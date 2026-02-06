# Config File Parser: Design Document


## Overview

A multi-format configuration file parser that supports INI, TOML, and YAML formats through unified parsing architecture. The key challenge is designing flexible tokenization and parsing components that can handle fundamentally different syntactic approaches while maintaining clean separation between lexical analysis and structural interpretation.


> This guide is meant to help you understand the big picture before diving into each milestone. Refer back to it whenever you need context on how components connect.


## Context and Problem Statement

> **Milestone(s):** All milestones (foundational understanding for INI Parser, TOML Tokenizer, TOML Parser, YAML Subset Parser)

Configuration file parsing represents one of the most deceptively complex challenges in software engineering. What appears to be a straightforward task—reading structured text and converting it into usable data structures—quickly reveals layers of intricate problems that demand sophisticated solutions. This complexity stems not from any single difficult algorithm, but from the intersection of multiple challenging domains: lexical analysis, syntax parsing, type inference, error recovery, and the fundamental differences between how humans write configuration files and how computers process structured data.

The core challenge lies in bridging the gap between human-friendly configuration syntax and machine-readable data structures while maintaining the flexibility and expressiveness that makes configuration files valuable in the first place. Configuration formats evolved to solve different problems for different communities, resulting in fundamentally incompatible approaches to representing the same underlying data. Building a unified parser that handles multiple formats requires understanding not just their syntactic differences, but the philosophical approaches that shaped their design.

### Why Configuration Parsing is Hard

Think of configuration file parsing like being a universal translator at the United Nations, but instead of translating between human languages, you're translating between different ways of thinking about structured data. Each configuration format represents a different cultural approach to organizing information—INI files reflect the Windows registry mindset of hierarchical sections, TOML embodies the programmer's desire for explicit types and unambiguous syntax, while YAML embraces the document author's preference for visual structure and minimal punctuation.

The fundamental challenge begins with **lexical ambiguity**—the same character sequence can mean entirely different things depending on context and format. Consider the innocent-looking string `key = "value"`. In INI format, this is a straightforward key-value assignment where the quotes are likely part of the value. In TOML, those quotes create a basic string with potential escape sequence processing. In YAML, this same line might be invalid because YAML prefers `key: value` syntax, or it could be interpreted as a mapping with a complex key containing an equals sign. A robust parser must not only recognize these differences but switch between entirely different parsing strategies based on format detection.

**Context sensitivity** presents another layer of complexity that novice parser implementers consistently underestimate. The meaning of characters and tokens changes dramatically based on their surrounding context. In TOML, the sequence `[[database]]` creates an array of tables entry, but `[database]` creates a simple table header, while `[ "database" ]` might be part of an inline array containing a single string. The parser must maintain sophisticated state tracking to distinguish between these contexts, often requiring lookahead parsing or backtracking when initial assumptions prove incorrect.

**Whitespace semantics** vary dramatically between formats in ways that create subtle but critical parsing challenges. INI files generally treat whitespace as optional padding around meaningful content, allowing liberal spacing that gets trimmed during processing. TOML follows similar principles but with stricter rules around string literals and multiline constructs. YAML, however, makes whitespace semantically meaningful—indentation levels determine nesting structure, and the difference between two spaces and four spaces can completely change the resulting data structure. A parser architecture must accommodate both whitespace-agnostic and whitespace-sensitive parsing modes, often within the same parsing session when handling mixed format scenarios.

The **type inference problem** reveals another dimension of complexity that extends beyond simple syntax parsing. Configuration files exist primarily for human authorship, which means they optimize for writing convenience rather than parsing simplicity. Humans write `timeout = 30` expecting the parser to understand this represents a numeric value, not a string containing digits. They write `enabled = yes` expecting boolean interpretation, and `created = 2023-10-15T14:30:00Z` expecting datetime parsing. Each format handles type inference differently—TOML provides explicit type syntax but still requires inference for basic literals, YAML attempts aggressive type inference that can surprise users, while INI files traditionally treat everything as strings, leaving type interpretation to the application layer.

**Error recovery and reporting** becomes exponentially more complex in configuration parsing because configuration files are primarily authored by humans, not generated by tools. Human-authored content contains creative deviations from formal grammar, partial completions during editing, and errors that reflect misunderstanding of format rules rather than simple typos. A parser must not only detect errors but provide meaningful feedback that helps users understand both what went wrong and how to fix it. This requires maintaining enough parsing context to generate suggestions, tracking line and column positions through complex tokenization processes, and distinguishing between recoverable syntax errors and fundamental format violations.

> **Critical Insight**: Configuration parsing difficulty stems not from complex algorithms but from the impedance mismatch between human-friendly syntax and machine-processable structure. Each format makes different trade-offs in this space, requiring parsers to switch between fundamentally different processing paradigms.

The **nested structure mapping problem** presents unique challenges that distinguish configuration parsing from simpler key-value processing. Modern configuration formats support deeply nested data structures that must be constructed incrementally as parsing proceeds. TOML's dotted key syntax like `database.connection.pool.size = 10` requires creating intermediate dictionary structures on demand while detecting conflicts with previously defined keys. YAML's indentation-based nesting requires stack-based tracking of scope levels with complex rules for when blocks end and new structures begin. INI files, despite their apparent simplicity, introduce nesting through section headers that create hierarchical organization.

**Format detection and switching** adds another layer of complexity that becomes critical in real-world applications. Users rarely want to specify the configuration format explicitly—they expect parsers to automatically detect whether a file is INI, TOML, or YAML based on content analysis. This detection must happen early enough to select the appropriate parsing strategy but late enough to have sufficient content for reliable identification. False positives in format detection can lead to confusing error messages when content gets parsed with the wrong grammar rules.

**Unicode and encoding handling** permeates every aspect of configuration parsing but often gets overlooked until problems emerge. Configuration files exist in international contexts with multi-byte character encodings, right-to-left text, combining characters, and normalization requirements. String literal processing must handle escape sequences that can introduce arbitrary Unicode code points, while maintaining proper character counting for error position reporting. Different formats have varying levels of Unicode sophistication—YAML requires full Unicode support for identifiers, while INI files traditionally assume ASCII-compatible encodings.

### Format Comparison Analysis

Understanding the specific characteristics and parsing requirements of each configuration format provides the foundation for designing a unified parsing architecture. Each format evolved to solve different problems, resulting in distinct approaches to syntax, semantics, and complexity management that directly influence parser design decisions.

| Format Aspect | INI | TOML | YAML Subset |
|---------------|-----|------|-------------|
| **Primary Design Goal** | Simple Windows registry-style configuration | Unambiguous configuration with explicit types | Human-readable documents with minimal syntax |
| **Syntax Philosophy** | Section-based key-value pairs | Programming language inspired explicit syntax | Indentation-based visual hierarchy |
| **Parsing Complexity** | Low - line-based processing | High - recursive descent with lookahead | Medium - indentation-sensitive with context |
| **Type System** | Implicit string-based | Explicit with inference | Aggressive implicit inference |
| **Nesting Support** | Section-based hierarchy only | Full nested structures with dotted keys | Arbitrary depth through indentation |
| **Comment Handling** | Semicolon and hash prefixes | Hash prefix only | Hash prefix with flow considerations |
| **String Literals** | Basic quoting with simple escapes | Multiple string types with complex rules | Quoted, unquoted, and multiline variants |
| **Error Recovery** | Forgiving - skip malformed lines | Strict - fail on ambiguity | Context-sensitive validation |

**INI Format Parsing Characteristics** reflect its origins as a simple configuration mechanism for early PC software. The format prioritizes ease of manual editing over sophisticated data representation, resulting in parsing requirements that favor line-by-line processing with minimal lookahead. INI parsers can process files streaming fashion, making decisions about each line independently based on simple pattern matching. This simplicity, however, creates ambiguities that require careful design decisions around edge cases.

The section-based organization of INI files creates a natural parsing structure where the parser maintains minimal state—primarily the current section name and accumulated key-value pairs. Section headers use bracket syntax `[section_name]` that's visually distinct from key-value pairs, enabling reliable pattern-based identification. Key-value pairs support both equals and colon separators (`key = value` and `key: value`) with whitespace trimming, but this flexibility introduces edge cases around keys or values that contain these separator characters.

INI comment handling appears straightforward but contains subtle complexities that impact parser design. Comments can begin with semicolons or hash symbols and extend to the end of the line, but the interaction between comments and quoted strings requires careful consideration. A line like `path = "C:\Program Files\App" ; comment` must distinguish between semicolons inside quoted strings and comment-starting semicolons, requiring some form of quoted string parsing even in this simple format.

**TOML Format Parsing Characteristics** represent the opposite extreme from INI simplicity, embracing explicit syntax that eliminates ambiguity at the cost of parsing complexity. TOML requires full tokenization with lookahead parsing, recursive descent for nested structures, and sophisticated type inference that respects explicit type annotations while handling implicit cases gracefully.

The tokenization requirements for TOML include multiple string literal types that each follow different processing rules. Basic strings use double quotes with escape sequence processing, literal strings use single quotes with minimal processing, and multiline variants of both types have complex rules for handling leading and trailing whitespace. Integer literals can include underscores for readability (`1_000_000`), floating-point numbers support scientific notation, and the format includes native boolean and datetime types that require specialized parsing logic.

TOML's table structure creates some of the most complex parsing challenges in configuration file processing. Simple tables use `[table_name]` syntax similar to INI sections, but nested tables use dotted notation `[table.subtable.deep]` that requires creating intermediate structures dynamically. Array-of-tables syntax `[[table_name]]` creates list structures containing dictionaries, with complex rules about when new entries are created versus when existing entries are extended. Dotted key assignments like `physical.color = "orange"` can create nested structures implicitly, but conflict with explicitly defined tables in ways that require careful validation.

| TOML Parsing Challenge | Complexity Level | Key Difficulty |
|------------------------|------------------|----------------|
| **Basic key-value pairs** | Low | Type inference and string literal handling |
| **Inline tables and arrays** | Medium | Balanced bracket parsing with nested values |
| **Table headers** | Medium | Nested structure creation and conflict detection |
| **Array of tables** | High | List management with dictionary entry semantics |
| **Dotted keys** | High | Implicit structure creation with conflict resolution |

**YAML Subset Parsing Characteristics** introduce indentation-sensitive parsing that requires fundamentally different algorithms from character-delimited formats. YAML parsers must track indentation levels using a stack-based approach, making parsing decisions based on the relationship between current and previous indentation rather than explicit delimiters.

The indentation sensitivity of YAML creates parsing challenges that don't exist in other formats. The parser must distinguish between spaces and tabs (YAML forbids tabs for indentation), track indentation levels precisely, and determine when indentation changes indicate structure transitions versus continuation of existing structures. A mapping entry like `key: value` at indentation level 2 might start a new mapping, continue an existing sequence, or represent a nested value depending on the preceding context.

YAML's type inference system attempts to automatically detect appropriate types for scalar values, but the rules can surprise users and complicate parsing. The string "yes" becomes boolean true, "1.0" becomes a floating-point number, and "2023-10-15" might become a date object depending on parser configuration. This aggressive inference requires parsers to implement pattern matching against multiple type signatures while providing mechanisms for users to override automatic detection through explicit quoting.

Flow syntax in YAML provides inline alternatives to block structure using familiar bracket and brace notation (`[1, 2, 3]` for sequences and `{key: value}` for mappings). Supporting flow syntax requires YAML parsers to handle both indentation-sensitive block parsing and delimiter-based parsing within the same document, often switching between modes multiple times as parsing proceeds.

> **Decision: Unified Parsing Architecture vs Format-Specific Parsers**
> - **Context**: Need to support multiple configuration formats with different parsing requirements and complexity levels
> - **Options Considered**: 
>   1. Completely separate parsers for each format with no shared code
>   2. Unified parser that handles all formats through extensive configuration
>   3. Shared tokenization layer with format-specific parsing logic
> - **Decision**: Shared tokenization foundation with format-specific parsing components
> - **Rationale**: Balances code reuse for common functionality (string handling, position tracking, error reporting) while acknowledging that parsing strategies differ fundamentally between formats
> - **Consequences**: Enables consistent error reporting and position tracking across formats, allows incremental implementation starting with simpler formats, but requires careful interface design to accommodate different tokenization needs

The architectural decision to use shared tokenization with format-specific parsers reflects the reality that while these formats differ significantly in syntax and semantics, they share common low-level requirements around string processing, position tracking, and error context management. This approach allows the implementation to start with the simpler INI format to establish core tokenization patterns, then extend the tokenizer capabilities as needed for TOML's more complex requirements, and finally adapt for YAML's unique indentation-sensitive needs.

**Error Handling Strategy Comparison** reveals how format characteristics influence error detection, recovery, and reporting approaches. Each format's syntax philosophy directly impacts what constitutes an error, how errors can be detected, and what recovery strategies are viable.

| Error Category | INI Approach | TOML Approach | YAML Approach |
|----------------|--------------|---------------|---------------|
| **Syntax Errors** | Skip malformed lines, continue processing | Fail fast with precise error location | Context-sensitive validation with recovery hints |
| **Type Errors** | No type validation (strings only) | Strict type checking with clear messages | Type inference conflicts with override suggestions |
| **Structure Errors** | Section redefinition warnings | Key redefinition failures with conflict location | Indentation inconsistencies with structure visualization |
| **Encoding Errors** | Best-effort ASCII compatible processing | UTF-8 validation with byte position reporting | Full Unicode support with normalization guidance |

This comprehensive comparison of format characteristics and parsing requirements establishes the foundation for designing a unified configuration parsing system that respects each format's unique properties while sharing common infrastructure where beneficial. The next sections will detail how these insights translate into specific architectural decisions and component designs.

### Implementation Guidance

The implementation of a multi-format configuration parser requires careful technology selection and project organization that accommodates the varying complexity levels of different formats while maintaining clean separation between shared and format-specific functionality.

**A. Technology Recommendations Table:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|----------------|
| **Tokenizer Core** | Regular expressions with manual state tracking | Hand-written state machine with character-level control |
| **String Processing** | Python built-in string methods with manual escape handling | Custom string literal parser with full Unicode support |
| **Data Structures** | Native dictionaries and lists with manual nesting | Custom tree structures with metadata tracking |
| **Error Reporting** | Exception-based with basic line number tracking | Structured error objects with position ranges and suggestions |
| **File I/O** | Direct file reading with encoding detection | Streaming parser with configurable buffer sizes |
| **Type Inference** | String-based with manual conversion functions | Plugin-based type detection with user customization |

**B. Recommended File/Module Structure:**

```
config-parser/
├── src/
│   ├── __init__.py                    ← public API exports
│   ├── core/
│   │   ├── __init__.py               ← shared parsing infrastructure
│   │   ├── tokenizer.py              ← base tokenization with position tracking
│   │   ├── errors.py                 ← error types and reporting utilities
│   │   ├── types.py                  ← token definitions and data structures
│   │   └── detector.py               ← format detection logic
│   ├── parsers/
│   │   ├── __init__.py               ← parser registry and factory
│   │   ├── ini_parser.py             ← INI format implementation (start here)
│   │   ├── toml_parser.py            ← TOML format implementation
│   │   └── yaml_parser.py            ← YAML subset implementation
│   └── utils/
│       ├── __init__.py               ← utility functions
│       ├── string_utils.py           ← string literal processing helpers
│       └── type_inference.py         ← automatic type detection
├── tests/
│   ├── test_data/                    ← sample configuration files
│   ├── test_tokenizer.py            ← tokenizer unit tests
│   ├── test_ini_parser.py           ← INI parser tests
│   ├── test_toml_parser.py          ← TOML parser tests
│   └── test_yaml_parser.py          ← YAML parser tests
└── examples/
    ├── basic_usage.py               ← simple parsing examples
    └── advanced_features.py        ← error handling and customization
```

**C. Infrastructure Starter Code:**

```python
# src/core/types.py - Complete token type definitions
from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Optional

class TokenType(Enum):
    # Common tokens across all formats
    STRING = auto()
    NUMBER = auto()
    BOOLEAN = auto()
    IDENTIFIER = auto()
    EQUALS = auto()
    COLON = auto()
    NEWLINE = auto()
    EOF = auto()
    COMMENT = auto()
    
    # Format-specific tokens
    SECTION_START = auto()      # [
    SECTION_END = auto()        # ]
    ARRAY_START = auto()        # [ in TOML/YAML context
    ARRAY_END = auto()          # ]
    OBJECT_START = auto()       # {
    OBJECT_END = auto()         # }
    COMMA = auto()
    DOT = auto()
    
    # YAML-specific
    INDENT = auto()
    DEDENT = auto()
    BLOCK_SEQUENCE = auto()     # -
    
    # Special tokens
    INVALID = auto()

@dataclass
class Position:
    """Tracks position in source file for error reporting."""
    line: int
    column: int
    offset: int
    
    def __str__(self) -> str:
        return f"line {self.line}, column {self.column}"

@dataclass
class Token:
    """Represents a parsed token with position and value information."""
    type: TokenType
    value: Any
    position: Position
    raw_text: str = ""
    
    def __str__(self) -> str:
        return f"{self.type.name}({self.value}) at {self.position}"
```

```python
# src/core/errors.py - Complete error handling infrastructure
from typing import List, Optional
from .types import Position

class ParseError(Exception):
    """Base class for all parsing errors with position information."""
    
    def __init__(self, message: str, position: Optional[Position] = None, 
                 suggestion: Optional[str] = None):
        self.message = message
        self.position = position
        self.suggestion = suggestion
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        msg = self.message
        if self.position:
            msg = f"At {self.position}: {msg}"
        if self.suggestion:
            msg = f"{msg}\nSuggestion: {self.suggestion}"
        return msg

class TokenError(ParseError):
    """Errors during tokenization phase."""
    pass

class SyntaxError(ParseError):
    """Errors in format-specific syntax."""
    pass

class StructureError(ParseError):
    """Errors in logical structure (key conflicts, invalid nesting)."""
    pass

def create_error_context(source: str, position: Position, 
                        context_lines: int = 2) -> str:
    """Generate visual error context showing problematic source location."""
    lines = source.split('\n')
    start_line = max(0, position.line - context_lines - 1)
    end_line = min(len(lines), position.line + context_lines)
    
    context = []
    for i in range(start_line, end_line):
        line_num = i + 1
        marker = ">>>" if line_num == position.line else "   "
        context.append(f"{marker} {line_num:3}: {lines[i]}")
        
        # Add column pointer for error line
        if line_num == position.line:
            pointer = " " * (7 + position.column - 1) + "^"
            context.append(pointer)
    
    return '\n'.join(context)
```

**D. Core Logic Skeleton Code:**

```python
# src/core/tokenizer.py - Base tokenizer with TODO implementation points
from typing import Iterator, Optional, List
from .types import Token, TokenType, Position
from .errors import TokenError

class BaseTokenizer:
    """Base tokenizer providing position tracking and common functionality."""
    
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
    
    def current_position(self) -> Position:
        """Get current position for error reporting."""
        return Position(self.line, self.column, self.position)
    
    def peek(self, offset: int = 0) -> str:
        """Look ahead at character without consuming it."""
        pos = self.position + offset
        if pos >= len(self.source):
            return '\0'  # EOF marker
        return self.source[pos]
    
    def advance(self) -> str:
        """Consume and return current character, updating position."""
        if self.position >= len(self.source):
            return '\0'
        
        char = self.source[self.position]
        self.position += 1
        
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
            
        return char
    
    def tokenize(self) -> List[Token]:
        """Main tokenization entry point - implement in subclasses."""
        # TODO 1: Initialize tokenization state
        # TODO 2: Loop through characters calling appropriate token handlers
        # TODO 3: Handle end-of-file and return complete token list
        # TODO 4: Ensure all tokens have proper position information
        raise NotImplementedError("Subclasses must implement tokenize()")
    
    def skip_whitespace(self) -> None:
        """Skip whitespace characters (except newlines in some formats)."""
        # TODO: Define which characters count as skippable whitespace
        # TODO: Handle format-specific whitespace rules (YAML indentation)
        # TODO: Update position tracking while skipping
        pass
    
    def read_string_literal(self, quote_char: str) -> Token:
        """Parse quoted string with escape sequence handling."""
        # TODO 1: Track starting position for error reporting
        # TODO 2: Consume opening quote
        # TODO 3: Process characters until closing quote
        # TODO 4: Handle escape sequences (\n, \t, \", \\, etc.)
        # TODO 5: Handle unterminated strings with helpful error messages
        # TODO 6: Return STRING token with processed value
        pass
    
    def read_number(self) -> Token:
        """Parse numeric literal (integer or float)."""
        # TODO 1: Collect digit characters and decimal points
        # TODO 2: Handle scientific notation (1e5, 1E-3)
        # TODO 3: Handle format-specific features (TOML underscores)
        # TODO 4: Distinguish between integers and floats
        # TODO 5: Validate number format and return NUMBER token
        pass
```

**E. Language-Specific Hints:**

- **String Processing**: Use Python's `str.translate()` for efficient character mapping during escape sequence processing
- **Position Tracking**: Consider using `enumerate()` with `splitlines(keepends=True)` for line-aware processing
- **Regular Expressions**: The `re` module's `finditer()` method provides position information useful for tokenization
- **Error Context**: Use `textwrap.dedent()` and `textwrap.indent()` for clean error message formatting
- **Type Inference**: Python's `ast.literal_eval()` can safely evaluate simple literals for type detection
- **File Encoding**: Always specify encoding explicitly when opening files: `open(filename, 'r', encoding='utf-8')`
- **Performance**: For large configuration files, consider using `io.StringIO` for string manipulation instead of concatenation

**F. Milestone Checkpoint:**

After implementing the basic tokenizer infrastructure:

**Test Command**: `python -m pytest tests/test_tokenizer.py -v`

**Expected Output**: All tokenizer tests should pass, demonstrating proper position tracking, basic token recognition, and error handling.

**Manual Verification**: Create a simple test script:
```python
from src.core.tokenizer import BaseTokenizer

# Test with a simple input
source = 'key = "value"\n# comment\n'
tokenizer = BaseTokenizer(source)  # Will fail until you implement a concrete subclass
tokens = tokenizer.tokenize()
for token in tokens:
    print(f"{token.type.name}: '{token.value}' at {token.position}")
```

**Signs Something is Wrong**:
- Position tracking jumps or becomes negative → Check advance() method character handling
- Tokens missing position information → Ensure current_position() called when creating tokens  
- String parsing fails on quotes → Verify escape sequence handling in read_string_literal()
- Performance issues on large files → Check for inefficient string concatenation in tokenizer loops


## Goals and Non-Goals

> **Milestone(s):** All milestones (foundational scoping for INI Parser, TOML Tokenizer, TOML Parser, YAML Subset Parser)

Understanding the precise scope of our configuration file parser is essential for making informed architectural decisions and avoiding feature creep during implementation. Think of this section as drawing the boundaries of a map—we need to know not just what territories we'll explore, but also what lies beyond our borders so we don't accidentally wander into complexity quicksand. The goals and non-goals serve as our north star throughout the implementation journey, helping us make consistent decisions when faced with ambiguous requirements or tempting feature additions.

The challenge in scoping a multi-format parser lies in the **impedance mismatch** between different configuration philosophies. INI files embrace simplicity with flat key-value pairs, TOML aims for human readability with explicit typing, and YAML prioritizes minimalist syntax with implicit typing. Each format makes different trade-offs between expressiveness, readability, and parsing complexity. Our scope must find the sweet spot that captures the essential utility of each format without drowning in edge cases that provide minimal real-world value.

### Functional Goals

The functional goals define the minimum viable product that delivers meaningful value to users while providing comprehensive learning experiences in parsing techniques. These goals were selected based on analyzing common configuration use cases and identifying the features that appear in 80% of real-world configuration files.

#### Core Format Support Requirements

Our parser must support three distinct configuration formats, each representing a different parsing paradigm. This multi-format approach provides exposure to line-based parsing (INI), recursive descent parsing (TOML), and indentation-sensitive parsing (YAML). The format support is intentionally comprehensive within each format's core feature set rather than attempting to cover exotic edge cases.

| Format | Core Features | Complexity Level | Learning Focus |
|--------|---------------|------------------|----------------|
| INI | Sections, key-value pairs, comments | Low | String processing, line parsing |
| TOML | Tables, arrays, type system, nested structures | High | Recursive descent, tokenization |
| YAML | Indentation-based nesting, mappings, sequences | Medium | Stack-based parsing, implicit typing |

**INI Format Support Goals:**

The INI parser serves as our foundational implementation, introducing core parsing concepts without overwhelming complexity. INI files follow a straightforward section-based organization that maps naturally to nested dictionaries. Our implementation must handle section headers using bracket notation `[section.subsection]`, creating appropriate nested dictionary structures. Key-value pairs support both equals and colon delimiters (`key=value` and `key: value`), with automatic whitespace trimming around keys and values.

Comment handling requires supporting both semicolon and hash prefixes (`; comment` and `# comment`), with comments allowed on their own lines and inline after values. String value parsing must handle both quoted and unquoted strings, with proper escape sequence processing for quoted strings including `\"`, `\\`, `\n`, `\r`, and `\t`. The parser should support multi-line value continuation using backslash line endings, properly joining continued lines with space characters.

Global key handling—keys that appear before any section header—requires creating an implicit root section or global namespace. This prevents data loss and provides consistent access patterns for simple configuration files that don't use sections.

**TOML Format Support Goals:**

TOML parsing represents the most complex component, introducing full tokenization and recursive descent parsing techniques. The tokenizer must recognize all TOML grammar elements including basic strings (quoted with double quotes), literal strings (quoted with single quotes), and their multiline variants (triple-quoted). Numeric parsing includes integers with optional underscores (`1_000_000`), floats with scientific notation, and hexadecimal/octal/binary integer formats.

Table parsing handles both simple tables `[table.name]` and array-of-tables `[[array.name]]` syntax. Simple tables create nested dictionary structures where dotted paths like `[server.database.connection]` create three levels of nesting. Array-of-tables creates lists of dictionaries, where each occurrence of `[[servers]]` adds a new dictionary entry to the servers array.

Inline syntax support includes inline tables `{ key1 = "value1", key2 = 42 }` and inline arrays `[ 1, 2, "three", true ]` with mixed type support. Dotted key notation like `physical.color = "orange"` automatically creates nested dictionary structures without requiring explicit table headers.

The TOML type system requires proper handling of strings, integers, floats, booleans, datetime values, and nested combinations of arrays and tables. Type inference converts literal syntax to appropriate Python types automatically.

**YAML Subset Support Goals:**

Our YAML implementation focuses on the core block syntax that represents 90% of configuration use cases, deliberately avoiding the more exotic flow syntax variants that add complexity without proportional learning value. The subset includes indentation-based block structure parsing where nesting depth is determined by consistent indentation levels. Mappings use the `key: value` syntax with proper handling of nested mappings through indentation increases.

Sequences use the dash prefix syntax (`- item`) for list creation, with support for both simple scalar sequences and complex nested sequences containing mappings or other sequences. Scalar type inference automatically converts unquoted values to appropriate types: numeric strings become integers or floats, `true`/`false`/`yes`/`no` become booleans, and everything else remains strings.

Flow syntax support is limited to basic inline arrays `[item1, item2, item3]` and inline objects `{key1: value1, key2: value2}` to handle simple embedded structures without requiring full flow syntax parsing.

#### Unified Interface Requirements

Beyond format-specific parsing, our system must provide a consistent interface that abstracts away format differences for consuming applications. This unified interface enables applications to work with configuration data without knowing the source format, promoting code reuse and simplifying configuration migration.

The primary interface is a `parse_config(content, format=None)` function that accepts configuration content as a string and an optional format specifier. When format is not specified, the parser performs **format detection** by examining content characteristics like the presence of section brackets (INI), table headers (TOML), or significant indentation (YAML).

All parsers return data in a **unified output format** consisting of nested dictionaries with string keys and mixed-type values (strings, numbers, booleans, lists, nested dictionaries). This standardized representation allows consuming code to navigate configuration hierarchies using consistent patterns regardless of source format.

Error handling follows a structured approach using custom exception types: `TokenError` for lexical analysis failures, `SyntaxError` for grammar violations, and `StructureError` for logical inconsistencies like duplicate keys or circular references. All errors include position information (`Position` with line, column, and offset fields) and suggested fixes when possible.

#### Development and Learning Goals

Since this project serves as a learning vehicle for parsing concepts, our functional goals include comprehensive educational outcomes alongside practical functionality. The implementation must demonstrate **recursive descent parsing** principles through the TOML parser, showing how complex nested structures can be parsed through function call recursion that mirrors grammar structure.

**Tokenization** concepts are thoroughly explored through the TOML lexer implementation, including state machine management for string literal parsing, lookahead techniques for disambiguating syntax, and position tracking for meaningful error reporting.

**Type inference** and automatic type conversion provide exposure to data type detection algorithms and the trade-offs between explicit and implicit typing systems. The different approaches across INI (mostly strings), TOML (explicit typing), and YAML (aggressive implicit typing) demonstrate the spectrum of type system design decisions.

Error recovery and diagnostic message generation represent critical parsing skills that extend beyond configuration files to any language processing task. The implementation must demonstrate how to detect errors early, provide meaningful context, and suggest concrete fixes rather than cryptic failure messages.

> **Decision: Comprehensive Format Coverage vs. Focused Implementation**
> - **Context**: Configuration parsers can either support many formats shallowly or fewer formats deeply
> - **Options Considered**: 
>   1. Support 5+ formats (INI, TOML, YAML, JSON, XML) with basic functionality
>   2. Support 3 formats (INI, TOML, YAML subset) with comprehensive feature coverage
>   3. Support 1 format (TOML only) with production-ready completeness
> - **Decision**: Option 2 - Three formats with comprehensive coverage
> - **Rationale**: Three formats provide exposure to fundamentally different parsing approaches (line-based, recursive descent, indentation-sensitive) while maintaining manageable complexity. Each format teaches distinct concepts without excessive overlap.
> - **Consequences**: Deeper learning experience with manageable scope, but requires careful subset selection for YAML to avoid overwhelming complexity

| Option | Parsing Techniques Learned | Implementation Complexity | Real-World Utility |
|--------|---------------------------|--------------------------|-------------------|
| Many formats shallow | Surface-level pattern matching | Low per format | Broad but shallow |
| Three formats deep | Line parsing, recursive descent, indentation parsing | Medium overall | Focused and practical |
| Single format complete | One technique deeply | Low overall | Deep but narrow |

### Non-Goals

The non-goals are equally important as the functional goals, establishing clear boundaries that prevent scope creep and maintain focus on core learning objectives. These exclusions are deliberate choices based on complexity analysis, learning value assessment, and implementation timeline considerations.

#### Advanced YAML Features

YAML's full specification includes numerous advanced features that add significant parsing complexity without proportional educational value for our core parsing learning objectives. We explicitly exclude **multiline string syntax** including folded scalars (`>`) and literal scalars (`|`) which require complex whitespace processing rules and line break interpretation logic.

**Flow syntax** beyond basic inline collections is excluded, meaning we don't support complex nested flow structures, flow mappings with complex keys, or mixed block/flow syntax within the same document. **Anchors and references** (`&anchor` and `*reference`) are excluded as they require symbol table management and cycle detection logic that distracts from core parsing concepts.

**Complex implicit typing** rules are simplified to basic cases. We don't support the full YAML type inference rules that can interpret strings like `2.10` as version numbers rather than floating-point numbers, or complex date/time parsing beyond basic ISO 8601 formats. **Document directives** (`%YAML 1.2`) and multiple documents in a single file are outside our scope.

These exclusions allow us to focus on the fundamental parsing concepts—indentation tracking, nested structure creation, and basic type inference—without getting lost in YAML's more esoteric features.

#### Performance Optimization Features

Our parser prioritizes educational clarity over performance optimization, deliberately excluding features that would complicate the core algorithms without teaching fundamentally new parsing concepts. **Streaming parsing** for large files is excluded as it requires complex buffering logic and stateful parsing that obscures the recursive descent algorithms we're trying to demonstrate.

**Memory optimization** techniques like string interning, parse tree compression, or lazy evaluation are outside scope. The implementation will create full parse trees and nested dictionary structures in memory, accepting higher memory usage in favor of simpler, more understandable algorithms.

**Incremental parsing** and **caching mechanisms** are excluded as they require sophisticated invalidation logic and change detection that distracts from core parsing implementation. Each `parse_config()` call processes the entire input from scratch.

**Parallel parsing** for multi-document files or concurrent tokenization/parsing pipelines are beyond scope, as the coordination complexity would overshadow the parsing algorithm learning objectives.

#### Production-Ready Features

While our parser will handle real configuration files correctly, several production-ready features are explicitly excluded to maintain focus on parsing fundamentals rather than software engineering concerns.

**Schema validation** and **configuration validation** are excluded despite their practical importance. Adding schema support would require a type system definition language, validation rule engines, and comprehensive error reporting for constraint violations. These features teach configuration management rather than parsing techniques.

**Plugin architectures** for custom format support are outside scope. While our design will be extensible in principle, we won't implement the dynamic loading, registration systems, or interface contracts needed for runtime format plugin support.

**Configuration merging** and **environment variable interpolation** represent configuration management concerns rather than parsing challenges. Features like `${ENV_VAR}` substitution, file inclusion (`!include other.yaml`), or configuration inheritance require post-processing logic that operates on parsed results rather than parsing algorithms themselves.

**Backwards compatibility** with legacy format variants is excluded. We implement current standard versions of each format rather than supporting deprecated syntax or vendor-specific extensions.

#### Advanced Error Recovery

While error detection and reporting are core goals, sophisticated **error recovery** mechanisms that attempt to continue parsing after syntax errors are excluded. Implementing recovery requires complex synchronization point detection, parser state restoration, and speculative parsing techniques that significantly complicate the core algorithms.

**Syntax error correction** and **"did you mean" suggestions** beyond basic cases are outside scope. We'll provide meaningful error messages with position information and simple suggestions, but not fuzzy matching or complex correction algorithms.

**Incremental error reporting** during parsing (reporting multiple errors in a single pass) is excluded in favor of **fail-fast behavior** that stops at the first error. This simplifies the parsing logic and error handling flow, though it means users might need multiple parse attempts to discover all syntax errors.

> **Decision: Educational Focus vs. Production Features**
> - **Context**: Learning projects can either simulate production requirements or focus purely on concept demonstration
> - **Options Considered**: 
>   1. Include production features (schema validation, plugins, performance optimization) for realism
>   2. Exclude production features to maintain focus on core parsing concepts
>   3. Hybrid approach with optional production feature modules
> - **Decision**: Option 2 - Exclude production features entirely
> - **Rationale**: Production features teach software engineering and API design rather than parsing algorithms. Including them would double the implementation complexity while diluting the core learning experience around tokenization, recursive descent parsing, and data structure mapping.
> - **Consequences**: Cleaner learning experience focused on parsing fundamentals, but resulting parser requires additional work for production use

| Feature Category | Educational Value | Implementation Complexity | Scope Decision |
|-----------------|-------------------|--------------------------|----------------|
| Core parsing algorithms | Very High | Medium | Include |
| Error detection/reporting | High | Low-Medium | Include |
| Performance optimization | Low | High | Exclude |
| Production features | Low-Medium | High | Exclude |
| Advanced error recovery | Medium | Very High | Exclude |

#### Format Subset Justification

Each format's feature exclusions follow specific principles designed to maximize learning value while minimizing implementation burden. The subset selections ensure that students encounter the core challenges of each parsing paradigm without getting overwhelmed by format-specific edge cases.

**INI Subset Rationale:**

INI format exclusions focus on eliminating legacy compatibility issues and vendor-specific extensions that don't teach fundamental parsing concepts. We exclude **case-insensitive key handling** as it adds string processing complexity without parsing algorithm learning value. **Custom delimiter support** beyond equals and colon is excluded as it requires configurable tokenization rules.

**Escape sequence handling** is limited to the standard set (`\"`, `\\`, `\n`, `\r`, `\t`) rather than supporting arbitrary Unicode escape sequences or custom escape definitions. This provides sufficient complexity to learn escape processing without requiring full Unicode lexical analysis.

**TOML Subset Rationale:**

TOML exclusions eliminate the most complex edge cases while retaining comprehensive coverage of the core type system and table structures. **Custom datetime formats** beyond ISO 8601 are excluded, as datetime parsing teaches string pattern matching rather than recursive parsing concepts.

**Complex array syntax** like arrays of arrays of tables with mixed inline/block syntax are excluded as they require extensive lookahead and backtracking logic that obscures the core recursive descent patterns we're demonstrating.

**Advanced numeric formats** like infinite/NaN float values are excluded as they add IEEE 754 floating-point knowledge requirements without contributing to parsing algorithm understanding.

**YAML Subset Rationale:**

YAML exclusions focus on eliminating the features that make YAML parsing notoriously complex while retaining the indentation-sensitive parsing challenges that provide educational value. **Complex key syntax** including multi-line keys, flow keys, and complex keys (non-string keys) are excluded as they require sophisticated key parsing logic.

**Advanced scalar syntax** including tagged types (`!!int`, `!!str`), complex multiline strings, and aggressive implicit typing for version numbers, IP addresses, and other domain-specific formats are excluded. Basic implicit typing for numbers, booleans, and null values is retained as it demonstrates type inference concepts.

#### Unified Interface Goals

The parser must provide a clean, consistent interface that demonstrates good API design principles while supporting the learning objectives around format abstraction and error handling.

| Interface Component | Requirement | Learning Objective |
|-------------------|-------------|-------------------|
| `parse_config()` function | Single entry point for all formats | API design consistency |
| Format auto-detection | Determine format from content analysis | Pattern recognition algorithms |
| Unified output structure | Nested dictionaries with consistent key access | Data structure standardization |
| Position-aware error reporting | Line/column information for all parse errors | Error context generation |
| Type-preserved values | Maintain numeric, boolean, and string types appropriately | Type system handling |

The unified output format must preserve type information where formats provide it explicitly (TOML), infer types where formats expect it (YAML), and provide reasonable defaults where formats are ambiguous (INI). This demonstrates the challenges of **type system impedance mismatch** across different configuration philosophies.

**Format Detection Strategy Goals:**

Automatic format detection teaches pattern recognition and heuristic algorithm development. The detection algorithm analyzes content characteristics to identify format signatures: INI files typically contain `[section]` headers or `key=value` patterns, TOML files contain explicit tables or arrays with TOML-specific syntax like `[[array.of.tables]]`, and YAML files exhibit significant indentation patterns and colon-space separators.

The format detection must be robust enough to handle edge cases like empty files, files with only comments, or files that could plausibly match multiple formats. The algorithm should prefer more specific format indicators over general ones, defaulting to the most permissive format (INI) when detection is ambiguous.

#### Error Handling and Diagnostics Goals

Comprehensive error handling serves dual purposes: demonstrating proper error management techniques and providing a debugging-friendly development experience. Our error handling goals focus on three categories of errors that teach different aspects of parser error management.

**Lexical Errors (`TokenError`):**

Lexical errors occur during tokenization when character sequences cannot be converted into valid tokens. Examples include unterminated string literals, invalid escape sequences, or malformed numeric literals. These errors must include precise position information and suggest specific fixes like adding closing quotes or correcting escape syntax.

**Syntax Errors (`SyntaxError`):**

Syntax errors occur when token sequences don't match the expected grammar rules. Examples include missing section closing brackets in INI files, invalid table header syntax in TOML files, or incorrect indentation patterns in YAML files. These errors require context-aware messaging that explains what was expected versus what was encountered.

**Structure Errors (`StructureError`):**

Structure errors occur when parsed content violates logical consistency rules like duplicate key definitions, conflicting type assignments, or circular reference creation. These errors teach the difference between syntactic validity and semantic correctness.

| Error Category | Example Triggers | Required Context | Learning Focus |
|----------------|-----------------|------------------|----------------|
| `TokenError` | Unterminated strings, invalid escapes | Character position, surrounding text | Lexical analysis debugging |
| `SyntaxError` | Grammar violations, unexpected tokens | Token position, expected vs actual | Grammar rule enforcement |
| `StructureError` | Duplicate keys, type conflicts | Logical position, conflicting definitions | Semantic validation |

#### Data Structure Mapping Goals

The parser must demonstrate effective techniques for mapping disparate source formats to a unified internal representation. This **nested structure mapping** teaches fundamental data transformation concepts that apply beyond configuration parsing to any data integration challenge.

**Hierarchical Structure Creation:**

All three formats must map to consistent nested dictionary structures despite using different syntax for hierarchy expression. INI sections become dictionary keys, TOML tables become nested dictionary paths, and YAML indentation becomes nested dictionary nesting. This mapping consistency allows consuming applications to navigate configuration hierarchies using uniform key access patterns.

**Type Preservation and Conversion:**

The mapping must preserve semantic information where possible while providing consistent access patterns. TOML's explicit typing is preserved directly, YAML's implicit typing is resolved during parsing, and INI's string-based values undergo basic type inference for numeric and boolean patterns.

**List and Dictionary Distinction:**

The parser must clearly distinguish between configuration structures that represent ordered lists versus unordered mappings, preserving this distinction in the output format. TOML arrays become Python lists, YAML sequences become Python lists, and INI multi-value keys (where supported) become Python lists.

> The fundamental principle guiding our scope decisions is **learning amplification**—every included feature should teach a distinct parsing concept or reinforce previously learned concepts through varied application. Features that add complexity without educational payoff are ruthlessly excluded, while features that demonstrate core parsing principles are included even when they increase implementation effort.

### Implementation Guidance

The implementation approach balances educational clarity with practical functionality, using Python's strengths in string processing and dynamic typing while demonstrating parsing concepts that transfer to other languages.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| String Processing | Built-in `str` methods with manual indexing | Regular expressions with `re` module |
| Data Structures | Native `dict` and `list` with manual type checking | `typing` module with type hints throughout |
| Error Handling | Custom exception classes with string messages | Rich context objects with position tracking |
| Testing Framework | Built-in `unittest` module | `pytest` with fixtures and parametrization |
| File I/O | Simple `open()` and `read()` operations | `pathlib` with encoding detection |

#### Recommended Project Structure

The project organization reflects the component separation discussed in the high-level architecture, with clear boundaries between format-specific implementations and shared infrastructure:

```
config-parser/
  src/
    config_parser/
      __init__.py              ← public API exports
      core/
        __init__.py
        tokenizer.py           ← BaseTokenizer and TokenType definitions
        errors.py              ← ParseError hierarchy
        position.py            ← Position tracking utilities
      formats/
        __init__.py
        ini_parser.py          ← INI-specific parsing logic
        toml_parser.py         ← TOML tokenizer and parser
        yaml_parser.py         ← YAML subset parser
      utils/
        __init__.py
        format_detection.py    ← Automatic format detection
        type_inference.py      ← Shared type conversion utilities
  tests/
    unit/
      test_tokenizer.py
      test_ini_parser.py
      test_toml_parser.py
      test_yaml_parser.py
    integration/
      test_unified_interface.py
      test_format_detection.py
    fixtures/
      sample_configs/         ← test configuration files
        basic.ini
        complex.toml
        nested.yaml
  examples/
    basic_usage.py
    error_handling_demo.py
    format_comparison.py
```

#### Core Infrastructure Starter Code

The following infrastructure provides the foundation for all format-specific parsers, handling position tracking, error management, and basic tokenization concepts:

```python
"""
Core parsing infrastructure providing position tracking, error handling,
and base tokenization functionality for all configuration format parsers.
"""
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Any, Union

class TokenType(Enum):
    """Token types recognized across all supported configuration formats."""
    STRING = auto()
    NUMBER = auto()
    BOOLEAN = auto()
    IDENTIFIER = auto()
    EQUALS = auto()
    COLON = auto()
    NEWLINE = auto()
    EOF = auto()
    COMMENT = auto()
    SECTION_START = auto()
    SECTION_END = auto()
    ARRAY_START = auto()
    ARRAY_END = auto()
    OBJECT_START = auto()
    OBJECT_END = auto()
    COMMA = auto()
    DOT = auto()
    INDENT = auto()
    DEDENT = auto()
    BLOCK_SEQUENCE = auto()
    INVALID = auto()

@dataclass
class Position:
    """Tracks position in source text for error reporting and debugging."""
    line: int
    column: int
    offset: int

@dataclass
class Token:
    """Individual lexical unit with type, value, and position information."""
    type: TokenType
    value: Any
    position: Position
    raw_text: str

class ParseError(Exception):
    """Base exception for all configuration parsing errors."""
    def __init__(self, message: str, position: Optional[Position] = None, 
                 suggestion: Optional[str] = None):
        self.message = message
        self.position = position
        self.suggestion = suggestion
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format error message with position and suggestion information."""
        # TODO: Implement error message formatting with position context
        # TODO: Include suggestion text when available
        # TODO: Add visual indicators for error location
        pass

class TokenError(ParseError):
    """Lexical analysis errors during tokenization phase."""
    pass

class SyntaxError(ParseError):
    """Grammar violations during parsing phase."""
    pass

class StructureError(ParseError):
    """Logical consistency violations in parsed structure."""
    pass

EOF_MARKER = '\0'

class BaseTokenizer:
    """Base tokenizer providing common functionality for all formats."""
    
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
    
    def current_position(self) -> Position:
        """Returns current parsing position for error reporting."""
        return Position(self.line, self.column, self.position)
    
    def peek(self, offset: int = 0) -> str:
        """Look ahead at character without consuming it."""
        pos = self.position + offset
        if pos >= len(self.source):
            return EOF_MARKER
        return self.source[pos]
    
    def advance(self) -> str:
        """Consume current character and update position tracking."""
        if self.position >= len(self.source):
            return EOF_MARKER
        
        char = self.source[self.position]
        self.position += 1
        
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        
        return char
    
    def tokenize(self) -> List[Token]:
        """Main tokenization entry point. Override in format-specific tokenizers."""
        # TODO: Implement format-specific tokenization logic
        # TODO: Handle whitespace according to format semantics
        # TODO: Recognize format-specific token patterns
        # TODO: Build token list with position tracking
        pass
    
    def skip_whitespace(self) -> None:
        """Skip whitespace characters that don't carry semantic meaning."""
        while self.peek() in ' \t' and self.peek() != EOF_MARKER:
            self.advance()
    
    def read_string_literal(self, quote_char: str) -> Token:
        """Parse quoted string with escape sequence processing."""
        # TODO: Consume opening quote
        # TODO: Process characters until closing quote
        # TODO: Handle escape sequences (\n, \t, \", \\, etc.)
        # TODO: Handle unterminated string error case
        # TODO: Return STRING token with unescaped value
        pass
    
    def read_number(self) -> Token:
        """Parse numeric literal with type inference."""
        # TODO: Collect digit characters and decimal points
        # TODO: Handle scientific notation (1e5, 1.2e-3)
        # TODO: Detect integer vs float based on decimal point presence
        # TODO: Handle number format errors (multiple decimal points)
        # TODO: Return NUMBER token with converted Python value
        pass

def create_error_context(source: str, position: Position, context_lines: int = 2) -> str:
    """Generate visual error context showing source location with line numbers."""
    lines = source.split('\n')
    start_line = max(0, position.line - context_lines - 1)
    end_line = min(len(lines), position.line + context_lines)
    
    context = []
    for i in range(start_line, end_line):
        line_num = i + 1
        line_content = lines[i] if i < len(lines) else ""
        marker = " -> " if line_num == position.line else "    "
        context.append(f"{marker}{line_num:4d} | {line_content}")
        
        if line_num == position.line:
            pointer = " " * (len(marker) + 7 + position.column - 1) + "^"
            context.append(pointer)
    
    return "\n".join(context)

# Public API function signatures (implement in respective milestone modules)
def parse_config(content: str, format: Optional[str] = None) -> dict:
    """Main entry point for configuration parsing with automatic format detection."""
    # TODO: Implement format detection when format=None
    # TODO: Delegate to appropriate format-specific parser
    # TODO: Handle parsing errors with rich context
    # TODO: Return unified nested dictionary structure
    pass

def detect_format(content: str) -> str:
    """Analyze content characteristics to determine configuration format."""
    # TODO: Look for INI-specific patterns ([sections], key=value)
    # TODO: Look for TOML-specific patterns ([[tables]], explicit arrays)
    # TODO: Look for YAML-specific patterns (indentation, key: value)
    # TODO: Return format string or raise detection error
    pass
```

#### Parser Implementation Skeleton

Each format-specific parser builds on the base tokenizer infrastructure while implementing parsing algorithms appropriate to the format's syntactic structure:

```python
"""
Format-specific parser skeletons demonstrating different parsing approaches.
Students implement the TODO sections using concepts from respective milestones.
"""

class INIParser:
    """Line-based parser for INI format configuration files."""
    
    def __init__(self, content: str):
        self.lines = content.split('\n')
        self.line_index = 0
        self.current_section = None
        self.result = {}
    
    def parse(self) -> dict:
        """Parse INI content into nested dictionary structure."""
        # TODO 1: Initialize result dict and set current_section to global
        # TODO 2: Iterate through lines, skipping empty lines and comments
        # TODO 3: Detect section headers and update current_section
        # TODO 4: Parse key-value pairs and add to current section
        # TODO 5: Handle multi-line continuations with backslash endings
        # TODO 6: Return completed nested dictionary structure
        pass
    
    def _parse_section_header(self, line: str) -> str:
        """Extract section name from [section] line."""
        # TODO: Validate bracket syntax and extract section name
        # TODO: Support nested sections with dot notation
        # TODO: Handle whitespace around section names
        pass
    
    def _parse_key_value_pair(self, line: str) -> tuple:
        """Parse key=value or key: value into (key, value) tuple."""
        # TODO: Split on = or : delimiter (handle multiple = in value)
        # TODO: Trim whitespace from key and value
        # TODO: Handle quoted values with escape sequences
        # TODO: Strip inline comments from unquoted values
        pass

class TOMLParser:
    """Recursive descent parser for TOML format with full tokenization."""
    
    def __init__(self, content: str):
        self.tokenizer = TOMLTokenizer(content)
        self.tokens = self.tokenizer.tokenize()
        self.token_index = 0
        self.result = {}
    
    def parse(self) -> dict:
        """Parse TOML tokens into nested dictionary structure."""
        # TODO 1: Initialize result dictionary and current table context
        # TODO 2: Process tokens sequentially, dispatching by token type
        # TODO 3: Handle table headers and array-of-tables headers
        # TODO 4: Parse key-value assignments with dotted key support
        # TODO 5: Handle inline tables and inline arrays recursively
        # TODO 6: Validate no key redefinition errors
        pass
    
    def _parse_table_header(self) -> str:
        """Parse [table.name] header and return dotted table path."""
        # TODO: Consume opening bracket token
        # TODO: Collect identifier and dot tokens for table path
        # TODO: Consume closing bracket token
        # TODO: Validate table path doesn't conflict with existing keys
        pass
    
    def _parse_inline_table(self) -> dict:
        """Parse {key1 = value1, key2 = value2} inline table syntax."""
        # TODO: Consume opening brace token
        # TODO: Parse key-value pairs separated by commas
        # TODO: Handle nested inline tables and arrays recursively
        # TODO: Consume closing brace token and return dictionary
        pass

class YAMLParser:
    """Stack-based parser for YAML subset with indentation sensitivity."""
    
    def __init__(self, content: str):
        self.lines = content.split('\n')
        self.line_index = 0
        self.indentation_stack = [0]
        self.result = {}
    
    def parse(self) -> dict:
        """Parse YAML content using indentation-based nesting."""
        # TODO 1: Initialize result and indentation tracking
        # TODO 2: Process lines, calculating indentation levels
        # TODO 3: Handle indentation increases (push to stack, nest deeper)
        # TODO 4: Handle indentation decreases (pop from stack, return to parent)
        # TODO 5: Parse mappings (key: value) and sequences (- item)
        # TODO 6: Handle flow syntax for inline collections
        pass
    
    def _parse_mapping_line(self, line: str, indent_level: int) -> tuple:
        """Parse 'key: value' mapping line."""
        # TODO: Split on first colon character
        # TODO: Trim whitespace from key and value parts
        # TODO: Handle quoted keys and values
        # TODO: Perform type inference on value part
        pass
    
    def _calculate_indentation(self, line: str) -> int:
        """Calculate indentation level and validate consistency."""
        # TODO: Count leading spaces (forbid tabs in YAML)
        # TODO: Validate indentation is consistent with previous levels
        # TODO: Return indentation level for stack management
        pass
```

#### Milestone Checkpoints

Each milestone includes specific verification points to ensure correct implementation before proceeding to the next phase:

**Milestone 1 Checkpoint (INI Parser):**
- Run: `python -m pytest tests/unit/test_ini_parser.py -v`
- Expected: All basic INI parsing tests pass including sections, comments, quoted strings
- Manual verification: Parse sample INI file and verify nested dictionary structure
- Debug check: Print parsed structure and confirm section nesting is correct

**Milestone 2 Checkpoint (TOML Tokenizer):**
- Run: `python -c "from config_parser.formats.toml_parser import TOMLTokenizer; t = TOMLTokenizer('key = \"value\"'); print(t.tokenize())"`
- Expected: List of tokens including IDENTIFIER, EQUALS, STRING, EOF with correct values
- Manual verification: Tokenize complex TOML sample and verify all token types are recognized
- Debug check: Verify position tracking is accurate by checking token position fields

**Milestone 3 Checkpoint (TOML Parser):**
- Run: `python -m pytest tests/unit/test_toml_parser.py -v`
- Expected: TOML parsing tests pass including tables, arrays-of-tables, dotted keys
- Manual verification: Parse complex TOML file with nested tables and verify structure
- Debug check: Test table redefinition error handling works correctly

**Milestone 4 Checkpoint (YAML Parser):**
- Run: `python -m pytest tests/unit/test_yaml_parser.py -v`
- Expected: YAML subset tests pass including indented blocks, sequences, mappings
- Manual verification: Parse nested YAML structure and verify indentation handling
- Debug check: Test indentation validation correctly rejects malformed input

#### Language-Specific Implementation Hints

**String Processing Efficiency:**
Use string slicing (`source[start:end]`) rather than character-by-character processing where possible. Python's string methods like `str.startswith()`, `str.strip()`, and `str.split()` are optimized for common parsing operations.

**Type Inference Implementation:**
Leverage Python's dynamic typing for simple type inference: use `int()`, `float()`, and `bool()` conversion functions with try-catch blocks to detect appropriate types. For YAML boolean inference, check against common boolean string representations: `{'true', 'false', 'yes', 'no', 'on', 'off'}`.

**Error Context Generation:**
Use `str.splitlines(keepends=True)` to preserve original line endings when generating error context displays. This maintains consistent character position calculations across different line ending styles.

**Dictionary Path Creation:**
For nested dictionary creation from dotted paths like `server.database.host`, use a helper function that creates intermediate dictionary levels as needed:

```python
def set_nested_dict_value(target_dict: dict, dotted_path: str, value: Any) -> None:
    """Set value in nested dictionary using dotted path notation."""
    # TODO: Split path on dots
    # TODO: Navigate/create intermediate dictionary levels
    # TODO: Set final key to value
    # TODO: Handle conflicts with existing non-dict values
    pass
```

**Position Tracking Accuracy:**
Maintain separate line and column counters during tokenization. Increment line counter on `\n` characters and reset column to 1. Increment column counter for all other characters including `\r` and `\t`. Track absolute byte offset separately for efficient source text slicing.

#### Common Implementation Pitfalls

⚠️ **Pitfall: Inconsistent Position Tracking**
Forgetting to update position information during tokenization leads to meaningless error messages. Always call `current_position()` before creating tokens, and ensure `advance()` properly updates line, column, and offset counters. Test position tracking by deliberately introducing syntax errors and verifying error messages point to the correct location.

⚠️ **Pitfall: Format Detection False Positives**
Simple format detection can misidentify files when formats share syntax elements. A TOML file with only key-value pairs might be detected as INI. Use multiple detection criteria and prefer more specific format indicators. Test detection with edge cases like empty files, comment-only files, and ambiguous minimal examples.

⚠️ **Pitfall: Type Inference Inconsistency**
Different formats have different type inference rules. INI typically treats everything as strings unless explicitly converted, TOML has explicit typing, and YAML aggressively infers types. Implement format-specific type inference rather than trying to unify the rules, as the differences are fundamental to each format's design philosophy.

⚠️ **Pitfall: Error Recovery Complexity**
Attempting sophisticated error recovery in a learning project leads to complex parser state management that obscures the core parsing algorithms. Use fail-fast error handling that stops at the first error with detailed diagnostics rather than trying to continue parsing after errors.

The implementation approach emphasizes correctness and educational clarity over performance optimization or advanced features. Focus on making the core parsing algorithms clear and well-tested before considering any optimizations or additional features beyond the defined scope.


## High-Level Architecture

> **Milestone(s):** All milestones (INI Parser, TOML Tokenizer, TOML Parser, YAML Subset Parser) - this section establishes the foundational architecture that supports all format-specific implementations

Configuration file parsing requires a carefully designed architecture that can handle three fundamentally different syntactic approaches while maintaining clean separation of concerns. Think of this architecture as a universal translator that needs to understand multiple languages (INI, TOML, YAML) but always produces the same kind of output (nested dictionaries). The key challenge lies in designing components that are both flexible enough to handle format-specific complexities and structured enough to share common parsing infrastructure.

![System Component Architecture](./diagrams/system-architecture.svg)

The architecture follows a classic **two-phase parsing approach**: lexical analysis (tokenization) followed by syntactic analysis (parsing). However, unlike traditional compilers that target machine code, our parser targets human-readable data structures. This creates unique design challenges around **impedance mismatch** - bridging the gap between human-friendly configuration syntax and machine-friendly nested dictionaries.

### Component Responsibilities

The system divides parsing responsibilities across five core components, each with distinct concerns and well-defined interfaces. This separation enables independent development and testing while maintaining clear data flow boundaries.

**Format Detection Component** serves as the system's entry point, automatically identifying which parser to invoke based on content analysis. Think of it as a postal sorting machine that examines envelope characteristics to determine the correct destination. This component analyzes syntactic markers, structural patterns, and format-specific signatures to make routing decisions without requiring full parsing.

| Responsibility | Description | Input | Output |
|----------------|-------------|-------|---------|
| Content Analysis | Examines file content for format-specific signatures | Raw configuration text | Confidence scores per format |
| Heuristic Application | Applies weighted heuristics for ambiguous cases | Syntactic patterns | Format identification |
| Fallback Strategy | Handles cases where format cannot be determined | Analysis results | Default format selection |
| Error Context | Provides meaningful errors when detection fails | Failed analysis state | Diagnostic information |

**Base Tokenizer Component** provides the foundation for lexical analysis across all formats. Rather than implementing three separate tokenizers, we use inheritance and composition to share common functionality while allowing format-specific extensions. This component handles the universal aspects of tokenization: character stream management, position tracking, and basic token creation.

| Core Method | Parameters | Returns | Description |
|-------------|------------|---------|-------------|
| `current_position()` | None | `Position` | Returns current line, column, and offset |
| `peek(offset=0)` | `offset: int` | `str` | Lookahead without consuming characters |
| `advance()` | None | `str` | Consume character and update position |
| `skip_whitespace()` | None | `None` | Skip spaces and tabs (not newlines) |
| `read_string_literal(quote_char)` | `quote_char: str` | `Token` | Parse quoted string with escape processing |
| `read_number()` | None | `Token` | Parse numeric literal with type inference |

**Format-Specific Parser Components** (INI, TOML, YAML) handle the syntactic analysis phase, converting token streams into structured data. Each parser implements a different parsing strategy optimized for its format's characteristics: line-based parsing for INI, recursive descent for TOML, and indentation-sensitive parsing for YAML.

The parsers share a common responsibility pattern but implement dramatically different algorithms:

| Parser Type | Strategy | Primary Challenge | Key Data Structure |
|-------------|----------|-------------------|-------------------|
| INI Parser | Line-based sequential | Section and key organization | Flat section map |
| TOML Parser | Recursive descent | Table nesting and type system | Symbol table with dotted keys |
| YAML Parser | Indentation stack | Block structure and implicit typing | Indentation level stack |

**Error Handling Component** coordinates error detection, enrichment, and reporting across all parsing phases. This component transforms low-level parsing failures into actionable user feedback with context and suggestions. Think of it as a medical diagnostic system that not only identifies symptoms but also suggests treatments.

| Error Type | Detection Phase | Enrichment Strategy | User Impact |
|------------|----------------|-------------------|-------------|
| `TokenError` | Lexical analysis | Character-level context | Syntax highlighting |
| `SyntaxError` | Structural parsing | Grammar rule violations | Format guidance |
| `StructureError` | Semantic validation | Logical inconsistencies | Schema suggestions |

> **Design Insight**: The key architectural decision is treating tokenization and parsing as separate concerns connected by a well-defined token stream interface. This enables us to swap tokenizers independently of parsers and simplifies testing by allowing isolated validation of lexical analysis separate from structural parsing.

**Architecture Decision: Shared Tokenizer Base vs Format-Specific Tokenizers**

> **Decision: Shared Base Tokenizer with Format-Specific Extensions**
> - **Context**: Configuration formats share many tokenization patterns (strings, numbers, identifiers) but have format-specific requirements (TOML datetime literals, YAML flow syntax, INI comment styles)
> - **Options Considered**:
>   1. Completely separate tokenizers for each format
>   2. Single universal tokenizer handling all formats
>   3. Base tokenizer class with format-specific extensions
> - **Decision**: Base tokenizer class with format-specific extensions
> - **Rationale**: Maximizes code reuse for common patterns while preserving flexibility for format-specific requirements. Reduces testing burden and maintenance overhead compared to separate tokenizers, while avoiding the complexity explosion of a single universal tokenizer trying to handle all format variations
> - **Consequences**: Requires careful interface design to support extension points. Creates inheritance relationships that must be managed, but significantly reduces code duplication and provides consistent position tracking and error reporting across formats

**Architecture Decision: Token Stream vs Parse Tree Interface**

> **Decision: Token Stream Interface Between Components**
> - **Context**: Need to connect tokenizers with parsers while maintaining component independence and testability
> - **Options Considered**:
>   1. Direct character stream parsing (no separate tokenization)
>   2. Full parse tree generation before data structure creation
>   3. Token stream interface with lazy evaluation
> - **Decision**: Token stream interface with lazy evaluation
> - **Rationale**: Token streams provide the right level of abstraction - more structured than character streams but lighter than full parse trees. Enables independent testing of tokenization logic and parser logic. Supports efficient memory usage through lazy token generation
> - **Consequences**: Requires careful token type design to support all formats. Creates additional interface complexity but enables better separation of concerns and more focused testing

### Recommended Module Structure

The codebase organization reflects the architectural separation of concerns while promoting code reuse and maintainability. The structure supports independent development of format-specific components while sharing common infrastructure.

```
config_parser/
├── __init__.py                 ← Main public API and format detection
├── core/                       ← Shared infrastructure
│   ├── __init__.py
│   ├── tokens.py              ← Token types and base tokenizer
│   ├── errors.py              ← Error hierarchy and context generation
│   ├── positions.py           ← Position tracking utilities
│   └── base_parser.py         ← Common parsing utilities
├── formats/                    ← Format-specific implementations
│   ├── __init__.py
│   ├── ini/
│   │   ├── __init__.py
│   │   ├── tokenizer.py       ← INI-specific tokenization
│   │   ├── parser.py          ← INI parsing logic
│   │   └── validator.py       ← INI-specific validation
│   ├── toml/
│   │   ├── __init__.py
│   │   ├── tokenizer.py       ← TOML tokenization with datetime/multiline
│   │   ├── parser.py          ← Recursive descent TOML parser
│   │   ├── tables.py          ← Table and array-of-tables handling
│   │   └── validator.py       ← TOML semantic validation
│   └── yaml/
│       ├── __init__.py
│       ├── tokenizer.py       ← YAML tokenization with indentation
│       ├── parser.py          ← Indentation-sensitive parser
│       ├── flow.py            ← Flow syntax handling
│       └── validator.py       ← YAML subset validation
├── utils/                      ← Utility functions
│   ├── __init__.py
│   ├── detection.py           ← Format detection heuristics
│   ├── type_inference.py      ← Automatic type conversion
│   └── string_utils.py        ← String processing utilities
└── tests/                      ← Comprehensive test suite
    ├── __init__.py
    ├── unit/                  ← Component-level tests
    ├── integration/           ← Cross-component tests
    ├── fixtures/              ← Test configuration files
    └── benchmarks/            ← Performance tests
```

**Module Responsibility Distribution** ensures each module has a single, well-defined purpose while maintaining clear dependency relationships:

| Module Category | Primary Responsibility | Dependencies | Extension Points |
|------------------|----------------------|-------------|------------------|
| `core/` | Shared parsing infrastructure | None (foundation layer) | Token types, error types |
| `formats/` | Format-specific parsing logic | Depends on `core/` | New format implementations |
| `utils/` | Cross-cutting utilities | Depends on `core/` | Detection heuristics |
| Main API | Public interface and orchestration | All modules | Configuration options |

**Architecture Decision: Flat vs Nested Module Structure**

> **Decision: Nested Module Structure with Format Separation**
> - **Context**: Need to organize format-specific code while maintaining discoverability and avoiding namespace pollution
> - **Options Considered**:
>   1. Flat structure with format prefixes (ini_parser.py, toml_parser.py)
>   2. Single formats.py module with all implementations
>   3. Nested structure with format-specific packages
> - **Decision**: Nested structure with format-specific packages
> - **Rationale**: Each format has multiple related files (tokenizer, parser, validator), making packages natural organizational units. Supports independent development and testing of formats. Easier to add new formats without impacting existing code. Cleaner imports and namespace management
> - **Consequences**: Slightly more complex import paths but much better organization. Enables format-specific testing and development workflows. Supports future packaging of formats as separate distributions if needed

The module structure supports several key **extensibility patterns**:

**Format Addition Pattern**: Adding new formats requires only creating a new package under `formats/` with the standard tokenizer/parser/validator structure. The main API automatically discovers new formats through package introspection.

**Component Extension Pattern**: Each core component can be extended through inheritance or composition. For example, adding new token types requires extending the `TokenType` enum and updating format-specific tokenizers to recognize them.

**Utility Integration Pattern**: Cross-cutting concerns like caching, streaming, or validation can be added through the `utils/` package without impacting core parsing logic.

**Dependency Management** follows a strict layered architecture:

```
┌─────────────────┐
│   Main API      │ ← Public interface, format detection
├─────────────────┤
│   Formats/      │ ← Format-specific parsers and tokenizers
├─────────────────┤
│   Utils/        │ ← Cross-cutting utilities and helpers
├─────────────────┤
│   Core/         │ ← Foundation: tokens, errors, base classes
└─────────────────┘
```

This layered approach ensures that core components remain stable while format-specific implementations can evolve independently. Dependencies flow downward only - core components never depend on format-specific code, ensuring reusability and testability.

**Architecture Decision: Package-Level vs Module-Level Organization**

> **Decision: Package-Level Organization for Format Implementations**
> - **Context**: Format implementations require multiple related modules (tokenizer, parser, validator) that should be grouped together
> - **Options Considered**:
>   1. All format code in single modules (ini.py, toml.py, yaml.py)
>   2. Package per format with internal module separation
>   3. Mixed approach with simple formats as modules, complex as packages
> - **Decision**: Package per format with internal module separation
> - **Rationale**: Even simple formats like INI benefit from separation of tokenization and parsing concerns. Consistent structure makes the codebase more predictable. Packages provide natural extension points for format-specific utilities and validation. Supports independent testing strategies per format
> - **Consequences**: More files and directories but much better organization. Enables format-specific development workflows and testing strategies. Makes it easier to contribute format-specific improvements without understanding other formats

### Implementation Guidance

The following guidance provides concrete steps for implementing the high-level architecture using Python, focusing on establishing the foundational structure that will support all format-specific implementations.

**A. Technology Recommendations**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Tokenization | Manual string parsing with position tracking | Regular expressions with compiled patterns |
| Error Handling | Basic exception hierarchy with string messages | Rich error objects with context and suggestions |
| Type Inference | Simple isinstance checks and string conversion | Configurable type system with custom converters |
| Testing | unittest with manual test data | pytest with property-based testing |
| Performance | Direct implementation focused on correctness | Caching, lazy evaluation, and memory optimization |

**B. Core Infrastructure Starter Code**

**Position Tracking (`core/positions.py`)**:
```python
"""Position tracking utilities for parsing error reporting."""

from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Position:
    """Represents a position in the source text with line, column, and offset."""
    line: int
    column: int
    offset: int
    
    def __str__(self) -> str:
        return f"line {self.line}, column {self.column}"

def create_error_context(source: str, position: Position, context_lines: int = 2) -> str:
    """Generate visual error context showing the error location in source text."""
    lines = source.split('\n')
    if position.line <= 0 or position.line > len(lines):
        return "Invalid position"
    
    # Calculate context window
    start_line = max(0, position.line - context_lines - 1)
    end_line = min(len(lines), position.line + context_lines)
    
    context_parts = []
    for i in range(start_line, end_line):
        line_num = i + 1
        line_content = lines[i]
        prefix = ">>> " if line_num == position.line else "    "
        context_parts.append(f"{prefix}{line_num:4d} | {line_content}")
        
        # Add pointer line for error position
        if line_num == position.line:
            pointer = " " * (len(prefix) + 7 + position.column - 1) + "^"
            context_parts.append(pointer)
    
    return "\n".join(context_parts)
```

**Error Hierarchy (`core/errors.py`)**:
```python
"""Error types for configuration parsing with context and suggestions."""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ParseError(Exception):
    """Base class for all parsing errors with position and suggestion support."""
    message: str
    position: Optional[Position] = None
    suggestion: Optional[str] = None
    
    def __str__(self) -> str:
        parts = [self.message]
        if self.position:
            parts.append(f" at {self.position}")
        if self.suggestion:
            parts.append(f"\nSuggestion: {self.suggestion}")
        return "".join(parts)

@dataclass
class TokenError(ParseError):
    """Errors during lexical analysis (tokenization)."""
    pass

@dataclass
class SyntaxError(ParseError):
    """Errors during syntactic analysis (parsing structure)."""
    pass

@dataclass
class StructureError(ParseError):
    """Errors in logical structure (semantic validation)."""
    pass
```

**Token System (`core/tokens.py`)**:
```python
"""Token definitions and base tokenizer for all configuration formats."""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, List, Optional
from .positions import Position

class TokenType(Enum):
    """All token types needed across INI, TOML, and YAML formats."""
    # Literal values
    STRING = auto()
    NUMBER = auto() 
    BOOLEAN = auto()
    IDENTIFIER = auto()
    
    # Operators and punctuation
    EQUALS = auto()         # =
    COLON = auto()          # :
    COMMA = auto()          # ,
    DOT = auto()            # .
    
    # Structure markers
    SECTION_START = auto()  # [
    SECTION_END = auto()    # ]
    ARRAY_START = auto()    # [ (in value context)
    ARRAY_END = auto()      # ] (in value context)
    OBJECT_START = auto()   # {
    OBJECT_END = auto()     # }
    
    # Whitespace and control
    NEWLINE = auto()
    INDENT = auto()         # YAML indentation increase
    DEDENT = auto()         # YAML indentation decrease
    BLOCK_SEQUENCE = auto() # YAML list item marker (-)
    
    # Special tokens
    COMMENT = auto()
    EOF = auto()
    INVALID = auto()

@dataclass
class Token:
    """A single token with type, value, position, and original text."""
    type: TokenType
    value: Any
    position: Position
    raw_text: str

class BaseTokenizer:
    """Foundation tokenizer providing common functionality for all formats."""
    
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
    
    def current_position(self) -> Position:
        """Returns current parsing position."""
        # TODO: Return Position object with current line, column, offset
        pass
    
    def peek(self, offset: int = 0) -> str:
        """Lookahead without consuming characters."""
        # TODO: Return character at position + offset, or EOF_MARKER if beyond end
        # TODO: Handle negative offsets for lookbehind
        pass
    
    def advance(self) -> str:
        """Consume character and update position tracking."""
        # TODO: Get current character before advancing
        # TODO: Update self.position, handle newline for line/column tracking
        # TODO: Return consumed character or EOF_MARKER
        pass
    
    def skip_whitespace(self) -> None:
        """Skip spaces and tabs but preserve newlines for structure."""
        # TODO: Advance while current character is space or tab
        # TODO: Do NOT skip newlines - they're structurally significant
        pass
    
    def read_string_literal(self, quote_char: str) -> Token:
        """Parse quoted string with escape sequence processing."""
        # TODO: Advance past opening quote
        # TODO: Build string value while processing escape sequences
        # TODO: Handle \n, \t, \r, \\, \", \' escape sequences  
        # TODO: Detect unterminated strings and create appropriate error
        # TODO: Return STRING token with processed value
        pass
    
    def read_number(self) -> Token:
        """Parse numeric literal with automatic type inference."""
        # TODO: Collect digits, handle decimal points for floats
        # TODO: Handle scientific notation (1e5, 1E-3)
        # TODO: Handle integer underscores in TOML (1_000_000)
        # TODO: Convert to appropriate Python type (int, float)
        # TODO: Return NUMBER token with converted value
        pass
    
    def tokenize(self) -> List[Token]:
        """Main tokenization entry point - implemented by format-specific subclasses."""
        # TODO: This is the main loop - format-specific tokenizers override this
        # TODO: Call appropriate read_* methods based on current character
        # TODO: Handle format-specific token types and syntax
        # TODO: Return complete list of tokens including final EOF
        raise NotImplementedError("Subclasses must implement tokenize()")

# Useful constants
EOF_MARKER = '\0'
```

**C. Main API Structure (`__init__.py`)**:
```python
"""Main configuration parser API with automatic format detection."""

from typing import Optional, Dict, Any
from .utils.detection import detect_format
from .formats.ini.parser import INIParser
from .formats.toml.parser import TOMLParser  
from .formats.yaml.parser import YAMLParser

def parse_config(content: str, format: Optional[str] = None) -> Dict[str, Any]:
    """Main configuration parsing entry point with automatic format detection.
    
    Args:
        content: Raw configuration file content
        format: Optional format override ('ini', 'toml', 'yaml')
    
    Returns:
        Nested dictionary representing the configuration structure
        
    Raises:
        ParseError: When content cannot be parsed in the specified/detected format
    """
    # TODO: If format not specified, call detect_format(content)
    # TODO: Create appropriate parser instance based on format
    # TODO: Call parser.parse(content) and return result
    # TODO: Wrap parser-specific errors in generic ParseError with context
    pass

# Export main API
__all__ = ['parse_config', 'detect_format', 'ParseError', 'TokenError', 'SyntaxError', 'StructureError']
```

**D. Format Detection Starter (`utils/detection.py`)**:
```python
"""Automatic configuration format detection using content analysis."""

import re
from typing import Dict

def detect_format(content: str) -> str:
    """Detect configuration format from content using weighted heuristics.
    
    Returns:
        Format string: 'ini', 'toml', or 'yaml'
    """
    scores = {
        'ini': _score_ini(content),
        'toml': _score_toml(content), 
        'yaml': _score_yaml(content)
    }
    
    # Return format with highest confidence score
    return max(scores.keys(), key=lambda k: scores[k])

def _score_ini(content: str) -> float:
    """Calculate confidence score for INI format."""
    score = 0.0
    # TODO: Look for [section] headers - strong INI indicator
    # TODO: Look for key=value pairs - moderate indicator
    # TODO: Look for ; comments - moderate INI indicator
    # TODO: Penalize TOML-specific syntax like [[array]]
    # TODO: Penalize YAML-specific syntax like list items with -
    return score

def _score_toml(content: str) -> float:
    """Calculate confidence score for TOML format."""
    score = 0.0  
    # TODO: Look for [[array.of.tables]] - very strong TOML indicator
    # TODO: Look for dotted.keys = value - strong TOML indicator
    # TODO: Look for inline tables { key = value } - strong indicator
    # TODO: Look for TOML datetime formats - moderate indicator
    # TODO: Penalize YAML indentation patterns
    return score

def _score_yaml(content: str) -> float:
    """Calculate confidence score for YAML format."""  
    score = 0.0
    # TODO: Analyze indentation patterns - strong YAML indicator
    # TODO: Look for list items with - prefix - strong indicator  
    # TODO: Look for key: value with colon - moderate indicator
    # TODO: Look for flow syntax [list] or {map} - moderate indicator
    # TODO: Penalize INI [sections] and TOML [[arrays]]
    return score
```

**E. Development Workflow Recommendations**

**Phase 1 - Foundation Setup**:
1. Implement the core infrastructure (positions, errors, tokens) completely
2. Create empty format packages with proper `__init__.py` files
3. Implement basic format detection with simple heuristics
4. Set up comprehensive testing structure with fixtures

**Phase 2 - Format Implementation Order**:
1. Start with INI parser (simplest format, builds confidence)
2. Move to TOML tokenizer (introduces complex tokenization concepts)  
3. Implement TOML parser (most complex parsing logic)
4. Finish with YAML parser (different paradigm, indentation-sensitive)

**Phase 3 - Integration and Polish**:
1. Connect all formats through main API
2. Enhance format detection with real-world test cases
3. Add comprehensive error handling with context
4. Performance optimization and edge case handling

**F. Common Architecture Pitfalls**

⚠️ **Pitfall: Mixing Tokenization and Parsing Logic**
Many implementations blur the line between lexical and syntactic analysis, making both components harder to test and debug. Keep tokenizers focused purely on character-to-token conversion without understanding structural meaning.

⚠️ **Pitfall: Inconsistent Error Position Tracking**
Failing to maintain accurate position information throughout the parsing pipeline makes debugging nearly impossible. Every token must carry position information, and every error must reference the relevant position.

⚠️ **Pitfall: Format-Specific Code in Shared Components**
Putting TOML-specific logic in the base tokenizer or YAML-specific handling in error reporting breaks the architectural separation. Keep shared components truly format-agnostic.

⚠️ **Pitfall: Inadequate Interface Design**
Designing token types or error interfaces that work for one format but need special cases for others indicates insufficient upfront analysis. Design interfaces to handle the union of all format requirements from the beginning.


## Data Model

> **Milestone(s):** All milestones (INI Parser, TOML Tokenizer, TOML Parser, YAML Subset Parser) - defines the core data structures used throughout all parsing implementations

The data model forms the foundation of our multi-format configuration parser, defining how we represent information as it flows through the parsing pipeline. Think of the data model as the common language that all components speak - just as different human languages can express the same concepts using different words and grammar, our INI, TOML, and YAML parsers must ultimately produce the same structured data despite processing fundamentally different syntactic approaches.

The data model consists of three primary layers, each serving a distinct purpose in the parsing pipeline. The **token layer** represents the lowest-level meaningful units extracted from raw text - individual symbols, keywords, literals, and operators that carry semantic meaning. The **parse tree layer** provides an intermediate structural representation that captures the syntactic relationships between tokens while remaining close to the original format's grammar. Finally, the **unified output layer** presents a consistent nested dictionary structure that client code can work with regardless of which configuration format was parsed.

This layered approach addresses one of the core challenges in multi-format parsing: the **impedance mismatch** between human-readable configuration syntax and machine-processable data structures. Each format has evolved different conventions for expressing hierarchy, data types, and structural relationships. INI files use section headers with flat key-value pairs, TOML employs table definitions with dotted key paths, and YAML relies on indentation-sensitive block structures. Our data model must bridge these syntactic differences while preserving the semantic intent of each format.

![Token Type Hierarchy](./diagrams/token-types.svg)

### Token Type Definitions

The tokenization layer transforms character streams into typed tokens that carry both lexical information and positional context. Understanding tokenization requires thinking of it as **pattern recognition with state tracking** - the tokenizer examines character sequences and classifies them into meaningful categories while maintaining awareness of where each token originated in the source text.

Every token in our system carries four essential pieces of information: its semantic type (what kind of language construct it represents), its processed value (the meaningful content after handling escapes and type conversion), its source position (for error reporting), and its raw text (the exact character sequence from the original input). This comprehensive token representation enables sophisticated error reporting and supports advanced features like syntax highlighting or source-to-output mapping.

The `TokenType` enumeration defines all possible token categories needed across INI, TOML, and YAML formats. This unified token vocabulary allows us to share tokenization logic where formats overlap while maintaining format-specific token types for unique constructs.

| Token Type | Description | Example Raw Text | Processed Value | Used By |
|------------|-------------|------------------|-----------------|---------|
| `STRING` | Quoted or unquoted string literal | `"hello\nworld"` | `hello\nworld` | All formats |
| `NUMBER` | Numeric literal (integer or float) | `42`, `3.14`, `1_000` | `42`, `3.14`, `1000` | TOML, YAML |
| `BOOLEAN` | Boolean literal | `true`, `false`, `yes`, `no` | `True`, `False` | TOML, YAML |
| `IDENTIFIER` | Unquoted name or key | `hostname`, `database` | `hostname`, `database` | All formats |
| `EQUALS` | Assignment operator | `=` | `=` | INI, TOML |
| `COLON` | Key-value separator | `:` | `:` | INI, YAML |
| `NEWLINE` | Line terminator | `\n`, `\r\n` | `\n` | All formats |
| `EOF` | End of file marker | (none) | `EOF_MARKER` | All formats |
| `COMMENT` | Comment text | `# This is a comment` | `This is a comment` | All formats |
| `SECTION_START` | INI section opening | `[` | `[` | INI |
| `SECTION_END` | INI section closing | `]` | `]` | INI |
| `ARRAY_START` | Array opening bracket | `[` | `[` | TOML, YAML |
| `ARRAY_END` | Array closing bracket | `]` | `]` | TOML, YAML |
| `OBJECT_START` | Object opening brace | `{` | `{` | TOML, YAML |
| `OBJECT_END` | Object closing brace | `}` | `}` | TOML, YAML |
| `COMMA` | Element separator | `,` | `,` | TOML, YAML |
| `DOT` | Dotted key separator | `.` | `.` | TOML |
| `INDENT` | Increased indentation level | (whitespace) | indent level | YAML |
| `DEDENT` | Decreased indentation level | (whitespace) | dedent count | YAML |
| `BLOCK_SEQUENCE` | YAML sequence marker | `-` | `-` | YAML |
| `INVALID` | Malformed or unrecognized token | `@#$%` | (original text) | All formats |

The `Position` structure tracks location information for every token, enabling precise error reporting and source mapping. Position tracking must handle the complexities of different line ending conventions (Unix `\n`, Windows `\r\n`, classic Mac `\r`) while maintaining accurate column counting in the presence of tab characters and Unicode code points.

| Field | Type | Description |
|-------|------|-------------|
| `line` | `int` | Line number (1-based) in source text |
| `column` | `int` | Column number (1-based) in source line |
| `offset` | `int` | Absolute character offset (0-based) from start of input |

The `Token` structure represents individual lexical units with their complete context. The separation between `value` and `raw_text` enables clean processing of escape sequences, type conversion, and normalization while preserving the original source text for error reporting and debugging.

| Field | Type | Description |
|-------|------|-------------|
| `type` | `TokenType` | Semantic category of this token |
| `value` | `Any` | Processed value after escape handling and type conversion |
| `position` | `Position` | Source location where this token originated |
| `raw_text` | `str` | Exact character sequence from input (before processing) |

> **Design Insight**: The distinction between `value` and `raw_text` in tokens is crucial for debugging and error reporting. When a user writes `"hello\nworld"` in their configuration file, they intend the string value `hello\nworld` (with a literal newline), but error messages should reference the original `"hello\nworld"` text they actually typed.

**Context sensitivity** presents one of the most significant challenges in tokenization across multiple formats. The same character sequence can represent different token types depending on its syntactic context. For example, the character `[` represents a section header in INI files, an array literal in TOML expressions, a table array definition in TOML headers, and a flow sequence in YAML. Our tokenizer design must either maintain sufficient context to disambiguate these cases or defer disambiguation to the parsing layer.

> **Decision: Context-Sensitive vs Context-Free Tokenization**
> - **Context**: Different formats assign different meanings to the same characters depending on syntactic position
> - **Options Considered**: 
>   1. Context-sensitive tokenizer that tracks parsing state and emits different tokens based on context
>   2. Context-free tokenizer that emits generic tokens and lets parsers handle disambiguation
>   3. Format-specific tokenizers with no shared token vocabulary
> - **Decision**: Context-free tokenizer with parser-level disambiguation
> - **Rationale**: Context-free tokenization is simpler to implement, test, and debug. Parser-level disambiguation allows each format parser to apply its own interpretation rules without complex tokenizer state management.
> - **Consequences**: Parsers must handle some ambiguous tokens, but tokenization logic remains clean and format-agnostic where possible.

**Lexical ambiguity** occurs when the same character sequence could represent multiple token types even within a single format. TOML demonstrates this complexity with its multiple string literal syntaxes: basic strings (`"hello"`), literal strings (`'hello'`), multi-line basic strings (`"""hello"""`), and multi-line literal strings (`'''hello'''`). Each syntax has different rules for escape sequence processing and line handling.

String literal tokenization requires careful state machine implementation to handle nested quotes, escape sequences, and multi-line continuation. The tokenizer must track whether it's inside a quoted string, which quote character initiated the string, whether escape processing is active, and how to handle embedded newlines.

| String Type | Quote Style | Escape Processing | Multi-line Support | Example |
|-------------|-------------|-------------------|-------------------|---------|
| TOML Basic | `"..."` | Yes (backslash escapes) | Single line only | `"hello\nworld"` |
| TOML Literal | `'...'` | No (raw text) | Single line only | `'C:\Users\name'` |
| TOML Multi-Basic | `"""..."""` | Yes (backslash escapes) | Yes (preserve newlines) | `"""line 1\nline 2"""` |
| TOML Multi-Literal | `'''...'''` | No (raw text) | Yes (preserve newlines) | `'''raw\text\here'''` |
| YAML Double | `"..."` | Yes (limited escape set) | With continuation | `"folded\nstring"` |
| YAML Single | `'...'` | Limited (only `''` → `'`) | With continuation | `'don''t escape'` |
| INI Quoted | `"..."` or `'...'` | Format-dependent | Usually single line | `"value with spaces"` |

**Type inference** during tokenization attempts to classify numeric and boolean literals without requiring explicit type annotations. This automatic classification simplifies the user experience but introduces complexity in handling edge cases and format-specific type systems.

Numeric type inference must distinguish between integers, floating-point numbers, and special numeric formats like TOML's integer underscores (`1_000_000`) or YAML's sexagesimal numbers (`90:30:15`). Boolean inference varies significantly between formats - YAML recognizes `yes`, `no`, `on`, `off`, `true`, and `false`, while TOML only accepts `true` and `false`.

> **Decision: Tokenization-Level vs Parse-Level Type Inference**
> - **Context**: Determining whether `"42"`, `42`, `true`, and `"true"` represent different data types
> - **Options Considered**:
>   1. Tokenizer performs type inference and emits strongly-typed tokens
>   2. Tokenizer emits raw values and parsers handle type inference
>   3. No type inference - everything remains as strings until explicit conversion
> - **Decision**: Tokenizer performs basic type inference for clearly-typed literals
> - **Rationale**: Type information at the token level simplifies parser logic and enables better error messages. Ambiguous cases can still be deferred to parsers.
> - **Consequences**: Tokenizer becomes slightly more complex but parsers can focus on structural analysis rather than type classification.

![Data Structure Transformation](./diagrams/data-structure-mapping.svg)

### Parse Tree Structure

The parse tree provides an intermediate structural representation that captures the hierarchical relationships between tokens while remaining faithful to each format's grammatical structure. Think of the parse tree as a **syntax-aware scaffolding** that organizes tokens according to their grammatical roles before transforming them into the final unified data structure.

Parse trees serve several critical functions in our architecture. They provide a stable interface between format-specific parsing logic and the unified output generation. They enable sophisticated error recovery by maintaining partial structural information even when parsing fails. They support advanced features like preserving comments, maintaining key ordering, or implementing syntax-aware editing operations.

The parse tree structure must accommodate the diverse grammatical approaches of our target formats while providing sufficient commonality for shared processing logic. INI files have a simple two-level hierarchy of sections containing key-value pairs. TOML supports arbitrary nesting through dotted keys and table definitions, with complex rules for table redefinition and array-of-tables. YAML uses indentation-sensitive block structures with implicit type inference and multiple syntactic styles for the same logical constructs.

> **Decision: Format-Specific vs Unified Parse Tree Structure**
> - **Context**: Balancing the need for format-faithful representation with shared processing logic
> - **Options Considered**:
>   1. Single unified parse tree structure that all formats must map to
>   2. Format-specific parse tree structures with conversion to unified output
>   3. Hybrid approach with common base structures and format-specific extensions
> - **Decision**: Hybrid approach with common base node types and format-specific specializations
> - **Rationale**: Enables shared processing logic for common constructs while allowing format-specific optimizations and features
> - **Consequences**: Moderate complexity increase but maximum flexibility for format-specific requirements

The base parse tree node structure provides common functionality needed across all formats. Every node tracks its source position for error reporting, maintains references to its constituent tokens for source mapping, and provides a consistent interface for tree traversal and manipulation.

| Field | Type | Description |
|-------|------|-------------|
| `node_type` | `str` | Discriminator indicating specific node subtype |
| `position` | `Position` | Source location of this syntactic construct |
| `tokens` | `List[Token]` | All tokens that contribute to this node |
| `children` | `List[ParseNode]` | Child nodes in syntax tree |
| `metadata` | `dict` | Format-specific annotations and processing hints |

**Section nodes** represent logical groupings of configuration data. In INI files, sections correspond directly to `[section_name]` headers. In TOML, sections represent table definitions including nested tables from dotted keys and array-of-tables. YAML doesn't have explicit sections, but top-level mapping keys function similarly.

| Field | Type | Description |
|-------|------|-------------|
| `section_name` | `str` | Fully-qualified section identifier |
| `key_path` | `List[str]` | Hierarchical path components for nested sections |
| `is_array_element` | `bool` | True if this section represents an array-of-tables entry |
| `key_value_pairs` | `List[KeyValueNode]` | Direct key-value assignments within this section |

**Key-value nodes** represent individual configuration assignments. These nodes must handle the variety of assignment operators (`=`, `:`), value types, and structural patterns used across formats.

| Field | Type | Description |
|-------|------|-------------|
| `key` | `str` | Configuration parameter name |
| `key_path` | `List[str]` | Dotted key components for nested assignment |
| `value` | `ValueNode` | Assigned value (scalar, array, or nested structure) |
| `assignment_operator` | `str` | Operator used for assignment (`=`, `:`) |
| `inline_comment` | `Optional[str]` | Comment appearing on same line as assignment |

**Value nodes** represent the diverse data types and structures supported across configuration formats. Value nodes must accommodate scalars (strings, numbers, booleans), collections (arrays, objects), and format-specific constructs like TOML inline tables or YAML flow sequences.

| Value Node Type | Description | Example Source | Parsed Structure |
|-----------------|-------------|----------------|------------------|
| `ScalarValue` | Single atomic value | `hostname = "server1"` | `{"type": "scalar", "value": "server1"}` |
| `ArrayValue` | Ordered list of values | `ports = [80, 443, 8080]` | `{"type": "array", "elements": [80, 443, 8080]}` |
| `ObjectValue` | Key-value mapping | `{name = "test", port = 80}` | `{"type": "object", "fields": {...}}` |
| `NestedValue` | Reference to nested section | `database.connection.host` | `{"type": "nested", "path": ["database", "connection", "host"]}` |

**Comment nodes** preserve documentation and annotations from the source configuration. Comment handling varies significantly between formats - INI and TOML support both line comments and inline comments, while YAML has more complex comment association rules.

| Field | Type | Description |
|-------|------|-------------|
| `comment_text` | `str` | Comment content (without comment markers) |
| `comment_style` | `str` | Comment syntax used (`#`, `;`, `//`) |
| `attachment` | `str` | How comment relates to nearby content (`above`, `inline`, `below`) |
| `associated_node` | `Optional[ParseNode]` | Parse node this comment documents |

**Array-of-tables nodes** handle TOML's unique `[[table.name]]` syntax for creating arrays of structured objects. This construct has no direct equivalent in INI or YAML, requiring specialized handling in the parse tree.

| Field | Type | Description |
|-------|------|-------------|
| `table_path` | `List[str]` | Hierarchical path to the array location |
| `entries` | `List[SectionNode]` | Individual table entries in the array |
| `entry_order` | `List[int]` | Explicit ordering for array elements |

The parse tree construction process varies significantly between formats due to their different grammatical approaches. **INI parsing** follows a simple line-oriented algorithm: scan for section headers, collect key-value pairs until the next section, handle comments and continuation lines. **TOML parsing** requires recursive descent techniques to handle nested table definitions, dotted key expansion, and the complex interaction between explicit tables and implicit tables created by dotted keys. **YAML parsing** uses an indentation-sensitive algorithm with a stack-based approach to track nesting levels and handle the transition between different indentation depths.

> **Design Insight**: Parse trees serve as a crucial debugging tool during implementation. When parsing produces incorrect output, examining the parse tree structure often reveals whether the problem lies in tokenization (wrong tokens), parsing logic (incorrect tree structure), or output conversion (correct tree, wrong transformation).

### Unified Output Format

The unified output format provides a consistent nested dictionary structure that client applications can work with regardless of which configuration format was originally parsed. Think of the unified output as a **format-agnostic data contract** - it abstracts away the syntactic differences between INI, TOML, and YAML while preserving the semantic intent of the original configuration.

The unified format addresses one of the core value propositions of our multi-format parser: enabling applications to support multiple configuration formats without implementing format-specific processing logic. A web application can read database connection parameters from `config.ini`, `config.toml`, or `config.yaml` using identical code, with the parser handling the format detection and conversion automatically.

The output structure consists of nested Python dictionaries with string keys and mixed-type values. This representation aligns with JSON semantics while supporting additional data types like datetime objects, large integers, and explicit type annotations that some configuration formats provide.

> **Decision: Output Data Structure Design**
> - **Context**: Choosing the unified representation that balances simplicity, type safety, and format compatibility
> - **Options Considered**:
>   1. Pure JSON-compatible structure (strings, numbers, booleans, arrays, objects only)
>   2. Python-native structure with full type system support (datetime, decimal, custom types)
>   3. Hybrid structure with JSON compatibility and optional type annotations
> - **Decision**: Python-native structure with rich type support
> - **Rationale**: Configuration files often contain dates, file paths, and numeric values that benefit from proper typing. Applications can serialize to JSON if needed.
> - **Consequences**: Output is more useful for Python applications but requires explicit conversion for JSON export.

**Nested structure mapping** transforms the various hierarchical representations used by different formats into a consistent nested dictionary structure. INI sections become top-level dictionary keys, TOML tables and dotted keys create nested dictionaries, and YAML block structures map directly to nested dictionaries and arrays.

| Configuration Source | Unified Output Structure |
|---------------------|--------------------------|
| INI: `[database]` `host=localhost` | `{"database": {"host": "localhost"}}` |
| TOML: `database.host = "localhost"` | `{"database": {"host": "localhost"}}` |
| YAML: `database:` `  host: localhost` | `{"database": {"host": "localhost"}}` |

**Type preservation** maintains the semantic intent of typed literals while providing consistent behavior across formats. Numbers remain as integers or floats, booleans are represented as Python `True`/`False`, dates and times use Python `datetime` objects, and strings preserve their exact content including Unicode characters.

The type mapping process must handle format-specific type systems while producing consistent output. TOML has an explicit type system with integers, floats, booleans, strings, arrays, and tables. YAML performs implicit type inference that can produce surprising results (`yes` becomes `True`, `1.0` becomes a float, `010` might become octal). INI files typically treat everything as strings unless explicit conversion is applied.

| Source Format | Type System | Example Literal | Unified Output Type | Unified Output Value |
|---------------|-------------|-----------------|-------------------|---------------------|
| TOML | Explicit typing | `port = 8080` | `int` | `8080` |
| YAML | Implicit inference | `port: 8080` | `int` | `8080` |
| INI | String-based | `port=8080` | `str` or `int` | `"8080"` or `8080` |
| TOML | Boolean literals | `enabled = true` | `bool` | `True` |
| YAML | Boolean inference | `enabled: yes` | `bool` | `True` |
| INI | String representation | `enabled=true` | `str` or `bool` | `"true"` or `True` |

**Array handling** unifies the different approaches formats use for representing ordered collections. TOML uses explicit array syntax with square brackets: `ports = [80, 443, 8080]`. YAML supports both flow syntax `[80, 443, 8080]` and block syntax with dash-prefixed items. INI files don't have native array support, but our parser can recognize comma-separated values or repeated keys as arrays.

| Format | Array Syntax | Source Example | Unified Output |
|--------|-------------|----------------|----------------|
| TOML | Square brackets | `ports = [80, 443]` | `{"ports": [80, 443]}` |
| YAML | Flow sequence | `ports: [80, 443]` | `{"ports": [80, 443]}` |
| YAML | Block sequence | `ports:` `- 80` `- 443` | `{"ports": [80, 443]}` |
| INI | Comma-separated | `ports=80,443` | `{"ports": [80, 443]}` |
| INI | Repeated keys | `port=80` `port=443` | `{"ports": [80, 443]}` |

**Object nesting** creates hierarchical dictionary structures from the various nesting mechanisms used by different formats. The unified output uses string keys and supports arbitrary nesting depth limited only by Python's recursion limits.

TOML dotted keys like `database.connection.host = "localhost"` create nested dictionaries: `{"database": {"connection": {"host": "localhost"}}}`. YAML indentation-based nesting maps directly to nested dictionaries. INI section names can be interpreted as namespace separators, so `[database.connection]` creates similar nesting.

**Key normalization** handles differences in identifier conventions between formats. Some formats are case-sensitive while others are case-insensitive. Key names might use different conventions (camelCase, snake_case, kebab-case) even within the same configuration file.

| Normalization Strategy | Description | Example Transformation | Use Case |
|----------------------|-------------|----------------------|----------|
| `preserve` | Maintain exact key names from source | `userName` → `userName` | Format-aware applications |
| `lowercase` | Convert all keys to lowercase | `userName` → `username` | Case-insensitive lookup |
| `snake_case` | Convert to underscore convention | `userName` → `user_name` | Python applications |
| `kebab_case` | Convert to dash convention | `userName` → `user-name` | Configuration standards |

**Error information preservation** maintains connection between the unified output and the original source text for debugging and error reporting. When possible, the unified output includes metadata about source locations, original formatting, and any type conversions that were applied.

The output structure supports optional metadata attachment through a special `__metadata__` key that contains position information, original formatting, comments, and conversion notes. This metadata enables advanced features like round-trip conversion, syntax-aware editing, and precise error reporting.

| Metadata Field | Type | Description | Example |
|----------------|------|-------------|---------|
| `source_format` | `str` | Original configuration format | `"toml"`, `"yaml"`, `"ini"` |
| `source_position` | `Position` | Location in original file | `{line: 15, column: 8}` |
| `original_key` | `str` | Key name before normalization | `"userName"` → normalized `"user_name"` |
| `type_conversion` | `str` | Applied type conversion | `"string_to_int"`, `"implicit_boolean"` |
| `comments` | `List[str]` | Associated comments | `["# Database configuration", "# Updated 2024-01-15"]` |

The unified output format serves as the foundation for advanced features like configuration validation, schema enforcement, and cross-format conversion. Applications can implement configuration schemas that work across multiple input formats, validate required fields and value ranges, and provide meaningful error messages that reference the original source text.

### Implementation Guidance

The data model implementation requires careful attention to type safety, memory efficiency, and extensibility. Python's dynamic typing system provides flexibility for handling mixed-type configuration values while its dataclass and enum features enable clean, self-documenting data structures.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Token Types | Python Enum | Custom classes with validation |
| Position Tracking | Named tuple | Dataclass with methods |
| Parse Trees | Dictionary-based | Custom node classes with inheritance |
| Type Conversion | Built-in functions | Custom type system with validation |
| Error Handling | Exception classes | Rich error objects with context |

**Recommended File Structure:**
```
config_parser/
  data_model/
    __init__.py              ← export main classes
    tokens.py                ← TokenType, Token, Position
    parse_tree.py            ← Parse tree node classes
    output.py                ← Unified output utilities
    errors.py                ← Error classes and context
    types.py                 ← Type conversion utilities
  tests/
    test_tokens.py           ← Token handling tests
    test_parse_tree.py       ← Parse tree construction tests
    test_output.py           ← Output format tests
```

**Core Token Infrastructure (Complete Implementation):**
```python
from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Optional, List, Union

class TokenType(Enum):
    """Token types for all supported configuration formats."""
    STRING = auto()
    NUMBER = auto()
    BOOLEAN = auto()
    IDENTIFIER = auto()
    EQUALS = auto()
    COLON = auto()
    NEWLINE = auto()
    EOF = auto()
    COMMENT = auto()
    SECTION_START = auto()
    SECTION_END = auto()
    ARRAY_START = auto()
    ARRAY_END = auto()
    OBJECT_START = auto()
    OBJECT_END = auto()
    COMMA = auto()
    DOT = auto()
    INDENT = auto()
    DEDENT = auto()
    BLOCK_SEQUENCE = auto()
    INVALID = auto()

@dataclass(frozen=True)
class Position:
    """Source position for error reporting and debugging."""
    line: int          # 1-based line number
    column: int        # 1-based column number  
    offset: int        # 0-based absolute offset
    
    def __str__(self) -> str:
        return f"line {self.line}, column {self.column}"

@dataclass(frozen=True)
class Token:
    """Individual lexical unit with complete context."""
    type: TokenType    # Semantic category
    value: Any         # Processed value (after escapes, conversion)
    position: Position # Source location
    raw_text: str      # Original character sequence
    
    def __str__(self) -> str:
        return f"{self.type.name}({self.value!r}) at {self.position}"

# Constants
EOF_MARKER = '\0'
```

**Parse Tree Node Infrastructure (Complete Implementation):**
```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod

@dataclass
class ParseNode(ABC):
    """Base class for all parse tree nodes."""
    node_type: str
    position: Position
    tokens: List[Token] = field(default_factory=list)
    children: List['ParseNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert this node to unified output format."""
        pass

@dataclass
class SectionNode(ParseNode):
    """Configuration section with key-value pairs."""
    section_name: str = ""
    key_path: List[str] = field(default_factory=list)
    is_array_element: bool = False
    key_value_pairs: List['KeyValueNode'] = field(default_factory=list)
    
    def __post_init__(self):
        self.node_type = "section"

@dataclass  
class KeyValueNode(ParseNode):
    """Individual configuration assignment."""
    key: str = ""
    key_path: List[str] = field(default_factory=list)
    value: Optional['ValueNode'] = None
    assignment_operator: str = "="
    inline_comment: Optional[str] = None
    
    def __post_init__(self):
        self.node_type = "key_value"

@dataclass
class ValueNode(ParseNode):
    """Configuration value of various types."""
    value_type: str = "scalar"  # scalar, array, object, nested
    processed_value: Any = None
    
    def __post_init__(self):
        self.node_type = "value"
```

**Error Handling Infrastructure (Complete Implementation):**
```python
from typing import Optional, List

class ParseError(Exception):
    """Base class for all parsing errors."""
    def __init__(self, message: str, position: Optional[Position] = None, 
                 suggestion: Optional[str] = None):
        self.message = message
        self.position = position  
        self.suggestion = suggestion
        super().__init__(self.format_error())
    
    def format_error(self) -> str:
        """Format error with position and suggestion."""
        result = self.message
        if self.position:
            result = f"{result} at {self.position}"
        if self.suggestion:
            result = f"{result}\nSuggestion: {self.suggestion}"
        return result

class TokenError(ParseError):
    """Error in tokenization phase."""
    pass

class SyntaxError(ParseError):
    """Error in parsing phase."""
    pass
    
class StructureError(ParseError):
    """Error in semantic analysis phase."""
    pass

def create_error_context(source: str, position: Position, 
                        context_lines: int = 2) -> str:
    """Generate visual error context showing problematic code."""
    lines = source.split('\n')
    start_line = max(0, position.line - context_lines - 1)
    end_line = min(len(lines), position.line + context_lines)
    
    context = []
    for i in range(start_line, end_line):
        line_num = i + 1
        marker = ">>> " if line_num == position.line else "    "
        context.append(f"{marker}{line_num:3d} | {lines[i]}")
        
        # Add column pointer for error line
        if line_num == position.line:
            pointer = " " * (8 + position.column - 1) + "^"
            context.append(pointer)
    
    return "\n".join(context)
```

**Core Tokenizer Skeleton:**
```python
class BaseTokenizer:
    """Base tokenizer for all configuration formats."""
    
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
    
    def current_position(self) -> Position:
        """Get current parsing position."""
        # TODO: Return Position object with current line, column, offset
        pass
    
    def peek(self, offset: int = 0) -> str:
        """Look ahead without consuming characters."""
        # TODO: Return character at position + offset, or EOF_MARKER if beyond end
        # TODO: Handle bounds checking gracefully
        pass
    
    def advance(self) -> str:
        """Consume current character and update position."""
        # TODO: Get current character
        # TODO: Update position counter (handle \n for line tracking)
        # TODO: Update column counter (reset on newline, increment otherwise)
        # TODO: Return consumed character
        pass
    
    def skip_whitespace(self) -> None:
        """Skip non-semantic whitespace."""
        # TODO: Advance through spaces and tabs
        # TODO: Preserve newlines for formats where they're significant
        # TODO: Handle different line ending conventions (\n, \r\n, \r)
        pass
    
    def read_string_literal(self, quote_char: str) -> Token:
        """Parse quoted string with escape sequences."""
        # TODO: Track starting position for token
        # TODO: Collect characters until matching quote
        # TODO: Handle escape sequences (\n, \t, \", \\, etc.)
        # TODO: Handle unterminated string error
        # TODO: Return STRING token with processed value
        pass
    
    def read_number(self) -> Token:
        """Parse numeric literal (integer or float)."""
        # TODO: Track starting position
        # TODO: Collect digits, handle decimal points
        # TODO: Handle scientific notation (1e10, 2.5E-3)
        # TODO: Handle format-specific features (TOML underscores: 1_000)
        # TODO: Convert to appropriate Python type (int or float)
        # TODO: Return NUMBER token with converted value
        pass
    
    def tokenize(self) -> List[Token]:
        """Main tokenization entry point."""
        # TODO: Initialize token list
        # TODO: Loop through source characters
        # TODO: Identify token boundaries and types
        # TODO: Handle format-specific token patterns
        # TODO: Add EOF token at end
        # TODO: Return complete token list
        pass
```

**Output Generation Utilities:**
```python
def normalize_key(key: str, strategy: str = "preserve") -> str:
    """Normalize key names according to specified strategy."""
    # TODO: Implement preserve (no change)
    # TODO: Implement lowercase conversion
    # TODO: Implement snake_case conversion (camelCase -> snake_case)
    # TODO: Implement kebab-case conversion (camelCase -> kebab-case)
    pass

def merge_nested_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two nested dictionaries."""
    # TODO: Handle overlapping keys
    # TODO: Recursively merge nested dictionaries
    # TODO: Handle array values (append vs replace)
    # TODO: Preserve type information
    pass

def convert_parse_tree_to_dict(root: ParseNode) -> Dict[str, Any]:
    """Convert parse tree to unified output format."""
    # TODO: Traverse parse tree depth-first
    # TODO: Convert section nodes to nested dictionaries
    # TODO: Convert key-value nodes to dictionary entries
    # TODO: Handle array-of-tables specially
    # TODO: Apply type conversions and normalizations
    pass
```

**Language-Specific Hints:**
- Use `dataclasses` for clean, self-documenting data structures with automatic `__init__`, `__repr__`, and equality methods
- Leverage `typing` module for precise type annotations that improve IDE support and catch bugs early
- Use `enum.auto()` for token types to avoid manual value assignment and reduce maintenance
- Consider `functools.lru_cache` for expensive operations like regex compilation in tokenizers
- Use `collections.defaultdict` for building nested structures incrementally during parsing

**Milestone Checkpoint:**
After implementing the data model components, verify correct behavior:

1. **Token Creation**: Create tokens of each type and verify the `__str__` representation shows useful information
2. **Position Tracking**: Create positions and verify they format correctly for error messages  
3. **Parse Tree Construction**: Build a simple parse tree manually and verify `to_dict()` conversion works
4. **Error Context**: Generate error context for a sample configuration file and verify the visual formatting
5. **Type Conversions**: Test numeric and boolean conversion with various input formats

Run: `python -m pytest tests/test_data_model.py -v` to verify all data model components work correctly.

**Common Implementation Pitfalls:**

⚠️ **Pitfall: Mutable Default Arguments in Dataclasses**
Using mutable defaults like `tokens: List[Token] = []` causes all instances to share the same list. Use `field(default_factory=list)` instead.

⚠️ **Pitfall: Position Tracking with Unicode**
Counting characters as single units breaks with multi-byte Unicode. Use `len(text.encode('utf-8'))` for byte offsets if needed, or stick to character-based counting for simplicity.

⚠️ **Pitfall: Token Value vs Raw Text Confusion** 
Mixing up processed values and raw text leads to double-escaping or incorrect error messages. Always use `raw_text` for error reporting and `value` for semantic processing.

⚠️ **Pitfall: Deep Copy Issues with Parse Trees**
Parse tree nodes contain references to other nodes, making deep copying complex. Implement custom `copy` methods or use immutable structures where possible.

⚠️ **Pitfall: Enum Comparison Mistakes**
Comparing `TokenType.STRING == "STRING"` fails because enums don't equal their string representations. Use `token.type == TokenType.STRING` or `token.type.name == "STRING"`.


## Tokenizer Component Design

> **Milestone(s):** TOML Tokenizer, with foundational concepts supporting INI Parser and YAML Subset Parser

The tokenizer serves as the lexical analysis engine that transforms raw character streams into meaningful typed tokens with precise position tracking. Think of the tokenizer as a sophisticated pattern recognition system that must simultaneously solve three fundamental challenges: identifying where meaningful units begin and end in continuous text, classifying what type of semantic meaning each unit carries, and maintaining perfect position tracking for error reporting. Unlike simple string splitting, tokenization must handle **lexical ambiguity** where the same character sequence can mean completely different things depending on context—a quote character might start a string literal, escape another quote, or appear as literal content within a different quoting style.

![Token Type Hierarchy](./diagrams/token-types.svg)

The tokenizer operates as a **context-sensitive** state machine that maintains awareness of its current parsing context to resolve these ambiguities. When the tokenizer encounters a quote character, its behavior depends entirely on whether it's currently inside a string literal, what type of string context it's in, and what escape rules apply. This context sensitivity becomes particularly complex when handling the different string literal syntaxes across INI, TOML, and YAML formats, where the same character sequence `"""` might indicate a multiline string start in TOML but could be three separate quoted empty strings in INI context.

The tokenizer must also solve the **impedance mismatch** between human-readable configuration syntax and machine-processable token streams. Configuration formats are designed for human readability, using intuitive conventions like indentation for nesting (YAML) or natural key=value syntax (INI). However, parsers need discrete, classified tokens with explicit type information and precise boundary definitions. The tokenizer bridges this gap by applying format-specific lexical rules that preserve the semantic intent while creating the structured token stream that downstream parsers require.

### Tokenization Mental Model

Understanding tokenization requires thinking about it as **pattern recognition with state tracking** rather than simple character classification. Imagine the tokenizer as a careful reader who must simultaneously identify word boundaries, understand context-dependent meanings, and take detailed notes about location and classification for later reference.

The mental model begins with the concept of a **scanning window** that moves through the character stream one position at a time. At each position, the tokenizer must decide whether the current character continues an existing token, starts a new token, or serves as a delimiter that separates tokens. This decision requires examining not just the current character, but also the **parsing context** maintained in the tokenizer's state machine. The context tracks information like "currently inside a quoted string," "at the start of a line," or "within a comment block."

Consider how a human reader processes the TOML line `name = "John \"Doe\""`. The reader automatically recognizes that the first quote starts a string, the backslash-quote sequence represents an escaped quote within the string content, and the final quote ends the string. The tokenizer must replicate this contextual understanding through explicit state tracking. When it encounters the first quote, it transitions to "inside string literal" state. When it sees the backslash, it transitions to "processing escape sequence" state, reads the escaped quote as literal content, returns to string literal state, and finally recognizes the closing quote as a string terminator.

The **lookahead** concept provides another essential mental model component. The tokenizer often needs to examine upcoming characters to make correct tokenization decisions. When processing the sequence `123.456e-7`, the tokenizer must look ahead to determine whether this represents a single floating-point number token or multiple separate tokens. The decimal point alone isn't sufficient—it must examine the subsequent characters to distinguish between a number continuation and a separate identifier that might follow.

**Position tracking** requires thinking of the tokenizer as maintaining a detailed location journal. Every token must carry precise position information indicating exactly where it appeared in the source text. This information becomes critical for error reporting, IDE integration, and debugging tools. The position tracking must handle format-specific complications like YAML's significant indentation (where column position affects semantic meaning) and multiline string literals (where internal newlines don't advance the logical line number for subsequent tokens).

The **error recovery** mental model treats tokenization errors as opportunities to provide helpful feedback rather than immediate failures. When the tokenizer encounters malformed input like an unterminated string literal, it should attempt to identify the likely intended structure, create appropriate error tokens, and continue processing to find additional issues. Think of this as a careful proofreader who marks errors but continues reading to provide comprehensive feedback rather than stopping at the first mistake.

### Tokenizer Interface Design

The tokenizer interface must provide a clean abstraction that supports both streaming tokenization (processing tokens one at a time) and batch tokenization (processing entire files), while maintaining consistent error handling and position tracking across all supported formats.

| Method Name | Parameters | Returns | Description |
|-------------|------------|---------|-------------|
| `current_position` | none | `Position` | Returns current parsing position with line, column, and byte offset |
| `peek` | `offset=0` | `str` | Look ahead at character without consuming, returns EOF_MARKER at end |
| `advance` | none | `str` | Consume current character, update position tracking, return consumed character |
| `tokenize` | none | `List[Token]` | Main entry point that processes entire input and returns complete token list |
| `next_token` | none | `Token` | Generate and return next token from current position, advance past it |
| `skip_whitespace` | none | `None` | Advance past all non-semantic whitespace characters |
| `read_string_literal` | `quote_char: str` | `Token` | Parse quoted string with escape sequence processing |
| `read_number` | none | `Token` | Parse numeric literal with type inference (int/float) |
| `read_identifier` | none | `Token` | Parse unquoted identifier or keyword |
| `read_comment` | `comment_char: str` | `Token` | Parse comment from delimiter to end of line |
| `create_error_token` | `message: str, raw_text: str` | `Token` | Create INVALID token with error information |
| `is_at_end` | none | `bool` | Check if tokenizer has reached end of input |

The interface design separates concerns between **navigation** methods (`peek`, `advance`, `current_position`) that handle character stream management and **recognition** methods (`read_string_literal`, `read_number`) that identify and extract specific token types. This separation allows the recognition methods to focus on format-specific parsing logic while relying on consistent navigation behavior.

> **Decision: Lookahead Interface Design**
> - **Context**: Different formats require varying amounts of lookahead for tokenization decisions
> - **Options Considered**: Fixed single-character lookahead, unlimited string lookahead, parameterized offset lookahead
> - **Decision**: Parameterized offset lookahead with `peek(offset=0)` method
> - **Rationale**: Provides flexibility for complex tokenization while maintaining predictable performance characteristics
> - **Consequences**: Enables proper handling of complex literals like scientific notation while avoiding unbounded memory usage

The `BaseTokenizer` maintains internal state that supports both the navigation and recognition interfaces:

| Field Name | Type | Description |
|------------|------|-------------|
| `source` | `str` | Complete input text being tokenized |
| `position` | `int` | Current byte offset in source text |
| `line` | `int` | Current line number (1-based) for error reporting |
| `column` | `int` | Current column number (1-based) for error reporting |
| `tokens` | `List[Token]` | Accumulated tokens from tokenization process |

The tokenizer interface supports **streaming processing** through the `next_token` method, allowing parsers to process tokens incrementally without loading the complete token list into memory. This approach becomes important for large configuration files or embedded systems with memory constraints. The streaming interface maintains the same error handling and position tracking guarantees as batch tokenization.

**Error token creation** represents a critical interface design decision. Rather than throwing exceptions immediately upon encountering malformed input, the tokenizer creates `INVALID` tokens that carry error information. This approach allows the parser to collect multiple errors in a single pass and provide comprehensive feedback to users. The error tokens include the malformed text, error description, and precise position information.

![Tokenizer State Machine](./diagrams/tokenizer-state-machine.svg)

The interface supports **format-specific customization** through polymorphism. Each format (INI, TOML, YAML) can extend `BaseTokenizer` and override specific recognition methods while inheriting the common navigation and position tracking behavior. For example, YAML tokenization requires custom `skip_whitespace` behavior that preserves significant indentation, while TOML needs specialized string literal processing for its multiple quoting styles.

### String Literal Handling

String literal tokenization represents the most complex aspect of lexical analysis due to the variety of quoting styles, escape sequence processing, and multiline handling requirements across different configuration formats. The complexity stems from **context sensitivity**—the same character sequence can have completely different meanings depending on the current string parsing state.

The string literal recognition system must handle multiple **quoting styles** with different semantic rules:

| Quote Style | Formats | Escape Processing | Multiline Support | Special Rules |
|-------------|---------|------------------|-------------------|---------------|
| Single quotes `'text'` | INI, TOML, YAML | TOML: none, others: basic | TOML only | TOML literal strings don't process escapes |
| Double quotes `"text"` | All formats | Full escape processing | No | Standard escape sequences: `\n`, `\t`, `\"`, `\\` |
| Triple single `'''text'''` | TOML only | None | Yes | First newline after opening quotes is trimmed |
| Triple double `"""text"""` | TOML only | Full escape processing | Yes | First newline after opening quotes is trimmed |
| No quotes (bare) | All formats | None | Context-dependent | YAML flow scalars, INI values, TOML bare keys |

The **escape sequence processing** requires a nested state machine within string literal parsing. When the tokenizer encounters a backslash character inside a double-quoted string, it must transition to escape processing mode and handle various escape sequences:

1. The tokenizer identifies the backslash as an escape indicator
2. It examines the following character to determine the escape type
3. For standard escapes (`\n`, `\t`, `\r`, `\"`, `\\`), it converts to the appropriate character
4. For Unicode escapes (`\u0041`, `\U00000041`), it processes the hexadecimal digits
5. For invalid escape sequences, it creates an error token with position information
6. It returns to normal string content processing mode

**Multiline string handling** introduces additional complexity because the tokenizer must track logical string content while maintaining accurate position information for error reporting. TOML's triple-quoted strings include special rules about newline handling: the first newline immediately after the opening quotes is automatically trimmed, but subsequent newlines are preserved as content. The tokenizer must implement this logic while correctly updating line and column tracking.

> **Decision: String Literal State Machine Design**
> - **Context**: String parsing requires handling nested state transitions for quotes, escapes, and multiline content
> - **Options Considered**: Single-function string parser, recursive state machine, explicit state tracking with switch statements
> - **Decision**: Explicit state tracking with enumerated states and transition table
> - **Rationale**: Provides clear visibility into parsing state, enables systematic testing, supports debugging
> - **Consequences**: More verbose implementation but significantly easier to debug and extend for new quote styles

The string literal parser maintains explicit state through an enumeration:

| State | Description | Valid Transitions | Exit Conditions |
|-------|-------------|------------------|-----------------|
| `NORMAL` | Processing regular string content | `ESCAPE`, `QUOTE_END` | Closing quote or end of input |
| `ESCAPE` | Processing escape sequence | `NORMAL`, `ERROR` | Valid escape processed or invalid sequence |
| `UNICODE_ESCAPE` | Processing `\uXXXX` sequence | `NORMAL`, `ERROR` | Four hex digits processed or invalid digit |
| `QUOTE_END` | Potential end of multiline string | `NORMAL`, `COMPLETE` | Triple quote completed or false alarm |
| `ERROR` | Invalid sequence encountered | `NORMAL`, `COMPLETE` | Error token created, attempt recovery |
| `COMPLETE` | String literal fully parsed | none | Return completed token |

**Error recovery** in string literal parsing requires special consideration because unterminated strings can consume the entire remaining input. When the tokenizer reaches the end of input while still inside a string literal, it must create an error token that includes all the consumed content and indicates the unterminated string issue. The error token should point to the opening quote position to help users identify where the string began.

The string literal handler must also manage **performance considerations** for large multiline strings. Rather than concatenating characters one at a time (which creates O(n²) complexity), it should identify string boundaries first, then extract the complete content in a single operation. This approach maintains linear performance even for very large configuration files with substantial multiline content.

**Quote character disambiguation** represents another critical challenge. When the tokenizer encounters a quote character, it must determine whether this starts a single-quoted string, a double-quoted string, or potentially a triple-quoted multiline string. This requires lookahead logic that examines the following characters to make the correct determination:

1. Single quote encountered: look ahead two characters to check for triple-quote pattern
2. If triple-quote detected: initialize multiline literal string parsing
3. If not triple-quote: initialize single-quote string parsing
4. Handle edge cases like `''` (empty string) vs `'''` (multiline string start)

### Tokenizer Architecture Decisions

The tokenizer architecture must balance **performance**, **maintainability**, and **extensibility** while handling the diverse requirements of INI, TOML, and YAML formats. The key architectural decisions affect how the tokenizer manages state, processes different formats, and integrates with the overall parsing pipeline.

> **Decision: Single Tokenizer vs Format-Specific Tokenizers**
> - **Context**: Each format has different lexical rules, keywords, and syntactic elements that require specialized handling
> - **Options Considered**: Single universal tokenizer with format flags, completely separate tokenizers, base tokenizer with format-specific extensions
> - **Decision**: Base tokenizer with format-specific extensions through inheritance
> - **Rationale**: Maximizes code reuse for common functionality while allowing format-specific customization where needed
> - **Consequences**: Shared navigation and position tracking logic, but specialized string and numeric literal handling per format

| Approach | Pros | Cons | Chosen? |
|----------|------|------|---------|
| Universal tokenizer | Single codebase, consistent behavior | Complex branching logic, hard to optimize per format | No |
| Separate tokenizers | Format-optimized, clear separation | Code duplication, inconsistent position tracking | No |
| Inheritance-based | Code reuse + specialization | Moderate complexity, clear extension points | **Yes** |

The inheritance-based approach creates a `BaseTokenizer` that handles universal concerns like position tracking, character navigation, and basic token creation. Format-specific tokenizers (`INITokenizer`, `TOMLTokenizer`, `YAMLTokenizer`) extend the base class and override methods that require specialized behavior. This architecture enables sharing common functionality while supporting format-specific requirements like YAML's indentation-sensitive tokenization or TOML's complex string literal rules.

> **Decision: Character Encoding and Unicode Support**
> - **Context**: Configuration files may contain Unicode characters, and different platforms handle encoding differently
> - **Options Considered**: ASCII-only support, UTF-8 with byte processing, full Unicode with character processing
> - **Decision**: Full Unicode support with character-based processing and UTF-8 encoding assumption
> - **Rationale**: Modern configuration files commonly contain Unicode characters for internationalization
> - **Consequences**: More complex position tracking (byte offset vs character offset), but supports real-world usage

The Unicode support decision affects position tracking implementation. The tokenizer must maintain both **character positions** (for human-readable error messages) and **byte offsets** (for efficient file access). When processing multibyte Unicode characters, the character count and byte count diverge, requiring careful tracking of both metrics.

> **Decision: Error Recovery Strategy**
> - **Context**: Malformed configuration files should provide helpful error messages rather than immediate failures
> - **Options Considered**: Fail-fast on first error, collect errors and continue, attempt automatic correction
> - **Decision**: Collect errors in special INVALID tokens and continue tokenization when possible
> - **Rationale**: Allows comprehensive error reporting and better user experience for debugging
> - **Consequences**: More complex parser logic but significantly better error messages

The error recovery approach creates `TokenError` objects that carry both the problematic text and contextual information about what was expected. The tokenizer continues processing after creating error tokens, allowing it to identify multiple issues in a single pass. This approach provides much better user experience compared to fail-fast behavior that requires multiple edit-test cycles to identify all issues.

> **Decision: Memory Management and Token Storage**
> - **Context**: Large configuration files could create memory pressure if all tokens are stored simultaneously
> - **Options Considered**: Store all tokens in memory, streaming tokenization only, hybrid approach with optional storage
> - **Decision**: Hybrid approach with batch tokenization for normal use and streaming interface for large files
> - **Rationale**: Most configuration files are small enough for in-memory processing, but large files need streaming
> - **Consequences**: Two code paths to maintain, but supports both typical usage and edge cases

The memory management decision creates two tokenization modes. The standard `tokenize()` method processes the entire input and returns a complete token list, suitable for typical configuration files under 1MB. The streaming `next_token()` interface allows processing arbitrarily large files by generating tokens on demand without storing the complete list.

**State machine implementation** represents another critical architectural decision. The tokenizer uses explicit state enumeration rather than implicit state in nested function calls. This approach provides better debuggability and makes the tokenization process easier to trace and test. Each state transition is explicitly modeled, making the tokenizer's behavior predictable and testable.

### Common Tokenizer Pitfalls

Tokenizer implementation involves several subtle but critical pitfalls that frequently trap developers, particularly around escape sequence processing, position tracking accuracy, and state management consistency.

⚠️ **Pitfall: Incorrect Escape Sequence Processing**

The most common tokenizer mistake involves improper handling of escape sequences within string literals. Developers often implement escape processing that fails to handle edge cases or processes escapes in contexts where they shouldn't be processed. For example, TOML literal strings (single-quoted) should not process escape sequences—the sequence `'C:\path\to\file'` should preserve the backslashes literally, not interpret them as escape attempts.

The error typically manifests when the tokenizer applies escape processing to all quoted strings regardless of quote type. This breaks TOML literal strings and can cause incorrect parsing of file paths, regular expressions, and other content that intentionally contains backslashes. The fix requires checking the string literal type before applying escape processing: only double-quoted strings in TOML should process escapes, while single-quoted strings preserve backslashes literally.

⚠️ **Pitfall: Position Tracking Inconsistencies**

Position tracking errors create confusing error messages that point to incorrect locations in the source file. The most frequent mistake involves failing to properly update line and column numbers when processing special characters like tabs, carriage returns, and multiline strings. When the tokenizer encounters a tab character, it should advance the column position according to tab width settings (typically 4 or 8 spaces), not increment by one character.

Another common position tracking error occurs with multiline string literals. When processing TOML triple-quoted strings, the tokenizer must correctly update line numbers for each newline within the string content while ensuring that subsequent tokens have accurate position information. The error typically causes error messages to point to the beginning of the multiline string rather than the actual error location.

⚠️ **Pitfall: State Machine Inconsistencies**

State management errors in string literal parsing often cause the tokenizer to get "stuck" in a particular state or fail to properly transition between states. A common example involves handling nested quote characters: when processing `"He said \"hello\""`, the tokenizer must properly transition to escape state when it encounters the backslash, process the escaped quote as content, and return to normal string processing state.

The error typically occurs when developers implement string parsing with ad-hoc conditional logic rather than explicit state tracking. The fix requires implementing a clear state machine with enumerated states and explicit transition conditions. Each state should have well-defined entry conditions, processing behavior, and exit conditions.

⚠️ **Pitfall: Lookahead Buffer Management**

Incorrect lookahead implementation can cause the tokenizer to miss token boundaries or incorrectly classify tokens. A common mistake involves implementing `peek()` that doesn't properly handle the end-of-file condition, causing array index errors or infinite loops when the tokenizer reaches the end of input.

Another lookahead error involves modifying the tokenizer state during lookahead operations. The `peek()` method should be purely observational—it should examine upcoming characters without advancing the current position or changing internal state. Lookahead that accidentally modifies state can cause tokens to be skipped or duplicated.

⚠️ **Pitfall: Comment Handling Edge Cases**

Comment processing errors frequently occur at line boundaries and in interaction with string literals. A common mistake involves recognizing comment delimiters (`#` or `;`) that appear inside string literals as actual comment starts. The line `message = "Error #404: Not found"` should not treat everything after the `#` as a comment—the hash character is part of the string content.

The fix requires checking the current parsing context before processing comment delimiters. Comment recognition should only occur when the tokenizer is not inside a string literal or other quoted context. Additionally, comment processing must properly handle different line ending styles (`\n`, `\r\n`, `\r`) to ensure comments are correctly terminated.

⚠️ **Pitfall: Number Format Recognition**

Numeric literal tokenization often fails to handle edge cases like scientific notation, hex literals, or numbers with underscores (allowed in TOML). A common error involves recognizing `1.23e-4` as three separate tokens (`1.23`, `e`, `-4`) instead of a single floating-point number in scientific notation.

The fix requires implementing proper lookahead in number recognition. When the tokenizer encounters a digit, it must examine the following characters to determine the complete numeric literal extent. This includes handling decimal points, exponent markers (`e` or `E`), sign characters in exponents, and format-specific features like TOML's underscore separators in large numbers.

### Implementation Guidance

The tokenizer implementation requires careful attention to character encoding, state management, and performance optimization. The following guidance provides concrete recommendations for building a robust tokenizer that handles all three configuration formats effectively.

**A. Technology Recommendations**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Character Processing | Basic string indexing with ord() | Unicode-aware processing with unicodedata module |
| State Management | Explicit state variables | State machine with enum states and transition table |
| Position Tracking | Simple line/column counters | Position objects with byte offset, line, column tracking |
| Error Handling | Exception throwing | Error token collection with context |

**B. Recommended File Structure**

```
src/
├── tokenizer/
│   ├── __init__.py              ← exports BaseTokenizer, Token, TokenType
│   ├── base_tokenizer.py        ← BaseTokenizer with core functionality
│   ├── tokens.py                ← Token, TokenType, Position definitions
│   ├── ini_tokenizer.py         ← INITokenizer with format-specific rules
│   ├── toml_tokenizer.py        ← TOMLTokenizer with complex string handling
│   ├── yaml_tokenizer.py        ← YAMLTokenizer with indentation processing
│   └── errors.py                ← TokenError and related error types
├── parsers/                     ← parser components use tokenizer
└── tests/
    └── tokenizer/
        ├── test_base_tokenizer.py
        ├── test_string_literals.py
        └── test_position_tracking.py
```

**C. Infrastructure Starter Code**

Complete token and position definitions that learners can use directly:

```python
from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, Optional

class TokenType(Enum):
    STRING = auto()
    NUMBER = auto()
    BOOLEAN = auto()
    IDENTIFIER = auto()
    EQUALS = auto()
    COLON = auto()
    NEWLINE = auto()
    EOF = auto()
    COMMENT = auto()
    SECTION_START = auto()      # [
    SECTION_END = auto()        # ]
    ARRAY_START = auto()        # [
    ARRAY_END = auto()          # ]
    OBJECT_START = auto()       # {
    OBJECT_END = auto()         # }
    COMMA = auto()
    DOT = auto()
    INDENT = auto()             # YAML indentation increase
    DEDENT = auto()             # YAML indentation decrease
    BLOCK_SEQUENCE = auto()     # YAML - marker
    INVALID = auto()            # Error token

@dataclass
class Position:
    line: int
    column: int
    offset: int
    
    def __str__(self) -> str:
        return f"line {self.line}, column {self.column}"

@dataclass
class Token:
    type: TokenType
    value: Any
    position: Position
    raw_text: str
    
    def __str__(self) -> str:
        return f"{self.type.name}({self.value}) at {self.position}"

# Character constants for tokenizer logic
EOF_MARKER = '\0'
WHITESPACE_CHARS = ' \t\r'
NEWLINE_CHARS = '\n'
QUOTE_CHARS = '"\'`'
NUMBER_START_CHARS = '0123456789+-'
IDENTIFIER_START_CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'
IDENTIFIER_CHARS = IDENTIFIER_START_CHARS + '0123456789-'
```

**D. Core Logic Skeleton Code**

BaseTokenizer implementation with detailed TODO comments for learner implementation:

```python
class BaseTokenizer:
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens = []
    
    def current_position(self) -> Position:
        """Returns current parsing position with line, column, and byte offset."""
        return Position(self.line, self.column, self.position)
    
    def peek(self, offset: int = 0) -> str:
        """Look ahead at character without consuming, returns EOF_MARKER at end."""
        # TODO 1: Calculate target position as self.position + offset
        # TODO 2: Check if target position is beyond source length
        # TODO 3: Return EOF_MARKER if beyond end, otherwise return character at target
        # Hint: Handle negative offsets by returning EOF_MARKER
        pass
    
    def advance(self) -> str:
        """Consume current character, update position tracking, return consumed char."""
        # TODO 1: Check if at end of source, return EOF_MARKER if so
        # TODO 2: Get current character at self.position
        # TODO 3: Increment self.position
        # TODO 4: Update line and column based on character type:
        #         - If newline: increment line, reset column to 1
        #         - If tab: advance column to next tab stop (typically +4 or +8)
        #         - Otherwise: increment column by 1
        # TODO 5: Return the consumed character
        # Hint: Handle both \n and \r\n newline styles
        pass
    
    def skip_whitespace(self) -> None:
        """Advance past all non-semantic whitespace characters."""
        # TODO 1: Loop while current character is in WHITESPACE_CHARS
        # TODO 2: Call advance() for each whitespace character
        # TODO 3: Stop when reaching non-whitespace or EOF_MARKER
        # Note: Don't skip newlines - they may be significant tokens
        pass
    
    def read_string_literal(self, quote_char: str) -> Token:
        """Parse quoted string with escape sequence processing."""
        start_pos = self.current_position()
        # TODO 1: Check if this might be a triple-quoted string (look ahead 2 chars)
        # TODO 2: If triple-quoted, call read_multiline_string instead
        # TODO 3: Advance past opening quote
        # TODO 4: Initialize empty content list for building string
        # TODO 5: Loop until closing quote or EOF:
        #         - If escape char (\), process escape sequence
        #         - If closing quote, break loop
        #         - If newline in single-quoted string, create error token
        #         - Otherwise add character to content
        # TODO 6: Advance past closing quote (if found)
        # TODO 7: Join content list and create STRING token
        # Hint: Different quote types have different escape rules
        pass
    
    def read_number(self) -> Token:
        """Parse numeric literal with type inference (int/float)."""
        # TODO 1: Determine if number starts with sign (+/-)
        # TODO 2: Read integer portion (digits, possibly with underscores in TOML)
        # TODO 3: Check for decimal point, read fractional portion if present
        # TODO 4: Check for exponent (e/E), read exponent with optional sign
        # TODO 5: Determine if result should be int or float based on format
        # TODO 6: Convert string to appropriate numeric type
        # TODO 7: Create NUMBER token with converted value
        # Hint: Handle scientific notation like 1.23e-4
        pass
    
    def tokenize(self) -> list[Token]:
        """Main tokenization entry point - processes entire input."""
        # TODO 1: Initialize empty tokens list
        # TODO 2: Loop until position reaches end of source:
        #         - Skip whitespace
        #         - Identify token type from current character
        #         - Call appropriate read_* method
        #         - Add resulting token to list
        # TODO 3: Add final EOF token
        # TODO 4: Return complete token list
        # Hint: Use character-based dispatch for token type identification
        pass
```

**E. Language-Specific Hints**

- Use `str.isdigit()`, `str.isalpha()`, `str.isalnum()` for character classification
- Handle Unicode properly with `len()` and string slicing - Python handles UTF-8 correctly
- Use `ord()` and `chr()` for escape sequence processing (e.g., `\n` → `chr(10)`)
- Consider using `unicodedata.category()` for advanced Unicode character classification
- Use `enum.auto()` for TokenType to avoid manual numbering
- Implement `__str__` and `__repr__` methods for debugging Token and Position classes

**F. Milestone Checkpoint**

After implementing the base tokenizer:

**Test Command:** `python -m pytest tests/tokenizer/test_base_tokenizer.py -v`

**Expected Output:**
```
test_peek_lookahead PASSED
test_advance_position_tracking PASSED  
test_string_literal_basic PASSED
test_string_literal_escapes PASSED
test_number_parsing PASSED
test_tokenize_complete_input PASSED
```

**Manual Verification:**
```python
tokenizer = BaseTokenizer('key = "value"')
tokens = tokenizer.tokenize()
print([str(token) for token in tokens])
# Expected: ['IDENTIFIER(key) at line 1, column 1', 'EQUALS(=) at line 1, column 5', ...]
```

**Signs of Problems:**
- Position tracking off by one → Check advance() method line/column updates
- Escape sequences not working → Verify quote type checking in read_string_literal
- Infinite loops → Check EOF_MARKER handling in peek() and advance()
- Missing tokens → Verify whitespace skipping doesn't consume significant characters


## INI Parser Component Design

> **Milestone(s):** INI Parser - implements section-based parsing with key-value pairs and comment handling

The INI parser represents our entry point into configuration file parsing, serving as the foundation for understanding core parsing concepts before tackling more complex formats. Think of INI parsing as learning to drive in an empty parking lot before venturing onto busy highways—the fundamental principles of tokenization, state management, and data structure mapping are all present, but in their simplest possible form.

The elegance of INI files lies in their human-readable simplicity: sections enclosed in square brackets, key-value pairs separated by equals signs or colons, and comments prefixed with semicolons or hash symbols. However, this apparent simplicity conceals several parsing challenges that make INI an excellent learning platform for understanding configuration file processing.

### INI Parsing Mental Model: Understanding INI as Section-Based Key-Value Organization

Understanding INI parsing requires thinking about it as a **hierarchical filing system** where documents are organized into labeled folders. Each section header acts like a manila folder tab, and the key-value pairs underneath are the documents filed within that folder. The parser's job is to walk through this filing system sequentially, creating digital folders and filing away documents in the correct locations.

The mental model extends further when we consider that some documents might exist outside of any folder (global keys), some folders might be referenced multiple times throughout the filing system, and some documents might have annotations (comments) that need to be preserved or ignored based on configuration.

This hierarchical organization creates several parsing contexts that must be tracked simultaneously. The **current section context** determines where newly encountered key-value pairs should be stored. The **comment context** affects whether a line should be processed or ignored. The **value context** influences how the right-hand side of assignments should be interpreted and type-converted.

Consider this fundamental INI structure that demonstrates the key parsing contexts:

```
# Global configuration
debug = true
port = 8080

[database]
host = localhost
port = 5432  # Different from global port
name = "my app"

[database.connection]
timeout = 30
```

The parser must recognize that we have three distinct namespaces: global scope, `database` section, and `database.connection` section. Keys can be repeated across namespaces (like `port`) without conflict. The final data structure should reflect this hierarchical organization through nested dictionaries.

The **line-oriented processing model** is crucial for INI parsing. Unlike formats that might span arbitrary boundaries, INI parsing can be conceptualized as processing one logical line at a time. Each line falls into one of several categories: blank lines (ignored), comment lines (ignored or preserved), section headers (context switches), key-value assignments (data storage), or continuation lines (value extension).

This line-oriented approach creates a natural state machine where the parser maintains current context and applies different processing rules based on line classification. The beauty of this model is its simplicity—most INI parsing bugs stem from incorrectly classifying lines or failing to maintain proper context across line boundaries.

### INI Parsing Algorithm: Step-by-Step Process for Handling Sections, Keys, Values, and Comments

The INI parsing algorithm follows a systematic approach that mirrors how humans naturally read these files: top to bottom, line by line, building understanding of structure as we encounter section boundaries. The algorithm maintains parsing state across line boundaries while making context-sensitive decisions about how to interpret each line.

The core algorithm can be broken down into distinct phases that handle different aspects of the parsing process:

1. **Preprocessing and Line Classification**: The raw input is split into logical lines (handling line continuation if supported), and each line is classified by its syntactic structure. This phase identifies section headers, key-value pairs, comments, and blank lines through pattern recognition.

2. **Context Management**: Based on line classification, the parser updates its current parsing context. Section headers cause context switches, while key-value pairs are processed within the current context. This phase maintains the section stack and current namespace information.

3. **Value Processing**: Key-value pairs undergo value extraction, type inference, and storage within the appropriate section. This phase handles quoted strings, escape sequences, and basic type coercion.

4. **Structure Building**: The parsed data is organized into the final nested dictionary structure, creating intermediate sections as needed and handling key path resolution for dotted notation.

Here's the detailed step-by-step algorithm for INI parsing:

1. **Initialize parsing state** by creating an empty result dictionary, setting current section to global scope (empty string), and preparing line processing infrastructure including line number tracking and error context accumulation.

2. **Split input into logical lines** while preserving line number information for error reporting. Handle different line ending conventions (CRLF, LF) and optionally support line continuation with backslash escapes if that feature is desired.

3. **For each logical line, classify its syntactic type** by examining leading characters after whitespace trimming. Lines starting with `[` are section headers, lines containing `=` or `:` are key-value pairs, lines starting with `;` or `#` are comments, and empty lines are ignored.

4. **Process section headers** by extracting the section name from between square brackets, validating the syntax, and updating the current section context. Create nested dictionary structure if the section name contains dots (like `database.connection`).

5. **Process key-value pairs** by splitting on the first occurrence of `=` or `:`, trimming whitespace from both sides, and applying value processing rules. Store the processed key-value pair in the current section of the result dictionary.

6. **Handle value processing** by detecting quoted strings and processing escape sequences, applying type inference to convert strings to appropriate Python types (integers, floats, booleans), and handling special cases like empty values or values containing the assignment operator.

7. **Manage inline comments** by detecting comment markers after values and either preserving them as metadata or ignoring them based on parser configuration. Be careful not to treat comment markers inside quoted strings as actual comments.

8. **Handle error recovery** by collecting syntax errors with line number information while continuing to parse subsequent lines when possible. This allows reporting multiple errors in a single parse run rather than failing on the first error.

9. **Finalize the result structure** by ensuring all sections exist in the final dictionary (even empty sections), applying any post-processing transformations like key normalization, and validating the overall structure for consistency.

10. **Return the parsed configuration** as a nested dictionary where top-level keys represent sections (with an empty string key for global values) and values are dictionaries containing the key-value pairs for each section.

The algorithm maintains several pieces of state throughout this process: the current section context (determining where new keys are stored), accumulated errors for comprehensive reporting, line number information for error context, and the growing result dictionary that represents the parsed structure.

### INI Architecture Decisions: Decisions Around Global Keys, Comment Handling, and Value Type Inference

The INI parser requires several architectural decisions that significantly impact both implementation complexity and user experience. These decisions represent trade-offs between simplicity, compatibility, and functionality. Each choice has downstream consequences for how the parser behaves in edge cases and how well it integrates with the broader configuration parsing system.

> **Decision: Global Key Handling Strategy**
> - **Context**: INI files often contain key-value pairs before any section headers, but different parsers handle these differently—some reject them, others create an implicit section, others treat them as truly global.
> - **Options Considered**: Reject global keys as syntax errors, create implicit "DEFAULT" section, store in special global namespace
> - **Decision**: Store global keys in the result dictionary with empty string as section key, accessible as `result[""]`
> - **Rationale**: This approach maintains the section-based mental model while accommodating real-world INI files that use global configuration. It's explicit (no hidden "DEFAULT" section) and predictable (always accessible the same way).
> - **Consequences**: Users must check for the empty string key to access global values, but the behavior is consistent and discoverable.

| Global Key Option | Pros | Cons | Compatibility |
|------------------|------|------|---------------|
| Reject as error | Simple, enforces structure | Breaks with common INI files | Low |
| Implicit DEFAULT section | Familiar to ConfigParser users | Hidden behavior, not obvious | Medium |
| Empty string key | Explicit, predictable | Slightly awkward access pattern | High |

> **Decision: Comment Preservation Strategy**
> - **Context**: Comments serve different purposes—some are documentation that should be preserved, others are temporary annotations. Different use cases benefit from different handling approaches.
> - **Options Considered**: Always ignore comments, always preserve as metadata, configurable preservation with default ignore
> - **Decision**: Default to ignoring comments with optional preservation mode that stores them as metadata in a parallel structure
> - **Rationale**: Most configuration consumers don't need comment data and benefit from cleaner output, but preserving comments enables round-trip editing and documentation tools.
> - **Consequences**: Simple use cases get clean data, advanced use cases can opt into comment preservation with additional complexity.

| Comment Strategy | Memory Usage | Output Complexity | Use Case Support |
|-----------------|--------------|-------------------|------------------|
| Always ignore | Low | Simple | Basic config loading |
| Always preserve | High | Complex | Documentation tools |
| Configurable | Variable | Medium | Both use cases |

> **Decision: Value Type Inference Rules**
> - **Context**: INI files store everything as strings, but users expect automatic conversion to appropriate Python types. Different inference rules affect both usability and predictability.
> - **Options Considered**: No type inference (all strings), aggressive inference with boolean/number detection, conservative inference with opt-in parsing
> - **Decision**: Conservative type inference that converts obvious numbers and booleans while preserving strings for ambiguous values
> - **Rationale**: Automatic conversion improves usability for common cases while avoiding surprising conversions that might break application logic.
> - **Consequences**: Users get convenient type conversion for clear cases but maintain control over ambiguous values through explicit quoting.

| Inference Level | Conversion Examples | Surprising Cases | User Control |
|----------------|-------------------|------------------|--------------|
| None | All strings | None | Full |
| Aggressive | "yes"→true, "1.0"→float | Many edge cases | Limited |
| Conservative | "123"→int, "true"→bool | Minimal | Good balance |

> **Decision: Section Nesting Strategy**
> - **Context**: Some INI files use dotted section names to imply hierarchy, while others treat dots as literal characters. The parser must choose how to interpret these patterns.
> - **Options Considered**: Always flatten section names, always create nested structure from dots, configurable nesting with sensible default
> - **Decision**: Create nested dictionary structure from dotted section names by default, with option to disable nesting for literal interpretation
> - **Rationale**: Nested structure matches user expectations for modern configuration and provides better integration with other formats, while the option preserves compatibility with legacy systems.
> - **Consequences**: Default behavior creates intuitive nested access patterns, but users working with legacy systems can opt out of nesting interpretation.

The type inference implementation uses a priority-based approach that attempts conversions in order of specificity. Integer conversion is attempted first (using Python's `int()` function), followed by float conversion, then boolean conversion using a predefined mapping of string values to boolean states. Only if all conversions fail does the value remain as a string.

| Value Pattern | Inference Result | Rationale |
|---------------|------------------|-----------|
| `123` | `int(123)` | Clear integer literal |
| `12.34` | `float(12.34)` | Clear float literal |
| `true`, `yes`, `on` | `True` | Common boolean representations |
| `false`, `no`, `off` | `False` | Common boolean representations |
| `"123"` | `str("123")` | Quoted values bypass inference |
| `1.0.0` | `str("1.0.0")` | Version strings stay as strings |

### Common INI Parsing Pitfalls: Issues with Inline Comments, Quoted Values, and Section Nesting

INI parsing appears deceptively simple, leading many developers to underestimate the edge cases and subtle behaviors that can cause parsing failures or incorrect data extraction. Understanding these common pitfalls helps build robust parsers and debug issues when they arise.

⚠️ **Pitfall: Inline Comment Detection Breaking on Quoted Values**

The most frequent mistake in INI parsing occurs when handling inline comments that appear after values. Naive implementations split lines on comment characters (`;` or `#`) without considering whether those characters appear inside quoted strings.

Consider this problematic line: `url = "postgres://user:pass#123@host/db" ; connection string`. A naive parser might treat `#123@host/db" ; connection string` as a comment, corrupting the URL value. The correct behavior requires tracking quote state while scanning for comment markers.

This issue manifests in several ways: quoted passwords containing special characters get truncated, file paths with hash symbols get corrupted, and SQL connection strings become malformed. The fix requires implementing stateful scanning that only recognizes comment markers outside of quoted contexts.

The robust approach processes each character sequentially, maintaining quote state and only treating comment markers as significant when not inside string literals. This requires handling escape sequences within quotes to properly track quote boundaries.

⚠️ **Pitfall: Assignment Operator Confusion in Values**

Another common error involves handling assignment operators (`=` or `:`) that appear within values rather than as key-value separators. Splitting lines on the first occurrence of these characters seems correct, but many implementations split on all occurrences, breaking values that contain assignment operators.

For example: `expression = x = y + z` should result in key `"expression"` with value `"x = y + z"`, not key `"expression"` with value `"x"` and some confused parsing of the remainder. The correct implementation splits only on the first occurrence of the assignment operator.

This pitfall extends to handling multiple assignment operators in configuration values. Mathematical expressions, code snippets, and connection strings frequently contain these characters as content rather than syntax.

⚠️ **Pitfall: Section Name Validation and Escaping**

Section headers require careful validation to prevent malformed input from corrupting the parser state or creating invalid nested structures. Common mistakes include accepting section names with unmatched brackets, allowing empty section names, or failing to handle escaped characters in section names.

Consider these problematic section headers: `[section[nested]]`, `[]`, `[section\nname]`. Each represents a different failure mode—nested brackets might indicate user confusion about syntax, empty names create ambiguous dictionary keys, and embedded newlines suggest multiline parsing errors.

The robust approach validates section names against a clear syntax specification, rejects malformed headers with helpful error messages, and handles any necessary escape sequences consistently with the overall parsing approach.

⚠️ **Pitfall: Global Key Context Management**

Managing the parsing context when transitioning between global keys and sectioned keys creates several failure modes. The most common involves incorrectly attributing keys to sections when encountering mixed global and sectioned content.

Consider this structure:
```ini
global_key = value
[section]
section_key = value
another_global = value
```

Incorrect implementations might attribute `another_global` to `[section]` because they fail to implement proper context switching. Keys appearing after sections should remain in their current section unless explicitly moved by another section header.

The fix requires maintaining clear section context throughout parsing and only changing context when section headers are encountered. Global keys appearing after sections indicate either user error (should be warned about) or intentional global configuration (should be supported).

⚠️ **Pitfall: Value Trimming and Whitespace Semantics**

Different approaches to whitespace handling create inconsistent behavior across different INI files. Some implementations trim all whitespace from values, others preserve it exactly, and still others apply complex rules about when trimming should occur.

The challenge emerges with values like `name = " John Doe "` where the spaces might be significant (indicating exact string content) or artifacts of formatting (should be trimmed). Quoted values suggest the spaces are intentional, but unquoted values are ambiguous.

A consistent approach defines clear rules: unquoted values have leading and trailing whitespace trimmed, quoted values preserve all internal whitespace including leading/trailing spaces, and the quoting characters themselves are removed during processing.

⚠️ **Pitfall: Line Continuation and Multiline Value Handling**

Some INI dialects support line continuation with backslash characters or multiline values with specific syntax. Implementing this incorrectly breaks both simple and complex configurations.

The most common error involves treating continuation characters inside quoted strings as actual continuation markers rather than literal content. Another frequent mistake is failing to properly join continued lines while preserving meaningful whitespace.

For implementations that choose to support continuation, the logic must carefully distinguish between syntactic continuation (backslash at line end) and literal backslashes that happen to appear at line boundaries.

### Implementation Guidance

The INI parser implementation focuses on building robust line-by-line processing while maintaining clean separation between tokenization, parsing, and data structure construction. This foundation prepares developers for more complex parsing challenges in TOML and YAML formats.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Line Processing | `str.splitlines()` with manual iteration | `io.StringIO` with buffered reading |
| Pattern Recognition | String methods (`startswith`, `find`, `strip`) | `re` module with compiled patterns |
| Value Type Conversion | Manual `int()`, `float()`, `bool()` attempts | `ast.literal_eval()` for safe evaluation |
| Error Collection | Simple list of error strings | Structured `ParseError` objects with position |
| String Parsing | Character-by-character state machine | `shlex` module for shell-like parsing |

#### Recommended File Structure

```
config_parser/
├── __init__.py
├── common/
│   ├── __init__.py
│   ├── types.py              # Token, Position, ParseError definitions
│   ├── tokenizer.py          # BaseTokenizer (from previous section)
│   └── utils.py              # Shared utilities
├── ini/
│   ├── __init__.py
│   ├── parser.py             # INIParser class - main implementation
│   ├── validator.py          # INI-specific validation rules
│   └── test_ini.py           # INI parser tests
├── toml/                     # Future TOML implementation
├── yaml/                     # Future YAML implementation
└── main.py                   # CLI interface and format detection
```

#### Infrastructure Starter Code

**File: `common/types.py`** (Complete implementation for shared types)

```python
"""
Shared type definitions for configuration parsers.
These types provide the foundation for all format-specific implementations.
"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List, Optional, Dict, Union


class TokenType(Enum):
    """Token types used across all configuration formats."""
    STRING = auto()
    NUMBER = auto() 
    BOOLEAN = auto()
    IDENTIFIER = auto()
    EQUALS = auto()
    COLON = auto()
    NEWLINE = auto()
    EOF = auto()
    COMMENT = auto()
    SECTION_START = auto()      # [
    SECTION_END = auto()        # ]
    ARRAY_START = auto()        # [
    ARRAY_END = auto()          # ]
    OBJECT_START = auto()       # {
    OBJECT_END = auto()         # }
    COMMA = auto()
    DOT = auto()
    INDENT = auto()
    DEDENT = auto()
    BLOCK_SEQUENCE = auto()     # YAML list marker
    INVALID = auto()


@dataclass
class Position:
    """Position information for tokens and errors."""
    line: int
    column: int
    offset: int

    def __str__(self) -> str:
        return f"line {self.line}, column {self.column}"


@dataclass  
class Token:
    """A lexical token with type, value, and position information."""
    type: TokenType
    value: Any
    position: Position
    raw_text: str

    def __str__(self) -> str:
        return f"{self.type.name}({self.value!r}) at {self.position}"


class ParseError(Exception):
    """Base class for all parsing errors."""
    def __init__(self, message: str, position: Optional[Position] = None, 
                 suggestion: Optional[str] = None):
        self.message = message
        self.position = position
        self.suggestion = suggestion
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        msg = self.message
        if self.position:
            msg = f"{msg} at {self.position}"
        if self.suggestion:
            msg = f"{msg}\nSuggestion: {self.suggestion}"
        return msg


class TokenError(ParseError):
    """Error during tokenization phase."""
    pass


class SyntaxError(ParseError):
    """Error in syntax structure."""
    pass


class StructureError(ParseError):
    """Error in logical structure (e.g., duplicate keys)."""
    pass


# Constants used across parsers
EOF_MARKER = '\0'
WHITESPACE_CHARS = ' \t\r'
NEWLINE_CHARS = '\n'
QUOTE_CHARS = '"\'`'
```

**File: `common/utils.py`** (Complete utility functions)

```python
"""
Utility functions shared across all configuration parsers.
"""
from typing import Dict, Any, List
from .types import Position, ParseError


def current_position(source: str, offset: int) -> Position:
    """Calculate line and column from string offset."""
    lines_before = source[:offset].count('\n')
    if lines_before == 0:
        column = offset
    else:
        last_newline = source.rfind('\n', 0, offset)
        column = offset - last_newline - 1
    
    return Position(
        line=lines_before + 1,
        column=column + 1, 
        offset=offset
    )


def create_error_context(source: str, position: Position, context_lines: int = 2) -> str:
    """Generate visual error context showing problematic line with pointer."""
    lines = source.splitlines()
    if not lines or position.line > len(lines):
        return "Error context unavailable"
    
    start_line = max(0, position.line - context_lines - 1)
    end_line = min(len(lines), position.line + context_lines)
    
    context_parts = []
    for i in range(start_line, end_line):
        line_num = i + 1
        prefix = ">>> " if line_num == position.line else "    "
        context_parts.append(f"{prefix}{line_num:4d}: {lines[i]}")
        
        # Add pointer line for error location
        if line_num == position.line:
            pointer_line = " " * (len(prefix) + 6 + position.column - 1) + "^"
            context_parts.append(pointer_line)
    
    return "\n".join(context_parts)


def normalize_key(key: str, strategy: str = "lowercase") -> str:
    """Apply key normalization strategy."""
    if strategy == "lowercase":
        return key.lower().strip()
    elif strategy == "preserve":
        return key.strip()
    elif strategy == "snake_case":
        return key.lower().replace("-", "_").replace(" ", "_").strip()
    else:
        raise ValueError(f"Unknown normalization strategy: {strategy}")


def merge_nested_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge nested dictionaries, with dict2 taking precedence."""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_nested_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def detect_format(content: str) -> str:
    """Automatically detect configuration format from content."""
    content = content.strip()
    if not content:
        return "ini"  # Default for empty files
    
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return "ini"
    
    # YAML indicators
    if any(line.startswith('- ') or ': ' in line and not line.startswith('[') 
           for line in lines):
        # Check for YAML-specific patterns
        if any(line and not line[0].isspace() and ': ' in line for line in lines):
            return "yaml"
    
    # TOML indicators  
    if any(line.startswith('[[') or ' = ' in line and '"' in line 
           for line in lines):
        return "toml"
    
    # Default to INI
    return "ini"
```

#### Core Logic Skeleton Code

**File: `ini/parser.py`** (Skeleton for learner implementation)

```python
"""
INI format configuration parser.
Implements line-based parsing for section headers and key-value pairs.
"""
from typing import Dict, Any, List, Optional, Tuple
from ..common.types import Position, ParseError, SyntaxError, StructureError
from ..common.utils import current_position, normalize_key, create_error_context


class INIParser:
    """
    Parser for INI format configuration files.
    
    Supports:
    - Section headers: [section.name]  
    - Key-value pairs: key = value, key: value
    - Comments: ; comment, # comment
    - Quoted values: key = "value with spaces"
    - Type inference: automatic conversion to int, float, bool
    """
    
    def __init__(self, preserve_comments: bool = False, 
                 enable_nesting: bool = True,
                 key_normalization: str = "preserve"):
        self.preserve_comments = preserve_comments
        self.enable_nesting = enable_nesting  
        self.key_normalization = key_normalization
        self.errors: List[ParseError] = []
        
    def parse(self, content: str) -> Dict[str, Any]:
        """
        Parse INI format configuration content.
        
        Returns nested dictionary with sections as keys.
        Global keys are stored under empty string key.
        """
        # TODO 1: Initialize parsing state (result dict, current section, line tracking)
        # TODO 2: Split content into logical lines while preserving line numbers  
        # TODO 3: For each line, classify its type (section, key-value, comment, blank)
        # TODO 4: Process each line type with appropriate handler method
        # TODO 5: Handle any accumulated errors and return result
        # Hint: Use self._classify_line() to determine line type
        # Hint: Track current section context throughout parsing
        pass
    
    def _classify_line(self, line: str, line_num: int) -> Tuple[str, str]:
        """
        Classify a line as section, keyvalue, comment, or blank.
        
        Returns (line_type, processed_content) tuple.
        line_type is one of: 'section', 'keyvalue', 'comment', 'blank'
        """
        # TODO 1: Strip whitespace and check for blank lines
        # TODO 2: Check for comment lines (starting with ; or #)
        # TODO 3: Check for section headers (enclosed in [])
        # TODO 4: Check for key-value pairs (containing = or :)
        # TODO 5: Return appropriate classification
        # Hint: Look for patterns after stripping whitespace
        # Hint: Be careful about quoted strings containing special characters
        pass
        
    def _process_section_header(self, line: str, line_num: int) -> str:
        """
        Extract section name from header line like [section.name].
        
        Returns the section name, creating nested structure if dots present.
        """
        # TODO 1: Extract content between [ and ] brackets
        # TODO 2: Validate section name (not empty, valid characters)
        # TODO 3: Handle dotted names for nested sections if nesting enabled
        # TODO 4: Return normalized section name
        # Hint: Check for unmatched brackets and report errors
        # Hint: Consider whether to allow spaces in section names
        pass
        
    def _process_key_value_pair(self, line: str, line_num: int, current_section: str, 
                               result: Dict[str, Any]) -> None:
        """
        Parse key-value pair and store in result dictionary.
        
        Handles both = and : delimiters, quoted values, inline comments.
        """
        # TODO 1: Find the assignment operator (= or :) - use first occurrence only
        # TODO 2: Split line into key and value parts
        # TODO 3: Process value part (handle quotes, inline comments, type inference)
        # TODO 4: Store in appropriate section of result dictionary
        # TODO 5: Handle any parsing errors gracefully
        # Hint: Use self._parse_value() to handle complex value processing
        # Hint: Be careful about assignment operators inside quoted strings
        pass
        
    def _parse_value(self, value_str: str, line_num: int) -> Tuple[Any, Optional[str]]:
        """
        Parse value string, handling quotes, escapes, and type inference.
        
        Returns (processed_value, inline_comment) tuple.
        """
        # TODO 1: Detect and handle quoted strings (preserve exact content)
        # TODO 2: Find inline comments (but not inside quoted strings)  
        # TODO 3: Apply type inference to unquoted values
        # TODO 4: Return processed value and any inline comment
        # Hint: Need to track quote state when looking for comment markers
        # Hint: Handle escape sequences in quoted strings
        pass
        
    def _infer_type(self, value: str) -> Any:
        """
        Convert string value to appropriate Python type.
        
        Attempts int, float, bool conversion in order.
        """
        # TODO 1: Try integer conversion first (for values like "123")
        # TODO 2: Try float conversion (for values like "12.34")  
        # TODO 3: Try boolean conversion (true, false, yes, no, on, off)
        # TODO 4: Fall back to string if all conversions fail
        # Hint: Use try/except for conversion attempts
        # Hint: Define boolean value mapping constants
        pass
        
    def _create_nested_section(self, result: Dict[str, Any], section_path: str) -> Dict[str, Any]:
        """
        Create nested dictionary structure for dotted section names.
        
        For section "database.connection", creates result["database"]["connection"].
        Returns the innermost dictionary where keys should be stored.
        """
        # TODO 1: Split section path on dots if nesting enabled
        # TODO 2: Walk through path parts, creating nested dicts as needed
        # TODO 3: Return reference to innermost dictionary
        # TODO 4: Handle case where intermediate path conflicts with existing keys
        # Hint: Check if intermediate keys are already non-dict values
        # Hint: Consider whether empty section names are valid
        pass

# Additional helper functions for string processing
def _find_comment_start(value: str, start_pos: int = 0) -> Optional[int]:
    """Find start of inline comment, respecting quoted string boundaries."""
    # TODO: Implement stateful scanning for comment markers outside quotes
    pass

def _process_quoted_string(quoted_str: str, quote_char: str) -> str:
    """Process escape sequences in quoted string."""
    # TODO: Handle \n, \t, \\, \", \' escape sequences
    pass
```

#### Language-Specific Hints

**Python Implementation Notes:**
- Use `str.partition('=')` instead of `str.split('=', 1)` for cleaner key-value splitting
- The `configparser` module in standard library provides reference behavior, but implement from scratch for learning
- `str.strip()` removes all whitespace; `str.lstrip(' \t')` removes only spaces and tabs
- For boolean inference: `{'true': True, 'false': False, 'yes': True, 'no': False, 'on': True, 'off': False}`
- Use `collections.defaultdict(dict)` to automatically create nested sections
- Regular expressions are overkill for INI parsing—string methods are sufficient and more readable

**Error Handling Patterns:**
```python
try:
    result[section][key] = self._infer_type(value)
except (ValueError, TypeError) as e:
    error = SyntaxError(f"Invalid value for {key}: {value}", 
                       position=current_position(content, line_start_offset),
                       suggestion="Check value format or use quotes for literal strings")
    self.errors.append(error)
    result[section][key] = value  # Fall back to string value
```

#### Milestone Checkpoint

After implementing the INI parser, verify correct behavior with these test cases:

**Test Command:** `python -m pytest ini/test_ini.py -v`

**Manual Verification:**
```python
from ini.parser import INIParser

parser = INIParser()
config = parser.parse("""
# Global config
debug = true  
port = 8080

[database]
host = localhost
port = 5432
password = "secret#123"  ; inline comment

[database.pool]
min_connections = 5
max_connections = 20
""")

# Expected structure:
assert config[""] == {"debug": True, "port": 8080}
assert config["database"]["host"] == "localhost" 
assert config["database"]["port"] == 5432
assert config["database"]["password"] == "secret#123"
assert config["database"]["pool"]["min_connections"] == 5
```

**Success Indicators:**
- Global keys accessible via `config[""]`
- Section nesting creates proper dictionary hierarchy  
- Type inference converts `"true"` to `True`, `"123"` to `123`
- Quoted strings preserve exact content including special characters
- Inline comments after values are ignored (unless preservation enabled)
- Parser continues after errors and collects multiple issues

**Common Issues to Debug:**
- If all values are strings: Check `_infer_type()` implementation
- If comments appear in values: Check quote state tracking in `_parse_value()`
- If section nesting fails: Verify `_create_nested_section()` path splitting
- If global keys missing: Ensure empty string key exists in result dictionary


## TOML Parser Component Design

> **Milestone(s):** TOML Tokenizer, TOML Parser - builds advanced tokenization capabilities and implements recursive descent parsing for tables, arrays, and complex type system

The TOML parser represents the most sophisticated component in our configuration parsing system, handling the complex interplay between explicit type systems, nested table structures, and array-of-tables syntax. Unlike the line-based simplicity of INI parsing, TOML parsing requires a full recursive descent approach that can manage hierarchical document structures while maintaining strict rules about key redefinition and table organization. This component demonstrates advanced parsing techniques including lookahead parsing, symbol table management, and context-sensitive grammar handling that form the foundation for understanding modern configuration language implementation.

The complexity of TOML parsing stems from its ambitious goal of being both human-readable and unambiguous for machine processing. While this creates a more robust configuration format, it introduces significant parsing challenges around table inheritance, dotted key expansion, and the distinction between inline tables and table headers. Understanding these concepts provides essential insight into how modern configuration languages balance expressiveness with parsing complexity.

### TOML Parsing Mental Model

Think of TOML parsing as **architectural blueprint interpretation** — you're reading a structured document that defines a building (your data structure) through a combination of room declarations (tables), furniture lists (arrays), and detailed specifications (key-value pairs). Just as an architect must understand that declaring a "kitchen.island" doesn't just create an island but also ensures the kitchen room exists, TOML parsing must understand that dotted keys implicitly create the table hierarchy they reference.

The key insight is that TOML operates on two levels simultaneously: the **document level** where you're declaring tables and organizing structure, and the **value level** where you're assigning specific data to keys. Unlike INI files where sections are simple containers, TOML tables have complex relationships — they can be nested, referenced by dotted paths, and must follow strict rules about redefinition and ordering.

Consider this mental model: TOML parsing is like **managing a hierarchical filing system** where each table declaration creates a folder, each key-value pair files a document, and dotted notation creates nested folder structures. The parser must ensure that no document is filed twice in the same location (key redefinition error) and that folder creation follows proper hierarchy rules. Array-of-tables syntax is like creating multiple folders with the same name but different numeric suffixes — `reports-1/`, `reports-2/`, etc.

The **explicit type system** in TOML means the parser acts as both a filing clerk and a data validator. Unlike YAML's implicit typing where "yes" might become a boolean, TOML requires that strings look like strings (`"hello"`), integers look like integers (`42`), and dates follow specific formats (`1979-05-27T07:32:00Z`). This eliminates parsing ambiguity but requires sophisticated tokenization to distinguish between different literal formats.

> The critical parsing insight is that TOML tables exist in a **global namespace** where dotted keys and table headers must be coordinated. When you write `physical.color = "orange"` followed by `[physical.size]`, the parser must recognize that both reference the same `physical` table and merge appropriately.

### Recursive Descent Algorithm

Recursive descent parsing provides the natural approach for handling TOML's nested structure by mapping grammatical rules directly to function calls. Each syntactic construct (table, key-value pair, array, inline table) corresponds to a parsing function that can call other parsing functions to handle nested elements. This creates a call stack that mirrors the document's hierarchical structure.

The **core recursive descent pattern** follows this structure: each parsing function consumes tokens that match its grammatical rule, recursively calls other parsing functions for nested elements, and returns a parse tree node representing the matched structure. For TOML, the top-level parser alternates between table declarations and key-value assignments, calling specialized functions for each construct type.

Here's the algorithmic breakdown for TOML recursive descent parsing:

1. **Initialize parsing state** with an empty symbol table for tracking defined keys and tables, current table context pointing to the root, and token stream positioned at the beginning
2. **Main parsing loop** examines the current token to determine the construct type — section header brackets indicate table declarations, identifiers suggest key-value pairs, and EOF terminates parsing
3. **Table declaration processing** extracts the table path from bracketed notation, validates that the path doesn't conflict with existing definitions, creates the table hierarchy if needed, and updates the current table context
4. **Key-value pair processing** parses the key (which may be dotted), validates the key isn't already defined in the current table, parses the value recursively based on its type, and stores the key-value mapping
5. **Dotted key expansion** treats dotted keys like nested table creation, ensuring each path segment creates a table if it doesn't exist, and placing the final key-value pair in the deepest table
6. **Value parsing delegation** examines the token type and calls appropriate specialized parsers — `parse_string()` for quoted literals, `parse_array()` for bracket notation, `parse_inline_table()` for brace notation
7. **Array-of-tables handling** creates a new table instance and appends it to an array stored under the table name, allowing multiple table definitions with the same path
8. **Error recovery** captures parsing failures with position information, attempts to resynchronize at the next table boundary or key-value pair, and continues parsing to collect multiple error reports

The **recursive call pattern** emerges naturally from TOML's grammar. When `parse_value()` encounters an array, it calls `parse_array()`, which repeatedly calls `parse_value()` for each element. When parsing inline tables, `parse_inline_table()` calls `parse_key_value_pair()` for each entry. This recursive structure handles arbitrary nesting depth without explicit stack management.

**Lookahead parsing** becomes essential for disambiguating TOML constructs. Distinguishing between table headers `[table]` and array-of-tables `[[table]]` requires examining two tokens ahead. Similarly, dotted keys require lookahead to determine whether `a.b.c` represents a single dotted key or the start of a more complex structure.

> The recursive descent approach naturally handles TOML's context sensitivity — each parsing function maintains the current table context and key path, allowing nested calls to understand their position in the document hierarchy.

| Parsing Function | Purpose | Recursive Calls | Returns |
|-----------------|---------|----------------|---------|
| `parse_document()` | Main entry point, processes top-level constructs | `parse_table_header()`, `parse_key_value_pair()` | `ParseNode` with document structure |
| `parse_table_header()` | Handles `[table]` and `[[table]]` declarations | `parse_key_path()` | `SectionNode` with table information |
| `parse_key_value_pair()` | Processes key assignments | `parse_key_path()`, `parse_value()` | `KeyValueNode` with key and value |
| `parse_value()` | Dispatches to type-specific parsers | `parse_array()`, `parse_inline_table()`, `parse_string()` | `ValueNode` with typed value |
| `parse_array()` | Handles `[item1, item2, item3]` syntax | `parse_value()` for each element | `ValueNode` with array contents |
| `parse_inline_table()` | Handles `{key = value, key2 = value2}` syntax | `parse_key_value_pair()` for each entry | `ValueNode` with table contents |
| `parse_key_path()` | Handles dotted keys like `a.b.c` | None (terminal parser) | List of key segments |

### Table and Array-of-Tables Logic

TOML's table system represents one of the most complex aspects of configuration file parsing, requiring careful management of hierarchical namespaces, implicit table creation, and the distinction between table headers and array-of-tables declarations. The parser must maintain a global view of the document structure while processing local key-value assignments.

**Table hierarchy management** operates through a symbol table that tracks all defined tables and keys. When the parser encounters a table header like `[database.connection]`, it must verify that neither `database` nor `database.connection` has been previously defined as a non-table value, create any missing intermediate tables, and establish the new current context for subsequent key-value pairs. This process involves both validation and construction phases.

The **implicit table creation** rule states that dotted keys automatically create the necessary table hierarchy. When processing `server.host = "localhost"` before any `[server]` declaration, the parser must create an implicit `server` table and place the `host` key within it. This implicit creation must be tracked separately from explicit table declarations to handle later conflicts correctly.

**Array-of-tables syntax** using double brackets `[[database.servers]]` creates a fundamentally different structure than regular tables. Each array-of-tables declaration appends a new table instance to an array stored under that key path. The parser must distinguish between extending an existing array-of-tables and conflicting with a previously defined single table or value.

Here's the detailed algorithm for table and array-of-tables processing:

1. **Table header detection** recognizes bracket notation and determines whether single brackets indicate a table declaration or double brackets indicate array-of-tables entry
2. **Path validation** checks that the table path doesn't conflict with existing definitions — a path can't be redefined as a different type, and array-of-tables paths must be consistent
3. **Intermediate table creation** ensures all parent tables in a dotted path exist, creating them implicitly if necessary, and marking them as implicit to allow later explicit redefinition
4. **Array-of-tables instantiation** creates a new table instance for double-bracket notation, appends it to the array stored under that path, and sets the new table as the current parsing context
5. **Context switching** updates the current table pointer to reflect the newly declared or accessed table, allowing subsequent key-value pairs to be stored in the correct location
6. **Conflict detection** validates that new table declarations don't redefine existing keys or tables, that array-of-tables declarations are consistent with previous usage, and that implicit tables can be explicitly redefined

**Dotted key expansion** requires special handling because it creates table structure inline with value assignment. When processing `physical.color = "orange"`, the parser must create a `physical` table if it doesn't exist, then assign the `color` key within that table. This expansion must respect the same conflict rules as explicit table declarations.

The **table redefinition rules** create complex validation requirements. A table can only be explicitly defined once, but keys can be added to tables from multiple locations. Array-of-tables can be extended with multiple `[[array.name]]` declarations, but the same path can't mix array-of-tables and regular table syntax.

> The key insight is that TOML tables form a **global namespace** where all dotted paths must be mutually consistent. Unlike INI sections that are independent containers, TOML tables are interconnected through their hierarchical relationships and shared namespace.

| Table Operation | Validation Required | Action Taken | Error Conditions |
|----------------|-------------------|--------------|------------------|
| `[explicit.table]` | Path not previously defined as table | Create table, set as current context | Path exists as value or different table type |
| `[[array.of.tables]]` | Path used consistently for arrays | Create table, append to array | Path exists as single table or value |
| Dotted key assignment | Intermediate paths are compatible | Create implicit tables, assign value | Intermediate path conflicts with existing value |
| Inline table `{k=v}` | Table contents don't conflict | Create table with all key-value pairs | Keys conflict with existing definitions in scope |

### TOML Architecture Decisions

The TOML parser's architecture must balance parsing complexity with maintainability, requiring careful decisions about tokenization depth, error handling strategies, and data structure representation. These architectural choices significantly impact both implementation difficulty and runtime performance.

> **Decision: Two-Pass vs Single-Pass Parsing**
> - **Context**: TOML's global namespace and table redefinition rules require comprehensive validation that may benefit from multiple parsing phases
> - **Options Considered**: Single-pass parsing with complex state tracking, two-pass parsing with structure building then validation, three-pass parsing with tokenization, structure building, and validation phases
> - **Decision**: Single-pass parsing with comprehensive symbol table management
> - **Rationale**: Single-pass parsing provides better memory efficiency and simpler error reporting, while comprehensive symbol tables can handle the required validation without multiple document traversals. The complexity of multi-pass coordination outweighs the benefits for our educational implementation.
> - **Consequences**: Requires sophisticated symbol table implementation but provides linear time complexity and straightforward error position reporting

| Parsing Approach | Memory Usage | Time Complexity | Error Reporting Quality | Implementation Complexity |
|------------------|--------------|-----------------|------------------------|---------------------------|
| Single-pass with symbol table | O(n) | O(n) | Excellent (exact positions) | High (complex state management) |
| Two-pass parsing | O(2n) | O(2n) | Good (requires position tracking) | Medium (separate phases) |
| Three-pass parsing | O(3n) | O(3n) | Excellent (multiple validation passes) | Low (simple individual passes) |

> **Decision: Recursive Descent vs Parser Generator**
> - **Context**: TOML's grammar complexity could be handled by hand-written recursive descent or automated parser generation tools
> - **Options Considered**: Hand-written recursive descent parser, ANTLR or similar parser generator, hybrid approach with generated lexer and hand-written parser
> - **Decision**: Hand-written recursive descent parser
> - **Rationale**: Educational value of understanding parsing mechanics outweighs the convenience of generated parsers. Recursive descent provides clear mapping from grammar rules to code, making debugging and extension more accessible to learners.
> - **Consequences**: Higher implementation effort but better learning outcomes and complete control over error messages and recovery strategies

> **Decision: Token Stream vs Character Stream Parsing**
> - **Context**: The parser can operate on pre-tokenized input from the tokenizer component or consume characters directly during parsing
> - **Options Considered**: Pre-tokenized input with full lookahead, character stream with on-demand tokenization, hybrid approach with token buffering
> - **Decision**: Pre-tokenized input with bounded lookahead
> - **Rationale**: Separation of lexical analysis and syntactic analysis improves modularity and debugging. Pre-tokenization enables better error messages with token-level position tracking and simplifies parser logic by eliminating character-level concerns.
> - **Consequences**: Higher memory usage for token storage but cleaner parser implementation and superior error reporting capabilities

**Symbol table architecture** requires careful consideration of how to represent the hierarchical namespace and track different types of definitions (explicit tables, implicit tables, array-of-tables, values). The symbol table must support efficient lookup, conflict detection, and context switching as the parser moves between different table scopes.

The chosen approach uses a **nested dictionary structure** mirroring the final output format, with additional metadata tracking the definition type and location for each entry. This provides O(1) lookup for conflict detection while building the final data structure incrementally during parsing.

> **Decision: Parse Tree vs Direct Output Generation**
> - **Context**: The parser can build an intermediate parse tree structure or generate the final nested dictionary directly during parsing
> - **Options Considered**: Full parse tree with separate transformation phase, direct output generation during parsing, hybrid approach with minimal intermediate representation
> - **Decision**: Direct output generation with conflict tracking metadata
> - **Rationale**: Direct generation reduces memory usage and eliminates an additional transformation pass. The conflict tracking metadata provides necessary validation capabilities without full parse tree overhead.
> - **Consequences**: More complex parsing logic but better performance and memory efficiency for large configuration files

**Error recovery strategy** determines how the parser responds to syntax errors and whether it attempts to continue parsing to report multiple errors. The architecture must balance comprehensive error reporting with parsing simplicity and reasonable recovery behavior.

The implemented approach uses **panic-mode recovery** where the parser skips tokens until it reaches a synchronization point (typically the start of the next table or key-value pair), then resumes parsing. This provides multiple error reports while maintaining reasonable parsing state consistency.

### Common TOML Parsing Pitfalls

Understanding the frequent mistakes in TOML parsing implementation helps avoid subtle bugs that can produce incorrect results or confusing error messages. These pitfalls often arise from the interaction between TOML's powerful features and the complexity they introduce in parser implementation.

⚠️ **Pitfall: Ignoring Table Redefinition Rules**

Many implementations incorrectly allow table redefinition or fail to properly distinguish between extending a table and redefining it. In TOML, once a table is explicitly declared with `[table.name]`, it cannot be redeclared, but additional keys can be added from other locations through dotted key notation.

The error manifests when parsing documents like:
```
[server]
host = "localhost"

[server]  # This should be an error - table redefinition
port = 8080
```

**Why it's wrong**: TOML specification explicitly forbids table redefinition to eliminate ambiguity about key placement and table structure. Allowing redefinition makes document interpretation dependent on declaration order.

**How to fix**: Maintain a set of explicitly declared table paths and check each new table declaration against this set. Distinguish between explicit table declarations (`[table]`) and implicit table creation through dotted keys.

⚠️ **Pitfall: Incorrect Array-of-Tables Mixing**

Implementations often fail to properly validate that array-of-tables declarations are used consistently. Mixing `[table]` and `[[table]]` syntax for the same path should produce an error, but many parsers incorrectly allow this or handle it inconsistently.

**Why it's wrong**: The distinction between single tables and array-of-tables is fundamental to TOML's type system. Mixing these creates ambiguous document interpretation and violates the specification's clarity goals.

**How to fix**: Track whether each table path is used as a single table or array-of-tables and validate consistency across all declarations. Maintain separate metadata for table type alongside the symbol table entries.

⚠️ **Pitfall: Improper Dotted Key Expansion**

Many implementations incorrectly handle the interaction between dotted keys and table declarations. When processing `a.b.c = "value"`, the parser must create implicit tables for `a` and `a.b`, but these implicit tables must be compatible with later explicit table declarations.

The complexity arises in sequences like:
```
physical.color = "orange"  # Creates implicit [physical] table
[physical.dimensions]      # Must extend the existing physical table
[physical]                 # Should be error - can't redefine implicit table explicitly
```

**Why it's wrong**: Incorrect dotted key expansion violates TOML's namespace consistency rules and can lead to key placement in wrong table locations.

**How to fix**: Mark implicitly created tables with metadata indicating their creation method, allow extension through additional dotted keys or sub-table declarations, but forbid explicit redeclaration of implicit tables.

⚠️ **Pitfall: Inadequate Inline Table Validation**

Inline tables using `{key = value, key2 = value2}` syntax have special scoping rules that many parsers implement incorrectly. Inline tables must be completely defined in their declaration and cannot be extended later through dotted keys or additional table declarations.

**Why it's wrong**: TOML treats inline tables as atomic units that cannot be modified after declaration, ensuring document clarity and preventing ambiguous key placement.

**How to fix**: Mark inline tables as immutable in the symbol table and validate that no subsequent operations attempt to add keys to or modify inline table contents.

⚠️ **Pitfall: Insufficient Unicode and Escape Sequence Handling**

TOML's string literals support complex escape sequences including Unicode code points, and many implementations handle these incorrectly or incompletely. This is particularly problematic for multiline strings where escape sequence rules differ between basic strings and literal strings.

**Why it's wrong**: Incorrect escape handling can corrupt string values, cause parsing failures on valid documents, or create security vulnerabilities through improper Unicode handling.

**How to fix**: Implement complete escape sequence processing according to TOML specification, handle Unicode normalization properly, and validate escape sequences during tokenization rather than during parsing.

⚠️ **Pitfall: Poor Error Message Context**

TOML parsing errors often involve complex interactions between different parts of the document (table redefinition, key conflicts, type mismatches), but many implementations provide error messages that don't include sufficient context to understand the problem.

**Why it's wrong**: Poor error messages make debugging configuration files extremely difficult, especially for users who may not understand TOML's complex rules about table definitions and key conflicts.

**How to fix**: Include both the error location and the conflicting previous definition location in error messages. Provide specific information about what rules were violated and suggestions for fixing the problem.

![TOML Table Parsing Sequence](./diagrams/toml-parsing-sequence.svg)

### Implementation Guidance

The TOML parser implementation requires sophisticated recursive descent techniques combined with comprehensive symbol table management. This guidance provides the foundation for building a parser that correctly handles TOML's complex table hierarchy and type system while maintaining clear separation between tokenization and parsing concerns.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Recursive Descent Framework | Manual recursive function calls with error handling | Parser combinator library with automatic backtracking |
| Symbol Table | Nested dictionaries with metadata annotations | Custom symbol table class with scope management |
| Type System | Python native types with isinstance() checking | Custom TOML type classes with validation methods |
| Error Reporting | Exception-based with position information | Error accumulation with multiple error reporting |
| Token Consumption | Linear token stream with index tracking | Token stream class with lookahead buffers |

#### Recommended File Structure

```
project-root/
  parsers/
    base_parser.py              ← Abstract base for all parsers
    toml_parser.py              ← Main TOML recursive descent parser
    toml_types.py               ← TOML-specific type definitions and validation
  tokenizers/
    toml_tokenizer.py           ← TOML tokenizer (from previous milestone)
  data_structures/
    parse_nodes.py              ← ParseNode hierarchy definitions
    symbol_table.py             ← Symbol table for conflict tracking
  tests/
    test_toml_parser.py         ← Comprehensive TOML parser tests
    fixtures/
      toml_examples/            ← TOML test files for each feature
        basic_tables.toml
        array_of_tables.toml
        dotted_keys.toml
        inline_structures.toml
```

#### Infrastructure Starter Code

**Symbol Table Implementation (`data_structures/symbol_table.py`):**
```python
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass

class DefinitionType(Enum):
    EXPLICIT_TABLE = "explicit_table"
    IMPLICIT_TABLE = "implicit_table"
    ARRAY_OF_TABLES = "array_of_tables"
    VALUE = "value"
    INLINE_TABLE = "inline_table"

@dataclass
class DefinitionInfo:
    definition_type: DefinitionType
    position: 'Position'
    is_mutable: bool = True
    
class SymbolTable:
    """Tracks all defined keys and tables for conflict detection."""
    
    def __init__(self):
        self.definitions: Dict[str, DefinitionInfo] = {}
        self.data: Dict[str, Any] = {}
        self.current_table_path: List[str] = []
    
    def register_definition(self, key_path: List[str], definition_type: DefinitionType, position: 'Position') -> None:
        """Register a new definition and validate for conflicts."""
        path_str = '.'.join(key_path)
        existing = self.definitions.get(path_str)
        
        if existing:
            # Validate compatibility based on TOML rules
            if definition_type == DefinitionType.EXPLICIT_TABLE and existing.definition_type == DefinitionType.IMPLICIT_TABLE:
                # Can explicitly define an implicit table
                self.definitions[path_str] = DefinitionInfo(definition_type, position)
                return
            elif definition_type == DefinitionType.ARRAY_OF_TABLES and existing.definition_type == DefinitionType.ARRAY_OF_TABLES:
                # Can extend array of tables
                return
            else:
                raise StructureError(f"Cannot redefine {path_str} as {definition_type}", position)
        
        self.definitions[path_str] = DefinitionInfo(definition_type, position)
    
    def create_nested_structure(self, key_path: List[str]) -> Dict[str, Any]:
        """Create nested dictionary structure for the given path."""
        current = self.data
        for segment in key_path[:-1]:
            if segment not in current:
                current[segment] = {}
            current = current[segment]
        return current
    
    def set_current_table(self, table_path: List[str]) -> None:
        """Set the current table context for key-value assignments."""
        self.current_table_path = table_path.copy()
    
    def get_current_table(self) -> Dict[str, Any]:
        """Get the dictionary representing the current table."""
        if not self.current_table_path:
            return self.data
        
        current = self.data
        for segment in self.current_table_path:
            if segment not in current:
                current[segment] = {}
            current = current[segment]
        return current
```

**TOML Type Validation (`parsers/toml_types.py`):**
```python
import re
from datetime import datetime, date, time
from typing import Any, Union, List, Dict

class TOMLTypeValidator:
    """Validates and converts TOML literal values to appropriate Python types."""
    
    # Regex patterns for TOML literals
    INTEGER_PATTERN = re.compile(r'^[+-]?(?:0|[1-9](?:_?\d)*)$')
    FLOAT_PATTERN = re.compile(r'^[+-]?(?:0|[1-9](?:_?\d)*)(?:\.(?:\d(?:_?\d)*)?)?(?:[eE][+-]?\d(?:_?\d)*)?$')
    BOOLEAN_PATTERN = re.compile(r'^(?:true|false)$')
    DATETIME_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$')
    
    @classmethod
    def validate_and_convert(cls, token: 'Token') -> Any:
        """Convert token value to appropriate Python type based on TOML rules."""
        if token.type == TokenType.STRING:
            return cls._process_string_literal(token)
        elif token.type == TokenType.NUMBER:
            return cls._process_numeric_literal(token)
        elif token.type == TokenType.BOOLEAN:
            return token.value == 'true'
        else:
            return token.value
    
    @classmethod
    def _process_string_literal(cls, token: 'Token') -> str:
        """Process string literal with proper escape sequence handling."""
        raw_value = token.raw_text
        
        if raw_value.startswith('"""') or raw_value.startswith("'''"):
            # Multiline string - handle special rules
            return cls._process_multiline_string(raw_value)
        elif raw_value.startswith('"'):
            # Basic string - process escape sequences
            return cls._process_basic_string(raw_value[1:-1])  # Remove quotes
        elif raw_value.startswith("'"):
            # Literal string - no escape processing
            return raw_value[1:-1]  # Remove quotes, no escape processing
        
        return raw_value
    
    @classmethod
    def _process_basic_string(cls, content: str) -> str:
        """Process escape sequences in basic strings."""
        # Handle standard escape sequences: \n, \t, \r, \\, \", etc.
        escape_map = {
            'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', '"': '"',
            'b': '\b', 'f': '\f', '/': '/'
        }
        
        result = []
        i = 0
        while i < len(content):
            if content[i] == '\\' and i + 1 < len(content):
                next_char = content[i + 1]
                if next_char in escape_map:
                    result.append(escape_map[next_char])
                    i += 2
                elif next_char == 'u' and i + 5 < len(content):
                    # Unicode escape sequence \uXXXX
                    unicode_hex = content[i+2:i+6]
                    result.append(chr(int(unicode_hex, 16)))
                    i += 6
                elif next_char == 'U' and i + 9 < len(content):
                    # Unicode escape sequence \UXXXXXXXX
                    unicode_hex = content[i+2:i+10]
                    result.append(chr(int(unicode_hex, 16)))
                    i += 10
                else:
                    # Invalid escape sequence
                    result.append(content[i])
                    i += 1
            else:
                result.append(content[i])
                i += 1
        
        return ''.join(result)
```

#### Core Logic Skeleton

**Main TOML Parser (`parsers/toml_parser.py`):**
```python
from typing import List, Dict, Any, Optional
from ..tokenizers.base_tokenizer import Token, TokenType
from ..data_structures.parse_nodes import ParseNode, SectionNode, KeyValueNode, ValueNode
from ..data_structures.symbol_table import SymbolTable, DefinitionType
from .base_parser import BaseParser

class TOMLParser(BaseParser):
    """Recursive descent parser for TOML configuration format."""
    
    def __init__(self, tokens: List[Token]):
        super().__init__(tokens)
        self.symbol_table = SymbolTable()
        self.current_token_index = 0
    
    def parse(self, content: str) -> Dict[str, Any]:
        """Main TOML parsing entry point - implements recursive descent algorithm."""
        # TODO 1: Initialize tokenizer and generate token stream from content
        # TODO 2: Reset parser state (symbol table, current position, etc.)
        # TODO 3: Enter main parsing loop - process document-level constructs
        # TODO 4: Handle end-of-file and validate final document state  
        # TODO 5: Return the nested dictionary structure from symbol table
        # Hint: Main loop alternates between table declarations and key-value pairs
        pass
    
    def parse_document_level_construct(self) -> Optional[ParseNode]:
        """Parse top-level TOML constructs (tables, key-value pairs, comments)."""
        # TODO 1: Skip whitespace and comments to find next meaningful token
        # TODO 2: Check for EOF condition and return None if document complete
        # TODO 3: Examine current token to determine construct type
        # TODO 4: Dispatch to appropriate parsing method based on token type
        # TODO 5: Handle unexpected tokens with informative error messages
        # Hint: Use lookahead to distinguish [table] from [[array-of-tables]]
        pass
    
    def parse_table_header(self) -> SectionNode:
        """Parse [table] and [[table.name]] declarations with hierarchy validation."""
        # TODO 1: Determine if single bracket [table] or double bracket [[table]]
        # TODO 2: Parse the table path (may be dotted like table.subtable.name)
        # TODO 3: Validate table path doesn't conflict with existing definitions
        # TODO 4: Register new table definition in symbol table with proper type
        # TODO 5: Create table hierarchy and set as current parsing context
        # TODO 6: Handle array-of-tables by creating new table instance and appending
        # Hint: Array-of-tables creates multiple table instances under same path
        pass
    
    def parse_table_path(self) -> List[str]:
        """Parse dotted table paths like 'database.connection.pool'."""
        # TODO 1: Start with first identifier token as initial path segment  
        # TODO 2: Check for DOT token indicating additional path segments
        # TODO 3: Consume DOT and parse next identifier in the path
        # TODO 4: Continue until no more DOT tokens found
        # TODO 5: Validate each path segment is a valid identifier
        # TODO 6: Return list of path segments for table creation
        # Hint: Handle quoted identifiers that may contain special characters
        pass
    
    def parse_key_value_pair(self) -> KeyValueNode:
        """Parse key assignments including dotted keys and complex values."""
        # TODO 1: Parse the key path (may be dotted like 'server.database.host')
        # TODO 2: Consume EQUALS token and validate assignment operator
        # TODO 3: Parse the value recursively based on its type
        # TODO 4: Handle dotted key expansion - create implicit tables as needed
        # TODO 5: Validate key doesn't already exist in target table
        # TODO 6: Store key-value mapping in appropriate table location
        # Hint: Dotted keys create nested table structure automatically
        pass
    
    def parse_value(self) -> ValueNode:
        """Dispatch to type-specific value parsers based on token examination."""
        # TODO 1: Examine current token type to determine value category
        # TODO 2: Dispatch to appropriate specialized parser method
        # TODO 3: Handle arrays by calling parse_array() for bracket notation
        # TODO 4: Handle inline tables by calling parse_inline_table() for braces
        # TODO 5: Handle literal values (strings, numbers, booleans, dates)
        # TODO 6: Validate parsed value conforms to TOML type system
        # Hint: Use lookahead for ambiguous cases like negative numbers vs expressions
        pass
    
    def parse_array(self) -> ValueNode:
        """Parse array literals [item1, item2, item3] with mixed type support."""
        # TODO 1: Consume opening ARRAY_START bracket token
        # TODO 2: Handle empty array case (immediate closing bracket)
        # TODO 3: Parse first array element by recursively calling parse_value()
        # TODO 4: Loop to parse additional elements separated by COMMA tokens
        # TODO 5: Handle trailing commas (optional in TOML arrays)
        # TODO 6: Consume closing ARRAY_END bracket and validate structure
        # TODO 7: Create ValueNode with array contents and return
        # Hint: TOML allows mixed-type arrays unlike some formats
        pass
    
    def parse_inline_table(self) -> ValueNode:
        """Parse inline table syntax {key1 = value1, key2 = value2}."""
        # TODO 1: Consume opening OBJECT_START brace token
        # TODO 2: Handle empty table case (immediate closing brace)
        # TODO 3: Parse first key-value pair within the inline table
        # TODO 4: Loop to parse additional pairs separated by COMMA tokens  
        # TODO 5: Validate inline table keys don't conflict with each other
        # TODO 6: Consume closing OBJECT_END brace and create table structure
        # TODO 7: Mark inline table as immutable in symbol table
        # Hint: Inline tables cannot be extended after declaration
        pass
    
    def validate_table_redefinition(self, table_path: List[str], definition_type: DefinitionType) -> None:
        """Validate new table definition against TOML redefinition rules."""
        # TODO 1: Convert table path to string representation for lookup
        # TODO 2: Check if path already exists in symbol table definitions
        # TODO 3: Apply TOML rules for valid redefinition cases
        # TODO 4: Allow explicit definition of previously implicit tables
        # TODO 5: Allow extension of array-of-tables with consistent syntax
        # TODO 6: Raise StructureError for invalid redefinition attempts
        # Hint: Different definition types have different redefinition rules
        pass
    
    def expand_dotted_key(self, key_path: List[str], value: Any) -> None:
        """Create nested table structure for dotted key assignment."""
        # TODO 1: Validate each segment of key path for conflicts
        # TODO 2: Create implicit tables for intermediate path segments
        # TODO 3: Register implicit table definitions in symbol table
        # TODO 4: Navigate to or create the target table for value assignment
        # TODO 5: Validate final key doesn't already exist in target table
        # TODO 6: Store the key-value pair in the appropriate nested location
        # Hint: Dotted keys can create deeply nested structures automatically
        pass
```

#### Milestone Checkpoints

**After implementing TOML tokenizer integration:**
```bash
python -m pytest tests/test_toml_parser.py::TestTOMLTokenizerIntegration -v
```
Expected: All token types correctly recognized, string literals properly parsed, numeric values tokenized with correct types.

**After implementing basic table parsing:**
```bash
python -c "
from parsers.toml_parser import TOMLParser
result = TOMLParser().parse('[server]\nhost = \"localhost\"\nport = 8080')
print('Success:', result == {'server': {'host': 'localhost', 'port': 8080}})
"
```
Expected: `Success: True` with proper nested dictionary structure.

**After implementing array-of-tables:**
```bash
python -c "
from parsers.toml_parser import TOMLParser
toml_content = '''
[[database.servers]]
ip = \"192.168.1.1\"
dc = \"eqdc10\"

[[database.servers]]  
ip = \"192.168.1.2\"
dc = \"eqdc10\"
'''
result = TOMLParser().parse(toml_content)
print('Array length:', len(result['database']['servers']))
print('First server IP:', result['database']['servers'][0]['ip'])
"
```
Expected: `Array length: 2` and `First server IP: 192.168.1.1`.

#### Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| "Key redefinition error" on valid TOML | Incorrect table conflict detection | Check symbol table state when error occurs | Review table redefinition validation rules |
| Array-of-tables creates single table instead of array | Missing double-bracket detection logic | Examine tokenizer output for bracket tokens | Implement proper lookahead for `[[` vs `[` |
| Dotted keys create incorrect nesting | Improper key path expansion | Trace key path parsing and table creation | Debug dotted key expansion algorithm step by step |
| Inline tables allow extension (should be immutable) | Missing immutability enforcement | Check if inline tables marked as immutable | Add immutability validation in symbol table |
| Complex nested structures parse incorrectly | Recursive descent state management issues | Print parser state at each recursive call | Review context switching and current table tracking |
| Unicode strings corrupted during parsing | Incorrect escape sequence processing | Test with known Unicode test cases | Debug string literal processing in type validator |


## YAML Parser Component Design

> **Milestone(s):** YAML Subset Parser - implements indentation-sensitive parsing for mappings, sequences, and scalar values with automatic type inference

The YAML parser represents the most conceptually different parsing challenge in our configuration file parser suite. While INI parsing follows line-based rules and TOML parsing uses explicit delimiters, YAML parsing requires understanding **indentation-driven hierarchical structure** where whitespace itself carries semantic meaning. This fundamental difference demands a completely different parsing approach built around stack-based nesting management and context-sensitive interpretation.

![YAML Indentation Stack Management](./diagrams/yaml-indentation-stack.svg)

### YAML Parsing Mental Model

Think of YAML parsing like reading a well-structured outline or book table of contents. The indentation level tells you exactly where you are in the hierarchy - a chapter, section, subsection, or bullet point. Just as you would track your current nesting level while reading an outline, the YAML parser maintains a **stack of indentation contexts** that grows and shrinks as the document flows through different nesting levels.

Consider this mental model: imagine you're organizing physical file folders on a desk. Each level of indentation represents placing a folder inside another folder. When you encounter a line with less indentation, you're "backing out" of nested folders until you reach the appropriate parent folder. When you encounter deeper indentation, you're "diving into" a subfolder structure. The parser maintains this folder stack in memory, always knowing exactly which "folder" (context) it's currently processing.

The **indentation-driven hierarchical structure** means that YAML parsing is fundamentally about tracking these context transitions. Unlike TOML where table boundaries are explicit (`[table.name]`) or INI where sections are clearly marked (`[section]`), YAML structure emerges from the spatial relationship between lines. This creates both elegance and complexity - the format reads naturally to humans but requires sophisticated state tracking for machines.

The parser must continuously answer three questions: "What nesting level am I at?", "Am I entering a deeper level or returning to a shallower one?", and "What type of structure am I building at this level?" This context sensitivity means that the same line can mean completely different things depending on the indentation stack state when it's encountered.

### Indentation-Based Parsing Algorithm

The **stack-based approach** forms the algorithmic foundation of YAML parsing. The parser maintains an **indentation stack** where each stack frame represents a nesting level with its indentation amount, structure type (mapping or sequence), and the data being built at that level. This approach handles the complex dance of tracking when structures begin, continue, and end based purely on whitespace changes.

The core algorithm operates through a **structure transition** state machine that processes each logical line by comparing its indentation against the current stack state:

1. **Tokenize the logical line** into its component parts: indentation amount, content type (key-value pair, sequence item, or scalar), and the actual content values. This tokenization must handle both block-style syntax (indentation-based) and flow-style syntax (bracket and brace delimited inline structures).

2. **Calculate indentation level** by counting leading whitespace characters. YAML strictly forbids mixing tabs and spaces for indentation, so this calculation must validate consistency and reject mixed whitespace. The indentation amount determines the structural relationship to previously parsed content.

3. **Compare current indentation to stack state** to determine the structural transition type. If indentation increases, we're entering a nested structure. If indentation decreases, we're exiting one or more nested structures. If indentation matches the top of stack, we're continuing the current structure at the same level.

4. **Handle stack transitions** based on the indentation comparison. For increased indentation, push a new stack frame for the nested structure. For decreased indentation, pop stack frames until reaching the matching indentation level, completing any nested structures during the unwinding process.

5. **Process the line content** within the appropriate structural context. Key-value pairs create or extend mapping structures. Sequence items (lines starting with dash) create or extend list structures. Scalar values get processed through type inference and stored in the current structure.

6. **Validate structural consistency** by ensuring that the indentation level matches exactly with a previous level when returning to a shallower nesting. YAML prohibits "dedenting" to indentation levels that were never established, preventing malformed structural transitions.

7. **Update parser state** by modifying the current stack frame's data structure with the processed line content and preparing for the next line. The stack top always represents the active parsing context for subsequent lines.

The **context-sensitive interpretation** means that identical content can create different structural results depending on the current stack state. A line containing `name: value` creates a top-level mapping entry when the stack is empty, adds a mapping entry to the current mapping when inside a mapping context, or creates a mapping value for a sequence item when inside a sequence context.

**Stack frame management** requires careful attention to data structure references. Each stack frame must maintain a reference to the actual data structure being built (dictionary for mappings, list for sequences) so that modifications during parsing affect the final output structure. The stack stores these references, not copies of the data.

The algorithm handles **implicit structure creation** by automatically determining structure types from content patterns. When encountering a key-value pair at a new indentation level, the parser creates a mapping structure. When encountering a sequence item (dash-prefixed line), the parser creates a sequence structure. This implicit creation eliminates the need for explicit structure declarations like TOML's table headers.

### YAML Type Inference Logic

YAML's **automatic type detection and conversion** system attempts to interpret scalar values as their most natural data types rather than treating everything as strings. This type inference creates intuitive behavior for users but introduces complexity in parsing logic that must recognize and convert various literal patterns.

The **scalar type inference** follows a precedence hierarchy that examines string content against increasingly specific patterns. The inference engine processes each scalar value through this decision tree:

| Pattern Type | Examples | Inferred Type | Conversion Logic |
|--------------|----------|---------------|------------------|
| Boolean literals | `true`, `false`, `yes`, `no`, `on`, `off` | Boolean | Case-insensitive string matching against known boolean values |
| Integer literals | `42`, `-17`, `0x1A`, `0o755`, `0b1010` | Integer | Regex pattern matching with base detection (decimal, hex, octal, binary) |
| Float literals | `3.14`, `-2.5e10`, `.5`, `1.2e-3` | Float | Scientific notation support with decimal point or exponent indicators |
| Date/time literals | `2023-12-31`, `12:30:45`, `2023-12-31T12:30:45Z` | Date/Time | ISO 8601 format detection with timezone support |
| Null literals | `null`, `~`, empty value | Null/None | Explicit null markers or empty values in certain contexts |
| String fallback | Everything else | String | Default case when no other patterns match |

The **boolean inference rules** recognize multiple conventional representations that users commonly expect to work as boolean values. The parser must handle case variations and cultural differences in boolean representation. Values like `TRUE`, `False`, `YES`, `No` all map to appropriate boolean values through case-insensitive comparison.

**Numeric type inference** involves pattern recognition that distinguishes integers from floating-point numbers while supporting alternative numeric bases. Integer patterns include standard decimal notation, hexadecimal (0x prefix), octal (0o prefix), and binary (0b prefix) representations. Float patterns recognize decimal points, scientific notation with e/E exponents, and special float values like infinity and NaN.

The **string escaping and quoting** system provides explicit control over type inference when automatic detection produces incorrect results. Single-quoted strings (`'value'`) preserve literal content without escape sequence processing and prevent type inference. Double-quoted strings (`"value"`) allow escape sequences like `\n`, `\t`, `\"`, and `\\` while still preventing type inference. Unquoted strings undergo full type inference processing.

**Multiline string handling** supports both literal block scalars (using `|` indicator) and folded block scalars (using `>` indicator). Literal blocks preserve line breaks and indentation exactly as written. Folded blocks collapse line breaks within paragraphs into spaces while preserving paragraph breaks indicated by blank lines.

Type inference must handle **edge cases** where the same string representation could reasonably map to multiple types. For example, `01:30:00` could represent a time value or a string. The parser uses context clues and follows YAML specification precedence rules to resolve these ambiguities consistently.

The **inference validation** process ensures that converted values make semantic sense and handles conversion errors gracefully. Invalid date formats, numeric overflows, and malformed patterns should produce clear error messages rather than silent failures or unexpected type assignments.

### YAML Architecture Decisions

The architecture decisions for YAML parsing center on managing the complexity of indentation-sensitive parsing while maintaining reasonable performance and error handling capabilities. These decisions directly impact both implementation complexity and user experience.

> **Decision: YAML Subset Selection**
> - **Context**: Full YAML specification includes advanced features like anchors, aliases, complex multiline syntax, and document streams that significantly increase implementation complexity without corresponding learning value for configuration parsing
> - **Options Considered**: Full YAML 1.2 specification, Common subset (mappings/sequences/scalars), Minimal subset (basic nesting only)
> - **Decision**: Implement common subset covering indentation-based mappings, sequences, basic scalars, and simple flow syntax
> - **Rationale**: The common subset covers 95% of real-world YAML configuration usage while keeping implementation complexity manageable for learning purposes. Advanced features like anchors and multi-document streams are rarely used in configuration files
> - **Consequences**: Users cannot use advanced YAML features, but implementation remains focused on core parsing concepts without getting lost in specification edge cases

| Subset Option | Features Included | Implementation Complexity | Real-World Coverage |
|---------------|-------------------|---------------------------|---------------------|
| Full YAML 1.2 | All specification features | Very High | 100% |
| Common Subset | Mappings, sequences, scalars, basic flow | Medium | 95% |
| Minimal Subset | Basic nesting only | Low | 60% |

> **Decision: Flow Syntax Support Level**
> - **Context**: YAML supports both block syntax (indentation-based) and flow syntax (JSON-like with brackets and braces). Flow syntax can appear inline within block structures, creating parsing complexity
> - **Options Considered**: No flow syntax support, Basic flow arrays/objects only, Full flow syntax with nesting
> - **Decision**: Support basic flow syntax for inline arrays and objects without deep nesting
> - **Rationale**: Basic flow syntax like `[item1, item2, item3]` and `{key: value, key2: value2}` appears frequently in real configurations and provides valuable parsing experience without excessive complexity
> - **Consequences**: Enables common inline syntax patterns while avoiding complex flow/block interaction edge cases that would complicate the parsing algorithm significantly

> **Decision: Indentation Validation Strictness**
> - **Context**: YAML requires consistent indentation patterns but allows flexibility in indentation amounts. Parser must decide how strictly to enforce indentation rules
> - **Options Considered**: Strict validation (exact indentation matching), Flexible validation (consistent increases/decreases), Lenient validation (best-effort interpretation)
> - **Decision**: Implement strict validation with clear error messages for indentation violations
> - **Rationale**: Strict validation helps users learn proper YAML formatting and prevents subtle bugs from inconsistent indentation. Clear error messages make the strictness helpful rather than frustrating
> - **Consequences**: Parser rejects malformed YAML that other parsers might accept, but provides better learning experience and more predictable behavior

The **multiline string handling** decision affects both parsing complexity and user experience with complex string values in configuration files:

> **Decision: Multiline String Support**
> - **Context**: YAML multiline strings use block scalar indicators (`|` and `>`) with complex indentation and line-ending rules that significantly complicate parsing logic
> - **Options Considered**: No multiline support, Basic literal blocks only, Full block scalar support with indicators
> - **Decision**: Support basic literal block scalars with `|` indicator for exact preservation
> - **Rationale**: Configuration files commonly need multiline strings for templates, SQL queries, or documentation. Literal blocks cover the most important use case with manageable parsing complexity
> - **Consequences**: Users can include multiline configuration values naturally, but advanced folding and block scalar features are not available

**Error recovery strategy** determines how the parser behaves when encountering invalid YAML structure:

| Recovery Strategy | Behavior on Error | User Experience | Implementation Complexity |
|-------------------|-------------------|-----------------|---------------------------|
| Fail Fast | Stop on first error | Clear failure point | Low |
| Continue Parsing | Collect multiple errors | Better error overview | Medium |
| Best-effort Recovery | Attempt to continue with assumptions | May mask real errors | High |

> **Decision: Error Recovery Approach**
> - **Context**: YAML parsing can fail in multiple ways (indentation errors, type conversion failures, syntax violations) and the parser must decide whether to stop immediately or attempt continued parsing
> - **Options Considered**: Fail-fast on first error, Continue parsing to collect multiple errors, Best-effort recovery with assumptions
> - **Decision**: Implement fail-fast approach with rich error context and suggestions for fix
> - **Rationale**: YAML's indentation sensitivity means that early errors often cascade into misleading later errors. Failing fast with clear context helps users fix the actual root cause rather than getting confused by secondary errors
> - **Consequences**: Users see one error at a time but get high-quality error messages that directly address the parsing failure cause

### Common YAML Parsing Pitfalls

YAML parsing introduces unique challenges that frequently trip up both parser implementers and users. Understanding these pitfalls helps create more robust parsers and better error messages.

⚠️ **Pitfall: Tab vs Space Indentation Mixing**

The most common YAML parsing error occurs when users mix tab and space characters for indentation. YAML specification explicitly prohibits this mixing, but many text editors make the distinction invisible to users. The parser encounters what appears to be consistent indentation but actually represents inconsistent whitespace character usage.

This manifests as parsing errors where the parser reports indentation level mismatches despite the visual appearance of correct indentation. Users often spend significant time checking their indentation manually without realizing the issue lies in invisible character differences.

**Detection approach**: During tokenization, track whether tabs or spaces are used for indentation and reject documents that mix both. Provide error messages that specifically identify the character type and line location where mixing occurs.

**Recovery strategy**: Report the exact character position and provide suggestions to convert all indentation to spaces (the recommended YAML convention) or use editor settings to visualize whitespace characters.

⚠️ **Pitfall: Implicit Type Conversion Surprises**

YAML's aggressive type inference creates unexpected behavior when string values accidentally match type inference patterns. Common examples include version numbers like `1.0` becoming floats instead of strings, boolean-like words becoming boolean values, and numeric strings being converted to integers.

Consider a configuration containing `version: 1.20` where the user expects string `"1.20"` but gets float `1.2` due to type inference. Similarly, values like `yes`, `no`, `on`, `off` automatically become boolean values even when users intended string values.

**Detection approach**: Implement type inference warnings or provide explicit control through quoted string syntax. When encountering common problematic patterns, consider generating warnings that inform users about automatic type conversion.

**Recovery strategy**: Document type inference rules clearly and show users how to use quoted strings (`"1.20"`) to force string type when automatic inference produces incorrect results. Consider providing a strict mode that requires explicit typing for ambiguous values.

⚠️ **Pitfall: Inconsistent Indentation Level Returns**

When returning to a previous indentation level, YAML requires exact matching with a previously established level. Users often "dedent" to indentation amounts that were never used, creating invalid structure transitions that the parser cannot interpret.

This occurs when users inconsistently indent nested structures, such as using 2 spaces for the first level, 4 spaces for the second level, and then 3 spaces when attempting to return to an intermediate level that was never established.

**Error pattern example**: Starting with 0 spaces (root), going to 2 spaces (level 1), then 6 spaces (level 2), and attempting to return to 4 spaces (invalid - level never existed).

**Detection approach**: Maintain a stack of established indentation levels and validate that any dedent operation returns to exactly one of those levels. Reject documents where dedent attempts target indentation amounts that don't match the stack.

**Recovery strategy**: Provide error messages showing the established indentation levels and suggest valid dedent targets. Include visual representation of the indentation stack state to help users understand the structural context.

⚠️ **Pitfall: Flow Syntax Context Confusion**

When YAML parsers support both block syntax (indentation-based) and flow syntax (bracket/brace delimited), users can create confusing combinations where flow syntax appears within block contexts or vice versa. The parser must correctly handle context switches between these two syntactic modes.

Complex cases arise when arrays or objects defined using flow syntax contain values that look like block syntax, or when block structures attempt to contain flow-syntax elements without proper nesting understanding.

**Context switching errors**: Users might start an array with flow syntax `[item1,` and then attempt to continue with block syntax indentation for subsequent items, creating syntactically invalid combinations.

**Detection approach**: Track parsing context to distinguish between block and flow modes. Validate that syntax elements are appropriate for the current context and provide clear errors when context violations occur.

**Recovery strategy**: Explain the difference between block and flow syntax clearly in error messages. Suggest consistent syntax usage and provide examples of correct flow-within-block or block-within-flow patterns.

⚠️ **Pitfall: Scalar Value Ambiguity in Lists**

YAML sequence items (list elements) can contain either simple scalar values or complex nested structures. The parser must correctly determine whether a sequence item is a simple value or the beginning of a nested structure, especially when the distinction isn't immediately clear from the first line.

Ambiguity occurs with patterns like:
```
items:
  - name: first item
    description: a complex item
  - simple item
  - name: third item
```

The parser must recognize that some sequence items are simple scalars while others are complex mappings, even though the determination might require lookahead parsing to make the distinction.

**Detection approach**: Use lookahead parsing to examine subsequent lines when processing sequence items. If the next line has greater indentation and contains a key-value pattern, treat the sequence item as a complex mapping. Otherwise, treat it as a simple scalar.

**Recovery strategy**: When ambiguity exists, provide clear error messages that explain how the parser interpreted the structure and suggest formatting changes to make the intent explicit.

### Implementation Guidance

The YAML parser implementation requires careful balance between handling indentation complexity and maintaining clean, debuggable code structure. The stack-based parsing approach translates naturally to object-oriented design with clear separation between tokenization, structure tracking, and value processing responsibilities.

#### Technology Recommendations

| Component | Simple Option | Advanced Option | Recommendation for Learning |
|-----------|---------------|-----------------|---------------------------|
| Indentation Tracking | Simple integer stack | Structured context objects | Structured context objects |
| Type Inference | Basic regex patterns | Comprehensive type system | Basic regex with clear rules |
| Error Handling | Exception-based | Result/Option types | Exception-based with context |
| Flow Syntax | Skip entirely | Full bracket/brace parsing | Basic inline arrays/objects |
| String Processing | Basic quote handling | Full escape sequence support | Full escape sequences |

#### Recommended File Structure

The YAML parser should integrate cleanly with the existing configuration parser architecture while maintaining clear separation of indentation-specific logic:

```
config-parser/
  parsers/
    yaml_parser.py              ← Main YAML parser entry point
    yaml_tokenizer.py           ← YAML-specific tokenization
    yaml_structures.py          ← Stack frame and context objects
    yaml_types.py               ← Type inference logic
  tests/
    test_yaml_parser.py         ← Comprehensive YAML tests
    fixtures/
      yaml/                     ← Test YAML files with edge cases
  examples/
    yaml_parsing_example.py     ← Demonstration of parser usage
```

This structure separates the indentation-sensitive parsing logic from the general tokenization framework while maintaining integration with the overall parser architecture.

#### Infrastructure Starter Code

Complete indentation stack management infrastructure that handles the complex state tracking required for YAML parsing:

```python
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass

class YAMLStructureType(Enum):
    """Types of YAML structures that can be nested."""
    MAPPING = "mapping"
    SEQUENCE = "sequence"
    DOCUMENT = "document"

@dataclass
class IndentationFrame:
    """Represents one level of nesting in the indentation stack."""
    indent_level: int
    structure_type: YAMLStructureType
    data: Union[Dict[str, Any], List[Any]]
    line_number: int
    
    def add_mapping_entry(self, key: str, value: Any) -> None:
        """Add a key-value pair to this mapping frame."""
        if self.structure_type != YAMLStructureType.MAPPING:
            raise ValueError(f"Cannot add mapping entry to {self.structure_type}")
        self.data[key] = value
    
    def add_sequence_item(self, item: Any) -> None:
        """Add an item to this sequence frame."""
        if self.structure_type != YAMLStructureType.SEQUENCE:
            raise ValueError(f"Cannot add sequence item to {self.structure_type}")
        self.data.append(item)

class IndentationStack:
    """Manages the stack of indentation contexts during YAML parsing."""
    
    def __init__(self):
        self.stack: List[IndentationFrame] = []
        self.established_levels: List[int] = [0]  # Track valid dedent levels
    
    def current_level(self) -> int:
        """Get the current indentation level."""
        return self.stack[-1].indent_level if self.stack else 0
    
    def current_frame(self) -> Optional[IndentationFrame]:
        """Get the current stack frame."""
        return self.stack[-1] if self.stack else None
    
    def push_frame(self, indent_level: int, structure_type: YAMLStructureType, line_number: int) -> IndentationFrame:
        """Push a new indentation frame onto the stack."""
        if structure_type == YAMLStructureType.MAPPING:
            data = {}
        elif structure_type == YAMLStructureType.SEQUENCE:
            data = []
        else:
            data = {}
        
        frame = IndentationFrame(indent_level, structure_type, data, line_number)
        self.stack.append(frame)
        
        if indent_level not in self.established_levels:
            self.established_levels.append(indent_level)
        
        return frame
    
    def pop_to_level(self, target_level: int) -> List[IndentationFrame]:
        """Pop frames until reaching the target indentation level."""
        if target_level not in self.established_levels:
            raise StructureError(
                f"Invalid dedent to level {target_level}. Valid levels: {self.established_levels}",
                position=None,
                suggestion=f"Use one of these indentation levels: {self.established_levels}"
            )
        
        popped_frames = []
        while self.stack and self.stack[-1].indent_level > target_level:
            popped_frames.append(self.stack.pop())
        
        return popped_frames
    
    def get_root_data(self) -> Dict[str, Any]:
        """Get the root document data structure."""
        return self.stack[0].data if self.stack else {}

# Type inference utilities for YAML scalar processing
class YAMLTypeInference:
    """Handles automatic type detection and conversion for YAML scalars."""
    
    BOOLEAN_VALUES = {
        'true': True, 'false': False,
        'yes': True, 'no': False,
        'on': True, 'off': False,
        'y': True, 'n': False
    }
    
    NULL_VALUES = {'null', '~', ''}
    
    @classmethod
    def infer_type(cls, value_str: str) -> Any:
        """Convert string value to appropriate Python type."""
        if not isinstance(value_str, str):
            return value_str
        
        # Remove leading/trailing whitespace
        clean_value = value_str.strip()
        
        # Handle empty values and null
        if clean_value in cls.NULL_VALUES:
            return None
        
        # Handle boolean values (case insensitive)
        lower_value = clean_value.lower()
        if lower_value in cls.BOOLEAN_VALUES:
            return cls.BOOLEAN_VALUES[lower_value]
        
        # Handle numeric values
        try:
            # Try integer first
            if '.' not in clean_value and 'e' not in lower_value:
                return int(clean_value)
            # Try float
            return float(clean_value)
        except ValueError:
            pass
        
        # Return as string if no other type matches
        return clean_value

# Complete error handling infrastructure for YAML-specific errors
class YAMLIndentationError(StructureError):
    """Error in YAML indentation structure."""
    
    def __init__(self, message: str, line_number: int, current_levels: List[int]):
        super().__init__(
            message=message,
            position=Position(line=line_number, column=0, offset=0),
            suggestion=f"Valid indentation levels are: {current_levels}"
        )
        self.current_levels = current_levels

class YAMLTypeError(ParseError):
    """Error in YAML type inference or conversion."""
    pass
```

#### Core Logic Skeleton Code

The main YAML parser implementation with detailed TODO comments mapping to the algorithm steps:

```python
class YAMLParser:
    """YAML subset parser with indentation-based structure tracking."""
    
    def __init__(self):
        self.stack = IndentationStack()
        self.type_inference = YAMLTypeInference()
        self.current_line = 0
    
    def parse(self, content: str) -> Dict[str, Any]:
        """
        Parse YAML content into nested dictionary structure.
        
        Args:
            content: Raw YAML text content
            
        Returns:
            Parsed configuration as nested dictionary
            
        Raises:
            YAMLIndentationError: For indentation structure violations
            YAMLTypeError: For type inference failures
            SyntaxError: For general syntax violations
        """
        # TODO 1: Split content into logical lines, handling line continuations
        # TODO 2: Initialize document root frame on indentation stack
        # TODO 3: Process each logical line through indentation analysis
        # TODO 4: Handle context transitions based on indentation changes
        # TODO 5: Parse line content based on current structural context
        # TODO 6: Validate final document structure completeness
        # TODO 7: Return root data structure from stack
        
        # Hint: Use self._process_line() for each line after indentation analysis
        # Hint: Track line numbers for error reporting throughout parsing
        pass
    
    def _analyze_line_indentation(self, line: str, line_number: int) -> Tuple[int, str]:
        """
        Analyze line indentation and extract content.
        
        Args:
            line: Raw line text
            line_number: Line number for error reporting
            
        Returns:
            Tuple of (indentation_level, content_text)
            
        Raises:
            YAMLIndentationError: For invalid indentation patterns
        """
        # TODO 1: Count leading whitespace characters (spaces only, reject tabs)
        # TODO 2: Validate indentation character consistency (no tab mixing)
        # TODO 3: Extract content portion after removing indentation
        # TODO 4: Return indentation level and cleaned content
        
        # Hint: Use enumerate() to track character positions for error reporting
        # Hint: Raise YAMLIndentationError for tab characters with helpful message
        pass
    
    def _handle_indentation_transition(self, current_indent: int, line_number: int) -> None:
        """
        Handle stack transitions based on indentation level changes.
        
        Args:
            current_indent: Indentation level of current line
            line_number: Current line number for error context
        """
        # TODO 1: Compare current indentation to stack top level
        # TODO 2: If deeper indentation, prepare for nested structure (don't push yet)
        # TODO 3: If same indentation, continue current structure context
        # TODO 4: If shallower indentation, pop stack frames to matching level
        # TODO 5: Validate dedent targets against established indentation levels
        
        # Hint: Use self.stack.pop_to_level() for dedent operations
        # Hint: Defer frame pushing until content type is determined
        pass
    
    def _process_line_content(self, content: str, indent_level: int, line_number: int) -> None:
        """
        Process line content based on YAML syntax patterns.
        
        Args:
            content: Line content after indentation removal
            indent_level: Indentation level of this line
            line_number: Line number for error reporting
        """
        # TODO 1: Skip empty lines and comment lines (starting with #)
        # TODO 2: Detect line type: mapping (key:), sequence (-), or scalar
        # TODO 3: Handle mapping entries by parsing key-value pairs
        # TODO 4: Handle sequence items by parsing list elements
        # TODO 5: Handle scalar values with type inference
        # TODO 6: Create appropriate stack frames for nested structures
        
        # Hint: Use regex patterns to identify mapping vs sequence vs scalar
        # Hint: Call self._parse_mapping_line() or self._parse_sequence_line()
        pass
    
    def _parse_mapping_line(self, content: str, indent_level: int, line_number: int) -> None:
        """Parse a mapping line (key: value format)."""
        # TODO 1: Split content on first colon to separate key and value
        # TODO 2: Validate key format and handle quoted keys
        # TODO 3: Trim whitespace from key and value portions  
        # TODO 4: If no current mapping frame, create one at this indent level
        # TODO 5: If value is empty, prepare for nested structure on next line
        # TODO 6: If value present, apply type inference and store mapping entry
        # TODO 7: Handle inline flow syntax in values (basic arrays/objects)
        
        # Hint: Use self.stack.current_frame() to check current context
        # Hint: Empty values after colon indicate nested structure follows
        pass
    
    def _parse_sequence_line(self, content: str, indent_level: int, line_number: int) -> None:
        """Parse a sequence line (- item format)."""
        # TODO 1: Remove leading dash and whitespace to get item content
        # TODO 2: If no current sequence frame, create one at this indent level
        # TODO 3: If item content is empty, prepare for nested structure
        # TODO 4: If item content present, apply type inference and add to sequence
        # TODO 5: Handle complex sequence items that are mappings
        
        # Hint: Sequence items can contain nested mappings or other sequences
        # Hint: Use lookahead to determine if item is scalar or complex structure
        pass
```

#### Language-Specific Implementation Hints

Python-specific techniques for effective YAML parser implementation:

- **String processing**: Use `str.lstrip()` to remove leading whitespace, but manually count characters to distinguish tabs from spaces
- **Stack management**: Python lists work perfectly as stacks with `append()` and `pop()` operations
- **Type inference**: Use `isinstance()` checks and `try/except` blocks for numeric conversion attempts  
- **Regular expressions**: Import `re` module for pattern matching mapping syntax (`key:`) and sequence syntax (`-`)
- **Error context**: Use `enumerate()` when iterating over lines to maintain line number tracking
- **Unicode handling**: Python 3 strings handle Unicode automatically, but be aware of Unicode whitespace characters beyond ASCII space and tab

#### Milestone Checkpoint

After implementing the YAML parser component, verify functionality with these concrete tests:

**Basic parsing verification:**
```bash
python -c "
from parsers.yaml_parser import YAMLParser
parser = YAMLParser()
result = parser.parse('name: test\nage: 25\nactive: true')
print('Parsed:', result)
assert result == {'name': 'test', 'age': 25, 'active': True}
print('✓ Basic parsing works')
"
```

**Indentation structure testing:**
```bash
python -c "
from parsers.yaml_parser import YAMLParser
yaml_content = '''
database:
  host: localhost
  port: 5432
  credentials:
    username: admin
    password: secret
'''
parser = YAMLParser()
result = parser.parse(yaml_content.strip())
print('Nested result:', result)
assert result['database']['credentials']['username'] == 'admin'
print('✓ Nested indentation works')
"
```

**Expected behavior verification:**
- Parser should handle 2-space, 4-space, or consistent indentation amounts
- Type inference should convert `25` to integer, `true` to boolean, quoted strings to strings
- Indentation errors should provide helpful error messages showing valid indent levels
- Mixed tab/space indentation should be rejected with clear error explaining the problem

**Signs of implementation problems:**
- "KeyError" exceptions usually indicate missing stack frame management
- "Indentation level mismatch" errors suggest issues with dedent level validation  
- Type conversion errors indicate problems in the type inference logic
- Stack overflow suggests infinite recursion in nested structure handling


## Interactions and Data Flow

> **Milestone(s):** All milestones (INI Parser, TOML Tokenizer, TOML Parser, YAML Subset Parser) - this section defines how all components work together to create a unified parsing system

Think of the complete parsing system as an assembly line in a specialized translation factory. Raw configuration files enter at one end as unstructured text, pass through a series of specialized workstations (format detection, tokenization, parsing, and structure conversion), and emerge as clean, standardized nested dictionaries. Each workstation has a specific expertise and passes its refined output to the next station, with quality control checkpoints that can halt the entire line if defects are detected. The beauty of this assembly line is that while each workstation uses different techniques internally, they all communicate through standardized interfaces, allowing the line to handle three completely different "product types" (INI, TOML, YAML) while producing identical output formats.

The interactions between components follow a carefully orchestrated pipeline where data flows through well-defined interfaces, errors are enriched with context as they propagate upward, and format detection automatically routes content to the appropriate specialized parser. This design enables clean separation of concerns while maintaining a unified experience for users of the parsing library.

![System Component Architecture](./diagrams/system-architecture.svg)

### Complete Parsing Pipeline

The complete parsing pipeline represents the end-to-end journey from raw configuration file content to fully processed nested dictionary structures. This pipeline consists of several distinct phases, each with specific responsibilities and well-defined interfaces for data exchange.

Think of this pipeline as a sophisticated mail sorting facility. Raw mail (configuration content) arrives in various formats and languages (INI, TOML, YAML). The facility first examines each piece to determine its origin language and routing requirements (format detection). Next, specialized language experts break down the content into meaningful units (tokenization). Then, grammar specialists organize these units into logical structures (parsing). Finally, translators convert everything into a common internal language (unified output format) that the rest of the organization can understand. At each stage, quality inspectors check for problems and can route defective items to error handling processes.

![Complete Parsing Pipeline Flow](./diagrams/parsing-pipeline.svg)

#### Pipeline Phase Breakdown

The parsing pipeline operates through five distinct phases, each building upon the output of the previous phase:

| Phase | Input | Output | Primary Responsibility | Error Types |
|-------|--------|---------|----------------------|-------------|
| Content Ingestion | File path or raw string | String content with encoding normalization | Read file content, handle encoding, normalize line endings | IO errors, encoding errors |
| Format Detection | String content | Format identifier (ini/toml/yaml) | Analyze content patterns to determine format | Ambiguous format errors |
| Tokenization | String content + format | List of typed tokens with positions | Break content into meaningful lexical units | Lexical errors, invalid characters |
| Parsing | Token stream + format | Parse tree structure | Build hierarchical structure from tokens | Syntax errors, structure errors |
| Output Conversion | Parse tree | Unified dictionary structure | Convert format-specific structures to common representation | Type conversion errors, structure mapping errors |

The main entry point `parse_config(content, format=None)` orchestrates this entire pipeline, handling the coordination between phases and managing error propagation. When a format is not explicitly specified, the pipeline automatically detects the format before proceeding to tokenization.

#### Data Transformation Flow

Each phase transforms data from one representation to another, with specific rules governing these transformations:

**Content Ingestion Transformation**: Raw file content undergoes normalization to ensure consistent processing regardless of source encoding or line ending conventions. The ingestion phase converts all content to UTF-8 encoding and normalizes line endings to Unix-style newlines. This normalization prevents encoding-related parsing errors in later phases.

**Format Detection Transformation**: Normalized content is analyzed using format-specific heuristics to produce a format identifier. This transformation is read-only - the content remains unchanged, but metadata about its format is added to guide subsequent processing decisions.

**Tokenization Transformation**: String content is decomposed into a sequence of typed tokens, each carrying a specific semantic meaning and position information. This transformation is lossy in the sense that whitespace semantics and comment placement may be simplified, but all semantically meaningful content is preserved with enhanced type information.

**Parsing Transformation**: The token stream is restructured into a hierarchical parse tree that reflects the logical organization of the configuration data. This transformation handles the complex task of resolving nested structures, managing scope transitions, and validating structural consistency according to format-specific rules.

**Output Conversion Transformation**: Format-specific parse tree structures are normalized into a unified nested dictionary representation. This transformation handles impedance mismatch between different format paradigms, ensuring that sections, tables, and mappings all produce equivalent dictionary structures.

#### Pipeline State Management

The parsing pipeline maintains state information that flows between phases to enable error reporting, debugging, and incremental processing:

| State Component | Purpose | Maintained By | Used By |
|-----------------|---------|---------------|---------|
| Source Position | Track current location in original content | Tokenizer | Parser, Error Reporter |
| Format Context | Remember detected format and configuration options | Format Detector | Tokenizer, Parser |
| Symbol Tables | Track defined keys and tables for conflict detection | Parser | Parser (validation) |
| Error Context | Accumulate errors with source positions | All phases | Error Reporter |
| Parse Stack | Maintain parsing context for nested structures | Parser | Parser (recursive descent) |

The pipeline uses a **context object** that flows through all phases, accumulating information and providing shared services. This context object implements the following interface:

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `get_position()` | None | `Position` | Get current source position |
| `set_position(pos)` | `pos: Position` | None | Update current source position |
| `add_error(error)` | `error: ParseError` | None | Record error with current context |
| `get_errors()` | None | `List[ParseError]` | Retrieve all accumulated errors |
| `get_format()` | None | `str` | Get detected format identifier |
| `set_format(fmt)` | `fmt: str` | None | Set format for pipeline |
| `get_options()` | None | `dict` | Get format-specific parsing options |

> **Design Insight**: The context object serves as both a communication channel and a memory mechanism, allowing later phases to access information discovered by earlier phases while maintaining loose coupling between components.

#### Error Recovery in Pipeline

The parsing pipeline implements **graceful degradation** where errors in one phase don't necessarily halt the entire pipeline. Each phase can choose to continue processing despite encountering errors, allowing the collection of multiple error reports in a single parsing attempt:

1. **Content Ingestion**: IO errors or encoding errors immediately halt the pipeline, as no further processing is possible without readable content.

2. **Format Detection**: Ambiguous format detection doesn't halt the pipeline; instead, the system defaults to INI format (simplest) and continues, recording a warning about the format assumption.

3. **Tokenization**: Invalid characters or lexical errors are recorded as `TokenError` instances, but tokenization continues by inserting `INVALID` tokens and resuming at the next recognizable boundary.

4. **Parsing**: Syntax errors are recorded as `SyntaxError` instances, and the parser attempts to resynchronize by finding the next stable parsing boundary (section header, top-level key, etc.).

5. **Output Conversion**: Type conversion errors are recorded as warnings, with problematic values preserved as strings in the final output.

This error recovery strategy enables the pipeline to provide comprehensive feedback about multiple issues in a single parsing attempt, significantly improving the developer experience when debugging configuration files.

### Format Detection Strategy

Format detection serves as the intelligent routing mechanism that automatically determines which specialized parser should handle incoming configuration content. Think of format detection as an experienced librarian who can instantly recognize whether a book is written in English, French, or German just by glancing at a few pages - they don't need to read the entire book, just identify the distinctive patterns that reveal the language.

The format detection system analyzes content using a combination of **syntactic signatures** and **statistical patterns** to make confident format determinations even with partial or ambiguous content. This approach is crucial because configuration files often contain overlapping syntactic elements (all three formats support key-value pairs and comments), requiring sophisticated heuristics to distinguish between them.

#### Detection Algorithm Strategy

The `detect_format(content)` function implements a **multi-pass analysis strategy** that examines different aspects of the content to build confidence in format identification:

1. **Signature Scanning Pass**: Look for unambiguous format-specific signatures that immediately identify the format
2. **Pattern Analysis Pass**: Analyze the frequency and distribution of format-specific patterns
3. **Conflict Resolution Pass**: Resolve ambiguities using format-specific tie-breaking rules
4. **Confidence Assessment Pass**: Evaluate the reliability of the detection and flag uncertain cases

| Detection Pass | TOML Signatures | INI Signatures | YAML Signatures | Weight |
|----------------|-----------------|----------------|-----------------|---------|
| Signature Scanning | `[[array.of.tables]]`, `key.dotted.path =`, `{ inline = "table" }` | `[section]` without dotted paths, `key=value` without quotes | `key: value`, `- list item`, indentation-based nesting | High (90% confidence) |
| Pattern Analysis | Quoted keys, date-time literals, mixed-type arrays | Semicolon comments, unquoted values, flat sections | Flow syntax `{}[]`, implicit type inference patterns | Medium (70% confidence) |
| Conflict Resolution | Complex table structures, type-specific literals | Simple key-value with minimal nesting | Indentation consistency, mapping vs sequence patterns | Low (50% confidence) |

The detection algorithm processes these passes sequentially, stopping early if high-confidence signatures are found, or continuing through all passes for ambiguous content.

#### Format-Specific Detection Rules

Each format has distinctive characteristics that enable reliable detection:

**TOML Detection Signatures**:
- **Definitive signatures**: Double-bracket array-of-tables syntax `[[database.servers]]`, dotted table paths `[tool.poetry.dependencies]`, and inline tables `person = { name = "Tom", age = 30 }`
- **Strong indicators**: Quoted keys using double or single quotes, date-time literals with timezone information, and mixed-type arrays with explicit type annotations
- **Supporting patterns**: Underscores in numeric literals `1_000_000`, multi-line strings with triple quotes, and boolean values using lowercase `true/false`

**INI Detection Signatures**:
- **Definitive signatures**: Simple bracketed sections `[Section Name]` without dots, semicolon-prefixed comments `;comment`, and unquoted string values
- **Strong indicators**: Key-value pairs using colon syntax `key: value`, global keys appearing before any section headers, and minimal use of nested structures
- **Supporting patterns**: Quoted string values using double quotes only, numeric values without type prefixes, and flat organizational structure

**YAML Detection Signatures**:
- **Definitive signatures**: Indentation-based nesting with consistent spaces, list items prefixed with dash and space `- item`, and mapping syntax using colon-space `key: value`
- **Strong indicators**: Flow syntax mixing with block syntax, implicit type inference for `yes/no/on/off`, and multi-document separators `---`
- **Supporting patterns**: Quoted strings using single or double quotes interchangeably, null values represented as `~` or `null`, and folded/literal string syntax `|` and `>`

> **Decision: Multi-Pass Detection Algorithm**
> - **Context**: Single-pass detection often fails on ambiguous files that mix format conventions or have minimal content
> - **Options Considered**: Single regex-based detection, machine learning classifier, multi-pass heuristic analysis
> - **Decision**: Multi-pass heuristic analysis with signature scanning and pattern analysis
> - **Rationale**: Provides high accuracy without external dependencies, handles edge cases gracefully, and offers explainable detection results
> - **Consequences**: More complex implementation but significantly better accuracy on real-world configuration files with mixed conventions

#### Detection Confidence and Fallback Strategy

The format detection system provides **confidence scoring** for its determinations, allowing the parsing pipeline to make informed decisions about how to handle uncertain cases:

| Confidence Level | Score Range | Action Taken | Error Handling |
|------------------|-------------|--------------|----------------|
| High Confidence | 90-100% | Proceed with detected format | Standard error reporting |
| Medium Confidence | 70-89% | Proceed with warning logged | Enhanced error context |
| Low Confidence | 50-69% | Proceed with user notification | Suggest manual format specification |
| Ambiguous | Below 50% | Default to INI with warning | Multiple format attempt on failure |

The fallback strategy implements **graceful degradation** where low-confidence detections still allow parsing to proceed, but with enhanced error reporting that includes format detection uncertainty in error messages.

#### Ambiguity Resolution Techniques

When content exhibits patterns consistent with multiple formats, the detection system applies format-specific tie-breaking rules:

**INI vs TOML Resolution**: INI format takes precedence when content uses simple bracketed sections without dotted paths and avoids TOML-specific features like inline tables or complex data types. TOML format takes precedence when dotted notation appears in section names or when complex data structures are present.

**YAML vs INI Resolution**: YAML format takes precedence when consistent indentation-based nesting is detected or when list syntax with dash prefixes appears. INI format takes precedence when content is predominantly flat with minimal nesting and uses semicolon-style comments.

**TOML vs YAML Resolution**: This represents the most complex ambiguity case. TOML format takes precedence when section headers use bracket notation and when explicit type annotations appear. YAML format takes precedence when indentation-based structure dominates and when implicit type inference patterns are prevalent.

### Error Information Flow

Error handling in the configuration parser implements a **enriched propagation model** where errors are detected at their source, progressively enhanced with context information as they move up the component stack, and finally transformed into user-friendly diagnostic messages. Think of this as a hospital's patient information system: when a problem is first detected in a lab test, it starts with raw technical data, but as it moves through specialists and ultimately to the patient's doctor, each step adds context, interpretation, and actionable guidance until the final message is both technically accurate and humanly comprehensible.

The error information flow ensures that users receive not just notification that something went wrong, but specific guidance about what the problem is, where it occurred, why it happened, and how to fix it. This comprehensive error reporting significantly reduces debugging time and improves the developer experience when working with configuration files.

![Error Detection and Reporting Flow](./diagrams/error-handling-flow.svg)

#### Error Detection Points

Errors can originate from multiple points in the parsing pipeline, each with different characteristics and context requirements:

| Detection Point | Error Types | Available Context | Enhancement Needed |
|-----------------|-------------|-------------------|-------------------|
| Content Ingestion | IO errors, encoding errors | File path, system error codes | User-friendly file access guidance |
| Format Detection | Ambiguous format, unknown patterns | Content sample, detection confidence scores | Format suggestion with reasoning |
| Tokenization | Invalid characters, malformed literals | Character position, token context | Visual indication of problem location |
| Parsing | Syntax errors, structure violations | Token stream, parsing state | Grammar explanation and fix suggestions |
| Type Conversion | Invalid values, type mismatches | Value context, expected type | Type conversion guidance and examples |

Each detection point creates errors using format-specific error types that inherit from the base `ParseError` class, ensuring consistent interface while enabling specialized handling.

#### Error Context Enrichment

As errors propagate up through the component stack, each level adds contextual information that makes the error more actionable for end users:

**Position Enhancement**: Raw character offsets from tokenization are converted to human-readable line and column numbers. The `current_position(source, offset)` function calculates these coordinates and creates `Position` objects that track line, column, and absolute offset information.

**Source Context Addition**: The `create_error_context(source, position, context_lines=2)` function generates visual error context by extracting surrounding lines from the source content and highlighting the specific problem location. This creates the familiar "caret pointer" display that shows exactly where the error occurred.

**Suggestion Generation**: Each component adds format-specific suggestions based on common error patterns. For example, TOML parsing errors include suggestions about table redefinition rules, while YAML parsing errors include guidance about indentation requirements.

**Error Classification Refinement**: Generic parsing errors are reclassified into specific error types (`TokenError`, `SyntaxError`, `StructureError`) that enable targeted error handling and user guidance.

#### Error Propagation Strategy

The error propagation system implements **structured error accumulation** where multiple errors can be collected and reported together, rather than halting on the first error encountered:

| Propagation Level | Responsibilities | Error Transformation | Context Added |
|------------------|------------------|---------------------|---------------|
| Component Level | Detect and classify errors | Raw errors → typed errors | Component-specific context |
| Parser Level | Coordinate error collection | Individual errors → error collections | Parsing state context |
| Pipeline Level | Aggregate cross-component errors | Component errors → unified reports | Source file context |
| API Level | Format for end-user consumption | Technical errors → user guidance | Help text and examples |

The propagation strategy uses **error aggregation objects** that collect related errors and provide unified reporting:

```
Error Report Structure:
- Primary Error: The main issue that prevented successful parsing
- Secondary Errors: Related issues that may have contributed to the primary error  
- Context Information: Source location, surrounding content, parsing state
- Suggestions: Actionable guidance for resolving the error
- Related Documentation: Links to format specifications or examples
```

#### Error Recovery and Continuation

The error handling system implements **intelligent recovery** that allows parsing to continue after encountering errors, enabling the collection of multiple error reports in a single parsing attempt:

**Tokenization Recovery**: When invalid characters are encountered, the tokenizer inserts `INVALID` tokens and attempts to resynchronize at the next recognizable token boundary. This allows parsing to continue and potentially identify additional errors downstream.

**Parsing Recovery**: When syntax errors occur, parsers attempt to resynchronize at stable parsing boundaries such as section headers, top-level keys, or significant structural markers. The recovery strategy varies by format:

- **INI Recovery**: Resynchronize at section headers `[section]` or line boundaries for key-value pairs
- **TOML Recovery**: Resynchronize at table declarations, top-level keys, or array-of-tables markers  
- **YAML Recovery**: Resynchronize at consistent indentation levels or document boundaries

**Structure Recovery**: When structural errors occur (like conflicting table definitions in TOML), the parser records the conflict but continues processing other parts of the file, allowing detection of multiple structural issues.

> **Design Insight**: Error recovery is particularly valuable during configuration development and debugging, where users benefit from seeing all issues at once rather than fixing them one at a time through multiple parsing attempts.

#### User-Friendly Error Formatting

The final stage of error information flow transforms technical error details into user-friendly diagnostic messages that follow established conventions for compiler and parser error reporting:

**Error Message Structure**:
1. **Problem Summary**: One-line description of what went wrong
2. **Location Information**: File, line, and column where the error occurred  
3. **Source Context**: Visual display of the problematic source with error highlighting
4. **Explanation**: Detailed description of why this is an error
5. **Suggestions**: Specific guidance on how to fix the problem
6. **Related Information**: Links to documentation or similar error examples

**Visual Error Context Example**:
```
Error: Invalid table redefinition in TOML
  → config.toml:15:1

   13 | [database]
   14 | host = "localhost"  
   15 | [database]
      | ^^^^^^^^^^
   16 | port = 5432

Explanation: Table 'database' has already been defined on line 13. TOML does not allow redefining tables once they have been created.

Suggestion: Use dotted notation [database.connection] to create subtables, or merge the key-value pairs into the existing [database] section.
```

This formatting approach provides immediate visual identification of the problem location, clear explanation of the underlying issue, and actionable guidance for resolution.

![Data Structure Transformation](./diagrams/data-structure-mapping.svg)

### Implementation Guidance

The interactions and data flow components require careful coordination between multiple parsing subsystems. This section provides the foundational infrastructure and integration patterns needed to build a cohesive parsing pipeline.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Pipeline Orchestration | Function-based pipeline with explicit error passing | Class-based pipeline with context objects and middleware |
| Format Detection | Regex-based pattern matching with scoring | Statistical analysis with machine learning features |
| Error Reporting | String-based error messages with position info | Rich error objects with suggestions and context |
| Context Management | Global variables or parameter threading | Context objects with scoped state management |
| Data Flow Coordination | Direct function calls with return values | Event-driven architecture with observer pattern |

#### Recommended Module Structure

```
config_parser/
  __init__.py                    ← Public API exports
  pipeline/
    __init__.py
    core.py                     ← Main pipeline orchestration
    context.py                  ← Pipeline context management
    errors.py                   ← Error handling infrastructure
  detection/
    __init__.py
    detector.py                 ← Format detection logic
    signatures.py              ← Format-specific signatures
    heuristics.py              ← Detection heuristics
  parsers/
    __init__.py
    ini_parser.py              ← INI format parser
    toml_parser.py             ← TOML format parser  
    yaml_parser.py             ← YAML format parser
    base.py                    ← Common parser interface
  tokenizer/
    __init__.py
    tokenizer.py               ← Main tokenization engine
    tokens.py                  ← Token definitions
  utils/
    __init__.py
    position.py                ← Position tracking utilities
    conversion.py              ← Data structure conversion
  tests/
    test_pipeline.py           ← Integration tests
    test_detection.py          ← Format detection tests
    test_errors.py             ← Error handling tests
```

#### Infrastructure Starter Code

**Pipeline Context Management** (Complete implementation):

```python
# pipeline/context.py
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from .errors import ParseError

@dataclass
class ParseContext:
    """Central context object that flows through the entire parsing pipeline."""
    
    source_content: str
    source_path: Optional[str] = None
    current_position: Optional['Position'] = None
    detected_format: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)
    errors: List[ParseError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def get_position(self) -> 'Position':
        """Get current source position, creating default if none exists."""
        if self.current_position is None:
            from ..utils.position import Position
            self.current_position = Position(line=1, column=1, offset=0)
        return self.current_position
    
    def set_position(self, pos: 'Position') -> None:
        """Update current source position."""
        self.current_position = pos
    
    def add_error(self, error: ParseError) -> None:
        """Record error with current context."""
        if error.position is None and self.current_position is not None:
            error.position = self.current_position
        self.errors.append(error)
    
    def add_warning(self, message: str) -> None:
        """Record warning message."""
        self.warnings.append(message)
    
    def has_errors(self) -> bool:
        """Check if any errors have been recorded."""
        return len(self.errors) > 0
    
    def get_errors(self) -> List[ParseError]:
        """Retrieve all accumulated errors."""
        return self.errors.copy()
    
    def get_format(self) -> Optional[str]:
        """Get detected format identifier."""
        return self.detected_format
    
    def set_format(self, fmt: str) -> None:
        """Set format for pipeline."""
        self.detected_format = fmt
    
    def get_options(self) -> Dict[str, Any]:
        """Get format-specific parsing options."""
        return self.options.copy()
    
    def set_option(self, key: str, value: Any) -> None:
        """Set format-specific parsing option."""
        self.options[key] = value

# pipeline/core.py  
from typing import Dict, Any, Optional, Union
from pathlib import Path
from .context import ParseContext
from .errors import ParseError, ConfigurationError
from ..detection.detector import detect_format
from ..parsers.ini_parser import INIParser
from ..parsers.toml_parser import TOMLParser  
from ..parsers.yaml_parser import YAMLParser

def parse_config(content: Union[str, Path], format: Optional[str] = None) -> Dict[str, Any]:
    """
    Main configuration parsing entry point.
    
    Args:
        content: Configuration content as string or file path
        format: Optional format specification ('ini', 'toml', 'yaml')
        
    Returns:
        Nested dictionary representing the configuration structure
        
    Raises:
        ConfigurationError: When parsing fails with detailed error information
    """
    # TODO 1: Handle content input - if Path, read file; if str, use directly
    # TODO 2: Create ParseContext with content and metadata
    # TODO 3: Detect format if not explicitly provided
    # TODO 4: Route to appropriate parser based on detected format
    # TODO 5: Handle parsing errors and create comprehensive error reports
    # TODO 6: Return unified dictionary structure
    
def _read_content(source: Union[str, Path]) -> tuple[str, Optional[str]]:
    """
    Read configuration content from file or use string directly.
    Returns (content, file_path) tuple.
    """
    # TODO: Implement file reading with encoding detection and normalization

def _create_parser(format_name: str) -> 'BaseParser':
    """Create appropriate parser instance based on format."""
    # TODO: Factory method for parser creation

def _handle_parsing_errors(context: ParseContext) -> None:
    """Process accumulated errors and create user-friendly reports."""
    # TODO: Transform technical errors into user guidance
```

**Error Infrastructure** (Complete implementation):

```python
# pipeline/errors.py
from typing import Optional, List
from dataclasses import dataclass

@dataclass 
class Position:
    """Source position information for error reporting."""
    line: int
    column: int  
    offset: int
    
    def __str__(self) -> str:
        return f"{self.line}:{self.column}"

class ParseError(Exception):
    """Base class for all parsing errors."""
    
    def __init__(self, message: str, position: Optional[Position] = None, 
                 suggestion: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.position = position
        self.suggestion = suggestion
    
    def __str__(self) -> str:
        result = self.message
        if self.position:
            result = f"Line {self.position}: {result}"
        if self.suggestion:
            result += f"\nSuggestion: {self.suggestion}"
        return result

class TokenError(ParseError):
    """Error during tokenization phase."""
    pass

class SyntaxError(ParseError):
    """Error during syntax analysis phase."""
    pass
    
class StructureError(ParseError):
    """Error during structure building phase."""
    pass

class ConfigurationError(Exception):
    """High-level configuration parsing failure with multiple error details."""
    
    def __init__(self, errors: List[ParseError], source_path: Optional[str] = None):
        self.errors = errors
        self.source_path = source_path
        
        # Create summary message
        if len(errors) == 1:
            message = f"Configuration parsing failed: {errors[0].message}"
        else:
            message = f"Configuration parsing failed with {len(errors)} errors"
            
        super().__init__(message)
    
    def format_detailed_report(self) -> str:
        """Generate comprehensive error report for user display."""
        # TODO: Create detailed multi-error report with source context
        pass

def create_error_context(source: str, position: Position, context_lines: int = 2) -> str:
    """Generate visual error context showing source location."""
    # TODO 1: Split source into lines
    # TODO 2: Calculate which lines to show (position +/- context_lines) 
    # TODO 3: Format with line numbers and error indicator
    # TODO 4: Return formatted context string
    pass
```

#### Core Logic Skeleton

**Format Detection Implementation**:

```python
# detection/detector.py
from typing import Dict, Tuple, List
import re
from enum import Enum

class FormatConfidence(Enum):
    HIGH = 90
    MEDIUM = 70  
    LOW = 50
    AMBIGUOUS = 25

def detect_format(content: str) -> str:
    """
    Automatically detect configuration file format from content.
    
    Args:
        content: Raw configuration file content
        
    Returns:
        Format identifier ('ini', 'toml', 'yaml')
        
    Raises:
        ValueError: When format cannot be determined with confidence
    """
    # TODO 1: Run signature scanning pass for definitive format markers
    # TODO 2: Run pattern analysis pass for format-specific characteristics  
    # TODO 3: Run conflict resolution pass for ambiguous cases
    # TODO 4: Evaluate confidence and return format or raise error
    # Hint: Use _scan_format_signatures() for each format, combine scores

def _scan_format_signatures(content: str) -> Dict[str, int]:
    """
    Scan for definitive format signatures and return confidence scores.
    Returns dict with format names as keys, confidence scores as values.
    """
    # TODO 1: Define regex patterns for TOML signatures ([[tables]], dotted.keys, etc.)
    # TODO 2: Define regex patterns for INI signatures ([sections], key=value, etc.) 
    # TODO 3: Define regex patterns for YAML signatures (key:, -, indentation, etc.)
    # TODO 4: Count matches for each format and calculate confidence scores
    # TODO 5: Return scores dict for conflict resolution

def _analyze_format_patterns(content: str) -> Dict[str, int]:
    """
    Analyze statistical patterns for format identification.
    Returns confidence adjustments based on pattern frequency.
    """
    # TODO: Implement pattern frequency analysis

def _resolve_format_conflicts(scores: Dict[str, int]) -> Tuple[str, int]:
    """
    Apply tie-breaking rules when multiple formats have similar scores.
    Returns (format_name, final_confidence_score).
    """
    # TODO: Implement conflict resolution logic
```

**Pipeline Integration Points**:

```python
# parsers/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any
from ..pipeline.context import ParseContext

class BaseParser(ABC):
    """Base interface for all format-specific parsers."""
    
    def __init__(self, context: ParseContext):
        self.context = context
    
    @abstractmethod
    def parse(self, content: str) -> Dict[str, Any]:
        """Parse content and return unified dictionary structure."""
        pass
    
    @abstractmethod  
    def get_format_name(self) -> str:
        """Return format identifier for this parser."""
        pass
    
    def add_error(self, error: 'ParseError') -> None:
        """Add error to parsing context."""
        self.context.add_error(error)
    
    def add_warning(self, message: str) -> None:
        """Add warning to parsing context."""
        self.context.add_warning(message)
```

#### Language-Specific Hints

**Python Implementation Notes**:
- Use `pathlib.Path` for file handling to maintain cross-platform compatibility
- Implement error context generation using `str.splitlines()` and list slicing for performance
- Use `dataclasses` for structured error objects to reduce boilerplate
- Consider using `typing.Union` for flexible input types (string content vs file paths)
- Use `enum.Enum` for format identifiers and confidence levels to prevent typos

**Error Handling Patterns**:
- Use context managers for file operations to ensure proper cleanup
- Implement error accumulation using lists rather than raising immediately 
- Use custom exception hierarchies to enable targeted error handling
- Include position information in all error objects for debugging

#### Milestone Checkpoint

After implementing the interactions and data flow components, verify the integration:

**Integration Test Command**:
```bash
python -m pytest tests/test_pipeline.py -v
```

**Expected Behavior**:
1. **Format Detection**: Create test files with clear format signatures. The detector should identify each format with high confidence.

2. **Pipeline Flow**: Test the complete pipeline with valid files of each format. All should produce equivalent nested dictionary outputs.

3. **Error Propagation**: Test with deliberately broken configuration files. Errors should include source positions, context, and helpful suggestions.

4. **Multi-Error Collection**: Test with files containing multiple errors. The parser should report all errors found, not just the first one.

**Manual Verification Steps**:
1. Create a simple test script that calls `parse_config()` with sample files
2. Verify that INI, TOML, and YAML files with equivalent content produce identical dictionary outputs
3. Test error reporting by introducing syntax errors - verify that error messages include line numbers and suggestions
4. Test format detection by removing format hints - verify correct automatic detection

**Success Indicators**:
- All three parsers integrate cleanly through the common pipeline
- Error messages include visual source context with line numbers
- Format detection works reliably on real configuration files
- Pipeline handles both file paths and string content seamlessly

#### Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Format detection always returns INI | Detection patterns too restrictive | Print intermediate scores from each format scanner | Adjust pattern weights and add more signature patterns |
| Errors missing source positions | Position not propagated through pipeline | Add logging to track position updates | Ensure position is set in context before creating errors |
| Pipeline crashes on malformed files | Missing error recovery in component | Add try/catch around each pipeline phase | Implement graceful error handling with continuation |
| Inconsistent output between formats | Structure conversion differences | Compare parse trees before conversion | Standardize conversion logic in base parser class |
| Memory usage grows with file size | Error context keeping full source | Profile memory usage during parsing | Limit error context to relevant lines only |


## Error Handling and Edge Cases

> **Milestone(s):** All milestones (INI Parser, TOML Tokenizer, TOML Parser, YAML Subset Parser) - comprehensive error detection, reporting, and recovery strategies across all formats

Configuration file parsing operates in an inherently error-prone environment where malformed input, ambiguous syntax, and edge cases are common. Think of error handling in parsing as building a safety net with multiple layers—each component must detect problems at its level, enrich the error information with context, and decide whether to recover gracefully or halt processing. Unlike typical application errors that occur during runtime, parsing errors must provide educational feedback to users who are editing configuration files, often by hand, and need specific guidance about what went wrong and how to fix it.

The complexity of error handling in our multi-format parser stems from the fundamental differences in how INI, TOML, and YAML express structure and meaning. Each format has distinct failure modes, from INI's simple line-based issues to TOML's complex table redefinition conflicts to YAML's indentation-driven structural problems. Our error handling system must provide format-specific diagnostics while maintaining consistent error reporting patterns across all parsers.

![Complete Parsing Pipeline Flow](./diagrams/parsing-pipeline.svg)

### Error Classification

Parsing errors fall into distinct categories based on where they occur in the processing pipeline and what type of problem they represent. Understanding these categories helps design appropriate detection strategies and recovery mechanisms for each error type.

**Lexical Errors (Token-Level)**

Lexical errors occur during tokenization when the character stream cannot be converted into valid tokens. These represent the most fundamental parsing failures, where the input violates basic format syntax rules at the character level. Think of lexical errors as problems a human would catch when reading character by character—invalid escape sequences, unterminated strings, or illegal characters in specific contexts.

| Error Type | Description | Common Causes | Detection Strategy |
|------------|-------------|---------------|-------------------|
| Unterminated String | String literal missing closing quote | Multiline strings without proper delimiters, escaped quotes at string end | Track quote nesting depth, validate at line/file boundaries |
| Invalid Escape Sequence | Backslash followed by invalid character | Incorrect escape codes like `\q` instead of `\n` | Pattern matching during string tokenization |
| Illegal Character | Character not allowed in current context | Unicode characters in identifiers, special chars in numbers | Context-aware character validation |
| Malformed Number | Invalid numeric literal format | Multiple decimal points, invalid scientific notation | Regex validation and numeric conversion attempts |
| Invalid Unicode | Malformed UTF-8 sequences or invalid codepoints | File encoding issues, manual hex editing | UTF-8 validation during character reading |

The tokenizer must detect these errors immediately when encountered because they prevent meaningful token creation. However, the challenge lies in providing helpful error messages that explain not just what character is invalid, but why it's invalid in the current parsing context.

**Syntactic Errors (Grammar-Level)**

Syntactic errors occur when tokens are valid individually but don't follow the grammar rules of the target format. These errors represent violations of format-specific structure rules—like missing equals signs in key-value pairs or incorrect bracket matching in arrays. Think of syntactic errors as problems a human would catch when reading for grammatical structure rather than individual words.

| Error Type | Description | Detection Method | Recovery Options |
|------------|-------------|------------------|------------------|
| Missing Assignment Operator | Key without equals or colon separator | Parser expects EQUALS or COLON after key token | Skip line, assume section header, insert default operator |
| Unmatched Brackets | Opening bracket without corresponding close | Bracket stack tracking during parsing | Insert missing bracket, truncate at boundary |
| Invalid Key Format | Key contains illegal characters or structure | Key validation against format rules | Sanitize key, reject entry, quote if possible |
| Malformed Table Header | Section header syntax violations | Bracket counting, path validation | Skip header, use previous section, create default |
| Invalid Array Structure | Mixed types or malformed array syntax | Type consistency checking during array parsing | Convert to strings, truncate at error, skip malformed elements |

Syntactic errors are particularly challenging because the parser has successfully tokenized the input but cannot interpret the token sequence according to format rules. The decision of whether to recover or abort depends on how fundamental the syntactic violation is to the document structure.

**Structural Errors (Semantic-Level)**

Structural errors occur when syntax is correct but the resulting data structure violates format-specific semantic rules. These are the most complex errors to detect because they require understanding the global document structure and format-specific constraints. Think of structural errors as problems that become apparent only when viewing the entire document—like trying to redefine a table that already exists or creating circular references in nested structures.

| Error Type | Format | Description | Detection Strategy |
|------------|--------|-------------|-------------------|
| Table Redefinition | TOML | Attempting to redefine existing table | Symbol table tracking with conflict detection |
| Key Path Conflict | TOML | Dotted key conflicts with existing table | Path resolution with type checking |
| Indentation Inconsistency | YAML | Mixed indentation levels or tab/space mixing | Indentation tracking with established level validation |
| Circular Structure | All | Self-referencing nested structures | Depth tracking and path cycle detection |
| Type Inconsistency | TOML/YAML | Array elements with incompatible types | Type inference tracking during array construction |
| Invalid Section Nesting | INI | Sections that create impossible hierarchies | Section path validation during nested structure creation |

Structural errors require maintaining parsing state across the entire document and applying format-specific validation rules as the data structure is constructed. These errors often indicate deeper misunderstandings of format semantics rather than simple syntax mistakes.

> **Design Insight**: The error classification hierarchy mirrors the parsing pipeline—lexical errors block tokenization, syntactic errors block parse tree construction, and structural errors block final data structure creation. Each layer must decide whether to recover locally or propagate the error upward with additional context.

**Context-Dependent Errors**

Some errors can only be detected by examining the relationship between different parts of the document or by understanding the broader parsing context. These errors represent violations of format-specific rules that span multiple lines or sections.

| Error Category | Description | Examples | Detection Approach |
|----------------|-------------|----------|-------------------|
| Cross-Reference Violations | References to undefined or invalid targets | TOML array-of-tables referencing undefined tables | Symbol table validation after parsing completion |
| Scope Violations | Content appearing in invalid contexts | Global INI keys after section definitions | Context stack validation during parsing |
| Ordering Violations | Content appearing in wrong sequence | YAML mapping keys out of alphabetical order (when required) | Sequence tracking with format-specific rules |
| Dependency Violations | Missing prerequisites for current content | Nested structure without parent definition | Dependency graph validation |

> **Decision: Error Classification Granularity**
> - **Context**: Errors can be classified at different levels of detail, from broad categories to specific error codes
> - **Options Considered**: Simple three-tier system (lexical/syntactic/structural), detailed error codes for each format, hybrid approach with categories and subcodes
> - **Decision**: Hierarchical classification with main categories and format-specific subcategories
> - **Rationale**: Allows consistent error handling patterns while preserving format-specific diagnostic information
> - **Consequences**: Enables both generic error recovery strategies and specialized format-specific handling

### Error Message Design

Effective error messages in configuration parsing must bridge the gap between technical parsing details and user-friendly guidance. The target audience includes both developers integrating configuration files and system administrators editing configuration by hand. Think of error messages as being written by an experienced colleague who understands both the technical requirements and the user's likely mental model of the configuration format.

**Error Message Components**

Every parsing error message should contain specific components that collectively provide enough information for the user to understand what went wrong and how to fix it. The challenge lies in presenting this information clearly without overwhelming the user with implementation details.

| Component | Purpose | Example Content | Design Guidelines |
|-----------|---------|-----------------|-------------------|
| Problem Summary | Clear statement of what went wrong | "Unterminated string literal" | Use domain language, avoid parser internals |
| Location Context | Where the error occurred | "Line 23, column 15" | Provide both line/column and visual context |
| Syntax Context | Surrounding code for visual reference | Show 2-3 lines around error position | Highlight exact error position with markers |
| Explanation | Why this is considered an error | "String literals must end with matching quote" | Reference format specification rules |
| Suggestion | Specific guidance for fixing the problem | "Add closing quote or use multiline string syntax" | Provide actionable steps, not vague advice |
| Related Information | Links to relevant documentation or similar errors | "See TOML string specification section 4.2" | Help users understand broader context |

The `create_error_context` function standardizes the visual presentation of error location by showing source lines with position markers and highlighting the specific character or token where the error occurred.

**Format-Specific Error Messaging**

Each configuration format has characteristic error patterns that require specialized messaging approaches. The error message must reflect the user's mental model of how the format works, not the internal parsing implementation.

**INI Format Error Messages**

INI errors typically involve line-based parsing issues that are relatively straightforward to diagnose and fix. The mental model for INI users is simple: sections contain key-value pairs, and comments are ignored. Error messages should reinforce this simplicity while providing specific guidance.

| Error Scenario | Technical Issue | User-Friendly Message | Recovery Guidance |
|-----------------|-----------------|----------------------|-------------------|
| Missing section header | Global key without section context | "Key-value pair 'database.host' appears before any section header" | "Move this line inside a [section] or add a [DEFAULT] section above it" |
| Malformed assignment | Line without equals or colon | "Line 'database host localhost' is not recognized as section header or key-value pair" | "Add '=' or ':' between key and value: 'database.host = localhost'" |
| Invalid section name | Section header with illegal characters | "Section header contains invalid characters: '[data<>base]'" | "Remove special characters: '[database]' or quote if necessary" |
| Inline comment confusion | Equals sign within quoted value | "Assignment appears to contain multiple '=' characters" | "Quote the entire value: 'url = \"http://example.com?id=123\"'" |

**TOML Format Error Messages**

TOML errors are often complex because of the format's rich type system and table semantics. Users frequently struggle with table redefinition rules and dotted key expansion. Error messages must explain not just what's wrong but how TOML's global namespace works.

| Error Scenario | Technical Issue | User-Friendly Message | Explanation |
|-----------------|-----------------|----------------------|-------------|
| Table redefinition | Explicit table conflicts with previous definition | "Cannot redefine table [database] (previously defined at line 15)" | "TOML tables can only be defined once. Use dotted keys to add more values or create a [database.connection] subtable" |
| Dotted key conflict | Dotted key path conflicts with existing table | "Key 'database.host' conflicts with table [database.host] at line 8" | "A key path cannot contain both a value and a subtable. Choose either 'database.host = \"value\"' or '[database.host]' with sub-keys" |
| Array-of-tables confusion | Mixed array and table syntax | "Cannot mix array-of-tables [[servers]] with regular table [servers]" | "Use either [[servers]] for multiple server entries or [servers] for a single server configuration, not both" |
| Type inconsistency | Array contains mixed types | "Array contains both integer (5) and string (\"five\") values" | "TOML arrays must contain values of the same type. Use strings for all values or create separate arrays" |

**YAML Format Error Messages**

YAML errors frequently involve indentation and implicit structure creation. Users often struggle with YAML's context-sensitive parsing where the same content can mean different things depending on indentation and surrounding structure.

| Error Scenario | Technical Issue | User-Friendly Message | Indentation Guidance |
|-----------------|-----------------|----------------------|----------------------|
| Indentation inconsistency | Current line doesn't match established levels | "Indentation of 3 spaces doesn't match any previous level (expected 0, 2, or 4)" | "Use consistent indentation increments. Choose either 2 or 4 spaces per level and stick with it" |
| Mixed tabs and spaces | Line contains both tab and space characters | "Line contains both tabs and spaces for indentation" | "YAML forbids mixing tabs and spaces. Use only spaces for indentation" |
| Structural ambiguity | Content could be interpreted multiple ways | "Mapping key 'items' appears at same level as sequence item" | "Indent the mapping key further to make it part of the sequence item, or outdent to make it a sibling" |
| Type inference surprise | Value converted to unexpected type | "Value 'yes' was interpreted as boolean true, not string" | "Quote string values that might be interpreted as other types: items: \"yes\"" |

> **Decision: Error Message Verbosity**
> - **Context**: Error messages can range from terse technical descriptions to verbose explanations with examples
> - **Options Considered**: Brief messages with error codes, verbose messages with full explanations, configurable verbosity levels
> - **Decision**: Verbose messages by default with option to reduce verbosity
> - **Rationale**: Configuration files are often edited manually by users who need educational feedback, not just error identification
> - **Consequences**: Larger error output but significantly improved user experience for configuration debugging

**Error Context Generation**

The `create_error_context` function generates visual representations of error locations that help users quickly identify and fix problems. This function must handle various edge cases while providing consistent output formatting.

| Context Scenario | Challenge | Solution Approach |
|------------------|-----------|-------------------|
| Error at line beginning | No preceding content for context | Show line with position marker at start |
| Error at line end | May be continuation or termination issue | Show line with marker and indicate if more content expected |
| Error in long line | Line too wide for terminal display | Truncate line intelligently, keeping error position visible |
| Error at file boundary | Beginning or end of file | Show available context, indicate file boundary |
| Multi-line error | Error spans multiple lines | Show all affected lines with range indicators |

The visual error context uses consistent formatting conventions: line numbers in brackets, position markers with carets or arrows, and highlighting of the specific characters involved in the error. This standardization helps users quickly parse error output across different error types.

![Error Detection and Reporting Flow](./diagrams/error-handling-flow.svg)

### Error Recovery Approaches

Error recovery in parsing determines whether the system continues processing after encountering errors, and if so, how it attempts to interpret subsequent input. The fundamental tension in error recovery is between providing comprehensive error reporting (finding multiple problems in one pass) and maintaining parsing accuracy (not generating spurious errors due to incorrect recovery assumptions).

**Recovery Strategy Classification**

Different types of parsing errors require different recovery strategies based on their scope and impact on subsequent parsing. The recovery approach must consider both the likelihood of successful continued parsing and the value of finding additional errors in the same document.

**Panic Mode Recovery**

Panic mode recovery involves discarding input tokens until reaching a known synchronization point where parsing can resume reliably. This approach works well for format-specific landmarks that clearly indicate structure boundaries.

| Format | Synchronization Points | Recovery Strategy | Reliability |
|--------|------------------------|-------------------|-------------|
| INI | Section headers, blank lines | Skip to next `[section]` or end of current section | High - sections are independent |
| TOML | Table headers, top-level assignments | Skip to next `[table]` or unindented key assignment | Medium - may skip related content |
| YAML | Document separators, unindented lines | Skip to next document `---` or zero-indentation content | Low - indentation context is critical |

Panic mode recovery is most effective when the error occurs in a self-contained syntactic unit that can be safely skipped without affecting the interpretation of subsequent content. The challenge lies in determining which synchronization points are truly safe versus which might lead to misinterpretation of later content.

**Error Production Recovery**

Error production recovery involves inserting assumed content (like missing punctuation) or making reasonable assumptions about user intent to continue parsing. This approach attempts to "fix" the input automatically while recording what assumptions were made.

| Error Type | Recovery Action | Assumption Made | Risk Level |
|------------|-----------------|-----------------|------------|
| Missing assignment operator | Insert `=` between key and value | User intended key-value pair | Low |
| Missing closing quote | Insert quote at line end | String was intended to be single-line | Medium |
| Missing closing bracket | Insert bracket at section end | User forgot to close array/table | High |
| Inconsistent indentation | Assume closest matching level | User made spacing mistake | High |

Error production recovery requires careful consideration of how likely the assumed fix is to match user intent. Conservative recovery (making minimal assumptions) is generally safer than aggressive recovery (making complex structural assumptions).

**Phrase-Level Recovery**

Phrase-level recovery attempts to identify the boundaries of malformed constructs and skip only the minimal amount of content necessary to resume parsing. This approach requires understanding the syntactic structure of the format to identify phrase boundaries.

| Format Construct | Boundary Indicators | Recovery Scope | Success Indicators |
|------------------|-------------------|----------------|-------------------|
| INI key-value pair | Line boundaries, section headers | Single line or continuation sequence | Next line parses as valid key-value or section |
| TOML value expression | Commas, newlines, closing brackets | Value portion of assignment | Assignment key is preserved, next token is valid |
| YAML mapping entry | Indentation changes, sequence markers | Single key-value pair | Indentation context remains consistent |
| YAML sequence item | Sequence markers, indentation outdent | Single list item | List structure is maintained |

The effectiveness of phrase-level recovery depends on the format's syntactic regularity. Formats with clear delimiters and boundaries support more reliable phrase-level recovery than formats with context-dependent parsing rules.

**Intelligent Recovery Strategies**

Advanced recovery strategies use knowledge about common error patterns and format-specific semantics to make informed decisions about how to continue parsing after errors.

**Format-Specific Recovery Patterns**

Each configuration format has characteristic error patterns that suggest specific recovery approaches based on observed user behavior and format complexity.

| Format | Common Error Pattern | Intelligent Recovery | Rationale |
|--------|---------------------|---------------------|-----------|
| INI | Quoted value with escaped quotes | Scan for next unescaped quote or line boundary | Nested quotes are often user intent, not syntax errors |
| TOML | Array with trailing comma | Accept trailing comma, continue parsing | Trailing commas are common in other languages |
| YAML | Inconsistent indentation increment | Calculate greatest common divisor of observed indentations | Users often pick inconsistent but mathematically related indentations |
| All | Unicode encoding issues | Attempt multiple encoding interpretations | Files are often edited with different tools that handle encoding differently |

Intelligent recovery leverages understanding of how users actually create configuration files, including common mistakes and patterns from other similar formats or programming languages.

**Multi-Pass Recovery Analysis**

For complex errors that affect document structure, a multi-pass approach can provide better recovery than single-pass strategies. The first pass identifies structural issues, and subsequent passes attempt parsing with different recovery assumptions.

| Pass Number | Analysis Focus | Recovery Goals | Success Criteria |
|-------------|----------------|----------------|------------------|
| Pass 1 | Strict parsing with no recovery | Identify all definite errors | Clear error classification and location |
| Pass 2 | Conservative recovery assumptions | Parse maximum safe content | No spurious errors introduced |
| Pass 3 | Aggressive recovery with user feedback | Extract any possible valid content | User can validate recovery assumptions |

Multi-pass analysis is particularly valuable for YAML documents where indentation errors can cascade and affect the interpretation of large portions of the document.

> **Decision: Recovery Aggressiveness**
> - **Context**: Recovery strategies range from conservative (minimal assumptions) to aggressive (maximal content extraction)
> - **Options Considered**: Always halt on first error, conservative recovery with user confirmation, aggressive recovery with detailed assumption logging
> - **Decision**: Conservative recovery by default with option for aggressive recovery mode
> - **Rationale**: Configuration files often contain critical system settings where incorrect assumptions can cause operational problems
> - **Consequences**: May require multiple parse attempts to extract maximum content, but reduces risk of silent misinterpretation

**Graceful Degradation Patterns**

When errors prevent complete parsing of a configuration document, graceful degradation strategies attempt to extract partial information that might still be valuable to the application.

| Degradation Level | Content Extracted | Application Guidance | Risk Assessment |
|-------------------|-------------------|---------------------|------------------|
| Section-Level | Complete valid sections only | Use partial configuration with defaults for missing sections | Low - missing sections are explicit |
| Key-Level | Valid key-value pairs within sections | Skip malformed keys, preserve valid ones | Medium - key relationships might be broken |
| Value-Level | Keys with parseable values | Use string fallback for unparseable values | High - type mismatches can cause runtime errors |

Graceful degradation must provide clear information to the calling application about what content was successfully parsed and what assumptions or defaults are being applied for missing or malformed content.

**Error Recovery State Management**

Effective error recovery requires maintaining additional parsing state to track recovery decisions and their impact on subsequent parsing. This state helps determine when recovery assumptions prove incorrect and parsing should be abandoned.

| State Information | Purpose | Update Triggers | Decision Points |
|------------------|---------|-----------------|-----------------|
| Recovery assumption log | Track what fixes were assumed | Each recovery action | Final validation of assumptions |
| Confidence level tracking | Measure parsing reliability | Each successful/failed parse decision | Threshold for abandoning recovery |
| Structural integrity markers | Validate document structure | Major structure completion | Consistency checking of recovered content |
| Alternative interpretation stack | Track parsing alternatives | Ambiguous syntax encounters | Backtracking to alternative interpretations |

This state management enables the parser to make informed decisions about when recovery is likely to succeed versus when the accumulated uncertainty makes continued parsing unreliable.

> **Common Pitfalls in Error Recovery**
> 
> ⚠️ **Pitfall: Over-Aggressive Recovery**
> Recovery logic that makes too many assumptions about user intent can silently convert malformed configuration into incorrect but syntactically valid results. This is particularly dangerous in configuration parsing where silent errors can cause production system failures.
> 
> ⚠️ **Pitfall: Recovery Error Cascades**
> Incorrect recovery from one error can cause subsequent parsing to misinterpret valid content as erroneous. Each recovery decision must be validated against following content to detect when recovery assumptions are incorrect.
> 
> ⚠️ **Pitfall: Context Loss During Recovery**
> Recovery strategies that discard too much parsing context (like jumping to the next section) can lose important structural information needed to correctly interpret subsequent content. Context preservation is critical for maintaining parsing accuracy.
> 
> ⚠️ **Pitfall: Inconsistent Recovery Behavior**
> Similar errors in different parts of the document should trigger consistent recovery behavior. Inconsistent recovery can confuse users and make error patterns harder to recognize and fix systematically.

### Implementation Guidance

The error handling implementation requires careful coordination between all parsing components to ensure consistent error detection, enriched propagation, and appropriate recovery strategies. The following guidance provides concrete implementations for the core error handling infrastructure.

**Technology Recommendations**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Error Types | Simple inheritance with base `ParseError` class | Rich error taxonomy with error codes and metadata |
| Error Context | String formatting with manual line extraction | Template-based error messages with structured context |
| Error Recovery | Fixed recovery strategies per error type | Configurable recovery policies with success tracking |
| Error Reporting | Direct exception raising with message | Structured error collection with severity levels |
| Position Tracking | Line/column counters during parsing | Full source mapping with character ranges |

**Recommended File Structure**

```python
config_parser/
├── errors/
│   ├── __init__.py              # Error type exports
│   ├── base_errors.py           # ParseError, TokenError, SyntaxError, StructureError
│   ├── error_context.py         # create_error_context, position tracking
│   ├── error_recovery.py        # Recovery strategies and state management
│   └── format_specific.py       # INI, TOML, YAML specific error types
├── parsers/
│   ├── base_parser.py           # BaseParser with error handling integration
│   ├── ini_parser.py            # INI parser with error recovery
│   ├── toml_parser.py           # TOML parser with error recovery
│   └── yaml_parser.py           # YAML parser with error recovery
└── tests/
    ├── test_error_handling.py   # Error detection and recovery tests
    └── error_test_cases/         # Test configuration files with various errors
```

**Core Error Infrastructure**

```python
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

@dataclass
class Position:
    """Represents a position in the source text with line, column, and offset information."""
    line: int
    column: int
    offset: int
    
    def __str__(self) -> str:
        return f"line {self.line}, column {self.column}"

class ParseError(Exception):
    """Base class for all parsing errors with rich context information."""
    
    def __init__(self, message: str, position: Optional[Position] = None, 
                 suggestion: Optional[str] = None):
        self.message = message
        self.position = position
        self.suggestion = suggestion
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        """Format error message with position and suggestion information."""
        # TODO: Create user-friendly error message combining message, position, and suggestion
        # TODO: Include position information in human-readable format
        # TODO: Append suggestion as actionable guidance when available
        # TODO: Use consistent formatting for all error types
        pass

class TokenError(ParseError):
    """Errors that occur during tokenization (lexical analysis)."""
    
    def __init__(self, message: str, position: Optional[Position] = None, 
                 invalid_text: str = "", suggestion: Optional[str] = None):
        self.invalid_text = invalid_text
        super().__init__(message, position, suggestion)

class SyntaxError(ParseError):
    """Errors that occur when tokens don't follow format grammar rules."""
    
    def __init__(self, message: str, position: Optional[Position] = None,
                 expected_tokens: List[str] = None, actual_token: str = "",
                 suggestion: Optional[str] = None):
        self.expected_tokens = expected_tokens or []
        self.actual_token = actual_token
        super().__init__(message, position, suggestion)

class StructureError(ParseError):
    """Errors that occur when document structure violates format semantics."""
    
    def __init__(self, message: str, position: Optional[Position] = None,
                 conflicting_position: Optional[Position] = None,
                 suggestion: Optional[str] = None):
        self.conflicting_position = conflicting_position
        super().__init__(message, position, suggestion)

def create_error_context(source_content: str, position: Position, 
                        context_lines: int = 2) -> str:
    """Generate visual error context showing source lines around error position."""
    # TODO: Split source content into lines and identify target line
    # TODO: Calculate start and end line numbers for context window
    # TODO: Format line numbers with consistent width and padding
    # TODO: Add position marker (caret or arrow) pointing to exact error location
    # TODO: Handle edge cases: file start/end, very long lines, empty lines
    # TODO: Return formatted string with line numbers, content, and position marker
    pass
```

**Error Recovery Infrastructure**

```python
from enum import Enum
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass, field

class RecoveryStrategy(Enum):
    """Available error recovery strategies."""
    HALT = "halt"                    # Stop parsing on first error
    PANIC_MODE = "panic_mode"        # Skip to synchronization point
    ERROR_PRODUCTION = "error_production"  # Insert assumed content
    PHRASE_LEVEL = "phrase_level"    # Skip minimal syntactic unit

@dataclass
class RecoveryDecision:
    """Represents a recovery decision made during parsing."""
    strategy: RecoveryStrategy
    assumption: str                  # What assumption was made
    confidence: float                # Confidence in recovery (0.0-1.0)
    tokens_skipped: int             # Number of tokens skipped
    content_inserted: str           # Any content inserted by recovery

@dataclass
class ErrorRecoveryState:
    """Tracks error recovery state during parsing."""
    decisions: List[RecoveryDecision] = field(default_factory=list)
    confidence_threshold: float = 0.7
    max_consecutive_recoveries: int = 5
    consecutive_recovery_count: int = 0
    
    def should_continue_recovery(self) -> bool:
        """Determine if parsing should continue after current error."""
        # TODO: Check if consecutive recovery count exceeds maximum
        # TODO: Calculate average confidence of recent recovery decisions
        # TODO: Return False if confidence drops below threshold
        # TODO: Consider total number of recovery attempts in document
        pass
    
    def record_recovery(self, decision: RecoveryDecision) -> None:
        """Record a recovery decision and update state."""
        # TODO: Append decision to decisions list
        # TODO: Update consecutive recovery count
        # TODO: Adjust confidence thresholds based on recovery success patterns
        # TODO: Reset consecutive count on successful parsing between errors
        pass

class ErrorRecoveryManager:
    """Manages error recovery strategies and decisions."""
    
    def __init__(self):
        self.recovery_strategies: Dict[type, Callable] = {}
        self.synchronization_points = {
            'ini': [r'\[.*\]', r'^\s*$'],           # Section headers, blank lines
            'toml': [r'\[.*\]', r'^[a-zA-Z].*='],   # Table headers, top-level keys
            'yaml': [r'^---', r'^[^\s]']            # Document separators, unindented content
        }
    
    def register_recovery_strategy(self, error_type: type, 
                                  strategy_func: Callable) -> None:
        """Register a recovery strategy for specific error type."""
        # TODO: Store strategy function for error type
        # TODO: Validate that strategy function has correct signature
        # TODO: Allow multiple strategies per error type with priority ordering
        pass
    
    def attempt_recovery(self, error: ParseError, parser_state: Dict[str, Any],
                        format_name: str) -> Optional[RecoveryDecision]:
        """Attempt to recover from parsing error using appropriate strategy."""
        # TODO: Determine recovery strategy based on error type and parser state
        # TODO: Check if recovery is advisable based on current recovery state
        # TODO: Execute recovery strategy and capture decision information
        # TODO: Validate recovery assumptions against subsequent content when possible
        # TODO: Return RecoveryDecision with strategy details and confidence level
        pass
```

**Format-Specific Error Handling**

```python
class INIParsingError(SyntaxError):
    """Specialized error for INI format parsing issues."""
    
    def __init__(self, message: str, line_number: int, line_content: str,
                 suggestion: Optional[str] = None):
        position = Position(line=line_number, column=1, offset=0)
        self.line_content = line_content
        super().__init__(message, position, suggestion=suggestion)

class TOMLTableRedefinitionError(StructureError):
    """Error for TOML table redefinition conflicts."""
    
    def __init__(self, table_path: List[str], original_position: Position,
                 redefinition_position: Position):
        table_name = ".".join(table_path)
        message = f"Cannot redefine table [{table_name}]"
        suggestion = f"Use dotted keys to add values or create subtable like [{table_name}.subtable]"
        super().__init__(message, redefinition_position, original_position, suggestion)

class YAMLIndentationError(StructureError):
    """Error for YAML indentation inconsistencies."""
    
    def __init__(self, current_indent: int, expected_indents: List[int],
                 position: Position):
        self.current_indent = current_indent
        self.expected_indents = expected_indents
        message = f"Indentation of {current_indent} spaces doesn't match established levels"
        suggestion = f"Use one of these indentation levels: {expected_indents}"
        super().__init__(message, position, suggestion=suggestion)

def create_format_specific_error(format_name: str, error_type: str,
                               context: Dict[str, Any]) -> ParseError:
    """Factory function for creating format-specific error instances."""
    # TODO: Dispatch to appropriate error class based on format and error type
    # TODO: Extract relevant context information for error class constructor
    # TODO: Generate format-appropriate error message and suggestion
    # TODO: Return properly constructed error instance with full context
    pass
```

**Error Collection and Reporting**

```python
@dataclass
class ConfigurationError:
    """Top-level error containing all parsing errors for a configuration file."""
    errors: List[ParseError]
    source_path: Optional[str] = None
    recovery_decisions: List[RecoveryDecision] = field(default_factory=list)
    
    def has_fatal_errors(self) -> bool:
        """Check if errors prevent using any configuration content."""
        # TODO: Classify errors by severity (fatal vs recoverable)
        # TODO: Return True if any StructureError or unrecovered TokenError present
        # TODO: Consider recovery decision confidence in fatal error determination
        pass
    
    def format_error_report(self, include_suggestions: bool = True,
                          include_context: bool = True) -> str:
        """Generate comprehensive error report for user consumption."""
        # TODO: Group related errors together (same line, same construct)
        # TODO: Sort errors by position (line number, then column)
        # TODO: Format each error with context and suggestions when requested
        # TODO: Include recovery decision summary if any recoveries were attempted
        # TODO: Provide document-level guidance for common error patterns
        pass

class ErrorCollector:
    """Collects and manages errors during parsing process."""
    
    def __init__(self, source_path: Optional[str] = None):
        self.errors: List[ParseError] = []
        self.source_path = source_path
        self.recovery_decisions: List[RecoveryDecision] = []
    
    def add_error(self, error: ParseError) -> None:
        """Add error to collection with automatic position tracking."""
        # TODO: Append error to errors list
        # TODO: Ensure error has position information when available
        # TODO: Check for duplicate errors at same position
        # TODO: Maintain errors in position order for consistent reporting
        pass
    
    def has_errors(self) -> bool:
        """Check if any errors have been collected."""
        return len(self.errors) > 0
    
    def create_configuration_error(self) -> ConfigurationError:
        """Create final ConfigurationError with all collected information."""
        # TODO: Return ConfigurationError with current errors and recovery decisions
        # TODO: Include source path information when available
        # TODO: Sort errors by position for consistent presentation
        pass
```

**Milestone Checkpoints**

**Milestone 1 - INI Error Handling**: Test error detection for malformed INI files:
```bash
# Test with malformed INI file containing various error types
python -m pytest tests/test_ini_error_handling.py -v
# Verify error messages are helpful and include suggestions
# Check that recovery allows parsing of valid sections despite errors
```

**Milestone 2-3 - TOML Error Handling**: Test complex TOML error scenarios:
```bash
# Test TOML table redefinition and dotted key conflicts
python -m pytest tests/test_toml_error_handling.py -v
# Verify symbol table conflict detection works correctly
# Check that error messages explain TOML's global namespace rules
```

**Milestone 4 - YAML Error Handling**: Test indentation and structure errors:
```bash
# Test YAML indentation consistency and type inference
python -m pytest tests/test_yaml_error_handling.py -v
# Verify indentation error messages suggest specific fixes
# Check that recovery maintains structural integrity
```

**Debugging Tips**

| Symptom | Likely Cause | Diagnosis Method | Fix |
|---------|--------------|------------------|-----|
| Error messages without position info | Position not tracked during parsing | Add logging to position tracking functions | Ensure every token creation updates position |
| Generic error messages | Error context not enriched during propagation | Check error creation calls for context info | Pass parser state to error constructors |
| Recovery causes spurious errors | Recovery assumptions don't match actual content | Log recovery decisions and validate against following tokens | Add recovery validation logic |
| Same error reported multiple times | Error detection in multiple parsing phases | Check error deduplication in error collector | Filter duplicate errors by position and type |
| Confusing error suggestions | Suggestions don't match actual error context | Review format-specific error message templates | Improve suggestion logic with context awareness |


## Testing Strategy

> **Milestone(s):** All milestones (INI Parser, TOML Tokenizer, TOML Parser, YAML Subset Parser) - comprehensive testing approach ensuring correct implementation and validation across all parsing components

Testing a multi-format configuration parser requires a sophisticated approach that validates correctness across fundamentally different syntactic paradigms while ensuring each component maintains its specific behavioral guarantees. Think of testing configuration parsers like quality assurance for a universal translator - we must verify that each language (format) is correctly understood, that the translation process preserves meaning accurately, and that error conditions are handled gracefully across all supported languages. The challenge lies in designing test strategies that capture both the unique characteristics of each format and the unified behavior expected from the complete parsing system.

The testing strategy must address multiple layers of complexity: lexical analysis correctness across different tokenization approaches, syntactic parsing accuracy for nested structures, semantic validation of type inference and data conversion, and integration behavior when components work together. Each milestone introduces specific testing requirements that build upon previous foundations while introducing new edge cases and validation needs.

### Test Categories and Coverage

Configuration parser testing requires a multi-dimensional approach that validates behavior across different abstraction levels, error conditions, and format-specific requirements. The testing matrix spans from low-level tokenization correctness to high-level integration scenarios, ensuring comprehensive coverage of the parsing pipeline.

**Unit Testing Coverage** forms the foundation of our testing strategy, focusing on individual component behavior in isolation. Each component must be tested against its specific responsibilities and interface contracts. Unit tests provide fast feedback during development and enable confident refactoring by establishing behavioral baselines.

| Component | Test Focus | Key Scenarios | Coverage Requirements |
|-----------|------------|---------------|---------------------|
| `BaseTokenizer` | Token generation accuracy | String literals, escape sequences, position tracking | All token types, edge cases, malformed input |
| `INIParser` | Section parsing, key-value extraction | Global keys, nested sections, comment handling | All INI syntax variants, whitespace handling |
| `TOMLParser` | Table creation, type inference | Dotted keys, array-of-tables, inline structures | TOML specification compliance, conflict detection |
| `YAMLParser` | Indentation tracking, structure building | Block syntax, flow syntax, type conversion | Indentation edge cases, mixed content types |
| Format Detection | Format identification accuracy | Ambiguous content, mixed syntax | High confidence detection, fallback behavior |

**Integration Testing Coverage** validates component interactions and end-to-end parsing behavior. Integration tests ensure that the tokenizer-parser pipeline produces correct results and that error information flows properly between components. These tests catch impedance mismatches between components and validate the complete parsing workflow.

| Integration Scenario | Test Focus | Validation Points | Error Conditions |
|---------------------|------------|-------------------|------------------|
| File-to-Dictionary Pipeline | Complete parsing accuracy | Input file → parsed structure consistency | Malformed files, encoding issues, large files |
| Cross-Format Consistency | Equivalent configurations produce same output | Semantic equivalence across INI/TOML/YAML | Format-specific limitations, type coercion differences |
| Error Context Propagation | Error information enrichment | Position accuracy, helpful messages | Multi-level errors, recovery context |
| Format Detection Integration | Automatic format selection | Correct parser selection, confidence levels | Ambiguous content, unknown formats |

**Property-Based Testing Coverage** validates parser behavior across broad input spaces by generating test cases that explore edge conditions and invariant properties. Property-based tests are particularly valuable for parsers because they can discover subtle bugs in tokenization state machines and recursive parsing algorithms.

| Property Category | Invariant Properties | Generation Strategy | Validation Approach |
|-------------------|---------------------|-------------------|-------------------|
| Tokenization Roundtrip | `tokenize(detokenize(tokens)) == tokens` | Random token sequences | Token equality, position consistency |
| Parse Tree Structure | Valid input produces well-formed tree | Grammar-compliant generation | Structure validation, parent-child consistency |
| Type Inference Consistency | Same semantic value infers same type | Value variation generation | Type stability, conversion accuracy |
| Error Recovery Consistency | Errors don't corrupt subsequent parsing | Malformed input injection | State isolation, recovery effectiveness |

**Regression Testing Coverage** ensures that fixes and enhancements don't break existing functionality. Regression tests capture specific bugs that were discovered and fixed, preventing their reintroduction. This category grows over time as edge cases are discovered and resolved.

| Regression Category | Source of Test Cases | Maintenance Strategy | Update Triggers |
|--------------------|---------------------|---------------------|-----------------|
| Bug Fix Validation | Discovered parsing failures | Permanent test retention | Every bug fix adds test case |
| Edge Case Preservation | Boundary condition discoveries | Categorized edge case library | Format specification updates |
| Performance Regression | Performance degradation incidents | Benchmark integration | Significant algorithmic changes |
| Compatibility Maintenance | Version upgrade issues | Cross-version test suite | Dependency updates, format changes |

> **Key Insight:** The test categorization strategy ensures that each type of failure is caught by the most appropriate testing approach. Unit tests catch component logic errors quickly, integration tests catch interface mismatches, property-based tests discover edge cases, and regression tests prevent backsliding.

**Error Condition Testing Coverage** validates parser behavior when encountering malformed input, resource constraints, and exceptional conditions. Error testing is particularly critical for parsers because they must handle arbitrary user input gracefully while providing helpful feedback for fixing configuration issues.

| Error Category | Test Scenarios | Expected Behavior | Recovery Validation |
|----------------|----------------|-------------------|-------------------|
| Syntax Errors | Invalid characters, malformed structures | Specific error messages, accurate positions | Parsing continuation, error accumulation |
| Type Errors | Invalid type conversions, incompatible values | Type-specific error details, suggested fixes | Type inference fallbacks, default handling |
| Structure Errors | Invalid nesting, conflicting definitions | Structural context, conflict locations | Partial structure preservation, data recovery |
| Resource Errors | Large files, deep nesting, memory pressure | Graceful degradation, resource reporting | Streaming alternatives, size limits |

**Format-Specific Testing Coverage** addresses the unique characteristics and edge cases of each supported configuration format. Format-specific tests ensure compliance with format specifications and handle format-specific corner cases that don't apply to other formats.

| Format | Specific Test Areas | Critical Edge Cases | Compliance Validation |
|--------|-------------------|-------------------|---------------------|
| INI | Section headers, key-value variants, comment styles | Global keys, inline comments, quoted values | Loose INI standard interpretations |
| TOML | Table definitions, array-of-tables, type system | Table redefinition, dotted key conflicts, datetime formats | TOML v1.0.0 specification compliance |
| YAML | Indentation semantics, flow/block mixing, implicit typing | Tab vs space indentation, implicit type surprises | YAML subset specification adherence |

### Milestone Verification Points

Each implementation milestone requires specific verification criteria that confirm successful completion of core functionality before proceeding to more advanced features. Milestone verification provides concrete checkpoints that validate both functional correctness and implementation quality.

**Milestone 1: INI Parser Verification** establishes the foundation for configuration parsing by validating line-based parsing, section organization, and comment handling. INI parser verification focuses on the fundamental concepts of section-based organization and key-value extraction that underlie more complex parsing scenarios.

| Verification Category | Success Criteria | Test Validation | Implementation Quality |
|---------------------|------------------|----------------|----------------------|
| Section Header Parsing | `[section]` creates nested dictionary entry | Section names extracted correctly, nested structure created | Bracket validation, whitespace handling, invalid section recovery |
| Key-Value Processing | Both `key=value` and `key: value` supported | Values parsed with whitespace trimming, type inference applied | Quote handling, escape sequence processing, inline comment separation |
| Comment Handling | `;` and `#` comments ignored appropriately | Comment lines skipped, inline comments preserved/ignored based on configuration | Comment detection accuracy, mixed comment style handling |
| Global Key Support | Keys before first section handled correctly | Global namespace created, keys accessible in output structure | Global section naming, namespace organization |

The INI parser milestone verification requires comprehensive testing of edge cases that commonly trip up implementations. Testing must validate that the parser handles keys outside of sections appropriately, processes inline comments correctly without breaking on `=` characters inside quoted strings, and supports both common INI delimiters (`=` and `:`) with consistent behavior.

**Milestone Verification Test Suite for INI Parser:**

| Test Case Category | Test Scenarios | Expected Results | Failure Indicators |
|-------------------|----------------|------------------|-------------------|
| Basic Structure | Simple sections with key-value pairs | Nested dictionary with section keys | Missing sections, flat structure, incorrect nesting |
| Edge Cases | Empty sections, keys with no values, quoted strings with delimiters | Appropriate defaults, quote processing, delimiter isolation | Parsing failures, incorrect value extraction, quote handling errors |
| Comment Processing | Line comments, inline comments, mixed comment styles | Comments ignored or preserved based on parser configuration | Comment content included in values, parsing errors on comment lines |
| Error Recovery | Malformed section headers, invalid key syntax | Specific error messages, parsing continuation | Parser crashes, generic error messages, parsing termination |

**Milestone 2: TOML Tokenizer Verification** validates lexical analysis capabilities required for structured format parsing. TOML tokenizer verification focuses on accurate token generation, complex string handling, and position tracking that enables meaningful error reporting.

| Verification Category | Success Criteria | Token Accuracy | Error Handling |
|---------------------|------------------|----------------|----------------|
| Basic Token Types | All TOML grammar elements tokenized correctly | `STRING`, `NUMBER`, `BOOLEAN`, `IDENTIFIER` tokens generated accurately | Invalid token detection, position tracking, recovery strategies |
| String Literal Handling | Basic, literal, and multiline strings processed correctly | Escape sequence processing, quote type differentiation, multiline joining | Quote mismatch detection, escape sequence validation, multiline boundary handling |
| Numeric Processing | Integers, floats, dates, times recognized as distinct types | Type-specific token generation, format validation, underscore handling | Invalid numeric formats, overflow detection, format compliance |
| Position Tracking | Line and column numbers accurate for all tokens | `Position` objects reflect true source locations | Position drift, multiline position errors, tab handling |

TOML tokenizer verification requires extensive testing of string literal variants because TOML supports multiple string syntaxes with different escape processing rules. The tokenizer must correctly differentiate between basic strings (which process escapes) and literal strings (which treat backslashes literally), while handling multiline variants of both string types according to TOML specification rules.

**Milestone Verification Test Suite for TOML Tokenizer:**

| Test Case Category | Token Scenarios | Validation Approach | Quality Indicators |
|-------------------|----------------|-------------------|-------------------|
| String Variants | Basic strings with escapes, literal strings, multiline strings | Token type accuracy, value processing correctness | Escape sequence handling, quote processing, newline normalization |
| Numeric Types | Integers with underscores, scientific notation floats, ISO dates | Type-specific token generation, format compliance | Underscore handling, exponent processing, datetime parsing |
| Structural Elements | Brackets, dots, equals, commas in various combinations | Token sequence accuracy, delimiter recognition | Tokenization order, structural token identification |
| Error Conditions | Unterminated strings, invalid numeric formats, illegal characters | Error token generation, position accuracy, recovery behavior | Error context quality, position precision, tokenization continuation |

**Milestone 3: TOML Parser Verification** validates recursive descent parsing capabilities for complex nested structures. TOML parser verification focuses on table creation, array-of-tables handling, and conflict detection that ensures TOML specification compliance.

| Verification Category | Success Criteria | Structure Validation | Conflict Detection |
|---------------------|------------------|-------------------|-------------------|
| Table Parsing | `[table]` and `[table.subtable]` create correct nested structure | Dictionary nesting accuracy, key path resolution | Table redefinition detection, implicit table conflicts |
| Array-of-Tables | `[[array.of.tables]]` creates list of dictionary entries | Array structure creation, element organization | Array redefinition as table, mixed array/table conflicts |
| Dotted Key Expansion | `physical.color = 'orange'` creates nested structure | Automatic structure creation, key path processing | Conflicting key definitions, type mismatches |
| Inline Structures | Inline tables `{key = value}` and arrays `[1, 2, 3]` parsed correctly | Nested structure creation, type inference | Syntax error recovery, nesting validation |

TOML parser verification requires sophisticated conflict detection testing because TOML has complex rules about table redefinition and key conflicts. The parser must detect when a dotted key attempts to redefine an existing table, when a table is defined multiple times, and when array-of-tables syntax conflicts with existing definitions.

**Milestone Verification Test Suite for TOML Parser:**

| Test Case Category | Parsing Scenarios | Structure Validation | Error Detection |
|-------------------|------------------|-------------------|----------------|
| Table Hierarchies | Nested table definitions, dotted table paths, implicit table creation | Nesting accuracy, path resolution, structure completeness | Table redefinition detection, path conflict identification |
| Value Types | Strings, numbers, booleans, arrays, inline tables | Type inference accuracy, nested structure creation | Type conversion errors, syntax validation |
| Complex Structures | Mixed arrays, nested inline tables, array-of-tables with complex values | Deep structure accuracy, reference integrity | Circular reference detection, depth validation |
| Specification Compliance | Edge cases from TOML specification, conflict scenarios | Specification adherence, edge case handling | Standard compliance validation, error message quality |

**Milestone 4: YAML Subset Parser Verification** validates indentation-sensitive parsing for hierarchical block structures. YAML parser verification focuses on indentation stack management, structure type inference, and scalar type conversion.

| Verification Category | Success Criteria | Structure Management | Type Inference |
|---------------------|------------------|-------------------|----------------|
| Indentation Processing | Block structure determined correctly from indentation | Stack-based nesting, level transitions, structure preservation | Indentation error detection, tab vs space validation |
| Mapping Processing | `key: value` pairs create dictionary entries | Dictionary structure, nested mappings, key uniqueness | Key conflict detection, value processing |
| Sequence Processing | `- item` lists create ordered arrays | Array structure, nested sequences, mixed content | Item processing, nesting validation |
| Flow Syntax | `[list]` and `{map}` inline syntax parsed correctly | Inline structure creation, flow/block mixing | Syntax validation, nesting consistency |

YAML parser verification requires extensive indentation testing because YAML's indentation sensitivity creates numerous edge cases around tab handling, inconsistent indentation levels, and mixed indentation styles. The parser must maintain strict indentation validation while providing helpful error messages for common indentation mistakes.

**Milestone Verification Test Suite for YAML Parser:**

| Test Case Category | Indentation Scenarios | Structure Validation | Error Handling |
|-------------------|---------------------|-------------------|----------------|
| Block Structures | Consistent indentation, nested mappings and sequences | Structure accuracy, nesting preservation | Indentation error detection, level validation |
| Mixed Content | Mappings containing sequences, sequences containing mappings | Content type handling, structure transitions | Type conflict detection, mixed content validation |
| Scalar Processing | Quoted strings, unquoted strings, numeric values, booleans | Type inference accuracy, value processing | Type conversion errors, ambiguous value handling |
| Edge Cases | Empty documents, single-item structures, deeply nested content | Minimal structure handling, depth management | Edge case recovery, structure validation |

### Test Data Strategy

Effective configuration parser testing requires carefully curated test datasets that systematically explore the input space while covering critical edge cases and error conditions. The test data strategy must balance comprehensive coverage with maintainable test organization, ensuring that test cases remain understandable and debuggable as the test suite grows.

**Structured Test Data Organization** provides a systematic approach to organizing test cases across multiple dimensions: format types, feature complexity, error conditions, and edge cases. The organization strategy enables efficient test maintenance and comprehensive coverage validation.

| Test Data Category | Organization Principle | File Structure | Content Strategy |
|-------------------|----------------------|---------------|------------------|
| Golden Path Cases | Common usage patterns for each format | `test-data/golden/[format]/[feature].ext` | Real-world configuration examples, typical use cases |
| Edge Case Library | Boundary conditions and corner cases | `test-data/edge-cases/[format]/[category]/` | Minimal reproducible cases, focused edge conditions |
| Error Case Collection | Invalid input scenarios | `test-data/errors/[format]/[error-type]/` | Systematic error exploration, recovery validation |
| Cross-Format Equivalence | Semantically equivalent configurations | `test-data/equivalence/[scenario]/` | Same logical configuration in multiple formats |

The structured organization enables systematic test coverage analysis and makes it easy to add new test cases as edge cases are discovered. Each category serves a specific testing purpose and can be processed with appropriate test harness logic.

**Golden Path Test Data Strategy** focuses on realistic configuration scenarios that represent common usage patterns. Golden path tests validate that the parser handles typical use cases correctly and produces expected output structures. These tests serve as acceptance criteria and regression protection for core functionality.

| Format | Golden Path Scenarios | Test Data Characteristics | Validation Approach |
|--------|---------------------|--------------------------|-------------------|
| INI | Application settings, database configuration, service parameters | Multiple sections, varied value types, mixed comment styles | Structure accuracy, value processing, type inference |
| TOML | Package configuration, build settings, service definitions | Nested tables, arrays, mixed value types, complex structures | Nesting validation, type system compliance, specification adherence |
| YAML | Application config, deployment descriptors, data serialization | Block structures, sequences, mappings, mixed content types | Indentation handling, type inference, structure preservation |

Golden path test data should represent configurations that users would actually write, not contrived examples that exist only to test specific features. The test data should include realistic key names, appropriate value ranges, and natural organization patterns that reflect how each format is typically used.

**Edge Case Test Data Strategy** systematically explores boundary conditions and corner cases that often reveal parsing bugs. Edge case testing requires carefully constructed minimal examples that isolate specific problematic conditions without introducing additional complexity.

| Edge Case Category | Test Data Design | Validation Focus | Coverage Goals |
|-------------------|----------------|------------------|---------------|
| Empty Content | Empty files, whitespace-only files, comment-only files | Parser initialization, minimal input handling | Zero-content graceful handling |
| Boundary Values | Maximum nesting depth, longest strings, largest numbers | Resource handling, algorithmic limits | Performance boundaries, memory usage |
| Whitespace Sensitivity | Mixed tabs/spaces, trailing whitespace, Unicode whitespace | Whitespace processing, normalization | Whitespace semantic preservation |
| Unicode Complexity | Non-ASCII characters, emoji, combining characters, RTL text | Unicode handling, encoding processing | International character support |

Edge case test data must be constructed systematically to ensure comprehensive coverage of boundary conditions. Each edge case should focus on a single problematic condition to make failures easy to diagnose and fix.

**Error Case Test Data Strategy** validates parser behavior when encountering invalid input by providing systematic exploration of error conditions. Error case testing ensures that parsers fail gracefully and provide helpful error messages that enable users to fix their configuration files.

| Error Category | Test Data Design | Error Validation | Recovery Testing |
|----------------|----------------|------------------|------------------|
| Syntax Errors | Invalid characters, malformed structures, missing delimiters | Error message quality, position accuracy | Parsing continuation, error accumulation |
| Type Errors | Invalid type conversions, incompatible value assignments | Type-specific error reporting, conversion failure handling | Type inference fallbacks, default value handling |
| Structure Errors | Invalid nesting, conflicting definitions, circular references | Structural validation, conflict detection | Partial structure preservation, data salvage |
| Resource Errors | Extremely large files, deeply nested structures, memory pressure | Resource exhaustion handling, graceful degradation | Resource limit enforcement, streaming alternatives |

Error case test data should be constructed to trigger specific error conditions while remaining understandable to developers debugging parsing failures. Each error case should include expected error messages and recovery behavior validation.

**Cross-Format Equivalence Test Strategy** validates that semantically equivalent configurations produce consistent results across different formats. Equivalence testing ensures that format choice doesn't affect application behavior when configurations represent the same logical settings.

| Equivalence Scenario | Format Variations | Semantic Validation | Difference Handling |
|---------------------|-------------------|-------------------|-------------------|
| Simple Key-Value | INI sections, TOML tables, YAML mappings | Value equality, structure equivalence | Format-specific limitations, type coercion differences |
| Nested Structures | INI dotted sections, TOML nested tables, YAML block nesting | Hierarchy preservation, access path consistency | Nesting depth limits, structure representation differences |
| Array Handling | INI repeated keys, TOML arrays, YAML sequences | Array content equality, ordering preservation | Format array support differences, mixed type handling |
| Type Representation | Format-specific type syntax, implicit vs explicit typing | Type consistency, conversion accuracy | Type system differences, inference variations |

Cross-format equivalence testing helps validate that the unified output format successfully abstracts away format differences while preserving semantic meaning. These tests catch cases where format-specific processing introduces unintended behavioral differences.

**Test Data Maintenance Strategy** ensures that test datasets remain current, comprehensive, and maintainable as the parser implementation evolves. Maintenance strategy addresses test data organization, update procedures, and quality validation.

| Maintenance Aspect | Strategy Approach | Automation Support | Quality Assurance |
|-------------------|------------------|-------------------|-------------------|
| Test Case Discovery | Systematic exploration of input space, bug-driven case addition | Automated test case generation, property-based case discovery | Coverage analysis, gap identification |
| Data Quality Validation | Format compliance checking, expected result verification | Automated validation pipelines, consistency checking | Regular audit procedures, quality metrics |
| Update Procedures | Version control integration, change tracking, regression prevention | Automated update validation, backwards compatibility testing | Change impact analysis, regression detection |
| Organization Maintenance | Consistent categorization, clear naming conventions, duplicate elimination | Automated organization validation, duplicate detection | Regular cleanup procedures, organization audits |

> **Critical Insight:** Test data strategy success depends on systematic organization and comprehensive coverage rather than large volumes of ad-hoc test cases. Well-organized test data enables efficient debugging, comprehensive validation, and maintainable test suites.

![Complete Parsing Pipeline Flow](./diagrams/parsing-pipeline.svg)

The test data strategy must support both automated testing workflows and manual debugging scenarios. Test data should be organized to enable easy identification of relevant test cases, efficient test execution, and clear failure diagnosis when tests fail.

**Performance Test Data Strategy** validates parser behavior under resource pressure and ensures that algorithmic complexity remains acceptable for realistic input sizes. Performance testing requires carefully constructed datasets that stress specific performance characteristics without introducing artificial complexity.

| Performance Aspect | Test Data Characteristics | Measurement Focus | Validation Criteria |
|-------------------|--------------------------|-------------------|-------------------|
| File Size Scaling | Incrementally larger configuration files | Memory usage, parsing time, throughput | Linear scaling behavior, memory efficiency |
| Nesting Depth | Increasingly deep nested structures | Stack usage, recursion handling, algorithm complexity | Graceful degradation, depth limit handling |
| Key Volume | Large numbers of keys and sections | Hash table performance, lookup efficiency | Consistent access times, memory organization |
| String Processing | Large string values, complex escape sequences | String processing efficiency, memory allocation | String handling optimization, garbage collection impact |

Performance test data should reflect realistic scaling scenarios that applications might encounter, rather than pathological cases designed to break the parser. The focus should be on ensuring acceptable performance for reasonable input sizes while establishing clear limits for extreme cases.

### Implementation Guidance

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Test Framework | `pytest` with fixtures and parametrize | `pytest` with `hypothesis` for property-based testing |
| Test Data Management | JSON files with expected results | YAML test definitions with embedded test cases |
| Coverage Analysis | `coverage.py` for line coverage | `coverage.py` + `pytest-cov` with branch coverage |
| Performance Testing | Simple timing with `time.time()` | `pytest-benchmark` with statistical analysis |
| Test Organization | Directory-based test separation | `pytest` marks and custom test collections |

**Recommended File Structure:**
```
config-parser/
  tests/
    unit/
      test_tokenizer.py          ← BaseTokenizer unit tests
      test_ini_parser.py         ← INIParser specific tests
      test_toml_parser.py        ← TOMLParser specific tests
      test_yaml_parser.py        ← YAMLParser specific tests
    integration/
      test_parsing_pipeline.py   ← End-to-end parsing tests
      test_format_detection.py   ← Format detection integration
      test_error_handling.py     ← Cross-component error flow
    data/
      golden/
        ini/                     ← Golden path INI test files
        toml/                    ← Golden path TOML test files
        yaml/                    ← Golden path YAML test files
      edge-cases/
        tokenizer/               ← Tokenization edge cases
        parsing/                 ← Parser-specific edge cases
      errors/
        syntax/                  ← Syntax error test cases
        structure/               ← Structure error test cases
      equivalence/
        basic-config/            ← Cross-format equivalent configs
    conftest.py                  ← Shared test fixtures
    test_utils.py                ← Testing utility functions
```

**Test Infrastructure Starter Code:**

```python
# tests/conftest.py - Shared test fixtures and utilities
import pytest
import json
from pathlib import Path
from typing import Dict, Any, List, Union
from dataclasses import dataclass

@dataclass
class TestCase:
    """Represents a single parser test case with input, expected output, and metadata."""
    name: str
    input_content: str
    expected_output: Dict[str, Any]
    expected_errors: List[str] = None
    format_hint: str = None
    description: str = ""

class TestDataLoader:
    """Loads and manages test data files with caching and validation."""
    
    def __init__(self, test_data_root: Path):
        self.test_data_root = test_data_root
        self._cache = {}
    
    def load_test_cases(self, category: str, format_type: str = None) -> List[TestCase]:
        """Load test cases from organized test data directory structure."""
        # TODO 1: Build path based on category and optional format_type
        # TODO 2: Scan directory for test case files (.json, .yaml, .toml, .ini)
        # TODO 3: Load each file and create TestCase objects
        # TODO 4: Cache loaded test cases for performance
        # TODO 5: Validate test case structure and expected results
        pass
    
    def load_equivalence_set(self, scenario_name: str) -> Dict[str, TestCase]:
        """Load cross-format equivalent test cases for validation."""
        # TODO 1: Load test cases for all formats in equivalence scenario
        # TODO 2: Validate that expected outputs are semantically equivalent
        # TODO 3: Return dictionary mapping format -> TestCase
        pass

@pytest.fixture
def test_data_loader():
    """Provides TestDataLoader instance for all tests."""
    test_data_root = Path(__file__).parent / "data"
    return TestDataLoader(test_data_root)

@pytest.fixture
def parser_factory():
    """Factory for creating parser instances with test configuration."""
    def create_parser(format_type: str, **options):
        if format_type == "ini":
            from config_parser.ini_parser import INIParser
            return INIParser(**options)
        elif format_type == "toml":
            from config_parser.toml_parser import TOMLParser
            return TOMLParser(**options)
        elif format_type == "yaml":
            from config_parser.yaml_parser import YAMLParser
            return YAMLParser(**options)
        else:
            raise ValueError(f"Unknown format: {format_type}")
    return create_parser

def assert_equivalent_structures(actual: Dict[str, Any], expected: Dict[str, Any], 
                                path: str = "root") -> None:
    """Deep comparison of nested dictionary structures with helpful error messages."""
    # TODO 1: Compare dictionary keys, reporting missing/extra keys with path context
    # TODO 2: Recursively compare nested dictionaries and lists
    # TODO 3: Handle type coercion differences (e.g., "1" vs 1) appropriately
    # TODO 4: Provide detailed error messages with path information for failures
    pass

def normalize_test_output(output: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize parser output for cross-format comparison."""
    # TODO 1: Sort dictionary keys recursively for consistent comparison
    # TODO 2: Normalize string representations of numbers and booleans
    # TODO 3: Handle format-specific type differences (e.g., date objects vs strings)
    # TODO 4: Remove format-specific metadata that shouldn't affect equivalence
    pass
```

**Unit Test Skeleton Code:**

```python
# tests/unit/test_tokenizer.py - Tokenizer unit tests
import pytest
from config_parser.tokenizer import BaseTokenizer, Token, TokenType, Position

class TestBaseTokenizer:
    """Unit tests for tokenizer functionality across all formats."""
    
    def test_basic_token_generation(self, test_data_loader):
        """Validate basic tokenization for all supported token types."""
        # TODO 1: Load tokenizer test cases with expected token sequences
        # TODO 2: Create tokenizer instance and tokenize test input
        # TODO 3: Compare generated tokens with expected tokens
        # TODO 4: Validate token types, values, and position information
        # TODO 5: Test edge cases like empty input, whitespace-only, comments
        pass
    
    def test_string_literal_handling(self):
        """Test complex string literal parsing including escapes and multiline."""
        test_cases = [
            ('basic_string', '"Hello World"', "Hello World"),
            ('escape_sequences', '"Line 1\\nLine 2\\t\\""', "Line 1\nLine 2\t\""),
            ('multiline_string', '"""Line 1\nLine 2"""', "Line 1\nLine 2"),
            ('literal_string', "'No\\nEscape'", "No\\nEscape"),
        ]
        
        for test_name, input_text, expected_value in test_cases:
            # TODO 1: Create tokenizer with string input
            # TODO 2: Tokenize and extract string token
            # TODO 3: Validate token type is STRING
            # TODO 4: Validate processed value matches expected result
            # TODO 5: Validate position tracking through string processing
            pass
    
    def test_position_tracking(self):
        """Validate accurate line and column tracking during tokenization."""
        multiline_input = """line1 = "value1"
        [section]
        line3 = 123"""
        
        # TODO 1: Tokenize multiline input and collect all tokens
        # TODO 2: Validate that line numbers increment correctly
        # TODO 3: Validate that column numbers reset after newlines
        # TODO 4: Validate position accuracy for tokens spanning multiple characters
        # TODO 5: Test position tracking with tabs, Unicode characters, and mixed line endings
        pass
    
    def test_error_recovery(self):
        """Test tokenizer behavior with invalid characters and malformed input."""
        error_cases = [
            "unterminated string \"never ends",
            "invalid \x00 null character",
            "unicode handling test 🚀",
            "mixed quotes 'started with single \"ended with double",
        ]
        
        for error_input in error_cases:
            # TODO 1: Create tokenizer with malformed input
            # TODO 2: Tokenize and expect appropriate error token generation
            # TODO 3: Validate error position accuracy
            # TODO 4: Verify tokenizer continues processing after errors
            # TODO 5: Test that subsequent valid tokens are processed correctly
            pass

# tests/unit/test_ini_parser.py - INI parser unit tests
class TestINIParser:
    """Unit tests for INI-specific parsing functionality."""
    
    @pytest.mark.parametrize("test_case", [
        pytest.param(TestCase("basic_sections", "[section1]\nkey=value", 
                             {"section1": {"key": "value"}}), id="basic"),
        pytest.param(TestCase("global_keys", "global=value\n[section]\nlocal=value",
                             {"global": "value", "section": {"local": "value"}}), id="global"),
    ])
    def test_section_parsing(self, test_case, parser_factory):
        """Test section header parsing and nested structure creation."""
        parser = parser_factory("ini")
        
        # TODO 1: Parse test case input content
        # TODO 2: Validate section structure matches expected output
        # TODO 3: Verify nested dictionary organization
        # TODO 4: Test section name extraction and normalization
        # TODO 5: Validate global key handling before first section
        pass
    
    def test_comment_handling(self, parser_factory):
        """Test comment processing with different comment styles."""
        ini_content = """
        ; This is a semicolon comment
        # This is a hash comment
        [section]
        key1 = value1  ; inline semicolon comment
        key2 = value2  # inline hash comment
        ; Another comment
        key3 = value3
        """
        
        # TODO 1: Parse INI content with mixed comment styles
        # TODO 2: Verify comments are ignored during parsing
        # TODO 3: Validate inline comments don't affect value processing
        # TODO 4: Test comment handling configuration options
        # TODO 5: Verify structure contains only actual key-value pairs
        pass
    
    def test_value_processing(self, parser_factory):
        """Test key-value parsing including quotes, escapes, and type inference."""
        test_cases = [
            ('unquoted_string', 'key = value', 'value'),
            ('quoted_string', 'key = "quoted value"', 'quoted value'),
            ('number_inference', 'key = 123', 123),
            ('boolean_inference', 'key = true', True),
            ('escaped_quotes', 'key = "say \\"hello\\""', 'say "hello"'),
        ]
        
        parser = parser_factory("ini")
        
        for test_name, ini_line, expected_value in test_cases:
            # TODO 1: Parse single key-value line in section context
            # TODO 2: Extract parsed value from result structure
            # TODO 3: Validate value matches expected result
            # TODO 4: Verify type inference worked correctly
            # TODO 5: Test whitespace trimming and quote processing
            pass
```

**Integration Test Skeleton Code:**

```python
# tests/integration/test_parsing_pipeline.py - End-to-end integration tests
class TestParsingPipeline:
    """Integration tests for complete parsing workflow."""
    
    def test_file_to_dictionary_pipeline(self, test_data_loader):
        """Test complete pipeline from configuration files to parsed dictionaries."""
        for format_type in ['ini', 'toml', 'yaml']:
            golden_cases = test_data_loader.load_test_cases('golden', format_type)
            
            for test_case in golden_cases:
                # TODO 1: Create parser for format type
                # TODO 2: Parse test case input content
                # TODO 3: Validate output structure matches expected result
                # TODO 4: Verify no errors occurred during parsing
                # TODO 5: Test with different parser configuration options
                pass
    
    def test_cross_format_equivalence(self, test_data_loader, parser_factory):
        """Validate semantically equivalent configurations produce consistent results."""
        equivalence_scenarios = [
            'basic_config', 'nested_structures', 'array_handling', 'type_inference'
        ]
        
        for scenario in equivalence_scenarios:
            equivalent_cases = test_data_loader.load_equivalence_set(scenario)
            
            # TODO 1: Parse same logical configuration in all supported formats
            # TODO 2: Normalize output structures for comparison
            # TODO 3: Validate semantic equivalence across formats
            # TODO 4: Document and validate acceptable format differences
            # TODO 5: Test edge cases where equivalence might not be perfect
            pass
    
    def test_error_context_propagation(self, parser_factory):
        """Test error information flow from tokenizer through parser to user."""
        malformed_configs = {
            'ini': '[broken section\nkey without section',
            'toml': '[table.redefinition]\nvalue = 1\n[table]\nconflict = 2',
            'yaml': 'mapping:\n\titem: value\n  other: value'  # mixed tabs/spaces
        }
        
        for format_type, malformed_content in malformed_configs.items():
            parser = parser_factory(format_type)
            
            # TODO 1: Parse malformed content and expect parsing errors
            # TODO 2: Validate error messages contain helpful context
            # TODO 3: Verify error positions are accurate
            # TODO 4: Test error recovery and continued parsing
            # TODO 5: Validate error message quality and actionability
            pass
```

**Milestone Checkpoints:**

**Milestone 1 Checkpoint - INI Parser:**
```bash
# Run INI parser tests
python -m pytest tests/unit/test_ini_parser.py -v

# Expected output:
# test_section_parsing[basic] PASSED
# test_section_parsing[global] PASSED
# test_comment_handling PASSED
# test_value_processing PASSED

# Manual verification:
python -c "
from config_parser.ini_parser import INIParser
parser = INIParser()
result = parser.parse('[database]\nhost = localhost\nport = 5432\n# comment line\nuser = admin')
print('Parsed structure:', result)
# Should output: {'database': {'host': 'localhost', 'port': 5432, 'user': 'admin'}}
"
```

**Milestone 2 Checkpoint - TOML Tokenizer:**
```bash
# Run tokenizer tests
python -m pytest tests/unit/test_tokenizer.py::TestBaseTokenizer::test_basic_token_generation -v

# Manual tokenization verification:
python -c "
from config_parser.tokenizer import BaseTokenizer
tokenizer = BaseTokenizer('[table]\nkey = \"value with \\\\ escapes\"')
tokens = tokenizer.tokenize()
for token in tokens:
    print(f'{token.type.name}: {token.value} at {token.position.line}:{token.position.column}')
"
```

**Performance Testing Integration:**
```python
# tests/performance/test_parser_performance.py
import pytest
import time
from pathlib import Path

class TestParserPerformance:
    """Performance validation for parser components."""
    
    @pytest.mark.benchmark
    def test_file_size_scaling(self, parser_factory, benchmark):
        """Validate parsing performance scales appropriately with file size."""
        # TODO 1: Generate configuration files of increasing sizes
        # TODO 2: Benchmark parsing time for each size
        # TODO 3: Validate linear scaling characteristics
        # TODO 4: Measure memory usage during parsing
        # TODO 5: Establish performance baselines for regression testing
        pass
    
    def test_nesting_depth_limits(self, parser_factory):
        """Test parser behavior with deeply nested structures."""
        max_depth = 100
        
        # TODO 1: Generate deeply nested test configurations
        # TODO 2: Parse configurations at increasing depth levels
        # TODO 3: Measure parsing time and memory usage
        # TODO 4: Identify practical depth limits
        # TODO 5: Validate graceful handling of extreme nesting
        pass


## Debugging Guide

> **Milestone(s):** All milestones (INI Parser, TOML Tokenizer, TOML Parser, YAML Subset Parser) - comprehensive debugging strategies, symptom diagnosis, and troubleshooting techniques for successful parsing implementation

Debugging configuration parsers presents unique challenges that differ significantly from typical application debugging. Unlike business logic where errors often manifest as obvious behavioral inconsistencies, parsing errors frequently emerge as subtle structural misalignments, tokenization edge cases, or context-sensitive interpretation failures. The multi-format nature of our parser amplifies these challenges by introducing format-specific edge cases alongside shared infrastructure bugs.

Think of debugging a parser as forensic investigation rather than traditional problem-solving. When a parser fails, you must trace backwards through multiple layers: the final output structure, the parse tree construction, the tokenization process, and the original character-by-character scanning. Each layer introduces potential failure points, and symptoms at the output level often obscure root causes buried deep in the tokenization logic. This section provides systematic approaches for conducting this forensic analysis effectively.

The complexity stems from parsing's inherently contextual nature. The same character sequence "key = value" might be valid in INI format but invalid in YAML context, while multiline strings behave completely differently across TOML and YAML. Context sensitivity means bugs often manifest inconsistently - working perfectly for simple cases but failing mysteriously when nesting levels change, indentation patterns shift, or specific character combinations appear.

### Common Bug Symptoms and Causes

Understanding the symptom-to-cause mapping for parsing bugs accelerates debugging by directing investigation toward the most likely root causes. Parsing failures typically fall into distinct categories with characteristic symptoms that point to specific implementation areas.

![Error Detection and Reporting Flow](./diagrams/error-handling-flow.svg)

#### Structural Misrepresentation Symptoms

**Missing Nested Structure**: When configuration content contains hierarchical structure but the parsed output flattens it into a single level, the bug typically originates in path expansion logic. For INI parsers, this manifests when dotted section names like `[database.connection.pool]` create a flat key instead of nested dictionaries. The root cause usually lies in `create_nested_section` not properly splitting section paths or `merge_nested_dicts` failing to create intermediate levels.

**Incorrect Nesting Depth**: Output shows wrong nesting levels - either too shallow or too deep. In YAML parsers, this symptom points to indentation stack management failures. The `IndentationStack` might incorrectly calculate target levels during `pop_to_level` operations, or `_handle_indentation_transition` might push frames when it should maintain current level. TOML parsers exhibit this when dotted key expansion in `expand_dotted_key` creates too many intermediate tables.

**Key Collision Overwriting**: Later-defined keys silently overwrite earlier ones instead of generating conflicts. This indicates insufficient symbol table tracking in TOML parsers. The `SymbolTable` should register every definition through `register_definition` and validate against redefinition rules. Missing validation allows conflicting definitions to proceed unchecked.

| Symptom | Format Context | Likely Root Cause | Diagnostic Focus |
|---------|----------------|-------------------|------------------|
| Flat output for nested syntax | INI dotted sections | Section path splitting failure | `create_nested_section` logic |
| Wrong nesting depth | YAML indentation | Stack management errors | `IndentationStack` operations |
| Silent key overwrites | TOML dotted keys | Missing conflict detection | `SymbolTable` validation |
| Array structure lost | TOML array-of-tables | Table vs array confusion | `parse_table_header` logic |

#### Value Processing Failures

**Type Conversion Errors**: Values appear as strings when they should be numbers, booleans, or other types. This symptom indicates failures in type inference logic. Each format's `infer_type` method might contain incomplete boolean detection, numeric parsing edge cases, or missing null value recognition. YAML's implicit typing is particularly susceptible due to complex conversion rules.

**String Escape Sequence Issues**: Literal backslashes appear in output instead of processed escapes, or escape sequences cause parsing to fail entirely. The root cause lies in `read_string_literal` implementation within the tokenizer. Multi-format parsers must handle different escape rules: INI uses minimal escaping, TOML has basic and literal string variants, while YAML has complex folding rules.

**Multiline Value Corruption**: Multiline strings split incorrectly, lose formatting, or include unintended content. This points to line continuation logic failures. Each format handles multilines differently: INI uses backslash continuation, TOML has multiline string delimiters, YAML uses folding indicators. The tokenizer's state machine must track multiline context accurately.

**Inline Comment Contamination**: Values include comment text that should be stripped. This indicates inadequate comment detection during value parsing. The `process_key_value_pair` logic must identify comment boundaries within lines while respecting quoted string contexts where hash or semicolon characters are literal.

#### Tokenization Boundary Errors

**Token Splitting Failures**: Single logical units split across multiple tokens, or multiple units merged into single tokens. This symptom reveals boundary detection errors in the tokenizer. Complex tokens like dotted identifiers, numeric literals with underscores, or quoted strings with embedded delimiters require sophisticated boundary logic.

**Context Sensitivity Violations**: Same character sequences tokenized differently in identical contexts, or differently in contexts where they should be identical. This points to inadequate context tracking in `BaseTokenizer`. State machines must maintain consistent context interpretation across equivalent parsing situations.

**Position Tracking Inaccuracies**: Error messages report wrong line/column positions, making debugging extremely difficult. The `Position` calculation logic in `current_position` must accurately track line breaks, character offsets, and column advancement through all tokenization paths including multiline constructs.

#### Format Detection Ambiguities

**Wrong Format Selection**: Parser selects incorrect format, causing interpretation failures. The `detect_format` algorithm relies on syntactic signatures that distinguish formats. Ambiguous content might match multiple format patterns, requiring confidence scoring to select the most appropriate parser.

**Format Switching Mid-Parse**: Parser begins with one format interpretation then shifts to another, causing structural inconsistencies. This indicates insufficient lookahead in format detection. The analysis window must examine enough content to establish format identity before committing to specific parsing logic.

> **Design Insight**: The most challenging bugs combine multiple symptom categories. For example, a TOML array-of-tables with complex nested structure might simultaneously exhibit tokenization boundary errors (array brackets), structural misrepresentation (table nesting), and value processing failures (nested values). Systematic diagnosis must examine each layer independently before considering interactions.

#### Common Error Propagation Patterns

Parsing errors often cascade through multiple system layers, making root cause identification challenging. Understanding propagation patterns helps focus debugging efforts on primary failure points rather than secondary symptoms.

**Tokenization → Structure Cascade**: Incorrect tokenization creates valid but wrong tokens that produce structurally incorrect parse trees. For example, a quoted string boundary missed by the tokenizer appears as separate IDENTIFIER and STRING tokens. The parser successfully processes these tokens but creates wrong structure. Diagnosis must verify token correctness before examining structural logic.

**Context → Value Cascade**: Wrong parsing context causes correct tokens to be interpreted inappropriately. A YAML scalar appears in sequence context but gets processed as mapping key, leading to structural errors. The tokenization is correct, but context tracking failures cause misinterpretation.

**Format → Everything Cascade**: Incorrect format detection causes the wrong parser to process content, leading to systematic failures across all processing layers. An INI file processed by the TOML parser generates numerous errors because fundamental syntax assumptions are violated. This pattern requires format detection validation as the first debugging step.

### Debugging Techniques for Parsers

Effective parser debugging requires specialized techniques that differ from general software debugging. The multi-layered nature of parsing - from character scanning through tokenization to structure building - demands systematic approaches for isolating failures at each level.

#### Layered Diagnosis Methodology

**Layer 1: Character Stream Analysis**: Begin debugging at the most fundamental level by examining the raw input character stream. Many parsing failures originate from assumptions about input encoding, line ending conventions, or whitespace handling that prove incorrect for specific content.

Create character-level inspection by implementing a diagnostic scanner that reveals non-printable characters, mixed encodings, and whitespace variations. Unicode normalization issues, byte order marks, or mixed line ending styles (Unix LF vs Windows CRLF) frequently cause parsing failures that manifest as mysterious structural errors.

Position tracking verification requires tracing through the character stream manually to confirm that `current_position` calculations match actual line and column positions. Off-by-one errors in position tracking make all subsequent error reporting unreliable, severely hampering debugging efforts.

**Layer 2: Token Stream Validation**: After confirming character stream integrity, examine the token stream produced by the tokenizer. This intermediate representation reveals whether lexical analysis correctly identifies meaningful units within the character stream.

Token boundary verification involves examining each token's `raw_text` field against its `position` information to confirm accurate extraction. Boundary errors often appear as tokens that include extra characters, exclude expected characters, or split logical units incorrectly.

Token type accuracy checking ensures that the tokenizer assigns correct `TokenType` values to extracted text. A numeric literal incorrectly classified as IDENTIFIER causes downstream parsing failures that appear as type conversion errors rather than tokenization bugs.

Context tracking validation confirms that tokenization state changes appropriately as content context shifts. Multiline strings, comment regions, and format-specific constructs require state machine transitions that maintain consistent interpretation rules.

**Layer 3: Parse Tree Structure Inspection**: With correct tokenization confirmed, examine the intermediate parse tree structure built by format-specific parsers. This layer reveals whether structural interpretation correctly translates token sequences into hierarchical representations.

Node hierarchy verification checks that parent-child relationships in the parse tree match expected nesting from the original content. Missing intermediate nodes, incorrect nesting depths, or orphaned nodes indicate structural interpretation failures.

Node metadata accuracy ensures that `ParseNode` instances contain correct position information, token references, and format-specific metadata. Inaccurate metadata complicates error reporting and downstream processing.

**Layer 4: Final Output Validation**: The final debugging layer examines the conversion from parse tree to the unified output format. This transformation must preserve all structural relationships while normalizing format-specific representations into consistent dictionary structures.

Key path accuracy verification confirms that nested dictionary keys match expected hierarchical paths from the original content. The `convert_parse_tree_to_dict` transformation must maintain all structural relationships without introducing spurious levels or losing intended nesting.

Value conversion correctness ensures that the type inference and conversion process produces appropriate Python types for each value. String-to-type conversion errors often appear as unexpected string values in contexts where numbers or booleans are expected.

#### Incremental Complexity Testing

**Minimal Reproduction Construction**: When debugging complex parsing failures, construct minimal examples that reproduce the same symptoms with the simplest possible input. This technique isolates the specific conditions that trigger failures without the complexity of realistic configuration files.

Start with single-line examples that exhibit the problematic behavior. If a complex nested TOML structure fails to parse correctly, create a minimal table definition that shows the same structural issues. Gradually increase complexity while maintaining the failure symptom to identify the exact complexity threshold where problems appear.

Format isolation testing processes the same logical content through different format parsers to identify format-specific versus general infrastructure bugs. If the same nested structure parses correctly in YAML but fails in TOML, the issue lies in TOML-specific logic rather than shared infrastructure.

**Progressive Feature Addition**: Build parser functionality incrementally, validating correct behavior at each step before adding complexity. This approach prevents multiple bugs from interacting and obscuring individual failure points.

Begin with basic key-value parsing without nesting, comments, or complex values. Confirm perfect behavior for simple cases before introducing sectioning, then nesting, then complex value types. Each addition point becomes a checkpoint for isolating newly introduced bugs.

Feature interaction testing examines combinations of parser features that work individually but fail when combined. Comments within multiline strings, nested structures with dotted keys, or array-of-tables with inline values represent interaction points where subtle bugs frequently emerge.

#### State Inspection and Tracing

**Parser State Snapshots**: Implement diagnostic capabilities that capture complete parser state at critical processing points. State snapshots provide detailed views of internal parser condition when failures occur.

The `ParseContext` should support diagnostic mode that records position progression, tokenization state, parse tree construction steps, and error accumulation. This information enables post-mortem analysis of parsing sessions to identify the exact point where processing diverged from expected behavior.

Symbol table inspection for TOML parsing reveals the current definition state, helping identify redefinition conflicts or missing implicit table creation. The `SymbolTable` diagnostic interface should expose all registered definitions with their types and positions.

Indentation stack analysis for YAML parsing shows the current nesting context and established indentation levels. The `IndentationStack` diagnostic view helps identify incorrect stack operations that lead to structural misinterpretation.

**Token Stream Replay**: Implement token stream recording and replay capabilities that enable re-processing specific token sequences with different parser configurations or enhanced diagnostics enabled.

Token stream serialization captures the complete sequence of tokens produced for problematic input, enabling offline analysis and regression testing. Replay functionality allows processing the same token stream multiple times with different diagnostic settings or parser modifications.

Interactive token inspection provides step-by-step token stream examination with parser state inspection at each token boundary. This technique helps identify the specific token where parsing logic makes incorrect decisions.

#### Error Message Archaeological Analysis

**Error Context Reconstruction**: Parser error messages often provide insufficient context for effective debugging. Implement enhanced error context generation that shows not just the immediate error location but the parsing context leading to the failure.

The `create_error_context` function should include previous parsing decisions, current parser state, and upcoming tokens to provide comprehensive situational awareness. Context windows should adapt to the specific error type - structural errors need broader context than tokenization errors.

Error correlation analysis identifies relationships between multiple errors that stem from single root causes. Cascading failures often generate numerous error messages that obscure the primary issue. Group related errors and present root cause analysis rather than symptom catalogs.

**Error Evolution Tracking**: Track how error conditions evolve during parsing to understand failure progression. Some errors represent recoverable conditions that become fatal due to inadequate error recovery, while others indicate fundamental structural problems.

Recovery decision analysis examines the effectiveness of error recovery strategies by tracking parsing progress after recovery attempts. The `ErrorRecoveryState` should maintain metrics on recovery success rates and provide feedback for recovery strategy refinement.

### Debugging Tools and Inspection

Effective parser debugging requires specialized tooling that provides visibility into the multi-layered parsing process. Unlike general application debugging where breakpoints and variable inspection suffice, parser debugging demands tools that can trace through character streams, token sequences, and structural transformations while maintaining context awareness.

#### Interactive Parser Inspection Framework

**Token-by-Token Stepping Interface**: Implement an interactive debugging interface that allows stepping through tokenization one token at a time while inspecting complete parser state. This capability proves essential for understanding how specific character sequences translate into token streams and how parser decisions evolve.

The stepping interface should display the current character position, upcoming character sequences, tokenizer state machine status, and accumulated token stream. At each step, inspect the decision logic for token boundary detection, type classification, and value extraction. This granular visibility reveals tokenization edge cases that are invisible in batch processing.

Context-aware display formatting presents different views optimized for different parsing phases. During tokenization, emphasize character boundaries and state transitions. During structural parsing, highlight nesting relationships and symbol table evolution. During error recovery, focus on recovery decision points and confidence metrics.

**Parse Tree Visualization Tools**: Develop visual representations of intermediate parse tree structures that reveal hierarchical relationships and node metadata. Text-based tree displays work well for automated testing, but interactive visual tools provide superior debugging capability for complex structures.

Node inspection capabilities should expose all `ParseNode` fields including position information, token references, child relationships, and format-specific metadata. Interactive expansion and collapse of subtrees helps manage complexity when debugging large configuration files.

Comparative tree visualization shows differences between expected and actual parse tree structures, highlighting specific nodes where structure diverges from expectations. This capability accelerates debugging of structural interpretation issues.

#### Tokenization Analysis Toolkit

**Character Stream Inspector**: Build diagnostic tools that reveal character-level details often hidden by text editors and terminal displays. Many parsing bugs stem from invisible characters, mixed encodings, or non-standard whitespace that standard tools don't reveal clearly.

The inspector should display hexadecimal character codes alongside visual representations, highlight different whitespace types distinctly, and identify potential encoding issues. Unicode normalization problems, zero-width characters, and mixed line ending styles become immediately visible.

Position mapping verification tools trace the relationship between character stream offsets and calculated line/column positions. Generate position maps that can be compared against expected values to identify off-by-one errors or miscalculated boundaries.

**Token Stream Analysis Suite**: Develop comprehensive tools for examining token streams in detail, including sequence analysis, boundary verification, and context tracking validation.

Token sequence comparison tools examine token streams produced from similar input to identify inconsistencies in tokenization behavior. This capability helps identify context sensitivity bugs where equivalent content tokenizes differently in different parsing contexts.

Boundary accuracy verification reconstructs original character sequences from token `raw_text` fields and position information, comparing against the original input to identify extraction errors. Gaps or overlaps in token coverage indicate boundary detection problems.

Token type distribution analysis reveals patterns in token classification that can identify systematic errors in type detection logic. Unexpected type distributions often point to classification rules that behave differently than intended.

#### Parser State Inspection Utilities

**Symbol Table Diagnostic Views**: For TOML parsing, implement comprehensive symbol table inspection that reveals definition conflicts, implicit table creation, and key path resolution. The `SymbolTable` diagnostic interface should support queries about definition history and conflict analysis.

Definition timeline views show the sequence of key and table definitions with their positions and types. This information helps identify redefinition violations and understand how implicit table creation interacts with explicit definitions.

Conflict detection analysis explains why specific key combinations are invalid, providing detailed explanations of TOML's redefinition rules. When debugging table conflicts, show the complete definition history that led to the conflict condition.

**Indentation Stack Analysis**: For YAML parsing, develop tools that visualize indentation stack evolution and validate stack operations. The `IndentationStack` diagnostic interface should expose stack frame details and transition logic.

Stack frame inspection shows the complete frame history including indentation levels, structure types, and line numbers where frames were established. This information helps identify incorrect stack operations that lead to nesting errors.

Indentation level analysis validates that calculated indentation levels match established patterns and identifies ambiguous indentation that could be interpreted multiple ways. Level transition visualization shows how stack operations respond to indentation changes.

#### Error Analysis and Recovery Tools

**Error Pattern Recognition**: Implement tools that analyze error patterns across multiple parsing attempts to identify systematic issues in parser logic or input handling. Pattern recognition helps distinguish between input-specific errors and parser implementation bugs.

Error clustering analysis groups similar errors by type, location patterns, and context to identify common failure modes. This analysis helps prioritize parser improvements by focusing on the most frequent error categories.

Recovery effectiveness measurement tracks the success rate of different error recovery strategies and provides data for refining recovery logic. The `ErrorRecoveryState` metrics help optimize recovery decision-making.

**Diagnostic Trace Generation**: Develop comprehensive tracing that captures the complete parsing decision sequence for post-mortem analysis. Traces should include character scanning, tokenization decisions, parser state transitions, and error handling decisions.

Execution path analysis identifies the specific code paths followed during parsing, helping isolate bugs to particular logic branches. This information proves especially valuable for debugging complex conditional logic in recursive descent parsers.

Decision point logging records the rationale for parser decisions at critical points, providing insight into why specific interpretation paths were chosen. This information helps identify cases where parser logic makes reasonable but incorrect decisions based on insufficient context.

#### Performance and Scalability Analysis

**Parsing Performance Profiler**: Implement profiling tools that identify performance bottlenecks in parsing logic, focusing on operations that scale poorly with input size or complexity.

Time allocation analysis shows how parsing time distributes across different operations: tokenization, structural analysis, value conversion, and output generation. This information helps identify optimization opportunities and scalability concerns.

Memory usage tracking reveals memory allocation patterns and identifies potential memory leaks or excessive allocation in parsing logic. Parser implementations should maintain bounded memory usage regardless of input complexity.

**Scalability Stress Testing**: Develop tools that generate configuration files of varying sizes and complexity to test parser behavior under stress conditions. Scalability testing reveals performance degradation patterns and memory usage growth.

Input complexity graduation creates test cases with systematically increasing complexity: nesting depth, table count, array size, and string length. This approach identifies complexity thresholds where parser performance degrades significantly.

Resource consumption monitoring tracks CPU usage, memory allocation, and parsing time across different input categories to establish performance baselines and identify regression conditions.

### Implementation Guidance

This implementation guidance provides practical tools and techniques for implementing comprehensive debugging capabilities for your configuration file parser. The focus is on building diagnostic infrastructure that integrates seamlessly with your parser implementation while providing powerful debugging capabilities.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Diagnostic Output | Print statements with structured formatting | Rich library for terminal formatting and interactive displays |
| State Inspection | JSON serialization of parser state | Custom inspection framework with interactive browsing |
| Token Visualization | Plain text token stream dumps | HTML/web-based token inspector with syntax highlighting |
| Error Analysis | Basic error categorization and counting | Statistical analysis with matplotlib for error pattern visualization |
| Interactive Debugging | Command-line REPL with parser commands | Web-based debugging interface with real-time visualization |

#### Recommended File/Module Structure

```
config-parser/
  src/
    parser/
      core/
        tokenizer.py              ← BaseTokenizer with diagnostic capabilities
        errors.py                 ← Error types and diagnostic functions
        context.py                ← ParseContext with debugging support
      formats/
        ini_parser.py             ← INI parser with diagnostic hooks
        toml_parser.py            ← TOML parser with symbol table inspection
        yaml_parser.py            ← YAML parser with stack analysis
      debugging/
        __init__.py               ← Public debugging API
        diagnostic_tools.py       ← Core diagnostic infrastructure
        token_inspector.py        ← Token stream analysis tools
        state_inspector.py        ← Parser state inspection utilities
        error_analyzer.py         ← Error pattern analysis and reporting
        interactive_debugger.py   ← Interactive debugging interface
      testing/
        debug_test_cases.py       ← Test cases specifically for debugging scenarios
        diagnostic_fixtures.py    ← Test fixtures with known debugging patterns
  examples/
    debug_examples/
      problematic_configs/        ← Example files that demonstrate common bugs
      debugging_sessions.py       ← Example debugging workflows
```

#### Core Diagnostic Infrastructure

```python
"""
Core diagnostic infrastructure for configuration parser debugging.
Provides comprehensive state inspection, error analysis, and interactive debugging capabilities.
"""

import json
import traceback
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

class DiagnosticLevel(Enum):
    MINIMAL = "minimal"      # Basic error information only
    STANDARD = "standard"    # Include context and suggestions
    COMPREHENSIVE = "comprehensive"  # Full state inspection and traces
    INTERACTIVE = "interactive"      # Enable interactive debugging features

@dataclass
class DiagnosticConfig:
    level: DiagnosticLevel = DiagnosticLevel.STANDARD
    include_token_stream: bool = True
    include_parse_tree: bool = False
    include_position_context: bool = True
    max_context_lines: int = 3
    enable_color_output: bool = True
    save_diagnostic_traces: bool = False
    trace_output_path: Optional[str] = None

class ParserDiagnostics:
    """
    Comprehensive diagnostic system for parser debugging and analysis.
    Integrates with all parser components to provide detailed inspection capabilities.
    """
    
    def __init__(self, config: DiagnosticConfig = None):
        self.config = config or DiagnosticConfig()
        self.diagnostic_data = {}
        self.error_patterns = []
        self.performance_metrics = {}
        
    def capture_parsing_session(self, content: str, format_hint: str = None) -> 'ParsingSession':
        """
        Create a comprehensive diagnostic session for parsing the given content.
        Returns a session object that tracks all parsing operations and state changes.
        """
        # TODO 1: Initialize session with input content and configuration
        # TODO 2: Set up diagnostic hooks for tokenizer, parser, and error handling
        # TODO 3: Enable state capture at each major parsing phase
        # TODO 4: Configure error tracking and recovery decision logging
        # TODO 5: Return configured session ready for parsing execution
        pass
    
    def analyze_tokenization_issues(self, content: str, expected_tokens: List[str] = None) -> 'TokenizationAnalysis':
        """
        Perform detailed analysis of tokenization behavior for diagnostic purposes.
        Identifies boundary issues, type classification problems, and context sensitivity bugs.
        """
        # TODO 1: Create tokenizer with full diagnostic logging enabled
        # TODO 2: Process content character-by-character with state tracking
        # TODO 3: Analyze token boundaries and type classification decisions
        # TODO 4: Compare against expected tokens if provided
        # TODO 5: Generate comprehensive analysis report with identified issues
        pass
    
    def inspect_parser_state(self, parser_instance: Any, checkpoint_name: str = None) -> Dict[str, Any]:
        """
        Capture complete parser state snapshot for detailed inspection.
        Includes symbol tables, indentation stacks, and all context information.
        """
        # TODO 1: Extract current position and context from parser
        # TODO 2: Serialize symbol table state (for TOML parser)
        # TODO 3: Capture indentation stack (for YAML parser)
        # TODO 4: Include token stream position and lookahead state
        # TODO 5: Format state information for human-readable inspection
        pass
    
    def trace_error_propagation(self, error: ParseError) -> 'ErrorTrace':
        """
        Analyze how errors propagate through parser components.
        Identifies root causes and distinguishes primary errors from cascading symptoms.
        """
        # TODO 1: Analyze error context and position information
        # TODO 2: Trace backwards through parsing decisions leading to error
        # TODO 3: Identify related errors that may stem from same root cause
        # TODO 4: Classify error as primary failure vs cascading symptom
        # TODO 5: Generate trace showing error evolution and propagation path
        pass

class TokenizationAnalysis:
    """Analysis results for tokenization diagnostic operations."""
    
    def __init__(self):
        self.character_analysis = []
        self.token_boundaries = []
        self.type_classification_issues = []
        self.context_sensitivity_violations = []
        self.position_tracking_errors = []
    
    def generate_report(self) -> str:
        """Generate comprehensive tokenization analysis report."""
        # TODO 1: Summarize character-level issues found
        # TODO 2: Detail token boundary problems with examples
        # TODO 3: List type classification errors with corrections
        # TODO 4: Highlight context sensitivity violations
        # TODO 5: Format as readable report with recommendations
        pass

class ParsingSession:
    """
    Comprehensive diagnostic session that tracks all aspects of a parsing operation.
    Provides detailed visibility into parser behavior for debugging purposes.
    """
    
    def __init__(self, content: str, config: DiagnosticConfig):
        self.content = content
        self.config = config
        self.timeline = []  # Chronological record of parsing operations
        self.state_snapshots = {}  # State captures at key points
        self.error_history = []  # Complete error tracking
        self.performance_data = {}  # Timing and resource usage
        
    def execute_parsing(self, parser_class, **parser_options) -> Dict[str, Any]:
        """
        Execute parsing with full diagnostic tracking enabled.
        Returns both parsing results and comprehensive diagnostic information.
        """
        # TODO 1: Initialize parser with diagnostic hooks enabled
        # TODO 2: Track parsing timeline with timestamps
        # TODO 3: Capture state snapshots at major parsing phases
        # TODO 4: Record all errors and recovery decisions
        # TODO 5: Generate final diagnostic report with recommendations
        pass
    
    def analyze_failure_points(self) -> List['FailureAnalysis']:
        """
        Identify specific points where parsing failed or made incorrect decisions.
        Provides targeted analysis for debugging specific issues.
        """
        # TODO 1: Examine error history for failure patterns
        # TODO 2: Identify decision points that led to incorrect results
        # TODO 3: Analyze context conditions at each failure point
        # TODO 4: Generate specific recommendations for each identified issue
        # TODO 5: Prioritize failure points by impact and fixing difficulty
        pass
```

#### Token Stream Analysis Tools

```python
"""
Specialized tools for analyzing token stream behavior and identifying tokenization issues.
"""

class TokenStreamInspector:
    """
    Comprehensive token stream analysis with boundary verification and context tracking.
    """
    
    def __init__(self, tokenizer_class):
        self.tokenizer_class = tokenizer_class
        
    def analyze_token_boundaries(self, content: str) -> 'BoundaryAnalysis':
        """
        Verify token boundary detection accuracy by comparing extracted tokens
        against original content positions.
        """
        # TODO 1: Tokenize content with position tracking enabled
        # TODO 2: Reconstruct original text from token raw_text and positions
        # TODO 3: Identify gaps or overlaps in token coverage
        # TODO 4: Validate position calculations against actual character positions
        # TODO 5: Generate boundary accuracy report with specific issues highlighted
        pass
    
    def compare_tokenization_contexts(self, test_cases: List[str]) -> 'ContextComparisonAnalysis':
        """
        Compare tokenization behavior across different contexts to identify
        context sensitivity bugs where equivalent content tokenizes differently.
        """
        # TODO 1: Process each test case through tokenizer
        # TODO 2: Identify equivalent content patterns across test cases
        # TODO 3: Compare token classification for equivalent patterns
        # TODO 4: Flag inconsistencies in context-sensitive interpretation
        # TODO 5: Generate comparison report highlighting inconsistent behavior
        pass
    
    def validate_string_literal_handling(self, test_strings: List[str]) -> 'StringHandlingAnalysis':
        """
        Comprehensive testing of string literal processing including escape sequences,
        multiline handling, and quote character processing.
        """
        # TODO 1: Test each string format supported by parser
        # TODO 2: Verify escape sequence processing accuracy
        # TODO 3: Validate multiline string boundary detection
        # TODO 4: Test quote character handling and nesting
        # TODO 5: Generate report on string processing capabilities and limitations
        pass

class InteractiveTokenDebugger:
    """
    Interactive debugging interface for step-by-step token stream analysis.
    """
    
    def __init__(self, content: str, tokenizer_class):
        self.content = content
        self.tokenizer = tokenizer_class(content)
        self.current_position = 0
        
    def start_interactive_session(self):
        """
        Launch interactive debugging session with command-line interface.
        Supports stepping through tokenization and state inspection.
        """
        # TODO 1: Initialize tokenizer with diagnostic mode enabled
        # TODO 2: Set up command processing loop
        # TODO 3: Implement commands: next, peek, inspect, context, reset
        # TODO 4: Provide help system and command completion
        # TODO 5: Enable state inspection and modification during session
        pass
    
    def step_to_next_token(self) -> Token:
        """
        Advance tokenizer by one token and display detailed processing information.
        Shows character consumption, state changes, and decision logic.
        """
        # TODO 1: Capture tokenizer state before processing
        # TODO 2: Advance tokenizer by one token
        # TODO 3: Display character consumption and boundary detection
        # TODO 4: Show state machine transitions and decision points
        # TODO 5: Present token result with classification rationale
        pass
```

#### Parser State Inspection Framework

```python
"""
Parser state inspection utilities for examining internal parser condition
during complex parsing operations.
"""

class TOMLParserInspector:
    """Specialized inspection tools for TOML parser state and symbol table analysis."""
    
    def __init__(self, parser_instance):
        self.parser = parser_instance
        
    def inspect_symbol_table(self) -> Dict[str, Any]:
        """
        Generate comprehensive view of current symbol table state including
        all definitions, their types, and conflict detection status.
        """
        # TODO 1: Extract all registered definitions from symbol table
        # TODO 2: Organize definitions by key path and definition type
        # TODO 3: Identify potential conflicts and redefinition violations
        # TODO 4: Show implicit table creation history
        # TODO 5: Format as hierarchical view showing definition relationships
        pass
    
    def analyze_table_conflicts(self, table_path: List[str]) -> 'ConflictAnalysis':
        """
        Analyze potential conflicts for a given table path and explain
        TOML redefinition rules in the context of current definitions.
        """
        # TODO 1: Examine existing definitions that overlap with table_path
        # TODO 2: Apply TOML redefinition rules to identify conflicts
        # TODO 3: Explain why conflicts exist with reference to specification
        # TODO 4: Suggest alternative approaches that avoid conflicts
        # TODO 5: Generate educational analysis explaining TOML semantics
        pass
    
    def trace_dotted_key_expansion(self, key_path: List[str]) -> 'ExpansionTrace':
        """
        Trace the process of expanding dotted keys into nested structure,
        showing intermediate table creation and final value assignment.
        """
        # TODO 1: Simulate dotted key expansion process step by step
        # TODO 2: Show intermediate table creation decisions
        # TODO 3: Identify implicit vs explicit table creation
        # TODO 4: Validate final structure matches expected nesting
        # TODO 5: Generate trace showing complete expansion process
        pass

class YAMLParserInspector:
    """Specialized inspection tools for YAML parser indentation stack and type inference."""
    
    def __init__(self, parser_instance):
        self.parser = parser_instance
        
    def inspect_indentation_stack(self) -> Dict[str, Any]:
        """
        Provide detailed view of current indentation stack state including
        all frames, established levels, and nesting context.
        """
        # TODO 1: Extract current indentation stack frames
        # TODO 2: Show established indentation levels and their contexts
        # TODO 3: Display structure types and data at each level
        # TODO 4: Highlight current frame and processing context
        # TODO 5: Format as visual stack representation with level indicators
        pass
    
    def analyze_indentation_transition(self, target_level: int) -> 'TransitionAnalysis':
        """
        Analyze what stack operations would be required to transition
        to target indentation level and validate transition correctness.
        """
        # TODO 1: Calculate required stack operations for target level
        # TODO 2: Validate target level against established level history
        # TODO 3: Identify ambiguous transitions that could be interpreted multiple ways
        # TODO 4: Show stack state before and after transition
        # TODO 5: Generate analysis explaining transition logic and potential issues
        pass
    
    def trace_type_inference(self, value_string: str) -> 'TypeInferenceTrace':
        """
        Trace type inference process for scalar values showing
        detection logic and conversion decisions.
        """
        # TODO 1: Apply each type detection rule to value_string
        # TODO 2: Show rule matching process and precedence handling
        # TODO 3: Explain conversion logic and edge case handling
        # TODO 4: Compare against YAML specification requirements
        # TODO 5: Generate trace showing complete inference and conversion process
        pass
```

#### Milestone Checkpoints

**Milestone 1 - Basic Diagnostic Infrastructure**: After implementing core diagnostic classes, verify functionality:
- Run `python -m parser.debugging.diagnostic_tools --test-basic` to validate diagnostic capture
- Test state inspection with simple INI content: capture should show section processing and key-value parsing
- Verify error tracking works by parsing invalid content and checking error accumulation

**Milestone 2 - Token Stream Analysis**: After implementing tokenization analysis tools:
- Test boundary analysis with complex strings containing quotes and escape sequences
- Verify context comparison identifies inconsistent tokenization behavior
- Check that position tracking validation catches off-by-one errors in line/column calculation

**Milestone 3 - Interactive Debugging**: After implementing interactive debugging interface:
- Launch interactive session and step through tokenization of sample content
- Verify state inspection shows accurate tokenizer and parser state at each step
- Test command completion and help system provide useful guidance

**Milestone 4 - Format-Specific Inspection**: After implementing TOML and YAML inspectors:
- Verify symbol table inspection shows all definitions and conflicts for complex TOML content
- Test indentation stack analysis correctly tracks YAML nesting transitions
- Confirm type inference tracing explains scalar conversion decisions accurately

#### Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Diagnostic tools crash on valid input | Exception in diagnostic code path | Enable exception logging in diagnostic framework | Add error handling to diagnostic methods |
| State inspection shows empty data | Diagnostic hooks not properly integrated | Verify diagnostic capture points in parser code | Add diagnostic calls at key parsing phases |
| Interactive debugger commands fail | Command parsing or execution errors | Test command parsing logic with simple inputs | Implement robust command validation and error reporting |
| Token boundary analysis reports false positives | Position calculation differences | Compare manual position calculation with automated results | Align position calculation methods between tools and parser |
| Performance degradation with diagnostics enabled | Excessive diagnostic overhead | Profile diagnostic operations vs parsing operations | Optimize diagnostic data collection and disable expensive operations in production |


## Future Extensions

> **Milestone(s):** All milestones (INI Parser, TOML Tokenizer, TOML Parser, YAML Subset Parser) - this section defines how the completed architecture can be extended with new formats, advanced features, and performance optimizations

The configuration file parser architecture we've designed provides a solid foundation that can grow with evolving requirements. Think of our current implementation as a well-designed foundation for a building — the structural elements are in place to support additional floors, rooms, and capabilities without requiring reconstruction of the core infrastructure. This extensibility comes from our careful separation of concerns, standardized interfaces, and unified data model that abstracts away format-specific details.

The extension strategies we'll explore fall into three categories: adding support for new configuration formats that follow similar parsing paradigms, integrating advanced features that enhance the parser's capabilities beyond basic configuration reading, and implementing performance optimizations that enable the parser to handle larger files and higher throughput scenarios. Each category requires different architectural considerations and presents unique challenges in maintaining backward compatibility while adding new functionality.

Our design's extensibility stems from several key architectural decisions. The `BaseTokenizer` abstraction provides a consistent interface that new format tokenizers can implement, while the `ParseContext` mechanism ensures that format-specific parsers can access shared functionality like error reporting and position tracking. The unified output format using nested dictionaries means that new formats only need to map their structures to this common representation, rather than requiring changes throughout the system.

### Adding New Configuration Formats

The architecture's modular design makes adding new configuration formats a straightforward process that follows established patterns. Consider the task of adding JSON5 support — JSON5 extends standard JSON with comments, trailing commas, and more flexible string literals. This format fits naturally into our tokenization-then-parsing pipeline because it shares structural similarities with existing formats while introducing format-specific syntactic features.

**Mental Model for Format Extension**: Think of adding a new format like adding a new language translator to an international conference. The conference already has the infrastructure (microphones, translation booths, audience seating), standardized protocols (when speakers talk, how questions are handled), and a common output format (simultaneous translation in multiple languages). Adding a new language requires creating a translator who understands that specific language but follows the same protocols and produces output in the same standardized format as existing translators.

The format extension process follows a predictable pattern that leverages our existing infrastructure. New formats require implementing a tokenizer that extends `BaseTokenizer` and produces tokens using our standardized `TokenType` enumeration. The format-specific parser must implement the unified `parse(content) -> Dict[str, Any]` interface and integrate with our `ParseContext` system for consistent error reporting and position tracking.

> **Decision: Standardized Format Registration System**
> - **Context**: New formats need to integrate with format detection and parser factory mechanisms
> - **Options Considered**: Static registration at compile time, dynamic registration via plugin system, configuration-driven format discovery
> - **Decision**: Static registration with factory method pattern
> - **Rationale**: Maintains type safety, enables compile-time verification, simplifies deployment while still allowing conditional compilation of format support
> - **Consequences**: Adding formats requires code changes but provides stronger guarantees about parser availability and reduces runtime complexity

The format registration system uses a factory pattern that maps format identifiers to parser creation functions. Each new format registers itself with the central `ConfigurationFormatRegistry` that provides format detection hints and parser instantiation services. This approach ensures that format detection logic remains centralized while allowing format-specific parsers to provide their own syntactic signatures for automatic detection.

| Format Extension Component | Responsibilities | Interface Requirements |
|---------------------------|------------------|----------------------|
| Format Tokenizer | Break character stream into format-specific tokens | Extend `BaseTokenizer`, implement `tokenize() -> List[Token]` |
| Format Parser | Convert tokens to unified data structure | Implement `parse(content) -> Dict[str, Any]` |
| Format Detector | Provide syntactic signatures for format identification | Implement `get_format_signatures() -> List[str]` |
| Format Registry Entry | Register format with detection and factory systems | Provide `create_parser(**options)` factory function |

**JSON5 Extension Example**: JSON5 support requires extending our tokenization capabilities to handle JavaScript-style comments and relaxed string literal syntax. The JSON5 tokenizer would recognize `//` and `/* */` comment styles, producing `COMMENT` tokens that the parser ignores. Trailing commas in arrays and objects require lookahead logic to distinguish between legitimate commas and trailing syntax. The parser implementation leverages our existing recursive descent patterns but adds format-specific handling for these syntactic extensions.

The JSON5 format detection relies on JavaScript-specific syntactic markers like unquoted object keys, single-quoted strings, and hexadecimal number literals. These signatures have high specificity — they're unlikely to appear in INI, TOML, or YAML files — making format detection reliable. The parser maps JSON5 structures directly to our nested dictionary representation since JSON5's object-array model aligns perfectly with our unified output format.

**HCL (HashiCorp Configuration Language) Extension**: HCL presents interesting challenges because it supports both JSON-compatible syntax and a more human-readable block-based syntax. The HCL tokenizer must handle this syntactic duality, potentially operating in different modes based on detected syntax style. HCL's block syntax requires parsing constructs like `resource "aws_instance" "example" { ... }` which don't map directly to simple key-value pairs.

The HCL parser addresses this complexity by treating blocks as nested dictionary structures where block types become keys and block labels become nested keys. For example, the resource block above creates a nested structure: `{"resource": {"aws_instance": {"example": { ... }}}}`. This mapping preserves HCL's semantic meaning while fitting into our unified output format.

> **Critical Design Insight**: Format extensions succeed when they embrace our unified data model rather than fighting it. Formats that map naturally to nested dictionaries integrate smoothly, while formats requiring fundamentally different output structures may indicate the need for architectural evolution rather than simple extension.

**Advanced Format Considerations**: Some configuration formats present challenges that push the boundaries of our current architecture. Formats with include mechanisms, template engines, or dynamic evaluation require extensions to our parsing pipeline. These advanced formats might require a two-phase parsing approach: first parsing the static structure using existing mechanisms, then post-processing for dynamic features.

Formats with schema requirements or validation constraints may benefit from integration with our planned schema validation extensions. The format extension architecture should anticipate these needs by providing hooks for post-parse processing and validation phases that can be optionally enabled based on format requirements and user preferences.

| Extension Challenge | Current Architecture Impact | Recommended Approach |
|-------------------|---------------------------|---------------------|
| Include file support | Requires file system access during parsing | Add optional `FileResolver` component to `ParseContext` |
| Template/variable expansion | Single-pass parsing insufficient | Implement two-phase parsing with post-processing stage |
| Schema validation | No built-in validation framework | Design validation extension points in parser interface |
| Custom data types | Limited to basic Python types | Extend value processing with pluggable type converters |

### Advanced Feature Extensions

The parser architecture provides several extension points for advanced features that enhance functionality beyond basic configuration file reading. These extensions transform the parser from a simple data extraction tool into a comprehensive configuration management system that can handle complex real-world requirements like schema validation, variable interpolation, and modular configuration organization.

**Schema Validation Extension**: Configuration files often require validation against predefined schemas to ensure they contain required fields, use acceptable value ranges, and maintain structural consistency. Think of schema validation like a quality control inspector at a manufacturing plant — it examines each component (configuration section) against predetermined specifications (schema rules) and flags any deviations before the product (configuration data) moves to the next stage (application startup).

The schema validation extension integrates with our parsing pipeline as a post-processing stage that operates on the unified dictionary output. This design choice preserves the separation between syntactic parsing and semantic validation, allowing schema validation to work consistently across all supported formats. The validation system uses a plugin architecture where different schema languages (JSON Schema, custom validation rules) can be plugged in based on user requirements.

| Schema Validation Component | Purpose | Integration Point |
|---------------------------|---------|-------------------|
| Schema Definition Parser | Parse schema files into validation rules | Standalone component, loads schema before configuration parsing |
| Validation Rule Engine | Apply validation rules to parsed configuration | Post-processing stage after format-specific parsing |
| Error Enrichment System | Add schema context to validation errors | Integrates with existing `ParseError` hierarchy |
| Validation Result Reporter | Generate comprehensive validation reports | Extends existing error reporting mechanisms |

The schema validation system maintains validation context that maps back to original source positions, enabling error messages that reference both the configuration file location and the schema requirement that was violated. For example, a validation error might report: "Line 15: Missing required field 'database.port' (required by schema section 'server-config')" providing both syntactic and semantic context.

**Variable Interpolation Extension**: Modern configuration management often requires dynamic value substitution where configuration values can reference other configuration values, environment variables, or external data sources. Consider variable interpolation as similar to mail merge in document processing — templates contain placeholders that get filled in with actual values from various sources to produce the final document.

The variable interpolation extension operates as a post-processing phase that scans the parsed configuration for interpolation expressions and resolves them using configured value sources. The interpolation engine supports multiple expression syntaxes (${var}, {{var}}, #{var}) to accommodate different format conventions and user preferences. Resolution occurs in dependency order, ensuring that variables are resolved before being used in other variable definitions.

> **Decision: Two-Phase Interpolation Processing**
> - **Context**: Variable interpolation requires dependency resolution and may involve external data sources
> - **Options Considered**: Single-pass interpolation during parsing, post-processing after parsing, lazy evaluation during configuration access
> - **Decision**: Post-processing with dependency graph resolution
> - **Rationale**: Separates concerns, enables cross-format interpolation, allows dependency cycle detection, supports external value sources
> - **Consequences**: Requires additional processing phase but provides more powerful and reliable interpolation capabilities

The interpolation system builds a dependency graph of variable references and performs topological sorting to determine resolution order. Circular dependencies are detected and reported as configuration errors with suggestions for restructuring. The system supports multiple value sources including configuration values, environment variables, file contents, and external service calls through a pluggable provider interface.

**Include File Extension**: Large configuration systems benefit from modular organization where configuration can be split across multiple files and composed at parse time. Think of include file support like a compiler's include mechanism — individual source files can reference other files, and the compilation process assembles them into a complete program.

The include file extension requires coordination between file system access and parsing operations. Include directives are recognized during parsing and trigger recursive parsing of referenced files. The included content is merged into the parent configuration using format-appropriate rules — INI sections merge, TOML tables merge with conflict detection, and YAML structures merge preserving hierarchy.

| Include Processing Component | Responsibilities | Implementation Considerations |
|---------------------------|------------------|------------------------------|
| Include Directive Detection | Recognize format-specific include syntax | Integrates with format-specific parsers |
| File Resolution System | Resolve relative paths and search paths | Handles file system access and path normalization |
| Recursive Parse Coordination | Manage parsing of included files | Prevents infinite recursion, maintains parse context |
| Configuration Merging Logic | Combine included content with parent configuration | Format-specific merging rules with conflict detection |

Include processing maintains a stack of currently processing files to detect and prevent circular includes. Error reporting preserves the include chain so users can trace errors back through the file inclusion hierarchy. For example: "Error in config.toml (included from main.toml:15): Invalid table definition at line 8".

**Configuration Template Extensions**: Some deployment scenarios benefit from template-based configuration generation where configuration files are generated from templates with environment-specific values. This extension transforms the parser into a configuration generation system that can produce format-specific output from template definitions.

The template extension operates in reverse from normal parsing — instead of reading configuration files and producing data structures, it takes data structures and template definitions to produce configuration files in specific formats. This bidirectional capability enables configuration round-tripping and format conversion workflows.

**Advanced Type System Extensions**: The current parser supports basic data types appropriate for configuration files, but some applications require more sophisticated type handling including custom objects, validated enumerations, and computed values. The advanced type system extension provides pluggable type converters that can transform string literals into domain-specific objects during parsing.

Custom type converters register with the value processing system and provide type detection patterns and conversion logic. For example, a duration converter might recognize strings like "30m", "2h45m", "1d" and convert them to appropriate duration objects. The type system maintains type metadata that can be used by schema validation and error reporting systems.

### Performance and Scalability Improvements

As configuration files grow larger and parsing becomes more frequent, performance optimizations become essential for maintaining responsive applications. The current parser architecture provides several optimization opportunities that can dramatically improve performance for large-scale usage scenarios without compromising correctness or maintainability.

**Streaming Parser Implementation**: Large configuration files can cause memory pressure when the entire file is loaded into memory for parsing. Think of streaming parsing like reading a book one page at a time instead of memorizing the entire book before understanding its content — you can begin processing and understanding information as soon as you receive it, rather than waiting for complete input.

The streaming parser extension modifies our tokenization and parsing pipeline to operate on character streams rather than complete string content. This approach enables processing configuration files that exceed available memory and reduces startup latency for applications that only need specific configuration sections.

| Streaming Component | Current Implementation | Streaming Enhancement |
|--------------------|----------------------|----------------------|
| Input Processing | Load complete file into memory string | Process character stream with buffered reading |
| Tokenization | Generate complete token list | Yield tokens on-demand with lookahead buffer |
| Parsing | Random access to token stream | Forward-only parsing with limited backtracking |
| Output Generation | Build complete result dictionary | Incremental result building with optional section filtering |

Streaming parsing requires careful consideration of lookahead requirements. TOML's complex syntax occasionally requires significant lookahead to disambiguate constructs, while YAML's indentation sensitivity requires maintaining context about previous lines. The streaming implementation uses bounded buffers that balance memory usage with parsing capability.

**Incremental Parsing for Configuration Updates**: Applications that monitor configuration files for changes benefit from incremental parsing that can update the parsed representation without re-parsing unchanged sections. Consider incremental parsing similar to smart document editing software that only reformats the paragraph you're currently editing rather than reformatting the entire document with every keystroke.

The incremental parsing extension requires maintaining parse tree metadata that maps sections of the output structure back to source file regions. When file changes are detected, the system identifies affected regions and re-parses only those sections, merging updated content with the cached parse results. This approach dramatically reduces parsing overhead for configuration hot-reloading scenarios.

> **Decision: Section-Level Incremental Granularity**
> - **Context**: Incremental parsing requires balancing granularity with complexity
> - **Options Considered**: Line-level incremental updates, section-level updates, full file re-parsing with caching
> - **Decision**: Section-level incremental updates with dependency tracking
> - **Rationale**: Provides significant performance benefits while maintaining manageable complexity, aligns with natural configuration organization boundaries
> - **Consequences**: Requires section dependency tracking but offers substantial performance improvements for large configuration files

Incremental parsing maintains dependency information between configuration sections so that changes in one section can trigger re-parsing of dependent sections. For example, changing a TOML table definition might require re-parsing sections that reference that table through dotted key notation.

**Parse Result Caching System**: Applications that parse the same configuration files repeatedly (such as microservices that restart frequently) benefit from caching parsed results indexed by file content hash. Think of parse result caching like a translator who keeps a glossary of previously translated phrases — common translations can be recalled instantly rather than re-performing the translation work.

The caching system computes content hashes for configuration files and stores serialized parse results in a cache (memory-based, disk-based, or distributed cache systems). Cache entries include metadata about parsing options and format detection results to ensure cache hits only occur for equivalent parsing scenarios.

**Parallel Parsing for Multi-File Configurations**: Configuration systems with many include files or modular organization can benefit from parallel parsing where independent configuration files are parsed simultaneously. The parallel parsing extension requires careful coordination to maintain dependency order while maximizing concurrency for independent parsing operations.

The parallel parsing coordinator analyzes include dependencies to create a parsing task graph where independent branches can be processed concurrently. Results are combined using the same merging logic as sequential include processing, but with coordination points that ensure dependency order is preserved.

| Performance Extension | Memory Impact | CPU Impact | Complexity Impact | Use Case |
|---------------------|---------------|------------|-------------------|----------|
| Streaming parsing | Significant reduction | Slight increase | Moderate increase | Very large files |
| Incremental parsing | Moderate increase | Significant reduction for updates | High increase | Frequently changing files |
| Result caching | Moderate increase | Significant reduction for repeated parsing | Low increase | Repeated parsing scenarios |
| Parallel parsing | No significant change | Significant reduction | High increase | Multi-file configurations |

**Memory Optimization Strategies**: The current parser creates rich object hierarchies for tokens, parse nodes, and error tracking that provide excellent debugging capabilities but consume significant memory for large configuration files. Memory optimization extensions provide configurable trade-offs between memory usage and debugging capability.

Lightweight parsing modes eliminate intermediate parse trees and error context information, producing only the final dictionary output. Token pooling reuses token objects to reduce garbage collection pressure. String interning reduces memory usage for repeated configuration keys and common values. These optimizations can reduce memory usage by 50-80% for typical configuration files while maintaining parsing correctness.

**Async/Await Integration**: Modern applications increasingly use asynchronous programming models where I/O operations should not block execution threads. The async parsing extension provides asynchronous versions of parsing operations that integrate cleanly with async/await frameworks while maintaining the same functionality and error handling characteristics.

Async parsing is particularly valuable when combined with streaming parsing and include file processing, where file I/O operations can be overlapped with parsing work. The async implementation maintains the same error handling and progress reporting capabilities while enabling better resource utilization in async applications.

### Implementation Guidance

**Technology Recommendations for Extensions:**

| Extension Category | Simple Option | Advanced Option |
|------------------|---------------|----------------|
| New Format Support | Direct parser implementation | Plugin architecture with dynamic loading |
| Schema Validation | JSON Schema library integration | Custom validation engine with DSL |
| Variable Interpolation | String template substitution | Expression evaluator with function support |
| Include Files | Recursive parsing with simple merging | Dependency graph with conflict resolution |
| Streaming Parsing | Generator-based token production | Coroutine-based parser with backtracking |
| Caching | In-memory dictionary cache | Redis/disk-based cache with TTL |

**Recommended Extension Structure:**
```
project-root/
  config_parser/
    core/                      ← existing core components
      tokenizer.py
      parsers/
        ini_parser.py
        toml_parser.py
        yaml_parser.py
    extensions/                ← extension system
      __init__.py             ← extension registry
      formats/                ← new format support
        json5_parser.py
        hcl_parser.py
        format_registry.py
      features/               ← advanced features
        schema_validator.py
        variable_interpolator.py
        include_processor.py
      performance/            ← performance extensions
        streaming_parser.py
        incremental_parser.py
        cache_manager.py
    examples/                 ← extension examples
      custom_format_example.py
      validation_example.py
```

**Extension Registry Infrastructure (Complete Implementation):**

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

class ExtensionType(Enum):
    FORMAT_PARSER = "format_parser"
    FEATURE_PROCESSOR = "feature_processor"
    PERFORMANCE_OPTIMIZER = "performance_optimizer"

@dataclass
class ExtensionInfo:
    name: str
    extension_type: ExtensionType
    version: str
    dependencies: List[str]
    factory: Callable[..., Any]
    config_schema: Optional[Dict[str, Any]] = None

class ConfigurationExtensionRegistry:
    def __init__(self):
        self._extensions: Dict[str, ExtensionInfo] = {}
        self._format_detectors: Dict[str, Callable[[str], float]] = {}
        
    def register_extension(self, extension_info: ExtensionInfo) -> None:
        """Register a new extension with the configuration system."""
        self._validate_dependencies(extension_info.dependencies)
        self._extensions[extension_info.name] = extension_info
        
        if extension_info.extension_type == ExtensionType.FORMAT_PARSER:
            # Format parsers should provide detection capabilities
            detector = getattr(extension_info.factory(), 'detect_format_confidence', None)
            if detector:
                self._format_detectors[extension_info.name] = detector
    
    def get_extension(self, name: str) -> Optional[ExtensionInfo]:
        return self._extensions.get(name)
    
    def create_parser(self, format_name: str, **options) -> Optional[Any]:
        extension = self.get_extension(format_name)
        if extension and extension.extension_type == ExtensionType.FORMAT_PARSER:
            return extension.factory(**options)
        return None
    
    def detect_format(self, content: str) -> Optional[str]:
        best_format = None
        best_confidence = 0.0
        
        for format_name, detector in self._format_detectors.items():
            confidence = detector(content)
            if confidence > best_confidence:
                best_confidence = confidence
                best_format = format_name
        
        return best_format if best_confidence > 0.7 else None

# Global registry instance
extension_registry = ConfigurationExtensionRegistry()
```

**Format Extension Skeleton (JSON5 Example):**

```python
from typing import Dict, Any, List
from config_parser.core.tokenizer import BaseTokenizer, Token, TokenType
from config_parser.core.base_parser import BaseParser

class JSON5Tokenizer(BaseTokenizer):
    def __init__(self, source: str):
        super().__init__(source)
        # TODO 1: Initialize JSON5-specific tokenization state
        # TODO 2: Set up comment recognition patterns (// and /* */)
        # TODO 3: Configure string literal handling for single quotes
    
    def tokenize(self) -> List[Token]:
        # TODO 1: Implement main tokenization loop
        # TODO 2: Handle JavaScript-style comments
        # TODO 3: Process unquoted object keys
        # TODO 4: Handle trailing commas in arrays/objects
        # TODO 5: Support hexadecimal number literals
        pass

class JSON5Parser(BaseParser):
    def __init__(self, **options):
        super().__init__()
        self.tokenizer = None
        # TODO 1: Initialize parser state for JSON5 syntax
        # TODO 2: Set up recursive descent parsing methods
    
    def parse(self, content: str) -> Dict[str, Any]:
        # TODO 1: Create JSON5 tokenizer instance
        # TODO 2: Tokenize input content
        # TODO 3: Parse JSON5 structure using recursive descent
        # TODO 4: Convert to unified dictionary format
        # TODO 5: Handle parsing errors with appropriate context
        pass
    
    @staticmethod
    def detect_format_confidence(content: str) -> float:
        # TODO 1: Look for JavaScript-style comments
        # TODO 2: Check for unquoted object keys
        # TODO 3: Detect trailing commas
        # TODO 4: Return confidence score (0.0 - 1.0)
        pass

# Registration with extension system
json5_extension = ExtensionInfo(
    name="json5",
    extension_type=ExtensionType.FORMAT_PARSER,
    version="1.0.0",
    dependencies=[],
    factory=JSON5Parser
)
extension_registry.register_extension(json5_extension)
```

**Schema Validation Extension Skeleton:**

```python
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from config_parser.core.errors import ParseError

class ValidationRule(ABC):
    @abstractmethod
    def validate(self, value: Any, path: str, context: Dict[str, Any]) -> List[str]:
        """Return list of validation error messages, empty if valid."""
        pass

class SchemaValidator:
    def __init__(self, schema_definition: Dict[str, Any]):
        self.schema = schema_definition
        self.rules: Dict[str, ValidationRule] = {}
        # TODO 1: Parse schema definition into validation rules
        # TODO 2: Build rule hierarchy for nested validation
        # TODO 3: Set up error message templates
    
    def validate_configuration(self, config: Dict[str, Any]) -> List[ParseError]:
        # TODO 1: Traverse configuration structure
        # TODO 2: Apply appropriate validation rules at each level
        # TODO 3: Collect validation errors with path context
        # TODO 4: Generate helpful error messages with suggestions
        # TODO 5: Return list of validation errors
        pass
    
    def add_custom_rule(self, path: str, rule: ValidationRule):
        # TODO 1: Register custom validation rule for specific configuration path
        # TODO 2: Integrate with existing rule hierarchy
        pass

# Integration point for adding validation to parsing pipeline
def parse_with_validation(content: str, schema_path: Optional[str] = None, 
                         format_hint: Optional[str] = None) -> Dict[str, Any]:
    # TODO 1: Parse configuration using standard pipeline
    # TODO 2: Load schema definition if provided
    # TODO 3: Create validator instance with schema
    # TODO 4: Run validation on parsed configuration
    # TODO 5: Report validation errors alongside parsing errors
    pass
```

**Performance Extension Checkpoints:**

After implementing streaming parsing:
- Test with configuration files larger than available memory (generate test files > 1GB)
- Verify memory usage remains constant regardless of file size: `python -m memory_profiler your_streaming_test.py`
- Expected behavior: Memory usage should plateau at buffer size rather than growing with file size
- Performance benchmark: Parse 100MB configuration file in under 30 seconds with less than 50MB memory usage

After implementing incremental parsing:
- Create test scenario with 1000-section configuration file
- Modify single section and measure re-parse time
- Expected behavior: Re-parse time should be < 5% of full parse time for single section changes
- Verify correctness: Output should be identical to full re-parse for all test cases

After implementing caching system:
- Test cache hit rates with repeated parsing of same content
- Expected behavior: Cache hit should be > 100x faster than full parsing
- Verify cache invalidation: Content changes should produce cache misses
- Test command: `python -m pytest tests/performance/test_caching.py -v --benchmark-only`

**Common Extension Pitfalls:**

⚠️ **Pitfall: Breaking Unified Output Format**
New format parsers sometimes introduce format-specific data types in the output dictionary, breaking compatibility with existing code that expects standard Python types. Always convert format-specific types to basic Python types (dict, list, str, int, float, bool) in the final output.

⚠️ **Pitfall: Inconsistent Error Reporting**
Extension components may implement their own error handling that doesn't integrate with the existing `ParseError` hierarchy, leading to inconsistent error messages and loss of source position information. Always use the established error classes and ensure position information propagates correctly.

⚠️ **Pitfall: Memory Leaks in Streaming Parsers**
Streaming parsers can accumulate state that isn't properly cleaned up, especially when parsing fails partway through large files. Implement proper cleanup in finally blocks and consider using context managers for resource management.

⚠️ **Pitfall: Schema Validation Performance**
Complex schema validation can become a performance bottleneck, especially with deeply nested configurations. Profile validation performance and consider implementing validation rule caching or lazy validation strategies for large configurations.


## Glossary

> **Milestone(s):** All milestones (INI Parser, TOML Tokenizer, TOML Parser, YAML Subset Parser) - comprehensive terminology reference supporting all implementation phases

A comprehensive configuration file parser requires precise vocabulary to communicate complex parsing concepts effectively. This glossary serves as the authoritative reference for all technical terms, parsing concepts, and domain-specific vocabulary used throughout the design document. Understanding these terms is essential for implementing the parser architecture and communicating about parsing challenges.

The terminology is organized into logical categories to support different aspects of the learning journey. Each definition includes context about where the term appears in the implementation pipeline and why it matters for configuration parsing specifically.

### Parsing Fundamentals

**Tokenization** refers to the process of breaking a character stream into meaningful lexical units called tokens. Think of tokenization as converting a continuous stream of characters into a sequence of building blocks that the parser can understand and manipulate. Each token represents a syntactic element like a string, number, operator, or delimiter that carries semantic meaning within the configuration format.

**Recursive descent** describes a parsing methodology where the parser uses function calls to handle nested grammatical structures. Each function in the parser corresponds to a grammatical rule, and when the parser encounters nested content, it calls the appropriate function recursively. This approach naturally handles the hierarchical structure found in configuration files, where tables can contain subtables, arrays can contain other arrays, and values can be complex nested structures.

**Lookahead parsing** involves examining upcoming tokens in the input stream before making parsing decisions. The parser maintains the ability to inspect future tokens without consuming them, allowing it to choose the correct parsing path based on upcoming context. This technique is crucial for resolving syntactic ambiguities and implementing robust error recovery.

**Parse tree** represents the intermediate structural representation of parsed syntax before conversion to the final data structure. The parse tree captures the grammatical structure of the input, maintaining information about how the parser interpreted each syntactic element. This intermediate representation allows for validation, transformation, and debugging before generating the unified output format.

**Unified output format** describes the consistent nested dictionary structure that all parsers produce regardless of input format. This standardization allows applications to work with configuration data uniformly, regardless of whether the source was INI, TOML, or YAML. The unified format uses Python dictionaries and lists to represent all hierarchical structures and data types.

### Tokenization Concepts

**Lexical ambiguity** occurs when the same character sequence can mean different things depending on the parsing context. For example, the sequence `"key"` might be a quoted string literal in one context but part of a larger quoted phrase in another context. Tokenizers must maintain sufficient context to resolve these ambiguities correctly.

**Context sensitivity** describes how the meaning of input changes based on surrounding parsing context. The same character sequence might tokenize as different token types depending on what the parser has already encountered. For instance, a colon character has different meanings in INI key-value pairs versus YAML mapping syntax.

**Whitespace semantics** refers to how different configuration formats treat whitespace as either meaningful or ignorable. INI files generally ignore whitespace around delimiters, while YAML uses indentation as syntactically significant for defining structure. Understanding these semantics is crucial for correct tokenization.

**String literal handling** encompasses the complex logic required to process quoted strings, escape sequences, and multiline variants across different formats. Each format has different rules for string delimiters, escape sequences, and multiline continuation, requiring format-specific tokenization logic.

**Scanning window** describes the tokenizer's moving view through the character stream as it identifies token boundaries. The tokenizer maintains position information and looks ahead through this window to determine where each token begins and ends.

### Data Structure Concepts

| Concept | Description | Usage Context |
|---------|-------------|---------------|
| **Token Type Definitions** | Enumerated values representing all possible token categories across formats | Tokenization phase for classifying lexical units |
| **Parse Node Structure** | Hierarchical representation capturing grammatical relationships | Intermediate parsing phase before final output |
| **Position Tracking** | Line and column information for error reporting and debugging | Throughout tokenization and parsing for diagnostics |
| **Symbol Table Management** | Registry of defined keys and tables for conflict detection | TOML parsing to enforce redefinition rules |
| **Indentation Frame** | Stack element tracking YAML nesting context | YAML parsing for managing hierarchical structure |

### INI Format Concepts

**Section-based organization** describes INI's hierarchical structure where named sections contain key-value pairs. Each section creates a namespace that groups related configuration values, and the parser must track the current section context when processing key-value assignments.

**Line-based parsing** refers to INI's approach of processing input one logical line at a time. The parser examines each line to determine whether it's a section header, key-value pair, comment, or continuation, then processes it according to its type.

**Global keys** are key-value pairs that appear before any section headers in INI files. These keys belong to a default or global namespace and require special handling in the parser's section tracking logic.

**Dotted notation** allows section names to use dot separators to create nested dictionary structures in the output. The parser must expand these dotted sections into hierarchical data structures during processing.

### TOML Format Concepts

**Table redefinition rules** are TOML's constraints preventing conflicting table declarations within the same document. Once a table path is defined explicitly, it cannot be redefined using a different declaration type, and the parser must track all definitions to enforce these rules.

**Array-of-tables** represents TOML's syntax for creating arrays of table instances using double bracket notation `[[table.name]]`. Each array-of-tables declaration creates a new table instance and appends it to an array at the specified path.

**Dotted key expansion** is the automatic creation of nested table structure from dotted key notation like `physical.color = "orange"`. The parser must create intermediate tables as needed while respecting existing table definitions.

**Inline tables** use TOML's single-line table syntax `{key = value, key2 = value2}` to define complete table structures within expressions. These tables cannot be modified after creation and require special parsing logic.

**Implicit table creation** occurs when dotted keys reference table paths that don't exist yet. The parser automatically creates the necessary intermediate table structure while tracking these implicit creations for conflict detection.

**Symbol table management** involves tracking all defined keys and tables throughout TOML parsing to detect conflicts and validate redefinition rules. The symbol table maintains metadata about how each key path was defined and where conflicts might occur.

### YAML Format Concepts

**Indentation-driven hierarchical structure** describes YAML's fundamental approach of using whitespace to define nesting relationships. The amount of indentation determines the hierarchical level of each element, and the parser must maintain strict tracking of indentation levels.

**Stack-based approach** refers to the parsing methodology that uses a stack data structure to track current nesting contexts. As indentation increases, the parser pushes new contexts onto the stack, and as indentation decreases, it pops contexts back to the appropriate level.

**Indentation stack** is the specific stack data structure that tracks current nesting contexts and established indentation levels. The stack contains frames representing each nesting level with information about the expected structure type and data being built.

**Structure transitions** are changes in nesting level that require stack operations to maintain correct parsing context. When the parser encounters different indentation levels, it must push or pop stack frames to match the new structure.

**Scalar type inference** involves automatically converting string literals to appropriate Python data types based on their format and content. YAML's implicit typing system requires the parser to recognize patterns like numbers, booleans, and null values.

**Flow syntax** refers to YAML's JSON-like inline syntax using brackets and braces for lists and mappings. This syntax provides an alternative to block syntax and requires different parsing logic.

**Block syntax** is YAML's primary indentation-based syntax for defining hierarchical structures using whitespace and line breaks rather than explicit delimiters.

### Error Handling Concepts

**Error recovery** describes the parser's ability to continue processing after encountering errors to find additional issues in the same parsing pass. Rather than stopping at the first error, the parser attempts to understand enough context to continue and report multiple problems.

**Panic mode recovery** involves discarding input tokens until reaching a known synchronization point where parsing can safely resume. This strategy helps the parser skip over malformed content and continue processing subsequent sections.

**Error production recovery** inserts assumed content when the parser encounters unexpected input but can infer what was likely intended. This approach allows parsing to continue while recording the assumption made.

**Phrase-level recovery** skips minimal malformed syntactic units while preserving as much surrounding structure as possible. This fine-grained approach minimizes the impact of local errors on overall parsing success.

**Graceful degradation** ensures the system continues operating despite errors, providing reduced functionality rather than complete failure. The parser returns partial results when possible, allowing applications to work with the correctly parsed portions.

**Structured error accumulation** involves collecting multiple related errors for unified reporting rather than stopping at each individual problem. This approach provides comprehensive feedback about all issues in a single parsing pass.

### Architecture Concepts

**Component responsibilities** define what each parser component owns in terms of data, processing, and state management. Clear responsibility boundaries prevent overlap and ensure each component has a well-defined purpose within the overall architecture.

**Nested structure mapping** describes the process of creating hierarchical data structures from various flat syntactic representations. Different formats express hierarchy differently, but all must map to the same nested dictionary output format.

**Format detection** involves automatically identifying configuration format from content analysis without explicit format specification. The system analyzes syntactic patterns and structural characteristics to determine the appropriate parser.

**Impedance mismatch** refers to fundamental incompatibilities between different format paradigms that complicate unified processing. For example, YAML's indentation sensitivity conflicts with INI's line-based approach, requiring different parsing strategies.

### Testing and Debugging Concepts

**Milestone verification points** are concrete checkpoints that confirm successful completion of each implementation milestone. These checkpoints provide measurable criteria for evaluating progress and identifying implementation gaps.

**Golden path test data** represents realistic configuration scenarios that reflect common usage patterns. This test data validates that the parser handles typical use cases correctly and produces expected results.

**Edge case test data** explores boundary conditions and corner cases that might reveal parsing bugs or unexpected behavior. This comprehensive testing approach ensures robust handling of unusual but valid input.

**Cross-format equivalence testing** validates that semantically equivalent configurations produce consistent results regardless of their source format. This testing ensures the unified output format truly provides format independence.

**Layered diagnosis methodology** provides a systematic debugging approach that examines parsing problems at multiple levels: character processing, tokenization, parse tree construction, and final output generation.

**Token stream replay** involves recording and re-processing token sequences for diagnostic purposes. This technique allows detailed analysis of tokenization behavior and helps identify the root causes of parsing failures.

**Interactive parser inspection framework** provides tools for step-by-step visibility into parsing operations. These debugging tools allow developers to examine parser state, token streams, and decision points interactively.

### Performance and Scalability Concepts

**Streaming parsing** refers to processing large configuration files without loading the entire content into memory. This approach enables handling of arbitrarily large files by processing content in chunks as it becomes available.

**Incremental parsing** updates parsed results when only part of a file changes, avoiding complete re-parsing of unchanged content. This optimization is valuable for applications that monitor configuration files for changes.

**Parse result caching** stores previously parsed results to avoid re-parsing unchanged files. The caching system must handle cache invalidation correctly when source files are modified.

**Extension architecture** describes the system design that supports pluggable extensions for new formats, features, and optimizations. The architecture provides well-defined extension points without requiring modifications to core parsing logic.

### Advanced Feature Concepts

**Schema validation** involves validating parsed configuration against predefined rules and constraints. This feature ensures configuration correctness beyond just syntactic validity, checking semantic rules and value constraints.

**Variable interpolation** provides dynamic value substitution in configuration files, allowing references to other configuration values or environment variables. This feature requires additional parsing passes to resolve dependencies correctly.

**Include file processing** supports modular configuration through file inclusion mechanisms. The parser must handle file resolution, circular dependency detection, and scope management for included content.

**Dependency graph resolution** ensures variables and includes are processed in the correct order when complex interdependencies exist. The parser must build and traverse dependency graphs to handle these relationships correctly.

### Implementation Guidance

The implementation of a comprehensive configuration parser requires careful attention to terminology consistency and conceptual clarity. Each term in this glossary represents a specific aspect of parsing theory or implementation practice that has precise meaning within the context of configuration file processing.

When implementing parser components, developers should use this terminology consistently to ensure clear communication about design decisions and implementation challenges. The terms provide a shared vocabulary for discussing parsing problems, architectural choices, and debugging strategies.

The glossary also serves as a reference for understanding the parsing literature and related tools. Many of these terms have broader applications in compiler design and language processing, making them valuable for developers who want to explore more advanced parsing topics.

#### Key Term Categories

The terminology is organized into several key categories that correspond to different aspects of the parsing implementation:

| Category | Focus Area | Key Terms |
|----------|------------|-----------|
| **Parsing Fundamentals** | Core concepts applicable to all parsers | Tokenization, recursive descent, lookahead parsing |
| **Format-Specific Concepts** | Terms specific to INI, TOML, or YAML | Section-based organization, table redefinition, indentation-driven structure |
| **Error Handling** | Error detection, recovery, and reporting | Panic mode recovery, graceful degradation, structured error accumulation |
| **Architecture** | System design and component interaction | Component responsibilities, format detection, impedance mismatch |
| **Testing and Debugging** | Validation and troubleshooting | Milestone verification, cross-format equivalence, layered diagnosis |
| **Advanced Features** | Extensions and optimizations | Schema validation, variable interpolation, streaming parsing |

#### Terminology Evolution

As the parser implementation progresses through different milestones, developers will encounter these terms in specific contexts that reinforce their meanings through practical application. The INI parser milestone introduces fundamental concepts like tokenization and line-based parsing, while the TOML parser milestone adds complexity with recursive descent and symbol table management.

The YAML parser milestone demonstrates how terminology evolves to handle different parsing paradigms, introducing concepts like indentation-driven structure and stack-based approaches. Throughout all milestones, error handling terminology becomes increasingly important as the parser implementations become more sophisticated and robust.

# AST Interpreter Architecture Guide

## Overview

This document provides a comprehensive architecture guide for building an AST (Abstract Syntax Tree) interpreter. The interpreter follows a classic pipeline: **source code** -> **tokens** -> **AST** -> **evaluation**.

> This guide is meant to help you understand the big picture before diving into each milestone. Refer back to it whenever you need context on how components connect.

## System Architecture

The interpreter consists of four main components arranged in a pipeline:

1. **Lexer** (Tokenizer) - Converts source text into tokens
2. **Parser** - Converts tokens into an Abstract Syntax Tree
3. **Evaluator** - Walks the AST and computes results
4. **Environment** - Manages variable scopes and bindings

### Data Flow

```
Source Code (string)
    |
    v
[Lexer] --> Token[]
    |
    v
[Parser] --> AST (tree of nodes)
    |
    v
[Evaluator + Environment] --> Result
```

## Component Details

### Lexer (Tokenizer)

The lexer scans the input string character by character and produces a flat list of tokens.

**Key types:**

```typescript
type TokenType =
  | 'NUMBER' | 'STRING' | 'IDENTIFIER'
  | 'PLUS' | 'MINUS' | 'STAR' | 'SLASH'
  | 'EQUALS' | 'LPAREN' | 'RPAREN'
  | 'LET' | 'IF' | 'ELSE' | 'FN' | 'RETURN'
  | 'EOF';

interface Token {
  type: TokenType;
  value: string;
  line: number;
  column: number;
}
```

**Design decisions:**
- Use a `position` cursor that advances through the source string
- Handle whitespace and comments during scanning (don't produce tokens for them)
- Keywords are initially scanned as identifiers, then checked against a keyword table

### Parser

The parser uses **recursive descent** to convert the token stream into a tree.

**AST node types:**

```typescript
type ASTNode =
  | NumberLiteral
  | StringLiteral
  | BinaryExpr
  | UnaryExpr
  | Identifier
  | LetStatement
  | AssignmentExpr
  | IfStatement
  | FunctionDecl
  | FunctionCall
  | ReturnStatement
  | Block;

interface BinaryExpr {
  type: 'BinaryExpr';
  op: string;
  left: ASTNode;
  right: ASTNode;
}
```

**Operator precedence** (lowest to highest):

| Precedence | Operators | Associativity |
|-----------|-----------|---------------|
| 1 | `=` | Right |
| 2 | `==`, `!=` | Left |
| 3 | `<`, `>`, `<=`, `>=` | Left |
| 4 | `+`, `-` | Left |
| 5 | `*`, `/` | Left |
| 6 | Unary `-`, `!` | Right |

### Evaluator

The evaluator is a **tree-walker** that recursively visits each node and returns a value.

```python
def evaluate(node, env):
    match node:
        case NumberLiteral(value):
            return value
        case BinaryExpr(op, left, right):
            l = evaluate(left, env)
            r = evaluate(right, env)
            return apply_op(op, l, r)
        case LetStatement(name, init):
            val = evaluate(init, env)
            env.define(name, val)
            return val
        case Identifier(name):
            return env.lookup(name)
        case FunctionCall(callee, args):
            fn = evaluate(callee, env)
            evaluated_args = [evaluate(a, env) for a in args]
            return call_function(fn, evaluated_args)
```

### Environment (Scope Management)

The environment forms a **chain of scopes** using a parent pointer:

```typescript
class Environment {
  private values: Map<string, Value>;
  private parent: Environment | null;

  constructor(parent?: Environment) {
    this.values = new Map();
    this.parent = parent ?? null;
  }

  define(name: string, value: Value): void {
    this.values.set(name, value);
  }

  lookup(name: string): Value {
    if (this.values.has(name)) {
      return this.values.get(name)!;
    }
    if (this.parent) {
      return this.parent.lookup(name);
    }
    throw new Error(`Undefined variable: ${name}`);
  }
}
```

**Scope rules:**
- Each function call creates a new child environment
- Block statements (`{ ... }`) create a new scope
- Variable lookup walks up the parent chain
- Closures capture their defining environment

## Error Handling Strategy

Errors should be clear and include source location:

```
Error at line 5, column 12:
  Unexpected token 'RPAREN', expected expression

  5 |   let x = (+ 3);
                 ^
```

**Error categories:**
- **Lexer errors**: Invalid characters, unterminated strings
- **Parser errors**: Unexpected tokens, missing delimiters
- **Runtime errors**: Type mismatches, undefined variables, division by zero

## Testing Strategy

Each component should be tested independently:

1. **Lexer tests**: Input string -> expected token list
2. **Parser tests**: Token list -> expected AST structure
3. **Evaluator tests**: AST -> expected result
4. **Integration tests**: Source string -> expected output

Example test:

```javascript
test('arithmetic expression', () => {
  const result = interpret('2 + 3 * 4');
  expect(result).toBe(14);
});

test('variable binding', () => {
  const result = interpret('let x = 10; x + 5');
  expect(result).toBe(15);
});
```

## Extension Points

Once the basic interpreter works, consider these extensions:

- **String operations**: Concatenation, length, slicing
- **Arrays/Lists**: Literal syntax, indexing, built-in functions
- **Closures**: Functions that capture their environment
- **Error recovery**: Parser continues after errors to report multiple issues
- **REPL**: Interactive read-eval-print loop

---

*This architecture document is a reference guide. The milestones below break the implementation into incremental steps.*

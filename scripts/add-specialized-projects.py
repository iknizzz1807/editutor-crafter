#!/usr/bin/env python3
"""
Add new specialized domain projects to the YAML file.
"""

import yaml
import os

class MyDumper(yaml.SafeDumper):
    pass

def str_representer(dumper, data):
    if '\n' in data or len(data) > 80:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    if any(c in data for c in [':', '#', '{', '}', '[', ']', ',', '&', '*', '?', '|', '-', '<', '>', '=', '!', '%', '@', '`']):
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='"')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

MyDumper.add_representer(str, str_representer)

# New projects to add
new_projects = {
    "chip8-emulator": {
        "id": "chip8-emulator",
        "name": "CHIP-8 Emulator",
        "description": "Build a CHIP-8 virtual machine emulator. Learn CPU emulation, instruction decoding, and graphics rendering through this classic 1970s VM architecture.",
        "difficulty": "intermediate",
        "estimated_hours": "20-30",
        "prerequisites": ["Binary/hex manipulation", "Basic graphics", "State machines"],
        "languages": {
            "recommended": ["C", "Rust", "Python"],
            "also_possible": ["JavaScript", "Go"]
        },
        "resources": [
            {"name": "Tobias V. Langhoff's CHIP-8 Guide", "url": "https://tobiasvl.github.io/blog/write-a-chip-8-emulator/", "type": "tutorial"},
            {"name": "CHIP-8 Technical Reference", "url": "http://devernay.free.fr/hacks/chip8/C8TECH10.HTM", "type": "reference"},
            {"name": "freeCodeCamp CHIP-8 Tutorial", "url": "https://www.freecodecamp.org/news/creating-your-very-own-chip-8-emulator/", "type": "tutorial"}
        ],
        "milestones": [
            {
                "id": 1,
                "name": "Memory and Registers",
                "description": "Initialize CHIP-8 memory (4KB), registers (V0-VF), index register, program counter, and stack.",
                "acceptance_criteria": [
                    "4096 bytes of RAM initialized",
                    "16 general-purpose 8-bit registers (V0-VF)",
                    "16-bit index register (I)",
                    "16-bit program counter starting at 0x200",
                    "Stack with 16 levels for subroutine calls",
                    "Font sprites loaded at 0x000-0x1FF"
                ],
                "hints": {
                    "level1": "CHIP-8 programs start at address 0x200. The first 512 bytes are reserved for the interpreter and font data.",
                    "level2": "VF register is special - it's used as a flag for carry, borrow, and collision detection. Don't use it for general storage.",
                    "level3": """struct Chip8 {
    uint8_t memory[4096];      // 4KB RAM
    uint8_t V[16];             // V0-VF registers
    uint16_t I;                // Index register
    uint16_t pc;               // Program counter
    uint16_t stack[16];        // Stack
    uint8_t sp;                // Stack pointer
    uint8_t delay_timer;       // Delay timer (60Hz)
    uint8_t sound_timer;       // Sound timer (60Hz)
    uint8_t display[64 * 32];  // 64x32 monochrome display
    uint8_t keypad[16];        // 16-key hexadecimal keypad
};

void init_chip8(Chip8* chip) {
    memset(chip, 0, sizeof(Chip8));
    chip->pc = 0x200;  // Programs start here

    // Load font sprites (0-F) at 0x000
    uint8_t fontset[80] = {
        0xF0, 0x90, 0x90, 0x90, 0xF0, // 0
        0x20, 0x60, 0x20, 0x20, 0x70, // 1
        // ... etc for 0-F
    };
    memcpy(chip->memory, fontset, 80);
}"""
                },
                "pitfalls": [
                    "Forgetting that programs start at 0x200, not 0x000",
                    "Not initializing font sprites in reserved memory",
                    "Using VF as a general register (it's used for flags)"
                ],
                "concepts": ["Memory-mapped I/O", "Register architecture", "Stack operations"],
                "estimated_hours": "3-4"
            },
            {
                "id": 2,
                "name": "Fetch-Decode-Execute Cycle",
                "description": "Implement the main emulation loop and instruction fetching. CHIP-8 instructions are 2 bytes (big-endian).",
                "acceptance_criteria": [
                    "Fetch 2-byte opcode from memory[PC]",
                    "Increment PC by 2 after fetch",
                    "Extract opcode components (nibbles)",
                    "Main loop runs at ~500-700 instructions/second"
                ],
                "hints": {
                    "level1": "CHIP-8 uses big-endian byte order. Combine two bytes: (memory[pc] << 8) | memory[pc+1]",
                    "level2": "Extract nibbles: first = (opcode >> 12), X = (opcode >> 8) & 0xF, Y = (opcode >> 4) & 0xF, N = opcode & 0xF",
                    "level3": """void cycle(Chip8* chip) {
    // Fetch
    uint16_t opcode = (chip->memory[chip->pc] << 8) | chip->memory[chip->pc + 1];
    chip->pc += 2;

    // Decode - extract components
    uint8_t first_nibble = (opcode >> 12) & 0xF;
    uint8_t X = (opcode >> 8) & 0xF;   // Second nibble (register)
    uint8_t Y = (opcode >> 4) & 0xF;   // Third nibble (register)
    uint8_t N = opcode & 0xF;          // Fourth nibble
    uint8_t NN = opcode & 0xFF;        // Last byte
    uint16_t NNN = opcode & 0xFFF;     // Last 12 bits (address)

    // Execute (switch on first_nibble)
    switch (first_nibble) {
        case 0x0:
            if (opcode == 0x00E0) clear_display(chip);
            else if (opcode == 0x00EE) return_from_subroutine(chip);
            break;
        case 0x1: chip->pc = NNN; break;  // Jump
        case 0x2: call_subroutine(chip, NNN); break;
        // ... more cases
    }
}"""
                },
                "pitfalls": [
                    "Wrong byte order when fetching opcode",
                    "Not incrementing PC after fetch (or incrementing wrong amount)",
                    "Running too fast or too slow (should be ~500-700 Hz)"
                ],
                "concepts": ["CPU fetch-decode-execute cycle", "Big-endian byte order", "Instruction decoding"],
                "estimated_hours": "4-5"
            },
            {
                "id": 3,
                "name": "Core Instructions",
                "description": "Implement the 35 CHIP-8 instructions including arithmetic, logic, jumps, and subroutines.",
                "acceptance_criteria": [
                    "All 35 standard opcodes implemented",
                    "Arithmetic: ADD, SUB, SHR, SHL with proper flag handling",
                    "Logic: AND, OR, XOR",
                    "Flow control: JP, CALL, RET, SE, SNE",
                    "Register operations: LD, RND"
                ],
                "hints": {
                    "level1": "Group instructions by first nibble. Most 8xxx instructions are arithmetic/logic between VX and VY.",
                    "level2": "For subtraction, VF=1 means NO borrow. For addition, VF=1 means carry. This is counterintuitive!",
                    "level3": """// Key instruction implementations
case 0x8:  // 8XYN - Arithmetic/Logic
    switch (N) {
        case 0x0: chip->V[X] = chip->V[Y]; break;           // LD
        case 0x1: chip->V[X] |= chip->V[Y]; break;          // OR
        case 0x2: chip->V[X] &= chip->V[Y]; break;          // AND
        case 0x3: chip->V[X] ^= chip->V[Y]; break;          // XOR
        case 0x4: {  // ADD with carry
            uint16_t sum = chip->V[X] + chip->V[Y];
            chip->V[0xF] = (sum > 255) ? 1 : 0;
            chip->V[X] = sum & 0xFF;
            break;
        }
        case 0x5: {  // SUB (VF=1 if NO borrow)
            chip->V[0xF] = (chip->V[X] >= chip->V[Y]) ? 1 : 0;
            chip->V[X] -= chip->V[Y];
            break;
        }
        case 0x6: {  // SHR (shift right, VF = bit shifted out)
            chip->V[0xF] = chip->V[X] & 0x1;
            chip->V[X] >>= 1;
            break;
        }
        case 0xE: {  // SHL (shift left)
            chip->V[0xF] = (chip->V[X] >> 7) & 0x1;
            chip->V[X] <<= 1;
            break;
        }
    }
    break;

case 0xC:  // CXNN - Random
    chip->V[X] = (rand() % 256) & NN;
    break;"""
                },
                "pitfalls": [
                    "Getting VF flag logic backwards for SUB/SUBN",
                    "Forgetting that VF is modified AFTER the operation (store temp first)",
                    "BNNN jump should use V0, not VX (ambiguous in old docs)"
                ],
                "concepts": ["ALU operations", "Flag registers", "Conditional branching"],
                "estimated_hours": "5-6"
            },
            {
                "id": 4,
                "name": "Graphics and Display",
                "description": "Implement the 64x32 monochrome display and DXYN draw instruction with XOR sprite drawing.",
                "acceptance_criteria": [
                    "64x32 pixel display buffer",
                    "00E0 clears the screen",
                    "DXYN draws N-byte sprite at (VX, VY)",
                    "XOR drawing with collision detection (VF=1 if pixel turned off)",
                    "Sprites wrap around screen edges"
                ],
                "hints": {
                    "level1": "Each sprite row is 8 pixels wide (1 byte). DXYN draws N rows starting from memory[I].",
                    "level2": "XOR means: if pixel is on and sprite bit is 1, pixel turns off (collision). If off and bit is 1, turns on.",
                    "level3": """void draw_sprite(Chip8* chip, uint8_t X, uint8_t Y, uint8_t N) {
    uint8_t x_pos = chip->V[X] % 64;  // Wrap around
    uint8_t y_pos = chip->V[Y] % 32;
    chip->V[0xF] = 0;  // Reset collision flag

    for (int row = 0; row < N; row++) {
        uint8_t sprite_byte = chip->memory[chip->I + row];

        for (int col = 0; col < 8; col++) {
            uint8_t sprite_pixel = (sprite_byte >> (7 - col)) & 1;

            int screen_x = (x_pos + col) % 64;
            int screen_y = (y_pos + row) % 32;
            int index = screen_y * 64 + screen_x;

            if (sprite_pixel) {
                if (chip->display[index]) {
                    chip->V[0xF] = 1;  // Collision!
                }
                chip->display[index] ^= 1;  // XOR
            }
        }
    }
}"""
                },
                "pitfalls": [
                    "Not wrapping coordinates around screen edges",
                    "Setting VF before checking all pixels (should be set if ANY collision)",
                    "Drawing bit order wrong (MSB is leftmost pixel)"
                ],
                "concepts": ["Framebuffer", "XOR graphics", "Sprite rendering", "Collision detection"],
                "estimated_hours": "4-5"
            },
            {
                "id": 5,
                "name": "Input and Timers",
                "description": "Implement 16-key hexadecimal keypad input and 60Hz delay/sound timers.",
                "acceptance_criteria": [
                    "16 keys (0-9, A-F) mapped to keyboard",
                    "EX9E/EXA1 skip if key pressed/not pressed",
                    "FX0A waits for key press (blocking)",
                    "Delay timer decrements at 60Hz",
                    "Sound timer plays tone when > 0"
                ],
                "hints": {
                    "level1": "Map keyboard keys to CHIP-8's 4x4 keypad. Common mapping: 1234/QWER/ASDF/ZXCV → 123C/456D/789E/A0BF",
                    "level2": "FX0A is blocking - the emulator should pause until a key is pressed, then store key in VX.",
                    "level3": """// Timer update (call at 60Hz, separate from instruction cycle)
void update_timers(Chip8* chip) {
    if (chip->delay_timer > 0) chip->delay_timer--;
    if (chip->sound_timer > 0) {
        if (chip->sound_timer == 1) play_beep();
        chip->sound_timer--;
    }
}

// Key instructions
case 0xE:
    if (NN == 0x9E) {  // Skip if key VX pressed
        if (chip->keypad[chip->V[X]]) chip->pc += 2;
    } else if (NN == 0xA1) {  // Skip if key VX not pressed
        if (!chip->keypad[chip->V[X]]) chip->pc += 2;
    }
    break;

case 0xF:
    switch (NN) {
        case 0x07: chip->V[X] = chip->delay_timer; break;
        case 0x15: chip->delay_timer = chip->V[X]; break;
        case 0x18: chip->sound_timer = chip->V[X]; break;
        case 0x0A: {  // Wait for key press
            bool pressed = false;
            for (int i = 0; i < 16; i++) {
                if (chip->keypad[i]) {
                    chip->V[X] = i;
                    pressed = true;
                    break;
                }
            }
            if (!pressed) chip->pc -= 2;  // Repeat this instruction
            break;
        }
    }
    break;"""
                },
                "pitfalls": [
                    "Running timers at instruction speed instead of 60Hz",
                    "Not handling FX0A blocking correctly",
                    "Key mapping confusion (CHIP-8 keys != keyboard layout)"
                ],
                "concepts": ["Hardware timers", "Input polling", "Blocking I/O"],
                "estimated_hours": "4-5"
            }
        ]
    },

    "config-parser": {
        "id": "config-parser",
        "name": "Config File Parser",
        "description": "Build a multi-format configuration file parser supporting INI, TOML, and YAML-subset. Learn recursive descent parsing and data structure mapping.",
        "difficulty": "intermediate",
        "estimated_hours": "15-25",
        "prerequisites": ["String manipulation", "Data structures", "Recursive thinking"],
        "languages": {
            "recommended": ["Python", "Go", "Rust"],
            "also_possible": ["JavaScript", "C"]
        },
        "resources": [
            {"name": "TOML Specification", "url": "https://toml.io/en/v1.0.0", "type": "specification"},
            {"name": "INI File Format", "url": "https://en.wikipedia.org/wiki/INI_file", "type": "reference"},
            {"name": "Writing a Parser in Python", "url": "https://www.freecodecamp.org/news/how-to-write-a-parser-in-python/", "type": "tutorial"}
        ],
        "milestones": [
            {
                "id": 1,
                "name": "INI Parser",
                "description": "Parse INI files with sections, key-value pairs, and comments.",
                "acceptance_criteria": [
                    "Parse [section] headers",
                    "Parse key=value and key: value pairs",
                    "Handle ; and # comments",
                    "Support quoted strings with escapes",
                    "Return nested dictionary structure"
                ],
                "hints": {
                    "level1": "INI is line-based. Process line by line: check if it's a section, comment, or key-value.",
                    "level2": "Use regex or simple string operations. Section: ^\\[(.+)\\]$, Comment: ^[;#], Key-value: split on first = or :",
                    "level3": """import re

def parse_ini(text: str) -> dict:
    result = {}
    current_section = None

    for line_num, line in enumerate(text.splitlines(), 1):
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith(';') or line.startswith('#'):
            continue

        # Section header
        if line.startswith('[') and line.endswith(']'):
            current_section = line[1:-1].strip()
            if current_section not in result:
                result[current_section] = {}
            continue

        # Key-value pair
        match = re.match(r'^([^=:]+)[=:](.*)$', line)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()

            # Remove quotes
            if (value.startswith('"') and value.endswith('"')) or \\
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]

            # Type coercion
            if value.lower() in ('true', 'yes', 'on'):
                value = True
            elif value.lower() in ('false', 'no', 'off'):
                value = False
            elif value.isdigit():
                value = int(value)

            target = result[current_section] if current_section else result
            target[key] = value
        else:
            raise ValueError(f"Invalid line {line_num}: {line}")

    return result"""
                },
                "pitfalls": [
                    "Not handling keys outside of sections (global keys)",
                    "Forgetting to handle inline comments after values",
                    "Breaking on = inside quoted strings"
                ],
                "concepts": ["Line-based parsing", "Regular expressions", "Type coercion"],
                "estimated_hours": "3-4"
            },
            {
                "id": 2,
                "name": "TOML Tokenizer",
                "description": "Build a lexer/tokenizer for TOML format.",
                "acceptance_criteria": [
                    "Tokenize: brackets, dots, equals, strings, numbers",
                    "Handle basic strings, literal strings, multiline strings",
                    "Recognize integers, floats, booleans, dates",
                    "Track line/column for error messages"
                ],
                "hints": {
                    "level1": "TOML has more complex string types than INI. Basic strings use \", literal strings use '.",
                    "level2": "Create Token class with type, value, line, column. Use a Lexer class that tracks position.",
                    "level3": """from enum import Enum
from dataclasses import dataclass

class TokenType(Enum):
    LBRACKET = '['
    RBRACKET = ']'
    LBRACE = '{'
    RBRACE = '}'
    DOT = '.'
    EQUALS = '='
    COMMA = ','
    STRING = 'STRING'
    INTEGER = 'INTEGER'
    FLOAT = 'FLOAT'
    BOOLEAN = 'BOOLEAN'
    DATETIME = 'DATETIME'
    BARE_KEY = 'BARE_KEY'
    NEWLINE = 'NEWLINE'
    EOF = 'EOF'

@dataclass
class Token:
    type: TokenType
    value: any
    line: int
    column: int

class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1

    def peek(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ''

    def advance(self) -> str:
        ch = self.peek()
        self.pos += 1
        if ch == '\\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return ch

    def read_string(self, quote: str) -> str:
        # Handle basic string (") with escapes
        # Handle literal string (') without escapes
        # Handle multiline (''' or \"\"\")
        pass

    def tokenize(self) -> list[Token]:
        tokens = []
        while self.pos < len(self.text):
            ch = self.peek()
            if ch in ' \\t':
                self.advance()
            elif ch == '\\n':
                tokens.append(Token(TokenType.NEWLINE, None, self.line, self.column))
                self.advance()
            elif ch == '#':
                while self.peek() and self.peek() != '\\n':
                    self.advance()
            # ... handle other token types
        return tokens"""
                },
                "pitfalls": [
                    "TOML multiline strings have complex rules for leading newlines",
                    "Literal strings don't process escapes (backslash is literal)",
                    "Integer underscores are allowed: 1_000_000"
                ],
                "concepts": ["Lexical analysis", "State machines", "Unicode handling"],
                "estimated_hours": "4-5"
            },
            {
                "id": 3,
                "name": "TOML Parser",
                "description": "Build recursive descent parser for TOML tables and arrays.",
                "acceptance_criteria": [
                    "Parse [table] and [table.subtable] headers",
                    "Parse [[array.of.tables]]",
                    "Handle inline tables { key = value }",
                    "Handle inline arrays [ 1, 2, 3 ]",
                    "Dotted keys: physical.color = 'orange'"
                ],
                "hints": {
                    "level1": "TOML keys can define nested structure: a.b.c = 1 creates {a: {b: {c: 1}}}",
                    "level2": "[[array]] appends to an array of tables. Each [[array]] block is a new element.",
                    "level3": """class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0
        self.result = {}
        self.current_table = self.result

    def parse_key(self) -> list[str]:
        '''Parse dotted key like a.b.c into ["a", "b", "c"]'''
        keys = []
        while True:
            tok = self.expect(TokenType.BARE_KEY, TokenType.STRING)
            keys.append(tok.value)
            if not self.match(TokenType.DOT):
                break
        return keys

    def set_nested(self, keys: list[str], value: any):
        '''Set value at nested key path'''
        target = self.current_table
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value

    def parse_table_header(self):
        '''Parse [table] or [[array.of.tables]]'''
        is_array = self.match(TokenType.LBRACKET)  # Second [
        keys = self.parse_key()
        self.expect(TokenType.RBRACKET)
        if is_array:
            self.expect(TokenType.RBRACKET)

        # Navigate to (or create) the table
        target = self.result
        for i, key in enumerate(keys):
            if key not in target:
                target[key] = [] if (is_array and i == len(keys)-1) else {}
            target = target[key]
            if isinstance(target, list):
                if i == len(keys) - 1:
                    target.append({})
                target = target[-1]

        self.current_table = target"""
                },
                "pitfalls": [
                    "Can't redefine a key that already exists",
                    "Tables can be defined implicitly by dotted keys",
                    "Array of tables vs array value have different syntax"
                ],
                "concepts": ["Recursive descent parsing", "Symbol tables", "Nested data structures"],
                "estimated_hours": "5-6"
            },
            {
                "id": 4,
                "name": "YAML Subset Parser",
                "description": "Parse a subset of YAML: indentation-based nesting, mappings, sequences.",
                "acceptance_criteria": [
                    "Indentation-based block structure",
                    "Key: value mappings",
                    "Sequences with - prefix",
                    "Quoted and unquoted strings",
                    "Flow style: [list] and {map}"
                ],
                "hints": {
                    "level1": "Track indentation level. Deeper indent = nested structure. Same indent = sibling.",
                    "level2": "Build a stack of (indent_level, container). When indent decreases, pop back to matching level.",
                    "level3": """def parse_yaml(text: str) -> dict:
    lines = text.splitlines()
    result = {}
    stack = [(0, result)]  # (indent_level, container)

    for line in lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith('#'):
            continue

        indent = len(line) - len(stripped)

        # Pop stack until we find appropriate parent
        while stack and stack[-1][0] >= indent:
            stack.pop()

        parent = stack[-1][1] if stack else result

        if stripped.startswith('- '):
            # Sequence item
            value = stripped[2:].strip()
            if isinstance(parent, dict):
                # First item, convert to list (need parent key)
                pass
            parent.append(parse_value(value) if value else {})
            if not value:
                stack.append((indent + 2, parent[-1]))
        elif ':' in stripped:
            # Mapping
            key, _, value = stripped.partition(':')
            key = key.strip()
            value = value.strip()

            if value:
                parent[key] = parse_value(value)
            else:
                parent[key] = {}
                stack.append((indent + 2, parent[key]))

    return result

def parse_value(s: str):
    if s.startswith('['):
        return parse_flow_sequence(s)
    if s.startswith('{'):
        return parse_flow_mapping(s)
    if s.lower() in ('true', 'yes'):
        return True
    if s.lower() in ('false', 'no'):
        return False
    if s.lower() == 'null':
        return None
    try:
        return int(s)
    except:
        try:
            return float(s)
        except:
            return s.strip('\"\\''')"""
                },
                "pitfalls": [
                    "Tabs vs spaces (YAML forbids tabs for indentation)",
                    "Implicit type detection can be surprising (yes = true, 1.0 = float)",
                    "Multiline strings have multiple syntaxes (|, >, etc.)"
                ],
                "concepts": ["Indentation-sensitive parsing", "Implicit typing", "Stack-based parsing"],
                "estimated_hours": "5-6"
            }
        ]
    },

    "diff-tool": {
        "id": "diff-tool",
        "name": "Diff Tool",
        "description": "Build a text diff tool using the Longest Common Subsequence algorithm and Myers' diff algorithm. Learn dynamic programming and edit distance concepts.",
        "difficulty": "beginner",
        "estimated_hours": "12-18",
        "prerequisites": ["Dynamic programming basics", "File I/O", "String manipulation"],
        "languages": {
            "recommended": ["Python", "JavaScript", "Go"],
            "also_possible": ["C", "Rust", "Java"]
        },
        "resources": [
            {"name": "Myers' Diff Algorithm Tutorial", "url": "http://simplygenius.net/Article/DiffTutorial1", "type": "tutorial"},
            {"name": "The Myers Difference Algorithm", "url": "https://nathaniel.ai/myers-diff/", "type": "article"},
            {"name": "Wikipedia - LCS", "url": "https://en.wikipedia.org/wiki/Longest_common_subsequence", "type": "reference"}
        ],
        "milestones": [
            {
                "id": 1,
                "name": "Line Tokenization",
                "description": "Read two files and split into line arrays for comparison.",
                "acceptance_criteria": [
                    "Read files handling different encodings",
                    "Split by newlines preserving empty lines",
                    "Handle different line endings (\\n, \\r\\n, \\r)",
                    "Report line counts for each file"
                ],
                "hints": {
                    "level1": "Use splitlines() in Python or split by regex /\\r?\\n/ in JavaScript.",
                    "level2": "Consider normalizing line endings first, or preserve them for accurate diff.",
                    "level3": """def read_lines(filepath: str) -> list[str]:
    '''Read file and return list of lines'''
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Normalize line endings
    content = content.replace('\\r\\n', '\\n').replace('\\r', '\\n')

    # Split into lines (keeping empty lines)
    lines = content.split('\\n')

    # Remove trailing empty line if file ends with newline
    if lines and lines[-1] == '':
        lines = lines[:-1]

    return lines

def main():
    file1_lines = read_lines(sys.argv[1])
    file2_lines = read_lines(sys.argv[2])

    print(f"File 1: {len(file1_lines)} lines")
    print(f"File 2: {len(file2_lines)} lines")"""
                },
                "pitfalls": [
                    "Binary files will cause encoding errors",
                    "Large files can exhaust memory",
                    "Trailing newline handling varies between tools"
                ],
                "concepts": ["File encoding", "Line endings", "Text normalization"],
                "estimated_hours": "2-3"
            },
            {
                "id": 2,
                "name": "LCS Algorithm",
                "description": "Implement Longest Common Subsequence using dynamic programming.",
                "acceptance_criteria": [
                    "Build LCS length matrix",
                    "Backtrack to find actual LCS",
                    "Handle sequences of different lengths",
                    "O(mn) time and space complexity"
                ],
                "hints": {
                    "level1": "LCS matrix: if items match, dp[i][j] = dp[i-1][j-1] + 1. Otherwise, max of left or up.",
                    "level2": "Backtrack from dp[m][n]: if match, include item and go diagonal. Otherwise, go to larger neighbor.",
                    "level3": """def lcs_matrix(seq1: list, seq2: list) -> list[list[int]]:
    '''Build LCS length matrix'''
    m, n = len(seq1), len(seq2)

    # dp[i][j] = length of LCS of seq1[:i] and seq2[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp

def backtrack_lcs(dp: list[list[int]], seq1: list, seq2: list) -> list:
    '''Backtrack to find LCS'''
    lcs = []
    i, j = len(seq1), len(seq2)

    while i > 0 and j > 0:
        if seq1[i-1] == seq2[j-1]:
            lcs.append((i-1, j-1, seq1[i-1]))  # (idx1, idx2, item)
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1

    lcs.reverse()
    return lcs"""
                },
                "pitfalls": [
                    "Off-by-one errors in matrix indexing",
                    "Not handling empty sequences",
                    "Memory explosion for large files (optimize with Hirschberg)"
                ],
                "concepts": ["Dynamic programming", "2D matrices", "Backtracking"],
                "estimated_hours": "4-5"
            },
            {
                "id": 3,
                "name": "Diff Generation",
                "description": "Convert LCS result into diff hunks with context.",
                "acceptance_criteria": [
                    "Mark lines as unchanged, added, or deleted",
                    "Generate unified diff format (-/+ prefixes)",
                    "Group changes into hunks with context lines",
                    "Include @@ line range markers"
                ],
                "hints": {
                    "level1": "Walk both sequences. If in LCS, it's unchanged. Otherwise, if only in seq1, deleted. Only in seq2, added.",
                    "level2": "Unified format: context lines have ' ' prefix, deletions '-', additions '+'.",
                    "level3": """def generate_diff(lines1: list[str], lines2: list[str], context: int = 3):
    '''Generate unified diff output'''
    lcs_indices = set()
    for idx1, idx2, _ in backtrack_lcs(lcs_matrix(lines1, lines2), lines1, lines2):
        lcs_indices.add(('1', idx1))
        lcs_indices.add(('2', idx2))

    # Build diff operations
    ops = []
    i, j = 0, 0

    while i < len(lines1) or j < len(lines2):
        if ('1', i) in lcs_indices and ('2', j) in lcs_indices:
            ops.append((' ', lines1[i]))
            i += 1
            j += 1
        elif ('1', i) not in lcs_indices and i < len(lines1):
            ops.append(('-', lines1[i]))
            i += 1
        elif ('2', j) not in lcs_indices and j < len(lines2):
            ops.append(('+', lines2[j]))
            j += 1

    # Group into hunks with context
    hunks = []
    current_hunk = []
    # ... group changes separated by more than context*2 unchanged lines

    # Output
    for hunk in hunks:
        print(f"@@ -{hunk.start1},{hunk.len1} +{hunk.start2},{hunk.len2} @@")
        for op, line in hunk.lines:
            print(f"{op}{line}")"""
                },
                "pitfalls": [
                    "Off-by-one in line numbers (diff format is 1-indexed)",
                    "Forgetting to handle files with no common lines",
                    "Context overlapping between hunks"
                ],
                "concepts": ["Diff algorithms", "Unified diff format", "Hunk generation"],
                "estimated_hours": "4-5"
            },
            {
                "id": 4,
                "name": "CLI and Color Output",
                "description": "Build command-line interface with colored output and options.",
                "acceptance_criteria": [
                    "Accept two file paths as arguments",
                    "Color output: red for deletions, green for additions",
                    "--no-color flag for plain output",
                    "--context N to set context lines",
                    "Exit code 0 if same, 1 if different"
                ],
                "hints": {
                    "level1": "Use ANSI escape codes: \\033[31m for red, \\033[32m for green, \\033[0m to reset.",
                    "level2": "Check if stdout is a TTY before using colors. Use --no-color to force plain.",
                    "level3": """import argparse
import sys

class Colors:
    RED = '\\033[31m'
    GREEN = '\\033[32m'
    CYAN = '\\033[36m'
    RESET = '\\033[0m'

    @classmethod
    def disable(cls):
        cls.RED = cls.GREEN = cls.CYAN = cls.RESET = ''

def main():
    parser = argparse.ArgumentParser(description='Compare two files')
    parser.add_argument('file1', help='First file')
    parser.add_argument('file2', help='Second file')
    parser.add_argument('-u', '--unified', type=int, default=3,
                        help='Number of context lines (default: 3)')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output')
    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    lines1 = read_lines(args.file1)
    lines2 = read_lines(args.file2)

    if lines1 == lines2:
        sys.exit(0)

    # Print header
    print(f"{Colors.CYAN}--- {args.file1}{Colors.RESET}")
    print(f"{Colors.CYAN}+++ {args.file2}{Colors.RESET}")

    for hunk in generate_hunks(lines1, lines2, args.unified):
        print(f"{Colors.CYAN}@@ ... @@{Colors.RESET}")
        for op, line in hunk:
            if op == '-':
                print(f"{Colors.RED}-{line}{Colors.RESET}")
            elif op == '+':
                print(f"{Colors.GREEN}+{line}{Colors.RESET}")
            else:
                print(f" {line}")

    sys.exit(1)"""
                },
                "pitfalls": [
                    "ANSI codes break when piped to file",
                    "Windows needs special handling for colors",
                    "Exit codes are important for scripting"
                ],
                "concepts": ["CLI argument parsing", "ANSI colors", "Exit codes", "TTY detection"],
                "estimated_hours": "3-4"
            }
        ]
    },

    "disassembler": {
        "id": "disassembler",
        "name": "x86 Disassembler",
        "description": "Build an x86/x64 instruction disassembler. Learn machine code encoding, instruction formats, and binary parsing.",
        "difficulty": "advanced",
        "estimated_hours": "35-50",
        "prerequisites": ["x86 assembly basics", "Binary file handling", "Bitwise operations"],
        "languages": {
            "recommended": ["C", "Rust", "Python"],
            "also_possible": ["Go", "C++"]
        },
        "resources": [
            {"name": "Intel x86 Manual Vol. 2", "url": "https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html", "type": "reference"},
            {"name": "Medium - Building x86-64 Disassembler", "url": "https://medium.com/@Koukyosyumei/learning-x86-64-machine-language-and-assembly-by-implementing-a-disassembler-dccc736ae85f", "type": "tutorial"},
            {"name": "x86 Instruction Encoding", "url": "http://wiki.osdev.org/X86-64_Instruction_Encoding", "type": "reference"}
        ],
        "milestones": [
            {
                "id": 1,
                "name": "Binary File Loading",
                "description": "Load and parse ELF or PE executable headers to find code sections.",
                "acceptance_criteria": [
                    "Parse ELF header (or PE for Windows)",
                    "Locate .text section",
                    "Extract code bytes and base address",
                    "Handle 32-bit and 64-bit binaries"
                ],
                "hints": {
                    "level1": "ELF magic: 0x7F 'E' 'L' 'F'. Check byte 5 for 32/64-bit. Section headers list all sections.",
                    "level2": "e_shoff points to section header table. e_shstrndx is section name string table index.",
                    "level3": """from dataclasses import dataclass
from struct import unpack

@dataclass
class Section:
    name: str
    addr: int
    offset: int
    size: int
    data: bytes

def parse_elf(data: bytes) -> tuple[int, list[Section]]:
    '''Parse ELF and return (bitness, sections)'''
    if data[:4] != b'\\x7fELF':
        raise ValueError("Not an ELF file")

    is_64bit = data[4] == 2
    is_little = data[5] == 1
    endian = '<' if is_little else '>'

    if is_64bit:
        # 64-bit header
        e_shoff = unpack(f'{endian}Q', data[40:48])[0]
        e_shentsize = unpack(f'{endian}H', data[58:60])[0]
        e_shnum = unpack(f'{endian}H', data[60:62])[0]
        e_shstrndx = unpack(f'{endian}H', data[62:64])[0]
    else:
        # 32-bit header
        e_shoff = unpack(f'{endian}I', data[32:36])[0]
        e_shentsize = unpack(f'{endian}H', data[46:48])[0]
        e_shnum = unpack(f'{endian}H', data[48:50])[0]
        e_shstrndx = unpack(f'{endian}H', data[50:52])[0]

    # Parse section headers
    sections = []
    # ... read section names from shstrtab, extract .text

    return (64 if is_64bit else 32, sections)"""
                },
                "pitfalls": [
                    "Endianness varies (most x86 is little-endian)",
                    "Virtual address != file offset",
                    "Some binaries are stripped (no symbol table)"
                ],
                "concepts": ["Executable formats", "Binary parsing", "Memory mapping"],
                "estimated_hours": "6-8"
            },
            {
                "id": 2,
                "name": "Instruction Prefixes",
                "description": "Decode x86 instruction prefixes: legacy, REX, VEX.",
                "acceptance_criteria": [
                    "Detect legacy prefixes (66, 67, F0, F2, F3, segment overrides)",
                    "Decode REX prefix (64-bit mode)",
                    "Extract REX.W, REX.R, REX.X, REX.B bits",
                    "Handle prefix ordering"
                ],
                "hints": {
                    "level1": "REX prefix: 0x40-0x4F in 64-bit mode. Format: 0100WRXB.",
                    "level2": "66h = operand size override, 67h = address size override. They change default sizes.",
                    "level3": """@dataclass
class Prefixes:
    rex: int = 0        # Full REX byte (0 if none)
    rex_w: bool = False # 64-bit operand
    rex_r: bool = False # Extends ModRM.reg
    rex_x: bool = False # Extends SIB.index
    rex_b: bool = False # Extends ModRM.rm or SIB.base
    operand_size: bool = False  # 66h prefix
    address_size: bool = False  # 67h prefix
    rep: int = 0        # F2/F3 prefix
    lock: bool = False  # F0 prefix
    segment: int = 0    # Segment override

def decode_prefixes(code: bytes, offset: int, is_64bit: bool) -> tuple[Prefixes, int]:
    '''Decode prefixes, return (Prefixes, bytes_consumed)'''
    prefixes = Prefixes()
    pos = offset

    while pos < len(code):
        byte = code[pos]

        if byte == 0x66:
            prefixes.operand_size = True
        elif byte == 0x67:
            prefixes.address_size = True
        elif byte == 0xF0:
            prefixes.lock = True
        elif byte in (0xF2, 0xF3):
            prefixes.rep = byte
        elif byte in (0x2E, 0x3E, 0x26, 0x64, 0x65, 0x36):
            prefixes.segment = byte
        elif is_64bit and 0x40 <= byte <= 0x4F:
            prefixes.rex = byte
            prefixes.rex_w = bool(byte & 0x08)
            prefixes.rex_r = bool(byte & 0x04)
            prefixes.rex_x = bool(byte & 0x02)
            prefixes.rex_b = bool(byte & 0x01)
        else:
            break  # Not a prefix

        pos += 1

    return prefixes, pos - offset"""
                },
                "pitfalls": [
                    "REX must be immediately before opcode",
                    "Some prefix combinations are invalid or have special meaning",
                    "VEX/EVEX prefixes in modern code need separate handling"
                ],
                "concepts": ["Prefix encoding", "Mode-dependent behavior", "Bit field extraction"],
                "estimated_hours": "6-8"
            },
            {
                "id": 3,
                "name": "Opcode Tables",
                "description": "Build opcode lookup tables for instruction identification.",
                "acceptance_criteria": [
                    "One-byte opcode table",
                    "Two-byte opcode table (0F xx)",
                    "Handle opcode extensions via ModRM.reg",
                    "Map opcodes to mnemonics"
                ],
                "hints": {
                    "level1": "Many opcodes share byte but differ by ModRM.reg field. E.g., 0x80 is ADD/OR/ADC/SBB/AND/SUB/XOR/CMP.",
                    "level2": "Create separate tables: one_byte_opcodes, two_byte_opcodes, extension_groups.",
                    "level3": """# Opcode table entry
@dataclass
class OpcodeEntry:
    mnemonic: str
    operands: str  # e.g., "Ev,Gv" or "rAX,Iz"
    flags: int = 0
    extension_group: int = 0  # 0 = no extension, 1-17 = group number

# One-byte opcode table (partial)
ONE_BYTE = {
    0x00: OpcodeEntry("ADD", "Eb,Gb"),
    0x01: OpcodeEntry("ADD", "Ev,Gv"),
    0x02: OpcodeEntry("ADD", "Gb,Eb"),
    0x03: OpcodeEntry("ADD", "Gv,Ev"),
    0x04: OpcodeEntry("ADD", "AL,Ib"),
    0x05: OpcodeEntry("ADD", "rAX,Iz"),
    # ...
    0x50: OpcodeEntry("PUSH", "rAX"),  # 50-57 are PUSH r16/r64
    # ...
    0x80: OpcodeEntry("", "Eb,Ib", extension_group=1),  # Group 1
    0x81: OpcodeEntry("", "Ev,Iz", extension_group=1),
    # ...
    0x0F: OpcodeEntry("", "", flags=TWO_BYTE),  # Escape to two-byte table
}

# Extension group 1: ADD/OR/ADC/SBB/AND/SUB/XOR/CMP
GROUP_1 = ["ADD", "OR", "ADC", "SBB", "AND", "SUB", "XOR", "CMP"]

def lookup_opcode(byte: int, modrm_reg: int = 0) -> OpcodeEntry:
    entry = ONE_BYTE.get(byte)
    if entry and entry.extension_group:
        mnemonic = GROUPS[entry.extension_group][modrm_reg]
        return OpcodeEntry(mnemonic, entry.operands)
    return entry"""
                },
                "pitfalls": [
                    "Opcode tables are large and error-prone to type manually",
                    "Some opcodes are invalid in certain modes",
                    "Three-byte opcodes exist (0F 38 xx, 0F 3A xx)"
                ],
                "concepts": ["Lookup tables", "Instruction encoding", "Opcode extensions"],
                "estimated_hours": "8-10"
            },
            {
                "id": 4,
                "name": "ModRM and SIB Decoding",
                "description": "Decode ModRM and SIB bytes for operand addressing.",
                "acceptance_criteria": [
                    "Parse ModRM: mod, reg, rm fields",
                    "Handle SIB when rm=100b",
                    "Decode all 16 addressing modes",
                    "Handle RIP-relative addressing (64-bit)"
                ],
                "hints": {
                    "level1": "ModRM = mod(2) | reg(3) | rm(3). mod=11 means register, otherwise memory.",
                    "level2": "SIB = scale(2) | index(3) | base(3). Address = base + index * scale + disp.",
                    "level3": """def decode_modrm(code: bytes, offset: int, prefixes: Prefixes, is_64bit: bool) -> tuple[str, int]:
    '''Decode ModRM, return (operand_string, bytes_consumed)'''
    modrm = code[offset]
    mod = (modrm >> 6) & 0x3
    reg = (modrm >> 3) & 0x7
    rm = modrm & 0x7

    # Apply REX extensions
    if prefixes.rex_r:
        reg |= 0x8
    if prefixes.rex_b:
        rm |= 0x8

    consumed = 1

    if mod == 0b11:
        # Register direct
        return get_register(rm, prefixes), consumed

    # Memory addressing
    if rm == 0b100 and not is_64bit:
        # SIB follows (32-bit) or rm=100 with REX.B (64-bit)
        sib = code[offset + 1]
        consumed += 1
        scale = 1 << ((sib >> 6) & 0x3)
        index = (sib >> 3) & 0x7
        base = sib & 0x7

        if prefixes.rex_x:
            index |= 0x8
        if prefixes.rex_b:
            base |= 0x8

        # Build address string
        # ...

    if mod == 0b00 and rm == 0b101:
        if is_64bit:
            # RIP-relative
            disp = unpack('<i', code[offset+consumed:offset+consumed+4])[0]
            consumed += 4
            return f"[rip + {disp:#x}]", consumed
        else:
            # 32-bit displacement only
            disp = unpack('<i', code[offset+consumed:offset+consumed+4])[0]
            consumed += 4
            return f"[{disp:#x}]", consumed

    # ... handle other mod values"""
                },
                "pitfalls": [
                    "RIP-relative is 64-bit only, in 32-bit mod=00 rm=101 is disp32",
                    "SIB with index=100 means no index (unless REX.X set)",
                    "REX extends registers to r8-r15"
                ],
                "concepts": ["Addressing modes", "Register encoding", "Memory operands"],
                "estimated_hours": "8-10"
            },
            {
                "id": 5,
                "name": "Output Formatting",
                "description": "Format disassembly output with addresses, bytes, and mnemonics.",
                "acceptance_criteria": [
                    "Show address, hex bytes, mnemonic, operands",
                    "Intel and AT&T syntax options",
                    "Label jumps to known addresses",
                    "Handle undefined/invalid opcodes gracefully"
                ],
                "hints": {
                    "level1": "Intel: op dst, src. AT&T: op src, dst (with size suffixes and % for registers).",
                    "level2": "Track branch targets to insert labels. Relative jumps need base address calculation.",
                    "level3": """class Disassembler:
    def __init__(self, code: bytes, base_addr: int, is_64bit: bool):
        self.code = code
        self.base = base_addr
        self.is_64bit = is_64bit
        self.labels = {}  # addr -> label name

    def disassemble(self) -> list[str]:
        # First pass: find branch targets
        offset = 0
        while offset < len(self.code):
            instr = self.decode_instruction(offset)
            if instr.is_branch:
                target = instr.branch_target
                if target not in self.labels:
                    self.labels[target] = f"loc_{target:x}"
            offset += instr.length

        # Second pass: generate output
        output = []
        offset = 0
        while offset < len(self.code):
            addr = self.base + offset

            # Insert label if this is a branch target
            if addr in self.labels:
                output.append(f"\\n{self.labels[addr]}:")

            instr = self.decode_instruction(offset)
            hex_bytes = self.code[offset:offset+instr.length].hex()

            output.append(f"{addr:08x}:  {hex_bytes:<20}  {instr}")
            offset += instr.length

        return output

    def format_intel(self, mnemonic: str, operands: list[str]) -> str:
        return f"{mnemonic} {', '.join(operands)}"

    def format_att(self, mnemonic: str, operands: list[str]) -> str:
        # Reverse operands, add suffixes
        return f"{mnemonic} {', '.join(reversed(operands))}"

# Output example:
# 00401000:  55                    push rbp
# 00401001:  48 89 e5              mov rbp, rsp
# 00401004:  e8 17 00 00 00        call loc_401020"""
                },
                "pitfalls": [
                    "Hex bytes should be padded/aligned for readability",
                    "Relative addresses need base address to calculate target",
                    "Invalid opcodes should print as .byte or db"
                ],
                "concepts": ["Assembly syntax", "Address calculation", "Output formatting"],
                "estimated_hours": "6-8"
            }
        ]
    },

    "hexdump": {
        "id": "hexdump",
        "name": "Hexdump Utility",
        "description": "Build a hexdump utility to display binary file contents in hexadecimal and ASCII. Learn binary file handling and formatted output.",
        "difficulty": "beginner",
        "estimated_hours": "8-12",
        "prerequisites": ["File I/O", "Binary vs text", "Formatted output"],
        "languages": {
            "recommended": ["C", "Python", "Rust"],
            "also_possible": ["Go", "JavaScript"]
        },
        "resources": [
            {"name": "Let's Build a Hexdump Utility in C", "url": "http://www.dmulholl.com/blog/lets-build-a-hexdump-utility.html", "type": "tutorial"},
            {"name": "Hexdump man page", "url": "https://man7.org/linux/man-pages/man1/hexdump.1.html", "type": "reference"}
        ],
        "milestones": [
            {
                "id": 1,
                "name": "Basic Hex Output",
                "description": "Read binary file and output bytes in hexadecimal format.",
                "acceptance_criteria": [
                    "Read file in binary mode",
                    "Output 16 bytes per line",
                    "Show offset at start of each line",
                    "Handle files of any size"
                ],
                "hints": {
                    "level1": "Read file in chunks of 16 bytes. Use format specifier %02x for hex output.",
                    "level2": "Track offset, incrementing by 16 each line. Handle partial last line.",
                    "level3": """def hexdump_basic(filepath: str):
    offset = 0
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(16)
            if not chunk:
                break

            # Format: offset: hex bytes
            hex_str = ' '.join(f'{b:02x}' for b in chunk)
            print(f'{offset:08x}: {hex_str}')

            offset += len(chunk)

# Output:
# 00000000: 7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00
# 00000010: 03 00 3e 00 01 00 00 00 40 10 40 00 00 00 00 00"""
                },
                "pitfalls": [
                    "Opening file in text mode mangles binary data",
                    "Last chunk may have fewer than 16 bytes",
                    "Large files should be streamed, not loaded entirely"
                ],
                "concepts": ["Binary file I/O", "Hexadecimal formatting", "Chunked reading"],
                "estimated_hours": "2-3"
            },
            {
                "id": 2,
                "name": "ASCII Column",
                "description": "Add ASCII representation alongside hex output.",
                "acceptance_criteria": [
                    "Show printable ASCII characters",
                    "Replace non-printable with '.'",
                    "Align ASCII column properly",
                    "Handle partial lines"
                ],
                "hints": {
                    "level1": "Printable ASCII: 0x20-0x7E. Use chr() to convert, '.' for others.",
                    "level2": "Pad hex section with spaces for short lines to keep ASCII aligned.",
                    "level3": """def hexdump_with_ascii(filepath: str):
    offset = 0
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(16)
            if not chunk:
                break

            # Hex part
            hex_parts = [f'{b:02x}' for b in chunk]
            hex_str = ' '.join(hex_parts)

            # Pad for alignment (each byte = 3 chars: "xx ")
            padding = '   ' * (16 - len(chunk))

            # ASCII part
            ascii_str = ''.join(
                chr(b) if 0x20 <= b <= 0x7e else '.'
                for b in chunk
            )

            print(f'{offset:08x}: {hex_str}{padding}  |{ascii_str}|')
            offset += len(chunk)

# Output:
# 00000000: 7f 45 4c 46 02 01 01 00 00 00 00 00 00 00 00 00  |.ELF............|
# 00000010: 03 00 3e 00 01 00 00 00 40 10 40 00 00 00 00 00  |..>.....@.@.....|"""
                },
                "pitfalls": [
                    "Don't print control characters (newlines, tabs corrupt output)",
                    "Alignment breaks on partial lines without padding",
                    "Unicode handling varies by terminal"
                ],
                "concepts": ["ASCII encoding", "String formatting", "Column alignment"],
                "estimated_hours": "2-3"
            },
            {
                "id": 3,
                "name": "Grouped Output",
                "description": "Group hex bytes (2, 4, or 8 byte groupings) for better readability.",
                "acceptance_criteria": [
                    "Support -g option for group size (1, 2, 4, 8)",
                    "Add extra space between groups",
                    "Default to 2-byte groups (like xxd)",
                    "Handle endianness display option"
                ],
                "hints": {
                    "level1": "Split 16-byte line into groups. For 2-byte groups: 8 groups per line.",
                    "level2": "Join bytes within group without space, separate groups with space.",
                    "level3": """def hexdump_grouped(filepath: str, group_size: int = 2):
    offset = 0
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(16)
            if not chunk:
                break

            # Group bytes
            groups = []
            for i in range(0, len(chunk), group_size):
                group = chunk[i:i+group_size]
                groups.append(''.join(f'{b:02x}' for b in group))

            hex_str = ' '.join(groups)

            # Calculate padding (each group = group_size*2 + 1 for space)
            full_groups = 16 // group_size
            actual_groups = (len(chunk) + group_size - 1) // group_size
            padding = ' ' * ((full_groups - actual_groups) * (group_size * 2 + 1))

            ascii_str = ''.join(
                chr(b) if 0x20 <= b <= 0x7e else '.'
                for b in chunk
            )

            print(f'{offset:08x}: {hex_str}{padding}  {ascii_str}')
            offset += len(chunk)

# With group_size=2:
# 00000000: 7f45 4c46 0201 0100 0000 0000 0000 0000  .ELF............
# 00000010: 0300 3e00 0100 0000 4010 4000 0000 0000  ..>.....@.@....."""
                },
                "pitfalls": [
                    "Group boundaries at end of file need special handling",
                    "Endianness affects display for multi-byte groups",
                    "Different tools use different default groupings"
                ],
                "concepts": ["Data grouping", "Byte order", "Flexible formatting"],
                "estimated_hours": "2-3"
            },
            {
                "id": 4,
                "name": "CLI Options",
                "description": "Add command-line options: offset, length, output format.",
                "acceptance_criteria": [
                    "-s/--skip: start offset",
                    "-n/--length: number of bytes",
                    "-C: canonical format (like hexdump -C)",
                    "Read from stdin if no file specified"
                ],
                "hints": {
                    "level1": "Use argparse (Python) or getopt (C). Seek to offset before reading.",
                    "level2": "Handle - as stdin. Count bytes for length limit.",
                    "level3": """import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Hexadecimal file dump')
    parser.add_argument('file', nargs='?', default='-',
                        help='File to dump (default: stdin)')
    parser.add_argument('-C', '--canonical', action='store_true',
                        help='Canonical hex+ASCII display')
    parser.add_argument('-g', '--group', type=int, default=2,
                        help='Bytes per group (default: 2)')
    parser.add_argument('-s', '--skip', type=int, default=0,
                        help='Skip bytes at start')
    parser.add_argument('-n', '--length', type=int, default=None,
                        help='Dump only N bytes')
    args = parser.parse_args()

    # Open file or stdin
    if args.file == '-':
        f = sys.stdin.buffer
    else:
        f = open(args.file, 'rb')

    try:
        # Skip to offset
        if args.skip and args.file != '-':
            f.seek(args.skip)
        elif args.skip:
            f.read(args.skip)  # Can't seek stdin

        bytes_read = 0
        while True:
            remaining = None
            if args.length:
                remaining = args.length - bytes_read
                if remaining <= 0:
                    break

            chunk_size = min(16, remaining) if remaining else 16
            chunk = f.read(chunk_size)
            if not chunk:
                break

            # Output...
            bytes_read += len(chunk)
    finally:
        if f != sys.stdin.buffer:
            f.close()"""
                },
                "pitfalls": [
                    "Can't seek on stdin (must read and discard)",
                    "Length limit interacts with offset",
                    "Binary mode required for stdin"
                ],
                "concepts": ["CLI design", "Standard input", "Argument parsing"],
                "estimated_hours": "2-3"
            }
        ]
    },

    "markdown-renderer": {
        "id": "markdown-renderer",
        "name": "Markdown Renderer",
        "description": "Build a Markdown to HTML converter. Learn text parsing, regular expressions, and document transformation.",
        "difficulty": "beginner",
        "estimated_hours": "12-18",
        "prerequisites": ["Regular expressions", "String manipulation", "HTML basics"],
        "languages": {
            "recommended": ["Python", "JavaScript", "Go"],
            "also_possible": ["Rust", "Ruby"]
        },
        "resources": [
            {"name": "CommonMark Spec", "url": "https://spec.commonmark.org/", "type": "specification"},
            {"name": "Building a Markdown Parser", "url": "https://dev.to/kawaljain/building-my-own-markdown-parser-a-developers-journey-3b26", "type": "tutorial"},
            {"name": "Sarvasv's MD Parser Notes", "url": "https://sarvasvkulpati.com/mdparser", "type": "tutorial"}
        ],
        "milestones": [
            {
                "id": 1,
                "name": "Block Elements",
                "description": "Parse block-level elements: headings, paragraphs, code blocks, horizontal rules.",
                "acceptance_criteria": [
                    "# to ###### headings → h1-h6",
                    "Blank-line separated paragraphs → p",
                    "``` or indented code → pre>code",
                    "--- or *** → hr"
                ],
                "hints": {
                    "level1": "Process line by line. Blank lines separate blocks. Accumulate paragraph lines.",
                    "level2": "Headings: /^(#{1,6})\\s+(.+)$/. Code blocks: track if inside ``` fence.",
                    "level3": """def parse_blocks(text: str) -> list[dict]:
    '''Parse text into block elements'''
    lines = text.split('\\n')
    blocks = []
    current_para = []
    in_code = False
    code_lines = []

    def flush_para():
        if current_para:
            blocks.append({'type': 'paragraph', 'content': ' '.join(current_para)})
            current_para.clear()

    for line in lines:
        # Code fence
        if line.startswith('```'):
            if in_code:
                blocks.append({'type': 'code', 'content': '\\n'.join(code_lines)})
                code_lines.clear()
                in_code = False
            else:
                flush_para()
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        # Heading
        match = re.match(r'^(#{1,6})\\s+(.+)$', line)
        if match:
            flush_para()
            level = len(match.group(1))
            blocks.append({'type': f'h{level}', 'content': match.group(2)})
            continue

        # Horizontal rule
        if re.match(r'^([-*_])\\1{2,}\\s*$', line):
            flush_para()
            blocks.append({'type': 'hr'})
            continue

        # Blank line
        if not line.strip():
            flush_para()
            continue

        # Paragraph continuation
        current_para.append(line)

    flush_para()
    return blocks"""
                },
                "pitfalls": [
                    "Setext headings (underline style) need lookahead",
                    "Indented code vs list continuation is complex",
                    "Fenced code can have language identifier"
                ],
                "concepts": ["Block parsing", "State machines", "Line-by-line processing"],
                "estimated_hours": "4-5"
            },
            {
                "id": 2,
                "name": "Inline Elements",
                "description": "Parse inline formatting: bold, italic, code, links, images.",
                "acceptance_criteria": [
                    "**bold** and __bold__ → strong",
                    "*italic* and _italic_ → em",
                    "`code` → code",
                    "[text](url) → a href",
                    "![alt](url) → img"
                ],
                "hints": {
                    "level1": "Process inline after block parsing. Use regex or character-by-character scan.",
                    "level2": "Handle nested: **bold _and italic_**. Process outer markers first.",
                    "level3": """def parse_inline(text: str) -> str:
    '''Convert inline markdown to HTML'''
    # Process in order: code (literal), then links, then emphasis

    # Inline code (must be first - content is literal)
    text = re.sub(r'`([^`]+)`', r'<code>\\1</code>', text)

    # Images (before links - similar syntax)
    text = re.sub(r'!\\[([^\\]]*)\]\\(([^)]+)\\)', r'<img src="\\2" alt="\\1">', text)

    # Links
    text = re.sub(r'\\[([^\\]]+)\\]\\(([^)]+)\\)', r'<a href="\\2">\\1</a>', text)

    # Bold (** or __)
    text = re.sub(r'\\*\\*([^*]+)\\*\\*', r'<strong>\\1</strong>', text)
    text = re.sub(r'__([^_]+)__', r'<strong>\\1</strong>', text)

    # Italic (* or _)
    text = re.sub(r'\\*([^*]+)\\*', r'<em>\\1</em>', text)
    text = re.sub(r'_([^_]+)_', r'<em>\\1</em>', text)

    return text

# Better approach: tokenize then render
def tokenize_inline(text: str) -> list:
    tokens = []
    i = 0
    while i < len(text):
        if text[i:i+2] == '**':
            # Find closing **
            end = text.find('**', i+2)
            if end != -1:
                tokens.append(('strong', text[i+2:end]))
                i = end + 2
                continue
        # ... handle other cases
        tokens.append(('text', text[i]))
        i += 1
    return tokens"""
                },
                "pitfalls": [
                    "Underscore in middle_of_words shouldn't trigger emphasis",
                    "Mismatched delimiters: **bold* is invalid",
                    "Escaping: \\* should be literal asterisk"
                ],
                "concepts": ["Inline parsing", "Regular expressions", "Nested formatting"],
                "estimated_hours": "4-5"
            },
            {
                "id": 3,
                "name": "Lists",
                "description": "Parse ordered and unordered lists with nesting.",
                "acceptance_criteria": [
                    "- or * for unordered lists → ul/li",
                    "1. 2. 3. for ordered lists → ol/li",
                    "Indented items create nested lists",
                    "Tight vs loose lists (with blank lines)"
                ],
                "hints": {
                    "level1": "Track indentation level. Deeper indent = nested list. Same indent = sibling.",
                    "level2": "Use a stack to track nested lists. Push on deeper indent, pop on shallower.",
                    "level3": """def parse_list(lines: list[str], start_idx: int) -> tuple[dict, int]:
    '''Parse list starting at start_idx, return (list_node, next_idx)'''
    first_line = lines[start_idx]

    # Determine list type
    if re.match(r'^\\d+\\.\\s', first_line.lstrip()):
        list_type = 'ol'
        pattern = r'^(\\s*)(\\d+)\\.\\s+(.*)$'
    else:
        list_type = 'ul'
        pattern = r'^(\\s*)([-*+])\\s+(.*)$'

    base_indent = len(first_line) - len(first_line.lstrip())
    items = []
    idx = start_idx

    while idx < len(lines):
        line = lines[idx]
        if not line.strip():
            idx += 1
            continue

        match = re.match(pattern, line)
        if not match:
            break

        indent = len(match.group(1))
        content = match.group(3)

        if indent < base_indent:
            break
        elif indent > base_indent:
            # Nested list - recursively parse
            nested, idx = parse_list(lines, idx)
            items[-1]['children'].append(nested)
        else:
            items.append({'content': content, 'children': []})
            idx += 1

    return {'type': list_type, 'items': items}, idx"""
                },
                "pitfalls": [
                    "Mixed list types in same level is invalid",
                    "Continuation lines must be indented",
                    "Loose lists have paragraphs in items"
                ],
                "concepts": ["Recursive parsing", "Indentation tracking", "Tree structures"],
                "estimated_hours": "4-5"
            },
            {
                "id": 4,
                "name": "HTML Generation",
                "description": "Convert parsed structure to valid HTML output.",
                "acceptance_criteria": [
                    "Generate valid HTML5",
                    "Escape special characters (&, <, >)",
                    "Proper nesting and indentation",
                    "Optional: add wrapper template"
                ],
                "hints": {
                    "level1": "Walk the parsed tree, emit HTML tags. Escape content but not generated tags.",
                    "level2": "Use html.escape() for content. Track indent level for pretty printing.",
                    "level3": """import html

class HtmlRenderer:
    def __init__(self, pretty: bool = True):
        self.pretty = pretty
        self.indent = 0

    def render(self, blocks: list[dict]) -> str:
        output = []
        for block in blocks:
            output.append(self.render_block(block))
        return '\\n'.join(output)

    def render_block(self, block: dict) -> str:
        t = block['type']

        if t == 'hr':
            return self._line('<hr>')

        if t.startswith('h'):
            content = self.render_inline(block['content'])
            return self._line(f'<{t}>{content}</{t}>')

        if t == 'paragraph':
            content = self.render_inline(block['content'])
            return self._line(f'<p>{content}</p>')

        if t == 'code':
            escaped = html.escape(block['content'])
            return self._line(f'<pre><code>{escaped}</code></pre>')

        if t in ('ul', 'ol'):
            return self.render_list(block)

        return ''

    def render_inline(self, text: str) -> str:
        # First escape HTML entities in the text
        text = html.escape(text)
        # Then apply inline markdown (careful not to escape our generated tags)
        # ... apply inline patterns
        return text

    def _line(self, content: str) -> str:
        indent = '  ' * self.indent if self.pretty else ''
        return f'{indent}{content}'"""
                },
                "pitfalls": [
                    "Double-escaping: escape content before inline parsing",
                    "Self-closing tags: <hr> not <hr></hr>",
                    "Inline HTML in markdown should pass through"
                ],
                "concepts": ["HTML generation", "Character escaping", "Tree traversal"],
                "estimated_hours": "3-4"
            }
        ]
    },

    "packet-sniffer": {
        "id": "packet-sniffer",
        "name": "Packet Sniffer",
        "description": "Build a network packet capture and analysis tool using raw sockets or libpcap. Learn network protocols and packet parsing.",
        "difficulty": "intermediate",
        "estimated_hours": "20-30",
        "prerequisites": ["Networking basics", "TCP/IP model", "Binary parsing"],
        "languages": {
            "recommended": ["C", "Python", "Go"],
            "also_possible": ["Rust"]
        },
        "resources": [
            {"name": "Libpcap Programming Tutorial", "url": "https://www.tcpdump.org/pcap.html", "type": "tutorial"},
            {"name": "Building a Packet Sniffer from Scratch", "url": "https://aidanvidal.github.io/posts/Packet_Sniffer.html", "type": "tutorial"},
            {"name": "Scapy Documentation", "url": "https://scapy.readthedocs.io/", "type": "reference"}
        ],
        "milestones": [
            {
                "id": 1,
                "name": "Packet Capture Setup",
                "description": "Set up packet capture using raw sockets or libpcap.",
                "acceptance_criteria": [
                    "List available network interfaces",
                    "Open interface for capture",
                    "Capture packets in promiscuous mode",
                    "Handle root/admin permissions"
                ],
                "hints": {
                    "level1": "libpcap: pcap_findalldevs() lists interfaces, pcap_open_live() opens for capture.",
                    "level2": "Promiscuous mode captures all traffic, not just for your host. Requires root.",
                    "level3": """# Python with scapy (simpler) or raw sockets
from scapy.all import sniff, get_if_list

def list_interfaces():
    return get_if_list()

def capture_packets(interface: str, count: int = 10):
    '''Capture packets and return them'''
    packets = sniff(iface=interface, count=count, promisc=True)
    return packets

# Or with raw sockets (Linux)
import socket

def raw_capture(interface: str):
    # Create raw socket
    sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0003))
    sock.bind((interface, 0))
    sock.setblocking(False)

    while True:
        try:
            packet, addr = sock.recvfrom(65535)
            yield packet
        except BlockingIOError:
            continue

# With libpcap (C)
'''
pcap_t *handle;
char errbuf[PCAP_ERRBUF_SIZE];
handle = pcap_open_live("eth0", BUFSIZ, 1, 1000, errbuf);
if (handle == NULL) {
    fprintf(stderr, "Couldn't open device: %s\\n", errbuf);
    return 1;
}
'''"""
                },
                "pitfalls": [
                    "Requires root/admin privileges",
                    "Interface names vary by OS (eth0, en0, etc.)",
                    "Virtual interfaces may not support promiscuous mode"
                ],
                "concepts": ["Raw sockets", "Network interfaces", "Privilege requirements"],
                "estimated_hours": "4-5"
            },
            {
                "id": 2,
                "name": "Ethernet Frame Parsing",
                "description": "Parse Ethernet frames to extract MAC addresses and protocol type.",
                "acceptance_criteria": [
                    "Extract destination MAC (6 bytes)",
                    "Extract source MAC (6 bytes)",
                    "Extract EtherType (2 bytes)",
                    "Format MACs as aa:bb:cc:dd:ee:ff"
                ],
                "hints": {
                    "level1": "Ethernet header is 14 bytes: dst(6) + src(6) + type(2). Type 0x0800 = IPv4.",
                    "level2": "Use struct.unpack() for parsing. MAC is 6 bytes, format each as hex.",
                    "level3": """from struct import unpack

class EthernetFrame:
    def __init__(self, data: bytes):
        # Unpack ethernet header (14 bytes)
        # ! = network byte order (big endian)
        self.dst_mac = data[0:6]
        self.src_mac = data[6:12]
        self.ethertype = unpack('!H', data[12:14])[0]
        self.payload = data[14:]

    @staticmethod
    def format_mac(mac: bytes) -> str:
        return ':'.join(f'{b:02x}' for b in mac)

    def __str__(self):
        proto = {0x0800: 'IPv4', 0x0806: 'ARP', 0x86dd: 'IPv6'}.get(
            self.ethertype, f'0x{self.ethertype:04x}'
        )
        return (f"Ethernet: {self.format_mac(self.src_mac)} -> "
                f"{self.format_mac(self.dst_mac)} ({proto})")

# Parse captured packet
frame = EthernetFrame(raw_packet)
print(frame)
# Ethernet: aa:bb:cc:dd:ee:ff -> 11:22:33:44:55:66 (IPv4)"""
                },
                "pitfalls": [
                    "802.1Q VLAN tags add 4 extra bytes",
                    "Jumbo frames can exceed 1500 bytes payload",
                    "Byte order is big-endian (network order)"
                ],
                "concepts": ["Ethernet framing", "MAC addresses", "EtherType"],
                "estimated_hours": "3-4"
            },
            {
                "id": 3,
                "name": "IP Header Parsing",
                "description": "Parse IPv4 headers to extract addresses and protocol.",
                "acceptance_criteria": [
                    "Extract version and header length",
                    "Extract total length, TTL, protocol",
                    "Extract source and destination IP",
                    "Handle IP options (variable header length)"
                ],
                "hints": {
                    "level1": "IPv4 header: first nibble = version (4), second nibble = IHL (header length in 32-bit words).",
                    "level2": "Protocol field: 1=ICMP, 6=TCP, 17=UDP. IPs are 4 bytes each at offset 12 and 16.",
                    "level3": """class IPv4Packet:
    def __init__(self, data: bytes):
        # First byte: version (4 bits) + IHL (4 bits)
        version_ihl = data[0]
        self.version = version_ihl >> 4
        self.ihl = version_ihl & 0x0F  # Header length in 32-bit words
        self.header_length = self.ihl * 4

        self.dscp_ecn = data[1]
        self.total_length = unpack('!H', data[2:4])[0]
        self.identification = unpack('!H', data[4:6])[0]

        flags_frag = unpack('!H', data[6:8])[0]
        self.flags = flags_frag >> 13
        self.fragment_offset = flags_frag & 0x1FFF

        self.ttl = data[8]
        self.protocol = data[9]
        self.checksum = unpack('!H', data[10:12])[0]

        self.src_ip = data[12:16]
        self.dst_ip = data[16:20]

        # Options (if any)
        self.options = data[20:self.header_length] if self.ihl > 5 else b''

        self.payload = data[self.header_length:]

    @staticmethod
    def format_ip(ip: bytes) -> str:
        return '.'.join(str(b) for b in ip)

    def __str__(self):
        proto = {1: 'ICMP', 6: 'TCP', 17: 'UDP'}.get(self.protocol, str(self.protocol))
        return (f"IPv4: {self.format_ip(self.src_ip)} -> "
                f"{self.format_ip(self.dst_ip)} ({proto}, TTL={self.ttl})")"""
                },
                "pitfalls": [
                    "Header length is in 32-bit words, not bytes",
                    "Fragmented packets need reassembly",
                    "Options make header length variable"
                ],
                "concepts": ["IP addressing", "Protocol numbers", "Header fields"],
                "estimated_hours": "4-5"
            },
            {
                "id": 4,
                "name": "TCP/UDP Parsing",
                "description": "Parse TCP and UDP headers for port information and flags.",
                "acceptance_criteria": [
                    "Extract source and destination ports",
                    "For TCP: extract flags (SYN, ACK, FIN, etc.)",
                    "For TCP: sequence and acknowledgment numbers",
                    "Identify common services by port"
                ],
                "hints": {
                    "level1": "TCP header: ports(4) + seq(4) + ack(4) + flags(2) + window(2) + checksum(2) + urgent(2) = 20+ bytes",
                    "level2": "UDP is simpler: ports(4) + length(2) + checksum(2) = 8 bytes total.",
                    "level3": """class TCPSegment:
    def __init__(self, data: bytes):
        self.src_port = unpack('!H', data[0:2])[0]
        self.dst_port = unpack('!H', data[2:4])[0]
        self.seq_num = unpack('!I', data[4:8])[0]
        self.ack_num = unpack('!I', data[8:12])[0]

        data_offset_flags = unpack('!H', data[12:14])[0]
        self.data_offset = (data_offset_flags >> 12) * 4  # In bytes
        self.flags = data_offset_flags & 0x1FF

        self.window = unpack('!H', data[14:16])[0]
        self.checksum = unpack('!H', data[16:18])[0]
        self.urgent = unpack('!H', data[18:20])[0]

        self.payload = data[self.data_offset:]

    @property
    def flag_str(self) -> str:
        flags = []
        if self.flags & 0x001: flags.append('FIN')
        if self.flags & 0x002: flags.append('SYN')
        if self.flags & 0x004: flags.append('RST')
        if self.flags & 0x008: flags.append('PSH')
        if self.flags & 0x010: flags.append('ACK')
        if self.flags & 0x020: flags.append('URG')
        return ','.join(flags) or 'none'

class UDPDatagram:
    def __init__(self, data: bytes):
        self.src_port = unpack('!H', data[0:2])[0]
        self.dst_port = unpack('!H', data[2:4])[0]
        self.length = unpack('!H', data[4:6])[0]
        self.checksum = unpack('!H', data[6:8])[0]
        self.payload = data[8:]

# Common ports
SERVICES = {80: 'HTTP', 443: 'HTTPS', 22: 'SSH', 53: 'DNS', 25: 'SMTP'}"""
                },
                "pitfalls": [
                    "TCP data offset is in 32-bit words",
                    "Port numbers are unsigned 16-bit",
                    "Don't confuse TCP and UDP despite similar port fields"
                ],
                "concepts": ["TCP flags", "Port numbers", "Transport layer"],
                "estimated_hours": "4-5"
            },
            {
                "id": 5,
                "name": "Filtering and Output",
                "description": "Add BPF filters and formatted output display.",
                "acceptance_criteria": [
                    "Support BPF filter expressions",
                    "Filter by protocol (tcp, udp, icmp)",
                    "Filter by port or IP",
                    "Formatted packet summary output"
                ],
                "hints": {
                    "level1": "BPF syntax: 'tcp port 80', 'host 192.168.1.1', 'udp and port 53'",
                    "level2": "pcap_compile() and pcap_setfilter() apply BPF filters before capture.",
                    "level3": """# Using scapy's BPF filter
from scapy.all import sniff

def capture_with_filter(interface: str, bpf_filter: str, callback):
    '''Capture packets matching BPF filter'''
    sniff(iface=interface, filter=bpf_filter, prn=callback)

def packet_callback(packet):
    '''Process and display packet'''
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]

    # Build summary
    summary = f"{timestamp} "

    if packet.haslayer('Ether'):
        eth = packet['Ether']

    if packet.haslayer('IP'):
        ip = packet['IP']
        summary += f"{ip.src} -> {ip.dst} "

        if packet.haslayer('TCP'):
            tcp = packet['TCP']
            flags = tcp.sprintf('%TCP.flags%')
            summary += f"TCP {tcp.sport} -> {tcp.dport} [{flags}]"
        elif packet.haslayer('UDP'):
            udp = packet['UDP']
            summary += f"UDP {udp.sport} -> {udp.dport}"
        elif packet.haslayer('ICMP'):
            icmp = packet['ICMP']
            summary += f"ICMP type={icmp.type}"

    print(summary)

# Example filters:
# "tcp port 80"       - HTTP traffic
# "udp port 53"       - DNS queries
# "host 192.168.1.1"  - All traffic to/from IP
# "tcp[tcpflags] & tcp-syn != 0"  - TCP SYN packets"""
                },
                "pitfalls": [
                    "BPF syntax errors are cryptic",
                    "High traffic can overwhelm display",
                    "Timestamps should be high-resolution"
                ],
                "concepts": ["Berkeley Packet Filter", "Traffic filtering", "Packet analysis"],
                "estimated_hours": "5-6"
            }
        ]
    },

    "protocol-buffer": {
        "id": "protocol-buffer",
        "name": "Protocol Buffer",
        "description": "Implement a binary serialization format similar to Protocol Buffers. Learn varints, wire types, and schema-driven encoding.",
        "difficulty": "advanced",
        "estimated_hours": "25-35",
        "prerequisites": ["Binary encoding", "Schema concepts", "Data structures"],
        "languages": {
            "recommended": ["Python", "Go", "Rust"],
            "also_possible": ["C", "Java"]
        },
        "resources": [
            {"name": "Protocol Buffers Encoding", "url": "https://protobuf.dev/programming-guides/encoding/", "type": "reference"},
            {"name": "Varint Encoding", "url": "https://developers.google.com/protocol-buffers/docs/encoding#varints", "type": "tutorial"}
        ],
        "milestones": [
            {
                "id": 1,
                "name": "Varint Encoding",
                "description": "Implement variable-length integer encoding used in Protocol Buffers.",
                "acceptance_criteria": [
                    "Encode unsigned integers as varints",
                    "Encode signed integers with ZigZag encoding",
                    "Decode varints from byte stream",
                    "Handle all 64-bit integer values"
                ],
                "hints": {
                    "level1": "Varint uses 7 bits per byte for data, MSB=1 means more bytes follow.",
                    "level2": "ZigZag maps signed to unsigned: 0->0, -1->1, 1->2, -2->3, etc. Formula: (n << 1) ^ (n >> 63)",
                    "level3": """def encode_varint(value: int) -> bytes:
    '''Encode unsigned integer as varint'''
    result = []
    while value > 127:
        result.append((value & 0x7F) | 0x80)  # 7 bits + continuation
        value >>= 7
    result.append(value)
    return bytes(result)

def decode_varint(data: bytes, offset: int = 0) -> tuple[int, int]:
    '''Decode varint, return (value, bytes_consumed)'''
    result = 0
    shift = 0
    pos = offset

    while True:
        byte = data[pos]
        result |= (byte & 0x7F) << shift
        pos += 1

        if not (byte & 0x80):  # No continuation bit
            break
        shift += 7

    return result, pos - offset

def zigzag_encode(value: int) -> int:
    '''Encode signed int using ZigZag'''
    return (value << 1) ^ (value >> 63)

def zigzag_decode(value: int) -> int:
    '''Decode ZigZag to signed int'''
    return (value >> 1) ^ -(value & 1)

# Test
assert encode_varint(1) == b'\\x01'
assert encode_varint(300) == b'\\xac\\x02'  # 300 = 0b100101100
assert zigzag_encode(-1) == 1
assert zigzag_encode(1) == 2"""
                },
                "pitfalls": [
                    "Signed integers need ZigZag, not two's complement",
                    "Varint can be up to 10 bytes for 64-bit values",
                    "Overflow possible if not handling properly"
                ],
                "concepts": ["Variable-length encoding", "Integer representation", "ZigZag encoding"],
                "estimated_hours": "4-5"
            },
            {
                "id": 2,
                "name": "Wire Types",
                "description": "Implement the wire type system for field encoding.",
                "acceptance_criteria": [
                    "Type 0: Varint (int32, int64, uint, bool, enum)",
                    "Type 1: 64-bit (fixed64, sfixed64, double)",
                    "Type 2: Length-delimited (string, bytes, embedded messages)",
                    "Type 5: 32-bit (fixed32, sfixed32, float)"
                ],
                "hints": {
                    "level1": "Field key = (field_number << 3) | wire_type. Wire type is in lower 3 bits.",
                    "level2": "Length-delimited: encode length as varint, then raw bytes.",
                    "level3": """from enum import IntEnum
from struct import pack, unpack

class WireType(IntEnum):
    VARINT = 0
    FIXED64 = 1
    LENGTH_DELIMITED = 2
    # 3, 4 deprecated (groups)
    FIXED32 = 5

def encode_field_key(field_number: int, wire_type: WireType) -> bytes:
    '''Encode field key'''
    key = (field_number << 3) | wire_type
    return encode_varint(key)

def decode_field_key(data: bytes, offset: int) -> tuple[int, WireType, int]:
    '''Decode field key, return (field_number, wire_type, bytes_consumed)'''
    key, consumed = decode_varint(data, offset)
    field_number = key >> 3
    wire_type = WireType(key & 0x07)
    return field_number, wire_type, consumed

def encode_field(field_number: int, value, value_type: str) -> bytes:
    '''Encode a field with type'''
    if value_type in ('int32', 'int64', 'uint32', 'uint64', 'bool'):
        wire_type = WireType.VARINT
        val_bytes = encode_varint(value if not isinstance(value, bool) else int(value))
    elif value_type in ('sint32', 'sint64'):
        wire_type = WireType.VARINT
        val_bytes = encode_varint(zigzag_encode(value))
    elif value_type == 'double':
        wire_type = WireType.FIXED64
        val_bytes = pack('<d', value)
    elif value_type == 'float':
        wire_type = WireType.FIXED32
        val_bytes = pack('<f', value)
    elif value_type in ('string', 'bytes'):
        wire_type = WireType.LENGTH_DELIMITED
        data = value.encode() if isinstance(value, str) else value
        val_bytes = encode_varint(len(data)) + data
    else:
        raise ValueError(f"Unknown type: {value_type}")

    return encode_field_key(field_number, wire_type) + val_bytes"""
                },
                "pitfalls": [
                    "Wire types 3 and 4 are deprecated (start/end group)",
                    "Fixed types are little-endian",
                    "Unknown fields should be preserved, not discarded"
                ],
                "concepts": ["Wire types", "Type-length-value encoding", "Binary protocols"],
                "estimated_hours": "5-6"
            },
            {
                "id": 3,
                "name": "Schema Parser",
                "description": "Parse a simple schema definition to drive encoding/decoding.",
                "acceptance_criteria": [
                    "Parse message definitions",
                    "Parse field declarations with types and numbers",
                    "Support repeated fields",
                    "Support nested messages"
                ],
                "hints": {
                    "level1": "Schema: message Name { type name = number; }. Start simple, then add features.",
                    "level2": "Build AST with Message nodes containing Field nodes.",
                    "level3": """from dataclasses import dataclass
from typing import Optional
import re

@dataclass
class Field:
    name: str
    type: str
    number: int
    repeated: bool = False

@dataclass
class Message:
    name: str
    fields: list[Field]
    nested: list['Message']

def parse_schema(text: str) -> list[Message]:
    '''Parse protobuf-like schema'''
    messages = []

    # Find all message definitions
    message_pattern = r'message\\s+(\\w+)\\s*\\{([^}]+)\\}'

    for match in re.finditer(message_pattern, text, re.DOTALL):
        name = match.group(1)
        body = match.group(2)

        fields = []
        # Field pattern: [repeated] type name = number;
        field_pattern = r'(repeated\\s+)?(\\w+)\\s+(\\w+)\\s*=\\s*(\\d+)\\s*;'

        for field_match in re.finditer(field_pattern, body):
            repeated = bool(field_match.group(1))
            field_type = field_match.group(2)
            field_name = field_match.group(3)
            field_number = int(field_match.group(4))

            fields.append(Field(field_name, field_type, field_number, repeated))

        messages.append(Message(name, fields, []))

    return messages

# Example schema
schema = '''
message Person {
    string name = 1;
    int32 age = 2;
    repeated string emails = 3;
}
'''

messages = parse_schema(schema)
# Message(name='Person', fields=[
#     Field(name='name', type='string', number=1),
#     Field(name='age', type='int32', number=2),
#     Field(name='emails', type='string', number=3, repeated=True)
# ])"""
                },
                "pitfalls": [
                    "Field numbers must be unique within a message",
                    "Field numbers 19000-19999 are reserved",
                    "Nested messages need recursive parsing"
                ],
                "concepts": ["Schema languages", "Code generation", "Parsing"],
                "estimated_hours": "5-6"
            },
            {
                "id": 4,
                "name": "Message Serialization",
                "description": "Serialize and deserialize messages according to schema.",
                "acceptance_criteria": [
                    "Serialize dict to bytes using schema",
                    "Deserialize bytes to dict using schema",
                    "Handle missing optional fields",
                    "Handle repeated fields (arrays)"
                ],
                "hints": {
                    "level1": "Fields can appear in any order. Repeated fields appear multiple times.",
                    "level2": "During decode, collect all values for repeated fields into a list.",
                    "level3": """class Serializer:
    def __init__(self, messages: list[Message]):
        self.messages = {m.name: m for m in messages}

    def serialize(self, message_name: str, data: dict) -> bytes:
        '''Serialize dict to protobuf bytes'''
        message = self.messages[message_name]
        result = b''

        for field in message.fields:
            value = data.get(field.name)
            if value is None:
                continue

            if field.repeated:
                for item in value:
                    result += encode_field(field.number, item, field.type)
            else:
                result += encode_field(field.number, value, field.type)

        return result

    def deserialize(self, message_name: str, data: bytes) -> dict:
        '''Deserialize protobuf bytes to dict'''
        message = self.messages[message_name]
        field_map = {f.number: f for f in message.fields}
        result = {}

        offset = 0
        while offset < len(data):
            field_number, wire_type, consumed = decode_field_key(data, offset)
            offset += consumed

            field = field_map.get(field_number)
            if not field:
                # Skip unknown field
                offset += skip_field(data, offset, wire_type)
                continue

            value, consumed = decode_value(data, offset, wire_type, field.type)
            offset += consumed

            if field.repeated:
                if field.name not in result:
                    result[field.name] = []
                result[field.name].append(value)
            else:
                result[field.name] = value

        return result

# Usage
serializer = Serializer(parse_schema(schema))
data = {'name': 'Alice', 'age': 30, 'emails': ['a@b.com', 'c@d.com']}
encoded = serializer.serialize('Person', data)
decoded = serializer.deserialize('Person', encoded)"""
                },
                "pitfalls": [
                    "Field order in wire format is arbitrary",
                    "Unknown fields should be skipped, not error",
                    "Empty repeated field = no entries (not encoded)"
                ],
                "concepts": ["Serialization", "Schema-driven encoding", "Forward compatibility"],
                "estimated_hours": "6-8"
            }
        ]
    },

    "terminal-multiplexer": {
        "id": "terminal-multiplexer",
        "name": "Terminal Multiplexer",
        "description": "Build a simple terminal multiplexer like tmux/screen. Learn PTY handling, terminal escape sequences, and process management.",
        "difficulty": "advanced",
        "estimated_hours": "30-40",
        "prerequisites": ["Unix processes", "Terminal basics", "File descriptors"],
        "languages": {
            "recommended": ["C", "Rust", "Go"],
            "also_possible": ["Python"]
        },
        "resources": [
            {"name": "PTY Programming", "url": "https://www.man7.org/linux/man-pages/man7/pty.7.html", "type": "reference"},
            {"name": "ANSI Escape Codes", "url": "https://en.wikipedia.org/wiki/ANSI_escape_code", "type": "reference"},
            {"name": "Building a Terminal Emulator", "url": "https://www.uninformativ.de/blog/postings/2018-02-24/0/POSTING-en.html", "type": "tutorial"}
        ],
        "milestones": [
            {
                "id": 1,
                "name": "PTY Creation",
                "description": "Create and manage pseudo-terminal pairs for running shell sessions.",
                "acceptance_criteria": [
                    "Open PTY master/slave pair",
                    "Fork child process with PTY as controlling terminal",
                    "Start shell in child process",
                    "Handle terminal size (TIOCSWINSZ)"
                ],
                "hints": {
                    "level1": "posix_openpt() or openpty() create PTY pairs. Child uses slave, parent uses master.",
                    "level2": "Child must setsid(), then open slave to make it controlling terminal.",
                    "level3": """import os
import pty
import fcntl
import termios
import struct

def create_pty_shell() -> tuple[int, int]:
    '''Create PTY and spawn shell, return (master_fd, child_pid)'''
    master_fd, slave_fd = pty.openpty()

    pid = os.fork()

    if pid == 0:
        # Child process
        os.close(master_fd)

        # Create new session
        os.setsid()

        # Slave becomes controlling terminal
        os.dup2(slave_fd, 0)  # stdin
        os.dup2(slave_fd, 1)  # stdout
        os.dup2(slave_fd, 2)  # stderr

        if slave_fd > 2:
            os.close(slave_fd)

        # Exec shell
        shell = os.environ.get('SHELL', '/bin/sh')
        os.execvp(shell, [shell])

    # Parent
    os.close(slave_fd)
    return master_fd, pid

def set_terminal_size(fd: int, rows: int, cols: int):
    '''Set terminal size via ioctl'''
    winsize = struct.pack('HHHH', rows, cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)

# C version:
'''
int master_fd = posix_openpt(O_RDWR | O_NOCTTY);
grantpt(master_fd);
unlockpt(master_fd);
char *slave_name = ptsname(master_fd);
'''"""
                },
                "pitfalls": [
                    "Must call setsid() in child before opening slave",
                    "File descriptors need careful management across fork",
                    "SIGCHLD handling for child termination"
                ],
                "concepts": ["Pseudo-terminals", "Process groups", "Controlling terminals"],
                "estimated_hours": "6-8"
            },
            {
                "id": 2,
                "name": "Terminal Emulation",
                "description": "Parse and render terminal escape sequences for display.",
                "acceptance_criteria": [
                    "Handle cursor movement (up, down, left, right)",
                    "Handle clear screen and clear line",
                    "Handle text attributes (bold, colors)",
                    "Maintain virtual screen buffer"
                ],
                "hints": {
                    "level1": "ANSI escapes start with ESC[ (0x1B 0x5B). Parse parameters between ESC[ and command letter.",
                    "level2": "Maintain grid of cells with character and attributes. Render to real terminal.",
                    "level3": """from dataclasses import dataclass
from enum import IntFlag

class Attr(IntFlag):
    NORMAL = 0
    BOLD = 1
    DIM = 2
    UNDERLINE = 4
    REVERSE = 8

@dataclass
class Cell:
    char: str = ' '
    fg: int = 7  # White
    bg: int = 0  # Black
    attr: Attr = Attr.NORMAL

class Screen:
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.cursor_row = 0
        self.cursor_col = 0
        self.cells = [[Cell() for _ in range(cols)] for _ in range(rows)]
        self.current_attr = Attr.NORMAL
        self.current_fg = 7
        self.current_bg = 0

    def write(self, data: bytes):
        '''Process output from PTY'''
        i = 0
        while i < len(data):
            if data[i] == 0x1B and i+1 < len(data) and data[i+1] == 0x5B:
                # ESC[ sequence
                end, params, cmd = self.parse_csi(data, i+2)
                self.handle_csi(params, cmd)
                i = end
            elif data[i] == ord('\\n'):
                self.cursor_row = min(self.cursor_row + 1, self.rows - 1)
                i += 1
            elif data[i] == ord('\\r'):
                self.cursor_col = 0
                i += 1
            else:
                self.put_char(chr(data[i]))
                i += 1

    def handle_csi(self, params: list[int], cmd: str):
        '''Handle CSI escape sequence'''
        if cmd == 'H':  # Cursor position
            row = params[0] - 1 if params else 0
            col = params[1] - 1 if len(params) > 1 else 0
            self.cursor_row = max(0, min(row, self.rows - 1))
            self.cursor_col = max(0, min(col, self.cols - 1))
        elif cmd == 'J':  # Clear screen
            mode = params[0] if params else 0
            if mode == 2:  # Clear all
                self.cells = [[Cell() for _ in range(self.cols)]
                              for _ in range(self.rows)]
        elif cmd == 'm':  # SGR (text attributes)
            self.handle_sgr(params)"""
                },
                "pitfalls": [
                    "Many escape sequences have optional parameters",
                    "UTF-8 characters can span multiple bytes",
                    "Some sequences are terminal-specific"
                ],
                "concepts": ["Terminal emulation", "ANSI escape codes", "Screen buffers"],
                "estimated_hours": "8-10"
            },
            {
                "id": 3,
                "name": "Window Management",
                "description": "Support multiple panes in a split-screen layout.",
                "acceptance_criteria": [
                    "Vertical and horizontal splits",
                    "Switch focus between panes",
                    "Resize panes",
                    "Each pane runs independent shell"
                ],
                "hints": {
                    "level1": "Each pane has its own PTY, screen buffer, and render region.",
                    "level2": "Use a tree structure: nodes are either splits (container) or panes (leaf).",
                    "level3": """from dataclasses import dataclass
from typing import Optional, Union

@dataclass
class Rect:
    x: int
    y: int
    width: int
    height: int

@dataclass
class Pane:
    id: int
    master_fd: int
    pid: int
    screen: Screen
    rect: Rect

@dataclass
class Split:
    direction: str  # 'horizontal' or 'vertical'
    ratio: float  # 0.0-1.0, position of split
    first: Union['Split', Pane]
    second: Union['Split', Pane]

class Multiplexer:
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.root: Union[Split, Pane] = None
        self.focused_pane: Optional[Pane] = None
        self.panes: list[Pane] = []
        self.next_pane_id = 0

    def create_pane(self, rect: Rect) -> Pane:
        '''Create new pane with PTY'''
        master_fd, pid = create_pty_shell()
        screen = Screen(rect.height, rect.width)
        set_terminal_size(master_fd, rect.height, rect.width)

        pane = Pane(self.next_pane_id, master_fd, pid, screen, rect)
        self.next_pane_id += 1
        self.panes.append(pane)
        return pane

    def split_pane(self, pane: Pane, direction: str):
        '''Split pane horizontally or vertically'''
        old_rect = pane.rect

        if direction == 'vertical':
            # Split left/right
            w1 = old_rect.width // 2
            w2 = old_rect.width - w1 - 1  # -1 for border
            rect1 = Rect(old_rect.x, old_rect.y, w1, old_rect.height)
            rect2 = Rect(old_rect.x + w1 + 1, old_rect.y, w2, old_rect.height)
        else:
            # Split top/bottom
            h1 = old_rect.height // 2
            h2 = old_rect.height - h1 - 1
            rect1 = Rect(old_rect.x, old_rect.y, old_rect.width, h1)
            rect2 = Rect(old_rect.x, old_rect.y + h1 + 1, old_rect.width, h2)

        # Resize existing pane
        pane.rect = rect1
        pane.screen = Screen(rect1.height, rect1.width)
        set_terminal_size(pane.master_fd, rect1.height, rect1.width)

        # Create new pane
        new_pane = self.create_pane(rect2)

        # Update tree structure
        # ..."""
                },
                "pitfalls": [
                    "Resize needs to update PTY size too",
                    "Border drawing reduces usable space",
                    "Focus tracking across splits is complex"
                ],
                "concepts": ["Window management", "Tree layouts", "PTY multiplexing"],
                "estimated_hours": "8-10"
            },
            {
                "id": 4,
                "name": "Key Bindings and UI",
                "description": "Add command mode and key bindings for pane management.",
                "acceptance_criteria": [
                    "Prefix key (like Ctrl-b) enters command mode",
                    "Bindings for split, close, navigate, resize",
                    "Status bar showing pane info",
                    "Render all panes to terminal"
                ],
                "hints": {
                    "level1": "Raw terminal mode to capture all keys. Check for prefix, then command key.",
                    "level2": "After prefix, next key is command. 'v' = vertical split, 's' = horizontal, arrows = navigate.",
                    "level3": """import select
import tty
import sys

class InputHandler:
    def __init__(self, mux: Multiplexer):
        self.mux = mux
        self.prefix = b'\\x02'  # Ctrl-B
        self.in_command_mode = False

        self.bindings = {
            ord('v'): self.split_vertical,
            ord('s'): self.split_horizontal,
            ord('x'): self.close_pane,
            ord('o'): self.next_pane,
        }

    def handle_input(self, data: bytes):
        '''Process keyboard input'''
        if self.in_command_mode:
            self.in_command_mode = False
            if data in self.bindings:
                self.bindings[data]()
            return

        if data == self.prefix:
            self.in_command_mode = True
            return

        # Forward to focused pane
        if self.mux.focused_pane:
            os.write(self.mux.focused_pane.master_fd, data)

def main_loop(mux: Multiplexer):
    '''Main event loop'''
    # Set terminal to raw mode
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin)

    handler = InputHandler(mux)

    try:
        while True:
            # Watch stdin and all PTY masters
            fds = [sys.stdin] + [p.master_fd for p in mux.panes]
            readable, _, _ = select.select(fds, [], [], 0.1)

            for fd in readable:
                if fd == sys.stdin:
                    data = os.read(sys.stdin.fileno(), 1024)
                    handler.handle_input(data)
                else:
                    # Find pane and update its screen
                    pane = next(p for p in mux.panes if p.master_fd == fd)
                    data = os.read(fd, 4096)
                    if data:
                        pane.screen.write(data)

            # Render all panes
            mux.render()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)"""
                },
                "pitfalls": [
                    "Raw mode disables Ctrl-C - need explicit handling",
                    "Must restore terminal settings on exit",
                    "Rendering must be efficient to avoid flicker"
                ],
                "concepts": ["Raw terminal mode", "Event loops", "Key binding systems"],
                "estimated_hours": "8-10"
            }
        ]
    }
}

# Load and update YAML
script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir, '..', 'data', 'projects.yaml')

with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

# Add new projects to expert_projects
for project_id, project_data in new_projects.items():
    if project_id not in data['expert_projects']:
        data['expert_projects'][project_id] = project_data
        print(f"Added: {project_id}")
    else:
        print(f"Already exists: {project_id}")

# Sort expert_projects alphabetically
data['expert_projects'] = dict(sorted(data['expert_projects'].items()))

# Write back
with open(yaml_path, 'w') as f:
    f.write("""# EduTutor Crafter - Projects Data (Unified)
# Source of truth for all domains, projects, and milestones
# Version: 3.0.0 - Merged from HTML visualizer data
#
# Structure:
#   1. DOMAINS - Full taxonomy with all difficulty levels
#   2. EXPERT_PROJECTS - Detailed specifications (alphabetically sorted)

""")
    yaml.dump(data, f, Dumper=MyDumper, default_flow_style=False, allow_unicode=True, sort_keys=False, width=100)

print(f"\nTotal expert_projects: {len(data['expert_projects'])}")

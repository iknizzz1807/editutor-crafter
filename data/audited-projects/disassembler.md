# AUDIT & FIX: disassembler

## CRITIQUE
- **Length-decoding loop not emphasized**: x86 instructions are variable-length (1-15 bytes). The milestones imply a linear pipeline (prefixes → opcodes → ModRM → output) but in reality, instruction length is determined DURING decoding. The decoder must determine the length of the current instruction to know where the next one starts. This critical loop is not emphasized in any AC.
- **M4 RIP-relative addressing**: The pitfall correctly notes RIP-relative addressing requires knowing the address of the NEXT instruction. However, since you're decoding linearly, you already know the current instruction's address and its length (just decoded), so the next instruction address is current + length. This should be an explicit AC, not just a pitfall.
- **M1 scope**: Parsing both ELF and PE in the first milestone is a lot of work. Students already built an ELF parser (prerequisite). This milestone should focus on reusing/extending that work, not building a PE parser from scratch.
- **M3 opcode table size**: Building complete one-byte and two-byte opcode tables manually is massive and error-prone. The AC should allow generating tables from a machine-readable specification or covering a representative subset.
- **VEX/EVEX in M3 deliverables**: VEX and EVEX prefix handling appears as a deliverable in M3 but VEX was supposedly covered in M2. The milestones have scope overlap.
- **Missing**: No AC for handling instruction-length limits (x86 instructions cannot exceed 15 bytes; longer sequences are #UD).
- **Missing**: No AC for linear sweep vs recursive descent disassembly strategy discussion.
- **Missing test methodology**: No AC requires comparing output against objdump or ndisasm for correctness verification.

## FIXED YAML
```yaml
id: disassembler
name: x86-64 Disassembler
description: Instruction decoder for x86-64 binaries
difficulty: advanced
estimated_hours: "35-50"
essence: >
  Variable-length instruction decoding requiring byte-level parsing of
  x86-64 encoding (prefixes, opcodes, ModRM/SIB bytes, displacement,
  immediates) with opcode table lookups to translate machine code into
  human-readable assembly mnemonics using a linear sweep strategy.
why_important: >
  Building a disassembler demystifies how CPUs execute code at the lowest
  level and provides foundational knowledge for reverse engineering,
  debuggers, binary analysis tools, and security research.
learning_outcomes:
  - Implement variable-length instruction decoding with a decode-length loop
  - Design opcode lookup tables mapping byte patterns to instruction mnemonics
  - Decode x86-64 addressing modes using ModRM and SIB bytes
  - Handle instruction prefixes including legacy, REX, and (optionally) VEX
  - Build a linear-sweep disassembler processing code sections from ELF binaries
  - Debug byte-level parsing by comparing output against objdump
  - Format disassembly output in Intel and AT&T syntax
skills:
  - Binary Format Parsing
  - Instruction Encoding
  - Opcode Tables
  - Bit Manipulation
  - State Machine Design
  - Low-level Debugging
tags:
  - advanced
  - binary
  - binary-analysis
  - c
  - instruction-decoding
  - opcodes
  - rust
architecture_doc: architecture-docs/disassembler/index.md
languages:
  recommended:
    - C
    - Rust
  also_possible:
    - Python
    - Go
    - C++
resources:
  - name: Intel x86 Manual Vol. 2
    url: "https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html"
    type: reference
  - name: "Medium - Building x86-64 Disassembler"
    url: "https://medium.com/@Koukyosyumei/learning-x86-64-machine-language-and-assembly-by-implementing-a-disassembler-dccc736ae85f"
    type: tutorial
  - name: x86 Instruction Encoding (OSDev)
    url: "http://wiki.osdev.org/X86-64_Instruction_Encoding"
    type: reference
prerequisites:
  - type: skill
    name: x86 assembly basics
  - type: project
    name: elf-parser
  - type: skill
    name: Bitwise operations
milestones:
  - id: disassembler-m1
    name: ELF Code Extraction
    description: Load ELF binary and extract .text section bytes for disassembly.
    estimated_hours: "4-6"
    concepts:
      - ELF section headers
      - Virtual address vs file offset
      - Code section identification
    skills:
      - ELF header parsing (from prerequisite project)
      - Section identification by type and flags
      - Address mapping
    acceptance_criteria:
      - Parse ELF header and section header table (reuse/extend ELF parser project code)
      - Locate .text section by name or by SHF_EXECINSTR flag; handle binaries with multiple executable sections
      - Extract code bytes and record the base virtual address for address display
      - Support both 32-bit and 64-bit ELF formats; auto-detect from ELF class field
      - Load symbol table if present for function name resolution in output
      - Handle stripped binaries (no .symtab) gracefully; disassembly still works, just without symbol names
      - Verify by comparing extracted bytes against `objdump -d -j .text` hex output
    pitfalls:
      - Virtual address (sh_addr) differs from file offset (sh_offset); use file offset for reading, virtual address for display
      - Some binaries use non-standard section names; prefer flag-based identification over name matching
      - PIE executables have low virtual addresses that get relocated; display the virtual address from the ELF, not the runtime address
    deliverables:
      - ELF loader extracting executable section bytes
      - Virtual address tracking for instruction address display
      - Symbol table loader for optional function name resolution
      - Support for 32-bit and 64-bit ELF formats

  - id: disassembler-m2
    name: Prefix and Opcode Decoding
    description: Decode instruction prefixes, primary opcodes, and establish the decode-length loop.
    estimated_hours: "8-10"
    concepts:
      - Variable-length instruction decoding
      - Legacy prefixes (groups 1-4)
      - REX prefix in 64-bit mode
      - Opcode tables (1-byte and 2-byte)
      - Instruction length determination
    skills:
      - Bit manipulation and masking
      - State machine for prefix consumption
      - Opcode table design and lookup
    acceptance_criteria:
      - "Implement the core decode loop: at current offset, decode one instruction determining its total length, advance offset by that length, repeat until end of section"
      - Consume legacy prefixes (66h, 67h, F0h/LOCK, F2h/REPNE, F3h/REP, segment overrides 26h/2Eh/36h/3Eh/64h/65h) in any order
      - Detect and decode REX prefix byte (40h-4Fh range, 64-bit mode only); extract W, R, X, B bit fields
      - REX must be the last prefix before the opcode; a legacy prefix after REX cancels the REX
      - Build one-byte opcode lookup table covering at minimum the most common x86-64 instructions (MOV, PUSH, POP, ADD, SUB, CMP, JMP, JCC, CALL, RET, NOP, LEA, TEST, XOR, AND, OR, INT)
      - Build two-byte opcode table (0F xx) covering at minimum MOVZX, MOVSX, conditional jumps (0F 80-8F), SETcc, CMOV
      - Handle opcode extensions via ModRM.reg field (e.g., opcode 80h-83h use reg field to select ADD/OR/ADC/SBB/AND/SUB/XOR/CMP)
      - Reject instruction lengths exceeding 15 bytes as invalid (#UD)
      - For unrecognized opcodes, emit `.byte 0xNN` and advance by 1 byte
      - Verify decoded instruction lengths against objdump output for a test binary with diverse instructions
    pitfalls:
      - REX prefix occupies the same byte range (40h-4Fh) as INC/DEC in 32-bit mode; must know the operating mode
      - Some prefix combinations are invalid or have special meaning (e.g., F2 0F = SSE prefix, not REPNE + 0F)
      - Three-byte opcodes (0F 38 xx, 0F 3A xx) exist; handle or at least recognize and skip correctly
      - The decode loop is the core architecture; getting instruction length wrong causes cascading errors for all subsequent instructions
    deliverables:
      - Core decode-length loop advancing through the code section
      - Prefix consumer handling legacy and REX prefixes
      - One-byte and two-byte opcode lookup tables
      - Opcode extension handling via ModRM.reg
      - Invalid/unknown opcode fallback

  - id: disassembler-m3
    name: ModRM, SIB, and Operand Decoding
    description: Decode ModRM and SIB bytes to determine instruction operands.
    estimated_hours: "8-10"
    concepts:
      - ModRM byte (mod, reg, rm fields)
      - SIB byte (scale, index, base)
      - Addressing modes (register, memory, RIP-relative)
      - Displacement and immediate values
    skills:
      - Complex addressing mode decoding
      - Register encoding with REX extensions
      - Effective address computation
    acceptance_criteria:
      - Parse ModRM byte into mod (2 bits), reg (3 bits), rm (3 bits) fields
      - Decode all mod values: 00 (memory, no disp), 01 (memory + disp8), 10 (memory + disp32), 11 (register)
      - When mod != 11 and rm == 100b, parse the SIB byte: scale (2 bits), index (3 bits), base (3 bits)
      - Handle SIB special case: base == 101b with mod == 00 means disp32 with no base register
      - Handle SIB special case: index == 100b means no index register (unless REX.X extends it)
      - "Decode RIP-relative addressing: in 64-bit mode, mod == 00 and rm == 101b means [RIP + disp32]. Compute the target address as (current_instruction_address + current_instruction_length + disp32)"
      - Apply REX.R, REX.B, REX.X to extend reg, rm, and SIB index/base to access r8-r15
      - Decode immediate operands (imm8, imm16, imm32, imm64) based on opcode and operand size
      - Handle operand size override (66h prefix) switching between 16-bit and 32-bit operands
      - Handle address size override (67h prefix) switching between 32-bit and 64-bit addressing
      - Verify operand decoding against objdump for a test binary with diverse addressing modes
    pitfalls:
      - "RIP-relative target computation requires knowing the TOTAL instruction length (including all prefixes, opcode, ModRM, SIB, displacement, and immediate). You must have fully decoded the instruction before computing the target."
      - In 32-bit mode, mod==00 rm==101b means [disp32] (absolute), NOT RIP-relative; mode matters
      - REX.W changes operand size to 64-bit for most instructions but NOT for MOV with immediate (some MOV variants use the REX.W bit differently)
      - Sign extension of 8-bit and 32-bit displacements to 64-bit is required for correct address computation
    deliverables:
      - ModRM byte parser with all mod/rm combinations
      - SIB byte parser with scale/index/base
      - RIP-relative address computation
      - REX register extension application
      - Displacement and immediate value extraction
      - Operand size and address size override handling

  - id: disassembler-m4
    name: Output Formatting and Verification
    description: Format disassembly output with addresses, bytes, mnemonics, and operands.
    estimated_hours: "6-8"
    concepts:
      - Intel vs AT&T syntax
      - Address calculation for branch targets
      - Symbol resolution
      - Output formatting
    skills:
      - Assembly syntax formatting
      - Hexadecimal output formatting
      - Branch target address arithmetic
      - Graceful error handling
    acceptance_criteria:
      - "Display each instruction as: virtual_address: raw_hex_bytes  mnemonic operands"
      - Default to Intel syntax (destination first); support AT&T syntax (source first, % register prefix, $ immediate prefix) via --att flag
      - Resolve relative branch and call targets to absolute virtual addresses; display as hex or symbol name if available
      - Insert symbol name labels (e.g., '<main>:') before instructions at known symbol addresses
      - Handle undefined/invalid opcodes by emitting `.byte 0xNN` without crashing
      - Pad hex bytes column to consistent width for visual alignment (e.g., max 15 bytes = 45 hex chars)
      - "Comprehensive verification: disassemble the .text section of /bin/ls (or similar) and diff against `objdump -d -M intel` output. Achieve >= 95% instruction-level match for the supported instruction subset"
      - Display summary statistics: total instructions decoded, unrecognized opcodes count, coverage percentage
    pitfalls:
      - Relative branch targets are signed offsets from the END of the current instruction (next instruction address), not from the start
      - AT&T syntax reverses operand order AND uses different size suffixes (movl, movq) vs Intel (mov dword, mov qword)
      - Very long instructions (many prefixes) can overflow a fixed-width hex bytes column
      - Invalid opcodes in data sections (if disassembling non-code by mistake) will produce garbage; linear sweep cannot distinguish code from data
    deliverables:
      - Intel syntax formatter with proper operand ordering
      - AT&T syntax formatter option
      - Branch target resolution and symbol labeling
      - Invalid opcode graceful handling
      - Verification against objdump with coverage report

  - id: disassembler-m5
    name: Extended Instruction Support
    description: Expand instruction coverage and add analysis features.
    estimated_hours: "6-8"
    concepts:
      - Three-byte opcodes (0F 38, 0F 3A)
      - Common SSE/SSE2 instructions
      - Basic control flow analysis
    skills:
      - Opcode table extension
      - Instruction set reference reading
      - Basic static analysis
    acceptance_criteria:
      - Add three-byte opcode tables (0F 38 xx and 0F 3A xx) covering common SSE4 instructions
      - Add common SSE/SSE2 instructions (MOVAPS, MOVUPS, ADDPS, MULPS, XORPS, PXOR, etc.) using 66/F2/F3 mandatory prefixes
      - Identify function boundaries using RET instructions and symbol table entries
      - Display cross-references for CALL instructions (list of functions called by each function)
      - Achieve >= 98% instruction match against objdump for standard system binaries
      - "Optional: implement basic VEX prefix (2-byte and 3-byte) decoding for AVX instructions"
    pitfalls:
      - SSE instructions reuse legacy prefixes (66h, F2h, F3h) as mandatory prefixes changing the opcode meaning, not as operand size overrides
      - VEX prefix encoding is complex (encodes REX bits, implied 0F/0F38/0F3A, and operand count)
      - Function boundary detection is unreliable with linear sweep; data embedded in code causes desync
    deliverables:
      - Extended opcode tables for SSE/SSE2 instructions
      - Three-byte opcode support
      - Function boundary identification
      - Cross-reference listing for CALL targets
      - Optional VEX prefix support

```
# AUDIT & FIX: build-linker

## CRITIQUE
- **Logical Gap (Confirmed - Symbol/Relocation Coupling):** Milestones 2 (Symbol Resolution) and 3 (Relocations) are tightly coupled; symbol resolution must provide the absolute addresses that the relocation phase then patches into the instruction stream. The original structure treats them as independent, but relocation cannot proceed without resolved symbol addresses.
- **Technical Inaccuracy (Confirmed - Entry Point):** Milestone 4 mentions PT_LOAD segments but for an executable to run 'without external tools' (as per AC), the linker must also define the entry point symbol (usually _start) in the ELF header. The original doesn't mention setting e_entry in the ELF header.
- **Hour Ranges:** Using ranges like '7-10' is imprecise. Converting to single estimates.
- **Estimated Hours:** 30-45 range is reasonable; estimate ~38 hours.

## FIXED YAML
```yaml
id: build-linker
name: Static Linker
description: >-
  Multi-file ELF linker with section merging, symbol resolution, relocation
  processing, and executable generation with proper entry point configuration.
difficulty: advanced
estimated_hours: 38
essence: >-
  Symbol table construction, address relocation, and section merging to
  transform position-independent object code into a fixed-address executable
  binary with resolved cross-file references and a defined entry point.
why_important: >-
  Building a linker demystifies the final stage of compilation and reveals how
  high-level code becomes machine-executable programs, a foundational
  understanding for systems programming, toolchain development, and low-level
  debugging.
learning_outcomes:
  - Parse ELF object files and extract sections, symbols, and relocations
  - Merge sections from multiple object files with alignment handling
  - Resolve symbols across translation units with strong/weak rules
  - Apply relocations to patch addresses in merged sections
  - Generate valid ELF executables with program headers and entry point
skills:
  - ELF Format
  - Symbol Resolution
  - Relocation Processing
  - Section Merging
  - Executable Generation
tags:
  - linker
  - elf
  - systems-programming
  - advanced
  - toolchain
architecture_doc: architecture-docs/build-linker/index.md
languages:
  recommended:
    - C
    - Rust
    - C++
  also_possible:
    - Go
    - Zig
resources:
  - name: Linkers and Loaders
    url: https://www.iecc.com/linker/
    type: book
  - name: ELF Format
    url: https://refspecs.linuxbase.org/elf/elf.pdf
    type: specification
  - name: Beginner's Guide to Linkers
    url: https://www.lurklurk.org/linkers/linkers.html
    type: tutorial
prerequisites:
  - type: skill
    name: C programming and pointers
  - type: skill
    name: Understanding of object files and compilation
  - type: skill
    name: Basic x86-64 assembly concepts
milestones:
  - id: build-linker-m1
    name: Section Merging
    description: >-
      Parse multiple ELF object files and merge corresponding sections.
    acceptance_criteria:
      - Read multiple ELF .o files and identify matching section types (.text, .data, .bss, .rodata)
      - Concatenate same-name sections from different files maintaining alignment requirements
      - Track section-to-output mapping (which input section maps to which output offset)
      - Generate output section table with correct sizes and offsets
      - Handle .bss sections which occupy no file space but need virtual address space allocation
    pitfalls:
      - Sections may require alignment padding between merged inputs—calculate padding correctly
      - .bss sections occupy no file space but need virtual address space allocation
      - Section flags (SHF_ALLOC, SHF_WRITE, SHF_EXECINSTR) must be consistent across merged sections
    concepts:
      - Sections are the basic units of content in object files
      - Merging combines same-named sections from multiple inputs
      - Alignment ensures sections start at valid addresses for the architecture
    skills:
      - Binary file parsing
      - Section header manipulation
      - Memory alignment computation
      - ELF format specification
    deliverables:
      - Multi-file ELF reader loading section data from multiple .o files
      - Section merging with alignment-aware concatenation
      - Input-to-output offset mapping table for relocation processing
      - Output section table with correct sizes, offsets, and flags
    estimated_hours: 10

  - id: build-linker-m2
    name: Symbol Resolution
    description: >-
      Build a global symbol table resolving references across object files.
    acceptance_criteria:
      - Collect all global symbols from all input files into a unified symbol table
      - Detect and report undefined symbols that have no definition in any input file
      - Handle duplicate symbol definitions following strong/weak symbol rules
      - Resolve local symbols within their defining translation unit only
      - Assign final virtual addresses to all global symbols based on their section location
    pitfalls:
      - Multiple strong definitions of the same symbol must be an error
      - Weak symbols are overridden by strong symbols but valid when alone
      - COMMON symbols (uninitialized globals) have special merging rules based on size (largest wins)
    concepts:
      - Symbol resolution determines the final address of each named entity
      - Strong vs weak symbols control priority when multiple definitions exist
      - Translation units isolate local symbols from each other
    skills:
      - Symbol table construction
      - Hash table implementation
      - Name mangling handling
      - Duplicate symbol detection
    deliverables:
      - Global symbol table aggregating symbols from all input object files
      - Undefined symbol detection and error reporting
      - Strong/weak symbol resolution following ELF linking rules
      - Local symbol scoping restricted to defining translation unit
      - Final address assignment for all resolved symbols
    estimated_hours: 10

  - id: build-linker-m3
    name: Relocation Processing
    description: >-
      Apply relocations to fix up addresses in merged sections using resolved
      symbols.
    acceptance_criteria:
      - Process R_X86_64_PC32 (relative) and R_X86_64_64 (absolute) relocation types
      - Calculate final symbol addresses from section base + offset in merged output
      - Patch relocation sites in section data with computed addresses
      - Handle addends from .rela sections correctly in address calculations
      - Process relocations in dependency order (symbols must be resolved first)
    pitfalls:
      - PC-relative relocations subtract the relocation site address from the symbol address
      - Addend is added to the computed address, not the symbol value alone
      - Truncation errors occur when 64-bit addresses don't fit in 32-bit relocation fields
      - Relocation processing requires resolved symbol addresses—do after symbol resolution
    concepts:
      - Relocations are fixup instructions for addresses unknown at compile time
      - PC-relative addressing uses offsets from the current instruction
      - Addends modify the base address for specific addressing modes
    skills:
      - Address calculation
      - Bitwise operations for patching
      - Relocation entry processing
      - Offset arithmetic
    deliverables:
      - Relocation processor handling R_X86_64_PC32 and R_X86_64_64 types
      - Address calculation using merged section bases and symbol offsets
      - Section data patching with computed relocation values
      - Overflow detection for truncating relocations
    estimated_hours: 10

  - id: build-linker-m4
    name: Executable Generation
    description: >-
      Write a complete ELF executable with program headers and entry point.
    acceptance_criteria:
      - Generate valid ELF header with correct entry point address (e_entry field)
      - Entry point is set to the _start symbol address (or main if no _start)
      - Create program headers (PT_LOAD) for text and data segments with proper permissions
      - Set up proper page-aligned segment layout for virtual memory loading
      - Text segment has PF_R|PF_X permissions, data segment has PF_R|PF_W permissions
      - Produced executable runs correctly on Linux without any external tools
    pitfalls:
      - PT_LOAD segments must be page-aligned (4096 bytes) for mmap loading
      - Entry point must point to _start symbol, not main (libc provides _start calling main)
      - Text segment needs PF_R|PF_X, data needs PF_R|PF_W—these must be separate PT_LOAD segments
      - For static executables without libc, _start is the true entry point
    concepts:
      - Program headers describe segments for the loader
      - Entry point is where execution begins
      - Segment permissions enforce memory protection
    skills:
      - Program header creation
      - Virtual address assignment
      - Segment permission setting
      - ELF file writing
    deliverables:
      - ELF executable writer producing valid header with e_entry set to _start address
      - Page-aligned segment layout for text and data
      - Entry point configuration pointing to _start symbol
      - Working executable that runs on Linux verified with simple test programs
    estimated_hours: 8
```

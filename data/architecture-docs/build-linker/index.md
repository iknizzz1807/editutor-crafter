# 🎯 Project Charter: Static Linker
## What You Are Building
A multi-file ELF64 static linker that transforms position-independent x86-64 object files (`.o`) into loadable Linux executables. Your linker will parse ELF relocatable files, merge sections from multiple translation units with proper alignment, resolve symbols across files following ELF strong/weak rules, apply relocations to patch addresses using PC-relative and absolute formulas, and generate a complete executable with valid program headers and entry point configuration. By the end, your linker will produce binaries that boot and run on real Linux systems.
## Why This Project Exists
Every program you've ever compiled passes through a linker, yet most developers treat it as a black box that magically produces executables. Building a linker from scratch reveals the hidden mechanics of how multi-file C programs become runnable binaries—why undefined symbol errors occur, how libraries work at the binary level, why linking order matters, and what the OS loader actually expects. This knowledge is foundational for systems programming, compiler development, binary analysis, security research, and debugging complex linking issues that would otherwise be incomprehensible.
## What You Will Be Able to Do When Done
- Parse ELF64 object files and extract sections, symbols, and relocation entries
- Merge sections from multiple object files while maintaining alignment constraints
- Build a global symbol table with strong/weak resolution and COMMON symbol handling
- Apply PC-relative and absolute relocations to patch code and data addresses
- Detect overflow errors when computed addresses don't fit in relocation fields
- Generate valid ELF executables with proper program headers and segment permissions
- Set the entry point to `_start` (or `main` as fallback) for execution
- Produce working Linux x86-64 executables that run without any external tools
- Debug linking errors by understanding symbol resolution and relocation mechanics
## Final Deliverable
A complete static linker (~2,000-3,000 lines of C code across 30+ source files) that takes multiple `.o` files as input and produces a working executable. The linker handles `.text`, `.data`, `.rodata`, and `.bss` sections with proper page alignment (4096 bytes), resolves all cross-file symbol references, applies relocations correctly, and generates an ELF file with PT_LOAD segments that the Linux loader can execute. Test with simple programs: link assembly files with `call` instructions across files, verify data references work, confirm the entry point is correct, and run the resulting executable.
## Is This Project For You?
**You should start this if you:**
- Are comfortable with C programming and pointer manipulation
- Understand basic x86-64 assembly (registers, instructions, calling conventions)
- Know what object files and compilation are at a high level
- Can work with binary data structures and bitwise operations
- Have debugged programs with segmentation faults and memory issues
**Come back after you've learned:**
- C pointers and memory management — [Learn C the Hard Way](https://learncodethehardway.org/c/) or K&R Chapter 5-6
- Basic assembly reading — [x86-64 Assembly Guide](https://www.cs.virginia.edu/~evans/cs216/guides/x86.html)
- Binary representation (hex, little-endian) — [Computer Systems: A Programmer's Perspective Chapter 2](https://csapp.cs.cmu.edu/)
## Estimated Effort
| Phase | Time |
|-------|------|
| Section Merging: Parse ELF headers, extract sections, build mapping table | ~10 hours |
| Symbol Resolution: Build global symbol table, resolve strong/weak/COMMON | ~10 hours |
| Relocation Processing: Apply patches with PC-relative and absolute formulas | ~10 hours |
| Executable Generation: Write ELF headers, program headers, segments | ~8 hours |
| **Total** | **~38 hours** |
## Definition of Done
The project is complete when:
- The linker successfully parses multiple x86-64 ELF relocatable object files without errors
- All sections are merged with correct alignment padding and input-to-output offset mapping
- The global symbol table resolves all non-weak undefined symbols and correctly handles strong/weak/COMMON rules
- All relocations are processed and patched in the output buffer without overflow errors
- The generated executable has valid ELF headers verifiable with `readelf -h`
- The generated executable has correct PT_LOAD segments verifiable with `readelf -l`
- The executable runs on Linux x86-64 and produces correct exit codes from test programs
- A simple two-file program with cross-file function calls links and executes correctly

---

# 📚 Before You Read This: Prerequisites & Further Reading
> **Read these first.** The Atlas assumes you are familiar with the foundations below.
> Resources are ordered by when you should encounter them — some before you start, some at specific milestones.
---
## Foundational Knowledge
### ELF Format Specification
**Read BEFORE starting this project — required foundational knowledge.**
| Resource | Type | Details |
|----------|------|---------|
| **ELF Format (Wikipedia)** | Reference | https://en.wikipedia.org/wiki/Executable_and_Linkable_Format |
| **System V ABI, Part 1** | Spec | Tool Interface Standards (TIS), Executable and Linking Format (ELF) Specification, Version 1.2 |
| **Linux ELF man page** | Reference | `man 5 elf` on any Linux system |
**Why**: The linker's entire job is to manipulate ELF files. You cannot understand what the linker does without understanding the container format — sections, segments, headers, and their relationships.
---
### Computer Systems: A Programmer's Perspective
**Read BEFORE starting this project — required foundational knowledge.**
| Resource | Type | Details |
|----------|------|---------|
| **CS:APP Chapter 7: Linking** | Book Chapter | Bryant & O'Hallaron, *Computer Systems: A Programmer's Perspective*, 3rd ed., Chapter 7 |
**Why**: The gold standard introduction to linking. Explains why linkers exist, how they work at a conceptual level, and introduces the key ideas (symbol resolution, relocation) you'll implement.
---
### x86-64 Instruction Encoding
**Read AFTER Milestone 3 (Relocation Processing) — you'll have enough context to appreciate the encoding details.**
| Resource | Type | Details |
|----------|------|---------|
| **Intel SDM, Vol. 2** | Spec | Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 2: Instruction Set Reference |
| **AMD APM, Vol. 3** | Spec | AMD64 Architecture Programmer's Manual, Volume 3: General-Purpose and System Instructions |
**Why**: When debugging why a `call` instruction's displacement is calculated wrong, you need to understand how x86-64 encodes RIP-relative addressing. These are the authoritative references.
---
## Deep Dive Resources
### ELF Symbol Resolution Rules
**Read BEFORE Milestone 2 (Symbol Resolution).**
| Resource | Type | Details |
|----------|------|---------|
| **Linkers and Loaders** | Book | Levine, John R. *Linkers and Loaders*. Morgan Kaufmann, 1999. Chapters 3-4. |
**Why**: The definitive book on linking. Chapter 4 covers symbol resolution rules (strong vs weak, COMMON merging) that Milestone 2 implements. Though predates modern ELF, the concepts are unchanged.
---
### Relocation Type Semantics
**Read AFTER Milestone 3 (Relocation Processing) — the formulas will make sense once you've implemented them.**
| Resource | Type | Details |
|----------|------|---------|
| **System V ABI, AMD64 Supplement** | Spec | System V Application Binary Interface, AMD64 Architecture Processor Supplement, Version 1.0, Section 4.4: "Relocation" |
**Why**: The authoritative reference for what each `R_X86_64_*` relocation type means. Documents the exact formulas (S + A, S + A - P) that your patch calculator implements.
---
### Linux Loader Behavior
**Read BEFORE Milestone 4 (Executable Generation).**
| Resource | Type | Details |
|----------|------|---------|
| **`fs/binfmt_elf.c`** | Code | Linux kernel source, `fs/binfmt_elf.c` — the ELF loader implementation |
**Why**: The loader is your "customer" in Milestone 4. Reading how `load_elf_binary()` processes program headers, validates alignment, and sets up memory mappings reveals what your generated executable must provide.
---
## Reference Implementation
### GNU ld (BFD Linker)
**Read AFTER completing the project — to compare approaches.**
| Resource | Type | Details |
|----------|------|---------|
| **GNU ld source** | Code | binutils-gdb repository, `ld/` directory. Key files: `ldmain.c`, `lang.c`, `emultempl/elf32.em` |
**Why**: The production linker you're building a simplified version of. After you complete each milestone, reading the corresponding code in `ld` reveals how a real linker handles edge cases you may have simplified.
---
### mold Linker
**Read AFTER completing the project — modern alternative design.**
| Resource | Type | Details |
|----------|------|---------|
| **mold source** | Code | https://github.com/rui314/mold. Key files: `src/input-sections.cc`, `src/symbol-table.cc`, `src/passes.cc` |
**Why**: A modern, high-performance linker with cleaner code than GNU ld. Demonstrates parallel linking, incremental updates, and other advanced techniques. Good for understanding how your simple sequential approach scales.
---
## Debugging Tools
### readelf and objdump
**Use throughout the project — essential for debugging.**
| Resource | Type | Details |
|----------|------|---------|
| **`readelf -h`** | Tool | Display ELF header |
| **`readelf -S`** | Tool | Display section headers |
| **`readelf -s`** | Tool | Display symbol table |
| **`readelf -r`** | Tool | Display relocations |
| **`readelf -l`** | Tool | Display program headers (segments) |
| **`objdump -d`** | Tool | Disassemble .text section |
| **`objdump -r`** | Tool | Display relocations with disassembly |
**Why**: Every milestone produces output that should be verified with these tools. If your linker produces an executable, `readelf -h` confirms the header is valid. If relocations are processed, `objdump -d` shows the patched addresses.
---
## Summary Reading Order
| When | What | Why |
|------|------|-----|
| Before starting | ELF Wikipedia + `man 5 elf` + CS:APP Ch. 7 | Foundation |
| Before Milestone 2 | Linkers and Loaders, Ch. 3-4 | Symbol resolution rules |
| Before Milestone 4 | `fs/binfmt_elf.c` (skim) | Understand the loader's expectations |
| After Milestone 3 | AMD64 ABI, §4.4 | Relocation formulas reference |
| After completing | mold source | Modern linker architecture |
| Throughout | `readelf`, `objdump` | Verify every milestone's output |

---

# Static Linker

A static linker is the final stage of the compilation pipeline that transforms individual object files into a cohesive executable. It performs three fundamental operations: section merging (combining code and data from multiple translation units), symbol resolution (connecting references to definitions across files), and relocation processing (patching addresses that couldn't be determined at compile time). The result is a valid ELF executable ready for the operating system loader.

This project reveals the hidden mechanics of how your multi-file C programs actually become runnable binaries. You'll understand why undefined symbol errors occur, how libraries work at the binary level, and why linking order matters. The linker sits at the intersection of the compiler's output and the loader's input, making it essential knowledge for debugging linking errors, understanding binary size, and working with custom toolchains.

Building a linker from scratch teaches you the ELF format in depth—one of the most important binary formats in modern computing. You'll handle alignment, address spaces, symbol visibility rules, and machine code patching. This knowledge transfers directly to debuggers, binary analyzers, JIT compilers, and security research.


<!-- MS_ID: build-linker-m1 -->
# Section Merging: Weaving Object Files into a Whole
You've compiled your C files. Each `.c` became a `.o` — an object file containing machine code, but with holes where addresses should be. Now you face the linker's first real challenge: taking these fragments and stitching them together into something coherent.
This milestone is about understanding what object files actually contain, why they're structured the way they are, and how to merge them while maintaining the precise bookkeeping that later stages depend on.
## The Tension: Fragments That Must Become Whole
Here's what you probably think happens during linking: the compiler outputs some code, the linker concatenates it, done. But that model fails immediately when you ask basic questions:
- `main()` in `main.o` calls `helper()` in `utils.o`. How does `main.o` know where `helper` will live in the final binary?
- `main.o` has 100 bytes of `.text`. `utils.o` has 50 bytes. Where does `utils.o`'s code end up? What if there's also a `lib.o`?
- `main.o`'s `.data` section needs 8-byte alignment. `utils.o`'s needs 16-byte alignment. How do you honor both?

![Section Merging State Machine](./diagrams/tdd-diag-007.svg)

![Linker Pipeline: Satellite View](./diagrams/diag-satellite-overview.svg)

The fundamental tension: **object files are designed to be position-independent, but executables must have fixed addresses**. Every byte in an object file might end up anywhere in the final executable, depending on what other files are linked and in what order. Your job is to:
1. Gather all the fragments
2. Assign each fragment a final location
3. Remember where everything went (because symbol resolution and relocation need this map)
The output of this milestone isn't an executable yet — it's a **merged layout** with a complete **input-to-output mapping table**. This table is the linker's internal bible; without it, you can't fix up addresses later.
## What's Inside an Object File?
Before you can merge sections, you need to understand what you're working with. An ELF object file (`ET_REL` type) isn't a runnable program — it's a collection of ingredients:

![File Offset vs Virtual Address Assignment](./diagrams/tdd-diag-014.svg)

> **🔑 Foundation: ELF header and section header table structure**
> 
> ## What It Is
The **ELF header** and **section header table** are the two bookends of an ELF (Executable and Linkable Format) binary file. Together, they define the file's structure and tell the loader/linker how to interpret everything in between.
**The ELF Header** sits at the very start of the file (byte 0). It's a fixed-size structure that answers three critical questions:
- Is this a valid ELF file? (magic bytes: `0x7f 'E' 'L' 'F'`)
- What kind of binary is this? (32-bit vs 64-bit, little vs big endian, executable vs shared object vs relocatable)
- Where do I find the rest of the structure? (entry point, program header table location, section header table location)
**The Section Header Table** sits at the end of the file. It's an array of section headers, each describing one "section" of the binary — a named, contiguous chunk of data with specific purpose (code, data, symbols, string tables, relocation entries, etc.).
```c
// 64-bit ELF Header (simplified)
typedef struct {
    unsigned char e_ident[16];    // Magic + class + endian + version
    uint16_t e_type;              // ET_EXEC, ET_DYN, ET_REL, etc.
    uint16_t e_machine;           // Architecture (EM_X86_64, EM_ARM, etc.)
    uint32_t e_version;           // ELF version
    uint64_t e_entry;             // Entry point virtual address
    uint64_t e_phoff;             // Program header table file offset
    uint64_t e_shoff;             // Section header table file offset
    uint32_t e_flags;             // Processor-specific flags
    uint16_t e_ehsize;            // ELF header size
    uint16_t e_phentsize;         // Program header entry size
    uint16_t e_phnum;             // Number of program headers
    uint16_t e_shentsize;         // Section header entry size
    uint16_t e_shnum;             // Number of section headers
    uint16_t e_shstrndx;          // Index of section name string table
} Elf64_Ehdr;
// Section Header (simplified)
typedef struct {
    uint32_t sh_name;             // Index into section name string table
    uint32_t sh_type;             // SHT_PROGBITS, SHT_SYMTAB, SHT_STRTAB, etc.
    uint64_t sh_flags;            // SHF_EXECINSTR, SHF_WRITE, SHF_ALLOC, etc.
    uint64_t sh_addr;             // Virtual address (if loaded)
    uint64_t sh_offset;           // File offset
    uint64_t sh_size;             // Section size in bytes
    uint32_t sh_link;             // Link to another section (type-dependent)
    uint32_t sh_info;             // Extra info (type-dependent)
    uint64_t sh_addralign;        // Alignment constraint
    uint64_t sh_entsize;          // Entry size (for tables)
} Elf64_Shdr;
```
## Why You Need This Right Now
If you're building a compiler, linker, debugger, binary analysis tool, or working on OS-level code — you're going to parse or generate ELF files directly. Understanding this structure is non-negotiable for:
- **Implementing a compiler backend**: Your code generator produces object files that must conform to ELF structure
- **Writing a linker**: You merge multiple ELF files by manipulating their sections
- **Building debuggers/profilers**: You read symbol tables and section maps to map addresses to source
- **Security analysis/exploitation**: You need to understand how malicious code might hide in or manipulate ELF structure
- **OS development**: Your loader must parse ELF headers to load executables into memory
The `e_shstrndx` field deserves special mention: it tells you which section contains the *names* of all other sections. Section headers only store an index into this string table, not the name itself. This indirection is a common source of bugs when parsing manually.
## Key Mental Model
**Think of an ELF file as a database with two indexes.**
The ELF header is the "metadata table" — it tells you the schema (architecture, file type) and where to find the two indexes.
The section header table is the "catalog" — it's an array of records, each pointing to a named blob somewhere in the file. The actual section data can be in any order; the headers tell you where each piece lives.
```
┌─────────────────────────────────────────┐
│ ELF Header (metadata + pointers)        │
├─────────────────────────────────────────┤
│ Program Headers (for loading)           │
├─────────────────────────────────────────┤
│ ... section data scattered throughout ...│
│ .text    .data    .bss    .symtab       │
│ .strtab  .rela.text  .debug_info  ...   │
├─────────────────────────────────────────┤
│ Section Header Table (the catalog)      │
│ [0] NULL section                        │
│ [1] .interp → offset 0x238, size 0x1c   │
│ [2] .note.ABI-tag → offset 0x254, ...   │
│ [n] ...                                 │
└─────────────────────────────────────────┘
```
**Critical insight**: Sections are a *link-time* concept. The section header table is optional for execution — the runtime loader uses *program headers* (segments), not sections. A stripped binary often has no section header table. But for any tool that needs to understand the binary's logical structure (debuggers, linkers, disassemblers), the section header table is essential.


![OutputSection Struct Memory Layout](./diagrams/tdd-diag-010.svg)

![ELF Object File Anatomy](./diagrams/diag-elf-object-anatomy.svg)

An object file contains:
- **ELF Header**: Magic number, architecture info, section header table location
- **Section Header Table**: An array describing every section (name, type, flags, offset, size, alignment)
- **Sections**: The actual content — `.text` (code), `.data` (initialized data), `.bss` (uninitialized data), `.rodata` (read-only data), `.symtab` (symbols), `.strtab` (strings), `.rela.*` (relocations)
The sections you care about for merging are:
| Section | Content | In File? | In Memory? | Flags |
|---------|---------|----------|------------|-------|
| `.text` | Executable code | Yes | Yes | `SHF_ALLOC | SHF_EXECINSTR` |
| `.data` | Initialized writable data | Yes | Yes | `SHF_ALLOC | SHF_WRITE` |
| `.rodata` | Read-only data (strings, const) | Yes | Yes | `SHF_ALLOC` |
| `.bss` | Uninitialized data (globals) | **No** | Yes | `SHF_ALLOC | SHF_WRITE` |
The `.bss` row should catch your eye: it occupies **no file space** but needs **virtual address space**. This is a crucial distinction we'll revisit.
### Parsing the ELF Header
Let's start with code. You need structures that match the ELF format exactly. On a 64-bit system (which we'll assume throughout — the linker targets x86-64):
```c
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
// ELF64 header — matches /usr/include/elf.h
typedef struct {
    uint8_t  e_ident[16];    // Magic number and info
    uint16_t e_type;         // Object file (ET_REL = 1), executable (ET_EXEC = 2), etc.
    uint16_t e_machine;      // Architecture (EM_X86_64 = 62)
    uint32_t e_version;      // ELF version
    uint64_t e_entry;        // Entry point (0 for object files)
    uint64_t e_phoff;        // Program header offset (0 for object files)
    uint64_t e_shoff;        // Section header table offset
    uint32_t e_flags;        // Processor flags
    uint16_t e_ehsize;       // ELF header size
    uint16_t e_phentsize;    // Program header entry size
    uint16_t e_phnum;        // Number of program headers (0 for objects)
    uint16_t e_shentsize;    // Section header entry size
    uint16_t e_shnum;        // Number of section headers
    uint16_t e_shstrndx;     // Section name string table index
} Elf64_Ehdr;
// Section header — describes one section
typedef struct {
    uint32_t sh_name;        // Section name (index into string table)
    uint32_t sh_type;        // Section type (SHT_PROGBITS, SHT_NOBITS, etc.)
    uint64_t sh_flags;       // Flags (SHF_ALLOC, SHF_WRITE, SHF_EXECINSTR)
    uint64_t sh_addr;        // Virtual address (0 for object files)
    uint64_t sh_offset;      // File offset
    uint64_t sh_size;        // Section size
    uint32_t sh_link;        // Link to another section
    uint32_t sh_info;        // Extra info
    uint64_t sh_addralign;   // Alignment requirement
    uint64_t sh_entsize;     // Entry size if section holds table
} Elf64_Shdr;
// Key constants
#define EI_NIDENT     16
#define ELFMAG        "\x7fELF"
#define ELFCLASS64    2
#define ELFDATA2LSB   1
#define ET_REL        1
#define EM_X86_64     62
#define SHT_NULL      0
#define SHT_PROGBITS  1    // .text, .data, .rodata
#define SHT_SYMTAB    2    // Symbol table
#define SHT_STRTAB    3    // String table
#define SHT_RELA      4    // Relocation entries with addends
#define SHT_NOBITS    8    // .bss — no file space
#define SHF_WRITE     0x1
#define SHF_ALLOC     0x2
#define SHF_EXECINSTR 0x4
```
Now let's parse an object file and extract its sections:
```c
typedef struct {
    char name[64];           // Section name
    uint32_t type;           // SHT_PROGBITS, SHT_NOBITS, etc.
    uint64_t flags;          // SHF_* flags
    uint64_t align;          // Alignment requirement
    uint64_t size;           // Size in bytes
    uint8_t *data;           // Section content (NULL for .bss)
    uint64_t file_offset;    // Where in input file
} InputSection;
typedef struct {
    char filename[256];
    Elf64_Ehdr ehdr;
    InputSection *sections;
    uint16_t section_count;
    char *shstrtab;          // Section name string table
} ObjectFile;
```
```c
int parse_object_file(const char *filename, ObjectFile *obj) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        perror("fopen");
        return -1;
    }
    // Read ELF header
    if (fread(&obj->ehdr, sizeof(Elf64_Ehdr), 1, f) != 1) {
        fprintf(stderr, "Failed to read ELF header\n");
        fclose(f);
        return -1;
    }
    // Validate magic
    if (memcmp(obj->ehdr.e_ident, ELFMAG, 4) != 0) {
        fprintf(stderr, "Not an ELF file: %s\n", filename);
        fclose(f);
        return -1;
    }
    // Verify it's a relocatable object file
    if (obj->ehdr.e_type != ET_REL) {
        fprintf(stderr, "Not a relocatable object file (type=%d)\n", 
                obj->ehdr.e_type);
        fclose(f);
        return -1;
    }
    strncpy(obj->filename, filename, sizeof(obj->filename) - 1);
    // Read section header string table first (to get section names)
    if (obj->ehdr.e_shstrndx == 0) {
        fprintf(stderr, "No section name string table\n");
        fclose(f);
        return -1;
    }
    // Seek to section header for string table
    fseek(f, obj->ehdr.e_shoff + obj->ehdr.e_shstrndx * obj->ehdr.e_shentsize, 
          SEEK_SET);
    Elf64_Shdr shstrtab_shdr;
    if (fread(&shstrtab_shdr, sizeof(Elf64_Shdr), 1, f) != 1) {
        fclose(f);
        return -1;
    }
    // Read the string table content
    obj->shstrtab = malloc(shstrtab_shdr.sh_size);
    fseek(f, shstrtab_shdr.sh_offset, SEEK_SET);
    if (fread(obj->shstrtab, shstrtab_shdr.sh_size, 1, f) != 1) {
        fclose(f);
        return -1;
    }
    // Now read all section headers
    obj->section_count = obj->ehdr.e_shnum;
    obj->sections = calloc(obj->section_count, sizeof(InputSection));
    fseek(f, obj->ehdr.e_shoff, SEEK_SET);
    for (int i = 0; i < obj->section_count; i++) {
        Elf64_Shdr shdr;
        if (fread(&shdr, sizeof(Elf64_Shdr), 1, f) != 1) {
            fclose(f);
            return -1;
        }
        // Get section name from string table
        const char *name = obj->shstrtab + shdr.sh_name;
        strncpy(obj->sections[i].name, name, sizeof(obj->sections[i].name) - 1);
        obj->sections[i].type = shdr.sh_type;
        obj->sections[i].flags = shdr.sh_flags;
        obj->sections[i].align = shdr.sh_addralign;
        obj->sections[i].size = shdr.sh_size;
        obj->sections[i].file_offset = shdr.sh_offset;
        // Read section data (except for .bss which has no file content)
        if (shdr.sh_type != SHT_NOBITS && shdr.sh_size > 0) {
            obj->sections[i].data = malloc(shdr.sh_size);
            long saved_pos = ftell(f);
            fseek(f, shdr.sh_offset, SEEK_SET);
            if (fread(obj->sections[i].data, shdr.sh_size, 1, f) != 1) {
                fclose(f);
                return -1;
            }
            fseek(f, saved_pos, SEEK_SET);
        } else {
            obj->sections[i].data = NULL;
        }
    }
    fclose(f);
    return 0;
}
```
This gives you an `ObjectFile` with all its sections loaded. But you'll quickly notice: **most sections aren't relevant for merging**.
### Filtering to Allocatable Sections
Object files contain metadata sections (`.symtab`, `.strtab`, `.rela.text`, `.debug_*`) that the linker uses but which don't become part of the final executable's memory image. The key filter is the `SHF_ALLOC` flag:
```c
int is_allocatable_section(const InputSection *sec) {
    // Only merge sections that will be loaded into memory
    return (sec->flags & SHF_ALLOC) != 0;
}
int is_mergeable_section(const InputSection *sec) {
    // Skip null section and non-allocatable sections
    if (sec->type == SHT_NULL) return 0;
    if (!is_allocatable_section(sec)) return 0;
    return 1;
}
```
The allocatable sections you'll typically see:
- `.text` — code (SHF_ALLOC | SHF_EXECINSTR)
- `.data` — initialized writable data (SHF_ALLOC | SHF_WRITE)
- `.rodata` — read-only data (SHF_ALLOC)
- `.bss` — uninitialized data (SHF_ALLOC | SHF_WRITE, but SHT_NOBITS)

![ELF Object File Parsing Data Flow](./diagrams/tdd-diag-002.svg)

![Section Flags and Consistency](./diagrams/diag-section-flags.svg)

## The Merge Strategy: Group, Pad, Concatenate
Now you have multiple object files, each with its own set of sections. The merge strategy:
1. **Group** sections by name across all files (all `.text` together, all `.data` together, etc.)
2. **Order** the groups (typically: `.text`, `.rodata`, `.data`, `.bss`)
3. **Concatenate** sections within each group, inserting **padding** to satisfy alignment
4. **Track** where every input byte ends up in the output
### Why Alignment Matters
Alignment isn't just a nice-to-have — it's a correctness requirement. Consider:
```c
// In utils.o
const char* error_messages[] = { "Error 1", "Error 2" };  // .rodata
// In main.o  
typedef struct { double x, y; } Point;  // needs 8-byte alignment
Point origin = { 0.0, 0.0 };  // .data
```
If `.data` starts at a misaligned address, accessing `origin.x` will crash on some architectures and be slow on others. The x86-64 ABI specifies:
- 16-byte alignment for SSE/AVX data
- 8-byte alignment for `double` and pointers
- 4-byte alignment for `int` and `float`
- Section alignment should be at least the maximum alignment of any contained data

![.bss Special Handling Data Flow](./diagrams/tdd-diag-011.svg)

![Alignment Padding Visualization](./diagrams/diag-alignment-padding.svg)

### Alignment Math: The Padding Formula
When concatenating sections, you need to calculate padding between them:
```c
// Calculate padding needed to align 'current_offset' to 'alignment'
uint64_t calc_padding(uint64_t current_offset, uint64_t alignment) {
    if (alignment == 0 || alignment == 1) return 0;
    uint64_t misalignment = current_offset % alignment;
    if (misalignment == 0) return 0;
    return alignment - misalignment;
}
// Align current_offset up to the next alignment boundary
uint64_t align_up(uint64_t current_offset, uint64_t alignment) {
    if (alignment == 0 || alignment == 1) return current_offset;
    return (current_offset + alignment - 1) & ~(alignment - 1);
}
```
The `align_up` function uses a bit trick: for power-of-two alignments, `~(alignment - 1)` creates a mask that clears the low bits. `(value + alignment - 1) & mask` rounds up.
**Example:**
```
Current offset: 0x1046
Required alignment: 16 (0x10)
calc_padding(0x1046, 16):
  misalignment = 0x1046 % 16 = 6
  padding = 16 - 6 = 10 (0xA)
align_up(0x1046, 16):
  = (0x1046 + 0xF) & ~0xF
  = 0x1055 & 0xFFF...FF0
  = 0x1050
```
### Building the Output Section Table
You need data structures that represent the merged output:
```c
// Tracks a single input section's placement in the output
typedef struct {
    char source_file[256];     // Which object file
    char section_name[64];     // Section name (.text, .data, etc.)
    uint64_t input_offset;     // Offset within input section (usually 0)
    uint64_t input_size;       // Size in input file
    uint64_t output_offset;    // Where it ends up in merged output
    uint64_t padding_before;   // Padding bytes added before this section
} SectionMapping;
// An output section (merged from multiple inputs)
typedef struct {
    char name[64];             // .text, .data, .rodata, .bss
    uint64_t flags;            // Merged flags
    uint64_t align;            // Maximum alignment of all inputs
    uint64_t file_size;        // Bytes in file (0 for .bss)
    uint64_t mem_size;         // Bytes in memory (includes .bss)
    uint64_t file_offset;      // Where in output file
    uint64_t virtual_addr;     // Assigned later during final layout
    SectionMapping *mappings;  // Array of input sections
    size_t mapping_count;
    size_t mapping_capacity;
    uint8_t *data;             // Merged content (NULL for .bss)
} OutputSection;
// The complete linker state for section merging
typedef struct {
    ObjectFile *inputs;        // Array of input object files
    size_t input_count;
    OutputSection *outputs;    // Array of output sections
    size_t output_count;
    size_t output_capacity;
    // Hash table for quick output section lookup by name
    OutputSection *section_by_name[64];  // Simple hash table
} LinkerContext;
```
### The Merge Algorithm
Here's the core logic:
```c
#define MAX_OUTPUT_SECTIONS 32
int init_linker_context(LinkerContext *ctx) {
    memset(ctx, 0, sizeof(LinkerContext));
    ctx->output_capacity = MAX_OUTPUT_SECTIONS;
    ctx->outputs = calloc(MAX_OUTPUT_SECTIONS, sizeof(OutputSection));
    return 0;
}
// Find or create an output section by name
OutputSection* get_or_create_output_section(LinkerContext *ctx, 
                                             const char *name,
                                             uint64_t flags,
                                             uint64_t align) {
    // Simple hash for lookup
    uint32_t hash = 0;
    for (const char *p = name; *p; p++) {
        hash = hash * 31 + *p;
    }
    hash %= 64;
    // Check if already exists
    for (size_t i = 0; i < ctx->output_count; i++) {
        if (strcmp(ctx->outputs[i].name, name) == 0) {
            // Update alignment to maximum
            if (align > ctx->outputs[i].align) {
                ctx->outputs[i].align = align;
            }
            return &ctx->outputs[i];
        }
    }
    // Create new output section
    if (ctx->output_count >= ctx->output_capacity) {
        return NULL;  // Too many sections
    }
    OutputSection *out = &ctx->outputs[ctx->output_count++];
    strncpy(out->name, name, sizeof(out->name) - 1);
    out->flags = flags;
    out->align = align;
    out->mapping_capacity = 16;
    out->mappings = calloc(out->mapping_capacity, sizeof(SectionMapping));
    return out;
}
// Add an input section to its corresponding output section
int add_section_to_output(LinkerContext *ctx, 
                          ObjectFile *input_file,
                          InputSection *input_sec) {
    OutputSection *out = get_or_create_output_section(
        ctx, 
        input_sec->name,
        input_sec->flags,
        input_sec->align
    );
    if (!out) {
        fprintf(stderr, "Failed to create output section for %s\n", 
                input_sec->name);
        return -1;
    }
    // Validate flag consistency (optional: warn or error on mismatch)
    // Allocatable sections should have consistent flags
    // Some linkers allow merging .rodata into .text (both read-only)
    // Calculate where this input section will go
    uint64_t current_size = out->mem_size;  // End of existing content
    uint64_t padding = calc_padding(current_size, input_sec->align);
    uint64_t new_offset = current_size + padding;
    // Grow mappings array if needed
    if (out->mapping_count >= out->mapping_capacity) {
        out->mapping_capacity *= 2;
        out->mappings = realloc(out->mappings, 
                                out->mapping_capacity * sizeof(SectionMapping));
    }
    // Record the mapping
    SectionMapping *map = &out->mappings[out->mapping_count++];
    strncpy(map->source_file, input_file->filename, sizeof(map->source_file) - 1);
    strncpy(map->section_name, input_sec->name, sizeof(map->section_name) - 1);
    map->input_offset = 0;
    map->input_size = input_sec->size;
    map->output_offset = new_offset;
    map->padding_before = padding;
    // Update output section size
    out->mem_size = new_offset + input_sec->size;
    if (input_sec->type != SHT_NOBITS) {
        out->file_size = new_offset + input_sec->size;
    }
    // Note: .bss contributes to mem_size but not file_size
    return 0;
}
```
### Processing All Input Files
```c
int merge_all_sections(LinkerContext *ctx) {
    // First pass: collect all sections into output groups
    for (size_t i = 0; i < ctx->input_count; i++) {
        ObjectFile *obj = &ctx->inputs[i];
        for (int j = 0; j < obj->section_count; j++) {
            InputSection *sec = &obj->sections[j];
            if (!is_mergeable_section(sec)) {
                continue;  // Skip non-allocatable sections
            }
            if (add_section_to_output(ctx, obj, sec) != 0) {
                return -1;
            }
        }
    }
    return 0;
}
```

![Alignment Padding Calculation Algorithm](./diagrams/tdd-diag-006.svg)

![Section Merging Data Flow](./diagrams/diag-section-merge-flow.svg)

![Section Flag Validation Matrix](./diagrams/tdd-diag-013.svg)

## The Input-to-Output Mapping Table
This is the most important data structure you're building in this milestone. The mapping table answers the question: "I have an offset in an input section; where is it in the output?"
Why does this matter? Because **symbol resolution** needs it:
```
Symbol 'helper' in utils.o:
  - In section .text at offset 0x20
  - .text from utils.o was placed at output offset 0x1050
  - Therefore 'helper' is at output offset 0x1070
```

![Mapping Table Hash Structure](./diagrams/tdd-diag-012.svg)

![Input-to-Output Offset Mapping Table](./diagrams/diag-input-output-mapping.svg)

Let's build a lookup function:
```c
// Find the output offset for an input section
int lookup_output_offset(LinkerContext *ctx,
                         const char *filename,
                         const char *section_name,
                         uint64_t input_offset,
                         uint64_t *output_offset_out) {
    // Find the output section
    for (size_t i = 0; i < ctx->output_count; i++) {
        OutputSection *out = &ctx->outputs[i];
        if (strcmp(out->name, section_name) != 0) continue;
        // Search for the specific input file's mapping
        for (size_t j = 0; j < out->mapping_count; j++) {
            SectionMapping *map = &out->mappings[j];
            if (strcmp(map->source_file, filename) == 0) {
                // Found it! Calculate output offset
                if (input_offset >= map->input_size) {
                    fprintf(stderr, "Input offset %lx exceeds section size %lx\n",
                            input_offset, map->input_size);
                    return -1;
                }
                *output_offset_out = map->output_offset + input_offset;
                return 0;
            }
        }
    }
    fprintf(stderr, "No mapping found for %s:%s\n", filename, section_name);
    return -1;
}
```
### Optimizing the Mapping Lookup
For a small linker, the linear search above is fine. But real linkers process thousands of input files with millions of sections. Common optimizations:
1. **Hash by (file, section)**: O(1) lookup instead of O(n)
2. **Sort + binary search**: O(log n) with better cache locality
3. **Range trees**: For finding which input section contains a given output offset (needed for debugging)
For this project, let's add a simple hash table:
```c
typedef struct {
    char key[320];  // "filename:section_name"
    SectionMapping *mapping;
    OutputSection *output;
} MappingEntry;
typedef struct {
    MappingEntry *entries;
    size_t size;
    size_t capacity;
} MappingHash;
uint32_t hash_key(const char *key) {
    uint32_t h = 5381;
    for (; *key; key++) {
        h = h * 33 + *key;
    }
    return h;
}
int build_mapping_hash(LinkerContext *ctx, MappingHash *hash) {
    // Count total mappings
    size_t total = 0;
    for (size_t i = 0; i < ctx->output_count; i++) {
        total += ctx->outputs[i].mapping_count;
    }
    hash->capacity = total * 2 + 1;  // Load factor < 0.5
    hash->entries = calloc(hash->capacity, sizeof(MappingEntry));
    hash->size = 0;
    // Populate
    for (size_t i = 0; i < ctx->output_count; i++) {
        OutputSection *out = &ctx->outputs[i];
        for (size_t j = 0; j < out->mapping_count; j++) {
            SectionMapping *map = &out->mappings[j];
            char key[320];
            snprintf(key, sizeof(key), "%s:%s", 
                     map->source_file, map->section_name);
            uint32_t idx = hash_key(key) % hash->capacity;
            // Linear probing
            while (hash->entries[idx].key[0] != '\0') {
                idx = (idx + 1) % hash->capacity;
            }
            strncpy(hash->entries[idx].key, key, sizeof(hash->entries[idx].key) - 1);
            hash->entries[idx].mapping = map;
            hash->entries[idx].output = out;
            hash->size++;
        }
    }
    return 0;
}
```
## The .bss Special Case
`.bss` is weird. It's an **allocatable section that has no file content**. This exists because:
- Uninitialized globals don't need to be stored in the file (waste of disk)
- But they need memory at runtime (the loader allocates it)
- The section exists to reserve address space, not file space

![Multi-File Section Grouping Sequence](./diagrams/tdd-diag-008.svg)

![.bss Section Handling: File vs Memory](./diagrams/diag-bss-handling.svg)

```c
// Example: handling .bss differently
int merge_section_data(OutputSection *out) {
    // Allocate buffer for merged data
    out->data = malloc(out->file_size);
    if (!out->data) return -1;
    memset(out->data, 0, out->file_size);  // Initialize to zero
    // Copy each input section's data
    for (size_t i = 0; i < out->mapping_count; i++) {
        SectionMapping *map = &out->mappings[i];
        // Find input section data
        // (In real code, you'd keep a pointer to the InputSection)
        // For .bss (SHT_NOBITS), there's no data to copy
        // The zeroed buffer already represents .bss content
        // Copy non-.bss data
        if (map->has_data) {
            memcpy(out->data + map->output_offset, 
                   map->input_data, 
                   map->input_size);
        }
    }
    return 0;
}
```
### Memory vs File Size Distinction
This is where many bugs creep in. Your output section needs two size fields:
```c
typedef struct {
    // ... other fields ...
    uint64_t file_size;   // Bytes to write to output file
    uint64_t mem_size;    // Bytes the loader allocates in memory
} OutputSection;
```
For `.text`, `.data`, `.rodata`: `file_size == mem_size`
For `.bss`: `file_size == 0`, `mem_size > 0`
When you have mixed content (e.g., `.data` followed by `.bss`), you need to handle this carefully. In practice, linkers keep `.bss` separate because:
1. It's typically placed last (after all file-backed sections)
2. The loader handles it differently (allocates zeroed pages)
3. It shouldn't pad file size unnecessarily
## Section Ordering and Virtual Address Assignment
After merging, you need to decide the **order** of output sections and assign them **virtual addresses**. The typical layout:
```
Virtual Address    Section     File Offset
0x400000           .text       0x1000
0x401000           .rodata     0x2000
0x402000           .data       0x3000
0x403000           .bss        (no file offset)
```
The ordering matters for:
1. **Cache behavior**: Code (.text) should be together
2. **Memory protection**: Read-only (.text, .rodata) separate from writable (.data, .bss)
3. **Page alignment**: Segments need page-aligned starts (4096 bytes typically)
```c
int assign_virtual_addresses(LinkerContext *ctx, uint64_t base_address) {
    // Standard ordering
    const char *order[] = {".text", ".rodata", ".data", ".bss"};
    int order_count = 4;
    uint64_t current_addr = base_address;
    uint64_t file_offset = 0x1000;  // Skip ELF header space
    for (int i = 0; i < order_count; i++) {
        const char *name = order[i];
        // Find output section
        OutputSection *out = NULL;
        for (size_t j = 0; j < ctx->output_count; j++) {
            if (strcmp(ctx->outputs[j].name, name) == 0) {
                out = &ctx->outputs[j];
                break;
            }
        }
        if (!out || out->mem_size == 0) continue;  // Section not present
        // Align virtual address
        current_addr = align_up(current_addr, out->align);
        // For non-.bss, also align file offset
        if (out->file_size > 0) {
            file_offset = align_up(file_offset, out->align);
            out->file_offset = file_offset;
        }
        out->virtual_addr = current_addr;
        current_addr += out->mem_size;
        file_offset += out->file_size;
    }
    return 0;
}
```
## A Complete Example: Linking Two Files
Let's trace through what happens when linking `main.o` and `utils.o`:
```
main.o:
  .text: 120 bytes, align 16
  .data: 32 bytes, align 8
utils.o:
  .text: 80 bytes, align 16
  .rodata: 64 bytes, align 8
  .bss: 16 bytes, align 8
```
**Step 1: Collect into output sections**
```
Output .text:
  - main.o/.text: 120 bytes, align 16
  - utils.o/.text: 80 bytes, align 16
Output .data:
  - main.o/.data: 32 bytes, align 8
Output .rodata:
  - utils.o/.rodata: 64 bytes, align 8
Output .bss:
  - utils.o/.bss: 16 bytes, align 8
```
**Step 2: Calculate placements**
```
Output .text (max align: 16):
  - main.o/.text: offset 0x0, size 0x78 (120)
  - utils.o/.text: offset 0x80, size 0x50 (80)
    (0x78 aligned to 16 = 0x80, padding = 8)
  Total mem_size: 0xD0 (208), file_size: 0xD0
Output .rodata (max align: 8):
  - utils.o/.rodata: offset 0x0, size 0x40 (64)
  Total mem_size: 0x40, file_size: 0x40
Output .data (max align: 8):
  - main.o/.data: offset 0x0, size 0x20 (32)
  Total mem_size: 0x20, file_size: 0x20
Output .bss (max align: 8):
  - utils.o/.bss: offset 0x0, size 0x10 (16)
  Total mem_size: 0x10, file_size: 0x0
```
**Step 3: Assign virtual addresses (base 0x400000)**
```
.text:  vaddr 0x400000, file_offset 0x1000, size 0xD0
.rodata: vaddr 0x4000D0, file_offset 0x10D0, size 0x40
.data:  vaddr 0x400120, file_offset 0x1120, size 0x20
.bss:   vaddr 0x400140, file_offset (none), size 0x10
```
**Step 4: Build mapping table**
```
(main.o, .text, 0x0) -> output .text at 0x0
(main.o, .text, 0x78) -> output .text at 0x78
(utils.o, .text, 0x0) -> output .text at 0x80
(utils.o, .text, 0x20) -> output .text at 0xA0
(main.o, .data, 0x0) -> output .data at 0x0
(utils.o, .rodata, 0x0) -> output .rodata at 0x0
(utils.o, .bss, 0x0) -> output .bss at 0x0
```

![InputSection Struct Memory Layout](./diagrams/tdd-diag-009.svg)

![Trace Example: Linking Two Files](./diagrams/diag-trace-simple-link.svg)

Now, when symbol resolution needs to find `main.o`'s function at `.text+0x10`, it looks up `(main.o, .text, 0x10)` and gets output offset `0x10`. When it needs `utils.o`'s function at `.text+0x20`, it gets `0xA0` (0x80 + 0x20).
## Putting It All Together
Here's the complete flow for section merging:
```c
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s file1.o file2.o ...\n", argv[0]);
        return 1;
    }
    LinkerContext ctx;
    init_linker_context(&ctx);
    // Parse all input files
    ctx.input_count = argc - 1;
    ctx.inputs = calloc(ctx.input_count, sizeof(ObjectFile));
    for (int i = 1; i < argc; i++) {
        printf("Parsing %s...\n", argv[i]);
        if (parse_object_file(argv[i], &ctx.inputs[i-1]) != 0) {
            fprintf(stderr, "Failed to parse %s\n", argv[i]);
            return 1;
        }
    }
    // Merge sections
    printf("Merging sections...\n");
    if (merge_all_sections(&ctx) != 0) {
        fprintf(stderr, "Section merge failed\n");
        return 1;
    }
    // Print merged section table
    printf("\nMerged sections:\n");
    for (size_t i = 0; i < ctx.output_count; i++) {
        OutputSection *out = &ctx.outputs[i];
        printf("  %s: mem_size=%lx, file_size=%lx, align=%ld, %ld mappings\n",
               out->name, out->mem_size, out->file_size, 
               out->align, out->mapping_count);
        for (size_t j = 0; j < out->mapping_count; j++) {
            SectionMapping *map = &out->mappings[j];
            printf("    %s: input_size=%lx -> output_offset=%lx (padding=%ld)\n",
                   map->source_file, map->input_size, 
                   map->output_offset, map->padding_before);
        }
    }
    // Assign virtual addresses
    assign_virtual_addresses(&ctx, 0x400000);
    printf("\nWith virtual addresses:\n");
    for (size_t i = 0; i < ctx.output_count; i++) {
        OutputSection *out = &ctx.outputs[i];
        printf("  %s: vaddr=%lx, file_offset=%lx\n",
               out->name, out->virtual_addr, out->file_offset);
    }
    return 0;
}
```
## Common Pitfalls and Debugging
### Pitfall 1: Forgetting Alignment Is a Power of 2
The alignment field might be 0, 1, or any power of 2. Never assume it's 16:
```c
// BAD: assumes power of 2
uint64_t aligned = (offset + 15) & ~15;
// GOOD: handles any alignment
uint64_t aligned = align_up(offset, section->align);
```
### Pitfall 2: Confusing File Offset and Virtual Address
These are different! A section at virtual address 0x401000 might be at file offset 0x1000 (after ELF header and program headers). The loader maps file contents to virtual addresses:
```c
// When writing relocation results:
// BAD: storing virtual address in file
*(uint32_t*)(data + file_offset) = symbol_virtual_addr;
// GOOD: this IS correct for the final executable
// (virtual addresses are what we want in the output)
*(uint32_t*)(data + file_offset) = symbol_virtual_addr;
```
### Pitfall 3: .bss in the Middle of File-Backed Sections
If you accidentally interleave `.bss` with `.data`, your file offsets will be wrong:
```
WRONG:
  .data: file_offset 0x1000, size 0x100
  .bss:  file_offset 0x1100, size 0x100 (but no file content!)
  .data2: file_offset 0x1200, size 0x100  <- gap in file!
CORRECT:
  .data: file_offset 0x1000, size 0x100
  .data2: file_offset 0x1100, size 0x100
  .bss:  (no file offset), mem_size 0x100
```
### Pitfall 4: Section Flag Mismatches
If `main.o`'s `.text` has `SHF_WRITE` (buggy compiler output?) and `utils.o`'s doesn't, what do you do? Options:
1. **Error**: Reject the input (safest)
2. **Warn + merge with union of flags**: Most permissive
3. **Trust first input**: What GNU ld does in some cases
For this project, option 2 is reasonable:
```c
// Merge flags: union of all input flags
out->flags |= input_sec->flags;
```
## Testing Your Section Merger
You need test inputs. Create simple assembly files and assemble them:
```bash
# Create test1.s
cat > test1.s << 'EOF'
.section .text
.globl func1
func1:
    ret
.section .data
.globl data1
data1:
    .quad 42
EOF
# Create test2.s  
cat > test2.s << 'EOF'
.section .text
.globl func2
func2:
    nop
    nop
    ret
.section .rodata
message:
    .ascii "Hello\0"
.section .bss
.globl buffer
buffer:
    .skip 64
EOF
# Assemble
as -o test1.o test1.s
as -o test2.o test2.s
# Run your linker
./mylinker test1.o test2.o
```
Expected output should show:
- `.text`: merged from both files
- `.data`: only from test1.o
- `.rodata`: only from test2.o
- `.bss`: only from test2.o (with file_size=0, mem_size=64)
## What's Next?
You've built the foundation: a system that reads object files, groups their sections, calculates placements with proper alignment, and builds a mapping table. But you have sections with **no symbols yet** and **no relocation fixups**.
In the next milestone, **Symbol Resolution**, you'll:
1. Extract symbols from each input's `.symtab`
2. Build a global symbol table
3. Detect undefined and duplicate symbols
4. Assign final addresses to every symbol using the mapping table you just built
The mapping table is the bridge — symbol resolution will use it constantly. When a symbol in `utils.o` is at `.text+0x20`, and you know `utils.o/.text` maps to output offset `0x80`, you can compute: this symbol lives at output address `0xA0`.
---
## Knowledge Cascade
**Position-Independent Code (PIC)** — Now that you see how object files are designed to be moved and rearranged, compiler flags like `-fPIC` make sense: they generate code that works at any address, using relative references instead of absolute ones. The linker's job is easier with PIC because fewer relocations need runtime fixups.
**Memory Layout (cross-domain, OS)** — The section ordering and alignment decisions you make here directly determine what the OS loader sees. When `exec()` loads your binary, it creates virtual memory regions mapped to your PT_LOAD segments. The permissions (read/write/execute) come from section flags. A linker bug becomes a loader crash becomes a confusing `SIGSEGV`.
**Linker Scripts** — What you've built is the *default* merging behavior. Real linkers like GNU ld accept linker scripts (`*.ld`) that specify custom section ordering, address assignments, and symbol aliases. Embedded systems use this heavily to place code at specific hardware addresses. Security-hardened builds use it to randomize section order (an early form of ASLR).
**Binary Size Optimization** — Understanding section merging enables `-ffunction-sections` and `-ffunction-sections` compiler flags. These put each function/variable in its own section, letting the linker garbage-collect unused ones (`.text.foo` gets merged only if `foo` is referenced). This is how you strip dead code at link time.
**Debug Information (DWARF)** — Debug sections (`.debug_*`) merge the same way as code sections. The DWARF info contains references to code addresses; when sections move, these references need updating too. This is why `objcopy --strip-debug` works — you can just delete those sections without affecting execution.
[[CRITERIA_JSON: {"milestone_id": "build-linker-m1", "criteria": ["Parse multiple ELF64 relocatable object files (.o) and extract section headers with name, type, flags, size, alignment, and content", "Filter sections to identify allocatable sections (SHF_ALLOC) for merging: .text, .data, .rodata, .bss", "Group sections by name across all input files, maintaining separate output sections for each unique section name", "Calculate and apply alignment padding between consecutive sections within each output group using power-of-2 alignment math", "Build input-to-output mapping table recording each input section's output offset for later relocation processing", "Handle .bss sections (SHT_NOBITS) correctly: track memory size separately from file size, allocate no file space but reserve virtual address space", "Assign output section sizes correctly: file_size includes only non-.bss content, mem_size includes all content including .bss", "Detect and validate section flag consistency across merged sections (SHF_WRITE, SHF_EXECINSTR)", "Assign preliminary virtual addresses to output sections in standard order (.text, .rodata, .data, .bss) with proper alignment", "Support lookup queries: given (input_file, section_name, input_offset) return the corresponding output section offset"]}]
<!-- END_MS -->


<!-- MS_ID: build-linker-m2 -->
# Symbol Resolution: The Global Address Book
You've merged sections. You know where every byte from every input file will live in the final executable. But there's a problem: your sections are anonymous. You have 208 bytes of `.text` but no idea which bytes belong to `main()`, which belong to `helper()`, and which belong to `compute()`.
More critically, `main.o` contains a `call` instruction that references `helper`. That instruction has a hole — a relocation — saying "put helper's address here." But what *is* helper's address? You can't answer that until you've found `helper` in some object file's symbol table and assigned it a final location.
This milestone is about building the linker's **global address book**: a unified symbol table that resolves every named entity across all translation units, enforces the rules of symbol priority, and produces final addresses for every function and variable your program uses.
## The Tension: Names Must Become Numbers
Here's what you probably think happens: you call `helper()`, the compiler outputs `call helper`, the linker finds `helper`, done. But that model hides the real complexity:
- `main.c` calls `helper()`. `helper.c` defines `helper()`. But what if `lib.c` *also* defines `helper()`? Which one wins?
- `utils.c` declares `int buffer[100];` without initializing it. `main.c` does the same. Is this one buffer or two? Or an error?
- `internal.c` has a `static void helper()` function. `other.c` also has `static void helper()`. These are different functions with the same name — how?
- `main.c` references `printf`, but no input file defines it. Is this an error? (Hint: maybe not — it might come from a library)

![Weak Undefined Symbol Resolution](./diagrams/tdd-diag-027.svg)


The fundamental tension: **source code uses human-readable names, but executables need numeric addresses**. The linker must translate every name to a number, and it must do so unambiguously even when:
- Multiple files define the same name
- Some names are meant to be private (file-local)
- Some definitions are "fallbacks" that stronger definitions override
- Some names reference things not in any input file
The output of this milestone is a **resolved global symbol table** where every symbol has a final virtual address. This table is the key that unlocks relocation processing — without it, you can't patch any address holes.
## What Is a Symbol, Really?
A symbol is a named location in an object file. It's the linker's representation of "something with a name that might be referenced from elsewhere."
> **🔑 Foundation: Symbol table entries (st_name, st_value, st_size, st_info, st_shndx)**
> 
> ## What It Is
> Every named entity in an ELF object file — functions, global variables, section labels, even the file itself — gets an entry in the symbol table (`.symtab` section). Each entry describes one symbol with a compact binary structure:
> 
> ```c
> typedef struct {
>     uint32_t st_name;   // Index into string table (.strtab) for symbol name
>     uint8_t  st_info;   // Type (4 bits) + Binding (4 bits)
>     uint8_t  st_other;  // Visibility (usually 0)
>     uint16_t st_shndx;  // Which section this symbol is in (or special value)
>     uint64_t st_value;  // Value: offset within section, or address
>     uint64_t st_size;   // Size in bytes (0 if unknown)
> } Elf64_Sym;
> ```
> 
> **st_name**: Not the name itself, but an index into the string table. The actual string "helper\0" lives in `.strtab`, and `st_name` tells you where to find it.
> 
> **st_info**: A packed byte encoding two properties:
> - **Binding** (high 4 bits): `STB_LOCAL` (0), `STB_GLOBAL` (1), `STB_WEAK` (2)
> - **Type** (low 4 bits): `STT_NOTYPE` (0), `STT_OBJECT` (1, variable), `STT_FUNC` (2), `STT_SECTION` (3), `STT_FILE` (4)
> 
> **st_shndx**: Section header index where this symbol lives. Special values:
> - `SHN_UNDEF` (0): Symbol is referenced but not defined in this file
> - `SHN_ABS` (0xFFF1): Absolute value, not in any section
> - `SHN_COMMON` (0xFFF2): COMMON symbol (uninitialized, needs allocation)
> 
> **st_value**: For defined symbols in object files, this is the offset within the section (`st_shndx`). For undefined symbols, it's 0. After linking, this becomes the final virtual address.
> 
> ## Why You Need This Right Now
> Symbol resolution is literally impossible without understanding this structure. You need to:
> - Iterate through `.symtab` entries in each input file
> - Extract names using `st_name` + `.strtab`
> - Determine visibility from `st_info` binding
> - Find where the symbol lives using `st_shndx`
> - Calculate final address using `st_value` + section's output offset
> 
> ## Key Mental Model
> **Think of the symbol table as a two-column spreadsheet: Name → Location.**
> 
> Each row (symbol entry) answers: "Where can I find the thing named X?"
> - If `st_shndx == SHN_UNDEF`, the answer is "I don't know — ask another file"
> - If `st_shndx` is a valid section, the answer is "in section Y at offset Z"
> - The binding field says whether other files can see this answer (LOCAL vs GLOBAL)

![Symbol Resolution Sequence Diagram](./diagrams/tdd-diag-025.svg)

![ELF Symbol Table Entry Structure](./diagrams/diag-symbol-table-entry.svg)

### Symbol Types You'll Encounter
Let's categorize what you'll find in a typical `.symtab`:
| Type | Binding | st_shndx | Meaning | Example |
|------|---------|----------|---------|---------|
| Defined global | GLOBAL | Section index | Available to all files | `void helper() {}` |
| Defined local | LOCAL | Section index | Only in this file | `static int count;` |
| Undefined | GLOBAL | SHN_UNDEF | Need definition elsewhere | `extern int errno;` |
| Weak defined | WEAK | Section index | Fallback if no strong | `__attribute__((weak)) void hook() {}` |
| Weak undefined | WEAK | SHN_UNDEF | Optional reference | `extern __attribute__((weak)) int opt;` |
| COMMON | GLOBAL | SHN_COMMON | Uninitialized global | `int buffer[100];` (tentative) |
| Section symbol | LOCAL | Section index | Section's own label | Used in relocations |
| File symbol | LOCAL | SHN_ABS | Source file name | Always first, for debugging |

![Undefined Symbol Error Attribution](./diagrams/tdd-diag-026.svg)

![Symbol Binding and Visibility Matrix](./diagrams/diag-symbol-binding-visibility.svg)

The **COMMON** row deserves attention. In C, a global variable declared without `extern` and without an initializer is a "tentative definition":
```c
// In a header included by multiple .c files
int global_counter;  // Tentative definition — is this a definition or declaration?
```
The C standard allows this to appear in multiple translation units. The linker must merge them into a single variable. This is why COMMON symbols exist — they represent "I might define this, but if someone else does, use theirs." The rule: **largest size wins**.
## Reading the Symbol Table
You need to parse `.symtab` from each input file. The symbol table section has a `sh_link` field pointing to its associated string table (`.strtab`):
```c
typedef struct {
    char name[64];            // Symbol name (from .strtab)
    uint8_t type;             // STT_*
    uint8_t binding;          // STB_*
    uint8_t visibility;       // STV_*
    uint16_t section_idx;     // st_shndx: section index or SHN_* special
    uint64_t value;           // Offset within section (or 0 for undefined)
    uint64_t size;            // Symbol size in bytes
    int source_file_idx;      // Which input file this came from
} Symbol;
```
```c
#define STB_LOCAL  0
#define STB_GLOBAL 1
#define STB_WEAK   2
#define STT_NOTYPE  0
#define STT_OBJECT  1
#define STT_FUNC    2
#define STT_SECTION 3
#define STT_FILE    4
#define SHN_UNDEF   0
#define SHN_ABS     0xFFF1
#define SHN_COMMON  0xFFF2
// Extract binding from st_info
#define ELF64_ST_BIND(i)   ((i) >> 4)
// Extract type from st_info
#define ELF64_ST_TYPE(i)   ((i) & 0xf)
```
Now let's read symbols from an object file:
```c
int parse_symbols(ObjectFile *obj) {
    // Find .symtab section
    int symtab_idx = -1;
    int strtab_idx = -1;
    for (int i = 0; i < obj->section_count; i++) {
        if (strcmp(obj->sections[i].name, ".symtab") == 0) {
            symtab_idx = i;
        }
        if (strcmp(obj->sections[i].name, ".strtab") == 0) {
            strtab_idx = i;
        }
    }
    if (symtab_idx < 0) {
        // No symbols (unlikely but possible for stripped objects)
        return 0;
    }
    InputSection *symtab_sec = &obj->sections[symtab_idx];
    char *strtab = (strtab_idx >= 0) ? (char*)obj->sections[strtab_idx].data : NULL;
    // Calculate number of symbols
    size_t sym_count = symtab_sec->size / sizeof(Elf64_Sym);
    Elf64_Sym *syms = (Elf64_Sym*)symtab_sec->data;
    // Allocate symbol array in object file
    obj->symbols = calloc(sym_count, sizeof(Symbol));
    obj->symbol_count = sym_count;
    for (size_t i = 0; i < sym_count; i++) {
        Elf64_Sym *esym = &syms[i];
        Symbol *sym = &obj->symbols[i];
        // Get name from string table
        if (strtab && esym->st_name < obj->sections[strtab_idx].size) {
            strncpy(sym->name, strtab + esym->st_name, sizeof(sym->name) - 1);
        }
        sym->type = ELF64_ST_TYPE(esym->st_info);
        sym->binding = ELF64_ST_BIND(esym->st_info);
        sym->visibility = esym->st_other;  // Usually 0
        sym->section_idx = esym->st_shndx;
        sym->value = esym->st_value;
        sym->size = esym->st_size;
    }
    return 0;
}
```
### Symbol Table Ordering: A Crucial Detail
ELF symbol tables have a specific ordering requirement:
1. **First entry (index 0)**: All zeros — the "null symbol"
2. **Local symbols**: All `STB_LOCAL` symbols, grouped together
3. **Global and weak symbols**: `STB_GLOBAL` and `STB_WEAK` symbols
The ELF header's `e_shndx` for `.symtab` has `sh_info` pointing to the index of the **first non-local symbol**. This lets you quickly iterate only global symbols when needed.
```c
// The section header's sh_info field tells us where locals end
// This is stored in the section's sh_info field for .symtab
```
This ordering matters because local symbols are resolved differently from global ones — they're invisible outside their translation unit.
## The Global Symbol Table
Now you need a data structure that aggregates symbols from all input files and resolves duplicates. This is the linker's internal symbol table:
```c
typedef enum {
    SYM_UNDEF,       // Referenced but not yet defined
    SYM_DEFINED,     // Has a definition (strong or weak)
    SYM_COMMON,      // COMMON symbol (needs allocation)
    SYM_RESOLVED     // Final address assigned
} SymbolState;
typedef struct {
    char name[64];            // Symbol name
    uint8_t type;             // STT_FUNC, STT_OBJECT, etc.
    uint8_t binding;          // Final binding after resolution
    SymbolState state;        // Current resolution state
    // For defined symbols:
    int source_file_idx;      // Which input file provides the definition
    int source_sym_idx;       // Index in that file's symbol table
    uint16_t section_idx;     // Which output section
    uint64_t section_offset;  // Offset within output section
    uint64_t final_address;   // Computed after address assignment
    // For COMMON symbols:
    uint64_t common_size;     // Size (largest wins)
    uint64_t common_align;    // Alignment requirement
    // Resolution tracking:
    int is_strong;            // 1 if strong definition, 0 if weak
    int ref_count;            // How many files reference this
} GlobalSymbol;
typedef struct {
    GlobalSymbol *symbols;
    size_t count;
    size_t capacity;
    // Hash table for O(1) lookup by name
    GlobalSymbol **hash_table;
    size_t hash_size;
} GlobalSymbolTable;
```
### Hash Table for Fast Lookup
Symbol resolution does a lot of name lookups. With hundreds of thousands of symbols in large programs, linear search is unacceptable. A hash table gives O(1) average lookup:
```c
#define HASH_PRIME 31
uint32_t hash_string(const char *s) {
    uint32_t h = 0;
    while (*s) {
        h = h * HASH_PRIME + (unsigned char)*s++;
    }
    return h;
}
int init_global_symbol_table(GlobalSymbolTable *tbl, size_t initial_capacity) {
    tbl->capacity = initial_capacity;
    tbl->count = 0;
    tbl->symbols = calloc(initial_capacity, sizeof(GlobalSymbol));
    tbl->hash_size = initial_capacity * 2;  // Load factor < 0.5
    tbl->hash_table = calloc(tbl->hash_size, sizeof(GlobalSymbol*));
    if (!tbl->symbols || !tbl->hash_table) {
        return -1;
    }
    return 0;
}
GlobalSymbol* lookup_symbol(GlobalSymbolTable *tbl, const char *name) {
    uint32_t h = hash_string(name) % tbl->hash_size;
    // Linear probing
    while (tbl->hash_table[h] != NULL) {
        if (strcmp(tbl->hash_table[h]->name, name) == 0) {
            return tbl->hash_table[h];
        }
        h = (h + 1) % tbl->hash_size;
    }
    return NULL;
}
GlobalSymbol* insert_symbol(GlobalSymbolTable *tbl, const char *name) {
    // Check if already exists
    GlobalSymbol *existing = lookup_symbol(tbl, name);
    if (existing) return existing;
    // Grow if needed
    if (tbl->count >= tbl->capacity) {
        // Resize logic omitted for brevity
        // Would double capacity and rehash
    }
    // Add to symbols array
    GlobalSymbol *sym = &tbl->symbols[tbl->count++];
    memset(sym, 0, sizeof(GlobalSymbol));
    strncpy(sym->name, name, sizeof(sym->name) - 1);
    sym->state = SYM_UNDEF;
    // Add to hash table
    uint32_t h = hash_string(name) % tbl->hash_size;
    while (tbl->hash_table[h] != NULL) {
        h = (h + 1) % tbl->hash_size;
    }
    tbl->hash_table[h] = sym;
    return sym;
}
```
## Symbol Resolution: The Rules
Now comes the core logic. When you encounter a symbol from an input file, you need to decide how it interacts with existing entries in the global table.

![COMMON Symbol Merging Flow](./diagrams/tdd-diag-021.svg)

![Strong/Weak Symbol Resolution Rules](./diagrams/diag-strong-weak-resolution.svg)

### Rule 1: Local Symbols Are Private
Local symbols (`STB_LOCAL`) are invisible outside their defining translation unit. They're resolved immediately to their section offset and never enter the global table:
```c
int process_local_symbol(LinkerContext *ctx, ObjectFile *obj, Symbol *sym) {
    // Local symbols don't participate in global resolution
    // Just record their output offset using the section mapping
    if (sym->section_idx == SHN_UNDEF || sym->section_idx == SHN_COMMON) {
        // This shouldn't happen for local symbols
        fprintf(stderr, "Warning: local symbol %s with UNDEF/COMMON section\n", 
                sym->name);
        return 0;
    }
    // Find the output section and offset
    InputSection *input_sec = &obj->sections[sym->section_idx];
    uint64_t output_offset;
    if (lookup_output_offset(ctx, obj->filename, input_sec->name,
                             sym->value, &output_offset) != 0) {
        fprintf(stderr, "Failed to find output mapping for local symbol %s\n",
                sym->name);
        return -1;
    }
    // Store locally in the object file for relocation use
    // (We'll add a local symbol cache to ObjectFile)
    sym->final_address = ctx->base_address + output_offset;
    return 0;
}
```
The key insight: **local symbols with the same name in different files are completely unrelated**. Each `static int counter` is a different variable.
### Rule 2: Strong + Strong = Error
Two strong (global) definitions of the same symbol is a fatal error:
```c
// main.c
int config = 42;
// utils.c  
int config = 100;  // ERROR: duplicate definition
```
```c
int check_strong_strong_conflict(GlobalSymbol *existing, Symbol *new_sym,
                                  const char *filename) {
    if (existing->state == SYM_DEFINED && existing->is_strong &&
        new_sym->binding == STB_GLOBAL && 
        new_sym->section_idx != SHN_UNDEF && 
        new_sym->section_idx != SHN_COMMON) {
        fprintf(stderr, "error: duplicate symbol '%s'\n", existing->name);
        fprintf(stderr, "  first defined in: %s\n", 
                existing->source_file_idx >= 0 ? 
                ctx->inputs[existing->source_file_idx].filename : "(unknown)");
        fprintf(stderr, "  also defined in: %s\n", filename);
        return -1;
    }
    return 0;
}
```
### Rule 3: Strong + Weak = Strong Wins
A strong definition overrides a weak definition. This is intentional — weak symbols are fallbacks:
```c
// libc provides weak default
__attribute__((weak)) void* malloc(size_t n) {
    return NULL;  // Weak default: always fails
}
// Your code provides strong override
void* malloc(size_t n) {
    return my_allocator_alloc(n);  // Strong definition wins
}
```
```c
int resolve_weak_strong(GlobalSymbol *existing, Symbol *new_sym,
                        int new_file_idx, int new_sym_idx) {
    int new_is_strong = (new_sym->binding == STB_GLOBAL &&
                         new_sym->section_idx != SHN_UNDEF &&
                         new_sym->section_idx != SHN_COMMON);
    int new_is_weak = (new_sym->binding == STB_WEAK &&
                       new_sym->section_idx != SHN_UNDEF);
    // Strong definition overrides weak
    if (new_is_strong && !existing->is_strong) {
        existing->is_strong = 1;
        existing->source_file_idx = new_file_idx;
        existing->source_sym_idx = new_sym_idx;
        existing->section_idx = new_sym->section_idx;  // Will be mapped later
        existing->section_offset = new_sym->value;
        existing->state = SYM_DEFINED;
        return 0;
    }
    // Weak definition doesn't override strong
    if (new_is_weak && existing->is_strong) {
        return 0;  // Keep existing
    }
    // Weak + weak: keep first (or either, they should be equivalent)
    if (new_is_weak && !existing->is_strong) {
        // Keep existing, or update — doesn't matter
        return 0;
    }
    return 0;
}
```
### Rule 4: Weak + Weak = Either (First Wins)
Multiple weak definitions are allowed. The linker picks one (typically the first encountered):
```c
// file1.c
__attribute__((weak)) int debug_level = 1;
// file2.c
__attribute__((weak)) int debug_level = 2;
// Both are valid; linker picks one. Don't rely on which!
```
### Rule 5: COMMON Symbols — Largest Size Wins
COMMON symbols represent tentative definitions. They're merged by taking the largest size:
```c
// file1.c
int buffer[100];  // COMMON, size 400 bytes
// file2.c  
int buffer[200];  // COMMON, size 800 bytes
// Result: buffer has 800 bytes
```

![Local vs Global Symbol Processing](./diagrams/tdd-diag-022.svg)

![COMMON Symbol Merging](./diagrams/diag-common-symbols.svg)

```c
int process_common_symbol(GlobalSymbol *existing, Symbol *new_sym,
                          int new_file_idx) {
    // New symbol is COMMON
    if (existing->state == SYM_UNDEF) {
        // First COMMON definition
        existing->state = SYM_COMMON;
        existing->common_size = new_sym->size;
        existing->source_file_idx = new_file_idx;
        existing->binding = STB_GLOBAL;
        existing->is_strong = 1;  // COMMON acts as strong for final output
        return 0;
    }
    if (existing->state == SYM_COMMON) {
        // Merge: largest size wins
        if (new_sym->size > existing->common_size) {
            existing->common_size = new_sym->size;
            // Keep the larger one's file reference for debug info
            existing->source_file_idx = new_file_idx;
        }
        return 0;
    }
    if (existing->state == SYM_DEFINED && existing->is_strong) {
        // Strong definition already exists; COMMON is ignored
        return 0;
    }
    return 0;
}
```
### Rule 6: Undefined Symbols — Must Be Resolved by End
An undefined symbol (one referenced but not defined in any input file) must either:
1. Be found in a library (not covered in this milestone)
2. Be weak (optional — can remain undefined if weak)
3. Cause a linker error
```c
int check_undefined_symbols(GlobalSymbolTable *tbl) {
    int errors = 0;
    for (size_t i = 0; i < tbl->count; i++) {
        GlobalSymbol *sym = &tbl->symbols[i];
        if (sym->state == SYM_UNDEF) {
            // Check if weak undefined (allowed)
            if (sym->binding == STB_WEAK) {
                sym->final_address = 0;  // Weak undefined resolves to 0
                sym->state = SYM_RESOLVED;
                continue;
            }
            fprintf(stderr, "error: undefined symbol '%s'\n", sym->name);
            fprintf(stderr, "  referenced by: %s\n", 
                    sym->source_file_idx >= 0 ?
                    ctx->inputs[sym->source_file_idx].filename : "(unknown)");
            errors++;
        }
    }
    return errors;
}
```

![GlobalSymbol Struct Memory Layout](./diagrams/tdd-diag-024.svg)

![Undefined Symbol Detection Flow](./diagrams/diag-undefined-symbol-detection.svg)

## The Resolution Algorithm
Putting it all together, here's the two-pass resolution process:
### Pass 1: Collect All Symbols
```c
int collect_symbols(LinkerContext *ctx) {
    for (size_t file_idx = 0; file_idx < ctx->input_count; file_idx++) {
        ObjectFile *obj = &ctx->inputs[file_idx];
        for (size_t sym_idx = 0; sym_idx < obj->symbol_count; sym_idx++) {
            Symbol *sym = &obj->symbols[sym_idx];
            // Skip null symbol (index 0)
            if (sym->name[0] == '\0') continue;
            // Handle local symbols separately
            if (sym->binding == STB_LOCAL) {
                process_local_symbol(ctx, obj, sym);
                continue;
            }
            // Skip section and file symbols (internal use)
            if (sym->type == STT_SECTION || sym->type == STT_FILE) {
                continue;
            }
            // Get or create global symbol entry
            GlobalSymbol *global = insert_symbol(&ctx->global_syms, sym->name);
            // Track references (undefined symbols)
            if (sym->section_idx == SHN_UNDEF) {
                if (global->state == SYM_UNDEF && global->ref_count == 0) {
                    // First reference — record where
                    global->source_file_idx = file_idx;
                }
                global->ref_count++;
                global->binding = sym->binding;  // Might be weak
                continue;
            }
            // Handle COMMON symbols
            if (sym->section_idx == SHN_COMMON) {
                process_common_symbol(global, sym, file_idx);
                continue;
            }
            // Regular definition
            int is_strong = (sym->binding == STB_GLOBAL);
            int is_weak = (sym->binding == STB_WEAK);
            // Check for conflicts
            if (global->state == SYM_DEFINED) {
                if (global->is_strong && is_strong) {
                    fprintf(stderr, "error: duplicate symbol '%s'\n", sym->name);
                    fprintf(stderr, "  defined in both %s and %s\n",
                            ctx->inputs[global->source_file_idx].filename,
                            obj->filename);
                    return -1;
                }
                // Strong overrides weak
                if (is_strong && !global->is_strong) {
                    global->is_strong = 1;
                    global->source_file_idx = file_idx;
                    global->source_sym_idx = sym_idx;
                    global->section_idx = sym->section_idx;
                    global->section_offset = sym->value;
                    global->type = sym->type;
                    global->size = sym->size;
                }
                // Otherwise keep existing
            } else {
                // First definition
                global->state = SYM_DEFINED;
                global->is_strong = is_strong;
                global->source_file_idx = file_idx;
                global->source_sym_idx = sym_idx;
                global->section_idx = sym->section_idx;
                global->section_offset = sym->value;
                global->type = sym->type;
                global->size = sym->size;
                global->binding = sym->binding;
            }
        }
    }
    return 0;
}
```
### Pass 2: Check for Undefined Symbols
```c
int verify_all_symbols_defined(LinkerContext *ctx) {
    int errors = 0;
    for (size_t i = 0; i < ctx->global_syms.count; i++) {
        GlobalSymbol *sym = &ctx->global_syms.symbols[i];
        if (sym->state == SYM_UNDEF) {
            if (sym->binding == STB_WEAK) {
                // Weak undefined is OK — resolves to 0
                sym->final_address = 0;
                sym->state = SYM_RESOLVED;
                continue;
            }
            fprintf(stderr, "undefined symbol: %s\n", sym->name);
            errors++;
        }
    }
    return errors;
}
```
## Assigning Final Addresses
Once all symbols are resolved (either defined or weak-undefined), you need to compute their final virtual addresses. This uses the section mapping table from Milestone 1:
```c
int assign_symbol_addresses(LinkerContext *ctx) {
    for (size_t i = 0; i < ctx->global_syms.count; i++) {
        GlobalSymbol *sym = &ctx->global_syms.symbols[i];
        if (sym->state == SYM_RESOLVED) {
            continue;  // Already has address (weak undefined = 0)
        }
        if (sym->state == SYM_COMMON) {
            // Allocate COMMON symbol in .bss
            // Find or create .bss output section
            OutputSection *bss = find_output_section(ctx, ".bss");
            if (!bss) {
                // Create .bss if it doesn't exist
                bss = create_output_section(ctx, ".bss", 
                                            SHF_ALLOC | SHF_WRITE, 
                                            sym->common_align > 0 ? sym->common_align : 8);
            }
            // Align and allocate
            uint64_t offset = align_up(bss->mem_size, 
                                       sym->common_align > 0 ? sym->common_align : 8);
            sym->section_idx = find_section_index(ctx, ".bss");
            sym->section_offset = offset;
            sym->final_address = bss->virtual_addr + offset;
            sym->state = SYM_RESOLVED;
            // Update .bss size
            bss->mem_size = offset + sym->common_size;
            continue;
        }
        if (sym->state == SYM_DEFINED) {
            // Find the output section for this symbol
            ObjectFile *obj = &ctx->inputs[sym->source_file_idx];
            // Map input section to output section
            // The section_idx in the symbol is the INPUT section index
            InputSection *input_sec = &obj->sections[sym->section_idx];
            // Find output section by name
            OutputSection *out_sec = find_output_section(ctx, input_sec->name);
            if (!out_sec) {
                fprintf(stderr, "error: symbol %s in unknown section %s\n",
                        sym->name, input_sec->name);
                return -1;
            }
            // Calculate output offset using mapping table
            uint64_t output_offset;
            if (lookup_output_offset(ctx, obj->filename, input_sec->name,
                                     sym->section_offset, &output_offset) != 0) {
                fprintf(stderr, "error: no mapping for symbol %s\n", sym->name);
                return -1;
            }
            // Final address = section virtual address + offset within section
            sym->final_address = out_sec->virtual_addr + output_offset;
            sym->section_idx = find_section_index(ctx, input_sec->name);
            sym->section_offset = output_offset;
            sym->state = SYM_RESOLVED;
        }
    }
    return 0;
}
```

![Symbol Address Calculation Algorithm](./diagrams/tdd-diag-023.svg)

![Final Address Assignment Process](./diagrams/diag-symbol-address-assignment.svg)

## Helper Functions
You need a few utility functions:
```c
OutputSection* find_output_section(LinkerContext *ctx, const char *name) {
    for (size_t i = 0; i < ctx->output_count; i++) {
        if (strcmp(ctx->outputs[i].name, name) == 0) {
            return &ctx->outputs[i];
        }
    }
    return NULL;
}
int find_section_index(LinkerContext *ctx, const char *name) {
    for (size_t i = 0; i < ctx->output_count; i++) {
        if (strcmp(ctx->outputs[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}
// Look up symbol by name for relocation processing
GlobalSymbol* resolve_symbol_by_name(LinkerContext *ctx, const char *name) {
    return lookup_symbol(&ctx->global_syms, name);
}
```
## A Complete Example: Resolving `main` and `helper`
Let's trace through resolution with our `main.o` and `utils.o` example:
```
main.o symbols:
  [0] (null)           LOCAL  NOTYPE   UNDEF
  [1] main.c           LOCAL  FILE     ABS
  [2] .text            LOCAL  SECTION  1
  [3] main             GLOBAL FUNC     .text+0x0  size=26
  [4] helper           GLOBAL NOTYPE   UNDEF      (reference)
utils.o symbols:
  [0] (null)           LOCAL  NOTYPE   UNDEF
  [1] utils.c          LOCAL  FILE     ABS
  [2] .text            LOCAL  SECTION  1
  [3] helper           GLOBAL FUNC     .text+0x0  size=12
  [4] data1            GLOBAL OBJECT   .data+0x0  size=8
```
**Processing main.o:**
1. `[1] main.c` (LOCAL FILE) — skip, internal
2. `[2] .text` (LOCAL SECTION) — skip, internal
3. `[3] main` (GLOBAL FUNC .text+0x0) — insert into global table:
   - `main`: state=DEFINED, strong=1, source=main.o, section=.text, offset=0
4. `[4] helper` (GLOBAL UNDEF) — insert into global table:
   - `helper`: state=UNDEF, ref_count=1, source=main.o (first reference)
**Processing utils.o:**
1. `[1] utils.c` (LOCAL FILE) — skip
2. `[2] .text` (LOCAL SECTION) — skip
3. `[3] helper` (GLOBAL FUNC .text+0x0) — lookup existing:
   - `helper` exists with state=UNDEF
   - New definition is strong, existing is undefined → define it
   - `helper`: state=DEFINED, strong=1, source=utils.o, section=.text, offset=0
4. `[4] data1` (GLOBAL OBJECT .data+0x0) — insert:
   - `data1`: state=DEFINED, strong=1, source=utils.o, section=.data, offset=0
**After collection:**
```
Global Symbol Table:
  main:    DEFINED, strong, main.o/.text+0x0
  helper:  DEFINED, strong, utils.o/.text+0x0
  data1:   DEFINED, strong, utils.o/.data+0x0
```
**Address assignment (using section mapping from M1):**
```
Output .text: vaddr=0x400000
  - main.o/.text at offset 0x0
  - utils.o/.text at offset 0x80
main:
  - In main.o/.text at 0x0
  - main.o/.text maps to output .text at offset 0x0
  - final_address = 0x400000 + 0x0 = 0x400000
helper:
  - In utils.o/.text at 0x0
  - utils.o/.text maps to output .text at offset 0x80
  - final_address = 0x400000 + 0x80 = 0x400080
data1:
  - In utils.o/.data at 0x0
  - utils.o/.data maps to output .data at offset 0x0
  - final_address = 0x400120 + 0x0 = 0x400120
```
**Final global symbol table:**
```
  main:    0x400000
  helper:  0x400080
  data1:   0x400120
```
Now when relocation processing needs to patch the `call helper` instruction in `main.o`, it looks up `helper` and finds `0x400080`.
## Error Messages That Actually Help
Good linker error messages are precious. Here's how to make them useful:
```c
void report_undefined_symbol(LinkerContext *ctx, GlobalSymbol *sym) {
    fprintf(stderr, "ld: error: undefined symbol: %s\n", sym->name);
    // Find all references to this symbol
    fprintf(stderr, ">>> referenced by:\n");
    for (size_t i = 0; i < ctx->input_count; i++) {
        ObjectFile *obj = &ctx->inputs[i];
        // Check relocations for references
        for (int j = 0; j < obj->section_count; j++) {
            InputSection *sec = &obj->sections[j];
            if (sec->type != SHT_REL && sec->type != SHT_RELA) continue;
            // Parse relocations and check symbol references
            // (Implementation depends on relocation format)
        }
    }
    // Suggest similar symbols
    fprintf(stderr, ">>> did you mean: ");
    suggest_similar_symbols(ctx, sym->name);
}
void suggest_similar_symbols(LinkerContext *ctx, const char *name) {
    int suggested = 0;
    for (size_t i = 0; i < ctx->global_syms.count && suggested < 5; i++) {
        GlobalSymbol *sym = &ctx->global_syms.symbols[i];
        if (sym->state != SYM_UNDEF && is_similar(name, sym->name)) {
            fprintf(stderr, "%s%s", suggested ? ", " : "", sym->name);
            suggested++;
        }
    }
    if (suggested) {
        fprintf(stderr, "?\n");
    } else {
        fprintf(stderr, "(no similar symbols found)\n");
    }
}
// Simple similarity: common prefix
int is_similar(const char *a, const char *b) {
    int common = 0;
    while (*a && *b && *a == *b) {
        common++;
        a++;
        b++;
    }
    return common >= 3;  // At least 3 common characters
}
```
### C++ Name Demangling
C++ symbols are mangled (e.g., `_Z4funci` for `func(int)`). For helpful error messages, you need to demangle:
```c
#include <cxxabi.h>  // GNU ABI
void print_demangled(const char *mangled) {
    int status;
    char *demangled = abi::__cxa_demangle(mangled, NULL, NULL, &status);
    if (status == 0 && demangled) {
        fprintf(stderr, "%s (%s)\n", demangled, mangled);
        free(demangled);
    } else {
        fprintf(stderr, "%s\n", mangled);
    }
}
```
This is why `extern "C"` exists in C++ — it disables name mangling for C compatibility:
```cpp
extern "C" void c_function();  // Symbol is just "c_function", not "_Z10c_functionv"
```
## Testing Symbol Resolution
Create test cases that exercise each rule:
```bash
# Test 1: Basic resolution
cat > test_main.c << 'EOF'
extern void helper(void);
int main() { helper(); return 0; }
EOF
cat > test_helper.c << 'EOF'
void helper(void) {}
EOF
gcc -c test_main.c test_helper.c
./mylinker test_main.o test_helper.o
# Should succeed
# Test 2: Duplicate symbol error
cat > test_dup1.c << 'EOF'
int shared = 42;
EOF
cat > test_dup2.c << 'EOF'
int shared = 100;
EOF
gcc -c test_dup1.c test_dup2.c
./mylinker test_dup1.o test_dup2.o
# Should error: duplicate symbol 'shared'
# Test 3: Weak symbol override
cat > test_weak.c << 'EOF'
__attribute__((weak)) int config = 1;
EOF
cat > test_strong.c << 'EOF'
int config = 2;
EOF
gcc -c test_weak.c test_strong.c
./mylinker test_weak.o test_strong.o
# Should succeed, config = 2 (strong wins)
# Test 4: Undefined symbol error
cat > test_undef.c << 'EOF'
int main() { missing_function(); return 0; }
EOF
gcc -c test_undef.c
./mylinker test_undef.o
# Should error: undefined symbol 'missing_function'
# Test 5: COMMON symbol merging
cat > test_common1.c << 'EOF'
int buffer[100];
EOF
cat > test_common2.c << 'EOF'
int buffer[200];
EOF
gcc -c test_common1.c test_common2.c
./mylinker test_common1.o test_common2.o
# Should succeed with buffer size 800 bytes (largest)
```
## Common Pitfalls
### Pitfall 1: Forgetting to Handle Section Symbols
Relocations often reference section symbols (`STT_SECTION`) instead of named symbols. Your resolution code must handle these:
```c
// In relocation processing (next milestone):
if (sym_idx == SHN_UNDEF) {
    // Relocation references the section itself, not a named symbol
    // Use section's base address
}
```
### Pitfall 2: Not Tracking Reference Counts
You need to know how many times a symbol is referenced to detect "defined but never used" warnings:
```c
if (sym->ref_count == 0 && sym->state == SYM_DEFINED) {
    fprintf(stderr, "warning: symbol '%s' defined but not used\n", sym->name);
}
```
### Pitfall 3: Misunderstanding Weak Undefined Symbols
A weak undefined symbol is allowed to remain undefined. It resolves to address 0:
```c
// Code that checks for optional feature
extern __attribute__((weak)) int optional_feature;
if (&optional_feature != NULL) {
    // Feature is available
}
```
### Pitfall 4: COMMON Symbols in .bss vs .data
COMMON symbols go in `.bss` (uninitialized data), but they might have an alignment requirement from the first definition that saw one:
```c
// Check if any definition had alignment info
if (sym->common_align == 0) {
    sym->common_align = 8;  // Default alignment
}
```
## Integration with the Linker Pipeline
Here's how symbol resolution fits into the overall flow:
```c
int link_files(LinkerContext *ctx, const char **inputs, int input_count) {
    // Phase 1: Parse all object files
    for (int i = 0; i < input_count; i++) {
        if (parse_object_file(inputs[i], &ctx->inputs[i]) != 0) {
            return -1;
        }
        if (parse_symbols(&ctx->inputs[i]) != 0) {
            return -1;
        }
    }
    // Phase 2: Merge sections (Milestone 1)
    if (merge_all_sections(ctx) != 0) {
        return -1;
    }
    // Phase 3: Assign virtual addresses to sections
    if (assign_virtual_addresses(ctx, 0x400000) != 0) {
        return -1;
    }
    // Phase 4: Symbol resolution (this milestone)
    if (collect_symbols(ctx) != 0) {
        return -1;
    }
    int undef_count = verify_all_symbols_defined(ctx);
    if (undef_count > 0) {
        return -1;
    }
    if (assign_symbol_addresses(ctx) != 0) {
        return -1;
    }
    // Phase 5: Relocation processing (next milestone)
    // ...
    // Phase 6: Write executable (Milestone 4)
    // ...
    return 0;
}
```
## What's Next?
You've built the global symbol table. Every named entity now has a final virtual address. But your sections still have holes — the `call` instructions still have placeholder values where `helper`'s address should go.
In the next milestone, **Relocation Processing**, you'll:
1. Parse relocation entries from `.rela.*` sections
2. Look up target symbols in the global table you just built
3. Calculate patch values (accounting for PC-relative addressing)
4. Write the computed addresses into section data at the right offsets
The symbol table is the bridge — relocation processing will query it constantly. When a relocation says "put the address of symbol 'helper' at offset 0x5 in .text", you look up `helper` in your global table, find `0x400080`, and patch that value into the code.
---
## Knowledge Cascade
**Static vs Dynamic Linking** — Symbol resolution is where the fundamental split occurs. Static linking resolves everything at link time (what you're building). Dynamic linking leaves some symbols undefined, to be resolved at load time by the dynamic linker (`ld.so`). The symbol table format is the same, but dynamic symbols live in `.dynsym` instead of `.symtab`, and undefined symbols are allowed if they're marked for dynamic resolution.
**C++ Name Mangling (cross-domain, compilers)** — You've seen mangled names like `_Z4funci`. The compiler mangles because the linker's symbol table is flat — it can't distinguish `func(int)` from `func(float)` by name alone. The mangling encodes parameter types into the symbol name. `extern "C"` disables mangling, which is why C++ can call C functions: both sides agree on the unmangled name. This is also why function overloading doesn't work in C — the linker can't tell the functions apart.
**Weak Symbols and Aliases** — The weak/strong mechanism you implemented enables powerful patterns. Function interposition: define a strong `malloc` that overrides libc's weak default. Plugin architectures: check `if (&plugin_hook)` to see if a plugin provided the hook. Generic defaults: library provides weak implementation, user can override. GCC's `__attribute__((alias("other_func")))` creates multiple names for the same address — both go in the symbol table pointing to the same location.
**Archive Extraction Semantics** — Static libraries (`.a` files) are archives of object files. The linker doesn't link every object in the archive — it only extracts objects that define symbols currently undefined. This is why linking order matters: if `libA.a` depends on `libB.a`, you must list `-lA -lB`. If you list `-lB -lA`, when the linker processes `-lB`, no symbols from `libA` are undefined yet, so nothing gets extracted. The symbol resolution algorithm you built is the engine behind this extraction decision.
**Symbol Versioning (advanced)** — glibc uses symbol versioning to maintain ABI compatibility across releases. `memcpy@GLIBC_2.2.5` and `memcpy@GLIBC_2.14` are different symbols in the same library. The symbol table has a version tag, and the dynamic linker resolves to the oldest version the executable was linked against. This is how old binaries keep working on new glibc — the old symbol versions are still there.
[[CRITERIA_JSON: {"milestone_id": "build-linker-m2", "criteria": ["Parse .symtab section from each input object file and extract symbol entries with name, type, binding, section index, value, and size", "Build a hash-table-backed global symbol table supporting O(1) lookup by symbol name", "Handle local symbols (STB_LOCAL) by resolving them within their translation unit only, not entering them into the global table", "Process undefined symbols (st_shndx == SHN_UNDEF) by recording references and tracking which files reference them", "Detect and report duplicate strong symbol definitions (STB_GLOBAL with same name defined in multiple files) as errors", "Implement weak symbol resolution: STB_WEAK definitions are overridden by STB_GLOBAL, multiple weak definitions coexist", "Handle COMMON symbols (st_shndx == SHN_COMMON) by merging with largest-size-wins rule and allocating in .bss", "Detect and report non-weak undefined symbols that have no definition in any input file", "Assign final virtual addresses to all resolved symbols using section mapping table (symbol's section offset + output section's virtual address + intra-section offset)", "Support lookup queries: given symbol name, return final virtual address for relocation processing", "Handle weak undefined symbols by resolving them to address 0 without error (optional symbol pattern)"]}]
<!-- END_MS -->


<!-- MS_ID: build-linker-m3 -->
# Relocation Processing: The Art of Fixing Broken Addresses
You've merged sections. You've resolved symbols. Every named entity has a final virtual address. But here's the problem: your code sections still contain **placeholder values** where addresses should be. The `call` instruction in `main.o` that invokes `helper()` doesn't contain `helper`'s address — it contains zeros, or a relative offset to nowhere.
This milestone is about the linker's most delicate operation: **patching binary code with the correct addresses**. Get one byte wrong, and the program crashes. Get the calculation wrong by even one bit, and execution goes off into the weeds. This is where the rubber meets the road — where all your careful symbol resolution and section mapping produces actual, runnable machine code.
## The Tension: Addresses Unknown at Compile Time
Here's what you probably think happens: the compiler generates a `call helper` instruction, the linker finds `helper`, done. But that model is fundamentally wrong. Let's see why:
```c
// main.c
void helper(void);  // Declaration
int main() {
    helper();       // Call site
    return 0;
}
```
When the compiler translates `main.c` to `main.o`, it encounters a problem: **it has no idea where `helper` will live**. The compiler only sees `main.c` — it doesn't know about `utils.c`, it doesn't know what other files will be linked, and it certainly doesn't know the final memory layout.

![Little-Endian Patch Write](./diagrams/tdd-diag-038.svg)


So what does the compiler do? It cheats:
1. Emits the instruction with a **placeholder value** (often zeros)
2. Creates a **relocation entry** that says: "Hey linker, once you know where `helper` ends up, patch this location with the right value"
The compiler is essentially leaving IOUs scattered throughout the object file. The linker's job is to **cash those IOUs** — to find every placeholder and replace it with the real address.
> **🔑 Revelation: The Hidden Truth**
> 
> Developers think function calls in object files already contain the target address. In reality, the compiler emits placeholder values (often zeros) and creates relocation entries saying "once you know where foo() ends up, patch this offset with the right address." The linker is the **first time** these addresses are known.
> 
> Consider what this means: every cross-file function call, every global variable reference, every address constant in your program — **all of them** are patched by the linker. The object file the compiler produces isn't runnable. It's a collection of fragments held together by relocation records, waiting for the linker to make them whole.
The fundamental tension: **compilation happens in isolation, but execution requires global knowledge**. The compiler can only see one translation unit at a time, but the final addresses depend on the entire program's layout. Relocation entries are the bridge — they carry the necessary context forward so the linker can finish what the compiler started.
## What Is a Relocation, Really?
A relocation is a **fixup instruction** embedded in the object file. It tells the linker:
> "At offset X in section Y, there's a placeholder value. Replace it with an address computed from symbol Z, using formula W."
The key insight: **relocations are metadata, not code**. They live in separate sections (`.rela.text`, `.rela.data`) and describe modifications to be made elsewhere. They're like margin notes on a manuscript — instructions for the editor, not part of the final text.
> **🔑 Foundation: Relocation entry format (r_offset, r_info, r_addend)**
> 
> ## What It Is
> Every relocation in an ELF file is represented by a structure that answers three questions:
> - **Where** do I patch? (`r_offset`)
> - **What** symbol's address do I use? (`r_info`)
> - **How** do I compute the final value? (`r_info` contains the type, plus `r_addend` for adjustments)
> 
> ```c
> // RELA format: relocation with explicit addend (64-bit x86-64 uses this)
> typedef struct {
>     uint64_t r_offset;   // Where to patch (offset within the section)
>     uint64_t r_info;     // Symbol index (top 32 bits) + relocation type (bottom 32 bits)
>     int64_t  r_addend;   // Value to add to symbol address
> } Elf64_Rela;
> 
> // REL format: relocation with implicit addend (stored in the target location)
> typedef struct {
>     uint64_t r_offset;
>     uint64_t r_info;
>     // No r_addend! The addend is whatever bytes are already at r_offset
> } Elf64_Rel;
> ```
> 
> **r_offset**: The byte offset within the section where the patch should be applied. Not a virtual address — this is relative to the section's start.
> 
> **r_info**: A packed 64-bit field. The high 32 bits contain the symbol table index — which symbol this relocation references. The low 32 bits contain the relocation type — what formula to use when computing the patch value.
> 
> **r_addend**: A signed value added to the symbol's address. This handles cases like "address of symbol plus 4" or "address of symbol minus the current instruction pointer." The addend is crucial for PC-relative addressing.
> 
> ## Why You Need This Right Now
> Without understanding relocation entries, you cannot write a linker. Period. Every cross-file reference in your program generates at least one relocation. A typical C program might have thousands of them. Your linker must:
> - Parse these entries from `.rela.*` sections
> - Look up the referenced symbols
> - Apply the correct formula based on relocation type
> - Write the computed value at the specified offset
> 
> ## Key Mental Model
> **Think of relocations as deferred computations.**
> 
> The compiler wanted to write `value = symbol_address + adjustment` but couldn't because `symbol_address` was unknown. So it wrote a relocation entry instead — a note that says "compute this later when you know the address."
> 
> The formula isn't stored explicitly — it's **implied by the relocation type**. Type `R_X86_64_64` means "write the full 64-bit address." Type `R_X86_64_PC32` means "write a 32-bit PC-relative offset." The type encodes the computation; the entry provides the parameters.

![Section Symbol Resolution Flow](./diagrams/tdd-diag-035.svg)

![Relocation Entry Structure](./diagrams/diag-relocation-entry-format.svg)

![call Instruction Relocation Example](./diagrams/tdd-diag-039.svg)

## The Relocation Section Structure
Relocations don't live in the code sections they modify — they live in separate sections named `.rela.<section>` or `.rel.<section>`. For example:
- `.rela.text` — relocations for the `.text` section
- `.rela.data` — relocations for the `.data` section
- `.rel.text` — same, but with implicit addends (older format)
The section header tells you which section the relocations apply to:
```c
// The sh_info field of a relocation section header points to the section it modifies
// For .rela.text, sh_info = index of .text section
// The sh_link field points to the symbol table to use for symbol lookups
```
Let's parse these sections:
```c
#define ELF64_R_SYM(i)    ((i) >> 32)    // Extract symbol index
#define ELF64_R_TYPE(i)   ((i) & 0xffffffff)  // Extract relocation type
// Relocation types we'll handle (x86-64)
#define R_X86_64_NONE     0   // No relocation
#define R_X86_64_64       1   // Direct 64-bit absolute address
#define R_X86_64_PC32     2   // PC-relative 32-bit signed
#define R_X86_64_32       10  // Direct 32-bit zero-extended
#define R_X86_64_32S      11  // Direct 32-bit sign-extended
typedef struct {
    uint64_t offset;        // Where to patch (section-relative)
    uint32_t sym_idx;       // Symbol table index
    uint32_t type;          // Relocation type (R_X86_64_*)
    int64_t addend;         // Value to add
    int target_section_idx; // Which section this relocation targets
} Relocation;
int parse_relocations(ObjectFile *obj) {
    // Find all relocation sections
    for (int i = 0; i < obj->section_count; i++) {
        InputSection *sec = &obj->sections[i];
        if (sec->type != SHT_RELA && sec->type != SHT_REL) {
            continue;
        }
        // This is a relocation section
        // sh_info tells us which section it targets
        // We need to look at the original section header for this
        int target_section_idx = /* from original Elf64_Shdr.sh_info */;
        // sh_link tells us which symbol table to use (usually .symtab)
        int symtab_idx = /* from original Elf64_Shdr.sh_link */;
        size_t entry_size = (sec->type == SHT_RELA) ? 
                            sizeof(Elf64_Rela) : sizeof(Elf64_Rel);
        size_t count = sec->size / entry_size;
        // Allocate relocations
        obj->relocations = realloc(obj->relocations, 
                                   (obj->relocation_count + count) * sizeof(Relocation));
        if (sec->type == SHT_RELA) {
            Elf64_Rela *relas = (Elf64_Rela*)sec->data;
            for (size_t j = 0; j < count; j++) {
                Relocation *rel = &obj->relocations[obj->relocation_count++];
                rel->offset = relas[j].r_offset;
                rel->sym_idx = ELF64_R_SYM(relas[j].r_info);
                rel->type = ELF64_R_TYPE(relas[j].r_info);
                rel->addend = relas[j].r_addend;
                rel->target_section_idx = target_section_idx;
            }
        } else {
            // REL format: addend is implicit (already in the code)
            Elf64_Rel *rels = (Elf64_Rel*)sec->data;
            for (size_t j = 0; j < count; j++) {
                Relocation *rel = &obj->relocations[obj->relocation_count++];
                rel->offset = rels[j].r_offset;
                rel->sym_idx = ELF64_R_SYM(rels[j].r_info);
                rel->type = ELF64_R_TYPE(rels[j].r_info);
                // Read addend from the target location
                // (This is tricky — we need to read from the target section)
                rel->addend = 0;  // Will be read during processing
                rel->target_section_idx = target_section_idx;
            }
        }
    }
    return 0;
}
```
## The Two Relocation Types You Must Handle
For a basic x86-64 linker, you need to handle two fundamental relocation types. These cover the vast majority of cases:
### R_X86_64_64: Absolute 64-bit Address
This is the simple case: write the symbol's full 64-bit virtual address at the relocation site.
**Formula:** `*location = symbol_address + addend`
**Use case:** Global variable references in data sections, function pointer initialization, vtables.
```c
// Example: function pointer in .data
void (*callback)(void) = helper;  // Store helper's address
// In .data, the compiler emits 8 zero bytes
// Relocation: R_X86_64_64, offset=0, symbol=helper, addend=0
// After linking: those 8 bytes contain helper's actual address
```
### R_X86_64_PC32: PC-Relative 32-bit Offset
This is where it gets interesting. The instruction doesn't want an absolute address — it wants the **distance from the current instruction** to the target.
**Formula:** `*location = symbol_address + addend - relocation_site_address`
> **🔑 Revelation: The PC-Relative Secret**
> 
> PC-relative relocations don't store absolute addresses — they store offsets from the instruction pointer. The formula is:
> 
> `final_value = symbol_address + addend - relocation_site_address`
> 
> This is why position-independent code prefers PC-relative addressing: **the relative distance doesn't change when the entire program shifts by a base address**. If `helper` is 100 bytes ahead of `main`, it's 100 bytes ahead whether the program loads at 0x400000 or 0x7fff0000.
> 
> The `addend` field typically accounts for the fact that when a `call` instruction executes, the instruction pointer (RIP) points to the *next* instruction, not the current one. So `addend = -4` is common for 32-bit PC-relative fields.
**Use case:** `call` and `jmp` instructions, RIP-relative addressing modes (most code references in x86-64).
```assembly
# Example: call instruction
call helper    # Encoded as: E8 xx xx xx xx (relative offset)
# At link time:
# - call instruction is at address 0x400100
# - helper is at address 0x400200
# - The relative offset field starts at 0x400101 (after the E8 opcode)
# - We need to store: 0x400200 - (0x400101 + 4) = 0xFB
#   (RIP points to next instruction, so we subtract the end of this one)
```

![Overflow Detection for 32-bit Relocations](./diagrams/tdd-diag-036.svg)

![x86-64 Relocation Type Reference](./diagrams/diag-relocation-types-x86-64.svg)

## The Relocation Processing Algorithm
Now let's implement the actual relocation processing. The key insight: **you must apply relocations to the merged output sections, not the input sections**. The input sections have been concatenated and rearranged; their content now lives at different offsets.
```c
typedef struct {
    // ... existing LinkerContext fields ...
    uint8_t *output_buffer;     // Buffer for final executable
    size_t output_size;
} LinkerContext;
int process_relocations(LinkerContext *ctx) {
    // Iterate through all input files
    for (size_t file_idx = 0; file_idx < ctx->input_count; file_idx++) {
        ObjectFile *obj = &ctx->inputs[file_idx];
        // Process each relocation
        for (size_t rel_idx = 0; rel_idx < obj->relocation_count; rel_idx++) {
            Relocation *rel = &obj->relocations[rel_idx];
            if (apply_relocation(ctx, obj, rel) != 0) {
                return -1;
            }
        }
    }
    return 0;
}
```
### Step 1: Find Where the Relocation Site Ended Up
The relocation's `offset` is relative to the **input** section. You need to translate it to the **output** section using your mapping table:
```c
int apply_relocation(LinkerContext *ctx, ObjectFile *obj, Relocation *rel) {
    // Step 1: Find the target section's name
    InputSection *input_sec = &obj->sections[rel->target_section_idx];
    const char *section_name = input_sec->name;
    // Step 2: Find the output section
    OutputSection *out_sec = find_output_section(ctx, section_name);
    if (!out_sec) {
        fprintf(stderr, "error: relocation targets unknown section %s\n", section_name);
        return -1;
    }
    // Step 3: Translate input offset to output offset
    uint64_t output_offset;
    if (lookup_output_offset(ctx, obj->filename, section_name, 
                             rel->offset, &output_offset) != 0) {
        fprintf(stderr, "error: cannot find output mapping for relocation\n");
        return -1;
    }
    // Step 4: Get the relocation site's virtual address
    uint64_t site_vaddr = out_sec->virtual_addr + output_offset;
    // ... continue with symbol lookup and patching ...
}
```

![Relocation Processing Sequence](./diagrams/tdd-diag-037.svg)

![Relocation Patching in Action](./diagrams/diag-relocation-patching.svg)

### Step 2: Resolve the Target Symbol
Now look up the symbol this relocation references:
```c
int apply_relocation(LinkerContext *ctx, ObjectFile *obj, Relocation *rel) {
    // ... (offset translation from above) ...
    // Get the symbol from the input file's symbol table
    if (rel->sym_idx >= obj->symbol_count) {
        fprintf(stderr, "error: invalid symbol index %u in relocation\n", rel->sym_idx);
        return -1;
    }
    Symbol *sym = &obj->symbols[rel->sym_idx];
    uint64_t symbol_vaddr;
    // Handle different symbol types
    if (sym->section_idx == SHN_UNDEF) {
        // External symbol — look up in global table
        GlobalSymbol *global = lookup_symbol(&ctx->global_syms, sym->name);
        if (!global || global->state == SYM_UNDEF) {
            fprintf(stderr, "error: relocation references undefined symbol '%s'\n",
                    sym->name);
            return -1;
        }
        symbol_vaddr = global->final_address;
    }
    else if (sym->section_idx == SHN_ABS) {
        // Absolute symbol — value is already an address
        symbol_vaddr = sym->value;
    }
    else if (sym->type == STT_SECTION) {
        // Section symbol — relocation references the section itself
        // The symbol's value is the offset within the section
        InputSection *sec_sym = &obj->sections[sym->section_idx];
        OutputSection *out_sec_sym = find_output_section(ctx, sec_sym->name);
        if (!out_sec_sym) {
            fprintf(stderr, "error: section symbol references unknown section\n");
            return -1;
        }
        symbol_vaddr = out_sec_sym->virtual_addr + sym->value;
    }
    else {
        // Regular symbol — should be in global table
        GlobalSymbol *global = lookup_symbol(&ctx->global_syms, sym->name);
        if (global) {
            symbol_vaddr = global->final_address;
        } else {
            // Local symbol — compute from section
            InputSection *sec = &obj->sections[sym->section_idx];
            OutputSection *out = find_output_section(ctx, sec->name);
            uint64_t out_off;
            lookup_output_offset(ctx, obj->filename, sec->name, 
                                 sym->value, &out_off);
            symbol_vaddr = out->virtual_addr + out_off;
        }
    }
    // ... continue with patching ...
}
```
### Step 3: Compute and Apply the Patch
Now apply the relocation formula based on type:
```c
int apply_relocation(LinkerContext *ctx, ObjectFile *obj, Relocation *rel) {
    // ... (offset translation and symbol resolution from above) ...
    // Assume: site_vaddr = virtual address of relocation site
    //         symbol_vaddr = virtual address of target symbol
    // Get pointer to the location in output buffer
    uint8_t *patch_location = ctx->output_buffer + out_sec->file_offset + output_offset;
    int64_t computed_value;
    size_t patch_size;
    switch (rel->type) {
        case R_X86_64_64:
            // Absolute 64-bit address
            computed_value = (int64_t)(symbol_vaddr + rel->addend);
            patch_size = 8;
            break;
        case R_X86_64_PC32:
            // PC-relative 32-bit offset
            computed_value = (int64_t)(symbol_vaddr + rel->addend - site_vaddr);
            patch_size = 4;
            // Check for overflow
            if (computed_value > INT32_MAX || computed_value < INT32_MIN) {
                fprintf(stderr, "error: PC-relative relocation overflow\n");
                fprintf(stderr, "  symbol: %s at 0x%lx\n", sym->name, symbol_vaddr);
                fprintf(stderr, "  site: 0x%lx\n", site_vaddr);
                fprintf(stderr, "  computed offset: %ld (doesn't fit in 32 bits)\n",
                        computed_value);
                return -1;
            }
            break;
        case R_X86_64_32:
            // Absolute 32-bit (zero-extended)
            computed_value = (int64_t)(symbol_vaddr + rel->addend);
            if ((uint64_t)computed_value > UINT32_MAX) {
                fprintf(stderr, "error: 32-bit relocation overflow (symbol out of range)\n");
                return -1;
            }
            patch_size = 4;
            break;
        case R_X86_64_32S:
            // Absolute 32-bit (sign-extended)
            computed_value = (int64_t)(symbol_vaddr + rel->addend);
            if (computed_value > INT32_MAX || computed_value < INT32_MIN) {
                fprintf(stderr, "error: 32-bit signed relocation overflow\n");
                return -1;
            }
            patch_size = 4;
            break;
        case R_X86_64_NONE:
            // No-op relocation (used for debugging)
            return 0;
        default:
            fprintf(stderr, "error: unsupported relocation type %d\n", rel->type);
            return -1;
    }
    // Apply the patch
    // Note: x86-64 is little-endian, so we write LSB first
    switch (patch_size) {
        case 4:
            patch_location[0] = (computed_value >> 0) & 0xFF;
            patch_location[1] = (computed_value >> 8) & 0xFF;
            patch_location[2] = (computed_value >> 16) & 0xFF;
            patch_location[3] = (computed_value >> 24) & 0xFF;
            break;
        case 8:
            patch_location[0] = (computed_value >> 0) & 0xFF;
            patch_location[1] = (computed_value >> 8) & 0xFF;
            patch_location[2] = (computed_value >> 16) & 0xFF;
            patch_location[3] = (computed_value >> 24) & 0xFF;
            patch_location[4] = (computed_value >> 32) & 0xFF;
            patch_location[5] = (computed_value >> 40) & 0xFF;
            patch_location[6] = (computed_value >> 48) & 0xFF;
            patch_location[7] = (computed_value >> 56) & 0xFF;
            break;
    }
    return 0;
}
```

![R_X86_64_PC32 PC-Relative Calculation](./diagrams/tdd-diag-032.svg)

![PC-Relative Relocation Calculation](./diagrams/diag-pc-relative-calculation.svg)

## Understanding the Addend: Why It Exists
The `addend` field often confuses people. Why do we need it?
**Scenario 1: Accessing a field within a struct**
```c
struct Config {
    int version;
    int flags;
    char name[32];
};
extern struct Config config;
int get_flags() {
    return config.flags;  // Need address of config + 4
}
```
The compiler generates a relocation for `config` with `addend = 4`. The formula becomes:
```
patch_value = config_address + 4
```
**Scenario 2: PC-relative `call` instruction**
The x86-64 `call` instruction (`E8 rel32`) uses RIP-relative addressing. When the CPU executes it, RIP points to the *next* instruction. So the relative offset needs to account for the 4 bytes of the offset field itself:
```assembly
# At address 0x400100:
E8 XX XX XX XX    # call helper (5 bytes total)
# Next instruction at 0x400105
# To call helper at 0x400200:
# offset = 0x400200 - 0x400105 = 0xFB
# But the compiler emitted offset = 0
# So addend = 0, and we compute: 0x400200 + 0 - 0x400101 - 4 = 0xFB
# Wait, that's confusing...
```
Actually, there's a subtlety. The relocation site is at offset 1 within the instruction (after the `E8` opcode). So:
```c
// site_vaddr = 0x400101 (where the 4-byte offset field starts)
// symbol_vaddr = 0x400200 (where helper is)
// addend = -4 (because we want RIP to point to next instruction)
// 
// computed = symbol_vaddr + addend - site_vaddr
//          = 0x400200 + (-4) - 0x400101
//          = 0xFB
// 
// When executed: RIP = 0x400105 (after the instruction)
// RIP + 0xFB = 0x400200 ✓
```
Actually, for most `call` instructions, the addend is typically -4 or 0 depending on how the compiler emits it. The key is: **the addend accounts for any offset between "symbol location" and "actual target"**.
**Scenario 3: Implicit addends (REL format)**
In the older REL format (without explicit `r_addend`), the addend is whatever bytes are already at the relocation site:
```c
// For REL format:
int64_t get_implicit_addend(uint8_t *location, uint32_t type) {
    switch (type) {
        case R_X86_64_32:
        case R_X86_64_PC32:
            // Read 32-bit little-endian value
            return (int32_t)(
                location[0] | 
                (location[1] << 8) | 
                (location[2] << 16) | 
                (location[3] << 24)
            );
        case R_X86_64_64:
            // Read 64-bit little-endian value
            return (int64_t)(
                location[0] |
                ((uint64_t)location[1] << 8) |
                // ... etc
            );
        default:
            return 0;
    }
}
```
The REL format saves 8 bytes per relocation but is more complex to handle. Modern x86-64 code uses RELA exclusively.

![Relocation Type Dispatcher State Machine](./diagrams/tdd-diag-034.svg)

![Overflow Detection for Truncating Relocations](./diagrams/diag-relocation-overflow.svg)

## Processing Order: Why It Matters
You might wonder: does the order in which we process relocations matter?
**For the relocations themselves: No.** Each relocation is independent. You can process them in any order.
**But there's a dependency on symbol resolution: Yes.** You must have resolved all symbols (Milestone 2) before processing relocations. If a relocation references an unresolved symbol, you can't compute the patch value.
```c
// WRONG: Processing relocations before symbol resolution
merge_sections(ctx);
process_relocations(ctx);  // ERROR: symbols not resolved!
resolve_symbols(ctx);
// CORRECT: Resolve symbols first
merge_sections(ctx);
resolve_symbols(ctx);      // Populate global symbol table
assign_addresses(ctx);     // Give every symbol a final address
process_relocations(ctx);  // Now we have addresses to patch
```

![Relocation Processing Order](./diagrams/diag-relocation-order.svg)

## Building the Output Buffer
Before you can patch relocations, you need to assemble the merged sections into a contiguous output buffer. This is different from the in-memory representation — this is the actual bytes that will be written to the executable file:
```c
int build_output_buffer(LinkerContext *ctx) {
    // Calculate total output size
    uint64_t max_offset = 0;
    for (size_t i = 0; i < ctx->output_count; i++) {
        OutputSection *out = &ctx->outputs[i];
        if (out->file_size > 0) {
            uint64_t end = out->file_offset + out->file_size;
            if (end > max_offset) {
                max_offset = end;
            }
        }
    }
    // Allocate buffer (plus space for ELF header and program headers)
    size_t header_space = 0x1000;  // Reserve first page for headers
    ctx->output_size = header_space + max_offset;
    ctx->output_buffer = calloc(ctx->output_size, 1);
    // Copy each output section's data
    for (size_t i = 0; i < ctx->output_count; i++) {
        OutputSection *out = &ctx->outputs[i];
        if (out->file_size == 0 || out->data == NULL) {
            continue;  // Skip .bss and empty sections
        }
        // Copy to the correct file offset (offset from start of file)
        uint8_t *dest = ctx->output_buffer + header_space + out->file_offset;
        memcpy(dest, out->data, out->file_size);
    }
    return 0;
}
```
Wait, we need to actually build the merged section data. Let's add that:
```c
int merge_section_data(LinkerContext *ctx) {
    for (size_t i = 0; i < ctx->output_count; i++) {
        OutputSection *out = &ctx->outputs[i];
        if (out->file_size == 0) {
            // .bss or empty — no data to allocate
            out->data = NULL;
            continue;
        }
        // Allocate buffer for merged data
        out->data = calloc(out->file_size, 1);  // Zero-initialized
        // Copy each input section's data to the correct position
        for (size_t j = 0; j < out->mapping_count; j++) {
            SectionMapping *map = &out->mappings[j];
            // Find the input section data
            ObjectFile *src_obj = NULL;
            InputSection *src_sec = NULL;
            // Find source object and section
            for (size_t k = 0; k < ctx->input_count; k++) {
                if (strcmp(ctx->inputs[k].filename, map->source_file) == 0) {
                    src_obj = &ctx->inputs[k];
                    break;
                }
            }
            if (!src_obj) continue;
            for (int k = 0; k < src_obj->section_count; k++) {
                if (strcmp(src_obj->sections[k].name, map->section_name) == 0) {
                    src_sec = &src_obj->sections[k];
                    break;
                }
            }
            if (!src_sec) continue;
            // Copy data (if not .bss)
            if (src_sec->type != SHT_NOBITS && src_sec->data) {
                memcpy(out->data + map->output_offset, 
                       src_sec->data, 
                       map->input_size);
            }
        }
    }
    return 0;
}
```
## A Complete Example: Tracing a `call` Instruction
Let's trace through what happens when we link a simple program:
```c
// main.c
extern void helper(void);
int main() {
    helper();
    return 0;
}
// utils.c
void helper(void) {
    // Does something
}
```
**Step 1: Compiler output (main.o)**
```assembly
# Disassembly of main.o
0000 <main>:
   0:   55                      push   rbp
   1:   48 89 e5                mov    rbp,rsp
   4:   e8 00 00 00 00          call   9 <main+0x9>  # Placeholder!
   9:   31 c0                   xor    eax,eax
   b:   5d                      pop    rbp
   c:   c3                      ret
```
The `call` at offset 4 contains `00 00 00 00` — the compiler had no idea where `helper` is.
**Relocation entry in main.o:**
```
Offset  Type          Symbol   Addend
0x0005  R_X86_64_PC32 helper   -4
```
Note: offset 0x5 is where the 4-byte relative offset starts (after the `E8` opcode).
**Step 2: Section merging**
```
main.o/.text: 13 bytes at output offset 0x0000
utils.o/.text: 20 bytes at output offset 0x0010
Output .text: 30 bytes total, virtual address 0x401000
```
**Step 3: Symbol resolution**
```
main:   defined in main.o/.text+0x0   → 0x401000
helper: defined in utils.o/.text+0x0  → 0x401010
```
**Step 4: Relocation processing**
```
Relocation: main.o, .text+0x5, R_X86_64_PC32, symbol=helper, addend=-4
1. Translate input offset to output:
   Input:  main.o/.text+0x5
   Output: .text+0x5 (main.o/.text starts at 0x0)
2. Calculate site virtual address:
   site_vaddr = 0x401000 + 0x5 = 0x401005
3. Look up symbol:
   symbol_vaddr = 0x401010 (helper's final address)
4. Apply formula (R_X86_64_PC32):
   computed = symbol_vaddr + addend - site_vaddr
            = 0x401010 + (-4) - 0x401005
            = 0x401010 - 4 - 0x401005
            = 0x7
5. Write to output:
   output_buffer[0x1005] = 0x07 0x00 0x00 0x00
```
**Step 5: Verify**
```assembly
# Final linked code
0000000000401000 <main>:
  401000:   55                      push   rbp
  401001:   48 89 e5                mov    rbp,rsp
  401004:   e8 07 00 00 00          call   401010 <helper>
  401009:   31 c0                   xor    eax,eax
  40100b:   5d                      pop    rbp
  40100c:   c3                      ret
0000000000401010 <helper>:
  401010:   ...
```
When the `call` executes:
- RIP = 0x401009 (next instruction after the call)
- Relative offset in instruction = 0x00000007
- Target = RIP + offset = 0x401009 + 0x7 = 0x401010 ✓

![Input to Output Offset Translation](./diagrams/tdd-diag-033.svg)


## Handling Edge Cases
### Edge Case 1: Weak Undefined Symbols
If a relocation references a weak undefined symbol, it resolves to 0:
```c
// Code
extern __attribute__((weak)) void optional_hook(void);
void try_hook() {
    if (optional_hook) optional_hook();
}
// If no strong definition exists, optional_hook resolves to 0
// The "if (optional_hook)" check will fail (0 is false)
```
```c
// In relocation processing:
if (sym->section_idx == SHN_UNDEF) {
    GlobalSymbol *global = lookup_symbol(&ctx->global_syms, sym->name);
    if (global && global->state == SYM_UNDEF) {
        if (global->binding == STB_WEAK) {
            // Weak undefined — resolve to 0
            symbol_vaddr = 0;
        } else {
            // Strong undefined — error!
            fprintf(stderr, "undefined symbol: %s\n", sym->name);
            return -1;
        }
    } else if (global) {
        symbol_vaddr = global->final_address;
    }
}
```
### Edge Case 2: Relocations in .bss
`.bss` has no file content, so you can't have relocations there... right?
Actually, you **can** have relocations targeting `.bss` if they reference things like section symbols. But you can't have relocations **inside** `.bss` because there's no content to patch. The linker should detect this:
```c
if (input_sec->type == SHT_NOBITS) {
    fprintf(stderr, "error: cannot apply relocation inside .bss section\n");
    return -1;
}
```
### Edge Case 3: Alignment and Cross-Section Relocations
A relocation in `.text` might reference a symbol in `.data`. The formula still works because we're using virtual addresses:
```c
// .text at 0x401000
// .data at 0x403000
// Code loading address of global_var
mov rax, [global_var]  # RIP-relative load
// Relocation: R_X86_64_PC32, symbol=global_var
// global_var is at 0x403000
// site is at 0x401010
// computed = 0x403000 + 0 - 0x401010 = 0x1FF0
// The relative distance is correctly computed across sections
```
### Edge Case 4: Negative Addends
The addend can be negative. This is common for "address of symbol minus some constant":
```c
// Accessing the last element of an array
extern int array[100];
int last = array[99];  // Address is &array + 99*4 = &array + 396
// Relocation: symbol=array, addend=396
// computed = array_addr + 396
```
But negative addends for PC-relative are trickier:
```assembly
# Jump backward
jmp label  # If label is before this instruction
# Relocation might have addend that accounts for the backward jump
# The computed offset will be negative (two's complement)
```
## Debugging Relocation Problems
When relocations go wrong, the symptoms are confusing:
- Program crashes with SIGSEGV at weird addresses
- Functions are called but execute garbage
- Data loads return wrong values
Here's a debugging helper:
```c
void debug_relocation(LinkerContext *ctx, ObjectFile *obj, Relocation *rel,
                      uint64_t site_vaddr, uint64_t symbol_vaddr, 
                      int64_t computed) {
    Symbol *sym = &obj->symbols[rel->sym_idx];
    const char *type_name = "UNKNOWN";
    switch (rel->type) {
        case R_X86_64_64:   type_name = "R_X86_64_64"; break;
        case R_X86_64_PC32: type_name = "R_X86_64_PC32"; break;
        case R_X86_64_32:   type_name = "R_X86_64_32"; break;
        case R_X86_64_32S:  type_name = "R_X86_64_32S"; break;
    }
    fprintf(stderr, "DEBUG: Relocation in %s\n", obj->filename);
    fprintf(stderr, "  Type: %s\n", type_name);
    fprintf(stderr, "  Target section: %s @ offset 0x%lx\n", 
            /* section name */, rel->offset);
    fprintf(stderr, "  Site virtual address: 0x%lx\n", site_vaddr);
    fprintf(stderr, "  Symbol: %s @ 0x%lx\n", sym->name, symbol_vaddr);
    fprintf(stderr, "  Addend: %ld (0x%lx)\n", rel->addend, rel->addend);
    fprintf(stderr, "  Computed value: %ld (0x%lx)\n", computed, computed);
    if (rel->type == R_X86_64_PC32) {
        fprintf(stderr, "  Verification: RIP after = 0x%lx, target = 0x%lx\n",
                site_vaddr + 4, site_vaddr + 4 + computed);
    }
}
```
### Common Bug: Off-by-One in Offset Translation
The most common bug is getting the output offset wrong. Remember: the relocation offset is from the **start of the input section**, and you need to add the **input section's position within the output section**:
```c
// WRONG: Using input section offset directly
patch_location = output_buffer + out_sec->file_offset + rel->offset;
// CORRECT: Translating through the mapping table
uint64_t input_sec_output_offset = /* find where input section starts in output */;
patch_location = output_buffer + out_sec->file_offset + 
                 input_sec_output_offset + rel->offset;
// OR: Using the lookup function
uint64_t output_offset;
lookup_output_offset(ctx, obj->filename, section_name, rel->offset, &output_offset);
patch_location = output_buffer + out_sec->file_offset + output_offset;
```
## Testing Relocation Processing
Create test cases that exercise each relocation type:
```bash
# Test 1: Basic PC-relative call
cat > test_call.c << 'EOF'
void helper(void) {}
int main() { helper(); return 0; }
EOF
gcc -c test_call.c
./mylinker test_call.o -o test_call
./test_call  # Should run without crashing
# Test 2: Absolute address in data
cat > test_data.c << 'EOF'
extern int global_var;
int *ptr = &global_var;
int global_var = 42;
int main() { return *ptr; }
EOF
gcc -c test_data.c
./mylinker test_data.o -o test_data
./test_data; echo $?  # Should print 42
# Test 3: Cross-section reference
cat > test_cross.c << 'EOF'
const char *message = "Hello";
int main() { return message[0] == 'H' ? 0 : 1; }
EOF
gcc -c test_cross.c
./mylinker test_cross.o -o test_cross
./test_cross; echo $?  # Should print 0
# Test 4: Overflow detection
cat > test_overflow.s << 'EOF'
.global _start
_start:
    mov rax, 0x123456789ABCDEF0  # 64-bit value
    mov dword [rel target], eax  # Try to store 64-bit addr in 32-bit
target:
    .quad 0
EOF
as -o test_overflow.o test_overflow.s
./mylinker test_overflow.o -o test_overflow
# Should error: relocation overflow
```
### Verifying with objdump
Always verify your linked executable with `objdump`:
```bash
objdump -d test_call  # Disassemble .text
objdump -r test_call.o  # Show relocations in object file
objdump -R test_call  # Show dynamic relocations (should be none for static)
readelf -s test_call  # Show symbol table
```
## Integration with the Full Linker Pipeline
Here's how relocation processing fits into the complete linker:
```c
int link_files(LinkerContext *ctx, const char **inputs, int input_count) {
    // Phase 1: Parse all object files
    for (int i = 0; i < input_count; i++) {
        if (parse_object_file(inputs[i], &ctx->inputs[i]) != 0) return -1;
        if (parse_symbols(&ctx->inputs[i]) != 0) return -1;
        if (parse_relocations(&ctx->inputs[i]) != 0) return -1;
    }
    ctx->input_count = input_count;
    // Phase 2: Merge sections (Milestone 1)
    if (merge_all_sections(ctx) != 0) return -1;
    if (merge_section_data(ctx) != 0) return -1;
    // Phase 3: Assign virtual addresses to sections
    if (assign_virtual_addresses(ctx, 0x400000) != 0) return -1;
    // Phase 4: Symbol resolution (Milestone 2)
    if (collect_symbols(ctx) != 0) return -1;
    if (verify_all_symbols_defined(ctx) != 0) return -1;
    if (assign_symbol_addresses(ctx) != 0) return -1;
    // Phase 5: Build output buffer
    if (build_output_buffer(ctx) != 0) return -1;
    // Phase 6: Relocation processing (this milestone)
    if (process_relocations(ctx) != 0) return -1;
    // Phase 7: Write executable (Milestone 4)
    // ...
    return 0;
}
```

![Relocation Processing Data Flow](./diagrams/tdd-diag-030.svg)


## What's Next?
You've completed the hardest part of linking: correctly patching binary code with computed addresses. Every `call`, every global variable reference, every function pointer — all of them now contain the correct values.
In the next milestone, **Executable Generation**, you'll:
1. Wrap the patched sections in an ELF header
2. Create program headers describing segments for the loader
3. Set the entry point to `_start` (or `main`)
4. Write the final executable file
The relocation processing you just did is the heart of the linker — everything else is bookkeeping. But that bookkeeping matters, because without valid ELF headers, the OS loader won't even look at your carefully patched code.
---
## Knowledge Cascade
**Position-Independent Code (PIC)** — The distinction between `R_X86_64_PC32` and `R_X86_64_64` determines whether code can run at any address. PIC uses PC-relative relocations exclusively for code references; the relative distances don't change when the entire program is loaded at a different base address. Non-PIC code uses absolute relocations that require fixups at load time. This is why shared libraries (.so files) are compiled with `-fPIC` — the loader can map them anywhere without modifying their code sections.
**Load-Time Relocation (cross-domain, OS)** — Dynamic linking applies similar relocations at load time, but with a twist: the code sections are mapped read-only, so relocations are applied through indirection tables (GOT — Global Offset Table). Instead of patching the code directly, the loader fills in the GOT entries, and the code references the GOT. This is why reading `gdb`'s disassembly of a shared library shows `call foo@plt` instead of `call foo` — the PLT (Procedure Linkage Table) is another level of indirection.
**Address Space Layout Randomization (ASLR)** — Understanding relocations explains why ASLR works and where it needs GOT/PLT support. ASLR loads executables at random base addresses. For PIC code with PC-relative references, this works transparently — relative distances don't change. For non-PIC code with absolute addresses, the loader must apply relocations at load time, which is slow and prevents sharing code pages between processes. Modern systems prefer PIC for security and efficiency.
**Link-Time Optimization (LTO)** — LTO moves optimization to link time precisely because final addresses enable better decisions. With LTO, the linker can inline functions across translation units (no `call` needed), reorder code for better cache locality (knowing final addresses), and eliminate dead code globally. The relocation entries serve as a "to-do list" for these optimizations — the linker can rewrite code, not just patch addresses.
**JIT Compilation (cross-domain, compilers)** — JITs perform relocation-like patching when emitting code that references runtime addresses. A JIT compiler for a dynamic language emits native code with "holes" for object addresses, then patches them when objects are allocated. The same formula (symbol + addend - site) appears in V8's code generation, LuaJIT's trace compilation, and the JVM's template interpreter. Understanding linker relocations is directly applicable to implementing a JIT.
[[CRITERIA_JSON: {"milestone_id": "build-linker-m3", "criteria": ["Parse .rela and .rel sections from input object files, extracting r_offset, r_info (symbol index and type), and r_addend for each relocation entry", "Implement relocation type dispatcher handling at minimum R_X86_64_64 (64-bit absolute) and R_X86_64_PC32 (32-bit PC-relative) relocation types", "Translate relocation site offsets from input section coordinates to output section coordinates using the section mapping table from Milestone 1", "Look up target symbols by index in the input file's symbol table, then resolve to final virtual addresses via the global symbol table from Milestone 2", "Handle section symbols (STT_SECTION) by resolving to their output section's virtual address plus symbol offset", "Compute R_X86_64_64 patch values as: symbol_address + addend (direct 64-bit absolute address)", "Compute R_X86_64_PC32 patch values as: symbol_address + addend - relocation_site_address (PC-relative 32-bit signed offset)", "Detect and report overflow errors for 32-bit relocations when computed values exceed INT32_MIN to INT32_MAX range", "Write computed relocation values to the output buffer at the correct file offset with proper little-endian byte order", "Handle weak undefined symbols by resolving their relocations to address 0 without error", "Process relocations only after symbol resolution is complete (dependency ordering)", "Support both RELA format (explicit addend) and REL format (implicit addend read from relocation site)"]}]
<!-- END_MS -->


<!-- MS_ID: build-linker-m4 -->
<!-- MS_ID: build-linker-m4 -->
# Executable Generation: Making It Run
You've parsed object files. You've merged sections. You've resolved symbols and patched relocations. Every byte of code and data now has a final virtual address, and every cross-reference has been fixed up. But here's the problem: you have a blob of bytes, not a runnable program.
The operating system loader doesn't know about your sections, your symbols, or your careful address calculations. It speaks one language: **ELF executable format**. Your job now is to package everything into a structure the loader understands, set the entry point to where execution should begin, and write bytes to disk that Linux will actually run.
This milestone is about the final transformation: from linker's internal representation to a valid, loadable, executable ELF file. Get one header field wrong, and the loader rejects your file. Get the segment layout wrong, and the program crashes before the first instruction. Get the entry point wrong, and execution starts in random memory.
## The Tension: Internal Order vs Loader Expectations
Here's what you probably think happens: the linker outputs code, the OS loads it, execution begins at `main()`. But every part of that sentence is wrong:
- The linker doesn't "output code" — it outputs an **ELF file** with specific structure
- The OS doesn't "load it" — it **parses program headers** and **mmaps segments** into memory
- Execution doesn't begin at `main()` — it begins at **`_start`**, which the C runtime provides


The fundamental tension: **the linker's internal organization (sections) is completely different from what the loader needs (segments)**. Sections are the compiler/linker's way of grouping related content: all code here, all read-only data there, all writable data somewhere else. Segments are the loader's way of describing memory regions: this range of file bytes should be mapped to this virtual address with these permissions.
You've spent three milestones thinking in sections. Now you must think in segments.
> **🔑 Foundation: Program headers vs section headers (segments vs sections)**
> 
> ## What It Is
> ELF files have two parallel systems for describing their contents:
> 
> **Section Headers** (for the linker): Describe the file's logical organization. Each section has a name (`.text`, `.data`, `.rodata`), a type, flags, and file offset. Section headers live at the end of the file and are optional for execution — stripped executables don't have them at all.
> 
> **Program Headers** (for the loader): Describe the file's memory image. Each program header (also called a "segment") describes a contiguous range of file bytes that should be loaded into memory at a specific virtual address with specific permissions. Program headers live near the beginning of the file and are **mandatory** for executables.
> 
> ```c
> // Section header: linker's view
> typedef struct {
>     uint32_t sh_name;      // Section name (string table index)
>     uint32_t sh_type;      // SHT_PROGBITS, SHT_NOBITS, etc.
>     uint64_t sh_flags;     // SHF_ALLOC, SHF_WRITE, SHF_EXECINSTR
>     uint64_t sh_addr;      // Virtual address
>     uint64_t sh_offset;    // File offset
>     uint64_t sh_size;      // Size in bytes
>     // ... more fields
> } Elf64_Shdr;
> 
> // Program header: loader's view
> typedef struct {
>     uint32_t p_type;       // PT_LOAD, PT_DYNAMIC, PT_INTERP, etc.
>     uint32_t p_flags;      // PF_R, PF_W, PF_X (permissions)
>     uint64_t p_offset;     // File offset
>     uint64_t p_vaddr;      // Virtual address to map at
>     uint64_t p_paddr;      // Physical address (usually same as vaddr)
>     uint64_t p_filesz;     // Size in file
>     uint64_t p_memsz;      // Size in memory (can be larger for .bss)
>     uint64_t p_align;      // Alignment requirement
> } Elf64_Phdr;
> ```
> 
> ## Why You Need This Right Now
> You've spent three milestones building section-based data structures. Now you must translate them to segment-based program headers. The loader will **only** look at program headers — it doesn't know or care about sections. If your program headers are wrong, your executable won't load, regardless of how correct your section merging was.
> 
> ## Key Mental Model
> **Sections are logical divisions; segments are memory mappings.**
> 
> Think of sections as "what's in the file" (organized by purpose: code, data, symbols) and segments as "what goes in memory" (organized by load behavior: this chunk maps here, that chunk maps there).
> 
> Multiple sections can be packed into one segment. For example, a typical `PT_LOAD` segment might contain `.text`, `.rodata`, and `.eh_frame` sections — they're all read-only and can share a memory mapping. The loader doesn't see the section boundaries; it just maps the whole segment.
> 
> ```
> Linker's View (Sections):        Loader's View (Segments):
> ┌─────────────────────┐         ┌─────────────────────────┐
> │ .text (code)        │         │ PT_LOAD (PF_R | PF_X)   │
> ├─────────────────────┤    →    │   maps .text + .rodata  │
> │ .rodata (strings)   │         │   to vaddr 0x401000     │
> ├─────────────────────┤         ├─────────────────────────┤
> │ .data (writable)    │         │ PT_LOAD (PF_R | PF_W)   │
> ├─────────────────────┤    →    │   maps .data + .bss     │
> │ .bss (uninitialized)│         │   to vaddr 0x403000     │
> └─────────────────────┘         └─────────────────────────┘
> ```

![Sections to Segments Mapping](./diagrams/tdd-diag-042.svg)

![Sections vs Segments: Dual View](./diagrams/diag-sections-vs-segments.svg)

The output of this milestone is a **valid ELF executable** that:
1. Has a correct ELF header identifying it as x86-64 executable
2. Has program headers describing all loadable segments
3. Has the entry point pointing to `_start` (or `main` if no `_start`)
4. Has proper page-aligned segments for mmap compatibility
5. **Actually runs on Linux**
## What Makes an ELF File "Executable"?
An ELF file can be several types:
- `ET_REL` (1): Relocatable object file (`.o`) — compiler output, not runnable
- `ET_EXEC` (2): Executable file — static, fixed addresses
- `ET_DYN` (3): Shared object (`.so`) or PIE executable — position-independent
- `ET_CORE` (4): Core dump — crash snapshot, not runnable
Your linker produces `ET_EXEC` (type 2). This tells the loader: "this file has fixed virtual addresses — map them exactly where requested."

![ELF Executable File Structure](./diagrams/diag-elf-executable-structure.svg)

### The ELF Header: The File's Identity Card
The ELF header sits at byte 0 of the file. It answers the loader's first questions:
```c
#define EI_NIDENT   16
typedef struct {
    uint8_t  e_ident[EI_NIDENT];  // Magic number and identification
    uint16_t e_type;              // File type (ET_EXEC = 2)
    uint16_t e_machine;           // Architecture (EM_X86_64 = 62)
    uint32_t e_version;           // ELF version (always 1)
    uint64_t e_entry;             // Entry point virtual address
    uint64_t e_phoff;             // Program header table file offset
    uint64_t e_shoff;             // Section header table file offset
    uint32_t e_flags;             // Processor-specific flags
    uint16_t e_ehsize;            // ELF header size (64 bytes)
    uint16_t e_phentsize;         // Program header entry size (56 bytes)
    uint16_t e_phnum;             // Number of program headers
    uint16_t e_shentsize;         // Section header entry size (64 bytes)
    uint16_t e_shnum;             // Number of section headers
    uint16_t e_shstrndx;          // Section name string table index
} Elf64_Ehdr;
// e_ident breakdown:
#define EI_MAG0     0   // 0x7f
#define EI_MAG1     1   // 'E'
#define EI_MAG2     2   // 'L'
#define EI_MAG3     3   // 'F'
#define EI_CLASS    4   // ELFCLASS64 = 2
#define EI_DATA     5   // ELFDATA2LSB = 1 (little-endian)
#define EI_VERSION  6   // 1
#define EI_OSABI    7   // ELFOSABI_NONE = 0 or ELFOSABI_LINUX = 3
#define EI_ABIVERSION 8 // ABI version (usually 0)
// Bytes 9-15: padding (zeros)
```
The `e_ident` array is special: it's designed to be machine-independent, so the loader can read it without knowing the target architecture. The magic bytes `0x7f 'E' 'L' 'F'` let the loader quickly verify this is an ELF file.

![ELF Header Key Fields](./diagrams/diag-elf-header-fields.svg)

### The Entry Point: Where Execution Begins
> **🔑 Revelation: The Entry Point Secret**
> 
> Developers think execution starts at `main()`. In reality, `main()` is just a function called by the C runtime. The true entry point — the address in `e_entry` — is `_start`.
> 
> When the loader finishes mapping your executable, it jumps to `e_entry`. At that moment:
> - The stack contains `argc`, `argv`, and environment variables
> - No initialization has happened (globals aren't constructed, stdio isn't set up)
> - `%rsp` points to the top of the stack
> 
> The `_start` function (provided by libc in a typical build, or written by you for freestanding code) does:
> 1. Set up the frame pointer and other ABI requirements
> 2. Initialize libc (thread-local storage, locale, etc.)
> 3. Call `__libc_csu_init` (runs C++ global constructors)
> 4. Call `main(argc, argv, envp)`
> 5. Call `exit()` with main's return value
> 
> If you're building a static executable **without libc**, you must provide your own `_start`:
> 
> ```assembly
> # Minimal _start for freestanding executable
> .global _start
> _start:
>     xor %rbp, %rbp        # Clear frame pointer
>     mov %rsp, %rdi        # argc in first argument
>     lea 8(%rsp), %rsi     # argv in second argument  
>     call main             # Call main(argc, argv)
>     mov %eax, %edi        # Return value to exit code
>     mov $60, %eax         # syscall number for exit
>     syscall               # exit(return_value)
> ```
The entry point must be set correctly. If you point to the wrong address:
- Point to 0: crash immediately (jump to null)
- Point to middle of data: crash (execute garbage)
- Point to wrong function: wrong behavior
For this project, your linker should:
1. Look for a symbol named `_start` in the global symbol table
2. If found, set `e_entry` to `_start`'s address
3. If not found, look for `main` (for simple test programs)
4. If neither exists, error

![Entry Point Resolution](./diagrams/diag-entry-point-resolution.svg)

## Program Headers: The Loader's Map
The program header table is an array of `Elf64_Phdr` structures, each describing one segment. For a basic executable, you need:
| Type | Purpose | Content |
|------|---------|---------|
| `PT_LOAD` (1) | Loadable segment | Code or data to mmap |
| `PT_GNU_STACK` | Stack permissions | Whether stack is executable |
| `PT_GNU_RELRO` | Relocation read-only | Mark GOT as read-only after init |
For this project, `PT_LOAD` segments are the critical ones. Each `PT_LOAD` tells the loader: "map `p_filesz` bytes from file offset `p_offset` to virtual address `p_vaddr`, then zero-fill `p_memsz - p_filesz` additional bytes."
```c
#define PT_NULL     0   // Unused entry
#define PT_LOAD     1   // Loadable segment
#define PT_DYNAMIC  2   // Dynamic linking info
#define PT_INTERP   3   // Interpreter path
#define PT_NOTE     4   // Auxiliary info
#define PT_PHDR     6   // Program header itself
#define PT_GNU_STACK 0x6474e551
#define PT_GNU_RELRO 0x6474e552
#define PF_X        0x1  // Execute
#define PF_W        0x2  // Write
#define PF_R        0x4  // Read
```

![Program Header Entry Layout](./diagrams/diag-program-header-structure.svg)

### The Two-Segment Layout
A minimal executable needs two `PT_LOAD` segments:
**Segment 1: Text (read + execute)**
- Contains: `.text`, `.rodata`, `.eh_frame`, and any other read-only sections
- Permissions: `PF_R | PF_X`
- Typically mapped at a low virtual address (e.g., `0x401000`)
**Segment 2: Data (read + write)**
- Contains: `.data`, `.bss`
- Permissions: `PF_R | PF_W`
- Mapped at a higher virtual address (e.g., `0x403000`)
The separation isn't just organizational — it's a security requirement. Modern systems enforce **W^X** (write XOR execute): memory pages cannot be both writable and executable. If you tried to create a single segment with `PF_R | PF_W | PF_X`, the loader would reject it (or silently drop the execute permission).

![Segment Permissions Layout](./diagrams/diag-segment-permissions.svg)

### Page Alignment: Why 4096 Bytes Matters
> **🔑 Revelation: The Page Alignment Requirement**
> 
> Program headers specify alignment (`p_align`), typically 4096 (0x1000) — the page size on x86-64. This isn't a suggestion; it's a hard requirement.
> 
> The loader uses `mmap()` to map segments into memory. `mmap()` works at page granularity: it maps entire 4096-byte pages. If your segment starts at file offset 0x1234 and you ask to map it to vaddr 0x401234, the loader must:
> 1. Round the file offset **down** to the nearest page boundary (0x1000)
> 2. Round the virtual address **down** to the nearest page boundary (0x401000)
> 3. Map the page-aligned range
> 
> This means:
> - `p_offset % p_align` must equal `p_vaddr % p_align`
> - Both file offset and virtual address must have the same offset within a page
> 
> If this constraint is violated, the loader will refuse to load your executable with "Exec format error."
> 
> In practice, you ensure this by:
> - Making `p_align = 0x1000` (page size)
> - Starting segments at file offsets that are multiples of 0x1000
> - Starting segments at virtual addresses that are multiples of 0x1000

![Page Alignment for PT_LOAD Segments](./diagrams/diag-page-alignment-requirement.svg)

### Memory Size vs File Size
The `p_memsz` field can be larger than `p_filesz`. This handles `.bss`:
```c
// Segment containing .data and .bss:
p_offset = 0x3000;     // Where .data starts in file
p_vaddr = 0x403000;    // Virtual address
p_filesz = 0x100;      // .data is 256 bytes in file
p_memsz = 0x1000;      // .bss adds more memory (not in file)
p_align = 0x1000;      // Page-aligned
```
The loader will:
1. Map 256 bytes from file offset 0x3000 to vaddr 0x403000
2. Zero-fill the remaining `0x1000 - 0x100 = 0xF00` bytes (the `.bss` area)
This is why `.bss` doesn't occupy file space — the loader creates it by zero-filling.

![Segment to Memory Mapping](./diagrams/diag-segment-memory-mapping.svg)

## Building the ELF Header
Let's implement the ELF header construction:
```c
#define ELF_MAGIC "\x7fELF"
#define ELFCLASS64  2
#define ELFDATA2LSB 1
#define ET_EXEC     2
#define EM_X86_64   62
int write_elf_header(LinkerContext *ctx, FILE *out) {
    Elf64_Ehdr ehdr;
    memset(&ehdr, 0, sizeof(ehdr));
    // Magic number
    ehdr.e_ident[0] = 0x7f;
    ehdr.e_ident[1] = 'E';
    ehdr.e_ident[2] = 'L';
    ehdr.e_ident[3] = 'F';
    // Class: 64-bit
    ehdr.e_ident[4] = ELFCLASS64;
    // Data: little-endian
    ehdr.e_ident[5] = ELFDATA2LSB;
    // Version: 1
    ehdr.e_ident[6] = 1;
    // OS/ABI: Linux (or System V, which is 0)
    ehdr.e_ident[7] = 0;  // ELFOSABI_NONE
    // File type: executable
    ehdr.e_type = ET_EXEC;
    // Machine: x86-64
    ehdr.e_machine = EM_X86_64;
    // Version
    ehdr.e_version = 1;
    // Entry point: find _start or main
    GlobalSymbol *entry_sym = lookup_symbol(&ctx->global_syms, "_start");
    if (!entry_sym || entry_sym->state == SYM_UNDEF) {
        entry_sym = lookup_symbol(&ctx->global_syms, "main");
    }
    if (!entry_sym || entry_sym->state == SYM_UNDEF) {
        fprintf(stderr, "error: no entry point symbol (_start or main)\n");
        return -1;
    }
    ehdr.e_entry = entry_sym->final_address;
    // Program header offset: right after ELF header
    ehdr.e_phoff = sizeof(Elf64_Ehdr);
    // Section header offset: we'll calculate this after writing segments
    // For now, set to 0 (we can omit section headers for minimal executable)
    ehdr.e_shoff = 0;
    // Flags: none for x86-64
    ehdr.e_flags = 0;
    // Header sizes
    ehdr.e_ehsize = sizeof(Elf64_Ehdr);
    ehdr.e_phentsize = sizeof(Elf64_Phdr);
    ehdr.e_phnum = ctx->segment_count;  // We'll calculate this
    ehdr.e_shentsize = sizeof(Elf64_Shdr);
    ehdr.e_shnum = 0;  // No section headers (optional for execution)
    ehdr.e_shstrndx = 0;
    // Write header
    if (fwrite(&ehdr, sizeof(ehdr), 1, out) != 1) {
        perror("fwrite elf header");
        return -1;
    }
    ctx->entry_point = ehdr.e_entry;
    printf("Entry point: 0x%lx (%s)\n", ehdr.e_entry, 
           entry_sym ? entry_sym->name : "unknown");
    return 0;
}
```
## Building Program Headers
Now we need to create `PT_LOAD` segments from our output sections:
```c
typedef struct {
    uint32_t p_type;
    uint32_t p_flags;
    uint64_t p_offset;
    uint64_t p_vaddr;
    uint64_t p_paddr;
    uint64_t p_filesz;
    uint64_t p_memsz;
    uint64_t p_align;
} OutputSegment;
int build_segments(LinkerContext *ctx) {
    // We'll create two segments: text (RX) and data (RW)
    ctx->segments = calloc(2, sizeof(OutputSegment));
    ctx->segment_count = 2;
    OutputSegment *text_seg = &ctx->segments[0];
    OutputSegment *data_seg = &ctx->segments[1];
    // === TEXT SEGMENT (RX) ===
    // Contains: .text, .rodata (anything read-only, executable or not)
    text_seg->p_type = PT_LOAD;
    text_seg->p_flags = PF_R | PF_X;
    text_seg->p_align = 0x1000;  // Page-aligned
    // Find lowest address and total size for text segment
    uint64_t text_vaddr_start = UINT64_MAX;
    uint64_t text_vaddr_end = 0;
    uint64_t text_file_start = UINT64_MAX;
    uint64_t text_file_end = 0;
    for (size_t i = 0; i < ctx->output_count; i++) {
        OutputSection *sec = &ctx->outputs[i];
        // Include in text segment if: allocatable, not writable
        if ((sec->flags & SHF_ALLOC) && !(sec->flags & SHF_WRITE)) {
            if (sec->virtual_addr < text_vaddr_start) {
                text_vaddr_start = sec->virtual_addr;
                text_file_start = sec->file_offset;
            }
            // Extend to end of this section
            uint64_t sec_end_vaddr = sec->virtual_addr + sec->mem_size;
            uint64_t sec_end_file = sec->file_offset + sec->file_size;
            if (sec_end_vaddr > text_vaddr_end) {
                text_vaddr_end = sec_end_vaddr;
            }
            if (sec_end_file > text_file_end) {
                text_file_end = sec_end_file;
            }
        }
    }
    if (text_vaddr_start != UINT64_MAX) {
        text_seg->p_vaddr = text_vaddr_start;
        text_seg->p_paddr = text_vaddr_start;  // Physical = virtual for us
        text_seg->p_offset = text_file_start;
        text_seg->p_filesz = text_file_end - text_file_start;
        text_seg->p_memsz = text_vaddr_end - text_vaddr_start;
    }
    // === DATA SEGMENT (RW) ===
    // Contains: .data, .bss
    data_seg->p_type = PT_LOAD;
    data_seg->p_flags = PF_R | PF_W;
    data_seg->p_align = 0x1000;
    uint64_t data_vaddr_start = UINT64_MAX;
    uint64_t data_vaddr_end = 0;
    uint64_t data_file_start = UINT64_MAX;
    uint64_t data_file_end = 0;
    for (size_t i = 0; i < ctx->output_count; i++) {
        OutputSection *sec = &ctx->outputs[i];
        // Include in data segment if: allocatable, writable
        if ((sec->flags & SHF_ALLOC) && (sec->flags & SHF_WRITE)) {
            if (sec->virtual_addr < data_vaddr_start) {
                data_vaddr_start = sec->virtual_addr;
                data_file_start = sec->file_offset;
            }
            uint64_t sec_end_vaddr = sec->virtual_addr + sec->mem_size;
            uint64_t sec_end_file = sec->file_offset + sec->file_size;
            if (sec_end_vaddr > data_vaddr_end) {
                data_vaddr_end = sec_end_vaddr;
            }
            if (sec_end_file > data_file_end) {
                data_file_end = sec_end_file;
            }
        }
    }
    if (data_vaddr_start != UINT64_MAX) {
        data_seg->p_vaddr = data_vaddr_start;
        data_seg->p_paddr = data_vaddr_start;
        data_seg->p_offset = data_file_start;
        data_seg->p_filesz = data_file_end - data_file_start;
        data_seg->p_memsz = data_vaddr_end - data_vaddr_start;
    } else {
        // No data sections — remove this segment
        ctx->segment_count = 1;
    }
    return 0;
}
```
### Validating Segment Alignment
Before writing, verify the alignment constraint:
```c
int validate_segment_alignment(LinkerContext *ctx) {
    for (size_t i = 0; i < ctx->segment_count; i++) {
        OutputSegment *seg = &ctx->segments[i];
        if (seg->p_type != PT_LOAD) continue;
        // Check: p_offset % p_align == p_vaddr % p_align
        uint64_t offset_mod = seg->p_offset % seg->p_align;
        uint64_t vaddr_mod = seg->p_vaddr % seg->p_align;
        if (offset_mod != vaddr_mod) {
            fprintf(stderr, "error: segment %zu alignment mismatch\n", i);
            fprintf(stderr, "  p_offset = 0x%lx (mod 0x%lx = 0x%lx)\n",
                    seg->p_offset, seg->p_align, offset_mod);
            fprintf(stderr, "  p_vaddr = 0x%lx (mod 0x%lx = 0x%lx)\n",
                    seg->p_vaddr, seg->p_align, vaddr_mod);
            return -1;
        }
    }
    return 0;
}
```
### Writing Program Headers
```c
int write_program_headers(LinkerContext *ctx, FILE *out) {
    for (size_t i = 0; i < ctx->segment_count; i++) {
        OutputSegment *seg = &ctx->segments[i];
        Elf64_Phdr phdr;
        memset(&phdr, 0, sizeof(phdr));
        phdr.p_type = seg->p_type;
        phdr.p_flags = seg->p_flags;
        phdr.p_offset = seg->p_offset;
        phdr.p_vaddr = seg->p_vaddr;
        phdr.p_paddr = seg->p_paddr;
        phdr.p_filesz = seg->p_filesz;
        phdr.p_memsz = seg->p_memsz;
        phdr.p_align = seg->p_align;
        if (fwrite(&phdr, sizeof(phdr), 1, out) != 1) {
            perror("fwrite program header");
            return -1;
        }
        printf("Segment %zu: type=%d flags=0x%x offset=0x%lx vaddr=0x%lx "
               "filesz=0x%lx memsz=0x%lx\n",
               i, phdr.p_type, phdr.p_flags, phdr.p_offset, phdr.p_vaddr,
               phdr.p_filesz, phdr.p_memsz);
    }
    return 0;
}
```
## Laying Out the File
Now we need to assign file offsets that satisfy the alignment requirements. The key insight: **virtual addresses determine file layout** because of the alignment constraint.
```c
int layout_executable(LinkerContext *ctx) {
    // File layout:
    // [0x0000] ELF header (64 bytes)
    // [0x0040] Program headers (56 bytes each)
    // [0x0040 + phnum*56] Padding to next page boundary
    // [0x1000] Text segment content (page-aligned)
    // [...]   Data segment content (page-aligned)
    size_t header_size = sizeof(Elf64_Ehdr) + 
                         ctx->segment_count * sizeof(Elf64_Phdr);
    // Round up to page boundary for first segment
    uint64_t first_segment_file_offset = 0x1000;
    // Calculate virtual address base
    // Text segment starts at a high virtual address (typical: 0x400000)
    // But we need the file content at offset 0x1000 to match
    // So: vaddr = base + (file_offset - header_offset_in_file)
    // Actually, let's reconsider. The constraint is:
    // p_offset % p_align == p_vaddr % p_align
    // 
    // If p_align = 0x1000 and p_offset = 0x1000:
    // Then p_vaddr must be 0x????1000 (any value where low 12 bits are 0x1000)
    // But we want p_vaddr = 0x401000 (page-aligned, low 12 bits = 0)
    // So p_offset must also have low 12 bits = 0
    // 
    // Solution: p_offset = 0x1000, p_vaddr = 0x401000
    // 0x1000 % 0x1000 = 0
    // 0x401000 % 0x1000 = 0
    // ✓ Matches!
    // Assign file offsets and virtual addresses to sections
    uint64_t current_file_offset = first_segment_file_offset;
    uint64_t current_vaddr = 0x401000;  // Base virtual address
    // Process sections in order: text, rodata, data, bss
    const char *section_order[] = {".text", ".rodata", ".data", ".bss"};
    int num_ordered = 4;
    for (int i = 0; i < num_ordered; i++) {
        const char *name = section_order[i];
        OutputSection *sec = find_output_section(ctx, name);
        if (!sec || sec->mem_size == 0) continue;
        // Align virtual address
        current_vaddr = align_up(current_vaddr, sec->align);
        // For non-.bss, align file offset
        if (sec->file_size > 0) {
            current_file_offset = align_up(current_file_offset, sec->align);
            sec->file_offset = current_file_offset;
        }
        sec->virtual_addr = current_vaddr;
        current_vaddr += sec->mem_size;
        current_file_offset += sec->file_size;
    }
    // Now build segments from the laid-out sections
    build_segments(ctx);
    // Validate alignment
    if (validate_segment_alignment(ctx) != 0) {
        return -1;
    }
    return 0;
}
```

![Segment Building Algorithm](./diagrams/tdd-diag-046.svg)

![Complete Executable Walkthrough](./diagrams/diag-complete-executable-example.svg)

## Writing the Complete Executable
Now we can write everything to the output file:
```c
int write_executable(LinkerContext *ctx, const char *output_path) {
    FILE *out = fopen(output_path, "wb");
    if (!out) {
        perror("fopen output");
        return -1;
    }
    // Write ELF header
    if (write_elf_header(ctx, out) != 0) {
        fclose(out);
        return -1;
    }
    // Write program headers
    if (write_program_headers(ctx, out) != 0) {
        fclose(out);
        return -1;
    }
    // Pad to first segment
    long current_pos = ftell(out);
    if (current_pos < 0x1000) {
        size_t padding = 0x1000 - current_pos;
        uint8_t *zeros = calloc(padding, 1);
        fwrite(zeros, 1, padding, out);
        free(zeros);
    }
    // Write segment content
    for (size_t i = 0; i < ctx->segment_count; i++) {
        OutputSegment *seg = &ctx->segments[i];
        if (seg->p_type != PT_LOAD) continue;
        if (seg->p_filesz == 0) continue;
        // Seek to segment's file offset
        fseek(out, seg->p_offset, SEEK_SET);
        // Write the segment content from output buffer
        // The output buffer was built in milestone 3
        fwrite(ctx->output_buffer + seg->p_offset, 1, seg->p_filesz, out);
    }
    fclose(out);
    // Make executable
    chmod(output_path, 0755);
    printf("Wrote executable: %s (%ld bytes)\n", output_path, ftell(out));
    return 0;
}
```
## A Complete Example: From Object Files to Running Program
Let's trace through the complete linking process for a minimal program:
**Source files:**
```c
// start.s - our own _start
.global _start
_start:
    xor %rbp, %rbp
    mov %rsp, %rdi       # argc
    lea 8(%rsp), %rsi    # argv
    call main
    mov %eax, %edi
    mov $60, %eax        # exit syscall
    syscall
// main.c
int main(int argc, char **argv) {
    return 42;
}
```
**After assembly and compilation:**
```bash
as -o start.o start.s
gcc -c -fno-builtin main.c -o main.o
```
**Linking process:**
1. **Parse object files:**
   - `start.o`: `.text` (32 bytes), symbol `_start` at `.text+0`
   - `main.o`: `.text` (24 bytes), symbol `main` at `.text+0`
2. **Merge sections:**
   - Output `.text`: 32 + 24 = 56 bytes
   - `start.o/.text` at output offset 0
   - `main.o/.text` at output offset 32
3. **Resolve symbols:**
   - `_start`: output `.text+0` → vaddr 0x401000
   - `main`: output `.text+32` → vaddr 0x401020
4. **Process relocations:**
   - `start.o` has `call main` at offset 12 (R_X86_64_PC32)
   - Patch site: 0x40100C, target: 0x401020
   - Computed: 0x401020 - 0x401010 = 0x10
5. **Layout executable:**
   - ELF header at offset 0 (64 bytes)
   - Program header at offset 64 (56 bytes)
   - Padding to 0x1000 (4032 bytes)
   - `.text` content at offset 0x1000 (56 bytes)
6. **Write ELF header:**
   - `e_entry = 0x401000` (address of `_start`)
   - `e_phoff = 64`
   - `e_phnum = 1`
7. **Write program header:**
   - `p_type = PT_LOAD`
   - `p_flags = PF_R | PF_X`
   - `p_offset = 0x1000`
   - `p_vaddr = 0x401000`
   - `p_filesz = 56`
   - `p_memsz = 56`
   - `p_align = 0x1000`
**Result:**
```bash
$ ./mylinker start.o main.o -o program
$ ./program; echo $?
42
```
The program runs and exits with code 42. Success!
## Testing Your Executable Generator
### Test 1: Minimal Freestanding Program
```bash
# Create minimal _start that just exits
cat > minimal.s << 'EOF'
.global _start
_start:
    mov $42, %edi    # exit code
    mov $60, %eax    # syscall: exit
    syscall
EOF
as -o minimal.o minimal.s
./mylinker minimal.o -o minimal
./minimal; echo $?
# Expected output: 42
```
### Test 2: Program with Data
```bash
cat > data_test.s << 'EOF'
.global _start
_start:
    mov value, %edi   # Load value from .data
    mov $60, %eax
    syscall
.section .data
value:
    .int 123
EOF
as -o data_test.o data_test.s
./mylinker data_test.o -o data_test
./data_test; echo $?
# Expected output: 123
```
### Test 3: Program with BSS
```bash
cat > bss_test.s << 'EOF'
.global _start
_start:
    movl $99, buffer    # Write to .bss
    mov buffer, %edi    # Read back
    mov $60, %eax
    syscall
.section .bss
buffer:
    .skip 4
EOF
as -o bss_test.o bss_test.s
./mylinker bss_test.o -o bss_test
./bss_test; echo $?
# Expected output: 99
```
### Test 4: Multi-File Program
```bash
cat > add.s << 'EOF'
.global add
add:
    lea (%rdi,%rsi,1), %eax
    ret
EOF
cat > main_add.s << 'EOF'
.global _start
_start:
    mov $10, %edi
    mov $32, %rsi
    call add
    mov %eax, %edi
    mov $60, %eax
    syscall
EOF
as -o add.o add.s
as -o main_add.o main_add.s
./mylinker add.o main_add.o -o add_test
./add_test; echo $?
# Expected output: 42
```
### Verifying with System Tools
```bash
# Check file type
file program
# Expected: ELF 64-bit LSB executable, x86-64
# Check program headers
readelf -l program
# Check entry point
readelf -h program | grep Entry
# Expected: Entry point address: 0x401000
# Disassemble
objdump -d program
# Run with strace to see syscalls
strace ./program
```
## Common Pitfalls
### Pitfall 1: Wrong Entry Point
If `e_entry` points to the wrong address, the program will crash or misbehave:
```c
// BAD: Hardcode entry point
ehdr.e_entry = 0x401000;  // What if _start is elsewhere?
// GOOD: Look up the symbol
GlobalSymbol *sym = lookup_symbol(&ctx->global_syms, "_start");
if (sym && sym->state == SYM_RESOLVED) {
    ehdr.e_entry = sym->final_address;
}
```
### Pitfall 2: Misaligned Segments
The loader requires `p_offset % p_align == p_vaddr % p_align`:
```c
// BAD: File offset doesn't match vaddr modulo alignment
seg->p_offset = 0x1234;
seg->p_vaddr = 0x401000;
seg->p_align = 0x1000;
// 0x1234 % 0x1000 = 0x234
// 0x401000 % 0x1000 = 0x0
// Mismatch! Loader will reject.
// GOOD: Both aligned to page boundary
seg->p_offset = 0x1000;
seg->p_vaddr = 0x401000;
seg->p_align = 0x1000;
```
### Pitfall 3: Forgetting .bss Size
`.bss` contributes to `p_memsz` but not `p_filesz`:
```c
// BAD: Only counting file size
seg->p_memsz = seg->p_filesz;  // .bss will be missing!
// GOOD: Include .bss in memory size
seg->p_memsz = data_section_mem_size;  // Includes .bss
seg->p_filesz = data_section_file_size;  // Excludes .bss
```
### Pitfall 4: Missing Executable Permission
If `.text` segment doesn't have `PF_X`, the code can't execute:
```c
// BAD: Read-only, not executable
text_seg->p_flags = PF_R;
// GOOD: Read and execute
text_seg->p_flags = PF_R | PF_X;
```
### Pitfall 5: Overlapping Segments
Segments must not overlap in virtual address space:
```c
// Validate no overlap
for (size_t i = 0; i < ctx->segment_count; i++) {
    for (size_t j = i + 1; j < ctx->segment_count; j++) {
        uint64_t start_i = ctx->segments[i].p_vaddr;
        uint64_t end_i = start_i + ctx->segments[i].p_memsz;
        uint64_t start_j = ctx->segments[j].p_vaddr;
        uint64_t end_j = start_j + ctx->segments[j].p_memsz;
        if (start_i < end_j && start_j < end_i) {
            fprintf(stderr, "error: segments %zu and %zu overlap\n", i, j);
            return -1;
        }
    }
}
```
## The Complete Linker Pipeline
Here's the full integration of all four milestones:
```c
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input1.o input2.o ... -o output\n", argv[0]);
        return 1;
    }
    LinkerContext ctx;
    init_linker_context(&ctx);
    // Parse input files
    int input_count = argc - 3;  // Subtract program name, -o, output
    ctx.inputs = calloc(input_count, sizeof(ObjectFile));
    ctx.input_count = input_count;
    for (int i = 0; i < input_count; i++) {
        printf("Parsing %s...\n", argv[i + 1]);
        if (parse_object_file(argv[i + 1], &ctx.inputs[i]) != 0) {
            return 1;
        }
        if (parse_symbols(&ctx.inputs[i]) != 0) {
            return 1;
        }
        if (parse_relocations(&ctx.inputs[i]) != 0) {
            return 1;
        }
    }
    // Milestone 1: Merge sections
    printf("Merging sections...\n");
    if (merge_all_sections(&ctx) != 0) {
        return 1;
    }
    if (merge_section_data(&ctx) != 0) {
        return 1;
    }
    // Assign virtual addresses (needed for symbol resolution)
    printf("Assigning addresses...\n");
    if (layout_executable(&ctx) != 0) {
        return 1;
    }
    // Milestone 2: Resolve symbols
    printf("Resolving symbols...\n");
    if (collect_symbols(&ctx) != 0) {
        return 1;
    }
    int undef = verify_all_symbols_defined(&ctx);
    if (undef > 0) {
        return 1;
    }
    if (assign_symbol_addresses(&ctx) != 0) {
        return 1;
    }
    // Build output buffer
    printf("Building output...\n");
    if (build_output_buffer(&ctx) != 0) {
        return 1;
    }
    // Milestone 3: Process relocations
    printf("Processing relocations...\n");
    if (process_relocations(&ctx) != 0) {
        return 1;
    }
    // Milestone 4: Write executable
    printf("Writing executable...\n");
    const char *output_path = argv[argc - 1];
    if (write_executable(&ctx, output_path) != 0) {
        return 1;
    }
    printf("Link successful: %s\n", output_path);
    return 0;
}
```
## Beyond the Basics: What Real Linkers Do
Your linker now produces working executables. But production linkers like GNU ld and lld do much more:
**Linker Scripts**: Custom layouts for embedded systems, security hardening, or special memory maps. A linker script can specify exact addresses for sections, create custom segments, and define symbol aliases.
**Static Libraries**: Archive files (`.a`) containing multiple object files. The linker only extracts objects that define undefined symbols, enabling "lazy" linking.
**Dynamic Linking**: Producing executables that reference shared libraries (`.so`). This requires `PT_INTERP` (specifying the dynamic loader), `PT_DYNAMIC` (dynamic linking info), and different relocation handling.
**Link-Time Optimization (LTO)**: Re-compiling at link time with whole-program visibility. Enables cross-module inlining, dead code elimination, and devirtualization.
**Exception Handling**: Generating `.eh_frame` sections with unwind information. Required for C++ exceptions and stack traces.
**Debug Information**: Merging DWARF debug info from input files, adjusting address ranges for the final layout.
**Thread-Local Storage**: Allocating and initializing TLS sections, generating the `__tls_get_addr` calls.
You now understand the foundation. These advanced features build on the same principles: sections, symbols, relocations, and segments.
---
## Knowledge Cascade
**Process Loading (cross-domain, OS)** — The `execve()` syscall is where your work meets the kernel. The kernel's binary handler reads your ELF header, validates the magic bytes, allocates virtual address space, `mmap()`s your `PT_LOAD` segments at their requested addresses, sets up the stack with `argc`/`argv`/`envp`, and finally jumps to `e_entry`. Every field you wrote — `e_entry`, `p_vaddr`, `p_flags` — is consumed by this code path. Understanding linking means understanding what the loader expects.
**Memory Protection (security)** — The segment permissions you set (`PF_R | PF_X` for text, `PF_R | PF_W` for data) are the foundation of exploit mitigation. W^X (write XOR execute) means no memory is both writable and executable, preventing code injection. NX bits (no-execute) in page tables enforce this at the hardware level. Modern hardening goes further: RELRO (relocation read-only) makes the GOT read-only after startup, preventing GOT overwrite attacks. Your segment layout directly enables these protections.
**Static vs Dynamic Executables** — Your linker produces static executables: everything needed to run is in one file. Dynamic executables are different: they have an interpreter (the dynamic linker, usually `/lib64/ld-linux-x86-64.so.2`) specified by `PT_INTERP`. The kernel loads both the executable and the interpreter; the interpreter then finds and loads required shared libraries, resolves symbols, and finally transfers control. Dynamic linking trades simplicity for flexibility: smaller executables, shared code pages, and updatable libraries, but with load-time overhead and ABI dependencies.
**ELF Injection (cross-domain, security)** — Understanding executable format enables both malware and analysis. An attacker with write access to an ELF file can inject code by: finding padding between sections, extending a segment, or adding a new `PT_LOAD` segment. The entry point is then redirected to the injected code. Defenders use the same knowledge to detect such modifications: unexpected segment counts, overlapping segments, or entry points outside `.text`. Tools like `readelf` and `objdump` are your first line of defense.
**Minimal Executables** — The ELF format is more flexible than most realize. It's possible to create a working executable under 100 bytes by: overlapping the ELF header with program headers, using the header itself as code, and exploiting the loader's leniency. The "ELFkickers" collection demonstrates techniques like overlapping segments and executing the header. While mostly a curiosity, understanding these edge cases deepens your grasp of what's truly required vs. what's convention.
[[CRITERIA_JSON: {"milestone_id": "build-linker-m4", "criteria": ["Generate valid ELF64 header with magic bytes 0x7f 'E' 'L' 'F', class ELFCLASS64, data ELFDATA2LSB, type ET_EXEC, and machine EM_X86_64", "Set e_entry field to the virtual address of _start symbol (or main if _start not found), reporting error if neither exists", "Write program header table at file offset specified by e_phoff, immediately following ELF header", "Create PT_LOAD segment for text/code sections with p_flags = PF_R | PF_X (readable and executable)", "Create PT_LOAD segment for data sections with p_flags = PF_R | PF_W (readable and writable)", "Ensure p_align is set to 0x1000 (page size) for all PT_LOAD segments", "Validate and enforce alignment constraint: p_offset % p_align == p_vaddr % p_align for each PT_LOAD segment", "Set p_memsz > p_filesz for data segment containing .bss, accounting for zero-filled uninitialized data", "Write segment content at correct file offsets matching p_offset values in program headers", "Apply executable file permissions (chmod 0755) to output file", "Produce executable that runs on Linux x86-64 without external tools, verifiable with ./executable", "Handle missing .bss section gracefully (single text segment executable)", "Report entry point symbol name and address during linking for verification"]}]
<!-- END_MS -->


## System Overview

![Virtual Address Space Layout](./diagrams/tdd-diag-050.svg)

![System Overview](./diagrams/system-overview.svg)


# TDD

A multi-file ELF linker that transforms position-independent object files into fixed-address executables. The linker performs four fundamental operations: section merging (combining code and data from multiple translation units with alignment handling), symbol resolution (connecting references to definitions across files with strong/weak rules), relocation processing (patching addresses using PC-relative and absolute formulas), and executable generation (producing valid ELF with program headers and entry point). The design emphasizes clear separation of concerns between parsing, merging, resolution, patching, and output phases, with comprehensive error reporting and a well-defined internal data model centered on the input-to-output mapping table.


<!-- TDD_MOD_ID: build-linker-m1 -->
# Section Merging: Technical Design Specification
## Module Charter
The Section Merging module is the foundational stage of the static linker, responsible for parsing multiple ELF64 relocatable object files and combining their sections into a unified memory layout. This module reads raw `.o` files, extracts section headers and content, groups sections by name across all input files, and concatenates them with proper alignment padding. It produces the **input-to-output mapping table** — a critical data structure that records where every byte from every input file ends up in the final output, enabling all subsequent relocation processing.
This module does NOT resolve symbols, patch addresses, or generate executable headers. Its sole responsibility is to establish the memory layout and provide the mapping infrastructure. Upstream, it depends only on the filesystem (reading `.o` files). Downstream, it feeds the Symbol Resolution and Relocation Processing modules with the mapping table they require. The module maintains strict invariants: every allocatable input section maps to exactly one output section, alignment constraints are never violated, and the mapping table is complete and consistent before any downstream processing begins.
## File Structure
```
linker/
├── 01_elf_types.h          # ELF64 structure definitions (header, section header)
├── 02_elf_parser.h         # Parser interface declarations
├── 03_elf_parser.c         # ELF header and section parsing implementation
├── 04_section_types.h      # Internal section representation types
├── 05_section_merger.h     # Merger interface declarations
├── 06_section_merger.c     # Section grouping and merging implementation
├── 07_mapping_table.h      # Input-to-output mapping data structures
├── 08_mapping_table.c      # Mapping table construction and lookup
├── 09_linker_context.h     # Global linker state
├── 10_section_merge_main.c # Test driver entry point
└── tests/
    ├── test_parser.c       # ELF parsing unit tests
    ├── test_merger.c       # Section merging unit tests
    └── fixtures/           # Test object files
        ├── simple.o        # Single .text section
        ├── multi.o         # .text, .data, .bss
        └── aligned.o       # Non-trivial alignment requirements
```
## Complete Data Model
### ELF64 Header Structure
The ELF64 header is exactly 64 bytes and appears at file offset 0:
```c
// File: 01_elf_types.h
#ifndef ELF_TYPES_H
#define ELF_TYPES_H
#include <stdint.h>
// ELF identification indices
#define EI_MAG0     0
#define EI_MAG1     1
#define EI_MAG2     2
#define EI_MAG3     3
#define EI_CLASS    4
#define EI_DATA     5
#define EI_VERSION  6
#define EI_OSABI    7
#define EI_NIDENT   16
// ELF magic bytes
#define ELFMAG0     0x7f
#define ELFMAG1     'E'
#define ELFMAG2     'L'
#define ELFMAG3     'F'
// ELF class
#define ELFCLASSNONE    0
#define ELFCLASS32      1
#define ELFCLASS64      2
// ELF data encoding
#define ELFDATANONE     0
#define ELFDATA2LSB     1   // Little-endian
#define ELFDATA2MSB     2   // Big-endian
// ELF file types
#define ET_NONE     0
#define ET_REL      1   // Relocatable
#define ET_EXEC     2   // Executable
#define ET_DYN      3   // Shared object
#define ET_CORE     4   // Core file
// ELF machine types
#define EM_NONE     0
#define EM_X86_64   62  // AMD x86-64
// Section header types
#define SHT_NULL        0
#define SHT_PROGBITS    1   // Program-defined content
#define SHT_SYMTAB      2   // Symbol table
#define SHT_STRTAB      3   // String table
#define SHT_RELA        4   // Relocation with addends
#define SHT_HASH        5   // Symbol hash table
#define SHT_DYNAMIC     6   // Dynamic linking info
#define SHT_NOTE        7   // Notes
#define SHT_NOBITS      8   // No file space (.bss)
#define SHT_REL         9   // Relocation without addends
// Section header flags
#define SHF_WRITE       0x00000001  // Writable
#define SHF_ALLOC       0x00000002  // Occupies memory
#define SHF_EXECINSTR   0x00000004  // Executable
#define SHF_MERGE       0x00000010  // Might be merged
#define SHF_STRINGS     0x00000020  // Contains null-terminated strings
#define SHF_INFO_LINK   0x00000040  // sh_info contains section index
#define SHF_LINK_ORDER  0x00000080  // Preserve ordering
#define SHF_TLS         0x00000400  // Thread-local storage
// Special section indices
#define SHN_UNDEF       0
#define SHN_LORESERVE   0xff00
#define SHN_ABS         0xfff1
#define SHN_COMMON      0xfff2
#define SHN_HIRESERVE   0xffff
// ELF64 Header - 64 bytes, packed, no padding between fields
typedef struct __attribute__((packed)) {
    uint8_t  e_ident[EI_NIDENT];    // +0x00: ELF identification
    uint16_t e_type;                 // +0x10: Object file type
    uint16_t e_machine;              // +0x12: Architecture
    uint32_t e_version;              // +0x14: Object file version
    uint64_t e_entry;                // +0x18: Entry point virtual address
    uint64_t e_phoff;                // +0x20: Program header table offset
    uint64_t e_shoff;                // +0x28: Section header table offset
    uint32_t e_flags;                // +0x30: Processor-specific flags
    uint16_t e_ehsize;               // +0x34: ELF header size (64)
    uint16_t e_phentsize;            // +0x36: Program header entry size
    uint16_t e_phnum;                // +0x38: Number of program headers
    uint16_t e_shentsize;            // +0x3A: Section header entry size (64)
    uint16_t e_shnum;                // +0x3C: Number of section headers
    uint16_t e_shstrndx;             // +0x3E: Section name string table index
} Elf64_Ehdr;  // Total: 64 bytes (0x40)
// Section Header - 64 bytes each
typedef struct __attribute__((packed)) {
    uint32_t sh_name;        // +0x00: Name (index into section string table)
    uint32_t sh_type;        // +0x04: Type (SHT_*)
    uint64_t sh_flags;       // +0x08: Flags (SHF_*)
    uint64_t sh_addr;        // +0x10: Virtual address (0 for .o files)
    uint64_t sh_offset;      // +0x18: File offset
    uint64_t sh_size;        // +0x20: Section size (bytes)
    uint32_t sh_link;        // +0x28: Link to another section
    uint32_t sh_info;        // +0x2C: Additional info
    uint64_t sh_addralign;   // +0x30: Alignment constraint
    uint64_t sh_entsize;     // +0x38: Entry size (if fixed-size entries)
} Elf64_Shdr;  // Total: 64 bytes (0x40)
#endif // ELF_TYPES_H
```
### Memory Layout of ELF64 Header
| Offset | Size | Field | Valid Values for x86-64 .o |
|--------|------|-------|---------------------------|
| 0x00 | 16 | e_ident[16] | [7f 45 4c 46 02 01 01 00 ...] |
| 0x10 | 2 | e_type | 1 (ET_REL) |
| 0x12 | 2 | e_machine | 62 (EM_X86_64) |
| 0x14 | 4 | e_version | 1 |
| 0x18 | 8 | e_entry | 0 (unused in relocatable) |
| 0x20 | 8 | e_phoff | 0 (no program headers) |
| 0x28 | 8 | e_shoff | Varies (typically near EOF) |
| 0x30 | 4 | e_flags | 0 |
| 0x34 | 2 | e_ehsize | 64 |
| 0x36 | 2 | e_phentsize | 0 |
| 0x38 | 2 | e_phnum | 0 |
| 0x3A | 2 | e_shentsize | 64 |
| 0x3C | 2 | e_shnum | Varies (≥3) |
| 0x3E | 2 | e_shstrndx | Index of .shstrtab |
### Internal Section Representation
```c
// File: 04_section_types.h
#ifndef SECTION_TYPES_H
#define SECTION_TYPES_H
#include <stdint.h>
#include <stddef.h>
#define MAX_SECTION_NAME    64
#define MAX_FILENAME        256
#define MAX_MAPPINGS        1024
#define MAX_OUTPUT_SECTIONS 16
#define MAX_INPUT_FILES     128
// Represents one section from an input object file
// This is our internal representation after parsing
typedef struct {
    char name[MAX_SECTION_NAME];    // Resolved section name (e.g., ".text")
    uint32_t type;                  // SHT_PROGBITS, SHT_NOBITS, etc.
    uint64_t flags;                 // SHF_* flags combined
    uint64_t align;                 // Alignment requirement (power of 2, or 0/1)
    uint64_t size;                  // Size in bytes
    uint64_t file_offset;           // Where in the input file this section starts
    uint8_t *data;                  // Pointer to section content (NULL for SHT_NOBITS)
    uint32_t original_index;        // Index in the input file's section header table
} InputSection;
// Represents one parsed object file
typedef struct {
    char filename[MAX_FILENAME];    // Source filename for error reporting
    Elf64_Ehdr ehdr;                // Parsed ELF header (for reference)
    InputSection *sections;         // Array of parsed sections
    uint16_t section_count;         // Number of sections
    uint16_t section_capacity;      // Allocated capacity
    char *shstrtab;                 // Section name string table (owned)
    size_t shstrtab_size;           // Size of string table
} ObjectFile;
// Tracks placement of one input section within output
// This is the critical mapping record for relocation processing
typedef struct {
    char source_file[MAX_FILENAME]; // Which input file
    char section_name[MAX_SECTION_NAME]; // Which section in that file
    uint64_t input_size;            // Size of input section
    uint64_t output_offset;         // Where in output section it starts
    uint64_t padding_before;        // Alignment padding bytes before this
    uint8_t *source_data;           // Pointer to input section data (not owned)
    uint32_t source_section_idx;    // Index in source ObjectFile
} SectionMapping;
// Represents one merged output section
typedef struct {
    char name[MAX_SECTION_NAME];    // Section name (e.g., ".text")
    uint64_t flags;                 // Combined flags from all inputs
    uint64_t align;                 // Maximum alignment of all inputs
    uint64_t file_size;             // Bytes in output file (excludes .bss)
    uint64_t mem_size;              // Bytes in memory (includes .bss)
    uint64_t file_offset;           // Where in output file (assigned later)
    uint64_t virtual_addr;          // Virtual address (assigned later)
    SectionMapping *mappings;       // Array of input section mappings
    size_t mapping_count;           // Number of mappings
    size_t mapping_capacity;        // Allocated capacity
    uint8_t *data;                  // Merged content buffer (NULL for .bss)
    int is_nobits;                  // 1 if this is a .bss-like section
} OutputSection;
// Hash table entry for mapping lookup
typedef struct {
    char key[MAX_FILENAME + MAX_SECTION_NAME + 2]; // "filename:section"
    SectionMapping *mapping;
    OutputSection *output;
    int occupied;                   // 1 if entry is in use
} MappingHashEntry;
// Input-to-output mapping table
// This is the core data structure for relocation processing
typedef struct {
    MappingHashEntry *entries;      // Hash table array
    size_t capacity;                // Hash table size
    size_t count;                   // Number of entries
} MappingTable;
// Global linker state for section merging phase
typedef struct {
    ObjectFile *inputs;             // Array of input object files
    size_t input_count;             // Number of input files
    size_t input_capacity;          // Allocated capacity
    OutputSection *outputs;         // Array of output sections
    size_t output_count;            // Number of output sections
    size_t output_capacity;         // Allocated capacity
    MappingTable mapping;           // Input-to-output mapping
    uint64_t base_address;          // Base virtual address (0x400000 typical)
    uint64_t total_file_size;       // Sum of all output file_size
    uint64_t total_mem_size;        // Sum of all output mem_size
} LinkerContext;
// Section categories for processing
typedef enum {
    SEC_CAT_IGNORE,     // Non-allocatable, not merged
    SEC_CAT_CODE,       // .text (SHF_ALLOC | SHF_EXECINSTR)
    SEC_CAT_RODATA,     // .rodata (SHF_ALLOC only)
    SEC_CAT_DATA,       // .data (SHF_ALLOC | SHF_WRITE)
    SEC_CAT_BSS,        // .bss (SHF_ALLOC | SHF_WRITE, SHT_NOBITS)
    SEC_CAT_OTHER       // Custom allocatable sections
} SectionCategory;
#endif // SECTION_TYPES_H
```

![Section Merging Module Architecture](./diagrams/tdd-diag-001.svg)

### Section Mapping Table Structure
The mapping table is the heart of the section merger. It enables O(1) lookup of output offsets:
```
Mapping Table (Hash: filename:section → MappingEntry)
┌─────────────────────────────────────────────────────────────┐
│ Hash Key: "main.o:.text"                                    │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ mapping.output_offset = 0x0000                          │ │
│ │ mapping.input_size = 0x80                               │ │
│ │ mapping.padding_before = 0                              │ │
│ │ output.name = ".text"                                   │ │
│ │ output.virtual_addr = 0x401000                          │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ Hash Key: "utils.o:.text"                                   │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ mapping.output_offset = 0x0080                          │ │
│ │ mapping.input_size = 0x40                               │ │
│ │ mapping.padding_before = 0                              │ │
│ │ output.name = ".text"                                   │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ Hash Key: "utils.o:.bss"                                    │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ mapping.output_offset = 0x0000                          │ │
│ │ mapping.input_size = 0x40                               │ │
│ │ mapping.padding_before = 0                              │ │
│ │ output.name = ".bss"                                    │ │
│ │ output.file_size = 0x00 (NOBITS!)                       │ │
│ │ output.mem_size = 0x40                                  │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```
## Interface Contracts
### ELF Parser Interface
```c
// File: 02_elf_parser.h
#ifndef ELF_PARSER_H
#define ELF_PARSER_H
#include "01_elf_types.h"
#include "04_section_types.h"
// Error codes from parser
typedef enum {
    PARSE_OK = 0,
    PARSE_ERR_FILE_NOT_FOUND,       // File does not exist
    PARSE_ERR_FILE_READ,            // Error reading file
    PARSE_ERR_INVALID_MAGIC,        // Magic bytes don't match ELF
    PARSE_ERR_INVALID_CLASS,        // Not ELFCLASS64
    PARSE_ERR_INVALID_ENDIAN,       // Not little-endian
    PARSE_ERR_INVALID_TYPE,         // Not ET_REL
    PARSE_ERR_INVALID_ARCH,         // Not EM_X86_64
    PARSE_ERR_INVALID_EHSIZE,       // e_ehsize != 64
    PARSE_ERR_INVALID_SHENTSIZE,    // e_shentsize != 64
    PARSE_ERR_NO_SECTIONS,          // e_shnum == 0
    PARSE_ERR_NO_SHSTRTAB,          // e_shstrndx == SHN_UNDEF
    PARSE_ERR_SHSTRTAB_MISSING,     // Section at e_shstrndx doesn't exist
    PARSE_ERR_SECTION_READ,         // Failed to read section header
    PARSE_ERR_DATA_READ,            // Failed to read section data
    PARSE_ERR_MEMORY,               // Memory allocation failed
    PARSE_ERR_ALIGNMENT_INVALID,    // sh_addralign not power of 2
    PARSE_ERR_SECTION_TOO_LARGE     // Section size > 1GB
} ParseError;
// Parse a single object file
// 
// Parameters:
//   filename - Path to .o file (must not be NULL, must exist)
//   obj      - Output structure to populate (must not be NULL)
//
// Returns:
//   PARSE_OK on success
//   Appropriate error code on failure
//
// On success, obj is fully initialized with:
//   - filename copied
//   - ehdr populated
//   - sections array allocated and populated
//   - shstrtab allocated and populated
//
// On failure, obj is in undefined state (caller should not use)
//
// Memory ownership:
//   - obj->sections is owned by caller, free with object_file_destroy()
//   - obj->shstrtab is owned by caller
//   - Each InputSection.data is owned by caller
ParseError parse_object_file(const char *filename, ObjectFile *obj);
// Get human-readable error message
// 
// Parameters:
//   err - Error code from parse_object_file
//
// Returns:
//   Static string describing the error (do not free)
const char *parse_error_string(ParseError err);
// Free resources associated with a parsed object file
// 
// Parameters:
//   obj - Object file to destroy (may be NULL)
//
// Safe to call on partially-initialized or failed parse results
void object_file_destroy(ObjectFile *obj);
// Validate an input section for merging
// 
// Parameters:
//   sec - Section to validate
//
// Returns:
//   1 if section is valid for merging
//   0 if section should be skipped
//
// A section is mergeable if:
//   - sh_type != SHT_NULL
//   - sh_flags & SHF_ALLOC
//   - sh_addralign is 0, 1, or a power of 2
int is_section_mergeable(const InputSection *sec);
// Determine section category for ordering
// 
// Parameters:
//   sec - Section to categorize
//
// Returns:
//   SectionCategory enum value
SectionCategory categorize_section(const InputSection *sec);
#endif // ELF_PARSER_H
```
### Section Merger Interface
```c
// File: 05_section_merger.h
#ifndef SECTION_MERGER_H
#define SECTION_MERGER_H
#include "04_section_types.h"
// Error codes from merger
typedef enum {
    MERGE_OK = 0,
    MERGE_ERR_MEMORY,               // Memory allocation failed
    MERGE_ERR_TOO_MANY_INPUTS,      // Exceeded MAX_INPUT_FILES
    MERGE_ERR_TOO_MANY_OUTPUTS,     // Exceeded MAX_OUTPUT_SECTIONS
    MERGE_ERR_TOO_MANY_MAPPINGS,    // Exceeded MAX_MAPPINGS per section
    MERGE_ERR_FLAG_CONFLICT,        // Incompatible section flags
    MERGE_ERR_ALIGN_CONFLICT,       // Alignment too large (>1GB)
    MERGE_ERR_DUPLICATE_INPUT,      // Same file added twice
    MERGE_ERR_NO_INPUTS,            // No input files to merge
    MERGE_ERR_MAPPING_BUILD         // Failed to build mapping table
} MergeError;
// Initialize a linker context
// 
// Parameters:
//   ctx - Context to initialize (must not be NULL)
//
// Returns:
//   MERGE_OK on success
//   MERGE_ERR_MEMORY on allocation failure
//
// Post-conditions:
//   - All arrays allocated with initial capacity
//   - Counts set to 0
//   - base_address set to 0x400000
MergeError linker_context_init(LinkerContext *ctx);
// Destroy a linker context and free all resources
// 
// Parameters:
//   ctx - Context to destroy (may be NULL)
//
// Frees:
//   - All input ObjectFiles
//   - All OutputSections and their mappings
//   - Mapping table
//   - Merged data buffers
void linker_context_destroy(LinkerContext *ctx);
// Add a parsed object file to the merge context
// 
// Parameters:
//   ctx - Linker context (must be initialized)
//   obj - Object file to add (ownership transferred to ctx)
//
// Returns:
//   MERGE_OK on success
//   MERGE_ERR_TOO_MANY_INPUTS if capacity exceeded
//   MERGE_ERR_DUPLICATE_INPUT if same filename already added
//
// Post-conditions:
//   - obj is now owned by ctx
//   - obj pointer should not be used by caller after this
MergeError linker_add_input(LinkerContext *ctx, ObjectFile *obj);
// Perform the section merge
// 
// Parameters:
//   ctx - Linker context with inputs added
//
// Returns:
//   MERGE_OK on success
//   MERGE_ERR_NO_INPUTS if no input files
//   MERGE_ERR_FLAG_CONFLICT if incompatible flags
//   MERGE_ERR_MEMORY on allocation failure
//
// This performs the full merge:
//   1. Group sections by name across all inputs
//   2. Calculate alignment padding for each placement
//   3. Compute output offsets
//   4. Build mapping table
//   5. Allocate merged data buffers
//
// After this call:
//   - All OutputSections are populated
//   - Mapping table is complete and queryable
//   - Merged data buffers contain concatenated content
MergeError linker_merge_sections(LinkerContext *ctx);
// Assign virtual addresses to output sections
// 
// Parameters:
//   ctx - Linker context (must have completed merge)
//
// Returns:
//   MERGE_OK on success
//   Appropriate error on failure
//
// Assigns virtual addresses in standard order:
//   .text → .rodata → .data → .bss
// Starting from ctx->base_address with proper alignment
MergeError linker_assign_addresses(LinkerContext *ctx);
// Get human-readable error message
const char *merge_error_string(MergeError err);
#endif // SECTION_MERGER_H
```
### Mapping Table Interface
```c
// File: 07_mapping_table.h
#ifndef MAPPING_TABLE_H
#define MAPPING_TABLE_H
#include "04_section_types.h"
// Error codes from mapping table operations
typedef enum {
    MAP_OK = 0,
    MAP_ERR_MEMORY,
    MAP_ERR_NOT_FOUND,
    MAP_ERR_INVALID_OFFSET,
    MAP_ERR_DUPLICATE
} MapError;
// Initialize a mapping table
// 
// Parameters:
//   table - Table to initialize
//   capacity - Initial hash table size (should be prime, > expected entries)
//
// Returns:
//   MAP_OK on success
//   MAP_ERR_MEMORY on failure
MapError mapping_table_init(MappingTable *table, size_t capacity);
// Destroy a mapping table
void mapping_table_destroy(MappingTable *table);
// Insert a mapping entry
// 
// Parameters:
//   table - Mapping table
//   filename - Input filename
//   section_name - Section name
//   mapping - Mapping record to insert (copied)
//   output - Output section this maps to
//
// Returns:
//   MAP_OK on success
//   MAP_ERR_DUPLICATE if key already exists
//   MAP_ERR_MEMORY on allocation failure
MapError mapping_table_insert(MappingTable *table,
                               const char *filename,
                               const char *section_name,
                               SectionMapping *mapping,
                               OutputSection *output);
// Lookup a mapping by (filename, section_name)
// 
// Parameters:
//   table - Mapping table
//   filename - Input filename
//   section_name - Section name
//   out_mapping - Output: pointer to mapping (may be NULL)
//   out_output - Output: pointer to output section (may be NULL)
//
// Returns:
//   MAP_OK if found
//   MAP_ERR_NOT_FOUND if no entry
MapError mapping_table_lookup(MappingTable *table,
                               const char *filename,
                               const char *section_name,
                               SectionMapping **out_mapping,
                               OutputSection **out_output);
// Translate an input offset to an output offset
// 
// Parameters:
//   ctx - Linker context with mapping table
//   filename - Input filename
//   section_name - Section name
//   input_offset - Offset within input section
//   out_output_offset - Output: corresponding output offset
//
// Returns:
//   MAP_OK on success
//   MAP_ERR_NOT_FOUND if (file, section) not in table
//   MAP_ERR_INVALID_OFFSET if input_offset >= section size
//
// Formula: output_offset = mapping.output_offset + input_offset
MapError translate_offset(LinkerContext *ctx,
                           const char *filename,
                           const char *section_name,
                           uint64_t input_offset,
                           uint64_t *out_output_offset);
// Translate to full virtual address
// 
// Parameters:
//   ctx - Linker context
//   filename - Input filename
//   section_name - Section name
//   input_offset - Offset within input section
//   out_vaddr - Output: final virtual address
//
// Returns:
//   MAP_OK on success
//   MAP_ERR_NOT_FOUND if not in table
//   MAP_ERR_INVALID_OFFSET if offset out of range
//
// Formula: vaddr = output.virtual_addr + mapping.output_offset + input_offset
MapError translate_to_vaddr(LinkerContext *ctx,
                             const char *filename,
                             const char *section_name,
                             uint64_t input_offset,
                             uint64_t *out_vaddr);
#endif // MAPPING_TABLE_H
```
## Algorithm Specification
### Algorithm 1: ELF Header Parsing
**Purpose**: Validate file is a valid x86-64 relocatable object and extract section table location.
**Input**: 
- `filename`: Path to potential object file
- `obj`: Output structure
**Output**: 
- `ParseError` code
- On success: `obj->ehdr` populated, `obj->filename` set
**Procedure**:
```
1. Open file for binary reading
   IF file not found: RETURN PARSE_ERR_FILE_NOT_FOUND
   IF open failed: RETURN PARSE_ERR_FILE_READ
2. Read exactly 64 bytes (sizeof(Elf64_Ehdr))
   IF read < 64 bytes: CLOSE file; RETURN PARSE_ERR_FILE_READ
3. Validate magic bytes (e_ident[0..3])
   IF e_ident[0] != 0x7f: RETURN PARSE_ERR_INVALID_MAGIC
   IF e_ident[1] != 'E': RETURN PARSE_ERR_INVALID_MAGIC
   IF e_ident[2] != 'L': RETURN PARSE_ERR_INVALID_MAGIC
   IF e_ident[3] != 'F': RETURN PARSE_ERR_INVALID_MAGIC
4. Validate class (e_ident[4])
   IF e_ident[EI_CLASS] != ELFCLASS64: RETURN PARSE_ERR_INVALID_CLASS
5. Validate endianness (e_ident[5])
   IF e_ident[EI_DATA] != ELFDATA2LSB: RETURN PARSE_ERR_INVALID_ENDIAN
6. Validate file type
   IF e_type != ET_REL: RETURN PARSE_ERR_INVALID_TYPE
7. Validate architecture
   IF e_machine != EM_X86_64: RETURN PARSE_ERR_INVALID_ARCH
8. Validate header sizes
   IF e_ehsize != 64: RETURN PARSE_ERR_INVALID_EHSIZE
   IF e_shentsize != 64: RETURN PARSE_ERR_INVALID_SHENTSIZE
9. Validate section table exists
   IF e_shnum == 0: RETURN PARSE_ERR_NO_SECTIONS
   IF e_shstrndx == SHN_UNDEF: RETURN PARSE_ERR_NO_SHSTRTAB
10. Copy filename and header to obj
    strncpy(obj->filename, filename, MAX_FILENAME-1)
    memcpy(&obj->ehdr, buffer, sizeof(Elf64_Ehdr))
11. CLOSE file
    RETURN PARSE_OK
```
**Invariants after execution**:
- File is closed
- `obj->ehdr` contains valid, validated header
- `obj->filename` contains source path
### Algorithm 2: Section Header Table Processing
**Purpose**: Read all section headers and resolve their names from string table.
**Input**:
- `obj`: Object file with valid ehdr
**Output**:
- `ParseError` code
- On success: `obj->sections` array populated, `obj->shstrtab` loaded
**Procedure**:
```
1. Seek to section header string table
   shstrtab_hdr_offset = e_shoff + e_shstrndx * 64
   SEEK to shstrtab_hdr_offset
   IF seek fails: RETURN PARSE_ERR_SECTION_READ
2. Read string table section header
   READ 64 bytes into Elf64_Shdr shstrtab_shdr
   IF read fails: RETURN PARSE_ERR_SECTION_READ
   IF shstrtab_shdr.sh_type != SHT_STRTAB: RETURN PARSE_ERR_SHSTRTAB_MISSING
3. Allocate and read string table content
   obj->shstrtab = malloc(shstrtab_shdr.sh_size)
   obj->shstrtab_size = shstrtab_shdr.sh_size
   SEEK to shstrtab_shdr.sh_offset
   READ shstrtab_shdr.sh_size bytes into obj->shstrtab
   IF read fails: FREE obj->shstrtab; RETURN PARSE_ERR_DATA_READ
4. Allocate section array
   obj->sections = calloc(e_shnum, sizeof(InputSection))
   obj->section_count = e_shnum
   obj->section_capacity = e_shnum
   IF allocation fails: FREE obj->shstrtab; RETURN PARSE_ERR_MEMORY
5. Read each section header
   FOR i = 0 TO e_shnum-1:
     a. SEEK to e_shoff + i * 64
     b. READ 64 bytes into Elf64_Shdr shdr
     c. IF read fails: GOTO cleanup_and_return_error
     d. Resolve section name
        name_offset = shdr.sh_name
        IF name_offset >= shstrtab_size:
            name = "(invalid)"
        ELSE:
            name = obj->shstrtab + name_offset
        strncpy(obj->sections[i].name, name, MAX_SECTION_NAME-1)
     e. Copy other fields
        obj->sections[i].type = shdr.sh_type
        obj->sections[i].flags = shdr.sh_flags
        obj->sections[i].align = shdr.sh_addralign
        obj->sections[i].size = shdr.sh_size
        obj->sections[i].file_offset = shdr.sh_offset
        obj->sections[i].original_index = i
        obj->sections[i].data = NULL  // Loaded separately
     f. Validate alignment
        IF sh_addralign != 0 AND sh_addralign != 1:
            IF NOT is_power_of_2(sh_addralign):
                RETURN PARSE_ERR_ALIGNMENT_INVALID
            IF sh_addralign > (1 << 30):  // 1GB
                RETURN PARSE_ERR_ALIGN_CONFLICT
     g. Validate size
        IF sh_size > (1 << 30):  // 1GB
            RETURN PARSE_ERR_SECTION_TOO_LARGE
6. RETURN PARSE_OK
```
**Invariants after execution**:
- All section headers loaded with resolved names
- String table loaded and owned
- Section data pointers are NULL (not yet loaded)
### Algorithm 3: Section Data Loading
**Purpose**: Load content for non-NOBITS sections.
**Input**:
- `obj`: Object file with sections populated
- `file`: Open file handle (or reopen)
**Output**:
- `ParseError` code
- On success: `InputSection.data` populated for PROGBITS sections
**Procedure**:
```
FOR each section i in obj->sections:
    1. Skip NOBITS sections (.bss)
       IF sections[i].type == SHT_NOBITS:
           sections[i].data = NULL
           CONTINUE
    2. Skip empty sections
       IF sections[i].size == 0:
           sections[i].data = NULL
           CONTINUE
    3. Allocate buffer
       sections[i].data = malloc(sections[i].size)
       IF allocation fails: RETURN PARSE_ERR_MEMORY
    4. Read section content
       SEEK to sections[i].file_offset
       READ sections[i].size bytes into sections[i].data
       IF read fails: 
           FREE sections[i].data
           RETURN PARSE_ERR_DATA_READ
RETURN PARSE_OK
```
**Memory ownership**:
- Each `sections[i].data` is owned by the ObjectFile
- Freed by `object_file_destroy()`
### Algorithm 4: Section Merging
**Purpose**: Combine sections from all inputs into output sections with proper alignment.
**Input**:
- `ctx`: Linker context with all inputs loaded
**Output**:
- `MergeError` code
- On success: `ctx->outputs` populated, `ctx->mapping` built
**Procedure**:
```
1. Validate inputs exist
   IF ctx->input_count == 0: RETURN MERGE_ERR_NO_INPUTS
2. Create output sections for each unique section name
   FOR each input file i:
     FOR each section j in inputs[i].sections:
       IF NOT is_section_mergeable(&section): CONTINUE
       a. Find or create output section
          out = find_output_section(ctx, section.name)
          IF out == NULL:
              out = create_output_section(ctx, section.name)
       b. Update output section alignment (max wins)
          IF section.align > out->align:
              out->align = section.align
       c. Merge flags (union)
          out->flags |= section.flags
3. Add input sections to output sections
   FOR each input file i (in order):
     FOR each section j (in file order):
       IF NOT is_section_mergeable(&section): CONTINUE
       out = find_output_section(ctx, section.name)
       a. Calculate padding
          current_size = out->mem_size
          padding = calc_padding(current_size, section.align)
       b. Calculate output offset
          output_offset = current_size + padding
       c. Create mapping record
          mapping.source_file = inputs[i].filename
          mapping.section_name = section.name
          mapping.input_size = section.size
          mapping.output_offset = output_offset
          mapping.padding_before = padding
          mapping.source_data = section.data
          mapping.source_section_idx = j
       d. Add mapping to output section
          APPEND mapping to out->mappings
          out->mapping_count++
       e. Update output section sizes
          out->mem_size = output_offset + section.size
          IF section.type != SHT_NOBITS:
              out->file_size = output_offset + section.size
          // Note: .bss adds to mem_size but NOT file_size
4. Build merged data buffers
   FOR each output section:
     IF output.file_size == 0: CONTINUE  // .bss or empty
     a. Allocate buffer
        output.data = calloc(output.file_size, 1)  // Zero-initialized
     b. Copy each input section's data
        FOR each mapping in output.mappings:
          IF mapping.source_data != NULL:
              dest = output.data + mapping.output_offset
              memcpy(dest, mapping.source_data, mapping.input_size)
5. Build mapping hash table
   total_mappings = 0
   FOR each output section:
     total_mappings += output.mapping_count
   hash_capacity = next_prime(total_mappings * 2 + 1)
   mapping_table_init(&ctx->mapping, hash_capacity)
   FOR each output section out:
     FOR each mapping m in out:
       key = format("%s:%s", m.source_file, m.section_name)
       mapping_table_insert(&ctx->mapping, key, &m, &out)
6. Calculate totals
   ctx->total_file_size = sum of all output.file_size
   ctx->total_mem_size = sum of all output.mem_size
RETURN MERGE_OK
```
**Alignment padding calculation**:
```c
static inline uint64_t calc_padding(uint64_t current, uint64_t align) {
    if (align == 0 || align == 1) return 0;
    uint64_t rem = current % align;
    if (rem == 0) return 0;
    return align - rem;
}
static inline uint64_t align_up(uint64_t value, uint64_t align) {
    if (align == 0 || align == 1) return value;
    return (value + align - 1) & ~(align - 1);
}
```

![ELF64 Header Memory Layout](./diagrams/tdd-diag-003.svg)

### Algorithm 5: Virtual Address Assignment
**Purpose**: Assign virtual addresses to output sections in standard order.
**Input**:
- `ctx`: Linker context with merged sections
**Output**:
- `MergeError` code
- On success: all `output.virtual_addr` set
**Procedure**:
```
1. Define standard section order
   section_order = [".text", ".rodata", ".data", ".bss"]
2. Start from base address
   current_vaddr = ctx->base_address  // Default: 0x400000
   current_file_offset = 0x1000  // Reserve space for ELF + program headers
3. Assign addresses in order
   FOR each section_name in section_order:
     out = find_output_section(ctx, section_name)
     IF out == NULL OR out->mem_size == 0: CONTINUE
     a. Align virtual address
        current_vaddr = align_up(current_vaddr, out->align)
     b. Align file offset (for non-.bss)
        IF out->file_size > 0:
            current_file_offset = align_up(current_file_offset, out->align)
            out->file_offset = current_file_offset
     c. Assign virtual address
        out->virtual_addr = current_vaddr
     d. Advance pointers
        current_vaddr += out->mem_size
        current_file_offset += out->file_size
4. Handle any remaining sections (custom names)
   FOR each output section NOT in standard_order:
     (Same alignment and assignment logic)
RETURN MERGE_OK
```
**Page alignment invariant**:
For segments (Milestone 4), `p_offset % p_align == p_vaddr % p_align` must hold. The current scheme ensures this because both start at multiples of their alignment.
## Error Handling Matrix
| Error | Detected By | Recovery | User-Visible? | System State |
|-------|-------------|----------|---------------|--------------|
| `PARSE_ERR_FILE_NOT_FOUND` | `fopen()` returns NULL | Abort, report filename | Yes: "file.o: No such file" | Clean, no allocations |
| `PARSE_ERR_INVALID_MAGIC` | Magic byte check | Abort, close file | Yes: "file.o: Not an ELF file" | File closed |
| `PARSE_ERR_INVALID_TYPE` | `e_type != ET_REL` | Abort | Yes: "file.o: Not relocatable (type=3)" | File closed |
| `PARSE_ERR_INVALID_ARCH` | `e_machine != 62` | Abort | Yes: "file.o: Wrong architecture (expected x86-64)" | File closed |
| `PARSE_ERR_NO_SHSTRTAB` | `e_shstrndx == 0` | Abort | Yes: "file.o: No section name table" | File closed |
| `PARSE_ERR_SECTION_READ` | `fread()` returns wrong count | Abort, free partial | Yes: "file.o: Section read error at offset N" | Partial cleanup |
| `PARSE_ERR_MEMORY` | `malloc()` returns NULL | Abort, cleanup | Yes: "Out of memory" | Full cleanup |
| `PARSE_ERR_ALIGNMENT_INVALID` | Non-power-of-2 alignment | Abort | Yes: "file.o: Invalid alignment 3 for section .foo" | Sections freed |
| `MERGE_ERR_FLAG_CONFLICT` | Incompatible SHF_* flags | Abort | Yes: "Cannot merge .text: conflicting flags (RWX vs RX)" | Clean |
| `MERGE_ERR_TOO_MANY_INPUTS` | `input_count >= MAX_INPUT_FILES` | Abort | Yes: "Too many input files (max 128)" | Clean |
| `MAP_ERR_NOT_FOUND` | Hash lookup returns empty | Return error | Yes: "No mapping for file.o:.text" | Clean |
| `MAP_ERR_INVALID_OFFSET` | `input_offset >= section_size` | Return error | Yes: "Offset 0x100 exceeds section size 0x80" | Clean |
### Cleanup Strategy
```c
// On any error, the cleanup order is:
// 1. Close file handles
// 2. Free section data buffers (each InputSection.data)
// 3. Free sections array
// 4. Free shstrtab
// 5. Zero the ObjectFile structure
void object_file_destroy(ObjectFile *obj) {
    if (obj == NULL) return;
    // Free section data
    if (obj->sections != NULL) {
        for (int i = 0; i < obj->section_count; i++) {
            if (obj->sections[i].data != NULL) {
                free(obj->sections[i].data);
            }
        }
        free(obj->sections);
    }
    // Free string table
    if (obj->shstrtab != NULL) {
        free(obj->shstrtab);
    }
    // Zero the structure
    memset(obj, 0, sizeof(ObjectFile));
}
```
## Implementation Sequence with Checkpoints
### Phase 1: ELF Header Parsing (2-3 hours)
**Files**: `01_elf_types.h`, `02_elf_parser.h`, `03_elf_parser.c`
**Implementation steps**:
1. Define all ELF constants and structures in `01_elf_types.h`
2. Declare parser interface in `02_elf_parser.h`
3. Implement `parse_elf_header()` function
4. Implement `parse_error_string()` for diagnostics
5. Write unit tests for header validation
**Checkpoint**: 
```bash
# Compile and test
gcc -c 03_elf_parser.c -o elf_parser.o
gcc tests/test_parser.c elf_parser.o -o test_parser
./test_parser
# Expected output:
# [PASS] Valid ELF header parsing
# [PASS] Invalid magic detection
# [PASS] Invalid class detection
# [PASS] Invalid type detection
# [PASS] Invalid architecture detection
# All tests passed!
```
At this point you can parse ELF headers but not yet read sections.
### Phase 2: Section Header Processing (2-3 hours)
**Files**: Continue `03_elf_parser.c`
**Implementation steps**:
1. Implement section header table reading
2. Implement string table loading and name resolution
3. Add section validation (alignment, size checks)
4. Extend unit tests for section parsing
**Checkpoint**:
```bash
./test_parser
# Expected output:
# [PASS] Section header table parsing
# [PASS] String table resolution
# [PASS] Section name extraction
# [PASS] Alignment validation
# All tests passed!
```
At this point you can read section headers with names, but no data loaded yet.
### Phase 3: Section Data Extraction (2-3 hours)
**Files**: Continue `03_elf_parser.c`, add `04_section_types.h`
**Implementation steps**:
1. Implement data loading for PROGBITS sections
2. Handle NOBITS sections specially (no file data)
3. Implement `object_file_destroy()` for cleanup
4. Test with various section types
**Checkpoint**:
```bash
# Create test object file
cat > test.s << 'EOF'
.section .text
    nop
.section .data
    .int 42
.section .bss
buffer:
    .skip 64
EOF
as -o test.o test.s
./test_parser test.o
# Expected output:
# Parsed test.o:
#   .text: type=PROGBITS, size=1, data_loaded=yes
#   .data: type=PROGBITS, size=4, data_loaded=yes
#   .bss: type=NOBITS, size=64, data_loaded=no
# All tests passed!
```
At this point you can fully parse object files.
### Phase 4: Section Grouping and Merging (2-3 hours)
**Files**: `05_section_merger.h`, `06_section_merger.c`
**Implementation steps**:
1. Implement `linker_context_init()` and `linker_context_destroy()`
2. Implement `linker_add_input()` to collect files
3. Implement section grouping by name
4. Implement alignment padding calculation
5. Implement offset computation and mapping record creation
6. Implement merged data buffer construction
**Checkpoint**:
```bash
# Create two test files
cat > a.s << 'EOF'
.section .text
.global func_a
func_a:
    ret
.section .data
.global var_a
var_a:
    .int 1
EOF
cat > b.s << 'EOF'
.section .text
.global func_b
func_b:
    ret
    ret
.section .rodata
message:
    .asciz "hello"
EOF
as -o a.o a.s
as -o b.o b.s
./test_merger a.o b.o
# Expected output:
# Merged sections:
#   .text: mem_size=3, file_size=3, align=16
#     a.o/.text: input_size=1 -> output_offset=0
#     b.o/.text: input_size=2 -> output_offset=16 (padding=15)
#   .data: mem_size=4, file_size=4
#     a.o/.data: input_size=4 -> output_offset=0
#   .rodata: mem_size=6, file_size=6
#     b.o/.rodata: input_size=6 -> output_offset=0
# Total: mem_size=13, file_size=13
# All tests passed!
```
At this point you can merge sections from multiple files.
### Phase 5: Mapping Table Construction (1-2 hours)
**Files**: `07_mapping_table.h`, `08_mapping_table.c`, `09_linker_context.h`
**Implementation steps**:
1. Implement hash function for (filename, section) keys
2. Implement `mapping_table_init()`, `mapping_table_destroy()`
3. Implement `mapping_table_insert()` with linear probing
4. Implement `mapping_table_lookup()` 
5. Implement `translate_offset()` and `translate_to_vaddr()`
6. Integrate with merger to build table during merge
**Checkpoint**:
```bash
./test_merger a.o b.o
# Expected output:
# (Previous output plus:)
# Mapping table: 4 entries
# Lookup a.o:.text -> output_offset=0
# Lookup b.o:.text -> output_offset=16
# Lookup a.o:.data -> output_offset=0
# Lookup b.o:.rodata -> output_offset=0
# 
# Offset translation:
#   a.o:.text+0 -> output .text+0
#   b.o:.text+0 -> output .text+16
#   b.o:.text+1 -> output .text+17
# All tests passed!
```
At this point the mapping table is complete and queryable.
### Phase 6: Virtual Address Assignment (1 hour)
**Files**: Continue `06_section_merger.c`
**Implementation steps**:
1. Implement standard section ordering
2. Implement address assignment with alignment
3. Handle custom sections
4. Validate page alignment constraints
**Checkpoint**:
```bash
./test_merger a.o b.o
# Expected output:
# (Previous output plus:)
# Virtual addresses:
#   .text: vaddr=0x401000, file_offset=0x1000
#   .rodata: vaddr=0x401020, file_offset=0x1020
#   .data: vaddr=0x403000, file_offset=0x3000
# 
# Address translation:
#   a.o:.text+0 -> vaddr 0x401000
#   b.o:.text+0 -> vaddr 0x401010
# All tests passed!
```
**Milestone 1 Complete**: Full section merging with mapping table ready for relocation processing.

![Section Header Entry Memory Layout](./diagrams/tdd-diag-004.svg)

## Test Specification
### Test Suite: ELF Parser
```c
// tests/test_parser.c
// Test: Valid ELF header parsing
void test_parse_valid_header(void) {
    // Create minimal valid .o file
    system("echo 'nop' | as -o /tmp/valid.o");
    ObjectFile obj;
    ParseError err = parse_object_file("/tmp/valid.o", &obj);
    ASSERT_EQ(err, PARSE_OK);
    ASSERT_EQ(obj.ehdr.e_type, ET_REL);
    ASSERT_EQ(obj.ehdr.e_machine, EM_X86_64);
    ASSERT_TRUE(obj.section_count > 0);
    object_file_destroy(&obj);
}
// Test: Invalid magic bytes
void test_parse_invalid_magic(void) {
    // Create file with wrong magic
    FILE *f = fopen("/tmp/bad_magic.o", "wb");
    fwrite("\x7fELG", 1, 4, f);  // 'G' instead of 'F'
    fclose(f);
    ObjectFile obj;
    ParseError err = parse_object_file("/tmp/bad_magic.o", &obj);
    ASSERT_EQ(err, PARSE_ERR_INVALID_MAGIC);
}
// Test: Non-relocatable file
void test_parse_non_relocatable(void) {
    // Compile to executable (not object)
    system("echo 'int main(){return 0;}' | gcc -x c - -o /tmp/exec");
    ObjectFile obj;
    ParseError err = parse_object_file("/tmp/exec", &obj);
    ASSERT_EQ(err, PARSE_ERR_INVALID_TYPE);
}
// Test: Wrong architecture
void test_parse_wrong_arch(void) {
    // This test requires a cross-compiled ARM object file
    // Skip if not available
    if (access("/tmp/arm.o", F_OK) != 0) {
        TEST_SKIP("ARM object file not available");
        return;
    }
    ObjectFile obj;
    ParseError err = parse_object_file("/tmp/arm.o", &obj);
    ASSERT_EQ(err, PARSE_ERR_INVALID_ARCH);
}
// Test: Section data loading
void test_section_data_loading(void) {
    system("as -o /tmp/data.o << 'EOF'\n"
           ".section .text\n"
           "    .byte 0xCC\n"
           ".section .data\n"
           "    .int 0xDEADBEEF\n"
           ".section .bss\n"
           "buffer: .skip 100\n"
           "EOF");
    ObjectFile obj;
    ParseError err = parse_object_file("/tmp/data.o", &obj);
    ASSERT_EQ(err, PARSE_OK);
    // Find sections
    InputSection *text = NULL, *data = NULL, *bss = NULL;
    for (int i = 0; i < obj.section_count; i++) {
        if (strcmp(obj.sections[i].name, ".text") == 0) text = &obj.sections[i];
        if (strcmp(obj.sections[i].name, ".data") == 0) data = &obj.sections[i];
        if (strcmp(obj.sections[i].name, ".bss") == 0) bss = &obj.sections[i];
    }
    // Verify data loading
    ASSERT_NOT_NULL(text);
    ASSERT_NOT_NULL(text->data);
    ASSERT_EQ(text->data[0], 0xCC);
    ASSERT_NOT_NULL(data);
    ASSERT_NOT_NULL(data->data);
    ASSERT_EQ(*(uint32_t*)data->data, 0xDEADBEEF);
    ASSERT_NOT_NULL(bss);
    ASSERT_NULL(bss->data);  // NOBITS has no file data
    ASSERT_EQ(bss->size, 100);
    object_file_destroy(&obj);
}
// Test: Alignment validation
void test_alignment_validation(void) {
    // Create object with unusual alignment
    system("as -o /tmp/align.o << 'EOF'\n"
           ".section .custom, \"a\", @progbits\n"
           ".align 256\n"
           "    .byte 0\n"
           "EOF");
    ObjectFile obj;
    ParseError err = parse_object_file("/tmp/align.o", &obj);
    ASSERT_EQ(err, PARSE_OK);
    // Find .custom section
    InputSection *custom = NULL;
    for (int i = 0; i < obj.section_count; i++) {
        if (strcmp(obj.sections[i].name, ".custom") == 0) {
            custom = &obj.sections[i];
            break;
        }
    }
    ASSERT_NOT_NULL(custom);
    ASSERT_EQ(custom->align, 256);
    object_file_destroy(&obj);
}
```
### Test Suite: Section Merger
```c
// tests/test_merger.c
// Test: Basic two-file merge
void test_basic_merge(void) {
    system("as -o /tmp/a.o << 'EOF'\n"
           ".section .text\n"
           "    nop\n"
           "EOF");
    system("as -o /tmp/b.o << 'EOF'\n"
           ".section .text\n"
           "    ret\n"
           "EOF");
    LinkerContext ctx;
    ASSERT_EQ(linker_context_init(&ctx), MERGE_OK);
    ObjectFile a, b;
    ASSERT_EQ(parse_object_file("/tmp/a.o", &a), PARSE_OK);
    ASSERT_EQ(parse_object_file("/tmp/b.o", &b), PARSE_OK);
    ASSERT_EQ(linker_add_input(&ctx, &a), MERGE_OK);
    ASSERT_EQ(linker_add_input(&ctx, &b), MERGE_OK);
    ASSERT_EQ(linker_merge_sections(&ctx), MERGE_OK);
    // Verify: one .text output section
    ASSERT_EQ(ctx.output_count, 1);
    ASSERT_STREQ(ctx.outputs[0].name, ".text");
    ASSERT_EQ(ctx.outputs[0].mapping_count, 2);
    // Verify: correct ordering
    ASSERT_EQ(ctx.outputs[0].mappings[0].output_offset, 0);
    ASSERT_EQ(ctx.outputs[0].mappings[1].output_offset, 1);  // After 1-byte nop
    linker_context_destroy(&ctx);
}
// Test: Alignment padding
void test_alignment_padding(void) {
    system("as -o /tmp/pad1.o << 'EOF'\n"
           ".section .text\n"
           ".align 16\n"
           "func1:\n"
           "    nop\n"  // 1 byte
           "EOF");
    system("as -o /tmp/pad2.o << 'EOF'\n"
           ".section .text\n"
           ".align 16\n"
           "func2:\n"
           "    ret\n"  // 1 byte
           "EOF");
    LinkerContext ctx;
    linker_context_init(&ctx);
    ObjectFile a, b;
    parse_object_file("/tmp/pad1.o", &a);
    parse_object_file("/tmp/pad2.o", &b);
    linker_add_input(&ctx, &a);
    linker_add_input(&ctx, &b);
    linker_merge_sections(&ctx);
    OutputSection *text = &ctx.outputs[0];
    // First section at offset 0
    ASSERT_EQ(text->mappings[0].output_offset, 0);
    ASSERT_EQ(text->mappings[0].padding_before, 0);
    // Second section padded to 16-byte alignment
    ASSERT_EQ(text->mappings[1].output_offset, 16);
    ASSERT_EQ(text->mappings[1].padding_before, 15);
    linker_context_destroy(&ctx);
}
// Test: .bss handling
void test_bss_handling(void) {
    system("as -o /tmp/bss.o << 'EOF'\n"
           ".section .data\n"
           "initialized: .int 42\n"
           ".section .bss\n"
           "uninitialized: .skip 100\n"
           "EOF");
    LinkerContext ctx;
    linker_context_init(&ctx);
    ObjectFile obj;
    parse_object_file("/tmp/bss.o", &obj);
    linker_add_input(&ctx, &obj);
    linker_merge_sections(&ctx);
    // Find .bss section
    OutputSection *bss = NULL;
    for (size_t i = 0; i < ctx.output_count; i++) {
        if (strcmp(ctx.outputs[i].name, ".bss") == 0) {
            bss = &ctx.outputs[i];
            break;
        }
    }
    ASSERT_NOT_NULL(bss);
    // .bss: mem_size > 0, file_size == 0
    ASSERT_EQ(bss->mem_size, 100);
    ASSERT_EQ(bss->file_size, 0);
    ASSERT_NULL(bss->data);  // No data buffer
    linker_context_destroy(&ctx);
}
// Test: Multiple section types
void test_multiple_section_types(void) {
    system("as -o /tmp/multi.o << 'EOF'\n"
           ".section .text\n"
           "    nop\n"
           ".section .rodata\n"
           "msg: .asciz \"hello\"\n"
           ".section .data\n"
           "value: .int 42\n"
           ".section .bss\n"
           "buffer: .skip 64\n"
           "EOF");
    LinkerContext ctx;
    linker_context_init(&ctx);
    ObjectFile obj;
    parse_object_file("/tmp/multi.o", &obj);
    linker_add_input(&ctx, &obj);
    linker_merge_sections(&ctx);
    // Should have 4 output sections
    ASSERT_EQ(ctx.output_count, 4);
    // Verify flags
    for (size_t i = 0; i < ctx.output_count; i++) {
        OutputSection *out = &ctx.outputs[i];
        if (strcmp(out->name, ".text") == 0) {
            ASSERT_TRUE(out->flags & SHF_EXECINSTR);
            ASSERT_FALSE(out->flags & SHF_WRITE);
        }
        if (strcmp(out->name, ".data") == 0 || strcmp(out->name, ".bss") == 0) {
            ASSERT_TRUE(out->flags & SHF_WRITE);
        }
        if (strcmp(out->name, ".rodata") == 0) {
            ASSERT_FALSE(out->flags & SHF_WRITE);
            ASSERT_FALSE(out->flags & SHF_EXECINSTR);
        }
    }
    linker_context_destroy(&ctx);
}
// Test: Non-allocatable sections ignored
void test_non_allocatable_ignored(void) {
    system("as -o /tmp/meta.o << 'EOF'\n"
           ".section .text\n"
           "    nop\n"
           ".section .comment\n"
           "    .asciz \"GCC\"\n"
           ".section .note\n"
           "    .int 0\n"
           "EOF");
    LinkerContext ctx;
    linker_context_init(&ctx);
    ObjectFile obj;
    parse_object_file("/tmp/meta.o", &obj);
    linker_add_input(&ctx, &obj);
    linker_merge_sections(&ctx);
    // Only .text should be merged
    ASSERT_EQ(ctx.output_count, 1);
    ASSERT_STREQ(ctx.outputs[0].name, ".text");
    linker_context_destroy(&ctx);
}
```
### Test Suite: Mapping Table
```c
// tests/test_mapping.c
// Test: Basic insert and lookup
void test_mapping_basic(void) {
    MappingTable table;
    ASSERT_EQ(mapping_table_init(&table, 16), MAP_OK);
    SectionMapping mapping = {.output_offset = 0x100};
    OutputSection output = {.name = ".text"};
    ASSERT_EQ(mapping_table_insert(&table, "file.o", ".text", &mapping, &output), MAP_OK);
    SectionMapping *found_mapping;
    OutputSection *found_output;
    ASSERT_EQ(mapping_table_lookup(&table, "file.o", ".text", &found_mapping, &found_output), MAP_OK);
    ASSERT_EQ(found_mapping->output_offset, 0x100);
    ASSERT_EQ(found_output, &output);
    mapping_table_destroy(&table);
}
// Test: Not found
void test_mapping_not_found(void) {
    MappingTable table;
    mapping_table_init(&table, 16);
    SectionMapping *found_mapping;
    ASSERT_EQ(mapping_table_lookup(&table, "nonexistent.o", ".text", &found_mapping, NULL), MAP_ERR_NOT_FOUND);
    mapping_table_destroy(&table);
}
// Test: Offset translation
void test_offset_translation(void) {
    LinkerContext ctx;
    linker_context_init(&ctx);
    // Manually set up a simple mapping
    // (In practice, this is done by linker_merge_sections)
    // ...
    uint64_t output_offset;
    ASSERT_EQ(translate_offset(&ctx, "test.o", ".text", 0x10, &output_offset), MAP_OK);
    // output_offset = mapping.output_offset + input_offset
    // = 0x1000 + 0x10 = 0x1010
    ASSERT_EQ(output_offset, 0x1010);
    linker_context_destroy(&ctx);
}
// Test: Virtual address translation
void test_vaddr_translation(void) {
    LinkerContext ctx;
    linker_context_init(&ctx);
    ctx.base_address = 0x400000;
    // Set up mapping with output section at 0x401000
    // ...
    uint64_t vaddr;
    ASSERT_EQ(translate_to_vaddr(&ctx, "test.o", ".text", 0x20, &vaddr), MAP_OK);
    // vaddr = output.virtual_addr + mapping.output_offset + input_offset
    // = 0x401000 + 0x1000 + 0x20 = 0x402020
    ASSERT_EQ(vaddr, 0x402020);
    linker_context_destroy(&ctx);
}
// Test: Invalid offset
void test_invalid_offset(void) {
    LinkerContext ctx;
    linker_context_init(&ctx);
    // Set up mapping with input_size = 0x100
    // ...
    uint64_t output_offset;
    ASSERT_EQ(translate_offset(&ctx, "test.o", ".text", 0x200, &output_offset), MAP_ERR_INVALID_OFFSET);
    // 0x200 > 0x100
    linker_context_destroy(&ctx);
}
// Test: Hash collision handling
void test_hash_collision(void) {
    MappingTable table;
    mapping_table_init(&table, 4);  // Small table to force collisions
    // Insert multiple entries
    for (int i = 0; i < 10; i++) {
        char key[32];
        snprintf(key, sizeof(key), "file%d.o:.text", i);
        SectionMapping mapping = {.output_offset = i * 0x100};
        ASSERT_EQ(mapping_table_insert(&table, "file", key, &mapping, NULL), MAP_OK);
    }
    // Verify all can be found
    for (int i = 0; i < 10; i++) {
        char key[32];
        snprintf(key, sizeof(key), "file%d.o:.text", i);
        SectionMapping *found;
        ASSERT_EQ(mapping_table_lookup(&table, "file", key, &found, NULL), MAP_OK);
        ASSERT_EQ(found->output_offset, (uint64_t)(i * 0x100));
    }
    mapping_table_destroy(&table);
}
```
## Performance Targets
| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| Parse single .o file | < 1ms per 100KB file | `time ./linker file.o` with gettimeofday |
| Parse 1000 .o files | < 5 seconds total | Benchmark with synthetic 1000-file input |
| Section merge (1000 files) | < 2 seconds | Measure after parsing complete |
| Mapping table lookup | < 100ns average | Microbenchmark with 1M lookups |
| Memory for merged sections | < 2x total input size | RSS measurement with /proc/self/status |
| Hash table load factor | 0.3 - 0.7 | Capacity / count ratio |
### Memory Budget
```
Per input file:
  ObjectFile struct:        ~300 bytes
  Per section:              ~128 bytes + data size
  String table:             ~1KB average
Per output section:
  OutputSection struct:     ~200 bytes
  Per mapping:              ~100 bytes
  Merged data:              sum of input sizes (for non-NOBITS)
Mapping table:
  Per entry:                ~150 bytes
  Hash overhead:            2-3x entries for load factor
Example for 100-file project with 500 sections:
  Input storage:    100 * 300 + 500 * 128 + 100KB data ≈ 200KB
  Output storage:   10 * 200 + 500 * 100 + 100KB data ≈ 160KB  
  Mapping table:    500 * 150 * 2.5 ≈ 190KB
  Total:            ~550KB
```

![Input Section to Output Section Mapping](./diagrams/tdd-diag-005.svg)

## Integration Notes
### Dependencies
- **Standard library**: `<stdint.h>`, `<stdio.h>`, `<stdlib.h>`, `<string.h>`, `<stdbool.h>`
- **System calls**: `open()`, `read()`, `close()`, `mmap()` (optional optimization)
- **No external libraries required**
### API for Milestone 2 (Symbol Resolution)
The Symbol Resolution module will:
1. Query `ctx->inputs[i].sections` to find `.symtab` and `.strtab`
2. Use `translate_to_vaddr()` to compute symbol addresses after resolution
3. Access `OutputSection.virtual_addr` for section base addresses
### API for Milestone 3 (Relocation Processing)
The Relocation Processing module will:
1. Query `translate_offset()` for every relocation site
2. Use `mapping_table_lookup()` to find which output section contains a relocation
3. Write to `OutputSection.data` buffer at computed offsets
### API for Milestone 4 (Executable Generation)
The Executable Generation module will:
1. Read `ctx->outputs` array for segment content
2. Use `OutputSection.file_offset` and `virtual_addr` for program headers
3. Write `OutputSection.data` buffers to the output file
[[CRITERIA_JSON: {"module_id": "build-linker-m1", "criteria": ["Parse ELF64 header from input file and validate magic bytes (0x7f ELF), class (ELFCLASS64=2), endianness (ELFDATA2LSB=1), type (ET_REL=1), and architecture (EM_X86_64=62)", "Read section header table from file offset e_shoff, parsing e_shnum entries each of e_shentsize (64) bytes", "Load section name string table from section at index e_shstrndx, resolving section names via sh_name offset into the string table", "Extract section data for SHT_PROGBITS and other non-NOBITS types by reading sh_size bytes from file offset sh_offset", "Handle SHT_NOBITS sections (.bss) by setting data pointer to NULL and tracking size separately from file content", "Filter sections to mergeable set: sh_type != SHT_NULL AND sh_flags & SHF_ALLOC", "Group input sections by name across all input files into corresponding output sections", "Calculate alignment padding using formula: padding = (align - (current_size % align)) % align for power-of-2 alignments", "Concatenate input sections within each output group, applying computed padding before each section", "Track input-to-output mapping: for each input section, record (source_file, section_name) → output_offset", "Build hash table mapping (filename:section_name) to SectionMapping for O(1) average lookup", "Distinguish file_size from mem_size in output sections: mem_size includes .bss, file_size excludes it", "Assign output section file offsets starting at page-aligned boundary (0x1000) with proper alignment", "Assign virtual addresses to output sections in standard order (.text, .rodata, .data, .bss) starting from base address 0x400000", "Support lookup query: given (filename, section_name, input_offset) return output section offset via mapping table", "Validate section flag consistency: warn if merged sections have conflicting SHF_WRITE or SHF_EXECINSTR flags"]}]
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: build-linker-m2 -->
# Symbol Resolution: Technical Design Specification
## Module Charter
The Symbol Resolution module builds the linker's global address book: a unified symbol table that maps every named entity across all input object files to its final virtual address. This module parses `.symtab` sections from each input file, resolves symbol references according to ELF linking rules (strong overrides weak, multiple strongs are errors, COMMON merges by largest size), and assigns final addresses using the section mapping table from Milestone 1. It does NOT process relocations or generate executable headers—its sole responsibility is producing a complete, consistent symbol table with every global symbol resolved to a concrete address.
Upstream, the module depends on parsed `ObjectFile` structures with loaded `.symtab` and `.strtab` sections, and the input-to-output mapping table from Section Merging. Downstream, it feeds the Relocation Processing module with symbol addresses and the Executable Generation module with the entry point. The module maintains strict invariants: no duplicate strong symbols exist in the final table, all non-weak undefined symbols are detected before address assignment, every resolved symbol has a valid virtual address, and local symbols never appear in the global table.
## File Structure
```
linker/
├── 11_symbol_types.h       # Symbol data structures and constants
├── 12_symbol_parser.h      # Symbol table parsing interface
├── 13_symbol_parser.c      # .symtab/.strtab parsing implementation
├── 14_global_symbols.h     # Global symbol table interface
├── 15_global_symbols.c     # Hash table and resolution logic
├── 16_symbol_resolver.h    # Resolution rules interface
├── 17_symbol_resolver.c    # Strong/weak/COMMON resolution
├── 18_address_assign.h     # Final address assignment interface
├── 19_address_assign.c     # Virtual address computation
├── 20_symbol_main.c        # Test driver for symbol resolution
└── tests/
    ├── test_symbol_parse.c # Symbol table parsing tests
    ├── test_resolution.c   # Resolution rules tests
    └── fixtures/
        ├── strong.o        # Strong global symbol
        ├── weak.o          # Weak symbol definitions
        ├── common.o        # COMMON symbols
        └── conflict.o      # Duplicate strong (error case)
```
## Complete Data Model
### ELF Symbol Table Entry Structure
The ELF64 symbol table entry (`Elf64_Sym`) is 24 bytes:
```c
// File: 11_symbol_types.h
#ifndef SYMBOL_TYPES_H
#define SYMBOL_TYPES_H
#include <stdint.h>
#include <stddef.h>
#include "04_section_types.h"  // For LinkerContext
// Symbol binding (st_info high 4 bits)
#define STB_LOCAL      0   // Local symbol, not visible outside this file
#define STB_GLOBAL     1   // Global symbol, visible to all object files
#define STB_WEAK       2   // Weak symbol, lower precedence than global
// Symbol type (st_info low 4 bits)
#define STT_NOTYPE     0   // Symbol type unspecified
#define STT_OBJECT     1   // Symbol is a data object (variable)
#define STT_FUNC       2   // Symbol is a function
#define STT_SECTION    3   // Symbol associated with a section
#define STT_FILE       4   // Symbol's name is file name
#define STT_COMMON     5   // Symbol is a common data object
#define STT_TLS        6   // Thread-local storage
// Symbol visibility (st_other)
#define STV_DEFAULT    0   // Default visibility
#define STV_INTERNAL   1   // Internal visibility
#define STV_HIDDEN     2   // Hidden from other components
#define STV_PROTECTED  3   // Cannot be preempted
// Special section indices
#define SHN_UNDEF      0       // Undefined/external symbol
#define SHN_LORESERVE  0xFF00  // Beginning of reserved indices
#define SHN_ABS        0xFFF1  // Absolute value (not section-relative)
#define SHN_COMMON     0xFFF2  // COMMON symbol (uninitialized)
// Macros to extract binding and type from st_info
#define ELF64_ST_BIND(i)    ((i) >> 4)
#define ELF64_ST_TYPE(i)    ((i) & 0xF)
#define ELF64_ST_INFO(b, t) (((b) << 4) | ((t) & 0xF))
// ELF64 Symbol Table Entry - 24 bytes, packed
typedef struct __attribute__((packed)) {
    uint32_t st_name;    // +0x00: Symbol name (index into string table)
    uint8_t  st_info;    // +0x04: Type and binding (see macros above)
    uint8_t  st_other;   // +0x05: Visibility (usually 0)
    uint16_t st_shndx;   // +0x06: Section index or SHN_* special value
    uint64_t st_value;   // +0x08: Value (section offset or address)
    uint64_t st_size;    // +0x10: Size in bytes (0 if unknown)
} Elf64_Sym;  // Total: 24 bytes (0x18)
// Symbol name maximum length
#define MAX_SYMBOL_NAME   256
// Resolution state of a symbol
typedef enum {
    SYM_STATE_UNDEF,     // Referenced but not yet defined
    SYM_STATE_DEFINED,   // Has a definition (strong or weak)
    SYM_STATE_COMMON,    // COMMON symbol awaiting allocation
    SYM_STATE_RESOLVED   // Final address assigned
} SymbolState;
// Represents a symbol from an input object file
// This is the parsed representation of an Elf64_Sym
typedef struct {
    char name[MAX_SYMBOL_NAME];    // Resolved symbol name from .strtab
    uint8_t type;                  // STT_* value
    uint8_t binding;               // STB_* value
    uint8_t visibility;            // STV_* value
    uint16_t section_idx;          // st_shndx: section index or SHN_*
    uint64_t value;                // st_value: offset within section
    uint64_t size;                 // st_size: symbol size in bytes
    // Tracking information
    int source_file_idx;           // Index in LinkerContext.inputs
    int source_sym_idx;            // Index in source file's symbol table
    uint64_t final_address;        // Computed after resolution (0 if not resolved)
} Symbol;
// Global symbol entry in the unified symbol table
// This represents the linker's knowledge about one symbol name
typedef struct {
    char name[MAX_SYMBOL_NAME];    // Symbol name (key for lookup)
    uint8_t type;                  // STT_* from the winning definition
    uint8_t binding;               // Final binding after resolution
    SymbolState state;             // Current resolution state
    // Definition information (if state != UNDEF)
    int source_file_idx;           // Which input file provides definition
    int source_sym_idx;            // Index in that file's symbol table
    uint16_t section_idx;          // INPUT section index (needs translation)
    uint64_t section_offset;       // Offset within input section (st_value)
    uint64_t size;                 // Symbol size
    // For resolved symbols
    uint64_t final_address;        // Computed virtual address
    // For COMMON symbols
    uint64_t common_size;          // Largest size seen (for merging)
    uint64_t common_align;         // Alignment requirement
    // Resolution tracking
    int is_strong;                 // 1 if strong (STB_GLOBAL) definition
    int ref_count;                 // Number of files referencing this symbol
    // Reference tracking for error messages
    int first_ref_file_idx;        // First file that referenced this
    int first_ref_sym_idx;         // Symbol index in that file
} GlobalSymbol;
// Hash table for global symbol lookup
// Uses open addressing with linear probing
typedef struct {
    GlobalSymbol **entries;        // Array of pointers to GlobalSymbol
    size_t capacity;               // Hash table size (power of 2)
    size_t count;                  // Number of entries
    size_t tombstones;             // Deleted entries (for resize decision)
} GlobalSymbolTable;
// Error codes for symbol operations
typedef enum {
    SYM_OK = 0,
    SYM_ERR_MEMORY,                // Memory allocation failed
    SYM_ERR_NOT_FOUND,             // Symbol not in table
    SYM_ERR_DUPLICATE_STRONG,      // Multiple strong definitions
    SYM_ERR_UNDEFINED,             // Symbol referenced but never defined
    SYM_ERR_INVALID_INDEX,         // Symbol index out of range
    SYM_ERR_INVALID_SECTION,       // st_shndx references invalid section
    SYM_ERR_PARSE_ERROR,           // Cannot parse symbol table entry
    SYM_ERR_STRTAB_MISSING,        // No string table for symbol names
    SYM_ERR_NO_INPUTS,             // No input files to process
    SYM_ERR_RESOLUTION_FAILED      // Resolution logic error
} SymbolError;
#endif // SYMBOL_TYPES_H
```
### Memory Layout of Elf64_Sym
| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0x00 | 4 | st_name | Index into `.strtab` for symbol name |
| 0x04 | 1 | st_info | Binding (high 4 bits) + Type (low 4 bits) |
| 0x05 | 1 | st_other | Visibility (usually 0 for STV_DEFAULT) |
| 0x06 | 2 | st_shndx | Section index or SHN_UNDEF/SHN_COMMON/SHN_ABS |
| 0x08 | 8 | st_value | For defined symbols: offset within section |
| 0x10 | 8 | st_size | Symbol size in bytes (0 if unknown) |
Total: 24 bytes (0x18)
### Symbol Table Ordering Invariant
ELF symbol tables have a required ordering:
1. **Index 0**: Null symbol (all zeros) — required by ELF spec
2. **Indices 1 to sh_info-1**: Local symbols (STB_LOCAL)
3. **Indices sh_info onwards**: Global and weak symbols
The `.symtab` section header's `sh_info` field contains the index of the first non-local symbol. This enables efficient iteration over only global symbols.
```c
// Symbol table access pattern
// sh_info from section header tells us where locals end
for (size_t i = 0; i < symtab_count; i++) {
    if (i < sh_info) {
        // Local symbol - file scope only
    } else {
        // Global/weak symbol - participates in resolution
    }
}
```

![Symbol Resolution Module Architecture](./diagrams/tdd-diag-015.svg)

### Global Symbol Table Structure
The global symbol table is a hash table with open addressing:
```
GlobalSymbolTable
┌─────────────────────────────────────────────────────────────────┐
│ capacity = 16, count = 5                                         │
├─────┬───────────────────────────────────────────────────────────┤
│  0  │ NULL                                                       │
│  1  │ GlobalSymbol* ──► [name="main", state=RESOLVED, addr=0x401000] │
│  2  │ GlobalSymbol* ──► [name="helper", state=RESOLVED, addr=0x401020] │
│  3  │ TOMBSTONE                                                  │
│  4  │ NULL                                                       │
│  5  │ GlobalSymbol* ──► [name="buffer", state=COMMON, size=256] │
│  6  │ NULL                                                       │
│  7  │ GlobalSymbol* ──► [name="printf", state=UNDEF, refs=3]    │
│  8  │ NULL                                                       │
│ ... │ NULL                                                       │
│ 15  │ GlobalSymbol* ──► [name="data", state=RESOLVED, addr=0x403000] │
└─────┴───────────────────────────────────────────────────────────┘
Hash function: hash_string(name) % capacity
Collision resolution: linear probing (i+1, i+2, ...)
Load factor: count / capacity (target < 0.7)
```
## Interface Contracts
### Symbol Parser Interface
```c
// File: 12_symbol_parser.h
#ifndef SYMBOL_PARSER_H
#define SYMBOL_PARSER_H
#include "11_symbol_types.h"
#include "04_section_types.h"
// Find .symtab and .strtab sections in an object file
//
// Parameters:
//   obj        - Parsed object file (must not be NULL)
//   symtab_idx - Output: index of .symtab section (may be NULL)
//   strtab_idx - Output: index of .strtab section (may be NULL)
//
// Returns:
//   SYM_OK if both sections found
//   SYM_ERR_PARSE_ERROR if .symtab not found
//   SYM_ERR_STRTAB_MISSING if .strtab not found
//
// Note: .strtab may be in obj->sections or as sh_link from .symtab
SymbolError find_symbol_tables(ObjectFile *obj, 
                                int *symtab_idx, 
                                int *strtab_idx);
// Parse all symbols from an object file's .symtab
//
// Parameters:
//   obj - Object file with sections loaded (must not be NULL)
//
// Returns:
//   SYM_OK on success
//   SYM_ERR_PARSE_ERROR if symbol table malformed
//   SYM_ERR_MEMORY if allocation fails
//
// On success:
//   - obj->symbols is allocated with obj->symbol_count entries
//   - Each Symbol.name is resolved from .strtab
//   - Each Symbol has source_file_idx set to -1 (caller must set)
//   - Ownership of symbols array transferred to caller
//
// On failure:
//   - obj->symbols is NULL
//   - No memory allocated
SymbolError parse_symbols(ObjectFile *obj);
// Get a symbol by index from an object file
//
// Parameters:
//   obj     - Object file with symbols parsed
//   idx     - Symbol index (0 to symbol_count-1)
//   out_sym - Output: pointer to symbol (may be NULL)
//
// Returns:
//   SYM_OK if index valid
//   SYM_ERR_INVALID_INDEX if idx >= symbol_count
SymbolError get_symbol(ObjectFile *obj, uint32_t idx, Symbol **out_sym);
// Check if a symbol is defined (has a section)
//
// Parameters:
//   sym - Symbol to check
//
// Returns:
//   1 if symbol is defined (st_shndx != SHN_UNDEF)
//   0 if symbol is undefined
static inline int symbol_is_defined(const Symbol *sym) {
    return sym->section_idx != SHN_UNDEF;
}
// Check if a symbol is a COMMON symbol
//
// Parameters:
//   sym - Symbol to check
//
// Returns:
//   1 if symbol is COMMON (st_shndx == SHN_COMMON)
//   0 otherwise
static inline int symbol_is_common(const Symbol *sym) {
    return sym->section_idx == SHN_COMMON;
}
// Check if a symbol is local (file-scope only)
//
// Parameters:
//   sym - Symbol to check
//
// Returns:
//   1 if symbol binding is STB_LOCAL
//   0 otherwise
static inline int symbol_is_local(const Symbol *sym) {
    return sym->binding == STB_LOCAL;
}
// Check if a symbol is strong (STB_GLOBAL with definition)
//
// Parameters:
//   sym - Symbol to check
//
// Returns:
//   1 if symbol is strong global definition
//   0 otherwise
static inline int symbol_is_strong(const Symbol *sym) {
    return sym->binding == STB_GLOBAL && 
           sym->section_idx != SHN_UNDEF &&
           sym->section_idx != SHN_COMMON;
}
// Check if a symbol is weak
//
// Parameters:
//   sym - Symbol to check
//
// Returns:
//   1 if symbol binding is STB_WEAK
//   0 otherwise
static inline int symbol_is_weak(const Symbol *sym) {
    return sym->binding == STB_WEAK;
}
// Get human-readable error string
const char *symbol_error_string(SymbolError err);
#endif // SYMBOL_PARSER_H
```
### Global Symbol Table Interface
```c
// File: 14_global_symbols.h
#ifndef GLOBAL_SYMBOLS_H
#define GLOBAL_SYMBOLS_H
#include "11_symbol_types.h"
// Initialize a global symbol table
//
// Parameters:
//   table    - Table to initialize (must not be NULL)
//   capacity - Initial capacity (will be rounded to power of 2)
//
// Returns:
//   SYM_OK on success
//   SYM_ERR_MEMORY on allocation failure
//
// Post-conditions:
//   - table->entries allocated with NULL pointers
//   - table->capacity set to power of 2 >= capacity
//   - table->count = 0
SymbolError global_symbol_table_init(GlobalSymbolTable *table, 
                                      size_t capacity);
// Destroy a global symbol table and free all symbols
//
// Parameters:
//   table - Table to destroy (may be NULL)
//
// Frees:
//   - All GlobalSymbol structures
//   - Entry array
void global_symbol_table_destroy(GlobalSymbolTable *table);
// Lookup a symbol by name
//
// Parameters:
//   table    - Symbol table (must be initialized)
//   name     - Symbol name to find (must not be NULL)
//   out_sym  - Output: pointer to symbol if found (may be NULL)
//
// Returns:
//   SYM_OK if found
//   SYM_ERR_NOT_FOUND if not in table
SymbolError global_symbol_lookup(GlobalSymbolTable *table,
                                  const char *name,
                                  GlobalSymbol **out_sym);
// Insert a new symbol or get existing
//
// Parameters:
//   table    - Symbol table (must be initialized)
//   name     - Symbol name (must not be NULL)
//   out_sym  - Output: pointer to inserted/existing symbol
//
// Returns:
//   SYM_OK on success
//   SYM_ERR_MEMORY if resize needed and failed
//
// If name already exists:
//   - Returns existing symbol (does not modify)
//   - out_sym points to existing entry
//
// If name is new:
//   - Creates new GlobalSymbol with state=SYM_STATE_UNDEF
//   - out_sym points to new entry
SymbolError global_symbol_insert(GlobalSymbolTable *table,
                                  const char *name,
                                  GlobalSymbol **out_sym);
// Remove a symbol from the table
//
// Parameters:
//   table - Symbol table
//   name  - Symbol name to remove
//
// Returns:
//   SYM_OK if removed
//   SYM_ERR_NOT_FOUND if not in table
//
// Note: Marks entry as TOMBSTONE, does not shrink table
SymbolError global_symbol_remove(GlobalSymbolTable *table, const char *name);
// Get all symbols as an array (for iteration)
//
// Parameters:
//   table     - Symbol table
//   out_count - Output: number of symbols
//
// Returns:
//   Array of pointers to GlobalSymbol (caller must NOT free)
//
// Note: Iterates through table to collect non-NULL entries
//       Returned array is owned by table, valid until next modification
GlobalSymbol **global_symbol_table_entries(GlobalSymbolTable *table,
                                            size_t *out_count);
// Resize the hash table
//
// Parameters:
//   table      - Table to resize
//   new_capacity - New capacity (will be rounded to power of 2)
//
// Returns:
//   SYM_OK on success
//   SYM_ERR_MEMORY on failure (table unchanged)
//
// Called automatically when load factor > 0.7
SymbolError global_symbol_table_resize(GlobalSymbolTable *table,
                                        size_t new_capacity);
#endif // GLOBAL_SYMBOLS_H
```
### Symbol Resolver Interface
```c
// File: 16_symbol_resolver.h
#ifndef SYMBOL_RESOLVER_H
#define SYMBOL_RESOLVER_H
#include "11_symbol_types.h"
#include "04_section_types.h"
// Process all symbols from all input files
//
// Parameters:
//   ctx - Linker context with inputs loaded and sections merged
//
// Returns:
//   SYM_OK on success
//   SYM_ERR_DUPLICATE_STRONG if multiple strong definitions
//   SYM_ERR_MEMORY on allocation failure
//   SYM_ERR_NO_INPUTS if ctx->input_count == 0
//
// This performs:
//   1. Parse symbols from each input file
//   2. Process local symbols (compute addresses, don't add to global)
//   3. Process global/weak symbols (add/merge in global table)
//   4. Detect duplicate strong definitions
//   5. Merge COMMON symbols by largest size
//
// After this call:
//   - All input ObjectFiles have symbols parsed
//   - ctx->global_syms is populated
//   - May have SYM_STATE_UNDEF entries (checked later)
SymbolError collect_all_symbols(LinkerContext *ctx);
// Check for undefined symbols and report errors
//
// Parameters:
//   ctx - Linker context after collect_all_symbols
//
// Returns:
//   Number of undefined symbol errors (0 = success)
//
// For each symbol with state == SYM_STATE_UNDEF:
//   - If binding == STB_WEAK: resolve to address 0 (OK)
//   - If binding == STB_GLOBAL: report error, track count
//
// Prints error messages to stderr with reference sources
int check_undefined_symbols(LinkerContext *ctx);
// Resolve all symbols to final state
//
// Parameters:
//   ctx - Linker context with symbols collected and verified
//
// Returns:
//   SYM_OK on success
//   SYM_ERR_RESOLUTION_FAILED if any symbol cannot be resolved
//
// This performs:
//   1. Allocate COMMON symbols in .bss
//   2. Translate section offsets to virtual addresses
//   3. Set state to SYM_STATE_RESOLVED for all symbols
//
// Requires:
//   - Section mapping table complete (Milestone 1)
//   - Virtual addresses assigned to output sections
SymbolError resolve_all_symbols(LinkerContext *ctx);
// Process a single local symbol
//
// Parameters:
//   ctx - Linker context
//   obj - Source object file
//   sym - Local symbol to process
//
// Returns:
//   SYM_OK on success
//   SYM_ERR_INVALID_SECTION if symbol's section is invalid
//
// Local symbols are resolved immediately using section mapping
// They do NOT go into the global symbol table
SymbolError process_local_symbol(LinkerContext *ctx,
                                  ObjectFile *obj,
                                  Symbol *sym);
// Process a single global/weak symbol definition
//
// Parameters:
//   ctx       - Linker context
//   obj       - Source object file
//   sym       - Symbol to process
//   file_idx  - Index of obj in ctx->inputs
//
// Returns:
//   SYM_OK on success
//   SYM_ERR_DUPLICATE_STRONG if conflicts with existing strong
//
// Resolution rules:
//   - Strong + Strong = ERROR
//   - Strong + Weak = Strong wins
//   - Weak + Weak = First wins (or either)
//   - Strong/Weak + Undefined = Define it
//   - COMMON + COMMON = Largest size wins
//   - Strong + COMMON = Strong wins
SymbolError process_global_symbol(LinkerContext *ctx,
                                   ObjectFile *obj,
                                   Symbol *sym,
                                   int file_idx);
// Record a reference to an undefined symbol
//
// Parameters:
//   ctx      - Linker context
//   sym      - Undefined symbol (st_shndx == SHN_UNDEF)
//   name     - Symbol name
//   file_idx - Index of file making reference
//
// Returns:
//   SYM_OK on success
//
// If symbol not in global table:
//   - Create entry with state=SYM_STATE_UNDEF
//   - Track this file as first reference
// If symbol exists:
//   - Increment ref_count
SymbolError record_symbol_reference(LinkerContext *ctx,
                                     Symbol *sym,
                                     const char *name,
                                     int file_idx);
#endif // SYMBOL_RESOLVER_H
```
### Address Assignment Interface
```c
// File: 18_address_assign.h
#ifndef ADDRESS_ASSIGN_H
#define ADDRESS_ASSIGN_H
#include "11_symbol_types.h"
#include "04_section_types.h"
// Assign final virtual addresses to all resolved symbols
//
// Parameters:
//   ctx - Linker context with symbols collected and verified
//
// Returns:
//   SYM_OK on success
//   SYM_ERR_RESOLUTION_FAILED if section mapping fails
//
// For each GlobalSymbol:
//   - If state == SYM_STATE_COMMON: allocate in .bss
//   - If state == SYM_STATE_DEFINED: translate to vaddr
//   - Set final_address field
//   - Set state to SYM_STATE_RESOLVED
SymbolError assign_symbol_addresses(LinkerContext *ctx);
// Look up a symbol's final address by name
//
// Parameters:
//   ctx       - Linker context with addresses assigned
//   name      - Symbol name
//   out_addr  - Output: final virtual address
//
// Returns:
//   SYM_OK if found and resolved
//   SYM_ERR_NOT_FOUND if not in table
//   SYM_ERR_RESOLUTION_FAILED if not yet resolved
SymbolError lookup_symbol_address(LinkerContext *ctx,
                                   const char *name,
                                   uint64_t *out_addr);
// Find the entry point symbol
//
// Parameters:
//   ctx - Linker context
//
// Returns:
//   Pointer to GlobalSymbol for entry point, or NULL if not found
//
// Lookup order:
//   1. "_start" (standard entry point)
//   2. "main" (fallback for simple programs)
GlobalSymbol *find_entry_point(LinkerContext *ctx);
// Allocate COMMON symbols in .bss section
//
// Parameters:
//   ctx - Linker context
//
// Returns:
//   SYM_OK on success
//   SYM_ERR_MEMORY if .bss creation fails
//
// For each symbol with state == SYM_STATE_COMMON:
//   - Find or create .bss output section
//   - Align to common_align
//   - Assign address in .bss
//   - Update .bss mem_size
SymbolError allocate_common_symbols(LinkerContext *ctx);
// Translate a symbol's section offset to virtual address
//
// Parameters:
//   ctx           - Linker context
//   sym           - Global symbol with section_idx and section_offset
//   source_file   - Object file containing the definition
//   out_vaddr     - Output: computed virtual address
//
// Returns:
//   SYM_OK on success
//   SYM_ERR_INVALID_SECTION if section_idx invalid
//   MAP_ERR_NOT_FOUND if section not in mapping table
//
// Formula:
//   vaddr = output_section.virtual_addr + 
//           mapping.output_offset + 
//           sym->section_offset
SymbolError translate_symbol_address(LinkerContext *ctx,
                                      GlobalSymbol *sym,
                                      ObjectFile *source_file,
                                      uint64_t *out_vaddr);
#endif // ADDRESS_ASSIGN_H
```

![Elf64_Sym Entry Memory Layout](./diagrams/tdd-diag-016.svg)

## Algorithm Specification
### Algorithm 1: Symbol Table Parsing
**Purpose**: Extract all symbols from an object file's `.symtab` section.
**Input**:
- `obj`: Object file with sections loaded
- Must have `.symtab` and `.strtab` sections
**Output**:
- `obj->symbols`: Array of parsed Symbol structures
- `obj->symbol_count`: Number of symbols
- Return: `SymbolError` code
**Procedure**:
```
1. Find symbol table section
   symtab_idx = -1
   strtab_idx = -1
   FOR i = 0 TO obj->section_count - 1:
     IF obj->sections[i].name == ".symtab":
       symtab_idx = i
     IF obj->sections[i].name == ".strtab":
       strtab_idx = i
   IF symtab_idx < 0:
     RETURN SYM_ERR_PARSE_ERROR  // No symbol table
   IF strtab_idx < 0:
     // Check sh_link from .symtab for associated strtab
     // (This is stored in original section header)
     RETURN SYM_ERR_STRTAB_MISSING
2. Get string table data
   strtab_data = obj->sections[strtab_idx].data
   strtab_size = obj->sections[strtab_idx].size
   IF strtab_data == NULL OR strtab_size == 0:
     RETURN SYM_ERR_STRTAB_MISSING
3. Calculate symbol count
   symtab_size = obj->sections[symtab_idx].size
   symbol_count = symtab_size / sizeof(Elf64_Sym)  // 24 bytes each
   IF symbol_count == 0:
     obj->symbols = NULL
     obj->symbol_count = 0
     RETURN SYM_OK
4. Allocate symbol array
   obj->symbols = calloc(symbol_count, sizeof(Symbol))
   IF obj->symbols == NULL:
     RETURN SYM_ERR_MEMORY
   obj->symbol_count = symbol_count
5. Parse each symbol entry
   raw_syms = (Elf64_Sym*)obj->sections[symtab_idx].data
   FOR i = 0 TO symbol_count - 1:
     raw = &raw_syms[i]
     sym = &obj->symbols[i]
     a. Extract name from string table
        name_offset = raw->st_name
        IF name_offset >= strtab_size:
          strncpy(sym->name, "(invalid)", MAX_SYMBOL_NAME-1)
        ELSE:
          // String is null-terminated in strtab
          char *name_start = strtab_data + name_offset
          strncpy(sym->name, name_start, MAX_SYMBOL_NAME-1)
          sym->name[MAX_SYMBOL_NAME-1] = '\0'
     b. Extract type and binding
        sym->type = ELF64_ST_TYPE(raw->st_info)
        sym->binding = ELF64_ST_BIND(raw->st_info)
        sym->visibility = raw->st_other
     c. Copy section index and value
        sym->section_idx = raw->st_shndx
        sym->value = raw->st_value
        sym->size = raw->st_size
     d. Initialize tracking fields
        sym->source_file_idx = -1  // Caller sets this
        sym->source_sym_idx = i
        sym->final_address = 0
6. RETURN SYM_OK
```
**Invariants after execution**:
- All symbols have names resolved from string table
- `source_file_idx` is -1 (caller must set)
- Symbol array is owned by ObjectFile
### Algorithm 2: Global Symbol Collection
**Purpose**: Process all symbols from all input files, building the global symbol table.
**Input**:
- `ctx`: Linker context with inputs loaded
**Output**:
- `ctx->global_syms`: Populated global symbol table
- Return: `SymbolError` code
**Procedure**:
```
1. Validate inputs exist
   IF ctx->input_count == 0:
     RETURN SYM_ERR_NO_INPUTS
2. Initialize global symbol table
   // Estimate: average 50 symbols per file
   estimated_symbols = ctx->input_count * 50
   global_symbol_table_init(&ctx->global_syms, estimated_symbols)
3. Process each input file
   FOR file_idx = 0 TO ctx->input_count - 1:
     obj = &ctx->inputs[file_idx]
     a. Parse symbols if not already parsed
        IF obj->symbols == NULL:
          err = parse_symbols(obj)
          IF err != SYM_OK:
            RETURN err
     b. Set source file index for all symbols
        FOR i = 0 TO obj->symbol_count - 1:
          obj->symbols[i].source_file_idx = file_idx
     c. Process each symbol
        FOR sym_idx = 0 TO obj->symbol_count - 1:
          sym = &obj->symbols[sym_idx]
          // Skip null symbol (index 0)
          IF sym->name[0] == '\0':
            CONTINUE
          // Skip internal symbols
          IF sym->type == STT_FILE OR sym->type == STT_SECTION:
            CONTINUE
          // Local symbols: process immediately
          IF sym->binding == STB_LOCAL:
            err = process_local_symbol(ctx, obj, sym)
            IF err != SYM_OK:
              RETURN err
            CONTINUE
          // Global/weak symbols: process for resolution
          err = process_global_symbol(ctx, obj, sym, file_idx)
          IF err != SYM_OK:
            RETURN err
4. RETURN SYM_OK
```
### Algorithm 3: Global Symbol Processing
**Purpose**: Apply ELF symbol resolution rules for one symbol.
**Input**:
- `ctx`: Linker context
- `obj`: Source object file
- `sym`: Symbol to process
- `file_idx`: Index of source file
**Output**:
- Return: `SymbolError` code
- Global symbol table updated
**Procedure**:
```
1. Handle undefined symbol reference
   IF sym->section_idx == SHN_UNDEF:
     err = record_symbol_reference(ctx, sym, sym->name, file_idx)
     RETURN err
2. Handle COMMON symbol
   IF sym->section_idx == SHN_COMMON:
     RETURN process_common_symbol(ctx, sym, file_idx)
3. Get or create global symbol entry
   err = global_symbol_insert(&ctx->global_syms, sym->name, &global)
   IF err != SYM_OK:
     RETURN err
4. Determine symbol strength
   is_strong = (sym->binding == STB_GLOBAL)
   is_weak = (sym->binding == STB_WEAK)
5. Apply resolution rules based on existing state
   SWITCH global->state:
   CASE SYM_STATE_UNDEF:
     // First definition - take it
     GOTO define_symbol
   CASE SYM_STATE_DEFINED:
     IF global->is_strong AND is_strong:
       // ERROR: Multiple strong definitions
       fprintf(stderr, "error: duplicate symbol '%s'\n", sym->name)
       fprintf(stderr, "  defined in %s\n", 
               ctx->inputs[global->source_file_idx].filename)
       fprintf(stderr, "  also defined in %s\n", obj->filename)
       RETURN SYM_ERR_DUPLICATE_STRONG
     IF is_strong AND NOT global->is_strong:
       // Strong overrides weak - replace
       GOTO define_symbol
     // Weak definition, existing is strong - keep existing
     // Weak definition, existing is weak - keep first
     RETURN SYM_OK
   CASE SYM_STATE_COMMON:
     IF is_strong:
       // Strong overrides COMMON - replace
       GOTO define_symbol
     // Otherwise keep COMMON for now
     RETURN SYM_OK
6. Define symbol (update global entry)
   LABEL define_symbol:
   global->state = SYM_STATE_DEFINED
   global->is_strong = is_strong
   global->type = sym->type
   global->binding = sym->binding
   global->size = sym->size
   global->source_file_idx = file_idx
   global->source_sym_idx = sym_idx
   global->section_idx = sym->section_idx
   global->section_offset = sym->value
   RETURN SYM_OK
```
### Algorithm 4: COMMON Symbol Processing
**Purpose**: Merge COMMON symbols by largest size.
**Input**:
- `ctx`: Linker context
- `sym`: COMMON symbol (st_shndx == SHN_COMMON)
- `file_idx`: Source file index
**Output**:
- Global symbol table updated
- Return: `SymbolError` code
**Procedure**:
```
1. Get or create global symbol entry
   err = global_symbol_insert(&ctx->global_syms, sym->name, &global)
   IF err != SYM_OK:
     RETURN err
2. Handle based on existing state
   SWITCH global->state:
   CASE SYM_STATE_UNDEF:
     // First COMMON definition
     global->state = SYM_STATE_COMMON
     global->common_size = sym->size
     global->common_align = sym->value  // Alignment stored in st_value for COMMON
     global->source_file_idx = file_idx
     // COMMON acts as strong for final resolution
     global->is_strong = 1
     RETURN SYM_OK
   CASE SYM_STATE_COMMON:
     // Merge: largest size wins
     IF sym->size > global->common_size:
       global->common_size = sym->size
       global->source_file_idx = file_idx  // Track largest source
     // Merge alignment: largest wins
     IF sym->value > global->common_align:
       global->common_align = sym->value
     RETURN SYM_OK
   CASE SYM_STATE_DEFINED:
     // Existing strong/weak definition takes precedence
     // COMMON is ignored
     RETURN SYM_OK
3. RETURN SYM_OK
```

![Symbol Table Parsing Data Flow](./diagrams/tdd-diag-017.svg)

### Algorithm 5: Undefined Symbol Detection
**Purpose**: Find all undefined symbols and report errors.
**Input**:
- `ctx`: Linker context with symbols collected
**Output**:
- Return: Count of errors (0 = success)
- Error messages printed to stderr
**Procedure**:
```
1. Initialize error count
   error_count = 0
2. Iterate through all global symbols
   entries = global_symbol_table_entries(&ctx->global_syms, &count)
   FOR i = 0 TO count - 1:
     global = entries[i]
     IF global->state != SYM_STATE_UNDEF:
       CONTINUE
     // Check if weak undefined (allowed)
     IF global->binding == STB_WEAK:
       // Weak undefined resolves to 0
       global->final_address = 0
       global->state = SYM_STATE_RESOLVED
       CONTINUE
     // Strong undefined - error
     error_count++
     fprintf(stderr, "undefined symbol: %s\n", global->name)
     // Report where it was referenced
     IF global->first_ref_file_idx >= 0:
       ref_file = ctx->inputs[global->first_ref_file_idx].filename
       fprintf(stderr, "  referenced by: %s\n", ref_file)
     // Suggest similar symbols
     suggest_similar_symbols(ctx, global->name)
3. RETURN error_count
```
**Similarity suggestion**:
```
suggest_similar_symbols(ctx, name):
   suggestions[5]
   count = 0
   entries = global_symbol_table_entries(&ctx->global_syms, &n)
   FOR i = 0 TO n - 1 AND count < 5:
     candidate = entries[i]
     IF candidate->state == SYM_STATE_UNDEF:
       CONTINUE
     // Simple similarity: common prefix length
     common = 0
     WHILE name[common] AND candidate->name[common] AND
           name[common] == candidate->name[common]:
       common++
     IF common >= 3:  // At least 3 matching chars
       suggestions[count++] = candidate->name
   IF count > 0:
     fprintf(stderr, "  did you mean: ")
     FOR i = 0 TO count - 1:
       fprintf(stderr, "%s%s", 
               i > 0 ? ", " : "", 
               suggestions[i])
     fprintf(stderr, "?\n")
```
### Algorithm 6: Final Address Assignment
**Purpose**: Compute virtual addresses for all resolved symbols.
**Input**:
- `ctx`: Linker context with symbols collected and verified
**Output**:
- All `GlobalSymbol.final_address` set
- All `GlobalSymbol.state` = `SYM_STATE_RESOLVED`
- Return: `SymbolError` code
**Procedure**:
```
1. Allocate COMMON symbols first
   err = allocate_common_symbols(ctx)
   IF err != SYM_OK:
     RETURN err
2. Assign addresses for all defined symbols
   entries = global_symbol_table_entries(&ctx->global_syms, &count)
   FOR i = 0 TO count - 1:
     global = entries[i]
     // Skip already resolved (weak undefined = 0)
     IF global->state == SYM_STATE_RESOLVED:
       CONTINUE
     // COMMON symbols handled in step 1
     IF global->state == SYM_STATE_COMMON:
       CONTINUE  // Should have been allocated
     IF global->state == SYM_STATE_DEFINED:
       a. Get source object file
          source_file = &ctx->inputs[global->source_file_idx]
       b. Translate to virtual address
          err = translate_symbol_address(ctx, global, source_file, 
                                          &global->final_address)
          IF err != SYM_OK:
            RETURN err
       c. Mark resolved
          global->state = SYM_STATE_RESOLVED
3. RETURN SYM_OK
```
### Algorithm 7: COMMON Symbol Allocation
**Purpose**: Allocate space in .bss for COMMON symbols.
**Input**:
- `ctx`: Linker context
**Output**:
- COMMON symbols allocated with addresses
- .bss section size updated
- Return: `SymbolError` code
**Procedure**:
```
1. Find or create .bss output section
   bss = find_output_section(ctx, ".bss")
   IF bss == NULL:
     bss = create_output_section(ctx, ".bss", 
                                  SHF_ALLOC | SHF_WRITE, 
                                  16)  // Default alignment
     IF bss == NULL:
       RETURN SYM_ERR_MEMORY
2. Process each COMMON symbol
   entries = global_symbol_table_entries(&ctx->global_syms, &count)
   FOR i = 0 TO count - 1:
     global = entries[i]
     IF global->state != SYM_STATE_COMMON:
       CONTINUE
     a. Align current .bss offset
        align = global->common_align
        IF align == 0:
          align = 8  // Default alignment
        current_offset = align_up(bss->mem_size, align)
     b. Assign address
        global->final_address = bss->virtual_addr + current_offset
        global->section_offset = current_offset
        global->state = SYM_STATE_RESOLVED
     c. Update .bss size
        bss->mem_size = current_offset + global->common_size
     d. Record mapping for potential debug info
        // (Optional: add mapping entry for COMMON symbol location)
3. RETURN SYM_OK
```
### Algorithm 8: Symbol Address Translation
**Purpose**: Compute final virtual address from section offset.
**Input**:
- `ctx`: Linker context
- `sym`: Global symbol with section_idx and section_offset
- `source_file`: Object file containing the definition
**Output**:
- `out_vaddr`: Computed virtual address
- Return: `SymbolError` code
**Procedure**:
```
1. Validate section index
   IF sym->section_idx >= source_file->section_count:
     RETURN SYM_ERR_INVALID_SECTION
2. Get input section
   input_sec = &source_file->sections[sym->section_idx]
   IF input_sec->name[0] == '\0':
     RETURN SYM_ERR_INVALID_SECTION
3. Find output section
   output_sec = find_output_section(ctx, input_sec->name)
   IF output_sec == NULL:
     RETURN SYM_ERR_INVALID_SECTION
4. Translate input offset to output offset
   // Use mapping table from Milestone 1
   err = translate_offset(ctx, 
                          source_file->filename, 
                          input_sec->name,
                          sym->section_offset,  // st_value
                          &output_offset)
   IF err != MAP_OK:
     RETURN SYM_ERR_RESOLUTION_FAILED
5. Compute final virtual address
   *out_vaddr = output_sec->virtual_addr + output_offset
   RETURN SYM_OK
```

![Symbol Binding and Type Classification](./diagrams/tdd-diag-018.svg)

## Error Handling Matrix
| Error | Detected By | Recovery | User-Visible? | System State |
|-------|-------------|----------|---------------|--------------|
| `SYM_ERR_MEMORY` | `malloc()`/`calloc()` returns NULL | Abort linking, cleanup | Yes: "Out of memory during symbol resolution" | Clean, all prior allocations freed |
| `SYM_ERR_DUPLICATE_STRONG` | Resolution rules check | Abort linking | Yes: "duplicate symbol 'foo'\n defined in a.o\n also defined in b.o" | Clean, global table intact |
| `SYM_ERR_UNDEFINED` | `check_undefined_symbols()` | Abort linking | Yes: "undefined symbol: 'bar'\n referenced by: main.o" | Clean, symbols collected but not resolved |
| `SYM_ERR_PARSE_ERROR` | Symbol table parsing | Abort linking | Yes: "file.o: Cannot parse symbol table" | File handle closed, no allocations |
| `SYM_ERR_STRTAB_MISSING` | Missing `.strtab` section | Abort linking | Yes: "file.o: No string table for symbol names" | Clean |
| `SYM_ERR_INVALID_INDEX` | Symbol index validation | Return error to caller | No (internal error) | Unchanged |
| `SYM_ERR_INVALID_SECTION` | Section index validation | Return error to caller | Yes: "file.o: Symbol 'foo' has invalid section index" | Unchanged |
| `SYM_ERR_NO_INPUTS` | Input count check | Abort linking | Yes: "No input files to link" | Clean |
| `SYM_ERR_NOT_FOUND` | Hash table lookup | Return to caller | No (caller handles) | Unchanged |
| `SYM_ERR_RESOLUTION_FAILED` | Address translation failure | Abort linking | Yes: "Failed to resolve symbol 'foo' address" | Clean |
### Error Message Format
```c
// Duplicate strong symbol
fprintf(stderr, "ld: error: duplicate symbol: %s\n", name);
fprintf(stderr, ">>> defined in %s\n", first_file);
fprintf(stderr, ">>> defined in %s\n", second_file);
// Undefined symbol
fprintf(stderr, "ld: error: undefined symbol: %s\n", name);
fprintf(stderr, ">>> referenced by %s\n", ref_file);
if (similar_count > 0) {
    fprintf(stderr, ">>> did you mean: %s?\n", similar_list);
}
// Parse error
fprintf(stderr, "ld: error: %s: %s\n", filename, description);
```
## Implementation Sequence with Checkpoints
### Phase 1: Symbol Table Parsing (2-3 hours)
**Files**: `11_symbol_types.h`, `12_symbol_parser.h`, `13_symbol_parser.c`
**Implementation steps**:
1. Define all symbol constants and structures in `11_symbol_types.h`
2. Declare parser interface in `12_symbol_parser.h`
3. Implement `find_symbol_tables()` to locate `.symtab` and `.strtab`
4. Implement `parse_symbols()` with full Elf64_Sym parsing
5. Implement helper functions (`symbol_is_defined`, etc.)
6. Write unit tests for parsing
**Checkpoint**:
```bash
gcc -c 13_symbol_parser.c -o symbol_parser.o
gcc tests/test_symbol_parse.c symbol_parser.o -o test_symbol_parse
./test_symbol_parse
# Expected output:
# [PASS] Symbol table location
# [PASS] Symbol name resolution
# [PASS] Symbol type/binding extraction
# [PASS] Null symbol handling
# [PASS] Local symbol detection
# [PASS] Global symbol detection
# [PASS] Weak symbol detection
# [PASS] COMMON symbol detection
# All tests passed!
```
At this point you can parse symbol tables but not resolve them.
### Phase 2: Global Symbol Table Construction (2-3 hours)
**Files**: `14_global_symbols.h`, `15_global_symbols.c`
**Implementation steps**:
1. Implement hash function (`hash_string`)
2. Implement `global_symbol_table_init()` and `_destroy()`
3. Implement `global_symbol_lookup()` with linear probing
4. Implement `global_symbol_insert()` with duplicate detection
5. Implement automatic resize when load factor > 0.7
6. Implement `global_symbol_remove()` with tombstone marking
7. Write unit tests for hash table operations
**Checkpoint**:
```bash
gcc -c 15_global_symbols.c -o global_symbols.o
gcc tests/test_hashtable.c global_symbols.o -o test_hashtable
./test_hashtable
# Expected output:
# [PASS] Table initialization
# [PASS] Basic insert and lookup
# [PASS] Duplicate insert returns existing
# [PASS] Not found returns error
# [PASS] Hash collision handling
# [PASS] Resize preserves entries
# [PASS] Remove and tombstone
# [PASS] Load factor calculation
# All tests passed!
```
At this point you have a working hash table for symbols.
### Phase 3: Symbol Resolution Rules (3-4 hours)
**Files**: `16_symbol_resolver.h`, `17_symbol_resolver.c`
**Implementation steps**:
1. Implement `collect_all_symbols()` orchestration
2. Implement `process_local_symbol()` with section offset translation
3. Implement `process_global_symbol()` with state machine
4. Implement `process_common_symbol()` with size merging
5. Implement `record_symbol_reference()` for undefined tracking
6. Add duplicate strong detection
7. Write tests for each resolution rule
**Checkpoint**:
```bash
# Create test object files
cat > strong1.c << 'EOF'
int symbol = 1;
EOF
cat > strong2.c << 'EOF'
int symbol = 2;
EOF
cat > weak.c << 'EOF'
__attribute__((weak)) int symbol = 3;
EOF
gcc -c strong1.c strong2.c weak.c
./test_resolution
# Expected output:
# [PASS] Strong + Undefined -> Strong defines
# [PASS] Weak + Undefined -> Weak defines
# [PASS] Strong + Weak -> Strong wins
# [PASS] Weak + Strong -> Strong wins
# [PASS] Weak + Weak -> First wins
# [PASS] Strong + Strong -> Error detected
# [PASS] COMMON + COMMON -> Largest wins
# [PASS] Strong + COMMON -> Strong wins
# [PASS] Local symbol scoping
# All tests passed!
```
At this point resolution rules are complete.
### Phase 4: Undefined Symbol Detection (1-2 hours)
**Files**: Continue `17_symbol_resolver.c`
**Implementation steps**:
1. Implement `check_undefined_symbols()` scan
2. Implement weak undefined handling (resolve to 0)
3. Implement error reporting with reference sources
4. Implement similarity suggestion
5. Write tests for undefined detection
**Checkpoint**:
```bash
cat > undef.c << 'EOF'
extern int missing_symbol;
int main() { return missing_symbol; }
EOF
gcc -c undef.c
./test_resolution
# Expected output:
# [PASS] Undefined symbol detection
# [PASS] Weak undefined resolves to 0
# [PASS] Error message includes reference source
# [PASS] Similarity suggestion
# [PASS] Multiple undefined symbols
# All tests passed!
```
At this point undefined symbol detection is complete.
### Phase 5: Final Address Assignment (2-3 hours)
**Files**: `18_address_assign.h`, `19_address_assign.c`
**Implementation steps**:
1. Implement `allocate_common_symbols()` in .bss
2. Implement `translate_symbol_address()` using mapping table
3. Implement `assign_symbol_addresses()` orchestration
4. Implement `lookup_symbol_address()` for relocation use
5. Implement `find_entry_point()` for `_start`/`main`
6. Write integration tests with mock mapping table
**Checkpoint**:
```bash
# Create complete test with multiple files
cat > file1.c << 'EOF'
int global_var = 42;
int get_var() { return global_var; }
EOF
cat > file2.c << 'EOF'
extern int get_var();
int main() { return get_var(); }
EOF
gcc -c file1.c file2.c
# Link with real section merger + symbol resolver
./test_address_assign
# Expected output:
# [PASS] COMMON allocation in .bss
# [PASS] Symbol address translation
# [PASS] Entry point detection
# [PASS] Local symbol addresses
# [PASS] Cross-file symbol resolution
# 
# Symbol table:
#   main: 0x401020 (FUNC)
#   get_var: 0x401000 (FUNC)
#   global_var: 0x403000 (OBJECT)
# All tests passed!
```
**Milestone 2 Complete**: Global symbol table with all symbols resolved to virtual addresses.

![Strong/Weak Symbol Resolution State Machine](./diagrams/tdd-diag-019.svg)

## Test Specification
### Test Suite: Symbol Table Parsing
```c
// tests/test_symbol_parse.c
void test_find_symbol_tables(void) {
    system("echo 'int x;' | gcc -c -x c - -o /tmp/symtest.o");
    ObjectFile obj;
    parse_object_file("/tmp/symtest.o", &obj);
    int symtab_idx, strtab_idx;
    SymbolError err = find_symbol_tables(&obj, &symtab_idx, &strtab_idx);
    ASSERT_EQ(err, SYM_OK);
    ASSERT_TRUE(symtab_idx >= 0);
    ASSERT_TRUE(strtab_idx >= 0);
    object_file_destroy(&obj);
}
void test_parse_symbols_basic(void) {
    system("as -o /tmp/syms.o << 'EOF'\n"
           ".globl global_func\n"
           "global_func:\n"
           "    ret\n"
           ".globl global_var\n"
           ".data\n"
           "global_var:\n"
           "    .int 42\n"
           "local_label:\n"
           "    .int 0\n"
           "EOF");
    ObjectFile obj;
    parse_object_file("/tmp/syms.o", &obj);
    SymbolError err = parse_symbols(&obj);
    ASSERT_EQ(err, SYM_OK);
    ASSERT_TRUE(obj.symbol_count > 0);
    // Find global_func
    Symbol *func = NULL;
    for (int i = 0; i < obj.symbol_count; i++) {
        if (strcmp(obj.symbols[i].name, "global_func") == 0) {
            func = &obj.symbols[i];
            break;
        }
    }
    ASSERT_NOT_NULL(func);
    ASSERT_EQ(func->binding, STB_GLOBAL);
    ASSERT_EQ(func->type, STT_FUNC);
    ASSERT_TRUE(symbol_is_defined(func));
    ASSERT_TRUE(symbol_is_strong(func));
    object_file_destroy(&obj);
}
void test_parse_undefined_symbol(void) {
    system("echo 'extern int missing; int get() { return missing; }' | "
           "gcc -c -x c - -o /tmp/undef.o");
    ObjectFile obj;
    parse_object_file("/tmp/undef.o", &obj);
    parse_symbols(&obj);
    // Find 'missing' - should be undefined
    Symbol *missing = NULL;
    for (int i = 0; i < obj.symbol_count; i++) {
        if (strcmp(obj.symbols[i].name, "missing") == 0) {
            missing = &obj.symbols[i];
            break;
        }
    }
    ASSERT_NOT_NULL(missing);
    ASSERT_EQ(missing->section_idx, SHN_UNDEF);
    ASSERT_FALSE(symbol_is_defined(missing));
    object_file_destroy(&obj);
}
void test_parse_common_symbol(void) {
    system("echo 'int buffer[100];' | gcc -c -x c - -o /tmp/common.o");
    ObjectFile obj;
    parse_object_file("/tmp/common.o", &obj);
    parse_symbols(&obj);
    // Find 'buffer' - should be COMMON
    Symbol *buffer = NULL;
    for (int i = 0; i < obj.symbol_count; i++) {
        if (strcmp(obj.symbols[i].name, "buffer") == 0) {
            buffer = &obj.symbols[i];
            break;
        }
    }
    ASSERT_NOT_NULL(buffer);
    ASSERT_TRUE(symbol_is_common(buffer));
    ASSERT_EQ(buffer->size, 400);  // 100 * sizeof(int)
    object_file_destroy(&obj);
}
void test_parse_weak_symbol(void) {
    system("echo '__attribute__((weak)) int weak_fn() { return 0; }' | "
           "gcc -c -x c - -o /tmp/weak.o");
    ObjectFile obj;
    parse_object_file("/tmp/weak.o", &obj);
    parse_symbols(&obj);
    Symbol *weak = NULL;
    for (int i = 0; i < obj.symbol_count; i++) {
        if (strcmp(obj.symbols[i].name, "weak_fn") == 0) {
            weak = &obj.symbols[i];
            break;
        }
    }
    ASSERT_NOT_NULL(weak);
    ASSERT_TRUE(symbol_is_weak(weak));
    ASSERT_TRUE(symbol_is_defined(weak));
    ASSERT_FALSE(symbol_is_strong(weak));
    object_file_destroy(&obj);
}
```
### Test Suite: Global Symbol Table
```c
// tests/test_hashtable.c
void test_basic_insert_lookup(void) {
    GlobalSymbolTable table;
    ASSERT_EQ(global_symbol_table_init(&table, 16), SYM_OK);
    GlobalSymbol *sym;
    ASSERT_EQ(global_symbol_insert(&table, "test", &sym), SYM_OK);
    ASSERT_NOT_NULL(sym);
    ASSERT_STREQ(sym->name, "test");
    ASSERT_EQ(sym->state, SYM_STATE_UNDEF);
    GlobalSymbol *found;
    ASSERT_EQ(global_symbol_lookup(&table, "test", &found), SYM_OK);
    ASSERT_EQ(found, sym);
    global_symbol_table_destroy(&table);
}
void test_duplicate_insert(void) {
    GlobalSymbolTable table;
    global_symbol_table_init(&table, 16);
    GlobalSymbol *sym1, *sym2;
    global_symbol_insert(&table, "foo", &sym1);
    sym1->state = SYM_STATE_DEFINED;
    ASSERT_EQ(global_symbol_insert(&table, "foo", &sym2), SYM_OK);
    ASSERT_EQ(sym1, sym2);  // Same pointer - existing returned
    ASSERT_EQ(sym2->state, SYM_STATE_DEFINED);  // Unchanged
    global_symbol_table_destroy(&table);
}
void test_not_found(void) {
    GlobalSymbolTable table;
    global_symbol_table_init(&table, 16);
    GlobalSymbol *found;
    ASSERT_EQ(global_symbol_lookup(&table, "nonexistent", &found), 
              SYM_ERR_NOT_FOUND);
    global_symbol_table_destroy(&table);
}
void test_hash_collisions(void) {
    GlobalSymbolTable table;
    global_symbol_table_init(&table, 4);  // Small table forces collisions
    // Insert more entries than capacity
    GlobalSymbol *syms[10];
    for (int i = 0; i < 10; i++) {
        char name[32];
        snprintf(name, sizeof(name), "symbol_%d", i);
        ASSERT_EQ(global_symbol_insert(&table, name, &syms[i]), SYM_OK);
    }
    // All should be findable
    for (int i = 0; i < 10; i++) {
        char name[32];
        snprintf(name, sizeof(name), "symbol_%d", i);
        GlobalSymbol *found;
        ASSERT_EQ(global_symbol_lookup(&table, name, &found), SYM_OK);
        ASSERT_EQ(found, syms[i]);
    }
    global_symbol_table_destroy(&table);
}
void test_auto_resize(void) {
    GlobalSymbolTable table;
    global_symbol_table_init(&table, 4);
    size_t initial_capacity = table.capacity;
    // Insert many symbols to trigger resize
    for (int i = 0; i < 20; i++) {
        char name[32];
        snprintf(name, sizeof(name), "sym_%d", i);
        GlobalSymbol *sym;
        global_symbol_insert(&table, name, &sym);
    }
    // Capacity should have increased
    ASSERT_TRUE(table.capacity > initial_capacity);
    // All symbols still findable
    for (int i = 0; i < 20; i++) {
        char name[32];
        snprintf(name, sizeof(name), "sym_%d", i);
        GlobalSymbol *found;
        ASSERT_EQ(global_symbol_lookup(&table, name, &found), SYM_OK);
    }
    global_symbol_table_destroy(&table);
}
```
### Test Suite: Symbol Resolution Rules
```c
// tests/test_resolution.c
void test_strong_overrides_weak(void) {
    system("as -o /tmp/weak_def.o << 'EOF'\n"
           ".weak symbol\n"
           "symbol:\n"
           "    mov $1, %eax\n"
           "    ret\n"
           "EOF");
    system("as -o /tmp/strong_def.o << 'EOF'\n"
           ".globl symbol\n"
           "symbol:\n"
           "    mov $2, %eax\n"
           "    ret\n"
           "EOF");
    LinkerContext ctx;
    linker_context_init(&ctx);
    ObjectFile weak_obj, strong_obj;
    parse_object_file("/tmp/weak_def.o", &weak_obj);
    parse_object_file("/tmp/strong_def.o", &strong_obj);
    linker_add_input(&ctx, &weak_obj);   // Weak first
    linker_add_input(&ctx, &strong_obj); // Strong second
    collect_all_symbols(&ctx);
    GlobalSymbol *sym;
    global_symbol_lookup(&ctx.global_syms, "symbol", &sym);
    ASSERT_EQ(sym->state, SYM_STATE_DEFINED);
    ASSERT_TRUE(sym->is_strong);
    // Should point to strong definition
    ASSERT_EQ(sym->source_file_idx, 1);
    linker_context_destroy(&ctx);
}
void test_duplicate_strong_error(void) {
    system("as -o /tmp/strong1.o << 'EOF'\n"
           ".globl conflict\n"
           "conflict:\n"
           "    ret\n"
           "EOF");
    system("as -o /tmp/strong2.o << 'EOF'\n"
           ".globl conflict\n"
           "conflict:\n"
           "    nop\n"
           "    ret\n"
           "EOF");
    LinkerContext ctx;
    linker_context_init(&ctx);
    ObjectFile obj1, obj2;
    parse_object_file("/tmp/strong1.o", &obj1);
    parse_object_file("/tmp/strong2.o", &obj2);
    linker_add_input(&ctx, &obj1);
    linker_add_input(&ctx, &obj2);
    SymbolError err = collect_all_symbols(&ctx);
    ASSERT_EQ(err, SYM_ERR_DUPLICATE_STRONG);
    linker_context_destroy(&ctx);
}
void test_common_merge_largest(void) {
    system("echo 'int buffer[100];' | gcc -c -x c - -o /tmp/common1.o");
    system("echo 'int buffer[200];' | gcc -c -x c - -o /tmp/common2.o");
    LinkerContext ctx;
    linker_context_init(&ctx);
    ObjectFile obj1, obj2;
    parse_object_file("/tmp/common1.o", &obj1);
    parse_object_file("/tmp/common2.o", &obj2);
    linker_add_input(&ctx, &obj1);
    linker_add_input(&ctx, &obj2);
    ASSERT_EQ(collect_all_symbols(&ctx), SYM_OK);
    GlobalSymbol *sym;
    global_symbol_lookup(&ctx.global_syms, "buffer", &sym);
    ASSERT_EQ(sym->state, SYM_STATE_COMMON);
    ASSERT_EQ(sym->common_size, 800);  // 200 * 4 (largest)
    linker_context_destroy(&ctx);
}
void test_local_symbol_scoping(void) {
    system("as -o /tmp/local1.o << 'EOF'\n"
           ".local helper\n"
           "helper:\n"
           "    ret\n"
           "EOF");
    system("as -o /tmp/local2.o << 'EOF'\n"
           ".local helper\n"
           "helper:\n"
           "    nop\n"
           "    ret\n"
           "EOF");
    LinkerContext ctx;
    linker_context_init(&ctx);
    ObjectFile obj1, obj2;
    parse_object_file("/tmp/local1.o", &obj1);
    parse_object_file("/tmp/local2.o", &obj2);
    linker_add_input(&ctx, &obj1);
    linker_add_input(&ctx, &obj2);
    // Should NOT error - locals are scoped to file
    ASSERT_EQ(collect_all_symbols(&ctx), SYM_OK);
    // 'helper' should NOT be in global table (local symbols)
    GlobalSymbol *sym;
    ASSERT_EQ(global_symbol_lookup(&ctx.global_syms, "helper", &sym),
              SYM_ERR_NOT_FOUND);
    linker_context_destroy(&ctx);
}
void test_weak_undefined_resolves_zero(void) {
    system("as -o /tmp/weak_ref.o << 'EOF'\n"
           ".weak optional\n"
           "optional:\n"
           "    ret\n"
           ".globl check\n"
           "check:\n"
           "    mov optional(%rip), %eax\n"
           "    ret\n"
           "EOF");
    LinkerContext ctx;
    linker_context_init(&ctx);
    ObjectFile obj;
    parse_object_file("/tmp/weak_ref.o", &obj);
    linker_add_input(&ctx, &obj);
    collect_all_symbols(&ctx);
    int errors = check_undefined_symbols(&ctx);
    ASSERT_EQ(errors, 0);  // No error for weak undefined
    GlobalSymbol *sym;
    global_symbol_lookup(&ctx.global_syms, "optional", &sym);
    // If it's undefined and weak, it should resolve to 0
    if (sym->state == SYM_STATE_UNDEF && sym->binding == STB_WEAK) {
        // Will be set to 0 during check_undefined_symbols
    }
    linker_context_destroy(&ctx);
}
```
### Test Suite: Address Assignment
```c
// tests/test_address_assign.c
void test_entry_point_detection(void) {
    system("as -o /tmp/start.o << 'EOF'\n"
           ".globl _start\n"
           "_start:\n"
           "    mov $60, %rax\n"
           "    syscall\n"
           "EOF");
    LinkerContext ctx;
    linker_context_init(&ctx);
    ObjectFile obj;
    parse_object_file("/tmp/start.o", &obj);
    linker_add_input(&ctx, &obj);
    linker_merge_sections(&ctx);
    linker_assign_addresses(&ctx);
    collect_all_symbols(&ctx);
    assign_symbol_addresses(&ctx);
    GlobalSymbol *entry = find_entry_point(&ctx);
    ASSERT_NOT_NULL(entry);
    ASSERT_STREQ(entry->name, "_start");
    ASSERT_EQ(entry->state, SYM_STATE_RESOLVED);
    ASSERT_TRUE(entry->final_address > 0);
    linker_context_destroy(&ctx);
}
void test_main_fallback_entry(void) {
    system("echo 'int main() { return 0; }' | gcc -c -x c - -o /tmp/main.o");
    LinkerContext ctx;
    linker_context_init(&ctx);
    ObjectFile obj;
    parse_object_file("/tmp/main.o", &obj);
    linker_add_input(&ctx, &obj);
    linker_merge_sections(&ctx);
    linker_assign_addresses(&ctx);
    collect_all_symbols(&ctx);
    assign_symbol_addresses(&ctx);
    GlobalSymbol *entry = find_entry_point(&ctx);
    ASSERT_NOT_NULL(entry);
    ASSERT_STREQ(entry->name, "main");  // Falls back to main
    linker_context_destroy(&ctx);
}
void test_address_translation(void) {
    // Create test with known layout
    system("as -o /tmp/trans.o << 'EOF'\n"
           ".section .text\n"
           ".globl func1\n"
           "func1:\n"
           "    ret\n"
           ".globl func2\n"
           "func2:\n"
           "    nop\n"
           "    ret\n"
           ".section .data\n"
           ".globl var1\n"
           "var1:\n"
           "    .int 42\n"
           "EOF");
    LinkerContext ctx;
    linker_context_init(&ctx);
    ctx.base_address = 0x400000;
    ObjectFile obj;
    parse_object_file("/tmp/trans.o", &obj);
    linker_add_input(&ctx, &obj);
    linker_merge_sections(&ctx);
    linker_assign_addresses(&ctx);
    collect_all_symbols(&ctx);
    assign_symbol_addresses(&ctx);
    // Verify addresses are in expected ranges
    GlobalSymbol *func1, *func2, *var1;
    global_symbol_lookup(&ctx.global_syms, "func1", &func1);
    global_symbol_lookup(&ctx.global_syms, "func2", &func2);
    global_symbol_lookup(&ctx.global_syms, "var1", &var1);
    ASSERT_TRUE(func1->final_address >= 0x401000);
    ASSERT_TRUE(func2->final_address > func1->final_address);
    ASSERT_TRUE(var1->final_address >= 0x403000);  // .data after .text
    linker_context_destroy(&ctx);
}
void test_common_allocation(void) {
    system("echo 'int buffer[100];' | gcc -c -x c - -o /tmp/comm.o");
    LinkerContext ctx;
    linker_context_init(&ctx);
    ctx.base_address = 0x400000;
    ObjectFile obj;
    parse_object_file("/tmp/comm.o", &obj);
    linker_add_input(&ctx, &obj);
    linker_merge_sections(&ctx);
    linker_assign_addresses(&ctx);
    collect_all_symbols(&ctx);
    // Before allocation, buffer is COMMON
    GlobalSymbol *buffer;
    global_symbol_lookup(&ctx.global_syms, "buffer", &buffer);
    ASSERT_EQ(buffer->state, SYM_STATE_COMMON);
    // Allocate
    assign_symbol_addresses(&ctx);
    // After allocation, buffer is RESOLVED with address in .bss
    ASSERT_EQ(buffer->state, SYM_STATE_RESOLVED);
    ASSERT_TRUE(buffer->final_address > 0);
    linker_context_destroy(&ctx);
}
```
## Performance Targets
| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| Parse symbols (per file) | < 0.5ms per 1000 symbols | `gettimeofday()` around parse_symbols() |
| Hash table insert | < 50ns average | Microbenchmark with 1M inserts |
| Hash table lookup | < 30ns average | Microbenchmark with 1M lookups |
| Full resolution (1000 files) | < 3 seconds | End-to-end timing |
| Memory per symbol | < 150 bytes | `/proc/self/status` RSS delta |
| Hash table load factor | 0.3 - 0.7 | `count / capacity` ratio |
### Memory Budget
```
Per input symbol (Symbol struct):
  Name buffer:           256 bytes
  Other fields:          ~40 bytes
  Total:                 ~300 bytes
Per global symbol (GlobalSymbol struct):
  Name buffer:           256 bytes
  Other fields:          ~100 bytes
  Total:                 ~350 bytes
Hash table overhead:
  Per entry pointer:     8 bytes
  Tombstones:            ~10% of capacity
  Load factor target:    0.5 (2x capacity)
Example for 1000-file project with 50,000 symbols:
  Input symbols:         50,000 * 300 = 15 MB
  Global symbols:        10,000 * 350 = 3.5 MB (unique names)
  Hash table:            20,000 * 8 = 160 KB
  Total:                 ~19 MB
```
## Integration Notes
### Dependencies
- **Milestone 1**: Requires `LinkerContext`, `ObjectFile`, `OutputSection`, mapping table functions
- **Standard library**: `<stdint.h>`, `<stdio.h>`, `<stdlib.h>`, `<string.h>`, `<stdbool.h>`
- **No external libraries**
### API for Milestone 3 (Relocation Processing)
The Relocation Processing module will:
1. Call `lookup_symbol_address(ctx, name, &addr)` to get symbol addresses
2. Call `global_symbol_lookup()` for detailed symbol information
3. Use `find_entry_point()` to set `e_entry` in executable header
### API for Milestone 4 (Executable Generation)
The Executable Generation module will:
1. Call `find_entry_point(ctx)` to get the entry point symbol
2. Read `entry->final_address` for `e_entry` field
3. Access global symbol table for symbol dump (optional)
### Thread Safety
The symbol resolution module is single-threaded by design. The resolution rules require ordered processing of input files. If parallel parsing is desired in the future:
- Symbol table parsing per file can be parallelized
- Global table insertion must be serialized (requires locking)
- Address assignment must be single-threaded (depends on complete table)

![Duplicate Symbol Detection Algorithm](./diagrams/tdd-diag-020.svg)

[[CRITERIA_JSON: {"module_id": "build-linker-m2", "criteria": ["Parse .symtab section from each input object file extracting Elf64_Sym entries with st_name, st_info, st_other, st_shndx, st_value, st_size fields", "Resolve symbol names from .strtab string table using st_name as byte offset into string table data", "Extract symbol type (STT_*) and binding (STB_*) from st_info field using ELF64_ST_TYPE and ELF64_ST_BIND macros", "Build hash-table-backed global symbol table with O(1) average lookup by symbol name using open addressing with linear probing", "Handle local symbols (STB_LOCAL) by resolving to section offset immediately, not entering them into the global symbol table", "Process undefined symbols (st_shndx == SHN_UNDEF) by creating SYM_STATE_UNDEF entries and tracking reference sources", "Detect duplicate strong symbol definitions (STB_GLOBAL with same name in multiple files) and report as errors with source file attribution", "Implement weak symbol resolution: STB_WEAK definitions overridden by STB_GLOBAL, multiple STB_WEAK coexist with first-wins semantics", "Handle COMMON symbols (st_shndx == SHN_COMMON) by merging with largest-size-wins rule and storing alignment from st_value", "Detect and report non-weak undefined symbols after all input files processed, with reference source attribution", "Allocate COMMON symbols in .bss output section with proper alignment before address assignment", "Assign final virtual addresses to resolved symbols using formula: output_section.vaddr + mapping.output_offset + symbol.section_offset", "Support lookup by symbol name returning final virtual address for relocation processing use", "Handle weak undefined symbols by resolving to address 0 without error", "Implement automatic hash table resize when load factor exceeds 0.7", "Track reference counts for each global symbol to support 'defined but unused' warnings"]}]
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: build-linker-m3 -->
# Relocation Processing: Technical Design Specification
## Module Charter
The Relocation Processing module is the linker's most delicate operation: patching binary code with computed addresses to transform position-independent object files into fixed-address executables. This module parses `.rela.*` sections from input object files, translates relocation site offsets from input section coordinates to output buffer coordinates using the mapping table from Milestone 1, resolves target symbols via the global symbol table from Milestone 2, computes patch values using architecture-specific formulas (absolute for `R_X86_64_64`, PC-relative for `R_X86_64_PC32`), and writes the computed values into the merged section data buffer with overflow detection. It does NOT parse object files, merge sections, resolve symbol definitions, or generate executable headers—its sole responsibility is applying address patches to already-merged data.
Upstream, the module depends on the section mapping table (for offset translation), the global symbol table with all symbols resolved to virtual addresses, and the merged output buffer containing concatenated section data. Downstream, it feeds the Executable Generation module with fully-patched section data ready for ELF wrapping. The module maintains strict invariants: all relocations are processed only after complete symbol resolution, no relocation site exceeds its section boundary, overflow is always detected before writing, and the output buffer remains consistent even on error (partial patches are not written).
## File Structure
```
linker/
├── 21_relocation_types.h      # Relocation constants and data structures
├── 22_relocation_parser.h     # Relocation section parsing interface
├── 23_relocation_parser.c     # .rela.* section parsing implementation
├── 24_relocation_resolver.h   # Symbol resolution for relocations interface
├── 25_relocation_resolver.c   # Target symbol lookup implementation
├── 26_patch_calculator.h      # Patch value computation interface
├── 27_patch_calculator.c      # R_X86_64_64 and R_X86_64_PC32 formulas
├── 28_patch_writer.h          # Output buffer patching interface
├── 29_patch_writer.c          # Little-endian write with bounds check
├── 30_relocation_processor.h  # Main processing orchestration interface
├── 31_relocation_processor.c  # Full relocation processing pipeline
├── 32_relocation_main.c       # Test driver entry point
└── tests/
    ├── test_reloc_parse.c     # Relocation entry parsing tests
    ├── test_reloc_calc.c      # Patch calculation tests
    ├── test_reloc_write.c     # Buffer patching tests
    ├── test_reloc_full.c      # End-to-end relocation tests
    └── fixtures/
        ├── call.o             # PC-relative call relocation
        ├── data_ref.o         # Absolute data reference
        ├── overflow.o         # Relocation overflow case
        └── multi.o            # Multiple relocation types
```
## Complete Data Model
### ELF64 Relocation Entry Structure
The ELF64 relocation entry with explicit addend (`Elf64_Rela`) is 24 bytes:
```c
// File: 21_relocation_types.h
#ifndef RELOCATION_TYPES_H
#define RELOCATION_TYPES_H
#include <stdint.h>
#include <stddef.h>
// x86-64 relocation types
#define R_X86_64_NONE       0   // No relocation
#define R_X86_64_64         1   // Direct 64-bit absolute address
#define R_X86_64_PC32       2   // PC-relative 32-bit signed
#define R_X86_64_GOT32      3   // GOT-relative (not implemented)
#define R_X86_64_PLT32      4   // PLT-relative (not implemented)
#define R_X86_64_32         10  // Direct 32-bit zero-extended
#define R_X86_64_32S        11  // Direct 32-bit sign-extended
#define R_X86_64_16         12  // Direct 16-bit zero-extended
#define R_X86_64_8          14  // Direct 8-bit
#define R_X86_64_PC64       24  // PC-relative 64-bit
#define R_X86_64_PC32       2   // Alias for clarity
// Macros to extract symbol index and type from r_info
#define ELF64_R_SYM(i)      ((i) >> 32)
#define ELF64_R_TYPE(i)     ((i) & 0xFFFFFFFF)
#define ELF64_R_INFO(s, t)  (((uint64_t)(s) << 32) | ((t) & 0xFFFFFFFF))
// ELF64 Relocation entry with explicit addend (RELA) - 24 bytes
typedef struct __attribute__((packed)) {
    uint64_t r_offset;    // +0x00: Offset within target section
    uint64_t r_info;      // +0x08: Symbol index (high 32) + type (low 32)
    int64_t  r_addend;    // +0x10: Constant addend for calculation
} Elf64_Rela;             // Total: 24 bytes (0x18)
// ELF64 Relocation entry with implicit addend (REL) - 16 bytes
// Addend is stored at the relocation site in the section data
typedef struct __attribute__((packed)) {
    uint64_t r_offset;    // +0x00: Offset within target section
    uint64_t r_info;      // +0x08: Symbol index + type
} Elf64_Rel;              // Total: 16 bytes (0x10)
// Maximum sizes for arrays
#define MAX_RELOCATIONS_PER_FILE   10000
#define MAX_RELOCATION_NAME        32
// Parsed relocation entry (internal representation)
typedef struct {
    uint64_t offset;           // r_offset: site within input section
    uint32_t sym_idx;          // Symbol index in input file's .symtab
    uint32_t type;             // R_X86_64_* relocation type
    int64_t addend;            // r_addend: value to add to symbol address
    int target_section_idx;    // Which input section this reloc targets
    char source_file[256];     // Source filename for error reporting
    char target_section[64];   // Target section name
} Relocation;
// Resolved relocation context (after symbol lookup)
typedef struct {
    Relocation *reloc;         // Original relocation entry
    uint64_t site_vaddr;       // Virtual address of relocation site
    uint64_t site_file_offset; // File offset in output buffer
    uint64_t symbol_vaddr;     // Target symbol's virtual address
    int64_t computed_value;    // Final patch value (before truncation)
    size_t patch_size;         // Bytes to write (4 or 8)
    int is_valid;              // 1 if computation succeeded
    char error_msg[256];       // Error description if is_valid == 0
} ResolvedRelocation;
// Relocation processing statistics
typedef struct {
    size_t total_count;        // Total relocations processed
    size_t success_count;      // Successfully applied
    size_t skipped_count;      // Skipped (R_X86_64_NONE)
    size_t error_count;        // Errors encountered
    size_t overflow_count;     // Overflow errors specifically
} RelocationStats;
// Error codes for relocation operations
typedef enum {
    RELOC_OK = 0,
    RELOC_ERR_MEMORY,              // Memory allocation failed
    RELOC_ERR_PARSE_ERROR,         // Cannot parse relocation entry
    RELOC_ERR_TYPE_UNSUPPORTED,    // Unknown or unimplemented relocation type
    RELOC_ERR_OVERFLOW,            // Computed value exceeds field size
    RELOC_ERR_SYMBOL_INVALID,      // Symbol index out of range
    RELOC_ERR_SYMBOL_UNRESOLVED,   // Symbol not in global table
    RELOC_ERR_TARGET_INVALID,      // Target section not found
    RELOC_ERR_TARGET_NOBITS,       // Cannot relocate into .bss
    RELOC_ERR_OFFSET_OUT_OF_RANGE, // r_offset exceeds section size
    RELOC_ERR_MAPPING_NOT_FOUND,   // No mapping for input section
    RELOC_ERR_WRITE_FAILED,        // Cannot write to output buffer
    RELOC_ERR_SECTION_SYMBOL,      // Cannot resolve section symbol
    RELOC_ERR_NO_INPUTS            // No input files to process
} RelocError;
// Section type for relocation section detection
typedef enum {
    RELOC_SECTION_NONE,
    RELOC_SECTION_REL,     // .rel.* - implicit addend
    RELOC_SECTION_RELA     // .rela.* - explicit addend
} RelocSectionType;
#endif // RELOCATION_TYPES_H
```
### Memory Layout of Elf64_Rela
| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0x00 | 8 | r_offset | Byte offset within target section where patch applies |
| 0x08 | 8 | r_info | Symbol index in bits 32-63, relocation type in bits 0-31 |
| 0x10 | 8 | r_addend | Signed value added to symbol address in formula |
Total: 24 bytes (0x18)
### Relocation Data Flow
```
Input Object File (.o)
┌─────────────────────────────────────────────────────────────────────┐
│ .text section                                                       │
│   offset 0x05: E8 00 00 00 00   (call with placeholder)            │
│                                                                     │
│ .rela.text section                                                  │
│   [0] r_offset=0x05, r_info=(sym=3, type=R_X86_64_PC32),           │
│       r_addend=-4                                                   │
│                                                                     │
│ .symtab section                                                     │
│   [3] name="helper", section=.text, offset=0x00                    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ MILESTONE 1: Section Merging                                        │
│   Input .text at offset 0x00 → Output .text at offset 0x1000       │
│   Mapping: (file.o, .text, 0x05) → output_offset 0x1005            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ MILESTONE 2: Symbol Resolution                                      │
│   "helper" resolved to virtual address 0x401020                     │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ MILESTONE 3: Relocation Processing (THIS MODULE)                    │
│                                                                     │
│ 1. Translate site offset:                                           │
│    site_vaddr = 0x401000 + 0x05 = 0x401005                          │
│                                                                     │
│ 2. Get symbol address:                                              │
│    symbol_vaddr = 0x401020                                          │
│                                                                     │
│ 3. Compute patch value (R_X86_64_PC32):                             │
│    computed = symbol_vaddr + addend - site_vaddr                    │
│             = 0x401020 + (-4) - 0x401005                            │
│             = 0x17                                                  │
│                                                                     │
│ 4. Write to output buffer at offset 0x1005:                         │
│    17 00 00 00 (little-endian 32-bit)                               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Output Buffer
┌─────────────────────────────────────────────────────────────────────┐
│ offset 0x1005: E8 17 00 00 00   (call with correct offset)          │
└─────────────────────────────────────────────────────────────────────┘
```

![Relocation Processing Module Architecture](./diagrams/tdd-diag-028.svg)

### Relocation Type Reference
```c
// Relocation type characteristics
typedef struct {
    uint32_t type;              // R_X86_64_* constant
    const char *name;           // Human-readable name
    size_t patch_size;          // Bytes written (1, 2, 4, or 8)
    int is_pc_relative;         // 1 if PC-relative (subtract site)
    int is_signed;              // 1 if signed (requires range check)
    int64_t min_value;          // Minimum valid value
    int64_t max_value;          // Maximum valid value
} RelocationTypeInfo;
// Relocation type lookup table
static const RelocationTypeInfo reloc_type_table[] = {
    // type               name           size  pc_rel  signed  min              max
    {R_X86_64_NONE,    "NONE",         0,    0,      0,      0,               0},
    {R_X86_64_64,      "R_X86_64_64",  8,    0,      0,      0,               INT64_MAX},
    {R_X86_64_PC32,    "R_X86_64_PC32",4,    1,      1,      INT32_MIN,       INT32_MAX},
    {R_X86_64_32,      "R_X86_64_32",  4,    0,      0,      0,               UINT32_MAX},
    {R_X86_64_32S,     "R_X86_64_32S", 4,    0,      1,      INT32_MIN,       INT32_MAX},
    {R_X86_64_16,      "R_X86_64_16",  2,    0,      0,      0,               UINT16_MAX},
    {R_X86_64_8,       "R_X86_64_8",   1,    0,      0,      0,               UINT8_MAX},
    {R_X86_64_PC64,    "R_X86_64_PC64",8,    1,      1,      INT64_MIN,       INT64_MAX},
};
// Formula reference:
// R_X86_64_64:   *loc = S + A
// R_X86_64_PC32: *loc = S + A - P
// R_X86_64_32:   *loc = (S + A) [truncate to 32-bit, zero-extend]
// R_X86_64_32S:  *loc = (S + A) [truncate to 32-bit, sign-extend]
//
// Where:
//   S = Symbol address (symbol_vaddr)
//   A = Addend (r_addend)
//   P = Place (site_vaddr) - address of relocation site
//   *loc = Location to patch
```
## Interface Contracts
### Relocation Parser Interface
```c
// File: 22_relocation_parser.h
#ifndef RELOCATION_PARSER_H
#define RELOCATION_PARSER_H
#include "21_relocation_types.h"
#include "04_section_types.h"
// Check if a section name indicates a relocation section
//
// Parameters:
//   name - Section name (e.g., ".rela.text", ".rel.data")
//
// Returns:
//   RELOC_SECTION_RELA for .rela.* sections
//   RELOC_SECTION_REL for .rel.* sections
//   RELOC_SECTION_NONE otherwise
RelocSectionType detect_relocation_section(const char *name);
// Parse all relocation sections from an object file
//
// Parameters:
//   obj - Object file with sections loaded (must not be NULL)
//
// Returns:
//   RELOC_OK on success
//   RELOC_ERR_MEMORY on allocation failure
//   RELOC_ERR_PARSE_ERROR if relocation entry is malformed
//
// On success:
//   - obj->relocations is allocated with obj->relocation_count entries
//   - Each Relocation has target_section_idx set
//   - Each Relocation has offset, sym_idx, type, addend populated
//
// Relocation sections are detected by:
//   - Section name starting with ".rela." (RELA format)
//   - Section name starting with ".rel." (REL format)
//   - sh_type == SHT_RELA or SHT_REL
RelocError parse_relocations(ObjectFile *obj);
// Get the target section index for a relocation section
//
// Parameters:
//   obj        - Object file with sections loaded
//   reloc_sec  - Relocation section (SHT_RELA or SHT_REL)
//   reloc_idx  - Index of relocation section in obj->sections
//
// Returns:
//   Index of target section (from sh_info field)
//   -1 if sh_info is invalid
//
// The sh_info field of a relocation section header contains the
// index of the section it applies to (e.g., .rela.text → .text)
int get_relocation_target_section(ObjectFile *obj, int reloc_idx);
// Parse a single RELA entry
//
// Parameters:
//   raw   - Raw Elf64_Rela structure from file
//   reloc - Output relocation entry to populate
//
// Post-conditions:
//   - reloc->offset = raw->r_offset
//   - reloc->sym_idx = ELF64_R_SYM(raw->r_info)
//   - reloc->type = ELF64_R_TYPE(raw->r_info)
//   - reloc->addend = raw->r_addend
void parse_rela_entry(const Elf64_Rela *raw, Relocation *reloc);
// Parse a single REL entry (implicit addend)
//
// Parameters:
//   raw         - Raw Elf64_Rel structure from file
//   target_data - Data of target section (to read implicit addend)
//   reloc       - Output relocation entry to populate
//
// Post-conditions:
//   - reloc->offset = raw->r_offset
//   - reloc->sym_idx = ELF64_R_SYM(raw->r_info)
//   - reloc->type = ELF64_R_TYPE(raw->r_info)
//   - reloc->addend = value read from target_data at r_offset
//
// Note: Addend size depends on relocation type (typically 4 or 8 bytes)
void parse_rel_entry(const Elf64_Rel *raw, const uint8_t *target_data,
                     Relocation *reloc);
// Get human-readable relocation type name
//
// Parameters:
//   type - R_X86_64_* constant
//
// Returns:
//   Static string (do not free)
//   "UNKNOWN" if type not recognized
const char *relocation_type_name(uint32_t type);
// Get human-readable error string
const char *reloc_error_string(RelocError err);
#endif // RELOCATION_PARSER_H
```
### Relocation Resolver Interface
```c
// File: 24_relocation_resolver.h
#ifndef RELOCATION_RESOLVER_H
#define RELOCATION_RESOLVER_H
#include "21_relocation_types.h"
#include "04_section_types.h"
#include "11_symbol_types.h"
// Resolve the target symbol for a relocation
//
// Parameters:
//   ctx   - Linker context with global symbol table
//   obj   - Source object file
//   reloc - Relocation entry with sym_idx set
//   out   - Output: resolved symbol address
//
// Returns:
//   RELOC_OK on success
//   RELOC_ERR_SYMBOL_INVALID if sym_idx >= symbol_count
//   RELOC_ERR_SYMBOL_UNRESOLVED if symbol not resolved
//   RELOC_ERR_SECTION_SYMBOL if section symbol cannot be resolved
//
// Handles symbol types:
//   - Regular symbols: lookup in global table
//   - STT_SECTION: use section's output address
//   - SHN_ABS symbols: use st_value directly
//   - Weak undefined: resolve to 0
RelocError resolve_relocation_symbol(LinkerContext *ctx,
                                      ObjectFile *obj,
                                      Relocation *reloc,
                                      uint64_t *out);
// Translate relocation site from input to output coordinates
//
// Parameters:
//   ctx           - Linker context with mapping table
//   obj           - Source object file
//   reloc         - Relocation with target_section_idx and offset
//   out_vaddr     - Output: virtual address of site
//   out_file_off  - Output: file offset in output buffer
//
// Returns:
//   RELOC_OK on success
//   RELOC_ERR_TARGET_INVALID if target section not found
//   RELOC_ERR_TARGET_NOBITS if target is .bss (cannot relocate)
//   RELOC_ERR_OFFSET_OUT_OF_RANGE if offset >= section size
//   RELOC_ERR_MAPPING_NOT_FOUND if no mapping entry
//
// Translation:
//   1. Get input section name from obj->sections[reloc->target_section_idx]
//   2. Find output section by name
//   3. Use mapping table to get output_offset for (file, section, offset)
//   4. site_vaddr = output_section.vaddr + output_offset
//   5. site_file_off = output_section.file_offset + output_offset
RelocError translate_relocation_site(LinkerContext *ctx,
                                      ObjectFile *obj,
                                      Relocation *reloc,
                                      uint64_t *out_vaddr,
                                      uint64_t *out_file_off);
// Fully resolve a relocation (symbol + site translation)
//
// Parameters:
//   ctx   - Linker context
//   obj   - Source object file
//   reloc - Relocation to resolve
//   out   - Output: fully resolved relocation context
//
// Returns:
//   RELOC_OK on success
//   Appropriate error on failure (out->error_msg populated)
//
// On success, out contains:
//   - site_vaddr: virtual address of patch location
//   - site_file_offset: where to write in output buffer
//   - symbol_vaddr: target symbol's address
//   - is_valid: 1
RelocError resolve_relocation(LinkerContext *ctx,
                               ObjectFile *obj,
                               Relocation *reloc,
                               ResolvedRelocation *out);
// Handle section symbols (STT_SECTION)
//
// Parameters:
//   ctx       - Linker context
//   obj       - Source object file
//   sym       - Symbol with type STT_SECTION
//   out_vaddr - Output: section base address
//
// Returns:
//   RELOC_OK on success
//   RELOC_ERR_SECTION_SYMBOL if section invalid
//
// For STT_SECTION:
//   - sym->section_idx contains the section number
//   - sym->value contains offset within that section
//   - Output: output_section.vaddr + mapping.output_offset + sym->value
RelocError resolve_section_symbol(LinkerContext *ctx,
                                   ObjectFile *obj,
                                   Symbol *sym,
                                   uint64_t *out_vaddr);
#endif // RELOCATION_RESOLVER_H
```
### Patch Calculator Interface
```c
// File: 26_patch_calculator.h
#ifndef PATCH_CALCULATOR_H
#define PATCH_CALCULATOR_H
#include "21_relocation_types.h"
// Get relocation type information
//
// Parameters:
//   type - R_X86_64_* constant
//
// Returns:
//   Pointer to type info (static storage)
//   NULL if type not recognized
const RelocationTypeInfo* get_relocation_type_info(uint32_t type);
// Check if a relocation type is supported
//
// Parameters:
//   type - R_X86_64_* constant
//
// Returns:
//   1 if supported (can be processed)
//   0 if unsupported (will cause error)
int is_relocation_type_supported(uint32_t type);
// Calculate patch value for R_X86_64_64
//
// Formula: *loc = S + A
//
// Parameters:
//   symbol_vaddr - Target symbol's virtual address (S)
//   addend       - Relocation addend (A)
//
// Returns:
//   Computed 64-bit value
static inline int64_t calc_r_x86_64_64(uint64_t symbol_vaddr, int64_t addend) {
    return (int64_t)(symbol_vaddr + addend);
}
// Calculate patch value for R_X86_64_PC32
//
// Formula: *loc = S + A - P
//
// Parameters:
//   symbol_vaddr - Target symbol's virtual address (S)
//   addend       - Relocation addend (A)
//   site_vaddr   - Relocation site virtual address (P)
//
// Returns:
//   Computed 32-bit signed offset
static inline int64_t calc_r_x86_64_pc32(uint64_t symbol_vaddr,
                                          int64_t addend,
                                          uint64_t site_vaddr) {
    return (int64_t)(symbol_vaddr + addend - site_vaddr);
}
// Calculate patch value for any supported type
//
// Parameters:
//   type         - R_X86_64_* constant
//   symbol_vaddr - Target symbol's virtual address
//   addend       - Relocation addend
//   site_vaddr   - Relocation site virtual address (for PC-relative)
//   out_value    - Output: computed patch value
//
// Returns:
//   RELOC_OK on success
//   RELOC_ERR_TYPE_UNSUPPORTED if type not implemented
RelocError calculate_patch_value(uint32_t type,
                                  uint64_t symbol_vaddr,
                                  int64_t addend,
                                  uint64_t site_vaddr,
                                  int64_t *out_value);
// Check for overflow in computed value
//
// Parameters:
//   type         - R_X86_64_* constant
//   value        - Computed patch value
//   out_overflow - Output: 1 if overflow detected
//
// Returns:
//   RELOC_OK (always succeeds)
//
// Overflow rules by type:
//   R_X86_64_64:   Never overflows (64-bit value into 64-bit field)
//   R_X86_64_PC32: Check INT32_MIN <= value <= INT32_MAX
//   R_X86_64_32:   Check 0 <= value <= UINT32_MAX
//   R_X86_64_32S:  Check INT32_MIN <= value <= INT32_MAX
RelocError check_patch_overflow(uint32_t type,
                                 int64_t value,
                                 int *out_overflow);
// Calculate and validate patch value
//
// Parameters:
//   type         - R_X86_64_* constant
//   symbol_vaddr - Target symbol's virtual address
//   addend       - Relocation addend
//   site_vaddr   - Relocation site virtual address
//   out_value    - Output: computed patch value
//
// Returns:
//   RELOC_OK on success
//   RELOC_ERR_TYPE_UNSUPPORTED if type not implemented
//   RELOC_ERR_OVERFLOW if value doesn't fit in field
RelocError calculate_and_validate_patch(uint32_t type,
                                         uint64_t symbol_vaddr,
                                         int64_t addend,
                                         uint64_t site_vaddr,
                                         int64_t *out_value);
#endif // PATCH_CALCULATOR_H
```
### Patch Writer Interface
```c
// File: 28_patch_writer.h
#ifndef PATCH_WRITER_H
#define PATCH_WRITER_H
#include "21_relocation_types.h"
// Write a patch value to the output buffer
//
// Parameters:
//   buffer     - Output buffer (must not be NULL)
//   buffer_size - Total size of output buffer
//   offset     - Byte offset within buffer to write
//   value      - Value to write
//   size       - Number of bytes (1, 2, 4, or 8)
//
// Returns:
//   RELOC_OK on success
//   RELOC_ERR_WRITE_FAILED if offset+size > buffer_size
//
// Writes in little-endian byte order (LSB first)
RelocError write_patch(uint8_t *buffer,
                        size_t buffer_size,
                        uint64_t offset,
                        int64_t value,
                        size_t size);
// Write a 32-bit little-endian value
//
// Parameters:
//   buffer - Output buffer
//   offset - Byte offset
//   value  - 32-bit value to write
static inline void write_le32(uint8_t *buffer, uint64_t offset, uint32_t value) {
    buffer[offset + 0] = (value >> 0) & 0xFF;
    buffer[offset + 1] = (value >> 8) & 0xFF;
    buffer[offset + 2] = (value >> 16) & 0xFF;
    buffer[offset + 3] = (value >> 24) & 0xFF;
}
// Write a 64-bit little-endian value
//
// Parameters:
//   buffer - Output buffer
//   offset - Byte offset
//   value  - 64-bit value to write
static inline void write_le64(uint8_t *buffer, uint64_t offset, uint64_t value) {
    buffer[offset + 0] = (value >> 0) & 0xFF;
    buffer[offset + 1] = (value >> 8) & 0xFF;
    buffer[offset + 2] = (value >> 16) & 0xFF;
    buffer[offset + 3] = (value >> 24) & 0xFF;
    buffer[offset + 4] = (value >> 32) & 0xFF;
    buffer[offset + 5] = (value >> 40) & 0xFF;
    buffer[offset + 6] = (value >> 48) & 0xFF;
    buffer[offset + 7] = (value >> 56) & 0xFF;
}
// Read a 32-bit little-endian value (for implicit addend in REL format)
//
// Parameters:
//   buffer - Input buffer
//   offset - Byte offset
//
// Returns:
//   32-bit value (sign-extended to int64_t for use as addend)
static inline int64_t read_le32_signed(const uint8_t *buffer, uint64_t offset) {
    uint32_t val = (uint32_t)buffer[offset + 0] |
                   ((uint32_t)buffer[offset + 1] << 8) |
                   ((uint32_t)buffer[offset + 2] << 16) |
                   ((uint32_t)buffer[offset + 3] << 24);
    return (int64_t)(int32_t)val;  // Sign-extend
}
// Read a 64-bit little-endian value (for implicit addend in REL format)
//
// Parameters:
//   buffer - Input buffer
//   offset - Byte offset
//
// Returns:
//   64-bit value
static inline int64_t read_le64(const uint8_t *buffer, uint64_t offset) {
    return (int64_t)(
        (uint64_t)buffer[offset + 0] |
        ((uint64_t)buffer[offset + 1] << 8) |
        ((uint64_t)buffer[offset + 2] << 16) |
        ((uint64_t)buffer[offset + 3] << 24) |
        ((uint64_t)buffer[offset + 4] << 32) |
        ((uint64_t)buffer[offset + 5] << 40) |
        ((uint64_t)buffer[offset + 6] << 48) |
        ((uint64_t)buffer[offset + 7] << 56)
    );
}
// Apply a resolved relocation to the output buffer
//
// Parameters:
//   buffer      - Output buffer with merged section data
//   buffer_size - Total buffer size
//   resolved    - Fully resolved relocation
//
// Returns:
//   RELOC_OK on success
//   RELOC_ERR_WRITE_FAILED if write fails
//
// This is the main entry point for applying a single relocation:
//   1. Calculate patch value
//   2. Check for overflow
//   3. Write to buffer at site_file_offset
RelocError apply_resolved_relocation(uint8_t *buffer,
                                      size_t buffer_size,
                                      ResolvedRelocation *resolved);
// Verify a patch was applied correctly (optional debugging)
//
// Parameters:
//   buffer      - Output buffer
//   buffer_size - Buffer size
//   offset      - Patch location
//   expected    - Expected value at location
//   size        - Value size
//
// Returns:
//   1 if buffer[offset:offset+size] == expected
//   0 otherwise
int verify_patch(const uint8_t *buffer,
                 size_t buffer_size,
                 uint64_t offset,
                 int64_t expected,
                 size_t size);
#endif // PATCH_WRITER_H
```

![Elf64_Rela Entry Memory Layout](./diagrams/tdd-diag-029.svg)

### Relocation Processor Interface
```c
// File: 30_relocation_processor.h
#ifndef RELOCATION_PROCESSOR_H
#define RELOCATION_PROCESSOR_H
#include "21_relocation_types.h"
#include "04_section_types.h"
// Process all relocations from all input files
//
// Parameters:
//   ctx - Linker context with:
//         - inputs[] parsed with symbols and relocations
//         - global_syms populated with resolved symbols
//         - outputs[] with virtual addresses assigned
//         - output_buffer with merged section data
//         - mapping table populated
//
// Returns:
//   RELOC_OK on success
//   RELOC_ERR_NO_INPUTS if no input files
//   RELOC_ERR_SYMBOL_UNRESOLVED if any symbol not resolved
//   RELOC_ERR_OVERFLOW if any relocation overflows
//
// This is the main entry point for relocation processing.
// Processes all relocations in order, applying patches to output_buffer.
//
// Prerequisites (must complete before calling):
//   1. Milestone 1: Section merging complete, mapping table built
//   2. Milestone 2: Symbol resolution complete, addresses assigned
//   3. Output buffer allocated with merged section data
RelocError process_all_relocations(LinkerContext *ctx);
// Process relocations from a single input file
//
// Parameters:
//   ctx     - Linker context
//   obj     - Input object file with relocations parsed
//   stats   - Statistics accumulator (may be NULL)
//
// Returns:
//   Number of errors encountered (0 = success)
//
// Processes each relocation:
//   1. Translate site offset to output coordinates
//   2. Resolve target symbol
//   3. Calculate patch value
//   4. Check for overflow
//   5. Write to output buffer
int process_file_relocations(LinkerContext *ctx,
                              ObjectFile *obj,
                              RelocationStats *stats);
// Process a single relocation
//
// Parameters:
//   ctx   - Linker context
//   obj   - Source object file
//   reloc - Relocation entry
//
// Returns:
//   RELOC_OK on success
//   Appropriate error on failure
//
// Full processing pipeline for one relocation
RelocError process_single_relocation(LinkerContext *ctx,
                                      ObjectFile *obj,
                                      Relocation *reloc);
// Get processing statistics
//
// Parameters:
//   ctx - Linker context after processing
//
// Returns:
//   Statistics structure with counts
RelocationStats get_relocation_stats(LinkerContext *ctx);
// Enable/disable verbose logging
//
// Parameters:
//   enabled - 1 to enable, 0 to disable
//
// When enabled, prints detailed info for each relocation:
//   - Type, symbol, addend
//   - Site address, symbol address
//   - Computed value
//   - Applied successfully or error
void set_relocation_verbose(int enabled);
// Set error callback for custom error handling
//
// Parameters:
//   callback - Function to call on each error (NULL to disable)
//
// Callback receives:
//   - Error code
//   - Source filename
//   - Relocation details
//   - Error message
typedef void (*RelocErrorCallback)(RelocError err,
                                    const char *file,
                                    const Relocation *reloc,
                                    const char *message);
void set_relocation_error_callback(RelocErrorCallback callback);
#endif // RELOCATION_PROCESSOR_H
```
## Algorithm Specification
### Algorithm 1: Relocation Section Parsing
**Purpose**: Extract all relocation entries from `.rela.*` and `.rel.*` sections.
**Input**:
- `obj`: Object file with sections loaded
- Must have section data for `.rela.*` / `.rel.*` sections
**Output**:
- `obj->relocations`: Array of parsed Relocation structures
- `obj->relocation_count`: Number of relocations
- Return: `RelocError` code
**Procedure**:
```
1. Count total relocations across all relocation sections
   total_count = 0
   FOR i = 0 TO obj->section_count - 1:
     sec = &obj->sections[i]
     IF sec->type == SHT_RELA:
       total_count += sec->size / sizeof(Elf64_Rela)  // 24 bytes each
     ELSE IF sec->type == SHT_REL:
       total_count += sec->size / sizeof(Elf64_Rel)   // 16 bytes each
   IF total_count == 0:
     obj->relocations = NULL
     obj->relocation_count = 0
     RETURN RELOC_OK
2. Allocate relocation array
   obj->relocations = calloc(total_count, sizeof(Relocation))
   IF obj->relocations == NULL:
     RETURN RELOC_ERR_MEMORY
   obj->relocation_count = 0
3. Parse each relocation section
   FOR i = 0 TO obj->section_count - 1:
     sec = &obj->sections[i]
     IF sec->type == SHT_RELA:
       a. Get target section index from original section header's sh_info
          // This was stored during parsing or needs to be re-read
          target_idx = get_target_section_index(obj, i)
       b. Calculate entry count
          count = sec->size / sizeof(Elf64_Rela)
       c. Parse each RELA entry
          raw_entries = (Elf64_Rela*)sec->data
          FOR j = 0 TO count - 1:
            reloc = &obj->relocations[obj->relocation_count]
            // Parse raw entry
            reloc->offset = raw_entries[j].r_offset
            reloc->sym_idx = ELF64_R_SYM(raw_entries[j].r_info)
            reloc->type = ELF64_R_TYPE(raw_entries[j].r_info)
            reloc->addend = raw_entries[j].r_addend
            reloc->target_section_idx = target_idx
            // Copy metadata
            strncpy(reloc->source_file, obj->filename, 255)
            IF target_idx >= 0 AND target_idx < obj->section_count:
              strncpy(reloc->target_section, 
                      obj->sections[target_idx].name, 63)
            obj->relocation_count++
     ELSE IF sec->type == SHT_REL:
       a. Get target section index
          target_idx = get_target_section_index(obj, i)
       b. Get target section data (for implicit addend)
          IF target_idx >= 0 AND target_idx < obj->section_count:
            target_data = obj->sections[target_idx].data
          ELSE:
            target_data = NULL
       c. Calculate entry count
          count = sec->size / sizeof(Elf64_Rel)
       d. Parse each REL entry
          raw_entries = (Elf64_Rel*)sec->data
          FOR j = 0 TO count - 1:
            reloc = &obj->relocations[obj->relocation_count]
            reloc->offset = raw_entries[j].r_offset
            reloc->sym_idx = ELF64_R_SYM(raw_entries[j].r_info)
            reloc->type = ELF64_R_TYPE(raw_entries[j].r_info)
            reloc->target_section_idx = target_idx
            // Read implicit addend from target section
            IF target_data != NULL AND reloc->offset + 4 <= target_sec->size:
              // Assume 32-bit for most REL relocations
              reloc->addend = read_le32_signed(target_data, reloc->offset)
            ELSE:
              reloc->addend = 0
            strncpy(reloc->source_file, obj->filename, 255)
            IF target_idx >= 0 AND target_idx < obj->section_count:
              strncpy(reloc->target_section,
                      obj->sections[target_idx].name, 63)
            obj->relocation_count++
4. RETURN RELOC_OK
```
**Invariants after execution**:
- All relocation entries have valid sym_idx (0 to symbol_count-1)
- All relocation entries have target_section_idx pointing to valid section
- Addend is correctly extracted (explicit for RELA, implicit for REL)
- Source file and target section names populated for error reporting
### Algorithm 2: Symbol Resolution for Relocation
**Purpose**: Resolve the target symbol for a relocation to its final virtual address.
**Input**:
- `ctx`: Linker context with global symbol table
- `obj`: Source object file
- `reloc`: Relocation with sym_idx set
**Output**:
- `out_vaddr`: Symbol's virtual address
- Return: `RelocError` code
**Procedure**:
```
1. Validate symbol index
   IF reloc->sym_idx >= obj->symbol_count:
     RETURN RELOC_ERR_SYMBOL_INVALID
2. Get symbol from object file's symbol table
   sym = &obj->symbols[reloc->sym_idx]
3. Handle different symbol types
   SWITCH sym->type:
   CASE STT_SECTION:
     // Section symbol - relocation targets section base
     err = resolve_section_symbol(ctx, obj, sym, out_vaddr)
     RETURN err
   CASE STT_FILE:
     // File symbol - should not appear in relocations
     RETURN RELOC_ERR_SYMBOL_INVALID
   DEFAULT:
     // Regular named symbol
   // Check symbol definition
   IF sym->section_idx == SHN_UNDEF:
     a. Lookup in global symbol table
        err = global_symbol_lookup(&ctx->global_syms, sym->name, &global)
        IF err != SYM_OK:
          RETURN RELOC_ERR_SYMBOL_UNRESOLVED
     b. Check resolution state
        IF global->state != SYM_STATE_RESOLVED:
          IF global->binding == STB_WEAK:
            // Weak undefined - resolve to 0
            *out_vaddr = 0
            RETURN RELOC_OK
          ELSE:
            RETURN RELOC_ERR_SYMBOL_UNRESOLVED
     c. Return resolved address
        *out_vaddr = global->final_address
        RETURN RELOC_OK
   ELSE IF sym->section_idx == SHN_ABS:
     // Absolute symbol - value is already an address
     *out_vaddr = sym->value
     RETURN RELOC_OK
   ELSE IF sym->section_idx == SHN_COMMON:
     // COMMON symbols should be resolved by now
     err = global_symbol_lookup(&ctx->global_syms, sym->name, &global)
     IF err != SYM_OK OR global->state != SYM_STATE_RESOLVED:
       RETURN RELOC_ERR_SYMBOL_UNRESOLVED
     *out_vaddr = global->final_address
     RETURN RELOC_OK
   ELSE:
     // Regular defined symbol
     a. Try global symbol table first
        err = global_symbol_lookup(&ctx->global_syms, sym->name, &global)
        IF err == SYM_OK AND global->state == SYM_STATE_RESOLVED:
          *out_vaddr = global->final_address
          RETURN RELOC_OK
     b. Local symbol - compute from section
        input_sec = &obj->sections[sym->section_idx]
        output_sec = find_output_section(ctx, input_sec->name)
        IF output_sec == NULL:
          RETURN RELOC_ERR_TARGET_INVALID
        // Translate input offset to output offset
        err = translate_offset(ctx, obj->filename, input_sec->name,
                               sym->value, &output_offset)
        IF err != MAP_OK:
          RETURN RELOC_ERR_MAPPING_NOT_FOUND
        *out_vaddr = output_sec->virtual_addr + output_offset
        RETURN RELOC_OK
```
### Algorithm 3: Section Symbol Resolution
**Purpose**: Resolve STT_SECTION symbols to their output section address.
**Input**:
- `ctx`: Linker context
- `obj`: Source object file
- `sym`: Symbol with type STT_SECTION
**Output**:
- `out_vaddr`: Section base address plus symbol offset
- Return: `RelocError` code
**Procedure**:
```
1. Validate section index
   IF sym->section_idx == SHN_UNDEF OR sym->section_idx >= obj->section_count:
     RETURN RELOC_ERR_SECTION_SYMBOL
2. Get input section
   input_sec = &obj->sections[sym->section_idx]
3. Find corresponding output section
   output_sec = find_output_section(ctx, input_sec->name)
   IF output_sec == NULL:
     RETURN RELOC_ERR_SECTION_SYMBOL
4. Get base offset of input section within output section
   err = mapping_table_lookup(&ctx->mapping, obj->filename, input_sec->name,
                               &mapping, NULL)
   IF err != MAP_OK:
     RETURN RELOC_ERR_MAPPING_NOT_FOUND
5. Compute final address
   // sym->value is offset within input section
   // mapping->output_offset is where input section starts in output
   // output_sec->virtual_addr is output section's base address
   *out_vaddr = output_sec->virtual_addr + mapping->output_offset + sym->value
   RETURN RELOC_OK
```
### Algorithm 4: Relocation Site Translation
**Purpose**: Map relocation site from input section coordinates to output buffer coordinates.
**Input**:
- `ctx`: Linker context with mapping table
- `obj`: Source object file
- `reloc`: Relocation with target_section_idx and offset
**Output**:
- `out_vaddr`: Virtual address of relocation site
- `out_file_off`: File offset in output buffer
- Return: `RelocError` code
**Procedure**:
```
1. Validate target section index
   IF reloc->target_section_idx < 0 OR 
      reloc->target_section_idx >= obj->section_count:
     RETURN RELOC_ERR_TARGET_INVALID
2. Get input section
   input_sec = &obj->sections[reloc->target_section_idx]
3. Check for .bss (NOBITS sections cannot have relocations applied to them)
   IF input_sec->type == SHT_NOBITS:
     RETURN RELOC_ERR_TARGET_NOBITS
4. Validate relocation offset within section bounds
   IF reloc->offset >= input_sec->size:
     RETURN RELOC_ERR_OFFSET_OUT_OF_RANGE
5. Find output section
   output_sec = find_output_section(ctx, input_sec->name)
   IF output_sec == NULL:
     RETURN RELOC_ERR_TARGET_INVALID
6. Translate offset using mapping table
   err = translate_offset(ctx, 
                          obj->filename, 
                          input_sec->name,
                          reloc->offset,
                          &output_offset)
   IF err != MAP_OK:
     RETURN RELOC_ERR_MAPPING_NOT_FOUND
7. Compute virtual address
   *out_vaddr = output_sec->virtual_addr + output_offset
8. Compute file offset
   *out_file_off = output_sec->file_offset + output_offset
9. RETURN RELOC_OK
```
### Algorithm 5: Patch Value Calculation
**Purpose**: Compute the patch value for each supported relocation type.
**Input**:
- `type`: R_X86_64_* relocation type
- `symbol_vaddr`: Target symbol's virtual address (S)
- `addend`: Relocation addend (A)
- `site_vaddr`: Relocation site virtual address (P)
**Output**:
- `out_value`: Computed patch value
- Return: `RelocError` code
**Procedure**:
```
SWITCH type:
CASE R_X86_64_NONE:
  // No-op relocation
  *out_value = 0
  RETURN RELOC_OK
CASE R_X86_64_64:
  // Direct 64-bit absolute address
  // Formula: *loc = S + A
  *out_value = (int64_t)(symbol_vaddr + addend)
  RETURN RELOC_OK
CASE R_X86_64_PC32:
  // PC-relative 32-bit signed offset
  // Formula: *loc = S + A - P
  *out_value = (int64_t)(symbol_vaddr + addend - site_vaddr)
  RETURN RELOC_OK
CASE R_X86_64_32:
  // Direct 32-bit zero-extended
  // Formula: *loc = (uint32_t)(S + A)
  *out_value = (int64_t)(symbol_vaddr + addend)
  // Overflow checked separately
  RETURN RELOC_OK
CASE R_X86_64_32S:
  // Direct 32-bit sign-extended
  // Formula: *loc = (int32_t)(S + A)
  *out_value = (int64_t)(symbol_vaddr + addend)
  // Overflow checked separately
  RETURN RELOC_OK
CASE R_X86_64_16:
  // Direct 16-bit zero-extended
  *out_value = (int64_t)(symbol_vaddr + addend)
  RETURN RELOC_OK
CASE R_X86_64_8:
  // Direct 8-bit
  *out_value = (int64_t)(symbol_vaddr + addend)
  RETURN RELOC_OK
CASE R_X86_64_PC64:
  // PC-relative 64-bit
  // Formula: *loc = S + A - P
  *out_value = (int64_t)(symbol_vaddr + addend - site_vaddr)
  RETURN RELOC_OK
DEFAULT:
  RETURN RELOC_ERR_TYPE_UNSUPPORTED
```
### Algorithm 6: Overflow Detection
**Purpose**: Verify computed value fits in the relocation field.
**Input**:
- `type`: R_X86_64_* relocation type
- `value`: Computed patch value
**Output**:
- `out_overflow`: 1 if overflow detected, 0 otherwise
- Return: `RelocError` code
**Procedure**:
```
SWITCH type:
CASE R_X86_64_NONE:
  *out_overflow = 0
  RETURN RELOC_OK
CASE R_X86_64_64:
  // 64-bit field can hold any 64-bit value
  *out_overflow = 0
  RETURN RELOC_OK
CASE R_X86_64_PC32:
CASE R_X86_64_32S:
  // Signed 32-bit: INT32_MIN to INT32_MAX
  IF value < INT32_MIN OR value > INT32_MAX:
    *out_overflow = 1
  ELSE:
    *out_overflow = 0
  RETURN RELOC_OK
CASE R_X86_64_32:
  // Unsigned 32-bit: 0 to UINT32_MAX
  IF value < 0 OR value > UINT32_MAX:
    *out_overflow = 1
  ELSE:
    *out_overflow = 0
  RETURN RELOC_OK
CASE R_X86_64_16:
  // Unsigned 16-bit
  IF value < 0 OR value > UINT16_MAX:
    *out_overflow = 1
  ELSE:
    *out_overflow = 0
  RETURN RELOC_OK
CASE R_X86_64_8:
  // Unsigned 8-bit
  IF value < 0 OR value > UINT8_MAX:
    *out_overflow = 1
  ELSE:
    *out_overflow = 0
  RETURN RELOC_OK
CASE R_X86_64_PC64:
  // 64-bit field
  *out_overflow = 0
  RETURN RELOC_OK
DEFAULT:
  *out_overflow = 0
  RETURN RELOC_OK
```
### Algorithm 7: Patch Application
**Purpose**: Write computed value to output buffer at relocation site.
**Input**:
- `buffer`: Output buffer with merged section data
- `buffer_size`: Total buffer size
- `offset`: File offset where patch applies
- `value`: Computed patch value
- `size`: Number of bytes to write (1, 2, 4, or 8)
**Output**:
- Return: `RelocError` code
- Buffer modified at offset
**Procedure**:
```
1. Validate offset and size
   IF offset + size > buffer_size:
     RETURN RELOC_ERR_WRITE_FAILED
2. Write value in little-endian byte order
   SWITCH size:
   CASE 1:
     buffer[offset] = (uint8_t)(value & 0xFF)
     BREAK
   CASE 2:
     buffer[offset + 0] = (value >> 0) & 0xFF
     buffer[offset + 1] = (value >> 8) & 0xFF
     BREAK
   CASE 4:
     buffer[offset + 0] = (value >> 0) & 0xFF
     buffer[offset + 1] = (value >> 8) & 0xFF
     buffer[offset + 2] = (value >> 16) & 0xFF
     buffer[offset + 3] = (value >> 24) & 0xFF
     BREAK
   CASE 8:
     buffer[offset + 0] = (value >> 0) & 0xFF
     buffer[offset + 1] = (value >> 8) & 0xFF
     buffer[offset + 2] = (value >> 16) & 0xFF
     buffer[offset + 3] = (value >> 24) & 0xFF
     buffer[offset + 4] = (value >> 32) & 0xFF
     buffer[offset + 5] = (value >> 40) & 0xFF
     buffer[offset + 6] = (value >> 48) & 0xFF
     buffer[offset + 7] = (value >> 56) & 0xFF
     BREAK
   DEFAULT:
     RETURN RELOC_ERR_WRITE_FAILED
3. RETURN RELOC_OK
```
### Algorithm 8: Full Relocation Processing Pipeline
**Purpose**: Process all relocations from all input files.
**Input**:
- `ctx`: Linker context with all prerequisites complete
**Output**:
- Output buffer with all patches applied
- Return: `RelocError` code
**Procedure**:
```
1. Validate prerequisites
   IF ctx->input_count == 0:
     RETURN RELOC_ERR_NO_INPUTS
   IF ctx->output_buffer == NULL:
     RETURN RELOC_ERR_WRITE_FAILED
2. Initialize statistics
   stats.total_count = 0
   stats.success_count = 0
   stats.error_count = 0
   stats.overflow_count = 0
3. Process each input file
   FOR file_idx = 0 TO ctx->input_count - 1:
     obj = &ctx->inputs[file_idx]
     a. Parse relocations if not already parsed
        IF obj->relocations == NULL:
          err = parse_relocations(obj)
          IF err != RELOC_OK:
            RETURN err
     b. Process each relocation
        FOR reloc_idx = 0 TO obj->relocation_count - 1:
          reloc = &obj->relocations[reloc_idx]
          stats.total_count++
          err = process_single_relocation(ctx, obj, reloc)
          IF err == RELOC_OK:
            stats.success_count++
          ELSE IF err == RELOC_ERR_OVERFLOW:
            stats.overflow_count++
            stats.error_count++
            // Report error but continue processing
            report_relocation_error(ctx, obj, reloc, err)
          ELSE:
            stats.error_count++
            report_relocation_error(ctx, obj, reloc, err)
4. Report summary
   IF verbose_mode:
     printf("Relocation processing complete:\n")
     printf("  Total: %zu\n", stats.total_count)
     printf("  Success: %zu\n", stats.success_count)
     printf("  Errors: %zu\n", stats.error_count)
     printf("  Overflow: %zu\n", stats.overflow_count)
5. RETURN (stats.error_count > 0) ? first_error : RELOC_OK
```
### Algorithm 9: Single Relocation Processing
**Purpose**: Process one relocation through the full pipeline.
**Input**:
- `ctx`: Linker context
- `obj`: Source object file
- `reloc`: Relocation entry
**Output**:
- Return: `RelocError` code
- Output buffer patched at relocation site
**Procedure**:
```
1. Skip R_X86_64_NONE relocations
   IF reloc->type == R_X86_64_NONE:
     RETURN RELOC_OK
2. Check if relocation type is supported
   IF NOT is_relocation_type_supported(reloc->type):
     RETURN RELOC_ERR_TYPE_UNSUPPORTED
3. Translate relocation site to output coordinates
   err = translate_relocation_site(ctx, obj, reloc,
                                    &site_vaddr, &site_file_off)
   IF err != RELOC_OK:
     RETURN err
4. Resolve target symbol
   err = resolve_relocation_symbol(ctx, obj, reloc, &symbol_vaddr)
   IF err != RELOC_OK:
     RETURN err
5. Get patch size for this relocation type
   type_info = get_relocation_type_info(reloc->type)
   IF type_info == NULL:
     RETURN RELOC_ERR_TYPE_UNSUPPORTED
   patch_size = type_info->patch_size
6. Calculate patch value
   err = calculate_patch_value(reloc->type,
                                symbol_vaddr,
                                reloc->addend,
                                site_vaddr,
                                &computed_value)
   IF err != RELOC_OK:
     RETURN err
7. Check for overflow
   int overflow = 0
   err = check_patch_overflow(reloc->type, computed_value, &overflow)
   IF overflow:
     report_overflow_error(ctx, obj, reloc, computed_value, type_info)
     RETURN RELOC_ERR_OVERFLOW
8. Apply patch to output buffer
   err = write_patch(ctx->output_buffer,
                     ctx->output_buffer_size,
                     site_file_off,
                     computed_value,
                     patch_size)
   IF err != RELOC_OK:
     RETURN err
9. Optional: Verify patch (debug mode)
   IF debug_verify_mode:
     IF NOT verify_patch(ctx->output_buffer, ctx->output_buffer_size,
                         site_file_off, computed_value, patch_size):
       RETURN RELOC_ERR_WRITE_FAILED
10. RETURN RELOC_OK
```

![R_X86_64_64 Absolute Relocation Calculation](./diagrams/tdd-diag-031.svg)

## Error Handling Matrix
| Error | Detected By | Recovery | User-Visible? | System State |
|-------|-------------|----------|---------------|--------------|
| `RELOC_ERR_MEMORY` | `calloc()` returns NULL | Abort linking | Yes: "Out of memory during relocation processing" | Clean, no partial patches |
| `RELOC_ERR_PARSE_ERROR` | Malformed relocation entry | Abort linking | Yes: "file.o: Invalid relocation entry at offset N" | Clean |
| `RELOC_ERR_TYPE_UNSUPPORTED` | Unknown relocation type | Skip relocation, report error | Yes: "file.o: Unsupported relocation type 42 at .text+0x10" | Clean, other relocations continue |
| `RELOC_ERR_OVERFLOW` | Value exceeds field size | Report error, skip patch | Yes: "file.o: Relocation overflow at .text+0x20: value 0x123456789 doesn't fit in 32 bits" | Clean, site unchanged |
| `RELOC_ERR_SYMBOL_INVALID` | Symbol index out of range | Abort linking | Yes: "file.o: Invalid symbol index 500 in relocation" | Clean |
| `RELOC_ERR_SYMBOL_UNRESOLVED` | Symbol not in global table | Abort linking | Yes: "file.o: Unresolved symbol 'foo' in relocation at .text+0x10" | Clean |
| `RELOC_ERR_TARGET_INVALID` | Target section not found | Abort linking | Yes: "file.o: Relocation targets unknown section" | Clean |
| `RELOC_ERR_TARGET_NOBITS` | Target is .bss | Abort linking | Yes: "file.o: Cannot apply relocation to .bss section" | Clean |
| `RELOC_ERR_OFFSET_OUT_OF_RANGE` | r_offset > section size | Abort linking | Yes: "file.o: Relocation offset 0x1000 exceeds section size 0x100" | Clean |
| `RELOC_ERR_MAPPING_NOT_FOUND` | No mapping for section | Abort linking | Yes: "file.o: No mapping found for section .text" | Clean |
| `RELOC_ERR_WRITE_FAILED` | Write beyond buffer | Abort linking | Yes: "file.o: Cannot write patch at offset 0xFFFF" | Clean |
| `RELOC_ERR_SECTION_SYMBOL` | Cannot resolve STT_SECTION | Abort linking | Yes: "file.o: Cannot resolve section symbol" | Clean |
| `RELOC_ERR_NO_INPUTS` | No input files | Abort linking | Yes: "No input files to process" | Clean |
### Error Message Format
```c
// Overflow error
fprintf(stderr, "ld: error: relocation overflow in %s\n", obj->filename);
fprintf(stderr, ">>> %s at %s+0x%lx\n", 
        relocation_type_name(reloc->type),
        reloc->target_section, reloc->offset);
fprintf(stderr, ">>> computed value: 0x%lx (%ld)\n", 
        computed_value, computed_value);
fprintf(stderr, ">>> valid range: %ld to %ld\n",
        type_info->min_value, type_info->max_value);
fprintf(stderr, ">>> symbol '%s' at 0x%lx\n", sym_name, symbol_vaddr);
fprintf(stderr, ">>> site at 0x%lx\n", site_vaddr);
// Unresolved symbol error
fprintf(stderr, "ld: error: unresolved symbol in relocation\n");
fprintf(stderr, ">>> file: %s\n", obj->filename);
fprintf(stderr, ">>> relocation: %s at %s+0x%lx\n",
        relocation_type_name(reloc->type),
        reloc->target_section, reloc->offset);
fprintf(stderr, ">>> symbol: '%s'\n", sym_name);
```
## Implementation Sequence with Checkpoints
### Phase 1: Relocation Section Parsing (2-3 hours)
**Files**: `21_relocation_types.h`, `22_relocation_parser.h`, `23_relocation_parser.c`
**Implementation steps**:
1. Define all relocation constants and structures in `21_relocation_types.h`
2. Declare parser interface in `22_relocation_parser.h`
3. Implement `detect_relocation_section()` to identify `.rela.*` and `.rel.*` sections
4. Implement `parse_rela_entry()` and `parse_rel_entry()` for raw entry parsing
5. Implement `parse_relocations()` to process all relocation sections
6. Implement `get_relocation_target_section()` to resolve sh_info
7. Write unit tests for parsing
**Checkpoint**:
```bash
gcc -c 23_relocation_parser.c -o relocation_parser.o
gcc tests/test_reloc_parse.c relocation_parser.o -o test_reloc_parse
./test_reloc_parse
# Expected output:
# [PASS] RELA section detection
# [PASS] REL section detection
# [PASS] RELA entry parsing
# [PASS] REL entry parsing with implicit addend
# [PASS] Target section resolution
# [PASS] Multiple relocation sections
# All tests passed!
```
At this point you can parse relocations but not resolve symbols or apply patches.
### Phase 2: Output Offset Translation (2-3 hours)
**Files**: `24_relocation_resolver.h`, `25_relocation_resolver.c`
**Implementation steps**:
1. Implement `translate_relocation_site()` using mapping table
2. Handle validation of target section (not .bss, within bounds)
3. Implement error reporting for invalid offsets
4. Test with mock mapping table
**Checkpoint**:
```bash
gcc -c 25_relocation_resolver.c -o relocation_resolver.o
gcc tests/test_offset_trans.c relocation_resolver.o -o test_offset_trans
./test_offset_trans
# Expected output:
# [PASS] Basic offset translation
# [PASS] .bss section rejection
# [PASS] Out-of-range offset detection
# [PASS] Missing mapping detection
# [PASS] Virtual address computation
# [PASS] File offset computation
# All tests passed!
```
At this point you can translate relocation sites to output coordinates.
### Phase 3: Symbol Resolution for Relocations (2-3 hours)
**Files**: Continue `25_relocation_resolver.c`
**Implementation steps**:
1. Implement `resolve_relocation_symbol()` for named symbols
2. Implement `resolve_section_symbol()` for STT_SECTION symbols
3. Handle weak undefined symbols (resolve to 0)
4. Handle absolute symbols (SHN_ABS)
5. Test with mock symbol table
**Checkpoint**:
```bash
./test_reloc_resolver
# Expected output:
# [PASS] Named symbol resolution
# [PASS] Section symbol resolution
# [PASS] Weak undefined symbol (resolves to 0)
# [PASS] Absolute symbol resolution
# [PASS] Invalid symbol index detection
# [PASS] Unresolved symbol detection
# All tests passed!
```
At this point you can resolve all relocation target symbols.
### Phase 4: R_X86_64_64 Implementation (1-2 hours)
**Files**: `26_patch_calculator.h`, `27_patch_calculator.c`
**Implementation steps**:
1. Implement `get_relocation_type_info()` lookup table
2. Implement `calc_r_x86_64_64()` inline function
3. Implement `calculate_patch_value()` dispatcher
4. Implement overflow check (always passes for 64-bit)
5. Write unit tests
**Checkpoint**:
```bash
gcc -c 27_patch_calculator.c -o patch_calculator.o
gcc tests/test_r64.c patch_calculator.o -o test_r64
./test_r64
# Expected output:
# [PASS] R_X86_64_64 basic calculation
# [PASS] R_X86_64_64 with positive addend
# [PASS] R_X86_64_64 with negative addend
# [PASS] R_X86_64_64 zero symbol address
# [PASS] R_X86_64_64 no overflow
# All tests passed!
```
At this point you can calculate 64-bit absolute patches.
### Phase 5: R_X86_64_PC32 Implementation (2-3 hours)
**Files**: Continue `27_patch_calculator.c`
**Implementation steps**:
1. Implement `calc_r_x86_64_pc32()` inline function
2. Implement overflow check for 32-bit signed range
3. Implement `check_patch_overflow()` for all types
4. Implement `calculate_and_validate_patch()` combined function
5. Write unit tests for edge cases
**Checkpoint**:
```bash
./test_pc32
# Expected output:
# [PASS] R_X86_64_PC32 forward reference
# [PASS] R_X86_64_PC32 backward reference
# [PASS] R_X86_64_PC32 with addend
# [PASS] R_X86_64_PC32 overflow detection (positive)
# [PASS] R_X86_64_PC32 overflow detection (negative)
# [PASS] R_X86_64_PC32 boundary values (INT32_MAX)
# [PASS] R_X86_64_PC32 boundary values (INT32_MIN)
# All tests passed!
```
At this point you can calculate and validate PC-relative patches.
### Phase 6: Patch Application and Verification (1-2 hours)
**Files**: `28_patch_writer.h`, `29_patch_writer.c`, `30_relocation_processor.h`, `31_relocation_processor.c`
**Implementation steps**:
1. Implement `write_le32()` and `write_le64()` inline functions
2. Implement `write_patch()` with bounds checking
3. Implement `apply_resolved_relocation()` orchestration
4. Implement `process_single_relocation()` full pipeline
5. Implement `process_all_relocations()` main entry point
6. Implement statistics tracking
7. Write end-to-end tests
**Checkpoint**:
```bash
# Create test object files
cat > test_call.s << 'EOF'
.globl _start
_start:
    call helper
    mov $60, %rax
    syscall
.globl helper
helper:
    ret
EOF
as -o test_call.o test_call.s
# Run full test
./test_reloc_full
# Expected output:
# [PASS] Single PC-relative call
# [PASS] Multiple relocations in sequence
# [PASS] Cross-section reference
# [PASS] Absolute address in data
# [PASS] Overflow detection
# [PASS] Statistics tracking
# 
# Relocation Statistics:
#   Total: 15
#   Success: 14
#   Skipped: 1
#   Errors: 0
# All tests passed!
```
**Milestone 3 Complete**: All relocations processed, output buffer fully patched.
## Test Specification
### Test Suite: Relocation Parsing
```c
// tests/test_reloc_parse.c
void test_rela_section_detection(void) {
    ASSERT_EQ(detect_relocation_section(".rela.text"), RELOC_SECTION_RELA);
    ASSERT_EQ(detect_relocation_section(".rela.data"), RELOC_SECTION_RELA);
    ASSERT_EQ(detect_relocation_section(".text"), RELOC_SECTION_NONE);
}
void test_rel_section_detection(void) {
    ASSERT_EQ(detect_relocation_section(".rel.text"), RELOC_SECTION_REL);
    ASSERT_EQ(detect_relocation_section(".rel.data"), RELOC_SECTION_REL);
}
void test_rela_entry_parsing(void) {
    // Create test object with known relocation
    system("as -o /tmp/rela.o << 'EOF'\n"
           ".text\n"
           "call external_func\n"
           "EOF");
    ObjectFile obj;
    parse_object_file("/tmp/rela.o", &obj);
    parse_symbols(&obj);
    RelocError err = parse_relocations(&obj);
    ASSERT_EQ(err, RELOC_OK);
    ASSERT_TRUE(obj.relocation_count > 0);
    // Verify first relocation
    Relocation *r = &obj.relocations[0];
    ASSERT_TRUE(r->type == R_X86_64_PC32 || r->type == R_X86_64_PLT32);
    ASSERT_TRUE(r->sym_idx > 0);  // Not the null symbol
    object_file_destroy(&obj);
}
void test_implicit_addend(void) {
    // Create object with REL format (implicit addend)
    // Note: Modern x86-64 typically uses RELA, but we should handle REL
    system("as -o /tmp/rel.o << 'EOF'\n"
           ".text\n"
           "movl $0x12345678, %eax\n"  // Has immediate value
           "EOF");
    ObjectFile obj;
    parse_object_file("/tmp/rel.o", &obj);
    RelocError err = parse_relocations(&obj);
    // May or may not have relocations depending on assembler
    ASSERT_EQ(err, RELOC_OK);
    object_file_destroy(&obj);
}
void test_multiple_relocation_sections(void) {
    system("as -o /tmp/multi.o << 'EOF'\n"
           ".text\n"
           "call func1\n"
           ".data\n"
           ".quad global_var\n"
           ".section .rodata\n"
           ".quad string_ptr\n"
           "EOF");
    ObjectFile obj;
    parse_object_file("/tmp/multi.o", &obj);
    parse_symbols(&obj);
    RelocError err = parse_relocations(&obj);
    ASSERT_EQ(err, RELOC_OK);
    // Should have relocations in both .rela.text and .rela.data
    object_file_destroy(&obj);
}
```
### Test Suite: Patch Calculation
```c
// tests/test_reloc_calc.c
void test_r_x86_64_64_basic(void) {
    uint64_t symbol = 0x401000;
    int64_t addend = 0;
    int64_t result = calc_r_x86_64_64(symbol, addend);
    ASSERT_EQ(result, 0x401000);
}
void test_r_x86_64_64_with_addend(void) {
    uint64_t symbol = 0x401000;
    int64_t addend = 100;
    int64_t result = calc_r_x86_64_64(symbol, addend);
    ASSERT_EQ(result, 0x401064);
}
void test_r_x86_64_64_negative_addend(void) {
    uint64_t symbol = 0x401000;
    int64_t addend = -16;
    int64_t result = calc_r_x86_64_64(symbol, addend);
    ASSERT_EQ(result, 0x400FF0);
}
void test_r_x86_64_pc32_forward(void) {
    // Symbol is ahead of site
    uint64_t symbol = 0x401100;  // Target
    int64_t addend = -4;         // Account for instruction size
    uint64_t site = 0x401000;    // Call instruction location
    int64_t result = calc_r_x86_64_pc32(symbol, addend, site);
    // Expected: 0x401100 + (-4) - 0x401000 = 0xFC
    ASSERT_EQ(result, 0xFC);
}
void test_r_x86_64_pc32_backward(void) {
    // Symbol is behind site (backward jump)
    uint64_t symbol = 0x400F00;  // Target (before site)
    int64_t addend = -4;
    uint64_t site = 0x401000;    // Jump instruction location
    int64_t result = calc_r_x86_64_pc32(symbol, addend, site);
    // Expected: 0x400F00 + (-4) - 0x401000 = -0x104 = 0xFFFFFEFC (as int32)
    ASSERT_EQ((int32_t)result, -0x104);
}
void test_r_x86_64_pc32_exact_boundary(void) {
    // Test INT32_MAX boundary
    int overflow;
    // Exactly at boundary - should not overflow
    check_patch_overflow(R_X86_64_PC32, INT32_MAX, &overflow);
    ASSERT_EQ(overflow, 0);
    check_patch_overflow(R_X86_64_PC32, INT32_MIN, &overflow);
    ASSERT_EQ(overflow, 0);
    // Beyond boundary - should overflow
    check_patch_overflow(R_X86_64_PC32, (int64_t)INT32_MAX + 1, &overflow);
    ASSERT_EQ(overflow, 1);
    check_patch_overflow(R_X86_64_PC32, (int64_t)INT32_MIN - 1, &overflow);
    ASSERT_EQ(overflow, 1);
}
void test_r_x86_64_32_unsigned(void) {
    int overflow;
    // Within unsigned 32-bit range
    check_patch_overflow(R_X86_64_32, 0, &overflow);
    ASSERT_EQ(overflow, 0);
    check_patch_overflow(R_X86_64_32, UINT32_MAX, &overflow);
    ASSERT_EQ(overflow, 0);
    // Negative should overflow (unsigned)
    check_patch_overflow(R_X86_64_32, -1, &overflow);
    ASSERT_EQ(overflow, 1);
    // Beyond 32-bit should overflow
    check_patch_overflow(R_X86_64_32, (int64_t)UINT32_MAX + 1, &overflow);
    ASSERT_EQ(overflow, 1);
}
```
### Test Suite: Patch Writing
```c
// tests/test_reloc_write.c
void test_write_le32(void) {
    uint8_t buffer[8] = {0};
    write_le32(buffer, 0, 0x12345678);
    ASSERT_EQ(buffer[0], 0x78);
    ASSERT_EQ(buffer[1], 0x56);
    ASSERT_EQ(buffer[2], 0x34);
    ASSERT_EQ(buffer[3], 0x12);
}
void test_write_le64(void) {
    uint8_t buffer[16] = {0};
    write_le64(buffer, 0, 0x0123456789ABCDEF);
    ASSERT_EQ(buffer[0], 0xEF);
    ASSERT_EQ(buffer[1], 0xCD);
    ASSERT_EQ(buffer[2], 0xAB);
    ASSERT_EQ(buffer[3], 0x89);
    ASSERT_EQ(buffer[4], 0x67);
    ASSERT_EQ(buffer[5], 0x45);
    ASSERT_EQ(buffer[6], 0x23);
    ASSERT_EQ(buffer[7], 0x01);
}
void test_read_le32_signed(void) {
    uint8_t buffer[] = {0xFC, 0xFE, 0xFF, 0xFF};  // -4 in little-endian
    int64_t result = read_le32_signed(buffer, 0);
    ASSERT_EQ(result, -4);
}
void test_write_patch_bounds_check(void) {
    uint8_t buffer[16] = {0};
    // Valid write
    RelocError err = write_patch(buffer, 16, 8, 0x12345678, 4);
    ASSERT_EQ(err, RELOC_OK);
    // Out of bounds
    err = write_patch(buffer, 16, 14, 0x12345678, 4);
    ASSERT_EQ(err, RELOC_ERR_WRITE_FAILED);
    // At boundary
    err = write_patch(buffer, 16, 12, 0x12345678, 4);
    ASSERT_EQ(err, RELOC_OK);
}
void test_apply_and_verify(void) {
    uint8_t buffer[32] = {0};
    // Apply a patch
    RelocError err = write_patch(buffer, 32, 8, 0xDEADBEEF, 4);
    ASSERT_EQ(err, RELOC_OK);
    // Verify it was written correctly
    int match = verify_patch(buffer, 32, 8, 0xDEADBEEF, 4);
    ASSERT_EQ(match, 1);
    // Verify wrong value fails
    match = verify_patch(buffer, 32, 8, 0xCAFEBABE, 4);
    ASSERT_EQ(match, 0);
}
```
### Test Suite: Full Relocation Processing
```c
// tests/test_reloc_full.c
void test_simple_call_relocation(void) {
    // Create two files: one with a call, one with the target
    system("as -o /tmp/caller.o << 'EOF'\n"
           ".globl _start\n"
           "_start:\n"
           "    call target_func\n"
           "    mov $60, %rax\n"
           "    xor %rdi, %rdi\n"
           "    syscall\n"
           "EOF");
    system("as -o /tmp/callee.o << 'EOF'\n"
           ".globl target_func\n"
           "target_func:\n"
           "    ret\n"
           "EOF");
    // Parse, merge, resolve, relocate
    LinkerContext ctx;
    linker_context_init(&ctx);
    ObjectFile caller, callee;
    parse_object_file("/tmp/caller.o", &caller);
    parse_object_file("/tmp/callee.o", &callee);
    linker_add_input(&ctx, &caller);
    linker_add_input(&ctx, &callee);
    linker_merge_sections(&ctx);
    linker_assign_addresses(&ctx);
    collect_all_symbols(&ctx);
    check_undefined_symbols(&ctx);
    assign_symbol_addresses(&ctx);
    // Build output buffer
    build_output_buffer(&ctx);
    // Process relocations
    RelocError err = process_all_relocations(&ctx);
    ASSERT_EQ(err, RELOC_OK);
    // Verify the call instruction was patched correctly
    // Find the call instruction in output buffer
    // It should now contain the correct relative offset
    linker_context_destroy(&ctx);
}
void test_data_reference_relocation(void) {
    system("as -o /tmp/data_ref.o << 'EOF'\n"
           ".data\n"
           ".globl ptr\n"
           "ptr:\n"
           "    .quad target_data\n"
           ".text\n"
           ".globl _start\n"
           "_start:\n"
           "    mov ptr(%rip), %rax\n"
           "    mov $60, %rax\n"
           "    syscall\n"
           ".globl target_data\n"
           "target_data:\n"
           "    .quad 42\n"
           "EOF");
    LinkerContext ctx;
    linker_context_init(&ctx);
    ObjectFile obj;
    parse_object_file("/tmp/data_ref.o", &obj);
    linker_add_input(&ctx, &obj);
    linker_merge_sections(&ctx);
    linker_assign_addresses(&ctx);
    collect_all_symbols(&ctx);
    assign_symbol_addresses(&ctx);
    build_output_buffer(&ctx);
    RelocError err = process_all_relocations(&ctx);
    ASSERT_EQ(err, RELOC_OK);
    // Verify ptr contains address of target_data
    GlobalSymbol *ptr_sym, *target_sym;
    global_symbol_lookup(&ctx.global_syms, "ptr", &ptr_sym);
    global_symbol_lookup(&ctx.global_syms, "target_data", &target_sym);
    // Read the value at ptr's location in output buffer
    // It should equal target_data's address
    linker_context_destroy(&ctx);
}
void test_overflow_detection(void) {
    // Create a relocation that will overflow
    system("as -o /tmp/overflow.o << 'EOF'\n"
           ".text\n"
           ".globl _start\n"
           "_start:\n"
           "    movl target_symbol, %eax  // 32-bit relocation\n"
           "    ret\n"
           "EOF");
    // Manually create a scenario where target is > 4GB away
    // This would require setting up addresses appropriately
    // For this test, we verify that overflow is detected
    // when computed value exceeds INT32_MAX or INT32_MIN
}
void test_section_symbol_relocation(void) {
    system("as -o /tmp/sec_sym.o << 'EOF'\n"
           ".section .rodata\n"
           "msg: .asciz \"Hello\"\n"
           ".text\n"
           ".globl _start\n"
           "_start:\n"
           "    lea msg(%rip), %rdi\n"
           "    ret\n"
           "EOF");
    LinkerContext ctx;
    linker_context_init(&ctx);
    ObjectFile obj;
    parse_object_file("/tmp/sec_sym.o", &obj);
    linker_add_input(&ctx, &obj);
    linker_merge_sections(&ctx);
    linker_assign_addresses(&ctx);
    collect_all_symbols(&ctx);
    assign_symbol_addresses(&ctx);
    build_output_buffer(&ctx);
    RelocError err = process_all_relocations(&ctx);
    ASSERT_EQ(err, RELOC_OK);
    linker_context_destroy(&ctx);
}
```
## Performance Targets
| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| Parse relocations (per file) | < 1ms per 1000 entries | `gettimeofday()` around parse_relocations() |
| Symbol lookup (per relocation) | < 100ns average | Hash table lookup microbenchmark |
| Patch calculation | < 50ns average | Inline function timing |
| Patch write | < 30ns average | Memory write microbenchmark |
| Full processing (10,000 relocs) | < 1 second | End-to-end timing |
| Memory per relocation | < 64 bytes | RSS delta measurement |
| Output buffer access | Cache-friendly | Sequential access pattern |
### Memory Budget
```
Per relocation entry (Relocation struct):
  Offset, addend, sym_idx, type:  ~32 bytes
  Metadata (file, section names): ~64 bytes
  Total:                          ~96 bytes
Per resolved relocation (ResolvedRelocation):
  Base struct:                    ~100 bytes
  Error message buffer:           ~256 bytes
  Total:                          ~356 bytes
Example for 100,000 relocations:
  Relocation entries:     100,000 * 96 = 9.6 MB
  Processing stack:       ~10 KB
  Output buffer access:   Direct pointer (no copy)
  Total:                  ~10 MB
```
### Cache Optimization
```c
// Process relocations in file order for cache locality
// Reuse ResolvedRelocation struct to avoid allocation
void process_file_relocations_optimized(LinkerContext *ctx, ObjectFile *obj) {
    ResolvedRelocation resolved;  // Stack-allocated, reused
    for (size_t i = 0; i < obj->relocation_count; i++) {
        // Reset resolved struct
        memset(&resolved, 0, sizeof(resolved));
        resolved.reloc = &obj->relocations[i];
        // Process with reused struct
        resolve_relocation(ctx, obj, resolved.reloc, &resolved);
        if (resolved.is_valid) {
            apply_resolved_relocation(ctx->output_buffer, 
                                       ctx->output_buffer_size, 
                                       &resolved);
        }
    }
}
```
## Integration Notes
### Dependencies
- **Milestone 1**: Requires `LinkerContext`, `ObjectFile`, `OutputSection`, mapping table functions (`translate_offset`)
- **Milestone 2**: Requires `GlobalSymbolTable`, `GlobalSymbol`, symbol lookup functions
- **Standard library**: `<stdint.h>`, `<stdio.h>`, `<stdlib.h>`, `<string.h>`, `<stdbool.h>`, `<limits.h>`
- **No external libraries**
### API for Milestone 4 (Executable Generation)
The Executable Generation module will:
1. Use `ctx->output_buffer` directly (already patched by this module)
2. Call `get_relocation_stats()` for diagnostics
3. No additional interface needed - output buffer is ready for writing
### Thread Safety
The relocation processing module is single-threaded by design. The processing order matters for:
- Error reporting (first error reported)
- Debug output (sequential logging)
If parallel processing is desired in the future:
- Each input file's relocations can be processed independently
- Symbol lookups require read-only access to global table (thread-safe)
- Output buffer writes must be synchronized (different regions per section)
[[CRITERIA_JSON: {"module_id": "build-linker-m3", "criteria": ["Parse .rela sections from input object files extracting Elf64_Rela entries with r_offset, r_info (symbol index via ELF64_R_SYM macro and type via ELF64_R_TYPE macro), and r_addend fields", "Parse .rel sections from input object files extracting Elf64_Rel entries with implicit addend read from relocation site in target section data", "Identify target section for each relocation section via sh_info field in section header pointing to the section the relocations apply to", "Implement relocation type dispatcher supporting at minimum R_X86_64_64 (type 1) and R_X86_64_PC32 (type 2) relocation types", "Translate relocation site offsets from input section coordinates to output buffer coordinates using the section mapping table from Milestone 1", "Look up target symbols by index in the input file's symbol table, then resolve to final virtual addresses via the global symbol table from Milestone 2", "Handle section symbols (STT_SECTION type) by resolving to their output section's virtual address plus the symbol's st_value offset", "Handle absolute symbols (SHN_ABS) by using st_value directly as the address", "Handle weak undefined symbols by resolving their relocations to address 0 without error", "Compute R_X86_64_64 patch values using formula: symbol_address + addend (direct 64-bit absolute address)", "Compute R_X86_64_PC32 patch values using formula: symbol_address + addend - site_address (PC-relative 32-bit signed offset)", "Detect and report overflow errors for 32-bit relocations when computed values exceed INT32_MIN to INT32_MAX range for signed types", "Detect and report overflow errors for R_X86_64_32 when computed values are negative or exceed UINT32_MAX", "Write computed relocation values to the output buffer at the correct file offset with little-endian byte order (LSB first)", "Validate relocation site offset does not exceed target section size before processing", "Reject relocations targeting .bss sections (SHT_NOBITS) with appropriate error", "Process relocations only after symbol resolution is complete (dependency ordering enforced)", "Support optional verification mode to read back and confirm patch values", "Track and report relocation processing statistics: total count, success count, error count, overflow count"]}]
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: build-linker-m4 -->
# Executable Generation: Technical Design Specification
## Module Charter
The Executable Generation module transforms the linker's internal representation—patched section data, resolved symbols, and virtual address assignments—into a valid ELF64 executable file that the Linux loader can execute. This module constructs the ELF header with correct identification and entry point, creates program headers (PT_LOAD segments) describing loadable memory regions with appropriate permissions, validates page alignment constraints for mmap compatibility, and writes the final binary to disk with executable permissions.
This module does NOT parse object files, merge sections, resolve symbols, or process relocations—those are complete before this module runs. Upstream, it depends on: the output buffer containing patched section data from Milestone 3, the global symbol table with resolved addresses from Milestone 2, and output sections with assigned virtual addresses from Milestone 1. Downstream, it produces a standalone executable file that the OS loader consumes directly. The module maintains strict invariants: all PT_LOAD segments satisfy the alignment constraint (p_offset % p_align == p_vaddr % p_align), the entry point points to a valid symbol address, segments do not overlap in virtual address space, and the file is complete and runnable even if chmod fails.
## File Structure
```
linker/
├── 33_elf_exec_types.h      # ELF executable structures and constants
├── 34_elf_header.h          # ELF header construction interface
├── 35_elf_header.c          # ELF header builder implementation
├── 36_segment_builder.h     # Segment construction interface
├── 37_segment_builder.c     # PT_LOAD segment builder implementation
├── 38_program_header.h      # Program header writer interface
├── 39_program_header.c      # Program header serialization
├── 40_entry_point.h         # Entry point resolution interface
├── 41_entry_point.c         # _start/main lookup implementation
├── 42_exec_writer.h         # Executable file writer interface
├── 43_exec_writer.c         # Complete ELF file assembly
├── 44_exec_main.c           # Test driver entry point
└── tests/
    ├── test_elf_header.c    # ELF header generation tests
    ├── test_segments.c      # Segment layout tests
    ├── test_entry.c         # Entry point resolution tests
    ├── test_exec_full.c     # End-to-end executable tests
    └── fixtures/
        ├── minimal.s        # Minimal _start-only program
        ├── with_main.s      # Program with main (no _start)
        ├── multi_segment.s  # Text + data + bss
        └── no_entry.s       # Missing entry point (error case)
```
## Complete Data Model
### ELF64 Header Structure
The ELF64 header is exactly 64 bytes and appears at file offset 0:
```c
// File: 33_elf_exec_types.h
#ifndef ELF_EXEC_TYPES_H
#define ELF_EXEC_TYPES_H
#include <stdint.h>
#include <stddef.h>
// ============================================================================
// ELF Identification Indices
// ============================================================================
#define EI_MAG0         0
#define EI_MAG1         1
#define EI_MAG2         2
#define EI_MAG3         3
#define EI_CLASS        4
#define EI_DATA         5
#define EI_VERSION      6
#define EI_OSABI        7
#define EI_ABIVERSION   8
#define EI_NIDENT       16
// ============================================================================
// ELF Magic Bytes
// ============================================================================
#define ELFMAG0         0x7F
#define ELFMAG1         'E'
#define ELFMAG2         'L'
#define ELFMAG3         'F'
// ============================================================================
// ELF Class
// ============================================================================
#define ELFCLASSNONE    0
#define ELFCLASS32      1
#define ELFCLASS64      2
// ============================================================================
// ELF Data Encoding
// ============================================================================
#define ELFDATANONE     0
#define ELFDATA2LSB     1   // Little-endian
#define ELFDATA2MSB     2   // Big-endian
// ============================================================================
// ELF File Types
// ============================================================================
#define ET_NONE         0   // No file type
#define ET_REL          1   // Relocatable
#define ET_EXEC         2   // Executable
#define ET_DYN          3   // Shared object
#define ET_CORE         4   // Core file
// ============================================================================
// ELF Machine Types
// ============================================================================
#define EM_NONE         0
#define EM_X86_64       62  // AMD x86-64 architecture
// ============================================================================
// Program Header Types
// ============================================================================
#define PT_NULL         0   // Unused entry
#define PT_LOAD         1   // Loadable segment
#define PT_DYNAMIC      2   // Dynamic linking information
#define PT_INTERP       3   // Interpreter pathname
#define PT_NOTE         4   // Auxiliary information
#define PT_PHDR         6   // Program header table
#define PT_GNU_EH_FRAME 0x6474e550
#define PT_GNU_STACK    0x6474e551
#define PT_GNU_RELRO    0x6474e552
// ============================================================================
// Program Header Flags
// ============================================================================
#define PF_X            0x1  // Execute permission
#define PF_W            0x2  // Write permission
#define PF_R            0x4  // Read permission
#define PF_MASKOS       0x0FF00000  // OS-specific flags
#define PF_MASKPROC     0xF0000000  // Processor-specific flags
// ============================================================================
// Section Header Types (for reference)
// ============================================================================
#define SHT_NULL        0
#define SHT_PROGBITS    1
#define SHT_SYMTAB      2
#define SHT_STRTAB      3
#define SHT_RELA        4
#define SHT_NOBITS      8
// ============================================================================
// Section Header Flags (for reference)
// ============================================================================
#define SHF_WRITE       0x1
#define SHF_ALLOC       0x2
#define SHF_EXECINSTR   0x4
// ============================================================================
// ELF64 Header Structure - 64 bytes, packed, no padding
// ============================================================================
typedef struct __attribute__((packed)) {
    uint8_t  e_ident[EI_NIDENT];    // +0x00: ELF identification bytes
    uint16_t e_type;                 // +0x10: Object file type (ET_EXEC = 2)
    uint16_t e_machine;              // +0x12: Architecture (EM_X86_64 = 62)
    uint32_t e_version;              // +0x14: Object file version (always 1)
    uint64_t e_entry;                // +0x18: Entry point virtual address
    uint64_t e_phoff;                // +0x20: Program header table offset
    uint64_t e_shoff;                // +0x28: Section header table offset
    uint32_t e_flags;                // +0x30: Processor-specific flags
    uint16_t e_ehsize;               // +0x34: ELF header size (64 bytes)
    uint16_t e_phentsize;            // +0x36: Program header entry size (56)
    uint16_t e_phnum;                // +0x38: Number of program headers
    uint16_t e_shentsize;            // +0x3A: Section header entry size (64)
    uint16_t e_shnum;                // +0x3C: Number of section headers
    uint16_t e_shstrndx;             // +0x3E: Section name string table index
} Elf64_Ehdr;                        // Total: 64 bytes (0x40)
// ============================================================================
// ELF64 Program Header Structure - 56 bytes, packed
// ============================================================================
typedef struct __attribute__((packed)) {
    uint32_t p_type;        // +0x00: Segment type (PT_LOAD, etc.)
    uint32_t p_flags;       // +0x04: Segment flags (PF_R, PF_W, PF_X)
    uint64_t p_offset;      // +0x08: Segment file offset
    uint64_t p_vaddr;       // +0x10: Segment virtual address
    uint64_t p_paddr;       // +0x18: Segment physical address (unused)
    uint64_t p_filesz;      // +0x20: Segment size in file
    uint64_t p_memsz;       // +0x28: Segment size in memory
    uint64_t p_align;       // +0x30: Segment alignment
} Elf64_Phdr;               // Total: 56 bytes (0x38)
// ============================================================================
// Internal Output Segment Structure
// ============================================================================
#define MAX_SEGMENTS        16
#define PAGE_SIZE           0x1000  // 4096 bytes
#define MAX_SEGMENT_NAME    32
typedef struct {
    uint32_t p_type;            // Segment type (PT_LOAD)
    uint32_t p_flags;           // Segment flags (PF_R | PF_W | PF_X)
    uint64_t p_offset;          // File offset
    uint64_t p_vaddr;           // Virtual address
    uint64_t p_paddr;           // Physical address (same as vaddr)
    uint64_t p_filesz;          // Size in file
    uint64_t p_memsz;           // Size in memory (includes .bss)
    uint64_t p_align;           // Alignment (PAGE_SIZE)
    char name[MAX_SEGMENT_NAME]; // Human-readable name for debugging
    int is_valid;               // 1 if segment has content
} OutputSegment;
// ============================================================================
// Executable Layout Structure
// ============================================================================
typedef struct {
    uint64_t elf_header_offset;     // Always 0
    uint64_t phdr_offset;           // Always 64 (sizeof(Elf64_Ehdr))
    uint64_t phdr_size;             // phnum * sizeof(Elf64_Phdr)
    uint64_t content_offset;        // First segment file offset (page-aligned)
    uint64_t total_file_size;       // Sum of all segment file sizes + headers
    uint64_t entry_point;           // Virtual address of _start or main
    uint16_t phnum;                 // Number of program headers
    int has_text_segment;           // 1 if text segment exists
    int has_data_segment;           // 1 if data segment exists
} ExecutableLayout;
// ============================================================================
// Error Codes
// ============================================================================
typedef enum {
    EXEC_OK = 0,
    EXEC_ERR_MEMORY,                // Memory allocation failed
    EXEC_ERR_ENTRY_NOT_FOUND,       // Neither _start nor main found
    EXEC_ERR_NO_SECTIONS,           // No output sections to emit
    EXEC_ERR_ALIGNMENT_INVALID,     // Page alignment constraint violated
    EXEC_ERR_SEGMENT_OVERLAP,       // Segments overlap in virtual address space
    EXEC_ERR_FILE_OPEN,             // Cannot open output file
    EXEC_ERR_FILE_WRITE,            // Write to output file failed
    EXEC_ERR_CHMOD,                 // Cannot set executable permissions
    EXEC_ERR_INVALID_LAYOUT,        // Internal layout calculation error
    EXEC_ERR_NO_INPUTS              // No input context provided
} ExecError;
// ============================================================================
// Statistics
// ============================================================================
typedef struct {
    size_t text_segment_size;       // Text segment file size
    size_t data_segment_size;       // Data segment file size
    size_t bss_size;                // .bss size (memory only)
    size_t total_file_size;         // Complete file size
    uint64_t entry_address;         // Entry point address
    const char *entry_symbol;       // Entry point symbol name
} ExecStats;
#endif // ELF_EXEC_TYPES_H
```
### Memory Layout of ELF64 Header
| Offset | Size | Field | Value for Static x86-64 Executable |
|--------|------|-------|-----------------------------------|
| 0x00 | 16 | e_ident[16] | [7F 45 4C 46 02 01 01 00 00 00 00 00 00 00 00 00] |
| 0x10 | 2 | e_type | 2 (ET_EXEC) |
| 0x12 | 2 | e_machine | 62 (EM_X86_64) |
| 0x14 | 4 | e_version | 1 |
| 0x18 | 8 | e_entry | Virtual address of _start |
| 0x20 | 8 | e_phoff | 64 (immediately after header) |
| 0x28 | 8 | e_shoff | 0 (no section headers for minimal) |
| 0x30 | 4 | e_flags | 0 |
| 0x34 | 2 | e_ehsize | 64 |
| 0x36 | 2 | e_phentsize | 56 |
| 0x38 | 2 | e_phnum | 1-3 (text, data, optional GNU_STACK) |
| 0x3A | 2 | e_shentsize | 64 |
| 0x3C | 2 | e_shnum | 0 |
| 0x3E | 2 | e_shstrndx | 0 |
Total: 64 bytes (0x40)
### Memory Layout of Elf64_Phdr
| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0x00 | 4 | p_type | PT_LOAD (1) for loadable segments |
| 0x04 | 4 | p_flags | PF_R | PF_W | PF_X combination |
| 0x08 | 8 | p_offset | File offset of segment start |
| 0x10 | 8 | p_vaddr | Virtual address to map at |
| 0x18 | 8 | p_paddr | Physical address (usually = p_vaddr) |
| 0x20 | 8 | p_filesz | Bytes in file |
| 0x28 | 8 | p_memsz | Bytes in memory (p_filesz + .bss) |
| 0x30 | 8 | p_align | Alignment (typically 0x1000) |
Total: 56 bytes (0x38)

![Entry Point Resolution Flow](./diagrams/tdd-diag-045.svg)

![Executable Generation Module Architecture](./diagrams/tdd-diag-040.svg)

### Segment Layout Model
```
Executable File Layout:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Offset 0x0000: ELF Header (64 bytes)                                        │
│   - e_ident: [7F 45 4C 46 02 01 01 00 ...]                                  │
│   - e_type: ET_EXEC (2)                                                     │
│   - e_machine: EM_X86_64 (62)                                               │
│   - e_entry: 0x401000 (address of _start)                                   │
│   - e_phoff: 64 (program headers start here)                                │
│   - e_phnum: 2 (text + data segments)                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Offset 0x0040: Program Header Table                                         │
│   [0] PT_LOAD (text)                                                        │
│     p_flags = PF_R | PF_X                                                   │
│     p_offset = 0x1000                                                       │
│     p_vaddr = 0x401000                                                      │
│     p_filesz = 0x200                                                        │
│     p_memsz = 0x200                                                         │
│     p_align = 0x1000                                                        │
│   [1] PT_LOAD (data)                                                        │
│     p_flags = PF_R | PF_W                                                   │
│     p_offset = 0x2000                                                       │
│     p_vaddr = 0x403000                                                      │
│     p_filesz = 0x100                                                        │
│     p_memsz = 0x200 (includes .bss)                                         │
│     p_align = 0x1000                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Offset 0x00D0: Padding (to 0x1000)                                          │
│   Zero bytes to align first segment                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ Offset 0x1000: Text Segment Content                                         │
│   - Merged .text section data                                               │
│   - Merged .rodata section data (if any)                                    │
│   - All patched code with resolved addresses                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ Offset 0x1200: Padding (to 0x2000)                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ Offset 0x2000: Data Segment Content                                         │
│   - Merged .data section data                                               │
│   - (.bss follows in memory but not in file)                                │
└─────────────────────────────────────────────────────────────────────────────┘
Memory Layout After Loading:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Virtual Address 0x401000: Text Segment (RX)                                 │
│   - Mapped from file offset 0x1000, size 0x200                              │
│   - Contains executable code and read-only data                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ Virtual Address 0x403000: Data Segment (RW)                                 │
│   - Mapped from file offset 0x2000, size 0x100                              │
│   - Contains initialized writable data                                      │
│   - Zero-filled extension for .bss (0x100 bytes)                            │
│   - Total memory size: 0x200                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```
### Page Alignment Constraint
The critical invariant for PT_LOAD segments:
```
p_offset % p_align == p_vaddr % p_align
For p_align = 0x1000 (page size):
  - If p_offset = 0x1000, then p_vaddr must end in 0x000 (e.g., 0x401000)
  - If p_offset = 0x2000, then p_vaddr must end in 0x000 (e.g., 0x403000)
This allows the loader to:
  1. mmap(file, p_offset & ~(page_size-1), ...)
  2. Map directly to (p_vaddr & ~(page_size-1))
  3. Both file offset and virtual address have same page offset
```

![Loader mmap Operation](./diagrams/tdd-diag-051.svg)

![Complete ELF Executable File Layout](./diagrams/tdd-diag-041.svg)

## Interface Contracts
### ELF Header Construction Interface
```c
// File: 34_elf_header.h
#ifndef ELF_HEADER_H
#define ELF_HEADER_H
#include "33_elf_exec_types.h"
#include "04_section_types.h"
#include "11_symbol_types.h"
// Initialize an ELF header with default values for static executable
//
// Parameters:
//   hdr - Header structure to initialize (must not be NULL)
//
// Post-conditions:
//   - e_ident populated with magic, class, data, version
//   - e_type = ET_EXEC
//   - e_machine = EM_X86_64
//   - e_version = 1
//   - e_ehsize = 64
//   - e_phentsize = 56
//   - e_shentsize = 64
//   - e_shnum = 0 (no section headers)
//   - e_shstrndx = 0
void elf_header_init(Elf64_Ehdr *hdr);
// Set the entry point in the ELF header
//
// Parameters:
//   hdr    - ELF header (must not be NULL)
//   vaddr  - Virtual address of entry point
//
// Post-conditions:
//   - hdr->e_entry = vaddr
void elf_header_set_entry(Elf64_Ehdr *hdr, uint64_t vaddr);
// Set program header table location
//
// Parameters:
//   hdr     - ELF header (must not be NULL)
//   offset  - File offset of program header table
//   count   - Number of program headers
//
// Post-conditions:
//   - hdr->e_phoff = offset
//   - hdr->e_phnum = count
void elf_header_set_phdr_info(Elf64_Ehdr *hdr, uint64_t offset, uint16_t count);
// Validate ELF header values
//
// Parameters:
//   hdr - ELF header to validate
//
// Returns:
//   1 if valid
//   0 if invalid (check specific fields)
//
// Validates:
//   - Magic bytes are correct
//   - e_type is ET_EXEC
//   - e_machine is EM_X86_64
//   - e_ehsize == 64
//   - e_phentsize == 56
//   - e_entry != 0
int elf_header_validate(const Elf64_Ehdr *hdr);
// Write ELF header to buffer
//
// Parameters:
//   hdr    - ELF header to write
//   buffer - Output buffer (must be at least 64 bytes)
//
// Returns:
//   EXEC_OK on success
//   EXEC_ERR_FILE_WRITE if buffer is NULL
//
// Writes exactly 64 bytes in little-endian format
ExecError elf_header_write(const Elf64_Ehdr *hdr, uint8_t *buffer);
// Write ELF header directly to file
//
// Parameters:
//   hdr   - ELF header to write
//   file  - Open file handle positioned at offset 0
//
// Returns:
//   EXEC_OK on success
//   EXEC_ERR_FILE_WRITE on write failure
ExecError elf_header_write_file(const Elf64_Ehdr *hdr, FILE *file);
#endif // ELF_HEADER_H
```
### Segment Builder Interface
```c
// File: 36_segment_builder.h
#ifndef SEGMENT_BUILDER_H
#define SEGMENT_BUILDER_H
#include "33_elf_exec_types.h"
#include "04_section_types.h"
// Initialize a segment with default values
//
// Parameters:
//   seg - Segment to initialize (must not be NULL)
//
// Post-conditions:
//   - All fields zeroed
//   - p_align = PAGE_SIZE
//   - is_valid = 0
void segment_init(OutputSegment *seg);
// Create text segment from code/read-only sections
//
// Parameters:
//   ctx - Linker context with output sections
//   seg - Output segment to populate
//
// Returns:
//   EXEC_OK on success
//   EXEC_ERR_NO_SECTIONS if no code sections exist
//
// Builds segment from sections with:
//   - SHF_ALLOC set
//   - SHF_EXECINSTR set OR (not SHF_WRITE)
//
// Result:
//   - p_type = PT_LOAD
//   - p_flags = PF_R | PF_X
//   - p_vaddr = lowest section vaddr
//   - p_filesz = total size of file-backed content
//   - p_memsz = total memory size
ExecError build_text_segment(LinkerContext *ctx, OutputSegment *seg);
// Create data segment from writable sections
//
// Parameters:
//   ctx - Linker context with output sections
//   seg - Output segment to populate
//
// Returns:
//   EXEC_OK on success
//   EXEC_ERR_NO_SECTIONS if no data sections exist
//
// Builds segment from sections with:
//   - SHF_ALLOC set
//   - SHF_WRITE set
//
// Result:
//   - p_type = PT_LOAD
//   - p_flags = PF_R | PF_W
//   - p_vaddr = lowest section vaddr
//   - p_filesz = file-backed size (excludes .bss)
//   - p_memsz = memory size (includes .bss)
ExecError build_data_segment(LinkerContext *ctx, OutputSegment *seg);
// Calculate segment file offset
//
// Parameters:
//   seg         - Segment with p_vaddr set
//   base_offset - Starting file offset for content
//   out_offset  - Output: computed file offset
//
// Returns:
//   EXEC_OK on success
//
// Calculates p_offset such that:
//   p_offset % p_align == p_vaddr % p_align
//
// Formula:
//   // Find smallest offset >= base_offset with correct alignment
//   vaddr_page_offset = p_vaddr % p_align
//   base_page = base_offset & ~(p_align - 1)
//   p_offset = base_page + vaddr_page_offset
//   if (p_offset < base_offset) {
//       p_offset += p_align
//   }
ExecError calculate_segment_offset(OutputSegment *seg,
                                    uint64_t base_offset,
                                    uint64_t *out_offset);
// Validate segment alignment constraint
//
// Parameters:
//   seg - Segment to validate
//
// Returns:
//   1 if p_offset % p_align == p_vaddr % p_align
//   0 otherwise
int segment_validate_alignment(const OutputSegment *seg);
// Check for segment overlap
//
// Parameters:
//   segments - Array of segments
//   count    - Number of segments
//
// Returns:
//   1 if any segments overlap in virtual address space
//   0 if no overlap
int segments_overlap(const OutputSegment *segments, size_t count);
// Build all segments for executable
//
// Parameters:
//   ctx      - Linker context
//   segments - Output array (must have MAX_SEGMENTS capacity)
//   count    - Output: number of segments created
//
// Returns:
//   EXEC_OK on success
//   EXEC_ERR_NO_SECTIONS if no allocatable sections
//   EXEC_ERR_SEGMENT_OVERLAP if segments would overlap
//   EXEC_ERR_ALIGNMENT_INVALID if alignment cannot be satisfied
//
// Creates segments in order:
//   1. Text segment (RX)
//   2. Data segment (RW)
//   3. Optional: PT_GNU_STACK (if needed)
ExecError build_all_segments(LinkerContext *ctx,
                              OutputSegment *segments,
                              size_t *count);
// Calculate total file size from segments
//
// Parameters:
//   segments - Array of segments
//   count    - Number of segments
//
// Returns:
//   Total file size (max of p_offset + p_filesz across all segments)
uint64_t calculate_total_file_size(const OutputSegment *segments, size_t count);
#endif // SEGMENT_BUILDER_H
```
### Program Header Writer Interface
```c
// File: 38_program_header.h
#ifndef PROGRAM_HEADER_H
#define PROGRAM_HEADER_H
#include "33_elf_exec_types.h"
// Initialize a program header for PT_LOAD segment
//
// Parameters:
//   phdr  - Program header to initialize
//   seg   - Source segment data
//
// Post-conditions:
//   - All fields copied from segment
//   - p_paddr = p_vaddr (physical = virtual)
void program_header_init(Elf64_Phdr *phdr, const OutputSegment *seg);
// Write program header to buffer
//
// Parameters:
//   phdr   - Program header to write
//   buffer - Output buffer (must be at least 56 bytes)
//   offset - Byte offset within buffer
//
// Returns:
//   EXEC_OK on success
//   EXEC_ERR_FILE_WRITE if buffer is NULL
//
// Writes exactly 56 bytes in little-endian format
ExecError program_header_write(const Elf64_Phdr *phdr,
                                uint8_t *buffer,
                                uint64_t offset);
// Write program header to file
//
// Parameters:
//   phdr  - Program header to write
//   file  - Open file handle
//
// Returns:
//   EXEC_OK on success
//   EXEC_ERR_FILE_WRITE on failure
ExecError program_header_write_file(const Elf64_Phdr *phdr, FILE *file);
// Write all program headers to buffer
//
// Parameters:
//   segments - Array of segments
//   count    - Number of segments
//   buffer   - Output buffer (must be at least count * 56 bytes)
//
// Returns:
//   EXEC_OK on success
//   EXEC_ERR_FILE_WRITE on failure
ExecError program_headers_write_all(const OutputSegment *segments,
                                     size_t count,
                                     uint8_t *buffer);
// Write all program headers to file
//
// Parameters:
//   segments - Array of segments
//   count    - Number of segments
//   file     - Open file handle positioned at e_phoff
//
// Returns:
//   EXEC_OK on success
//   EXEC_ERR_FILE_WRITE on failure
ExecError program_headers_write_all_file(const OutputSegment *segments,
                                          size_t count,
                                          FILE *file);
// Get string name for segment type
//
// Parameters:
//   p_type - Segment type constant
//
// Returns:
//   Static string name (do not free)
const char *segment_type_name(uint32_t p_type);
// Get string representation of flags
//
// Parameters:
//   p_flags - Segment flags
//   buffer  - Output buffer for string
//   size    - Buffer size
//
// Returns:
//   Pointer to buffer (contains "R", "W", "X" combinations)
char *segment_flags_string(uint32_t p_flags, char *buffer, size_t size);
#endif // PROGRAM_HEADER_H
```
### Entry Point Resolution Interface
```c
// File: 40_entry_point.h
#ifndef ENTRY_POINT_H
#define ENTRY_POINT_H
#include "33_elf_exec_types.h"
#include "11_symbol_types.h"
#include "04_section_types.h"
// Find entry point symbol in global symbol table
//
// Parameters:
//   ctx - Linker context with resolved symbols
//
// Returns:
//   Pointer to GlobalSymbol for entry point
//   NULL if not found
//
// Lookup order:
//   1. "_start" (standard Unix entry point)
//   2. "main" (fallback for simple programs)
//
// Note: Returns first match found; does not validate symbol type
GlobalSymbol *find_entry_point(LinkerContext *ctx);
// Get entry point address
//
// Parameters:
//   ctx       - Linker context
//   out_addr  - Output: virtual address of entry point
//   out_name  - Output: symbol name (optional, may be NULL)
//
// Returns:
//   EXEC_OK if entry point found
//   EXEC_ERR_ENTRY_NOT_FOUND if neither _start nor main exists
//
// On success:
//   - *out_addr = entry symbol's final_address
//   - *out_name = "main" or "_start" (static string, do not free)
ExecError get_entry_point_address(LinkerContext *ctx,
                                   uint64_t *out_addr,
                                   const char **out_name);
// Validate entry point
//
// Parameters:
//   ctx   - Linker context
//   sym   - Entry point symbol
//
// Returns:
//   1 if valid entry point
//   0 if invalid
//
// Valid if:
//   - Symbol is resolved (state == SYM_STATE_RESOLVED)
//   - Symbol address != 0
//   - Symbol is in an executable section (optional check)
int validate_entry_point(LinkerContext *ctx, GlobalSymbol *sym);
// Report entry point resolution for debugging
//
// Parameters:
//   ctx - Linker context
//
// Prints:
//   - Entry point symbol name
//   - Entry point virtual address
//   - Which section contains the entry point
void report_entry_point(LinkerContext *ctx);
#endif // ENTRY_POINT_H
```
### Executable Writer Interface
```c
// File: 42_exec_writer.h
#ifndef EXEC_WRITER_H
#define EXEC_WRITER_H
#include "33_elf_exec_types.h"
#include "04_section_types.h"
// Calculate complete executable layout
//
// Parameters:
//   ctx    - Linker context with merged sections and addresses
//   layout - Output layout structure
//
// Returns:
//   EXEC_OK on success
//   EXEC_ERR_NO_SECTIONS if no output sections
//   EXEC_ERR_INVALID_LAYOUT on calculation error
//
// Computes:
//   - Program header count and offset
//   - Content start offset (page-aligned)
//   - Total file size
//   - Segment placement
ExecError calculate_executable_layout(LinkerContext *ctx,
                                       ExecutableLayout *layout);
// Build complete executable in memory
//
// Parameters:
//   ctx          - Linker context
//   buffer       - Output buffer (must be layout->total_file_size bytes)
//   buffer_size  - Size of buffer
//   layout       - Pre-calculated layout
//   segments     - Pre-built segments
//   segment_count - Number of segments
//
// Returns:
//   EXEC_OK on success
//   EXEC_ERR_FILE_WRITE if buffer too small
//   EXEC_ERR_INVALID_LAYOUT if layout is inconsistent
//
// Assembles:
//   1. ELF header at offset 0
//   2. Program headers at offset 64
//   3. Padding to first segment
//   4. Segment content at calculated offsets
ExecError build_executable_buffer(LinkerContext *ctx,
                                   uint8_t *buffer,
                                   size_t buffer_size,
                                   const ExecutableLayout *layout,
                                   const OutputSegment *segments,
                                   size_t segment_count);
// Write complete executable to file
//
// Parameters:
//   ctx         - Linker context with all data ready
//   output_path - Path for output executable
//   stats       - Output: statistics (may be NULL)
//
// Returns:
//   EXEC_OK on success
//   EXEC_ERR_ENTRY_NOT_FOUND if no entry point
//   EXEC_ERR_NO_SECTIONS if no content
//   EXEC_ERR_FILE_OPEN if cannot create file
//   EXEC_ERR_FILE_WRITE if write fails
//   EXEC_ERR_CHMOD if cannot set permissions (file still created)
//
// Full pipeline:
//   1. Find entry point
//   2. Build segments
//   3. Calculate layout
//   4. Allocate buffer
//   5. Write headers and content
//   6. Write to file
//   7. Set executable permissions (chmod 0755)
ExecError write_executable(LinkerContext *ctx,
                            const char *output_path,
                            ExecStats *stats);
// Get statistics about generated executable
//
// Parameters:
//   ctx      - Linker context
//   segments - Built segments
//   count    - Number of segments
//   stats    - Output statistics
void get_executable_stats(LinkerContext *ctx,
                           const OutputSegment *segments,
                           size_t count,
                           ExecStats *stats);
// Get human-readable error string
//
// Parameters:
//   err - Error code
//
// Returns:
//   Static string description (do not free)
const char *exec_error_string(ExecError err);
// Verify generated executable (optional debugging)
//
// Parameters:
//   path - Path to executable
//
// Returns:
//   EXEC_OK if executable appears valid
//   Appropriate error if verification fails
//
// Runs:
//   - File command to verify ELF type
//   - readelf -h to check header
//   - readelf -l to check segments
ExecError verify_executable(const char *path);
#endif // EXEC_WRITER_H
```
## Algorithm Specification
### Algorithm 1: ELF Header Initialization
**Purpose**: Create a valid ELF64 header for a static executable.
**Input**:
- `hdr`: Elf64_Ehdr structure to initialize
**Output**:
- Initialized header with all required fields
**Procedure**:
```
1. Clear header structure
   memset(hdr, 0, sizeof(Elf64_Ehdr))
2. Set ELF identification bytes
   hdr->e_ident[EI_MAG0] = 0x7F
   hdr->e_ident[EI_MAG1] = 'E'
   hdr->e_ident[EI_MAG2] = 'L'
   hdr->e_ident[EI_MAG3] = 'F'
   hdr->e_ident[EI_CLASS] = ELFCLASS64
   hdr->e_ident[EI_DATA] = ELFDATA2LSB
   hdr->e_ident[EI_VERSION] = 1
   hdr->e_ident[EI_OSABI] = 0  // ELFOSABI_NONE
   hdr->e_ident[EI_ABIVERSION] = 0
   // Bytes 9-15 remain zero (padding)
3. Set file type and architecture
   hdr->e_type = ET_EXEC
   hdr->e_machine = EM_X86_64
   hdr->e_version = 1
4. Initialize header sizes
   hdr->e_ehsize = 64
   hdr->e_phentsize = 56
   hdr->e_shentsize = 64
5. Clear optional fields
   hdr->e_entry = 0  // Set later
   hdr->e_phoff = 0  // Set later
   hdr->e_shoff = 0  // No section headers
   hdr->e_flags = 0
   hdr->e_phnum = 0  // Set later
   hdr->e_shnum = 0
   hdr->e_shstrndx = 0
6. RETURN
```
**Invariants after execution**:
- Header is valid for a 64-bit little-endian static executable
- Entry point must be set before writing
- Program header info must be set before writing
### Algorithm 2: Entry Point Resolution
**Purpose**: Find the entry point symbol and return its address.
**Input**:
- `ctx`: Linker context with global symbol table
**Output**:
- `out_addr`: Virtual address of entry point
- `out_name`: Symbol name ("_start" or "main")
- Return: `ExecError` code
**Procedure**:
```
1. Try to find _start symbol
   err = global_symbol_lookup(&ctx->global_syms, "_start", &sym)
   IF err == SYM_OK AND sym != NULL:
     a. Validate symbol is resolved
        IF sym->state != SYM_STATE_RESOLVED:
          GOTO try_main
     b. Validate address is non-zero
        IF sym->final_address == 0:
          GOTO try_main
     c. Return _start
        *out_addr = sym->final_address
        *out_name = "_start"
        RETURN EXEC_OK
2. LABEL try_main:
   err = global_symbol_lookup(&ctx->global_syms, "main", &sym)
   IF err == SYM_OK AND sym != NULL:
     a. Validate symbol is resolved
        IF sym->state != SYM_STATE_RESOLVED:
          GOTO not_found
     b. Validate address is non-zero
        IF sym->final_address == 0:
          GOTO not_found
     c. Return main
        *out_addr = sym->final_address
        *out_name = "main"
        RETURN EXEC_OK
3. LABEL not_found:
   *out_addr = 0
   *out_name = NULL
   RETURN EXEC_ERR_ENTRY_NOT_FOUND
```
**Error handling**:
- `EXEC_ERR_ENTRY_NOT_FOUND`: Neither `_start` nor `main` exists or resolved
### Algorithm 3: Text Segment Construction
**Purpose**: Build the PT_LOAD segment for code and read-only data.
**Input**:
- `ctx`: Linker context with output sections
**Output**:
- `seg`: OutputSegment populated with text segment info
- Return: `ExecError` code
**Procedure**:
```
1. Initialize segment
   segment_init(seg)
   strncpy(seg->name, "text", MAX_SEGMENT_NAME - 1)
2. Find all code/read-only sections
   min_vaddr = UINT64_MAX
   max_vaddr = 0
   total_file_size = 0
   total_mem_size = 0
   FOR i = 0 TO ctx->output_count - 1:
     sec = &ctx->outputs[i]
     a. Check if section belongs in text segment
        // Must be allocatable
        IF NOT (sec->flags & SHF_ALLOC):
          CONTINUE
        // Must be executable OR read-only (not writable)
        IF (sec->flags & SHF_WRITE):
          CONTINUE
        // Skip empty sections
        IF sec->mem_size == 0:
          CONTINUE
     b. Track bounds
        IF sec->virtual_addr < min_vaddr:
          min_vaddr = sec->virtual_addr
        sec_end = sec->virtual_addr + sec->mem_size
        IF sec_end > max_vaddr:
          max_vaddr = sec_end
     c. Accumulate sizes
        total_file_size += sec->file_size
        total_mem_size += sec->mem_size
3. Validate we found sections
   IF min_vaddr == UINT64_MAX:
     seg->is_valid = 0
     RETURN EXEC_ERR_NO_SECTIONS
4. Set segment properties
   seg->p_type = PT_LOAD
   seg->p_flags = PF_R | PF_X
   seg->p_vaddr = min_vaddr
   seg->p_paddr = min_vaddr
   seg->p_filesz = max_vaddr - min_vaddr  // Total span
   seg->p_memsz = max_vaddr - min_vaddr
   seg->p_align = PAGE_SIZE
   seg->is_valid = 1
5. RETURN EXEC_OK
```
**Note**: `p_filesz` is the span from min_vaddr to max_vaddr, which correctly handles gaps between sections due to alignment.
### Algorithm 4: Data Segment Construction
**Purpose**: Build the PT_LOAD segment for writable data and .bss.
**Input**:
- `ctx`: Linker context with output sections
**Output**:
- `seg`: OutputSegment populated with data segment info
- Return: `ExecError` code
**Procedure**:
```
1. Initialize segment
   segment_init(seg)
   strncpy(seg->name, "data", MAX_SEGMENT_NAME - 1)
2. Find all writable sections
   min_vaddr = UINT64_MAX
   max_vaddr = 0
   max_file_vaddr = 0  // End of file-backed content
   FOR i = 0 TO ctx->output_count - 1:
     sec = &ctx->outputs[i]
     a. Check if section belongs in data segment
        // Must be allocatable AND writable
        IF NOT (sec->flags & SHF_ALLOC):
          CONTINUE
        IF NOT (sec->flags & SHF_WRITE):
          CONTINUE
        // Skip empty sections
        IF sec->mem_size == 0:
          CONTINUE
     b. Track bounds
        IF sec->virtual_addr < min_vaddr:
          min_vaddr = sec->virtual_addr
        sec_end = sec->virtual_addr + sec->mem_size
        IF sec_end > max_vaddr:
          max_vaddr = sec_end
     c. Track file-backed end (excludes .bss)
        IF sec->file_size > 0:
          file_end = sec->virtual_addr + sec->file_size
          IF file_end > max_file_vaddr:
            max_file_vaddr = file_end
3. Validate we found sections
   IF min_vaddr == UINT64_MAX:
     seg->is_valid = 0
     RETURN EXEC_ERR_NO_SECTIONS
4. Set segment properties
   seg->p_type = PT_LOAD
   seg->p_flags = PF_R | PF_W
   seg->p_vaddr = min_vaddr
   seg->p_paddr = min_vaddr
   seg->p_filesz = max_file_vaddr - min_vaddr  // File-backed only
   seg->p_memsz = max_vaddr - min_vaddr        // Includes .bss
   seg->p_align = PAGE_SIZE
   seg->is_valid = 1
5. RETURN EXEC_OK
```
**Key insight**: `p_memsz > p_filesz` when .bss is present. The loader zero-fills the difference.

![Minimal Executable Example Trace](./diagrams/tdd-diag-052.svg)

![Elf64_Phdr Program Header Memory Layout](./diagrams/tdd-diag-043.svg)

### Algorithm 5: Segment File Offset Calculation
**Purpose**: Calculate file offset for a segment satisfying alignment constraint.
**Input**:
- `seg`: Segment with p_vaddr and p_align set
- `base_offset`: Minimum acceptable file offset
**Output**:
- `out_offset`: Computed file offset
- `seg->p_offset` set
- Return: `ExecError` code
**Procedure**:
```
1. Validate alignment is power of 2
   IF seg->p_align == 0 OR (seg->p_align & (seg->p_align - 1)) != 0:
     RETURN EXEC_ERR_ALIGNMENT_INVALID
2. Calculate required page offset
   // p_vaddr % p_align tells us the offset within a page
   vaddr_page_offset = seg->p_vaddr % seg->p_align
3. Find page-aligned base
   base_page = base_offset & ~(seg->p_align - 1)
4. Calculate candidate offset
   candidate = base_page + vaddr_page_offset
5. If candidate is before base_offset, advance to next page
   IF candidate < base_offset:
     candidate += seg->p_align
6. Set segment offset
   seg->p_offset = candidate
   *out_offset = candidate
7. RETURN EXEC_OK
```
**Example**:
```
p_vaddr = 0x401000
p_align = 0x1000
base_offset = 0x00D0 (after headers)
vaddr_page_offset = 0x401000 % 0x1000 = 0x000
base_page = 0x00D0 & ~0xFFF = 0x0000
candidate = 0x0000 + 0x000 = 0x0000
Since 0x0000 < 0x00D0:
  candidate = 0x0000 + 0x1000 = 0x1000
Result: p_offset = 0x1000
Check: 0x1000 % 0x1000 == 0x401000 % 0x1000 ✓
```
### Algorithm 6: Segment Overlap Detection
**Purpose**: Ensure no two segments overlap in virtual address space.
**Input**:
- `segments`: Array of segments
- `count`: Number of segments
**Output**:
- Returns 1 if overlap detected, 0 if no overlap
**Procedure**:
```
1. For each pair of segments
   FOR i = 0 TO count - 1:
     IF NOT segments[i].is_valid:
       CONTINUE
     IF segments[i].p_type != PT_LOAD:
       CONTINUE
     FOR j = i + 1 TO count - 1:
       IF NOT segments[j].is_valid:
         CONTINUE
       IF segments[j].p_type != PT_LOAD:
         CONTINUE
       a. Get address ranges
          start_i = segments[i].p_vaddr
          end_i = start_i + segments[i].p_memsz
          start_j = segments[j].p_vaddr
          end_j = start_j + segments[j].p_memsz
       b. Check for overlap
          IF start_i < end_j AND start_j < end_i:
            // Overlap detected
            RETURN 1
2. No overlap found
   RETURN 0
```
### Algorithm 7: Executable Layout Calculation
**Purpose**: Compute complete file layout for the executable.
**Input**:
- `ctx`: Linker context with merged sections
**Output**:
- `layout`: ExecutableLayout structure
- Return: `ExecError` code
**Procedure**:
```
1. Initialize layout
   layout->elf_header_offset = 0
   layout->phdr_offset = sizeof(Elf64_Ehdr)  // 64
2. Build all segments
   err = build_all_segments(ctx, segments, &segment_count)
   IF err != EXEC_OK:
     RETURN err
3. Count valid segments
   phnum = 0
   FOR i = 0 TO segment_count - 1:
     IF segments[i].is_valid:
       phnum++
   layout->phnum = phnum
4. Calculate program header table size
   layout->phdr_size = phnum * sizeof(Elf64_Phdr)  // 56 bytes each
5. Calculate first segment offset
   // Headers end at: phdr_offset + phdr_size
   headers_end = layout->phdr_offset + layout->phdr_size
   // Content starts at next page boundary
   layout->content_offset = align_up(headers_end, PAGE_SIZE)
6. Assign file offsets to each segment
   current_offset = layout->content_offset
   FOR i = 0 TO segment_count - 1:
     IF NOT segments[i].is_valid:
       CONTINUE
     err = calculate_segment_offset(&segments[i], current_offset, &actual_offset)
     IF err != EXEC_OK:
       RETURN err
     current_offset = actual_offset + segments[i].p_filesz
7. Calculate total file size
   layout->total_file_size = 0
   FOR i = 0 TO segment_count - 1:
     IF segments[i].is_valid:
       seg_end = segments[i].p_offset + segments[i].p_filesz
       IF seg_end > layout->total_file_size:
         layout->total_file_size = seg_end
8. Validate no overlap
   IF segments_overlap(segments, segment_count):
     RETURN EXEC_ERR_SEGMENT_OVERLAP
9. Store entry point
   err = get_entry_point_address(ctx, &layout->entry_point, NULL)
   IF err != EXEC_OK:
     RETURN err
10. Set segment flags
    layout->has_text_segment = 0
    layout->has_data_segment = 0
    FOR i = 0 TO segment_count - 1:
      IF segments[i].is_valid:
        IF segments[i].p_flags & PF_X:
          layout->has_text_segment = 1
        IF segments[i].p_flags & PF_W:
          layout->has_data_segment = 1
11. RETURN EXEC_OK
```

![Page Alignment Constraint Validation](./diagrams/tdd-diag-044.svg)

### Algorithm 8: Executable Buffer Assembly
**Purpose**: Assemble complete executable in memory buffer.
**Input**:
- `ctx`: Linker context
- `buffer`: Output buffer
- `buffer_size`: Size of buffer
- `layout`: Pre-calculated layout
- `segments`: Pre-built segments
**Output**:
- Buffer filled with executable content
- Return: `ExecError` code
**Procedure**:
```
1. Validate buffer size
   IF buffer_size < layout->total_file_size:
     RETURN EXEC_ERR_FILE_WRITE
2. Zero entire buffer
   memset(buffer, 0, buffer_size)
3. Build and write ELF header
   Elf64_Ehdr ehdr
   elf_header_init(&ehdr)
   elf_header_set_entry(&ehdr, layout->entry_point)
   elf_header_set_phdr_info(&ehdr, layout->phdr_offset, layout->phnum)
   elf_header_write(&ehdr, buffer)
4. Write program headers
   phdr_buffer = buffer + layout->phdr_offset
   FOR i = 0 TO segment_count - 1:
     IF NOT segments[i].is_valid:
       CONTINUE
     Elf64_Phdr phdr
     program_header_init(&phdr, &segments[i])
     program_header_write(&phdr, phdr_buffer, i * sizeof(Elf64_Phdr))
5. Copy segment content from output buffer
   FOR i = 0 TO segment_count - 1:
     IF NOT segments[i].is_valid:
       CONTINUE
     IF segments[i].p_filesz == 0:
       CONTINUE  // Skip .bss-only segments
     a. Find sections belonging to this segment
        FOR j = 0 TO ctx->output_count - 1:
          sec = &ctx->outputs[j]
          // Check section belongs to this segment
          IF sec->virtual_addr < segments[i].p_vaddr:
            CONTINUE
          IF sec->virtual_addr >= segments[i].p_vaddr + segments[i].p_memsz:
            CONTINUE
          IF sec->file_size == 0:
            CONTINUE  // Skip .bss
          b. Calculate where to copy
             sec_offset_in_seg = sec->virtual_addr - segments[i].p_vaddr
             file_dest = buffer + segments[i].p_offset + sec_offset_in_seg
          c. Copy section data
             IF sec->data != NULL:
               memcpy(file_dest, sec->data, sec->file_size)
6. RETURN EXEC_OK
```
### Algorithm 9: Complete Executable File Writing
**Purpose**: Write complete executable to file with proper permissions.
**Input**:
- `ctx`: Linker context with all data ready
- `output_path`: Path for output file
**Output**:
- Executable file written to disk
- `stats`: Statistics about generated file (optional)
- Return: `ExecError` code
**Procedure**:
```
1. Find entry point
   err = get_entry_point_address(ctx, &entry_addr, &entry_name)
   IF err != EXEC_OK:
     fprintf(stderr, "error: no entry point symbol (_start or main)\n")
     RETURN err
2. Build segments
   OutputSegment segments[MAX_SEGMENTS]
   size_t segment_count = 0
   err = build_all_segments(ctx, segments, &segment_count)
   IF err != EXEC_OK:
     RETURN err
3. Calculate layout
   ExecutableLayout layout
   err = calculate_executable_layout(ctx, &layout)
   IF err != EXEC_OK:
     RETURN err
4. Allocate output buffer
   buffer = calloc(layout.total_file_size, 1)
   IF buffer == NULL:
     RETURN EXEC_ERR_MEMORY
5. Build executable in buffer
   err = build_executable_buffer(ctx, buffer, layout.total_file_size,
                                  &layout, segments, segment_count)
   IF err != EXEC_OK:
     free(buffer)
     RETURN err
6. Open output file
   file = fopen(output_path, "wb")
   IF file == NULL:
     free(buffer)
     RETURN EXEC_ERR_FILE_OPEN
7. Write buffer to file
   bytes_written = fwrite(buffer, 1, layout.total_file_size, file)
   fclose(file)
   IF bytes_written != layout.total_file_size:
     free(buffer)
     RETURN EXEC_ERR_FILE_WRITE
8. Free buffer
   free(buffer)
9. Set executable permissions
   result = chmod(output_path, 0755)
   IF result != 0:
     // File is still valid, just not executable
     fprintf(stderr, "warning: could not set executable permissions\n")
     // Don't fail - user can chmod manually
10. Populate statistics
    IF stats != NULL:
      get_executable_stats(ctx, segments, segment_count, stats)
      stats->total_file_size = layout.total_file_size
      stats->entry_address = entry_addr
      stats->entry_symbol = entry_name
11. Print summary
    printf("Generated executable: %s\n", output_path)
    printf("  Entry point: %s @ 0x%lx\n", entry_name, entry_addr)
    printf("  File size: %lu bytes\n", layout.total_file_size)
    printf("  Segments: %d\n", layout.phnum)
12. RETURN EXEC_OK
```
## Error Handling Matrix
| Error | Detected By | Recovery | User-Visible? | System State |
|-------|-------------|----------|---------------|--------------|
| `EXEC_ERR_MEMORY` | `calloc()` returns NULL | Abort, report error | Yes: "Out of memory" | Clean, no partial file |
| `EXEC_ERR_ENTRY_NOT_FOUND` | `find_entry_point()` returns NULL | Abort, report error | Yes: "error: no entry point symbol (_start or main)" | Clean |
| `EXEC_ERR_NO_SECTIONS` | No allocatable sections found | Abort | Yes: "No output sections to emit" | Clean |
| `EXEC_ERR_ALIGNMENT_INVALID` | Alignment constraint violated | Abort, report details | Yes: "Segment alignment constraint violated: p_offset=0x1234, p_vaddr=0x401000" | Clean |
| `EXEC_ERR_SEGMENT_OVERLAP` | Two segments overlap | Abort, report which | Yes: "Segments overlap in memory: text and data" | Clean |
| `EXEC_ERR_FILE_OPEN` | `fopen()` returns NULL | Abort | Yes: "Cannot create output file: permission denied" | Clean |
| `EXEC_ERR_FILE_WRITE` | `fwrite()` returns short count | Abort, remove partial | Yes: "Write failed: disk full" | Partial file may exist |
| `EXEC_ERR_CHMOD` | `chmod()` returns -1 | Continue with warning | Yes: "warning: could not set executable permissions" | File exists, may need chmod |
| `EXEC_ERR_INVALID_LAYOUT` | Internal calculation error | Abort | Yes: "Internal error: invalid layout calculation" | Clean |
| `EXEC_ERR_NO_INPUTS` | `ctx == NULL` or no inputs | Abort | Yes: "No linker context provided" | Clean |
### Error Message Format
```c
// Entry point not found
fprintf(stderr, "ld: error: no entry point symbol\n");
fprintf(stderr, ">>> looking for: _start or main\n");
fprintf(stderr, ">>> available symbols:\n");
// List a few global symbols as hints
// Alignment error
fprintf(stderr, "ld: error: segment alignment constraint violated\n");
fprintf(stderr, ">>> segment: %s\n", seg->name);
fprintf(stderr, ">>> p_offset = 0x%lx\n", seg->p_offset);
fprintf(stderr, ">>> p_vaddr = 0x%lx\n", seg->p_vaddr);
fprintf(stderr, ">>> p_align = 0x%lx\n", seg->p_align);
fprintf(stderr, ">>> p_offset %% p_align = 0x%lx\n", seg->p_offset % seg->p_align);
fprintf(stderr, ">>> p_vaddr %% p_align = 0x%lx\n", seg->p_vaddr % seg->p_align);
// Segment overlap
fprintf(stderr, "ld: error: segments overlap in virtual address space\n");
fprintf(stderr, ">>> %s: 0x%lx - 0x%lx\n", seg1->name, seg1->p_vaddr, 
        seg1->p_vaddr + seg1->p_memsz);
fprintf(stderr, ">>> %s: 0x%lx - 0x%lx\n", seg2->name, seg2->p_vaddr,
        seg2->p_vaddr + seg2->p_memsz);
```
## Implementation Sequence with Checkpoints
### Phase 1: ELF Header Construction (1-2 hours)
**Files**: `33_elf_exec_types.h`, `34_elf_header.h`, `35_elf_header.c`
**Implementation steps**:
1. Define all ELF constants and structures in `33_elf_exec_types.h`
2. Declare header interface in `34_elf_header.h`
3. Implement `elf_header_init()` with all required fields
4. Implement `elf_header_set_entry()` and `elf_header_set_phdr_info()`
5. Implement `elf_header_validate()` for sanity checking
6. Implement `elf_header_write()` for buffer serialization
7. Write unit tests for header construction
**Checkpoint**:
```bash
gcc -c 35_elf_header.c -o elf_header.o
gcc tests/test_elf_header.c elf_header.o -o test_elf_header
./test_elf_header
# Expected output:
# [PASS] Header initialization
# [PASS] Magic bytes correct
# [PASS] Entry point setting
# [PASS] Program header info setting
# [PASS] Header validation
# [PASS] Buffer serialization
# [PASS] Little-endian byte order
# All tests passed!
```
At this point you can construct valid ELF headers.
### Phase 2: Entry Point Resolution (1 hour)
**Files**: `40_entry_point.h`, `41_entry_point.c`
**Implementation steps**:
1. Implement `find_entry_point()` with _start/main priority
2. Implement `get_entry_point_address()` wrapper
3. Implement `validate_entry_point()` checks
4. Implement `report_entry_point()` for debugging
5. Write unit tests with mock symbol table
**Checkpoint**:
```bash
gcc -c 41_entry_point.c -o entry_point.o
gcc tests/test_entry.c entry_point.o -o test_entry
./test_entry
# Expected output:
# [PASS] Find _start symbol
# [PASS] Fallback to main
# [PASS] No entry point error
# [PASS] Unresolved entry point error
# [PASS] Entry point validation
# All tests passed!
```
At this point you can resolve entry points.
### Phase 3: Segment Construction (2-3 hours)
**Files**: `36_segment_builder.h`, `37_segment_builder.c`
**Implementation steps**:
1. Implement `segment_init()` helper
2. Implement `build_text_segment()` with section filtering
3. Implement `build_data_segment()` with .bss handling
4. Implement `calculate_segment_offset()` with alignment constraint
5. Implement `segment_validate_alignment()` checker
6. Implement `segments_overlap()` detection
7. Implement `build_all_segments()` orchestration
8. Write unit tests for segment construction
**Checkpoint**:
```bash
gcc -c 37_segment_builder.c -o segment_builder.o
gcc tests/test_segments.c segment_builder.o -o test_segments
./test_segments
# Expected output:
# [PASS] Text segment construction
# [PASS] Data segment construction
# [PASS] Segment offset calculation
# [PASS] Alignment validation
# [PASS] Alignment constraint satisfaction
# [PASS] Overlap detection
# [PASS] .bss memory size vs file size
# [PASS] Empty section handling
# All tests passed!
```
At this point you can construct valid segments.
### Phase 4: Program Header Writing (1-2 hours)
**Files**: `38_program_header.h`, `39_program_header.c`
**Implementation steps**:
1. Implement `program_header_init()` from segment
2. Implement `program_header_write()` for single header
3. Implement `program_headers_write_all()` for array
4. Implement `segment_type_name()` for debugging
5. Implement `segment_flags_string()` for debugging
6. Write unit tests for program header serialization
**Checkpoint**:
```bash
gcc -c 39_program_header.c -o program_header.o
gcc tests/test_phdr.c program_header.o -o test_phdr
./test_phdr
# Expected output:
# [PASS] Program header initialization
# [PASS] PT_LOAD header writing
# [PASS] Flags serialization
# [PASS] Multiple headers
# [PASS] Little-endian byte order
# [PASS] Segment type names
# [PASS] Flags string representation
# All tests passed!
```
At this point you can serialize program headers.
### Phase 5: Executable Assembly and Writing (2-3 hours)
**Files**: `42_exec_writer.h`, `43_exec_writer.c`
**Implementation steps**:
1. Implement `calculate_executable_layout()` orchestration
2. Implement `build_executable_buffer()` assembly
3. Implement `write_executable()` full pipeline
4. Implement `get_executable_stats()` statistics
5. Implement `exec_error_string()` error messages
6. Implement `verify_executable()` optional verification
7. Write end-to-end tests
**Checkpoint**:
```bash
# Create minimal test program
cat > /tmp/minimal.s << 'EOF'
.globl _start
_start:
    mov $42, %rdi
    mov $60, %rax
    syscall
EOF
as -o /tmp/minimal.o /tmp/minimal.s
# Run full pipeline test
gcc tests/test_exec_full.c elf_header.o entry_point.o segment_builder.o \
    program_header.o exec_writer.o -o test_exec_full
./test_exec_full
# Expected output:
# [PASS] Complete executable generation
# [PASS] Entry point resolution
# [PASS] Segment layout
# [PASS] File writing
# [PASS] Executable permissions
# [PASS] File execution
# 
# Executing generated program...
# Exit code: 42
# All tests passed!
```
**Milestone 4 Complete**: Full executable generation producing runnable Linux binaries.
## Test Specification
### Test Suite: ELF Header Generation
```c
// tests/test_elf_header.c
void test_header_init(void) {
    Elf64_Ehdr hdr;
    elf_header_init(&hdr);
    // Check magic bytes
    ASSERT_EQ(hdr.e_ident[EI_MAG0], 0x7F);
    ASSERT_EQ(hdr.e_ident[EI_MAG1], 'E');
    ASSERT_EQ(hdr.e_ident[EI_MAG2], 'L');
    ASSERT_EQ(hdr.e_ident[EI_MAG3], 'F');
    // Check class and data
    ASSERT_EQ(hdr.e_ident[EI_CLASS], ELFCLASS64);
    ASSERT_EQ(hdr.e_ident[EI_DATA], ELFDATA2LSB);
    // Check type and machine
    ASSERT_EQ(hdr.e_type, ET_EXEC);
    ASSERT_EQ(hdr.e_machine, EM_X86_64);
    // Check sizes
    ASSERT_EQ(hdr.e_ehsize, 64);
    ASSERT_EQ(hdr.e_phentsize, 56);
}
void test_entry_point_setting(void) {
    Elf64_Ehdr hdr;
    elf_header_init(&hdr);
    elf_header_set_entry(&hdr, 0x401000);
    ASSERT_EQ(hdr.e_entry, 0x401000);
    elf_header_set_entry(&hdr, 0x7FFFFFFFFFFF);
    ASSERT_EQ(hdr.e_entry, 0x7FFFFFFFFFFF);
}
void test_phdr_info_setting(void) {
    Elf64_Ehdr hdr;
    elf_header_init(&hdr);
    elf_header_set_phdr_info(&hdr, 64, 2);
    ASSERT_EQ(hdr.e_phoff, 64);
    ASSERT_EQ(hdr.e_phnum, 2);
}
void test_header_validation(void) {
    Elf64_Ehdr hdr;
    elf_header_init(&hdr);
    elf_header_set_entry(&hdr, 0x401000);
    elf_header_set_phdr_info(&hdr, 64, 2);
    ASSERT_TRUE(elf_header_validate(&hdr));
    // Invalidate by clearing magic
    hdr.e_ident[EI_MAG0] = 0;
    ASSERT_FALSE(elf_header_validate(&hdr));
    // Restore and invalidate by wrong type
    hdr.e_ident[EI_MAG0] = 0x7F;
    hdr.e_type = ET_REL;
    ASSERT_FALSE(elf_header_validate(&hdr));
}
void test_header_serialization(void) {
    Elf64_Ehdr hdr;
    elf_header_init(&hdr);
    elf_header_set_entry(&hdr, 0x401234);
    uint8_t buffer[64];
    ExecError err = elf_header_write(&hdr, buffer);
    ASSERT_EQ(err, EXEC_OK);
    // Verify magic at offset 0
    ASSERT_EQ(buffer[0], 0x7F);
    ASSERT_EQ(buffer[1], 'E');
    ASSERT_EQ(buffer[2], 'L');
    ASSERT_EQ(buffer[3], 'F');
    // Verify entry point at offset 0x18 (little-endian)
    uint64_t entry = (uint64_t)buffer[0x18] |
                     ((uint64_t)buffer[0x19] << 8) |
                     ((uint64_t)buffer[0x1A] << 16) |
                     ((uint64_t)buffer[0x1B] << 24) |
                     ((uint64_t)buffer[0x1C] << 32) |
                     ((uint64_t)buffer[0x1D] << 40) |
                     ((uint64_t)buffer[0x1E] << 48) |
                     ((uint64_t)buffer[0x1F] << 56);
    ASSERT_EQ(entry, 0x401234);
}
```
### Test Suite: Entry Point Resolution
```c
// tests/test_entry.c
void test_find_start_symbol(void) {
    // Create mock context with _start
    LinkerContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    global_symbol_table_init(&ctx.global_syms, 16);
    GlobalSymbol *start;
    global_symbol_insert(&ctx.global_syms, "_start", &start);
    start->state = SYM_STATE_RESOLVED;
    start->final_address = 0x401000;
    GlobalSymbol *found = find_entry_point(&ctx);
    ASSERT_NOT_NULL(found);
    ASSERT_STREQ(found->name, "_start");
    ASSERT_EQ(found->final_address, 0x401000);
    global_symbol_table_destroy(&ctx.global_syms);
}
void test_fallback_to_main(void) {
    LinkerContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    global_symbol_table_init(&ctx.global_syms, 16);
    // Only add main, not _start
    GlobalSymbol *main_sym;
    global_symbol_insert(&ctx.global_syms, "main", &main_sym);
    main_sym->state = SYM_STATE_RESOLVED;
    main_sym->final_address = 0x401020;
    GlobalSymbol *found = find_entry_point(&ctx);
    ASSERT_NOT_NULL(found);
    ASSERT_STREQ(found->name, "main");
    global_symbol_table_destroy(&ctx.global_syms);
}
void test_start_priority_over_main(void) {
    LinkerContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    global_symbol_table_init(&ctx.global_syms, 16);
    // Add both
    GlobalSymbol *start, *main_sym;
    global_symbol_insert(&ctx.global_syms, "_start", &start);
    global_symbol_insert(&ctx.global_syms, "main", &main_sym);
    start->state = SYM_STATE_RESOLVED;
    start->final_address = 0x401000;
    main_sym->state = SYM_STATE_RESOLVED;
    main_sym->final_address = 0x401020;
    GlobalSymbol *found = find_entry_point(&ctx);
    ASSERT_STREQ(found->name, "_start");  // _start wins
    global_symbol_table_destroy(&ctx.global_syms);
}
void test_no_entry_point(void) {
    LinkerContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    global_symbol_table_init(&ctx.global_syms, 16);
    GlobalSymbol *found = find_entry_point(&ctx);
    ASSERT_NULL(found);
    uint64_t addr;
    ExecError err = get_entry_point_address(&ctx, &addr, NULL);
    ASSERT_EQ(err, EXEC_ERR_ENTRY_NOT_FOUND);
    global_symbol_table_destroy(&ctx.global_syms);
}
```
### Test Suite: Segment Construction
```c
// tests/test_segments.c
void test_text_segment_construction(void) {
    // Create mock context with text section
    LinkerContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.outputs = calloc(2, sizeof(OutputSection));
    ctx.output_count = 2;
    // .text section
    strncpy(ctx.outputs[0].name, ".text", 63);
    ctx.outputs[0].flags = SHF_ALLOC | SHF_EXECINSTR;
    ctx.outputs[0].virtual_addr = 0x401000;
    ctx.outputs[0].file_size = 0x100;
    ctx.outputs[0].mem_size = 0x100;
    ctx.outputs[0].data = malloc(0x100);
    // .rodata section
    strncpy(ctx.outputs[1].name, ".rodata", 63);
    ctx.outputs[1].flags = SHF_ALLOC;
    ctx.outputs[1].virtual_addr = 0x401100;
    ctx.outputs[1].file_size = 0x50;
    ctx.outputs[1].mem_size = 0x50;
    ctx.outputs[1].data = malloc(0x50);
    OutputSegment seg;
    ExecError err = build_text_segment(&ctx, &seg);
    ASSERT_EQ(err, EXEC_OK);
    ASSERT_TRUE(seg.is_valid);
    ASSERT_EQ(seg.p_type, PT_LOAD);
    ASSERT_EQ(seg.p_flags, PF_R | PF_X);
    ASSERT_EQ(seg.p_vaddr, 0x401000);
    ASSERT_EQ(seg.p_memsz, 0x150);  // Total span
    // Cleanup
    free(ctx.outputs[0].data);
    free(ctx.outputs[1].data);
    free(ctx.outputs);
}
void test_data_segment_with_bss(void) {
    LinkerContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.outputs = calloc(2, sizeof(OutputSection));
    ctx.output_count = 2;
    // .data section
    strncpy(ctx.outputs[0].name, ".data", 63);
    ctx.outputs[0].flags = SHF_ALLOC | SHF_WRITE;
    ctx.outputs[0].virtual_addr = 0x403000;
    ctx.outputs[0].file_size = 0x100;
    ctx.outputs[0].mem_size = 0x100;
    ctx.outputs[0].data = malloc(0x100);
    // .bss section
    strncpy(ctx.outputs[1].name, ".bss", 63);
    ctx.outputs[1].flags = SHF_ALLOC | SHF_WRITE;
    ctx.outputs[1].virtual_addr = 0x403100;
    ctx.outputs[1].file_size = 0;  // No file content
    ctx.outputs[1].mem_size = 0x100;
    ctx.outputs[1].data = NULL;
    OutputSegment seg;
    ExecError err = build_data_segment(&ctx, &seg);
    ASSERT_EQ(err, EXEC_OK);
    ASSERT_TRUE(seg.is_valid);
    ASSERT_EQ(seg.p_type, PT_LOAD);
    ASSERT_EQ(seg.p_flags, PF_R | PF_W);
    ASSERT_EQ(seg.p_vaddr, 0x403000);
    ASSERT_EQ(seg.p_filesz, 0x100);  // Only .data
    ASSERT_EQ(seg.p_memsz, 0x200);   // .data + .bss
    free(ctx.outputs[0].data);
    free(ctx.outputs);
}
void test_segment_offset_alignment(void) {
    OutputSegment seg;
    segment_init(&seg);
    seg.p_vaddr = 0x401000;
    seg.p_align = 0x1000;
    uint64_t offset;
    ExecError err = calculate_segment_offset(&seg, 0x00D0, &offset);
    ASSERT_EQ(err, EXEC_OK);
    ASSERT_EQ(offset, 0x1000);  // Aligned to page boundary
    // Verify alignment constraint
    ASSERT_EQ(offset % seg.p_align, seg.p_vaddr % seg.p_align);
}
void test_segment_overlap_detection(void) {
    OutputSegment segments[2];
    // First segment: 0x401000 - 0x402000
    segment_init(&segments[0]);
    segments[0].p_type = PT_LOAD;
    segments[0].p_vaddr = 0x401000;
    segments[0].p_memsz = 0x1000;
    segments[0].is_valid = 1;
    // Second segment: 0x401800 - 0x402800 (overlaps!)
    segment_init(&segments[1]);
    segments[1].p_type = PT_LOAD;
    segments[1].p_vaddr = 0x401800;
    segments[1].p_memsz = 0x1000;
    segments[1].is_valid = 1;
    ASSERT_TRUE(segments_overlap(segments, 2));
    // Fix: non-overlapping
    segments[1].p_vaddr = 0x402000;
    ASSERT_FALSE(segments_overlap(segments, 2));
}
```
### Test Suite: Full Executable Generation
```c
// tests/test_exec_full.c
void test_minimal_executable(void) {
    // Create minimal assembly file
    system("as -o /tmp/minimal.o << 'EOF'\n"
           ".globl _start\n"
           "_start:\n"
           "    mov $42, %rdi\n"
           "    mov $60, %rax\n"
           "    syscall\n"
           "EOF");
    // Full linker pipeline
    LinkerContext ctx;
    linker_context_init(&ctx);
    ObjectFile obj;
    parse_object_file("/tmp/minimal.o", &obj);
    parse_symbols(&obj);
    parse_relocations(&obj);
    linker_add_input(&ctx, &obj);
    linker_merge_sections(&ctx);
    linker_assign_addresses(&ctx);
    collect_all_symbols(&ctx);
    check_undefined_symbols(&ctx);
    assign_symbol_addresses(&ctx);
    build_output_buffer(&ctx);
    process_all_relocations(&ctx);
    // Write executable
    ExecStats stats;
    ExecError err = write_executable(&ctx, "/tmp/minimal", &stats);
    ASSERT_EQ(err, EXEC_OK);
    ASSERT_EQ(stats.entry_address > 0, 1);
    // Verify file exists and is executable
    ASSERT_EQ(access("/tmp/minimal", X_OK), 0);
    // Run the executable
    int status = system("/tmp/minimal");
    int exit_code = WEXITSTATUS(status);
    ASSERT_EQ(exit_code, 42);
    linker_context_destroy(&ctx);
}
void test_executable_with_data(void) {
    system("as -o /tmp/data.o << 'EOF'\n"
           ".data\n"
           ".globl value\n"
           "value:\n"
           "    .quad 123\n"
           ".text\n"
           ".globl _start\n"
           "_start:\n"
           "    mov value(%rip), %rdi\n"
           "    mov $60, %rax\n"
           "    syscall\n"
           "EOF");
    LinkerContext ctx;
    linker_context_init(&ctx);
    ObjectFile obj;
    parse_object_file("/tmp/data.o", &obj);
    parse_symbols(&obj);
    parse_relocations(&obj);
    linker_add_input(&ctx, &obj);
    linker_merge_sections(&ctx);
    linker_assign_addresses(&ctx);
    collect_all_symbols(&ctx);
    assign_symbol_addresses(&ctx);
    build_output_buffer(&ctx);
    process_all_relocations(&ctx);
    ExecError err = write_executable(&ctx, "/tmp/data_exec", NULL);
    ASSERT_EQ(err, EXEC_OK);
    int status = system("/tmp/data_exec");
    ASSERT_EQ(WEXITSTATUS(status), 123);
    linker_context_destroy(&ctx);
}
void test_main_fallback(void) {
    // Program with main but no _start
    system("as -o /tmp/main_only.o << 'EOF'\n"
           ".globl main\n"
           "main:\n"
           "    mov $99, %rdi\n"
           "    mov $60, %rax\n"
           "    syscall\n"
           "EOF");
    LinkerContext ctx;
    linker_context_init(&ctx);
    ObjectFile obj;
    parse_object_file("/tmp/main_only.o", &obj);
    parse_symbols(&obj);
    linker_add_input(&ctx, &obj);
    linker_merge_sections(&ctx);
    linker_assign_addresses(&ctx);
    collect_all_symbols(&ctx);
    assign_symbol_addresses(&ctx);
    build_output_buffer(&ctx);
    ExecStats stats;
    ExecError err = write_executable(&ctx, "/tmp/main_exec", &stats);
    ASSERT_EQ(err, EXEC_OK);
    ASSERT_STREQ(stats.entry_symbol, "main");
    linker_context_destroy(&ctx);
}
void test_no_entry_point_error(void) {
    // Program with no entry point
    system("as -o /tmp/no_entry.o << 'EOF'\n"
           ".globl helper\n"
           "helper:\n"
           "    ret\n"
           "EOF");
    LinkerContext ctx;
    linker_context_init(&ctx);
    ObjectFile obj;
    parse_object_file("/tmp/no_entry.o", &obj);
    parse_symbols(&obj);
    linker_add_input(&ctx, &obj);
    linker_merge_sections(&ctx);
    linker_assign_addresses(&ctx);
    collect_all_symbols(&ctx);
    assign_symbol_addresses(&ctx);
    build_output_buffer(&ctx);
    ExecError err = write_executable(&ctx, "/tmp/no_entry_exec", NULL);
    ASSERT_EQ(err, EXEC_ERR_ENTRY_NOT_FOUND);
    linker_context_destroy(&ctx);
}
void test_verify_with_readelf(void) {
    // Generate executable and verify with readelf
    system("as -o /tmp/verify.o << 'EOF'\n"
           ".globl _start\n"
           "_start:\n"
           "    xor %rdi, %rdi\n"
           "    mov $60, %rax\n"
           "    syscall\n"
           "EOF");
    LinkerContext ctx;
    linker_context_init(&ctx);
    ObjectFile obj;
    parse_object_file("/tmp/verify.o", &obj);
    parse_symbols(&obj);
    linker_add_input(&ctx, &obj);
    linker_merge_sections(&ctx);
    linker_assign_addresses(&ctx);
    collect_all_symbols(&ctx);
    assign_symbol_addresses(&ctx);
    build_output_buffer(&ctx);
    process_all_relocations(&ctx);
    write_executable(&ctx, "/tmp/verify_exec", NULL);
    // Verify with file command
    FILE *fp = popen("file /tmp/verify_exec", "r");
    char output[256];
    fgets(output, sizeof(output), fp);
    pclose(fp);
    ASSERT_TRUE(strstr(output, "ELF 64-bit") != NULL);
    ASSERT_TRUE(strstr(output, "executable") != NULL);
    // Verify with readelf
    fp = popen("readelf -h /tmp/verify_exec 2>/dev/null | grep Entry", "r");
    fgets(output, sizeof(output), fp);
    pclose(fp);
    ASSERT_TRUE(strstr(output, "Entry point address:") != NULL);
    linker_context_destroy(&ctx);
}
```

![Executable File Write Sequence](./diagrams/tdd-diag-049.svg)

![p_memsz vs p_filesz for .bss](./diagrams/tdd-diag-047.svg)

## Performance Targets
| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| ELF header construction | < 1μs | `gettimeofday()` around `elf_header_init()` |
| Entry point lookup | < 10μs | Hash table lookup timing |
| Segment construction | < 100μs | `build_all_segments()` timing |
| Layout calculation | < 100μs | `calculate_executable_layout()` timing |
| Buffer assembly | < 1ms for 1MB | `build_executable_buffer()` timing |
| File writing | Disk I/O limited | `fwrite()` timing |
| Complete generation | < 10ms for 1MB executable | End-to-end timing |
| Memory overhead | < 1KB beyond output buffer | RSS delta |
### Memory Budget
```
ELF header:                    64 bytes
Program headers (max 16):      16 * 56 = 896 bytes
OutputSegment array:           16 * 88 = ~1.4 KB
ExecutableLayout:              ~100 bytes
Temporary buffers:             ~1 KB
Total fixed overhead:          ~3.5 KB
Per-section tracking:          ~100 bytes each
Per-segment content:           Uses existing output buffer (no copy until write)
Total for typical executable:  ~5 KB overhead + output buffer
```
## Integration Notes
### Dependencies
- **Milestone 1**: Requires `LinkerContext`, `OutputSection`, section flags and virtual addresses
- **Milestone 2**: Requires `GlobalSymbolTable`, `GlobalSymbol`, resolved symbol addresses
- **Milestone 3**: Requires `output_buffer` with patched section data
- **Standard library**: `<stdint.h>`, `<stdio.h>`, `<stdlib.h>`, `<string.h>`, `<sys/stat.h>`
- **System calls**: `fopen()`, `fwrite()`, `fclose()`, `chmod()`
### API Usage Example
```c
// Complete linker pipeline with executable generation
int main(int argc, char **argv) {
    LinkerContext ctx;
    linker_context_init(&ctx);
    // Milestone 1: Parse and merge
    for (int i = 1; i < argc - 1; i++) {
        ObjectFile obj;
        parse_object_file(argv[i], &obj);
        parse_symbols(&obj);
        parse_relocations(&obj);
        linker_add_input(&ctx, &obj);
    }
    linker_merge_sections(&ctx);
    linker_assign_addresses(&ctx);
    // Milestone 2: Symbol resolution
    collect_all_symbols(&ctx);
    check_undefined_symbols(&ctx);
    assign_symbol_addresses(&ctx);
    // Milestone 3: Relocation processing
    build_output_buffer(&ctx);
    process_all_relocations(&ctx);
    // Milestone 4: Executable generation
    ExecStats stats;
    ExecError err = write_executable(&ctx, argv[argc - 1], &stats);
    if (err != EXEC_OK) {
        fprintf(stderr, "Failed: %s\n", exec_error_string(err));
        return 1;
    }
    printf("Generated: %s (%zu bytes)\n", argv[argc - 1], stats.total_file_size);
    printf("Entry: %s @ 0x%lx\n", stats.entry_symbol, stats.entry_address);
    linker_context_destroy(&ctx);
    return 0;
}
```
### Thread Safety
The executable generation module is single-threaded by design. All operations are sequential:
1. Layout calculation depends on complete section data
2. Buffer assembly requires stable segment info
3. File writing is inherently sequential
No thread safety considerations are needed.

![Segment Permission Assignment](./diagrams/tdd-diag-048.svg)

[[CRITERIA_JSON: {"module_id": "build-linker-m4", "criteria": ["Generate valid ELF64 header with magic bytes 0x7F 'E' 'L' 'F', class ELFCLASS64 (2), data ELFDATA2LSB (1), version 1, type ET_EXEC (2), machine EM_X86_64 (62), e_ehsize=64, e_phentsize=56", "Set e_entry field to the virtual address of _start symbol (or main if _start not found), reporting EXEC_ERR_ENTRY_NOT_FOUND error if neither exists", "Set e_phoff to 64 (immediately after ELF header) and e_phnum to the count of valid PT_LOAD segments", "Set e_shoff, e_shnum, e_shstrndx to 0 (no section headers in minimal executable)", "Write program header table at file offset e_phoff (64), with e_phnum entries each of 56 bytes", "Create PT_LOAD segment for text/code sections with p_type=1, p_flags=PF_R|PF_X (5), p_align=0x1000", "Create PT_LOAD segment for data sections with p_type=1, p_flags=PF_R|PF_W (6), p_align=0x1000", "Set p_memsz > p_filesz for data segment containing .bss, where difference represents zero-filled uninitialized data", "Validate and enforce alignment constraint: p_offset % p_align == p_vaddr % p_align for each PT_LOAD segment", "Calculate p_offset for each segment ensuring page-aligned file offsets (multiples of 0x1000)", "Detect and report segment overlap error if two PT_LOAD segments have overlapping virtual address ranges", "Write segment content at file offsets matching p_offset values in program headers", "Apply executable file permissions (chmod 0755) to output file, continuing with warning if chmod fails", "Produce executable that runs on Linux x86-64 without external tools, verifiable by executing ./output", "Handle missing .bss section gracefully (single text segment executable)", "Report entry point symbol name and address during linking for verification", "Support statistics output including total file size, text segment size, data segment size, and bss size"]}]
<!-- END_TDD_MOD -->


# Project Structure: Static Linker
## Directory Tree
```
linker/
├── 01_elf_types.h              # ELF64 structure definitions (M1: ELF header, section header)
├── 02_elf_parser.h             # Parser interface declarations (M1)
├── 03_elf_parser.c             # ELF header and section parsing implementation (M1)
├── 04_section_types.h          # Internal section representation types (M1)
├── 05_section_merger.h         # Merger interface declarations (M1)
├── 06_section_merger.c         # Section grouping and merging implementation (M1)
├── 07_mapping_table.h          # Input-to-output mapping data structures (M1)
├── 08_mapping_table.c          # Mapping table construction and lookup (M1)
├── 09_linker_context.h         # Global linker state (M1)
├── 10_section_merge_main.c     # Section merge test driver (M1)
├── 11_symbol_types.h           # Symbol data structures and constants (M2)
├── 12_symbol_parser.h          # Symbol table parsing interface (M2)
├── 13_symbol_parser.c          # .symtab/.strtab parsing implementation (M2)
├── 14_global_symbols.h         # Global symbol table interface (M2)
├── 15_global_symbols.c         # Hash table and resolution logic (M2)
├── 16_symbol_resolver.h        # Resolution rules interface (M2)
├── 17_symbol_resolver.c        # Strong/weak/COMMON resolution (M2)
├── 18_address_assign.h         # Final address assignment interface (M2)
├── 19_address_assign.c         # Virtual address computation (M2)
├── 20_symbol_main.c            # Symbol resolution test driver (M2)
├── 21_relocation_types.h       # Relocation constants and data structures (M3)
├── 22_relocation_parser.h      # Relocation section parsing interface (M3)
├── 23_relocation_parser.c      # .rela.* section parsing implementation (M3)
├── 24_relocation_resolver.h    # Symbol resolution for relocations (M3)
├── 25_relocation_resolver.c    # Target symbol lookup implementation (M3)
├── 26_patch_calculator.h       # Patch value computation interface (M3)
├── 27_patch_calculator.c       # R_X86_64_64 and R_X86_64_PC32 formulas (M3)
├── 28_patch_writer.h           # Output buffer patching interface (M3)
├── 29_patch_writer.c           # Little-endian write with bounds check (M3)
├── 30_relocation_processor.h   # Main processing orchestration (M3)
├── 31_relocation_processor.c   # Full relocation pipeline (M3)
├── 32_relocation_main.c        # Relocation test driver (M3)
├── 33_elf_exec_types.h         # ELF executable structures and constants (M4)
├── 34_elf_header.h             # ELF header construction interface (M4)
├── 35_elf_header.c             # ELF header builder implementation (M4)
├── 36_segment_builder.h        # Segment construction interface (M4)
├── 37_segment_builder.c        # PT_LOAD segment builder implementation (M4)
├── 38_program_header.h         # Program header writer interface (M4)
├── 39_program_header.c         # Program header serialization (M4)
├── 40_entry_point.h            # Entry point resolution interface (M4)
├── 41_entry_point.c            # _start/main lookup implementation (M4)
├── 42_exec_writer.h            # Executable file writer interface (M4)
├── 43_exec_writer.c            # Complete ELF file assembly (M4)
├── 44_exec_main.c              # Executable generation test driver (M4)
├── main.c                      # Main linker entry point
├── Makefile                    # Build system
└── tests/
    ├── test_parser.c           # ELF parsing unit tests (M1)
    ├── test_merger.c           # Section merging unit tests (M1)
    ├── test_symbol_parse.c     # Symbol table parsing tests (M2)
    ├── test_resolution.c       # Resolution rules tests (M2)
    ├── test_reloc_parse.c      # Relocation entry parsing tests (M3)
    ├── test_reloc_calc.c       # Patch calculation tests (M3)
    ├── test_reloc_write.c      # Buffer patching tests (M3)
    ├── test_reloc_full.c       # End-to-end relocation tests (M3)
    ├── test_elf_header.c       # ELF header generation tests (M4)
    ├── test_segments.c         # Segment layout tests (M4)
    ├── test_entry.c            # Entry point resolution tests (M4)
    ├── test_exec_full.c        # End-to-end executable tests (M4)
    └── fixtures/
        ├── simple.o            # Single .text section (M1)
        ├── multi.o             # .text, .data, .bss (M1)
        ├── aligned.o           # Non-trivial alignment requirements (M1)
        ├── strong.o            # Strong global symbol (M2)
        ├── weak.o              # Weak symbol definitions (M2)
        ├── common.o            # COMMON symbols (M2)
        ├── conflict.o          # Duplicate strong (error case) (M2)
        ├── call.o              # PC-relative call relocation (M3)
        ├── data_ref.o          # Absolute data reference (M3)
        ├── overflow.o          # Relocation overflow case (M3)
        ├── minimal.s           # Minimal _start-only program (M4)
        ├── with_main.s         # Program with main (no _start) (M4)
        ├── multi_segment.s     # Text + data + bss (M4)
        └── no_entry.s          # Missing entry point (error case) (M4)
```
## Creation Order
1. **Project Setup** (15 min)
   - Create `linker/` directory
   - Create `tests/` and `tests/fixtures/` directories
   - Create `Makefile` with basic compilation rules
2. **Milestone 1: Section Merging** (8-10 hours)
   - `01_elf_types.h` — ELF64 structure definitions
   - `02_elf_parser.h`, `03_elf_parser.c` — ELF parsing
   - `04_section_types.h` — Internal types
   - `05_section_merger.h`, `06_section_merger.c` — Section merging
   - `07_mapping_table.h`, `08_mapping_table.c` — Mapping table
   - `09_linker_context.h` — Global state
   - `10_section_merge_main.c` — Test driver
   - `tests/test_parser.c`, `tests/test_merger.c` — Unit tests
3. **Milestone 2: Symbol Resolution** (8-10 hours)
   - `11_symbol_types.h` — Symbol structures
   - `12_symbol_parser.h`, `13_symbol_parser.c` — Symbol parsing
   - `14_global_symbols.h`, `15_global_symbols.c` — Global symbol table
   - `16_symbol_resolver.h`, `17_symbol_resolver.c` — Resolution rules
   - `18_address_assign.h`, `19_address_assign.c` — Address assignment
   - `20_symbol_main.c` — Test driver
   - `tests/test_symbol_parse.c`, `tests/test_resolution.c` — Unit tests
4. **Milestone 3: Relocation Processing** (8-10 hours)
   - `21_relocation_types.h` — Relocation structures
   - `22_relocation_parser.h`, `23_relocation_parser.c` — Relocation parsing
   - `24_relocation_resolver.h`, `25_relocation_resolver.c` — Symbol resolution
   - `26_patch_calculator.h`, `27_patch_calculator.c` — Patch computation
   - `28_patch_writer.h`, `29_patch_writer.c` — Buffer patching
   - `30_relocation_processor.h`, `31_relocation_processor.c` — Pipeline
   - `32_relocation_main.c` — Test driver
   - `tests/test_reloc_*.c` — Unit tests
5. **Milestone 4: Executable Generation** (6-8 hours)
   - `33_elf_exec_types.h` — Executable structures
   - `34_elf_header.h`, `35_elf_header.c` — ELF header
   - `36_segment_builder.h`, `37_segment_builder.c` — Segment building
   - `38_program_header.h`, `39_program_header.c` — Program headers
   - `40_entry_point.h`, `41_entry_point.c` — Entry point resolution
   - `42_exec_writer.h`, `43_exec_writer.c` — File writing
   - `44_exec_main.c` — Test driver
   - `tests/test_elf_header.c`, `tests/test_segments.c`, etc.
6. **Integration** (2-3 hours)
   - `main.c` — Main entry point linking all modules
   - Complete `Makefile` with all targets
   - End-to-end testing
## File Count Summary
- Total source files: 44
- Header files: 22
- Implementation files: 22
- Test files: 12
- Test fixtures: 12
- Build files: 1 (Makefile)
- **Estimated lines of code: ~8,000-10,000**
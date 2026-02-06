# Static Linker: Design Document


## Overview

This document designs a static linker that merges multiple compiled object files into a single executable ELF binary. The key architectural challenge is resolving symbolic references across separate compilation units while maintaining proper memory layout, requiring careful section concatenation, global symbol table construction, and address relocation patching.


> This guide is meant to help you understand the big picture before diving into each milestone. Refer back to it whenever you need context on how components connect.


## Context and Problem Statement

> **Milestone(s):** This section provides foundational understanding for all milestones.

Software compilation is typically a multi-stage process: source code is transformed into machine code, but this machine code is not yet a runnable program. The critical final step that bridges this gap is **linking**. This section explores why linking exists, the fundamental problems it solves, and the architectural challenges involved in merging separate compilation units into a cohesive executable.

### The Assembly Puzzle Analogy

Imagine assembling a complex jigsaw puzzle where each piece represents a single compiled object file (`.o` file). Each puzzle piece has two types of connectors:

1. **Protruding Connectors (Symbol Definitions):** Functions and variables *defined* within that object file. For example, a function `calculate_total` implemented in `math.o`.
2. **Recessed Connectors (Symbol References):** Calls to functions or uses of variables that are *declared* but not defined in that object file. For example, a call to `printf` in `main.o`.

The linking process is analogous to assembling this puzzle by matching every recessed connector (reference) with a corresponding protruding connector (definition) from another piece. If a recessed connector has no matching protruding connector anywhere in the puzzle set, the puzzle is incomplete and cannot be assembled—this is an **undefined symbol error**.

However, linking involves more than just matching shapes. The assembler must also:
- **Arrange the pieces in a specific layout** (section merging with alignment)
- **Label each connector with its final position** (address resolution via relocations)
- **Create instructions for the final display** (executable generation with program headers)

This analogy highlights the core linking problem: independently compiled modules are designed to connect, but they lack the contextual information about where other modules will be placed in memory. The linker provides this global view and performs the necessary adjustments to make all connections work correctly.

### Core Technical Challenges

Transforming multiple object files into a single executable involves three interdependent technical challenges that must be solved simultaneously.

#### 1. Section Merging with Alignment Constraints

Object files contain code and data organized into **sections** (`.text` for executable code, `.data` for initialized global variables, `.rodata` for read-only data, `.bss` for uninitialized data). The linker must combine corresponding sections from all input files into contiguous blocks in the output executable.

**Primary Challenge:** Sections have **alignment requirements** (e.g., 16-byte boundaries for `.text` to optimize CPU cache lines). When concatenating sections from different files, the linker must insert padding between them to satisfy these requirements. Furthermore, `.bss` sections contain no actual data in the object file—they only specify how much zero-initialized memory to allocate at runtime—which requires special handling during layout.

**Consequences of Failure:** Incorrect alignment causes CPU misalignment faults or severe performance degradation. Mishandling `.bss` results in either wasted space or overlapping data.

#### 2. Cross-File Symbol Resolution

Each object file defines some symbols (functions, global variables) and references others. The linker must build a **global symbol table** across all input files to:
- Match each reference with its corresponding definition
- Detect undefined symbols (references without definitions)
- Handle multiple definitions according to **strong/weak symbol rules** (strong definitions override weak ones)
- Resolve **COMMON symbols** (tentative definitions of uninitialized globals) by allocating space for the largest size encountered

**Primary Challenge:** Symbol resolution must respect **linking scope rules**: local symbols (static functions/variables) are visible only within their defining object file, while global symbols are visible across all files. The linker must distinguish between these visibility classes and handle the complex interplay of strong, weak, and COMMON symbol semantics.

**Consequences of Failure:** Undetected undefined symbols cause runtime crashes. Incorrect strong/weak resolution leads to wrong function implementations being used. COMMON symbol mishandling causes memory corruption.

#### 3. Address Relocation Patching

Object files contain **relocation entries** that mark places in the code/data where addresses need to be filled in once the final memory layout is known. For example:
- A function call to an external function needs the target address
- A reference to a global variable needs its absolute address
- PC-relative addressing (common on x86-64) needs the offset between the instruction and the target

**Primary Challenge:** Relocations require precise calculations that depend on:
- The final address of the target symbol (from the global symbol table)
- The address of the relocation site itself (where the patch is applied)
- The **addend** value stored in the relocation entry (an offset to add to the symbol value)
- The relocation type (e.g., `R_X86_64_PC32` for 32-bit PC-relative, `R_X86_64_64` for 64-bit absolute)

**Consequences of Failure:** Incorrect relocation calculations produce executables that crash or compute wrong addresses, leading to segmentation faults or data corruption.

### Existing Approaches Comparison

The linking problem has been addressed through various architectural approaches, each with different trade-offs in complexity, performance, and flexibility.

> **Decision: Static Linking Approach**
> - **Context**: We need to choose a linking strategy suitable for educational implementation while covering fundamental linking concepts.
> - **Options Considered**: 
>   1. **Traditional Static Linking** (as used by `ld`, `gold`): All code and data from object files and libraries are combined into a single executable at build time.
>   2. **Dynamic Linking**: References to shared libraries are resolved at load time or runtime via indirection tables (PLT/GOT).
>   3. **Incremental/Lazy Linking**: Only necessary parts are linked initially, with additional linking occurring on-demand.
> - **Decision**: Traditional static linking.
> - **Rationale**: Static linking provides the clearest pedagogical path to understanding core linking concepts (section merging, symbol resolution, relocation) without the complexity of runtime indirection, position-independent code, or shared library management. It produces self-contained executables that are easier to debug and verify.
> - **Consequences**: The resulting linker will produce larger executables (no code sharing between processes) and cannot link against shared libraries without implementing additional complexity. However, it provides a solid foundation for understanding all linking approaches.

| Approach | Pros | Cons | Educational Value |
|----------|------|------|-------------------|
| **Traditional Static Linking** (Our choice) | • Complete control over final binary<br>• No runtime dependencies<br>• Simpler implementation<br>• Clear separation of linking phases | • Larger executable size<br>• Cannot update libraries without re-linking<br>• Limited code sharing between processes | **Highest** - directly exposes section merging, symbol resolution, and relocation concepts |
| **Dynamic Linking** | • Smaller executables<br>• Library updates without re-linking<br>• Memory sharing between processes | • Complex implementation (PLT/GOT, interposition)<br>• Runtime overhead<br>• Dependency management issues | Medium - requires understanding indirection and runtime resolution |
| **Modern Linkers (lld, mold)** | • Extremely fast<br>• Advanced optimizations (LTO, ICF)<br>• Better diagnostics | • Complex algorithms<br>• Heavy dependency on compiler toolchain<br>• Many advanced features | Low - optimization complexity obscures core concepts |

**Traditional Static Linker Architecture:**
Traditional linkers like `ld` (the GNU linker) and `gold` (a faster alternative) follow a similar multi-pass architecture:
1. **Read and parse** all input object files and libraries
2. **Merge sections** by type, applying alignment constraints
3. **Resolve symbols** across all inputs, handling strong/weak rules
4. **Apply relocations** using resolved symbol addresses
5. **Generate executable** with proper headers and segment layout

This architecture maps directly to our milestone structure and provides a clear separation of concerns that we can implement incrementally.

**Comparison with Our Design:**
While we follow the traditional architecture for clarity, we make several simplifications for educational purposes:

| Feature | Traditional Linkers (ld/gold) | Our Linker |
|---------|-----------------------------|------------|
| Input formats | ELF, archive (.a), shared objects (.so) | ELF object files only |
| Relocation types | Hundreds (all architectures) | 2 x86-64 types (PC32, 64) |
| Optimization | Dead code elimination, ICF, LTO | None |
| Output format | Executable, shared object, relocatable | Executable only |
| Library support | Archive and shared libraries | Manual object file listing |

These simplifications allow us to focus on the essential linking algorithms while still producing working executables that run on modern Linux systems.

> **Key Insight**: All linking approaches share the same fundamental concepts—section layout, symbol resolution, and address relocation. Mastering static linking provides the conceptual foundation needed to understand any linking system, including dynamic linkers and advanced optimizers.

**Common Pitfalls in Understanding Linking:**

⚠️ **Pitfall: Confusing Compilation and Linking**
- **Description**: Thinking that object files contain fully resolved addresses or that the compiler handles cross-file references.
- **Why It's Wrong**: Compilers work on one translation unit at a time and leave placeholders (relocations) for addresses that can only be determined when all object files are combined.
- **How to Avoid**: Remember that object files are **relocatable**—they contain code that can be placed at any memory address, with relocations indicating where addresses need to be filled in later.

⚠️ **Pitfall: Misunderstanding Symbol Visibility**
- **Description**: Assuming all symbols follow the same resolution rules or that static (local) symbols can be referenced across files.
- **Why It's Wrong**: C's `static` keyword makes symbols visible only within their translation unit. The linker must enforce this by not resolving cross-file references to static symbols.
- **How to Avoid**: Carefully track symbol binding (`STB_LOCAL` vs `STB_GLOBAL`) in the symbol table and only resolve references to global symbols.

⚠️ **Pitfall: Ignoring Alignment Requirements**
- **Description**: Simply concatenating section data without inserting padding between sections from different files.
- **Why It's Wrong**: CPUs and memory subsystems require certain data alignments for performance and correctness. Misaligned accesses can cause crashes or severe slowdowns.
- **How to Avoid**: Always compute the next aligned address when transitioning between sections: `aligned_addr = (current_addr + alignment - 1) & ~(alignment - 1)`.

The following diagram illustrates the high-level component architecture we'll implement to solve these challenges:

![Static Linker Component Diagram](./diagrams/component-diagram.svg)

Each component addresses specific aspects of the linking problem, working together through a well-defined data flow to transform multiple object files into a single executable. The subsequent sections of this document will delve into each component's design, starting with the data structures that form the foundation of our implementation.


## Goals and Non-Goals

> **Milestone(s):** This section establishes the foundational scope for all milestones, defining what the static linker must achieve and what boundaries exist in its implementation.

### Goals

Think of our static linker as a **specialized assembly line in a factory**. Raw components (object files) arrive from different production lines (compilers), each containing partially assembled machine code with missing connections. Our assembly line must: 1) sort components into bins by type (section merging), 2) create a master wiring diagram showing how all components connect (symbol resolution), 3) solder the actual connections between components (relocation processing), and 4) package the final product in a shipping container that the operating system's loading dock can efficiently unpack (executable generation). Each station in this assembly line corresponds to a core capability the linker must possess.

The following table defines the essential capabilities our static linker must implement, serving as the acceptance criteria for the complete project:

| Goal | Description | Success Criteria | Key Components Involved |
|------|-------------|------------------|--------------------------|
| **Multi-file ELF Parsing** | Read and interpret multiple ELF object files (`.o` files) produced by compilers like GCC or Clang. Each file contains compiled code, data, symbol definitions, and relocation instructions. | Successfully load all sections, symbols, and relocation entries from multiple input files without crashing on valid ELF input. | ELF Reader |
| **Section Merging** | Combine corresponding sections (`.text`, `.data`, `.rodata`, `.bss`) from different object files into contiguous blocks in the output executable. Must respect alignment requirements and track the mapping from input sections to final output locations. | Produce a single, logically concatenated output for each section type with correct padding, and maintain accurate input-to-output offset mapping for relocation processing. | Section Merger |
| **Symbol Resolution** | Build a global symbol table across all input files, resolving references to external symbols. Must handle strong vs. weak symbols, detect undefined symbols, and scope local symbols appropriately. | Report clear errors for undefined symbols, correctly resolve duplicate symbols according to ELF rules, and produce a final symbol table with accurate virtual addresses. | Symbol Resolver |
| **Relocation Processing** | Apply relocation entries to patch addresses in the merged sections. Must calculate correct absolute and relative addresses based on final symbol locations and apply them to the appropriate positions in section data. | Successfully patch relocation sites in `.text` and `.data` sections so that references to symbols point to their correct final addresses, enabling the code to execute correctly. | Relocation Applier |
| **Executable Generation** | Produce a valid ELF executable file with proper headers and memory layout. Must create program headers that tell the OS loader how to map the file into memory, set the correct entry point, and separate code and data into distinct memory segments with appropriate permissions. | Generate an executable that runs correctly on Linux (x86_64) when invoked, passing simple test cases without requiring external fixup tools. | Executable Writer |

> **Design Insight:** The sequence of goals follows the natural dependency chain of linking: you cannot resolve symbols until you know what sections exist, cannot apply relocations until symbols are resolved, and cannot generate a valid executable until all sections are merged and relocations applied. This dependency graph dictates the component architecture.

**Key Architectural Decisions Supporting These Goals:**

> **Decision: Educational-First Implementation Over Production Robustness**
> - **Context**: This project exists primarily for learning linker internals, not for replacing production linkers like `ld`.
> - **Options Considered**:
>     1. **Implement full ELF spec with all edge cases**: Would be exhaustive but overwhelming for learners.
>     2. **Implement minimal subset needed for educational test cases**: Focuses on core concepts while leaving out complex but less-essential features.
> - **Decision**: Implement the minimal subset of ELF features required to link simple C programs, with clear error messages for unsupported cases.
> - **Rationale**: The cognitive load of understanding linking concepts is high; adding support for every ELF feature would distract from core educational objectives. A minimal implementation that works for deliberately simple test programs allows learners to focus on the architectural flow.
> - **Consequences**: The linker will fail on complex real-world object files (e.g., those with TLS, complex relocations, or unusual sections), but will succeed on the test programs provided in the project materials. This trade-off is acceptable for the learning context.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Full ELF spec implementation | Can link real-world programs; comprehensive learning | Massive scope; high complexity; long development time | ❌ |
| Minimal educational subset | Focuses on core concepts; manageable implementation; clear learning path | Cannot link complex programs; limited practical utility | ✅ |

### Non-Goals

Imagine our assembly line is designed specifically for **basic mechanical watches**—it excels at assembling gears, springs, and hands into a working timepiece. It is explicitly **not** designed to assemble digital smartwatches with circuit boards, nor is it a universal factory that can assemble any product. These boundaries are crucial for managing scope and ensuring we build something that teaches the intended concepts without becoming unmanageably complex.

The following table defines explicit boundaries—features and capabilities that this static linker **will not implement**, along with justification for their exclusion:

| Non-Goal | What It Means | Why Excluded (Scope/Rationale) | Alternative for Learners |
|----------|---------------|--------------------------------|--------------------------|
| **Dynamic Linking** | Supporting shared libraries (`.so` files) and runtime symbol resolution via PLT/GOT. | Dynamic linking introduces completely different architectural patterns (lazy binding, interposition, runtime relocations) that would double the project scope. The static linking concepts are foundational; dynamic linking can be a separate advanced topic. | Study `ld.so` and ELF dynamic sections as a follow-on project. |
| **Shared Library Generation** | Creating `.so` files as output (position-independent code, symbol versioning). | Requires handling of PIC-specific relocations and symbol visibility controls, which are distinct from static executable generation. | Focus on producing static executables first; shared libraries represent a different output format with different constraints. |
| **Cross-Architecture Linking** | Supporting object files for architectures other than x86_64 (e.g., ARM, RISC-V). | Each architecture has its own relocation types, calling conventions, and alignment requirements. Supporting multiple architectures would require an abstraction layer that obscures the concrete learning of x86_64 specifics. | The learned concepts are transferable; implementation details differ per architecture. |
| **Compiler Integration** | Acting as a full toolchain that invokes the compiler or processes source code directly. | The linker operates on compiler output (object files); integrating compilation would blur the separation of concerns and introduce massive additional complexity (parsing C, optimization, etc.). | Use existing compilers (gcc, clang) to produce `.o` files as input. |
| **Archive (`*.a`) File Processing** | Reading static library archives and extracting member object files automatically. | While static libraries are collections of `.o` files, archive format parsing adds filesystem complexity without teaching new linking concepts. Learners can manually specify all `.o` files. | List all required `.o` files explicitly on the command line instead of using `-l` flags. |
| **Debug Information Preservation** | Carrying over DWARF or other debug sections from input objects to the output executable. | Debug sections are large and complex; their processing doesn't contribute to core linking concepts (they are essentially treated as passive data). Would increase complexity without educational benefit. | Strip debug sections from input files if present, or ignore them. |
| **Advanced Optimization** | Performing link-time optimization (LTO), garbage collection of unused sections, or code reordering. | These are advanced features of production linkers that require deep analysis and transformation of code, far beyond the basic "combine and resolve" model. | Recognize that production linkers have these capabilities, but they are out of scope for learning fundamentals. |
| **Complex Relocation Types** | Supporting the full range of x86_64 relocations (e.g., RIP-relative 32-bit, GOTPCREL, etc.) beyond the two basic types. | The two relocations chosen (R_X86_64_PC32 and R_X86_64_64) illustrate the core concepts of relative and absolute addressing. Adding more types increases implementation complexity without proportionally increasing educational value. | Implement only the two specified relocation types; others can be reported as unsupported errors. |
| **C++ Feature Support** | Handling name mangling, template instantiation, or exception handling tables. | C++ introduces language-specific complexities that are implemented by the compiler and linker collaboratively. Supporting them would require understanding Itanium ABI details beyond ELF basics. | Test with C programs only; C++ programs would require a C++ compiler and runtime, which are out of scope. |
| **Windows/PE Format Support** | Generating Portable Executable (PE) files for Windows. | This project focuses on the ELF format used by Linux and other Unix-like systems. PE has different structures and semantics; supporting both would require a format-abstracting layer. | The architectural concepts are similar, but implementation details differ—focus on ELF as a concrete learning vehicle. |

> **Design Insight:** By explicitly declaring these non-goals, we create a **protected learning environment**. Learners can focus on understanding the core pipeline without being overwhelmed by the immense complexity of production-grade linkers. Each non-goal represents a potential future extension point once the fundamentals are mastered.

**Architectural Impact of Non-Goals:**

The exclusion of these features significantly simplifies the component designs:

- The **ELF Reader** only needs to parse the minimal set of sections and relocation types, ignoring dynamic sections, debug sections, and architecture-specific extensions.
- The **Symbol Resolver** doesn't need to handle symbol versioning, visibility attributes beyond basic global/local, or complex C++ mangling.
- The **Relocation Applier** implements exactly two relocation types instead of dozens, avoiding the need for a complex dispatch mechanism.
- The **Executable Writer** generates simple two-segment executables without complex segment types (e.g., `PT_DYNAMIC`, `PT_TLS`, `PT_GNU_STACK`).

This focused scope allows each component to be implemented in a straightforward, pedagogical manner while still producing working executables for the intended test cases.

### Implementation Guidance

#### Technology Recommendations Table

| Component | Simple Option | Advanced Option | Rationale for Choice |
|-----------|---------------|-----------------|----------------------|
| **ELF Parsing** | Manual byte-by-byte parsing using `<stdint.h>` types and pointer arithmetic. | Using a library like `libelf` or `elf.h` system headers. | **Manual parsing** is chosen for maximum educational value—learners directly interact with ELF structures, reinforcing format understanding. |
| **Data Structures** | Plain C structs matching ELF specifications, with custom linked lists for collections. | Generic containers from libraries (e.g., GLib). | **Plain structs + custom lists** keep dependencies minimal and expose all implementation details for learning. |
| **File I/O** | Standard C file operations (`fopen`, `fread`, `fwrite`) with binary mode. | Memory-mapped I/O (`mmap`) for performance. | **Standard file I/O** is simpler, more portable, and sufficient for educational scale. |
| **Error Handling** | Simple error codes with descriptive `fprintf(stderr, ...)` messages and program exit. | Structured error types with propagation and recovery. | **Simple exit-on-error** keeps focus on main logic; production linkers would need more sophisticated error recovery. |

#### Recommended File/Module Structure

The goals and non-goals influence how we organize the codebase. Since we're implementing a focused subset of functionality, we can keep the module structure clean and linear:

```
static-linker/
├── include/                    # Public headers (if any)
│   └── linker.h               # Main linker API (optional)
├── src/                       # Core implementation
│   ├── main.c                 # Command-line interface
│   ├── elf_reader.c           # Goal 1: Multi-file ELF parsing
│   ├── section_merger.c       # Goal 2: Section merging
│   ├── symbol_resolver.c      # Goal 3: Symbol resolution  
│   ├── relocation_applier.c   # Goal 4: Relocation processing
│   └── executable_writer.c    # Goal 5: Executable generation
├── lib/                       # Internal utilities
│   ├── elf_definitions.h      # ELF structure definitions
│   ├── common.c               # Shared helpers (alignment, etc.)
│   └── common.h
└── tests/                     # Test programs
    ├── simple_program.c       # Basic test case
    ├── extern_test.c          # Tests external references
    └── expected_output/       # Expected executable outputs
```

#### Infrastructure Starter Code

Since manual ELF parsing is a prerequisite but not the core learning goal of linking logic, provide complete helper code for reading basic ELF structures:

**File: `lib/elf_definitions.h`**
```c
#ifndef ELF_DEFINITIONS_H
#define ELF_DEFINITIONS_H

#include <stdint.h>

// ELF identification constants
#define EI_NIDENT 16
#define ELFMAG0 0x7F
#define ELFMAG1 'E'
#define ELFMAG2 'L'
#define ELFMAG3 'F'
#define ELFCLASS64 2
#define ELFDATA2LSB 1
#define EV_CURRENT 1
#define ET_REL 1
#define EM_X86_64 62

// Section header types
#define SHT_PROGBITS 1
#define SHT_SYMTAB 2
#define SHT_STRTAB 3
#define SHT_RELA 4
#define SHT_NOBITS 8

// Section flags
#define SHF_ALLOC 0x2
#define SHF_EXECINSTR 0x4
#define SHF_WRITE 0x1

// Symbol bindings
#define STB_LOCAL 0
#define STB_GLOBAL 1
#define STB_WEAK 2

// Symbol types  
#define STT_NOTYPE 0
#define STT_OBJECT 1
#define STT_FUNC 2
#define STT_SECTION 3

// Relocation types
#define R_X86_64_64 1
#define R_X86_64_PC32 2

// ELF64 structures (packed to match on-disk layout)
typedef struct {
    unsigned char e_ident[EI_NIDENT];
    uint16_t e_type;
    uint16_t e_machine;
    uint32_t e_version;
    uint64_t e_entry;
    uint64_t e_phoff;
    uint64_t e_shoff;
    uint32_t e_flags;
    uint16_t e_ehsize;
    uint16_t e_phentsize;
    uint16_t e_phnum;
    uint16_t e_shentsize;
    uint16_t e_shnum;
    uint16_t e_shstrndx;
} __attribute__((packed)) Elf64_Ehdr;

typedef struct {
    uint32_t sh_name;
    uint32_t sh_type;
    uint64_t sh_flags;
    uint64_t sh_addr;
    uint64_t sh_offset;
    uint64_t sh_size;
    uint32_t sh_link;
    uint32_t sh_info;
    uint64_t sh_addralign;
    uint64_t sh_entsize;
} __attribute__((packed)) Elf64_Shdr;

typedef struct {
    uint32_t st_name;
    unsigned char st_info;
    unsigned char st_other;
    uint16_t st_shndx;
    uint64_t st_value;
    uint64_t st_size;
} __attribute__((packed)) Elf64_Sym;

typedef struct {
    uint64_t r_offset;
    uint64_t r_info;
    int64_t r_addend;
} __attribute__((packed)) Elf64_Rela;

// Helper macros for relocation entries
#define ELF64_R_SYM(i) ((i) >> 32)
#define ELF64_R_TYPE(i) ((i) & 0xffffffffL)

#endif // ELF_DEFINITIONS_H
```

**File: `lib/common.h`**
```c
#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <stddef.h>

// Alignment helper
static inline uint64_t align_to(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

// Error handling
void fatal_error(const char* fmt, ...);

#endif // COMMON_H
```

**File: `lib/common.c`**
```c
#include "common.h"
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>

void fatal_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "linker error: ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
    exit(1);
}
```

#### Core Logic Skeleton Code

While the full component implementations will come later, here's a skeleton showing how the main linking procedure reflects our goals:

**File: `src/main.c`**
```c
#include <stdio.h>
#include <stdlib.h>
#include "elf_definitions.h"
#include "common.h"

// Forward declarations of component interfaces
struct ObjectFile;
struct MergedSections;
struct SymbolTable;
struct OutputExecutable;

struct ObjectFile* read_elf_file(const char* filename);
struct MergedSections* merge_all_sections(struct ObjectFile** objects, int count);
struct SymbolTable* resolve_all_symbols(struct ObjectFile** objects, int count, 
                                       struct MergedSections* merged);
void apply_all_relocations(struct ObjectFile** objects, int count,
                          struct MergedSections* merged,
                          struct SymbolTable* symbols);
struct OutputExecutable* generate_executable(struct MergedSections* merged,
                                            struct SymbolTable* symbols);
void write_executable(struct OutputExecutable* exec, const char* filename);

int main(int argc, char** argv) {
    if (argc < 3) {
        fatal_error("usage: %s <output> <input1.o> [input2.o ...]", argv[0]);
    }
    
    const char* output_filename = argv[1];
    int input_count = argc - 2;
    
    // TODO 1: Read all input object files (Goal: Multi-file ELF parsing)
    struct ObjectFile** objects = malloc(sizeof(struct ObjectFile*) * input_count);
    for (int i = 0; i < input_count; i++) {
        objects[i] = read_elf_file(argv[i + 2]);
        if (!objects[i]) {
            fatal_error("failed to read file: %s", argv[i + 2]);
        }
    }
    
    // TODO 2: Merge sections from all objects (Goal: Section merging)
    struct MergedSections* merged = merge_all_sections(objects, input_count);
    
    // TODO 3: Resolve symbols across all objects (Goal: Symbol resolution)
    struct SymbolTable* symbols = resolve_all_symbols(objects, input_count, merged);
    
    // TODO 4: Apply relocations using resolved symbols (Goal: Relocation processing)
    apply_all_relocations(objects, input_count, merged, symbols);
    
    // TODO 5: Generate final executable (Goal: Executable generation)
    struct OutputExecutable* exec = generate_executable(merged, symbols);
    
    // TODO 6: Write executable to disk
    write_executable(exec, output_filename);
    
    printf("Successfully linked %d files into %s\n", input_count, output_filename);
    return 0;
}
```

#### Language-Specific Hints (C)

- **Endianness**: x86_64 uses little-endian. Use `le32toh`, `le64toh` (from `<endian.h>`) or manually swap bytes if portability is needed, but for simplicity, assume host is little-endian.
- **Structure Packing**: Use `__attribute__((packed))` for ELF structures to match on-disk layout without padding.
- **File Operations**: Always open files in binary mode (`"rb"`, `"wb"`) to prevent text mode conversions on Windows.
- **Memory Management**: Since the linker is a one-pass tool, simple `malloc/free` is sufficient; no need for complex memory pools.
- **Error Messages**: Print errors to `stderr` with descriptive context (file name, symbol name, section name) to help debugging.

#### Milestone Checkpoint

After implementing all goals, you should be able to link and run a simple test:

1. **Create test program** (`tests/hello.c`):
   ```c
   #include <unistd.h>
   
   const char* message = "Hello, linker!\n";
   
   void _start() {
       write(1, message, 15);
       _exit(0);
   }
   ```

2. **Compile and link**:
   ```bash
   gcc -c -nostdlib -o hello.o tests/hello.c
   ./linker hello hello.o
   chmod +x hello
   ```

3. **Expected behavior**:
   - Linker runs without errors
   - Produces executable `hello`
   - Running `./hello` prints "Hello, linker!" and exits with code 0
   - `readelf -l hello` shows two PT_LOAD segments with correct permissions
   - `objdump -d hello` shows resolved addresses in disassembly

#### Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| "Segmentation fault" when running output | Incorrect relocation or entry point | Use `objdump -d` to check if `_start` jumps to a reasonable address; check relocation calculations | Verify symbol resolution and PC-relative relocation math |
| "File not in ELF format" error | Corrupted or wrong file type | Check first 4 bytes with `hexdump -C file.o | head -1` (should be 7F 45 4C 46) | Ensure input files are actual `.o` files from compiler |
| "Undefined symbol" when symbol exists | Symbol not marked global in source | Use `readelf -s file.o` to check binding (should be STB_GLOBAL, not STB_LOCAL) | Add `extern` or remove `static` in source; recompile |
| Executable runs but produces wrong output | Data relocations incorrect | Check `.data` section contents with `objdump -s -j .data`; verify data symbol addresses | Ensure data symbols get correct absolute relocations |
| "Section flags mismatch" error | Trying to merge incompatible sections | Check section flags with `readelf -S` for each input file | Ensure only sections with identical flags are merged |


## High-Level Architecture

> **Milestone(s):** This section provides the overarching architectural blueprint for all milestones, defining the system components, their responsibilities, and interactions that collectively transform multiple object files into a single executable.

Think of the static linker as a **factory assembly line** that takes individual components (object files) and transforms them into a complete product (executable). Each workstation along the assembly line performs a specific transformation, with components passing along a conveyor belt that gradually assembles the final product. The assembly line must handle precise measurements (address calculations), ensure parts fit together (symbol resolution), and create proper packaging for delivery (executable format).

### Component Overview

The static linker architecture comprises five core components that work sequentially to transform object files into an executable. Each component owns specific responsibilities and data transformations, maintaining clear separation of concerns.

#### ELF Reader: The Librarian
**Mental Model:** Imagine a librarian who receives new books (object files) and creates detailed catalog cards for each. The librarian doesn't modify the books—they merely index their contents (sections, symbols, relocation instructions) so other components can find what they need efficiently.

| Component | Responsibility | Key Inputs | Key Outputs |
|-----------|----------------|------------|-------------|
| **ELF Reader** | Parses ELF object files to extract sections, symbols, and relocations without modifying content | File paths to `.o` files | In-memory representation of each `ObjectFile` with parsed headers, section data, symbol tables, and relocation entries |
| **Section Merger** | Groups similar sections from multiple files into contiguous output sections with proper alignment | Multiple `ObjectFile` instances | `MergedSections` layout mapping input sections to output addresses |
| **Symbol Resolver** | Builds global symbol table, resolving references across files and handling duplicate definitions | Multiple `ObjectFile` instances, `MergedSections` layout | `SymbolTable` with final addresses for all global symbols |
| **Relocation Applier** | Patches addresses in merged section data using relocation entries and resolved symbol addresses | Multiple `ObjectFile` instances, `MergedSections` layout, `SymbolTable` | Modified section data with addresses resolved |
| **Executable Writer** | Generates final ELF executable with proper headers and segment layout for OS loading | `MergedSections` layout, `SymbolTable` | `OutputExecutable` ready for writing to disk |

> **Decision: Pipeline Architecture vs Integrated Processing**  
> - **Context:** We need to process multiple object files through several distinct transformations.  
> - **Options Considered:**  
>   1. **Pipeline Architecture:** Each component processes all files for one transformation before passing to next component  
>   2. **Integrated Processing:** Single component processes each file through all transformations sequentially  
>   3. **Streaming Architecture:** Process files as they arrive with minimal buffering  
> - **Decision:** Pipeline architecture with clear component boundaries  
> - **Rationale:** Pipeline provides natural checkpointing for debugging, matches educational milestones precisely, and separates concerns cleanly. Each component's output becomes the next's input, making data flow explicit and testable.  
> - **Consequences:** Requires full loading of all object files initially (higher memory) but simplifies logic and debugging. Intermediate data structures (`MergedSections`, `SymbolTable`) must be well-defined.

| Architecture Option | Pros | Cons | Why Not Chosen |
|-------------------|------|------|----------------|
| Pipeline Architecture | Clear separation of concerns, matches milestones, easy debugging | Higher memory usage (loads all files), intermediate data structures needed | **CHOSEN** - Best for educational clarity |
| Integrated Processing | Potentially lower memory, single pass through data | Mixes concerns, harder to debug and test | Too complex for learners |
| Streaming Architecture | Minimal memory, processes as data arrives | Complex state management, hard error recovery | Overly complex for educational goals |

### Recommended File Structure

**Mental Model:** Imagine organizing a workshop with separate stations for different tasks—one area for disassembly (reading), another for sorting parts (merging), a third for quality control (resolution), etc. Each station has its own tools and workbenches, logically separated but connected by conveyors.

The project should follow a modular structure that mirrors the pipeline architecture, with each component living in its own module/file. This promotes separation of concerns, simplifies testing, and makes the codebase navigable.

```
static-linker/
├── include/                          # Public header files (if creating a library)
│   └── linker.h
├── src/                              # Main source directory
│   ├── main.c                        # Entry point: coordinates the pipeline
│   ├── elf_reader.c                  # ELF Reader component
│   ├── elf_reader.h                  # ELF Reader interface and structures
│   ├── section_merger.c              # Section Merger component
│   ├── section_merger.h              # Section Merger interface
│   ├── symbol_resolver.c             # Symbol Resolver component
│   ├── symbol_resolver.h             # Symbol Resolver interface
│   ├── relocation_applier.c          # Relocation Applier component
│   ├── relocation_applier.h          # Relocation Applier interface
│   ├── executable_writer.c           # Executable Writer component
│   ├── executable_writer.h           # Executable Writer interface
│   ├── utils.c                       # Shared utilities (alignment, error handling)
│   └── utils.h
├── tests/                            # Test programs and verification
│   ├── test_programs/                # Simple C programs for testing linker
│   │   ├── simple.c                  # Single file test
│   │   ├── extern_test.c             # External reference test
│   │   └── weak_test.c               # Weak symbol test
│   └── run_tests.sh                  # Test runner script
└── Makefile                          # Build configuration
```

**Key Interface Files and Their Responsibilities:**

| Header File | Primary Types | Primary Functions | Purpose |
|-------------|---------------|-------------------|---------|
| `elf_reader.h` | `ObjectFile`, `ElfSection`, `ElfSymbol`, `ElfRelocation` | `read_elf_file()`, `free_object_file()` | Parse ELF files into in-memory structures |
| `section_merger.h` | `MergedSections`, `OutputSection` | `merge_all_sections()`, `get_output_offset()` | Merge sections and track input→output mapping |
| `symbol_resolver.h` | `SymbolTable`, `SymbolEntry` | `resolve_all_symbols()`, `lookup_symbol()` | Build global symbol table with addresses |
| `relocation_applier.h` | (none - operations on existing data) | `apply_all_relocations()` | Patch addresses in merged section data |
| `executable_writer.h` | `OutputExecutable` | `generate_executable()`, `write_executable()` | Create ELF headers and write final binary |
| `utils.h` | (none) | `align_to()`, `fatal_error()`, `read_file()` | Shared utilities for alignment, errors, I/O |

> **Decision: Monolithic vs Modular Header Structure**  
> - **Context:** Components need to share common type definitions (like `Elf64_Sym`) while maintaining clear interfaces.  
> - **Options Considered:**  
>   1. **Single Header:** All types and functions in one `linker.h` file  
>   2. **Modular Headers:** Each component has its own header with only necessary exports  
>   3. **Two-Layer Headers:** Common types in base header, component-specific in separate headers  
> - **Decision:** Modular headers with minimal cross-inclusion  
> - **Rationale:** Modular headers enforce interface boundaries, reduce compilation dependencies, and make component responsibilities explicit. Learners can examine each header to understand what a component provides without seeing unrelated details.  
> - **Consequences:** Some type duplication may occur, but this is acceptable for educational clarity. Circular dependencies must be avoided.

### Linking Workflow

**Mental Model:** Picture an assembly line where each object file is a kit of parts. The ELF Reader unpacks each kit and sorts parts onto pallets (sections). The Section Merger takes pallets of the same type from all kits and stacks them in the warehouse with proper spacing. The Symbol Resolver creates a master inventory showing where each part (symbol) ended up. The Relocation Applier goes through assembly instructions (relocations) and updates part numbers to match the new warehouse locations. Finally, the Executable Writer creates shipping labels and packaging (headers) so the final product can be delivered to the customer (OS).

The linking process follows a strict sequence where each component's output becomes the next component's input. This data flow is crucial for understanding dependencies and ensuring correctness.

![Component Diagram](./diagrams/component-diagram.svg)

#### Step-by-Step Linking Sequence

1. **Initialization Phase**
   - The main program receives command-line arguments listing input `.o` files and output executable name
   - Memory is allocated for tracking the linking process state

2. **ELF Reading Phase** (Milestone 1)
   - For each input file path:
     1. Call `read_elf_file()` to parse the ELF file
     2. Validate ELF magic numbers, class, and endianness
     3. Parse section headers to build array of `ElfSection` structures
     4. Load section data (`.text`, `.data`, `.rodata`) into memory buffers
     5. Parse symbol table (`.symtab`) into `ElfSymbol` array
     6. Parse relocation tables (`.rela.text`, `.rela.data`) into `ElfRelocation` array
     7. Store everything in an `ObjectFile` structure
   - Result: Array of `ObjectFile` pointers, each representing a fully parsed object file

3. **Section Merging Phase** (Milestone 1)
   - Call `merge_all_sections()` with array of `ObjectFile` pointers
   - For each section type (`.text`, `.data`, `.rodata`, `.bss`):
     1. Group all sections of this type from all object files
     2. Verify compatibility (flags like `SHF_EXECINSTR` must match)
     3. Sort sections by alignment requirements (higher alignment first)
     4. Compute output layout: start with current offset, apply `align_to()` for each section, add section size, track padding
     5. For `.bss` sections: allocate virtual address space but no file space
   - Build `MergedSections` structure containing:
     - Array of `OutputSection` entries (one per merged section type)
     - Mapping from input (file_index, section_index) to output (section_id, offset)
   - Result: Complete layout of the final executable's sections

4. **Symbol Resolution Phase** (Milestone 2)
   - Call `resolve_all_symbols()` with array of `ObjectFile` pointers and `MergedSections`
   - First pass: Collect all global symbols
     1. Iterate through each object file's symbol table
     2. For each symbol with `STB_GLOBAL` or `STB_WEAK` binding:
        - Look up symbol name in string table
        - Record symbol definition (value, size, type, binding) along with source file
   - Second pass: Resolve symbols
     1. Apply strong/weak rules: strong symbols override weak ones
     2. Handle COMMON symbols (special merging by largest size)
     3. For defined symbols: compute final address = output section base + input section offset + symbol offset
     4. For undefined symbols: check if defined elsewhere; if not, report error
     5. Local symbols (`STB_LOCAL`) are discarded (not needed for linking)
   - Build `SymbolTable` structure with:
     - Hash table mapping symbol names to `SymbolEntry` (address, size, type)
     - Error tracking for unresolved symbols
   - Result: Complete mapping from symbol names to final virtual addresses

5. **Relocation Application Phase** (Milestone 3)
   - Call `apply_all_relocations()` with array of `ObjectFile` pointers, `MergedSections`, and `SymbolTable`
   - For each object file and each of its relocation tables:
     1. Determine which output section contains the relocation site
     2. For each relocation entry (`Elf64_Rela`):
        - Look up referenced symbol in `SymbolTable` to get target address
        - Compute relocation site address = output section base + relocation offset
        - Calculate value based on relocation type:
          - `R_X86_64_64`: value = symbol_address + addend
          - `R_X86_64_PC32`: value = symbol_address - relocation_site_address + addend - 4
        - Patch the value into the section data at the relocation offset
        - Check for overflow (32-bit field containing 64-bit value)
   - Result: Merged section data now contains correct absolute and relative addresses

6. **Executable Generation Phase** (Milestone 4)
   - Call `generate_executable()` with `MergedSections` and `SymbolTable`
     1. Create ELF header (`Elf64_Ehdr`):
        - Set `e_type` to `ET_EXEC` (2)
        - Set `e_entry` to address of `_start` symbol from `SymbolTable`
        - Set `e_phoff` to position of program headers
        - Set `e_shoff` to 0 (no section headers in executable)
     2. Create program headers (`Elf64_Phdr` array):
        - Text segment: `PT_LOAD`, covers `.text` and `.rodata`, permissions `PF_R | PF_X`
        - Data segment: `PT_LOAD`, covers `.data` and `.bss`, permissions `PF_R | PF_W`
        - Each segment aligned to 4096-byte page boundary
     3. Layout segments in memory:
        - Text segment at virtual address 0x400000 (typical Linux default)
        - Data segment immediately after text segment, page-aligned
        - File offsets correspond to segment contents with proper alignment
   - Result: `OutputExecutable` structure containing headers and segment data

7. **File Writing Phase** (Milestone 4)
   - Call `write_executable()` with `OutputExecutable` and output filename
     1. Open output file for binary writing
     2. Write ELF header
     3. Write program headers
     4. For each segment, seek to file offset and write segment contents
        - Text segment: write merged `.text` and `.rodata` data
        - Data segment: write merged `.data` data (`.bss` has no file content)
     5. Pad file to next page boundary if needed
     6. Close file
   - Result: Valid ELF executable file on disk

![Linking Sequence](./diagrams/linking-sequence.svg)

#### Data Transformations Through the Pipeline

| Stage | Input Data | Transformation | Output Data |
|-------|------------|----------------|-------------|
| **ELF Reading** | Raw bytes from `.o` files | Parse ELF structures, load sections | `ObjectFile` with sections, symbols, relocations |
| **Section Merging** | Multiple `ObjectFile` instances | Concatenate similar sections with padding | `MergedSections` layout and input→output mapping |
| **Symbol Resolution** | Multiple `ObjectFile` instances + `MergedSections` | Resolve cross-file references, compute addresses | `SymbolTable` with final symbol addresses |
| **Relocation Application** | `ObjectFile` instances + `MergedSections` + `SymbolTable` | Patch addresses in section data | Modified section data with resolved addresses |
| **Executable Generation** | `MergedSections` + `SymbolTable` | Create ELF headers, organize segments | `OutputExecutable` with headers and segment data |
| **File Writing** | `OutputExecutable` | Serialize to disk with proper offsets | ELF executable file |

> **Key Insight:** The pipeline creates increasingly refined representations of the program. Raw object files contain "local truth" (addresses relative to each file's sections). After merging, we have "global layout" (where everything goes in the final executable). After symbol resolution, we have "global addressing" (where each symbol lives). After relocation, we have "localized code" (instructions using final addresses). After executable generation, we have "loadable format" (headers telling the OS how to load everything).

#### Component Interactions and Data Dependencies

The components form a directed acyclic graph with the following dependencies:

```
ObjectFile[1..N] → ELF Reader
      ↓
ObjectFile[1..N] → Section Merger → MergedSections
      ↓                                    ↓
ObjectFile[1..N] → Symbol Resolver ← MergedSections → SymbolTable
      ↓                                    ↓               ↓
ObjectFile[1..N] → Relocation Applier ← MergedSections ← SymbolTable
                                             ↓               ↓
                                    Executable Writer ← SymbolTable
                                             ↓
                                     OutputExecutable
                                             ↓
                                       File Writer
```

**Critical Data Flow Notes:**
1. `ObjectFile` structures are read-only after creation—components never modify them
2. `MergedSections` is created by Section Merger and read by Symbol Resolver, Relocation Applier, and Executable Writer
3. `SymbolTable` is created by Symbol Resolver and read by Relocation Applier and Executable Writer
4. Relocation Applier modifies the section data buffers inside `MergedSections` in-place
5. Executable Writer creates a new `OutputExecutable` without modifying `MergedSections`

### Implementation Guidance

#### A. Technology Recommendations Table

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| **ELF Reader** | Manual parsing with `fread()` and struct casting | Using `libelf` or similar library |
| **Section Merger** | Arrays and loops for section grouping | Hash maps for section name lookup |
| **Symbol Resolver** | Linear search through symbol arrays | Hash table for O(1) symbol lookup |
| **Relocation Applier** | Switch statement for relocation types | Table-driven dispatch with function pointers |
| **Executable Writer** | Direct struct writing to file | Stream-based writing with buffering |

#### B. Recommended File/Module Structure

Create the following files with their respective responsibilities:

**`src/main.c`** - Orchestrates the pipeline:
```c
#include "elf_reader.h"
#include "section_merger.h"
#include "symbol_resolver.h"
#include "relocation_applier.h"
#include "executable_writer.h"
#include "utils.h"

int main(int argc, char** argv) {
    // TODO 1: Parse command line arguments
    // TODO 2: Allocate array for ObjectFile pointers
    // TODO 3: For each input file: read_elf_file()
    // TODO 4: merge_all_sections() → MergedSections*
    // TODO 5: resolve_all_symbols() → SymbolTable*
    // TODO 6: apply_all_relocations()
    // TODO 7: generate_executable() → OutputExecutable*
    // TODO 8: write_executable()
    // TODO 9: Free all allocated memory
    return 0;
}
```

**`src/utils.h`** - Shared utilities:
```c
#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <stdio.h>

// Alignment helper
uint64_t align_to(uint64_t value, uint64_t alignment);

// Error handling
void fatal_error(const char* format, ...);

// File reading
void* read_file(const char* filename, size_t* out_size);

#endif
```

**`src/utils.c`** - Implementation:
```c
#include "utils.h"
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

uint64_t align_to(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

void fatal_error(const char* format, ...) {
    va_list args;
    va_start(args, format);
    fprintf(stderr, "ERROR: ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
    exit(1);
}

void* read_file(const char* filename, size_t* out_size) {
    FILE* f = fopen(filename, "rb");
    if (!f) fatal_error("Cannot open file: %s", filename);
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    void* buffer = malloc(size);
    if (!buffer) fatal_error("Out of memory reading %s", filename);
    
    size_t read = fread(buffer, 1, size, f);
    if (read != (size_t)size) fatal_error("Short read on %s", filename);
    
    fclose(f);
    *out_size = size;
    return buffer;
}
```

#### C. Infrastructure Starter Code

**Complete ELF Structure Definitions** (place in `elf_reader.h`):

```c
#ifndef ELF_READER_H
#define ELF_READER_H

#include <stdint.h>

// ELF header structure (matches ELF64 specification)
typedef struct {
    unsigned char e_ident[16];
    uint16_t e_type;
    uint16_t e_machine;
    uint32_t e_version;
    uint64_t e_entry;
    uint64_t e_phoff;
    uint64_t e_shoff;
    uint32_t e_flags;
    uint16_t e_ehsize;
    uint16_t e_phentsize;
    uint16_t e_phnum;
    uint16_t e_shentsize;
    uint16_t e_shnum;
    uint16_t e_shstrndx;
} Elf64_Ehdr;

// Section header structure
typedef struct {
    uint32_t sh_name;
    uint32_t sh_type;
    uint64_t sh_flags;
    uint64_t sh_addr;
    uint64_t sh_offset;
    uint64_t sh_size;
    uint32_t sh_link;
    uint32_t sh_info;
    uint64_t sh_addralign;
    uint64_t sh_entsize;
} Elf64_Shdr;

// Symbol table entry
typedef struct {
    uint32_t st_name;
    unsigned char st_info;
    unsigned char st_other;
    uint16_t st_shndx;
    uint64_t st_value;
    uint64_t st_size;
} Elf64_Sym;

// Relocation entry with addend
typedef struct {
    uint64_t r_offset;
    uint64_t r_info;
    int64_t r_addend;
} Elf64_Rela;

// Parsed section information
typedef struct {
    Elf64_Shdr header;
    char* name;               // Section name from string table
    uint8_t* data;            // Raw section data (NULL for .bss)
    size_t data_size;         // Size of data buffer
} ElfSection;

// Parsed symbol information
typedef struct {
    Elf64_Sym sym;            // Raw symbol entry
    char* name;               // Symbol name
    ElfSection* section;      // Pointer to containing section (if any)
} ElfSymbol;

// Parsed relocation information
typedef struct {
    Elf64_Rela rela;          // Raw relocation entry
    ElfSymbol* symbol;        // Pointer to referenced symbol
    ElfSection* target_section; // Section containing relocation site
} ElfRelocation;

// Complete parsed object file
typedef struct {
    char* filename;
    Elf64_Ehdr header;
    
    ElfSection* sections;     // Array of sections
    uint16_t num_sections;
    
    ElfSymbol* symbols;       // Array of symbols
    uint32_t num_symbols;
    
    ElfRelocation* relocations; // Array of relocations
    uint32_t num_relocations;
    
    char* shstrtab;           // Section header string table
    char* strtab;             // Symbol string table
} ObjectFile;

// Function declarations
ObjectFile* read_elf_file(const char* filename);
void free_object_file(ObjectFile* obj);

#endif
```

#### D. Core Logic Skeleton Code

**`src/elf_reader.c`** - Main parsing logic:

```c
#include "elf_reader.h"
#include "utils.h"
#include <string.h>

ObjectFile* read_elf_file(const char* filename) {
    // TODO 1: Allocate and initialize ObjectFile structure
    // TODO 2: Read file into memory using read_file()
    // TODO 3: Validate ELF magic numbers (ELFMAG0-3)
    // TODO 4: Check ELF class (ELFCLASS64) and endianness (ELFDATA2LSB)
    // TODO 5: Parse ELF header (Elf64_Ehdr) from buffer
    // TODO 6: Load section header string table (.shstrtab)
    // TODO 7: Parse all section headers, storing Elf64_Shdr copies
    // TODO 8: For each PROGBITS section, load data into buffer
    // TODO 9: Find .symtab section, parse all Elf64_Sym entries
    // TODO 10: Load symbol string table (.strtab) for symbol names
    // TODO 11: Find .rela.* sections, parse all Elf64_Rela entries
    // TODO 12: Link relocations to their target symbols
    // TODO 13: Return populated ObjectFile*
    return NULL;
}
```

**`src/section_merger.c`** - Section merging logic:

```c
#include "section_merger.h"
#include "utils.h"
#include <stdlib.h>

typedef struct {
    uint32_t file_index;      // Which ObjectFile
    uint32_t section_index;   // Which section within that file
    uint64_t output_offset;   // Offset within output section
} InputSectionMapping;

typedef struct {
    char* name;               // Output section name (e.g., ".text")
    uint64_t sh_type;         // Section type (SHT_PROGBITS, etc.)
    uint64_t sh_flags;        // Section flags (SHF_ALLOC, etc.)
    uint64_t sh_addralign;    // Alignment requirement
    uint64_t file_offset;     // Offset in output file
    uint64_t virtual_addr;    // Virtual address in memory
    uint64_t size;            // Total size including padding
    uint8_t* data;            // Merged data (NULL for .bss)
} OutputSection;

struct MergedSections {
    OutputSection* sections;
    uint32_t num_sections;
    InputSectionMapping* mappings;
    uint32_t num_mappings;
};

MergedSections* merge_all_sections(ObjectFile** objects, uint32_t count) {
    // TODO 1: Allocate MergedSections structure
    // TODO 2: For each section type (.text, .data, .rodata, .bss):
    //   a. Find all input sections of this type across all files
    //   b. Verify compatible flags (SHF_EXECINSTR, SHF_WRITE, etc.)
    //   c. Sort by alignment requirement (highest first)
    //   d. Compute output layout: current_offset = align_to(current_offset, section.alignment)
    //   e. Record mapping from input section to output offset
    //   f. current_offset += section.size
    //   g. For .bss: track size but don't allocate file space
    // TODO 3: Allocate buffers for PROGBITS sections and copy input data
    // TODO 4: Apply padding bytes between sections as needed
    // TODO 5: Return MergedSections* with complete layout
    return NULL;
}
```

#### E. Language-Specific Hints (C)

1. **Memory Management:** Use `malloc()`/`free()` consistently. Consider implementing a simple arena allocator for better performance.
2. **Endianness:** x86-64 uses little-endian. Use `__attribute__((packed))` or `#pragma pack(1)` for structs to avoid padding issues.
3. **File I/O:** Use `fopen()` with `"rb"` for reading, `"wb"` for writing binary files.
4. **Error Handling:** Use `fatal_error()` wrapper for unrecoverable errors. Return error codes for recoverable ones.
5. **Data Structures:** Use arrays for simplicity initially. Later optimize with hash tables (consider `uthash` library if needed).

#### F. Milestone Checkpoint

After implementing the high-level architecture, verify the pipeline connects correctly:

```bash
# Compile with skeleton implementations
gcc -c src/*.c -I. -o linker.o
gcc linker.o -o linker

# Test with empty main (should fail gracefully)
./linker test.o -o test
# Expected: Error message about file not found or parsing error
# This confirms the pipeline structure compiles and runs
```

**Signs of Correct Architecture:**
- Each `.c` file includes only the headers it needs
- `main.c` calls functions in the correct order
- No circular dependencies between components
- Compilation succeeds with skeleton implementations

**Common Architecture Mistakes:**
- Component A directly modifies Component B's internal data
- Missing `#include` causing implicit function declarations
- Global variables used for inter-component communication
- Memory leaks from not freeing intermediate structures


## Data Model

> **Milestone(s):** This section defines the foundational data structures that enable all linking operations, spanning Milestones 1-4.

The static linker's architecture revolves around transforming input object files into a cohesive executable through a series of data transformations. This transformation requires precise modeling of ELF components, intermediate linking state, and output structures. The data model serves as the connective tissue between components—each component consumes and produces instances of these structures. A clear mental model emerges: think of the linker as a **document assembly system** where object files are source documents with chapters (sections), cross-references (symbols), and placeholder annotations (relocations). The data model defines how to catalog these documents, create a master outline (merged sections), resolve all cross-references (symbol table), and produce the final assembled document (executable).

Understanding these data structures is crucial because they embody the entire linking state. The `ObjectFile` structures capture everything about input files without modification, preserving the "raw materials." The `MergedSections` structure tracks how these raw materials are rearranged in the output. The `SymbolTable` serves as the central directory that maps names to final locations. Each structure maintains precise relationships—what input section maps to which output offset, which symbol belongs to which section, and which relocation targets which location.

### ELF File Representation

The ELF file representation structures mirror the on-disk ELF format exactly, providing a parsed, in-memory view of input object files. These structures follow the ELF standard but add convenience fields for easier processing.

#### Mental Model: Library Index Cards

Imagine each object file as a library book. The ELF representation is like an **index card system** for that book: it doesn't contain the book's full content, but it catalogs every chapter (section), every important term (symbol), and every cross-reference (relocation) with page numbers. Each index card points to the actual content in the book, allowing you to work with the catalog without modifying the original pages.

#### Core Structures

| Structure Name | Purpose | Key Relationships |
|----------------|---------|-------------------|
| `Elf64_Ehdr` | ELF file header identifying file type, architecture, and locating section/segment tables | Contains pointers to section and program header tables |
| `Elf64_Shdr` | Section header describing a single section's type, flags, size, alignment, and location | Points to string table entries for names; links to related sections |
| `Elf64_Sym` | Symbol table entry defining a named symbol with binding, type, size, and section association | References a section via `st_shndx`; name stored in string table |
| `Elf64_Rela` | Relocation with addend specifying where and how to patch an address | Contains offset within section and symbol index to use |
| `ElfSection` | Enhanced section representation with header, name, and actual data | Wraps `Elf64_Shdr` with convenient access to data and name |
| `ElfSymbol` | Enhanced symbol with parsed name and direct section reference | Wraps `Elf64_Sym` with resolved name string and section pointer |
| `ElfRelocation` | Enhanced relocation with resolved symbol and target section pointers | Wraps `Elf64_Rela` with direct references instead of indices |
| `ObjectFile` | Complete parsed representation of an input ELF object file | Contains arrays of sections, symbols, relocations, plus string tables |

#### Detailed Field Specifications

**`Elf64_Ehdr` (ELF File Header)**
| Field | Type | Description |
|-------|------|-------------|
| `e_ident` | `unsigned char[16]` | ELF identification bytes (magic number, class, data encoding, version) |
| `e_type` | `uint16_t` | File type (1 = ET_REL for relocatable object files) |
| `e_machine` | `uint16_t` | Target architecture (62 = EM_X86_64 for x86-64) |
| `e_version` | `uint32_t` | ELF version (1 = EV_CURRENT) |
| `e_entry` | `uint64_t` | Entry point address (0 for object files) |
| `e_phoff` | `uint64_t` | Program header table offset (0 for object files) |
| `e_shoff` | `uint64_t` | Section header table offset (bytes from file start) |
| `e_flags` | `uint32_t` | Processor-specific flags |
| `e_ehsize` | `uint16_t` | ELF header size in bytes |
| `e_phentsize` | `uint16_t` | Program header entry size (0 for object files) |
| `e_phnum` | `uint16_t` | Number of program headers (0 for object files) |
| `e_shentsize` | `uint16_t` | Section header entry size in bytes |
| `e_shnum` | `uint16_t` | Number of section headers |
| `e_shstrndx` | `uint16_t` | Section header string table index |

**`Elf64_Shdr` (Section Header)**
| Field | Type | Description |
|-------|------|-------------|
| `sh_name` | `uint32_t` | Offset into section name string table |
| `sh_type` | `uint32_t` | Section type (1 = SHT_PROGBITS, 2 = SHT_SYMTAB, 3 = SHT_STRTAB, 4 = SHT_RELA, 8 = SHT_NOBITS) |
| `sh_flags` | `uint64_t` | Section attributes (SHF_ALLOC = 0x2, SHF_WRITE = 0x1, SHF_EXECINSTR = 0x4) |
| `sh_addr` | `uint64_t` | Virtual address in memory (0 for object files) |
| `sh_offset` | `uint64_t` | Offset of section data in file |
| `sh_size` | `uint64_t` | Section size in bytes |
| `sh_link` | `uint32_t` | Link to another section (depends on type) |
| `sh_info` | `uint32_t` | Additional section information (depends on type) |
| `sh_addralign` | `uint64_t` | Section alignment requirement (power of 2) |
| `sh_entsize` | `uint64_t` | Entry size for tables (0 if not a table) |

**`Elf64_Sym` (Symbol Table Entry)**
| Field | Type | Description |
|-------|------|-------------|
| `st_name` | `uint32_t` | Offset into symbol name string table |
| `st_info` | `unsigned char` | Symbol binding (bits 4-7) and type (bits 0-3) |
| `st_other` | `unsigned char` | Symbol visibility (usually 0) |
| `st_shndx` | `uint16_t` | Section index containing symbol (special values: 0 = undefined, 0xfff1 = absolute) |
| `st_value` | `uint64_t` | Symbol value (offset within section or absolute value) |
| `st_size` | `uint64_t` | Symbol size in bytes |

**`Elf64_Rela` (Relocation with Addend)**
| Field | Type | Description |
|-------|------|-------------|
| `r_offset` | `uint64_t` | Offset within target section to patch |
| `r_info` | `uint64_t` | Symbol index (bits 0-31) and relocation type (bits 32-63) |
| `r_addend` | `int64_t` | Constant addend added to symbol value |

**`ElfSection` (Enhanced Section Representation)**
| Field | Type | Description |
|-------|------|-------------|
| `header` | `Elf64_Shdr` | Raw section header |
| `name` | `char*` | Null-terminated section name (parsed from string table) |
| `data` | `uint8_t*` | Pointer to section data (allocated memory copy) |
| `data_size` | `size_t` | Size of allocated data (may be 0 for SHT_NOBITS) |

**`ElfSymbol` (Enhanced Symbol Representation)**
| Field | Type | Description |
|-------|------|-------------|
| `sym` | `Elf64_Sym` | Raw symbol table entry |
| `name` | `char*` | Null-terminated symbol name (parsed from string table) |
| `section` | `ElfSection*` | Pointer to containing section (or NULL for absolute/undefined) |

**`ElfRelocation` (Enhanced Relocation Representation)**
| Field | Type | Description |
|-------|------|-------------|
| `rela` | `Elf64_Rela` | Raw relocation entry |
| `symbol` | `ElfSymbol*` | Pointer to referenced symbol (resolved during parsing) |
| `target_section` | `ElfSection*` | Pointer to section containing relocation site |

**`ObjectFile` (Complete Object File Representation)**
| Field | Type | Description |
|-------|------|-------------|
| `filename` | `char*` | Source filename for error reporting |
| `header` | `Elf64_Ehdr` | ELF file header |
| `sections` | `ElfSection*` | Array of parsed sections |
| `num_sections` | `uint16_t` | Number of sections in array |
| `symbols` | `ElfSymbol*` | Array of parsed symbols |
| `num_symbols` | `uint32_t` | Number of symbols in array |
| `relocations` | `ElfRelocation*` | Array of parsed relocations |
| `num_relocations` | `uint32_t` | Number of relocations in array |
| `shstrtab` | `char*` | Section name string table data |
| `strtab` | `char*` | Symbol name string table data |

#### Architecture Decision: Direct Pointer Resolution vs. Index-Based Lookup

> **Decision: Resolve String and Section References to Direct Pointers During Parsing**
> - **Context**: ELF files use indices (string table offsets, section indices) for references, requiring lookup during processing. The linker must frequently access symbol names and section data.
> - **Options Considered**:
>   1. **Store raw indices only**: Keep the ELF structures exactly as on disk, looking up strings and sections each time they're needed
>   2. **Resolve to pointers during parsing**: Convert all indices to direct pointers when loading the object file
>   3. **Hybrid lazy resolution**: Store indices but provide accessor functions that resolve on demand
> - **Decision**: Resolve to direct pointers during parsing (Option 2)
> - **Rationale**: The linker performs thousands of symbol lookups and section accesses. Direct pointers eliminate repeated table lookups, simplifying code and improving performance. Since object files are read once and never modified, the one-time resolution cost is negligible. The memory overhead of storing pointers is acceptable given typical object file sizes.
> - **Consequences**: Faster symbol resolution and relocation processing; simpler code without repeated index lookups; memory overhead from storing pointers; parsing phase becomes slightly more complex.

| Option | Pros | Cons | Rejected? |
|--------|------|------|-----------|
| Raw indices only | Minimal memory, simpler parsing | Repeated lookups slow down processing, complex access patterns | Yes |
| **Resolve during parsing** | **Fast access, simpler processing logic** | **One-time resolution cost, extra memory for pointers** | **Chosen** |
| Hybrid lazy resolution | Best of both worlds in theory | Complex caching logic, unpredictable performance | Yes |

#### Common Pitfalls in ELF Representation

⚠️ **Pitfall: Forgetting to Handle SHT_NOBITS Sections Differently**
- **Description**: Treating `.bss` sections (type `SHT_NOBITS`) like regular sections with file data
- **Why it's wrong**: `SHT_NOBITS` sections occupy memory but no file space; their `sh_offset` field may be meaningless or zero. Copying supposed "data" from the file will read garbage or cause buffer overflows
- **Fix**: Check `sh_type` when loading section data: for `SHT_NOBITS`, set `data` pointer to `NULL` and `data_size` to 0, but track `sh_size` for memory allocation

⚠️ **Pitfall: Assuming String Tables are Null-Terminated**
- **Description**: Treating string table data as a single null-terminated string
- **Why it's wrong**: ELF string tables contain multiple null-terminated strings packed together. The `sh_size` includes all strings plus their terminators. Using standard string functions on the entire buffer will stop at the first null
- **Fix**: When looking up strings by offset, validate that `offset < sh_size`, then treat `&strtab[offset]` as the start of that specific string

⚠️ **Pitfall: Ignoring Section Alignment Requirements**
- **Description**: Not checking `sh_addralign` when accessing or processing section data
- **Why it's wrong**: Some sections (especially those containing machine code or aligned data) require specific alignment for correct operation. Violating alignment can cause crashes or incorrect behavior on some architectures
- **Fix**: Always align section data addresses and sizes according to `sh_addralign` using the `align_to` helper

### Internal Linking Structures

The internal linking structures track the linker's intermediate state—how input components are transformed and combined. These structures evolve as linking progresses and ultimately define the output executable.

#### Mental Model: Construction Site Blueprint

Imagine building a house from prefabricated modules (object files). The internal linking structures are the **construction site blueprint**: they show where each module will be placed in the final structure (`MergedSections`), a directory of all named components and their final locations (`SymbolTable`), and instructions for connecting wires between modules (`InputSectionMapping`). Unlike the static catalog of input files, these structures are dynamic—they're drawn up, revised, and finalized during construction.

#### Core Structures

| Structure Name | Purpose | Key Relationships |
|----------------|---------|-------------------|
| `OutputSection` | Represents a section in the final executable | Contains concatenated data from multiple input sections |
| `InputSectionMapping` | Maps input sections to their location in output sections | Links `ObjectFile` + section index to `OutputSection` + offset |
| `MergedSections` | Collection of all output sections and their mappings | The central layout plan for section merging |
| `SymbolTable` | Resolved global symbol table | Maps symbol names to final addresses in output |
| `OutputExecutable` | Complete executable ready for writing | Contains headers, segments, and final section data |

#### Detailed Field Specifications

**`OutputSection` (Final Executable Section)**
| Field | Type | Description |
|-------|------|-------------|
| `name` | `char*` | Section name (e.g., ".text", ".data") |
| `sh_type` | `uint64_t` | Section type (matches input section types) |
| `sh_flags` | `uint64_t` | Section flags (combined from merged sections) |
| `sh_addralign` | `uint64_t` | Maximum alignment of constituent sections |
| `file_offset` | `uint64_t` | Offset of section data in output file |
| `virtual_addr` | `uint64_t` | Virtual address in memory for loaded executable |
| `size` | `uint64_t` | Total size of section (including padding) |
| `data` | `uint8_t*` | Concatenated section data (allocated buffer) |

**`InputSectionMapping` (Input-to-Output Location Map)**
| Field | Type | Description |
|-------|------|-------------|
| `file_index` | `uint32_t` | Index into array of `ObjectFile*` inputs |
| `section_index` | `uint32_t` | Index within that file's `sections` array |
| `output_offset` | `uint64_t` | Byte offset within the containing `OutputSection` |

**`MergedSections` (Complete Section Layout)**
| Field | Type | Description |
|-------|------|-------------|
| `sections` | `OutputSection*` | Array of output sections in layout order |
| `num_sections` | `uint32_t` | Number of output sections |
| `mappings` | `InputSectionMapping*` | Array of all input-to-output mappings |
| `num_mappings` | `uint32_t` | Number of mappings (equals total input sections) |

**`SymbolTable` (Resolved Global Symbol Directory)**
> *Note: `SymbolTable` is a placeholder type—its implementation varies based on resolution strategy. The table below shows typical contents.*

| Field | Type | Description |
|-------|------|-------------|
| `entries` | `SymbolEntry*` | Array of resolved symbol entries |
| `num_entries` | `uint32_t` | Number of symbols in table |
| `by_name` | `HashTable` | Hash table for name lookup (implementation dependent) |

**Typical `SymbolEntry` Structure (Not in naming conventions, but implied)**
| Field | Type | Description |
|-------|------|-------------|
| `name` | `char*` | Symbol name |
| `value` | `uint64_t` | Final virtual address in output |
| `size` | `uint64_t` | Symbol size |
| `binding` | `uint8_t` | Symbol binding (STB_GLOBAL, STB_WEAK, etc.) |
| `type` | `uint8_t` | Symbol type (STT_FUNC, STT_OBJECT, etc.) |
| `defined` | `bool` | Whether symbol has a definition |
| `output_section` | `OutputSection*` | Pointer to containing output section |
| `offset_in_section` | `uint64_t` | Offset within output section |

**`OutputExecutable` (Final Executable Representation)**
> *Note: `OutputExecutable` is a placeholder type—its contents depend on writing strategy.*

| Field | Type | Description |
|-------|------|-------------|
| `header` | `Elf64_Ehdr` | ELF executable header |
| `program_headers` | `Elf64_Phdr*` | Array of program headers (for PT_LOAD segments) |
| `num_program_headers` | `uint16_t` | Number of program headers |
| `sections` | `OutputSection*` | Final section data (same as in `MergedSections`) |
| `num_sections` | `uint32_t` | Number of sections |
| `entry_point` | `uint64_t` | Virtual address of entry point (`_start`) |

#### Architecture Decision: Two-Level Section Mapping

> **Decision: Maintain Both Output Sections and Explicit Input Mappings**
> - **Context**: The linker must track where each input section ends up in the output, both for relocation calculations and for debugging. This requires mapping (file_index, section_index) → output location.
> - **Options Considered**:
>   1. **Implicit calculation only**: Store only output sections; calculate input positions by summing sizes of preceding sections from same input
>   2. **Explicit mapping array**: Maintain separate `InputSectionMapping` array for direct lookup
>   3. **Augmented output sections**: Store input section ranges within each `OutputSection`
> - **Decision**: Explicit mapping array (Option 2)
> - **Rationale**: Relocation processing requires frequent lookup of "where did this input section go?" An explicit array provides O(1) lookup by (file_index, section_index). The memory overhead is minimal (12 bytes per input section). Implicit calculation would require scanning through sections for each lookup, which is O(n) and complex when sections from one file aren't contiguous in output.
> - **Consequences**: Fast relocation processing; simple, clear data structure; small memory overhead; need to build and maintain mapping during merging.

| Option | Pros | Cons | Rejected? |
|--------|------|------|-----------|
| Implicit calculation | Minimal memory | Slow lookups, complex logic, fragile to layout changes | Yes |
| **Explicit mapping array** | **Fast O(1) lookups, simple logic** | **Memory overhead, must build mappings** | **Chosen** |
| Augmented output sections | Groups related data together | Complex data structure, still requires searching within ranges | Yes |

#### Relationship Diagram

The data structures form a clear hierarchy and reference network:

![Data Model Relationships](./diagrams/data-model-diagram.svg)

**Key Relationships:**
1. **ObjectFile contains** → ElfSection, ElfSymbol, ElfRelocation
2. **ElfRelocation references** → ElfSymbol (target) and ElfSection (location)
3. **ElfSymbol references** → ElfSection (containing section)
4. **MergedSections contains** → OutputSection and InputSectionMapping
5. **InputSectionMapping references** → (implicitly) ObjectFile + ElfSection via indices
6. **SymbolTable entries reference** → OutputSection + offset
7. **OutputExecutable contains** → OutputSection data

#### Common Pitfalls in Internal Structures

⚠️ **Pitfall: Not Tracking Both File and Virtual Address Offsets**
- **Description**: Storing only file offsets in `OutputSection`, forgetting virtual addresses
- **Why it's wrong**: Executables have two address spaces: file offsets (where data lives on disk) and virtual addresses (where sections load in memory). They differ due to alignment padding and segment layout. Relocations need virtual addresses, while writing needs file offsets
- **Fix**: Store both `file_offset` and `virtual_addr` in `OutputSection`, and calculate both during layout

⚠️ **Pitfall: Mixing Alignment Types**
- **Description**: Using the same alignment for file offset and virtual address alignment
- **Why it's wrong**: File offset alignment ensures efficient disk I/O (often 1-byte or small blocks), while virtual address alignment ensures correct memory mapping (often page-aligned to 4096 bytes). They have different requirements
- **Fix**: Apply `sh_addralign` to virtual addresses, but use minimal padding (1-byte) for file offsets unless specifically required

⚠️ **Pitfall: Forgetting to Combine Section Flags Correctly**
- **Description**: Simply copying flags from first input section to output section
- **Why it's wrong**: Different input sections of the same name might have slightly different flags (e.g., one .rodata marked SHF_ALLOC, another not). The output section must have flags that are a valid superset
- **Fix**: Compute output section flags as the bitwise AND of all merged input sections' flags—only flags present in ALL inputs should remain

### Implementation Guidance

#### A. Technology Recommendations Table

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| ELF Parsing | Manual parsing with `fread`/`memcpy` | Use libelf or similar library |
| Data Structures | Simple arrays with linear search | Hash tables for symbol lookup, balanced trees for mappings |
| Memory Management | Manual `malloc`/`free` with careful tracking | Reference counting or arena allocator |
| Alignment | Custom `align_to` function | Compiler intrinsics (`__builtin_align_up`) |

#### B. Recommended File/Module Structure

```
linker/
├── include/
│   ├── elfdefs.h          ← ELF constants and raw structure definitions
│   ├── datamodel.h        ← Enhanced data structures (ObjectFile, OutputSection, etc.)
│   └── linker.h           ← Main linker interface functions
├── src/
│   ├── elfreader.c        ← ELF parsing and ObjectFile creation
│   ├── datamodel.c        ← Memory management for data structures
│   ├── section_merger.c   ← Section merging logic
│   ├── symbol_resolver.c  ← Symbol resolution logic
│   ├── relocations.c      ← Relocation application
│   ├── executable.c       ← Executable generation
│   └── utils.c           ← Alignment, error reporting helpers
├── tests/
│   ├── test_elfreader.c
│   └── test_datamodel.c
└── main.c                ← Command-line interface
```

#### C. Infrastructure Starter Code

**`include/elfdefs.h` - ELF Constants and Raw Structures:**
```c
#ifndef ELFDEFS_H
#define ELFDEFS_H

#include <stdint.h>

/* ELF identification indices */
#define EI_NIDENT 16
#define ELFMAG0   0x7F
#define ELFMAG1   'E'
#define ELFMAG2   'L'
#define ELFMAG3   'F'
#define ELFCLASS64 2
#define ELFDATA2LSB 1
#define EV_CURRENT 1

/* File types */
#define ET_REL 1

/* Machines */
#define EM_X86_64 62

/* Section types */
#define SHT_NULL     0
#define SHT_PROGBITS 1
#define SHT_SYMTAB   2
#define SHT_STRTAB   3
#define SHT_RELA     4
#define SHT_NOBITS   8

/* Section flags */
#define SHF_WRITE     0x1
#define SHF_ALLOC     0x2
#define SHF_EXECINSTR 0x4

/* Symbol bindings */
#define STB_LOCAL  0
#define STB_GLOBAL 1
#define STB_WEAK   2

/* Symbol types */
#define STT_NOTYPE  0
#define STT_OBJECT  1
#define STT_FUNC    2
#define STT_SECTION 3

/* Relocation types for x86-64 */
#define R_X86_64_64   1
#define R_X86_64_PC32 2

/* Raw ELF structures (packed to match on-disk layout) */
typedef struct __attribute__((packed)) {
    unsigned char e_ident[EI_NIDENT];
    uint16_t e_type;
    uint16_t e_machine;
    uint32_t e_version;
    uint64_t e_entry;
    uint64_t e_phoff;
    uint64_t e_shoff;
    uint32_t e_flags;
    uint16_t e_ehsize;
    uint16_t e_phentsize;
    uint16_t e_phnum;
    uint16_t e_shentsize;
    uint16_t e_shnum;
    uint16_t e_shstrndx;
} Elf64_Ehdr;

typedef struct __attribute__((packed)) {
    uint32_t sh_name;
    uint32_t sh_type;
    uint64_t sh_flags;
    uint64_t sh_addr;
    uint64_t sh_offset;
    uint64_t sh_size;
    uint32_t sh_link;
    uint32_t sh_info;
    uint64_t sh_addralign;
    uint64_t sh_entsize;
} Elf64_Shdr;

typedef struct __attribute__((packed)) {
    uint32_t st_name;
    unsigned char st_info;
    unsigned char st_other;
    uint16_t st_shndx;
    uint64_t st_value;
    uint64_t st_size;
} Elf64_Sym;

typedef struct __attribute__((packed)) {
    uint64_t r_offset;
    uint64_t r_info;
    int64_t r_addend;
} Elf64_Rela;

#endif /* ELFDEFS_H */
```

**`include/datamodel.h` - Enhanced Data Structures:**
```c
#ifndef DATAMODEL_H
#define DATAMODEL_H

#include "elfdefs.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* Forward declarations */
typedef struct ObjectFile ObjectFile;
typedef struct ElfSection ElfSection;
typedef struct ElfSymbol ElfSymbol;
typedef struct ElfRelocation ElfRelocation;
typedef struct OutputSection OutputSection;
typedef struct InputSectionMapping InputSectionMapping;
typedef struct MergedSections MergedSections;
typedef struct SymbolTable SymbolTable;
typedef struct OutputExecutable OutputExecutable;

/* Enhanced section with data and parsed name */
struct ElfSection {
    Elf64_Shdr header;
    char* name;           /* Null-terminated section name */
    uint8_t* data;        /* Section data (NULL for SHT_NOBITS) */
    size_t data_size;     /* Size of allocated data buffer */
};

/* Enhanced symbol with parsed name and section pointer */
struct ElfSymbol {
    Elf64_Sym sym;
    char* name;           /* Null-terminated symbol name */
    ElfSection* section;  /* Containing section (NULL if undefined/absolute) */
};

/* Enhanced relocation with direct pointers */
struct ElfRelocation {
    Elf64_Rela rela;
    ElfSymbol* symbol;        /* Referenced symbol */
    ElfSection* target_section; /* Section containing relocation site */
};

/* Complete object file representation */
struct ObjectFile {
    char* filename;           /* Source filename for error reporting */
    Elf64_Ehdr header;
    ElfSection* sections;     /* Array of sections */
    uint16_t num_sections;
    ElfSymbol* symbols;       /* Array of symbols */
    uint32_t num_symbols;
    ElfRelocation* relocations; /* Array of relocations */
    uint32_t num_relocations;
    char* shstrtab;           /* Section name string table */
    char* strtab;             /* Symbol name string table */
};

/* Output section in final executable */
struct OutputSection {
    char* name;
    uint64_t sh_type;
    uint64_t sh_flags;
    uint64_t sh_addralign;
    uint64_t file_offset;     /* Offset in output file */
    uint64_t virtual_addr;    /* Virtual address in memory */
    uint64_t size;            /* Total size including padding */
    uint8_t* data;            /* Concatenated section data */
};

/* Mapping from input section to output location */
struct InputSectionMapping {
    uint32_t file_index;      /* Index in objects array */
    uint32_t section_index;   /* Index in that file's sections array */
    uint64_t output_offset;   /* Offset within containing OutputSection */
};

/* Complete merged section layout */
struct MergedSections {
    OutputSection* sections;  /* Array of output sections */
    uint32_t num_sections;
    InputSectionMapping* mappings; /* Array of all input mappings */
    uint32_t num_mappings;
};

/* Placeholder for symbol table - actual implementation varies */
struct SymbolTable {
    /* To be defined in symbol_resolver.c */
    void* internal_data;
};

/* Placeholder for executable - actual implementation varies */
struct OutputExecutable {
    /* To be defined in executable.c */
    void* internal_data;
};

/* Memory management functions */
ObjectFile* create_object_file(const char* filename);
void free_object_file(ObjectFile* obj);
MergedSections* create_merged_sections(void);
void free_merged_sections(MergedSections* merged);

#endif /* DATAMODEL_H */
```

**`src/utils.c` - Alignment and Error Helpers:**
```c
#include "linker.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

/* Align value to given boundary (power of 2) */
uint64_t align_to(uint64_t value, uint64_t alignment) {
    if (alignment == 0) return value;
    return (value + alignment - 1) & ~(alignment - 1);
}

/* Fatal error reporting with format string */
void fatal_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "linker error: ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
    exit(1);
}

/* Safe memory allocation with error checking */
void* xmalloc(size_t size) {
    void* ptr = malloc(size);
    if (!ptr && size > 0) {
        fatal_error("out of memory allocating %zu bytes", size);
    }
    return ptr;
}

void* xcalloc(size_t count, size_t size) {
    void* ptr = calloc(count, size);
    if (!ptr && count > 0 && size > 0) {
        fatal_error("out of memory allocating %zu * %zu bytes", count, size);
    }
    return ptr;
}

void* xrealloc(void* ptr, size_t size) {
    void* new_ptr = realloc(ptr, size);
    if (!new_ptr && size > 0) {
        fatal_error("out of memory reallocating to %zu bytes", size);
    }
    return new_ptr;
}
```

#### D. Core Logic Skeleton Code

**`src/datamodel.c` - Memory Management for Data Structures:**
```c
#include "datamodel.h"
#include "linker.h"
#include <stdlib.h>
#include <string.h>

/* Create and initialize a new ObjectFile structure */
ObjectFile* create_object_file(const char* filename) {
    ObjectFile* obj = (ObjectFile*)xcalloc(1, sizeof(ObjectFile));
    
    // TODO 1: Allocate and copy filename string
    // TODO 2: Initialize arrays to NULL and counts to 0
    // TODO 3: Set default values in header
    
    return obj;
}

/* Free all memory associated with an ObjectFile */
void free_object_file(ObjectFile* obj) {
    if (!obj) return;
    
    // TODO 1: Free filename string
    // TODO 2: For each section: free name string and data buffer
    // TODO 3: Free sections array
    // TODO 4: For each symbol: free name string
    // TODO 5: Free symbols array
    // TODO 6: Free relocations array
    // TODO 7: Free string tables (shstrtab, strtab)
    // TODO 8: Free ObjectFile structure itself
}

/* Create empty MergedSections structure */
MergedSections* create_merged_sections(void) {
    MergedSections* merged = (MergedSections*)xcalloc(1, sizeof(MergedSections));
    
    // TODO 1: Initialize sections array to NULL
    // TODO 2: Initialize mappings array to NULL
    // TODO 3: Set counts to 0
    
    return merged;
}

/* Free MergedSections and all contained data */
void free_merged_sections(MergedSections* merged) {
    if (!merged) return;
    
    // TODO 1: For each OutputSection: free name string and data buffer
    // TODO 2: Free sections array
    // TODO 3: Free mappings array
    // TODO 4: Free MergedSections structure
}
```

**`include/linker.h` - Main Interface Functions:**
```c
#ifndef LINKER_H
#define LINKER_H

#include "datamodel.h"

/* Main linker interface functions */
ObjectFile* read_elf_file(const char* filename);
MergedSections* merge_all_sections(ObjectFile** objects, uint32_t count);
SymbolTable* resolve_all_symbols(ObjectFile** objects, uint32_t count, 
                                 MergedSections* merged);
void apply_all_relocations(ObjectFile** objects, uint32_t count,
                          MergedSections* merged, SymbolTable* symbols);
OutputExecutable* generate_executable(MergedSections* merged, 
                                     SymbolTable* symbols);
void write_executable(OutputExecutable* exec, const char* filename);

/* Utility functions */
uint64_t align_to(uint64_t value, uint64_t alignment);
void fatal_error(const char* fmt, ...);

/* Memory helpers */
void* xmalloc(size_t size);
void* xcalloc(size_t count, size_t size);
void* xrealloc(void* ptr, size_t size);

#endif /* LINKER_H */
```

#### E. Language-Specific Hints

- **Use `memcpy` for structure loading**: ELF structures must be copied byte-for-byte from file to memory to handle endianness correctly
- **String handling**: Always allocate new strings with `strdup` when parsing from string tables—don't keep pointers into the original buffer which will be freed
- **Flexible array members**: Consider using flexible array members for variable-sized data in structures to reduce memory fragmentation
- **Pointer arithmetic**: When calculating offsets in section data, use `uint8_t*` pointers and explicit casts to avoid alignment issues

#### F. Milestone Checkpoint - Data Model Validation

**After implementing data structures (pre-Milestone 1):**
1. **Compile test program**: `gcc -c test_datamodel.c datamodel.c utils.c -o test_datamodel`
2. **Run basic tests**: Program should create and free structures without memory leaks
3. **Verify with valgrind**: `valgrind --leak-check=full ./test_datamodel` should show "no leaks are possible"
4. **Expected behavior**: Structures allocate and deallocate correctly; alignment function produces correct results
5. **Signs of problems**: Segmentation faults (uninitialized pointers), memory leaks (forgetting to free), alignment errors (wrong padding calculations)

#### G. Debugging Tips for Data Model Issues

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Segmentation fault when accessing symbol name | String pointer not initialized or incorrectly parsed | Check `read_elf_file` string table parsing; verify offset is within string table bounds | Ensure string table data is copied and null-terminated |
| Memory leak showing in valgrind | Missing `free` for allocated arrays or strings | Add missing `free` calls in reverse order of allocation | Implement comprehensive cleanup functions |
| Alignment produces wrong offsets | Off-by-one error in `align_to` function | Test with values: `align_to(0, 16)`, `align_to(15, 16)`, `align_to(16, 16)` | Use `(value + alignment - 1) & ~(alignment - 1)` |
| Relocation can't find input section | `InputSectionMapping` not built correctly or indices wrong | Print mapping table after merging; verify indices match `ObjectFile` arrays | Build mappings when adding sections to output |
| Symbol resolution extremely slow | Using linear search in large symbol table | Profile code; if >1000 symbols, linear search is inefficient | Implement hash table for symbol lookup |


## Component: ELF Reader
> **Milestone(s):** Milestone 1: Section Merging

The **ELF Reader** is the foundational component of the static linker, responsible for loading and interpreting individual object files in ELF (Executable and Linkable Format). It acts as the input processor that transforms raw binary files into structured in-memory representations that downstream components can analyze and manipulate. This component must accurately parse the complex ELF format, extract all relevant metadata (sections, symbols, relocations), and present this information in a normalized form for subsequent linking stages.

### Mental Model: Library Catalog

Imagine walking into a large library where each book represents a compiled object file (`.o` file). The ELF Reader acts as the **library catalog system** that indexes every book's contents without modifying the books themselves. For each book, the catalog:

1. **Records the table of contents** (section headers) showing chapter locations and types
2. **Creates an index of important terms** (symbol table) marking where specific concepts are defined and referenced
3. **Notes cross-references between chapters** (relocations) that point to terms in other books
4. **Extracts the actual chapter text** (section data) for potential copying

Just as a library catalog allows you to find information across multiple books without rearranging the physical shelves, the ELF Reader enables the linker to analyze multiple object files simultaneously. It provides read-only access to each file's internal structure, creating the necessary foundation for the linker to decide how to merge these "books" into a single "compiled volume" (the executable).

### Interface Specification

The ELF Reader exposes a minimal API focused on loading object files and providing structured access to their contents. The primary interface consists of functions that parse ELF files and return normalized data structures.

**Table: ELF Reader Interface Methods**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `read_elf_file` | `filename: const char*` | `ObjectFile*` | Primary entry point that parses a single ELF object file from disk into an in-memory `ObjectFile` structure. Returns `NULL` on failure. |
| `free_object_file` | `obj: ObjectFile*` | `void` | Releases all memory associated with a previously loaded `ObjectFile`, including section data, symbol tables, and the structure itself. |
| `get_section_by_name` | `obj: ObjectFile*, name: const char*` | `ElfSection*` | Helper function (not required by downstream components) that retrieves a section by its name string using the section header string table. |
| `get_section_by_index` | `obj: ObjectFile*, index: uint16_t` | `ElfSection*` | Helper function to retrieve a section by its numerical index in the section header table. |

**Table: Key Data Structures Exposed by ELF Reader**

| Structure | Field | Type | Description |
|-----------|-------|------|-------------|
| `ObjectFile` | `filename` | `char*` | Path to the source file for debugging and error messages |
| | `header` | `Elf64_Ehdr` | Complete ELF file header with identification and layout metadata |
| | `sections` | `ElfSection*` | Array of parsed section structures (dynamically allocated) |
| | `num_sections` | `uint16_t` | Number of valid entries in the `sections` array |
| | `symbols` | `ElfSymbol*` | Array of symbol entries extracted from `.symtab` section |
| | `num_symbols` | `uint32_t` | Count of symbols in the `symbols` array |
| | `relocations` | `ElfRelocation*` | Array of relocation entries from `.rela` sections |
| | `num_relocations` | `uint32_t` | Count of relocation entries across all sections |
| | `shstrtab` | `char*` | Section header string table data (for section names) |
| | `strtab` | `char*` | Symbol string table data (for symbol names) |
| `ElfSection` | `header` | `Elf64_Shdr` | Raw ELF section header with type, flags, and layout information |
| | `name` | `char*` | Null-terminated section name (points into `shstrtab`) |
| | `data` | `uint8_t*` | Raw section contents (dynamically allocated copy) |
| | `data_size` | `size_t` | Size of allocated data buffer (may be 0 for `SHT_NOBITS`) |
| `ElfSymbol` | `sym` | `Elf64_Sym` | Raw ELF symbol entry with binding, type, and value fields |
| | `name` | `char*` | Null-terminated symbol name (points into `strtab`) |
| | `section` | `ElfSection*` | Pointer to the section where this symbol is defined (or `NULL` for undefined) |
| `ElfRelocation` | `rela` | `Elf64_Rela` | Raw relocation entry with offset, info, and addend fields |
| | `symbol` | `ElfSymbol*` | Pointer to the symbol this relocation references |
| | `target_section` | `ElfSection*` | Pointer to the section containing the relocation site (where patching occurs) |

The `ObjectFile` structure serves as the complete representation of a single input file, with all parsed data normalized and cross-linked for easy traversal. Downstream components receive arrays of `ObjectFile` structures and use the contained information to perform linking operations.

### Internal Behavior

The ELF Reader follows a deterministic multi-stage parsing process that transforms raw file bytes into structured data. This process must handle the ELF format's intricacies while providing robust error detection and clear failure messages.

**Numbered Parsing Algorithm:**

1. **File Opening and Validation**
   - Open the specified file in binary read mode
   - Read the first 64 bytes (minimum ELF header size) into a buffer
   - Verify the **ELF magic number** (`0x7F 'E' 'L' 'F'`) in `e_ident[0..3]`
   - Check that the file is **64-bit** (`ELFCLASS64` in `e_ident[EI_CLASS]`)
   - Verify **little-endian** format (`ELFDATA2LSB` in `e_ident[EI_DATA]`) for x86-64
   - Confirm current version (`EV_CURRENT` in `e_ident[EI_VERSION]`)
   - Validate file type is relocatable object file (`ET_REL` in `e_type`)
   - Confirm machine architecture is x86-64 (`EM_X86_64` in `e_machine`)

2. **Complete Header Parsing**
   - Read the remaining `Elf64_Ehdr` fields from the file
   - Validate critical offsets: `e_shoff` (section header table) must be non-zero
   - Check that `e_shnum` (section count) and `e_shentsize` (section header size) are reasonable
   - Verify `e_shstrndx` (section name string table index) points to a valid section

3. **Section Header Table Loading**
   - Seek to file offset `e_shoff`
   - Allocate array for `e_shnum` section headers
   - Read each `Elf64_Shdr` structure from the file
   - Load the **section header string table** (`shstrtab`) using index `e_shstrndx`
   - For each section header, extract the name by indexing into `shstrtab` using `sh_name`

4. **Section Data Extraction**
   - For each section where `sh_type != SHT_NOBITS` and `sh_size > 0`:
     - Allocate buffer of size `sh_size`
     - Seek to file offset `sh_offset`
     - Read `sh_size` bytes into the buffer
     - Store pointer in corresponding `ElfSection.data` field
   - For `SHT_NOBITS` sections (like `.bss`), set `data` to `NULL` and `data_size` to `sh_size`

5. **Symbol Table Processing**
   - Locate the `.symtab` section (type `SHT_SYMTAB`)
   - Load the **symbol string table** (`.strtab`) referenced by `sh_link` in the `.symtab` header
   - Calculate number of symbols: `section_size / sizeof(Elf64_Sym)`
   - For each symbol entry:
     - Extract name from string table using `st_name` offset
     - Determine binding (`STB_GLOBAL`, `STB_WEAK`, `STB_LOCAL`) from `st_info`
     - Determine type (`STT_NOTYPE`, `STT_OBJECT`, `STT_FUNC`, `STT_SECTION`) from `st_info`
     - Link symbol to its defining section using `st_shndx` (special values like `SHN_UNDEF` for undefined)

6. **Relocation Table Processing**
   - Iterate through sections with type `SHT_RELA`
   - For each relocation entry in the section:
     - Extract relocation type from `r_info` (e.g., `R_X86_64_PC32`, `R_X86_64_64`)
     - Extract symbol index from `r_info`
     - Look up the corresponding symbol from the symbol table
     - Identify the target section (where relocation applies) as the section containing the `.rela` section
     - Store cross-linked `ElfRelocation` entry

7. **Cross-Reference Resolution**
   - Link each symbol to its defining `ElfSection` (using `st_shndx`)
   - Link each relocation to its target section and referenced symbol
   - Validate that all internal references are consistent (no dangling pointers)

8. **Normalization and Cleanup**
   - Ensure all string pointers reference the appropriate string table data
   - Sort sections by type for easier downstream processing
   - Set up sentinel values for undefined symbols (`section = NULL`)
   - Return the fully populated `ObjectFile` structure

> **Key Insight:** The ELF Reader performs **read-only parsing**—it never modifies the original file content or the in-memory representations. This separation of concerns allows the linker to analyze multiple files simultaneously before committing to any merging decisions.

**Concrete Example Walkthrough:**

Consider parsing a simple object file `math.o` containing a function `square()` and a global variable `pi`. The ELF Reader would:

1. Validate the ELF magic and confirm 64-bit x86-64 format
2. Discover 8 sections: `.text`, `.data`, `.rodata`, `.bss`, `.symtab`, `.strtab`, `.shstrtab`, `.rela.text`
3. Extract `.text` section data containing `square()` machine code
4. Extract `.data` section data containing initialized `pi` value
5. Parse `.symtab` to find:
   - `square`: binding=`STB_GLOBAL`, type=`STT_FUNC`, section=`.text`
   - `pi`: binding=`STB_GLOBAL`, type=`STT_OBJECT`, section=`.data`
6. Parse `.rela.text` to find any relocations (none in this simple case)
7. Return an `ObjectFile` with all sections, symbols, and empty relocations array

### Architecture Decision: ELF Parsing Library vs Manual Parsing

> **Decision: Manual ELF Parsing for Educational Value**
> - **Context**: The linker needs to parse ELF object files to extract sections, symbols, and relocations. This could be implemented using existing ELF parsing libraries (like `libelf`) or through manual parsing of the ELF format.
> - **Options Considered**:
>   1. **Use existing ELF library** (e.g., `libelf`, `elf.h` with system loader)
>   2. **Manual parsing with standard I/O**
>   3. **Hybrid approach using system headers but manual data extraction**
> - **Decision**: Implement manual ELF parsing using only standard C file I/O and the ELF structure definitions from `elf.h`.
> - **Rationale**: 
>   - **Educational transparency**: Manual parsing forces understanding of ELF format internals, which is core to the learning objectives
>   - **Portability**: Avoiding external library dependencies simplifies build process and cross-platform compatibility
>   - **Control**: Direct access to raw structures enables custom optimizations and debugging output
>   - **Minimalism**: The linker only needs a subset of ELF features; full libraries add unnecessary complexity
> - **Consequences**:
>   - **Positive**: Deep understanding of ELF layout, easier debugging of parsing issues, no external dependencies
>   - **Negative**: More code to maintain, must handle edge cases manually, potential for bugs in format handling

**Table: Options Comparison**

| Option | Pros | Cons | Why Not Chosen |
|--------|------|------|----------------|
| Existing ELF library (`libelf`) | Robust, handles all edge cases, well-tested | Obscures learning, external dependency, potential license issues | Defeats educational purpose of understanding ELF internals |
| Manual parsing | Full control, educational, no dependencies | More code, must handle all edge cases manually | **CHOSEN**: Best aligns with learning objectives |
| Hybrid (system headers + manual I/O) | Some abstraction while maintaining control | Still requires significant manual work, inconsistent abstraction | Less coherent architecture than full manual approach |

The decision to implement manual parsing aligns with the project's educational goals. By working directly with the ELF structures, developers gain intimate knowledge of the format that will help them debug linking issues and understand toolchain behavior throughout their careers.

### Common Pitfalls

ELF parsing involves numerous subtle details that can lead to incorrect behavior if mishandled. Below are the most common pitfalls encountered when implementing the ELF Reader.

⚠️ **Pitfall: Endianness Assumptions**
- **Description**: Assuming the host system's endianness matches the ELF file's endianness.
- **Why it's wrong**: ELF files can be big-endian or little-endian. x86-64 uses little-endian, but the linker should verify this rather than assume.
- **Fix**: Check `e_ident[EI_DATA]` for `ELFDATA2LSB` (little-endian) and implement byte-swapping if `ELFDATA2MSB` (big-endian) is encountered.

⚠️ **Pitfall: Incorrect Section Alignment**
- **Description**: Treating section file offsets as directly usable memory addresses.
- **Why it's wrong**: Section data in object files may not be aligned to their `sh_addralign` requirements; alignment is enforced at load time, not in the file.
- **Fix**: Use `sh_offset` for file reading, but track `sh_addralign` for later merging. Don't assume alignment in the object file.

⚠️ **Pitfall: Mishandling SHT_NOBITS Sections**
- **Description**: Attempting to read file content for `.bss` sections (type `SHT_NOBITS`).
- **Why it's wrong**: `SHT_NOBITS` sections occupy memory but no file space. `sh_offset` may be zero or point to unrelated data.
- **Fix**: Check `sh_type == SHT_NOBITS` and skip file reading; allocate zero-filled buffer in memory if needed.

⚠️ **Pitfall: String Table Index Errors**
- **Description**: Incorrectly calculating string offsets in `shstrtab` and `.strtab`.
- **Why it's wrong**: String table offsets are zero-based from the start of the table, not from the current position.
- **Fix**: Validate offset < table size, then compute `table_base + offset`. Handle zero offset (null string) specially.

⚠️ **Pitfall: Symbol Table Entry Misalignment**
- **Description**: Assuming symbol table entries are tightly packed without considering `sh_entsize`.
- **Why it's wrong**: While typically `sizeof(Elf64_Sym)`, the entry size field allows for format extensions.
- **Fix**: Use `sh_entsize` to calculate entry count and stride through the table.

⚠️ **Pitfall: Relocation Symbol Index Confusion**
- **Description**: Misinterpreting the `r_info` field for symbol index extraction.
- **Why it's wrong**: On x86-64, `r_info` contains both type (low 32 bits) and symbol index (high 32 bits), not a simple offset.
- **Fix**: Use `ELF64_R_SYM(r_info)` macro (or manual shift) to extract symbol index, and `ELF64_R_TYPE(r_info)` for type.

⚠️ **Pitfall: Header Version Mismatch**
- **Description**: Ignoring version fields in ELF headers.
- **Why it's wrong**: Future ELF versions may change format; ignoring version risks incompatibility.
- **Fix**: Check `e_ident[EI_VERSION]` and `e_version` against `EV_CURRENT`, rejecting unsupported versions.

### Implementation Guidance

This section provides concrete implementation advice for building the ELF Reader component in C, following the naming conventions and structure established in the design.

#### A. Technology Recommendations Table

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| File I/O | Standard `fopen`/`fread` with buffering | Memory-mapped files for large objects |
| Endianness | Assume little-endian (x86-64) | Runtime endianness detection and conversion |
| Error Handling | `fatal_error` macro with `__FILE__`/`__LINE__` | Structured error codes with recovery paths |
| Data Structures | Simple arrays with linear search | Hash tables for symbol name lookup |

#### B. Recommended File/Module Structure

```
static-linker/
├── include/
│   ├── elf_reader.h           # Public interface for ELF Reader
│   └── elf_definitions.h      # ELF constants and type definitions
├── src/
│   ├── main.c                 # Linker entry point (future)
│   ├── elf_reader.c           # ELF Reader implementation
│   ├── util.c                 # Utilities (xmalloc, align_to, etc.)
│   └── error.c                # Error reporting (fatal_error)
└── test/
    ├── test_elf_reader.c      # Unit tests for ELF parsing
    └── test_files/            # Sample .o files for testing
```

#### C. Infrastructure Starter Code

**File: `include/elf_definitions.h`** - Complete ELF constant definitions:

```c
#ifndef ELF_DEFINITIONS_H
#define ELF_DEFINITIONS_H

#include <stdint.h>

/* ELF identification indices */
#define EI_NIDENT 16
#define EI_MAG0 0
#define EI_MAG1 1
#define EI_MAG2 2
#define EI_MAG3 3
#define EI_CLASS 4
#define EI_DATA 5
#define EI_VERSION 6

/* ELF magic numbers */
#define ELFMAG0 0x7F
#define ELFMAG1 'E'
#define ELFMAG2 'L'
#define ELFMAG3 'F'

/* ELF class */
#define ELFCLASSNONE 0
#define ELFCLASS32 1
#define ELFCLASS64 2

/* ELF data encoding */
#define ELFDATANONE 0
#define ELFDATA2LSB 1
#define ELFDATA2MSB 2

/* ELF version */
#define EV_NONE 0
#define EV_CURRENT 1

/* ELF file types */
#define ET_NONE 0
#define ET_REL 1
#define ET_EXEC 2
#define ET_DYN 3
#define ET_CORE 4

/* Machine types */
#define EM_NONE 0
#define EM_X86_64 62

/* Section types */
#define SHT_NULL 0
#define SHT_PROGBITS 1
#define SHT_SYMTAB 2
#define SHT_STRTAB 3
#define SHT_RELA 4
#define SHT_NOBITS 8

/* Section flags */
#define SHF_WRITE 0x1
#define SHF_ALLOC 0x2
#define SHF_EXECINSTR 0x4

/* Symbol binding */
#define STB_LOCAL 0
#define STB_GLOBAL 1
#define STB_WEAK 2

/* Symbol type */
#define STT_NOTYPE 0
#define STT_OBJECT 1
#define STT_FUNC 2
#define STT_SECTION 3

/* Special section indices */
#define SHN_UNDEF 0
#define SHN_ABS 0xfff1
#define SHN_COMMON 0xfff2

/* Relocation types for x86-64 */
#define R_X86_64_NONE 0
#define R_X86_64_64 1
#define R_X86_64_PC32 2

/* ELF64 type definitions (simplified from standard elf.h) */
typedef struct {
    unsigned char e_ident[EI_NIDENT];
    uint16_t e_type;
    uint16_t e_machine;
    uint32_t e_version;
    uint64_t e_entry;
    uint64_t e_phoff;
    uint64_t e_shoff;
    uint32_t e_flags;
    uint16_t e_ehsize;
    uint16_t e_phentsize;
    uint16_t e_phnum;
    uint16_t e_shentsize;
    uint16_t e_shnum;
    uint16_t e_shstrndx;
} Elf64_Ehdr;

typedef struct {
    uint32_t sh_name;
    uint32_t sh_type;
    uint64_t sh_flags;
    uint64_t sh_addr;
    uint64_t sh_offset;
    uint64_t sh_size;
    uint32_t sh_link;
    uint32_t sh_info;
    uint64_t sh_addralign;
    uint64_t sh_entsize;
} Elf64_Shdr;

typedef struct {
    uint32_t st_name;
    unsigned char st_info;
    unsigned char st_other;
    uint16_t st_shndx;
    uint64_t st_value;
    uint64_t st_size;
} Elf64_Sym;

typedef struct {
    uint64_t r_offset;
    uint64_t r_info;
    int64_t r_addend;
} Elf64_Rela;

/* Helper macros for relocation info */
#define ELF64_R_SYM(i) ((i) >> 32)
#define ELF64_R_TYPE(i) ((i) & 0xffffffff)

#endif /* ELF_DEFINITIONS_H */
```

**File: `src/util.c`** - Essential utility functions:

```c
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "elf_definitions.h"

void* xmalloc(size_t size) {
    void* ptr = malloc(size);
    if (!ptr && size > 0) {
        fprintf(stderr, "Fatal: memory allocation failed for %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void* xcalloc(size_t count, size_t size) {
    void* ptr = calloc(count, size);
    if (!ptr && count > 0 && size > 0) {
        fprintf(stderr, "Fatal: calloc failed for %zu elements of %zu bytes\n", count, size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void* xrealloc(void* ptr, size_t size) {
    void* new_ptr = realloc(ptr, size);
    if (!new_ptr && size > 0) {
        fprintf(stderr, "Fatal: realloc failed for %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return new_ptr;
}

uint64_t align_to(uint64_t value, uint64_t alignment) {
    if (alignment == 0) return value;
    uint64_t remainder = value % alignment;
    if (remainder == 0) return value;
    return value + (alignment - remainder);
}

void fatal_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "Error: ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
    exit(EXIT_FAILURE);
}
```

#### D. Core Logic Skeleton Code

**File: `include/elf_reader.h`** - Public interface:

```c
#ifndef ELF_READER_H
#define ELF_READER_H

#include "elf_definitions.h"

/* Forward declarations of internal structures */
typedef struct ElfSection ElfSection;
typedef struct ElfSymbol ElfSymbol;
typedef struct ElfRelocation ElfRelocation;
typedef struct ObjectFile ObjectFile;

/* Public API */
ObjectFile* read_elf_file(const char* filename);
void free_object_file(ObjectFile* obj);

/* Accessor helpers (for debugging) */
const ElfSection* get_section_by_name(const ObjectFile* obj, const char* name);
const ElfSection* get_section_by_index(const ObjectFile* obj, uint16_t index);

#endif /* ELF_READER_H */
```

**File: `src/elf_reader.c`** - Implementation skeleton with TODOs:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "elf_reader.h"
#include "elf_definitions.h"

/* Internal structure definitions */
struct ElfSection {
    Elf64_Shdr header;
    char* name;
    uint8_t* data;
    size_t data_size;
};

struct ElfSymbol {
    Elf64_Sym sym;
    char* name;
    ElfSection* section;
};

struct ElfRelocation {
    Elf64_Rela rela;
    ElfSymbol* symbol;
    ElfSection* target_section;
};

struct ObjectFile {
    char* filename;
    Elf64_Ehdr header;
    ElfSection* sections;
    uint16_t num_sections;
    ElfSymbol* symbols;
    uint32_t num_symbols;
    ElfRelocation* relocations;
    uint32_t num_relocations;
    char* shstrtab;
    char* strtab;
};

/* Helper function declarations */
static int validate_elf_header(const Elf64_Ehdr* header);
static char* load_string_table(FILE* file, const Elf64_Shdr* shdr);
static ElfSection* load_section(FILE* file, const Elf64_Shdr* shdr, const char* name);
static void load_symbols(ObjectFile* obj, FILE* file, const ElfSection* symtab_section);
static void load_relocations(ObjectFile* obj, FILE* file, const ElfSection* rela_section);

ObjectFile* read_elf_file(const char* filename) {
    // TODO 1: Open file in binary mode with error checking
    // TODO 2: Read ELF header and validate with validate_elf_header()
    // TODO 3: Allocate and initialize ObjectFile structure
    // TODO 4: Load section header table (array of Elf64_Shdr)
    // TODO 5: Load section header string table (shstrtab)
    // TODO 6: For each section header:
    //   a) Extract name from shstrtab using sh_name offset
    //   b) Load section data with load_section() (skip for SHT_NOBITS)
    // TODO 7: Locate and load symbol table (.symtab) with load_symbols()
    // TODO 8: Locate and load relocation tables (.rela.*) with load_relocations()
    // TODO 9: Cross-link symbols to their sections using st_shndx
    // TODO 10: Cross-link relocations to symbols and target sections
    // TODO 11: Close file and return populated ObjectFile*
    return NULL; // Placeholder
}

void free_object_file(ObjectFile* obj) {
    if (!obj) return;
    // TODO 1: Free filename string if allocated
    // TODO 2: For each section: free name and data if allocated
    // TODO 3: Free sections array
    // TODO 4: For each symbol: free name if allocated
    // TODO 5: Free symbols array
    // TODO 6: Free relocations array
    // TODO 7: Free shstrtab and strtab if allocated
    // TODO 8: Free ObjectFile structure itself
}

static int validate_elf_header(const Elf64_Ehdr* header) {
    // TODO 1: Check ELF magic bytes (0x7F, 'E', 'L', 'F')
    // TODO 2: Verify ELFCLASS64 (64-bit format)
    // TODO 3: Verify ELFDATA2LSB (little-endian for x86-64)
    // TODO 4: Check version is EV_CURRENT
    // TODO 5: Verify file type is ET_REL (relocatable object)
    // TODO 6: Verify machine is EM_X86_64
    // TODO 7: Validate section header offset and counts are reasonable
    // TODO 8: Return 1 if valid, 0 otherwise
    return 0; // Placeholder
}

static char* load_string_table(FILE* file, const Elf64_Shdr* shdr) {
    // TODO 1: Check for valid section (type SHT_STRTAB, size > 0)
    // TODO 2: Allocate buffer of shdr->sh_size bytes
    // TODO 3: Seek to shdr->sh_offset in file
    // TODO 4: Read shdr->sh_size bytes into buffer
    // TODO 5: Ensure null termination at end of buffer
    // TODO 6: Return buffer (caller must free)
    return NULL; // Placeholder
}

static ElfSection* load_section(FILE* file, const Elf64_Shdr* shdr, const char* name) {
    // TODO 1: Allocate ElfSection structure
    // TODO 2: Copy header and name (duplicate string)
    // TODO 3: If sh_type != SHT_NOBITS and sh_size > 0:
    //   a) Allocate data buffer of sh_size bytes
    //   b) Seek to sh_offset in file
    //   c) Read sh_size bytes into buffer
    // TODO 4: For SHT_NOBITS: set data to NULL, data_size to sh_size
    // TODO 5: Return populated ElfSection*
    return NULL; // Placeholder
}

static void load_symbols(ObjectFile* obj, FILE* file, const ElfSection* symtab_section) {
    // TODO 1: Verify section type is SHT_SYMTAB
    // TODO 2: Calculate symbol count: section_size / sh_entsize (or sizeof(Elf64_Sym))
    // TODO 3: Allocate symbols array of appropriate size
    // TODO 4: Load string table using sh_link index
    // TODO 5: For each symbol entry in symtab_section->data:
    //   a) Parse Elf64_Sym from raw bytes
    //   b) Extract name from string table using st_name
    //   c) Store in symbols array (section pointer will be filled later)
    // TODO 6: Update obj->symbols and obj->num_symbols
}

static void load_relocations(ObjectFile* obj, FILE* file, const ElfSection* rela_section) {
    // TODO 1: Verify section type is SHT_RELA
    // TODO 2: Calculate relocation count: section_size / sizeof(Elf64_Rela)
    // TODO 3: Allocate relocations array
    // TODO 4: For each Elf64_Rela in section data:
    //   a) Extract symbol index using ELF64_R_SYM(rela.r_info)
    //   b) Look up symbol in obj->symbols array (validate index)
    //   c) Determine target section (section containing this .rela section)
    //   d) Store populated ElfRelocation in array
    // TODO 5: Update obj->relocations and obj->num_relocations
}

/* Public helper functions */
const ElfSection* get_section_by_name(const ObjectFile* obj, const char* name) {
    // TODO: Linear search through sections array comparing names
    return NULL;
}

const ElfSection* get_section_by_index(const ObjectFile* obj, uint16_t index) {
    // TODO: Bounds check and return section at index
    return NULL;
}
```

#### E. Language-Specific Hints

1. **File I/O**: Use `fopen` with `"rb"` mode for binary reading. Always check return values and use `fseek`/`ftell` for positioning.

2. **Memory Management**: Follow the pattern: allocate with `xmalloc`/`xcalloc`, free in reverse order in `free_object_file`. Use `size_t` for sizes to avoid overflow.

3. **Structure Packing**: ELF structures are tightly packed with 1-byte alignment. Use `#pragma pack(1)` or compiler attributes if your compiler adds padding.

4. **Endianness**: On x86-64, host is little-endian matching ELF format. For portability, use `ntohll`/`htonll` for 64-bit values when reading.

5. **String Handling**: ELF strings are null-terminated but tables contain multiple strings. Always validate offsets before accessing.

#### F. Milestone Checkpoint

**After implementing ELF Reader**, you should be able to:

1. **Test with a simple object file:**
   ```bash
   # Create a test C file
   echo "int global_var = 42; int main() { return global_var; }" > test.c
   # Compile to object file
   gcc -c test.c -o test.o
   # Run your linker's ELF reader (assuming you have a test program)
   ./linker_test test.o
   ```

2. **Expected behavior:**
   - Program should parse `test.o` without errors
   - Should report section count (at least `.text`, `.data`, `.symtab`, `.strtab`, `.shstrtab`)
   - Should list symbols including `global_var` and `main`
   - Should show section sizes and types

3. **Verification with standard tools:**
   ```bash
   # Compare your parser's output with readelf
   readelf -h test.o        # Compare header fields
   readelf -S test.o        # Compare sections
   readelf -s test.o        # Compare symbols
   ```

4. **Signs of problems:**
   - **Crash on startup**: Likely file opening or header validation issue
   - **Wrong section count**: `e_shnum` parsing error or endianness issue
   - **Garbage symbol names**: String table offset calculation wrong
   - **Missing relocations**: `.rela` section parsing not implemented

#### G. Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| "Invalid ELF magic" error on valid .o file | Wrong byte order in magic check | Hexdump first 4 bytes: `hexdump -C test.o \| head -1` | Check byte-by-byte: `0x7F 'E' 'L' 'F'` |
| Section names are garbage | Wrong `shstrtab` loading or offset | Print `shstrtab` content, check `e_shstrndx` value | Verify `sh_name` is offset from table start, not absolute |
| Symbols point to wrong sections | `st_shndx` misinterpretation | Compare with `readelf -s`: section indices differ | Handle special indices (`SHN_UNDEF`, `SHN_ABS`) correctly |
| Relocation count is zero | Missing `.rela` section parsing | Check if section type `SHT_RELA` is being recognized | Implement `load_relocations` and call it for all `SHT_RELA` sections |
| Memory leak on repeated parsing | `free_object_file` not implemented | Use `valgrind ./linker_test test.o` | Implement complete cleanup in `free_object_file` |
| Crash when accessing symbol name | String offset out of bounds | Check `st_name < strtab_size` before accessing | Add bounds checking in symbol loading |


## Component: Section Merger

> **Milestone(s):** Milestone 1: Section Merging

The **Section Merger** is the architectural component responsible for transforming the fragmented memory layout of multiple input object files into a unified, contiguous layout suitable for a single executable. Its core challenge lies in concatenating similar sections from different compilation units while respecting hardware alignment requirements and maintaining accurate address mappings for subsequent relocation processing. This component serves as the bridge between the raw ELF file parsing performed by the ELF Reader and the symbolic resolution that follows, establishing the physical and virtual memory foundation upon which the entire executable is built.

### Mental Model: Warehouse Consolidation

Imagine you are managing inventory across several small warehouses (object files), each containing boxes (sections) of different types: machinery parts (`.text`), raw materials (`.data`), finished goods (`.rodata`), and empty slots reserved for future stock (`.bss`). Your task is to consolidate all inventory into one large, efficient warehouse (the output executable). You cannot simply dump boxes together; you must:

1. **Group similar items**: All machinery parts from every warehouse go into one designated area, all raw materials into another, and so on.
2. **Reserve aisle space**: Some boxes require specific alignment—heavy machinery might need to be placed on 8-foot centers. You must insert empty padding between boxes from different warehouses to meet these requirements.
3. **Maintain a manifest**: As you place each incoming box, you record its new location (output offset) in the consolidated warehouse. This manifest (`InputSectionMapping`) is crucial for later operations, like relabeling the contents of the boxes.
4. **Handle special cases**: Empty slots (`.bss`) reserve space in the warehouse but don't occupy physical shelf space in the shipping container (file); they only exist in the warehouse's floor plan (virtual memory).

This warehouse consolidation mental model captures the essence of section merging: logical grouping, alignment-aware spacing, and meticulous tracking of input-to-output location mappings.

### Interface Specification

The Section Merger exposes a clean, functional interface focused on layout computation and data aggregation. Its primary consumer is the linking driver that orchestrates the overall pipeline, and its outputs feed directly into the Symbol Resolver and Relocation Applier.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `create_merged_sections` | `void` | `MergedSections*` | Allocates and initializes an empty `MergedSections` structure to hold the output layout and mappings. |
| `add_sections_from_object` | `MergedSections* merged`, `const ObjectFile* obj`, `uint32_t file_index` | `void` | Processes all allocatable sections from a single `ObjectFile`, adding them to the appropriate output section groups based on type/name/flags. Updates internal grouping tables. |
| `compute_merged_layout` | `MergedSections* merged` | `void` | Finalizes the layout: computes final sizes, applies alignment padding between input sections, assigns output file offsets and virtual addresses. Builds the `OutputSection` array and `InputSectionMapping` table. |
| `get_output_section_data` | `const MergedSections* merged`, `uint32_t out_sect_index`, `uint8_t** data`, `size_t* size` | `void` | Retrieves the concatenated raw byte data for a specific output section after layout. The caller must free the data. |
| `free_merged_sections` | `MergedSections* merged` | `void` | Deallocates all memory owned by the `MergedSections` structure, including internal arrays and section data. |

The key data structures produced by this component are:

**`OutputSection` (Per final output section):**
| Field | Type | Description |
|-------|------|-------------|
| `name` | `char*` | Name of the output section (e.g., ".text", ".data"). Derived from the common name of merged input sections. |
| `sh_type` | `uint64_t` | ELF section type (e.g., `SHT_PROGBITS`, `SHT_NOBITS`). Must be consistent across all merged inputs. |
| `sh_flags` | `uint64_t` | Section flags (e.g., `SHF_ALLOC | SHF_EXECINSTR`). Must be consistent across all merged inputs. |
| `sh_addralign` | `uint64_t` | Alignment requirement (maximum alignment among merged input sections). |
| `file_offset` | `uint64_t` | Starting byte offset of this section within the output ELF file. |
| `virtual_addr` | `uint64_t` | Starting virtual address for this section when loaded into memory. |
| `size` | `uint64_t` | Total size in bytes of this output section (including all input data and padding). |
| `data` | `uint8_t*` | Pointer to the concatenated raw byte data (for `SHT_PROGBITS` sections). `NULL` for `SHT_NOBITS` (`.bss`). |

**`InputSectionMapping` (Per input section contributed to output):**
| Field | Type | Description |
|-------|------|-------------|
| `file_index` | `uint32_t` | Index of the source object file within the input array. |
| `section_index` | `uint32_t` | Index of the section within that object file's section table. |
| `output_offset` | `uint64_t` | Byte offset *within its output section* where this input section's data begins (after any preceding padding). |

The `MergedSections` container holds arrays of these structures:
- `sections`: `OutputSection*` – Array of final output sections.
- `num_sections`: `uint32_t` – Count of output sections.
- `mappings`: `InputSectionMapping*` – Array mapping every input section to its output location.
- `num_mappings`: `uint32_t` – Count of input sections processed.

### Internal Behavior

The Section Merger operates in three distinct phases: **Collection**, **Layout Computation**, and **Data Assembly**. This separation ensures all alignment constraints are known before any addresses are assigned.

#### Phase 1: Collection and Grouping

For each input object file (via `add_sections_from_object`):

1. **Filter allocatable sections**: Iterate through the object's sections. Only process sections with the `SHF_ALLOC` flag set (`.text`, `.data`, `.bss`, `.rodata`). Ignore debug sections, string tables, and symbol tables.
2. **Create section signature**: For each allocatable section, generate a grouping key based on:
   - Section type (`sh_type`)
   - Section flags (`sh_flags`)
   - Section name (for special handling: `.text`, `.data`, `.bss`, `.rodata` get separate groups)
3. **Add to group**: Place the input section reference (file index, section index) into an internal table keyed by this signature. Maintain the original order of sections within each group as they appear in input files (important for linker determinism).

#### Phase 2: Layout Computation (via `compute_merged_layout`)

1. **Initialize output sections**: For each unique grouping signature, create a corresponding `OutputSection` prototype. Set its `name`, `sh_type`, `sh_flags` from the signature. Compute `sh_addralign` as the maximum alignment value among all input sections in the group.
2. **Assign output section order**: Determine the final order of output sections in the executable. A typical layout order is: `.text` (executable code), `.rodata` (read-only data), `.data` (initialized writable data), `.bss` (uninitialized writable data). This order respects typical virtual memory segment groupings (text segment: RX, data segment: RW).
3. **Compute offsets within each output section**:
   - Set current offset `offset = 0`.
   - For each input section in the group (in the order collected):
     - Calculate padding: `padding = align_to(offset, input_section.sh_addralign) - offset`.
     - Increment `offset += padding`.
     - Record mapping: Create an `InputSectionMapping` entry with `output_offset = offset`.
     - Increment `offset += input_section.sh_size` (note: for `SHT_NOBITS` sections, this affects virtual size but not file offset).
   - Set the output section's total `size = offset`.
4. **Assign file offsets and virtual addresses**:
   - Set `current_file_offset = size_of_headers` (estimated; final adjustment occurs in Executable Writer).
   - Set `current_virtual_addr = 0x400000` (typical Linux x86_64 executable base address).
   - For each output section in layout order:
     - Align `current_file_offset` to the section's `sh_addralign` (for file representation).
     - Align `current_virtual_addr` to the section's `sh_addralign` (for memory representation).
     - Set section's `file_offset = current_file_offset`.
     - Set section's `virtual_addr = current_virtual_addr`.
     - Increment `current_file_offset += section.size` (but only add file size for `SHT_PROGBITS`; `SHT_NOBITS` sections occupy zero file space).
     - Increment `current_virtual_addr += section.size`.

![Section Merging Layout Diagram](./diagrams/section-merging-layout.svg)

#### Phase 3: Data Assembly (via `get_output_section_data`)

When requested (typically by the Executable Writer), the merger assembles the raw byte content for each output section:

1. Allocate a buffer of size `output_section.size`.
2. For each input section mapped to this output section (in order):
   - Calculate target pointer: `buffer + mapping.output_offset`.
   - If input section is `SHT_PROGBITS`: copy `input_section.data` (size `input_section.sh_size`) to target.
   - If input section is `SHT_NOBITS`: no copy; the region in buffer remains uninitialized (typically zero-filled later).
3. Return the buffer.

### Architecture Decision: Lazy vs Eager Section Processing

> **Decision: Two-Pass Layout-First Processing**
>
> - **Context**: The linker must combine sections from multiple object files, each with alignment requirements. The final layout (file offsets, virtual addresses) depends on the concatenation order and padding between sections. We need to know the final size and position of every section before we can resolve symbol addresses or apply relocations, which depend on those positions.
> - **Options Considered**:
>   1. **Eager (single-pass streaming)**: Copy section data immediately as files are read, inserting padding on-the-fly. Assign addresses incrementally.
>   2. **Lazy (two-pass layout-first)**: First, collect all section metadata (size, alignment). Compute complete layout with padding. Then, in a second pass, copy data into the final buffer using the computed offsets.
> - **Decision**: We chose **Lazy two-pass layout-first processing**.
> - **Rationale**:
>   - **Symbol resolution dependency**: The Symbol Resolver needs to know the final base address of each output section to compute absolute symbol addresses. With eager processing, symbol addresses would be computed incrementally and might change if later sections require unexpected padding due to alignment, causing cascading recalculations.
>   - **Relocation accuracy**: Relocation entries refer to offsets within input sections. Applying relocations requires knowing the final output location of the target symbol *and* the relocation site. With a complete layout map, both can be computed directly.
>   - **Simplicity and determinism**: Two-pass processing separates concerns cleanly: layout computation is a pure function of section metadata; data assembly is a simple copy using the layout. This avoids complex state management and backtracking.
> - **Consequences**:
>   - **Memory overhead**: The linker must hold section metadata for all input files simultaneously before computing layout. For typical project sizes, this is negligible.
>   - **Performance**: Requires an extra pass over section data, but the copying pass is linear and cache-friendly. The simplicity outweighs any minor performance cost for an educational linker.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| **Eager (single-pass)** | - Potentially lower memory footprint (can stream data)<br>- Single pass over input files | - Complex address management (addresses may shift)<br>- Difficult to handle alignment padding optimally<br>- Harder to debug layout issues | No |
| **Lazy (two-pass)** | - Clear separation of layout and data<br>- Deterministic, reproducible layout<br>- Easy to compute symbol addresses after layout<br>- Simplified relocation processing | - Requires storing section metadata for all files<br>- Extra pass over data for copying | **Yes** |

### Common Pitfalls

⚠️ **Pitfall: Incorrect Alignment Padding**
- **Description**: Forgetting to insert padding between input sections when their alignment requirements are not met by the current offset. For example, a `.text` section with `sh_addralign=16` placed at file offset 100 would incorrectly start at 100 instead of 112 (since `align_to(100, 16) = 112`).
- **Why it's wrong**: The processor or OS loader may crash when trying to load misaligned sections, as some architectures require specific alignment for code or data (e.g., SSE instructions require 16-byte alignment). Even on x86_64, misalignment can cause severe performance penalties.
- **How to fix**: Always compute padding before placing each input section: `padding = align_to(current_offset, section_alignment) - current_offset`. Use the `align_to` helper function consistently.

⚠️ **Pitfall: Mishandling .bss (SHT_NOBITS) Sections**
- **Description**: Treating `.bss` sections as if they contain file data (copying garbage bytes) or forgetting to allocate virtual address space for them.
- **Why it's wrong**: `.bss` sections represent uninitialized data (e.g., `int global_array[1000];`). They occupy zero bytes in the object file (`sh_size` indicates memory size, not file size) but must reserve virtual address space when loaded. If omitted, symbols in `.bss` will point to invalid addresses.
- **How to fix**: During layout, include `.bss` input sections in size calculation for virtual address space (`current_virtual_addr += size`). During file offset calculation, skip adding their size to file offset (`current_file_offset += 0`). During data assembly, do not copy any data for `SHT_NOBITS` sections.

⚠️ **Pitfall: Mixing Incompatible Section Flags**
- **Description**: Attempting to merge sections with different flags, such as combining a writable `.data` section (`SHF_WRITE`) with a read-only `.rodata` section (no `SHF_WRITE`).
- **Why it's wrong**: The merged section would have ambiguous memory permissions. The OS loader sets page protections based on section flags; mixing flags could result in incorrectly marking read-only data as writable (security risk) or executable code as non-executable (crash).
- **How to fix**: Use the section signature (type + flags + name) as the grouping key. Sections with different `sh_flags` will form separate output sections automatically. Add validation to reject merging if flags differ within a group.

⚠️ **Pitfall: Forgetting to Map Input Sections**
- **Description**: Failing to create `InputSectionMapping` entries that record where each input section lands in the output.
- **Why it's wrong**: The Relocation Applier needs to know the output location of every symbol, which is computed as `output_section.virtual_addr + mapping.output_offset + symbol_offset`. Without mappings, relocations cannot be applied correctly.
- **How to fix**: Create a mapping entry for every input section added to a group, recording the `output_offset` within its output section. Store these in the `MergedSections` structure for later use.

### Implementation Guidance

#### A. Technology Recommendations Table

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| **Section Data Storage** | `uint8_t*` dynamic arrays per output section | Memory-mapped regions for large sections |
| **Grouping Data Structure** | Array of linked lists (one per output section type) | Hash table keyed by (type, flags, name) for O(1) lookup |
| **Alignment Helper** | Custom `align_to` function using bitwise operations | Platform-specific `posix_memalign` for actual allocations |

#### B. Recommended File/Module Structure

Extend the project structure established in the ELF Reader component:

```
static-linker/
├── include/
│   ├── linker.h          # Public API, type declarations
│   └── internal.h        # Internal structures, helper prototypes
├── src/
│   ├── main.c            # Driver, command-line interface
│   ├── elf_reader.c      # ELF parsing (previous component)
│   ├── section_merger.c  # This component ← NEW
│   ├── symbol_resolver.c # (Milestone 2)
│   ├── relocation.c      # (Milestone 3)
│   ├── executable.c      # (Milestone 4)
│   └── util.c            # Shared utilities (xmalloc, align_to, etc.)
└── tests/
    ├── test_merger.c     # Unit tests for section merging
    └── test_programs/    # Simple .c files for integration tests
```

#### C. Infrastructure Starter Code

The following utility functions are prerequisites used by the Section Merger:

```c
/* src/util.c */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "internal.h"

void* xmalloc(size_t size) {
    void* ptr = malloc(size);
    if (!ptr) {
        fatal_error("Out of memory: malloc failed for %zu bytes\n", size);
    }
    return ptr;
}

void* xcalloc(size_t count, size_t size) {
    void* ptr = calloc(count, size);
    if (!ptr) {
        fatal_error("Out of memory: calloc failed for %zu bytes\n", count * size);
    }
    return ptr;
}

void* xrealloc(void* ptr, size_t size) {
    void* new_ptr = realloc(ptr, size);
    if (!new_ptr && size > 0) {
        fatal_error("Out of memory: realloc failed for %zu bytes\n", size);
    }
    return new_ptr;
}

uint64_t align_to(uint64_t value, uint64_t alignment) {
    if (alignment == 0) return value;
    uint64_t remainder = value % alignment;
    if (remainder == 0) return value;
    return value + (alignment - remainder);
}

void fatal_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "linker error: ");
    vfprintf(stderr, fmt, args);
    va_end(args);
    exit(1);
}
```

#### D. Core Logic Skeleton Code

```c
/* src/section_merger.c */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "internal.h"

/* Internal grouping structure: holds list of input sections for one output section */
typedef struct InputSectionRef {
    uint32_t file_idx;
    uint32_t sect_idx;
    struct InputSectionRef* next;
} InputSectionRef;

typedef struct OutputSectionGroup {
    char* name;
    uint64_t sh_type;
    uint64_t sh_flags;
    uint64_t sh_addralign;  /* Max alignment in group */
    InputSectionRef* first_input;
    InputSectionRef* last_input;
    uint32_t input_count;
} OutputSectionGroup;

/* Internal state of the merger during collection phase */
typedef struct MergerState {
    OutputSectionGroup* groups;
    uint32_t groups_capacity;
    uint32_t groups_count;
    uint32_t total_input_sections;
} MergerState;

MergedSections* create_merged_sections(void) {
    MergedSections* merged = xcalloc(1, sizeof(MergedSections));
    merged->sections = NULL;
    merged->num_sections = 0;
    merged->mappings = NULL;
    merged->num_mappings = 0;
    return merged;
}

void free_merged_sections(MergedSections* merged) {
    if (!merged) return;
    for (uint32_t i = 0; i < merged->num_sections; i++) {
        free(merged->sections[i].name);
        free(merged->sections[i].data);
    }
    free(merged->sections);
    free(merged->mappings);
    free(merged);
}

void add_sections_from_object(MergedSections* merged, const ObjectFile* obj, uint32_t file_index) {
    /* TODO 1: Initialize MergerState on first call (use static or attach to MergedSections) */
    /* TODO 2: Iterate through obj->sections (0..obj->num_sections-1) */
    /* TODO 3: For each section with SHF_ALLOC flag set:
        - Skip sections with type SHT_SYMTAB, SHT_STRTAB, SHT_RELA (non-allocatable)
        - Determine grouping key: combination of sh_type, sh_flags, and name
        - For simplicity, use predefined groups: .text, .rodata, .data, .bss
          (Map section name to group: if name is ".text", ".text.*", etc., use .text group)
        - Find or create OutputSectionGroup for this key
        - Create InputSectionRef with file_idx=file_index, sect_idx=section_index
        - Append to group's linked list, update group's max alignment
        - Increment total_input_sections counter
    */
    /* TODO 4: Store MergerState back into MergedSections for use in compute_merged_layout */
}

void compute_merged_layout(MergedSections* merged) {
    /* TODO 1: Retrieve internal MergerState (groups, total_input_sections) */
    /* TODO 2: Allocate merged->sections array (one per OutputSectionGroup) */
    /* TODO 3: Allocate merged->mappings array (size = total_input_sections) */
    /* TODO 4: For each group (in order: .text, .rodata, .data, .bss):
        - Create OutputSection: set name, sh_type, sh_flags, sh_addralign
        - Compute layout within this output section:
            * current_offset = 0
            * For each InputSectionRef in group (in order of addition):
                - padding = align_to(current_offset, input_section.sh_addralign) - current_offset
                - current_offset += padding
                - Create InputSectionMapping entry:
                    file_index = ref->file_idx
                    section_index = ref->sect_idx
                    output_offset = current_offset
                - current_offset += input_section.sh_size
        - Set output section size = current_offset
    */
    /* TODO 5: Assign file offsets and virtual addresses to output sections:
        - file_offset = size_of_headers (estimate, e.g., 4096 for now)
        - virtual_addr = 0x400000 (typical base address)
        - For each output section in order:
            * Align file_offset and virtual_addr to section's sh_addralign
            * section.file_offset = file_offset
            * section.virtual_addr = virtual_addr
            * file_offset += (section.sh_type == SHT_NOBITS) ? 0 : section.size
            * virtual_addr += section.size
    */
}

void get_output_section_data(const MergedSections* merged, uint32_t out_sect_index,
                             uint8_t** data, size_t* size) {
    /* TODO 1: Validate out_sect_index */
    /* TODO 2: Allocate buffer of size merged->sections[out_sect_index].size */
    /* TODO 3: Zero-initialize buffer (important for padding regions) */
    /* TODO 4: Iterate through all InputSectionMapping entries that belong to this output section:
        - For each mapping where the input section is SHT_PROGBITS:
            * source = input_object_file.sections[section_index].data
            * dest = buffer + mapping.output_offset
            * copy source data of size input_section.sh_size
        - For SHT_NOBITS: do nothing (buffer region remains zero)
    */
    /* TODO 5: Set *data = buffer, *size = section size */
}
```

#### E. Language-Specific Hints (C)

- **Memory Management**: Use `xcalloc` for zero-initialized allocations (e.g., for output section data buffers). This ensures padding bytes are zero.
- **String Handling**: When setting output section names, use `strdup` (or custom `xstrdup`) to copy constant strings like `".text"`. Remember to free these in `free_merged_sections`.
- **Linked List Operations**: For grouping input sections, a simple singly-linked list with head and tail pointers is sufficient. Append new nodes to the tail to maintain input order.
- **Alignment Calculation**: The `align_to` function should handle alignment=0 (some sections may have alignment 0 or 1). Treat alignment 0 as alignment 1 (no padding).
- **Efficient Group Lookup**: For simplicity, use a linear search through groups array (max ~4 groups). For advanced implementation, use a hash table keyed by `(sh_type << 32) | sh_flags` plus name comparison.

#### F. Milestone Checkpoint

After implementing the Section Merger, verify correctness with the following test:

1. **Create test object files**:
   ```bash
   echo '.global main; main: mov $0, %eax; ret' > test1.s
   echo '.global helper; helper: mov $1, %eax; ret' > test2.s
   as test1.s -o test1.o
   as test2.s -o test2.o
   ```

2. **Run the linker through Milestone 1** (your driver should call `read_elf_file` on both objects, then `merge_all_sections`):
   ```bash
   ./linker test1.o test2.o -o merged.bin
   ```
   This should produce a file `merged.bin` containing raw merged sections (not yet a valid ELF executable).

3. **Inspect the output**:
   - Use `readelf -S test1.o test2.o` to see original section sizes and alignments.
   - Write a small utility to print the `MergedSections` structure:
     - Output sections count should be 1 (`.text`) if only code sections exist.
     - Total size should be sum of both `.text` sections plus any alignment padding.
     - Each input section should have a mapping entry with correct `output_offset`.

4. **Expected behavior**: The merger should not crash, should compute a layout where the combined `.text` size is at least the sum of both input `.text` sections, and should produce mapping entries that correctly map `test1.o:.text` to offset 0 and `test2.o:.text` to offset `align_to(size_of_test1, alignment_of_test2)`.

#### G. Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| **Segmentation fault in `add_sections_from_object`** | Accessing invalid section index or null pointer. | Add bounds checking: `if (section_index >= obj->num_sections)`. Use debugger to see which section is being processed. | Validate indices before use; check that ELF Reader correctly populated `obj->sections`. |
| **Output section size smaller than expected** | Forgetting to add padding between input sections. | Print `current_offset` before and after each input section placement. Check that `padding` is calculated correctly. | Ensure `align_to` is called with the *input section's* alignment, not the output section's alignment. |
| **`.bss` section appears in output file with garbage data** | Treating `SHT_NOBITS` as `SHT_PROGBITS` during data assembly. | Check `sh_type` when copying data: only copy if `sh_type == SHT_PROGBITS`. | Add condition in `get_output_section_data` to skip copy for `SHT_NOBITS`. |
| **Symbol addresses later computed incorrectly** | Mapping entries have wrong `output_offset`. | Print mapping table: for each input section, show `output_offset`. Compare with manual calculation. | Verify that `output_offset` is set *after* adding padding but *before* adding the input section size. |
| **Alignment padding causes empty gaps in final executable** | Padding bytes are uninitialized (contain garbage). | Hexdump the output section data: look for non-zero bytes between sections. | Zero-initialize the entire output buffer in `get_output_section_data`. |


## Component: Symbol Resolver
> **Milestone(s):** Milestone 2: Symbol Resolution

The **Symbol Resolver** is the linchpin component of the static linker, responsible for creating a unified view of all named memory locations across disparate compilation units. While the ELF Reader extracts symbols from individual object files and the Section Merger organizes raw data, the Symbol Resolver connects the logical world of symbolic references (`main`, `printf`, `global_var`) to the physical world of memory addresses in the final executable. This component builds a global namespace by collecting all symbols, resolving conflicting definitions, and establishing definitive addresses that subsequent relocation can use. Its primary challenge is implementing the precise rules of ELF linking for strong/weak symbols, COMMON symbols, and undefined references, ensuring the resulting executable has exactly one definition for every referenced symbol.

### Mental Model: Conference Name Badges
Imagine organizing a large multi-track conference where each attendee (a symbol) must have a unique name badge. Different departments (object files) submit their attendance lists, which include both presenters (defined symbols with physical presence) and audience members (undefined symbols that need to find a presenter). The Symbol Resolver acts as the conference registration desk that:

1. **Collects all badges** from every department's submission lists.
2. **Resolves name conflicts** using precedence rules: keynote speakers (strong symbols) override workshop presenters (weak symbols), but workshop presenters are kept if no keynote exists.
3. **Handles provisional badges** for people who might show up (COMMON symbols) by allocating the largest requested space.
4. **Identifies missing presenters** (undefined symbols) and flags them as errors if no one can fill that role.
5. **Issues final badge numbers** (addresses) that everyone can reference when trying to find a particular person.

Each reference to a symbol in the code is like an attendee asking "Where can I find person X?" The resolver provides the definitive badge number (address) that the asker can use to locate them in the conference hall (memory). This mental model clarifies why local symbols (like people only known within their department) don't get global badges—they're irrelevant outside their own translation unit.

### Interface Specification
The Symbol Resolver exposes a minimal interface to the main linking pipeline, focused on constructing and querying the global symbol table. It operates on the parsed `ObjectFile` structures from the ELF Reader and the layout information from the `MergedSections` to compute final symbol addresses.

| Method Name | Parameters | Returns | Description |
|-------------|------------|---------|-------------|
| `resolve_all_symbols` | `ObjectFile** objects`, `uint32_t count`, `MergedSections* merged` | `SymbolTable*` | Main entry point: processes all symbols from all object files, resolves conflicts, and returns the complete global symbol table with assigned addresses. The `merged` parameter provides section layout for computing addresses of section-based symbols. |
| `find_symbol` | `SymbolTable* table`, `const char* name` | `SymbolEntry*` | Looks up a symbol by name in the resolved global table, returning `NULL` if not found. Used internally by the relocation applier to get target addresses. |
| `free_symbol_table` | `SymbolTable* table` | `void` | Deallocates the symbol table and all its entries. Called by the main linker after relocation is complete. |
| `report_undefined` | `SymbolTable* table` | `void` | Diagnostic function that scans for undefined symbols and prints error messages. Called before relocation if undefined symbols should be treated as fatal errors. |

The `SymbolTable` type returned by `resolve_all_symbols` is an opaque structure containing the complete resolved symbol mapping. Internally, it's implemented as a hash table keyed by symbol name for efficient lookup during relocation.

### Internal Behavior
The Symbol Resolver implements a **two-pass resolution algorithm** that first collects all symbol definitions across files, then resolves references and assigns final addresses. This approach ensures all definitions are known before attempting to resolve references, which is necessary for handling forward references across files.

#### Symbol Collection and Resolution Algorithm
1. **Initialize Global Symbol Table**
   - Create an empty hash table (e.g., using open addressing or chaining) with case-sensitive string keys (symbol names).
   - Preallocate sufficient capacity for the total number of global symbols across all input files.

2. **First Pass: Collect All Global Symbols**
   - For each `ObjectFile` in the input array:
     - Iterate through its `symbols` array (obtained from `.symtab` section).
     - For each `ElfSymbol`:
       - Extract binding (`STB_GLOBAL` or `STB_WEAK`), type, name, and defining section index.
       - Skip `STB_LOCAL` symbols entirely—they're invisible to other translation units.
       - If symbol is defined (`st_shndx != SHN_UNDEF` and not `SHN_COMMON`):
         - Create a tentative `SymbolEntry` with the symbol's value as offset within its input section.
         - Determine if this is a **strong** (`STB_GLOBAL`) or **weak** (`STB_WEAK`) definition.
         - Check global table for existing entry with same name:
           - **No existing entry**: Insert this symbol.
           - **Existing weak, new strong**: Replace weak definition with strong one.
           - **Existing strong, new weak**: Keep strong, ignore weak.
           - **Both strong**: Report fatal "multiple definition" error.
           - **Both weak**: Keep the first encountered (or pick arbitrarily—both are equally valid).
         - For `SHN_COMMON` symbols (uninitialized globals with tentative size):
           - Store as a special "COMMON" entry with size from `st_size`.
           - Apply **COMMON merging rule**: keep the largest size among all COMMON definitions of the same name.

3. **Second Pass: Process Undefined References**
   - For each `ObjectFile` again:
     - For each `ElfSymbol` with `st_shndx == SHN_UNDEF` (undefined reference):
       - Look up name in global table.
       - If not found, add as an **undefined symbol entry** marked `defined=false`.
       - If found but also undefined (possible with multiple files referencing same undefined symbol), merge entries.
   - After processing all files, scan global table for any symbols still marked undefined.
   - Report fatal "undefined symbol" error for any remaining undefined symbols unless explicitly allowed (e.g., for dynamic linking, though not in our scope).

4. **Address Assignment**
   - For each `SymbolEntry` in the global table that is defined (not undefined):
     - If symbol is section-relative (`st_shndx` indexes a normal section):
       - Use `MergedSections` mappings to translate `(file_index, section_index, offset)` to final virtual address.
       - Formula: `final_address = output_section->virtual_addr + offset_in_section`.
     - If symbol is absolute (`st_shndx == SHN_ABS`): address is simply `st_value`.
     - If symbol is COMMON (`st_shndx == SHN_COMMON`):
       - Allocate space in the `.bss` section of the output (handled by Section Merger with special COMMON allocation).
       - Address becomes `.bss` base + allocated offset.
   - Store computed address in `SymbolEntry.value`.

> **Key Insight:** The two-pass approach is essential because symbol references in one file may be defined in a later file in the input list. By collecting all definitions first, we ensure we have complete information before attempting resolution. This mirrors how traditional linkers like `ld` process archive libraries.

#### Data Structures for Internal State
The resolver maintains these core structures during processing:

| Structure | Fields | Purpose |
|-----------|--------|---------|
| `SymbolTable` (opaque) | `entries: SymbolEntry**`, `capacity: uint32_t`, `count: uint32_t`, `hash_seed: uint64_t` | Main symbol table using open-addressing hash table. Contains all resolved global symbols. |
| `SymbolEntry` | `name: char*`, `value: uint64_t`, `size: uint64_t`, `binding: uint8_t`, `type: uint8_t`, `defined: bool`, `is_common: bool`, `output_section: OutputSection*`, `offset_in_section: uint64_t`, `src_file: ObjectFile*`, `src_symbol_idx: uint32_t` | Complete information for one resolved symbol. `value` holds final virtual address after assignment. |
| `ResolutionState` | `current_file: ObjectFile*`, `file_index: uint32_t`, `phase: enum { COLLECTING, RESOLVING }` | Transient state used during the two passes to track progress and provide context for error messages. |

The hash table implementation uses FNV-1a or DJB2 hashing for symbol names, with linear probing for collision resolution. The table resizes when load factor exceeds 70%.

### Architecture Decision: Two-Pass Symbol Resolution

> **Decision: Two-Pass Resolution with Complete Definition Collection**
> - **Context**: The linker must resolve symbolic references across multiple object files where definitions and references can appear in any order. Forward references (reference before definition) are common, and symbol strength rules (strong vs weak) require comparing all definitions before making decisions.
> - **Options Considered**:
>   1. **Single-pass streaming**: Process files sequentially, resolving symbols as encountered. Update addresses when definitions are found.
>   2. **Two-pass with definition collection**: First pass collects all definitions, second pass resolves references using complete definition set.
>   3. **Multi-pass with dependency analysis**: Build dependency graph between symbols and resolve in topological order.
> - **Decision**: Two-pass with definition collection (Option 2).
> - **Rationale**:
>   - **Strong/Weak Resolution**: To correctly apply "strong overrides weak" rule, we must see ALL definitions before deciding which one wins. Single-pass might incorrectly choose a weak definition only to later discover a strong one.
>   - **COMMON Symbol Merging**: COMMON symbols require comparing sizes from all definitions to allocate the largest. This requires seeing all definitions.
>   - **Undefined Symbol Detection**: We cannot reliably detect undefined symbols until we've seen ALL files—a symbol undefined in file A might be defined in file B.
>   - **Simplicity**: Two-pass is simpler to implement and debug than dependency analysis, while handling all ELF requirements correctly.
>   - **Performance**: Acceptable for educational linker; the O(2N) cost is negligible for typical input sizes.
> - **Consequences**:
>   - **Memory Overhead**: Must store all symbol entries in memory before resolution completes.
>   - **File Re-reading**: Need to traverse each file's symbol table twice (or cache it).
>   - **Deterministic**: Always produces same result regardless of input file order for strong/weak resolution.

| Option | Pros | Cons | Why Not Chosen |
|--------|------|------|----------------|
| Single-pass streaming | Lower memory, single file traversal | Cannot handle strong/weak correctly, COMMON merging problematic, undefined detection unreliable | Fails core ELF semantics |
| Two-pass with definition collection | Correct strong/weak, correct COMMON, reliable undefined detection | Double traversal, memory for all symbols | **CHOSEN**: Correctness trumps minor overhead |
| Multi-pass dependency analysis | Handles circular references, optimal for complex cases | Complex implementation, overkill for static linking | Too complex for educational linker |

![Symbol Resolution Flowchart](./diagrams/symbol-resolution-flowchart.svg)

### Common Pitfalls

⚠️ **Pitfall: Incorrect Strong/Weak Resolution**
- **Description**: Treating all global symbols as equal precedence, or incorrectly allowing multiple strong definitions.
- **Why it's Wrong**: ELF specifies that strong symbols (`STB_GLOBAL`) override weak symbols (`STB_WEAK`), but multiple strong definitions of the same symbol constitute an error. Getting this wrong produces executables with non-deterministic behavior or misses legitimate errors.
- **How to Fix**: Always check binding type during first-pass collection. Maintain state for each symbol name: `NO_DEF`, `WEAK_DEF`, or `STRONG_DEF`. When encountering a new definition:
  - `NO_DEF` → store definition with its strength.
  - `WEAK_DEF` + new strong → replace with strong definition.
  - `STRONG_DEF` + new weak → keep strong, ignore weak.
  - `STRONG_DEF` + new strong → fatal error.

⚠️ **Pitfall: Mishandling COMMON Symbols**
- **Description**: Treating `SHN_COMMON` symbols like normal definitions or like undefined symbols.
- **Why it's Wrong**: COMMON symbols (uninitialized globals like `int global;`) have special semantics: they represent tentative definitions that should be merged, allocating space equal to the LARGEST size found across all object files. Treating them as normal definitions causes multiple allocations; treating them as undefined causes missing definitions.
- **How to Fix**: During first pass, detect `st_shndx == SHN_COMMON`. Create a special COMMON entry tracking the maximum `st_size`. During address assignment, allocate space in `.bss` using the maximum size (Section Merger must handle this specially).

⚠️ **Pitfall: Confusing Local vs Global Symbol Visibility**
- **Description**: Including `STB_LOCAL` symbols in the global symbol table or resolving references to them across files.
- **Why it's Wrong**: Local symbols (static functions/variables) are intentionally limited to their translation unit. Including them globally causes name collisions (two different `static int count` variables would conflict) and violates C semantics.
- **How to Fix**: During symbol collection, skip any symbol with `STB_LOCAL` binding. These symbols are only relevant for relocation WITHIN their original object file, handled later by the Relocation Applier using local symbol tables.

⚠️ **Pitfall: Incorrect Address Calculation for Section Symbols**
- **Description**: Computing symbol addresses using input-section offsets without applying section merging transformations.
- **Why it's Wrong**: After section merging, each input section is relocated to a new virtual address. A symbol defined at offset 0x10 in input `.text` might end up at 0x401010 in the final executable, not 0x10.
- **How to Fix**: During address assignment, use the `InputSectionMapping` structures from `MergedSections` to translate `(file_index, section_index, st_value)` triple to final virtual address:
  ```
  final_addr = merged->sections[output_sect_idx].virtual_addr + 
               mapping->output_offset + 
               (st_value - input_section_base)
  ```

### Implementation Guidance

**Technology Recommendations**
| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Hash Table | Open addressing with linear probing | Separate chaining with linked lists |
| String Storage | Store symbol names as `strdup` copies | Use string interning pool for deduplication |
| Error Reporting | Print to stderr and exit | Collect all errors before exiting, with source location |

**Recommended File Structure**
```
static-linker/
├── include/
│   └── linker.h              # Public interfaces for all components
├── src/
│   ├── main.c                # Main driver
│   ├── elf_reader.c          # From Milestone 1
│   ├── section_merger.c      # From Milestone 1
│   ├── symbol_resolver.c     # This component (NEW)
│   ├── relocation.c          # Milestone 3
│   ├── executable_writer.c   # Milestone 4
│   └── util/
│       ├── hash_table.c      # Generic hash table utilities
│       └── errors.c          # Error reporting utilities
```

**Infrastructure Starter Code: Hash Table**
```c
// src/util/hash_table.h
#ifndef HASH_TABLE_H
#define HASH_TABLE_H

#include <stdint.h>
#include <stdbool.h>

typedef struct HashTable HashTable;

// Create/destroy
HashTable* hash_table_create(uint32_t initial_capacity);
void hash_table_destroy(HashTable* table);

// Insert/lookup/remove
bool hash_table_insert(HashTable* table, const char* key, void* value);
void* hash_table_lookup(const HashTable* table, const char* key);
bool hash_table_remove(HashTable* table, const char* key);

// Iteration
typedef struct HashTableIterator HashTableIterator;
HashTableIterator* hash_table_iterate(const HashTable* table);
bool hash_table_next(HashTableIterator* iter, const char** key, void** value);
void hash_table_iterator_destroy(HashTableIterator* iter);

#endif // HASH_TABLE_H
```

```c
// src/util/hash_table.c
#include "hash_table.h"
#include <stdlib.h>
#include <string.h>

#define FNV_OFFSET_BASIS 0xcbf29ce484222325ULL
#define FNV_PRIME 0x100000001b3ULL

typedef struct HashEntry {
    char* key;
    void* value;
    bool occupied;
    bool deleted;
} HashEntry;

struct HashTable {
    HashEntry* entries;
    uint32_t capacity;
    uint32_t count;
    uint32_t deleted_count;
};

// FNV-1a hash function
static uint64_t hash_string(const char* str) {
    uint64_t hash = FNV_OFFSET_BASIS;
    while (*str) {
        hash ^= (uint8_t)(*str++);
        hash *= FNV_PRIME;
    }
    return hash;
}

HashTable* hash_table_create(uint32_t initial_capacity) {
    HashTable* table = (HashTable*)xmalloc(sizeof(HashTable));
    table->capacity = initial_capacity > 0 ? initial_capacity : 16;
    table->count = 0;
    table->deleted_count = 0;
    table->entries = (HashEntry*)xcalloc(table->capacity, sizeof(HashEntry));
    return table;
}

// ... (rest of implementation with open addressing, resizing at 70% load factor)
// Full implementation available in project resources
```

**Core Logic Skeleton Code**
```c
// src/symbol_resolver.c
#include "linker.h"
#include "util/hash_table.h"

typedef enum {
    SYM_NO_DEF,
    SYM_WEAK_DEF,
    SYM_STRONG_DEF,
    SYM_COMMON_DEF,
    SYM_UNDEFINED
} SymbolStrength;

typedef struct SymbolState {
    SymbolEntry entry;
    SymbolStrength strength;
    uint32_t file_index;
    uint32_t sym_index;
} SymbolState;

// Main resolution function
SymbolTable* resolve_all_symbols(ObjectFile** objects, uint32_t count, 
                                 MergedSections* merged) {
    HashTable* global_table = hash_table_create(count * 32); // Estimate 32 symbols/file
    
    // PASS 1: Collect all definitions
    for (uint32_t file_idx = 0; file_idx < count; file_idx++) {
        ObjectFile* obj = objects[file_idx];
        for (uint32_t sym_idx = 0; sym_idx < obj->num_symbols; sym_idx++) {
            ElfSymbol* sym = &obj->symbols[sym_idx];
            uint8_t binding = ELF64_ST_BIND(sym->sym.st_info);
            
            // TODO 1: Skip local symbols (STB_LOCAL)
            // TODO 2: Check if symbol is defined (st_shndx != SHN_UNDEF)
            // TODO 3: Handle SHN_COMMON symbols specially (tentative definitions)
            // TODO 4: Determine symbol strength: strong (STB_GLOBAL), weak (STB_WEAK)
            // TODO 5: Look up existing entry in global_table
            // TODO 6: Apply strong/weak resolution rules:
            //         - No existing entry → insert
            //         - Existing weak, new strong → replace
            //         - Existing strong, new weak → keep strong
            //         - Both strong → fatal_error("multiple definition")
            //         - Both weak → keep first
            // TODO 7: For COMMON symbols, track maximum size
        }
    }
    
    // PASS 2: Process undefined references
    for (uint32_t file_idx = 0; file_idx < count; file_idx++) {
        ObjectFile* obj = objects[file_idx];
        for (uint32_t sym_idx = 0; sym_idx < obj->num_symbols; sym_idx++) {
            ElfSymbol* sym = &obj->symbols[sym_idx];
            // TODO 8: Skip local symbols again
            // TODO 9: If symbol is undefined (st_shndx == SHN_UNDEF)
            // TODO 10: Look up in global_table
            // TODO 11: If not found, add as undefined symbol entry
            // TODO 12: If found but also undefined, merge (or keep as is)
        }
    }
    
    // TODO 13: Scan global_table for any symbols still undefined
    // TODO 14: If found, report fatal error with list of undefined symbols
    
    // PASS 3: Assign addresses
    HashTableIterator* iter = hash_table_iterate(global_table);
    const char* key;
    SymbolState* state;
    while (hash_table_next(iter, &key, (void**)&state)) {
        // TODO 15: Skip undefined symbols (they should have been caught)
        // TODO 16: If symbol is COMMON: allocate space in .bss via merged sections
        // TODO 17: If symbol is absolute (SHN_ABS): address = st_value
        // TODO 18: If symbol is section-relative:
        //          a. Find input section from file_idx/sym_idx
        //          b. Use merged->mappings to find output section and offset
        //          c. Calculate: address = output_section->virtual_addr + 
        //                           mapping->output_offset + (st_value - section_base)
        // TODO 19: Store final address in state->entry.value
    }
    hash_table_iterator_destroy(iter);
    
    // Convert HashTable to SymbolTable (opaque type)
    SymbolTable* result = (SymbolTable*)global_table;
    return result;
}

// Lookup function for relocations
SymbolEntry* find_symbol(SymbolTable* table, const char* name) {
    HashTable* ht = (HashTable*)table;
    SymbolState* state = (SymbolState*)hash_table_lookup(ht, name);
    return state ? &state->entry : NULL;
}
```

**Language-Specific Hints for C**
- **Symbol Binding Macros**: Use `ELF64_ST_BIND(st_info)` and `ELF64_ST_TYPE(st_info)` macros to extract binding and type from `st_info` field. These are typically defined in `<elf.h>` but you can define them if not available.
- **String Management**: Use `strdup` for symbol names but remember to free them in `free_symbol_table`. Consider using a string arena allocator if memory usage is a concern.
- **Error Messages**: Include the object filename and symbol name in error messages: `fatal_error("undefined symbol '%s' in %s", symname, obj->filename)`.
- **Type Casting**: When storing `SymbolState*` in the hash table's `void*` value, ensure proper alignment. Use `malloc` for each `SymbolState` rather than storing them by value in the hash table entries.

**Milestone Checkpoint**
After implementing the Symbol Resolver, test with these commands:
1. **Basic Resolution**: Create two C files: `a.c` with `int global = 42;` and `b.c` with `extern int global; int main() { return global; }`. Compile to object files (`gcc -c a.c b.c`) and run your linker. It should succeed without "undefined symbol" errors.
2. **Strong/Weak Test**: Create `weak.c` with `__attribute__((weak)) int foo() { return 1; }` and `strong.c` with `int foo() { return 2; }`. Link both with a main that calls `foo()`. The linker should use the strong definition, and the program should return 2.
3. **Multiple Definition Error**: Create two files both defining `int conflict = 5;` (without `static`). Linking them should produce a fatal "multiple definition" error.
4. **COMMON Symbol Test**: Create two files with `int common_var;` (uninitialized global). The linker should allocate a single `.bss` entry of size 4 bytes (not two separate entries).

Use `readelf -s` on your output executable to verify symbol addresses are correctly computed and match expectations. The symbol table should show `global` and `main` with valid virtual addresses, and no undefined symbols.


## Component: Relocation Applier
> **Milestone(s):** Milestone 3: Relocations

The **Relocation Applier** is the component responsible for fixing symbolic references within the merged section data. After sections have been concatenated and symbols have been resolved to final addresses, this component processes relocation entries that instruct it to patch specific locations within the code and data with the computed addresses. This transforms position-dependent object files into a position-independent executable where all internal references correctly point to their ultimate destinations.

### Mental Model: Address Labeling Service

Imagine you've just moved to a new city and need to update all your correspondence. You've previously given out your old address on business cards, letters, and packages. A mail forwarding service receives your mail, examines each item, finds the old address written on it, and replaces it with your new address before forwarding it to you. The **Relocation Applier** operates similarly:

- **Old addresses in object files** are placeholders (often zero or dummy values) that compilers embed when they don't know final symbol locations.
- **Relocation entries** are the forwarding instructions that specify where these old addresses are located (the offset within a section) and which symbol they refer to.
- **Resolved symbol addresses** are the new, definitive locations calculated by the Symbol Resolver.
- **The patching process** involves visiting each relocation site, computing the correct final address based on symbol location and relocation type, and overwriting the placeholder with the computed value.

This mental model clarifies that relocation is fundamentally a **data transformation** applied to the raw bytes of merged sections, guided by explicit metadata (relocation entries) and the global address map (symbol table).

### Interface Specification

The Relocation Applier provides a single primary interface method that orchestrates the entire relocation process, plus internal helper functions for specific relocation types. It operates on the already-merged sections and the resolved symbol table.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `apply_all_relocations` | `objects`: array of `ObjectFile*`, `count`: number of objects, `merged`: `MergedSections*`, `symbols`: `SymbolTable*` | `void` | Main entry point. Iterates through all object files, processes their relocation sections, and patches the corresponding merged section data. |
| `apply_relocation_to_section` (internal) | `rela`: `Elf64_Rela`, `symbol_addr`: `uint64_t`, `section_base`: `uint64_t`, `patch_addr`: `uint8_t*` | `bool` (success) | Applies a single relocation of a specific type. Calculates the value to write and patches the memory at `patch_addr`. Handles type-specific calculations (PC-relative vs absolute). |
| `compute_absolute_relocation` (internal) | `symbol_addr`: `uint64_t`, `addend`: `int64_t` | `uint64_t` | Computes the value for an absolute relocation: `symbol_addr + addend`. |
| `compute_pcrel_relocation` (internal) | `symbol_addr`: `uint64_t`, `section_base`: `uint64_t`, `rel_offset`: `uint64_t`, `addend`: `int64_t` | `uint64_t` | Computes the value for a PC-relative relocation: `symbol_addr - (section_base + rel_offset) + addend`. |
| `write_value_at_address` (internal) | `patch_addr`: `uint8_t*`, `value`: `uint64_t`, `width_bytes`: `int` | `void` | Writes `value` to `patch_addr` using little-endian byte order, truncating to `width_bytes` (4 or 8). Includes overflow checking for 32-bit writes. |

The component expects that:
1. `merged` contains the final section layout with `virtual_addr` set for each output section.
2. `symbols` contains fully resolved symbols with final virtual addresses (`value` field).
3. The data pointers in `merged->sections[].data` point to modifiable memory (allocated via `xmalloc`).

### Internal Behavior

The relocation algorithm follows a systematic, nested iteration pattern: for each object file, for each relocation section, for each relocation entry, apply the relocation. Here is the step-by-step procedure:

1. **Iterate over input object files**: The linker processes each input `ObjectFile` to access its relocation sections.
2. **Identify relocation sections**: For each section in the object file, check if `sh_type == SHT_RELA`. These sections contain arrays of `Elf64_Rela` entries.
3. **Determine target section**: Each relocation section is associated with a specific section being relocated (the "target section"). This is indicated by `sh_info` field in the relocation section header, which contains the index of the target section within that object file.
4. **Map to merged output section**: Using the `InputSectionMapping` table in `merged`, find which output section the target section was merged into, and compute the offset within that output section where the target section's data begins.
5. **Process each relocation entry**:
   1. Extract the `Elf64_Rela` structure containing:
      - `r_offset`: offset within the target section where the patch should be applied.
      - `r_info`: encodes both the symbol index (high 32 bits) and relocation type (low 32 bits).
      - `r_addend`: constant value to add to the computed address.
   2. Decode the symbol index from `r_info` and look up the corresponding `ElfSymbol` in the object file's symbol table.
   3. Using the symbol's name, query the global `SymbolTable` to obtain the final `SymbolEntry` with its resolved virtual address.
   4. Calculate the **patch location** in memory: 
      - Output section base virtual address (`merged->sections[output_idx].virtual_addr`)
      - Plus the target section's offset within that output section (`mapping.output_offset`)
      - Plus the relocation offset within the target section (`r_offset`)
   5. Determine the relocation type from `r_info` (e.g., `R_X86_64_64` or `R_X86_64_PC32`).
   6. Compute the value to write based on relocation type:
      - For `R_X86_64_64` (absolute 64-bit): `symbol_addr + addend`
      - For `R_X86_64_PC32` (PC-relative 32-bit): `symbol_addr - (section_base + rel_offset) + addend` where `section_base` is the virtual address of the output section containing the relocation site.
   7. Check for overflow if the value must be truncated (e.g., 64-bit value into 32-bit field). For `R_X86_64_PC32`, ensure the signed 32-bit result fits within `[-2^31, 2^31-1]`.
   8. Write the computed value to the patch location in little-endian byte order, with the appropriate width (8 bytes for `R_X86_64_64`, 4 bytes for `R_X86_64_PC32`).

6. **Repeat** until all relocation entries from all object files have been processed.

> **Key Insight:** The most subtle aspect is the calculation for PC-relative relocations. The offset being patched is relative to the *next instruction's address*, but `r_offset` points to the location of the relocation itself (the displacement field within the instruction). The formula `symbol_addr - (section_base + rel_offset) + addend` correctly computes the displacement from the next instruction because x86-64 PC-relative addressing uses the address of the *next* instruction, not the address of the displacement field. The `addend` allows the compiler to encode an initial displacement (often zero).

### Architecture Decision: In-Place vs Copy Relocation

> **Decision: In-Place Relocation**
> - **Context**: After section merging, we have contiguous blocks of memory containing the raw data from each input section. Relocations require modifying specific bytes within these blocks. We must choose whether to modify the original merged data directly or create a separate relocated copy.
> - **Options Considered**:
>   1. **In-Place Modification**: Directly patch the bytes in the merged section data arrays.
>   2. **Copy-and-Relocate**: Allocate fresh memory for each output section, copy the original data, and apply relocations to the copy.
>   3. **Streaming Relocation**: Apply relocations as sections are being written to the final executable file, never holding fully relocated sections in memory.
> - **Decision**: Use **In-Place Modification** of the merged section data.
> - **Rationale**:
>   - **Simplicity**: The merged sections already exist in memory as mutable arrays. In-place modification requires no additional memory allocation or copying.
>   - **Predictable workflow**: The Section Merger produces the final layout; the Relocation Applier finalizes the content; the Executable Writer simply serializes the already-relocated data. This clear separation of phases aids debugging.
>   - **Educational value**: Seeing the actual bytes change in the section data helps learners understand how relocations physically alter the binary.
> - **Consequences**:
>   - The merged section data becomes "dirty" and cannot be reused for multiple output formats (not needed for our single-executable output).
>   - Relocation must occur after all sections are merged and before writing the executable, creating a strict pipeline order.
>   - If a relocation calculation fails (e.g., overflow), the section data may be partially modified, but since we abort on error, this is acceptable.

| Option | Pros | Cons | Why Not Chosen |
|--------|------|------|----------------|
| In-Place Modification | Simple, memory-efficient, clear phase separation | Irreversible changes, can't retry with different layouts | **Chosen** for its simplicity and educational clarity |
| Copy-and-Relocate | Preserves original data, allows multiple relocation passes | Doubles memory usage, adds copy overhead | Unnecessary memory overhead for our single-pass linker |
| Streaming Relocation | Minimal memory footprint, enables linking of huge files | Complex to implement, intertwines relocation with writing | Overly complex for educational purposes; obscures the relocation step |

### Common Pitfalls

⚠️ **Pitfall: Incorrect PC-relative calculation using wrong base address**
- **Description**: Using the output section's virtual address directly as the base in PC-relative calculations, rather than the address of the relocation site itself.
- **Why it's wrong**: PC-relative relocations compute the offset from the *next instruction* to the symbol. The formula must subtract the address where the relocation is applied (`section_base + r_offset`), not just the section start.
- **Fix**: Always compute `relocation_site_addr = output_section_base + target_section_offset + r_offset`. For `R_X86_64_PC32`, use `symbol_addr - relocation_site_addr + addend`.

⚠️ **Pitfall: Ignoring the addend or applying it incorrectly**
- **Description**: Adding the addend to the symbol address before or after the PC-relative adjustment inconsistently, or forgetting it entirely.
- **Why it's wrong**: The addend is part of the compiler-generated relocation specification and must be included in the final calculation exactly as specified. For `R_X86_64_PC32`, the addend is typically -4 because the displacement is measured from the next instruction, but the relocation offset points to the displacement field itself (4 bytes earlier). Omitting it produces addresses off by 4 bytes.
- **Fix**: Follow the standard formulas precisely: absolute = `symbol_addr + addend`; PC-relative = `symbol_addr - relocation_site_addr + addend`. Use the `r_addend` field directly; don't assume it's zero.

⚠️ **Pitfall: Truncation overflow in 32-bit relocations**
- **Description**: Computing a 64-bit address difference that exceeds the signed 32-bit range when writing a `R_X86_64_PC32` relocation.
- **Why it's wrong**: The displacement field in the instruction is only 32 bits (signed). If the symbol is more than ±2GB away from the relocation site, the value won't fit, causing runtime jumps to wrong addresses.
- **Fix**: After computing the PC-relative value, check if it fits in `int32_t` range (`-2147483648` to `2147483647`). If not, report an overflow error. This typically indicates the linked executable is too large or sections are laid out too far apart.

⚠️ **Pitfall: Byte order (endianness) confusion when patching**
- **Description**: Writing multi-byte values in host byte order (which may be big-endian) instead of the target's little-endian format.
- **Why it's wrong**: x86-64 is little-endian. The bytes `0x1122334455667788` should be stored in memory as `0x88 0x77 0x66 0x55 0x44 0x33 0x22 0x11`. Writing in host byte order produces incorrect addresses on little-endian hosts and completely broken addresses on big-endian hosts.
- **Fix**: Always write values in little-endian byte order when patching. Use helper functions that explicitly store bytes from least significant to most significant.

⚠️ **Pitfall: Relocating against wrong symbol due to symbol table index mismatch**
- **Description**: Using the symbol index from `r_info` directly in the global symbol table instead of looking up the symbol in the object file's symbol table first to get its name, then querying the global table.
- **Why it's wrong**: Symbol indices are local to each object file's symbol table. The global symbol table is indexed by name, not by original indices. Using the index directly will access wrong or out-of-bounds entries.
- **Fix**: Always follow the chain: relocation entry → symbol index → object file's `symbols[index]` → symbol name → global `SymbolTable` lookup.

### Implementation Guidance

#### A. Technology Recommendations Table
| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Relocation Calculation | Manual implementation of `R_X86_64_64` and `R_X86_64_PC32` | Table-driven dispatcher supporting many relocation types |
| Byte Patching | Direct pointer arithmetic with bit shifts | `memcpy` with union-based type punning |
| Overflow Detection | Explicit range checks before truncation | Compiler intrinsics for signed overflow detection |

#### B. Recommended File/Module Structure
```
static-linker/
├── include/
│   ├── linker.h           # Main interface declarations
│   └── reloc.h            # Relocation-specific types and functions
├── src/
│   ├── main.c             # CLI driver
│   ├── elf_reader.c       # ELF Reader component
│   ├── section_merger.c   # Section Merger component
│   ├── symbol_resolver.c  # Symbol Resolver component
│   ├── reloc_applier.c    # Relocation Applier component (this section)
│   ├── exec_writer.c      # Executable Writer component
│   ├── utils.c            # Shared utilities (align_to, xmalloc, etc.)
│   └── hash_table.c       # Hash table implementation
└── test/
    ├── test_reloc.c       # Unit tests for relocation
    └── test_programs/     # Source code for test linking
```

#### C. Infrastructure Starter Code

**Complete byte-order handling utilities** (add to `utils.c`):

```c
#include <stdint.h>
#include <string.h>

// Write a 64-bit value in little-endian byte order
void write_le64(uint8_t *ptr, uint64_t value) {
    for (int i = 0; i < 8; i++) {
        ptr[i] = (value >> (i * 8)) & 0xFF;
    }
}

// Write a 32-bit value in little-endian byte order  
void write_le32(uint8_t *ptr, uint32_t value) {
    for (int i = 0; i < 4; i++) {
        ptr[i] = (value >> (i * 8)) & 0xFF;
    }
}

// Read a 64-bit value in little-endian byte order
uint64_t read_le64(const uint8_t *ptr) {
    uint64_t result = 0;
    for (int i = 0; i < 8; i++) {
        result |= ((uint64_t)ptr[i]) << (i * 8);
    }
    return result;
}

// Check if a signed 64-bit value fits in 32 bits
int fits_in_int32(int64_t value) {
    return value >= INT32_MIN && value <= INT32_MAX;
}
```

#### D. Core Logic Skeleton Code

**Main relocation application function** (in `reloc_applier.c`):

```c
#include "linker.h"
#include "reloc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Apply all relocations from all object files to the merged sections
void apply_all_relocations(ObjectFile **objects, uint32_t num_objects,
                           MergedSections *merged, SymbolTable *global_syms) {
    // TODO 1: Iterate through each object file (for i = 0 to num_objects-1)
    // TODO 2: For each section in object file, check if sh_type == SHT_RELA
    // TODO 3: For each relocation section, get target section index from sh_info
    // TODO 4: Find InputSectionMapping for this (file_index, target_section_index)
    // TODO 5: Determine output section index and offset from mapping
    // TODO 6: Get output section base address from merged->sections[output_idx].virtual_addr
    // TODO 7: Get pointer to output section data: merged->sections[output_idx].data
    // TODO 8: For each Elf64_Rela in relocation section:
    //   a. Extract symbol index from r_info (high 32 bits: (r_info >> 32))
    //   b. Get ElfSymbol from object's symbol table using symbol index
    //   c. Look up symbol name in global SymbolTable to get SymbolEntry
    //   d. Calculate patch address: section_data_ptr + mapping.output_offset + rela.r_offset
    //   e. Determine relocation type (low 32 bits of r_info: (r_info & 0xFFFFFFFF))
    //   f. Call apply_relocation_to_section() with appropriate parameters
    //   g. If apply_relocation_to_section returns false, report error and abort
    // TODO 9: Continue until all relocation sections processed
}

// Internal: Apply a single relocation
static bool apply_relocation_to_section(const Elf64_Rela *rela,
                                        uint64_t symbol_addr,
                                        uint64_t section_base,
                                        uint8_t *patch_addr) {
    uint32_t type = rela->r_info & 0xFFFFFFFF;
    
    // TODO 1: Switch on relocation type
    // TODO 2: For R_X86_64_64:
    //   a. Compute value = symbol_addr + rela->r_addend
    //   b. Call write_le64(patch_addr, value)
    //   c. Return true
    // TODO 3: For R_X86_64_PC32:
    //   a. Compute relocation_site_addr = section_base + rela->r_offset
    //   b. Compute value = symbol_addr - relocation_site_addr + rela->r_addend
    //   c. Check if value fits in int32_t using fits_in_int32()
    //   d. If not fits, print error about overflow and return false
    //   e. Write lower 32 bits using write_le32(patch_addr, (uint32_t)value)
    //   f. Return true
    // TODO 4: For unsupported type: print error and return false
}
```

#### E. Language-Specific Hints (C)

- **Pointer Arithmetic**: Use `uint8_t*` for byte-wise pointer arithmetic when calculating patch addresses. Cast section data pointers to `uint8_t*` for flexibility.
- **Symbol Index Extraction**: Use `(rela->r_info >> 32)` to get symbol index and `(rela->r_info & 0xFFFFFFFF)` to get type. These follow ELF specification.
- **Error Messages**: Print detailed error messages including symbol name, relocation type, and computed value to aid debugging.
- **Type Casting**: When checking 32-bit overflow, cast the computed `int64_t` value to `int32_t` and back, comparing for equality to detect truncation.

#### F. Milestone Checkpoint

After implementing the Relocation Applier, test with a simple program that uses both absolute and PC-relative relocations:

1. **Create test program** (`test_reloc.c`):
```c
extern int global_var;
void external_func();

int main() {
    global_var = 42;
    external_func();
    return 0;
}
```

2. **Compile separately**:
```bash
gcc -c test_reloc.c -o test_reloc.o
# Create a dummy external_func and global_var in another file
```

3. **Link with your linker**:
```bash
./linker test_reloc.o other.o -o test_reloc
```

4. **Verify**:
   - Use `objdump -d test_reloc` to check that call to `external_func` has a proper relative displacement.
   - Use `readelf -r test_reloc.o` to see original relocations, then verify they're absent in the executable (they should be resolved).
   - Run the executable (if all symbols defined) to ensure it doesn't crash.

#### G. Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Executable crashes with segmentation fault immediately | Wrong address patched in PC-relative jump | Use `objdump -d` to see the instruction bytes at the call site. Check if the displacement points outside the text segment. | Verify PC-relative calculation includes the addend correctly. |
| Global variable contains wrong value | Absolute relocation computed incorrectly | Examine the data section with `hexdump -C` to see the stored address. Compare with symbol address from `readelf -s`. | Check that `symbol_addr + addend` uses the right symbol address and includes addend. |
| Linker reports "relocation overflow" | Symbols too far apart in virtual memory | Check the distance between sections in merged layout. Symbols more than ±2GB apart cannot use 32-bit PC-relative relocations. | Adjust section ordering to keep related code closer, or implement linker relaxation (advanced). |
| Relocation writes zeros | Symbol not found in global table | Print the symbol name being looked up for each relocation. Check if it's defined in any input. | Ensure symbol resolution completed successfully and all global symbols are in the table. |


## Component: Executable Writer

> **Milestone(s):** Milestone 4: Executable Generation

The **Executable Writer** is the final component in the static linker's pipeline, responsible for transforming the merged and relocated section data into a fully-formed ELF executable that can be loaded and run by the operating system. While previous components dealt with the logical organization of code and data, this component focuses on the physical layout and packaging requirements that the OS loader expects. It creates the **ELF header** that identifies the file type, the **program headers** that describe how the file should be mapped into memory, and writes the actual bytes to disk in the proper ELF format.

![Executable Segment Layout](./diagrams/segment-layout-diagram.svg)

### Mental Model: Building Blueprint

Think of the Executable Writer as an **architect creating construction blueprints for a house**. The merged sections (.text, .data, .rodata, .bss) are like the rooms of the house (kitchen, bedroom, garage). The architect's job is to:

1. **Create a site plan (ELF header)** that identifies the property as a residential building (executable file) and points to the blueprint set (program header table).
2. **Draw floor plans (program headers)** that specify exactly how each part of the house should be constructed in its final location:
   - Which rooms go together in which wing (segment grouping)
   - The exact dimensions and position of each wing (virtual address and size)
   - What each wing is used for: living spaces (executable code), storage (data), or unfinished areas (.bss)
   - The access rules for each wing: read-only, write-only, or executable (permission flags)
3. **Ensure proper spacing between wings (page alignment)** so that construction equipment (the OS loader using `mmap`) can work efficiently.
4. **Mark the front door (entry point)** so people know where to enter the house.

Just as a builder uses blueprints to construct the actual house in the right location with the right materials, the OS loader uses the ELF headers to map the binary into memory at the correct addresses with the correct permissions. A missing or incorrect blueprint causes the builder to either refuse construction or build something unusable.

### Interface Specification

The Executable Writer exposes a minimal interface focused on generation and output, accepting the fully-processed linking artifacts and producing a runnable file.

| Method Name | Parameters | Returns | Description |
|-------------|------------|---------|-------------|
| `generate_executable` | `merged` (const MergedSections*), `symbols` (const SymbolTable*) | `OutputExecutable*` | Creates an in-memory representation of the final executable from merged sections and resolved symbols. This includes computing segment layout, constructing headers, and preparing section data for writing. |
| `write_executable` | `exec` (const OutputExecutable*), `filename` (const char*) | `void` | Serializes the in-memory executable representation to disk as a valid ELF file. Writes all headers and section data to the specified file path. |
| `free_output_executable` | `exec` (OutputExecutable*) | `void` | Deallocates all memory associated with an `OutputExecutable` structure, including header copies and section data buffers. |

The `OutputExecutable` type serves as an opaque container for the fully-formed executable data. Its internal structure (not exposed to other components) typically includes:

| Field Name | Type | Description |
|------------|------|-------------|
| `header` | `Elf64_Ehdr` | The ELF file header identifying the file as an executable. |
| `phdrs` | `Elf64_Phdr*` | Array of program headers describing loadable segments. |
| `num_phdrs` | `uint16_t` | Count of program headers in the array. |
| `segment_data` | `uint8_t**` | Array of pointers to raw data for each loadable segment. |
| `segment_sizes` | `size_t*` | Array of sizes for each segment's data. |
| `entry_point` | `uint64_t` | Virtual address of the entry point (_start symbol). |
| `text_segment_vaddr` | `uint64_t` | Starting virtual address of the text (RX) segment. |
| `data_segment_vaddr` | `uint64_t` | Starting virtual address of the data (RW) segment. |

### Internal Behavior

The Executable Writer operates through a precise sequence of layout calculations and header construction. Its internal workflow can be broken down into distinct phases:

1. **Segment Planning and Grouping**
   - Examine all output sections from the `MergedSections` structure and group them by memory permissions and access patterns.
   - Create at least two **PT_LOAD** segments:
     - **Text segment**: Contains all sections with `SHF_ALLOC` and `SHF_EXECINSTR` flags (typically `.text`), plus read-only allocated sections (`.rodata`). This segment will have permissions `PF_R | PF_X` (read and execute).
     - **Data segment**: Contains all sections with `SHF_ALLOC` and `SHF_WRITE` flags (typically `.data`), plus the `.bss` section (which occupies memory but no file space). This segment will have permissions `PF_R | PF_W` (read and write).
   - For each group, compute the **total size in memory** (including `.bss` expansion) and **total size in file** (excluding `.bss`).

2. **Address and Alignment Calculation**
   - Assign virtual addresses starting from a conventional base address (e.g., `0x400000` for x86-64 Linux executables).
   - Apply strict **page alignment** (4096 bytes) to segment start addresses in memory (`p_vaddr`) and in file (`p_offset`). Use the `align_to` helper function to round up addresses.
   - Ensure that sections within a segment are placed at correct offsets relative to the segment base, maintaining their individual alignment requirements (`sh_addralign`).
   - Calculate the **memory size** (`p_memsz`) for each segment: the sum of all section sizes including `.bss`.
   - Calculate the **file size** (`p_filesz`) for each segment: the sum of all section sizes excluding `.bss` (since `.bss` occupies no file space).

3. **Header Construction**
   - **ELF Header**:
     - Set `e_ident` fields: magic number, class (64-bit), data (little-endian), version, OS ABI (typically 0 for System V).
     - Set `e_type = ET_EXEC` (2) for an executable.
     - Set `e_machine = EM_X86_64`.
     - Set `e_version = EV_CURRENT`.
     - Set `e_entry` to the resolved virtual address of the `_start` symbol (obtained from the `SymbolTable`).
     - Set `e_phoff` to the offset of the program header table (immediately after the ELF header).
     - Set `e_shoff = 0` (executables typically omit section headers).
     - Set `e_ehsize = sizeof(Elf64_Ehdr)`.
     - Set `e_phentsize = sizeof(Elf64_Phdr)`.
     - Set `e_phnum` to the number of program headers.
     - Set other fields to 0 or appropriate defaults.
   - **Program Headers** (for each PT_LOAD segment):
     - Set `p_type = PT_LOAD` (1).
     - Set `p_flags`: `PF_X | PF_R` for text segment, `PF_R | PF_W` for data segment.
     - Set `p_offset` to the file offset where segment data begins (page-aligned).
     - Set `p_vaddr` and `p_paddr` to the virtual address where the segment loads (page-aligned).
     - Set `p_filesz` to the size of segment data in the file.
     - Set `p_memsz` to the size of segment data in memory (larger than `p_filesz` if `.bss` is present).
     - Set `p_align = 4096` (page alignment).

4. **Data Packaging and Serialization**
   - Concatenate section data for each segment into contiguous buffers, inserting any necessary intra-segment padding to maintain section alignment.
   - For the data segment, ensure the `.bss` portion is not written to file (it will be zero-initialized by the OS loader).
   - Write the final file in the following order:
     1. ELF header
     2. Program header table
     3. Padding to reach `p_offset` of first segment
     4. Text segment data (.text + .rodata)
     5. Padding to reach `p_offset` of second segment
     6. Data segment data (.data only, no .bss)
   - All numeric values must be written in **little-endian** byte order.

> **Key Insight:** The separation between `p_filesz` and `p_memsz` is crucial for handling `.bss` and other `SHT_NOBITS` sections. The file contains only initialized data; the loader expands the memory image by zero-initializing the difference between memory size and file size. This optimization saves disk space for uninitialized global variables.

### Architecture Decision: Single vs Multiple PT_LOAD Segments

> **Decision: Multiple PT_LOAD Segments for Permission Separation**
> - **Context**: The linker must create an executable that can be loaded into memory with proper page protection (read, write, execute). Modern operating systems use memory protection units to enforce security policies like W^X (write XOR execute), which prevents memory pages from being both writable and executable simultaneously to mitigate code injection attacks.
> - **Options Considered**:
>   1. **Single PT_LOAD segment**: Combine all allocated sections into one contiguous block with combined permissions (`PF_R | PF_W | PF_X`).
>   2. **Two PT_LOAD segments**: Separate executable code (RX) from writable data (RW) into distinct loadable segments.
>   3. **Three or more PT_LOAD segments**: Further separate read-only data (R) from executable code (RX) and writable data (RW).
> - **Decision**: Use **two PT_LOAD segments** – one for text (RX) and one for data (RW). This is the minimal configuration that satisfies W^X requirements while remaining simple to implement.
> - **Rationale**:
>   - **Security**: A single segment with RWX permissions would create an executable that violates modern security hardening standards and might be rejected by some systems.
>   - **Simplicity**: Two segments provide clear separation without overcomplicating the layout algorithm. Three segments (separating .rodata) would require more complex grouping logic for minimal practical benefit in this educational project.
>   - **Compatibility**: The two-segment model matches what typical toolchains (like `ld`) produce for simple executables and is universally accepted by Linux loaders.
>   - **Educational Value**: Implementing segment separation teaches crucial concepts about memory protection and loader expectations.
> - **Consequences**:
>   - The linker must group sections by permission flags and compute separate alignments for each segment.
>   - Virtual address calculations become slightly more complex, as we must place segments at page-aligned boundaries with possible gaps between them.
>   - The executable file will have some internal fragmentation due to page alignment padding between segments in the file.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Single PT_LOAD segment | Simplest implementation; fewer calculations; smaller file size (no alignment gaps) | Violates W^X security principle; may be rejected by secure systems; doesn't teach segment separation | No |
| Two PT_LOAD segments (RX + RW) | Satisfies W^X; matches typical toolchain output; clear security separation | Requires permission-based grouping; introduces alignment gaps in file | **Yes** |
| Three PT_LOAD segments (R + RX + RW) | Maximum permission granularity; matches some optimized toolchains | More complex grouping; additional alignment gaps; minimal benefit for simple programs | No |

### Common Pitfalls

⚠️ **Pitfall: Incorrect Page Alignment**
- **Description**: Setting `p_vaddr` or `p_offset` to non-page-aligned addresses (not multiples of 4096).
- **Why It's Wrong**: The OS loader uses `mmap` to load segments, which requires page-aligned offsets. Non-aligned addresses cause `mmap` to fail or map incorrect memory regions.
- **Fix**: Always use `align_to(address, 4096)` when setting `p_vaddr`, `p_paddr`, and `p_offset`. Ensure the alignment is applied consistently across all segment calculations.

⚠️ **Pitfall: Wrong Entry Point Address**
- **Description**: Setting `e_entry` to the address of `main` instead of `_start`.
- **Why It's Wrong**: The entry point is where the OS transfers control after loading. On Linux with glibc, `_start` is the actual entry point that initializes the C runtime before calling `main`. Directly jumping to `main` skips critical initialization.
- **Fix**: Look up the symbol `_start` in the resolved `SymbolTable` and use its virtual address. If `_start` is not defined (e.g., when linking without libc), use the beginning of the `.text` section as a fallback, but document this limitation clearly.

⚠️ **Pitfall: Mixing Segment Permissions Incorrectly**
- **Description**: Placing a writable section (like `.data`) in the text segment (RX) or an executable section (`.text`) in the data segment (RW).
- **Why It's Wrong**: This violates the permission separation that segments are designed to enforce. The OS will apply the segment's flags to all pages in that range, potentially making data executable or code writable.
- **Fix**: Group sections strictly by their flags: sections with `SHF_EXECINSTR` go to text segment; sections with `SHF_WRITE` go to data segment; sections with only `SHF_ALLOC` (like `.rodata`) typically go to text segment (read-only).

⚠️ **Pitfall: Mishandling .bss in File Size**
- **Description**: Including `.bss` size in `p_filesz` or writing `.bss` zeros to the output file.
- **Why It's Wrong**: `.bss` sections have type `SHT_NOBITS`, meaning they occupy memory but no file space. Including them in the file wastes disk space and causes the loader to misinterpret the segment layout.
- **Fix**: Set `p_filesz` to only include sections with actual file content. Set `p_memsz` to include all sections (including `.bss`). When writing segment data to file, skip the portions corresponding to `.bss`.

⚠️ **Pitfall: Endianness Mismatch in Header Writing**
- **Description**: Writing multi-byte header fields in host byte order instead of the target ELF's specified endianness.
- **Why It's Wrong**: ELF files specify their endianness in `e_ident[EI_DATA]`. x86-64 uses little-endian. Writing in host order (which may be big-endian on some systems) produces unreadable files.
- **Fix**: Use dedicated helper functions `write_le32` and `write_le64` for all 32-bit and 64-bit values when writing to the output file, regardless of host architecture.

### Implementation Guidance

#### Technology Recommendations Table

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| File I/O | Standard C library (`fopen`, `fwrite`, `fclose`) | Memory-mapped file I/O (`mmap`, `munmap`) for large executables |
| Endian Handling | Manual byte swapping using helper functions | Compiler intrinsics (`__builtin_bswap32/64`) with preprocessor detection |
| Address Calculation | Straightforward loops and accumulators | More sophisticated layout managers with constraint satisfaction |

#### Recommended File/Module Structure

The Executable Writer fits into the project structure as follows:

```
static-linker/
├── include/
│   ├── elf_reader.h        # ELF structure definitions
│   ├── section_merger.h    # MergedSections type
│   ├── symbol_resolver.h   # SymbolTable type
│   └── executable_writer.h # Executable Writer interface
├── src/
│   ├── main.c              # Command-line interface
│   ├── elf_reader.c
│   ├── section_merger.c
│   ├── symbol_resolver.c
│   ├── relocation_applier.c
│   └── executable_writer.c # This component's implementation
└── test/
    └── test_programs/      # Simple C programs for validation
```

#### Infrastructure Starter Code

Here is a complete, ready-to-use implementation of the byte ordering helpers and alignment utility required by the Executable Writer:

```c
/* include/elf_common.h - Common ELF utilities */
#ifndef ELF_COMMON_H
#define ELF_COMMON_H

#include <stdint.h>
#include <stddef.h>

/* Alignment helper - aligns value upward to the given boundary */
static inline uint64_t align_to(uint64_t value, uint64_t alignment) {
    if (alignment == 0) return value;
    uint64_t remainder = value % alignment;
    if (remainder == 0) return value;
    return value + (alignment - remainder);
}

/* Write 32-bit value in little-endian order */
static inline void write_le32(uint8_t* ptr, uint32_t value) {
    ptr[0] = (uint8_t)(value & 0xFF);
    ptr[1] = (uint8_t)((value >> 8) & 0xFF);
    ptr[2] = (uint8_t)((value >> 16) & 0xFF);
    ptr[3] = (uint8_t)((value >> 24) & 0xFF);
}

/* Write 64-bit value in little-endian order */
static inline void write_le64(uint8_t* ptr, uint64_t value) {
    write_le32(ptr, (uint32_t)(value & 0xFFFFFFFF));
    write_le32(ptr + 4, (uint32_t)(value >> 32));
}

/* Read 32-bit value in little-endian order */
static inline uint32_t read_le32(const uint8_t* ptr) {
    return (uint32_t)ptr[0] | ((uint32_t)ptr[1] << 8) |
           ((uint32_t)ptr[2] << 16) | ((uint32_t)ptr[3] << 24);
}

/* Read 64-bit value in little-endian order */
static inline uint64_t read_le64(const uint8_t* ptr) {
    uint64_t lo = read_le32(ptr);
    uint64_t hi = read_le32(ptr + 4);
    return (hi << 32) | lo;
}

/* Check if a value fits in signed 32-bit range (for PC-relative relocations) */
static inline int fits_in_int32(int64_t value) {
    return value >= INT32_MIN && value <= INT32_MAX;
}

#endif /* ELF_COMMON_H */
```

#### Core Logic Skeleton Code

Below is the skeleton implementation for the Executable Writer's core functions. The TODOs map directly to the algorithm steps described in the Internal Behavior section.

```c
/* src/executable_writer.c - Core implementation */

#include "executable_writer.h"
#include "elf_common.h"
#include "section_merger.h"
#include "symbol_resolver.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Internal structure for OutputExecutable (opaque in header) */
struct OutputExecutable {
    Elf64_Ehdr header;
    Elf64_Phdr* phdrs;
    uint16_t num_phdrs;
    uint8_t** segment_data;
    size_t* segment_sizes;
    uint64_t entry_point;
    uint64_t text_segment_vaddr;
    uint64_t data_segment_vaddr;
};

/* Helper: Determine segment for a section based on flags */
static int section_belongs_in_text_segment(const OutputSection* sect) {
    // TODO: Return 1 if section should go in text (RX) segment
    // Sections with SHF_EXECINSTR go in text segment
    // Sections with only SHF_ALLOC (like .rodata) also go in text segment
    // Sections with SHF_WRITE go in data segment
    return (sect->sh_flags & SHF_EXECINSTR) || 
           ((sect->sh_flags & (SHF_ALLOC | SHF_WRITE)) == SHF_ALLOC);
}

OutputExecutable* generate_executable(const MergedSections* merged,
                                      const SymbolTable* symbols) {
    OutputExecutable* exec = xcalloc(1, sizeof(OutputExecutable));
    
    // TODO 1: Plan segments - group output sections by permissions
    // Iterate through merged->sections, count sections for each segment
    // Determine total file size and memory size for text segment
    // Determine total file size and memory size for data segment
    
    // TODO 2: Calculate virtual addresses and file offsets
    // Start text segment at conventional base (e.g., 0x400000)
    // Align text segment start to page boundary (4096)
    // Calculate data segment start after text segment with page alignment
    // Calculate file offsets with page alignment
    
    // TODO 3: Find entry point symbol (_start)
    // Look up "_start" in symbol table using hash_table_lookup
    // If found, get its virtual address (segment base + offset)
    // If not found, use beginning of .text section as fallback
    // Set exec->entry_point
    
    // TODO 4: Construct ELF header
    // Set e_ident magic numbers and identification fields
    // Set e_type = ET_EXEC, e_machine = EM_X86_64
    // Set e_entry to entry point address
    // Set e_phoff = sizeof(Elf64_Ehdr)
    // Set e_phentsize = sizeof(Elf64_Phdr), e_phnum = 2
    // Set other fields appropriately
    
    // TODO 5: Construct program headers (2 PT_LOAD entries)
    // Text segment: p_type = PT_LOAD, p_flags = PF_R | PF_X
    // Data segment: p_type = PT_LOAD, p_flags = PF_R | PF_W
    // Set p_offset, p_vaddr, p_filesz, p_memsz, p_align for each
    
    // TODO 6: Package section data into segment buffers
    // Allocate buffers for each segment
    // Copy section data into buffers, maintaining intra-segment offsets
    // For data segment, leave space for .bss (don't write zeros)
    
    return exec;
}

void write_executable(const OutputExecutable* exec, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        fatal_error("Failed to open output file: %s", filename);
    }
    
    // TODO 1: Write ELF header
    // Create a buffer for the header
    // Use write_le32/write_le64 for all multi-byte fields
    // Write e_ident byte-by-byte
    // Write the buffer to file
    
    // TODO 2: Write program headers
    // Iterate through exec->phdrs
    // Serialize each Elf64_Phdr using little-endian writes
    // Write to file immediately after ELF header
    
    // TODO 3: Write segment data at correct file offsets
    // For each segment:
    //   Calculate padding needed to reach p_offset
    //   Write zeros as padding
    //   Write segment data (exec->segment_data[i], exec->segment_sizes[i])
    
    fclose(f);
}

void free_output_executable(OutputExecutable* exec) {
    if (!exec) return;
    
    // TODO: Free all allocated resources
    // Free program header array
    // Free each segment data buffer
    // Free segment sizes array
    // Free the OutputExecutable structure itself
}
```

#### Language-Specific Hints

- **File I/O**: Use `fopen` with mode `"wb"` (write binary) to ensure correct handling on Windows as well as Unix-like systems. Always check the return value.
- **Memory Management**: Since the Executable Writer creates several dynamic allocations (segment buffers, header arrays), implement careful cleanup in `free_output_executable` to avoid memory leaks.
- **Error Handling**: Use the provided `fatal_error` function for unrecoverable errors (file I/O failures, allocation failures). For logic errors (e.g., missing `_start` symbol), consider printing a warning but continuing with a reasonable default.
- **Portability**: The `write_le32`/`write_le64` functions work correctly on both little-endian and big-endian hosts because they always write in little-endian order. This matches the x86-64 ELF specification.

#### Milestone Checkpoint

After implementing the Executable Writer, verify correctness with these steps:

1. **Build a test program**:
   ```c
   /* test_programs/hello.c */
   const char* message = "Hello, World!\n";
   
   void _start() {
       // Write message to stdout (system call without libc)
       asm volatile (
           "mov $1, %%rax\n"   // syscall number for write
           "mov $1, %%rdi\n"   // file descriptor 1 (stdout)
           "mov %0, %%rsi\n"   // message pointer
           "mov $14, %%rdx\n"  // message length
           "syscall\n"
           "mov $60, %%rax\n"  // syscall number for exit
           "xor %%rdi, %%rdi\n" // exit code 0
           "syscall\n"
           :: "r"(message) : "%rax", "%rdi", "%rsi", "%rdx"
       );
   }
   ```

2. **Compile and link**:
   ```bash
   # Compile to object file (using system compiler)
   gcc -c -fno-pie -no-pie hello.c -o hello.o
   
   # Link with your static linker
   ./static-linker hello.o -o hello
   
   # Check the output
   file hello
   # Should report: "hello: ELF 64-bit LSB executable, x86-64, version 1 (SYSV), statically linked"
   
   readelf -l hello
   # Should show 2 PT_LOAD segments with correct permissions and alignment
   
   readelf -h hello | grep Entry
   # Should show entry point address pointing to _start
   ```

3. **Run the executable**:
   ```bash
   ./hello
   # Should print: "Hello, World!"
   ```

**Signs of Success**: The executable runs without segmentation faults and produces the expected output. `readelf` shows proper segment layout with page-aligned addresses and correct permissions (text: R+X, data: R+W).

**Common Failure Modes**:
- **Permission denied**: Check if the text segment has execute permission (`PF_X` flag).
- **Segmentation fault**: Check if entry point is valid and sections are placed at correct virtual addresses. Use `objdump -d hello` to disassemble and verify code placement.
- **File not executable**: Ensure the output file has the executable bit set (may need `chmod +x hello` on some systems).
- **Loader error**: Use `strace ./hello 2>&1 | head -20` to see system calls; look for `mmap` failures which indicate alignment issues.

#### Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Executable crashes immediately | Incorrect entry point | Use `readelf -h` to check `e_entry`; compare to `objdump -t | grep _start` | Ensure `_start` symbol is resolved and address calculated correctly |
| "Permission denied" when running | Missing execute permission on text segment | Use `readelf -l` to check `p_flags` for text segment | Ensure text segment has `PF_X` flag set |
| "Exec format error" | Malformed ELF headers | Use `hexdump -C hello | head -50` to see raw bytes; check magic number | Verify ELF header fields written with correct endianness |
| Data variables contain garbage | Data segment not loaded at correct address | Use `readelf -l` to check data segment `p_vaddr`; compare to relocation calculations | Ensure data segment virtual address matches addresses used in relocations |
| Executable much larger than expected | Incorrect handling of .bss in file | Use `ls -l` to see file size; compare to `readelf -l` output for `p_filesz` | Ensure `.bss` is not included in `p_filesz` and not written to file |
| Text and data overlap in memory | Incorrect segment address calculation | Use `readelf -l` to see segment `p_vaddr` and `p_memsz` values | Ensure segments don't overlap and have proper page-aligned spacing |

---


## Interactions and Data Flow

> **Milestone(s):** Milestones 1-4 (cross-cutting)

The static linker follows a **pipeline architecture** where components transform object files through sequential stages into an executable. Understanding the flow of data and control between components is critical for implementing the linking process correctly. This section provides a detailed view of how components communicate, the sequence of operations, and how raw ELF data evolves into a runnable program.

![Static Linker Component Diagram](./diagrams/component-diagram.svg)

### Linking Sequence

The linking process follows a strict sequential flow where each component depends on outputs from previous components. The overall sequence resembles an assembly line where raw object files enter at one end and a complete executable emerges at the other. This linear progression is necessary because later stages require information computed in earlier stages—for example, relocations cannot be applied until symbols are resolved, and symbol resolution depends on knowing where sections will be placed in the output.

> **Architecture Decision: Strict Sequential Pipeline vs. Iterative Processing**
> - **Context:** The linker must process multiple object files with interdependencies (symbols reference each other across files, relocations depend on symbol addresses).
> - **Options Considered:**
>   1. **Strict sequential pipeline** (chosen): Process all files through each stage in order: read → merge → resolve symbols → apply relocations → write executable.
>   2. **Iterative multi-pass:** Process files iteratively, resolving symbols as they're encountered and adjusting layout dynamically.
> - **Decision:** Strict sequential pipeline with two passes over symbols (collect then resolve).
> - **Rationale:** Simpler to implement and debug, deterministic behavior, matches traditional linker architectures. Iterative approaches are more complex and can lead to non-deterministic outcomes.
> - **Consequences:** Requires storing intermediate data structures for all files simultaneously, but provides clear separation of concerns and predictable execution flow.

The linking sequence unfolds in five distinct phases, each managed by a dedicated component:

| Phase | Component | Primary Input | Primary Output | Key Transformation |
|-------|-----------|---------------|----------------|-------------------|
| 1. Input Loading | ELF Reader | `.o` file bytes on disk | `ObjectFile` structures in memory | Raw ELF binary → parsed sections, symbols, relocations |
| 2. Layout Planning | Section Merger | Multiple `ObjectFile` instances | `MergedSections` with layout | Individual sections → grouped output sections with offsets |
| 3. Symbol Resolution | Symbol Resolver | `ObjectFile` instances + `MergedSections` | `SymbolTable` with resolved addresses | Symbol references → concrete virtual addresses |
| 4. Address Fixup | Relocation Applier | `ObjectFile` instances + `MergedSections` + `SymbolTable` | Patched section data in `MergedSections` | Relocation instructions → patched bytes in output sections |
| 5. Executable Assembly | Executable Writer | `MergedSections` + `SymbolTable` | ELF executable file on disk | In-memory sections → valid ELF binary with headers |

The following numbered steps detail the complete linking sequence:

1. **Initialize linking context**: The main program allocates arrays to hold all `ObjectFile` structures and creates an empty `MergedSections` container.

2. **Load all input object files**: For each input `.o` file specified on the command line:
   - Call `read_elf_file(filename)` which validates the ELF header, parses section headers, loads section data, and extracts symbol and relocation tables.
   - Store the resulting `ObjectFile*` in an array for subsequent processing.
   - Validate that all object files share compatible ELF properties (64-bit, little-endian, x86-64 architecture).

3. **Merge sections from all files**: Pass the array of `ObjectFile` pointers to `merge_all_sections()`:
   - The Section Merger groups allocatable sections (`SHF_ALLOC` flag) by name and type (`.text`, `.data`, `.rodata`, `.bss`).
   - For each group, it concatenates sections with proper alignment padding, tracking each input section's output offset via `InputSectionMapping`.
   - It computes final output section sizes, file offsets, and virtual addresses (starting from a base like `0x400000`).
   - Returns a `MergedSections*` containing the layout plan and empty buffers for section data.

4. **Resolve symbols across files**: Call `resolve_all_symbols()` with the object files and merged sections:
   - First pass: Collect all global symbols (binding `STB_GLOBAL` or `STB_WEAK`) from all object files into a preliminary table.
   - Apply strong/weak resolution rules: Strong symbols override weak symbols; duplicate strong symbols cause an error.
   - Handle COMMON symbols (special uninitialized globals) by allocating space in `.bss` with proper alignment.
   - Assign each defined symbol a final virtual address = `output_section->virtual_addr + offset_in_section`.
   - Second pass: Scan symbol references, ensuring every referenced symbol has a definition (or report undefined symbol error).
   - Return a `SymbolTable*` mapping symbol names to `SymbolEntry` structures with resolved addresses.

5. **Apply relocations to merged data**: Call `apply_all_relocations()` with object files, merged sections, and symbol table:
   - For each object file, iterate through its relocation entries (`ElfRelocation` array).
   - For each relocation, compute the target address in the output using the symbol's resolved address from the `SymbolTable`.
   - Apply relocation-specific calculations (`R_X86_64_64` for absolute, `R_X86_64_PC32` for PC-relative).
   - Patch the computed value into the appropriate location in the merged section data buffer.
   - Validate that 32-bit relocations don't overflow (values fit in signed 32-bit range).

6. **Generate executable structure**: Call `generate_executable()` with merged sections and symbol table:
   - Determine entry point address by looking up the `_start` symbol in the symbol table (or fallback to `.text` start).
   - Create program headers (`Elf64_Phdr`): at least two `PT_LOAD` segments (text/RX and data/RW) with page-aligned boundaries.
   - Set segment permissions: text segment = `PF_R | PF_X`, data segment = `PF_R | PF_W`.
   - Calculate `p_filesz` and `p_memsz` accounting for `.bss` (file size zero but memory size non-zero).
   - Assemble the complete `OutputExecutable` structure with headers and section data.

7. **Write executable to disk**: Call `write_executable()` with the `OutputExecutable` and output filename:
   - Write the ELF header at file offset 0.
   - Write program headers immediately after the ELF header.
   - Write section data at their calculated file offsets.
   - Ensure the file is properly closed with correct permissions (executable bit set).

8. **Cleanup resources**: Free all allocated structures: `ObjectFile`s, `MergedSections`, `SymbolTable`, and `OutputExecutable`.

The sequence diagram below illustrates this flow visually:

![Linking Process Sequence Diagram](./diagrams/linking-sequence.svg)

### Data Transformations

Throughout the linking pipeline, data undergoes several structural transformations. Understanding these transformations is key to debugging and implementing each component correctly.

#### Phase 1: From Raw Bytes to Structured ObjectFile

The ELF Reader transforms raw file bytes into structured in-memory representations:

| Input | Transformation Process | Output |
|-------|------------------------|--------|
| Raw ELF file bytes | Parse `Elf64_Ehdr`, validate magic numbers, endianness, architecture | Validated ELF header |
| Section header table bytes | Parse array of `Elf64_Shdr`, locate section names via `.shstrtab` | Array of `ElfSection` with headers |
| Section content bytes | Copy `.text`, `.data`, `.rodata` sections to memory buffers; `.bss` gets zero size | `ElfSection.data` pointers with loaded content |
| Symbol table bytes | Parse `Elf64_Sym` entries, resolve symbol names via `.strtab` | Array of `ElfSymbol` with name pointers |
| Relocation table bytes | Parse `Elf64_Rela` entries, link to target sections and symbols | Array of `ElfRelocation` with references |

**Concrete Example Walkthrough:** Consider a simple object file `main.o` compiled from:
```c
extern int shared;
int main() { return shared + 1; }
```

The `ObjectFile` for `main.o` would contain:
- A `.text` section with compiled machine code containing a `mov` instruction that references `shared`
- A symbol table with: `main` (defined, `STB_GLOBAL`, `STT_FUNC`), `shared` (undefined, `STB_GLOBAL`)
- A relocation entry: `R_X86_64_32` targeting the offset of `shared` reference in `.text`

#### Phase 2: From Individual Sections to Merged Layout

The Section Merger transforms arrays of individual sections into a unified layout:

| Input | Transformation Process | Output |
|-------|------------------------|--------|
| Multiple `ElfSection` arrays from different files | Group by name/type, sort by alignment requirements | `OutputSectionGroup` structures |
| Individual section sizes and alignments | Concatenate with padding: `offset = align_to(prev_offset + prev_size, current_alignment)` | `OutputSection` with `size` and `file_offset` |
| Input section references | Create mapping: `(file_index, section_index) → (output_section, output_offset)` | `InputSectionMapping` array |
| `.bss` sections (type `SHT_NOBITS`) | Account for memory size but not file size | `OutputSection` with `size > 0` but `data = NULL` |

**Data Structure Evolution:** An `OutputSection` representing `.text` might contain:
- `size = 0x150` (sum of three `.text` sections from different files plus padding)
- `data =` concatenated bytes from all three input `.text` sections
- `virtual_addr = 0x4001e0` (calculated base address)
- `sh_addralign = 16` (maximum alignment of input sections)

#### Phase 3: From Symbol References to Resolved Addresses

The Symbol Resolver transforms symbolic references into concrete virtual addresses:

| Input | Transformation Process | Output |
|-------|------------------------|--------|
| `ElfSymbol` arrays from all files | Filter for global symbols (`STB_GLOBAL`, `STB_WEAK`) | Preliminary symbol list |
| Symbol definitions with section indices | Resolve section-relative offsets to absolute addresses using `MergedSections` mappings | `SymbolEntry` with `value = virtual_addr` |
| Multiple definitions of same name | Apply strong/weak rules: strong wins, duplicate strong = error | Single `SymbolEntry` per name |
| COMMON symbols (`SHN_COMMON`) | Allocate space in `.bss` with size = max of all COMMON definitions | `SymbolEntry` pointing to `.bss` offset |
| Undefined symbol references | Check all references have definitions; report errors otherwise | Complete `SymbolTable` or error |

**Concrete Example:** With `main.o` and `lib.o` (defining `shared`), resolution produces:
- `main` → address `0x4001e0` (start of `.text` from `main.o`)
- `shared` → address `0x600120` (offset in `.data` from `lib.o`)
- All references to `shared` now point to `0x600120`

#### Phase 4: From Relocation Instructions to Patched Bytes

The Relocation Applier transforms relocation metadata into actual byte modifications:

| Input | Transformation Process | Output |
|-------|------------------------|--------|
| `ElfRelocation` entries | Look up target symbol in `SymbolTable` for resolved address | Symbol value (`S`) |
| Relocation type and addend | Compute value: `V = S + A` (absolute) or `V = S + A - P` (PC-relative) | Calculated patch value |
| Target offset in section | Locate patch location: `patch_addr = section_base + r_offset` | Pointer to bytes in merged data |
| Patch width (32/64-bit) | Write value in little-endian format, check for overflow | Modified section bytes |

**Relocation Calculation Examples:**
- `R_X86_64_64`: `V = S + A` written as 64-bit absolute address at patch location
- `R_X86_64_PC32`: `V = S + A - P` written as 32-bit signed offset from next instruction

Where:
- `S` = symbol address (from `SymbolTable`)
- `A` = addend (from `r_addend`)
- `P` = relocation site address (where the relocation is applied)

#### Phase 5: From Merged Sections to Executable File

The Executable Writer transforms in-memory structures to disk format:

| Input | Transformation Process | Output |
|-------|------------------------|--------|
| `OutputSection` array | Group into segments: executable sections → text segment, writable → data segment | `Elf64_Phdr` entries |
| Section virtual addresses | Align to page boundaries (4096), ensure no overlap | Segment `p_vaddr` and `p_paddr` |
| Section file data | Write to contiguous file regions, leaving gaps for alignment | File bytes on disk |
| Entry point symbol | Look up `_start` address or use `.text` start | `e_entry` in ELF header |

**Memory Layout Example:**
```
Text segment (RX): 0x400000-0x401000
  .text at 0x4001e0 (size 0x150)
  .rodata at 0x400330 (size 0x80)

Data segment (RW): 0x600000-0x601000  
  .data at 0x600000 (size 0x200)
  .bss at 0x600200 (size 0x100) - zero-initialized in memory
```

#### Data Flow Dependencies

Critical data dependencies exist between components that must be respected:

| Dependency | From Component | To Component | Data Passed | Timing |
|------------|----------------|--------------|-------------|--------|
| Section layout | Section Merger | Symbol Resolver | `MergedSections*` with `InputSectionMapping` | Before symbol resolution |
| Symbol addresses | Symbol Resolver | Relocation Applier | `SymbolTable*` with `value` fields | Before relocation application |
| Patched data | Relocation Applier | Executable Writer | `MergedSections.data` buffers with fixes | Before executable generation |
| Entry point | Symbol Resolver | Executable Writer | Address of `_start` symbol | During executable generation |

> **Key Insight:** The pipeline is deliberately designed with unidirectional data flow—each component only depends on outputs from previous components, never on later components. This makes the system easier to reason about and debug, as issues can be isolated to specific stages.

#### Error Propagation Through the Pipeline

Errors detected at later stages often originate from earlier stages. The pipeline design enables early error detection:

| Error Type | Likely Origin | Detection Point | Prevention Strategy |
|------------|---------------|-----------------|---------------------|
| Undefined symbol | Missing object file | Symbol Resolver | Comprehensive symbol collection in first pass |
| Relocation overflow | Incorrect symbol address calculation | Relocation Applier | Validate 32-bit range before patching |
| Section flag mismatch | Incompatible object files | Section Merger | Validate `sh_flags` when grouping sections |
| Alignment impossible | Extremely large alignment requirement | Section Merger | Check alignment <= page size (4096) |

### Implementation Guidance

#### Technology Recommendations Table

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Data Flow Control | Sequential function calls in `main()` | Pipeline with context structure passed between stages |
| Error Propagation | Return error codes, centralized error handler | Structured error types with context information |
| Intermediate Data Storage | Global variables for current linking state | Context structure passed to each component |
| Memory Management | Manual `malloc`/`free` with careful tracking | Arena allocator for temporary linking data |

#### Recommended File/Module Structure

```
static-linker/
├── include/
│   ├── linker.h           # Main public API for the linker
│   ├── elf_reader.h       # ELF Reader component interface
│   ├── section_merger.h   # Section Merger component interface  
│   ├── symbol_resolver.h  # Symbol Resolver component interface
│   ├── relocation_applier.h # Relocation Applier component interface
│   └── executable_writer.h # Executable Writer component interface
├── src/
│   ├── main.c             # Command-line interface and orchestration
│   ├── elf_reader.c       # ELF Reader implementation
│   ├── section_merger.c   # Section Merger implementation
│   ├── symbol_resolver.c  # Symbol Resolver implementation
│   ├── relocation_applier.c # Relocation Applier implementation
│   ├── executable_writer.c # Executable Writer implementation
│   └── utils/
│       ├── alloc.c        # xmalloc, xcalloc, xrealloc helpers
│       ├── align.c        # align_to utility function
│       └── error.c        # fatal_error and warning functions
└── tests/
    ├── test_programs/     # Simple C programs for testing
    └── integration.c      # End-to-end tests
```

#### Infrastructure Starter Code

The main orchestration file that implements the linking sequence:

```c
/* src/main.c - Main linker orchestration */
#include <stdio.h>
#include <stdlib.h>
#include "linker.h"
#include "elf_reader.h"
#include "section_merger.h"
#include "symbol_resolver.h"
#include "relocation_applier.h"
#include "executable_writer.h"
#include "utils/alloc.h"
#include "utils/error.h"

/* Context structure holding all linking state */
typedef struct LinkContext {
    ObjectFile** objects;
    uint32_t object_count;
    MergedSections* merged;
    SymbolTable* symbols;
    OutputExecutable* executable;
} LinkContext;

static LinkContext* create_link_context(void) {
    LinkContext* ctx = xmalloc(sizeof(LinkContext));
    ctx->objects = NULL;
    ctx->object_count = 0;
    ctx->merged = NULL;
    ctx->symbols = NULL;
    ctx->executable = NULL;
    return ctx;
}

static void free_link_context(LinkContext* ctx) {
    if (!ctx) return;
    
    // Free in reverse order of creation
    if (ctx->executable) free_output_executable(ctx->executable);
    if (ctx->symbols) free_symbol_table(ctx->symbols);
    if (ctx->merged) free_merged_sections(ctx->merged);
    
    for (uint32_t i = 0; i < ctx->object_count; i++) {
        if (ctx->objects[i]) free_object_file(ctx->objects[i]);
    }
    free(ctx->objects);
    free(ctx);
}

/* Main linking function implementing the sequence */
int link_files(const char** input_files, uint32_t file_count, 
               const char* output_file) {
    LinkContext* ctx = create_link_context();
    int result = 0;
    
    // Phase 1: Load all input files
    printf("Loading %u object files...\n", file_count);
    ctx->objects = xcalloc(file_count, sizeof(ObjectFile*));
    ctx->object_count = file_count;
    
    for (uint32_t i = 0; i < file_count; i++) {
        ctx->objects[i] = read_elf_file(input_files[i]);
        if (!ctx->objects[i]) {
            fatal_error("Failed to read ELF file: %s\n", input_files[i]);
        }
    }
    
    // Phase 2: Merge sections
    printf("Merging sections...\n");
    ctx->merged = merge_all_sections(ctx->objects, file_count);
    if (!ctx->merged) {
        fatal_error("Failed to merge sections\n");
    }
    
    // Phase 3: Resolve symbols
    printf("Resolving symbols...\n");
    ctx->symbols = resolve_all_symbols(ctx->objects, file_count, ctx->merged);
    if (!ctx->symbols) {
        fatal_error("Failed to resolve symbols\n");
    }
    
    // Check for undefined symbols
    report_undefined(ctx->symbols);
    
    // Phase 4: Apply relocations
    printf("Applying relocations...\n");
    apply_all_relocations(ctx->objects, file_count, ctx->merged, ctx->symbols);
    
    // Phase 5: Generate executable
    printf("Generating executable...\n");
    ctx->executable = generate_executable(ctx->merged, ctx->symbols);
    if (!ctx->executable) {
        fatal_error("Failed to generate executable\n");
    }
    
    // Phase 6: Write to disk
    printf("Writing executable to: %s\n", output_file);
    write_executable(ctx->executable, output_file);
    
    // Set executable permission
    chmod(output_file, 0755);
    
    printf("Linking completed successfully!\n");
    free_link_context(ctx);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <output> <input1.o> [input2.o ...]\n", argv[0]);
        return 1;
    }
    
    const char* output_file = argv[1];
    const char** input_files = (const char**)&argv[2];
    uint32_t file_count = argc - 2;
    
    return link_files(input_files, file_count, output_file);
}
```

#### Core Logic Skeleton Code

The core linking orchestration with TODO markers for key steps:

```c
/* src/linker.c - Core linking logic (alternative to main.c) */
#include "linker.h"

LinkResult link_multiple_files(const char** inputs, uint32_t count, 
                               const char* output) {
    // TODO 1: Validate input parameters (count > 0, inputs not NULL)
    // TODO 2: Create arrays to hold ObjectFile pointers for all inputs
    // TODO 3: For each input file, call read_elf_file() and store result
    // TODO 4: Verify all object files have compatible architecture (x86-64)
    // TODO 5: Call merge_all_sections() with object array and count
    // TODO 6: Check merge_all_sections() returned non-NULL MergedSections
    // TODO 7: Call resolve_all_symbols() with objects, count, and merged sections
    // TODO 8: If resolve_all_symbols() returns NULL, report undefined symbols
    // TODO 9: Call apply_all_relocations() with objects, count, merged, symbols
    // TODO 10: Call generate_executable() with merged sections and symbols
    // TODO 11: Call write_executable() with the generated executable and output path
    // TODO 12: Clean up all allocated resources in reverse creation order
    // TODO 13: Return success/failure indication
}
```

#### Language-Specific Hints (C)

1. **Memory Management:** Use wrapper functions `xmalloc`, `xcalloc`, `xrealloc` that call `fatal_error` on allocation failure to simplify error handling.

2. **Error Propagation:** Consider returning `NULL` for functions that can fail, with error messages printed inside the failing function.

3. **Context Passing:** Create a `LinkContext` structure that holds all intermediate state to avoid global variables.

4. **File I/O:** Use `fopen` with `"rb"` for reading object files and `"wb"` for writing the executable.

5. **Endianness:** Since we're targeting x86-64 (little-endian), use helper functions `read_le32`, `write_le32`, etc., for consistent byte order handling.

#### Milestone Checkpoint

After implementing the complete linking sequence, test with a simple multi-file program:

1. **Create test files:**
```c
/* test1.c */
extern int shared_var;
int main() { return shared_var + 1; }

/* test2.c */
int shared_var = 42;
```

2. **Compile and link:**
```bash
gcc -c test1.c -o test1.o
gcc -c test2.c -o test2.o
./linker test.bin test1.o test2.o
```

3. **Verify the executable:**
```bash
chmod +x test.bin
./test.bin
echo $?  # Should print 43 (42 + 1)
readelf -h test.bin  # Should show ELF64 executable
objdump -d test.bin  # Should show patched machine code
```

#### Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Executable crashes immediately | Wrong entry point address | Use `readelf -h` to check `e_entry`, compare with `_start` address | Ensure entry point points to `_start` symbol or `.text` start |
| Relocated values incorrect | Symbol addresses wrong | Dump symbol table with `nm` equivalent, check calculations | Verify symbol resolution uses correct output section base |
| Segments overlap in memory | Incorrect segment size calculations | Use `readelf -l` to examine program headers, check `p_vaddr` and `p_memsz` | Ensure page alignment and non-overlapping ranges |
| Relocation overflow error | 64-bit address in 32-bit field | Check symbol is within ±2GB of relocation site for PC-relative | Rearrange section layout or use different relocation type |
| Undefined symbol error | Missing object file | List all symbols with `--print-symbols` debug flag | Add required object file or library |
| Section data corrupted | Incorrect padding during merge | Hexdump merged section data, compare with original | Fix alignment padding calculation in `merge_all_sections` |


## Error Handling and Edge Cases

> **Milestone(s):** Milestones 1-4 (cross-cutting)

The **Error Handling and Edge Cases** design is the safety net of the static linker architecture. While the core linking pipeline focuses on successful transformations, real-world scenarios inevitably present malformed inputs, conflicting definitions, and resource constraints that must be gracefully managed. This section systematically categorizes failure modes, defines detection strategies, and establishes recovery approaches that maintain diagnostic clarity without sacrificing architectural simplicity.

Effective error handling in a linker is fundamentally a **diagnostic-first philosophy**: rather than silently continuing with potentially corrupted output, the linker must detect issues early, provide actionable error messages with precise context, and terminate cleanly before producing invalid executables. The architectural challenge lies in balancing comprehensive validation with performance, and providing sufficient debugging context without leaking implementation details.

### Mental Model: Airport Security Screening

Think of the linker's error handling as an airport security screening process. Each component acts as a specialized checkpoint examining different aspects of travel documents (object files) and luggage (section data):

1. **Document Validation Checkpoint (ELF Reader)**: Verifies passports (ELF headers) are authentic and properly formatted before allowing entry
2. **Luggage Consolidation Checkpoint (Section Merger)**: Ensures prohibited items (incompatible sections) aren't merged and that luggage dimensions (alignment) meet requirements
3. **Passenger Manifest Checkpoint (Symbol Resolver)**: Cross-references all passenger names (symbols) for duplicates, missing passengers, and conflicting reservations
4. **Address Labeling Checkpoint (Relocation Applier)**: Verifies all forwarding addresses (relocation targets) exist and can be properly labeled
5. **Flight Manifest Checkpoint (Executable Writer)**: Ensures the final flight plan (executable layout) meets all safety regulations (ELF specifications)

Just as security checkpoints must provide clear reasons for rejection ("your passport is expired" rather than "document invalid"), each linker component must emit specific, contextual error messages that guide developers to the root cause in their source code or build configuration.

### Error Categories

The linker's error conditions can be systematically classified into four orthogonal categories based on their origin in the linking pipeline and severity. Each category has distinct detection strategies, recovery possibilities, and diagnostic requirements.

| Category | Origin Phase | Typical Causes | Severity | Recovery Possibility |
|----------|--------------|----------------|----------|----------------------|
| **Input Validation Errors** | ELF Reading | Corrupted/malformed ELF headers, unsupported architectures, file I/O failures, missing sections | Fatal | None – cannot proceed with invalid inputs |
| **Symbol Resolution Errors** | Symbol Resolution | Undefined symbols, multiple strong definitions, incompatible symbol types, COMMON symbol size conflicts | Fatal | None – semantic contradictions in program |
| **Relocation Errors** | Relocation Application | Address overflow in truncated fields, misaligned relocation sites, invalid symbol references, PC-relative out-of-range | Fatal | None – would produce incorrect code |
| **Layout & Resource Errors** | Section Merging & Executable Generation | Alignment impossible within address space, segment overflow beyond 32-bit range, memory exhaustion during processing | Fatal | None – physical/logical constraints violated |
| **Warning Conditions** | All phases | Weak symbol overridden, size mismatches in COMMON merging, unusual but valid alignment requirements | Non-fatal | Continue with diagnostic messages |

> **Key Insight:** The linker adopts a **fail-fast, fail-cleanly** philosophy. Unlike compilers that can sometimes generate placeholder code for missing symbols, a linker has no safe fallback for unresolved references or malformed binaries. Early detection with clear diagnostics is paramount, as late-stage errors in executables manifest as cryptic segmentation faults at runtime rather than build-time messages.

#### Input Validation Errors
These errors occur during the initial parsing of object files and represent fundamental violations of the ELF format specification or system constraints.

| Failure Mode | Detection Strategy | Diagnostic Context Needed |
|--------------|-------------------|---------------------------|
| Invalid ELF magic bytes | Check first 4 bytes against `{0x7F, 'E', 'L', 'F'}` | Filename, offset, actual byte values |
| Unsupported architecture (`e_machine != EM_X86_64`) | Validate `e_machine` field | Filename, actual architecture value, supported architectures |
| Endianness mismatch (`e_ident[EI_DATA] != ELFDATA2LSB`) | Check endianness field | Filename, actual endianness, expected endianness |
| Corrupted section header table | Verify `e_shoff + e_shnum*e_shentsize` ≤ file size | Filename, calculated offset, file size |
| Missing required sections (.text, .symtab) | Scan sections for required types by linking phase | Filename, missing section type, phase requirement |
| File I/O failures (permission denied, not found) | Check return values of `fopen()`, `fread()`, `fseek()` | Filename, `errno` value, operation attempted |

#### Symbol Resolution Errors
These errors represent semantic contradictions in the program being linked, where symbolic references cannot be consistently resolved.

| Failure Mode | Detection Strategy | Diagnostic Context Needed |
|--------------|-------------------|---------------------------|
| Undefined symbol reference | Symbol with `STB_GLOBAL` binding has `shndx == SHN_UNDEF` after all files processed | Symbol name, referencing object file, location in source (if debug info available) |
| Multiple strong definitions | Two or more symbols with same name, both with `STB_GLOBAL` binding and non-weak definitions | Symbol name, defining object files (all occurrences), symbol types |
| Incompatible symbol types | Definition and reference have mismatched types (`STT_FUNC` vs `STT_OBJECT`) | Symbol name, expected type, actual type, locations |
| COMMON symbol size conflict | Multiple COMMON symbols with different sizes where largest doesn't satisfy all references | Symbol name, sizes from each file, maximum size required |
| Weak symbol definition after strong | Weak definition ignored due to earlier strong definition (warning only) | Symbol name, weak definition location, strong definition location |

#### Relocation Errors
These errors occur when address calculations or memory patching cannot be correctly performed due to mathematical or semantic constraints.

| Failure Mode | Detection Strategy | Diagnostic Context Needed |
|--------------|-------------------|---------------------------|
| Truncation overflow in 32-bit field | Check if computed `symbol_addr + addend - base` fits in `int32_t` range | Relocation type, symbol name, computed value, valid range |
| PC-relative out of range | For `R_X86_64_PC32`, verify displacement fits in signed 32-bit range | Symbol name, relocation offset, target section, distance |
| Invalid symbol for relocation type | Absolute relocation to undefined symbol or wrong symbol type | Relocation type, symbol name, symbol binding/type |
| Misaligned relocation site | Check `r_offset` alignment matches relocation requirements (e.g., 4-byte for `R_X86_64_PC32`) | Relocation type, offset value, required alignment |
| Section-relative relocation to wrong section | Symbol's section doesn't match relocation's expected target | Symbol section, relocation section, relocation type requirements |

#### Layout & Resource Errors
These errors represent physical or logical constraints that cannot be satisfied during memory layout or resource allocation.

| Failure Mode | Detection Strategy | Diagnostic Context Needed |
|--------------|-------------------|---------------------------|
| Alignment impossible | Cannot satisfy `sh_addralign` requirement within address space constraints | Section name, alignment requirement, current address |
| Address space overflow | Section or segment extends beyond 32-bit or 64-bit address limits | Section/segment name, end address, address space limit |
| Segment size exceeds page boundaries | `p_memsz` causes segment to span non-contiguous pages incorrectly | Segment type, memory size, page boundary crossings |
| Memory exhaustion during processing | `malloc()` or `calloc()` returns `NULL` | Allocation size, current heap usage, operation context |
| Entry point not in any loadable segment | `e_entry` address falls outside all `PT_LOAD` segments | Entry point address, segment ranges, segment permissions |

> **Architecture Decision: Error Handling Strategy**

> **Decision: Fail-Fast with Contextual Diagnostics**
> - **Context**: The linker processes multiple object files through sequential transformations where later stages depend on earlier stages being correct. Errors can cascade, making root cause diagnosis difficult if processing continues.
> - **Options Considered**:
>   1. **Fail-Fast with Abort**: Immediately terminate on first error with detailed message
>   2. **Continue with Best-Effort**: Attempt to continue processing, collect multiple errors, then abort
>   3. **Recovery with Placeholders**: Generate placeholder values for missing symbols and continue
> - **Decision**: Option 1 (Fail-Fast with Abort) with enhanced contextual diagnostics
> - **Rationale**: 
>   - Linker errors are fundamental semantic issues in the program; continuing produces unusable executables
>   - Cascading errors from a single root cause produce confusing diagnostic spam
>   - Placeholder generation would produce executables with undefined behavior at runtime
>   - Early termination provides faster feedback in development cycles
> - **Consequences**:
>   - Simplifies implementation (no need to maintain partial state after errors)
>   - Requires excellent first-error diagnostics since only one error is reported
>   - May frustrate users who want to see all errors at once (mitigated by clear messages)

| Option | Pros | Cons | Why Not Chosen |
|--------|------|------|----------------|
| **Fail-Fast with Abort** | Simple implementation, clear causality, prevents wasted processing | Only reports first error, may require multiple rebuilds | **CHOSEN** - Best for educational tool and production use |
| **Continue with Best-Effort** | Collects multiple errors, comprehensive diagnostics | Complex error state management, cascading errors obscure root causes | Too complex for learning project, confusing output |
| **Recovery with Placeholders** | Always produces some output, enables partial testing | Output is semantically incorrect, masks serious bugs | Produces broken executables, violates safety principles |

### Detection and Recovery

The linker implements a layered detection strategy where each component validates its inputs, invariants, and outputs before passing data downstream. Recovery is generally not attempted beyond clean resource deallocation, as linker errors represent fundamental contradictions in the input program.

#### Detection Architecture

Error detection follows a **defense-in-depth** approach with three validation layers:

1. **Syntax Validation** (ELF Reader): Verifies structural correctness of input files
2. **Semantic Validation** (Symbol Resolver): Verifies logical consistency across compilation units
3. **Physical Validation** (Section Merger, Executable Writer): Verifies feasibility within memory and file constraints

Each validation layer uses a consistent error reporting infrastructure that captures:
- Error category and severity
- Component where error occurred
- Specific object file(s) involved
- Symbol/section names when applicable
- Numerical values and constraints
- Source context if debug information available

> **Key Insight:** Error messages should answer not just "what broke" but "why it matters." Instead of "symbol 'foo' undefined," provide "function 'foo' called from main.o but defined nowhere - check spelling and linking order."

#### Error Reporting Interface

The linker uses a centralized error reporting mechanism with the following interface:

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `report_error()` | `ErrorCategory category`, `const char* component`, `const char* format`, `...` | `void` | Formats and prints error message to stderr, increments error count |
| `report_warning()` | `const char* component`, `const char* format`, `...` | `void` | Formats and prints warning to stderr, increments warning count |
| `fatal_error()` | `const char* format`, `...` | `void` (does not return) | Prints error and terminates program immediately |
| `has_errors()` | `void` | `bool` | Returns true if any errors reported since last reset |
| `get_error_count()` | `void` | `int` | Returns total errors reported |

This centralized approach enables consistent formatting, error counting, and potential future extension to machine-readable error output formats.

#### Detection Strategies by Component

**ELF Reader Detection Strategy**:
1. **Magic Number Validation**: Byte-by-byte comparison with ELF signature
2. **Field Range Checking**: Validate `e_shnum`, `e_phnum` against file size limits
3. **Section Header Consistency**: Verify section headers don't overlap, point within file
4. **String Table Integrity**: Check string indices don't exceed table bounds
5. **Symbol Table Coherence**: Verify symbol indices reference valid sections

```c
// Example detection logic for ELF magic bytes
if (header->e_ident[EI_MAG0] != ELFMAG0 ||
    header->e_ident[EI_MAG1] != ELFMAG1 ||
    header->e_ident[EI_MAG2] != ELFMAG2 ||
    header->e_ident[EI_MAG3] != ELFMAG3) {
    report_error(ERROR_INPUT, "ELF Reader", 
                 "%s: not an ELF file (bad magic bytes: 0x%02x%02x%02x%02x)",
                 filename,
                 header->e_ident[0], header->e_ident[1],
                 header->e_ident[2], header->e_ident[3]);
    return NULL;
}
```

**Section Merger Detection Strategy**:
1. **Alignment Compatibility**: Verify merged sections have compatible `sh_addralign` (must be power of two, merged alignment is LCM)
2. **Flag Consistency**: Ensure `SHF_WRITE`, `SHF_EXECINSTR`, `SHF_ALLOC` flags match for all sections in group
3. **Address Space Bounds**: Check cumulative size + alignment padding doesn't exceed addressable range
4. **Section Type Compatibility**: Validate only compatible section types (`SHT_PROGBITS`, `SHT_NOBITS`) merge together

**Symbol Resolver Detection Strategy**:
1. **Two-Pass Collection**: First pass collects all definitions, second pass verifies all references resolvable
2. **Strength Conflict Detection**: Track strongest binding encountered for each symbol, flag conflicts
3. **Type Compatibility Matrix**: Validate symbol types match (function vs data, size compatibility)
4. **COMMON Symbol Merging**: Apply size-based conflict resolution with diagnostic warnings

**Relocation Applier Detection Strategy**:
1. **Range Checking Pre-validation**: Before patching, compute relocation value and verify fits in target field
2. **Symbol State Validation**: Ensure target symbol is defined and has appropriate binding
3. **Alignment Verification**: Check relocation site alignment matches architecture requirements
4. **Overflow Detection**: Mathematical validation of 64→32 bit truncation with sign-extension awareness

**Executable Writer Detection Strategy**:
1. **Segment Boundary Validation**: Verify segments don't overlap, have proper page alignment
2. **Permission Consistency**: Check segment permissions match contained section flags
3. **Entry Point Reachability**: Validate `e_entry` points within executable segment with execute permission
4. **File Offset Alignment**: Ensure program headers and segments follow ELF specification alignment rules

#### Recovery and Cleanup Strategy

Since recovery from linker errors is generally impossible (the input program has fundamental issues), the focus shifts to **graceful degradation**:

1. **Error Propagation**: Once an error is detected, set error state and short-circuit further processing in current component
2. **Resource Cleanup**: All allocated resources (memory, file handles) must be freed before exit
3. **Diagnostic Output**: Provide clear, actionable error message pointing to source of problem
4. **Exit Status**: Return non-zero exit code to signal failure to build system

The cleanup strategy follows **ownership semantics**:
- Each component owns resources it allocates
- `free_*()` functions handle recursive deallocation of nested structures
- Error paths call appropriate cleanup before returning error indication
- Top-level `link_files()` orchestrates cleanup of all components on error

```c
// Example cleanup orchestration in link_files()
int link_files(const char** input_files, int file_count, const char* output_file) {
    ObjectFile** objects = NULL;
    MergedSections* merged = NULL;
    SymbolTable* symbols = NULL;
    OutputExecutable* exec = NULL;
    
    objects = xmalloc(sizeof(ObjectFile*) * file_count);
    for (int i = 0; i < file_count; i++) {
        objects[i] = read_elf_file(input_files[i]);
        if (!objects[i] || has_errors()) {
            // Clean up what we've allocated so far
            for (int j = 0; j < i; j++) free_object_file(objects[j]);
            free(objects);
            return 1; // Error exit code
        }
    }
    
    merged = merge_all_sections(objects, file_count);
    if (!merged || has_errors()) {
        cleanup_all(objects, file_count, merged, symbols, exec);
        return 1;
    }
    
    // ... continue with other phases
    
    // Success path
    cleanup_all(objects, file_count, merged, symbols, exec);
    return 0;
}
```

#### Common Error Scenarios and Diagnostics

| Scenario | Faulty Input Example | Ideal Error Message | Why Better |
|----------|----------------------|---------------------|------------|
| Undefined symbol | `main.o` calls `printf()` but no libc linked | `main.o: undefined reference to 'printf'`<br>`  called from function 'main' at main.c:5`<br>`  hint: add -lc to link with C library` | Points to source location, suggests fix |
| Multiple definitions | `foo.o` and `bar.o` both define `global_var` | `multiple definition of 'global_var'`<br>`  first defined in foo.o (data, 4 bytes)`<br>`  also defined in bar.o (data, 8 bytes)`<br>`  size mismatch suggests different types` | Shows all locations, notes incompatibility |
| Relocation overflow | Large array causes PC-relative jump > ±2GB | `relocation R_X86_64_PC32 overflow against symbol 'large_array'`<br>`  in section .text of foo.o at offset 0x45`<br>`  distance: 0x80000001 (needs < 0x7fffffff)` | Shows exact distance, constraint |
| Bad ELF file | Corrupted section header offset | `foo.o: invalid ELF file: section header at offset 0xffffffff exceeds file size (1024 bytes)`<br>`  file may be corrupted or truncated` | Provides specific offset, suggests cause |

### Common Pitfalls in Error Handling

⚠️ **Pitfall: Silent Truncation in Relocations**
- **Description**: Computing relocation value without checking if it fits in target field, silently truncating high bits
- **Why Wrong**: Produces incorrect executables that jump to wrong addresses, causing subtle memory corruption
- **Fix**: Always validate `fits_in_int32()` before `write_le32()`, report overflow with context

⚠️ **Pitfall: Continuing After First Error**
- **Description**: Attempting to process remaining files/symbols after encountering an error to "collect all errors"
- **Why Wrong**: Later errors often result from earlier unresolved symbols, producing confusing cascading diagnostics
- **Fix**: Implement fail-fast, set error flag after first error, short-circuit remaining processing

⚠️ **Pitfall: Generic Error Messages**
- **Description**: Reporting "symbol error" or "ELF error" without specific details
- **Why Wrong**: Developer must guess which symbol/file/line caused issue, wasting debugging time
- **Fix**: Include filename, symbol name, offset, expected vs actual values in all error messages

⚠️ **Pitfall: Memory Leaks on Error Paths**
- **Description**: Allocating resources but not freeing them when error occurs early
- **Why Wrong**: Long-running build processes accumulate memory leaks, eventually exhausting system memory
- **Fix**: Implement consistent `free_*()` functions for all data structures, call in error cleanup paths

⚠️ **Pitfall: Ignoring Alignment in Error Checks**
- **Description**: Validating addresses/sizes without considering alignment requirements
- **Why Wrong**: May report success but produce executables that crash due to misaligned accesses
- **Fix**: Include alignment in all layout calculations, validate `address % alignment == 0`

### Implementation Guidance

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Error Reporting | `fprintf(stderr, ...)` with custom formatting | Structured logging with JSON output for IDE integration |
| Error Context Tracking | Pass filename/symbol name as parameters | Error stack with component traceback |
| Memory Validation | Manual `NULL` checks after allocations | Address sanitizer (ASan) integration for development |
| Input Validation | Byte-by-byte manual checking | Formal verification of ELF constraints with validation library |

#### Recommended File/Module Structure

```
static-linker/
  src/
    error.c              ← Central error reporting utilities
    error.h              ← Error categories, reporting interface
    cleanup.c            ← Orchestrated cleanup functions
    cleanup.h
    
    components/
      elf_reader.c       ← ELF-specific validation
      section_merger.c   ← Layout validation
      symbol_resolver.c  ← Symbol conflict detection
      relocation.c       ← Relocation validation
      executable_writer.c ← Executable validation
      
  tests/
    test_errors.c        ← Error condition tests
    corrupt_files/       ← Test inputs with deliberate errors
```

#### Infrastructure Starter Code

**Complete Error Reporting Implementation** (`error.h` and `error.c`):

```c
/* error.h - Central error reporting interface */
#ifndef ERROR_H
#define ERROR_H

#include <stdio.h>
#include <stdbool.h>

typedef enum {
    ERROR_NONE = 0,
    ERROR_INPUT_VALIDATION,   // Bad ELF files, I/O errors
    ERROR_SYMBOL_RESOLUTION,  // Undefined/multiple definitions
    ERROR_RELOCATION,         // Overflow, invalid relocations
    ERROR_LAYOUT,             // Alignment, address space issues
    ERROR_RESOURCE,           // Memory exhaustion, system limits
    ERROR_INTERNAL           // Bug in linker itself
} ErrorCategory;

void report_error(ErrorCategory category, const char* component, 
                  const char* format, ...);
void report_warning(const char* component, const char* format, ...);
void fatal_error(const char* format, ...);

bool has_errors(void);
int get_error_count(void);
int get_warning_count(void);
void reset_error_counters(void);

#endif /* ERROR_H */
```

```c
/* error.c - Error reporting implementation */
#include "error.h"
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

static int error_count = 0;
static int warning_count = 0;

void report_error(ErrorCategory category, const char* component, 
                  const char* format, ...) {
    const char* category_names[] = {
        "none", "input", "symbol", "relocation", 
        "layout", "resource", "internal"
    };
    
    fprintf(stderr, "error: %s: %s: ", 
            category_names[category], component);
    
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    
    fprintf(stderr, "\n");
    error_count++;
}

void report_warning(const char* component, const char* format, ...) {
    fprintf(stderr, "warning: %s: ", component);
    
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    
    fprintf(stderr, "\n");
    warning_count++;
}

void fatal_error(const char* format, ...) {
    fprintf(stderr, "fatal error: ");
    
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    
    fprintf(stderr, "\n");
    exit(EXIT_FAILURE);
}

bool has_errors(void) {
    return error_count > 0;
}

int get_error_count(void) {
    return error_count;
}

int get_warning_count(void) {
    return warning_count;
}

void reset_error_counters(void) {
    error_count = 0;
    warning_count = 0;
}
```

#### Core Logic Skeleton Code

**Symbol Resolution with Error Detection** (`symbol_resolver.c`):

```c
/* Symbol resolution with comprehensive error checking */
SymbolTable* resolve_all_symbols(ObjectFile** objects, int count, 
                                 MergedSections* merged) {
    SymbolTable* table = hash_table_create(128);
    if (!table) {
        fatal_error("failed to allocate symbol table");
    }
    
    // TODO 1: First pass - collect all symbol definitions
    // For each object file in objects[0..count-1]:
    //   For each symbol in objects[i]->symbols[0..num_symbols-1]:
    //     If symbol has STB_GLOBAL or STB_WEAK binding and not SHN_UNDEF:
    //       Look up existing entry in table with hash_table_lookup()
    //       If exists:
    //         Call resolve_symbol_conflict(existing, new_symbol, objects, i)
    //       Else:
    //         Create new SymbolEntry, insert with hash_table_insert()
    //     Note: STB_LOCAL symbols are not added to global table
    
    // TODO 2: Second pass - verify all references can be resolved
    // For each object file in objects[0..count-1]:
    //   For each symbol in objects[i]->symbols[0..num_symbols-1]:
    //     If symbol has STB_GLOBAL binding and SHN_UNDEF:
    //       Look up in table with hash_table_lookup()
    //       If not found:
    //         report_error(ERROR_SYMBOL_RESOLUTION, "Symbol Resolver",
    //                      "%s: undefined reference to '%s'",
    //                      objects[i]->filename, symbol_name)
    //       Else if type mismatch:
    //         report_error(ERROR_SYMBOL_RESOLUTION, "Symbol Resolver",
    //                      "%s: incompatible types for '%s' "
    //                      "(expected %s, got %s)",
    //                      objects[i]->filename, symbol_name,
    //                      type_to_str(expected), type_to_str(actual))
    
    // TODO 3: If any errors reported, cleanup and return NULL
    if (has_errors()) {
        hash_table_destroy(table);
        return NULL;
    }
    
    return table;
}
```

**Relocation Application with Overflow Checking** (`relocation.c`):

```c
/* Apply single relocation with validation */
static void apply_relocation_to_section(Elf64_Rela* rela, 
                                        uint64_t symbol_addr,
                                        uint8_t* section_base,
                                        uint8_t* patch_addr) {
    uint32_t relocation_type = ELF64_R_TYPE(rela->r_info);
    int64_t addend = rela->r_addend;
    
    switch (relocation_type) {
        case R_X86_64_64:
            // TODO 1: Compute absolute address: symbol_addr + addend
            // TODO 2: Write 64-bit value using write_le64()
            break;
            
        case R_X86_64_PC32:
            // TODO 1: Compute PC-relative displacement:
            //   displacement = symbol_addr + addend - (patch_addr - section_base)
            // TODO 2: Validate displacement fits in int32_t range using fits_in_int32()
            //   If not: report_error(ERROR_RELOCATION, "Relocation Applier",
            //             "R_X86_64_PC32 overflow: displacement 0x%lx "
            //             "exceeds ±2GB range", displacement)
            // TODO 3: If valid, write 32-bit value using write_le32()
            break;
            
        default:
            report_error(ERROR_RELOCATION, "Relocation Applier",
                        "unsupported relocation type %u", relocation_type);
    }
}
```

#### Language-Specific Hints (C)

- **Memory Allocation Checking**: Always check `malloc()`/`calloc()` return values:
  ```c
  void* ptr = malloc(size);
  if (!ptr && size > 0) {
      fatal_error("out of memory: failed to allocate %zu bytes", size);
  }
  ```

- `errno` for System Errors: When file operations fail, use `perror()` or `strerror(errno)`:
  ```c
  FILE* f = fopen(filename, "rb");
  if (!f) {
      report_error(ERROR_INPUT_VALIDATION, "ELF Reader",
                   "cannot open '%s': %s", filename, strerror(errno));
      return NULL;
  }
  ```

- **Signed vs Unsigned Overflow**: Use explicit checks rather than relying on wrap-around:
  ```c
  // Instead of: uint64_t end = offset + size; (may overflow)
  // Use:
  if (offset > UINT64_MAX - size) {
      report_error(ERROR_INPUT_VALIDATION, "ELF Reader",
                   "section overflow: offset 0x%lx + size 0x%lx > maximum",
                   offset, size);
  }
  uint64_t end = offset + size;
  ```

- **Portable Integer Sizes**: Use `<stdint.h>` types (`uint32_t`, `int64_t`) rather than `int`, `long` for binary data.

#### Milestone Checkpoint: Error Handling Verification

**Test Command**:
```bash
# Build linker with error support
make linker

# Test 1: Undefined symbol
gcc -c test_undef.c -o test_undef.o
./linker test_undef.o -o test_undef 2>&1 | grep "undefined reference"
# Expected: "error: symbol: Symbol Resolver: test_undef.o: undefined reference to 'missing_func'"

# Test 2: Multiple definitions
gcc -c test_multidef1.c -o test_multidef1.o
gcc -c test_multidef2.c -o test_multidef2.o
./linker test_multidef1.o test_multidef2.o -o test_multidef 2>&1 | grep "multiple definition"
# Expected: "error: symbol: Symbol Resolver: multiple definition of 'global_var'"

# Test 3: Bad ELF file
dd if=/dev/zero of=corrupted.o bs=1 count=100
./linker corrupted.o -o corrupted 2>&1 | grep "not an ELF file"
# Expected: "error: input: ELF Reader: corrupted.o: not an ELF file"

# Test 4: Successful link should produce no errors
gcc -c test_good1.c -o test_good1.o
gcc -c test_good2.c -o test_good2.o
./linker test_good1.o test_good2.o -o test_good
echo $?
# Expected: 0 (success)
```

**Success Indicators**:
- Error messages are specific and include component name, filename, symbol names
- Exit code is non-zero when errors occur
- No memory leaks reported by Valgrind: `valgrind --leak-check=full ./linker ...`
- Warnings don't prevent successful linking (only errors do)

**Debugging Tips**:

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Linker crashes with segfault on valid input | Memory corruption or NULL pointer dereference | Run with `valgrind`, check all malloc() return values | Add NULL checks, fix buffer overruns |
| Error message mentions wrong filename | File indexing error in error reporting | Check file_index tracking in error context | Pass filename directly to report_error() |
| "Undefined symbol" when symbol exists | Symbol binding misinterpreted | Dump symbol table with `STB_*` values | Ensure `STB_GLOBAL` vs `STB_WEAK` handled correctly |
| Relocation produces wrong addresses | Endianness or addend handling error | Use `objdump -r` to compare expected vs actual | Check byte swapping, addend application |
| Cleanup causes double-free | Ownership confusion in cleanup chain | Add debug prints to each free_*() function | Implement clear ownership rules, NULL after free |

#### Edge Case Test Suite

Create these test files to verify error handling:

**`test_undef.c`**:
```c
extern void missing_func(void);
int main() { missing_func(); return 0; }
```

**`test_multidef1.c`**:
```c
int global_var = 42;
```

**`test_multidef2.c`**:
```c
int global_var = 99;  // Same name, different value
```

**`test_reloc_overflow.c`**:
```c
// Generate large displacement that won't fit in 32-bit
extern char huge_array[0x80000000];  // 2GB array
void foo(void) {
    // PC-relative access that will overflow
    char c = huge_array[0x7fffffff];
}
```

By implementing this comprehensive error handling architecture, the static linker transforms from a fragile prototype into a robust development tool that provides actionable feedback and gracefully handles the complexities of real-world linking scenarios.


## Testing Strategy

> **Milestone(s):** Milestones 1-4 (cross-cutting)

The **Testing Strategy** is the verification framework that ensures each component of the static linker works correctly individually and integrates properly into a functional whole. Testing a linker is particularly challenging because it operates at the binary level—errors manifest as incorrect addresses, misaligned sections, or runtime crashes rather than clear semantic failures. This section provides a systematic approach to validate each milestone through targeted test programs and inspection tools.

### Mental Model: Bridge Stress Testing
Imagine building a multi-span bridge where each section must precisely align with the next. To test it, you would:
1. **Load test individual spans** (Milestone 1: verify each section merges correctly)
2. **Check connector integrity** (Milestone 2: ensure symbols connect properly across spans)
3. **Validate load distribution** (Milestone 3: confirm relocations transfer weight appropriately)
4. **Run vehicles across the bridge** (Milestone 4: test the complete executable runs safely)

Just as bridge engineers use progressively more realistic tests—from material strength tests to full-scale load tests—linker testing advances from binary inspection to running actual programs, with each milestone building verification upon the previous.

### Milestone Checkpoints

Each milestone has specific acceptance criteria that translate to concrete test scenarios. The table below outlines verification objectives, test commands using standard Unix tools, and expected outcomes for each development phase.

| Milestone | Verification Objective | Test Commands & Inspection | Expected Behavior | Failure Indicators |
|-----------|----------------------|----------------------------|-------------------|-------------------|
| **Milestone 1: Section Merging** | Merged sections maintain correct layout, alignment, and mapping | 1. Compile test objects: `gcc -c test1.c test2.c`<br>2. Run linker: `./linker test1.o test2.o -o merged.o`<br>3. Inspect: `readelf -S merged.o`<br>4. Verify mappings: Custom debug output | - Single .text, .data, .rodata sections<br>- Section sizes equal sum of input sections plus alignment padding<br>- Section offsets follow alignment constraints<br>- .bss has file size 0 but non-zero memory size<br>- Section flags consistent (e.g., all .text have SHF_ALLOC\|SHF_EXECINSTR) | - Section size mismatch<br>- Misaligned section offsets (not multiple of sh_addralign)<br>- Missing or duplicate sections<br>- Mixed flags in merged sections |
| **Milestone 2: Symbol Resolution** | Global symbols resolved correctly; undefined symbols detected; strong/weak rules enforced | 1. Create objects with strong/weak/undefined symbols<br>2. Run linker with `--debug-symbols` flag<br>3. Inspect: `readelf -s output.o \| grep -E '(GLOBAL\|WEAK)'`<br>4. Test error cases: `./linker undef.o 2>&1` | - Strong symbols override weak ones<br>- Multiple strong definitions produce error<br>- Undefined symbols produce clear error messages<br>- COMMON symbols merge to largest size<br>- Local symbols (STB_LOCAL) not visible globally | - Weak symbol incorrectly overriding strong<br>- Undefined symbols silently accepted<br>- Duplicate strong symbols not detected<br>- Local symbols leaked to global scope |
| **Milestone 3: Relocations** | Addresses patched correctly in merged sections; PC-relative and absolute calculations accurate | 1. Create test with cross-file function calls and data references<br>2. Link and inspect patched sections: `objdump -d output.o`<br>3. Verify relocation sites: `readelf -r input.o` vs `objdump -d output.o`<br>4. Test overflow detection with large address offsets | - Function call offsets compute correctly<br>- Data references point to correct merged location<br>- PC-relative relocations account for instruction pointer<br>- Addends included in final calculation<br>- 32-bit overflow detected and reported | - Jump targets point to wrong addresses<br>- PC-relative offsets ignore addend<br>- Absolute relocations use wrong base address<br>- Overflow not detected in 32-bit fields |
| **Milestone 4: Executable Generation** | Produced ELF runs correctly; segments properly aligned; entry point valid | 1. Link complete program: `./linker *.o -o program`<br>2. Verify ELF: `readelf -l program`<br>3. Run: `./program ; echo $?`<br>4. Loader test: `ldd program` (should show "statically linked") | - Program executes and returns expected exit code<br>- Two PT_LOAD segments: text (RX), data (RW)<br>- Segment virtual addresses page-aligned (0x1000 boundaries)<br>- Entry point points to _start symbol<br>- File size matches p_filesz for each segment | - Segmentation fault on execution<br>- Program headers missing or malformed<br>- Text and data segments incorrectly merged<br>- Entry point points to main instead of _start |

#### Architecture Decision: Incremental vs. Integrated Testing

> **Decision: Incremental Component Testing with Integrated End-to-End Validation**
> - **Context**: The linker's pipeline architecture means errors in early components cascade to later stages. We need to detect issues as early as possible while also verifying the complete system works.
> - **Options Considered**:
>   1. **Pure end-to-end testing**: Only test final executable generation with complete programs. Simple but provides poor isolation and debugging.
>   2. **Unit testing each component in isolation**: Mock dependencies between components. Provides isolation but requires extensive mocking of binary formats.
>   3. **Incremental pipeline testing**: Test each milestone's output using real object files, verifying intermediate representations while progressing toward final executable.
> - **Decision**: Adopt incremental pipeline testing with inspection checkpoints after each milestone.
> - **Rationale**: This approach mirrors the natural development progression, allows inspection of intermediate states with standard tools (readelf, objdump), and provides clear failure localization. It balances isolation with realistic data flow.
> - **Consequences**: Test programs must be designed to exercise specific milestones while remaining compatible with incomplete linker implementations. Debug output flags become essential for inspecting intermediate states.

The incremental testing strategy relies on carefully crafted test programs that expose specific linker behaviors while remaining simple enough to manually verify expected outcomes.

### Test Programs

Test programs are minimal C programs designed to isolate specific linking behaviors. Each program exercises a particular aspect of the linker, from basic section merging to complex relocation scenarios. The following table describes the purpose, construction, and verification approach for each test category.

| Test Category | Purpose | Sample C Code Structure | Expected Linking Behavior | Verification Method |
|--------------|---------|-------------------------|--------------------------|-------------------|
| **Basic Section Merge** | Verify sections concatenate correctly | `test1.c`: `int global1 = 42;`<br>`test2.c`: `int global2 = 99;` | Single .data section containing both variables at offsets respecting alignment | Inspect merged .data section with `readelf -x .data output.o`, verify both values present |
| **Alignment Padding** | Test sh_addralign requirements | `align.c`: `int aligned_var __attribute__((aligned(32))) = 7;` | Output section offset increases to meet 32-byte alignment, with padding bytes between sections | Check section offsets are multiples of 32; verify padding bytes are zeros in hex dump |
| **.bss Handling** | Verify zero-initialized data handling | `bss1.c`: `char buffer[1024];` (uninitialized)<br>`bss2.c`: `long array[256];` | .bss section with size 1024+2048=3072 bytes (assuming 8-byte long), but zero file size | Confirm `readelf -S` shows .bss with sh_type=SHT_NOBITS, sh_size=3072, sh_offset within file but no actual data |
| **Strong/Weak Symbols** | Test symbol resolution precedence | `strong.c`: `int symbol = 1;` (global)<br>`weak.c`: `__attribute__((weak)) int symbol = 2;`<br>`use.c`: `extern int symbol;` | Final symbol value = 1 (strong overrides weak); no duplicate definition error | Check symbol table: `readelf -s output.o \| grep symbol` shows single GLOBAL definition with value 1 |
| **Undefined Symbol Detection** | Verify missing definition reporting | `main.c`: `extern void missing_func(); int main() { missing_func(); }` | Linker exits with error message identifying 'missing_func' as undefined | Stderr contains "undefined symbol: missing_func" with non-zero exit code |
| **COMMON Symbol Merging** | Test tentative definition resolution | `common1.c`: `int common_var;` (no initializer)<br>`common2.c`: `long common_var = 5;` (initialized) | Initialized definition wins; COMMON symbol resolved to .data with value 5 | Symbol table shows common_var in .data section with value 5, not COMMON |
| **Intra-file Relocation** | Verify relocation within single file | `reloc.c`: `static int local; int* ptr = &local;` | Absolute relocation for ptr pointing to local's address within .data | objdump -d shows ptr initialized with address of local variable |
| **Cross-file PC-relative** | Test function calls between files | `caller.c`: `extern void target(); void caller() { target(); }`<br>`target.c`: `void target() {}` | PC-relative relocation in caller's .text pointing to target function | Disassembly shows `call` instruction with correct offset to target |
| **Addend Handling** | Verify relocation addends included | `addend.c`: `extern int arr; int* ptr = &arr + 3;` (addend = 3*sizeof(int)) | Final ptr value = address of arr + 12 bytes (for 4-byte ints) | Inspect .data section: ptr contains arr's address + 0xC |
| **Entry Point** | Test executable entry configuration | `start.S`: assembly defining `_start` calling `main`<br>`main.c`: `int main() { return 42; }` | Executable ELF header's e_entry points to _start, not main | `readelf -h program` shows entry address matching _start symbol address |

#### Architecture Decision: Manual vs. Automated Test Verification

> **Decision: Hybrid Manual Inspection with Scriptable Verification**
> - **Context**: Linker output verification requires examining binary structures that are not easily captured in simple pass/fail tests. Automated comparison of binary files is brittle due to address variations.
> - **Options Considered**:
>   1. **Pure manual inspection**: Developer runs commands and visually verifies output. Provides maximum flexibility but is time-consuming and error-prone.
>   2. **Full automation with golden files**: Compare linker output against pre-computed "golden" binaries. Requires exact address matching, which fails when layout changes.
>   3. **Semantic verification scripts**: Write scripts that parse ELF output and check semantic properties (e.g., "symbol X has value Y", "section S contains bytes B").
> - **Decision**: Use semantic verification scripts for regression testing, supplemented by manual inspection during development.
> - **Rationale**: Semantic scripts can verify the essential correctness properties (symbol values, section contents, relocation calculations) without requiring byte-for-byte identical output. Manual inspection remains valuable for debugging and understanding tool output.
> - **Consequences**: Test infrastructure needs helper scripts to parse ELF files and extract semantic information. Test programs must be designed with predictable, verifiable properties.

The following numbered procedure outlines the recommended testing workflow for each milestone:

1. **Create test objects**: Compile minimal C programs with `gcc -c -fno-pie -no-pie` to produce straightforward relocatable objects without position-independent complexity.
2. **Run linker with debug flags**: Invoke the linker with flags like `--dump-sections`, `--dump-symbols`, or `--dump-relocations` to output intermediate states for inspection.
3. **Inspect with standard tools**: Use `readelf`, `objdump`, and `hexdump` to examine the output, comparing against expected properties.
4. **Run semantic verification**: Use custom scripts to parse ELF output and verify critical properties programmatically.
5. **Execute final programs**: For milestone 4, run the produced executable and verify it behaves correctly.

### Common Pitfalls in Linker Testing

⚠️ **Pitfall: Assuming successful compilation implies correct linking**
- **Description**: A test program compiles to object files without error, and the linker produces an output file, but the output is subtly wrong (e.g., misaligned sections, incorrect relocations).
- **Why it's wrong**: Linker errors often manifest only at runtime or with specific address patterns. A working compilation doesn't guarantee correct section merging or symbol resolution.
- **Fix**: Always inspect the linker's output with ELF analysis tools. For each test, manually verify at least one critical property using `readelf` or `objdump`.

⚠️ **Pitfall: Testing only simple cases**
- **Description**: Testing with single-file programs or programs without external references misses complex cross-file interactions.
- **Why it's wrong**: The linker's primary purpose is resolving cross-file dependencies. Single-file tests don't exercise symbol resolution or cross-file relocations.
- **Fix**: Create test suites with at least two object files that reference each other's symbols. Include edge cases like circular references, weak symbols, and COMMON symbols.

⚠️ **Pitfall: Ignoring alignment requirements**
- **Description**: Tests pass with small data types but fail when larger alignments are required (e.g., SSE vectors with 16-byte alignment).
- **Why it's wrong**: Alignment affects section offsets and virtual addresses. Misalignment can cause segmentation faults or performance penalties.
- **Fix**: Test with variables declared with explicit alignment (`__attribute__((aligned(N)))`) and verify section offsets meet requirements.

⚠️ **Pitfall: Overlooking .bss section handling**
- **Description**: Tests focus on .data and .text but neglect uninitialized data sections.
- **Why it's wrong**: .bss sections have unique properties (SHT_NOBITS type, zero file size but non-zero memory size). Mishandling causes memory corruption.
- **Fix**: Include test programs with large uninitialized arrays. Verify .bss appears in section headers with correct size but doesn't consume file space.

⚠️ **Pitfall: Confusing virtual addresses with file offsets**
- **Description**: Test verification uses file offsets where virtual addresses are needed, or vice versa.
- **Why it's wrong**: In executables, sections have both file offsets (where they live in the file) and virtual addresses (where they load in memory). Relocations use virtual addresses.
- **Fix**: When checking relocation values, compute based on virtual addresses, not file offsets. Use `readelf -l` to see the mapping between file offsets and virtual addresses.

### Implementation Guidance

#### Technology Recommendations Table

| Component | Simple Option | Advanced Option |
|-----------|--------------|-----------------|
| Test Infrastructure | Bash scripts with `readelf`/`objdump` parsing | Python with `pyelftools` library for ELF analysis |
| Test Runner | Makefile with per-milestone targets | Custom test harness with TAP (Test Anything Protocol) output |
| Binary Inspection | Manual `readelf`/`objdump` commands | Automated comparison with `diff` against expected `readelf` output |
| Debug Output | Simple `printf` debugging in linker | Structured logging with levels (INFO, DEBUG, ERROR) |

#### Recommended File Structure for Tests

```
project-root/
  src/                    # Linker source code
    linker.c              # Main entry point
    elf_reader.c          # ELF Reader component
    section_merger.c      # Section Merger component
    symbol_resolver.c     # Symbol Resolver component  
    relocation_applier.c  # Relocation Applier component
    exec_writer.c         # Executable Writer component
    utils.c               # Shared utilities
  tests/                  # Test infrastructure
    test_programs/        # Source code for test programs
      milestone1/         # Tests for section merging
        basic_merge.c
        alignment.c
        bss_test.c
      milestone2/         # Tests for symbol resolution
        strong_weak.c
        undefined.c
        common.c
      milestone3/         # Tests for relocations
        pc_relative.c
        absolute.c
        addend.c
      milestone4/         # Tests for executable generation
        simple_program.c
        start.S           # Assembly _start for entry point test
    scripts/              # Verification scripts
      verify_sections.py  # Parse ELF and verify section properties
      verify_symbols.py   # Verify symbol table contents
      verify_relocs.py    # Check relocation calculations
    Makefile              # Build test programs and run tests
  build/                  # Build directory (generated)
```

#### Test Infrastructure Starter Code

The following complete test runner script provides a foundation for automated testing. It compiles test programs, runs the linker, and verifies output using semantic checks.

```bash
#!/bin/bash
# tests/run_tests.sh - Basic test runner for static linker milestones

set -e  # Exit on any error

LINKER="../build/linker"
CC="gcc"
CFLAGS="-c -fno-pie -no-pie -O0"

# Create build directory
mkdir -p build/tests

echo "=== Testing Static Linker ==="

# Milestone 1: Section Merging Tests
echo "--- Milestone 1: Section Merging ---"
$CC $CFLAGS tests/test_programs/milestone1/basic_merge1.c -o build/tests/m1_test1.o
$CC $CFLAGS tests/test_programs/milestone1/basic_merge2.c -o build/tests/m1_test2.o

echo "Test 1: Basic section merge"
$LINKER build/tests/m1_test1.o build/tests/m1_test2.o -o build/tests/m1_output.o --dump-sections

# Verify .data section contains both variables
echo "Verifying .data section..."
readelf -x .data build/tests/m1_output.o | grep -q "0x0000002a.*0x00000063" && echo "PASS" || echo "FAIL"

# Milestone 2: Symbol Resolution Tests  
echo "--- Milestone 2: Symbol Resolution ---"
$CC $CFLAGS tests/test_programs/milestone2/strong.c -o build/tests/m2_strong.o
$CC $CFLAGS tests/test_programs/milestone2/weak.c -o build/tests/m2_weak.o

echo "Test 2: Strong overrides weak"
$LINKER build/tests/m2_strong.o build/tests/m2_weak.o -o build/tests/m2_output.o --dump-symbols 2>&1 | grep -q "symbol.*GLOBAL.*value:" && echo "PASS" || echo "FAIL"

echo "All tests completed!"
```

#### Core Test Verification Skeleton

For more sophisticated verification, this Python script skeleton uses command-line tools to parse ELF output and check properties programmatically:

```python
#!/usr/bin/env python3
# tests/scripts/verify_sections.py - Verify section properties in ELF file

import subprocess
import sys
import re

def readelf_sections(elf_file):
    """Run readelf -S and parse section table into list of dicts."""
    result = subprocess.run(['readelf', '-S', '--wide', elf_file], 
                          capture_output=True, text=True, check=True)
    
    sections = []
    in_table = False
    
    for line in result.stdout.split('\n'):
        # Parse line like "[ 1] .text PROGBITS 0000000000000000 000040 000015 00 AX 0 0 16"
        if '[' in line and ']' in line and 'Name' not in line:
            parts = re.split(r'\s+', line.strip())
            if len(parts) >= 7:
                sections.append({
                    'idx': parts[0].strip('[]'),
                    'name': parts[1],
                    'type': parts[2],
                    'addr': int(parts[3], 16),
                    'offset': int(parts[4], 16),
                    'size': int(parts[5], 16),
                    'align': int(parts[9], 16) if len(parts) > 9 else 1
                })
    
    return sections

def verify_section_alignment(sections):
    """Verify all sections are properly aligned."""
    errors = []
    for sect in sections:
        if sect['align'] > 0 and sect['addr'] % sect['align'] != 0:
            errors.append(f"Section {sect['name']}: address 0x{sect['addr']:x} not aligned to {sect['align']}")
    
    return errors

# TODO 1: Add function to verify section size equals sum of input sections
# TODO 2: Add function to check .bss has type NOBITS and file offset within segment
# TODO 3: Add function to verify merged section flags are consistent
# TODO 4: Add function to check section ordering (code before data)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <elf_file>")
        sys.exit(1)
    
    sections = readelf_sections(sys.argv[1])
    errors = verify_section_alignment(sections)
    
    if errors:
        print("FAILED:")
        for err in errors:
            print(f"  {err}")
        sys.exit(1)
    else:
        print("PASS: All sections properly aligned")
```

#### Language-Specific Hints for Testing

- **C Test Programs**: Use `__attribute__((weak))` for weak symbols, `__attribute__((aligned(N)))` for alignment tests, and `extern` declarations for external references.
- **Assembly Integration**: Write simple assembly files for testing entry points and exact instruction patterns. Use `.global _start` to define entry symbol.
- **Error Detection**: Check linker exit codes (`$?` in bash) for error conditions. Successful linking should return 0, errors should return non-zero.
- **Temporary Files**: Use `mktemp` for temporary object files to avoid polluting the source directory.

#### Milestone Checkpoint Verification Commands

After implementing each milestone, run these commands to verify basic functionality:

**Milestone 1 Checkpoint**:
```bash
# Create test objects
echo "int a=1;" > test1.c; echo "int b=2;" > test2.c
gcc -c test1.c test2.c

# Run linker with debug output
./linker test1.o test2.o -o merged.o --dump-sections

# Verify output
readelf -S merged.o | grep -E "\.(text|data|rodata|bss)"
# Should see one of each section type, not duplicates
```

**Milestone 2 Checkpoint**:
```bash
# Create objects with strong/weak symbols
echo "int symbol = 42;" > strong.c
echo "__attribute__((weak)) int symbol = 99;" > weak.c  
echo "extern int symbol; int main() { return symbol; }" > main.c
gcc -c strong.c weak.c main.c

# Link and check symbol value
./linker strong.o weak.o main.o -o resolved.o --dump-symbols 2>&1 | grep symbol
# Should show symbol = 42 (strong overrides weak)
```

**Milestone 3 Checkpoint**:
```bash
# Create cross-file function call test
echo "void target() {}" > target.c
echo "extern void target(); void caller() { target(); }" > caller.c
gcc -c target.c caller.c

# Link and disassemble
./linker target.o caller.o -o relocated.o
objdump -d relocated.o | grep -A2 "<caller>:"
# Should show 'call' instruction with correct offset to target
```

**Milestone 4 Checkpoint**:
```bash
# Create minimal executable test
echo ".global _start; _start: mov \$60, %rax; mov \$0, %rdi; syscall" > start.S
gcc -c start.S

# Link as executable
./linker start.o -o program

# Verify executable properties
readelf -h program | grep "Entry point"
readelf -l program | grep "LOAD"
# Should show entry point address and two LOAD segments

# Run program (should exit with code 0)
./program; echo "Exit code: $?"
```

#### Debugging Tips Table

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| **Segmentation fault on execution** | Entry point points to wrong address or text segment misaligned | 1. Check entry point: `readelf -h program \| grep Entry`<br>2. Verify _start symbol exists: `readelf -s program \| grep _start`<br>3. Check segment alignment: `readelf -l program \| grep -A1 LOAD` | Ensure entry point points to _start virtual address. Align segments to 0x1000 boundaries. |
| **Function calls jump to wrong location** | PC-relative relocation calculation error | 1. Disassembly: `objdump -d program \| grep -B2 -A2 "call"`<br>2. Check relocation: PC-relative = symbol - (relocation_site + 4) + addend | Verify relocation calculation uses instruction-relative addressing (RIP-relative on x86-64). |
| **Data references contain zero or garbage** | Absolute relocation not applied or wrong section base | 1. Hex dump data section: `readelf -x .data program`<br>2. Check symbol addresses: `readelf -s program \| grep <symbol>` | Ensure absolute relocations use symbol's virtual address + addend, not file offset. |
| **"Undefined symbol" error for defined symbol** | Symbol binding mismatch (local vs global) or section index wrong | 1. Check input symbol table: `readelf -s input.o \| grep <symbol>`<br>2. Verify binding (GLOBAL/WEAK) and section index (not UNDEF or ABS) | Ensure global symbols have STB_GLOBAL binding and defined section index (not SHN_UNDEF). |
| **Merged section size smaller than expected** | Alignment padding missing or .bss counted incorrectly | 1. Check individual section sizes: `readelf -S input*.o \| grep <section>`<br>2. Verify alignment: `readelf -S output.o \| grep -A1 <section>` | Add alignment padding between concatenated sections: `offset = align_to(prev_end, alignment)`. |
| **Executable runs but returns wrong value** | Data relocations wrong or .bss not zero-initialized | 1. Trace execution with `strace` or `gdb`<br>2. Check .bss initialization: linker should zero .bss memory | Ensure .bss memory size (p_memsz) > file size (p_filesz); loader will zero-fill difference. |
| **"File truncated" or "Exec format error"** | ELF headers malformed or program headers misaligned | 1. Basic ELF validation: `readelf -a program > /dev/null`<br>2. Check header offsets: `od -x -N 64 program` | Verify ELF magic bytes, header sizes, and that program headers don't exceed file size. |

These testing strategies provide a comprehensive framework for validating the static linker at each development stage, ensuring robust functionality before progressing to the next milestone.


## Debugging Guide
> **Milestone(s):** Milestones 1-4 (cross-cutting)

The **Debugging Guide** is the diagnostic toolkit for understanding and fixing failures in the static linker. Unlike typical software where runtime behavior reveals bugs, linking errors manifest as malformed binaries that fail to load, produce incorrect output, or crash with segmentation faults. This section provides structured approaches to diagnose linking problems through observable symptoms, inspection techniques, and systematic troubleshooting.

### Symptom → Cause → Fix Table

This table maps observable failures during linking or execution to their root causes and remediation strategies. The table follows the pipeline architecture: symptoms progress from parsing errors through section merging, symbol resolution, relocation, and final executable generation.

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| **Parsing/Validation Errors** |
| "Not a valid ELF file" or "Invalid ELF magic" when reading .o files | Input file is not an ELF object file (possibly source or binary file) or file corrupted | Ensure input files are compiled object files (`gcc -c`). Use `file` command to verify ELF format. Check file read operations handle binary mode correctly. |
| "Section header offset out of bounds" or "String table index invalid" | Malformed ELF header values or incorrect endianness handling | Validate `e_shoff`, `e_shnum`, `e_shentsize` before accessing. Implement proper little-endian byte swapping (`read_le64`, `read_le32`). |
| **Section Merging Errors** |
| Output executable has wrong section sizes (e.g., .text too small) | Alignment padding miscalculation or missing section data | Verify `align_to` function rounds up correctly: `(offset + align - 1) & ~(align - 1)`. Ensure all `SHF_ALLOC` sections are processed, including `.rodata` and `.bss`. |
| Segfault when accessing global variables or code jumps to wrong locations | Incorrect `InputSectionMapping` offsets cause symbols to point to wrong addresses | Validate mapping calculations: `output_offset = current_output_size` (before adding input). Debug by printing mapping table and comparing with `objdump -r` expectations. |
| `.bss` section occupies file space in output (non-zero file size) | Treating `SHT_NOBITS` sections as having file data | Check `sh_type == SHT_NOBITS` and set `data = NULL`, `data_size = 0` for output. Only allocate virtual memory space (`p_memsz`) not file space (`p_filesz`). |
| **Symbol Resolution Errors** |
| "Undefined symbol: main" or similar missing symbol errors | No definition for referenced symbol in any input file | Ensure all required object files provided. Check symbol visibility: only `STB_GLOBAL`/`STB_WEAK` symbols are visible across files. Verify `SHN_UNDEF` handling. |
| "Multiple definition of symbol 'x'" when linking | Two strong definitions (`STB_GLOBAL`) for same symbol | One source file must remove duplicate global definition. Use `static` for file-local, or declare one as weak (`__attribute__((weak))` in C). |
| Program uses wrong function version (weak instead of strong) | Weak symbol overriding strong due to resolution order | Implement strong-over-weak rule: when adding symbol, if existing is weak and new is strong, replace. Track `STB_WEAK` vs `STB_GLOBAL` in `SymbolEntry`. |
| Uninitialized global variables get zero values instead of merged | COMMON symbol merging not implemented (size-based resolution) | For `SHN_COMMON` symbols, choose largest size definition. Convert to regular `STB_GLOBAL` definition in `.bss` with aligned size. |
| **Relocation Errors** |
| Executable crashes with "Illegal instruction" (SIGILL) or jumps to random address | Incorrect relocation patching corrupts instruction stream | Verify relocation type handling: `R_X86_64_PC32` uses 32-bit signed offset relative to next instruction. Use `compute_pcrel_relocation`. |
| Program computes wrong addresses (e.g., accessing wrong array element) | Addend handling error: forgetting to include `r_addend` in calculation | Relocation formula: `S + A` (absolute) or `S + A - P` (PC-relative). `A` is `r_addend`. Extract from `Elf64_Rela`. |
| "Relocation overflow" or "truncation error" for large binaries | 64-bit address truncated to 32-bit field in `R_X86_64_PC32` | Validate with `fits_in_int32` before writing. If overflow, linker should report error (cannot create position-independent code for large offsets). |
| Relocation applies to wrong location in section | `r_offset` misinterpreted as file offset instead of section-relative offset | `r_offset` is offset from start of containing section. Add section's output base address to get final virtual address for patching. |
| **Executable Generation Errors** |
| "Cannot execute binary file: Exec format error" on Linux | Invalid ELF header (wrong `e_type`, `e_machine`, or `e_version`) | Set `e_type = ET_EXEC`, `e_machine = EM_X86_64`, `e_version = EV_CURRENT`. Ensure ELF header at file start with proper `e_ident` magic. |
| Segmentation fault immediately on execution (`SIGSEGV`) | Entry point (`e_entry`) points to non-executable memory or invalid address | Entry point should be virtual address of `_start` symbol (not `main`). Verify `_start` resolved and placed in executable segment (`PF_X`). |
| Program loads but text/data segments overlap in memory | Incorrect segment layout: `p_vaddr` and `p_memsz` cause overlap | Separate text (RX) and data (RW) into different `PT_LOAD` segments. Ensure `p_vaddr + p_memsz` of text < data segment `p_vaddr`. |
| `.data` section appears read-only (writes cause SIGSEGV) | Data segment missing write permission (`PF_W`) | Set `p_flags = PF_R | PF_W` for data segment (containing `.data`, `.bss`). Text segment gets `PF_R | PF_X`. |
| Executable file size excessively large (multiple MB for small program) | Incorrect `p_filesz` includes `.bss` or alignment padding as file data | `p_filesz` = size of segment in file (exclude `.bss`). `p_memsz` = size in memory (include `.bss` zero-initialized). Align both to page boundary (4096). |
| **Runtime Data Corruption** |
| Global variables initialized incorrectly (wrong values) | Relocation patching overwrites initialized data values | Ensure relocations only patch locations specified by `r_offset`. Use separate read/write passes or copy data before patching. |
| String literals appear corrupted or wrong | `.rodata` placed in writable segment, or merged with `.data` | Place `.rodata` in text segment (RX) not data segment. Check section flags: `SHF_ALLOC` but not `SHF_WRITE`. |
| Stack alignment issues causing SSE instructions to crash | Entry point (`_start`) doesn't maintain 16-byte stack alignment | Ensure `_start` begins with `push rbp` (saves frame pointer) which aligns stack to 16 bytes per System V ABI requirement. |

### Binary Inspection Techniques

Debugging a linker requires examining binary artifacts at various stages. Three essential tools provide complementary views: **readelf** for structural analysis, **objdump** for disassembly and relocations, and **hexdump** for raw byte-level inspection.

#### Using readelf for Structural Validation

`readelf` displays ELF file structure without requiring debug symbols. It's the primary tool for verifying section layout, symbol tables, and segment organization.

**Key Commands and Interpretation:**

| Command | Purpose | What to Look For |
|---------|---------|------------------|
| `readelf -h file.o` | Examine ELF header | `Type: REL (Relocatable file)` for inputs, `Type: EXEC (Executable file)` for output. Verify `Machine: Advanced Micro Devices X86-64`. |
| `readelf -S file.o` | List section headers | Sections `.text`, `.data`, `.rodata`, `.bss` present. Note sizes and alignments. Check `.symtab` and `.rela.text`/.`rela.data` exist. |
| `readelf -l executable` | Display program headers | Two `LOAD` segments: one `R E` (text), one `RW` (data). `VirtAddr` and `PhysAddr` identical for static. `MemSiz` > `FileSiz` for `.bss`. |
| `readelf -s file.o` | Show symbol table | Global symbols (`GLOBAL` binding), undefined symbols (`UND` section). Weak symbols marked `WEAK`. Verify `main` and `_start`. |
| `readelf -r file.o` | Display relocations | `R_X86_64_PC32` and `R_X86_64_64` entries. Check `Offset` matches code locations, `Addend` values correct. |

**Example Diagnostic Session:**
```
# Check input object file structure
$ readelf -S input.o
Section Headers:
  [Nr] Name   Type       Addr     Off    Size   ES Flg Lk Inf Al
  [ 1] .text  PROGBITS   00000000 000040 000015 00  AX  0   0  1
  [ 2] .data  PROGBITS   00000000 000058 000008 00  WA  0   0  4
  [ 3] .bss   NOBITS     00000000 000060 000010 00  WA  0   0  4
  [ 4] .rela.t RELA      00000000 000200 000018 18   I  9   1  8
  [ 5] .symtab SYMTAB    00000000 000100 000090 18     10   9  8

# Check output executable segments
$ readelf -l output
Program Headers:
  Type     Offset   VirtAddr   PhysAddr   FileSiz  MemSiz   Flg Align
  LOAD     0x001000 0x400000   0x400000   0x0000a0 0x0000a0 R E 0x1000
  LOAD     0x002000 0x401000   0x401000   0x000008 0x000018 RW  0x1000
```
Interpretation: The output has proper segments: text at 0x400000 (RX), data at 0x401000 (RW). Data segment has `MemSiz=0x18` (includes 0x10 .bss) but `FileSiz=0x8` (only .data).

#### Using objdump for Code and Relocation Analysis

`objdump` provides disassembly and detailed relocation views, essential for verifying instruction patching.

**Key Commands and Interpretation:**

| Command | Purpose | What to Look For |
|---------|---------|------------------|
| `objdump -d file.o` | Disassemble sections | View raw instructions before relocation. Look for `00 00 00 00` placeholders where relocations will patch. |
| `objdump -r file.o` | Show relocations (same as readelf -r) | Cross-reference with disassembly: relocation offset should match address of placeholder bytes. |
| `objdump -d executable` | Disassemble final executable | Verify instructions have correct absolute addresses (e.g., `call 400012 <main>`). No zeros in address fields. |
| `objdump -x executable` | All headers summary | Check `start address` matches entry point. Verify sections mapped to correct addresses. |

**Example Diagnostic Session:**
```
# Examine relocations in object file
$ objdump -r input.o
RELOCATION RECORDS FOR [.text]:
OFFSET   TYPE              VALUE
00000007 R_X86_64_PC32     foo-0x0000000000000004
0000000e R_X86_64_64       bar

# Disassemble to see relocation sites
$ objdump -d input.o
0000000000000000 <main>:
   0: 55                    push   %rbp
   1: 48 89 e5              mov    %rsp,%rbp
   4: e8 00 00 00 00        callq  9 <main+0x9>  # Relocation at offset 5 (e8 opcode + rel32)
   9: 48 8b 05 00 00 00 00  mov    0x0(%rip),%rax # Relocation at offset b (R_X86_64_PC32)
  10: c9                    leaveq
  11: c3                    retq
```
Interpretation: The `callq` at offset 4 has opcode `e8` followed by 4 zero bytes (offset 5-8). This matches relocation offset 7? Wait, offset calculation: objdump shows offset 7 for the relocation, but disassembly shows zeros at bytes 5-8. Actually, `e8` is at address 4, then 4-byte rel32 at addresses 5-8 = offset 5. The discrepancy is because objdump's offset is from section start, and sometimes includes addend in display. Check carefully: `R_X86_64_PC32` relocation offset should be the address of the 4-byte field to patch.

#### Using hexdump for Byte-Level Inspection

When structural tools show anomalies, `hexdump` (or `xxd`) reveals raw bytes to diagnose corruption, endianness, or padding issues.

**Key Commands and Interpretation:**

| Command | Purpose | What to Look For |
|---------|---------|------------------|
| `hexdump -C -n 64 file.o` | First 64 bytes with ASCII view | Verify ELF magic: `7f 45 4c 46`. Check section data appears at offsets matching `sh_offset`. |
| `hexdump -C -s <offset> -n 16 file` | Specific region inspection | Examine relocation patching: at `r_offset`, bytes should match computed address (little-endian). |
| `hexdump -C -v executable \| head -20` | Full file view (verbose) | Identify unexpected zeros, alignment padding patterns (`00` bytes between sections). |

**Example Diagnostic Session:**
```
# Check ELF header
$ hexdump -C -n 64 input.o
00000000  7f 45 4c 46 02 01 01 00  00 00 00 00 00 00 00 00  |.ELF............|
00000010  01 00 3e 00 01 00 00 00  00 00 00 00 00 00 00 00  |..>.............|
00000020  00 00 00 00 00 00 00 00  40 00 00 00 00 00 00 00  |........@.......|

# Check patched relocation site (executable, virtual address 0x400010)
$ objdump -d output | grep -A2 400010
  400010: 48 8b 05 e9 0f 00 00  mov 0xfe9(%rip),%rax # 401000 <global_var>

$ hexdump -C -s 0x1010 -n 8 output  # File offset = virtual address - segment Vaddr + segment offset
00001010  e9 0f 00 00 00 00 00 00                           |........|
```
Interpretation: The bytes at file offset 0x1010 are `e9 0f 00 00` (little-endian 0xfe9), matching the PC-relative offset from instruction at 0x400010 to symbol at 0x401000: 0x401000 - (0x400010 + 7) = 0xfe9.

#### Integrated Diagnostic Workflow

When encountering a linking error, follow this systematic inspection sequence:

1. **Validate Inputs:** Use `readelf -h` on all `.o` files to confirm they are valid ELF relocatable files for x86-64.
2. **Check Symbol Resolution:** Run `readelf -s` on each `.o` and compare with linker's symbol table. Verify undefined symbols have definitions elsewhere.
3. **Inspect Merged Layout:** For linker-internal debugging, dump `MergedSections` mappings. Externally, check output sections with `readelf -S`.
4. **Verify Relocations:** Compare `objdump -r` input relocations with final patched instructions in executable. Use `hexdump` at relocation sites.
5. **Validate Executable Structure:** `readelf -l` must show proper segment layout with correct permissions and alignment.
6. **Runtime Debugging:** If executable runs but produces wrong output, use `gdb` to disassemble at runtime, check register values, and memory contents.

> **Debugging Philosophy:** Linker bugs often cascade—a single error in section merging causes incorrect symbol addresses, which breaks relocations, producing a malformed executable. Always debug **upstream** first: fix section merging before symbol resolution, symbols before relocations. Use the component architecture to isolate failures: test ELF Reader independently with known object files, then Section Merger with simple inputs, etc.

### Implementation Guidance

#### A. Technology Recommendations Table

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Debug Output | `fprintf(stderr, ...)` with conditional compilation | Custom logging framework with levels (DEBUG, INFO, ERROR) and component tags |
| Binary Inspection | Manual `hexdump` analysis | Integrated hex dump utility in linker (`--dump-sections`, `--dump-symbols`) |
| Validation | `assert()` for invariants | Comprehensive error checking with descriptive messages and recovery |
| Diagnostics | Print to stderr during linking | Generate linker map file (`-Map`) showing section layout and symbol addresses |

#### B. Recommended File/Module Structure

Add debugging utilities to a separate module that can be conditionally compiled:

```
static-linker/
  src/
    main.c                    # Entry point, argument parsing
    elf_reader.c              # ELF parsing (Milestone 1)
    section_merger.c          # Section merging (Milestone 1)
    symbol_resolver.c         # Symbol resolution (Milestone 2)
    relocation_applier.c      # Relocation processing (Milestone 3)
    executable_writer.c       # Executable generation (Milestone 4)
    debug.c                   # Debugging utilities (THIS SECTION)
    debug.h                   # Debug function declarations
    utils.c                   # Shared utilities (xmalloc, align_to, etc.)
    linker.c                  # Orchestration (link_files)
  include/
    elf_types.h               # ELF constants and type definitions
    common.h                  # Common structures and forward declarations
```

#### C. Infrastructure Starter Code (Complete Debug Utilities)

**File: `debug.h`**
```c
#ifndef DEBUG_H
#define DEBUG_H

#include <stdio.h>
#include <stdint.h>
#include "common.h"

// Debug levels
#define DEBUG_LEVEL_NONE    0
#define DEBUG_LEVEL_ERROR   1
#define DEBUG_LEVEL_WARNING 2
#define DEBUG_LEVEL_INFO    3
#define DEBUG_LEVEL_DEBUG   4

extern int debug_level;

// Set debug level (0-4)
void set_debug_level(int level);

// Debug print macros
#define DEBUG_ERROR(fmt, ...) \
    do { if (debug_level >= DEBUG_LEVEL_ERROR) \
        fprintf(stderr, "ERROR: " fmt, ##__VA_ARGS__); } while(0)

#define DEBUG_WARN(fmt, ...) \
    do { if (debug_level >= DEBUG_LEVEL_WARNING) \
        fprintf(stderr, "WARN: " fmt, ##__VA_ARGS__); } while(0)

#define DEBUG_INFO(fmt, ...) \
    do { if (debug_level >= DEBUG_LEVEL_INFO) \
        fprintf(stderr, "INFO: " fmt, ##__VA_ARGS__); } while(0)

#define DEBUG_DBG(fmt, ...) \
    do { if (debug_level >= DEBUG_LEVEL_DEBUG) \
        fprintf(stderr, "DBG: " fmt, ##__VA_ARGS__); } while(0)

// Diagnostic dump functions
void dump_elf_header(const Elf64_Ehdr* hdr);
void dump_section_header(const Elf64_Shdr* shdr, const char* name);
void dump_symbol(const Elf64_Sym* sym, const char* name, const char* section_name);
void dump_relocation(const Elf64_Rela* rela, const char* sym_name);
void dump_merged_layout(const MergedSections* merged);
void dump_symbol_table(const SymbolTable* table);

// Hex dump utility
void hexdump(const uint8_t* data, size_t size, uint64_t base_addr);

#endif // DEBUG_H
```

**File: `debug.c`** (partial implementation)
```c
#include "debug.h"
#include "common.h"
#include <elf_types.h>
#include <inttypes.h>

int debug_level = DEBUG_LEVEL_ERROR;

void set_debug_level(int level) {
    debug_level = level;
}

void dump_elf_header(const Elf64_Ehdr* hdr) {
    DEBUG_INFO("ELF Header:\n");
    DEBUG_INFO("  Magic:   %02x %02x %02x %02x\n", 
               hdr->e_ident[0], hdr->e_ident[1], hdr->e_ident[2], hdr->e_ident[3]);
    DEBUG_INFO("  Class:   %s\n", hdr->e_ident[EI_CLASS] == ELFCLASS64 ? "ELF64" : "Unknown");
    DEBUG_INFO("  Type:    %s\n", hdr->e_type == ET_REL ? "REL (Relocatable)" : 
                                 hdr->e_type == ET_EXEC ? "EXEC (Executable)" : "Unknown");
    DEBUG_INFO("  Entry:   0x%" PRIx64 "\n", hdr->e_entry);
}

void hexdump(const uint8_t* data, size_t size, uint64_t base_addr) {
    for (size_t i = 0; i < size; i += 16) {
        printf("%08" PRIx64 ": ", base_addr + i);
        for (size_t j = 0; j < 16; j++) {
            if (i + j < size) {
                printf("%02x ", data[i + j]);
            } else {
                printf("   ");
            }
            if (j == 7) printf(" ");
        }
        printf(" |");
        for (size_t j = 0; j < 16 && i + j < size; j++) {
            uint8_t c = data[i + j];
            printf("%c", (c >= 32 && c < 127) ? c : '.');
        }
        printf("|\n");
    }
}
```

#### D. Core Logic Skeleton Code (Debug Dump Functions)

**Add to `debug.c`:**

```c
// TODO 1: Implement dump_merged_layout to show output sections and input mappings
void dump_merged_layout(const MergedSections* merged) {
    if (debug_level < DEBUG_LEVEL_DEBUG) return;
    
    DEBUG_DBG("=== Merged Sections Layout ===\n");
    DEBUG_DBG("%-10s %-10s %-10s %-10s %-10s\n", 
              "Name", "VAddr", "Size", "FileOff", "Align");
    
    // TODO 2: Iterate through merged->sections array
    // For each OutputSection:
    //   DEBUG_DBG("%-10s 0x%08" PRIx64 " 0x%08" PRIx64 " 0x%08" PRIx64 " %" PRIu64 "\n",
    //             sect->name, sect->virtual_addr, sect->size, 
    //             sect->file_offset, sect->sh_addralign);
    
    DEBUG_DBG("\n=== Input Section Mappings ===\n");
    DEBUG_DBG("%-10s %-10s %-10s\n", "File:Sec", "OutSect", "Offset");
    
    // TODO 3: Iterate through merged->mappings array
    // For each InputSectionMapping:
    //   DEBUG_DBG("  %d:%d      %d         0x%08" PRIx64 "\n",
    //             map->file_index, map->section_index,
    //             output_sect_index, map->output_offset);
}

// TODO 4: Implement dump_symbol_table to show resolved symbols
void dump_symbol_table(const SymbolTable* table) {
    if (debug_level < DEBUG_LEVEL_INFO) return;
    
    DEBUG_INFO("=== Symbol Table ===\n");
    DEBUG_INFO("%-20s %-10s %-10s %-10s %s\n", 
               "Name", "Value", "Size", "Binding", "Defined");
    
    // TODO 5: Use hash_table_iterate to walk all symbols
    // For each SymbolEntry:
    //   const char* binding_str = (entry->binding == STB_GLOBAL) ? "GLOBAL" :
    //                             (entry->binding == STB_WEAK) ? "WEAK" : "LOCAL";
    //   DEBUG_INFO("%-20s 0x%08" PRIx64 " 0x%08" PRIx64 " %-10s %s\n",
    //              entry->name, entry->value, entry->size,
    //              binding_str, entry->defined ? "YES" : "NO");
}

// TODO 6: Implement dump_relocation for debugging relocation processing
void dump_relocation(const Elf64_Rela* rela, const char* sym_name) {
    if (debug_level < DEBUG_LEVEL_DEBUG) return;
    
    uint32_t type = ELF64_R_TYPE(rela->r_info);
    const char* type_str = (type == R_X86_64_64) ? "R_X86_64_64" :
                          (type == R_X86_64_PC32) ? "R_X86_64_PC32" : "UNKNOWN";
    
    DEBUG_DBG("  Reloc: offset=0x%08" PRIx64 " type=%s sym=%s addend=%" PRId64 "\n",
              rela->r_offset, type_str, sym_name ? sym_name : "NULL", rela->r_addend);
}
```

#### E. Language-Specific Hints (C)

- **Conditional Debugging:** Use preprocessor macros to compile out debug code: `#ifdef LINKER_DEBUG` or use runtime `debug_level` variable as shown.
- **Print Format Specifiers:** For `uint64_t`, use `PRIx64` from `<inttypes.h>`: `printf("0x%" PRIx64, value);`.
- **Endianness Helpers:** When dumping raw bytes, remember ELF data is in file's endianness (little-endian for x86-64). Use `read_le32()`/`write_le32()` for consistency.
- **Error Reporting:** Use `__FILE__` and `__LINE__` macros in debug macros to pinpoint location.

#### F. Milestone Checkpoint Debugging

| Milestone | What to Check | Command | Expected Outcome |
|-----------|---------------|---------|------------------|
| 1 | Section merging preserves alignment | `readelf -S output.o \| grep -A1 .text` | Output section alignment ≥ any input alignment |
| 1 | Input mappings correct | Linker with `--debug` flag | Mappings show each input section → output offset |
| 2 | Symbol resolution | `readelf -s output` | No undefined symbols (except from libc if not linking), weak/strong resolved |
| 3 | Relocation patching | `objdump -d output \| grep -B2 -A2 call` | Call addresses point to valid functions, not zeros |
| 4 | Executable loads | `./output ; echo $?` | Program runs and returns exit code (compile with `_start` returning value) |

#### G. Debugging Tips Table

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Linker crashes with segfault in `read_elf_file` | Buffer overflow reading section data | Use `valgrind ./linker input.o` or add bounds checks before `memcpy` | Validate `sh_offset + sh_size ≤ file_size` |
| Output executable has zero size for .text | Section data not copied during merging | Call `hexdump` on output section data in `get_output_section_data` | Ensure `memcpy` copies from each input section's data |
| Program prints wrong values from global variables | Relocation addend ignored or wrong symbol address | Add debug print in `apply_relocation_to_section` showing `S`, `A`, `P` | Verify `symbol_addr` calculation includes output section base |
| Executable runs but `printf` output doesn't appear | Missing libc startup files (crt1.o, etc.) | Static linking without libc requires minimal `_start` | Provide simple `_start` in assembly that calls `main` then `exit` syscall |


## Future Extensions

> **Milestone(s):** This section describes potential enhancements beyond the core milestones, showing how the architectural design can accommodate future evolution.

The static linker's modular **pipeline architecture** provides a robust foundation that can be extended with additional functionality without requiring major architectural redesign. By maintaining clear separation between components—ELF reading, section merging, symbol resolution, relocation application, and executable writing—each extension can be implemented as a focused enhancement to specific components while preserving the overall linking workflow. This section explores several meaningful extensions that build upon the core linking functionality, demonstrating how the existing design accommodates evolution and how each enhancement would integrate with the current architecture.

### Extension Ideas

The core linker implements the minimum viable functionality required to produce working executables from multiple object files. However, real-world linkers support dozens of additional features that address practical software development needs. These extensions represent natural progression points where learners can deepen their understanding of linking concepts while expanding the linker's capabilities.

#### Support for Additional Relocation Types

**Mental Model: Expanding the Address Translation Dictionary**  
Think of relocation types as different grammatical rules in a language translation dictionary. The core linker implements two basic "grammar rules" (`R_X86_64_PC32` for relative addressing and `R_X86_64_64` for absolute addressing), much like learning present tense before mastering past and future tenses. Each new relocation type represents another grammatical construction that follows predictable patterns but requires understanding specific calculation rules and constraints.

**Current Limitation and Extension Scope**  
The initial implementation supports only two x86-64 relocation types, which suffices for simple programs but cannot handle position-independent code, thread-local storage, or 32-bit absolute addressing. Real linkers support dozens of architecture-specific relocation types (over 30 for x86-64 alone) that handle various addressing modes, data widths, and special cases.

**Architecture Decision: Relocation Type Handler Registry**

> **Decision: Plugin-Style Relocation Handler Registry**
> - **Context**: The linker needs to support many relocation types with different calculation formulas and validation rules. Hardcoding each type in a large switch statement becomes unmaintainable and makes cross-architecture support difficult.
> - **Options Considered**:
>   1. **Monolithic Switch Statement**: Single function with switch-case for each relocation type (current implementation for two types).
>   2. **Function Pointer Table**: Array of handler functions indexed by relocation type, with NULL entries for unsupported types.
>   3. **Registration-Based Plugin System**: Dynamic registration of handlers with metadata about supported architectures and relocation types.
> - **Decision**: Use a **function pointer table** approach for simplicity with architecture-specific extension points.
> - **Rationale**: The function pointer table provides clean separation between relocation calculation logic while maintaining static type safety and compile-time validation. It's simpler than a full plugin system but more extensible than a monolithic switch statement. Each handler can be implemented in a separate source file, enabling incremental addition of new relocation types.
> - **Consequences**: New relocation types can be added by implementing a handler function and adding it to the table. Architecture support becomes explicit through separate tables. The design naturally supports both x86-64 and potential future architectures like AArch64 or RISC-V.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Monolithic Switch Statement | Simple for few types, no indirection overhead | Scales poorly, mixes unrelated logic, hard to test individually | ❌ |
| Function Pointer Table | Clean separation, easy to add new types, testable handlers | Requires careful initialization, indirection overhead minimal | ✅ |
| Registration-Based Plugin System | Most flexible, runtime configurable, ideal for library | Overly complex for static linker, runtime overhead | ❌ |

**Data Structure Extensions**  
To support multiple relocation types with varying characteristics, the data model requires these additions:

| Structure Name | New Fields | Purpose |
|----------------|------------|---------|
| `RelocationHandler` | `type uint32_t`, `name const char*`, `handler_func RelocHandlerFunc`, `width_bits uint8_t`, `is_pc_relative bool`, `is_signed bool`, `check_overflow bool` | Metadata and handler for a specific relocation type |
| `ArchitectureInfo` | `machine uint16_t`, `word_size uint8_t`, `endianness uint8_t`, `reloc_handlers RelocationHandler*`, `num_handlers uint32_t` | Architecture-specific configuration including supported relocations |
| `RelocationContext` (extended) | `arch const ArchitectureInfo*`, `reloc_type uint32_t`, `handler const RelocationHandler*` | Context passed to relocation handler with architecture info |

**Implementation Integration Points**  
1. **ELF Reader Enhancement**: The `load_relocations` function must preserve the relocation type (`r_info` field) without assuming only two types are present.
2. **Relocation Applier Modification**: The `apply_relocation_to_section` function would consult the architecture's handler table instead of using hardcoded logic:
   ```c (conceptual, not in main body)
   // Pseudo-code for illustration
   const RelocationHandler* handler = find_handler(arch, reloc_type);
   if (!handler) {
       report_error(ERROR_RELOCATION, "Unsupported relocation type %u", reloc_type);
       return;
   }
   uint64_t value = handler->handler_func(reloc, symbol_addr, context);
   ```
3. **New Handler Functions**: Each relocation type gets its own implementation following a standard interface:
   - `handle_R_X86_64_32`: 32-bit absolute relocation with overflow checking
   - `handle_R_X86_64_PLT32`: PLT-relative offset for function calls
   - `handle_R_X86_64_GOTPCREL`: GOT-relative access for position-independent code

**Common Relocation Types to Implement**  
| Relocation Type | Calculation | Purpose | Complexity |
|-----------------|-------------|---------|------------|
| `R_X86_64_32` | `S + A` (32-bit absolute) | 32-bit absolute addressing in small code model | Medium (requires overflow check) |
| `R_X86_64_PLT32` | `L + A - P` (PLT-relative) | Function calls via Procedure Linkage Table | High (requires PLT generation) |
| `R_X86_64_GOTPCREL` | `G + GOT + A - P` | Position-independent access to global data | High (requires GOT generation) |
| `R_X86_64_TLSGD` | `GD + A` (TLS global dynamic) | Thread-local storage access | Very High (requires TLS support) |

**Example Walk-Through: Adding R_X86_64_32 Support**  
Consider a program that uses 32-bit absolute addresses (common in kernel code or embedded systems). The extension would:
1. Add a handler function `handle_R_X86_64_32` that:
   - Calculates `symbol_address + addend`
   - Verifies the result fits in 32 bits using `fits_in_uint32(value)`
   - Writes the truncated 32-bit value using `write_le32(patch_addr, value)`
2. Register this handler in the x86-64 architecture table
3. Update relocation processing to call this handler when encountering type `R_X86_64_32` (value 10)
4. Add test programs that use 32-bit absolute addressing to verify correctness

> **Design Insight**: Relocation handlers should be **pure functions** that take symbol address, relocation context, and addend, then return the value to patch. This makes them easily testable and avoids side effects.

#### Simple Archive Library Support

**Mental Model: Library as a Filing Cabinet of Object Files**  
Imagine archive libraries (.a files) as filing cabinets where each drawer contains related object files. Instead of specifying every individual object file to link, you can specify the cabinet (library), and the linker opens it, examines the index, and extracts only the object files needed to satisfy unresolved symbols—much like looking up topics in a library card catalog and retrieving only relevant books.

**Current Limitation and Extension Scope**  
The current linker requires explicit listing of all object files. Real-world development uses library archives that group related object files with an index of exported symbols. Supporting thin archive libraries (the simplest format) allows the linker to accept `.a` files as input and perform **archive member extraction** based on symbol needs.

**Architecture Decision: Two-Pass Archive Processing**

> **Decision: Lazy Archive Member Extraction with Symbol-Driven Pull**
> - **Context**: Archive libraries contain many object files, but typically only a subset are needed to satisfy undefined references. Loading all members upfront wastes memory and processing time.
> - **Options Considered**:
>   1. **Flatten Archives Early**: Extract all object files from archives before main linking begins (simplest but inefficient).
>   2. **Symbol-Driven Pull**: Process archives during symbol resolution, extracting only members that define needed symbols (efficient but more complex).
>   3. **Multi-Pass with Progress**: Iterate through archives multiple times until no new symbols are resolved (handles mutual dependencies).
> - **Decision**: Use **symbol-driven pull with single pass** for simplicity, falling back to multi-pass if mutual dependencies are detected.
> - **Rationale**: The symbol-driven approach matches traditional Unix linker behavior and minimizes unnecessary work. Starting with single-pass keeps implementation simple, with a clear upgrade path to multi-pass if needed. Archives are processed after regular object files but before reporting undefined symbols.
> - **Consequences**: The linker must understand archive file format, maintain a set of unresolved symbols, and iterate through archive members selectively. Error reporting becomes more complex when symbols remain unresolved after archive processing.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Flatten Archives Early | Simple implementation, deterministic behavior | Wastes resources, doesn't scale for large libraries | ❌ |
| Symbol-Driven Pull | Efficient, matches traditional linkers, minimal memory | Complex implementation, requires archive format parsing | ✅ |
| Multi-Pass with Progress | Handles mutual dependencies between archives | Most complex, potentially multiple iterations | ❌ (initially) |

**Archive File Format Overview**  
Unix archive files (`.a`) have a simple structure that the linker must parse:
- **Global Header**: 8-byte magic string `"!<arch>\n"`
- **Member Headers**: 60-byte records describing each member file
- **Symbol Index**: Special member `"/SYM64/"` or `"/"` containing symbol-to-member mapping
- **Member Data**: Actual object file contents padded to even byte boundaries

**Data Structure Extensions**  
| Structure Name | New Fields | Purpose |
|----------------|------------|---------|
| `ArchiveFile` | `filename char*`, `file_handle FILE*`, `members ArchiveMember*`, `num_members uint32_t`, `symbol_index ArchiveSymbol*`, `num_symbols uint32_t` | In-memory representation of an archive file |
| `ArchiveMember` | `header ArchiveMemberHeader`, `name char*`, `offset uint64_t`, `size uint64_t`, `object ObjectFile*` (lazy-loaded) | Archive member metadata and cached object |
| `ArchiveSymbol` | `name char*`, `member_index uint32_t`, `offset_in_member uint64_t` | Mapping from symbol name to archive member |
| `LinkContext` (extended) | `archives ArchiveFile**`, `num_archives uint32_t`, `unresolved_symbols HashTable*` | Track archives and unresolved symbols during resolution |

**Linking Sequence with Archives**  
The overall linking workflow expands to incorporate archive processing:

1. **Initial Symbol Resolution Pass**: Process all explicitly specified object files, building symbol table with defined and undefined symbols
2. **Archive Processing Phase**:
   - For each archive file in command-line order:
     - Parse archive header and symbol index (but not member contents)
     - For each undefined symbol in unresolved set:
       - Look up symbol in archive index
       - If found, extract the corresponding member (parse as `ObjectFile`)
       - Add member to processing queue as if it were specified on command line
   - Reprocess newly extracted object files through standard resolution
3. **Final Undefined Check**: After all archives processed, report any remaining undefined symbols

**Implementation Integration Points**  
1. **Command Line Parsing Enhancement**: Recognize `.a` files as archives rather than object files, store them separately in `LinkContext`
2. **Symbol Resolver Modification**: `resolve_all_symbols` gains an archive processing phase between initial resolution and final undefined check
3. **New Archive Parser Module**: Functions to parse archive format:
   - `parse_archive_file(filename) returns ArchiveFile*`
   - `extract_archive_member(archive, member_index) returns ObjectFile*`
   - `free_archive_file(archive) returns void`
4. **Circular Dependency Handling**: Basic implementation can warn about mutual dependencies; advanced implementation could implement multiple passes until convergence

**Example Walk-Through: Linking with libc.a**  
Consider linking a simple program that calls `printf`:
1. Command line: `linker main.o libc.a`
2. Initial resolution: `main.o` defines `main`, references `printf` (undefined)
3. Archive processing: Parse `libc.a` index, find `printf` defined in `printf.o` member
4. Extract `printf.o`, add to object list, resolve its symbols
5. `printf.o` may reference other undefined symbols (e.g., `write`), causing further extraction
6. Process continues until all symbols resolved or no more archive members match unresolved symbols

> **Design Insight**: Archive processing exemplifies the **lazy evaluation** pattern—object files are only loaded when their symbols are actually needed. This optimization is particularly valuable for large libraries like libc.

#### Debug Section Preservation

**Mental Model: Preserving Source Map Information**  
Debug sections are like detailed floor plans and construction notes attached to a building blueprint. While the building (executable) functions fine without them, architects (debuggers) need these annotations to understand how source code maps to machine instructions and how variables map to memory locations. Preserving debug information means carefully copying these annotations through the linking process without disrupting their intricate relationships.

**Current Limitation and Extension Scope**  
The core linker strips or ignores debug sections (`.debug_*`, `.zdebug_*`, `.line`, `.strtab`, etc.), producing executables without debugging information. Real linkers preserve these sections, concatenating them like regular sections but with special handling for cross-references between debug sections and the need to update location references within debug data.

**Architecture Decision: Debug Section Pass-Through with Relocation**

> **Decision: Transparent Debug Section Concatenation with Limited Processing**
> - **Context**: Debug sections contain complex, format-specific data (DWARF) that requires sophisticated processing to update location references. Full DWARF processing is extremely complex and beyond educational scope.
> - **Options Considered**:
>   1. **Strip All Debug Sections**: Ignore debug sections entirely (current implementation).
>   2. **Pass-Through Concatenation**: Merge debug sections verbatim without processing internal references.
>   3. **Partial DWARF Relocation**: Update simple location references (addresses) in debug info while ignoring complex relationships.
>   4. **Full DWARF Processing**: Complete DWARF section merging and reference updating (like production linkers).
> - **Decision**: Implement **pass-through concatenation** with optional simple address patching for `.debug_line` sections.
> - **Rationale**: Pass-through concatenation provides working debug information for simple cases (single object file or non-overlapping ranges) while keeping implementation manageable. Adding limited address patching for line number information gives practical utility for debugging. Full DWARF processing would exceed project scope by orders of magnitude.
> - **Consequences**: Debuggers may work for simple programs but fail with complex ones. The extension primarily demonstrates section preservation concepts rather than providing production-quality debug linking.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Strip All Debug Sections | Simplest, no extra code | No debug support, limits usefulness | ❌ |
| Pass-Through Concatenation | Simple implementation, works for simple cases | Debug info may be incorrect for complex programs | ✅ |
| Partial DWARF Relocation | More useful debug info, handles common cases | Complex to implement, partial solution | ❌ |
| Full DWARF Processing | Production-quality debug support | Extremely complex, weeks of work | ❌ |

**Debug Section Types and Requirements**  
| Section Name | Purpose | Processing Required |
|--------------|---------|-------------------|
| `.debug_info` | DWARF debugging information entries | Address updates, reference preservation (complex) |
| `.debug_abbrev` | DWARF abbreviation tables | Concatenation only |
| `.debug_line` | Line number program | Address and offset updates (moderate complexity) |
| `.debug_str` | Debug string table | Concatenation with deduplication |
| `.debug_loc` | Location lists | Address updates (complex) |
| `.debug_ranges` | Address ranges | Address updates (moderate) |

**Data Structure Extensions**  
| Structure Name | New Fields | Purpose |
|----------------|------------|---------|
| `DebugSectionInfo` | `name char*`, `requires_processing bool`, `is_string_table bool`, `merge_strategy DebugMergeStrategy` | Metadata about how to handle specific debug section types |
| `DebugLinkState` | `debug_sections OutputSectionGroup*`, `num_debug_sections uint32_t`, `string_table_offsets HashTable*` | State for tracking debug section merging |
| `OutputSection` (extended) | `is_debug bool`, `debug_format DebugFormat`, `source_sections DebugSourceSection*` | Flag and metadata for debug sections |

**Implementation Strategy**  
The implementation follows a layered approach:

1. **Identification Layer**: Recognize debug sections by name prefix (`.debug_`, `.zdebug_`, `.line`) and set appropriate handling flags
2. **Basic Concatenation**: For simple debug sections (`.debug_abbrev`, `.debug_str`), use standard section merging with special handling:
   - `.debug_str` sections need string deduplication across files
   - Other sections can be concatenated with possible alignment
3. **Address Patching Layer**: For `.debug_line` sections, implement minimal processing:
   - Parse DWARF line number program header
   - Update address registers when encountering `DW_LNE_set_address` ops
   - Leave other opcodes unchanged
4. **Placement Strategy**: Debug sections typically go in non-allocatable segments at the end of the file

**Integration with Existing Components**  
1. **Section Merger Enhancement**: `add_sections_from_object` checks section names and routes debug sections to specialized handling
2. **New Debug Merger Module**: Functions for debug-specific merging:
   - `merge_debug_sections(debug_state, objects, count) returns void`
   - `patch_debug_line_sections(debug_state, symbol_table) returns void`
   - `concatenate_debug_strings(debug_state) returns void`
3. **Executable Writer Modification**: Ensure debug sections are included in output but not in `PT_LOAD` segments

**Example Walk-Through: Preserving .debug_line**  
Consider two object files with debug line information:
1. `main.o`: `.debug_line` section mapping addresses 0x0-0x50 to main.c lines 1-20
2. `util.o`: `.debug_line` section mapping addresses 0x0-0x30 to util.c lines 1-15
3. After merging: `.text` section has main.o code at 0x1000, util.o code at 0x1050
4. Debug merger must:
   - Concatenate `.debug_line` sections
   - Update address in main.o's debug line program by adding 0x1000
   - Update address in util.o's debug line program by adding 0x1050
   - Leave line number information unchanged

> **Design Insight**: Debug information preservation demonstrates that linking involves **multiple parallel transformations**—while code and data sections undergo relocation patching, debug sections need coordinated updates to maintain consistency with the new layout.

#### Cross-Architecture Linking Support

**Mental Model: Universal Translator Between Machine Languages**  
Cross-architecture linking is like translating a book while also converting between measurement systems (metric to imperial) and cultural references. The linker must not only resolve symbols but also handle fundamental differences in instruction sets, data alignment, and relocation types between source and target architectures.

**Current Limitation and Extension Scope**  
The linker is hardcoded for x86-64 (ELF64, little-endian, specific relocation types). Supporting other architectures (32-bit x86, ARM, RISC-V) requires abstracting architecture-specific details while preserving the core linking algorithms.

**Architecture Decision: Architecture Abstraction Layer**

> **Decision: Pluggable Architecture Provider Interface**
> - **Context**: Different CPU architectures have different word sizes, endianness, relocation types, and ABI requirements. Hardcoding x86-64 assumptions throughout the code makes supporting other architectures difficult.
> - **Options Considered**:
>   1. **Conditional Compilation**: `#ifdef X86_64` / `#ifdef ARM64` blocks throughout codebase.
>   2. **Runtime Architecture Detection**: Single codebase with architecture determined at runtime from ELF headers.
>   3. **Abstract Architecture Interface**: Architecture-agnostic core with pluggable providers for architecture-specific operations.
> - **Decision**: Implement **abstract architecture interface** with provider functions for architecture-specific operations.
> - **Rationale**: The interface approach provides clean separation of concerns, enables testing with mock architectures, and allows incremental addition of new architectures. It's more maintainable than conditional compilation and more flexible than runtime detection alone.
> - **Consequences**: Core linking algorithms become parameterized by architecture providers. Initial complexity increases but pays off when adding second architecture.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Conditional Compilation | Simple conceptually, compile-time optimized | Code duplication, hard to maintain, doesn't scale | ❌ |
| Runtime Architecture Detection | Single binary supports multiple architectures | Mixed concerns, harder to debug | ❌ |
| Abstract Architecture Interface | Clean separation, testable, extensible | More upfront design, indirect function calls | ✅ |

**Architecture Interface Definition**  
The architecture abstraction would define these operations:

| Operation | Signature | Purpose |
|-----------|-----------|---------|
| `word_size` | `uint8_t (*)(void)` | Returns 4 for 32-bit, 8 for 64-bit architectures |
| `endianness` | `Endian (*)(void)` | Returns `ENDIAN_LITTLE` or `ENDIAN_BIG` |
| `default_machine` | `uint16_t (*)(void)` | Returns `EM_X86_64`, `EM_ARM`, etc. |
| `get_relocation_handler` | `RelocationHandler* (*)(uint32_t type)` | Returns handler for specific relocation type |
| `create_program_headers` | `Elf64_Phdr* (*)(const MergedSections*, size_t*)` | Generates appropriate program headers for architecture |
| `apply_abi_rules` | `void (*)(SymbolTable*, MergedSections*)` | Applies ABI-specific rules (stack alignment, TLS, etc.) |

**Data Model Extensions**  
| Structure Name | New Fields | Purpose |
|----------------|------------|---------|
| `Architecture` | `name const char*`, `machine uint16_t`, `ops ArchitectureOps*` | Architecture descriptor with function table |
| `LinkContext` (extended) | `target_arch const Architecture*` | Target architecture for linking |
| `ObjectFile` (extended) | `source_arch const Architecture*` | Architecture of input object file (for validation) |

**Implementation Approach**  
1. **Architecture Detection**: Determine target architecture from:
   - Command-line option (`--architecture=x86_64`)
   - First input object file's `e_machine` field
   - Default to host architecture
2. **Validation Layer**: Verify all input files match target architecture (or implement fat binary support)
3. **Parameterized Components**:
   - **ELF Reader**: Uses architecture's endianness for reading multi-byte values
   - **Relocation Applier**: Uses architecture's relocation handler table
   - **Executable Writer**: Uses architecture's program header creation
4. **Architecture Implementations**:
   - `arch_x86_64.c`: x86-64 specific implementations
   - `arch_aarch64.c`: ARM64 implementations (future)
   - `arch_riscv64.c`: RISC-V 64-bit implementations (future)

**Cross-Architecture Scenarios**  
1. **Same Architecture Linking**: All input files match target architecture (normal case)
2. **Mixed Architecture Detection**: Input files with different `e_machine` values trigger error
3. **Cross-Compilation Support**: Linking 32-bit x86 objects on 64-bit host requires:
   - 32-bit word size handling
   - Different relocation types (`R_386_PC32` vs `R_X86_64_PC32`)
   - Different program header expectations

**Example Walk-Through: Adding ARM64 Support**  
To add ARM AArch64 support:
1. Create `arch_aarch64.c` implementing `ArchitectureOps`:
   - `word_size`: returns 8
   - `endianness`: returns `ENDIAN_LITTLE` (typical)
   - `machine`: returns `EM_AARCH64` (183)
   - Relocation handlers for `R_AARCH64_ABS64`, `R_AARCH64_PREL32`, etc.
2. Update ELF reader to handle AArch64 relocations section format
3. Add AArch64 test programs compiled with `aarch64-linux-gnu-gcc`
4. Verify output with `readelf` and `qemu-aarch64`

> **Design Insight**: The architecture abstraction demonstrates the **strategy pattern**—defining a family of algorithms (architecture-specific operations), encapsulating each one, and making them interchangeable. The linking algorithm becomes a template method that calls architecture-specific operations at well-defined points.

#### Integration with Compiler Toolchain

**Mental Model: Linker as Specialized Factory in Manufacturing Pipeline**  
In a manufacturing pipeline, the linker is the final assembly station that receives semi-finished components (object files) from earlier stations (compiler, assembler) and produces finished products (executables). Integration means establishing standard interfaces and protocols between stations so components fit together correctly, with proper documentation (debug info) and quality control (validation).

**Current Limitation and Extension Scope**  
The standalone linker operates independently without awareness of the broader compilation ecosystem. Real linkers integrate with compilers through start files (`crt0.o`), compiler runtime libraries (`libgcc.a`), and language-specific initialization. They also support linker scripts for detailed control over memory layout.

**Simple Integration Extensions**  
1. **Start File Support**: Automatic inclusion of C runtime initialization code
2. **Compiler Runtime Libraries**: Automatic linking of `libgcc` for arithmetic operations
3. **Linker Scripts**: Basic support for memory layout control via simple script language
4. **Response Files**: Support for `@file` syntax to handle large numbers of input files

**Implementation Priorities**  
| Extension | Complexity | Value | Recommended Order |
|-----------|------------|-------|------------------|
| Response Files | Low | High (practical utility) | 1 |
| Start File Support | Medium | Medium (enables standalone C programs) | 2 |
| Basic Linker Scripts | High | High (layout control) | 3 |
| Compiler Runtime | Medium | Low (only needed for certain operations) | 4 |

**Data Flow with Toolchain Integration**  
The enhanced linker would participate in a larger ecosystem:

```
Compiler Driver (gcc/clang)
    ↓ (invokes with correct options)
Static Linker (ours)
    ↓ (automatically includes)
Start Files (crt1.o, crti.o, crtn.o)
    ↓ (processes according to)
Linker Script (optional)
    ↓ (produces)
Executable with proper startup code
```

> **Design Insight**: Toolchain integration transforms the linker from a **standalone utility** to a **cooperative component** in a larger system. This requires thinking about interfaces, conventions, and error reporting that work well in automated build environments.

### Implementation Guidance

**Technology Recommendations for Extensions**  
| Extension Area | Simple Option | Advanced Option |
|----------------|---------------|-----------------|
| Relocation Types | Hand-coded function table with switch fallback | Auto-generated handlers from architecture specification files |
| Archive Libraries | Basic BSD archive format only | Support for both BSD and GNU (thin) archive formats |
| Debug Sections | Pass-through concatenation only | Partial DWARF processing using libdwarf library |
| Cross-Architecture | Runtime detection with if-else chains | Abstract interface with plugin loading |
| Toolchain Integration | Hardcoded start file paths | Configuration file with search paths |

**Recommended File Structure for Extensions**  
```
project-root/
  src/
    core/                    # Existing core components
      elf_reader.c
      section_merger.c
      symbol_resolver.c
      relocation_applier.c
      executable_writer.c
    extensions/             # New extension modules
      relocation_handlers/  # Additional relocation types
        reloc_x86_64.c      # x86-64 specific handlers
        reloc_common.c      # Common relocation utilities
        reloc_table.c       # Handler registry
      archives/             # Archive library support
        archive_parser.c    # Archive format parsing
        archive_extract.c   # Member extraction logic
      debug/                # Debug section handling
        debug_merger.c      # Debug section concatenation
        dwarf_utils.c       # Simple DWARF helpers
      architectures/        # Cross-architecture support
        arch_interface.h    # Abstract architecture interface
        arch_x86_64.c       # x86-64 implementation
        arch_aarch64.c      # ARM64 implementation (future)
    include/
      linker.h              # Main public header
      extensions.h          # Extension function prototypes
  tests/
    extensions/             # Tests for extensions
      test_archives.c
      test_debug.c
```

**Infrastructure Starter Code: Archive Parser**  
Here's complete, working code for parsing simple BSD archive format:

```c
/* archive_parser.c - Basic BSD archive format parser */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "archive_parser.h"

/* BSD archive member header (60 bytes) */
typedef struct {
    char name[16];      /* File name (null-padded) */
    char timestamp[12]; /* Decimal modification time */
    char owner[6];      /* Decimal user ID */
    char group[6];      /* Decimal group ID */
    char mode[8];       /* Octal file mode */
    char size[10];      /* Decimal size in bytes */
    char magic[2];      /* `\n` */
} ArchiveMemberHeader;

/* Parse decimal string with possible trailing spaces */
static uint64_t parse_decimal(const char* str, size_t len) {
    uint64_t value = 0;
    for (size_t i = 0; i < len && str[i] >= '0' && str[i] <= '9'; i++) {
        value = value * 10 + (str[i] - '0');
    }
    return value;
}

/* Check if header is special index member */
static int is_special_member(const ArchiveMemberHeader* hdr) {
    return (hdr->name[0] == '/' && hdr->name[1] != ' ');
}

ArchiveFile* parse_archive_file(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        report_error(ERROR_INPUT_VALIDATION, "Archive", 
                     "Cannot open archive file: %s", filename);
        return NULL;
    }
    
    /* Check global magic */
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "!<arch>\n", 8) != 0) {
        fclose(f);
        report_error(ERROR_INPUT_VALIDATION, "Archive",
                     "Invalid archive magic in %s", filename);
        return NULL;
    }
    
    ArchiveFile* archive = xcalloc(1, sizeof(ArchiveFile));
    archive->filename = strdup(filename);
    archive->file_handle = f;
    
    /* First pass: count members */
    uint32_t member_count = 0;
    long pos = 8;  /* Skip global magic */
    
    while (1) {
        ArchiveMemberHeader hdr;
        if (fseek(f, pos, SEEK_SET) != 0) break;
        if (fread(&hdr, sizeof(hdr), 1, f) != 1) break;
        
        uint64_t size = parse_decimal(hdr.size, sizeof(hdr.size));
        if (size == 0) break;
        
        member_count++;
        pos += sizeof(ArchiveMemberHeader) + ((size + 1) & ~1); /* Even padding */
    }
    
    /* Allocate member array */
    archive->members = xcalloc(member_count, sizeof(ArchiveMember));
    archive->num_members = member_count;
    
    /* Second pass: populate member info */
    pos = 8;
    for (uint32_t i = 0; i < member_count; i++) {
        ArchiveMemberHeader hdr;
        fseek(f, pos, SEEK_SET);
        fread(&hdr, sizeof(hdr), 1, f);
        
        uint64_t size = parse_decimal(hdr.size, sizeof(hdr.size));
        archive->members[i].offset = pos + sizeof(ArchiveMemberHeader);
        archive->members[i].size = size;
        
        /* Extract member name */
        if (hdr.name[0] == '/' && isdigit(hdr.name[1])) {
            /* Long name reference - would need string table */
            archive->members[i].name = strdup("<longname>");
        } else {
            /* Regular name (null-terminated within 16 chars) */
            char name[17];
            memcpy(name, hdr.name, 16);
            name[16] = '\0';
            /* Trim trailing spaces */
            size_t len = strcspn(name, " ");
            name[len] = '\0';
            archive->members[i].name = strdup(name);
        }
        
        pos += sizeof(ArchiveMemberHeader) + ((size + 1) & ~1);
    }
    
    return archive;
}

ObjectFile* extract_archive_member(ArchiveFile* archive, uint32_t member_idx) {
    if (member_idx >= archive->num_members) return NULL;
    
    ArchiveMember* member = &archive->members[member_idx];
    
    /* Lazy loading: parse object file on demand */
    if (member->object == NULL) {
        FILE* f = archive->file_handle;
        uint8_t* data = xmalloc(member->size);
        
        fseek(f, member->offset, SEEK_SET);
        if (fread(data, 1, member->size, f) != member->size) {
            free(data);
            return NULL;
        }
        
        /* Create temporary file or parse from memory */
        char tempname[64];
        snprintf(tempname, sizeof(tempname), "/tmp/archive_member_%u.o", member_idx);
        FILE* tmp = fopen(tempname, "wb");
        fwrite(data, 1, member->size, tmp);
        fclose(tmp);
        
        member->object = read_elf_file(tempname);
        unlink(tempname);
        free(data);
    }
    
    return member->object;
}

void free_archive_file(ArchiveFile* archive) {
    if (!archive) return;
    
    for (uint32_t i = 0; i < archive->num_members; i++) {
        free(archive->members[i].name);
        if (archive->members[i].object) {
            free_object_file(archive->members[i].object);
        }
    }
    
    free(archive->members);
    free(archive->filename);
    if (archive->file_handle) fclose(archive->file_handle);
    free(archive);
}
```

**Core Logic Skeleton: Relocation Handler Registry**  
```c
/* reloc_table.c - Relocation handler registry implementation */

typedef struct {
    uint32_t type;
    const char* name;
    RelocHandlerFunc handler;
    uint8_t width_bits;     /* Width of field to patch (8, 16, 32, 64) */
    bool is_pc_relative;
    bool is_signed;
    bool check_overflow;
} RelocationHandler;

/* Global handler table for x86-64 */
static RelocationHandler x86_64_handlers[] = {
    {R_X86_64_64, "R_X86_64_64", handle_R_X86_64_64, 64, false, false, false},
    {R_X86_64_PC32, "R_X86_64_PC32", handle_R_X86_64_PC32, 32, true, true, true},
    {R_X86_64_32, "R_X86_64_32", handle_R_X86_64_32, 32, false, false, true},
    {R_X86_64_32S, "R_X86_64_32S", handle_R_X86_64_32S, 32, false, true, true},
    /* Add more handlers here as implemented */
};

/* Architecture descriptor for x86-64 */
static const Architecture x86_64_arch = {
    .name = "x86_64",
    .machine = EM_X86_64,
    .word_size = 8,
    .endianness = ENDIAN_LITTLE,
    .reloc_handlers = x86_64_handlers,
    .num_handlers = sizeof(x86_64_handlers) / sizeof(x86_64_handlers[0])
};

const RelocationHandler* find_relocation_handler(const Architecture* arch, 
                                                 uint32_t reloc_type) {
    for (uint32_t i = 0; i < arch->num_handlers; i++) {
        if (arch->reloc_handlers[i].type == reloc_type) {
            return &arch->reloc_handlers[i];
        }
    }
    return NULL;
}

/* Example handler implementation for R_X86_64_32 */
uint64_t handle_R_X86_64_32(const RelocationContext* ctx, 
                           uint64_t symbol_addr, 
                           int64_t addend) {
    uint64_t value = symbol_addr + addend;
    
    /* TODO 1: Check for overflow in 32-bit unsigned range */
    /* if (value > UINT32_MAX) report overflow error */
    
    /* TODO 2: Apply any ABI-specific adjustments */
    /* None needed for R_X86_64_32 */
    
    /* TODO 3: Return value to be written */
    return value;
}

/* Updated relocation application using handler registry */
void apply_relocation_with_handler(const RelocationContext* ctx,
                                   const RelocationHandler* handler,
                                   uint8_t* patch_addr,
                                   uint64_t symbol_addr,
                                   int64_t addend) {
    /* TODO 1: Calculate value using handler function */
    /* uint64_t value = handler->handler(ctx, symbol_addr, addend); */
    
    /* TODO 2: Check overflow if handler requires it */
    /* if (handler->check_overflow) { ... } */
    
    /* TODO 3: Write value with appropriate width and signedness */
    /* switch (handler->width_bits) { case 32: write_le32(...); break; } */
    
    /* TODO 4: Update statistics or debug information */
}
```

**Language-Specific Hints for C Extensions**  
- **Endianness Handling**: Use `htole64`, `le64toh` functions from `<endian.h>` for portable endian conversion when adding cross-architecture support
- **Dynamic Dispatch**: Implement architecture function tables using structs of function pointers for clean abstraction
- **Memory Management**: For archive support, use `mmap` for large archive files instead of reading entire files into memory
- **Debug Section Parsing**: Consider using `libdwarf` and `libelf` for production-quality debug handling rather than reimplementing DWARF
- **Configuration**: Use `getauxval(AT_PLATFORM)` on Linux to detect host architecture at runtime

**Milestone Checkpoint for Archive Extension**  
After implementing archive support:
- **Test Command**: `./linker test/main.o lib/libutil.a`
- **Expected Behavior**: Linker should extract only needed object files from libutil.a
- **Verification**: Run `readelf -s output.exe | grep util_` to verify utility functions are present
- **Debugging**: Use `--verbose` flag to see which archive members are extracted
- **Common Issue**: If undefined symbols remain, check archive index parsing; use `ar t libutil.a` and `nm libutil.a` to verify archive contents

**Debugging Tips for Extensions**  
| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Relocation overflow error with new relocation type | Handler calculating wrong value or overflow check too strict | Use `hexdump` to see calculated vs expected value, check addend handling | Adjust calculation formula or overflow bounds |
| Archive member not extracted | Symbol not in archive index or index parsing bug | Run `nm -s lib.a` to see archive index, compare with linker's parsed index | Fix index parsing or symbol lookup logic |
| Debugger shows wrong line numbers | Debug line section addresses not updated | Use `readelf -wl output.exe` to see line table, check address ranges | Implement `.debug_line` address patching |
| Executable crashes on different architecture | Wrong program headers or alignment | Compare `readelf -l` output with working executable from native linker | Fix architecture-specific segment creation |
| Linker uses wrong architecture | Detection logic error or mixed input files | Check `e_machine` field of all inputs with `readelf -h`, verify detection logic | Improve architecture detection or add `--architecture` flag |

---


## Glossary

> **Milestone(s):** This section provides a reference for all technical terms used throughout the document, supporting understanding across all milestones.

The static linker domain contains specialized terminology that can be challenging for developers new to systems programming and binary formats. This glossary provides precise definitions for all key terms, acronyms, and domain-specific vocabulary used throughout this design document. Each entry includes the term's definition and a reference to where it first appears or receives detailed explanation in the document structure.

### Terminology Table

| Term | Definition | First Appears In |
|------|------------|------------------|
| **Absolute relocation** | A relocation type that requires the linker to compute the absolute virtual address of a symbol and write it directly into the relocation site. Contrasts with PC-relative relocation. | Component: Relocation Applier |
| **Addend** | A constant value stored in the relocation entry (`r_addend` field) that is added to the computed symbol address during relocation calculation. The addend allows the compiler to encode offsets like `&array[5]` as `array + 20` (assuming 4-byte elements). | Component: Relocation Applier |
| **Alignment padding** | Empty bytes (typically zeros) inserted between sections or within section data to ensure the next section or data structure starts at an address that meets its alignment requirements (`sh_addralign`). | Component: Section Merger |
| **Archive file** | A collection of object files stored in a single file using the BSD archive format (also known as a static library, typically with `.a` extension). The linker can extract specific object files from the archive based on symbol dependencies. | Future Extensions |
| **Archive member extraction** | The process of selectively loading object files from an archive based on which symbols they define that are needed to resolve undefined references. This avoids linking unnecessary object files into the final executable. | Future Extensions |
| **Architecture abstraction layer** | A design pattern that isolates architecture-specific operations (like relocation handling and byte order) from the core linking logic, enabling support for multiple CPU architectures without rewriting the entire linker. | Future Extensions |
| **BSD archive format** | The traditional Unix archive format used for static libraries, consisting of a global file signature followed by member headers and file contents. Each member header contains metadata like file name, size, and modification timestamp. | Future Extensions |
| **Cascading errors** | Secondary errors that occur as a consequence of an earlier unresolved issue. For example, an undefined symbol error might cause multiple relocation errors at all sites that reference that symbol. | Error Handling and Edge Cases |
| **Cleanup orchestration** | The coordinated deallocation of all allocated resources (memory, file handles, etc.) when the linker finishes processing, whether due to successful completion or error termination. | Error Handling and Edge Cases |
| **COMMON symbol** | A special type of uninitialized global variable symbol (with section index `SHN_COMMON`) that represents a "tentative definition." COMMON symbols from multiple object files are merged into a single definition with the largest size among all definitions. | Component: Symbol Resolver |
| **Component** | A self-contained module with specific responsibility in the linker architecture, following the single responsibility principle. The five main components are ELF Reader, Section Merger, Symbol Resolver, Relocation Applier, and Executable Writer. | High-Level Architecture |
| **Cross-architecture linking** | Linking object files compiled for different CPU architectures (e.g., x86-64 and ARM64). This is not supported by the basic static linker but could be added through the architecture abstraction layer. | Future Extensions |
| **Data flow** | The movement of data structures between components during the linking process, showing how information transforms from raw object files through intermediate representations to the final executable. | Interactions and Data Flow |
| **Debug section pass-through** | A strategy for handling debug sections (like `.debug_info`) where the sections are copied verbatim from input object files to the output executable without attempting to process or merge internal references, since debug information is consumed by debuggers, not the runtime loader. | Future Extensions |
| **Debugging** | The process of diagnosing and fixing errors in the linker implementation or in the programs being linked, using tools like `readelf`, `objdump`, and custom diagnostics. | Debugging Guide |
| **Defense-in-depth** | An error handling strategy employing multiple validation layers at different abstraction levels to catch issues early, such as validating ELF headers, checking section flags consistency, and verifying relocation calculations. | Error Handling and Edge Cases |
| **Diagnostic** | Information produced by the linker to aid debugging, including error messages, warnings, and informational output about the linking process. | Debugging Guide |
| **Diagnostic context** | Additional information included in error messages to aid debugging, such as file names, line numbers (if available), symbol names, section names, and offset values that help pinpoint the source of problems. | Error Handling and Edge Cases |
| **DWARF** | Debugging With Arbitrary Record Format, the standard format for debug information in ELF files. DWARF sections contain information about source code structure, variables, and line numbers for use by debuggers. | Future Extensions |
| **ELF** | Executable and Linkable Format, the standard binary file format used by Unix-like systems for object files, executables, shared libraries, and core dumps. The format defines headers, section tables, program headers, and data organization. | Context and Problem Statement |
| **Endianness** | The byte ordering of multi-byte values in memory and files. Little-endian stores the least significant byte first (used by x86-64), while big-endian stores the most significant byte first. The linker must handle the endianness of the target architecture. | Component: ELF Reader |
| **Entry point** | The virtual address where execution begins when the operating system transfers control to the executable, specified in the ELF header's `e_entry` field. For standalone programs without libc, this typically points to the `_start` symbol. | Component: Executable Writer |
| **Error category** | Classification of errors by origin and nature, used to organize error handling and reporting. Categories include input validation, symbol resolution, relocation, layout, resource, and internal errors. | Error Handling and Edge Cases |
| **Error propagation** | A mechanism where errors detected in one component prevent further processing in the pipeline, avoiding cascading errors and wasted computation on invalid intermediate states. | Error Handling and Edge Cases |
| **Fail-fast** | An error handling philosophy where the linker immediately terminates with detailed diagnostics upon detecting the first error, rather than attempting to continue and potentially produce misleading secondary errors. | Error Handling and Edge Cases |
| **Graceful degradation** | The linker's ability to clean up allocated resources and provide informative error output before terminating, rather than crashing or leaking resources. | Error Handling and Edge Cases |
| **Hexdump** | A tool or function that displays binary data in hexadecimal and ASCII representation, useful for examining the raw bytes of sections, headers, and patched data during debugging. | Debugging Guide |
| **InputSectionMapping** | A data structure that records the mapping between an input section (from a specific object file) and its location in the merged output section, including the file index, section index, and offset within the output section. | Data Model |
| **Linking sequence** | The ordered steps of the linking process from reading input object files through section merging, symbol resolution, relocation application, to writing the executable. | Interactions and Data Flow |
| **Little-endian** | Byte order where the least significant byte of a multi-byte value comes first in memory. x86-64 processors use little-endian ordering, and the ELF format specifies the endianness in the header. | Component: ELF Reader |
| **Ownership semantics** | Clear rules defining which component is responsible for allocating and freeing resources (memory, file handles, etc.), preventing memory leaks and double-free errors. | Error Handling and Edge Cases |
| **Page alignment** | The requirement that virtual addresses and file offsets for memory segments be multiples of the system page size (typically 4096 bytes). This is necessary because the operating system loads program segments using memory-mapping operations that work at page granularity. | Component: Executable Writer |
| **PC-relative** | Addressing mode where the target address is specified relative to the program counter (instruction pointer). PC-relative relocations compute the difference between the symbol address and the relocation site address (plus addend). | Component: Relocation Applier |
| **Pipeline architecture** | An architectural style where components are arranged in a sequence where the output of one component becomes the input to the next, with minimal feedback loops. The static linker follows this pattern from ELF reading through executable writing. | High-Level Architecture |
| **Program headers** | ELF structures (type `Elf64_Phdr`) that describe how parts of the file should be loaded into memory. Each program header defines a segment with type, permissions, file offset, virtual address, and size information for the operating system loader. | Component: Executable Writer |
| **PT_LOAD** | Program header type (value 1) indicating a loadable segment that should be mapped into memory by the operating system loader. Executables typically have at least two PT_LOAD segments: one for text (RX) and one for data (RW). | Component: Executable Writer |
| **PT_LOAD segment** | A loadable program segment that will be mapped into memory when the executable is loaded. Each PT_LOAD segment has associated permissions (read, write, execute) and must be page-aligned in both file offset and virtual address. | Component: Executable Writer |
| **p_filesz** | Field in a program header specifying the size of the segment in the file. For sections like `.bss` that occupy memory but not file space, `p_filesz` may be smaller than `p_memsz`, with the difference representing zero-initialized memory. | Component: Executable Writer |
| **p_memsz** | Field in a program header specifying the size of the segment in memory. This includes both file-backed data and zero-initialized memory (like `.bss`). The loader allocates `p_memsz` bytes in virtual memory, zeroing any bytes beyond `p_filesz`. | Component: Executable Writer |
| **Relocation** | An instruction to the linker to patch an address or offset in the section data. Relocation entries specify the location to patch (offset within a section), the symbol whose address is needed, the type of calculation required, and an optional addend. | Context and Problem Statement |
| **Relocation calculation** | The process of computing the value to write into a relocation site, based on the symbol's final address, the relocation type, the addend, and (for PC-relative relocations) the address of the relocation site itself. | Component: Relocation Applier |
| **Relocation handler registry** | A table mapping relocation type numbers (like `R_X86_64_PC32`) to handler functions that know how to compute and apply that specific relocation type. This enables extensible support for multiple relocation types across different architectures. | Future Extensions |
| **Relocation site** | The location in code or data where an address needs to be patched during relocation processing. The relocation entry's `r_offset` field specifies this location relative to the beginning of its containing section. | Component: Relocation Applier |
| **Response files** | Files containing command-line arguments, used to overcome operating system command-line length limits when linking large numbers of object files or libraries. | Future Extensions |
| **Section** | A contiguous chunk of data or code in an ELF file with a specific purpose, such as executable instructions (`.text`), initialized data (`.data`), read-only data (`.rodata`), or uninitialized data (`.bss`). Sections are described by section headers. | Data Model |
| **Section merging** | The process of concatenating sections of the same type/name from multiple object files into contiguous output sections in the final executable, applying proper alignment padding between contributions from different files. | Context and Problem Statement |
| **sh_addralign** | Field in a section header specifying the alignment requirement for the section's starting address. The value must be a power of two, and the linker must ensure the section starts at an address that is a multiple of this value. | Component: Section Merger |
| **SHT_NOBITS** | Section type (value 8) indicating that the section occupies no space in the file but occupies memory during execution. The `.bss` section has this type, representing uninitialized data that should be zero-filled at load time. | Component: Section Merger |
| **Static linker** | A program that combines multiple compiled object files into a single executable binary by resolving symbolic references, merging sections, applying relocations, and generating executable headers. Contrasts with dynamic linker that loads shared libraries at runtime. | Context and Problem Statement |
| **Strong symbol** | A symbol definition with global binding (`STB_GLOBAL`) that takes precedence over weak symbols (`STB_WEAK`) during symbol resolution. If multiple strong symbols with the same name exist, it's a multiple definition error. | Component: Symbol Resolver |
| **Symbol** | A named reference to a memory location, function, or data object. Symbols have attributes like name, binding (local/global/weak), type (function/object), size, and value (address or size for COMMON symbols). | Data Model |
| **Symbol binding** | ELF attribute (`st_info` field) indicating the visibility and linking behavior of a symbol: `STB_LOCAL` (visible only within the object file), `STB_GLOBAL` (visible to all object files), or `STB_WEAK` (global but overridable). | Component: Symbol Resolver |
| **Symbol resolution** | The process of matching symbol references (uses) to symbol definitions across multiple object files, handling duplicate definitions according to strong/weak rules, and detecting undefined references that lack any definition. | Context and Problem Statement |
| **Symbol Resolver** | The component that builds a global symbol table across all input object files, resolves references to definitions, handles strong/weak symbol conflicts, and detects undefined symbols. | Component: Symbol Resolver |
| **Symbol-driven pull** | An archive processing approach where members of an archive (static library) are extracted only when they define symbols needed to resolve currently undefined references, minimizing the size of the final executable. | Future Extensions |
| **Tentative definition** | A COMMON symbol representing an uninitialized global variable declared without an initializer. Tentative definitions from multiple translation units are merged into a single definition with the largest size. | Component: Symbol Resolver |
| **Toolchain integration** | Coordinating the linker with other build tools like the compiler, assembler, and archiver, potentially through standardized interfaces, response files, or common metadata formats. | Future Extensions |
| **Truncation overflow** | An error that occurs when a computed address value does not fit in the target field specified by a relocation type (e.g., when a 64-bit address must be stored in a 32-bit field and the value exceeds the 32-bit range). | Component: Relocation Applier |
| **Two-pass resolution** | A symbol resolution algorithm that collects all symbol definitions from all object files in the first pass, then resolves references in the second pass with complete knowledge of all available definitions. This handles forward references correctly. | Component: Symbol Resolver |
| **Validation layer** | A component or function that checks specific constraints before processing continues, such as validating ELF magic numbers, checking section flag consistency, or verifying that relocation values fit in their target fields. | Error Handling and Edge Cases |
| **Weak symbol** | A symbol definition with weak binding (`STB_WEAK`) that can be overridden by a strong symbol with the same name. If no strong definition exists, the weak definition is used. Multiple weak definitions typically choose the first encountered. | Component: Symbol Resolver |

---

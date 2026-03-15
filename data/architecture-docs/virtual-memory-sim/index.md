# 🎯 Project Charter: Virtual Memory Simulator
## What You Are Building
A complete virtual memory subsystem simulator that processes memory access traces through the full translation pipeline: virtual address decomposition, TLB caching, multi-level page table walks, demand paging with page fault handling, and swap-backed page replacement. Your simulator will implement four replacement algorithms (FIFO, LRU, Clock, Optimal) and produce detailed statistics demonstrating real-world performance phenomena like TLB miss storms, Bélády's anomaly, and thrashing detection.
## Why This Project Exists
Every pointer you've ever used is a virtual address—a fictional number that the CPU translates to a completely different physical address on every memory access. This translation is the foundation of process isolation, memory protection, and efficient resource sharing. Yet most developers treat virtual memory as a black box. Building this simulator exposes the hardware-software contract that makes modern computing possible: how page tables map virtual to physical, why TLBs are critical for performance, how multi-level tables compress sparse address spaces, and why page replacement algorithms can make or break system performance.
## What You Will Be Able to Do When Done
- Parse memory access traces and translate virtual addresses to physical addresses
- Implement single-level and multi-level (2-3 level) page tables with on-demand allocation
- Build a TLB cache with ASID tagging and LRU eviction
- Detect and handle page faults with demand paging frame allocation
- Implement four page replacement algorithms (FIFO, LRU, Clock, Optimal) and compare their behavior
- Simulate swap space for dirty page write-back and reload on fault
- Demonstrate Bélády's anomaly where adding frames can *increase* page faults
- Detect thrashing when working set exceeds available physical memory
- Measure and compare TLB hit rates, page fault rates, and memory overhead
## Final Deliverable
~3,000-4,000 lines of C across 15+ source files implementing a complete virtual memory simulator. Processes trace files in `R/W 0xADDRESS` format. Reports translation statistics, TLB hit rates, page fault rates, memory overhead comparisons, and algorithm performance. Demonstrates the same phenomena (TLB miss storms, Bélády's anomaly, thrashing) that production systems encounter.
## Is This Project For You?
**You should start this if you:**
- Are comfortable with C pointers, bitwise operations, and memory management
- Understand basic data structures (arrays, linked lists, hash tables)
- Have some exposure to memory hierarchy concepts (cache, RAM, disk)
- Can work with hexadecimal numbers and bit manipulation
**Come back after you've learned:**
- C pointers and dynamic memory allocation (`malloc`, `free`, pointer arithmetic)
- Bitwise operations (shift, AND, OR, mask extraction)
- Basic file I/O in C (`fopen`, `fgets`, `sscanf`)
## Estimated Effort
| Phase | Time |
|-------|------|
| Single-Level Page Table and Address Translation | ~4-6 hours |
| TLB (Translation Lookaside Buffer) | ~5-7 hours |
| Multi-Level Page Tables | ~5-8 hours |
| Page Replacement and Swap Simulation | ~8-12 hours |
| **Total** | **~25-40 hours** |
## Definition of Done
The project is complete when:
- Trace file parser correctly reads `R/W 0xADDRESS` format with comments and blank lines
- Virtual addresses are correctly decomposed into VPN/offset (single-level) or directory/table/offset (multi-level)
- TLB achieves >95% hit rate on locality-friendly traces and correctly handles ASID context switches
- All four replacement algorithms (FIFO, LRU, Clock, Optimal) produce correct fault counts on standard test traces
- Bélády's anomaly is demonstrable: FIFO with 4 frames produces more faults than 3 frames on the specific anomaly trace
- Dirty pages are written to simulated swap on eviction and correctly reloaded on subsequent page faults
- Working set tracker detects thrashing when distinct pages accessed exceed available frames
- Statistics reporter outputs total accesses, page faults, TLB hit rate, swap reads/writes, and memory overhead comparison

---

# 📚 Before You Read This: Prerequisites & Further Reading
> **Read these first.** The Atlas assumes you are familiar with the foundations below.
> Resources are ordered by when you should encounter them — some before you start, some at specific milestones.
---
## Foundational Operating Systems Concepts
### 📖 Read BEFORE Starting This Project
**What Every Computer Scientist Should Know About Virtual Memory**
- **Paper**: Ulrich Drepper, "What Every Programmer Should Know About Memory" (2007)
- **Best Explanation**: Section 3.3 on virtual memory and page tables
- **Why**: The definitive guide to how memory actually works on modern systems, from CPU caches through virtual memory to NUMA. Read sections 1-3 before starting Milestone 1.
- **When**: Before Milestone 1 — provides the mental model for why virtual memory exists.
**Operating Systems: Three Easy Pieces — Virtualization**
- **Book**: Remzi Arpaci-Dusseau and Andrea Arpaci-Dusseau, *Operating Systems: Three Easy Pieces* (free online)
- **Best Explanation**: Chapters 18-22 on paging and address translation
- **Why**: The clearest pedagogical explanation of virtual-to-physical translation, multi-level page tables, and TLBs. Work through the exercises.
- **When**: Before Milestone 1 — required foundational knowledge.
---
## Milestone 1: Single-Level Page Tables
### 📖 Read Before Milestone 1
**The x86 Paging Mechanism**
- **Spec**: Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 3, Chapter 4 ("Paging")
- **Best Explanation**: Sections 4.1-4.3 on 32-bit paging with 4KB pages
- **Why**: The actual hardware specification you're simulating. Understanding the real PTE bit layout (present, R/W, dirty, accessed) makes the implementation concrete.
- **When**: Before Milestone 1 — reference while implementing PTE structure.
### 🔍 Deep Dive: Address Translation
**Linux Kernel Page Table Walking**
- **Code**: Linux kernel source, `arch/x86/mm/pagetable.c` — function `walk_pte()`
- **Why**: Production-quality page table walking code. Shows how the kernel handles translation, permission checking, and atomicity.
- **When**: After Milestone 1 — compare your implementation to the real thing.
---
## Milestone 2: TLB (Translation Lookaside Buffer)
### 📖 Read Before Milestone 2
**TLB Design and Performance**
- **Paper**: Jeffrey C. Mogul and Anita Borg, "The Impact of TLB Miss Handling on Kernel Performance" (1991), USENIX
- **Best Explanation**: Section 2 on TLB organization and miss costs
- **Why**: Classic paper establishing why TLBs matter. Quantifies the 10-100x cost difference between TLB hits and misses.
- **When**: Before Milestone 2 — motivates why you're implementing TLB caching.
### 🔍 Deep Dive: TLB Coherency
**Translation Lookaside Buffer Management in Shared Memory Systems**
- **Paper**: Alan L. Cox and Willy Zwaenepoel, "Address Translation for Shared Memory Multiprocessors" (1996)
- **Best Explanation**: Section 3 on TLB shootdowns and coherency
- **Why**: Explores the hard problem of keeping TLBs consistent when page tables change. Your `tlb_invalidate_entry()` function solves a subset of this.
- **When**: After Milestone 2 — when you understand the basic TLB, this shows the real-world complexity.
### 💻 Reference Implementation
**QEMU TLB Implementation**
- **Code**: QEMU source, `accel/tcg/cputlb.c` — functions `tlb_lookup()` and `tlb_fill()`
- **Why**: Software TLB implementation used in one of the most popular emulators. Shows how virtualization handles TLB misses.
- **When**: After Milestone 2 — study a production TLB implementation.
---
## Milestone 3: Multi-Level Page Tables
### 📖 Read Before Milestone 3
**The Case for Hierarchical Page Tables**
- **Paper**: David R. Cheriton and Willy Zwaenepoel, "The Design and Implementation of a Segmented Virtual Memory System" (1985)
- **Best Explanation**: Section 2 on hierarchical page table structure
- **Why**: Original motivation for multi-level tables — shows the memory savings calculation you'll demonstrate in this milestone.
- **When**: Before Milestone 3 — explains why flat tables don't scale.
**Understanding the Linux Virtual Memory Manager**
- **Book**: Mel Gorman, *Understanding the Linux Virtual Memory Manager* (2004, free online)
- **Best Explanation**: Chapter 3 on Page Table Management
- **Why**: Linux uses 4-level page tables on x86-64. Shows how real systems handle sparse address spaces.
- **When**: Before Milestone 3 — read while designing your two-level structure.
### 🔍 Deep Dive: Sparse Address Spaces
**Page Table Compression**
- **Paper**: Michael H. Butler, "Page Table Structure and the Cost of Memory References" (1993)
- **Best Explanation**: Analysis of inverted vs. hierarchical page tables
- **Why**: Quantifies the memory/speed tradeoffs that led to multi-level tables winning on modern systems.
- **When**: After Milestone 3 — understand why multi-level became the standard.
---
## Milestone 4: Page Replacement and Swap
### 📖 Read Before Milestone 4
**The Working Set Model for Program Behavior**
- **Paper**: Peter J. Denning, "The Working Set Model for Program Behavior" (1968), CACM
- **Best Explanation**: The original paper — all of it
- **Why**: Foundational work defining "working set" and "thrashing." Every replacement algorithm since builds on this insight.
- **When**: Before Milestone 4 — required reading for understanding replacement.
**Bélády's Anomaly**
- **Paper**: László A. Bélády, "A Study of Replacement Algorithms for Virtual-Storage Computer Systems" (1966), IBM Systems Journal
- **Best Explanation**: Section 3 demonstrating the anomaly with the famous 1-2-3-4-1-2-5 pattern
- **Why**: The original discovery that more memory can mean MORE page faults with FIFO. Your simulator will reproduce this result.
- **When**: Before Milestone 4 — understand the anomaly before implementing it.
### 🔍 Deep Dive: Replacement Algorithms
**The Clock Algorithm**
- **Paper**: F. J. Corbató, "A Paging Experiment with the Multics System" (1969)
- **Best Explanation**: Description of the "clock hand" replacement policy
- **Why**: The original Clock algorithm paper. The referenced-bit sweep you implement comes from this work.
- **When**: After implementing FIFO and LRU in Milestone 4 — see the historical context.
**LRU Implementation and Approximation**
- **Paper**: Elizabeth J. O'Neil, et al., "The LRU-K Page Replacement Algorithm for Database Disk Buffering" (1993), SIGMOD
- **Best Explanation**: Section 2 on why true LRU is impractical and how to approximate it
- **Why**: Shows LRU approximations used in real systems. Your timestamp-based LRU is a simplified version.
- **When**: After Milestone 4 — understand production-grade replacements.
### 💻 Reference Implementation
**Linux Page Replacement (Clock-Pro)**
- **Code**: Linux kernel source, `mm/vmscan.c` — functions `shrink_active_list()` and `shrink_inactive_list()`
- **Why**: Linux's two-handed clock implementation. Shows how production systems balance recency and frequency.
- **When**: After Milestone 4 — compare your algorithms to the real thing.
---
## Cross-Cutting References
### 📖 Read Anytime
**Computer Systems: A Programmer's Perspective**
- **Book**: Randal E. Bryant and David R. O'Hallaron, *Computer Systems: A Programmer's Perspective* (3rd ed.)
- **Best Explanation**: Chapter 9 on Virtual Memory
- **Why**: Comprehensive treatment of the entire memory hierarchy from caches through virtual memory. Excellent exercises.
- **When**: Read alongside any milestone — especially valuable for Milestones 1 and 4.
**Designing Data-Intensive Applications**
- **Book**: Martin Kleppmann, *Designing Data-Intensive Applications*
- **Best Explanation**: Chapter 3 on Storage and Retrieval (B-trees vs. LSM trees)
- **Why**: Multi-level page tables are essentially B-trees optimized for bit-prefix keys. This chapter provides the broader context.
- **When**: After Milestone 3 — see the connection between page tables and database indexing.
### 🔍 Deep Dive: Real-World Impact
**KPTI (Kernel Page Table Isolation) and Meltdown**
- **Paper**: Moritz Lipp, et al., "Meltdown: Reading Kernel Memory from User Space" (2018)
- **Best Explanation**: Section 4 on the KPTI mitigation
- **Why**: Shows the real-world cost of TLB flushes — KPTI added 5-10% overhead to syscall-heavy workloads by forcing TLB invalidation on every kernel entry.
- **When**: After Milestone 2 — understand why TLB performance matters in security contexts.
---
## Summary Reading Order
| Milestone | Before Starting | During Implementation | After Completion |
|-----------|-----------------|----------------------|------------------|
| 1 | OSTEP Ch. 18-20, Drepper §3 | Intel SDM Vol. 3 Ch. 4 | Linux `pagetable.c` |
| 2 | Mogul & Borg (1991) | OSTEP Ch. 21 (TLBs) | QEMU `cputlb.c`, Cox & Zwaenepoel |
| 3 | OSTEP Ch. 20 (multi-level), Gorman Ch. 3 | Cheriton & Zwaenepoel (1985) | Linux 4-level tables |
| 4 | Denning (1968), Bélády (1966) | Corbató (1969), O'Neil et al. (1993) | Linux `vmscan.c` |

---

# Virtual Memory Simulator

Build a complete virtual memory subsystem simulator that mirrors how real operating systems manage memory. You'll implement the full address translation pipeline—from the CPU's perspective of virtual addresses through multi-level page tables, TLB caching, and swap-backed page replacement. This isn't a toy: trace-driven testing reveals the same performance phenomena (TLB miss storms, page fault cascades, Bélády's anomaly) that production systems encounter. The simulator processes memory access traces (read/write operations with virtual addresses) and produces detailed statistics on translation efficiency, fault rates, and replacement algorithm behavior.



<!-- MS_ID: virtual-memory-sim-m1 -->
# Milestone 1: Single-Level Page Table and Address Translation
## The Lie Your Pointers Tell You
When you write `int *ptr = 0x1000;` in your C program, you believe you're pointing at memory address 4096. You assume that when you dereference `*ptr`, the CPU walks over to physical location 0x1000 in RAM and grabs the value.
**This is a lie.**
Every pointer you've ever used is a *virtual address*—a fictional number that your process invented. The CPU takes that fictional number and, on every single memory access, translates it to a completely different *physical address* where the data actually lives. Your `0x1000` might translate to physical `0x7F3A2000`. Another process's `0x1000` might translate to `0x8B1C4000`. Same virtual number, completely different physical locations.

![Statistics Collection Points](./diagrams/tdd-diag-m1-10.svg)

![Virtual Memory Simulator: System Overview](./diagrams/diag-satellite-system.svg)

This translation isn't a performance optimization you can opt out of. It's the fundamental mechanism that makes modern computing possible:
- **Isolation**: Your process cannot corrupt another process's memory because your virtual addresses don't map to their physical frames
- **Relocation**: Programs can load anywhere in physical memory without recompilation—the page table handles the indirection
- **Sharing**: The same physical frame can appear at different virtual addresses in different processes (shared libraries, memory-mapped files)
- **Lazy allocation**: The OS can promise memory (virtual addresses exist) without providing it (physical frames allocated on first touch)

> **🔑 Foundation: The three fundamental reasons virtual-to-physical translatio**
>
> Virtual-to-physical address translation allows multiple programs to believe they have exclusive access to the entire physical memory space. This illusion is vital for operating systems to isolate processes, preventing them from corrupting each other's data or the kernel. We need address translation in our OS project right now because we want to run multiple user-space programs concurrently without them interfering with each other or crashing the system. The key insight is that address translation decouples a process's view of memory (its virtual address space) from the actual physical layout, enabling resource management and security.



![VPN to PFN Translation Example](./diagrams/tdd-diag-m1-11.svg)

> **🔑 Foundation: How the OS creates the illusion that each process has its own private memory space**
> 
> ## What It Is
An **address space** is the OS's way of giving each process the illusion that it has its own private, contiguous memory — starting at address 0 and extending to some maximum address. The process thinks it has the entire machine's memory to itself.
In reality, physical RAM is a shared resource. Multiple processes run simultaneously, each believing it owns memory starting at address 0. The OS and hardware (specifically, the Memory Management Unit or MMU) collaborate to translate these "virtual" addresses into actual physical locations — mapping process A's address 0x1000 to physical address 0x5000, while process B's address 0x1000 maps to physical address 0xA000.
The address space is typically organized into segments:
- **Code (text)**: The program's instructions, usually read-only
- **Heap**: Dynamically allocated data, grows upward
- **Stack**: Local variables and call frames, grows downward
- **Shared libraries**: Mapped in at specific ranges
## Why You Need It Right Now
Understanding address spaces is foundational to almost everything else in systems programming:
- **Debugging**: When you see a segmentation fault at address 0x0, you're seeing a virtual address — knowing this helps you interpret crashes correctly
- **Pointer safety**: Understanding that pointers are virtual addresses explains why dangling pointers can corrupt your own data but not other processes
- **Memory-mapped I/O**: When you `mmap()` a file, you're extending your address space to include file contents
- **Inter-process communication**: Shared memory works by mapping the *same* physical pages into different processes' address spaces
- **Security**: Buffer overflows and return-oriented programming attacks all operate within the address space abstraction
## Key Mental Model
Think of an address space as a **per-process map or translation table**, not as actual memory. Each process carries its own map. When the CPU switches from one process to another (context switch), the OS swaps in a different map — suddenly all those address 0x1000 references point somewhere entirely different.
The genius of this abstraction is that the process never needs to know or care where its memory physically resides. It simply refers to addresses 0 through N, and the OS handles the complexity of multiplexing physical RAM among all running processes. This is isolation by design: process A cannot name (and therefore cannot access) process B's memory, because those addresses *mean different things* in each process's context.

![Page Fault Handling Sequence](./diagrams/tdd-diag-m1-06.svg)

In this milestone, you'll build the translation machinery yourself. You'll parse memory access traces, decompose virtual addresses, consult a page table, handle faults, and track statistics. By the end, you won't just understand address translation—you'll have implemented it.
---
## The Physical Constraint: Memory Doesn't Come in Bytes
Memory chips don't address individual bytes efficiently. A memory controller that tracked every byte would need massive addressing overhead. Instead, physical memory is organized into **frames**—contiguous chunks that the hardware can manage efficiently.
The standard frame size on modern systems is 4 KB (4096 bytes). This isn't arbitrary:
- **Small enough**: Internal fragmentation (wasted space when allocations don't fill a frame) is bounded at 4 KB
- **Large enough**: Amortizes the overhead of tracking metadata across many bytes
- **Hardware-aligned**: Matches disk sector sizes, cache line boundaries, and TLB designs

> **🔑 Foundation: The distinction between pages**
>
> A "page" is a fixed-size block of virtual memory, representing a contiguous range of addresses within a process's address space. A "frame," on the other hand, is a fixed-size block of physical memory where page data can be stored. We're implementing paging to allow for efficient memory allocation and protection, so understanding this distinction is crucial for mapping virtual addresses to physical locations. The mental model is that pages are the "logical" units of memory, while frames are the actual physical containers; the page table bridges the gap between them.


**The tension**: Programs allocate memory in arbitrary sizes (17 bytes, 3 MB, 1 byte). Hardware manages memory in 4 KB chunks. The page table is the translator that bridges this gap.
Your virtual address space is divided into **pages** (4 KB chunks of virtual memory). Physical memory is divided into **frames** (4 KB chunks of physical memory). The page table maps each page to a frame—or marks it as unmapped.
```
Virtual Address Space          Physical Memory
┌─────────────────┐           ┌─────────────────┐
│ Page 0: 0x0000  │ ─────────▶│ Frame 42        │
├─────────────────┤           ├─────────────────┤
│ Page 1: 0x1000  │ ─────────▶│ Frame 7         │
├─────────────────┤           ├─────────────────┤
│ Page 2: 0x2000  │     ?     │ Frame 103       │
├─────────────────┤           ├─────────────────┤
│ Page 3: 0x3000  │ ─────────▶│ Frame 255       │
└─────────────────┘           └─────────────────┘
```
---
## Anatomy of a Virtual Address
A 32-bit virtual address isn't a monolithic number—it's two fields packed together:
```
┌────────────────────────┬────────────────────┐
│   Virtual Page Number  │      Offset        │
│        (VPN)           │                    │
│      20 bits           │     12 bits        │
└────────────────────────┴────────────────────┘
         31              11 10               0
```

![Virtual Address Bit Layout](./diagrams/tdd-diag-m1-02.svg)

![Virtual Address Bit Layout](./diagrams/diag-address-decomposition.svg)

![Dirty and Referenced Bit State Machine](./diagrams/tdd-diag-m1-12.svg)

- **VPN (bits 31-12)**: Which page contains the data? This indexes into the page table.
- **Offset (bits 11-0)**: Which byte within the 4 KB page? This passes through unchanged.
**Why 12 bits for offset?** Because 2¹² = 4096 = 4 KB. The offset can address every byte within a page.
**Why 20 bits for VPN?** Because 32 - 12 = 20. This means 2²⁰ = 1,048,576 possible pages in a 32-bit address space.
To translate a virtual address:
1. **Extract VPN**: `vpn = virtual_address >> 12` (shift right by 12 bits)
2. **Extract offset**: `offset = virtual_address & 0xFFF` (mask bottom 12 bits)
3. **Look up PFN**: `pfn = page_table[vpn].frame_number`
4. **Compose physical address**: `physical_address = (pfn << 12) | offset`
The offset survives translation unchanged because both pages and frames are 4 KB—the byte position within the chunk is the same whether you're talking virtual or physical.
```c
// The fundamental translation formula
#define PAGE_SIZE 4096
#define PAGE_SHIFT 12
#define PAGE_MASK 0xFFF
uint32_t translate(uint32_t va, page_table_entry_t *page_table) {
    uint32_t vpn = va >> PAGE_SHIFT;        // Which page?
    uint32_t offset = va & PAGE_MASK;       // Where in the page?
    uint32_t pfn = page_table[vpn].pfn;     // Where in physical memory?
    return (pfn << PAGE_SHIFT) | offset;    // Combine frame + offset
}
```
---
## The Page Table Entry: Metadata That Controls Everything
A page table entry (PTE) is more than just a frame number. It's a control structure that encodes permissions, state, and access history.

![Translation Pipeline Flow](./diagrams/tdd-diag-m1-05.svg)

![Page Table Entry (PTE) Bit Fields](./diagrams/diag-pte-structure.svg)

Here's the structure you'll implement:
```c
typedef struct {
    uint32_t pfn;        // Physical Frame Number (which frame backs this page)
    bool valid;          // Is this page present in physical memory?
    bool readable;       // Can the process read from this page?
    bool writable;       // Can the process write to this page?
    bool dirty;          // Has this page been written to?
    bool referenced;     // Has this page been accessed (read or write)?
} page_table_entry_t;
```
Let's examine each field:
### Valid Bit
- **Meaning**: Does this page have a physical frame allocated?
- **If 0**: Accessing this page triggers a **page fault**—the OS must intervene
- **If 1**: Translation can proceed using the PFN
This is the mechanism behind **demand paging**: the OS sets up virtual addresses (allocates VPNs in the page table) but doesn't allocate physical frames until the process actually touches the memory. The valid bit is 0 initially; on first access, the page fault handler allocates a frame and sets valid=1.
### Permission Bits (Readable, Writable)
- **Meaning**: What operations is the process allowed to perform?
- **Violation**: Accessing with wrong permissions triggers a **protection fault** (segfault in user space)
These bits enable memory protection: read-only data (code, constants), read-write data (heap, stack), and even execute-only memory for security.
### Dirty Bit
- **Set when**: A write operation modifies the page
- **Never cleared by hardware**: Only the OS resets this after handling the page
This bit is critical for swap: when evicting a page from memory, the OS only writes it to disk if it's dirty. Clean pages can be discarded and reloaded from their original source.
### Referenced (Accessed) Bit
- **Set when**: Any access (read or write) to the page
- **Used by**: Page replacement algorithms to identify "hot" pages
This is the hardware's contribution to LRU-like algorithms. Pages with referenced=1 have been used recently; pages with referenced=0 are candidates for eviction.
---
## The Translation Pipeline: From Virtual to Physical

![Page Table Entry Structure](./diagrams/tdd-diag-m1-03.svg)

![Single-Level Address Translation Flow](./diagrams/diag-translation-pipeline.svg)

Every memory access follows this decision tree:
```
1. Extract VPN and offset from virtual address
2. Look up PTE: pte = page_table[vpn]
3. Check valid bit:
   - If 0: PAGE FAULT → OS handler
   - If 1: Continue
4. Check permissions:
   - Read access but !readable? PROTECTION FAULT → OS handler
   - Write access but !writable? PROTECTION FAULT → OS handler
   - Otherwise: Continue
5. Update access bits:
   - If write: pte.dirty = 1
   - Always: pte.referenced = 1
6. Compose physical address: pa = (pte.pfn << 12) | offset
7. Return physical address
```
Notice the order: **validity before permissions**. A page can't violate permissions if it doesn't exist. This matters because:
- Page fault handlers might change permissions (e.g., copy-on-write)
- The distinction affects what statistics you track
Here's the complete translation function with all checks:
```c
typedef enum {
    TRANSLATE_OK,           // Success: physical address returned
    TRANSLATE_PAGE_FAULT,   // Valid=0: page not in memory
    TRANSLATE_PROTECTION    // Permission violation
} translate_result_t;
typedef struct {
    translate_result_t result;
    uint32_t physical_address;
} translate_out_t;
translate_out_t translate_address(
    uint32_t va,
    bool is_write,
    page_table_entry_t *page_table,
    stats_t *stats
) {
    translate_out_t out;
    uint32_t vpn = va >> PAGE_SHIFT;
    uint32_t offset = va & PAGE_MASK;
    stats->total_accesses++;
    // Step 1: Check valid bit
    if (!page_table[vpn].valid) {
        stats->page_faults++;
        out.result = TRANSLATE_PAGE_FAULT;
        out.physical_address = 0;
        return out;
    }
    // Step 2: Check permissions
    if (is_write && !page_table[vpn].writable) {
        stats->protection_faults++;
        out.result = TRANSLATE_PROTECTION;
        out.physical_address = 0;
        return out;
    }
    // Step 3: Update access bits
    page_table[vpn].referenced = true;
    if (is_write) {
        page_table[vpn].dirty = true;
    }
    // Step 4: Compose physical address
    out.physical_address = (page_table[vpn].pfn << PAGE_SHIFT) | offset;
    out.result = TRANSLATE_OK;
    return out;
}
```
---
## The Trace File Format: Reproducible Test Input
You'll need a way to feed memory accesses into your simulator. The standard format is a text file where each line specifies one access:
```
R 0x00001000
W 0x00001004
R 0x00002000
W 0x00003000
R 0x00001000
```

![System Component Overview](./diagrams/tdd-diag-m1-01.svg)

![Memory Access Trace File Format](./diagrams/diag-trace-format.svg)

- **First character**: Operation type (`R` for read, `W` for write)
- **Space**: Separator
- **Hexadecimal address**: The virtual address being accessed
This format is:
- **Human-readable**: You can manually construct test cases
- **Reproducible**: Same trace file produces same results (deterministic testing)
- **Flexible**: Easy to generate programmatically for stress tests
```c
typedef struct {
    bool is_write;
    uint32_t virtual_address;
} memory_access_t;
bool parse_trace_line(const char *line, memory_access_t *access) {
    char op;
    unsigned int addr;
    if (sscanf(line, "%c 0x%X", &op, &addr) != 2) {
        return false;  // Parse error
    }
    access->is_write = (op == 'W' || op == 'w');
    access->virtual_address = addr;
    return true;
}
```
For your first milestone, start with simple hand-crafted traces:
```
# Simple trace: sequential access to 3 pages
R 0x00001000    # Access page 1 (VPN=1), will fault
W 0x00001400    # Write to same page, sets dirty bit
R 0x00002000    # Access page 2 (VPN=2), will fault
R 0x00003000    # Access page 3 (VPN=3), will fault
R 0x00001000    # Re-access page 1, should NOT fault (valid=1 now)
```
---
## Demand Paging: Being Lazy on Purpose

![Trace File Format](./diagrams/tdd-diag-m1-08.svg)

![Demand Paging: First Access Sequence](./diagrams/diag-demand-paging-timeline.svg)

![Demand Paging Timeline](./diagrams/tdd-diag-m1-09.svg)

When a process allocates memory (e.g., `malloc(1GB)`), the OS doesn't immediately reserve 1 GB of physical RAM. That would be wasteful—most allocations are never fully used. Instead:
1. The OS creates page table entries with **valid=0**
2. No physical frames are allocated yet
3. On first access to each page, a **page fault** occurs
4. The page fault handler allocates a frame, sets the PFN, and sets **valid=1**
5. The faulting instruction is restarted
This is **demand paging**: physical resources are allocated only when demanded by actual access.
**Why this matters**: Programs routinely allocate more memory than they use. A web browser might reserve address space for a 2 GB cache but only touch 200 MB. Without demand paging, the OS would need to reserve 2 GB of physical RAM that would never be used.
Your simulator will implement this:
```c
typedef struct {
    uint32_t *frames;      // Array of physical frames
    uint32_t total_frames;
    uint32_t next_free;    // Simple bump allocator
} physical_memory_t;
void handle_page_fault(
    uint32_t vpn,
    page_table_entry_t *page_table,
    physical_memory_t *phys_mem
) {
    // Check if we have free frames
    if (phys_mem->next_free >= phys_mem->total_frames) {
        // No free frames! This is where page replacement comes in
        // (Milestone 4). For now, we'll just error out.
        fprintf(stderr, "Out of physical memory!\n");
        exit(1);
    }
    // Allocate a free frame
    uint32_t pfn = phys_mem->next_free++;
    // Update the page table
    page_table[vpn].pfn = pfn;
    page_table[vpn].valid = true;
    page_table[vpn].readable = true;
    page_table[vpn].writable = true;  // Simplified: all pages are RW
    page_table[vpn].dirty = false;
    page_table[vpn].referenced = false;
    printf("Page fault: VPN %u -> PFN %u\n", vpn, pfn);
}
```
---
## Protection Faults: When Access Is Denied

![Physical Memory Frame Pool](./diagrams/tdd-diag-m1-04.svg)

![Protection Fault Detection Flowchart](./diagrams/diag-protection-fault-flow.svg)

![Protection Fault Detection](./diagrams/tdd-diag-m1-07.svg)

A protection fault is different from a page fault:
- **Page fault**: "This page isn't in memory yet" → Can be resolved by loading the page
- **Protection fault**: "You're not allowed to do this" → Cannot be resolved; it's a policy violation
Protection faults are what trigger `SIGSEGV` (segmentation fault) in real programs. Common causes:
- Writing to read-only memory (code section, string literals)
- Reading from kernel memory from user space
- Executing code from a non-executable page (DEP/NX bit)
Your simulator should distinguish these clearly:
```c
void log_protection_fault(uint32_t va, bool is_write, page_table_entry_t *pte) {
    printf("PROTECTION FAULT at VA 0x%08X\n", va);
    printf("  Attempted operation: %s\n", is_write ? "WRITE" : "READ");
    printf("  Page permissions: R=%d W=%d\n", pte->readable, pte->writable);
    printf("  This is a fatal error - the access cannot proceed.\n");
}
```
For testing, create a trace that triggers a protection fault:
```
# Assume VPN 1 is mapped read-only
W 0x00001000    # Write to read-only page → PROTECTION FAULT
```
---
## Hardware Soul: What Happens on Every Memory Access
Let's trace a single memory access through the hardware layers to understand the real cost.
### Level 1: Application (Software)
```c
int value = *ptr;  // ptr = 0x00001234
```
The compiler emits a load instruction. The CPU sees virtual address 0x00001234.
### Level 2: Memory Management Unit (Hardware)
The MMU receives the virtual address and:
1. Extracts VPN = 0x1, offset = 0x234
2. Checks the TLB (Translation Lookaside Buffer) — a hardware cache of recent translations
3. **TLB miss**: Walks the page table in memory
4. Reads PTE from memory (this is a memory access!)
5. Checks valid bit, permission bits
6. Updates dirty/referenced bits (another memory write if these changed!)
7. Caches the translation in the TLB
8. Returns physical address
### Level 3: Cache Hierarchy (Hardware)
With the physical address, the CPU:
1. Checks L1 data cache (4-8 cycles)
2. **L1 miss**: Check L2 cache (10-20 cycles)
3. **L2 miss**: Check L3 cache (30-50 cycles)
4. **L3 miss**: Go to main memory (100-200 cycles)
**The hidden cost of a page fault**: If the page isn't in memory (valid=0), the OS takes over:
- Context switch to kernel (save registers, change privilege level)
- Disk I/O to load the page (10,000,000+ cycles — milliseconds)
- Update page tables
- Context switch back to user process
- Retry the instruction
A page fault is **100,000× slower** than a cache hit. This is why working set size matters: if your active data exceeds physical memory, the system thrashes, spending all its time waiting for disk.
---
## Implementation: The Complete Single-Level Simulator
Let's put it all together into a working simulator. This implementation will:
1. Parse a trace file
2. Maintain a flat page table
3. Handle translations with fault detection
4. Implement demand paging
5. Track and report statistics
### Data Structures
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
// Configuration constants
#define PAGE_SIZE 4096
#define PAGE_SHIFT 12
#define PAGE_MASK 0xFFF
#define MAX_VPN 1048576  // 2^20 for 32-bit address space
// Page Table Entry
typedef struct {
    uint32_t pfn;        // Physical Frame Number
    bool valid;          // Page present in memory
    bool readable;       // Read permission
    bool writable;       // Write permission
    bool dirty;          // Modified by write
    bool referenced;     // Accessed (read or write)
} pte_t;
// Physical Memory
typedef struct {
    uint8_t **frames;    // Array of frame pointers (NULL = free)
    uint32_t total_frames;
    uint32_t *free_list; // Stack of free frame indices
    uint32_t free_count;
} physical_memory_t;
// Statistics
typedef struct {
    uint64_t total_accesses;
    uint64_t page_faults;
    uint64_t protection_faults;
    uint64_t tlb_hits;      // For Milestone 2
    uint64_t tlb_misses;    // For Milestone 2
} stats_t;
// Simulator State
typedef struct {
    pte_t *page_table;
    physical_memory_t phys_mem;
    stats_t stats;
} simulator_t;
```
### Initialization
```c
simulator_t *simulator_create(uint32_t num_frames) {
    simulator_t *sim = calloc(1, sizeof(simulator_t));
    if (!sim) return NULL;
    // Allocate flat page table (sparse in practice, but simple)
    sim->page_table = calloc(MAX_VPN, sizeof(pte_t));
    if (!sim->page_table) {
        free(sim);
        return NULL;
    }
    // Initialize all PTEs as invalid
    for (uint32_t i = 0; i < MAX_VPN; i++) {
        sim->page_table[i].valid = false;
    }
    // Allocate physical memory tracking
    sim->phys_mem.total_frames = num_frames;
    sim->phys_mem.frames = calloc(num_frames, sizeof(uint8_t*));
    sim->phys_mem.free_list = calloc(num_frames, sizeof(uint32_t));
    sim->phys_mem.free_count = num_frames;
    // Initialize free list (all frames are free initially)
    for (uint32_t i = 0; i < num_frames; i++) {
        sim->phys_mem.frames[i] = NULL;
        sim->phys_mem.free_list[i] = i;
    }
    return sim;
}
void simulator_destroy(simulator_t *sim) {
    if (!sim) return;
    // Free allocated frames
    for (uint32_t i = 0; i < sim->phys_mem.total_frames; i++) {
        free(sim->phys_mem.frames[i]);
    }
    free(sim->page_table);
    free(sim->phys_mem.frames);
    free(sim->phys_mem.free_list);
    free(sim);
}
```
### Frame Allocation
```c
// Allocate a free frame; returns -1 if none available
int32_t allocate_frame(simulator_t *sim) {
    if (sim->phys_mem.free_count == 0) {
        return -1;  // Out of memory
    }
    // Pop from free list
    uint32_t pfn = sim->phys_mem.free_list[--sim->phys_mem.free_count];
    // Allocate the actual frame memory
    sim->phys_mem.frames[pfn] = calloc(PAGE_SIZE, sizeof(uint8_t));
    return (int32_t)pfn;
}
// Free a frame (for future use with page replacement)
void free_frame(simulator_t *sim, uint32_t pfn) {
    if (sim->phys_mem.frames[pfn]) {
        free(sim->phys_mem.frames[pfn]);
        sim->phys_mem.frames[pfn] = NULL;
    }
    sim->phys_mem.free_list[sim->phys_mem.free_count++] = pfn;
}
```
### Translation with Fault Handling
```c
typedef enum {
    TRANS_OK,
    TRANS_PAGE_FAULT,
    TRANS_PROTECTION_FAULT
} trans_result_t;
typedef struct {
    trans_result_t result;
    uint32_t physical_address;
} trans_output_t;
trans_output_t translate(
    simulator_t *sim,
    uint32_t va,
    bool is_write
) {
    trans_output_t out = {0};
    uint32_t vpn = va >> PAGE_SHIFT;
    uint32_t offset = va & PAGE_MASK;
    sim->stats.total_accesses++;
    // Bounds check
    if (vpn >= MAX_VPN) {
        sim->stats.protection_faults++;
        out.result = TRANS_PROTECTION_FAULT;
        return out;
    }
    pte_t *pte = &sim->page_table[vpn];
    // Check valid bit (page fault if not present)
    if (!pte->valid) {
        sim->stats.page_faults++;
        // Demand paging: allocate a frame on fault
        int32_t pfn = allocate_frame(sim);
        if (pfn < 0) {
            // No free frames - in Milestone 4, we'd evict here
            fprintf(stderr, "OUT OF MEMORY: No free frames for VPN %u\n", vpn);
            out.result = TRANS_PAGE_FAULT;
            return out;
        }
        // Set up the PTE
        pte->pfn = (uint32_t)pfn;
        pte->valid = true;
        pte->readable = true;
        pte->writable = true;  // Simplified: all pages RW
        pte->dirty = false;
        pte->referenced = false;
        printf("[Page Fault] VPN %u -> PFN %u\n", vpn, pfn);
    }
    // Re-fetch PTE (might have been updated by page fault handler)
    pte = &sim->page_table[vpn];
    // Check permissions
    if (is_write && !pte->writable) {
        sim->stats.protection_faults++;
        out.result = TRANS_PROTECTION_FAULT;
        printf("[Protection Fault] Write to read-only page VPN %u\n", vpn);
        return out;
    }
    if (!is_write && !pte->readable) {
        sim->stats.protection_faults++;
        out.result = TRANS_PROTECTION_FAULT;
        printf("[Protection Fault] Read from non-readable page VPN %u\n", vpn);
        return out;
    }
    // Update access bits
    pte->referenced = true;
    if (is_write) {
        pte->dirty = true;
    }
    // Compose physical address
    out.physical_address = (pte->pfn << PAGE_SHIFT) | offset;
    out.result = TRANS_OK;
    return out;
}
```
### Trace Processing
```c
bool process_trace_file(simulator_t *sim, const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Failed to open trace file");
        return false;
    }
    char line[256];
    int line_num = 0;
    while (fgets(line, sizeof(line), fp)) {
        line_num++;
        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\0') {
            continue;
        }
        // Parse: R/W 0xADDRESS
        char op;
        unsigned int addr;
        if (sscanf(line, "%c 0x%X", &op, &addr) != 2) {
            fprintf(stderr, "Parse error at line %d: %s", line_num, line);
            continue;
        }
        bool is_write = (op == 'W' || op == 'w');
        printf("[Line %d] %c 0x%08X\n", line_num, is_write ? 'W' : 'R', addr);
        trans_output_t result = translate(sim, addr, is_write);
        if (result.result == TRANS_OK) {
            printf("  -> Physical: 0x%08X\n", result.physical_address);
        } else if (result.result == TRANS_PAGE_FAULT) {
            printf("  -> PAGE FAULT (handled)\n");
        } else {
            printf("  -> PROTECTION FAULT (fatal)\n");
            // In a real simulator, we might continue or abort
        }
        printf("\n");
    }
    fclose(fp);
    return true;
}
```
### Statistics Reporting
```c
void print_statistics(simulator_t *sim) {
    printf("\n");
    printf("========================================\n");
    printf("         SIMULATION STATISTICS          \n");
    printf("========================================\n");
    printf("Total memory accesses:  %lu\n", sim->stats.total_accesses);
    printf("Page faults:            %lu\n", sim->stats.page_faults);
    printf("Protection faults:      %lu\n", sim->stats.protection_faults);
    if (sim->stats.total_accesses > 0) {
        double fault_rate = (double)sim->stats.page_faults / sim->stats.total_accesses * 100.0;
        printf("Page fault rate:        %.2f%%\n", fault_rate);
    }
    printf("Physical frames used:   %u / %u\n",
           sim->phys_mem.total_frames - sim->phys_mem.free_count,
           sim->phys_mem.total_frames);
    printf("========================================\n");
}
```
### Main Entry Point
```c
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <trace_file> <num_frames>\n", argv[0]);
        printf("Example: %s traces/simple.trace 64\n", argv[0]);
        return 1;
    }
    const char *trace_file = argv[1];
    uint32_t num_frames = atoi(argv[2]);
    if (num_frames == 0) {
        fprintf(stderr, "Invalid number of frames\n");
        return 1;
    }
    printf("Virtual Memory Simulator\n");
    printf("Page size: %d bytes\n", PAGE_SIZE);
    printf("Physical frames: %u\n", num_frames);
    printf("Trace file: %s\n\n", trace_file);
    simulator_t *sim = simulator_create(num_frames);
    if (!sim) {
        fprintf(stderr, "Failed to create simulator\n");
        return 1;
    }
    bool success = process_trace_file(sim, trace_file);
    if (success) {
        print_statistics(sim);
    }
    simulator_destroy(sim);
    return success ? 0 : 1;
}
```
---
## Testing Your Simulator
### Test Case 1: Basic Demand Paging
Create `traces/basic.trace`:
```
# Basic demand paging test
# First access to each page triggers a fault
R 0x00001000
R 0x00002000
R 0x00003000
# Second access to same pages - no faults
R 0x00001000
R 0x00002000
R 0x00003000
```
**Expected output**:
- 3 page faults (first three accesses)
- 6 total accesses
- 50% page fault rate
### Test Case 2: Dirty Bit Tracking
Create `traces/dirty.trace`:
```
# Dirty bit test
R 0x00001000    # Fault, not dirty
W 0x00001000    # No fault, sets dirty bit
R 0x00001000    # No fault, dirty stays set
W 0x00002000    # Fault, immediately sets dirty
```
**Verify**: After running, check that VPN 1 and VPN 2 have dirty=true.
### Test Case 3: Protection Faults
To test protection faults, you'll need to modify your simulator to support read-only pages. Add a function:
```c
void set_page_read_only(simulator_t *sim, uint32_t vpn) {
    if (vpn < MAX_VPN && sim->page_table[vpn].valid) {
        sim->page_table[vpn].writable = false;
    }
}
```
Then create a trace that writes to a read-only page:
```
# Protection fault test
R 0x00001000    # Fault, allocate page (RW by default)
# [Manually set page to read-only here]
W 0x00001000    # Protection fault!
```
### Test Case 4: Address Decomposition
Verify your bit extraction is correct:
```c
void test_address_decomposition(void) {
    // Test case: 0x00001234
    // VPN should be 0x1, offset should be 0x234
    assert((0x00001234 >> PAGE_SHIFT) == 0x1);
    assert((0x00001234 & PAGE_MASK) == 0x234);
    // Test case: 0xFFFFF000 (last page, last byte)
    // VPN should be 0xFFFFF, offset should be 0x000
    assert((0xFFFFF000 >> PAGE_SHIFT) == 0xFFFFF);
    assert((0xFFFFF000 & PAGE_MASK) == 0x000);
    // Test case: 0x00000FFF
    // VPN should be 0x0, offset should be 0xFFF
    assert((0x00000FFF >> PAGE_SHIFT) == 0x0);
    assert((0x00000FFF & PAGE_MASK) == 0xFFF);
    printf("All address decomposition tests passed!\n");
}
```
---
## Common Pitfalls and How to Avoid Them
### Pitfall 1: Off-by-One in Bit Shifting
**The mistake**:
```c
uint32_t vpn = va >> 11;  // WRONG! Should be 12
```
**Why it's wrong**: 4 KB = 4096 = 2^12, so you need 12 bits for the offset, meaning you shift by 12.
**The fix**:
```c
#define PAGE_SHIFT 12  // Define once, use everywhere
uint32_t vpn = va >> PAGE_SHIFT;
```
### Pitfall 2: Destroying the Offset
**The mistake**:
```c
uint32_t physical = (pfn << 12);  // WRONG! Lost the offset
```
**Why it's wrong**: The physical address needs both the frame number AND the offset within the frame.
**The fix**:
```c
uint32_t physical = (pfn << PAGE_SHIFT) | offset;
```
### Pitfall 3: Setting Dirty on Read
**The mistake**:
```c
pte->dirty = true;  // Always set dirty
```
**Why it's wrong**: Dirty means "modified." Read operations don't modify.
**The fix**:
```c
if (is_write) {
    pte->dirty = true;
}
```
### Pitfall 4: Confusing Page Fault and Protection Fault
**The mistake**: Treating all faults the same way.
**The distinction**:
- **Page fault** (valid=0): The page exists but isn't in memory. The OS can fix this by loading the page.
- **Protection fault** (valid=1, wrong permissions): The process is violating memory policy. The OS kills the process.
**The fix**: Separate counters, separate handling paths, separate log messages.
---
## Design Decisions: Why This, Not That?
| Option | Pros | Cons | Used By |
|--------|------|------|---------|
| **Flat Page Table ✓** | O(1) lookup, simple implementation | 4 MB per process (wasteful for sparse address spaces) | Early Unix, embedded systems |
| Hashed Page Table | Memory-efficient for sparse spaces | O(n) worst case, collision handling | Early Linux (PAE) |
| Inverted Page Table | Single table for all processes, memory-efficient | O(n) search, complex sharing | PowerPC, IA-64 |
| Multi-Level Page Tables | Memory-efficient, hierarchical | Multiple memory accesses per translation | x86-64, ARM64 |
For this milestone, we use a flat page table because:
1. **Simplicity**: One array lookup, no pointer chasing
2. **Correctness first**: Get the translation logic right before optimizing
3. **Comparison baseline**: In Milestone 3, you'll implement multi-level tables and compare memory usage
---
## What You've Built
At this point, your simulator can:
1. **Parse memory access traces** from text files
2. **Decompose virtual addresses** into VPN and offset
3. **Perform address translation** via a flat page table
4. **Detect page faults** (valid=0) and allocate frames on demand
5. **Detect protection faults** (permission violations) and log them
6. **Track access bits** (dirty, referenced) for each page
7. **Collect statistics** on accesses, faults, and memory usage
This is the foundation of virtual memory. Every operating system—from the Linux kernel to the microcontroller in your washing machine—has code that does exactly what you just implemented.
---
## Knowledge Cascade: Where This Leads
### Same Domain Connections
**→ TLB Caching (Milestone 2)**: Your simulator now walks the page table on every access. Real CPUs cache translations in a TLB (Translation Lookaside Buffer). A TLB hit avoids the page table walk entirely—this is why TLB miss rates dominate performance.
**→ Multi-Level Page Tables (Milestone 3)**: A flat page table for a 64-bit address space would be impossibly large (2^52 entries × 8 bytes = 36 petabytes). Multi-level tables solve this by only allocating page table structures for regions that are actually used.
**→ Page Replacement (Milestone 4)**: When physical memory fills up, the OS must choose which page to evict. The referenced and dirty bits you're tracking are the key inputs to LRU, Clock, and other replacement algorithms.
### Cross-Domain Connections
**→ Cache Coherence Protocols**: CPU caches track physical addresses, not virtual ones. When one core modifies a physical frame, all caches with that frame must be invalidated. This is why cache coherence is a "physical address" problem—the same virtual address in different processes maps to different physical frames.
**→ Memory-Mapped Files**: The same page fault mechanism you implemented is how `mmap()` works. When you map a file into memory, accessing that memory triggers page faults, and the OS loads file data instead of zeroed memory. Your page fault handler is one `read()` syscall away from being a memory-mapped file system.
**→ Garbage Collector Write Barriers**: Generational garbage collectors need to track which heap pages contain modified objects. Sound familiar? The dirty bit mechanism is exactly what they use. A GC can protect the "old generation" and use protection faults to detect cross-generational pointers.
**→ Container Memory Limits (cgroups)**: Docker containers have memory limits enforced by the kernel. The kernel tracks how many physical frames a cgroup is using and can trigger OOM kills. This accounting happens at the frame allocation level—exactly where your `allocate_frame()` function sits.
**→ Security: ASLR**: Address Space Layout Randomization works because translation decouples virtual from physical. The stack can appear at any virtual address; the same physical stack frame could be at 0x7FFE0000 in one run and 0x2A3B0000 in another. The page table makes this indirection transparent to the process.
### Historical Context
**→ Segmentation vs. Paging**: Before paging became dominant, x86 used segmentation—base + limit addressing where each segment had a starting address and size. Segmentation had external fragmentation (holes in memory) and couldn't support sparse address spaces efficiently. Paging won because fixed-size frames eliminate external fragmentation and the page table provides a clean indirection layer.
**→ The 640 KB Limit**: Early PCs had a 640 KB "conventional memory" limit because the 8086 used 20-bit addresses (1 MB total), and IBM reserved the upper 384 KB for ROM and memory-mapped I/O. Expanded memory (EMS) and extended memory (XMS) were hacks to work around this. The 80386's 32-bit paging finally solved it properly—with exactly the mechanism you just implemented.
---
## Looking Ahead
In Milestone 2, you'll add a TLB (Translation Lookaside Buffer) between your address requests and the page table. This will reveal:
- Why TLB hit rates are the most important metric for memory-intensive workloads
- How ASID tagging allows TLB entries from multiple processes to coexist
- The subtle coherency problems when page table entries change but TLB entries become stale
The translation function you wrote today (`translate()`) will gain a new code path: check TLB first, fall back to page table on miss. Same result, dramatically different performance.
[[CRITERIA_JSON: {"milestone_id": "virtual-memory-sim-m1", "criteria": ["Trace file parser correctly reads memory access sequences in format 'R/W 0xADDRESS' with comment and whitespace handling", "Virtual address decomposition extracts VPN using right shift by PAGE_SHIFT (default 12) and offset using PAGE_MASK (0xFFF)", "Page table entry structure contains pfn (uint32_t), valid (bool), readable (bool), writable (bool), dirty (bool), and referenced (bool) fields", "Valid address translation returns physical address as concatenation of (pfn << PAGE_SHIFT) | offset", "Page fault is detected when valid=0 and triggers frame allocation with PTE update (demand paging)", "Protection fault is detected when permissions don't match operation (write to read-only page) and is logged separately from page faults", "Dirty bit is set only on write operations; referenced bit is set on any access (read or write)", "Statistics tracking includes total_accesses (uint64_t), page_faults (uint64_t), and protection_faults (uint64_t)", "Physical memory frame pool uses configurable frame count with free list or bump allocator for frame allocation", "Page fault handler allocates free frame, updates PTE with pfn, sets valid=1, and initializes permission bits", "Translation function returns result enum distinguishing TRANS_OK, TRANS_PAGE_FAULT, and TRANS_PROTECTION_FAULT", "Statistics reporter outputs total accesses, page faults, protection faults, and page fault rate percentage"]}]
<!-- END_MS -->


<!-- MS_ID: virtual-memory-sim-m2 -->
# Milestone 2: TLB (Translation Lookaside Buffer)
## The Hidden Cost You've Been Paying
In Milestone 1, you built a page table and translated virtual addresses. Every translation was an array lookup: `page_table[vpn]`. Simple, right? Just one memory access to get the frame number.
**This is the lie that hides the real cost.**
When you write `page_table[vpn]` in C, you're reading from *virtual* memory. That means the CPU must first translate `page_table[vpn]`'s virtual address to a physical address before it can read the PTE. And how does it do that? By walking the page table. Which requires another translation. Which requires...
Wait. Let's step back and count the actual memory accesses for a single load instruction.

![TLB Miss Slow Path](./diagrams/tdd-diag-m2-05.svg)

![TLB Hit vs Miss: Memory Access Count](./diagrams/diag-tlb-vs-page-table-access.svg)

![LRU Victim Selection Algorithm](./diagrams/tdd-diag-m2-06.svg)

### The True Cost of a Page Table Walk
When your program executes `int x = *ptr` where `ptr = 0x00001234`:
1. **First memory access**: Read the page table entry for VPN 0x1
   - But wait—the page table itself lives in memory at some virtual address
   - The CPU needs to translate the page table's address first
2. **Second memory access** (if multi-level): Read the page directory entry
   - Same problem—another virtual address to translate
3. **Third memory access**: Finally read the actual data at the translated physical address
For a two-level page table on a 32-bit system, a single memory access becomes **three memory accesses**. For a four-level page table on x86-64 (the real world), it becomes **five memory accesses**.
And here's the kicker: each of those memory accesses can miss in L1, L2, and L3 cache, forcing a trip to main memory at 100-200 cycles each. A single `mov` instruction could cost **500-1000 cycles** just for address translation.
**This is the fundamental tension**: software wants to treat memory as a flat array of bytes, but hardware needs hierarchical indirection to provide isolation and virtualization. Without caching, that indirection would make every memory access 3-5× slower.
The Translation Lookaside Buffer (TLB) is the hardware's answer to this tension.
---
## What the TLB Actually Is
A TLB is a small, fast cache that stores recent virtual-to-physical address translations. It sits between the CPU's address generation unit and the page table in memory.
```
Virtual Address
      │
      ▼
┌─────────────┐     HIT      ┌──────────────┐
│     TLB     │─────────────▶│ Physical Addr│
│  (16-64     │              └──────────────┘
│   entries)  │
└─────────────┘
      │ MISS
      ▼
┌─────────────┐
│ Page Table  │
│  (in RAM)   │
└─────────────┘
      │
      ▼
┌──────────────┐
│ Physical Addr│
└──────────────┘
```
**Key insight**: The TLB doesn't cache data. It caches *mappings*. A TLB entry says "virtual page 0x1 maps to physical frame 0x7F" — not "the byte at address 0x1234 contains the value 42."

> **🔑 Foundation: Why caches work at all: the principle of locality**
>
> A cache operates by storing copies of frequently accessed data from main memory in a faster, smaller storage space. A "cache hit" occurs when requested data is found in the cache, while a "cache miss" happens when the data must be retrieved from main memory, incurring a significant performance penalty. We need to understand cache behavior to optimize our memory access patterns and avoid performance bottlenecks in our OS. The principle of locality – that programs tend to access data and instructions that are near those they have recently accessed (temporal and spatial locality) – is what makes caching effective.


This distinction matters because:
- **Data caches** (L1/L2/L3) cache the *contents* of memory
- **The TLB** caches the *path* to memory
- A TLB miss doesn't mean the data isn't in RAM—it means the CPU doesn't know *where* in RAM it is
### The Numbers That Make TLBs Essential
| Operation | Latency (approximate) |
|-----------|----------------------|
| TLB hit | 1-2 cycles |
| TLB miss (page table in L1) | 5-10 cycles |
| TLB miss (page table in L2) | 15-25 cycles |
| TLB miss (page table in L3) | 40-60 cycles |
| TLB miss (page table in RAM) | 100-200 cycles |
| Page fault (disk) | 10,000,000+ cycles |
A well-tuned program with good locality can achieve **99%+ TLB hit rates**. At that point, the translation overhead becomes negligible—almost every access costs just 1-2 cycles for the TLB lookup.
But if your access pattern has poor locality (random jumps across many pages), TLB misses dominate. Your program spends more time finding memory than using it.
---
## TLB Entry Structure
Each TLB entry needs to store enough information to perform a translation and validate it:

![TLB Hit Fast Path](./diagrams/tdd-diag-m2-04.svg)

![TLB Entry Structure with ASID Tag](./diagrams/diag-tlb-structure.svg)

![Translation Pipeline with TLB](./diagrams/tdd-diag-m2-10.svg)

```c
typedef struct {
    uint32_t vpn;        // Virtual Page Number (tag)
    uint32_t pfn;        // Physical Frame Number (the translation)
    uint32_t asid;       // Address Space ID (which process owns this)
    bool valid;          // Is this entry meaningful?
    bool readable;       // Read permission
    bool writable;       // Write permission
    bool dirty;          // Has the page been written to?
    bool referenced;     // Has the page been accessed?
} tlb_entry_t;
```
Let's examine each field:
### VPN (Virtual Page Number)
This is the "key" we're looking up. When the CPU receives virtual address `0x00001234`, it extracts VPN `0x1` and searches the TLB for an entry where `entry.vpn == 0x1`.
### PFN (Physical Frame Number)
This is the "value" we want. If we find a matching VPN, the PFN tells us which physical frame contains the data.
### ASID (Address Space ID)
This tag identifies which process the entry belongs to. Without ASIDs, every context switch would require flushing the entire TLB—a huge performance penalty. With ASIDs, entries from multiple processes can coexist, and the CPU simply ignores entries with the wrong ASID.
### Valid Bit
Indicates whether this entry contains useful data. An invalid entry is treated as "not present"—the TLB continues searching or reports a miss.
### Permission Bits (Readable, Writable)
Cached copies of the page table's permission bits. The TLB checks permissions on every access, so it needs this information locally.
### Dirty and Referenced Bits
These are where things get tricky. The TLB must track whether the page has been written to (dirty) or accessed at all (referenced). But these bits also exist in the page table! Keeping them synchronized is a coherency problem we'll explore later.
---
## The TLB Lookup Algorithm

![TLB Lookup Decision Tree](./diagrams/tdd-diag-m2-03.svg)

![TLB Lookup and Miss Handling](./diagrams/diag-tlb-lookup-flow.svg)

![TLB Invalidation Flow](./diagrams/tdd-diag-m2-09.svg)

Every address translation now follows this sequence:
```
1. Extract VPN and offset from virtual address
2. Search TLB for entry where:
   - entry.vpn == our VPN
   - entry.asid == current ASID (or ASID disabled)
   - entry.valid == true
3. If found (TLB HIT):
   a. Check permissions (read/write)
   b. If violation: PROTECTION FAULT
   c. Update dirty/referenced bits in TLB entry
   d. Return PFN (skip page table entirely)
4. If not found (TLB MISS):
   a. Walk the page table (expensive!)
   b. Check valid bit, permissions in PTE
   c. Insert new translation into TLB
   d. May need to evict an existing entry
   e. Return PFN from page table
5. Compose physical address: (PFN << 12) | offset
```
The critical optimization: **on a TLB hit, we never touch the page table**. All the information we need is in the TLB entry.
Here's the implementation:
```c
typedef struct {
    tlb_entry_t *entries;
    uint32_t capacity;
    uint32_t current_asid;
    uint64_t hits;
    uint64_t misses;
    uint64_t flushes;
} tlb_t;
// Search the TLB for a translation
tlb_entry_t* tlb_lookup(tlb_t *tlb, uint32_t vpn) {
    for (uint32_t i = 0; i < tlb->capacity; i++) {
        tlb_entry_t *entry = &tlb->entries[i];
        if (entry->valid && 
            entry->vpn == vpn && 
            entry->asid == tlb->current_asid) {
            return entry;  // HIT
        }
    }
    return NULL;  // MISS
}
```
This is a **fully associative** search—we check every entry. Real TLBs use set-associative designs (e.g., 4-way or 8-way) to parallelize the search, but the principle is the same.
---
## TLB Miss Handling: The Expensive Path
When the TLB doesn't contain the translation, we must fall back to the page table. This is called a **TLB miss** or **TLB fault**.
```c
typedef enum {
    TLB_HIT,
    TLB_MISS,
    TLB_PROTECTION_FAULT
} tlb_result_t;
typedef struct {
    tlb_result_t result;
    uint32_t pfn;
    bool from_tlb;  // Did this come from TLB or page table?
} tlb_translate_out_t;
tlb_translate_out_t tlb_translate(
    tlb_t *tlb,
    uint32_t va,
    bool is_write,
    pte_t *page_table,
    stats_t *stats
) {
    tlb_translate_out_t out = {0};
    uint32_t vpn = va >> PAGE_SHIFT;
    uint32_t offset = va & PAGE_MASK;
    stats->total_accesses++;
    // Step 1: Check TLB first
    tlb_entry_t *tlb_entry = tlb_lookup(tlb, vpn);
    if (tlb_entry != NULL) {
        // TLB HIT - fast path!
        stats->tlb_hits++;
        // Check permissions
        if (is_write && !tlb_entry->writable) {
            stats->protection_faults++;
            out.result = TLB_PROTECTION_FAULT;
            return out;
        }
        // Update access bits
        tlb_entry->referenced = true;
        if (is_write) {
            tlb_entry->dirty = true;
        }
        out.pfn = tlb_entry->pfn;
        out.from_tlb = true;
        out.result = TLB_HIT;
        return out;
    }
    // TLB MISS - slow path
    stats->tlb_misses++;
    // Walk the page table
    pte_t *pte = &page_table[vpn];
    // Check valid bit
    if (!pte->valid) {
        stats->page_faults++;
        // Demand paging would go here
        out.result = TLB_MISS;  // Caller must handle
        return out;
    }
    // Check permissions
    if (is_write && !pte->writable) {
        stats->protection_faults++;
        out.result = TLB_PROTECTION_FAULT;
        return out;
    }
    // Update page table access bits
    pte->referenced = true;
    if (is_write) {
        pte->dirty = true;
    }
    // Insert into TLB for future accesses
    tlb_insert(tlb, vpn, pte->pfn, pte, tlb->current_asid);
    out.pfn = pte->pfn;
    out.from_tlb = false;
    out.result = TLB_MISS;  // Miss, but successfully resolved
    return out;
}
```
The key insight: **a TLB miss doesn't mean failure**. It just means we took the slow path. The translation still succeeds; we just had to work harder for it.
---
## TLB Insertion and Eviction
When we bring a new translation into the TLB, we might need to evict an existing entry. This is where cache replacement policies come into play.
### The Eviction Problem
The TLB has a fixed number of entries (16, 32, 64, etc.). When it's full and a new translation arrives, we must choose which existing entry to evict.
```c
int tlb_find_victim(tlb_t *tlb) {
    // Simple random replacement
    // (In practice, LRU or pseudo-LRU is more common)
    return rand() % tlb->capacity;
}
void tlb_insert(
    tlb_t *tlb,
    uint32_t vpn,
    uint32_t pfn,
    pte_t *pte,
    uint32_t asid
) {
    // Find a slot (either empty or victim)
    int slot = -1;
    // First, look for an empty slot
    for (uint32_t i = 0; i < tlb->capacity; i++) {
        if (!tlb->entries[i].valid) {
            slot = i;
            break;
        }
    }
    // If no empty slot, evict someone
    if (slot == -1) {
        slot = tlb_find_victim(tlb);
        // CRITICAL: Write back dirty bit to page table before eviction
        tlb_entry_t *victim = &tlb->entries[slot];
        if (victim->valid && victim->dirty) {
            // Propagate dirty bit to page table
            page_table[victim->vpn].dirty = true;
        }
        if (victim->valid && victim->referenced) {
            // Propagate referenced bit to page table
            page_table[victim->vpn].referenced = true;
        }
    }
    // Install the new entry
    tlb_entry_t *entry = &tlb->entries[slot];
    entry->vpn = vpn;
    entry->pfn = pfn;
    entry->asid = asid;
    entry->valid = true;
    entry->readable = pte->readable;
    entry->writable = pte->writable;
    entry->dirty = false;      // Clean in TLB (will be set on write)
    entry->referenced = false; // Will be set on first access
}
```
### Replacement Policies
**Random**: Simple, fast, and surprisingly effective. With a seeded PRNG, results are reproducible for testing.
```c
int tlb_find_victim_random(tlb_t *tlb) {
    static unsigned int seed = 12345;
    seed = seed * 1103515245 + 12345;
    return seed % tlb->capacity;
}
```
**LRU (Least Recently Used)**: Evict the entry that hasn't been used for the longest time. Requires tracking access order.
```c
typedef struct {
    tlb_entry_t *entries;
    uint32_t capacity;
    uint64_t *last_used;  // Timestamp of last access
    uint64_t clock;       // Monotonic counter
    // ... other fields
} tlb_lru_t;
int tlb_find_victim_lru(tlb_lru_t *tlb) {
    uint64_t oldest = UINT64_MAX;
    int victim = 0;
    for (uint32_t i = 0; i < tlb->capacity; i++) {
        if (tlb->last_used[i] < oldest) {
            oldest = tlb->last_used[i];
            victim = i;
        }
    }
    return victim;
}
// Call this on every TLB hit
void tlb_update_lru(tlb_lru_t *tlb, uint32_t index) {
    tlb->last_used[index] = ++tlb->clock;
}
```
**Comparison**:
| Policy | Implementation Complexity | Predictability | Real Usage |
|--------|--------------------------|----------------|------------|
| Random | O(1), trivial | Non-deterministic | Simple embedded systems |
| LRU | O(n) scan or O(1) with list | Deterministic | Most modern CPUs (pseudo-LRU) |
| FIFO | O(1) queue | Deterministic | Some embedded systems |
Real CPUs typically use **pseudo-LRU** approximations because true LRU requires too much hardware. A 64-entry TLB with true LRU would need 64 timestamp registers and comparison logic—a significant area cost.
---
## ASID: Why Context Switches Don't Always Flush
In the old days, every context switch required a complete TLB flush. Process A's translations were invalidated, and Process B started with an empty TLB. This worked, but it was expensive:
1. Process B's first few memory accesses would all be TLB misses
2. The TLB would slowly warm up as Process B ran
3. Context switch back to Process A → flush again → warm up again
Modern systems use **ASID (Address Space ID)** tagging to avoid this penalty.

![TLB Entry Structure with ASID](./diagrams/tdd-diag-m2-01.svg)

![Context Switch with ASID Tagging](./diagrams/diag-context-switch-asid.svg)

![Context Switch with ASID](./diagrams/tdd-diag-m2-07.svg)

### How ASIDs Work
Each process is assigned a unique ASID (typically 8-16 bits). Every TLB entry stores the ASID of the process that owns it. On lookup, the CPU compares the current process's ASID with the entry's ASID:
```c
tlb_entry_t* tlb_lookup_with_asid(tlb_t *tlb, uint32_t vpn, uint32_t asid) {
    for (uint32_t i = 0; i < tlb->capacity; i++) {
        tlb_entry_t *entry = &tlb->entries[i];
        if (entry->valid && 
            entry->vpn == vpn && 
            entry->asid == asid) {  // ASID must match!
            return entry;
        }
    }
    return NULL;
}
```
Now, on a context switch:
```c
void context_switch(tlb_t *tlb, uint32_t new_asid) {
    // With ASID support: just change the current ASID
    // Entries from the old process stay in the TLB but won't match
    tlb->current_asid = new_asid;
    // No flush needed! The old entries are preserved.
}
```
### ASID Exhaustion
ASIDs are a limited resource. An 8-bit ASID field means only 256 unique processes can have entries in the TLB simultaneously. When you run out of ASIDs:
```c
uint32_t allocate_asid(tlb_t *tlb) {
    static uint32_t next_asid = 0;
    static bool asid_in_use[256] = {false};
    // Find a free ASID
    for (int i = 0; i < 256; i++) {
        uint32_t candidate = (next_asid + i) % 256;
        if (!asid_in_use[candidate]) {
            asid_in_use[candidate] = true;
            next_asid = (candidate + 1) % 256;
            return candidate;
        }
    }
    // All ASIDs in use - must flush and recycle
    tlb_flush(tlb);
    memset(asid_in_use, 0, sizeof(asid_in_use));
    uint32_t recycled = next_asid;
    asid_in_use[recycled] = true;
    next_asid = (recycled + 1) % 256;
    return recycled;
}
```
When ASIDs are exhausted, we flush the entire TLB and start fresh. This is rare—most systems have far fewer than 256 simultaneously active processes.
---
## TLB Flush: When Everything Must Go
Sometimes we need to invalidate TLB entries explicitly:
```c
void tlb_flush(tlb_t *tlb) {
    for (uint32_t i = 0; i < tlb->capacity; i++) {
        // Write back dirty bits before invalidating
        if (tlb->entries[i].valid && tlb->entries[i].dirty) {
            uint32_t vpn = tlb->entries[i].vpn;
            page_table[vpn].dirty = true;
        }
        if (tlb->entries[i].valid && tlb->entries[i].referenced) {
            uint32_t vpn = tlb->entries[i].vpn;
            page_table[vpn].referenced = true;
        }
        tlb->entries[i].valid = false;
    }
    tlb->flushes++;
}
void tlb_flush_asid(tlb_t *tlb, uint32_t asid) {
    // Flush only entries belonging to a specific process
    for (uint32_t i = 0; i < tlb->capacity; i++) {
        if (tlb->entries[i].asid == asid) {
            // Write back dirty bits
            if (tlb->entries[i].dirty) {
                page_table[tlb->entries[i].vpn].dirty = true;
            }
            tlb->entries[i].valid = false;
        }
    }
    tlb->flushes++;
}
void tlb_invalidate(tlb_t *tlb, uint32_t vpn, uint32_t asid) {
    // Invalidate a single entry (used when PTE changes)
    for (uint32_t i = 0; i < tlb->capacity; i++) {
        if (tlb->entries[i].valid &&
            tlb->entries[i].vpn == vpn &&
            tlb->entries[i].asid == asid) {
            if (tlb->entries[i].dirty) {
                page_table[vpn].dirty = true;
            }
            tlb->entries[i].valid = false;
            return;
        }
    }
}
```
---
## TLB Coherency: The Stale Entry Problem

![TLB Architecture](./diagrams/tdd-diag-m2-02.svg)

![TLB Coherency: Stale Entry Problem](./diagrams/diag-tlb-coherency.svg)

![ASID Allocation and Exhaustion](./diagrams/tdd-diag-m2-12.svg)

![TLB Statistics State Machine](./diagrams/tdd-diag-m2-11.svg)

![TLB Coherency Problem](./diagrams/tdd-diag-m2-08.svg)

Here's a subtle bug that trips up many implementers: **the TLB can become inconsistent with the page table.**
### Scenario: Page Eviction
1. Process accesses VPN 0x1, TLB caches the translation (PFN 0x7F)
2. Later, the OS evicts page 0x1 to make room for another page
3. The OS updates the page table: `page_table[0x1].valid = false`
4. **But the TLB still has the old entry!** VPN 0x1 → PFN 0x7F with valid=true
5. Process accesses VPN 0x1 again
6. TLB hit! Returns PFN 0x7F... which now contains completely different data
This is a **TLB coherency bug**. The TLB is serving stale translations.
### The Fix: Invalidation on Page Table Changes
Whenever the OS modifies a page table entry, it must invalidate the corresponding TLB entry:
```c
void evict_page(uint32_t vpn, pte_t *page_table, tlb_t *tlb) {
    // Step 1: Invalidate TLB entry BEFORE modifying page table
    tlb_invalidate(tlb, vpn, tlb->current_asid);
    // Step 2: Now safe to modify page table
    page_table[vpn].valid = false;
    page_table[vpn].pfn = 0;
    // The TLB will re-fetch this translation on next access
    // (which will trigger a page fault if the page isn't reloaded)
}
```
### The Write-Back Problem
The reverse problem exists too: the TLB's dirty/referenced bits might be more up-to-date than the page table.
```c
void tlb_invalidate(tlb_t *tlb, uint32_t vpn, uint32_t asid) {
    for (uint32_t i = 0; i < tlb->capacity; i++) {
        if (tlb->entries[i].valid &&
            tlb->entries[i].vpn == vpn &&
            tlb->entries[i].asid == asid) {
            // CRITICAL: Propagate bits to page table before invalidating
            if (tlb->entries[i].dirty) {
                page_table[vpn].dirty = true;
            }
            if (tlb->entries[i].referenced) {
                page_table[vpn].referenced = true;
            }
            tlb->entries[i].valid = false;
            return;
        }
    }
}
```
This is called **write-back** of TLB metadata. Without it, a page could be clean in the page table but dirty in the TLB. If the OS evicts the page without checking the TLB, it won't write the dirty data to swap—data loss!
---
## Hardware Soul: What Really Happens in Silicon
Let's trace a memory access through the actual hardware to understand where the cycles go.
### Level 1: Application (Software)
```c
int value = *ptr;  // ptr = 0x00001234
```
The compiler emits something like:
```asm
mov eax, [ptr]    ; Load pointer value
mov ebx, [eax]    ; Dereference - this is where translation happens
```
### Level 2: Memory Management Unit (Hardware)
When the CPU executes `mov ebx, [eax]`:
1. **Extract VPN**: 0x00001234 → VPN = 0x1, offset = 0x234
2. **TLB Lookup** (parallel comparison across all entries)
   - TLB hit: 1-2 cycles, we have the PFN
   - TLB miss: 20-200 cycles for page table walk
3. **Permission Check**: Read allowed? (part of TLB entry)
4. **Physical Address Composition**: (PFN << 12) | offset
### Level 3: Cache Hierarchy (Hardware)
With the physical address:
1. **L1 Data Cache Lookup**: 4-8 cycles
   - Hit: data returned
   - Miss: proceed to L2
2. **L2 Cache Lookup**: 10-20 cycles
3. **L3 Cache Lookup**: 30-50 cycles
4. **Main Memory**: 100-200 cycles
### The Combined Cost
| Scenario | Total Latency |
|----------|--------------|
| TLB hit + L1 hit | 5-10 cycles |
| TLB hit + L1 miss + L2 hit | 15-25 cycles |
| TLB miss + page table in cache + L1 hit | 25-50 cycles |
| TLB miss + page table walk to RAM + L1 hit | 150-300 cycles |
| TLB miss + page fault | 10,000,000+ cycles |
The TLB is the difference between "memory access" being 5 cycles or 300 cycles. That's a **60× speedup** for hot translations.
---
## Implementation: The Complete TLB
Let's put it all together into a working TLB implementation.
### Data Structures
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
// TLB Entry
typedef struct {
    uint32_t vpn;
    uint32_t pfn;
    uint32_t asid;
    bool valid;
    bool readable;
    bool writable;
    bool dirty;
    bool referenced;
    uint64_t last_used;  // For LRU
} tlb_entry_t;
// TLB Structure
typedef struct {
    tlb_entry_t *entries;
    uint32_t capacity;
    uint32_t current_asid;
    uint64_t clock;
    // Statistics
    uint64_t hits;
    uint64_t misses;
    uint64_t flushes;
    uint64_t evictions;
} tlb_t;
// Page Table Entry (from Milestone 1)
typedef struct {
    uint32_t pfn;
    bool valid;
    bool readable;
    bool writable;
    bool dirty;
    bool referenced;
} pte_t;
// Statistics
typedef struct {
    uint64_t total_accesses;
    uint64_t page_faults;
    uint64_t protection_faults;
    uint64_t tlb_hits;
    uint64_t tlb_misses;
    double tlb_hit_rate;
} stats_t;
```
### TLB Creation and Destruction
```c
tlb_t* tlb_create(uint32_t capacity) {
    tlb_t *tlb = calloc(1, sizeof(tlb_t));
    if (!tlb) return NULL;
    tlb->entries = calloc(capacity, sizeof(tlb_entry_t));
    if (!tlb->entries) {
        free(tlb);
        return NULL;
    }
    tlb->capacity = capacity;
    tlb->current_asid = 0;
    tlb->clock = 0;
    // Initialize all entries as invalid
    for (uint32_t i = 0; i < capacity; i++) {
        tlb->entries[i].valid = false;
    }
    return tlb;
}
void tlb_destroy(tlb_t *tlb) {
    if (tlb) {
        free(tlb->entries);
        free(tlb);
    }
}
```
### TLB Lookup
```c
// Returns index of entry if found, -1 if miss
int tlb_lookup_index(tlb_t *tlb, uint32_t vpn, uint32_t asid) {
    for (uint32_t i = 0; i < tlb->capacity; i++) {
        tlb_entry_t *entry = &tlb->entries[i];
        if (entry->valid &&
            entry->vpn == vpn &&
            entry->asid == asid) {
            return i;
        }
    }
    return -1;
}
tlb_entry_t* tlb_lookup(tlb_t *tlb, uint32_t vpn) {
    int idx = tlb_lookup_index(tlb, vpn, tlb->current_asid);
    if (idx >= 0) {
        return &tlb->entries[idx];
    }
    return NULL;
}
```
### LRU Victim Selection
```c
int tlb_find_victim_lru(tlb_t *tlb, pte_t *page_table) {
    uint64_t oldest = UINT64_MAX;
    int victim = 0;
    for (uint32_t i = 0; i < tlb->capacity; i++) {
        if (!tlb->entries[i].valid) {
            // Found an empty slot - no need to evict
            return i;
        }
        if (tlb->entries[i].last_used < oldest) {
            oldest = tlb->entries[i].last_used;
            victim = i;
        }
    }
    // Write back dirty/referenced bits before eviction
    tlb_entry_t *entry = &tlb->entries[victim];
    if (entry->dirty) {
        page_table[entry->vpn].dirty = true;
    }
    if (entry->referenced) {
        page_table[entry->vpn].referenced = true;
    }
    tlb->evictions++;
    return victim;
}
```
### TLB Insertion
```c
void tlb_insert(
    tlb_t *tlb,
    uint32_t vpn,
    uint32_t pfn,
    pte_t *pte,
    pte_t *page_table
) {
    // Find a victim slot (may write back dirty bits)
    int slot = tlb_find_victim_lru(tlb, page_table);
    // Install the new entry
    tlb_entry_t *entry = &tlb->entries[slot];
    entry->vpn = vpn;
    entry->pfn = pfn;
    entry->asid = tlb->current_asid;
    entry->valid = true;
    entry->readable = pte->readable;
    entry->writable = pte->writable;
    entry->dirty = false;
    entry->referenced = false;
    entry->last_used = ++tlb->clock;
}
```
### Complete Translation with TLB
```c
typedef enum {
    TRANS_OK,
    TRANS_PAGE_FAULT,
    TRANS_PROTECTION_FAULT
} trans_result_t;
typedef struct {
    trans_result_t result;
    uint32_t physical_address;
    bool tlb_hit;
} trans_output_t;
trans_output_t translate_with_tlb(
    tlb_t *tlb,
    uint32_t va,
    bool is_write,
    pte_t *page_table,
    stats_t *stats
) {
    trans_output_t out = {0};
    uint32_t vpn = va >> PAGE_SHIFT;
    uint32_t offset = va & PAGE_MASK;
    stats->total_accesses++;
    // Step 1: Check TLB
    tlb_entry_t *tlb_entry = tlb_lookup(tlb, vpn);
    if (tlb_entry != NULL) {
        // TLB HIT - fast path!
        stats->tlb_hits++;
        out.tlb_hit = true;
        // Update LRU timestamp
        tlb_entry->last_used = ++tlb->clock;
        // Check permissions
        if (is_write && !tlb_entry->writable) {
            stats->protection_faults++;
            out.result = TRANS_PROTECTION_FAULT;
            return out;
        }
        // Update access bits in TLB
        tlb_entry->referenced = true;
        if (is_write) {
            tlb_entry->dirty = true;
        }
        // Compose physical address
        out.physical_address = (tlb_entry->pfn << PAGE_SHIFT) | offset;
        out.result = TRANS_OK;
        return out;
    }
    // TLB MISS - slow path
    stats->tlb_misses++;
    out.tlb_hit = false;
    // Walk the page table
    pte_t *pte = &page_table[vpn];
    // Check valid bit
    if (!pte->valid) {
        stats->page_faults++;
        out.result = TRANS_PAGE_FAULT;
        return out;
    }
    // Check permissions
    if (is_write && !pte->writable) {
        stats->protection_faults++;
        out.result = TRANS_PROTECTION_FAULT;
        return out;
    }
    // Update page table access bits
    pte->referenced = true;
    if (is_write) {
        pte->dirty = true;
    }
    // Insert into TLB
    tlb_insert(tlb, vpn, pte->pfn, pte, page_table);
    // Compose physical address
    out.physical_address = (pte->pfn << PAGE_SHIFT) | offset;
    out.result = TRANS_OK;
    return out;
}
```
### TLB Flush Operations
```c
void tlb_flush(tlb_t *tlb, pte_t *page_table) {
    for (uint32_t i = 0; i < tlb->capacity; i++) {
        tlb_entry_t *entry = &tlb->entries[i];
        if (entry->valid) {
            // Write back dirty/referenced bits
            if (entry->dirty) {
                page_table[entry->vpn].dirty = true;
            }
            if (entry->referenced) {
                page_table[entry->vpn].referenced = true;
            }
            entry->valid = false;
        }
    }
    tlb->flushes++;
}
void tlb_flush_asid(tlb_t *tlb, uint32_t asid, pte_t *page_table) {
    for (uint32_t i = 0; i < tlb->capacity; i++) {
        tlb_entry_t *entry = &tlb->entries[i];
        if (entry->valid && entry->asid == asid) {
            if (entry->dirty) {
                page_table[entry->vpn].dirty = true;
            }
            if (entry->referenced) {
                page_table[entry->vpn].referenced = true;
            }
            entry->valid = false;
        }
    }
    tlb->flushes++;
}
void tlb_invalidate_entry(tlb_t *tlb, uint32_t vpn, uint32_t asid, pte_t *page_table) {
    int idx = tlb_lookup_index(tlb, vpn, asid);
    if (idx >= 0) {
        tlb_entry_t *entry = &tlb->entries[idx];
        // Write back dirty/referenced bits
        if (entry->dirty) {
            page_table[entry->vpn].dirty = true;
        }
        if (entry->referenced) {
            page_table[entry->vpn].referenced = true;
        }
        entry->valid = false;
    }
}
```
### Context Switch Support
```c
void context_switch(tlb_t *tlb, uint32_t new_asid, pte_t *page_table) {
    // With ASID support: just change the ASID
    // Old entries remain but won't match lookups
    tlb->current_asid = new_asid;
    // Without ASID support, you would need:
    // tlb_flush(tlb, page_table);
}
// Simulate running multiple processes
typedef struct {
    uint32_t pid;
    uint32_t asid;
    pte_t *page_table;
    uint64_t instruction_count;
} process_t;
void run_process(
    tlb_t *tlb,
    process_t *proc,
    memory_access_t *accesses,
    size_t num_accesses,
    stats_t *stats
) {
    // Context switch to this process
    context_switch(tlb, proc->asid, proc->page_table);
    // Process memory accesses
    for (size_t i = 0; i < num_accesses; i++) {
        translate_with_tlb(
            tlb,
            accesses[i].virtual_address,
            accesses[i].is_write,
            proc->page_table,
            stats
        );
    }
}
```
### Statistics Reporting
```c
void print_tlb_statistics(tlb_t *tlb, stats_t *stats) {
    printf("\n");
    printf("========================================\n");
    printf("          TLB STATISTICS                \n");
    printf("========================================\n");
    printf("TLB capacity:           %u entries\n", tlb->capacity);
    printf("Total accesses:         %lu\n", stats->total_accesses);
    printf("TLB hits:               %lu\n", stats->tlb_hits);
    printf("TLB misses:             %lu\n", stats->tlb_misses);
    printf("TLB flushes:            %lu\n", tlb->flushes);
    printf("TLB evictions:          %lu\n", tlb->evictions);
    if (stats->total_accesses > 0) {
        stats->tlb_hit_rate = (double)stats->tlb_hits / stats->total_accesses * 100.0;
        printf("TLB hit rate:           %.2f%%\n", stats->tlb_hit_rate);
    }
    printf("Page faults:            %lu\n", stats->page_faults);
    printf("Protection faults:      %lu\n", stats->protection_faults);
    printf("========================================\n");
}
```
---
## Testing Your TLB
### Test Case 1: Basic TLB Hit/Miss
Create a trace that accesses the same page multiple times:
```
# TLB hit test
R 0x00001000    # Miss (first access), inserts into TLB
R 0x00001004    # Hit (same page, VPN=1)
R 0x00001008    # Hit
R 0x00002000    # Miss (different page, VPN=2)
R 0x00002004    # Hit
R 0x00001000    # Hit (back to VPN=1, still in TLB)
```
**Expected**: 2 TLB misses, 4 TLB hits, 66.7% hit rate.
### Test Case 2: TLB Capacity and Eviction
With a 4-entry TLB:
```
# Fill the TLB
R 0x00001000    # Miss, insert VPN 1
R 0x00002000    # Miss, insert VPN 2
R 0x00003000    # Miss, insert VPN 3
R 0x00004000    # Miss, insert VPN 4 (TLB now full)
R 0x00005000    # Miss, evict LRU, insert VPN 5
R 0x00001000    # Miss (VPN 1 was evicted), re-insert
```
**Expected**: All misses (or 5 misses + 1 hit if the re-access found VPN 1).
### Test Case 3: ASID Context Switch
```
# Process A (ASID 1)
R 0x00001000    # Miss, insert with ASID 1
R 0x00001000    # Hit (same ASID)
# Context switch to Process B (ASID 2)
R 0x00001000    # Miss (same VPN but different ASID!)
R 0x00001000    # Hit (now cached for ASID 2)
# Context switch back to Process A (ASID 1)
R 0x00001000    # Hit (ASID 1 entry still in TLB!)
```
**Key insight**: Without ASIDs, the third access would miss because we flushed. With ASIDs, it hits.
### Test Case 4: TLB Coherency
```c
void test_tlb_coherency(void) {
    tlb_t *tlb = tlb_create(16);
    pte_t *page_table = calloc(1024, sizeof(pte_t));
    stats_t stats = {0};
    // Set up a page
    page_table[1].pfn = 42;
    page_table[1].valid = true;
    page_table[1].readable = true;
    page_table[1].writable = true;
    // Access the page (inserts into TLB)
    trans_output_t result = translate_with_tlb(
        tlb, 0x1000, false, page_table, &stats
    );
    assert(result.result == TRANS_OK);
    assert(!result.tlb_hit);  // First access is a miss
    // Second access should hit
    result = translate_with_tlb(tlb, 0x1000, false, page_table, &stats);
    assert(result.result == TRANS_OK);
    assert(result.tlb_hit);
    // Now invalidate the page in the page table
    page_table[1].valid = false;
    // CRITICAL: Must invalidate TLB entry!
    tlb_invalidate_entry(tlb, 1, 0, page_table);
    // Third access should now see the invalid page
    result = translate_with_tlb(tlb, 0x1000, false, page_table, &stats);
    assert(result.result == TRANS_PAGE_FAULT);
    printf("TLB coherency test passed!\n");
    tlb_destroy(tlb);
    free(page_table);
}
```
---
## Common Pitfalls
### Pitfall 1: Forgetting to Write Back Dirty Bits
**The mistake**: Evicting a TLB entry without propagating the dirty bit.
```c
// WRONG!
void tlb_evict_wrong(tlb_t *tlb, int index) {
    tlb->entries[index].valid = false;  // Data loss if dirty!
}
```
**The fix**: Always write back before eviction or invalidation.
```c
void tlb_evict_correct(tlb_t *tlb, int index, pte_t *page_table) {
    tlb_entry_t *entry = &tlb->entries[index];
    if (entry->dirty) {
        page_table[entry->vpn].dirty = true;
    }
    entry->valid = false;
}
```
### Pitfall 2: Not Updating Page Table Bits on TLB Hits
**The mistake**: Only updating dirty/referenced in the page table, not the TLB.
```c
// WRONG! Only updates page table
if (is_write) {
    page_table[vpn].dirty = true;
}
```
**The problem**: On a TLB hit, we never touch the page table. The dirty bit stays false in the page table. When the page is evicted from the TLB later, we might not write back.
**The fix**: Update both places.
```c
// Update TLB entry
tlb_entry->dirty = true;
tlb_entry->referenced = true;
// On eviction, propagate to page table
if (tlb_entry->dirty) {
    page_table[tlb_entry->vpn].dirty = true;
}
```
### Pitfall 3: Random Replacement Without Seeding
**The mistake**: Using `rand()` without a seed, making tests non-deterministic.
**The fix**: Use a seeded PRNG for reproducibility.
```c
// In your test harness
srand(12345);  // Fixed seed for reproducible tests
```
### Pitfall 4: ASID Width Limits
**The mistake**: Assuming ASIDs are unlimited.
**The reality**: 8-bit ASIDs = 256 processes. When you exceed this, you must flush and recycle.
---
## Design Decisions: Why This, Not That
| Option | Pros | Cons | Used By |
|--------|------|------|---------|
| **Fully Associative ✓** | Best hit rate, no conflict misses | O(n) search, complex hardware | Small TLBs (16-32 entries) |
| Set Associative (4-way) | Parallel lookup, balanced | Conflict misses possible | Most modern CPUs |
| Direct Mapped | Simplest, fastest | Worst conflict misses | Rarely used for TLBs |
| Replacement Policy | Pros | Cons | Used By |
|-------------------|------|------|---------|
| **LRU ✓** | Best locality exploitation | O(n) scan, or complex list | Most modern CPUs (pseudo-LRU) |
| Random | O(1), simple | Ignores locality | Embedded systems, testing |
| FIFO | O(1), simple | Bélády's anomaly | Rarely used |
| ASID Support | Pros | Cons | Used By |
|--------------|------|------|---------|
| **Yes ✓** | No flush on context switch | Limited ASID space, complexity | All modern CPUs |
| No | Simpler hardware | Full flush on every switch | Very old or embedded systems |
---
## What You've Built
At this point, your simulator has:
1. **A TLB cache** that stores recent virtual-to-physical translations
2. **ASID tagging** that allows multiple processes' entries to coexist
3. **LRU (or random) eviction** when the TLB is full
4. **Write-back of dirty/referenced bits** on eviction and invalidation
5. **TLB coherency** through explicit invalidation when page tables change
6. **Statistics tracking** for hits, misses, hit rate, and flushes
This is exactly what real hardware does. The Intel x86-64 architecture has:
- L1 dTLB: 64 entries, 4-way set associative
- L1 iTLB: 128 entries, 4-way set associative
- L2 TLB: 1536 entries, unified for instructions and data
- 12-bit ASIDs (called PCIDs in Intel terminology)
Your simulator mirrors this architecture, just with configurable parameters.
---
## Knowledge Cascade: Where This Leads
### Same Domain Connections
**→ Multi-Level Page Tables (Milestone 3)**: TLBs become even more critical with multi-level tables. A four-level page table walk requires four memory accesses. The TLB collapses all four into a single lookup. This is why TLB coverage (how much of the address space is cached) matters more than raw entry count.
**→ Page Replacement (Milestone 4)**: The referenced and dirty bits you're tracking in the TLB are the key inputs to page replacement algorithms. LRU needs to know which pages have been accessed recently. The referenced bit is exactly that information, collected by hardware for free.
**→ Huge Pages**: Modern systems support 2 MB or 1 GB pages. A single TLB entry for a 2 MB page covers 512× more address space than a 4 KB entry. This dramatically increases TLB coverage. The tradeoff: internal fragmentation (a 2 MB page for a 4 KB allocation wastes 2 MB - 4 KB).
### Cross-Domain Connections
**→ Database Buffer Pools**: A database's buffer pool is to disk pages what the TLB is to page tables. Both cache expensive lookups (disk I/O vs. page table walks) with limited capacity and eviction policies. The same LRU vs. Clock vs. LFU debates apply. When PostgreSQL evicts a page from the buffer pool to make room for another, it's making the same tradeoff as the TLB evicting a translation.
**→ KPTI (Kernel Page Table Isolation)**: The Meltdown vulnerability (2018) forced OSes to separate user and kernel page tables. On every syscall, the OS switches page tables and flushes the TLB. This added 5-10% overhead to syscall-heavy workloads. The TLB flush cost was a measurable performance regression—a real-world demonstration of why ASIDs matter.
**→ Game Engine Spatial Hashing**: Open-world games map 3D coordinates to terrain chunks. The lookup (which chunk contains position X,Y,Z?) is cached in a hash table with limited size. Evicting a chunk from this cache forces a slow load from disk. Same pattern: cache expensive lookups, evict when full, locality determines hit rate.
**→ Redis Key Hashing**: Redis clusters distribute keys across nodes using hash slots. Clients cache the mapping (key → node). When the cluster rebalances, the cache becomes stale—exactly like a TLB after page table modification. Redis clients must invalidate their cache entries, just like TLB invalidation.
**→ CDN Cache Keys**: A CDN caches (URL, headers) → (content). The "translation" is from request to response. Cache eviction policies (LRU, LFU) determine what stays. The hit rate determines how many requests go to the origin server vs. being served from cache. Same economics, different scale.
### Historical Context
**→ The Rise of Paging (1980s)**: Early PCs (8086, 80286) used segmentation, not paging. The 80386 introduced paging with a single-level page table but no TLB. Every memory access walked the page table in memory. Performance was terrible. The 80486 added a small TLB (32 entries), and memory performance improved dramatically.
**→ The ASID Innovation**: Early TLBs didn't have ASIDs. Every context switch flushed the entire TLB. This worked for single-user DOS systems but became a bottleneck for multi-user Unix systems. MIPS introduced ASIDs in the R2000 (1986), and other architectures followed. The performance gain for context-switch-heavy workloads was substantial.
**→ The TLB Shootdown Problem**: On multi-core systems, when one core modifies a page table, it must invalidate TLB entries on *all* cores that might have cached that translation. This is called a TLB shootdown, and it requires inter-processor interrupts. The cost scales with core count—this is why huge pages (fewer translations to invalidate) matter more on many-core systems.
---
## Looking Ahead
In Milestone 3, you'll implement multi-level page tables. This reveals:
- Why a 64-bit flat page table is impossible (2^52 entries × 8 bytes = 36 petabytes)
- How hierarchical tables only allocate structures for regions that are actually used
- The CR3 register and how context switches change page table roots
- Why the TLB is even more critical when page table walks require multiple memory accesses
The translation function you've built (`translate_with_tlb()`) will gain a new layer underneath: instead of `page_table[vpn]`, you'll traverse `pdpt → page_directory → page_table → pte`. The TLB hides this complexity—a hit still returns in 1-2 cycles regardless of how many levels the page table has.
[[CRITERIA_JSON: {"milestone_id": "virtual-memory-sim-m2", "criteria": ["TLB lookup is checked before page table on every address translation; TLB hit returns PFN without page table access", "TLB miss triggers full page table walk and inserts the resulting translation (VPN → PFN + permissions) into the TLB", "TLB capacity is configurable (16, 32, 64 entries); when full, victim selection uses LRU (with last_used timestamps) or seeded random replacement", "TLB entries store ASID (uint32_t) alongside VPN; lookup requires both VPN and current ASID to match for a hit", "Context switch changes current_asid field; entries with different ASIDs remain in TLB but are treated as misses without flush", "TLB flush operation invalidates all entries and increments flush counter; flush_asid variant invalidates only entries matching a specific ASID", "Statistics tracked: tlb_hits (uint64_t), tlb_misses (uint64_t), tlb_hit_rate (double percentage), flushes (uint64_t), evictions (uint64_t)", "Dirty bit is propagated from TLB entry to page_table[vpn].dirty on eviction and invalidation; referenced bit similarly propagated", "TLB invalidation function (tlb_invalidate_entry) removes single entry by VPN+ASID after writing back dirty/referenced bits", "Permission checks occur on TLB hits using cached readable/writable bits; protection fault triggered on violation", "TLB entry update on hit: referenced bit set on any access, dirty bit set only on write, last_used timestamp updated for LRU", "Page table access bits (dirty, referenced) are updated on TLB miss before insertion into TLB"]}]
<!-- END_MS -->


<!-- MS_ID: virtual-memory-sim-m3 -->
# Milestone 3: Multi-Level Page Tables
## The Impossibility You've Been Ignoring
In Milestone 1, you built a flat page table: a simple array where `page_table[vpn]` gives you the PTE. Clean, simple, O(1) lookup. You might have thought: "Why would anyone make this more complicated?"
Here's the problem you've been avoiding.
A 32-bit address space with 4 KB pages has 2²⁰ = 1,048,576 pages. Each PTE is about 4 bytes (PFN + flags). That's **4 MB of page table per process**. Manageable.
Now consider a 64-bit address space. With 4 KB pages, you have 2⁵² possible pages (using only 52 bits for physical addresses). Each PTE is 8 bytes. That's:
```
2^52 entries × 8 bytes = 36,028,797,018,963,968 bytes
                    = 36 petabytes
```
**A single flat page table would be 36 petabytes.** Per process. That's more RAM than exists in most data centers.

![Multi-Level Architecture](./diagrams/tdd-diag-m3-08.svg)

![Memory Overhead: Flat vs Multi-Level for Sparse Address Space](./diagrams/diag-sparse-vs-dense-memory.svg)

![Sparse vs Dense Access Patterns](./diagrams/tdd-diag-m3-07.svg)

The lie you've been telling yourself is that the address space is "full." It isn't. Real programs use a tiny fraction of their address space:
- **Code**: Maybe 1-10 MB at address 0x00400000
- **Heap**: Grows upward from ~0x01000000, but rarely exceeds a few GB
- **Shared libraries**: Scattered at various addresses, maybe 50-100 MB total
- **Stack**: Grows downward from near 0x7FFFFFFFFFFF, typically < 10 MB
- **Everything else**: Empty. Unmapped. Never touched.
A typical process might touch 1 GB of a 256 TB address space. That's **0.0004% utilization**. A flat page table allocates metadata for 100% of the space to serve 0.0004% of it.
**This is the fundamental tension**: hardware provides a massive, sparse address space (64-bit = 18 exabytes theoretically), but programs use only tiny, scattered regions. A flat data structure can't efficiently represent sparse data.
Multi-level page tables are the compression scheme that solves this.
---
## The Compression Insight: Only Allocate What You Use
The key realization is that page table entries are only needed for pages that *exist*. If a region of the address space is unmapped, we shouldn't need to store PTEs for it.
But we can't just use a sparse array—that would require searching on every access. We need a structure that:
1. Supports O(1) or O(depth) lookup (not O(n) search)
2. Only allocates storage for mapped regions
3. Works with the bit-extraction approach hardware uses
The solution is **hierarchical indirection**.
### The Two-Level Scheme
Instead of one giant array, we split the virtual page number (VPN) into two parts:
```
┌─────────────────┬─────────────────┬────────────────────┐
│  Directory Index│   Table Index   │      Offset        │
│    (10 bits)    │    (10 bits)    │     (12 bits)      │
└─────────────────┴─────────────────┴────────────────────┘
        31        22 21            12 11                 0
```

![Two-Level Page Table Bit Extraction](./diagrams/tdd-diag-m3-01.svg)

![Two-Level Page Table: Bit Extraction](./diagrams/diag-multi-level-bit-extraction.svg)

![Page Table Memory Accounting](./diagrams/tdd-diag-m3-12.svg)

- **Directory Index (bits 31-22)**: Which entry in the top-level *page directory*?
- **Table Index (bits 21-12)**: Which entry in the second-level *page table*?
- **Offset (bits 11-0)**: Which byte within the page? (unchanged from before)
With 10 bits for directory index, we have 2¹⁰ = 1024 directory entries. Each directory entry points to a second-level page table (or is NULL if that entire region is unmapped).
With 10 bits for table index, each second-level table has 2¹⁰ = 1024 entries. Each entry is a PTE pointing to a physical frame.
**The compression magic**: Second-level tables are only allocated when needed. If directory entry 5 is NULL, that means the entire region covered by that entry (4 MB of virtual address space) is unmapped—no storage needed.
### The Math: Why This Saves Memory
Let's calculate memory usage for a sparse address space:
**Flat page table (32-bit):**
- 1,048,576 entries × 4 bytes = **4 MB** (always)
**Two-level page table:**
- Page directory: 1024 entries × 4 bytes = 4 KB (always)
- Each second-level table: 1024 entries × 4 bytes = 4 KB
- Total: 4 KB + (number of second-level tables × 4 KB)
If a process uses 10 distinct regions (code, heap, stack, a few libraries), it might need 10 second-level tables:
```
4 KB + (10 × 4 KB) = 44 KB
```
**44 KB vs 4 MB** — that's a 90× reduction in memory overhead.
For a 64-bit address space, the savings are even more dramatic. A four-level page table (what x86-64 actually uses) might only need a few kilobytes for a typical process, compared to the impossible 36 petabytes for a flat structure.
---
## The Page Directory Entry: Pointer with Permissions
A page directory entry (PDE) looks similar to a PTE, but it points to a page table instead of a frame:
```c
typedef struct {
    uint32_t page_table_addr;  // Physical address of second-level table
    bool present;              // Is this directory entry valid?
    bool writable;             // Can pages in this region be written?
    bool user_accessible;      // Can user-mode access this region?
    // ... other flags (cache disabled, accessed, etc.)
} pde_t;
```
The critical field is `page_table_addr` — it's a physical address pointing to a second-level page table. If `present = false`, the entire region is unmapped.
**Key insight**: The PDE is the "compression point." When present=0, we don't just say "one page is unmapped" — we say "1024 pages are unmapped, and we're not storing any of them."
---
## The Page Table Walk: Following the Pointers

![Memory Overhead Comparison](./diagrams/tdd-diag-m3-06.svg)

![Page Table Walk: CR3 → Physical Address](./diagrams/diag-cr3-to-pfn-walk.svg)

![Walk Failure Cases](./diagrams/tdd-diag-m3-09.svg)

Translation now requires multiple memory accesses:
```
1. Read CR3 register → physical address of page directory
2. Extract directory index from virtual address
3. Read PDE: pde = page_directory[directory_index]
4. If pde.present == 0: PAGE FAULT (entire region unmapped)
5. Extract table index from virtual address
6. Read PTE: pte = page_table[table_index]
7. If pte.valid == 0: PAGE FAULT (specific page unmapped)
8. Compose physical address: pa = (pte.pfn << 12) | offset
```
Here's the implementation:
```c
// Bit extraction for two-level page table
#define PAGE_SHIFT      12
#define PAGE_SIZE       4096
#define PAGE_MASK       0xFFF
#define DIR_SHIFT       22
#define DIR_MASK        0x3FF        // 10 bits
#define TABLE_SHIFT     12
#define TABLE_MASK      0x3FF        // 10 bits
typedef struct {
    uint32_t page_table_pfn;   // PFN of second-level table (not byte address!)
    bool present;
    bool writable;
    bool user_accessible;
} pde_t;
typedef struct {
    uint32_t pfn;
    bool valid;
    bool readable;
    bool writable;
    bool dirty;
    bool referenced;
} pte_t;
// Extract indices from virtual address
static inline uint32_t get_directory_index(uint32_t va) {
    return (va >> DIR_SHIFT) & DIR_MASK;
}
static inline uint32_t get_table_index(uint32_t va) {
    return (va >> TABLE_SHIFT) & TABLE_MASK;
}
static inline uint32_t get_offset(uint32_t va) {
    return va & PAGE_MASK;
}
```
### The Complete Walk Function
```c
typedef enum {
    WALK_OK,
    WALK_DIR_NOT_PRESENT,   // Directory entry missing
    WALK_PAGE_FAULT,        // PTE valid=0
    WALK_PROTECTION         // Permission violation
} walk_result_t;
typedef struct {
    walk_result_t result;
    uint32_t physical_address;
} walk_output_t;
walk_output_t walk_two_level(
    uint32_t cr3,                    // Physical address of page directory
    uint32_t va,
    bool is_write,
    pde_t **directories,             // Array of page directories (simulated physical memory)
    pte_t **page_tables,             // Array of page tables (simulated physical memory)
    stats_t *stats
) {
    walk_output_t out = {0};
    stats->total_accesses++;
    uint32_t dir_index = get_directory_index(va);
    uint32_t table_index = get_table_index(va);
    uint32_t offset = get_offset(va);
    // Level 1: Check page directory
    pde_t *page_directory = directories[cr3];  // cr3 is index into our simulated physical memory
    if (page_directory == NULL) {
        stats->protection_faults++;
        out.result = WALK_DIR_NOT_PRESENT;
        return out;
    }
    pde_t *pde = &page_directory[dir_index];
    if (!pde->present) {
        // Entire region unmapped - this could be a page fault
        // (in a real system, the OS might decide to allocate a page table here)
        stats->page_faults++;
        out.result = WALK_DIR_NOT_PRESENT;
        return out;
    }
    // Check directory-level write permission
    if (is_write && !pde->writable) {
        stats->protection_faults++;
        out.result = WALK_PROTECTION;
        return out;
    }
    // Level 2: Check page table
    pte_t *page_table = page_tables[pde->page_table_pfn];
    if (page_table == NULL) {
        // This shouldn't happen if PDE is present, but handle it
        stats->page_faults++;
        out.result = WALK_PAGE_FAULT;
        return out;
    }
    pte_t *pte = &page_table[table_index];
    if (!pte->valid) {
        stats->page_faults++;
        out.result = WALK_PAGE_FAULT;
        return out;
    }
    // Check page-level permissions
    if (is_write && !pte->writable) {
        stats->protection_faults++;
        out.result = WALK_PROTECTION;
        return out;
    }
    // Update access bits
    pte->referenced = true;
    if (is_write) {
        pte->dirty = true;
    }
    // Success: compose physical address
    out.physical_address = (pte->pfn << PAGE_SHIFT) | offset;
    out.result = WALK_OK;
    return out;
}
```
---
## CR3: The Per-Process Root

![CR3 to Physical Address Walk](./diagrams/tdd-diag-m3-03.svg)

![Context Switch: CR3 Update and TLB Flush](./diagrams/diag-context-switch-cr3.svg)

The **CR3 register** (Control Register 3, also called the Page Directory Base Register or PDBR) holds the physical address of the current process's page directory. This is the "root" of the entire address translation tree.
When the OS performs a context switch:
```c
typedef struct {
    uint32_t pid;
    uint32_t cr3;              // Physical address of this process's page directory
    pde_t *page_directory;     // Pointer to the directory (in our simulation)
    // ... other process state
} process_t;
void context_switch(
    uint32_t *current_cr3,
    process_t *new_process,
    tlb_t *tlb,
    pte_t *page_table  // For TLB write-back (simplified)
) {
    // CR3 is changing - this is the fundamental context switch operation
    *current_cr3 = new_process->cr3;
    // TLB entries from the old process are now invalid
    // (Unless we're using ASIDs - see Milestone 2)
    tlb_flush(tlb, page_table);
    printf("[Context Switch] CR3 -> 0x%X (PID %u)\n", 
           new_process->cr3, new_process->pid);
}
```
**Why CR3 matters**: Every process has its own page directory. Switching CR3 instantly changes the entire virtual-to-physical mapping. The same virtual address `0x00400000` in Process A and Process B will translate to completely different physical addresses because they start from different roots.
This is the hardware mechanism that provides **memory isolation**. Process A cannot construct a virtual address that translates to Process B's physical frames because Process A's page directory doesn't contain mappings to those frames.
---
## On-Demand Table Allocation

![Context Switch CR3 Update](./diagrams/tdd-diag-m3-05.svg)

![On-Demand Second-Level Table Allocation](./diagrams/diag-on-demand-allocation.svg)

The true power of multi-level tables is revealed when we allocate second-level tables lazily:
```c
// Allocate a second-level page table on demand
uint32_t allocate_page_table(
    pde_t *page_directory,
    uint32_t dir_index,
    physical_memory_t *phys_mem
) {
    pde_t *pde = &page_directory[dir_index];
    if (pde->present) {
        // Already allocated
        return pde->page_table_pfn;
    }
    // Allocate a frame for the new page table
    int32_t pfn = allocate_frame(phys_mem);
    if (pfn < 0) {
        return (uint32_t)-1;  // Out of memory
    }
    // Initialize the page table (all entries invalid)
    pte_t *new_table = phys_mem->frames[pfn];
    for (int i = 0; i < 1024; i++) {
        new_table[i].valid = false;
        new_table[i].pfn = 0;
        new_table[i].readable = true;
        new_table[i].writable = true;
        new_table[i].dirty = false;
        new_table[i].referenced = false;
    }
    // Link it into the directory
    pde->page_table_pfn = pfn;
    pde->present = true;
    pde->writable = true;
    pde->user_accessible = true;
    printf("[Allocated Page Table] Dir %u -> PFN %u\n", dir_index, pfn);
    return pfn;
}
```
Now the page fault handler becomes:
```c
walk_output_t handle_page_fault(
    uint32_t cr3,
    uint32_t va,
    bool is_write,
    pde_t **directories,
    pte_t **page_tables,
    physical_memory_t *phys_mem,
    stats_t *stats
) {
    walk_output_t out = {0};
    uint32_t dir_index = get_directory_index(va);
    uint32_t table_index = get_table_index(va);
    uint32_t offset = get_offset(va);
    pde_t *page_directory = directories[cr3];
    pde_t *pde = &page_directory[dir_index];
    // Step 1: Ensure the page table exists
    if (!pde->present) {
        uint32_t pt_pfn = allocate_page_table(page_directory, dir_index, phys_mem);
        if (pt_pfn == (uint32_t)-1) {
            out.result = WALK_PAGE_FAULT;
            return out;
        }
    }
    // Step 2: Now handle the page fault in the second-level table
    pte_t *page_table = page_tables[pde->page_table_pfn];
    pte_t *pte = &page_table[table_index];
    if (!pte->valid) {
        // Allocate a physical frame for this page
        int32_t frame_pfn = allocate_frame(phys_mem);
        if (frame_pfn < 0) {
            out.result = WALK_PAGE_FAULT;
            return out;
        }
        pte->pfn = frame_pfn;
        pte->valid = true;
        pte->readable = true;
        pte->writable = true;
        pte->dirty = false;
        pte->referenced = false;
        printf("[Page Fault Handled] VA 0x%08X -> PFN %u\n", va, frame_pfn);
    }
    // Retry the translation
    out.physical_address = (pte->pfn << PAGE_SHIFT) | offset;
    out.result = WALK_OK;
    return out;
}
```
The beauty: if a process never touches directory 0 (addresses 0x00000000 - 0x003FFFFF), no page table is ever allocated for it. The PDE stays present=0, and the 4 KB for that table is saved.
---
## Memory Overhead Comparison
Let's measure the actual memory usage difference. We'll track:
- **Page directory**: Always 4 KB (1024 entries × 4 bytes)
- **Second-level tables**: 4 KB each, allocated on demand
- **Total overhead**: Sum of above
```c
typedef struct {
    uint64_t page_directory_bytes;
    uint64_t page_table_bytes;
    uint64_t total_overhead;
    uint32_t num_page_tables;
} memory_overhead_t;
memory_overhead_t measure_overhead(
    pde_t *page_directory,
    pte_t **page_tables,
    uint32_t num_possible_tables
) {
    memory_overhead_t overhead = {0};
    // Page directory is always allocated
    overhead.page_directory_bytes = 1024 * sizeof(pde_t);
    overhead.num_page_tables = 0;
    // Count allocated second-level tables
    for (uint32_t i = 0; i < 1024; i++) {
        if (page_directory[i].present && page_tables[page_directory[i].page_table_pfn] != NULL) {
            overhead.page_table_bytes += 1024 * sizeof(pte_t);
            overhead.num_page_tables++;
        }
    }
    overhead.total_overhead = overhead.page_directory_bytes + overhead.page_table_bytes;
    return overhead;
}
void print_overhead_comparison(
    pde_t *page_directory,
    pte_t **page_tables,
    uint32_t num_possible_tables,
    uint64_t flat_table_bytes
) {
    memory_overhead_t multi = measure_overhead(page_directory, page_tables, num_possible_tables);
    printf("\n");
    printf("========================================\n");
    printf("       MEMORY OVERHEAD COMPARISON       \n");
    printf("========================================\n");
    printf("Flat page table:        %lu bytes (%.2f MB)\n", 
           flat_table_bytes, flat_table_bytes / (1024.0 * 1024.0));
    printf("Multi-level overhead:   %lu bytes (%.2f KB)\n", 
           multi.total_overhead, multi.total_overhead / 1024.0);
    printf("  - Page directory:     %lu bytes\n", multi.page_directory_bytes);
    printf("  - Page tables:        %lu bytes (%u tables)\n", 
           multi.page_table_bytes, multi.num_page_tables);
    printf("Savings:                %.2fx (%.1f%% reduction)\n",
           (double)flat_table_bytes / multi.total_overhead,
           100.0 * (1.0 - (double)multi.total_overhead / flat_table_bytes));
    printf("========================================\n");
}
```
### Example: Sparse vs Dense Access Patterns
```c
void demonstrate_overhead(void) {
    // Simulate a 32-bit address space
    uint64_t flat_bytes = 1024 * 1024 * sizeof(pte_t);  // 4 MB
    printf("=== Sparse Address Space Test ===\n");
    // Access: code at 0x00400000, heap at 0x01000000, stack at 0x7FFF0000
    // These span only 3 directory entries
    // ...
    // Expected: 4 KB + (3 × 4 KB) = 16 KB vs 4 MB = 256x savings
    printf("\n=== Dense Address Space Test ===\n");
    // Access: sequential from 0x00000000 to 0x10000000 (256 MB)
    // This spans 64 directory entries
    // ...
    // Expected: 4 KB + (64 × 4 KB) = 260 KB vs 4 MB = 16x savings
}
```
**The pattern**: The sparser the address space, the bigger the savings. For realistic workloads (code, heap, stack, a few libraries), multi-level tables typically use 10-100× less memory than flat tables.
---
## Three-Level Page Tables: The Stretch Goal
For 32-bit systems, two levels work well. But what about 64-bit systems? Even with two levels, a fully populated 64-bit address space would need impossibly many second-level tables.

![On-Demand Table Allocation](./diagrams/tdd-diag-m3-04.svg)

![Three-Level Page Table Structure (Stretch)](./diagrams/diag-three-level-stretch.svg)

![Three-Level Stretch Goal Structure](./diagrams/tdd-diag-m3-11.svg)

The solution is **more levels**. x86-64 uses four levels:
```
┌──────────┬──────────┬──────────┬──────────┬────────────────────┐
│ PML4 idx │ PDPT idx │  PD idx  │  PT idx  │      Offset        │
│ (9 bits) │ (9 bits) │ (9 bits) │ (9 bits) │     (12 bits)      │
└──────────┴──────────┴──────────┴──────────┴────────────────────┘
```
For this project, implementing three levels is a stretch goal. Here's the structure:
```c
// Three-level page table: 2+9+9+12 for 32-bit with 4KB pages
// This allows only 4 PML4 entries (bits 31-30)
// Each PML4 entry covers 1 GB of address space
#define PML4_SHIFT      30
#define PML4_MASK       0x3         // 2 bits = 4 entries
#define PDPT_SHIFT      21
#define PDPT_MASK       0x1FF       // 9 bits = 512 entries
#define PT_SHIFT        12
#define PT_MASK         0x1FF       // 9 bits = 512 entries
typedef struct {
    uint32_t pdpt_pfn;
    bool present;
} pml4e_t;
typedef struct {
    uint32_t page_table_pfn;
    bool present;
} pdpte_t;
// Three-level walk
walk_output_t walk_three_level(
    uint32_t cr3,                    // PML4 physical address
    uint32_t va,
    bool is_write,
    pml4e_t *pml4,
    pdpte_t **pdpts,
    pte_t **page_tables,
    stats_t *stats
) {
    walk_output_t out = {0};
    uint32_t pml4_index = (va >> PML4_SHIFT) & PML4_MASK;
    uint32_t pdpt_index = (va >> PDPT_SHIFT) & PDPT_MASK;
    uint32_t pt_index = (va >> PT_SHIFT) & PT_MASK;
    uint32_t offset = va & PAGE_MASK;
    // Level 1: PML4
    if (!pml4[pml4_index].present) {
        stats->page_faults++;
        out.result = WALK_DIR_NOT_PRESENT;
        return out;
    }
    // Level 2: PDPT
    pdpte_t *pdpt = pdpts[pml4[pml4_index].pdpt_pfn];
    if (!pdpt[pdpt_index].present) {
        stats->page_faults++;
        out.result = WALK_DIR_NOT_PRESENT;
        return out;
    }
    // Level 3: Page table
    pte_t *pt = page_tables[pdpt[pdpt_index].page_table_pfn];
    pte_t *pte = &pt[pt_index];
    if (!pte->valid) {
        stats->page_faults++;
        out.result = WALK_PAGE_FAULT;
        return out;
    }
    if (is_write && !pte->writable) {
        stats->protection_faults++;
        out.result = WALK_PROTECTION;
        return out;
    }
    pte->referenced = true;
    if (is_write) pte->dirty = true;
    out.physical_address = (pte->pfn << PAGE_SHIFT) | offset;
    out.result = WALK_OK;
    return out;
}
```
The three-level structure is essentially a **trie** (prefix tree) with fixed depth. Each level consumes some bits of the virtual address to index into the next level.
---
## Hardware Soul: The Real Cost of Multiple Levels
Let's trace what happens at the hardware level when we add more indirection.
### Level 1: Application (Software)
```c
int value = *ptr;  // ptr = 0x00401234
```
### Level 2: Memory Management Unit (Hardware)
For a two-level page table, the MMU must:
1. **Extract directory index** (bits 31-22): `0x00401234 >> 22 = 0x1`
2. **Read PDE from memory**: Access `page_directory[0x1]`
   - This is a **memory access** (unless cached)
3. **Extract table index** (bits 21-12): `(0x00401234 >> 12) & 0x3FF = 0x1`
4. **Read PTE from memory**: Access `page_table[pde.table_pfn][0x1]`
   - This is another **memory access**
5. **Compose physical address**: `(pte.pfn << 12) | offset`
6. **Access the data**: Read from physical address
   - Third **memory access**
**A single load instruction requires three memory accesses!**
### Level 3: Cache Hierarchy
If each of these accesses hits in L1 cache (4-8 cycles each), the translation overhead is manageable. But if they miss:
| Access | L1 Hit | L1 Miss, L2 Hit | L1/L2 Miss, RAM |
|--------|--------|-----------------|-----------------|
| PDE read | 4 cycles | 20 cycles | 150 cycles |
| PTE read | 4 cycles | 20 cycles | 150 cycles |
| Data read | 4 cycles | 20 cycles | 150 cycles |
| **Total (hit)** | **12 cycles** | | |
| **Total (worst)** | | | **450 cycles** |
This is why the **TLB is critical** for multi-level page tables. A TLB hit collapses all the levels into a single lookup. Without the TLB, a four-level page table walk (x86-64) could require four memory accesses before we even touch the data.
---
## The Complete Multi-Level Simulator
Let's put it all together:
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
// Configuration
#define PAGE_SHIFT          12
#define PAGE_SIZE           4096
#define PAGE_MASK           0xFFF
#define DIR_SHIFT           22
#define DIR_MASK            0x3FF
#define DIR_ENTRIES         1024
#define TABLE_SHIFT         12
#define TABLE_MASK          0x3FF
#define TABLE_ENTRIES       1024
// Page Directory Entry
typedef struct {
    uint32_t page_table_pfn;   // Which page table (by frame number)
    bool present;
    bool writable;
    bool user_accessible;
} pde_t;
// Page Table Entry (same as Milestone 1)
typedef struct {
    uint32_t pfn;
    bool valid;
    bool readable;
    bool writable;
    bool dirty;
    bool referenced;
} pte_t;
// Physical Memory
typedef struct {
    uint8_t **frames;          // Frame data
    pte_t **page_tables;       // Page tables stored in frames
    pde_t **page_directories;  // Page directories stored in frames
    uint32_t total_frames;
    uint32_t *free_list;
    uint32_t free_count;
    uint32_t next_page_table_frame;
    uint32_t next_directory_frame;
} physical_memory_t;
// Process
typedef struct {
    uint32_t pid;
    uint32_t cr3;              // Index of page directory in phys_mem
    char name[32];
} process_t;
// Statistics
typedef struct {
    uint64_t total_accesses;
    uint64_t page_faults;
    uint64_t protection_faults;
    uint64_t dir_not_present;
    uint64_t tables_allocated;
    uint64_t pages_allocated;
} stats_t;
// Simulator State
typedef struct {
    physical_memory_t phys_mem;
    process_t *processes;
    uint32_t num_processes;
    uint32_t current_process;
    uint32_t cr3;              // Current CR3 value
    stats_t stats;
} simulator_t;
```
### Initialization
```c
simulator_t* simulator_create(uint32_t num_frames, uint32_t max_processes) {
    simulator_t *sim = calloc(1, sizeof(simulator_t));
    if (!sim) return NULL;
    // Physical memory setup
    sim->phys_mem.total_frames = num_frames;
    sim->phys_mem.frames = calloc(num_frames, sizeof(uint8_t*));
    sim->phys_mem.page_tables = calloc(num_frames, sizeof(pte_t*));
    sim->phys_mem.page_directories = calloc(num_frames, sizeof(pde_t*));
    sim->phys_mem.free_list = calloc(num_frames, sizeof(uint32_t));
    sim->phys_mem.free_count = num_frames;
    for (uint32_t i = 0; i < num_frames; i++) {
        sim->phys_mem.free_list[i] = i;
    }
    // Process table
    sim->processes = calloc(max_processes, sizeof(process_t));
    sim->num_processes = 0;
    return sim;
}
uint32_t allocate_frame(simulator_t *sim) {
    if (sim->phys_mem.free_count == 0) {
        return (uint32_t)-1;
    }
    return sim->phys_mem.free_list[--sim->phys_mem.free_count];
}
void free_frame(simulator_t *sim, uint32_t pfn) {
    sim->phys_mem.free_list[sim->phys_mem.free_count++] = pfn;
}
// Allocate a page directory
uint32_t allocate_page_directory(simulator_t *sim) {
    uint32_t pfn = allocate_frame(sim);
    if (pfn == (uint32_t)-1) return pfn;
    pde_t *dir = calloc(DIR_ENTRIES, sizeof(pde_t));
    for (uint32_t i = 0; i < DIR_ENTRIES; i++) {
        dir[i].present = false;
    }
    sim->phys_mem.page_directories[pfn] = dir;
    return pfn;
}
// Allocate a page table
uint32_t allocate_page_table(simulator_t *sim, pde_t *directory, uint32_t dir_index) {
    if (directory[dir_index].present) {
        return directory[dir_index].page_table_pfn;
    }
    uint32_t pfn = allocate_frame(sim);
    if (pfn == (uint32_t)-1) return pfn;
    pte_t *table = calloc(TABLE_ENTRIES, sizeof(pte_t));
    for (uint32_t i = 0; i < TABLE_ENTRIES; i++) {
        table[i].valid = false;
    }
    sim->phys_mem.page_tables[pfn] = table;
    directory[dir_index].page_table_pfn = pfn;
    directory[dir_index].present = true;
    directory[dir_index].writable = true;
    directory[dir_index].user_accessible = true;
    sim->stats.tables_allocated++;
    return pfn;
}
// Create a new process
process_t* process_create(simulator_t *sim, const char *name) {
    process_t *proc = &sim->processes[sim->num_processes];
    proc->pid = sim->num_processes;
    proc->cr3 = allocate_page_directory(sim);
    strncpy(proc->name, name, sizeof(proc->name) - 1);
    sim->num_processes++;
    return proc;
}
```
### Translation with Demand Allocation
```c
typedef enum {
    TRANS_OK,
    TRANS_PAGE_FAULT,
    TRANS_DIR_NOT_PRESENT,
    TRANS_PROTECTION_FAULT
} trans_result_t;
typedef struct {
    trans_result_t result;
    uint32_t physical_address;
} trans_output_t;
trans_output_t translate_multi_level(
    simulator_t *sim,
    uint32_t va,
    bool is_write
) {
    trans_output_t out = {0};
    sim->stats.total_accesses++;
    uint32_t dir_index = (va >> DIR_SHIFT) & DIR_MASK;
    uint32_t table_index = (va >> TABLE_SHIFT) & TABLE_MASK;
    uint32_t offset = va & PAGE_MASK;
    // Get current page directory
    pde_t *directory = sim->phys_mem.page_directories[sim->cr3];
    if (!directory) {
        sim->stats.protection_faults++;
        out.result = TRANS_PROTECTION_FAULT;
        return out;
    }
    pde_t *pde = &directory[dir_index];
    // Level 1: Check directory entry
    if (!pde->present) {
        // Try to allocate a page table on demand
        uint32_t pt_pfn = allocate_page_table(sim, directory, dir_index);
        if (pt_pfn == (uint32_t)-1) {
            sim->stats.dir_not_present++;
            out.result = TRANS_DIR_NOT_PRESENT;
            return out;
        }
    }
    // Check directory-level write permission
    if (is_write && !pde->writable) {
        sim->stats.protection_faults++;
        out.result = TRANS_PROTECTION_FAULT;
        return out;
    }
    // Level 2: Check page table
    pte_t *table = sim->phys_mem.page_tables[pde->page_table_pfn];
    pte_t *pte = &table[table_index];
    if (!pte->valid) {
        // Demand paging: allocate a frame
        uint32_t frame_pfn = allocate_frame(sim);
        if (frame_pfn == (uint32_t)-1) {
            sim->stats.page_faults++;
            out.result = TRANS_PAGE_FAULT;
            return out;
        }
        // Allocate frame data
        sim->phys_mem.frames[frame_pfn] = calloc(PAGE_SIZE, 1);
        pte->pfn = frame_pfn;
        pte->valid = true;
        pte->readable = true;
        pte->writable = true;
        pte->dirty = false;
        pte->referenced = false;
        sim->stats.pages_allocated++;
        printf("[Page Fault] VA 0x%08X -> PFN %u\n", va, frame_pfn);
    }
    // Check page-level permissions
    if (is_write && !pte->writable) {
        sim->stats.protection_faults++;
        out.result = TRANS_PROTECTION_FAULT;
        return out;
    }
    // Update access bits
    pte->referenced = true;
    if (is_write) {
        pte->dirty = true;
    }
    // Compose physical address
    out.physical_address = (pte->pfn << PAGE_SHIFT) | offset;
    out.result = TRANS_OK;
    return out;
}
```
### Context Switch
```c
void context_switch(simulator_t *sim, uint32_t pid) {
    if (pid >= sim->num_processes) {
        fprintf(stderr, "Invalid PID %u\n", pid);
        return;
    }
    process_t *proc = &sim->processes[pid];
    sim->current_process = pid;
    sim->cr3 = proc->cr3;
    printf("[Context Switch] Now running '%s' (PID %u, CR3=0x%X)\n",
           proc->name, proc->pid, proc->cr3);
}
```
### Memory Overhead Measurement
```c
void measure_and_print_overhead(simulator_t *sim) {
    uint64_t directory_bytes = DIR_ENTRIES * sizeof(pde_t);
    uint64_t table_bytes = 0;
    uint32_t tables_used = 0;
    // Count tables for current process
    pde_t *directory = sim->phys_mem.page_directories[sim->cr3];
    for (uint32_t i = 0; i < DIR_ENTRIES; i++) {
        if (directory[i].present) {
            table_bytes += TABLE_ENTRIES * sizeof(pte_t);
            tables_used++;
        }
    }
    uint64_t multi_level_total = directory_bytes + table_bytes;
    uint64_t flat_total = DIR_ENTRIES * TABLE_ENTRIES * sizeof(pte_t);
    printf("\n");
    printf("========================================\n");
    printf("      MEMORY OVERHEAD COMPARISON        \n");
    printf("========================================\n");
    printf("Process: %s (PID %u)\n", 
           sim->processes[sim->current_process].name,
           sim->current_process);
    printf("----------------------------------------\n");
    printf("Flat page table:        %lu bytes (%.2f MB)\n",
           flat_total, flat_total / (1024.0 * 1024.0));
    printf("Multi-level total:      %lu bytes (%.2f KB)\n",
           multi_level_total, multi_level_total / 1024.0);
    printf("  - Page directory:     %lu bytes\n", directory_bytes);
    printf("  - Page tables:        %lu bytes (%u tables)\n",
           table_bytes, tables_used);
    printf("----------------------------------------\n");
    printf("Space savings:          %.2fx (%.1f%%)\n",
           (double)flat_total / multi_level_total,
           100.0 * (1.0 - (double)multi_level_total / flat_total));
    printf("========================================\n");
}
```
### Statistics
```c
void print_statistics(simulator_t *sim) {
    printf("\n");
    printf("========================================\n");
    printf("         SIMULATION STATISTICS          \n");
    printf("========================================\n");
    printf("Total accesses:         %lu\n", sim->stats.total_accesses);
    printf("Page faults:            %lu\n", sim->stats.page_faults);
    printf("Dir not present:        %lu\n", sim->stats.dir_not_present);
    printf("Protection faults:      %lu\n", sim->stats.protection_faults);
    printf("Tables allocated:       %lu\n", sim->stats.tables_allocated);
    printf("Pages allocated:        %lu\n", sim->stats.pages_allocated);
    printf("Frames used:            %u / %u\n",
           sim->phys_mem.total_frames - sim->phys_mem.free_count,
           sim->phys_mem.total_frames);
    printf("========================================\n");
}
```
---
## Testing Multi-Level Tables
### Test Case 1: Sparse Address Space
```c
void test_sparse_address_space(void) {
    printf("=== Test: Sparse Address Space ===\n\n");
    simulator_t *sim = simulator_create(256, 4);
    process_t *proc = process_create(sim, "sparse_test");
    context_switch(sim, proc->pid);
    // Access widely scattered addresses
    uint32_t addresses[] = {
        0x00400000,  // Code (directory 1)
        0x01000000,  // Heap (directory 4)
        0x10000000,  // Shared lib (directory 64)
        0x7FFF0000,  // Stack (directory 511)
    };
    for (int i = 0; i < 4; i++) {
        trans_output_t result = translate_multi_level(sim, addresses[i], false);
        printf("0x%08X -> 0x%08X (result=%d)\n",
               addresses[i], result.physical_address, result.result);
    }
    measure_and_print_overhead(sim);
    // Expected: 4 page tables allocated, ~20 KB total vs 4 MB flat
    // Cleanup handled by simulator_destroy
}
```
### Test Case 2: Multiple Processes
```c
void test_multiple_processes(void) {
    printf("=== Test: Multiple Processes ===\n\n");
    simulator_t *sim = simulator_create(256, 4);
    // Create two processes
    process_t *proc_a = process_create(sim, "process_A");
    process_t *proc_b = process_create(sim, "process_B");
    // Process A accesses 0x00400000
    context_switch(sim, proc_a->pid);
    trans_output_t result = translate_multi_level(sim, 0x00400000, true);
    printf("Process A: 0x00400000 -> PFN %u\n", result.physical_address >> 12);
    // Process B accesses same virtual address
    context_switch(sim, proc_b->pid);
    result = translate_multi_level(sim, 0x00400000, true);
    printf("Process B: 0x00400000 -> PFN %u\n", result.physical_address >> 12);
    // Different CR3 = different page directory = different physical address!
    // The same virtual address in different processes maps to different frames.
    // Switch back to A
    context_switch(sim, proc_a->pid);
    result = translate_multi_level(sim, 0x00400000, false);
    printf("Process A (again): 0x00400000 -> PFN %u (same as before)\n",
           result.physical_address >> 12);
}
```
### Test Case 3: Bit Extraction Verification
```c
void test_bit_extraction(void) {
    printf("=== Test: Bit Extraction ===\n\n");
    uint32_t test_addr = 0x00401234;
    uint32_t dir_index = (test_addr >> DIR_SHIFT) & DIR_MASK;
    uint32_t table_index = (test_addr >> TABLE_SHIFT) & TABLE_MASK;
    uint32_t offset = test_addr & PAGE_MASK;
    printf("Address: 0x%08X\n", test_addr);
    printf("  Directory index: %u (0x%X)\n", dir_index, dir_index);
    printf("  Table index:     %u (0x%X)\n", table_index, table_index);
    printf("  Offset:          %u (0x%X)\n", offset, offset);
    // Expected:
    // dir_index = 1 (0x00400000 >> 22 = 1)
    // table_index = 1 (0x401234 >> 12 & 0x3FF = 0x1)
    // offset = 0x234
    assert(dir_index == 1);
    assert(table_index == 1);
    assert(offset == 0x234);
    printf("All assertions passed!\n");
}
```
---
## Common Pitfalls
### Pitfall 1: Index Extraction Order
**The mistake**: Confusing which bits go where.
```c
// WRONG! Table index uses the wrong bits
uint32_t table_index = (va >> 10) & 0x3FF;  // Should be >> 12
```
**Why it's wrong**: The offset is 12 bits (0-11). Table index starts at bit 12, not bit 10.
**The fix**:
```c
uint32_t table_index = (va >> TABLE_SHIFT) & TABLE_MASK;  // TABLE_SHIFT = 12
```
### Pitfall 2: Confusing PDE and PTE
**The mistake**: Treating PDEs and PTEs as interchangeable.
```c
// WRONG! PDE points to a page table, not a frame
uint32_t physical_addr = (pde->page_table_pfn << 12) | offset;
```
**Why it's wrong**: The PDE's PFN field points to a *page table*, which is a data structure, not user data. You need to complete the walk to get the actual frame.
**The mental model**:
- **PDE**: "Where is the page table for this region?"
- **PTE**: "Where is the actual data frame?"
### Pitfall 3: Forgetting to Allocate Page Tables
**The mistake**: Only allocating frames, not page tables.
```c
// WRONG! Assumes page table already exists
pte_t *pte = &page_table[table_index];
```
**Why it's wrong**: The page table might not exist if this is the first access to that region. The PDE could have present=0.
**The fix**: Check and allocate on demand:
```c
if (!pde->present) {
    allocate_page_table(sim, directory, dir_index);
}
```
### Pitfall 4: Not Accounting for Table Memory
**The mistake**: Only counting user data frames, not page table structures.
**Why it's wrong**: Page directories and page tables consume physical memory too. A process with many scattered allocations might use more memory for page tables than for data.
**The fix**: Track both:
```c
uint64_t total_memory = user_data_bytes + page_table_bytes + directory_bytes;
```
---
## Design Decisions: Why This, Not That
| Option | Pros | Cons | Used By |
|--------|------|------|---------|
| **Two-Level (32-bit) ✓** | Simple, good for 32-bit spaces | Doesn't scale to 64-bit | x86 PAE, early 32-bit Unix |
| Four-Level (64-bit) | Handles full 64-bit space | More memory accesses per walk | x86-64, ARM64 |
| Five-Level (64-bit) | Extends addressable space | Even more indirection | Recent x86-64 (LA57) |
| Hashed Page Table | Memory-efficient, O(1) expected | Collision handling, poor locality | Early Linux PAE, PowerPC |
| Inverted Page Table | Single table for all processes | O(n) search, complex sharing | IA-64, some embedded |
| Allocation Strategy | Pros | Cons | Used By |
|---------------------|------|------|---------|
| **On-Demand ✓** | Minimal memory usage | Allocation latency on first access | All modern OSes |
| Eager (all tables) | Predictable latency | Wastes memory for sparse spaces | Rarely used |
| Preallocated + Swap | Fixed memory budget | Complex, less flexible | Some embedded systems |
---
## What You've Built
At this point, your simulator implements:
1. **Two-level hierarchical page tables** with page directories and page tables
2. **CR3 register simulation** for per-process address space roots
3. **On-demand allocation** of second-level tables only when regions are accessed
4. **Memory overhead measurement** comparing multi-level to flat structures
5. **Context switch** that changes CR3 and switches to a different address space
6. **Bit extraction** correctly decomposing virtual addresses into directory, table, and offset indices
You've now implemented the same fundamental structure that x86-64 and ARM64 use, just with fewer levels. The principle—hierarchical indirection to compress sparse address spaces—is identical.
---
## Knowledge Cascade: Where This Leads
### Same Domain Connections
**→ TLB Importance (Milestone 2)**: With multi-level tables, the TLB becomes even more critical. A four-level walk (x86-64) requires four memory accesses. A TLB hit collapses all four into one lookup. This is why TLB coverage—how much address space a TLB can cache—matters more than raw entry count.
**→ Page Replacement (Milestone 4)**: The referenced and dirty bits in your PTEs are exactly what page replacement algorithms need. When physical memory fills up and the OS must evict pages, it uses these bits to make intelligent decisions. Your multi-level structure makes it easy to find all pages belonging to a process (walk its tables).
**→ Huge Pages**: Modern systems support 2 MB or 1 GB pages. A 2 MB page is mapped by a PDE directly (no second-level table needed). This eliminates one level of indirection and increases TLB coverage. The tradeoff: internal fragmentation. Your multi-level structure can be extended to support this by adding a "large page" flag to PDEs.
### Cross-Domain Connections
**→ B-Trees in Databases**: A multi-level page table is essentially a B-tree optimized for a specific key distribution (bit prefixes). The fan-out (1024 for directory → table), height (2-5 levels), and the principle of only allocating nodes for populated keys—all identical. Database index design uses the same math you just applied to page tables.

![Page Directory Entry Structure](./diagrams/tdd-diag-m3-02.svg)

> **🔑 Foundation: Tries: prefix trees that use key bits/characters for navigation**
> 
> ## What It Is
A **trie** (pronounced "try") is a tree data structure where each node represents a prefix of a key, and the path from root to node spells out the complete key. Unlike binary search trees that compare entire keys at each node, tries navigate one bit (or character) at a time.
**Bit-indexed tries** use individual bits of a key to decide left/right navigation:
- Bit = 0 → go left
- Bit = 1 → go right
```
Key: 0b1010 (decimal 10)
         root
        /    \
      0        1    ← bit 3 (MSB)
              /
            0        ← bit 2
              \
                1    ← bit 1
              /
            0        ← bit 0 → VALUE stored here
```
## Why You Need It Right Now
For network packet processing, tries enable **longest prefix matching** — the core operation for IP route lookups. When a packet arrives with destination 192.168.1.42, the router must find the most specific matching route (e.g., 192.168.1.0/24 beats 192.168.0.0/16).
Tries make this O(k) where k = prefix length (typically 32 for IPv4), regardless of how many routes exist. Hash tables can't do prefix matching; sorted arrays require O(log n) but can't handle variable-length prefixes elegantly.
## Key Insight
**Think of a trie as a "decision compressor."** Each level answers one binary question about the key. The genius is that you stop as soon as you've distinguished your key from all others — common prefixes share nodes, saving memory and comparison work.
In networking specifically: the trie depth equals the prefix length you're matching, not the number of routes in your table. A /24 route requires 24 decisions whether you have 100 routes or 100,000 routes.

![Per-Process Address Space Isolation](./diagrams/tdd-diag-m3-10.svg)

**→ Tries and Radix Trees**: Your page directory → page table → PTE structure is exactly a fixed-depth trie. Each level consumes some bits of the key (virtual address) to index into the next level. This is why multi-level page tables are sometimes called "radix trees" in OS literature.
**→ JVM Compressed OOPs**: The JVM uses a trick called "Compressed Ordinary Object Pointers" to use 32-bit pointers in a 64-bit JVM. This works because the OS places the heap in the low 32 GB of address space, which can be addressed with 32-bit offsets from a base. Multi-level page tables make this cheap—the sparse upper address space costs nothing.
**→ Filesystem Directory Structures**: When you resolve `/home/user/docs/file.txt`, the filesystem does exactly what your page table walk does:
1. Read root directory → find "home"
2. Read /home → find "user"
3. Read /home/user → find "docs"
4. Read /home/user/docs → find "file.txt"
5. Read the file's inode
Each path component is an index into the next level. The parallel is exact.
**→ Virtualization: Nested Page Tables**: Running a VM adds another level of translation. The guest OS thinks it's managing physical memory, but those "physical" addresses are actually virtual to the host. So the full walk becomes:
- Guest virtual → Guest physical (guest page table walk)
- Guest physical → Host physical (host page table walk)
This is called "two-dimensional paging" and can require 24+ memory accesses for a single translation without TLBs. Hardware-assisted nested paging (AMD RVI, Intel EPT) adds dedicated TLBs for the second level.
### Historical Context
**→ Segmentation vs. Paging**: Before paging became dominant, x86 used segmentation—base + limit addressing where each segment had a starting address and size. Segmentation had external fragmentation (holes in memory) and couldn't support sparse address spaces efficiently. Paging won because fixed-size frames eliminate external fragmentation, and the hierarchical table structure handles sparsity.
**→ The 640 KB Limit**: Early PCs had a 640 KB "conventional memory" limit because the 8086 used 20-bit addresses (1 MB total), and IBM reserved the upper 384 KB for ROM and memory-mapped I/O. The 80386's 32-bit paging with two-level tables finally provided a clean solution—each process could have its own address space mapped wherever convenient in physical memory.
**→ PAE (Physical Address Extension)**: The Pentium Pro introduced 36-bit physical addresses (up to 64 GB RAM), but 32-bit virtual addresses. This required a three-level page table (2+9+9+12) to map the larger physical space. Your three-level stretch goal uses the same structure.
---
## Looking Ahead
In Milestone 4, you'll implement page replacement and swap simulation. This reveals:
- What happens when `allocate_frame()` returns -1 (out of memory)
- How the referenced bit enables LRU-like algorithms
- Why dirty pages must be written to swap before eviction
- The phenomenon of thrashing when working set exceeds physical memory
The multi-level page table you built today is the data structure that replacement algorithms traverse to find eviction candidates. Your `measure_overhead()` function will help demonstrate why thrashing is so devastating: when the OS spends all its time evicting and reloading pages, the page table structure itself becomes a source of overhead.
[[CRITERIA_JSON: {"milestone_id": "virtual-memory-sim-m3", "criteria": ["CR3 register simulation holds physical address (frame index) of the root page directory for the active process", "Virtual address decomposition extracts directory index (bits 31-22), table index (bits 21-12), and offset (bits 11-0) for two-level 32-bit page tables", "Page directory entry (PDE) structure contains page_table_pfn (uint32_t), present (bool), writable (bool), and user_accessible (bool) fields", "PDE with present=false indicates entire 4 MB region is unmapped; no second-level table allocated for that directory entry", "Second-level page tables (1024 PTEs each) are allocated on demand when first access to a directory's region occurs", "Page table walk traverses CR3 → page_directory[dir_index] → page_table[table_index] → PTE → physical_address", "Memory overhead measurement function calculates total bytes for page directory plus all allocated second-level tables", "Memory overhead comparison demonstrates multi-level tables use significantly less memory than flat tables for sparse address spaces (report savings factor and percentage)", "Context switch function updates CR3 to point to new process's page directory and logs the switch", "TLB flush or ASID switch occurs on context switch (integrates with Milestone 2 TLB implementation)", "Three-level page table stretch goal implements 2+9+9+12 bit split with PML4, PDPT, and page table levels", "Statistics tracked include tables_allocated (uint64_t) in addition to page faults and protection faults from earlier milestones", "Translation function handles three distinct failure cases: directory not present, page not valid, and protection violation"]}]
<!-- END_MS -->


<!-- MS_ID: virtual-memory-sim-m4 -->
# Milestone 4: Page Replacement and Swap Simulation
## The Lie About "Out of Memory"
You've probably seen this error before:
```
Out of memory
```
And you thought: "My program needs more RAM than the machine has. Time to buy more memory or optimize."
**This is almost never what's happening.**
When your program "runs out of memory," the machine almost always has plenty of free RAM. What actually happened is subtler—and far more interesting.

![Swap Mapping Structure](./diagrams/tdd-diag-m4-18.svg)

![Physical Memory Frame Pool Management](./diagrams/diag-physical-memory-frame-pool.svg)

Modern operating systems practice **memory overcommitment**: they promise more memory to processes than physically exists. When you call `malloc(1GB)` on a machine with 4GB RAM, the OS says "sure, here's your 1GB"—even though it doesn't have 1GB to give you. It's betting you won't actually touch all those pages.
Most of the time, the OS wins this bet. Programs allocate far more than they use. But sometimes—when you actually do touch all those pages—the OS must make a hard choice:
> Which pages stay in precious RAM, and which get evicted to disk?
This choice is made by a **page replacement algorithm**, and the algorithm you choose can matter more than the amount of RAM you have. Here's the truly mind-bending part: **with some algorithms, adding MORE memory can make performance WORSE.**
This isn't theoretical. It's called **Bélády's anomaly**, and you're about to implement it, witness it, and understand why it happens.
---
## The Fundamental Tension: Infinite Demand, Finite Supply
Every process wants infinite memory. Your web browser would love to cache every webpage you've ever visited. Your database wants to keep all indexes in RAM. Your ML training wants the entire dataset loaded at once.
Hardware provides a fixed amount of physical frames. A machine with 16 GB RAM has about 4 million frames of 4 KB each. That's it. No more.

![Replacement Algorithm Comparison Architecture](./diagrams/tdd-diag-m4-14.svg)

![Thrashing: When Working Set Exceeds Memory](./diagrams/diag-thrashing-detection.svg)

![Thrashing Detection Flow](./diagrams/tdd-diag-m4-13.svg)

The OS sits between infinite demand and finite supply, constantly making triage decisions:
1. **Page fault occurs**: Process A needs page X, but all frames are occupied
2. **Find a victim**: Which page should we evict to make room?
3. **Write back if dirty**: If the victim was modified, write it to swap
4. **Load the new page**: Read the faulting page from swap (or zero-fill if new)
5. **Update page tables**: Point the PTE to the new frame
6. **Resume the process**: The faulting instruction retries
The **page replacement algorithm** is step 2. Everything else is mechanics. But step 2 determines whether your system runs smoothly or grinds to a halt.
### The Cost Model
Why does the algorithm matter? Because the costs are asymmetric by six orders of magnitude:
| Operation | Latency | Relative Cost |
|-----------|---------|---------------|
| Access page in RAM | 100 ns | 1× |
| TLB miss (page table walk) | 50-200 ns | 1-2× |
| Page fault (load from SSD) | 50-100 μs | 500-1,000× |
| Page fault (load from HDD) | 5-10 ms | 50,000-100,000× |
| Page fault (write + read) | 10-20 ms | 100,000-200,000× |
A single wrong eviction decision can cost **100,000×** more than a good one. If your algorithm evicts a page that will be needed 10 instructions from now, you've just wasted milliseconds. Do this repeatedly, and your "fast" program spends 99% of its time waiting for disk.
This is **thrashing**: when the working set (pages actively being used) exceeds physical memory, and the system spends all its time evicting and reloading pages rather than doing useful work. The computer doesn't crash—it just runs 100× slower.
---
## Swap Space: The Overflow Parking Lot
Before we can evict pages, we need somewhere to put them. That's **swap space**: a reserved area on disk (or a dedicated partition) where evicted pages are stored.

![Page Reload from Swap](./diagrams/tdd-diag-m4-09.svg)

![Simulated Swap Space Organization](./diagrams/diag-swap-space-layout.svg)

Your simulator will model swap as a simple array or file:
```c
// Swap slot: stores the data for one evicted page
typedef struct {
    uint8_t data[PAGE_SIZE];    // The actual page contents
    uint32_t vpn;               // Which virtual page this was
    uint32_t owning_process;    // Which process owns it
    bool occupied;              // Is this slot in use?
} swap_slot_t;
typedef struct {
    swap_slot_t *slots;
    uint32_t total_slots;
    uint32_t *free_list;
    uint32_t free_count;
} swap_space_t;
swap_space_t* swap_create(uint32_t num_slots) {
    swap_space_t *swap = calloc(1, sizeof(swap_space_t));
    swap->slots = calloc(num_slots, sizeof(swap_slot_t));
    swap->free_list = calloc(num_slots, sizeof(uint32_t));
    swap->total_slots = num_slots;
    swap->free_count = num_slots;
    for (uint32_t i = 0; i < num_slots; i++) {
        swap->slots[i].occupied = false;
        swap->free_list[i] = i;
    }
    return swap;
}
// Allocate a swap slot for an evicted page
int32_t swap_alloc(swap_space_t *swap) {
    if (swap->free_count == 0) {
        return -1;  // Swap full!
    }
    return swap->free_list[--swap->free_count];
}
void swap_free(swap_space_t *swap, uint32_t slot_index) {
    swap->slots[slot_index].occupied = false;
    swap->free_list[swap->free_count++] = slot_index;
}
```
### The Write-Back Path
When evicting a page, we must check the **dirty bit**:
```c
typedef struct {
    uint32_t swap_slot_index;   // Where in swap this page resides
    bool in_swap;               // Is there a copy in swap?
} swap_mapping_t;
void evict_to_swap(
    simulator_t *sim,
    uint32_t vpn,
    pte_t *pte,
    swap_space_t *swap,
    swap_mapping_t *swap_map
) {
    // Allocate a swap slot
    int32_t slot = swap_alloc(swap);
    if (slot < 0) {
        fprintf(stderr, "FATAL: Swap space exhausted!\n");
        exit(1);
    }
    // If the page is dirty, write it to swap
    if (pte->dirty) {
        uint8_t *frame_data = sim->phys_mem.frames[pte->pfn];
        memcpy(swap->slots[slot].data, frame_data, PAGE_SIZE);
        sim->stats.swap_writes++;
        printf("[Swap Write] VPN %u -> Slot %d (dirty)\n", vpn, slot);
    } else {
        // Clean page: swap already has the correct data (or it's all zeros)
        // No need to write anything
        sim->stats.swap_discards++;
        printf("[Swap Discard] VPN %u (clean, no write needed)\n", vpn);
    }
    // Record where this page now lives in swap
    swap_map[vpn].swap_slot_index = slot;
    swap_map[vpn].in_swap = true;
    swap->slots[slot].vpn = vpn;
    swap->slots[slot].occupied = true;
    // Invalidate the page table entry
    pte->valid = false;
    pte->pfn = 0;
}
```

![Swap Space Organization](./diagrams/tdd-diag-m4-03.svg)

![Dirty Page Write-Back Path](./diagrams/diag-dirty-page-writeback.svg)

![Complete Eviction Sequence](./diagrams/tdd-diag-m4-10.svg)

**The critical insight**: clean pages can be discarded without writing to swap because their data already exists somewhere (the executable file, memory-mapped file, or was never modified). Only dirty pages require disk writes. This is why the dirty bit matters—it can save 50% of swap I/O.
---
## The Replacement Algorithms: Four Philosophies
Now the core question: when all frames are full and a page fault occurs, **which page do we evict?**
Different algorithms answer this differently. You'll implement all four and compare them.
### FIFO: First In, First Out
**Philosophy**: Evict the page that has been in memory the longest.
```c
typedef struct {
    uint32_t *queue;        // Circular buffer of VPNs in load order
    uint32_t capacity;
    uint32_t head;          // Oldest entry
    uint32_t tail;          // Next insertion point
    uint32_t count;
} fifo_replacer_t;
void fifo_init(fifo_replacer_t *fifo, uint32_t capacity) {
    fifo->queue = calloc(capacity, sizeof(uint32_t));
    fifo->capacity = capacity;
    fifo->head = 0;
    fifo->tail = 0;
    fifo->count = 0;
}
void fifo_insert(fifo_replacer_t *fifo, uint32_t vpn) {
    fifo->queue[fifo->tail] = vpn;
    fifo->tail = (fifo->tail + 1) % fifo->capacity;
    fifo->count++;
}
uint32_t fifo_evict(fifo_replacer_t *fifo) {
    uint32_t victim = fifo->queue[fifo->head];
    fifo->head = (fifo->head + 1) % fifo->capacity;
    fifo->count--;
    return victim;
}
```

![FIFO Queue State Evolution](./diagrams/tdd-diag-m4-04.svg)

![FIFO Replacement: Queue Evolution](./diagrams/diag-fifo-queue-state.svg)

![Statistics State Transitions](./diagrams/tdd-diag-m4-20.svg)

**Appeal**: Simple, fair, requires almost no metadata.
**Problem**: A page loaded 5 minutes ago but accessed every microsecond will be evicted before a page loaded 1 minute ago but never touched. FIFO ignores recency entirely.
**Worse problem**: Bélády's anomaly—adding frames can *increase* page faults.
### LRU: Least Recently Used
**Philosophy**: Evict the page that hasn't been accessed for the longest time.
```c
typedef struct {
    uint32_t vpn;
    uint64_t last_access_time;
} lru_entry_t;
typedef struct {
    lru_entry_t *entries;
    uint32_t capacity;
    uint32_t count;
    uint64_t clock;         // Monotonic timestamp
} lru_replacer_t;
void lru_init(lru_replacer_t *lru, uint32_t capacity) {
    lru->entries = calloc(capacity, sizeof(lru_entry_t));
    lru->capacity = capacity;
    lru->count = 0;
    lru->clock = 0;
    for (uint32_t i = 0; i < capacity; i++) {
        lru->entries[i].vpn = UINT32_MAX;
    }
}
void lru_access(lru_replacer_t *lru, uint32_t vpn) {
    lru->clock++;
    // Find the entry and update its timestamp
    for (uint32_t i = 0; i < lru->count; i++) {
        if (lru->entries[i].vpn == vpn) {
            lru->entries[i].last_access_time = lru->clock;
            return;
        }
    }
}
void lru_insert(lru_replacer_t *lru, uint32_t vpn) {
    if (lru->count < lru->capacity) {
        lru->entries[lru->count].vpn = vpn;
        lru->entries[lru->count].last_access_time = ++lru->clock;
        lru->count++;
    }
}
uint32_t lru_evict(lru_replacer_t *lru) {
    // Find the entry with oldest access time
    uint64_t oldest = UINT64_MAX;
    uint32_t victim_idx = 0;
    for (uint32_t i = 0; i < lru->count; i++) {
        if (lru->entries[i].last_access_time < oldest) {
            oldest = lru->entries[i].last_access_time;
            victim_idx = i;
        }
    }
    uint32_t victim_vpn = lru->entries[victim_idx].vpn;
    // Remove by swapping with last element
    lru->entries[victim_idx] = lru->entries[--lru->count];
    return victim_vpn;
}
```

![LRU Stack Evolution](./diagrams/tdd-diag-m4-05.svg)

![LRU Replacement: Stack/List Evolution](./diagrams/diag-lru-stack.svg)

**Appeal**: Uses actual access patterns. Pages accessed recently are likely to be accessed again soon (the principle that recently accessed items are likely to be accessed again).
**Problem**: Requires tracking access time for every page. On every memory access, you must update a timestamp. Hardware doesn't do this—the referenced bit is a coarse approximation.
**Implementation note**: The O(n) scan above is simplified. Real systems use doubly-linked lists (move accessed page to head, evict from tail) for O(1) operations.
### Clock: The Hardware-Friendly Approximation
**Philosophy**: Approximate LRU using the hardware's referenced bit, without tracking exact timestamps.
The Clock algorithm (also called "Second Chance") treats all pages as arranged in a circular buffer. A "clock hand" pointer sweeps through them:
1. If the page's referenced bit is 1, clear it and move on (give it a "second chance")
2. If the referenced bit is 0, evict this page
3. Wrap around when reaching the end
```c
typedef struct {
    uint32_t *vpn_list;     // Pages in the circular buffer
    bool *referenced;       // Mirrors of the PTE referenced bits
    uint32_t capacity;
    uint32_t count;
    uint32_t hand;          // Clock hand position
} clock_replacer_t;
void clock_init(clock_replacer_t *clock, uint32_t capacity) {
    clock->vpn_list = calloc(capacity, sizeof(uint32_t));
    clock->referenced = calloc(capacity, sizeof(bool));
    clock->capacity = capacity;
    clock->count = 0;
    clock->hand = 0;
}
void clock_insert(clock_replacer_t *clock, uint32_t vpn) {
    if (clock->count < clock->capacity) {
        clock->vpn_list[clock->count] = vpn;
        clock->referenced[clock->count] = true;  // Newly loaded, was just accessed
        clock->count++;
    }
}
void clock_update_referenced(clock_replacer_t *clock, uint32_t vpn, bool ref) {
    for (uint32_t i = 0; i < clock->count; i++) {
        if (clock->vpn_list[i] == vpn) {
            clock->referenced[i] = ref;
            return;
        }
    }
}
uint32_t clock_evict(clock_replacer_t *clock, pte_t *page_table) {
    // Sweep through pages, looking for one with referenced = 0
    uint32_t pages_checked = 0;
    while (pages_checked < clock->count) {
        uint32_t current_vpn = clock->vpn_list[clock->hand];
        if (!clock->referenced[clock->hand]) {
            // Found a victim!
            uint32_t victim = clock->vpn_list[clock->hand];
            // Remove by shifting remaining elements
            for (uint32_t i = clock->hand; i < clock->count - 1; i++) {
                clock->vpn_list[i] = clock->vpn_list[i + 1];
                clock->referenced[i] = clock->referenced[i + 1];
            }
            clock->count--;
            // Adjust hand if needed
            if (clock->hand >= clock->count) {
                clock->hand = 0;
            }
            return victim;
        }
        // Give this page a second chance: clear referenced and move on
        clock->referenced[clock->hand] = false;
        clock->hand = (clock->hand + 1) % clock->count;
        pages_checked++;
    }
    // All pages were referenced - evict the one at hand position
    uint32_t victim = clock->vpn_list[clock->hand];
    for (uint32_t i = clock->hand; i < clock->count - 1; i++) {
        clock->vpn_list[i] = clock->vpn_list[i + 1];
        clock->referenced[i] = clock->referenced[i + 1];
    }
    clock->count--;
    clock->hand = clock->hand % clock->count;
    return victim;
}
```

![Clock Algorithm Visualization](./diagrams/tdd-diag-m4-06.svg)

![Clock (Second-Chance) Algorithm Visualization](./diagrams/diag-clock-algorithm.svg)

**Appeal**: Uses only the referenced bit that hardware already provides. O(1) amortized—most evictions require just a few hand movements.
**Problem**: Approximation can be coarse. A page accessed 1 microsecond before the hand reaches it gets the same "second chance" as a page accessed 1 hour before.
**Real-world note**: Most operating systems use variants of Clock (Linux uses a two-handed clock, Windows uses something similar). It's the practical compromise between LRU's accuracy and FIFO's simplicity.
### Optimal (Bélády's Algorithm): The Theoretical Bound
**Philosophy**: Evict the page that will be used farthest in the future.
This algorithm requires knowing the future. It's impossible to implement in a real system—but it provides a **lower bound** on page faults. If Optimal gets 50 faults on a trace, no algorithm can do better.
```c
typedef struct {
    const memory_access_t *trace;
    size_t trace_length;
    size_t current_position;
} optimal_replacer_t;
void optimal_init(optimal_replacer_t *opt, 
                  const memory_access_t *trace, 
                  size_t trace_length) {
    opt->trace = trace;
    opt->trace_length = trace_length;
    opt->current_position = 0;
}
// Find when each page will next be accessed
int64_t find_next_access(const memory_access_t *trace, 
                         size_t trace_length, 
                         size_t current_pos, 
                         uint32_t vpn) {
    for (size_t i = current_pos; i < trace_length; i++) {
        uint32_t access_vpn = trace[i].virtual_address >> PAGE_SHIFT;
        if (access_vpn == vpn) {
            return (int64_t)i;
        }
    }
    return -1;  // Never accessed again
}
uint32_t optimal_evict(optimal_replacer_t *opt, 
                       uint32_t *loaded_pages, 
                       uint32_t num_loaded) {
    int64_t farthest = -1;
    uint32_t victim = loaded_pages[0];
    for (uint32_t i = 0; i < num_loaded; i++) {
        uint32_t vpn = loaded_pages[i];
        int64_t next_use = find_next_access(opt->trace, opt->trace_length, 
                                            opt->current_position, vpn);
        if (next_use == -1) {
            // This page is never used again - perfect victim!
            return vpn;
        }
        if (next_use > farthest) {
            farthest = next_use;
            victim = vpn;
        }
    }
    return victim;
}
```
**Appeal**: Provably optimal. Provides a benchmark for comparing other algorithms.
**Problem**: Requires future knowledge. Only usable for offline analysis with pre-recorded traces.
**Why implement it**: When testing your other algorithms, Optimal tells you how close they are to perfect. If FIFO gets 100 faults and Optimal gets 30, there's room for improvement. If LRU gets 32 faults, LRU is doing almost perfectly.
---
## Bélády's Anomaly: When More Memory Hurts
Here's the result that surprises everyone:
> With FIFO replacement, adding more frames can **increase** the number of page faults.

![Optimal Algorithm Lookahead](./diagrams/tdd-diag-m4-07.svg)

![Bélády's Anomaly: More Frames, More Faults](./diagrams/diag-belady-anomaly.svg)

Let's trace through a concrete example. Consider this access pattern:
```
1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5
```
**With 3 frames (FIFO):**
| Access | Frame 0 | Frame 1 | Frame 2 | Fault? |
|--------|---------|---------|---------|--------|
| 1      | 1       | -       | -       | Yes    |
| 2      | 1       | 2       | -       | Yes    |
| 3      | 1       | 2       | 3       | Yes    |
| 4      | 4       | 2       | 3       | Yes (evict 1) |
| 1      | 4       | 1       | 3       | Yes (evict 2) |
| 2      | 4       | 1       | 2       | Yes (evict 3) |
| 5      | 5       | 1       | 2       | Yes (evict 4) |
| 1      | 5       | 1       | 2       | No     |
| 2      | 5       | 1       | 2       | No     |
| 3      | 5       | 3       | 2       | Yes (evict 1) |
| 4      | 5       | 3       | 4       | Yes (evict 2) |
| 5      | 5       | 3       | 4       | No     |
**Total: 9 page faults**
**With 4 frames (FIFO):**
| Access | Frame 0 | Frame 1 | Frame 2 | Frame 3 | Fault? |
|--------|---------|---------|---------|---------|--------|
| 1      | 1       | -       | -       | -       | Yes    |
| 2      | 1       | 2       | -       | -       | Yes    |
| 3      | 1       | 2       | 3       | -       | Yes    |
| 4      | 1       | 2       | 3       | 4       | Yes    |
| 1      | 1       | 2       | 3       | 4       | No     |
| 2      | 1       | 2       | 3       | 4       | No     |
| 5      | 5       | 2       | 3       | 4       | Yes (evict 1) |
| 1      | 5       | 1       | 3       | 4       | Yes (evict 2) |
| 2      | 5       | 1       | 2       | 4       | Yes (evict 3) |
| 3      | 5       | 1       | 2       | 3       | Yes (evict 4) |
| 4      | 5       | 1       | 2       | 3       | No     |
| 5      | 5       | 1       | 2       | 3       | No     |
**Total: 10 page faults**
**Adding a frame INCREASED faults from 9 to 10!**
This happens because FIFO's eviction decision depends only on load order, not access pattern. With 4 frames, we keep pages around longer, which changes which pages get evicted at critical moments. It's a pathological interaction between the queue state and the access pattern.
**The fix**: LRU and Optimal don't suffer from Bélády's anomaly. They're **stack algorithms**—the set of pages in memory with N+1 frames is always a superset of pages with N frames. This property guarantees that more frames never hurts.
Your simulator will demonstrate this anomaly automatically when you run the comparison tests.
---
## Working Set Size: The Real Performance Predictor
Page fault counts tell you what happened, but not what's *going to happen*. For that, you need the **working set**: the set of pages a process actively uses within a time window.

![Bélády's Anomaly Demonstration](./diagrams/tdd-diag-m4-11.svg)

![Working Set Size Tracking](./diagrams/diag-working-set-window.svg)

![Working Set Sliding Window](./diagrams/tdd-diag-m4-12.svg)

```c
typedef struct {
    uint32_t *recent_vpns;      // Circular buffer of recent VPNs
    uint32_t window_size;
    uint32_t head;
    uint32_t count;
} working_set_tracker_t;
void working_set_init(working_set_tracker_t *ws, uint32_t window_size) {
    ws->recent_vpns = calloc(window_size, sizeof(uint32_t));
    ws->window_size = window_size;
    ws->head = 0;
    ws->count = 0;
    for (uint32_t i = 0; i < window_size; i++) {
        ws->recent_vpns[i] = UINT32_MAX;
    }
}
void working_set_record(working_set_tracker_t *ws, uint32_t vpn) {
    ws->recent_vpns[ws->head] = vpn;
    ws->head = (ws->head + 1) % ws->window_size;
    if (ws->count < ws->window_size) {
        ws->count++;
    }
}
uint32_t working_set_size(working_set_tracker_t *ws) {
    // Count distinct VPNs in the window
    uint32_t distinct = 0;
    uint32_t *seen = calloc(65536, sizeof(uint32_t));  // Simple hash for demo
    for (uint32_t i = 0; i < ws->count; i++) {
        uint32_t vpn = ws->recent_vpns[i];
        if (vpn != UINT32_MAX && !seen[vpn % 65536]) {
            seen[vpn % 65536] = 1;
            distinct++;
        }
    }
    free(seen);
    return distinct;
}
```
**The thrashing condition**: If `working_set_size > num_frames`, the process will thrash. Every access will evict a page that's still in the working set, causing another fault.
You can detect this in your simulator:
```c
void check_thrashing(simulator_t *sim, working_set_tracker_t *ws) {
    uint32_t ws_size = working_set_size(ws);
    uint32_t frames_available = sim->phys_mem.total_frames;
    if (ws_size > frames_available) {
        printf("[WARNING] Potential thrashing! Working set (%u) > frames (%u)\n",
               ws_size, frames_available);
        sim->stats.thrashing_warnings++;
    }
}
```
---
## The Complete Page Replacement Simulator
Let's integrate everything into a complete simulator:
```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
// Configuration constants (from earlier milestones)
#define PAGE_SHIFT      12
#define PAGE_SIZE       4096
#define PAGE_MASK       0xFFF
#define MAX_VPN         1048576
// Page Table Entry
typedef struct {
    uint32_t pfn;
    bool valid;
    bool readable;
    bool writable;
    bool dirty;
    bool referenced;
} pte_t;
// Swap Space
typedef struct {
    uint8_t data[PAGE_SIZE];
    uint32_t vpn;
    uint32_t process_id;
    bool occupied;
} swap_slot_t;
typedef struct {
    swap_slot_t *slots;
    uint32_t total_slots;
    uint32_t *free_list;
    uint32_t free_count;
} swap_space_t;
// Replacement Algorithm Types
typedef enum {
    REPL_FIFO,
    REPL_LRU,
    REPL_CLOCK,
    REPL_OPTIMAL
} repl_algo_t;
// FIFO State
typedef struct {
    uint32_t *queue;
    uint32_t capacity;
    uint32_t head;
    uint32_t tail;
    uint32_t count;
} fifo_t;
// LRU State
typedef struct {
    uint32_t *vpns;
    uint64_t *access_times;
    uint32_t count;
    uint64_t clock;
} lru_t;
// Clock State
typedef struct {
    uint32_t *vpns;
    bool *referenced;
    uint32_t count;
    uint32_t hand;
} clock_t;
// Optimal State
typedef struct {
    const memory_access_t *trace;
    size_t trace_length;
    size_t current_pos;
} optimal_t;
// Physical Memory
typedef struct {
    uint8_t **frames;
    uint32_t *frame_to_vpn;     // Reverse mapping: frame -> VPN
    uint32_t *frame_to_process; // Which process owns each frame
    uint32_t total_frames;
    uint32_t *free_list;
    uint32_t free_count;
} physical_memory_t;
// Statistics
typedef struct {
    uint64_t total_accesses;
    uint64_t page_faults;
    uint64_t protection_faults;
    uint64_t swap_writes;
    uint64_t swap_reads;
    uint64_t swap_discards;
    uint64_t thrashing_warnings;
    // Per-algorithm tracking
    uint64_t fifo_faults;
    uint64_t lru_faults;
    uint64_t clock_faults;
    uint64_t optimal_faults;
} stats_t;
// Swap Mapping (VPN -> swap slot)
typedef struct {
    int32_t swap_slot;
    bool in_swap;
} swap_mapping_t;
// Simulator State
typedef struct {
    pte_t *page_table;
    physical_memory_t phys_mem;
    swap_space_t *swap;
    swap_mapping_t *swap_map;
    stats_t stats;
    // Replacement algorithm state
    repl_algo_t current_algo;
    fifo_t fifo;
    lru_t lru;
    clock_t clock;
    optimal_t optimal;
    // Working set tracking
    working_set_tracker_t working_set;
} simulator_t;
```
### Initialization with Replacement Support
```c
simulator_t* simulator_create(uint32_t num_frames, 
                              uint32_t num_swap_slots,
                              repl_algo_t algo,
                              const memory_access_t *trace,
                              size_t trace_length) {
    simulator_t *sim = calloc(1, sizeof(simulator_t));
    if (!sim) return NULL;
    // Page table
    sim->page_table = calloc(MAX_VPN, sizeof(pte_t));
    for (uint32_t i = 0; i < MAX_VPN; i++) {
        sim->page_table[i].valid = false;
    }
    // Physical memory
    sim->phys_mem.total_frames = num_frames;
    sim->phys_mem.frames = calloc(num_frames, sizeof(uint8_t*));
    sim->phys_mem.frame_to_vpn = calloc(num_frames, sizeof(uint32_t));
    sim->phys_mem.frame_to_process = calloc(num_frames, sizeof(uint32_t));
    sim->phys_mem.free_list = calloc(num_frames, sizeof(uint32_t));
    sim->phys_mem.free_count = num_frames;
    for (uint32_t i = 0; i < num_frames; i++) {
        sim->phys_mem.free_list[i] = i;
    }
    // Swap space
    sim->swap = swap_create(num_swap_slots);
    sim->swap_map = calloc(MAX_VPN, sizeof(swap_mapping_t));
    for (uint32_t i = 0; i < MAX_VPN; i++) {
        sim->swap_map[i].in_swap = false;
        sim->swap_map[i].swap_slot = -1;
    }
    // Replacement algorithm
    sim->current_algo = algo;
    switch (algo) {
        case REPL_FIFO:
            fifo_init(&sim->fifo, num_frames);
            break;
        case REPL_LRU:
            lru_init(&sim->lru, num_frames);
            break;
        case REPL_CLOCK:
            clock_init(&sim->clock, num_frames);
            break;
        case REPL_OPTIMAL:
            optimal_init(&sim->optimal, trace, trace_length);
            break;
    }
    // Working set tracker
    working_set_init(&sim->working_set, 100);  // Track last 100 accesses
    return sim;
}
```
### Frame Allocation with Eviction
```c
int32_t allocate_frame_with_eviction(simulator_t *sim, uint32_t vpn) {
    // Try to get a free frame first
    if (sim->phys_mem.free_count > 0) {
        return sim->phys_mem.free_list[--sim->phys_mem.free_count];
    }
    // No free frames - must evict!
    uint32_t victim_vpn;
    switch (sim->current_algo) {
        case REPL_FIFO:
            victim_vpn = fifo_evict(&sim->fifo);
            break;
        case REPL_LRU:
            victim_vpn = lru_evict(&sim->lru);
            break;
        case REPL_CLOCK:
            victim_vpn = clock_evict(&sim->clock, sim->page_table);
            break;
        case REPL_OPTIMAL:
            // Need to pass loaded pages list
            victim_vpn = optimal_evict(&sim->optimal, 
                                       sim->fifo.queue, 
                                       sim->fifo.count);
            break;
        default:
            return -1;
    }
    // Evict the victim
    pte_t *victim_pte = &sim->page_table[victim_vpn];
    uint32_t victim_pfn = victim_pte->pfn;
    printf("[Eviction] Replacing VPN %u (PFN %u) with VPN %u using %s\n",
           victim_vpn, victim_pfn, vpn, 
           sim->current_algo == REPL_FIFO ? "FIFO" :
           sim->current_algo == REPL_LRU ? "LRU" :
           sim->current_algo == REPL_CLOCK ? "Clock" : "Optimal");
    // Write to swap if dirty
    if (victim_pte->dirty) {
        int32_t slot = swap_alloc(sim->swap);
        if (slot < 0) {
            fprintf(stderr, "Swap full - cannot evict!\n");
            return -1;
        }
        memcpy(sim->swap->slots[slot].data, 
               sim->phys_mem.frames[victim_pfn], 
               PAGE_SIZE);
        sim->swap->slots[slot].vpn = victim_vpn;
        sim->swap->slots[slot].occupied = true;
        sim->swap_map[victim_vpn].swap_slot = slot;
        sim->swap_map[victim_vpn].in_swap = true;
        sim->stats.swap_writes++;
    } else {
        sim->stats.swap_discards++;
    }
    // Invalidate the victim's PTE
    victim_pte->valid = false;
    victim_pte->pfn = 0;
    victim_pte->dirty = false;
    victim_pte->referenced = false;
    // Update replacement structures
    switch (sim->current_algo) {
        case REPL_LRU:
            // Remove from LRU tracking
            for (uint32_t i = 0; i < sim->lru.count; i++) {
                if (sim->lru.vpns[i] == victim_vpn) {
                    sim->lru.vpns[i] = sim->lru.vpns[--sim->lru.count];
                    break;
                }
            }
            break;
        case REPL_CLOCK:
            // Already handled in clock_evict
            break;
        default:
            break;
    }
    return (int32_t)victim_pfn;
}
```
### Complete Translation with Replacement
```c
typedef enum {
    TRANS_OK,
    TRANS_PAGE_FAULT,
    TRANS_PROTECTION_FAULT,
    TRANS_SWAP_ERROR
} trans_result_t;
typedef struct {
    trans_result_t result;
    uint32_t physical_address;
    bool was_fault;
    bool was_swap_in;
} trans_output_t;
trans_output_t translate_with_replacement(
    simulator_t *sim,
    uint32_t va,
    bool is_write
) {
    trans_output_t out = {0};
    sim->stats.total_accesses++;
    uint32_t vpn = va >> PAGE_SHIFT;
    uint32_t offset = va & PAGE_MASK;
    // Update working set tracker
    working_set_record(&sim->working_set, vpn);
    // Check if page is valid
    pte_t *pte = &sim->page_table[vpn];
    if (!pte->valid) {
        // Page fault!
        sim->stats.page_faults++;
        out.was_fault = true;
        // Get a frame (possibly evicting)
        int32_t pfn = allocate_frame_with_eviction(sim, vpn);
        if (pfn < 0) {
            out.result = TRANS_SWAP_ERROR;
            return out;
        }
        // Allocate frame data if needed
        if (!sim->phys_mem.frames[pfn]) {
            sim->phys_mem.frames[pfn] = calloc(PAGE_SIZE, 1);
        }
        // Check if page is in swap
        if (sim->swap_map[vpn].in_swap) {
            // Reload from swap
            int32_t slot = sim->swap_map[vpn].swap_slot;
            memcpy(sim->phys_mem.frames[pfn], 
                   sim->swap->slots[slot].data, 
                   PAGE_SIZE);
            swap_free(sim->swap, slot);
            sim->swap_map[vpn].in_swap = false;
            sim->stats.swap_reads++;
            out.was_swap_in = true;
            printf("[Swap In] VPN %u from slot %d to PFN %u\n", vpn, slot, pfn);
        }
        // Update PTE
        pte->pfn = pfn;
        pte->valid = true;
        pte->readable = true;
        pte->writable = true;
        pte->dirty = false;
        pte->referenced = false;
        // Update reverse mappings
        sim->phys_mem.frame_to_vpn[pfn] = vpn;
        // Insert into replacement structure
        switch (sim->current_algo) {
            case REPL_FIFO:
                fifo_insert(&sim->fifo, vpn);
                break;
            case REPL_LRU:
                lru_insert(&sim->lru, vpn);
                break;
            case REPL_CLOCK:
                clock_insert(&sim->clock, vpn);
                break;
            default:
                break;
        }
    }
    // Update access bits
    pte->referenced = true;
    if (is_write) {
        pte->dirty = true;
    }
    // Update LRU timestamp on access
    if (sim->current_algo == REPL_LRU) {
        lru_access(&sim->lru, vpn);
    }
    // Compose physical address
    out.physical_address = (pte->pfn << PAGE_SHIFT) | offset;
    out.result = TRANS_OK;
    // Check for thrashing
    check_thrashing(sim, &sim->working_set);
    return out;
}
```
---
## Comparative Statistics and Analysis

![Physical Memory Frame Pool with Reverse Mapping](./diagrams/tdd-diag-m4-01.svg)

![Page Replacement Algorithms: Visual Comparison](./diagrams/diag-replacement-algorithm-comparison.svg)

![Full Translation Pipeline Summary](./diagrams/tdd-diag-m4-19.svg)

```c
typedef struct {
    repl_algo_t algo;
    uint64_t page_faults;
    uint64_t swap_writes;
    uint64_t swap_reads;
    double avg_working_set;
    double fault_rate;
} algo_result_t;
void run_comparison(const memory_access_t *trace, 
                    size_t trace_length,
                    uint32_t num_frames,
                    uint32_t num_swap_slots) {
    algo_result_t results[4];
    const char *algo_names[] = {"FIFO", "LRU", "Clock", "Optimal"};
    repl_algo_t algos[] = {REPL_FIFO, REPL_LRU, REPL_CLOCK, REPL_OPTIMAL};
    printf("\n");
    printf("============================================================\n");
    printf("     PAGE REPLACEMENT ALGORITHM COMPARISON                  \n");
    printf("============================================================\n");
    printf("Trace length:    %zu accesses\n", trace_length);
    printf("Physical frames: %u\n", num_frames);
    printf("Swap slots:      %u\n", num_swap_slots);
    printf("------------------------------------------------------------\n\n");
    for (int i = 0; i < 4; i++) {
        simulator_t *sim = simulator_create(num_frames, num_swap_slots, 
                                           algos[i], trace, trace_length);
        // Run the trace
        for (size_t j = 0; j < trace_length; j++) {
            translate_with_replacement(sim, 
                                       trace[j].virtual_address,
                                       trace[j].is_write);
            // Update optimal's position
            if (algos[i] == REPL_OPTIMAL) {
                sim->optimal.current_pos = j + 1;
            }
        }
        // Record results
        results[i].algo = algos[i];
        results[i].page_faults = sim->stats.page_faults;
        results[i].swap_writes = sim->stats.swap_writes;
        results[i].swap_reads = sim->stats.swap_reads;
        results[i].fault_rate = (double)sim->stats.page_faults / trace_length * 100.0;
        printf("%-8s: %5lu faults (%.2f%%), %3lu swap writes, %3lu swap reads\n",
               algo_names[i], 
               results[i].page_faults,
               results[i].fault_rate,
               results[i].swap_writes,
               results[i].swap_reads);
        // Cleanup
        // (simulator_destroy implementation omitted for brevity)
    }
    printf("\n");
    printf("Optimal is the theoretical lower bound.\n");
    printf("LRU/Clock within 10%% of Optimal indicates good locality.\n");
    printf("FIFO significantly worse than Optimal suggests access patterns\n");
    printf("  that conflict with FIFO's load-order bias.\n");
    printf("============================================================\n");
}
```
### Demonstrating Bélády's Anomaly
```c
void demonstrate_belady_anomaly(void) {
    // The classic trace that triggers Bélády's anomaly
    memory_access_t trace[] = {
        {false, 0x00001000},  // VPN 1
        {false, 0x00002000},  // VPN 2
        {false, 0x00003000},  // VPN 3
        {false, 0x00004000},  // VPN 4
        {false, 0x00001000},  // VPN 1
        {false, 0x00002000},  // VPN 2
        {false, 0x00005000},  // VPN 5
        {false, 0x00001000},  // VPN 1
        {false, 0x00002000},  // VPN 2
        {false, 0x00003000},  // VPN 3
        {false, 0x00004000},  // VPN 4
        {false, 0x00005000},  // VPN 5
    };
    size_t trace_len = sizeof(trace) / sizeof(trace[0]);
    printf("\n");
    printf("============================================================\n");
    printf("        BÉLÁDY'S ANOMALY DEMONSTRATION                      \n");
    printf("============================================================\n");
    printf("Access pattern: 1,2,3,4,1,2,5,1,2,3,4,5\n\n");
    printf("Testing FIFO with varying frame counts:\n");
    printf("----------------------------------------\n");
    for (uint32_t frames = 3; frames <= 6; frames++) {
        simulator_t *sim = simulator_create(frames, 100, REPL_FIFO, trace, trace_len);
        for (size_t i = 0; i < trace_len; i++) {
            translate_with_replacement(sim, trace[i].virtual_address, trace[i].is_write);
        }
        printf("  %u frames: %lu page faults\n", frames, sim->stats.page_faults);
    }
    printf("\n");
    printf("Notice: 4 frames has MORE faults than 3 frames!\n");
    printf("This is Bélády's anomaly - unique to FIFO.\n\n");
    printf("Now testing LRU (immune to anomaly):\n");
    printf("----------------------------------------\n");
    for (uint32_t frames = 3; frames <= 6; frames++) {
        simulator_t *sim = simulator_create(frames, 100, REPL_LRU, trace, trace_len);
        for (size_t i = 0; i < trace_len; i++) {
            translate_with_replacement(sim, trace[i].virtual_address, trace[i].is_write);
        }
        printf("  %u frames: %lu page faults\n", frames, sim->stats.page_faults);
    }
    printf("\nLRU faults decrease monotonically with more frames.\n");
    printf("============================================================\n");
}
```
---
## Hardware Soul: What Really Happens During Replacement
Let's trace a page fault with eviction through the hardware layers.
### Level 1: Application (Software)
```c
int value = *ptr;  // ptr = 0x00001234, VPN 1
```
The CPU issues a load for virtual address 0x00001234.
### Level 2: Memory Management Unit (Hardware)
1. **TLB miss**: VPN 1 not in TLB
2. **Page table walk**: Read PTE for VPN 1
3. **PTE valid = 0**: Page not in memory!
4. **Page fault exception**: CPU traps to OS kernel
### Level 3: Operating System (Software, Ring 0)
The page fault handler runs:
1. **Identify faulting VPN**: Extract VPN 1 from fault address
2. **Check if valid address**: Is VPN 1 in process's address space?
3. **Find a free frame**: Check free list... empty!
4. **Run replacement algorithm**: FIFO says evict VPN 3
5. **Check dirty bit**: VPN 3's PTE has dirty=1
6. **Write to swap**: DMA write of frame 42 to swap slot 7
   - This is a **disk I/O**: 5-10 ms
7. **Invalidate TLB entry**: Remove VPN 3 from TLB
8. **Load new page**: 
   - Is VPN 1 in swap? Yes, slot 3
   - DMA read from swap slot 3 to frame 42
   - This is another **disk I/O**: 5-10 ms
9. **Update page tables**: 
   - VPN 3: valid=0, pfn=0
   - VPN 1: valid=1, pfn=42
10. **Invalidate TLB for VPN 3**: (already done)
11. **Resume user process**: Return from exception
### Level 4: Disk Controller (Hardware)
The disk controller receives DMA commands:
- Write 4 KB from frame 42 to swap slot 7
- Read 4 KB from swap slot 3 to frame 42
Each operation involves:
- Seek time (HDD): 3-5 ms
- Rotational latency (HDD): 2-4 ms  
- Transfer time: 0.04 ms for 4 KB at 100 MB/s
**Total time for one page fault with dirty eviction**: 10-20 ms
During this time, the CPU could have executed **50-100 million instructions**. Instead, it's idle, waiting for disk.
### The SSD Advantage
Modern SSDs reduce this dramatically:
| Storage | Read Latency | Write Latency | Fault Cost |
|---------|--------------|---------------|------------|
| HDD | 5-10 ms | 5-10 ms | 10-20 ms |
| SATA SSD | 50-100 μs | 100-200 μs | 150-300 μs |
| NVMe SSD | 10-25 μs | 20-50 μs | 30-75 μs |
SSDs make page faults "only" 10,000× more expensive than memory accesses instead of 100,000×. Still a massive penalty, but the difference between "unusable" and "slow."
---
## Testing Your Replacement Algorithms
### Test Case 1: Sequential Access
```
R 0x00001000
R 0x00002000
R 0x00003000
R 0x00004000
R 0x00005000
R 0x00001000  # Revisit first page
R 0x00002000
...
```
**With 3 frames**: All algorithms should behave similarly (sequential access has no locality to exploit).
### Test Case 2: Looping Access
```
R 0x00001000
R 0x00002000
R 0x00003000
R 0x00001000
R 0x00002000
R 0x00003000
... (repeat 10 times)
```
**With 3 frames**: All algorithms should get exactly 3 page faults (one per unique page).
**With 2 frames**: FIFO will thrash! The loop of 3 pages with 2 frames causes constant eviction.
### Test Case 3: Dirty Page Tracking
```
W 0x00001000  # Write, sets dirty
R 0x00002000  # Read, no dirty
W 0x00003000  # Write, sets dirty
R 0x00004000  # Trigger eviction (with 3 frames)
```
**Verify**: VPN 1 and VPN 3 should have been written to swap. VPN 2 should have been discarded (clean).
### Test Case 4: Working Set Detection
```
# Access 5 pages in a loop (working set = 5)
R 0x00001000
R 0x00002000
R 0x00003000
R 0x00004000
R 0x00005000
(repeat)
```
**With 3 frames**: Working set tracker should report 5, triggering thrashing warning.
**With 5 frames**: Working set = frames, no thrashing.
---
## Common Pitfalls
### Pitfall 1: Not Writing Dirty Pages to Swap
**The mistake**: Evicting without checking dirty bit.
```c
// WRONG! Data loss!
void evict_wrong(pte_t *pte) {
    pte->valid = false;  // If dirty, modifications are lost!
}
```
**The fix**: Always check and write back.
```c
void evict_correct(pte_t *pte, swap_space_t *swap, uint32_t vpn) {
    if (pte->dirty) {
        write_to_swap(swap, vpn, pte->pfn);
    }
    pte->valid = false;
}
```
### Pitfall 2: Forgetting to Update Replacement Structures
**The mistake**: Evicting from page table but not from FIFO queue / LRU list.
**Symptom**: Replacement algorithm tries to evict an already-invalid page.
**The fix**: Always synchronize all data structures on eviction.
### Pitfall 3: Not Invalidating TLB on Eviction
**The mistake**: Evicting a page but leaving its TLB entry.
**Symptom**: TLB hit returns stale PFN that now belongs to a different page.
**The fix**: Call `tlb_invalidate_entry()` after eviction.
### Pitfall 4: Swap Space Exhaustion
**The mistake**: Assuming swap is infinite.
**Reality**: Swap has limits too. When swap fills, the OS kills processes (OOM killer).
**The fix**: Check swap_alloc() return value and handle failure gracefully.
---
## Design Decisions: Why This, Not That
| Algorithm | Pros | Cons | Used By |
|-----------|------|------|---------|
| **FIFO** | Simple, no metadata | Bélády's anomaly, ignores recency | Rarely used alone |
| **LRU** | Optimal for temporal locality | Requires timestamp tracking, O(n) naive | Often approximated |
| **Clock** | Hardware-friendly, O(1) | Coarse approximation | Linux, Windows |
| **Optimal** | Theoretical lower bound | Requires future knowledge | Analysis only |
| **LFU** | Good for frequency-based access | Ignores recency, easy to pollute | Some databases |
| **ARC** | Balanced recency + frequency | Complex, patent issues | ZFS |
| Swap Strategy | Pros | Cons | Used By |
|---------------|------|------|---------|
| **Partition** | Predictable performance | Fixed size | Traditional Unix |
| **Swap file** | Flexible size | Slight overhead | Linux, Windows |
| **Swap to SSD** | Much faster | Wear concerns | Modern systems |
| **No swap** | Simpler | Can't overcommit | Embedded, some containers |
---
## What You've Built
At this point, your simulator implements:
1. **Physical memory frame pool** with configurable size
2. **Swap space simulation** for storing evicted pages
3. **FIFO replacement** with queue-based eviction
4. **LRU replacement** with timestamp tracking
5. **Clock (Second-Chance) replacement** with referenced-bit sweeping
6. **Optimal (Bélády's) replacement** with look-ahead for benchmarking
7. **Dirty page write-back** to swap on eviction
8. **Page reload from swap** on fault
9. **Working set tracking** with thrashing detection
10. **Comparative statistics** across all algorithms
You've built the complete memory overcommitment pipeline that every modern operating system uses.
---
## Knowledge Cascade: Where This Leads
### Same Domain Connections
**→ Working Set Model**: The working set tracker you implemented is the foundation for **working set paging**, an alternative to global replacement where each process is allocated frames proportional to its working set. This prevents one process from stealing all memory from others.
**→ Prefetching**: Once you can predict which pages will be needed (Optimal algorithm), you can **prefetch** them before the fault occurs. Modern CPUs do this with instruction prefetch; OSes do it with memory-mapped files (readahead).
**→ Memory Compression**: Before evicting to slow swap, some systems (macOS, Linux zswap) **compress** pages in memory. A compressed page might use 50% less RAM, letting you keep more pages before evicting. Your dirty bit tracking is exactly what's needed to decide what to compress.
### Cross-Domain Connections
**→ CDN Cache Eviction**: A CDN edge server is a cache for web content. When the cache fills, what do we evict? The same algorithms apply:
- LRU: Evict the object least recently requested
- LFU: Evict the object with fewest total requests
- TTL-based: Evict objects past their expiration time
The CDN "working set" is the set of hot objects that users are actively requesting. A cache miss storm (sudden surge in requests for uncached content) is exactly analogous to a page fault storm.
**→ Browser Tab Discarding**: Chrome's memory manager uses page replacement logic. When memory pressure is high, Chrome:
1. Identifies "inactive" tabs (no recent user interaction)
2. Ranks them by memory usage and importance
3. Discards the least important, keeping a placeholder
4. Reloads from disk/network when the user switches to that tab
The "suspended" tab is evicted to "swap" (serialized to disk). The algorithm is LRU-ish with heuristics for media tabs, pinned tabs, etc.
**→ Redis maxmemory-policy**: Redis implements multiple eviction policies for when it hits its memory limit:
- `volatile-lru`: Evict least-recently-used among keys with expiration
- `allkeys-lru`: Evict any key using LRU
- `volatile-lfu`: Evict least-frequently-used among expiring keys
- `allkeys-random`: Evict randomly
The tradeoffs are identical to page replacement: LRU favors temporal locality, LFU favors frequency, random is simple but ignores patterns.
**→ CPU Cache Replacement**: L1/L2 caches use variants of these algorithms, but in hardware:
- True LRU requires too much state (N! states for N ways)
- Pseudo-LRU trees approximate with O(log N) bits
- PLRU (Pseudo-LRU) is the hardware equivalent of Clock
Understanding page replacement gives you intuition for why CPU cache behavior is often non-intuitive—the same algorithmic tradeoffs apply.
**→ Machine Learning Data Loading**: PyTorch DataLoader and TensorFlow tf.data use caching and prefetching with similar principles:
- Cache frequently accessed samples (LRU-like)
- Prefetch next batch while GPU trains on current batch (working set awareness)
- Shuffle buffer is essentially a cache of upcoming samples
The "working set" of a training epoch is the dataset size. If it fits in RAM, cache everything. If not, use replacement to decide what stays.
### Historical Context
**→ The Working Set Model (1968)**: Peter Denning introduced the working set model to explain why some programs thrashed while others didn't. His insight: page faults aren't about total memory usage, but about the **gap between working set and available frames**. This shifted OS design from "allocate as much as possible" to "allocate proportional to working set."
**→ The Clock Algorithm (1970s)**: The Clock algorithm was developed when hardware had only a single referenced bit per page. The insight that you could approximate LRU with just this bit was crucial—it made virtual memory practical on early minicomputers with limited hardware support.
**→ Linux's Two-Handed Clock**: Linux uses a variant called the "two-handed clock": one hand clears referenced bits, another hand (following behind) evicts pages whose bits are still clear. The distance between hands determines how long a page gets its "second chance." This balances recency against the cost of scanning.
---
## Looking Ahead
You've completed the Virtual Memory Simulator. You now understand:
- How virtual addresses become physical addresses
- How TLBs accelerate translation
- How multi-level tables compress sparse address spaces
- How replacement algorithms decide who stays and who goes
The system you've built mirrors what runs on every computer today. When you see a program slow down mysteriously, you now have the mental model to diagnose it:
- Check TLB miss rates (perf stat -e dTLB-load-misses)
- Check page fault rates (ps -o majflt,minflt)
- Check swap activity (vmstat, swapon -s)
- Consider whether the working set fits in memory
More importantly, you understand that memory isn't just "RAM"—it's a hierarchy of caches (TLB → L1 → L2 → L3 → RAM → Swap), and performance depends on how well your access patterns match each level's characteristics.

![Dirty Page Write-Back Path](./diagrams/tdd-diag-m4-08.svg)

![Complete Translation Pipeline: VA to PA](./diagrams/diag-full-translation-pipeline.svg)


![Translation with Replacement Pipeline](./diagrams/tdd-diag-m4-17.svg)

![Statistics Collection Architecture](./diagrams/diag-statistics-collection-points.svg)

![Statistics Collection Points](./diagrams/tdd-diag-m4-15.svg)

[[CRITERIA_JSON: {"milestone_id": "virtual-memory-sim-m4", "criteria": ["Physical memory modeled as configurable frame pool (uint32_t total_frames, free_list, free_count); page fault triggers replacement when free_count == 0", "Swap space structure (swap_slot_t with data[PAGE_SIZE], vpn, occupied) stores evicted page contents; swap_alloc() and swap_free() manage slot lifecycle", "FIFO replacement uses circular queue (head, tail, count); evict returns queue[head] with head = (head + 1) % capacity", "LRU replacement tracks last_access_time (uint64_t) per page; evict scans for minimum timestamp, returns that VPN", "Clock replacement uses circular buffer with referenced[] array; evict clears referenced bits on first pass, evicts first page with referenced=false on second pass", "Optimal (Bélády's) replacement requires trace lookahead; evict selects page with farthest future access or never-accessed-again page", "Dirty page write-back: if pte->dirty on eviction, memcpy frame data to swap slot and increment stats.swap_writes; clean pages increment stats.swap_discards", "Page reload on fault: if swap_map[vpn].in_swap, memcpy from swap slot to frame, increment stats.swap_reads, free the swap slot", "Comparative statistics printed: page_faults, swap_writes, swap_reads per algorithm; fault rate as percentage of total_accesses", "Bélády's anomaly demonstration shows FIFO fault count non-monotonic with increasing frame count; LRU faults decrease monotonically", "Working set tracker maintains sliding window of recent VPNs; working_set_size() returns count of distinct VPNs in window", "Thrashing detection compares working_set_size() against num_frames; logs warning when working set exceeds available memory", "Swap mapping structure (swap_mapping_t with swap_slot index and in_swap bool) tracks which pages have swap copies", "Statistics structure includes total_accesses, page_faults, swap_writes, swap_reads, swap_discards, thrashing_warnings as uint64_t fields"]}]
<!-- END_MS -->


## System Overview

![Swap Slot Structure](./diagrams/tdd-diag-m4-02.svg)

![System Overview](./diagrams/system-overview.svg)




# TDD

Build a complete virtual memory subsystem simulator that mirrors real OS memory management. The simulator processes memory access traces through a full translation pipeline: virtual address decomposition, TLB caching, multi-level page table walks, demand paging with page fault handling, and swap-backed page replacement algorithms. This reveals the same performance phenomena (TLB miss storms, page fault cascades, Bélády's anomaly, thrashing) that production systems encounter.



![Algorithm Comparison Output Format](./diagrams/tdd-diag-m4-16.svg)

<!-- TDD_MOD_ID: virtual-memory-sim-m1 -->
# Technical Design Document: Single-Level Page Table and Address Translation
**Module ID:** `virtual-memory-sim-m1`  
**Language:** C (BINDING)  
**Difficulty:** Intermediate → Advanced transition point
---
## 1. Module Charter
This module implements a flat (single-level) page table with complete address translation, permission checking, demand paging, and page fault detection. It processes memory access traces (text files with R/W operations and virtual addresses) and produces physical addresses or fault codes.
**What it does:**
- Parses trace files in format `R/W 0xADDRESS`
- Decomposes 32-bit virtual addresses into VPN and offset using configurable page size
- Performs O(1) page table lookups to translate VPN → PFN
- Detects page faults (valid=0) and triggers demand paging frame allocation
- Detects protection faults (permission violations) and logs them distinctly
- Tracks dirty/referenced bits per PTE
- Collects and reports translation statistics
**What it does NOT do:**
- TLB caching (Milestone 2)
- Multi-level hierarchical page tables (Milestone 3)
- Page replacement/swap (Milestone 4)
**Upstream dependencies:** None — this is the foundation module.
**Downstream dependencies:** Milestones 2-4 extend this module by adding TLB caching before page table lookup, converting to multi-level tables, and adding replacement algorithms when frames exhaust.
**Invariants that must always hold:**
1. `valid == false` implies the PFN field contains meaningless data
2. `dirty == true` implies the page has been written to at least once
3. `referenced == true` implies the page has been accessed (read or write) at least once
4. Physical frame numbers in PTEs always point to allocated frames (no dangling PFNs)
5. Free list count + allocated frame count == total frames (conservation of frames)
---
## 2. File Structure
Create files in this exact order:
```
01. include/config.h           — Constants, macros, page size configuration
02. include/types.h            — All struct definitions (PTE, stats, physical memory)
03. include/translate.h        — Translation function declarations
04. include/parser.h           — Trace parser declarations
05. include/stats.h            — Statistics collection declarations
06. src/translate.c            — Address translation implementation
07. src/parser.c               — Trace file parsing implementation
08. src/stats.c                — Statistics reporting implementation
09. src/main.c                 — Entry point, CLI argument handling
10. tests/test_basic.c         — Unit tests for address decomposition
11. tests/test_translation.c   — Unit tests for translation logic
12. traces/basic.trace         — Simple test trace (demand paging)
13. traces/dirty.trace         — Test trace for dirty bit tracking
14. traces/protection.trace    — Test trace for protection faults
15. Makefile                   — Build system
```
---
## 3. Complete Data Model
### 3.1 Configuration Constants (`config.h`)
```c
#ifndef CONFIG_H
#define CONFIG_H
#include <stdint.h>
/*
 * Page configuration.
 * Default: 4 KB pages = 2^12 bytes.
 * PAGE_SHIFT determines bit position where VPN ends and offset begins.
 * 
 * Memory layout for 32-bit address:
 * ┌────────────────────────┬────────────────────┐
 * │   VPN (20 bits)        │   Offset (12 bits) │
 * │   bits 31-12           │   bits 11-0        │
 * └────────────────────────┴────────────────────┘
 */
#define PAGE_SHIFT          12
#define PAGE_SIZE           (1u << PAGE_SHIFT)   // 4096 bytes
#define PAGE_MASK           (PAGE_SIZE - 1)       // 0x00000FFF
/*
 * Address space limits.
 * 32-bit virtual address space = 2^32 bytes = 4 GB.
 * Number of pages = 2^32 / 2^12 = 2^20 = 1,048,576.
 */
#define VIRTUAL_ADDRESS_BITS    32
#define MAX_VPN                 (1u << (VIRTUAL_ADDRESS_BITS - PAGE_SHIFT))  // 2^20
/*
 * Physical memory defaults.
 * These can be overridden via command-line arguments.
 */
#define DEFAULT_NUM_FRAMES      64
#define MAX_TRACE_LINE_LENGTH   256
#endif // CONFIG_H
```
### 3.2 Type Definitions (`types.h`)
```c
#ifndef TYPES_H
#define TYPES_H
#include <stdint.h>
#include <stdbool.h>
/*
 * Page Table Entry (PTE)
 * 
 * Each PTE represents the mapping for one virtual page.
 * Memory layout (logical, not packed):
 * 
 * ┌─────────────────────────────────────────────────────────────┐
 * │ pfn (32 bits) │ flags...                                    │
 * └─────────────────────────────────────────────────────────────┘
 * 
 * WHY each field exists:
 * - pfn:        The physical frame number where this page's data lives.
 *               Combined with offset to form physical address.
 * - valid:      Critical for demand paging. 0 = page not in memory,
 *               triggers page fault on access.
 * - readable:   Permission bit. 0 = read access causes protection fault.
 * - writable:   Permission bit. 0 = write access causes protection fault.
 * - dirty:      Set by hardware on write. Used by OS to know if page
 *               must be written to swap on eviction (vs just discarded).
 * - referenced: Set by hardware on any access. Used by page replacement
 *               algorithms to identify "hot" pages.
 * 
 * Total size: 32 + 5 = ~40 bits, padded to 8 bytes for alignment.
 */
typedef struct {
    uint32_t pfn;          // Physical Frame Number (offset 0x00, 4 bytes)
                           // Meaningful only when valid == true
    bool valid;            // Offset 0x04, 1 byte - Page present in memory?
    bool readable;         // Offset 0x05, 1 byte - Read permission
    bool writable;         // Offset 0x06, 1 byte - Write permission
    bool dirty;            // Offset 0x07, 1 byte - Has been written to?
    bool referenced;       // Offset 0x08, 1 byte - Has been accessed?
} pte_t;                   // Total: ~12 bytes (with alignment padding)
/*
 * Physical Memory Frame Pool
 * 
 * Models physical RAM as a fixed array of frames.
 * Each frame is PAGE_SIZE bytes (4096 by default).
 * 
 * WHY each field exists:
 * - frames:       Array of pointers to actual frame data (NULL = unallocated).
 *                 We use pointers so frames can be allocated on demand.
 * - total_frames: Capacity of the frame pool. Fixed at simulator creation.
 * - free_list:    Stack of available frame indices. Enables O(1) allocation.
 * - free_count:   Number of entries in free_list. 0 = memory exhausted.
 */
typedef struct {
    uint8_t **frames;         // Array of frame pointers (offset 0x00)
                              // frames[pfn] points to PAGE_SIZE bytes
    uint32_t total_frames;    // Capacity (offset 0x08, 4 bytes)
    uint32_t *free_list;      // Stack of free PFNs (offset 0x10)
    uint32_t free_count;      // Free list size (offset 0x18, 4 bytes)
} physical_memory_t;
/*
 * Translation Statistics
 * 
 * Counters for measuring simulator behavior.
 * All counters use 64-bit integers to handle long traces without overflow.
 * 
 * WHY each field exists:
 * - total_accesses:    Denominator for fault rate calculations.
 * - page_faults:       Count of valid=0 encounters (demand paging events).
 * - protection_faults: Count of permission violations (segfault equivalents).
 * - frames_used:       Current memory utilization.
 */
typedef struct {
    uint64_t total_accesses;      // Total translation attempts
    uint64_t page_faults;         // valid=0 faults (demand paging)
    uint64_t protection_faults;   // Permission violations
    uint64_t frames_used;         // Currently allocated frames
} stats_t;
/*
 * Memory Access (parsed from trace)
 * 
 * Represents a single memory operation from the trace file.
 */
typedef struct {
    bool is_write;               // true = write, false = read
    uint32_t virtual_address;    // The VA being accessed
} memory_access_t;
/*
 * Translation Result Codes
 * 
 * Enumerates all possible outcomes of a translation attempt.
 */
typedef enum {
    TRANS_OK,                    // Success: physical address valid
    TRANS_PAGE_FAULT,            // valid=0: page not in memory
    TRANS_PROTECTION_FAULT       // Permission violation
} trans_result_t;
/*
 * Translation Output
 * 
 * Bundles result code with the physical address (if successful).
 */
typedef struct {
    trans_result_t result;       // What happened?
    uint32_t physical_address;   // Valid only when result == TRANS_OK
} trans_output_t;
/*
 * Simulator State (top-level structure)
 * 
 * Bundles all simulator components into one object.
 * This is the "context" passed to most functions.
 */
typedef struct {
    pte_t *page_table;                // Flat array of PTEs, indexed by VPN
    physical_memory_t phys_mem;       // Frame pool
    stats_t stats;                    // Counters
} simulator_t;
#endif // TYPES_H
```
### 3.3 Memory Layout Diagram
```
Virtual Address Space (32-bit, 4 GB):
┌─────────────────────────────────────────────────────────────┐
│ VPN 0x00000: 0x00000000 - 0x00000FFF (Page 0)               │
├─────────────────────────────────────────────────────────────┤
│ VPN 0x00001: 0x00001000 - 0x00001FFF (Page 1)               │
├─────────────────────────────────────────────────────────────┤
│ VPN 0x00002: 0x00002000 - 0x00002FFF (Page 2)               │
├─────────────────────────────────────────────────────────────┤
│ ...                                                         │
├─────────────────────────────────────────────────────────────┤
│ VPN 0xFFFFE: 0xFFFFF000 - 0xFFFFEFFF (Page 1,048,574)       │
├─────────────────────────────────────────────────────────────┤
│ VPN 0xFFFFF: 0xFFFFF000 - 0xFFFFFFFF (Page 1,048,575)       │
└─────────────────────────────────────────────────────────────┘
         │
         │ Page Table Lookup: page_table[vpn]
         ▼
┌─────────────────────────────────────────────────────────────┐
│ PTE: [PFN=0x42 | valid=1 | readable=1 | writable=1 |        │
│       dirty=0 | referenced=1]                               │
└─────────────────────────────────────────────────────────────┘
         │
         │ Physical address = (PFN << 12) | offset
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Physical Memory (simplified view):                          │
│ ┌─────────────────┐                                         │
│ │ Frame 0x00      │ ← free_list[0]                          │
│ ├─────────────────┤                                         │
│ │ Frame 0x01      │ ← free_list[1]                          │
│ ├─────────────────┤                                         │
│ │ ...             │                                         │
│ ├─────────────────┤                                         │
│ │ Frame 0x42      │ ← Contains VPN 0x00002's data           │
│ ├─────────────────┤                                         │
│ │ ...             │                                         │
│ ├─────────────────┤                                         │
│ │ Frame 0xFF      │ ← free_list[n-1]                        │
│ └─────────────────┘                                         │
└─────────────────────────────────────────────────────────────┘
```
---
## 4. Interface Contracts
### 4.1 Simulator Lifecycle
```c
/*
 * simulator_create
 * 
 * Creates and initializes a new simulator instance.
 * 
 * Parameters:
 *   num_frames - Number of physical frames to allocate.
 *                Must be > 0 and <= MAX_VPN.
 *                DEFAULT_NUM_FRAMES (64) is a good starting point.
 * 
 * Returns:
 *   Pointer to initialized simulator_t on success.
 *   NULL on failure (memory allocation error or invalid parameters).
 * 
 * Postconditions:
 *   - page_table is allocated with MAX_VPN entries, all valid=false
 *   - phys_mem.frames is allocated with num_frames slots, all NULL
 *   - phys_mem.free_list contains all frame indices [0, num_frames)
 *   - phys_mem.free_count == num_frames
 *   - All stats counters are zero
 * 
 * Memory ownership:
 *   Caller owns returned pointer and must free with simulator_destroy().
 */
simulator_t* simulator_create(uint32_t num_frames);
/*
 * simulator_destroy
 * 
 * Releases all resources associated with a simulator.
 * 
 * Parameters:
 *   sim - Pointer to simulator created by simulator_create().
 *         May be NULL (no-op in that case).
 * 
 * Side effects:
 *   - Frees page_table array
 *   - Frees all allocated frame data (frames[i] where non-NULL)
 *   - Frees phys_mem.frames array
 *   - Frees phys_mem.free_list array
 *   - Frees simulator_t struct itself
 * 
 * Postconditions:
 *   - sim pointer is invalid after this call (dangling!)
 *   - All memory returned to heap
 */
void simulator_destroy(simulator_t *sim);
```
### 4.2 Address Translation
```c
/*
 * translate_address
 * 
 * Performs virtual-to-physical address translation with all checks.
 * 
 * This is the core function of the simulator. It implements the full
 * translation pipeline:
 *   1. Extract VPN and offset from virtual address
 *   2. Check PTE valid bit (page fault if 0)
 *   3. Check permissions (protection fault if violated)
 *   4. Update access bits (dirty on write, referenced always)
 *   5. Compose physical address
 * 
 * Parameters:
 *   sim       - Simulator state (page table, physical memory, stats)
 *   va        - Virtual address to translate (32-bit)
 *   is_write  - true if this is a write operation, false for read
 * 
 * Returns:
 *   trans_output_t struct with:
 *   - result: TRANS_OK, TRANS_PAGE_FAULT, or TRANS_PROTECTION_FAULT
 *   - physical_address: Valid only when result == TRANS_OK
 * 
 * Statistics updates (side effects):
 *   - stats->total_accesses always incremented
 *   - stats->page_faults incremented on valid=0
 *   - stats->protection_faults incremented on permission violation
 *   - stats->frames_used incremented when frame allocated (demand paging)
 * 
 * PTE mutations (side effects):
 *   - pte->valid set to true on demand paging
 *   - pte->referenced set to true on any access
 *   - pte->dirty set to true on write access
 *   - pte->pfn set to allocated frame on demand paging
 * 
 * Error handling:
 *   - VPN out of bounds: Treated as protection fault
 *   - No free frames: Page fault returned, no allocation occurs
 *     (In Milestone 4, this triggers replacement instead)
 * 
 * Invariants preserved:
 *   - PTE with valid=false has meaningless pfn
 *   - Allocated frames are tracked in frames_used
 *   - Free list accurately reflects available frames
 */
trans_output_t translate_address(simulator_t *sim, uint32_t va, bool is_write);
```
### 4.3 Frame Allocation (Internal)
```c
/*
 * allocate_frame
 * 
 * Allocates a physical frame from the free pool.
 * 
 * Parameters:
 *   sim - Simulator with phys_mem.free_list and free_count
 * 
 * Returns:
 *   Frame index (PFN) on success, in range [0, total_frames)
 *   -1 (cast to int32_t) if no frames available
 * 
 * Side effects:
 *   - Decrements sim->phys_mem.free_count
 *   - Allocates sim->phys_mem.frames[pfn] with calloc(PAGE_SIZE, 1)
 *   - Frame data is zero-initialized
 * 
 * Precondition: sim != NULL
 * Postcondition: If successful, frames[pfn] is non-NULL and zeroed
 */
int32_t allocate_frame(simulator_t *sim);
/*
 * handle_page_fault
 * 
 * Demand paging handler. Called when valid=0 encountered.
 * 
 * Parameters:
 *   sim   - Simulator state
 *   vpn   - Virtual page number that faulted
 * 
 * Returns:
 *   true if fault was handled (frame allocated, PTE updated)
 *   false if no frames available
 * 
 * Side effects on success:
 *   - Allocates a frame via allocate_frame()
 *   - Sets pte->pfn to allocated frame
 *   - Sets pte->valid = true
 *   - Sets pte->readable = true, pte->writable = true (simplified)
 *   - Sets pte->dirty = false, pte->referenced = false
 *   - Increments sim->stats.frames_used
 * 
 * Note: In this milestone, all demand-paged pages are RW.
 *       Protection is only set explicitly for testing.
 */
bool handle_page_fault(simulator_t *sim, uint32_t vpn);
```
### 4.4 Trace Parsing
```c
/*
 * parse_trace_file
 * 
 * Reads and processes an entire trace file.
 * 
 * Parameters:
 *   sim      - Simulator state
 *   filename - Path to trace file
 * 
 * Returns:
 *   true on success (file processed completely)
 *   false on error (file not found, parse error)
 * 
 * Trace file format:
 *   - One access per line
 *   - Format: "R 0xADDRESS" or "W 0xADDRESS"
 *   - R = read, W = write
 *   - ADDRESS is hexadecimal (case-insensitive)
 *   - Lines starting with '#' are comments (ignored)
 *   - Empty lines are ignored
 *   - Whitespace around tokens is ignored
 * 
 * Example trace:
 *   # This is a comment
 *   R 0x00001000    # First access
 *   W 0x00001004    # Write to same page
 *   R 0x00002000    # Different page
 * 
 * Side effects:
 *   - Calls translate_address() for each parsed access
 *   - Updates all statistics
 *   - May allocate frames via demand paging
 *   - Prints translation results to stdout (verbose mode)
 * 
 * Error handling:
 *   - File not found: Prints error to stderr, returns false
 *   - Malformed line: Prints warning to stderr, skips line, continues
 *   - Out of memory: Prints error, returns false
 */
bool parse_trace_file(simulator_t *sim, const char *filename);
/*
 * parse_trace_line
 * 
 * Parses a single trace line into a memory_access_t.
 * 
 * Parameters:
 *   line   - Null-terminated string (one line from trace file)
 *   access - Output struct to populate
 * 
 * Returns:
 *   true on successful parse
 *   false if line is malformed or is a comment/blank
 * 
 * Precondition: line != NULL, access != NULL
 * Postcondition: On success, access fields are populated
 */
bool parse_trace_line(const char *line, memory_access_t *access);
```
### 4.5 Statistics
```c
/*
 * stats_init
 * 
 * Initializes statistics structure to all zeros.
 */
void stats_init(stats_t *stats);
/*
 * stats_print
 * 
 * Prints formatted statistics report to stdout.
 * 
 * Output format:
 *   ========================================
 *            SIMULATION STATISTICS
 *   ========================================
 *   Total memory accesses:  <count>
 *   Page faults:            <count>
 *   Protection faults:      <count>
 *   Page fault rate:        <percentage>%
 *   Frames used:            <used> / <total>
 *   ========================================
 */
void stats_print(const simulator_t *sim);
```
---
## 5. Algorithm Specification
### 5.1 Address Decomposition
```
ALGORITHM: decompose_virtual_address
INPUT:  va - 32-bit virtual address
OUTPUT: vpn - Virtual Page Number (bits 31-12)
        offset - Byte offset within page (bits 11-0)
PROCEDURE:
  1. vpn ← va >> PAGE_SHIFT
     // Right shift by 12 discards offset bits, leaving VPN
     // Example: 0x00001234 >> 12 = 0x1
  2. offset ← va AND PAGE_MASK
     // Mask with 0xFFF extracts bottom 12 bits
     // Example: 0x00001234 AND 0xFFF = 0x234
  3. RETURN (vpn, offset)
INVARIANTS:
  - vpn ∈ [0, MAX_VPN-1] for valid 32-bit addresses
  - offset ∈ [0, PAGE_SIZE-1]
  - va == (vpn << PAGE_SHIFT) | offset (recomposition property)
EDGE CASES:
  - va = 0x00000000 → vpn = 0, offset = 0
  - va = 0xFFFFF000 → vpn = 0xFFFFF, offset = 0
  - va = 0xFFFFFFFF → vpn = 0xFFFFF, offset = 0xFFF
```
### 5.2 Translation Pipeline
```
ALGORITHM: translate_address
INPUT:  sim - Simulator state
        va - Virtual address to translate
        is_write - Boolean, true for write operation
OUTPUT: trans_output_t with result code and physical address
PROCEDURE:
  1. INCREMENT sim->stats.total_accesses
  2. // Decompose virtual address
     vpn ← va >> PAGE_SHIFT
     offset ← va AND PAGE_MASK
  3. // Bounds check on VPN
     IF vpn >= MAX_VPN THEN
       INCREMENT sim->stats.protection_faults
       RETURN {TRANS_PROTECTION_FAULT, 0}
     END IF
  4. // Get PTE pointer
     pte ← ADDRESSOF sim->page_table[vpn]
  5. // Check valid bit (demand paging)
     IF pte->valid == false THEN
       INCREMENT sim->stats.page_faults
       // Attempt to handle fault by allocating frame
       success ← handle_page_fault(sim, vpn)
       IF success == false THEN
         // No frames available
         RETURN {TRANS_PAGE_FAULT, 0}
       END IF
       // PTE is now valid, re-fetch
       pte ← ADDRESSOF sim->page_table[vpn]
     END IF
  6. // Check permissions
     IF is_write == true AND pte->writable == false THEN
       INCREMENT sim->stats.protection_faults
       RETURN {TRANS_PROTECTION_FAULT, 0}
     END IF
     IF is_write == false AND pte->readable == false THEN
       INCREMENT sim->stats.protection_faults
       RETURN {TRANS_PROTECTION_FAULT, 0}
     END IF
  7. // Update access bits
     pte->referenced ← true
     IF is_write == true THEN
       pte->dirty ← true
     END IF
  8. // Compose physical address
     physical_address ← (pte->pfn << PAGE_SHIFT) OR offset
  9. RETURN {TRANS_OK, physical_address}
INVARIANTS AFTER EXECUTION:
  - If result == TRANS_OK:
    * pte->valid == true
    * pte->referenced == true
    * pte->dirty == true (if is_write was true)
    * physical_address is valid
  - If result == TRANS_PAGE_FAULT:
    * pte->valid == false (unless handle_page_fault succeeded but then failed)
  - If result == TRANS_PROTECTION_FAULT:
    * No state changes to PTE
```
### 5.3 Frame Allocation
```
ALGORITHM: allocate_frame
INPUT:  sim - Simulator with phys_mem
OUTPUT: pfn - Physical Frame Number, or -1 if exhausted
PROCEDURE:
  1. IF sim->phys_mem.free_count == 0 THEN
       RETURN -1  // No frames available
     END IF
  2. // Pop from free list (stack discipline)
     sim->phys_mem.free_count ← sim->phys_mem.free_count - 1
     pfn ← sim->phys_mem.free_list[free_count]
  3. // Allocate frame data
     sim->phys_mem.frames[pfn] ← calloc(PAGE_SIZE, 1)
     IF sim->phys_mem.frames[pfn] == NULL THEN
       // Memory allocation failed - push back to free list
       sim->phys_mem.free_list[free_count] ← pfn
       sim->phys_mem.free_count ← free_count + 1
       RETURN -1
     END IF
  4. RETURN pfn
POSTCONDITIONS:
  - frames[pfn] points to PAGE_SIZE bytes of zeroed memory
  - pfn is no longer in free_list (until explicitly freed)
```
### 5.4 Demand Paging Handler
```
ALGORITHM: handle_page_fault
INPUT:  sim - Simulator state
        vpn - Virtual page number that faulted
OUTPUT: Boolean success/failure
PROCEDURE:
  1. // Attempt frame allocation
     pfn ← allocate_frame(sim)
     IF pfn == -1 THEN
       RETURN false  // Out of memory
     END IF
  2. // Get PTE
     pte ← ADDRESSOF sim->page_table[vpn]
  3. // Update PTE with new mapping
     pte->pfn ← pfn
     pte->valid ← true
     pte->readable ← true   // Simplified: all pages RW
     pte->writable ← true
     pte->dirty ← false
     pte->referenced ← false
  4. // Update statistics
     sim->stats.frames_used ← sim->stats.frames_used + 1
  5. RETURN true
POSTCONDITIONS:
  - pte->valid == true
  - pte->pfn points to an allocated frame
  - Frame data is zero-initialized
```
### 5.5 Trace Line Parsing
```
ALGORITHM: parse_trace_line
INPUT:  line - Null-terminated string
        access - Output struct
OUTPUT: Boolean success/failure
PROCEDURE:
  1. // Skip leading whitespace
     WHILE line[0] == ' ' OR line[0] == '\t' DO
       line ← line + 1
     END WHILE
  2. // Skip comments and empty lines
     IF line[0] == '#' OR line[0] == '\n' OR line[0] == '\0' THEN
       RETURN false  // Not an error, just not an access
     END IF
  3. // Parse operation type and address
     // Expected format: "R 0x1234" or "W 0xABCD"
     result ← sscanf(line, "%c 0x%X", &operation, &address)
     IF result != 2 THEN
       // Try parsing without 0x prefix
       result ← sscanf(line, "%c %X", &operation, &address)
       IF result != 2 THEN
         RETURN false  // Parse error
       END IF
     END IF
  4. // Convert operation to is_write boolean
     access->is_write ← (operation == 'W' OR operation == 'w')
     access->virtual_address ← address
  5. RETURN true
EDGE CASES:
  - "r 0x1000" → treated as read (lowercase accepted)
  - "w 0x1000" → treated as write (lowercase accepted)
  - "X 0x1000" → parse error (invalid operation)
  - "R 0x" → parse error (missing address)
  - "R 0x1000 extra" → parsed successfully (extra ignored)
```
---
## 6. Error Handling Matrix
| Error | Detected By | Recovery | User-Visible? | System State |
|-------|-------------|----------|---------------|--------------|
| VPN out of bounds | `translate_address` step 3 | Return `TRANS_PROTECTION_FAULT` | Yes, in statistics | Unchanged |
| Page not valid (valid=0) | `translate_address` step 5 | Call `handle_page_fault`, allocate frame | Yes, "Page Fault" log | PTE updated, frame allocated |
| No free frames | `allocate_frame` step 1 | Return -1, propagate as `TRANS_PAGE_FAULT` | Yes, "Out of memory" error | Unchanged |
| Write to read-only page | `translate_address` step 6 | Return `TRANS_PROTECTION_FAULT` | Yes, "Protection Fault" log | Unchanged |
| Read from non-readable page | `translate_address` step 6 | Return `TRANS_PROTECTION_FAULT` | Yes, "Protection Fault" log | Unchanged |
| Trace file not found | `parse_trace_file` | Print error to stderr, return false | Yes, error message | Simulator unchanged |
| Malformed trace line | `parse_trace_line` | Print warning, skip line, continue | Yes, warning message | Line skipped |
| Memory allocation failure | `allocate_frame` step 3 | Push frame back to free list, return -1 | Yes, error message | Free list restored |
| NULL simulator pointer | Any function | Check at entry, return error | Yes, assertion or error | N/A |
---
## 7. Implementation Sequence with Checkpoints
### Phase 1: Data Structures (1-2 hours)
**Files:** `include/config.h`, `include/types.h`
**Tasks:**
1. Create `config.h` with PAGE_SHIFT, PAGE_SIZE, PAGE_MASK, MAX_VPN
2. Create `types.h` with pte_t, physical_memory_t, stats_t, memory_access_t
3. Create trans_result_t enum and trans_output_t struct
4. Create simulator_t as top-level container
**Checkpoint:**
```bash
# Compile header files only (check for syntax errors)
gcc -c -fsyntax-only include/config.h include/types.h
# Expected: No errors or warnings
```
---
### Phase 2: Address Decomposition (0.5-1 hour)
**Files:** `src/translate.c`
**Tasks:**
1. Implement inline helper functions for VPN/offset extraction
2. Implement recomposition function for testing
**Code to implement:**
```c
static inline uint32_t extract_vpn(uint32_t va) {
    return va >> PAGE_SHIFT;
}
static inline uint32_t extract_offset(uint32_t va) {
    return va & PAGE_MASK;
}
static inline uint32_t compose_physical(uint32_t pfn, uint32_t offset) {
    return (pfn << PAGE_SHIFT) | offset;
}
```
**Checkpoint:**
```c
// Add to tests/test_basic.c
void test_address_decomposition(void) {
    // Test case 1: 0x00001234
    assert(extract_vpn(0x00001234) == 0x1);
    assert(extract_offset(0x00001234) == 0x234);
    // Test case 2: Last page, first byte
    assert(extract_vpn(0xFFFFF000) == 0xFFFFF);
    assert(extract_offset(0xFFFFF000) == 0x0);
    // Test case 3: First page, last byte
    assert(extract_vpn(0x00000FFF) == 0x0);
    assert(extract_offset(0x00000FFF) == 0xFFF);
    // Test case 4: Recomposition
    assert(compose_physical(0x42, 0x234) == 0x420234);
    printf("Address decomposition tests PASSED\n");
}
```
```bash
gcc -o test_basic tests/test_basic.c src/translate.c -I include
./test_basic
# Expected: "Address decomposition tests PASSED"
```
---
### Phase 3: Trace Parser (1-2 hours)
**Files:** `include/parser.h`, `src/parser.c`
**Tasks:**
1. Implement `parse_trace_line` for single-line parsing
2. Implement `parse_trace_file` for file processing
3. Handle comments, blank lines, whitespace
**Checkpoint:**
```c
// Add to tests/test_basic.c
void test_trace_parsing(void) {
    memory_access_t access;
    // Test read operation
    assert(parse_trace_line("R 0x1000", &access) == true);
    assert(access.is_write == false);
    assert(access.virtual_address == 0x1000);
    // Test write operation
    assert(parse_trace_line("W 0x2000", &access) == true);
    assert(access.is_write == true);
    assert(access.virtual_address == 0x2000);
    // Test comment (should return false)
    assert(parse_trace_line("# comment", &access) == false);
    // Test blank line (should return false)
    assert(parse_trace_line("", &access) == false);
    printf("Trace parsing tests PASSED\n");
}
```
```bash
gcc -o test_parser tests/test_basic.c src/parser.c -I include
./test_parser
# Expected: "Trace parsing tests PASSED"
```
---
### Phase 4: Translation Function (2-3 hours)
**Files:** `src/translate.c`, `include/translate.h`
**Tasks:**
1. Implement `simulator_create` and `simulator_destroy`
2. Implement `allocate_frame`
3. Implement `handle_page_fault`
4. Implement `translate_address` with all checks
**Checkpoint:**
```c
// Add to tests/test_translation.c
void test_translation_basic(void) {
    simulator_t *sim = simulator_create(64);
    assert(sim != NULL);
    // First access to VPN 1 should trigger page fault
    trans_output_t result = translate_address(sim, 0x1000, false);
    assert(result.result == TRANS_OK);  // Demand paging handles it
    assert(sim->stats.page_faults == 1);
    // Second access to same page should NOT fault
    result = translate_address(sim, 0x1004, false);
    assert(result.result == TRANS_OK);
    assert(sim->stats.page_faults == 1);  // Unchanged
    // Physical addresses should have same PFN, different offset
    uint32_t pa1 = (result.result == TRANS_OK) ? result.physical_address : 0;
    result = translate_address(sim, 0x1000, false);
    uint32_t pa2 = (result.result == TRANS_OK) ? result.physical_address : 0;
    assert((pa1 >> PAGE_SHIFT) == (pa2 >> PAGE_SHIFT));
    simulator_destroy(sim);
    printf("Basic translation tests PASSED\n");
}
```
```bash
gcc -o test_translation tests/test_translation.c src/translate.c -I include
./test_translation
# Expected: "Basic translation tests PASSED"
```
---
### Phase 5: Statistics (0.5-1 hour)
**Files:** `src/stats.c`, `include/stats.h`
**Tasks:**
1. Implement `stats_init`
2. Implement `stats_print`
**Checkpoint:**
```c
void test_statistics(void) {
    simulator_t *sim = simulator_create(64);
    // Perform some accesses
    translate_address(sim, 0x1000, false);  // Page fault
    translate_address(sim, 0x1000, true);   // Hit, sets dirty
    translate_address(sim, 0x2000, false);  // Page fault
    assert(sim->stats.total_accesses == 3);
    assert(sim->stats.page_faults == 2);
    assert(sim->stats.frames_used == 2);
    stats_print(sim);
    simulator_destroy(sim);
    printf("Statistics tests PASSED\n");
}
```
---
### Phase 6: Main Entry Point (0.5-1 hour)
**Files:** `src/main.c`, `Makefile`
**Tasks:**
1. Parse command-line arguments (trace file, frame count)
2. Create simulator, run trace, print stats
3. Create Makefile with targets: all, clean, test
**Checkpoint:**
```bash
make
./vm_sim traces/basic.trace 64
# Expected: Statistics output showing page faults and hit rates
```
---
### Phase 7: Integration Testing (1-2 hours)
**Files:** `traces/basic.trace`, `traces/dirty.trace`, `traces/protection.trace`
**Tasks:**
1. Create test traces
2. Run full simulation on each trace
3. Verify statistics match expected values
**Checkpoint:**
```bash
# Run all tests
make test
# Expected: All tests pass
```
---
## 8. Test Specification
### 8.1 Unit Tests for Address Decomposition
| Test Case | Input | Expected VPN | Expected Offset |
|-----------|-------|--------------|-----------------|
| First page, first byte | 0x00000000 | 0x00000 | 0x000 |
| First page, last byte | 0x00000FFF | 0x00000 | 0xFFF |
| Second page, first byte | 0x00001000 | 0x00001 | 0x000 |
| Arbitrary address | 0x00001234 | 0x00001 | 0x234 |
| Last page, first byte | 0xFFFFF000 | 0xFFFFF | 0x000 |
| Last page, last byte | 0xFFFFFFFF | 0xFFFFF | 0xFFF |
### 8.2 Unit Tests for Translation
| Test Case | Initial State | Input | Expected Result | Expected Stats Change |
|-----------|---------------|-------|-----------------|----------------------|
| First access to page | PTE valid=false | VA=0x1000, read | TRANS_OK (demand paging) | page_faults++, frames_used++ |
| Second access to page | PTE valid=true | VA=0x1000, read | TRANS_OK | no change |
| Write sets dirty | PTE valid=true, dirty=false | VA=0x1000, write | TRANS_OK | dirty bit set |
| Read doesn't set dirty | PTE valid=true, dirty=false | VA=0x1000, read | TRANS_OK | dirty bit unchanged |
| Protection fault | PTE valid=true, writable=false | VA=0x1000, write | TRANS_PROTECTION_FAULT | protection_faults++ |
| Out of memory | All frames used | VA=new_page | TRANS_PAGE_FAULT | page_faults++ (no frame allocated) |
### 8.3 Unit Tests for Trace Parsing
| Test Case | Input Line | Expected Result | access.is_write | access.virtual_address |
|-----------|------------|-----------------|-----------------|------------------------|
| Valid read | "R 0x1000" | true | false | 0x1000 |
| Valid write | "W 0x2000" | true | true | 0x2000 |
| Lowercase op | "r 0x1000" | true | false | 0x1000 |
| Comment line | "# comment" | false | - | - |
| Empty line | "" | false | - | - |
| Whitespace line | "   " | false | - | - |
| Missing address | "R" | false | - | - |
| Invalid format | "X 0x1000" | false | - | - |
### 8.4 Integration Tests
**Test Trace 1: Basic Demand Paging (`traces/basic.trace`)**
```
# Sequential access to 3 pages
R 0x00001000
R 0x00002000
R 0x00003000
# Re-access same pages
R 0x00001000
R 0x00002000
R 0x00003000
```
**Expected Statistics:**
- total_accesses: 6
- page_faults: 3
- protection_faults: 0
- frames_used: 3
**Test Trace 2: Dirty Bit Tracking (`traces/dirty.trace`)**
```
# Read doesn't set dirty
R 0x00001000
# Write sets dirty
W 0x00001000
# Read again
R 0x00001000
# Write to new page
W 0x00002000
```
**Expected Statistics:**
- total_accesses: 4
- page_faults: 2
- VPN 1: dirty=true
- VPN 2: dirty=true
**Test Trace 3: Protection Faults (`traces/protection.trace`)**
```
# Requires manual PTE modification in test code
# After setting page 1 read-only:
W 0x00001000
```
**Expected Statistics:**
- protection_faults: 1
---
## 9. Performance Targets
| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| Address decomposition (VPN + offset) | < 10 CPU cycles | Inline functions, compiler optimization |
| PTE lookup | O(1), ~5 cycles | Array index, single memory access |
| Translation (valid page, hit) | < 50 cycles | Includes bit extraction, checks, composition |
| Translation (page fault) | ~500 cycles | Excludes frame allocation time |
| Frame allocation | ~100 cycles | Stack pop + calloc |
| Trace line parsing | ~100 ns per line | sscanf overhead dominates |
| Memory overhead (page table) | 4 MB per process | MAX_VPN × sizeof(pte_t) ≈ 1M × 4 bytes |
**Cache Behavior Analysis:**
- Sequential VPN access: Prefetch-friendly. Each PTE is 12 bytes, so 5 PTEs per 64-byte cache line.
- Random VPN access: Cache-hostile. Each access may miss L1, L2.
- PTE access pattern mirrors trace access pattern.
---
## 10. State Machine (PTE Lifecycle)
```
                    ┌─────────────────────────────────────┐
                    │          PTE STATES                  │
                    └─────────────────────────────────────┘
     ┌──────────────────┐
     │    INVALID       │  valid=false, pfn=undefined
     │   (unallocated)  │
     └────────┬─────────┘
              │
              │ handle_page_fault() allocates frame
              │ (demand paging on first access)
              ▼
     ┌──────────────────┐
     │    VALID-CLEAN   │  valid=true, dirty=false
     │   (in memory)    │
     └────────┬─────────┘
              │
              │ Write access
              │ (sets dirty bit)
              ▼
     ┌──────────────────┐
     │    VALID-DIRTY   │  valid=true, dirty=true
     │   (modified)     │
     └──────────────────┘
              │
              │ (In Milestone 4: eviction to swap)
              │
              ▼
     ┌──────────────────┐
     │    INVALID       │  valid=false (back to start)
     │   (evicted)      │
     └──────────────────┘
TRANSITIONS:
  INVALID → VALID-CLEAN:  First access (page fault)
  VALID-CLEAN → VALID-DIRTY: Write access
  VALID-DIRTY → INVALID: Eviction (Milestone 4)
ILLEGAL TRANSITIONS:
  VALID → VALID with different PFN (would corrupt data)
  dirty=true → dirty=false (dirty bit is monotonic until eviction)
```
---
## 11. Makefile
```makefile
CC = gcc
CFLAGS = -Wall -Wextra -Werror -std=c11 -g -O0
INCLUDES = -I include
SRCS = src/translate.c src/parser.c src/stats.c src/main.c
OBJS = $(SRCS:.c=.o)
TARGET = vm_sim
TEST_SRCS = tests/test_basic.c tests/test_translation.c
TEST_TARGETS = test_basic test_translation
.PHONY: all clean test
all: $(TARGET)
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<
test_basic: tests/test_basic.c src/translate.c src/parser.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^
test_translation: tests/test_translation.c src/translate.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^
test: test_basic test_translation
	./test_basic
	./test_translation
	./$(TARGET) traces/basic.trace 64
clean:
	rm -f $(OBJS) $(TARGET) $(TEST_TARGETS)
```
---
## 12. Hardware Soul: Cache and Branch Analysis
### 12.1 Cache Lines Touched Per Operation
| Operation | Cache Lines | Hot/Cold | Notes |
|-----------|-------------|----------|-------|
| PTE lookup | 1 | Depends on access pattern | 12-byte PTE fits in 64-byte line with 4 neighbors |
| Frame data access | 1 | Depends | Zeroed frames are cold initially |
| Stats update | 1 | Hot | Same cache line touched repeatedly |
| Free list pop | 1 | Hot | Stack top is cache-resident |
### 12.2 Branch Prediction
| Branch | Predictability | Misprediction Cost | Frequency |
|--------|----------------|-------------------|-----------|
| `if (!pte->valid)` | Highly predictable | 15 cycles | Most pages valid after warmup |
| `if (is_write && !pte->writable)` | Highly predictable | 15 cycles | Most writes to writable pages |
| `if (free_count == 0)` | Highly predictable | 15 cycles | Rarely true until memory full |
### 12.3 Memory Access Pattern
- **Sequential trace**: VPNs increase monotonically → sequential PTE access → prefetcher friendly
- **Random trace**: Random VPNs → random PTE access → cache hostile, high miss rate
- **Looping trace**: Repeated VPNs → temporal locality → cache hits after warmup
---
## 13. Sample Implementation: translate_address
```c
#include "types.h"
#include "config.h"
#include "translate.h"
trans_output_t translate_address(simulator_t *sim, uint32_t va, bool is_write) {
    trans_output_t out = {0};
    // Step 1: Count access
    sim->stats.total_accesses++;
    // Step 2: Decompose virtual address
    uint32_t vpn = va >> PAGE_SHIFT;
    uint32_t offset = va & PAGE_MASK;
    // Step 3: Bounds check
    if (vpn >= MAX_VPN) {
        sim->stats.protection_faults++;
        out.result = TRANS_PROTECTION_FAULT;
        return out;
    }
    // Step 4: Get PTE
    pte_t *pte = &sim->page_table[vpn];
    // Step 5: Check valid bit
    if (!pte->valid) {
        sim->stats.page_faults++;
        // Demand paging: allocate frame
        if (!handle_page_fault(sim, vpn)) {
            // Out of memory
            out.result = TRANS_PAGE_FAULT;
            return out;
        }
        // Re-fetch PTE after modification
        pte = &sim->page_table[vpn];
    }
    // Step 6: Check permissions
    if (is_write && !pte->writable) {
        sim->stats.protection_faults++;
        out.result = TRANS_PROTECTION_FAULT;
        return out;
    }
    if (!is_write && !pte->readable) {
        sim->stats.protection_faults++;
        out.result = TRANS_PROTECTION_FAULT;
        return out;
    }
    // Step 7: Update access bits
    pte->referenced = true;
    if (is_write) {
        pte->dirty = true;
    }
    // Step 8: Compose physical address
    out.physical_address = (pte->pfn << PAGE_SHIFT) | offset;
    out.result = TRANS_OK;
    return out;
}
```
---
[[CRITERIA_JSON: {"module_id": "virtual-memory-sim-m1", "criteria": ["Trace file parser correctly reads memory access sequences in format 'R/W 0xADDRESS' with comment (# prefix) and blank line handling", "Virtual address decomposition extracts VPN using right shift by PAGE_SHIFT (default 12) and offset using PAGE_MASK (0xFFF)", "Page table entry (pte_t) structure contains pfn (uint32_t), valid (bool), readable (bool), writable (bool), dirty (bool), and referenced (bool) fields", "Valid address translation returns physical address as concatenation of (pfn << PAGE_SHIFT) | offset", "Page fault is detected when valid=0 and triggers frame allocation with PTE update (demand paging)", "Protection fault is detected when permissions do not match operation (write to read-only page) and is logged separately from page faults", "Dirty bit is set only on write operations; referenced bit is set on any access (read or write)", "Statistics tracking (stats_t) includes total_accesses (uint64_t), page_faults (uint64_t), and protection_faults (uint64_t)", "Physical memory frame pool uses configurable frame count with free_list (uint32_t array) and free_count for O(1) frame allocation", "Page fault handler (handle_page_fault) allocates free frame via allocate_frame, updates PTE with pfn, sets valid=1, readable=1, writable=1", "Translation function (translate_address) returns trans_output_t with result enum (TRANS_OK, TRANS_PAGE_FAULT, TRANS_PROTECTION_FAULT) and physical_address field", "Statistics reporter (stats_print) outputs total accesses, page faults, protection faults, and page fault rate percentage to stdout"]}]
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: virtual-memory-sim-m2 -->
# Technical Design Document: TLB (Translation Lookaside Buffer)
**Module ID:** `virtual-memory-sim-m2`  
**Language:** C (BINDING)  
**Difficulty:** Intermediate → Advanced transition point
---
## 1. Module Charter
This module implements a Translation Lookaside Buffer (TLB) cache that accelerates virtual-to-physical address translation by caching recent translations. The TLB sits between address translation requests and the page table, returning the physical frame number (PFN) directly on a hit without consulting the page table.
**What it does:**
- Implements a fully-associative TLB with configurable entry count (16, 32, 64)
- Supports ASID (Address Space ID) tagging for context switch without full flush
- Implements LRU (Least Recently Used) victim selection using timestamp tracking
- Propagates dirty/referenced bits from TLB entries back to page table on eviction
- Provides flush operations: full flush, ASID-specific flush, single-entry invalidation
- Maintains coherency between TLB entries and page table entries
- Tracks TLB statistics: hits, misses, hit rate, flushes, evictions
**What it does NOT do:**
- Multi-level page table walks (Milestone 3)
- Page replacement when frames exhaust (Milestone 4)
- Hardware-style parallel associative lookup (simulated with linear scan)
**Upstream dependencies:** Milestone 1 (page table, PTE structure, translation pipeline)
**Downstream dependencies:** Milestone 3 (multi-level page tables increase TLB importance), Milestone 4 (replacement algorithms use referenced bits that TLB tracks)
**Invariants that must always hold:**
1. TLB entry with `valid=false` contains meaningless VPN/PFN data
2. A TLB entry's VPN+ASID combination is unique within the TLB (no duplicates)
3. Dirty bit in TLB implies page has been written to since TLB entry was created
4. Referenced bit in TLB implies page has been accessed since TLB entry was created
5. After eviction, dirty/referenced bits are propagated to page table before entry is invalidated
6. After page table modification (PTE.valid = false, permissions change), corresponding TLB entry must be invalidated
---
## 2. File Structure
Create files in this exact order:
```
01. include/tlb_config.h        — TLB-specific constants and configuration
02. include/tlb_types.h         — TLB entry, TLB structure, statistics
03. include/tlb.h               — TLB public interface declarations
04. src/tlb.c                   — TLB implementation (lookup, insert, evict, flush)
05. src/tlb_translate.c         — Translation function with TLB integration
06. tests/test_tlb_basic.c      — Unit tests for TLB lookup and insertion
07. tests/test_tlb_lru.c        — Unit tests for LRU victim selection
08. tests/test_tlb_coherency.c  — Unit tests for TLB-page table coherency
09. tests/test_tlb_context.c    — Unit tests for ASID context switching
10. traces/tlb_locality.trace   — Test trace for high locality (high hit rate)
11. traces/tlb_random.trace     — Test trace for random access (low hit rate)
12. traces/tlb_context.trace    — Test trace with context switches
13. Makefile                    — Updated build system with TLB targets
```
---
## 3. Complete Data Model
### 3.1 TLB Configuration Constants (`tlb_config.h`)
```c
#ifndef TLB_CONFIG_H
#define TLB_CONFIG_H
#include <stdint.h>
/*
 * TLB Configuration
 * 
 * These constants define the TLB's capacity and behavior.
 * Real x86-64 processors typically have:
 *   - L1 dTLB: 64 entries, 4-way set associative
 *   - L1 iTLB: 128 entries, 4-way set associative
 *   - L2 TLB: 1536 entries, unified
 * 
 * Our simulation uses a simpler fully-associative design.
 */
/* Default TLB capacity (number of entries) */
#define DEFAULT_TLB_CAPACITY    32
/* Maximum TLB capacity (for static allocation if needed) */
#define MAX_TLB_CAPACITY        256
/* ASID configuration */
#define ASID_BITS               8
#define MAX_ASID                ((1u << ASID_BITS) - 1)  // 255
#define ASID_INVALID            MAX_ASID
/* 
 * LRU timestamp type.
 * Using 64-bit timestamps to handle long traces without overflow.
 * At 1 access per nanosecond, overflow would take ~584 years.
 */
typedef uint64_t tlb_timestamp_t;
/* Initial timestamp value (prevents issues with zero-initialized entries) */
#define TLB_TIMESTAMP_INITIAL   1
#endif // TLB_CONFIG_H
```
### 3.2 TLB Type Definitions (`tlb_types.h`)
```c
#ifndef TLB_TYPES_H
#define TLB_TYPES_H
#include <stdint.h>
#include <stdbool.h>
#include "config.h"      // PAGE_SHIFT, PAGE_SIZE from Milestone 1
#include "tlb_config.h"
/*
 * TLB Entry Structure
 * 
 * Each entry caches a complete virtual-to-physical translation along with
 * permission bits and access tracking metadata.
 * 
 * Memory layout:
 * ┌────────────────────────────────────────────────────────────────────┐
 * │ vpn (4) │ pfn (4) │ asid (4) │ last_used (8) │ flags (5 bytes)     │
 * └────────────────────────────────────────────────────────────────────┘
 * 
 * WHY each field exists:
 * - vpn:         Virtual Page Number (tag). This is what we search for.
 * - pfn:         Physical Frame Number (value). This is what we return on hit.
 * - asid:        Address Space ID. Identifies which process owns this entry.
 *                Enables context switch without full TLB flush.
 * - valid:       Entry contains valid data. Invalid entries are skipped.
 * - readable:    Cached copy of PTE's read permission bit.
 * - writable:    Cached copy of PTE's write permission bit.
 * - dirty:       Set on write access. Must be written back to PTE on eviction.
 * - referenced:  Set on any access. Must be written back to PTE on eviction.
 * - last_used:   Timestamp for LRU eviction. Updated on every hit.
 * 
 * Total size: 4 + 4 + 4 + 8 + 5 = 25 bytes, padded to 32 bytes for alignment.
 * With 32-byte entries, a 64-entry TLB = 2 KB, fits comfortably in L1 cache.
 */
typedef struct {
    uint32_t vpn;              // Offset 0x00, 4 bytes - Virtual Page Number
    uint32_t pfn;              // Offset 0x04, 4 bytes - Physical Frame Number
    uint32_t asid;             // Offset 0x08, 4 bytes - Address Space ID
    tlb_timestamp_t last_used; // Offset 0x0C, 8 bytes - LRU timestamp
    // Permission and state flags (5 bytes)
    bool valid;                // Offset 0x14, 1 byte - Entry is valid?
    bool readable;             // Offset 0x15, 1 byte - Read permission
    bool writable;             // Offset 0x16, 1 byte - Write permission
    bool dirty;                // Offset 0x17, 1 byte - Written since load?
    bool referenced;           // Offset 0x18, 1 byte - Accessed since load?
    // Padding to 32 bytes for cache alignment
    uint8_t _padding[3];       // Offset 0x19, 3 bytes
} tlb_entry_t;                 // Total: 32 bytes
_Static_assert(sizeof(tlb_entry_t) == 32, "TLB entry must be 32 bytes");
/*
 * TLB Statistics
 * 
 * Performance counters for measuring TLB effectiveness.
 * These metrics are critical for understanding memory access patterns.
 */
typedef struct {
    uint64_t hits;             // TLB hits (fast path taken)
    uint64_t misses;           // TLB misses (page table walk required)
    uint64_t flushes;          // Full TLB flush operations
    uint64_t evictions;        // Entries evicted to make room
    uint64_t partial_flushes;  // ASID-specific or single-entry invalidations
    uint64_t dirty_writebacks; // Dirty bits propagated to page table
} tlb_stats_t;
/*
 * TLB Structure
 * 
 * Main TLB state container. Uses a simple array for fully-associative
 * lookup (linear scan). Real hardware uses content-addressable memory
 * (CAM) for parallel lookup, but we simulate with sequential search.
 * 
 * WHY each field exists:
 * - entries:       Array of TLB entries. Fixed size at creation.
 * - capacity:      Number of entries in the array.
 * - current_asid:  ASID of the currently running process.
 * - clock:         Monotonic timestamp counter for LRU ordering.
 * - stats:         Performance counters.
 * - page_table:    Back-reference for dirty bit write-back.
 */
typedef struct {
    tlb_entry_t *entries;      // Offset 0x00, 8 bytes - Entry array
    uint32_t capacity;         // Offset 0x08, 4 bytes - Array size
    uint32_t current_asid;     // Offset 0x0C, 4 bytes - Active ASID
    tlb_timestamp_t clock;     // Offset 0x10, 8 bytes - LRU clock
    tlb_stats_t stats;         // Offset 0x18, 48 bytes - Statistics
    struct pte_t *page_table;  // Offset 0x48, 8 bytes - Page table ref
} tlb_t;
/*
 * TLB Lookup Result
 * 
 * Returned by tlb_lookup to indicate whether a translation was found.
 */
typedef enum {
    TLB_HIT,                   // Translation found in TLB
    TLB_MISS                   // Translation not in TLB
} tlb_lookup_result_t;
/*
 * TLB Translation Result
 * 
 * Extended result type for the full translation pipeline.
 */
typedef enum {
    TLB_TRANS_OK,              // Success: physical address valid
    TLB_TRANS_MISS,            // TLB miss, but successfully resolved via page table
    TLB_TRANS_PAGE_FAULT,      // Page fault (PTE valid=0)
    TLB_TRANS_PROTECTION_FAULT // Permission violation
} tlb_trans_result_t;
/*
 * TLB Translation Output
 * 
 * Bundles result code with physical address and TLB hit/miss information.
 */
typedef struct {
    tlb_trans_result_t result; // What happened?
    uint32_t physical_address; // Valid when result == TLB_TRANS_OK or TLB_TRANS_MISS
    bool tlb_hit;              // Did this come from TLB (true) or page table (false)?
} tlb_trans_output_t;
#endif // TLB_TYPES_H
```
### 3.3 Memory Layout Diagram
```
TLB Structure (32 entries, 1 KB total):
┌─────────────────────────────────────────────────────────────────────┐
│                          TLB State                                  │
├─────────────────────────────────────────────────────────────────────┤
│ entries ──────────────┐                                             │
│ capacity: 32          │                                             │
│ current_asid: 0x01    │                                             │
│ clock: 1523           │                                             │
│ stats: {...}          │                                             │
│ page_table: ─────┐    │                                             │
└──────────────────┼────┼─────────────────────────────────────────────┘
                   │    │
                   │    └──────────────────────────────────────┐
                   │                                           │
                   ▼                                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        TLB Entry Array                              │
├─────────────────────────────────────────────────────────────────────┤
│ Entry 0:  [vpn=0x001 | pfn=0x42 | asid=0x01 | valid=1 | ref=1 | ...]│
├─────────────────────────────────────────────────────────────────────┤
│ Entry 1:  [vpn=0x002 | pfn=0x17 | asid=0x01 | valid=1 | ref=1 | ...]│
├─────────────────────────────────────────────────────────────────────┤
│ Entry 2:  [vpn=0x000 | pfn=0x00 | asid=0x00 | valid=0 | ...]        │
├─────────────────────────────────────────────────────────────────────┤
│ Entry 3:  [vpn=0x0FF | pfn=0x99 | asid=0x02 | valid=1 | ref=0 | ...]│
├─────────────────────────────────────────────────────────────────────┤
│ ...                                                                 │
├─────────────────────────────────────────────────────────────────────┤
│ Entry 31: [vpn=0x003 | pfn=0xAB | asid=0x01 | valid=1 | dirty=1]    │
└─────────────────────────────────────────────────────────────────────┘
Lookup Flow:
                    Virtual Address
                          │
                          ▼
            ┌─────────────────────────┐
            │   Extract VPN (20 bits) │
            └────────────┬────────────┘
                         │
                         ▼
    ┌────────────────────────────────────────────────┐
    │              TLB Lookup (Linear Scan)          │
    │                                                │
    │  FOR each entry i:                             │
    │    IF entry[i].valid AND                       │
    │       entry[i].vpn == search_vpn AND           │
    │       entry[i].asid == current_asid:           │
    │      → HIT: return entry[i].pfn                │
    │                                                │
    │  → MISS: walk page table                       │
    └────────────────────────────────────────────────┘
                         │
           ┌─────────────┴─────────────┐
           │                           │
        HIT│                           │MISS
           ▼                           ▼
    ┌─────────────┐          ┌─────────────────────┐
    │ Return PFN  │          │   Page Table Walk   │
    │ Update LRU  │          │   Insert into TLB   │
    │ Set ref/dir │          │   Return PFN        │
    └─────────────┘          └─────────────────────┘
```
### 3.4 Integration with Milestone 1 Types
```c
/*
 * Forward declarations for Milestone 1 types.
 * These are defined in types.h from the previous module.
 */
typedef struct {
    uint32_t pfn;
    bool valid;
    bool readable;
    bool writable;
    bool dirty;
    bool referenced;
} pte_t;
typedef struct {
    uint64_t total_accesses;
    uint64_t page_faults;
    uint64_t protection_faults;
    uint64_t frames_used;
    // TLB statistics (new in Milestone 2)
    uint64_t tlb_hits;
    uint64_t tlb_misses;
} stats_t;
typedef struct {
    pte_t *page_table;
    physical_memory_t phys_mem;
    stats_t stats;
    // TLB pointer (new in Milestone 2)
    tlb_t *tlb;
} simulator_t;
```
---
## 4. Interface Contracts
### 4.1 TLB Lifecycle
```c
/*
 * tlb_create
 * 
 * Creates and initializes a new TLB instance.
 * 
 * Parameters:
 *   capacity    - Number of TLB entries. Must be > 0 and <= MAX_TLB_CAPACITY.
 *                 DEFAULT_TLB_CAPACITY (32) is recommended.
 *   page_table  - Pointer to the page table for dirty bit write-back.
 *                 Must not be NULL.
 * 
 * Returns:
 *   Pointer to initialized tlb_t on success.
 *   NULL on failure (invalid parameters or memory allocation error).
 * 
 * Postconditions:
 *   - entries array is allocated with 'capacity' slots
 *   - All entries have valid=false
 *   - current_asid = 0
 *   - clock = TLB_TIMESTAMP_INITIAL
 *   - All stats counters are zero
 *   - page_table reference is stored
 * 
 * Memory ownership:
 *   Caller owns returned pointer and must free with tlb_destroy().
 * 
 * Time complexity: O(capacity) for initialization
 */
tlb_t* tlb_create(uint32_t capacity, pte_t *page_table);
/*
 * tlb_destroy
 * 
 * Releases all resources associated with a TLB.
 * 
 * IMPORTANT: Before destruction, this function writes back all dirty
 * and referenced bits to the page table to avoid losing access metadata.
 * 
 * Parameters:
 *   tlb - Pointer to TLB created by tlb_create().
 *         May be NULL (no-op in that case).
 * 
 * Side effects:
 *   - Writes back dirty/referenced bits for all valid entries
 *   - Frees entries array
 *   - Frees tlb_t struct itself
 * 
 * Postconditions:
 *   - tlb pointer is invalid after this call
 *   - Page table dirty/referenced bits are updated
 * 
 * Time complexity: O(capacity)
 */
void tlb_destroy(tlb_t *tlb);
```
### 4.2 TLB Lookup
```c
/*
 * tlb_lookup
 * 
 * Searches the TLB for a translation matching the given VPN and current ASID.
 * 
 * This is the fast path: on a hit, returns the PFN immediately without
 * consulting the page table.
 * 
 * Parameters:
 *   tlb - TLB state
 *   vpn - Virtual Page Number to search for
 * 
 * Returns:
 *   Pointer to tlb_entry_t if found (HIT)
 *   NULL if not found (MISS)
 * 
 * Side effects on HIT:
 *   - entry->last_used is updated to current clock value
 *   - entry->referenced is set to true
 *   - tlb->stats.hits is incremented
 *   - tlb->clock is incremented
 * 
 * Side effects on MISS:
 *   - tlb->stats.misses is incremented
 *   - No other changes
 * 
 * Note: This function does NOT check permissions. The caller must verify
 *       entry->readable and entry->writable after a successful lookup.
 * 
 * Time complexity: O(capacity) - linear scan
 */
tlb_entry_t* tlb_lookup(tlb_t *tlb, uint32_t vpn);
/*
 * tlb_lookup_with_asid
 * 
 * Variant that allows specifying ASID explicitly (for cross-process lookups).
 * 
 * Parameters:
 *   tlb  - TLB state
 *   vpn  - Virtual Page Number to search for
 *   asid - Address Space ID to match
 * 
 * Returns:
 *   Pointer to tlb_entry_t if found
 *   NULL if not found
 * 
 * Use case: TLB flush by ASID needs to iterate entries matching a specific ASID.
 */
tlb_entry_t* tlb_lookup_with_asid(tlb_t *tlb, uint32_t vpn, uint32_t asid);
```
### 4.3 TLB Insertion and Eviction
```c
/*
 * tlb_insert
 * 
 * Inserts a new translation into the TLB.
 * 
 * If the TLB is full, evicts an entry using LRU policy. Before eviction,
 * propagates dirty/referenced bits back to the page table.
 * 
 * Parameters:
 *   tlb  - TLB state
 *   vpn  - Virtual Page Number
 *   pfn  - Physical Frame Number
 *   pte  - Page Table Entry to copy permissions from
 * 
 * Returns:
 *   true on successful insertion
 *   false on error (should not happen with valid inputs)
 * 
 * Side effects:
 *   - If full, evicts LRU entry (writes back dirty/referenced bits)
 *   - Creates new entry with:
 *     * vpn, pfn, asid set from parameters
 *     * permissions copied from pte
 *     * dirty=false, referenced=false
 *     * last_used set to current clock
 *   - Increments tlb->stats.evictions if eviction occurred
 *   - Increments tlb->stats.dirty_writebacks if evicted entry was dirty
 * 
 * Precondition: pte != NULL and pte->valid == true
 * 
 * Time complexity: O(capacity) for LRU scan
 */
bool tlb_insert(tlb_t *tlb, uint32_t vpn, uint32_t pfn, const pte_t *pte);
/*
 * tlb_find_victim_lru
 * 
 * Internal function: finds the best entry to evict using LRU policy.
 * 
 * Parameters:
 *   tlb - TLB state
 * 
 * Returns:
 *   Index of entry to evict (always valid, even if TLB has empty slots)
 * 
 * Algorithm:
 *   1. First, scan for an invalid entry (valid=false) - free slot
 *   2. If no invalid entries, find entry with smallest last_used timestamp
 *   3. Before returning, write back dirty/referenced bits to page table
 * 
 * Time complexity: O(capacity)
 */
uint32_t tlb_find_victim_lru(tlb_t *tlb);
```
### 4.4 TLB Flush Operations
```c
/*
 * tlb_flush
 * 
 * Invalidates all TLB entries.
 * 
 * Use cases:
 *   - ASID exhaustion (need to recycle ASIDs)
 *   - Major page table changes (e.g., exec() syscall)
 *   - Explicit flush requested by OS
 * 
 * Parameters:
 *   tlb - TLB state
 * 
 * Side effects:
 *   - All entries have valid=false
 *   - Dirty/referenced bits written back to page table for all valid entries
 *   - tlb->stats.flushes incremented
 *   - tlb->stats.dirty_writebacks incremented for each dirty entry
 * 
 * Time complexity: O(capacity)
 */
void tlb_flush(tlb_t *tlb);
/*
 * tlb_flush_asid
 * 
 * Invalidates all entries belonging to a specific ASID.
 * 
 * Use case: Process termination - remove its entries without affecting others.
 * 
 * Parameters:
 *   tlb  - TLB state
 *   asid - Address Space ID to flush
 * 
 * Side effects:
 *   - Entries with matching asid have valid=false
 *   - Dirty/referenced bits written back for flushed entries
 *   - tlb->stats.partial_flushes incremented
 * 
 * Time complexity: O(capacity)
 */
void tlb_flush_asid(tlb_t *tlb, uint32_t asid);
/*
 * tlb_invalidate_entry
 * 
 * Invalidates a single TLB entry by VPN and ASID.
 * 
 * CRITICAL: This must be called whenever the corresponding PTE is modified:
 *   - PTE.valid changed (page evicted or loaded)
 *   - PTE permissions changed (mprotect)
 *   - Page shared/unshared (copy-on-write)
 * 
 * Failure to call this leads to TLB coherency bugs where the TLB returns
 * stale translations.
 * 
 * Parameters:
 *   tlb  - TLB state
 *   vpn  - Virtual Page Number of entry to invalidate
 *   asid - Address Space ID of entry to invalidate
 * 
 * Side effects:
 *   - Matching entry (if found) has valid=false
 *   - Dirty/referenced bits written back before invalidation
 *   - tlb->stats.partial_flushes incremented if entry found
 * 
 * Time complexity: O(capacity)
 */
void tlb_invalidate_entry(tlb_t *tlb, uint32_t vpn, uint32_t asid);
```
### 4.5 Context Switch Support
```c
/*
 * tlb_context_switch
 * 
 * Changes the current ASID for the TLB.
 * 
 * With ASID support, entries from the old process remain in the TLB
 * but won't match lookups (different ASID). This avoids the performance
 * penalty of a full flush on every context switch.
 * 
 * Parameters:
 *   tlb      - TLB state
 *   new_asid - ASID of the incoming process
 * 
 * Side effects:
 *   - tlb->current_asid = new_asid
 *   - No flush performed (entries preserved)
 * 
 * Precondition: new_asid <= MAX_ASID
 * 
 * Time complexity: O(1)
 */
void tlb_context_switch(tlb_t *tlb, uint32_t new_asid);
/*
 * tlb_allocate_asid
 * 
 * Allocates a new ASID for a process.
 * 
 * When ASIDs are exhausted (all 255 in use), performs a full TLB flush
 * and resets the ASID allocation state.
 * 
 * Parameters:
 *   tlb - TLB state
 * 
 * Returns:
 *   Newly allocated ASID (0 to MAX_ASID-1)
 * 
 * Side effects (on ASID exhaustion):
 *   - Full TLB flush
 *   - ASID tracking state reset
 * 
 * Time complexity: O(1) normally, O(capacity) on ASID exhaustion
 */
uint32_t tlb_allocate_asid(tlb_t *tlb);
```
### 4.6 Translation with TLB
```c
/*
 * tlb_translate
 * 
 * Performs virtual-to-physical translation with TLB caching.
 * 
 * This is the main translation function for Milestone 2. It replaces
 * the direct page table lookup from Milestone 1 with a TLB-first approach.
 * 
 * Pipeline:
 *   1. Check TLB for cached translation
 *   2. On HIT: verify permissions, update access bits in TLB, return PFN
 *   3. On MISS: walk page table (from Milestone 1)
 *      - Handle page faults and protection faults as before
 *      - Insert resulting translation into TLB
 *      - Return PFN
 * 
 * Parameters:
 *   sim       - Simulator state (contains page table, TLB, stats)
 *   va        - Virtual address to translate
 *   is_write  - true for write operation, false for read
 * 
 * Returns:
 *   tlb_trans_output_t with:
 *   - result: TLB_TRANS_OK, TLB_TRANS_MISS, TLB_TRANS_PAGE_FAULT, 
 *             or TLB_TRANS_PROTECTION_FAULT
 *   - physical_address: valid when result is OK or MISS
 *   - tlb_hit: true if translation came from TLB, false if from page table
 * 
 * Statistics updates:
 *   - sim->stats.total_accesses incremented
 *   - sim->stats.tlb_hits incremented on TLB hit
 *   - sim->stats.tlb_misses incremented on TLB miss
 *   - sim->stats.page_faults incremented on page fault
 *   - sim->stats.protection_faults incremented on protection fault
 * 
 * TLB entry mutations on HIT:
 *   - entry->referenced = true
 *   - entry->dirty = true (if is_write)
 *   - entry->last_used updated
 * 
 * TLB entry creation on MISS:
 *   - New entry inserted with permissions from PTE
 *   - dirty=false, referenced=false initially
 * 
 * Time complexity: O(1) on TLB hit, O(capacity) on miss (insertion)
 */
tlb_trans_output_t tlb_translate(simulator_t *sim, uint32_t va, bool is_write);
```
### 4.7 Statistics
```c
/*
 * tlb_stats_init
 * 
 * Initializes TLB statistics structure to all zeros.
 */
void tlb_stats_init(tlb_stats_t *stats);
/*
 * tlb_stats_print
 * 
 * Prints formatted TLB statistics report to stdout.
 * 
 * Output format:
 *   ========================================
 *             TLB STATISTICS
 *   ========================================
 *   TLB capacity:           <capacity> entries
 *   Total accesses:         <count>
 *   TLB hits:               <count>
 *   TLB misses:             <count>
 *   TLB hit rate:           <percentage>%
 *   TLB flushes:            <count>
 *   TLB evictions:          <count>
 *   Dirty writebacks:       <count>
 *   ========================================
 */
void tlb_stats_print(const tlb_t *tlb);
/*
 * tlb_get_hit_rate
 * 
 * Calculates the TLB hit rate as a percentage.
 * 
 * Returns:
 *   Hit rate (0.0 to 100.0)
 *   0.0 if no accesses have occurred
 */
double tlb_get_hit_rate(const tlb_t *tlb);
```
---
## 5. Algorithm Specification
### 5.1 TLB Lookup Algorithm
```
ALGORITHM: tlb_lookup
INPUT:  tlb - TLB state
        vpn - Virtual Page Number to find
OUTPUT: Pointer to tlb_entry_t if found, NULL if not found
PROCEDURE:
  1. // Increment clock for LRU tracking
     tlb->clock ← tlb->clock + 1
  2. // Linear scan through all entries
     FOR i ← 0 TO tlb->capacity - 1 DO
       entry ← &tlb->entries[i]
       // Check if entry matches
       IF entry->valid == true AND
          entry->vpn == vpn AND
          entry->asid == tlb->current_asid THEN
         // HIT! Update LRU timestamp
         entry->last_used ← tlb->clock
         // Update statistics
         tlb->stats.hits ← tlb->stats.hits + 1
         RETURN entry  // Pointer to found entry
       END IF
     END FOR
  3. // MISS - entry not found
     tlb->stats.misses ← tlb->stats.misses + 1
     RETURN NULL
INVARIANTS:
  - On HIT: entry->last_used is the most recent timestamp
  - On MISS: TLB state is unchanged (except stats and clock)
  - ASID matching ensures isolation between processes
TIME COMPLEXITY: O(capacity) - must scan all entries
```
### 5.2 LRU Victim Selection Algorithm
```
ALGORITHM: tlb_find_victim_lru
INPUT:  tlb - TLB state
OUTPUT: Index of entry to evict (0 to capacity-1)
PROCEDURE:
  1. // First pass: look for invalid entry (free slot)
     FOR i ← 0 TO tlb->capacity - 1 DO
       IF tlb->entries[i].valid == false THEN
         RETURN i  // Free slot, no eviction needed
       END IF
     END FOR
  2. // No free slots - must evict using LRU
     oldest_timestamp ← UINT64_MAX
     victim_index ← 0
     FOR i ← 0 TO tlb->capacity - 1 DO
       entry ← &tlb->entries[i]
       IF entry->last_used < oldest_timestamp THEN
         oldest_timestamp ← entry->last_used
         victim_index ← i
       END IF
     END FOR
  3. // Write back dirty/referenced bits before eviction
     victim ← &tlb->entries[victim_index]
     page_table_entry ← &tlb->page_table[victim->vpn]
     IF victim->dirty == true THEN
       page_table_entry->dirty ← true
       tlb->stats.dirty_writebacks ← tlb->stats.dirty_writebacks + 1
     END IF
     IF victim->referenced == true THEN
       page_table_entry->referenced ← true
     END IF
  4. // Update eviction counter
     tlb->stats.evictions ← tlb->stats.evictions + 1
  5. RETURN victim_index
INVARIANTS:
  - Returned index is always valid (0 to capacity-1)
  - Victim's dirty/referenced bits are propagated to page table
  - If invalid entry exists, it is preferred over valid entries
EDGE CASES:
  - All entries have same timestamp (unlikely): first entry evicted
  - TLB is empty (all invalid): first entry returned (no eviction cost)
TIME COMPLEXITY: O(capacity) - two passes in worst case
```
### 5.3 TLB Insertion Algorithm
```
ALGORITHM: tlb_insert
INPUT:  tlb - TLB state
        vpn - Virtual Page Number
        pfn - Physical Frame Number
        pte - Page Table Entry (source of permissions)
OUTPUT: Boolean success (always true for valid inputs)
PROCEDURE:
  1. // Find a slot (may trigger eviction)
     slot ← tlb_find_victim_lru(tlb)
  2. // Get pointer to the entry
     entry ← &tlb->entries[slot]
  3. // Initialize entry fields
     entry->vpn ← vpn
     entry->pfn ← pfn
     entry->asid ← tlb->current_asid
     entry->valid ← true
     // Copy permissions from PTE
     entry->readable ← pte->readable
     entry->writable ← pte->writable
     // Access bits start clean (will be set on first access)
     entry->dirty ← false
     entry->referenced ← false
     // Set LRU timestamp
     entry->last_used ← tlb->clock
  4. RETURN true
INVARIANTS:
  - Entry is valid after insertion
  - Entry's ASID matches current process
  - Permissions match source PTE
  - Entry is "clean" (dirty=false, referenced=false)
TIME COMPLEXITY: O(capacity) due to victim selection
```
### 5.4 Translation Pipeline with TLB
```
ALGORITHM: tlb_translate
INPUT:  sim - Simulator state
        va - Virtual address
        is_write - Boolean for operation type
OUTPUT: tlb_trans_output_t with result, physical address, and hit flag
PROCEDURE:
  1. // Count access
     sim->stats.total_accesses ← sim->stats.total_accesses + 1
  2. // Extract VPN and offset
     vpn ← va >> PAGE_SHIFT
     offset ← va & PAGE_MASK
  3. // === FAST PATH: TLB Lookup ===
     tlb_entry ← tlb_lookup(sim->tlb, vpn)
     IF tlb_entry != NULL THEN
       // TLB HIT
       output.tlb_hit ← true
       sim->stats.tlb_hits ← sim->stats.tlb_hits + 1
       // Check permissions using cached bits
       IF is_write == true AND tlb_entry->writable == false THEN
         sim->stats.protection_faults ← sim->stats.protection_faults + 1
         output.result ← TLB_TRANS_PROTECTION_FAULT
         RETURN output
       END IF
       IF is_write == false AND tlb_entry->readable == false THEN
         sim->stats.protection_faults ← sim->stats.protection_faults + 1
         output.result ← TLB_TRANS_PROTECTION_FAULT
         RETURN output
       END IF
       // Update access bits in TLB entry
       tlb_entry->referenced ← true
       IF is_write == true THEN
         tlb_entry->dirty ← true
       END IF
       // Compose physical address
       output.physical_address ← (tlb_entry->pfn << PAGE_SHIFT) | offset
       output.result ← TLB_TRANS_OK
       RETURN output
     END IF
  4. // === SLOW PATH: Page Table Walk ===
     output.tlb_hit ← false
     sim->stats.tlb_misses ← sim->stats.tlb_misses + 1
     // Get PTE from page table
     pte ← &sim->page_table[vpn]
     // Check valid bit (page fault handling from Milestone 1)
     IF pte->valid == false THEN
       sim->stats.page_faults ← sim->stats.page_faults + 1
       // Demand paging would go here
       output.result ← TLB_TRANS_PAGE_FAULT
       RETURN output
     END IF
     // Check permissions in PTE
     IF is_write == true AND pte->writable == false THEN
       sim->stats.protection_faults ← sim->stats.protection_faults + 1
       output.result ← TLB_TRANS_PROTECTION_FAULT
       RETURN output
     END IF
     // Update PTE access bits
     pte->referenced ← true
     IF is_write == true THEN
       pte->dirty ← true
     END IF
     // Insert translation into TLB for future accesses
     tlb_insert(sim->tlb, vpn, pte->pfn, pte)
     // Compose physical address
     output.physical_address ← (pte->pfn << PAGE_SHIFT) | offset
     output.result ← TLB_TRANS_MISS  // Miss but successfully resolved
     RETURN output
INVARIANTS:
  - On TLB_TRANS_OK: tlb_hit=true, physical_address valid
  - On TLB_TRANS_MISS: tlb_hit=false, physical_address valid, new TLB entry created
  - On TLB_TRANS_PAGE_FAULT: page not in memory, no TLB entry created
  - On TLB_TRANS_PROTECTION_FAULT: permission violation, no state change
TIME COMPLEXITY:
  - TLB hit: O(capacity) for lookup
  - TLB miss: O(capacity) for lookup + O(capacity) for insertion
```
### 5.5 TLB Flush Algorithm
```
ALGORITHM: tlb_flush
INPUT:  tlb - TLB state
OUTPUT: None (side effects only)
PROCEDURE:
  1. FOR i ← 0 TO tlb->capacity - 1 DO
       entry ← &tlb->entries[i]
       IF entry->valid == true THEN
         // Write back dirty/referenced bits
         pte ← &tlb->page_table[entry->vpn]
         IF entry->dirty == true THEN
           pte->dirty ← true
           tlb->stats.dirty_writebacks ← tlb->stats.dirty_writebacks + 1
         END IF
         IF entry->referenced == true THEN
           pte->referenced ← true
         END IF
         // Invalidate entry
         entry->valid ← false
       END IF
     END FOR
  2. // Update statistics
     tlb->stats.flushes ← tlb->stats.flushes + 1
INVARIANTS:
  - All entries have valid=false after flush
  - Dirty/referenced bits are not lost (written to page table)
  - Page table state is updated before TLB entries invalidated
TIME COMPLEXITY: O(capacity)
```
### 5.6 Single Entry Invalidation Algorithm
```
ALGORITHM: tlb_invalidate_entry
INPUT:  tlb - TLB state
        vpn - Virtual Page Number
        asid - Address Space ID
OUTPUT: None (side effects only)
PROCEDURE:
  1. // Search for matching entry
     FOR i ← 0 TO tlb->capacity - 1 DO
       entry ← &tlb->entries[i]
       IF entry->valid == true AND
          entry->vpn == vpn AND
          entry->asid == asid THEN
         // Found! Write back dirty/referenced bits
         pte ← &tlb->page_table[entry->vpn]
         IF entry->dirty == true THEN
           pte->dirty ← true
           tlb->stats.dirty_writebacks ← tlb->stats.dirty_writebacks + 1
         END IF
         IF entry->referenced == true THEN
           pte->referenced ← true
         END IF
         // Invalidate entry
         entry->valid ← false
         // Update statistics
         tlb->stats.partial_flushes ← tlb->stats.partial_flushes + 1
         RETURN  // Entry found and invalidated
       END IF
     END FOR
  2. // Entry not found - no action needed
     RETURN
INVARIANTS:
  - If entry exists, it is invalidated
  - Dirty/referenced bits are preserved in page table
  - If entry doesn't exist, no state changes
TIME COMPLEXITY: O(capacity)
```
---
## 6. Error Handling Matrix
| Error | Detected By | Recovery | User-Visible? | System State |
|-------|-------------|----------|---------------|--------------|
| NULL TLB pointer | Any TLB function at entry | Return error code or assert | Yes, assertion failure or error return | Unchanged |
| Invalid capacity (0 or > MAX) | `tlb_create` | Return NULL | Yes, creation fails | No TLB created |
| NULL page table reference | `tlb_create` | Return NULL | Yes, creation fails | No TLB created |
| Memory allocation failure | `tlb_create` | Return NULL | Yes, creation fails | No TLB created |
| ASID exceeds MAX_ASID | `tlb_context_switch`, `tlb_allocate_asid` | Assert or return error | Yes, invalid ASID | Unchanged |
| PTE modification without TLB invalidation | (Design-time error) | Stale TLB entries served | Maybe (wrong translations) | TLB inconsistent with page table |
| LRU timestamp overflow | `tlb_lookup` (clock increment) | Wrap around, relative order preserved | No | All entries appear "old" relative to new ones |
| Insert with invalid PTE | `tlb_insert` | Undefined behavior (precondition violation) | Maybe | TLB entry with invalid mapping |
| Flush during lookup | (Concurrency issue) | N/A (single-threaded) | N/A | N/A |
---
## 7. Implementation Sequence with Checkpoints
### Phase 1: Define TLB Entry and TLB Structures (1-2 hours)
**Files:** `include/tlb_config.h`, `include/tlb_types.h`
**Tasks:**
1. Create `tlb_config.h` with capacity limits, ASID configuration
2. Create `tlb_types.h` with `tlb_entry_t`, `tlb_stats_t`, `tlb_t`
3. Add static assertion for `tlb_entry_t` size (32 bytes)
4. Define result enums and output structs
**Checkpoint:**
```bash
# Compile header files only
gcc -c -fsyntax-only include/tlb_config.h include/tlb_types.h -I include
# Expected: No errors or warnings
# Verify struct sizes
cat > test_sizes.c << 'EOF'
#include <stdio.h>
#include "tlb_types.h"
int main() {
    printf("tlb_entry_t: %zu bytes\n", sizeof(tlb_entry_t));
    printf("tlb_t: %zu bytes\n", sizeof(tlb_t));
    printf("tlb_stats_t: %zu bytes\n", sizeof(tlb_stats_t));
    return 0;
}
EOF
gcc -o test_sizes test_sizes.c -I include && ./test_sizes
# Expected: tlb_entry_t: 32 bytes
```
---
### Phase 2: Implement TLB Lookup with ASID Matching (1-2 hours)
**Files:** `src/tlb.c`, `include/tlb.h`
**Tasks:**
1. Implement `tlb_create` with entry array allocation
2. Implement `tlb_destroy` with dirty bit write-back
3. Implement `tlb_lookup` with linear scan and ASID matching
4. Implement `tlb_lookup_with_asid` variant
**Checkpoint:**
```c
// Add to tests/test_tlb_basic.c
void test_tlb_lookup_basic(void) {
    pte_t page_table[1024] = {0};
    tlb_t *tlb = tlb_create(16, page_table);
    assert(tlb != NULL);
    // Manually insert an entry for testing
    tlb->entries[0].vpn = 0x10;
    tlb->entries[0].pfn = 0x42;
    tlb->entries[0].asid = 0;
    tlb->entries[0].valid = true;
    tlb->entries[0].readable = true;
    tlb->entries[0].writable = true;
    // Should find it
    tlb_entry_t *entry = tlb_lookup(tlb, 0x10);
    assert(entry != NULL);
    assert(entry->pfn == 0x42);
    // Should not find different VPN
    entry = tlb_lookup(tlb, 0x20);
    assert(entry == NULL);
    // Should not find with different ASID
    tlb->current_asid = 1;
    entry = tlb_lookup(tlb, 0x10);
    assert(entry == NULL);
    tlb_destroy(tlb);
    printf("TLB lookup basic tests PASSED\n");
}
```
```bash
gcc -o test_tlb_basic tests/test_tlb_basic.c src/tlb.c -I include
./test_tlb_basic
# Expected: "TLB lookup basic tests PASSED"
```
---
### Phase 3: Implement LRU Victim Selection (1-2 hours)
**Files:** `src/tlb.c`
**Tasks:**
1. Implement `tlb_find_victim_lru` with timestamp comparison
2. Handle empty slot preference over valid entries
3. Implement dirty/referenced bit write-back in victim selection
**Checkpoint:**
```c
void test_tlb_lru(void) {
    pte_t page_table[1024] = {0};
    tlb_t *tlb = tlb_create(4, page_table);
    // Fill all entries
    for (int i = 0; i < 4; i++) {
        tlb->entries[i].vpn = i;
        tlb->entries[i].valid = true;
        tlb->entries[i].last_used = i * 100;  // Entry 0 is oldest
    }
    // Find victim - should be entry 0 (oldest)
    uint32_t victim = tlb_find_victim_lru(tlb);
    assert(victim == 0);
    // Update entry 0's timestamp
    tlb->entries[0].last_used = 1000;
    // Now entry 1 should be victim
    victim = tlb_find_victim_lru(tlb);
    assert(victim == 1);
    tlb_destroy(tlb);
    printf("TLB LRU tests PASSED\n");
}
```
---
### Phase 4: Implement TLB Insertion with Eviction (1-2 hours)
**Files:** `src/tlb.c`
**Tasks:**
1. Implement `tlb_insert` using victim selection
2. Copy permissions from PTE to TLB entry
3. Initialize dirty/referenced to false
**Checkpoint:**
```c
void test_tlb_insert(void) {
    pte_t page_table[1024] = {0};
    tlb_t *tlb = tlb_create(4, page_table);
    // Create a PTE to insert
    pte_t pte = {.pfn = 0x99, .valid = true, .readable = true, .writable = true};
    // Insert should succeed
    bool result = tlb_insert(tlb, 0x10, 0x99, &pte);
    assert(result == true);
    // Should find it now
    tlb_entry_t *entry = tlb_lookup(tlb, 0x10);
    assert(entry != NULL);
    assert(entry->pfn == 0x99);
    assert(entry->readable == true);
    // Fill remaining slots
    for (int i = 1; i < 4; i++) {
        pte.pfn = i;
        tlb_insert(tlb, i, i, &pte);
    }
    // One more insert should trigger eviction
    pte.pfn = 0xAA;
    result = tlb_insert(tlb, 0xFF, 0xAA, &pte);
    assert(result == true);
    assert(tlb->stats.evictions == 1);
    tlb_destroy(tlb);
    printf("TLB insert tests PASSED\n");
}
```
---
### Phase 5: Implement Dirty/Referenced Bit Write-Back (1-2 hours)
**Files:** `src/tlb.c`
**Tasks:**
1. Ensure `tlb_find_victim_lru` writes back dirty/referenced bits
2. Ensure `tlb_flush` writes back all dirty/referenced bits
3. Ensure `tlb_invalidate_entry` writes back before invalidating
**Checkpoint:**
```c
void test_tlb_writeback(void) {
    pte_t page_table[1024] = {0};
    tlb_t *tlb = tlb_create(4, page_table);
    // Insert an entry
    pte_t pte = {.pfn = 0x42, .valid = true, .readable = true, .writable = true};
    tlb_insert(tlb, 0x10, 0x42, &pte);
    // Simulate write access (set dirty in TLB)
    tlb_entry_t *entry = tlb_lookup(tlb, 0x10);
    entry->dirty = true;
    entry->referenced = true;
    // PTE should not have dirty bit yet
    assert(page_table[0x10].dirty == false);
    // Evict by inserting 4 more entries
    for (int i = 0; i < 4; i++) {
        pte.pfn = i + 1;
        tlb_insert(tlb, 0x20 + i, i + 1, &pte);
    }
    // PTE should now have dirty bit
    assert(page_table[0x10].dirty == true);
    assert(page_table[0x10].referenced == true);
    assert(tlb->stats.dirty_writebacks >= 1);
    tlb_destroy(tlb);
    printf("TLB writeback tests PASSED\n");
}
```
---
### Phase 6: Implement TLB Flush Operations (1-2 hours)
**Files:** `src/tlb.c`
**Tasks:**
1. Implement `tlb_flush` (full flush)
2. Implement `tlb_flush_asid` (ASID-specific)
3. Implement `tlb_invalidate_entry` (single entry)
**Checkpoint:**
```c
void test_tlb_flush(void) {
    pte_t page_table[1024] = {0};
    tlb_t *tlb = tlb_create(16, page_table);
    // Insert entries for different ASIDs
    pte_t pte = {.valid = true, .readable = true, .writable = true};
    tlb->current_asid = 0;
    tlb_insert(tlb, 0x10, 0x42, &pte);
    tlb->entries[0].dirty = true;
    tlb->current_asid = 1;
    tlb_insert(tlb, 0x20, 0x43, &pte);
    tlb->entries[1].dirty = true;
    // Flush ASID 0
    tlb_flush_asid(tlb, 0);
    assert(tlb->entries[0].valid == false);
    assert(tlb->entries[1].valid == true);
    assert(page_table[0x10].dirty == true);
    // Full flush
    tlb_flush(tlb);
    assert(tlb->entries[1].valid == false);
    assert(page_table[0x20].dirty == true);
    tlb_destroy(tlb);
    printf("TLB flush tests PASSED\n");
}
```
---
### Phase 7: Integrate TLB into Translation Pipeline (1-2 hours)
**Files:** `src/tlb_translate.c`, updated `src/translate.c`
**Tasks:**
1. Implement `tlb_translate` function
2. Update `simulator_t` to include TLB pointer
3. Update `simulator_create` to create TLB
4. Update `simulator_destroy` to destroy TLB
**Checkpoint:**
```c
void test_tlb_translate(void) {
    simulator_t *sim = simulator_create(64);
    sim->tlb = tlb_create(32, sim->page_table);
    // First access - should be TLB miss, page fault, then insert
    tlb_trans_output_t result = tlb_translate(sim, 0x1000, false);
    assert(result.result == TLB_TRANS_OK || result.result == TLB_TRANS_MISS);
    assert(result.tlb_hit == false);  // First access
    assert(sim->stats.tlb_misses == 1);
    // Second access to same page - should be TLB hit
    result = tlb_translate(sim, 0x1004, false);
    assert(result.tlb_hit == true);
    assert(sim->stats.tlb_hits == 1);
    // Verify physical addresses have same PFN
    uint32_t pfn1 = result.physical_address >> PAGE_SHIFT;
    result = tlb_translate(sim, 0x1000, true);
    uint32_t pfn2 = result.physical_address >> PAGE_SHIFT;
    assert(pfn1 == pfn2);
    tlb_destroy(sim->tlb);
    simulator_destroy(sim);
    printf("TLB translate tests PASSED\n");
}
```
---
### Phase 8: Write Tests for Hit/Miss, Coherency, Context Switch (2-3 hours)
**Files:** All test files
**Tasks:**
1. Complete `test_tlb_basic.c` with lookup tests
2. Complete `test_tlb_lru.c` with eviction order tests
3. Complete `test_tlb_coherency.c` with stale entry tests
4. Complete `test_tlb_context.c` with ASID switching tests
5. Create test traces for integration testing
**Checkpoint:**
```bash
make test
# Expected: All tests pass
./vm_sim traces/tlb_locality.trace 64
# Expected: TLB hit rate > 95%
./vm_sim traces/tlb_random.trace 64
# Expected: TLB hit rate < 50%
```
---
## 8. Test Specification
### 8.1 Unit Tests for TLB Lookup
| Test Case | Initial State | Input | Expected Result | Notes |
|-----------|---------------|-------|-----------------|-------|
| Basic hit | Entry with VPN=0x10, ASID=0 exists | VPN=0x10, ASID=0 | Pointer to entry | |
| Basic miss | No entry for VPN | VPN=0x20 | NULL | |
| ASID mismatch | Entry with VPN=0x10, ASID=0 | VPN=0x10, ASID=1 | NULL | Isolation |
| Invalid entry skipped | Entry valid=false | VPN matching invalid entry | NULL | |
| LRU update on hit | Entry exists | Lookup that hits | last_used updated | |
| Stats increment | Any lookup | Any | hits or misses incremented | |
### 8.2 Unit Tests for LRU Victim Selection
| Test Case | Initial State | Expected Victim | Notes |
|-----------|---------------|-----------------|-------|
| Empty TLB | All entries valid=false | Index 0 | Free slot preferred |
| One entry, oldest | Entry 0 has smallest timestamp | Index 0 | |
| Middle entry oldest | Entry 1 has smallest timestamp | Index 1 | |
| All same timestamp | All entries same last_used | Index 0 | Tie-break |
| Dirty entry eviction | Victim has dirty=true | Page table updated | Write-back |
### 8.3 Unit Tests for TLB Insertion
| Test Case | Initial State | Input | Expected Result | Notes |
|-----------|---------------|-------|-----------------|-------|
| Insert into empty | TLB empty | VPN=0x10 | Entry created | |
| Insert triggers eviction | TLB full | New VPN | Eviction occurs | |
| Permissions copied | PTE with readable=0 | Insert | Entry.readable=0 | |
| Access bits reset | Any insert | Entry.dirty=false | Clean start | |
### 8.4 Unit Tests for Coherency
| Test Case | Initial State | Action | Expected Result |
|-----------|---------------|--------|-----------------|
| PTE invalidation | TLB has entry | Set PTE.valid=false, call invalidate | Entry.valid=false |
| Permission change | TLB has entry | Change PTE.writable, call invalidate | Entry removed |
| No invalidation (bug) | TLB has entry | Change PTE.valid | Stale entry served (detectable) |
| Dirty write-back | TLB entry dirty | Evict or invalidate | PTE.dirty=true |
### 8.5 Unit Tests for Context Switch
| Test Case | Initial State | Action | Expected Result |
|-----------|---------------|--------|-----------------|
| Basic switch | ASID=0 | Switch to ASID=1 | current_asid=1 |
| Lookup after switch | Entry ASID=0, current=1 | Lookup VPN | NULL (miss) |
| Switch back | Entry ASID=0 preserved | Switch to ASID=0 | Hit on old entry |
| ASID exhaustion | 255 ASIDs used | Allocate 256th | Full flush, reset |
### 8.6 Integration Tests
**Test Trace 1: High Locality (`traces/tlb_locality.trace`)**
```
# Access same 4 pages repeatedly (4 entries in 32-entry TLB)
R 0x00001000
R 0x00002000
R 0x00003000
R 0x00004000
R 0x00001000
R 0x00002000
... (repeat 100 times)
```
**Expected:** TLB hit rate > 95%
**Test Trace 2: Random Access (`traces/tlb_random.trace`)**
```
# Access many different pages (exceeds TLB capacity)
R 0x00001000
R 0x00002000
R 0x00003000
... (64+ unique pages)
```
**Expected:** TLB hit rate < 50% with 32-entry TLB
**Test Trace 3: Context Switch (`traces/tlb_context.trace`)**
```
# Process A (ASID 0)
R 0x00001000
R 0x00001000  # Hit
# [Context switch to Process B]
R 0x00001000  # Miss (different ASID)
# [Context switch back to Process A]
R 0x00001000  # Hit (entry preserved)
```
---
## 9. Performance Targets
| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| TLB lookup (hit) | < 5 simulated cycles | Linear scan through 32 entries |
| TLB lookup (miss) | < 5 simulated cycles | Same scan, no entry found |
| TLB insert | < 50 simulated cycles | Includes O(32) victim scan |
| Full flush | < 100 simulated cycles | O(32) scan with write-back |
| LRU victim selection | < 50 simulated cycles | O(32) timestamp comparison |
| TLB hit rate (locality trace) | > 95% | Statistics comparison |
| TLB hit rate (random trace) | < 50% | Statistics comparison |
| Memory overhead | 1 KB (32 entries × 32 bytes) | Struct size calculation |
**Cache Behavior Analysis:**
- TLB entry array: 32 × 32 = 1024 bytes, fits in single 64-byte cache line × 16
- Sequential scan: Prefetch-friendly, predictable access pattern
- LRU scan: Sequential read of timestamps, prefetch-friendly
---
## 10. State Machine (TLB Entry Lifecycle)
```
                    ┌─────────────────────────────────────┐
                    │         TLB ENTRY STATES            │
                    └─────────────────────────────────────┘
     ┌──────────────────┐
     │    INVALID       │  valid=false, vpn/pfn undefined
     │   (empty slot)   │
     └────────┬─────────┘
              │
              │ tlb_insert() on TLB miss
              │ (copy from PTE)
              ▼
     ┌──────────────────┐
     │ VALID-CLEAN-UNREF│  valid=true, dirty=false, ref=false
     │  (just inserted) │
     └────────┬─────────┘
              │
              │ First access (lookup hit)
              │ sets referenced=true
              ▼
     ┌──────────────────┐
     │ VALID-CLEAN-REF  │  valid=true, dirty=false, ref=true
     │   (read-only)    │
     └────────┬─────────┘
              │
              │ Write access (lookup hit)
              │ sets dirty=true
              ▼
     ┌──────────────────┐
     │ VALID-DIRTY-REF  │  valid=true, dirty=true, ref=true
     │   (modified)     │
     └────────┬─────────┘
              │
              │ Eviction or invalidation
              │ (write back dirty/ref to PTE)
              ▼
     ┌──────────────────┐
     │    INVALID       │  valid=false
     │   (evicted)      │
     └──────────────────┘
TRANSITIONS:
  INVALID → VALID-CLEAN-UNREF:  TLB miss, insert from page table
  VALID-CLEAN-UNREF → VALID-CLEAN-REF:  First access (read or write)
  VALID-CLEAN-* → VALID-DIRTY-*:  Write access
  ANY-VALID → INVALID:  Eviction, flush, or invalidation
CRITICAL INVARIANTS:
  - Eviction ALWAYS writes dirty bit to page table first
  - Invalidation ALWAYS writes dirty bit to page table first
  - PTE modification MUST trigger TLB invalidation (caller responsibility)
COHERENCY BUG PATTERN:
  If PTE is modified (e.g., page evicted) without calling
  tlb_invalidate_entry(), the TLB will continue serving the
  old translation → STALE ENTRY BUG
```
---
## 11. Hardware Soul: Cache and Branch Analysis
### 11.1 Cache Lines Touched Per Operation
| Operation | Cache Lines | Hot/Cold | Notes |
|-----------|-------------|----------|-------|
| TLB lookup (scan) | 16 lines (1 KB / 64) | Hot after warmup | Sequential scan prefetches well |
| TLB entry update | 1 line | Hot | Same line touched repeatedly |
| Page table access (miss) | 1+ lines | Depends on access pattern | May cascade to multiple levels |
| Stats update | 1 line | Hot | Same cache line for all stats |
### 11.2 Branch Prediction
| Branch | Predictability | Misprediction Cost | Frequency |
|--------|----------------|-------------------|-----------|
| `if (entry->valid && entry->vpn == vpn && ...)` | Moderate | 15 cycles | Once per entry scanned |
| `if (entry->valid)` in flush | Highly predictable | 15 cycles | Most entries valid after warmup |
| `if (tlb_entry != NULL)` in translate | Depends on locality | 15 cycles | Predictable for locality-friendly traces |
### 11.3 Memory Access Pattern
- **TLB lookup**: Sequential scan through entry array. Prefetcher can recognize pattern after first few accesses.
- **Page table access**: Random access indexed by VPN. No prefetch benefit.
- **LRU scan**: Sequential read of timestamps. Prefetch-friendly.
### 11.4 Comparison to Real Hardware
| Aspect | Our Simulation | Real Hardware |
|--------|----------------|---------------|
| Lookup | O(n) linear scan | O(1) parallel CAM |
| Associativity | Fully associative | Set-associative (4-way, 8-way) |
| Replacement | LRU with timestamps | Pseudo-LRU tree |
| ASID width | 8 bits (256 processes) | 12 bits (4096 processes) on x86-64 |
| Entries | 32-64 | 64-1536 (L1 + L2) |
---
## 12. Sample Implementation: tlb_translate
```c
#include "types.h"
#include "tlb_types.h"
#include "tlb.h"
#include "config.h"
tlb_trans_output_t tlb_translate(simulator_t *sim, uint32_t va, bool is_write) {
    tlb_trans_output_t output = {0};
    // Step 1: Count access
    sim->stats.total_accesses++;
    // Step 2: Extract VPN and offset
    uint32_t vpn = va >> PAGE_SHIFT;
    uint32_t offset = va & PAGE_MASK;
    // Step 3: Check TLB first (fast path)
    tlb_entry_t *entry = tlb_lookup(sim->tlb, vpn);
    if (entry != NULL) {
        // TLB HIT
        output.tlb_hit = true;
        sim->stats.tlb_hits++;
        // Check permissions using cached bits
        if (is_write && !entry->writable) {
            sim->stats.protection_faults++;
            output.result = TLB_TRANS_PROTECTION_FAULT;
            return output;
        }
        if (!is_write && !entry->readable) {
            sim->stats.protection_faults++;
            output.result = TLB_TRANS_PROTECTION_FAULT;
            return output;
        }
        // Update access bits in TLB entry
        entry->referenced = true;
        if (is_write) {
            entry->dirty = true;
        }
        // Compose physical address
        output.physical_address = (entry->pfn << PAGE_SHIFT) | offset;
        output.result = TLB_TRANS_OK;
        return output;
    }
    // Step 4: TLB MISS - walk page table (slow path)
    output.tlb_hit = false;
    sim->stats.tlb_misses++;
    // Bounds check
    if (vpn >= MAX_VPN) {
        sim->stats.protection_faults++;
        output.result = TLB_TRANS_PROTECTION_FAULT;
        return output;
    }
    pte_t *pte = &sim->page_table[vpn];
    // Check valid bit
    if (!pte->valid) {
        sim->stats.page_faults++;
        output.result = TLB_TRANS_PAGE_FAULT;
        return output;
    }
    // Check permissions in PTE
    if (is_write && !pte->writable) {
        sim->stats.protection_faults++;
        output.result = TLB_TRANS_PROTECTION_FAULT;
        return output;
    }
    if (!is_write && !pte->readable) {
        sim->stats.protection_faults++;
        output.result = TLB_TRANS_PROTECTION_FAULT;
        return output;
    }
    // Update PTE access bits
    pte->referenced = true;
    if (is_write) {
        pte->dirty = true;
    }
    // Insert into TLB for future accesses
    tlb_insert(sim->tlb, vpn, pte->pfn, pte);
    // Compose physical address
    output.physical_address = (pte->pfn << PAGE_SHIFT) | offset;
    output.result = TLB_TRANS_MISS;
    return output;
}
```
---
## 13. Updated Makefile
```makefile
# Add to existing Makefile from Milestone 1
CC = gcc
CFLAGS = -Wall -Wextra -Werror -std=c11 -g -O0
INCLUDES = -I include
# Milestone 1 sources
SRCS_M1 = src/translate.c src/parser.c src/stats.c
# Milestone 2 sources
SRCS_M2 = src/tlb.c src/tlb_translate.c
# All sources
SRCS = $(SRCS_M1) $(SRCS_M2) src/main.c
OBJS = $(SRCS:.c=.o)
TARGET = vm_sim
# Test targets
TEST_M2_BASIC = test_tlb_basic
TEST_M2_LRU = test_tlb_lru
TEST_M2_COHERENCY = test_tlb_coherency
TEST_M2_CONTEXT = test_tlb_context
TEST_M2_ALL = $(TEST_M2_BASIC) $(TEST_M2_LRU) $(TEST_M2_COHERENCY) $(TEST_M2_CONTEXT)
.PHONY: all clean test test_m1 test_m2
all: $(TARGET)
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<
# Milestone 2 tests
$(TEST_M2_BASIC): tests/test_tlb_basic.c src/tlb.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^
$(TEST_M2_LRU): tests/test_tlb_lru.c src/tlb.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^
$(TEST_M2_COHERENCY): tests/test_tlb_coherency.c src/tlb.c src/translate.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^
$(TEST_M2_CONTEXT): tests/test_tlb_context.c src/tlb.c src/tlb_translate.c src/translate.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^
test_m2: $(TEST_M2_ALL)
	@echo "=== Running TLB Tests ==="
	./$(TEST_M2_BASIC)
	./$(TEST_M2_LRU)
	./$(TEST_M2_COHERENCY)
	./$(TEST_M2_CONTEXT)
	@echo "=== TLB Tests Complete ==="
test: test_m1 test_m2
	./$(TARGET) traces/tlb_locality.trace 64
	./$(TARGET) traces/tlb_random.trace 64
clean:
	rm -f $(OBJS) $(TARGET) $(TEST_M2_ALL)
```
---
[[CRITERIA_JSON: {"module_id": "virtual-memory-sim-m2", "criteria": ["TLB entry structure (tlb_entry_t) contains vpn (uint32_t), pfn (uint32_t), asid (uint32_t), last_used (uint64_t), valid (bool), readable (bool), writable (bool), dirty (bool), referenced (bool)", "TLB structure (tlb_t) contains entries array, capacity (uint32_t), current_asid (uint32_t), clock (uint64_t), stats (tlb_stats_t), and page_table back-reference", "TLB lookup function (tlb_lookup) checks entries in O(capacity) time; returns pointer on hit (vpn AND asid match with valid=true), NULL on miss", "TLB hit increments stats.hits, updates entry->last_used to current clock, sets entry->referenced=true, and sets entry->dirty=true on write", "TLB miss increments stats.misses and returns NULL without modifying TLB state", "LRU victim selection (tlb_find_victim_lru) scans for minimum last_used timestamp; prefers invalid entries over valid ones", "TLB insertion (tlb_insert) finds victim slot, writes back dirty/referenced bits to page table, creates new entry with permissions from PTE", "Dirty bit write-back: on eviction, if entry->dirty then page_table[vpn].dirty = true; same for referenced bit", "TLB flush (tlb_flush) invalidates all entries after writing back dirty/referenced bits; increments stats.flushes", "TLB flush by ASID (tlb_flush_asid) invalidates only entries matching specified asid; increments stats.partial_flushes", "Single entry invalidation (tlb_invalidate_entry) removes entry by vpn+asid after write-back; increments stats.partial_flushes if found", "Context switch (tlb_context_switch) changes current_asid field; preserves all entries (no flush)", "Translation with TLB (tlb_translate) checks TLB first; on hit returns immediately, on miss walks page table and inserts result", "TLB statistics (tlb_stats_t) includes hits, misses, flushes, evictions, partial_flushes, dirty_writebacks as uint64_t fields", "TLB coherency: tlb_invalidate_entry must be called when corresponding PTE is modified (valid bit change, permission change)"]}]
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: virtual-memory-sim-m3 -->
# Technical Design Document: Multi-Level Page Tables
**Module ID:** `virtual-memory-sim-m3`  
**Language:** C (BINDING)  
**Difficulty:** Advanced
---
## 1. Module Charter
This module implements two-level hierarchical page tables with CR3 register simulation, replacing the flat page table from Milestone 1 with a memory-efficient sparse structure. The page directory (first level) contains entries pointing to second-level page tables, which are allocated on-demand only when their corresponding address region is accessed. This dramatically reduces memory overhead for realistic workloads where processes use scattered, small regions of their 4 GB virtual address space.
**What it does:**
- Implements page directory entries (PDEs) that point to second-level page tables
- Decomposes 32-bit virtual addresses into directory index (10 bits), table index (10 bits), and offset (12 bits)
- Performs two-level page table walks: CR3 → PDE → PTE → physical address
- Allocates second-level page tables on-demand when first access to a directory's region occurs
- Simulates CR3 register for per-process page directory roots
- Provides context switch that changes CR3 and flushes/switches TLB
- Measures and reports memory overhead comparison (flat vs multi-level)
**What it does NOT do:**
- Page replacement when physical memory exhausts (Milestone 4)
- Three-level page tables (stretch goal only)
- Inverted or hashed page table schemes
**Upstream dependencies:** Milestone 1 (PTE structure, physical memory, statistics), Milestone 2 (TLB caching, ASID support)
**Downstream dependencies:** Milestone 4 (page replacement algorithms traverse page tables to find eviction candidates)
**Invariants that must always hold:**
1. PDE with `present=false` indicates the entire 4 MB region is unmapped; no second-level table exists
2. PDE with `present=true` must have a valid `page_table_pfn` pointing to an allocated page table
3. CR3 always points to a valid page directory for the current process
4. Each process has exactly one page directory; directories are not shared
5. Free frame count + allocated frame count + page table frames = total frames (conservation)
6. On-demand allocation never fails silently—returns error code if out of memory
---
## 2. File Structure
Create files in this exact order:
```
01. include/mlpt_config.h        — Multi-level page table constants and bit masks
02. include/mlpt_types.h         — PDE structure, process structure, overhead metrics
03. include/mlpt.h               — Multi-level walk and context switch declarations
04. src/mlpt_walk.c              — Two-level page table walk implementation
05. src/mlpt_alloc.c             — On-demand page table allocation
06. src/context_switch.c         — CR3 management and process switching
07. src/memory_overhead.c        — Memory overhead measurement and comparison
08. tests/test_bit_extraction.c  — Unit tests for directory/table index extraction
09. tests/test_mlpt_walk.c       — Unit tests for two-level walks
10. tests/test_on_demand.c       — Unit tests for on-demand allocation
11. tests/test_context_switch.c  — Unit tests for CR3 and process isolation
12. tests/test_overhead.c        — Unit tests for memory overhead comparison
13. traces/sparse.trace          — Test trace with sparse address access
14. traces/dense.trace           — Test trace with dense address access
15. Makefile                     — Updated build system with MLPT targets
```
---
## 3. Complete Data Model
### 3.1 Multi-Level Page Table Configuration (`mlpt_config.h`)
```c
#ifndef MLPT_CONFIG_H
#define MLPT_CONFIG_H
#include <stdint.h>
/*
 * Two-Level Page Table Bit Layout (32-bit address, 4 KB pages)
 *
 * Virtual Address Decomposition:
 * ┌─────────────────┬─────────────────┬────────────────────┐
 * │  Directory Index│   Table Index   │      Offset        │
 * │    (10 bits)    │    (10 bits)    │     (12 bits)      │
 * └─────────────────┴─────────────────┴────────────────────┘
 *        31        22 21            12 11                 0
 *
 * - Directory Index: bits 31-22 (10 bits = 1024 entries)
 * - Table Index:     bits 21-12 (10 bits = 1024 entries per table)
 * - Offset:          bits 11-0  (12 bits = 4096 bytes per page)
 *
 * Address Space Coverage:
 * - Each directory entry covers: 1024 pages × 4 KB = 4 MB
 * - Total address space: 1024 × 4 MB = 4 GB
 */
/* Page configuration (inherited from Milestone 1) */
#define PAGE_SHIFT              12
#define PAGE_SIZE               (1u << PAGE_SHIFT)    // 4096 bytes
#define PAGE_MASK               (PAGE_SIZE - 1)        // 0x00000FFF
/* Directory configuration */
#define DIR_SHIFT               22                     // Bits 31-22
#define DIR_ENTRIES             1024                   // 2^10
#define DIR_MASK                0x3FF                  // 10 bits
/* Second-level table configuration */
#define TABLE_SHIFT             12                     // Bits 21-12
#define TABLE_ENTRIES           1024                   // 2^10
#define TABLE_MASK              0x3FF                  // 10 bits
/* Memory overhead constants */
#define BYTES_PER_PTE           12                     // sizeof(pte_t) with padding
#define BYTES_PER_PDE           8                      // sizeof(pde_t) with padding
/* Coverage calculations (for documentation) */
#define BYTES_PER_DIR_ENTRY     (TABLE_ENTRIES * PAGE_SIZE)  // 4 MB
#define TOTAL_ADDRESS_SPACE     (DIR_ENTRIES * BYTES_PER_DIR_ENTRY)  // 4 GB
#endif // MLPT_CONFIG_H
```
### 3.2 Multi-Level Page Table Types (`mlpt_types.h`)
```c
#ifndef MLPT_TYPES_H
#define MLPT_TYPES_H
#include <stdint.h>
#include <stdbool.h>
#include "config.h"
#include "mlpt_config.h"
/*
 * Page Directory Entry (PDE)
 *
 * First-level entry pointing to a second-level page table.
 * Each PDE covers 4 MB of virtual address space.
 *
 * Memory layout:
 * ┌────────────────────────────────────────────────────────────┐
 * │ page_table_pfn (4) │ present (1) │ writable (1) │ pad (2)  │
 * └────────────────────────────────────────────────────────────┘
 *
 * WHY each field exists:
 * - page_table_pfn:  Physical frame number of the second-level page table.
 *                    This is an index into the physical memory frame array,
 *                    NOT a byte address.
 * - present:         Does this directory entry have an allocated page table?
 *                    If false, the entire 4 MB region is unmapped.
 * - writable:        Directory-level write permission. Combined with PTE's
 *                    writable bit (both must be true for write access).
 * - user_accessible: Can user-mode access this region? (Simplified: always true)
 *
 * Total size: 8 bytes (aligned for cache efficiency)
 */
typedef struct {
    uint32_t page_table_pfn;    // Offset 0x00, 4 bytes - PFN of page table
    bool present;               // Offset 0x04, 1 byte - Page table exists?
    bool writable;              // Offset 0x05, 1 byte - Write permission for region
    bool user_accessible;       // Offset 0x06, 1 byte - User-mode access (simplified)
    uint8_t _padding[1];        // Offset 0x07, 1 byte - Alignment padding
} pde_t;                        // Total: 8 bytes
_Static_assert(sizeof(pde_t) == 8, "PDE must be 8 bytes");
/*
 * Process Control Block (Simplified)
 *
 * Represents a single process with its own address space.
 * The CR3 field is the key to address space isolation.
 *
 * WHY each field exists:
 * - pid:           Unique process identifier for logging and debugging.
 * - cr3:           Physical frame number of this process's page directory.
 *                  This is the "root" of the address space.
 * - name:          Human-readable name for debugging output.
 * - page_directory: Pointer to the page directory array (in our simulated
 *                   physical memory). NULL if not yet allocated.
 */
typedef struct {
    uint32_t pid;                       // Offset 0x00, 4 bytes
    uint32_t cr3;                       // Offset 0x04, 4 bytes - Page directory PFN
    char name[32];                      // Offset 0x08, 32 bytes - Process name
    pde_t *page_directory;              // Offset 0x28, 8 bytes - Pointer to directory
} process_t;                            // Total: 48 bytes
/*
 * Multi-Level Page Table Simulator State
 *
 * Extends the basic simulator with multi-level structures.
 *
 * WHY each field exists:
 * - directories:     Array of page directory pointers, indexed by PFN.
 *                    directories[cr3] gives the current process's directory.
 * - page_tables:     Array of second-level page table pointers, indexed by PFN.
 *                    A page table's PFN is stored in its parent PDE.
 * - processes:       Array of process control blocks.
 * - num_processes:   Current number of processes.
 * - current_pid:     Which process is currently running.
 * - cr3:             Current CR3 value (cached for fast access).
 * - stats:           Extended statistics including table allocation counts.
 */
typedef struct {
    /* Page table storage (indexed by PFN) */
    pde_t **directories;                // Offset 0x00, 8 bytes
    pte_t **page_tables;                // Offset 0x08, 8 bytes
    /* Physical memory (from Milestone 1) */
    physical_memory_t phys_mem;         // Offset 0x10
    /* Process management */
    process_t *processes;               // Offset 0x??, 8 bytes
    uint32_t num_processes;             // Offset 0x??, 4 bytes
    uint32_t current_pid;               // Offset 0x??, 4 bytes
    uint32_t cr3;                       // Offset 0x??, 4 bytes - Current CR3
    /* Statistics */
    struct {
        uint64_t total_accesses;
        uint64_t page_faults;
        uint64_t protection_faults;
        uint64_t dir_not_present;       // NEW: PDE present=0 count
        uint64_t tables_allocated;      // NEW: Second-level tables created
        uint64_t pages_allocated;       // From demand paging
        uint64_t frames_used;
    } stats;
    /* TLB reference (from Milestone 2) */
    tlb_t *tlb;
} mlpt_simulator_t;
/*
 * Multi-Level Walk Result Codes
 */
typedef enum {
    WALK_OK,                    // Success: physical address valid
    WALK_DIR_NOT_PRESENT,       // PDE present=0: entire region unmapped
    WALK_PAGE_FAULT,            // PTE valid=0: specific page unmapped
    WALK_PROTECTION,            // Permission violation at any level
    WALK_OUT_OF_MEMORY          // No frames for page table allocation
} walk_result_t;
/*
 * Multi-Level Walk Output
 */
typedef struct {
    walk_result_t result;       // What happened?
    uint32_t physical_address;  // Valid only when result == WALK_OK
    uint32_t dir_index;         // Extracted directory index (for debugging)
    uint32_t table_index;       // Extracted table index (for debugging)
    uint32_t offset;            // Extracted offset (for debugging)
} walk_output_t;
/*
 * Memory Overhead Metrics
 */
typedef struct {
    uint64_t directory_bytes;       // Page directory size (always 8 KB)
    uint64_t page_table_bytes;      // Sum of all second-level tables
    uint64_t total_overhead;        // directory + page tables
    uint32_t num_page_tables;       // Count of allocated second-level tables
    uint32_t num_directories;       // Always 1 per process
    /* Comparison metrics */
    uint64_t flat_table_bytes;      // What flat table would cost
    double savings_factor;          // flat / multi-level
    double savings_percent;         // (1 - multi/flat) * 100
} memory_overhead_t;
#endif // MLPT_TYPES_H
```
### 3.3 Memory Layout Diagram
```
Two-Level Page Table Structure:
                         CR3 Register
                              │
                              │ (PFN of page directory)
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Page Directory (8 KB)                            │
│                        1024 × 8 bytes = 8192 bytes                      │
├─────────────────────────────────────────────────────────────────────────┤
│ Entry 0:   [pt_pfn=── | present=0 | ...]  ← Region 0x00000000 unmapped  │
├─────────────────────────────────────────────────────────────────────────┤
│ Entry 1:   [pt_pfn=42 | present=1 | ...]  ← Points to page table @ PFN42│
├─────────────────────────────────────────────────────────────────────────┤
│ Entry 2:   [pt_pfn=── | present=0 | ...]  ← Region 0x00800000 unmapped  │
├─────────────────────────────────────────────────────────────────────────┤
│ ...                                                                     │
├─────────────────────────────────────────────────────────────────────────┤
│ Entry 511: [pt_pfn=17 | present=1 | ...]  ← Stack region (high memory) │
├─────────────────────────────────────────────────────────────────────────┤
│ ...                                                                     │
├─────────────────────────────────────────────────────────────────────────┤
│ Entry 1023:[pt_pfn=── | present=0 | ...]                                │
└─────────────────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┴──────────────────┐
           │ Entry 1 points to PFN 42            │ Entry 511 points to PFN 17
           ▼                                     ▼
┌─────────────────────────┐           ┌─────────────────────────┐
│ Page Table @ PFN 42     │           │ Page Table @ PFN 17     │
│ 1024 × 12 = 12288 bytes │           │ 1024 × 12 = 12288 bytes │
├─────────────────────────┤           ├─────────────────────────┤
│ PTE 0: [pfn=100 | v=1]  │           │ PTE 0: [pfn=200 | v=1]  │
│ PTE 1: [pfn=101 | v=1]  │           │ PTE 1: [pfn=── | v=0]   │
│ ...                     │           │ ...                     │
│ PTE 1023: [pfn=...|...] │           │ PTE 1023: [pfn=...|...] │
└─────────────────────────┘           └─────────────────────────┘
           │                                     │
           ▼                                     ▼
    Physical Frames                      Physical Frames
    (User Data)                          (User Data)
Virtual Address Translation Example:
Virtual Address: 0x00401234
┌─────────────────────────────────────────────────────────────────────────┐
│ 31    22 │ 21        12 │ 11                                          0 │
│ 0000000001│ 000000000001 │ 0001001000110100                            │
│  Dir=1    │  Table=1     │  Offset=0x234                               │
└─────────────────────────────────────────────────────────────────────────┘
      │            │              │
      │            │              │
      ▼            ▼              ▼
┌──────────┐ ┌──────────┐ ┌──────────────────────────────────────────────┐
│directory │ │  table   │ │ Physical Address = (PFN << 12) | 0x234       │
 │  [1]    │ │   [1]    │ │                                              │
└──────────┘ └──────────┘ └──────────────────────────────────────────────┘
      │            │
      │            │
      ▼            ▼
   PDE @ 1     PTE @ 1
   pt_pfn=42   pfn=0x17
Physical Address = (0x17 << 12) | 0x234 = 0x170234
```
### 3.4 Bit Extraction Visualization
```
Virtual Address: 0x00401234 (binary representation)
Bit Position:  31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
               ─────────────────────────────────────────────────────────────────────────────────────
               │<── Directory Index (10 bits) ──>│<── Table Index (10 bits) ──>│<─ Offset (12 bits)─>│
               │  0  0  0  0  0  0  0  0  0  1 │  0  0  0  0  0  0  0  0  0  1 │ 0 0 1 0 0 0 1 1 0 1 0 0 │
               └────────────────────────────────────────────────────────────────────────────────────
Extraction:
  dir_index   = (0x00401234 >> 22) & 0x3FF = 0x1
  table_index = (0x00401234 >> 12) & 0x3FF = 0x1
  offset      =  0x00401234       & 0xFFF = 0x234
Verification:
  Reconstructed = (dir_index << 22) | (table_index << 12) | offset
                = (0x1 << 22) | (0x1 << 12) | 0x234
                = 0x00400000 | 0x00001000 | 0x234
                = 0x00401234  ✓
```
---
## 4. Interface Contracts
### 4.1 Multi-Level Simulator Lifecycle
```c
/*
 * mlpt_simulator_create
 *
 * Creates and initializes a multi-level page table simulator.
 *
 * Parameters:
 *   num_frames     - Number of physical frames for user data + page tables.
 *                    Page tables consume frames from this pool.
 *   max_processes  - Maximum number of concurrent processes.
 *
 * Returns:
 *   Pointer to initialized mlpt_simulator_t on success.
 *   NULL on failure (invalid parameters or memory allocation error).
 *
 * Postconditions:
 *   - directories array allocated with num_frames slots, all NULL
 *   - page_tables array allocated with num_frames slots, all NULL
 *   - phys_mem initialized with free frames
 *   - No processes created yet (call mlpt_process_create)
 *   - cr3 = 0 (invalid until first process created)
 *
 * Memory ownership:
 *   Caller owns returned pointer and must free with mlpt_simulator_destroy().
 */
mlpt_simulator_t* mlpt_simulator_create(uint32_t num_frames, uint32_t max_processes);
/*
 * mlpt_simulator_destroy
 *
 * Releases all resources associated with a multi-level simulator.
 *
 * Parameters:
 *   sim - Pointer to simulator created by mlpt_simulator_create().
 *         May be NULL (no-op).
 *
 * Side effects:
 *   - Frees all page directories
 *   - Frees all page tables
 *   - Frees all frame data
 *   - Frees process array
 *   - Frees simulator struct
 */
void mlpt_simulator_destroy(mlpt_simulator_t *sim);
```
### 4.2 Bit Extraction Functions
```c
/*
 * extract_directory_index
 *
 * Extracts the page directory index from a 32-bit virtual address.
 *
 * Parameters:
 *   va - Virtual address
 *
 * Returns:
 *   Directory index in range [0, 1023]
 *
 * Implementation:
 *   return (va >> DIR_SHIFT) & DIR_MASK;
 *
 * Time complexity: O(1)
 */
static inline uint32_t extract_directory_index(uint32_t va);
/*
 * extract_table_index
 *
 * Extracts the page table index from a 32-bit virtual address.
 *
 * Parameters:
 *   va - Virtual address
 *
 * Returns:
 *   Table index in range [0, 1023]
 *
 * Implementation:
 *   return (va >> TABLE_SHIFT) & TABLE_MASK;
 *
 * Time complexity: O(1)
 */
static inline uint32_t extract_table_index(uint32_t va);
/*
 * extract_offset
 *
 * Extracts the page offset from a 32-bit virtual address.
 *
 * Parameters:
 *   va - Virtual address
 *
 * Returns:
 *   Offset in range [0, 4095]
 *
 * Implementation:
 *   return va & PAGE_MASK;
 *
 * Time complexity: O(1)
 */
static inline uint32_t extract_offset(uint32_t va);
```
### 4.3 Two-Level Page Table Walk
```c
/*
 * mlpt_walk
 *
 * Performs a two-level page table walk to translate a virtual address.
 *
 * This is the core translation function for Milestone 3. It traverses:
 *   CR3 → page_directory[dir_index] → page_table[table_index] → physical_address
 *
 * Parameters:
 *   sim       - Multi-level simulator state
 *   va        - Virtual address to translate
 *   is_write  - true for write operation, false for read
 *
 * Returns:
 *   walk_output_t with:
 *   - result: WALK_OK, WALK_DIR_NOT_PRESENT, WALK_PAGE_FAULT, 
 *             WALK_PROTECTION, or WALK_OUT_OF_MEMORY
 *   - physical_address: Valid only when result == WALK_OK
 *   - dir_index, table_index, offset: Extracted components (for debugging)
 *
 * Walk steps:
 *   1. Extract dir_index, table_index, offset from va
 *   2. Get page directory: directory = sim->directories[sim->cr3]
 *   3. Check PDE: pde = &directory[dir_index]
 *      - If !pde->present: attempt on-demand allocation
 *      - If allocation fails: return WALK_OUT_OF_MEMORY
 *   4. Check PDE write permission if is_write
 *   5. Get page table: table = sim->page_tables[pde->page_table_pfn]
 *   6. Check PTE: pte = &table[table_index]
 *      - If !pte->valid: return WALK_PAGE_FAULT (demand paging handled by caller)
 *   7. Check PTE write permission if is_write
 *   8. Update access bits (referenced, dirty)
 *   9. Compose physical address: (pte->pfn << PAGE_SHIFT) | offset
 *  10. Return WALK_OK with physical_address
 *
 * Statistics updates:
 *   - sim->stats.total_accesses always incremented
 *   - sim->stats.dir_not_present incremented on PDE present=0
 *   - sim->stats.page_faults incremented on PTE valid=0
 *   - sim->stats.protection_faults incremented on permission violation
 *   - sim->stats.tables_allocated incremented on on-demand allocation
 *
 * Time complexity: O(1) for successful walk, O(1) + allocation for on-demand
 */
walk_output_t mlpt_walk(mlpt_simulator_t *sim, uint32_t va, bool is_write);
```
### 4.4 On-Demand Page Table Allocation
```c
/*
 * allocate_page_directory
 *
 * Allocates a page directory for a new process.
 *
 * Parameters:
 *   sim - Multi-level simulator state
 *
 * Returns:
 *   PFN of allocated page directory on success
 *   (uint32_t)-1 on failure (out of memory)
 *
 * Side effects:
 *   - Allocates a frame from phys_mem
 *   - Allocates and zero-initializes a pde_t array
 *   - Stores pointer in sim->directories[pfn]
 *   - All entries initialized with present=false
 *
 * Time complexity: O(1) + O(DIR_ENTRIES) for initialization
 */
uint32_t allocate_page_directory(mlpt_simulator_t *sim);
/*
 * allocate_page_table
 *
 * Allocates a second-level page table on demand.
 *
 * Called when a page walk encounters a PDE with present=false.
 *
 * Parameters:
 *   sim        - Multi-level simulator state
 *   directory  - Page directory containing the PDE
 *   dir_index  - Index of the PDE in the directory
 *
 * Returns:
 *   PFN of allocated page table on success
 *   (uint32_t)-1 on failure (out of memory)
 *
 * Side effects on success:
 *   - Allocates a frame from phys_mem
 *   - Allocates and zero-initializes a pte_t array
 *   - Stores pointer in sim->page_tables[pfn]
 *   - Updates PDE: present=true, page_table_pfn=pfn
 *   - Increments sim->stats.tables_allocated
 *
 * Precondition: directory != NULL, dir_index < DIR_ENTRIES
 * Postcondition: PDE at dir_index has present=true
 *
 * Idempotent: If PDE already present, returns existing PFN without allocation.
 *
 * Time complexity: O(1) + O(TABLE_ENTRIES) for initialization
 */
uint32_t allocate_page_table(mlpt_simulator_t *sim, pde_t *directory, uint32_t dir_index);
```
### 4.5 Process Management and Context Switch
```c
/*
 * mlpt_process_create
 *
 * Creates a new process with its own page directory.
 *
 * Parameters:
 *   sim  - Multi-level simulator state
 *   name - Human-readable process name (copied, max 31 chars)
 *
 * Returns:
 *   Pointer to process_t on success
 *   NULL on failure (out of memory or max processes reached)
 *
 * Side effects:
 *   - Allocates a page directory via allocate_page_directory()
 *   - Creates process_t entry with unique PID
 *   - Sets process->cr3 to the directory's PFN
 *   - Increments sim->num_processes
 *
 * Postconditions:
 *   - process->pid is unique
 *   - process->cr3 points to a valid page directory
 *   - process->page_directory pointer is valid
 *
 * Time complexity: O(1) + allocation cost
 */
process_t* mlpt_process_create(mlpt_simulator_t *sim, const char *name);
/*
 * mlpt_context_switch
 *
 * Switches to a different process's address space.
 *
 * Parameters:
 *   sim     - Multi-level simulator state
 *   new_pid - PID of the process to switch to
 *
 * Returns:
 *   true on success
 *   false if new_pid is invalid
 *
 * Side effects:
 *   - Updates sim->current_pid = new_pid
 *   - Updates sim->cr3 = process->cr3
 *   - If TLB exists: calls tlb_flush() or tlb_context_switch()
 *   - Logs context switch message
 *
 * CRITICAL: The CR3 change must be followed by TLB invalidation.
 * Without ASIDs: full TLB flush required.
 * With ASIDs: just change current_asid.
 *
 * Time complexity: O(1) + TLB flush cost
 */
bool mlpt_context_switch(mlpt_simulator_t *sim, uint32_t new_pid);
```
### 4.6 Memory Overhead Measurement
```c
/*
 * measure_memory_overhead
 *
 * Calculates memory overhead for the current process's page tables.
 *
 * Parameters:
 *   sim - Multi-level simulator state
 *
 * Returns:
 *   memory_overhead_t structure with:
 *   - directory_bytes: Always 8 KB (1024 × 8 bytes)
 *   - page_table_bytes: Sum of all allocated second-level tables
 *   - total_overhead: directory + page tables
 *   - num_page_tables: Count of PDEs with present=true
 *   - flat_table_bytes: 4 MB (what flat table would cost)
 *   - savings_factor: flat / multi-level
 *   - savings_percent: Percentage saved
 *
 * Algorithm:
 *   1. directory_bytes = DIR_ENTRIES * sizeof(pde_t)
 *   2. For each PDE in current directory:
 *      if present: page_table_bytes += TABLE_ENTRIES * sizeof(pte_t)
 *   3. total_overhead = directory_bytes + page_table_bytes
 *   4. flat_table_bytes = MAX_VPN * sizeof(pte_t) = 4 MB
 *   5. savings_factor = flat_table_bytes / total_overhead
 *   6. savings_percent = (1 - total_overhead / flat_table_bytes) * 100
 *
 * Time complexity: O(DIR_ENTRIES)
 */
memory_overhead_t measure_memory_overhead(mlpt_simulator_t *sim);
/*
 * print_overhead_comparison
 *
 * Prints a formatted comparison of flat vs multi-level overhead.
 *
 * Parameters:
 *   sim - Multi-level simulator state
 *
 * Output format:
 *   ========================================
 *       MEMORY OVERHEAD COMPARISON
 *   ========================================
 *   Process: <name> (PID <pid>)
 *   ----------------------------------------
 *   Flat page table:        <bytes> (<MB> MB)
 *   Multi-level total:      <bytes> (<KB> KB)
 *     - Page directory:     <bytes>
 *     - Page tables:        <bytes> (<count> tables)
 *   ----------------------------------------
 *   Space savings:          <factor>x (<percent>%)
 *   ========================================
 */
void print_overhead_comparison(mlpt_simulator_t *sim);
```
---
## 5. Algorithm Specification
### 5.1 Bit Extraction
```
ALGORITHM: extract_indices
INPUT:  va - 32-bit virtual address
OUTPUT: dir_index, table_index, offset
PROCEDURE:
  1. dir_index ← (va >> 22) AND 0x3FF
     // Shift right by 22 positions to move directory bits to LSB
     // Mask with 0x3FF (10 ones) to extract only 10 bits
     // Example: 0x00401234 >> 22 = 0x1
  2. table_index ← (va >> 12) AND 0x3FF
     // Shift right by 12 positions to move table bits to LSB
     // Mask with 0x3FF to extract only 10 bits
     // Example: 0x00401234 >> 12 = 0x401, AND 0x3FF = 0x1
  3. offset ← va AND 0xFFF
     // Mask with 0xFFF (12 ones) to extract bottom 12 bits
     // Example: 0x00401234 AND 0xFFF = 0x234
  4. RETURN (dir_index, table_index, offset)
INVARIANTS:
  - dir_index ∈ [0, 1023]
  - table_index ∈ [0, 1023]
  - offset ∈ [0, 4095]
  - va == (dir_index << 22) | (table_index << 12) | offset (recomposition)
EDGE CASES:
  - va = 0x00000000 → dir=0, table=0, offset=0
  - va = 0xFFFFFFFF → dir=1023, table=1023, offset=4095
  - va = 0x003FFFFF → dir=0, table=1023, offset=4095 (end of dir 0)
  - va = 0x00400000 → dir=1, table=0, offset=0 (start of dir 1)
```
### 5.2 Two-Level Page Table Walk
```
ALGORITHM: mlpt_walk
INPUT:  sim - Multi-level simulator state
        va - Virtual address
        is_write - Boolean for operation type
OUTPUT: walk_output_t with result and physical address
PROCEDURE:
  1. // Initialize output
     output ← ZERO_INITIALIZED
     sim->stats.total_accesses ← sim->stats.total_accesses + 1
  2. // Extract address components
     dir_index ← (va >> 22) AND 0x3FF
     table_index ← (va >> 12) AND 0x3FF
     offset ← va AND 0xFFF
     output.dir_index ← dir_index
     output.table_index ← table_index
     output.offset ← offset
  3. // Get current page directory
     directory ← sim->directories[sim->cr3]
     IF directory == NULL THEN
       // CR3 points to non-existent directory (should never happen)
       sim->stats.protection_faults ← sim->stats.protection_faults + 1
       output.result ← WALK_PROTECTION
       RETURN output
     END IF
  4. // Level 1: Check Page Directory Entry
     pde ← &directory[dir_index]
     IF pde->present == false THEN
       // Attempt on-demand allocation
       sim->stats.dir_not_present ← sim->stats.dir_not_present + 1
       pfn ← allocate_page_table(sim, directory, dir_index)
       IF pfn == (uint32_t)-1 THEN
         output.result ← WALK_OUT_OF_MEMORY
         RETURN output
       END IF
       // Re-fetch PDE after allocation
       pde ← &directory[dir_index]
     END IF
  5. // Check directory-level write permission
     IF is_write == true AND pde->writable == false THEN
       sim->stats.protection_faults ← sim->stats.protection_faults + 1
       output.result ← WALK_PROTECTION
       RETURN output
     END IF
  6. // Level 2: Get Page Table
     page_table ← sim->page_tables[pde->page_table_pfn]
     IF page_table == NULL THEN
       // PDE present but no page table (invariant violation)
       output.result ← WALK_PAGE_FAULT
       RETURN output
     END IF
  7. // Check Page Table Entry
     pte ← &page_table[table_index]
     IF pte->valid == false THEN
       sim->stats.page_faults ← sim->stats.page_faults + 1
       output.result ← WALK_PAGE_FAULT
       RETURN output
     END IF
  8. // Check page-level write permission
     IF is_write == true AND pte->writable == false THEN
       sim->stats.protection_faults ← sim->stats.protection_faults + 1
       output.result ← WALK_PROTECTION
       RETURN output
     END IF
  9. // Update access bits
     pte->referenced ← true
     IF is_write == true THEN
       pte->dirty ← true
     END IF
  10. // Compose physical address
      output.physical_address ← (pte->pfn << 12) OR offset
      output.result ← WALK_OK
      RETURN output
INVARIANTS AFTER EXECUTION:
  - If result == WALK_OK:
    * PTE valid=true, referenced=true
    * PTE dirty=true if is_write was true
    * physical_address is valid
  - If result == WALK_DIR_NOT_PRESENT (now handled by on-demand):
    * Page table was allocated
    * PDE now has present=true
  - If result == WALK_PAGE_FAULT:
    * PTE valid=false
    * Caller must handle demand paging
  - If result == WALK_PROTECTION:
    * No state changes
TIME COMPLEXITY:
  - Best case (all present): O(1)
  - With on-demand allocation: O(1) + O(TABLE_ENTRIES) for initialization
```
### 5.3 On-Demand Page Table Allocation
```
ALGORITHM: allocate_page_table
INPUT:  sim - Simulator state
        directory - Page directory pointer
        dir_index - Which PDE to populate
OUTPUT: PFN of allocated table, or -1 on failure
PROCEDURE:
  1. // Check if already allocated (idempotent)
     pde ← &directory[dir_index]
     IF pde->present == true THEN
       RETURN pde->page_table_pfn  // Already exists
     END IF
  2. // Allocate a frame for the page table
     pfn ← allocate_frame(&sim->phys_mem)
     IF pfn == (uint32_t)-1 THEN
       RETURN (uint32_t)-1  // Out of memory
     END IF
  3. // Allocate the page table array
     page_table ← calloc(TABLE_ENTRIES, sizeof(pte_t))
     IF page_table == NULL THEN
       // Rollback: return frame to free list
       free_frame(&sim->phys_mem, pfn)
       RETURN (uint32_t)-1
     END IF
  4. // Initialize page table entries
     FOR i ← 0 TO TABLE_ENTRIES - 1 DO
       page_table[i].valid ← false
       page_table[i].pfn ← 0
       page_table[i].readable ← true   // Default permissions
       page_table[i].writable ← true
       page_table[i].dirty ← false
       page_table[i].referenced ← false
     END FOR
  5. // Store in simulator's page table array
     sim->page_tables[pfn] ← page_table
  6. // Update the PDE
     pde->page_table_pfn ← pfn
     pde->present ← true
     pde->writable ← true
     pde->user_accessible ← true
  7. // Update statistics
     sim->stats.tables_allocated ← sim->stats.tables_allocated + 1
  8. RETURN pfn
INVARIANTS:
  - After success: PDE present=true, page_table_pfn valid
  - Page table is zero-initialized (all entries valid=false)
  - Idempotent: multiple calls with same dir_index return same PFN
ROLLBACK:
  - If calloc fails, frame is returned to free list
  - PDE remains present=false
```
### 5.4 Context Switch
```
ALGORITHM: mlpt_context_switch
INPUT:  sim - Simulator state
        new_pid - Process ID to switch to
OUTPUT: Boolean success
PROCEDURE:
  1. // Validate PID
     IF new_pid >= sim->num_processes THEN
       RETURN false
     END IF
  2. // Get process
     process ← &sim->processes[new_pid]
     IF process->page_directory == NULL THEN
       RETURN false  // Process not fully initialized
     END IF
  3. // Update CR3 (the key operation)
     old_cr3 ← sim->cr3
     sim->cr3 ← process->cr3
     sim->current_pid ← new_pid
  4. // Handle TLB coherency
     IF sim->tlb != NULL THEN
       // Option A: With ASID support
       tlb_context_switch(sim->tlb, new_pid)
       // Option B: Without ASID (full flush)
       // tlb_flush(sim->tlb)
     END IF
  5. // Log the switch
     PRINT "[Context Switch] PID %u -> PID %u (CR3: 0x%X -> 0x%X)\n",
           sim->current_pid, new_pid, old_cr3, sim->cr3
  6. RETURN true
INVARIANTS:
  - After switch: sim->cr3 points to new process's page directory
  - TLB entries from old process are either:
    * Invalidated (full flush)
    * Tagged with different ASID (won't match lookups)
  - Old process's page tables remain intact
CRITICAL: Without proper TLB handling, the new process could access
          the old process's cached translations → SECURITY BUG
```
---
## 6. Error Handling Matrix
| Error | Detected By | Recovery | User-Visible? | System State |
|-------|-------------|----------|---------------|--------------|
| CR3 points to NULL directory | `mlpt_walk` step 3 | Return `WALK_PROTECTION` | Yes, in statistics | Unchanged |
| PDE present=0 | `mlpt_walk` step 4 | Attempt on-demand allocation | Yes, "tables_allocated" stat | New table allocated if memory available |
| On-demand allocation fails | `allocate_page_table` step 2 | Return `WALK_OUT_OF_MEMORY` | Yes, error message | Unchanged, PDE still present=false |
| Page table array allocation fails | `allocate_page_table` step 3 | Rollback frame allocation, return -1 | Yes, error message | Frame returned to free list |
| PTE valid=0 | `mlpt_walk` step 7 | Return `WALK_PAGE_FAULT` | Yes, "page_faults" stat | Unchanged (caller handles demand paging) |
| Write to read-only page (PDE) | `mlpt_walk` step 5 | Return `WALK_PROTECTION` | Yes, "protection_faults" stat | Unchanged |
| Write to read-only page (PTE) | `mlpt_walk` step 8 | Return `WALK_PROTECTION` | Yes, "protection_faults" stat | Unchanged |
| Invalid PID in context switch | `mlpt_context_switch` step 1 | Return false | Yes, error message | Unchanged, no switch |
| Process directory not initialized | `mlpt_context_switch` step 2 | Return false | Yes, error message | Unchanged |
| Directory allocation fails | `allocate_page_directory` | Return -1 | Yes, error message | Process not created |
| Max processes reached | `mlpt_process_create` | Return NULL | Yes, error message | No new process |
---
## 7. Implementation Sequence with Checkpoints
### Phase 1: Define PDE Structure and Bit Extraction Macros (1-2 hours)
**Files:** `include/mlpt_config.h`, `include/mlpt_types.h`
**Tasks:**
1. Create `mlpt_config.h` with DIR_SHIFT, DIR_MASK, TABLE_SHIFT, TABLE_MASK
2. Define PDE structure with page_table_pfn, present, writable, user_accessible
3. Add static assertion for PDE size (8 bytes)
4. Define process_t and mlpt_simulator_t structures
5. Define walk_result_t and walk_output_t
6. Define memory_overhead_t for comparison metrics
**Checkpoint:**
```bash
# Compile header files only
gcc -c -fsyntax-only include/mlpt_config.h include/mlpt_types.h -I include
# Verify struct sizes
cat > test_sizes.c << 'EOF'
#include <stdio.h>
#include "mlpt_types.h"
int main() {
    printf("pde_t: %zu bytes\n", sizeof(pde_t));
    printf("process_t: %zu bytes\n", sizeof(process_t));
    printf("DIR_ENTRIES: %d\n", DIR_ENTRIES);
    printf("TABLE_ENTRIES: %d\n", TABLE_ENTRIES);
    printf("Bytes per dir entry coverage: %d MB\n", 
           BYTES_PER_DIR_ENTRY / (1024*1024));
    return 0;
}
EOF
gcc -o test_sizes test_sizes.c -I include && ./test_sizes
# Expected: pde_t: 8 bytes, DIR_ENTRIES: 1024, TABLE_ENTRIES: 1024
```
---
### Phase 2: Implement Directory and Table Index Extraction (0.5-1 hour)
**Files:** `src/mlpt_walk.c`
**Tasks:**
1. Implement `extract_directory_index(va)` as inline function
2. Implement `extract_table_index(va)` as inline function
3. Implement `extract_offset(va)` as inline function
4. Add compile-time verification of bit masks
**Checkpoint:**
```c
// Add to tests/test_bit_extraction.c
void test_bit_extraction(void) {
    // Test case 1: 0x00401234
    uint32_t va = 0x00401234;
    assert(extract_directory_index(va) == 0x1);
    assert(extract_table_index(va) == 0x1);
    assert(extract_offset(va) == 0x234);
    // Test case 2: Boundary between directories
    va = 0x003FFFFF;  // End of directory 0
    assert(extract_directory_index(va) == 0x0);
    assert(extract_table_index(va) == 0x3FF);
    assert(extract_offset(va) == 0xFFF);
    va = 0x00400000;  // Start of directory 1
    assert(extract_directory_index(va) == 0x1);
    assert(extract_table_index(va) == 0x0);
    assert(extract_offset(va) == 0x0);
    // Test case 3: Highest address
    va = 0xFFFFFFFF;
    assert(extract_directory_index(va) == 0x3FF);
    assert(extract_table_index(va) == 0x3FF);
    assert(extract_offset(va) == 0xFFF);
    // Test case 4: Zero address
    va = 0x00000000;
    assert(extract_directory_index(va) == 0x0);
    assert(extract_table_index(va) == 0x0);
    assert(extract_offset(va) == 0x0);
    // Test case 5: Recomposition
    for (int d = 0; d < 1024; d += 100) {
        for (int t = 0; t < 1024; t += 100) {
            for (int o = 0; o < 4096; o += 100) {
                uint32_t reconstructed = (d << 22) | (t << 12) | o;
                assert(extract_directory_index(reconstructed) == (uint32_t)d);
                assert(extract_table_index(reconstructed) == (uint32_t)t);
                assert(extract_offset(reconstructed) == (uint32_t)o);
            }
        }
    }
    printf("Bit extraction tests PASSED\n");
}
```
```bash
gcc -o test_bit_extraction tests/test_bit_extraction.c src/mlpt_walk.c -I include
./test_bit_extraction
# Expected: "Bit extraction tests PASSED"
```
---
### Phase 3: Implement Two-Level Page Table Walk (2-3 hours)
**Files:** `src/mlpt_walk.c`, `include/mlpt.h`
**Tasks:**
1. Implement `mlpt_simulator_create` and `mlpt_simulator_destroy`
2. Implement `mlpt_walk` with full two-level traversal
3. Handle WALK_DIR_NOT_PRESENT, WALK_PAGE_FAULT, WALK_PROTECTION cases
4. Update statistics in walk function
5. Implement permission checking at both levels
**Checkpoint:**
```c
// Add to tests/test_mlpt_walk.c
void test_mlpt_walk_basic(void) {
    mlpt_simulator_t *sim = mlpt_simulator_create(256, 4);
    assert(sim != NULL);
    // Create a process
    process_t *proc = mlpt_process_create(sim, "test");
    assert(proc != NULL);
    mlpt_context_switch(sim, proc->pid);
    // Access an address - should trigger on-demand allocation
    walk_output_t result = mlpt_walk(sim, 0x00401234, false);
    assert(result.result == WALK_PAGE_FAULT);  // PTE not yet valid
    assert(result.dir_index == 0x1);
    assert(result.table_index == 0x1);
    assert(result.offset == 0x234);
    // Verify PDE was allocated
    pde_t *dir = sim->directories[sim->cr3];
    assert(dir[0x1].present == true);
    assert(sim->stats.tables_allocated == 1);
    // Manually set up the PTE for testing
    pte_t *table = sim->page_tables[dir[0x1].page_table_pfn];
    table[0x1].valid = true;
    table[0x1].pfn = 0x42;
    table[0x1].readable = true;
    table[0x1].writable = true;
    // Now walk should succeed
    result = mlpt_walk(sim, 0x00401234, false);
    assert(result.result == WALK_OK);
    assert(result.physical_address == (0x42 << 12) | 0x234);
    mlpt_simulator_destroy(sim);
    printf("MLPT walk basic tests PASSED\n");
}
```
---
### Phase 4: Implement On-Demand Second-Level Table Allocation (2-3 hours)
**Files:** `src/mlpt_alloc.c`
**Tasks:**
1. Implement `allocate_page_directory`
2. Implement `allocate_page_table` with idempotent behavior
3. Handle frame allocation failure with rollback
4. Initialize page tables with all entries valid=false
5. Update statistics on allocation
**Checkpoint:**
```c
// Add to tests/test_on_demand.c
void test_on_demand_allocation(void) {
    mlpt_simulator_t *sim = mlpt_simulator_create(256, 4);
    process_t *proc = mlpt_process_create(sim, "test");
    mlpt_context_switch(sim, proc->pid);
    pde_t *dir = sim->directories[sim->cr3];
    // Directory 0 should not have a page table yet
    assert(dir[0].present == false);
    // Trigger allocation
    uint32_t pfn = allocate_page_table(sim, dir, 0);
    assert(pfn != (uint32_t)-1);
    assert(dir[0].present == true);
    assert(dir[0].page_table_pfn == pfn);
    assert(sim->stats.tables_allocated == 1);
    // Page table should be initialized
    pte_t *table = sim->page_tables[pfn];
    assert(table != NULL);
    for (int i = 0; i < TABLE_ENTRIES; i++) {
        assert(table[i].valid == false);
    }
    // Idempotent: second call should return same PFN
    uint32_t pfn2 = allocate_page_table(sim, dir, 0);
    assert(pfn2 == pfn);
    assert(sim->stats.tables_allocated == 1);  // Not incremented
    // Allocate multiple tables
    for (int i = 1; i <= 5; i++) {
        pfn = allocate_page_table(sim, dir, i);
        assert(pfn != (uint32_t)-1);
        assert(dir[i].present == true);
    }
    assert(sim->stats.tables_allocated == 6);
    mlpt_simulator_destroy(sim);
    printf("On-demand allocation tests PASSED\n");
}
```
---
### Phase 5: Implement CR3 Register and Context Switch (1-2 hours)
**Files:** `src/context_switch.c`
**Tasks:**
1. Implement `mlpt_process_create` with directory allocation
2. Implement `mlpt_context_switch` with CR3 update
3. Integrate with TLB flush (if TLB exists)
4. Log context switch operations
**Checkpoint:**
```c
// Add to tests/test_context_switch.c
void test_context_switch(void) {
    mlpt_simulator_t *sim = mlpt_simulator_create(256, 4);
    // Create two processes
    process_t *proc_a = mlpt_process_create(sim, "process_A");
    process_t *proc_b = mlpt_process_create(sim, "process_B");
    assert(proc_a->pid == 0);
    assert(proc_b->pid == 1);
    assert(proc_a->cr3 != proc_b->cr3);  // Different directories
    // Switch to process A
    assert(mlpt_context_switch(sim, proc_a->pid) == true);
    assert(sim->current_pid == proc_a->pid);
    assert(sim->cr3 == proc_a->cr3);
    // Access an address in process A
    walk_output_t result = mlpt_walk(sim, 0x00400000, false);
    // (Will trigger allocation and fault)
    // Set up a mapping in process A
    pde_t *dir_a = sim->directories[proc_a->cr3];
    allocate_page_table(sim, dir_a, 1);
    pte_t *table_a = sim->page_tables[dir_a[1].page_table_pfn];
    table_a[0].valid = true;
    table_a[0].pfn = 0x42;
    // Switch to process B
    assert(mlpt_context_switch(sim, proc_b->pid) == true);
    assert(sim->cr3 == proc_b->cr3);
    // Access same virtual address in process B
    // Should NOT see process A's mapping
    dir_a = sim->directories[proc_a->cr3];  // Still have reference
    pde_t *dir_b = sim->directories[proc_b->cr3];
    assert(dir_b[1].present == false);  // No table allocated yet
    // Allocate table in process B
    allocate_page_table(sim, dir_b, 1);
    pte_t *table_b = sim->page_tables[dir_b[1].page_table_pfn];
    table_b[0].valid = true;
    table_b[0].pfn = 0x99;  // Different frame!
    // Switch back to process A
    mlpt_context_switch(sim, proc_a->pid);
    result = mlpt_walk(sim, 0x00400000, false);
    assert(result.result == WALK_OK);
    assert(result.physical_address >> 12 == 0x42);  // Still A's mapping
    mlpt_simulator_destroy(sim);
    printf("Context switch tests PASSED\n");
}
```
---
### Phase 6: Integrate with TLB (1-2 hours)
**Files:** Updated `src/context_switch.c`, `src/mlpt_walk.c`
**Tasks:**
1. Add TLB pointer to mlpt_simulator_t
2. Call tlb_flush or tlb_context_switch on context switch
3. Call tlb_invalidate_entry when PTE is modified
4. Update walk to check TLB first (if present)
**Checkpoint:**
```c
void test_tlb_integration(void) {
    mlpt_simulator_t *sim = mlpt_simulator_create(256, 4);
    sim->tlb = tlb_create(32, NULL);  // Create TLB
    process_t *proc = mlpt_process_create(sim, "test");
    mlpt_context_switch(sim, proc->pid);
    // Set up a mapping
    pde_t *dir = sim->directories[sim->cr3];
    allocate_page_table(sim, dir, 1);
    pte_t *table = sim->page_tables[dir[1].page_table_pfn];
    table[0].valid = true;
    table[0].pfn = 0x42;
    // First access - TLB miss
    walk_output_t result = mlpt_walk(sim, 0x00400000, false);
    assert(result.result == WALK_OK);
    // Second access - should hit in TLB (if integrated)
    // (This requires TLB integration in walk function)
    // Context switch should flush TLB
    process_t *proc2 = mlpt_process_create(sim, "test2");
    mlpt_context_switch(sim, proc2->pid);
    // TLB should now be flushed or have different ASID
    mlpt_simulator_destroy(sim);
    printf("TLB integration tests PASSED\n");
}
```
---
### Phase 7: Implement Memory Overhead Measurement (1-2 hours)
**Files:** `src/memory_overhead.c`
**Tasks:**
1. Implement `measure_memory_overhead` function
2. Count present PDEs to determine table count
3. Calculate bytes for directory and tables
4. Compare against flat table (4 MB)
5. Implement `print_overhead_comparison`
**Checkpoint:**
```c
void test_memory_overhead(void) {
    mlpt_simulator_t *sim = mlpt_simulator_create(256, 4);
    process_t *proc = mlpt_process_create(sim, "test");
    mlpt_context_switch(sim, proc->pid);
    // Initially: only directory allocated
    memory_overhead_t overhead = measure_memory_overhead(sim);
    assert(overhead.num_page_tables == 0);
    assert(overhead.directory_bytes == 1024 * 8);  // 8 KB
    assert(overhead.page_table_bytes == 0);
    // Allocate 3 page tables (sparse access)
    pde_t *dir = sim->directories[sim->cr3];
    allocate_page_table(sim, dir, 0);    // Addresses 0x00000000-0x003FFFFF
    allocate_page_table(sim, dir, 1);    // Addresses 0x00400000-0x007FFFFF
    allocate_page_table(sim, dir, 511);  // Addresses 0x7FC00000-0x7FFFFFFF
    overhead = measure_memory_overhead(sim);
    assert(overhead.num_page_tables == 3);
    assert(overhead.page_table_bytes == 3 * 1024 * 12);  // 3 * 12 KB
    assert(overhead.total_overhead == 8 * 1024 + 36 * 1024);  // ~44 KB
    // Compare to flat table (4 MB)
    assert(overhead.flat_table_bytes == 1024 * 1024 * 12);  // ~12 MB
    assert(overhead.savings_factor > 200);  // At least 200x savings
    assert(overhead.savings_percent > 99);  // > 99% savings
    print_overhead_comparison(sim);
    mlpt_simulator_destroy(sim);
    printf("Memory overhead tests PASSED\n");
}
```
---
### Phase 8: Write Comparison Tests (2-3 hours)
**Files:** `tests/test_overhead.c`, `traces/sparse.trace`, `traces/dense.trace`
**Tasks:**
1. Create sparse access trace (few directories touched)
2. Create dense access trace (many directories touched)
3. Run both traces and compare overhead
4. Verify sparse access has much lower overhead
5. Test Bélády's anomaly demonstration (optional)
**Test Traces:**
`traces/sparse.trace`:
```
# Sparse access: only 4 distinct directories
# Directory 1: 0x00400000 (code)
R 0x00400000
R 0x00401000
R 0x00402000
# Directory 4: 0x01000000 (heap)
R 0x01000000
W 0x01001000
# Directory 64: 0x10000000 (shared library)
R 0x10000000
R 0x10001000
# Directory 511: 0x7FC00000 (stack)
R 0x7FC00000
W 0x7FC01000
# Re-access (should not allocate new tables)
R 0x00400000
R 0x01000000
```
`traces/dense.trace`:
```
# Dense access: sequential through many directories
R 0x00000000
R 0x00400000
R 0x00800000
R 0x00C00000
... (continue for 64+ directories)
```
**Checkpoint:**
```bash
./vm_sim traces/sparse.trace 64
# Expected: 4 tables allocated, ~44 KB overhead, >200x savings
./vm_sim traces/dense.trace 256
# Expected: Many tables allocated, lower savings factor
```
---
## 8. Test Specification
### 8.1 Unit Tests for Bit Extraction
| Test Case | Input VA | Expected Dir | Expected Table | Expected Offset |
|-----------|----------|--------------|----------------|-----------------|
| Zero address | 0x00000000 | 0 | 0 | 0 |
| First page, last byte | 0x00000FFF | 0 | 0 | 0xFFF |
| Second directory, first page | 0x00400000 | 1 | 0 | 0 |
| Arbitrary address | 0x00401234 | 1 | 1 | 0x234 |
| Directory boundary (end) | 0x003FFFFF | 0 | 0x3FF | 0xFFF |
| Directory boundary (start) | 0x00400000 | 1 | 0 | 0 |
| Last directory, last page | 0xFFFFF000 | 0x3FF | 0x3FF | 0 |
| Max address | 0xFFFFFFFF | 0x3FF | 0x3FF | 0xFFF |
### 8.2 Unit Tests for Two-Level Walk
| Test Case | Initial State | Input | Expected Result | Notes |
|-----------|---------------|-------|-----------------|-------|
| Walk to unmapped directory | PDE present=0 | VA in dir 5 | WALK_PAGE_FAULT after allocation | On-demand allocates table |
| Walk to unmapped page | PDE present=1, PTE valid=0 | VA | WALK_PAGE_FAULT | Caller handles |
| Successful walk | Both levels valid | VA | WALK_OK | Physical address returned |
| Write to read-only page | PTE writable=0 | VA, is_write=true | WALK_PROTECTION | |
| Out of memory | No free frames | VA in new dir | WALK_OUT_OF_MEMORY | |
| Multiple processes | Two processes | Same VA | Different physical | Isolation verified |
### 8.3 Unit Tests for On-Demand Allocation
| Test Case | Initial State | Action | Expected Result |
|-----------|---------------|--------|-----------------|
| First allocation | PDE present=0 | allocate_page_table | PFN returned, present=1 |
| Repeated allocation | PDE present=1 | allocate_page_table | Same PFN returned |
| Out of frames | free_count=0 | allocate_page_table | -1 returned |
| Multiple allocations | Empty directory | Allocate 5 tables | 5 distinct PFNs |
| Table initialization | After allocation | Check all PTEs | All valid=false |
### 8.4 Unit Tests for Context Switch
| Test Case | Initial State | Action | Expected Result |
|-----------|---------------|--------|-----------------|
| Basic switch | Running PID 0 | Switch to PID 1 | cr3 updated, current_pid=1 |
| Same process | Running PID 0 | Switch to PID 0 | No change (or allowed) |
| Invalid PID | 2 processes | Switch to PID 99 | Returns false |
| CR3 isolation | Two processes | Access same VA | Different physical |
| Switch back | After switch | Switch back | Original mapping works |
### 8.5 Integration Tests
**Sparse Address Space Test:**
```
Access 4 widely scattered addresses → expect 4 tables allocated
Expected overhead: 8 KB + 4 × 12 KB = 56 KB
Expected savings vs flat: 4 MB / 56 KB ≈ 71x
```
**Dense Address Space Test:**
```
Access 64 sequential directories → expect 64 tables allocated
Expected overhead: 8 KB + 64 × 12 KB = 776 KB
Expected savings vs flat: 4 MB / 776 KB ≈ 5x
```
**Process Isolation Test:**
```
Create two processes, map same VA to different PFNs
Verify each process sees only its own mapping
```
---
## 9. Performance Targets
| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| Bit extraction (inline) | < 5 CPU cycles | Single shift + mask |
| Two-level walk (cached) | < 50 cycles | 2 array lookups + checks |
| Two-level walk (on-demand) | ~1000 cycles | Allocation + initialization |
| Page table allocation | ~500 cycles | calloc + initialization loop |
| Context switch | < 100 cycles | CR3 update + TLB flush |
| Memory overhead (sparse) | < 100 KB | 4-5 tables for typical process |
| Memory overhead (dense) | Proportional to coverage | Linear in directories used |
| Savings factor (sparse) | > 50x | 4 MB / overhead |
| Savings factor (dense, 64 dirs) | > 4x | 4 MB / overhead |
**Cache Behavior Analysis:**
| Access Pattern | Cache Lines | Notes |
|----------------|-------------|-------|
| Walk to same directory | 1 (PDE) + 1 (PTE) | PDE cached after first access |
| Walk to different directory, same table | 1 (PDE) + 1 (PTE) | May share PTE cache line |
| Walk to different directory, different table | 1 (PDE) + 1 (PTE) | No sharing |
| Sequential VPNs in same directory | 1 (PDE) + varying (PTEs) | PDE cached, PTEs prefetch |
---
## 10. State Machine (Page Table Lifecycle)
```
                    ┌─────────────────────────────────────┐
                    │     MULTI-LEVEL STRUCTURE STATES    │
                    └─────────────────────────────────────┘
PAGE DIRECTORY ENTRY (PDE) States:
==================================
     ┌──────────────────┐
     │    NOT_PRESENT   │  present=false, page_table_pfn undefined
     │ (region unmapped)│
     └────────┬─────────┘
              │
              │ allocate_page_table() on first access
              │ to this 4 MB region
              ▼
     ┌──────────────────┐
     │    PRESENT       │  present=true, page_table_pfn valid
     │ (table allocated)│
     └──────────────────┘
PAGE TABLE ENTRY (PTE) States:
==============================
(Same as Milestone 1, but accessed via two-level walk)
     ┌──────────────────┐
     │    INVALID       │  valid=false
     └────────┬─────────┘
              │
              │ Demand paging on first access
              │ (via mlpt_walk returning WALK_PAGE_FAULT)
              ▼
     ┌──────────────────┐
     │    VALID-CLEAN   │  valid=true, dirty=false
     └────────┬─────────┘
              │
              │ Write access
              ▼
     ┌──────────────────┐
     │    VALID-DIRTY   │  valid=true, dirty=true
     └──────────────────┘
CONTEXT SWITCH States:
======================
     ┌──────────────────┐
     │   Process A      │  cr3 = A's directory PFN
     │   Running        │  TLB entries tagged with A's ASID
     └────────┬─────────┘
              │
              │ mlpt_context_switch(B)
              │
              ▼
     ┌──────────────────┐
     │   Process B      │  cr3 = B's directory PFN
     │   Running        │  TLB flushed or ASID changed
     └──────────────────┘
ILLEGAL TRANSITIONS:
====================
- PDE present=1 → present=0 without freeing page table (memory leak)
- PDE page_table_pfn changed while present=1 (dangling table)
- CR3 changed without TLB flush (stale translations)
- Process directory freed while process still exists (dangling CR3)
```
---
## 11. Hardware Soul: Cache and Branch Analysis
### 11.1 Cache Lines Touched Per Operation
| Operation | Cache Lines | Hot/Cold | Notes |
|-----------|-------------|----------|-------|
| PDE lookup | 1 line | Hot after first access to directory | 8 PDEs per 64-byte line |
| PTE lookup | 1 line | Depends on access pattern | ~5 PTEs per 64-byte line |
| Page table allocation | Many lines | Cold | calloc zeros all lines |
| Directory traversal (overhead calc) | 16 lines | Cold | Sequential scan of 1024 PDEs |
### 11.2 Branch Prediction
| Branch | Predictability | Misprediction Cost | Frequency |
|--------|----------------|-------------------|-----------|
| `if (!pde->present)` | Highly predictable | 15 cycles | Most directories unmapped initially, then present |
| `if (!pte->valid)` | Highly predictable | 15 cycles | Most pages valid after warmup |
| `if (is_write && !pte->writable)` | Highly predictable | 15 cycles | Most pages writable |
| `if (sim->tlb != NULL)` | Constant | 0 cycles | Always same result |
### 11.3 Memory Access Pattern
**Page Table Walk (two-level):**
- PDE access: Sequential in directory index space, but directory is contiguous in memory
- PTE access: Random in physical memory (page tables scattered across frames)
- Pattern is a "pointer chase": CR3 → PDE → PTE → data
**Comparison to Flat Table:**
- Flat: Single array access, contiguous memory
- Multi-level: Two+ indirect accesses, scattered memory
- TLB is critical: hit collapses all levels into O(1)
### 11.4 TLB Considerations
With multi-level tables, TLB becomes even more critical:
- Two-level walk = 2 memory accesses before data
- Four-level walk (x86-64) = 4 memory accesses before data
- TLB hit = 0 page table memory accesses
This is why TLB coverage (entries × page size) matters more for multi-level systems.
---
## 12. Sample Implementation: mlpt_walk
```c
#include "mlpt_types.h"
#include "mlpt_config.h"
#include "mlpt.h"
#include <stdio.h>
walk_output_t mlpt_walk(mlpt_simulator_t *sim, uint32_t va, bool is_write) {
    walk_output_t output = {0};
    // Step 1: Count access
    sim->stats.total_accesses++;
    // Step 2: Extract address components
    output.dir_index = (va >> DIR_SHIFT) & DIR_MASK;
    output.table_index = (va >> TABLE_SHIFT) & TABLE_MASK;
    output.offset = va & PAGE_MASK;
    // Step 3: Get current page directory
    pde_t *directory = sim->directories[sim->cr3];
    if (directory == NULL) {
        sim->stats.protection_faults++;
        output.result = WALK_PROTECTION;
        return output;
    }
    // Step 4: Level 1 - Check PDE
    pde_t *pde = &directory[output.dir_index];
    if (!pde->present) {
        // On-demand allocation
        sim->stats.dir_not_present++;
        uint32_t pfn = allocate_page_table(sim, directory, output.dir_index);
        if (pfn == (uint32_t)-1) {
            output.result = WALK_OUT_OF_MEMORY;
            return output;
        }
        // Re-fetch PDE after allocation
        pde = &directory[output.dir_index];
    }
    // Step 5: Check directory-level write permission
    if (is_write && !pde->writable) {
        sim->stats.protection_faults++;
        output.result = WALK_PROTECTION;
        return output;
    }
    // Step 6: Level 2 - Get page table
    pte_t *page_table = sim->page_tables[pde->page_table_pfn];
    if (page_table == NULL) {
        // Invariant violation - PDE present but no table
        output.result = WALK_PAGE_FAULT;
        return output;
    }
    // Step 7: Check PTE
    pte_t *pte = &page_table[output.table_index];
    if (!pte->valid) {
        sim->stats.page_faults++;
        output.result = WALK_PAGE_FAULT;
        return output;
    }
    // Step 8: Check page-level write permission
    if (is_write && !pte->writable) {
        sim->stats.protection_faults++;
        output.result = WALK_PROTECTION;
        return output;
    }
    // Step 9: Update access bits
    pte->referenced = true;
    if (is_write) {
        pte->dirty = true;
    }
    // Step 10: Compose physical address
    output.physical_address = (pte->pfn << PAGE_SHIFT) | output.offset;
    output.result = WALK_OK;
    return output;
}
```
---
## 13. Updated Makefile
```makefile
# Add to existing Makefile
# Milestone 3 sources
SRCS_M3 = src/mlpt_walk.c src/mlpt_alloc.c src/context_switch.c src/memory_overhead.c
# All sources
SRCS = $(SRCS_M1) $(SRCS_M2) $(SRCS_M3) src/main.c
# Test targets for Milestone 3
TEST_M3_BIT = test_bit_extraction
TEST_M3_WALK = test_mlpt_walk
TEST_M3_ONDEMAND = test_on_demand
TEST_M3_CONTEXT = test_context_switch
TEST_M3_OVERHEAD = test_overhead
TEST_M3_ALL = $(TEST_M3_BIT) $(TEST_M3_WALK) $(TEST_M3_ONDEMAND) $(TEST_M3_CONTEXT) $(TEST_M3_OVERHEAD)
.PHONY: test_m3
# Milestone 3 tests
$(TEST_M3_BIT): tests/test_bit_extraction.c src/mlpt_walk.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^
$(TEST_M3_WALK): tests/test_mlpt_walk.c src/mlpt_walk.c src/mlpt_alloc.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^
$(TEST_M3_ONDEMAND): tests/test_on_demand.c src/mlpt_alloc.c src/mlpt_walk.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^
$(TEST_M3_CONTEXT): tests/test_context_switch.c src/context_switch.c src/mlpt_walk.c src/mlpt_alloc.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^
$(TEST_M3_OVERHEAD): tests/test_overhead.c src/memory_overhead.c src/mlpt_walk.c src/mlpt_alloc.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^
test_m3: $(TEST_M3_ALL)
	@echo "=== Running Multi-Level Page Table Tests ==="
	./$(TEST_M3_BIT)
	./$(TEST_M3_WALK)
	./$(TEST_M3_ONDEMAND)
	./$(TEST_M3_CONTEXT)
	./$(TEST_M3_OVERHEAD)
	@echo "=== Multi-Level Tests Complete ==="
test: test_m1 test_m2 test_m3
	./$(TARGET) traces/sparse.trace 64
	./$(TARGET) traces/dense.trace 256
clean:
	rm -f $(OBJS) $(TARGET) $(TEST_M2_ALL) $(TEST_M3_ALL)
```
---
[[CRITERIA_JSON: {"module_id": "virtual-memory-sim-m3", "criteria": ["CR3 register simulation holds physical frame number (PFN) of the root page directory for the active process in uint32_t cr3 field", "Virtual address decomposition extracts directory index (bits 31-22 using >> 22 & 0x3FF), table index (bits 21-12 using >> 12 & 0x3FF), and offset (bits 11-0 using & 0xFFF) for two-level 32-bit page tables", "Page directory entry (pde_t) structure contains page_table_pfn (uint32_t), present (bool), writable (bool), and user_accessible (bool) fields with total size 8 bytes", "PDE with present=false indicates entire 4 MB region (1024 pages) is unmapped with no second-level table allocated", "Second-level page tables (1024 pte_t entries each) are allocated on demand via allocate_page_table() when first access to a directory's region occurs", "Page table walk traverses CR3 → page_directory[dir_index] → page_table[table_index] → PTE → physical_address in mlpt_walk() function", "Memory overhead measurement function measure_memory_overhead() calculates directory_bytes (always 8 KB) plus page_table_bytes (num_tables × 12 KB) as total_overhead", "Memory overhead comparison demonstrates multi-level tables use significantly less memory than flat tables for sparse address spaces with savings_factor > 10x and savings_percent reported", "Context switch function mlpt_context_switch() updates CR3 to point to new process's page directory and logs the switch with old and new CR3 values", "TLB flush (tlb_flush) or ASID switch (tlb_context_switch) occurs on context switch when TLB exists in simulator", "Statistics tracked include tables_allocated (uint64_t) in addition to page faults and protection faults, plus dir_not_present counter", "Translation function mlpt_walk() returns walk_output_t with result enum distinguishing WALK_OK, WALK_DIR_NOT_PRESENT, WALK_PAGE_FAULT, WALK_PROTECTION, and WALK_OUT_OF_MEMORY", "Walk output includes extracted dir_index, table_index, and offset fields for debugging and verification", "On-demand allocation function allocate_page_table() is idempotent (returns existing PFN if table already allocated) and initializes all PTEs with valid=false"]}]
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: virtual-memory-sim-m4 -->
# Technical Design Document: Page Replacement and Swap Simulation
**Module ID:** `virtual-memory-sim-m4`  
**Language:** C (BINDING)  
**Difficulty:** Advanced
---
## 1. Module Charter
This module implements page replacement algorithms (FIFO, LRU, Clock, Optimal) with simulated swap space for evicting and reloading pages when physical memory is exhausted. When `allocate_frame()` returns -1 (no free frames), the replacement algorithm selects a victim page to evict; dirty pages are written to swap before eviction, and evicted pages are reloaded from swap on subsequent page faults. This completes the virtual memory simulation by demonstrating memory overcommitment, the cost of dirty page write-back, and performance differences between replacement policies including Bélády's anomaly where FIFO can produce MORE faults with MORE frames.
**What it does:**
- Models physical memory as a fixed frame pool with configurable size
- Implements swap space as an array of slots storing evicted page contents
- Provides four replacement algorithms: FIFO (circular queue), LRU (timestamp tracking), Clock (referenced-bit sweeping), Optimal (lookahead for benchmarking)
- Tracks dirty pages and writes them to swap on eviction (clean pages are simply discarded)
- Reloads pages from swap when a page fault occurs on a previously evicted page
- Tracks working set size using a sliding window of recent VPNs
- Detects and reports thrashing when working set exceeds available frames
- Demonstrates Bélády's anomaly where FIFO fault count is non-monotonic with frame count
**What it does NOT do:**
- Actual disk I/O delays (swap operations are counted but instantaneous)
- Prefetching or readahead optimizations
- Memory compression before swap
- Priority-based page selection (mlock, mprotect effects)
**Upstream dependencies:** Milestone 1 (PTE structure, physical memory, translation), Milestone 2 (TLB invalidation on eviction), Milestone 3 (page table traversal for finding pages)
**Downstream dependencies:** None — this is the final module completing the simulator
**Invariants that must always hold:**
1. Dirty pages MUST be written to swap before eviction — failure means data loss
2. Swap slot allocation on eviction MUST succeed or be treated as fatal error
3. After eviction, PTE.valid=0 and swap_map[vpn].in_swap=true
4. On reload from swap, the same swap slot is freed and made available
5. Free frame count + allocated frame count == total frames (conservation)
6. Replacement algorithm state is synchronized with page table state (no orphan entries)
7. TLB entries for evicted pages are invalidated before PTE modification
---
## 2. File Structure
Create files in this exact order:
```
01. include/swap_config.h         — Swap space constants and limits
02. include/swap_types.h          — Swap slot, swap space, swap mapping structures
03. include/replacement_config.h  — Replacement algorithm enumeration and config
04. include/replacement_types.h   — FIFO, LRU, Clock, Optimal state structures
05. include/replacement.h         — Replacement algorithm interface declarations
06. include/working_set.h         — Working set tracker declarations
07. src/swap.c                    — Swap space operations (alloc, free, read, write)
08. src/repl_fifo.c               — FIFO replacement implementation
09. src/repl_lru.c                — LRU replacement implementation
10. src/repl_clock.c              — Clock (Second-Chance) replacement implementation
11. src/repl_optimal.c            — Optimal (Bélády's) replacement implementation
12. src/replacement_manager.c     — Unified replacement interface and eviction logic
13. src/working_set.c             — Working set tracking implementation
14. src/translate_with_repl.c     — Translation function with replacement support
15. src/thrashing.c               — Thrashing detection and reporting
16. src/comparison.c              — Algorithm comparison and Bélády demonstration
17. tests/test_swap.c             — Unit tests for swap space operations
18. tests/test_fifo.c             — Unit tests for FIFO replacement
19. tests/test_lru.c              — Unit tests for LRU replacement
20. tests/test_clock.c            — Unit tests for Clock replacement
21. tests/test_optimal.c          — Unit tests for Optimal replacement
22. tests/test_writeback.c        — Unit tests for dirty page write-back
23. tests/test_reload.c           — Unit tests for swap reload
24. tests/test_belady.c           — Unit tests for Bélády's anomaly demonstration
25. tests/test_thrashing.c        — Unit tests for working set and thrashing
26. traces/loop_3pages.trace      — Test trace for Bélády's anomaly
27. traces/sequential.trace       — Test trace for sequential access
28. traces/looping.trace          — Test trace for looping access
29. traces/random.trace           — Test trace for random access
30. Makefile                      — Updated build system
```
---
## 3. Complete Data Model
### 3.1 Swap Configuration (`swap_config.h`)
```c
#ifndef SWAP_CONFIG_H
#define SWAP_CONFIG_H
#include <stdint.h>
/*
 * Swap Space Configuration
 *
 * Swap simulates disk storage for evicted pages. Each slot holds
 * exactly one page (PAGE_SIZE bytes). The swap space size determines
 * how many pages can be evicted before swap exhaustion.
 *
 * Real systems: swap is typically 1-2x RAM size.
 * Our simulation: configurable, default 256 slots (1 MB with 4KB pages).
 */
/* Default swap configuration */
#define DEFAULT_SWAP_SLOTS      256
#define MAX_SWAP_SLOTS          4096
/* Swap slot identifier type */
typedef uint32_t swap_slot_id_t;
#define SWAP_SLOT_INVALID       ((swap_slot_id_t)-1)
/* Swap statistics tracking */
typedef struct {
    uint64_t writes;            // Pages written to swap (dirty evictions)
    uint64_t reads;             // Pages read from swap (reloads)
    uint64_t discards;          // Clean pages discarded without write
    uint64_t slot_allocations;  // Total slots allocated
    uint64_t slot_frees;        // Total slots freed
} swap_stats_t;
#endif // SWAP_CONFIG_H
```
### 3.2 Swap Types (`swap_types.h`)
```c
#ifndef SWAP_TYPES_H
#define SWAP_TYPES_H
#include <stdint.h>
#include <stdbool.h>
#include "config.h"
#include "swap_config.h"
/*
 * Swap Slot Structure
 *
 * Each swap slot stores the complete contents of one evicted page
 * along with metadata identifying which page it belongs to.
 *
 * Memory layout:
 * ┌─────────────────────────────────────────────────────────────────┐
 * │ data[4096] (4096 bytes) │ vpn (4) │ process_id (4) │ flags (4) │
 * └─────────────────────────────────────────────────────────────────┘
 *
 * WHY each field exists:
 * - data:          The actual page contents (4 KB). Preserved across
 *                  eviction/reload cycles so modifications aren't lost.
 * - vpn:           Virtual Page Number of the page stored here.
 *                  Used to verify correct page on reload.
 * - process_id:    Which process owns this page. Needed because swap
 *                  is global (shared across all processes).
 * - occupied:      Is this slot in use? False = free, True = allocated.
 *
 * Total size: 4096 + 4 + 4 + 1 = 4105 bytes, padded to 4108 for alignment.
 */
typedef struct {
    uint8_t data[PAGE_SIZE];        // Offset 0x0000, 4096 bytes - Page contents
    uint32_t vpn;                   // Offset 0x1000, 4 bytes - Virtual Page Number
    uint32_t process_id;            // Offset 0x1004, 4 bytes - Owning process
    bool occupied;                  // Offset 0x1008, 1 byte - Slot in use?
    uint8_t _padding[3];            // Offset 0x1009, 3 bytes - Alignment
} swap_slot_t;                      // Total: 4108 bytes
/*
 * Swap Space Structure
 *
 * Manages the pool of swap slots with a free list for O(1) allocation.
 *
 * WHY each field exists:
 * - slots:         Array of all swap slots. Fixed size at creation.
 * - total_slots:   Capacity of the swap space.
 * - free_list:     Stack of available slot indices. Enables O(1) alloc.
 * - free_count:    Number of entries in free_list. 0 = swap full.
 * - stats:         Performance counters for swap operations.
 */
typedef struct {
    swap_slot_t *slots;             // Offset 0x00, 8 bytes - Slot array
    uint32_t total_slots;           // Offset 0x08, 4 bytes - Capacity
    uint32_t *free_list;            // Offset 0x0C, 8 bytes - Free slot stack
    uint32_t free_count;            // Offset 0x14, 4 bytes - Available slots
    swap_stats_t stats;             // Offset 0x18, 40 bytes - Statistics
} swap_space_t;                     // Total: ~64 bytes + slot array
/*
 * Swap Mapping Structure
 *
 * Per-VPN tracking of swap state. Each virtual page has one of these
 * to track whether it has a copy in swap and where.
 *
 * WHY each field exists:
 * - swap_slot:     Index into swap_space_t.slots where this page's
 *                  data is stored. Valid only when in_swap=true.
 * - in_swap:       Does this page have a valid copy in swap?
 *                  False after reload, True after eviction.
 */
typedef struct {
    swap_slot_id_t swap_slot;       // Offset 0x00, 4 bytes - Slot index
    bool in_swap;                   // Offset 0x04, 1 byte - Has swap copy?
    uint8_t _padding[3];            // Offset 0x05, 3 bytes - Alignment
} swap_mapping_t;                   // Total: 8 bytes
#endif // SWAP_TYPES_H
```
### 3.3 Replacement Configuration (`replacement_config.h`)
```c
#ifndef REPLACEMENT_CONFIG_H
#define REPLACEMENT_CONFIG_H
#include <stdint.h>
/*
 * Replacement Algorithm Enumeration
 *
 * Identifies which algorithm to use for victim selection.
 */
typedef enum {
    REPL_FIFO,          // First In, First Out
    REPL_LRU,           // Least Recently Used
    REPL_CLOCK,         // Second-Chance (Clock) algorithm
    REPL_OPTIMAL        // Bélády's optimal algorithm (requires lookahead)
} repl_algo_t;
/* String names for logging */
static const char* REPL_ALGO_NAMES[] = {
    "FIFO",
    "LRU",
    "Clock",
    "Optimal"
};
/*
 * Replacement Statistics
 *
 * Counters for measuring replacement algorithm behavior.
 */
typedef struct {
    uint64_t evictions;             // Total pages evicted
    uint64_t dirty_evictions;       // Evictions that required swap write
    uint64_t clean_evictions;       // Evictions that were discarded
    uint64_t swap_reads;            // Pages reloaded from swap
    uint64_t page_faults;           // Total page faults (for this algorithm)
} repl_stats_t;
#endif // REPLACEMENT_CONFIG_H
```
### 3.4 Replacement Types (`replacement_types.h`)
```c
#ifndef REPLACEMENT_TYPES_H
#define REPLACEMENT_TYPES_H
#include <stdint.h>
#include <stdbool.h>
#include "replacement_config.h"
#include "types.h"
/*
 * FIFO Replacement State
 *
 * Uses a circular queue to track page load order.
 * Victim is always the page at the head (oldest loaded).
 *
 * WHY each field exists:
 * - queue:         Circular buffer of VPNs in load order.
 * - capacity:      Maximum entries in queue (== number of frames).
 * - head:          Index of oldest entry (next victim).
 * - tail:          Index where next entry will be inserted.
 * - count:         Current number of entries in queue.
 */
typedef struct {
    uint32_t *queue;                // Offset 0x00, 8 bytes - VPN circular buffer
    uint32_t capacity;              // Offset 0x08, 4 bytes - Buffer size
    uint32_t head;                  // Offset 0x0C, 4 bytes - Oldest entry index
    uint32_t tail;                  // Offset 0x10, 4 bytes - Next insertion index
    uint32_t count;                 // Offset 0x14, 4 bytes - Current entries
} fifo_state_t;                     // Total: 24 bytes + queue
/*
 * LRU Replacement State
 *
 * Tracks last access time for each loaded page.
 * Victim is the page with the oldest timestamp.
 *
 * WHY each field exists:
 * - vpns:          Array of VPNs currently in memory.
 * - timestamps:    Parallel array of last access times.
 * - count:         Number of valid entries in arrays.
 * - clock:         Monotonic counter incremented on each access.
 */
typedef struct {
    uint32_t *vpns;                 // Offset 0x00, 8 bytes - Loaded VPNs
    uint64_t *timestamps;           // Offset 0x08, 8 bytes - Last access times
    uint32_t count;                 // Offset 0x10, 4 bytes - Entry count
    uint32_t capacity;              // Offset 0x14, 4 bytes - Array capacity
    uint64_t clock;                 // Offset 0x18, 8 bytes - Monotonic timestamp
} lru_state_t;                      // Total: 32 bytes + arrays
/*
 * Clock (Second-Chance) Replacement State
 *
 * Uses a circular buffer with reference bits.
 * Hand sweeps through pages; referenced pages get a "second chance".
 *
 * WHY each field exists:
 * - vpns:          Array of VPNs currently in memory.
 * - referenced:    Parallel array of reference bit mirrors.
 * - count:         Number of valid entries.
 * - hand:          Current position of the clock hand.
 */
typedef struct {
    uint32_t *vpns;                 // Offset 0x00, 8 bytes - Loaded VPNs
    bool *referenced;               // Offset 0x08, 8 bytes - Reference bits
    uint32_t count;                 // Offset 0x10, 4 bytes - Entry count
    uint32_t capacity;              // Offset 0x14, 4 bytes - Array capacity
    uint32_t hand;                  // Offset 0x18, 4 bytes - Clock hand position
} clock_state_t;                    // Total: 28 bytes + arrays
/*
 * Optimal Replacement State
 *
 * Requires knowledge of future memory accesses.
 * Victim is the page used farthest in the future (or never).
 *
 * WHY each field exists:
 * - trace:         Pointer to the complete memory access trace.
 * - trace_length:  Number of entries in the trace.
 * - current_pos:   Current position in the trace (for lookahead).
 */
typedef struct {
    const void *trace;              // Offset 0x00, 8 bytes - Trace pointer
    size_t trace_length;            // Offset 0x08, 8 bytes - Trace size
    size_t current_pos;             // Offset 0x10, 8 bytes - Current position
} optimal_state_t;                  // Total: 24 bytes
/*
 * Unified Replacement Manager
 *
 * Bundles all replacement algorithm states and the current selection.
 * Only one algorithm's state is active at a time.
 */
typedef struct {
    repl_algo_t current_algo;       // Offset 0x00, 4 bytes - Active algorithm
    repl_stats_t stats;             // Offset 0x04, 40 bytes - Statistics
    /* Algorithm-specific state (union to save memory) */
    union {
        fifo_state_t fifo;
        lru_state_t lru;
        clock_state_t clock;
        optimal_state_t optimal;
    } state;                        // Offset 0x2C, ~32 bytes
    /* Back-reference to page table for dirty bit checking */
    pte_t *page_table;              // Offset 0x4C, 8 bytes
} replacement_manager_t;            // Total: ~84 bytes + arrays
#endif // REPLACEMENT_TYPES_H
```
### 3.5 Working Set Types (`working_set.h`)
```c
#ifndef WORKING_SET_H
#define WORKING_SET_H
#include <stdint.h>
#include <stdbool.h>
/*
 * Working Set Tracker
 *
 * Maintains a sliding window of recent VPNs accessed.
 * Used to detect thrashing when working set > available frames.
 *
 * WHY each field exists:
 * - recent_vpns:   Circular buffer of recently accessed VPNs.
 * - window_size:   Size of the sliding window (e.g., last 100 accesses).
 * - head:          Next write position in the circular buffer.
 * - count:         Number of valid entries (grows to window_size).
 */
typedef struct {
    uint32_t *recent_vpns;          // Offset 0x00, 8 bytes - VPN circular buffer
    uint32_t window_size;           // Offset 0x08, 4 bytes - Window size
    uint32_t head;                  // Offset 0x0C, 4 bytes - Write position
    uint32_t count;                 // Offset 0x10, 4 bytes - Valid entries
} working_set_t;                    // Total: 20 bytes + buffer
/*
 * Thrashing Detection Result
 */
typedef struct {
    bool is_thrashing;              // True if working set > frames
    uint32_t working_set_size;      // Distinct pages in window
    uint32_t available_frames;      // Physical frames available
    double pressure_ratio;          // working_set / frames (>1.0 = thrashing)
} thrashing_result_t;
#endif // WORKING_SET_H
```
### 3.6 Complete Simulator State with Replacement
```c
/*
 * Extended Statistics (for Milestone 4)
 */
typedef struct {
    /* From Milestone 1 */
    uint64_t total_accesses;
    uint64_t page_faults;
    uint64_t protection_faults;
    /* From Milestone 2 */
    uint64_t tlb_hits;
    uint64_t tlb_misses;
    /* From Milestone 4 (NEW) */
    uint64_t swap_writes;           // Dirty pages written to swap
    uint64_t swap_reads;            // Pages reloaded from swap
    uint64_t swap_discards;         // Clean pages discarded
    uint64_t evictions;             // Total pages evicted
    uint64_t dirty_evictions;       // Evictions requiring swap write
    uint64_t thrashing_warnings;    // Thrashing detection count
    /* Per-algorithm comparison (populated by comparison runner) */
    uint64_t fifo_faults;
    uint64_t lru_faults;
    uint64_t clock_faults;
    uint64_t optimal_faults;
} stats_t;
/*
 * Complete Simulator State
 */
typedef struct {
    /* From Milestones 1-3 */
    pte_t *page_table;
    physical_memory_t phys_mem;
    tlb_t *tlb;
    /* Frame-to-VPN reverse mapping (NEW in M4) */
    uint32_t *frame_to_vpn;         // Maps PFN → VPN for eviction
    uint32_t *frame_to_process;     // Maps PFN → process ID
    /* Milestone 4 additions */
    swap_space_t *swap;             // Swap space
    swap_mapping_t *swap_map;       // Per-VPN swap state
    replacement_manager_t *repl;    // Replacement manager
    working_set_t *working_set;     // Working set tracker
    stats_t stats;
} simulator_t;
```
### 3.7 Memory Layout Diagram
```
Page Replacement and Swap Data Flow:
┌─────────────────────────────────────────────────────────────────────────┐
│                        PHYSICAL MEMORY                                  │
│                      (Frame Pool: 64 frames)                            │
├─────────────────────────────────────────────────────────────────────────┤
│ Frame 0:  [User Data Page] ←─┐                                         │
│ Frame 1:  [User Data Page]   │                                         │
│ Frame 2:  [User Data Page]   │ mapped by                               │
│ ...                         ──┘ page_table[vpn].pfn                     │
│ Frame 63: [User Data Page]                                             │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    │ On eviction (all frames full, new page needed)
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      REPLACEMENT DECISION                               │
│                                                                         │
│   FIFO:   queue[head] = oldest loaded page                             │
│   LRU:    min(timestamps[]) = least recently used                       │
│   Clock:  first vpn with referenced=false after hand sweep              │
│   Optimal: farthest future use (or never)                               │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    │ Victim VPN selected
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      DIRTY BIT CHECK                                    │
│                                                                         │
│   if (page_table[victim_vpn].dirty == true):                           │
│       → WRITE to swap (preserve modifications)                          │
│   else:                                                                 │
│       → DISCARD (clean page, data unchanged)                            │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    │ If dirty
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        SWAP SPACE                                       │
│                      (256 slots × 4 KB)                                 │
├─────────────────────────────────────────────────────────────────────────┤
│ Slot 0:   [Page data │ vpn=0x10 │ pid=0 │ occupied=true]               │
│ Slot 1:   [Page data │ vpn=0x20 │ pid=0 │ occupied=true]               │
│ Slot 2:   [FREE]                                                       │
│ Slot 3:   [Page data │ vpn=0x05 │ pid=1 │ occupied=true]               │
│ ...                                                                     │
│ Slot 255: [FREE]                                                       │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    │ After eviction: swap_map[victim_vpn].in_swap = true
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        PAGE TABLE UPDATE                                │
│                                                                         │
│   page_table[victim_vpn].valid = false                                  │
│   page_table[victim_vpn].pfn = 0                                        │
│   tlb_invalidate_entry(victim_vpn)                                      │
│                                                                         │
│   // Frame now free for new page                                        │
│   page_table[new_vpn].valid = true                                      │
│   page_table[new_vpn].pfn = freed_frame                                 │
└─────────────────────────────────────────────────────────────────────────┘
Bélády's Anomaly Visualization (FIFO):
Access Pattern: 1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5
3 Frames (FIFO):
┌─────┬─────┬─────┬────────┐
│ Acc │ F0  │ F1  │ Fault? │
├─────┼─────┼─────┼────────┤
│  1  │  1  │  -  │  Yes   │
│  2  │  1  │  2  │  Yes   │
│  3  │  1  │  2  │  Yes   │
│  4  │  4  │  2  │  Yes   │ ← Evict 1 (oldest)
│  1  │  4  │  1  │  Yes   │ ← Evict 2
│  2  │  4  │  1  │  Yes   │ ← Evict 3
│  5  │  5  │  1  │  Yes   │ ← Evict 4
│  1  │  5  │  1  │   No   │
│  2  │  5  │  1  │   No   │
│  3  │  5  │  3  │  Yes   │ ← Evict 1
│  4  │  5  │  3  │  Yes   │ ← Evict 2
│  5  │  5  │  3  │   No   │
└─────┴─────┴─────┴────────┘
Total: 9 faults
4 Frames (FIFO):
┌─────┬─────┬─────┬─────┬─────┬────────┐
│ Acc │ F0  │ F1  │ F2  │ F3  │ Fault? │
├─────┼─────┼─────┼─────┼─────┼────────┤
│  1  │  1  │  -  │  -  │  -  │  Yes   │
│  2  │  1  │  2  │  -  │  -  │  Yes   │
│  3  │  1  │  2  │  3  │  -  │  Yes   │
│  4  │  1  │  2  │  3  │  4  │  Yes   │
│  1  │  1  │  2  │  3  │  4  │   No   │
│  2  │  1  │  2  │  3  │  4  │   No   │
│  5  │  5  │  2  │  3  │  4  │  Yes   │ ← Evict 1
│  1  │  5  │  1  │  3  │  4  │  Yes   │ ← Evict 2
│  2  │  5  │  1  │  2  │  4  │  Yes   │ ← Evict 3
│  3  │  5  │  1  │  2  │  3  │  Yes   │ ← Evict 4
│  4  │  5  │  1  │  2  │  3  │   No   │
│  5  │  5  │  1  │  2  │  3  │   No   │
└─────┴─────┴─────┴─────┴─────┴────────┘
Total: 10 faults ← MORE than with 3 frames!
```
---
## 4. Interface Contracts
### 4.1 Swap Space Operations
```c
/*
 * swap_create
 *
 * Creates and initializes a swap space with the specified number of slots.
 *
 * Parameters:
 *   num_slots - Number of 4 KB slots to allocate. Must be > 0.
 *
 * Returns:
 *   Pointer to initialized swap_space_t on success.
 *   NULL on failure (invalid parameters or memory allocation error).
 *
 * Postconditions:
 *   - slots array is allocated with num_slots entries
 *   - All slots have occupied=false
 *   - free_list contains all slot indices [0, num_slots)
 *   - free_count == num_slots
 *   - All stats counters are zero
 *
 * Memory ownership:
 *   Caller owns returned pointer and must free with swap_destroy().
 *
 * Time complexity: O(num_slots)
 */
swap_space_t* swap_create(uint32_t num_slots);
/*
 * swap_destroy
 *
 * Releases all resources associated with a swap space.
 *
 * Parameters:
 *   swap - Pointer to swap space created by swap_create().
 *          May be NULL (no-op).
 *
 * Side effects:
 *   - Frees slots array
 *   - Frees free_list array
 *   - Frees swap_space_t struct
 */
void swap_destroy(swap_space_t *swap);
/*
 * swap_alloc
 *
 * Allocates a free swap slot for storing an evicted page.
 *
 * Parameters:
 *   swap - Swap space state
 *
 * Returns:
 *   Slot index (0 to total_slots-1) on success
 *   SWAP_SLOT_INVALID (-1 cast to swap_slot_id_t) if swap is full
 *
 * Side effects:
 *   - Decrements swap->free_count
 *   - Marks slot as occupied
 *   - Increments swap->stats.slot_allocations
 *
 * Time complexity: O(1)
 */
swap_slot_id_t swap_alloc(swap_space_t *swap);
/*
 * swap_free
 *
 * Returns a swap slot to the free pool after page reload.
 *
 * Parameters:
 *   swap      - Swap space state
 *   slot_id   - Slot index to free
 *
 * Side effects:
 *   - Increments swap->free_count
 *   - Marks slot as not occupied
 *   - Increments swap->stats.slot_frees
 *
 * Precondition: slot_id < swap->total_slots and slot is occupied
 *
 * Time complexity: O(1)
 */
void swap_free(swap_space_t *swap, swap_slot_id_t slot_id);
/*
 * swap_write_page
 *
 * Writes a page's contents to a swap slot.
 *
 * Parameters:
 *   swap      - Swap space state
 *   slot_id   - Destination slot (must be allocated)
 *   page_data - Pointer to PAGE_SIZE bytes of page data
 *   vpn       - Virtual Page Number (for metadata)
 *   process_id - Owning process ID (for metadata)
 *
 * Side effects:
 *   - Copies page_data into swap->slots[slot_id].data
 *   - Sets slot metadata (vpn, process_id, occupied)
 *   - Increments swap->stats.writes
 *
 * Precondition: slot_id is valid and allocated
 *
 * Time complexity: O(1) (memcpy of 4 KB)
 */
void swap_write_page(swap_space_t *swap, 
                     swap_slot_id_t slot_id,
                     const uint8_t *page_data,
                     uint32_t vpn,
                     uint32_t process_id);
/*
 * swap_read_page
 *
 * Reads a page's contents from a swap slot.
 *
 * Parameters:
 *   swap      - Swap space state
 *   slot_id   - Source slot
 *   page_data - Destination buffer (must be PAGE_SIZE bytes)
 *
 * Side effects:
 *   - Copies swap->slots[slot_id].data into page_data
 *   - Increments swap->stats.reads
 *
 * Precondition: slot_id is valid and occupied
 *
 * Time complexity: O(1) (memcpy of 4 KB)
 */
void swap_read_page(swap_space_t *swap,
                    swap_slot_id_t slot_id,
                    uint8_t *page_data);
```
### 4.2 Replacement Algorithm Interface
```c
/*
 * replacement_init
 *
 * Initializes the replacement manager with the specified algorithm.
 *
 * Parameters:
 *   mgr       - Replacement manager to initialize
 *   algo      - Which algorithm to use
 *   num_frames - Number of physical frames (determines queue/array sizes)
 *   page_table - Back-reference for dirty bit checking
 *   trace     - Memory access trace (required for OPTIMAL only)
 *   trace_length - Length of trace (required for OPTIMAL only)
 *
 * Returns:
 *   true on success
 *   false on failure (invalid parameters or allocation error)
 *
 * Postconditions:
 *   - Algorithm-specific state is initialized
 *   - Stats are zeroed
 *   - current_algo is set
 */
bool replacement_init(replacement_manager_t *mgr,
                      repl_algo_t algo,
                      uint32_t num_frames,
                      pte_t *page_table,
                      const void *trace,
                      size_t trace_length);
/*
 * replacement_destroy
 *
 * Releases resources associated with a replacement manager.
 */
void replacement_destroy(replacement_manager_t *mgr);
/*
 * replacement_on_load
 *
 * Called when a page is loaded into a frame (after page fault handling).
 * Updates replacement algorithm state to track the new page.
 *
 * Parameters:
 *   mgr - Replacement manager state
 *   vpn - Virtual Page Number of loaded page
 *
 * Side effects (algorithm-dependent):
 *   - FIFO: Adds VPN to tail of queue
 *   - LRU: Adds VPN to array with current timestamp
 *   - Clock: Adds VPN to array with referenced=true
 *   - Optimal: No state change
 *
 * Time complexity: O(1) for FIFO/Clock, O(n) for LRU (array scan)
 */
void replacement_on_load(replacement_manager_t *mgr, uint32_t vpn);
/*
 * replacement_on_access
 *
 * Called when a page is accessed (hit in page table).
 * Updates replacement algorithm state for access tracking.
 *
 * Parameters:
 *   mgr       - Replacement manager state
 *   vpn       - Virtual Page Number of accessed page
 *   is_write  - true if write access
 *
 * Side effects (algorithm-dependent):
 *   - FIFO: No state change
 *   - LRU: Updates timestamp for this VPN
 *   - Clock: Sets referenced=true for this VPN
 *   - Optimal: Advances current_pos
 *
 * Time complexity: O(n) for LRU/Clock (array scan), O(1) for others
 */
void replacement_on_access(replacement_manager_t *mgr, uint32_t vpn, bool is_write);
/*
 * replacement_select_victim
 *
 * Selects a page to evict using the configured algorithm.
 *
 * Parameters:
 *   mgr - Replacement manager state
 *
 * Returns:
 *   VPN of the page to evict
 *   (uint32_t)-1 if no pages are loaded (should not happen in normal operation)
 *
 * Side effects:
 *   - FIFO: Advances head pointer
 *   - LRU: Removes entry from tracking array
 *   - Clock: Removes entry from tracking array
 *   - Optimal: Removes entry from tracking array
 *   - Increments mgr->stats.evictions
 *
 * CRITICAL: This function ONLY selects the victim. The caller must:
 *   1. Check dirty bit and write to swap if needed
 *   2. Update page table (valid=0)
 *   3. Invalidate TLB entry
 *   4. Update swap_map
 *
 * Time complexity:
 *   - FIFO: O(1)
 *   - LRU: O(n) for scan
 *   - Clock: O(n) worst case, O(1) amortized
 *   - Optimal: O(n × remaining_trace)
 */
uint32_t replacement_select_victim(replacement_manager_t *mgr);
```
### 4.3 Working Set and Thrashing
```c
/*
 * working_set_init
 *
 * Initializes a working set tracker.
 *
 * Parameters:
 *   ws          - Working set tracker to initialize
 *   window_size - Number of recent accesses to track
 *
 * Returns:
 *   true on success, false on allocation failure
 */
bool working_set_init(working_set_t *ws, uint32_t window_size);
/*
 * working_set_destroy
 *
 * Releases resources for a working set tracker.
 */
void working_set_destroy(working_set_t *ws);
/*
 * working_set_record
 *
 * Records a VPN access in the sliding window.
 *
 * Parameters:
 *   ws  - Working set tracker
 *   vpn - Virtual Page Number accessed
 *
 * Side effects:
 *   - Adds vpn to circular buffer at head position
 *   - Advances head (wrapping)
 *   - Increments count (up to window_size)
 *
 * Time complexity: O(1)
 */
void working_set_record(working_set_t *ws, uint32_t vpn);
/*
 * working_set_size
 *
 * Calculates the current working set size (distinct pages in window).
 *
 * Parameters:
 *   ws - Working set tracker
 *
 * Returns:
 *   Number of distinct VPNs in the sliding window
 *
 * Time complexity: O(window_size × log(window_size)) with sorting
 *                 O(window_size) with hash set
 */
uint32_t working_set_size(working_set_t *ws);
/*
 * check_thrashing
 *
 * Detects if the system is thrashing.
 *
 * Parameters:
 *   ws            - Working set tracker
 *   num_frames    - Number of available physical frames
 *
 * Returns:
 *   thrashing_result_t with:
 *   - is_thrashing: true if working_set_size > num_frames
 *   - working_set_size: Distinct pages in window
 *   - available_frames: num_frames (echoed)
 *   - pressure_ratio: working_set_size / num_frames
 *
 * Time complexity: O(window_size)
 */
thrashing_result_t check_thrashing(working_set_t *ws, uint32_t num_frames);
```
### 4.4 Frame Allocation with Eviction
```c
/*
 * allocate_frame_with_eviction
 *
 * Allocates a physical frame, evicting a page if necessary.
 *
 * This is the main entry point for frame allocation in Milestone 4.
 * It first tries the free list; if empty, it evicts a page using
 * the configured replacement algorithm.
 *
 * Parameters:
 *   sim   - Simulator state (contains page table, swap, replacement mgr)
 *   vpn   - VPN that needs a frame (for logging)
 *
 * Returns:
 *   PFN of allocated frame on success
 *   (uint32_t)-1 on failure (swap full, cannot evict)
 *
 * Side effects on eviction:
 *   1. Selects victim via replacement_select_victim()
 *   2. If victim is dirty:
 *      - Allocates swap slot
 *      - Writes page data to swap
 *      - Updates swap_map
 *      - Increments swap_writes stat
 *   3. If victim is clean:
 *      - Increments swap_discards stat
 *   4. Invalidates victim's PTE (valid=false)
 *   5. Invalidates victim's TLB entry
 *   6. Updates replacement state (removes victim)
 *   7. Returns victim's PFN
 *
 * Side effects on success (no eviction needed):
 *   - Decrements phys_mem.free_count
 *   - Allocates and zeros frame data
 *
 * CRITICAL: This function handles ALL eviction bookkeeping.
 *           The caller only needs to update the new page's PTE.
 *
 * Time complexity: O(1) if free frame, O(n) if eviction needed
 */
uint32_t allocate_frame_with_eviction(simulator_t *sim, uint32_t vpn);
/*
 * handle_page_fault_with_swap
 *
 * Extended page fault handler that checks for swap reload.
 *
 * Parameters:
 *   sim       - Simulator state
 *   vpn       - VPN that faulted
 *   is_write  - Is this a write access?
 *
 * Returns:
 *   true if fault was handled (frame allocated, page loaded)
 *   false if out of memory (allocation failed)
 *
 * Side effects:
 *   - Allocates frame (may trigger eviction)
 *   - If page is in swap:
 *     * Reads page data from swap slot
 *     * Frees the swap slot
 *     * Increments swap_reads stat
 *   - If page is NOT in swap (new page):
 *     * Zeros the frame (handled by allocate_frame)
 *   - Updates PTE (valid=true, pfn set)
 *   - Updates replacement state (on_load)
 *   - Updates working set tracker
 *
 * Time complexity: O(1) + O(eviction) + O(swap_read)
 */
bool handle_page_fault_with_swap(simulator_t *sim, uint32_t vpn, bool is_write);
```
### 4.5 Comparative Statistics
```c
/*
 * run_algorithm_comparison
 *
 * Runs all four replacement algorithms on the same trace and compares results.
 *
 * Parameters:
 *   trace        - Memory access trace
 *   trace_length - Number of entries in trace
 *   num_frames   - Number of physical frames
 *   num_swap_slots - Number of swap slots
 *
 * Output:
 *   Prints formatted comparison table to stdout:
 *
 *   ========================================
 *     PAGE REPLACEMENT ALGORITHM COMPARISON
 *   ========================================
 *   Trace length:      <count>
 *   Physical frames:   <count>
 *   ----------------------------------------
 *   Algorithm  Faults  Writes  Reads  Rate
 *   ----------------------------------------
 *   FIFO       <n>     <n>     <n>    <%>
 *   LRU        <n>     <n>     <n>    <%>
 *   Clock      <n>     <n>     <n>    <%>
 *   Optimal    <n>     <n>     <n>    <%>
 *   ----------------------------------------
 *   Optimal is the theoretical lower bound.
 *   ========================================
 */
void run_algorithm_comparison(const void *trace,
                              size_t trace_length,
                              uint32_t num_frames,
                              uint32_t num_swap_slots);
/*
 * demonstrate_belady_anomaly
 *
 * Demonstrates Bélády's anomaly with FIFO on a specific trace.
 *
 * Shows that FIFO with 4 frames can produce MORE page faults
 * than FIFO with 3 frames for certain access patterns.
 *
 * Output:
 *   Prints detailed trace of FIFO behavior with different frame counts.
 */
void demonstrate_belady_anomaly(void);
```
---
## 5. Algorithm Specification
### 5.1 FIFO Replacement
```
ALGORITHM: fifo_select_victim
INPUT:  mgr - Replacement manager with FIFO state
OUTPUT: VPN of victim page
PROCEDURE:
  1. // Check if any pages are loaded
     IF mgr->state.fifo.count == 0 THEN
       RETURN (uint32_t)-1  // No pages to evict
     END IF
  2. // Victim is at head of queue (oldest loaded)
     victim_vpn ← mgr->state.fifo.queue[mgr->state.fifo.head]
  3. // Advance head (circular)
     mgr->state.fifo.head ← (mgr->state.fifo.head + 1) MOD mgr->state.fifo.capacity
  4. // Decrement count
     mgr->state.fifo.count ← mgr->state.fifo.count - 1
  5. // Update statistics
     mgr->stats.evictions ← mgr->stats.evictions + 1
  6. RETURN victim_vpn
INVARIANTS:
  - Head always points to oldest loaded page
  - Count accurately reflects loaded pages
  - Queue is circular (wraps at capacity)
EDGE CASES:
  - Single page: head advances, count becomes 0
  - Full queue: head advances, count decrements
TIME COMPLEXITY: O(1)
```
### 5.2 LRU Replacement
```
ALGORITHM: lru_select_victim
INPUT:  mgr - Replacement manager with LRU state
OUTPUT: VPN of victim page
PROCEDURE:
  1. // Check if any pages are loaded
     IF mgr->state.lru.count == 0 THEN
       RETURN (uint32_t)-1
     END IF
  2. // Find entry with minimum timestamp
     oldest_time ← UINT64_MAX
     victim_idx ← 0
     FOR i ← 0 TO mgr->state.lru.count - 1 DO
       IF mgr->state.lru.timestamps[i] < oldest_time THEN
         oldest_time ← mgr->state.lru.timestamps[i]
         victim_idx ← i
       END IF
     END FOR
  3. // Get victim VPN
     victim_vpn ← mgr->state.lru.vpns[victim_idx]
  4. // Remove victim from arrays (swap with last element)
     last_idx ← mgr->state.lru.count - 1
     mgr->state.lru.vpns[victim_idx] ← mgr->state.lru.vpns[last_idx]
     mgr->state.lru.timestamps[victim_idx] ← mgr->state.lru.timestamps[last_idx]
     mgr->state.lru.count ← mgr->state.lru.count - 1
  5. // Update statistics
     mgr->stats.evictions ← mgr->stats.evictions + 1
  6. RETURN victim_vpn
INVARIANTS:
  - Each loaded page appears exactly once in vpns array
  - Timestamps are updated on every access
  - Minimum timestamp identifies LRU page
EDGE CASES:
  - All same timestamp: first entry evicted
  - Single page: count becomes 0 after removal
TIME COMPLEXITY: O(n) where n = number of loaded pages
```
### 5.3 Clock (Second-Chance) Replacement
```
ALGORITHM: clock_select_victim
INPUT:  mgr - Replacement manager with Clock state
OUTPUT: VPN of victim page
PROCEDURE:
  1. // Check if any pages are loaded
     IF mgr->state.clock.count == 0 THEN
       RETURN (uint32_t)-1
     END IF
  2. // Sweep through pages looking for victim
     pages_checked ← 0
     WHILE pages_checked < mgr->state.clock.count DO
       // Get current page at hand position
       current_vpn ← mgr->state.clock.vpns[mgr->state.clock.hand]
       current_ref ← mgr->state.clock.referenced[mgr->state.clock.hand]
       IF current_ref == false THEN
         // Found victim! It hasn't been referenced since last sweep
         GOTO FOUND_VICTIM
       END IF
       // Give this page a "second chance": clear reference bit
       mgr->state.clock.referenced[mgr->state.clock.hand] ← false
       // Also update the PTE's referenced bit (write-back)
       mgr->page_table[current_vpn].referenced ← true
       // Advance hand
       mgr->state.clock.hand ← (mgr->state.clock.hand + 1) MOD mgr->state.clock.count
       pages_checked ← pages_checked + 1
     END WHILE
  3. // All pages were referenced - evict the one at hand
     // (This is the "emergency" case)
     FOUND_VICTIM:
     victim_idx ← mgr->state.clock.hand
     victim_vpn ← mgr->state.clock.vpns[victim_idx]
  4. // Remove victim from arrays (shift or swap)
     // Using swap-with-last for simplicity:
     last_idx ← mgr->state.clock.count - 1
     mgr->state.clock.vpns[victim_idx] ← mgr->state.clock.vpns[last_idx]
     mgr->state.clock.referenced[victim_idx] ← mgr->state.clock.referenced[last_idx]
     mgr->state.clock.count ← mgr->state.clock.count - 1
     // Adjust hand if needed
     IF mgr->state.clock.hand >= mgr->state.clock.count AND mgr->state.clock.count > 0 THEN
       mgr->state.clock.hand ← 0
     END IF
  5. // Update statistics
     mgr->stats.evictions ← mgr->stats.evictions + 1
  6. RETURN victim_vpn
INVARIANTS:
  - Referenced bits mirror PTE referenced bits (synchronized on access)
  - Hand sweeps through all pages before returning to start
  - Page with ref=false is evicted; ref=true pages get "second chance"
EDGE CASES:
  - All ref=true: first page evicted after one full sweep
  - Single page: evicted regardless of ref bit if needed
TIME COMPLEXITY: O(n) worst case, O(1) amortized (most evictions quick)
```
### 5.4 Optimal (Bélády's) Replacement
```
ALGORITHM: optimal_select_victim
INPUT:  mgr - Replacement manager with Optimal state
OUTPUT: VPN of victim page
PROCEDURE:
  1. // Check if any pages are loaded
     IF mgr->state.optimal.current_loaded_count == 0 THEN
       RETURN (uint32_t)-1
     END IF
  2. // For each loaded page, find when it will next be accessed
     farthest_future ← -1  // -1 means "never"
     victim_vpn ← loaded_pages[0]
     FOR each vpn IN loaded_pages DO
       next_use ← find_next_access(vpn, mgr->state.optimal)
       IF next_use == -1 THEN
         // This page is NEVER used again - perfect victim!
         victim_vpn ← vpn
         GOTO FOUND_VICTIM
       END IF
       IF next_use > farthest_future THEN
         farthest_future ← next_use
         victim_vpn ← vpn
       END IF
     END FOR
  3. FOUND_VICTIM:
     // Remove victim from tracking
     remove_from_loaded_pages(victim_vpn)
  4. // Update statistics
     mgr->stats.evictions ← mgr->stats.evictions + 1
  5. RETURN victim_vpn
HELPER: find_next_access
INPUT:  vpn - Page to search for
        state - Optimal state with trace
OUTPUT: Index of next access, or -1 if never
PROCEDURE:
  FOR i ← state.current_pos TO state.trace_length - 1 DO
    IF trace[i].vpn == vpn THEN
      RETURN i
    END IF
  END FOR
  RETURN -1  // Never accessed again
INVARIANTS:
  - current_pos advances with each access
  - Lookahead never examines past accesses
  - "Never used again" is the best victim
EDGE CASES:
  - Multiple pages never used: first one found is victim
  - All pages used in future: farthest is victim
TIME COMPLEXITY: O(n × remaining_trace) - expensive!
NOTE: This is for BENCHMARKING only, not practical use.
      It requires complete knowledge of future accesses.
```
### 5.5 Dirty Page Write-Back and Eviction
```
ALGORITHM: evict_page
INPUT:  sim - Simulator state
        victim_vpn - Page to evict
OUTPUT: PFN of freed frame
PROCEDURE:
  1. // Get PTE for victim
     pte ← &sim->page_table[victim_vpn]
     victim_pfn ← pte->pfn
  2. // Check dirty bit
     IF pte->dirty == true THEN
       // Page has been modified - must write to swap
       // Allocate swap slot
       slot_id ← swap_alloc(sim->swap)
       IF slot_id == SWAP_SLOT_INVALID THEN
         // FATAL: Cannot evict dirty page without swap space
         RETURN (uint32_t)-1
       END IF
       // Write page data to swap
       frame_data ← sim->phys_mem.frames[victim_pfn]
       process_id ← sim->frame_to_process[victim_pfn]
       swap_write_page(sim->swap, slot_id, frame_data, victim_vpn, process_id)
       // Update swap mapping
       sim->swap_map[victim_vpn].swap_slot ← slot_id
       sim->swap_map[victim_vpn].in_swap ← true
       // Update statistics
       sim->stats.swap_writes ← sim->stats.swap_writes + 1
       sim->stats.dirty_evictions ← sim->stats.dirty_evictions + 1
       LOG "[Swap Write] VPN %u -> Slot %u\n", victim_vpn, slot_id
     ELSE
       // Page is clean - can discard without write
       sim->stats.swap_discards ← sim->stats.swap_discards + 1
       sim->stats.clean_evictions ← sim->stats.clean_evictions + 1
       LOG "[Swap Discard] VPN %u (clean)\n", victim_vpn
     END IF
  3. // Invalidate PTE
     pte->valid ← false
     pte->pfn ← 0
     pte->dirty ← false
     pte->referenced ← false
  4. // Invalidate TLB entry (if TLB exists)
     IF sim->tlb != NULL THEN
       tlb_invalidate_entry(sim->tlb, victim_vpn, current_asid)
     END IF
  5. // Clear reverse mapping
     sim->frame_to_vpn[victim_pfn] ← 0
     sim->frame_to_process[victim_pfn] ← 0
  6. // Update statistics
     sim->stats.evictions ← sim->stats.evictions + 1
  7. RETURN victim_pfn
INVARIANTS:
  - Dirty pages ALWAYS written to swap before eviction
  - PTE marked invalid after eviction
  - TLB entry removed (no stale translations)
  - Swap mapping updated for potential reload
CRITICAL: Steps 3 and 4 must happen BEFORE the frame is reused.
          Otherwise, concurrent lookups could see inconsistent state.
```
### 5.6 Page Reload from Swap
```
ALGORITHM: reload_from_swap
INPUT:  sim - Simulator state
        vpn - VPN that faulted (has swap copy)
        pfn - Newly allocated frame for this page
OUTPUT: None (side effects only)
PROCEDURE:
  1. // Verify page is in swap
     IF sim->swap_map[vpn].in_swap == false THEN
       ERROR "reload_from_swap called for page not in swap"
       RETURN
     END IF
  2. // Get swap slot
     slot_id ← sim->swap_map[vpn].swap_slot
  3. // Read page data from swap into frame
     frame_data ← sim->phys_mem.frames[pfn]
     swap_read_page(sim->swap, slot_id, frame_data)
  4. // Free the swap slot (page is now in memory)
     swap_free(sim->swap, slot_id)
     sim->swap_map[vpn].in_swap ← false
     sim->swap_map[vpn].swap_slot ← SWAP_SLOT_INVALID
  5. // Update statistics
     sim->stats.swap_reads ← sim->stats.swap_reads + 1
  6. LOG "[Swap Read] VPN %u from Slot %u to PFN %u\n", vpn, slot_id, pfn
INVARIANTS:
  - Swap slot is freed after reload (no duplicate storage)
  - Page data is preserved across eviction/reload cycle
  - in_swap flag is cleared after reload
```
---
## 6. Error Handling Matrix
| Error | Detected By | Recovery | User-Visible? | System State |
|-------|-------------|----------|---------------|--------------|
| Swap space full (swap_alloc returns INVALID) | `evict_page` step 2 | Return -1, abort eviction | Yes, "Swap full - cannot evict" error | Page not evicted, PTE unchanged |
| Page not in swap on reload attempt | `reload_from_swap` step 1 | Log error, return | Yes, error message | Frame allocated but contains zeros |
| Replacement algorithm returns invalid VPN | `allocate_frame_with_eviction` | Return -1, allocation fails | Yes, "No pages to evict" error | Unchanged |
| Frame-to-VPN mapping inconsistent | Various (sanity checks) | Log warning, continue | Yes, warning message | May have incorrect statistics |
| Working set tracker overflow | `working_set_record` | Wrap around (circular buffer) | No | Oldest entries overwritten |
| TLB invalidation fails | `evict_page` step 4 | Continue (TLB will miss on next access) | No | Potential stale entry (handled on miss) |
| Page fault on out-of-swap page | `handle_page_fault_with_swap` | Allocate zeroed frame | No | New page created |
| Replacement state corrupted (count < 0) | `replacement_select_victim` | Assert/crash | Yes, assertion failure | Fatal error |
| Double eviction of same page | `evict_page` (pte->valid check) | Skip eviction, log warning | Yes, warning | Frame returned but not used |
---
## 7. Implementation Sequence with Checkpoints
### Phase 1: Define Swap Space Structures and Operations (1-2 hours)
**Files:** `include/swap_config.h`, `include/swap_types.h`, `src/swap.c`
**Tasks:**
1. Create `swap_config.h` with constants and swap_stats_t
2. Create `swap_types.h` with swap_slot_t, swap_space_t, swap_mapping_t
3. Implement `swap_create` with slot array and free list initialization
4. Implement `swap_destroy`
5. Implement `swap_alloc` (stack pop from free list)
6. Implement `swap_free` (stack push to free list)
7. Implement `swap_write_page` (memcpy + metadata)
8. Implement `swap_read_page` (memcpy)
**Checkpoint:**
```c
// Add to tests/test_swap.c
void test_swap_basic(void) {
    swap_space_t *swap = swap_create(16);
    assert(swap != NULL);
    assert(swap->free_count == 16);
    // Allocate a slot
    swap_slot_id_t slot = swap_alloc(swap);
    assert(slot < 16);
    assert(swap->free_count == 15);
    assert(swap->slots[slot].occupied == true);
    // Write page data
    uint8_t test_data[PAGE_SIZE];
    memset(test_data, 0xAB, PAGE_SIZE);
    swap_write_page(swap, slot, test_data, 0x10, 0);
    assert(swap->slots[slot].vpn == 0x10);
    assert(swap->stats.writes == 1);
    // Read page data
    uint8_t read_data[PAGE_SIZE];
    swap_read_page(swap, slot, read_data);
    assert(memcmp(test_data, read_data, PAGE_SIZE) == 0);
    assert(swap->stats.reads == 1);
    // Free slot
    swap_free(swap, slot);
    assert(swap->slots[slot].occupied == false);
    assert(swap->free_count == 16);
    swap_destroy(swap);
    printf("Swap basic tests PASSED\n");
}
```
```bash
gcc -o test_swap tests/test_swap.c src/swap.c -I include
./test_swap
# Expected: "Swap basic tests PASSED"
```
---
### Phase 2: Implement FIFO Replacement (1-2 hours)
**Files:** `src/repl_fifo.c`
**Tasks:**
1. Implement `fifo_init` to allocate circular queue
2. Implement `fifo_on_load` to add VPN to tail
3. Implement `fifo_select_victim` to return head VPN
4. Track count to know how many pages are loaded
**Checkpoint:**
```c
void test_fifo_basic(void) {
    replacement_manager_t mgr;
    pte_t page_table[1024] = {0};
    assert(replacement_init(&mgr, REPL_FIFO, 4, page_table, NULL, 0));
    // Load pages in order: 1, 2, 3, 4
    replacement_on_load(&mgr, 1);
    replacement_on_load(&mgr, 2);
    replacement_on_load(&mgr, 3);
    replacement_on_load(&mgr, 4);
    assert(mgr.state.fifo.count == 4);
    // Evict should return 1 (oldest)
    uint32_t victim = replacement_select_victim(&mgr);
    assert(victim == 1);
    assert(mgr.state.fifo.count == 3);
    // Load another page
    replacement_on_load(&mgr, 5);
    assert(mgr.state.fifo.count == 4);
    // Evict should return 2 (next oldest)
    victim = replacement_select_victim(&mgr);
    assert(victim == 2);
    replacement_destroy(&mgr);
    printf("FIFO basic tests PASSED\n");
}
```
---
### Phase 3: Implement LRU Replacement (2-3 hours)
**Files:** `src/repl_lru.c`
**Tasks:**
1. Implement `lru_init` to allocate VPN and timestamp arrays
2. Implement `lru_on_load` to add entry with current timestamp
3. Implement `lru_on_access` to update timestamp
4. Implement `lru_select_victim` to scan for minimum timestamp
**Checkpoint:**
```c
void test_lru_basic(void) {
    replacement_manager_t mgr;
    pte_t page_table[1024] = {0};
    assert(replacement_init(&mgr, REPL_LRU, 4, page_table, NULL, 0));
    // Load pages
    replacement_on_load(&mgr, 1);
    replacement_on_load(&mgr, 2);
    replacement_on_load(&mgr, 3);
    // Access page 1 (makes it most recent)
    replacement_on_access(&mgr, 1, false);
    // Access page 3
    replacement_on_access(&mgr, 3, false);
    // Evict should return 2 (least recently used)
    uint32_t victim = replacement_select_victim(&mgr);
    assert(victim == 2);
    // Load another, access it, then evict
    replacement_on_load(&mgr, 4);
    replacement_on_access(&mgr, 4, false);
    // Page 1 and 3 still there, 1 was accessed before 3
    victim = replacement_select_victim(&mgr);
    assert(victim == 3);  // 1 was accessed more recently than 3
    replacement_destroy(&mgr);
    printf("LRU basic tests PASSED\n");
}
```
---
### Phase 4: Implement Clock Replacement (2-3 hours)
**Files:** `src/repl_clock.c`
**Tasks:**
1. Implement `clock_init` to allocate VPN and reference bit arrays
2. Implement `clock_on_load` to add entry with ref=true
3. Implement `clock_on_access` to set ref=true
4. Implement `clock_select_victim` with hand sweep and second-chance logic
**Checkpoint:**
```c
void test_clock_basic(void) {
    replacement_manager_t mgr;
    pte_t page_table[1024] = {0};
    assert(replacement_init(&mgr, REPL_CLOCK, 4, page_table, NULL, 0));
    // Load pages
    replacement_on_load(&mgr, 1);  // ref=true initially
    replacement_on_load(&mgr, 2);
    replacement_on_load(&mgr, 3);
    // Clear ref bit on page 1
    mgr.state.clock.referenced[0] = false;
    // Access page 2 (set ref=true)
    replacement_on_access(&mgr, 2, false);
    // Evict should skip 2 (ref=true), find 1 (ref=false)
    uint32_t victim = replacement_select_victim(&mgr);
    assert(victim == 1);  // First with ref=false
    // Load another, don't access it
    replacement_on_load(&mgr, 4);
    mgr.state.clock.referenced[mgr.state.clock.count - 1] = false;
    // Evict should find 3 or 4 (depending on hand position)
    victim = replacement_select_victim(&mgr);
    // After evicting 1, hand might point to 2 or 3
    // 2 has ref=true, so hand will clear it and continue
    replacement_destroy(&mgr);
    printf("Clock basic tests PASSED\n");
}
```
---
### Phase 5: Implement Optimal Replacement (2-3 hours)
**Files:** `src/repl_optimal.c`
**Tasks:**
1. Implement `optimal_init` to store trace pointer and length
2. Implement `optimal_on_load` to track loaded pages
3. Implement `optimal_on_access` to advance current_pos
4. Implement `optimal_select_victim` with lookahead scan
5. Implement `find_next_access` helper
**Checkpoint:**
```c
void test_optimal_basic(void) {
    // Trace: 1, 2, 3, 4, 1, 2, 5
    memory_access_t trace[] = {
        {false, 0x1000}, {false, 0x2000}, {false, 0x3000},
        {false, 0x4000}, {false, 0x1000}, {false, 0x2000},
        {false, 0x5000}
    };
    size_t trace_len = sizeof(trace) / sizeof(trace[0]);
    replacement_manager_t mgr;
    pte_t page_table[1024] = {0};
    assert(replacement_init(&mgr, REPL_OPTIMAL, 3, page_table, trace, trace_len));
    // Load pages 1, 2, 3 (VPNs extracted from addresses)
    replacement_on_load(&mgr, 1);
    replacement_on_load(&mgr, 2);
    replacement_on_load(&mgr, 3);
    // At position 3, next access for:
    // 1: position 4
    // 2: position 5
    // 3: never
    // Optimal should evict 3 (never used again)
    mgr.state.optimal.current_pos = 3;
    uint32_t victim = replacement_select_victim(&mgr);
    assert(victim == 3);
    replacement_destroy(&mgr);
    printf("Optimal basic tests PASSED\n");
}
```
---
### Phase 6: Implement Dirty Page Write-Back (1-2 hours)
**Files:** `src/replacement_manager.c`
**Tasks:**
1. Implement `evict_page` function
2. Check dirty bit and call swap_write_page if needed
3. Update swap_map
4. Invalidate PTE and TLB entry
5. Track statistics
**Checkpoint:**
```c
void test_writeback(void) {
    simulator_t *sim = simulator_create(4, 1);
    sim->swap = swap_create(16);
    sim->swap_map = calloc(1024, sizeof(swap_mapping_t));
    sim->repl = calloc(1, sizeof(replacement_manager_t));
    replacement_init(sim->repl, REPL_FIFO, 4, sim->page_table, NULL, 0);
    // Allocate and dirty a page
    translate_address(sim, 0x1000, true);  // Write, sets dirty
    uint32_t vpn = 1;
    assert(sim->page_table[vpn].dirty == true);
    // Manually trigger eviction
    uint32_t victim_pfn = evict_page(sim, vpn);
    assert(victim_pfn != (uint32_t)-1);
    // Verify swap write occurred
    assert(sim->stats.swap_writes == 1);
    assert(sim->swap_map[vpn].in_swap == true);
    // Verify PTE invalidated
    assert(sim->page_table[vpn].valid == false);
    swap_destroy(sim->swap);
    free(sim->swap_map);
    replacement_destroy(sim->repl);
    simulator_destroy(sim);
    printf("Write-back tests PASSED\n");
}
```
---
### Phase 7: Implement Page Reload from Swap (1-2 hours)
**Files:** `src/translate_with_repl.c`
**Tasks:**
1. Implement `handle_page_fault_with_swap` function
2. Check `swap_map[vpn].in_swap` on page fault
3. If in swap, call `reload_from_swap`
4. Otherwise, zero-fill the frame
5. Update PTE and replacement state
**Checkpoint:**
```c
void test_reload(void) {
    simulator_t *sim = simulator_create(4, 1);
    sim->swap = swap_create(16);
    sim->swap_map = calloc(1024, sizeof(swap_mapping_t));
    sim->repl = calloc(1, sizeof(replacement_manager_t));
    replacement_init(sim->repl, REPL_FIFO, 4, sim->page_table, NULL, 0);
    // Load and dirty a page
    translate_address(sim, 0x1000, true);
    // Write some data to the frame
    uint32_t pfn = sim->page_table[1].pfn;
    sim->phys_mem.frames[pfn][0] = 0x42;
    sim->phys_mem.frames[pfn][100] = 0x99;
    // Evict it
    evict_page(sim, 1);
    assert(sim->swap_map[1].in_swap == true);
    // Access again - should reload from swap
    trans_output_t result = translate_address(sim, 0x1000, false);
    assert(result.result == TRANS_OK);
    assert(sim->stats.swap_reads == 1);
    // Verify data was preserved
    pfn = sim->page_table[1].pfn;
    assert(sim->phys_mem.frames[pfn][0] == 0x42);
    assert(sim->phys_mem.frames[pfn][100] == 0x99);
    // Verify swap slot was freed
    assert(sim->swap_map[1].in_swap == false);
    swap_destroy(sim->swap);
    free(sim->swap_map);
    replacement_destroy(sim->repl);
    simulator_destroy(sim);
    printf("Reload tests PASSED\n");
}
```
---
### Phase 8: Implement Working Set Tracker (1-2 hours)
**Files:** `src/working_set.c`
**Tasks:**
1. Implement `working_set_init` with circular buffer allocation
2. Implement `working_set_record` to add VPN to window
3. Implement `working_set_size` to count distinct VPNs
4. Use hash set or sorting for distinct count
**Checkpoint:**
```c
void test_working_set(void) {
    working_set_t ws;
    assert(working_set_init(&ws, 10));
    // Record 10 accesses to 3 distinct pages
    working_set_record(&ws, 1);
    working_set_record(&ws, 2);
    working_set_record(&ws, 3);
    working_set_record(&ws, 1);
    working_set_record(&ws, 2);
    working_set_record(&ws, 3);
    working_set_record(&ws, 1);
    working_set_record(&ws, 2);
    working_set_record(&ws, 3);
    working_set_record(&ws, 1);
    uint32_t size = working_set_size(&ws);
    assert(size == 3);  // 3 distinct pages
    // Add more accesses, window should slide
    working_set_record(&ws, 4);
    working_set_record(&ws, 5);
    size = working_set_size(&ws);
    // Window now contains: 3, 1, 2, 3, 1, 2, 3, 1, 4, 5
    // Distinct: 1, 2, 3, 4, 5 = 5
    assert(size == 5);
    working_set_destroy(&ws);
    printf("Working set tests PASSED\n");
}
```
---
### Phase 9: Implement Thrashing Detection (0.5-1 hour)
**Files:** `src/thrashing.c`
**Tasks:**
1. Implement `check_thrashing` function
2. Compare working set size to frame count
3. Calculate pressure ratio
4. Log warning when thrashing detected
**Checkpoint:**
```c
void test_thrashing(void) {
    working_set_t ws;
    working_set_init(&ws, 10);
    // Record accesses to 5 distinct pages
    for (int i = 0; i < 10; i++) {
        working_set_record(&ws, (i % 5) + 1);
    }
    // With 3 frames: thrashing
    thrashing_result_t result = check_thrashing(&ws, 3);
    assert(result.is_thrashing == true);
    assert(result.working_set_size == 5);
    assert(result.pressure_ratio > 1.0);
    // With 10 frames: not thrashing
    result = check_thrashing(&ws, 10);
    assert(result.is_thrashing == false);
    assert(result.pressure_ratio <= 1.0);
    working_set_destroy(&ws);
    printf("Thrashing tests PASSED\n");
}
```
---
### Phase 10: Implement Comparative Statistics and Bélády Demonstration (2-3 hours)
**Files:** `src/comparison.c`
**Tasks:**
1. Implement `run_algorithm_comparison` function
2. Run trace through all four algorithms
3. Collect and display statistics
4. Implement `demonstrate_belady_anomaly` with specific trace
5. Show FIFO fault count for 3, 4, 5 frames
**Checkpoint:**
```c
void test_belady(void) {
    // Classic Bélády trace: 1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5
    memory_access_t trace[] = {
        {false, 0x1000}, {false, 0x2000}, {false, 0x3000},
        {false, 0x4000}, {false, 0x1000}, {false, 0x2000},
        {false, 0x5000}, {false, 0x1000}, {false, 0x2000},
        {false, 0x3000}, {false, 0x4000}, {false, 0x5000}
    };
    size_t trace_len = 12;
    // Run FIFO with 3 frames
    uint64_t faults_3 = run_single_algorithm(trace, trace_len, 3, REPL_FIFO);
    // Run FIFO with 4 frames
    uint64_t faults_4 = run_single_algorithm(trace, trace_len, 4, REPL_FIFO);
    printf("FIFO with 3 frames: %lu faults\n", faults_3);
    printf("FIFO with 4 frames: %lu faults\n", faults_4);
    // Bélády's anomaly: 4 frames should have MORE faults!
    assert(faults_4 > faults_3);
    // Verify LRU does NOT exhibit anomaly
    faults_3 = run_single_algorithm(trace, trace_len, 3, REPL_LRU);
    faults_4 = run_single_algorithm(trace, trace_len, 4, REPL_LRU);
    assert(faults_4 <= faults_3);  // LRU is monotonic
    printf("Bélády's anomaly tests PASSED\n");
}
```
---
### Phase 11: Write Comprehensive Tests (2-3 hours)
**Files:** All test files
**Tasks:**
1. Create test traces for different access patterns
2. Test all algorithms with same trace
3. Verify statistics are correct
4. Test swap full error handling
5. Test working set edge cases
**Test Traces:**
`traces/loop_3pages.trace`:
```
# Loop through 3 pages (Bélády test)
R 0x00001000
R 0x00002000
R 0x00003000
R 0x00004000
R 0x00001000
R 0x00002000
R 0x00005000
R 0x00001000
R 0x00002000
R 0x00003000
R 0x00004000
R 0x00005000
```
`traces/sequential.trace`:
```
# Sequential access through many pages
R 0x00001000
R 0x00002000
R 0x00003000
... (continue for 64 pages)
```
`traces/looping.trace`:
```
# Loop through same 4 pages
R 0x00001000
R 0x00002000
R 0x00003000
R 0x00004000
... (repeat 20 times)
```
**Checkpoint:**
```bash
make test
# Expected: All tests pass
./vm_sim traces/loop_3pages.trace 3 fifo
# Expected: 9 faults
./vm_sim traces/loop_3pages.trace 4 fifo
# Expected: 10 faults (Bélády's anomaly!)
./vm_sim traces/looping.trace 4 lru
# Expected: 4 faults (exactly one per unique page)
```
---
## 8. Test Specification
### 8.1 Unit Tests for Swap Operations
| Test Case | Initial State | Action | Expected Result |
|-----------|---------------|--------|-----------------|
| Allocate slot | Free slots available | swap_alloc() | Valid slot ID returned |
| Swap full | All slots occupied | swap_alloc() | SWAP_SLOT_INVALID returned |
| Write and read | Slot allocated | Write data, read back | Data matches |
| Free slot | Slot occupied | swap_free() | Slot becomes available |
| Metadata preserved | After write | Check slot.vpn | Matches written VPN |
### 8.2 Unit Tests for FIFO Replacement
| Test Case | Loaded Pages | Action | Expected Victim |
|-----------|--------------|--------|-----------------|
| Basic eviction | [1, 2, 3, 4] (in order) | select_victim | 1 (oldest) |
| After load | [2, 3, 4, 5] | select_victim | 2 |
| Empty queue | [] | select_victim | -1 (error) |
| Single page | [1] | select_victim | 1 |
### 8.3 Unit Tests for LRU Replacement
| Test Case | Loaded Pages | Access Pattern | Expected Victim |
|-----------|--------------|----------------|-----------------|
| Basic LRU | [1, 2, 3] | Access 1, 3 | 2 (least recent) |
| All accessed equally | [1, 2, 3] | Access all | First loaded (tie-break) |
| Recent access protects | [1, 2, 3] | Access 2 only | 1 or 3 (not 2) |
### 8.4 Unit Tests for Clock Replacement
| Test Case | Ref Bits | Action | Expected Victim |
|-----------|----------|--------|-----------------|
| Clear ref bit | [1:0, 2:1, 3:1] | sweep | 1 (ref=false) |
| All ref set | [1:1, 2:1, 3:1] | sweep | First (after clearing all) |
| Second chance | [1:1, 2:0, 3:1] | sweep | 2 (skips 1 with ref=1) |
### 8.5 Unit Tests for Dirty Page Write-Back
| Test Case | PTE State | Action | Expected Result |
|-----------|-----------|--------|-----------------|
| Dirty page | dirty=true | evict | swap_writes++ |
| Clean page | dirty=false | evict | swap_discards++ |
| Swap full | dirty=true | evict | Returns -1, no eviction |
### 8.6 Unit Tests for Bélády's Anomaly
| Frames | Algorithm | Trace | Expected Faults |
|--------|-----------|-------|-----------------|
| 3 | FIFO | 1,2,3,4,1,2,5,1,2,3,4,5 | 9 |
| 4 | FIFO | Same | 10 (MORE!) |
| 3 | LRU | Same | X |
| 4 | LRU | Same | ≤X (monotonic) |
---
## 9. Performance Targets
| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| FIFO victim selection | < 10 cycles | Queue head read + increment |
| LRU victim selection | O(n) scan | Timestamp comparison loop |
| Clock victim selection | O(1) amortized | Hand sweep with early exit |
| Optimal victim selection | O(n × trace) | Full lookahead scan |
| Swap write (4 KB) | ~500 cycles | memcpy + metadata update |
| Swap read (4 KB) | ~500 cycles | memcpy |
| Working set size calc | O(window_size) | Distinct count with hash |
| Full translation (hit) | < 100 cycles | TLB + PTE check |
| Full translation (fault + reload) | ~2000 cycles | Fault + eviction + swap read |
**Cache Behavior Analysis:**
| Data Structure | Cache Lines | Access Pattern |
|----------------|-------------|----------------|
| FIFO queue | 1-2 lines | Sequential, hot |
| LRU timestamps | n/8 lines | Sequential scan, warm |
| Clock vpns + refs | n/8 lines | Sequential sweep |
| Swap slot array | Many lines | Random access, cold |
| Working set buffer | 1-2 lines | Sequential, hot |
---
## 10. State Machine (Page Lifecycle with Swap)
```
                    ┌─────────────────────────────────────┐
                    │     PAGE LIFECYCLE WITH SWAP        │
                    └─────────────────────────────────────┘
     ┌──────────────────┐
     │    UNMAPPED      │  PTE.valid=false, not in swap
     │  (never loaded)  │  swap_map[vpn].in_swap=false
     └────────┬─────────┘
              │
              │ First access (page fault)
              │ handle_page_fault_with_swap()
              ▼
     ┌──────────────────┐
     │    IN_MEMORY     │  PTE.valid=true, in frame
     │    (clean)       │  PTE.dirty=false
     └────────┬─────────┘
              │
              │ Write access
              │ sets PTE.dirty=true
              ▼
     ┌──────────────────┐
     │    IN_MEMORY     │  PTE.valid=true, in frame
     │    (dirty)       │  PTE.dirty=true
     └────────┬─────────┘
              │
              │ Eviction (replacement algorithm)
              │ evict_page()
              │
              ├─────────────────────────────┐
              │                             │
              │ if dirty                    │ if clean
              ▼                             ▼
     ┌──────────────────┐         ┌──────────────────┐
     │    IN_SWAP       │         │    UNMAPPED      │
     │ (dirty evicted)  │         │ (clean evicted)  │
     │ in_swap=true     │         │ in_swap=false    │
     │ swap_slot valid  │         │ data lost        │
     └────────┬─────────┘         └──────────────────┘
              │
              │ Subsequent access (page fault)
              │ reload_from_swap()
              ▼
     ┌──────────────────┐
     │    IN_MEMORY     │  PTE.valid=true again
     │  (reloaded)      │  Data restored from swap
     │ in_swap=false    │  swap_slot freed
     └──────────────────┘
CRITICAL TRANSITIONS:
  IN_MEMORY (dirty) → IN_SWAP:  MUST write to swap before invalidating PTE
  IN_SWAP → IN_MEMORY:          MUST free swap slot after loading
  IN_MEMORY (clean) → UNMAPPED: Safe to discard (data unchanged)
ILLEGAL TRANSITIONS:
  IN_MEMORY → UNMAPPED (if dirty): Data loss!
  IN_SWAP → UNMAPPED: Data loss! (forgetting swap location)
  Any → IN_MEMORY without frame allocation: Invalid PTE
```
---
## 11. Hardware Soul: Cache and Branch Analysis
### 11.1 Cache Lines Touched Per Operation
| Operation | Cache Lines | Hot/Cold | Notes |
|-----------|-------------|----------|-------|
| FIFO queue access | 1-2 | Hot | Small, frequently accessed |
| LRU timestamp scan | n/8 (n=loaded pages) | Warm | Sequential scan prefetches |
| Clock array sweep | n/16 | Warm | vpns + refs interleaved |
| Swap slot allocation | 1 | Hot | Free list top cached |
| Swap page write | 64 (4 KB / 64 B) | Cold | Full page memcpy |
| Working set update | 1 | Hot | Circular buffer head |
| Working set size calc | window/8 | Warm | Sequential scan |
### 11.2 Branch Prediction
| Branch | Predictability | Misprediction Cost | Frequency |
|--------|----------------|-------------------|-----------|
| `if (pte->dirty)` in eviction | Moderate | 15 cycles | ~50% dirty in typical workloads |
| `if (free_count == 0)` | High | 15 cycles | Rarely true after warmup |
| `if (in_swap)` in fault handler | Moderate | 15 cycles | Depends on eviction rate |
| `if (referenced[i])` in Clock | Moderate | 15 cycles | ~70% true in typical workloads |
| `if (timestamps[i] < oldest)` in LRU | Unpredictable | 15 cycles | Random comparison results |
### 11.3 Memory Access Patterns
**FIFO:** Sequential queue access, excellent prefetch behavior.
**LRU:** Random timestamp comparison, cache-unfriendly scan.
**Clock:** Sequential sweep with random hand position, moderate locality.
**Swap:** Random slot access, large data (4 KB), completely cache-cold.
**Working set:** Circular buffer, sequential with wrap, prefetch-friendly.
### 11.4 Real-World Cost Comparison
| Operation | Simulated Cost | Real Hardware Cost |
|-----------|----------------|-------------------|
| Page fault (no swap) | ~500 cycles | ~10,000 cycles (kernel entry) |
| Swap write (4 KB) | ~500 cycles | 5-10 ms HDD, 50-100 μs SSD |
| Swap read (4 KB) | ~500 cycles | 5-10 ms HDD, 50-100 μs SSD |
| Eviction | ~1000 cycles | ~20 ms HDD, ~300 μs SSD |
| Thrashing detection | ~100 cycles | N/A (software) |
The simulator counts operations but doesn't delay for I/O. Real systems would stall for milliseconds on swap operations.
---
## 12. Sample Implementation: allocate_frame_with_eviction
```c
#include "types.h"
#include "swap_types.h"
#include "replacement_types.h"
#include "working_set.h"
uint32_t allocate_frame_with_eviction(simulator_t *sim, uint32_t vpn) {
    // Step 1: Try free list first
    if (sim->phys_mem.free_count > 0) {
        uint32_t pfn = sim->phys_mem.free_list[--sim->phys_mem.free_count];
        // Allocate frame data if not already allocated
        if (sim->phys_mem.frames[pfn] == NULL) {
            sim->phys_mem.frames[pfn] = calloc(PAGE_SIZE, 1);
            if (sim->phys_mem.frames[pfn] == NULL) {
                // Allocation failed - restore free list
                sim->phys_mem.free_list[sim->phys_mem.free_count++] = pfn;
                return (uint32_t)-1;
            }
        }
        return pfn;
    }
    // Step 2: No free frames - must evict
    printf("[Eviction] No free frames, selecting victim using %s\n",
           REPL_ALGO_NAMES[sim->repl->current_algo]);
    // Select victim using replacement algorithm
    uint32_t victim_vpn = replacement_select_victim(sim->repl);
    if (victim_vpn == (uint32_t)-1) {
        fprintf(stderr, "ERROR: No pages to evict!\n");
        return (uint32_t)-1;
    }
    pte_t *victim_pte = &sim->page_table[victim_vpn];
    uint32_t victim_pfn = victim_pte->pfn;
    printf("[Eviction] Victiming VPN %u (PFN %u), dirty=%d\n",
           victim_vpn, victim_pfn, victim_pte->dirty);
    // Step 3: Write to swap if dirty
    if (victim_pte->dirty) {
        swap_slot_id_t slot = swap_alloc(sim->swap);
        if (slot == SWAP_SLOT_INVALID) {
            fprintf(stderr, "FATAL: Swap full, cannot evict dirty page!\n");
            // Restore victim to replacement state (abort eviction)
            replacement_on_load(sim->repl, victim_vpn);
            return (uint32_t)-1;
        }
        // Write page data to swap
        uint8_t *frame_data = sim->phys_mem.frames[victim_pfn];
        uint32_t process_id = sim->frame_to_process[victim_pfn];
        swap_write_page(sim->swap, slot, frame_data, victim_vpn, process_id);
        // Update swap mapping
        sim->swap_map[victim_vpn].swap_slot = slot;
        sim->swap_map[victim_vpn].in_swap = true;
        // Update statistics
        sim->stats.swap_writes++;
        sim->stats.dirty_evictions++;
        printf("[Swap Write] VPN %u -> Slot %u\n", victim_vpn, slot);
    } else {
        sim->stats.swap_discards++;
        sim->stats.clean_evictions++;
        printf("[Swap Discard] VPN %u (clean)\n", victim_vpn);
    }
    // Step 4: Invalidate PTE
    victim_pte->valid = false;
    victim_pte->pfn = 0;
    victim_pte->dirty = false;
    victim_pte->referenced = false;
    // Step 5: Invalidate TLB entry
    if (sim->tlb != NULL) {
        tlb_invalidate_entry(sim->tlb, victim_vpn, 
                            sim->tlb->current_asid);
    }
    // Step 6: Clear reverse mapping
    sim->frame_to_vpn[victim_pfn] = 0;
    sim->frame_to_process[victim_pfn] = 0;
    // Step 7: Update statistics
    sim->stats.evictions++;
    // Step 8: Return the freed frame
    return victim_pfn;
}
```
---
## 13. Updated Makefile
```makefile
# Add to existing Makefile
CC = gcc
CFLAGS = -Wall -Wextra -Werror -std=c11 -g -O0
INCLUDES = -I include
# Milestone 4 sources
SRCS_M4 = src/swap.c \
          src/repl_fifo.c \
          src/repl_lru.c \
          src/repl_clock.c \
          src/repl_optimal.c \
          src/replacement_manager.c \
          src/working_set.c \
          src/translate_with_repl.c \
          src/thrashing.c \
          src/comparison.c
# All sources
SRCS = $(SRCS_M1) $(SRCS_M2) $(SRCS_M3) $(SRCS_M4) src/main.c
OBJS = $(SRCS:.c=.o)
TARGET = vm_sim
# Test targets for Milestone 4
TEST_M4_SWAP = test_swap
TEST_M4_FIFO = test_fifo
TEST_M4_LRU = test_lru
TEST_M4_CLOCK = test_clock
TEST_M4_OPTIMAL = test_optimal
TEST_M4_WRITEBACK = test_writeback
TEST_M4_RELOAD = test_reload
TEST_M4_BELADY = test_belady
TEST_M4_THRASHING = test_thrashing
TEST_M4_ALL = $(TEST_M4_SWAP) $(TEST_M4_FIFO) $(TEST_M4_LRU) \
              $(TEST_M4_CLOCK) $(TEST_M4_OPTIMAL) $(TEST_M4_WRITEBACK) \
              $(TEST_M4_RELOAD) $(TEST_M4_BELADY) $(TEST_M4_THRASHING)
.PHONY: test_m4
# Milestone 4 tests
$(TEST_M4_SWAP): tests/test_swap.c src/swap.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^
$(TEST_M4_FIFO): tests/test_fifo.c src/repl_fifo.c src/replacement_manager.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^
$(TEST_M4_LRU): tests/test_lru.c src/repl_lru.c src/replacement_manager.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^
$(TEST_M4_CLOCK): tests/test_clock.c src/repl_clock.c src/replacement_manager.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^
$(TEST_M4_OPTIMAL): tests/test_optimal.c src/repl_optimal.c src/replacement_manager.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^
$(TEST_M4_WRITEBACK): tests/test_writeback.c src/swap.c src/replacement_manager.c src/translate_with_repl.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^
$(TEST_M4_RELOAD): tests/test_reload.c src/swap.c src/replacement_manager.c src/translate_with_repl.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^
$(TEST_M4_BELADY): tests/test_belady.c src/comparison.c src/swap.c src/repl_fifo.c src/repl_lru.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^
$(TEST_M4_THRASHING): tests/test_thrashing.c src/working_set.c src/thrashing.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^
test_m4: $(TEST_M4_ALL)
	@echo "=== Running Milestone 4 Tests ==="
	./$(TEST_M4_SWAP)
	./$(TEST_M4_FIFO)
	./$(TEST_M4_LRU)
	./$(TEST_M4_CLOCK)
	./$(TEST_M4_OPTIMAL)
	./$(TEST_M4_WRITEBACK)
	./$(TEST_M4_RELOAD)
	./$(TEST_M4_BELADY)
	./$(TEST_M4_THRASHING)
	@echo "=== Milestone 4 Tests Complete ==="
test: test_m1 test_m2 test_m3 test_m4
	./$(TARGET) traces/loop_3pages.trace 3 fifo
	./$(TARGET) traces/loop_3pages.trace 4 fifo
	./$(TARGET) traces/looping.trace 4 lru
clean:
	rm -f $(OBJS) $(TARGET) $(TEST_M2_ALL) $(TEST_M3_ALL) $(TEST_M4_ALL)
```
---
[[CRITERIA_JSON: {"module_id": "virtual-memory-sim-m4", "criteria": ["Physical memory modeled as configurable frame pool (uint32_t total_frames, free_list array, free_count); page fault triggers replacement when free_count == 0", "Swap space structure (swap_slot_t with data[PAGE_SIZE], vpn, process_id, occupied fields) stores evicted page contents; swap_alloc() and swap_free() manage slot lifecycle", "FIFO replacement uses circular queue (head, tail, count fields in fifo_state_t); evict returns queue[head] with head = (head + 1) % capacity", "LRU replacement tracks last_used timestamp (uint64_t per page in lru_state_t); evict scans for minimum timestamp, returns that VPN", "Clock replacement uses circular buffer with referenced[] array in clock_state_t; evict clears referenced bits on first pass, evicts first page with referenced=false on second pass", "Optimal (Bélády's) replacement requires trace lookahead in optimal_state_t; evict selects page with farthest future access or never-accessed-again page", "Dirty page write-back: if pte->dirty on eviction, memcpy frame data to swap slot and increment stats.swap_writes; clean pages increment stats.swap_discards", "Page reload on fault: if swap_map[vpn].in_swap is true, memcpy from swap slot to frame, increment stats.swap_reads, free the swap slot", "Comparative statistics printed: page_faults, swap_writes, swap_reads per algorithm; fault rate as percentage of total_accesses", "Bélády's anomaly demonstration shows FIFO fault count non-monotonic with increasing frame count (4 frames can have MORE faults than 3); LRU faults decrease monotonically", "Working set tracker maintains sliding window of recent VPNs in working_set_t; working_set_size() returns count of distinct VPNs in window", "Thrashing detection compares working_set_size() against num_frames; logs warning when working set exceeds available memory", "Swap mapping structure (swap_mapping_t with swap_slot index and in_swap bool) tracks which pages have swap copies", "Statistics structure (stats_t) includes total_accesses, page_faults, swap_writes, swap_reads, swap_discards, thrashing_warnings as uint64_t fields", "Frame allocation with eviction (allocate_frame_with_eviction) tries free list first, then calls replacement_select_victim and handles dirty write-back", "TLB invalidation (tlb_invalidate_entry) called on eviction before PTE is modified to prevent stale translations"]}]
<!-- END_TDD_MOD -->


# Project Structure: Virtual Memory Simulator
## Directory Tree
```
virtual-memory-sim/
├── include/                          # Header files (all modules)
│   ├── config.h                      # M1: Page size, address space constants
│   ├── types.h                       # M1: PTE, physical_memory_t, stats_t, simulator_t
│   ├── translate.h                   # M1: Translation function declarations
│   ├── parser.h                      # M1: Trace parser declarations
│   ├── stats.h                       # M1: Statistics collection declarations
│   ├── tlb_config.h                  # M2: TLB capacity, ASID configuration
│   ├── tlb_types.h                   # M2: tlb_entry_t, tlb_t, tlb_stats_t
│   ├── tlb.h                         # M2: TLB public interface
│   ├── mlpt_config.h                 # M3: Multi-level bit masks, directory/table config
│   ├── mlpt_types.h                  # M3: PDE, process_t, mlpt_simulator_t
│   ├── mlpt.h                        # M3: Multi-level walk declarations
│   ├── working_set.h                 # M4: Working set tracker declarations
│   ├── swap_config.h                 # M4: Swap space constants, swap_stats_t
│   ├── swap_types.h                  # M4: swap_slot_t, swap_space_t, swap_mapping_t
│   ├── replacement_config.h          # M4: repl_algo_t enum, repl_stats_t
│   └── replacement_types.h           # M4: FIFO, LRU, Clock, Optimal state structs
├── src/                              # Source files
│   ├── translate.c                   # M1: Address translation, frame allocation
│   ├── parser.c                      # M1: Trace file parsing
│   ├── stats.c                       # M1: Statistics reporting
│   ├── main.c                        # M1: Entry point, CLI handling
│   ├── tlb.c                         # M2: TLB create, lookup, insert, flush
│   ├── tlb_translate.c               # M2: Translation with TLB integration
│   ├── mlpt_walk.c                   # M3: Two-level page table walk
│   ├── mlpt_alloc.c                  # M3: On-demand table allocation
│   ├── context_switch.c              # M3: CR3 management, process switching
│   ├── memory_overhead.c             # M3: Memory overhead measurement
│   ├── swap.c                        # M4: Swap space operations
│   ├── repl_fifo.c                   # M4: FIFO replacement
│   ├── repl_lru.c                    # M4: LRU replacement
│   ├── repl_clock.c                  # M4: Clock (Second-Chance) replacement
│   ├── repl_optimal.c                # M4: Optimal (Bélády's) replacement
│   ├── replacement_manager.c         # M4: Unified replacement interface
│   ├── working_set.c                 # M4: Working set tracking
│   ├── translate_with_repl.c         # M4: Translation with replacement support
│   ├── thrashing.c                   # M4: Thrashing detection
│   └── comparison.c                  # M4: Algorithm comparison, Bélády demo
├── tests/                            # Test files
│   ├── test_basic.c                  # M1: Address decomposition tests
│   ├── test_translation.c            # M1: Translation logic tests
│   ├── test_tlb_basic.c              # M2: TLB lookup/insertion tests
│   ├── test_tlb_lru.c                # M2: LRU victim selection tests
│   ├── test_tlb_coherency.c          # M2: TLB-page table coherency tests
│   ├── test_tlb_context.c            # M2: ASID context switch tests
│   ├── test_bit_extraction.c         # M3: Directory/table index extraction tests
│   ├── test_mlpt_walk.c              # M3: Two-level walk tests
│   ├── test_on_demand.c              # M3: On-demand allocation tests
│   ├── test_context_switch.c         # M3: CR3 and process isolation tests
│   ├── test_overhead.c               # M3: Memory overhead comparison tests
│   ├── test_swap.c                   # M4: Swap space operation tests
│   ├── test_fifo.c                   # M4: FIFO replacement tests
│   ├── test_lru.c                    # M4: LRU replacement tests
│   ├── test_clock.c                  # M4: Clock replacement tests
│   ├── test_optimal.c                # M4: Optimal replacement tests
│   ├── test_writeback.c              # M4: Dirty page write-back tests
│   ├── test_reload.c                 # M4: Swap reload tests
│   ├── test_belady.c                 # M4: Bélády's anomaly tests
│   └── test_thrashing.c              # M4: Working set/thrashing tests
├── traces/                           # Test trace files
│   ├── basic.trace                   # M1: Simple demand paging test
│   ├── dirty.trace                   # M1: Dirty bit tracking test
│   ├── protection.trace              # M1: Protection fault test
│   ├── tlb_locality.trace            # M2: High locality (high hit rate)
│   ├── tlb_random.trace              # M2: Random access (low hit rate)
│   ├── tlb_context.trace             # M2: Context switch test
│   ├── sparse.trace                  # M3: Sparse address access
│   ├── dense.trace                   # M3: Dense address access
│   ├── loop_3pages.trace             # M4: Bélády's anomaly test
│   ├── sequential.trace              # M4: Sequential access test
│   ├── looping.trace                 # M4: Looping access test
│   └── random.trace                  # M4: Random access test
├── diagrams/                         # SVG diagrams for documentation
├── Makefile                          # Build system
└── README.md                         # Project overview
```
## Creation Order
### Phase 1: Project Setup (15 min)
- Create directory structure (`include/`, `src/`, `tests/`, `traces/`)
- Create `Makefile` with basic targets
- Create `README.md` with project description
### Phase 2: Milestone 1 Foundation (3-4 hours)
1. `include/config.h` — Page size, masks, address space limits
2. `include/types.h` — PTE, physical_memory_t, stats_t, simulator_t
3. `include/translate.h` — Translation function declarations
4. `include/parser.h` — Trace parser declarations
5. `include/stats.h` — Statistics declarations
6. `src/translate.c` — Address decomposition, translation, frame allocation
7. `src/parser.c` — Trace file parsing
8. `src/stats.c` — Statistics reporting
9. `src/main.c` — Entry point, CLI
10. `tests/test_basic.c` — Address decomposition tests
11. `tests/test_translation.c` — Translation tests
12. `traces/basic.trace`, `traces/dirty.trace`, `traces/protection.trace`
### Phase 3: Milestone 2 TLB (2-3 hours)
1. `include/tlb_config.h` — TLB capacity, ASID config
2. `include/tlb_types.h` — tlb_entry_t, tlb_t, tlb_stats_t
3. `include/tlb.h` — TLB interface
4. `src/tlb.c` — TLB create, lookup, insert, flush, LRU
5. `src/tlb_translate.c` — Translation with TLB
6. `tests/test_tlb_basic.c` — Lookup/insert tests
7. `tests/test_tlb_lru.c` — LRU tests
8. `tests/test_tlb_coherency.c` — Coherency tests
9. `tests/test_tlb_context.c` — Context switch tests
10. `traces/tlb_locality.trace`, `traces/tlb_random.trace`, `traces/tlb_context.trace`
### Phase 4: Milestone 3 Multi-Level Tables (2-3 hours)
1. `include/mlpt_config.h` — Bit masks, directory/table config
2. `include/mlpt_types.h` — PDE, process_t, mlpt_simulator_t
3. `include/mlpt.h` — Multi-level walk declarations
4. `src/mlpt_walk.c` — Bit extraction, two-level walk
5. `src/mlpt_alloc.c` — On-demand table allocation
6. `src/context_switch.c` — CR3, process management
7. `src/memory_overhead.c` — Overhead measurement
8. `tests/test_bit_extraction.c` — Index extraction tests
9. `tests/test_mlpt_walk.c` — Walk tests
10. `tests/test_on_demand.c` — Allocation tests
11. `tests/test_context_switch.c` — Context switch tests
12. `tests/test_overhead.c` — Overhead comparison tests
13. `traces/sparse.trace`, `traces/dense.trace`
### Phase 5: Milestone 4 Replacement & Swap (3-4 hours)
1. `include/swap_config.h` — Swap constants
2. `include/swap_types.h` — swap_slot_t, swap_space_t, swap_mapping_t
3. `include/replacement_config.h` — repl_algo_t, repl_stats_t
4. `include/replacement_types.h` — FIFO/LRU/Clock/Optimal structs
5. `include/working_set.h` — Working set tracker
6. `src/swap.c` — Swap operations
7. `src/repl_fifo.c` — FIFO replacement
8. `src/repl_lru.c` — LRU replacement
9. `src/repl_clock.c` — Clock replacement
10. `src/repl_optimal.c` — Optimal replacement
11. `src/replacement_manager.c` — Unified interface
12. `src/working_set.c` — Working set tracking
13. `src/translate_with_repl.c` — Translation with replacement
14. `src/thrashing.c` — Thrashing detection
15. `src/comparison.c` — Algorithm comparison
16. `tests/test_swap.c` through `tests/test_thrashing.c`
17. `traces/loop_3pages.trace`, `traces/sequential.trace`, `traces/looping.trace`, `traces/random.trace`
### Phase 6: Integration Testing (1-2 hours)
- Run full test suite: `make test`
- Verify Bélády's anomaly demonstration
- Compare algorithm performance across traces
## File Count Summary
- **Total files**: 62
- **Header files**: 13
- **Source files**: 21
- **Test files**: 21
- **Trace files**: 12
- **Build/Docs**: 2 (Makefile, README.md)
- **Estimated lines of code**: ~8,000-10,000
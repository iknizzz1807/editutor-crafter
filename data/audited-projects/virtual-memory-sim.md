# AUDIT & FIX: virtual-memory-sim

## CRITIQUE
- CRITICAL: No mention of swap space or simulated disk I/O. Page replacement is meaningless without a backing store — when a page is evicted, where does its data go? When a page fault occurs on a previously evicted page, where does the data come from? The milestones treat page replacement as a pure algorithmic exercise disconnected from storage.
- Multi-level page tables milestone doesn't mention a CR3/Page Directory Base Register simulation. The walk must start somewhere — typically a register pointing to the root of the page table hierarchy. Without this, the student has no understanding of how context switching changes the address space.
- TLB milestone mentions 'ASID management' in pitfalls but doesn't require it in AC. ASIDs are what allow TLB entries to survive context switches without a full flush. This is a critical optimization concept.
- The Optimal (Bélády's) replacement algorithm is mentioned in the essence but not required in any AC. It's mentioned in M4 title/description area but the ACs list FIFO, LRU, and Clock only.
- Page Replacement M4 AC #4 says 'Track working set size' but working set tracking is a distinct concept from page replacement. It should be its own AC or removed to reduce scope confusion.
- The dirty bit is mentioned only in M1 (page table entry flags) but dirty page handling during eviction (write-back to swap) is never tested.
- No mention of access pattern trace files as input to the simulator. Without a defined input format, testing is ad hoc.
- M1 says 'Handle page faults by detecting invalid page table entries on lookup' but doesn't define what happens after detection (the fault handler is only in M4).
- Missing: demand paging concept — pages should be loaded on first access (lazy), not pre-loaded.

## FIXED YAML
```yaml
id: virtual-memory-sim
name: "Virtual Memory Simulator"
description: "Multi-level page tables, TLB caching, page replacement, and swap simulation"
difficulty: advanced
estimated_hours: "25-40"
essence: >
  Multi-level page table address translation with a simulated CR3 base register,
  TLB caching for fast virtual-to-physical mapping, demand paging with page fault
  handling, and page replacement algorithms (FIFO, LRU, Clock, Optimal) backed by
  simulated swap space for evicted dirty pages.
why_important: >
  Building this teaches you how operating systems provide memory isolation and
  efficient address translation — skills essential for systems programming, OS
  development, performance debugging of memory-intensive applications, and
  understanding why cache misses and page faults dominate real-world performance.
learning_outcomes:
  - Implement single-level and multi-level page table structures with address translation
  - Design a TLB cache with hit/miss handling, eviction, and ASID-based context switch support
  - Implement page replacement algorithms (FIFO, LRU, Clock, Optimal) with comparative analysis
  - Build a simulated swap space for storing evicted dirty pages and reloading them on fault
  - Simulate a CR3/PDBR register for page table base switching on context switch
  - Measure and compare TLB hit rates, page fault rates, and dirty page write-back counts
  - Process memory access trace files as simulator input for reproducible testing
skills:
  - Virtual memory management
  - Address translation
  - Page table structures
  - TLB caching
  - Page replacement algorithms
  - Swap space simulation
  - Performance profiling
tags:
  - advanced
  - c
  - page-tables
  - python
  - replacement
  - rust
  - tlb
  - swap
architecture_doc: architecture-docs/virtual-memory-sim/index.md
languages:
  recommended:
    - C
    - Rust
    - Python
  also_possible:
    - Go
    - Java
resources:
  - type: book
    name: "OSTEP - Address Translation"
    url: "https://pages.cs.wisc.edu/~remzi/OSTEP/vm-mechanism.pdf"
  - type: book
    name: "OSTEP - Paging: Faster Translations (TLBs)"
    url: "https://pages.cs.wisc.edu/~remzi/OSTEP/vm-tlbs.pdf"
  - type: book
    name: "OSTEP - Beyond Physical Memory: Mechanisms"
    url: "https://pages.cs.wisc.edu/~remzi/OSTEP/vm-beyondphys.pdf"
  - type: book
    name: "OSTEP - Page Replacement Policies"
    url: "https://pages.cs.wisc.edu/~remzi/OSTEP/vm-beyondphys-policy.pdf"
prerequisites:
  - type: skill
    name: "Binary/hex arithmetic and bit manipulation"
  - type: skill
    name: "Basic data structures (arrays, hash maps, linked lists)"
  - type: skill
    name: "Understanding of memory hierarchy (cache, RAM, disk)"
milestones:
  - id: virtual-memory-sim-m1
    name: "Single-Level Page Table and Address Translation"
    description: >
      Implement a flat page table with address translation, permission checking,
      and page fault detection. Define the trace file input format.
    acceptance_criteria:
      - Define a memory access trace format (e.g., 'R 0x1A3F' or 'W 0x4B00') specifying read/write and virtual address; simulator reads trace file as input
      - Virtual address is split into virtual page number (VPN) and offset using configurable page size (default 4KB = 12-bit offset)
      - Page table entry (PTE) contains physical frame number, valid bit, dirty bit, referenced bit, and read/write permission bits
      - Valid address translation returns concatenation of physical frame number + offset
      - Access to a page with valid=0 triggers a page fault; the fault is logged and the page is loaded (demand paging — allocate a free frame and set valid=1)
      - Write access to a read-only page triggers a protection fault and is logged as an error
      - Write access sets the dirty bit on the PTE; any access sets the referenced bit
      - Statistics are tracked: total accesses, page faults, protection faults
    pitfalls:
      - "Off-by-one in bit shifting for VPN extraction; page size 4096 = 2^12 means shift right by 12, not 11"
      - "Forgetting to preserve the offset bits during translation; the offset passes through unchanged"
      - "Setting dirty bit on read access (only writes should set dirty)"
      - "Not distinguishing between page fault (valid=0) and protection fault (valid=1 but wrong permissions)"
    concepts:
      - Address translation (VPN → PFN + offset)
      - Page table entries and flag bits
      - Demand paging
      - Memory protection
    skills:
      - Bitwise operations for address decomposition
      - Data structure design for page tables
      - Trace file parsing
      - Statistics collection
    deliverables:
      - Trace file parser reading memory access sequences
      - Page table array indexed by VPN with PTE structures
      - Address translation function returning physical address or fault
      - Page fault handler allocating a free frame and updating PTE
      - Protection fault detection and logging
      - Statistics reporter (accesses, faults, protection violations)
    estimated_hours: "4-6"

  - id: virtual-memory-sim-m2
    name: "TLB (Translation Lookaside Buffer)"
    description: >
      Add a TLB cache between address requests and the page table to
      accelerate translations. Support ASID tagging for context switches.
    acceptance_criteria:
      - TLB is checked first on every address translation; TLB hit returns the PFN directly without consulting the page table
      - TLB miss triggers a full page table lookup; the resulting translation is inserted into the TLB
      - TLB has a configurable number of entries (e.g., 16, 32, 64); when full, eviction uses LRU or random replacement
      - TLB entries include an ASID (Address Space ID) tag; entries with a different ASID are treated as misses without requiring a full flush
      - Context switch simulation changes the active ASID; with ASID support, TLB entries from other processes are preserved but inactive
      - TLB flush operation invalidates all entries (used when ASIDs are exhausted or explicitly requested)
      - Statistics tracked: TLB hits, TLB misses, TLB hit rate percentage, TLB flushes
      - TLB dirty bit is propagated back to the page table on eviction (write-back of metadata)
    pitfalls:
      - TLB coherency: when a PTE is modified (e.g., page evicted), the corresponding TLB entry must be invalidated or the TLB serves stale data
      - "Not updating the page table's referenced/dirty bits on TLB hits; the TLB must either write-through these bits or flush them on eviction"
      - "ASID width limits (e.g., 8-bit ASID = 256 processes before recycling requires flush)"
      - "Random replacement is simpler but makes testing non-deterministic; use a seeded PRNG for reproducibility"
    concepts:
      - Caching and locality of reference
      - Fully associative vs set-associative TLB
      - ASID-tagged TLB entries
      - TLB shootdown concept
    skills:
      - Cache implementation
      - Performance optimization through caching
      - Context switch simulation
      - Associative lookup structures
    deliverables:
      - TLB structure with configurable entry count and ASID support
      - TLB lookup function with ASID-tagged matching
      - TLB miss handler triggering page table walk and TLB insertion
      - TLB eviction (LRU or random with seed)
      - TLB flush and ASID-based invalidation
      - TLB statistics (hits, misses, hit rate)
    estimated_hours: "5-7"

  - id: virtual-memory-sim-m3
    name: "Multi-Level Page Tables"
    description: >
      Implement two-level (or three-level) hierarchical page tables to
      reduce memory overhead for sparse address spaces. Simulate a CR3/PDBR register.
    acceptance_criteria:
      - A simulated CR3 (Page Directory Base Register) holds the physical address of the root page directory for the active process
      - Virtual address is split into page directory index, page table index, and offset (e.g., 10+10+12 for two-level with 4KB pages on 32-bit)
      - Page directory entries point to second-level page tables; NULL entries indicate unmapped regions (no second-level table allocated)
      - Second-level page tables are allocated on demand only when the first access to that region occurs
      - Page table walk traverses all levels from CR3 → page directory → page table → PTE → physical frame + offset
      - Memory overhead is measured and compared to single-level: a sparse address space with few mapped pages should use significantly less memory
      - Context switch changes CR3 to point to a different process's page directory; TLB is flushed or ASID is switched
      - Three-level page table support is implemented as a stretch goal (e.g., 2+9+9+12 for a 32-bit space with 4KB pages)
    pitfalls:
      - Index extraction order: most significant bits select the directory, middle bits select the table, least significant bits are the offset
      - "Confusing page directory entries (which point to page tables) with page table entries (which point to frames)"
      - "Not accounting for the memory consumed by the page table structures themselves in overhead calculations"
      - "Forgetting to handle the case where the page directory entry is NULL (entire region unmapped) vs the PTE being invalid (single page unmapped)"
    concepts:
      - Hierarchical page tables
      - Sparse address space optimization
      - CR3/PDBR register
      - On-demand table allocation
    skills:
      - Multi-level data structure traversal
      - Bit field extraction
      - Memory-efficient sparse structures
      - Pointer indirection
    deliverables:
      - CR3 register simulation holding root page directory base address
      - Two-level page table with directory and table index extraction
      - On-demand second-level page table allocation
      - Page table walk function traversing all levels
      - Context switch simulation changing CR3 and flushing/switching TLB
      - Memory overhead comparison report (single-level vs multi-level for sparse access patterns)
    estimated_hours: "5-8"

  - id: virtual-memory-sim-m4
    name: "Page Replacement and Swap Simulation"
    description: >
      Implement page replacement algorithms with simulated swap space
      for evicting and reloading pages when physical memory is exhausted.
    acceptance_criteria:
      - Physical memory is modeled as a fixed number of frames (configurable, e.g., 16, 64, 256); page faults when all frames are occupied trigger replacement
      - Simulated swap space (a file or array) stores evicted page data; dirty pages are written to swap on eviction and read back on reload
      - FIFO replacement evicts the page loaded earliest; verify against known trace outputs
      - LRU replacement evicts the page with the oldest last-access time; implemented with timestamps or a doubly-linked list
      - Clock (Second-Chance) algorithm uses a circular buffer with reference bits; clears reference bit on first pass, evicts on second pass
      - Optimal (Bélády's) algorithm evicts the page that will not be used for the longest time in the future; requires look-ahead over the remaining trace (used as a lower bound, not practical)
      - Dirty pages incur an extra write-back cost to swap on eviction; clean pages are simply discarded (their data is unchanged from swap/initial load)
      - Comparative statistics are printed: page faults per algorithm, dirty write-backs, and optional Bélády's anomaly demonstration with FIFO
      - Working set size is tracked as the number of distinct pages accessed in a sliding window of the last N accesses
    pitfalls:
      - Bélády's anomaly: FIFO can produce MORE page faults with MORE frames; demonstrate with a specific trace to build intuition
      - "Not writing dirty pages to swap on eviction means data is silently lost; reload from swap returns stale/zero data"
      - "LRU with timestamps can be expensive (O(n) scan); consider approximate LRU with the Clock algorithm as the practical alternative"
      - "Optimal algorithm requires the full future access trace; it's a benchmark, not a real algorithm. Students sometimes try to implement it as online."
      - Thrashing: if working set exceeds physical memory, every access is a fault. Detect and report this condition.
    concepts:
      - Page replacement policies
      - Swap space and disk I/O simulation
      - Dirty page write-back
      - Working set model
      - Bélády's anomaly
    skills:
      - Algorithm implementation and comparison
      - Queue and list management (FIFO, LRU)
      - Statistics collection and analysis
      - Dirty page tracking
    deliverables:
      - Physical memory frame pool with configurable size
      - Simulated swap space (array or file) for evicted page data
      - FIFO page replacement implementation
      - LRU page replacement implementation
      - Clock (Second-Chance) page replacement implementation
      - Optimal (Bélády's) page replacement implementation (look-ahead based)
      - Dirty page write-back on eviction and reload from swap on fault
      - Comparative statistics report across all algorithms for the same trace
    estimated_hours: "8-12"
```
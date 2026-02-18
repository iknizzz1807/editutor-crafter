# AUDIT & FIX: simple-gc

## CRITIQUE
- **Technical Inaccuracy (Confirmed - Metadata/Type Descriptors):** Milestone 1 defines the object model but fails to include 'Metadata/Type Descriptors' that allow the Mark Phase to identify which fields in an object are pointers vs. primitives. Without this information, the GC cannot safely traverse object graphs—it won't know which fields to follow.
- **Logical Gap (Confirmed - Stop-the-World):** Missing a 'Stop-the-world' or 'Safe-point' mechanism. A GC cannot safely walk the stack or mutate pointers while the mutator (the main program) is actively changing references. The implementation must coordinate with the mutator, either by pausing entirely (stop-the-world) or at defined safe points.
- **Hour Ranges:** Using ranges like '3-4' is imprecise. Converting to single estimates.
- **Estimated Hours:** 15-25 range is reasonable; with added metadata and safe-point requirements, estimate ~20 hours.

## FIXED YAML
```yaml
id: simple-gc
name: Simple GC
description: >-
  Mark-sweep garbage collector with type descriptors for pointer identification,
  root scanning, and stop-the-world coordination for safe collection cycles.
difficulty: advanced
estimated_hours: 20
essence: >-
  Automatic memory reclamation through graph traversal algorithms that
  distinguish live objects from garbage, requiring intricate pointer
  manipulation, type metadata for safe traversal, and careful coordination
  between allocation, marking, and deallocation phases to prevent memory
  corruption.
why_important: >-
  Building a garbage collector reveals how high-level languages manage memory
  automatically and teaches systems programming fundamentals that apply to
  performance optimization, debugging memory leaks, and understanding runtime
  behavior in production systems.
learning_outcomes:
  - Design object layouts with GC metadata and type descriptors
  - Implement root discovery from stack and global variables
  - Build mark phase with cycle-safe traversal using mark bits
  - Implement sweep phase to reclaim unreachable objects
  - Coordinate GC cycles with mutator via stop-the-world mechanism
skills:
  - Memory Management
  - Graph Traversal
  - Type Descriptors
  - Root Scanning
  - Stop-the-World Coordination
tags:
  - garbage-collection
  - memory-management
  - systems-programming
  - advanced
  - mark-sweep
architecture_doc: architecture-docs/simple-gc/index.md
languages:
  recommended:
    - C
    - Rust
    - C++
  also_possible:
    - Zig
resources:
  - name: Garbage Collection Handbook
    url: https://gchandbook.org/
    type: book
  - name: Simple GC Tutorial
    url: http://www.jpaulmorrison.com/fuzzylogic.shtml
    type: tutorial
  - name: Boehm GC
    url: https://www.hboehm.info/gc/
    type: code
prerequisites:
  - type: skill
    name: C programming and pointers
  - type: skill
    name: Memory allocation (malloc/free)
  - type: skill
    name: Understanding of call stack
milestones:
  - id: simple-gc-m1
    name: Object Model with Type Descriptors
    description: >-
      Define object representation with GC metadata and type information for
      pointer identification.
    acceptance_criteria:
      - Object header includes type tag for runtime dispatch
      - Store mark bit in each object header for garbage collection
      - Type descriptor identifies which fields in an object are pointers vs primitives
      - Track total allocated object size for GC threshold triggering
      - Support at least three different object types (int, pair, string) with distinct layouts
    pitfalls:
      - Header alignment: object header must be at a known offset for the GC to find
      - Type safety: GC must not misinterpret primitive data as pointers—use type descriptors
      - Object size calculation: variable-sized objects (arrays, strings) need special handling
    concepts:
      - Object headers contain GC metadata (mark bit, type pointer, size)
      - Type descriptors enable precise pointer identification
      - Allocation tracking enables GC triggering at thresholds
    skills:
      - Memory layout design
      - Pointer arithmetic and alignment
      - Low-level data structure implementation
      - Type system design
    deliverables:
      - Object header structure containing type tag, mark bit, and size
      - Type descriptor system identifying pointer vs primitive fields for each type
      - Reference tracking linking objects that point to other objects
      - Object allocation function creating new objects on the heap with proper header initialization
    estimated_hours: 5

  - id: simple-gc-m2
    name: Root Discovery
    description: >-
      Identify and enumerate all GC roots including stack, globals, and
      registers.
    acceptance_criteria:
      - Identify and enumerate all GC root references on the stack
      - Include global variable references in the root set
      - Support register-based roots when applicable to target architecture
      - Distinguish precise roots (known pointers) from conservative scanning
      - Stop-the-world mechanism pauses mutator before root scanning begins
    pitfalls:
      - Missing roots: any missed root causes live objects to be incorrectly collected
      - Stack scanning direction: stacks grow in different directions on different platforms
      - Interior pointers: pointers into the middle of objects require finding the object start
      - Mutator coordination: GC MUST NOT run while mutator is modifying references
    concepts:
      - Root set is the starting point for reachability analysis
      - Precise scanning knows exactly where pointers are; conservative guesses
      - Stop-the-world ensures consistent state during collection
    skills:
      - Stack introspection and traversal
      - Pointer identification techniques
      - Memory scanning strategies
      - Platform-specific calling conventions
      - Mutator coordination
    deliverables:
      - Stack scanning identifying root references from call stack frames
      - Global variable roots tracking references in static or global scope
      - Register contents inspection for architectures storing roots in registers
      - Root set construction building complete list of live reference origins
      - Stop-the-world mechanism to pause mutator during collection
    estimated_hours: 5

  - id: simple-gc-m3
    name: Mark Phase
    description: >-
      Traverse and mark all reachable objects with cycle safety.
    acceptance_criteria:
      - Recursively mark all objects reachable from root set
      - Handle reference cycles without infinite loop using mark bits
      - Set and check mark bits to avoid revisiting already-marked objects
      - Use explicit worklist instead of recursion to avoid stack overflow on deep structures
      - Type descriptors guide traversal of pointer fields within objects
    pitfalls:
      - Stack overflow on deep structures: recursive marking can overflow—use explicit worklist
      - Forgetting object types: must use type descriptors to find pointer fields
      - Double marking: harmless but wastes time—check mark bit before recursing
    concepts:
      - Graph traversal from roots marks all reachable objects
      - Mark bits prevent infinite loops in cyclic structures
      - Worklist (BFS) avoids deep recursion stack overflow
    skills:
      - Recursive graph algorithms
      - Iterative traversal with explicit stacks
      - Bit manipulation for marking
      - Cycle detection in object graphs
    deliverables:
      - Object graph traversal starting from root set references
      - Mark bit setting flagging each reachable object as live
      - Explicit worklist processing avoiding deep recursion stack overflow
      - Type-directed traversal using type descriptors to find pointer fields
      - Reachability analysis determining which objects are still in use
    estimated_hours: 5

  - id: simple-gc-m4
    name: Sweep Phase
    description: >-
      Reclaim unmarked objects, reset marks, and resume mutator.
    acceptance_criteria:
      - Walk the complete allocation list checking each object's mark bit
      - Free all unmarked objects and reclaim their memory
      - Reset mark bits on surviving objects for next collection cycle
      - Update allocation linked list to skip freed object entries
      - Resume mutator after sweep completes
    pitfalls:
      - List corruption: removing freed objects from allocation list must not break traversal
      - Forgetting to clear marks: uncleared marks cause incorrect behavior in next cycle
      - Memory leaks in objects: swept objects must have their destructors called if applicable
    concepts:
      - Sweep phase frees all unmarked (unreachable) objects
      - Mark bit reset prepares for next collection cycle
      - Free list management returns memory to the allocator
    skills:
      - Free list management
      - Memory coalescing techniques
      - Linked list manipulation without allocation
      - GC heuristics and tuning
    deliverables:
      - Heap traversal walking entire allocation list checking mark bits
      - Unmarked object collection freeing memory of unreachable objects
      - Free list update relinking allocation list after removing freed objects
      - Mark bit reset clearing marks on surviving objects
      - Memory reclamation returning freed bytes to allocator for reuse
      - Mutator resume to continue program execution
    estimated_hours: 5
```

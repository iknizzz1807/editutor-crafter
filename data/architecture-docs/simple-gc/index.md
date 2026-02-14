# Simple Mark-Sweep Garbage Collector

A foundational garbage collector implementing the mark-sweep algorithm. This system automatically reclaims memory by identifying reachable objects from root references, marking them as live, and sweeping unreachable objects to free memory. The architecture demonstrates core GC concepts including object graphs, root scanning, tri-color marking, and memory compaction strategies.



# System Architecture Overview

<div id="ms-system-overview"></div>

## The City Sanitation Analogy

Imagine a bustling city where buildings (objects) are constantly being constructed and abandoned. Your garbage collector is the **sanitation department** that must:

1. **Identify which buildings are still in use** by tracing roads (pointers) from government offices (root references)
2. **Mark active buildings** with a flag so they're not demolished
3. **Sweep through the city** and demolish abandoned buildings
4. **Optionally compact** the remaining buildings to eliminate 

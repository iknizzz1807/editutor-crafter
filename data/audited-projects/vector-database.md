# AUDIT & FIX: vector-database

## CRITIQUE
- **Logical Gap (Confirmed - Quantization):** Product Quantization (PQ) and Scalar Quantization (SQ) are standard for memory-efficient large-scale vector search. Without quantization, a vector database storing 100M 768-dimensional float32 vectors requires ~286GB of RAM. This is not an edge case—it's a core production requirement. A quantization milestone is mandatory.
- **Technical Inaccuracy (Confirmed - Cosine Similarity):** The AC states 'cosine similarity returning values in the range of -1 to 1'—while mathematically correct for the cosine of the angle, distance-based graph structures like HNSW use cosine *distance* (1 - cosine_similarity) which ranges [0, 2]. The current AC would produce inverted ordering in the HNSW graph. The storage and search layers must agree on whether they use similarity (higher is better) or distance (lower is better).
- **M1 Vector Storage:** 'Handle vector normalization converting raw vectors to unit length on insert' should be optional behavior, not default. Euclidean distance search on pre-normalized vectors gives different results than on raw vectors. Normalization should only be applied for cosine similarity.
- **M3 Brute Force:** 'Use as baseline for ANN accuracy' is not a measurable AC. It should specify recall@k measurement comparing HNSW results against brute-force ground truth.
- **M4 HNSW:** 'Node insertion with level assignment' deliverable is incomplete—must specify the probabilistic level assignment formula (level = floor(-ln(uniform_random) * mL) where mL = 1/ln(M)).
- **M5 Query API:** 'Implement hybrid search (vector + keyword)' is a massive feature crammed into one AC with no prerequisite for a keyword index. This should be scoped down or removed.
- **Missing: Metadata Filtering during ANN search** is architecturally complex—pre-filtering vs post-filtering significantly affects recall. No milestone addresses this tradeoff.
- **Estimated Hours:** 5 × 16 = 80 hours. Adding quantization pushes to ~96 hours.

## FIXED YAML
```yaml
id: vector-database
name: Vector Database
description: >-
  Similarity search engine with vector storage, distance metrics, brute-force
  baseline, HNSW approximate nearest neighbor index, vector quantization, and
  a query API.
difficulty: advanced
estimated_hours: 96
essence: >-
  Multi-dimensional vector storage with graph-based approximate nearest neighbor
  search using hierarchical navigable small world graphs (HNSW), distance metric
  computations (cosine distance, L2, dot product), vector quantization for memory
  efficiency, and metadata filtering—trading exact accuracy for sub-linear query
  time complexity.
why_important: >-
  Vector similarity search is fundamental to modern AI applications including
  semantic search, recommendation systems, and RAG pipelines. Building a vector
  database from scratch teaches core data structures (graphs, quantization),
  systems programming (memory management, SIMD optimization), and performance
  engineering critical for infrastructure roles.
learning_outcomes:
  - Design memory-aligned contiguous vector storage with efficient batch I/O
  - Implement distance metrics (cosine distance, L2, dot product) with SIMD optimization
  - Build exact brute-force KNN as a correctness baseline for measuring ANN recall
  - Implement HNSW graph construction and search with configurable M, efConstruction, and efSearch parameters
  - Implement scalar and product quantization for memory-efficient large-scale search
  - Build a query API with metadata filtering, batch operations, and concurrent access
  - Measure and optimize recall@k, query latency, and memory usage under load
skills:
  - Vector similarity
  - HNSW algorithm
  - Distance metrics
  - Approximate nearest neighbor
  - Vector quantization
  - Memory-mapped storage
  - Index persistence
  - SIMD optimization
tags:
  - advanced
  - ai-ml
  - databases
  - embeddings
  - indexing
  - search
  - similarity-search
  - build-from-scratch
architecture_doc: architecture-docs/vector-database/index.md
languages:
  recommended:
    - Rust
    - Go
    - C++
  also_possible:
    - Python
resources:
  - name: HNSW Algorithm Paper
    url: https://arxiv.org/abs/1603.09320
    type: paper
  - name: Product Quantization for Nearest Neighbor Search (Jégou et al.)
    url: https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf
    type: paper
  - name: Faiss Documentation
    url: https://faiss.ai/index.html
    type: documentation
  - name: hnswlib Implementation
    url: https://github.com/nmslib/hnswlib
    type: tool
  - name: Understanding HNSW
    url: https://www.pinecone.io/learn/series/faiss/hnsw/
    type: tutorial
prerequisites:
  - type: skill
    name: Data structures (graphs, heaps)
  - type: skill
    name: Linear algebra basics (dot product, norms)
  - type: skill
    name: Systems programming (memory layout, file I/O)
milestones:
  - id: vector-database-m1
    name: Vector Storage Engine
    description: >-
      Build efficient contiguous storage for fixed-dimension vectors with
      metadata, supporting batch operations, persistence, and compaction.
    acceptance_criteria:
      - Vectors are stored in contiguous memory with fixed dimensionality configured at collection creation time
      - Storage is aligned to 32-byte (AVX2) or 64-byte (AVX-512) boundaries for SIMD compatibility
      - Batch insert adds N vectors in a single operation, measured to be at least 5x faster than N individual inserts for N=1000
      - Vector retrieval by ID returns the vector and associated metadata in O(1) via an ID-to-offset index
      - Deletion marks vectors as tombstoned; storage compaction reclaims space by rewriting live vectors contiguously
      - Memory-mapped file storage enables efficient access to datasets larger than available RAM
      - Persistence survives process restart—stored vectors are recoverable from disk
    pitfalls:
      - Not aligning vectors to cache line or SIMD boundaries kills distance computation performance by 2-4x
      - Memory mapping without proper error handling on disk full causes silent corruption
      - Dynamic allocation per vector (e.g., individual heap allocations) fragments memory and destroys cache locality
      - Partial writes during serialization corrupt the storage file—use atomic rename on fsync'd temp file
      - Tombstone accumulation without compaction wastes space and slows sequential scans
    concepts:
      - Memory alignment for SIMD operations (AVX2 needs 32-byte, AVX-512 needs 64-byte)
      - Contiguous storage layout for cache-efficient sequential access
      - Memory-mapped file lifecycle (mmap, msync, munmap)
      - Tombstone-based deletion with background compaction
    skills:
      - Memory layout and alignment
      - Memory-mapped file I/O
      - Batch operation design
      - Storage compaction
    deliverables:
      - Fixed-dimension vector storage engine with configurable dimensionality
      - SIMD-aligned contiguous memory layout
      - Vector ID-to-offset index for O(1) retrieval
      - Batch insert operation for bulk loading
      - Tombstone deletion with compaction process
      - Memory-mapped file persistence with crash-safe writes
    estimated_hours: 16

  - id: vector-database-m2
    name: Distance Metrics
    description: >-
      Implement distance/similarity functions with SIMD optimization and
      establish clear conventions for similarity vs distance ordering.
    acceptance_criteria:
      - Cosine distance computed as 1.0 - (dot(a,b) / (norm(a) * norm(b))), returning values in [0, 2] where 0 means identical
      - Euclidean (L2) distance computed as sqrt(sum((a_i - b_i)^2)), returning non-negative values
      - Dot product (inner product) similarity computed correctly; note that higher values mean more similar (ordering is inverted vs distance)
      - All metrics produce results within 1e-6 of a reference implementation for a test suite of at least 100 vector pairs
      - SIMD-optimized implementations are at least 3x faster than naive scalar loops for 768-dimensional vectors (measured via benchmark)
      - Pre-normalized vector detection skips redundant normalization for cosine distance, verified to produce identical results
      - Batch distance computation (1 query vs N vectors) processes at least 1M 128-dim comparisons per second
    pitfalls:
      - Confusing similarity (higher=better) with distance (lower=better) causes inverted rankings—establish a convention and enforce it everywhere
      - Not normalizing vectors before cosine computation produces incorrect results; alternatively use cosine distance formula that handles unnormalized inputs
      - Float accumulation in high dimensions loses precision; use compensated summation (Kahan) for >512 dimensions
      - Comparing distances from different metrics directly (e.g., L2 vs cosine) is mathematically meaningless
      - SIMD intrinsics are platform-specific; provide scalar fallback for portability
    concepts:
      - Cosine distance vs cosine similarity (1 - similarity = distance)
      - SIMD vectorization (AVX2, NEON) for parallel float arithmetic
      - Numerical stability in floating-point accumulation
      - Distance metric properties: symmetry, non-negativity, triangle inequality (L2 and cosine distance satisfy these; dot product does not)
    skills:
      - SIMD intrinsics programming
      - Numerical stability
      - Benchmarking methodology
      - Distance metric theory
    deliverables:
      - Cosine distance computation (1 - cosine_similarity)
      - Euclidean L2 distance computation
      - Dot product similarity computation
      - SIMD-optimized implementations with scalar fallback
      - Batch distance computation (1-vs-N)
      - Pre-normalized vector fast path
      - Benchmark suite comparing SIMD vs scalar performance
    estimated_hours: 14

  - id: vector-database-m3
    name: Brute Force KNN (Baseline)
    description: >-
      Implement exact K-nearest neighbor search as a correctness baseline
      for measuring HNSW recall in subsequent milestones.
    acceptance_criteria:
      - Exact KNN returns the true top-k nearest vectors by computing distance to every stored vector
      - Top-k selection uses a max-heap of size k, achieving O(N log k) complexity instead of O(N log N) full sort
      - Metadata filtering (pre-filter) restricts candidates before distance computation based on field predicates
      - Batch query support executes M queries returning results equivalent to M individual queries
      - Performance baseline is established: query latency measured at 10K, 100K, and 1M vectors for k=10
      - Ground truth export generates a file mapping query vectors to their exact top-k results for recall measurement
    pitfalls:
      - Using full sort instead of heap-based top-k selection is O(N log N) instead of O(N log k)—significant for large k
      - Scanning vectors in non-sequential memory order destroys cache performance; iterate contiguously
      - Pre-filtering that evaluates metadata predicates *after* distance computation wastes CPU; filter first when selectivity is high
      - Copying vectors into result arrays instead of returning indices wastes memory
    concepts:
      - Heap-based top-k selection maintains a bounded result set during linear scan
      - Cache-efficient sequential scanning maximizes memory bandwidth
      - Pre-filtering vs post-filtering tradeoff for metadata predicates
      - Ground truth generation for recall@k measurement
    skills:
      - Heap data structure
      - Linear scan optimization
      - Metadata filtering
      - Benchmarking and measurement
    deliverables:
      - Exact KNN search with max-heap top-k selection
      - Metadata pre-filtering restricting candidates by field predicates
      - Batch query execution for multiple queries
      - Performance benchmark at 10K, 100K, 1M scale
      - Ground truth export for recall measurement
      - Threshold-based search returning all vectors within a distance cutoff
    estimated_hours: 14

  - id: vector-database-m4
    name: HNSW Index
    description: >-
      Implement the Hierarchical Navigable Small World (HNSW) graph for
      approximate nearest neighbor search with sub-linear query complexity.
    acceptance_criteria:
      - HNSW graph construction assigns layers using the probabilistic formula: level = floor(-ln(uniform_random()) * mL) where mL = 1/ln(M)
      - Configurable parameters: M (max connections per node, default 16), efConstruction (build-time exploration width, default 200), efSearch (query-time exploration width, default 50)
      - Graph construction maintains bidirectional edges and enforces maximum M connections per node via neighbor selection heuristic
      - Greedy search traverses from entry point at top layer, descending to base layer, then expanding with efSearch candidates
      - Incremental insert adds new vectors without full index rebuild, maintaining graph connectivity invariants
      - Recall@10 ≥ 0.95 on a 100K vector dataset when efSearch=100, measured against brute-force ground truth from M3
      - Query latency is at least 10x faster than brute-force on 100K vectors for k=10 (measured via benchmark)
      - Index serialization persists the full graph structure to disk; deserialization restores identical search behavior
    pitfalls:
      - Not maintaining bidirectional edges during insertion causes unreachable nodes and recall collapse
      - Not limiting max connections per node (M_max = M for base layer, 2*M for upper layers per original paper) causes memory bloat
      - Greedy search without backtracking (using a priority queue of candidates) gets stuck in local optima
      - Neighbor selection heuristic must prioritize diversity (not just proximity) to maintain navigable small world properties
      - efSearch too low gives poor recall; too high degrades to near-brute-force latency. Tune empirically.
    concepts:
      - Probabilistic skip list-like layer structure for hierarchical navigation
      - Greedy best-first search with candidate priority queue and backtracking
      - Neighbor selection heuristic balancing proximity and diversity
      - Layer probability distribution controls graph height and connectivity
      - Recall@k measures fraction of true nearest neighbors found by ANN
    skills:
      - Graph construction algorithms
      - Priority queue-based greedy search
      - Probabilistic data structures
      - Recall measurement methodology
    deliverables:
      - Multi-layer graph structure with probabilistic level assignment
      - Node insertion with bidirectional edge maintenance and neighbor pruning
      - Greedy search with layer descent and efSearch-controlled base layer exploration
      - Configurable M, efConstruction, and efSearch parameters
      - Recall@k measurement against brute-force ground truth
      - Index serialization and deserialization
      - Benchmark comparing HNSW vs brute-force latency and recall
    estimated_hours: 20

  - id: vector-database-m5
    name: Vector Quantization
    description: >-
      Implement scalar quantization (SQ) and product quantization (PQ) for
      memory-efficient approximate distance computation, enabling large-scale
      search that would not fit in RAM with full-precision vectors.
    acceptance_criteria:
      - Scalar quantization (SQ8) maps float32 values to uint8 using per-dimension min/max scaling, reducing memory by 4x
      - Product quantization (PQ) splits vectors into M subspaces, quantizes each to one of 256 centroids (trained via k-means), and stores M uint8 codes per vector
      - Asymmetric Distance Computation (ADC) computes approximate distances using precomputed lookup tables, avoiding full vector decompression
      - SQ8 recall@10 ≥ 0.90 on the same 100K dataset used in M3/M4 (measured against brute-force ground truth)
      - PQ with M=8 subspaces achieves at least 16x memory reduction compared to float32 storage (measured)
      - Quantization can be combined with HNSW—HNSW graph uses quantized distance for graph traversal, then re-ranks top candidates with exact distance
      - Training (codebook generation for PQ) completes in under 60 seconds for 100K training vectors
    pitfalls:
      - Per-dimension min/max for scalar quantization must be computed from training data, not hardcoded—data distribution varies
      - PQ codebook training with too few iterations of k-means produces poor centroids and low recall
      - Asymmetric distance computation requires precomputing M×256 lookup table per query—this amortizes only with large candidate sets
      - Combining quantized distance with HNSW requires careful threshold tuning—quantization errors can cause HNSW to skip true nearest neighbors
      - Quantization error is higher for vectors with high variance across dimensions; monitor per-dimension error distribution
    concepts:
      - Scalar quantization maps continuous values to discrete integer levels
      - Product quantization decomposes high-dimensional space into lower-dimensional subspaces
      - Codebook training uses k-means clustering to learn representative centroids per subspace
      - Asymmetric Distance Computation (ADC) uses lookup tables for fast approximate distance
      - Two-phase search: quantized recall candidates, then exact re-ranking of top candidates
    skills:
      - K-means clustering
      - Quantization theory
      - Lookup table optimization
      - Memory-accuracy tradeoff analysis
    deliverables:
      - Scalar quantization (SQ8) with per-dimension min/max calibration
      - Product quantization with configurable subspace count and codebook training
      - Asymmetric distance computation using precomputed lookup tables
      - HNSW + PQ integration for memory-efficient approximate search
      - Recall and memory reduction measurements comparing FP32, SQ8, and PQ
      - Benchmark suite measuring recall, latency, and memory usage across configurations
    estimated_hours: 16

  - id: vector-database-m6
    name: Query API & Server
    description: >-
      Expose vector operations via a REST or gRPC API with concurrent access,
      metadata filtering, and batch operations.
    acceptance_criteria:
      - REST or gRPC API supports insert, upsert, search, and delete operations with OpenAPI or proto schema
      - Search API accepts query vector, k, distance metric, optional metadata filter predicates, and optional efSearch override
      - Metadata filtering during ANN search supports pre-filtering (filter before search) and post-filtering (filter after search) modes with configurable strategy
      - Batch operations process multiple inserts or queries in a single request, measured to reduce per-operation latency by at least 50% vs individual calls
      - Concurrent reads and writes are handled safely via read-write locks or lock-free structures; verified by concurrent stress test with no data corruption
      - Collection management API supports create (with dimension, metric, and index config), list, and delete operations
      - Query timeout terminates long-running searches after a configurable deadline
    pitfalls:
      - Blocking all reads during index updates causes availability gaps—use copy-on-write or RW-lock with read preference
      - Not setting query timeouts allows runaway searches to consume resources indefinitely
      - Pre-filtering with very selective predicates can reduce the candidate set so much that recall drops—document this tradeoff
      - Deserializing entire request body before validation wastes memory on malformed requests—validate incrementally
      - Missing API versioning makes backward-incompatible changes dangerous
    concepts:
      - Read-write locking for concurrent vector access
      - Pre-filtering vs post-filtering tradeoff for metadata-constrained ANN
      - gRPC streaming for efficient batch operations
      - Backpressure handling for high query loads
      - API versioning for forward compatibility
    skills:
      - API design (REST/gRPC)
      - Concurrent access control
      - Request validation
      - Batch operation design
    deliverables:
      - Insert/upsert API accepting vectors with metadata
      - Search API with k, metric, filter, and efSearch parameters
      - Delete API removing vectors by ID
      - Collection management (create, list, delete)
      - Pre-filter and post-filter modes for metadata-constrained search
      - Batch insert and batch query endpoints
      - Concurrent stress test verifying data integrity under load
    estimated_hours: 16
```
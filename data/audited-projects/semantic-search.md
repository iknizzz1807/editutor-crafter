# AUDIT & FIX: semantic-search

## CRITIQUE
- **Critical Logical Gap**: Milestone 1 jumps straight to 'Embedding Index' without any data ingestion, preprocessing, or chunking milestone. Transformer models have token limits (typically 256-512 tokens for SBERT). Without a chunking strategy, embedding long documents produces degraded representations or silent truncation. This is the #1 failure mode in production semantic search.
- **Misleading AC in M1**: 'Index millions of document vectors' as the *first* milestone is absurdly ambitious scope-wise and conflates embedding generation with index construction. These are distinct concerns.
- **Technical Inaccuracy in M2**: 'Negative query terms subtracting unwanted concepts from results' is described as simple vector subtraction, but in practice this is algebraically unreliable in high-dimensional embedding spaces. Vector subtraction can flip the query into an unrelated region. The AC should specify metadata/filter-based exclusion as the primary mechanism, with vector arithmetic as experimental.
- **Missing Embedding Generation Step**: M1's deliverables mention 'Document embedding pipeline' but the AC focuses entirely on indexing (HNSW, IVF). The actual transformer model loading, batching, and embedding generation is never explicitly validated.
- **Tag Pollution**: Tags include 'game-dev' and 'go' which are irrelevant to a semantic search project.
- **M3 Overreach**: 'Learn from click-through rate data' and 'personalization signals' are production ML features that require user interaction logging infrastructure not built in any prior milestone. This is scope creep for a 25-40 hour project.
- **M4 Missing Auth/Rate Limiting**: Production API has no mention of authentication, rate limiting, or pagination for search results.
- **Estimated Hours Inconsistency**: M1 (6-10h) claims to build an embedding pipeline AND index millions of vectors, while M4 (8-12h) for a REST API is allocated more time. The time allocation is inverted.
- **No Evaluation Milestone**: There is no milestone for measuring search quality (MRR, NDCG, recall@k). Without evaluation, you cannot validate that the system works.

## FIXED YAML
```yaml
id: semantic-search
name: Semantic Search Engine
description: >-
  Build a semantic search engine that understands meaning, not just keywords.
  Covers the full stack from document preprocessing through embedding generation,
  vector indexing, hybrid ranking, and a production search API.
difficulty: intermediate
estimated_hours: "35-50"
essence: >-
  Transform text into dense vector embeddings using transformer models (Sentence-BERT),
  implement approximate nearest neighbor search with spatial indexing structures
  (FAISS HNSW/IVF), and fuse keyword-based ranking (BM25) with cosine similarity
  scoring to retrieve semantically relevant results beyond exact string matches.
why_important: >-
  Semantic search powers modern information retrieval from Google to ChatGPT's RAG
  pipelines. Building this teaches the complete ML-powered search stack—from text
  preprocessing and embedding models to vector indexes and hybrid ranking—skills
  directly applicable to LLM applications, recommendation systems, and AI products.
learning_outcomes:
  - Preprocess and chunk documents for optimal transformer embedding quality
  - Implement vector embeddings using Sentence-BERT to capture semantic meaning
  - Build approximate nearest neighbor indexes with FAISS for sub-linear similarity search
  - Design hybrid ranking systems combining BM25 keyword scoring with cosine similarity
  - Optimize index structures through dimensionality reduction and quantization
  - Implement metadata filtering alongside vector similarity for precision retrieval
  - Evaluate search quality using MRR, NDCG, and recall@k metrics
  - Build production search APIs with sub-100ms latency using batching and caching
skills:
  - Text Preprocessing & Chunking
  - Vector Embeddings
  - Approximate Nearest Neighbor Search
  - Transformer Models (SBERT)
  - FAISS Indexing
  - Hybrid Search Ranking (BM25 + Cosine)
  - Information Retrieval Evaluation Metrics
  - REST API Design
tags:
  - embeddings
  - intermediate
  - python
  - search
  - sentence-transformers
  - similarity
  - vector-search
  - information-retrieval
architecture_doc: architecture-docs/semantic-search/index.md
languages:
  recommended:
    - Python
  also_possible:
    - Go
    - TypeScript
    - Rust
resources:
  - name: Sentence Transformers
    url: https://www.sbert.net/
    type: documentation
  - name: FAISS Library
    url: https://github.com/facebookresearch/faiss
    type: documentation
  - name: BM25 Algorithm Explained
    url: https://en.wikipedia.org/wiki/Okapi_BM25
    type: reference
prerequisites:
  - type: skill
    name: Python
  - type: skill
    name: Embeddings basics
  - type: skill
    name: Database fundamentals
milestones:
  - id: semantic-search-m1
    name: Data Preprocessing & Chunking
    description: >-
      Build a document ingestion pipeline that cleans, chunks, and prepares text
      for transformer embedding. This is the foundation—garbage in, garbage out.
    acceptance_criteria:
      - Ingest documents from at least two formats (plain text, JSON) with metadata extraction
      - Chunk documents into segments of configurable token length (default 256 tokens) with configurable overlap (default 50 tokens)
      - Chunking respects sentence boundaries so no chunk splits mid-sentence
      - Each chunk retains a reference to its source document ID and positional offset
      - Text cleaning removes HTML tags, normalizes unicode, and collapses whitespace
      - Pipeline processes at least 10,000 documents and outputs a chunk dataset with statistics (total chunks, avg length, length distribution)
    pitfalls:
      - Splitting mid-sentence destroys semantic coherence and degrades embedding quality
      - Using character-based chunking instead of token-based chunking causes silent truncation by the transformer tokenizer
      - Forgetting overlap between chunks loses context at chunk boundaries
      - Not tracking chunk-to-document mapping makes result attribution impossible
      - Ignoring encoding issues (UTF-8 BOM, mixed encodings) causes silent data corruption
    concepts:
      - Text chunking strategies (fixed-size, sentence-aware, recursive)
      - Tokenizer alignment between chunker and embedding model
      - Document-chunk lineage tracking
      - Text normalization and cleaning
    skills:
      - Text preprocessing pipelines
      - Tokenizer usage (HuggingFace tokenizers)
      - Data pipeline design
      - Metadata management
    deliverables:
      - Document loader supporting plain text and JSON input formats
      - Sentence-aware text chunker with configurable token length and overlap
      - Text cleaner removing HTML, normalizing unicode, and handling encoding issues
      - Chunk metadata store linking each chunk back to its source document and position
      - Pipeline statistics reporter showing chunk count, length distribution, and processing time
    estimated_hours: "5-7"

  - id: semantic-search-m2
    name: Embedding Generation & Vector Index
    description: >-
      Generate dense vector embeddings for all chunks using a transformer model,
      then build an efficient approximate nearest neighbor index for similarity search.
    acceptance_criteria:
      - Load a Sentence-BERT model and generate 384+ dimensional embeddings for all chunks
      - Batch embedding generation processes chunks in configurable batch sizes (default 64) to maximize GPU/CPU throughput
      - All embedding vectors are L2-normalized before indexing to enable cosine similarity via inner product
      - Build an HNSW or IVF-based FAISS index over the embedding vectors
      - Index supports at least 100,000 vectors with sub-200ms query latency for top-10 retrieval
      - Index is serializable to disk and loadable without re-embedding
      - Incremental addition of new vectors without full index reconstruction (for HNSW; IVF requires re-training acknowledgment)
      - Numerical correctness verified by comparing FAISS results against brute-force exact search on a 1,000-vector subset (recall@10 >= 0.95)
    pitfalls:
      - Not normalizing vectors before indexing makes cosine similarity impossible with IndexFlatIP
      - IVF indexes require a training step on a representative sample before vectors can be added—skipping this produces garbage results
      - Setting HNSW M parameter too high (>64) causes memory explosion with diminishing recall gains
      - ID mapping between FAISS internal IDs and your chunk IDs gets out of sync if not managed explicitly
      - Embedding generation without batching is 10-100x slower than batched inference
      - Using the wrong distance metric (L2 vs inner product) silently produces incorrect rankings
    concepts:
      - Sentence-BERT embedding models
      - HNSW algorithm and its M/ef parameters
      - IVF indexing with nprobe tuning
      - Vector normalization for cosine similarity
      - Index persistence and memory-mapped access
    skills:
      - Transformer model inference and batching
      - FAISS index construction and tuning
      - Vector normalization and distance metrics
      - Binary serialization and deserialization
      - Memory-mapped file handling
    deliverables:
      - Embedding generator that batches chunks through Sentence-BERT and outputs normalized vectors
      - FAISS index builder supporting HNSW and IVF index types with configurable parameters
      - Index persistence module for saving and loading indexes to/from disk
      - ID mapping store linking FAISS internal IDs to chunk metadata
      - Recall validation script comparing ANN results against brute-force exact search
    estimated_hours: "7-10"

  - id: semantic-search-m3
    name: Query Processing & Hybrid Ranking
    description: >-
      Implement query embedding, BM25 keyword search, and a hybrid ranking system
      that fuses lexical and semantic signals for optimal relevance.
    acceptance_criteria:
      - Query text is embedded using the same Sentence-BERT model and normalization as document chunks
      - BM25 keyword index is built over the chunk corpus using tokenized text
      - Hybrid search combines BM25 score and cosine similarity score using a configurable weighted sum (default alpha=0.5)
      - Multi-stage ranking retrieves top-100 candidates from vector search, then re-ranks with combined score, returning top-10
      - Metadata filtering (e.g., by source document, date range, category) is applied as a pre-filter or post-filter on results
      - Query expansion with synonyms is implemented as an optional step, disabled by default to avoid intent dilution
      - Negative filtering is implemented via metadata exclusion filters, NOT vector subtraction (document this design decision)
      - Query embedding cache stores embeddings for repeated queries with TTL-based invalidation
    pitfalls:
      - BM25 and cosine similarity scores are on different scales—normalize both to [0,1] before combining
      - Vector subtraction for "negative queries" is unreliable in high dimensions and can flip the query into unrelated semantic space
      - Over-aggressive query expansion dilutes the original intent and hurts precision
      - Cross-encoder re-ranking on all candidates is too slow; restrict to top-N from the retrieval stage
      - Cache invalidation must account for embedding model changes, not just query text
      - Metadata filtering before vector search reduces recall; filtering after search reduces precision—choose deliberately
    concepts:
      - Hybrid search (lexical + semantic fusion)
      - BM25 scoring algorithm
      - Reciprocal Rank Fusion (RRF)
      - Score normalization techniques
      - Metadata filtering strategies
    skills:
      - BM25 implementation or library usage (rank_bm25)
      - Score normalization and fusion
      - Query embedding and caching
      - Filter predicate evaluation
      - Cross-encoder re-ranking
    deliverables:
      - Query embedding module using the same SBERT model as document embedding
      - BM25 keyword index built over the chunk corpus
      - Hybrid scorer combining normalized BM25 and cosine similarity with configurable weights
      - Multi-stage ranking pipeline (retrieve candidates → re-rank → return top-K)
      - Metadata filter engine supporting field-level inclusion and exclusion predicates
      - Query embedding cache with TTL-based invalidation
    estimated_hours: "7-10"

  - id: semantic-search-m4
    name: Evaluation & Quality Metrics
    description: >-
      Build an evaluation harness to measure search quality. Without metrics,
      you cannot know if your system works or if changes improve it.
    acceptance_criteria:
      - Evaluation dataset contains at least 50 queries with manually annotated relevant document IDs
      - MRR (Mean Reciprocal Rank) is computed over the evaluation set and reported
      - NDCG@10 is computed with graded relevance judgments
      - Recall@K is computed for K=1, 5, 10, 20 showing retrieval coverage
      - Comparison report shows metric differences between pure semantic, pure BM25, and hybrid search
      - Evaluation runs are reproducible with fixed random seeds and versioned datasets
    pitfalls:
      - Using too few evaluation queries (<20) produces unreliable metric estimates
      - Not using graded relevance (binary only) misses nuance in ranking quality
      - Evaluating on the same data used to tune hybrid weights is data leakage
      - Forgetting to fix random seeds makes evaluation non-reproducible
    concepts:
      - Information retrieval evaluation metrics (MRR, NDCG, Recall@K)
      - Relevance judgment annotation
      - Statistical significance of metric differences
      - Evaluation dataset design
    skills:
      - IR evaluation metric implementation
      - Annotation workflow design
      - Statistical comparison of search configurations
      - Reproducible experiment design
    deliverables:
      - Evaluation dataset with queries and annotated relevant document IDs
      - MRR, NDCG@10, and Recall@K metric calculators
      - Evaluation runner that scores a search configuration against the evaluation set
      - Comparison report generator showing metrics across search variants (semantic, BM25, hybrid)
    estimated_hours: "5-7"

  - id: semantic-search-m5
    name: Search API & Result Presentation
    description: >-
      Build a production-ready REST API serving search results with highlighting,
      pagination, and basic analytics.
    acceptance_criteria:
      - RESTful search endpoint accepts query string, filters, page number, and page size as parameters
      - Response includes ranked results with title, text snippet, relevance score, and source document metadata
      - Query term highlighting marks matched words in result snippets using configurable markers
      - Pagination returns total result count and supports cursor-based or offset-based paging
      - API responds with p95 latency under 200ms for queries against a 100K chunk index
      - Search analytics log records every query, result count, latency, and whether results were clicked (if UI exists)
      - Zero-result queries are logged separately for vocabulary gap analysis
      - API returns proper error responses (400 for bad input, 500 for internal errors) with descriptive messages
    pitfalls:
      - Not implementing pagination causes memory issues with large result sets
      - Highlighting that operates on raw HTML breaks entity encoding
      - Facet count computation is expensive—precompute or cache aggressively
      - Not logging zero-result queries means you miss the biggest opportunity for improvement
      - Sub-100ms autocomplete requires a separate lightweight index, not the full search pipeline
    concepts:
      - REST API design for search
      - Result snippet generation
      - Query logging and analytics
      - Pagination strategies
    skills:
      - REST API implementation (FastAPI/Flask)
      - Response serialization
      - Text highlighting algorithms
      - Logging and analytics pipeline
    deliverables:
      - RESTful search endpoint with query, filter, and pagination parameters
      - Result formatter producing title, snippet, score, and metadata per result
      - Query term highlighter marking matched words in result text
      - Pagination support with total count and page navigation
      - Search analytics logger recording query text, result count, and latency
      - Error handling middleware returning structured error responses
    estimated_hours: "6-9"
```
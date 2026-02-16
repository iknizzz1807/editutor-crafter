# AUDIT & FIX: search-engine

## CRITIQUE
- **Logical Gap (Confirmed):** The original jumps straight to 'Inverted Index' without a dedicated document preprocessing pipeline. Tokenization, stemming, stop-word removal, and Unicode normalization are prerequisites to building any meaningful index. Bundling them into Milestone 1 is insufficient—they deserve their own milestone because the correctness of every downstream component depends on the quality of text preprocessing.
- **Technical Inaccuracy (Confirmed):** BM25 requires average document length (avgdl) as a global corpus statistic. The original M2 mentions 'length normalization' in passing but never mandates computing, persisting, or incrementally updating avgdl. Without this, the BM25 implementation is mathematically incomplete.
- **Missing Measurability:** AC items like 'Handle large vocabularies with efficient memory usage' are vague. What is 'large'? What is 'efficient'? No benchmark targets are given anywhere.
- **Incorrect Tags:** 'game-dev' and 'framework' are irrelevant to a search engine project.
- **Architectural Weakness:** No mention of a Pager/Buffer Manager for the index. Memory-mapped I/O is listed as a skill but never appears in any milestone's deliverables or AC.
- **Missing Skip Lists/Posting List Compression:** The learning outcomes mention 'skip list indexing' and 'index compression' but no milestone actually requires implementing skip pointers for posting list intersection acceleration.
- **Estimated Hours Mismatch:** 4 milestones × 14 hours = 56 hours, which roughly matches the stated 55, but adding a preprocessing milestone pushes this to ~70 hours. The estimate must be updated.
- **Fuzzy Matching Milestone (M3):** Claims to use BK-trees or tries for optimization but the AC only requires basic Levenshtein distance. No AC validates sub-linear candidate filtering.
- **Query Parser (M4):** No AC for wildcard queries even though the pitfall mentions them. The pitfall about 'NOT without other terms returns entire index' is correct but no AC enforces a safeguard against it.

## FIXED YAML
```yaml
id: search-engine
name: Search Engine
description: >-
  Full-text search engine with document preprocessing, inverted index,
  probabilistic ranking (TF-IDF/BM25), fuzzy matching, and boolean query parsing.
difficulty: expert
estimated_hours: 70
essence: >-
  Document preprocessing pipeline (tokenization, normalization, stemming) feeding
  into inverted index data structures with posting lists and skip pointers,
  combined with BM25 probabilistic ranking using corpus-level statistics (IDF,
  average document length), approximate string matching via edit distance with
  candidate pre-filtering (BK-trees), and boolean/phrase query parsing for
  expressive retrieval over large corpora.
why_important: >-
  Building a full-text search engine teaches core information retrieval
  algorithms used by production systems like Elasticsearch and Lucene, while
  developing expertise in performance-critical systems programming with
  memory-efficient data structures and I/O-optimized index storage.
learning_outcomes:
  - Implement a text preprocessing pipeline with tokenization, Unicode normalization, stemming, and configurable stop-word removal
  - Build inverted index data structures with posting lists storing document IDs and term positions
  - Implement skip pointers in posting lists for efficient multi-term intersection
  - Design and optimize BM25 ranking with corpus statistics (IDF, average document length) and tunable parameters
  - Build fuzzy matching with Levenshtein distance and optimize candidate filtering using BK-trees
  - Implement prefix-based autocomplete with trie structures
  - Develop query parsers supporting boolean operators, phrase queries, and field-specific filters
  - Optimize index persistence with memory-mapped file I/O and posting list compression
  - Profile and benchmark search latency and throughput under load
skills:
  - Text Preprocessing
  - Inverted Index Design
  - Information Retrieval Algorithms
  - BM25 Ranking
  - Fuzzy String Matching
  - Tokenization and Text Normalization
  - Memory-Mapped I/O
  - Query Language Parsing
  - Performance Optimization
tags:
  - expert
  - indexing
  - information-retrieval
  - inverted-index
  - nlp
  - ranking
  - search
  - tokenization
  - build-from-scratch
architecture_doc: architecture-docs/search-engine/index.md
languages:
  recommended:
    - Rust
    - Go
    - C
  also_possible:
    - Python
    - Java
resources:
  - name: Stanford IR Book - Inverted Index
    url: https://nlp.stanford.edu/IR-book/html/htmledition/a-first-take-at-building-an-inverted-index-1.html
    type: documentation
  - name: Elasticsearch BM25 Guide
    url: https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables
    type: article
  - name: Inverted Index Tutorial
    url: https://www.baeldung.com/cs/indexing-inverted-index
    type: tutorial
  - name: Levenshtein Distance Implementation
    url: https://www.digitalocean.com/community/tutorials/levenshtein-distance-python
    type: tutorial
  - name: Meilisearch Documentation
    url: https://www.meilisearch.com/docs/home
    type: documentation
milestones:
  - id: search-engine-m1
    name: Document Preprocessing Pipeline
    description: >-
      Build a complete text preprocessing pipeline that converts raw document text
      into normalized, stemmed tokens suitable for indexing. This is the foundation
      for every subsequent milestone.
    acceptance_criteria:
      - Tokenizer splits raw text on whitespace and punctuation, producing individual word tokens, verified on at least 1000 documents
      - Unicode normalization (NFC) and case folding produce identical tokens for equivalent Unicode representations (e.g., 'café' in composed vs decomposed form)
      - Stemmer reduces inflected words to root forms using Porter or Snowball algorithm, verified with a test suite of at least 50 known stem mappings
      - Configurable stop-word list removes high-frequency noise words, with per-language list support for at least English
      - Pipeline processes at least 10,000 documents per second on a single core (measured via benchmark)
      - Pipeline is deterministic: identical input always produces identical token output
    pitfalls:
      - Stemming can over-stem (e.g., 'university' and 'universe' collapsing to same root)—validate with known false-positive pairs
      - Unicode normalization is not optional—without NFC normalization, visually identical strings produce different tokens
      - Stop-word removal before stemming vs after stemming produces different results; choose consistently
      - Punctuation handling must account for hyphens, apostrophes, and periods in abbreviations (e.g., 'U.S.A.')
      - Language detection is out of scope but stop-word lists must be language-specific
    concepts:
      - Tokenization splits text into atomic units for indexing
      - Unicode NFC normalization ensures canonical character representation
      - Case folding converts all characters to lowercase for case-insensitive matching
      - Stemming algorithms (Porter, Snowball) reduce words to approximate root forms
      - Stop-word removal eliminates high-frequency low-information terms
    skills:
      - Text tokenization
      - Unicode handling
      - Stemming algorithms
      - Pipeline architecture
    deliverables:
      - Tokenizer module splitting raw text into word tokens with configurable delimiters
      - Unicode normalizer applying NFC normalization and case folding
      - Stemmer implementation (Porter or Snowball) with test suite
      - Configurable stop-word filter with per-language word lists
      - Preprocessing pipeline composing tokenizer, normalizer, stemmer, and stop-word filter into a single callable unit
      - Benchmark harness measuring documents-per-second throughput
    estimated_hours: 10

  - id: search-engine-m2
    name: Inverted Index
    description: >-
      Build an inverted index from preprocessed tokens, supporting efficient
      term-to-document lookups, positional information for phrase queries,
      and persistent storage with compression.
    acceptance_criteria:
      - Inverted index maps each unique term to a posting list containing document IDs and within-document term positions
      - Index construction processes a corpus of at least 10,000 documents and completes within a measured time budget
      - Document addition appends to existing index without full rebuild; document deletion marks postings as removed
      - Posting lists are sorted by document ID for efficient intersection
      - Skip pointers are inserted every sqrt(N) postings to accelerate multi-term intersection to sub-linear in posting list length
      - Index serialization to disk uses variable-byte encoding for posting list gaps, reducing on-disk size by at least 40% compared to uncompressed storage (measured)
      - Index deserialization restores full index from disk with correct query results verified against in-memory version
    pitfalls:
      - Forgetting to store term positions makes phrase queries impossible later—always store positions from the start
      - Index updates (add/delete) are expensive; use batch operations and tombstone-based deletion with periodic compaction
      - Variable-byte encoding must handle edge cases (value 0, maximum integer values)
      - Skip pointer interval must be tuned—too frequent wastes space, too sparse gives no speedup
      - Memory usage can explode with large vocabularies; monitor and set limits
    concepts:
      - Posting lists map terms to sorted document IDs with positions
      - Skip pointers enable O(sqrt(N)) intersection on sorted posting lists
      - Variable-byte (VByte) encoding compresses integer gaps in posting lists
      - Batch index updates amortize I/O cost across many document additions
      - Tombstone-based deletion marks documents as removed without immediate physical removal
    skills:
      - Inverted index construction
      - Posting list data structures
      - Skip pointer implementation
      - Variable-byte encoding
      - Index persistence
    deliverables:
      - Term-to-posting-list mapping built from preprocessed document tokens
      - Posting list structure storing sorted document IDs and per-document term positions
      - Skip pointer layer on posting lists for accelerated intersection
      - Index construction pipeline processing document batches into inverted index
      - Variable-byte encoded index serialization to disk
      - Index deserialization and loading from disk
      - Batch document addition and tombstone-based deletion support
    estimated_hours: 16

  - id: search-engine-m3
    name: TF-IDF & BM25 Ranking
    description: >-
      Implement relevance ranking using TF-IDF as a baseline and BM25 as the
      primary ranking algorithm, including all required corpus-level statistics.
    acceptance_criteria:
      - TF (term frequency) is computed as raw count of term occurrences within each document
      - IDF (inverse document frequency) is computed as log(N/df_t) where N is total documents and df_t is document frequency of term t, precomputed and cached
      - Average document length (avgdl) is computed across the entire corpus and updated incrementally on document add/delete
      - BM25 score is computed using the formula with k1 (default 1.2) and b (default 0.75) parameters, matching reference implementation output within 0.001 tolerance on a shared test corpus
      - Search results are returned in descending BM25 score order
      - Multi-term queries compute BM25 as the sum of per-term BM25 scores
      - Ranking latency for a 3-term query over a 100K document corpus is under 50ms (measured via benchmark)
    pitfalls:
      - Forgetting to compute and persist avgdl makes BM25 length normalization incorrect—this is a hard requirement
      - IDF values must be precomputed and updated on corpus changes, not recomputed per query
      - Very short documents (1-2 words) get artificially inflated scores; verify with edge cases
      - BM25 k1 and b parameter tuning is corpus-dependent; provide configurable values with sensible defaults
      - Division by zero when document length is 0 or df_t is 0; guard all calculations
    concepts:
      - Term frequency (TF) measures how often a term appears in a document
      - Inverse document frequency (IDF) weighs rare terms higher than common ones
      - BM25 extends TF-IDF with term frequency saturation (controlled by k1) and document length normalization (controlled by b, using avgdl)
      - Average document length (avgdl) is a corpus-level statistic required by BM25
      - Field boosting assigns different weights to title, body, metadata fields
    skills:
      - TF-IDF computation
      - BM25 algorithm
      - Relevance scoring
      - Corpus statistics management
    deliverables:
      - TF computation per term per document
      - IDF computation with precomputed and cached values per term
      - Average document length (avgdl) computation and incremental update logic
      - BM25 scoring function with configurable k1 and b parameters
      - Multi-term BM25 aggregation summing per-term scores
      - Ranked result list sorted by descending BM25 score
      - Benchmark suite measuring ranking latency on a 100K document corpus
    estimated_hours: 14

  - id: search-engine-m4
    name: Fuzzy Matching & Autocomplete
    description: >-
      Implement typo-tolerant search using edit distance algorithms and
      prefix-based autocomplete using trie or BK-tree structures for
      sub-linear candidate filtering.
    acceptance_criteria:
      - Levenshtein distance correctly computes minimum edit distance between two strings, verified against a test suite of at least 20 known pairs
      - Fuzzy search returns index terms within a configurable edit distance (default max 2) of the query term
      - BK-tree (or equivalent) pre-filters candidates, reducing comparisons to sub-linear in vocabulary size—verified by counting distance computations vs brute-force
      - Edit distance threshold scales with term length: max 1 edit for terms ≤4 chars, max 2 edits for terms >4 chars
      - Prefix-based autocomplete returns top-10 completions ranked by document frequency, with latency under 10ms for a 500K term vocabulary
      - Damerau-Levenshtein distance (with transpositions) is supported as an alternative metric
    pitfalls:
      - Naive O(n*m) Levenshtein against every vocabulary term is prohibitively slow—BK-tree or n-gram pre-filtering is mandatory for large vocabularies
      - Allowing 2 edits on 3-character words produces excessive false matches; enforce length-dependent thresholds
      - Transpositions are the most common typo type; Damerau-Levenshtein handles them with one operation instead of two
      - Trie memory usage can be large for big vocabularies; consider compressed tries (DAWG/FST) if memory constrained
      - Autocomplete must be fast enough for interactive use (<10ms); lazy loading and caching are essential
    concepts:
      - Levenshtein distance counts insertions, deletions, and substitutions
      - Damerau-Levenshtein adds transposition as a single operation
      - BK-tree uses triangle inequality of edit distance for sub-linear candidate filtering
      - Trie (prefix tree) enables O(k) prefix lookups where k is prefix length
      - N-gram indexing breaks terms into character n-grams for approximate matching
    skills:
      - Edit distance algorithms
      - BK-tree data structure
      - Trie/prefix tree implementation
      - Fuzzy search optimization
    deliverables:
      - Levenshtein distance computation with dynamic programming
      - Damerau-Levenshtein distance supporting transpositions
      - BK-tree construction from vocabulary for sub-linear candidate filtering
      - Fuzzy search integrating BK-tree pre-filtering with edit distance verification
      - Length-dependent edit distance thresholds
      - Trie-based prefix autocomplete returning top-k completions ranked by frequency
      - Benchmark comparing BK-tree filtered search vs brute-force on a 500K term vocabulary
    estimated_hours: 16

  - id: search-engine-m5
    name: Query Parser & Filters
    description: >-
      Implement a query language parser supporting boolean operators, phrase
      queries, field-specific filters, and safety guards against expensive
      query patterns.
    acceptance_criteria:
      - Recursive descent parser converts query strings into an abstract syntax tree (AST)
      - Boolean AND combines posting lists via sorted intersection; OR via sorted union; NOT via sorted difference
      - Phrase queries verify term adjacency using positional index data, returning only documents where terms appear consecutively in order
      - Field-specific filters (e.g., 'author:smith', 'title:database') restrict search to named fields
      - Numeric range queries (e.g., 'year:2020..2024') filter by min/max values
      - Standalone NOT queries are rejected with an error message (prevents returning entire index)
      - Query depth is limited to a configurable maximum (default 10) to prevent stack overflow from deeply nested expressions
      - Leading wildcard queries (e.g., '*tion') are rejected or flagged as expensive
    pitfalls:
      - Phrase search requires positional data in the index—if M2 skipped positions, this milestone is blocked
      - NOT without a positive clause returns the complement of the entire corpus; always require at least one positive term
      - Deeply nested boolean expressions can cause stack overflow; enforce a recursion depth limit
      - Leading wildcards require scanning every term in the vocabulary; either reject them or implement n-gram index
      - Operator precedence: NOT > AND > OR; mishandling this produces incorrect results
    concepts:
      - Recursive descent parsing converts query strings into ASTs
      - Boolean operators (AND, OR, NOT) combine posting lists via set operations
      - Phrase queries require positional indexes to verify term adjacency
      - Field-specific search partitions the index by document field
      - Range filters use numeric comparisons on indexed field values
    skills:
      - Query parsing
      - Boolean set operations on posting lists
      - Phrase query evaluation with positional data
      - Field and range filtering
    deliverables:
      - Recursive descent query parser producing an AST from query strings
      - Boolean AND/OR/NOT evaluation using posting list set operations
      - Phrase query evaluation verifying consecutive term positions
      - Field-specific filter restricting search to named document fields
      - Numeric range filter on indexed numeric fields
      - Query validation rejecting standalone NOT, excessive depth, and leading wildcards
      - Integration test suite with at least 30 query patterns covering all operators and edge cases
    estimated_hours: 14
```
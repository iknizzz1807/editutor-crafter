# AUDIT & FIX: rag-system

## CRITIQUE
- **BM25 Requires Inverted Index Not Built**: The audit correctly identifies that M3 requires hybrid search with BM25, but M1 (Document Ingestion) doesn't build an inverted index. BM25 is a sparse retrieval method that requires term frequency statistics and an inverted index over the corpus. This is architecturally significant and must be addressed.
- **Re-ranking Mentioned but Not a Milestone**: M3 deliverables include 'Re-ranking retrieved results using cross-encoder model' but this is a deliverable, not an AC, and the complexity of cross-encoder re-ranking (loading a separate model, computing pairwise scores) is significant enough to deserve proper treatment.
- **Prompt Injection Risk Understated**: M4 pitfall mentions 'Prompt injection from retrieved content' but no AC requires mitigation. Retrieved documents can contain adversarial text that hijacks the LLM prompt. This is a real production security concern.
- **Missing Query Augmentation**: The project mentions 'query augmentation' in learning outcomes but no milestone implements it. Techniques like HyDE (Hypothetical Document Embeddings) or query expansion significantly improve retrieval quality.
- **Evaluation Milestone Conflates Retrieval and Generation Metrics**: M5 mixes retrieval evaluation (recall@K, MRR) with generation evaluation (faithfulness, relevance). These are different evaluation stages with different methodologies and should be clearly separated.
- **No Chunking Strategy Evaluation in M1**: M1 implements chunking strategies but doesn't require comparing them. Without evaluation, how does the learner know which strategy works best?
- **Embedding Normalization**: M2 pitfall mentions normalization but no AC requires it. For cosine similarity (which many vector DBs use), embeddings must be L2-normalized.
- **Context Window Management Oversimplified**: M4 says 'Handle context window limits by truncating excess chunks' but truncation loses potentially relevant information. Better strategies include relevance-based selection and summarization.

## FIXED YAML
```yaml
id: rag-system
name: "RAG System (Retrieval Augmented Generation)"
description: "Build a production RAG pipeline: document ingestion, chunking, embedding, hybrid search, re-ranking, and LLM generation with evaluation."
difficulty: intermediate
estimated_hours: "35-55"
essence: >
  Semantic similarity search through dense vector embeddings combined with
  sparse keyword retrieval (BM25), cross-encoder re-ranking for precision,
  and context-augmented LLM prompting—with systematic evaluation of retrieval
  quality and answer faithfulness.
why_important: >
  RAG is the foundation for most real-world AI applications in 2024+, from
  chatbots to enterprise knowledge bases. Building one from scratch teaches
  embedding pipelines, vector databases, hybrid search, and LLM integration—
  the core skills for AI engineering roles.
learning_outcomes:
  - Implement document chunking strategies with overlap and semantic boundaries
  - Generate and normalize vector embeddings using transformer models
  - Build hybrid search combining dense vector similarity and sparse BM25 retrieval
  - Implement cross-encoder re-ranking for retrieval precision improvement
  - Integrate retrieval context into LLM prompts with token budget management
  - Evaluate RAG quality using retrieval metrics (recall@K, MRR) and generation metrics (faithfulness)
  - Handle production concerns: prompt injection mitigation, caching, and streaming
skills:
  - Vector Embeddings
  - Semantic Search
  - Document Processing
  - LLM Prompting
  - Vector Databases
  - BM25 Information Retrieval
  - Cross-Encoder Re-ranking
  - RAG Evaluation
tags:
  - ai-ml
  - embeddings
  - intermediate
  - python
  - retrieval
  - search
  - rag
  - llm
architecture_doc: architecture-docs/rag-system/index.md
languages:
  recommended:
    - Python
  also_possible:
    - TypeScript
resources:
  - name: LangChain RAG Tutorial""
    url: https://python.langchain.com/docs/tutorials/rag/
    type: tutorial
  - name: OpenAI Embeddings Guide""
    url: https://platform.openai.com/docs/guides/embeddings
    type: documentation
  - name: BM25 Algorithm Explained""
    url: https://en.wikipedia.org/wiki/Okapi_BM25
    type: article
  - name: Cross-Encoder Re-ranking (SBERT)""
    url: https://www.sbert.net/examples/applications/cross-encoder/README.html
    type: documentation
  - name: RAGAS Evaluation Framework""
    url: https://docs.ragas.io/
    type: documentation
prerequisites:
  - type: skill
    name: "Python"
  - type: skill
    name: "REST API consumption"
  - type: skill
    name: "Basic ML concepts (vectors, similarity)"
  - type: skill
    name: "Database basics (CRUD operations)"
milestones:
  - id: rag-system-m1
    name: "Document Ingestion & Chunking"
    description: >
      Load documents from multiple formats, extract text with metadata,
      and split into chunks using configurable strategies with overlap.
    acceptance_criteria:
      - "Document loader reads PDF (via PyPDF2 or pdfplumber), Markdown, HTML, and plain text files"
      - Text extraction preserves document metadata: source filename, page number (for PDF), section headings
      - "Fixed-size chunking splits text into chunks of configurable token count (default 512 tokens) with configurable overlap (default 50 tokens)"
      - "Recursive character splitting breaks text at paragraph → sentence → word boundaries, respecting a maximum chunk size"
      - Each chunk object contains: chunk text, chunk ID, source document metadata, and character offset within the source
      - Chunk statistics are reported: total chunks generated, average chunk length, min/max chunk length
      - "UTF-8 encoding is handled correctly; non-UTF-8 files produce a clear error message"
    pitfalls:
      - "Chunks too small (<100 tokens) lose semantic context; chunks too large (>1000 tokens) may exceed embedding model limits and dilute relevance"
      - "PDF extraction loses tables, images, and formatting—document these limitations explicitly"
      - "Overlap too large wastes storage and embedding compute; too small loses cross-boundary context"
      - "Not tracking source metadata makes it impossible to provide citations in generated answers"
    concepts:
      - Document parsing and extraction
      - Text chunking strategies (fixed, recursive, semantic)
      - Chunk overlap for context continuity
      - Metadata tracking for provenance
    skills:
      - File I/O and parsing (PDF, HTML, Markdown)
      - Text preprocessing and cleaning
      - Tokenization and character counting
      - Metadata management
    deliverables:
      - "Multi-format document loader (PDF, Markdown, HTML, TXT)"
      - "Fixed-size chunker with configurable token count and overlap"
      - "Recursive character splitter respecting paragraph/sentence boundaries"
      - "Chunk metadata tracking source, page, and offset information"
    estimated_hours: "5-8"

  - id: rag-system-m2
    name: "Embedding Generation & Indexing"
    description: >
      Convert text chunks to vector embeddings using transformer models,
      build both a vector index and an inverted index (for BM25), and
      store them for retrieval.
    acceptance_criteria:
      - "Embeddings are generated using OpenAI API or a local sentence-transformer model (e.g., all-MiniLM-L6-v2)"
      - "Embeddings are L2-normalized so that cosine similarity equals dot product similarity"
      - "Batch processing generates embeddings in configurable batch sizes (default 32) to respect API rate limits"
      - "Embedding cache persists computed vectors to disk (pickle or numpy) to avoid redundant API calls on re-run"
      - BM25 inverted index is built from chunk text: for each term, store the list of chunk IDs containing it with term frequency
      - "Vector store (Chroma, FAISS, or pgvector) indexes all chunk embeddings with associated metadata"
      - Metadata filtering is supported: queries can restrict search to specific source documents or date ranges
    pitfalls:
      - "Not normalizing embeddings causes cosine similarity to incorrectly favor longer vectors"
      - "API rate limits without retry logic cause silent data loss—implement exponential backoff"
      - "Embedding model dimension mismatch between indexing and query time causes runtime errors—store model name in index metadata"
      - "BM25 index must be rebuilt when new documents are added—design for incremental updates or full rebuild"
    concepts:
      - Dense embeddings from transformer models
      - L2 normalization for cosine similarity
      - Inverted index for sparse retrieval (BM25)
      - Vector database indexing
    skills:
      - Embedding API integration
      - Batch processing with rate limiting
      - Vector database operations
      - Inverted index construction
    deliverables:
      - "Embedding generator with batch processing and API rate limit handling"
      - "L2 normalization of all embedding vectors before storage"
      - "Vector store integration (Chroma/FAISS/pgvector) with metadata"
      - "BM25 inverted index built from chunk text"
      - "Embedding cache for avoiding redundant computation"
    estimated_hours: "6-9"

  - id: rag-system-m3
    name: "Hybrid Retrieval & Re-ranking"
    description: >
      Implement hybrid search combining dense vector similarity and sparse
      BM25 retrieval, with cross-encoder re-ranking for improved precision.
    acceptance_criteria:
      - "Dense retrieval returns top-K chunks by cosine similarity from the vector store (default K=20)"
      - "Sparse retrieval returns top-K chunks by BM25 score from the inverted index (default K=20)"
      - Hybrid search merges dense and sparse results using Reciprocal Rank Fusion (RRF): score = Σ 1/(k + rank_i) across retrieval methods
      - "Cross-encoder re-ranker scores each (query, chunk) pair using a pre-trained model (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2) and returns top-N by re-ranked score (default N=5)"
      - Re-ranking improves precision: on a test set of 20 queries with labeled relevant chunks, re-ranked top-5 has higher precision than unreranked top-5
      - "Metadata filters can be applied before or after retrieval to restrict results by source or attribute"
    pitfalls:
      - "BM25 and dense scores are on different scales—RRF avoids the need for score normalization by using ranks"
      - "Cross-encoder is slow (O(K) model forward passes per query)—limit the number of candidates passed to re-ranker"
      - "Stale BM25 index after document updates returns outdated results—rebuild on ingestion"
      - "Re-ranker model downloads can be large (~100MB)—handle first-run download gracefully"
    concepts:
      - Dense vs. sparse retrieval
      - Reciprocal Rank Fusion (RRF)
      - Cross-encoder re-ranking
      - Two-stage retrieval pipeline
    skills:
      - Vector similarity search
      - BM25 scoring implementation
      - Score fusion algorithms
      - Pre-trained model integration
    deliverables:
      - "Dense retrieval via vector store cosine similarity search"
      - "Sparse retrieval via BM25 scoring over inverted index"
      - "Reciprocal Rank Fusion merging dense and sparse results"
      - "Cross-encoder re-ranker scoring and re-ordering merged candidates"
    estimated_hours: "7-10"

  - id: rag-system-m4
    name: "LLM Integration & Prompting"
    description: >
      Generate answers using retrieved context injected into LLM prompts,
      with token budget management, streaming, and prompt injection mitigation.
    acceptance_criteria:
      - RAG prompt template includes: system instruction, retrieved context chunks with source citations, and user query
      - "System instruction directs the LLM to answer only from provided context and cite sources"
      - "Token budget manager calculates available tokens for context after accounting for system prompt, user query, and expected response length"
      - "Context is selected by re-ranked relevance score until the token budget is filled; excess chunks are dropped"
      - "LLM responses are streamed token-by-token to the client for perceived latency reduction"
      - Basic prompt injection mitigation: retrieved chunk text is enclosed in clearly delimited markers (e.g., <context>...</context>) and the system prompt instructs the LLM to treat content within markers as data, not instructions
      - "LLM API errors and timeouts return a graceful error message to the user, not a stack trace"
    pitfalls:
      - "Exceeding context window silently truncates input and produces incoherent answers—must count tokens accurately"
      - "LLM may hallucinate despite 'only use context' instruction—this is a known limitation, not a solvable bug"
      - "Streaming error mid-response (e.g., API timeout) leaves partial response—handle with error suffix"
      - Prompt injection from retrieved content: a malicious document could contain 'Ignore previous instructions'—delimiter-based mitigation reduces but doesn't eliminate risk
    concepts:
      - Prompt engineering for RAG
      - Token budget management
      - Streaming LLM responses
      - Prompt injection defense
    skills:
      - LLM API integration (OpenAI, Anthropic)
      - Token counting (tiktoken)
      - Prompt template design
      - Streaming response handling
      - Input sanitization
    deliverables:
      - "RAG prompt template with system instruction, context section, and query formatting"
      - "Token budget calculator selecting context chunks within available token limit"
      - "LLM API wrapper with streaming support and error handling"
      - "Prompt injection mitigation using context delimiters and system-level instruction"
      - "Source citation extraction from LLM response"
    estimated_hours: "6-9"

  - id: rag-system-m5
    name: "Evaluation & Optimization"
    description: >
      Systematically evaluate and optimize RAG quality using retrieval metrics,
      generation quality metrics, and configuration experiments.
    acceptance_criteria:
      - "Evaluation dataset contains at least 30 questions with labeled relevant chunk IDs and expected answer summaries"
      - Retrieval metrics are computed: Recall@K (proportion of relevant chunks in top-K), Mean Reciprocal Rank (MRR), and Precision@K
      - Generation quality metrics are computed: faithfulness (answer is supported by retrieved context) and answer relevance (answer addresses the question)—using LLM-as-judge or RAGAS framework
      - Configuration experiment: at least 3 chunk size configurations (e.g., 256, 512, 1024 tokens) are compared on retrieval metrics
      - Configuration experiment: hybrid search (dense + BM25 + re-ranking) is compared against dense-only search on retrieval metrics
      - "Results are presented in a comparison table showing metric values for each configuration"
      - "Best configuration is selected and documented with justification"
    pitfalls:
      - "Evaluation set too small (<20 questions) produces high-variance metrics—aim for 30+"
      - "LLM-as-judge biases toward verbose, confident-sounding responses regardless of correctness—validate a sample manually"
      - "Overfitting to evaluation set by tuning parameters specifically to maximize evaluation metrics—use separate dev and test sets if possible"
      - "Not versioning evaluation datasets makes results non-reproducible"
    concepts:
      - Information retrieval evaluation (Recall, MRR, Precision)
      - Generation evaluation (faithfulness, relevance)
      - LLM-as-judge evaluation methodology
      - Hyperparameter optimization for RAG
    skills:
      - Computing retrieval metrics
      - LLM-based evaluation pipelines
      - Experiment design and comparison
      - Results documentation and analysis
    deliverables:
      - "Labeled evaluation dataset with questions, relevant chunk IDs, and expected answers"
      - "Retrieval metric calculator (Recall@K, MRR, Precision@K)"
      - "Generation quality scorer (faithfulness and relevance via LLM-as-judge)"
      - "Configuration comparison table across chunk sizes and retrieval strategies"
      - "Optimization report documenting best configuration with evidence"
    estimated_hours: "8-12"
```
# AUDIT & FIX: word2vec

## CRITIQUE
- **Softmax → Negative Sampling Transition Unclear**: The audit correctly identifies that M2 implements full softmax and M3 introduces negative sampling as a replacement. The milestones should make it explicit that negative sampling is an approximation that replaces the intractable full softmax, not an additional component. M2 should implement the conceptual model, and M3 should replace the training objective.
- **Subsampling Timing**: The audit notes that subsampling of frequent words should be done during pair generation, not as a separate preprocessing step. Subsampling removes words from the corpus with probability proportional to their frequency, and this affects the context window—doing it as preprocessing changes window boundaries.
- **M2 Full Softmax is Impractical**: For any vocabulary larger than ~10K words, computing softmax over the entire vocabulary is computationally prohibitive. M2 should acknowledge this is only feasible for tiny vocabularies or toy datasets, and frame it as conceptual before M3's optimization.
- **Cross-Entropy Loss in M3 is Technically Binary Cross-Entropy**: The loss for negative sampling is binary cross-entropy (sigmoid cross-entropy), not the same as categorical cross-entropy used in full softmax. These are different loss functions and the terminology should be precise.
- **No Mention of Window Weighting**: In the original Word2Vec, closer context words are weighted more heavily. This is a detail that affects embedding quality but is missing.
- **Analogy Evaluation Metrics Missing**: M4 says 'Analogy task: king - man + woman = queen' but doesn't specify accuracy metrics (e.g., analogy accuracy percentage on a standard analogy benchmark).
- **Embedding Dimension Not Discussed**: No guidance on choosing embedding dimensions (50-300 typical) and the tradeoff between capacity and overfitting.
- **Missing: Training on a Real Corpus**: No AC specifies what corpus to train on. Without a meaningful corpus (e.g., Wikipedia subset, text8), the embeddings won't be semantically interesting.

## FIXED YAML
```yaml
id: word2vec
name: "Word Embeddings (Word2Vec)"
description: "Implement Word2Vec skip-gram with negative sampling from scratch, producing dense word vectors that capture semantic relationships."
difficulty: intermediate
estimated_hours: "18-28"
essence: >
  Training a shallow neural network to predict context words from target words
  (skip-gram), producing dense vector representations where geometric
  relationships encode semantic meaning, with negative sampling replacing
  the intractable full-vocabulary softmax computation.
why_important: >
  Word embeddings are foundational to modern NLP. Implementing Word2Vec from
  scratch teaches how neural networks learn distributed representations,
  efficient training with negative sampling, and the mathematical principles
  behind semantic vector spaces used in production systems.
learning_outcomes:
  - Build text preprocessing pipelines with tokenization, vocabulary construction, and frequency-based filtering
  - Implement skip-gram context window pair generation with subsampling of frequent words
  - Understand why full softmax is intractable and how negative sampling approximates it
  - Implement negative sampling training with binary cross-entropy loss
  - Optimize embedding weights using stochastic gradient descent with backpropagation
  - Evaluate embedding quality using cosine similarity and word analogy tasks
  - Visualize high-dimensional embeddings using dimensionality reduction (t-SNE/PCA)
skills:
  - Neural Network Training
  - Embedding Representations
  - Negative Sampling
  - Text Preprocessing
  - Vector Space Models
  - Stochastic Gradient Descent
  - Semantic Similarity
  - Dimensionality Reduction
tags:
  - embeddings
  - intermediate
  - nlp
  - python
  - skip-gram
  - word-vectors
  - negative-sampling
architecture_doc: architecture-docs/word2vec/index.md
languages:
  recommended:
    - Python
  also_possible:
    - Julia
    - C++
resources:
  - name: "Efficient Estimation of Word Representations (Mikolov et al.)"
    url: https://arxiv.org/abs/1301.3781
    type: paper
  - name: "Distributed Representations of Words (Negative Sampling paper)"
    url: https://arxiv.org/abs/1310.4546
    type: paper
  - name: "Word2Vec Tutorial (TensorFlow)"
    url: https://www.tensorflow.org/tutorials/text/word2vec
    type: tutorial
  - name: "text8 Dataset (100MB Wikipedia text)"
    url: http://mattmahoney.net/dc/textdata.html
    type: documentation
prerequisites:
  - type: skill
    name: "Neural network basics (forward/backward pass)"
  - type: skill
    name: "Linear algebra (dot products, matrix operations)"
  - type: skill
    name: "Python and NumPy"
milestones:
  - id: word2vec-m1
    name: "Text Preprocessing & Pair Generation"
    description: >
      Prepare a text corpus for training: tokenize, build vocabulary with
      frequency filtering, generate skip-gram training pairs with subsampling
      of frequent words.
    acceptance_criteria:
      - "Tokenizer splits raw text into lowercase word tokens with punctuation removed"
      - "Vocabulary is built with word-to-index and index-to-word mappings, filtering words below minimum frequency threshold (default 5)"
      - "Subsampling of frequent words is applied during pair generation: each word is kept with probability P(w) = sqrt(t/f(w)) where t is a threshold (default 1e-5) and f(w) is the word's corpus frequency"
      - "Skip-gram pairs are generated using a sliding context window of configurable radius (default 5): for each target word, create (target, context) pairs for each word within the window"
      - "Subsampled words are removed from the sequence before windowing, so window neighbors are the remaining words—not the original positional neighbors"
      - "Training corpus is at least 1MB of text (e.g., text8 dataset or Wikipedia subset) for meaningful embeddings"
    pitfalls:
      - "Vocabulary too large consumes excessive memory—enforce minimum frequency threshold to cap vocabulary at ~50K words"
      - "Subsampling must happen during pair generation (removing words from the token stream before windowing), not as a separate preprocessing step—otherwise context windows span removed positions incorrectly"
      - "Memory issues with large corpus: generate pairs lazily (generator) instead of materializing all pairs in memory"
      - "Rare words provide almost no training signal and bloat the vocabulary—filter aggressively"
    concepts:
      - Tokenization and vocabulary construction
      - Frequency-based word filtering
      - Skip-gram context window
      - Subsampling of frequent words
    skills:
      - Text preprocessing
      - Vocabulary management with index mappings
      - Generator-based lazy data loading
      - Frequency statistics computation
    deliverables:
      - "Tokenizer splitting raw text into normalized word tokens"
      - "Vocabulary builder with frequency filtering and bidirectional word↔index mapping"
      - "Skip-gram pair generator producing (target, context) pairs from sliding window"
      - "Frequency-based subsampling applied during pair generation"
    estimated_hours: "3-5"

  - id: word2vec-m2
    name: "Skip-gram Model Architecture"
    description: >
      Implement the skip-gram neural network with input and output embedding
      matrices, and understand why full softmax is intractable for large vocabularies.
    acceptance_criteria:
      - "Input embedding matrix W_in of shape (vocab_size, embedding_dim) maps target word index to a dense vector; embedding dimension is configurable (default 100)"
      - "Output embedding matrix W_out of shape (vocab_size, embedding_dim) maps context word index to a dense vector"
      - "Both matrices are initialized with small random values (e.g., uniform in [-0.5/dim, 0.5/dim])"
      - "Forward pass computes the dot product between target's input embedding and context's output embedding: score = W_in[target] · W_out[context]"
      - "Full softmax probability is implemented for verification on a tiny vocabulary (<1000 words): P(context|target) = exp(score) / Σ_w exp(W_in[target] · W_out[w])"
      - "Demonstration that full softmax is O(V) per training pair (where V is vocabulary size), making it impractical for large vocabularies"
    pitfalls:
      - "Full softmax is only feasible for toy vocabularies—don't try to train with it on real data"
      - "Embedding dimension too small (<50) produces low-quality embeddings; too large (>300) overfits on small corpora"
      - "Numerical instability in softmax: subtract max score before exp() to prevent overflow"
      - "Confusing W_in and W_out: they are separate matrices—the final word embeddings are typically W_in"
    concepts:
      - Embedding matrices (input and output)
      - Skip-gram prediction objective
      - Softmax as a probability distribution
      - Computational cost of full softmax
    skills:
      - Matrix initialization and indexing
      - Dot product computation
      - Softmax implementation with numerical stability
      - Computational complexity analysis
    deliverables:
      - "Input and output embedding matrices with configurable dimension"
      - "Embedding lookup returning the dense vector for a given word index"
      - "Forward pass computing dot product score between target and context embeddings"
      - "Full softmax implementation (for verification on tiny vocab only) with numerical stability"
    estimated_hours: "3-4"

  - id: word2vec-m3
    name: "Training with Negative Sampling"
    description: >
      Replace the intractable full softmax with negative sampling: for each
      positive (target, context) pair, sample K negative context words and
      train with binary cross-entropy loss.
    acceptance_criteria:
      - "Negative sampler draws K negative words (default K=5) per positive pair from a noise distribution proportional to word frequency raised to the 3/4 power: P_noise(w) ∝ count(w)^0.75"
      - "Loss function is binary cross-entropy (sigmoid cross-entropy): L = -log(σ(score_positive)) - Σ_k log(σ(-score_negative_k)) where σ is the sigmoid function"
      - "Gradient computation correctly updates both W_in[target] and W_out[context] (and W_out[negative_k] for each negative sample)"
      - "SGD parameter updates: each embedding vector is updated by subtracting learning_rate * gradient after each training pair"
      - "Training over the full corpus for at least 5 epochs shows decreasing average loss"
      - "Learning rate linearly decays from initial value (default 0.025) to near-zero over the course of training"
    pitfalls:
      - "Sampling distribution must use the 3/4 power of frequency, not raw frequency—this gives rare words more representation as negatives"
      - "Drawing the target word itself as a negative sample wastes computation—filter it out"
      - "Sigmoid overflow for large scores: clip input to sigmoid to [-10, 10] range"
      - "Learning rate too high causes embeddings to diverge; too low makes training impractically slow—use linear decay schedule"
    concepts:
      - Negative sampling as softmax approximation
      - Binary cross-entropy loss
      - Noise contrastive estimation
      - Learning rate scheduling
    skills:
      - Sampling from weighted distributions
      - Sigmoid and binary cross-entropy implementation
      - Gradient derivation for embedding updates
      - Learning rate decay schedules
    deliverables:
      - "Noise distribution table for efficient weighted negative sampling (precomputed)"
      - "Binary cross-entropy loss for positive pair and K negative samples"
      - "Gradient computation updating target input embedding and all context output embeddings"
      - "SGD training loop with linear learning rate decay over epochs"
    estimated_hours: "5-7"

  - id: word2vec-m4
    name: "Evaluation & Visualization"
    description: >
      Evaluate embedding quality using similarity queries and analogy tasks,
      and visualize the embedding space.
    acceptance_criteria:
      - "Cosine similarity function computes sim(a,b) = (a·b) / (||a|| * ||b||) between two word vectors"
      - "most_similar(word, k=10) returns the K nearest words by cosine similarity, excluding the query word itself"
      - "Word analogy function computes a - b + c and returns the nearest word to the result vector (e.g., king - man + woman ≈ queen)"
      - "Analogy evaluation on at least 20 test analogies reports accuracy (percentage of correct top-1 predictions)"
      - "t-SNE or PCA visualization projects embeddings to 2D and displays a scatter plot with word labels for a selected subset (~100 words)"
      - "Trained embeddings are saved to disk in a standard format (text: word followed by space-separated floats, one word per line)"
      - "Embeddings can be loaded from disk and used for similarity/analogy queries without retraining"
    pitfalls:
      - "Not L2-normalizing embedding vectors before cosine similarity causes incorrect rankings"
      - "Including the query word in similarity results is a common bug—explicitly exclude it"
      - "Analogy results are poor with small corpora or few training epochs—set realistic expectations"
      - "t-SNE is stochastic: different runs produce different layouts—set random seed for reproducibility"
    concepts:
      - Cosine similarity in embedding space
      - Word analogy via vector arithmetic
      - Dimensionality reduction for visualization
      - Embedding serialization
    skills:
      - Vector similarity computation
      - Nearest neighbor search
      - t-SNE/PCA visualization
      - File I/O for embedding format
    deliverables:
      - "Cosine similarity function with L2 normalization"
      - "most_similar query returning top-K nearest words"
      - "Word analogy evaluator with accuracy metric on test analogies"
      - "t-SNE/PCA 2D visualization of selected word embeddings"
      - "Embedding save/load in standard text format"
    estimated_hours: "3-5"
```
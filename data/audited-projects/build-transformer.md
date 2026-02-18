# AUDIT & FIX: build-transformer

## CRITIQUE
- **High Redundancy with transformer-scratch**: This project and 'Transformer from Scratch' (transformer-scratch) overlap approximately 80%. Both implement self-attention, multi-head attention, FFN, and training. The key difference is this project focuses on decoder-only (GPT-style) while transformer-scratch covers encoder-decoder. This distinction must be made explicit.
- **Missing Pre-LN vs Post-LN in AC**: The audit correctly identifies that LayerNorm placement is a critical architectural decision affecting training stability. The AC in M2 should explicitly require choosing and implementing one, with documentation of why.
- **M1 Combines Two Milestones**: Self-attention AND multi-head attention in a single milestone is too much. The transformer-scratch project correctly separates them. However, since this is an expert-level project, combining them may be acceptable if the AC is rigorous.
- **M3 Training Missing Key Details**: No mention of learning rate warmup (critical for transformer training), gradient clipping, or mixed precision. The pitfalls mention 'no gradient clipping' but the AC doesn't require it.
- **M4 KV Cache Missing Detail**: KV cache is mentioned but the AC doesn't specify the memory savings or require benchmarking. KV caching is the #1 inference optimization and deserves more rigor.
- **Missing Positional Encoding Milestone**: Token embeddings and positional encoding are lumped into M2 as an afterthought. Given this is a GPT implementation, the embedding layer (token + position) should be explicitly addressed.
- **No Verification Against Reference**: Unlike transformer-scratch (fixed version), this project doesn't require comparing outputs against a reference implementation.
- **Description Too Terse**: 'Full GPT implementation' and 'Attention mechanism' are unhelpfully vague descriptions.

## FIXED YAML
```yaml
id: build-transformer
name: Build Your Own GPT
description: >-
  Implement a decoder-only GPT-style transformer from scratch, train it on text
  data, and generate coherent text with various sampling strategies.
difficulty: expert
estimated_hours: "50-80"
essence: >-
  Decoder-only transformer with causal self-attention computing query-key-value
  transformations, multi-head parallelization, learned positional embeddings,
  and autoregressive next-token prediction trained with cross-entropy loss
  and optimized with KV caching for efficient generation.
why_important: >-
  GPT-style transformers power the most capable language models (GPT-4, Claude,
  Llama). Building one from scratch teaches the exact architecture, training
  procedure, and inference optimizations used in production LLMs—essential
  knowledge for any ML engineer working with generative AI.
learning_outcomes:
  - Implement causal self-attention with proper masking for autoregressive models
  - Build multi-head attention with parallel head computation
  - Design transformer blocks with FFN, layer normalization, and residual connections
  - Implement Pre-LN architecture for training stability
  - Train a language model with next-token prediction and learning rate warmup
  - Implement autoregressive generation with temperature, top-k, and top-p sampling
  - Optimize inference with KV caching for efficient generation
  - Compare outputs against minGPT or PyTorch reference implementations
skills:
  - Causal Self-Attention
  - Multi-Head Attention
  - Neural Architecture Design
  - Autoregressive Generation
  - KV Caching
  - PyTorch Implementation
  - Language Modeling
  - Sampling Strategies
tags:
  - ai-ml
  - attention
  - build-from-scratch
  - expert
  - gpt
  - nlp
  - python
  - language-modeling
architecture_doc: architecture-docs/build-transformer/index.md
languages:
  recommended:
    - Python
  also_possible:
    - Julia
    - Rust
resources:
  - type: paper
    name: Attention Is All You Need
    url: https://arxiv.org/abs/1706.03762
  - type: paper
    name: Language Models are Unsupervised Multitask Learners (GPT-2)
    url: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
  - type: video
    name: Let's build GPT by Karpathy
    url: https://www.youtube.com/watch?v=kCc8FmEb1nY
  - type: repository
    name: minGPT
    url: https://github.com/karpathy/minGPT
  - type: repository
    name: nanoGPT
    url: https://github.com/karpathy/nanoGPT
prerequisites:
  - type: skill
    name: Neural networks (backpropagation, loss functions)
  - type: skill
    name: Attention mechanism basics
  - type: skill
    name: PyTorch (nn.Module, autograd, training loops)
  - type: skill
    name: NLP fundamentals (tokenization, language modeling)
milestones:
  - id: build-transformer-m1
    name: Causal Self-Attention & Multi-Head Attention
    description: >-
      Implement causal (masked) self-attention and multi-head attention—the
      core computation of GPT. This is decoder-only: every position can only
      attend to itself and earlier positions.
    acceptance_criteria:
      - Q, K, V projections from input via learned linear layers produce matrices with correct dimensions
      - Attention scores scaled by 1/sqrt(d_k) before softmax to prevent gradient saturation
      - Causal mask applies -inf to all positions above the diagonal before softmax, ensuring each token only attends to past and present
      - Multi-head attention splits d_model into h heads (d_k = d_model / h), computes attention per head in a single batched operation
      - Head outputs concatenated and projected through W_O output layer; final shape matches input [batch, seq, d_model]
      - Numerical verification: output matches PyTorch's nn.MultiheadAttention (with causal mask) within 1e-4 tolerance on random inputs
      - Attention weights visualizable: extract and plot attention patterns for a sample input sequence
    pitfalls:
      - Applying causal mask AFTER softmax produces incorrect attention (positions attend to future)
      - Dimension transposition errors between [batch, heads, seq, d_k] and [batch, seq, heads, d_k] are the #1 bug
      - Forgetting .contiguous() before .view() causes runtime errors on non-contiguous tensors
      - Not scaling by sqrt(d_k) causes softmax to produce near-one-hot distributions with vanishing gradients
    concepts:
      - Causal (autoregressive) self-attention
      - Multi-head attention
      - Scaled dot-product attention
      - Attention masking
    skills:
      - Matrix multiplication and broadcasting
      - Tensor reshaping for multi-head computation
      - Softmax numerical stability
      - Reference implementation verification
    deliverables:
      - Q, K, V projection layers
      - Scaled dot-product attention with causal masking
      - Multi-head attention splitting, computing, concatenating, and projecting
      - Attention weight extraction for visualization
      - Verification script comparing against PyTorch nn.MultiheadAttention
    estimated_hours: "10-15"

  - id: build-transformer-m2
    name: Transformer Block & Embeddings
    description: >-
      Build a complete GPT transformer block (attention + FFN + LayerNorm +
      residuals) and the embedding layer (token + position).
    acceptance_criteria:
      - Pre-LN architecture: LayerNorm applied BEFORE each sub-layer (attention and FFN), not after (document why: Pre-LN is more training-stable for GPT-style models, used in GPT-2+)
      - Feed-forward network expands to 4x d_model dimension, applies GELU activation, and projects back
      - Residual connections add sub-layer input to its output (x + sublayer(LN(x)) for Pre-LN)
      - Dropout applied after attention weights, after FFN, and after residual addition
      - Token embedding maps vocabulary indices to d_model-dimensional learned vectors
      - Position embedding: learned positional embeddings for positions 0 to max_seq_len-1 (GPT-2 style, not sinusoidal)
      - Combined embedding: token_embed + position_embed with dropout
      - N blocks stacked sequentially (N configurable, default 6); gradient flows through all blocks verified
    pitfalls:
      - Pre-LN vs Post-LN: Post-LN (original paper) is less stable for deep models; GPT-2 uses Pre-LN for this reason
      - Missing residual connections cause vanishing gradients in deep stacks; training loss plateaus
      - Wrong FFN expansion ratio (should be 4x d_model for standard GPT)
      - Using sinusoidal instead of learned positional embeddings for GPT (sinusoidal is for the original encoder-decoder Transformer)
      - Dropout rate too high (>0.2) causes underfitting; too low (<0.05) causes overfitting on small data
    concepts:
      - Pre-LN transformer block
      - GELU activation
      - Residual connections for gradient flow
      - Learned positional embeddings
    skills:
      - Layer normalization implementation
      - Residual connection patterns
      - Embedding layer design
      - Module stacking and composition
    deliverables:
      - TransformerBlock with Pre-LN attention and FFN sub-layers with residual connections
      - Feed-forward network with 4x expansion and GELU activation
      - Token embedding layer
      - Learned positional embedding layer
      - Combined embedding (token + position + dropout)
      - GPT model class stacking N transformer blocks
      - Gradient flow verification script
    estimated_hours: "12-18"

  - id: build-transformer-m3
    name: Training Pipeline
    description: >-
      Implement the training pipeline with tokenization, data loading,
      cross-entropy loss, learning rate warmup, and convergence monitoring.
    acceptance_criteria:
      - Tokenizer: character-level (simple) or BPE (using tiktoken or sentencepiece) converting text to integer token IDs
      - Data loader yields batches of [batch_size, seq_len] token sequences with targets shifted by one position (input[t] → target[t+1])
      - Output projection: linear layer mapping d_model → vocab_size logits; optionally weight-tied with token embedding
      - Cross-entropy loss computed over logits vs target tokens, averaged over non-padding positions
      - Learning rate warmup for first N steps (default 10% of total) followed by cosine decay to 10% of peak LR
      - Gradient clipping (max norm 1.0) applied every step
      - Training loss decreases consistently; on Shakespeare or similar small corpus, loss reaches <1.5 within 5000 steps
      - Validation loss evaluated every N steps; training stopped if overfitting detected
      - Checkpoint saves model weights and optimizer state at configurable intervals
    pitfalls:
      - Target labels must be shifted: input tokens [0:T-1] predict targets [1:T]. Getting the offset wrong trains on incorrect supervision
      - Without learning rate warmup, initial gradient updates are too large and destabilize training
      - Without gradient clipping, loss spikes and NaN gradients occur in early training
      - Weight tying (sharing embedding and output projection weights) reduces parameters but requires careful initialization
      - Training on very small datasets without dropout/regularization causes rapid overfitting
    concepts:
      - Next-token prediction objective
      - Teacher forcing with shifted targets
      - Learning rate warmup and cosine decay
      - Weight tying
    skills:
      - Tokenization (character or BPE)
      - Data loading with sequence batching
      - Cross-entropy loss for language modeling
      - Training loop with warmup, clipping, and checkpointing
    deliverables:
      - Tokenizer converting text to/from integer token sequences
      - Data loader producing batched token sequences with shifted targets
      - Output projection layer (d_model → vocab_size), optionally weight-tied
      - Cross-entropy loss function
      - Learning rate scheduler with warmup and cosine decay
      - Training loop with gradient clipping, loss logging, and checkpointing
      - Validation evaluation at configurable intervals
    estimated_hours: "12-18"

  - id: build-transformer-m4
    name: Text Generation & KV Caching
    description: >-
      Implement autoregressive text generation with multiple sampling strategies
      and KV caching for efficient inference.
    acceptance_criteria:
      - Greedy decoding selects argmax token at each step, producing deterministic output
      - Temperature parameter scales logits before softmax; temperature approaching 0 produces greedy-like output (handle division-by-zero)
      - Top-k sampling restricts candidates to k highest-probability tokens before sampling
      - Top-p (nucleus) sampling restricts candidates to the smallest set whose cumulative probability exceeds p
      - KV cache stores previously computed key-value tensors; each generation step processes only the new token
      - KV cache benchmark: generating 100 tokens is at least 3x faster with cache than without (full re-encoding)
      - Generated text from a trained model is coherent at short lengths (20-50 tokens) and stylistically matches training data
      - End-of-sequence token terminates generation; max_length parameter caps output length
      - Repetition penalty optionally reduces probability of recently generated tokens
    pitfalls:
      - Temperature of exactly 0.0 causes division by zero; use greedy decoding for temperature <= epsilon (1e-8)
      - KV cache shape mismatch when the cache doesn't grow correctly with each new token
      - Top-p sampling requires sorting probabilities, which is computationally expensive; use efficient implementation
      - Not running encoder only once in encoder-decoder models—but for GPT (decoder-only) this doesn't apply
      - Repetitive text without any sampling strategy; greedy decoding on small models produces degenerate loops
    concepts:
      - Autoregressive generation loop
      - Temperature, top-k, and top-p sampling
      - KV caching for inference efficiency
      - Repetition penalty
    skills:
      - Sampling strategy implementation
      - KV cache management across generation steps
      - Inference benchmarking
      - Text post-processing
    deliverables:
      - Greedy decoder selecting highest-probability token per step
      - Temperature scaling of logits
      - Top-k sampling implementation
      - Top-p (nucleus) sampling implementation
      - KV cache storing and reusing key-value tensors across generation steps
      - Generation benchmark comparing speed with and without KV cache
      - Text generation function producing output from a given prompt string
      - Repetition penalty (optional) reducing probability of recent tokens
    estimated_hours: "12-18"
```
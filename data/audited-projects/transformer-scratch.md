# AUDIT & FIX: transformer-scratch

## CRITIQUE
- **Logical Gap – Inference/Generation**: The project ends at 'Training' with a vague AC ('Train on translation or LM task') but never requires implementing actual inference/generation. Greedy decoding, beam search, and KV caching are critical for using the model. The deliverables in M5 mention 'Inference module' but the AC doesn't rigorously validate it.
- **High Redundancy with build-transformer**: This project and 'Build Your Own Transformer' cover nearly identical ground. This version includes encoder-decoder architecture depth, while the other focuses on GPT-style decoder-only. The distinction should be sharper.
- **M1 AC Vagueness**: 'Compute Q, K, V from input' lacks any measurable criteria. What does 'correct' mean? There should be numerical verification against a reference implementation.
- **M5 AC Weakness**: 'Train on translation or LM task' is too vague. What dataset? What loss value indicates success? What generation quality is expected?
- **Embedding Scaling Note**: The audit correctly identifies that sqrt(d_model) scaling is paper-specific and dropped in many modern implementations. The AC should note it as historically accurate but optional.
- **Missing Numerical Stability**: Softmax overflow/underflow with large attention scores is a common bug not addressed in any pitfall until you realize the 'scaling by sqrt(d_k)' IS the numerical stability fix, but the connection isn't made explicit.
- **Pre-norm vs Post-norm**: M4 AC says 'Layer normalization is applied before or after each sublayer per configuration' which is good, but the pitfall just says 'Pre-norm vs post-norm' without explaining that Pre-LN is more stable for training while Post-LN matches the original paper.
- **No Verification Strategy**: None of the milestones require comparing outputs against a reference implementation (e.g., PyTorch's nn.MultiheadAttention). This is essential for a 'from scratch' project.
- **M3 Missing Dropout**: The pitfalls mention 'Forgetting dropout' but the AC doesn't require dropout implementation. Dropout is critical for regularization in transformers.

## FIXED YAML
```yaml
id: transformer-scratch
name: Transformer from Scratch
description: >-
  Implement the complete Transformer architecture (encoder-decoder) from first
  principles, including attention, positional encodings, training, and inference.
difficulty: advanced
estimated_hours: "35-55"
essence: >-
  Scaled dot-product attention with query-key-value matrix operations, multi-head
  parallel processing, sinusoidal positional encoding, and stacked encoder-decoder
  layers with residual connections and layer normalization—trained on a
  sequence-to-sequence task and validated with autoregressive generation.
why_important: >-
  Transformers power modern LLMs (GPT, BERT, T5). Understanding the architecture
  from first principles—attention mechanisms, positional encodings, encoder-decoder
  design, and generation strategies—is essential for any ML engineer working with
  NLP, whether fine-tuning models, optimizing inference, or researching new architectures.
learning_outcomes:
  - Implement scaled dot-product attention with numerical stability considerations
  - Build multi-head attention with parallel head projections and concatenation
  - Design sinusoidal positional encodings to inject sequence order information
  - Construct encoder and decoder layers with residual connections and layer normalization
  - Implement masked self-attention for autoregressive decoding
  - Train a transformer on a sequence-to-sequence task with proper loss computation
  - Implement greedy and beam search decoding for inference
  - Verify correctness by comparing outputs against PyTorch reference implementations
skills:
  - Self-Attention Mechanisms
  - Matrix Operations & Linear Algebra
  - Neural Architecture Design
  - Sequence Modeling
  - Positional Encoding
  - Gradient Debugging
  - PyTorch Implementation
  - Autoregressive Generation
tags:
  - advanced
  - ai-ml
  - attention
  - embeddings
  - positional-encoding
  - python
  - self-attention
  - build-from-scratch
architecture_doc: architecture-docs/transformer-scratch/index.md
languages:
  recommended:
    - Python
  also_possible:
    - Julia
resources:
  - name: Attention Is All You Need Paper
    url: https://arxiv.org/abs/1706.03762
    type: paper
  - name: The Illustrated Transformer
    url: https://jalammar.github.io/illustrated-transformer/
    type: article
  - name: Annotated Transformer (Harvard NLP)
    url: https://nlp.seas.harvard.edu/annotated-transformer/
    type: tutorial
prerequisites:
  - type: skill
    name: Neural networks (backpropagation, loss functions)
  - type: skill
    name: Linear algebra (matrix multiplication, transpose, softmax)
  - type: skill
    name: Python and PyTorch basics (tensors, autograd, nn.Module)
milestones:
  - id: transformer-scratch-m1
    name: Scaled Dot-Product Attention
    description: >-
      Implement the core attention mechanism. Verify numerical correctness
      against PyTorch's built-in implementation.
    acceptance_criteria:
      - Compute Q, K, V from input embeddings via three separate learned linear projections (nn.Linear)
      - Scaled dot-product computes softmax(QK^T / sqrt(d_k))V with correct output dimensions [batch, seq_len, d_v]
      - Padding mask sets attention weights to zero for padding tokens by applying -inf before softmax
      - Causal mask prevents attending to future positions by applying -inf to upper-triangle entries before softmax
      - Implementation is fully vectorized—no Python loops over batch or sequence dimensions
      - Numerical correctness verified: output matches PyTorch's F.scaled_dot_product_attention within 1e-5 tolerance on random inputs
      - Softmax numerical stability ensured by subtracting max before exponentiation (or verifying PyTorch's softmax does this)
    pitfalls:
      - Forgetting to scale by sqrt(d_k) causes softmax to saturate, producing near-one-hot attention weights with vanishing gradients
      - Applying mask AFTER softmax instead of before produces incorrect attention distributions
      - Dimension mismatch between Q [batch, seq, d_k] and K^T [batch, d_k, seq] is the most common tensor bug
      - Not testing with actual padding tokens means the mask logic is unverified
    concepts:
      - Scaled dot-product attention
      - Query-Key-Value decomposition
      - Padding and causal masking
      - Softmax numerical stability
    skills:
      - Matrix multiplication and broadcasting
      - Softmax implementation and stability
      - Mask construction and application
      - Numerical verification against reference
    deliverables:
      - Q, K, V linear projection layers
      - Scaled dot-product attention function computing softmax(QK^T/sqrt(d_k))V
      - Padding mask builder creating masks from sequence lengths
      - Causal mask builder creating upper-triangular -inf masks
      - Verification script comparing output against PyTorch reference
    estimated_hours: "4-6"

  - id: transformer-scratch-m2
    name: Multi-Head Attention
    description: >-
      Implement multi-head attention that runs multiple attention heads
      in parallel and concatenates their outputs.
    acceptance_criteria:
      - Split d_model into h heads with d_k = d_model / h dimensions per head; assert d_model % h == 0
      - Separate learned W_Q, W_K, W_V projection matrices for each head (or equivalently one large projection reshaped)
      - All h attention heads computed in a single batched operation via reshape/transpose (no loop over heads)
      - Head outputs concatenated along feature dimension and projected through W_O output linear layer
      - Output shape matches input shape [batch, seq_len, d_model]
      - Numerical correctness verified against PyTorch's nn.MultiheadAttention within 1e-5 tolerance
    pitfalls:
      - Reshape vs view errors when tensor is not contiguous—use .contiguous() before .view()
      - Transposing the wrong dimensions (seq_len vs num_heads) is the #1 shape bug
      - Forgetting the output projection W_O after concatenation loses the ability to mix information across heads
      - Not verifying that d_model is divisible by num_heads causes silent dimension errors
    concepts:
      - Multi-head attention mechanism
      - Parallel head computation via tensor reshaping
      - Linear projections (W_Q, W_K, W_V, W_O)
    skills:
      - Tensor reshaping and dimension management
      - Efficient parallel computation via batched operations
      - Linear layer design
      - Reference verification
    deliverables:
      - Head splitting logic reshaping [batch, seq, d_model] to [batch, h, seq, d_k]
      - Batched attention computation across all heads simultaneously
      - Head concatenation merging [batch, h, seq, d_k] back to [batch, seq, d_model]
      - Output projection layer W_O
      - Verification script comparing against nn.MultiheadAttention
    estimated_hours: "4-6"

  - id: transformer-scratch-m3
    name: Feed-Forward Network, Embeddings & Positional Encoding
    description: >-
      Implement the position-wise FFN, token embeddings, sinusoidal positional
      encoding, and dropout regularization.
    acceptance_criteria:
      - Two-layer FFN with ReLU (or GELU) activation: FFN(x) = W2 * ReLU(W1 * x + b1) + b2, with configurable inner dimension (default 4 * d_model)
      - Token embedding layer maps vocabulary indices to d_model-dimensional learned vectors
      - Sinusoidal positional encoding computes PE(pos, 2i) = sin(pos/10000^(2i/d_model)) and PE(pos, 2i+1) = cos(...)
      - Positional encoding is NOT learned (registered as buffer, not parameter); precomputed for max_seq_len positions
      - Embedding scaling multiplies token embeddings by sqrt(d_model) before adding positional encoding (note: this follows the original paper; modern implementations often omit this—document the choice)
      - Dropout applied after embedding+PE combination AND after each FFN and attention sublayer
      - Dropout is disabled during evaluation (model.eval())
    pitfalls:
      - Positional encoding dimension indexing error (even=sin, odd=cos) is extremely common
      - Forgetting dropout is the #1 cause of overfitting in small transformer models
      - Embedding scale factor sqrt(d_model) is specific to the original paper; omitting it changes the effective learning rate of embeddings vs positional encodings
      - Registering positional encoding as a parameter instead of buffer causes it to be updated by the optimizer
    concepts:
      - Sinusoidal positional encoding
      - Position-wise feed-forward network
      - Token embedding with scaling
      - Dropout regularization
    skills:
      - Sinusoidal encoding implementation
      - FFN layer design
      - Embedding and buffer registration in PyTorch
      - Dropout placement strategy
    deliverables:
      - Feed-forward network with two linear layers, ReLU activation, and dropout
      - Token embedding lookup table with sqrt(d_model) scaling (documented as optional)
      - Sinusoidal positional encoding registered as non-learnable buffer
      - Combined embedding layer adding scaled token embeddings and positional encoding with dropout
    estimated_hours: "4-6"

  - id: transformer-scratch-m4
    name: Encoder & Decoder Layers
    description: >-
      Compose attention, FFN, normalization, and residual connections into
      complete encoder and decoder layers. Stack N layers into encoder and decoder.
    acceptance_criteria:
      - Encoder layer: self-attention → residual+norm → FFN → residual+norm (Post-LN, matching original paper)
      - Pre-LN variant (norm → sublayer → residual) implemented as configurable alternative, with documentation explaining Pre-LN is more training-stable for deep models
      - Decoder layer: masked self-attention → residual+norm → cross-attention (attending to encoder output) → residual+norm → FFN → residual+norm
      - Cross-attention uses decoder output as Q and encoder output as K, V
      - Causal mask in decoder self-attention prevents attending to future positions
      - N encoder layers stacked sequentially; N decoder layers stacked sequentially (N configurable, default 6)
      - Encoder output is passed to every decoder layer's cross-attention (not just the last)
      - Gradient flow verified: loss.backward() produces non-zero gradients for all parameters in all layers
    pitfalls:
      - Pre-LN vs Post-LN is a critical stability decision: Post-LN matches the paper but is harder to train deep; Pre-LN is used in GPT-2+ for stability
      - Cross-attention K and V come from encoder output, not decoder output—getting this wrong is a silent semantic error
      - Causal mask must be regenerated or sliced correctly for different sequence lengths during training vs generation
      - Not checking gradient flow through all layers misses dead layers or vanishing gradient issues
    concepts:
      - Encoder-decoder architecture
      - Pre-LN vs Post-LN normalization
      - Cross-attention mechanics
      - Residual connections for gradient flow
    skills:
      - Layer composition and stacking
      - Layer normalization integration
      - Cross-attention implementation
      - Gradient flow verification
    deliverables:
      - Encoder layer with self-attention, FFN, residual connections, and layer norm
      - Decoder layer with masked self-attention, cross-attention, FFN, residual connections, and layer norm
      - Pre-LN/Post-LN configuration switch
      - Layer stacker composing N identical layers into encoder and decoder stacks
      - Gradient flow verification script checking all parameters receive gradients
    estimated_hours: "5-8"

  - id: transformer-scratch-m5
    name: Full Transformer Assembly & Training
    description: >-
      Assemble the complete encoder-decoder transformer. Train on a small
      sequence-to-sequence task (e.g., copy task, small translation, or
      number reversal) and demonstrate decreasing loss.
    acceptance_criteria:
      - Full transformer wires encoder stack, decoder stack, source/target embeddings, and output projection
      - Output projection maps decoder output to vocabulary-sized logits via linear layer
      - Cross-entropy loss computed on decoder output vs target tokens, ignoring padding positions
      - Training on a synthetic task (e.g., sequence copy or reversal) shows loss decreasing below 0.1 within 1000 steps
      - Learning rate warmup for first N steps followed by inverse-sqrt decay (as in original paper) or cosine schedule
      - Gradient clipping (max norm 1.0) applied to prevent training instability
      - Label smoothing (epsilon=0.1) implemented as optional regularization
      - Training loop logs loss and learning rate every N steps
    pitfalls:
      - Target labels must be shifted by one position relative to decoder input (teacher forcing offset)
      - Loss must mask out padding positions or the model optimizes for predicting padding tokens
      - Learning rate warmup is critical—without it, early gradient updates are too large and destabilize training
      - Not using gradient clipping with Post-LN causes gradient explosions in early training
    concepts:
      - Full transformer assembly
      - Teacher forcing and label shifting
      - Learning rate scheduling (warmup + decay)
      - Label smoothing regularization
    skills:
      - End-to-end model assembly
      - Training loop implementation
      - Loss function design with masking
      - Learning rate scheduling
    deliverables:
      - Complete encoder-decoder transformer class
      - Output projection layer (d_model → vocab_size)
      - Masked cross-entropy loss function ignoring padding positions
      - Training loop with forward pass, loss, backward, optimizer step, and gradient clipping
      - Learning rate scheduler with warmup
      - Training script for synthetic task demonstrating convergence
    estimated_hours: "6-10"

  - id: transformer-scratch-m6
    name: Inference & Generation
    description: >-
      Implement autoregressive decoding with greedy search, beam search, and
      KV caching for efficient generation.
    acceptance_criteria:
      - Greedy decoding generates output sequence by selecting argmax token at each step until EOS or max length
      - Beam search with configurable beam width (default 4) explores multiple hypotheses and returns top-K completed sequences
      - KV cache stores previously computed key-value pairs so each generation step only processes the new token (not full sequence)
      - KV cache reduces generation time measurably: benchmark shows at least 2x speedup for 100-token generation vs naive re-encoding
      - Generated output on the trained synthetic task is correct for at least 90% of test inputs
      - Temperature parameter scales logits before softmax, with temperature=0 equivalent to greedy decoding
      - Length penalty in beam search adjustable to control output length preference
    pitfalls:
      - KV cache dimension mismatch when the cache shape doesn't match the current sequence length plus one
      - Beam search requires careful bookkeeping of partial hypotheses, their scores, and whether they've emitted EOS
      - Temperature of exactly 0 causes division by zero—use greedy decoding when temperature <= epsilon
      - Not updating the causal mask to account for growing sequence length during generation
      - Forgetting to run the encoder only once and reuse its output across all decoder steps
    concepts:
      - Autoregressive decoding
      - Beam search algorithm
      - KV caching for efficient inference
      - Temperature and sampling strategies
    skills:
      - Greedy and beam search implementation
      - KV cache management
      - Inference benchmarking
      - Sequence generation loop design
    deliverables:
      - Greedy decoder generating tokens autoregressively until EOS
      - Beam search decoder exploring multiple hypotheses with configurable beam width
      - KV cache storing and reusing key-value pairs across generation steps
      - Temperature-based sampling option
      - Inference benchmark comparing generation speed with and without KV cache
      - End-to-end demo generating outputs from the trained model
    estimated_hours: "6-10"
```
# DOMAIN PROFILE: AI & Machine Learning
# Applies to: ai-ml
# Projects: neural network, transformer, RAG, recommendation, distributed training, MoE, etc.

## Fundamental Tension Type
Mathematical and computational constraints. MODEL CAPACITY (learn complex patterns) vs GENERALIZATION (work on unseen data). Bias-variance tradeoff — every architecture decision is a specific answer.

Secondary: attention O(n²) limits context, deeper = better features but harder to train (vanishing gradients), larger batch = better GPU util but worse generalization, model size vs inference latency.

## Three-Level View
- **Level 1 — Model Architecture**: Layers, attention heads, activations — what model IS
- **Level 2 — Training Dynamics**: Loss landscape, gradient flow, optimizer behavior — how model LEARNS
- **Level 3 — Compute Infrastructure**: GPU memory, tensor/data parallelism, mixed precision — what makes it FAST

## Soul Section: "Math Soul"
- What is this layer COMPUTING? (actual matrix multiplication, not just "applies attention")
- Why this activation? (ReLU sparsity, GELU smooth gradients — when matters?)
- Gradient flow backward through this? (vanishing? exploding? normalization helps how?)
- Loss landscape shape? (convex? local minima? saddle points?)
- Dimensions at each step? (batch × seq_len × hidden_dim — trace shapes)

Include key equations in LaTeX where they clarify, ALWAYS paired with intuitive explanation.

## Alternative Reality Comparisons
PyTorch (autograd), TensorFlow/JAX (XLA), HuggingFace Transformers, GPT-2/3/4 (decoder), BERT/RoBERTa (encoder), T5 (enc-dec), Mamba/RWKV (state-space), LoRA/QLoRA, vLLM, DeepSpeed/FSDP.

## TDD Emphasis
- Tensor shapes: MANDATORY — input/output shapes with named dims per layer
- Forward pass: step-by-step with shapes (einops notation)
- Gradient flow analysis: vanish/explode paths
- Numerical stability: float32 vs float16, overflow/underflow spots
- GPU memory budget: params + activations + gradients + optimizer state
- Hyperparameters: lr, batch, warmup — with justification
- Tests: known input → expected output per layer (small dims)
- Memory layout: SKIP (tensor shapes instead)
- Cache line: SKIP
- Lock ordering: ONLY for distributed training sync
- Benchmarks: loss milestones, inference latency, tokens/sec, GPU util %

## Cross-Domain Notes
Borrow from systems-lowlevel when: CUDA kernels, memory management, GPU pipeline.
Borrow from distributed when: distributed training (all-reduce, ring topology, gradient sync).
Borrow from data-storage when: embedding storage, vector retrieval, dataset pipeline.
Borrow from compiler-language when: custom operators, graph compilation (XLA, TorchScript).

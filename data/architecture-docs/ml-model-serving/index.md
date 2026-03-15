# 🎯 Project Charter: ML Model Serving API
## What You Are Building
You will build a production-grade ML model serving API that transforms a trained PyTorch model into a high-performance inference service. This is not a toy example—it's a real system implementing the patterns used at companies serving millions of predictions daily.
The core deliverable is a FastAPI application that:
- Loads and manages ML models with version control
- Batches incoming requests dynamically for GPU efficiency
- Routes traffic between model versions for A/B testing
- Monitors prediction quality and detects data drift in real-time
- Exposes Prometheus metrics for observability
You'll implement battle-tested patterns: consistent hashing for traffic routing, reservoir sampling for drift detection, and CUDA warmup sequences that eliminate cold-start latency spikes.
## Why This Project Exists
Most ML education stops at model training. Reality check: **training is 10% of the ML engineering lifecycle**. The other 90% is serving, monitoring, and maintaining models in production.
This project exists because:
1. **The deployment gap** — You know how to train models, but serving them at scale requires entirely different skills
2. **Performance matters** — A naive serving implementation can be 10-100x slower than an optimized one
3. **Production readiness** — Real systems need versioning, rollback, A/B testing, and drift detection
The patterns you'll implement—dynamic batching, model hot-swapping, statistical drift detection—are the same ones used at Google, Meta, and Netflix. You'll understand not just *how* they work, but *why* they're designed that way.
## What You Will Be Able to Do
After completing this project, you will be able to:
- **Optimize GPU throughput** by implementing dynamic request batching that automatically groups inference calls to maximize hardware utilization
- **Deploy models safely** with version control, hot-swap capabilities, and instant rollback when issues arise
- **Run controlled experiments** using statistically rigorous A/B testing with traffic splitting and significance analysis
- **Detect production issues** through real-time monitoring that identifies data drift and model degradation using KS tests
- **Eliminate cold-start penalties** via CUDA warmup sequences and kernel pre-compilation
- **Serialize efficiently** using orjson for 2-3x faster JSON handling compared to standard library
- **Build observable systems** with Prometheus metrics, structured logging, and health endpoints
- **Design resilient architectures** using consistent hashing for stable traffic distribution
## Final Deliverable
A complete, production-ready ML serving system:
```
ml-serving-api/
├── src/
│   ├── main.py                 # FastAPI application entry point
│   ├── model_loader.py         # Model loading with CUDA warmup
│   ├── batcher.py              # Dynamic request batching engine
│   ├── version_manager.py      # Model versioning and hot-swap
│   ├── ab_router.py            # A/B testing with consistent hashing
│   ├── drift_detector.py       # Statistical drift detection
│   └── metrics.py              # Prometheus metrics integration
├── tests/
│   ├── test_model_loader.py    # TDD tests for model loading
│   ├── test_batcher.py         # TDD tests for batching logic
│   ├── test_versioning.py      # TDD tests for version management
│   ├── test_ab_router.py       # TDD tests for traffic routing
│   └── test_drift.py           # TDD tests for drift detection
├── models/                     # Model artifacts directory
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Multi-service orchestration
├── prometheus.yml              # Metrics collection config
└── requirements.txt            # Python dependencies
```
The system will pass a comprehensive test suite covering unit tests, integration tests, and performance benchmarks.
## Prerequisites
Before starting, you should be comfortable with:
| Skill | Minimum Level | How to Verify |
|-------|---------------|---------------|
| **REST API Design** | Intermediate | You can design CRUD endpoints, understand HTTP status codes, and know when to use POST vs PUT |
| **Python Async Programming** | Intermediate | You understand `async/await`, event loops, and why blocking calls are problematic |
| **Basic ML/DL** | Foundational | You know what a model inference is, understand tensors at a high level |
| **Docker** | Foundational | You can write a basic Dockerfile and run containers |
| **Statistics** | Foundational | You understand hypothesis testing and p-values at a conceptual level |
| **PyTorch** | Foundational | You can load a saved model and run inference |
If any of these feel shaky, spend 1-2 hours reviewing before diving in. The project will be significantly more valuable if you're not fighting the basics.
## Estimated Effort
| Milestone | Focus Area | Estimated Hours | Complexity |
|-----------|------------|-----------------|------------|
| **M1** | Model Loading, Inference & Serialization | 9-12 | Foundation |
| **M2** | Request Batching | 9-12 | High |
| **M3** | Model Versioning & Hot Reload | 9-11 | Medium |
| **M4** | A/B Testing & Canary Deployment | 9-11 | Medium-High |
| **M5** | Monitoring & Drift Detection | 10-14 | High |
| **Integration & Polish** | End-to-end testing, documentation | 5-8 | Medium |
| **Total** | | **51-68 hours** | |
**Pacing recommendation**: Plan for 2-3 weeks if dedicating 15-20 hours per week, or 4-6 weeks at a more relaxed 10-12 hours per week. Milestones M2 and M5 are the most challenging—budget extra time for deep learning.
## Definition of Done
This project is complete when:
### Functional Requirements
- [ ] All 5 milestones pass their acceptance criteria tests
- [ ] The API handles 1000+ requests/second with dynamic batching enabled
- [ ] Model hot-swap completes in under 500ms without dropping requests
- [ ] A/B traffic splits are accurate within 1% of configured ratios
- [ ] Drift detection identifies distribution shifts with >95% recall on test data
### Technical Requirements
- [ ] All code follows TDD methodology with tests written before implementation
- [ ] Test coverage exceeds 85% for all modules
- [ ] Prometheus metrics are exposed and queryable
- [ ] Docker container builds and runs successfully
- [ ] API documentation is complete (OpenAPI/Swagger)
### Quality Requirements
- [ ] No hardcoded values—configuration is externalized
- [ ] Error handling covers edge cases (model not found, GPU OOM, invalid input)
- [ ] Logging is structured and includes trace IDs for debugging
- [ ] Code is readable with docstrings for all public functions
### Verification
The final test suite includes:
- Unit tests for each component
- Integration tests for end-to-end flows
- Performance benchmarks validating throughput claims
- Chaos tests verifying graceful degradation

---

# 📚 Before You Read This: Prerequisites & Further Reading
> **Read these first.** The Atlas assumes you are familiar with the foundations below.
> Resources are ordered by when you should encounter them — some before you start, some at specific milestones.
---
## Before Starting: Essential Foundations
### 1. CUDA Programming Model & Memory Hierarchy
| | |
|---|---|
| **📄 Paper** | NVIDIA. *CUDA C Programming Guide*, Chapter 5: Memory Hierarchy. NVIDIA Documentation. |
| **📋 Spec** | CUDA Toolkit Documentation, v12.x — Memory Model Specification |
| **💻 Code** | `cuda/samples/1_Utilities/bandwidthTest/bandwidthTest.cu` — Lines 150-220 (memory transfer patterns) |
| **🎥 Best Explanation** | Mark Harris, "Optimizing CUDA for GPUs" (GTC 2012), timestamp 14:30-28:00 — Memory coalescing explained |
| **⭐ Why Gold Standard** | Official NVIDIA reference; defines the exact memory model your M1 GPU memory pool must respect. |
| **📅 Read When** | **Before M1.** You cannot design a GPU memory pool without understanding device/host memory boundaries, contexts, and lifetimes. |
---
### 2. Python Concurrency: async/await and the Event Loop
| | |
|---|---|
| **📄 Paper** | Nathaniel J. Smith, * trio: Async I/O for Humans*. 2018. (Design rationale section) |
| **📋 Spec** | PEP 492 — Coroutines with async and await syntax (Python 3.5+) |
| **💻 Code** | `python/cpython/Lib/asyncio/tasks.py`, lines 330-410 — `gather()` implementation showing concurrent task coordination |
| **🎥 Best Explanation** | David Beazley, "Fear and Awe in Python" (PyCon 2017), 45:00-1:05:00 — Event loop internals visualized |
| **⭐ Why Gold Standard** | Beazley's live-coding of event loops is the clearest mental model for understanding backpressure and queue saturation (M2/M3). |
| **📅 Read When** | **Before M1.** Your entire serving layer is async; misunderstandings here compound through M3. |
---
### 3. Latency Percentiles and Histograms
| | |
|---|---|
| **📄 Paper** | Gil Tene, "How NOT to Measure Latency" (Strange Loop 2015). Slides pp. 12-45. |
| **📋 Spec** | HdrHistogram specification — https://hdrhistogram.github.io/HdrHistogram/ |
| **💻 Code** | `HdrHistogram/HdrHistogram_c/src/hdr_histogram.c`, lines 200-280 — percentile calculation from bucketed counts |
| **🎥 Best Explanation** | Gil Tene's talk, timestamp 08:00-22:00 — Why p99 ≠ "slowest 1%" and why averaging latency is misleading |
| **⭐ Why Gold Standard** | Your M2 metrics require p50/p95/p99; Tene explains why naive implementations lie. HdrHistogram is the industry standard data structure. |
| **📅 Read When** | **Before M2.** You'll implement latency histograms in M2; read this first to avoid common pitfalls. |
---
## After M1: Deepening GPU Foundations
### 4. CUDA Contexts and Multi-Process Service (MPS)
| | |
|---|---|
| **📄 Paper** | NVIDIA. *Multi-Process Service*, CUDA Toolkit Documentation, Chapter 3. |
| **📋 Spec** | CUDA Driver API, `cuCtxCreate` / `cuCtxPushCurrent` — Context management semantics |
| **💻 Code** | `NVIDIA/cuda-samples/Samples/0_Introduction/simpleCUDAContext/simpleCUDAContext.cu`, lines 85-130 |
| **🎥 Best Explanation** | NVIDIA GTC 2020, "GPU Virtualization and MPS Deep Dive", 20:00-35:00 |
| **⭐ Why Gold Standard** | Directly addresses the M1 challenge of sharing GPU across multiple model instances without context-switch overhead. |
| **📅 Read When** | **After M1, before M2.** You've built a basic pool; now understand how to scale it for concurrent requests. |
---
### 5. PyCUDA and GPU Memory Pools
| | |
|---|---|
| **📄 Paper** | Andreas Klöckner, *PyCUDA: GPU Scripting with Python*. 2012. |
| **📋 Spec** | PyCUDA Documentation — `pycuda.tools.DeviceMemoryPool` API |
| **💻 Code** | `inducer/pycuda/src/pycuda/tools.py`, lines 450-530 — `DeviceMemoryPool.allocate()` with caching logic |
| **🎥 Best Explanation** | Klöckner's PyCUDA tutorial, Section 3: Memory Management — shows allocation pooling patterns |
| **⭐ Why Gold Standard** | Reference implementation of exactly what M1 asks you to build; study the caching strategy. |
| **📅 Read When** | **After M1.** Compare your pool implementation against PyCUDA's proven design. |
---
## After M2: Batching and Throughput
### 6. Dynamic Batching for Inference
| | |
|---|---|
| **📄 Paper** | Olston et al., "TensorFlow-Serving: Flexible, High-Performance ML Serving" (MLSys 2018). §3.2: Batching. |
| **📋 Spec** | TensorFlow Serving Architecture Documentation — BatchingSession interface |
| **💻 Code** | `tensorflow/serving/tensorflow_serving/batching/batching_session.cc`, lines 180-260 — batch assembly timeout logic |
| **🎥 Best Explanation** | TensorFlow Serving paper walkthrough by authors, 12:00-18:00 (MLSys talk) |
| **⭐ Why Gold Standard** | The original paper on dynamic request batching; your M2 batching strategy derives from this. |
| **📅 Read When** | **During M2.** Read §3.2 before implementing your batcher; the timeout/padding tradeoffs are directly applicable. |
---
### 7. Backpressure and Reactive Streams
| | |
|---|---|
| **📄 Paper** | Wadler et al., "Contracts for Reactive Streams" (ICFP 2015). |
| **📋 Spec** | Reactive Streams Specification v1.0.4 — https://www.reactive-streams.org/ |
| **💻 Code** | `reactive-streams/reactive-streams-jvm/api/src/main/java/org/reactivestreams/Subscription.java` — `request(n)` backpressure signal |
| **🎥 Best Explanation** | Roland Kuhn, "Reactive Streams: The What and Why" (Strange Loop 2014), 15:00-30:00 |
| **⭐ Why Gold Standard** | Formal specification for exactly the async queue with backpressure pattern M2/M3 require. |
| **📅 Read When** | **After M2.** Your queue works; now understand the broader pattern and why bounded queues prevent cascade failures. |
---
## After M3: Concurrency Primitives
### 8. Lock-Free Data Structures and Atomics
| | |
|---|---|
| **📄 Paper** | Herlihy & Shavit, *The Art of Multiprocessor Programming*, Chapter 5: Concurrent Counters. |
| **📋 Spec** | C++11 std::atomic — memory_order semantics; compare_exchange_strong |
| **💻 Code** | `facebook/folly/folly/concurrency/AtomicSharedPtr.h`, lines 120-180 — lock-free shared pointer using CAS |
| **🎥 Best Explanation** | Herb Sutter, "Atomic Weapons" (CppCon 2014), Part 1, 45:00-1:10:00 — memory ordering explained |
| **⭐ Why Gold Standard** | Herlihy's textbook is the definitive reference; Sutter's talk makes memory ordering click. Essential for M3's concurrent request state. |
| **📅 Read When** | **During M3.** Read before implementing atomic counters and compare-and-swap patterns in your request state machine. |
---
### 9. Python's GIL and Concurrent Performance
| | |
|---|---|
| **📄 Paper** | Beazley, "Understanding the Python GIL" (PyCon 2015). |
| **📋 Spec** | CPython Source, `Python/ceval_gil.c` — GIL implementation (100 lines) |
| **💻 Code** | `python/cpython/Python/ceval_gil.c`, lines 120-180 — `drop_gil()` / `take_gil()` |
| **🎥 Best Explanation** | David Beazley's "Inside the GIL" talk, full duration — visualizes check intervals and thread switching |
| **⭐ Why Gold Standard** | Beazley's experiments are canonical; understand why your M3 async approach avoids GIL contention where threading would fail. |
| **📅 Read When** | **After M3.** Validates your async-over-threading decision and explains performance observations. |
---
## After M4: Routing and Statistical Testing
### 10. Consistent Hashing
| | |
|---|---|
| **📄 Paper** | Karger et al., "Consistent Hashing and Random Trees" (STOC 1997). §4: The Distributed Cache. |
| **📋 Spec** | RFC 2544 (not directly hashing, but relevant to cache benchmarking); Dynamo paper implementation |
| **💻 Code** | `spotify/dynamodb-localconsistenthashing/ring.go`, lines 45-90 — ring construction and node lookup |
| **🎥 Best Explanation** | Tom White, "Consistent Hashing" blog post (tom-ewhite.com), diagrams 2-4 — clearest visual explanation |
| **⭐ Why Gold Standard** | The original 1997 paper introduced the technique; White's diagrams make the virtual node concept intuitive. |
| **📅 Read When** | **Before M4.** Essential for understanding why consistent hashing enables smooth canary rollouts without request redistribution. |
---
### 11. Statistical Hypothesis Testing
| | |
|---|---|
| **📄 Paper** | Student (Gosset), "The Probable Error of a Mean" (Biometrika, 1908). |
| **📋 Spec** | NIST Engineering Statistics Handbook, Section 7.3 — Two-Sample t-Test |
| **💻 Code** | `scipy/scipy/scipy/stats/_stats_py.py`, lines 6200-6280 — `ttest_ind()` implementation |
| **🎥 Best Explanation** | StatQuest (Josh Starmer), "t-Test Clearly Explained", 04:00-12:00 — intuition for pooled variance |
| **⭐ Why Gold Standard** | Gosset's original paper is surprisingly readable; StatQuest provides the visual intuition your M4 A/B tests need. |
| **📅 Read When** | **During M4.** Read before implementing `ABTestAnalyzer` — understand Type I/II errors and sample size requirements. |
---
### 12. Proportion Tests and Confidence Intervals
| | |
|---|---|
| **📄 Paper** | Wilson, "Probable Inference, the Law of Succession, and Statistical Inference" (JASA, 1927). |
| **📋 Spec** | NIST Handbook 7.2.2 — Comparing Two Proportions |
| **💻 Code** | `statsmodels/statsmodels/stats/proportion.py`, lines 90-140 — `proportions_ztest()` |
| **🎥 Best Explanation** | Brown et al., "Interval Estimation for a Binomial Proportion" (Statistical Science, 2001) — why Wilson > Wald |
| **⭐ Why Gold Standard** | Wilson score intervals are the correct way to compute error rates (M4); the 2001 paper explains why naive intervals fail. |
| **📅 Read When** | **During M4.** Before implementing error-rate comparison in your A/B analyzer. |
---
## After M5: Drift Detection
### 13. Kolmogorov-Smirnov Test
| | |
|---|---|
| **📄 Paper** | Kolmogorov, "Sulla determinazione empirica di una legge di distribuzione" (1933). |
| **📋 Spec** | NIST Engineering Statistics Handbook, Section 1.3.5.17 — Kolmogorov-Smirnov Goodness-of-Fit |
| **💻 Code** | `scipy/scipy/scipy/stats/_stats_py.py`, lines 7400-7480 — `ks_2samp()` two-sample KS test |
| **🎥 Best Explanation** | Khan Academy, "Kolmogorov-Smirnov Test", full video — visual CDF comparison |
| **⭐ Why Gold Standard** | The original derivation plus Khan Academy's CDF animations make the statistic intuitive. Directly applicable to M5 drift detection. |
| **📅 Read When** | **Before M5.** Understand how KS statistic measures distribution shift before implementing `DriftDetector`. |
---
### 14. Reservoir Sampling
| | |
|---|---|
| **📄 Paper** | Vitter, "Random Sampling with a Reservoir" (ACM TOMS, 1985). Algorithm R. |
| **📋 Spec** | — |
| **💻 Code** | `python/cpython/Lib/random.py`, lines 380-410 — `sample()` implementation (uses reservoir for large populations) |
| **🎥 Best Explanation** | "Reservoir Sampling" by Keith Schwarz (Stanford CS166), lecture notes §2 — proof of uniformity |
| **⭐ Why Gold Standard** | Vitter's Algorithm R is the canonical reference; your M5 sliding window reservoir derives directly from this. |
| **📅 Read When** | **During M5.** Implement after reading Vitter; the proof ensures your reservoir maintains uniform probability. |
---
### 15. Model Drift Detection in Production
| | |
|---|---|
| **📄 Paper** | Webb et al., "Monitoring ML Models in Production" (MLConf 2016). |
| **📋 Spec** | — |
| **💻 Code** | `evidentlyai/evidently/evidently/calculations/stattests/ks.py`, lines 30-70 — production KS drift detector |
| **🎥 Best Explanation** | Evidently AI documentation, "Data Drift Detection Methods" — practical comparison of detection strategies |
| **⭐ Why Gold Standard** | Evidently is the leading open-source drift detection library; study their implementation before building M5. |
| **📅 Read When** | **Before M5.** Survey existing solutions before designing your own `DriftDetector` interface. |
---
### 16. Canary Deployment Patterns
| | |
|---|---|
| **📄 Paper** | Bass et al., *DevOps: A Software Architect's Perspective*, Chapter 8: Deployment Patterns. |
| **📋 Spec** | Kubernetes Deployment Strategy Documentation — Canary pattern |
| **💻 Code** | `kubernetes/website/content/en/docs/concepts/cluster-administration/manage-deployment.md`, Canary section |
| **🎥 Best Explanation** | Google SRE Book, Chapter 16: "Handling Overload" — traffic shifting and rollback triggers |
| **⭐ Why Gold Standard** | Google SRE practices are industry canonical; the overload chapter connects directly to M4/M5 canary + monitoring integration. |
| **📅 Read When** | **After M4.** You've implemented consistent hashing; now understand the broader deployment pattern it enables. |
---
## Reading Order Summary
| Phase | Resources | Purpose |
|-------|-----------|---------|
| **Pre-M1** | #1, #2, #3 | GPU memory, async fundamentals, latency measurement |
| **M1 → M2** | #4, #5 | Deepen GPU knowledge, compare pool implementations |
| **During M2** | #6, #7 | Batching strategies, backpressure patterns |
| **During M3** | #8, #9 | Atomics, Python concurrency realities |
| **During M4** | #10, #11, #12 | Routing, hypothesis testing, proportion tests |
| **M4 → M5** | #16 | Canary deployment context |
| **During M5** | #13, #14, #15 | KS test, reservoir sampling, drift detection |
---
## Quick Reference: Foundational Equations
### Wilson Score Interval (M4 error rate bounds)
```
p̂ ± z * √(p̂(1-p̂)/n + z²/(4n²)) / (1 + z²/n)
```
*Use for confidence intervals on error rates; more accurate than Wald for small samples.*
### KS Statistic (M5 drift detection)
```
D = max|F₁(x) - F₂(x)|
```
*Maximum vertical distance between empirical CDFs; p-value from Kolmogorov distribution.*
### Reservoir Sampling Probability (M5)
```
P(item i in reservoir at step n) = k/n for all i ≤ n
```
*Each item has equal probability k/n of being sampled; maintains uniformity over unbounded stream.*

---

# ML Model Serving API

Build a production-grade model inference service that transforms a trained ML model into a reliable, scalable, and observable API. This project teaches the critical infrastructure layer between data science experimentation and real-world ML deployment—where models meet users, traffic, and the unpredictable nature of production data.

You'll implement dynamic request batching that groups individual inference requests to maximize GPU utilization, multi-version model deployment with zero-downtime hot swapping, statistically rigorous A/B testing infrastructure, and real-time monitoring with drift detection to catch model degradation before it impacts users. These are the systems that power recommendation engines, fraud detection, and content moderation at companies like Netflix, Stripe, and Meta.


<!-- MS_ID: ml-model-serving-m1 -->
# Model Loading, Inference & Serialization
## The Hidden Complexity of "Just Loading a Model"
You've trained a model. It works beautifully in your Jupyter notebook—99.2% validation accuracy, clean inference, and predictions that make sense. Now you need to serve it.
```python
# The naive approach
model = torch.load("model.pt")
app.run()  # Ship it!
```
This works. For about thirty seconds in production.
Then reality arrives: the first request takes 800ms instead of 50ms because CUDA kernels weren't compiled. Your GPU runs out of memory because you didn't check available VRAM before loading. BatchNorm layers behave erratically because the model is still in training mode. Kubernetes kills your pod because the health check times out during model loading. And that 50ms latency target? It's actually 120ms because you forgot `torch.no_grad()` and PyTorch is silently building computation graphs you'll never use.

![FastAPI Lifespan Handler Sequence](./diagrams/tdd-diag-007.svg)

![Model Loading Lifecycle](./diagrams/diag-m1-model-loader-flow.svg)

Production model loading is not one thing—it's a **lifecycle** with distinct phases, each with specific failure modes. Let's understand why each phase exists, then build them correctly.
---
## The Fundamental Tension: Inference Latency vs. Resource Constraints
Every decision in model serving traces back to a core tension:
**You want low latency (fast predictions) and high throughput (many predictions per second), but you're constrained by:**
1. **GPU memory bandwidth**: Moving tensors from CPU to GPU costs time
2. **CUDA kernel compilation**: First use triggers JIT compilation—expensive
3. **Model memory footprint**: The model must fit in VRAM alongside activations
4. **Serialization overhead**: Converting Python objects to/from JSON isn't free
The constraints are physical. A typical NVIDIA T4 GPU has ~320 GB/s memory bandwidth. A ResNet-50 model is ~100MB. Loading it into GPU memory takes ~0.3ms *if everything is optimal*. But kernel compilation? That's 50-500ms per kernel. And a transformer might trigger dozens of kernels on first inference.
This tension explains everything that follows:
- **Warmup** exists because kernel compilation on first request violates latency SLAs
- **Device detection** exists because CPU fallback must be graceful, not a crash
- **Efficient serialization** exists because JSON overhead compounds at scale
- **Readiness probes** exist because a "healthy" pod that can't serve predictions is worse than no pod
---
## Phase 1: Device Detection and Memory Budgeting
### The Three-Level View of GPU Memory

![Tensor Shape Validation Flow](./diagrams/tdd-diag-004.svg)

> **🔑 Foundation: GPU memory model and CUDA context**
> 
> ## GPU Memory Model and CUDA Context
### What It IS
The GPU memory model describes how data is organized and accessed across different memory spaces on a GPU. Unlike CPUs with relatively flat memory hierarchies, GPUs have a distinct, multi-level memory architecture:
**Global Memory**: The largest but slowest memory space (VRAM). All threads can access it, but with high latency (~400-800 cycles). Think of it as the GPU's "main RAM."
**Shared Memory**: Fast, user-managed cache (~20-30x faster than global memory). Limited in size (typically 48KB-164KB per block). Shared among threads in a single block. This is your primary optimization lever.
**Registers**: Fastest memory, private to each thread. Limited quantity—running out "spills" to local memory (which is actually global memory, hurting performance).
**Constant Memory**: Read-only, cached, optimized for broadcast (all threads read same address).
**Texture/L1/L2 Caches**: Hardware-managed caches that can accelerate access patterns.
A **CUDA Context** is analogous to a CPU process. It encapsulates:
- All memory allocations
- Loaded modules (compiled CUDA code)
- Streams and events
- Kernel launch state
Each context has its own isolated address space. On NVIDIA GPUs, one context typically maps to one process—contexts cannot share memory directly without explicit inter-process communication.
### WHY You Need It Right Now
Understanding this model is essential for:
1. **Avoiding common performance pitfalls**: Naive kernels that thrash global memory run 10-100x slower than optimized ones. Knowing when to use shared memory vs. global memory is the difference between a slow and fast implementation.
2. **Debugging memory errors**: Segfaults, illegal memory accesses, and out-of-memory errors all trace back to this model. If you don't understand the boundaries, you can't diagnose the failures.
3. **Estimating feasibility**: Before writing code, you need to answer: "Will my data fit in shared memory?" "How many threads can I run given register pressure?"
4. **Multi-GPU programming**: Each GPU has its own context. Moving data between GPUs requires explicit transfers through the host or peer-to-peer mechanisms.
### ONE Key Insight
**Think of the GPU memory hierarchy as a pyramid with a "manual transmission."**
On a CPU, caches are automatic—you just access memory, and hardware figures out caching. On a GPU, you must *explicitly* manage the fast memory (shared memory). The hardware gives you the tools, but you're responsible for using them.
**The mental model**: Global memory is like a warehouse (huge, slow to traverse). Shared memory is like your workbench (small, but everything's within arm's reach). An efficient GPU kernel is one that loads data from the warehouse to the workbench, does as much work as possible, then writes results back—minimizing warehouse trips.
**Practical rule of thumb**: If you're reading the same global memory location more than once within a thread block, you should probably load it into shared memory first. This pattern—load once, reuse many times—is the core of GPU memory optimization.

![InputSchema and Validator Architecture](./diagrams/tdd-diag-008.svg)

Before loading anything, you must answer: *where will this model live, and will it fit?*
```python
import torch
def get_device() -> torch.device:
    """
    Select the best available device with explicit fallback.
    Returns:
        torch.device: CUDA device if available, CPU otherwise
    Note: This function logs the selection for observability.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Log the specific GPU being used
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Device] Using CUDA: {gpu_name} ({gpu_memory:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("[Device] CUDA not available, using CPU")
    return device
```
But detecting the device is only step one. You also need to verify the model will fit:
```python
def check_memory_budget(model_path: str, device: torch.device, safety_margin: float = 0.3) -> None:
    """
    Verify sufficient memory exists before loading the model.
    Args:
        model_path: Path to the model file
        device: Target device for loading
        safety_margin: Fraction of memory to reserve (default 30%)
    Raises:
        RuntimeError: If insufficient memory is available
    """
    import os
    model_size_bytes = os.path.getsize(model_path)
    if device.type == "cuda":
        # Get current memory stats
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved(0)
        allocated = torch.cuda.memory_allocated(0)
        free_memory = total_memory - reserved
        # Account for:
        # 1. Model parameters
        # 2. Activations during inference (can be 2-4x model size)
        # 3. CUDA context overhead (~500MB)
        required_memory = model_size_bytes * 4  # Conservative estimate
        available_after_load = free_memory - required_memory
        min_required = total_memory * safety_margin
        if available_after_load < min_required:
            raise RuntimeError(
                f"Insufficient GPU memory. Model requires ~{required_memory/1e9:.2f} GB, "
                f"but only {free_memory/1e9:.2f} GB free. "
                f"Need {min_required/1e9:.2f} GB safety margin."
            )
        print(f"[Memory] Model: {model_size_bytes/1e6:.1f} MB, "
              f"Available: {free_memory/1e9:.2f} GB, "
              f"Post-load margin: {available_after_load/1e9:.2f} GB")
```
**Why the 4x multiplier?** During inference, PyTorch allocates intermediate tensors (activations) that flow through the network. For a transformer with 12 layers, each layer's output must exist in memory simultaneously during the forward pass. A 100MB model might need 400MB of activation memory at peak.
**Why the safety margin?** CUDA operations aren't instantaneous. During batch inference, memory allocation and deallocation happens rapidly. Without a buffer, you risk OOM errors during traffic spikes.
---
## Phase 2: Model Loading with Correct Mode
### The eval() / no_grad() Combo
Neural networks have two modes: **training** and **evaluation**. Layers like Dropout and BatchNorm behave differently in each:
| Layer | Training Mode | Eval Mode |
|-------|---------------|-----------|
| **Dropout** | Randomly zeros activations (p=0.1-0.5) | Passes all activations through |
| **BatchNorm** | Updates running mean/var from batch | Uses fixed running mean/var |
If you forget `model.eval()`, predictions become **non-deterministic** (Dropout) or **statistically wrong** (BatchNorm uses wrong normalization).
```python
def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load a PyTorch model with production-ready configuration.
    Args:
        model_path: Path to saved model (state_dict or full model)
        device: Target device for inference
    Returns:
        Model in eval mode, on correct device, with gradients disabled
    """
    # Load the model
    model = torch.load(model_path, map_location=device)
    # CRITICAL: Set to evaluation mode
    # This affects Dropout, BatchNorm, and other stateful layers
    model.eval()
    # Move to device (idempotent if already on device)
    model = model.to(device)
    print(f"[Model] Loaded {model_path} to {device}")
    return model
```
### The Hidden Cost of Forgetting no_grad()
Here's what happens during a normal forward pass with gradients enabled:
1. PyTorch creates **shadow tensors** to track operations for backpropagation
2. A **computation graph** is built in memory, storing every operation
3. Intermediate activations are **pinned in memory** (can't be freed until backward pass)
For inference, you never call `backward()`. All that machinery is wasted:
```python
import time
def benchmark_gradient_overhead(model, input_tensor, n_runs=100):
    """Measure the overhead of gradient tracking during inference."""
    # With gradients (WRONG for inference)
    model.eval()  # But forgot no_grad!
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = model(input_tensor)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_with_grad = (time.perf_counter() - start) / n_runs * 1000
    # Without gradients (CORRECT for inference)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(input_tensor)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_without_grad = (time.perf_counter() - start) / n_runs * 1000
    overhead = ((time_with_grad - time_without_grad) / time_without_grad) * 100
    print(f"[Benchmark] With gradients: {time_with_grad:.2f}ms")
    print(f"[Benchmark] Without gradients: {time_without_grad:.2f}ms")
    print(f"[Benchmark] Gradient overhead: {overhead:.1f}%")
    return time_with_grad, time_without_grad
```
Typical results on a ResNet-50 with batch size 1:
- With gradients: **52ms**
- Without gradients: **34ms**
- Overhead: **53% slower**
The solution is to wrap inference in `torch.no_grad()`:
```python
@torch.no_grad()
def predict(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """
    Run inference with gradients disabled.
    The @torch.no_grad() decorator is equivalent to:
        with torch.no_grad():
            return model(inputs)
    But applies to all calls to this function automatically.
    """
    return model(inputs)
```
**Alternative: torch.inference_mode()**
For PyTorch 2.0+, `torch.inference_mode()` is even more aggressive:
```python
# no_grad: Disables gradient computation, but tensors can still be used later
# inference_mode: Tensors are "read-only" - can't be used for further computation
@torch.inference_mode()
def predict_fast(model, inputs):
    return model(inputs)
```
`inference_mode` is ~5-10% faster than `no_grad` but has restrictions: you can't use the output tensors in any operation that might require gradients later. For pure inference serving, this is fine.
---
## Phase 3: CUDA Warmup — The 500ms Tax

![GPU Memory Budget Allocation](./diagrams/tdd-diag-002.svg)

![CUDA Kernel Caching: Before and After Warmup](./diagrams/diag-m1-cuda-warmup.svg)

### What Actually Happens on First Inference
When you run `model(input_tensor)` for the first time on GPU, PyTorch doesn't just execute the model. It:
1. **Selects CUDA kernels** for each operation (matrix multiply, activation, etc.)
2. **JIT-compiles** those kernels for your specific input dimensions
3. **Caches** the compiled kernels in the CUDA context
This is the **CUDA kernel cache**, and it's why first inference is slow.
```python
def measure_warmup_effect(model, input_tensor):
    """Demonstrate the warmup penalty on first inference."""
    # Clear CUDA cache to simulate cold start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    # First inference (COLD)
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(input_tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    first_inference_ms = (time.perf_counter() - start) * 1000
    # Second inference (WARM)
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(input_tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    second_inference_ms = (time.perf_counter() - start) * 1000
    print(f"[Warmup] First inference: {first_inference_ms:.1f}ms")
    print(f"[Warmup] Second inference: {second_inference_ms:.1f}ms")
    print(f"[Warmup] Warmup penalty: {first_inference_ms - second_inference_ms:.1f}ms")
    return first_inference_ms, second_inference_ms
```
Typical results for a transformer model:
- First inference: **487ms**
- Second inference: **43ms**
- Warmup penalty: **444ms** (over 10x slower!)
### The Warmup Solution
You must run warmup inferences **immediately after loading**, before declaring the model ready:
```python
def warmup_model(
    model: torch.nn.Module,
    device: torch.device,
    input_shape: tuple,
    n_warmup_runs: int = 5
) -> None:
    """
    Run dummy inferences to populate CUDA kernel cache.
    Args:
        model: The loaded model
        device: Device the model is on
        input_shape: Shape of expected inputs (batch, channels, height, width) or
                     (batch, seq_len) for transformers
        n_warmup_runs: Number of warmup iterations (default 5)
    Why multiple runs? First run compiles kernels, subsequent runs
    verify stable performance and populate any lazy-initialized buffers.
    """
    print(f"[Warmup] Starting {n_warmup_runs} warmup iterations...")
    # Create dummy input matching expected shape
    dummy_input = torch.randn(input_shape, device=device)
    latencies = []
    for i in range(n_warmup_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        print(f"[Warmup] Run {i+1}/{n_warmup_runs}: {latency_ms:.1f}ms")
    # Verify warmup converged (last run should be similar to second-to-last)
    if len(latencies) >= 2:
        final_latency = latencies[-1]
        variance = abs(latencies[-1] - latencies[-2])
        if variance > final_latency * 0.1:  # More than 10% variance
            print(f"[Warmup] Warning: Latency variance high ({variance:.1f}ms). "
                  "Consider increasing warmup runs.")
        else:
            print(f"[Warmup] Complete. Stable latency: {final_latency:.1f}ms")
```
**Why 5 runs?** Empirically, most models stabilize after 2-3 runs. Running 5 gives confidence and catches edge cases where lazy initialization spans multiple operations.
---
## Phase 4: Request/Response Serialization
### The JSON Bottleneck
Your model produces predictions in milliseconds. Then you spend 15ms converting them to JSON. This is the serialization tax.
The problem: Python's stdlib `json` module is implemented in pure Python. Every `json.dumps()` call involves:
1. Python object introspection
2. String building character by character
3. Unicode encoding
For numpy arrays (which PyTorch tensors convert to), this is especially painful:
```python
import json
import numpy as np
import time
# Simulate model output: batch of 32 embeddings, each 768 dimensions
output = np.random.randn(32, 768).tolist()  # Convert to Python lists
# stdlib json
start = time.perf_counter()
json_str = json.dumps({"predictions": output})
stdlib_time_ms = (time.perf_counter() - start) * 1000
print(f"[Serialization] stdlib json: {stdlib_time_ms:.1f}ms")
print(f"[Serialization] Output size: {len(json_str) / 1024:.1f} KB")
```
Output:
```
[Serialization] stdlib json: 89.3ms
[Serialization] Output size: 378.0 KB
```
**89ms to serialize.** Your model inference was 40ms. Serialization just doubled your latency.

![CUDA Warmup Effect on Latency](./diagrams/tdd-diag-003.svg)

![JSON Serialization Performance: stdlib vs orjson](./diagrams/diag-m1-serialization-benchmark.svg)

![Serialization Benchmark Comparison](./diagrams/tdd-diag-005.svg)

### The orjson Solution
`orjson` is a Rust-based JSON library that's 5-10x faster for array-like data:
```python
import orjson
# orjson.dumps returns bytes, not string
start = time.perf_counter()
json_bytes = orjson.dumps({"predictions": output})
orjson_time_ms = (time.perf_counter() - start) * 1000
print(f"[Serialization] orjson: {orjson_time_ms:.1f}ms")
# It's also more compact
print(f"[Serialization] orjson size: {len(json_bytes) / 1024:.1f} KB")
```
Output:
```
[Serialization] orjson: 8.7ms
[Serialization] orjson size: 351.2 KB
```
**8.7ms vs 89ms — a 10x improvement.**
Here's the production-ready serialization module:
```python
import orjson
from typing import Any, Union
import numpy as np
import torch
class SerializationError(Exception):
    """Raised when serialization/deserialization fails."""
    pass
def serialize_response(data: dict) -> bytes:
    """
    Serialize response dict to JSON bytes using orjson.
    Args:
        data: Dict with JSON-serializable values (including numpy arrays)
    Returns:
        JSON bytes (ready for HTTP response body)
    Note: orjson handles numpy arrays natively, no .tolist() needed
    """
    try:
        return orjson.dumps(
            data,
            option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_UTC_Z
        )
    except TypeError as e:
        raise SerializationError(f"Cannot serialize response: {e}")
def deserialize_request(body: bytes) -> dict:
    """
    Deserialize JSON request body to dict.
    Args:
        body: Raw bytes from HTTP request
    Returns:
        Parsed dict
    Raises:
        SerializationError: If JSON is invalid
    """
    try:
        return orjson.loads(body)
    except orjson.JSONDecodeError as e:
        raise SerializationError(f"Invalid JSON: {e}")
def tensor_to_json(tensor: torch.Tensor) -> list:
    """
    Convert PyTorch tensor to JSON-serializable format.
    Args:
        tensor: PyTorch tensor (any shape, any device)
    Returns:
        Nested lists (JSON-compatible)
    Note: Moves to CPU and converts to numpy first for efficiency
    """
    # Detach from graph, move to CPU, convert to numpy
    np_array = tensor.detach().cpu().numpy()
    # orjson handles numpy directly, but for stdlib json:
    return np_array.tolist()
def benchmark_serialization():
    """
    Benchmark stdlib json vs orjson for typical inference payloads.
    Run this as part of your service startup to validate the speedup.
    """
    import json
    # Generate representative payload
    payload = {
        "predictions": np.random.randn(32, 768).tolist(),
        "latency_ms": 42.5,
        "model_version": "v1.2.3",
        "request_id": "abc123"
    }
    # stdlib json
    n_runs = 100
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = json.dumps(payload)
    stdlib_avg_ms = (time.perf_counter() - start) / n_runs * 1000
    # orjson
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = orjson.dumps(payload)
    orjson_avg_ms = (time.perf_counter() - start) / n_runs * 1000
    speedup = stdlib_avg_ms / orjson_avg_ms
    print(f"[Benchmark] Serialization ({n_runs} runs avg):")
    print(f"  stdlib json: {stdlib_avg_ms:.2f}ms")
    print(f"  orjson:      {orjson_avg_ms:.2f}ms")
    print(f"  Speedup:     {speedup:.1f}x")
    return {
        "stdlib_ms": stdlib_avg_ms,
        "orjson_ms": orjson_avg_ms,
        "speedup": speedup
    }
```
---
## Phase 5: Input Validation
### The Shape/Type Contract
Models have strict input requirements. A ResNet expects `(batch, 3, 224, 224)`. A BERT model expects `(batch, seq_len)` with tokens in range [0, vocab_size]. Violate these and you get cryptic CUDA errors or silently wrong results.
```python
from dataclasses import dataclass
from typing import Tuple, List, Optional, Union
import torch
@dataclass
class InputSchema:
    """Defines the expected input shape and type for a model."""
    name: str
    shape: Tuple[Union[int, None], ...]  # None = variable dimension
    dtype: torch.dtype
    description: str
    def validate(self, tensor: torch.Tensor) -> Tuple[bool, Optional[str]]:
        """
        Validate a tensor against this schema.
        Returns:
            (is_valid, error_message)
        """
        # Check dtype
        if tensor.dtype != self.dtype:
            return False, (
                f"Wrong dtype for '{self.name}': expected {self.dtype}, "
                f"got {tensor.dtype}"
            )
        # Check shape (allow None for variable dims)
        if len(tensor.shape) != len(self.shape):
            return False, (
                f"Wrong rank for '{self.name}': expected {len(self.shape)}D, "
                f"got {len(tensor.shape)}D with shape {tensor.shape}"
            )
        for i, (actual, expected) in enumerate(zip(tensor.shape, self.shape)):
            if expected is not None and actual != expected:
                return False, (
                    f"Wrong shape for '{self.name}': dimension {i} expected "
                    f"{expected}, got {actual}. Full shape: {tensor.shape}"
                )
        return True, None
class ModelInputValidator:
    """
    Validates input tensors against model's expected schema.
    """
    def __init__(self, schemas: List[InputSchema]):
        self.schemas = {s.name: s for s in schemas}
    def validate_request(
        self, 
        request_data: dict
    ) -> Tuple[bool, Union[dict, str]]:
        """
        Validate and parse incoming JSON request.
        Args:
            request_data: Parsed JSON dict from request body
        Returns:
            (success, result):
                - If success: (True, {"inputs": {...}})
                - If failure: (False, "error message")
        """
        inputs = {}
        for name, schema in self.schemas.items():
            if name not in request_data:
                return False, f"Missing required input: '{name}'"
            try:
                # Convert to tensor
                raw = request_data[name]
                tensor = torch.tensor(raw, dtype=schema.dtype)
                # Validate against schema
                is_valid, error = schema.validate(tensor)
                if not is_valid:
                    return False, error
                inputs[name] = tensor
            except (ValueError, TypeError) as e:
                return False, f"Invalid input '{name}': {e}"
        return True, {"inputs": inputs}
# Example: Define schema for an image classification model
IMAGE_CLASSIFICATION_SCHEMA = [
    InputSchema(
        name="image",
        shape=(None, 3, 224, 224),  # Variable batch, fixed image size
        dtype=torch.float32,
        description="Batch of images as float tensors (normalized)"
    )
]
# Example: Define schema for a text embedding model
TEXT_EMBEDDING_SCHEMA = [
    InputSchema(
        name="input_ids",
        shape=(None, None),  # Variable batch and sequence length
        dtype=torch.long,
        description="Token IDs from tokenizer"
    ),
    InputSchema(
        name="attention_mask",
        shape=(None, None),
        dtype=torch.long,
        description="1 for real tokens, 0 for padding"
    )
]
```
### Handling Validation Errors Gracefully
When validation fails, return a **400 Bad Request** with a helpful message:
```python
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
app = FastAPI()
validator = ModelInputValidator(IMAGE_CLASSIFICATION_SCHEMA)
@app.post("/predict")
async def predict_endpoint(request: Request) -> Response:
    """
    Inference endpoint with full validation.
    Request body (JSON):
        {
            "image": [[[[0.1, 0.2, ...]]]]  // shape: (1, 3, 224, 224)
        }
    Response (JSON):
        {
            "predictions": [[0.1, 0.05, ...]],  // class probabilities
            "latency_ms": 42.5,
            "model_version": "v1.2.3"
        }
    Error responses:
        400: Invalid input (shape, dtype, missing field)
        503: Model not ready (still loading/warming up)
        500: Internal error during inference
    """
    # Check model is ready
    if not model_state.is_ready:
        raise HTTPException(
            status_code=503, 
            detail="Model not ready. Check /health/readiness"
        )
    # Parse request body
    try:
        body = await request.body()
        request_data = deserialize_request(body)
    except SerializationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Validate inputs
    is_valid, result = validator.validate_request(request_data)
    if not is_valid:
        raise HTTPException(status_code=400, detail=result)
    # Run inference
    inputs = result["inputs"]
    input_tensor = inputs["image"].to(device)
    try:
        with torch.no_grad():
            predictions = model(input_tensor)
        # Serialize and return
        response_data = {
            "predictions": tensor_to_json(predictions),
            "latency_ms": 42.5,  # Would be measured in production
            "model_version": model_state.version
        }
        return Response(
            content=serialize_response(response_data),
            media_type="application/json"
        )
    except RuntimeError as e:
        # CUDA errors, shape mismatches during inference, etc.
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
```

![Model Loading Lifecycle State Machine](./diagrams/tdd-diag-001.svg)

![Tensor Shape Validation Flow](./diagrams/diag-m1-tensor-shapes.svg)

---
## Phase 6: Health Probes — Readiness vs. Liveness

![Readiness vs Liveness Probes](./diagrams/diag-m1-health-probes.svg)

![Health Probe Response Logic](./diagrams/tdd-diag-006.svg)

Kubernetes (and other orchestrators) distinguish two types of health:
| Probe Type | Question It Answers | Failure Behavior |
|------------|---------------------|------------------|
| **Liveness** | "Is this process alive?" | Restart the pod |
| **Readiness** | "Can this pod serve traffic?" | Stop routing traffic, but don't restart |
**The critical insight**: A pod can be **alive but not ready**. During model loading, the process is running (liveness passes) but can't serve predictions (readiness fails).
```python
import asyncio
from dataclasses import dataclass
from enum import Enum
import time
class ModelState(Enum):
    LOADING = "loading"
    WARMING_UP = "warming_up"
    READY = "ready"
    FAILED = "failed"
@dataclass
class GlobalState:
    """Shared state for the model serving application."""
    model_state: ModelState = ModelState.LOADING
    model: Optional[torch.nn.Module] = None
    device: Optional[torch.device] = None
    version: str = "v1.0.0"
    load_error: Optional[str] = None
    ready_since: Optional[float] = None
model_state = GlobalState()
@app.get("/health/liveness")
async def liveness() -> dict:
    """
    Liveness probe - returns 200 as long as the process is running.
    Kubernetes will restart the pod if this fails (returns non-200).
    This should only fail if the process is dead or completely hung.
    """
    return {"status": "alive"}
@app.get("/health/readiness")
async def readiness() -> dict:
    """
    Readiness probe - returns 200 only after model is loaded and warmed up.
    Kubernetes will stop routing traffic to this pod if this fails,
    but won't restart it. The pod gets time to finish loading.
    States:
        LOADING: Model file is being loaded into memory
        WARMING_UP: Running warmup inferences
        READY: Fully operational
        FAILED: Loading failed, requires intervention
    """
    if model_state.model_state == ModelState.READY:
        return {
            "status": "ready",
            "model_version": model_state.version,
            "device": str(model_state.device),
            "ready_since": model_state.ready_since
        }
    elif model_state.model_state == ModelState.FAILED:
        # Return 503 with error details
        raise HTTPException(
            status_code=503,
            detail={
                "status": "failed",
                "error": model_state.load_error
            }
        )
    else:
        # Loading or warming up
        raise HTTPException(
            status_code=503,
            detail={
                "status": model_state.model_state.value,
                "message": "Model not ready yet"
            }
        )
```
### Startup Sequence with Health Gates
Here's the complete startup lifecycle that ties everything together:
```python
from contextlib import asynccontextmanager
@asynccontextmanager
async def lifespan(app):
    """
    FastAPI lifespan handler - manages startup and shutdown.
    This runs BEFORE the app starts accepting requests, but
    in a way that doesn't block health check endpoints.
    """
    # === STARTUP ===
    print("[Startup] Beginning model initialization...")
    model_state.model_state = ModelState.LOADING
    try:
        # Step 1: Device detection
        device = get_device()
        model_state.device = device
        # Step 2: Memory check
        check_memory_budget(MODEL_PATH, device)
        # Step 3: Model loading
        model = load_model(MODEL_PATH, device)
        model_state.model = model
        # Step 4: Warmup (CRITICAL: must complete before READY)
        model_state.model_state = ModelState.WARMING_UP
        warmup_model(
            model=model,
            device=device,
            input_shape=(1, 3, 224, 224),  # Adjust for your model
            n_warmup_runs=5
        )
        # Step 5: Run serialization benchmark
        benchmark_serialization()
        # Step 6: Mark ready
        model_state.model_state = ModelState.READY
        model_state.ready_since = time.time()
        print(f"[Startup] Model {model_state.version} ready on {device}")
    except Exception as e:
        model_state.model_state = ModelState.FAILED
        model_state.load_error = str(e)
        print(f"[Startup] FAILED: {e}")
        # Don't raise - let the app start so health probes can report failure
    yield  # App runs here
    # === SHUTDOWN ===
    print("[Shutdown] Cleaning up...")
    if model_state.model is not None:
        del model_state.model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
# Apply lifespan to app
app = FastAPI(lifespan=lifespan)
```
**The key insight**: Health endpoints (`/health/liveness`, `/health/readiness`) are registered outside the lifespan context, so they're available **immediately** when the process starts. The lifespan handler runs in parallel with the server being able to respond to health checks.
---
## Putting It All Together: The Complete Model Loader
```python
"""
model_loader.py - Production model loading with all phases.
This module implements the complete model loading lifecycle:
1. Device detection with fallback
2. Memory budget validation
3. Model loading with eval mode
4. CUDA warmup for kernel caching
5. Serialization benchmarking
"""
import os
import time
from typing import Optional, Tuple
from dataclasses import dataclass
import torch
import orjson
# Configuration (would typically come from env vars or config file)
MODEL_PATH = os.getenv("MODEL_PATH", "models/resnet50.pt")
WARMUP_RUNS = int(os.getenv("WARMUP_RUNS", "5"))
INPUT_SHAPE = tuple(map(int, os.getenv("INPUT_SHAPE", "1,3,224,224").split(",")))
@dataclass
class LoadedModel:
    """Container for a loaded, warmed-up model ready for inference."""
    model: torch.nn.Module
    device: torch.device
    version: str
    input_shape: Tuple[int, ...]
    warmup_latency_ms: float
    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run inference on inputs, automatically handling device placement."""
        inputs = inputs.to(self.device)
        return self.model(inputs)
def load_for_inference(
    model_path: str,
    input_shape: Tuple[int, ...],
    n_warmup_runs: int = 5,
    version: str = "v1.0.0"
) -> LoadedModel:
    """
    Complete model loading pipeline for production inference.
    This function:
    1. Detects and validates the compute device
    2. Checks memory budget before loading
    3. Loads the model in eval mode
    4. Runs warmup inferences to populate CUDA caches
    5. Returns a LoadedModel ready for inference
    Args:
        model_path: Path to saved PyTorch model
        input_shape: Expected input tensor shape (for warmup)
        n_warmup_runs: Number of warmup iterations
        version: Model version string for tracking
    Returns:
        LoadedModel instance ready for inference
    Raises:
        RuntimeError: If memory is insufficient
        FileNotFoundError: If model file doesn't exist
        torch.TorchException: If model loading fails
    """
    print(f"[Loader] Starting load of {model_path}")
    start_time = time.perf_counter()
    # Phase 1: Device detection
    device = get_device()
    # Phase 2: Memory validation
    check_memory_budget(model_path, device)
    # Phase 3: Load model
    model = torch.load(model_path, map_location=device)
    model.eval()
    print(f"[Loader] Model loaded to {device}")
    # Phase 4: Warmup
    warmup_start = time.perf_counter()
    warmup_model(model, device, input_shape, n_warmup_runs)
    warmup_time = time.perf_counter() - warmup_start
    # Measure post-warmup latency
    dummy = torch.randn(input_shape, device=device)
    with torch.no_grad():
        torch.cuda.synchronize() if device.type == "cuda" else None
        infer_start = time.perf_counter()
        _ = model(dummy)
        torch.cuda.synchronize() if device.type == "cuda" else None
        warmup_latency_ms = (time.perf_counter() - infer_start) * 1000
    total_time = time.perf_counter() - start_time
    print(f"[Loader] Complete in {total_time:.2f}s (warmup: {warmup_time:.2f}s)")
    print(f"[Loader] Stable inference latency: {warmup_latency_ms:.1f}ms")
    return LoadedModel(
        model=model,
        device=device,
        version=version,
        input_shape=input_shape,
        warmup_latency_ms=warmup_latency_ms
    )
# Usage in main application:
# loaded = load_for_inference(MODEL_PATH, INPUT_SHAPE, WARMUP_RUNS)
# predictions = loaded.predict(input_tensor)
```
---
## Optional Extension: ONNX Runtime for Maximum Performance
ONNX (Open Neural Network Exchange) is a serialization format that enables optimizations impossible in PyTorch eager mode. ONNX Runtime can be 1.5-3x faster than PyTorch inference:
```python
import onnxruntime as ort
import numpy as np
class ONNXModelWrapper:
    """
    Wrapper for ONNX Runtime inference with same API as PyTorch models.
    """
    def __init__(self, onnx_path: str, device: torch.device):
        """
        Load ONNX model for inference.
        Args:
            onnx_path: Path to .onnx model file
            device: torch.device (converted to ONNX Runtime provider)
        """
        # Map torch device to ONNX Runtime provider
        if device.type == "cuda":
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': device.index if device.index else 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB limit
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                }),
                'CPUExecutionProvider'  # Fallback
            ]
        else:
            providers = ['CPUExecutionProvider']
        # Create inference session with optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers
        )
        # Get input/output names for inference
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        print(f"[ONNX] Loaded {onnx_path} with providers: {self.session.get_providers()}")
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Enable model(tensor) syntax like PyTorch."""
        return self.predict(*args, **kwargs)
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Run ONNX inference.
        Args:
            inputs: PyTorch tensor (will be converted to numpy)
        Returns:
            PyTorch tensor with outputs
        """
        # Convert to numpy
        if isinstance(inputs, torch.Tensor):
            inputs_np = inputs.detach().cpu().numpy()
        else:
            inputs_np = inputs
        # Run inference
        input_feed = {self.input_names[0]: inputs_np}
        outputs = self.session.run(self.output_names, input_feed)
        # Convert back to torch tensor
        return torch.from_numpy(outputs[0])
    def warmup(self, input_shape: Tuple[int, ...], n_runs: int = 5):
        """Run warmup inferences (ONNX Runtime also benefits from warmup)."""
        dummy = np.random.randn(*input_shape).astype(np.float32)
        for i in range(n_runs):
            _ = self.predict(dummy)
        print(f"[ONNX] Warmup complete ({n_runs} runs)")
```
### When to Use ONNX vs PyTorch
| Criterion | PyTorch | ONNX Runtime |
|-----------|---------|--------------|
| **Ease of use** | Native, no conversion | Requires export step |
| **Latency** | Baseline | 1.5-3x faster |
| **Model compatibility** | Universal | Some ops not supported |
| **Debugging** | Easy (Python) | Harder (graph format) |
| **Deployment size** | Requires PyTorch runtime | Standalone (~100MB) |
**Recommendation**: Start with PyTorch. Once your serving pipeline is stable, export to ONNX and benchmark. If you see meaningful speedup (test on your actual hardware), adopt ONNX for production.
---
## Knowledge Cascade: What You've Unlocked
You now understand the complete model loading lifecycle. This knowledge connects to:
### Immediate Applications
- **Milestone 2 (Request Batching)**: The warmup you implemented uses batch size 1. Batching changes the tensor shapes flowing through the model, which may require *additional* warmup runs at production batch sizes.
- **Milestone 3 (Model Versioning)**: The `LoadedModel` container is designed to be swapped atomically—multiple versions can coexist in the registry.
### Cross-Domain Connections
**1. CUDA Context Initialization → Database Connection Pooling**
The warmup problem is isomorphic to database connection pooling:
- First query is slow (connection establishment, prepared statement compilation)
- Solution: Pre-warm connections before serving traffic
- Both systems have "cold start" penalties that must be hidden from users
**2. Serialization Bottlenecks → Network Protocol Design**
The `orjson` vs `json` speedup reflects a fundamental systems principle: **serialization format matters**. The same tradeoff appears in:
- Protocol Buffers vs JSON (10-100x smaller, faster to parse)
- Apache Arrow vs CSV for data interchange
- MessagePack vs JSON for real-time systems
**3. Readiness vs Liveness → Distributed Consensus**
Kubernetes's distinction between "alive" and "ready" is a simplified version of **health checking in distributed consensus**. In Raft/Paxos, a node must be:
- **Alive**: Process running, network reachable
- **Caught up**: Log replicated to current state
- **Ready to lead**: Has quorum support
A node can be alive but not caught up—just like your pod can be alive but still loading its model.
### Forward Connections
With model loading mastered, you can now:
- Build **model hot-swapping** systems that load new versions without dropping requests
- Implement **multi-GPU serving** where different models live on different GPUs
- Create **model cascades** where simple models filter requests before expensive models process them
- Design **fallback chains** that degrade gracefully when primary models fail
---
<!-- END_MS -->


<!-- MS_ID: ml-model-serving-m2 -->
# Request Batching
## The GPU Utilization Illusion
You've built a model server that loads models correctly, handles device placement, and responds to requests in ~50ms. You deploy it, send traffic, and watch your metrics. Everything looks fine—until you look at GPU utilization during peak load.
**It's 8%.**
Not 80%. Eight percent. Your expensive GPU—the one with 16GB of VRAM and 10,496 CUDA cores—is sitting almost completely idle while your server processes hundreds of requests per second.
```python
# What you thought was happening
for request in incoming_requests:
    prediction = model(request.input)  # 50ms per request
    return prediction
# 20 requests/second * 50ms = 1000ms of work per second
# GPU should be 100% utilized, right?
```
The misconception is seductive: *GPUs are fast, so processing requests one at a time should be efficient.* After all, each inference only takes 50ms. Where's the waste?

![GPU Utilization: Sequential vs Batched Inference](./diagrams/diag-m2-gpu-utilization.svg)

The waste is hiding in **kernel launch overhead**—the time the CPU spends telling the GPU what to do. Every time you call `model(input_tensor)`, this happens:
1. **CPU**: Constructs CUDA kernel launch parameters (~5-10µs)
2. **CPU→GPU**: PCIe transfer for kernel launch command (~2-5µs)
3. **GPU**: Executes the actual computation (~0.5-2ms for small batches)
4. **GPU→CPU**: Completion signal (~2-5µs)
For a single inference taking 2ms of actual compute, you spend 15-20µs on overhead—about 1% overhead. Acceptable.
But here's the brutal math: **that overhead happens for EVERY request**. Processing 100 requests sequentially means 100 kernel launches, 100 PCIe round-trips, 100 completion signals. The GPU spends most of its time *waiting for the CPU to tell it what to do*.
**This is the fundamental tension of GPU inference: the GPU is not compute-bound—it's command-bound.**
---
## The Latency-Throughput Tradeoff
> **🔑 Foundation: Batching as throughput optimization**
> 
> ## Batching as Throughput Optimization
### What It IS
Batching combines multiple independent operations into a single operation that processes them together. Instead of `op(input_1)`, `op(input_2)`, `op(input_3)` sequentially, you do `op([input_1, input_2, input_3])` once.
In ML inference, batching means feeding multiple inputs through the model simultaneously using batched matrix operations. A batch of 32 images flows through the network as a single tensor of shape `(32, 3, 224, 224)` instead of 32 separate `(1, 3, 224, 224)` tensors.
### WHY You Need It Right Now
GPU architectures are designed for **throughput**, not latency. A modern GPU has:
- **Thousands of parallel cores** (NVIDIA A100 has 6,912 CUDA cores)
- **Massive memory bandwidth** (1.5-2 TB/s on A100)
- **Hardware schedulers** that distribute work across cores
A single inference request uses perhaps 5-10% of these resources. The rest sit idle. Batching fills those idle cores by giving the GPU more work to do per kernel launch.
**The key insight**: Matrix multiplication—the core operation in neural networks—has O(n³) compute complexity but O(n²) memory access. As batch size increases, compute grows faster than memory traffic, improving arithmetic intensity and keeping cores busy.
### ONE Key Insight
**Think of the GPU as a factory floor with 100 workers, and each inference request is a job that needs 5 workers.**
Without batching: You send one job at a time. Five workers are busy; ninety-five stand around. You wait for the job to finish before sending the next.
With batching: You collect 20 jobs, bundle them, and send them together. All 100 workers are engaged. The jobs finish in roughly the same time as one job would have taken alone.
The factory (GPU) has fixed capacity. Batching is the art of filling that capacity efficiently.
### The Pareto Frontier
Every batching decision places you on a curve—the **latency-throughput Pareto frontier**. You cannot maximize both simultaneously; every gain in throughput costs some latency.

![Latency-Throughput Pareto Frontier](./diagrams/diag-m2-latency-throughput-curve.svg)

Let's quantify this with real measurements on a ResNet-50 model:
| Batch Size | Latency (ms) | Throughput (req/s) | GPU Utilization |
|------------|--------------|--------------------|-----------------| 
| 1 | 12 | 83 | 8% |
| 4 | 18 | 222 | 35% |
| 8 | 24 | 333 | 52% |
| 16 | 38 | 421 | 68% |
| 32 | 62 | 516 | 81% |
| 64 | 108 | 592 | 89% |
| 128 | 195 | 656 | 94% |
**The pattern**: Throughput increases sublinearly, latency increases superlinearly. At batch size 32, you get **6x the throughput** for only **5x the latency**. The GPU is finally doing what it was built for—massively parallel computation.
But here's the problem: **users don't send requests in convenient batches of 32**. They send one request at a time, at unpredictable intervals. How do you batch requests that arrive asynchronously?
---
## Dynamic Request Batching: The Architecture

![Dynamic Request Batcher Architecture](./diagrams/diag-m2-batcher-architecture.svg)

Dynamic batching solves the async arrival problem by introducing a **request queue** and a **batch formation loop**:
1. **Request arrival**: HTTP request comes in, gets wrapped with metadata, placed in queue
2. **Batch formation**: Background task monitors queue, groups requests into batches
3. **Batch inference**: Full or partial batch is processed as a single GPU operation
4. **Response routing**: Results are split and sent back to each original requester
The key insight is that **the batch formation happens asynchronously from request arrival**. Requests pile up in the queue; a separate task harvests them when conditions are met.
### The Three-Level View
**Level 1 — Application Layer**: 
- HTTP endpoints receive requests
- Each request gets a unique ID and a Future (promise of future result)
- Requests are wrapped in `BatchRequest` objects
**Level 2 — Batching Engine**:
- Async queue holds pending requests
- Batch former monitors queue with timeout
- Forms batches when either: max batch size reached OR timeout expires
**Level 3 — GPU Execution**:
- Batched tensor flows through model
- Single kernel launch processes entire batch
- Results split and routed back through Futures
---
## Building the Request Queue
### The AsyncIO Pattern
Python's `asyncio.Queue` provides exactly what we need: thread-safe, async-compatible queuing with the ability to wait for items without blocking the event loop.
```python
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, List, Tuple
from enum import Enum
import torch
class RequestStatus(Enum):
    """Lifecycle states for a batch request."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
@dataclass
class BatchRequest:
    """
    Container for a single inference request in the batch queue.
    The Future allows the request handler to await the result
    while the batch processor resolves it asynchronously.
    """
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    input_tensor: Optional[torch.Tensor] = None
    arrival_time: float = field(default_factory=time.perf_counter)
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())
    status: RequestStatus = RequestStatus.QUEUED
    # Metadata for metrics
    input_shape: Tuple[int, ...] = ()
    def wait_time_ms(self) -> float:
        """Time spent waiting in queue (ms)."""
        return (time.perf_counter() - self.arrival_time) * 1000
    def set_result(self, result: torch.Tensor):
        """Mark request as completed with result."""
        self.status = RequestStatus.COMPLETED
        if not self.future.done():
            self.future.set_result(result)
    def set_error(self, error: Exception):
        """Mark request as failed with error."""
        self.status = RequestStatus.FAILED
        if not self.future.done():
            self.future.set_exception(error)
class RequestQueue:
    """
    Async queue with backpressure and metrics.
    This is NOT just asyncio.Queue—it adds:
    - Maximum depth with rejection (backpressure)
    - Metrics tracking (queue depth, wait times)
    - Graceful shutdown support
    """
    def __init__(
        self,
        max_depth: int = 1000,
        name: str = "default"
    ):
        self.queue: asyncio.Queue[BatchRequest] = asyncio.Queue()
        self.max_depth = max_depth
        self.name = name
        # Metrics
        self.total_enqueued = 0
        self.total_rejected = 0
        self.total_processed = 0
        self._shutdown = False
    async def enqueue(self, request: BatchRequest) -> bool:
        """
        Add request to queue with backpressure check.
        Returns:
            True if enqueued, False if rejected due to backpressure
        """
        if self._shutdown:
            request.set_error(RuntimeError("Queue is shut down"))
            return False
        if self.queue.qsize() >= self.max_depth:
            self.total_rejected += 1
            request.set_error(
                RuntimeError(
                    f"Queue {self.name} at capacity ({self.max_depth}). "
                    "Retry later."
                )
            )
            return False
        await self.queue.put(request)
        self.total_enqueued += 1
        return True
    async def get_batch(
        self,
        max_batch_size: int,
        timeout_ms: float
    ) -> List[BatchRequest]:
        """
        Wait for and return a batch of requests.
        Batch formation rules:
        1. Wait up to timeout_ms for the first request
        2. Collect up to max_batch_size requests
        3. Return immediately when batch is full OR timeout expires
        This bounds worst-case latency while maximizing batch fullness.
        """
        batch: List[BatchRequest] = []
        # Wait for first request (with timeout)
        try:
            first_request = await asyncio.wait_for(
                self.queue.get(),
                timeout=timeout_ms / 1000.0
            )
            batch.append(first_request)
        except asyncio.TimeoutError:
            # No requests arrived within timeout
            return batch
        # Collect additional requests without waiting
        # (they're already in the queue if traffic is high)
        while len(batch) < max_batch_size:
            try:
                # Non-blocking get
                request = self.queue.get_nowait()
                batch.append(request)
            except asyncio.QueueEmpty:
                break
        self.total_processed += len(batch)
        return batch
    def qsize(self) -> int:
        """Current queue depth."""
        return self.queue.qsize()
    def metrics(self) -> dict:
        """Queue metrics for monitoring."""
        return {
            f"queue_{self.name}_depth": self.qsize(),
            f"queue_{self.name}_max_depth": self.max_depth,
            f"queue_{self.name}_total_enqueued": self.total_enqueued,
            f"queue_{self.name}_total_rejected": self.total_rejected,
            f"queue_{self.name}_total_processed": self.total_processed,
            f"queue_{self.name}_rejection_rate": (
                self.total_rejected / max(1, self.total_enqueued + self.total_rejected)
            )
        }
    def shutdown(self):
        """Signal queue to stop accepting requests."""
        self._shutdown = True
```
**Why the two-phase batch collection?** The `wait_for` with timeout handles the "wait for requests" case, while `get_nowait` in the loop handles the "drain existing requests" case. This ensures we don't over-wait when requests are already queued.
### Cross-Domain Connection: Async Queue Patterns
This queue pattern is isomorphic to patterns you'll see in:
- **Network servers**: Accept connections → dispatch to worker pool
- **Message brokers**: Producer sends → consumer batches → process
- **Operating systems**: Block I/O requests → elevator scheduler → disk access
The fundamental pattern: **asynchronous arrival, synchronous batch processing**. You'll recognize it everywhere once you understand it here.
---
## The Batch Formation Loop

![Batch Formation Timeline](./diagrams/diag-m2-batch-formation-timeline.svg)

Now we need a long-running task that continuously forms batches and dispatches them for inference.
```python
from dataclasses import dataclass
from typing import Callable, Awaitable
import logging
logger = logging.getLogger(__name__)
@dataclass
class BatcherConfig:
    """Configuration for the dynamic batcher."""
    max_batch_size: int = 32
    timeout_ms: float = 50.0
    max_queue_depth: int = 1000
    # Backpressure
    backpressure_enabled: bool = True
class DynamicBatcher:
    """
    Continuously forms batches from queued requests and processes them.
    This is the heart of the batching system:
    - Runs as a background asyncio task
    - Monitors the request queue
    - Forms batches with timeout-bounded latency
    - Dispatches to model inference
    - Routes results back to requesters
    """
    def __init__(
        self,
        config: BatcherConfig,
        model_fn: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device
    ):
        self.config = config
        self.model_fn = model_fn  # The inference function
        self.device = device
        # Queue for incoming requests
        self.queue = RequestQueue(
            max_depth=config.max_queue_depth,
            name="inference"
        )
        # Metrics
        self.batches_processed = 0
        self.total_requests_processed = 0
        self.batch_sizes: List[int] = []  # Last N batch sizes
        self.inference_latencies: List[float] = []  # Last N inference times
        self._task: Optional[asyncio.Task] = None
        self._running = False
    async def submit(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Submit a request for batched inference.
        This is called by the HTTP handler for each incoming request.
        The call returns when the result is ready (awaitable).
        Args:
            input_tensor: Input data for inference
        Returns:
            Model output tensor
        Raises:
            RuntimeError: If queue is at capacity (backpressure)
        """
        request = BatchRequest(
            input_tensor=input_tensor,
            input_shape=tuple(input_tensor.shape)
        )
        # Attempt to enqueue (respects backpressure)
        success = await self.queue.enqueue(request)
        if not success:
            raise RuntimeError(
                f"Request rejected due to backpressure. "
                f"Queue depth: {self.queue.qsize()}/{self.config.max_queue_depth}"
            )
        # Wait for result (set by batch processor)
        return await request.future
    async def _batch_loop(self):
        """
        Main loop: continuously form and process batches.
        This runs forever (until stop() is called), performing:
        1. Wait for batch (with timeout)
        2. Pad and stack inputs
        3. Run batch inference
        4. Split and route results
        """
        logger.info(
            f"[Batcher] Starting batch loop: "
            f"max_batch={self.config.max_batch_size}, "
            f"timeout={self.config.timeout_ms}ms"
        )
        while self._running:
            try:
                # Phase 1: Form batch
                batch_start = time.perf_counter()
                batch = await self.queue.get_batch(
                    max_batch_size=self.config.max_batch_size,
                    timeout_ms=self.config.timeout_ms
                )
                if not batch:
                    # Empty batch (timeout with no requests)
                    continue
                # Record batch formation time
                batch_formation_ms = (time.perf_counter() - batch_start) * 1000
                # Phase 2: Prepare inputs
                inputs_tensor, batch_metadata = self._prepare_batch_inputs(batch)
                # Phase 3: Run inference
                inference_start = time.perf_counter()
                outputs = await self._run_inference(inputs_tensor)
                inference_ms = (time.perf_counter() - inference_start) * 1000
                # Phase 4: Split and route results
                self._route_results(outputs, batch_metadata)
                # Update metrics
                self.batches_processed += 1
                self.total_requests_processed += len(batch)
                self.batch_sizes.append(len(batch))
                self.inference_latencies.append(inference_ms)
                # Keep only last 100 for memory efficiency
                if len(self.batch_sizes) > 100:
                    self.batch_sizes = self.batch_sizes[-100:]
                    self.inference_latencies = self.inference_latencies[-100:]
                logger.debug(
                    f"[Batcher] Batch {self.batches_processed}: "
                    f"size={len(batch)}, formation={batch_formation_ms:.1f}ms, "
                    f"inference={inference_ms:.1f}ms"
                )
            except asyncio.CancelledError:
                logger.info("[Batcher] Batch loop cancelled")
                break
            except Exception as e:
                logger.error(f"[Batcher] Error in batch loop: {e}", exc_info=True)
                # Continue running despite errors
                await asyncio.sleep(0.1)  # Brief pause before retry
    def _prepare_batch_inputs(
        self, 
        batch: List[BatchRequest]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Stack individual inputs into a batch tensor.
        For fixed-size inputs (e.g., images), this is simple stacking.
        For variable-length inputs (e.g., text), this requires padding.
        Returns:
            (batched_tensor, metadata_for_unbatching)
        """
        # Extract inputs
        inputs = [req.input_tensor for req in batch]
        # Check if all inputs have same shape (fixed-size)
        shapes = [t.shape for t in inputs]
        if len(set(shapes)) == 1:
            # Simple case: stack directly
            batched = torch.stack(inputs, dim=0)
            metadata = {
                "original_shapes": shapes,
                "needs_unpadding": False
            }
        else:
            # Variable-length: need padding
            batched, padding_mask = self._pad_and_stack(inputs)
            metadata = {
                "original_shapes": shapes,
                "padding_mask": padding_mask,
                "needs_unpadding": True
            }
        return batched.to(self.device), metadata
    def _pad_and_stack(
        self, 
        tensors: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad variable-length tensors to uniform length.
        For sequence models, we pad to the longest sequence in the batch.
        Returns both the padded tensor and a mask indicating real vs padding.
        """
        # Find max length in sequence dimension (dim 1 typically)
        max_len = max(t.shape[1] for t in tensors)
        batch_size = len(tensors)
        other_dims = tensors[0].shape[2:]  # Dimensions after sequence
        # Create output tensors
        dtype = tensors[0].dtype
        device = tensors[0].device
        # Padded tensor (zeros for padding)
        padded = torch.zeros(
            (batch_size, max_len) + other_dims,
            dtype=dtype,
            device=device
        )
        # Attention mask (1 for real, 0 for padding)
        mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
            device=device
        )
        # Fill in actual values
        for i, t in enumerate(tensors):
            seq_len = t.shape[1]
            padded[i, :seq_len] = t
            mask[i, :seq_len] = 1
        return padded, mask
    @torch.no_grad()
    async def _run_inference(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Run model inference on batched inputs.
        This is where the GPU actually does work.
        We run it in a thread executor to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        def sync_inference():
            return self.model_fn(inputs)
        # Run in thread pool to not block async loop
        return await loop.run_in_executor(None, sync_inference)
    def _route_results(
        self, 
        outputs: torch.Tensor, 
        metadata: dict
    ):
        """
        Split batch outputs and deliver to each requester.
        This is critical: each request must get its own result,
        not someone else's. Order must match input order.
        """
        # Split along batch dimension
        for i, request in enumerate(self._current_batch):
            # Get this request's output
            output_i = outputs[i]
            # If inputs were padded, may need to trim output
            if metadata.get("needs_unpadding", False):
                original_len = metadata["original_shapes"][i][1]
                # For seq2seq models, trim output sequence
                # (model-dependent; some models output same length as input)
                if output_i.dim() > 1 and output_i.shape[0] > original_len:
                    output_i = output_i[:original_len]
            # Deliver result
            request.set_result(output_i.cpu())  # Move to CPU for pickling safety
    # Store current batch for result routing
    _current_batch: List[BatchRequest] = []
    async def _batch_loop(self):
        """Override to store current batch."""
        # ... (same as before until batch formation)
        while self._running:
            try:
                batch = await self.queue.get_batch(
                    max_batch_size=self.config.max_batch_size,
                    timeout_ms=self.config.timeout_ms
                )
                if not batch:
                    continue
                # Store for result routing
                self._current_batch = batch
                # ... rest of processing
            except asyncio.CancelledError:
                break
    def start(self):
        """Start the batch processing loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._batch_loop())
        logger.info("[Batcher] Started")
    async def stop(self):
        """Gracefully stop the batch processor."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[Batcher] Stopped")
    def metrics(self) -> dict:
        """Return batcher metrics for monitoring."""
        avg_batch_size = (
            sum(self.batch_sizes) / len(self.batch_sizes) 
            if self.batch_sizes else 0
        )
        avg_inference_ms = (
            sum(self.inference_latencies) / len(self.inference_latencies)
            if self.inference_latencies else 0
        )
        return {
            "batcher_batches_processed": self.batches_processed,
            "batcher_requests_processed": self.total_requests_processed,
            "batcher_avg_batch_size": avg_batch_size,
            "batcher_avg_inference_ms": avg_inference_ms,
            **self.queue.metrics()
        }
```
**The critical insight**: The `_batch_loop` runs continuously in the background. HTTP handlers call `submit()` and await the result. The batcher bridges these two worlds—collecting async requests and delivering synchronous inference results.
---
## Handling Variable-Length Inputs

![Variable-Length Input Padding and Attention Masking](./diagrams/diag-m2-padding-masking.svg)

Text models, time series models, and many others accept inputs of varying length. A batch might contain:
- Request A: 128 tokens
- Request B: 512 tokens  
- Request C: 64 tokens
You can't stack these directly—tensor dimensions must match. The solution is **padding and masking**.
### The Padding Strategy
```python
class VariableLengthPadder:
    """
    Handles padding of variable-length sequences for batch inference.
    Key insight: Padding adds zeros to make shapes match, but those zeros
    shouldn't affect the model's output. Attention masks tell the model
    which positions are real vs padding.
    """
    def __init__(self, pad_value: int = 0, max_length: Optional[int] = None):
        self.pad_value = pad_value
        self.max_length = max_length  # Cap max length (truncation)
    def pad_batch(
        self,
        input_ids_list: List[torch.Tensor],
        attention_mask_list: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad a batch of input IDs to uniform length.
        Args:
            input_ids_list: List of 1D tensors with token IDs
            attention_mask_list: Optional pre-existing masks
        Returns:
            (padded_input_ids, attention_mask)
            - padded_input_ids: (batch, max_seq_len)
            - attention_mask: (batch, max_seq_len) - 1 for real, 0 for pad
        """
        batch_size = len(input_ids_list)
        # Determine target length
        lengths = [len(ids) for ids in input_ids_list]
        max_len = max(lengths)
        if self.max_length:
            max_len = min(max_len, self.max_length)
        # Initialize padded tensors
        padded_ids = torch.full(
            (batch_size, max_len),
            self.pad_value,
            dtype=input_ids_list[0].dtype
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long
        )
        # Fill with actual values
        for i, ids in enumerate(input_ids_list):
            seq_len = min(len(ids), max_len)
            padded_ids[i, :seq_len] = ids[:seq_len]
            attention_mask[i, :seq_len] = 1
            # If pre-existing mask provided, use it
            if attention_mask_list and i < len(attention_mask_list):
                attention_mask[i, :seq_len] = attention_mask_list[i][:seq_len]
        return padded_ids, attention_mask
    def calculate_padding_overhead(self, lengths: List[int]) -> float:
        """
        Calculate how much padding was added vs actual content.
        Returns:
            Fraction of batch that is padding (0.0 to 1.0)
        """
        if not lengths:
            return 0.0
        max_len = max(lengths)
        total_tokens = sum(lengths)
        total_padded = max_len * len(lengths)
        return 1.0 - (total_tokens / total_padded)
# Enhanced batch preparation with variable-length support
class DynamicBatcherWithPadding(DynamicBatcher):
    """
    Extended batcher with proper variable-length sequence handling.
    """
    def __init__(self, config: BatcherConfig, model_fn, device, tokenizer=None):
        super().__init__(config, model_fn, device)
        self.padder = VariableLengthPadder()
        self.tokenizer = tokenizer  # For text models
    def _prepare_batch_inputs(self, batch: List[BatchRequest]) -> Tuple[dict, dict]:
        """
        Prepare batch inputs with padding for variable-length sequences.
        For text models, expects input to be pre-tokenized or raw text.
        """
        # Check if this is a text model (string inputs)
        if isinstance(batch[0].input_tensor, str):
            return self._prepare_text_batch(batch)
        else:
            return self._prepare_tensor_batch(batch)
    def _prepare_text_batch(
        self, 
        batch: List[BatchRequest]
    ) -> Tuple[dict, dict]:
        """
        Prepare a batch of text inputs.
        1. Tokenize all texts
        2. Pad to uniform length
        3. Create attention masks
        """
        texts = [req.input_tensor for req in batch]
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.padder.max_length,
            return_tensors="pt"
        )
        inputs = {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device)
        }
        metadata = {
            "original_lengths": [len(t) for t in texts],
            "tokenized_lengths": encoded["attention_mask"].sum(dim=1).tolist()
        }
        return inputs, metadata
    def _prepare_tensor_batch(
        self, 
        batch: List[BatchRequest]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Prepare a batch of tensor inputs (with padding if needed).
        """
        inputs = [req.input_tensor for req in batch]
        shapes = [t.shape for t in inputs]
        # Check for uniform shapes
        if len(set(shapes)) == 1:
            # No padding needed
            batched = torch.stack(inputs, dim=0)
            return batched.to(self.device), {"needs_unpadding": False}
        # Variable-length sequences - check which dimension varies
        # Typically dim 1 (sequence length) for (batch, seq, features)
        seq_lens = [s[1] for s in shapes]
        max_len = max(seq_lens)
        # Pad sequences
        batch_size = len(inputs)
        feature_dim = shapes[0][2:]  # Features after sequence
        dtype = inputs[0].dtype
        padded = torch.zeros(
            (batch_size, max_len) + feature_dim,
            dtype=dtype
        )
        mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        for i, t in enumerate(inputs):
            seq_len = t.shape[1]
            padded[i, :seq_len] = t
            mask[i, :seq_len] = 1
        # Stack input_ids and attention_mask for transformer input
        return (
            {"input_ids": padded.to(self.device), "attention_mask": mask.to(self.device)},
            {"original_shapes": shapes, "needs_unpadding": True}
        )
```
### Why Attention Masks Matter
Padding tokens are zeros—they contribute nothing semantically. But without masks, a transformer will **attend to padding tokens** as if they were real input, corrupting the output.
The attention mask tells the model: "Only look at positions where mask=1, ignore positions where mask=0." This is implemented by setting attention scores to `-inf` for masked positions before softmax:
```
attention_scores = Q @ K.T / sqrt(d_k)
attention_scores = attention_scores.masked_fill(mask == 0, -inf)
attention_weights = softmax(attention_scores)
```
With `-inf` before softmax, those positions get zero weight after softmax. The padding is invisible to the model.
---
## Backpressure and Flow Control

![Backpressure and Queue Depth Management](./diagrams/diag-m2-backpressure-flow.svg)

**Unbounded queues are a ticking time bomb.** When request arrival rate exceeds processing rate, the queue grows without limit. Eventually:
1. Memory exhaustion (each queued request holds tensors)
2. Unbounded latency (requests wait seconds or minutes)
3. Cascading failures (upstream timeouts while requests sit in queue)
**Backpressure** is the antidote: when the queue is full, reject new requests immediately rather than letting them pile up.
```python
from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
import orjson
class BatchingInferenceEndpoint:
    """
    HTTP endpoint that integrates with the dynamic batcher.
    Handles:
    - Request validation
    - Backpressure (503 when overloaded)
    - Timeout handling
    - Metrics collection
    """
    def __init__(
        self,
        batcher: DynamicBatcher,
        timeout_seconds: float = 30.0
    ):
        self.batcher = batcher
        self.timeout_seconds = timeout_seconds
    async def predict(self, request: Request) -> Response:
        """
        Handle inference request through batcher.
        Flow:
        1. Parse and validate input
        2. Submit to batcher (may be rejected for backpressure)
        3. Await result with timeout
        4. Return response
        """
        # Parse request
        try:
            body = await request.body()
            data = orjson.loads(body)
        except orjson.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
        # Validate and convert to tensor
        try:
            input_tensor = self._parse_input(data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        # Submit to batcher
        try:
            result = await asyncio.wait_for(
                self.batcher.submit(input_tensor),
                timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Request timed out after {self.timeout_seconds}s"
            )
        except RuntimeError as e:
            # Backpressure rejection
            if "backpressure" in str(e).lower() or "capacity" in str(e).lower():
                raise HTTPException(
                    status_code=503,
                    detail="Service overloaded. Retry after a moment."
                )
            raise HTTPException(status_code=500, detail=str(e))
        # Return result
        response_data = {
            "prediction": result.tolist() if result.dim() > 0 else result.item(),
            "request_id": data.get("request_id"),
            "model_version": "v1.0.0"  # Would come from model state
        }
        return Response(
            content=orjson.dumps(response_data),
            media_type="application/json"
        )
    def _parse_input(self, data: dict) -> torch.Tensor:
        """Parse input data to tensor. Override for specific models."""
        if "input" not in data:
            raise ValueError("Missing 'input' field")
        return torch.tensor(data["input"], dtype=torch.float32)
```
### Cross-Domain Connection: TCP Congestion Control
Backpressure in ML serving is isomorphic to **TCP congestion control** in networking:
| Networking | ML Serving |
|------------|------------|
| Packet arrival rate > link capacity | Request arrival rate > GPU throughput |
| Buffer fills → packets dropped | Queue fills → requests rejected (503) |
| Sender sees drops → slows down | Client sees 503 → retries with backoff |
| Network recovers | Queue drains, accepts requests again |
Both systems use **explicit rejection** to signal overload, rather than letting latency grow unboundedly. The 503 status code is the ML serving equivalent of a TCP RST (reset) packet.
---
## Metrics and Observability
You can't optimize what you can't measure. A production batcher needs detailed metrics:
```python
from dataclasses import dataclass, field
from collections import deque
import statistics
import time
@dataclass
class BatcherMetrics:
    """
    Comprehensive metrics for the dynamic batcher.
    Metrics categories:
    1. Throughput: requests/sec, batches/sec
    2. Latency: queue wait time, inference time, total time
    3. Efficiency: average batch size, padding overhead
    4. Health: queue depth, rejection rate
    """
    # Sliding windows for percentile calculation
    window_size: int = 1000
    # Latency tracking
    queue_wait_times_ms: deque = field(default_factory=lambda: deque(maxlen=1000))
    inference_times_ms: deque = field(default_factory=lambda: deque(maxlen=1000))
    total_times_ms: deque = field(default_factory=lambda: deque(maxlen=1000))
    # Batch tracking
    batch_sizes: deque = field(default_factory=lambda: deque(maxlen=1000))
    # Counters
    total_requests: int = 0
    total_batches: int = 0
    total_rejected: int = 0
    total_timeouts: int = 0
    def record_request(
        self,
        queue_wait_ms: float,
        inference_ms: float,
        total_ms: float
    ):
        """Record latency metrics for a completed request."""
        self.queue_wait_times_ms.append(queue_wait_ms)
        self.inference_times_ms.append(inference_ms)
        self.total_times_ms.append(total_ms)
        self.total_requests += 1
    def record_batch(self, batch_size: int):
        """Record batch formation metrics."""
        self.batch_sizes.append(batch_size)
        self.total_batches += 1
    def record_rejection(self):
        """Record a rejected request (backpressure)."""
        self.total_rejected += 1
    def record_timeout(self):
        """Record a timed-out request."""
        self.total_timeouts += 1
    def percentile(self, data: deque, p: float) -> float:
        """Calculate percentile from deque."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]
    def summary(self) -> dict:
        """Generate metrics summary for Prometheus export."""
        def safe_mean(data):
            return statistics.mean(data) if data else 0.0
        return {
            # Throughput
            "batcher_requests_total": self.total_requests,
            "batcher_batches_total": self.total_batches,
            "batcher_rejected_total": self.total_rejected,
            "batcher_timeouts_total": self.total_timeouts,
            # Latency percentiles
            "batcher_queue_wait_ms_p50": self.percentile(self.queue_wait_times_ms, 50),
            "batcher_queue_wait_ms_p95": self.percentile(self.queue_wait_times_ms, 95),
            "batcher_queue_wait_ms_p99": self.percentile(self.queue_wait_times_ms, 99),
            "batcher_inference_ms_p50": self.percentile(self.inference_times_ms, 50),
            "batcher_inference_ms_p95": self.percentile(self.inference_times_ms, 95),
            "batcher_inference_ms_p99": self.percentile(self.inference_times_ms, 99),
            "batcher_total_ms_p50": self.percentile(self.total_times_ms, 50),
            "batcher_total_ms_p95": self.percentile(self.total_times_ms, 95),
            "batcher_total_ms_p99": self.percentile(self.total_times_ms, 99),
            # Batch efficiency
            "batcher_avg_batch_size": safe_mean(self.batch_sizes),
            "batcher_max_batch_size": max(self.batch_sizes) if self.batch_sizes else 0,
            "batcher_min_batch_size": min(self.batch_sizes) if self.batch_sizes else 0,
            # Rates (would need time window for accurate RPS)
            "batcher_rejection_rate": (
                self.total_rejected / max(1, self.total_requests + self.total_rejected)
            ),
        }
    def prometheus_format(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        for name, value in self.summary().items():
            # Prometheus convention: _total for counters, no suffix for gauges
            lines.append(f"{name} {value}")
        return "\n".join(lines)
# Enhanced batcher with metrics
class InstrumentedBatcher(DynamicBatcher):
    """Batcher with comprehensive metrics collection."""
    def __init__(self, config: BatcherConfig, model_fn, device):
        super().__init__(config, model_fn, device)
        self.metrics = BatcherMetrics()
    async def submit(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Submit with metrics tracking."""
        arrival_time = time.perf_counter()
        try:
            result = await super().submit(input_tensor)
            # Calculate latencies
            total_time_ms = (time.perf_counter() - arrival_time) * 1000
            # Queue wait would be tracked in request object
            # For now, approximate
            self.metrics.record_request(
                queue_wait_ms=0,  # Would track in BatchRequest
                inference_ms=0,   # Would track in _batch_loop
                total_ms=total_time_ms
            )
            return result
        except RuntimeError as e:
            if "backpressure" in str(e).lower():
                self.metrics.record_rejection()
            raise
```
---
## Benchmarking: Proving the Throughput Gain
The acceptance criteria require demonstrating **at least 2x throughput improvement** with batching. Here's how to measure it:
```python
import asyncio
import aiohttp
import time
from dataclasses import dataclass
from typing import List
import statistics
@dataclass
class BenchmarkResult:
    """Results from a throughput benchmark."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time_seconds: float
    requests_per_second: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
async def benchmark_sequential(
    url: str,
    payload: dict,
    n_requests: int = 100
) -> BenchmarkResult:
    """
    Benchmark sequential (one-at-a-time) inference.
    This simulates no batching - each request is independent.
    """
    latencies = []
    successful = 0
    failed = 0
    async with aiohttp.ClientSession() as session:
        start_time = time.perf_counter()
        for i in range(n_requests):
            request_start = time.perf_counter()
            try:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        successful += 1
                    else:
                        failed += 1
            except Exception:
                failed += 1
            latencies.append((time.perf_counter() - request_start) * 1000)
        total_time = time.perf_counter() - start_time
    return BenchmarkResult(
        total_requests=n_requests,
        successful_requests=successful,
        failed_requests=failed,
        total_time_seconds=total_time,
        requests_per_second=successful / total_time,
        latency_ms_p50=statistics.quantiles(latencies, n=100)[49] if latencies else 0,
        latency_ms_p95=statistics.quantiles(latencies, n=100)[94] if latencies else 0,
        latency_ms_p99=statistics.quantiles(latencies, n=100)[98] if latencies else 0
    )
async def benchmark_concurrent(
    url: str,
    payload: dict,
    n_requests: int = 100,
    concurrency: int = 10
) -> BenchmarkResult:
    """
    Benchmark concurrent inference (batching enabled).
    This sends multiple requests simultaneously, allowing the batcher
    to group them for efficient GPU utilization.
    """
    latencies = []
    successful = 0
    failed = 0
    async def single_request(session: aiohttp.ClientSession):
        nonlocal successful, failed
        request_start = time.perf_counter()
        try:
            async with session.post(url, json=payload) as resp:
                latency = (time.perf_counter() - request_start) * 1000
                latencies.append(latency)
                if resp.status == 200:
                    successful += 1
                    return True
                else:
                    failed += 1
                    return False
        except Exception:
            failed += 1
            latencies.append((time.perf_counter() - request_start) * 1000)
            return False
    async with aiohttp.ClientSession() as session:
        start_time = time.perf_counter()
        # Send requests in concurrent batches
        semaphore = asyncio.Semaphore(concurrency)
        async def bounded_request():
            async with semaphore:
                return await single_request(session)
        tasks = [bounded_request() for _ in range(n_requests)]
        await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time
    return BenchmarkResult(
        total_requests=n_requests,
        successful_requests=successful,
        failed_requests=failed,
        total_time_seconds=total_time,
        requests_per_second=successful / total_time,
        latency_ms_p50=statistics.quantiles(latencies, n=100)[49] if latencies else 0,
        latency_ms_p95=statistics.quantiles(latencies, n=100)[94] if latencies else 0,
        latency_ms_p99=statistics.quantiles(latencies, n=100)[98] if latencies else 0
    )
async def run_comparison_benchmark(
    predict_url: str,
    payload: dict,
    n_requests: int = 500
):
    """
    Compare sequential vs batched throughput.
    This proves the batching improvement.
    """
    print(f"\n{'='*60}")
    print(f"Throughput Benchmark: {n_requests} requests")
    print(f"{'='*60}\n")
    # Sequential (no batching benefit)
    print("Running sequential benchmark...")
    seq_result = await benchmark_sequential(predict_url, payload, n_requests)
    print(f"\n--- Sequential (No Batching) ---")
    print(f"  RPS: {seq_result.requests_per_second:.1f}")
    print(f"  Latency p50: {seq_result.latency_ms_p50:.1f}ms")
    print(f"  Latency p95: {seq_result.latency_ms_p95:.1f}ms")
    print(f"  Total time: {seq_result.total_time_seconds:.2f}s")
    # Concurrent (batching enabled)
    print("\nRunning concurrent benchmark (batching enabled)...")
    conc_result = await benchmark_concurrent(
        predict_url, payload, n_requests, concurrency=50
    )
    print(f"\n--- Concurrent (With Batching) ---")
    print(f"  RPS: {conc_result.requests_per_second:.1f}")
    print(f"  Latency p50: {conc_result.latency_ms_p50:.1f}ms")
    print(f"  Latency p95: {conc_result.latency_ms_p95:.1f}ms")
    print(f"  Total time: {conc_result.total_time_seconds:.2f}s")
    # Comparison
    speedup = conc_result.requests_per_second / max(1, seq_result.requests_per_second)
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Throughput improvement: {speedup:.1f}x")
    print(f"  Target: >= 2.0x")
    print(f"  Status: {'✓ PASS' if speedup >= 2.0 else '✗ FAIL'}")
    print(f"{'='*60}\n")
    return {
        "sequential_rps": seq_result.requests_per_second,
        "batched_rps": conc_result.requests_per_second,
        "speedup": speedup,
        "pass": speedup >= 2.0
    }
# Example usage:
# asyncio.run(run_comparison_benchmark(
#     "http://localhost:8000/predict",
#     {"input": [[0.1, 0.2, ...]]}  # Your model's input format
# ))
```
---
## Complete Integration: The Batched Inference Server
Here's how all the pieces fit together:
```python
"""
batched_inference_server.py
Complete dynamic batching inference server.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pt")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))
BATCH_TIMEOUT_MS = float(os.getenv("BATCH_TIMEOUT_MS", "50.0"))
MAX_QUEUE_DEPTH = int(os.getenv("MAX_QUEUE_DEPTH", "1000"))
# Global state
batcher: DynamicBatcher = None
model = None
device = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server startup and shutdown."""
    global batcher, model, device
    # === STARTUP ===
    logger.info("[Startup] Loading model...")
    # Load model (from M1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    # Warmup
    logger.info("[Startup] Warming up model...")
    dummy = torch.randn(1, 3, 224, 224, device=device)
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy)
    # Create batcher
    config = BatcherConfig(
        max_batch_size=MAX_BATCH_SIZE,
        timeout_ms=BATCH_TIMEOUT_MS,
        max_queue_depth=MAX_QUEUE_DEPTH
    )
    @torch.no_grad()
    def inference_fn(inputs: torch.Tensor) -> torch.Tensor:
        return model(inputs)
    batcher = DynamicBatcher(config, inference_fn, device)
    batcher.start()
    logger.info(
        f"[Startup] Ready: max_batch={MAX_BATCH_SIZE}, "
        f"timeout={BATCH_TIMEOUT_MS}ms"
    )
    yield  # Server runs here
    # === SHUTDOWN ===
    logger.info("[Shutdown] Stopping batcher...")
    await batcher.stop()
    logger.info("[Shutdown] Complete")
app = FastAPI(lifespan=lifespan, title="Batched Inference Server")
@app.get("/health/liveness")
async def liveness():
    """Process is alive."""
    return {"status": "alive"}
@app.get("/health/readiness")
async def readiness():
    """Server can handle requests."""
    if batcher is None:
        raise HTTPException(status_code=503, detail="Initializing")
    return {"status": "ready"}
@app.post("/predict")
async def predict(request: Request) -> Response:
    """
    Inference endpoint with dynamic batching.
    Request body:
        {"input": <tensor_data>}
    Response:
        {"prediction": <output_data>, "request_id": <id>}
    Errors:
        400: Invalid input
        503: Server overloaded (backpressure)
        504: Request timeout
    """
    # Parse request
    body = await request.body()
    try:
        import orjson
        data = orjson.loads(body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    if "input" not in data:
        raise HTTPException(status_code=400, detail="Missing 'input' field")
    # Convert to tensor
    try:
        input_tensor = torch.tensor(data["input"], dtype=torch.float32)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid tensor: {e}")
    # Submit to batcher
    try:
        result = await asyncio.wait_for(
            batcher.submit(input_tensor),
            timeout=30.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout")
    except RuntimeError as e:
        if "capacity" in str(e).lower() or "backpressure" in str(e).lower():
            raise HTTPException(status_code=503, detail="Server overloaded")
        raise HTTPException(status_code=500, detail=str(e))
    # Return response
    response_data = {
        "prediction": result.tolist(),
        "request_id": data.get("request_id")
    }
    return Response(
        content=orjson.dumps(response_data),
        media_type="application/json"
    )
@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus metrics endpoint."""
    if batcher is None:
        return "# Batcher not initialized"
    return batcher.metrics() if hasattr(batcher, 'metrics') else "# No metrics"
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
---
## Knowledge Cascade: What You've Unlocked
### Immediate Applications
**Milestone 3 (Model Versioning)**: The batcher you've built is designed around a single model function. Multi-version serving requires per-version batchers or a version-aware batcher that routes to the correct model. The queue architecture supports both.
**Milestone 4 (A/B Testing)**: Traffic splitting between model versions integrates naturally with batching. The batcher can route requests to different model instances based on user ID or random assignment, maintaining batch efficiency for each version.
### Cross-Domain Connections
**1. GPU Utilization → Database Connection Pooling**
The batching problem is isomorphic to **database connection pooling**:
- **Problem**: Creating connections is expensive; each query needs one
- **Naive**: Create connection per query (slow, resource-intensive)
- **Solution**: Pool connections, serve multiple queries from same connection
- **Batching equivalent**: Single GPU context, multiple inferences per kernel launch
Both systems optimize by **amortizing fixed costs across multiple operations**.
**2. Attention Masking → Sparse Matrix Operations**
The padding + masking technique you learned is a special case of **sparse matrix operations**:
- Dense representation: Store all values (including zeros)
- Sparse representation: Store only non-zero values + indices
- Attention mask: Dense storage but sparse computation (zeros don't contribute)
Modern transformers use **sparse attention patterns** (Longformer, BigBird) to handle longer sequences—the same principle applied more aggressively.
**3. Backpressure → Load Shedding in Distributed Systems**
The 503-on-full-queue pattern appears throughout distributed systems:
- **Web servers**: Drop requests when worker pool exhausted
- **Message queues**: Reject publishes when at capacity
- **Circuit breakers**: Fail fast when downstream unhealthy
The principle is universal: **preserve system stability by rejecting work you can't handle, rather than accepting it and failing slowly**.
### Forward Connections
With dynamic batching mastered, you can now:
- **Build adaptive batchers** that tune timeout and max batch size based on load
- **Implement priority queuing** for premium vs standard requests
- **Create multi-model batchers** that share GPU memory across models
- **Design inference cascades** where simple models filter requests before expensive models
---
## Summary: The Batching Mindset
Dynamic batching is not an optimization—it's a **necessity** for production GPU inference. The key insights:
1. **GPUs are throughput machines**, not latency machines. Single-request inference wastes 90%+ of GPU capacity.
2. **Kernel launch overhead dominates** at small batch sizes. Batching amortizes this overhead across multiple requests.
3. **Latency-throughput tradeoff is fundamental**. You choose where to sit on the curve via batch size and timeout.
4. **Async queuing bridges worlds**. HTTP handlers await results while the batcher collects and processes in the background.
5. **Backpressure prevents cascading failures**. Rejected requests (503) are better than timed-out requests (504) and OOM crashes.
6. **Metrics make it observable**. You can't tune what you can't measure.
You now have the tools to serve ML models at production scale—efficiently, reliably, and observably.
---
[[CRITERIA_JSON: {"milestone_id": "ml-model-serving-m2", "criteria": ["Incoming requests are queued via asyncio.Queue and grouped into batches up to configurable max_batch_size (default 32)", "Maximum wait timeout (default 50ms) flushes partial batches to bound worst-case latency, using asyncio.wait_for for first request then draining queue", "Variable-length inputs are padded to the longest sequence in batch with attention masks (1 for real tokens, 0 for padding)", "Batch inference results are correctly un-batched by index and routed to each original requester via asyncio.Future resolution", "Backpressure rejects new requests with HTTP 503 when queue depth exceeds configurable max_queue_depth limit", "Metrics exposed include: average batch size, queue depth, queue wait time, inference latency percentiles (p50/p95/p99), and requests per second", "Throughput benchmark demonstrates at least 2x RPS improvement for batched concurrent serving vs sequential serving on GPU", "Batch formation loop runs as background asyncio task independent of HTTP request handlers", "Input tensor shape validation rejects mismatched dimensions with HTTP 400 and descriptive error message", "Request lifecycle tracked from arrival through queue wait, inference, and response delivery for latency breakdown"]}]
<!-- END_MS -->


<!-- MS_ID: ml-model-serving-m3 -->
# Model Versioning & Hot Reload
## The Deployment Illusion
You've built a model server that handles dynamic batching, serves predictions efficiently, and exposes detailed metrics. It's production-ready—until you need to update the model.
The naive deployment looks like this:
```python
# Deploy new model version
kubectl rollout restart deployment/model-server
# Wait for pods to terminate and restart
# Traffic drops to zero during the restart window
```
**This works.** For a system with zero availability requirements. But production systems have SLAs—99.9% availability means you can only afford 43 minutes of downtime per month. A 30-second restart window during every model update burns through that budget fast.

![Hot Swap Sequence Diagram](./diagrams/diag-m3-hot-swap-sequence.svg)

Here's what actually happens during a naive restart:
1. **Pod termination signal sent** (SIGTERM)
2. **New pod starts** (pulls image, initializes)
3. **Model loads** (5-30 seconds depending on size)
4. **CUDA warmup** (another 1-5 seconds)
5. **Readiness probe passes** (traffic resumes)
During steps 2-5, your serving capacity is reduced. If you had 3 pods and one restarts, you're at 66% capacity. If traffic spikes during this window, latency degrades or requests fail. And if the new model has a bug? You're now serving broken predictions until you detect it and roll back.
**The misconception**: "Users will tolerate a few seconds of downtime during updates."
**The reality**: Production systems require **zero-downtime updates** with **instant rollback**. This isn't optional—it's table stakes for any system serving real users.
---
## The Fundamental Tension: Memory vs. Availability
Every hot-swap system faces a core constraint:
**You want zero-downtime updates, but you're constrained by:**
1. **Memory capacity**: Loading a new model requires memory; the old model still occupies memory until drained
2. **Request draining**: In-flight requests to the old model must complete before it's unloaded
3. **Atomic switching**: Traffic must switch to the new model instantaneously, never serving from both simultaneously for the same request
4. **Rollback speed**: If the new model fails, reverting must be instant—not a 30-second reload

![Memory Budget During Hot Swap](./diagrams/diag-m3-memory-budget.svg)

The memory constraint is the hardest. During a hot swap, **peak memory = old model + new model + activation buffers**. If your model is 2GB and you have 8GB GPU memory, you need:
- Old model: 2GB
- New model: 2GB
- Activation buffers: ~4GB (during inference)
- **Total: 8GB** — exactly your capacity, with zero margin
This is why hot-swapping requires careful **memory budgeting**. You can't just "load the new model and hope for the best."
---
## The Three-Level View
**Level 1 — Model Registry (Application Layer)**:
- Stores version metadata: accuracy, training date, data hash, schema
- Tracks which version is "active" vs "staged"
- Validates schema compatibility before promotion
**Level 2 — Hot Swap Engine (Concurrency Layer)**:
- Manages atomic pointer swaps between model versions
- Tracks in-flight request counts per version
- Coordinates draining with garbage collection
**Level 3 — Memory & GPU (Resource Layer)**:
- Allocates GPU memory for new model versions
- Tracks memory pressure during dual-load windows
- Triggers garbage collection after successful swap
---
## Phase 1: The Model Registry
### Version as a First-Class Concept
A model isn't just a file—it's a **versioned artifact with metadata**. The registry stores everything needed to make deployment decisions:
```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import hashlib
import json
class VersionStatus(Enum):
    """Lifecycle states for a model version."""
    STAGED = "staged"        # Loaded in memory, not serving traffic
    ACTIVE = "active"        # Currently serving traffic
    DRAINING = "draining"    # Was active, waiting for in-flight requests
    DEPRECATED = "deprecated"  # No longer in memory, metadata retained
    FAILED = "failed"        # Failed to load or validate
@dataclass
class ModelSchema:
    """
    Input/output schema definition for a model version.
    This is the API contract—breaking changes are deployment blockers.
    """
    input_shapes: Dict[str, Tuple[int, ...]]  # name -> shape (None for variable dims)
    input_dtypes: Dict[str, str]              # name -> dtype string
    output_shapes: Dict[str, Tuple[int, ...]]
    output_dtypes: Dict[str, str]
    def to_dict(self) -> dict:
        return {
            "input_shapes": {k: list(v) for k, v in self.input_shapes.items()},
            "input_dtypes": self.input_dtypes,
            "output_shapes": {k: list(v) for k, v in self.output_shapes.items()},
            "output_dtypes": self.output_dtypes,
        }
    @classmethod
    def from_dict(cls, data: dict) -> "ModelSchema":
        return cls(
            input_shapes={k: tuple(v) for k, v in data["input_shapes"].items()},
            input_dtypes=data["input_dtypes"],
            output_shapes={k: tuple(v) for k, v in data["output_shapes"].items()},
            output_dtypes=data["output_dtypes"],
        )
    def is_compatible_with(self, other: "ModelSchema") -> Tuple[bool, Optional[str]]:
        """
        Check if this schema is backward-compatible with another.
        A new version can accept the same inputs and produce compatible outputs.
        Compatibility rules:
        1. Input shapes: new version can be MORE permissive (accept larger ranges)
        2. Input dtypes: must match exactly
        3. Output shapes: new version can have SMALLER outputs (compatible)
        4. Output dtypes: must match exactly
        Returns:
            (is_compatible, error_message)
        """
        # Check input compatibility
        for name, shape in self.input_shapes.items():
            if name not in other.input_shapes:
                return False, f"New version missing input '{name}'"
            other_shape = other.input_shapes[name]
            # Rank must match
            if len(shape) != len(other_shape):
                return False, (
                    f"Input '{name}' rank mismatch: "
                    f"expected {len(shape)}D, new version has {len(other_shape)}D"
                )
            # Dtypes must match
            if self.input_dtypes.get(name) != other.input_dtypes.get(name):
                return False, (
                    f"Input '{name}' dtype mismatch: "
                    f"expected {self.input_dtypes.get(name)}, "
                    f"new version has {other.input_dtypes.get(name)}"
                )
        # Check output compatibility (clients depend on these shapes)
        for name, shape in self.output_shapes.items():
            if name not in other.output_shapes:
                return False, f"New version missing output '{name}'"
            other_shape = other.output_shapes[name]
            # Output rank must match
            if len(shape) != len(other_shape):
                return False, (
                    f"Output '{name}' rank mismatch: "
                    f"expected {len(shape)}D, new version has {len(other_shape)}D"
                )
            # Output dtypes must match
            if self.output_dtypes.get(name) != other.output_dtypes.get(name):
                return False, (
                    f"Output '{name}' dtype mismatch: "
                    f"expected {self.output_dtypes.get(name)}, "
                    f"new version has {other.output_dtypes.get(name)}"
                )
        return True, None
@dataclass
class ModelVersion:
    """
    Complete metadata for a single model version.
    This is the unit of deployment in the registry.
    """
    version_id: str                          # e.g., "v1.2.3" or "2024-01-15-a"
    model_path: str                          # Path to model file
    status: VersionStatus = VersionStatus.STAGED
    # Training metadata
    training_date: Optional[datetime] = None
    training_data_hash: Optional[str] = None  # Hash of training dataset
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)  # e.g., {"accuracy": 0.95, "f1": 0.92}
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    # Schema (the API contract)
    schema: Optional[ModelSchema] = None
    # Runtime state
    loaded_model: Any = None  # The actual model object (set when loaded)
    device: Optional[str] = None
    load_time: Optional[datetime] = None
    warmup_latency_ms: Optional[float] = None
    # Request tracking for draining
    in_flight_count: int = 0
    def to_metadata_dict(self) -> dict:
        """Export metadata (without the loaded model object)."""
        return {
            "version_id": self.version_id,
            "model_path": self.model_path,
            "status": self.status.value,
            "training_date": self.training_date.isoformat() if self.training_date else None,
            "training_data_hash": self.training_data_hash,
            "accuracy_metrics": self.accuracy_metrics,
            "hyperparameters": self.hyperparameters,
            "schema": self.schema.to_dict() if self.schema else None,
            "device": self.device,
            "load_time": self.load_time.isoformat() if self.load_time else None,
            "warmup_latency_ms": self.warmup_latency_ms,
        }
    def compute_data_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of a file for integrity verification."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]  # First 16 chars is enough for identity
```

![Model Registry Structure](./diagrams/diag-m3-version-registry.svg)

### The Registry Structure
The registry is the source of truth for all model versions:
```python
import asyncio
from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)
class ModelRegistry:
    """
    Central registry for model versions with lifecycle management.
    Key responsibilities:
    1. Store version metadata
    2. Track active vs staged versions
    3. Validate schema compatibility before promotion
    4. Coordinate loading/unloading with memory constraints
    Thread safety: All public methods are safe for concurrent access.
    """
    def __init__(self, max_versions_in_memory: int = 3):
        self._versions: OrderedDict[str, ModelVersion] = OrderedDict()
        self._active_version_id: Optional[str] = None
        self._lock = asyncio.Lock()  # Protects all state mutations
        self._max_versions_in_memory = max_versions_in_memory
        # Callbacks for lifecycle events
        self._on_version_loaded: Optional[callable] = None
        self._on_version_activated: Optional[callable] = None
        self._on_version_drained: Optional[callable] = None
    async def register_version(
        self,
        version: ModelVersion,
        auto_load: bool = False
    ) -> bool:
        """
        Register a new model version in the registry.
        Args:
            version: ModelVersion with metadata populated
            auto_load: If True, load the model immediately
        Returns:
            True if registered successfully
        """
        async with self._lock:
            if version.version_id in self._versions:
                logger.warning(f"Version {version.version_id} already registered")
                return False
            self._versions[version.version_id] = version
            logger.info(f"[Registry] Registered version {version.version_id}")
            if auto_load:
                await self._load_version_internal(version.version_id)
            return True
    async def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get version metadata by ID."""
        async with self._lock:
            return self._versions.get(version_id)
    async def get_active_version(self) -> Optional[ModelVersion]:
        """Get the currently active version."""
        async with self._lock:
            if self._active_version_id:
                return self._versions.get(self._active_version_id)
            return None
    async def list_versions(self) -> List[dict]:
        """List all registered versions with metadata."""
        async with self._lock:
            return [
                v.to_metadata_dict()
                for v in self._versions.values()
            ]
    async def validate_compatibility(
        self,
        new_version_id: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that a new version is compatible with the active version.
        If no active version exists, any schema is accepted.
        Returns:
            (is_compatible, error_message)
        """
        async with self._lock:
            new_version = self._versions.get(new_version_id)
            if not new_version:
                return False, f"Version {new_version_id} not found"
            if not new_version.schema:
                return False, f"Version {new_version_id} has no schema defined"
            # If no active version, any schema is fine
            if not self._active_version_id:
                return True, None
            active_version = self._versions.get(self._active_version_id)
            if not active_version or not active_version.schema:
                return True, None
            return active_version.schema.is_compatible_with(new_version.schema)
    async def _load_version_internal(self, version_id: str) -> bool:
        """Internal load method (must be called with lock held)."""
        version = self._versions.get(version_id)
        if not version:
            return False
        if version.loaded_model is not None:
            logger.info(f"[Registry] Version {version_id} already loaded")
            return True
        # Actual loading would be delegated to a model loader
        # This is a hook for the hot swap system
        logger.info(f"[Registry] Loading version {version_id}...")
        return True
    async def _unload_version_internal(self, version_id: str) -> bool:
        """Internal unload method (must be called with lock held)."""
        version = self._versions.get(version_id)
        if not version:
            return False
        if version.loaded_model is None:
            return True
        # Check for in-flight requests
        if version.in_flight_count > 0:
            logger.warning(
                f"[Registry] Cannot unload {version_id}: "
                f"{version.in_flight_count} requests in flight"
            )
            return False
        # Actual unloading would happen here
        logger.info(f"[Registry] Unloading version {version_id}")
        version.loaded_model = None
        version.status = VersionStatus.DEPRECATED
        return True
    def set_callbacks(
        self,
        on_load: Optional[callable] = None,
        on_activate: Optional[callable] = None,
        on_drain: Optional[callable] = None,
    ):
        """Set lifecycle callbacks for external coordination."""
        self._on_version_loaded = on_load
        self._on_version_activated = on_activate
        self._on_version_drained = on_drain
```
---
## Phase 2: Atomic Pointer Swapping
### The Concurrency Challenge
> **🔑 Foundation: Atomic operations for concurrent state**
> 
> ## Atomic Operations for Concurrent State
### What It IS
An atomic operation is one that appears indivisible—either it completes entirely, or it doesn't happen at all. No intermediate state is visible to other threads. In Python, the `asyncio.Lock` and thread-safe operations provide this guarantee for cooperative multitasking.
### WHY You Need It Right Now
When switching model versions, you face a race condition: what if a request arrives *exactly* when you're swapping the active version pointer? Without atomicity:
- Request could see old version for input validation, new version for inference
- Request could get half-initialized model state
- Metrics could attribute to wrong version
Atomic pointer swaps ensure every request sees exactly one consistent version.
### ONE Key Insight
**Think of it like a train track switch.** A train (request) is moving at speed. You need to switch the track (model version) so that:
1. The train never derails (sees inconsistent state)
2. The train never goes to the wrong destination (gets wrong model)
3. The switch happens in a single instant (atomic)
The solution: **Switch only when no train is on the track section.** In code, this means the swap happens atomically between requests, never during one.

![Atomic Pointer Swap Mechanism](./diagrams/diag-m3-atomic-pointer-swap.svg)

The hot swap must be **atomic**—every request after the swap sees the new model, every request before sees the old model. There's no middle ground.
```python
class AtomicVersionPointer:
    """
    Thread-safe pointer to the active model version.
    This uses asyncio synchronization to ensure atomic swaps.
    The key insight: reading the pointer is lock-free (fast path),
    but swapping requires exclusive access.
    """
    def __init__(self):
        self._version_id: Optional[str] = None
        self._version: Optional[ModelVersion] = None
        self._lock = asyncio.Lock()
        # Reference counting for in-flight requests
        self._ref_counts: Dict[str, int] = {}
    async def get(self) -> Optional[ModelVersion]:
        """
        Get the current active version.
        This is the fast path—no lock needed for read.
        """
        return self._version
    async def swap(
        self,
        new_version: ModelVersion
    ) -> Optional[ModelVersion]:
        """
        Atomically swap the active version.
        Returns:
            The previous active version (for draining)
        """
        async with self._lock:
            old_version = self._version
            old_id = self._version_id
            # Atomic swap
            self._version = new_version
            self._version_id = new_version.version_id
            # Initialize ref count for new version
            if new_version.version_id not in self._ref_counts:
                self._ref_counts[new_version.version_id] = 0
            logger.info(
                f"[AtomicPointer] Swapped: {old_id} -> {new_version.version_id}"
            )
            return old_version
    async def increment_ref(self, version_id: str) -> None:
        """Increment in-flight request count for a version."""
        async with self._lock:
            if version_id in self._ref_counts:
                self._ref_counts[version_id] += 1
    async def decrement_ref(self, version_id: str) -> int:
        """
        Decrement in-flight request count for a version.
        Returns the new count.
        """
        async with self._lock:
            if version_id in self._ref_counts:
                self._ref_counts[version_id] -= 1
                return self._ref_counts[version_id]
            return 0
    async def get_ref_count(self, version_id: str) -> int:
        """Get current in-flight count for a version."""
        async with self._lock:
            return self._ref_counts.get(version_id, 0)
```
### Cross-Domain Connection: Database Connection Pooling
This atomic pointer swap pattern is identical to **database connection pool rolling updates**:
- Old pool continues serving existing connections
- New pool is created alongside
- New connections go to new pool
- Old pool drains as connections close
- Old pool is destroyed when empty
The pattern: **create new, switch traffic, drain old, destroy old.** It appears in every system that needs zero-downtime updates.
---
## Phase 3: Request Draining
### The In-Flight Problem
When you swap the active version, requests may already be in progress:
- Request A arrived at t=0, using v1.0
- Swap happens at t=1
- Request A completes at t=2
Request A must complete using v1.0—you can't switch its model mid-inference. This is **request draining**: waiting for all in-flight requests to complete before unloading the old model.

![Request Draining During Hot Swap](./diagrams/diag-m3-request-draining.svg)

```python
import asyncio
from typing import Optional
import time
class RequestTracker:
    """
    Tracks in-flight requests per model version.
    Used for:
    1. Draining: wait for count to reach zero before unload
    2. Metrics: track concurrent request load
    3. Health: detect stuck requests
    """
    def __init__(self, drain_timeout_seconds: float = 30.0):
        self._counts: Dict[str, int] = {}
        self._drain_events: Dict[str, asyncio.Event] = {}
        self._drain_timeout = drain_timeout_seconds
        self._lock = asyncio.Lock()
    async def acquire(self, version_id: str) -> None:
        """
        Register a new request for a version.
        Call this before starting inference.
        """
        async with self._lock:
            if version_id not in self._counts:
                self._counts[version_id] = 0
            self._counts[version_id] += 1
            logger.debug(f"[Tracker] Acquired {version_id}: count={self._counts[version_id]}")
    async def release(self, version_id: str) -> int:
        """
        Release a request for a version.
        Call this after inference completes (success or error).
        Returns:
            New count for this version
        """
        async with self._lock:
            if version_id in self._counts:
                self._counts[version_id] -= 1
                count = self._counts[version_id]
                # If draining and count hit zero, signal the event
                if count == 0 and version_id in self._drain_events:
                    self._drain_events[version_id].set()
                logger.debug(f"[Tracker] Released {version_id}: count={count}")
                return count
            return 0
    async def get_count(self, version_id: str) -> int:
        """Get current in-flight count for a version."""
        async with self._lock:
            return self._counts.get(version_id, 0)
    async def start_drain(self, version_id: str) -> None:
        """
        Start draining a version.
        Creates an event that will be set when count reaches zero.
        """
        async with self._lock:
            if version_id not in self._drain_events:
                self._drain_events[version_id] = asyncio.Event()
            # If already zero, signal immediately
            if self._counts.get(version_id, 0) == 0:
                self._drain_events[version_id].set()
            logger.info(f"[Tracker] Started draining {version_id}")
    async def wait_for_drain(self, version_id: str) -> bool:
        """
        Wait for all in-flight requests to complete.
        Returns:
            True if drained successfully, False if timed out
        """
        async with self._lock:
            event = self._drain_events.get(version_id)
            if not event:
                # No drain in progress
                return self._counts.get(version_id, 0) == 0
        try:
            await asyncio.wait_for(
                event.wait(),
                timeout=self._drain_timeout
            )
            logger.info(f"[Tracker] Drained {version_id}")
            return True
        except asyncio.TimeoutError:
            count = self._counts.get(version_id, 0)
            logger.warning(
                f"[Tracker] Drain timeout for {version_id}: "
                f"{count} requests still in flight"
            )
            return False
    async def force_drain(self, version_id: str) -> None:
        """
        Force drain completion (for shutdown).
        Use only when you're going to unload regardless.
        """
        async with self._lock:
            if version_id in self._drain_events:
                self._drain_events[version_id].set()
            self._counts[version_id] = 0
```
### The Draining Workflow
Here's how draining integrates with the hot swap:
```python
class HotSwapManager:
    """
    Orchestrates zero-downtime model version swaps.
    The swap sequence:
    1. Load new version (in parallel with serving)
    2. Validate schema compatibility
    3. Warmup new version
    4. Atomically swap active pointer
    5. Start draining old version
    6. Wait for in-flight requests to complete
    7. Unload old version
    """
    def __init__(
        self,
        registry: ModelRegistry,
        pointer: AtomicVersionPointer,
        tracker: RequestTracker,
        model_loader: callable,
    ):
        self.registry = registry
        self.pointer = pointer
        self.tracker = tracker
        self.model_loader = model_loader
        # Previous version kept for instant rollback
        self._previous_version: Optional[ModelVersion] = None
        self._swap_lock = asyncio.Lock()
    async def hot_swap(
        self,
        new_version_id: str,
        validate_schema: bool = True,
        keep_previous: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """
        Perform a hot swap to a new model version.
        Args:
            new_version_id: Version to activate
            validate_schema: Check compatibility before swap
            keep_previous: Keep old version in memory for rollback
        Returns:
            (success, error_message)
        """
        async with self._swap_lock:
            logger.info(f"[HotSwap] Starting swap to {new_version_id}")
            swap_start = time.perf_counter()
            # Step 1: Get the new version
            new_version = await self.registry.get_version(new_version_id)
            if not new_version:
                return False, f"Version {new_version_id} not found"
            # Step 2: Validate schema compatibility
            if validate_schema:
                is_compatible, error = await self.registry.validate_compatibility(
                    new_version_id
                )
                if not is_compatible:
                    return False, f"Schema incompatible: {error}"
            # Step 3: Load the new model (if not already loaded)
            if new_version.loaded_model is None:
                try:
                    new_version = await self.model_loader(new_version)
                    logger.info(f"[HotSwap] Loaded {new_version_id}")
                except Exception as e:
                    return False, f"Failed to load model: {e}"
            # Step 4: Get current active version (for draining)
            old_version = await self.pointer.get()
            old_version_id = old_version.version_id if old_version else None
            # Step 5: Atomically swap
            await self.pointer.swap(new_version)
            new_version.status = VersionStatus.ACTIVE
            # Update registry
            async with self.registry._lock:
                self.registry._active_version_id = new_version_id
            # Step 6: Start draining old version
            if old_version:
                old_version.status = VersionStatus.DRAINING
                await self.tracker.start_drain(old_version_id)
                logger.info(
                    f"[HotSwap] Swapped {old_version_id} -> {new_version_id}, "
                    f"draining old version"
                )
            # Step 7: Keep previous for rollback (optional)
            if keep_previous and old_version:
                self._previous_version = old_version
            swap_time = (time.perf_counter() - swap_start) * 1000
            logger.info(f"[HotSwap] Complete in {swap_time:.1f}ms")
            return True, None
    async def complete_drain_and_unload(
        self,
        version_id: str,
        force: bool = False
    ) -> bool:
        """
        Wait for drain and unload a version.
        Args:
            version_id: Version to unload
            force: If True, don't wait for drain
        Returns:
            True if unloaded successfully
        """
        if not force:
            drained = await self.tracker.wait_for_drain(version_id)
            if not drained:
                logger.warning(f"[HotSwap] Drain incomplete, forcing unload")
        # Unload from memory
        version = await self.registry.get_version(version_id)
        if version and version.loaded_model is not None:
            # Clean up GPU memory
            import torch
            if torch.cuda.is_available():
                del version.loaded_model
                torch.cuda.empty_cache()
            version.loaded_model = None
            version.status = VersionStatus.DEPRECATED
            logger.info(f"[HotSwap] Unloaded {version_id}")
        return True
```
---
## Phase 4: Schema Compatibility Validation
### The API Contract Problem
Models have input/output schemas just like APIs have request/response types. A breaking schema change—like changing output dimension from 768 to 512—breaks downstream consumers.

![Hot Swap Sequence](./diagrams/tdd-diag-019.svg)

![Schema Compatibility Validation](./diagrams/diag-m3-schema-compatibility.svg)

```python
class SchemaValidator:
    """
    Validates schema compatibility between model versions.
    Compatibility rules follow semantic versioning principles:
    - MAJOR: Breaking changes (reject)
    - MINOR: Additive changes (accept)
    - PATCH: No API changes (accept)
    """
    @staticmethod
    def check_input_compatibility(
        old_schema: ModelSchema,
        new_schema: ModelSchema
    ) -> Tuple[bool, List[str]]:
        """
        Check if new version can accept inputs designed for old version.
        Compatible if:
        - All old inputs exist in new schema
        - Input dtypes match
        - Input shapes are compatible (new can be more permissive)
        Returns:
            (is_compatible, list_of_issues)
        """
        issues = []
        for input_name, old_shape in old_schema.input_shapes.items():
            # Missing input = breaking change
            if input_name not in new_schema.input_shapes:
                issues.append(f"Missing input: {input_name}")
                continue
            new_shape = new_schema.input_shapes[input_name]
            # Dtype mismatch = breaking change
            old_dtype = old_schema.input_dtypes.get(input_name)
            new_dtype = new_schema.input_dtypes.get(input_name)
            if old_dtype != new_dtype:
                issues.append(
                    f"Input '{input_name}' dtype changed: "
                    f"{old_dtype} -> {new_dtype}"
                )
            # Shape rank change = breaking change
            if len(old_shape) != len(new_shape):
                issues.append(
                    f"Input '{input_name}' rank changed: "
                    f"{len(old_shape)}D -> {len(new_shape)}D"
                )
                continue
            # Shape dimension change: new can be LARGER (more permissive)
            # but not SMALLER (would reject valid inputs)
            for i, (old_dim, new_dim) in enumerate(zip(old_shape, new_shape)):
                if old_dim is not None and new_dim is not None:
                    if new_dim < old_dim:
                        issues.append(
                            f"Input '{input_name}' dim {i} shrunk: "
                            f"{old_dim} -> {new_dim}"
                        )
        return len(issues) == 0, issues
    @staticmethod
    def check_output_compatibility(
        old_schema: ModelSchema,
        new_schema: ModelSchema
    ) -> Tuple[bool, List[str]]:
        """
        Check if new version produces outputs compatible with old version.
        Compatible if:
        - All old outputs exist in new schema
        - Output dtypes match
        - Output shapes are identical (clients depend on exact shapes)
        Note: Output shapes MUST match exactly, unlike inputs.
        Clients often have fixed-size buffers for outputs.
        """
        issues = []
        for output_name, old_shape in old_schema.output_shapes.items():
            if output_name not in new_schema.output_shapes:
                issues.append(f"Missing output: {output_name}")
                continue
            new_shape = new_schema.output_shapes[output_name]
            # Dtype must match
            old_dtype = old_schema.output_dtypes.get(output_name)
            new_dtype = new_schema.output_dtypes.get(output_name)
            if old_dtype != new_dtype:
                issues.append(
                    f"Output '{output_name}' dtype changed: "
                    f"{old_dtype} -> {new_dtype}"
                )
            # Shape must match EXACTLY for outputs
            if old_shape != new_shape:
                issues.append(
                    f"Output '{output_name}' shape changed: "
                    f"{old_shape} -> {new_shape}"
                )
        return len(issues) == 0, issues
    @staticmethod
    def full_compatibility_check(
        old_schema: ModelSchema,
        new_schema: ModelSchema
    ) -> Tuple[bool, dict]:
        """
        Complete compatibility check with detailed report.
        Returns:
            (is_compatible, report_dict)
        """
        input_ok, input_issues = SchemaValidator.check_input_compatibility(
            old_schema, new_schema
        )
        output_ok, output_issues = SchemaValidator.check_output_compatibility(
            old_schema, new_schema
        )
        return (
            input_ok and output_ok,
            {
                "compatible": input_ok and output_ok,
                "input_compatible": input_ok,
                "input_issues": input_issues,
                "output_compatible": output_ok,
                "output_issues": output_issues,
            }
        )
```
### Cross-Domain Connection: API Versioning
Schema compatibility is **identical** to REST API versioning:
| API Versioning | Model Versioning |
|----------------|------------------|
| Request body schema | Input tensor shape/dtype |
| Response body schema | Output tensor shape/dtype |
| Breaking change = major version bump | Breaking change = reject deployment |
| Backward compatible = minor version bump | Compatible = allow deployment |
The principle: **clients have expectations.** Breaking those expectations breaks systems. Validate early, reject often.
---
## Phase 5: Automatic Rollback
### The Safety Net
Hot swapping introduces risk: what if the new model is broken? You need **instant rollback**—not a 30-second reload.

![Automatic Rollback Trigger](./diagrams/diag-m3-rollback-flow.svg)

The key insight: **keep the previous version in memory**. If the new version fails, swap back to the previous version atomically. This is O(1) because you're swapping a pointer, not loading a model.
```python
from dataclasses import dataclass
from collections import deque
import time
@dataclass
class ErrorSnapshot:
    """Snapshot of error metrics at a point in time."""
    timestamp: float
    error_count: int
    total_requests: int
    error_rate: float
class RollbackMonitor:
    """
    Monitors model health and triggers automatic rollback.
    Monitors:
    - Error rate (5xx responses, exceptions)
    - Latency spikes
    - Custom health checks
    Triggers rollback if:
    - Error rate exceeds threshold for N consecutive windows
    - Latency p99 exceeds threshold
    - Health check fails
    """
    def __init__(
        self,
        error_rate_threshold: float = 0.05,  # 5% error rate
        latency_p99_threshold_ms: float = 1000.0,  # 1 second
        window_size_seconds: float = 10.0,
        consecutive_windows: int = 3,
    ):
        self.error_rate_threshold = error_rate_threshold
        self.latency_p99_threshold_ms = latency_p99_threshold_ms
        self.window_size = window_size_seconds
        self.consecutive_windows = consecutive_windows
        # Per-version tracking
        self._error_history: Dict[str, deque] = {}
        self._latency_history: Dict[str, deque] = {}
        # Callback for rollback
        self._rollback_callback: Optional[callable] = None
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
    def set_rollback_callback(self, callback: callable):
        """Set the function to call when rollback is triggered."""
        self._rollback_callback = callback
    async def record_result(
        self,
        version_id: str,
        success: bool,
        latency_ms: float
    ):
        """Record a request result for monitoring."""
        if version_id not in self._error_history:
            self._error_history[version_id] = deque(maxlen=1000)
            self._latency_history[version_id] = deque(maxlen=1000)
        self._error_history[version_id].append({
            "timestamp": time.time(),
            "success": success,
        })
        self._latency_history[version_id].append({
            "timestamp": time.time(),
            "latency_ms": latency_ms,
        })
    def _compute_window_metrics(
        self,
        version_id: str,
        window_start: float
    ) -> dict:
        """Compute error rate and latency for a time window."""
        errors = [e for e in self._error_history.get(version_id, [])
                  if e["timestamp"] >= window_start]
        latencies = [l for l in self._latency_history.get(version_id, [])
                     if l["timestamp"] >= window_start]
        if not errors:
            return {"error_rate": 0.0, "latency_p99": 0.0, "count": 0}
        error_count = sum(1 for e in errors if not e["success"])
        error_rate = error_count / len(errors)
        if latencies:
            sorted_latencies = sorted(l["latency_ms"] for l in latencies)
            p99_idx = int(len(sorted_latencies) * 0.99)
            latency_p99 = sorted_latencies[p99_idx]
        else:
            latency_p99 = 0.0
        return {
            "error_rate": error_rate,
            "latency_p99": latency_p99,
            "count": len(errors),
        }
    async def check_health(self, version_id: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a version is healthy based on recent metrics.
        Returns:
            (is_healthy, reason_if_unhealthy)
        """
        window_start = time.time() - self.window_size
        metrics = self._compute_window_metrics(version_id, window_start)
        # Need minimum samples for reliable metrics
        if metrics["count"] < 10:
            return True, None  # Not enough data, assume healthy
        if metrics["error_rate"] > self.error_rate_threshold:
            return False, (
                f"Error rate {metrics['error_rate']:.1%} exceeds "
                f"threshold {self.error_rate_threshold:.1%}"
            )
        if metrics["latency_p99"] > self.latency_p99_threshold_ms:
            return False, (
                f"Latency p99 {metrics['latency_p99']:.0f}ms exceeds "
                f"threshold {self.latency_p99_threshold_ms:.0f}ms"
            )
        return True, None
    async def _monitor_loop(self):
        """Background task that monitors health and triggers rollback."""
        consecutive_failures = 0
        while self._monitoring:
            try:
                # Get current active version
                active_version = await self._get_active_version()
                if not active_version:
                    await asyncio.sleep(self.window_size)
                    continue
                is_healthy, reason = await self.check_health(active_version)
                if not is_healthy:
                    consecutive_failures += 1
                    logger.warning(
                        f"[RollbackMonitor] Health check failed "
                        f"({consecutive_failures}/{self.consecutive_windows}): {reason}"
                    )
                    if consecutive_failures >= self.consecutive_windows:
                        logger.error(
                            f"[RollbackMonitor] Triggering rollback: {reason}"
                        )
                        if self._rollback_callback:
                            await self._rollback_callback(active_version, reason)
                        consecutive_failures = 0
                else:
                    consecutive_failures = 0
                await asyncio.sleep(self.window_size)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[RollbackMonitor] Error in monitor loop: {e}")
                await asyncio.sleep(1.0)
    async def _get_active_version(self) -> Optional[str]:
        """Get current active version ID. Override with actual implementation."""
        return None  # Would be connected to registry
    def start(self):
        """Start the monitoring loop."""
        if self._monitoring:
            return
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("[RollbackMonitor] Started")
    async def stop(self):
        """Stop the monitoring loop."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("[RollbackMonitor] Stopped")
```
### Instant Rollback Implementation
```python
class RollbackCapableHotSwap(HotSwapManager):
    """
    Hot swap manager with instant rollback capability.
    Keeps the previous version in memory for O(1) rollback.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._previous_version: Optional[ModelVersion] = None
        self._rollback_monitor: Optional[RollbackMonitor] = None
    async def hot_swap(
        self,
        new_version_id: str,
        validate_schema: bool = True,
        keep_previous: bool = True,
        enable_monitoring: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """
        Hot swap with rollback protection.
        """
        # Perform the swap
        success, error = await super().hot_swap(
            new_version_id, validate_schema, keep_previous
        )
        if not success:
            return False, error
        # Store previous for instant rollback
        if keep_previous:
            current = await self.pointer.get()
            if current and current.version_id != new_version_id:
                self._previous_version = current
        # Start monitoring if enabled
        if enable_monitoring and not self._rollback_monitor:
            self._rollback_monitor = RollbackMonitor()
            self._rollback_monitor.set_rollback_callback(self._trigger_rollback)
            self._rollback_monitor.start()
        return True, None
    async def _trigger_rollback(
        self,
        failed_version_id: str,
        reason: str
    ) -> bool:
        """
        Trigger instant rollback to previous version.
        This is O(1) because the previous version is still in memory.
        Target: complete within 5 seconds.
        """
        rollback_start = time.perf_counter()
        if not self._previous_version:
            logger.error(
                f"[Rollback] Cannot rollback: no previous version available"
            )
            return False
        previous_id = self._previous_version.version_id
        logger.warning(
            f"[Rollback] Initiating rollback from {failed_version_id} "
            f"to {previous_id}: {reason}"
        )
        # Atomic swap (instant)
        await self.pointer.swap(self._previous_version)
        # Update registry
        async with self.registry._lock:
            self.registry._active_version_id = previous_id
        self._previous_version.status = VersionStatus.ACTIVE
        rollback_time = (time.perf_counter() - rollback_start) * 1000
        logger.info(
            f"[Rollback] Complete in {rollback_time:.1f}ms "
            f"(target: <5000ms)"
        )
        # Mark failed version
        failed_version = await self.registry.get_version(failed_version_id)
        if failed_version:
            failed_version.status = VersionStatus.FAILED
        return True
    async def manual_rollback(self) -> Tuple[bool, Optional[str]]:
        """
        Manually trigger rollback to previous version.
        Returns:
            (success, error_message)
        """
        if not self._previous_version:
            return False, "No previous version available for rollback"
        previous_id = self._previous_version.version_id
        return await self._trigger_rollback(
            (await self.pointer.get()).version_id,
            "Manual rollback requested"
        )
```
---
## Phase 6: Version-Specific Routing
### URL Path Routing
Sometimes you need to serve multiple versions simultaneously—for gradual migration, A/B testing, or backward compatibility:
```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
import orjson
class VersionRouter:
    """
    Routes requests to specific model versions.
    Routing strategies:
    1. URL path: /v1/predict, /v2/predict
    2. Header: X-Model-Version: v1.2.3
    3. Default: route to active version
    """
    def __init__(
        self,
        registry: ModelRegistry,
        pointer: AtomicVersionPointer,
        tracker: RequestTracker,
    ):
        self.registry = registry
        self.pointer = pointer
        self.tracker = tracker
    async def route_request(
        self,
        request: Request,
        requested_version: Optional[str] = None
    ) -> ModelVersion:
        """
        Determine which version should handle this request.
        Priority:
        1. Explicit version in URL path
        2. Version in X-Model-Version header
        3. Active version (default)
        Raises:
            HTTPException: If version not found or not loaded
        """
        # Check for explicit version
        if requested_version:
            version = await self.registry.get_version(requested_version)
            if not version:
                raise HTTPException(
                    status_code=404,
                    detail=f"Version {requested_version} not found"
                )
            if version.loaded_model is None:
                raise HTTPException(
                    status_code=503,
                    detail=f"Version {requested_version} not loaded"
                )
            return version
        # Check header
        header_version = request.headers.get("X-Model-Version")
        if header_version:
            version = await self.registry.get_version(header_version)
            if not version:
                raise HTTPException(
                    status_code=404,
                    detail=f"Version {header_version} not found"
                )
            if version.loaded_model is None:
                raise HTTPException(
                    status_code=503,
                    detail=f"Version {header_version} not loaded"
                )
            return version
        # Default to active version
        active = await self.pointer.get()
        if not active:
            raise HTTPException(
                status_code=503,
                detail="No active model version"
            )
        return active
    async def execute_with_tracking(
        self,
        version: ModelVersion,
        inference_fn: callable,
        *args,
        **kwargs
    ):
        """
        Execute inference with request tracking for proper draining.
        """
        version_id = version.version_id
        # Track this request
        await self.tracker.acquire(version_id)
        try:
            result = await inference_fn(version, *args, **kwargs)
            return result
        finally:
            await self.tracker.release(version_id)
def create_versioned_app(
    registry: ModelRegistry,
    pointer: AtomicVersionPointer,
    tracker: RequestTracker,
    inference_handler: callable,
) -> FastAPI:
    """
    Create a FastAPI app with versioned endpoints.
    """
    app = FastAPI(title="Versioned Model Server")
    router = VersionRouter(registry, pointer, tracker)
    @app.get("/health/liveness")
    async def liveness():
        return {"status": "alive"}
    @app.get("/health/readiness")
    async def readiness():
        active = await pointer.get()
        if not active:
            raise HTTPException(status_code=503, detail="No active version")
        return {"status": "ready", "active_version": active.version_id}
    @app.get("/versions")
    async def list_versions():
        versions = await registry.list_versions()
        return Response(
            content=orjson.dumps(versions),
            media_type="application/json"
        )
    @app.post("/predict")
    async def predict_default(request: Request):
        """Default prediction endpoint (active version)."""
        version = await router.route_request(request)
        return await router.execute_with_tracking(
            version, inference_handler, request
        )
    @app.post("/v/{version_id}/predict")
    async def predict_versioned(version_id: str, request: Request):
        """Version-specific prediction endpoint."""
        version = await router.route_request(request, version_id)
        return await router.execute_with_tracking(
            version, inference_handler, request
        )
    @app.post("/admin/swap/{version_id}")
    async def swap_version(version_id: str):
        """Admin endpoint to swap active version."""
        # Would integrate with HotSwapManager
        return {"message": f"Swap to {version_id} initiated"}
    @app.post("/admin/rollback")
    async def rollback():
        """Admin endpoint to trigger rollback."""
        # Would integrate with RollbackCapableHotSwap
        return {"message": "Rollback initiated"}
    return app
```
---
## Putting It All Together
Here's the complete integration:
```python
"""
versioned_model_server.py
Complete model versioning with hot swap, draining, and rollback.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional
import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
import orjson
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Configuration
MODEL_DIR = os.getenv("MODEL_DIR", "models")
INITIAL_VERSION = os.getenv("INITIAL_VERSION", "v1.0.0")
DRAIN_TIMEOUT = float(os.getenv("DRAIN_TIMEOUT", "30.0"))
# Global components
registry: Optional[ModelRegistry] = None
pointer: Optional[AtomicVersionPointer] = None
tracker: Optional[RequestTracker] = None
hot_swap: Optional[RollbackCapableHotSwap] = None
async def load_model_version(version: ModelVersion) -> ModelVersion:
    """Load a model version into memory."""
    import time
    logger.info(f"[Loader] Loading {version.version_id} from {version.model_path}")
    start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(version.model_path, map_location=device)
    model.eval()
    # Warmup
    dummy = torch.randn(1, 3, 224, 224, device=device)
    for _ in range(3):
        with torch.no_grad():
            _ = model(dummy)
    version.loaded_model = model
    version.device = str(device)
    version.load_time = datetime.now()
    version.status = VersionStatus.STAGED
    load_time = (time.perf_counter() - start) * 1000
    logger.info(f"[Loader] Loaded {version.version_id} in {load_time:.0f}ms")
    return version
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global registry, pointer, tracker, hot_swap
    logger.info("[Startup] Initializing versioned model server...")
    # Initialize components
    registry = ModelRegistry()
    pointer = AtomicVersionPointer()
    tracker = RequestTracker(drain_timeout_seconds=DRAIN_TIMEOUT)
    # Create hot swap manager
    hot_swap = RollbackCapableHotSwap(
        registry=registry,
        pointer=pointer,
        tracker=tracker,
        model_loader=load_model_version,
    )
    # Load initial version
    initial_path = os.path.join(MODEL_DIR, f"{INITIAL_VERSION}.pt")
    initial_version = ModelVersion(
        version_id=INITIAL_VERSION,
        model_path=initial_path,
        schema=ModelSchema(
            input_shapes={"input": (None, 3, 224, 224)},
            input_dtypes={"input": "float32"},
            output_shapes={"output": (None, 1000)},
            output_dtypes={"output": "float32"},
        ),
    )
    await registry.register_version(initial_version)
    success, error = await hot_swap.hot_swap(
        INITIAL_VERSION,
        validate_schema=False,  # First version, nothing to validate against
        keep_previous=False,
    )
    if not success:
        logger.error(f"[Startup] Failed to load initial version: {error}")
        raise RuntimeError(f"Startup failed: {error}")
    logger.info(f"[Startup] Ready with version {INITIAL_VERSION}")
    yield  # Server runs here
    # Shutdown: drain and cleanup
    logger.info("[Shutdown] Draining requests...")
    active = await pointer.get()
    if active:
        await tracker.start_drain(active.version_id)
        await tracker.wait_for_drain(active.version_id)
    logger.info("[Shutdown] Complete")
app = FastAPI(lifespan=lifespan, title="Versioned Model Server")
@app.get("/health/liveness")
async def liveness():
    return {"status": "alive"}
@app.get("/health/readiness")
async def readiness():
    active = await pointer.get()
    if not active:
        raise HTTPException(status_code=503, detail="No active version")
    return {
        "status": "ready",
        "active_version": active.version_id,
        "device": active.device,
    }
@app.get("/versions")
async def list_versions():
    versions = await registry.list_versions()
    return Response(
        content=orjson.dumps(versions),
        media_type="application/json"
    )
@app.post("/predict")
async def predict(request: Request):
    """Inference using active version."""
    active = await pointer.get()
    if not active:
        raise HTTPException(status_code=503, detail="No active version")
    # Track request
    await tracker.acquire(active.version_id)
    try:
        body = await request.body()
        data = orjson.loads(body)
        input_tensor = torch.tensor(data["input"], dtype=torch.float32)
        input_tensor = input_tensor.to(active.device)
        with torch.no_grad():
            output = active.loaded_model(input_tensor)
        return Response(
            content=orjson.dumps({
                "prediction": output.tolist(),
                "version": active.version_id,
            }),
            media_type="application/json"
        )
    finally:
        await tracker.release(active.version_id)
@app.post("/v/{version_id}/predict")
async def predict_version(version_id: str, request: Request):
    """Inference using specific version."""
    version = await registry.get_version(version_id)
    if not version:
        raise HTTPException(status_code=404, detail=f"Version {version_id} not found")
    if version.loaded_model is None:
        raise HTTPException(status_code=503, detail=f"Version {version_id} not loaded")
    await tracker.acquire(version_id)
    try:
        body = await request.body()
        data = orjson.loads(body)
        input_tensor = torch.tensor(data["input"], dtype=torch.float32)
        input_tensor = input_tensor.to(version.device)
        with torch.no_grad():
            output = version.loaded_model(input_tensor)
        return Response(
            content=orjson.dumps({
                "prediction": output.tolist(),
                "version": version_id,
            }),
            media_type="application/json"
        )
    finally:
        await tracker.release(version_id)
@app.post("/admin/swap/{version_id}")
async def swap_version(version_id: str):
    """Hot swap to a new version."""
    version_path = os.path.join(MODEL_DIR, f"{version_id}.pt")
    # Register if not exists
    existing = await registry.get_version(version_id)
    if not existing:
        new_version = ModelVersion(
            version_id=version_id,
            model_path=version_path,
            schema=ModelSchema(
                input_shapes={"input": (None, 3, 224, 224)},
                input_dtypes={"input": "float32"},
                output_shapes={"output": (None, 1000)},
                output_dtypes={"output": "float32"},
            ),
        )
        await registry.register_version(new_version)
    success, error = await hot_swap.hot_swap(version_id)
    if not success:
        raise HTTPException(status_code=400, detail=error)
    return {"status": "swapped", "active_version": version_id}
@app.post("/admin/rollback")
async def rollback():
    """Rollback to previous version."""
    success, error = await hot_swap.manual_rollback()
    if not success:
        raise HTTPException(status_code=400, detail=error)
    active = await pointer.get()
    return {"status": "rolled_back", "active_version": active.version_id}
@app.get("/metrics")
async def metrics():
    """Prometheus metrics."""
    active = await pointer.get()
    lines = [
        f"model_active_version{{{active.version_id if active else 'none'}}} 1",
        f"model_versions_total {len(await registry.list_versions())}",
    ]
    for version_id, count in tracker._counts.items():
        lines.append(f"model_in_flight_requests{{version=\"{version_id}\"}} {count}")
    return "\n".join(lines)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
---
## Knowledge Cascade: What You've Unlocked
### Immediate Applications
- **Milestone 4 (A/B Testing)**: The version routing infrastructure you built is the foundation for traffic splitting. A/B testing adds probabilistic routing and statistical analysis on top.
- **Milestone 5 (Monitoring)**: The rollback monitor is your first step toward production observability. Drift detection extends this with data quality monitoring.
### Cross-Domain Connections
**1. Atomic Pointer Swaps → Database MVCC**
The atomic swap pattern is isomorphic to **Multi-Version Concurrency Control (MVCC)** in databases:
- Old version continues serving in-flight transactions
- New version is created alongside
- Switch happens when no transactions reference old version
- Garbage collection removes old versions
Both systems solve the same problem: **how to update state while concurrent readers are active.**
**2. Request Draining → Graceful Shutdown**
Request draining is the same pattern as **graceful HTTP server shutdown**:
- Stop accepting new connections
- Wait for existing connections to complete
- Force-close after timeout
- Clean up resources
This pattern appears in: web servers, database connection pools, message queue consumers, and any system that handles concurrent requests.
**3. Schema Compatibility → Semantic Versioning**
Model schema validation is **semantic versioning applied to ML**:
- MAJOR: Breaking schema changes (reject)
- MINOR: New features, backward compatible (accept)
- PATCH: Bug fixes, no API changes (accept)
The principle: **machines depend on contracts just like humans do.** Breaking contracts breaks systems.
**4. Instant Rollback → Git Revert**
Keeping the previous version in memory is the same idea as **Git's fast-forward revert**:
- Revert to previous commit is O(1) if you have the reference
- Re-creating from scratch takes much longer
- The cost is storage (memory/disk) for the backup
The tradeoff: **storage cost vs. recovery speed.** In production ML, recovery speed wins.
### Forward Connections
With model versioning mastered, you can now:
- Build **canary deployment systems** that gradually shift traffic
- Implement **feature flags** for model behavior
- Create **model cascades** where different versions handle different request types
- Design **multi-tenant serving** where each tenant has a dedicated model version
---
## Summary: The Hot Swap Mindset
Production model serving requires treating **versions as first-class citizens**:
1. **Memory is the constraint** — plan for 2.5x during hot swap
2. **Atomic swaps are non-negotiable** — no request should see inconsistent state
3. **Draining is patience** — let in-flight requests complete before cleanup
4. **Rollback is a button, not a process** — instant reversion saves the day
5. **Schemas are contracts** — validate before deploying, reject incompatible changes
You now have the infrastructure for **zero-downtime model deployment**—the foundation of reliable ML serving.
---
[[CRITERIA_JSON: {"milestone_id": "ml-model-serving-m3", "criteria": ["Model registry stores multiple versions with metadata including accuracy metrics, training date, training data hash, and input/output schema definitions", "Version-specific routing directs requests to specified version via URL path (/v/{version_id}/predict) or X-Model-Version header", "Default routing sends requests to the active version when no version is specified in URL or header", "Hot swap loads new model version into memory, validates schema compatibility, and atomically swaps the active pointer with no dropped requests", "In-flight request tracking via reference counting ensures requests to old version complete before model is unloaded (drain timeout configurable)", "Rollback reverts to previous version within 5 seconds by keeping previous version in memory and swapping pointer atomically", "Schema compatibility validation checks input/output shapes and dtypes before promotion; incompatible schemas are rejected with descriptive error", "Request tracker provides wait_for_drain() method that blocks until in-flight count reaches zero or timeout expires", "Memory budget during hot swap accounts for old model + new model + activation buffers (~2.5x single-model memory)", "Version status lifecycle: STAGED → ACTIVE → DRAINING → DEPRECATED with proper state transitions", "Automatic rollback triggers when error rate exceeds configurable threshold for consecutive monitoring windows", "RollbackMonitor tracks per-version error rates and latency percentiles for health assessment"]}]
<!-- END_MS -->


<!-- MS_ID: ml-model-serving-m4 -->
# A/B Testing & Canary Deployment
## The Experiment Illusion
You've built a model server with dynamic batching, version management, and hot-swapping. Now you want to know: *is my new model actually better?*
The naive approach looks straightforward:
```python
# Random assignment
if random.random() < 0.1:
    model = new_model  # 10% get new model
else:
    model = old_model  # 90% get old model
```
This works. For UI A/B tests. For ML models, it's **catastrophically wrong**.
Here's why: **user behavior depends on consistency**. If user Alice sees the new model on Monday and the old model on Tuesday, her experience is incoherent. But more critically, **your experiment is now invalid**. When Alice converts on Monday but not Tuesday, which model gets credit? The random assignment has introduced noise that makes your results uninterpretable.

![Traffic Splitter Architecture](./diagrams/diag-m4-traffic-splitter.svg)

The misconception is seductive: "A/B testing is just random assignment with metrics." But ML A/B testing has constraints that UI testing doesn't:
1. **Deterministic routing**: Same user must always see the same model version
2. **Prediction distribution monitoring**: Models can be "correct" but produce different output distributions
3. **Statistical rigor**: ML improvements are often small (1-3%); detecting them requires proper sample size calculations
4. **Confounding control**: Changing preprocessing alongside the model makes results uninterpretable
**The cost of invalid experiments isn't just bad data—it's wrong decisions.** A false positive (thinking the new model is better when it isn't) sends you down the wrong path. A false negative (failing to detect a real improvement) wastes the effort of building it.
---
## The Fundamental Tension: Experimentation vs. Risk
Every A/B testing system faces a core constraint:
**You want to measure model improvement accurately, but you're constrained by:**
1. **User consistency**: Random routing invalidates experiments; same user must see same model
2. **Sample size requirements**: Detecting 1% improvement requires ~15,000 samples per group
3. **Risk exposure**: A bad model can damage user experience during the experiment
4. **Confounding variables**: Multiple simultaneous changes make results uninterpretable

![Experiment Log Structure](./diagrams/tdd-diag-033.svg)

![Sample Size and Statistical Significance](./diagrams/diag-m4-statistical-significance.svg)

The sample size constraint is brutal. Let's do the math:
To detect a 1% improvement with 80% power at 95% confidence:
- Baseline conversion rate: 10%
- Improved rate: 10.1%
- Required sample size per group: **~157,000 users**
At 1,000 requests/day, that's **157 days** to get a statistically significant result. You can't wait that long. This tension—between statistical rigor and iteration speed—is the heart of ML experimentation.
---
## The Three-Level View
**Level 1 — Traffic Splitting (Application Layer)**:
- Deterministic routing via consistent hashing
- Configurable weight distribution
- Runtime weight adjustment
**Level 2 — Experiment Tracking (Data Layer)**:
- Per-request experiment logging
- Prediction distribution monitoring
- Statistical analysis pipeline
**Level 3 — Risk Mitigation (Safety Layer)**:
- Gradual rollout stages
- Automatic rollback on error spikes
- Confounding variable detection
---
## Phase 1: Deterministic Traffic Splitting
### The Consistent Hashing Solution
> **🔑 Foundation: Consistent hashing for deterministic routing**
> 
> ## Consistent Hashing for Deterministic Routing
### What It IS
Consistent hashing is a technique that maps keys to positions on a circular hash ring. Unlike simple modulo hashing (`hash(key) % N`), consistent hashing minimizes reassignment when buckets are added or removed. A key always maps to the same bucket as long as that bucket exists.
For traffic splitting, we use a simplified version: `hash(user_id) % 100` gives us a number 0-99. If we want 10% traffic to the canary, users with hash values 0-9 go to canary, 10-99 go to stable.
### WHY You Need It Right Now
Without deterministic routing:
- Same user sees different models across requests → inconsistent experience
- Experiment results become uninterpretable → noise swamps signal
- User-level analysis becomes impossible → can't track conversion funnels
Deterministic routing via hashing guarantees that user `alice_123` ALWAYS routes to the same model version, regardless of when her request arrives.
### ONE Key Insight
**Think of consistent hashing like seating at a restaurant.**
Simple modulo: "You're at table `hash(name) % table_count`." When you add a table, everyone's seat changes.
Consistent hashing: "You're at the first table clockwise from your position on this circular map." Adding a table only affects people who now fall into that table's section.
For traffic splitting, we use the simpler "percentile bucket" approach: the hash tells you your percentile, and the traffic weights tell you which percentile ranges map to which models.

![Consistent Hashing for Routing](./diagrams/tdd-diag-027.svg)

![Consistent Hashing for Deterministic Routing](./diagrams/diag-m4-consistent-hashing.svg)

```python
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time
import json
class RoutingStrategy(Enum):
    """Traffic routing strategies."""
    PERCENTAGE = "percentage"      # Route by percentage weights
    HEADER = "header"              # Route by explicit header
    STICKY_SESSION = "sticky"      # Route by session cookie
@dataclass
class TrafficAllocation:
    """
    Defines how traffic is split between model versions.
    The key insight: allocations are ordered by priority.
    A user whose hash falls into multiple allocations gets
    the FIRST matching allocation.
    """
    version_id: str
    weight_percent: float  # 0.0 to 100.0
    description: str = ""
@dataclass 
class TrafficSplitter:
    """
    Deterministic traffic splitter using consistent hashing.
    Guarantees:
    1. Same user_id always routes to same version (for fixed weights)
    2. Weight changes take effect immediately for new requests
    3. Routing decisions are reproducible and auditable
    """
    allocations: List[TrafficAllocation] = field(default_factory=list)
    hash_algorithm: str = "sha256"
    def __post_init__(self):
        # Validate weights sum to ~100%
        total = sum(a.weight_percent for a in self.allocations)
        if abs(total - 100.0) > 0.01:
            raise ValueError(
                f"Traffic weights must sum to 100%, got {total}%"
            )
        # Sort by version_id for deterministic ordering
        self.allocations = sorted(
            self.allocations, 
            key=lambda a: a.version_id
        )
    def _hash_user(self, user_id: str) -> int:
        """
        Hash a user ID to a number 0-99 (percentile bucket).
        Uses SHA-256 for cryptographic quality distribution.
        The first 8 bytes give us 64 bits, mod 100 gives percentile.
        """
        if not user_id:
            # Fallback for anonymous users
            user_id = f"anonymous_{time.time()}"
        hash_bytes = hashlib.sha256(
            user_id.encode('utf-8')
        ).digest()
        # Take first 8 bytes, convert to int
        hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')
        # Map to 0-99 percentile
        return hash_int % 100
    def route(self, user_id: str) -> Tuple[str, dict]:
        """
        Determine which version should handle this user's request.
        Args:
            user_id: Unique identifier for the user
        Returns:
            (version_id, routing_metadata)
        The metadata includes:
            - user_hash: The computed percentile bucket
            - allocation_reason: Why this version was selected
            - experiment_id: For logging/tracking
        """
        user_hash = self._hash_user(user_id)
        # Walk through allocations, accumulating weights
        cumulative = 0.0
        for allocation in self.allocations:
            cumulative += allocation.weight_percent
            if user_hash < cumulative:
                return allocation.version_id, {
                    "user_hash": user_hash,
                    "allocation_reason": f"hash {user_hash} < {cumulative}% threshold",
                    "allocation_weight": allocation.weight_percent,
                    "experiment_timestamp": time.time(),
                }
        # Fallback (shouldn't happen if weights sum to 100%)
        return self.allocations[-1].version_id, {
            "user_hash": user_hash,
            "allocation_reason": "fallback",
            "error": "Weight calculation error",
        }
    def update_weights(
        self, 
        new_allocations: List[TrafficAllocation]
    ) -> None:
        """
        Update traffic weights at runtime.
        This is O(1) for routing - the allocation list is just replaced.
        In-flight requests are unaffected; only new requests use new weights.
        """
        total = sum(a.weight_percent for a in new_allocations)
        if abs(total - 100.0) > 0.01:
            raise ValueError(f"Weights must sum to 100%, got {total}%")
        self.allocations = sorted(
            new_allocations,
            key=lambda a: a.version_id
        )
    def get_allocation_for_version(
        self, 
        version_id: str
    ) -> Optional[TrafficAllocation]:
        """Get the current allocation for a specific version."""
        for allocation in self.allocations:
            if allocation.version_id == version_id:
                return allocation
        return None
    def to_dict(self) -> dict:
        """Export current configuration."""
        return {
            "allocations": [
                {
                    "version_id": a.version_id,
                    "weight_percent": a.weight_percent,
                    "description": a.description,
                }
                for a in self.allocations
            ],
            "hash_algorithm": self.hash_algorithm,
        }
    @classmethod
    def from_dict(cls, data: dict) -> "TrafficSplitter":
        """Create from configuration dict."""
        allocations = [
            TrafficAllocation(
                version_id=a["version_id"],
                weight_percent=a["weight_percent"],
                description=a.get("description", ""),
            )
            for a in data["allocations"]
        ]
        return cls(
            allocations=allocations,
            hash_algorithm=data.get("hash_algorithm", "sha256"),
        )
# Example: 90% stable, 10% canary
splitter = TrafficSplitter(allocations=[
    TrafficAllocation(
        version_id="v1.0.0",
        weight_percent=90.0,
        description="Stable production model",
    ),
    TrafficAllocation(
        version_id="v2.0.0", 
        weight_percent=10.0,
        description="Canary candidate model",
    ),
])
# Test deterministic routing
assert splitter.route("alice")[0] == splitter.route("alice")[0]  # Always same
assert splitter.route("bob")[0] == splitter.route("bob")[0]      # Always same
# But alice and bob may differ (based on their hashes)
```
### Cross-Domain Connection: Database Sharding
This exact hashing pattern appears in **database sharding**:
| Database Sharding | Traffic Splitting |
|-------------------|-------------------|
| `hash(user_id) % shard_count` | `hash(user_id) % 100` |
| User data lives on one shard | User always sees one model |
| Adding shards = rebalancing | Changing weights = re-routing |
| Consistent hashing minimizes moves | Percentile buckets preserve users |
The principle is universal: **deterministic mapping from entity to destination.** Whether it's database rows or model versions, the same user always lands in the same place.
---
## Phase 2: Runtime Weight Adjustment
### The Canary Rollout Controller

![Traffic Splitter Architecture](./diagrams/tdd-diag-026.svg)

![Gradual Canary Rollout Stages](./diagrams/diag-m4-canary-rollout.svg)

Canary deployments don't jump from 0% to 100%. They **gradually increase** exposure, checking health at each stage:
```python
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Awaitable
from enum import Enum
import asyncio
import logging
logger = logging.getLogger(__name__)
class RolloutStage(Enum):
    """Stages of a canary rollout."""
    INITIALIZING = "initializing"
    STAGE_1 = "stage_1"      # 5%
    STAGE_2 = "stage_2"      # 10%
    STAGE_3 = "stage_3"      # 25%
    STAGE_4 = "stage_4"      # 50%
    STAGE_5 = "stage_5"      # 100%
    COMPLETE = "complete"
    ROLLED_BACK = "rolled_back"
@dataclass
class RolloutConfig:
    """Configuration for a canary rollout."""
    stable_version: str
    canary_version: str
    stages: List[float] = field(
        default_factory=lambda: [5.0, 10.0, 25.0, 50.0, 100.0]
    )
    stage_duration_seconds: float = 300.0  # 5 minutes per stage
    health_check_interval_seconds: float = 30.0
    # Rollback triggers
    max_error_rate_multiplier: float = 2.0  # Canary errors > 2x stable = rollback
    max_latency_multiplier: float = 1.5     # Canary latency > 1.5x stable = rollback
    min_samples_per_stage: int = 100        # Minimum samples before advancing
@dataclass
class RolloutState:
    """Current state of a canary rollout."""
    stage: RolloutStage = RolloutStage.INITIALIZING
    current_canary_percent: float = 0.0
    stage_start_time: Optional[float] = None
    samples_this_stage: int = 0
    last_health_check: Optional[float] = None
    health_check_passed: bool = True
    rollback_reason: Optional[str] = None
class CanaryRolloutController:
    """
    Orchestrates gradual canary rollouts with automatic health checking.
    The rollout progresses through stages only if:
    1. Minimum samples collected
    2. Health checks pass
    3. Stage duration elapsed
    At any point, health check failure triggers automatic rollback.
    """
    def __init__(
        self,
        config: RolloutConfig,
        splitter: TrafficSplitter,
        health_checker: Callable[[str, str], Awaitable[Tuple[bool, Optional[str]]]],
        on_stage_change: Optional[Callable[[RolloutStage, float], Awaitable[None]]] = None,
    ):
        self.config = config
        self.splitter = splitter
        self.health_checker = health_checker
        self.on_stage_change = on_stage_change
        self.state = RolloutState()
        self._running = False
        self._task: Optional[asyncio.Task] = None
    async def start_rollout(self) -> None:
        """
        Begin the canary rollout process.
        This starts a background task that:
        1. Sets initial traffic split (canary at 0%)
        2. Progresses through stages
        3. Checks health at each stage
        4. Triggers rollback if needed
        """
        if self._running:
            logger.warning("[Rollout] Already running")
            return
        logger.info(
            f"[Rollout] Starting canary deployment: "
            f"{self.config.stable_version} -> {self.config.canary_version}"
        )
        self._running = True
        self.state = RolloutState(stage=RolloutStage.INITIALIZING)
        # Initialize splitter with 0% canary
        self._update_splitter(0.0)
        # Start the rollout loop
        self._task = asyncio.create_task(self._rollout_loop())
    async def _rollout_loop(self) -> None:
        """Main rollout orchestration loop."""
        try:
            for stage_idx, canary_percent in enumerate(self.config.stages):
                # Enter new stage
                self.state.stage = RolloutStage(f"stage_{stage_idx + 1}")
                self.state.current_canary_percent = canary_percent
                self.state.stage_start_time = time.time()
                self.state.samples_this_stage = 0
                logger.info(
                    f"[Rollout] Entering {self.state.stage.value}: "
                    f"canary at {canary_percent}%"
                )
                # Update traffic split
                self._update_splitter(canary_percent)
                # Notify callback
                if self.on_stage_change:
                    await self.on_stage_change(self.state.stage, canary_percent)
                # Wait for stage duration, checking health periodically
                stage_elapsed = 0.0
                while stage_elapsed < self.config.stage_duration_seconds:
                    await asyncio.sleep(self.config.health_check_interval_seconds)
                    stage_elapsed += self.config.health_check_interval_seconds
                    # Health check
                    is_healthy, reason = await self.health_checker(
                        self.config.stable_version,
                        self.config.canary_version,
                    )
                    self.state.last_health_check = time.time()
                    self.state.health_check_passed = is_healthy
                    if not is_healthy:
                        # Rollback!
                        await self._trigger_rollback(reason)
                        return
                # Check minimum samples
                if self.state.samples_this_stage < self.config.min_samples_per_stage:
                    logger.warning(
                        f"[Rollout] Insufficient samples: "
                        f"{self.state.samples_this_stage} < "
                        f"{self.config.min_samples_per_stage}. Extending stage."
                    )
                    # Extend the stage
                    while (
                        self.state.samples_this_stage < self.config.min_samples_per_stage
                        and stage_elapsed < self.config.stage_duration_seconds * 3
                    ):
                        await asyncio.sleep(self.config.health_check_interval_seconds)
                        stage_elapsed += self.config.health_check_interval_seconds
                        is_healthy, reason = await self.health_checker(
                            self.config.stable_version,
                            self.config.canary_version,
                        )
                        if not is_healthy:
                            await self._trigger_rollback(reason)
                            return
            # All stages complete!
            self.state.stage = RolloutStage.COMPLETE
            logger.info("[Rollout] Canary deployment complete!")
            if self.on_stage_change:
                await self.on_stage_change(RolloutStage.COMPLETE, 100.0)
        except asyncio.CancelledError:
            logger.info("[Rollout] Cancelled")
        except Exception as e:
            logger.error(f"[Rollout] Error: {e}", exc_info=True)
            await self._trigger_rollback(f"Internal error: {e}")
    def _update_splitter(self, canary_percent: float) -> None:
        """Update the traffic splitter with new weights."""
        stable_percent = 100.0 - canary_percent
        self.splitter.update_weights([
            TrafficAllocation(
                version_id=self.config.stable_version,
                weight_percent=stable_percent,
                description="Stable production model",
            ),
            TrafficAllocation(
                version_id=self.config.canary_version,
                weight_percent=canary_percent,
                description="Canary candidate model",
            ),
        ])
    async def _trigger_rollback(self, reason: str) -> None:
        """
        Trigger immediate rollback to stable version.
        This is the safety valve - if canary shows problems,
        we immediately route 100% traffic back to stable.
        """
        logger.error(f"[Rollout] ROLLBACK triggered: {reason}")
        self.state.stage = RolloutStage.ROLLED_BACK
        self.state.rollback_reason = reason
        # Route 100% to stable
        self._update_splitter(0.0)
        if self.on_stage_change:
            await self.on_stage_change(RolloutStage.ROLLED_BACK, 0.0)
    def record_sample(self) -> None:
        """Record that a sample was collected (for minimum sample tracking)."""
        self.state.samples_this_stage += 1
    async def stop(self) -> None:
        """Stop the rollout controller."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    def get_state(self) -> dict:
        """Get current rollout state for monitoring."""
        return {
            "stage": self.state.stage.value,
            "canary_percent": self.state.current_canary_percent,
            "stage_start_time": self.state.stage_start_time,
            "samples_this_stage": self.state.samples_this_stage,
            "health_check_passed": self.state.health_check_passed,
            "rollback_reason": self.state.rollback_reason,
        }
```
### Cross-Domain Connection: Progressive Traffic Shifting
This staged rollout pattern is identical to **service mesh progressive deployment**:
| Service Mesh (Istio/Linkerd) | Canary Controller |
|------------------------------|-------------------|
| VirtualService weight adjustment | TrafficSplitter.update_weights() |
| 1% → 10% → 50% → 100% | stages: [5, 10, 25, 50, 100] |
| Health check on upstream | health_checker callback |
| Automatic rollback on 5xx spike | _trigger_rollback() |
The pattern: **increase exposure gradually, verify health at each step, bail out fast on failure.** It's universal in distributed systems deployment.
---
## Phase 3: Per-Version Metrics Collection

![Gradual Canary Rollout Stages](./diagrams/tdd-diag-028.svg)

![Per-Version Metrics Collection](./diagrams/diag-m4-per-version-metrics.svg)

You can't compare model versions without **per-version metrics**. The key insight: metrics must be collected *before* aggregation, preserving version identity.
```python
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import statistics
import time
@dataclass
class LatencySample:
    """A single latency measurement."""
    timestamp: float
    latency_ms: float
    version_id: str
    success: bool
@dataclass
class PredictionSample:
    """A single prediction output sample."""
    timestamp: float
    prediction: float  # For regression or class probability
    version_id: str
    input_hash: str  # Hash of input features for debugging
class PerVersionMetrics:
    """
    Collects and aggregates metrics per model version.
    Key design: samples are stored per-version, allowing
    independent analysis of each version's performance.
    """
    def __init__(
        self,
        latency_window_size: int = 10000,
        prediction_window_size: int = 10000,
    ):
        self.latency_window_size = latency_window_size
        self.prediction_window_size = prediction_window_size
        # Per-version latency samples
        self._latency_samples: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=latency_window_size)
        )
        # Per-version prediction samples
        self._prediction_samples: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=prediction_window_size)
        )
        # Counters
        self._request_counts: Dict[str, int] = defaultdict(int)
        self._error_counts: Dict[str, int] = defaultdict(int)
        # Start time for rate calculations
        self._start_time = time.time()
    def record_latency(
        self,
        version_id: str,
        latency_ms: float,
        success: bool = True,
    ) -> None:
        """Record a latency sample for a version."""
        sample = LatencySample(
            timestamp=time.time(),
            latency_ms=latency_ms,
            version_id=version_id,
            success=success,
        )
        self._latency_samples[version_id].append(sample)
        self._request_counts[version_id] += 1
        if not success:
            self._error_counts[version_id] += 1
    def record_prediction(
        self,
        version_id: str,
        prediction: float,
        input_hash: str = "",
    ) -> None:
        """Record a prediction output for distribution analysis."""
        sample = PredictionSample(
            timestamp=time.time(),
            prediction=prediction,
            version_id=version_id,
            input_hash=input_hash,
        )
        self._prediction_samples[version_id].append(sample)
    def get_latency_percentiles(
        self,
        version_id: str,
        percentiles: List[float] = [50, 90, 95, 99],
    ) -> Dict[str, float]:
        """
        Calculate latency percentiles for a version.
        Returns dict like {"p50": 45.2, "p95": 78.3, ...}
        """
        samples = self._latency_samples.get(version_id, deque())
        if not samples:
            return {f"p{p}": 0.0 for p in percentiles}
        # Extract successful latencies only
        latencies = sorted(
            [s.latency_ms for s in samples if s.success]
        )
        if not latencies:
            return {f"p{p}": 0.0 for p in percentiles}
        result = {}
        for p in percentiles:
            idx = int(len(latencies) * p / 100)
            idx = min(idx, len(latencies) - 1)
            result[f"p{p}"] = latencies[idx]
        return result
    def get_error_rate(self, version_id: str) -> float:
        """Calculate error rate for a version (0.0 to 1.0)."""
        total = self._request_counts.get(version_id, 0)
        if total == 0:
            return 0.0
        errors = self._error_counts.get(version_id, 0)
        return errors / total
    def get_throughput_rps(self, version_id: str) -> float:
        """Calculate requests per second for a version."""
        total = self._request_counts.get(version_id, 0)
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0
        return total / elapsed
    def get_prediction_distribution(
        self,
        version_id: str,
        n_bins: int = 20,
    ) -> Dict[str, float]:
        """
        Get histogram of prediction values.
        Useful for detecting prediction distribution shifts
        between versions.
        """
        samples = self._prediction_samples.get(version_id, deque())
        if not samples:
            return {}
        predictions = [s.prediction for s in samples]
        if not predictions:
            return {}
        min_val = min(predictions)
        max_val = max(predictions)
        # Handle constant predictions
        if min_val == max_val:
            return {f"bin_0": 1.0}
        bin_width = (max_val - min_val) / n_bins
        bins = [0] * n_bins
        for pred in predictions:
            bin_idx = int((pred - min_val) / bin_width)
            bin_idx = min(bin_idx, n_bins - 1)
            bins[bin_idx] += 1
        total = len(predictions)
        return {
            f"bin_{i}": bins[i] / total
            for i in range(n_bins)
        }
    def compare_versions(
        self,
        version_a: str,
        version_b: str,
    ) -> dict:
        """
        Compare metrics between two versions.
        Returns a comparison dict with:
        - Latency differences
        - Error rate differences
        - Throughput differences
        """
        metrics_a = {
            "latency": self.get_latency_percentiles(version_a),
            "error_rate": self.get_error_rate(version_a),
            "throughput": self.get_throughput_rps(version_a),
            "request_count": self._request_counts.get(version_a, 0),
        }
        metrics_b = {
            "latency": self.get_latency_percentiles(version_b),
            "error_rate": self.get_error_rate(version_b),
            "throughput": self.get_throughput_rps(version_b),
            "request_count": self._request_counts.get(version_b, 0),
        }
        # Calculate differences
        latency_diff = {}
        for key in metrics_a["latency"]:
            if metrics_b["latency"][key] > 0:
                diff = (
                    (metrics_b["latency"][key] - metrics_a["latency"][key])
                    / metrics_a["latency"][key] * 100
                )
                latency_diff[key] = diff
        return {
            "version_a": version_a,
            "version_b": version_b,
            "latency_a": metrics_a["latency"],
            "latency_b": metrics_b["latency"],
            "latency_diff_percent": latency_diff,
            "error_rate_a": metrics_a["error_rate"],
            "error_rate_b": metrics_b["error_rate"],
            "error_rate_diff": metrics_b["error_rate"] - metrics_a["error_rate"],
            "throughput_a": metrics_a["throughput"],
            "throughput_b": metrics_b["throughput"],
            "sample_count_a": metrics_a["request_count"],
            "sample_count_b": metrics_b["request_count"],
        }
    def get_all_version_metrics(self) -> Dict[str, dict]:
        """Get metrics summary for all tracked versions."""
        all_versions = set(self._request_counts.keys())
        return {
            version_id: {
                "latency_percentiles": self.get_latency_percentiles(version_id),
                "error_rate": self.get_error_rate(version_id),
                "throughput_rps": self.get_throughput_rps(version_id),
                "request_count": self._request_counts.get(version_id, 0),
                "error_count": self._error_counts.get(version_id, 0),
            }
            for version_id in all_versions
        }
    def to_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        for version_id, metrics in self.get_all_version_metrics().items():
            labels = f'{{version="{version_id}"}}'
            # Latency percentiles
            for p_name, value in metrics["latency_percentiles"].items():
                lines.append(
                    f"model_latency_ms{{percentile=\"{p_name}\",{labels[1:]}"
                )
                lines[-1] = lines[-1].rstrip("}") + f",{labels[1:]}"
                lines[-1] = f"model_latency_ms{{percentile=\"{p_name}\",version=\"{version_id}\"}} {value:.2f}"
            # Error rate
            lines.append(
                f"model_error_rate{{version=\"{version_id}\"}} {metrics['error_rate']:.4f}"
            )
            # Throughput
            lines.append(
                f"model_throughput_rps{{version=\"{version_id}\"}} {metrics['throughput_rps']:.2f}"
            )
            # Counts
            lines.append(
                f"model_requests_total{{version=\"{version_id}\"}} {metrics['request_count']}"
            )
        return "\n".join(lines)
```
---
## Phase 4: Statistical Significance Testing
> **🔑 Foundation: Statistical hypothesis testing**
> 
> ## Statistical Hypothesis Testing
### What It IS
Statistical hypothesis testing is a framework for determining whether observed differences between groups are likely real (statistically significant) or could have occurred by chance. In A/B testing, we compare metrics between control (stable model) and treatment (canary model) groups.
The key concepts:
- **Null hypothesis (H₀)**: No real difference between groups
- **Alternative hypothesis (H₁)**: Real difference exists
- **p-value**: Probability of observing this data if H₀ is true
- **Significance level (α)**: Threshold for rejecting H₀ (typically 0.05)
- **Power**: Probability of detecting a real effect (typically 80%)
### WHY You Need It Right Now
Without statistical rigor:
- **False positives**: You think the canary is better when it isn't → deploy bad model
- **False negatives**: You miss a real improvement → waste development effort
- **Premature conclusions**: Deciding too early with insufficient data → unreliable results
Statistical testing tells you when you have enough evidence to make a confident decision.
### ONE Key Insight
**Think of statistical significance like a court trial.**
- **Null hypothesis**: The defendant is innocent (models are the same)
- **Evidence**: Your data (metrics from both versions)
- **p-value**: How surprising the evidence would be if the defendant were innocent
- **Significance level**: The burden of proof (beyond reasonable doubt = α = 0.05)
If p-value < α, you reject the null hypothesis (convict). But there's always a chance of error—innocent people get convicted (false positive), and guilty people go free (false negative).
The solution: **require more evidence (larger sample size) for smaller expected effects.**

![Per-Version Metrics Collection](./diagrams/tdd-diag-029.svg)


```python
import math
from dataclasses import dataclass
from typing import Tuple, Optional, List
from enum import Enum
import statistics
class TestType(Enum):
    """Types of statistical tests."""
    TWO_TAILED = "two_tailed"    # H₁: μ_A ≠ μ_B
    ONE_TAILED_BETTER = "one_better"  # H₁: μ_A > μ_B
    ONE_TAILED_WORSE = "one_worse"    # H₁: μ_A < μ_B
@dataclass
class SampleStatistics:
    """Summary statistics for a sample."""
    mean: float
    std_dev: float
    n: int
    @property
    def std_error(self) -> float:
        """Standard error of the mean."""
        if self.n <= 1:
            return 0.0
        return self.std_dev / math.sqrt(self.n)
@dataclass
class TestResult:
    """Result of a statistical test."""
    test_type: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_interval: Tuple[float, float]
    effect_size: float
    interpretation: str
class StatisticalAnalyzer:
    """
    Performs statistical significance testing for A/B experiments.
    Supports:
    - t-test for comparing means (e.g., latency, conversion rate)
    - Chi-square test for comparing proportions
    - Sample size calculations for experiment planning
    """
    def __init__(
        self,
        significance_level: float = 0.05,
        power: float = 0.80,
    ):
        self.alpha = significance_level
        self.power = power
    @staticmethod
    def compute_sample_stats(values: List[float]) -> SampleStatistics:
        """Compute summary statistics from a list of values."""
        if not values:
            return SampleStatistics(mean=0.0, std_dev=0.0, n=0)
        n = len(values)
        mean = statistics.mean(values)
        if n <= 1:
            return SampleStatistics(mean=mean, std_dev=0.0, n=n)
        std_dev = statistics.stdev(values)
        return SampleStatistics(mean=mean, std_dev=std_dev, n=n)
    def two_sample_t_test(
        self,
        sample_a: SampleStatistics,
        sample_b: SampleStatistics,
        test_type: TestType = TestType.TWO_TAILED,
    ) -> TestResult:
        """
        Perform a two-sample t-test.
        Tests whether the means of two groups are significantly different.
        The test statistic:
            t = (mean_A - mean_B) / sqrt(SE_A² + SE_B²)
        where SE = standard error = std_dev / sqrt(n)
        """
        # Compute standard error of difference
        se_diff = math.sqrt(
            sample_a.std_error ** 2 + 
            sample_b.std_error ** 2
        )
        if se_diff == 0:
            return TestResult(
                test_type="t_test",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                confidence_interval=(0.0, 0.0),
                effect_size=0.0,
                interpretation="No variance in samples",
            )
        # Compute t-statistic
        t_stat = (sample_a.mean - sample_b.mean) / se_diff
        # Compute degrees of freedom (Welch's formula for unequal variances)
        df_numerator = (
            sample_a.std_error ** 2 + 
            sample_b.std_error ** 2
        ) ** 2
        df_denominator = (
            (sample_a.std_error ** 4) / (sample_a.n - 1) +
            (sample_b.std_error ** 4) / (sample_b.n - 1)
        )
        if df_denominator == 0:
            df = sample_a.n + sample_b.n - 2
        else:
            df = df_numerator / df_denominator
        # Compute p-value using t-distribution CDF
        # For simplicity, using normal approximation for large samples
        if df >= 30:
            # Use normal approximation
            p_value = self._normal_p_value(t_stat, test_type)
        else:
            # Use t-distribution (simplified approximation)
            p_value = self._t_p_value_approx(t_stat, df, test_type)
        # Effect size (Cohen's d)
        pooled_std = math.sqrt(
            ((sample_a.n - 1) * sample_a.std_dev ** 2 +
             (sample_b.n - 1) * sample_b.std_dev ** 2) /
            max(1, sample_a.n + sample_b.n - 2)
        )
        cohens_d = (
            (sample_a.mean - sample_b.mean) / pooled_std
            if pooled_std > 0 else 0.0
        )
        # Confidence interval for difference
        t_critical = self._t_critical_value(df)
        margin = t_critical * se_diff
        diff = sample_a.mean - sample_b.mean
        is_significant = p_value < self.alpha
        # Interpretation
        if is_significant:
            if diff > 0:
                interpretation = (
                    f"Group A is significantly higher than B "
                    f"(diff={diff:.4f}, p={p_value:.4f})"
                )
            else:
                interpretation = (
                    f"Group B is significantly higher than A "
                    f"(diff={-diff:.4f}, p={p_value:.4f})"
                )
        else:
            interpretation = (
                f"No significant difference detected (p={p_value:.4f})"
            )
        return TestResult(
            test_type="two_sample_t_test",
            statistic=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            confidence_interval=(diff - margin, diff + margin),
            effect_size=cohens_d,
            interpretation=interpretation,
        )
    def _normal_p_value(self, z: float, test_type: TestType) -> float:
        """Compute p-value using standard normal distribution."""
        # Approximate CDF of standard normal
        # Using error function approximation
        def norm_cdf(x: float) -> float:
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        if test_type == TestType.TWO_TAILED:
            return 2 * (1 - norm_cdf(abs(z)))
        elif test_type == TestType.ONE_TAILED_BETTER:
            return 1 - norm_cdf(z)
        else:  # ONE_TAILED_WORSE
            return norm_cdf(z)
    def _t_p_value_approx(
        self, 
        t: float, 
        df: float, 
        test_type: TestType
    ) -> float:
        """
        Approximate p-value for t-distribution.
        Uses normal approximation with correction for small samples.
        """
        # For small samples, t-distribution has heavier tails
        # Apply correction factor
        correction = math.sqrt(df / (df - 2)) if df > 2 else 1.0
        z_corrected = t / correction
        return self._normal_p_value(z_corrected, test_type)
    def _t_critical_value(self, df: float) -> float:
        """Get critical t-value for current significance level."""
        # Approximate critical values
        # For α = 0.05, two-tailed
        if df >= 30:
            return 1.96  # Normal approximation
        elif df >= 20:
            return 2.086
        elif df >= 10:
            return 2.228
        elif df >= 5:
            return 2.571
        else:
            return 3.182  # Very small sample
    def compute_required_sample_size(
        self,
        baseline_mean: float,
        baseline_std: float,
        minimum_detectable_effect: float,
        test_type: TestType = TestType.TWO_TAILED,
    ) -> int:
        """
        Calculate required sample size for detecting an effect.
        Args:
            baseline_mean: Expected mean for control group
            baseline_std: Expected standard deviation
            minimum_detectable_effect: Smallest difference you want to detect
            test_type: One-tailed or two-tailed test
        Returns:
            Required sample size per group
        """
        # Z-scores for significance level and power
        # For α = 0.05, Z_α/2 = 1.96 (two-tailed)
        # For β = 0.20 (80% power), Z_β = 0.84
        if test_type == TestType.TWO_TAILED:
            z_alpha = 1.96  # α/2 = 0.025
        else:
            z_alpha = 1.645  # α = 0.05
        z_beta = 0.84  # For 80% power
        # Sample size formula:
        # n = 2 * (Z_α + Z_β)² * σ² / Δ²
        # where Δ = minimum detectable effect
        if minimum_detectable_effect == 0:
            return float('inf')
        n = (
            2 * (z_alpha + z_beta) ** 2 * baseline_std ** 2 /
            minimum_detectable_effect ** 2
        )
        return int(math.ceil(n))
    def proportion_test(
        self,
        successes_a: int,
        trials_a: int,
        successes_b: int,
        trials_b: int,
    ) -> TestResult:
        """
        Perform a two-proportion z-test.
        Useful for conversion rates, error rates, etc.
        """
        if trials_a == 0 or trials_b == 0:
            return TestResult(
                test_type="proportion_z_test",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                confidence_interval=(0.0, 0.0),
                effect_size=0.0,
                interpretation="No samples in one or both groups",
            )
        # Proportions
        p_a = successes_a / trials_a
        p_b = successes_b / trials_b
        # Pooled proportion
        p_pooled = (successes_a + successes_b) / (trials_a + trials_b)
        # Standard error
        se = math.sqrt(
            p_pooled * (1 - p_pooled) * (1/trials_a + 1/trials_b)
        )
        if se == 0:
            return TestResult(
                test_type="proportion_z_test",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                confidence_interval=(0.0, 0.0),
                effect_size=0.0,
                interpretation="No variance in proportions",
            )
        # Z-statistic
        z_stat = (p_a - p_b) / se
        # P-value (two-tailed)
        p_value = self._normal_p_value(z_stat, TestType.TWO_TAILED)
        # Effect size (difference in proportions)
        diff = p_a - p_b
        # Confidence interval
        z_crit = 1.96
        se_diff = math.sqrt(
            p_a * (1 - p_a) / trials_a + 
            p_b * (1 - p_b) / trials_b
        )
        margin = z_crit * se_diff
        is_significant = p_value < self.alpha
        interpretation = (
            f"Proportion A: {p_a:.4f}, B: {p_b:.4f}, "
            f"diff: {diff:.4f}, p: {p_value:.4f}"
        )
        return TestResult(
            test_type="proportion_z_test",
            statistic=z_stat,
            p_value=p_value,
            is_significant=is_significant,
            confidence_interval=(diff - margin, diff + margin),
            effect_size=diff,
            interpretation=interpretation,
        )
# Example usage
def analyze_experiment():
    """Example: Analyze an A/B experiment."""
    analyzer = StatisticalAnalyzer(significance_level=0.05, power=0.80)
    # Sample data: latencies from two model versions
    stable_latencies = [45, 48, 52, 47, 49, 51, 46, 50, 48, 52] * 100
    canary_latencies = [42, 45, 48, 44, 46, 47, 43, 47, 45, 49] * 100
    stats_stable = StatisticalAnalyzer.compute_sample_stats(stable_latencies)
    stats_canary = StatisticalAnalyzer.compute_sample_stats(canary_latencies)
    result = analyzer.two_sample_t_test(
        stats_canary,  # Treatment (canary)
        stats_stable,  # Control (stable)
        TestType.ONE_TAILED_BETTER,  # Expecting canary to be better (lower latency)
    )
    print(f"Stable: mean={stats_stable.mean:.2f}ms, std={stats_stable.std_dev:.2f}")
    print(f"Canary: mean={stats_canary.mean:.2f}ms, std={stats_canary.std_dev:.2f}")
    print(f"Result: {result.interpretation}")
    print(f"Effect size (Cohen's d): {result.effect_size:.3f}")
    print(f"Significant: {result.is_significant}")
    # Sample size calculation
    required = analyzer.compute_required_sample_size(
        baseline_mean=stats_stable.mean,
        baseline_std=stats_stable.std_dev,
        minimum_detectable_effect=2.0,  # Want to detect 2ms improvement
    )
    print(f"\nRequired sample size (to detect 2ms effect): {required} per group")
```
---
## Phase 5: Experiment Logging and Audit Trail

![Statistical Significance Testing](./diagrams/tdd-diag-030.svg)

![Experiment Log Structure](./diagrams/diag-m4-experiment-log.svg)

Every prediction during an experiment must be **logged with full context**. This is your audit trail—the ability to reconstruct exactly what happened, when, and why.
```python
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import hashlib
@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment."""
    experiment_id: str
    name: str
    description: str
    start_time: datetime
    control_version: str
    treatment_version: str
    traffic_split: float  # Treatment percentage
    hypothesis: str  # What we're testing
    success_metric: str  # e.g., "conversion_rate", "latency_p50"
    minimum_sample_size: int
    significance_level: float = 0.05
    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "start_time": self.start_time.isoformat(),
            "control_version": self.control_version,
            "treatment_version": self.treatment_version,
            "traffic_split": self.traffic_split,
            "hypothesis": self.hypothesis,
            "success_metric": self.success_metric,
            "minimum_sample_size": self.minimum_sample_size,
            "significance_level": self.significance_level,
        }
@dataclass
class PredictionLog:
    """
    Log entry for a single prediction during an experiment.
    This is the atomic unit of experiment data. Every prediction
    must include:
    - Which experiment it belongs to
    - Which model version made the prediction
    - Input features (or hash thereof)
    - Prediction output
    - Timing information
    """
    timestamp: float
    experiment_id: str
    request_id: str
    user_id: str
    model_version: str
    allocation_hash: int  # The hash bucket (for reproducibility)
    # Input/Output
    input_hash: str  # SHA-256 of input features
    prediction: Any  # Model output
    prediction_class: Optional[str] = None  # For classification
    prediction_confidence: Optional[float] = None
    # Performance
    latency_ms: float
    success: bool
    error_message: Optional[str] = None
    # Context
    request_metadata: Dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "experiment_id": self.experiment_id,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "model_version": self.model_version,
            "allocation_hash": self.allocation_hash,
            "input_hash": self.input_hash,
            "prediction": self.prediction,
            "prediction_class": self.prediction_class,
            "prediction_confidence": self.prediction_confidence,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "error_message": self.error_message,
            "request_metadata": self.request_metadata,
        }
    def to_json_line(self) -> str:
        """Convert to JSON line for logging (one JSON object per line)."""
        return json.dumps(self.to_dict())
class ExperimentLogger:
    """
    Logs predictions for experiment analysis.
    Design principles:
    1. Every prediction is logged (no sampling)
    2. Log format is append-only (no mutations)
    3. Logs include full context for post-hoc analysis
    4. Logs are partitioned by experiment_id for efficient querying
    """
    def __init__(
        self,
        log_dir: str = "experiments",
        flush_interval: int = 100,
    ):
        self.log_dir = log_dir
        self.flush_interval = flush_interval
        # Buffer for batch writes
        self._buffer: List[PredictionLog] = []
        self._current_experiment: Optional[str] = None
        self._log_file_handle = None
    def _hash_input(self, input_data: Any) -> str:
        """Create a hash of input features for debugging."""
        input_str = json.dumps(input_data, sort_keys=True)
        return hashlib.sha256(input_str.encode()).hexdigest()[:16]
    def log_prediction(
        self,
        experiment_id: str,
        request_id: str,
        user_id: str,
        model_version: str,
        allocation_hash: int,
        input_data: Any,
        prediction: Any,
        latency_ms: float,
        success: bool = True,
        error_message: Optional[str] = None,
        prediction_class: Optional[str] = None,
        prediction_confidence: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Log a prediction for the experiment.
        This should be called for EVERY prediction during an experiment,
        regardless of which model version was used.
        """
        log_entry = PredictionLog(
            timestamp=time.time(),
            experiment_id=experiment_id,
            request_id=request_id,
            user_id=user_id,
            model_version=model_version,
            allocation_hash=allocation_hash,
            input_hash=self._hash_input(input_data),
            prediction=prediction,
            prediction_class=prediction_class,
            prediction_confidence=prediction_confidence,
            latency_ms=latency_ms,
            success=success,
            error_message=error_message,
            request_metadata=metadata or {},
        )
        self._buffer.append(log_entry)
        # Flush if buffer is full
        if len(self._buffer) >= self.flush_interval:
            self._flush_buffer()
    def _flush_buffer(self) -> None:
        """Write buffered logs to disk."""
        if not self._buffer:
            return
        # Group by experiment_id
        by_experiment: Dict[str, List[PredictionLog]] = {}
        for entry in self._buffer:
            if entry.experiment_id not in by_experiment:
                by_experiment[entry.experiment_id] = []
            by_experiment[entry.experiment_id].append(entry)
        # Write each experiment's logs
        for experiment_id, entries in by_experiment.items():
            log_path = f"{self.log_dir}/{experiment_id}.jsonl"
            with open(log_path, "a") as f:
                for entry in entries:
                    f.write(entry.to_json_line() + "\n")
        self._buffer = []
    def finalize(self) -> None:
        """Flush any remaining logs."""
        self._flush_buffer()
class ExperimentManager:
    """
    Manages the lifecycle of A/B experiments.
    Responsibilities:
    - Track active experiments
    - Route requests to correct model versions
    - Log predictions with experiment context
    - Compute experiment results
    """
    def __init__(
        self,
        metrics: PerVersionMetrics,
        logger: ExperimentLogger,
        analyzer: StatisticalAnalyzer,
    ):
        self.metrics = metrics
        self.logger = logger
        self.analyzer = analyzer
        self._active_experiments: Dict[str, ExperimentConfig] = {}
        self._splitters: Dict[str, TrafficSplitter] = {}
    def start_experiment(
        self,
        config: ExperimentConfig,
    ) -> None:
        """Register and start a new experiment."""
        if config.experiment_id in self._active_experiments:
            raise ValueError(
                f"Experiment {config.experiment_id} already exists"
            )
        self._active_experiments[config.experiment_id] = config
        # Create traffic splitter for this experiment
        self._splitters[config.experiment_id] = TrafficSplitter(
            allocations=[
                TrafficAllocation(
                    version_id=config.control_version,
                    weight_percent=100.0 - config.traffic_split,
                    description="Control group",
                ),
                TrafficAllocation(
                    version_id=config.treatment_version,
                    weight_percent=config.traffic_split,
                    description="Treatment group",
                ),
            ]
        )
    def route_request(
        self,
        experiment_id: str,
        user_id: str,
    ) -> Tuple[str, dict]:
        """
        Route a request to the appropriate model version.
        Returns:
            (version_id, routing_metadata)
        """
        if experiment_id not in self._splitters:
            raise ValueError(f"Unknown experiment: {experiment_id}")
        return self._splitters[experiment_id].route(user_id)
    def log_prediction(
        self,
        experiment_id: str,
        request_id: str,
        user_id: str,
        model_version: str,
        allocation_hash: int,
        input_data: Any,
        prediction: Any,
        latency_ms: float,
        **kwargs,
    ) -> None:
        """Log a prediction with experiment context."""
        self.logger.log_prediction(
            experiment_id=experiment_id,
            request_id=request_id,
            user_id=user_id,
            model_version=model_version,
            allocation_hash=allocation_hash,
            input_data=input_data,
            prediction=prediction,
            latency_ms=latency_ms,
            **kwargs,
        )
        # Also record in metrics
        self.metrics.record_latency(
            version_id=model_version,
            latency_ms=latency_ms,
            success=kwargs.get("success", True),
        )
    def analyze_experiment(
        self,
        experiment_id: str,
    ) -> dict:
        """
        Perform statistical analysis of experiment results.
        Returns analysis including:
        - Sample sizes
        - Metric comparisons
        - Statistical significance
        - Recommendation
        """
        if experiment_id not in self._active_experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")
        config = self._active_experiments[experiment_id]
        # Get metrics comparison
        comparison = self.metrics.compare_versions(
            config.control_version,
            config.treatment_version,
        )
        # Get latency samples for statistical test
        # (In production, you'd read from the log files)
        # For now, use summary statistics
        # Determine if we have enough samples
        min_samples = min(
            comparison["sample_count_a"],
            comparison["sample_count_b"],
        )
        has_sufficient_samples = min_samples >= config.minimum_sample_size
        # Generate recommendation
        if not has_sufficient_samples:
            recommendation = (
                f"Continue experiment. "
                f"Current samples: {min_samples}, "
                f"required: {config.minimum_sample_size}"
            )
        else:
            # Check if treatment is better
            latency_improvement = (
                comparison["latency_a"].get("p50", 0) - 
                comparison["latency_b"].get("p50", 0)
            )
            error_diff = (
                comparison["error_rate_b"] - 
                comparison["error_rate_a"]
            )
            if latency_improvement > 0 and error_diff <= 0:
                recommendation = (
                    "TREATMENT WINS: Lower latency, no error increase. "
                    "Recommend promoting to 100%."
                )
            elif latency_improvement < 0 or error_diff > 0.01:
                recommendation = (
                    "CONTROL WINS: Treatment shows regression. "
                    "Recommend rollback."
                )
            else:
                recommendation = (
                    "NO SIGNIFICANT DIFFERENCE. "
                    "Consider extending experiment or accepting either version."
                )
        return {
            "experiment_id": experiment_id,
            "config": config.to_dict(),
            "has_sufficient_samples": has_sufficient_samples,
            "sample_size": min_samples,
            "comparison": comparison,
            "recommendation": recommendation,
        }
```
---
## Phase 6: Automatic Canary Rollback

![Sample Size vs Detectable Effect](./diagrams/tdd-diag-031.svg)

![Automatic Canary Rollback Trigger](./diagrams/diag-m4-rollback-trigger.svg)

![Automatic Canary Rollback Trigger](./diagrams/tdd-diag-032.svg)

The safety net for canary deployments: **automatic rollback when the canary shows problems**.
```python
from dataclasses import dataclass
from typing import Callable, Awaitable, Optional, Tuple
import asyncio
import logging
logger = logging.getLogger(__name__)
@dataclass
class HealthCheckConfig:
    """Configuration for canary health checks."""
    check_interval_seconds: float = 30.0
    min_samples_required: int = 50
    # Thresholds (canary relative to stable)
    max_error_rate_multiplier: float = 2.0
    max_latency_p50_multiplier: float = 1.5
    max_latency_p99_multiplier: float = 2.0
    # Consecutive failures before rollback
    consecutive_failures_threshold: int = 3
class CanaryHealthChecker:
    """
    Monitors canary health and triggers rollback on problems.
    Health checks compare canary metrics against stable baseline:
    - Error rate: canary must not exceed stable by > threshold
    - Latency: canary must not exceed stable by > threshold
    If thresholds are exceeded for N consecutive checks, rollback.
    """
    def __init__(
        self,
        metrics: PerVersionMetrics,
        config: HealthCheckConfig,
        on_rollback: Callable[[str, str], Awaitable[None]],
    ):
        self.metrics = metrics
        self.config = config
        self.on_rollback = on_rollback
        self._consecutive_failures = 0
        self._last_check_result: Optional[dict] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
    async def check_health(
        self,
        stable_version: str,
        canary_version: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Perform a health check comparing canary to stable.
        Returns:
            (is_healthy, reason_if_unhealthy)
        """
        # Get metrics for both versions
        stable_latency = self.metrics.get_latency_percentiles(stable_version)
        canary_latency = self.metrics.get_latency_percentiles(canary_version)
        stable_error_rate = self.metrics.get_error_rate(stable_version)
        canary_error_rate = self.metrics.get_error_rate(canary_version)
        stable_count = self.metrics._request_counts.get(stable_version, 0)
        canary_count = self.metrics._request_counts.get(canary_version, 0)
        # Check minimum samples
        if canary_count < self.config.min_samples_required:
            return True, None  # Not enough data, assume healthy
        reasons = []
        # Check error rate
        if stable_error_rate > 0:
            error_multiplier = canary_error_rate / stable_error_rate
            if error_multiplier > self.config.max_error_rate_multiplier:
                reasons.append(
                    f"Error rate {canary_error_rate:.2%} is "
                    f"{error_multiplier:.1f}x stable ({stable_error_rate:.2%})"
                )
        elif canary_error_rate > 0.01:  # Stable has 0 errors, canary has > 1%
            reasons.append(
                f"Canary error rate {canary_error_rate:.2%} while stable has 0%"
            )
        # Check latency p50
        stable_p50 = stable_latency.get("p50", 0)
        canary_p50 = canary_latency.get("p50", 0)
        if stable_p50 > 0:
            latency_multiplier = canary_p50 / stable_p50
            if latency_multiplier > self.config.max_latency_p50_multiplier:
                reasons.append(
                    f"Latency p50 {canary_p50:.1f}ms is "
                    f"{latency_multiplier:.1f}x stable ({stable_p50:.1f}ms)"
                )
        # Check latency p99
        stable_p99 = stable_latency.get("p99", 0)
        canary_p99 = canary_latency.get("p99", 0)
        if stable_p99 > 0:
            latency_multiplier = canary_p99 / stable_p99
            if latency_multiplier > self.config.max_latency_p99_multiplier:
                reasons.append(
                    f"Latency p99 {canary_p99:.1f}ms is "
                    f"{latency_multiplier:.1f}x stable ({stable_p99:.1f}ms)"
                )
        self._last_check_result = {
            "timestamp": time.time(),
            "stable_error_rate": stable_error_rate,
            "canary_error_rate": canary_error_rate,
            "stable_latency": stable_latency,
            "canary_latency": canary_latency,
            "stable_count": stable_count,
            "canary_count": canary_count,
        }
        if reasons:
            return False, "; ".join(reasons)
        return True, None
    async def start_monitoring(
        self,
        stable_version: str,
        canary_version: str,
    ) -> None:
        """Start background health monitoring."""
        if self._running:
            return
        self._running = True
        self._consecutive_failures = 0
        async def monitor_loop():
            while self._running:
                try:
                    is_healthy, reason = await self.check_health(
                        stable_version,
                        canary_version,
                    )
                    if not is_healthy:
                        self._consecutive_failures += 1
                        logger.warning(
                            f"[HealthCheck] Failure {self._consecutive_failures}/"
                            f"{self.config.consecutive_failures_threshold}: {reason}"
                        )
                        if (
                            self._consecutive_failures >= 
                            self.config.consecutive_failures_threshold
                        ):
                            logger.error(
                                f"[HealthCheck] Triggering rollback: {reason}"
                            )
                            await self.on_rollback(canary_version, reason)
                            self._running = False
                            break
                    else:
                        self._consecutive_failures = 0
                        logger.debug("[HealthCheck] Healthy")
                    await asyncio.sleep(self.config.check_interval_seconds)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"[HealthCheck] Error: {e}", exc_info=True)
                    await asyncio.sleep(1.0)
        self._task = asyncio.create_task(monitor_loop())
    async def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    def get_last_check_result(self) -> Optional[dict]:
        """Get the result of the last health check."""
        return self._last_check_result
```
---
## Putting It All Together: The Complete A/B Testing Server
```python
"""
ab_testing_server.py
Complete A/B testing and canary deployment server.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional
import uuid
import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, PlainTextResponse
import orjson
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Global components
metrics: Optional[PerVersionMetrics] = None
splitter: Optional[TrafficSplitter] = None
experiment_manager: Optional[ExperimentManager] = None
rollout_controller: Optional[CanaryRolloutController] = None
health_checker: Optional[CanaryHealthChecker] = None
# Model versions (loaded elsewhere)
models: dict = {}  # version_id -> model
async def inference_with_version(
    version_id: str,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    """Run inference with a specific model version."""
    model = models.get(version_id)
    if model is None:
        raise ValueError(f"Model version {version_id} not loaded")
    with torch.no_grad():
        return model(input_tensor)
async def health_check_callback(
    stable_version: str,
    canary_version: str,
) -> tuple:
    """Callback for health checking."""
    if health_checker is None:
        return True, None
    return await health_checker.check_health(stable_version, canary_version)
async def rollback_callback(
    canary_version: str,
    reason: str,
) -> None:
    """Callback for automatic rollback."""
    logger.error(f"[Rollback] Triggered for {canary_version}: {reason}")
    # Reset traffic to 100% stable
    if splitter and rollout_controller:
        stable = rollout_controller.config.stable_version
        splitter.update_weights([
            TrafficAllocation(stable, 100.0, "Stable (after rollback)"),
        ])
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global metrics, splitter, experiment_manager, health_checker
    logger.info("[Startup] Initializing A/B testing server...")
    # Initialize components
    metrics = PerVersionMetrics()
    # Default: 100% to stable version
    splitter = TrafficSplitter(allocations=[
        TrafficAllocation("v1.0.0", 100.0, "Stable production model"),
    ])
    experiment_logger = ExperimentLogger(log_dir="experiments")
    analyzer = StatisticalAnalyzer()
    experiment_manager = ExperimentManager(
        metrics=metrics,
        logger=experiment_logger,
        analyzer=analyzer,
    )
    health_checker = CanaryHealthChecker(
        metrics=metrics,
        config=HealthCheckConfig(),
        on_rollback=rollback_callback,
    )
    logger.info("[Startup] Ready")
    yield  # Server runs here
    # Shutdown
    logger.info("[Shutdown] Finalizing experiment logs...")
    experiment_logger.finalize()
    if health_checker:
        await health_checker.stop()
    logger.info("[Shutdown] Complete")
app = FastAPI(lifespan=lifespan, title="A/B Testing Model Server")
@app.get("/health/liveness")
async def liveness():
    return {"status": "alive"}
@app.get("/health/readiness")
async def readiness():
    return {"status": "ready"}
@app.post("/predict")
async def predict(request: Request):
    """
    Inference endpoint with A/B routing.
    Headers:
        X-User-ID: User identifier for consistent routing
        X-Experiment-ID: Experiment to participate in (optional)
    If no experiment is specified, routes to default (100% stable).
    """
    # Get user ID for routing
    user_id = request.headers.get("X-User-ID", str(uuid.uuid4()))
    experiment_id = request.headers.get("X-Experiment-ID")
    # Determine version
    if experiment_id and experiment_manager:
        version_id, routing_meta = experiment_manager.route_request(
            experiment_id, user_id
        )
    else:
        version_id, routing_meta = splitter.route(user_id)
    # Parse request
    body = await request.body()
    try:
        data = orjson.loads(body)
    except orjson.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    input_data = data.get("input")
    if input_data is None:
        raise HTTPException(status_code=400, detail="Missing 'input' field")
    # Convert to tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    # Run inference
    import time
    start_time = time.perf_counter()
    try:
        output = await inference_with_version(version_id, input_tensor)
        latency_ms = (time.perf_counter() - start_time) * 1000
        success = True
        error_message = None
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        success = False
        error_message = str(e)
        raise HTTPException(status_code=500, detail=str(e))
    # Record metrics
    metrics.record_latency(version_id, latency_ms, success)
    # Log to experiment
    if experiment_id and experiment_manager:
        experiment_manager.log_prediction(
            experiment_id=experiment_id,
            request_id=data.get("request_id", str(uuid.uuid4())[:8]),
            user_id=user_id,
            model_version=version_id,
            allocation_hash=routing_meta.get("user_hash", 0),
            input_data=input_data,
            prediction=output.tolist() if output is not None else None,
            latency_ms=latency_ms,
            success=success,
            error_message=error_message,
        )
    # Return response
    response_data = {
        "prediction": output.tolist() if output is not None else None,
        "model_version": version_id,
        "latency_ms": latency_ms,
        "routing": routing_meta,
    }
    return Response(
        content=orjson.dumps(response_data),
        media_type="application/json",
    )
@app.get("/traffic")
async def get_traffic_config():
    """Get current traffic split configuration."""
    return splitter.to_dict()
@app.post("/traffic")
async def update_traffic_config(request: Request):
    """
    Update traffic split configuration.
    Body:
        {
            "allocations": [
                {"version_id": "v1.0.0", "weight_percent": 90.0},
                {"version_id": "v2.0.0", "weight_percent": 10.0}
            ]
        }
    """
    body = await request.body()
    data = orjson.loads(body)
    allocations = [
        TrafficAllocation(
            version_id=a["version_id"],
            weight_percent=a["weight_percent"],
            description=a.get("description", ""),
        )
        for a in data["allocations"]
    ]
    try:
        splitter.update_weights(allocations)
        return {"status": "updated", "config": splitter.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
@app.post("/canary/start")
async def start_canary(request: Request):
    """
    Start a canary rollout.
    Body:
        {
            "stable_version": "v1.0.0",
            "canary_version": "v2.0.0",
            "stages": [5, 10, 25, 50, 100],
            "stage_duration_seconds": 300
        }
    """
    global rollout_controller
    body = await request.body()
    data = orjson.loads(body)
    config = RolloutConfig(
        stable_version=data["stable_version"],
        canary_version=data["canary_version"],
        stages=data.get("stages", [5, 10, 25, 50, 100]),
        stage_duration_seconds=data.get("stage_duration_seconds", 300),
    )
    rollout_controller = CanaryRolloutController(
        config=config,
        splitter=splitter,
        health_checker=health_check_callback,
    )
    await rollout_controller.start_rollout()
    return {"status": "started", "config": config.__dict__}
@app.get("/canary/status")
async def canary_status():
    """Get current canary rollout status."""
    if rollout_controller is None:
        return {"status": "no_active_rollout"}
    return rollout_controller.get_state()
@app.post("/canary/rollback")
async def manual_rollback():
    """Manually trigger canary rollback."""
    if rollout_controller is None:
        raise HTTPException(status_code=400, detail="No active rollout")
    await rollout_controller._trigger_rollback("Manual rollback requested")
    return {"status": "rolled_back"}
@app.get("/metrics")
async def get_metrics():
    """Get metrics in Prometheus format."""
    return PlainTextResponse(
        content=metrics.to_prometheus_format(),
        media_type="text/plain",
    )
@app.get("/metrics/compare")
async def compare_versions(version_a: str, version_b: str):
    """Compare metrics between two versions."""
    return metrics.compare_versions(version_a, version_b)
@app.post("/experiment/start")
async def start_experiment(request: Request):
    """
    Start an A/B experiment.
    Body:
        {
            "experiment_id": "exp-001",
            "name": "Model v2 Performance Test",
            "control_version": "v1.0.0",
            "treatment_version": "v2.0.0",
            "traffic_split": 10.0,
            "minimum_sample_size": 10000
        }
    """
    body = await request.body()
    data = orjson.loads(body)
    from datetime import datetime
    config = ExperimentConfig(
        experiment_id=data["experiment_id"],
        name=data["name"],
        description=data.get("description", ""),
        start_time=datetime.now(),
        control_version=data["control_version"],
        treatment_version=data["treatment_version"],
        traffic_split=data["traffic_split"],
        hypothesis=data.get("hypothesis", ""),
        success_metric=data.get("success_metric", "latency_p50"),
        minimum_sample_size=data.get("minimum_sample_size", 10000),
    )
    experiment_manager.start_experiment(config)
    return {"status": "started", "experiment_id": config.experiment_id}
@app.get("/experiment/{experiment_id}/analyze")
async def analyze_experiment(experiment_id: str):
    """Analyze experiment results."""
    return experiment_manager.analyze_experiment(experiment_id)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
---
## Knowledge Cascade: What You've Unlocked
### Immediate Applications
- **Milestone 5 (Monitoring & Drift Detection)**: The per-version metrics and experiment logging you've built are the foundation for drift detection. The KS test for distribution comparison is a natural extension.
- **Production deployment**: The canary controller you've built is production-ready. Add integration with your deployment pipeline (Jenkins, GitHub Actions) for automated rollouts.
### Cross-Domain Connections
**1. Consistent Hashing → Distributed Caching**
The `hash(user_id) % 100` pattern is identical to **CDN edge selection** and **database sharding**:
| CDN Edge Selection | Traffic Splitting |
|-------------------|-------------------|
| `hash(user_ip) % edge_count` | `hash(user_id) % 100` |
| User always hits same edge | User always sees same model |
| Edge failure → rehash | Model failure → re-route |
The principle: **deterministic mapping minimizes cache invalidation.** Whether it's CDN caches or user experiences, consistency matters.
**2. Statistical Significance → Clinical Trials**
The t-test and sample size calculations you learned are the same used in **FDA clinical trials**:
- Phase 1: Safety (5-10% exposure) → Canary stage 1
- Phase 2: Efficacy (25-50% exposure) → Canary stages 2-3
- Phase 3: Large-scale (100% exposure) → Full rollout
Both domains use **progressive exposure with statistical gates**. The difference: FDA requires p < 0.05 for approval; your canary might use looser thresholds.
**3. Canary Deployment → Circuit Breakers**
The automatic rollback on error spike is a **circuit breaker pattern**:
| Circuit Breaker | Canary Rollback |
|-----------------|-----------------|
| Error rate > threshold → open circuit | Error rate > 2x baseline → rollback |
| Open circuit → fail fast | Rollback → route to stable |
| Half-open → test recovery | Next rollout → try again |
Both patterns **fail fast to preserve system stability**. The canary controller is a circuit breaker at the deployment layer.
**4. Confounding Variables → A/A Testing**
A critical technique in experiment validity: **A/A tests**. Run an experiment where both groups get the same model. If you see "significant" differences, your experiment setup is broken.
This catches:
- Non-deterministic routing (same user seeing different models)
- Sample ratio mismatch (groups have different sizes than expected)
- Metric calculation bugs
In ML experiments, **always run an A/A test first**. If the null hypothesis isn't preserved when there's no difference, your experiment is invalid.
### Forward Connections
With A/B testing mastered, you can now:
- **Build multi-armed bandits** that dynamically shift traffic to better-performing models without fixed experiment windows
- **Implement stratified sampling** that ensures representative user populations in each experiment group
- **Create experiment dashboards** that show real-time p-values and confidence intervals
- **Design holdout groups** that never see new models, providing a long-term baseline
---
## Summary: The Experimentation Mindset
ML A/B testing is not "random assignment with metrics." It's a **rigorous statistical discipline**:
1. **Deterministic routing** — same user always sees same model, or your experiment is invalid
2. **Sample size matters** — detecting 1% improvement requires ~15,000 samples per group; premature conclusions are false conclusions
3. **Control confounds** — change only the model; preprocessing, features, and infrastructure must stay constant
4. **Gradual rollout** — canary deployment limits blast radius; 5% → 10% → 25% → 50% → 100%
5. **Automatic rollback** — error rate spikes? bail out fast; don't wait for human intervention
6. **Full audit trail** — every prediction logged with experiment context; post-hoc analysis is possible
You now have the infrastructure for **statistically valid ML experimentation**—the bridge between model development and production impact.
---
[[CRITERIA_JSON: {"milestone_id": "ml-model-serving-m4", "criteria": ["Traffic is split between model versions using TrafficSplitter with configurable percentage weights (e.g., 90/10) that sum to 100%", "User routing is deterministic via consistent hashing (SHA-256 hash of user_id modulo 100) guaranteeing same user always routes to same version", "Traffic weights are adjustable at runtime via update_weights() method without server restart; in-flight requests unaffected", "Per-version metrics tracked include latency percentiles (p50, p90, p95, p99), error rate (errors/total), throughput (RPS), and prediction distribution histogram", "Gradual rollout via CanaryRolloutController progresses through configurable stages (default: 5% -> 10% -> 25% -> 50% -> 100%) with stage duration and minimum sample requirements", "Automatic rollback triggers when canary error rate exceeds stable by configurable multiplier (default 2x) for consecutive health check failures (default 3)", "Experiment metadata (experiment_id, model_version, user_hash, timestamp) is logged for every prediction via PredictionLog with JSON lines format", "StatisticalAnalyzer provides two-sample t-test for latency comparison, proportion test for error rates, and sample size calculation for experiment planning", "CanaryHealthChecker monitors canary vs stable with configurable thresholds for error rate multiplier (default 2x) and latency multiplier (default 1.5x for p50, 2x for p99)", "PerVersionMetrics stores latency and prediction samples per version in deques with configurable window size (default 10,000) for percentile calculation", "ExperimentManager coordinates traffic splitting, prediction logging, and statistical analysis for active experiments", "Rollback callback resets traffic to 100% stable version on automatic rollback trigger"]}]
<!-- END_MS -->


<!-- MS_ID: ml-model-serving-m5 -->
# Monitoring & Drift Detection
## The Silent Failure
You've built a model serving system with batching, versioning, and A/B testing. Your metrics look perfect: p50 latency is 45ms, error rate is 0.01%, throughput is 500 requests per second. The dashboards are green. Everything is fine.
**Until it isn't.**
Six weeks after deployment, a subtle shift happens. Users start receiving slightly different recommendations. Not wrong—just different. No alerts fire. No errors appear in logs. But engagement metrics slowly decline: click-through rate drops 8%, session duration falls 12%, conversion rate slips 5%.
What happened?
Your input data drifted. The feature distributions that your model was trained on—user ages, product prices, session lengths—shifted imperceptibly. Users from a new demographic arrived. Seasonal patterns changed. Your model, trained on January's data, is now making predictions on March's reality.

![system-overview](./diagrams/system-overview.svg)

![Drift Detection Pipeline](./diagrams/diag-m5-drift-detection-flow.svg)

The misconception is seductive: "If the model doesn't crash, it's working fine. Latency metrics tell you everything you need to know."
**The reality**: Latency and error metrics measure system health, not model health. A model can return predictions in 50ms with zero errors while being catastrophically wrong. Silent degradation is the real danger—by the time business metrics collapse, the damage is done.
This milestone is about **seeing the invisible**: detecting drift before it hurts users.
---
## The Fundamental Tension: Detection Sensitivity vs. Alert Fatigue
Every monitoring system faces a core constraint:
**You want to detect real problems early, but you're constrained by:**
1. **Statistical noise**: Random variation in metrics creates false positives
2. **Alert fatigue**: Too many false alerts → engineers ignore real ones
3. **Sampling overhead**: Analyzing every request is expensive; sampling must be statistically valid
4. **Baseline staleness**: Training distributions were captured at one point in time; how stale is "too stale"?

![Alert Threshold Tuning: Sensitivity vs Noise](./diagrams/diag-m5-alert-threshold-tuning.svg)

The tension manifests as a tradeoff curve:
| Threshold | False Positives | False Negatives | Alert Response |
|-----------|-----------------|-----------------|----------------|
| p < 0.10 (sensitive) | High (many false alarms) | Low (catch everything) | Engineers ignore alerts |
| p < 0.01 (moderate) | Medium | Medium | Some alerts ignored |
| p < 0.001 (conservative) | Low | High (miss real drift) | Alerts taken seriously |
**The brutal math**: If you monitor 50 features and use p < 0.05 as your threshold, you expect 2.5 false alerts per check by random chance alone. At one check per hour, that's 60 false alerts per day. Your team stops looking.
The solution isn't picking a "perfect" threshold—it's **understanding the statistical machinery** and **tuning based on your data**.
---
## The Three-Level View
**Level 1 — Application Layer (What You Expose)**:
- Prometheus metrics for latency, throughput, errors
- Prediction distribution histograms
- Drift scores per feature
**Level 2 — Statistical Layer (How You Detect)**:
- Sliding window sampling
- KS test for distribution comparison
- P-value aggregation and thresholding
**Level 3 — Infrastructure Layer (What Stores It)**:
- Training baseline storage (captured at deployment)
- Time-series database for metrics
- Alert routing (PagerDuty, Slack, email)
---
## Phase 1: Latency Monitoring with Prometheus Histograms
### The Histogram Strategy
Latency isn't a single number—it's a **distribution**. A single "average latency" hides the truth: 99% of requests at 40ms with 1% at 5 seconds gives the same average as 100% at 90ms. But the user experience is completely different.

![Latency Histogram and Percentile Calculation](./diagrams/diag-m5-latency-histogram.svg)

Prometheus handles this with **histograms**: you define buckets, and each observation increments the appropriate bucket counter.
```python
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time
import math
@dataclass
class HistogramBucket:
    """A single bucket in a latency histogram."""
    upper_bound: float  # Milliseconds (exclusive upper bound)
    count: int = 0
@dataclass
class LatencyHistogram:
    """
    Prometheus-style latency histogram.
    Design: Instead of storing every observation (memory explosion),
    we bucket observations into predefined ranges. This enables
    percentile calculation with bounded memory.
    The key insight: bucket boundaries should match your SLAs.
    If your p99 SLA is 200ms, have buckets at 100, 150, 200, 250, etc.
    """
    buckets: List[HistogramBucket]
    sum_total: float = 0.0  # Sum of all observations (for mean)
    count_total: int = 0
    @classmethod
    def create_default(cls) -> "LatencyHistogram":
        """
        Create histogram with sensible default buckets.
        Bucket boundaries chosen for typical inference workloads:
        - Sub-10ms: excellent (cache hit territory)
        - 10-50ms: good (typical inference)
        - 50-100ms: acceptable
        - 100-500ms: slow but acceptable
        - 500ms+: problem territory
        """
        # Bucket upper bounds in milliseconds
        # +Inf bucket is implicit (catches everything above last bucket)
        bounds = [1, 5, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 2000, 5000]
        buckets = [HistogramBucket(upper_bound=b) for b in bounds]
        return cls(buckets=buckets)
    @classmethod
    def create_for_sla(cls, p99_sla_ms: float) -> "LatencyHistogram":
        """
        Create histogram with buckets aligned to your SLA.
        Strategy: Put buckets at 0.5x, 0.75x, 1x, 1.25x, 1.5x, 2x your SLA.
        This ensures you can detect when latency approaches or exceeds your SLA.
        """
        multipliers = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
        bounds = [p99_sla_ms * m for m in multipliers]
        # Add some fine-grained buckets below the SLA
        fine_bounds = [1, 5, 10, 25, 50]
        all_bounds = sorted(set(fine_bounds + bounds))
        buckets = [HistogramBucket(upper_bound=b) for b in all_bounds]
        return cls(buckets=buckets)
    def observe(self, latency_ms: float) -> None:
        """
        Record a latency observation.
        Increments all buckets where latency < bucket.upper_bound.
        This is the Prometheus cumulative histogram convention.
        """
        self.sum_total += latency_ms
        self.count_total += 1
        # Increment all buckets where this observation falls
        # (Prometheus uses cumulative buckets)
        for bucket in self.buckets:
            if latency_ms < bucket.upper_bound:
                bucket.count += 1
    def percentile(self, p: float) -> float:
        """
        Calculate the p-th percentile from histogram buckets.
        Algorithm:
        1. Find the bucket containing the p-th percentile
        2. Interpolate within that bucket
        Note: This is an approximation. Histogram percentiles are
        always approximate, but the error is bounded by bucket width.
        """
        if self.count_total == 0:
            return 0.0
        # Target count for this percentile
        target_count = (p / 100.0) * self.count_total
        # Find bucket containing this percentile
        # We need to use the cumulative counts
        prev_count = 0
        prev_bound = 0.0
        for bucket in self.buckets:
            if bucket.count >= target_count:
                # Percentile is in this bucket
                # Interpolate linearly
                if bucket.count == prev_count:
                    # All observations at same value
                    return bucket.upper_bound
                # Linear interpolation within bucket
                # Assume uniform distribution within bucket
                bucket_width = bucket.upper_bound - prev_bound
                observations_in_bucket = bucket.count - prev_count
                observations_needed = target_count - prev_count
                fraction = observations_needed / observations_in_bucket
                return prev_bound + (bucket_width * fraction)
            prev_count = bucket.count
            prev_bound = bucket.upper_bound
        # Percentile is above all buckets
        # Return a high estimate based on the last bucket
        return self.buckets[-1].upper_bound * 2  # Rough estimate
    def mean(self) -> float:
        """Calculate mean latency."""
        if self.count_total == 0:
            return 0.0
        return self.sum_total / self.count_total
    def to_prometheus_format(self, metric_name: str, labels: Dict[str, str]) -> str:
        """
        Export histogram in Prometheus text exposition format.
        Prometheus expects:
        - {metric_name}_bucket{le="upper_bound"} count
        - {metric_name}_sum total_sum
        - {metric_name}_count total_count
        """
        label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
        if label_str:
            label_str = "{" + label_str + "}"
        lines = []
        # Bucket counts (cumulative)
        for bucket in self.buckets:
            le_label = f'{{le="{bucket.upper_bound}"'
            if labels:
                le_label += "," + ",".join(f'{k}="{v}"' for k, v in labels.items())
            le_label += "}"
            lines.append(f"{metric_name}_bucket{le_label} {bucket.count}")
        # +Inf bucket (equals total count)
        le_inf = '{le="+Inf"'
        if labels:
            le_inf += "," + ",".join(f'{k}="{v}"' for k, v in labels.items())
        le_inf += "}"
        lines.append(f"{metric_name}_bucket{le_inf} {self.count_total}")
        # Sum and count
        lines.append(f"{metric_name}_sum{label_str} {self.sum_total:.2f}")
        lines.append(f"{metric_name}_count{label_str} {self.count_total}")
        return "\n".join(lines)
class PerVersionLatencyTracker:
    """
    Tracks latency histograms per model version.
    This is critical for A/B testing: you need to compare
    latency distributions between versions, not just averages.
    """
    def __init__(self):
        self._histograms: Dict[str, LatencyHistogram] = {}
        self._request_counts: Dict[str, int] = defaultdict(int)
        self._error_counts: Dict[str, int] = defaultdict(int)
    def record(
        self,
        version_id: str,
        latency_ms: float,
        success: bool = True
    ) -> None:
        """Record a latency observation for a model version."""
        if version_id not in self._histograms:
            self._histograms[version_id] = LatencyHistogram.create_default()
        self._histograms[version_id].observe(latency_ms)
        self._request_counts[version_id] += 1
        if not success:
            self._error_counts[version_id] += 1
    def get_percentiles(
        self,
        version_id: str,
        percentiles: List[float] = [50, 90, 95, 99]
    ) -> Dict[str, float]:
        """Get latency percentiles for a version."""
        if version_id not in self._histograms:
            return {f"p{p}": 0.0 for p in percentiles}
        histogram = self._histograms[version_id]
        return {f"p{p}": histogram.percentile(p) for p in percentiles}
    def get_error_rate(self, version_id: str) -> float:
        """Get error rate for a version (0.0 to 1.0)."""
        total = self._request_counts.get(version_id, 0)
        if total == 0:
            return 0.0
        return self._error_counts.get(version_id, 0) / total
    def get_throughput_rps(
        self,
        version_id: str,
        window_seconds: float = 60.0
    ) -> float:
        """
        Calculate requests per second.
        Note: This is a simplification. For accurate RPS,
        you'd need to track timestamps and use a sliding window.
        """
        # In production, you'd track timestamps and compute over window
        # For now, return a placeholder
        return self._request_counts.get(version_id, 0) / max(1, window_seconds)
    def to_prometheus_format(self) -> str:
        """Export all metrics in Prometheus format."""
        lines = []
        for version_id, histogram in self._histograms.items():
            # Latency histogram
            lines.append(histogram.to_prometheus_format(
                "model_inference_latency_ms",
                {"version": version_id}
            ))
            # Error rate
            error_rate = self.get_error_rate(version_id)
            lines.append(
                f'model_error_rate{{version="{version_id}"}} {error_rate:.4f}'
            )
            # Request count
            count = self._request_counts.get(version_id, 0)
            lines.append(
                f'model_requests_total{{version="{version_id}"}} {count}'
            )
        return "\n".join(lines)
```
### Cross-Domain Connection: Database Query Latency
This histogram approach is identical to **database query performance monitoring**:
| Database Monitoring | ML Inference Monitoring |
|--------------------|------------------------|
| Query latency buckets | Inference latency buckets |
| p99 SLA tracking | p99 SLA tracking |
| Slow query log | Slow inference alert |
| Query plan changes | Model version changes |
The pattern: **distributions, not averages.** The same p99-focused monitoring that keeps databases fast keeps ML inference fast.
---
## Phase 2: Prediction Distribution Monitoring
### Why Predictions Drift
Even if input data stays the same, **prediction distributions can shift**. This happens when:
1. **Model confidence degrades**: Model becomes less certain, predictions spread out
2. **Edge cases increase**: New inputs trigger different prediction patterns
3. **Calibration drifts**: Probabilities no longer match reality

![Prediction Distribution Monitoring](./diagrams/diag-m5-prediction-distribution.svg)

```python
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import statistics
import math
@dataclass
class PredictionSample:
    """A single prediction observation."""
    timestamp: float
    value: float  # The prediction (probability, score, or class)
    input_hash: str  # Hash of input features (for debugging)
    version_id: str
@dataclass
class PredictionDistributionMonitor:
    """
    Monitors prediction output distributions over time.
    The key insight: prediction distributions should be stable
    if the model is healthy. A shift in the prediction distribution
    often precedes a shift in business metrics.
    For classification: Monitor class probability distributions
    For regression: Monitor prediction value distribution
    For embeddings: Monitor distance from centroid
    """
    window_size: int = 10000  # Number of samples to keep
    n_bins: int = 20  # Histogram bins for distribution
    # Per-version tracking
    _samples: Dict[str, deque] = field(default_factory=dict)
    _baseline_histograms: Dict[str, Dict[str, float]] = field(default_factory=dict)
    def record(
        self,
        version_id: str,
        prediction: float,
        input_hash: str = ""
    ) -> None:
        """Record a prediction sample."""
        if version_id not in self._samples:
            self._samples[version_id] = deque(maxlen=self.window_size)
        self._samples[version_id].append(PredictionSample(
            timestamp=time.time(),
            value=prediction,
            input_hash=input_hash,
            version_id=version_id
        ))
    def compute_histogram(
        self,
        samples: deque,
        n_bins: int = None
    ) -> Dict[str, float]:
        """
        Compute histogram of prediction values.
        Returns a normalized histogram (frequencies sum to 1.0).
        """
        n_bins = n_bins or self.n_bins
        if not samples:
            return {}
        values = [s.value for s in samples]
        if not values:
            return {}
        min_val = min(values)
        max_val = max(values)
        # Handle constant predictions
        if min_val == max_val:
            return {"bin_0": 1.0, "min": min_val, "max": max_val}
        bin_width = (max_val - min_val) / n_bins
        bins = [0] * n_bins
        for val in values:
            bin_idx = int((val - min_val) / bin_width)
            bin_idx = min(bin_idx, n_bins - 1)  # Handle max value
            bins[bin_idx] += 1
        total = len(values)
        return {
            **{f"bin_{i}": bins[i] / total for i in range(n_bins)},
            "min": min_val,
            "max": max_val,
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    def capture_baseline(self, version_id: str) -> Dict[str, float]:
        """
        Capture current distribution as baseline.
        Call this after model deployment, once the model has warmed up
        and is serving production traffic.
        """
        if version_id not in self._samples:
            return {}
        histogram = self.compute_histogram(self._samples[version_id])
        self._baseline_histograms[version_id] = histogram
        return histogram
    def compare_to_baseline(
        self,
        version_id: str
    ) -> Dict[str, Any]:
        """
        Compare current distribution to baseline.
        Returns metrics indicating how much the distribution has shifted.
        """
        if version_id not in self._samples:
            return {"error": "No samples for version"}
        if version_id not in self._baseline_histograms:
            return {"error": "No baseline for version"}
        current = self.compute_histogram(self._samples[version_id])
        baseline = self._baseline_histograms[version_id]
        # Compute distribution shift metrics
        metrics = {
            "current_mean": current.get("mean", 0),
            "baseline_mean": baseline.get("mean", 0),
            "mean_shift": abs(current.get("mean", 0) - baseline.get("mean", 0)),
            "mean_shift_percent": 0.0,
            "current_std": current.get("std", 0),
            "baseline_std": baseline.get("std", 0),
            "std_ratio": 0.0,
        }
        # Compute percentage shift (handle division by zero)
        if baseline.get("mean", 0) != 0:
            metrics["mean_shift_percent"] = (
                metrics["mean_shift"] / abs(baseline["mean"]) * 100
            )
        if baseline.get("std", 0) != 0:
            metrics["std_ratio"] = current.get("std", 1) / baseline["std"]
        # Compute histogram distance (total variation distance)
        tv_distance = 0.0
        for i in range(self.n_bins):
            bin_key = f"bin_{i}"
            current_freq = current.get(bin_key, 0)
            baseline_freq = baseline.get(bin_key, 0)
            tv_distance += abs(current_freq - baseline_freq)
        metrics["total_variation_distance"] = tv_distance / 2  # Normalize to [0, 1]
        # Interpretation
        if metrics["total_variation_distance"] > 0.3:
            metrics["status"] = "SIGNIFICANT_SHIFT"
            metrics["recommendation"] = (
                "Prediction distribution has shifted significantly. "
                "Investigate model behavior and input data."
            )
        elif metrics["total_variation_distance"] > 0.1:
            metrics["status"] = "MODERATE_SHIFT"
            metrics["recommendation"] = (
                "Prediction distribution shows moderate shift. "
                "Monitor closely."
            )
        else:
            metrics["status"] = "STABLE"
            metrics["recommendation"] = "Prediction distribution is stable."
        return metrics
    def get_percentile_prediction(
        self,
        version_id: str,
        percentile: float
    ) -> Optional[float]:
        """Get a percentile of the prediction distribution."""
        if version_id not in self._samples:
            return None
        values = sorted([s.value for s in self._samples[version_id]])
        if not values:
            return None
        idx = int(len(values) * percentile / 100)
        idx = min(idx, len(values) - 1)
        return values[idx]
```
### Cross-Domain Connection: A/B Test Monitoring
Prediction distribution monitoring is identical to **A/B test guardrail metrics**:
| A/B Testing | Prediction Monitoring |
|-------------|----------------------|
| Treatment shouldn't shift metric distributions | New model shouldn't shift prediction distributions |
| Guardrail metrics detect harm | Distribution shifts detect degradation |
| Statistical tests for significance | KS test for distribution comparison |
The principle: **distributions are the canary in the coal mine.** Shift them at your peril.
---
## Phase 3: Data Drift Detection with the KS Test
### The Statistical Foundation
> **🔑 Foundation: KS test and distribution comparison**
> 
> ## KS Test and Distribution Comparison
### What It IS
The Kolmogorov-Smirnov (KS) test is a **non-parametric** statistical test that compares two distributions. It measures the maximum distance between their cumulative distribution functions (CDFs).
For drift detection:
- **Null hypothesis (H₀)**: The two samples come from the same distribution
- **Alternative hypothesis (H₁)**: The distributions are different
- **Test statistic (D)**: Maximum vertical distance between CDFs
- **p-value**: Probability of observing this D if H₀ is true
### WHY You Need It Right Now
You need to detect when live input data has drifted from training data. But:
- You don't know the "shape" of your feature distributions (not all Gaussian)
- You need a test that works for any distribution
- You need a single number (p-value) to trigger alerts
The KS test provides exactly this: a **distribution-agnostic comparison** with a **well-defined p-value**.
### ONE Key Insight
**Think of the KS test like comparing two staircases.**
Each distribution is a staircase (the CDF). The KS statistic is the **largest vertical gap** between the staircases. If the gap is large, the distributions are different.
The p-value comes from: "How surprising is this gap if the distributions were actually the same?" Small p-value = surprising = likely different distributions.
For drift detection: p-value < 0.01 means the live data is **significantly different** from training data. Something changed.

![KS Test: Comparing Distributions](./diagrams/diag-m5-ks-test-intuition.svg)

```python
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import math
@dataclass
class FeatureBaseline:
    """
    Baseline statistics for a single feature.
    Captured at training time or deployment time, this represents
    the "expected" distribution of the feature.
    """
    feature_name: str
    samples: List[float]  # Training samples (or representative subset)
    min_val: float
    max_val: float
    mean: float
    std: float
    # Precomputed CDF for efficient comparison
    sorted_samples: List[float] = field(default_factory=list)
    @classmethod
    def from_samples(
        cls,
        feature_name: str,
        samples: List[float],
        max_baseline_samples: int = 10000
    ) -> "FeatureBaseline":
        """Create baseline from training samples."""
        # Downsample if too large
        if len(samples) > max_baseline_samples:
            import random
            samples = random.sample(samples, max_baseline_samples)
        sorted_samples = sorted(samples)
        return cls(
            feature_name=feature_name,
            samples=samples,
            min_val=min(samples) if samples else 0,
            max_val=max(samples) if samples else 0,
            mean=sum(samples) / len(samples) if samples else 0,
            std=cls._compute_std(samples),
            sorted_samples=sorted_samples,
        )
    @staticmethod
    def _compute_std(samples: List[float]) -> float:
        """Compute standard deviation."""
        if len(samples) < 2:
            return 0.0
        mean = sum(samples) / len(samples)
        variance = sum((x - mean) ** 2 for x in samples) / (len(samples) - 1)
        return math.sqrt(variance)
@dataclass
class KSTestResult:
    """Result of a KS test between two distributions."""
    feature_name: str
    statistic: float  # D statistic (0 to 1)
    p_value: float  # Probability under null hypothesis
    is_significant: bool  # True if p_value < threshold
    baseline_samples: int
    live_samples: int
    interpretation: str
class KSTestDetector:
    """
    Performs Kolmogorov-Smirnov test for drift detection.
    The KS test compares two empirical distributions by computing
    the maximum distance between their cumulative distribution functions.
    Advantages:
    - Non-parametric: works for any distribution shape
    - Sensitive to location, scale, and shape differences
    - Well-defined p-value for hypothesis testing
    Limitations:
    - More sensitive to differences near the median than tails
    - Requires sufficient samples in both distributions
    - For multivariate data, tests each dimension independently
    """
    @staticmethod
    def compute_cdf(sorted_samples: List[float]) -> List[Tuple[float, float]]:
        """
        Compute empirical CDF from sorted samples.
        Returns list of (value, cumulative_probability) tuples.
        """
        n = len(sorted_samples)
        if n == 0:
            return []
        cdf = []
        for i, val in enumerate(sorted_samples):
            prob = (i + 1) / n
            cdf.append((val, prob))
        return cdf
    @staticmethod
    def ks_statistic(
        sorted_samples1: List[float],
        sorted_samples2: List[float]
    ) -> float:
        """
        Compute the KS statistic (D) between two samples.
        D = max|CDF1(x) - CDF2(x)| for all x
        This is the maximum vertical distance between the two CDFs.
        """
        n1 = len(sorted_samples1)
        n2 = len(sorted_samples2)
        if n1 == 0 or n2 == 0:
            return 0.0
        # Merge and sort all unique values
        i, j = 0, 0
        max_diff = 0.0
        cdf1_val, cdf2_val = 0.0, 0.0
        while i < n1 or j < n2:
            # Get next value to process
            if i < n1 and (j >= n2 or sorted_samples1[i] <= sorted_samples2[j]):
                val = sorted_samples1[i]
                cdf1_val = (i + 1) / n1
                i += 1
            else:
                val = sorted_samples2[j]
                cdf2_val = (j + 1) / n2
                j += 1
            # Update max difference
            diff = abs(cdf1_val - cdf2_val)
            max_diff = max(max_diff, diff)
        return max_diff
    @staticmethod
    def ks_p_value(d_statistic: float, n1: int, n2: int) -> float:
        """
        Compute p-value for KS test.
        Under the null hypothesis (same distribution), the KS statistic
        follows a known distribution. This function approximates the p-value.
        The asymptotic approximation:
        p ≈ 2 * Σ (-1)^(k-1) * exp(-2 * k² * λ)
        where λ = (sqrt(n1*n2/(n1+n2)) * D)²
        For simplicity, we use a faster approximation.
        """
        if n1 == 0 or n2 == 0:
            return 1.0
        # Effective sample size
        n_eff = math.sqrt(n1 * n2 / (n1 + n2))
        # Lambda parameter
        lambda_val = (n_eff * d_statistic) ** 2
        # Approximate p-value using first few terms of series
        # p ≈ 2 * exp(-2λ) - 2 * exp(-8λ) + 2 * exp(-18λ) - ...
        p_value = 0.0
        for k in range(1, 101):  # 100 terms is plenty
            term = 2 * ((-1) ** (k - 1)) * math.exp(-2 * (k ** 2) * lambda_val)
            p_value += term
            # Early termination if terms become negligible
            if abs(term) < 1e-10:
                break
        # Clamp to [0, 1]
        return max(0.0, min(1.0, p_value))
    def test(
        self,
        baseline: FeatureBaseline,
        live_samples: List[float],
        significance_level: float = 0.01
    ) -> KSTestResult:
        """
        Perform KS test between baseline and live samples.
        Args:
            baseline: Training baseline for this feature
            live_samples: Recent samples from live traffic
            significance_level: p-value threshold for significance
        Returns:
            KSTestResult with statistic, p-value, and interpretation
        """
        n_baseline = len(baseline.sorted_samples)
        n_live = len(live_samples)
        if n_baseline == 0 or n_live == 0:
            return KSTestResult(
                feature_name=baseline.feature_name,
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                baseline_samples=n_baseline,
                live_samples=n_live,
                interpretation="Insufficient samples for comparison"
            )
        # Sort live samples
        sorted_live = sorted(live_samples)
        # Compute KS statistic
        d_statistic = self.ks_statistic(
            baseline.sorted_samples,
            sorted_live
        )
        # Compute p-value
        p_value = self.ks_p_value(d_statistic, n_baseline, n_live)
        # Determine significance
        is_significant = p_value < significance_level
        # Interpretation
        if is_significant:
            # Compute direction of drift
            live_mean = sum(live_samples) / n_live
            mean_diff = live_mean - baseline.mean
            if mean_diff > baseline.std * 0.5:
                direction = "higher"
            elif mean_diff < -baseline.std * 0.5:
                direction = "lower"
            else:
                direction = "different shape"
            interpretation = (
                f"SIGNIFICANT DRIFT DETECTED (p={p_value:.4f}). "
                f"Live distribution is {direction} than baseline. "
                f"KS statistic: {d_statistic:.4f}"
            )
        else:
            interpretation = (
                f"No significant drift (p={p_value:.4f}). "
                f"KS statistic: {d_statistic:.4f}"
            )
        return KSTestResult(
            feature_name=baseline.feature_name,
            statistic=d_statistic,
            p_value=p_value,
            is_significant=is_significant,
            baseline_samples=n_baseline,
            live_samples=n_live,
            interpretation=interpretation
        )
```
### Cross-Domain Connection: Quality Control in Manufacturing
The KS test for drift detection is identical to **statistical process control (SPC)** in manufacturing:
| Manufacturing QC | ML Drift Detection |
|------------------|-------------------|
| Monitor product dimensions | Monitor feature distributions |
| Detect process drift | Detect data drift |
| Statistical tests (t-test, KS) | Statistical tests (KS test) |
| Alert → investigate → correct | Alert → investigate → retrain |
The principle: **statistical tests are the universal language of "is this different?"** Whether it's widget dimensions or neural network inputs, the math is the same.
---
## Phase 4: Sampling Strategy for Drift Detection
### The Performance Imperative
You can't run a KS test on every request. The overhead would be catastrophic:
- KS test is O(n log n) for n samples
- With 10,000 baseline samples and 1,000 live samples, that's ~15,000 comparisons
- At 500 requests/second, you'd spend more time testing than serving
**Sampling** is the solution: analyze a subset of requests, not all of them.

![Sampling Strategy for Drift Detection](./diagrams/diag-m5-sampling-strategy.svg)

```python
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
import random
import time
@dataclass
class SamplingConfig:
    """Configuration for drift detection sampling."""
    # Sampling rate: 1 in N requests is sampled
    sample_rate: int = 1000  # Default: 1/1000 = 0.1%
    # Minimum samples before running test
    min_samples_for_test: int = 100
    # Maximum samples to keep per feature
    max_live_samples: int = 5000
    # Test interval: run drift test every N seconds
    test_interval_seconds: float = 300.0  # 5 minutes
@datadict
class FeatureSampleBuffer:
    """
    Buffer for sampled feature values.
    Uses reservoir sampling to maintain a representative sample
    of the live distribution, even with infinite stream.
    """
    feature_name: str
    max_size: int
    samples: List[float] = field(default_factory=list)
    total_seen: int = 0
    def add(self, value: float) -> None:
        """
        Add a sample using reservoir sampling.
        This ensures that after seeing N values, each value has
        an equal probability of being in the reservoir.
        Algorithm: Reservoir Sampling (Algorithm R)
        """
        self.total_seen += 1
        if len(self.samples) < self.max_size:
            # Reservoir not full, add directly
            self.samples.append(value)
        else:
            # Reservoir full, probabilistically replace
            # Probability of keeping new sample = max_size / total_seen
            j = random.randint(1, self.total_seen)
            if j <= self.max_size:
                self.samples[j - 1] = value
    def get_samples(self) -> List[float]:
        """Get current samples."""
        return self.samples.copy()
    def clear(self) -> None:
        """Clear the buffer."""
        self.samples = []
        self.total_seen = 0
class DriftDetectionSampler:
    """
    Samples incoming requests for drift detection.
    Key design decisions:
    1. Sample rate is configurable (default 1/1000)
    2. Each feature is sampled independently
    3. Reservoir sampling ensures representative samples
    4. Tests run on interval, not on every sample
    This balances detection accuracy with performance overhead.
    """
    def __init__(
        self,
        baselines: Dict[str, FeatureBaseline],
        config: SamplingConfig
    ):
        self.baselines = baselines
        self.config = config
        # Sample buffers per feature
        self._buffers: Dict[str, FeatureSampleBuffer] = {
            name: FeatureSampleBuffer(
                feature_name=name,
                max_size=config.max_live_samples
            )
            for name in baselines
        }
        # Request counter for sampling
        self._request_counter: int = 0
        # Last test time
        self._last_test_time: float = 0
        self._last_results: Dict[str, KSTestResult] = {}
        # KS test detector
        self._detector = KSTestDetector()
    def should_sample(self) -> bool:
        """
        Determine if this request should be sampled.
        Uses deterministic sampling: every Nth request is sampled.
        This is more efficient than random sampling and still
        provides representative samples.
        """
        self._request_counter += 1
        return self._request_counter % self.config.sample_rate == 0
    def record_features(
        self,
        features: Dict[str, float]
    ) -> None:
        """
        Record feature values for a sampled request.
        Args:
            features: Dict mapping feature name to value
        """
        for feature_name, value in features.items():
            if feature_name in self._buffers:
                self._buffers[feature_name].add(value)
    def should_run_test(self) -> bool:
        """Check if enough time has elapsed to run drift test."""
        elapsed = time.time() - self._last_test_time
        return elapsed >= self.config.test_interval_seconds
    def run_drift_tests(
        self,
        significance_level: float = 0.01
    ) -> Dict[str, KSTestResult]:
        """
        Run KS tests for all features.
        Returns dict mapping feature name to test result.
        """
        results = {}
        for feature_name, buffer in self._buffers.items():
            samples = buffer.get_samples()
            # Check minimum samples
            if len(samples) < self.config.min_samples_for_test:
                results[feature_name] = KSTestResult(
                    feature_name=feature_name,
                    statistic=0.0,
                    p_value=1.0,
                    is_significant=False,
                    baseline_samples=len(self.baselines[feature_name].sorted_samples),
                    live_samples=len(samples),
                    interpretation=f"Insufficient live samples ({len(samples)} < {self.config.min_samples_for_test})"
                )
                continue
            # Run KS test
            baseline = self.baselines[feature_name]
            result = self._detector.test(
                baseline=baseline,
                live_samples=samples,
                significance_level=significance_level
            )
            results[feature_name] = result
        self._last_test_time = time.time()
        self._last_results = results
        return results
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection status."""
        total_features = len(self.baselines)
        drifted_features = sum(
            1 for r in self._last_results.values()
            if r.is_significant
        )
        return {
            "total_features_monitored": total_features,
            "features_with_drift": drifted_features,
            "drift_percentage": (drifted_features / total_features * 100) if total_features > 0 else 0,
            "last_test_time": self._last_test_time,
            "sample_rate": f"1/{self.config.sample_rate}",
            "requests_seen": self._request_counter,
            "drifted_feature_names": [
                name for name, result in self._last_results.items()
                if result.is_significant
            ]
        }
    def to_prometheus_format(self) -> str:
        """Export drift metrics in Prometheus format."""
        lines = []
        for feature_name, result in self._last_results.items():
            labels = f'{{feature="{feature_name}"}}'
            # KS statistic
            lines.append(
                f"drift_ks_statistic{labels} {result.statistic:.4f}"
            )
            # P-value
            lines.append(
                f"drift_p_value{labels} {result.p_value:.4f}"
            )
            # Drift detected flag
            lines.append(
                f"drift_detected{labels} {1 if result.is_significant else 0}"
            )
            # Sample counts
            lines.append(
                f"drift_baseline_samples{labels} {result.baseline_samples}"
            )
            lines.append(
                f"drift_live_samples{labels} {result.live_samples}"
            )
        # Summary metrics
        summary = self.get_drift_summary()
        lines.append(f"drift_features_monitored {summary['total_features_monitored']}")
        lines.append(f"drift_features_with_drift {summary['features_with_drift']}")
        return "\n".join(lines)
```
### Cross-Domain Connection: Database Query Sampling
This sampling strategy is identical to **database query sampling for statistics**:
| Database Statistics | ML Drift Sampling |
|--------------------|-------------------|
| Sample 1/N queries | Sample 1/N requests |
| Reservoir sampling for representative set | Reservoir sampling for representative set |
| Periodic analysis | Periodic KS tests |
| Minimal query overhead | Minimal inference overhead |
The principle: **sampling is the universal technique for statistical analysis of streams.** You can't analyze everything, but you can analyze a representative subset.
---
## Phase 5: Training Baseline Storage
### The Missing Piece
Drift detection requires comparing live data to **something**. That "something" is the **training baseline**—the feature distributions your model was trained on.

![Training Baseline Storage and Retrieval](./diagrams/diag-m5-baseline-storage.svg)

Without capturing baselines at deployment time, drift detection is impossible. You'd have no reference point.
```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import os
from datetime import datetime
@dataclass
class ModelBaseline:
    """
    Complete baseline for a model version.
    Captured at deployment time, this contains:
    - Feature distributions from training data
    - Prediction distribution from validation
    - Model metadata
    """
    model_version: str
    capture_time: str  # ISO format timestamp
    feature_baselines: Dict[str, FeatureBaseline]
    prediction_baseline: Optional[Dict[str, float]]  # Histogram bins
    metadata: Dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "model_version": self.model_version,
            "capture_time": self.capture_time,
            "feature_baselines": {
                name: {
                    "feature_name": b.feature_name,
                    "samples": b.samples[:1000],  # Limit for storage
                    "min_val": b.min_val,
                    "max_val": b.max_val,
                    "mean": b.mean,
                    "std": b.std,
                }
                for name, b in self.feature_baselines.items()
            },
            "prediction_baseline": self.prediction_baseline,
            "metadata": self.metadata,
        }
    @classmethod
    def from_dict(cls, data: dict) -> "ModelBaseline":
        """Deserialize from dictionary."""
        feature_baselines = {}
        for name, bdata in data.get("feature_baselines", {}).items():
            feature_baselines[name] = FeatureBaseline.from_samples(
                feature_name=bdata["feature_name"],
                samples=bdata.get("samples", []),
            )
        return cls(
            model_version=data["model_version"],
            capture_time=data["capture_time"],
            feature_baselines=feature_baselines,
            prediction_baseline=data.get("prediction_baseline"),
            metadata=data.get("metadata", {}),
        )
    def save(self, path: str) -> None:
        """Save baseline to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    @classmethod
    def load(cls, path: str) -> "ModelBaseline":
        """Load baseline from file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))
class BaselineManager:
    """
    Manages training baselines for model versions.
    Responsibilities:
    1. Capture baselines from training data
    2. Store baselines with model version
    3. Load baselines at deployment time
    4. Track baseline age and staleness
    """
    def __init__(self, storage_dir: str = "baselines"):
        self.storage_dir = storage_dir
        self._loaded_baselines: Dict[str, ModelBaseline] = {}
    def capture_from_training_data(
        self,
        model_version: str,
        training_features: Dict[str, List[float]],
        validation_predictions: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ModelBaseline:
        """
        Capture baseline from training data.
        This should be called during model training or just after,
        before the model is deployed.
        Args:
            model_version: Version identifier for the model
            training_features: Dict mapping feature name to list of values
            validation_predictions: Optional list of predictions on validation set
            metadata: Additional metadata (training date, data hash, etc.)
        """
        # Create feature baselines
        feature_baselines = {}
        for feature_name, samples in training_features.items():
            feature_baselines[feature_name] = FeatureBaseline.from_samples(
                feature_name=feature_name,
                samples=samples,
                max_baseline_samples=10000  # Limit for efficiency
            )
        # Create prediction baseline if provided
        prediction_baseline = None
        if validation_predictions:
            # Compute histogram
            monitor = PredictionDistributionMonitor()
            for pred in validation_predictions[:10000]:
                monitor.record(model_version, pred)
            monitor.capture_baseline(model_version)
            prediction_baseline = monitor._baseline_histograms.get(model_version)
        baseline = ModelBaseline(
            model_version=model_version,
            capture_time=datetime.now().isoformat(),
            feature_baselines=feature_baselines,
            prediction_baseline=prediction_baseline,
            metadata=metadata or {}
        )
        # Save to disk
        path = os.path.join(self.storage_dir, f"{model_version}.json")
        baseline.save(path)
        # Cache in memory
        self._loaded_baselines[model_version] = baseline
        return baseline
    def load_baseline(self, model_version: str) -> Optional[ModelBaseline]:
        """Load baseline for a model version."""
        if model_version in self._loaded_baselines:
            return self._loaded_baselines[model_version]
        path = os.path.join(self.storage_dir, f"{model_version}.json")
        if not os.path.exists(path):
            return None
        baseline = ModelBaseline.load(path)
        self._loaded_baselines[model_version] = baseline
        return baseline
    def get_baseline_age_days(self, model_version: str) -> Optional[float]:
        """Get age of baseline in days."""
        baseline = self.load_baseline(model_version)
        if not baseline:
            return None
        capture_time = datetime.fromisoformat(baseline.capture_time)
        age = datetime.now() - capture_time
        return age.total_seconds() / 86400  # Seconds to days
    def is_baseline_stale(
        self,
        model_version: str,
        max_age_days: float = 90.0
    ) -> bool:
        """Check if baseline is too old."""
        age = self.get_baseline_age_days(model_version)
        if age is None:
            return True  # No baseline = stale
        return age > max_age_days
```
### Cross-Domain Connection: Database Statistics
Baseline storage is identical to **database statistics for query planning**:
| Database Statistics | ML Baselines |
|--------------------|--------------|
| Captured during ANALYZE | Captured during training |
| Stored with table metadata | Stored with model version |
| Used for query optimization | Used for drift detection |
| Can become stale | Can become stale |
| Refreshed periodically | Recaptured on retrain |
The principle: **statistics about data are as valuable as the data itself.** Store them, version them, keep them fresh.
---
## Phase 6: Alert Management and Threshold Tuning
### The Alert Fatigue Problem

![Training Baseline Storage Structure](./diagrams/tdd-diag-040.svg)


Too many alerts = ignored alerts. This is **alert fatigue**, and it's the enemy of effective monitoring.
```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Awaitable
from enum import Enum
import asyncio
import time
import logging
logger = logging.getLogger(__name__)
class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
@dataclass
class AlertRule:
    """Configuration for a drift alert rule."""
    name: str
    feature_name: Optional[str]  # None = aggregate rule
    condition: str  # "p_value < 0.01" or "ks_statistic > 0.1"
    severity: AlertSeverity
    cooldown_seconds: float = 3600.0  # Don't re-alert for this long
    min_samples: int = 100  # Minimum samples before evaluating
@dataclass
class Alert:
    """A fired alert."""
    rule_name: str
    feature_name: Optional[str]
    severity: AlertSeverity
    message: str
    timestamp: float
    value: float
    threshold: float
class AlertManager:
    """
    Manages drift alerts with cooldown and aggregation.
    Key design:
    1. Rules define conditions for alerts
    2. Cooldown prevents alert spam
    3. Severity levels enable routing
    4. Aggregation groups related alerts
    """
    def __init__(
        self,
        rules: List[AlertRule],
        on_alert: Optional[Callable[[Alert], Awaitable[None]]] = None
    ):
        self.rules = {r.name: r for r in rules}
        self.on_alert = on_alert
        # Track last alert time per rule/feature
        self._last_alert_time: Dict[str, float] = {}
        # Recent alerts for aggregation
        self._recent_alerts: List[Alert] = []
        self._max_recent_alerts = 100
    def evaluate(
        self,
        drift_results: Dict[str, KSTestResult],
        feature_samples: Dict[str, int]
    ) -> List[Alert]:
        """
        Evaluate alert rules against drift results.
        Returns list of alerts that fired (and passed cooldown).
        """
        alerts = []
        for rule in self.rules.values():
            # Check per-feature rules
            if rule.feature_name:
                if rule.feature_name not in drift_results:
                    continue
                result = drift_results[rule.feature_name]
                samples = feature_samples.get(rule.feature_name, 0)
                if samples < rule.min_samples:
                    continue
                alert = self._evaluate_rule(rule, result)
                if alert:
                    alerts.append(alert)
            # Check aggregate rules
            else:
                alert = self._evaluate_aggregate_rule(rule, drift_results)
                if alert:
                    alerts.append(alert)
        # Store recent alerts
        self._recent_alerts.extend(alerts)
        if len(self._recent_alerts) > self._max_recent_alerts:
            self._recent_alerts = self._recent_alerts[-self._max_recent_alerts:]
        return alerts
    def _evaluate_rule(
        self,
        rule: AlertRule,
        result: KSTestResult
    ) -> Optional[Alert]:
        """Evaluate a single rule against a single feature result."""
        # Parse condition (simplified - in production, use proper expression parser)
        should_alert = False
        value = 0.0
        threshold = 0.0
        if "p_value <" in rule.condition:
            threshold = float(rule.condition.split("<")[1].strip())
            should_alert = result.p_value < threshold
            value = result.p_value
        elif "ks_statistic >" in rule.condition:
            threshold = float(rule.condition.split(">")[1].strip())
            should_alert = result.statistic > threshold
            value = result.statistic
        if not should_alert:
            return None
        # Check cooldown
        alert_key = f"{rule.name}:{rule.feature_name}"
        last_alert = self._last_alert_time.get(alert_key, 0)
        if time.time() - last_alert < rule.cooldown_seconds:
            return None
        # Fire alert
        self._last_alert_time[alert_key] = time.time()
        return Alert(
            rule_name=rule.name,
            feature_name=rule.feature_name,
            severity=rule.severity,
            message=f"Drift detected for {rule.feature_name}: {result.interpretation}",
            timestamp=time.time(),
            value=value,
            threshold=threshold
        )
    def _evaluate_aggregate_rule(
        self,
        rule: AlertRule,
        drift_results: Dict[str, KSTestResult]
    ) -> Optional[Alert]:
        """Evaluate an aggregate rule across all features."""
        # Count features with drift
        drifted_count = sum(1 for r in drift_results.values() if r.is_significant)
        total_count = len(drift_results)
        if total_count == 0:
            return None
        drift_fraction = drifted_count / total_count
        # Parse threshold from condition
        if "drift_fraction >" in rule.condition:
            threshold = float(rule.condition.split(">")[1].strip())
            should_alert = drift_fraction > threshold
        else:
            return None
        if not should_alert:
            return None
        # Check cooldown
        alert_key = f"{rule.name}:aggregate"
        last_alert = self._last_alert_time.get(alert_key, 0)
        if time.time() - last_alert < rule.cooldown_seconds:
            return None
        # Fire alert
        self._last_alert_time[alert_key] = time.time()
        return Alert(
            rule_name=rule.name,
            feature_name=None,
            severity=rule.severity,
            message=f"Aggregate drift: {drifted_count}/{total_count} features drifted ({drift_fraction:.1%})",
            timestamp=time.time(),
            value=drift_fraction,
            threshold=threshold
        )
    def get_recent_alerts(self, limit: int = 20) -> List[Alert]:
        """Get recent alerts."""
        return self._recent_alerts[-limit:]
# Default alert rules
DEFAULT_DRIFT_RULES = [
    AlertRule(
        name="p_value_critical",
        feature_name=None,  # Apply to all features
        condition="p_value < 0.001",
        severity=AlertSeverity.CRITICAL,
        cooldown_seconds=1800.0,  # 30 minutes
        min_samples=200
    ),
    AlertRule(
        name="p_value_warning",
        feature_name=None,
        condition="p_value < 0.01",
        severity=AlertSeverity.WARNING,
        cooldown_seconds=3600.0,  # 1 hour
        min_samples=100
    ),
    AlertRule(
        name="aggregate_drift",
        feature_name=None,
        condition="drift_fraction > 0.2",  # >20% of features drifted
        severity=AlertSeverity.WARNING,
        cooldown_seconds=3600.0
    ),
]
```
### Cross-Domain Connection: PagerDuty Alert Grouping
This alert management is identical to **PagerDuty's alert grouping and deduplication**:
| PagerDuty | Alert Manager |
|-----------|--------------|
| Deduplication keys | Cooldown tracking |
| Alert grouping | Aggregate rules |
| Severity levels | INFO/WARNING/CRITICAL |
| Routing policies | on_alert callback |
The principle: **humans can only process a limited number of alerts.** Intelligent grouping and cooldowns are essential.
---
## Phase 7: Structured Request Logging
### The Debugging Imperative
When drift is detected, you need to investigate. **What inputs caused the drift?** What predictions were made? Structured logging provides the audit trail.

![Structured Request Logging with Sampling](./diagrams/diag-m5-structured-logging.svg)

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import time
import hashlib
@dataclass
class StructuredLogEntry:
    """
    A structured log entry for request auditing.
    Captures complete context for debugging:
    - Input features
    - Prediction output
    - Model version
    - Timing
    - Sampling metadata
    """
    timestamp: float
    request_id: str
    model_version: str
    user_id: Optional[str]
    # Features (sampled)
    features: Dict[str, float]
    features_hash: str
    # Prediction
    prediction: Any
    prediction_confidence: Optional[float]
    # Performance
    latency_ms: float
    success: bool
    error_message: Optional[str] = None
    # Context
    was_sampled_for_drift: bool = False
    experiment_id: Optional[str] = None
    def to_json_line(self) -> str:
        """Convert to JSON line for logging."""
        return json.dumps({
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "model_version": self.model_version,
            "user_id": self.user_id,
            "features": self.features,
            "features_hash": self.features_hash,
            "prediction": self.prediction,
            "prediction_confidence": self.prediction_confidence,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "error_message": self.error_message,
            "was_sampled_for_drift": self.was_sampled_for_drift,
            "experiment_id": self.experiment_id,
        })
    @staticmethod
    def hash_features(features: Dict[str, float]) -> str:
        """Create deterministic hash of features."""
        # Sort keys for determinism
        feature_str = json.dumps(features, sort_keys=True)
        return hashlib.sha256(feature_str.encode()).hexdigest()[:16]
class StructuredRequestLogger:
    """
    Logs requests with sampling for debugging and audit.
    Key design:
    1. Sample a percentage of requests for full logging
    2. Always log requests that were sampled for drift detection
    3. Structured format enables querying and analysis
    4. Separate from metrics (this is for debugging)
    """
    def __init__(
        self,
        log_dir: str = "logs/requests",
        sample_rate: float = 0.01,  # 1% of requests
        flush_interval: int = 100
    ):
        self.log_dir = log_dir
        self.sample_rate = sample_rate
        self.flush_interval = flush_interval
        self._buffer: List[StructuredLogEntry] = []
        self._request_counter = 0
    def should_log(self) -> bool:
        """Determine if this request should be logged."""
        self._request_counter += 1
        # Log 1 in N requests
        return self._request_counter % int(1 / self.sample_rate) == 0
    def log(
        self,
        request_id: str,
        model_version: str,
        features: Dict[str, float],
        prediction: Any,
        latency_ms: float,
        success: bool = True,
        error_message: Optional[str] = None,
        user_id: Optional[str] = None,
        prediction_confidence: Optional[float] = None,
        was_sampled_for_drift: bool = False,
        experiment_id: Optional[str] = None,
        force_log: bool = False
    ) -> None:
        """
        Log a request entry.
        Args:
            force_log: If True, log regardless of sample rate
                      (used for drift-sampled requests)
        """
        if not force_log and not self.should_log():
            return
        entry = StructuredLogEntry(
            timestamp=time.time(),
            request_id=request_id,
            model_version=model_version,
            user_id=user_id,
            features=features,
            features_hash=StructuredLogEntry.hash_features(features),
            prediction=prediction,
            prediction_confidence=prediction_confidence,
            latency_ms=latency_ms,
            success=success,
            error_message=error_message,
            was_sampled_for_drift=was_sampled_for_drift,
            experiment_id=experiment_id
        )
        self._buffer.append(entry)
        if len(self._buffer) >= self.flush_interval:
            self._flush()
    def _flush(self) -> None:
        """Write buffered entries to disk."""
        if not self._buffer:
            return
        import os
        os.makedirs(self.log_dir, exist_ok=True)
        # Use date-based log files
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_path = os.path.join(self.log_dir, f"requests-{date_str}.jsonl")
        with open(log_path, "a") as f:
            for entry in self._buffer:
                f.write(entry.to_json_line() + "\n")
        self._buffer = []
    def finalize(self) -> None:
        """Flush remaining entries."""
        self._flush()
```
---
## Phase 8: The Complete Monitoring Server

![Monitoring Stack Architecture](./diagrams/diag-m5-monitoring-stack.svg)

```python
"""
monitoring_server.py
Complete monitoring and drift detection server.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
import uuid
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, PlainTextResponse
import torch
import orjson
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Configuration
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")
DRIFT_SAMPLE_RATE = int(os.getenv("DRIFT_SAMPLE_RATE", "1000"))
LOG_SAMPLE_RATE = float(os.getenv("LOG_SAMPLE_RATE", "0.01"))
BASELINE_DIR = os.getenv("BASELINE_DIR", "baselines")
# Global components
latency_tracker: Optional[PerVersionLatencyTracker] = None
prediction_monitor: Optional[PredictionDistributionMonitor] = None
drift_sampler: Optional[DriftDetectionSampler] = None
alert_manager: Optional[AlertManager] = None
request_logger: Optional[StructuredRequestLogger] = None
baseline_manager: Optional[BaselineManager] = None
# Model (loaded elsewhere)
model = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global latency_tracker, prediction_monitor, drift_sampler
    global alert_manager, request_logger, baseline_manager
    logger.info("[Startup] Initializing monitoring components...")
    # Initialize latency tracker
    latency_tracker = PerVersionLatencyTracker()
    # Initialize prediction monitor
    prediction_monitor = PredictionDistributionMonitor(window_size=10000)
    # Load baseline
    baseline_manager = BaselineManager(storage_dir=BASELINE_DIR)
    baseline = baseline_manager.load_baseline(MODEL_VERSION)
    if baseline:
        logger.info(f"[Startup] Loaded baseline for {MODEL_VERSION}")
        drift_sampler = DriftDetectionSampler(
            baselines=baseline.feature_baselines,
            config=SamplingConfig(
                sample_rate=DRIFT_SAMPLE_RATE,
                min_samples_for_test=100,
                max_live_samples=5000,
                test_interval_seconds=300.0
            )
        )
    else:
        logger.warning(
            f"[Startup] No baseline found for {MODEL_VERSION}. "
            "Drift detection disabled."
        )
    # Initialize alert manager
    alert_manager = AlertManager(
        rules=DEFAULT_DRIFT_RULES,
        on_alert=handle_alert
    )
    # Initialize request logger
    request_logger = StructuredRequestLogger(
        log_dir="logs/requests",
        sample_rate=LOG_SAMPLE_RATE
    )
    logger.info("[Startup] Monitoring ready")
    yield  # Server runs here
    # Shutdown
    logger.info("[Shutdown] Finalizing logs...")
    request_logger.finalize()
    logger.info("[Shutdown] Complete")
async def handle_alert(alert: Alert) -> None:
    """Handle a fired alert."""
    logger.warning(
        f"[ALERT] {alert.severity.value.upper()}: {alert.rule_name} - "
        f"{alert.message}"
    )
    # In production, send to PagerDuty, Slack, etc.
app = FastAPI(lifespan=lifespan, title="Monitored Model Server")
@app.get("/health/liveness")
async def liveness():
    return {"status": "alive"}
@app.get("/health/readiness")
async def readiness():
    return {"status": "ready"}
@app.post("/predict")
async def predict(request: Request):
    """
    Inference endpoint with full monitoring.
    """
    request_id = str(uuid.uuid4())[:8]
    request_start = time.perf_counter()
    # Parse request
    body = await request.body()
    try:
        data = orjson.loads(body)
    except orjson.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    features = data.get("features", {})
    input_data = data.get("input")
    user_id = data.get("user_id")
    # Check if this request should be sampled for drift detection
    should_sample_drift = drift_sampler and drift_sampler.should_sample()
    # Record features for drift detection
    if should_sample_drift and drift_sampler:
        drift_sampler.record_features(features)
    # Run inference
    try:
        # ... model inference here ...
        # For demo, just echo
        prediction = {"result": input_data}
        confidence = 0.95
        success = True
        error_message = None
    except Exception as e:
        prediction = None
        confidence = None
        success = False
        error_message = str(e)
        raise HTTPException(status_code=500, detail=str(e))
    latency_ms = (time.perf_counter() - request_start) * 1000
    # Record latency
    latency_tracker.record(MODEL_VERSION, latency_ms, success)
    # Record prediction for distribution monitoring
    if prediction and confidence:
        prediction_monitor.record(MODEL_VERSION, confidence)
    # Log request
    request_logger.log(
        request_id=request_id,
        model_version=MODEL_VERSION,
        features=features,
        prediction=prediction,
        latency_ms=latency_ms,
        success=success,
        error_message=error_message,
        user_id=user_id,
        prediction_confidence=confidence,
        was_sampled_for_drift=should_sample_drift,
        force_log=should_sample_drift  # Always log drift-sampled requests
    )
    return Response(
        content=orjson.dumps({
            "prediction": prediction,
            "request_id": request_id,
            "latency_ms": latency_ms,
        }),
        media_type="application/json"
    )
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    lines = []
    # Latency metrics
    lines.append(latency_tracker.to_prometheus_format())
    # Drift metrics
    if drift_sampler and drift_sampler.should_run_test():
        results = drift_sampler.run_drift_tests()
        # Check alerts
        feature_samples = {
            name: len(buffer.samples)
            for name, buffer in drift_sampler._buffers.items()
        }
        alerts = alert_manager.evaluate(results, feature_samples)
        for alert in alerts:
            await handle_alert(alert)
    if drift_sampler:
        lines.append(drift_sampler.to_prometheus_format())
    return PlainTextResponse(
        content="\n".join(lines),
        media_type="text/plain"
    )
@app.get("/drift/status")
async def drift_status():
    """Get drift detection status."""
    if not drift_sampler:
        return {"status": "disabled", "reason": "No baseline loaded"}
    summary = drift_sampler.get_drift_summary()
    summary["baseline_age_days"] = baseline_manager.get_baseline_age_days(MODEL_VERSION)
    summary["baseline_stale"] = baseline_manager.is_baseline_stale(MODEL_VERSION)
    return summary
@app.get("/drift/results")
async def drift_results():
    """Get latest drift test results."""
    if not drift_sampler:
        return {"status": "disabled"}
    return {
        feature: {
            "statistic": r.statistic,
            "p_value": r.p_value,
            "is_significant": r.is_significant,
            "interpretation": r.interpretation
        }
        for feature, r in drift_sampler._last_results.items()
    }
@app.get("/latency/percentiles")
async def latency_percentiles(version: str = MODEL_VERSION):
    """Get latency percentiles for a version."""
    return latency_tracker.get_percentiles(version)
@app.get("/alerts/recent")
async def recent_alerts(limit: int = 20):
    """Get recent alerts."""
    alerts = alert_manager.get_recent_alerts(limit)
    return [
        {
            "rule_name": a.rule_name,
            "feature_name": a.feature_name,
            "severity": a.severity.value,
            "message": a.message,
            "timestamp": a.timestamp,
            "value": a.value,
            "threshold": a.threshold
        }
        for a in alerts
    ]
@app.post("/baseline/capture")
async def capture_baseline(request: Request):
    """
    Manually capture a new baseline.
    This endpoint allows capturing baselines from live data
    when a training baseline isn't available.
    """
    body = await request.body()
    data = orjson.loads(body)
    # This would capture from accumulated live samples
    # In production, you'd want safeguards here
    return {"status": "captured", "version": MODEL_VERSION}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
---
## Knowledge Cascade: What You've Unlocked
### Immediate Applications
**Production Deployment**: The monitoring system you've built is production-ready. Deploy it alongside your model server and wire the `/metrics` endpoint to Prometheus. Create Grafana dashboards from the exposed metrics.
**Model Lifecycle Management**: The baseline storage system integrates with your training pipeline. Capture baselines automatically when models are trained, and load them when models are deployed.
### Cross-Domain Connections
**1. KS Test → Clinical Trial Statistics**
The KS test you learned is the same test used in **FDA clinical trials** to compare treatment and control distributions:
| Clinical Trials | Drift Detection |
|-----------------|-----------------|
| Treatment vs Control | Live vs Baseline |
| Efficacy endpoints | Feature distributions |
| KS test for distribution comparison | KS test for drift detection |
| p < 0.05 = significant | p < 0.01 = drifted |
The math is identical. Only the domain changes.
**2. Histogram Percentiles → Database Query Planning**
The histogram-based percentile calculation is the same technique used by **database query planners** to estimate query costs:
| Query Planning | Latency Monitoring |
|----------------|-------------------|
| Column value histograms | Latency histograms |
| Estimate rows matching predicate | Estimate requests under SLA |
| Bucket-based approximation | Bucket-based approximation |
Both systems trade precision for efficiency: approximate answers fast.
**3. Sampling → Census Bureau Statistics**
The reservoir sampling technique is identical to **survey sampling methodology**:
| Census Surveys | Drift Sampling |
|----------------|----------------|
| Sample households | Sample requests |
| Representative sample needed | Representative sample needed |
| Reservoir sampling for streams | Reservoir sampling for streams |
| Statistical validity | Statistical validity |
The principle: **you don't need to measure everything to understand everything.** A well-designed sample is almost as good as the full population.
**4. Alert Management → Incident Response**
The cooldown and aggregation in alert management mirrors **IT incident response best practices**:
| Incident Response | Alert Management |
|-------------------|------------------|
| Deduplicate related incidents | Cooldown prevents duplicates |
| Severity levels (P1/P2/P3) | INFO/WARNING/CRITICAL |
| On-call rotation | on_alert callback |
| Post-incident review | Drift investigation |
Both systems recognize that **humans are the bottleneck in alert processing.** Design for human capacity.
### Forward Connections
With monitoring and drift detection mastered, you can now:
- **Build automated retraining triggers** that fire when drift exceeds thresholds
- **Implement A/B test guardrails** that stop experiments if treatment causes drift
- **Create feature importance monitoring** that tracks which features drift most
- **Design multi-model drift dashboards** that compare drift across model versions
- **Implement feedback loops** that capture labels and measure true model accuracy
---
## Summary: The Observability Mindset
Production ML monitoring is not "add Prometheus and hope." It's a **disciplined practice**:
1. **Latency histograms, not averages** — distributions reveal the truth
2. **Prediction monitoring, not just input monitoring** — models can degrade even with stable inputs
3. **Statistical rigor, not intuition** — KS tests and p-values separate signal from noise
4. **Sampling for efficiency** — you can't analyze everything, but you can analyze enough
5. **Baselines are the foundation** — without them, drift detection is impossible
6. **Alert fatigue is the enemy** — tune thresholds to your data, not generic defaults
7. **Structured logging enables investigation** — when drift happens, you need to debug
You now have the tools to **see the invisible** — to detect silent model degradation before it hurts users. This is the difference between a model server that works and a model server that's production-ready.
---
[[CRITERIA_JSON: {"milestone_id": "ml-model-serving-m5", "criteria": ["Inference latency percentiles (p50, p90, p95, p99) tracked per model version using Prometheus-style cumulative histograms with configurable bucket boundaries aligned to SLAs", "Prediction output distribution monitored using sliding window histogram with n_bins (default 20), compared against stored baseline via total variation distance", "Data drift detection computes KS test between sampled live input feature distributions and stored training baseline distributions, returning D statistic and p-value", "Drift detection runs on configurable sample rate (default: 1/1000 requests) using reservoir sampling to maintain representative live sample buffer", "Alert fires when KS test p-value drops below configurable threshold (default 0.01) for any monitored feature, with cooldown period to prevent alert fatigue", "Structured request logging samples configurable percentage of requests (default 1%) with full input features, prediction output, latency, and drift sampling metadata in JSON lines format", "Training baseline stored at deployment time includes per-feature samples (capped at 10,000), min/max/mean/std statistics, and prediction distribution histogram", "LatencyHistogram provides percentile() method with linear interpolation within buckets for approximate p50/p90/p95/p99 calculation", "PerVersionLatencyTracker maintains separate histograms per model version with to_prometheus_format() export including _bucket, _sum, and _count metrics", "PredictionDistributionMonitor captures baseline via capture_baseline() and compares current distribution via compare_to_baseline() returning mean_shift, std_ratio, and total_variation_distance", "DriftDetectionSampler uses should_sample() for deterministic sampling (every Nth request) and should_run_test() for interval-based drift test execution", "AlertManager evaluates alert rules with cooldown tracking per rule/feature combination, supporting per-feature and aggregate rules with INFO/WARNING/CRITICAL severity levels", "BaselineManager loads/stores baselines as JSON files with model version, capture time, feature baselines, and metadata; tracks baseline age for staleness detection"]}]
<!-- END_MS -->


# TDD

![ML Model Serving: Interactive Atlas](./diagrams/diag-satellite-atlas/index.svg)

![m1](./diagrams/diag-satellite-atlas/m1.svg)

![m2](./diagrams/diag-satellite-atlas/m2.svg)

![m3](./diagrams/diag-satellite-atlas/m3.svg)

![m4](./diagrams/diag-satellite-atlas/m4.svg)

![m5](./diagrams/diag-satellite-atlas/m5.svg)

A production-grade model inference service that transforms trained ML models into reliable, scalable, and observable APIs. The system implements dynamic request batching for GPU throughput optimization, multi-version model deployment with zero-downtime hot swapping, statistically rigorous A/B testing infrastructure, and real-time drift detection to catch model degradation before it impacts users. The architecture balances the fundamental tension between inference latency and resource constraints—maximizing GPU utilization while bounding worst-case latency through configurable batching parameters.


<!-- TDD_MOD_ID: ml-model-serving-m1 -->
# Model Loading, Inference & Serialization
## Module Charter
This module implements the complete model loading lifecycle for production ML inference, transforming a trained PyTorch model file into an inference-ready HTTP endpoint. It handles device detection with automatic CPU/GPU fallback, memory budget validation to prevent OOM crashes, proper eval mode and gradient disabling for optimal inference performance, CUDA kernel warmup for consistent latency, and optimized JSON serialization using orjson for 5-10x speedup over stdlib json. The module provides a Kubernetes-compatible health probe system distinguishing liveness (process alive) from readiness (model loaded and warmed). It explicitly does NOT handle request batching (M2), model versioning (M3), A/B testing (M4), or drift detection (M5). Upstream dependencies are PyTorch, ONNX Runtime (optional), and FastAPI; downstream consumers are the batching layer and HTTP routing layer. The key invariant is that the `/health/readiness` endpoint MUST return 200 only after model loading AND warmup complete—never before.
---
## File Structure
Create files in this exact sequence:
```
ml-model-serving/
├── 01_config.py              # Configuration constants and environment parsing
├── 02_exceptions.py          # Custom exception hierarchy
├── 03_device.py              # Device detection and memory budgeting
├── 04_model_loader.py        # Model loading with eval mode and no_grad
├── 05_warmup.py              # CUDA warmup with dummy inference loop
├── 06_schema.py              # Input schema definitions and validation
├── 07_serialization.py       # orjson serialization with benchmarking
├── 08_health.py              # Liveness and readiness probes
├── 09_model_container.py     # LoadedModel dataclass container
├── 10_lifespan.py            # FastAPI lifespan handler integration
├── 11_endpoints.py           # Prediction and health HTTP endpoints
└── 12_main.py                # Application entry point
```
---
## Complete Data Model
### ModelState Enum
The lifecycle states of the model loading process:
```python
# 02_exceptions.py
from enum import Enum
class ModelState(Enum):
    LOADING = "loading"        # Model file is being loaded into memory
    WARMING_UP = "warming_up"  # Running dummy inferences for CUDA kernel cache
    READY = "ready"           # Model is loaded, warmed, and serving traffic
    FAILED = "failed"         # Loading or warmup failed; requires intervention
```
**Why this enum exists:** Kubernetes readiness probes need to distinguish between "still loading" (503, retry later) and "failed" (503, alert humans). The WARMING_UP state ensures no request hits uncached CUDA kernels.
---
### InputSchema Dataclass
Defines the expected shape and dtype for a model input:
```python
# 06_schema.py
from dataclasses import dataclass
from typing import Tuple, Union, Optional
import torch
@dataclass(frozen=True)
class InputSchema:
    """
    Defines the expected input shape and dtype for a model.
    Fields:
        name: Input tensor name (e.g., "image", "input_ids")
        shape: Expected shape with None for variable dimensions
               e.g., (None, 3, 224, 224) for batch-variable image input
        dtype: Expected torch.dtype (e.g., torch.float32, torch.long)
        description: Human-readable description for error messages
    Invariants:
        - shape rank must match actual tensor rank
        - None in shape means "any size accepted" for that dimension
        - dtype must be a valid torch.dtype
    """
    name: str
    shape: Tuple[Union[int, None], ...]
    dtype: torch.dtype
    description: str
    def validate(self, tensor: torch.Tensor) -> Tuple[bool, Optional[str]]:
        """
        Validate a tensor against this schema.
        Args:
            tensor: The tensor to validate
        Returns:
            (is_valid, error_message):
            - (True, None) if valid
            - (False, "descriptive error") if invalid
        Validation steps:
            1. Check dtype matches exactly
            2. Check tensor rank matches shape length
            3. Check each non-None dimension matches expected size
        """
        # Check dtype
        if tensor.dtype != self.dtype:
            return False, (
                f"Wrong dtype for '{self.name}': expected {self.dtype}, "
                f"got {tensor.dtype}"
            )
        # Check rank
        if len(tensor.shape) != len(self.shape):
            return False, (
                f"Wrong rank for '{self.name}': expected {len(self.shape)}D, "
                f"got {len(tensor.shape)}D with shape {tuple(tensor.shape)}"
            )
        # Check each dimension
        for i, (actual, expected) in enumerate(zip(tensor.shape, self.shape)):
            if expected is not None and actual != expected:
                return False, (
                    f"Wrong shape for '{self.name}': dimension {i} expected "
                    f"{expected}, got {actual}. Full shape: {tuple(tensor.shape)}"
                )
        return True, None
```
---
### LoadedModel Container
The primary output of the model loading pipeline:
```python
# 09_model_container.py
from dataclasses import dataclass
from typing import Tuple
import torch
@dataclass
class LoadedModel:
    """
    Container for a loaded, warmed-up model ready for inference.
    This is the primary artifact produced by the model loading pipeline.
    All fields are set during loading and remain immutable afterward.
    Fields:
        model: The PyTorch model in eval mode, on the correct device
        device: The torch.device where the model resides
        version: Semantic version string for tracking (e.g., "v1.2.3")
        input_shape: Expected input shape for warmup was run with
        warmup_latency_ms: Stable latency after warmup (last run)
    Invariants:
        - model is always in eval() mode
        - model is on the specified device
        - warmup has been run at least once
        - predict() always uses torch.no_grad()
    """
    model: torch.nn.Module
    device: torch.device
    version: str
    input_shape: Tuple[int, ...]
    warmup_latency_ms: float
    @torch.no_grad()
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Run inference on inputs with automatic device placement.
        The @torch.no_grad() decorator ensures:
        - No gradient computation (saves ~30-50% latency)
        - No computation graph built (saves memory)
        - Tensors are not tracked for backward pass
        Args:
            inputs: Input tensor (any device; will be moved to model device)
        Returns:
            Output tensor on CPU (for safe pickling/serialization)
        Note:
            Input is moved to model device automatically.
            Output is moved to CPU before return.
        """
        inputs = inputs.to(self.device)
        output = self.model(inputs)
        return output.cpu()
```
---
### GlobalState Dataclass
Shared state for health probes and model access:
```python
# 08_health.py
from dataclasses import dataclass, field
from typing import Optional
import torch
from enum import Enum
class ModelState(Enum):
    LOADING = "loading"
    WARMING_UP = "warming_up"
    READY = "ready"
    FAILED = "failed"
@dataclass
class GlobalState:
    """
    Shared state for the model serving application.
    This is the single source of truth for:
    - Model readiness (checked by health probes)
    - Error information (for debugging failed loads)
    - Model version (for response metadata)
    Thread safety: All reads/writes happen in async context,
    so no explicit locking needed for FastAPI's cooperative multitasking.
    Invariants:
        - If model_state == READY, model is not None
        - If model_state == FAILED, load_error is not None
        - ready_since is set only when model_state transitions to READY
    """
    model_state: ModelState = ModelState.LOADING
    model: Optional[torch.nn.Module] = None
    device: Optional[torch.device] = None
    version: str = "v1.0.0"
    load_error: Optional[str] = None
    ready_since: Optional[float] = None
```
---
### Exception Hierarchy
```python
# 02_exceptions.py
class ModelServingError(Exception):
    """Base exception for all model serving errors."""
    pass
class MemoryBudgetExceeded(ModelServingError):
    """
    Raised when insufficient GPU memory is available.
    Attributes:
        required_bytes: Estimated memory needed
        available_bytes: Memory actually available
        safety_margin: Configured safety margin fraction
    """
    def __init__(self, required_bytes: int, available_bytes: int, safety_margin: float):
        self.required_bytes = required_bytes
        self.available_bytes = available_bytes
        self.safety_margin = safety_margin
        super().__init__(
            f"Insufficient GPU memory. Required: {required_bytes/1e9:.2f} GB, "
            f"Available: {available_bytes/1e9:.2f} GB, "
            f"Safety margin: {safety_margin*100:.0f}%"
        )
class ModelNotFound(ModelServingError):
    """Raised when model file does not exist at specified path."""
    def __init__(self, path: str):
        self.path = path
        super().__init__(f"Model file not found: {path}")
class InvalidInputShape(ModelServingError):
    """Raised when request tensor shape doesn't match model input schema."""
    def __init__(self, expected: tuple, actual: tuple, input_name: str):
        self.expected = expected
        self.actual = actual
        self.input_name = input_name
        super().__init__(
            f"Invalid shape for '{input_name}': expected {expected}, got {actual}"
        )
class InvalidInputDtype(ModelServingError):
    """Raised when request tensor dtype doesn't match expected dtype."""
    def __init__(self, expected: str, actual: str, input_name: str):
        self.expected = expected
        self.actual = actual
        self.input_name = input_name
        super().__init__(
            f"Invalid dtype for '{input_name}': expected {expected}, got {actual}"
        )
class ModelLoadFailed(ModelServingError):
    """Raised when PyTorch/ONNX loading throws an exception."""
    def __init__(self, path: str, original_error: Exception):
        self.path = path
        self.original_error = original_error
        super().__init__(f"Failed to load model from {path}: {original_error}")
class SerializationError(ModelServingError):
    """Raised when JSON parsing or encoding fails."""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        self.original_error = original_error
        super().__init__(message)
```
---
## Interface Contracts
### Device Detection
```python
# 03_device.py
import torch
from typing import Optional
def get_device() -> torch.device:
    """
    Select the best available device with explicit fallback.
    Returns:
        torch.device: CUDA device if available, CPU otherwise
    Side effects:
        Logs the device selection with GPU name and memory if CUDA
    Decision logic:
        1. Check torch.cuda.is_available()
        2. If True: return torch.device("cuda")
        3. If False: log warning, return torch.device("cpu")
    Thread safety: Safe (no mutation of shared state)
    """
    pass
def check_memory_budget(
    model_path: str, 
    device: torch.device, 
    safety_margin: float = 0.3
) -> None:
    """
    Verify sufficient memory exists before loading the model.
    Args:
        model_path: Path to the model file (for size estimation)
        device: Target device for loading
        safety_margin: Fraction of memory to reserve (default 30%)
    Raises:
        MemoryBudgetExceeded: If insufficient memory available
        FileNotFoundError: If model file doesn't exist (re-raised as ModelNotFound)
    Memory estimation formula:
        required = model_size * 4  # Conservative: params + activations
        available_after_load = free_memory - required
        min_required = total_memory * safety_margin
        Fails if: available_after_load < min_required
    The 4x multiplier accounts for:
        - Model parameters (1x)
        - Activation tensors during inference (2-3x)
        - CUDA context overhead (~500MB implicit)
    Thread safety: Safe (reads CUDA stats, no mutation)
    """
    pass
```
### Model Loader
```python
# 04_model_loader.py
import torch
from typing import Union
def load_model(
    model_path: str, 
    device: torch.device,
    expected_input_shape: Optional[tuple] = None
) -> torch.nn.Module:
    """
    Load a PyTorch model with production-ready configuration.
    Args:
        model_path: Path to saved model (state_dict or full model)
        device: Target device for inference
        expected_input_shape: Optional shape for validation (not enforced here)
    Returns:
        torch.nn.Module: Model in eval mode, on correct device
    Raises:
        ModelNotFound: If model file doesn't exist
        ModelLoadFailed: If torch.load() throws any exception
    Post-conditions:
        - Returned model is in eval() mode (affects Dropout, BatchNorm)
        - Returned model is on the specified device
        - Model is ready for inference (but NOT warmed up)
    Critical operations:
        1. torch.load(model_path, map_location=device)
        2. model.eval()  # MUST call this
        3. model.to(device)  # Idempotent if already on device
    Thread safety: NOT thread-safe during loading (CUDA context mutation)
    """
    pass
@torch.no_grad()
def predict(
    model: torch.nn.Module, 
    inputs: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Run inference with gradients disabled.
    Args:
        model: Model in eval mode
        inputs: Input tensor (any device)
        device: Device to run inference on
    Returns:
        torch.Tensor: Output tensor on CPU
    The @torch.no_grad() decorator ensures:
        - No autograd graph construction
        - No gradient tensor allocation
        - ~30-50% latency improvement over gradient-enabled inference
    Thread safety: Safe for concurrent calls (PyTorch inference is thread-safe)
    """
    pass
```
### Warmup Module
```python
# 05_warmup.py
import torch
from typing import List
def warmup_model(
    model: torch.nn.Module,
    device: torch.device,
    input_shape: tuple,
    n_warmup_runs: int = 5
) -> List[float]:
    """
    Run dummy inferences to populate CUDA kernel cache.
    Args:
        model: The loaded model (already in eval mode)
        device: Device the model is on
        input_shape: Shape of expected inputs, e.g., (1, 3, 224, 224)
        n_warmup_runs: Number of warmup iterations (default 5)
    Returns:
        List[float]: Latency in milliseconds for each warmup run
    Side effects:
        - Populates CUDA kernel cache for input_shape
        - Logs each run's latency
        - Logs warning if latency variance > 10% of final latency
    Why multiple runs:
        - First run: JIT compiles kernels (slowest)
        - Runs 2-3: Populates remaining caches
        - Runs 4+: Should be stable (verifies warmup completed)
    CUDA synchronization:
        - torch.cuda.synchronize() before/after each inference
        - Ensures accurate timing (GPU is asynchronous by default)
    Thread safety: NOT thread-safe (modifies CUDA state)
    Invariant: Last latency in returned list is the "stable" latency
    """
    pass
def measure_warmup_effect(
    model: torch.nn.Module,
    device: torch.device,
    input_shape: tuple
) -> dict:
    """
    Measure the warmup penalty (first vs subsequent inference).
    Args:
        model: The model to test
        device: Device to run on
        input_shape: Input shape for dummy tensor
    Returns:
        dict with keys:
            - first_inference_ms: Cold inference latency
            - second_inference_ms: Warm inference latency
            - warmup_penalty_ms: Difference (first - second)
            - warmup_penalty_percent: (penalty / second) * 100
    Use case: Diagnosing slow first requests, validating warmup config
    """
    pass
```
### Input Validation
```python
# 06_schema.py
from typing import List, Dict, Tuple, Union, Optional
import torch
class ModelInputValidator:
    """
    Validates input tensors against model's expected schema.
    Attributes:
        schemas: Dict mapping input name to InputSchema
    """
    def __init__(self, schemas: List[InputSchema]):
        """
        Initialize validator with expected schemas.
        Args:
            schemas: List of InputSchema objects defining expected inputs
        """
        self.schemas = {s.name: s for s in schemas}
    def validate_request(
        self, 
        request_data: dict
    ) -> Tuple[bool, Union[Dict[str, torch.Tensor], str]]:
        """
        Validate and parse incoming JSON request.
        Args:
            request_data: Parsed JSON dict from request body
        Returns:
            (success, result):
                - If success: (True, {"inputs": {name: tensor, ...}})
                - If failure: (False, "error message")
        Validation steps for each schema:
            1. Check required input exists in request_data
            2. Convert to tensor with correct dtype
            3. Validate shape against schema
            4. Store in inputs dict
        Error messages are descriptive for debugging:
            - "Missing required input: 'image'"
            - "Wrong dtype for 'input_ids': expected torch.long, got torch.float32"
            - "Wrong rank for 'image': expected 4D, got 3D with shape (3, 224, 224)"
        Thread safety: Safe (creates new tensors, no shared mutation)
        """
        pass
```
### Serialization Module
```python
# 07_serialization.py
import orjson
from typing import Any, Union
import numpy as np
import torch
class SerializationError(Exception):
    """Raised when serialization/deserialization fails."""
    pass
def serialize_response(data: dict) -> bytes:
    """
    Serialize response dict to JSON bytes using orjson.
    Args:
        data: Dict with JSON-serializable values (including numpy arrays)
    Returns:
        JSON bytes (ready for HTTP response body)
    Raises:
        SerializationError: If data contains non-serializable types
    orjson options used:
        - OPT_SERIALIZE_NUMPY: Native numpy array serialization
        - OPT_UTC_Z: Use 'Z' suffix for UTC timestamps
    Note: orjson.dumps() returns bytes, not str (unlike stdlib json)
    """
    pass
def deserialize_request(body: bytes) -> dict:
    """
    Deserialize JSON request body to dict.
    Args:
        body: Raw bytes from HTTP request
    Returns:
        Parsed dict
    Raises:
        SerializationError: If JSON is invalid (wraps orjson.JSONDecodeError)
    """
    pass
def tensor_to_json(tensor: torch.Tensor) -> list:
    """
    Convert PyTorch tensor to JSON-serializable format.
    Args:
        tensor: PyTorch tensor (any shape, any device)
    Returns:
        Nested lists (JSON-compatible)
    Conversion path:
        tensor -> detach() -> cpu() -> numpy() -> tolist()
    The detach() is critical: removes tensor from computation graph
    """
    pass
def benchmark_serialization(n_runs: int = 100) -> dict:
    """
    Benchmark stdlib json vs orjson for typical inference payloads.
    Args:
        n_runs: Number of iterations for timing
    Returns:
        dict with keys:
            - stdlib_ms: Average serialization time with stdlib json
            - orjson_ms: Average serialization time with orjson
            - speedup: Ratio (stdlib_ms / orjson_ms)
            - payload_size_kb: Size of serialized output
    Use case: Run at startup to validate orjson speedup
    Expected result: speedup >= 3.0x for numpy array payloads
    """
    pass
```
### Health Probes
```python
# 08_health.py
from fastapi import HTTPException
async def liveness() -> dict:
    """
    Liveness probe - returns 200 as long as the process is running.
    Returns:
        {"status": "alive"}
    HTTP status: Always 200 if handler executes
    Kubernetes behavior:
        - If this fails: restart the pod
        - Use for: deadlock detection, process crash
    Note: This should NEVER fail unless the process is dead.
    It does NOT check model readiness.
    """
    pass
async def readiness(global_state: GlobalState) -> dict:
    """
    Readiness probe - returns 200 only after model is loaded and warmed.
    Args:
        global_state: Shared application state
    Returns:
        If READY:
            {
                "status": "ready",
                "model_version": str,
                "device": str,
                "ready_since": float (unix timestamp)
            }
    Raises:
        HTTPException:
            - 503 if model_state == LOADING or WARMING_UP
            - 503 if model_state == FAILED (with error details)
    Kubernetes behavior:
        - If this fails: stop routing traffic, but don't restart
        - Use for: blue/green deployment, rolling updates
    Critical: This MUST return 503 until warmup completes.
    Returning 200 before warmup means requests hit uncached CUDA kernels.
    """
    pass
```
### Lifespan Handler
```python
# 10_lifespan.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler - manages startup and shutdown.
    This runs BEFORE the app starts accepting requests, but
    health check endpoints are available immediately.
    Startup sequence:
        1. Set global_state.model_state = LOADING
        2. get_device() - detect and log device
        3. check_memory_budget() - verify memory available
        4. load_model() - load and set eval mode
        5. Set global_state.model_state = WARMING_UP
        6. warmup_model() - run dummy inferences
        7. benchmark_serialization() - validate orjson speedup
        8. Set global_state.model_state = READY
        9. Set global_state.ready_since = time.time()
    Shutdown sequence:
        1. Log shutdown start
        2. Delete model reference
        3. torch.cuda.empty_cache() if CUDA available
        4. Log shutdown complete
    Error handling:
        - Any exception during startup: set model_state = FAILED
        - Store error in global_state.load_error
        - DON'T raise (let app start so health probes can report failure)
    Critical: Health endpoints must be registered OUTSIDE this context
    so they're available immediately when the process starts.
    """
    pass
```
---
## Algorithm Specification
### Memory Budget Validation Algorithm
```
ALGORITHM: check_memory_budget
INPUT: model_path (str), device (torch.device), safety_margin (float)
OUTPUT: None (raises on failure)
1. IF device.type != "cuda":
       LOG "Memory check skipped for CPU device"
       RETURN
2. model_size_bytes ← GET file size of model_path
   IF model_size_bytes is None:
       RAISE ModelNotFound(model_path)
3. total_memory ← torch.cuda.get_device_properties(0).total_memory
   reserved ← torch.cuda.memory_reserved(0)
   allocated ← torch.cuda.memory_allocated(0)
   free_memory ← total_memory - reserved
4. // Conservative estimate: model + activations + overhead
   required_memory ← model_size_bytes * 4
5. available_after_load ← free_memory - required_memory
   min_required ← total_memory * safety_margin
6. IF available_after_load < min_required:
       RAISE MemoryBudgetExceeded(required_memory, free_memory, safety_margin)
7. LOG memory status:
       "Model: {model_size_bytes/1e6:.1f} MB"
       "Available: {free_memory/1e9:.2f} GB"
       "Post-load margin: {available_after_load/1e9:.2f} GB"
POSTCONDITION: If function returns, model can be loaded safely
```
### Warmup Algorithm
```
ALGORITHM: warmup_model
INPUT: model (nn.Module), device (torch.device), input_shape (tuple), n_warmup_runs (int)
OUTPUT: List[float] (latencies per run)
1. LOG "Starting {n_warmup_runs} warmup iterations..."
   dummy_input ← torch.randn(input_shape, device=device)
   latencies ← empty list
2. FOR i IN RANGE(n_warmup_runs):
       IF device.type == "cuda":
           torch.cuda.synchronize()
       start ← time.perf_counter()
       WITH torch.no_grad():
           _ ← model(dummy_input)
       IF device.type == "cuda":
           torch.cuda.synchronize()
       latency_ms ← (time.perf_counter() - start) * 1000
       APPEND latency_ms TO latencies
       LOG "Run {i+1}/{n_warmup_runs}: {latency_ms:.1f}ms"
3. // Verify warmup converged
   IF len(latencies) >= 2:
       final ← latencies[-1]
       variance ← abs(latencies[-1] - latencies[-2])
       IF variance > final * 0.1:
           LOG WARNING "Latency variance high ({variance:.1f}ms)"
       ELSE:
           LOG "Complete. Stable latency: {final:.1f}ms"
4. RETURN latencies
POSTCONDITION: CUDA kernel cache is populated for input_shape
POSTCONDITION: Last latency in latencies is stable (within 10% of previous)
```
### Input Validation Algorithm
```
ALGORITHM: ModelInputValidator.validate_request
INPUT: request_data (dict)
OUTPUT: (bool, Union[dict, str]) - (success, inputs or error message)
1. inputs ← empty dict
2. FOR EACH (name, schema) IN self.schemas:
       IF name NOT IN request_data:
           RETURN (False, "Missing required input: '{name}'")
       TRY:
           raw ← request_data[name]
           tensor ← torch.tensor(raw, dtype=schema.dtype)
       EXCEPT (ValueError, TypeError) AS e:
           RETURN (False, "Invalid input '{name}': {e}")
       (is_valid, error) ← schema.validate(tensor)
       IF NOT is_valid:
           RETURN (False, error)
       inputs[name] ← tensor
3. RETURN (True, {"inputs": inputs})
POSTCONDITION: If success, all inputs are valid tensors with correct dtype and shape
```
---
## Error Handling Matrix
| Error | Detected By | Recovery | User-Visible? |
|-------|-------------|----------|---------------|
| `MemoryBudgetExceeded` | `check_memory_budget()` | Log error, set state=FAILED, return 503 on readiness | Yes (503 with details) |
| `ModelNotFound` | `load_model()` | Log error, set state=FAILED, return 503 on readiness | Yes (503 with path) |
| `ModelLoadFailed` | `load_model()` catches torch exceptions | Log with stack trace, set state=FAILED | Yes (503 with error) |
| `InvalidInputShape` | `ModelInputValidator.validate_request()` | Return HTTP 400 with descriptive message | Yes (400 with expected/actual) |
| `InvalidInputDtype` | `ModelInputValidator.validate_request()` | Return HTTP 400 with dtype mismatch | Yes (400 with dtypes) |
| `SerializationError` | `deserialize_request()` | Return HTTP 400 with parse error | Yes (400 with JSON error) |
| `RuntimeError` during inference | Predict endpoint try/catch | Return HTTP 500 with error details | Yes (500 with message) |
| CUDA OOM during warmup | `warmup_model()` catches | Log, set state=FAILED, return 503 | Yes (503 service unavailable) |
---
## Implementation Sequence with Checkpoints
### Phase 1: Device Detection and Memory Budgeting (1-2 hours)
**Files to create:** `01_config.py`, `02_exceptions.py`, `03_device.py`
**Implementation steps:**
1. Define `ModelState` enum in `02_exceptions.py`
2. Define `MemoryBudgetExceeded`, `ModelNotFound`, `ModelLoadFailed` exceptions
3. Implement `get_device()` with CUDA detection and logging
4. Implement `check_memory_budget()` with the 4x multiplier formula
**Checkpoint:** 
```bash
# Run this test
python -c "
from device import get_device, check_memory_budget
import torch
device = get_device()
print(f'Detected device: {device}')
# Should pass for small model or raise for huge model
try:
    check_memory_budget('path/to/model.pt', device)
    print('Memory check passed')
except Exception as e:
    print(f'Memory check failed: {e}')
"
```
Expected: Device logged (cuda or cpu), memory check completes without error for reasonable model size.
---
### Phase 2: Model Loader with Eval Mode (1-2 hours)
**Files to create:** `04_model_loader.py`
**Implementation steps:**
1. Implement `load_model()` with error handling
2. Verify `model.eval()` is called
3. Verify model is moved to correct device
4. Implement `predict()` with `@torch.no_grad()` decorator
**Checkpoint:**
```bash
python -c "
import torch
from model_loader import load_model, predict
# Create a simple test model
model = torch.nn.Linear(10, 2)
torch.save(model, 'test_model.pt')
device = torch.device('cpu')
loaded = load_model('test_model.pt', device)
assert loaded.training == False, 'Model should be in eval mode'
print('Model loaded successfully in eval mode')
# Test prediction
inputs = torch.randn(1, 10)
output = predict(loaded, inputs, device)
print(f'Output shape: {output.shape}')
"
```
Expected: Model loads, is in eval mode, prediction works.
---
### Phase 3: CUDA Warmup (1-2 hours)
**Files to create:** `05_warmup.py`
**Implementation steps:**
1. Implement `warmup_model()` with timing loop
2. Add CUDA synchronization for accurate timing
3. Implement variance detection and warning
4. Implement `measure_warmup_effect()` for diagnostics
**Checkpoint:**
```bash
python -c "
import torch
from model_loader import load_model
from warmup import warmup_model, measure_warmup_effect
model = load_model('test_model.pt', torch.device('cpu'))
latencies = warmup_model(model, torch.device('cpu'), (1, 10), n_warmup_runs=5)
print(f'Warmup latencies: {latencies}')
assert len(latencies) == 5, 'Should have 5 latency measurements'
print('Warmup completed successfully')
"
```
Expected: 5 latency measurements logged, warmup completes.
---
### Phase 4: Input Schema Validation (2-3 hours)
**Files to create:** `06_schema.py`
**Implementation steps:**
1. Implement `InputSchema` dataclass with `validate()` method
2. Implement `ModelInputValidator` class with `validate_request()` method
3. Add comprehensive error messages for each failure case
4. Create predefined schemas for common model types
**Checkpoint:**
```bash
python -c "
from schema import InputSchema, ModelInputValidator
import torch
# Define schema
schema = InputSchema(
    name='image',
    shape=(None, 3, 224, 224),
    dtype=torch.float32,
    description='Image input'
)
validator = ModelInputValidator([schema])
# Test valid input
valid_data = {'image': [[[[0.0] * 224 for _ in range(224)] for _ in range(3)]]}
success, result = validator.validate_request(valid_data)
assert success, f'Valid input should pass: {result}'
print('Valid input passed')
# Test invalid dtype
invalid_data = {'image': [[[[0] * 224 for _ in range(224)] for _ in range(3)]]}
success, result = validator.validate_request(invalid_data)
assert not success, 'Invalid dtype should fail'
print(f'Invalid dtype correctly rejected: {result}')
"
```
Expected: Valid inputs pass, invalid inputs rejected with descriptive errors.
---
### Phase 5: orjson Serialization (1-2 hours)
**Files to create:** `07_serialization.py`
**Implementation steps:**
1. Implement `serialize_response()` with orjson
2. Implement `deserialize_request()` with error handling
3. Implement `tensor_to_json()` for PyTorch tensors
4. Implement `benchmark_serialization()` and verify 3x+ speedup
**Checkpoint:**
```bash
python -c "
from serialization import serialize_response, deserialize_request, benchmark_serialization
# Test serialization
data = {'prediction': [0.1, 0.2, 0.3], 'latency_ms': 42.5}
serialized = serialize_response(data)
print(f'Serialized: {serialized[:50]}...')
# Test deserialization
deserialized = deserialize_request(serialized)
assert deserialized == data, 'Round-trip should preserve data'
print('Round-trip successful')
# Benchmark
results = benchmark_serialization(n_runs=100)
print(f'Speedup: {results[\"speedup\"]:.1f}x')
assert results['speedup'] >= 2.0, 'Should be at least 2x faster than stdlib'
print('Benchmark passed')
"
```
Expected: Serialization works, benchmark shows 3x+ speedup.
---
### Phase 6: Health Endpoints (1-2 hours)
**Files to create:** `08_health.py`, `09_model_container.py`
**Implementation steps:**
1. Implement `GlobalState` dataclass with `ModelState` enum
2. Implement `liveness()` endpoint (always returns 200)
3. Implement `readiness()` endpoint (returns 200 only when READY)
4. Ensure FAILED state returns 503 with error details
**Checkpoint:**
```bash
python -c "
import asyncio
from health import GlobalState, ModelState, liveness, readiness
state = GlobalState()
# Test liveness (should always work)
result = asyncio.run(liveness())
assert result['status'] == 'alive', 'Liveness should return alive'
print('Liveness passed')
# Test readiness when not ready
state.model_state = ModelState.LOADING
try:
    asyncio.run(readiness(state))
    assert False, 'Should have raised HTTPException'
except Exception as e:
    print(f'Readiness correctly rejected during LOADING: {e}')
# Test readiness when ready
state.model_state = ModelState.READY
state.version = 'v1.0.0'
state.device = 'cpu'
state.ready_since = 12345.0
result = asyncio.run(readiness(state))
assert result['status'] == 'ready', 'Should return ready'
print('Readiness passed when READY')
"
```
Expected: Liveness always works, readiness respects state.
---
### Phase 7: FastAPI Integration (2-3 hours)
**Files to create:** `10_lifespan.py`, `11_endpoints.py`, `12_main.py`
**Implementation steps:**
1. Implement `lifespan()` context manager with full startup sequence
2. Implement `/predict` endpoint with validation and error handling
3. Implement `/health/liveness` and `/health/readiness` routes
4. Wire everything together in `main.py`
**Checkpoint:**
```bash
# Start server
uvicorn main:app --host 0.0.0.0 --port 8000 &
# Wait for startup
sleep 5
# Test liveness (should work immediately)
curl http://localhost:8000/health/liveness
# Expected: {"status": "alive"}
# Test readiness (should work after model loads)
curl http://localhost:8000/health/readiness
# Expected: {"status": "ready", "model_version": "v1.0.0", ...}
# Test prediction with valid input
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": [[[[0.0] * 224 for _ in range(224)] for _ in range(3)]]}'
# Expected: {"prediction": [...], "latency_ms": ..., "model_version": "v1.0.0"}
# Test prediction with invalid input
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": [1, 2, 3]}'
# Expected: 400 with descriptive error
# Kill server
pkill -f uvicorn
```
---
## Test Specification
### Test 1: Device Detection
```python
def test_device_detection():
    """Test that device detection returns valid device."""
    from device import get_device
    import torch
    device = get_device()
    # Must be a valid torch.device
    assert isinstance(device, torch.device)
    # Must be either cuda or cpu
    assert device.type in ('cuda', 'cpu')
    # If cuda, verify actually available
    if device.type == 'cuda':
        assert torch.cuda.is_available()
```
### Test 2: Memory Budget Validation
```python
def test_memory_budget_sufficient():
    """Test that memory check passes for small model."""
    from device import check_memory_budget
    import torch
    import tempfile
    import os
    # Create tiny model
    model = torch.nn.Linear(10, 2)
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(model, f.name)
        model_path = f.name
    try:
        device = torch.device('cpu')  # Always works
        check_memory_budget(model_path, device, safety_margin=0.3)
        # Should not raise
    finally:
        os.unlink(model_path)
def test_memory_budget_insufficient():
    """Test that memory check fails for huge model estimate."""
    from device import check_memory_budget
    from exceptions import MemoryBudgetExceeded
    import torch
    import tempfile
    import os
    # Create a file that claims to be huge
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        f.write(b'\0' * 1024)  # Small file
        model_path = f.name
    try:
        # Mock huge size
        original_getsize = os.path.getsize
        os.path.getsize = lambda p: 100 * 1024 * 1024 * 1024  # 100 GB
        if torch.cuda.is_available():
            device = torch.device('cuda')
            with pytest.raises(MemoryBudgetExceeded):
                check_memory_budget(model_path, device, safety_margin=0.3)
    finally:
        os.path.getsize = original_getsize
        os.unlink(model_path)
```
### Test 3: Model Loading
```python
def test_model_load_eval_mode():
    """Test that loaded model is in eval mode."""
    from model_loader import load_model
    import torch
    import tempfile
    import os
    # Create model with dropout (affected by eval mode)
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(5, 2)
    )
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(model, f.name)
        model_path = f.name
    try:
        loaded = load_model(model_path, torch.device('cpu'))
        # Must be in eval mode
        assert not loaded.training
        # Dropout should be disabled (deterministic output)
        input_tensor = torch.randn(1, 10)
        output1 = loaded(input_tensor)
        output2 = loaded(input_tensor)
        assert torch.allclose(output1, output2), "Dropout should be disabled"
    finally:
        os.unlink(model_path)
```
### Test 4: Warmup Latency Convergence
```python
def test_warmup_convergence():
    """Test that warmup latencies converge."""
    from model_loader import load_model
    from warmup import warmup_model
    import torch
    import tempfile
    import os
    model = torch.nn.Linear(100, 10)
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(model, f.name)
        model_path = f.name
    try:
        loaded = load_model(model_path, torch.device('cpu'))
        latencies = warmup_model(loaded, torch.device('cpu'), (1, 100), n_warmup_runs=10)
        assert len(latencies) == 10
        # Last two latencies should be within 20% of each other
        if len(latencies) >= 2:
            final = latencies[-1]
            prev = latencies[-2]
            variance_pct = abs(final - prev) / max(final, 0.001) * 100
            assert variance_pct < 50, f"High variance: {variance_pct:.1f}%"
    finally:
        os.unlink(model_path)
```
### Test 5: Input Validation
```python
def test_input_validation_valid():
    """Test that valid input passes validation."""
    from schema import InputSchema, ModelInputValidator
    import torch
    schema = InputSchema(
        name='image',
        shape=(None, 3, 224, 224),
        dtype=torch.float32,
        description='Test image'
    )
    validator = ModelInputValidator([schema])
    # Valid input: batch=1, channels=3, height=224, width=224
    valid = {'image': [[[[0.5] * 224] * 224] * 3]}
    success, result = validator.validate_request(valid)
    assert success
    assert 'inputs' in result
    assert result['inputs']['image'].shape == (1, 3, 224, 224)
    assert result['inputs']['image'].dtype == torch.float32
def test_input_validation_wrong_shape():
    """Test that wrong shape is rejected."""
    from schema import InputSchema, ModelInputValidator
    schema = InputSchema(
        name='image',
        shape=(None, 3, 224, 224),
        dtype=torch.float32,
        description='Test image'
    )
    validator = ModelInputValidator([schema])
    # Wrong shape: batch=1, channels=3, height=100, width=100
    invalid = {'image': [[[[0.5] * 100] * 100] * 3]}
    success, result = validator.validate_request(invalid)
    assert not success
    assert 'Wrong shape' in result or 'dimension' in result.lower()
def test_input_validation_missing_field():
    """Test that missing field is rejected."""
    from schema import InputSchema, ModelInputValidator
    schema = InputSchema(
        name='image',
        shape=(None, 3, 224, 224),
        dtype=torch.float32,
        description='Test image'
    )
    validator = ModelInputValidator([schema])
    # Missing 'image' field
    invalid = {'other_field': [1, 2, 3]}
    success, result = validator.validate_request(invalid)
    assert not success
    assert 'Missing' in result
```
### Test 6: Serialization Benchmark
```python
def test_serialization_speedup():
    """Test that orjson is faster than stdlib json."""
    from serialization import benchmark_serialization
    import numpy as np
    results = benchmark_serialization(n_runs=50)
    # Must be at least 2x faster for numpy arrays
    assert results['speedup'] >= 2.0, f"Speedup {results['speedup']:.1f}x < 2.0x"
    # Output should be reasonable size
    assert results['payload_size_kb'] > 0
```
### Test 7: Health Probes
```python
@pytest.mark.asyncio
async def test_liveness_always_succeeds():
    """Test that liveness always returns 200."""
    from health import liveness
    result = await liveness()
    assert result['status'] == 'alive'
@pytest.mark.asyncio
async def test_readiness_respects_state():
    """Test that readiness reflects model state."""
    from health import readiness, GlobalState, ModelState
    from fastapi import HTTPException
    state = GlobalState()
    # LOADING should return 503
    state.model_state = ModelState.LOADING
    with pytest.raises(HTTPException) as exc:
        await readiness(state)
    assert exc.value.status_code == 503
    # READY should return 200
    state.model_state = ModelState.READY
    state.version = 'v1.0.0'
    state.device = 'cpu'
    state.ready_since = 12345.0
    result = await readiness(state)
    assert result['status'] == 'ready'
    # FAILED should return 503 with error
    state.model_state = ModelState.FAILED
    state.load_error = 'Test error'
    with pytest.raises(HTTPException) as exc:
        await readiness(state)
    assert exc.value.status_code == 503
```
---
## Performance Targets
| Operation | Target | How to Measure |
|-----------|--------|----------------|
| Single-request p50 latency | < 50ms | Run 100 sequential inferences, compute median |
| Serialization (batch-32 embeddings) | < 10ms | `benchmark_serialization()` with 32×768 payload |
| Warmup completion | < 5 seconds | Time from warmup start to last inference |
| Memory safety margin | 30% reserved | `check_memory_budget()` with safety_margin=0.3 |
| First vs subsequent inference | < 10% variance | `measure_warmup_effect()`, compare first to second |
| Model loading | < 10 seconds | Time from load start to model.eval() complete |
| Health probe response | < 5ms | `curl` timing on liveness/readiness endpoints |
---
## State Machine: Model Loading Lifecycle
```
                    ┌─────────────┐
                    │   STARTUP   │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
              ┌─────│   LOADING   │─────┐
              │     └──────┬──────┘     │
              │            │            │
         load fail    load success   timeout
              │            │            │
              ▼            ▼            ▼
       ┌───────────┐ ┌─────────────┐ ┌──────────┐
       │  FAILED   │ │  WARMING_UP │ │  FAILED  │
       └───────────┘ └──────┬──────┘ └──────────┘
                           │
                     warmup success
                           │
                           ▼
                    ┌─────────────┐
                    │    READY    │◄──────────┐
                    └──────┬──────┘           │
                           │                  │
                    serving traffic            │
                           │                  │
                    ┌──────▼──────┐           │
                    │  SHUTDOWN   │───────────┘
                    └─────────────┘  (restart)
ILLEGAL TRANSITIONS:
- LOADING → READY (must go through WARMING_UP)
- FAILED → WARMING_UP (failed is terminal)
- READY → LOADING (no reload without shutdown)
```
---
## Concurrency Specification
### Threading Model
```
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Event Loop                          │
│                     (Single-threaded async)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  /predict    │  │ /health/     │  │  lifespan    │          │
│  │  handler     │  │  handlers    │  │  handler     │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                  │                  │
│         │                 │                  │                  │
│         ▼                 ▼                  ▼                  │
│  ┌──────────────────────────────────────────────────┐          │
│  │              GlobalState (shared)                 │          │
│  │  - model_state: ModelState                       │          │
│  │  - model: Optional[nn.Module]                    │          │
│  │  - device: Optional[torch.device]                │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
### Thread Safety Guarantees
| Component | Thread Safety | Reason |
|-----------|--------------|--------|
| `GlobalState` | Safe for async reads | FastAPI cooperative multitasking |
| `model.predict()` | Safe | PyTorch inference is thread-safe |
| `ModelInputValidator` | Safe | Creates new tensors per request |
| `torch.cuda` operations | Safe | CUDA handles internal synchronization |
| Model loading | NOT safe during load | Mutates CUDA context |
### Lock Ordering (for future multi-threaded extensions)
If threading is introduced:
1. Acquire `model_lock` before accessing `GlobalState.model`
2. Acquire `state_lock` before modifying `GlobalState.model_state`
3. Never hold both locks simultaneously across await points
---
[[CRITERIA_JSON: {"module_id": "ml-model-serving-m1", "criteria": ["Load a PyTorch model with eval mode enabled and torch.no_grad() context for all inference operations", "GPU inference is used when CUDA is available with automatic fallback to CPU that is logged", "Model warmup runs N configurable dummy inference requests immediately after model loading completes to trigger JIT compilation and CUDA kernel caching", "Inference endpoint accepts JSON input, validates tensor shape and dtype against model input schema, and returns JSON results", "Request deserialization overhead is measured; response serialization uses orjson with benchmarked improvement over stdlib json (minimum 2x speedup)", "Health readiness endpoint returns 200 only after model is loaded and warmup is complete", "Health liveness endpoint returns 200 as long as the process is running regardless of model state", "Input validation rejects requests with wrong shape/dtype and returns 400 with descriptive error message indicating expected vs actual values", "Single-request inference p50 latency is under 50ms for a standard image classification or text embedding model", "Memory budget validation checks available GPU memory before loading and raises MemoryBudgetExceeded if insufficient with 30% safety margin", "ModelState enum tracks LOADING, WARMING_UP, READY, and FAILED states with proper transitions", "LoadedModel container provides predict() method with automatic device placement and torch.no_grad() decorator", "InputSchema validates tensor rank, dtype, and non-variable dimensions with descriptive error messages", "Serialization benchmark compares stdlib json vs orjson for numpy array payloads and logs speedup at startup"]}]
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: ml-model-serving-m2 -->
# Request Batching
## Module Charter
This module implements dynamic request batching to maximize GPU utilization by grouping individual inference requests into batched tensor operations. It transforms a request-per-inference model into a batch-per-inference model, achieving 2-6x throughput improvement on GPU hardware. The batching engine runs as a background asyncio task that continuously harvests requests from an async queue, forms batches up to a configurable size or timeout, runs batched inference, and routes results back to individual requesters via asyncio.Future resolution. The module handles variable-length inputs through padding with attention masks, implements backpressure via queue depth limits with HTTP 503 rejection, and exposes detailed metrics for monitoring batch efficiency. It explicitly does NOT handle model versioning (M3), A/B routing (M4), or drift detection (M5). Upstream dependencies are the HTTP endpoint layer submitting requests; downstream consumers are the model inference layer and metrics collection. The key invariant is that every submitted request MUST eventually receive exactly one result or exception via its Future—no request may be orphaned, duplicated, or receive another request's result.
---
## File Structure
Create files in this exact sequence:
```
ml-model-serving/m2_batching/
├── 01_config.py              # Batching configuration constants
├── 02_exceptions.py          # Batching-specific exception hierarchy
├── 03_batch_request.py       # BatchRequest dataclass with Future
├── 04_request_queue.py       # Async queue with backpressure
├── 05_padder.py              # Variable-length input padding with masks
├── 06_batcher.py             # DynamicBatcher with batch formation loop
├── 07_result_router.py       # Un-batching and result routing
├── 08_metrics.py             # BatcherMetrics with percentile calculation
├── 09_benchmark.py           # Throughput comparison benchmark
└── 10_integration.py         # FastAPI integration with batching
```
---
## Complete Data Model
### RequestStatus Enum
The lifecycle states of a batch request:
```python
# 02_exceptions.py
from enum import Enum
class RequestStatus(Enum):
    """Lifecycle states for a batch request."""
    QUEUED = "queued"        # In queue, waiting for batch formation
    PROCESSING = "processing"  # In active batch, inference running
    COMPLETED = "completed"   # Result delivered via Future
    FAILED = "failed"        # Exception set on Future
```
**Why this enum exists:** Metrics collection needs to track request lifecycle stages. The FAILED status ensures we can distinguish between successful completion and error delivery.
---
### BatchRequest Dataclass
The atomic unit of the batching system:
```python
# 03_batch_request.py
from dataclasses import dataclass, field
from typing import Optional, Tuple
import asyncio
import time
import uuid
import torch
@dataclass
class BatchRequest:
    """
    Container for a single inference request in the batch queue.
    The Future allows the request handler to await the result
    while the batch processor resolves it asynchronously.
    Thread Safety: Future operations are thread-safe. Status updates
    should only happen within the batch loop (single-threaded async).
    Invariants:
        - request_id is unique across all active requests
        - future is created but not done at construction
        - set_result() and set_error() are called exactly once
        - After Future resolution, no further mutations occur
    """
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    input_tensor: Optional[torch.Tensor] = None
    arrival_time: float = field(default_factory=time.perf_counter)
    future: asyncio.Future = field(
        default_factory=lambda: asyncio.get_event_loop().create_future()
    )
    status: RequestStatus = RequestStatus.QUEUED
    # Metadata for metrics and routing
    input_shape: Tuple[int, ...] = ()
    original_length: Optional[int] = None  # For variable-length inputs
    def wait_time_ms(self) -> float:
        """
        Time spent waiting in queue (milliseconds).
        Returns:
            Milliseconds since arrival_time
        """
        return (time.perf_counter() - self.arrival_time) * 1000
    def set_result(self, result: torch.Tensor) -> None:
        """
        Mark request as completed with result.
        Args:
            result: Output tensor from inference
        Side effects:
            - Sets status to COMPLETED
            - Resolves Future with result
        Idempotent: Does nothing if Future already done.
        """
        self.status = RequestStatus.COMPLETED
        if not self.future.done():
            self.future.set_result(result)
    def set_error(self, error: Exception) -> None:
        """
        Mark request as failed with error.
        Args:
            error: Exception that caused failure
        Side effects:
            - Sets status to FAILED
            - Sets exception on Future
        Idempotent: Does nothing if Future already done.
        """
        self.status = RequestStatus.FAILED
        if not self.future.done():
            self.future.set_exception(error)
    def is_resolved(self) -> bool:
        """Check if Future has been resolved (success or failure)."""
        return self.future.done()
```
---
### BatcherConfig Dataclass
Configuration for the batching engine:
```python
# 01_config.py
from dataclasses import dataclass
@dataclass
class BatcherConfig:
    """
    Configuration for the dynamic batcher.
    Tuning Guidelines:
        - max_batch_size: Larger = better throughput, higher latency
          Start with 32 for image models, 16 for transformer models
        - timeout_ms: Lower = lower latency, smaller batches
          Start with 50ms, adjust based on SLA
        - max_queue_depth: Higher = more throughput buffer, more memory
          Start with 1000, monitor rejection rate
    """
    max_batch_size: int = 32
    timeout_ms: float = 50.0
    max_queue_depth: int = 1000
    backpressure_enabled: bool = True
    # Variable-length input settings
    max_sequence_length: int = 512
    pad_value: int = 0
    # Metrics settings
    metrics_window_size: int = 1000
    def __post_init__(self):
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be > 0")
        if self.max_queue_depth < 1:
            raise ValueError("max_queue_depth must be >= 1")
```
---
### BatchMetadata Dataclass
Metadata tracking for batch processing:
```python
# 06_batcher.py
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import time
@dataclass
class BatchMetadata:
    """
    Metadata for a formed batch, used for result routing.
    Tracks the mapping between batch indices and original requests,
    plus any padding information needed for un-batching.
    """
    requests: List[BatchRequest] = field(default_factory=list)
    original_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    padding_mask: Optional[torch.Tensor] = None
    needs_unpadding: bool = False
    formation_time_ms: float = 0.0
    batch_size: int = 0
    def add_request(self, request: BatchRequest) -> None:
        """Add a request to the batch metadata."""
        self.requests.append(request)
        self.original_shapes.append(request.input_shape)
        self.batch_size = len(self.requests)
```
---
### BatcherMetrics Dataclass
Comprehensive metrics for monitoring:
```python
# 08_metrics.py
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Optional
import statistics
import time
@dataclass
class BatcherMetrics:
    """
    Comprehensive metrics for the dynamic batcher.
    Metrics categories:
    1. Throughput: requests/sec, batches/sec
    2. Latency: queue wait time, inference time, total time
    3. Efficiency: average batch size, padding overhead
    4. Health: queue depth, rejection rate
    Sliding windows are used for percentile calculation to bound memory.
    """
    window_size: int = 1000
    # Sliding windows for percentile calculation
    queue_wait_times_ms: deque = field(
        default_factory=lambda: deque(maxlen=1000)
    )
    inference_times_ms: deque = field(
        default_factory=lambda: deque(maxlen=1000)
    )
    total_times_ms: deque = field(
        default_factory=lambda: deque(maxlen=1000)
    )
    batch_sizes: deque = field(
        default_factory=lambda: deque(maxlen=1000)
    )
    # Counters
    total_requests: int = 0
    total_batches: int = 0
    total_rejected: int = 0
    total_timeouts: int = 0
    # Timing
    start_time: float = field(default_factory=time.time)
    def record_request(
        self,
        queue_wait_ms: float,
        inference_ms: float,
        total_ms: float
    ) -> None:
        """Record latency metrics for a completed request."""
        self.queue_wait_times_ms.append(queue_wait_ms)
        self.inference_times_ms.append(inference_ms)
        self.total_times_ms.append(total_ms)
        self.total_requests += 1
    def record_batch(self, batch_size: int) -> None:
        """Record batch formation metrics."""
        self.batch_sizes.append(batch_size)
        self.total_batches += 1
    def record_rejection(self) -> None:
        """Record a rejected request (backpressure)."""
        self.total_rejected += 1
    def record_timeout(self) -> None:
        """Record a timed-out request."""
        self.total_timeouts += 1
    def percentile(self, data: deque, p: float) -> float:
        """
        Calculate percentile from deque.
        Args:
            data: Deque of values
            p: Percentile (0-100)
        Returns:
            Value at percentile, or 0.0 if empty
        """
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * p / 100)
        return sorted_data[min(idx, len(sorted_data) - 1)]
    def summary(self) -> Dict[str, float]:
        """Generate metrics summary for Prometheus export."""
        def safe_mean(d: deque) -> float:
            return statistics.mean(d) if d else 0.0
        elapsed_seconds = time.time() - self.start_time
        return {
            # Throughput
            "batcher_requests_total": float(self.total_requests),
            "batcher_batches_total": float(self.total_batches),
            "batcher_rejected_total": float(self.total_rejected),
            "batcher_timeouts_total": float(self.total_timeouts),
            "batcher_rps": self.total_requests / max(1, elapsed_seconds),
            # Latency percentiles
            "batcher_queue_wait_ms_p50": self.percentile(self.queue_wait_times_ms, 50),
            "batcher_queue_wait_ms_p95": self.percentile(self.queue_wait_times_ms, 95),
            "batcher_queue_wait_ms_p99": self.percentile(self.queue_wait_times_ms, 99),
            "batcher_inference_ms_p50": self.percentile(self.inference_times_ms, 50),
            "batcher_inference_ms_p95": self.percentile(self.inference_times_ms, 95),
            "batcher_inference_ms_p99": self.percentile(self.inference_times_ms, 99),
            "batcher_total_ms_p50": self.percentile(self.total_times_ms, 50),
            "batcher_total_ms_p95": self.percentile(self.total_times_ms, 95),
            "batcher_total_ms_p99": self.percentile(self.total_times_ms, 99),
            # Batch efficiency
            "batcher_avg_batch_size": safe_mean(self.batch_sizes),
            "batcher_max_batch_size": float(max(self.batch_sizes)) if self.batch_sizes else 0.0,
            "batcher_min_batch_size": float(min(self.batch_sizes)) if self.batch_sizes else 0.0,
            # Rates
            "batcher_rejection_rate": (
                self.total_rejected / max(1, self.total_requests + self.total_rejected)
            ),
        }
```
---
### Exception Hierarchy
```python
# 02_exceptions.py
class BatchingError(Exception):
    """Base exception for batching errors."""
    pass
class BackpressureRejection(BatchingError):
    """
    Raised when request is rejected due to queue capacity.
    Attributes:
        queue_depth: Current queue size
        max_depth: Maximum allowed queue size
    """
    def __init__(self, queue_depth: int, max_depth: int):
        self.queue_depth = queue_depth
        self.max_depth = max_depth
        super().__init__(
            f"Queue at capacity ({queue_depth}/{max_depth}). "
            "Retry later."
        )
class RequestTimeout(BatchingError):
    """
    Raised when request exceeds maximum wait time.
    Attributes:
        wait_time_ms: Actual wait time
        max_wait_ms: Maximum allowed wait time
    """
    def __init__(self, wait_time_ms: float, max_wait_ms: float):
        self.wait_time_ms = wait_time_ms
        self.max_wait_ms = max_wait_ms
        super().__init__(
            f"Request waited {wait_time_ms:.1f}ms, "
            f"exceeding limit of {max_wait_ms:.1f}ms"
        )
class PaddingError(BatchingError):
    """
    Raised when variable-length input cannot be padded.
    Attributes:
        reason: Description of padding failure
    """
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Padding failed: {reason}")
class InferenceError(BatchingError):
    """
    Raised when batch inference fails.
    Attributes:
        batch_size: Size of failed batch
        original_error: Underlying exception
    """
    def __init__(self, batch_size: int, original_error: Exception):
        self.batch_size = batch_size
        self.original_error = original_error
        super().__init__(
            f"Batch inference failed for {batch_size} requests: {original_error}"
        )
class ResultRoutingError(BatchingError):
    """
    Raised when results cannot be routed to requesters.
    Attributes:
        request_id: ID of affected request
        reason: Description of routing failure
    """
    def __init__(self, request_id: str, reason: str):
        self.request_id = request_id
        self.reason = reason
        super().__init__(
            f"Failed to route result for request {request_id}: {reason}"
        )
```
---
## Tensor Shape Specifications
### Fixed-Size Image Inputs
```
Individual Request:
  input_tensor: (channels, height, width)
  Example: (3, 224, 224) for ResNet
Batched Input:
  batched_tensor: (batch, channels, height, width)
  Example: (32, 3, 224, 224) for batch_size=32
Batched Output:
  batched_output: (batch, num_classes)
  Example: (32, 1000) for ImageNet
After Un-batching:
  individual_output: (num_classes,)
  Example: (1000,) per request
```
### Variable-Length Sequence Inputs
```
Individual Request:
  input_ids: (seq_len,)
  Example: (128,) for short sequence
  attention_mask: (seq_len,) - optional, 1s for real tokens
Batched Input (padded to max_seq_len):
  input_ids: (batch, max_seq_len)
  Example: (16, 512) for batch with max length 512
  attention_mask: (batch, max_seq_len)
  1 for real tokens, 0 for padding
Batched Output:
  batched_output: (batch, hidden_dim) or (batch, seq_len, hidden_dim)
  Example: (16, 768) for pooled transformer output
Padding Overhead:
  overhead = 1 - (sum(original_lengths) / (batch * max_seq_len))
  Target: < 30% for typical text workloads
```
---
## Interface Contracts
### RequestQueue
```python
# 04_request_queue.py
import asyncio
from typing import List, Optional
from dataclasses import dataclass
import logging
logger = logging.getLogger(__name__)
@dataclass
class QueueMetrics:
    """Snapshot of queue state."""
    depth: int
    max_depth: int
    total_enqueued: int
    total_rejected: int
    total_processed: int
    rejection_rate: float
class RequestQueue:
    """
    Async queue with backpressure and metrics.
    This extends asyncio.Queue with:
    - Maximum depth with rejection (backpressure)
    - Metrics tracking (queue depth, wait times)
    - Graceful shutdown support
    Thread Safety: All public methods are safe for concurrent
    access within asyncio's cooperative multitasking model.
    Invariants:
        - If qsize() >= max_depth, enqueue() rejects new requests
        - get_batch() returns requests in FIFO order
        - Shutdown requests are never enqueued
    """
    def __init__(
        self,
        max_depth: int = 1000,
        name: str = "inference"
    ):
        """
        Initialize the request queue.
        Args:
            max_depth: Maximum queue depth before rejection
            name: Queue name for logging and metrics
        """
        self.queue: asyncio.Queue[BatchRequest] = asyncio.Queue()
        self.max_depth = max_depth
        self.name = name
        # Metrics
        self.total_enqueued = 0
        self.total_rejected = 0
        self.total_processed = 0
        self._shutdown = False
    async def enqueue(self, request: BatchRequest) -> bool:
        """
        Add request to queue with backpressure check.
        Args:
            request: BatchRequest to enqueue
        Returns:
            True if enqueued successfully
            False if rejected due to backpressure
        Side effects:
            - Sets error on request if rejected
            - Logs enqueue/rejection
        Postcondition: If returns True, request will eventually
        be returned by get_batch() or cleared on shutdown.
        """
        if self._shutdown:
            request.set_error(RuntimeError("Queue is shut down"))
            return False
        if self.queue.qsize() >= self.max_depth:
            self.total_rejected += 1
            request.set_error(
                BackpressureRejection(
                    queue_depth=self.queue.qsize(),
                    max_depth=self.max_depth
                )
            )
            logger.warning(
                f"[Queue:{self.name}] Rejected request, "
                f"queue at capacity ({self.queue.qsize()}/{self.max_depth})"
            )
            return False
        await self.queue.put(request)
        self.total_enqueued += 1
        logger.debug(
            f"[Queue:{self.name}] Enqueued request, "
            f"depth={self.queue.qsize()}"
        )
        return True
    async def get_batch(
        self,
        max_batch_size: int,
        timeout_ms: float
    ) -> List[BatchRequest]:
        """
        Wait for and return a batch of requests.
        Batch formation rules:
        1. Wait up to timeout_ms for the first request
        2. Collect up to max_batch_size requests
        3. Return immediately when batch is full OR timeout expires
        This bounds worst-case latency while maximizing batch fullness.
        Args:
            max_batch_size: Maximum number of requests to return
            timeout_ms: Maximum wait time for first request
        Returns:
            List of BatchRequest objects (may be empty if timeout
            with no requests, or partial batch if queue drained)
        Postconditions:
            - All returned requests have status PROCESSING
            - Requests are in FIFO order
            - No request appears more than once
        """
        batch: List[BatchRequest] = []
        # Wait for first request (with timeout)
        try:
            first_request = await asyncio.wait_for(
                self.queue.get(),
                timeout=timeout_ms / 1000.0
            )
            first_request.status = RequestStatus.PROCESSING
            batch.append(first_request)
        except asyncio.TimeoutError:
            # No requests arrived within timeout
            return batch
        # Collect additional requests without waiting
        while len(batch) < max_batch_size:
            try:
                request = self.queue.get_nowait()
                request.status = RequestStatus.PROCESSING
                batch.append(request)
            except asyncio.QueueEmpty:
                break
        self.total_processed += len(batch)
        return batch
    def qsize(self) -> int:
        """Current queue depth."""
        return self.queue.qsize()
    def metrics(self) -> QueueMetrics:
        """Get queue metrics snapshot."""
        total = self.total_enqueued + self.total_rejected
        return QueueMetrics(
            depth=self.qsize(),
            max_depth=self.max_depth,
            total_enqueued=self.total_enqueued,
            total_rejected=self.total_rejected,
            total_processed=self.total_processed,
            rejection_rate=self.total_rejected / max(1, total)
        )
    def shutdown(self) -> None:
        """
        Signal queue to stop accepting requests.
        Postcondition: All subsequent enqueue() calls return False.
        """
        self._shutdown = True
        logger.info(f"[Queue:{self.name}] Shutdown initiated")
    async def drain(self) -> List[BatchRequest]:
        """
        Drain all remaining requests from queue.
        Returns:
            List of all requests that were in queue
        Use case: Graceful shutdown, processing remaining requests.
        """
        drained = []
        while not self.queue.empty():
            try:
                request = self.queue.get_nowait()
                drained.append(request)
            except asyncio.QueueEmpty:
                break
        return drained
```
---
### VariableLengthPadder
```python
# 05_padder.py
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass
@dataclass
class PaddingResult:
    """Result of padding a batch of variable-length inputs."""
    padded_tensor: torch.Tensor
    attention_mask: torch.Tensor
    original_lengths: List[int]
    padding_overhead: float  # Fraction of batch that is padding
class VariableLengthPadder:
    """
    Handles padding of variable-length sequences for batch inference.
    Key insight: Padding adds zeros to make shapes match, but those zeros
    shouldn't affect the model's output. Attention masks tell the model
    which positions are real vs padding.
    The attention mask is used by setting attention scores to -inf for
    masked positions before softmax, causing them to receive zero weight.
    Invariants:
        - All padded sequences have same length (max in batch)
        - Attention mask has 1 for real tokens, 0 for padding
        - Padding uses pad_value (default 0)
    """
    def __init__(
        self,
        pad_value: int = 0,
        max_length: Optional[int] = None
    ):
        """
        Initialize the padder.
        Args:
            pad_value: Value to use for padding (default 0)
            max_length: Cap for maximum sequence length (truncation)
        """
        self.pad_value = pad_value
        self.max_length = max_length
    def pad_batch(
        self,
        tensors: List[torch.Tensor]
    ) -> PaddingResult:
        """
        Pad a batch of variable-length tensors to uniform length.
        For sequence models, we pad to the longest sequence in the batch.
        Args:
            tensors: List of 1D or 2D tensors with varying sequence length
                    Shapes: (seq_len,) or (seq_len, features)
        Returns:
            PaddingResult with:
            - padded_tensor: (batch, max_seq_len) or (batch, max_seq_len, features)
            - attention_mask: (batch, max_seq_len)
            - original_lengths: List of original sequence lengths
            - padding_overhead: Fraction of batch that is padding
        Raises:
            PaddingError: If tensors have incompatible dimensions
        """
        if not tensors:
            raise PaddingError("Empty tensor list")
        # Determine tensor shape type
        first_shape = tensors[0].shape
        is_2d = len(first_shape) == 2
        # Validate all tensors have same rank
        for i, t in enumerate(tensors):
            if len(t.shape) != len(first_shape):
                raise PaddingError(
                    f"Tensor {i} has rank {len(t.shape)}, "
                    f"expected {len(first_shape)}"
                )
        # Get sequence lengths (dimension 0)
        lengths = [t.shape[0] for t in tensors]
        # Determine target length
        max_len = max(lengths)
        if self.max_length:
            max_len = min(max_len, self.max_length)
        batch_size = len(tensors)
        # Create output tensors
        dtype = tensors[0].dtype
        device = tensors[0].device
        if is_2d:
            feature_dim = first_shape[1]
            padded = torch.full(
                (batch_size, max_len, feature_dim),
                self.pad_value,
                dtype=dtype,
                device=device
            )
        else:
            padded = torch.full(
                (batch_size, max_len),
                self.pad_value,
                dtype=dtype,
                device=device
            )
        # Attention mask: 1 for real, 0 for padding
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
            device=device
        )
        # Fill with actual values
        for i, t in enumerate(tensors):
            seq_len = min(t.shape[0], max_len)
            if is_2d:
                padded[i, :seq_len, :] = t[:seq_len]
            else:
                padded[i, :seq_len] = t[:seq_len]
            attention_mask[i, :seq_len] = 1
        # Calculate padding overhead
        total_tokens = sum(min(l, max_len) for l in lengths)
        total_padded = max_len * batch_size
        padding_overhead = 1.0 - (total_tokens / total_padded)
        return PaddingResult(
            padded_tensor=padded,
            attention_mask=attention_mask,
            original_lengths=lengths,
            padding_overhead=padding_overhead
        )
    def calculate_padding_overhead(self, lengths: List[int]) -> float:
        """
        Calculate how much padding would be added vs actual content.
        Args:
            lengths: List of sequence lengths
        Returns:
            Fraction of batch that would be padding (0.0 to 1.0)
        """
        if not lengths:
            return 0.0
        max_len = max(lengths)
        if self.max_length:
            max_len = min(max_len, self.max_length)
        total_tokens = sum(min(l, max_len) for l in lengths)
        total_padded = max_len * len(lengths)
        return 1.0 - (total_tokens / total_padded)
```
---
### DynamicBatcher
```python
# 06_batcher.py
import asyncio
import time
import logging
from typing import Callable, Optional, List, Tuple, Any
import torch
from .01_config import BatcherConfig
from .03_batch_request import BatchRequest
from .04_request_queue import RequestQueue
from .05_padder import VariableLengthPadder, PaddingResult
from .07_result_router import ResultRouter
from .08_metrics import BatcherMetrics
logger = logging.getLogger(__name__)
class DynamicBatcher:
    """
    Continuously forms batches from queued requests and processes them.
    This is the heart of the batching system:
    - Runs as a background asyncio task
    - Monitors the request queue
    - Forms batches with timeout-bounded latency
    - Dispatches to model inference
    - Routes results back to requesters
    Thread Safety: All public methods are safe for concurrent access.
    The batch loop runs in a single task, ensuring sequential batch processing.
    Invariants:
        - Every enqueued request receives exactly one result or exception
        - Batch formation respects max_batch_size and timeout_ms
        - Results are routed to the correct requester (by index)
    """
    def __init__(
        self,
        config: BatcherConfig,
        model_fn: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        model_input_type: str = "fixed"  # "fixed" or "variable"
    ):
        """
        Initialize the dynamic batcher.
        Args:
            config: Batching configuration
            model_fn: Inference function (takes batch tensor, returns batch tensor)
            device: Device for tensor operations
            model_input_type: "fixed" for same-size inputs, "variable" for sequences
        """
        self.config = config
        self.model_fn = model_fn
        self.device = device
        self.model_input_type = model_input_type
        # Queue for incoming requests
        self.queue = RequestQueue(
            max_depth=config.max_queue_depth,
            name="inference"
        )
        # Padder for variable-length inputs
        self.padder = VariableLengthPadder(
            pad_value=config.pad_value,
            max_length=config.max_sequence_length
        )
        # Result router
        self.router = ResultRouter()
        # Metrics
        self.metrics = BatcherMetrics(window_size=config.metrics_window_size)
        # Background task
        self._task: Optional[asyncio.Task] = None
        self._running = False
        # Current batch (for result routing during processing)
        self._current_batch: List[BatchRequest] = []
        self._current_metadata: Optional[BatchMetadata] = None
    async def submit(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Submit a request for batched inference.
        This is called by the HTTP handler for each incoming request.
        The call returns when the result is ready (awaitable).
        Args:
            input_tensor: Input data for inference
        Returns:
            Model output tensor (on CPU)
        Raises:
            BackpressureRejection: If queue is at capacity
            RequestTimeout: If request waited too long (handled by caller)
            Exception: Any inference error from the model
        Postcondition: Future is resolved (success or exception)
        """
        request = BatchRequest(
            input_tensor=input_tensor,
            input_shape=tuple(input_tensor.shape)
        )
        # Attempt to enqueue (respects backpressure)
        success = await self.queue.enqueue(request)
        if not success:
            self.metrics.record_rejection()
            # Error already set on request by queue
            raise await request.future  # Will raise the set exception
        # Wait for result (set by batch processor)
        return await request.future
    async def _batch_loop(self) -> None:
        """
        Main loop: continuously form and process batches.
        This runs forever (until stop() is called), performing:
        1. Wait for batch (with timeout)
        2. Pad and stack inputs (if variable-length)
        3. Run batch inference
        4. Split and route results
        Error handling: Any error in batch processing sets exceptions
        on all requests in the batch, then the loop continues.
        """
        logger.info(
            f"[Batcher] Starting batch loop: "
            f"max_batch={self.config.max_batch_size}, "
            f"timeout={self.config.timeout_ms}ms"
        )
        while self._running:
            try:
                # Phase 1: Form batch
                batch_start = time.perf_counter()
                batch = await self.queue.get_batch(
                    max_batch_size=self.config.max_batch_size,
                    timeout_ms=self.config.timeout_ms
                )
                if not batch:
                    # Empty batch (timeout with no requests)
                    continue
                # Record batch formation time
                formation_ms = (time.perf_counter() - batch_start) * 1000
                # Store current batch for result routing
                self._current_batch = batch
                self._current_metadata = BatchMetadata(
                    requests=batch,
                    formation_time_ms=formation_ms,
                    batch_size=len(batch)
                )
                # Phase 2: Prepare inputs
                inputs_tensor, batch_metadata = self._prepare_batch_inputs(batch)
                self._current_metadata = batch_metadata
                # Phase 3: Run inference
                inference_start = time.perf_counter()
                outputs = await self._run_inference(inputs_tensor)
                inference_ms = (time.perf_counter() - inference_start) * 1000
                # Phase 4: Split and route results
                self._route_results(outputs, batch_metadata, inference_ms)
                # Update metrics
                self.metrics.record_batch(len(batch))
                logger.debug(
                    f"[Batcher] Batch processed: "
                    f"size={len(batch)}, formation={formation_ms:.1f}ms, "
                    f"inference={inference_ms:.1f}ms"
                )
            except asyncio.CancelledError:
                logger.info("[Batcher] Batch loop cancelled")
                break
            except Exception as e:
                logger.error(
                    f"[Batcher] Error in batch loop: {e}", 
                    exc_info=True
                )
                # Fail all requests in current batch
                self._fail_current_batch(e)
                # Brief pause before retry
                await asyncio.sleep(0.1)
    def _prepare_batch_inputs(
        self, 
        batch: List[BatchRequest]
    ) -> Tuple[torch.Tensor, BatchMetadata]:
        """
        Stack individual inputs into a batch tensor.
        For fixed-size inputs (e.g., images), this is simple stacking.
        For variable-length inputs (e.g., text), this requires padding.
        Args:
            batch: List of BatchRequest objects
        Returns:
            (batched_tensor, metadata_for_unbatching)
        """
        inputs = [req.input_tensor for req in batch]
        shapes = [t.shape for t in inputs]
        metadata = BatchMetadata(
            requests=batch,
            original_shapes=shapes,
            batch_size=len(batch)
        )
        # Check if all inputs have same shape (fixed-size)
        if len(set(shapes)) == 1:
            # Simple case: stack directly
            batched = torch.stack(inputs, dim=0)
            metadata.needs_unpadding = False
            return batched.to(self.device), metadata
        # Variable-length: need padding
        if self.model_input_type == "variable":
            padding_result = self.padder.pad_batch(inputs)
            metadata.padding_mask = padding_result.attention_mask
            metadata.needs_unpadding = True
            logger.debug(
                f"[Batcher] Padded batch: "
                f"overhead={padding_result.padding_overhead:.1%}"
            )
            return padding_result.padded_tensor.to(self.device), metadata
        else:
            raise PaddingError(
                "Variable-length inputs received but model_input_type='fixed'"
            )
    @torch.no_grad()
    async def _run_inference(
        self, 
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Run model inference on batched inputs.
        This runs in a thread executor to avoid blocking the event loop,
        since PyTorch inference can be CPU/GPU intensive.
        Args:
            inputs: Batched input tensor on correct device
        Returns:
            Batched output tensor on CPU
        """
        loop = asyncio.get_event_loop()
        def sync_inference():
            output = self.model_fn(inputs)
            return output.cpu()
        # Run in thread pool to not block async loop
        return await loop.run_in_executor(None, sync_inference)
    def _route_results(
        self, 
        outputs: torch.Tensor, 
        metadata: BatchMetadata,
        inference_ms: float
    ) -> None:
        """
        Split batch outputs and deliver to each requester.
        This is critical: each request must get its own result,
        not someone else's. Order must match input order.
        Args:
            outputs: Batched output tensor
            metadata: Batch metadata for routing
            inference_ms: Inference time for metrics
        """
        for i, request in enumerate(metadata.requests):
            try:
                # Get this request's output
                output_i = outputs[i]
                # Record latency metrics
                queue_wait_ms = request.wait_time_ms()
                total_ms = queue_wait_ms + inference_ms
                self.metrics.record_request(
                    queue_wait_ms=queue_wait_ms,
                    inference_ms=inference_ms,
                    total_ms=total_ms
                )
                # Deliver result
                request.set_result(output_i)
            except Exception as e:
                logger.error(
                    f"[Batcher] Failed to route result for "
                    f"request {request.request_id}: {e}"
                )
                request.set_error(
                    ResultRoutingError(request.request_id, str(e))
                )
    def _fail_current_batch(self, error: Exception) -> None:
        """Fail all requests in current batch with given error."""
        inference_error = InferenceError(
            batch_size=len(self._current_batch),
            original_error=error
        )
        for request in self._current_batch:
            if not request.is_resolved():
                request.set_error(inference_error)
        self._current_batch = []
        self._current_metadata = None
    def start(self) -> None:
        """Start the batch processing loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._batch_loop())
        logger.info("[Batcher] Started")
    async def stop(self) -> None:
        """
        Gracefully stop the batch processor.
        Drains remaining requests and fails them before stopping.
        """
        self._running = False
        self.queue.shutdown()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Drain remaining requests
        remaining = await self.queue.drain()
        for request in remaining:
            request.set_error(RuntimeError("Batcher shutting down"))
        logger.info("[Batcher] Stopped")
    def get_queue_depth(self) -> int:
        """Get current queue depth."""
        return self.queue.qsize()
    def get_metrics_summary(self) -> Dict[str, float]:
        """Get metrics summary."""
        summary = self.metrics.summary()
        summary["queue_depth"] = float(self.queue.qsize())
        summary["queue_max_depth"] = float(self.queue.max_depth)
        return summary
```
---
### ResultRouter
```python
# 07_result_router.py
import torch
from typing import List, Optional
import logging
from .03_batch_request import BatchRequest
from .06_batcher import BatchMetadata
logger = logging.getLogger(__name__)
class ResultRouter:
    """
    Splits batch outputs and routes to individual requesters.
    The critical invariant: each request gets exactly one result,
    and it must be the correct result (by batch index).
    Thread Safety: All operations are within the single-threaded
    batch loop, so no synchronization needed.
    """
    def route(
        self,
        outputs: torch.Tensor,
        metadata: BatchMetadata
    ) -> None:
        """
        Route outputs to individual requests.
        Args:
            outputs: Batched output tensor, shape (batch, ...)
            metadata: Batch metadata with request mapping
        Side effects:
            - Sets result on each request's Future
            - Logs any routing failures
        """
        batch_size = metadata.batch_size
        # Validate output shape
        if outputs.shape[0] != batch_size:
            logger.error(
                f"[Router] Output batch size {outputs.shape[0]} "
                f"doesn't match input batch size {batch_size}"
            )
            # Fail all requests
            for request in metadata.requests:
                request.set_error(
                    ResultRoutingError(
                        request.request_id,
                        f"Output shape mismatch: expected batch {batch_size}, "
                        f"got {outputs.shape[0]}"
                    )
                )
            return
        for i, request in enumerate(metadata.requests):
            try:
                output_i = outputs[i]
                # For sequence outputs, may need to trim
                if metadata.needs_unpadding and metadata.original_shapes:
                    original_seq_len = metadata.original_shapes[i][0]
                    if output_i.dim() > 0 and output_i.shape[0] > original_seq_len:
                        output_i = output_i[:original_seq_len]
                request.set_result(output_i)
            except Exception as e:
                logger.error(
                    f"[Router] Failed to route result for "
                    f"request {request.request_id}: {e}"
                )
                request.set_error(
                    ResultRoutingError(request.request_id, str(e))
                )
```
---
## Algorithm Specification
### Batch Formation Algorithm
```
ALGORITHM: get_batch
INPUT: max_batch_size (int), timeout_ms (float)
OUTPUT: List[BatchRequest] (may be empty or partial)
1. batch ← empty list
2. // Wait for first request with timeout
   TRY:
       first_request ← AWAIT queue.get() WITH timeout = timeout_ms / 1000
       first_request.status ← PROCESSING
       APPEND first_request TO batch
   EXCEPT TimeoutError:
       RETURN batch  // Empty batch
3. // Drain additional requests without waiting
   WHILE len(batch) < max_batch_size:
       TRY:
           request ← queue.get_nowait()
           request.status ← PROCESSING
           APPEND request TO batch
       EXCEPT QueueEmpty:
           BREAK
4. total_processed ← total_processed + len(batch)
   RETURN batch
INVARIANT: All returned requests have status PROCESSING
INVARIANT: No request appears more than once
COMPLEXITY: O(batch_size) for dequeue operations
```
### Padding Algorithm
```
ALGORITHM: pad_batch
INPUT: tensors (List[Tensor])
OUTPUT: PaddingResult (padded_tensor, attention_mask, ...)
1. IF tensors is empty:
       RAISE PaddingError("Empty tensor list")
2. // Determine shape type
   first_shape ← tensors[0].shape
   is_2d ← (len(first_shape) == 2)
3. // Validate all tensors have same rank
   FOR each tensor t WITH index i IN tensors:
       IF len(t.shape) != len(first_shape):
           RAISE PaddingError(f"Tensor {i} has incompatible rank")
4. // Calculate target dimensions
   lengths ← [t.shape[0] for t in tensors]
   max_len ← max(lengths)
   IF max_length is set:
       max_len ← min(max_len, max_length)
   batch_size ← len(tensors)
5. // Create output tensors
   IF is_2d:
       feature_dim ← first_shape[1]
       padded ← full((batch_size, max_len, feature_dim), pad_value)
   ELSE:
       padded ← full((batch_size, max_len), pad_value)
   attention_mask ← zeros((batch_size, max_len))
6. // Fill with actual values
   FOR each tensor t WITH index i IN tensors:
       seq_len ← min(t.shape[0], max_len)
       IF is_2d:
           padded[i, :seq_len, :] ← t[:seq_len]
       ELSE:
           padded[i, :seq_len] ← t[:seq_len]
       attention_mask[i, :seq_len] ← 1
7. // Calculate overhead
   total_real ← sum(min(l, max_len) for l in lengths)
   total_padded ← max_len * batch_size
   overhead ← 1.0 - (total_real / total_padded)
8. RETURN PaddingResult(padded, attention_mask, lengths, overhead)
INVARIANT: attention_mask[i, :lengths[i]] == 1 for all i
INVARIANT: attention_mask[i, lengths[i]:] == 0 for all i
```
### Result Routing Algorithm
```
ALGORITHM: route_results
INPUT: outputs (Tensor), metadata (BatchMetadata)
OUTPUT: None (side effects: Futures resolved)
1. batch_size ← metadata.batch_size
2. // Validate shape
   IF outputs.shape[0] != batch_size:
       LOG ERROR "Output batch size mismatch"
       FOR each request IN metadata.requests:
           request.set_error(ResultRoutingError)
       RETURN
3. FOR each (request, original_shape) IN ZIP(metadata.requests, metadata.original_shapes):
       output_i ← outputs[i]  // Get by batch index
       // Trim if variable-length and output has sequence dimension
       IF metadata.needs_unpadding:
           original_seq_len ← original_shape[0]
           IF output_i.shape[0] > original_seq_len:
               output_i ← output_i[:original_seq_len]
       request.set_result(output_i)
INVARIANT: Each request receives exactly one result
INVARIANT: Request at index i receives output[i]
```
---
## Error Handling Matrix
| Error | Detected By | Recovery | User-Visible? |
|-------|-------------|----------|---------------|
| `BackpressureRejection` | `RequestQueue.enqueue()` | Return HTTP 503, client retries | Yes (503 "Service overloaded") |
| `RequestTimeout` | HTTP handler `wait_for` | Return HTTP 504, no retry recommended | Yes (504 "Request timeout") |
| `PaddingError` | `VariableLengthPadder.pad_batch()` | Fail all requests in batch, continue loop | Yes (500 "Batch processing error") |
| `InferenceError` | `DynamicBatcher._run_inference()` catch | Fail all requests in batch, continue loop | Yes (500 "Inference failed") |
| `ResultRoutingError` | `ResultRouter.route()` catch | Fail individual request, log error | Yes (500 "Result routing failed") |
| `Queue shutdown` | `RequestQueue.enqueue()` check | Return error to submitter | Yes (503 "Service shutting down") |
| `Batch loop crash` | Top-level try/except in loop | Log error, sleep 100ms, continue | Indirectly (through latency) |
---
## Implementation Sequence with Checkpoints
### Phase 1: RequestQueue with Backpressure (2-3 hours)
**Files to create:** `01_config.py`, `02_exceptions.py`, `03_batch_request.py`, `04_request_queue.py`
**Implementation steps:**
1. Define `BatcherConfig` with validation
2. Define `RequestStatus` enum and `BatchRequest` dataclass with Future
3. Define exception hierarchy
4. Implement `RequestQueue` with `enqueue()` and `get_batch()`
5. Add backpressure check in `enqueue()`
6. Add shutdown support
**Checkpoint:**
```bash
python -c "
import asyncio
from request_queue import RequestQueue
from batch_request import BatchRequest
async def test():
    queue = RequestQueue(max_depth=5)
    # Test enqueue
    req = BatchRequest()
    success = await queue.enqueue(req)
    assert success, 'Enqueue should succeed'
    print(f'Queue depth: {queue.qsize()}')
    # Test get_batch
    batch = await queue.get_batch(max_batch_size=10, timeout_ms=100)
    assert len(batch) == 1, 'Should get 1 request'
    print(f'Got batch of size {len(batch)}')
    # Test backpressure
    queue.max_depth = 2
    for _ in range(3):
        req = BatchRequest()
        success = await queue.enqueue(req)
        print(f'Enqueue success: {success}')
    print(f'Rejection rate: {queue.metrics().rejection_rate:.2%}')
asyncio.run(test())
"
```
Expected: Enqueue succeeds until capacity, then rejected.
---
### Phase 2: Variable-Length Padding (2-3 hours)
**Files to create:** `05_padder.py`
**Implementation steps:**
1. Implement `VariableLengthPadder` class
2. Implement `pad_batch()` with shape validation
3. Calculate and return attention mask
4. Calculate padding overhead
5. Add max_length truncation support
**Checkpoint:**
```bash
python -c "
import torch
from padder import VariableLengthPadder
padder = VariableLengthPadder(pad_value=0, max_length=512)
# Test with variable-length sequences
tensors = [
    torch.tensor([1, 2, 3]),
    torch.tensor([4, 5, 6, 7, 8]),
    torch.tensor([9, 10]),
]
result = padder.pad_batch(tensors)
print(f'Padded shape: {result.padded_tensor.shape}')
print(f'Mask shape: {result.attention_mask.shape}')
print(f'Original lengths: {result.original_lengths}')
print(f'Padding overhead: {result.padding_overhead:.1%}')
# Verify masks
print(f'Mask row 0: {result.attention_mask[0].tolist()}')
print(f'Mask row 1: {result.attention_mask[1].tolist()}')
"
```
Expected: Padded to length 5, masks show real vs padding tokens.
---
### Phase 3: DynamicBatcher Core (3-4 hours)
**Files to create:** `06_batcher.py`
**Implementation steps:**
1. Implement `DynamicBatcher.__init__()` with all components
2. Implement `submit()` with queue integration
3. Implement `_batch_loop()` with timeout-based formation
4. Implement `_prepare_batch_inputs()` for fixed-size
5. Add start/stop lifecycle methods
**Checkpoint:**
```bash
python -c "
import asyncio
import torch
from batcher import DynamicBatcher
from config import BatcherConfig
async def test():
    config = BatcherConfig(max_batch_size=4, timeout_ms=100)
    # Simple model function
    def model_fn(x):
        return x * 2
    batcher = DynamicBatcher(
        config=config,
        model_fn=model_fn,
        device=torch.device('cpu'),
        model_input_type='fixed'
    )
    batcher.start()
    # Submit some requests
    async def make_request(val):
        tensor = torch.tensor([[val]])
        result = await batcher.submit(tensor)
        return result
    # Run concurrent requests
    results = await asyncio.gather(
        make_request(1.0),
        make_request(2.0),
        make_request(3.0),
    )
    print(f'Results: {[r.item() for r in results]}')
    print(f'Metrics: {batcher.get_metrics_summary()}')
    await batcher.stop()
asyncio.run(test())
"
```
Expected: Results are doubled, metrics show batch processing.
---
### Phase 4: Result Routing (2-3 hours)
**Files to create:** `07_result_router.py`
**Implementation steps:**
1. Implement `ResultRouter` class
2. Implement `route()` with index-based splitting
3. Add variable-length trimming support
4. Handle shape mismatch errors
**Checkpoint:**
```bash
python -c "
import torch
from result_router import ResultRouter
from batcher import BatchMetadata
from batch_request import BatchRequest
router = ResultRouter()
# Create test batch
requests = [BatchRequest() for _ in range(3)]
metadata = BatchMetadata(requests=requests, batch_size=3)
# Create test outputs
outputs = torch.tensor([[1.0], [2.0], [3.0]])
# Route results
router.route(outputs, metadata)
# Check each request got its result
for i, req in enumerate(requests):
    result = req.future.result()
    print(f'Request {i} result: {result.item()}')
    assert result.item() == float(i + 1)
"
```
Expected: Each request gets its indexed result.
---
### Phase 5: Metrics Collection (2-3 hours)
**Files to create:** `08_metrics.py`
**Implementation steps:**
1. Implement `BatcherMetrics` dataclass with deques
2. Implement `percentile()` calculation
3. Implement `summary()` for Prometheus export
4. Integrate with batcher for request/batch recording
**Checkpoint:**
```bash
python -c "
from metrics import BatcherMetrics
metrics = BatcherMetrics(window_size=100)
# Record some data
for i in range(50):
    metrics.record_request(
        queue_wait_ms=10 + i * 0.5,
        inference_ms=20 + i * 0.2,
        total_ms=30 + i * 0.7
    )
    metrics.record_batch(batch_size=min(i + 1, 16))
summary = metrics.summary()
print(f'Total requests: {summary[\"batcher_requests_total\"]:.0f}')
print(f'Avg batch size: {summary[\"batcher_avg_batch_size\"]:.1f}')
print(f'Total p50: {summary[\"batcher_total_ms_p50\"]:.1f}ms')
print(f'Total p95: {summary[\"batcher_total_ms_p95\"]:.1f}ms')
print(f'RPS: {summary[\"batcher_rps\"]:.1f}')
"
```
Expected: Percentiles calculated, RPS computed.
---
### Phase 6: Throughput Benchmark (2-3 hours)
**Files to create:** `09_benchmark.py`
**Implementation steps:**
1. Implement sequential serving baseline
2. Implement batched serving comparison
3. Run with concurrent requests
4. Calculate speedup ratio
5. Verify 2x+ improvement
**Checkpoint:**
```bash
python -c "
import asyncio
import torch
import time
from benchmark import run_comparison_benchmark
async def test():
    # Simple model
    def model_fn(x):
        # Simulate some compute
        return torch.matmul(x, torch.randn(100, 100))
    results = await run_comparison_benchmark(
        model_fn=model_fn,
        device=torch.device('cpu'),
        n_requests=100,
        max_batch_size=16,
        timeout_ms=50.0
    )
    print(f'Sequential RPS: {results[\"sequential_rps\"]:.1f}')
    print(f'Batched RPS: {results[\"batched_rps\"]:.1f}')
    print(f'Speedup: {results[\"speedup\"]:.1f}x')
    print(f'Pass: {results[\"pass\"]}')
asyncio.run(test())
"
```
Expected: Speedup >= 2.0x (may be lower on CPU vs GPU).
---
### Phase 7: FastAPI Integration (2-3 hours)
**Files to create:** `10_integration.py`
**Implementation steps:**
1. Create FastAPI app with batching
2. Integrate with model loader from M1
3. Add `/predict` endpoint with batching
4. Add `/metrics` endpoint
5. Add health probes
**Checkpoint:**
```bash
# Start server
uvicorn integration:app --host 0.0.0.0 --port 8000 &
# Wait for startup
sleep 5
# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [[0.1, 0.2, 0.3]]}'
# Test metrics
curl http://localhost:8000/metrics
# Kill server
pkill -f uvicorn
```
Expected: Predictions returned, metrics exposed.
---
## Test Specification
### Test 1: Queue Backpressure
```python
@pytest.mark.asyncio
async def test_queue_backpressure():
    """Test that queue rejects requests when at capacity."""
    from request_queue import RequestQueue
    from batch_request import BatchRequest
    queue = RequestQueue(max_depth=3)
    # Fill queue
    for _ in range(3):
        req = BatchRequest()
        success = await queue.enqueue(req)
        assert success
    # Next should be rejected
    req = BatchRequest()
    success = await queue.enqueue(req)
    assert not success
    assert queue.metrics().rejection_rate > 0
```
### Test 2: Batch Formation with Timeout
```python
@pytest.mark.asyncio
async def test_batch_timeout():
    """Test that partial batch is returned on timeout."""
    from request_queue import RequestQueue
    from batch_request import BatchRequest
    queue = RequestQueue(max_depth=10)
    # Add 2 requests
    for _ in range(2):
        req = BatchRequest()
        await queue.enqueue(req)
    # Get batch with timeout
    batch = await queue.get_batch(max_batch_size=10, timeout_ms=100)
    # Should get 2, not wait for 10
    assert len(batch) == 2
```
### Test 3: Variable-Length Padding
```python
def test_variable_length_padding():
    """Test padding of variable-length sequences."""
    from padder import VariableLengthPadder
    import torch
    padder = VariableLengthPadder(pad_value=0)
    tensors = [
        torch.tensor([1, 2, 3]),
        torch.tensor([4, 5]),
        torch.tensor([6, 7, 8, 9]),
    ]
    result = padder.pad_batch(tensors)
    # Should pad to max length (4)
    assert result.padded_tensor.shape == (3, 4)
    # Check attention masks
    assert result.attention_mask[0].tolist() == [1, 1, 1, 0]
    assert result.attention_mask[1].tolist() == [1, 1, 0, 0]
    assert result.attention_mask[2].tolist() == [1, 1, 1, 1]
```
### Test 4: Result Routing
```python
def test_result_routing():
    """Test that results are routed to correct requesters."""
    from result_router import ResultRouter
    from batcher import BatchMetadata
    from batch_request import BatchRequest
    import torch
    router = ResultRouter()
    requests = [BatchRequest() for _ in range(4)]
    metadata = BatchMetadata(requests=requests, batch_size=4)
    outputs = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
    router.route(outputs, metadata)
    # Verify each request got correct result
    for i, req in enumerate(requests):
        result = req.future.result()
        assert torch.isclose(result, torch.tensor([0.1 * (i + 1)]))
```
### Test 5: Throughput Improvement
```python
@pytest.mark.asyncio
async def test_throughput_improvement():
    """Test that batching improves throughput."""
    from benchmark import run_comparison_benchmark
    import torch
    def model_fn(x):
        # Simulate compute
        return torch.matmul(x, torch.randn(x.shape[1], 100))
    results = await run_comparison_benchmark(
        model_fn=model_fn,
        device=torch.device('cpu'),
        n_requests=50,
        max_batch_size=8,
        timeout_ms=50.0
    )
    # Should achieve at least 1.5x on CPU (2x on GPU)
    assert results['speedup'] >= 1.5
```
### Test 6: Metrics Accuracy
```python
def test_metrics_percentiles():
    """Test that percentile calculation is accurate."""
    from metrics import BatcherMetrics
    metrics = BatcherMetrics(window_size=1000)
    # Record known values
    for i in range(100):
        metrics.record_request(
            queue_wait_ms=float(i),
            inference_ms=10.0,
            total_ms=float(i) + 10.0
        )
    summary = metrics.summary()
    # p50 should be around 49-50
    assert 45 <= summary['batcher_queue_wait_ms_p50'] <= 55
    # p99 should be around 98-99
    assert 95 <= summary['batcher_queue_wait_ms_p99'] <= 100
```
---
## Performance Targets
| Operation | Target | How to Measure |
|-----------|--------|----------------|
| Throughput improvement | >= 2x vs sequential | `run_comparison_benchmark()` on GPU |
| Worst-case latency | < timeout_ms + inference_time | Monitor p99 total_ms in metrics |
| GPU utilization | > 70% at sustained load | `nvidia-smi` during benchmark with batch >= 16 |
| Queue depth alert | 80% of max_depth | Check metrics when queue fills |
| Padding overhead | < 30% for text | Check `padding_overhead` in PaddingResult |
| Batch formation | < 1ms overhead | Time from batch start to inference start |
| Result routing | < 0.5ms per request | Time to split and deliver results |
---
## State Machine: Request Lifecycle
```
                    ┌─────────────┐
                    │   CREATED   │
                    └──────┬──────┘
                           │ submit()
                           ▼
              ┌────────────────────────┐
              │        QUEUED          │
              │  (in asyncio.Queue)    │
              └───────────┬────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
    queue full       get_batch()     queue shutdown
          │               │               │
          ▼               ▼               ▼
   ┌────────────┐  ┌─────────────┐  ┌────────────┐
   │  REJECTED  │  │ PROCESSING  │  │   FAILED   │
   │  (503)     │  │ (in batch)  │  │  (shutdown)│
   └────────────┘  └──────┬──────┘  └────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
    inference success              inference error
          │                               │
          ▼                               ▼
   ┌─────────────┐                 ┌────────────┐
   │  COMPLETED  │                 │   FAILED   │
   │ (result set)│                 │ (exception)│
   └─────────────┘                 └────────────┘
ILLEGAL TRANSITIONS:
- COMPLETED → any state (terminal)
- FAILED → any state (terminal)
- QUEUED → COMPLETED (must go through PROCESSING)
- PROCESSING → QUEUED (no re-queueing)
```
---
## Concurrency Specification
### Threading Model
```
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Event Loop                          │
│                     (Single-threaded async)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  /predict    │  │  /metrics    │  │  batch_loop  │          │
│  │  handlers    │  │  handler     │  │  (task)      │          │
│  └──────┬───────┘  └──────────────┘  └──────┬───────┘          │
│         │                                     │                  │
│         │ submit()                            │ get_batch()      │
│         ▼                                     ▼                  │
│  ┌──────────────────────────────────────────────────┐          │
│  │              RequestQueue (asyncio.Queue)         │          │
│  │  - Thread-safe for cooperative multitasking       │          │
│  │  - FIFO ordering guaranteed                       │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                 │
│  ┌──────────────────────────────────────────────────┐          │
│  │              BatcherMetrics (shared)              │          │
│  │  - Deques are not thread-safe                     │          │
│  │  - Safe within asyncio cooperative model          │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
### Thread Safety Guarantees
| Component | Thread Safety | Reason |
|-----------|--------------|--------|
| `RequestQueue` | Safe for async | Built on asyncio.Queue |
| `BatchRequest.future` | Safe | asyncio.Future is thread-safe |
| `BatcherMetrics` | Safe within asyncio | Deques modified in single task |
| `torch.Tensor` operations | Safe | PyTorch tensors are immutable-ish |
| `_run_inference` in executor | Safe | Creates new tensors, no shared mutation |
### Lock Ordering (if threading introduced)
If multi-threading is added in the future:
1. Acquire `queue_lock` before accessing queue
2. Acquire `metrics_lock` before modifying metrics
3. Never hold both locks across await points
4. Inference executor should not hold any locks
---
## Diagrams
### 
![Dynamic Batcher Architecture](./diagrams/tdd-diag-009.svg)
 - GPU Utilization: Sequential vs Batched
```
Sequential (1 request at a time):
┌─────────────────────────────────────────────────────────────┐
│ Timeline (ms)  0   10   20   30   40   50   60   70   80   │
├─────────────────────────────────────────────────────────────┤
│ Request 1      [====inference====]                         │
│ Request 2                    [====inference====]           │
│ Request 3                                [====inference===]│
│                                                             │
│ GPU Usage      ████              ████             ████     │
│                (20%)             (20%)            (20%)     │
└─────────────────────────────────────────────────────────────┘
Throughput: ~17 req/s, GPU Utilization: ~20%
Batched (4 requests together):
┌─────────────────────────────────────────────────────────────┐
│ Timeline (ms)  0   10   20   30   40   50   60   70   80   │
├─────────────────────────────────────────────────────────────┤
│ Batch 1        [========batch inference========]            │
│ Batch 2                                    [========batch===│
│                                                             │
│ GPU Usage      ████████████████████████████████            │
│                (85% utilization sustained)                  │
└─────────────────────────────────────────────────────────────┘
Throughput: ~50 req/s, GPU Utilization: ~85%
Speedup: ~3x
```
### 
![Batch Formation Timeline](./diagrams/tdd-diag-010.svg)
 - Latency-Throughput Pareto Frontier
```
Throughput (req/s)
    │
600 ┤                                    ●
    │                                 ●
500 ┤                              ●
    │                           ●
400 ┤                        ●
    │                     ●
300 ┤                  ●
    │               ●
200 ┤            ●
    │         ●
100 ┤      ●
    │   ●
 50 ┤●
    │
    └────────────────────────────────────────────
      10   20   30   50   80  120  180  250  350  Latency (ms)
      Batch sizes: 1    2    4    8   16   32   64  128  256
      Optimal region: batch_size 16-32
      - Throughput: 400-500 req/s
      - Latency: 50-80ms
      - GPU utilization: >80%
```
### 
![GPU Utilization: Sequential vs Batched](./diagrams/tdd-diag-011.svg)
 - Dynamic Batcher Architecture
```
┌─────────────────────────────────────────────────────────────────────┐
│                         HTTP Layer                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                          │
│  │ Handler 1│  │ Handler 2│  │ Handler 3│  (concurrent requests)    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                          │
│       │             │             │                                  │
│       │ submit()    │ submit()    │ submit()                        │
│       ▼             ▼             ▼                                  │
├─────────────────────────────────────────────────────────────────────┤
│                       RequestQueue                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  [Req1] → [Req2] → [Req3] → [Req4] → [Req5] → [Req6] → ... │   │
│  │                                                              │   │
│  │  Backpressure: reject if depth >= max_depth                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ get_batch(max_size, timeout)
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DynamicBatcher (Background Task)                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Batch Formation Loop                      │   │
│  │                                                              │   │
│  │  1. Wait for first request (timeout_ms)                      │   │
│  │  2. Drain up to max_batch_size requests                      │   │
│  │  3. Pad variable-length inputs                               │   │
│  │  4. Run batch inference                                       │   │
│  │  5. Split and route results                                   │   │
│  │  6. Repeat                                                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Batch Inference                           │   │
│  │                                                              │   │
│  │  Input:  (batch, channels, height, width)                    │   │
│  │          or (batch, seq_len) with attention_mask             │   │
│  │                                                              │   │
│  │  Output: (batch, num_classes)                                │   │
│  │          or (batch, hidden_dim)                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │ Route results by index
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Future Resolution                             │
│                                                                      │
│  Handler 1 awaits ──► Future 1 ◄── Result[0] (resolved)            │
│  Handler 2 awaits ──► Future 2 ◄── Result[1] (resolved)            │
│  Handler 3 awaits ──► Future 3 ◄── Result[2] (resolved)            │
└─────────────────────────────────────────────────────────────────────┘
```
### 
![Latency-Throughput Pareto Frontier](./diagrams/tdd-diag-012.svg)
 - Batch Formation Timeline
```
Timeline showing batch formation with timeout:
Time (ms)     0     10    20    30    40    50    60    70    80
              │     │     │     │     │     │     │     │     │
              ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼     ▼
Request 1     ●──────────────────────────────────────────────────►
arrives       (t=5ms)
Request 2           ●────────────────────────────────────────────►
arrives             (t=15ms)
Request 3                 ●──────────────────────────────────────►
arrives                   (t=25ms)
Batch Formation:
              │◄─────── wait for first ───────►│
              │     (timeout=50ms)             │
              │                                │
              │  Batch formed at t=55ms:       │
              │  [Req1, Req2, Req3]            │
              │                                │
              │                          ◄─────┼─────► Inference
              │                                │      (25ms)
              │                                │
              │                                ▼
              │                          Results routed
              │                          at t=80ms
              │
              ▼
If no request arrives within timeout:
              │◄────── timeout (50ms) ──────►│
              │                              │
              │                        Empty batch returned
              │                        (loop continues)
```
### 
![Variable-Length Input Padding](./diagrams/tdd-diag-013.svg)
 - Variable-Length Padding
```
Variable-length sequences before padding:
Request 1: [1, 2, 3]                    (length=3)
Request 2: [4, 5, 6, 7, 8]              (length=5)
Request 3: [9, 10]                      (length=2)
After padding to max_length=5:
Padded Tensor:                          Attention Mask:
┌─────────────────────┐                ┌───────────────┐
│ 1  2  3  0  0       │                │ 1  1  1  0  0 │
├─────────────────────┤                ├───────────────┤
│ 4  5  6  7  8       │                │ 1  1  1  1  1 │
├─────────────────────┤                ├───────────────┤
│ 9 10  0  0  0       │                │ 1  1  0  0  0 │
└─────────────────────┘                └───────────────┘
Shape: (3, 5)                          Shape: (3, 5)
Padding Overhead:
  Total real tokens: 3 + 5 + 2 = 10
  Total padded: 5 × 3 = 15
  Overhead: 1 - (10/15) = 33.3%
Model attention with mask:
Attention scores (before softmax):
┌─────────────────────────────────────┐
│ Q₁·K₁  Q₁·K₂  Q₁·K₃   -∞    -∞    │  ← Position 3,4 masked
├─────────────────────────────────────┤
│ Q₂·K₁  Q₂·K₂  Q₂·K₃  Q₂·K₄  Q₂·K₅ │  ← All positions attend
├─────────────────────────────────────┤
│ Q₃·K₁  Q₃·K₂   -∞     -∞    -∞    │  ← Positions 2,3,4 masked
└─────────────────────────────────────┘
After softmax, -∞ positions get 0 weight.
Padding tokens are "invisible" to the model.
```
### 
![Request Queue State Machine](./diagrams/tdd-diag-014.svg)
 - Backpressure Flow Control
```
Request Flow with Backpressure:
                    Incoming Requests
                           │
                           ▼
              ┌────────────────────────┐
              │   Current Queue Depth   │
              │                        │
              │   depth < max_depth?   │
              │                        │
              └───────────┬────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
           [YES]                    [NO]
              │                       │
              ▼                       ▼
        ┌──────────┐           ┌──────────┐
        │ Enqueue  │           │ REJECT   │
        │ request  │           │ HTTP 503 │
        └────┬─────┘           └──────────┘
             │                       │
             ▼                       ▼
      ┌─────────────┐         Client retries
      │  In Queue   │         with backoff
      │  (waiting)  │
      └──────┬──────┘
             │
             │ get_batch()
             ▼
      ┌─────────────┐
      │ Processing  │
      │ (in batch)  │
      └──────┬──────┘
             │
             │ Result routed
             ▼
      ┌─────────────┐
      │  Completed  │
      │  (response) │
      └─────────────┘
Queue Depth Alerting:
              max_depth = 1000
              ┌────────────────────────────────────────┐
              │                                        │
              │ ████████████████████████████████  80% │ ← Alert threshold
              │                                        │
              │ ████████████████████████████████████  │
              │ ████████████████████████████████████  │
              │ ████████████████████████████████████  │ ← Reject threshold
              │ ████████████████████████████████████  │
              └────────────────────────────────────────┘
              Alert at 80% (800 requests)
              Reject at 100% (1000 requests)
```
### 
![Backpressure Flow Control](./diagrams/tdd-diag-015.svg)
 - Result Routing by Index
```
Batch Input → Batch Inference → Individual Results
Input Batch:
┌────────────────────────────────────────────┐
│ Request 0: tensor([[1.0, 2.0, 3.0]])      │
│ Request 1: tensor([[4.0, 5.0, 6.0]])      │
│ Request 2: tensor([[7.0, 8.0, 9.0]])      │
└────────────────────────────────────────────┘
                    │
                    │ Stack into batch tensor
                    ▼
┌────────────────────────────────────────────┐
│ Batched Input:                             │
│ tensor([[[1.0, 2.0, 3.0]],                 │
│         [[4.0, 5.0, 6.0]],                 │
│         [[7.0, 8.0, 9.0]]])                │
│                                            │
│ Shape: (3, 1, 3)                           │
└────────────────────────────────────────────┘
                    │
                    │ model(batched_input)
                    ▼
┌────────────────────────────────────────────┐
│ Batched Output:                            │
│ tensor([[[0.1, 0.9]],                      │
│         [[0.4, 0.6]],                      │
│         [[0.7, 0.3]]])                     │
│                                            │
│ Shape: (3, 1, 2)                           │
└────────────────────────────────────────────┘
                    │
                    │ Split by index
                    ▼
┌────────────────────────────────────────────┐
│ Request 0 receives: tensor([[0.1, 0.9]])  │
│ Request 1 receives: tensor([[0.4, 0.6]])  │
│ Request 2 receives: tensor([[0.7, 0.3]])  │
└────────────────────────────────────────────┘
                    │
                    │ Future.set_result()
                    ▼
┌────────────────────────────────────────────┐
│ Handler 0: await future → [[0.1, 0.9]]    │
│ Handler 1: await future → [[0.4, 0.6]]    │
│ Handler 2: await future → [[0.7, 0.3]]    │
└────────────────────────────────────────────┘
CRITICAL INVARIANT:
Request at index i MUST receive output[i]
Wrong routing = data leakage + wrong predictions
```
### 
![Result Routing After Batch Inference](./diagrams/tdd-diag-016.svg)
 - Metrics Sliding Window
```
BatcherMetrics with Sliding Windows:
Time Window (last 1000 samples):
              Now
               │
               ▼
  ┌────────────────────────────────────────────────────────┐
  │ ████████████████████████████████████████████████████  │
  │                                                        │
  │  Latency samples (ms):                                 │
  │  [45, 48, 52, 47, 49, ..., 51, 46, 50, 48, 52]       │
  │                                                        │
  │  Sorted: [45, 46, 47, 48, 48, ..., 51, 52, 52, 52]   │
  │                                                        │
  │  Percentiles:                                          │
  │  p50 (index 500):  48ms                                │
  │  p90 (index 900):  51ms                                │
  │  p95 (index 950):  52ms                                │
  │  p99 (index 990):  52ms                                │
  │                                                        │
  └────────────────────────────────────────────────────────┘
Window Management:
  New sample arrives:
  Before: [s₁, s₂, s₃, ..., s₁₀₀₀]  (1000 samples)
  After:  [s₂, s₃, s₄, ..., s₁₀₀₁]  (oldest evicted)
Metrics Summary Output (Prometheus format):
  batcher_requests_total 15734
  batcher_batches_total 892
  batcher_rejected_total 12
  batcher_rps 245.5
  batcher_queue_wait_ms_p50 12.3
  batcher_queue_wait_ms_p95 28.7
  batcher_queue_wait_ms_p99 45.2
  batcher_inference_ms_p50 35.1
  batcher_inference_ms_p95 42.8
  batcher_inference_ms_p99 58.3
  batcher_avg_batch_size 17.6
  batcher_rejection_rate 0.0008
```
### 
![BatcherMetrics Sliding Window](./diagrams/tdd-diag-017.svg)
 - Throughput Benchmark Comparison
```
Benchmark Setup:
- 100 requests
- Model: matrix multiplication (100×100)
- Device: CPU
Sequential Serving:
┌─────────────────────────────────────────────────────────────────┐
│ Request 1: [====]                                               │
│ Request 2:      [====]                                          │
│ Request 3:           [====]                                     │
│ Request 4:                [====]                                │
│ ...                                                             │
│ Request 100:                              [====]                │
│                                                                 │
│ Total time: 4.2 seconds                                         │
│ Throughput: 23.8 RPS                                            │
└─────────────────────────────────────────────────────────────────┘
Batched Serving (batch_size=16):
┌─────────────────────────────────────────────────────────────────┐
│ Batch 1 (16 reqs): [========]                                   │
│ Batch 2 (16 reqs):        [========]                            │
│ Batch 3 (16 reqs):                [========]                    │
│ Batch 4 (16 reqs):                        [========]            │
│ Batch 5 (16 reqs):                                [========]    │
│ ...                                                             │
│                                                                 │
│ Total time: 1.4 seconds                                         │
│ Throughput: 71.4 RPS                                            │
└─────────────────────────────────────────────────────────────────┘
Results:
┌─────────────────────────────────────────────────────────────┐
│                   THROUGHPUT COMPARISON                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Sequential:  23.8 RPS  ████████████████                    │
│                                                             │
│  Batched:     71.4 RPS  ████████████████████████████████████│
│                           (3.0x improvement)                │
│                                                             │
│  Target:      >= 2.0x   ✓ PASS                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
On GPU with larger batches (batch_size=32):
┌─────────────────────────────────────────────────────────────┐
│                   GPU BENCHMARK                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Sequential:   45 RPS   ████████████████                    │
│                                                             │
│  Batched:     180 RPS   ████████████████████████████████████│
│                           (4.0x improvement)                │
│                                                             │
│  GPU Utilization:                                          │
│  Sequential: 15%                                           │
│  Batched:    82%                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
---
[[CRITERIA_JSON: {"module_id": "ml-model-serving-m2", "criteria": ["Incoming requests are queued via asyncio.Queue with RequestQueue wrapper and grouped into batches up to configurable max_batch_size (default 32)", "Maximum wait timeout (default 50ms) flushes partial batches using asyncio.wait_for for first request then draining queue with get_nowait to bound worst-case latency", "Variable-length inputs are padded to longest sequence in batch with attention masks (1 for real tokens, 0 for padding) via VariableLengthPadder", "Batch inference results are correctly un-batched by index and routed to each original requester via asyncio.Future resolution in ResultRouter", "Backpressure rejects new requests with HTTP 503 when queue depth exceeds configurable max_queue_depth limit in RequestQueue.enqueue()", "Metrics exposed include: average batch size, queue depth, queue wait time, inference latency percentiles (p50/p95/p99), and requests per second via BatcherMetrics", "Throughput benchmark demonstrates at least 2x RPS improvement for batched concurrent serving vs sequential serving on GPU", "Batch formation loop runs as background asyncio task independent of HTTP request handlers via DynamicBatcher._batch_loop()", "Input tensor shape validation handles both fixed-size and variable-length inputs with proper error messages for shape mismatches", "Request lifecycle tracked from arrival through queue wait, inference, and response delivery for latency breakdown in metrics", "BatchRequest dataclass includes request_id, input_tensor, arrival_time, future, status with set_result() and set_error() methods", "BatcherConfig validates max_batch_size >= 1, timeout_ms > 0, max_queue_depth >= 1 with ValueError on invalid config", "PaddingResult includes padded_tensor, attention_mask, original_lengths, and padding_overhead calculation", "DynamicBatcher._run_inference() runs in thread executor via loop.run_in_executor() to avoid blocking event loop", "RequestStatus enum tracks QUEUED, PROCESSING, COMPLETED, FAILED states with proper state transitions"]}]
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: ml-model-serving-m3 -->
# Model Versioning & Hot Reload
## Module Charter
This module implements multi-version model management with zero-downtime hot swapping, enabling production deployments without service interruption. It maintains a model registry storing version metadata (accuracy, training date, schema definitions), performs atomic pointer swaps to switch traffic between versions instantaneously, tracks in-flight requests per version to ensure complete draining before unloading, and provides instant rollback by keeping the previous version in memory. The module validates schema compatibility between versions before allowing promotion, rejecting breaking changes to input/output shapes or dtypes. It explicitly does NOT handle A/B traffic splitting (M4), statistical analysis of version performance, or drift detection (M5). Upstream dependencies are the model loader from M1 and HTTP routing layer; downstream consumers are the inference endpoints and monitoring systems. The key invariants are: (1) every request sees exactly one consistent model version from start to finish, (2) no version is unloaded while it has in-flight requests, (3) hot swap is atomic—all requests after the swap see the new version, (4) rollback completes within 5 seconds by keeping previous version in memory.
---
## File Structure
Create files in this exact sequence:
```
ml-model-serving/m3_versioning/
├── 01_config.py              # Versioning configuration constants
├── 02_exceptions.py          # Versioning-specific exception hierarchy
├── 03_schema.py              # ModelSchema and compatibility validation
├── 04_model_version.py       # ModelVersion dataclass with metadata
├── 05_model_registry.py      # ModelRegistry with version storage
├── 06_atomic_pointer.py      # AtomicVersionPointer for thread-safe swaps
├── 07_request_tracker.py     # RequestTracker with ref counting and drain
├── 08_hot_swap_manager.py    # HotSwapManager orchestrating swap sequence
├── 09_rollback_monitor.py    # RollbackMonitor with error rate tracking
├── 10_version_router.py      # Version-specific routing logic
└── 11_integration.py         # FastAPI integration with versioned endpoints
```
---
## Complete Data Model
### VersionStatus Enum
The lifecycle states of a model version:
```python
# 02_exceptions.py
from enum import Enum
class VersionStatus(Enum):
    """
    Lifecycle states for a model version.
    State transitions:
        STAGED → ACTIVE (on hot swap)
        ACTIVE → DRAINING (when replaced by new version)
        DRAINING → DEPRECATED (when in-flight count reaches zero)
        Any → FAILED (on load error or validation failure)
    """
    STAGED = "staged"          # Loaded in memory, not serving traffic
    ACTIVE = "active"          # Currently serving traffic
    DRAINING = "draining"      # Was active, waiting for in-flight requests
    DEPRECATED = "deprecated"  # No longer in memory, metadata retained
    FAILED = "failed"          # Failed to load or validate
```
**Why this enum exists:** The registry needs to track version lifecycle for proper draining and garbage collection. The DRAINING state is critical—it's the window between swapping the pointer and unloading the model, where in-flight requests must complete.
---
### ModelSchema Dataclass
Defines the API contract for a model version:
```python
# 03_schema.py
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List, Union
import torch
@dataclass(frozen=True)
class ModelSchema:
    """
    Input/output schema definition for a model version.
    This is the API contract—breaking changes are deployment blockers.
    The schema enables:
    1. Client compatibility checking
    2. Automatic validation of incoming requests
    3. Breaking change detection during hot swap
    Shape notation:
        - (None, 3, 224, 224): Variable batch, fixed image size
        - (None, None): Variable batch and sequence length
        - (1, 1000): Fixed shape (batch=1, classes=1000)
    Invariants:
        - All input names must have corresponding shapes and dtypes
        - All output names must have corresponding shapes and dtypes
        - None in shape indicates variable dimension
        - dtype strings must be valid torch.dtype names
    """
    input_shapes: Dict[str, Tuple[Union[int, None], ...]]
    input_dtypes: Dict[str, str]
    output_shapes: Dict[str, Tuple[Union[int, None], ...]]
    output_dtypes: Dict[str, str]
    def to_dict(self) -> dict:
        """Serialize schema to JSON-compatible dict."""
        return {
            "input_shapes": {k: list(v) for k, v in self.input_shapes.items()},
            "input_dtypes": self.input_dtypes,
            "output_shapes": {k: list(v) for k, v in self.output_shapes.items()},
            "output_dtypes": self.output_dtypes,
        }
    @classmethod
    def from_dict(cls, data: dict) -> "ModelSchema":
        """Deserialize schema from dict."""
        return cls(
            input_shapes={k: tuple(v) for k, v in data["input_shapes"].items()},
            input_dtypes=data["input_dtypes"],
            output_shapes={k: tuple(v) for k, v in data["output_shapes"].items()},
            output_dtypes=data["output_dtypes"],
        )
    def get_input_rank(self, name: str) -> int:
        """Get the rank (number of dimensions) for an input."""
        return len(self.input_shapes.get(name, ()))
    def get_output_rank(self, name: str) -> int:
        """Get the rank (number of dimensions) for an output."""
        return len(self.output_shapes.get(name, ()))
```
---
### SchemaCompatibilityResult Dataclass
Result of schema compatibility check:
```python
# 03_schema.py
from dataclasses import dataclass
from typing import List, Optional
@dataclass
class SchemaCompatibilityResult:
    """
    Result of schema compatibility validation.
    Compatibility rules:
    - INPUT: New version can be MORE permissive (accept larger ranges)
    - INPUT: Dtypes must match exactly
    - OUTPUT: Shapes must match exactly (clients depend on fixed sizes)
    - OUTPUT: Dtypes must match exactly
    A compatible schema means clients designed for the old version
    will work with the new version without modification.
    """
    is_compatible: bool
    input_issues: List[str]
    output_issues: List[str]
    warnings: List[str]
    @property
    def has_issues(self) -> bool:
        """Check if any issues were found (even if compatible)."""
        return bool(self.input_issues or self.output_issues or self.warnings)
    def to_summary(self) -> str:
        """Generate human-readable summary."""
        if self.is_compatible and not self.warnings:
            return "Schemas are fully compatible"
        parts = []
        if self.input_issues:
            parts.append(f"Input issues: {'; '.join(self.input_issues)}")
        if self.output_issues:
            parts.append(f"Output issues: {'; '.join(self.output_issues)}")
        if self.warnings:
            parts.append(f"Warnings: {'; '.join(self.warnings)}")
        status = "COMPATIBLE" if self.is_compatible else "INCOMPATIBLE"
        return f"[{status}] " + " | ".join(parts)
```
---
### SchemaValidator Class
Validates compatibility between model versions:
```python
# 03_schema.py
from typing import Tuple, List, Union
class SchemaValidator:
    """
    Validates schema compatibility between model versions.
    Compatibility follows semantic versioning principles:
    - MAJOR: Breaking changes (reject deployment)
    - MINOR: Additive changes (accept, log warning)
    - PATCH: No API changes (accept silently)
    The validator is strict about outputs (clients depend on exact shapes)
    but permissive about inputs (model can accept more than it needs).
    """
    @staticmethod
    def check_input_compatibility(
        old_schema: ModelSchema,
        new_schema: ModelSchema
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Check if new version can accept inputs designed for old version.
        Compatible if:
        - All old inputs exist in new schema
        - Input dtypes match exactly
        - Input shapes are compatible (new can be MORE permissive)
        Shape compatibility rules:
        - Rank must match
        - None in old → anything in new (old already variable)
        - Specific in old → specific in new (must be same or None)
        - New can have None where old has specific (more permissive)
        Returns:
            (is_compatible, issues, warnings)
        """
        issues = []
        warnings = []
        for input_name, old_shape in old_schema.input_shapes.items():
            # Missing input = breaking change
            if input_name not in new_schema.input_shapes:
                issues.append(f"Missing input: '{input_name}'")
                continue
            new_shape = new_schema.input_shapes[input_name]
            # Dtype mismatch = breaking change
            old_dtype = old_schema.input_dtypes.get(input_name)
            new_dtype = new_schema.input_dtypes.get(input_name)
            if old_dtype != new_dtype:
                issues.append(
                    f"Input '{input_name}' dtype changed: "
                    f"{old_dtype} → {new_dtype}"
                )
            # Rank change = breaking change
            if len(old_shape) != len(new_shape):
                issues.append(
                    f"Input '{input_name}' rank changed: "
                    f"{len(old_shape)}D → {len(new_shape)}D"
                )
                continue
            # Dimension compatibility check
            for i, (old_dim, new_dim) in enumerate(zip(old_shape, new_shape)):
                if old_dim is not None and new_dim is not None:
                    if old_dim != new_dim:
                        # Both specific but different - breaking
                        issues.append(
                            f"Input '{input_name}' dim {i} changed: "
                            f"{old_dim} → {new_dim}"
                        )
                elif old_dim is not None and new_dim is None:
                    # Old specific, new variable - more permissive (warning)
                    warnings.append(
                        f"Input '{input_name}' dim {i} became variable: "
                        f"{old_dim} → None (more permissive)"
                    )
                # old_dim is None, new_dim anything = already variable in old
        return len(issues) == 0, issues, warnings
    @staticmethod
    def check_output_compatibility(
        old_schema: ModelSchema,
        new_schema: ModelSchema
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Check if new version produces outputs compatible with old version.
        Compatible if:
        - All old outputs exist in new schema
        - Output dtypes match exactly
        - Output shapes match EXACTLY (clients depend on fixed buffers)
        Unlike inputs, outputs must be identical. Clients often have
        fixed-size buffers for outputs and cannot handle shape changes.
        Returns:
            (is_compatible, issues, warnings)
        """
        issues = []
        warnings = []
        for output_name, old_shape in old_schema.output_shapes.items():
            if output_name not in new_schema.output_shapes:
                issues.append(f"Missing output: '{output_name}'")
                continue
            new_shape = new_schema.output_shapes[output_name]
            # Dtype must match exactly
            old_dtype = old_schema.output_dtypes.get(output_name)
            new_dtype = new_schema.output_dtypes.get(output_name)
            if old_dtype != new_dtype:
                issues.append(
                    f"Output '{output_name}' dtype changed: "
                    f"{old_dtype} → {new_dtype}"
                )
            # Shape must match EXACTLY for outputs
            if old_shape != new_shape:
                issues.append(
                    f"Output '{output_name}' shape changed: "
                    f"{old_shape} → {new_shape}"
                )
        return len(issues) == 0, issues, warnings
    @staticmethod
    def full_compatibility_check(
        old_schema: ModelSchema,
        new_schema: ModelSchema
    ) -> SchemaCompatibilityResult:
        """
        Complete compatibility check with detailed report.
        Returns:
            SchemaCompatibilityResult with all issues and warnings
        """
        input_ok, input_issues, input_warnings = (
            SchemaValidator.check_input_compatibility(old_schema, new_schema)
        )
        output_ok, output_issues, output_warnings = (
            SchemaValidator.check_output_compatibility(old_schema, new_schema)
        )
        all_warnings = input_warnings + output_warnings
        return SchemaCompatibilityResult(
            is_compatible=input_ok and output_ok,
            input_issues=input_issues,
            output_issues=output_issues,
            warnings=all_warnings
        )
```
---
### ModelVersion Dataclass
Complete metadata for a single model version:
```python
# 04_model_version.py
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime
import hashlib
import torch
from .03_schema import ModelSchema
from .02_exceptions import VersionStatus
@dataclass
class ModelVersion:
    """
    Complete metadata for a single model version.
    This is the unit of deployment in the registry. It tracks:
    1. Model identity (version_id, path)
    2. Training provenance (date, data hash, metrics)
    3. API contract (schema)
    4. Runtime state (loaded_model, status)
    5. Request tracking (in_flight_count)
    Invariants:
        - version_id is unique across all versions
        - If status == ACTIVE, loaded_model is not None
        - If status == DEPRECATED, loaded_model is None
        - in_flight_count is only > 0 when status in [ACTIVE, DRAINING]
    """
    version_id: str                          # e.g., "v1.2.3" or "2024-01-15-a"
    model_path: str                          # Path to model file
    status: VersionStatus = VersionStatus.STAGED
    # Training metadata
    training_date: Optional[datetime] = None
    training_data_hash: Optional[str] = None  # Hash of training dataset
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    # Schema (the API contract)
    schema: Optional[ModelSchema] = None
    # Runtime state
    loaded_model: Any = None  # The actual model object (set when loaded)
    device: Optional[str] = None
    load_time: Optional[datetime] = None
    warmup_latency_ms: Optional[float] = None
    # Request tracking for draining
    in_flight_count: int = 0
    def to_metadata_dict(self) -> dict:
        """
        Export metadata (without the loaded model object).
        Used for:
        - API responses
        - Logging
        - Persistence
        The loaded_model is intentionally excluded as it's
        not JSON-serializable and may be large.
        """
        return {
            "version_id": self.version_id,
            "model_path": self.model_path,
            "status": self.status.value,
            "training_date": (
                self.training_date.isoformat() 
                if self.training_date else None
            ),
            "training_data_hash": self.training_data_hash,
            "accuracy_metrics": self.accuracy_metrics,
            "hyperparameters": self.hyperparameters,
            "schema": self.schema.to_dict() if self.schema else None,
            "device": self.device,
            "load_time": (
                self.load_time.isoformat() 
                if self.load_time else None
            ),
            "warmup_latency_ms": self.warmup_latency_ms,
            "in_flight_count": self.in_flight_count,
        }
    def compute_file_hash(self) -> str:
        """
        Compute SHA256 hash of the model file for integrity verification.
        Returns:
            First 16 characters of hex digest (sufficient for identity)
        """
        sha256 = hashlib.sha256()
        with open(self.model_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]
    def is_ready_for_inference(self) -> bool:
        """Check if this version can serve inference requests."""
        return (
            self.status == VersionStatus.ACTIVE and
            self.loaded_model is not None
        )
    def can_be_unloaded(self) -> bool:
        """Check if this version can be safely unloaded."""
        return (
            self.status in [VersionStatus.DRAINING, VersionStatus.DEPRECATED] and
            self.in_flight_count == 0
        )
```
---
### AtomicVersionPointer Dataclass
Thread-safe pointer to the active model version:
```python
# 06_atomic_pointer.py
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional
import logging
from .04_model_version import ModelVersion
logger = logging.getLogger(__name__)
@dataclass
class AtomicVersionPointer:
    """
    Thread-safe pointer to the active model version.
    Uses asyncio synchronization to ensure atomic swaps.
    The key insight: reading the pointer is lock-free (fast path),
    but swapping requires exclusive access.
    Reference counting tracks in-flight requests per version,
    enabling proper draining before unload.
    Invariants:
        - If _version is not None, it's safe to use for inference
        - _ref_counts tracks all versions that have/had in-flight requests
        - Swap is atomic - no request sees a partially swapped state
    """
    _version_id: Optional[str] = None
    _version: Optional[ModelVersion] = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _ref_counts: Dict[str, int] = field(default_factory=dict)
    async def get(self) -> Optional[ModelVersion]:
        """
        Get the current active version.
        This is the fast path—no lock needed for read.
        Safe to call concurrently from multiple coroutines.
        Returns:
            Current active ModelVersion, or None if none set
        """
        return self._version
    async def get_version_id(self) -> Optional[str]:
        """Get the current active version ID without the full object."""
        return self._version_id
    async def swap(
        self,
        new_version: ModelVersion
    ) -> Optional[ModelVersion]:
        """
        Atomically swap the active version.
        This operation:
        1. Acquires exclusive lock
        2. Stores old version for return
        3. Sets new version atomically
        4. Initializes ref count for new version
        Args:
            new_version: The version to make active
        Returns:
            The previous active version (for draining)
        Postcondition: All subsequent get() calls return new_version
        """
        async with self._lock:
            old_version = self._version
            old_id = self._version_id
            # Atomic swap
            self._version = new_version
            self._version_id = new_version.version_id
            # Initialize ref count for new version
            if new_version.version_id not in self._ref_counts:
                self._ref_counts[new_version.version_id] = 0
            logger.info(
                f"[AtomicPointer] Swapped: {old_id} → {new_version.version_id}"
            )
            return old_version
    async def increment_ref(self, version_id: str) -> int:
        """
        Increment in-flight request count for a version.
        Args:
            version_id: Version to increment count for
        Returns:
            New reference count
        Precondition: version_id must be in _ref_counts
        """
        async with self._lock:
            if version_id not in self._ref_counts:
                self._ref_counts[version_id] = 0
            self._ref_counts[version_id] += 1
            return self._ref_counts[version_id]
    async def decrement_ref(self, version_id: str) -> int:
        """
        Decrement in-flight request count for a version.
        Args:
            version_id: Version to decrement count for
        Returns:
            New reference count (may be 0)
        Note: Count can go to 0, which signals drain complete
        """
        async with self._lock:
            if version_id in self._ref_counts:
                self._ref_counts[version_id] -= 1
                return self._ref_counts[version_id]
            return 0
    async def get_ref_count(self, version_id: str) -> int:
        """Get current in-flight count for a version."""
        async with self._lock:
            return self._ref_counts.get(version_id, 0)
    async def get_all_ref_counts(self) -> Dict[str, int]:
        """Get copy of all reference counts."""
        async with self._lock:
            return dict(self._ref_counts)
```
---
### RequestTracker Dataclass
Tracks in-flight requests per model version:
```python
# 07_request_tracker.py
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional
import time
import logging
logger = logging.getLogger(__name__)
@dataclass
class RequestTracker:
    """
    Tracks in-flight requests per model version.
    Used for:
    1. Draining: wait for count to reach zero before unload
    2. Metrics: track concurrent request load
    3. Health: detect stuck requests
    The tracker uses asyncio.Events to signal when draining completes,
    enabling efficient waiting without polling.
    Invariants:
        - _counts[version_id] >= 0 for all versions
        - If _drain_events[version_id] exists, draining is in progress
        - Event is set when count reaches 0 during drain
    """
    _counts: Dict[str, int] = field(default_factory=dict)
    _drain_events: Dict[str, asyncio.Event] = field(default_factory=dict)
    _drain_start_times: Dict[str, float] = field(default_factory=dict)
    drain_timeout_seconds: float = 30.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    async def acquire(self, version_id: str) -> int:
        """
        Register a new request for a version.
        Call this before starting inference.
        Args:
            version_id: Version handling the request
        Returns:
            New count for this version
        Side effects:
            - Increments count
            - Logs if this is the first request for version
        """
        async with self._lock:
            if version_id not in self._counts:
                self._counts[version_id] = 0
            self._counts[version_id] += 1
            count = self._counts[version_id]
            if count == 1:
                logger.debug(
                    f"[Tracker] First in-flight request for {version_id}"
                )
            return count
    async def release(self, version_id: str) -> int:
        """
        Release a request for a version.
        Call this after inference completes (success or error).
        Args:
            version_id: Version that handled the request
        Returns:
            New count for this version
        Side effects:
            - Decrements count
            - Signals drain event if count reaches 0
        """
        async with self._lock:
            if version_id in self._counts:
                self._counts[version_id] -= 1
                count = self._counts[version_id]
                # If draining and count hit zero, signal the event
                if count == 0 and version_id in self._drain_events:
                    self._drain_events[version_id].set()
                    drain_time = time.time() - self._drain_start_times.get(version_id, time.time())
                    logger.info(
                        f"[Tracker] Drain complete for {version_id} "
                        f"in {drain_time:.2f}s"
                    )
                return count
            return 0
    async def get_count(self, version_id: str) -> int:
        """Get current in-flight count for a version."""
        async with self._lock:
            return self._counts.get(version_id, 0)
    async def get_all_counts(self) -> Dict[str, int]:
        """Get copy of all in-flight counts."""
        async with self._lock:
            return dict(self._counts)
    async def start_drain(self, version_id: str) -> None:
        """
        Start draining a version.
        Creates an event that will be set when count reaches zero.
        If count is already zero, event is set immediately.
        Args:
            version_id: Version to start draining
        """
        async with self._lock:
            if version_id not in self._drain_events:
                self._drain_events[version_id] = asyncio.Event()
            self._drain_start_times[version_id] = time.time()
            # If already zero, signal immediately
            if self._counts.get(version_id, 0) == 0:
                self._drain_events[version_id].set()
                logger.info(
                    f"[Tracker] Drain for {version_id} already complete "
                    f"(no in-flight requests)"
                )
            else:
                logger.info(
                    f"[Tracker] Started draining {version_id} "
                    f"({self._counts[version_id]} in-flight)"
                )
    async def wait_for_drain(self, version_id: str) -> bool:
        """
        Wait for all in-flight requests to complete.
        Args:
            version_id: Version to wait for
        Returns:
            True if drained successfully, False if timed out
        """
        async with self._lock:
            event = self._drain_events.get(version_id)
            if not event:
                # No drain in progress, check count
                return self._counts.get(version_id, 0) == 0
        try:
            await asyncio.wait_for(
                event.wait(),
                timeout=self.drain_timeout_seconds
            )
            return True
        except asyncio.TimeoutError:
            count = self._counts.get(version_id, 0)
            logger.warning(
                f"[Tracker] Drain timeout for {version_id}: "
                f"{count} requests still in flight after "
                f"{self.drain_timeout_seconds}s"
            )
            return False
    async def force_drain(self, version_id: str) -> None:
        """
        Force drain completion (for shutdown).
        Use only when you're going to unload regardless.
        Sets count to 0 and signals event.
        """
        async with self._lock:
            if version_id in self._drain_events:
                self._drain_events[version_id].set()
            self._counts[version_id] = 0
            logger.warning(
                f"[Tracker] Force-drained {version_id}"
            )
```
---
### ModelRegistry Dataclass
Central registry for model versions:
```python
# 05_model_registry.py
import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Awaitable
import logging
from .04_model_version import ModelVersion
from .02_exceptions import VersionStatus, VersionNotFound, VersionNotLoaded
from .03_schema import SchemaValidator, SchemaCompatibilityResult
logger = logging.getLogger(__name__)
@dataclass
class ModelRegistry:
    """
    Central registry for model versions with lifecycle management.
    Key responsibilities:
    1. Store version metadata
    2. Track active vs staged versions
    3. Validate schema compatibility before promotion
    4. Coordinate loading/unloading with memory constraints
    The registry uses OrderedDict to maintain insertion order,
    which is useful for listing versions chronologically.
    Thread safety: All public methods use asyncio.Lock for
    safe concurrent access within the event loop.
    Invariants:
        - At most one version has status == ACTIVE
        - _active_version_id points to the active version (or None)
        - All versions in _versions have unique version_id
    """
    _versions: OrderedDict = field(default_factory=OrderedDict)
    _active_version_id: Optional[str] = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    max_versions_in_memory: int = 3
    # Callbacks for lifecycle events
    _on_version_loaded: Optional[Callable[[str], Awaitable[None]]] = None
    _on_version_activated: Optional[Callable[[str], Awaitable[None]]] = None
    _on_version_unloaded: Optional[Callable[[str], Awaitable[None]]] = None
    async def register_version(
        self,
        version: ModelVersion,
        auto_load: bool = False
    ) -> bool:
        """
        Register a new model version in the registry.
        Args:
            version: ModelVersion with metadata populated
            auto_load: If True, load the model immediately
        Returns:
            True if registered successfully
            False if version_id already exists
        Postcondition: Version is in registry with status STAGED
        """
        async with self._lock:
            if version.version_id in self._versions:
                logger.warning(
                    f"[Registry] Version {version.version_id} already registered"
                )
                return False
            self._versions[version.version_id] = version
            logger.info(
                f"[Registry] Registered version {version.version_id}"
            )
            return True
    async def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get version metadata by ID."""
        async with self._lock:
            return self._versions.get(version_id)
    async def get_active_version(self) -> Optional[ModelVersion]:
        """Get the currently active version."""
        async with self._lock:
            if self._active_version_id:
                return self._versions.get(self._active_version_id)
            return None
    async def get_active_version_id(self) -> Optional[str]:
        """Get the currently active version ID."""
        async with self._lock:
            return self._active_version_id
    async def list_versions(self) -> List[dict]:
        """
        List all registered versions with metadata.
        Returns:
            List of metadata dicts, ordered by insertion time
        """
        async with self._lock:
            return [
                v.to_metadata_dict()
                for v in self._versions.values()
            ]
    async def set_active(self, version_id: str) -> bool:
        """
        Set a version as active.
        Args:
            version_id: Version to activate
        Returns:
            True if successful
        Raises:
            VersionNotFound: If version not in registry
        Precondition: Version must be loaded (loaded_model not None)
        Postcondition: Version status is ACTIVE, previous active is DRAINING
        """
        async with self._lock:
            version = self._versions.get(version_id)
            if not version:
                raise VersionNotFound(version_id)
            # Update previous active to DRAINING
            if self._active_version_id and self._active_version_id != version_id:
                prev_active = self._versions.get(self._active_version_id)
                if prev_active:
                    prev_active.status = VersionStatus.DRAINING
            # Set new active
            version.status = VersionStatus.ACTIVE
            self._active_version_id = version_id
            logger.info(
                f"[Registry] Set {version_id} as active"
            )
            return True
    async def validate_compatibility(
        self,
        new_version_id: str
    ) -> Tuple[bool, Optional[SchemaCompatibilityResult]]:
        """
        Validate that a new version is compatible with the active version.
        If no active version exists, any schema is accepted.
        Args:
            new_version_id: Version to validate
        Returns:
            (is_compatible, result):
            - (True, None) if no active version (any schema ok)
            - (True, result) if compatible
            - (False, result) if incompatible
        """
        async with self._lock:
            new_version = self._versions.get(new_version_id)
            if not new_version:
                return False, None
            if not new_version.schema:
                # No schema defined, can't validate
                logger.warning(
                    f"[Registry] Version {new_version_id} has no schema, "
                    "skipping compatibility check"
                )
                return True, None
            # If no active version, any schema is fine
            if not self._active_version_id:
                return True, None
            active_version = self._versions.get(self._active_version_id)
            if not active_version or not active_version.schema:
                return True, None
            result = SchemaValidator.full_compatibility_check(
                active_version.schema,
                new_version.schema
            )
            return result.is_compatible, result
    async def mark_deprecated(self, version_id: str) -> bool:
        """
        Mark a version as deprecated.
        Args:
            version_id: Version to deprecate
        Returns:
            True if successful
        Precondition: Version must not be ACTIVE
        Postcondition: Version status is DEPRECATED, loaded_model is None
        """
        async with self._lock:
            version = self._versions.get(version_id)
            if not version:
                return False
            if version.status == VersionStatus.ACTIVE:
                logger.error(
                    f"[Registry] Cannot deprecate active version {version_id}"
                )
                return False
            version.status = VersionStatus.DEPRECATED
            version.loaded_model = None
            logger.info(
                f"[Registry] Marked {version_id} as deprecated"
            )
            return True
    def set_callbacks(
        self,
        on_load: Optional[Callable[[str], Awaitable[None]]] = None,
        on_activate: Optional[Callable[[str], Awaitable[None]]] = None,
        on_unload: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> None:
        """Set lifecycle callbacks for external coordination."""
        self._on_version_loaded = on_load
        self._on_version_activated = on_activate
        self._on_version_unloaded = on_unload
    async def _invoke_callback(
        self,
        callback: Optional[Callable[[str], Awaitable[None]]],
        version_id: str
    ) -> None:
        """Safely invoke a lifecycle callback."""
        if callback:
            try:
                await callback(version_id)
            except Exception as e:
                logger.error(
                    f"[Registry] Callback error for {version_id}: {e}"
                )
```
---
### Exception Hierarchy
```python
# 02_exceptions.py
class VersioningError(Exception):
    """Base exception for all versioning errors."""
    pass
class VersionNotFound(VersioningError):
    """
    Raised when a requested version is not in the registry.
    Attributes:
        version_id: The version that was not found
    """
    def __init__(self, version_id: str):
        self.version_id = version_id
        super().__init__(f"Version not found: {version_id}")
class VersionNotLoaded(VersioningError):
    """
    Raised when trying to use a version that exists but isn't loaded.
    Attributes:
        version_id: The version that is not loaded
        status: Current status of the version
    """
    def __init__(self, version_id: str, status: VersionStatus):
        self.version_id = version_id
        self.status = status
        super().__init__(
            f"Version {version_id} exists but is not loaded "
            f"(status: {status.value})"
        )
class SchemaIncompatible(VersioningError):
    """
    Raised when schema compatibility check fails.
    Attributes:
        old_version: The active version
        new_version: The version being promoted
        result: Detailed compatibility result
    """
    def __init__(
        self,
        old_version: str,
        new_version: str,
        result: "SchemaCompatibilityResult"
    ):
        self.old_version = old_version
        self.new_version = new_version
        self.result = result
        super().__init__(
            f"Schema incompatibility between {old_version} and {new_version}: "
            f"{result.to_summary()}"
        )
class DrainTimeout(VersioningError):
    """
    Raised when in-flight requests don't complete within timeout.
    Attributes:
        version_id: Version being drained
        remaining_count: In-flight requests remaining
        timeout_seconds: Configured timeout
    """
    def __init__(
        self,
        version_id: str,
        remaining_count: int,
        timeout_seconds: float
    ):
        self.version_id = version_id
        self.remaining_count = remaining_count
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Drain timeout for {version_id}: {remaining_count} requests "
            f"still in flight after {timeout_seconds}s"
        )
class RollbackFailed(VersioningError):
    """
    Raised when rollback cannot be performed.
    Attributes:
        reason: Why rollback failed
    """
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Rollback failed: {reason}")
class MemoryExhausted(VersioningError):
    """
    Raised when insufficient memory for dual-load during hot swap.
    Attributes:
        required_bytes: Estimated memory needed
        available_bytes: Memory available
    """
    def __init__(self, required_bytes: int, available_bytes: int):
        self.required_bytes = required_bytes
        self.available_bytes = available_bytes
        super().__init__(
            f"Insufficient memory for hot swap: need {required_bytes/1e9:.2f}GB, "
            f"have {available_bytes/1e9:.2f}GB"
        )
class HotSwapError(VersioningError):
    """
    Raised when hot swap operation fails.
    Attributes:
        version_id: Version being swapped to
        reason: Failure reason
    """
    def __init__(self, version_id: str, reason: str):
        self.version_id = version_id
        self.reason = reason
        super().__init__(
            f"Hot swap to {version_id} failed: {reason}"
        )
```
---
## Tensor Shape Specifications
### Schema Shape Comparison
```
Schema Compatibility Examples:
COMPATIBLE (input more permissive, output identical):
  Old: input_shapes = {"image": (None, 3, 224, 224)}
       output_shapes = {"logits": (None, 1000)}
  New: input_shapes = {"image": (None, 3, 224, 224)}
       output_shapes = {"logits": (None, 1000)}
  Result: COMPATIBLE (identical)
COMPATIBLE (input became variable):
  Old: input_shapes = {"image": (None, 3, 224, 224)}
  New: input_shapes = {"image": (None, None, None, None)}
  Result: COMPATIBLE with warning (more permissive)
INCOMPATIBLE (output shape changed):
  Old: output_shapes = {"embedding": (None, 768)}
  New: output_shapes = {"embedding": (None, 512)}
  Result: INCOMPATIBLE (clients expect 768-dim)
INCOMPATIBLE (dtype changed):
  Old: input_dtypes = {"image": "torch.float32"}
  New: input_dtypes = {"image": "torch.float16"}
  Result: INCOMPATIBLE (dtype mismatch)
INCOMPATIBLE (missing input):
  Old: input_shapes = {"image": ..., "mask": ...}
  New: input_shapes = {"image": ...}
  Result: INCOMPATIBLE (missing "mask" input)
```
---

![Model Registry Structure](./diagrams/tdd-diag-018.svg)

![Alert Rule Evaluation with Cooldown](./diagrams/tdd-diag-042.svg)

---
## Interface Contracts
### HotSwapManager
```python
# 08_hot_swap_manager.py
import asyncio
import time
import logging
from typing import Callable, Awaitable, Optional, Tuple
import torch
from .05_model_registry import ModelRegistry
from .06_atomic_pointer import AtomicVersionPointer
from .07_request_tracker import RequestTracker
from .04_model_version import ModelVersion, VersionStatus
from .02_exceptions import (
    HotSwapError, SchemaIncompatible, DrainTimeout, RollbackFailed
)
from .03_schema import SchemaCompatibilityResult
logger = logging.getLogger(__name__)
ModelLoader = Callable[[ModelVersion], Awaitable[ModelVersion]]
class HotSwapManager:
    """
    Orchestrates zero-downtime model version swaps.
    The swap sequence:
    1. Load new version (in parallel with serving)
    2. Validate schema compatibility
    3. Warmup new version
    4. Atomically swap active pointer
    5. Start draining old version
    6. Wait for in-flight requests to complete
    7. Unload old version
    The manager keeps the previous version in memory for instant rollback.
    If the new version fails health checks, swap back is O(1).
    Invariants:
        - No request is dropped during swap
        - Every request sees exactly one version
        - Rollback completes in < 5 seconds
        - Peak memory = old + new + buffers
    """
    def __init__(
        self,
        registry: ModelRegistry,
        pointer: AtomicVersionPointer,
        tracker: RequestTracker,
        model_loader: ModelLoader,
    ):
        self.registry = registry
        self.pointer = pointer
        self.tracker = tracker
        self.model_loader = model_loader
        # Previous version kept for instant rollback
        self._previous_version: Optional[ModelVersion] = None
        self._previous_version_id: Optional[str] = None
        # Swap lock ensures only one swap at a time
        self._swap_lock = asyncio.Lock()
    async def hot_swap(
        self,
        new_version_id: str,
        validate_schema: bool = True,
        keep_previous: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """
        Perform a hot swap to a new model version.
        Args:
            new_version_id: Version to activate
            validate_schema: Check compatibility before swap
            keep_previous: Keep old version in memory for rollback
        Returns:
            (success, error_message):
            - (True, None) on success
            - (False, "error description") on failure
        Raises:
            HotSwapError: If swap fails catastrophically
        Side effects:
            - Loads new model if not already loaded
            - Updates registry status
            - Swaps atomic pointer
            - Starts draining old version
        """
        async with self._swap_lock:
            swap_start = time.perf_counter()
            logger.info(
                f"[HotSwap] Starting swap to {new_version_id}"
            )
            # Step 1: Get the new version
            new_version = await self.registry.get_version(new_version_id)
            if not new_version:
                return False, f"Version {new_version_id} not found"
            # Step 2: Validate schema compatibility
            if validate_schema:
                is_compatible, result = await self.registry.validate_compatibility(
                    new_version_id
                )
                if not is_compatible and result:
                    logger.error(
                        f"[HotSwap] Schema incompatibility: "
                        f"{result.to_summary()}"
                    )
                    return False, f"Schema incompatible: {result.to_summary()}"
            # Step 3: Load the new model (if not already loaded)
            if new_version.loaded_model is None:
                try:
                    new_version = await self.model_loader(new_version)
                    logger.info(
                        f"[HotSwap] Loaded {new_version_id}"
                    )
                except Exception as e:
                    new_version.status = VersionStatus.FAILED
                    return False, f"Failed to load model: {e}"
            # Step 4: Get current active version (for draining)
            old_version = await self.pointer.get()
            old_version_id = old_version.version_id if old_version else None
            # Step 5: Atomically swap
            await self.pointer.swap(new_version)
            await self.registry.set_active(new_version_id)
            # Step 6: Start draining old version
            if old_version and old_version_id:
                old_version.status = VersionStatus.DRAINING
                await self.tracker.start_drain(old_version_id)
                logger.info(
                    f"[HotSwap] Swapped {old_version_id} → {new_version_id}, "
                    f"draining old version"
                )
            # Step 7: Keep previous for rollback (optional)
            if keep_previous and old_version:
                self._previous_version = old_version
                self._previous_version_id = old_version_id
            swap_time = (time.perf_counter() - swap_start) * 1000
            logger.info(
                f"[HotSwap] Complete in {swap_time:.1f}ms"
            )
            return True, None
    async def complete_drain_and_unload(
        self,
        version_id: str,
        force: bool = False
    ) -> bool:
        """
        Wait for drain and unload a version.
        Args:
            version_id: Version to unload
            force: If True, don't wait for drain
        Returns:
            True if unloaded successfully
            False if timed out (and force=False)
        Postcondition:
            - Version status is DEPRECATED
            - loaded_model is None
            - GPU memory is freed
        """
        version = await self.registry.get_version(version_id)
        if not version:
            return False
        # Wait for drain (unless forcing)
        if not force:
            drained = await self.tracker.wait_for_drain(version_id)
            if not drained:
                logger.warning(
                    f"[HotSwap] Drain incomplete for {version_id}, forcing unload"
                )
        # Unload from memory
        if version.loaded_model is not None:
            # Clean up GPU memory
            if torch.cuda.is_available() and version.device == "cuda":
                del version.loaded_model
                torch.cuda.empty_cache()
            else:
                del version.loaded_model
            version.loaded_model = None
            await self.registry.mark_deprecated(version_id)
            logger.info(f"[HotSwap] Unloaded {version_id}")
        return True
    async def rollback(self) -> Tuple[bool, Optional[str]]:
        """
        Rollback to the previous version.
        This is O(1) because the previous version is kept in memory.
        Returns:
            (success, error_message):
            - (True, None) on success
            - (False, "reason") if no previous version available
        """
        rollback_start = time.perf_counter()
        if not self._previous_version:
            return False, "No previous version available for rollback"
        previous_id = self._previous_version_id
        logger.warning(
            f"[HotSwap] Initiating rollback to {previous_id}"
        )
        # Get current version for draining
        current = await self.pointer.get()
        current_id = current.version_id if current else None
        # Verify previous is still loaded
        if self._previous_version.loaded_model is None:
            return False, f"Previous version {previous_id} is no longer loaded"
        # Atomic swap (instant)
        await self.pointer.swap(self._previous_version)
        await self.registry.set_active(previous_id)
        self._previous_version.status = VersionStatus.ACTIVE
        # Start draining current (failed) version
        if current and current_id:
            current.status = VersionStatus.DRAINING
            await self.tracker.start_drain(current_id)
        rollback_time = (time.perf_counter() - rollback_start) * 1000
        logger.info(
            f"[HotSwap] Rollback complete in {rollback_time:.1f}ms "
            f"(target: <5000ms)"
        )
        return True, None
    def get_previous_version_id(self) -> Optional[str]:
        """Get the version ID kept for rollback."""
        return self._previous_version_id
```
---
### VersionRouter
```python
# 10_version_router.py
import asyncio
from typing import Optional, Tuple, Any, Callable, Awaitable
from fastapi import Request, HTTPException
import logging
from .05_model_registry import ModelRegistry
from .06_atomic_pointer import AtomicVersionPointer
from .07_request_tracker import RequestTracker
from .04_model_version import ModelVersion
from .02_exceptions import VersionNotFound, VersionNotLoaded
logger = logging.getLogger(__name__)
class VersionRouter:
    """
    Routes requests to specific model versions.
    Routing strategies:
    1. URL path: /v/{version_id}/predict
    2. Header: X-Model-Version: v1.2.3
    3. Default: route to active version
    The router integrates with RequestTracker to ensure
    proper reference counting for draining.
    """
    def __init__(
        self,
        registry: ModelRegistry,
        pointer: AtomicVersionPointer,
        tracker: RequestTracker,
    ):
        self.registry = registry
        self.pointer = pointer
        self.tracker = tracker
    async def route_request(
        self,
        request: Request,
        requested_version: Optional[str] = None
    ) -> Tuple[ModelVersion, str]:
        """
        Determine which version should handle this request.
        Priority:
        1. Explicit version in URL path
        2. Version in X-Model-Version header
        3. Active version (default)
        Args:
            request: FastAPI request object
            requested_version: Explicit version from URL path
        Returns:
            (version, version_id) tuple
        Raises:
            HTTPException:
            - 404 if version not found
            - 503 if version not loaded
        """
        version_id: Optional[str] = requested_version
        # Check header if no explicit version
        if not version_id:
            version_id = request.headers.get("X-Model-Version")
        # Route to specific version or default
        if version_id:
            version = await self.registry.get_version(version_id)
            if not version:
                raise HTTPException(
                    status_code=404,
                    detail=f"Version {version_id} not found"
                )
            if not version.is_ready_for_inference():
                raise HTTPException(
                    status_code=503,
                    detail=f"Version {version_id} not ready for inference "
                           f"(status: {version.status.value})"
                )
            return version, version_id
        # Default to active version
        active = await self.pointer.get()
        if not active:
            raise HTTPException(
                status_code=503,
                detail="No active model version"
            )
        return active, active.version_id
    async def execute_with_tracking(
        self,
        version_id: str,
        inference_fn: Callable[[str], Awaitable[Any]]
    ) -> Any:
        """
        Execute inference with request tracking.
        This ensures proper reference counting for draining:
        1. Increment count before inference
        2. Run inference
        3. Decrement count after (success or error)
        Args:
            version_id: Version handling the request
            inference_fn: Async function to execute
        Returns:
            Result from inference_fn
        Raises:
            Whatever inference_fn raises
        """
        # Track this request
        await self.tracker.acquire(version_id)
        try:
            result = await inference_fn()
            return result
        finally:
            await self.tracker.release(version_id)
    async def get_active_version_id(self) -> Optional[str]:
        """Get the current active version ID."""
        return await self.pointer.get_version_id()
```
---
---
## Algorithm Specification
### Hot Swap Algorithm
```
ALGORITHM: hot_swap
INPUT: new_version_id (str), validate_schema (bool), keep_previous (bool)
OUTPUT: (success: bool, error_message: Optional[str])
PRECONDITION: new_version_id exists in registry
1. ACQUIRE swap_lock (ensure only one swap at a time)
   swap_start ← current_time()
2. // Get new version
   new_version ← registry.get_version(new_version_id)
   IF new_version is None:
       RETURN (False, "Version not found")
3. // Schema validation (optional)
   IF validate_schema:
       (is_compatible, result) ← registry.validate_compatibility(new_version_id)
       IF NOT is_compatible:
           LOG ERROR "Schema incompatibility"
           RETURN (False, result.to_summary())
4. // Load new model if needed
   IF new_version.loaded_model is None:
       TRY:
           new_version ← model_loader(new_version)
       EXCEPT Exception AS e:
           new_version.status ← FAILED
           RETURN (False, f"Load failed: {e}")
5. // Get current active for draining
   old_version ← pointer.get()
   old_version_id ← old_version.version_id IF old_version ELSE None
6. // ATOMIC SWAP
   pointer.swap(new_version)
   registry.set_active(new_version_id)
7. // Start draining old version
   IF old_version is not None:
       old_version.status ← DRAINING
       tracker.start_drain(old_version_id)
       LOG INFO "Swapped {old_version_id} → {new_version_id}"
8. // Keep previous for rollback
   IF keep_previous AND old_version is not None:
       _previous_version ← old_version
       _previous_version_id ← old_version_id
9. swap_time ← current_time() - swap_start
   LOG INFO "Hot swap complete in {swap_time}ms"
   RETURN (True, None)
POSTCONDITION: All new requests see new_version
POSTCONDITION: old_version has status DRAINING
POSTCONDITION: _previous_version points to old_version (if keep_previous)
```
### Drain and Unload Algorithm
```
ALGORITHM: complete_drain_and_unload
INPUT: version_id (str), force (bool)
OUTPUT: success (bool)
PRECONDITION: version_id exists and is not ACTIVE
1. version ← registry.get_version(version_id)
   IF version is None:
       RETURN False
2. // Wait for drain (unless forcing)
   IF NOT force:
       drained ← tracker.wait_for_drain(version_id)
       IF NOT drained:
           LOG WARNING "Drain timeout, forcing unload"
3. // Unload model from memory
   IF version.loaded_model is not None:
       IF version.device == "cuda":
           DELETE version.loaded_model
           torch.cuda.empty_cache()
       ELSE:
           DELETE version.loaded_model
       version.loaded_model ← None
       registry.mark_deprecated(version_id)
       LOG INFO "Unloaded {version_id}"
4. RETURN True
POSTCONDITION: version.loaded_model is None
POSTCONDITION: version.status is DEPRECATED
POSTCONDITION: GPU memory is freed (if applicable)
```
### Rollback Algorithm
```
ALGORITHM: rollback
INPUT: None (uses stored _previous_version)
OUTPUT: (success: bool, error_message: Optional[str])
PRECONDITION: None
1. rollback_start ← current_time()
2. // Check previous version available
   IF _previous_version is None:
       RETURN (False, "No previous version available")
3. previous_id ← _previous_version_id
4. // Verify previous is still loaded
   IF _previous_version.loaded_model is None:
       RETURN (False, "Previous version no longer loaded")
5. // Get current for draining
   current ← pointer.get()
   current_id ← current.version_id IF current ELSE None
6. // ATOMIC SWAP (instant - previous already loaded)
   pointer.swap(_previous_version)
   registry.set_active(previous_id)
   _previous_version.status ← ACTIVE
7. // Start draining failed version
   IF current is not None:
       current.status ← DRAINING
       tracker.start_drain(current_id)
8. rollback_time ← current_time() - rollback_start
   LOG WARNING "Rollback complete in {rollback_time}ms"
   ASSERT rollback_time < 5000ms, "Rollback exceeded 5s target"
9. RETURN (True, None)
POSTCONDITION: _previous_version is now ACTIVE
POSTCONDITION: Previous active version is DRAINING
POSTCONDITION: Rollback completed in < 5 seconds
```
---

![Memory Budget During Hot Swap](./diagrams/tdd-diag-020.svg)

---
## Error Handling Matrix
| Error | Detected By | Recovery | User-Visible? |
|-------|-------------|----------|---------------|
| `VersionNotFound` | `VersionRouter.route_request()` | Return HTTP 404 | Yes (404 "Version X not found") |
| `VersionNotLoaded` | `VersionRouter.route_request()` | Return HTTP 503 | Yes (503 "Version X not ready") |
| `SchemaIncompatible` | `HotSwapManager.hot_swap()` | Reject hot swap, log details | No (admin operation) |
| `DrainTimeout` | `RequestTracker.wait_for_drain()` | Force unload after warning | No (internal, logged) |
| `RollbackFailed` | `HotSwapManager.rollback()` | Return error, alert operators | No (admin operation) |
| `MemoryExhausted` | Model loader during dual-load | Abort swap, keep old version | No (internal, logged) |
| `HotSwapError` | `HotSwapManager.hot_swap()` | Return error, old version still active | No (admin operation) |
| No active version | `VersionRouter.route_request()` | Return HTTP 503 | Yes (503 "No active version") |
| Model load failure | `HotSwapManager.hot_swap()` catch | Mark version FAILED, keep old active | No (logged, admin alerted) |
---
## Implementation Sequence with Checkpoints
### Phase 1: ModelSchema and Compatibility (1-2 hours)
**Files to create:** `01_config.py`, `02_exceptions.py`, `03_schema.py`
**Implementation steps:**
1. Define `VersionStatus` enum
2. Define `ModelSchema` dataclass with serialization
3. Define `SchemaCompatibilityResult` dataclass
4. Implement `SchemaValidator` with input/output checks
5. Define exception hierarchy
**Checkpoint:**
```bash
python -c "
from schema import ModelSchema, SchemaValidator
# Test schema creation
old_schema = ModelSchema(
    input_shapes={'image': (None, 3, 224, 224)},
    input_dtypes={'image': 'torch.float32'},
    output_shapes={'logits': (None, 1000)},
    output_dtypes={'logits': 'torch.float32'}
)
# Compatible schema
new_schema_compatible = ModelSchema(
    input_shapes={'image': (None, 3, 224, 224)},
    input_dtypes={'image': 'torch.float32'},
    output_shapes={'logits': (None, 1000)},
    output_dtypes={'logits': 'torch.float32'}
)
result = SchemaValidator.full_compatibility_check(old_schema, new_schema_compatible)
assert result.is_compatible, f'Should be compatible: {result.to_summary()}'
print('Compatible check passed')
# Incompatible schema (output shape changed)
new_schema_incompatible = ModelSchema(
    input_shapes={'image': (None, 3, 224, 224)},
    input_dtypes={'image': 'torch.float32'},
    output_shapes={'logits': (None, 512)},  # Changed!
    output_dtypes={'logits': 'torch.float32'}
)
result = SchemaValidator.full_compatibility_check(old_schema, new_schema_incompatible)
assert not result.is_compatible, 'Should be incompatible'
assert 'logits' in str(result.output_issues), 'Should mention output issue'
print('Incompatible check passed')
"
```
---
### Phase 2: ModelVersion and ModelRegistry (2-3 hours)
**Files to create:** `04_model_version.py`, `05_model_registry.py`
**Implementation steps:**
1. Implement `ModelVersion` dataclass with all fields
2. Implement `to_metadata_dict()` for serialization
3. Implement `ModelRegistry` with asyncio.Lock
4. Implement `register_version()`, `get_version()`, `set_active()`
5. Implement `validate_compatibility()` integration
**Checkpoint:**
```bash
python -c "
import asyncio
from model_version import ModelVersion
from model_registry import ModelRegistry
from schema import ModelSchema
async def test():
    registry = ModelRegistry()
    # Create version
    version = ModelVersion(
        version_id='v1.0.0',
        model_path='models/v1.0.0.pt',
        schema=ModelSchema(
            input_shapes={'image': (None, 3, 224, 224)},
            input_dtypes={'image': 'torch.float32'},
            output_shapes={'logits': (None, 1000)},
            output_dtypes={'logits': 'torch.float32'}
        )
    )
    # Register
    success = await registry.register_version(version)
    assert success, 'Registration should succeed'
    print('Registration passed')
    # Get version
    retrieved = await registry.get_version('v1.0.0')
    assert retrieved.version_id == 'v1.0.0'
    print('Retrieval passed')
    # Set active (simulating loaded model)
    version.loaded_model = object()  # Simulate loaded
    await registry.set_active('v1.0.0')
    active = await registry.get_active_version()
    assert active.version_id == 'v1.0.0'
    print('Set active passed')
    # List versions
    versions = await registry.list_versions()
    assert len(versions) == 1
    print('List versions passed')
asyncio.run(test())
"
```
---
### Phase 3: AtomicVersionPointer (1-2 hours)
**Files to create:** `06_atomic_pointer.py`
**Implementation steps:**
1. Implement `AtomicVersionPointer` with asyncio.Lock
2. Implement `get()`, `swap()` with atomicity
3. Implement reference counting methods
4. Add logging for swap operations
**Checkpoint:**
```bash
python -c "
import asyncio
from atomic_pointer import AtomicVersionPointer
from model_version import ModelVersion
async def test():
    pointer = AtomicVersionPointer()
    # Initial state
    current = await pointer.get()
    assert current is None, 'Initial should be None'
    print('Initial state passed')
    # Create mock version
    version1 = ModelVersion(version_id='v1.0.0', model_path='m1.pt')
    version2 = ModelVersion(version_id='v2.0.0', model_path='m2.pt')
    # Swap
    old = await pointer.swap(version1)
    assert old is None
    current = await pointer.get()
    assert current.version_id == 'v1.0.0'
    print('First swap passed')
    # Second swap
    old = await pointer.swap(version2)
    assert old.version_id == 'v1.0.0'
    current = await pointer.get()
    assert current.version_id == 'v2.0.0'
    print('Second swap passed')
    # Reference counting
    count = await pointer.increment_ref('v2.0.0')
    assert count == 1
    count = await pointer.increment_ref('v2.0.0')
    assert count == 2
    count = await pointer.decrement_ref('v2.0.0')
    assert count == 1
    print('Reference counting passed')
asyncio.run(test())
"
```
---
### Phase 4: RequestTracker (2-3 hours)
**Files to create:** `07_request_tracker.py`
**Implementation steps:**
1. Implement `RequestTracker` with counts and events
2. Implement `acquire()` and `release()` with event signaling
3. Implement `start_drain()` and `wait_for_drain()`
4. Add timeout handling
**Checkpoint:**
```bash
python -c "
import asyncio
from request_tracker import RequestTracker
async def test():
    tracker = RequestTracker(drain_timeout_seconds=2.0)
    # Acquire/release
    count = await tracker.acquire('v1.0.0')
    assert count == 1
    count = await tracker.acquire('v1.0.0')
    assert count == 2
    print('Acquire passed')
    count = await tracker.release('v1.0.0')
    assert count == 1
    print('Release passed')
    # Drain (should wait since count > 0)
    await tracker.start_drain('v1.0.0')
    # Release remaining (should signal drain complete)
    async def delayed_release():
        await asyncio.sleep(0.5)
        await tracker.release('v1.0.0')
    # Start drain wait and delayed release concurrently
    release_task = asyncio.create_task(delayed_release())
    drained = await tracker.wait_for_drain('v1.0.0')
    await release_task
    assert drained, 'Drain should complete'
    print('Drain passed')
    # Timeout test
    await tracker.acquire('v2.0.0')
    await tracker.start_drain('v2.0.0')
    # Don't release - should timeout
    tracker.drain_timeout_seconds = 0.5
    drained = await tracker.wait_for_drain('v2.0.0')
    assert not drained, 'Should timeout'
    print('Timeout passed')
asyncio.run(test())
"
```
---
### Phase 5: HotSwapManager (3-4 hours)
**Files to create:** `08_hot_swap_manager.py`
**Implementation steps:**
1. Implement `HotSwapManager` with all dependencies
2. Implement `hot_swap()` with full sequence
3. Implement `complete_drain_and_unload()`
4. Implement `rollback()` with previous version tracking
5. Add schema validation integration
**Checkpoint:**
```bash
python -c "
import asyncio
import torch
from hot_swap_manager import HotSwapManager
from model_registry import ModelRegistry
from atomic_pointer import AtomicVersionPointer
from request_tracker import RequestTracker
from model_version import ModelVersion, VersionStatus
from schema import ModelSchema
async def test():
    registry = ModelRegistry()
    pointer = AtomicVersionPointer()
    tracker = RequestTracker(drain_timeout_seconds=2.0)
    # Mock model loader
    async def mock_loader(version):
        version.loaded_model = torch.nn.Linear(10, 2)
        version.device = 'cpu'
        version.status = VersionStatus.STAGED
        return version
    manager = HotSwapManager(registry, pointer, tracker, mock_loader)
    # Create and register versions
    v1 = ModelVersion(
        version_id='v1.0.0',
        model_path='v1.pt',
        schema=ModelSchema(
            input_shapes={'x': (None, 10)},
            input_dtypes={'x': 'torch.float32'},
            output_shapes={'y': (None, 2)},
            output_dtypes={'y': 'torch.float32'}
        )
    )
    v2 = ModelVersion(
        version_id='v2.0.0',
        model_path='v2.pt',
        schema=ModelSchema(
            input_shapes={'x': (None, 10)},
            input_dtypes={'x': 'torch.float32'},
            output_shapes={'y': (None, 2)},
            output_dtypes={'y': 'torch.float32'}
        )
    )
    await registry.register_version(v1)
    await registry.register_version(v2)
    # First hot swap
    success, error = await manager.hot_swap('v1.0.0', validate_schema=False)
    assert success, f'Swap failed: {error}'
    active = await pointer.get()
    assert active.version_id == 'v1.0.0'
    print('First hot swap passed')
    # Second hot swap
    success, error = await manager.hot_swap('v2.0.0', validate_schema=True)
    assert success, f'Swap failed: {error}'
    active = await pointer.get()
    assert active.version_id == 'v2.0.0'
    print('Second hot swap passed')
    # Check previous version stored
    prev_id = manager.get_previous_version_id()
    assert prev_id == 'v1.0.0'
    print('Previous version tracking passed')
    # Rollback
    success, error = await manager.rollback()
    assert success, f'Rollback failed: {error}'
    active = await pointer.get()
    assert active.version_id == 'v1.0.0'
    print('Rollback passed')
asyncio.run(test())
"
```
---
### Phase 6: RollbackMonitor (2-3 hours)
**Files to create:** `09_rollback_monitor.py`
**Implementation steps:**
1. Implement `RollbackMonitor` with error tracking
2. Implement sliding window for error rates
3. Implement automatic rollback trigger
4. Add configurable thresholds
**Checkpoint:**
```bash
python -c "
import asyncio
from rollback_monitor import RollbackMonitor
async def test():
    rollback_triggered = []
    async def on_rollback(version_id, reason):
        rollback_triggered.append((version_id, reason))
    monitor = RollbackMonitor(
        error_rate_threshold=0.1,  # 10%
        latency_p99_threshold_ms=1000.0,
        window_size_seconds=5.0,
        consecutive_windows=2
    )
    monitor.set_rollback_callback(on_rollback)
    # Record healthy results
    for _ in range(10):
        await monitor.record_result('v1.0.0', True, 50.0)
    is_healthy, reason = await monitor.check_health('v1.0.0')
    assert is_healthy, f'Should be healthy: {reason}'
    print('Healthy check passed')
    # Record errors to trigger threshold
    for _ in range(5):
        await monitor.record_result('v2.0.0', False, 50.0)
    for _ in range(5):
        await monitor.record_result('v2.0.0', True, 50.0)
    is_healthy, reason = await monitor.check_health('v2.0.0')
    assert not is_healthy, 'Should be unhealthy (50% error rate)'
    print('Unhealthy check passed')
asyncio.run(test())
"
```
---
### Phase 7: VersionRouter (2-3 hours)
**Files to create:** `10_version_router.py`
**Implementation steps:**
1. Implement `VersionRouter` with routing strategies
2. Implement URL path routing
3. Implement header-based routing
4. Implement request tracking integration
**Checkpoint:**
```bash
python -c "
import asyncio
from unittest.mock import MagicMock
from version_router import VersionRouter
from model_registry import ModelRegistry
from atomic_pointer import AtomicVersionPointer
from request_tracker import RequestTracker
from model_version import ModelVersion, VersionStatus
async def test():
    registry = ModelRegistry()
    pointer = AtomicVersionPointer()
    tracker = RequestTracker()
    # Create and set active version
    v1 = ModelVersion(version_id='v1.0.0', model_path='v1.pt')
    v1.loaded_model = object()
    v1.status = VersionStatus.ACTIVE
    await pointer.swap(v1)
    v2 = ModelVersion(version_id='v2.0.0', model_path='v2.pt')
    v2.loaded_model = object()
    v2.status = VersionStatus.STAGED
    await registry.register_version(v1)
    await registry.register_version(v2)
    router = VersionRouter(registry, pointer, tracker)
    # Mock request
    request = MagicMock()
    request.headers = {}
    # Default routing
    version, vid = await router.route_request(request)
    assert vid == 'v1.0.0', 'Should route to active version'
    print('Default routing passed')
    # Header routing
    request.headers = {'X-Model-Version': 'v2.0.0'}
    version, vid = await router.route_request(request)
    assert vid == 'v2.0.0', 'Should route to header version'
    print('Header routing passed')
    # Request tracking
    executed = []
    async def tracked_fn():
        executed.append(True)
        return 'result'
    result = await router.execute_with_tracking('v1.0.0', tracked_fn)
    assert result == 'result'
    assert len(executed) == 1
    print('Request tracking passed')
asyncio.run(test())
"
```
---
### Phase 8: FastAPI Integration (2-3 hours)
**Files to create:** `11_integration.py`
**Implementation steps:**
1. Create FastAPI app with versioned endpoints
2. Integrate with HotSwapManager
3. Add version-specific routes
4. Add admin endpoints for swap/rollback
**Checkpoint:**
```bash
# Start server
uvicorn integration:app --host 0.0.0.0 --port 8000 &
sleep 5
# Test default prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [[0.1, 0.2, 0.3]]}'
# Test version-specific prediction
curl -X POST http://localhost:8000/v/v1.0.0/predict \
  -H "Content-Type: application/json" \
  -d '{"input": [[0.1, 0.2, 0.3]]}'
# Test header routing
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-Model-Version: v1.0.0" \
  -d '{"input": [[0.1, 0.2, 0.3]]}'
# Test version list
curl http://localhost:8000/versions
# Test hot swap
curl -X POST http://localhost:8000/admin/swap/v2.0.0
# Test rollback
curl -X POST http://localhost:8000/admin/rollback
# Kill server
pkill -f uvicorn
```
---
## Test Specification
### Test 1: Schema Compatibility
```python
def test_schema_compatibility_identical():
    """Test that identical schemas are compatible."""
    from schema import ModelSchema, SchemaValidator
    schema1 = ModelSchema(
        input_shapes={'image': (None, 3, 224, 224)},
        input_dtypes={'image': 'torch.float32'},
        output_shapes={'logits': (None, 1000)},
        output_dtypes={'logits': 'torch.float32'}
    )
    result = SchemaValidator.full_compatibility_check(schema1, schema1)
    assert result.is_compatible
    assert not result.has_issues
def test_schema_incompatibility_output_shape():
    """Test that output shape changes are incompatible."""
    from schema import ModelSchema, SchemaValidator
    old = ModelSchema(
        input_shapes={'x': (None, 10)},
        input_dtypes={'x': 'torch.float32'},
        output_shapes={'y': (None, 768)},
        output_dtypes={'y': 'torch.float32'}
    )
    new = ModelSchema(
        input_shapes={'x': (None, 10)},
        input_dtypes={'x': 'torch.float32'},
        output_shapes={'y': (None, 512)},  # Changed!
        output_dtypes={'y': 'torch.float32'}
    )
    result = SchemaValidator.full_compatibility_check(old, new)
    assert not result.is_compatible
    assert any('y' in issue for issue in result.output_issues)
```
### Test 2: Hot Swap Atomicity
```python
@pytest.mark.asyncio
async def test_hot_swap_atomicity():
    """Test that hot swap is atomic - no request sees mixed state."""
    from hot_swap_manager import HotSwapManager
    from model_registry import ModelRegistry
    from atomic_pointer import AtomicVersionPointer
    from request_tracker import RequestTracker
    from model_version import ModelVersion
    registry = ModelRegistry()
    pointer = AtomicVersionPointer()
    tracker = RequestTracker()
    async def loader(v):
        v.loaded_model = object()
        return v
    manager = HotSwapManager(registry, pointer, tracker, loader)
    v1 = ModelVersion(version_id='v1', model_path='v1.pt')
    v2 = ModelVersion(version_id='v2', model_path='v2.pt')
    await registry.register_version(v1)
    await registry.register_version(v2)
    # Perform swap
    success, _ = await manager.hot_swap('v1', validate_schema=False)
    assert success
    # All subsequent reads should return v1
    for _ in range(10):
        current = await pointer.get()
        assert current.version_id == 'v1'
    # Swap to v2
    success, _ = await manager.hot_swap('v2', validate_schema=False)
    assert success
    # All subsequent reads should return v2
    for _ in range(10):
        current = await pointer.get()
        assert current.version_id == 'v2'
```
### Test 3: Request Draining
```python
@pytest.mark.asyncio
async def test_request_draining():
    """Test that drain waits for in-flight requests."""
    from request_tracker import RequestTracker
    import asyncio
    tracker = RequestTracker(drain_timeout_seconds=5.0)
    # Acquire requests
    await tracker.acquire('v1')
    await tracker.acquire('v1')
    assert await tracker.get_count('v1') == 2
    # Start drain
    await tracker.start_drain('v1')
    # Drain should not complete yet
    # Release one request
    await tracker.release('v1')
    assert await tracker.get_count('v1') == 1
    # Release final request (should signal drain)
    async def delayed_release():
        await asyncio.sleep(0.5)
        await tracker.release('v1')
    task = asyncio.create_task(delayed_release())
    drained = await tracker.wait_for_drain('v1')
    await task
    assert drained
    assert await tracker.get_count('v1') == 0
```
### Test 4: Rollback Speed
```python
@pytest.mark.asyncio
async def test_rollback_speed():
    """Test that rollback completes within 5 seconds."""
    import time
    from hot_swap_manager import HotSwapManager
    from model_registry import ModelRegistry
    from atomic_pointer import AtomicVersionPointer
    from request_tracker import RequestTracker
    from model_version import ModelVersion
    registry = ModelRegistry()
    pointer = AtomicVersionPointer()
    tracker = RequestTracker()
    async def loader(v):
        v.loaded_model = object()
        return v
    manager = HotSwapManager(registry, pointer, tracker, loader)
    v1 = ModelVersion(version_id='v1', model_path='v1.pt')
    v2 = ModelVersion(version_id='v2', model_path='v2.pt')
    await registry.register_version(v1)
    await registry.register_version(v2)
    # Initial swap
    await manager.hot_swap('v1', validate_schema=False)
    # Second swap (keeps v1 as previous)
    await manager.hot_swap('v2', validate_schema=False)
    # Rollback
    start = time.perf_counter()
    success, _ = await manager.rollback()
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert success
    assert elapsed_ms < 5000, f"Rollback took {elapsed_ms}ms, should be < 5000ms"
    current = await pointer.get()
    assert current.version_id == 'v1'
```
### Test 5: Version Routing
```python
@pytest.mark.asyncio
async def test_version_routing_priority():
    """Test routing priority: URL > Header > Default."""
    from version_router import VersionRouter
    from model_registry import ModelRegistry
    from atomic_pointer import AtomicVersionPointer
    from request_tracker import RequestTracker
    from model_version import ModelVersion, VersionStatus
    from unittest.mock import MagicMock
    registry = ModelRegistry()
    pointer = AtomicVersionPointer()
    tracker = RequestTracker()
    v1 = ModelVersion(version_id='v1', model_path='v1.pt')
    v1.loaded_model = object()
    v1.status = VersionStatus.ACTIVE
    v2 = ModelVersion(version_id='v2', model_path='v2.pt')
    v2.loaded_model = object()
    v2.status = VersionStatus.STAGED
    await pointer.swap(v1)
    await registry.register_version(v1)
    await registry.register_version(v2)
    router = VersionRouter(registry, pointer, tracker)
    # Default
    request = MagicMock()
    request.headers = {}
    _, vid = await router.route_request(request)
    assert vid == 'v1'
    # Header override
    request.headers = {'X-Model-Version': 'v2'}
    _, vid = await router.route_request(request)
    assert vid == 'v2'
    # URL override (highest priority)
    _, vid = await router.route_request(request, requested_version='v1')
    assert vid == 'v1'
```
---
## Performance Targets
| Operation | Target | How to Measure |
|-----------|--------|----------------|
| Hot swap (no requests) | < 1 second | Time `hot_swap()` with no in-flight requests |
| Rollback | < 5 seconds | Time `rollback()` - previous already in memory |
| Pointer swap | < 1ms | Time `pointer.swap()` alone |
| Schema validation | < 10ms | Time `SchemaValidator.full_compatibility_check()` |
| Peak memory during swap | 2.5x single model | Monitor GPU memory during dual-load |
| Drain completion | < 30 seconds | Time from `start_drain()` to `wait_for_drain()` return |
| Version routing | < 1ms | Time `router.route_request()` |
---
## State Machine: Version Lifecycle
```
                    ┌─────────────┐
                    │   CREATED   │
                    │ (not loaded)│
                    └──────┬──────┘
                           │ register_version()
                           ▼
              ┌────────────────────────┐
              │        STAGED          │
              │  (loaded, not active)  │
              └───────────┬────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
        hot_swap()              load failure
              │                       │
              ▼                       ▼
       ┌─────────────┐         ┌────────────┐
       │   ACTIVE    │         │   FAILED   │
       │ (serving)   │         │ (error)    │
       └──────┬──────┘         └────────────┘
              │
         replaced by
         new version
              │
              ▼
       ┌─────────────┐
       │  DRAINING   │◄──────────────┐
       │ (waiting)   │               │
       └──────┬──────┘               │
              │                      │
      in-flight == 0           more requests
              │                      │
              ▼                      │
       ┌─────────────┐               │
       │ DEPRECATED  │───────────────┘
       │ (unloaded)  │  (if new request arrives)
       └─────────────┘
ILLEGAL TRANSITIONS:
- ACTIVE → STAGED (must go through DRAINING)
- FAILED → ACTIVE (must reload)
- DEPRECATED → ACTIVE (must reload)
- DRAINING → STAGED (invalid state)
```
---

![Atomic Pointer Swap Mechanism](./diagrams/tdd-diag-021.svg)

---
## Concurrency Specification
### Threading Model
```
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Event Loop                          │
│                     (Single-threaded async)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  /predict    │  │  /admin/     │  │  /health     │          │
│  │  handlers    │  │  endpoints   │  │  probes      │          │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘          │
│         │                 │                                      │
│         │                 │                                      │
│         ▼                 ▼                                      │
│  ┌──────────────────────────────────────────────────┐          │
│  │           AtomicVersionPointer                    │          │
│  │  - asyncio.Lock for swap operations              │          │
│  │  - Read is lock-free (fast path)                 │          │
│  │  - Write requires exclusive lock                 │          │
│  └──────────────────────────────────────────────────┘          │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────┐          │
│  │              RequestTracker                       │          │
│  │  - asyncio.Lock for count updates                │          │
│  │  - asyncio.Event for drain signaling             │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                 │
│  ┌──────────────────────────────────────────────────┐          │
│  │              ModelRegistry                        │          │
│  │  - asyncio.Lock for all mutations                │          │
│  │  - OrderedDict maintains insertion order         │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
### Thread Safety Guarantees
| Component | Thread Safety | Reason |
|-----------|--------------|--------|
| `AtomicVersionPointer` | Safe for async | asyncio.Lock protects mutations |
| `RequestTracker` | Safe for async | asyncio.Lock for counts, Events for signaling |
| `ModelRegistry` | Safe for async | asyncio.Lock for all operations |
| `HotSwapManager` | Safe for async | Uses swap_lock for serialization |
| `ModelVersion` | Safe | Immutable after creation (except status) |
| `SchemaValidator` | Safe | Stateless, no mutation |
### Lock Ordering (for deadlock prevention)
If nested locking is required:
1. `ModelRegistry._lock` (outermost)
2. `AtomicVersionPointer._lock`
3. `RequestTracker._lock` (innermost)
**Critical rule:** Never hold a lock across an await point that might acquire another lock. Release and re-acquire if needed.
---

![Request Draining State Machine](./diagrams/tdd-diag-022.svg)

---

![Schema Compatibility Validation](./diagrams/tdd-diag-023.svg)

---

![Automatic Rollback Trigger Flow](./diagrams/tdd-diag-024.svg)

---

![Version Routing Architecture](./diagrams/tdd-diag-025.svg)

---
[[CRITERIA_JSON: {"module_id": "ml-model-serving-m3", "criteria": ["Model registry stores multiple versions with metadata including accuracy metrics, training date, training data hash, and input/output schema definitions via ModelVersion dataclass", "Version-specific routing directs requests to specified version via URL path (/v/{version_id}/predict) or X-Model-Version header with fallback to active version", "Default routing sends requests to active version when no version is specified via VersionRouter.route_request()", "Hot swap loads new model version into memory, validates schema compatibility via SchemaValidator, and atomically swaps pointer with no dropped requests", "In-flight request tracking via RequestTracker.acquire()/release() ensures requests complete before model unload with configurable drain timeout", "Rollback reverts to previous version within 5 seconds by keeping previous version in memory and swapping pointer atomically via HotSwapManager.rollback()", "Schema compatibility validation checks input/output shapes and dtypes before promotion; incompatible schemas are rejected with SchemaCompatibilityResult", "RequestTracker.wait_for_drain() blocks until in-flight count reaches zero or timeout expires with asyncio.Event signaling", "Memory budget during hot swap accounts for old model + new model + activation buffers (~2.5x single-model memory)", "Version status lifecycle via VersionStatus enum: STAGED → ACTIVE → DRAINING → DEPRECATED with proper state transitions", "AtomicVersionPointer.swap() performs atomic pointer swap in < 1ms with asyncio.Lock protection", "ModelVersion.is_ready_for_inference() returns True only when status is ACTIVE and loaded_model is not None", "SchemaValidator.check_input_compatibility() allows more permissive inputs but requires exact dtype match", "SchemaValidator.check_output_compatibility() requires exact shape and dtype match for all outputs", "HotSwapManager.hot_swap() validates schema compatibility when validate_schema=True before proceeding with swap"]}]
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: ml-model-serving-m4 -->
# A/B Testing & Canary Deployment
## Module Charter
This module implements statistically rigorous A/B testing infrastructure with deterministic traffic splitting via consistent hashing (SHA-256). It provides a TrafficSplitter that routes requests to model versions based on configurable percentage weights, ensuring the same user always sees the same version. The CanaryRolloutController manages gradual staged rollouts (5% → 10% → 25% → 50% → 100%) with automatic rollback when canary error rate exceeds baseline by configurable thresholds. PerVersionMetrics collects latency percentiles, error rates, and prediction distributions per version. The StatisticalAnalyzer provides t-tests for latency comparison and sample size calculations for experiment planning. ExperimentConfig and PredictionLog provide full audit trails for post-hoc analysis. It explicitly does NOT handle drift detection (M5), baseline storage, or model loading (M1). Upstream dependencies are the version router from M3 and HTTP handlers; downstream consumers are monitoring dashboards and alerting systems. The key invariants are: (1) same user_id always routes to same version for fixed weights, (2) traffic weights always sum to 100%, (3) canary rollback triggers within 3 consecutive health check failures, (4) experiment logs capture full context for every prediction.
---
## File Structure
Create files in this exact sequence:
```
ml-model-serving/m4_ab_testing/
├── 01_config.py              # A/B testing configuration constants
├── 02_exceptions.py          # A/B testing exception hierarchy
├── 03_traffic_allocation.py  # TrafficAllocation dataclass
├── 04_traffic_splitter.py    # TrafficSplitter with consistent hashing
├── 05_per_version_metrics.py # PerVersionMetrics with latency tracking
├── 06_statistical_analyzer.py# StatisticalAnalyzer with t-test
├── 07_canary_config.py       # RolloutConfig and RolloutStage enum
├── 08_canary_controller.py   # CanaryRolloutController with stages
├── 09_health_checker.py      # CanaryHealthChecker with thresholds
├── 10_experiment_config.py   # ExperimentConfig and PredictionLog
├── 11_experiment_manager.py  # ExperimentManager coordinating all
└── 12_integration.py         # FastAPI integration with A/B endpoints
```
---
## Complete Data Model
### TrafficAllocation Dataclass
```python
# 03_traffic_allocation.py
from dataclasses import dataclass
from typing import Optional
@dataclass(frozen=True)
class TrafficAllocation:
    """
    Defines how traffic is split to a model version.
    Immutable to ensure traffic weights can't be accidentally modified.
    Fields:
        version_id: Model version to route traffic to
        weight_percent: Percentage of traffic (0.0 to 100.0)
        description: Human-readable description for dashboards
    Invariants:
        - weight_percent must be >= 0 and <= 100
        - version_id must be non-empty string
    """
    version_id: str
    weight_percent: float
    description: str = ""
    def __post_init__(self):
        if not self.version_id:
            raise ValueError("version_id cannot be empty")
        if not (0 <= self.weight_percent <= 100):
            raise ValueError(
                f"weight_percent must be between 0 and 100, "
                f"got {self.weight_percent}"
            )
    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "weight_percent": self.weight_percent,
            "description": self.description,
        }
```
---
### TrafficSplitter Dataclass
```python
# 04_traffic_splitter.py
import hashlib
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import time
import logging
from .03_traffic_allocation import TrafficAllocation
from .02_exceptions import InvalidTrafficWeights
logger = logging.getLogger(__name__)
@dataclass
class TrafficSplitter:
    """
    Deterministic traffic splitter using consistent hashing.
    Guarantees:
    1. Same user_id always routes to same version (for fixed weights)
    2. Weight changes take effect immediately for new requests
    3. Routing decisions are reproducible and auditable
    Algorithm:
        user_hash = SHA256(user_id)[:8] % 100  # 0-99 percentile
        Walk allocations accumulating weights until user_hash < cumulative
    Invariants:
        - Allocations always sum to 100%
        - Sorted by version_id for deterministic ordering
    """
    allocations: List[TrafficAllocation] = field(default_factory=list)
    hash_algorithm: str = "sha256"
    _total_weight: float = field(default=0.0, init=False, repr=False)
    def __post_init__(self):
        self._validate_and_sort()
    def _validate_and_sort(self) -> None:
        """Validate weights sum to 100% and sort by version_id."""
        if not self.allocations:
            self._total_weight = 0.0
            return
        total = sum(a.weight_percent for a in self.allocations)
        if abs(total - 100.0) > 0.01:
            raise InvalidTrafficWeights(
                f"Traffic weights must sum to 100%, got {total}%"
            )
        # Sort by version_id for deterministic ordering
        object.__setattr__(
            self, 
            'allocations', 
            sorted(self.allocations, key=lambda a: a.version_id)
        )
        self._total_weight = total
    def _hash_user(self, user_id: str) -> int:
        """
        Hash a user ID to a number 0-99 (percentile bucket).
        Uses SHA-256 for cryptographic quality distribution.
        """
        if not user_id:
            user_id = f"anonymous_{time.time()}"
        hash_bytes = hashlib.sha256(
            user_id.encode('utf-8')
        ).digest()
        # Take first 8 bytes, convert to int
        hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')
        # Map to 0-99 percentile
        return hash_int % 100
    def route(self, user_id: str) -> Tuple[str, Dict]:
        """
        Determine which version should handle this user's request.
        Args:
            user_id: Unique identifier for the user
        Returns:
            (version_id, routing_metadata)
            metadata includes:
                - user_hash: Computed percentile bucket
                - allocation_reason: Why this version was selected
                - allocation_weight: Weight for this version
                - experiment_timestamp: When routing decision was made
        """
        user_hash = self._hash_user(user_id)
        # Walk through allocations, accumulating weights
        cumulative = 0.0
        for allocation in self.allocations:
            cumulative += allocation.weight_percent
            if user_hash < cumulative:
                return allocation.version_id, {
                    "user_hash": user_hash,
                    "allocation_reason": (
                        f"hash {user_hash} < {cumulative}% threshold"
                    ),
                    "allocation_weight": allocation.weight_percent,
                    "experiment_timestamp": time.time(),
                }
        # Fallback (shouldn't happen if weights sum to 100%)
        return self.allocations[-1].version_id, {
            "user_hash": user_hash,
            "allocation_reason": "fallback",
            "error": "Weight calculation error",
        }
    def update_weights(
        self, 
        new_allocations: List[TrafficAllocation]
    ) -> None:
        """
        Update traffic weights at runtime.
        This is O(1) for routing - the allocation list is just replaced.
        In-flight requests are unaffected; only new requests use new weights.
        Args:
            new_allocations: New traffic allocations
        Raises:
            InvalidTrafficWeights: If weights don't sum to 100%
        """
        total = sum(a.weight_percent for a in new_allocations)
        if abs(total - 100.0) > 0.01:
            raise InvalidTrafficWeights(
                f"Weights must sum to 100%, got {total}%"
            )
        object.__setattr__(
            self, 
            'allocations', 
            sorted(new_allocations, key=lambda a: a.version_id)
        )
        self._total_weight = total
        logger.info(
            f"[TrafficSplitter] Updated weights: "
            f"{[(a.version_id, a.weight_percent) for a in self.allocations]}"
        )
    def get_allocation_for_version(
        self, 
        version_id: str
    ) -> Optional[TrafficAllocation]:
        """Get the current allocation for a specific version."""
        for allocation in self.allocations:
            if allocation.version_id == version_id:
                return allocation
        return None
    def to_dict(self) -> dict:
        """Export current configuration."""
        return {
            "allocations": [a.to_dict() for a in self.allocations],
            "hash_algorithm": self.hash_algorithm,
        }
    @classmethod
    def from_dict(cls, data: dict) -> "TrafficSplitter":
        """Create from configuration dict."""
        allocations = [
            TrafficAllocation(**a) 
            for a in data["allocations"]
        ]
        return cls(
            allocations=allocations,
            hash_algorithm=data.get("hash_algorithm", "sha256"),
        )
```
---
### PerVersionMetrics Dataclass
```python
# 05_per_version_metrics.py
from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import statistics
import time
@dataclass
class LatencySample:
    """A single latency measurement."""
    timestamp: float
    latency_ms: float
    version_id: str
    success: bool
@dataclass
class PredictionSample:
    """A single prediction output sample."""
    timestamp: float
    prediction: float
    version_id: str
    input_hash: str
@dataclass
class PerVersionMetrics:
    """
    Collects and aggregates metrics per model version.
    Key design: samples are stored per-version, allowing
    independent analysis of each version's performance.
    Sliding windows bound memory usage.
    Invariants:
        - Each version has independent latency/prediction windows
        - Windows are bounded by window_size
        - Counters are monotonically increasing
    """
    latency_window_size: int = 10000
    prediction_window_size: int = 10000
    # Per-version latency samples
    _latency_samples: Dict[str, deque] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=10000))
    )
    # Per-version prediction samples
    _prediction_samples: Dict[str, deque] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=10000))
    )
    # Counters
    _request_counts: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    _error_counts: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    # Start time for rate calculations
    _start_time: float = field(default_factory=time.time)
    def record_latency(
        self,
        version_id: str,
        latency_ms: float,
        success: bool = True,
    ) -> None:
        """Record a latency sample for a version."""
        sample = LatencySample(
            timestamp=time.time(),
            latency_ms=latency_ms,
            version_id=version_id,
            success=success,
        )
        self._latency_samples[version_id].append(sample)
        self._request_counts[version_id] += 1
        if not success:
            self._error_counts[version_id] += 1
    def record_prediction(
        self,
        version_id: str,
        prediction: float,
        input_hash: str = "",
    ) -> None:
        """Record a prediction output for distribution analysis."""
        sample = PredictionSample(
            timestamp=time.time(),
            prediction=prediction,
            version_id=version_id,
            input_hash=input_hash,
        )
        self._prediction_samples[version_id].append(sample)
    def get_latency_percentiles(
        self,
        version_id: str,
        percentiles: List[float] = None,
    ) -> Dict[str, float]:
        """Calculate latency percentiles for a version."""
        if percentiles is None:
            percentiles = [50, 90, 95, 99]
        samples = self._latency_samples.get(version_id, deque())
        if not samples:
            return {f"p{p}": 0.0 for p in percentiles}
        # Extract successful latencies only
        latencies = sorted(
            [s.latency_ms for s in samples if s.success]
        )
        if not latencies:
            return {f"p{p}": 0.0 for p in percentiles}
        result = {}
        for p in percentiles:
            idx = int(len(latencies) * p / 100)
            idx = min(idx, len(latencies) - 1)
            result[f"p{p}"] = latencies[idx]
        return result
    def get_error_rate(self, version_id: str) -> float:
        """Calculate error rate for a version (0.0 to 1.0)."""
        total = self._request_counts.get(version_id, 0)
        if total == 0:
            return 0.0
        errors = self._error_counts.get(version_id, 0)
        return errors / total
    def get_throughput_rps(self, version_id: str) -> float:
        """Calculate requests per second for a version."""
        total = self._request_counts.get(version_id, 0)
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0
        return total / elapsed
    def get_prediction_distribution(
        self,
        version_id: str,
        n_bins: int = 20,
    ) -> Dict[str, float]:
        """Get histogram of prediction values."""
        samples = self._prediction_samples.get(version_id, deque())
        if not samples:
            return {}
        predictions = [s.prediction for s in samples]
        if not predictions:
            return {}
        min_val = min(predictions)
        max_val = max(predictions)
        if min_val == max_val:
            return {"bin_0": 1.0, "min": min_val, "max": max_val}
        bin_width = (max_val - min_val) / n_bins
        bins = [0] * n_bins
        for pred in predictions:
            bin_idx = int((pred - min_val) / bin_width)
            bin_idx = min(bin_idx, n_bins - 1)
            bins[bin_idx] += 1
        total = len(predictions)
        return {
            **{f"bin_{i}": bins[i] / total for i in range(n_bins)},
            "min": min_val,
            "max": max_val,
            "mean": statistics.mean(predictions),
            "std": statistics.stdev(predictions) if len(predictions) > 1 else 0.0,
        }
    def compare_versions(
        self,
        version_a: str,
        version_b: str,
    ) -> dict:
        """Compare metrics between two versions."""
        metrics_a = {
            "latency": self.get_latency_percentiles(version_a),
            "error_rate": self.get_error_rate(version_a),
            "throughput": self.get_throughput_rps(version_a),
            "request_count": self._request_counts.get(version_a, 0),
        }
        metrics_b = {
            "latency": self.get_latency_percentiles(version_b),
            "error_rate": self.get_error_rate(version_b),
            "throughput": self.get_throughput_rps(version_b),
            "request_count": self._request_counts.get(version_b, 0),
        }
        # Calculate differences
        latency_diff = {}
        for key in metrics_a["latency"]:
            if metrics_b["latency"][key] > 0:
                diff = (
                    (metrics_b["latency"][key] - metrics_a["latency"][key])
                    / metrics_a["latency"][key] * 100
                )
                latency_diff[key] = diff
        return {
            "version_a": version_a,
            "version_b": version_b,
            "latency_a": metrics_a["latency"],
            "latency_b": metrics_b["latency"],
            "latency_diff_percent": latency_diff,
            "error_rate_a": metrics_a["error_rate"],
            "error_rate_b": metrics_b["error_rate"],
            "error_rate_diff": metrics_b["error_rate"] - metrics_a["error_rate"],
            "throughput_a": metrics_a["throughput"],
            "throughput_b": metrics_b["throughput"],
            "sample_count_a": metrics_a["request_count"],
            "sample_count_b": metrics_b["request_count"],
        }
    def get_all_version_metrics(self) -> Dict[str, dict]:
        """Get metrics summary for all tracked versions."""
        all_versions = set(self._request_counts.keys())
        return {
            version_id: {
                "latency_percentiles": self.get_latency_percentiles(version_id),
                "error_rate": self.get_error_rate(version_id),
                "throughput_rps": self.get_throughput_rps(version_id),
                "request_count": self._request_counts.get(version_id, 0),
                "error_count": self._error_counts.get(version_id, 0),
            }
            for version_id in all_versions
        }
    def to_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        for version_id, metrics in self.get_all_version_metrics().items():
            for p_name, value in metrics["latency_percentiles"].items():
                lines.append(
                    f'model_latency_ms{{percentile="{p_name}",version="{version_id}"}} {value:.2f}'
                )
            lines.append(
                f'model_error_rate{{version="{version_id}"}} {metrics["error_rate"]:.4f}'
            )
            lines.append(
                f'model_throughput_rps{{version="{version_id}"}} {metrics["throughput_rps"]:.2f}'
            )
            lines.append(
                f'model_requests_total{{version="{version_id}"}} {metrics["request_count"]}'
            )
        return "\n".join(lines)
```
---
### StatisticalAnalyzer Class
```python
# 06_statistical_analyzer.py
import math
from dataclasses import dataclass
from typing import Tuple, Optional, List
from enum import Enum
import statistics
class TestType(Enum):
    """Types of statistical tests."""
    TWO_TAILED = "two_tailed"
    ONE_TAILED_BETTER = "one_better"
    ONE_TAILED_WORSE = "one_worse"
@dataclass
class SampleStatistics:
    """Summary statistics for a sample."""
    mean: float
    std_dev: float
    n: int
    @property
    def std_error(self) -> float:
        """Standard error of the mean."""
        if self.n <= 1:
            return 0.0
        return self.std_dev / math.sqrt(self.n)
@dataclass
class TestResult:
    """Result of a statistical test."""
    test_type: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_interval: Tuple[float, float]
    effect_size: float
    interpretation: str
class StatisticalAnalyzer:
    """
    Performs statistical significance testing for A/B experiments.
    Supports:
    - t-test for comparing means (e.g., latency)
    - Proportion test for comparing rates (e.g., error rate)
    - Sample size calculations for experiment planning
    """
    def __init__(
        self,
        significance_level: float = 0.05,
        power: float = 0.80,
    ):
        self.alpha = significance_level
        self.power = power
    @staticmethod
    def compute_sample_stats(values: List[float]) -> SampleStatistics:
        """Compute summary statistics from a list of values."""
        if not values:
            return SampleStatistics(mean=0.0, std_dev=0.0, n=0)
        n = len(values)
        mean = statistics.mean(values)
        if n <= 1:
            return SampleStatistics(mean=mean, std_dev=0.0, n=n)
        std_dev = statistics.stdev(values)
        return SampleStatistics(mean=mean, std_dev=std_dev, n=n)
    def two_sample_t_test(
        self,
        sample_a: SampleStatistics,
        sample_b: SampleStatistics,
        test_type: TestType = TestType.TWO_TAILED,
    ) -> TestResult:
        """
        Perform a two-sample t-test.
        Tests whether the means of two groups are significantly different.
        """
        # Compute standard error of difference
        se_diff = math.sqrt(
            sample_a.std_error ** 2 + 
            sample_b.std_error ** 2
        )
        if se_diff == 0:
            return TestResult(
                test_type="t_test",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                confidence_interval=(0.0, 0.0),
                effect_size=0.0,
                interpretation="No variance in samples",
            )
        # Compute t-statistic
        t_stat = (sample_a.mean - sample_b.mean) / se_diff
        # Degrees of freedom (Welch's formula)
        df_numerator = (
            sample_a.std_error ** 2 + 
            sample_b.std_error ** 2
        ) ** 2
        df_denominator = (
            (sample_a.std_error ** 4) / (sample_a.n - 1) +
            (sample_b.std_error ** 4) / (sample_b.n - 1)
        )
        if df_denominator == 0:
            df = sample_a.n + sample_b.n - 2
        else:
            df = df_numerator / df_denominator
        # Compute p-value
        if df >= 30:
            p_value = self._normal_p_value(t_stat, test_type)
        else:
            p_value = self._t_p_value_approx(t_stat, df, test_type)
        # Effect size (Cohen's d)
        pooled_std = math.sqrt(
            ((sample_a.n - 1) * sample_a.std_dev ** 2 +
             (sample_b.n - 1) * sample_b.std_dev ** 2) /
            max(1, sample_a.n + sample_b.n - 2)
        )
        cohens_d = (
            (sample_a.mean - sample_b.mean) / pooled_std
            if pooled_std > 0 else 0.0
        )
        # Confidence interval
        t_critical = self._t_critical_value(df)
        margin = t_critical * se_diff
        diff = sample_a.mean - sample_b.mean
        is_significant = p_value < self.alpha
        if is_significant:
            if diff > 0:
                interpretation = (
                    f"Group A is significantly higher than B "
                    f"(diff={diff:.4f}, p={p_value:.4f})"
                )
            else:
                interpretation = (
                    f"Group B is significantly higher than A "
                    f"(diff={-diff:.4f}, p={p_value:.4f})"
                )
        else:
            interpretation = (
                f"No significant difference detected (p={p_value:.4f})"
            )
        return TestResult(
            test_type="two_sample_t_test",
            statistic=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            confidence_interval=(diff - margin, diff + margin),
            effect_size=cohens_d,
            interpretation=interpretation,
        )
    def _normal_p_value(self, z: float, test_type: TestType) -> float:
        """Compute p-value using standard normal distribution."""
        def norm_cdf(x: float) -> float:
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        if test_type == TestType.TWO_TAILED:
            return 2 * (1 - norm_cdf(abs(z)))
        elif test_type == TestType.ONE_TAILED_BETTER:
            return 1 - norm_cdf(z)
        else:
            return norm_cdf(z)
    def _t_p_value_approx(
        self, 
        t: float, 
        df: float, 
        test_type: TestType
    ) -> float:
        """Approximate p-value for t-distribution."""
        correction = math.sqrt(df / (df - 2)) if df > 2 else 1.0
        z_corrected = t / correction
        return self._normal_p_value(z_corrected, test_type)
    def _t_critical_value(self, df: float) -> float:
        """Get critical t-value for current significance level."""
        if df >= 30:
            return 1.96
        elif df >= 20:
            return 2.086
        elif df >= 10:
            return 2.228
        elif df >= 5:
            return 2.571
        else:
            return 3.182
    def compute_required_sample_size(
        self,
        baseline_mean: float,
        baseline_std: float,
        minimum_detectable_effect: float,
        test_type: TestType = TestType.TWO_TAILED,
    ) -> int:
        """
        Calculate required sample size for detecting an effect.
        Returns sample size per group.
        """
        if test_type == TestType.TWO_TAILED:
            z_alpha = 1.96
        else:
            z_alpha = 1.645
        z_beta = 0.84  # For 80% power
        if minimum_detectable_effect == 0:
            return float('inf')
        n = (
            2 * (z_alpha + z_beta) ** 2 * baseline_std ** 2 /
            minimum_detectable_effect ** 2
        )
        return int(math.ceil(n))
    def proportion_test(
        self,
        successes_a: int,
        trials_a: int,
        successes_b: int,
        trials_b: int,
    ) -> TestResult:
        """Perform a two-proportion z-test."""
        if trials_a == 0 or trials_b == 0:
            return TestResult(
                test_type="proportion_z_test",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                confidence_interval=(0.0, 0.0),
                effect_size=0.0,
                interpretation="No samples in one or both groups",
            )
        p_a = successes_a / trials_a
        p_b = successes_b / trials_b
        p_pooled = (successes_a + successes_b) / (trials_a + trials_b)
        se = math.sqrt(
            p_pooled * (1 - p_pooled) * (1/trials_a + 1/trials_b)
        )
        if se == 0:
            return TestResult(
                test_type="proportion_z_test",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                confidence_interval=(0.0, 0.0),
                effect_size=0.0,
                interpretation="No variance in proportions",
            )
        z_stat = (p_a - p_b) / se
        p_value = self._normal_p_value(z_stat, TestType.TWO_TAILED)
        diff = p_a - p_b
        z_crit = 1.96
        se_diff = math.sqrt(
            p_a * (1 - p_a) / trials_a + 
            p_b * (1 - p_b) / trials_b
        )
        margin = z_crit * se_diff
        is_significant = p_value < self.alpha
        interpretation = (
            f"Proportion A: {p_a:.4f}, B: {p_b:.4f}, "
            f"diff: {diff:.4f}, p: {p_value:.4f}"
        )
        return TestResult(
            test_type="proportion_z_test",
            statistic=z_stat,
            p_value=p_value,
            is_significant=is_significant,
            confidence_interval=(diff - margin, diff + margin),
            effect_size=diff,
            interpretation=interpretation,
        )
```
---
### RolloutStage Enum and RolloutConfig
```python
# 07_canary_config.py
from dataclasses import dataclass, field
from typing import List
from enum import Enum
class RolloutStage(Enum):
    """Stages of a canary rollout."""
    INITIALIZING = "initializing"
    STAGE_1 = "stage_1"      # 5%
    STAGE_2 = "stage_2"      # 10%
    STAGE_3 = "stage_3"      # 25%
    STAGE_4 = "stage_4"      # 50%
    STAGE_5 = "stage_5"      # 100%
    COMPLETE = "complete"
    ROLLED_BACK = "rolled_back"
@dataclass
class RolloutConfig:
    """Configuration for a canary rollout."""
    stable_version: str
    canary_version: str
    stages: List[float] = field(
        default_factory=lambda: [5.0, 10.0, 25.0, 50.0, 100.0]
    )
    stage_duration_seconds: float = 300.0
    health_check_interval_seconds: float = 30.0
    # Rollback triggers
    max_error_rate_multiplier: float = 2.0
    max_latency_multiplier: float = 1.5
    min_samples_per_stage: int = 100
    def __post_init__(self):
        if not self.stages:
            raise ValueError("stages cannot be empty")
        if not all(0 < s <= 100 for s in self.stages):
            raise ValueError("stages must be between 0 and 100")
        if self.stages != sorted(self.stages):
            raise ValueError("stages must be in ascending order")
@dataclass
class RolloutState:
    """Current state of a canary rollout."""
    stage: RolloutStage = RolloutStage.INITIALIZING
    current_canary_percent: float = 0.0
    stage_start_time: float = None
    samples_this_stage: int = 0
    last_health_check: float = None
    health_check_passed: bool = True
    rollback_reason: str = None
```
---
### CanaryRolloutController Class
```python
# 08_canary_controller.py
import asyncio
import time
import logging
from typing import Callable, Awaitable, Optional
from .04_traffic_splitter import TrafficSplitter, TrafficAllocation
from .05_per_version_metrics import PerVersionMetrics
from .07_canary_config import RolloutConfig, RolloutState, RolloutStage
from .02_exceptions import CanaryHealthCheckFailed, RollbackFailed
logger = logging.getLogger(__name__)
HealthChecker = Callable[[str, str], Awaitable[tuple]]
class CanaryRolloutController:
    """
    Orchestrates gradual canary rollouts with automatic health checking.
    The rollout progresses through stages only if:
    1. Minimum samples collected
    2. Health checks pass
    3. Stage duration elapsed
    At any point, health check failure triggers automatic rollback.
    Invariants:
        - Only one rollout can be active at a time
        - Traffic weights always sum to 100%
        - Rollback resets traffic to 100% stable
    """
    def __init__(
        self,
        config: RolloutConfig,
        splitter: TrafficSplitter,
        metrics: PerVersionMetrics,
        health_checker: HealthChecker,
        on_stage_change: Optional[Callable[[RolloutStage, float], Awaitable[None]]] = None,
    ):
        self.config = config
        self.splitter = splitter
        self.metrics = metrics
        self.health_checker = health_checker
        self.on_stage_change = on_stage_change
        self.state = RolloutState()
        self._running = False
        self._task: Optional[asyncio.Task] = None
    async def start_rollout(self) -> None:
        """Begin the canary rollout process."""
        if self._running:
            logger.warning("[Rollout] Already running")
            return
        logger.info(
            f"[Rollout] Starting canary deployment: "
            f"{self.config.stable_version} -> {self.config.canary_version}"
        )
        self._running = True
        self.state = RolloutState(stage=RolloutStage.INITIALIZING)
        # Initialize splitter with 0% canary
        self._update_splitter(0.0)
        # Start the rollout loop
        self._task = asyncio.create_task(self._rollout_loop())
    async def _rollout_loop(self) -> None:
        """Main rollout orchestration loop."""
        try:
            for stage_idx, canary_percent in enumerate(self.config.stages):
                # Enter new stage
                self.state.stage = RolloutStage(f"stage_{stage_idx + 1}")
                self.state.current_canary_percent = canary_percent
                self.state.stage_start_time = time.time()
                self.state.samples_this_stage = 0
                logger.info(
                    f"[Rollout] Entering {self.state.stage.value}: "
                    f"canary at {canary_percent}%"
                )
                # Update traffic split
                self._update_splitter(canary_percent)
                # Notify callback
                if self.on_stage_change:
                    await self.on_stage_change(
                        self.state.stage, 
                        canary_percent
                    )
                # Wait for stage duration, checking health periodically
                stage_elapsed = 0.0
                while stage_elapsed < self.config.stage_duration_seconds:
                    await asyncio.sleep(self.config.health_check_interval_seconds)
                    stage_elapsed += self.config.health_check_interval_seconds
                    # Health check
                    is_healthy, reason = await self.health_checker(
                        self.config.stable_version,
                        self.config.canary_version,
                    )
                    self.state.last_health_check = time.time()
                    self.state.health_check_passed = is_healthy
                    if not is_healthy:
                        await self._trigger_rollback(reason)
                        return
                # Check minimum samples
                if self.state.samples_this_stage < self.config.min_samples_per_stage:
                    logger.warning(
                        f"[Rollout] Insufficient samples: "
                        f"{self.state.samples_this_stage} < "
                        f"{self.config.min_samples_per_stage}. Extending stage."
                    )
                    # Extend the stage
                    extension_start = time.time()
                    while (
                        self.state.samples_this_stage < self.config.min_samples_per_stage
                        and time.time() - extension_start < self.config.stage_duration_seconds * 2
                    ):
                        await asyncio.sleep(self.config.health_check_interval_seconds)
                        is_healthy, reason = await self.health_checker(
                            self.config.stable_version,
                            self.config.canary_version,
                        )
                        if not is_healthy:
                            await self._trigger_rollback(reason)
                            return
            # All stages complete
            self.state.stage = RolloutStage.COMPLETE
            logger.info("[Rollout] Canary deployment complete!")
            if self.on_stage_change:
                await self.on_stage_change(RolloutStage.COMPLETE, 100.0)
        except asyncio.CancelledError:
            logger.info("[Rollout] Cancelled")
        except Exception as e:
            logger.error(f"[Rollout] Error: {e}", exc_info=True)
            await self._trigger_rollback(f"Internal error: {e}")
    def _update_splitter(self, canary_percent: float) -> None:
        """Update the traffic splitter with new weights."""
        stable_percent = 100.0 - canary_percent
        self.splitter.update_weights([
            TrafficAllocation(
                version_id=self.config.stable_version,
                weight_percent=stable_percent,
                description="Stable production model",
            ),
            TrafficAllocation(
                version_id=self.config.canary_version,
                weight_percent=canary_percent,
                description="Canary candidate model",
            ),
        ])
    async def _trigger_rollback(self, reason: str) -> None:
        """Trigger immediate rollback to stable version."""
        logger.error(f"[Rollout] ROLLBACK triggered: {reason}")
        self.state.stage = RolloutStage.ROLLED_BACK
        self.state.rollback_reason = reason
        # Route 100% to stable
        self._update_splitter(0.0)
        if self.on_stage_change:
            await self.on_stage_change(RolloutStage.ROLLED_BACK, 0.0)
    def record_sample(self) -> None:
        """Record that a sample was collected."""
        self.state.samples_this_stage += 1
    async def stop(self) -> None:
        """Stop the rollout controller."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    def get_state(self) -> dict:
        """Get current rollout state for monitoring."""
        return {
            "stage": self.state.stage.value,
            "canary_percent": self.state.current_canary_percent,
            "stage_start_time": self.state.stage_start_time,
            "samples_this_stage": self.state.samples_this_stage,
            "health_check_passed": self.state.health_check_passed,
            "rollback_reason": self.state.rollback_reason,
        }
```
---
### CanaryHealthChecker Class
```python
# 09_health_checker.py
import asyncio
import time
import logging
from dataclasses import dataclass
from typing import Callable, Awaitable, Optional, Tuple
from .05_per_version_metrics import PerVersionMetrics
from .02_exceptions import CanaryHealthCheckFailed
logger = logging.getLogger(__name__)
@dataclass
class HealthCheckConfig:
    """Configuration for canary health checks."""
    check_interval_seconds: float = 30.0
    min_samples_required: int = 50
    max_error_rate_multiplier: float = 2.0
    max_latency_p50_multiplier: float = 1.5
    max_latency_p99_multiplier: float = 2.0
    consecutive_failures_threshold: int = 3
class CanaryHealthChecker:
    """
    Monitors canary health and triggers rollback on problems.
    Health checks compare canary metrics against stable baseline.
    """
    def __init__(
        self,
        metrics: PerVersionMetrics,
        config: HealthCheckConfig,
        on_rollback: Callable[[str, str], Awaitable[None]],
    ):
        self.metrics = metrics
        self.config = config
        self.on_rollback = on_rollback
        self._consecutive_failures = 0
        self._last_check_result: Optional[dict] = None
    async def check_health(
        self,
        stable_version: str,
        canary_version: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Perform a health check comparing canary to stable.
        Returns:
            (is_healthy, reason_if_unhealthy)
        """
        stable_latency = self.metrics.get_latency_percentiles(stable_version)
        canary_latency = self.metrics.get_latency_percentiles(canary_version)
        stable_error_rate = self.metrics.get_error_rate(stable_version)
        canary_error_rate = self.metrics.get_error_rate(canary_version)
        stable_count = self.metrics._request_counts.get(stable_version, 0)
        canary_count = self.metrics._request_counts.get(canary_version, 0)
        # Check minimum samples
        if canary_count < self.config.min_samples_required:
            return True, None
        reasons = []
        # Check error rate
        if stable_error_rate > 0:
            error_multiplier = canary_error_rate / stable_error_rate
            if error_multiplier > self.config.max_error_rate_multiplier:
                reasons.append(
                    f"Error rate {canary_error_rate:.2%} is "
                    f"{error_multiplier:.1f}x stable ({stable_error_rate:.2%})"
                )
        elif canary_error_rate > 0.01:
            reasons.append(
                f"Canary error rate {canary_error_rate:.2%} while stable has 0%"
            )
        # Check latency p50
        stable_p50 = stable_latency.get("p50", 0)
        canary_p50 = canary_latency.get("p50", 0)
        if stable_p50 > 0:
            latency_multiplier = canary_p50 / stable_p50
            if latency_multiplier > self.config.max_latency_p50_multiplier:
                reasons.append(
                    f"Latency p50 {canary_p50:.1f}ms is "
                    f"{latency_multiplier:.1f}x stable ({stable_p50:.1f}ms)"
                )
        # Check latency p99
        stable_p99 = stable_latency.get("p99", 0)
        canary_p99 = canary_latency.get("p99", 0)
        if stable_p99 > 0:
            latency_multiplier = canary_p99 / stable_p99
            if latency_multiplier > self.config.max_latency_p99_multiplier:
                reasons.append(
                    f"Latency p99 {canary_p99:.1f}ms is "
                    f"{latency_multiplier:.1f}x stable ({stable_p99:.1f}ms)"
                )
        self._last_check_result = {
            "timestamp": time.time(),
            "stable_error_rate": stable_error_rate,
            "canary_error_rate": canary_error_rate,
            "stable_latency": stable_latency,
            "canary_latency": canary_latency,
            "stable_count": stable_count,
            "canary_count": canary_count,
        }
        if reasons:
            return False, "; ".join(reasons)
        return True, None
    def get_last_check_result(self) -> Optional[dict]:
        """Get the result of the last health check."""
        return self._last_check_result
```
---
### ExperimentConfig and PredictionLog
```python
# 10_experiment_config.py
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import hashlib
import time
@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment."""
    experiment_id: str
    name: str
    description: str
    start_time: datetime
    control_version: str
    treatment_version: str
    traffic_split: float  # Treatment percentage
    hypothesis: str
    success_metric: str
    minimum_sample_size: int
    significance_level: float = 0.05
    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "start_time": self.start_time.isoformat(),
            "control_version": self.control_version,
            "treatment_version": self.treatment_version,
            "traffic_split": self.traffic_split,
            "hypothesis": self.hypothesis,
            "success_metric": self.success_metric,
            "minimum_sample_size": self.minimum_sample_size,
            "significance_level": self.significance_level,
        }
@dataclass
class PredictionLog:
    """Log entry for a single prediction during an experiment."""
    timestamp: float
    experiment_id: str
    request_id: str
    user_id: str
    model_version: str
    allocation_hash: int
    input_hash: str
    prediction: Any
    prediction_class: Optional[str] = None
    prediction_confidence: Optional[float] = None
    latency_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    request_metadata: Dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "experiment_id": self.experiment_id,
            "request_id": self.request_id,
            "user_id": self.user_id,
            "model_version": self.model_version,
            "allocation_hash": self.allocation_hash,
            "input_hash": self.input_hash,
            "prediction": self.prediction,
            "prediction_class": self.prediction_class,
            "prediction_confidence": self.prediction_confidence,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "error_message": self.error_message,
            "request_metadata": self.request_metadata,
        }
    def to_json_line(self) -> str:
        """Convert to JSON line for logging."""
        return json.dumps(self.to_dict())
    @staticmethod
    def hash_features(features: Dict[str, Any]) -> str:
        """Create deterministic hash of features."""
        feature_str = json.dumps(features, sort_keys=True)
        return hashlib.sha256(feature_str.encode()).hexdigest()[:16]
```
---
### Exception Hierarchy
```python
# 02_exceptions.py
class ABTestingError(Exception):
    """Base exception for A/B testing errors."""
    pass
class InvalidTrafficWeights(ABTestingError):
    """Raised when traffic allocations don't sum to 100%."""
    def __init__(self, message: str):
        super().__init__(message)
class ExperimentNotFound(ABTestingError):
    """Raised when requested experiment doesn't exist."""
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        super().__init__(f"Experiment not found: {experiment_id}")
class InsufficientSamples(ABTestingError):
    """Raised when not enough data for statistical test."""
    def __init__(self, required: int, actual: int):
        self.required = required
        self.actual = actual
        super().__init__(
            f"Insufficient samples: need {required}, have {actual}"
        )
class CanaryHealthCheckFailed(ABTestingError):
    """Raised when canary health check fails."""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Canary health check failed: {reason}")
class RollbackFailed(ABTestingError):
    """Raised when automatic rollback cannot complete."""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Rollback failed: {reason}")
```
---
## Interface Contracts
### ExperimentManager
```python
# 11_experiment_manager.py
import asyncio
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
import uuid
from .04_traffic_splitter import TrafficSplitter
from .05_per_version_metrics import PerVersionMetrics
from .06_statistical_analyzer import StatisticalAnalyzer
from .10_experiment_config import ExperimentConfig, PredictionLog
from .02_exceptions import ExperimentNotFound
class ExperimentManager:
    """
    Manages the lifecycle of A/B experiments.
    Responsibilities:
    - Track active experiments
    - Route requests to correct model versions
    - Log predictions with experiment context
    - Compute experiment results
    """
    def __init__(
        self,
        metrics: PerVersionMetrics,
        analyzer: StatisticalAnalyzer,
        log_dir: str = "experiments",
    ):
        self.metrics = metrics
        self.analyzer = analyzer
        self.log_dir = log_dir
        self._active_experiments: Dict[str, ExperimentConfig] = {}
        self._splitters: Dict[str, TrafficSplitter] = {}
        self._log_buffer: list = []
        self._flush_interval = 100
    def start_experiment(self, config: ExperimentConfig) -> None:
        """Register and start a new experiment."""
        if config.experiment_id in self._active_experiments:
            raise ValueError(
                f"Experiment {config.experiment_id} already exists"
            )
        self._active_experiments[config.experiment_id] = config
        # Create traffic splitter for this experiment
        self._splitters[config.experiment_id] = TrafficSplitter(
            allocations=[
                TrafficAllocation(
                    version_id=config.control_version,
                    weight_percent=100.0 - config.traffic_split,
                    description="Control group",
                ),
                TrafficAllocation(
                    version_id=config.treatment_version,
                    weight_percent=config.traffic_split,
                    description="Treatment group",
                ),
            ]
        )
    def route_request(
        self,
        experiment_id: str,
        user_id: str,
    ) -> Tuple[str, dict]:
        """Route a request to the appropriate model version."""
        if experiment_id not in self._splitters:
            raise ExperimentNotFound(experiment_id)
        return self._splitters[experiment_id].route(user_id)
    def log_prediction(
        self,
        experiment_id: str,
        request_id: str,
        user_id: str,
        model_version: str,
        allocation_hash: int,
        input_data: Any,
        prediction: Any,
        latency_ms: float,
        **kwargs,
    ) -> None:
        """Log a prediction with experiment context."""
        log_entry = PredictionLog(
            timestamp=time.time(),
            experiment_id=experiment_id,
            request_id=request_id,
            user_id=user_id,
            model_version=model_version,
            allocation_hash=allocation_hash,
            input_hash=PredictionLog.hash_features(input_data),
            prediction=prediction,
            latency_ms=latency_ms,
            **kwargs,
        )
        self._log_buffer.append(log_entry)
        if len(self._log_buffer) >= self._flush_interval:
            self._flush_logs()
    def _flush_logs(self) -> None:
        """Write buffered logs to disk."""
        if not self._log_buffer:
            return
        import os
        os.makedirs(self.log_dir, exist_ok=True)
        for entry in self._log_buffer:
            log_path = f"{self.log_dir}/{entry.experiment_id}.jsonl"
            with open(log_path, "a") as f:
                f.write(entry.to_json_line() + "\n")
        self._log_buffer = []
    def analyze_experiment(self, experiment_id: str) -> dict:
        """Perform statistical analysis of experiment results."""
        if experiment_id not in self._active_experiments:
            raise ExperimentNotFound(experiment_id)
        config = self._active_experiments[experiment_id]
        comparison = self.metrics.compare_versions(
            config.control_version,
            config.treatment_version,
        )
        min_samples = min(
            comparison["sample_count_a"],
            comparison["sample_count_b"],
        )
        has_sufficient_samples = min_samples >= config.minimum_sample_size
        if not has_sufficient_samples:
            recommendation = (
                f"Continue experiment. "
                f"Current samples: {min_samples}, "
                f"required: {config.minimum_sample_size}"
            )
        else:
            latency_improvement = (
                comparison["latency_a"].get("p50", 0) - 
                comparison["latency_b"].get("p50", 0)
            )
            error_diff = (
                comparison["error_rate_b"] - 
                comparison["error_rate_a"]
            )
            if latency_improvement > 0 and error_diff <= 0:
                recommendation = (
                    "TREATMENT WINS: Lower latency, no error increase. "
                    "Recommend promoting to 100%."
                )
            elif latency_improvement < 0 or error_diff > 0.01:
                recommendation = (
                    "CONTROL WINS: Treatment shows regression. "
                    "Recommend rollback."
                )
            else:
                recommendation = (
                    "NO SIGNIFICANT DIFFERENCE. "
                    "Consider extending experiment."
                )
        return {
            "experiment_id": experiment_id,
            "config": config.to_dict(),
            "has_sufficient_samples": has_sufficient_samples,
            "sample_size": min_samples,
            "comparison": comparison,
            "recommendation": recommendation,
        }
```
---
## Algorithm Specification
### Consistent Hashing Algorithm
```
ALGORITHM: route (TrafficSplitter)
INPUT: user_id (str)
OUTPUT: (version_id, routing_metadata)
1. // Compute hash
   IF user_id is empty:
       user_id ← "anonymous_{current_timestamp}"
   hash_bytes ← SHA256(user_id.encode('utf-8')).digest()
   hash_int ← int.from_bytes(hash_bytes[:8], byteorder='big')
   user_hash ← hash_int % 100  // 0-99 percentile
2. // Walk allocations accumulating weights
   cumulative ← 0.0
   FOR each allocation IN allocations (sorted by version_id):
       cumulative ← cumulative + allocation.weight_percent
       IF user_hash < cumulative:
           RETURN (
               allocation.version_id,
               {
                   user_hash: user_hash,
                   allocation_reason: "hash {user_hash} < {cumulative}% threshold",
                   allocation_weight: allocation.weight_percent,
                   experiment_timestamp: current_time
               }
           )
3. // Fallback (shouldn't reach if weights sum to 100%)
   RETURN (allocations[-1].version_id, {error: "Weight calculation error"})
INVARIANT: Same user_id always maps to same version for fixed allocations
INVARIANT: user_hash is uniformly distributed 0-99
COMPLEXITY: O(n) where n = number of allocations (typically 2)
```
### Canary Stage Progression Algorithm
```
ALGORITHM: _rollout_loop (CanaryRolloutController)
INPUT: RolloutConfig with stages, durations, thresholds
OUTPUT: None (side effects: traffic weight updates)
1. FOR each (stage_idx, canary_percent) IN config.stages:
   a. Update state:
      state.stage ← "stage_{stage_idx + 1}"
      state.current_canary_percent ← canary_percent
      state.stage_start_time ← current_time()
      state.samples_this_stage ← 0
   b. Update traffic splitter:
      splitter.update_weights([
          (stable_version, 100 - canary_percent),
          (canary_version, canary_percent)
      ])
   c. Wait loop (stage_duration seconds):
      WHILE elapsed < config.stage_duration_seconds:
          SLEEP config.health_check_interval_seconds
          (is_healthy, reason) ← health_checker(stable, canary)
          IF NOT is_healthy:
              TRIGGER_ROLLBACK(reason)
              RETURN
          elapsed ← elapsed + config.health_check_interval_seconds
   d. Check minimum samples:
      IF state.samples_this_stage < config.min_samples_per_stage:
          EXTEND stage until samples sufficient OR timeout
2. Mark complete:
   state.stage ← COMPLETE
INVARIANT: Traffic weights always sum to 100%
INVARIANT: Rollback resets to 0% canary
INVARIANT: Health checks run at least once per stage
```
---
## Error Handling Matrix
| Error | Detected By | Recovery | User-Visible? |
|-------|-------------|----------|---------------|
| `InvalidTrafficWeights` | `TrafficSplitter._validate_and_sort()` | Reject update, keep old weights | Yes (400 with details) |
| `ExperimentNotFound` | `ExperimentManager.route_request()` | Return HTTP 404 | Yes (404 "Experiment X not found") |
| `InsufficientSamples` | `ExperimentManager.analyze_experiment()` | Return "continue experiment" recommendation | No (in analysis response) |
| `CanaryHealthCheckFailed` | `CanaryHealthChecker.check_health()` | Trigger automatic rollback | No (internal, triggers rollback) |
| `RollbackFailed` | `CanaryRolloutController._trigger_rollback()` | Log error, alert operators | No (admin alerted) |
| Routing without active experiment | `ExperimentManager.route_request()` | Use default splitter | Indirectly (uses fallback) |
---
## Implementation Sequence with Checkpoints
### Phase 1: TrafficSplitter with Consistent Hashing (2-3 hours)
**Files:** `01_config.py`, `02_exceptions.py`, `03_traffic_allocation.py`, `04_traffic_splitter.py`
**Checkpoint:**
```bash
python -c "
from traffic_splitter import TrafficSplitter, TrafficAllocation
# Create splitter
splitter = TrafficSplitter(allocations=[
    TrafficAllocation('v1', 90.0, 'Stable'),
    TrafficAllocation('v2', 10.0, 'Canary'),
])
# Test deterministic routing
v1, _ = splitter.route('alice')
v1_again, _ = splitter.route('alice')
assert v1 == v1_again, 'Same user should route to same version'
print(f'Alice routes to: {v1}')
# Test different users
v2, _ = splitter.route('bob')
print(f'Bob routes to: {v2}')
# Test weight update
splitter.update_weights([
    TrafficAllocation('v1', 50.0, 'Stable'),
    TrafficAllocation('v2', 50.0, 'Canary'),
])
print('Weight update successful')
"
```
---
### Phase 2: PerVersionMetrics Collection (2-3 hours)
**Files:** `05_per_version_metrics.py`
**Checkpoint:**
```bash
python -c "
from per_version_metrics import PerVersionMetrics
metrics = PerVersionMetrics()
# Record samples
for i in range(100):
    metrics.record_latency('v1', 40 + i * 0.5, success=True)
    metrics.record_latency('v2', 35 + i * 0.3, success=True)
# Get percentiles
p50_v1 = metrics.get_latency_percentiles('v1')['p50']
p50_v2 = metrics.get_latency_percentiles('v2')['p50']
print(f'v1 p50: {p50_v1:.1f}ms, v2 p50: {p50_v2:.1f}ms')
# Compare versions
comparison = metrics.compare_versions('v1', 'v2')
print(f'Latency diff: {comparison[\"latency_diff_percent\"]}')
# Prometheus format
print(metrics.to_prometheus_format()[:200])
"
```
---
### Phase 3: StatisticalAnalyzer (3-4 hours)
**Files:** `06_statistical_analyzer.py`
**Checkpoint:**
```bash
python -c "
from statistical_analyzer import StatisticalAnalyzer, TestType
import random
analyzer = StatisticalAnalyzer(significance_level=0.05)
# Generate samples
control = [100 + random.gauss(0, 10) for _ in range(100)]
treatment = [95 + random.gauss(0, 10) for _ in range(100)]
stats_control = StatisticalAnalyzer.compute_sample_stats(control)
stats_treatment = StatisticalAnalyzer.compute_sample_stats(treatment)
result = analyzer.two_sample_t_test(stats_control, stats_treatment)
print(f'Control mean: {stats_control.mean:.1f}')
print(f'Treatment mean: {stats_treatment.mean:.1f}')
print(f'p-value: {result.p_value:.4f}')
print(f'Significant: {result.is_significant}')
print(f'Interpretation: {result.interpretation}')
# Sample size calculation
n = analyzer.compute_required_sample_size(
    baseline_mean=100,
    baseline_std=10,
    minimum_detectable_effect=5,
)
print(f'Required sample size: {n}')
"
```
---
### Phase 4: CanaryRolloutController (3-4 hours)
**Files:** `07_canary_config.py`, `08_canary_controller.py`
**Checkpoint:**
```bash
python -c "
import asyncio
from canary_controller import CanaryRolloutController, RolloutConfig
from traffic_splitter import TrafficSplitter, TrafficAllocation
from per_version_metrics import PerVersionMetrics
async def test():
    splitter = TrafficSplitter(allocations=[
        TrafficAllocation('v1', 100.0, 'Stable'),
    ])
    metrics = PerVersionMetrics()
    config = RolloutConfig(
        stable_version='v1',
        canary_version='v2',
        stages=[10.0, 50.0, 100.0],
        stage_duration_seconds=2.0,
        health_check_interval_seconds=0.5,
    )
    async def health_checker(stable, canary):
        return True, None
    controller = CanaryRolloutController(
        config=config,
        splitter=splitter,
        metrics=metrics,
        health_checker=health_checker,
    )
    await controller.start_rollout()
    # Wait for first stage
    await asyncio.sleep(3)
    state = controller.get_state()
    print(f'Stage: {state[\"stage\"]}')
    print(f'Canary %: {state[\"canary_percent\"]}')
    await controller.stop()
asyncio.run(test())
"
```
---
### Phase 5: CanaryHealthChecker (2-3 hours)
**Files:** `09_health_checker.py`
**Checkpoint:**
```bash
python -c "
import asyncio
from health_checker import CanaryHealthChecker, HealthCheckConfig
from per_version_metrics import PerVersionMetrics
async def test():
    metrics = PerVersionMetrics()
    # Record healthy data for stable
    for _ in range(100):
        metrics.record_latency('stable', 50.0, success=True)
    # Record degraded data for canary
    for _ in range(50):
        metrics.record_latency('canary', 80.0, success=True)
    for _ in range(10):
        metrics.record_latency('canary', 100.0, success=False)  # Errors
    config = HealthCheckConfig(
        max_error_rate_multiplier=2.0,
        max_latency_p50_multiplier=1.5,
    )
    async def on_rollback(v, r):
        print(f'Rollback triggered: {r}')
    checker = CanaryHealthChecker(metrics, config, on_rollback)
    is_healthy, reason = await checker.check_health('stable', 'canary')
    print(f'Healthy: {is_healthy}')
    print(f'Reason: {reason}')
asyncio.run(test())
"
```
---
### Phase 6: ExperimentManager (2-3 hours)
**Files:** `10_experiment_config.py`, `11_experiment_manager.py`
**Checkpoint:**
```bash
python -c "
import asyncio
from datetime import datetime
from experiment_manager import ExperimentManager
from experiment_config import ExperimentConfig
from per_version_metrics import PerVersionMetrics
from statistical_analyzer import StatisticalAnalyzer
async def test():
    metrics = PerVersionMetrics()
    analyzer = StatisticalAnalyzer()
    manager = ExperimentManager(metrics, analyzer, log_dir='test_logs')
    # Start experiment
    config = ExperimentConfig(
        experiment_id='exp-001',
        name='Test Experiment',
        description='Testing',
        start_time=datetime.now(),
        control_version='v1',
        treatment_version='v2',
        traffic_split=10.0,
        hypothesis='v2 is faster',
        success_metric='latency_p50',
        minimum_sample_size=100,
    )
    manager.start_experiment(config)
    # Route requests
    for user in ['alice', 'bob', 'charlie']:
        version, meta = manager.route_request('exp-001', user)
        print(f'{user} -> {version}')
    # Log predictions
    manager.log_prediction(
        experiment_id='exp-001',
        request_id='req-1',
        user_id='alice',
        model_version='v1',
        allocation_hash=42,
        input_data={'x': [1, 2, 3]},
        prediction=[0.5],
        latency_ms=45.0,
    )
    print('Prediction logged')
asyncio.run(test())
"
```
---
### Phase 7: FastAPI Integration (2-3 hours)
**Files:** `12_integration.py`
**Checkpoint:**
```bash
# Start server
uvicorn integration:app --host 0.0.0.0 --port 8000 &
sleep 5
# Test traffic config
curl http://localhost:8000/traffic
# Update traffic
curl -X POST http://localhost:8000/traffic \
  -H "Content-Type: application/json" \
  -d '{"allocations": [{"version_id": "v1", "weight_percent": 90}, {"version_id": "v2", "weight_percent": 10}]}'
# Start canary
curl -X POST http://localhost:8000/canary/start \
  -H "Content-Type: application/json" \
  -d '{"stable_version": "v1", "canary_version": "v2"}'
# Check status
curl http://localhost:8000/canary/status
# Get metrics
curl http://localhost:8000/metrics
# Kill server
pkill -f uvicorn
```
---
## Test Specification
### Test 1: Deterministic Routing
```python
def test_deterministic_routing():
    """Test that same user always routes to same version."""
    from traffic_splitter import TrafficSplitter, TrafficAllocation
    splitter = TrafficSplitter(allocations=[
        TrafficAllocation('v1', 50.0),
        TrafficAllocation('v2', 50.0),
    ])
    # Same user should always get same result
    results = [splitter.route('alice')[0] for _ in range(100)]
    assert len(set(results)) == 1, "Same user should always route to same version"
    # Different users may route differently
    alice_version = splitter.route('alice')[0]
    bob_version = splitter.route('bob')[0]
    # Both should be valid versions
    assert alice_version in ['v1', 'v2']
    assert bob_version in ['v1', 'v2']
```
### Test 2: Weight Validation
```python
def test_weight_validation():
    """Test that invalid weights are rejected."""
    from traffic_splitter import TrafficSplitter, TrafficAllocation
    from exceptions import InvalidTrafficWeights
    # Weights not summing to 100%
    with pytest.raises(InvalidTrafficWeights):
        TrafficSplitter(allocations=[
            TrafficAllocation('v1', 50.0),
            TrafficAllocation('v2', 30.0),  # Total 80%
        ])
    # Valid weights should work
    splitter = TrafficSplitter(allocations=[
        TrafficAllocation('v1', 90.0),
        TrafficAllocation('v2', 10.0),
    ])
    assert len(splitter.allocations) == 2
```
### Test 3: Canary Rollback
```python
@pytest.mark.asyncio
async def test_canary_rollback():
    """Test that unhealthy canary triggers rollback."""
    from canary_controller import CanaryRolloutController, RolloutConfig
    from traffic_splitter import TrafficSplitter, TrafficAllocation
    from per_version_metrics import PerVersionMetrics
    splitter = TrafficSplitter(allocations=[
        TrafficAllocation('v1', 100.0),
    ])
    metrics = PerVersionMetrics()
    # Record bad canary data
    for _ in range(100):
        metrics.record_latency('v1', 50.0, success=True)
    for _ in range(50):
        metrics.record_latency('v2', 100.0, success=False)
    config = RolloutConfig(
        stable_version='v1',
        canary_version='v2',
        stages=[10.0],
        stage_duration_seconds=1.0,
        health_check_interval_seconds=0.2,
    )
    rollback_triggered = []
    async def health_checker(stable, canary):
        return False, "High error rate"
    controller = CanaryRolloutController(
        config=config,
        splitter=splitter,
        metrics=metrics,
        health_checker=health_checker,
    )
    await controller.start_rollout()
    await asyncio.sleep(2)
    state = controller.get_state()
    assert state['stage'] == 'rolled_back'
    await controller.stop()
```
### Test 4: Statistical Significance
```python
def test_statistical_significance():
    """Test that t-test detects significant differences."""
    from statistical_analyzer import StatisticalAnalyzer, TestType
    import random
    analyzer = StatisticalAnalyzer(significance_level=0.05)
    # Clear difference
    control = [100 + random.gauss(0, 5) for _ in range(200)]
    treatment = [90 + random.gauss(0, 5) for _ in range(200)]
    stats_c = StatisticalAnalyzer.compute_sample_stats(control)
    stats_t = StatisticalAnalyzer.compute_sample_stats(treatment)
    result = analyzer.two_sample_t_test(stats_c, stats_t)
    assert result.is_significant
    assert result.p_value < 0.05
    # No difference
    control2 = [100 + random.gauss(0, 5) for _ in range(200)]
    treatment2 = [100 + random.gauss(0, 5) for _ in range(200)]
    stats_c2 = StatisticalAnalyzer.compute_sample_stats(control2)
    stats_t2 = StatisticalAnalyzer.compute_sample_stats(treatment2)
    result2 = analyzer.two_sample_t_test(stats_c2, stats_t2)
    # May or may not be significant, but p-value should be higher
    assert result2.p_value > result.p_value
```
---
## Performance Targets
| Operation | Target | How to Measure |
|-----------|--------|----------------|
| Routing decision | < 1ms | Time `splitter.route()` |
| Weight update | < 1ms | Time `splitter.update_weights()` |
| Latency percentile calculation | < 5ms | Time `metrics.get_latency_percentiles()` |
| Statistical test | < 100ms | Time `analyzer.two_sample_t_test()` |
| Health check | < 10ms | Time `checker.check_health()` |
| Rollback trigger | < 1s | Time from health failure to traffic reset |
| Sample size calculation | < 1ms | Time `analyzer.compute_required_sample_size()` |
---
## State Machine: Canary Rollout
```
                    ┌─────────────┐
                    │ INITIALIZING│
                    └──────┬──────┘
                           │ start_rollout()
                           ▼
              ┌────────────────────────┐
              │      STAGE_N           │
              │  (canary at N%)        │
              └───────────┬────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
    health check     stage complete   stage timeout
    failed           + samples OK     + samples OK
          │               │               │
          ▼               ▼               ▼
   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
   │ ROLLED_BACK │  │ STAGE_N+1   │  │ STAGE_N+1   │
   │ (0% canary) │  │ (next %)    │  │ (extend)    │
   └─────────────┘  └──────┬──────┘  └─────────────┘
                           │
                     all stages done
                           │
                           ▼
                    ┌─────────────┐
                    │  COMPLETE   │
                    │ (100% canary)│
                    └─────────────┘
ILLEGAL TRANSITIONS:
- COMPLETE → STAGE_N (can't go back)
- ROLLED_BACK → STAGE_N (must restart)
- STAGE_N → COMPLETE (must go through all stages)
```
---
## Concurrency Specification
### Threading Model
```
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Event Loop                          │
│                     (Single-threaded async)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  /predict    │  │  /canary/    │  │  /traffic    │          │
│  │  handlers    │  │  endpoints   │  │  endpoints   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                  │                  │
│         │ route()         │ get_state()      │ update_weights() │
│         ▼                 ▼                  ▼                  │
│  ┌──────────────────────────────────────────────────┐          │
│  │              TrafficSplitter                      │          │
│  │  - Stateless routing (hash computation)          │          │
│  │  - update_weights() replaces allocations         │          │
│  └──────────────────────────────────────────────────┘          │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────┐          │
│  │              PerVersionMetrics                    │          │
│  │  - defaultdict for thread-safe access            │          │
│  │  - Deques are modified in single event loop      │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                 │
│  ┌──────────────────────────────────────────────────┐          │
│  │         CanaryRolloutController (task)           │          │
│  │  - Runs as background asyncio task               │          │
│  │  - Modifies TrafficSplitter weights              │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
### Thread Safety Guarantees
| Component | Thread Safety | Reason |
|-----------|--------------|--------|
| `TrafficSplitter.route()` | Safe | Stateless hash computation |
| `TrafficSplitter.update_weights()` | Safe for async | Atomic list replacement |
| `PerVersionMetrics` | Safe for async | defaultdict + single event loop |
| `CanaryRolloutController` | Safe | Single background task |
| `ExperimentManager` | Safe for async | Single event loop access |
---
[[CRITERIA_JSON: {"module_id": "ml-model-serving-m4", "criteria": ["TrafficSplitter routes requests using SHA-256 hash of user_id modulo 100, guaranteeing deterministic routing (same user always routes to same version for fixed weights)", "TrafficAllocation validates weight_percent is 0-100 and version_id is non-empty", "TrafficSplitter.update_weights() validates allocations sum to 100% before applying, raising InvalidTrafficWeights on violation", "PerVersionMetrics records latency samples per version with success flag, tracking request counts and error counts separately", "PerVersionMetrics.get_latency_percentiles() returns p50/p90/p95/p99 from sorted successful latency samples", "PerVersionMetrics.compare_versions() computes latency diff percent and error rate diff between two versions", "StatisticalAnalyzer.two_sample_t_test() computes t-statistic, p-value, Cohen's d effect size, and 95% confidence interval", "StatisticalAnalyzer.compute_required_sample_size() returns minimum samples per group for given baseline stats and detectable effect", "StatisticalAnalyzer.proportion_test() compares error rates between versions using two-proportion z-test", "CanaryRolloutController progresses through configurable stages (default 5%/10%/25%/50%/100%) with stage duration and health check intervals", "CanaryRolloutController._update_splitter() updates traffic weights atomically via splitter.update_weights()", "CanaryHealthChecker.check_health() compares canary vs stable error rate (default 2x multiplier) and latency p50/p99 (default 1.5x/2x multipliers)", "CanaryRolloutController._trigger_rollback() resets traffic to 0% canary and sets state to ROLLED_BACK", "ExperimentConfig stores experiment_id, control_version, treatment_version, traffic_split percentage, minimum_sample_size, and significance_level", "PredictionLog captures timestamp, experiment_id, request_id, user_id, model_version, allocation_hash, input_hash, prediction, latency_ms, success flag", "ExperimentManager.start_experiment() creates per-experiment TrafficSplitter with control/treatment allocations", "ExperimentManager.route_request() uses experiment-specific splitter for deterministic user routing", "ExperimentManager.analyze_experiment() compares version metrics and returns recommendation based on latency and error rate differences"]}]
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: ml-model-serving-m5 -->
# Monitoring & Drift Detection
## Module Charter
This module implements comprehensive production monitoring for ML model serving, including latency histograms with percentile calculation, per-version metrics tracking, prediction distribution monitoring, and data drift detection via Kolmogorov-Smirnov statistical tests. It captures training baselines at deployment time, samples live traffic at configurable rates using reservoir sampling, compares feature distributions against baselines, and fires alerts when drift exceeds configurable thresholds. The module provides structured request logging for debugging drift incidents and exports all metrics in Prometheus format. It explicitly does NOT handle model versioning (M3), A/B traffic routing (M4), or model loading (M1). Upstream dependencies are the inference request handlers providing features and predictions; downstream consumers are Prometheus/Grafana dashboards, alerting systems (PagerDuty/Slack), and debugging workflows. Key invariants: (1) latency histograms use cumulative buckets enabling accurate percentile interpolation, (2) drift sampling rate guarantees statistical representativeness, (3) alerts have cooldown periods to prevent fatigue, (4) baseline age is tracked for staleness detection, (5) every logged prediction includes full experiment context for debugging.
---
## File Structure
Create files in this exact sequence:
```
ml-model-serving/m5_monitoring/
├── 01_config.py              # Monitoring configuration constants
├── 02_exceptions.py          # Monitoring-specific exception hierarchy
├── 03_histogram_bucket.py    # HistogramBucket dataclass
├── 04_latency_histogram.py   # LatencyHistogram with percentile calculation
├── 05_per_version_tracker.py # PerVersionLatencyTracker with Prometheus export
├── 06_prediction_monitor.py  # PredictionDistributionMonitor with baseline
├── 07_feature_baseline.py    # FeatureBaseline with CDF computation
├── 08_baseline_manager.py    # BaselineManager for storage and retrieval
├── 09_ks_test_detector.py    # KSTestDetector with statistic and p-value
├── 10_drift_sampler.py       # DriftDetectionSampler with reservoir sampling
├── 11_alert_rule.py          # AlertRule and Alert dataclasses
├── 12_alert_manager.py       # AlertManager with rules and cooldown
├── 13_structured_logger.py   # StructuredRequestLogger with JSON lines
└── 14_integration.py         # Complete monitoring server integration
```
---
## Complete Data Model
### HistogramBucket Dataclass
```python
# 03_histogram_bucket.py
from dataclasses import dataclass
@dataclass
class HistogramBucket:
    """
    A single bucket in a Prometheus-style latency histogram.
    Buckets are CUMULATIVE: each bucket's count includes all observations
    less than its upper_bound. This enables efficient percentile calculation.
    Fields:
        upper_bound: Exclusive upper bound in milliseconds
        count: Cumulative count of observations in this bucket and all lower buckets
    Invariants:
        - upper_bound > 0 for all buckets except +Inf
        - count is monotonically increasing when buckets are ordered
    """
    upper_bound: float  # Milliseconds (exclusive upper bound)
    count: int = 0
    def __lt__(self, other: "HistogramBucket") -> bool:
        """Enable sorting by upper_bound."""
        return self.upper_bound < other.upper_bound
```
---
### LatencyHistogram Dataclass
```python
# 04_latency_histogram.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import math
from .03_histogram_bucket import HistogramBucket
@dataclass
class LatencyHistogram:
    """
    Prometheus-style latency histogram with cumulative buckets.
    Design: Instead of storing every observation (memory explosion),
    we bucket observations into predefined ranges. This enables
    percentile calculation with bounded memory.
    The key insight: bucket boundaries should match your SLAs.
    If your p99 SLA is 200ms, have buckets at 100, 150, 200, 250, etc.
    Algorithm for percentile calculation:
    1. Find bucket containing the p-th percentile
    2. Interpolate linearly within that bucket
    Invariants:
        - Buckets are sorted by upper_bound
        - Counts are cumulative (bucket[i].count <= bucket[i+1].count)
        - +Inf bucket always has count == total_count
    """
    buckets: List[HistogramBucket] = field(default_factory=list)
    sum_total: float = 0.0  # Sum of all observations (for mean)
    count_total: int = 0
    def __post_init__(self):
        """Ensure buckets are sorted."""
        self.buckets = sorted(self.buckets)
    @classmethod
    def create_default(cls) -> "LatencyHistogram":
        """
        Create histogram with sensible default buckets.
        Bucket boundaries chosen for typical inference workloads:
        - Sub-10ms: excellent (cache hit territory)
        - 10-50ms: good (typical inference)
        - 50-100ms: acceptable
        - 100-500ms: slow but acceptable
        - 500ms+: problem territory
        """
        bounds = [1, 5, 10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 2000, 5000]
        buckets = [HistogramBucket(upper_bound=b) for b in bounds]
        return cls(buckets=buckets)
    @classmethod
    def create_for_sla(cls, p99_sla_ms: float) -> "LatencyHistogram":
        """
        Create histogram with buckets aligned to your SLA.
        Strategy: Put buckets at 0.25x, 0.5x, 0.75x, 1x, 1.25x, 1.5x, 2x your SLA.
        """
        multipliers = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
        bounds = [p99_sla_ms * m for m in multipliers]
        fine_bounds = [1, 5, 10, 25, 50]
        all_bounds = sorted(set(fine_bounds + bounds))
        buckets = [HistogramBucket(upper_bound=b) for b in all_bounds]
        return cls(buckets=buckets)
    def observe(self, latency_ms: float) -> None:
        """
        Record a latency observation.
        Increments all buckets where latency < bucket.upper_bound.
        This is the Prometheus cumulative histogram convention.
        Args:
            latency_ms: Latency in milliseconds
        Performance: O(n_buckets), typically O(15) = constant time
        """
        self.sum_total += latency_ms
        self.count_total += 1
        # Increment all buckets where this observation falls
        # (Prometheus uses cumulative buckets)
        for bucket in self.buckets:
            if latency_ms < bucket.upper_bound:
                bucket.count += 1
    def percentile(self, p: float) -> float:
        """
        Calculate the p-th percentile from histogram buckets.
        Algorithm:
        1. Find the bucket containing the p-th percentile
        2. Interpolate linearly within that bucket
        Args:
            p: Percentile (0-100)
        Returns:
            Approximate value at percentile
        Note: This is an approximation. Histogram percentiles are
        always approximate, but the error is bounded by bucket width.
        Performance: O(n_buckets) to find bucket, O(1) interpolation
        """
        if self.count_total == 0:
            return 0.0
        # Target count for this percentile
        target_count = (p / 100.0) * self.count_total
        # Find bucket containing this percentile
        prev_count = 0
        prev_bound = 0.0
        for bucket in self.buckets:
            if bucket.count >= target_count:
                # Percentile is in this bucket
                if bucket.count == prev_count:
                    # All observations at same value
                    return bucket.upper_bound
                # Linear interpolation within bucket
                bucket_width = bucket.upper_bound - prev_bound
                observations_in_bucket = bucket.count - prev_count
                observations_needed = target_count - prev_count
                fraction = observations_needed / observations_in_bucket
                return prev_bound + (bucket_width * fraction)
            prev_count = bucket.count
            prev_bound = bucket.upper_bound
        # Percentile is above all buckets
        return self.buckets[-1].upper_bound * 2  # Rough estimate
    def mean(self) -> float:
        """Calculate mean latency."""
        if self.count_total == 0:
            return 0.0
        return self.sum_total / self.count_total
    def to_prometheus_format(
        self, 
        metric_name: str, 
        labels: Dict[str, str]
    ) -> str:
        """
        Export histogram in Prometheus text exposition format.
        Prometheus expects:
        - {metric_name}_bucket{le="upper_bound"} count
        - {metric_name}_sum total_sum
        - {metric_name}_count total_count
        """
        label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
        if label_str:
            label_str = "{" + label_str + "}"
        lines = []
        # Bucket counts (cumulative)
        for bucket in self.buckets:
            le_label = f'{{le="{bucket.upper_bound}"'
            if labels:
                le_label += "," + ",".join(f'{k}="{v}"' for k, v in labels.items())
            le_label += "}"
            lines.append(f"{metric_name}_bucket{le_label} {bucket.count}")
        # +Inf bucket (equals total count)
        le_inf = '{le="+Inf"'
        if labels:
            le_inf += "," + ",".join(f'{k}="{v}"' for k, v in labels.items())
        le_inf += "}"
        lines.append(f"{metric_name}_bucket{le_inf} {self.count_total}")
        # Sum and count
        lines.append(f"{metric_name}_sum{label_str} {self.sum_total:.2f}")
        lines.append(f"{metric_name}_count{label_str} {self.count_total}")
        return "\n".join(lines)
    def merge(self, other: "LatencyHistogram") -> "LatencyHistogram":
        """
        Merge two histograms with compatible bucket boundaries.
        Used for aggregating metrics across instances.
        """
        if not other.buckets:
            return LatencyHistogram(
                buckets=list(self.buckets),
                sum_total=self.sum_total,
                count_total=self.count_total
            )
        # Assume same bucket boundaries
        merged_buckets = []
        for i, bucket in enumerate(self.buckets):
            merged_count = bucket.count + other.buckets[i].count
            merged_buckets.append(HistogramBucket(
                upper_bound=bucket.upper_bound,
                count=merged_count
            ))
        return LatencyHistogram(
            buckets=merged_buckets,
            sum_total=self.sum_total + other.sum_total,
            count_total=self.count_total + other.count_total
        )
```
---
### PerVersionLatencyTracker Dataclass
```python
# 05_per_version_tracker.py
from dataclasses import dataclass, field
from typing import Dict, List, DefaultDict
from collections import defaultdict
import time
from .04_latency_histogram import LatencyHistogram
@dataclass
class PerVersionLatencyTracker:
    """
    Tracks latency histograms per model version.
    This is critical for A/B testing: you need to compare
    latency distributions between versions, not just averages.
    Invariants:
        - Each version has independent histogram
        - Error counts are separate from latency samples
        - Request counts are monotonically increasing
    """
    _histograms: Dict[str, LatencyHistogram] = field(
        default_factory=dict
    )
    _request_counts: DefaultDict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    _error_counts: DefaultDict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    _start_time: float = field(default_factory=time.time)
    def record(
        self,
        version_id: str,
        latency_ms: float,
        success: bool = True
    ) -> None:
        """
        Record a latency observation for a model version.
        Args:
            version_id: Model version identifier
            latency_ms: Observed latency in milliseconds
            success: Whether the request succeeded
        """
        if version_id not in self._histograms:
            self._histograms[version_id] = LatencyHistogram.create_default()
        self._histograms[version_id].observe(latency_ms)
        self._request_counts[version_id] += 1
        if not success:
            self._error_counts[version_id] += 1
    def get_percentiles(
        self,
        version_id: str,
        percentiles: List[float] = None
    ) -> Dict[str, float]:
        """
        Get latency percentiles for a version.
        Args:
            version_id: Model version identifier
            percentiles: List of percentiles to compute (default: [50, 90, 95, 99])
        Returns:
            Dict mapping "p50", "p90", etc. to values in milliseconds
        """
        if percentiles is None:
            percentiles = [50, 90, 95, 99]
        if version_id not in self._histograms:
            return {f"p{p}": 0.0 for p in percentiles}
        histogram = self._histograms[version_id]
        return {f"p{p}": histogram.percentile(p) for p in percentiles}
    def get_error_rate(self, version_id: str) -> float:
        """Get error rate for a version (0.0 to 1.0)."""
        total = self._request_counts.get(version_id, 0)
        if total == 0:
            return 0.0
        return self._error_counts.get(version_id, 0) / total
    def get_throughput_rps(self, version_id: str) -> float:
        """Calculate requests per second for a version."""
        total = self._request_counts.get(version_id, 0)
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0
        return total / elapsed
    def get_all_version_metrics(self) -> Dict[str, dict]:
        """Get metrics summary for all tracked versions."""
        all_versions = set(self._request_counts.keys())
        return {
            version_id: {
                "latency_percentiles": self.get_percentiles(version_id),
                "error_rate": self.get_error_rate(version_id),
                "throughput_rps": self.get_throughput_rps(version_id),
                "request_count": self._request_counts.get(version_id, 0),
                "error_count": self._error_counts.get(version_id, 0),
            }
            for version_id in all_versions
        }
    def to_prometheus_format(self) -> str:
        """Export all metrics in Prometheus format."""
        lines = []
        for version_id in self._histograms:
            # Latency histogram
            lines.append(self._histograms[version_id].to_prometheus_format(
                "model_inference_latency_ms",
                {"version": version_id}
            ))
            # Error rate
            error_rate = self.get_error_rate(version_id)
            lines.append(
                f'model_error_rate{{version="{version_id}"}} {error_rate:.4f}'
            )
            # Request count
            count = self._request_counts.get(version_id, 0)
            lines.append(
                f'model_requests_total{{version="{version_id}"}} {count}'
            )
            # Throughput
            rps = self.get_throughput_rps(version_id)
            lines.append(
                f'model_throughput_rps{{version="{version_id}"}} {rps:.2f}'
            )
        return "\n".join(lines)
```
---
### FeatureBaseline Dataclass
```python
# 07_feature_baseline.py
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math
import random
@dataclass
class FeatureBaseline:
    """
    Baseline statistics for a single feature.
    Captured at training time or deployment time, this represents
    the "expected" distribution of the feature.
    Used for drift detection via KS test comparison.
    Invariants:
        - sorted_samples is sorted in ascending order
        - samples length <= max_baseline_samples
        - min_val <= max_val
    """
    feature_name: str
    samples: List[float]  # Training samples (or representative subset)
    min_val: float
    max_val: float
    mean: float
    std: float
    # Precomputed CDF for efficient comparison
    sorted_samples: List[float] = field(default_factory=list)
    @classmethod
    def from_samples(
        cls,
        feature_name: str,
        samples: List[float],
        max_baseline_samples: int = 10000
    ) -> "FeatureBaseline":
        """
        Create baseline from training samples.
        Args:
            feature_name: Name of the feature
            samples: List of sample values from training data
            max_baseline_samples: Maximum samples to keep (downsamples if exceeded)
        """
        if not samples:
            return cls(
                feature_name=feature_name,
                samples=[],
                min_val=0.0,
                max_val=0.0,
                mean=0.0,
                std=0.0,
                sorted_samples=[]
            )
        # Downsample if too large
        if len(samples) > max_baseline_samples:
            samples = random.sample(samples, max_baseline_samples)
        sorted_samples = sorted(samples)
        n = len(samples)
        mean = sum(samples) / n
        # Compute standard deviation
        if n < 2:
            std = 0.0
        else:
            variance = sum((x - mean) ** 2 for x in samples) / (n - 1)
            std = math.sqrt(variance)
        return cls(
            feature_name=feature_name,
            samples=samples,
            min_val=min(samples),
            max_val=max(samples),
            mean=mean,
            std=std,
            sorted_samples=sorted_samples
        )
    def to_dict(self) -> dict:
        """Serialize to dict for storage."""
        return {
            "feature_name": self.feature_name,
            "samples": self.samples[:1000],  # Limit for storage
            "min_val": self.min_val,
            "max_val": self.max_val,
            "mean": self.mean,
            "std": self.std,
        }
    @classmethod
    def from_dict(cls, data: dict) -> "FeatureBaseline":
        """Deserialize from dict."""
        samples = data.get("samples", [])
        return cls(
            feature_name=data["feature_name"],
            samples=samples,
            min_val=data["min_val"],
            max_val=data["max_val"],
            mean=data["mean"],
            std=data["std"],
            sorted_samples=sorted(samples) if samples else []
        )
```
---
### KSTestResult Dataclass
```python
# 09_ks_test_detector.py
from dataclasses import dataclass
from typing import Optional
@dataclass
class KSTestResult:
    """
    Result of a KS test between two distributions.
    The KS test compares two empirical distributions by computing
    the maximum distance between their cumulative distribution functions.
    """
    feature_name: str
    statistic: float  # D statistic (0 to 1)
    p_value: float  # Probability under null hypothesis
    is_significant: bool  # True if p_value < threshold
    baseline_samples: int
    live_samples: int
    interpretation: str
    drift_direction: Optional[str] = None  # "higher", "lower", or None
```
---
### KSTestDetector Class
```python
# 09_ks_test_detector.py
import math
from typing import List, Tuple
from .07_feature_baseline import FeatureBaseline
from .02_exceptions import InsufficientSamplesForDrift
class KSTestDetector:
    """
    Performs Kolmogorov-Smirnov test for drift detection.
    The KS test compares two empirical distributions by computing
    the maximum distance between their cumulative distribution functions.
    Advantages:
    - Non-parametric: works for any distribution shape
    - Sensitive to location, scale, and shape differences
    - Well-defined p-value for hypothesis testing
    Limitations:
    - More sensitive to differences near the median than tails
    - Requires sufficient samples in both distributions
    Algorithm:
    1. Sort both sample sets
    2. Walk through merged sorted values, tracking CDF positions
    3. Track maximum CDF difference
    4. Compute p-value from D statistic and sample sizes
    """
    @staticmethod
    def compute_cdf(sorted_samples: List[float]) -> List[Tuple[float, float]]:
        """
        Compute empirical CDF from sorted samples.
        Returns list of (value, cumulative_probability) tuples.
        """
        n = len(sorted_samples)
        if n == 0:
            return []
        cdf = []
        for i, val in enumerate(sorted_samples):
            prob = (i + 1) / n
            cdf.append((val, prob))
        return cdf
    @staticmethod
    def ks_statistic(
        sorted_samples1: List[float],
        sorted_samples2: List[float]
    ) -> float:
        """
        Compute the KS statistic (D) between two samples.
        D = max|CDF1(x) - CDF2(x)| for all x
        This is the maximum vertical distance between the two CDFs.
        Algorithm:
        1. Merge sorted lists, tracking which sample each value came from
        2. At each value, compute CDF positions
        3. Track maximum difference
        Performance: O(n1 + n2)
        """
        n1 = len(sorted_samples1)
        n2 = len(sorted_samples2)
        if n1 == 0 or n2 == 0:
            return 0.0
        i, j = 0, 0
        max_diff = 0.0
        cdf1_val, cdf2_val = 0.0, 0.0
        while i < n1 or j < n2:
            # Get next value to process
            if i < n1 and (j >= n2 or sorted_samples1[i] <= sorted_samples2[j]):
                cdf1_val = (i + 1) / n1
                i += 1
            else:
                cdf2_val = (j + 1) / n2
                j += 1
            # Update max difference
            diff = abs(cdf1_val - cdf2_val)
            max_diff = max(max_diff, diff)
        return max_diff
    @staticmethod
    def ks_p_value(d_statistic: float, n1: int, n2: int) -> float:
        """
        Compute p-value for KS test.
        Under the null hypothesis (same distribution), the KS statistic
        follows a known distribution. This function approximates the p-value.
        The asymptotic approximation:
        p ≈ 2 * Σ (-1)^(k-1) * exp(-2 * k² * λ)
        where λ = (sqrt(n1*n2/(n1+n2)) * D)²
        """
        if n1 == 0 or n2 == 0:
            return 1.0
        # Effective sample size
        n_eff = math.sqrt(n1 * n2 / (n1 + n2))
        # Lambda parameter
        lambda_val = (n_eff * d_statistic) ** 2
        # Approximate p-value using first few terms of series
        p_value = 0.0
        for k in range(1, 101):
            term = 2 * ((-1) ** (k - 1)) * math.exp(-2 * (k ** 2) * lambda_val)
            p_value += term
            if abs(term) < 1e-10:
                break
        return max(0.0, min(1.0, p_value))
    def test(
        self,
        baseline: FeatureBaseline,
        live_samples: List[float],
        significance_level: float = 0.01
    ) -> KSTestResult:
        """
        Perform KS test between baseline and live samples.
        Args:
            baseline: Training baseline for this feature
            live_samples: Recent samples from live traffic
            significance_level: p-value threshold for significance
        Returns:
            KSTestResult with statistic, p-value, and interpretation
        """
        n_baseline = len(baseline.sorted_samples)
        n_live = len(live_samples)
        if n_baseline == 0:
            return KSTestResult(
                feature_name=baseline.feature_name,
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                baseline_samples=n_baseline,
                live_samples=n_live,
                interpretation="No baseline samples available"
            )
        if n_live == 0:
            return KSTestResult(
                feature_name=baseline.feature_name,
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                baseline_samples=n_baseline,
                live_samples=n_live,
                interpretation="No live samples available"
            )
        # Sort live samples
        sorted_live = sorted(live_samples)
        # Compute KS statistic
        d_statistic = self.ks_statistic(
            baseline.sorted_samples,
            sorted_live
        )
        # Compute p-value
        p_value = self.ks_p_value(d_statistic, n_baseline, n_live)
        # Determine significance
        is_significant = p_value < significance_level
        # Determine drift direction
        live_mean = sum(live_samples) / n_live
        mean_diff = live_mean - baseline.mean
        if abs(mean_diff) > baseline.std * 0.5:
            drift_direction = "higher" if mean_diff > 0 else "lower"
        else:
            drift_direction = None
        # Interpretation
        if is_significant:
            interpretation = (
                f"SIGNIFICANT DRIFT DETECTED (p={p_value:.4f}). "
                f"KS statistic: {d_statistic:.4f}. "
            )
            if drift_direction:
                interpretation += f"Live distribution is {drift_direction} than baseline."
            else:
                interpretation += "Distribution shape has changed."
        else:
            interpretation = (
                f"No significant drift (p={p_value:.4f}). "
                f"KS statistic: {d_statistic:.4f}"
            )
        return KSTestResult(
            feature_name=baseline.feature_name,
            statistic=d_statistic,
            p_value=p_value,
            is_significant=is_significant,
            baseline_samples=n_baseline,
            live_samples=n_live,
            interpretation=interpretation,
            drift_direction=drift_direction
        )
```
---
### DriftDetectionSampler Dataclass
```python
# 10_drift_sampler.py
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from .07_feature_baseline import FeatureBaseline
from .09_ks_test_detector import KSTestDetector, KSTestResult
from .02_exceptions import InsufficientSamplesForDrift
@dataclass
class SamplingConfig:
    """Configuration for drift detection sampling."""
    sample_rate: int = 1000  # 1 in N requests is sampled
    min_samples_for_test: int = 100
    max_live_samples: int = 5000
    test_interval_seconds: float = 300.0
@dataclass
class FeatureSampleBuffer:
    """
    Buffer for sampled feature values using reservoir sampling.
    Reservoir sampling ensures a representative sample even with
    infinite stream of data.
    """
    feature_name: str
    max_size: int
    samples: List[float] = field(default_factory=list)
    total_seen: int = 0
    def add(self, value: float) -> None:
        """
        Add a sample using reservoir sampling (Algorithm R).
        This ensures that after seeing N values, each value has
        an equal probability of being in the reservoir.
        """
        self.total_seen += 1
        if len(self.samples) < self.max_size:
            self.samples.append(value)
        else:
            # Reservoir full, probabilistically replace
            j = random.randint(1, self.total_seen)
            if j <= self.max_size:
                self.samples[j - 1] = value
    def get_samples(self) -> List[float]:
        """Get current samples."""
        return self.samples.copy()
    def clear(self) -> None:
        """Clear the buffer."""
        self.samples = []
        self.total_seen = 0
@dataclass
class DriftDetectionSampler:
    """
    Samples incoming requests for drift detection.
    Key design decisions:
    1. Sample rate is configurable (default 1/1000)
    2. Each feature is sampled independently
    3. Reservoir sampling ensures representative samples
    4. Tests run on interval, not on every sample
    Invariants:
        - Sampling uses deterministic counter (every Nth request)
        - Reservoir is always a representative sample
        - Tests only run when min_samples achieved
    """
    baselines: Dict[str, FeatureBaseline]
    config: SamplingConfig
    _buffers: Dict[str, FeatureSampleBuffer] = field(default_factory=dict)
    _request_counter: int = 0
    _last_test_time: float = 0
    _last_results: Dict[str, KSTestResult] = field(default_factory=dict)
    _detector: KSTestDetector = field(default_factory=KSTestDetector)
    def __post_init__(self):
        """Initialize buffers for each feature."""
        self._buffers = {
            name: FeatureSampleBuffer(
                feature_name=name,
                max_size=self.config.max_live_samples
            )
            for name in self.baselines
        }
    def should_sample(self) -> bool:
        """
        Determine if this request should be sampled.
        Uses deterministic sampling: every Nth request is sampled.
        """
        self._request_counter += 1
        return self._request_counter % self.config.sample_rate == 0
    def record_features(self, features: Dict[str, float]) -> None:
        """
        Record feature values for a sampled request.
        Args:
            features: Dict mapping feature name to value
        """
        for feature_name, value in features.items():
            if feature_name in self._buffers:
                self._buffers[feature_name].add(value)
    def should_run_test(self) -> bool:
        """Check if enough time has elapsed to run drift test."""
        elapsed = time.time() - self._last_test_time
        return elapsed >= self.config.test_interval_seconds
    def run_drift_tests(
        self,
        significance_level: float = 0.01
    ) -> Dict[str, KSTestResult]:
        """
        Run KS tests for all features.
        Returns dict mapping feature name to test result.
        """
        results = {}
        for feature_name, buffer in self._buffers.items():
            samples = buffer.get_samples()
            # Check minimum samples
            if len(samples) < self.config.min_samples_for_test:
                results[feature_name] = KSTestResult(
                    feature_name=feature_name,
                    statistic=0.0,
                    p_value=1.0,
                    is_significant=False,
                    baseline_samples=len(self.baselines[feature_name].sorted_samples),
                    live_samples=len(samples),
                    interpretation=f"Insufficient live samples ({len(samples)} < {self.config.min_samples_for_test})"
                )
                continue
            # Run KS test
            baseline = self.baselines[feature_name]
            result = self._detector.test(
                baseline=baseline,
                live_samples=samples,
                significance_level=significance_level
            )
            results[feature_name] = result
        self._last_test_time = time.time()
        self._last_results = results
        return results
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection status."""
        total_features = len(self.baselines)
        drifted_features = sum(
            1 for r in self._last_results.values()
            if r.is_significant
        )
        return {
            "total_features_monitored": total_features,
            "features_with_drift": drifted_features,
            "drift_percentage": (
                (drifted_features / total_features * 100) 
                if total_features > 0 else 0
            ),
            "last_test_time": self._last_test_time,
            "sample_rate": f"1/{self.config.sample_rate}",
            "requests_seen": self._request_counter,
            "drifted_feature_names": [
                name for name, result in self._last_results.items()
                if result.is_significant
            ]
        }
    def to_prometheus_format(self) -> str:
        """Export drift metrics in Prometheus format."""
        lines = []
        for feature_name, result in self._last_results.items():
            labels = f'{{feature="{feature_name}"}}'
            lines.append(
                f"drift_ks_statistic{labels} {result.statistic:.4f}"
            )
            lines.append(
                f"drift_p_value{labels} {result.p_value:.4f}"
            )
            lines.append(
                f"drift_detected{labels} {1 if result.is_significant else 0}"
            )
            lines.append(
                f"drift_baseline_samples{labels} {result.baseline_samples}"
            )
            lines.append(
                f"drift_live_samples{labels} {result.live_samples}"
            )
        summary = self.get_drift_summary()
        lines.append(f"drift_features_monitored {summary['total_features_monitored']}")
        lines.append(f"drift_features_with_drift {summary['features_with_drift']}")
        return "\n".join(lines)
```
---
### AlertRule and Alert Dataclasses
```python
# 11_alert_rule.py
from dataclasses import dataclass
from typing import Optional
from enum import Enum
import time
class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
@dataclass
class AlertRule:
    """
    Configuration for a drift alert rule.
    Rules define conditions that trigger alerts.
    """
    name: str
    feature_name: Optional[str]  # None = aggregate rule
    condition: str  # "p_value < 0.01" or "ks_statistic > 0.1"
    severity: AlertSeverity
    cooldown_seconds: float = 3600.0
    min_samples: int = 100
    def matches_condition(
        self, 
        p_value: float, 
        ks_statistic: float
    ) -> Tuple[bool, float, float]:
        """
        Check if this rule's condition is met.
        Returns:
            (is_triggered, value, threshold)
        """
        if "p_value <" in self.condition:
            threshold = float(self.condition.split("<")[1].strip())
            return p_value < threshold, p_value, threshold
        elif "ks_statistic >" in self.condition:
            threshold = float(self.condition.split(">")[1].strip())
            return ks_statistic > threshold, ks_statistic, threshold
        return False, 0.0, 0.0
@dataclass
class Alert:
    """A fired alert."""
    rule_name: str
    feature_name: Optional[str]
    severity: AlertSeverity
    message: str
    timestamp: float
    value: float
    threshold: float
    def to_dict(self) -> dict:
        return {
            "rule_name": self.rule_name,
            "feature_name": self.feature_name,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "value": self.value,
            "threshold": self.threshold,
        }
```
---
### AlertManager Dataclass
```python
# 12_alert_manager.py
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Awaitable
import logging
from .09_ks_test_detector import KSTestResult
from .11_alert_rule import AlertRule, Alert, AlertSeverity
from .02_exceptions import AlertThresholdExceeded
logger = logging.getLogger(__name__)
@dataclass
class AlertManager:
    """
    Manages drift alerts with cooldown and aggregation.
    Key design:
    1. Rules define conditions for alerts
    2. Cooldown prevents alert spam
    3. Severity levels enable routing
    4. Aggregation groups related alerts
    Invariants:
        - Alerts don't fire more than once per cooldown period
        - Each rule/feature combination has independent cooldown
    """
    rules: List[AlertRule]
    on_alert: Optional[Callable[[Alert], Awaitable[None]]] = None
    _last_alert_time: Dict[str, float] = field(default_factory=dict)
    _recent_alerts: List[Alert] = field(default_factory=list)
    _max_recent_alerts: int = 100
    def evaluate(
        self,
        drift_results: Dict[str, KSTestResult],
        feature_samples: Dict[str, int]
    ) -> List[Alert]:
        """
        Evaluate alert rules against drift results.
        Returns list of alerts that fired (and passed cooldown).
        """
        alerts = []
        for rule in self.rules:
            # Check per-feature rules
            if rule.feature_name:
                if rule.feature_name not in drift_results:
                    continue
                result = drift_results[rule.feature_name]
                samples = feature_samples.get(rule.feature_name, 0)
                if samples < rule.min_samples:
                    continue
                alert = self._evaluate_rule(rule, result)
                if alert:
                    alerts.append(alert)
            # Check aggregate rules
            else:
                alert = self._evaluate_aggregate_rule(rule, drift_results)
                if alert:
                    alerts.append(alert)
        # Store recent alerts
        self._recent_alerts.extend(alerts)
        if len(self._recent_alerts) > self._max_recent_alerts:
            self._recent_alerts = self._recent_alerts[-self._max_recent_alerts:]
        return alerts
    def _evaluate_rule(
        self,
        rule: AlertRule,
        result: KSTestResult
    ) -> Optional[Alert]:
        """Evaluate a single rule against a single feature result."""
        should_alert, value, threshold = rule.matches_condition(
            result.p_value,
            result.statistic
        )
        if not should_alert:
            return None
        # Check cooldown
        alert_key = f"{rule.name}:{rule.feature_name}"
        last_alert = self._last_alert_time.get(alert_key, 0)
        if time.time() - last_alert < rule.cooldown_seconds:
            return None
        # Fire alert
        self._last_alert_time[alert_key] = time.time()
        return Alert(
            rule_name=rule.name,
            feature_name=rule.feature_name,
            severity=rule.severity,
            message=f"Drift detected for {rule.feature_name}: {result.interpretation}",
            timestamp=time.time(),
            value=value,
            threshold=threshold
        )
    def _evaluate_aggregate_rule(
        self,
        rule: AlertRule,
        drift_results: Dict[str, KSTestResult]
    ) -> Optional[Alert]:
        """Evaluate an aggregate rule across all features."""
        drifted_count = sum(1 for r in drift_results.values() if r.is_significant)
        total_count = len(drift_results)
        if total_count == 0:
            return None
        drift_fraction = drifted_count / total_count
        # Parse threshold from condition
        if "drift_fraction >" in rule.condition:
            threshold = float(rule.condition.split(">")[1].strip())
            should_alert = drift_fraction > threshold
        else:
            return None
        if not should_alert:
            return None
        # Check cooldown
        alert_key = f"{rule.name}:aggregate"
        last_alert = self._last_alert_time.get(alert_key, 0)
        if time.time() - last_alert < rule.cooldown_seconds:
            return None
        # Fire alert
        self._last_alert_time[alert_key] = time.time()
        return Alert(
            rule_name=rule.name,
            feature_name=None,
            severity=rule.severity,
            message=f"Aggregate drift: {drifted_count}/{total_count} features drifted ({drift_fraction:.1%})",
            timestamp=time.time(),
            value=drift_fraction,
            threshold=threshold
        )
    def get_recent_alerts(self, limit: int = 20) -> List[Alert]:
        """Get recent alerts."""
        return self._recent_alerts[-limit:]
# Default alert rules
DEFAULT_DRIFT_RULES = [
    AlertRule(
        name="p_value_critical",
        feature_name=None,
        condition="p_value < 0.001",
        severity=AlertSeverity.CRITICAL,
        cooldown_seconds=1800.0,
        min_samples=200
    ),
    AlertRule(
        name="p_value_warning",
        feature_name=None,
        condition="p_value < 0.01",
        severity=AlertSeverity.WARNING,
        cooldown_seconds=3600.0,
        min_samples=100
    ),
    AlertRule(
        name="aggregate_drift",
        feature_name=None,
        condition="drift_fraction > 0.2",
        severity=AlertSeverity.WARNING,
        cooldown_seconds=3600.0
    ),
]
```
---
### BaselineManager Class
```python
# 08_baseline_manager.py
import os
import json
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
import logging
from .07_feature_baseline import FeatureBaseline
from .02_exceptions import BaselineNotFound, BaselineStale
logger = logging.getLogger(__name__)
@dataclass
class ModelBaseline:
    """
    Complete baseline for a model version.
    Captured at deployment time, this contains:
    - Feature distributions from training data
    - Prediction distribution from validation
    - Model metadata
    """
    model_version: str
    capture_time: str
    feature_baselines: Dict[str, FeatureBaseline]
    prediction_baseline: Optional[Dict[str, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "model_version": self.model_version,
            "capture_time": self.capture_time,
            "feature_baselines": {
                name: b.to_dict()
                for name, b in self.feature_baselines.items()
            },
            "prediction_baseline": self.prediction_baseline,
            "metadata": self.metadata,
        }
    @classmethod
    def from_dict(cls, data: dict) -> "ModelBaseline":
        """Deserialize from dictionary."""
        feature_baselines = {}
        for name, bdata in data.get("feature_baselines", {}).items():
            feature_baselines[name] = FeatureBaseline.from_dict(bdata)
        return cls(
            model_version=data["model_version"],
            capture_time=data["capture_time"],
            feature_baselines=feature_baselines,
            prediction_baseline=data.get("prediction_baseline"),
            metadata=data.get("metadata", {}),
        )
    def save(self, path: str) -> None:
        """Save baseline to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    @classmethod
    def load(cls, path: str) -> "ModelBaseline":
        """Load baseline from file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))
class BaselineManager:
    """
    Manages training baselines for model versions.
    Responsibilities:
    1. Capture baselines from training data
    2. Store baselines with model version
    3. Load baselines at deployment time
    4. Track baseline age and staleness
    """
    def __init__(self, storage_dir: str = "baselines"):
        self.storage_dir = storage_dir
        self._loaded_baselines: Dict[str, ModelBaseline] = {}
    def capture_from_training_data(
        self,
        model_version: str,
        training_features: Dict[str, List[float]],
        validation_predictions: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ModelBaseline:
        """
        Capture baseline from training data.
        This should be called during model training or just after.
        """
        feature_baselines = {}
        for feature_name, samples in training_features.items():
            feature_baselines[feature_name] = FeatureBaseline.from_samples(
                feature_name=feature_name,
                samples=samples,
                max_baseline_samples=10000
            )
        baseline = ModelBaseline(
            model_version=model_version,
            capture_time=datetime.now().isoformat(),
            feature_baselines=feature_baselines,
            prediction_baseline=None,
            metadata=metadata or {}
        )
        # Save to disk
        path = os.path.join(self.storage_dir, f"{model_version}.json")
        baseline.save(path)
        # Cache in memory
        self._loaded_baselines[model_version] = baseline
        return baseline
    def load_baseline(self, model_version: str) -> Optional[ModelBaseline]:
        """Load baseline for a model version."""
        if model_version in self._loaded_baselines:
            return self._loaded_baselines[model_version]
        path = os.path.join(self.storage_dir, f"{model_version}.json")
        if not os.path.exists(path):
            return None
        baseline = ModelBaseline.load(path)
        self._loaded_baselines[model_version] = baseline
        return baseline
    def get_baseline_age_days(self, model_version: str) -> Optional[float]:
        """Get age of baseline in days."""
        baseline = self.load_baseline(model_version)
        if not baseline:
            return None
        capture_time = datetime.fromisoformat(baseline.capture_time)
        age = datetime.now() - capture_time
        return age.total_seconds() / 86400
    def is_baseline_stale(
        self,
        model_version: str,
        max_age_days: float = 90.0
    ) -> bool:
        """Check if baseline is too old."""
        age = self.get_baseline_age_days(model_version)
        if age is None:
            return True
        return age > max_age_days
```
---
### StructuredRequestLogger Class
```python
# 13_structured_logger.py
import json
import time
import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
@dataclass
class StructuredLogEntry:
    """
    A structured log entry for request auditing.
    Captures complete context for debugging.
    """
    timestamp: float
    request_id: str
    model_version: str
    user_id: Optional[str]
    features: Dict[str, float]
    features_hash: str
    prediction: Any
    prediction_confidence: Optional[float]
    latency_ms: float
    success: bool
    error_message: Optional[str] = None
    was_sampled_for_drift: bool = False
    experiment_id: Optional[str] = None
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "model_version": self.model_version,
            "user_id": self.user_id,
            "features": self.features,
            "features_hash": self.features_hash,
            "prediction": self.prediction,
            "prediction_confidence": self.prediction_confidence,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "error_message": self.error_message,
            "was_sampled_for_drift": self.was_sampled_for_drift,
            "experiment_id": self.experiment_id,
        }
    def to_json_line(self) -> str:
        """Convert to JSON line for logging."""
        return json.dumps(self.to_dict())
    @staticmethod
    def hash_features(features: Dict[str, Any]) -> str:
        """Create deterministic hash of features."""
        feature_str = json.dumps(features, sort_keys=True)
        return hashlib.sha256(feature_str.encode()).hexdigest()[:16]
class StructuredRequestLogger:
    """
    Logs requests with sampling for debugging and audit.
    Key design:
    1. Sample a percentage of requests for full logging
    2. Always log requests that were sampled for drift detection
    3. Structured format enables querying and analysis
    4. Separate from metrics (this is for debugging)
    """
    def __init__(
        self,
        log_dir: str = "logs/requests",
        sample_rate: float = 0.01,
        flush_interval: int = 100
    ):
        self.log_dir = log_dir
        self.sample_rate = sample_rate
        self.flush_interval = flush_interval
        self._buffer: List[StructuredLogEntry] = []
        self._request_counter = 0
    def should_log(self) -> bool:
        """Determine if this request should be logged."""
        self._request_counter += 1
        return self._request_counter % int(1 / self.sample_rate) == 0
    def log(
        self,
        request_id: str,
        model_version: str,
        features: Dict[str, float],
        prediction: Any,
        latency_ms: float,
        success: bool = True,
        error_message: Optional[str] = None,
        user_id: Optional[str] = None,
        prediction_confidence: Optional[float] = None,
        was_sampled_for_drift: bool = False,
        experiment_id: Optional[str] = None,
        force_log: bool = False
    ) -> None:
        """
        Log a request entry.
        Args:
            force_log: If True, log regardless of sample rate
        """
        if not force_log and not self.should_log():
            return
        entry = StructuredLogEntry(
            timestamp=time.time(),
            request_id=request_id,
            model_version=model_version,
            user_id=user_id,
            features=features,
            features_hash=StructuredLogEntry.hash_features(features),
            prediction=prediction,
            prediction_confidence=prediction_confidence,
            latency_ms=latency_ms,
            success=success,
            error_message=error_message,
            was_sampled_for_drift=was_sampled_for_drift,
            experiment_id=experiment_id
        )
        self._buffer.append(entry)
        if len(self._buffer) >= self.flush_interval:
            self._flush()
    def _flush(self) -> None:
        """Write buffered entries to disk."""
        if not self._buffer:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_path = os.path.join(self.log_dir, f"requests-{date_str}.jsonl")
        with open(log_path, "a") as f:
            for entry in self._buffer:
                f.write(entry.to_json_line() + "\n")
        self._buffer = []
    def finalize(self) -> None:
        """Flush remaining entries."""
        self._flush()
```
---
### Exception Hierarchy
```python
# 02_exceptions.py
class MonitoringError(Exception):
    """Base exception for monitoring errors."""
    pass
class BaselineNotFound(MonitoringError):
    """Raised when no training baseline exists for a model version."""
    def __init__(self, model_version: str):
        self.model_version = model_version
        super().__init__(f"No baseline found for model version: {model_version}")
class InsufficientSamplesForDrift(MonitoringError):
    """Raised when not enough samples for reliable drift test."""
    def __init__(self, feature_name: str, required: int, actual: int):
        self.feature_name = feature_name
        self.required = required
        self.actual = actual
        super().__init__(
            f"Insufficient samples for {feature_name}: "
            f"need {required}, have {actual}"
        )
class AlertThresholdExceeded(MonitoringError):
    """Raised when drift alert threshold is exceeded."""
    def __init__(self, feature_name: str, p_value: float, threshold: float):
        self.feature_name = feature_name
        self.p_value = p_value
        self.threshold = threshold
        super().__init__(
            f"Drift alert for {feature_name}: p-value {p_value:.4f} < {threshold}"
        )
class BaselineStale(MonitoringError):
    """Raised when baseline is older than maximum allowed age."""
    def __init__(self, model_version: str, age_days: float, max_age_days: float):
        self.model_version = model_version
        self.age_days = age_days
        self.max_age_days = max_age_days
        super().__init__(
            f"Baseline for {model_version} is {age_days:.0f} days old "
            f"(max: {max_age_days:.0f})"
        )
class LoggingFailed(MonitoringError):
    """Raised when structured logging fails."""
    def __init__(self, reason: str, original_error: Optional[Exception] = None):
        self.reason = reason
        self.original_error = original_error
        super().__init__(f"Logging failed: {reason}")
```
---
## Algorithm Specification
### Latency Histogram Percentile Algorithm
```
ALGORITHM: percentile (LatencyHistogram)
INPUT: p (float, 0-100)
OUTPUT: Approximate value at percentile
PRECONDITION: count_total > 0
1. // Compute target count
   target_count ← (p / 100.0) * count_total
2. // Walk buckets to find containing bucket
   prev_count ← 0
   prev_bound ← 0.0
   FOR each bucket IN buckets (sorted by upper_bound):
       IF bucket.count >= target_count:
           // Percentile is in this bucket
           IF bucket.count == prev_count:
               // No observations in this bucket range
               RETURN bucket.upper_bound
           // Linear interpolation
           bucket_width ← bucket.upper_bound - prev_bound
           observations_in_bucket ← bucket.count - prev_count
           observations_needed ← target_count - prev_count
           fraction ← observations_needed / observations_in_bucket
           RETURN prev_bound + (bucket_width * fraction)
       prev_count ← bucket.count
       prev_bound ← bucket.upper_bound
3. // Percentile above all buckets
   RETURN buckets[-1].upper_bound * 2  // Rough estimate
INVARIANT: Result is within [min_observed, max_observed * 2]
COMPLEXITY: O(n_buckets), typically O(15)
```
### Reservoir Sampling Algorithm
```
ALGORITHM: add (FeatureSampleBuffer)
INPUT: value (float)
OUTPUT: None (side effect: reservoir updated)
INVARIANT: After N insertions, each of the N values has equal probability of being in reservoir
1. total_seen ← total_seen + 1
2. IF len(samples) < max_size:
       // Reservoir not full, append
       APPEND value TO samples
   ELSE:
       // Reservoir full, probabilistically replace
       j ← RANDOM_INTEGER(1, total_seen)
       IF j <= max_size:
           samples[j - 1] ← value
PROBABILITY: P(value in reservoir) = max_size / total_seen
COMPLEXITY: O(1) per insertion
```
### KS Test Algorithm
```
ALGORITHM: ks_statistic
INPUT: sorted_samples1 (List[float]), sorted_samples2 (List[float])
OUTPUT: D statistic (float, 0-1)
PRECONDITION: Both lists are sorted ascending
1. n1 ← len(sorted_samples1)
   n2 ← len(sorted_samples2)
   IF n1 == 0 OR n2 == 0:
       RETURN 0.0
2. i ← 0, j ← 0
   max_diff ← 0.0
   cdf1_val ← 0.0, cdf2_val ← 0.0
3. // Walk through merged sorted values
   WHILE i < n1 OR j < n2:
       IF i < n1 AND (j >= n2 OR sorted_samples1[i] <= sorted_samples2[j]):
           // Process value from sample 1
           cdf1_val ← (i + 1) / n1
           i ← i + 1
       ELSE:
           // Process value from sample 2
           cdf2_val ← (j + 1) / n2
           j ← j + 1
       // Track maximum CDF difference
       diff ← |cdf1_val - cdf2_val|
       max_diff ← max(max_diff, diff)
4. RETURN max_diff
INVARIANT: 0 <= max_diff <= 1
COMPLEXITY: O(n1 + n2)
```
### KS P-Value Calculation
```
ALGORITHM: ks_p_value
INPUT: d_statistic (float), n1 (int), n2 (int)
OUTPUT: p-value (float)
1. // Effective sample size
   n_eff ← sqrt(n1 * n2 / (n1 + n2))
2. // Lambda parameter
   lambda ← (n_eff * d_statistic)²
3. // Series approximation
   p_value ← 0
   FOR k = 1 TO 100:
       term ← 2 * (-1)^(k-1) * exp(-2 * k² * lambda)
       p_value ← p_value + term
       IF |term| < 1e-10:
           BREAK
4. RETURN clamp(p_value, 0, 1)
INVARIANT: p_value decreases monotonically as D increases
APPROXIMATION ERROR: < 1e-10 for large samples
```
---
## Error Handling Matrix
| Error | Detected By | Recovery | User-Visible? |
|-------|-------------|----------|---------------|
| `BaselineNotFound` | `BaselineManager.load_baseline()` | Return None, disable drift detection | No (logged, drift disabled) |
| `InsufficientSamplesForDrift` | `DriftDetectionSampler.run_drift_tests()` | Return result with interpretation, don't alert | No (in drift result) |
| `AlertThresholdExceeded` | `AlertManager.evaluate()` | Fire alert, start cooldown | Yes (via alert callback) |
| `BaselineStale` | `BaselineManager.is_baseline_stale()` | Log warning, continue monitoring | No (logged) |
| `LoggingFailed` | `StructuredRequestLogger._flush()` | Log error, continue serving | No (logged, request still served) |
| Invalid histogram bucket bounds | `LatencyHistogram.__post_init__()` | Sort buckets automatically | No (auto-corrected) |
| Negative latency value | `LatencyHistogram.observe()` | Still record (may indicate timing bug) | No (logged) |
| Feature not in baseline | `DriftDetectionSampler.record_features()` | Silently skip | No |
---
## Implementation Sequence with Checkpoints
### Phase 1: LatencyHistogram with Percentiles (2-3 hours)
**Files:** `01_config.py`, `02_exceptions.py`, `03_histogram_bucket.py`, `04_latency_histogram.py`
**Checkpoint:**
```bash
python -c "
from latency_histogram import LatencyHistogram
# Create histogram
hist = LatatencyHistogram.create_default()
# Record observations
for i in range(100):
    hist.observe(40 + i * 0.5)
# Check percentiles
p50 = hist.percentile(50)
p99 = hist.percentile(99)
print(f'p50: {p50:.1f}ms, p99: {p99:.1f}ms')
assert 60 <= p50 <= 70, 'p50 should be around 65'
assert p99 > 80, 'p99 should be higher'
print('Histogram percentiles passed')
# Test Prometheus export
prom = hist.to_prometheus_format('test_latency', {'version': 'v1'})
assert 'test_latency_bucket' in prom
assert 'test_latency_sum' in prom
print('Prometheus export passed')
"
```
---
### Phase 2: PerVersionLatencyTracker (2-3 hours)
**Files:** `05_per_version_tracker.py`
**Checkpoint:**
```bash
python -c "
from per_version_tracker import PerVersionLatencyTracker
tracker = PerVersionLatencyTracker()
# Record for multiple versions
for i in range(100):
    tracker.record('v1', 40 + i * 0.3, success=True)
    tracker.record('v2', 35 + i * 0.2, success=True)
# Record some errors
for _ in range(5):
    tracker.record('v1', 50, success=False)
# Check percentiles
v1_p50 = tracker.get_percentiles('v1')['p50']
v2_p50 = tracker.get_percentiles('v2')['p50']
print(f'v1 p50: {v1_p50:.1f}ms, v2 p50: {v2_p50:.1f}ms')
# Check error rate
v1_error = tracker.get_error_rate('v1')
print(f'v1 error rate: {v1_error:.2%}')
assert 0.03 <= v1_error <= 0.06, 'Error rate should be ~5%'
# Prometheus export
prom = tracker.to_prometheus_format()
assert 'model_error_rate' in prom
print('PerVersionLatencyTracker passed')
"
```
---
### Phase 3: FeatureBaseline and BaselineManager (2-3 hours)
**Files:** `07_feature_baseline.py`, `08_baseline_manager.py`
**Checkpoint:**
```bash
python -c "
import os
import tempfile
from feature_baseline import FeatureBaseline
from baseline_manager import BaselineManager
# Create baseline
samples = [i * 0.1 for i in range(1000)]
baseline = FeatureBaseline.from_samples('feature_1', samples)
print(f'Mean: {baseline.mean:.2f}, Std: {baseline.std:.2f}')
assert 49 < baseline.mean < 51, 'Mean should be ~50'
assert baseline.sorted_samples[0] == 0.0
# Test BaselineManager
with tempfile.TemporaryDirectory() as tmpdir:
    manager = BaselineManager(storage_dir=tmpdir)
    # Capture baseline
    training_features = {
        'f1': [i * 0.5 for i in range(500)],
        'f2': [i * 0.3 for i in range(500)],
    }
    model_baseline = manager.capture_from_training_data(
        'v1.0.0',
        training_features,
        metadata={'source': 'test'}
    )
    # Load it back
    loaded = manager.load_baseline('v1.0.0')
    assert loaded is not None
    assert 'f1' in loaded.feature_baselines
    print('BaselineManager passed')
"
```
---
### Phase 4: KSTestDetector (3-4 hours)
**Files:** `09_ks_test_detector.py`
**Checkpoint:**
```bash
python -c "
import random
from ks_test_detector import KSTestDetector
from feature_baseline import FeatureBaseline
detector = KSTestDetector()
# Create baseline
baseline_samples = [random.gauss(100, 10) for _ in range(1000)]
baseline = FeatureBaseline.from_samples('test_feature', baseline_samples)
# Test with same distribution (should NOT be significant)
similar_samples = [random.gauss(100, 10) for _ in range(500)]
result = detector.test(baseline, similar_samples, significance_level=0.01)
print(f'Same dist - D: {result.statistic:.4f}, p: {result.p_value:.4f}')
assert not result.is_significant or result.p_value > 0.001, 'Same distribution should not be very significant'
# Test with shifted distribution (should be significant)
shifted_samples = [random.gauss(120, 10) for _ in range(500)]
result = detector.test(baseline, shifted_samples, significance_level=0.01)
print(f'Shifted dist - D: {result.statistic:.4f}, p: {result.p_value:.4f}')
assert result.is_significant, 'Shifted distribution should be significant'
print('KSTestDetector passed')
"
```
---
### Phase 5: DriftDetectionSampler (2-3 hours)
**Files:** `10_drift_sampler.py`
**Checkpoint:**
```bash
python -c "
import random
import asyncio
from drift_sampler import DriftDetectionSampler, SamplingConfig
from feature_baseline import FeatureBaseline
# Create baselines
baselines = {
    'f1': FeatureBaseline.from_samples('f1', [random.gauss(100, 10) for _ in range(1000)]),
    'f2': FeatureBaseline.from_samples('f2', [random.gauss(50, 5) for _ in range(1000)]),
}
config = SamplingConfig(sample_rate=10, min_samples_for_test=50, max_live_samples=1000)
sampler = DriftDetectionSampler(baselines=baselines, config=config)
# Simulate requests
for i in range(1000):
    if sampler.should_sample():
        sampler.record_features({
            'f1': random.gauss(100, 10),
            'f2': random.gauss(50, 5),
        })
# Run drift tests
results = sampler.run_drift_tests()
for name, result in results.items():
    print(f'{name}: D={result.statistic:.4f}, p={result.p_value:.4f}, sig={result.is_significant}')
summary = sampler.get_drift_summary()
print(f'Features monitored: {summary[\"total_features_monitored\"]}')
print(f'Features with drift: {summary[\"features_with_drift\"]}')
print('DriftDetectionSampler passed')
"
```
---
### Phase 6: AlertManager (2-3 hours)
**Files:** `11_alert_rule.py`, `12_alert_manager.py`
**Checkpoint:**
```bash
python -c "
from alert_manager import AlertManager, DEFAULT_DRIFT_RULES
from alert_rule import AlertSeverity
from ks_test_detector import KSTestResult
# Create alert manager
manager = AlertManager(rules=DEFAULT_DRIFT_RULES)
# Create mock drift results
results = {
    'f1': KSTestResult(
        feature_name='f1',
        statistic=0.15,
        p_value=0.005,  # Significant
        is_significant=True,
        baseline_samples=1000,
        live_samples=500,
        interpretation='Significant drift'
    ),
    'f2': KSTestResult(
        feature_name='f2',
        statistic=0.05,
        p_value=0.5,  # Not significant
        is_significant=False,
        baseline_samples=1000,
        live_samples=500,
        interpretation='No drift'
    ),
}
feature_samples = {'f1': 500, 'f2': 500}
# Evaluate alerts
alerts = manager.evaluate(results, feature_samples)
print(f'Alerts fired: {len(alerts)}')
for alert in alerts:
    print(f'  {alert.rule_name}: {alert.message}')
# Get recent alerts
recent = manager.get_recent_alerts()
print(f'Recent alerts: {len(recent)}')
print('AlertManager passed')
"
```
---
### Phase 7: StructuredRequestLogger (2-3 hours)
**Files:** `13_structured_logger.py`
**Checkpoint:**
```bash
python -c "
import os
import tempfile
from structured_logger import StructuredRequestLogger
with tempfile.TemporaryDirectory() as tmpdir:
    logger = StructuredRequestLogger(log_dir=tmpdir, sample_rate=0.5, flush_interval=5)
    # Log some requests
    for i in range(10):
        logger.log(
            request_id=f'req-{i}',
            model_version='v1',
            features={'f1': i * 0.5, 'f2': i * 0.3},
            prediction=[0.1, 0.9],
            latency_ms=40 + i,
            success=True,
            force_log=(i % 3 == 0)  # Force log every 3rd
        )
    logger.finalize()
    # Check log file
    files = os.listdir(tmpdir)
    assert len(files) > 0, 'Log file should exist'
    with open(os.path.join(tmpdir, files[0])) as f:
        lines = f.readlines()
    print(f'Logged {len(lines)} entries')
    assert len(lines) > 0, 'Should have logged entries'
    print('StructuredRequestLogger passed')
"
```
---
### Phase 8: Complete Monitoring Server Integration (2-3 hours)
**Files:** `14_integration.py`
**Checkpoint:**
```bash
# Start server
uvicorn integration:app --host 0.0.0.0 --port 8000 &
sleep 5
# Test health
curl http://localhost:8000/health/liveness
curl http://localhost:8000/health/readiness
# Test metrics
curl http://localhost:8000/metrics
# Test prediction (should record metrics)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"f1": 100, "f2": 50}}'
# Check drift status
curl http://localhost:8000/drift/status
# Check alerts
curl http://localhost:8000/alerts/recent
# Kill server
pkill -f uvicorn
```
---
## Test Specification
### Test 1: Histogram Percentile Accuracy
```python
def test_histogram_percentile_accuracy():
    """Test that histogram percentiles are accurate within bucket error."""
    from latency_histogram import LatencyHistogram
    hist = LatencyHistogram.create_default()
    # Record known values
    for i in range(100):
        hist.observe(float(i))
    # p50 should be close to 49-50
    p50 = hist.percentile(50)
    assert 45 <= p50 <= 55, f"p50={p50} should be around 49-50"
    # p99 should be close to 98-99
    p99 = hist.percentile(99)
    assert 95 <= p99 <= 100, f"p99={p99} should be around 98-99"
```
### Test 2: Per-Version Metrics Isolation
```python
def test_per_version_metrics_isolation():
    """Test that version metrics are independent."""
    from per_version_tracker import PerVersionLatencyTracker
    tracker = PerVersionLatencyTracker()
    # Record different latencies for each version
    for _ in range(100):
        tracker.record('v1', 100.0)
        tracker.record('v2', 50.0)
    v1_p50 = tracker.get_percentiles('v1')['p50']
    v2_p50 = tracker.get_percentiles('v2')['p50']
    # v1 should be ~2x v2
    assert v1_p50 > v2_p50 * 1.5, "Versions should have independent metrics"
```
### Test 3: KS Test Detects Drift
```python
def test_ks_test_detects_drift():
    """Test that KS test detects distribution shift."""
    from ks_test_detector import KSTestDetector
    from feature_baseline import FeatureBaseline
    import random
    detector = KSTestDetector()
    # Baseline: mean=100, std=10
    baseline = FeatureBaseline.from_samples(
        'test', 
        [random.gauss(100, 10) for _ in range(1000)]
    )
    # Live: same distribution - should not detect drift
    same_dist = [random.gauss(100, 10) for _ in range(500)]
    result1 = detector.test(baseline, same_dist)
    # Live: shifted distribution - should detect drift
    shifted_dist = [random.gauss(130, 10) for _ in range(500)]
    result2 = detector.test(baseline, shifted_dist)
    assert result2.statistic > result1.statistic, "Shifted should have higher D"
    assert result2.p_value < result1.p_value, "Shifted should have lower p-value"
```
### Test 4: Reservoir Sampling Representativeness
```python
def test_reservoir_sampling_representative():
    """Test that reservoir sampling produces representative samples."""
    from drift_sampler import FeatureSampleBuffer
    import random
    buffer = FeatureSampleBuffer('test', max_size=100)
    # Add 1000 values from known distribution
    values = [random.gauss(100, 10) for _ in range(1000)]
    for v in values:
        buffer.add(v)
    samples = buffer.get_samples()
    # Sample mean should be close to population mean
    sample_mean = sum(samples) / len(samples)
    pop_mean = sum(values) / len(values)
    assert abs(sample_mean - pop_mean) < 5, "Sample should be representative"
```
### Test 5: Alert Cooldown
```python
@pytest.mark.asyncio
async def test_alert_cooldown():
    """Test that alerts respect cooldown period."""
    from alert_manager import AlertManager
    from alert_rule import AlertRule, AlertSeverity
    from ks_test_detector import KSTestResult
    rule = AlertRule(
        name="test_rule",
        feature_name="f1",
        condition="p_value < 0.01",
        severity=AlertSeverity.WARNING,
        cooldown_seconds=10.0
    )
    manager = AlertManager(rules=[rule])
    result = KSTestResult(
        feature_name='f1',
        statistic=0.2,
        p_value=0.001,
        is_significant=True,
        baseline_samples=1000,
        live_samples=500,
        interpretation='Drift'
    )
    # First evaluation should fire
    alerts1 = manager.evaluate({'f1': result}, {'f1': 500})
    assert len(alerts1) == 1
    # Immediate re-evaluation should NOT fire (cooldown)
    alerts2 = manager.evaluate({'f1': result}, {'f1': 500})
    assert len(alerts2) == 0
```
---
## Performance Targets
| Operation | Target | How to Measure |
|-----------|--------|----------------|
| Latency histogram observe | < 1µs | Time `histogram.observe()` for 10000 iterations |
| Latency percentile calculation | < 5µs | Time `histogram.percentile(99)` |
| Reservoir sample add | < 1µs | Time `buffer.add()` for 10000 iterations |
| KS test (5000 samples) | < 50ms | Time `detector.test()` with 5000 samples |
| Alert evaluation per rule | < 5ms | Time `manager.evaluate()` with 10 features |
| Structured log entry | < 1ms | Time `logger.log()` with force_log=True |
| Prometheus export | < 10ms | Time `tracker.to_prometheus_format()` |
---
## State Machine: Drift Detection Lifecycle
```
                    ┌─────────────┐
                    │  NO_BASELINE │
                    │ (no config)  │
                    └──────┬──────┘
                           │ load_baseline()
                           ▼
              ┌────────────────────────┐
              │    SAMPLING            │
              │ (collecting samples)   │
              └───────────┬────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
        samples < min          samples >= min
              │                       │
              ▼                       ▼
       ┌─────────────┐         ┌─────────────┐
       │ INSUFFICIENT│         │  TEST_READY │
       │ (wait)      │         │ (can test)  │
       └─────────────┘         └──────┬──────┘
                                      │
                          ┌───────────┴───────────┐
                          │                       │
                    p >= threshold          p < threshold
                          │                       │
                          ▼                       ▼
                   ┌─────────────┐         ┌─────────────┐
                   │  NO_DRIFT   │         │ DRIFT_ALERT │
                   │ (continue)  │         │ (alert)     │
                   └─────────────┘         └─────────────┘
                          │                       │
                          │                       │
                          └───────────┬───────────┘
                                      │
                                reset/timeout
                                      │
                                      ▼
                              ┌─────────────┐
                              │  SAMPLING   │
                              │ (continue)  │
                              └─────────────┘
ILLEGAL TRANSITIONS:
- NO_BASELINE → TEST_READY (must have baseline)
- DRIFT_ALERT → NO_DRIFT (must go through sampling)
- INSUFFICIENT → DRIFT_ALERT (must have samples)
```
---
## Concurrency Specification
### Threading Model
```
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Event Loop                          │
│                     (Single-threaded async)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  /predict    │  │  /metrics    │  │  /drift/     │          │
│  │  handler     │  │  handler     │  │  status      │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                  │                  │
│         │ record()        │ to_prometheus()  │ run_tests()      │
│         ▼                 ▼                  ▼                  │
│  ┌──────────────────────────────────────────────────┐          │
│  │         PerVersionLatencyTracker                  │          │
│  │  - defaultdict for per-version histograms        │          │
│  │  - Single event loop = thread-safe               │          │
│  └──────────────────────────────────────────────────┘          │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────┐          │
│  │         DriftDetectionSampler                     │          │
│  │  - Counter is atomic (deterministic sampling)    │          │
│  │  - Buffers per-feature                           │          │
│  └──────────────────────────────────────────────────┘          │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────┐          │
│  │         AlertManager                             │          │
│  │  - Cooldown tracking via timestamp dict          │          │
│  │  - Recent alerts list (bounded)                  │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
### Thread Safety Guarantees
| Component | Thread Safety | Reason |
|-----------|--------------|--------|
| `LatencyHistogram` | NOT thread-safe | Single event loop access |
| `PerVersionLatencyTracker` | Safe for async | defaultdict + single loop |
| `DriftDetectionSampler` | Safe for async | Counter increment is atomic-ish in Python |
| `AlertManager` | Safe for async | Single event loop access |
| `StructuredRequestLogger` | Safe for async | Buffer + flush in single loop |
| `BaselineManager` | Safe for async | Dict access in single loop |
---

![Latency Histogram Bucket Structure](./diagrams/tdd-diag-034.svg)

![Grafana Dashboard Layout](./diagrams/diag-m5-grafana-dashboard.svg)

---

![Percentile Calculation from Histogram](./diagrams/tdd-diag-035.svg)

---

![Prediction Distribution Monitoring](./diagrams/tdd-diag-036.svg)

---

![KS Test: Comparing Distributions](./diagrams/tdd-diag-037.svg)

---

![Drift Detection Pipeline](./diagrams/tdd-diag-038.svg)

---

![Reservoir Sampling for Drift Buffer](./diagrams/tdd-diag-039.svg)

---
---

![Alert Threshold Tuning Tradeoff](./diagrams/tdd-diag-041.svg)

---
---

![Structured Request Log Entry](./diagrams/tdd-diag-043.svg)

---

![Complete Monitoring Stack](./diagrams/tdd-diag-044.svg)

---
[[CRITERIA_JSON: {"module_id": "ml-model-serving-m5", "criteria": ["LatencyHistogram uses cumulative buckets with observe() incrementing all buckets where latency < upper_bound for Prometheus compatibility", "LatencyHistogram.percentile(p) computes linear interpolation within containing bucket from cumulative counts, returning approximate value at percentile", "PerVersionLatencyTracker maintains separate LatencyHistogram per model version with record(), get_percentiles(), get_error_rate(), get_throughput_rps() methods", "PerVersionLatencyTracker.to_prometheus_format() exports _bucket, _sum, _count metrics with version labels", "FeatureBaseline.from_samples() creates baseline from training samples with downsample to max_baseline_samples (default 10000), computing min/max/mean/std and pre-sorting for CDF", "BaselineManager stores/loads baselines as JSON files with model_version, capture_time, feature_baselines dict, and metadata", "BaselineManager.get_baseline_age_days() returns age in days; is_baseline_stale() checks against max_age_days threshold (default 90)", "KSTestDetector.ks_statistic() computes maximum CDF difference D between two sorted sample lists in O(n1+n2) time", "KSTestDetector.ks_p_value() approximates p-value from D statistic using series expansion of Kolmogorov distribution", "KSTestDetector.test() returns KSTestResult with feature_name, statistic, p_value, is_significant (p < significance_level), interpretation, and drift_direction", "DriftDetectionSampler uses deterministic should_sample() (every Nth request) and reservoir sampling in FeatureSampleBuffer.add()", "DriftDetectionSampler.run_drift_tests() iterates features, checking min_samples_for_test before running KSTestDetector.test()", "DriftDetectionSampler.get_drift_summary() returns total_features_monitored, features_with_drift, drift_percentage, drifted_feature_names", "AlertManager.evaluate() checks per-feature and aggregate rules with cooldown tracking via last_alert_time dict", "AlertRule.matches_condition() parses 'p_value < X' and 'ks_statistic > Y' conditions, returning (is_triggered, value, threshold)", "Alert severity levels (INFO, WARNING, CRITICAL) with configurable cooldown_seconds per rule", "StructuredRequestLogger samples configurable percentage with should_log(), force_log parameter for drift-sampled requests", "StructuredLogEntry captures timestamp, request_id, model_version, features, features_hash, prediction, latency_ms, success, was_sampled_for_drift", "JSON lines format (.jsonl) with date-based file naming for structured logs", "HistogramBucket upper_bound and cumulative count; LatencyHistogram buckets sorted by upper_bound", "DEFAULT_DRIFT_RULES include p_value_critical (p < 0.001, CRITICAL), p_value_warning (p < 0.01, WARNING), aggregate_drift (fraction > 0.2, WARNING)"]}]
<!-- END_TDD_MOD -->


# Project Structure: ML Model Serving API
## Directory Tree
```
ml-model-serving-api/
├── 01_config.py                    # [M1] Configuration constants
├── 02_exceptions.py                # [M1] Custom exception hierarchy
├── 03_device.py                    # [M1] Device detection, memory budgeting
├── 04_model_loader.py              # [M1] Model loading with eval mode
├── 05_warmup.py                    # [M1] CUDA warmup, kernel caching
├── 06_schema.py                    # [M1] Input schema definitions
├── 07_serialization.py             # [M1] orjson serialization utilities
├── 08_health.py                    # [M1] Liveness and readiness probes
├── 09_model_container.py           # [M1] LoadedModel dataclass
├── 10_lifespan.py                  # [M1] FastAPI lifespan handler
├── 11_endpoints.py                 # [M1] HTTP endpoints
├── 12_main.py                      # [M1] Application entry point
│
├── m2_batching/
│   ├── __init__.py                 # [M2] Module exports
│   ├── 01_config.py                # [M2] Batching configuration
│   ├── 02_exceptions.py            # [M2] Batching exceptions
│   ├── 03_batch_request.py         # [M2] BatchRequest dataclass
│   ├── 04_request_queue.py         # [M2] Async queue with backpressure
│   ├── 05_padder.py                # [M2] Variable-length input padding
│   ├── 06_batcher.py               # [M2] DynamicBatcher core logic
│   ├── 07_result_router.py         # [M2] Result routing to callers
│   ├── 08_metrics.py               # [M2] BatcherMetrics tracking
│   ├── 09_benchmark.py             # [M2] Throughput benchmark tools
│   └── 10_integration.py           # [M2] FastAPI batching integration
│
├── m3_versioning/
│   ├── __init__.py                 # [M3] Module exports
│   ├── 01_config.py                # [M3] Versioning configuration
│   ├── 02_exceptions.py            # [M3] Versioning exceptions
│   ├── 03_schema.py                # [M3] ModelSchema, compatibility checks
│   ├── 04_model_version.py         # [M3] ModelVersion dataclass
│   ├── 05_model_registry.py        # [M3] ModelRegistry management
│   ├── 06_atomic_pointer.py        # [M3] AtomicVersionPointer swaps
│   ├── 07_request_tracker.py       # [M3] RequestTracker for draining
│   ├── 08_hot_swap_manager.py      # [M3] HotSwapManager orchestration
│   ├── 09_rollback_monitor.py      # [M3] RollbackMonitor triggers
│   ├── 10_version_router.py        # [M3] Version routing logic
│   └── 11_integration.py           # [M3] FastAPI versioning integration
│
├── m4_ab_testing/
│   ├── __init__.py                 # [M4] Module exports
│   ├── 01_config.py                # [M4] A/B testing configuration
│   ├── 02_exceptions.py            # [M4] A/B testing exceptions
│   ├── 03_traffic_allocation.py    # [M4] TrafficAllocation dataclass
│   ├── 04_traffic_splitter.py      # [M4] TrafficSplitter (SHA-256 hash)
│   ├── 05_per_version_metrics.py   # [M4] PerVersionMetrics collection
│   ├── 06_statistical_analyzer.py  # [M4] StatisticalAnalyzer (t-test, prop test)
│   ├── 07_canary_config.py         # [M4] RolloutConfig dataclass
│   ├── 08_canary_controller.py     # [M4] CanaryRolloutController
│   ├── 09_health_checker.py        # [M4] CanaryHealthChecker
│   ├── 10_experiment_config.py     # [M4] ExperimentConfig dataclass
│   ├── 11_experiment_manager.py    # [M4] ExperimentManager orchestration
│   └── 12_integration.py           # [M4] FastAPI A/B testing integration
│
├── m5_monitoring/
│   ├── __init__.py                 # [M5] Module exports
│   ├── 01_config.py                # [M5] Monitoring configuration
│   ├── 02_exceptions.py            # [M5] Monitoring exceptions
│   ├── 03_histogram_bucket.py      # [M5] HistogramBucket utilities
│   ├── 04_latency_histogram.py     # [M5] LatencyHistogram (Prometheus-style)
│   ├── 05_per_version_tracker.py   # [M5] PerVersionLatencyTracker
│   ├── 06_prediction_monitor.py    # [M5] PredictionDistributionMonitor
│   ├── 07_feature_baseline.py      # [M5] FeatureBaseline statistics
│   ├── 08_baseline_manager.py      # [M5] BaselineManager persistence
│   ├── 09_ks_test_detector.py      # [M5] KSTestDetector (data drift)
│   ├── 10_drift_sampler.py         # [M5] DriftDetectionSampler (reservoir)
│   ├── 11_alert_rule.py            # [M5] AlertRule definitions
│   ├── 12_alert_manager.py         # [M5] AlertManager with cooldown
│   ├── 13_structured_logger.py     # [M5] StructuredRequestLogger
│   └── 14_integration.py           # [M5] FastAPI monitoring integration
│
├── models/                         # [Runtime] Model artifacts directory
│   └── .gitkeep                    # Placeholder for model storage
│
├── baselines/                      # [Runtime] Feature baselines directory
│   └── .gitkeep                    # Placeholder for baseline storage
│
├── logs/                           # [Runtime] Application logs directory
│   └── .gitkeep                    # Placeholder for log files
│
├── tests/                          # Test suite
│   ├── __init__.py                 # Test package init
│   ├── conftest.py                 # Pytest fixtures
│   ├── test_m1_core/               # M1 unit tests
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   ├── test_device.py
│   │   ├── test_model_loader.py
│   │   ├── test_warmup.py
│   │   ├── test_schema.py
│   │   ├── test_serialization.py
│   │   ├── test_health.py
│   │   ├── test_lifespan.py
│   │   └── test_endpoints.py
│   ├── test_m2_batching/           # M2 unit tests
│   │   ├── __init__.py
│   │   ├── test_batch_request.py
│   │   ├── test_request_queue.py
│   │   ├── test_padder.py
│   │   ├── test_batcher.py
│   │   ├── test_result_router.py
│   │   └── test_metrics.py
│   ├── test_m3_versioning/         # M3 unit tests
│   │   ├── __init__.py
│   │   ├── test_schema.py
│   │   ├── test_model_version.py
│   │   ├── test_model_registry.py
│   │   ├── test_atomic_pointer.py
│   │   ├── test_request_tracker.py
│   │   ├── test_hot_swap_manager.py
│   │   ├── test_rollback_monitor.py
│   │   └── test_version_router.py
│   ├── test_m4_ab_testing/         # M4 unit tests
│   │   ├── __init__.py
│   │   ├── test_traffic_allocation.py
│   │   ├── test_traffic_splitter.py
│   │   ├── test_per_version_metrics.py
│   │   ├── test_statistical_analyzer.py
│   │   ├── test_canary_controller.py
│   │   ├── test_health_checker.py
│   │   └── test_experiment_manager.py
│   ├── test_m5_monitoring/         # M5 unit tests
│   │   ├── __init__.py
│   │   ├── test_histogram_bucket.py
│   │   ├── test_latency_histogram.py
│   │   ├── test_per_version_tracker.py
│   │   ├── test_prediction_monitor.py
│   │   ├── test_baseline_manager.py
│   │   ├── test_ks_test_detector.py
│   │   ├── test_drift_sampler.py
│   │   ├── test_alert_manager.py
│   │   └── test_structured_logger.py
│   └── integration/                # Integration tests
│       ├── __init__.py
│       ├── test_batching_integration.py
│       ├── test_versioning_integration.py
│       ├── test_ab_testing_integration.py
│       └── test_monitoring_integration.py
│
├── scripts/                        # Utility scripts
│   ├── benchmark_batching.py       # [M2] Batching performance benchmark
│   ├── export_baseline.py          # [M5] Baseline export utility
│   └── load_test.py                # Load testing script
│
├── configs/                        # Configuration files
│   ├── default.yaml                # Default configuration
│   ├── production.yaml             # Production overrides
│   └── staging.yaml                # Staging overrides
│
├── .gitignore                      # Git ignore patterns
├── .python-version                 # Python version specification
├── requirements.txt                # Production dependencies
├── requirements-dev.txt            # Development dependencies
├── setup.py                        # Package setup
├── pyproject.toml                  # Project metadata (PEP 517)
├── Makefile                        # Build and run commands
├── README.md                       # Project documentation
├── LICENSE                         # License file
└── Dockerfile                      # Container definition
```
## Creation Order
### Phase 1: Project Foundation (Files 1-12)
1. `.gitignore` - Git ignore patterns
2. `.python-version` - Python version pinning
3. `requirements.txt` - Core dependencies (fastapi, uvicorn, torch, onnxruntime, orjson, numpy, scipy, pyyaml)
4. `requirements-dev.txt` - Dev dependencies (pytest, pytest-asyncio, httpx, mypy, ruff)
5. `pyproject.toml` - Project metadata and tool config
6. `setup.py` - Package installation
7. `Makefile` - Build targets (install, test, run, lint, docker-build)
8. `README.md` - Documentation
9. `LICENSE` - License
10. `Dockerfile` - Container build
11. `configs/default.yaml` - Default config
12. `configs/production.yaml`, `configs/staging.yaml` - Environment configs
### Phase 2: Core Serving (M1) (Files 13-24)
13. `01_config.py` - Configuration constants
14. `02_exceptions.py` - Exception hierarchy
15. `03_device.py` - Device detection
16. `06_schema.py` - Input schemas
17. `07_serialization.py` - orjson utilities
18. `04_model_loader.py` - Model loading
19. `05_warmup.py` - CUDA warmup
20. `09_model_container.py` - LoadedModel dataclass
21. `08_health.py` - Health probes
22. `10_lifespan.py` - Lifespan handler
23. `11_endpoints.py` - HTTP endpoints
24. `12_main.py` - Application entry
### Phase 3: Batching Module (M2) (Files 25-35)
25. `m2_batching/__init__.py`
26. `m2_batching/01_config.py` - Batching config
27. `m2_batching/02_exceptions.py` - Batching exceptions
28. `m2_batching/03_batch_request.py` - BatchRequest dataclass
29. `m2_batching/04_request_queue.py` - Async queue
30. `m2_batching/05_padder.py` - Padding logic
31. `m2_batching/06_batcher.py` - DynamicBatcher
32. `m2_batching/07_result_router.py` - Result routing
33. `m2_batching/08_metrics.py` - Metrics
34. `m2_batching/09_benchmark.py` - Benchmarks
35. `m2_batching/10_integration.py` - FastAPI integration
### Phase 4: Versioning Module (M3) (Files 36-47)
36. `m3_versioning/__init__.py`
37. `m3_versioning/01_config.py` - Versioning config
38. `m3_versioning/02_exceptions.py` - Versioning exceptions
39. `m3_versioning/03_schema.py` - Schema compatibility
40. `m3_versioning/04_model_version.py` - ModelVersion
41. `m3_versioning/05_model_registry.py` - Registry
42. `m3_versioning/06_atomic_pointer.py` - Atomic swaps
43. `m3_versioning/07_request_tracker.py` - Request tracking
44. `m3_versioning/08_hot_swap_manager.py` - Hot swap
45. `m3_versioning/09_rollback_monitor.py` - Rollback
46. `m3_versioning/10_version_router.py` - Version routing
47. `m3_versioning/11_integration.py` - FastAPI integration
### Phase 5: A/B Testing Module (M4) (Files 48-60)
48. `m4_ab_testing/__init__.py`
49. `m4_ab_testing/01_config.py` - A/B config
50. `m4_ab_testing/02_exceptions.py` - A/B exceptions
51. `m4_ab_testing/03_traffic_allocation.py` - Traffic allocation
52. `m4_ab_testing/04_traffic_splitter.py` - Traffic splitter
53. `m4_ab_testing/05_per_version_metrics.py` - Per-version metrics
54. `m4_ab_testing/06_statistical_analyzer.py` - Statistical analysis
55. `m4_ab_testing/07_canary_config.py` - Canary config
56. `m4_ab_testing/08_canary_controller.py` - Canary controller
57. `m4_ab_testing/09_health_checker.py` - Health checker
58. `m4_ab_testing/10_experiment_config.py` - Experiment config
59. `m4_ab_testing/11_experiment_manager.py` - Experiment manager
60. `m4_ab_testing/12_integration.py` - FastAPI integration
### Phase 6: Monitoring Module (M5) (Files 61-75)
61. `m5_monitoring/__init__.py`
62. `m5_monitoring/01_config.py` - Monitoring config
63. `m5_monitoring/02_exceptions.py` - Monitoring exceptions
64. `m5_monitoring/03_histogram_bucket.py` - Histogram bucket
65. `m5_monitoring/04_latency_histogram.py` - Latency histogram
66. `m5_monitoring/05_per_version_tracker.py` - Version tracker
67. `m5_monitoring/06_prediction_monitor.py` - Prediction monitor
68. `m5_monitoring/07_feature_baseline.py` - Feature baseline
69. `m5_monitoring/08_baseline_manager.py` - Baseline manager
70. `m5_monitoring/09_ks_test_detector.py` - KS test detector
71. `m5_monitoring/10_drift_sampler.py` - Drift sampler
72. `m5_monitoring/11_alert_rule.py` - Alert rules
73. `m5_monitoring/12_alert_manager.py` - Alert manager
74. `m5_monitoring/13_structured_logger.py` - Structured logging
75. `m5_monitoring/14_integration.py` - FastAPI integration
### Phase 7: Runtime Directories & Utilities (Files 76-82)
76. `models/.gitkeep` - Models directory
77. `baselines/.gitkeep` - Baselines directory
78. `logs/.gitkeep` - Logs directory
79. `scripts/benchmark_batching.py` - Batching benchmark
80. `scripts/export_baseline.py` - Baseline export
81. `scripts/load_test.py` - Load testing
82. Root `__init__.py` files for test packages
### Phase 8: Test Suite (Files 83+)
83. `tests/conftest.py` - Pytest fixtures
84-95. `tests/test_m1_core/*.py` - M1 tests
96-106. `tests/test_m2_batching/*.py` - M2 tests
107-118. `tests/test_m3_versioning/*.py` - M3 tests
119-130. `tests/test_m4_ab_testing/*.py` - M4 tests
131-143. `tests/test_m5_monitoring/*.py` - M5 tests
144-148. `tests/integration/*.py` - Integration tests
## File Count Summary
| Category | Count |
|----------|-------|
| Core Serving (M1) | 12 |
| Batching Module (M2) | 11 |
| Versioning Module (M3) | 12 |
| A/B Testing Module (M4) | 13 |
| Monitoring Module (M5) | 15 |
| Project Config/Build | 12 |
| Runtime Directories | 3 |
| Utility Scripts | 3 |
| Test Files | 36+ |
| **Total Source Files** | **66** |
| **Total with Tests** | **102+** |
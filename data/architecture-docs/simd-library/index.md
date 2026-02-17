# simd-library







# TDD

To provide a zero-cost abstraction layer for SIMD operations across x86 (SSE/AVX/AVX-512) and ARM (NEON) architectures. The library focuses on deterministic performance, compile-time dispatch optimization, and strict memory alignment safety to prevent segmentation faults and bus errors during high-throughput data processing.



<!-- TDD_MOD_ID: mod-hardware-dispatch -->
# Module: Hardware Detection & Dispatch Engine (HDDE)

## 1. Technical Specification
The Hardware Detection & Dispatch Engine (HDDE) serves as the foundational decision-maker for the `simd-library`. Its primary responsibility is the identification of CPU Instruction Set Architecture (ISA) extensions (e.g., AVX-512, NEON) and the subsequent orchestration of the "Best-Fit" execution path.

The engine must resolve two conflicting requirements:
1.  **Runtime Versatility**: The ability to run a single binary on multiple CPU generations.
2.  **Zero-Overhead Execution**: Ensuring that the dispatch mechanism does not introduce branch mispredictions or pipeline stalls in the "Hot Path."

## 2. Abstraction Layers
The HDDE is structured into three discrete layers:
*   **Layer 0: The Prober**: Architecture-specific assembly (`cpuid` for x86, `mrs` or `/proc/self/auxv` for ARM) that extracts raw feature bits.
*   **Layer 1: The Registry**: A thread-safe, immutable bit-field representing the global hardware state.
*   **Layer 2: The Dispatcher**: A functional bridge that uses either Function Multiversioning (FMV) or static function pointers to route data to the optimal SIMD implementation.

## 3. Struct & Interface Definitions

### 3.1. Hardware State Representation
To ensure cache-line efficiency, the hardware state is encapsulated in a bit-packed structure, aligned to avoid false sharing.

```cpp
// Alignment to 64 bytes to prevent cache-line contention in multi-threaded environments
struct alignas(64) CPUFeatureSet {
    uint64_t x86_extensions; // Bits for SSE, AVX, AVX2, AVX512(F,CD,BW,DQ,VL)
    uint64_t arm_extensions; // Bits for NEON, ASIMD, SVE, SVE2
    uint32_t cache_line_size;
    uint32_t logical_cores;
    
    // Memory Alignment: 
    // [8 bytes: x86] [8 bytes: arm] [4 bytes: cache] [4 bytes: cores] 
    // [40 bytes: padding/reserved]
};
```

### 3.2. Dispatcher Interface
The dispatcher utilizes a function pointer table or a static global pointer resolved at load-time (via `__attribute__((constructor))`).

```cpp
typedef void (*SimdKernel)(const float* __restrict src, float* __restrict dst, size_t len);

struct KernelRegistry {
    SimdKernel current_best_kernel;
    uint32_t isa_level; // Numeric rank for comparison
};
```

{{DIAGRAM:hdde-class-hierarchy}}

![Component Interaction](./diagrams/tdd-hdde-001.svg)


## 4. Algorithm Pseudo-code

### 4.1. The Hot Path: Static-Lazy Dispatch
This logic occurs at the first invocation of a SIMD-accelerated function. We use the "Double-Checked Locking" pattern or atomic `std::call_once`.

```cpp
// Global atomic pointer to the active kernel
static std::atomic<SimdKernel> global_dispatch_ptr{nullptr};

void dispatch_init() {
    CPUFeatureSet features = probe_hardware();
    
    if (features.x86_extensions & FEATURE_AVX512) {
        global_dispatch_ptr.store(kernel_avx512_impl, std::memory_order_release);
    } else if (features.x86_extensions & FEATURE_AVX2) {
        global_dispatch_ptr.store(kernel_avx2_impl, std::memory_order_release);
    } else {
        global_dispatch_ptr.store(kernel_scalar_fallback, std::memory_order_release);
    }
}

// Hot Path Wrapper
void execute_simd_op(const float* src, float* dst, size_t len) {
    // Micro-optimization: Use local pointer to avoid multiple atomic loads
    SimdKernel func = global_dispatch_ptr.load(std::memory_order_acquire);
    
    if (__builtin_expect(!func, 0)) {
        dispatch_init();
        func = global_dispatch_ptr.load(std::memory_order_relaxed);
    }
    
    // Tail-call optimization target
    func(src, dst, len);
}
```

## 5. Engineering Constraints & Hazards

### 5.1. Concurrency & Initialization
*   **Race Conditions**: Hardware probing must be idempotent and thread-safe. Static constructors are preferred for deterministic initialization before `main()` is entered.
*   **Memory Ordering**: The `global_dispatch_ptr` must use `memory_order_acquire/release` to ensure that once the function pointer is visible, the underlying ISA-specific machine code is also fully loaded and executable.

### 5.2. Memory & Alignment
*   **Data Alignment**: All SIMD kernels assume `alignas(32)` or `alignas(64)` for input buffers. The HDDE must provide a `malloc` wrapper or an alignment validator.
*   **Fault Handling**: If a kernel is dispatched to an ISA not supported by the hardware (e.g., AVX-512 on an AVX2 chip), the resulting `SIGILL` must be prevented via strict runtime masking in `probe_hardware()`.

### 5.3. Micro-Optimization Corner
1.  **Branch Target Buffer (BTB) Warming**: For critical loops, the HDDE should be initialized during the application's "cold start" phase to ensure the BTB is primed with the correct kernel address, reducing the cost of the indirect function call.
2.  **IFUNC (Indirect Functions)**: On GNU/Linux systems, the HDDE will leverage `__attribute__((ifunc))` to allow the dynamic linker (ld.so) to resolve the optimal function at load-time. This replaces the `if(!func)` check with a direct PLT (Procedure Linkage Table) entry, reducing the dispatch overhead to exactly zero at runtime.
3.  **No-Inline Policy**: SIMD kernels themselves should be marked `__attribute__((noinline))` to prevent the compiler from bloating the dispatcher with multiple ISA versions, which would trash the Instruction Cache (I-Cache).

{{DIAGRAM:hdde-sequence-ifunc}}
{{DIAGRAM:tdd-hdde-002}}


<!-- TDD_MOD_ID: mod-memory-management -->
# Module: Aligned Memory & Buffer Manager

## 1. Technical Specification
The Aligned Memory & Buffer Manager (AMBM) provides a deterministic, high-performance memory allocation subsystem specifically designed for SIMD workloads. Standard heap allocators (e.g., `malloc`) typically guarantee 8 or 16-byte alignment; however, modern SIMD instructions (AVX-256, AVX-512) require 32 or 64-byte alignment to avoid `GPF` (General Protection Faults) or performance-degrading split-load penalties.

The AMBM's primary responsibilities include:
1.  **Strict Alignment Enforcement**: Guarantees alignment on $2^n$ boundaries (16, 32, 64 bytes).
2.  **Metadata Management**: Efficiently tracking "original" pointers vs. "aligned" pointers for safe deallocation.
3.  **Buffer Lifecycle Control**: Providing RAII-compliant wrappers to prevent memory leaks in complex SIMD pipelines.

## 2. Abstraction Layers
The AMBM is divided into three functional strata:
*   **Layer 0: OS Abstraction (Low Level)**: Wraps platform-specific calls like `posix_memalign` (POSIX), `_aligned_malloc` (MSVC), or `mmap` with `MAP_HUGETLB`.
*   **Layer 1: Aligned Pointer Arithmetic (Manual Alignment)**: Logic to transform an arbitrary pointer into an aligned one by over-allocating and storing metadata.
*   **Layer 2: Buffer Templates (High Level)**: C++ templates (`AlignedVector<T>`) that interface with the Hardware Detection Engine to determine optimal alignment at compile/link time.

## 3. Struct & Interface Definitions

### 3.1. Aligned Block Metadata
To support manual alignment on platforms without intrinsic aligned allocators, each block must store the offset to the original heap-allocated address.

```cpp
// Packed to 16 bytes to ensure the header itself doesn't cause 
// massive fragmentation.
struct AlignedBlockHeader {
    void* original_ptr;      // 8 bytes: Address returned by malloc()
    size_t requested_size;   // 8 bytes: Size in bytes
    uint32_t alignment_offset; // 4 bytes: Distance from original to aligned
    uint32_t magic_canary;   // 4 bytes: Debug-only validity check (0xDEADBEEF)
};
```

### 3.2. Primary Manager Interface
```cpp
class AlignedBufferManager {
public:
    /**
     * @brief Allocates memory aligned to the specified boundary.
     * @param size Total bytes to allocate.
     * @param alignment Must be a power of 2 (16, 32, 64).
     */
    static void* Allocate(size_t size, size_t alignment);

    /**
     * @brief Frees memory allocated via Allocate.
     */
    static void Free(void* aligned_ptr);

    /**
     * @brief Validates if a pointer is aligned to a specific boundary.
     */
    static bool IsAligned(const void* ptr, size_t alignment) noexcept {
        return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
    }
};
```

{{DIAGRAM:ambm-allocation-flow}}
{{DIAGRAM:tdd-ambm-001}}

## 4. Algorithm Pseudo-code

### 4.1. The Hot Path: Manual Alignment Logic
When `_aligned_malloc` is unavailable, we use the "Over-allocation" strategy.

```cpp
void* AlignedBufferManager::Allocate(size_t size, size_t alignment) {
    // 1. Calculate total size including potential padding and metadata
    // alignment - 1 (max padding) + sizeof(void*) (to store original ptr)
    size_t total_size = size + alignment + sizeof(void*);

    // 2. Standard heap allocation
    void* raw_ptr = std::malloc(total_size);
    if (!raw_ptr) return nullptr;

    // 3. Pointer Arithmetic
    // Move forward by size of a pointer, then align up
    uintptr_t raw_addr = reinterpret_cast<uintptr_t>(raw_ptr);
    uintptr_t start_addr = raw_addr + sizeof(void*);
    uintptr_t aligned_addr = (start_addr + alignment - 1) & ~(alignment - 1);

    // 4. Metadata Storage
    // Store original_ptr right before the aligned_addr
    void** metadata_loc = reinterpret_cast<void**>(aligned_addr - sizeof(void*));
    *metadata_loc = raw_ptr;

    return reinterpret_cast<void*>(aligned_addr);
}

void AlignedBufferManager::Free(void* aligned_ptr) {
    if (!aligned_ptr) return;

    // Retrieve original pointer from the slot immediately preceding the aligned address
    void** metadata_loc = reinterpret_cast<void**>(reinterpret_cast<uintptr_t>(aligned_ptr) - sizeof(void*));
    void* original_ptr = *metadata_loc;

    std::free(original_ptr);
}
```

## 5. Engineering Constraints & Hazards

### 5.1. Performance & Cache Locality
*   **False Sharing**: If multiple `AlignedBlockHeaders` are allocated on the same cache line as frequently written data in a multi-threaded context, performance will degrade.
*   **TLB Misses**: For very large buffers (e.g., > 2MB), the manager should optionally use Huge Pages (`mmap` with `MAP_HUGETLB`) to reduce Translation Lookaside Buffer (TLB) pressure.

### 5.2. Safety Hazards
*   **Pointer Mismatch**: Passing a pointer allocated with `std::malloc` to `AlignedBufferManager::Free` will result in a segmentation fault or heap corruption, as it will attempt to read metadata from invalid memory.
*   **Alignment Granularity**: Requesting an alignment that is not a power of 2 will result in undefined behavior in the bitwise logic (`& ~(alignment - 1)`).

### 5.3. Micro-Optimization Corner
1.  **Prefetching Hint**: The `Allocate` function should ideally call `_mm_prefetch` (or equivalent) on the first few cache lines of the returned buffer to warm the L1 cache before the SIMD kernel begins execution.
2.  **Size Rounding**: To prevent "tail-end" complexity in SIMD loops (handling the remaining `N % vector_width` elements), the manager should round up the `requested_size` to the nearest multiple of the vector width (e.g., 64 bytes for AVX-512). This allows for "over-reading" into safe padding without triggering page faults.
3.  **Zero-Initialization (Lazy)**: Use `mmap` with `MAP_ANONYMOUS` which utilizes the OS "zero-page" mechanism. This avoids the cost of `memset(0)` until the memory is actually touched.

{{DIAGRAM:ambm-memory-hierarchy}}
{{DIAGRAM:tdd-ambm-002}}


<!-- TDD_MOD_ID: mod-vector-abstraction -->
<!-- TDD_MOD_ID: mod-unified-vector-interface -->
# Module: Unified Vector Interface (UVI)

## 1. Technical Specification
The Unified Vector Interface (UVI) provides the high-level, type-safe API for the `simd-library`. While the **HDDE** handles dispatch and the **AMBM** handles memory, the UVI defines the semantic operations performed on data. It abstracts architecture-specific SIMD registers (e.g., `ymm` on x86, `v` on ARM) into a uniform `Vector<T, N>` template.

The UVI's primary requirements are:
1.  **Semantic Uniformity**: A single source-code expression (e.g., `a + b`) must compile to the optimal ISA-specific intrinsic (e.g., `_mm256_add_ps` or `vaddq_f32`).
2.  **Type Safety**: Prevent illegal operations (e.g., adding a 256-bit float vector to a 128-bit integer vector) at compile time.
3.  **Zero-Overhead Wrappers**: Ensure that the abstraction layer collapses into raw assembly with no temporary object overhead or unnecessary register spills.

## 2. Abstraction Layers
The UVI is architected in a "Trait-to-Implementation" hierarchy:
*   **Layer 0: Semantic Traits**: A set of template specializations that map `<Type, Width>` pairs to underlying intrinsic types (e.g., `<float, 8>` -> `__m256`).
*   **Layer 1: Intrinsic Wrappers**: Inline functions that wrap raw intrinsics in a unified naming convention (`simd_add`, `simd_load`).
*   **Layer 2: The Vector Object**: The public-facing `Vector<T, N>` class that provides operator overloads and functional programming primitives (map, reduce).

## 3. Struct & Interface Definitions

### 3.1. Vector Storage & Alignment
The `Vector` struct is the core primitive. It must be explicitly aligned to ensure it can be mapped directly to SIMD registers.

```cpp
template <typename T, size_t N>
struct alignas(sizeof(T) * N) Vector {
    // The underlying hardware register type (resolved via traits)
    using InternalType = typename SimdTraits<T, N>::NativeType;
    
    InternalType data;

    // Constraints: 
    // 1. N must be a power of 2.
    // 2. sizeof(T) * N must be supported by the current ISA (16, 32, 64 bytes).
    
    // Interface
    static Vector Load(const T* ptr);
    static Vector LoadAligned(const T* ptr);
    void Store(T* ptr) const;
};
```

### 3.2. SimdTraits Specialization (Example)
Mapping logic used by the compiler to select the register width.

```cpp
// Generic template
template <typename T, size_t N> struct SimdTraits;

// x86 AVX Specialization
template <> struct SimdTraits<float, 8> {
    using NativeType = __m256;
    static constexpr size_t Alignment = 32;
    static constexpr const char* ISA = "AVX/AVX2";
};

// ARM NEON Specialization
template <> struct SimdTraits<float, 4> {
    using NativeType = float32x4_t;
    static constexpr size_t Alignment = 16;
    static constexpr const char* ISA = "NEON";
};
```

{{DIAGRAM:uvi-class-abstraction}}
{{DIAGRAM:tdd-uvi-001}}

## 4. Algorithm Pseudo-code

### 4.1. The Hot Path: Generic Vectorized Accumulation
This pseudo-code demonstrates how the UVI interacts with the AMBM to process an array of data using the widest available vector.

```cpp
template <typename T>
void vector_add_pipeline(const T* __restrict a, const T* __restrict b, T* __restrict res, size_t count) {
    // 1. Determine optimal width from HDDE
    constexpr size_t W = HardwareCapabilities::GetOptimalWidth<T>();
    using VType = Vector<T, W>;

    size_t i = 0;
    // 2. Main Loop: Process W elements per iteration
    for (; i <= count - W; i += W) {
        // Load (Assume AMBM provided aligned pointers)
        VType va = VType::LoadAligned(&a[i]);
        VType vb = VType::LoadAligned(&b[i]);
        
        // Operation (Overloaded operator + calls underlying intrinsics)
        VType vr = va + vb;
        
        // Store
        vr.Store(&res[i]);
    }

    // 3. Tail Handling: Scalar fallback for remaining (count % W) elements
    for (; i < count; ++i) {
        res[i] = a[i] + b[i];
    }
}
```

## 5. Engineering Constraints & Hazards

### 5.1. Concurrency: Register Pressure
*   **Hazard**: Over-using complex UVI objects in a single function can lead to "Register Spilling," where the compiler is forced to move SIMD register data to the stack (memory).
*   **Mitigation**: UVI functions must be marked `__attribute__((always_inline))` and the number of live `Vector` objects in a scope should be monitored.

### 5.2. Memory: Strict Aliasing & Restrict
*   **Constraint**: The UVI assumes that input and output buffers do not overlap.
*   **Logic**: Use the `__restrict` keyword in all UVI interfaces to allow the compiler to reorder instructions without worrying about pointer aliasing, which is critical for pipelining SIMD loads/stores.

### 5.3. Micro-Optimization Corner
1.  **FMA (Fused Multiply-Add)**: The UVI should provide a `Vector::MultiplyAdd(a, b, c)` method. On supported hardware (AVX2+), this maps to a single `vfmadd` instruction ($a \times b + c$), which has the same latency as a single addition but double the throughput and higher numerical precision.
2.  **Loop Unrolling**: For the Hot Path, the UVI recommends unrolling loops by a factor of 2 or 4 (e.g., processing $4 \times 512$ bits per iteration). This hides the latency of the `Load` instructions by utilizing the CPU's multiple execution ports.
3.  **Non-Temporal Stores**: For very large datasets that exceed the L3 cache, the UVI provides `StoreNT()`. This uses "Streaming Stores" (`movntps`) which bypass the cache hierarchy and write directly to RAM, preventing the "Cache Pollution" that occurs when processing multi-gigabyte buffers.


![SIMD Pipelining](./diagrams/tdd-uvi-002.svg)


### 5.4. Masked Operations (Predication)
For architectures like AVX-512 and ARM SVE, the UVI supports **Masked Loads/Stores**.
*   **Logic**: Instead of a scalar tail loop, a bitmask is used to enable/disable specific lanes in a single vector operation.
*   **Efficiency**: Reduces the "Tail Penalty" to near-zero by allowing the final partial block to be processed by the same SIMD hardware.

{{DIAGRAM:uvi-masked-operation-flow}}
{{DIAGRAM:tdd-uvi-003}}


<!-- TDD_MOD_ID: mod-kernel-compute -->
<!-- TDD_MOD_ID: mod-compute-kernel-engine -->
# Module: Compute Kernel Engine (CKE)

## 1. Technical Specification
The Compute Kernel Engine (CKE) is the orchestration layer of the `simd-library`. While the **UVI** provides raw vector primitives, the CKE transforms these primitives into high-level, composite mathematical operations (e.g., GEMM, 1D/2D Convolutions, Signal Transducers). 

The CKE's primary responsibilities are:
1.  **Pipeline Orchestration**: Managing the execution of multiple SIMD kernels in a single pass to maximize L1/L2 cache residency.
2.  **Loop Transformation**: Automatically applying loop unrolling, software prefetching, and tail-handling strategies based on the ISA reported by the **HDDE**.
3.  **Numerical Stability**: Ensuring that vectorized implementations maintain parity with IEEE-754 standards, specifically regarding Fused Multiply-Add (FMA) accumulations.

## 2. Abstraction Layers
The CKE operates through three distinct functional tiers:
*   **Layer 0: Kernel Primitives**: Low-level "Micro-Kernels" that perform a single operation on a fixed number of registers (e.g., a $4 \times 4$ matrix-vector block).
*   **Layer 1: The Iteration Engine**: Logic responsible for traversing memory buffers, handling alignment boundaries via the **AMBM**, and managing the "Horizontal-to-Vertical" data transformations.
*   **Layer 2: Kernel DSL/Interface**: The high-level API where users define complex pipelines (e.g., `Map().Filter().Reduce()`) which the CKE compiles into an optimized instruction stream.

## 3. Struct & Interface Definitions

### 3.1. Kernel Execution Context
To minimize stack frame overhead, the execution state is encapsulated in a cache-aligned context.

```cpp
// Aligned to 64 bytes to match AVX-512 cache-line and prevent false sharing
struct alignas(64) KernelContext {
    const float* __restrict src_a;
    const float* __restrict src_b;
    float* __restrict out;
    size_t length;
    
    // Stride information for multidimensional kernels
    struct {
        size_t row_stride;
        size_t col_stride;
    } metrics;

    // Prefetching configuration
    uint32_t prefetch_distance; // Measured in cache lines
    
    // [Padding to 64 bytes to ensure subsequent contexts in an array are line-aligned]
};
```

### 3.2. Micro-Kernel Interface
Kernels are defined as stateless functors to allow the compiler to inline them directly into the CKE's main loop.

```cpp
template <typename T, size_t VectorWidth>
struct DotProductKernel {
    using V = Vector<T, VectorWidth>;

    // Accumulator state held in registers
    V acc0, acc1, acc2, acc3;

    inline void Initialize() {
        acc0 = V::Zero(); acc1 = V::Zero();
        acc2 = V::Zero(); acc3 = V::Zero();
    }

    // Process 4 vectors per call (Unroll factor 4)
    inline void Step(const T* a, const T* b) {
        acc0 = V::FusedMultiplyAdd(V::LoadAligned(a + 0*VectorWidth), V::LoadAligned(b + 0*VectorWidth), acc0);
        acc1 = V::FusedMultiplyAdd(V::LoadAligned(a + 1*VectorWidth), V::LoadAligned(b + 1*VectorWidth), acc1);
        acc2 = V::FusedMultiplyAdd(V::LoadAligned(a + 2*VectorWidth), V::LoadAligned(b + 2*VectorWidth), acc2);
        acc3 = V::FusedMultiplyAdd(V::LoadAligned(a + 3*VectorWidth), V::LoadAligned(b + 3*VectorWidth), acc3);
    }

    inline T Finalize() {
        V total = (acc0 + acc1) + (acc2 + acc3);
        return total.HorizontalSum();
    }
};
```

{{DIAGRAM:cke-kernel-hierarchy}}
{{DIAGRAM:tdd-cke-001}}

## 4. Algorithm Pseudo-code

### 4.1. The Hot Path: Super-Scalar Iteration Engine
This logic handles the main loop, utilizing 4x unrolling and software prefetching to saturate the CPU's execution ports.

```cpp
template <typename Kernel, typename T>
T execute_kernel_pipeline(KernelContext& ctx) {
    Kernel k;
    k.Initialize();

    const size_t W = HardwareCapabilities::GetOptimalWidth<T>();
    const size_t UNROLL = 4;
    const size_t BLOCK_SIZE = W * UNROLL;

    size_t i = 0;
    
    // 1. Main Unrolled Loop
    for (; i <= ctx.length - BLOCK_SIZE; i += BLOCK_SIZE) {
        // Software Prefetching: Bring data into L1/L2 before needed
        __builtin_prefetch(ctx.src_a + i + ctx.prefetch_distance, 0, 3);
        __builtin_prefetch(ctx.src_b + i + ctx.prefetch_distance, 0, 3);

        // Core Micro-kernel step
        k.Step(ctx.src_a + i, ctx.src_b + i);
    }

    // 2. Remainder Handling (The "Tail")
    // If the ISA supports masking (AVX-512/SVE), use masked Load/Step
    if constexpr (SupportsMasking<T, W>()) {
        size_t remaining = ctx.length - i;
        if (remaining > 0) {
            auto mask = CreateMask<T, W>(remaining);
            k.StepMasked(ctx.src_a + i, ctx.src_b + i, mask);
        }
    } else {
        // Fallback to scalar for architectures without masking
        for (; i < ctx.length; ++i) {
            k.StepScalar(ctx.src_a[i], ctx.src_b[i]);
        }
    }

    return k.Finalize();
}
```

## 5. Engineering Constraints & Hazards

### 5.1. Concurrency: Data Dependencies
*   **Hazard (Read-After-Write)**: In kernels that perform reductions (like Dot Product), the `acc = fma(a, b, acc)` operation creates a dependency chain.
*   **Mitigation**: The CKE uses multiple accumulators (`acc0` through `acc3`). This allows the CPU's Out-of-Order (OoO) engine to schedule independent FMA instructions while waiting for the latency of the previous accumulation (typically 4-6 cycles) to resolve.

### 5.2. Memory: The "Three-Array" Problem
*   **Constraint**: Operations like $C = A + B$ are limited by memory bandwidth, not compute throughput.
*   **Strategy**: When the CKE detects that data exceeds the L3 cache size, it switches the **UVI** to use Non-Temporal Stores (`StoreNT`) to prevent evicting useful data from the cache hierarchy.

### 5.3. Micro-Optimization Corner
1.  **Loop Peeling**: If the `src` pointers provided by the user are not aligned to 64 bytes, the CKE performs "Loop Peeling"—running a few scalar iterations until the pointer hits an alignment boundary—before engaging the SIMD Hot Path.
2.  **Instruction Balancing**: The engine attempts to balance the ratio of `Load` instructions to `Arithmetic` instructions. For AVX-256, the goal is to keep 2 Loads and 1 Compute instruction in flight per cycle to match the port distribution of Zen/Skylake architectures.
3.  **Kernel Fusion**: The CKE can fuse multiple operations (e.g., `Add` followed by `ReLU`) into a single pass. This reduces the number of times data is loaded from RAM, transforming a "Memory-Bound" problem into a "Compute-Bound" one.

{{DIAGRAM:cke-pipeline-sequence}}
{{DIAGRAM:tdd-cke-002}}

### 5.4. Hazards: Denormal Floats
*   **Hazard**: Floating point numbers very close to zero (denormals) can cause SIMD units to drop in performance by 100x as they fallback to microcode.
*   **Mitigation**: The CKE provides a `FlushToZero` (FTZ) and `DenormalsAreZero` (DAZ) toggle in the `KernelContext` to ensure deterministic timing in real-time signal processing applications.


# 📚 Beyond the Atlas: Further Reading

| Concept | The Paper / Specification | The Implementation | Short Summary |
| :--- | :--- | :--- | :--- |
| **Performance-Portable SIMD** | [Highway: Fast, portable SIMD](https://github.com/google/highway/blob/master/g3doc/design.md) | [google/highway](https://github.com/google/highway) | The Gold Standard for creating a single SIMD source that scales from SSE4 to AVX-512 and ARM SVE without performance loss. |
| **Runtime Dispatch (IFUNC)** | [GNU IFUNC Specification](https://sourceware.org/glibc/wiki/GNU_IFUNC) | [glibc (sysdeps/x86)](https://sourceware.org/git/?p=glibc.git;a=tree;f=sysdeps/x86) | The definitive mechanism for resolving the optimal ISA-specific function at load-time with zero runtime branching overhead. |
| **Micro-Kernel Orchestration** | [Anatomy of High-Performance Matrix Multiplication](https://www.cs.utexas.edu/~flame/pubs/GotoBLAS-revisited.pdf) | [OpenBLAS](https://github.com/xianyi/OpenBLAS) | The foundational research on using small, register-blocked "micro-kernels" to achieve near-theoretical peak performance on modern CPUs. |
| **SIMD Memory Alignment** | [Intel® 64 and IA-32 Architectures Optimization Reference Manual](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html) | [jemalloc (Aligned Alloc)](https://github.com/jemalloc/jemalloc) | The industry-standard guide for understanding cache-line splits, 64-byte alignment, and the hardware penalties of unaligned SIMD loads. |
| **Scalable Vector Abstraction** | [Arm Scalable Vector Extension (SVE) Reference](https://developer.arm.com/documentation/ddi0584/latest/) | [LLVM SVE Backend](https://github.com/llvm/llvm-project/tree/main/llvm/lib/Target/AArch64) | The primary specification for "Vector Length Agnostic" programming, which moves beyond fixed 128/256-bit widths to hardware-determined widths. |
| **Numerical Stability (IEEE-754)** | [IEEE 754-2019 Standard](https://ieeexplore.ieee.org/document/8766229) | [Berkeley SoftFloat](http://www.jhauser.us/arithmetic/SoftFloat.html) | The critical specification for floating-point behavior, essential for handling the "Denormal Floats" and FMA precision issues mentioned in the CKE. |
| **Cross-Platform SIMD Mapping** | [SIMD Everywhere (SIMDe) Design](https://github.com/simd-everywhere/simde) | [simd-everywhere/simde](https://github.com/simd-everywhere/simde) | The most comprehensive implementation of "header-only" translation layers, mapping one ISA's intrinsics (e.g., SSE) to another (e.g., NEON) with zero cost. |
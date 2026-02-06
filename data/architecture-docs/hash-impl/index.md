# SHA-256 Hash Function: Design Document


## Overview

This system implements the SHA-256 cryptographic hash function from the NIST specification, transforming arbitrary-length messages into fixed 256-bit hash values. The key architectural challenge is correctly implementing the precise bitwise operations, padding algorithms, and compression rounds required by the cryptographic standard.


> This guide is meant to help you understand the big picture before diving into each milestone. Refer back to it whenever you need context on how components connect.


## Context and Problem Statement

> **Milestone(s):** Foundational understanding for all milestones (1-4)

### Mental Model: Digital Fingerprints

Think of SHA-256 as creating **digital fingerprints** for any piece of data. Just as your physical fingerprint uniquely identifies you among billions of people, a SHA-256 hash creates a unique 256-bit "fingerprint" for any digital message, whether it's a single character, an entire book, or a massive database file.

This fingerprint analogy reveals several crucial properties. First, **uniqueness**: just as no two people share the same fingerprint (with astronomical probability), no two different messages should produce the same SHA-256 hash. Second, **consistency**: your fingerprint remains the same whether it's taken on Monday or Friday—similarly, SHA-256 will always produce the same hash for the same input. Third, **irreversibility**: you cannot reconstruct a person from their fingerprint alone, and you cannot reverse-engineer the original message from its SHA-256 hash.

However, digital fingerprints have one property that physical fingerprints lack: **sensitivity to the smallest changes**. While your physical fingerprint remains recognizable even with minor cuts or dirt, changing even a single bit in a digital message produces a completely different SHA-256 hash. This property, called the **avalanche effect**, makes SHA-256 invaluable for detecting any modification to data, no matter how minor.

The mathematical precision required to achieve these properties means that implementing SHA-256 is fundamentally different from writing typical application code. Every bit operation must be exact, every padding calculation must be precise, and every arithmetic operation must handle overflow correctly. Unlike application code where "close enough" might work, cryptographic implementations demand mathematical perfection.

### Cryptographic Requirements

SHA-256 belongs to the family of **cryptographic hash functions**, which must satisfy three fundamental mathematical properties that distinguish them from simple checksums or general-purpose hash functions used in data structures.

**Preimage Resistance** forms the foundation of cryptographic security. Given a hash output, it must be computationally infeasible to find any input that produces that hash. This property ensures that SHA-256 acts as a one-way function—easy to compute forward (message to hash) but practically impossible to reverse (hash to message). The term "computationally infeasible" means that even with the most powerful computers available today, finding a preimage would require more time than the age of the universe.

| Property | Definition | Practical Implication | Attack Resistance |
|----------|------------|----------------------|-------------------|
| Preimage Resistance | Given hash h, finding message m where SHA256(m) = h is infeasible | Cannot reverse-engineer original data from hash | Protects against rainbow table attacks |
| Second Preimage Resistance | Given message m1, finding different m2 where SHA256(m1) = SHA256(m2) is infeasible | Cannot forge alternative data with same hash | Prevents targeted collision attacks |
| Collision Resistance | Finding any two different messages m1, m2 where SHA256(m1) = SHA256(m2) is infeasible | No two different inputs should produce same hash | Strongest property; implies the other two |

**Second Preimage Resistance** protects against targeted forgery attacks. If an attacker knows a specific message and its hash, they should not be able to craft a different message that produces the same hash value. This property is crucial for digital signatures and document integrity verification, where an attacker might try to substitute malicious content while maintaining the same hash.

**Collision Resistance** represents the strongest requirement and is the most challenging to achieve mathematically. It demands that finding any pair of different messages that produce the same hash should be computationally infeasible. This property is stronger than the previous two because if you can find collisions easily, you can break both preimage and second preimage resistance.

The **avalanche effect** serves as a practical indicator of these cryptographic properties. When implemented correctly, changing a single input bit should, on average, flip half of the output bits. This sensitivity ensures that related inputs (which might be more vulnerable to mathematical attacks) produce completely unrelated outputs.

> **Design Insight**: The 256-bit output length was specifically chosen to provide 128 bits of collision resistance due to the birthday paradox. While 2^256 operations would be needed for preimage attacks, only 2^128 operations would be expected to find collisions, making 256 bits the minimum acceptable length for long-term cryptographic security.

**Determinism** requires that SHA-256 produce identical outputs for identical inputs across all implementations, platforms, and time periods. This property seems obvious but demands careful attention to implementation details like endianness, padding algorithms, and arithmetic overflow handling. Even minor deviations from the NIST specification can break interoperability or create security vulnerabilities.

The **compression property** enables SHA-256 to accept inputs of any length while always producing exactly 256 bits of output. This fixed-length output is achieved through the Merkle-Damgård construction, which processes the input in fixed-size blocks while maintaining cryptographic security even when multiple blocks are involved.

### Comparison with Other Hash Functions

Understanding SHA-256's position in the cryptographic landscape helps clarify its design decisions and implementation requirements. Each hash function generation emerged to address specific weaknesses discovered in its predecessors, leading to increasingly sophisticated internal structures.

**MD5 (Message Digest 5)** represents the earlier generation of cryptographic hash functions. Designed in 1991, MD5 produces 128-bit hashes through a four-round compression function operating on 512-bit message blocks. While MD5's basic structure influenced later designs, cryptanalysts discovered practical collision attacks by 2004, making MD5 unsuitable for cryptographic applications.

| Hash Function | Output Size | Block Size | Rounds | Status | Primary Weakness |
|---------------|-------------|------------|--------|--------|------------------|
| MD5 | 128 bits | 512 bits | 64 (4×16) | Broken | Collision attacks found (2004) |
| SHA-1 | 160 bits | 512 bits | 80 | Deprecated | Collision attacks demonstrated (2017) |
| SHA-256 | 256 bits | 512 bits | 64 | Secure | No practical attacks known |
| SHA-3 (Keccak) | Variable | 1600 bits | 24 | Secure | Different construction (sponge) |

**SHA-1 (Secure Hash Algorithm 1)** extended MD5's approach with a 160-bit output and 80 compression rounds, providing increased security margins. SHA-1 dominated cryptographic applications throughout the 1990s and 2000s. However, theoretical collision attacks emerged in 2005, followed by the first practical collision demonstration in 2017. Major technology companies and standards bodies have since deprecated SHA-1 in favor of SHA-2 family functions.

**SHA-256** belongs to the SHA-2 family, designed by the NSA and published by NIST in 2001. SHA-256 incorporates lessons learned from attacks on earlier hash functions through several key improvements: increased output length (256 bits), more complex round functions with additional bitwise operations, and enhanced message schedule generation. The SHA-2 family includes variants with different output lengths (SHA-224, SHA-256, SHA-384, SHA-512) but shares the same fundamental structure.

> **Decision: Why SHA-256 Over Other SHA-2 Variants**
> - **Context**: The SHA-2 family offers multiple output lengths, each with different performance and security characteristics
> - **Options Considered**: SHA-224 (fewer bits), SHA-256 (balanced), SHA-512 (more bits but different word size)
> - **Decision**: Implement SHA-256 as the primary learning target
> - **Rationale**: SHA-256 provides the optimal balance of security (128-bit collision resistance), widespread adoption in real systems, and implementation complexity suitable for learning. SHA-224 offers insufficient security margins for modern applications, while SHA-512 operates on 64-bit words requiring different arithmetic handling
> - **Consequences**: Students learn the most practically relevant variant while maintaining manageable implementation complexity

**SHA-3 (Keccak)** represents a fundamental departure from the Merkle-Damgård construction used by MD5, SHA-1, and SHA-2. Designed through NIST's open competition (2007-2012), SHA-3 uses the **sponge construction** with a much larger internal state (1600 bits) and fewer rounds (24). This different approach provides diversity in case weaknesses are discovered in the SHA-2 design paradigm.

The **internal structure differences** between these hash functions reveal the evolution of cryptographic design principles:

**MD5 and SHA-1** rely heavily on addition, left rotation, and simple logical functions (AND, OR, XOR). Their message schedules use straightforward permutations of input words with minimal expansion. The compression functions process working variables through predictable patterns that later proved vulnerable to differential cryptanalysis.

**SHA-256** introduces several sophisticated improvements. The message schedule expands 16 input words to 64 schedule words using complex sigma functions that combine rotation, shifting, and XOR operations. The compression function employs more diverse logical functions (Choice and Majority) alongside enhanced Sigma functions. These design changes increase the **diffusion** of input changes throughout the hash computation, making cryptanalytic attacks significantly more difficult.

**SHA-3** abandons block-cipher-inspired designs entirely, instead using the Keccak permutation applied to a large state array. This sponge construction provides theoretical advantages in security proofs and enables additional features like extendable output functions (SHAKE variants).

**Performance characteristics** vary significantly across hash functions, influencing their practical adoption:

| Function | Software Speed | Hardware Cost | Cryptanalysis Resistance | Standardization |
|----------|----------------|---------------|--------------------------|-----------------|
| MD5 | Fastest | Lowest | Broken | Legacy only |
| SHA-1 | Fast | Low | Weak | Deprecated |
| SHA-256 | Moderate | Moderate | Strong | Widely adopted |
| SHA-3 | Slower | Higher | Strong | Limited adoption |

**Implementation complexity** also differs markedly between hash functions. MD5 and SHA-1 can be implemented with relatively simple operations, making them attractive for educational purposes but insufficient for security. SHA-256 requires more careful handling of the expanded message schedule and additional round functions, striking a balance between educational value and real-world relevance. SHA-3's sponge construction involves substantially different algorithms that provide less transferable knowledge to other cryptographic primitives.

> **Cryptographic Evolution Insight**: Each generation of hash functions addresses specific attack vectors discovered in previous designs. MD5's collision vulnerabilities led to SHA-1's additional rounds and larger output. SHA-1's differential attack susceptibility drove SHA-2's enhanced diffusion and round functions. SHA-2's theoretical concerns about extension attacks motivated SHA-3's fundamentally different sponge construction. This evolution demonstrates how cryptographic standards continuously adapt to advancing attack techniques.

**Real-world adoption patterns** reflect both security requirements and implementation inertia. SHA-256 dominates modern applications including Bitcoin blockchain, TLS certificates, code signing, and digital forensics. SHA-1 remains in legacy systems despite deprecation warnings. SHA-3 adoption has been slower due to the continued security of SHA-2 and the costs of migrating existing infrastructure.

For educational purposes, SHA-256 offers the optimal combination of practical relevance, manageable complexity, and transferable concepts. Students implementing SHA-256 learn bitwise operations, endianness handling, modular arithmetic, and algorithm specification compliance—skills directly applicable to other cryptographic primitives and security implementations.

### Implementation Guidance

Understanding the context and requirements establishes the foundation for implementing SHA-256 correctly. This guidance bridges the gap between cryptographic theory and practical coding, providing concrete tools and structures for building a compliant implementation.

#### A. Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Primary Language | Python with built-in integers | Python with `ctypes` for 32-bit arithmetic |
| Testing Framework | Built-in `unittest` module | `pytest` with parameterized test cases |
| Binary Operations | Native bit operators (`&`, `\|`, `^`, `<<`, `>>`) | Custom bit manipulation utilities |
| Test Data Format | Hardcoded NIST vectors | JSON test files with automated loading |
| Output Formatting | String formatting with `format()` | `struct` module for binary packing |

For beginners, Python's built-in arbitrary precision integers simplify development by eliminating overflow concerns during development, though proper 32-bit masking remains essential for SHA-256 compliance. The native bit operators provide direct translation from the NIST specification's mathematical notation.

#### B. Recommended Project Structure

Organizing the SHA-256 implementation into logical modules helps maintain clarity and supports incremental development aligned with the milestone structure:

```
sha256_project/
├── src/
│   ├── __init__.py
│   ├── sha256.py           ← Main hash function interface
│   ├── preprocessing.py    ← Message padding and block parsing (Milestone 1)
│   ├── message_schedule.py ← Word expansion and sigma functions (Milestone 2)
│   ├── compression.py      ← Main compression function (Milestone 3)
│   └── utils.py           ← Bit operations and constants
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_schedule.py
│   ├── test_compression.py
│   ├── test_integration.py
│   └── nist_vectors.py    ← Official test vectors
└── examples/
    ├── basic_usage.py
    └── performance_test.py
```

This structure separates concerns by milestone while maintaining clear interfaces between components. Each module handles one major aspect of the SHA-256 algorithm, enabling focused development and testing.

#### C. Core Constants and Data Types

Every SHA-256 implementation requires the standardized constants and clear data type definitions. These form the foundation for all subsequent operations:

```python
# utils.py - Complete constants and utilities
"""SHA-256 constants and utility functions."""

# Initial hash values (first 32 bits of fractional parts of square roots of first 8 primes)
INITIAL_HASH_VALUES = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ad, 0x5be0cd19
]

# Round constants (first 32 bits of fractional parts of cube roots of first 64 primes)
ROUND_CONSTANTS = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]

def right_rotate_32(value, amount):
    """Rotate a 32-bit value right by specified amount."""
    # TODO: Implement 32-bit right rotation using bitwise operations
    # TODO: Mask input value to 32 bits
    # TODO: Handle rotation amount modulo 32
    # TODO: Combine high and low parts after rotation
    pass

def mask_32_bits(value):
    """Ensure value fits in 32 bits."""
    # TODO: Apply bitwise AND with 0xFFFFFFFF to mask to 32 bits
    pass
```

#### D. Core Component Interfaces

Each major component should expose a clean interface that matches the milestone structure and enables independent testing:

```python
# sha256.py - Main interface
"""SHA-256 hash function implementation."""

class SHA256:
    """SHA-256 hash function with step-by-step processing."""
    
    def __init__(self):
        """Initialize hash state with NIST-specified initial values."""
        # TODO: Initialize hash state with INITIAL_HASH_VALUES
        # TODO: Set up state for processing multiple messages
        pass
    
    def hash_message(self, message):
        """Compute SHA-256 hash of message string.
        
        Args:
            message: Input string to hash
            
        Returns:
            64-character lowercase hexadecimal string
        """
        # TODO: Convert message string to bytes
        # TODO: Apply preprocessing (padding and block parsing)
        # TODO: Process each 512-bit block through compression function
        # TODO: Format final hash state as hexadecimal string
        pass
    
    def hash_bytes(self, data):
        """Compute SHA-256 hash of byte array."""
        # TODO: Apply preprocessing to byte data
        # TODO: Process blocks and return formatted hash
        pass
```

#### E. Debugging and Validation Utilities

Implementing SHA-256 requires extensive intermediate value verification. These utilities enable step-by-step validation against known reference implementations:

```python
# tests/nist_vectors.py - Test vectors for validation
"""NIST SHA-256 test vectors for compliance verification."""

NIST_TEST_VECTORS = [
    {
        'message': '',
        'expected_hash': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
        'description': 'Empty string'
    },
    {
        'message': 'abc',
        'expected_hash': 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
        'description': 'Three character message'
    },
    {
        'message': 'abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq',
        'expected_hash': '248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1',
        'description': 'Multi-block message'
    }
]

def run_compliance_tests(sha256_implementation):
    """Run NIST test vectors against implementation."""
    # TODO: Iterate through test vectors
    # TODO: Compute hash for each test message
    # TODO: Compare against expected results
    # TODO: Report detailed results for debugging
    pass
```

#### F. Milestone Checkpoints

Each milestone should produce verifiable intermediate results that confirm correct implementation before proceeding to the next stage:

**Milestone 1 Checkpoint (Preprocessing):**
- Input: Empty string `""`
- Expected: Single 512-bit block with correct padding pattern
- Validation: Block should end with 64-bit big-endian length (0x0000000000000000)
- Debug command: Print binary representation of padded block

**Milestone 2 Checkpoint (Message Schedule):**
- Input: First block from empty string preprocessing
- Expected: 64-word schedule with specific values for words 16-19
- Validation: Schedule[16] should equal 0x80000000 after sigma function application
- Debug command: Print first 20 schedule words in hexadecimal

**Milestone 3 Checkpoint (Compression):**
- Input: Schedule from Milestone 2 + initial hash values
- Expected: Specific intermediate hash values after 64 rounds
- Validation: Hash state should match reference implementation after first block
- Debug command: Print working variables after rounds 0, 16, 32, 48, 63

**Milestone 4 Checkpoint (Final Output):**
- Input: Complete hash state from Milestone 3
- Expected: Final hash matches NIST test vector for empty string
- Validation: Output equals `"e3b0c442...52b855"`
- Debug command: Compare byte-by-byte with expected output

#### G. Common Implementation Pitfalls

⚠️ **Pitfall: Incorrect Bit vs Byte Handling**
Students frequently confuse bit and byte operations, especially during preprocessing. The NIST specification describes operations in terms of bits, but most programming languages work with bytes. Always track whether lengths and offsets refer to bits or bytes, and convert explicitly when crossing boundaries.

⚠️ **Pitfall: Endianness Confusion**
SHA-256 requires big-endian byte ordering throughout, but most modern processors use little-endian representation. This mismatch causes incorrect results when converting between integers and byte arrays. Use explicit big-endian conversion functions rather than assuming native byte order.

⚠️ **Pitfall: 32-bit Overflow Assumptions**
Python's arbitrary precision integers can mask overflow bugs that would appear in other languages. Always apply 32-bit masking (`value & 0xFFFFFFFF`) after arithmetic operations to simulate the 32-bit arithmetic specified by NIST.

⚠️ **Pitfall: Right Rotate vs Right Shift**
The sigma functions require right rotation (circular shift) rather than right shift (with zero fill). Right rotation preserves all bits by moving them from the high end to the low end, while right shift discards high-order bits. This distinction is crucial for cryptographic security.


## Goals and Non-Goals

> **Milestone(s):** Foundational understanding for all milestones (1-4)

### Mental Model: Academic vs Production Implementation

Think of this SHA-256 implementation as building a **reference model** rather than a production engine. Imagine the difference between a detailed anatomical model used in medical school versus the actual surgical instruments used in an operating room. The anatomical model prioritizes educational clarity, accurate representation of all components, and step-by-step understanding of how each part functions. It's not optimized for speed or efficiency—it's optimized for learning and verification.

Similarly, our SHA-256 implementation serves as a cryptographic learning laboratory where every step of the NIST specification becomes visible and understandable. Just as medical students need to understand anatomy before performing surgery, developers need to understand cryptographic primitives before building production security systems. This implementation prioritizes correctness, clarity, and compliance over performance optimization.

### Functional Goals

The primary functional goals of this SHA-256 implementation establish clear success criteria that align with educational objectives and cryptographic standards. Each goal addresses a specific aspect of building a correct, understandable, and verifiable hash function implementation.

#### NIST Specification Compliance

**Perfect adherence to the NIST FIPS 180-4 specification** represents the cornerstone functional goal. This means implementing every algorithmic step exactly as specified in the official cryptographic standard, without shortcuts or optimizations that might alter the mathematical behavior. The implementation must produce identical outputs to the reference specification for all possible inputs.

The compliance requirement extends beyond just getting the right final answer—it encompasses implementing the correct intermediate steps, using the specified constants, following the exact bit manipulation sequences, and maintaining the prescribed data structures throughout the computation. This includes using the exact `INITIAL_HASH_VALUES` from the NIST specification, applying all 64 `ROUND_CONSTANTS` in the correct order, and implementing the precise sigma function definitions without mathematical shortcuts.

Verification of NIST compliance occurs through systematic testing against official test vectors, starting with the well-known empty string hash and the "abc" test case, then expanding to longer messages and edge cases. The implementation must handle all message lengths correctly, from empty inputs to multi-gigabyte files, always producing results that match reference implementations.

> **Decision: Strict NIST Compliance Over Performance**
> - **Context**: Could implement optimized variants that produce correct results faster
> - **Options Considered**: 
>   1. Strict NIST compliance with readable intermediate steps
>   2. Optimized implementation with lookup tables and vectorization
>   3. Hybrid approach with optional optimization flags
> - **Decision**: Strict NIST compliance with no performance optimizations
> - **Rationale**: Educational value requires understanding every specification detail; optimizations obscure the underlying mathematics and make debugging harder; compliance verification becomes more complex with multiple code paths
> - **Consequences**: Slower execution but complete algorithmic transparency; easier debugging through step-by-step verification; simplified testing against reference vectors

| Compliance Aspect | Requirement | Verification Method |
|-------------------|-------------|-------------------|
| Initial Hash Values | Use exact NIST constants | Compare against specification table |
| Round Constants | All 64 K values match specification | Unit test each constant value |
| Message Padding | Follow 448 mod 512 rule precisely | Test various message lengths |
| Endianness | Big-endian throughout | Compare intermediate values |
| Bit Operations | Exact rotate/shift/XOR sequences | Step-by-step test vector validation |
| Output Format | 256-bit hash as 64-character hex | String format and length verification |

#### Correctness Verification Through Testing

**Comprehensive correctness verification** ensures that the implementation produces mathematically correct results across all input scenarios. This goes beyond basic functionality to include systematic validation of intermediate computations, edge case handling, and boundary condition testing.

The verification strategy employs multiple layers of testing, from unit tests that validate individual functions like `right_rotate_32` and the sigma functions, to integration tests that verify complete message processing pipelines. Each milestone introduces additional verification checkpoints that confirm the implementation remains correct as complexity increases.

Test vector validation forms the foundation of correctness verification. The implementation must pass all official NIST test vectors, including the standard examples for empty string, single-character messages, and the 448-bit boundary cases that stress the padding algorithm. Additionally, the implementation should correctly handle messages that span multiple 512-bit blocks, verifying that the hash state properly carries forward between block processing iterations.

Intermediate value logging provides crucial debugging support for correctness verification. The implementation should offer the ability to inspect hash state values, message schedule arrays, and working variables at each compression round. This visibility enables developers to identify exactly where their implementation diverges from reference calculations, accelerating the debugging process significantly.

| Testing Layer | Focus Area | Success Criteria |
|---------------|------------|------------------|
| Unit Tests | Individual functions | All bitwise operations produce expected outputs |
| Integration Tests | Complete pipeline | Known inputs produce correct final hashes |
| Edge Case Tests | Boundary conditions | Empty strings, maximum lengths, padding boundaries |
| NIST Vector Tests | Official compliance | All reference test vectors pass |
| Intermediate Validation | Step-by-step verification | Hash state matches reference at each round |
| Regression Tests | Consistency over time | Results remain stable across code changes |

#### Educational Clarity and Understanding

**Maximizing educational value** drives architectural decisions toward code readability, comprehensive documentation, and step-by-step algorithmic transparency. The implementation should serve as a learning tool that helps developers understand not just how to use SHA-256, but how SHA-256 actually works at the bit manipulation level.

Code structure prioritizes readability over efficiency, with descriptive variable names, extensive comments, and clear separation between different algorithmic phases. Each major function includes documentation that explains its role in the overall SHA-256 algorithm, its inputs and outputs, and any non-obvious implementation details. The code should read like a textbook implementation that closely mirrors the NIST specification text.

Function organization follows the natural flow of the SHA-256 algorithm, making it easy for learners to trace message processing from initial input through final hash output. Related functions are grouped logically, with clear interfaces between preprocessing, message scheduling, compression, and output formatting components. This organization helps learners understand the algorithm's structure and the purpose of each processing stage.

Educational comments within the code explain not just what each operation does, but why it's necessary for cryptographic security. For example, comments should explain that the message padding prevents length extension attacks, that the message schedule expansion increases diffusion, and that the compression rounds provide the avalanche effect crucial for hash function security.

> The educational goal requires balancing mathematical precision with accessibility. Junior developers should be able to follow the implementation logic without getting lost in cryptographic theory, while still gaining appreciation for the sophisticated mathematics underlying modern cryptographic systems.

| Educational Aspect | Implementation Approach | Learning Outcome |
|-------------------|------------------------|-------------------|
| Algorithm Structure | Clear separation of four main phases | Understanding of SHA-256 pipeline |
| Bit Operations | Explicit rotate/shift/XOR implementations | Mastery of cryptographic primitives |
| Constant Usage | Named constants with explanatory comments | Knowledge of specification details |
| State Evolution | Visible hash state at each processing step | Understanding of iterative construction |
| Error Patterns | Common pitfall documentation | Awareness of implementation challenges |
| Testing Strategy | Progressive milestone validation | Systematic verification skills |

### Non-Goals

Explicitly defining what this implementation will not attempt prevents scope creep and maintains focus on the core educational and correctness objectives. These non-goals represent important considerations for production systems that are deliberately excluded to keep the implementation approachable for learning purposes.

#### Performance Optimization

**High-performance execution** is explicitly excluded from this implementation's objectives. While production SHA-256 implementations employ numerous optimization techniques—including lookup tables for rotation operations, SIMD vectorization for parallel computation, and specialized algorithms for different message sizes—this implementation prioritizes algorithmic clarity over execution speed.

Performance optimizations typically involve mathematical transformations that produce identical results through different computational paths. For example, precomputed lookup tables can accelerate bit rotation operations, but they obscure the underlying mathematical operations that learners need to understand. Similarly, loop unrolling and vectorization can dramatically improve throughput, but they make the code much harder to follow and debug.

The decision to avoid performance optimization simplifies the testing and verification process significantly. With a single, straightforward computational path, debugging becomes much easier because there's only one way the algorithm can execute. Performance optimizations often introduce conditional execution paths that require additional test coverage and complicate the verification of correctness.

Memory usage optimization also falls outside the scope of this implementation. While production systems might implement streaming algorithms that process large files without loading them entirely into memory, this implementation can assume that input messages fit comfortably in available RAM. This assumption eliminates complex buffering logic and allows the implementation to focus on the core cryptographic computation.

| Optimization Category | Production Approach | Educational Approach | Rationale for Exclusion |
|----------------------|-------------------|---------------------|------------------------|
| Bit Operations | Lookup tables, SIMD instructions | Direct mathematical operations | Obscures underlying mathematics |
| Memory Usage | Streaming, in-place updates | Full message buffering | Adds complexity without educational value |
| Loop Structure | Unrolling, vectorization | Clear iterative logic | Makes debugging much harder |
| Data Structures | Cache-optimized layouts | Readable field organization | Optimized layouts reduce clarity |
| Algorithm Variants | Specialized paths for different inputs | Single unified implementation | Multiple paths complicate testing |

#### Hardware Acceleration

**Hardware-specific optimizations and acceleration** are explicitly excluded from the implementation scope. Modern processors offer specialized instructions for cryptographic operations, including Intel's SHA-NI instruction set extensions that can dramatically accelerate SHA-256 computation. Graphics processing units (GPUs) can parallelize hash computation across thousands of simultaneous threads for certain use cases.

While hardware acceleration provides substantial performance benefits in production environments, it introduces platform dependencies that complicate the educational mission. Students working on different hardware platforms—Intel versus ARM processors, different operating systems, or systems without specialized cryptographic instructions—would encounter varying behavior and performance characteristics.

Hardware acceleration also requires specialized knowledge of processor architectures, instruction set extensions, and parallel programming models that extend far beyond the scope of learning SHA-256 specifically. The focus should remain on understanding the cryptographic algorithm itself, not on the intricacies of hardware optimization techniques.

Cross-platform compatibility becomes much more challenging when hardware-specific optimizations are involved. The educational implementation should run consistently across different platforms, producing identical results and exhibiting similar behavior regardless of the underlying hardware capabilities. This consistency helps ensure that all learners have equivalent experiences with the implementation.

> **Decision: Software-Only Implementation**
> - **Context**: Hardware acceleration can provide 10-100x performance improvements for SHA-256
> - **Options Considered**:
>   1. Pure software implementation
>   2. Optional hardware acceleration with fallback
>   3. Hardware-optimized primary implementation
> - **Decision**: Pure software implementation only
> - **Rationale**: Educational focus should remain on algorithm understanding; hardware specifics introduce platform dependencies; debugging hardware-accelerated code requires specialized tools and knowledge
> - **Consequences**: Slower execution but universal compatibility; simplified debugging and testing; focus remains on cryptographic concepts rather than hardware optimization

#### Production Security Hardening

**Production-grade security measures** beyond algorithmic correctness are outside the implementation scope. Production cryptographic libraries implement numerous security hardening measures designed to protect against side-channel attacks, fault injection, and other sophisticated attack vectors that target the implementation rather than the underlying mathematical algorithm.

Side-channel attack protection involves implementing constant-time algorithms that prevent timing analysis, power analysis, and electromagnetic analysis attacks. These protections require careful attention to conditional branching, memory access patterns, and instruction timing characteristics. While crucial for production security, these considerations add significant complexity that obscures the core algorithmic learning objectives.

Memory protection measures in production systems include secure memory allocation, explicit memory clearing, and protection against memory disclosure vulnerabilities. These measures prevent sensitive intermediate values from persisting in memory where they might be accessible to attackers. For an educational implementation, these security measures add complexity without contributing to cryptographic understanding.

Input validation and sanitization represent another category of production security measures that are simplified in the educational context. Production implementations must defend against malicious inputs designed to exploit parsing vulnerabilities, trigger buffer overflows, or cause denial-of-service conditions. The educational implementation can assume well-intentioned inputs and focus on the cryptographic computation rather than input security.

Error handling in production systems must avoid information disclosure through timing variations, error message content, or exception details that might assist attackers. Educational implementations can provide detailed error information that helps with debugging and learning, even though such information might represent a security risk in production environments.

| Security Aspect | Production Requirement | Educational Approach | Exclusion Rationale |
|------------------|----------------------|-------------------|-------------------|
| Timing Attacks | Constant-time operations | Natural algorithm timing | Constant-time implementation obscures algorithm flow |
| Memory Protection | Secure allocation, explicit clearing | Standard memory usage | Memory security adds complexity without educational value |
| Input Validation | Comprehensive malicious input defense | Basic type and format checking | Focus should remain on cryptographic computation |
| Error Disclosure | Minimal information leakage | Detailed debugging information | Detailed errors aid learning and debugging |
| Fault Tolerance | Redundant computation, integrity checks | Single computation path | Redundancy complicates verification and testing |

#### Multi-Algorithm Cryptographic Suite

**Comprehensive cryptographic library functionality** extends beyond the scope of this focused SHA-256 implementation. While production cryptographic libraries typically provide numerous hash functions (MD5, SHA-1, SHA-224, SHA-384, SHA-512, SHA-3 variants), digital signature algorithms, and symmetric encryption capabilities, this implementation concentrates exclusively on SHA-256.

The decision to focus on a single algorithm allows for deeper exploration of SHA-256 specifics rather than broader but shallower coverage of multiple cryptographic primitives. Learners benefit more from thoroughly understanding one hash function implementation than from superficial exposure to many different algorithms. The concepts learned from SHA-256 provide a strong foundation for understanding other cryptographic hash functions later.

Generic cryptographic interfaces and algorithm-agnostic frameworks introduce abstraction layers that, while valuable for production code reuse, obscure the specific details that make each cryptographic algorithm unique. The educational implementation benefits from concrete, specific code that directly implements SHA-256 without generic abstractions.

Algorithm comparison and selection logic becomes complex when supporting multiple hash functions with different security properties, performance characteristics, and use case suitability. Focusing on SHA-256 eliminates these complications and allows learners to concentrate on mastering a single, widely-used, and cryptographically sound hash function.

### Implementation Guidance

This SHA-256 implementation serves as a comprehensive educational tool, requiring careful balance between functional completeness and learning accessibility. The following guidance helps establish the technical foundation for achieving the functional goals while respecting the defined non-goals.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Core Implementation | Pure Python with standard library | Python with optional C extensions for comparison |
| Testing Framework | Built-in unittest module | pytest with coverage reporting |
| Documentation | Inline comments and docstrings | Sphinx-generated API documentation |
| Validation Tools | Manual test vector verification | Automated NIST test suite runner |
| Debugging Support | Print statements and logging | Step-by-step debugger integration |
| Performance Measurement | Simple timing with time.time() | Detailed profiling with cProfile |

#### Recommended File Structure

```
sha256-implementation/
├── src/
│   ├── sha256/
│   │   ├── __init__.py              # Main hash_message() interface
│   │   ├── preprocessing.py         # Message padding and block parsing
│   │   ├── message_schedule.py      # Schedule generation and sigma functions
│   │   ├── compression.py           # Main compression function and rounds
│   │   ├── output.py               # Final hash formatting and validation
│   │   ├── constants.py            # INITIAL_HASH_VALUES and ROUND_CONSTANTS
│   │   └── utilities.py            # right_rotate_32, mask_32_bits helpers
│   └── tests/
│       ├── test_preprocessing.py    # Milestone 1 validation
│       ├── test_message_schedule.py # Milestone 2 validation
│       ├── test_compression.py      # Milestone 3 validation
│       ├── test_integration.py      # Milestone 4 validation
│       └── nist_test_vectors.py     # Official compliance testing
├── examples/
│   ├── basic_usage.py              # Simple hash_message() examples
│   └── step_by_step_debug.py       # Intermediate value inspection
├── docs/
│   ├── algorithm_walkthrough.md    # Step-by-step SHA-256 explanation
│   └── debugging_guide.md          # Common issues and solutions
└── README.md                       # Quick start and overview
```

This file organization separates concerns clearly while maintaining educational accessibility. Each module corresponds to a specific milestone, making it easy for learners to focus on one algorithmic phase at a time while understanding how the pieces integrate.

#### Infrastructure Starter Code

**Constants and Basic Types** (constants.py):
```python
"""SHA-256 constants from NIST FIPS 180-4 specification."""

# Initial hash values (first 32 bits of fractional parts of square roots of first 8 primes)
INITIAL_HASH_VALUES = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
]

# Round constants (first 32 bits of fractional parts of cube roots of first 64 primes)
ROUND_CONSTANTS = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]

# Algorithm constants
BLOCK_SIZE_BITS = 512
WORD_SIZE_BITS = 32
HASH_SIZE_BITS = 256
BLOCK_SIZE_BYTES = BLOCK_SIZE_BITS // 8
WORD_SIZE_BYTES = WORD_SIZE_BITS // 8
```

**Utility Functions** (utilities.py):
```python
"""Utility functions for SHA-256 bit manipulation operations."""

def right_rotate_32(value: int, amount: int) -> int:
    """
    Perform 32-bit right rotation on a value.
    
    Args:
        value: 32-bit integer to rotate
        amount: Number of positions to rotate right (0-31)
    
    Returns:
        32-bit integer result of right rotation
    """
    # Ensure value is within 32-bit range
    value = mask_32_bits(value)
    amount = amount % 32  # Handle rotation amounts > 32
    
    # Right rotation: (value >> amount) | (value << (32 - amount))
    return mask_32_bits((value >> amount) | (value << (32 - amount)))

def mask_32_bits(value: int) -> int:
    """
    Mask a value to 32 bits using bitwise AND.
    
    Args:
        value: Integer value to mask
    
    Returns:
        Value masked to 32 bits (0x00000000 to 0xFFFFFFFF)
    """
    return value & 0xFFFFFFFF

def bytes_to_words(byte_data: bytes) -> list[int]:
    """
    Convert byte array to 32-bit words in big-endian order.
    
    Args:
        byte_data: Bytes to convert (length must be multiple of 4)
    
    Returns:
        List of 32-bit words
    """
    if len(byte_data) % 4 != 0:
        raise ValueError("Byte data length must be multiple of 4")
    
    words = []
    for i in range(0, len(byte_data), 4):
        # Combine 4 bytes into 32-bit word (big-endian)
        word = (byte_data[i] << 24) | (byte_data[i+1] << 16) | (byte_data[i+2] << 8) | byte_data[i+3]
        words.append(word)
    
    return words

def words_to_hex_string(words: list[int]) -> str:
    """
    Convert list of 32-bit words to lowercase hexadecimal string.
    
    Args:
        words: List of 32-bit words
    
    Returns:
        Lowercase hexadecimal string representation
    """
    hex_parts = []
    for word in words:
        hex_parts.append(f"{word:08x}")  # 8 hex digits, zero-padded, lowercase
    
    return ''.join(hex_parts)
```

#### Core Logic Skeleton Code

**Main Interface** (__init__.py):
```python
"""SHA-256 Hash Function Implementation - Main Interface"""

from .preprocessing import preprocess_message
from .message_schedule import generate_message_schedule
from .compression import compress_block
from .output import finalize_hash
from .constants import INITIAL_HASH_VALUES

def hash_message(message: str) -> str:
    """
    Compute SHA-256 hash of a string message.
    
    Args:
        message: Input string to hash
    
    Returns:
        64-character lowercase hexadecimal hash string
    """
    # TODO 1: Convert message string to bytes using UTF-8 encoding
    # TODO 2: Call preprocess_message to get list of 512-bit blocks
    # TODO 3: Initialize hash_state with INITIAL_HASH_VALUES
    # TODO 4: For each message block:
    #   TODO 4a: Generate message schedule from block
    #   TODO 4b: Compress block with current hash state
    #   TODO 4c: Update hash_state with compression result
    # TODO 5: Call finalize_hash to convert final state to hex string
    # TODO 6: Return final hash string
    
    pass  # Learner implements this

def hash_bytes(data: bytes) -> str:
    """
    Compute SHA-256 hash of byte data.
    
    Args:
        data: Input bytes to hash
    
    Returns:
        64-character lowercase hexadecimal hash string
    """
    # TODO 1: Call preprocess_message with byte data
    # TODO 2: Follow same algorithm as hash_message
    # Hint: This eliminates the UTF-8 encoding step
    
    pass  # Learner implements this
```

#### Language-Specific Implementation Hints

**Python-Specific Considerations:**
- Use `int.to_bytes(length, byteorder='big')` for converting integers to big-endian byte representation
- Python's unlimited integer precision means all arithmetic operations need explicit 32-bit masking with `mask_32_bits()`
- Use `len(message_string).to_bytes(8, 'big')` to create the 64-bit length field for padding
- List comprehensions work well for bulk operations: `[mask_32_bits(word) for word in word_list]`
- The `struct` module provides alternative packing/unpacking: `struct.pack('>I', word)` for big-endian 32-bit integers

**Common Python Pitfalls:**
- Python's `>>` operator performs arithmetic right shift, not logical right shift, but for positive 32-bit values this doesn't matter
- String formatting with f"{value:08x}" produces lowercase hex with zero-padding to 8 characters
- List slicing `data[i:i+4]` for extracting 4-byte chunks is more readable than manual indexing
- Use `isinstance(message, str)` to distinguish string vs bytes input handling

#### Milestone Checkpoint Validation

**Milestone 1 Checkpoint:**
After implementing message preprocessing, run:
```bash
python -m pytest tests/test_preprocessing.py -v
```

Expected behavior:
- Empty string should pad to exactly 512 bits (64 bytes)
- "abc" should pad to 512 bits with length 3 at the end
- Messages near 448-bit boundary should extend to 1024 bits (two blocks)
- All padding should end with 64-bit big-endian length field

**Milestone 2 Checkpoint:**
Test message schedule generation:
```bash
python -c "
from src.sha256.message_schedule import generate_message_schedule
from src.sha256.preprocessing import preprocess_message
blocks = preprocess_message(b'abc')
schedule = generate_message_schedule(blocks[0])
print(f'Schedule length: {len(schedule)}')
print(f'First word: 0x{schedule[0]:08x}')
print(f'Last word: 0x{schedule[63]:08x}')
"
```

Expected output should show 64 schedule words with first word matching the first 32 bits of the "abc" block.

**Milestone 3 Checkpoint:**
Verify compression function produces different hash state:
```python
from src.sha256.constants import INITIAL_HASH_VALUES
from src.sha256.compression import compress_block
# Should see hash_state values change after compression
```

**Milestone 4 Checkpoint:**
Final integration test:
```python
from src.sha256 import hash_message
assert hash_message("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
assert hash_message("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


## High-Level Architecture

> **Milestone(s):** Foundational understanding for all milestones (1-4)

### Four-Stage Processing Pipeline: Preprocessing, message schedule, compression, and finalization stages

#### Mental Model: Assembly Line Manufacturing

Think of SHA-256 as a **four-stage assembly line** in a precision manufacturing facility. Just as an assembly line transforms raw materials into finished products through specialized stations, SHA-256 transforms arbitrary input messages into fixed-size hash values through four distinct processing stages. Each stage has a specific responsibility and produces intermediate products that feed into the next stage. The raw material (input message) enters at one end, passes through preprocessing (formatting), schedule generation (preparation), compression (transformation), and finalization (quality control) to emerge as a finished product (256-bit hash).

![SHA-256 System Architecture](./diagrams/system-architecture.svg)

The SHA-256 algorithm follows the **Merkle-Damgård construction**, which processes input data in fixed-size blocks through an iterative compression function. This construction provides crucial cryptographic properties including **collision resistance** and **preimage resistance** by ensuring that each bit of input influences the final output through multiple rounds of mixing operations.

The four-stage pipeline processes data sequentially, with each stage having distinct inputs, outputs, and responsibilities:

| Stage | Input | Output | Primary Responsibility |
|-------|-------|--------|----------------------|
| Preprocessing | Raw message bytes | Array of 512-bit blocks | Message padding and block alignment |
| Schedule Generation | 512-bit block | 64-word message schedule | Word expansion using sigma functions |
| Compression | Message schedule + Hash state | Updated hash state | 64 rounds of cryptographic mixing |
| Finalization | Final hash state | 256-bit hash output | Format conversion and validation |

> **Decision: Sequential Pipeline Architecture**
> - **Context**: SHA-256 specification requires specific order of operations with intermediate state dependencies
> - **Options Considered**: 
>   1. Monolithic single-function approach with all operations inline
>   2. Sequential pipeline with clear stage boundaries
>   3. Parallel processing with independent workers
> - **Decision**: Sequential pipeline with clear stage boundaries
> - **Rationale**: The Merkle-Damgård construction requires strict ordering where each block's compression depends on the previous block's hash state. Additionally, within each block, schedule generation must complete before compression can begin. Clear stage boundaries improve testability and debugging.
> - **Consequences**: Enables independent testing of each stage, simplifies debugging through intermediate value inspection, but prevents parallelization opportunities within a single message hash computation.

#### Stage 1: Message Preprocessing

The preprocessing stage transforms variable-length input messages into standardized 512-bit blocks suitable for the compression function. This stage implements the critical **padding algorithm** that ensures input messages conform to the block structure required by the Merkle-Damgård construction.

The preprocessing component accepts raw message bytes and performs three essential operations: binary conversion with padding bit insertion, length encoding with proper endianness handling, and block parsing into fixed-size chunks. The padding algorithm follows a precise mathematical formula to ensure that the padded message length is congruent to 448 modulo 512 bits, leaving exactly 64 bits for the original message length encoding.

| Preprocessing Operation | Input | Output | Purpose |
|------------------------|-------|--------|---------|
| Binary conversion | Raw message string/bytes | Bit array with '1' padding | Convert to binary representation |
| Zero padding | Bit array | 448 mod 512 bit alignment | Ensure consistent block structure |
| Length encoding | Original message length | 64-bit big-endian integer | Preserve original message size |
| Block parsing | Padded bit stream | Array of `MessageBlock` objects | Create processing units |

> The preprocessing stage is where most implementation errors occur due to bit-versus-byte confusion and endianness mistakes. The NIST specification operates at the bit level, but most programming languages work with bytes, requiring careful conversion and alignment calculations.

#### Stage 2: Message Schedule Generation

The schedule generation stage expands each 512-bit block into a 64-word **message schedule** that feeds the compression rounds. This expansion process uses the **sigma functions** (lowercase σ₀ and σ₁) to create additional words from the initial 16 words parsed from each block.

The expansion follows the SHA-256 recurrence relation: `W[t] = σ₁(W[t-2]) + W[t-7] + σ₀(W[t-15]) + W[t-16]` for t = 16 to 63. Each sigma function combines right-rotation and XOR operations to create **avalanche effect** where small changes in input words produce large changes in the expanded schedule.

| Schedule Component | Function | Implementation | Purpose |
|-------------------|----------|----------------|---------|
| Initial words (0-15) | Block parsing | Direct 32-bit word extraction | Foundation for expansion |
| Sigma-0 function | σ₀(x) = ROTR₇(x) ⊕ ROTR₁₈(x) ⊕ SHR₃(x) | Right-rotate and XOR | Diffusion of bit changes |
| Sigma-1 function | σ₁(x) = ROTR₁₇(x) ⊕ ROTR₁₉(x) ⊕ SHR₁₀(x) | Right-rotate and XOR | Additional bit mixing |
| Extended words (16-63) | Recurrence relation | Four-word combination | Schedule completion |

The schedule generation creates **temporal dependencies** where later schedule words depend on multiple earlier words through the sigma functions. This dependency chain ensures that modifications to any input bit propagate through multiple schedule positions, contributing to the cryptographic strength of the hash function.

#### Stage 3: Main Compression Function

The compression function implements the cryptographic core of SHA-256 through 64 rounds of intensive bit manipulation operations. Each round updates eight **working variables** (labeled a through h) using the current schedule word, round-specific constants, and four auxiliary functions: Ch (Choice), Maj (Majority), Σ₀ (uppercase Sigma-0), and Σ₁ (uppercase Sigma-1).

The compression function maintains the **hash state** as eight 32-bit words that accumulate the cryptographic mixing effects across all processed blocks. After processing each 512-bit block, the compression function adds the final working variable values to the current hash state, creating the iterative structure that enables processing of arbitrary-length messages.

| Compression Element | Definition | Cryptographic Purpose | Implementation Notes |
|--------------------|------------|----------------------|---------------------|
| Choice function | Ch(x,y,z) = (x ∧ y) ⊕ (¬x ∧ z) | Conditional bit selection | x chooses between y and z |
| Majority function | Maj(x,y,z) = (x ∧ y) ⊕ (x ∧ z) ⊕ (y ∧ z) | Majority vote per bit position | Consensus-based mixing |
| Upper Sigma-0 | Σ₀(x) = ROTR₂(x) ⊕ ROTR₁₃(x) ⊕ ROTR₂₂(x) | Hash state diffusion | Different from schedule σ₀ |
| Upper Sigma-1 | Σ₁(x) = ROTR₆(x) ⊕ ROTR₁₁(x) ⊕ ROTR₂₅(x) | Working variable mixing | Different from schedule σ₁ |

> **Critical Design Insight**: The compression function uses different Sigma functions (uppercase Σ) than the schedule generation (lowercase σ). Confusing these functions is a common implementation error that produces incorrect hash values while still appearing to function normally.

The 64 rounds use the **round constants** `K[0]` through `K[63]`, which are the first 32 bits of the fractional parts of the cube roots of the first 64 prime numbers. These constants ensure that no weak keys or patterns exist in the compression function, providing **nothing-up-my-sleeve** numbers that prevent backdoors or hidden weaknesses.

#### Stage 4: Final Hash Output Generation

The finalization stage converts the final hash state into the standard 256-bit output format and performs validation against known test vectors. After processing all message blocks through the compression function, the eight 32-bit hash state words are concatenated and formatted as a 64-character lowercase hexadecimal string.

The finalization component handles multiple output format options and provides validation mechanisms to ensure correct implementation. The standard output format is a hexadecimal string, but the component also supports binary byte arrays for applications requiring raw hash values.

| Finalization Operation | Input | Output | Validation Method |
|----------------------|-------|--------|------------------|
| Hash concatenation | Eight 32-bit words | 256-bit binary hash | Bit-level assembly |
| Hexadecimal formatting | 256-bit binary hash | 64-character hex string | Character encoding validation |
| Test vector verification | Computed hash | Pass/fail result | NIST reference comparison |
| Empty input handling | Zero-length message | Standard empty hash | Known value verification |

The finalization stage implements proper **endianness handling** to ensure that the output format matches the NIST specification exactly. Each 32-bit hash word is converted to big-endian byte order before hexadecimal formatting, ensuring compatibility with standard SHA-256 implementations across different platforms and languages.

### Component Responsibilities: Clear separation between padding, scheduling, compression, and output formatting

The SHA-256 implementation uses **separation of concerns** to isolate different algorithmic responsibilities into focused components. Each component has a single primary responsibility and well-defined interfaces that enable independent development, testing, and debugging.

> **Decision: Component-Based Architecture**
> - **Context**: SHA-256 involves complex algorithms with distinct mathematical operations and error-prone bit manipulation
> - **Options Considered**:
>   1. Single monolithic function with all operations inline
>   2. Component-based architecture with clear responsibilities
>   3. Object-oriented approach with state machines
> - **Decision**: Component-based architecture with clear responsibilities
> - **Rationale**: Each SHA-256 stage has distinct inputs, outputs, and mathematical operations. Separating them enables focused testing, easier debugging, and better code organization. The algorithmic complexity requires isolation to prevent errors from spreading across stages.
> - **Consequences**: Enables independent testing and validation of each algorithmic stage, simplifies debugging through intermediate value inspection, but requires careful interface design to maintain data flow integrity.

#### Preprocessor Component Responsibilities

The Preprocessor component owns all message formatting operations required to convert arbitrary input messages into standardized block format. This component encapsulates the complex padding algorithm and handles the error-prone bit-level operations required by the NIST specification.

| Responsibility | Scope | Input Validation | Error Handling |
|----------------|-------|------------------|----------------|
| Message encoding | Convert strings to byte arrays | Character encoding verification | Invalid encoding rejection |
| Padding algorithm | Implement 448 mod 512 alignment | Length boundary checking | Overflow prevention |
| Length encoding | 64-bit big-endian conversion | Maximum message size validation | Size limit enforcement |
| Block parsing | 512-bit block creation | Alignment verification | Malformed block detection |

The Preprocessor maintains **stateless operation** to enable thread-safe usage and prevent state pollution between hash computations. All operations are pure functions that produce deterministic outputs based solely on input parameters, with no hidden dependencies or side effects.

The component provides multiple input format options including string messages with UTF-8 encoding and raw byte arrays for binary data. Input validation ensures that character encoding is handled correctly and that message lengths remain within the implementation's supported range.

⚠️ **Pitfall: Bit vs Byte Confusion**
The NIST specification describes padding at the bit level, but most programming languages operate on bytes. Implementers frequently make mistakes when converting between bit positions and byte positions, especially when calculating padding lengths and block boundaries. Always work in consistent units and use explicit conversion functions.

#### Message Scheduler Component Responsibilities

The Message Scheduler component owns the word expansion process that generates 64-word schedules from 512-bit input blocks. This component implements the sigma functions and manages the recurrence relation that creates temporal dependencies across schedule positions.

| Responsibility | Scope | Mathematical Operations | Validation Requirements |
|----------------|-------|------------------------|------------------------|
| Word extraction | Parse 16 initial words | Big-endian conversion | Word boundary alignment |
| Sigma function implementation | σ₀ and σ₁ operations | Right-rotate and XOR | 32-bit overflow masking |
| Schedule expansion | Generate words 16-63 | Recurrence relation | Intermediate result validation |
| Schedule validation | Verify expansion correctness | Known vector comparison | Error detection |

The Message Scheduler implements **bit-rotation utilities** that handle 32-bit word rotation operations correctly across different programming languages and platforms. These utilities mask intermediate results to prevent arithmetic overflow and ensure consistent behavior regardless of the underlying integer implementation.

The component provides debug output options that enable inspection of intermediate schedule values for debugging and validation purposes. This capability is essential for verifying correct implementation of the sigma functions and identifying errors in the word expansion process.

⚠️ **Pitfall: Rotation vs Shift Confusion**
Right-rotate operations are different from right-shift operations. Rotation moves bits that fall off the right end back to the left end, while shifting discards them. Using shift operations instead of rotation in the sigma functions produces completely incorrect schedule values that are difficult to debug.

#### Compression Engine Component Responsibilities

The Compression Engine component owns the 64-round compression algorithm that implements the cryptographic mixing operations at the heart of SHA-256. This component manages the working variables, implements the round functions, and maintains the hash state across multiple block processing operations.

| Responsibility | Scope | State Management | Round Operations |
|----------------|-------|-----------------|------------------|
| Working variable initialization | Copy hash state to a-h | State isolation per block | Proper variable assignment |
| Round function implementation | Ch, Maj, Σ₀, Σ₁ functions | Stateless bit operations | Correct parameter ordering |
| Round iteration | Execute 64 compression rounds | Sequential round processing | Schedule word consumption |
| Hash state update | Add working variables to state | Modular arithmetic handling | 32-bit overflow management |

The Compression Engine maintains the **round constants** array containing the 64 predetermined values specified in the NIST standard. These constants are loaded once during component initialization and used repeatedly during round processing to ensure consistent cryptographic mixing.

The component implements **intermediate result masking** to ensure that all arithmetic operations remain within 32-bit bounds. This masking is critical for maintaining correct behavior across programming languages with different integer overflow semantics.

⚠️ **Pitfall: Function Parameter Confusion**
The Choice and Majority functions have specific parameter orders that must be maintained exactly. Swapping parameters produces valid-looking results but incorrect hash values that are extremely difficult to debug. Always use named parameters or clear variable names to prevent ordering mistakes.

#### Output Formatter Component Responsibilities

The Output Formatter component owns the final hash generation and formatting operations that convert the internal hash state into standard output formats. This component handles endianness conversion, format validation, and test vector verification.

| Responsibility | Scope | Format Options | Validation Methods |
|----------------|-------|---------------|-------------------|
| Hash finalization | Concatenate final hash state | 256-bit binary assembly | State completeness verification |
| Format conversion | Multiple output formats | Hex string, byte array | Format correctness validation |
| Endianness handling | Big-endian output compliance | Consistent byte ordering | Cross-platform compatibility |
| Test vector validation | NIST compliance verification | Known value comparison | Implementation correctness |

The Output Formatter provides **multiple format options** to support different application requirements. The primary format is a 64-character lowercase hexadecimal string, but the component also supports uppercase hexadecimal, raw byte arrays, and base64 encoding for specific use cases.

The component implements **test vector validation** using the standard NIST test cases including the empty string hash and the "abc" input hash. These validations provide immediate feedback about implementation correctness and help identify errors early in the development process.

### Recommended File Organization: Module layout for maintainable implementation

The file organization follows **domain-driven design** principles where each component is isolated in its own module with clear dependencies and minimal coupling. This organization enables independent development and testing while maintaining clear data flow relationships between components.

> **Decision: Module-Per-Component Organization**
> - **Context**: SHA-256 implementation involves multiple complex algorithms with distinct responsibilities and different testing requirements
> - **Options Considered**:
>   1. Single file with all functions together
>   2. Module per component with clear separation
>   3. Layer-based organization (utilities, algorithms, interfaces)
> - **Decision**: Module per component with clear separation
> - **Rationale**: Each SHA-256 component has distinct algorithmic complexity, different testing requirements, and separate debugging needs. Isolating components in separate modules enables focused development, simplifies testing, and improves maintainability. The clear separation also helps learners understand component boundaries.
> - **Consequences**: Enables independent component development and testing, simplifies debugging through component isolation, but requires careful interface design to maintain proper data flow between modules.

#### Core Module Structure

The recommended file organization separates algorithmic components from supporting utilities and provides clear entry points for different usage patterns. The structure enables both library usage and command-line execution while maintaining clean separation of concerns.

```
sha256-implementation/
├── README.md                     # Implementation guide and usage examples
├── requirements.txt              # Python dependencies for testing
├── main.py                       # Command-line interface and examples
├── sha256/
│   ├── __init__.py              # Public API exports
│   ├── hash_function.py         # Main SHA256 class and public interface
│   ├── preprocessor.py          # Message preprocessing and padding
│   ├── scheduler.py             # Message schedule generation
│   ├── compression.py           # Compression function and round operations
│   ├── formatter.py             # Output formatting and validation
│   ├── constants.py             # NIST constants and configuration values
│   └── utilities.py             # Shared bit manipulation utilities
├── tests/
│   ├── __init__.py
│   ├── test_preprocessor.py     # Unit tests for preprocessing
│   ├── test_scheduler.py        # Unit tests for schedule generation
│   ├── test_compression.py      # Unit tests for compression function
│   ├── test_formatter.py        # Unit tests for output formatting
│   ├── test_integration.py      # End-to-end integration tests
│   └── test_vectors.py          # NIST test vector validation
└── examples/
    ├── basic_usage.py           # Simple hash computation examples
    ├── performance_test.py      # Performance benchmarking
    └── debugging_example.py     # Debugging and introspection tools
```

#### Component Module Responsibilities

Each module has a focused responsibility and well-defined interfaces that enable independent development and testing. The modules follow a layered dependency structure where higher-level components depend on lower-level utilities but not on peer components.

| Module | Primary Class/Function | Dependencies | Public Interface |
|--------|----------------------|--------------|------------------|
| `hash_function.py` | `SHA256` class | All components | `hash_message()`, `hash_bytes()` |
| `preprocessor.py` | `MessagePreprocessor` | `utilities.py`, `constants.py` | `preprocess_message()` |
| `scheduler.py` | `MessageScheduler` | `utilities.py`, `constants.py` | `generate_schedule()` |
| `compression.py` | `CompressionEngine` | `utilities.py`, `constants.py` | `compress_block()` |
| `formatter.py` | `OutputFormatter` | `constants.py` | `format_hash()`, `validate_output()` |
| `constants.py` | Module-level constants | None | `INITIAL_HASH_VALUES`, `ROUND_CONSTANTS` |
| `utilities.py` | Bit manipulation functions | None | `right_rotate_32()`, `mask_32_bits()` |

#### Interface Design Principles

The module interfaces follow **dependency inversion** principles where high-level components depend on abstractions rather than concrete implementations. Each component exposes a clean functional interface that accepts well-defined input types and produces predictable outputs without side effects.

| Interface Principle | Implementation | Benefit | Example |
|---------------------|----------------|---------|---------|
| Stateless operations | Pure functions only | Thread safety and predictability | `preprocess_message(data) -> blocks` |
| Type safety | Explicit parameter types | Error prevention and documentation | `MessageBlock`, `HashState` types |
| Single responsibility | One primary function per module | Focused testing and debugging | Preprocessor only handles padding |
| Clear error handling | Explicit exception types | Predictable failure modes | `InvalidMessageError`, `PaddingError` |

The interfaces use **value types** rather than reference types to prevent accidental state modification and ensure that component operations are pure and deterministic. All data structures are immutable or treated as immutable to prevent side effects that could introduce subtle bugs.

#### Testing Organization Strategy

The testing organization enables both unit testing of individual components and integration testing of the complete pipeline. Each component module has corresponding unit tests that verify algorithmic correctness using known intermediate values and edge cases.

| Test Category | File Location | Test Scope | Validation Method |
|---------------|---------------|------------|------------------|
| Unit tests | `test_[component].py` | Individual component functions | Known intermediate values |
| Integration tests | `test_integration.py` | Complete hash pipeline | NIST test vectors |
| Performance tests | `examples/performance_test.py` | Execution timing and memory | Benchmark comparison |
| Debugging tests | `examples/debugging_example.py` | Intermediate value inspection | Manual verification |

The test organization includes **milestone validation** that enables learners to verify correct implementation after completing each development stage. These validation tests provide specific checkpoints with expected outputs that confirm algorithmic correctness before proceeding to the next implementation milestone.

> The testing strategy emphasizes validation at component boundaries rather than internal implementation details. This approach enables refactoring of internal algorithms while maintaining confidence in overall correctness through interface-level validation.

### Implementation Guidance

#### Technology Recommendations

| Component | Simple Option | Advanced Option | Recommended for Learning |
|-----------|---------------|-----------------|------------------------|
| Bit Operations | Built-in operators (`&`, `|`, `^`, `<<`, `>>`) | NumPy arrays for vectorization | Built-in operators |
| Integer Handling | Standard int with manual masking | ctypes for fixed-width types | Standard int with masking |
| Testing Framework | Built-in unittest module | pytest with fixtures | unittest for simplicity |
| Performance Profiling | Basic timing with `time.time()` | cProfile for detailed analysis | Basic timing initially |
| Documentation | Inline comments and docstrings | Sphinx for API documentation | Inline comments |

#### Recommended File Structure Implementation

```python
# sha256/__init__.py
"""SHA-256 implementation following NIST specification."""
from .hash_function import SHA256, hash_message, hash_bytes

__version__ = "1.0.0"
__all__ = ["SHA256", "hash_message", "hash_bytes"]

# sha256/constants.py
"""NIST SHA-256 constants and configuration values."""

# Initial hash values - first 32 bits of fractional parts of square roots of first 8 primes
INITIAL_HASH_VALUES = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9c5, 0x5be0cd19
]

# Round constants - first 32 bits of fractional parts of cube roots of first 64 primes
ROUND_CONSTANTS = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    # ... (complete array of 64 constants)
]

# Block and word size constants
BLOCK_SIZE_BITS = 512
WORD_SIZE_BITS = 32
HASH_SIZE_BITS = 256
BLOCK_SIZE_BYTES = BLOCK_SIZE_BITS // 8
WORD_SIZE_BYTES = WORD_SIZE_BITS // 8

# sha256/utilities.py
"""Shared bit manipulation utilities for SHA-256 implementation."""

def right_rotate_32(value: int, amount: int) -> int:
    """Perform right rotation of 32-bit value by specified amount.
    
    Args:
        value: 32-bit integer to rotate
        amount: Number of bit positions to rotate right (0-31)
        
    Returns:
        Rotated 32-bit value
    """
    # Ensure value is masked to 32 bits
    value = value & 0xFFFFFFFF
    amount = amount % 32
    
    # Rotate: bits shifted right OR bits wrapped from right to left
    return ((value >> amount) | (value << (32 - amount))) & 0xFFFFFFFF

def mask_32_bits(value: int) -> int:
    """Mask value to 32 bits to handle integer overflow.
    
    Args:
        value: Integer value that may exceed 32 bits
        
    Returns:
        Value masked to 32 bits (0 to 0xFFFFFFFF)
    """
    return value & 0xFFFFFFFF

def bytes_to_words(byte_array: bytes) -> list[int]:
    """Convert byte array to array of 32-bit words in big-endian order.
    
    Args:
        byte_array: Bytes to convert (length must be multiple of 4)
        
    Returns:
        List of 32-bit words
    """
    if len(byte_array) % 4 != 0:
        raise ValueError("Byte array length must be multiple of 4")
    
    words = []
    for i in range(0, len(byte_array), 4):
        # Convert 4 bytes to 32-bit word in big-endian order
        word = (byte_array[i] << 24) | (byte_array[i+1] << 16) | \
               (byte_array[i+2] << 8) | byte_array[i+3]
        words.append(word)
    
    return words

def words_to_bytes(words: list[int]) -> bytes:
    """Convert array of 32-bit words to byte array in big-endian order.
    
    Args:
        words: List of 32-bit words
        
    Returns:
        Byte array representation
    """
    byte_array = bytearray()
    for word in words:
        # Convert 32-bit word to 4 bytes in big-endian order
        byte_array.append((word >> 24) & 0xFF)
        byte_array.append((word >> 16) & 0xFF)
        byte_array.append((word >> 8) & 0xFF)
        byte_array.append(word & 0xFF)
    
    return bytes(byte_array)
```

#### Core Logic Skeleton Structure

```python
# sha256/hash_function.py
"""Main SHA-256 hash function interface."""
from .preprocessor import MessagePreprocessor
from .scheduler import MessageScheduler  
from .compression import CompressionEngine
from .formatter import OutputFormatter
from .constants import INITIAL_HASH_VALUES

class SHA256:
    """SHA-256 hash function implementation."""
    
    def __init__(self):
        self.preprocessor = MessagePreprocessor()
        self.scheduler = MessageScheduler()
        self.compression = CompressionEngine()
        self.formatter = OutputFormatter()
    
    def hash_message(self, message: str) -> str:
        """Compute SHA-256 hash of string message.
        
        Args:
            message: Input string to hash
            
        Returns:
            64-character lowercase hexadecimal hash
        """
        # TODO 1: Convert string to bytes using UTF-8 encoding
        # TODO 2: Call hash_bytes with the byte representation
        # TODO 3: Return the hexadecimal hash result
        pass
    
    def hash_bytes(self, data: bytes) -> str:
        """Compute SHA-256 hash of byte data.
        
        Args:
            data: Input bytes to hash
            
        Returns:
            64-character lowercase hexadecimal hash
        """
        # TODO 1: Preprocess message into 512-bit blocks using self.preprocessor
        # TODO 2: Initialize hash state to INITIAL_HASH_VALUES
        # TODO 3: For each block:
        #   - Generate message schedule using self.scheduler
        #   - Compress block using self.compression with current hash state
        #   - Update hash state with compression result
        # TODO 4: Format final hash state using self.formatter
        # TODO 5: Return formatted hexadecimal string
        pass

# Convenience functions for direct usage
def hash_message(message: str) -> str:
    """Compute SHA-256 hash of string message."""
    sha256 = SHA256()
    return sha256.hash_message(message)

def hash_bytes(data: bytes) -> str:
    """Compute SHA-256 hash of byte data."""
    sha256 = SHA256()
    return sha256.hash_bytes(data)
```

#### Component Interface Templates

```python
# sha256/preprocessor.py  
"""Message preprocessing and padding implementation."""
from .constants import BLOCK_SIZE_BITS, BLOCK_SIZE_BYTES
from .utilities import mask_32_bits

class MessageBlock:
    """Represents a 512-bit message block."""
    def __init__(self, data: bytes):
        if len(data) != BLOCK_SIZE_BYTES:
            raise ValueError(f"Block must be exactly {BLOCK_SIZE_BYTES} bytes")
        self.data = data

class MessagePreprocessor:
    """Handles message padding and block parsing."""
    
    def preprocess_message(self, message: bytes) -> list[MessageBlock]:
        """Convert message to array of 512-bit blocks with proper padding.
        
        Args:
            message: Raw message bytes
            
        Returns:
            List of MessageBlock objects
        """
        # TODO 1: Calculate original message length in bits
        # TODO 2: Append single '1' bit to message  
        # TODO 3: Calculate zero padding needed to reach 448 mod 512 bits
        # TODO 4: Append zero padding to align to 448 mod 512
        # TODO 5: Append original length as 64-bit big-endian integer
        # TODO 6: Verify total length is multiple of 512 bits
        # TODO 7: Split padded message into 512-bit MessageBlock objects
        # TODO 8: Return list of MessageBlock objects
        pass

# sha256/scheduler.py
"""Message schedule generation implementation."""
from .utilities import right_rotate_32, mask_32_bits
from .constants import WORD_SIZE_BITS

class MessageSchedule:
    """Represents 64-word message schedule for compression."""
    def __init__(self, words: list[int]):
        if len(words) != 64:
            raise ValueError("Schedule must contain exactly 64 words")
        self.words = words

class MessageScheduler:
    """Generates 64-word message schedules from 512-bit blocks."""
    
    def generate_schedule(self, block: 'MessageBlock') -> MessageSchedule:
        """Generate 64-word schedule from 512-bit block.
        
        Args:
            block: MessageBlock to expand
            
        Returns:
            MessageSchedule with 64 32-bit words
        """
        # TODO 1: Parse block into 16 initial 32-bit words (big-endian)
        # TODO 2: Initialize schedule array with 64 positions
        # TODO 3: Copy initial 16 words to schedule positions 0-15
        # TODO 4: For positions 16-63:
        #   - Calculate sigma1(W[t-2]) using _sigma1 function
        #   - Add W[t-7] 
        #   - Add sigma0(W[t-15]) using _sigma0 function
        #   - Add W[t-16]
        #   - Mask result to 32 bits and store in W[t]
        # TODO 5: Return MessageSchedule object with completed schedule
        pass
    
    def _sigma0(self, x: int) -> int:
        """Lower-case sigma-0 function for schedule generation."""
        # TODO: Implement ROTR(7) XOR ROTR(18) XOR SHR(3)
        pass
    
    def _sigma1(self, x: int) -> int:
        """Lower-case sigma-1 function for schedule generation."""
        # TODO: Implement ROTR(17) XOR ROTR(19) XOR SHR(10)
        pass
```

#### Milestone Validation Checkpoints

**After Milestone 1 (Preprocessing):**
```python
# Test basic preprocessing
preprocessor = MessagePreprocessor()
blocks = preprocessor.preprocess_message(b"abc")
print(f"Number of blocks: {len(blocks)}")  # Should be 1
print(f"Block length: {len(blocks[0].data)} bytes")  # Should be 64
```

**After Milestone 2 (Schedule Generation):**
```python
# Test schedule generation  
scheduler = MessageScheduler()
schedule = scheduler.generate_schedule(blocks[0])
print(f"Schedule length: {len(schedule.words)}")  # Should be 64
print(f"First word: 0x{schedule.words[0]:08x}")  # Should match expected
```

**After Milestone 3 (Compression):**
```python
# Test compression function
engine = CompressionEngine()
initial_state = INITIAL_HASH_VALUES.copy()
final_state = engine.compress_block(schedule, initial_state)
print(f"Hash state words: {[f'0x{word:08x}' for word in final_state]}")
```

**After Milestone 4 (Complete Implementation):**
```python
# Test complete implementation against known vectors
sha256 = SHA256()
empty_hash = sha256.hash_message("")
abc_hash = sha256.hash_message("abc")
print(f"Empty string: {empty_hash}")
print(f"Expected:     e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
print(f"ABC string:   {abc_hash}")  
print(f"Expected:     ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
```

#### Python-Specific Implementation Notes

- **Integer Handling**: Python integers have arbitrary precision, so use `& 0xFFFFFFFF` to mask to 32 bits after arithmetic operations
- **Byte Order**: Use `int.from_bytes(data, 'big')` and `int.to_bytes(4, 'big')` for endianness conversion  
- **Bit Operations**: Python's `>>` and `<<` operators work correctly, but be careful with negative numbers in right shifts
- **String Encoding**: Use `message.encode('utf-8')` to convert strings to bytes for hashing
- **Performance**: For production use, consider using the `hashlib` module, but implement from scratch for learning

#### Common Implementation Pitfalls

| Symptom | Likely Cause | Diagnosis | Fix |
|---------|--------------|-----------|-----|
| Wrong hash for empty string | Padding calculation error | Check bit vs byte conversion | Use bit-level padding math |
| Incorrect "abc" hash | Sigma function confusion | Verify uppercase vs lowercase sigma | Use correct rotation amounts |
| Hash changes between runs | Mutable state pollution | Check for shared state | Make all operations stateless |
| Integer overflow errors | Missing 32-bit masking | Add logging for large values | Mask after every arithmetic operation |


## Data Model and Types

> **Milestone(s):** Foundational understanding for all milestones (1-4)

### Mental Model: Recipe Ingredients and Containers

Think of the SHA-256 data model as a **professional kitchen with specialized containers** for different ingredients and stages of a complex recipe. Just as a chef uses specific bowls for raw ingredients, mixing bowls for preparation, and serving dishes for final presentation, SHA-256 uses precisely defined data types for each stage of hash computation. Each container has exact dimensions (bit sizes), specific contents (data formats), and clear purposes (processing stages). The raw message is like ingredients that must be measured, prepared, and combined using exact procedures to produce the final hash "dish."

This analogy helps us understand why SHA-256 requires such precise data structures: just as baking requires exact measurements and proper containers, cryptographic hashing demands exact bit layouts and type safety to maintain security properties.

### Core Data Types

The SHA-256 algorithm operates on several fundamental data types that represent different stages of the hash computation pipeline. Each type serves a specific purpose and maintains precise bit-level formatting to ensure cryptographic correctness.

**Message and Block Representations**

The input message undergoes several transformations, each requiring specific data structures. The algorithm processes data in fixed-size chunks, similar to how assembly lines process items in standardized batches.

| Type | Size | Purpose | Format |
|------|------|---------|--------|
| `MessageBlock` | 512 bits | Single processing unit | Array of 16 32-bit words in big-endian |
| `HashState` | 256 bits | Current hash values | Array of 8 32-bit words (h0-h7) |
| `MessageSchedule` | 2048 bits | Expanded block data | Array of 64 32-bit words |
| `WorkingVariables` | 256 bits | Temporary compression state | 8 variables (a,b,c,d,e,f,g,h) |

The `MessageBlock` represents exactly 512 bits of padded message data. This fixed size enables the Merkle-Damgård construction to process arbitrarily long messages in uniform chunks. Each block contains 16 words, where each word is a 32-bit unsigned integer stored in big-endian byte order.

> **Key Insight**: The 512-bit block size was chosen to balance security and performance. Larger blocks would require more memory and computation, while smaller blocks would increase the number of compression rounds needed for long messages.

**Hash State Evolution**

The `HashState` type maintains the evolving hash values throughout block processing. This represents the "memory" of the hash function, carrying forward the cumulative effect of all previously processed blocks.

| Field | Type | Description | Initial Value |
|-------|------|-------------|---------------|
| h0 | 32-bit word | Hash value 0 | 0x6a09e667 |
| h1 | 32-bit word | Hash value 1 | 0xbb67ae85 |
| h2 | 32-bit word | Hash value 2 | 0x3c6ef372 |
| h3 | 32-bit word | Hash value 3 | 0xa54ff53a |
| h4 | 32-bit word | Hash value 4 | 0x510e527f |
| h5 | 32-bit word | Hash value 5 | 0x9b05688c |
| h6 | 32-bit word | Hash value 6 | 0x1f83d9ab |
| h7 | 32-bit word | Hash value 7 | 0x5be0cd19 |

**Working Variables and Temporary State**

During compression, the algorithm uses working variables to perform the 64 rounds of mixing operations. These variables evolve through each round, combining the current hash state with the message schedule and round constants.

| Variable | Purpose | Updates Per Round | Final Action |
|----------|---------|-------------------|--------------|
| a | Primary accumulator | Receives new mixed value | Added to h0 |
| b | Secondary buffer | Rotated from a | Added to h1 |
| c | Tertiary buffer | Rotated from b | Added to h2 |
| d | Quaternary buffer | Rotated from c | Added to h3 |
| e | Fifth buffer | Mixed with compression result | Added to h4 |
| f | Sixth buffer | Rotated from e | Added to h5 |
| g | Seventh buffer | Rotated from f | Added to h6 |
| h | Eighth buffer | Rotated from g | Added to h7 |

> **Architecture Decision: 32-bit Word Size**
> - **Context**: SHA-256 operations could use 8-bit, 16-bit, 32-bit, or 64-bit word sizes
> - **Options Considered**: 
>   - 8-bit words: Simple but requires many operations
>   - 16-bit words: Moderate complexity
>   - 32-bit words: Balances efficiency and security
>   - 64-bit words: Used in SHA-512 but unnecessary for 256-bit output
> - **Decision**: 32-bit words throughout
> - **Rationale**: Matches common processor architectures, provides sufficient mixing in reasonable rounds, and aligns with 256-bit output size (8 × 32 = 256)
> - **Consequences**: Enables efficient implementation on 32-bit and 64-bit systems, requires exactly 64 compression rounds

### SHA-256 Constants

The SHA-256 specification defines precise mathematical constants that provide security properties and prevent malicious constant selection. These constants derive from mathematical properties of prime numbers and fractional parts of square roots and cube roots.

**Initial Hash Values**

The `INITIAL_HASH_VALUES` represent the fractional parts of square roots of the first eight prime numbers. This mathematical derivation ensures no hidden patterns or backdoors exist in the constants.

| Constant | Mathematical Source | Hex Value | Purpose |
|----------|-------------------|-----------|---------|
| h0_init | sqrt(2) fractional part | 0x6a09e667 | Initial h0 |
| h1_init | sqrt(3) fractional part | 0xbb67ae85 | Initial h1 |
| h2_init | sqrt(5) fractional part | 0x3c6ef372 | Initial h2 |
| h3_init | sqrt(7) fractional part | 0xa54ff53a | Initial h3 |
| h4_init | sqrt(11) fractional part | 0x510e527f | Initial h4 |
| h5_init | sqrt(13) fractional part | 0x9b05688c | Initial h5 |
| h6_init | sqrt(17) fractional part | 0x1f83d9ab | Initial h6 |
| h7_init | sqrt(19) fractional part | 0x5be0cd19 | Initial h7 |

The mathematical derivation follows this process: take the square root of prime p, subtract the integer part, multiply by 2^32, and truncate to get a 32-bit constant. This ensures the constants have no exploitable mathematical structure while providing sufficient randomness for cryptographic security.

**Round Constants**

The `ROUND_CONSTANTS` array contains 64 values derived from cube roots of the first 64 prime numbers. Each round of compression uses one constant to break symmetry and ensure different rounds produce different mixing patterns.

| Index Range | Mathematical Source | Example Values | Usage |
|-------------|-------------------|----------------|-------|
| K[0-15] | Cube roots of primes 2-47 | K[0]=0x428a2f98, K[1]=0x71374491 | Rounds 0-15 |
| K[16-31] | Cube roots of primes 53-127 | K[16]=0xe9b5dba5, K[17]=0x3956c25b | Rounds 16-31 |
| K[32-47] | Cube roots of primes 131-211 | K[32]=0x748f82ee, K[33]=0x78a5636f | Rounds 32-47 |
| K[48-63] | Cube roots of primes 223-311 | K[48]=0x90befffa, K[63]=0xc67178f2 | Rounds 48-63 |

The cube root derivation ensures each constant is mathematically independent and cryptographically strong. The process takes cube root of prime p, subtracts integer part, multiplies by 2^32, and truncates to 32 bits.

> **Security Insight**: Using mathematically derived constants prevents the "nothing up my sleeve" attack where algorithm designers might choose constants with hidden backdoors. The mathematical derivation proves no secret patterns exist.

**Algorithmic Constants**

Additional constants define the algorithm's structural parameters and ensure consistent implementation across different platforms and languages.

| Constant | Value | Purpose | Critical Property |
|----------|--------|---------|-------------------|
| `BLOCK_SIZE_BITS` | 512 | Message block size | Must align with Merkle-Damgård construction |
| `WORD_SIZE_BITS` | 32 | Word size for operations | Must match processor efficiency |
| `HASH_SIZE_BITS` | 256 | Final output size | Determines security level |
| `SCHEDULE_SIZE_WORDS` | 64 | Message schedule length | Must provide sufficient mixing rounds |
| `HASH_SIZE_WORDS` | 8 | Number of hash state words | 8 × 32 = 256 bits |

### Type Relationships and Flow

![Data Type Relationships](./diagrams/data-model-relationships.svg)

The SHA-256 data types form a directed acyclic graph of transformations, where each type represents a specific stage in the hash computation pipeline. Understanding these relationships helps implement correct data flow and type conversions.

**Input to Block Transformation**

The initial transformation converts variable-length input messages into standardized blocks suitable for cryptographic processing.

| Source Type | Transformation | Target Type | Key Operations |
|-------------|----------------|-------------|----------------|
| String/Bytes | Padding + Parsing | `MessageBlock[]` | Bit padding, length encoding, block splitting |
| `MessageBlock` | Word extraction | 32-bit words[16] | Big-endian conversion, word alignment |

The padding transformation ensures all messages, regardless of original length, convert to an exact multiple of 512-bit blocks. This uniformity enables the compression function to operate on consistent data structures without special cases for short or odd-length inputs.

**Schedule Generation Flow**

Each message block undergoes expansion to create the message schedule, providing 64 words of mixed data for the compression rounds.

| Input | Process | Output | Bit Operations |
|-------|---------|--------|----------------|
| `MessageBlock` (16 words) | Word expansion | `MessageSchedule` (64 words) | Right rotate, XOR, addition |
| Words W[0-15] | Direct copy | Schedule S[0-15] | No transformation |
| Words W[16-63] | Sigma functions | Schedule S[16-63] | σ0, σ1, modular addition |

The expansion process creates non-linear dependencies between input bits and schedule words. Each of the 48 generated words depends on multiple previous words through rotation and XOR operations, ensuring avalanche effect propagation throughout the schedule.

**Compression State Evolution**

The compression function maintains evolving state through 64 rounds of transformation, combining schedule words with hash state and round constants.

| State Component | Evolution Pattern | Update Frequency | Dependencies |
|-----------------|------------------|------------------|--------------|
| Working Variables (a-h) | Round-by-round rotation | Every round | Previous variables, schedule word, round constant |
| Hash State (h0-h7) | Block-by-block addition | Every block | Final working variables, previous hash values |
| Intermediate Values | Temporary calculation | Within rounds | Sigma functions, choice/majority functions |

The working variables serve as a mixing chamber where each round thoroughly combines new input (schedule word), accumulated state (variables), and cryptographic randomness (round constant). The eight-variable structure ensures each bit position influences multiple other positions through the round function networks.

**Output Finalization Chain**

The final hash output requires careful type conversion to ensure consistent representation across different systems and applications.

| Internal Type | Conversion | Output Format | Encoding Requirements |
|---------------|------------|---------------|----------------------|
| `HashState` (8 × 32-bit) | Concatenation | 256-bit binary | Big-endian word order |
| 256-bit binary | Hex encoding | 64-character string | Lowercase hexadecimal |
| 256-bit binary | Byte array | 32-byte array | Platform-specific byte order |

> **Architecture Decision: Big-Endian Throughout**
> - **Context**: Hash functions must produce identical outputs across different processor architectures
> - **Options Considered**:
>   - Native endianness: Fast but inconsistent across platforms  
>   - Little-endian: Common on x86 but not universal
>   - Big-endian: Network byte order, cryptographic standard
> - **Decision**: Big-endian byte order for all multi-byte values
> - **Rationale**: Ensures identical hash outputs on all platforms, matches network protocols, follows cryptographic conventions
> - **Consequences**: May require byte-swapping on little-endian systems, but guarantees correctness

**Type Safety and Overflow Handling**

SHA-256 arithmetic operates in finite fields, requiring careful overflow management to maintain mathematical properties and ensure identical results across implementations.

| Operation Type | Overflow Behavior | Masking Required | Security Impact |
|----------------|-------------------|------------------|-----------------|
| Word addition | Modulo 2^32 | Mask to 32 bits | Prevents bias in mixing |
| Bit rotation | No overflow | No masking needed | Preserves bit patterns |
| XOR operations | No overflow | No masking needed | Linear operation safety |
| Schedule generation | Modulo 2^32 | Mask intermediate results | Maintains expansion properties |

The modular arithmetic ensures all operations remain within defined bit ranges, preventing undefined behavior that could compromise cryptographic security. Each implementation must mask results to exactly 32 bits after addition operations.

**Memory Layout Considerations**

Efficient SHA-256 implementation requires understanding how data types map to memory structures, particularly for performance-critical applications.

| Type | Memory Size | Alignment | Cache Behavior |
|------|-------------|-----------|----------------|
| `MessageBlock` | 64 bytes | 4-byte word alignment | Single cache line (typical) |
| `MessageSchedule` | 256 bytes | 4-byte word alignment | Multiple cache lines |
| `HashState` | 32 bytes | 4-byte word alignment | Single cache line |
| `WorkingVariables` | 32 bytes | 4-byte word alignment | Single cache line |

Understanding memory layout helps optimize implementation for cache efficiency and memory bandwidth, particularly important for high-throughput applications processing many messages.

⚠️ **Pitfall: Integer Overflow Neglect**
Many implementations fail to properly mask 32-bit arithmetic operations, leading to incorrect results on 64-bit systems where intermediate calculations exceed 32-bit ranges. Always apply `& 0xFFFFFFFF` masks after addition operations to ensure modulo 2^32 arithmetic. Forgetting this mask causes hash values to differ between 32-bit and 64-bit platforms, breaking interoperability.

⚠️ **Pitfall: Endianness Confusion**
Converting between byte arrays and 32-bit words requires careful attention to byte order. The NIST specification mandates big-endian byte order, but many platforms use little-endian natively. Always use explicit big-endian conversion functions rather than casting pointers, which produces platform-dependent results.

⚠️ **Pitfall: Type Size Assumptions**
Assuming `int` or `long` types have specific bit sizes leads to portability issues. Always use explicitly sized types (32-bit unsigned integers) or verify type sizes match requirements. Hash correctness depends on exact bit widths, not approximate sizes.

### Implementation Guidance

**A. Technology Recommendations**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Integer Types | Built-in 32-bit integers with manual masking | Custom Word32 class with automatic overflow |
| Byte Arrays | Native byte arrays with manual endian conversion | Struct.pack/unpack for automatic endian handling |
| Type Safety | Manual type checking and assertions | Type hints with mypy static analysis |
| Memory Management | Standard lists and basic arrays | NumPy arrays for cache-efficient operations |

**B. Recommended File Structure**

```
sha256/
    __init__.py              ← main hash interface
    types.py                 ← data type definitions
    constants.py             ← NIST constants and values
    preprocessor.py          ← message padding and parsing
    scheduler.py             ← message schedule generation  
    compressor.py            ← compression function
    formatter.py             ← output formatting
    utils.py                 ← bit operations and helpers
    test_vectors.py          ← NIST test cases
```

**C. Infrastructure Starter Code**

```python
# types.py - Complete data type definitions
from typing import List, NewType
import struct

# Type aliases for clarity and type safety
Word32 = NewType('Word32', int)
MessageBlock = NewType('MessageBlock', List[Word32])
MessageSchedule = NewType('MessageSchedule', List[Word32])
HashState = NewType('HashState', List[Word32])

class SHA256Types:
    """Container for SHA-256 type utilities and validation."""
    
    @staticmethod
    def mask_32_bits(value: int) -> Word32:
        """Ensure value fits in exactly 32 bits."""
        return Word32(value & 0xFFFFFFFF)
    
    @staticmethod
    def bytes_to_words_big_endian(data: bytes) -> List[Word32]:
        """Convert byte array to 32-bit words in big-endian order."""
        words = []
        for i in range(0, len(data), 4):
            word_bytes = data[i:i+4]
            if len(word_bytes) < 4:
                word_bytes += b'\x00' * (4 - len(word_bytes))
            word = struct.unpack('>I', word_bytes)[0]
            words.append(Word32(word))
        return words
    
    @staticmethod
    def words_to_bytes_big_endian(words: List[Word32]) -> bytes:
        """Convert 32-bit words to byte array in big-endian order."""
        result = b''
        for word in words:
            result += struct.pack('>I', word)
        return result
    
    @staticmethod
    def validate_hash_state(state: List[Word32]) -> bool:
        """Verify hash state has exactly 8 32-bit words."""
        return len(state) == 8 and all(0 <= word <= 0xFFFFFFFF for word in state)
    
    @staticmethod
    def validate_message_block(block: List[Word32]) -> bool:
        """Verify message block has exactly 16 32-bit words."""
        return len(block) == 16 and all(0 <= word <= 0xFFFFFFFF for word in block)

# constants.py - Complete NIST constants
INITIAL_HASH_VALUES: List[Word32] = [
    Word32(0x6a09e667), Word32(0xbb67ae85), Word32(0x3c6ef372), Word32(0xa54ff53a),
    Word32(0x510e527f), Word32(0x9b05688c), Word32(0x1f83d9ab), Word32(0x5be0cd19)
]

ROUND_CONSTANTS: List[Word32] = [
    Word32(0x428a2f98), Word32(0x71374491), Word32(0xb5c0fbcf), Word32(0xe9b5dba5),
    Word32(0x3956c25b), Word32(0x59f111f1), Word32(0x923f82a4), Word32(0xab1c5ed5),
    Word32(0xd807aa98), Word32(0x12835b01), Word32(0x243185be), Word32(0x550c7dc3),
    Word32(0x72be5d74), Word32(0x80deb1fe), Word32(0x9bdc06a7), Word32(0xc19bf174),
    Word32(0xe49b69c1), Word32(0xefbe4786), Word32(0x0fc19dc6), Word32(0x240ca1cc),
    Word32(0x2de92c6f), Word32(0x4a7484aa), Word32(0x5cb0a9dc), Word32(0x76f988da),
    Word32(0x983e5152), Word32(0xa831c66d), Word32(0xb00327c8), Word32(0xbf597fc7),
    Word32(0xc6e00bf3), Word32(0xd5a79147), Word32(0x06ca6351), Word32(0x14292967),
    Word32(0x27b70a85), Word32(0x2e1b2138), Word32(0x4d2c6dfc), Word32(0x53380d13),
    Word32(0x650a7354), Word32(0x766a0abb), Word32(0x81c2c92e), Word32(0x92722c85),
    Word32(0xa2bfe8a1), Word32(0xa81a664b), Word32(0xc24b8b70), Word32(0xc76c51a3),
    Word32(0xd192e819), Word32(0xd6990624), Word32(0xf40e3585), Word32(0x106aa070),
    Word32(0x19a4c116), Word32(0x1e376c08), Word32(0x2748774c), Word32(0x34b0bcb5),
    Word32(0x391c0cb3), Word32(0x4ed8aa4a), Word32(0x5b9cca4f), Word32(0x682e6ff3),
    Word32(0x748f82ee), Word32(0x78a5636f), Word32(0x84c87814), Word32(0x8cc70208),
    Word32(0x90befffa), Word32(0xa4506ceb), Word32(0xbef9a3f7), Word32(0xc67178f2)
]

BLOCK_SIZE_BITS = 512
WORD_SIZE_BITS = 32 
HASH_SIZE_BITS = 256
SCHEDULE_SIZE_WORDS = 64
HASH_SIZE_WORDS = 8

# utils.py - Complete bit operation utilities
def right_rotate_32(value: Word32, amount: int) -> Word32:
    """Perform 32-bit right rotation."""
    amount = amount % 32  # Handle rotation amounts > 32
    return Word32(((value >> amount) | (value << (32 - amount))) & 0xFFFFFFFF)

def right_shift_32(value: Word32, amount: int) -> Word32:
    """Perform 32-bit right shift (zero-fill)."""
    return Word32((value >> amount) & 0xFFFFFFFF)
```

**D. Core Logic Skeleton Code**

```python
# sha256.py - Main interface (TODO: implement core logic)
from typing import List, Union
from .types import *
from .constants import *

class SHA256:
    """SHA-256 hash function implementation."""
    
    def __init__(self):
        self.hash_state: HashState = HashState(INITIAL_HASH_VALUES.copy())
    
    def hash_message(self, message: Union[str, bytes]) -> str:
        """Compute SHA-256 hash of message and return hex string."""
        # TODO 1: Convert string to bytes if needed (use UTF-8 encoding)
        # TODO 2: Call preprocess_message to get list of MessageBlocks
        # TODO 3: For each block: generate schedule, compress, update hash_state
        # TODO 4: Call format_hash to convert final hash_state to hex string
        # TODO 5: Reset hash_state to initial values for next use
        pass
    
    def hash_bytes(self, data: bytes) -> str:
        """Compute SHA-256 hash of byte data."""
        # TODO: This should call hash_message after converting bytes appropriately
        pass
    
    def preprocess_message(self, message: bytes) -> List[MessageBlock]:
        """Convert message to padded 512-bit blocks."""
        # TODO 1: Convert message to bit representation
        # TODO 2: Append single '1' bit to message
        # TODO 3: Calculate padding needed to reach 448 mod 512 bits
        # TODO 4: Append zero bits for padding alignment
        # TODO 5: Append original message length as 64-bit big-endian integer
        # TODO 6: Split padded message into 512-bit blocks
        # TODO 7: Convert each block to MessageBlock (16 32-bit words)
        pass
    
    def generate_schedule(self, block: MessageBlock) -> MessageSchedule:
        """Generate 64-word message schedule from 16-word block."""
        # TODO 1: Copy first 16 words directly from block to schedule
        # TODO 2: For words 16-63, use schedule expansion formula:
        #         W[i] = σ1(W[i-2]) + W[i-7] + σ0(W[i-15]) + W[i-16]
        # TODO 3: Implement σ0(x) = ROTR(x,7) XOR ROTR(x,18) XOR SHR(x,3)  
        # TODO 4: Implement σ1(x) = ROTR(x,17) XOR ROTR(x,19) XOR SHR(x,10)
        # TODO 5: Mask all intermediate results to 32 bits
        pass
    
    def compress_block(self, schedule: MessageSchedule, hash_state: HashState) -> HashState:
        """Execute 64 rounds of compression on message schedule."""
        # TODO 1: Initialize working variables a,b,c,d,e,f,g,h from hash_state
        # TODO 2: For rounds 0-63:
        #   TODO 2a: Calculate T1 = h + Σ1(e) + Ch(e,f,g) + K[round] + W[round]
        #   TODO 2b: Calculate T2 = Σ0(a) + Maj(a,b,c)
        #   TODO 2c: Update variables: h=g, g=f, f=e, e=d+T1, d=c, c=b, b=a, a=T1+T2
        #   TODO 2d: Mask all values to 32 bits
        # TODO 3: Add compressed working variables to hash_state values
        # TODO 4: Return new hash_state (mask final additions to 32 bits)
        pass
    
    def format_hash(self, hash_state: HashState) -> str:
        """Convert hash state to 64-character hex string."""
        # TODO 1: Convert each hash word to big-endian bytes
        # TODO 2: Concatenate all 8 words into 32-byte array
        # TODO 3: Convert byte array to lowercase hexadecimal string
        # TODO 4: Verify result is exactly 64 characters
        pass
```

**E. Language-Specific Hints**

- **Bit Operations**: Use `>>` for right shift, `<<` for left shift, `^` for XOR, `&` for AND, `|` for OR
- **32-bit Masking**: Always apply `& 0xFFFFFFFF` after arithmetic operations to ensure 32-bit results
- **Big-Endian Conversion**: Use `struct.pack('>I', word)` and `struct.unpack('>I', bytes)[0]` for consistent endianness
- **Hex Formatting**: Use `f"{value:08x}"` to format 32-bit words as 8-character lowercase hex strings
- **Type Safety**: Enable mypy checking with `python -m mypy sha256/` to catch type errors early

**F. Milestone Checkpoints**

After implementing each milestone, verify correct behavior:

**Milestone 1 Checkpoint (Preprocessing)**:
- Empty string should pad to exactly one 512-bit block
- String "abc" should pad to exactly one 512-bit block  
- Check final 64 bits encode original message length correctly
- Verify block parsing produces MessageBlock objects with 16 words each

**Milestone 2 Checkpoint (Message Schedule)**:
- First 16 schedule words should match block words exactly
- Schedule words 16-63 should differ from block words (expansion working)
- Verify schedule has exactly 64 32-bit words
- Test sigma functions produce expected bit patterns

**Milestone 3 Checkpoint (Compression)**:
- Working variables should change after each compression round
- Final hash state should differ from initial values (compression working)
- Verify all intermediate calculations stay within 32-bit bounds
- Check round constants access correctly (no index errors)

**Milestone 4 Checkpoint (Output)**:
- Empty string hash: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- String "abc" hash: `ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad`
- Verify output is exactly 64 lowercase hex characters
- Test multiple consecutive hashes produce consistent results


## Message Preprocessing Component

> **Milestone(s):** Milestone 1 - Message Preprocessing

### Mental Model: Document Formatting

Think of message preprocessing as **preparing documents for industrial printing**. Just as documents must be formatted to exact specifications before entering a printing press - with consistent margins, page breaks, and header information - SHA-256 requires messages to be formatted into precise 512-bit blocks before cryptographic processing can begin.

When you send a document to a professional printer, the document goes through preprocessing: adding padding to reach standard page sizes, inserting page numbers, and ensuring consistent formatting. Similarly, SHA-256's preprocessing takes your arbitrary-length message and transforms it into standardized blocks that the compression function can process reliably.

The analogy extends further: just as a printer needs to know the total page count before starting (to manage paper feeding and collation), SHA-256 embeds the original message length into each padded message. This prevents attackers from crafting messages where different inputs produce the same block structure - a critical security property called **length extension attack resistance**.

The preprocessing stage serves as the **format standardizer** that ensures every message, whether it's a single character or a multi-gigabyte file, gets transformed into uniform 512-bit blocks that flow smoothly through the remaining SHA-256 pipeline.

### Padding Algorithm

The SHA-256 padding algorithm follows a precise three-step process defined in the NIST specification. This algorithm ensures that every input message, regardless of length, gets transformed into complete 512-bit blocks suitable for cryptographic processing.

The padding algorithm addresses a fundamental challenge in cryptographic hash functions: messages arrive in arbitrary lengths, but the compression function operates on fixed-size blocks. Without standardized padding, different messages could produce identical block sequences, creating security vulnerabilities.

> **Decision: Three-Stage Padding Algorithm**
> - **Context**: Need to convert arbitrary-length messages into 512-bit aligned blocks while preserving security properties
> - **Options Considered**: Simple zero-padding, length-prefixed padding, NIST three-stage padding
> - **Decision**: NIST three-stage padding with bit-level precision
> - **Rationale**: Prevents length extension attacks, ensures unambiguous message boundaries, maintains collision resistance properties
> - **Consequences**: More complex implementation but provides cryptographic security guarantees required for SHA-256

The three stages of the padding algorithm work together to create unambiguous message boundaries:

**Stage 1: Mandatory '1' Bit Append**
Every message receives exactly one '1' bit immediately after the final message bit. This creates an unambiguous end-of-message marker that prevents different messages from producing identical internal representations. Even if the original message ends with zeros, the appended '1' bit ensures the padded message differs from any message that naturally ends with those zeros followed by a '1'.

**Stage 2: Zero-Fill to Alignment Boundary**
After appending the mandatory '1' bit, zero bits are added until the total length reaches exactly 448 bits modulo 512. This specific alignment leaves exactly 64 bits remaining in the current 512-bit block for the length encoding. The calculation determines how many zero bits to add: `zero_bits = (448 - (message_length + 1)) % 512`.

**Stage 3: 64-Bit Length Encoding**
The final 64 bits of the padded message contain the original message length encoded as a big-endian unsigned integer. This length represents the bit count of the original message before any padding was applied. The 64-bit length field provides enough range to handle messages up to 2^64 - 1 bits (approximately 2.3 exabytes).

| Padding Stage | Purpose | Operation | Bit Count |
|---------------|---------|-----------|-----------|
| Bit Append | End marker | Add single '1' bit | 1 |
| Zero Fill | Block alignment | Add zeros to reach 448 mod 512 | Variable (0-511) |
| Length Encode | Original size | Add 64-bit big-endian length | 64 |

The algorithm handles edge cases systematically:

**Empty Message Handling**: An empty message (0 bits) receives the mandatory '1' bit, then 447 zero bits to reach the 448-bit boundary, followed by a 64-bit encoding of length zero. This produces exactly one 512-bit block.

**Block Boundary Cases**: When a message length plus the mandatory '1' bit would exceed the 448-bit boundary within the current block, padding extends into the next block. For example, a 448-bit message receives the '1' bit (reaching 449 bits), then 511 zero bits to reach the next 448-bit boundary, then the 64-bit length encoding.

**Maximum Length Handling**: The 64-bit length field theoretically supports messages up to 2^64 - 1 bits, but practical implementations often impose smaller limits to prevent resource exhaustion attacks.

Here's a concrete walk-through example for the message "abc":

1. Original message "abc" = 24 bits (0x616263 in hexadecimal)
2. Append mandatory '1' bit: 24 + 1 = 25 bits
3. Calculate zero padding: (448 - 25) % 512 = 423 zero bits
4. Total after zero padding: 25 + 423 = 448 bits
5. Append 64-bit length: 448 + 64 = 512 bits (exactly one block)
6. Length encoding: 24 as 64-bit big-endian = 0x0000000000000018

The resulting 512-bit block contains the original message bits, the '1' delimiter, 423 zero bits, and the big-endian length encoding - all precisely positioned according to the NIST specification.

### Endianness and Length Encoding

**Endianness** refers to the byte order used when storing multi-byte values in memory or during transmission. SHA-256 exclusively uses **big-endian** byte ordering throughout all operations, where the most significant byte appears first in memory layout.

Understanding endianness is crucial for SHA-256 implementation because the algorithm operates on 32-bit words that must be constructed from 8-bit bytes in the correct order. Incorrect endianness handling produces completely different hash values, making this one of the most critical correctness requirements.

> **Critical Security Insight**
> Endianness errors don't just produce wrong results - they can create security vulnerabilities. If different parts of an implementation use inconsistent byte ordering, attackers might exploit these inconsistencies to create hash collisions or bypass integrity checks.

**Big-Endian Byte Ordering in SHA-256**

SHA-256's big-endian requirement affects three key areas:

**32-Bit Word Construction**: When parsing message blocks into 32-bit words, bytes must be combined in big-endian order. For example, the byte sequence [0x61, 0x62, 0x63, 0x64] becomes the 32-bit word 0x61626364, not 0x64636261.

**Length Field Encoding**: The 64-bit message length uses big-endian encoding where the most significant byte appears first. A message length of 24 bits encodes as the 8-byte sequence [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18].

**Hash Output Generation**: The final eight 32-bit hash values are concatenated using big-endian byte order to produce the 256-bit output hash.

| Component | Byte Order | Example Input | Big-Endian Output |
|-----------|------------|---------------|-------------------|
| Message Words | Big-endian | [0x61, 0x62, 0x63, 0x64] | 0x61626364 |
| Length Field | Big-endian | 24 bits | 0x0000000000000018 |
| Hash Output | Big-endian | 0x12345678 | [0x12, 0x34, 0x56, 0x78] |

**64-Bit Length Encoding Specifications**

The length encoding follows specific rules that ensure consistent behavior across implementations:

**Bit Count Representation**: The 64-bit field contains the count of bits in the original message, not bytes. This bit-level precision allows SHA-256 to handle messages that aren't byte-aligned, though most practical implementations work with byte-aligned inputs.

**Overflow Handling**: For messages longer than 2^64 - 1 bits, the NIST specification doesn't define behavior. Most implementations either reject such inputs or use modular arithmetic, but this rarely occurs in practice since 2^64 bits represents approximately 2.3 exabytes.

**Zero-Length Encoding**: Empty messages encode their length as 64 zero bits (0x0000000000000000), following the same big-endian format as non-empty messages.

The length encoding algorithm proceeds as follows:

1. Calculate the bit count of the original message before padding
2. Convert the bit count to a 64-bit unsigned integer
3. Split the 64-bit integer into 8 bytes using big-endian ordering
4. Append these 8 bytes as the final 64 bits of the padded message

Consider encoding a 1000-bit message length:

1. Message bit count: 1000 (decimal)
2. 64-bit representation: 0x00000000000003E8
3. Big-endian byte sequence: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0xE8]
4. These bytes occupy the final 64 bits of the last message block

**Cross-Platform Endianness Considerations**

Different computer architectures use different native endianness. Intel x86/x64 processors use little-endian, while some ARM processors and network protocols use big-endian. SHA-256 implementations must explicitly handle endianness conversion regardless of the host architecture.

The implementation strategy involves explicit byte-order conversion functions that ensure consistent behavior across platforms. Never rely on native integer storage formats - always perform explicit big-endian encoding and decoding operations.

### Common Pitfalls

⚠️ **Pitfall: Bit vs Byte Confusion in Length Calculation**

The most frequent preprocessing error involves confusing bit counts with byte counts when calculating message length and padding requirements. The NIST specification operates at bit-level precision, but many developers instinctively work with bytes.

**Why This Breaks**: Using byte counts instead of bit counts produces incorrect padding lengths and wrong length encodings. For example, a 3-byte message has length 24 bits, not 3 bits. Encoding 3 instead of 24 in the length field creates an invalid hash that fails NIST test vector validation.

**Detection**: Hash outputs don't match expected test vectors, particularly for simple inputs like "abc" where the correct hash is well-known.

**Fix**: Always calculate and store message length in bits. Multiply byte counts by 8 when converting from byte-oriented input processing. Double-check length field encoding uses bit counts, not byte counts.

⚠️ **Pitfall: Off-By-One Errors in Padding Calculation**

The padding calculation `(448 - (message_length + 1)) % 512` is prone to off-by-one errors when developers forget the mandatory '1' bit or miscalculate the modular arithmetic.

**Why This Breaks**: Incorrect padding length causes misaligned blocks where the 64-bit length field doesn't appear in the expected position. This corrupts the block structure and produces wrong hash values.

**Common Mistakes**:
- Forgetting to add 1 for the mandatory '1' bit: `(448 - message_length) % 512`
- Using 512 instead of 448 in the calculation: `(512 - (message_length + 1)) % 512`
- Handling the modulus incorrectly for messages that require no zero padding

**Detection**: Message blocks have incorrect sizes or the length field appears in wrong positions. Hash outputs fail test vector validation.

**Fix**: Carefully implement the three-stage algorithm with explicit handling of each component. Test with edge cases like messages of length 447 bits (requires no zero padding) and 448 bits (requires maximum zero padding).

⚠️ **Pitfall: Endianness Confusion in Multi-Byte Values**

Mixing up big-endian and little-endian byte ordering when constructing 32-bit words from bytes or encoding the 64-bit length field is extremely common and produces completely wrong results.

**Why This Breaks**: SHA-256's security properties depend on precise bit manipulation. Incorrect endianness changes the values of all intermediate computations, producing hash outputs that bear no cryptographic relationship to the correct values.

**Detection Symptoms**:
- Hash outputs are wrong but internally consistent (wrong endianness applied consistently)
- Simple test cases like empty string produce completely incorrect results
- Implementation works on little-endian systems but fails on big-endian systems (or vice versa)

**Fix**: Implement explicit endianness conversion functions that work identically on all platforms. Never rely on native integer representations. Test on both little-endian and big-endian systems if possible.

⚠️ **Pitfall: String Encoding Assumptions**

Assuming message strings use ASCII encoding when they might use UTF-8, UTF-16, or other encodings leads to incorrect byte sequences being processed by the hash function.

**Why This Breaks**: Different string encodings produce different byte sequences for the same characters. For example, accented characters encode differently in UTF-8 vs UTF-16, leading to different hash values for the same logical message.

**Detection**: Hash values differ between systems with different default string encodings, or when processing international text.

**Fix**: Explicitly specify and document the expected string encoding (typically UTF-8 for modern systems). Convert strings to byte arrays using the specified encoding before hash processing begins.

⚠️ **Pitfall: Block Boundary Edge Cases**

Failing to correctly handle messages whose length plus padding would exactly fill one block or require exactly one additional block creates subtle bugs in boundary conditions.

**Critical Cases**:
- Message length 447 bits: needs 1 bit + 0 zero bits + 64 length bits = exactly 512 bits
- Message length 448 bits: needs 1 bit + 511 zero bits + 64 length bits = 1024 bits (two blocks)
- Empty message: needs 1 bit + 447 zero bits + 64 length bits = 512 bits

**Why This Breaks**: Incorrect boundary handling can create blocks with wrong sizes or place the length field in incorrect positions, corrupting the entire hash computation.

**Fix**: Implement comprehensive test cases covering all boundary conditions. Use modular arithmetic carefully and verify block sizes equal exactly 512 bits.

### Implementation Guidance

![End-to-End Data Flow Pipeline](./diagrams/data-flow-pipeline.svg)

![Message Preprocessing Steps](./diagrams/message-preprocessing-flow.svg)

The message preprocessing component serves as the entry point for SHA-256 computation, transforming arbitrary input messages into standardized 512-bit blocks ready for cryptographic processing. This implementation guidance provides complete working code for the preprocessing pipeline while leaving core learning opportunities for the student to implement.

**Technology Recommendations**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Input Handling | String → bytes with UTF-8 encoding | Multiple encoding support with explicit specification |
| Bit Manipulation | Python int with bit operations | Custom bit array class with overflow protection |
| Block Storage | List of integers | NumPy arrays for performance |
| Endianness | Manual bit shifting | struct.pack/unpack with explicit formats |

**Recommended File Structure**

The preprocessing component integrates into the overall SHA-256 project structure as the first stage of the processing pipeline:

```
sha256_project/
├── src/
│   ├── __init__.py
│   ├── sha256_main.py              ← Main hash interface
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── message_preprocessor.py ← Core preprocessing logic
│   │   ├── padding_utils.py        ← Padding algorithm helpers
│   │   └── endian_utils.py        ← Endianness conversion utilities
│   ├── constants.py                ← SHA-256 constants
│   └── types.py                    ← Data type definitions
├── tests/
│   ├── test_preprocessing.py       ← Preprocessing unit tests
│   └── test_vectors/               ← NIST test vectors
└── examples/
    └── preprocessing_demo.py       ← Usage examples
```

This structure separates concerns while maintaining clear dependencies. The preprocessing module depends only on constants and types, making it easy to test in isolation.

**Infrastructure Starter Code**

Here's complete working infrastructure code that handles the non-core aspects of preprocessing:

**File: `src/types.py`**
```python
from typing import List, NewType
from dataclasses import dataclass

# Type aliases for clarity and type safety
Word32 = NewType('Word32', int)  # 32-bit unsigned integer
MessageBlock = List[Word32]      # 16 32-bit words = 512 bits

# Constants for preprocessing
BLOCK_SIZE_BITS = 512
WORD_SIZE_BITS = 32
WORDS_PER_BLOCK = BLOCK_SIZE_BITS // WORD_SIZE_BITS  # 16 words
MESSAGE_LENGTH_BITS = 64
PADDING_BOUNDARY = BLOCK_SIZE_BITS - MESSAGE_LENGTH_BITS  # 448 bits

@dataclass
class PreprocessingResult:
    """Result of message preprocessing containing blocks ready for hashing."""
    blocks: List[MessageBlock]
    original_length_bits: int
    block_count: int
    
    def __post_init__(self):
        """Validate preprocessing result consistency."""
        assert len(self.blocks) > 0, "Must have at least one block"
        assert self.block_count == len(self.blocks), "Block count mismatch"
        for block in self.blocks:
            assert len(block) == WORDS_PER_BLOCK, f"Block must have {WORDS_PER_BLOCK} words"
```

**File: `src/preprocessing/endian_utils.py`**
```python
"""
Endianness utilities for SHA-256 preprocessing.
Provides platform-independent big-endian conversion functions.
"""

def bytes_to_word32_be(byte_sequence: bytes, offset: int = 0) -> int:
    """
    Convert 4 bytes to 32-bit word using big-endian byte order.
    
    Args:
        byte_sequence: Byte array containing data
        offset: Starting position in byte array
        
    Returns:
        32-bit word constructed from 4 bytes in big-endian order
        
    Raises:
        IndexError: If insufficient bytes available at offset
    """
    if offset + 4 > len(byte_sequence):
        raise IndexError(f"Need 4 bytes at offset {offset}, only {len(byte_sequence)} available")
    
    # Big-endian: most significant byte first
    return (byte_sequence[offset] << 24 |
            byte_sequence[offset + 1] << 16 |
            byte_sequence[offset + 2] << 8 |
            byte_sequence[offset + 3])

def word32_to_bytes_be(word: int) -> bytes:
    """
    Convert 32-bit word to 4 bytes using big-endian byte order.
    
    Args:
        word: 32-bit unsigned integer
        
    Returns:
        4-byte sequence in big-endian order
    """
    # Ensure word fits in 32 bits
    word = word & 0xFFFFFFFF
    
    return bytes([
        (word >> 24) & 0xFF,  # Most significant byte
        (word >> 16) & 0xFF,
        (word >> 8) & 0xFF,
        word & 0xFF           # Least significant byte
    ])

def length_to_64bit_be(bit_length: int) -> bytes:
    """
    Convert message bit length to 64-bit big-endian byte sequence.
    
    Args:
        bit_length: Original message length in bits
        
    Returns:
        8-byte sequence representing length in big-endian format
        
    Raises:
        ValueError: If bit_length exceeds 64-bit maximum
    """
    if bit_length < 0:
        raise ValueError("Message length cannot be negative")
    if bit_length >= (1 << 64):
        raise ValueError("Message length exceeds 64-bit maximum")
    
    # Convert to 8 bytes, most significant byte first
    return bytes([
        (bit_length >> 56) & 0xFF,
        (bit_length >> 48) & 0xFF,
        (bit_length >> 40) & 0xFF,
        (bit_length >> 32) & 0xFF,
        (bit_length >> 24) & 0xFF,
        (bit_length >> 16) & 0xFF,
        (bit_length >> 8) & 0xFF,
        bit_length & 0xFF
    ])

def validate_endianness():
    """
    Test endianness utilities with known values.
    Useful for verifying correct implementation.
    """
    # Test word conversion
    test_bytes = bytes([0x61, 0x62, 0x63, 0x64])  # "abcd"
    expected_word = 0x61626364
    actual_word = bytes_to_word32_be(test_bytes)
    assert actual_word == expected_word, f"Expected {expected_word:08x}, got {actual_word:08x}"
    
    # Test reverse conversion
    converted_bytes = word32_to_bytes_be(expected_word)
    assert converted_bytes == test_bytes, f"Round-trip conversion failed"
    
    # Test length encoding
    test_length = 24  # bits
    expected_length_bytes = bytes([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18])
    actual_length_bytes = length_to_64bit_be(test_length)
    assert actual_length_bytes == expected_length_bytes, "Length encoding failed"
    
    print("✓ All endianness tests passed")

if __name__ == "__main__":
    validate_endianness()
```

**Core Logic Skeleton**

The following skeleton provides the structure for the main preprocessing logic that students should implement:

**File: `src/preprocessing/message_preprocessor.py`**
```python
"""
SHA-256 Message Preprocessing Implementation

This module implements the NIST-specified preprocessing algorithm that converts
arbitrary-length messages into 512-bit blocks ready for hash computation.

The preprocessing follows three stages:
1. Append mandatory '1' bit after message
2. Pad with zeros to reach 448 bits mod 512
3. Append 64-bit big-endian message length
"""

from typing import List
from ..types import MessageBlock, PreprocessingResult, Word32, BLOCK_SIZE_BITS, PADDING_BOUNDARY
from .endian_utils import bytes_to_word32_be, length_to_64bit_be

class MessagePreprocessor:
    """Handles SHA-256 message preprocessing according to NIST specification."""
    
    def preprocess_message(self, message: str) -> PreprocessingResult:
        """
        Preprocess message string into 512-bit blocks for SHA-256 computation.
        
        Args:
            message: Input message string (UTF-8 encoded)
            
        Returns:
            PreprocessingResult containing padded message blocks
            
        Raises:
            ValueError: For invalid input messages
        """
        # TODO 1: Convert message string to bytes using UTF-8 encoding
        # TODO 2: Calculate original message length in bits (not bytes!)
        # TODO 3: Apply three-stage padding algorithm
        # TODO 4: Parse padded message into 512-bit blocks
        # TODO 5: Return PreprocessingResult with blocks and metadata
        # 
        # Hint: Use len(message.encode('utf-8')) * 8 for bit length
        # Hint: Handle empty string case explicitly
        pass
    
    def _apply_padding(self, message_bytes: bytes) -> bytes:
        """
        Apply SHA-256 padding algorithm to message bytes.
        
        Implements three-stage NIST padding:
        1. Append single '1' bit
        2. Append zeros to reach 448 mod 512 bits
        3. Append 64-bit big-endian length
        
        Args:
            message_bytes: Original message as byte array
            
        Returns:
            Padded message bytes aligned to 512-bit boundaries
        """
        # TODO 1: Calculate original message length in bits
        # TODO 2: Append mandatory '1' bit (0x80 byte if byte-aligned)
        # TODO 3: Calculate zero padding needed: (448 - (length + 1)) % 512 bits
        # TODO 4: Append calculated number of zero bits/bytes
        # TODO 5: Append 64-bit big-endian length using length_to_64bit_be()
        # TODO 6: Verify final length is multiple of 512 bits
        # 
        # Hint: For byte-aligned messages, append 0x80 for the '1' bit
        # Hint: Zero padding might be 0 bits if message length is exactly right
        # Hint: Use divmod() for cleaner modular arithmetic
        pass
    
    def _parse_blocks(self, padded_bytes: bytes) -> List[MessageBlock]:
        """
        Parse padded message bytes into array of 512-bit blocks.
        
        Each block contains 16 32-bit words in big-endian byte order.
        
        Args:
            padded_bytes: Padded message with length multiple of 64 bytes
            
        Returns:
            List of MessageBlock, each containing 16 32-bit words
            
        Raises:
            ValueError: If padded_bytes length not multiple of 64
        """
        # TODO 1: Validate input length is multiple of 64 bytes (512 bits)
        # TODO 2: Calculate number of 512-bit blocks
        # TODO 3: For each 64-byte chunk, create one MessageBlock
        # TODO 4: Within each block, parse 16 32-bit words using big-endian order
        # TODO 5: Use bytes_to_word32_be() for endianness-correct word construction
        # TODO 6: Return list of MessageBlock objects
        # 
        # Hint: range(0, len(padded_bytes), 64) iterates by 64-byte blocks
        # Hint: range(0, 64, 4) within each block iterates by 4-byte words
        pass
    
    def _calculate_zero_padding_bits(self, message_bit_length: int) -> int:
        """
        Calculate number of zero bits needed for SHA-256 padding.
        
        Formula: (448 - (message_length + 1)) % 512
        The +1 accounts for the mandatory '1' bit.
        
        Args:
            message_bit_length: Original message length in bits
            
        Returns:
            Number of zero bits to append (0-511)
        """
        # TODO 1: Add 1 to message length for mandatory '1' bit
        # TODO 2: Calculate how many bits needed to reach 448 mod 512
        # TODO 3: Use modular arithmetic: (448 - total_so_far) % 512
        # TODO 4: Return result (should be 0-511)
        # 
        # Hint: Python's % operator handles negative numbers correctly
        # Hint: Test with edge cases like 447-bit messages (needs 0 zero bits)
        pass

def preprocess_message(message: str) -> List[MessageBlock]:
    """
    Convenience function for message preprocessing.
    
    Args:
        message: Input message string
        
    Returns:
        List of 512-bit blocks ready for SHA-256 processing
    """
    preprocessor = MessagePreprocessor()
    result = preprocessor.preprocess_message(message)
    return result.blocks
```

**Language-Specific Implementation Hints**

**Python Bit Manipulation Tips:**
- Use `message.encode('utf-8')` for consistent string-to-bytes conversion
- Python integers have unlimited precision, so manual 32-bit masking isn't needed during preprocessing
- The `bytes()` constructor and byte array slicing handle endianness conversion efficiently
- Use `assert` statements liberally during development to catch padding errors early

**Working with Binary Data:**
- Python's `int.from_bytes(byte_data, 'big')` provides endianness-aware integer construction
- The `struct` module offers another approach: `struct.unpack('>I', four_bytes)[0]` for big-endian 32-bit words
- List comprehensions efficiently process multiple blocks: `[process_block(data[i:i+64]) for i in range(0, len(data), 64)]`

**Debugging Binary Operations:**
- Use `format(value, '08x')` to display 32-bit values as 8-digit hexadecimal
- Print message lengths in both bytes and bits to catch bit/byte confusion
- Log the first few bytes of padded messages in hex to verify padding correctness

**Milestone Checkpoint**

After implementing the preprocessing component, verify correct behavior with these checkpoints:

**Unit Test Validation:**
Run `python -m pytest tests/test_preprocessing.py -v` and expect all tests to pass, particularly:
- Empty string produces exactly one 512-bit block
- "abc" message produces expected padding pattern
- Block boundaries handle edge cases correctly
- Endianness conversion matches expected values

**Manual Verification Commands:**
```python
from src.preprocessing.message_preprocessor import preprocess_message

# Test empty string
blocks = preprocess_message("")
assert len(blocks) == 1, "Empty string should produce one block"
assert len(blocks[0]) == 16, "Block should contain 16 words"

# Test "abc" - should produce one block with known pattern
blocks = preprocess_message("abc")
print(f"'abc' produces {len(blocks)} block(s)")
print(f"First word: 0x{blocks[0][0]:08x}")  # Should start with 'abc' + padding
```

**Expected Behavior Signs:**
- **Correct Implementation**: NIST test vectors pass, padding calculations match manual verification, consistent results across runs
- **Wrong Implementation**: Off-by-one padding errors, endianness-swapped words, incorrect block counts for edge cases

**Common Debug Symptoms and Fixes:**

| Symptom | Likely Cause | Diagnosis | Fix |
|---------|-------------|-----------|-----|
| Empty string fails | Wrong padding calculation | Check if produces exactly 448 zeros + 64-bit length | Verify (448 - 1) % 512 = 447 zero bits |
| "abc" block wrong size | Bit/byte confusion | Print message length in bits vs bytes | Use `len(message.encode()) * 8` for bits |
| Words have wrong values | Endianness error | Compare first word with expected 0x61626380 | Use big-endian byte-to-word conversion |
| Too many/few blocks | Block boundary error | Print padded message length | Ensure final length divisible by 512 |

The preprocessing component serves as the foundation for the entire SHA-256 implementation. Correct preprocessing ensures that subsequent components receive properly formatted input, while preprocessing errors propagate through the entire hash computation and produce completely wrong results.


## Message Schedule Generation

> **Milestone(s):** Milestone 2 - Message Schedule

### Mental Model: Recipe Preparation

Think of message schedule generation as **preparing ingredients for a complex recipe**. Just as a master chef takes basic ingredients and transforms them into mise en place - carefully prepared components ready for cooking - the message scheduler takes raw 512-bit blocks and transforms them into precisely prepared 64-word schedules.

In cooking, you might start with basic ingredients like flour, eggs, and butter, but through careful preparation (sifting, tempering, clarifying), you create specialized ingredients that interact perfectly during cooking. Similarly, SHA-256 starts with 16 basic 32-bit words from each message block, but through careful mathematical preparation using **sigma functions** (specialized mixing operations), it creates 64 carefully crafted schedule words that will interact perfectly during the compression rounds.

The sigma functions act like specialized kitchen techniques - just as you might fold ingredients in a specific pattern to maintain texture, or whip in a particular direction to achieve proper aeration, the sigma functions combine rotation, shifting, and XOR operations in precise patterns to achieve the cryptographic properties needed for secure hashing. Each of the 64 schedule words is prepared with the exact mathematical "seasoning" needed for its role in the compression function.

### Word Expansion Algorithm

The **word expansion algorithm** transforms each 512-bit message block into a 64-word message schedule through a carefully designed recurrence relation. This expansion is fundamental to SHA-256's security, as it ensures that every bit of the original message influences multiple rounds of the compression function through the avalanche effect.

The algorithm begins with **initial word extraction**, where the 512-bit block is parsed into 16 initial 32-bit words using big-endian byte ordering. These initial words, labeled W₀ through W₁₅, directly represent the message content and serve as the foundation for schedule generation. The big-endian ordering ensures consistent interpretation across different computing platforms and matches the NIST specification exactly.

> **Critical Design Insight:** The choice to expand from 16 to 64 words isn't arbitrary - it ensures that each bit of the original message influences at least 4 compression rounds through the mathematical dependencies in the recurrence relation.

The **schedule recurrence relation** generates the remaining 48 words (W₁₆ through W₆₃) using the formula:

W_t = (σ₁(W_{t-2}) + W_{t-7} + σ₀(W_{t-15}) + W_{t-16}) mod 2³²

This recurrence creates complex mathematical dependencies where each new word depends on four previously computed words at specific intervals. The intervals (2, 7, 15, 16) were carefully chosen by NIST cryptographers to maximize diffusion while maintaining computational efficiency.

| Word Index Range | Generation Method | Dependencies | Purpose |
|------------------|-------------------|--------------|---------|
| W₀ to W₁₅ | Direct block parsing | Original message block | Preserve message content |
| W₁₆ to W₆₃ | Recurrence relation | W_{t-2}, W_{t-7}, W_{t-15}, W_{t-16} | Create diffusion and avalanche |
| All words | 32-bit masking | Previous computation results | Prevent overflow artifacts |

The **dependency pattern** in the recurrence relation ensures that small changes in the original message propagate throughout the entire schedule. When a single bit changes in the input message, it affects one of the initial 16 words, which then influences multiple words in the expanded schedule through the recursive dependencies. This creates the avalanche effect essential for cryptographic security.

**Modular arithmetic handling** requires careful attention to 32-bit overflow behavior. All intermediate computations must be masked to 32 bits using bitwise AND with 0xFFFFFFFF to ensure consistent behavior across different programming languages and platforms. This masking simulates the natural overflow behavior of 32-bit unsigned integers.

> **Decision: Recurrence Relation Parameters**
> - **Context**: Need to expand 16 words to 64 while maximizing cryptographic diffusion
> - **Options Considered**: Linear expansion, Fibonacci-like sequences, Complex multi-dependency relations
> - **Decision**: Use (t-2, t-7, t-15, t-16) dependency pattern with sigma function mixing
> - **Rationale**: Provides optimal balance between security (maximum diffusion), performance (reasonable computational cost), and implementation simplicity (straightforward recurrence)
> - **Consequences**: Each message bit influences multiple compression rounds while maintaining linear computational complexity

### Sigma Function Implementation

The **sigma functions** (σ₀ and σ₁) are specialized bitwise operations that provide the cryptographic mixing needed for secure schedule expansion. These functions combine right rotation and right shift operations with XOR to create complex bit-level transformations that resist cryptographic attacks.

**Lower-sigma-0 (σ₀) function** operates on 32-bit words using the formula:
σ₀(x) = ROTR⁷(x) ⊕ ROTR¹⁸(x) ⊕ SHR³(x)

This function takes an input word x and combines three different bit transformations: a 7-bit right rotation, an 18-bit right rotation, and a 3-bit right shift. The rotation operations preserve all bits while changing their positions, while the shift operation introduces zeros in the most significant positions. The XOR operation combines these transformations to create a result where each output bit depends on multiple input bit positions.

**Lower-sigma-1 (σ₁) function** uses a different parameter set:
σ₁(x) = ROTR¹⁷(x) ⊕ ROTR¹⁹(x) ⊕ SHR¹⁰(x)

The different rotation and shift amounts (17, 19, 10) ensure that σ₀ and σ₁ create distinct bit-mixing patterns. This diversity is crucial for preventing cryptographic weaknesses that could arise if both functions used identical transformation patterns.

| Function | Rotation 1 | Rotation 2 | Shift Amount | Usage Context |
|----------|------------|------------|--------------|---------------|
| σ₀(x) | ROTR⁷(x) | ROTR¹⁸(x) | SHR³(x) | Applied to W_{t-15} in recurrence |
| σ₁(x) | ROTR¹⁷(x) | ROTR¹⁹(x) | SHR¹⁰(x) | Applied to W_{t-2} in recurrence |

**Right rotation implementation** requires careful distinction from right shift operations. Right rotation (ROTR) moves bits circularly - bits shifted off the right end wrap around to the left end, preserving all information. This can be implemented using the formula:
ROTR^n(x) = (x >> n) | (x << (32-n))

The bitwise OR operation combines the shifted bits with the wrapped bits, and the (32-n) calculation ensures proper wrap-around for 32-bit words. All intermediate results must be masked to 32 bits to handle potential overflow in the left shift operation.

**Right shift implementation** (SHR) permanently discards bits shifted off the right end, introducing zeros on the left. This irreversible transformation adds cryptographic strength by preventing certain types of mathematical analysis. The implementation is straightforward: SHR^n(x) = x >> n.

**XOR combination logic** merges the three transformed values to produce the final sigma function result. XOR was chosen because it provides perfect bit diffusion (each output bit depends on multiple input bits) while maintaining computational efficiency. The XOR operation also has the mathematical property that A ⊕ B ⊕ C distributes evenly across all possible output values when the inputs have good statistical properties.

> **Decision: Sigma Function Parameters**
> - **Context**: Need bit-mixing functions that resist cryptographic analysis while remaining computationally efficient
> - **Options Considered**: Simple rotations only, Addition-based mixing, Complex nonlinear functions
> - **Decision**: Use rotation + shift + XOR with asymmetric parameters (7,18,3) and (17,19,10)
> - **Rationale**: Provides strong bit diffusion, prevents differential attacks, maintains linear computational complexity, and has undergone extensive cryptographic analysis
> - **Consequences**: Creates secure bit mixing while allowing efficient implementation in both software and hardware

### Common Pitfalls

⚠️ **Pitfall: Right Rotate vs Right Shift Confusion**

Many implementers mistakenly use right shift (>>) where right rotation is required. Right shift permanently discards bits by shifting zeros in from the left, while right rotation preserves all bits by wrapping them around. Using shift instead of rotation in the ROTR operations produces completely incorrect sigma function results that fail all test vectors.

The confusion often stems from programming languages providing shift operators (>> and <<) as built-ins while requiring manual implementation of rotation. To avoid this pitfall, implement rotation functions first and test them independently with known values before using them in sigma functions.

| Operation | Formula | Bit Preservation | Cryptographic Purpose |
|-----------|---------|------------------|----------------------|
| Right Shift | x >> n | Loses n bits | Irreversible mixing |
| Right Rotation | (x >> n) \| (x << (32-n)) | Preserves all bits | Reversible positioning |

⚠️ **Pitfall: 32-Bit Overflow in Schedule Computation**

The schedule recurrence relation involves adding four 32-bit values, which can produce results exceeding 32-bit range. Failing to mask intermediate results to 32 bits causes incorrect schedule generation that produces wrong final hash values. This pitfall is particularly common in languages like Python that support arbitrary-precision integers by default.

The issue manifests as hash values that don't match NIST test vectors, even when all other logic is correct. The solution is to mask every addition result using bitwise AND with 0xFFFFFFFF (4,294,967,295) to simulate 32-bit unsigned integer overflow behavior.

⚠️ **Pitfall: Incorrect Sigma Function Parameter Values**

The sigma functions use specific rotation and shift amounts that must match the NIST specification exactly. Common errors include swapping parameters between σ₀ and σ₁, using wrong rotation amounts (like 8 instead of 7), or applying shifts where rotations belong. Even single-bit errors in these parameters produce completely different hash results.

To avoid parameter errors, create named constants for all rotation and shift amounts, implement each sigma function in a separate function with clear documentation, and test each function individually against known intermediate values before integration.

| Common Error | Correct Value | Error Impact |
|--------------|---------------|--------------|
| σ₀ uses ROTR⁸ instead of ROTR⁷ | ROTR⁷(x) | Complete hash failure |
| σ₁ uses SHR³ instead of SHR¹⁰ | SHR¹⁰(x) | Wrong bit diffusion |
| Swapping σ₀ and σ₁ positions | σ₁ for t-2, σ₀ for t-15 | Incorrect dependencies |

⚠️ **Pitfall: Schedule Array Indexing Errors**

Off-by-one errors in schedule array indexing cause incorrect word dependencies in the recurrence relation. The most common mistake is using zero-based indexing inconsistently - for example, computing W[16] using W[1] instead of W[0] for the "t-16" dependency. This subtle error produces a completely different schedule that fails test vector validation.

The pitfall often occurs when translating mathematical notation (W₁₆ depends on W₀) to programming language arrays (W[16] depends on W[0]). Always use explicit index calculations like `t-16` rather than hardcoded values, and verify that W[16] correctly uses W[0], W[1], W[9], and W[14] as dependencies.

⚠️ **Pitfall: Endianness Errors in Initial Word Extraction**

The initial 16 words must be extracted from the 512-bit block using big-endian byte ordering, regardless of the host system's native endianness. Many implementers forget to handle endianness conversion, causing the initial schedule words to have incorrect values that propagate through the entire expansion process.

This pitfall produces hash results that are consistently wrong but internally consistent - the implementation generates the same wrong result repeatedly. The solution is to explicitly convert each 4-byte sequence to a big-endian 32-bit integer using appropriate byte ordering functions for your programming language.

### Implementation Guidance

![Message Schedule Generation Sequence](./diagrams/schedule-generation-sequence.svg)

The message schedule generation component bridges the gap between preprocessed message blocks and the compression function. This implementation guidance provides complete utilities for bit manipulation and a structured approach to schedule array construction.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Bit Operations | Built-in language operators with manual masking | Dedicated bitwise utility library |
| Array Storage | Standard language arrays/lists | Typed arrays with bounds checking |
| Constants | Module-level constants | Immutable configuration objects |
| Testing | Print-based debugging with known values | Comprehensive unit test suite |

#### Recommended File Structure

```
project-root/
  src/
    sha256/
      preprocessing.py          ← from previous milestone
      schedule.py               ← THIS COMPONENT
      compression.py            ← next milestone
      output.py                 ← final milestone
      constants.py              ← shared constants
      utils.py                  ← bit operation utilities
    test/
      test_schedule.py          ← schedule-specific tests
      test_vectors.py           ← NIST test vectors
  main.py                       ← command-line interface
```

#### Infrastructure Code: Bit Operation Utilities

```python
"""
Bit manipulation utilities for SHA-256 implementation.
These functions handle 32-bit arithmetic and rotation operations.
"""

def mask_32_bits(value: int) -> int:
    """
    Mask a value to 32 bits using bitwise AND.
    Simulates 32-bit unsigned integer overflow behavior.
    
    Args:
        value: Integer value that may exceed 32-bit range
    
    Returns:
        Value masked to 32-bit range (0 to 4,294,967,295)
    """
    return value & 0xFFFFFFFF

def right_rotate_32(value: int, amount: int) -> int:
    """
    Perform right rotation on a 32-bit value.
    Bits shifted off the right wrap around to the left.
    
    Args:
        value: 32-bit integer to rotate
        amount: Number of bit positions to rotate (0-31)
    
    Returns:
        Value rotated right by amount positions
    """
    # Ensure value is 32-bit and amount is valid
    value = mask_32_bits(value)
    amount = amount % 32
    
    # Perform rotation: shift right and wrap bits around
    return mask_32_bits((value >> amount) | (value << (32 - amount)))

def right_shift_32(value: int, amount: int) -> int:
    """
    Perform right shift on a 32-bit value.
    Bits shifted off the right are lost, zeros fill from left.
    
    Args:
        value: 32-bit integer to shift  
        amount: Number of bit positions to shift (0-31)
    
    Returns:
        Value shifted right by amount positions
    """
    return mask_32_bits(value) >> amount

# Test the bit operations with known values
if __name__ == "__main__":
    # Test right rotation
    test_value = 0x80000000  # Binary: 10000000000000000000000000000000
    rotated = right_rotate_32(test_value, 1)  # Should be 0x40000000
    print(f"Rotate test: 0x{test_value:08x} -> 0x{rotated:08x}")
    
    # Test right shift  
    shifted = right_shift_32(test_value, 1)  # Should be 0x40000000
    print(f"Shift test: 0x{test_value:08x} -> 0x{shifted:08x}")
    
    # Test 32-bit masking
    large_value = 0x100000000  # 33 bits
    masked = mask_32_bits(large_value)  # Should be 0x00000000
    print(f"Mask test: 0x{large_value:x} -> 0x{masked:08x}")
```

#### Infrastructure Code: SHA-256 Constants

```python
"""
SHA-256 constants from NIST specification.
All values are 32-bit unsigned integers in hexadecimal format.
"""

# Initial hash values - first 32 bits of fractional parts of square roots of first 8 primes
INITIAL_HASH_VALUES = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
]

# Round constants - first 32 bits of fractional parts of cube roots of first 64 primes  
ROUND_CONSTANTS = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]

# Block and word size constants
BLOCK_SIZE_BITS = 512
WORD_SIZE_BITS = 32
HASH_SIZE_BITS = 256
SCHEDULE_SIZE_WORDS = 64
HASH_SIZE_WORDS = 8
```

#### Core Logic Skeleton: Message Schedule Generation

```python
"""
Message Schedule Generation for SHA-256.
Expands 512-bit message blocks into 64-word schedules using sigma functions.
"""

from typing import List
from .utils import mask_32_bits, right_rotate_32, right_shift_32
from .constants import SCHEDULE_SIZE_WORDS

class MessageSchedule:
    """64-word message schedule generated from a single message block."""
    
    def __init__(self, words: List[int]):
        if len(words) != SCHEDULE_SIZE_WORDS:
            raise ValueError(f"Schedule must contain exactly {SCHEDULE_SIZE_WORDS} words")
        self.words = words
    
    def __getitem__(self, index: int) -> int:
        return self.words[index]
    
    def __len__(self) -> int:
        return len(self.words)

class MessageScheduler:
    """Generates message schedules from preprocessed message blocks."""
    
    def generate_schedule(self, block: 'MessageBlock') -> MessageSchedule:
        """
        Generate 64-word message schedule from 512-bit message block.
        
        Args:
            block: Message block containing 16 32-bit words
        
        Returns:
            Message schedule with 64 32-bit words
        """
        # TODO 1: Create schedule array with 64 empty slots
        # Hint: Initialize list with 64 zeros: [0] * SCHEDULE_SIZE_WORDS
        
        # TODO 2: Copy initial 16 words from message block to schedule positions 0-15
        # Hint: schedule[i] = block.words[i] for i in range(16)
        
        # TODO 3: Generate remaining 48 words (indices 16-63) using recurrence relation
        # Hint: for t in range(16, SCHEDULE_SIZE_WORDS):
        
        # TODO 4: For each word position t, compute W[t] using the formula:
        # W[t] = (sigma1(W[t-2]) + W[t-7] + sigma0(W[t-15]) + W[t-16]) mod 2^32
        
        # TODO 5: Apply sigma1 function to W[t-2]
        # Hint: sig1_result = self._sigma1(schedule[t-2])
        
        # TODO 6: Apply sigma0 function to W[t-15]  
        # Hint: sig0_result = self._sigma0(schedule[t-15])
        
        # TODO 7: Add all four terms and mask to 32 bits
        # Hint: schedule[t] = mask_32_bits(sig1_result + schedule[t-7] + sig0_result + schedule[t-16])
        
        # TODO 8: Return completed MessageSchedule object
        # Hint: return MessageSchedule(schedule)
        
        pass
    
    def _sigma0(self, x: int) -> int:
        """
        Lower-sigma-0 function: σ₀(x) = ROTR⁷(x) ⊕ ROTR¹⁸(x) ⊕ SHR³(x)
        
        Args:
            x: 32-bit input word
        
        Returns:
            Result of sigma0 transformation
        """
        # TODO 1: Apply 7-bit right rotation to input
        # Hint: rotr7 = right_rotate_32(x, 7)
        
        # TODO 2: Apply 18-bit right rotation to input
        # Hint: rotr18 = right_rotate_32(x, 18)
        
        # TODO 3: Apply 3-bit right shift to input
        # Hint: shr3 = right_shift_32(x, 3)
        
        # TODO 4: XOR all three results together
        # Hint: return rotr7 ^ rotr18 ^ shr3
        
        pass
    
    def _sigma1(self, x: int) -> int:
        """
        Lower-sigma-1 function: σ₁(x) = ROTR¹⁷(x) ⊕ ROTR¹⁹(x) ⊕ SHR¹⁰(x)
        
        Args:
            x: 32-bit input word
        
        Returns:
            Result of sigma1 transformation
        """
        # TODO 1: Apply 17-bit right rotation to input
        # Hint: rotr17 = right_rotate_32(x, 17)
        
        # TODO 2: Apply 19-bit right rotation to input  
        # Hint: rotr19 = right_rotate_32(x, 19)
        
        # TODO 3: Apply 10-bit right shift to input
        # Hint: shr10 = right_shift_32(x, 10)
        
        # TODO 4: XOR all three results together
        # Hint: return rotr17 ^ rotr19 ^ shr10
        
        pass

# Helper function for testing individual sigma functions
def test_sigma_functions():
    """Test sigma functions with known values."""
    scheduler = MessageScheduler()
    
    # Test sigma0 with known value
    test_value = 0x61626380  # First word from "abc" message
    sigma0_result = scheduler._sigma0(test_value)
    print(f"σ₀(0x{test_value:08x}) = 0x{sigma0_result:08x}")
    
    # Test sigma1 with known value  
    sigma1_result = scheduler._sigma1(test_value)
    print(f"σ₁(0x{test_value:08x}) = 0x{sigma1_result:08x}")

if __name__ == "__main__":
    test_sigma_functions()
```

#### Language-Specific Hints for Python

**Integer Handling**: Python's arbitrary-precision integers require explicit 32-bit masking. Always use `mask_32_bits()` after arithmetic operations to simulate 32-bit unsigned integer behavior. The expression `value & 0xFFFFFFFF` efficiently masks to 32 bits.

**List Initialization**: Create the 64-word schedule array using `schedule = [0] * SCHEDULE_SIZE_WORDS`. This creates a list with 64 zero-initialized elements that can be indexed and modified in place.

**Bit Operations**: Python's bitwise operators (`&`, `|`, `^`, `<<`, `>>`) work on arbitrary-precision integers. The shift operators (`<<` and `>>`) automatically handle bit positioning, but rotation requires manual implementation using the provided utility functions.

**Type Hints**: Use `List[int]` for schedule arrays and `int` for 32-bit words. The type hints help catch errors during development and make the code more maintainable.

**Testing Individual Functions**: Test each sigma function separately before integration. Use known input values and verify outputs match expected results to catch parameter errors early.

#### Milestone Checkpoint

After implementing the message schedule generation component, verify correct behavior using these checkpoints:

**Unit Test Execution**: Run `python -m pytest test/test_schedule.py -v` to execute schedule-specific tests. All tests should pass, particularly those verifying sigma function outputs and schedule array generation.

**Known Value Verification**: Test schedule generation with the "abc" message first block. The initial 16 words should be: `[0x61626380, 0x00000000, ..., 0x00000018]`. The 17th word (W[16]) should be computed as: `σ₁(0x00000000) + 0x00000000 + σ₀(0x61626380) + 0x61626380`.

**Sigma Function Testing**: Verify individual sigma functions with test values:
- `σ₀(0x61626380)` should produce a specific result based on the rotation/shift pattern
- `σ₁(0x00000000)` should always return 0 since all operations on zero produce zero
- Test with `0xFFFFFFFF` to verify all bits participate in transformations

**Schedule Array Length**: Confirm that generated schedules always contain exactly 64 words. Index out-of-bounds errors indicate problems with array initialization or loop ranges.

**Integration Readiness**: The `MessageSchedule` object should integrate cleanly with the compression function from the next milestone. Test that schedule words can be accessed by index and that the object properly encapsulates the 64-word array.


## Main Compression Function

> **Milestone(s):** Milestone 3 - Compression Function

### Mental Model: Industrial Mixer

Think of the SHA-256 compression function as a **sophisticated industrial mixer** that thoroughly combines ingredients through repeated blending cycles. Just as an industrial mixer uses multiple mixing blades, paddles, and agitators working in precise coordination to ensure uniform distribution of all ingredients, the compression function uses multiple **round functions** (Choice, Majority, and Sigma operations) working together across 64 mixing rounds to thoroughly blend the message schedule words with the current hash state.

The eight **working variables** (`a` through `h`) act like the mixing chamber contents that get continuously stirred, folded, and recombined. Each of the 64 rounds is like one complete mixing cycle where ingredients (schedule words and round constants) are added while the mixing blades (round functions) perform specific blending operations. The **Choice function** acts like a selective blade that picks ingredients based on conditions, while the **Majority function** operates like a consensus mechanism that blends based on what most ingredients agree on.

Just as industrial mixing ensures that no pocket of unmixed material remains and small changes in input ingredients affect the entire final mixture, the compression function's 64 rounds ensure that every input bit influences every output bit through the **avalanche effect**. The round constants act like specialized mixing agents added at precise intervals to prevent patterns and ensure thorough distribution.

After all 64 mixing cycles complete, the final working variable contents are combined with the original hash state (like adding the new mixture to an existing base), creating the updated hash state that serves as input for the next message block or the final hash output.

### Round Function Definitions

The compression function relies on four carefully designed **bitwise round functions** that provide the cryptographic strength of SHA-256. These functions use only basic bitwise operations (AND, OR, XOR, NOT, rotate) but combine them in ways that create complex, non-linear relationships between inputs and outputs.

#### Choice Function (Ch)

The **Choice function** implements a bitwise conditional selection mechanism where the first parameter (`x`) acts as a selector that chooses between the second parameter (`y`) and third parameter (`z`) on a bit-by-bit basis.

| Function | Formula | Bit-Level Behavior | Cryptographic Purpose |
|----------|---------|-------------------|----------------------|
| `Ch(x, y, z)` | `(x & y) ^ (~x & z)` | If x bit is 1, choose y bit; if x bit is 0, choose z bit | Creates non-linear dependency between three inputs |

The Choice function operates as a **bitwise multiplexer**. For each bit position, if the corresponding bit in `x` is 1, the output bit comes from `y`. If the corresponding bit in `x` is 0, the output bit comes from `z`. This creates a complex dependency where the output depends on all three inputs in a non-linear fashion, contributing to SHA-256's **preimage resistance**.

> **Design Insight: Non-Linear Mixing**
> The Choice function's conditional selection creates non-linear relationships that prevent attackers from easily predicting how input changes affect outputs. Unlike simple XOR operations, the Choice function's output cannot be easily inverted because it depends on three variables in a conditional manner.

#### Majority Function (Maj)

The **Majority function** implements a bitwise voting mechanism where the output bit is determined by the majority vote among the three input bits at each position.

| Function | Formula | Bit-Level Behavior | Cryptographic Purpose |
|----------|---------|-------------------|----------------------|
| `Maj(x, y, z)` | `(x & y) ^ (x & z) ^ (y & z)` | Output bit is 1 if at least 2 of 3 input bits are 1 | Provides balanced mixing and prevents bias |

The Majority function creates **democratic consensus** at the bit level. For each bit position, if two or more of the corresponding input bits are 1, the output bit is 1. If two or more input bits are 0, the output bit is 0. This ensures balanced mixing without favoring any particular input pattern.

The mathematical equivalence `(x & y) ^ (x & z) ^ (y & z)` efficiently computes the majority vote using only AND and XOR operations, making it suitable for both software and hardware implementation.

#### Upper-Sigma Functions (Σ₀ and Σ₁)

The **upper-Sigma functions** (capitalized Σ, distinct from the lower-sigma functions used in message scheduling) perform complex bit diffusion through combinations of **right rotation** and XOR operations.

| Function | Formula | Rotation/Shift Operations | Purpose |
|----------|---------|--------------------------|---------|
| `Σ₀(x)` | `ROTR(x,2) ⊕ ROTR(x,13) ⊕ ROTR(x,22)` | Right rotate by 2, 13, and 22 bits | Diffuses bits from working variable `a` |
| `Σ₁(x)` | `ROTR(x,6) ⊕ ROTR(x,11) ⊕ ROTR(x,25)` | Right rotate by 6, 11, and 25 bits | Diffuses bits from working variable `e` |

These functions create **bit diffusion** by rotating the input word by different amounts and XORing the results together. Each rotation moves bits to different positions, and the XOR combination ensures that each output bit depends on multiple input bit positions. The rotation amounts (2, 13, 22 for Σ₀ and 6, 11, 25 for Σ₁) are specifically chosen to maximize diffusion while avoiding patterns.

> **Architectural Decision: Rotation vs. Shift Operations**
> - **Context**: Need operations that spread bit influence while preserving information
> - **Options Considered**: 
>   1. Right shift operations (SHR) that introduce zeros
>   2. Right rotation operations (ROTR) that preserve all bits
>   3. Left rotation operations (ROTL)
> - **Decision**: Use right rotation operations exclusively
> - **Rationale**: Right rotation preserves all input bits while achieving maximum diffusion. Right shift would lose information by introducing zeros, weakening cryptographic properties. The NIST specification requires right rotation with specific amounts optimized through cryptanalysis.
> - **Consequences**: Requires careful implementation to distinguish rotation from shift operations, but provides optimal bit diffusion and cryptographic strength.

### 64-Round Compression Algorithm

The compression algorithm transforms the current hash state by processing it through 64 identical rounds, each incorporating one word from the message schedule and one round constant. This iterative process ensures thorough mixing of all input bits with the hash state.

#### Working Variables Initialization

The compression process begins by copying the current **hash state** into eight **working variables** that serve as the compression chamber contents.

| Working Variable | Source | Purpose | Update Pattern |
|------------------|---------|---------|----------------|
| `a` | `h0` (hash state word 0) | Primary accumulator, receives Σ₀ diffusion | Updated every round, becomes `h0` after all rounds |
| `b` | `h1` | Secondary accumulator | Receives previous `a` value each round |
| `c` | `h2` | Tertiary storage | Receives previous `b` value each round |
| `d` | `h3` | Choice function input | Receives previous `c` value each round |
| `e` | `h4` | Choice selector, receives Σ₁ diffusion | Updated with temp2 calculation each round |
| `f` | `h5` | Choice true branch | Receives previous `e` value each round |
| `g` | `h6` | Choice false branch | Receives previous `f` value each round |
| `h` | `h7` | Round accumulator | Receives previous `g` value each round |

This initialization creates a **shift register pattern** where values flow through the working variables during compression, ensuring that every bit of the original hash state influences the final result through multiple paths.

#### Single Round Operation

Each compression round follows an identical pattern that combines the current working variables with a message schedule word and round constant through the defined round functions.

The round algorithm proceeds through these precise steps:

1. **Calculate Temporary Value 1 (temp1)**: Combine the `h` working variable with Σ₁ diffusion of `e`, the Choice function output, the current schedule word, and round constant.
2. **Calculate Temporary Value 2 (temp2)**: Combine Σ₀ diffusion of `a` with the Majority function output.
3. **Shift Working Variables**: Move values through the working variable chain to implement the shift register pattern.
4. **Update Primary Variables**: Set new values for `a` and `e` using the calculated temporary values.
5. **Apply 32-bit Masking**: Ensure all arithmetic operations wrap correctly within 32-bit bounds.

The mathematical formulation for each round `i` (where `i` ranges from 0 to 63):

```
temp1 = h + Σ₁(e) + Ch(e,f,g) + K[i] + W[i]
temp2 = Σ₀(a) + Maj(a,b,c)
h = g
g = f
f = e
e = d + temp1
d = c
c = b
b = a
a = temp1 + temp2
```

All additions are performed **modulo 2³²** to maintain 32-bit word boundaries and prevent integer overflow.

#### Round Constant Integration

The **round constants** (`K[0]` through `K[63]`) are predetermined values derived from the cube roots of the first 64 prime numbers, providing **nothing-up-my-sleeve numbers** that prevent hidden backdoors or patterns.

| Constant Purpose | Generation Method | Cryptographic Role |
|------------------|-------------------|-------------------|
| Prevent patterns | Cube roots of primes 2, 3, 5, 7, 11, ... | Ensures each round has unique characteristics |
| Add entropy | Fractional parts × 2³² | Introduces mathematical constants with no special structure |
| Avoid symmetry | Different value per round | Prevents rounds from behaving identically |

These constants ensure that identical message schedule words processed in different rounds produce different results, breaking potential symmetries that could be exploited by attackers.

#### Hash State Update

After all 64 rounds complete, the final working variables are **added** (modulo 2³²) to the original hash state values to produce the updated hash state. This addition operation implements the **Davies-Meyer construction**, a proven method for building secure hash functions from block ciphers.

| Hash State Update | Formula | Purpose |
|------------------|---------|---------|
| `h0 = (h0 + a) mod 2³²` | Add final `a` to original `h0` | Incorporate round results into persistent state |
| `h1 = (h1 + b) mod 2³²` | Add final `b` to original `h1` | Preserve hash chain integrity |
| `h2 = (h2 + c) mod 2³²` | Add final `c` to original `h2` | Ensure feedforward security |
| `h3 = (h3 + d) mod 2³²` | Add final `d` to original `h3` | Maintain diffusion properties |
| `h4 = (h4 + e) mod 2³²` | Add final `e` to original `h4` | Continue hash chain |
| `h5 = (h5 + f) mod 2³²` | Add final `f` to original `h5` | Preserve state evolution |
| `h6 = (h6 + g) mod 2³²` | Add final `g` to original `h6` | Maintain cryptographic strength |
| `h7 = (h7 + h) mod 2³²` | Add final `h` to original `h7` | Complete hash state update |

This feedforward addition ensures that the compression function behaves as a **one-way function** where recovering the input from the output requires inverting both the 64-round compression and the final addition, making SHA-256 resistant to preimage attacks.

![Single Compression Round Flow](./diagrams/compression-round-flow.svg)

![Hash State Evolution](./diagrams/hash-state-machine.svg)

### Common Pitfalls

#### ⚠️ **Pitfall: Confusing Upper and Lower Sigma Functions**

Many implementers mistakenly interchange the **upper-Sigma functions** (Σ₀, Σ₁) used in compression with the **lower-sigma functions** (σ₀, σ₁) used in message schedule generation. These functions have completely different rotation parameters and serve different purposes.

**Why it's wrong**: The upper and lower sigma functions use different rotation amounts and apply to different data. Upper-Sigma functions operate on working variables during compression with rotations like (2,13,22) and (6,11,25). Lower-sigma functions operate on schedule words during expansion with rotations like (7,18,3) and (17,19,10).

**How to fix**: Create separate, clearly-named functions for each sigma type. Use distinctive names like `upper_sigma_0`, `upper_sigma_1` for compression functions and `lower_sigma_0`, `lower_sigma_1` for schedule functions. Never reuse sigma function implementations between compression and schedule generation.

**Detection method**: If test vectors fail for known inputs like "abc", compare your sigma function outputs against reference implementations. Incorrect sigma functions typically cause systematic failures across all test cases.

#### ⚠️ **Pitfall: Using Wrong Round Constant Values**

The 64 round constants must match the NIST specification exactly, but implementers often use incorrect values due to transcription errors, wrong precision, or confusion between hexadecimal and decimal representations.

**Why it's wrong**: Round constants are derived from cube roots of prime numbers and converted to 32-bit hexadecimal values. Even a single incorrect constant breaks the cryptographic properties and causes hash outputs to differ from the standard. The constants provide mathematical assurance against backdoors.

**How to fix**: Copy the round constants directly from the NIST specification or a trusted reference implementation. Use hexadecimal literals (0x428a2f98, 0x71374491, etc.) rather than attempting to recalculate from cube roots. Verify your constant array by computing SHA-256 of known test vectors.

**Detection method**: Incorrect round constants cause consistent failures across all inputs. Compare your `ROUND_CONSTANTS` array against the NIST specification. A single wrong constant typically causes completely different hash outputs.

#### ⚠️ **Pitfall: Forgetting 32-bit Arithmetic Masking**

SHA-256 requires all arithmetic operations to wrap at 32-bit boundaries (modulo 2³²), but many languages use 64-bit integers by default. Failing to mask intermediate results to 32 bits causes incorrect hash computation.

**Why it's wrong**: The compression algorithm depends on precise 32-bit arithmetic overflow behavior. When additions exceed 2³² - 1, they must wrap to zero and continue counting. Using 64-bit arithmetic changes the mathematical behavior and produces wrong results.

**How to fix**: Apply `& 0xFFFFFFFF` masking after every addition operation to ensure 32-bit wraparound. Create a helper function `mask_32_bits(value)` that enforces this constraint consistently. Mask the results of `temp1`, `temp2`, and all hash state updates.

**Detection method**: 64-bit arithmetic typically causes failures on longer inputs where wraparound occurs. Short inputs like "" or "abc" might work correctly by accident, but longer inputs fail. Check if your implementation produces correct results for multi-block messages.

#### ⚠️ **Pitfall: Incorrect Right Rotation Implementation**

The upper-Sigma functions require **right rotation** (ROTR) operations, but implementers often accidentally use **right shift** (SHR) operations that introduce zeros instead of preserving all bits.

**Why it's wrong**: Right rotation moves bits circularly, so bits shifted off the right end wrap around to the left end. Right shift discards bits shifted off the right end and introduces zeros from the left. This changes the diffusion properties and breaks cryptographic security.

**How to fix**: Implement right rotation as `((value >> amount) | (value << (32 - amount))) & 0xFFFFFFFF`. The right shift moves bits right, the left shift moves the wrapped bits to the left side, and the OR combines them. The mask ensures 32-bit boundaries.

**Detection method**: Wrong rotation typically causes systematic test failures. Verify rotation by testing `right_rotate_32(0x80000000, 1)` should return `0x40000000`, not `0x40000000` (which would indicate correct behavior) vs `0x00000000` (which would indicate incorrect right shift).

#### ⚠️ **Pitfall: Working Variable Update Order Dependencies**

The working variable updates must occur simultaneously (or with careful ordering) to avoid using updated values where original values are required. Implementers often update variables in sequence, causing each update to see modified values from previous updates.

**Why it's wrong**: The algorithm specification assumes all working variable reads occur before any writes within a single round. If you update `a` first, then use the new `a` value when calculating subsequent updates, the compression behavior changes and produces incorrect results.

**How to fix**: Use temporary variables to store all new values, then assign them to working variables after all calculations complete. Alternatively, update variables in reverse order (`h` through `a`) since the algorithm flows forward. Never update `a` or `e` before reading their original values for other calculations.

**Detection method**: Order dependency errors often cause subtle failures that vary by input. Test with simple inputs first, then verify that changing update order doesn't affect results (it should not if implemented correctly).

### Implementation Guidance

#### Technology Recommendations

| Component | Simple Option | Advanced Option | Rationale |
|-----------|---------------|-----------------|-----------|
| Bitwise Operations | Native language operators (`&`, `|`, `^`, `~`) | Optimized bit manipulation libraries | Native operators sufficient for learning; libraries add complexity |
| Arithmetic | Standard integer arithmetic with manual masking | Fixed-width integer types (uint32) | Manual masking teaches overflow behavior; fixed types hide important concepts |
| Function Organization | Single compression function with inline round functions | Separate modules for each round function type | Inline approach shows algorithm flow; modules aid testing and debugging |
| State Management | Simple variables for working state | Structured types for working variables | Simple variables match algorithm description; structures add abstraction layer |

#### Recommended File Organization

```
sha256/
├── sha256_core.py              ← main hash interface
├── compression/
│   ├── __init__.py
│   ├── compression_engine.py   ← CompressionEngine class (this component)
│   ├── round_functions.py      ← Ch, Maj, Σ₀, Σ₁ implementations
│   └── constants.py            ← ROUND_CONSTANTS and related values
├── preprocessing/
│   └── preprocessor.py         ← from previous milestone
├── schedule/
│   └── scheduler.py            ← from previous milestone
└── tests/
    ├── test_compression.py     ← unit tests for compression
    ├── test_round_functions.py ← tests for individual round functions
    └── test_vectors.py         ← NIST test vector validation
```

#### Infrastructure Starter Code

**File: `compression/constants.py`** - Complete round constants implementation:

```python
"""SHA-256 compression constants and initialization values."""

# Round constants K[0..63] - cube roots of first 64 primes
ROUND_CONSTANTS = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]

# Initial hash values H[0..7] - square roots of first 8 primes
INITIAL_HASH_VALUES = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
]

# Validation function for constants
def validate_constants():
    """Verify round constants and initial values are correct."""
    assert len(ROUND_CONSTANTS) == 64, f"Expected 64 round constants, got {len(ROUND_CONSTANTS)}"
    assert len(INITIAL_HASH_VALUES) == 8, f"Expected 8 initial hash values, got {len(INITIAL_HASH_VALUES)}"
    
    # Verify first few constants for basic correctness
    assert ROUND_CONSTANTS[0] == 0x428a2f98, "First round constant incorrect"
    assert INITIAL_HASH_VALUES[0] == 0x6a09e667, "First initial hash value incorrect"
    
    print("✓ Constants validation passed")

if __name__ == "__main__":
    validate_constants()
```

**File: `compression/round_functions.py`** - Complete round function implementations:

```python
"""SHA-256 round functions: Choice, Majority, and upper-Sigma functions."""

def mask_32_bits(value):
    """Ensure value fits in 32-bit unsigned integer range."""
    return value & 0xFFFFFFFF

def right_rotate_32(value, amount):
    """Perform 32-bit right rotation (ROTR operation)."""
    value = mask_32_bits(value)
    amount = amount % 32  # Handle rotation amounts >= 32
    return mask_32_bits((value >> amount) | (value << (32 - amount)))

def choice(x, y, z):
    """
    Choice function: Ch(x,y,z) = (x & y) ^ (~x & z)
    If x bit is 1, choose y bit; if x bit is 0, choose z bit.
    """
    return mask_32_bits((x & y) ^ (~x & z))

def majority(x, y, z):
    """
    Majority function: Maj(x,y,z) = (x & y) ^ (x & z) ^ (y & z)
    Output bit is 1 if majority of input bits are 1.
    """
    return mask_32_bits((x & y) ^ (x & z) ^ (y & z))

def upper_sigma_0(x):
    """
    Upper-Sigma-0: Σ₀(x) = ROTR(x,2) ⊕ ROTR(x,13) ⊕ ROTR(x,22)
    Used in compression rounds with working variable 'a'.
    """
    return mask_32_bits(
        right_rotate_32(x, 2) ^ 
        right_rotate_32(x, 13) ^ 
        right_rotate_32(x, 22)
    )

def upper_sigma_1(x):
    """
    Upper-Sigma-1: Σ₁(x) = ROTR(x,6) ⊕ ROTR(x,11) ⊕ ROTR(x,25)
    Used in compression rounds with working variable 'e'.
    """
    return mask_32_bits(
        right_rotate_32(x, 6) ^ 
        right_rotate_32(x, 11) ^ 
        right_rotate_32(x, 25)
    )

# Test functions for validation
def test_round_functions():
    """Basic validation tests for round functions."""
    # Test rotation
    assert right_rotate_32(0x80000000, 1) == 0x40000000, "Right rotation failed"
    
    # Test choice function with known values
    assert choice(0xFFFFFFFF, 0x12345678, 0x87654321) == 0x12345678, "Choice function failed"
    assert choice(0x00000000, 0x12345678, 0x87654321) == 0x87654321, "Choice function failed"
    
    # Test majority function
    assert majority(0xFFFFFFFF, 0xFFFFFFFF, 0x00000000) == 0xFFFFFFFF, "Majority function failed"
    assert majority(0xFFFFFFFF, 0x00000000, 0x00000000) == 0x00000000, "Majority function failed"
    
    print("✓ Round functions validation passed")

if __name__ == "__main__":
    test_round_functions()
```

#### Core Logic Skeleton Code

**File: `compression/compression_engine.py`** - Core compression implementation for learner to complete:

```python
"""SHA-256 compression engine - core learning component."""

from typing import List
from .round_functions import choice, majority, upper_sigma_0, upper_sigma_1, mask_32_bits
from .constants import ROUND_CONSTANTS, INITIAL_HASH_VALUES

class WorkingVariables:
    """Eight working variables for compression rounds."""
    def __init__(self, hash_state: List[int]):
        # TODO 1: Initialize working variables a-h from hash_state[0-7]
        # Hint: self.a = hash_state[0], self.b = hash_state[1], etc.
        pass
    
    def to_list(self) -> List[int]:
        """Convert working variables back to list format."""
        # TODO 2: Return list [a, b, c, d, e, f, g, h]
        pass

class CompressionEngine:
    """Handles 64-round compression of message schedule with hash state."""
    
    def __init__(self):
        self.hash_state = INITIAL_HASH_VALUES.copy()
    
    def compress_block(self, message_schedule: List[int]) -> List[int]:
        """
        Compress one message block through 64 rounds.
        
        Args:
            message_schedule: List of 64 32-bit words from message scheduler
            
        Returns:
            Updated hash state after compression
        """
        # TODO 3: Validate message_schedule has exactly 64 words
        # Hint: assert len(message_schedule) == 64
        
        # TODO 4: Initialize working variables from current hash state
        # Hint: working_vars = WorkingVariables(self.hash_state)
        
        # TODO 5: Execute 64 compression rounds
        # Hint: for round_num in range(64): self._execute_round(...)
        
        # TODO 6: Update hash state with final working variables (Davies-Meyer)
        # Hint: self.hash_state[i] = mask_32_bits(self.hash_state[i] + working_vars.to_list()[i])
        
        return self.hash_state.copy()
    
    def _execute_round(self, working_vars: WorkingVariables, schedule_word: int, round_constant: int):
        """
        Execute one compression round updating working variables.
        
        This is the core cryptographic operation - implement carefully!
        """
        # TODO 7: Calculate temp1 = h + Σ₁(e) + Ch(e,f,g) + round_constant + schedule_word
        # Hint: Use upper_sigma_1() and choice() functions, mask result with mask_32_bits()
        
        # TODO 8: Calculate temp2 = Σ₀(a) + Maj(a,b,c)  
        # Hint: Use upper_sigma_0() and majority() functions, mask result
        
        # TODO 9: Shift working variables: h=g, g=f, f=e, d=c, c=b, b=a
        # Hint: Update in reverse order to avoid overwriting values you still need
        
        # TODO 10: Update e = d + temp1 (with 32-bit masking)
        # TODO 11: Update a = temp1 + temp2 (with 32-bit masking)
        
        pass
    
    def get_hash_state(self) -> List[int]:
        """Get current hash state for output formatting."""
        return self.hash_state.copy()
    
    def reset(self):
        """Reset hash state to initial values for new message."""
        # TODO 12: Reset self.hash_state to copy of INITIAL_HASH_VALUES
        pass

# Test harness for milestone validation
def test_compression_engine():
    """Test compression engine with known values."""
    engine = CompressionEngine()
    
    # Test initial state
    initial_state = engine.get_hash_state()
    assert initial_state == INITIAL_HASH_VALUES, "Initial state incorrect"
    
    # TODO: Add tests with known message schedules and expected outputs
    # This requires completing the implementation first
    
    print("✓ Compression engine basic validation passed")

if __name__ == "__main__":
    test_compression_engine()
```

#### Language-Specific Hints

**Python Implementation Details:**
- Use `& 0xFFFFFFFF` for 32-bit masking since Python integers are unlimited precision
- Right rotation: `((x >> n) | (x << (32-n))) & 0xFFFFFFFF` handles bit wraparound correctly  
- List slicing `hash_state[:]` creates copies to avoid accidental mutation
- Use `assert` statements liberally during development to catch errors early
- The `struct` module can help with endianness: `struct.pack('>I', word)` for big-endian 32-bit

**Debugging Utilities:**
```python
def debug_working_vars(working_vars, round_num):
    """Print working variables for debugging."""
    print(f"Round {round_num:2d}: a={working_vars.a:08x} b={working_vars.b:08x} "
          f"c={working_vars.c:08x} d={working_vars.d:08x} e={working_vars.e:08x} "
          f"f={working_vars.f:08x} g={working_vars.g:08x} h={working_vars.h:08x}")

def debug_round_calculation(temp1, temp2, schedule_word, round_constant, round_num):
    """Print round calculations for debugging."""
    print(f"Round {round_num:2d}: temp1={temp1:08x} temp2={temp2:08x} "
          f"W[{round_num}]={schedule_word:08x} K[{round_num}]={round_constant:08x}")
```

#### Milestone Checkpoint

After implementing the compression engine, verify correct behavior:

**Unit Test Execution:**
```bash
python -m compression.compression_engine
python -m compression.round_functions  
python -m compression.constants
```

**Expected Output:**
```
✓ Round functions validation passed
✓ Constants validation passed  
✓ Compression engine basic validation passed
```

**Integration Test with Previous Milestones:**
Create a simple end-to-end test combining preprocessing, scheduling, and compression:

```python
# Test with empty string (should eventually produce SHA-256 of "")
from preprocessing.preprocessor import preprocess_message
from schedule.scheduler import generate_schedule
from compression.compression_engine import CompressionEngine

message = ""
blocks = preprocess_message(message)
engine = CompressionEngine()

for block in blocks:
    schedule = generate_schedule(block)
    engine.compress_block(schedule)

final_hash = engine.get_hash_state()
print(f"Hash state: {[f'{word:08x}' for word in final_hash]}")
```

**Signs of Correct Implementation:**
- Round function tests pass with expected bit patterns
- Working variables shift correctly through rounds  
- Hash state updates show cumulative changes
- No assertion failures during block processing
- Intermediate values match reference implementation patterns

**Common Debugging Indicators:**

| Symptom | Likely Cause | Check This |
|---------|--------------|------------|
| All zeros output | Forgot to update hash state after compression | Verify Davies-Meyer addition in compress_block |
| Same output for different inputs | Working variables not updating | Check _execute_round implementation |
| Assertion failures on schedule length | Wrong schedule size expectation | Verify message scheduler produces 64 words |
| Integer overflow errors | Missing 32-bit masking | Add mask_32_bits() to all arithmetic |
| Pattern in output bits | Wrong round constants | Compare ROUND_CONSTANTS with NIST specification |


## Final Hash Output Generation

> **Milestone(s):** Milestone 4 - Final Hash Output

### Mental Model: Photo Development

Think of final hash output generation as **developing a photograph in a darkroom**. After all the complex chemical processes have occurred in the developer solution (the compression rounds), you now need to properly fix, wash, and present the final photograph. The hash state at this point contains all the information needed for your final image, but it exists in an internal format that needs careful conversion to produce a viewable result. Just as a photographer must follow precise steps to avoid ruining the developed image during the final processing stages, SHA-256 requires exact procedures to convert the internal hash state into the standard 64-character hexadecimal representation that applications expect.

The finalization process is deceptively simple but critically important - a single mistake in byte ordering, formatting, or validation can produce a completely incorrect hash that appears valid but fails to match any standard implementation. This is the stage where your careful work throughout the previous milestones either comes together perfectly or reveals subtle bugs that must be identified and corrected through systematic testing against known reference values.

### Hash Finalization Process

The **hash finalization process** transforms the internal hash state from the compression function into the final 256-bit SHA-256 output. After processing all message blocks through the compression rounds, the hash state contains eight 32-bit words that represent the final hash values. These values must be concatenated in the correct order and converted to the appropriate output format while maintaining the big-endian byte ordering required by the SHA-256 specification.

The finalization process begins immediately after the last message block has been processed through the compression function. At this point, the `HashState` contains the eight final 32-bit hash values h₀ through h₇, each representing a 32-bit portion of the final 256-bit hash. These values have been accumulated and updated throughout the processing of all message blocks, with each block contributing to the final result through the davies-meyer construction that adds the compressed working variables back to the original hash state.

The concatenation process must preserve the exact bit ordering specified in the NIST standard. The final hash is formed by concatenating the eight hash values in order: h₀ || h₁ || h₂ || h₃ || h₄ || h₅ || h₆ || h₇, where || represents concatenation. Each 32-bit hash value must be converted to its big-endian byte representation before concatenation, ensuring that the most significant byte of each word appears first in the final output sequence.

> **Decision: Internal Hash State Representation**
> - **Context**: The compression function produces eight 32-bit values that must be converted to final output
> - **Options Considered**: Store as byte array, store as 32-bit integers, store as single 256-bit value
> - **Decision**: Store as list of eight 32-bit integers until final conversion
> - **Rationale**: Matches the mathematical specification, simplifies debugging by allowing inspection of individual hash values, enables efficient processing during compression rounds
> - **Consequences**: Requires explicit conversion to bytes during finalization, enables clear separation between internal processing and output formatting

| Hash Finalization Step | Input | Process | Output | Critical Requirements |
|------------------------|-------|---------|--------|----------------------|
| Hash State Extraction | Eight 32-bit integers h₀-h₇ | Extract final values from compression engine | List of hash values | Values must reflect all processed blocks |
| Endianness Conversion | 32-bit integers | Convert each to 4-byte big-endian sequence | Byte arrays per word | Most significant byte first |
| Hash Concatenation | Eight 4-byte sequences | Concatenate in order h₀ to h₇ | 32-byte array | Preserve exact byte ordering |
| Format Conversion | 32-byte binary array | Convert to target format | String or bytes | Match expected output type |
| Validation Check | Formatted output | Compare against known test vectors | Pass/fail result | Ensure implementation correctness |

The hash state extraction step retrieves the final hash values from the `CompressionEngine` after all message blocks have been processed. These values represent the cumulative result of all compression rounds across all blocks, with each value being a 32-bit unsigned integer that has been masked to prevent overflow during the arithmetic operations performed throughout the compression process.

Endianness conversion is a critical step that frequently causes implementation errors. Each 32-bit hash value must be converted to its big-endian byte representation, meaning the most significant byte (bits 24-31) becomes the first byte in the sequence, followed by bits 16-23, then bits 8-15, and finally bits 0-7. This conversion must be performed consistently for all eight hash values to maintain the standard SHA-256 output format.

The concatenation step combines the eight 4-byte sequences into a single 32-byte array representing the complete 256-bit hash. The order is critical: h₀ occupies bytes 0-3, h₁ occupies bytes 4-7, continuing through h₇ which occupies bytes 28-31. This 32-byte array represents the canonical binary form of the SHA-256 hash that can be converted to various output formats.

### Output Format Options

SHA-256 implementations typically support multiple output formats to accommodate different application requirements. The two most common formats are **hexadecimal string representation** and **binary byte array representation**. Each format serves specific use cases and requires different conversion processes from the internal 32-byte hash array.

The **hexadecimal string format** is the most widely recognized SHA-256 output format, producing a 64-character lowercase hexadecimal string. This format converts each byte of the 32-byte hash array into two hexadecimal characters, with each nibble (4 bits) represented by a single hex digit (0-9, a-f). The hexadecimal format is human-readable, easily transmitted as text, and commonly used in applications, debugging output, and cryptographic protocols that require string-based hash representation.

> The hexadecimal output format is the de facto standard for SHA-256 because it provides a compact, readable representation that preserves all 256 bits of information while being safe to transmit through text-based protocols and easy for humans to compare visually.

The conversion process for hexadecimal output requires careful attention to formatting consistency. Each byte must produce exactly two hexadecimal characters, with leading zeros preserved for bytes with values less than 16. The choice between uppercase and lowercase hex characters should be consistent throughout the implementation, with lowercase being the conventional choice for SHA-256 implementations to match common tools and specifications.

| Output Format | Size | Encoding | Use Cases | Advantages | Disadvantages |
|---------------|------|----------|-----------|------------|---------------|
| Hexadecimal String | 64 characters | ASCII hex digits | Web APIs, debugging, file integrity | Human readable, text-safe, compact | Larger than binary, requires parsing |
| Binary Byte Array | 32 bytes | Raw binary | Cryptographic operations, storage | Compact, efficient processing | Not human readable, transmission issues |
| Base64 String | 44 characters | Base64 encoding | JSON APIs, URLs | Text-safe, compact | Less familiar, requires encoding knowledge |
| Uppercase Hex | 64 characters | ASCII uppercase | Legacy compatibility | Matches some standards | Less common for SHA-256 |

The **binary byte array format** preserves the hash in its most compact form as a 32-byte array of raw binary data. This format is ideal for applications that perform further cryptographic operations on the hash value, store hashes in binary database fields, or require maximum storage efficiency. The binary format requires no conversion beyond the endianness handling already performed during finalization, making it the most computationally efficient output option.

Applications often need both output formats, requiring implementations to support format selection through method parameters or separate interface methods. The internal 32-byte array serves as the canonical representation from which both hex strings and binary arrays can be generated without loss of precision or additional computation.

> **Decision: Default Output Format**
> - **Context**: Applications expect different hash output formats for different use cases
> - **Options Considered**: Always return hex string, always return binary, support both formats
> - **Decision**: Primary interface returns hex string, secondary method returns binary
> - **Rationale**: Hex strings match user expectations and common usage patterns, while binary format supports advanced use cases
> - **Consequences**: Slight performance overhead for applications needing binary format, but improved usability for common cases

The format conversion implementations must handle edge cases correctly, particularly for bytes with special values. Zero bytes must produce "00" in hex format rather than "0", and all bytes must be converted consistently regardless of their numeric value. The conversion process should be efficient and avoid string concatenation in loops, instead using appropriately sized buffers or built-in formatting functions.

### NIST Test Vector Validation

**NIST test vector validation** provides the definitive verification that a SHA-256 implementation produces correct results according to the official cryptographic standard. The National Institute of Standards and Technology provides official test vectors that specify exact input values and their corresponding expected SHA-256 output hashes. These test vectors serve as the authoritative reference for implementation correctness and must be used to validate every SHA-256 implementation.

The most fundamental test vectors that every implementation must pass are the **empty string test** and the **"abc" string test**. The empty string (zero-length input) must produce the hash value "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855". This test validates that the padding algorithm correctly handles the minimum case where only padding bits and length encoding are processed. The "abc" input must produce "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", validating basic message processing with minimal content.

| Test Vector | Input | Expected Output | Validation Focus |
|-------------|-------|----------------|------------------|
| Empty String | "" | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 | Padding algorithm, minimal processing |
| Single Letter | "a" | ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb | Single character handling |
| Short String | "abc" | ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad | Basic message processing |
| Longer String | "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq" | 248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1 | Multi-word message processing |
| Boundary Case | 55-byte message | Various expected values | Padding boundary conditions |
| Multi-Block | 64+ byte message | Various expected values | Multiple block processing |

Additional test vectors should include boundary cases that stress the padding algorithm, such as messages that are exactly 55 bytes long (which require minimum padding) and messages that are exactly 56 bytes long (which require a full additional block for padding). These boundary cases validate that the padding calculation correctly determines when an additional 512-bit block is needed to accommodate the length encoding.

The validation process must compare both the final hash output and, ideally, intermediate values at key stages of the algorithm. Comparing intermediate hash states after processing each block can help identify exactly where an implementation diverges from the expected behavior, significantly reducing debugging time when test vectors fail.

> Critical validation insight: If an implementation fails the empty string test, the error is almost certainly in the padding algorithm or initial hash values. If it passes empty string but fails "abc", the error is likely in the message schedule generation or compression function. This systematic approach to test vector analysis accelerates debugging.

Test vector validation should be automated and run as part of the development process, not just at the end of implementation. Each milestone should include validation checkpoints using appropriate test vectors to catch errors early rather than debugging a complete but incorrect implementation. The validation framework should provide detailed error messages indicating which test failed and, if possible, at what stage the divergence occurred.

Advanced implementations may include the full NIST test suite, which covers hundreds of test cases with various message lengths, patterns, and edge cases. However, for learning purposes, validating against the basic test vectors (empty string, "abc", and a few boundary cases) provides sufficient confidence in implementation correctness while keeping the testing manageable.

⚠️ **Pitfall: Byte vs Character Confusion**
When processing string inputs like "abc", implementations often confuse byte length with character length. The string "abc" should be processed as three bytes (0x61, 0x62, 0x63), not as three Unicode code points. For ASCII text, these are equivalent, but the distinction becomes critical with non-ASCII inputs. Always convert string inputs to byte arrays using UTF-8 encoding and process the resulting byte sequence.

⚠️ **Pitfall: Case Sensitivity in Hex Output**
Test vectors use lowercase hexadecimal representation, but some implementations produce uppercase hex output, causing validation failures despite correct computation. Ensure your hex formatting produces lowercase characters (0-9, a-f) to match the standard test vector format. This is a formatting issue, not a computational error, but it will cause test failures if not addressed.

### Implementation Guidance

The final hash output generation represents the culmination of the SHA-256 implementation, requiring careful attention to format conversion, validation testing, and error handling. This guidance provides complete implementation patterns for output formatting, comprehensive test vector validation, and debugging utilities to ensure your implementation produces correct results.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Format Conversion | String concatenation with built-in hex formatting | Byte buffer operations with pre-allocated arrays |
| Test Framework | Manual test vector checking with assertions | Automated test suite with detailed failure reporting |
| Validation | Basic pass/fail test result | Intermediate value logging for debugging |
| Output Interface | Single method returning hex string | Multiple methods supporting different formats |

#### Recommended File Organization

The output formatting and validation components should be organized to separate concerns while maintaining clear interfaces to the core hash computation:

```
sha256/
  sha256_hash.py           ← main hash interface (hash_message, hash_bytes)
  output_formatter.py      ← OutputFormatter class for format conversion
  test_vectors.py          ← NIST test vector constants and validation
  validation.py            ← test execution and debugging utilities
  tests/
    test_output.py         ← unit tests for output formatting
    test_vectors.py        ← NIST test vector validation tests
    test_integration.py    ← end-to-end integration tests
```

#### Infrastructure Starter Code

Here's the complete `OutputFormatter` class that handles all format conversion requirements:

```python
"""
SHA-256 Output Formatting Module

Handles conversion of internal hash state to various output formats
including hexadecimal strings and binary byte arrays.
"""

from typing import List, Union
import struct

class OutputFormatter:
    """Converts SHA-256 hash state to standard output formats."""
    
    def __init__(self):
        pass
    
    def format_hash_hex(self, hash_state: List[int]) -> str:
        """
        Convert hash state to 64-character lowercase hexadecimal string.
        
        Args:
            hash_state: List of 8 32-bit integers representing final hash values h0-h7
            
        Returns:
            64-character lowercase hexadecimal string representation
            
        Raises:
            ValueError: If hash_state doesn't contain exactly 8 values
        """
        if len(hash_state) != 8:
            raise ValueError(f"Hash state must contain exactly 8 values, got {len(hash_state)}")
        
        # Convert each 32-bit word to 4-byte big-endian sequence, then to hex
        hex_parts = []
        for word in hash_state:
            # Ensure word is 32-bit unsigned integer
            word = word & 0xFFFFFFFF
            # Convert to 4-byte big-endian bytes
            word_bytes = struct.pack('>I', word)
            # Convert bytes to lowercase hex (2 chars per byte)
            hex_part = word_bytes.hex()
            hex_parts.append(hex_part)
        
        return ''.join(hex_parts)
    
    def format_hash_bytes(self, hash_state: List[int]) -> bytes:
        """
        Convert hash state to 32-byte binary array.
        
        Args:
            hash_state: List of 8 32-bit integers representing final hash values h0-h7
            
        Returns:
            32-byte binary representation of the hash
            
        Raises:
            ValueError: If hash_state doesn't contain exactly 8 values
        """
        if len(hash_state) != 8:
            raise ValueError(f"Hash state must contain exactly 8 values, got {len(hash_state)}")
        
        # Convert each 32-bit word to 4-byte big-endian sequence
        byte_parts = []
        for word in hash_state:
            # Ensure word is 32-bit unsigned integer
            word = word & 0xFFFFFFFF
            # Convert to 4-byte big-endian bytes
            word_bytes = struct.pack('>I', word)
            byte_parts.append(word_bytes)
        
        return b''.join(byte_parts)
    
    def validate_hash_state(self, hash_state: List[int]) -> bool:
        """
        Validate that hash state contains valid 32-bit unsigned integers.
        
        Args:
            hash_state: Hash state to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(hash_state, list):
            return False
        
        if len(hash_state) != 8:
            return False
        
        for word in hash_state:
            if not isinstance(word, int):
                return False
            if word < 0 or word > 0xFFFFFFFF:
                return False
        
        return True

# Test vector constants for validation
NIST_TEST_VECTORS = {
    "empty_string": {
        "input": "",
        "expected": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    "single_a": {
        "input": "a",
        "expected": "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"
    },
    "abc": {
        "input": "abc",
        "expected": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    },
    "long_string": {
        "input": "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        "expected": "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    }
}

class TestVectorValidator:
    """Validates SHA-256 implementation against NIST test vectors."""
    
    def __init__(self, hash_function):
        """
        Initialize validator with hash function to test.
        
        Args:
            hash_function: Function that takes string input and returns hex hash
        """
        self.hash_function = hash_function
        self.formatter = OutputFormatter()
    
    def run_all_tests(self) -> bool:
        """
        Run all NIST test vectors and return overall pass/fail.
        
        Returns:
            True if all tests pass, False if any test fails
        """
        all_passed = True
        
        for test_name, test_data in NIST_TEST_VECTORS.items():
            try:
                result = self.hash_function(test_data["input"])
                expected = test_data["expected"]
                
                if result == expected:
                    print(f"✓ {test_name}: PASSED")
                else:
                    print(f"✗ {test_name}: FAILED")
                    print(f"  Input: '{test_data['input']}'")
                    print(f"  Expected: {expected}")
                    print(f"  Got:      {result}")
                    all_passed = False
                    
            except Exception as e:
                print(f"✗ {test_name}: ERROR - {str(e)}")
                all_passed = False
        
        return all_passed
    
    def debug_test_failure(self, test_name: str, debug_callback=None):
        """
        Run single test with detailed debugging output.
        
        Args:
            test_name: Name of test vector to debug
            debug_callback: Optional function to call for intermediate values
        """
        if test_name not in NIST_TEST_VECTORS:
            print(f"Unknown test vector: {test_name}")
            return
        
        test_data = NIST_TEST_VECTORS[test_name]
        print(f"\nDebugging test: {test_name}")
        print(f"Input: '{test_data['input']}'")
        print(f"Input length: {len(test_data['input'])} characters")
        print(f"Expected: {test_data['expected']}")
        
        try:
            if debug_callback:
                debug_callback(test_data["input"])
            
            result = self.hash_function(test_data["input"])
            print(f"Actual:   {result}")
            print(f"Match: {'YES' if result == test_data['expected'] else 'NO'}")
            
        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            import traceback
            traceback.print_exc()
```

#### Core Logic Skeleton Code

Here's the skeleton for the main hash interface that learners should complete:

```python
def hash_message(message: str) -> str:
    """
    Compute SHA-256 hash of input message and return as hex string.
    
    Args:
        message: Input string to hash
        
    Returns:
        64-character lowercase hexadecimal SHA-256 hash
        
    Raises:
        TypeError: If message is not a string
        UnicodeEncodeError: If message contains invalid Unicode
    """
    # TODO 1: Validate input message is a string type
    # TODO 2: Convert message string to UTF-8 byte array
    # TODO 3: Call hash_bytes with the byte array
    # TODO 4: Return the resulting hex string
    # Hint: Use message.encode('utf-8') for string to bytes conversion
    pass

def hash_bytes(data: bytes) -> str:
    """
    Compute SHA-256 hash of byte array input and return as hex string.
    
    Args:
        data: Input bytes to hash
        
    Returns:
        64-character lowercase hexadecimal SHA-256 hash
        
    Raises:
        TypeError: If data is not bytes type
    """
    # TODO 1: Validate input data is bytes type
    # TODO 2: Initialize hash state with INITIAL_HASH_VALUES
    # TODO 3: Preprocess message into 512-bit blocks
    # TODO 4: For each block: generate schedule, compress block, update hash state
    # TODO 5: Use OutputFormatter to convert final hash state to hex string
    # TODO 6: Return the hex string result
    # Hint: Copy INITIAL_HASH_VALUES to avoid modifying the original
    pass

def format_hash(hash_state: List[int]) -> str:
    """
    Convert internal hash state to standard hex string format.
    
    Args:
        hash_state: List of 8 32-bit integers h0-h7
        
    Returns:
        64-character lowercase hexadecimal string
        
    Raises:
        ValueError: If hash_state format is invalid
    """
    # TODO 1: Validate hash_state contains exactly 8 integers
    # TODO 2: Validate each integer is valid 32-bit unsigned value
    # TODO 3: Create OutputFormatter instance
    # TODO 4: Use formatter.format_hash_hex() to convert to string
    # TODO 5: Return the formatted hex string
    # Hint: Use OutputFormatter.validate_hash_state() for validation
    pass

def get_hash_bytes(hash_state: List[int]) -> bytes:
    """
    Convert internal hash state to 32-byte binary array.
    
    Args:
        hash_state: List of 8 32-bit integers h0-h7
        
    Returns:
        32-byte binary hash representation
        
    Raises:
        ValueError: If hash_state format is invalid
    """
    # TODO 1: Validate hash_state using same checks as format_hash
    # TODO 2: Create OutputFormatter instance  
    # TODO 3: Use formatter.format_hash_bytes() to convert to bytes
    # TODO 4: Return the binary result
    # Hint: This provides binary output option for advanced use cases
    pass
```

#### Language-Specific Implementation Hints

**Python-Specific Guidelines:**

1. **String Encoding**: Always use `.encode('utf-8')` when converting strings to bytes for hashing. This ensures consistent behavior across different platforms and Python versions.

2. **Integer Masking**: Use `word & 0xFFFFFFFF` to ensure 32-bit unsigned integer behavior, as Python integers have arbitrary precision by default.

3. **Struct Module**: Use `struct.pack('>I', value)` for big-endian 32-bit integer to bytes conversion. The `>` specifies big-endian byte order, and `I` specifies unsigned 32-bit integer.

4. **Hex Formatting**: The `.hex()` method on bytes objects automatically produces lowercase hexadecimal output, which matches NIST test vector format.

5. **Type Hints**: Use type hints consistently to catch type-related errors early and improve code documentation.

6. **Error Handling**: Raise specific exception types (`ValueError`, `TypeError`) rather than generic `Exception` to help users understand and handle errors appropriately.

#### Milestone Checkpoint

After implementing the final hash output generation, verify your implementation with these specific checkpoints:

**Test Command**: Run the complete test suite with NIST validation
```bash
python -m pytest tests/ -v
python test_integration.py  # Run the integration test with all test vectors
```

**Expected Behavior**: 
- All NIST test vectors should pass (empty string, "a", "abc", long string)
- Both hex string and binary byte output formats should work correctly
- Error handling should properly catch invalid inputs and hash states

**Manual Verification**: Test the main interface functions directly
```python
from sha256_hash import hash_message, hash_bytes

# These should produce the expected NIST test vector results
print(hash_message(""))     # Should output: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
print(hash_message("abc"))  # Should output: ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
print(hash_bytes(b"abc"))   # Should produce same result as hash_message("abc")
```

**Signs of Problems and Debugging**:
- **All hashes are wrong**: Check INITIAL_HASH_VALUES and ensure you're not modifying the original array
- **Empty string fails**: Verify padding algorithm handles zero-length input correctly
- **Hex output has wrong case**: Ensure you're using lowercase hex formatting
- **Wrong output length**: Check that all 8 hash values are being included in the final output
- **Endianness errors**: Verify you're using big-endian byte ordering throughout

![Hash State Evolution](./diagrams/hash-state-machine.svg)

#### Common Pitfalls and Debugging

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| All test vectors fail with consistent pattern | Wrong initial hash values or endianness | Print INITIAL_HASH_VALUES and compare to spec | Use correct initial values, check byte ordering |
| Empty string test fails, others pass | Padding algorithm error with zero-length input | Debug preprocessing with empty input | Handle zero-length case in padding calculation |
| Hex output is 63 or 65 characters | Leading zero handling error in hex formatting | Check output length and examine short values | Use proper formatting to preserve leading zeros |
| Output has uppercase hex characters | Wrong formatting method or string conversion | Compare output case to test vectors | Use lowercase hex formatting consistently |
| Binary output wrong but hex correct | Format conversion error in bytes method | Compare binary and hex outputs byte-by-byte | Check struct.pack parameters and byte ordering |
| Test passes individually but fails in batch | Hash state not being reset between calls | Test with fresh instances for each hash | Ensure clean state initialization for each hash |

The most common error at this stage is **endianness confusion**, where the internal hash computation is correct but the final byte ordering is wrong. This typically manifests as test vector failures where the output appears to be a valid hash but doesn't match the expected values. Debug this by comparing your intermediate hash values (h₀ through h₇) with reference implementations and ensuring the byte-level output matches exactly.

Another frequent issue is **incomplete hash state inclusion**, where only some of the eight hash values are included in the final output, typically due to loop errors or array slicing mistakes. This produces output that is too short or missing parts of the final hash. Always verify that your output includes all 256 bits from all eight 32-bit hash values.


## Component Interactions and Data Flow

> **Milestone(s):** All milestones (1-4) - Understanding how components work together

### Mental Model: Assembly Line Production

Think of SHA-256 as a **sophisticated assembly line** where raw materials (input messages) undergo precise transformations at each station before emerging as finished products (256-bit hashes). Each station has specialized equipment and trained operators who know exactly what to receive from the previous station, how to process it, and what to deliver to the next station.

Just as an automotive assembly line has quality control checkpoints, material handling protocols, and coordination between stations, the SHA-256 pipeline has strict component interfaces, state management protocols, and data validation at each stage. The assembly line supervisor (main hash function) orchestrates the entire process, ensuring that each component receives properly formatted inputs and delivers outputs that meet the next component's requirements.

The key insight is that **each component is both a customer and a supplier** - it has strict requirements for what it will accept and firm commitments for what it will deliver. This contract-based approach ensures that complex cryptographic processing can be broken down into manageable, testable, and debuggable pieces.

### End-to-End Pipeline Flow

The SHA-256 pipeline transforms input messages through four distinct stages, with each stage building upon the previous stage's output. Understanding this flow is crucial because each component must maintain precise data contracts with its neighbors.

The pipeline begins when the main `hash_message` or `hash_bytes` function receives an input string or byte array. This input undergoes a series of transformations that progressively convert human-readable data into a cryptographically secure 256-bit fingerprint.

**Stage 1: Message Preprocessing**
The `MessagePreprocessor` component receives the raw input message and transforms it into one or more standardized 512-bit blocks. This stage handles three critical transformations: binary conversion of the input data, application of the three-phase padding algorithm (append '1', zero-fill to boundary, append length), and parsing of the padded message into uniform blocks.

The preprocessor must handle variable-length inputs ranging from empty strings to arbitrarily large messages. For inputs longer than 447 bits (56 bytes minus the '1' padding bit), multiple 512-bit blocks are created. Each block becomes an independent unit for subsequent processing stages.

**Stage 2: Message Schedule Generation**
The `MessageScheduler` component receives each 512-bit `MessageBlock` and expands it into a 64-word `MessageSchedule`. This expansion process takes the 16 initial 32-bit words from the block and applies the recurrence relation to generate 48 additional words.

This stage implements the word expansion algorithm that provides cryptographic diffusion - small changes in the input block create large, unpredictable changes in the schedule. The sigma functions (`_sigma0` and `_sigma1`) perform the bit rotation and XOR operations that achieve this diffusion property.

**Stage 3: Main Compression**
The `CompressionEngine` receives each `MessageSchedule` and the current `HashState`, then executes 64 rounds of intensive bitwise mixing. This stage implements the heart of the SHA-256 algorithm - the Davies-Meyer construction that processes each schedule word through the round functions.

The compression engine maintains eight working variables (a through h) that undergo continuous transformation through the choice, majority, and upper-sigma functions. After 64 rounds, the final working variables are added back to the hash state using modular arithmetic.

**Stage 4: Hash Finalization**
The `OutputFormatter` receives the final `HashState` after all message blocks have been processed and converts it into the desired output format. This stage handles endianness conversion, concatenation of the eight 32-bit hash values, and formatting as either hexadecimal strings or binary byte arrays.

![End-to-End Data Flow Pipeline](./diagrams/data-flow-pipeline.svg)

> **Critical Design Insight**: The pipeline is designed as a **single-pass algorithm** - each message block flows through all four stages before the next block begins processing. This approach minimizes memory usage and enables streaming processing of large inputs, but requires careful state management between blocks.

**Multi-Block Processing Flow**
For messages that require multiple blocks, the pipeline exhibits a specific coordination pattern. The preprocessor generates all blocks upfront and stores them in a list. The main hash function then iterates through this list, sending each block through stages 2 and 3 while maintaining the evolving hash state.

The coordination follows this precise sequence:
1. Initialize hash state with `INITIAL_HASH_VALUES`
2. For each block in the preprocessed block list:
   - Generate message schedule from current block
   - Compress schedule with current hash state
   - Update hash state with compression results
3. Format final hash state as output

This pattern ensures that information from each block influences the processing of subsequent blocks through the evolving hash state, creating the chaining behavior essential for cryptographic security.

**Data Validation Checkpoints**
At each stage boundary, the pipeline includes implicit validation checkpoints that verify data integrity and format compliance:

| Stage Boundary | Validation Performed | Recovery Action |
|----------------|---------------------|----------------|
| Input → Preprocessing | Message encoding validation | Raise encoding error |
| Preprocessing → Scheduling | Block size verification (512 bits) | Internal consistency error |
| Scheduling → Compression | Schedule length verification (64 words) | Internal consistency error |
| Compression → Output | Hash state length verification (8 words) | Internal consistency error |
| Output → Return | Format validation (hex length, byte count) | Internal formatting error |

### Hash State Management

Hash state management represents one of the most critical aspects of SHA-256 implementation because the hash state carries cryptographic information between message blocks and maintains the algorithm's security properties. The hash state serves as both the algorithm's memory and its primary defense against cryptographic attacks.

**Hash State Structure and Evolution**
The `HashState` consists of eight 32-bit words (h0 through h7) that represent the cumulative cryptographic digest of all processed message blocks. These words are initialized to carefully chosen constants derived from the fractional parts of square roots of the first eight prime numbers, providing cryptographic strength and preventing trivial attacks.

The hash state undergoes continuous evolution as each message block is processed. This evolution follows a precise mathematical pattern that ensures small changes in any input block create large, unpredictable changes in the final hash value - the avalanche effect that makes SHA-256 cryptographically secure.

**State Initialization and Constants**
Hash state initialization uses the `INITIAL_HASH_VALUES` constants, which are not arbitrary but derived from mathematical principles. These values provide cryptographic strength by eliminating any hidden structure or backdoors that could be exploited by attackers.

| Hash Value | Initialization Constant | Mathematical Origin |
|------------|------------------------|-------------------|
| h0 | 0x6a09e667 | √2 fractional part |
| h1 | 0xbb67ae85 | √3 fractional part |
| h2 | 0x3c6ef372 | √5 fractional part |
| h3 | 0xa54ff53a | √7 fractional part |
| h4 | 0x510e527f | √11 fractional part |
| h5 | 0x9b05688c | √13 fractional part |
| h6 | 0x1f83d9ab | √17 fractional part |
| h7 | 0x5be0cd19 | √19 fractional part |

These constants ensure that SHA-256 produces deterministic results across all implementations while providing no computational advantage to attackers who might try to exploit the initial state.

**State Update Protocol**
The hash state update follows the Davies-Meyer construction pattern, which provides provable security properties under certain mathematical assumptions. After each compression round, the new hash state is computed by adding the compressed working variables to the previous hash state using 32-bit modular arithmetic.

The update protocol implements this precise sequence:
1. Store current hash state as h0_prev through h7_prev
2. Execute 64 compression rounds with current message schedule
3. Extract final working variables a through h from compression
4. Compute new hash state: h0_new = (h0_prev + a) mod 2³²
5. Continue for all eight hash values
6. Store updated hash state for next block processing

This feedforward addition pattern ensures that information from previous blocks influences the processing of subsequent blocks, creating the chaining behavior essential for collision resistance and preimage resistance.

**Concurrency and State Isolation**
Hash state management must account for potential concurrent usage patterns, even though SHA-256 itself is inherently sequential. Different hash computations must maintain completely isolated hash states to prevent cross-contamination between concurrent hash operations.

> **Architecture Decision: Stateless Component Design**
> - **Context**: SHA-256 components could maintain internal state or operate statelessly with explicit state parameters
> - **Options Considered**: 
>   1. Stateful components with internal hash state storage
>   2. Stateless components with explicit state passing
>   3. Hybrid approach with optional state caching
> - **Decision**: Pure stateless design with explicit state passing
> - **Rationale**: Stateless components eliminate concurrency issues, simplify testing, and enable functional composition patterns. State passing makes data flow explicit and debuggable.
> - **Consequences**: Slightly more verbose function signatures, but dramatically improved testability and thread safety

**State Persistence and Serialization**
While this implementation focuses on complete hash computation rather than incremental hashing, the hash state structure supports serialization for debugging and testing purposes. The state can be captured at any point in the pipeline and restored for replay or analysis.

The hash state serialization includes both the eight hash values and metadata about the processing context:

| Field | Type | Purpose |
|-------|------|---------|
| hash_values | List[int] | Eight 32-bit hash state words |
| blocks_processed | int | Count of completed message blocks |
| total_message_length | int | Original message length in bits |
| current_block_index | int | Index of block currently being processed |

**State Validation and Integrity Checking**
Hash state validation ensures that the state structure remains consistent throughout processing. The `validate_hash_state` function performs comprehensive checks on state integrity:

1. Verify exactly eight hash values are present
2. Confirm each hash value fits in 32 bits (0 ≤ value < 2³²)
3. Check that hash values are integers, not floating-point numbers
4. Validate metadata consistency (block counts, message lengths)

State validation catches implementation errors early and provides clear diagnostic information when problems occur. This validation is particularly important during development and testing phases.

![System Architecture](./diagrams/system-architecture.svg)

### Component Interface Contracts

Component interface contracts define the precise input/output specifications, preconditions, and postconditions that govern how SHA-256 components interact. These contracts serve as both implementation guides and testing specifications, ensuring that components can be developed and tested independently while maintaining system-wide correctness.

**Message Preprocessor Interface Contract**
The `MessagePreprocessor` component serves as the pipeline's entry point and must handle the widest variety of input formats while producing standardized output for downstream components.

| Method | Parameters | Returns | Preconditions | Postconditions |
|--------|------------|---------|---------------|----------------|
| `preprocess_message` | message: str | List[MessageBlock] | Message is valid UTF-8 string | Returns 1+ blocks, each exactly 512 bits |
| `_apply_padding` | message_bytes: bytes | bytes | Input is valid byte sequence | Output length ≡ 0 (mod 64 bytes) |
| `_parse_blocks` | padded_bytes: bytes | List[MessageBlock] | Input length divisible by 64 | Each block contains exactly 16 32-bit words |
| `_calculate_zero_padding_bits` | message_bit_length: int | int | Length ≥ 0 | Returns bits needed to reach 448 mod 512 |

The preprocessor contract guarantees that output blocks conform to the SHA-256 specification requirements. Each `MessageBlock` contains exactly 16 32-bit words in big-endian byte order, and the final block correctly encodes the original message length in the last 64 bits.

**Critical Contract Requirements**:
- Input validation must reject invalid encodings and raise clear error messages
- Padding algorithm must produce bit-identical results to NIST reference implementations
- Block parsing must preserve big-endian byte ordering throughout
- Length encoding must handle messages up to 2⁶⁴-1 bits (though practical limits apply)

**Message Scheduler Interface Contract**
The `MessageScheduler` transforms uniform message blocks into cryptographically mixed word schedules using the SHA-256 expansion algorithm.

| Method | Parameters | Returns | Preconditions | Postconditions |
|--------|------------|---------|---------------|----------------|
| `generate_schedule` | block: MessageBlock | MessageSchedule | Block has exactly 16 words | Schedule has exactly 64 words |
| `_sigma0` | x: int | int | 0 ≤ x < 2³² | Returns σ₀(x) = ROTR₇(x) ⊕ ROTR₁₈(x) ⊕ SHR₃(x) |
| `_sigma1` | x: int | int | 0 ≤ x < 2³² | Returns σ₁(x) = ROTR₁₇(x) ⊕ ROTR₁₉(x) ⊕ SHR₁₀(x) |

The scheduler contract ensures that word expansion follows the precise mathematical definition from the NIST specification. The sigma functions implement exact bit rotation and shift operations, and the schedule generation produces deterministic, repeatable results.

**Schedule Generation Algorithm Contract**:
1. Copy 16 input words to schedule positions 0-15 without modification
2. For positions 16-63, apply recurrence: W[t] = σ₁(W[t-2]) + W[t-7] + σ₀(W[t-15]) + W[t-16]
3. All arithmetic operations use 32-bit modular arithmetic (automatic overflow)
4. Sigma functions use exact rotation counts from NIST specification

**Compression Engine Interface Contract**
The `CompressionEngine` implements the cryptographic core of SHA-256, executing 64 rounds of intensive bitwise mixing that provides the algorithm's security properties.

| Method | Parameters | Returns | Preconditions | Postconditions |
|--------|------------|---------|---------------|----------------|
| `compress_block` | schedule: MessageSchedule, hash_state: HashState | HashState | Schedule has 64 words, state has 8 values | Returns updated 8-word hash state |
| `choice` | x: int, y: int, z: int | int | All inputs 0 ≤ value < 2³² | Returns Ch(x,y,z) = (x∧y) ⊕ (¬x∧z) |
| `majority` | x: int, y: int, z: int | int | All inputs 0 ≤ value < 2³² | Returns Maj(x,y,z) = (x∧y) ⊕ (x∧z) ⊕ (y∧z) |
| `upper_sigma_0` | x: int | int | 0 ≤ x < 2³² | Returns Σ₀(x) = ROTR₂(x) ⊕ ROTR₁₃(x) ⊕ ROTR₂₂(x) |
| `upper_sigma_1` | x: int | int | 0 ≤ x < 2³² | Returns Σ₁(x) = ROTR₆(x) ⊕ ROTR₁₁(x) ⊕ ROTR₂₅(x) |

The compression contract guarantees that each round follows the exact SHA-256 round function definition and that the final hash state update uses the Davies-Meyer construction pattern.

**Compression Round Function Requirements**:
- Working variables a-h must be initialized from current hash state
- Each round must use the correct schedule word W[t] and round constant K[t]
- Round functions must implement exact bitwise operations from specification
- Final hash update must use modular addition: h_new[i] = (h_old[i] + working[i]) mod 2³²

**Output Formatter Interface Contract**
The `OutputFormatter` converts internal hash states into user-visible output formats while maintaining compatibility with standard SHA-256 representations.

| Method | Parameters | Returns | Preconditions | Postconditions |
|--------|------------|---------|---------------|----------------|
| `format_hash_hex` | hash_state: HashState | str | State has 8 32-bit values | Returns 64-char lowercase hex string |
| `format_hash_bytes` | hash_state: HashState | bytes | State has 8 32-bit values | Returns 32-byte big-endian byte array |
| `validate_hash_state` | hash_state: HashState | bool | State is list or tuple | Returns True if state format is valid |

The formatter contract ensures that output representations match standard SHA-256 conventions and can be verified against NIST test vectors or other reference implementations.

**Format Conversion Requirements**:
- Hexadecimal output must use lowercase letters a-f (not uppercase A-F)
- Big-endian byte ordering must be preserved in both hex and binary formats
- Each 32-bit hash value contributes exactly 8 hex characters or 4 bytes to output
- Output concatenation follows h0||h1||h2||h3||h4||h5||h6||h7 order

> **Architecture Decision: Explicit Error Handling Strategy**
> - **Context**: Components can fail due to invalid inputs, implementation bugs, or resource constraints
> - **Options Considered**:
>   1. Silent failure with default return values
>   2. Exception-based error propagation
>   3. Result types with explicit success/failure states
> - **Decision**: Exception-based error handling with specific exception types
> - **Rationale**: Exceptions provide clear failure semantics, enable stack trace debugging, and prevent silent corruption. Cryptographic code should fail loudly on any anomaly.
> - **Consequences**: Callers must handle exceptions, but failures are immediately visible and debuggable.

**Cross-Component Data Validation**
Interface contracts include validation requirements that detect implementation errors and ensure data integrity across component boundaries. Each component validates its inputs and guarantees its outputs meet the next component's requirements.

| Component | Input Validation | Output Guarantee |
|-----------|------------------|------------------|
| MessagePreprocessor | UTF-8 encoding validity | Blocks are exactly 512 bits |
| MessageScheduler | Block has 16 32-bit words | Schedule has 64 32-bit words |
| CompressionEngine | Schedule length, hash state format | Hash state has 8 32-bit values |
| OutputFormatter | Hash state structure | Output matches standard format |

**Interface Evolution and Versioning**
Component interfaces are designed for stability and backward compatibility. Changes to interface contracts require careful consideration because they affect all downstream components and test cases.

Interface evolution follows these principles:
1. **Additive changes only**: New optional parameters or methods can be added
2. **Semantic preservation**: Existing method behavior cannot change
3. **Validation strengthening**: Input validation can become stricter, but output guarantees cannot weaken
4. **Documentation updates**: All changes require corresponding documentation updates

### Common Pitfalls

⚠️ **Pitfall: State Mutation Across Blocks**
Developers often accidentally modify the hash state object in place, causing incorrect results for multi-block messages. The hash state should be treated as immutable within each compression operation, with updates creating new state objects rather than modifying existing ones. Fix this by explicitly copying state before modification or using immutable data structures.

⚠️ **Pitfall: Component Interface Mismatches**
Components may produce outputs that don't match the input requirements of downstream components, particularly around endianness and word size assumptions. This creates silent data corruption that's difficult to debug. Fix this by implementing comprehensive interface validation at each component boundary and using explicit type checking.

⚠️ **Pitfall: Missing Data Flow Validation**
Skipping validation of intermediate data structures leads to cryptographic failures that manifest as incorrect final hash values. Each component should validate both its inputs and outputs to catch implementation errors early. Implement validation functions that check data structure integrity at each pipeline stage.

⚠️ **Pitfall: Concurrency State Contamination**
If components maintain internal state, concurrent hash computations can interfere with each other, producing incorrect results. Design all components as stateless functions that receive all necessary data through parameters and return results without side effects.

### Implementation Guidance

**A. Technology Recommendations**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Data Flow Control | Direct function calls | Generator-based pipeline |
| State Management | Explicit parameter passing | Immutable state objects |
| Validation | Assert statements | Custom exception hierarchy |
| Testing | Simple unit tests | Property-based testing |
| Debugging | Print statements | Structured logging with state dumps |

**B. Recommended File Organization**

```
sha256/
├── __init__.py                    # Main public interface
├── core/
│   ├── __init__.py
│   ├── preprocessor.py            # MessagePreprocessor component
│   ├── scheduler.py               # MessageScheduler component  
│   ├── compression.py             # CompressionEngine component
│   └── formatter.py               # OutputFormatter component
├── utils/
│   ├── __init__.py
│   ├── constants.py               # INITIAL_HASH_VALUES, ROUND_CONSTANTS
│   ├── bitops.py                  # right_rotate_32, mask_32_bits utilities
│   └── validation.py              # Input validation and error checking
├── types/
│   ├── __init__.py
│   └── data_types.py              # HashState, MessageBlock, etc.
└── tests/
    ├── __init__.py
    ├── test_integration.py        # End-to-end pipeline tests
    ├── test_preprocessor.py       # Component-specific tests
    ├── test_scheduler.py
    ├── test_compression.py
    └── test_formatter.py
```

**C. Infrastructure Starter Code (Complete)**

```python
# utils/validation.py - Complete validation utilities
from typing import List, Union

class SHA256ValidationError(Exception):
    """Base exception for SHA-256 validation failures."""
    pass

class InvalidMessageError(SHA256ValidationError):
    """Raised when input message format is invalid."""
    pass

class ComponentInterfaceError(SHA256ValidationError):
    """Raised when component inputs/outputs don't meet contract requirements."""
    pass

def validate_message_input(message: Union[str, bytes]) -> bytes:
    """
    Validate and normalize input message to bytes.
    
    Args:
        message: Input message as string or bytes
        
    Returns:
        Normalized message as bytes
        
    Raises:
        InvalidMessageError: If message cannot be encoded as bytes
    """
    if isinstance(message, str):
        try:
            return message.encode('utf-8')
        except UnicodeEncodeError as e:
            raise InvalidMessageError(f"Cannot encode string as UTF-8: {e}")
    elif isinstance(message, bytes):
        return message
    else:
        raise InvalidMessageError(f"Message must be str or bytes, got {type(message)}")

def validate_hash_state_format(hash_state: List[int]) -> bool:
    """
    Validate hash state structure and value ranges.
    
    Args:
        hash_state: List of hash values to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(hash_state, (list, tuple)):
        return False
    if len(hash_state) != 8:
        return False
    for value in hash_state:
        if not isinstance(value, int):
            return False
        if not (0 <= value < 2**32):
            return False
    return True

def validate_component_boundary(component_name: str, data_type: str, data) -> None:
    """
    Validate data at component boundaries to catch interface contract violations.
    
    Args:
        component_name: Name of component being validated
        data_type: Type of data being validated
        data: The data to validate
        
    Raises:
        ComponentInterfaceError: If data doesn't meet interface requirements
    """
    if data_type == "message_block" and len(data) != 16:
        raise ComponentInterfaceError(f"{component_name}: MessageBlock must have exactly 16 words, got {len(data)}")
    elif data_type == "message_schedule" and len(data) != 64:
        raise ComponentInterfaceError(f"{component_name}: MessageSchedule must have exactly 64 words, got {len(data)}")
    elif data_type == "hash_state" and not validate_hash_state_format(data):
        raise ComponentInterfaceError(f"{component_name}: Invalid hash state format")
```

```python
# utils/bitops.py - Complete bitwise operation utilities
def right_rotate_32(value: int, amount: int) -> int:
    """
    Perform 32-bit right rotation (circular shift).
    
    Args:
        value: 32-bit integer to rotate
        amount: Number of positions to rotate right
        
    Returns:
        Rotated 32-bit integer
    """
    # Ensure value is within 32-bit range
    value = mask_32_bits(value)
    # Normalize rotation amount
    amount = amount % 32
    # Perform rotation: shift right and OR with wrapped bits
    return ((value >> amount) | (value << (32 - amount))) & 0xFFFFFFFF

def mask_32_bits(value: int) -> int:
    """
    Mask integer to 32-bit unsigned range.
    
    Args:
        value: Integer value to mask
        
    Returns:
        Value masked to 32-bit range (0 to 2^32-1)
    """
    return value & 0xFFFFFFFF

def bytes_to_words_32(data: bytes) -> List[int]:
    """
    Convert byte array to list of 32-bit words in big-endian order.
    
    Args:
        data: Byte array with length divisible by 4
        
    Returns:
        List of 32-bit words
        
    Raises:
        ValueError: If data length is not divisible by 4
    """
    if len(data) % 4 != 0:
        raise ValueError(f"Data length must be divisible by 4, got {len(data)}")
    
    words = []
    for i in range(0, len(data), 4):
        # Extract 4 bytes and combine into 32-bit big-endian word
        word = (data[i] << 24) | (data[i+1] << 16) | (data[i+2] << 8) | data[i+3]
        words.append(word)
    return words

def words_32_to_bytes(words: List[int]) -> bytes:
    """
    Convert list of 32-bit words to byte array in big-endian order.
    
    Args:
        words: List of 32-bit words
        
    Returns:
        Big-endian byte array
    """
    data = bytearray()
    for word in words:
        # Convert word to 4 bytes in big-endian order
        data.extend([
            (word >> 24) & 0xFF,
            (word >> 16) & 0xFF,
            (word >> 8) & 0xFF,
            word & 0xFF
        ])
    return bytes(data)
```

**D. Core Logic Skeleton Code**

```python
# __init__.py - Main pipeline coordination
from .core.preprocessor import MessagePreprocessor
from .core.scheduler import MessageScheduler
from .core.compression import CompressionEngine
from .core.formatter import OutputFormatter
from .utils.constants import INITIAL_HASH_VALUES
from .utils.validation import validate_message_input

def hash_message(message: str) -> str:
    """
    Compute SHA-256 hash of string message.
    
    Args:
        message: Input string to hash
        
    Returns:
        64-character lowercase hexadecimal hash string
        
    Raises:
        InvalidMessageError: If message encoding is invalid
    """
    # TODO 1: Validate and convert message to bytes using validate_message_input
    # TODO 2: Initialize hash state with INITIAL_HASH_VALUES
    # TODO 3: Preprocess message into blocks using MessagePreprocessor
    # TODO 4: For each block:
    #   TODO 4a: Generate message schedule using MessageScheduler
    #   TODO 4b: Compress block with current hash state using CompressionEngine
    #   TODO 4c: Update hash state with compression result
    # TODO 5: Format final hash state as hex string using OutputFormatter
    # TODO 6: Return formatted hash string
    pass

def hash_bytes(data: bytes) -> str:
    """
    Compute SHA-256 hash of byte data.
    
    Args:
        data: Input bytes to hash
        
    Returns:
        64-character lowercase hexadecimal hash string
    """
    # TODO 1: Follow same pipeline as hash_message but skip encoding validation
    # TODO 2: Use data directly for preprocessing
    # Hint: Consider refactoring common pipeline logic into internal helper function
    pass
```

```python
# core/preprocessor.py - Message preprocessing skeleton
from typing import List
from ..types.data_types import MessageBlock
from ..utils.validation import validate_component_boundary
from ..utils.bitops import bytes_to_words_32

class MessagePreprocessor:
    @staticmethod
    def preprocess_message(message_bytes: bytes) -> List[MessageBlock]:
        """
        Convert message bytes to list of 512-bit message blocks.
        
        Args:
            message_bytes: Raw message data
            
        Returns:
            List of MessageBlock objects (each containing 16 32-bit words)
        """
        # TODO 1: Calculate original message length in bits
        # TODO 2: Apply three-phase padding algorithm using _apply_padding
        # TODO 3: Parse padded bytes into 512-bit blocks using _parse_blocks
        # TODO 4: Validate each block meets interface contract requirements
        # TODO 5: Return list of MessageBlock objects
        # Hint: Use validate_component_boundary to check output format
        pass

    @staticmethod
    def _apply_padding(message_bytes: bytes) -> bytes:
        """
        Apply SHA-256 padding algorithm to message bytes.
        
        Args:
            message_bytes: Original message data
            
        Returns:
            Padded message with length divisible by 64 bytes
        """
        # TODO 1: Calculate message length in bits
        # TODO 2: Convert message to mutable bytearray
        # TODO 3: Append mandatory '1' padding bit (0x80 byte)
        # TODO 4: Calculate zero padding needed to reach 448 mod 512 bits
        # TODO 5: Append calculated zero bytes
        # TODO 6: Append original length as 64-bit big-endian integer
        # TODO 7: Return padded bytes
        # Hint: Use struct.pack('>Q', length_bits) for 64-bit big-endian encoding
        pass

    @staticmethod
    def _parse_blocks(padded_bytes: bytes) -> List[MessageBlock]:
        """
        Parse padded message into 512-bit blocks.
        
        Args:
            padded_bytes: Padded message with length divisible by 64
            
        Returns:
            List of MessageBlock objects
        """
        # TODO 1: Validate input length is divisible by 64 bytes
        # TODO 2: Split bytes into 64-byte chunks
        # TODO 3: Convert each chunk to 16 32-bit words using bytes_to_words_32
        # TODO 4: Create MessageBlock object for each chunk
        # TODO 5: Return list of blocks
        # Hint: range(0, len(padded_bytes), 64) for 64-byte chunks
        pass
```

**E. Language-Specific Hints**

- **Bitwise Operations**: Python's `>>` and `<<` operators work correctly for SHA-256, but always use `& 0xFFFFFFFF` to mask results to 32 bits
- **Byte Handling**: Use `struct.pack('>Q', value)` for big-endian 64-bit integers and `struct.pack('>I', value)` for 32-bit words
- **List Comprehensions**: Useful for block processing: `[generate_schedule(block) for block in blocks]`
- **Exception Handling**: Use specific exception types from `utils.validation` rather than generic `ValueError`
- **Type Hints**: Include complete type annotations for all function parameters and return values to catch interface contract violations

**F. Milestone Checkpoints**

**After implementing end-to-end pipeline flow**:
- Run: `python -c "from sha256 import hash_message; print(hash_message('abc'))"`
- Expected: `ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad`
- Verify: Hash matches NIST test vector for 'abc'
- Debug: If wrong, check component interface contracts and data flow validation

**After implementing hash state management**:
- Run: `python -c "from sha256 import hash_message; print(hash_message('abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq'))"`
- Expected: `248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1`
- Verify: Multi-block processing maintains correct state evolution
- Debug: If wrong, examine hash state between blocks and validate state update arithmetic

**After implementing component interface contracts**:
- Run: `python -m pytest tests/test_integration.py -v`
- Expected: All integration tests pass with component boundary validation
- Verify: Components handle invalid inputs with clear error messages
- Debug: If failures, check input validation and interface contract enforcement


## Error Handling and Edge Cases

> **Milestone(s):** All milestones (1-4) - Ensuring robust implementation across all components

### Mental Model: Quality Control Inspector

Think of error handling in SHA-256 as a **quality control inspector** working alongside an assembly line. Just as an inspector checks raw materials before they enter production, validates each assembly step, and ensures the final product meets specifications, our error handling system validates inputs before processing, monitors each computation stage for anomalies, and guarantees the output meets cryptographic standards. The inspector doesn't just catch defective products—they prevent defects from propagating through the system and corrupting downstream operations.

Error handling in cryptographic implementations is particularly critical because silent failures can produce seemingly valid but incorrect hash values, which could compromise security in production systems. Our approach focuses on three key areas: rigorous input validation to catch problems early, careful edge case handling to ensure correct behavior in boundary conditions, and arithmetic overflow protection to maintain the mathematical integrity required by the NIST specification.

### Input Validation Strategy

The **input validation strategy** serves as the first line of defense against invalid data entering our SHA-256 pipeline. Since cryptographic hash functions must produce deterministic results, any ambiguity or inconsistency in input handling could lead to incorrect outputs that fail to match reference implementations.

> **Decision: Comprehensive Input Validation with Early Rejection**
> - **Context**: SHA-256 must handle arbitrary byte sequences, but Python's string handling introduces encoding complexities that could lead to inconsistent results across different environments
> - **Options Considered**: 
>   1. Accept any Python object and attempt conversion
>   2. Accept only UTF-8 strings and convert to bytes
>   3. Accept both strings and bytes with explicit validation
> - **Decision**: Accept both strings and bytes with explicit validation and normalization
> - **Rationale**: Provides flexibility for users while ensuring deterministic byte-level processing that matches NIST test vectors
> - **Consequences**: Requires careful encoding handling but eliminates ambiguity in input interpretation

The validation strategy implements a three-tier approach: type validation, encoding validation, and boundary condition checking. Each tier serves a specific purpose in ensuring input data can be reliably processed by the downstream components.

| Validation Tier | Purpose | Validation Rules | Error Response |
|-----------------|---------|------------------|----------------|
| Type Validation | Ensure supported input types | Accept `str` or `bytes` objects only | Raise `InvalidMessageError` with type information |
| Encoding Validation | Normalize string inputs to bytes | Convert strings using UTF-8 encoding | Raise `InvalidMessageError` for encoding failures |
| Boundary Validation | Check input size limits | Verify input length ≤ 2^55 bytes (SHA-256 limit) | Raise `InvalidMessageError` with size information |
| Content Validation | Verify byte sequence integrity | Ensure bytes object contains valid data | Raise `InvalidMessageError` for corrupted data |

The `validate_message_input` function serves as the single entry point for all input validation, ensuring consistent behavior across all hash operations. This function performs comprehensive checks while providing detailed error messages that help developers identify and correct input issues.

**Type checking and normalization** begins by examining the input object's type and applying appropriate conversion logic. String inputs undergo UTF-8 encoding to produce the byte sequence that will be hashed, while byte inputs are validated for integrity. This approach ensures that the string "hello" always produces the same hash value regardless of the Python environment or locale settings.

**Encoding validation** addresses the complexity of converting Unicode strings to byte sequences. Since SHA-256 operates on bytes, not characters, we must ensure consistent encoding behavior. UTF-8 encoding provides the standard approach used by most cryptographic libraries and ensures compatibility with NIST test vectors that specify input as byte sequences.

**Size limit validation** protects against inputs that exceed SHA-256's theoretical maximum message length of 2^64 - 1 bits (approximately 2^61 bytes). While Python's memory limitations make such large inputs impractical, explicit checking prevents integer overflow in length calculations and provides clear error messages for unreasonably large inputs.

| Input Scenario | Validation Behavior | Expected Result | Error Handling |
|----------------|---------------------|-----------------|----------------|
| Valid UTF-8 string | Convert to bytes using UTF-8 encoding | Return normalized bytes | None |
| Valid bytes object | Validate integrity and pass through | Return original bytes | None |
| Invalid UTF-8 string | Detect encoding errors during conversion | N/A | Raise `InvalidMessageError` |
| Non-string/bytes input | Type check fails immediately | N/A | Raise `InvalidMessageError` |
| Oversized input | Size check fails during validation | N/A | Raise `InvalidMessageError` |
| Corrupted bytes | Integrity check fails during validation | N/A | Raise `InvalidMessageError` |

⚠️ **Pitfall: Silent Encoding Failures**
A common mistake is allowing Python's default error handling to convert invalid UTF-8 sequences using replacement characters. This produces valid bytes that hash successfully but don't represent the original input. Always use strict encoding validation to catch these issues early rather than allowing corrupted data to propagate through the hash computation.

### Edge Case Handling

**Edge case handling** addresses the boundary conditions and special scenarios that standard implementations often overlook. These cases are particularly important for SHA-256 because the NIST specification defines exact behavior for all possible inputs, including edge cases that rarely occur in typical usage.

> **Decision: Explicit Edge Case Implementation with Reference Validation**
> - **Context**: SHA-256 specification defines behavior for edge cases like empty inputs and maximum-length messages, but these cases stress test implementation correctness
> - **Options Considered**:
>   1. Handle edge cases implicitly through general algorithms
>   2. Add special-case code paths for edge conditions
>   3. Implement edge cases explicitly with dedicated test validation
> - **Decision**: Implement edge cases explicitly with dedicated test validation
> - **Rationale**: Ensures correctness for all inputs and provides confidence that the implementation handles the full specification
> - **Consequences**: Adds complexity but guarantees compliance with NIST test vectors and prevents subtle bugs in production

The edge case handling strategy focuses on three critical areas: empty input processing, maximum length inputs, and boundary conditions in the padding algorithm. Each area requires special attention because these cases often expose implementation bugs that don't manifest with typical inputs.

**Empty string handling** represents the most important edge case because the SHA-256 hash of an empty input is a well-known constant that serves as a basic correctness test. The empty string must be padded to exactly 512 bits (one block) consisting of a single '1' bit followed by 447 zero bits, then the 64-bit length encoding of zero.

| Edge Case Category | Specific Cases | Special Handling Required | Validation Method |
|-------------------|----------------|---------------------------|-------------------|
| Empty Inputs | Empty string `""`, empty bytes `b""` | Single block padding with zero length | Compare against NIST empty hash |
| Single Character | Single byte inputs like `"a"` or `b"x"` | Standard padding but minimal content | Verify padding calculation |
| Boundary Lengths | Inputs requiring exactly 448 bits | Padding triggers additional block | Check block count calculation |
| Maximum Length | Theoretical 2^64-1 bit inputs | Length encoding edge cases | Validate length field overflow |
| Block Boundaries | Inputs exactly 512*N bits | No zero padding required | Verify correct block parsing |
| Unicode Edge Cases | Strings with multi-byte UTF-8 characters | Consistent encoding behavior | Compare byte-level results |

**Large input handling** addresses inputs that approach or exceed practical memory limits. While SHA-256 theoretically supports messages up to 2^64 - 1 bits, Python implementations must handle memory constraints gracefully. The streaming approach processes large inputs in chunks rather than loading entire messages into memory.

**Padding boundary conditions** require careful attention because the padding algorithm behaves differently depending on the input length. When the input length leaves exactly 64 bits available in the final block (length ≡ 448 mod 512), no zero padding is needed. When insufficient space remains for the length encoding, an additional block must be added.

The padding calculation uses modular arithmetic to determine the required zero padding:

1. Calculate the message bit length including the mandatory '1' padding bit
2. Determine the remainder when dividing by 512 (block size)
3. If remainder ≤ 448, add zeros to reach exactly 448 bits in the current block
4. If remainder > 448, add zeros to complete the current block, then add a new block with zeros to reach 448 bits
5. Append the 64-bit big-endian length encoding to complete the final block

| Input Length (bits) | Padding Bits Added | Total Blocks | Special Handling |
|---------------------|-------------------|--------------|------------------|
| 0 | 1 + 447 + 64 = 512 | 1 | Empty input case |
| 448 | 1 + 63 + 64 = 128 | 1 | Exactly fits boundary |
| 449 | 1 + 510 + 64 = 575 | 2 | Requires extra block |
| 511 | 1 + 448 + 64 = 513 | 2 | One bit under boundary |
| 512 | 1 + 447 + 64 = 512 | 2 | Exact block boundary |

⚠️ **Pitfall: Off-by-One in Boundary Calculations**
The most common error in padding implementation is miscalculating when an additional block is required. Remember that the '1' padding bit must be added before calculating remaining space. An input of exactly 447 bits becomes 448 bits after the '1' bit, requiring an additional block for the length encoding.

**Unicode and encoding edge cases** deserve special attention because they can produce subtle bugs that only manifest with international text inputs. Multi-byte UTF-8 characters must be handled consistently to ensure the same logical string always produces the same hash value regardless of the processing environment.

### Arithmetic Overflow Protection

**Arithmetic overflow protection** ensures that all numerical computations maintain the mathematical properties required by the SHA-256 specification. Since the algorithm relies heavily on 32-bit modular arithmetic, any deviation from the specified overflow behavior can produce incorrect hash values.

> **Decision: Explicit 32-bit Masking with Verification**
> - **Context**: Python's unlimited integer precision doesn't naturally implement 32-bit overflow behavior required by SHA-256 specification
> - **Options Considered**:
>   1. Rely on Python's arbitrary precision and mask only at the end
>   2. Use ctypes or struct module for native 32-bit arithmetic
>   3. Implement explicit masking after every operation
> - **Decision**: Implement explicit masking after every operation with verification
> - **Rationale**: Provides maximum control and predictability while making overflow behavior explicit and debuggable
> - **Consequences**: Requires discipline to mask consistently but eliminates subtle overflow bugs

The overflow protection strategy implements three key mechanisms: consistent 32-bit masking, operation-specific overflow handling, and verification against known test vectors. Each mechanism serves to maintain the mathematical integrity required for cryptographic correctness.

**32-bit masking operations** ensure that all intermediate values remain within the valid range for unsigned 32-bit integers (0 to 4,294,967,295). Python's unlimited integer precision means that arithmetic operations can produce values larger than 32 bits, which must be reduced using modular arithmetic.

| Operation Type | Overflow Risk | Masking Requirement | Implementation Pattern |
|----------------|---------------|---------------------|------------------------|
| Addition | Two 32-bit values can sum to 33 bits | Mask after each addition | `(a + b) & 0xFFFFFFFF` |
| Rotation | No overflow risk, preserves bit count | No masking needed | Direct bit manipulation |
| XOR | No overflow risk, bitwise operation | No masking needed | Direct XOR operation |
| Shift Operations | Can reduce bits but not increase | No masking needed | Direct shift operation |
| Word Expansion | Multiple additions in sequence | Mask final result | Apply after complete calculation |
| Round Functions | Complex expressions with additions | Mask intermediate results | Multiple mask operations |

The `mask_32_bits` function provides a centralized implementation of the masking operation, ensuring consistent behavior throughout the codebase. This function takes any integer value and returns the equivalent value in the range [0, 2^32 - 1] using bitwise AND operation with the mask 0xFFFFFFFF.

**Operation-specific overflow handling** addresses the unique requirements of different SHA-256 operations. While simple operations like XOR cannot overflow, complex operations like the round function calculations involve multiple additions that must be carefully managed.

The **word expansion calculations** in message schedule generation perform multiple arithmetic operations in sequence, each of which could potentially overflow. The sigma functions combine rotation and XOR operations (which don't overflow) with addition operations (which can overflow). The recurrence relation for generating schedule words adds four 32-bit values, potentially producing results up to 34 bits.

**Round function overflow management** requires particular attention because each compression round performs several 32-bit additions in the context of updating working variables. The temporary calculations involve adding up to five 32-bit values: the current working variable, the result of the choice or majority function, the round constant, the schedule word, and the result of the upper-sigma function.

| Calculation Step | Maximum Bit Width | Overflow Protection | Masking Points |
|------------------|-------------------|---------------------|----------------|
| Sigma function result | 32 bits (XOR/rotate only) | No overflow possible | None required |
| Choice/majority result | 32 bits (bitwise only) | No overflow possible | None required |
| Temporary sum 1 | 33 bits (two 32-bit additions) | Mask intermediate result | After each addition |
| Temporary sum 2 | 34 bits (three 32-bit additions) | Mask intermediate result | After each addition |
| Final working variable | 35 bits (five 32-bit additions) | Mask final result | After complete calculation |

⚠️ **Pitfall: Inconsistent Masking in Complex Expressions**
A common error is applying masking inconsistently in complex expressions. Consider the expression `(a + b + c) & 0xFFFFFFFF`. If `a + b` already exceeds 32 bits, the intermediate result affects the final outcome. Always mask intermediate results: `((a + b) & 0xFFFFFFFF + c) & 0xFFFFFFFF`.

**Length encoding overflow protection** addresses the special case of encoding message lengths as 64-bit big-endian integers. While Python handles large integers naturally, the bit length calculation must account for potential edge cases where the input approaches the theoretical SHA-256 maximum of 2^64 - 1 bits.

The length encoding process converts the message length in bits to a 64-bit big-endian representation. This requires careful handling of the conversion from Python's arbitrary precision integers to the fixed-width format required by the specification.

**Verification mechanisms** provide ongoing assurance that overflow protection is working correctly. The implementation includes checkpoint functions that verify intermediate values against known correct results from reference implementations. These checkpoints catch overflow-related errors early in the development process rather than allowing them to propagate to the final hash output.

| Verification Point | Check Type | Reference Source | Failure Response |
|-------------------|------------|------------------|------------------|
| Word expansion | Compare 64-word schedule | NIST intermediate values | Raise `ComponentInterfaceError` |
| Round calculations | Compare working variables | Reference implementation | Raise `ComponentInterfaceError` |
| Hash state updates | Compare hash state after each block | Known block-by-block results | Raise `ComponentInterfaceError` |
| Final output | Compare complete hash | NIST test vectors | Raise `SHA256ValidationError` |

### Implementation Guidance

The error handling implementation requires careful coordination between validation functions, exception classes, and component interfaces to ensure comprehensive coverage without performance penalties.

**A. Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|------------------|
| Input Validation | Basic type checking with manual encoding | `schema` library with custom validators |
| Error Messages | String formatting with error codes | Structured logging with context objects |
| Overflow Protection | Manual masking operations | `ctypes` for native 32-bit arithmetic |
| Test Validation | Assert statements with custom messages | `pytest` with parameterized test fixtures |

**B. Recommended File Structure:**
```
sha256/
  validation/
    __init__.py              ← validation module exports
    input_validator.py       ← input validation and normalization
    edge_cases.py           ← edge case handling utilities
    overflow_protection.py   ← arithmetic overflow safeguards
    test_vectors.py         ← NIST test vector validation
  exceptions/
    __init__.py             ← exception class exports
    validation_errors.py    ← validation-specific exceptions
    component_errors.py     ← component interface exceptions
  tests/
    test_validation.py      ← validation unit tests
    test_edge_cases.py      ← edge case testing
    test_overflow.py        ← overflow protection tests
```

**C. Infrastructure Starter Code:**

```python
# validation/input_validator.py - Complete validation infrastructure
import sys
from typing import Union, bytes as BytesType

class SHA256ValidationError(Exception):
    """Base exception for SHA-256 validation failures."""
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.error_code = error_code

class InvalidMessageError(SHA256ValidationError):
    """Raised when input message format is invalid."""
    pass

class ComponentInterfaceError(SHA256ValidationError):
    """Raised when component interface contracts are violated."""
    pass

def validate_message_input(message: Union[str, BytesType]) -> BytesType:
    """
    Validate and normalize input message to bytes.
    
    Args:
        message: Input message as string or bytes
        
    Returns:
        Normalized byte sequence ready for hashing
        
    Raises:
        InvalidMessageError: For invalid input types or encoding failures
    """
    # Type validation
    if not isinstance(message, (str, bytes)):
        raise InvalidMessageError(
            f"Input must be str or bytes, got {type(message).__name__}",
            error_code="INVALID_TYPE"
        )
    
    # String to bytes conversion with encoding validation
    if isinstance(message, str):
        try:
            normalized_bytes = message.encode('utf-8')
        except UnicodeEncodeError as e:
            raise InvalidMessageError(
                f"Failed to encode string as UTF-8: {e}",
                error_code="ENCODING_ERROR"
            )
    else:
        normalized_bytes = message
    
    # Size limit validation (2^61 bytes maximum for practical purposes)
    max_bytes = 2 ** 61
    if len(normalized_bytes) > max_bytes:
        raise InvalidMessageError(
            f"Input too large: {len(normalized_bytes)} bytes (max {max_bytes})",
            error_code="SIZE_LIMIT_EXCEEDED"
        )
    
    return normalized_bytes

# validation/overflow_protection.py - Complete overflow protection utilities
def mask_32_bits(value: int) -> int:
    """
    Mask integer value to 32-bit unsigned range.
    
    Args:
        value: Integer value of any size
        
    Returns:
        Value masked to range [0, 2^32 - 1]
    """
    return value & 0xFFFFFFFF

def safe_add_32(*values: int) -> int:
    """
    Add multiple 32-bit values with overflow protection.
    
    Args:
        *values: Variable number of integer values
        
    Returns:
        Sum masked to 32-bit range
    """
    result = 0
    for value in values:
        result = mask_32_bits(result + mask_32_bits(value))
    return result

def validate_32_bit_value(value: int, name: str = "value") -> int:
    """
    Validate that a value fits in 32-bit unsigned range.
    
    Args:
        value: Integer value to validate
        name: Name of value for error messages
        
    Returns:
        Original value if valid
        
    Raises:
        ComponentInterfaceError: If value exceeds 32-bit range
    """
    if not (0 <= value <= 0xFFFFFFFF):
        raise ComponentInterfaceError(
            f"{name} {value:x} exceeds 32-bit range [0, 0xFFFFFFFF]",
            error_code="VALUE_OUT_OF_RANGE"
        )
    return value
```

**D. Core Logic Skeleton Code:**

```python
# validation/edge_cases.py - Edge case handling implementation
def handle_empty_input() -> bytes:
    """
    Handle empty input edge case.
    
    Returns:
        Empty byte sequence for standard processing
    """
    # TODO 1: Return empty bytes object b''
    # TODO 2: Ensure this flows through standard padding algorithm
    # TODO 3: Verify result matches NIST empty string hash
    pass

def calculate_padding_requirements(message_bit_length: int) -> tuple[int, int]:
    """
    Calculate padding requirements for message of given bit length.
    
    Args:
        message_bit_length: Length of message in bits
        
    Returns:
        Tuple of (zero_padding_bits, total_blocks_needed)
    """
    # TODO 1: Add 1 for mandatory padding bit
    # TODO 2: Calculate bits used in final block (including length field)
    # TODO 3: Determine if additional block needed for length encoding
    # TODO 4: Calculate exact zero padding required
    # TODO 5: Return zero padding count and total block count
    pass

def validate_component_boundary(component_name: str, data_type: str, data) -> None:
    """
    Validate data integrity at component boundaries.
    
    Args:
        component_name: Name of component for error messages
        data_type: Expected data type description
        data: Data to validate
        
    Raises:
        ComponentInterfaceError: If data doesn't match expected format
    """
    # TODO 1: Check data type matches expected type
    # TODO 2: Validate data structure (list lengths, value ranges)
    # TODO 3: Verify all values are properly masked to required bit widths
    # TODO 4: Raise detailed error with component and data type information
    pass

# validation/test_vectors.py - NIST test vector validation
NIST_TEST_VECTORS = {
    "empty": {
        "input": "",
        "expected": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    "abc": {
        "input": "abc", 
        "expected": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    },
    "448_bits": {
        "input": "a" * 56,  # Exactly 448 bits
        "expected": "b35439a8ac6e3fb28d67c33a59d2031f7560f93b5efb2d5cb45b9ddab2009b6c"
    }
}

def run_all_tests() -> bool:
    """
    Execute all NIST test vectors for validation.
    
    Returns:
        True if all tests pass, False otherwise
    """
    # TODO 1: Import main hash_message function
    # TODO 2: Iterate through all test vectors in NIST_TEST_VECTORS
    # TODO 3: Call hash_message with test input
    # TODO 4: Compare result with expected hash value
    # TODO 5: Log detailed results for any failures
    # TODO 6: Return overall pass/fail status
    pass
```

**E. Language-Specific Hints:**

- Use `int.bit_length()` to calculate message length in bits efficiently
- Python's `bytes` objects are immutable - use `bytearray` for building padded messages
- The `struct` module provides `>Q` format for 64-bit big-endian integers
- Use `hex()` function with string slicing to format hash outputs: `hex(value)[2:].zfill(8)`
- Python's `&` operator performs bitwise AND for masking operations
- Use `isinstance(obj, (str, bytes))` to check for multiple acceptable types

**F. Milestone Checkpoints:**

| Milestone | Checkpoint Command | Expected Behavior | Failure Indicators |
|-----------|-------------------|-------------------|-------------------|
| All | `python -m pytest tests/test_validation.py` | All validation tests pass | Type errors, encoding failures |
| 1 | `python -c "from sha256 import validate_message_input; print(len(validate_message_input('')))"` | Prints `0` | Exceptions or wrong length |
| 2-4 | `python -c "from sha256 import mask_32_bits; print(hex(mask_32_bits(0x1FFFFFFFF)))"` | Prints `0xffffffff` | Wrong masking result |
| 4 | `python -c "from sha256.validation import run_all_tests; print(run_all_tests())"` | Prints `True` | Hash mismatches with NIST vectors |

**G. Debugging Tips:**

| Symptom | Likely Cause | Diagnosis | Fix |
|---------|-------------|-----------|-----|
| Wrong hash for empty string | Padding calculation error | Check bit length calculation | Verify empty input produces 512-bit padded block |
| Hash varies between runs | Input encoding inconsistency | Print byte representation of input | Use explicit UTF-8 encoding for strings |
| Arithmetic overflow errors | Missing 32-bit masking | Log intermediate calculation values | Add mask_32_bits after each addition |
| Component interface failures | Data type mismatches | Check types at component boundaries | Add validate_component_boundary calls |
| Test vector failures | Multiple possible causes | Run tests individually | Compare intermediate values with reference |


## Testing Strategy and Milestone Validation

> **Milestone(s):** All milestones (1-4) - Comprehensive testing approach ensuring correct implementation at each stage

### Mental Model: Quality Assurance Assembly Line

Think of testing SHA-256 as operating a **quality assurance assembly line** in a precision manufacturing facility. Just as a factory tests components at each assembly stage before final product validation, our testing strategy validates each milestone independently before integration testing. Unit tests act like **individual component inspectors** checking each part's specifications. Integration tests function as **assembly station supervisors** ensuring components work together correctly. NIST compliance testing serves as the **final quality certification** against international standards, like ISO certification for manufactured goods. Each milestone validation checkpoint acts as a **quality gate** - no component advances to the next stage until it passes all required tests. This layered testing approach catches defects early when they're easier and cheaper to fix, rather than discovering them during final assembly when the entire system must be debugged.

### Test Categories

Our testing strategy employs three distinct categories of tests, each targeting different aspects of SHA-256 implementation correctness. These categories work together to provide comprehensive validation coverage from individual function behavior to full system compliance.

#### Unit Tests

Unit tests validate individual functions and components in isolation, focusing on correctness of specific algorithms and edge case handling. These tests execute quickly and provide immediate feedback during development, making them ideal for test-driven development workflows.

| Test Category | Target Components | Validation Focus | Example Test Cases |
|---------------|-------------------|------------------|-------------------|
| Bitwise Operations | `right_rotate_32`, `mask_32_bits` | Bit manipulation correctness | Rotate by 0, 16, 31 positions; overflow masking |
| Sigma Functions | `_sigma0`, `_sigma1`, `upper_sigma_0`, `upper_sigma_1` | Mathematical function accuracy | Known input/output pairs from specification |
| Round Functions | `choice`, `majority` | Boolean logic implementation | Truth table validation for all bit combinations |
| Padding Algorithm | `_apply_padding`, `_calculate_zero_padding_bits` | Boundary condition handling | Empty input, 447-bit input, 448-bit input |
| Word Conversion | `bytes_to_words_32`, `words_32_to_bytes` | Endianness handling | Big-endian conversion with known byte patterns |
| Arithmetic Safety | `safe_add_32`, `validate_32_bit_value` | Overflow protection | Maximum value addition, out-of-range detection |

Each unit test should validate both normal operation and edge cases. For example, the `right_rotate_32` function requires tests for zero rotation (identity operation), full rotation (32 positions returning to original), and rotation amounts exceeding 32 (ensuring modulo behavior). The padding algorithm tests must cover the critical boundary at 447 bits where an additional block is required, as this is a common source of implementation errors.

> **Design Insight**: Unit tests serve as executable documentation for each function's behavior. When a unit test fails, it immediately identifies which specific algorithm or calculation is incorrect, dramatically reducing debugging time compared to integration test failures.

#### Integration Tests

Integration tests validate component interactions and data flow through the complete SHA-256 pipeline. These tests ensure that components communicate correctly and maintain proper state throughout multi-block processing.

| Integration Scope | Components Tested | Validation Approach | Key Scenarios |
|------------------|-------------------|-------------------|---------------|
| Preprocessing Pipeline | `MessagePreprocessor` → Block Parsing | End-to-end message preparation | Single block, multi-block, boundary sizes |
| Schedule Generation | Block Parser → `MessageScheduler` | Block-to-schedule transformation | 16 words expanding to 64 words correctly |
| Compression Pipeline | `MessageScheduler` → `CompressionEngine` | Schedule consumption and state update | Hash state evolution through 64 rounds |
| Multi-Block Processing | All components | State persistence across blocks | Hash state carried between block processing |
| Output Generation | `CompressionEngine` → `OutputFormatter` | Final hash construction | Hash state to hexadecimal conversion |
| Full Pipeline | All components | Complete message processing | End-to-end transformation validation |

Integration tests must validate state management between components. For multi-block messages, the hash state output from processing the first block becomes the initial state for the second block. A common integration bug occurs when components modify shared state incorrectly, causing later blocks to process with corrupted initial values.

The integration test suite should include **state isolation tests** that process multiple messages concurrently or sequentially to ensure no cross-contamination occurs. These tests catch subtle bugs where global state or improperly scoped variables cause one hash operation to interfere with another.

#### Compliance Verification

Compliance verification tests validate the complete SHA-256 implementation against official NIST test vectors and cryptographic requirements. These tests provide definitive proof that the implementation produces correct outputs for known inputs.

| Compliance Category | Test Source | Validation Criteria | Pass/Fail Determination |
|--------------------|-------------|-------------------|------------------------|
| NIST Test Vectors | FIPS 180-4 specification | Exact hash match | Binary string comparison |
| Empty Input Handling | Cryptographic standards | Specific hash for empty string | Known value validation |
| Single Block Messages | NIST short message tests | Messages under 512 bits | Hash output verification |
| Multi-Block Messages | NIST long message tests | Messages over 512 bits | Cross-block state validation |
| Boundary Conditions | Edge case analysis | 447, 448, 512-bit inputs | Padding behavior verification |
| ASCII vs Binary Input | Character encoding tests | UTF-8, binary data handling | Encoding consistency validation |

The `NIST_TEST_VECTORS` constant contains the official test cases that every compliant SHA-256 implementation must pass. These include the empty string (producing hash `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`), the single-character string "a", the string "abc", and longer test messages designed to exercise multi-block processing.

### Milestone Validation Checkpoints

Each milestone has specific validation checkpoints that confirm correct implementation before proceeding to the next stage. These checkpoints provide concrete success criteria and help identify exactly where implementation problems occur.

#### Milestone 1: Message Preprocessing Validation

The message preprocessing milestone focuses on correct padding algorithm implementation and block parsing functionality. Validation confirms that messages are properly formatted for subsequent processing stages.

| Validation Test | Input | Expected Behavior | Success Criteria |
|-----------------|-------|-------------------|------------------|
| Empty String Padding | `""` (empty bytes) | Padding to 64 bytes (512 bits) | Last 8 bytes contain `0x0000000000000000` |
| Single Character Padding | `"a"` (1 byte) | Padding to 64 bytes total | Last 8 bytes contain `0x0000000000000008` |
| 55-Byte Message | 55 'a' characters | Single 64-byte block | Length field shows 440 bits (55 * 8) |
| 56-Byte Message | 56 'a' characters | Two 64-byte blocks | Second block contains only padding and length |
| Block Boundary Test | 64-byte message | Two blocks required | First block is message, second is padding only |
| Multi-Block Parsing | 128-byte message | Three blocks total | Two message blocks plus one padding block |

**Validation Commands:**
```
python -m pytest tests/test_preprocessing.py::test_padding_algorithm -v
python -m pytest tests/test_preprocessing.py::test_block_parsing -v
```

**Expected Checkpoint Output:**
The preprocessing validation should demonstrate correct bit-level padding implementation. For the empty string test, the padded output should be exactly 64 bytes: one `0x80` byte (the '1' padding bit), 55 zero bytes, and 8 bytes containing the original length (0) in big-endian format. The 55-byte test is critical because it represents the largest message that fits in a single block after padding.

> **Critical Checkpoint**: If padding produces incorrect block counts, all subsequent processing will fail. The boundary at 55 bytes (440 bits) is where messages transition from single-block to multi-block processing. Implementation errors here typically manifest as off-by-one errors in block counting.

#### Milestone 2: Message Schedule Validation

Message schedule generation validation confirms correct word expansion from 16 initial words to 64 schedule words using the sigma functions and recurrence relation.

| Validation Test | Input Block | Validation Focus | Expected Verification |
|-----------------|-------------|------------------|---------------------|
| All-Zero Block | 64 zero bytes | Sigma function behavior with zero | Predictable schedule pattern |
| All-Ones Block | 64 bytes of 0xFF | Maximum value handling | Overflow masking verification |
| NIST Test Block | First block from "abc" | Standard test case | Known schedule values |
| Incremental Pattern | Words 0-15 as sequential integers | Recurrence relation correctness | Mathematical sequence validation |
| Single Bit Set | Only bit 0 of word 0 set | Bit propagation through schedule | Avalanche effect demonstration |

**Schedule Generation Validation Process:**
1. Parse the test block into 16 initial 32-bit words using big-endian byte ordering
2. Apply the word expansion algorithm to generate words 16 through 63
3. Verify each generated word against known correct values or mathematical properties
4. Confirm that sigma functions produce expected bit patterns for test inputs
5. Validate that all intermediate calculations use proper 32-bit masking

**Expected Milestone Behavior:**
For the "abc" test case, the first block contains the padded message: "abc" followed by padding. The initial 16 words should match the expected binary representation of this padded block. Words 16-63 should follow the recurrence relation exactly, with each word depending on four previous words through the sigma functions. Any deviation indicates sigma function implementation errors or incorrect recurrence application.

#### Milestone 3: Compression Function Validation

Compression function validation ensures correct implementation of the 64-round compression algorithm, including proper working variable updates and round function calculations.

| Validation Component | Test Approach | Verification Method | Success Indicators |
|--------------------|---------------|-------------------|-------------------|
| Working Variables Initialization | Known hash state input | Variable assignment verification | a-h match input hash values |
| Round Function Implementation | Truth table testing | Exhaustive bit pattern validation | Ch, Maj, Σ functions correct |
| Single Round Execution | Step-by-step round processing | Intermediate value checking | Working variables update correctly |
| 64-Round Compression | Complete block compression | Hash state transformation | Output matches expected values |
| Multiple Block Processing | Sequential block compression | State carryover validation | Hash state properly maintained |

**Round-by-Round Validation:**
The compression function validation should include intermediate value checking for at least the first few rounds and the final few rounds. This allows detection of accumulating errors that might not be obvious in the final output. For the "abc" test case, specific working variable values are known after rounds 1, 2, and 64, providing concrete validation points.

**Compression Function Debugging Checkpoints:**
- After round 1: Variables a-h should match specific expected values
- After round 32: Intermediate hash state should show proper mixing
- After round 64: Final working variables should match known test case results
- Hash state update: Addition of working variables to initial hash state

> **Implementation Insight**: Compression function errors often compound across rounds, making early detection crucial. If round 1 produces incorrect values, all subsequent rounds will be wrong. Conversely, if rounds 1-63 are correct but round 64 fails, the error is likely in the final variable assignment or hash state update logic.

#### Milestone 4: Final Hash Output Validation

Final output validation confirms correct hash construction, format conversion, and NIST test vector compliance. This milestone represents complete SHA-256 implementation validation.

| Test Vector | Input Message | Expected Hash (Hexadecimal) | Validation Purpose |
|-------------|---------------|----------------------------|-------------------|
| Empty String | `""` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | Empty input handling |
| Single Character | `"a"` | `ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb` | Basic single-block processing |
| Three Characters | `"abc"` | `ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad` | Standard test case |
| Longer Message | `"message digest"` | `f7846f55cf23e14eebeab5b4e1550cad5b509e3348fbc4efa3a1413d393cb650` | Multi-block processing |
| Boundary Case | 55 'a' characters | Known specific hash | Single-block boundary |
| Binary Data | Bytes 0x00-0xFF | Known specific hash | Non-ASCII input handling |

**Output Format Validation:**
1. **Hexadecimal String Format**: 64 characters, lowercase, no spaces or delimiters
2. **Binary Byte Array**: 32 bytes in big-endian order matching hex interpretation
3. **Endianness Consistency**: Most significant byte first in both formats
4. **Character Encoding**: Only valid hexadecimal characters (0-9, a-f)

**Final Validation Commands:**
```
python -m pytest tests/test_nist_vectors.py -v
python -m pytest tests/test_output_formats.py -v
python sha256_implementation.py --test-all-vectors
```

**Complete Implementation Checkpoint:**
A fully correct SHA-256 implementation should pass all NIST test vectors without exception. Any failure indicates a fundamental implementation error that must be resolved. The test vector validation should include both automated test execution and manual verification of key test cases.

### NIST Compliance Testing

NIST compliance testing provides the definitive validation that our SHA-256 implementation meets the official cryptographic standard. This testing uses the exact test vectors specified in FIPS 180-4 and validates against the precise requirements of the cryptographic specification.

#### Official Test Vector Categories

The NIST specification provides several categories of test vectors, each designed to validate specific aspects of SHA-256 implementation correctness. Our compliance testing must cover all categories to ensure complete standard conformance.

| Vector Category | Test Count | Message Length Range | Validation Focus |
|----------------|------------|---------------------|------------------|
| Short Messages | 65 vectors | 0 to 55 bytes | Single-block processing and padding |
| Medium Messages | 64 vectors | 56 to 111 bytes | Two-block processing and state transition |
| Long Messages | 100 vectors | 112+ bytes | Multi-block processing and extended operation |
| Boundary Cases | 20 vectors | Specific bit lengths | Edge case and boundary validation |
| Monte Carlo | 100 iterations | Iterative hash chains | Long-term stability and accumulation errors |
| Pseudo-Random | 1000 vectors | Random content | Statistical validation and corner cases |

**Short Message Vector Validation:**
Short message vectors focus on single-block processing where the entire message, padding, and length field fit within 512 bits. These vectors are critical for validating the padding algorithm and basic compression function operation. The most important short message vectors include:

- **Empty string**: Tests padding-only block processing
- **Single bit message**: Tests minimal message with maximum padding
- **447-bit message**: Tests largest single-block message before requiring additional block
- **ASCII alphabet**: Tests common character processing
- **Binary patterns**: Tests non-ASCII byte value handling

**Monte Carlo Test Validation:**
The Monte Carlo test validates long-term implementation stability by performing iterative hash operations where each hash output becomes the input for the next iteration. This test catches subtle errors that accumulate over many operations, such as state corruption or arithmetic overflow issues.

The Monte Carlo test procedure:
1. Start with a known seed message
2. Compute SHA-256 hash of the seed
3. Use the hash output as input for the next iteration
4. Repeat for 100,000 iterations
5. Compare final result to known expected value

Monte Carlo test failures typically indicate:
- **State corruption**: Hash state not properly isolated between operations
- **Overflow errors**: Arithmetic operations not properly masked to 32 bits
- **Memory management**: Improper buffer reuse or initialization
- **Endianness errors**: Byte order inconsistencies in multi-iteration processing

#### Test Vector Implementation Structure

The `TestVectorValidator` component provides structured access to NIST test vectors and automated validation functionality. This component encapsulates all official test data and provides standardized validation methods.

| Validator Method | Purpose | Input Parameters | Return Value |
|-----------------|---------|------------------|--------------|
| `run_all_tests()` | Execute complete NIST validation | None | Boolean success/failure |
| `validate_short_messages()` | Test single-block processing | Vector subset selection | Test results dictionary |
| `validate_long_messages()` | Test multi-block processing | Message length range | Test results dictionary |
| `validate_monte_carlo()` | Test iterative stability | Iteration count | Final hash comparison |
| `validate_custom_vector(input, expected)` | Test specific case | Message, expected hash | Boolean match result |
| `generate_test_report()` | Create detailed results | Test execution results | Formatted report string |

**NIST Test Vector Data Structure:**
```
NIST_TEST_VECTORS = {
    "short_messages": [
        {"input": "", "expected": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"input": "a", "expected": "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"},
        {"input": "abc", "expected": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        # ... additional short message vectors
    ],
    "medium_messages": [
        # 56-111 byte test messages with expected hashes
    ],
    "long_messages": [
        # 112+ byte test messages with expected hashes
    ],
    "monte_carlo": {
        "seed": "initial_seed_message",
        "iterations": 100000,
        "expected": "final_expected_hash_after_iterations"
    }
}
```

#### Compliance Validation Process

The compliance validation process executes systematically through all test vector categories, providing detailed reporting of any failures and comprehensive validation of standard conformance.

**Validation Execution Steps:**
1. **Environment Preparation**: Initialize clean test environment with isolated SHA-256 instance
2. **Vector Loading**: Load all NIST test vectors from specification data files
3. **Category Execution**: Process each test vector category in sequence
4. **Result Comparison**: Compare actual outputs to expected values using binary string comparison
5. **Failure Analysis**: For any failures, capture detailed diagnostic information
6. **Report Generation**: Create comprehensive validation report with pass/fail status

**Failure Diagnosis and Reporting:**
When NIST compliance tests fail, the validator should provide detailed diagnostic information to help identify the specific implementation error:

| Failure Type | Diagnostic Information | Common Causes | Debugging Approach |
|--------------|----------------------|---------------|-------------------|
| Hash Mismatch | Expected vs actual hash values | Algorithm implementation error | Compare intermediate values step-by-step |
| Processing Error | Exception during hash computation | Input validation or type error | Check input preprocessing and validation |
| Format Error | Output format incorrect | Hexadecimal formatting issue | Validate output conversion functions |
| Performance Timeout | Test execution exceeds time limit | Infinite loop or inefficient implementation | Profile execution and check termination conditions |

**Comprehensive Test Execution Command:**
The complete NIST compliance validation should be executable through a single command that processes all test vector categories and provides summary results:

```
Expected Command Output:
NIST SHA-256 Compliance Validation Results:
==========================================
Short Messages:     65/65 PASSED
Medium Messages:    64/64 PASSED  
Long Messages:     100/100 PASSED
Boundary Cases:     20/20 PASSED
Monte Carlo:         1/1 PASSED
Pseudo-Random:    1000/1000 PASSED
==========================================
OVERALL: PASSED - Full NIST Compliance Achieved
```

Any test failure should provide specific diagnostic information:
```
FAILURE in Short Messages - Test Vector 3:
Input: "abc"
Expected: ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
Actual:   ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ac
Error: Final byte mismatch (expected 'ad', got 'ac')
Likely Cause: Final hash value calculation or hexadecimal formatting error
```

> **Compliance Certification**: Passing all NIST test vectors provides definitive proof that the SHA-256 implementation meets the international cryptographic standard. This validation is essential for any implementation intended for security-critical applications or educational demonstration of correct cryptographic implementation.

### Common Pitfalls

⚠️ **Pitfall: Test-Driven Development Abandonment**
Many learners start with good testing practices but abandon them when encountering complex implementation challenges. They begin implementing core logic without maintaining corresponding tests, leading to integration failures that are difficult to debug. This happens because writing tests for bitwise operations and cryptographic functions feels tedious compared to implementing the "interesting" algorithms. However, SHA-256's complexity makes comprehensive testing absolutely critical. Without tests, a single bit error in one component can cascade through the entire system, making the root cause nearly impossible to identify.

**Fix**: Maintain strict test-first discipline, especially for bitwise operations. Write tests for every helper function before implementing it. Use the milestone validation checkpoints as mandatory gates - never proceed to the next milestone until all tests pass completely.

⚠️ **Pitfall: Insufficient Edge Case Coverage**
Learners often test only "happy path" scenarios like the standard "abc" test case while neglecting edge cases that reveal implementation flaws. Common overlooked edge cases include empty input (requires padding-only block), messages exactly at block boundaries (55-56 byte transition), and inputs with unusual bit patterns (all zeros, all ones, alternating patterns). These edge cases frequently expose off-by-one errors in padding calculations, endianness mistakes in word conversion, and overflow issues in arithmetic operations.

**Fix**: Systematically test boundary conditions for every component. Create specific test cases for: empty input, single-bit input, 55-byte input (largest single block), 56-byte input (smallest two-block), and patterns that exercise bit manipulation edge cases. Use property-based testing tools to generate random inputs and validate invariant properties.

⚠️ **Pitfall: Manual Test Vector Validation**
Some learners manually copy-paste expected hash values and visually compare outputs instead of implementing automated test validation. This approach is error-prone and doesn't scale to the hundreds of NIST test vectors required for full compliance validation. Manual validation also misses subtle differences in output formatting, such as uppercase vs lowercase hexadecimal or extra whitespace characters.

**Fix**: Implement automated binary string comparison for all test vectors. Use exact string matching with no tolerance for formatting differences. Store all NIST test vectors in structured data formats (JSON or CSV) that can be programmatically processed. Create helper functions that handle format normalization consistently across all test cases.

⚠️ **Pitfall: Integration Test Gaps**
While unit tests validate individual components and compliance tests validate final outputs, learners often neglect integration testing that validates component interfaces and data flow. Integration bugs frequently occur at component boundaries where data format assumptions differ, such as endianness mismatches between preprocessing and scheduling, or state management errors between compression rounds.

**Fix**: Design integration tests that specifically validate component interfaces. Test data flow through component pairs (preprocessing → scheduling, scheduling → compression) before testing the complete pipeline. Create integration tests that process multiple blocks sequentially to validate state management between blocks.

⚠️ **Pitfall: Debugging Without Intermediate Value Inspection**
When NIST test vectors fail, learners often focus on the final output difference without examining intermediate values throughout the processing pipeline. SHA-256's complexity means errors can occur at any stage, and the final hash difference provides little information about where the error originated. Attempting to debug by changing implementation code without understanding the failure point leads to trial-and-error fixes that often introduce additional bugs.

**Fix**: Implement comprehensive logging and intermediate value inspection throughout the pipeline. Create debug modes that output hash state after each block, working variables after each compression round, and schedule values for each block. Compare these intermediate values against known reference implementations to identify exactly where divergence occurs.

⚠️ **Pitfall: Test Environment Contamination**
Learners sometimes reuse SHA-256 instances or global state between test cases, causing test contamination where one test's execution affects another test's results. This is particularly problematic in SHA-256 implementations that maintain internal state or use mutable data structures. Test contamination leads to intermittent failures that are difficult to reproduce and debug.

**Fix**: Ensure complete test isolation by creating fresh SHA-256 instances for each test case. Reset all global state between tests. Use test frameworks that provide automatic test isolation. Implement explicit state validation that confirms clean initial conditions before each test execution.

### Implementation Guidance

The testing strategy implementation requires careful organization of test code, systematic validation approaches, and robust debugging capabilities. This guidance provides concrete tools and structures for implementing comprehensive SHA-256 validation.

#### Testing Framework Organization

Organize test files to mirror the component structure and provide clear separation between different types of validation:

```
tests/
  unit/
    test_bitwise_operations.py      ← Individual function validation
    test_sigma_functions.py         ← Cryptographic function testing
    test_padding_algorithm.py       ← Preprocessing unit tests
    test_word_expansion.py          ← Message schedule unit tests
    test_compression_rounds.py      ← Compression function unit tests
    test_output_formatting.py       ← Final output unit tests
  integration/
    test_preprocessing_pipeline.py  ← Component interaction tests
    test_multiblock_processing.py   ← Multi-block state management
    test_end_to_end_pipeline.py     ← Complete pipeline validation
  compliance/
    test_nist_vectors.py            ← Official test vector validation
    test_monte_carlo.py             ← Long-term stability testing
    nist_test_data.json             ← NIST test vector data
  utilities/
    test_helpers.py                 ← Common test utilities
    debug_tools.py                  ← Debugging and inspection tools
    vector_loader.py                ← Test vector loading utilities
```

#### Complete Test Infrastructure

**Test Vector Data Structure (nist_test_data.json):**
```python
# Complete NIST test vector data structure
NIST_TEST_VECTORS = {
    "short_messages": [
        {
            "input": "",
            "input_hex": "",
            "expected": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "description": "Empty string test case"
        },
        {
            "input": "a", 
            "input_hex": "61",
            "expected": "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb",
            "description": "Single character test case"
        },
        {
            "input": "abc",
            "input_hex": "616263", 
            "expected": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "description": "Standard three-character test case"
        }
        # Additional test vectors...
    ],
    "medium_messages": [
        # 56-111 byte test cases
    ],
    "boundary_cases": [
        {
            "input": "a" * 55,  # Largest single-block message
            "expected": "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318",
            "description": "55-byte boundary case"
        },
        {
            "input": "a" * 56,  # Smallest two-block message
            "expected": "b35439a123ac6f0c2bce2d7a1b280a75ea729ab5a2e4a6dc1d8b41a8bb7c5e8c",
            "description": "56-byte boundary case"
        }
    ]
}

# Test vector validator implementation
class TestVectorValidator:
    def __init__(self, sha256_implementation):
        self.sha256 = sha256_implementation
        self.test_vectors = NIST_TEST_VECTORS
        self.results = {}
    
    def run_all_tests(self) -> bool:
        """Execute complete NIST compliance validation."""
        # TODO 1: Load all test vector categories from NIST_TEST_VECTORS
        # TODO 2: Execute each category systematically (short, medium, long, boundary)
        # TODO 3: Collect detailed results for each test case
        # TODO 4: Generate comprehensive pass/fail report
        # TODO 5: Return overall success status
        pass
    
    def validate_test_category(self, category_name: str) -> dict:
        """Validate specific test vector category."""
        # TODO 1: Extract test vectors for specified category
        # TODO 2: Initialize result tracking for category
        # TODO 3: Execute each test vector in category
        # TODO 4: Compare actual output to expected hash
        # TODO 5: Record detailed results including failure diagnostics
        pass
    
    def execute_single_test(self, input_data: str, expected_hash: str) -> dict:
        """Execute single test case with detailed validation."""
        # TODO 1: Convert input data to appropriate format (string or bytes)
        # TODO 2: Execute SHA-256 hash computation
        # TODO 3: Compare result to expected hash using exact string match
        # TODO 4: Capture diagnostic information if test fails
        # TODO 5: Return detailed test result dictionary
        pass
```

**Unit Test Template Structure:**
```python
# Complete unit test implementation for bitwise operations
import unittest
from src.sha256_implementation import right_rotate_32, mask_32_bits, safe_add_32

class TestBitwiseOperations(unittest.TestCase):
    
    def test_right_rotate_32_zero_rotation(self):
        """Test identity operation with zero rotation."""
        # TODO 1: Test various input values with zero rotation
        # TODO 2: Verify output equals input for all test values
        # TODO 3: Test boundary values (0, 0xFFFFFFFF, powers of 2)
        pass
    
    def test_right_rotate_32_full_rotation(self):
        """Test full 32-bit rotation returns to original value."""
        # TODO 1: Test rotation by 32 positions equals identity
        # TODO 2: Test with various bit patterns
        # TODO 3: Verify mathematical property: ROTR_32(x, 32) = x
        pass
    
    def test_right_rotate_32_known_patterns(self):
        """Test rotation with known bit patterns."""
        test_cases = [
            (0x80000000, 1, 0x40000000),  # MSB to bit 30
            (0x00000001, 1, 0x80000000),  # LSB to MSB
            (0xF0F0F0F0, 4, 0x0F0F0F0F),  # Nibble shift
            (0x12345678, 8, 0x78123456),  # Byte rotation
        ]
        # TODO 1: Execute each test case systematically
        # TODO 2: Verify actual output matches expected output exactly
        # TODO 3: Add diagnostic output for any failures
        pass
    
    def test_mask_32_bits_overflow_handling(self):
        """Test 32-bit masking with overflow values."""
        # TODO 1: Test values exceeding 32-bit range
        # TODO 2: Verify results are properly masked to 32 bits
        # TODO 3: Test edge cases like 0x100000000 (2^32)
        # TODO 4: Verify mathematical property: result < 2^32
        pass
```

**Integration Test Implementation:**
```python
# Multi-block processing integration test
class TestMultiBlockIntegration(unittest.TestCase):
    
    def setUp(self):
        """Initialize clean test environment for each test."""
        # TODO 1: Create fresh SHA-256 instance
        # TODO 2: Reset any global state or configuration
        # TODO 3: Initialize test data structures
        pass
    
    def test_two_block_processing(self):
        """Test hash state management across two blocks."""
        # TODO 1: Create test message requiring exactly two blocks
        # TODO 2: Process first block and capture intermediate hash state
        # TODO 3: Verify hash state matches expected intermediate values
        # TODO 4: Process second block using intermediate state as input
        # TODO 5: Verify final hash matches expected result
        pass
    
    def test_state_isolation_between_messages(self):
        """Test that processing multiple messages doesn't cause contamination."""
        # TODO 1: Process first message and record result
        # TODO 2: Process second message using same SHA-256 instance
        # TODO 3: Process first message again and verify identical result
        # TODO 4: Verify no state leakage between message processing
        pass
```

#### Milestone Validation Implementation

**Milestone Checkpoint Automation:**
```python
# Automated milestone validation system
class MilestoneValidator:
    def __init__(self, sha256_implementation):
        self.sha256 = sha256_implementation
        self.milestone_results = {}
    
    def validate_milestone_1_preprocessing(self) -> bool:
        """Validate message preprocessing milestone."""
        test_cases = [
            {"input": "", "expected_blocks": 1, "expected_length_bits": 0},
            {"input": "a", "expected_blocks": 1, "expected_length_bits": 8},
            {"input": "a" * 55, "expected_blocks": 1, "expected_length_bits": 440},
            {"input": "a" * 56, "expected_blocks": 2, "expected_length_bits": 448},
        ]
        
        # TODO 1: Execute preprocessing for each test case
        # TODO 2: Verify correct number of blocks generated
        # TODO 3: Verify length encoding in final 8 bytes
        # TODO 4: Verify padding bit placement and zero-fill
        # TODO 5: Record detailed results for milestone checkpoint
        pass
    
    def validate_milestone_2_schedule_generation(self) -> bool:
        """Validate message schedule generation milestone."""
        # TODO 1: Create test blocks with known patterns
        # TODO 2: Generate 64-word schedules for each test block
        # TODO 3: Verify initial 16 words match block content
        # TODO 4: Verify words 16-63 follow recurrence relation
        # TODO 5: Verify sigma functions produce expected bit patterns
        pass
    
    def validate_milestone_3_compression(self) -> bool:
        """Validate compression function milestone."""
        # TODO 1: Initialize working variables with known hash state
        # TODO 2: Execute 64 compression rounds with test schedule
        # TODO 3: Verify intermediate working variable values at key rounds
        # TODO 4: Verify final hash state update is correct
        # TODO 5: Test round functions with truth table validation
        pass
    
    def validate_milestone_4_final_output(self) -> bool:
        """Validate final hash output milestone."""
        # TODO 1: Execute complete SHA-256 on NIST test vectors
        # TODO 2: Verify hexadecimal output format (64 lowercase chars)
        # TODO 3: Verify binary output format (32 bytes big-endian)
        # TODO 4: Verify endianness consistency across formats
        # TODO 5: Validate against all required NIST test cases
        pass
```

#### Debugging and Diagnostic Tools

**Intermediate Value Inspector:**
```python
# Comprehensive debugging tools for SHA-256 implementation
class SHA256Debugger:
    def __init__(self, sha256_implementation):
        self.sha256 = sha256_implementation
        self.debug_log = []
        self.intermediate_values = {}
    
    def enable_detailed_logging(self):
        """Enable comprehensive intermediate value logging."""
        # TODO 1: Configure SHA-256 implementation for debug mode
        # TODO 2: Enable logging at all component boundaries
        # TODO 3: Set up intermediate value capture points
        # TODO 4: Initialize diagnostic data structures
        pass
    
    def compare_intermediate_values(self, reference_implementation):
        """Compare intermediate values against reference implementation."""
        # TODO 1: Execute both implementations with identical input
        # TODO 2: Compare hash state after each block processing
        # TODO 3: Compare working variables after key compression rounds
        # TODO 4: Identify first point of divergence between implementations
        # TODO 5: Generate detailed diagnostic report
        pass
    
    def validate_component_interfaces(self):
        """Validate data integrity at all component boundaries."""
        # TODO 1: Check preprocessing output format before scheduling
        # TODO 2: Validate schedule generation input/output consistency
        # TODO 3: Verify compression function state management
        # TODO 4: Validate output formatter input requirements
        pass
```

**Test Execution and Reporting:**
```python
# Comprehensive test execution script
def run_complete_validation():
    """Execute complete SHA-256 validation suite."""
    print("SHA-256 Implementation Validation")
    print("=" * 40)
    
    # TODO 1: Initialize test environment and load test vectors
    # TODO 2: Execute milestone validation checkpoints in sequence
    # TODO 3: Execute unit test suite with detailed reporting
    # TODO 4: Execute integration test suite with state validation
    # TODO 5: Execute NIST compliance testing with all vector categories
    # TODO 6: Generate comprehensive validation report
    # TODO 7: Provide specific debugging guidance for any failures
    
    # Expected output format:
    # Milestone 1 (Preprocessing): PASSED
    # Milestone 2 (Schedule Gen):  PASSED  
    # Milestone 3 (Compression):  PASSED
    # Milestone 4 (Final Output): PASSED
    # Unit Tests:                 45/45 PASSED
    # Integration Tests:          12/12 PASSED
    # NIST Compliance:           1249/1249 PASSED
    # OVERALL STATUS:             FULL COMPLIANCE ACHIEVED
    pass

if __name__ == "__main__":
    run_complete_validation()
```

This comprehensive testing strategy ensures systematic validation at every implementation stage, providing concrete checkpoints for learning progress and definitive validation of cryptographic correctness through official NIST compliance testing.

![Testing and Validation Flow](./diagrams/testing-validation-flow.svg)


## Debugging Guide

> **Milestone(s):** All milestones (1-4) - Comprehensive debugging support for implementation issues across all components

### Mental Model: Medical Diagnosis

Think of debugging SHA-256 implementation as **medical diagnosis**. Just as a doctor uses symptoms to identify underlying conditions and prescribe treatments, debugging follows a systematic approach: observe symptoms (wrong outputs, crashes, test failures), diagnose root causes (padding errors, bit operation mistakes, state corruption), and apply targeted fixes. Like medical diagnosis, effective debugging requires understanding both the normal "healthy" operation of each component and the specific ways each can malfunction.

The SHA-256 debugging process mirrors medical triage: first identify which component is affected (preprocessing, scheduling, compression, or output), then narrow down to specific functions within that component, and finally examine the precise bit-level operations where errors occur. Just as doctors use diagnostic tools (X-rays, blood tests), SHA-256 debugging relies on intermediate value inspection, state logging, and systematic test vector validation.

### Symptom-Cause-Fix Reference

The following structured approach provides systematic troubleshooting for common SHA-256 implementation failures. Each entry follows the medical diagnosis model: symptom observation, root cause analysis, diagnostic verification, and targeted treatment.

| Symptom | Root Cause | Diagnostic Steps | Fix Implementation |
|---------|------------|------------------|-------------------|
| Wrong hash for empty string | Incorrect empty input handling or padding boundary calculation | 1. Check if `handle_empty_input()` returns correct zero-length bytes. 2. Verify padding adds exactly 448 zero bits plus 64-bit length encoding. 3. Confirm length encoding shows 0x0000000000000000. | Implement proper empty input detection and ensure padding algorithm handles zero-length messages correctly |
| Hash incorrect for "abc" test vector | Endianness errors in preprocessing or output formatting | 1. Log intermediate values after each preprocessing step. 2. Check byte order in `bytes_to_words_32()`. 3. Verify big-endian conversion in `format_hash_hex()`. | Fix endianness conversion: use big-endian throughout, especially in length encoding and final output |
| Padding length calculation wrong | Off-by-one errors in bit vs byte conversion | 1. Calculate expected padding manually. 2. Log `message_bit_length` and `zero_padding_bits`. 3. Verify 448 mod 512 boundary alignment. | Ensure bit length calculations account for the mandatory '1' padding bit before zero padding |
| Message schedule values incorrect | Sigma function implementation errors | 1. Test `_sigma0()` and `_sigma1()` with known values. 2. Verify right rotate vs right shift operations. 3. Check XOR parameter order in sigma functions. | Implement sigma functions with correct ROTR amounts: σ₀(ROTR7,ROTR18,SHR3), σ₁(ROTR17,ROTR19,SHR10) |
| Word expansion produces wrong values | 32-bit overflow not properly masked | 1. Log intermediate addition results in schedule generation. 2. Check if values exceed 32-bit range before masking. 3. Verify `mask_32_bits()` implementation. | Apply 32-bit masking after every arithmetic operation using `value & 0xFFFFFFFF` |
| Compression rounds fail test vectors | Round function implementation errors | 1. Test `choice()`, `majority()`, `upper_sigma_0()`, `upper_sigma_1()` individually. 2. Verify round constant array values. 3. Check working variable update order. | Fix round functions: Ch(x,y,z) = (x&y)⊕(~x&z), Maj(x,y,z) = (x&y)⊕(x&z)⊕(y&z) |
| Hash state corruption between blocks | State not properly carried forward | 1. Log hash state before and after each block. 2. Check if working variables reset correctly. 3. Verify davies-meyer feedforward addition. | Ensure hash state updates by adding compressed working variables to previous hash state |
| Constant-time violations in production | Non-constant time operations | 1. Profile execution time with different inputs. 2. Check for conditional operations based on data. 3. Verify uniform memory access patterns. | Implement constant-time bit operations and avoid data-dependent branches |
| Memory corruption with large inputs | Buffer overflow in preprocessing | 1. Test with messages near block boundaries. 2. Check array bounds in block parsing. 3. Verify memory allocation for large messages. | Implement proper bounds checking and safe memory allocation for variable-length inputs |
| Test vector failures on specific platforms | Platform-specific integer behavior | 1. Check sizeof(int) on target platform. 2. Verify unsigned integer arithmetic behavior. 3. Test endianness detection. | Use explicit 32-bit types (uint32_t) and platform-independent bit operations |

⚠️ **Pitfall: Debugging Without Reference Values**

Many implementers attempt to debug by staring at hexadecimal outputs without reference points. This is like trying to diagnose an illness without knowing what healthy vital signs look like. Always debug with known intermediate values from NIST test vectors or reference implementations. The NIST specification includes detailed intermediate values for the "abc" test case, providing checkpoints after each major component.

⚠️ **Pitfall: Fixing Symptoms Instead of Root Causes**

A common debugging mistake is applying band-aid fixes that make tests pass without addressing underlying issues. For example, hard-coding correct outputs for specific test cases or manually adjusting intermediate values. This approach fails when new test cases are introduced. Always identify and fix the algorithmic error causing the symptom.

### SHA-256 Specific Debugging Techniques

SHA-256 debugging requires specialized techniques that account for the algorithm's bitwise operations, multi-stage pipeline, and cryptographic properties. These techniques leverage the deterministic nature of cryptographic functions and the availability of official test vectors.

**Intermediate Value Logging Strategy**

The most effective SHA-256 debugging technique is systematic intermediate value logging at each pipeline stage. Unlike general software debugging, cryptographic debugging benefits from complete state visibility because any single bit error propagates through subsequent stages due to the avalanche effect.

The logging strategy follows the four-stage pipeline:

1. **Preprocessing Stage Logging**: Log the original message bytes, padded message with bit padding visible, length encoding as 64-bit big-endian, and final parsed blocks as 16 32-bit words each. This catches endianness errors, padding boundary mistakes, and length calculation errors early in the pipeline.

2. **Schedule Generation Logging**: Log the initial 16 words from each block, intermediate sigma function results for word expansion, and the complete 64-word schedule. This isolates sigma function implementation errors and 32-bit overflow issues during word expansion.

3. **Compression Round Logging**: Log working variables (a through h) after each of the 64 rounds, intermediate round function results (choice, majority, upper-sigma values), and the schedule word and round constant used in each round. This detailed logging identifies specific rounds where corruption occurs.

4. **Output Generation Logging**: Log the final hash state before formatting, intermediate steps in hex conversion, and byte-level output formatting. This catches endianness errors in the final output stage.

**Test Vector Comparison Methodology**

Effective SHA-256 debugging leverages the comprehensive NIST test vectors that provide not only input/output pairs but also intermediate values for key test cases. The "abc" test vector is particularly valuable because NIST provides intermediate values after each major stage.

The comparison methodology involves implementing a reference value checker that compares intermediate values against known correct values at each stage:

| Stage | Reference Checkpoint | Validation Method |
|-------|---------------------|-------------------|
| Preprocessing | "abc" padded to 512 bits | Compare byte arrays after padding |
| Initial Words | First 16 32-bit words from "abc" | Word-by-word comparison |
| Schedule Generation | Complete 64-word schedule | Array comparison with tolerance for endianness |
| Round 1 State | Working variables after first compression round | 8-value state comparison |
| Round 64 State | Working variables after final compression round | Final state validation |
| Hash Output | Expected 256-bit hash for "abc" | String comparison of hex output |

**Bitwise Operation Verification**

SHA-256's reliance on bitwise operations requires specialized debugging techniques for rotate, shift, and XOR operations. Standard debuggers often display integers in decimal format, making bit-level errors difficult to identify.

Effective bitwise debugging involves implementing verification functions that test each bitwise operation with known input/output pairs:

The sigma function verification tests each component operation separately. For example, verifying `_sigma0(x)` by testing the three component operations (ROTR 7, ROTR 18, SHR 3) individually before combining with XOR. This isolates whether the error lies in the rotate implementation, shift implementation, or XOR combination.

Round function verification tests the choice, majority, and upper-sigma functions with carefully chosen test values that exercise all bit positions. For example, testing the choice function with inputs where x has alternating 1/0 bits allows verification that the conditional selection works correctly for each bit position.

**State Evolution Tracking**

SHA-256's hash state evolves predictably through multiple blocks, and tracking this evolution helps identify state corruption issues. The debugging technique involves maintaining a complete history of hash state changes and verifying the davies-meyer construction at each block boundary.

State evolution tracking logs the hash state initialization with `INITIAL_HASH_VALUES`, the state after processing each 512-bit block, and the davies-meyer feedforward addition that combines compressed working variables with the previous hash state. This tracking quickly identifies whether errors occur within block processing or at block boundaries.

### State Inspection Tools

Comprehensive state inspection tools provide the instrumentation needed for effective SHA-256 debugging. These tools go beyond simple logging to offer interactive state examination, automated validation, and systematic component testing.

**SHA-256 Debug Console Implementation**

The debug console provides interactive access to SHA-256 internal state and intermediate values during execution. Unlike static logging, the console allows dynamic inspection and what-if analysis during debugging sessions.

| Debug Console Command | Purpose | Usage Example |
|----------------------|---------|---------------|
| `inspect_hash_state()` | Display current hash state values | Shows all 8 32-bit hash values in hex format |
| `log_schedule_generation(block)` | Trace schedule generation step-by-step | Shows word expansion for specific block |
| `compare_round_state(round_num)` | Compare working variables against reference | Validates state after specific compression round |
| `trace_preprocessing(message)` | Step through padding algorithm | Shows each step of message preprocessing |
| `validate_sigma_functions(test_value)` | Test sigma functions with known inputs | Verifies bitwise operation implementations |
| `dump_component_state(component)` | Export complete component state | Saves internal state for external analysis |
| `run_partial_hash(stop_point)` | Execute hash up to specific stage | Allows testing individual pipeline stages |
| `inject_test_state(state_data)` | Load known state for testing | Enables testing from specific intermediate points |

**Automated Validation Framework**

The validation framework provides systematic verification of component behavior against known correct values. This framework automates the tedious process of manual state inspection by implementing comprehensive validation suites for each component.

```
Component Validation Framework:
├── PreprocessingValidator
│   ├── validate_padding_algorithm()
│   ├── validate_length_encoding()
│   └── validate_block_parsing()
├── ScheduleValidator
│   ├── validate_sigma_functions()
│   ├── validate_word_expansion()
│   └── validate_complete_schedule()
├── CompressionValidator
│   ├── validate_round_functions()
│   ├── validate_working_variables()
│   └── validate_state_evolution()
└── OutputValidator
    ├── validate_hash_formatting()
    ├── validate_endianness_conversion()
    └── validate_final_output()
```

Each validator implements both positive testing (known good inputs produce expected outputs) and negative testing (known bad inputs trigger appropriate error handling). The validators also implement boundary condition testing for edge cases like empty inputs, single-bit messages, and maximum-length inputs.

**Runtime State Monitoring**

Runtime state monitoring provides continuous validation during hash computation, catching errors as they occur rather than after final output comparison. This monitoring is particularly valuable for identifying the exact point where errors are introduced.

The monitoring system implements watchpoints on critical data structures and operations:

| Monitored Element | Validation Check | Error Detection |
|-------------------|------------------|-----------------|
| Hash State Values | Range validation (32-bit unsigned) | Detects overflow or corruption |
| Working Variables | State transition validation | Catches invalid variable updates |
| Schedule Words | 32-bit boundary verification | Identifies overflow in word expansion |
| Block Boundaries | Proper state carryover | Detects state corruption between blocks |
| Bitwise Operations | Input/output relationship validation | Catches rotate/shift implementation errors |
| Memory Buffers | Bounds checking and initialization | Prevents buffer overflows and uninitialized data |

**Component Interface Boundary Checking**

Interface boundary checking validates that data flowing between SHA-256 components maintains proper format and integrity. This checking catches errors that occur at component interfaces rather than within component implementations.

The boundary checking system validates every data transfer between components:

From `MessagePreprocessor` to `MessageScheduler`: Validates that message blocks contain exactly 16 32-bit words in proper big-endian format. Checks that block count matches expected value based on original message length.

From `MessageScheduler` to `CompressionEngine`: Validates that message schedules contain exactly 64 32-bit words within proper range. Verifies that sigma function outputs are properly masked to 32 bits.

From `CompressionEngine` to `OutputFormatter`: Validates that hash state contains exactly 8 32-bit words representing valid hash values. Confirms that working variables were properly added back to hash state.

**State Persistence and Replay**

State persistence allows capturing complete SHA-256 internal state at any point during execution for later replay and analysis. This capability is essential for reproducing intermittent errors and sharing debugging information between developers.

The persistence system captures snapshots of complete internal state including hash state values, working variables, current message block, schedule generation progress, and compression round state. These snapshots can be saved to files and later loaded for replay debugging sessions.

Replay debugging allows starting execution from any captured state point, enabling focused debugging on specific components or operations without re-executing the entire hash computation from the beginning.

### Implementation Guidance

The debugging implementation provides comprehensive support for troubleshooting SHA-256 implementations through systematic state inspection, automated validation, and interactive debugging tools.

**Technology Recommendations for Debugging Tools**

| Tool Category | Simple Option | Advanced Option |
|---------------|---------------|-----------------|
| Logging Framework | Python `logging` module with custom formatters | Structured logging with JSON output for analysis |
| State Inspection | Dictionary-based state dumps with pretty printing | Object introspection with automatic validation |
| Test Validation | Simple assertion-based testing with clear error messages | Property-based testing with automatic test case generation |
| Interactive Debugging | Python `pdb` with custom commands for SHA-256 | Custom debug console with domain-specific commands |
| Performance Profiling | Basic timing measurements with `time.time()` | Statistical profiling with confidence intervals |
| Memory Analysis | Manual memory usage tracking | Automatic memory leak detection and reporting |

**Debugging Module File Structure**

```
sha256_project/
├── src/sha256/
│   ├── __init__.py
│   ├── sha256_core.py          ← main implementation
│   ├── preprocessing.py        ← message preprocessing
│   ├── scheduling.py           ← message schedule generation
│   ├── compression.py          ← compression function
│   ├── output.py              ← hash output formatting
│   └── debugging/             ← debugging support
│       ├── __init__.py
│       ├── debugger.py        ← main debugging interface
│       ├── validators.py      ← component validation
│       ├── state_inspector.py ← state inspection tools
│       ├── test_vectors.py    ← NIST test vector data
│       └── logging_config.py  ← logging configuration
├── tests/
│   ├── test_debugging.py      ← debugging tool tests
│   ├── integration/           ← end-to-end validation
│   └── reference_data/        ← known good intermediate values
└── examples/
    ├── debug_session.py       ← interactive debugging example
    └── validation_report.py   ← automated validation example
```

**Complete SHA-256 Debugger Infrastructure**

```python
"""
SHA-256 Debugging Infrastructure
Provides comprehensive debugging support for SHA-256 implementation
"""

import logging
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import hashlib
from enum import Enum

class DebugLevel(Enum):
    """Debug verbosity levels"""
    BASIC = 1      # Component boundaries only
    DETAILED = 2   # Function-level logging
    VERBOSE = 3    # All intermediate values
    TRACE = 4      # Bit-level operation tracing

@dataclass
class DebugState:
    """Complete SHA-256 internal state snapshot"""
    hash_state: List[int]
    working_variables: Optional[List[int]]
    current_block: Optional[List[int]]
    current_schedule: Optional[List[int]]
    round_number: int
    component: str
    stage: str
    
class SHA256Debugger:
    """Main debugging interface for SHA-256 implementation"""
    
    def __init__(self, debug_level: DebugLevel = DebugLevel.DETAILED):
        self.debug_level = debug_level
        self.state_history = []
        self.validation_enabled = True
        self.reference_implementation = hashlib.sha256
        self.setup_logging()
    
    def setup_logging(self):
        """Configure detailed logging for SHA-256 debugging"""
        # TODO: Configure Python logging with custom formatter for hex values
        # TODO: Set up different log levels based on debug_level
        # TODO: Create separate loggers for each component
        # TODO: Configure output format for readable hex dumps
        pass
    
    def capture_state(self, hash_state: List[int], component: str, stage: str, **kwargs):
        """Capture complete internal state snapshot"""
        # TODO: Create DebugState object with current values
        # TODO: Add timestamp and sequence number
        # TODO: Append to state_history
        # TODO: Log state capture if debug_level >= DETAILED
        pass
    
    def validate_component_boundary(self, component_name: str, data_type: str, data: Any):
        """Validate data at component interface boundaries"""
        # TODO: Check data format matches expected type
        # TODO: Validate value ranges (e.g., 32-bit unsigned integers)
        # TODO: Verify array lengths match specifications
        # TODO: Log validation results
        pass
    
    def compare_intermediate_values(self, stage: str, computed_values: Any, expected_values: Any):
        """Compare computed values against reference implementation"""
        # TODO: Compare values with tolerance for floating-point differences
        # TODO: Identify specific differences and log details
        # TODO: Generate detailed comparison report
        # TODO: Return boolean success/failure result
        pass
    
    def inspect_hash_state(self, hash_state: List[int]) -> Dict[str, str]:
        """Inspect and format hash state for debugging"""
        # TODO: Format each 32-bit word as 8-character hex string
        # TODO: Create readable display with word labels (h0-h7)
        # TODO: Calculate overall hash if in final state
        # TODO: Return formatted dictionary for display
        pass
    
    def trace_preprocessing(self, message: Union[str, bytes]) -> Dict[str, Any]:
        """Trace message preprocessing step-by-step"""
        # TODO: Log original message in multiple formats
        # TODO: Trace padding bit addition
        # TODO: Show zero padding calculation
        # TODO: Display length encoding
        # TODO: Show final block parsing
        pass
    
    def validate_sigma_functions(self, test_value: int) -> Dict[str, bool]:
        """Test sigma function implementations with known values"""
        # TODO: Test _sigma0 with test_value
        # TODO: Test _sigma1 with test_value  
        # TODO: Verify against reference implementation
        # TODO: Return pass/fail results for each function
        pass
    
    def run_component_validation(self, component: str) -> Dict[str, Any]:
        """Run comprehensive validation for specific component"""
        # TODO: Load appropriate test vectors for component
        # TODO: Execute component-specific validation tests
        # TODO: Compare against reference values
        # TODO: Generate validation report
        pass

class ComponentValidator:
    """Validates individual SHA-256 components"""
    
    def __init__(self, debugger: SHA256Debugger):
        self.debugger = debugger
        self.nist_test_vectors = self.load_nist_vectors()
    
    def load_nist_vectors(self) -> Dict[str, Any]:
        """Load NIST test vectors and intermediate values"""
        # TODO: Load test vectors from reference_data/
        # TODO: Include intermediate values for "abc" test case
        # TODO: Parse into structured format for easy access
        # TODO: Return dictionary indexed by test case name
        pass
    
    def validate_preprocessing_component(self, preprocessor) -> Dict[str, bool]:
        """Validate message preprocessing implementation"""
        # TODO: Test with empty string input
        # TODO: Test with "abc" input  
        # TODO: Test with boundary cases (447, 448, 449 bit messages)
        # TODO: Verify padding algorithm correctness
        # TODO: Check endianness in length encoding
        pass
    
    def validate_schedule_generation(self, scheduler) -> Dict[str, bool]:
        """Validate message schedule generation"""
        # TODO: Test sigma functions with known inputs/outputs
        # TODO: Validate word expansion for "abc" test case
        # TODO: Check 32-bit masking in arithmetic operations
        # TODO: Verify complete 64-word schedule correctness
        pass
    
    def validate_compression_function(self, compression_engine) -> Dict[str, bool]:
        """Validate main compression function"""
        # TODO: Test individual round functions (Ch, Maj, Σ)
        # TODO: Validate single compression round
        # TODO: Test complete 64-round compression
        # TODO: Verify working variable updates
        pass

class StateInspector:
    """Interactive state inspection tools"""
    
    def __init__(self, debugger: SHA256Debugger):
        self.debugger = debugger
        self.current_state = None
    
    def start_interactive_session(self):
        """Start interactive debugging session"""
        # TODO: Display available commands
        # TODO: Set up command loop with error handling
        # TODO: Provide help system for debugging commands
        # TODO: Allow state loading/saving
        pass
    
    def dump_complete_state(self, filename: str):
        """Export complete internal state to file"""
        # TODO: Serialize all state_history to JSON
        # TODO: Include metadata (timestamp, version, test case)
        # TODO: Write to specified filename
        # TODO: Log export completion
        pass
    
    def load_state_snapshot(self, filename: str) -> DebugState:
        """Load previously saved state snapshot"""
        # TODO: Read JSON file and parse
        # TODO: Validate loaded data format
        # TODO: Reconstruct DebugState object
        # TODO: Return loaded state for analysis
        pass

# NIST Test Vector Data (Complete reference values)
NIST_TEST_VECTORS = {
    "empty_string": {
        "input": "",
        "expected_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "intermediate_values": {
            "padded_length_bits": 512,
            "block_count": 1,
            "initial_schedule_words": [0x80000000, 0x00000000, 0x00000000, 0x00000000,
                                     0x00000000, 0x00000000, 0x00000000, 0x00000000,
                                     0x00000000, 0x00000000, 0x00000000, 0x00000000,
                                     0x00000000, 0x00000000, 0x00000000, 0x00000000]
        }
    },
    "abc": {
        "input": "abc",
        "expected_hash": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "intermediate_values": {
            "padded_length_bits": 512,
            "block_count": 1,
            "initial_schedule_words": [0x61626380, 0x00000000, 0x00000000, 0x00000000,
                                     0x00000000, 0x00000000, 0x00000000, 0x00000000,
                                     0x00000000, 0x00000000, 0x00000000, 0x00000000,
                                     0x00000000, 0x00000000, 0x00000000, 0x00000018]
        }
    }
}
```

**Core Debugging Logic Skeleton**

```python
def enable_detailed_logging():
    """Enable comprehensive SHA-256 operation logging"""
    # TODO: Configure logging level to capture all intermediate values
    # TODO: Set up custom formatters for hex value display
    # TODO: Create separate log files for each component
    # TODO: Enable timestamp and function name logging
    pass

def validate_milestone_implementation(milestone_number: int) -> Dict[str, bool]:
    """Validate specific milestone implementation"""
    # TODO: Load test cases appropriate for milestone
    # TODO: Execute milestone-specific validation tests
    # TODO: Compare results against expected values
    # TODO: Generate milestone completion report
    # TODO: Return pass/fail status for each test
    pass

def debug_hash_computation(message: str, stop_at_component: str = None):
    """Debug complete hash computation with optional stopping point"""
    # TODO: Initialize debugger with VERBOSE level
    # TODO: Process message through each component
    # TODO: Capture state snapshots at each stage
    # TODO: Stop at specified component if provided
    # TODO: Display complete debugging report
    pass

def compare_with_reference(message: str) -> Dict[str, Any]:
    """Compare implementation against Python hashlib reference"""
    # TODO: Compute hash using our implementation
    # TODO: Compute hash using hashlib.sha256
    # TODO: Compare final results
    # TODO: If different, trace through components to find discrepancy
    # TODO: Return detailed comparison report
    pass
```

**Milestone Validation Checkpoints**

After implementing the debugging infrastructure, validate with these specific checkpoints:

1. **Basic Logging Test**: Run `enable_detailed_logging()` followed by `hash_message("abc")`. Verify that log output shows preprocessing, scheduling, compression, and output stages with intermediate hex values displayed.

2. **State Inspection Test**: Use `inspect_hash_state()` with the initial hash values. Verify output shows all 8 hash values (h0 through h7) formatted as readable hex strings.

3. **Component Validation Test**: Run `validate_milestone_implementation(1)` after implementing preprocessing. Verify that all preprocessing tests pass and any failures show specific error details.

4. **Interactive Debugging Test**: Start an interactive session and manually step through processing the "abc" test case. Verify that each stage produces expected intermediate values.

**Language-Specific Debugging Hints**

- Use Python's `logging` module with custom formatters for hex value display: `logging.Formatter('%(message)s - Hash: %(hexvalue)08x')`
- Implement custom `__repr__` methods for SHA-256 data classes to show hex values automatically in debuggers
- Use `dataclasses.asdict()` for easy state serialization to JSON for external analysis tools  
- Leverage Python's `pdb.set_trace()` for interactive debugging at specific components
- Use `hex()` function for readable hex output: `hex(0xdeadbeef)` produces '0xdeadbeef'
- Implement context managers (`with` statements) for automatic state capture and logging

![Testing and Validation Flow](./diagrams/testing-validation-flow.svg)


## Future Extensions

> **Milestone(s):** Post-implementation - Expanding beyond the basic SHA-256 implementation after completing all core milestones (1-4)

### Mental Model: Growing from Prototype to Production

Think of your SHA-256 implementation as a **working prototype car** that successfully gets you from point A to point B. The future extensions are like the path from prototype to production vehicle - adding performance optimizations (turbocharger, aerodynamics), expanding the product line (SUVs, trucks, electric variants), and hardening for real-world conditions (safety features, weatherproofing, security systems). Your educational implementation proves the concept works correctly, but production cryptographic systems require additional considerations for performance, family compatibility, and security hardening.

The progression follows a natural evolution: first you ensure correctness through NIST compliance, then you optimize for performance when processing large volumes of data, then you leverage your SHA-256 foundation to implement related algorithms, and finally you address the security considerations required for production deployment. Each extension builds upon your solid foundation while addressing different real-world requirements.

### Performance Optimization Opportunities

Your educational SHA-256 implementation prioritizes clarity and correctness over speed, but production systems often need to hash large volumes of data efficiently. Performance optimization involves applying well-established techniques that maintain cryptographic correctness while dramatically improving throughput.

> **Decision: Performance vs Clarity Trade-offs**
> - **Context**: Educational implementations optimize for readability while production systems need high throughput for large data volumes
> - **Options Considered**: Keep simple implementation, add performance optimizations as separate module, rewrite optimized version
> - **Decision**: Maintain educational version and create separate optimized implementations
> - **Rationale**: Preserves learning value while demonstrating real-world performance techniques
> - **Consequences**: Enables performance comparison between approaches and maintains both educational and practical value

#### Vectorization and SIMD Instructions

Modern processors provide **Single Instruction, Multiple Data (SIMD)** capabilities that can process multiple hash operations simultaneously. SHA-256's bitwise operations map naturally to vector instructions, enabling significant speedups when processing multiple messages or multiple blocks within a single message.

The vectorization approach involves restructuring your compression function to operate on multiple data streams simultaneously. Instead of processing one 32-bit word through the `choice` function, you process four or eight words using 128-bit or 256-bit SIMD registers. This technique proves particularly effective for applications like blockchain mining or large file hashing where you process many independent inputs.

| Optimization Target | Traditional Approach | Vectorized Approach | Expected Speedup |
|-------------------|---------------------|-------------------|------------------|
| Multiple Messages | Sequential processing one message at a time | Process 4-8 messages in parallel using SIMD | 4x-8x for small messages |
| Block Processing | One compression round per instruction cycle | Multiple blocks processed simultaneously | 2x-4x for large files |
| Bit Rotations | Individual rotate operations per word | Vector rotate operations on multiple words | 3x-6x for rotate-heavy operations |
| XOR Operations | Scalar XOR on 32-bit values | Vector XOR on 128-bit or 256-bit chunks | 4x-8x improvement |

The implementation complexity increases significantly, requiring platform-specific intrinsics (like Intel's SSE/AVX instructions or ARM's NEON) and careful memory alignment management. However, cryptographic libraries like OpenSSL achieve substantial performance gains through these techniques.

#### Lookup Tables and Precomputation

Certain SHA-256 operations can benefit from **precomputed lookup tables** that trade memory for computational speed. While SHA-256's design intentionally avoids large lookup tables (to resist certain cryptanalytic attacks), some optimizations remain beneficial.

The round constants `ROUND_CONSTANTS` array represents the simplest lookup table - precomputing the 64 cube roots rather than calculating them during execution. More advanced optimizations involve precomputing combinations of rotation operations or partial results of the `choice` and `majority` functions.

| Precomputation Target | Memory Cost | Speed Improvement | Security Consideration |
|----------------------|-------------|------------------|----------------------|
| Rotation Combinations | 4KB-16KB tables | 20-30% faster rotations | No security impact |
| Partial Choice/Majority | 64KB-256KB tables | 15-25% faster rounds | Potential cache timing attacks |
| Round Constant Variants | 1KB additional storage | 5-10% overall improvement | No security impact |
| Sigma Function Results | 16KB-64KB tables | 30-40% faster schedule generation | Minimal timing leak risk |

The trade-off involves balancing speed gains against memory usage and potential security vulnerabilities. Large lookup tables can introduce **cache timing attacks** where attackers infer secret information by measuring memory access patterns.

#### Hardware Acceleration Support

Modern processors include dedicated **cryptographic instruction sets** that accelerate SHA-256 computation dramatically. Intel's SHA Extensions and ARM's Cryptography Extensions provide single-cycle instructions for SHA-256 operations.

These hardware accelerations require detecting processor capabilities at runtime and providing fallback implementations for unsupported processors. The performance improvement can be dramatic - often 5x-10x faster than optimized software implementations.

| Hardware Feature | Availability | Performance Gain | Implementation Complexity |
|-----------------|-------------|-----------------|-------------------------|
| Intel SHA Extensions | Intel processors 2013+ | 8x-12x improvement | Moderate - requires runtime detection |
| ARM Crypto Extensions | ARM Cortex-A53+ processors | 6x-10x improvement | Moderate - ARM-specific intrinsics |
| GPU Parallel Processing | Most modern GPUs | 50x-200x for batch processing | High - requires CUDA/OpenCL |
| FPGA Custom Logic | Specialized hardware | 100x-1000x for dedicated applications | Very high - hardware design required |

The implementation strategy involves creating a **runtime capability detection system** that automatically selects the fastest available implementation. Your code structure might include multiple implementation variants with a dispatcher that chooses the optimal version based on detected hardware features.

> **Critical Insight**: Hardware acceleration provides massive performance improvements but introduces platform-specific complexity. Production cryptographic libraries typically maintain multiple implementation paths - pure software, optimized software, and hardware-accelerated versions - with automatic selection based on runtime detection.

#### Memory Access Optimization

SHA-256's memory access patterns can be optimized to improve **cache efficiency** and reduce memory bandwidth requirements. The standard implementation processes data sequentially, but careful attention to memory layout can improve performance substantially.

Cache-friendly implementations focus on **data locality** - ensuring that frequently accessed data resides in fast processor caches rather than slower main memory. This involves structuring your `HashState` and `WorkingVariables` to fit within cache line boundaries and minimizing memory allocations during processing.

| Memory Optimization | Benefit | Implementation Effort | Compatibility Impact |
|-------------------|---------|---------------------|-------------------|
| Cache Line Alignment | 10-20% faster memory access | Low - compiler attributes | No compatibility issues |
| Memory Pool Allocation | Eliminates allocation overhead | Medium - custom allocators | Requires memory management |
| In-place Block Processing | Reduces memory copies | Medium - algorithm restructuring | API changes required |
| Streaming Interface | Constant memory usage | High - state machine redesign | Major API changes |

### Other SHA-2 Family Implementations

Your SHA-256 implementation provides an excellent foundation for implementing the complete **SHA-2 cryptographic family**. These algorithms share core design principles while varying in word size, block size, and output length.

Understanding the relationship between SHA-2 variants helps you appreciate the scalability of cryptographic design and provides practical experience with different security/performance trade-offs.

#### SHA-224 Implementation

**SHA-224** represents the simplest extension of your SHA-256 implementation. It uses identical preprocessing, message scheduling, and compression logic but differs only in initial hash values and output truncation.

The implementation requires minimal changes to your existing SHA-256 code:

| Component | SHA-256 Behavior | SHA-224 Modification | Implementation Effort |
|-----------|-----------------|-------------------|---------------------|
| `INITIAL_HASH_VALUES` | Standard 8 initial values | Different 8 initial values from NIST specification | Trivial - constant array change |
| Message Preprocessing | 512-bit blocks, same padding | Identical preprocessing logic | No changes required |
| Message Schedule | 64-word expansion with sigma functions | Identical schedule generation | No changes required |
| Compression Function | 64 rounds with same round functions | Identical compression logic | No changes required |
| Output Generation | Return all 8 hash words (256 bits) | Return only first 7 words (224 bits) | Trivial - truncate output |

The SHA-224 variant demonstrates how cryptographic standards provide multiple output lengths from the same underlying algorithm. Financial and embedded systems sometimes prefer SHA-224's shorter output for bandwidth or storage constraints while maintaining strong security properties.

> **Design Insight**: SHA-224 proves that cryptographic flexibility doesn't require algorithm complexity. By changing only initialization and output formatting, you create a completely different hash function with different security characteristics.

#### SHA-384 and SHA-512 Implementation

**SHA-384 and SHA-512** represent more substantial extensions that operate on 64-bit words instead of 32-bit words. These implementations require significant changes to your data model but follow identical algorithmic patterns.

The 64-bit variants process 1024-bit message blocks and use 80-round compression functions instead of 64 rounds. However, the fundamental operations remain the same - just scaled to larger word sizes.

| Implementation Aspect | SHA-256/224 Design | SHA-384/512 Design | Architectural Impact |
|----------------------|-------------------|-------------------|-------------------|
| Word Size | 32-bit `Word32` type | 64-bit `Word64` type | All arithmetic operations change |
| Block Size | 512-bit `MessageBlock` | 1024-bit `MessageBlock` | Double the preprocessing complexity |
| Schedule Size | 64 words | 80 words | Extended message schedule generation |
| Compression Rounds | 64 rounds | 80 rounds | Additional round constants and processing |
| Initial Hash Values | 8 32-bit values | 8 64-bit values | Different constant values |
| Round Constants | 64 32-bit constants | 80 64-bit constants | Extended constant array |

The rotation amounts in the **sigma functions** also differ between 32-bit and 64-bit variants. Your `right_rotate_32` function needs a `right_rotate_64` counterpart with different rotation counts specified in the NIST standard.

Creating a **unified implementation architecture** that supports multiple SHA-2 variants requires careful abstraction of word size and algorithm parameters:

| Architectural Component | Design Strategy | Benefits | Implementation Complexity |
|------------------------|----------------|----------|--------------------------|
| Generic Word Operations | Template/generic functions for word size | Single codebase for all variants | High - requires generic programming |
| Algorithm Parameter Tables | Configuration-driven constants and sizes | Easy addition of new variants | Medium - table-driven design |
| Shared Core Logic | Common preprocessing and compression patterns | Reduced code duplication | Medium - careful interface design |
| Variant-Specific Modules | Separate implementation for each algorithm | Simple, clear implementations | Low - independent modules |

> **Decision: Unified vs Separate SHA-2 Implementations**
> - **Context**: SHA-2 family shares algorithmic patterns but differs in word sizes and parameters
> - **Options Considered**: Single generic implementation, separate implementations per variant, hybrid approach
> - **Decision**: Separate implementations sharing common utility functions
> - **Rationale**: Maintains clarity for learning while reusing well-tested components like padding and output formatting
> - **Consequences**: Easier to understand and debug, some code duplication, simpler testing strategy

#### Implementation Strategy for SHA-2 Family

The recommended approach involves creating a **shared infrastructure layer** that handles common operations while maintaining separate algorithm implementations for clarity.

Your existing SHA-256 implementation provides these reusable components:

| Reusable Component | SHA-2 Family Usage | Required Modifications | Sharing Strategy |
|-------------------|-------------------|----------------------|-----------------|
| Input Validation | Identical across all variants | None | Direct reuse |
| Hex Output Formatting | Same formatting logic | Support variable output lengths | Parameterized function |
| Test Vector Validation | Same validation approach | Different expected outputs | Test data tables |
| Endianness Handling | Same big-endian requirements | Support 64-bit words | Generic conversion functions |
| Debugging Infrastructure | Same debugging needs | Support different word sizes | Parameterized logging |

The algorithmic components require variant-specific implementations:

| Algorithm Component | Customization Required | Implementation Approach |
|-------------------|----------------------|------------------------|
| Round Constants | Different values and counts per variant | Constant arrays per algorithm |
| Initial Hash Values | Different values per variant | Initialization functions per algorithm |
| Sigma Functions | Different rotation amounts | Variant-specific function implementations |
| Message Scheduling | Different schedule sizes | Algorithm-specific schedule generators |
| Compression Rounds | Different round counts | Variant-specific compression functions |

### Production Security Considerations

Your educational SHA-256 implementation focuses on correctness and clarity, but production cryptographic systems must address additional security threats that don't affect algorithm correctness but can leak sensitive information through implementation details.

Production security involves protecting against **side-channel attacks** that infer secret information by observing timing, power consumption, electromagnetic emissions, or cache access patterns during cryptographic operations.

> **Security Context**: Implementation attacks represent a different threat model than cryptanalytic attacks. While cryptanalysis targets mathematical properties of SHA-256 itself, implementation attacks exploit information leaked through physical characteristics of computation.

#### Side-Channel Protection and Constant-Time Operations

**Timing attacks** represent the most practical side-channel threat for hash functions. If your SHA-256 implementation's execution time depends on input content, attackers might infer information about secret data being hashed.

Your current implementation likely contains several timing vulnerabilities:

| Potential Timing Leak | Current Implementation Behavior | Attack Scenario | Mitigation Strategy |
|----------------------|------------------------------|----------------|-------------------|
| Input Length Processing | Variable padding computation time | Infer approximate message lengths | Constant-time padding calculation |
| Early Termination Paths | Different execution for edge cases | Distinguish empty vs normal inputs | Eliminate conditional branches |
| Memory Access Patterns | Sequential processing of message blocks | Infer block boundaries | Uniform memory access patterns |
| Optimization Shortcuts | Skip computation for zero values | Infer partial input content | Always execute full computation |

**Constant-time implementation** requires eliminating all data-dependent branches, memory accesses, and computation shortcuts. This often conflicts with performance optimization but becomes critical when hashing secret or sensitive data.

The transformation involves replacing conditional operations with bitwise operations that always execute the same instructions regardless of data values:

| Operation Type | Time-Variable Implementation | Constant-Time Equivalent | Performance Impact |
|---------------|----------------------------|-------------------------|------------------|
| Conditional Assignment | `if (condition) x = a; else x = b;` | `x = (condition & a) | (~condition & b);` | Minimal overhead |
| Early Loop Exit | `for (i=0; i<n && condition; i++)` | `for (i=0; i<MAX_N; i++) { ... }` | Worst-case time always |
| Table Lookups | `result = table[secret_index];` | Scan entire table with masking | Significant overhead |
| Memory Allocation | Variable-size based on input | Fixed maximum-size buffers | Memory overhead |

#### Memory Protection and Sensitive Data Handling

Production cryptographic implementations must carefully manage **sensitive data in memory** to prevent information leakage through memory dumps, swap files, or memory reuse by other processes.

Your educational implementation stores intermediate values in regular program variables, but production systems require explicit memory protection:

| Memory Security Concern | Risk | Protection Mechanism | Implementation Complexity |
|------------------------|-----|---------------------|-------------------------|
| Sensitive Data in Swap | Operating system writes memory to disk | Mark pages non-swappable (`mlock`) | Medium - OS-specific APIs |
| Memory Dumps in Crashes | Debug dumps contain sensitive values | Zero memory before deallocation | Low - explicit clearing |
| Memory Reuse by Other Processes | Previous data visible to new allocations | Secure memory allocators | Medium - custom allocators |
| Compiler Optimizations | Dead code elimination removes clearing | Volatile memory barriers | Low - compiler attributes |

The implementation strategy involves creating a **secure memory management layer** that ensures sensitive data never persists longer than necessary:

| Protection Layer | Responsibility | Implementation Requirements |
|-----------------|---------------|----------------------------|
| Secure Allocation | Obtain non-swappable memory | OS-specific memory locking APIs |
| Automatic Clearing | Zero sensitive data on scope exit | RAII patterns or explicit cleanup |
| Compiler Barriers | Prevent optimization of security-critical operations | Memory barriers and volatile annotations |
| Access Control | Restrict memory access to authorized code | Memory protection and process isolation |

> **Critical Security Principle**: Assume that all memory containing sensitive data will eventually be accessed by attackers through memory dumps, debugging, or process inspection. Design your memory management to minimize sensitive data lifetime and ensure reliable cleanup.

#### Fault Injection and Physical Attack Resistance

Advanced attackers can manipulate cryptographic computations by inducing **hardware faults** through voltage glitching, electromagnetic interference, or thermal manipulation. While these attacks typically target embedded systems and smart cards, understanding the principles helps design robust implementations.

Fault-resistant SHA-256 implementation involves redundancy and integrity checking that detects and responds to computational errors:

| Fault Attack Vector | Attack Method | Detection Strategy | Response Action |
|-------------------|--------------|------------------|----------------|
| Computation Corruption | Voltage glitching during arithmetic | Duplicate computation with comparison | Halt on mismatch |
| Memory Corruption | Electromagnetic interference | Error-correcting codes or checksums | Retry with fresh data |
| Control Flow Modification | Instruction skip attacks | Control flow integrity checking | Secure failure mode |
| Clock Manipulation | Frequency changes during computation | Timing verification | Reset and retry |

The protection mechanisms add substantial overhead but become necessary for high-security applications:

| Protection Mechanism | Security Benefit | Performance Cost | Implementation Effort |
|---------------------|-----------------|----------------|---------------------|
| Dual Computation | Detects single computation faults | 2x processing time | Low - duplicate existing logic |
| Error Correction | Corrects small memory errors | 20-30% memory overhead | High - ECC implementation |
| Control Flow Integrity | Prevents instruction skip attacks | 10-15% performance penalty | Medium - compiler support needed |
| Temporal Redundancy | Repeat operations over time | Variable delay overhead | Medium - state management required |

#### Secure Random Number Generation

While SHA-256 is deterministic and doesn't directly require randomness, production implementations often integrate with **cryptographically secure random number generators** for salting, nonce generation, or other security purposes.

Understanding the integration between deterministic hash functions and random number generation helps design complete cryptographic systems:

| Randomness Application | Purpose | Quality Requirements | Integration Complexity |
|------------------------|---------|-------------------|---------------------|
| Salt Generation | Prevent rainbow table attacks | High entropy, unpredictable | Low - call during initialization |
| Nonce Creation | Ensure unique hash inputs | Non-repeating, sufficient entropy | Medium - state management |
| Key Derivation | Generate encryption keys from passwords | Cryptographic quality randomness | High - security-critical |
| Padding Randomization | Obscure message lengths | Moderate entropy requirements | Low - replace deterministic padding |

### Common Implementation Pitfalls

⚠️ **Pitfall: Premature Performance Optimization**
Many developers attempt to optimize their SHA-256 implementation before verifying correctness against NIST test vectors. Performance optimizations often introduce subtle bugs that break cryptographic compliance. Always establish a working, test-validated implementation before adding optimizations. Create performance-optimized variants as separate modules that can be validated against your reference implementation.

⚠️ **Pitfall: Platform-Specific Code Without Fallbacks**
Hardware acceleration and SIMD optimizations are platform-specific and may not be available on all target systems. Implementations that assume specific processor features will fail on unsupported hardware. Always provide software fallback implementations and runtime capability detection. Test your code on multiple processor architectures to ensure compatibility.

⚠️ **Pitfall: Insecure Memory Management in Security-Critical Applications**
Using standard memory allocation for sensitive cryptographic data creates security vulnerabilities through memory dumps and process inspection. If your SHA-256 implementation will process sensitive data (passwords, keys, personal information), implement secure memory allocation with explicit clearing and non-swappable pages.

⚠️ **Pitfall: Timing Attack Vulnerabilities in Sensitive Contexts**
Standard implementations contain data-dependent timing variations that can leak information about secret inputs. While this doesn't affect typical file hashing applications, it becomes critical when hashing passwords, keys, or other sensitive data. Evaluate your threat model and implement constant-time operations when necessary.

### Implementation Guidance

The future extensions represent advanced topics that build upon your solid SHA-256 foundation. Each extension addresses different real-world requirements while maintaining the cryptographic correctness you've achieved.

#### Technology Recommendations

| Extension Category | Simple Approach | Advanced Approach |
|-------------------|----------------|------------------|
| Performance Optimization | Profile existing code, optimize hotspots | SIMD vectorization with runtime dispatch |
| SHA-2 Family Extensions | Separate implementation per variant | Generic template-based unified implementation |
| Security Hardening | Basic input validation and error handling | Constant-time operations with secure memory management |
| Hardware Acceleration | Software-only with compiler optimizations | Native cryptographic instruction sets with fallbacks |

#### Recommended Extension Structure

Your future extensions can be organized as separate modules that build upon your core SHA-256 implementation:

```
project-root/
  sha256/
    core/                    ← your original implementation
      sha256.py              ← basic SHA-256
      test_vectors.py        ← NIST test validation
    performance/
      simd_sha256.py         ← vectorized implementation
      benchmark.py           ← performance comparison
    family/
      sha224.py              ← SHA-224 implementation
      sha384.py              ← SHA-384 implementation
      sha512.py              ← SHA-512 implementation
      common.py              ← shared utilities
    security/
      constant_time.py       ← side-channel resistant version
      secure_memory.py       ← memory protection utilities
    hardware/
      accelerated.py         ← hardware-optimized versions
      capability_detect.py   ← runtime feature detection
```

#### Performance Optimization Starter Code

```python
"""
SHA-256 Performance Optimization Framework
Provides infrastructure for benchmarking and optimizing SHA-256 implementations.
"""

import time
import statistics
from typing import List, Callable, Dict, Any
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    """Performance measurement results for SHA-256 implementations."""
    implementation_name: str
    messages_per_second: float
    bytes_per_second: float
    average_latency_ns: float
    memory_usage_kb: float

class SHA256Benchmark:
    """Benchmark different SHA-256 implementations for performance comparison."""
    
    def __init__(self):
        self.implementations: Dict[str, Callable[[bytes], str]] = {}
        self.test_data: List[bytes] = []
    
    def register_implementation(self, name: str, hash_func: Callable[[bytes], str]) -> None:
        """Register a SHA-256 implementation for benchmarking."""
        # TODO: Store implementation function with name for testing
        # TODO: Validate that implementation produces correct SHA-256 hashes
        # TODO: Add implementation to benchmark suite
        pass
    
    def generate_test_data(self, sizes: List[int], count: int = 100) -> None:
        """Generate test data of various sizes for comprehensive benchmarking."""
        # TODO: Create test messages of specified sizes
        # TODO: Include empty messages, small messages, and large messages
        # TODO: Generate random content to avoid optimization artifacts
        # TODO: Store test data for repeated benchmarking
        pass
    
    def benchmark_implementation(self, name: str, iterations: int = 1000) -> BenchmarkResult:
        """Benchmark a single implementation with statistical analysis."""
        # TODO: Warm up implementation to stabilize performance
        # TODO: Measure multiple iterations to get statistical validity
        # TODO: Calculate throughput in messages/second and bytes/second
        # TODO: Measure memory usage during execution
        # TODO: Return structured benchmark results
        pass
    
    def compare_implementations(self) -> Dict[str, BenchmarkResult]:
        """Compare all registered implementations and return performance analysis."""
        # TODO: Run benchmarks on all registered implementations
        # TODO: Validate that all implementations produce identical results
        # TODO: Generate performance comparison report
        # TODO: Identify fastest implementation for each message size category
        pass
```

#### SHA-2 Family Extension Skeleton

```python
"""
SHA-2 Family Implementation Framework
Extends SHA-256 foundation to support SHA-224, SHA-384, and SHA-512 variants.
"""

from typing import List, Union, Protocol
from abc import ABC, abstractmethod

class SHA2Variant(ABC):
    """Abstract base class for all SHA-2 family implementations."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this SHA-2 variant."""
        pass
    
    @property
    @abstractmethod
    def output_size_bits(self) -> int:
        """Output size in bits for this variant."""
        pass
    
    @property
    @abstractmethod
    def word_size_bits(self) -> int:
        """Word size in bits (32 for SHA-256/224, 64 for SHA-384/512)."""
        pass
    
    @abstractmethod
    def hash_message(self, message: Union[str, bytes]) -> str:
        """Hash a message and return hexadecimal digest."""
        pass
    
    @abstractmethod
    def hash_bytes(self, data: bytes) -> bytes:
        """Hash data and return binary digest."""
        pass

class SHA224(SHA2Variant):
    """SHA-224 implementation extending SHA-256 logic."""
    
    def __init__(self):
        # TODO: Initialize SHA-224 specific constants
        # TODO: Set initial hash values per NIST specification
        # TODO: Configure output truncation to 224 bits
        pass
    
    @property
    def name(self) -> str:
        return "SHA-224"
    
    @property
    def output_size_bits(self) -> int:
        return 224
    
    @property
    def word_size_bits(self) -> int:
        return 32
    
    def hash_message(self, message: Union[str, bytes]) -> str:
        """Hash message using SHA-224 (truncated SHA-256)."""
        # TODO: Reuse SHA-256 preprocessing logic
        # TODO: Reuse SHA-256 compression function
        # TODO: Use SHA-224 specific initial values
        # TODO: Truncate output to 224 bits (7 words instead of 8)
        pass
    
    def hash_bytes(self, data: bytes) -> bytes:
        """Return SHA-224 hash as binary data."""
        # TODO: Call hash_message and convert hex to binary
        # TODO: Ensure output is exactly 28 bytes (224 bits)
        pass

class SHA2Factory:
    """Factory for creating SHA-2 variant implementations."""
    
    VARIANTS = {
        'sha224': SHA224,
        # TODO: Add SHA384 class when implemented
        # TODO: Add SHA512 class when implemented
    }
    
    @classmethod
    def create(cls, variant_name: str) -> SHA2Variant:
        """Create a SHA-2 variant implementation by name."""
        # TODO: Validate variant name exists
        # TODO: Instantiate and return appropriate implementation
        # TODO: Raise descriptive error for unknown variants
        pass
    
    @classmethod
    def list_variants(cls) -> List[str]:
        """List all available SHA-2 variant names."""
        return list(cls.VARIANTS.keys())
```

#### Security Hardening Infrastructure

```python
"""
SHA-256 Security Hardening Framework
Provides constant-time operations and secure memory management.
"""

import ctypes
import os
from typing import Optional
import weakref

class SecureMemory:
    """Secure memory allocation with automatic clearing and non-swappable pages."""
    
    def __init__(self, size_bytes: int):
        self.size = size_bytes
        self.buffer: Optional[bytearray] = None
        self._locked = False
    
    def allocate(self) -> bytearray:
        """Allocate secure memory with operating system protection."""
        # TODO: Allocate memory using OS-specific APIs
        # TODO: Lock memory pages to prevent swapping (mlock on Unix)
        # TODO: Register cleanup handler for automatic zeroing
        # TODO: Return protected memory buffer
        pass
    
    def clear(self) -> None:
        """Securely clear memory contents to prevent data leakage."""
        # TODO: Zero all bytes in buffer
        # TODO: Use memory barriers to prevent compiler optimization
        # TODO: Unlock memory pages
        # TODO: Mark buffer as cleared
        pass
    
    def __del__(self):
        """Ensure memory is cleared when object is destroyed."""
        if self.buffer is not None:
            self.clear()

class ConstantTimeSHA256:
    """Constant-time SHA-256 implementation resistant to timing attacks."""
    
    def __init__(self):
        self.secure_memory = SecureMemory(1024)  # Buffer for sensitive operations
    
    def constant_time_compare(self, a: bytes, b: bytes) -> bool:
        """Compare two byte arrays in constant time."""
        # TODO: Ensure comparison takes same time regardless of where difference occurs
        # TODO: Use bitwise operations instead of conditional branches
        # TODO: Return boolean result without early termination
        pass
    
    def constant_time_select(self, condition: int, true_val: int, false_val: int) -> int:
        """Select value based on condition in constant time."""
        # TODO: Use bitwise operations to avoid conditional branches
        # TODO: Ensure same execution time regardless of condition value
        # TODO: Return selected value using bit masking
        pass
    
    def secure_hash_message(self, message: Union[str, bytes]) -> str:
        """Hash message with timing attack protection."""
        # TODO: Use secure memory for intermediate values
        # TODO: Implement constant-time operations throughout
        # TODO: Clear all intermediate values after computation
        # TODO: Return hash result with secure cleanup
        pass
```

#### Milestone Validation for Extensions

After implementing performance optimizations:
- **Benchmark Validation**: Your optimized implementation should produce identical results to your reference implementation while showing measurable performance improvements
- **Performance Testing**: Create test cases with various message sizes (1KB, 1MB, 100MB) to measure throughput improvements
- **Compatibility Verification**: Ensure optimizations work across different processor architectures and operating systems

After implementing SHA-2 family extensions:
- **Cross-Validation**: All SHA-2 variants should pass their respective NIST test vectors
- **Shared Component Testing**: Verify that common utilities work correctly across different word sizes
- **Family Consistency**: Ensure consistent API design across all SHA-2 variant implementations

After implementing security hardening:
- **Timing Analysis**: Measure execution time variation across different inputs to verify constant-time behavior
- **Memory Security Testing**: Verify that sensitive data is properly cleared and memory pages are protected
- **Side-Channel Testing**: Use specialized tools to detect potential information leakage through implementation details

#### Language-Specific Extension Hints

**Python Performance Extensions:**
- Use `numpy` arrays for vectorized operations on large datasets
- Consider `Cython` for performance-critical constant-time operations
- Use `ctypes` for integrating with optimized C libraries
- Profile with `cProfile` to identify optimization targets

**Security Implementation:**
- Use `secrets` module for cryptographically secure random number generation
- Consider `mmap` with `MAP_LOCKED` for secure memory allocation on Unix systems
- Use `memoryview` objects to minimize memory copies
- Implement secure comparison using `hmac.compare_digest` as reference

**Cross-Platform Considerations:**
- Test memory protection APIs on different operating systems
- Verify SIMD instruction availability using runtime detection
- Provide graceful fallbacks for unsupported hardware features
- Ensure consistent behavior across Python versions and implementations


## Glossary

> **Milestone(s):** All milestones (1-4) - Essential terminology for understanding SHA-256 implementation across all components

### Mental Model: Technical Reference Library

Think of this glossary as a **specialized technical reference library** for cryptographic hash functions. Just as a medical dictionary provides precise definitions for anatomical terms that might be misunderstood in casual conversation, this glossary ensures that every cryptographic concept, bitwise operation, and SHA-256 specific term has an unambiguous meaning. When you encounter terms like "preimage resistance" or "endianness" in the implementation, you can reference these definitions to understand both the technical meaning and the practical implications for your code.

The glossary serves as the **common vocabulary** that bridges theoretical cryptography with practical implementation details. Each term includes not just its definition, but also context about how it applies specifically to SHA-256 implementation and where it appears in the four-stage processing pipeline.

### Cryptographic Terms

Cryptographic hash functions embody several mathematical properties that make them suitable for security applications. These properties distinguish cryptographic hashes from simple checksums or non-cryptographic hash functions used in data structures.

| Term | Definition | SHA-256 Context | Practical Significance |
|------|------------|-----------------|----------------------|
| **Hash Function** | Mathematical function that maps arbitrary-length input data to fixed-length output values in a deterministic, irreversible manner | SHA-256 maps any input to exactly 256 bits (64 hex characters) | Forms the foundation of digital signatures, password storage, and blockchain mining |
| **Collision Resistance** | Cryptographic property making it computationally infeasible to find two different inputs that produce identical hash outputs | SHA-256 requires approximately 2^128 operations to find a collision | Prevents attackers from substituting malicious data with the same hash value |
| **Preimage Resistance** | Cryptographic property making it computationally infeasible to find any input that produces a specific given hash output | SHA-256 requires approximately 2^256 operations to find a preimage | Enables secure password storage and commitment schemes |
| **Second Preimage Resistance** | Cryptographic property making it computationally infeasible to find a different input that produces the same hash as a given input | SHA-256 requires approximately 2^256 operations to find a second preimage | Protects against targeted forgery attacks on specific documents |
| **Avalanche Effect** | Property where small changes in input cause large, unpredictable changes in output | Changing one bit in SHA-256 input changes approximately half the output bits | Ensures that similar inputs produce completely different hash values |
| **Deterministic** | Property where identical inputs always produce identical outputs | Same message always produces same SHA-256 hash | Enables verification and comparison of hash values across systems |
| **One-Way Function** | Mathematical function that is easy to compute forward but computationally infeasible to reverse | Easy to compute SHA-256 hash from message, infeasible to recover message from hash | Foundation of cryptographic security assumptions |
| **Cryptographic Primitive** | Basic building block used to construct cryptographic protocols and systems | SHA-256 serves as primitive for HMAC, digital signatures, and blockchain protocols | Provides verified security foundation for higher-level constructions |

The **Merkle-Damgård construction** underlies SHA-256's design, processing input in fixed-size blocks with chaining between blocks. This construction ensures that the security properties of the underlying compression function extend to messages of arbitrary length.

> **Design Insight**: The mathematical properties above are not just theoretical concepts—they directly impact implementation requirements. For example, collision resistance requires that your implementation produce bit-identical results to the NIST specification, since even tiny implementation errors could theoretically weaken the security guarantees.

**Comparison with Other Hash Function Families:**

| Property | MD5 | SHA-1 | SHA-256 | SHA-3 |
|----------|-----|-------|---------|--------|
| Output Size | 128 bits | 160 bits | 256 bits | Variable |
| Construction | Merkle-Damgård | Merkle-Damgård | Merkle-Damgård | Sponge |
| Security Status | Broken | Deprecated | Secure | Secure |
| Performance | Fast | Fast | Moderate | Moderate |
| Block Size | 512 bits | 512 bits | 512 bits | Variable |

The **sponge construction** used by SHA-3 represents an alternative to Merkle-Damgård, absorbing input and then squeezing output through a permutation function. However, SHA-256's Merkle-Damgård approach remains the industry standard for most applications.

### Technical Implementation Terms

Implementation of SHA-256 requires precise understanding of low-level computational concepts that affect correctness and security. These terms describe the mechanical aspects of transforming bytes and bits according to the NIST specification.

| Term | Definition | SHA-256 Usage | Implementation Impact |
|------|------------|---------------|---------------------|
| **Endianness** | Byte ordering convention for multi-byte values in memory - big-endian stores most significant byte first | SHA-256 uses big-endian for all multi-byte values including lengths and hash output | Incorrect endianness produces completely wrong hash values |
| **Padding** | Process of extending message to required length using specific bit patterns | SHA-256 appends '1' bit, zero bits for alignment, then 64-bit length field | Must reach exactly 448 mod 512 bits before adding length |
| **Message Block** | Fixed-size chunk of padded message data processed as a unit | SHA-256 processes exactly 512-bit (64-byte) blocks sequentially | Each block updates hash state through compression function |
| **Word** | Fixed-size data unit used in hash computation | SHA-256 uses 32-bit words throughout all operations | All arithmetic and bitwise operations occur on 32-bit boundaries |
| **Compression Function** | Core algorithm that combines message block with current hash state | SHA-256 compression executes 64 rounds of bitwise mixing per block | Heart of the hash algorithm - where security properties emerge |
| **Right Rotation** | Circular bit shift operation preserving all bits by wrapping around | Used in sigma functions - bits shifted off right end wrap to left end | Different from right shift which introduces zeros |
| **Right Shift** | Linear bit shift operation introducing zeros on the left | Used in sigma functions - bits shifted off right end are lost | Combined with rotation in XOR operations for diffusion |
| **Modular Arithmetic** | Arithmetic operations with wraparound at specified boundary | All SHA-256 addition uses modulo 2^32 arithmetic | Prevents integer overflow - values wrap at 32-bit boundary |
| **32-bit Masking** | Operation reducing values to 32-bit unsigned range | Apply bitwise AND with 0xFFFFFFFF after arithmetic operations | Essential for preventing implementation-dependent overflow behavior |

**Bit and Byte Representation Concepts:**

| Concept | Description | SHA-256 Application | Common Pitfalls |
|---------|-------------|-------------------|-----------------|
| **Binary Representation** | Expressing data as sequences of 0 and 1 bits | Message converted to binary before padding and processing | Confusion between bit length and byte length calculations |
| **Hexadecimal Output** | Base-16 representation using digits 0-9 and letters a-f | Final hash displayed as 64-character hex string | Uppercase vs lowercase conventions, leading zero handling |
| **Big-Endian Encoding** | Multi-byte values stored with most significant byte first | Length field, hash values, and all words use big-endian order | Platform-dependent endianness can cause silent failures |
| **Block Alignment** | Ensuring data fits exactly into required block boundaries | Messages padded to multiple of 512 bits for block processing | Off-by-one errors in padding calculations |
| **Length Encoding** | Representing original message length as fixed-size binary value | 64-bit big-endian length appended after zero padding | Must represent bit length, not byte length |

**SHA-256 Specific Algorithm Components:**

The SHA-256 algorithm employs specialized functions and constants that require precise implementation:

| Component | Technical Definition | Mathematical Representation | Implementation Notes |
|-----------|---------------------|----------------------------|-------------------|
| **Lower-Sigma Functions** | Diffusion functions used in message schedule generation | σ₀(x) = ROTR₇(x) ⊕ ROTR₁₈(x) ⊕ SHR₃(x) | Used in word expansion - note different rotation amounts |
| **Upper-Sigma Functions** | Diffusion functions used in compression rounds | Σ₀(x) = ROTR₂(x) ⊕ ROTR₁₃(x) ⊕ ROTR₂₂(x) | Used in main compression - different from lower-sigma |
| **Choice Function** | Bitwise conditional selection based on first argument | Ch(x,y,z) = (x ∧ y) ⊕ (¬x ∧ z) | x chooses between y and z bit-by-bit |
| **Majority Function** | Bitwise voting mechanism selecting most common bit | Maj(x,y,z) = (x ∧ y) ⊕ (x ∧ z) ⊕ (y ∧ z) | Result bit is 1 if majority of input bits are 1 |
| **Word Expansion** | Algorithm extending 16 message words to 64 schedule words | W[i] = σ₁(W[i-2]) + W[i-7] + σ₀(W[i-15]) + W[i-16] | Recurrence relation for i = 16 to 63 |
| **Working Variables** | Eight temporary variables updated during compression rounds | a, b, c, d, e, f, g, h - each 32-bit unsigned integers | Initialized from hash state, updated each round |
| **Round Constants** | 64 predefined constants used in compression rounds | K[0] through K[63] - fractional parts of cube roots of first 64 primes | Must match NIST specification exactly |
| **Initial Hash Values** | Eight starting hash state values | H[0] through H[7] - fractional parts of square roots of first 8 primes | Forms initial state before processing any blocks |

**Data Flow and State Management Terms:**

| Term | Definition | Usage Context | State Relationships |
|------|------------|---------------|-------------------|
| **Hash State** | Current set of eight 32-bit values representing intermediate hash result | Maintained across block processing, updated after each compression | Carries cryptographic state between message blocks |
| **Message Schedule** | Array of 64 32-bit words derived from current message block | Generated fresh for each block from 16 input words | Provides round-specific input for compression function |
| **Processing Pipeline** | Four-stage transformation from input message to final hash | Preprocessing → Schedule → Compression → Output | Each stage has specific input/output contracts |
| **Block Chaining** | Method of incorporating results from previous blocks | Hash state from block N becomes input for block N+1 | Ensures all message content affects final hash |
| **Davies-Meyer Construction** | Feedforward pattern adding block input to compression output | Hash[i+1] = Hash[i] + Compression(Hash[i], Block[i]) | Prevents length extension and strengthens security |

**Test and Validation Terminology:**

| Term | Purpose | NIST Context | Validation Process |
|------|---------|-------------|------------------|
| **Test Vector** | Official input/output pair for implementation verification | NIST provides standard test cases with known results | Compare implementation output to expected values |
| **NIST Compliance** | Conformance to official SHA-256 specification | Implementation must match reference exactly | Required for cryptographic security guarantees |
| **Intermediate Values** | Internal computation states during hash processing | NIST vectors include hash state after each block | Enable debugging of specific algorithm stages |
| **Monte Carlo Testing** | Iterative validation using hash output as next input | Tests stability and absence of weak input patterns | Reveals implementation flaws not caught by simple vectors |
| **Cross-Platform Validation** | Verification across different hardware and software environments | Ensures consistent results regardless of implementation platform | Critical for interoperability |

> **Implementation Warning**: Many of these technical terms have precise mathematical definitions that differ from colloquial usage. For example, "rotation" specifically means circular bit shifting, while "shift" means linear bit shifting with zero fill. Using the wrong operation will produce incorrect results that may not be immediately obvious during testing.

**Error Handling and Edge Case Terminology:**

Understanding failure modes and boundary conditions requires specific vocabulary:

| Term | Scenario | Detection Method | Recovery Approach |
|------|----------|-----------------|------------------|
| **Arithmetic Overflow** | 32-bit addition results exceed representable range | Values larger than 0xFFFFFFFF after operations | Apply 32-bit masking to maintain modular arithmetic |
| **Endianness Mismatch** | Multi-byte values interpreted with wrong byte order | Hash output differs from expected NIST test vectors | Convert all multi-byte values to big-endian explicitly |
| **Padding Miscalculation** | Message length calculation error in preprocessing | Block count doesn't match expected or padding incorrect | Verify bit length calculation and boundary alignment |
| **State Contamination** | Hash state persists between independent hash operations | Second hash operation produces incorrect result | Reset hash state to initial values before each message |
| **Precision Loss** | Intermediate calculations lose bits due to type limitations | Sigma functions or arithmetic produce wrong bit patterns | Use appropriate integer types with explicit bit masking |

### Common Implementation Pitfalls

Understanding where implementations typically fail helps avoid subtle bugs that may not surface in basic testing:

⚠️ **Pitfall: Bit vs Byte Length Confusion**
The most common error involves confusing bit lengths with byte lengths during padding calculations. SHA-256 padding requires the original message length in **bits**, not bytes. A 10-byte message has bit length 80, not 10. The 64-bit length field appended during padding must contain 80 (0x0000000000000050 in big-endian), not 10.

⚠️ **Pitfall: Endianness Inconsistency**
SHA-256 requires big-endian byte ordering throughout, but many platforms default to little-endian. Simply reversing bytes is insufficient—you must understand exactly which values need conversion. The message length field, all 32-bit words during processing, and the final hash output all require big-endian representation.

⚠️ **Pitfall: Function Parameter Confusion**
The sigma functions use different rotation and shift amounts that are easy to mix up. Lower-sigma-0 uses rotations of 7 and 18 bits plus shift of 3 bits, while lower-sigma-1 uses rotations of 17 and 19 bits plus shift of 10 bits. Upper-sigma functions have completely different parameters. Verify each function against the specification independently.

⚠️ **Pitfall: Modular Arithmetic Oversight**
All SHA-256 addition operations use modulo 2^32 arithmetic, meaning results wrap around at the 32-bit boundary. Programming languages handle integer overflow differently—some throw exceptions, others produce undefined behavior. Explicitly mask all arithmetic results to 32 bits using bitwise AND with 0xFFFFFFFF.

⚠️ **Pitfall: State Persistence Between Operations**
Hash objects that process multiple messages must reset their internal state completely between operations. Failing to reset to the initial hash values causes the second hash operation to continue from the final state of the first operation, producing completely incorrect results.

### Implementation Guidance

The terminology defined above forms the foundation for implementing SHA-256 correctly. Understanding these concepts prevents the subtle errors that are difficult to debug and may only surface with specific input patterns.

#### Key Term Usage Patterns

When writing SHA-256 implementation code, certain terms appear in predictable contexts:

**Message Processing Context:**
- Use "preprocessing" when discussing the padding and block parsing stage
- Use "message schedule" when referring to the 64-word array generation
- Use "compression function" when discussing the main hash mixing algorithm
- Use "hash finalization" when discussing output formatting

**Bitwise Operation Context:**
- Use "right rotation" for circular bit shifts that preserve all bits
- Use "right shift" for linear bit shifts that introduce zeros
- Use "32-bit masking" when reducing arithmetic results to valid range
- Use "big-endian encoding" when discussing multi-byte value representation

**Cryptographic Security Context:**
- Use "preimage resistance" when discussing the difficulty of hash inversion
- Use "collision resistance" when discussing the difficulty of finding duplicate inputs
- Use "avalanche effect" when discussing input sensitivity properties
- Use "NIST compliance" when discussing specification conformance requirements

#### Technology Recommendations

| Component | Terminology Focus | Implementation Priority |
|-----------|------------------|------------------------|
| Bit Operations | Right rotation, masking, endianness | Critical - affects correctness |
| Hash State | Working variables, state management | High - affects multiple blocks |
| Test Validation | NIST vectors, compliance testing | High - verifies implementation |
| Error Handling | Overflow protection, edge cases | Medium - affects robustness |

#### Recommended Module Structure

Organize terminology and documentation to support code organization:

```
sha256_implementation/
  docs/
    glossary.md              ← this comprehensive terminology reference
    algorithm_walkthrough.md ← step-by-step process with term definitions
  src/
    constants.py            ← INITIAL_HASH_VALUES, ROUND_CONSTANTS with documentation
    types.py               ← MessageBlock, HashState, WorkingVariables type definitions
    preprocessing.py       ← padding, endianness, block parsing functions
    schedule.py           ← word expansion, sigma functions
    compression.py        ← round functions, main compression algorithm
    output.py            ← hash finalization, format conversion
  tests/
    test_terminology.py   ← verify understanding of key concepts
    test_nist_vectors.py  ← validate against official test cases
```

#### Terminology Validation Utilities

**A. Concept Verification Functions:**

```python
def validate_terminology_understanding():
    """
    Verify understanding of key SHA-256 terminology through interactive tests.
    Tests bit vs byte calculations, endianness conversions, and function differences.
    """
    # TODO: Test bit length calculation for various message sizes
    # TODO: Test endianness conversion for 32-bit values
    # TODO: Test sigma function parameter recognition
    # TODO: Verify understanding of rotation vs shift operations

def explain_term_in_context(term_name, implementation_stage):
    """
    Provide contextual explanation of terminology within specific implementation stages.
    Helps developers understand when and how to apply concepts correctly.
    """
    # TODO: Map term to relevant implementation milestone
    # TODO: Provide stage-specific usage examples
    # TODO: List common mistakes for this term in this context
    # TODO: Reference related terms and concepts
```

**B. Implementation Checkpoint Functions:**

```python
def check_terminology_usage(code_module):
    """
    Analyze code for correct usage of SHA-256 terminology and concepts.
    Identifies potential misunderstanding of cryptographic or technical terms.
    """
    # TODO: Scan for correct bit vs byte handling
    # TODO: Verify endianness consistency
    # TODO: Check sigma function implementations
    # TODO: Validate hash state management

def generate_terminology_report(implementation_status):
    """
    Generate report showing which terminology concepts are correctly implemented.
    Maps implementation progress to terminology understanding.
    """
    # TODO: List implemented concepts with terminology verification
    # TODO: Identify missing concepts based on terminology gaps
    # TODO: Suggest next implementation steps based on terminology mastery
    # TODO: Recommend additional study for weak terminology areas
```

#### Debugging with Terminology

Understanding terminology enables more effective debugging by providing precise vocabulary for describing problems:

| Problem Description | Terminology Used | Debugging Approach |
|-------------------|-----------------|-------------------|
| "Hash output wrong" | NIST compliance, test vector validation, endianness | Compare against official vectors, check byte ordering |
| "Second hash incorrect" | State contamination, hash state management | Verify state reset between operations |
| "Padding seems wrong" | Message preprocessing, block alignment, length encoding | Verify bit length calculation and boundary conditions |
| "Sigma functions broken" | Lower-sigma vs upper-sigma, rotation parameters | Check each function's rotation and shift amounts separately |

#### Milestone Terminology Checkpoints

After completing each implementation milestone, verify terminology understanding:

**Milestone 1 Checkpoint - Preprocessing Terminology:**
- Can you explain the difference between bit length and byte length?
- Do you understand why SHA-256 uses big-endian byte ordering?
- Can you describe the three stages of message padding?
- Do you know what "448 mod 512" means in padding context?

**Milestone 2 Checkpoint - Schedule Generation Terminology:**
- Can you distinguish between lower-sigma-0 and lower-sigma-1 functions?
- Do you understand the difference between right rotation and right shift?
- Can you explain what "word expansion" means in SHA-256 context?
- Do you know why the schedule has exactly 64 words?

**Milestone 3 Checkpoint - Compression Terminology:**
- Can you distinguish between upper-Sigma and lower-sigma functions?
- Do you understand the choice and majority functions?
- Can you explain what "working variables" represent?
- Do you know why there are exactly 64 compression rounds?

**Milestone 4 Checkpoint - Output Terminology:**
- Can you explain what "hash finalization" involves?
- Do you understand the difference between hex string and binary output?
- Can you describe what NIST test vector validation proves?
- Do you know what "deterministic" means for hash functions?

This comprehensive terminology foundation ensures that learners can communicate precisely about their implementation, debug problems effectively, and understand the cryptographic principles underlying each algorithmic step.

# AUDIT & FIX: hash-impl

## CRITIQUE
- Milestone 1 has redundant acceptance criteria: AC2 ('Append 1 bit followed by zeros to reach 448 mod 512 bit alignment') and AC3 ('Pad with zeros to 448 mod 512 bits then append original length as 64-bit big-endian') overlap significantly. AC1 ('Convert message to binary representation and append 1 padding bit') is partially redundant with AC2. AC4 ('Append original message length as 64-bit big-endian integer at block end') is the tail end of AC3. These should be consolidated.
- Missing: the edge case where the message length modulo 512 is > 447 bits, requiring an additional padding block. This is a critical implementation detail that catches many beginners.
- Milestone 2 AC says 'Words stored as 32-bit unsigned integers with overflow masked to 32 bits' — this is only needed in languages without native 32-bit unsigned overflow (Python, JavaScript). Should be stated as language-conditional.
- Milestone 3 is solid but missing an AC for verifying intermediate hash values against NIST example computation appendix.
- Milestone 4 AC 'Handle empty input correctly' is good but should specify the exact expected hash (e5e7...69d7 for SHA-256 of empty string) as a measurable criterion.
- The project is labeled 'beginner' which is appropriate for the algorithm itself but the bit manipulation requirements are intermediate in most languages. Python makes it beginner-friendly; C makes it intermediate.
- No mention of streaming/chunked hashing (processing input larger than memory), which is an important practical consideration.
- Missing pitfall: in Python/JavaScript, integers are arbitrary precision, so forgetting to mask to 32 bits after every addition causes wrong results. This is the #1 beginner mistake.

## FIXED YAML
```yaml
id: hash-impl
name: SHA-256 Hash Function
description: >-
  Implement SHA-256 from the NIST FIPS 180-4 specification, covering message
  preprocessing, message schedule generation, compression function, and
  final hash output with test vector validation.
difficulty: beginner
estimated_hours: "10-15"
essence: >-
  Merkle-Damgård construction using iterative compression with bitwise
  logical functions (Ch, Maj, Σ, σ), modular 32-bit addition, and message
  scheduling to produce collision-resistant 256-bit digests from arbitrary
  input through 64 rounds of compression per 512-bit block.
why_important: >-
  Building SHA-256 from scratch teaches low-level bit manipulation,
  cryptographic algorithm implementation, and how to translate formal
  mathematical specifications (NIST FIPS 180-4) into working code —
  essential skills for security engineering and systems programming.
learning_outcomes:
  - Implement message padding with length encoding to create 512-bit aligned blocks
  - Design message schedule generation using bitwise rotation and XOR operations
  - Build compression function with Ch, Maj, Σ0, Σ1, σ0, and σ1 logical functions
  - Apply modular 32-bit arithmetic with proper overflow masking
  - Translate NIST FIPS specification pseudocode into executable code
  - Debug bit-level operations using intermediate value validation against spec examples
  - Handle endianness conversion for cross-platform correctness
  - Verify cryptographic correctness against standard NIST test vectors
skills:
  - Bitwise Operations
  - Cryptographic Algorithms
  - Specification Implementation
  - Modular Arithmetic
  - Binary Data Handling
  - Algorithm Verification
  - Test-Driven Development
tags:
  - beginner-friendly
  - c
  - cryptography
  - hashing
  - implementation
  - javascript
  - python
  - sha-256
architecture_doc: architecture-docs/hash-impl/index.md
languages:
  recommended:
    - Python
    - JavaScript
    - C
  also_possible:
    - Rust
    - Go
    - Java
resources:
  - name: "SHA-256 Step by Step"
    url: https://blog.boot.dev/cryptography/how-sha-2-works-step-by-step-sha-256/
    type: tutorial
  - name: "NIST FIPS 180-4 (SHA-256 Specification)"
    url: https://csrc.nist.gov/publications/detail/fips/180/4/final
    type: specification
  - name: "NIST SHA-256 Test Vectors"
    url: https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Standards-and-Guidelines/documents/examples/SHA256.pdf
    type: reference
prerequisites:
  - type: skill
    name: Binary and hexadecimal representation
  - type: skill
    name: Bitwise operations (AND, OR, XOR, shift, rotate)
  - type: skill
    name: Basic understanding of hash functions
milestones:
  - id: hash-impl-m1
    name: "Message Preprocessing and Padding"
    description: >-
      Implement SHA-256 message padding: append the '1' bit, pad with zeros
      to 448 mod 512 bits, append the 64-bit big-endian message length,
      and parse into 512-bit blocks.
    acceptance_criteria:
      - >-
        Message is converted to its byte/bit representation, a single '1'
        bit is appended, followed by enough '0' bits so that the total
        message length is congruent to 448 mod 512 bits, and finally the
        original message length in bits is appended as a 64-bit big-endian
        integer, producing a message whose total length is a multiple of
        512 bits.
      - >-
        When the original message length mod 512 is greater than 447 bits
        (i.e., fewer than 65 bits remain for padding + length), an
        additional 512-bit block is correctly appended to accommodate the
        padding and length field.
      - >-
        Padded message is parsed into an array of 512-bit (64-byte) blocks
        for sequential processing.
      - >-
        Empty input (0 bytes) produces exactly one 512-bit padded block
        with the '1' bit at position 0, zeros through bit 447, and a
        64-bit zero length field.
      - >-
        Test: the string "abc" (24 bits) produces one 512-bit block;
        a 55-byte message produces one block; a 56-byte message produces
        two blocks (boundary case).
    pitfalls:
      - >-
        Off-by-one in padding length calculation: the padding must account
        for the '1' bit AND the 64-bit length field. The number of zero
        padding bits is (447 - message_bit_length) mod 512, not
        (448 - message_bit_length) mod 512, because the '1' bit is
        separate.
      - >-
        Endianness: SHA-256 uses big-endian byte ordering throughout. On
        little-endian systems (x86), bytes must be swapped when reading
        words from the padded message.
      - >-
        Forgetting the extra block: when the message is 56-64 bytes,
        there isn't enough room in the first block for padding + length,
        requiring a second block. This is the most common beginner bug.
    concepts:
      - SHA-256 message padding rules
      - Bit/byte boundary handling
      - Big-endian byte ordering
      - Block boundary edge cases
    skills:
      - Binary data manipulation
      - Bitwise operations
      - Padding algorithms
      - Byte-order conversion
    deliverables:
      - Padding function appending '1' bit, zero-fill, and 64-bit length
      - Block parser splitting padded message into 512-bit chunks
      - Edge case handling for messages requiring an extra padding block
      - Test cases for empty, short, and boundary-length inputs
    estimated_hours: "2-3"

  - id: hash-impl-m2
    name: "Message Schedule Generation"
    description: >-
      For each 512-bit block, generate the 64-word message schedule using
      the SHA-256 recurrence relation with σ0 and σ1 functions.
    acceptance_criteria:
      - >-
        512-bit block is parsed into 16 initial 32-bit words (W[0]..W[15])
        in big-endian order.
      - >-
        Words W[16]..W[63] are computed using the recurrence:
        W[t] = σ1(W[t-2]) + W[t-7] + σ0(W[t-15]) + W[t-16], all mod 2^32.
      - >-
        σ0(x) = ROTR(x,7) XOR ROTR(x,18) XOR SHR(x,3) produces correct
        values verified against NIST example computation.
      - >-
        σ1(x) = ROTR(x,17) XOR ROTR(x,19) XOR SHR(x,10) produces correct
        values verified against NIST example computation.
      - >-
        All 64 words are stored as 32-bit unsigned integers. In languages
        with arbitrary-precision integers (Python, JavaScript), every
        addition is masked with & 0xFFFFFFFF.
    pitfalls:
      - >-
        In Python/JavaScript: forgetting to mask to 32 bits (& 0xFFFFFFFF)
        after EVERY addition. This is the #1 mistake because the intermediate
        values look plausible but diverge from the spec after several rounds.
      - >-
        Confusing right-rotate (ROTR) with right-shift (SHR). ROTR wraps
        bits around; SHR fills with zeros. σ functions use BOTH, and mixing
        them up produces wrong schedule values.
      - >-
        Wrong σ function rotation/shift constants: σ0 uses (7,18,3) and
        σ1 uses (17,19,10). Swapping these produces a valid-looking but
        completely wrong hash.
    concepts:
      - Right-rotate vs. right-shift operations
      - XOR combination of rotated/shifted values
      - Message schedule expansion
      - 32-bit modular arithmetic
    skills:
      - Bitwise rotation and shifting
      - Logical operations (XOR, AND, OR)
      - 32-bit modular arithmetic
      - Array manipulation
    deliverables:
      - Word extraction parsing 512-bit block into 16 big-endian 32-bit words
      - σ0 and σ1 functions with correct rotate and shift constants
      - Message schedule expansion from 16 to 64 words
      - 32-bit overflow masking for languages without native 32-bit unsigned
    estimated_hours: "2-3"

  - id: hash-impl-m3
    name: "Compression Function"
    description: >-
      Implement the SHA-256 compression function: 64 rounds of state
      transformation using Ch, Maj, Σ0, Σ1, round constants K, and
      message schedule words.
    acceptance_criteria:
      - >-
        Working variables (a through h) are initialized from the current
        hash values (H0..H7) at the start of each block's compression.
      - >-
        64 rounds of compression execute, each computing:
        T1 = h + Σ1(e) + Ch(e,f,g) + K[t] + W[t]
        T2 = Σ0(a) + Maj(a,b,c)
        then shifting variables: h=g, g=f, f=e, e=d+T1, d=c, c=b, b=a,
        a=T1+T2, all mod 2^32.
      - >-
        Ch(x,y,z) = (x AND y) XOR (NOT x AND z) produces correct values.
      - >-
        Maj(x,y,z) = (x AND y) XOR (x AND z) XOR (y AND z) produces
        correct values.
      - >-
        Σ0(a) = ROTR(a,2) XOR ROTR(a,13) XOR ROTR(a,22) and
        Σ1(e) = ROTR(e,6) XOR ROTR(e,11) XOR ROTR(e,25) produce correct
        values.
      - >-
        All 64 round constants K[0]..K[63] are the first 32 bits of the
        fractional parts of the cube roots of the first 64 primes, matching
        the values in FIPS 180-4 Section 4.2.2.
      - >-
        After all 64 rounds, the hash values are updated:
        H0 += a, H1 += b, ..., H7 += h (all mod 2^32).
      - >-
        Intermediate hash values after processing the "abc" test vector's
        first block match the NIST example computation appendix values.
    pitfalls:
      - >-
        Mixing up Σ (upper-case sigma, used in compression) with σ
        (lower-case sigma, used in message schedule). They use different
        rotation constants.
      - >-
        Wrong K constant values: copying from an incorrect source or
        truncating to wrong precision. Always verify against FIPS 180-4
        Table of K constants.
      - >-
        Not masking intermediate T1 and T2 values to 32 bits in
        arbitrary-precision languages, causing divergence in later rounds.
      - >-
        Variable rotation error: forgetting that 'e = d + T1' (not
        'd + T1 + T2') and 'a = T1 + T2'.
    concepts:
      - Compression function rounds
      - Choice (Ch) and Majority (Maj) functions
      - Hash state transformation
      - Round constants from prime cube roots
    skills:
      - Cryptographic round functions
      - State transformation chains
      - Working with lookup tables
      - Modular addition with masking
    deliverables:
      - Ch, Maj, Σ0, and Σ1 bitwise functions
      - 64-element K constants array from FIPS 180-4
      - 64-round compression loop with variable rotation
      - Hash state update after block compression
      - Intermediate value validation against NIST example
    estimated_hours: "4-5"

  - id: hash-impl-m4
    name: "Final Hash Output and Validation"
    description: >-
      Process all blocks sequentially, produce the final 256-bit hash,
      format as hexadecimal, and validate against NIST test vectors.
    acceptance_criteria:
      - >-
        Hash state (H0..H7) is initialized to the SHA-256 initial hash
        values from FIPS 180-4 Section 5.3.3 before processing any blocks.
      - >-
        All message blocks are processed sequentially, each updating the
        hash state through the compression function.
      - >-
        Final hash is the concatenation of H0..H7 as 32-bit big-endian
        values, producing a 256-bit (32-byte) digest.
      - >-
        Output is formatted as a 64-character lowercase hexadecimal string.
      - >-
        SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
      - >-
        SHA-256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
      - >-
        SHA-256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") =
        248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1
      - >-
        Function handles multiple independent invocations correctly: hash
        state is fully reset between calls, producing identical output for
        identical input.
      - >-
        Streaming/chunked API: input can be fed in arbitrary-sized chunks
        (update/finalize pattern), producing the same hash as single-call
        processing.
    pitfalls:
      - >-
        Not resetting hash state between calls: if H0..H7 retain values
        from a previous hash computation, subsequent hashes are wrong.
      - >-
        Endianness in output: H0..H7 must be written as big-endian bytes
        in the final concatenation, even on little-endian platforms.
      - >-
        Streaming API: the update function must buffer partial blocks
        internally and only process complete 512-bit blocks. The finalize
        function must pad the remaining buffer. Getting the buffer
        management wrong is a common source of off-by-one errors.
    concepts:
      - Hash finalization
      - Test vector validation
      - Hexadecimal encoding
      - Streaming hash API design
    skills:
      - Hash output formatting
      - Hexadecimal encoding
      - State management and reset
      - Test-driven validation
    deliverables:
      - Initial hash value constants from FIPS 180-4
      - Sequential block processing loop
      - 64-character hexadecimal output formatter
      - Streaming API with update() and finalize() methods
      - Test vector validation against all three NIST examples above
    estimated_hours: "2-4"
```
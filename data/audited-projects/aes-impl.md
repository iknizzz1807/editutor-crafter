# AUDIT & FIX: aes-impl

## CRITIQUE
- No mention of constant-time implementation for S-box lookups, which is a major side-channel vulnerability. Table-based S-box lookups are vulnerable to cache-timing attacks. The project should at least educate about this and optionally require a bitsliced or constant-time S-box implementation.
- Milestone 4 mentions CBC mode but does not address padding oracle attacks, which are the most famous practical attack against CBC-mode encryption. This is a critical educational gap.
- Milestone 4 AC mentions PKCS7 padding but deliverables list CTR mode instead of PKCS7 decryption padding validation, creating an inconsistency between ACs and deliverables.
- GCM mode is mentioned in learning outcomes but not in any milestone. GCM is the most important authenticated encryption mode and should be included or explicitly deferred.
- Milestone 3 AC says 'Rcon are correctly computed as successive powers of 2 in GF(2^8)' — this is imprecise. Rcon[i] = x^(i-1) in GF(2^8), which is powers of x (the polynomial), not powers of 2 (the integer). Rcon[1]=0x01, Rcon[2]=0x02, Rcon[3]=0x04, ..., Rcon[8]=0x80, Rcon[9]=0x1B (reduction). This matters.
- Milestone 1 says 'Multiplicative inverse lookup returns the correct inverse for every non-zero element' but doesn't mention that the S-box also applies an affine transformation AFTER the inverse. The S-box is not just the GF(2^8) inverse.
- No mention of decryption (InvSubBytes, InvShiftRows, InvMixColumns) — the project only talks about encryption.
- Estimated hours (15-25) is reasonable for encryption only but tight if decryption and multiple modes are included.

## FIXED YAML
```yaml
id: aes-impl
name: AES Implementation
description: >-
  Implement AES-128/192/256 from the NIST FIPS 197 specification, including
  GF(2^8) arithmetic, all four round transformations, key expansion,
  encryption, decryption, and cipher modes (ECB, CBC, CTR).
difficulty: intermediate
estimated_hours: "20-30"
essence: >-
  Galois field GF(2^8) arithmetic over the irreducible polynomial
  x^8+x^4+x^3+x+1, substitution-permutation network with SubBytes,
  ShiftRows, MixColumns, and AddRoundKey transformations, Rijndael key
  scheduling, and block cipher modes of operation — producing a complete
  encryption/decryption implementation validated against NIST test vectors.
why_important: >-
  Building AES from scratch demystifies the cryptographic primitive
  underlying most modern secure communications (TLS, disk encryption,
  VPNs), teaching finite field mathematics, bitwise manipulation, and
  the security mindset required for implementation where a single bug
  can compromise an entire system.
learning_outcomes:
  - Implement GF(2^8) addition and multiplication with irreducible polynomial reduction
  - Build SubBytes (S-box), ShiftRows, MixColumns, and AddRoundKey transformations
  - Implement inverse transformations for decryption
  - Build key expansion for AES-128, AES-192, and AES-256
  - Implement ECB, CBC, and CTR cipher modes
  - Understand why table-based S-box implementations are vulnerable to cache-timing attacks
  - Validate correctness using NIST FIPS 197 test vectors
  - Understand padding oracle attacks on CBC mode and why authenticated encryption matters
skills:
  - Finite Field Arithmetic
  - Block Cipher Design
  - Bitwise Operations
  - Cryptographic Algorithms
  - Side-Channel Awareness
  - Standards Compliance
  - Decryption Implementation
  - Security Implementation
tags:
  - block-cipher
  - c
  - cryptography
  - galois-field
  - implementation
  - intermediate
  - python
  - rust
  - security
architecture_doc: architecture-docs/aes-impl/index.md
languages:
  recommended:
    - C
    - Rust
    - Python
  also_possible:
    - Go
    - Java
resources:
  - name: "FIPS 197 (AES Specification)"
    url: https://csrc.nist.gov/publications/detail/fips/197/final
    type: specification
  - name: "A Stick Figure Guide to AES"
    url: http://www.moserware.com/2009/09/stick-figure-guide-to-advanced.html
    type: article
  - name: "NIST AES Test Vectors"
    url: https://csrc.nist.gov/CSRC/media/Projects/Cryptographic-Algorithm-Validation-Program/documents/aes/AESAVS.pdf
    type: reference
prerequisites:
  - type: skill
    name: Bitwise operations (AND, OR, XOR, shift)
  - type: skill
    name: Modular arithmetic
  - type: skill
    name: Basic understanding of cryptographic concepts
milestones:
  - id: aes-impl-m1
    name: "GF(2^8) Arithmetic and S-Box Construction"
    description: >-
      Implement Galois field GF(2^8) operations and construct the AES
      S-box from the multiplicative inverse and affine transformation.
    acceptance_criteria:
      - >-
        GF(2^8) addition is implemented as bitwise XOR of two bytes.
      - >-
        GF(2^8) multiplication produces correct results using the
        irreducible polynomial x^8+x^4+x^3+x+1 (0x11B) for reduction,
        verified against known test vectors (e.g., 0x57 * 0x83 = 0xC1).
      - >-
        Multiplicative inverse is computed for every non-zero element
        in GF(2^8), with 0 mapping to 0 by convention.
      - >-
        S-box is constructed by applying the multiplicative inverse
        followed by the affine transformation specified in FIPS 197
        Section 5.1.1. S-box[0x00]=0x63, S-box[0x53]=0xED, etc.
      - >-
        Inverse S-box is constructed for decryption (applies inverse
        affine then multiplicative inverse). InvS-box[0x63]=0x00.
      - >-
        Pre-computed S-box and inverse S-box tables (256 bytes each)
        match all 256 values in FIPS 197.
    pitfalls:
      - >-
        The S-box is NOT just the GF(2^8) multiplicative inverse. After
        computing the inverse, an affine transformation (bit-level matrix
        multiply + XOR with 0x63) must be applied. Missing this step
        produces a valid-looking but completely wrong S-box.
      - >-
        GF(2^8) multiplication overflow: the xtime function (multiply by
        x, i.e., left-shift and conditional XOR with 0x1B) is the building
        block. Getting the conditional reduction wrong corrupts all
        dependent computations.
      - >-
        Using the wrong irreducible polynomial: AES specifically uses
        0x11B. Other GF(2^8) polynomials exist but produce different
        field arithmetic.
    concepts:
      - Galois field GF(2^8) arithmetic
      - Irreducible polynomial reduction
      - S-box construction (inverse + affine)
      - Lookup table precomputation
    skills:
      - Finite field arithmetic
      - Bitwise operations
      - Lookup table construction
      - Affine transformation implementation
    deliverables:
      - GF(2^8) addition (XOR) and multiplication (with 0x11B reduction)
      - xtime function (multiply by x in GF(2^8))
      - Multiplicative inverse computation for all 256 elements
      - S-box and inverse S-box tables matching FIPS 197
      - Verification against known GF(2^8) multiplication results
    estimated_hours: "3-5"

  - id: aes-impl-m2
    name: "AES Round Operations (Encrypt and Decrypt)"
    description: >-
      Implement all four AES encryption transformations (SubBytes, ShiftRows,
      MixColumns, AddRoundKey) and their inverses for decryption.
    acceptance_criteria:
      - >-
        SubBytes replaces every byte in the 4×4 state matrix with its
        S-box substitution value. InvSubBytes uses the inverse S-box.
      - >-
        ShiftRows cyclically left-shifts row i by i positions (row 0
        unchanged, row 1 by 1, row 2 by 2, row 3 by 3). InvShiftRows
        right-shifts by the same amounts.
      - >-
        MixColumns multiplies each column vector by the fixed polynomial
        matrix [{02,03,01,01},{01,02,03,01},{01,01,02,03},{03,01,01,02}]
        in GF(2^8). InvMixColumns uses the inverse matrix
        [{0e,0b,0d,09},{09,0e,0b,0d},{0d,09,0e,0b},{0b,0d,09,0e}].
      - >-
        AddRoundKey XORs each byte of the state with the corresponding
        byte of the round key. (Self-inverse: same operation for
        encrypt and decrypt.)
      - >-
        Each transformation is independently verified against FIPS 197
        Appendix B intermediate values for the AES-128 example.
      - >-
        State matrix uses column-major ordering as specified in FIPS 197
        (state[row][col], stored column-by-column in memory).
    pitfalls:
      - >-
        State matrix orientation: FIPS 197 uses column-major layout.
        Confusing row-major and column-major causes ShiftRows and
        MixColumns to operate on wrong elements.
      - >-
        ShiftRows direction: encryption shifts LEFT, decryption shifts
        RIGHT. Getting the direction wrong is a common mistake.
      - >-
        MixColumns coefficients for decryption (0x0e, 0x0b, 0x0d, 0x09)
        require multi-step GF(2^8) multiplication. Using encryption
        coefficients for decryption produces wrong results.
      - >-
        Side-channel vulnerability: table-based S-box lookups are
        vulnerable to cache-timing attacks. Document this risk. For
        educational purposes, table lookups are acceptable, but note
        that production implementations use bitsliced or constant-time
        approaches.
    concepts:
      - Substitution (SubBytes) and permutation (ShiftRows)
      - Diffusion (MixColumns)
      - Key mixing (AddRoundKey)
      - Inverse operations for decryption
      - Cache-timing side-channel awareness
    skills:
      - State array manipulation
      - Matrix transformations in GF(2^8)
      - Implementing inverse operations
      - Side-channel awareness
    deliverables:
      - SubBytes and InvSubBytes transformations
      - ShiftRows and InvShiftRows transformations
      - MixColumns and InvMixColumns transformations
      - AddRoundKey transformation
      - Intermediate value verification against FIPS 197 Appendix B
      - Documentation of cache-timing side-channel risk for table lookups
    estimated_hours: "5-7"

  - id: aes-impl-m3
    name: "Key Expansion"
    description: >-
      Implement the AES key schedule for AES-128, AES-192, and AES-256,
      producing the correct number of round keys.
    acceptance_criteria:
      - >-
        AES-128 (16-byte key) expansion produces 44 words (11 round keys
        × 4 words each = 176 bytes).
      - >-
        AES-192 (24-byte key) expansion produces 52 words (13 round keys
        × 4 words each = 208 bytes).
      - >-
        AES-256 (32-byte key) expansion produces 60 words (15 round keys
        × 4 words each = 240 bytes).
      - >-
        RotWord rotates a 4-byte word left by one byte.
        SubWord applies the S-box to each byte of a 4-byte word.
        Both produce correct intermediate values per FIPS 197 examples.
      - >-
        Round constants Rcon[i] = [x^(i-1) in GF(2^8), 0, 0, 0] are
        correctly computed. Rcon[1]=[01,00,00,00],
        Rcon[2]=[02,00,00,00], ..., Rcon[8]=[80,00,00,00],
        Rcon[9]=[1B,00,00,00], Rcon[10]=[36,00,00,00].
      - >-
        AES-256 additional SubWord step (applied when i mod 8 == 4) is
        correctly implemented.
      - >-
        Expanded key words match FIPS 197 Appendix A test vectors for
        all three key sizes.
    pitfalls:
      - >-
        Rcon computation: Rcon values are powers of x in GF(2^8), NOT
        powers of 2 in integer arithmetic. After Rcon[8]=0x80,
        Rcon[9]=0x1B (due to reduction by 0x11B), not 0x100.
      - >-
        AES-256 has an extra SubWord step that AES-128/192 do not.
        Forgetting this produces correct key expansion for 128/192 but
        wrong for 256.
      - >-
        Word vs. byte indexing: the key schedule operates on 32-bit
        words, but the round key is applied byte-by-byte. Confusing
        the two causes misaligned key application.
    concepts:
      - Rijndael key schedule
      - Round constants (Rcon) in GF(2^8)
      - RotWord and SubWord operations
      - Key-size-dependent schedule variations
    skills:
      - Key scheduling algorithms
      - GF(2^8) exponentiation for Rcon
      - Word-level operations
      - Multi-key-size implementation
    deliverables:
      - RotWord and SubWord helper functions
      - Rcon generation (powers of x in GF(2^8))
      - Key expansion for AES-128, AES-192, and AES-256
      - Verification against FIPS 197 Appendix A test vectors
    estimated_hours: "3-5"

  - id: aes-impl-m4
    name: "Encryption, Decryption, and Cipher Modes"
    description: >-
      Assemble the complete AES encrypt/decrypt functions and implement
      ECB, CBC, and CTR cipher modes with proper padding and IV handling.
    acceptance_criteria:
      - >-
        AES-128 encryption produces correct ciphertext matching FIPS 197
        Appendix B test vector (plaintext 00112233...eeff with key
        000102...0f).
      - >-
        AES-128 decryption produces correct plaintext from the above
        ciphertext. Decryption applies InvShiftRows, InvSubBytes,
        AddRoundKey, InvMixColumns in the correct order (equivalent
        inverse cipher or direct inverse cipher per FIPS 197).
      - >-
        Number of rounds is correct: 10 for AES-128, 12 for AES-192,
        14 for AES-256. The final round omits MixColumns/InvMixColumns.
      - >-
        ECB mode encrypts each 16-byte block independently. Demonstrate
        ECB's weakness by encrypting an image and showing the visible
        pattern leakage (or equivalent test).
      - >-
        CBC mode XORs each plaintext block with the previous ciphertext
        block (or IV for the first block) before encryption. Decryption
        reverses this correctly.
      - >-
        PKCS7 padding is correctly applied during encryption and validated
        during decryption. Invalid padding (e.g., tampered ciphertext)
        is rejected with an error.
      - >-
        CTR mode encrypts a counter (nonce + counter) to produce a
        keystream, XORed with plaintext. CTR mode does not require
        padding.
      - >-
        IV/nonce is generated using CSPRNG for CBC and CTR modes. IV
        reuse is documented as a critical vulnerability.
    pitfalls:
      - >-
        ECB mode vulnerability: ECB is included for educational purposes
        only. It MUST NOT be used for encrypting data larger than one
        block in any real application because identical plaintext blocks
        produce identical ciphertext blocks.
      - >-
        Padding oracle attacks on CBC: if the decryption code reveals
        whether padding is valid (through error messages or timing
        differences), an attacker can decrypt arbitrary ciphertext
        without the key. Always use authenticated encryption (AES-GCM)
        in production.
      - >-
        IV reuse in CBC or CTR mode completely destroys security. CBC
        with repeated IV leaks plaintext XOR; CTR with repeated nonce
        leaks plaintext XOR directly.
      - >-
        Decryption round order: the AES inverse cipher applies
        transformations in a different order than the forward cipher.
        FIPS 197 Section 5.3 specifies the exact order.
    concepts:
      - Block cipher modes of operation (ECB, CBC, CTR)
      - PKCS7 padding and padding oracle attacks
      - IV and nonce management
      - Authenticated encryption motivation
    skills:
      - Block cipher implementation
      - IV/nonce generation and management
      - Padding scheme implementation
      - Understanding cipher mode security properties
    deliverables:
      - Full AES encrypt and decrypt for 128/192/256-bit keys
      - ECB mode with educational demonstration of pattern leakage
      - CBC mode with PKCS7 padding and IV handling
      - CTR mode with nonce-counter construction
      - NIST test vector validation for all key sizes and modes
      - Documentation of padding oracle attack risk for CBC mode
    estimated_hours: "5-8"
```
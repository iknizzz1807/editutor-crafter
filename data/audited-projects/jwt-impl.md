# AUDIT & FIX: jwt-impl

## CRITIQUE
- Missing: Asymmetric signing (RS256 with RSA, ES256 with ECDSA, EdDSA) is completely absent. HMAC-only JWTs are used for same-service authentication but cross-service/microservice architectures universally use asymmetric algorithms. This is a major gap for an 'intermediate' project.
- Missing: No AC for rejecting 'alg: none' tokens, which is the single most exploited JWT vulnerability (CVE-2015-9235). This MUST be included.
- Missing: No AC for preventing algorithm confusion attacks (e.g., attacker changes alg from RS256 to HS256 and uses the public key as HMAC secret). The pitfall mentions it but there's no measurable AC.
- Milestone 1 AC says 'Header JSON contains the correct algorithm identifier' but doesn't specify that the algorithm field must be validated against an allowlist during verification.
- Milestone 2 AC says 'Tokens signed with HMAC-SHA256 produce signatures matching RFC 7515 test vectors' — RFC 7515 is JWS, which is correct, but the project should specify which exact test vector (Appendix A.1 of RFC 7515).
- No mention of token size considerations or the difference between JWS (signed) and JWE (encrypted) tokens.
- Clock skew tolerance is mentioned as a pitfall in M3 but not as a configurable parameter in the ACs.
- Estimated hours (8-12) is too low if asymmetric signing is added. Should be 12-18.

## FIXED YAML
```yaml
id: jwt-impl
name: JWT Library
description: >-
  Build a JWT library supporting HMAC (HS256) and RSA (RS256) signing,
  claims validation, and critical security defenses against algorithm
  confusion and 'alg: none' attacks.
difficulty: intermediate
estimated_hours: "12-18"
essence: >-
  HMAC-SHA256 and RSA-SHA256 signature generation/verification over
  Base64URL-encoded JSON payloads, with time-based claim validation,
  algorithm allowlisting, and defense against 'alg: none' and algorithm
  confusion attacks for stateless authentication tokens.
why_important: >-
  JWTs are the dominant token format for API authentication and
  authorization in modern distributed systems. Building one from scratch
  teaches cryptographic signing, secure token validation, and the critical
  security pitfalls that have led to real-world exploits.
learning_outcomes:
  - Implement Base64URL encoding/decoding for compact token representation
  - Build HMAC-SHA256 (HS256) signature generation and verification
  - Implement RSA-SHA256 (RS256) asymmetric signing and verification
  - Implement time-based claim validation (exp, nbf, iat) with clock skew tolerance
  - Defend against 'alg: none' attack by always requiring signature verification
  - Defend against algorithm confusion attacks with strict algorithm allowlisting
  - Implement constant-time signature comparison
  - Handle issuer, audience, and custom claim validation
skills:
  - HMAC Cryptography
  - RSA Signatures
  - Base64URL Encoding
  - Claims Validation
  - Token Authentication
  - Security Vulnerability Defense
  - JSON Serialization
  - Algorithm Allowlisting
tags:
  - claims
  - go
  - implementation
  - intermediate
  - javascript
  - python
  - security
  - signatures
  - tokens
architecture_doc: architecture-docs/jwt-impl/index.md
languages:
  recommended:
    - Python
    - JavaScript
    - Go
  also_possible:
    - Java
    - Rust
resources:
  - name: "JWT Specification (RFC 7519)"
    url: https://tools.ietf.org/html/rfc7519
    type: specification
  - name: "JWS Specification (RFC 7515)"
    url: https://tools.ietf.org/html/rfc7515
    type: specification
  - name: "JWT.io Debugger"
    url: https://jwt.io/
    type: tool
  - name: "Critical vulnerabilities in JSON Web Token libraries"
    url: https://auth0.com/blog/critical-vulnerabilities-in-json-web-token-libraries/
    type: article
prerequisites:
  - type: skill
    name: JSON serialization/deserialization
  - type: skill
    name: Base64 encoding concepts
  - type: skill
    name: HMAC basics (keyed hashing)
milestones:
  - id: jwt-impl-m1
    name: "JWT Structure and Base64URL Encoding"
    description: >-
      Implement JWT encoding: header and payload JSON serialization,
      Base64URL encoding, and the three-part dot-separated token format.
    acceptance_criteria:
      - >-
        Header JSON contains 'alg' (algorithm) and 'typ' ('JWT') fields.
        Supported algorithms: HS256, RS256, and 'none' (for testing only;
        rejected during verification).
      - >-
        Payload encodes registered claims (iss, sub, aud, exp, nbf, iat,
        jti) and arbitrary custom claims as valid JSON.
      - >-
        Base64URL encoding uses URL-safe alphabet (- instead of +,
        _ instead of /) with no padding characters (= stripped).
      - >-
        Base64URL decoding correctly handles input both with and without
        padding characters.
      - >-
        Assembled token follows the format: Base64URL(header) + '.' +
        Base64URL(payload) + '.' + Base64URL(signature).
      - >-
        Round-trip test: encode a token, decode it, verify all header
        and payload fields are preserved exactly.
    pitfalls:
      - >-
        Regular Base64 vs. Base64URL: '+' becomes '-', '/' becomes '_',
        and '=' padding is removed. Using standard Base64 produces tokens
        that break in URLs and HTTP headers.
      - >-
        JSON key ordering: while JWT spec doesn't require ordered keys,
        some implementations produce non-deterministic JSON serialization,
        which breaks signature verification if the signing and verifying
        sides serialize differently. Use canonical/sorted JSON or sign
        the exact serialized bytes.
      - >-
        UTF-8 encoding: JSON payloads must be UTF-8 encoded before
        Base64URL encoding. Incorrect encoding causes cross-platform
        verification failures.
    concepts:
      - JWT three-part structure (header.payload.signature)
      - Base64URL encoding (RFC 4648 Section 5)
      - Registered vs. custom claims
    skills:
      - Base64URL encoding/decoding
      - JSON serialization
      - String encoding handling
      - Token format assembly
    deliverables:
      - Base64URL encoder/decoder with padding removal
      - Header encoder producing {"alg":"...","typ":"JWT"}
      - Payload encoder with registered and custom claims
      - Token assembler (header.payload.signature format)
      - Round-trip encode/decode test
    estimated_hours: "2-3"

  - id: jwt-impl-m2
    name: "HMAC-SHA256 and RSA-SHA256 Signing"
    description: >-
      Implement HS256 (symmetric) and RS256 (asymmetric) signing and
      verification with constant-time comparison and 'alg: none' defense.
    acceptance_criteria:
      - >-
        HS256 signing computes HMAC-SHA256 over the signing input
        (Base64URL(header) + '.' + Base64URL(payload)) using a secret key,
        producing a signature matching RFC 7515 Appendix A.1 test vector.
      - >-
        HS256 verification recomputes the HMAC and compares using
        constant-time comparison (hmac.compare_digest or equivalent).
      - >-
        RS256 signing computes RSASSA-PKCS1-v1_5 with SHA-256 over the
        signing input using an RSA private key (minimum 2048-bit).
      - >-
        RS256 verification uses the RSA public key to verify the
        signature. Verification uses the crypto library's built-in
        verify function (not decrypt-and-compare).
      - >-
        Tokens with 'alg: none' are ALWAYS rejected during verification
        with an explicit error, regardless of whether a signature is
        present. There is no code path that skips signature verification.
      - >-
        Algorithm allowlisting: the verifier accepts a configured list
        of allowed algorithms and rejects any token whose 'alg' header
        is not in the allowlist, preventing algorithm confusion attacks.
      - >-
        Tokens with tampered headers, payloads, or signatures are
        rejected with descriptive error messages.
      - >-
        HS256 key minimum length is enforced: keys shorter than 256 bits
        (32 bytes) are rejected with an error.
    pitfalls:
      - >-
        'alg: none' vulnerability (CVE-2015-9235): if the verifier reads
        the algorithm from the token header and respects 'none', an
        attacker can forge any token by removing the signature and setting
        alg to 'none'. NEVER trust the token's alg field for choosing
        the verification algorithm.
      - >-
        Algorithm confusion attack: attacker changes alg from RS256 to
        HS256 and uses the RSA public key (which is public!) as the HMAC
        secret. The verifier must use the algorithm from its own
        configuration, not from the token header.
      - >-
        Timing attacks: using '==' for signature comparison leaks
        information about how many bytes match, allowing byte-by-byte
        forgery. Always use constant-time comparison.
      - >-
        HMAC key encoding: the secret must be used as raw bytes, not as
        a UTF-8 or hex string. Inconsistent encoding between signing and
        verifying causes verification failures.
    concepts:
      - HMAC-SHA256 symmetric signing
      - RSA-SHA256 asymmetric signing
      - 'alg: none' vulnerability
      - Algorithm confusion attacks
      - Constant-time comparison
    skills:
      - HMAC-SHA256 computation
      - RSA signature generation and verification
      - Constant-time comparison
      - Algorithm allowlisting
    deliverables:
      - HS256 sign and verify with constant-time comparison
      - RS256 sign (private key) and verify (public key)
      - Algorithm allowlist enforcement in verifier
      - 'alg: none' rejection with explicit error
      - Key validation (minimum length for HS256, minimum size for RSA)
      - Test suite covering valid signatures, tampered tokens, alg:none, and algorithm confusion
    estimated_hours: "4-5"

  - id: jwt-impl-m3
    name: "Claims Validation"
    description: >-
      Implement standard JWT claim validation with time checks, issuer/
      audience verification, and configurable clock skew tolerance.
    acceptance_criteria:
      - >-
        Tokens with exp claim in the past (current_time > exp) are rejected
        as expired with an 'TokenExpired' error.
      - >-
        Tokens with nbf claim in the future (current_time < nbf) are
        rejected as not-yet-valid with a 'TokenNotYetValid' error.
      - >-
        Clock skew tolerance is configurable (default 60 seconds) and
        applied to both exp and nbf checks, so a token expired 30 seconds
        ago with 60-second tolerance is still accepted.
      - >-
        iss (issuer) claim is validated against a configured allowlist.
        Tokens from unknown issuers are rejected with an 'InvalidIssuer'
        error.
      - >-
        aud (audience) claim is validated against the expected audience.
        Both single-string and array-of-strings aud values are handled
        correctly per RFC 7519 Section 4.1.3.
      - >-
        Missing required claims (exp, iss, aud when configured as required)
        are rejected with a 'MissingClaim' error specifying which claim
        is absent.
      - >-
        Validation is performed AFTER signature verification succeeds.
        Claims are never trusted from an unsigned or tampered token.
    pitfalls:
      - >-
        Clock skew: distributed systems have unsynchronized clocks. Without
        tolerance, valid tokens are rejected by servers whose clocks are
        slightly behind. But too much tolerance (>5 minutes) weakens
        expiration security.
      - >-
        aud as array: RFC 7519 allows aud to be either a single string
        or an array of strings. Code that only handles one form will
        fail on the other.
      - >-
        Validating claims before verifying signature: an attacker could
        craft a token with valid claims but an invalid signature. Always
        verify signature FIRST.
      - >-
        iat (issued-at) validation: some implementations reject tokens
        with iat in the future, but the RFC does not mandate this.
        Document your policy explicitly.
    concepts:
      - Time-based claim validation
      - Clock skew tolerance
      - Issuer and audience validation
      - Claim requirement enforcement
    skills:
      - Timestamp validation
      - Clock skew tolerance implementation
      - Issuer/audience allowlisting
      - Error classification and reporting
    deliverables:
      - exp validation with configurable clock skew tolerance
      - nbf validation with clock skew tolerance
      - iss validation against issuer allowlist
      - aud validation supporting both string and array formats
      - Required claim enforcement with descriptive error messages
      - Test suite covering expired, not-yet-valid, wrong issuer, wrong audience, and missing claims
    estimated_hours: "3-4"
```
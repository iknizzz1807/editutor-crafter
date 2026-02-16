# AUDIT & FIX: build-tls

## CRITIQUE
- **'Pre-master Secret' Terminology Wrong for TLS 1.3**: TLS 1.3 replaces the 'pre-master secret' concept with a staged secret hierarchy: (0 → Early Secret), (ECDHE shared secret → Handshake Secret), (0 → Master Secret). Each stage uses HKDF-Extract with the previous stage's derived secret as salt. Using 'pre-master secret' is a TLS 1.2 term that doesn't apply.
- **Missing Key Update Mechanism**: TLS 1.3 supports post-handshake KeyUpdate messages (RFC 8446 Section 4.6.3) to rotate traffic keys without renegotiation. This is absent.
- **Missing Alert Protocol**: Alerts are mentioned in pitfalls (M1) but never appear in acceptance criteria. TLS alerts are essential for error handling, close notification, and security (bad_record_mac, unexpected_message, etc.).
- **OCSP Stapling / CRL Missing**: Certificate verification milestone doesn't include any revocation checking. In practice, CRL and OCSP are complex, but at minimum OCSP stapling (certificate status from the server's CertificateStatus extension) should be mentioned.
- **Record Layer Encryption Missing from M1**: M1 lists AEAD encryption as a concept but the record layer milestone only handles plaintext. Encrypted record layer (inner content type, padding) should be addressed when records start being encrypted (post-ServerHello).
- **Handshake Transcript Hash Not Explicitly Required**: The transcript hash is FUNDAMENTAL to TLS 1.3 key derivation and Finished verification, but it's not called out as an explicit deliverable or AC.
- **0-RTT / Early Data Missing**: The project mentions 'zero-RTT resumption with replay attack mitigation' in learning outcomes but no milestone implements it. This is an advanced feature and could be optional, but it shouldn't be in learning outcomes if not implemented.
- **Milestone Sizing Uneven**: M3 (Handshake Protocol) is 15-25 hours while M1 (Record Layer) is 8-12. The handshake is complex but 25 hours for a single milestone is too large; should be split.
- **Version Field Confusion Not Addressed**: In TLS 1.3, the record layer version field is ALWAYS 0x0303 (TLS 1.2). The ClientHello legacy_version is also 0x0303. Actual version is only in supported_versions extension. This is a major confusion source not addressed.

## FIXED YAML
```yaml
id: build-tls
name: Build Your Own TLS 1.3
description: >-
  Full TLS 1.3 implementation supporting both client and server roles with
  record layer, ECDHE key exchange, multi-stage key schedule, certificate
  verification, and encrypted communication.
difficulty: expert
estimated_hours: "60-90"
essence: >-
  Binary protocol state machine implementing TLS 1.3 handshake with transcript
  hash tracking, multi-stage HKDF key derivation (Early → Handshake → Master
  Secret), ECDHE key agreement, X.509 certificate chain-of-trust verification,
  AEAD-authenticated record encryption with inner content type, and alert
  protocol handling over fragmented network streams.
why_important: >-
  Building TLS 1.3 from scratch demystifies how HTTPS secures the web by
  forcing you to implement modern cryptographic protocols, teaching skills
  directly applicable to security-critical systems and low-level protocol
  engineering.
learning_outcomes:
  - Implement TLS 1.3 record layer with plaintext and encrypted record handling
  - Design binary protocol parsers with fragmentation, reassembly, and content type routing
  - Integrate cryptographic primitives (X25519, HKDF, AES-GCM) into protocol flow
  - Implement TLS 1.3 multi-stage key schedule via HKDF-Extract and HKDF-Expand-Label
  - Build X.509 certificate chain verification with hostname validation
  - Implement handshake state machine with strict message ordering enforcement
  - Handle TLS alert protocol for error reporting and graceful termination
  - Maintain transcript hash across all handshake messages for key derivation integrity
skills:
  - TLS 1.3 Protocol State Machine
  - Binary Protocol Parsing & Serialization
  - ECDHE Key Exchange (X25519)
  - Multi-Stage HKDF Key Derivation
  - AEAD Encryption (AES-GCM)
  - X.509 Certificate Verification
  - Transcript Hash Management
  - Alert Protocol Handling
tags:
  - build-from-scratch
  - c
  - certificates
  - encryption
  - expert
  - go
  - handshake
  - pki
  - rust
  - security
  - tls-1.3
architecture_doc: architecture-docs/build-tls/index.md
languages:
  recommended:
    - Rust
    - Go
    - C
  also_possible:
    - Python
resources:
  - type: specification
    name: RFC 8446 - TLS 1.3
    url: https://datatracker.ietf.org/doc/html/rfc8446
  - type: tutorial
    name: Illustrated TLS 1.3
    url: https://tls13.xargs.org/
  - type: specification
    name: RFC 5280 - X.509 Certificates
    url: https://datatracker.ietf.org/doc/html/rfc5280
  - type: specification
    name: RFC 7748 - X25519
    url: https://datatracker.ietf.org/doc/html/rfc7748
prerequisites:
  - type: skill
    name: TCP socket programming
  - type: skill
    name: Symmetric encryption (AES-GCM)
  - type: skill
    name: Elliptic curve basics (Diffie-Hellman)
  - type: skill
    name: Binary data manipulation (struct packing, big-endian)
milestones:
  - id: build-tls-m1
    name: Record Layer & Alert Protocol
    description: >-
      Implement TLS record protocol for framing, content type routing,
      fragment reassembly, and the alert protocol for error handling.
    acceptance_criteria:
      - Parse 5-byte record header: content_type (1 byte), legacy_record_version (2 bytes, always 0x0303), length (2 bytes big-endian)
      - Enforce maximum plaintext record size of 16384 bytes (2^14); reject oversized records with record_overflow alert
      - Route records by content_type: handshake (22), alert (21), change_cipher_spec (20), application_data (23)
      - Reassemble fragmented handshake messages spanning multiple records into complete messages
      - Handle multiple handshake messages packed into a single record
      - Implement alert protocol: parse alert level (warning=1, fatal=2) and description codes
      - Handle close_notify alert for graceful shutdown (both sending and receiving)
      - On fatal alert received, close connection immediately and report error
      - Send appropriate alerts on protocol errors (unexpected_message, decode_error, record_overflow)
      - Record writer constructs properly framed records with header and payload
      - Note: legacy_record_version is ALWAYS 0x0303 in TLS 1.3, not 0x0304
    pitfalls:
      - Version field 0x0303 in record header does NOT mean TLS 1.2; in TLS 1.3 it's always 0x0303 for middlebox compatibility
      - First ClientHello record may use legacy_record_version 0x0301 per RFC 8446 Section 5.1
      - Record length field is 2 bytes big-endian (max 16384 + 256 for encrypted records with content type + padding)
      - Encrypted records use content_type=application_data (23) for ALL encrypted content; inner content type is inside
      - Alert messages are exactly 2 bytes; receiving partial alert is a protocol error
    concepts:
      - TLS record protocol framing and multiplexing
      - Fragment reassembly for handshake messages
      - Alert protocol levels and descriptions
      - Version field semantics in TLS 1.3
    skills:
      - Binary protocol parsing with big-endian fields
      - State management across fragmented messages
      - Error handling and alert generation
      - Buffer management for variable-size records
    deliverables:
      - Record header parser and writer
      - Content type router dispatching to handshake, alert, and data handlers
      - Fragment reassembly buffer for multi-record handshake messages
      - Alert protocol handler (send and receive)
      - Record size validation enforcing limits
    estimated_hours: "8-12"

  - id: build-tls-m2
    name: ECDHE Key Exchange & Key Schedule
    description: >-
      Implement X25519 ECDHE key exchange and the TLS 1.3 multi-stage
      key schedule deriving handshake and application traffic keys.
    acceptance_criteria:
      - Generate ephemeral X25519 key pair (32-byte private key with clamping per RFC 7748, 32-byte public key)
      - Compute ECDH shared secret from local private key and peer's public key
      - Validate shared secret is not all-zeros (reject invalid peer keys)
      - Implement HKDF-Extract(salt, ikm) per RFC 5869
      - Implement HKDF-Expand-Label(secret, label, context, length) with TLS 1.3 specific 'tls13 ' label prefix
      - Derive Early Secret: HKDF-Extract(salt=0x00*HashLen, ikm=0x00*HashLen)
      - Derive Handshake Secret: HKDF-Extract(salt=Derive-Secret(early_secret, "derived", ""), ikm=shared_secret)
      - Derive client_handshake_traffic_secret and server_handshake_traffic_secret from Handshake Secret + transcript hash at ServerHello
      - Derive Master Secret: HKDF-Extract(salt=Derive-Secret(handshake_secret, "derived", ""), ikm=0x00*HashLen)
      - Derive client_application_traffic_secret_0 and server_application_traffic_secret_0 from Master Secret + transcript hash at server Finished
      - Expand each traffic secret into write_key (16 bytes for AES-128) and write_iv (12 bytes) using HKDF-Expand-Label
      - Implement Derive-Secret(secret, label, messages) = HKDF-Expand-Label(secret, label, Hash(messages), HashLen)
      - Verify key derivation produces correct output against RFC 8446 Appendix B test vectors
    pitfalls:
      - TLS 1.3 has NO 'pre-master secret'; the hierarchy is Early Secret → Handshake Secret → Master Secret
      - HKDF-Expand-Label uses a specific encoding: 2-byte length + 1-byte label_length + 'tls13 '+label + 1-byte context_length + context
      - Transcript hash must be computed incrementally; missing or misordering any handshake message produces wrong keys
      - X25519 private key clamping: clear bits 0,1,2 of first byte; set bit 254; clear bit 255
      - Derive-Secret hashes the concatenated handshake messages, not the individual hashes
      - Test vectors in RFC 8446 Appendix B are essential for verifying correctness at each stage
    concepts:
      - TLS 1.3 three-stage key schedule
      - HKDF-Extract and HKDF-Expand-Label
      - ECDHE key exchange with X25519
      - Transcript hash dependency chain
      - Forward secrecy through ephemeral keys
    skills:
      - Elliptic curve Diffie-Hellman computation
      - HKDF implementation per RFC 5869
      - TLS 1.3 specific label encoding
      - Test vector verification
    deliverables:
      - X25519 key pair generation with proper clamping
      - ECDH shared secret computation
      - HKDF-Extract and HKDF-Expand-Label implementation
      - Three-stage key schedule (Early → Handshake → Master)
      - Traffic secret derivation for both directions
      - Key and IV expansion from traffic secrets
      - Test vector verification against RFC 8446 Appendix B
    estimated_hours: "12-18"

  - id: build-tls-m3
    name: Handshake Protocol & State Machine
    description: >-
      Implement the full TLS 1.3 handshake including ClientHello,
      ServerHello, EncryptedExtensions, and Finished messages with
      strict state machine enforcement.
    acceptance_criteria:
      - "CLIENT ROLE:"
      - Build ClientHello with: legacy_version=0x0303, random (32 bytes), legacy_session_id (non-empty for middlebox compat), cipher_suites (TLS_AES_128_GCM_SHA256), extensions (supported_versions with 0x0304, key_share with X25519 public key, SNI, signature_algorithms)
      - Parse ServerHello extracting server_random, selected cipher suite, and key_share extension
      - Verify ServerHello's supported_versions extension contains 0x0304
      - Switch to handshake traffic keys after ServerHello for decrypting subsequent messages
      - Process EncryptedExtensions (may be empty but must be handled)
      - Process server Certificate message extracting X.509 certificate chain
      - Process CertificateVerify verifying server's signature over transcript
      - Process server Finished verifying HMAC over transcript matches expected value
      - Send client Finished message with HMAC computed using client_handshake_traffic_secret
      - Switch to application traffic keys after sending client Finished
      - "SERVER ROLE (if implementing both sides):"
      - Parse ClientHello and select cipher suite and key share
      - Build and send ServerHello, EncryptedExtensions, Certificate, CertificateVerify, Finished
      - "STATE MACHINE:"
      - Enforce strict handshake message ordering; reject out-of-order messages with unexpected_message alert
      - Handshake state machine tracks: START → WAIT_SERVER_HELLO → WAIT_ENCRYPTED_EXTENSIONS → WAIT_CERTIFICATE → WAIT_CERTIFICATE_VERIFY → WAIT_FINISHED → CONNECTED
      - Maintain running transcript hash updated with each handshake message (type + length + body)
      - Handle HelloRetryRequest by detecting special server_random value and restarting handshake
    pitfalls:
      - After ServerHello, ALL remaining handshake messages are encrypted with handshake traffic keys
      - Encrypted handshake records use content_type=application_data (23) with inner content type = handshake (22)
      - Transcript hash must be updated BEFORE deriving keys that depend on it (order is critical)
      - CertificateVerify signature covers a specific context string ('server' or 'client' + 0x20*64 + 0x00 + transcript_hash)
      - Extension ordering in ClientHello is flexible per spec but some servers are sensitive
      - change_cipher_spec record may be received after ServerHello for middlebox compatibility; silently ignore it
    concepts:
      - TLS 1.3 handshake message sequence
      - Handshake state machine with strict ordering
      - Encrypted handshake phase (post-ServerHello)
      - CertificateVerify signature context
      - Transcript hash accumulation
    skills:
      - State machine design and enforcement
      - Protocol message construction and parsing
      - Encrypted message processing
      - Signature verification
    deliverables:
      - ClientHello builder with all required TLS 1.3 extensions
      - ServerHello parser with supported_versions validation
      - Handshake state machine enforcing message ordering
      - Transcript hash tracker updated with each handshake message
      - EncryptedExtensions processor
      - Finished message computation and verification (both sides)
      - Handshake-to-application traffic key transition
    estimated_hours: "15-22"

  - id: build-tls-m4
    name: Certificate Chain Verification
    description: >-
      Verify X.509 certificate chains from the server's Certificate
      message including chain building, signature verification, hostname
      matching, and CertificateVerify validation.
    acceptance_criteria:
      - Parse DER-encoded X.509 certificates from Certificate handshake message
      - Extract from each certificate: subject, issuer, validity (notBefore, notAfter), public key, extensions (SAN, Basic Constraints, Key Usage)
      - Build certificate chain from leaf certificate to trusted root CA (using system trust store or configured CA bundle)
      - Verify each certificate's digital signature using the issuer's public key (support RSA-PSS-SHA256 and ECDSA-SHA256 at minimum)
      - Reject expired certificates (notAfter < now) and not-yet-valid certificates (notBefore > now)
      - Verify leaf certificate hostname against Subject Alternative Name (SAN) extension (NOT Common Name per RFC 6125)
      - Support wildcard matching in SAN: *.example.com matches foo.example.com but NOT foo.bar.example.com
      - Verify Basic Constraints: intermediate CAs must have cA=TRUE; leaf must not
      - Verify CertificateVerify message: signature over (0x20*64 + context_string + 0x00 + transcript_hash) using server's public key
      - Reject self-signed certificates not present in trust store
      - Handle missing intermediate certificates gracefully (clear error message)
    pitfalls:
      - X.509 parsing requires ASN.1/DER decoding which is complex; consider using a library for this part
      - Certificate chain may be sent in wrong order by server; must sort by issuer/subject relationship
      - Hostname MUST be checked against SAN, not CN; CN matching is deprecated per RFC 6125
      - CertificateVerify context string is different for server ('TLS 1.3, server CertificateVerify') and client
      - Root CAs are self-signed; their signature is verified against their own public key
      - Revocation checking (CRL/OCSP) is important in production but complex; document as known limitation if not implemented
    concepts:
      - X.509 certificate structure and ASN.1/DER encoding
      - Certificate chain of trust model
      - Subject Alternative Name hostname matching
      - CertificateVerify signature in TLS 1.3
      - Basic Constraints and Key Usage extensions
    skills:
      - ASN.1/DER parsing (or library integration)
      - Certificate chain building
      - Digital signature verification (RSA-PSS, ECDSA)
      - Hostname verification per RFC 6125
    deliverables:
      - X.509 certificate parser (DER/ASN.1) extracting subject, issuer, validity, public key, extensions
      - Certificate chain builder from leaf to trusted root
      - Signature verification for each certificate in chain
      - Hostname verification against SAN with wildcard support
      - Validity date checking
      - CertificateVerify validation with TLS 1.3 context string
      - Basic Constraints and Key Usage verification
    estimated_hours: "12-18"

  - id: build-tls-m5
    name: Encrypted Records & Application Data
    description: >-
      Implement AEAD-encrypted record layer with inner content type,
      KeyUpdate support, and bidirectional encrypted communication.
    acceptance_criteria:
      - Encrypt outgoing records using AES-128-GCM with traffic key and nonce = XOR(write_iv, sequence_number)
      - Sequence numbers are 64-bit per-direction counters starting from 0; MUST NOT wrap
      - Encrypted record format: encrypt(plaintext + inner_content_type + padding) with additional data = record header (5 bytes)
      - Outer record always uses content_type=application_data (23) regardless of inner content type
      - Decrypt incoming records: decrypt, strip trailing zero padding, extract inner content type from last non-zero byte
      - Reject records that fail AEAD authentication (bad_record_mac alert)
      - Implement KeyUpdate message (type=24): on receipt, derive new read traffic secret; on send, derive new write traffic secret
      - KeyUpdate uses HKDF-Expand-Label(current_secret, "traffic upd", "", HashLen) to derive next secret
      - After KeyUpdate, reset sequence number to 0 for the updated direction
      - Send and receive application data (e.g., HTTP request/response) over encrypted channel
      - Implement graceful shutdown: send close_notify, wait for peer's close_notify, then close TCP
      - End-to-end test: complete TLS 1.3 handshake with a real server (e.g., example.com:443) and exchange HTTP data
    pitfalls:
      - Nonce reuse with AES-GCM is CATASTROPHIC; it allows authentication key recovery and arbitrary forgery
      - Inner content type byte is AFTER the plaintext and BEFORE padding in the encrypted record
      - Padding is optional zero bytes after inner content type; strip from right until non-zero byte found (that's the content type)
      - AEAD additional data is the 5-byte record header of the OUTER (encrypted) record, not the inner content type
      - KeyUpdate request_update flag (update_requested vs update_not_requested) determines if peer must also update
      - After switching to application keys, handshake-level messages (NewSessionTicket, KeyUpdate) arrive as encrypted records with inner type=handshake
    concepts:
      - AEAD encryption with nonce construction
      - TLS 1.3 encrypted record format with inner content type
      - KeyUpdate for traffic key rotation
      - Graceful TLS termination via close_notify
    skills:
      - AES-GCM authenticated encryption and decryption
      - Nonce management with sequence numbers
      - Post-handshake message handling
      - Encrypted record format with padding
    deliverables:
      - AEAD record encryption with AES-128-GCM and proper nonce construction
      - AEAD record decryption with authentication verification and padding removal
      - Inner content type handling (append on encrypt, extract on decrypt)
      - KeyUpdate message handling for traffic key rotation
      - Sequence number management with per-direction counters
      - Graceful shutdown with close_notify exchange
      - End-to-end test: full handshake + HTTP exchange with real TLS 1.3 server
    estimated_hours: "12-18"
```
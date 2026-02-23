# DOMAIN PROFILE: Security & Cryptography
# Applies to: security
# Projects: SHA-256, AES, TLS, sandbox, fuzzer, HTTPS, RBAC, vuln scanner, anti-cheat, etc.

## Fundamental Tension Type
Cryptographic and trust constraints. SECURITY vs USABILITY/PERFORMANCE. A single subtle bug makes the entire system worthless.

Secondary: defense depth vs complexity, confidentiality vs auditability, zero-trust vs performance.

## Three-Level View
- **Level 1 — Threat Model**: Attacker identity, capabilities, attack surface
- **Level 2 — Protocol/Algorithm Design**: Key exchange, auth, authorization flows
- **Level 3 — Implementation Pitfalls**: Side channels, timing attacks, padding oracles, overflows

## Soul Section: "Adversary Soul"
Think like an attacker: what can they observe (timing, errors, patterns), control (input, network, env), exploit (malformed input → overflow? info leak?). Is this constant-time? What's the weakest link? What assumptions could break (quantum, new math)?

For crypto-heavy projects (AES, TLS, ECDHE): also include math intuition — finite fields, elliptic curves, modular arithmetic. Explain the math the reader needs, with equations + plain-language pairing.

## Alternative Reality Comparisons
OpenSSL/BoringSSL, WireGuard, Signal Protocol, seccomp/AppArmor/SELinux, HashiCorp Vault, Chromium sandbox, AWS IAM, NaCl/libsodium.

## TDD Emphasis
- Threat model: MANDATORY
- Protocol state machine: MANDATORY for crypto protocols
- Bit-level spec: YES for crypto primitives
- Constant-time analysis: MANDATORY for code handling secrets
- Error handling: CRITICAL — errors must not leak info
- Tests: known-answer tests (KATs), malformed input, timing uniformity
- Memory layout: YES for crypto/protocol messages. NO for policy logic.
- Benchmarks: crypto ops/sec, handshake latency, scan coverage

## Cross-Domain Awareness
May need systems knowledge (sandboxing, namespaces, cgroups, kernel modules), web knowledge (CSRF/XSS, OAuth), or distributed concepts (mTLS, distributed auth).



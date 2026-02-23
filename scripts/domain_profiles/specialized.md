# DOMAIN PROFILE: Specialized & Cross-Domain
# Applies to: specialized (fallback for diverse projects)
# Projects: BitTorrent, browser, DNS, debugger, emulator, Git, LSP, VPN, WASM runtime, SLAM, etc.

## About This Profile
Diverse projects — each is its own sub-domain. Identify which existing profile is CLOSEST and borrow its conventions.

## Fundamental Tension Type
Project-specific:
- **Protocol implementations** (DNS, BitTorrent, VPN, QUIC): Spec compliance vs real-world compatibility.
- **Tool implementations** (debugger, LSP, editor, Git): Power vs responsiveness on huge inputs.
- **Simulation/Science** (SLAM, bioinformatics, finance): Accuracy vs computational feasibility.
- **Runtimes/Emulators** (WASM, emulator): Spec faithfulness vs execution speed.

## Three-Level View (choose per project)
- Protocol: Wire Format → State Machine → Network/OS
- Tool: User API → Core Engine → File System/OS
- Simulation: Math Model → Numerical Methods → Compute
- Runtime: Guest Program → Engine → Host Hardware

## Soul Section
- Protocol → Specification Soul: What does the RFC say vs what real impls do?
- Tool → Responsiveness Soul: Latency budget, million-line files, UI responsive while processing?
- Simulation → Accuracy Soul: Error bounds, approximation quality, "good enough" threshold?
- Runtime → Compliance Soul: Which spec tests pass? Edge cases? Undefined behavior?

## Alternative Reality (find 3-4 per project)
DNS: BIND, CoreDNS, Unbound. Git: libgit2, JGit. Browser: Blink, Gecko, Servo. WASM: Wasmtime, Wasmer, V8. Editor: Vim, Helix, Zed. Debugger: GDB, LLDB, rr.

## TDD Emphasis
- Protocol: wire format byte-level, state machine, test against reference
- Tool: I/O spec, large-input benchmarks, plugin API
- Simulation: math spec with error bounds, validation vs known solutions
- Runtime: spec compliance suite, edge case catalog
- Memory layout: YES for wire formats, emulator memory maps. NO for tool logic.

## Cross-Domain Awareness
This domain borrows heavily. The project likely touches networking (→ systems), file formats (→ storage), parsing (→ compilers), rendering (→ game), math (→ AI/ML), or protocols (→ distributed/security).




## Artist Examples for This Domain
- **data_walk**: Specific logic flow for the niche domain.
- **structure_layout**: Domain-specific data structures.
- **state_evolution**: Lifecycle of specialized entities.

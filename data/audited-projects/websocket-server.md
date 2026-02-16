# AUDIT & FIX: websocket-server

## CRITIQUE
- **No Audit Findings:** The audit returned no findings, but the project has issues.
- **M3/M4 Overlap:** M3 (Connection Management) and M4 (Ping/Pong Heartbeat) have significant overlap. M3 deliverables already include 'Ping and pong frames for keep-alive' and 'Close handshake implementation', yet M4 is entirely dedicated to ping/pong. This is redundant. M3 also mentions broadcasting in its concepts but has no broadcasting AC—broadcasting was removed but the concept lingers.
- **M3 Missing Broadcasting AC:** The deliverables and concepts mention broadcasting/fan-out but none of the ACs test broadcasting a message to multiple connections. For a WebSocket server, this is a core capability.
- **M4 Scope Too Narrow:** An entire milestone (7.5 hours) dedicated solely to ping/pong is excessive. Ping/pong is a single control frame type. This should be folded into M3.
- **Missing Security:** No milestone addresses origin validation, message size limits, or rate limiting for connected clients. A WebSocket server without these is trivially DoS-able.
- **Missing Text vs Binary Distinction:** M2 AC mentions 'Support text and binary frame types with correct UTF-8 validation for text' which is good, but there's no AC for sending frames (only parsing/receiving). The server needs to construct and send frames too.
- **SHA-1 Concern:** The essence and skills mention SHA-1 hashing, which is correct for the WebSocket handshake (RFC 6455 mandates it). However, it should be noted that this is protocol-mandated and NOT a general security recommendation—SHA-1 is cryptographically broken for signatures.
- **Estimated Hours:** All four milestones are exactly 7.5 hours each. This uniform distribution is unrealistic; frame parsing is significantly more complex than the handshake.
- **Overall:** The project would benefit from consolidating M3 and M4 and adding a proper milestone for message sending, broadcasting, and security hardening.

## FIXED YAML
```yaml
id: websocket-server
name: WebSocket Server (From Scratch)
description: >-
  Implement a WebSocket server from the ground up, handling the HTTP upgrade
  handshake, binary frame parsing and construction, connection lifecycle
  management, and multi-client broadcasting.
difficulty: intermediate
estimated_hours: "25-35"
essence: >-
  Binary framing protocol with masking/unmasking operations, SHA-1-based
  handshake negotiation over HTTP upgrade, and stateful connection lifecycle
  management over raw TCP sockets with opcode-driven message demultiplexing
  and fan-out broadcasting to multiple concurrent connections.
why_important: >-
  Implementing a WebSocket server from scratch teaches you binary protocol
  engineering, TCP connection management, and the mechanics behind real-time
  bidirectional communication that powers chat apps, live updates, gaming,
  and collaborative tools.
learning_outcomes:
  - Implement the WebSocket handshake including SHA-1 key derivation per RFC 6455
  - Parse and construct WebSocket frames with correct masking and payload length encoding
  - Handle text and binary message types with UTF-8 validation and fragmentation
  - Manage multiple concurrent connections with proper lifecycle tracking
  - Implement ping/pong heartbeat for dead connection detection
  - Build message broadcasting to multiple connected clients
  - Secure the server with origin validation, message size limits, and per-connection rate limiting
skills:
  - Binary Protocol Design
  - HTTP Upgrade Handshake
  - Frame Parsing & Construction
  - Connection State Management
  - SHA-1 Hashing (protocol-mandated)
  - TCP Socket Programming
  - Heartbeat Mechanisms
  - Concurrent Connection Handling
tags:
  - bidirectional
  - frames
  - intermediate
  - networking
  - protocols
  - real-time
  - service
  - upgrade
architecture_doc: architecture-docs/websocket-server/index.md
languages:
  recommended:
    - Go
    - Python
    - Rust
  also_possible:
    - C
    - Java
resources:
  - name: RFC 6455 - The WebSocket Protocol
    url: https://datatracker.ietf.org/doc/html/rfc6455
    type: documentation
  - name: Writing WebSocket Servers - MDN
    url: https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API/Writing_WebSocket_servers
    type: tutorial
  - name: WebSocket Tutorial - JavaScript.info
    url: https://javascript.info/websocket
    type: tutorial
prerequisites:
  - type: project
    id: http-server-basic
    name: HTTP Server (Basic)
  - type: skill
    name: TCP socket programming
  - type: skill
    name: Binary/bitwise operations
milestones:
  - id: websocket-server-m1
    name: HTTP Upgrade Handshake
    description: >-
      Implement the WebSocket handshake by parsing the HTTP upgrade request,
      validating headers, computing Sec-WebSocket-Accept, and switching
      protocols.
    acceptance_criteria:
      - "Server listens on a TCP socket and reads incoming HTTP requests line by line"
      - "Upgrade request is detected by checking for Connection: Upgrade and Upgrade: websocket headers (case-insensitive)"
      - "Sec-WebSocket-Version header is validated; versions other than 13 are rejected with 426 Upgrade Required"
      - "Sec-WebSocket-Key header is extracted, concatenated with the magic GUID '258EAFA5-E914-47DA-95CA-5AB0DC85B11', SHA-1 hashed, and Base64-encoded to produce Sec-WebSocket-Accept"
      - "Server responds with '101 Switching Protocols' including Connection: Upgrade, Upgrade: websocket, and Sec-WebSocket-Accept headers"
      - "After the 101 response, the connection switches to WebSocket framing mode; subsequent data is interpreted as WebSocket frames"
      - "Non-upgrade HTTP requests receive a 400 Bad Request response"
    pitfalls:
      - "Missing CRLF (\\r\\n) termination on response headers causes handshake failure with most clients"
      - "Case-sensitive header name comparison breaks with clients that send lowercase headers; always compare case-insensitively"
      - "SHA-1 is used here because RFC 6455 mandates it for the handshake; this is NOT a security hash—do not use SHA-1 for other purposes"
      - "Not validating Sec-WebSocket-Version allows connections with unsupported protocol versions"
    concepts:
      - HTTP header parsing per RFC 2616/7230
      - SHA-1 hash for WebSocket key derivation (protocol-mandated)
      - Base64 encoding
      - HTTP upgrade mechanism and protocol switching
    skills:
      - HTTP Protocol Implementation
      - Cryptographic Hash Functions
      - String Parsing
      - Protocol Handshake Design
    deliverables:
      - TCP listener accepting incoming connections
      - HTTP request parser extracting method, path, and headers
      - Upgrade detection and Sec-WebSocket-Accept computation
      - 101 Switching Protocols response with correct headers
    estimated_hours: "5-7"

  - id: websocket-server-m2
    name: Frame Parsing & Construction
    description: >-
      Parse incoming WebSocket frames and construct outgoing frames,
      handling masking, payload length encoding, opcodes, and fragmentation.
    acceptance_criteria:
      - "Incoming frames are parsed extracting: FIN bit, opcode (4 bits), MASK bit, payload length (7-bit, 16-bit, or 64-bit extended), masking key (4 bytes if masked), and payload data"
      - "Client frames are unmasked by XOR-ing each payload byte with the corresponding byte of the 4-byte masking key; frames from clients without masking are rejected with 1002 Protocol Error close"
      - "Server-to-client frames are constructed WITHOUT masking (servers MUST NOT mask per RFC 6455)"
      - "Text frames (opcode 0x1) have their payload validated as valid UTF-8; invalid UTF-8 triggers a 1007 Invalid Payload close"
      - "Binary frames (opcode 0x2) are handled without UTF-8 validation"
      - "Fragmented messages (FIN=0 with continuation frames opcode 0x0) are buffered and reassembled into a complete message when the final FIN=1 frame arrives"
      - "Control frames (ping 0x9, pong 0xA, close 0x8) are handled even when interleaved between fragmented message frames"
      - "Payload lengths exceeding a configurable maximum (e.g., 1MB) are rejected to prevent memory exhaustion"
    pitfalls:
      - "Forgetting to unmask client frames produces garbled data; clients MUST mask per RFC 6455"
      - "64-bit payload length can cause integer overflow on 32-bit systems or memory exhaustion without size limits"
      - "Not handling fragmented messages (FIN=0 continuation frames) breaks compatibility with clients that fragment large messages"
      - "Control frames (ping/pong/close) can appear between fragments of a data message and must be handled immediately"
      - "Server frames MUST NOT be masked; masking server frames causes compliant clients to reject them"
    concepts:
      - WebSocket frame binary structure (2+ byte header)
      - XOR masking/unmasking algorithm
      - Variable-length payload encoding (7-bit, 16-bit, 64-bit)
      - Opcodes for data (text, binary) and control (ping, pong, close) frames
      - Message fragmentation and reassembly
    skills:
      - Binary Protocol Parsing
      - Bitwise Operations
      - Buffer Management
      - UTF-8 Validation
    deliverables:
      - Frame parser extracting all header fields and payload
      - XOR unmasking for client frames
      - Frame constructor for server-to-client messages (text, binary, control)
      - Fragmented message reassembly buffer
      - Maximum payload size enforcement
    estimated_hours: "8-10"

  - id: websocket-server-m3
    name: Connection Management, Heartbeat & Broadcasting
    description: >-
      Manage multiple concurrent WebSocket connections with lifecycle tracking,
      ping/pong heartbeat, clean close handshake, and message broadcasting.
    acceptance_criteria:
      - "Each connection is assigned a unique ID and tracked in a thread-safe connection registry with metadata (connected_at, remote_addr)"
      - "Connection states (OPEN, CLOSING, CLOSED) are tracked; operations on non-OPEN connections are rejected"
      - "Server sends ping frames at a configurable interval (e.g., 30 seconds); pong responses reset the liveness timer"
      - "Connections that do not respond to ping within a configurable timeout are terminated and removed from the registry"
      - "Close handshake: when server initiates close, it sends a close frame with status code, waits for the client's close frame response, then closes the TCP connection"
      - "Close handshake: when client sends a close frame, server responds with a close frame echoing the status code, then closes the TCP connection"
      - "Broadcast method sends a message to all OPEN connections except the sender; errors on individual connections do not affect other recipients"
      - "Disconnected connections are cleaned up from the registry immediately; no resource leaks"
    pitfalls:
      - "Using wall-clock time for ping intervals fails with system clock changes; use monotonic time"
      - "Not initializing last_pong_time on connection establishment causes immediate timeout"
      - "Too aggressive ping interval (< 10s) wastes bandwidth; too passive (> 60s) delays dead connection detection"
      - "Modifying the connection registry while iterating over it for broadcast causes concurrent modification errors"
      - "Broadcast errors on one connection (write to closed socket) must not abort the entire broadcast loop"
      - "Memory leak from not removing closed connections; always clean up in a finally/defer block"
    concepts:
      - Concurrent connection handling with event loops or threads
      - Thread-safe data structures for connection registry
      - Ping/pong control frames for keepalive (RFC 6455 §5.5.2/5.5.3)
      - Close handshake protocol (RFC 6455 §7)
      - Fan-out broadcasting pattern
    skills:
      - Concurrency and Synchronization
      - Resource Lifecycle Management
      - Timer-Based Health Checking
      - Broadcasting to Multiple Connections
      - Graceful Shutdown
    deliverables:
      - Thread-safe connection registry with add, remove, and iterate operations
      - Connection lifecycle state machine (OPEN → CLOSING → CLOSED)
      - Ping sender with configurable interval and pong-based liveness tracking
      - Dead connection detection and cleanup
      - Close handshake implementation (server-initiated and client-initiated)
      - Broadcast method with error isolation per connection
    estimated_hours: "8-10"

  - id: websocket-server-m4
    name: Security Hardening & Integration Testing
    description: >-
      Add security protections and verify the complete server implementation
      with integration tests against real WebSocket clients.
    acceptance_criteria:
      - "Origin header is validated against a configurable allowlist during the handshake; disallowed origins receive 403 Forbidden"
      - "Per-connection message rate limiting rejects clients sending more than a configurable number of messages per second with a 1008 Policy Violation close"
      - "Maximum message size (after reassembly of fragments) is enforced; oversized messages trigger a 1009 Message Too Big close"
      - "Integration test: standard WebSocket client (e.g., browser, wscat) successfully completes handshake and exchanges text messages"
      - "Integration test: fragmented message sent by client is correctly reassembled by server"
      - "Integration test: server detects dead connection via ping timeout and cleans up within 2x the configured timeout"
      - "Integration test: broadcast message is received by all connected clients except the sender"
      - "Integration test: close handshake completes cleanly with correct status code exchange"
    pitfalls:
      - "Not validating Origin allows cross-site WebSocket hijacking attacks"
      - "Rate limiting must be per-connection, not global; a single abusive client should not affect others"
      - "Integration tests with real WebSocket clients can be flaky due to timing; use reasonable timeouts and retries"
      - "Forgetting to test binary frames, not just text frames"
    concepts:
      - Origin-based access control
      - Per-connection rate limiting
      - Message size enforcement
      - Cross-site WebSocket hijacking prevention
      - Integration testing with real protocol clients
    skills:
      - Security Hardening
      - Rate Limiting per Connection
      - Integration Testing
      - Protocol Compliance Verification
    deliverables:
      - Origin validation during handshake with configurable allowlist
      - Per-connection message rate limiter
      - Message size limit enforcement with appropriate close codes
      - Integration test suite verifying handshake, messaging, fragmentation, heartbeat, broadcast, and close
    estimated_hours: "5-8"
```
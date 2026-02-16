# AUDIT & FIX: http2-server

## CRITIQUE
- CRITICAL: Missing Connection Preface milestone. RFC 7540 §3.5 mandates the client send a 24-byte magic string ('PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n') before any frames. Without validating this, the server cannot distinguish HTTP/2 from HTTP/1.1 or garbage. This is not optional.
- CRITICAL: Missing SETTINGS exchange. RFC 7540 §3.5 requires both endpoints to send a SETTINGS frame immediately after the connection preface. The server MUST acknowledge the client's SETTINGS with a SETTINGS ACK. Without this, negotiation of SETTINGS_MAX_CONCURRENT_STREAMS, SETTINGS_INITIAL_WINDOW_SIZE, SETTINGS_HEADER_TABLE_SIZE, etc. never occurs, making all subsequent milestones operate on undefined assumptions.
- The original Milestone 1 jumps straight to frame parsing without establishing the connection lifecycle. A server that can parse frames but doesn't validate the connection preface is non-compliant and will fail against any real HTTP/2 client.
- Milestone 3 mentions 'Priority handling' and 'dependency tree' but RFC 9113 (which supersedes RFC 7540) deprecates stream priority. The project should note this but still implement it for educational purposes against RFC 7540.
- Milestone 4 AC says 'WINDOW_UPDATE frames correctly increment the receiver's available send window' — this is backwards. WINDOW_UPDATE increments the SENDER's available window as seen by the receiver. The wording conflates perspectives.
- Flow control estimated hours (9-18) have an unreasonably wide range suggesting poorly defined scope.
- No mention of CONTINUATION frames, which are required when HEADERS don't fit in a single frame.
- No mention of GOAWAY handling for graceful shutdown.
- No mention of TLS/ALPN negotiation which is how HTTP/2 is actually initiated in practice (h2 vs h2c).
- Resources reference RFC 7540 but not RFC 9113 which is the current HTTP/2 specification.
- HPACK milestone doesn't mention SETTINGS_HEADER_TABLE_SIZE acknowledgment affecting dynamic table size.

## FIXED YAML
```yaml
id: http2-server
name: "HTTP/2 Server"
description: "Binary framing, HPACK compression, multiplexed streams, and flow control"
difficulty: advanced
estimated_hours: "40-60"
essence: >
  Connection preface validation and SETTINGS negotiation bootstrapping the protocol,
  binary framing layer with 9-byte header parsing and frame type dispatch,
  multiplexed stream state machines managing concurrent request/response pairs over
  a single TCP connection, HPACK header compression using static/dynamic table
  indexing with Huffman coding, and dual-level window-based flow control preventing
  buffer overflow at both connection and individual stream granularity.
why_important: >
  Building an HTTP/2 server demystifies modern web infrastructure by teaching you
  how browsers and servers actually communicate at the binary protocol level,
  preparing you for performance optimization, protocol design, and low-level
  network programming roles.
learning_outcomes:
  - Validate the HTTP/2 connection preface and perform SETTINGS handshake
  - Implement binary frame parsing and serialization for all HTTP/2 frame types
  - Design state machines to manage stream lifecycle transitions per RFC 7540 §5.1
  - Build HPACK encoder/decoder with static table, dynamic table, and Huffman coding
  - Implement dual-level window-based flow control with WINDOW_UPDATE handling
  - Handle stream prioritization using dependency trees and weight calculations
  - Debug binary protocol implementations using frame-level inspection tools
  - Implement concurrent stream multiplexing over a single TCP connection
  - Design robust error handling with GOAWAY and RST_STREAM for protocol violations
skills:
  - Binary Protocol Design
  - State Machine Implementation
  - Header Compression (HPACK)
  - Flow Control Algorithms
  - Stream Multiplexing
  - Network Protocol Debugging
  - Concurrent Programming
  - Performance Optimization
tags:
  - advanced
  - binary
  - c
  - go
  - hpack
  - multiplexing
  - networking
  - rust
  - service
  - streaming
  - streams
architecture_doc: architecture-docs/http2-server/index.md
languages:
  recommended:
    - Go
    - Rust
    - C
  also_possible:
    - Java
    - Python
resources:
  - type: specification
    name: "RFC 9113 - HTTP/2 (current)"
    url: "https://httpwg.org/specs/rfc9113.html"
  - type: specification
    name: "RFC 7540 - HTTP/2 (original)"
    url: "https://httpwg.org/specs/rfc7540.html"
  - type: specification
    name: "RFC 7541 - HPACK"
    url: "https://httpwg.org/specs/rfc7541.html"
  - type: tool
    name: "h2spec - HTTP/2 conformance testing tool"
    url: "https://github.com/summerwind/h2spec"
prerequisites:
  - type: skill
    name: "HTTP/1.1 server implementation"
  - type: skill
    name: "TLS fundamentals"
  - type: skill
    name: "Binary protocol experience"
milestones:
  - id: http2-server-m1
    name: "Connection Preface & SETTINGS Handshake"
    description: >
      Accept TCP connections, validate the 24-byte HTTP/2 client connection preface,
      exchange initial SETTINGS frames, and acknowledge with SETTINGS ACK.
    acceptance_criteria:
      - Server reads exactly 24 bytes and validates against the magic string 'PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n'; rejects connection on mismatch
      - Server sends its own SETTINGS frame immediately upon connection establishment before reading any client frames
      - Server parses the client's initial SETTINGS frame extracting key parameters (SETTINGS_MAX_CONCURRENT_STREAMS, SETTINGS_INITIAL_WINDOW_SIZE, SETTINGS_HEADER_TABLE_SIZE)
      - Server sends SETTINGS ACK (SETTINGS frame with ACK flag set, zero-length payload) within a reasonable timeout after receiving client SETTINGS
      - Server stores negotiated settings and applies them to all subsequent frame processing (e.g., initial window size, max concurrent streams)
      - Connection is terminated with GOAWAY and PROTOCOL_ERROR if preface is invalid or SETTINGS ACK is not received within timeout
    pitfalls:
      - Reading the 24-byte preface in multiple TCP reads due to Nagle or partial delivery; must buffer until complete
      - Sending frames before receiving the client preface, violating protocol ordering
      - Confusing SETTINGS_INITIAL_WINDOW_SIZE (per-stream) with the connection-level flow control window (always 65535 initially)
      - Forgetting that SETTINGS_HEADER_TABLE_SIZE changes must be acknowledged by the HPACK decoder via a dynamic table size update
    concepts:
      - HTTP/2 connection lifecycle
      - Protocol negotiation
      - SETTINGS parameter semantics
    skills:
      - TCP stream reading and buffering
      - Protocol state machine bootstrapping
      - Configuration negotiation
    deliverables:
      - TCP listener accepting connections on a configurable port
      - Connection preface validator reading and comparing the 24-byte magic string
      - SETTINGS frame parser extracting parameter ID-value pairs from the payload
      - SETTINGS ACK sender responding to client SETTINGS with an acknowledgment frame
      - Negotiated settings storage applied to connection-scoped parameters
    estimated_hours: "3-5"

  - id: http2-server-m2
    name: "Binary Framing Layer"
    description: >
      Parse and serialize HTTP/2 frames with the 9-byte header format.
      Support all core frame types.
    acceptance_criteria:
      - Frame header parser reads exactly 9 bytes extracting 24-bit length, 8-bit type, 8-bit flags, 1-bit reserved, and 31-bit stream ID with correct network byte order (big-endian)
      - Parser supports frame types DATA (0x0), HEADERS (0x1), PRIORITY (0x2), RST_STREAM (0x3), SETTINGS (0x4), PUSH_PROMISE (0x5), PING (0x6), GOAWAY (0x7), WINDOW_UPDATE (0x8), CONTINUATION (0x9)
      - Frame flags are correctly interpreted per frame type (e.g., END_STREAM=0x1 on DATA/HEADERS, END_HEADERS=0x4 on HEADERS/CONTINUATION, PADDED=0x8, PRIORITY=0x20)
      - Payload extraction reads exactly the number of bytes specified by the length field; connection error on payload exceeding SETTINGS_MAX_FRAME_SIZE (default 16384)
      - Frame serializer produces correctly formatted binary output that round-trips through the parser without data loss
      - Reserved bit in stream ID field is ignored on read and set to zero on write
    pitfalls:
      - Endianness errors: HTTP/2 uses network byte order (big-endian) for all multi-byte fields; x86 is little-endian
      - Forgetting CONTINUATION frames which must follow HEADERS/PUSH_PROMISE when headers exceed max frame size; no other frame types may be interleaved
      - Off-by-one in the 24-bit length field extraction (it's 3 bytes, not 4)
      - Not enforcing SETTINGS_MAX_FRAME_SIZE causing buffer overflows on malicious payloads
      - Treating the reserved bit as part of the stream ID, producing incorrect stream identification
    concepts:
      - Binary frame format (RFC 7540 §4.1)
      - Frame type dispatch
      - Protocol-level validation
    skills:
      - Binary protocol parsing
      - Byte-level manipulation and endianness
      - Network protocol serialization
    deliverables:
      - 9-byte frame header parser with big-endian field extraction
      - Frame type enum covering all 10 standard frame types
      - Frame serializer encoding frames into binary wire format
      - Frame validator enforcing max frame size and reserved bit constraints
      - CONTINUATION frame handling for multi-frame header blocks
    estimated_hours: "5-8"

  - id: http2-server-m3
    name: "HPACK Header Compression"
    description: >
      Implement HPACK (RFC 7541) header compression with static table,
      dynamic table, Huffman coding, and integer encoding.
    acceptance_criteria:
      - Static table lookup resolves all 61 predefined header name-value entries (RFC 7541 Appendix A) by index
      - Dynamic table adds new entries at the front (index 62+), evicts oldest entries (FIFO from the back) when table size exceeds SETTINGS_HEADER_TABLE_SIZE, and correctly handles zero-size dynamic table
      - Huffman decoder decompresses encoded header values using the static Huffman table from RFC 7541 Appendix B; encoder produces valid Huffman-coded output with EOS padding
      - Integer encoder/decoder handles prefixed integers with 5-bit, 6-bit, and 7-bit prefix sizes including multi-byte encoding for values >= 2^N-1
      - Header field representations are correctly parsed: indexed (bit pattern 1xxxxxxx), literal with incremental indexing (01xxxxxx), literal without indexing (0000xxxx), and literal never indexed (0001xxxx)
      - Dynamic table size update signals (001xxxxx) are processed and the decoder adjusts its maximum table size accordingly
      - Encoder and decoder round-trip test passes: encoding a header list then decoding produces the identical header list
    pitfalls:
      - Off-by-one errors in static table indexing (index 0 is invalid, index 1 is the first entry)
      - Not handling dynamic table size update signals which the encoder MUST send when SETTINGS_HEADER_TABLE_SIZE changes
      - Huffman boundary handling with EOS symbol padding (must be at most 7 bits of 1s)
      - Integer encoding edge case when value exactly equals 2^N-1 requiring one additional byte
      - Sensitive headers (e.g., cookies, auth tokens) should use 'never indexed' representation to prevent CRIME-style attacks
    concepts:
      - Header compression (RFC 7541)
      - Static and dynamic table indexing
      - Huffman coding
      - Prefix-coded integers
    skills:
      - Compression algorithm implementation
      - Dynamic data structure management with size constraints
      - Huffman tree traversal and bit-level I/O
      - Specification compliance testing
    deliverables:
      - Static table with all 61 predefined header entries indexed correctly
      - Dynamic table with FIFO eviction, configurable maximum size, and size update signaling
      - Huffman encoder/decoder using the RFC 7541 Appendix B static Huffman table
      - Integer encoder/decoder for 5-bit, 6-bit, and 7-bit prefix formats
      - Header block encoder supporting indexed, literal-with-indexing, and literal-never-indexed representations
      - Header block decoder reconstructing full header lists from compressed wire format
    estimated_hours: "8-12"

  - id: http2-server-m4
    name: "Stream State Machine & Multiplexing"
    description: >
      Implement the HTTP/2 stream lifecycle state machine and handle
      multiplexed concurrent streams over a single TCP connection.
    acceptance_criteria:
      - Stream state machine implements all transitions: idle → open (via HEADERS), open → half-closed(local) (via END_STREAM sent), open → half-closed(remote) (via END_STREAM received), half-closed → closed, with proper handling of RST_STREAM from any state
      - Client-initiated streams use odd-numbered stream IDs; server-initiated (push) streams use even-numbered IDs; stream 0 is reserved for connection-level frames
      - New stream IDs are monotonically increasing; receiving a stream ID lower than a previously seen maximum triggers PROTOCOL_ERROR
      - Concurrent stream limit enforced per SETTINGS_MAX_CONCURRENT_STREAMS; new streams beyond the limit receive RST_STREAM with REFUSED_STREAM
      - Server correctly processes a complete request (HEADERS + optional DATA with END_STREAM) and sends a response (HEADERS + optional DATA with END_STREAM) on the same stream
      - Multiple streams are processed concurrently (interleaved frame processing) over the single TCP connection without head-of-line blocking between streams
    pitfalls:
      - State machine violations causing STREAM_CLOSED errors when sending frames on already-closed streams
      - Stream ID exhaustion on long-lived connections (2^31-1 maximum); must GOAWAY and reconnect
      - Race between receiving RST_STREAM and sending pending frames on the same stream
      - Forgetting that stream 0 frames (SETTINGS, PING, GOAWAY) are connection-level and must not carry a non-zero stream ID
      - Not handling priority/dependency fields in HEADERS frames (even if ignoring priority, the bytes must be parsed)
    concepts:
      - Stream state machine (RFC 7540 §5.1)
      - Multiplexing and concurrency
      - Connection vs stream scope
    skills:
      - Finite state machine implementation
      - Concurrent stream management
      - Resource lifecycle and cleanup
    deliverables:
      - Stream state machine with idle, reserved, open, half-closed(local/remote), and closed states
      - Stream registry tracking all active streams with their state and metadata
      - Request/response processing receiving HEADERS+DATA and sending response HEADERS+DATA per stream
      - RST_STREAM handling for stream-level error signaling and cancellation
      - GOAWAY frame sending for graceful connection shutdown with last-stream-ID
      - Concurrent frame interleaving across multiple active streams
    estimated_hours: "8-12"

  - id: http2-server-m5
    name: "Flow Control"
    description: >
      Implement connection-level and stream-level flow control using
      WINDOW_UPDATE frames to prevent buffer overflow.
    acceptance_criteria:
      - Initial flow control window is 65535 bytes for both connection-level and each new stream, as mandated by RFC 7540 §6.9.2
      - Sending a DATA frame decrements both the stream-level and connection-level send windows by the frame's payload length (including padding if present)
      - Receiving a WINDOW_UPDATE frame on stream 0 increments the connection-level send window; on a specific stream ID, increments that stream's send window
      - Sender queues DATA frames when the send window (stream or connection) reaches zero; queued frames are sent when a WINDOW_UPDATE restores available capacity
      - Flow control window overflow (increment causing window to exceed 2^31-1) triggers FLOW_CONTROL_ERROR on the stream or connection as appropriate
      - Server sends WINDOW_UPDATE frames to the client to indicate consumed DATA has been processed and more data can be sent
      - SETTINGS_INITIAL_WINDOW_SIZE changes retroactively adjust all open stream windows by the delta between old and new values
    pitfalls:
      - Window underflow: sending more DATA bytes than the current window allows, violating flow control
      - Deadlock when both connection and stream windows are exhausted and neither side sends WINDOW_UPDATE
      - Forgetting that flow control applies to DATA frames only, not HEADERS or other control frames
      - Not accounting for padding bytes in flow control window calculations
      - SETTINGS_INITIAL_WINDOW_SIZE change can make existing stream windows negative, which is legal but means no DATA can be sent until WINDOW_UPDATE brings it positive
    concepts:
      - Dual-level flow control (connection + stream)
      - Backpressure and buffering
      - Window arithmetic and overflow protection
    skills:
      - Window-based flow control algorithms
      - Backpressure handling and queuing
      - Concurrent resource accounting
      - Edge case handling for overflow/underflow
    deliverables:
      - Connection-level flow control window tracking aggregate DATA capacity across all streams
      - Per-stream flow control window tracking individual stream DATA capacity
      - WINDOW_UPDATE frame handler incrementing appropriate window on receipt
      - WINDOW_UPDATE frame sender signaling consumed capacity to the peer
      - Send queue holding DATA frames blocked by insufficient window capacity
      - SETTINGS_INITIAL_WINDOW_SIZE change handler adjusting all open stream windows
    estimated_hours: "8-12"
```
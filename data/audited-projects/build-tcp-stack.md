# AUDIT & FIX: build-tcp-stack

## CRITIQUE
- **TCP Checksum Omission**: Milestone 3 (TCP Connection Management) completely omits TCP checksum calculation and verification. TCP checksums are more complex than IP checksums because they require a pseudo-header containing IP source/destination addresses, protocol, and TCP length. This is a critical correctness issue — without checksum verification, the stack will accept corrupted segments.
- **No MSS Negotiation**: Maximum Segment Size (MSS) negotiation happens during the SYN/SYN-ACK exchange via TCP options. Without it, the stack will either send segments too large (causing IP fragmentation) or too small (wasting bandwidth). The original milestones completely ignore TCP options parsing.
- **IP Fragmentation Handling Missing**: Milestone 2 lists fragmentation as a pitfall but has zero ACs for reassembly. Real-world stacks must handle fragmented packets or they will silently drop legitimate traffic.
- **Sequence Number Wraparound**: Listed as a pitfall in M3 and M4 but never addressed in ACs. RFC 1323 PAWS (Protection Against Wrapped Sequences) and RFC 7323 timestamps are essential for correctness on high-bandwidth links.
- **No Mention of TIME_WAIT State**: The four-way FIN handshake AC doesn't mention TIME_WAIT (2*MSL), which is critical for preventing stale segment acceptance on connection reuse.
- **Congestion Control Scope Creep**: Milestone 4 mixes flow control (receiver window) with congestion control (slow start, congestion avoidance). These are distinct mechanisms. Slow start is a congestion control algorithm (RFC 5681), not flow control. The AC for slow start is incomplete — no mention of congestion avoidance phase, ssthresh, or fast retransmit/fast recovery.
- **No Out-of-Order Segment Handling AC**: Listed as a skill in M4 but no AC requires buffering and reordering out-of-sequence segments.
- **Missing RTO Calculation**: Retransmission timeout is mentioned but there's no AC for Karn's algorithm or Jacobson's RTO estimation (RFC 6298), which are essential for correct timer behavior.
- **Estimated Hours Seem Low**: 60-100 hours for a full TCP/IP stack is aggressive. The TCP data transfer milestone alone (sliding window + congestion control + retransmission) is easily 30+ hours for correct implementation.

## FIXED YAML
```yaml
id: build-tcp-stack
name: Build Your Own TCP/IP Stack
description: >-
  Implement a userspace TCP/IP network stack from scratch, handling Ethernet frame
  parsing, ARP resolution, IP routing, ICMP, and a full TCP implementation with
  connection management, reliable delivery, and flow/congestion control.
difficulty: expert
estimated_hours: "80-120"
essence: >-
  Byte-level protocol parsing with correct endianness handling, state machine
  management for TCP connection lifecycle, and implementing reliable ordered
  delivery over unreliable packet networks through sequence numbers,
  acknowledgments, retransmission timers, sliding window flow control, and
  congestion control algorithms.
why_important: >-
  Building a TCP/IP stack from scratch teaches fundamental network programming
  concepts that underpin all modern distributed systems, from databases to
  microservices, and provides deep insight into how operating systems manage
  network communication at the kernel level.
learning_outcomes:
  - Implement Ethernet frame parsing and ARP protocol for hardware address resolution
  - Design and implement IP packet routing with ICMP echo request/reply (ping)
  - Handle IP fragmentation and reassembly for packets exceeding MTU
  - Build TCP connection establishment using 3-way handshake and full state machine transitions including TIME_WAIT
  - Implement TCP checksum calculation with pseudo-header covering IP fields
  - Negotiate TCP options including Maximum Segment Size (MSS) during connection setup
  - Implement sliding window flow control with sequence numbers and acknowledgments
  - Handle out-of-order segment buffering and reordering
  - Implement retransmission with adaptive RTO calculation (Jacobson/Karn's algorithm)
  - Build congestion control with slow start, congestion avoidance, fast retransmit, and fast recovery
  - Debug low-level network protocols using packet capture and analysis tools
  - Manage concurrent connection state across multiple TCP sessions
skills:
  - Network Protocol Implementation
  - Binary Protocol Parsing
  - State Machine Design
  - Socket Programming
  - Flow Control Algorithms
  - Congestion Control Algorithms
  - Packet Analysis
  - Low-Level Systems Programming
  - Concurrency Management
tags:
  - build-from-scratch
  - c
  - congestion
  - data-structures
  - expert
  - flow-control
  - go
  - ip
  - networking
  - packets
  - rust
architecture_doc: architecture-docs/build-tcp-stack/index.md
languages:
  recommended:
    - C
    - Rust
    - Go
  also_possible: []
resources:
  - type: book
    name: "TCP/IP Illustrated Vol 1"
    url: https://www.amazon.com/TCP-Illustrated-Vol-Addison-Wesley-Professional/dp/0201633469
  - type: specification
    name: "RFC 793 - TCP"
    url: https://tools.ietf.org/html/rfc793
  - type: specification
    name: "RFC 5681 - TCP Congestion Control"
    url: https://tools.ietf.org/html/rfc5681
  - type: specification
    name: "RFC 6298 - Computing TCP Retransmission Timer"
    url: https://tools.ietf.org/html/rfc6298
  - type: tutorial
    name: "Let's code a TCP/IP stack"
    url: https://www.saminiir.com/lets-code-tcp-ip-stack-1-ethernet-arp/
prerequisites:
  - type: skill
    name: Networking fundamentals (OSI model, basic socket programming)
  - type: skill
    name: C or Rust programming (pointer arithmetic, memory management)
  - type: skill
    name: Packet analysis (Wireshark/tcpdump usage)
  - type: skill
    name: Finite state machines
milestones:
  - id: build-tcp-stack-m1
    name: "Ethernet & ARP"
    description: >-
      Set up raw packet I/O via TAP device or raw socket, parse Ethernet frames,
      and implement ARP request/reply for IP-to-MAC address resolution.
    estimated_hours: "10-15"
    concepts:
      - Layer 2 networking
      - Address resolution protocol
      - Ethernet frame structure
      - Network byte order (big-endian)
    skills:
      - Raw packet manipulation
      - Binary protocol implementation
      - Network hardware interfacing
      - Low-level memory management
    acceptance_criteria:
      - Raw socket or TAP device successfully receives and sends Ethernet frames on the network interface
      - Ethernet parser correctly extracts 6-byte source MAC, 6-byte destination MAC, and 2-byte EtherType field
      - Parser correctly handles IEEE 802.1Q VLAN-tagged frames by detecting EtherType 0x8100 and skipping the 4-byte tag
      - ARP reply with correct hardware and protocol addresses is sent when an ARP request matches our configured IP
      - ARP table caches resolved entries and evicts stale entries after a configurable timeout (default 60 seconds)
      - ARP request is sent for unknown destination IPs, and outbound packets are queued until resolution completes or times out
      - All multi-byte fields are converted between network byte order (big-endian) and host byte order using explicit conversion functions
    pitfalls:
      - Byte order: all Ethernet/ARP fields are big-endian on the wire; forgetting ntohs/ntohl causes silent corruption
      - ARP cache poisoning is trivial — do not trust unsolicited ARP replies in production, but accept them for this project scope
      - Broadcast frames (destination FF: FF:FF:FF:FF:FF) must be handled; dropping them breaks ARP
      - TAP device MTU defaults may differ from physical interface MTU, causing silent truncation
      - Forgetting to set the correct EtherType (0x0806 for ARP, 0x0800 for IPv4) causes frames to be dropped by receivers
    deliverables:
      - Raw socket or TAP device setup for capturing and injecting network frames
      - Ethernet frame parser extracting source MAC, destination MAC, EtherType, and payload
      - ARP request/reply handler resolving IP addresses to MAC addresses per RFC 826
      - ARP cache with configurable TTL expiration and pending-request queue for outbound packets

  - id: build-tcp-stack-m2
    name: "IP Layer & ICMP"
    description: >-
      Implement IPv4 packet parsing, header checksum calculation, IP fragmentation
      reassembly, basic routing table lookup, and ICMP echo request/reply (ping).
    estimated_hours: "14-20"
    concepts:
      - Layer 3 networking
      - IP header checksum (ones' complement)
      - IP fragmentation and reassembly
      - ICMP error and informational messages
      - Routing table and longest prefix match
    skills:
      - IP packet routing and forwarding
      - Checksum computation and verification
      - Packet fragmentation and reassembly
      - Network layer error handling
    acceptance_criteria:
      - IPv4 parser correctly extracts version, IHL, total length, identification, flags, fragment offset, TTL, protocol, source IP, and destination IP
      - IP header checksum is computed using RFC 1071 ones' complement sum and verified on every received packet; packets with invalid checksums are dropped silently
      - ICMP echo reply is sent with correct identifier and sequence number in response to incoming echo request (ping responds within 10ms on local network)
      - IP fragmentation reassembly correctly reconstructs original datagram from fragments using identification field and fragment offset; incomplete fragments are discarded after a 30-second timeout
      - Routing table supports static route entries and selects the correct next-hop gateway using longest prefix match
      - TTL is decremented on forwarded packets; packets with TTL=0 are dropped and ICMP Time Exceeded (Type 11) is sent back to source
      - Packets destined for our own IP addresses are delivered to the upper-layer protocol handler (TCP/ICMP)
    pitfalls:
      - IP header checksum covers only the header, not the payload — a common mistake is checksumming the entire packet
      - IHL (Internet Header Length) is in 4-byte words, not bytes; forgetting to multiply by 4 causes incorrect payload offset
      - Fragment reassembly must handle overlapping fragments and out-of-order arrival; a simple array indexed by offset works
      - TTL must be decremented before forwarding, not after; decrementing after forwarding violates RFC 791
      - ICMP checksum is separate from IP checksum and covers the entire ICMP message including data
    deliverables:
      - IPv4 packet parser and builder with full header field extraction and construction
      - IP header checksum calculator using RFC 1071 ones' complement algorithm
      - IP fragmentation reassembly module with timeout-based cleanup of incomplete datagrams
      - ICMP echo request/reply handler implementing ping functionality
      - Static routing table with longest prefix match lookup and next-hop selection

  - id: build-tcp-stack-m3
    name: "TCP Connection Management"
    description: >-
      Implement TCP segment parsing with checksum verification using the IP
      pseudo-header, the full TCP state machine including 3-way handshake, MSS
      option negotiation, and graceful connection teardown with TIME_WAIT.
    estimated_hours: "20-28"
    concepts:
      - TCP state machine (RFC 793 Figure 6)
      - TCP pseudo-header checksum
      - 3-way handshake (SYN, SYN-ACK, ACK)
      - 4-way connection teardown (FIN handshake)
      - TCP options (MSS, Window Scale)
      - TIME_WAIT state and 2*MSL timer
    skills:
      - Finite state machine implementation
      - TCP checksum with pseudo-header
      - Connection lifecycle management
      - Sequence number arithmetic (modulo 2^32)
      - Concurrent connection handling
    acceptance_criteria:
      - TCP segment parser extracts source port, destination port, sequence number, acknowledgment number, data offset, flags (SYN/ACK/FIN/RST/PSH), window size, checksum, and urgent pointer
      - TCP checksum is computed over the pseudo-header (source IP, destination IP, zero byte, protocol 6, TCP length) plus the TCP segment; segments with invalid checksums are dropped
      - 3-way handshake completes successfully — server transitions LISTEN→SYN_RECEIVED→ESTABLISHED; client transitions CLOSED→SYN_SENT→ESTABLISHED
      - MSS option is parsed from SYN/SYN-ACK segments and used to limit outbound segment payload size; default MSS of 536 bytes is used when option is absent
      - State machine implements all 11 TCP states from RFC 793 with correct transitions, including simultaneous open and simultaneous close
      - RST segments are handled correctly — received RST aborts the connection; RST is sent in response to segments received in invalid states
      - 4-way FIN handshake gracefully closes connection; the closing side enters TIME_WAIT for 2*MSL (default 60 seconds) to absorb delayed segments
      - Connection table supports multiple concurrent connections keyed by (local IP, local port, remote IP, remote port) tuple
    pitfalls:
      - TCP checksum pseudo-header is the most error-prone part — it requires fields from the IP layer, breaking clean layer separation
      - Sequence number arithmetic must use modulo 2^32 comparison (e.g., SEQ_LT, SEQ_GT macros) because sequence numbers wrap around
      - Forgetting TIME_WAIT causes connection reuse to accept stale segments from previous connections
      - Simultaneous open (both sides send SYN) is rare but must be handled or the state machine will deadlock
      - RST handling is asymmetric — receiving RST in SYN_RECEIVED returns to LISTEN, but in ESTABLISHED it aborts
      - ISN (Initial Sequence Number) should not be zero or predictable; use a time-based or random generator per RFC 6528
    deliverables:
      - TCP segment parser and builder with full header field extraction including options
      - TCP checksum calculator using IP pseudo-header per RFC 793
      - MSS option negotiation during SYN/SYN-ACK exchange
      - Full TCP state machine implementing all 11 states with transition validation
      - Connection teardown with FIN handshake and TIME_WAIT timer
      - Connection table managing concurrent sessions by 4-tuple

  - id: build-tcp-stack-m4
    name: "TCP Reliable Delivery & Flow Control"
    description: >-
      Implement sliding window protocol for flow control, out-of-order segment
      buffering, retransmission with adaptive RTO, and basic congestion control
      (slow start, congestion avoidance, fast retransmit/fast recovery).
    estimated_hours: "25-35"
    concepts:
      - Sliding window flow control
      - Receiver advertised window
      - Retransmission timeout (RTO) estimation (RFC 6298)
      - Karn's algorithm (don't sample retransmitted segments)
      - Congestion window (cwnd) and slow start threshold (ssthresh)
      - Slow start and congestion avoidance (RFC 5681)
      - Fast retransmit on 3 duplicate ACKs
      - Fast recovery (RFC 5681 Section 3.2)
    skills:
      - Sliding window protocol implementation
      - Adaptive RTO calculation (SRTT, RTTVAR)
      - Out-of-order segment buffering and reassembly
      - Congestion control algorithms
      - Timer management
    acceptance_criteria:
      - Sender sliding window allows sending up to min(cwnd, rwnd) bytes of unacknowledged data before blocking
      - Receiver advertised window (rwnd) is respected; sender never has more unacknowledged bytes than the receiver's advertised window
      - Out-of-order segments are buffered at the receiver and delivered to the application in-order once gaps are filled
      - RTO is calculated using Jacobson's algorithm (SRTT, RTTVAR) per RFC 6298; initial RTO is 1 second, minimum is 1 second
      - Karn's algorithm is applied — RTT samples are not taken from retransmitted segments to avoid ambiguity
      - Unacknowledged segments are retransmitted when RTO expires; RTO is doubled (exponential backoff) on each successive timeout for the same segment
      - Slow start increases cwnd by 1 MSS per ACK received (exponential growth) until cwnd reaches ssthresh
      - Congestion avoidance increases cwnd by MSS*(MSS/cwnd) per ACK (approximately 1 MSS per RTT) when cwnd >= ssthresh
      - Fast retransmit triggers on receipt of 3 duplicate ACKs, retransmitting the missing segment without waiting for RTO
      - Fast recovery sets ssthresh = cwnd/2 and cwnd = ssthresh + 3*MSS after fast retransmit, then inflates cwnd by 1 MSS per additional duplicate ACK
      - End-to-end test transfers a 1MB file over the stack and verifies byte-for-byte correctness with SHA-256 comparison
    pitfalls:
      - Timer management is the #1 source of bugs — use a single timer wheel or min-heap, not per-segment timers
      - Window calculation must use modulo 2^32 arithmetic; naive subtraction overflows on sequence wraparound
      - Nagle's algorithm (RFC 896) can interact badly with delayed ACKs causing 200ms delays; consider implementing or explicitly disabling
      - Zero-window probing is needed when receiver advertises rwnd=0; without it, the connection deadlocks
      - Silly window syndrome occurs when receiver advertises tiny windows; Clark's algorithm or sender-side Nagle mitigates this
      - Fast retransmit/recovery must not be triggered during slow start or when there's insufficient duplicate ACK data
    deliverables:
      - Sender sliding window tracking bytes in flight against min(cwnd, rwnd)
      - Receiver-side out-of-order segment buffer with gap tracking and in-order delivery
      - Adaptive RTO calculator implementing Jacobson's algorithm with Karn's amendment
      - Retransmission timer with exponential backoff on repeated timeouts
      - Congestion control module implementing slow start, congestion avoidance, fast retransmit, and fast recovery per RFC 5681
      - Zero-window probe mechanism preventing deadlock when receiver window closes
      - End-to-end file transfer test verifying reliable, ordered delivery
```
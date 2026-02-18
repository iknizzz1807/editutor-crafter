# AUDIT & FIX: build-vpn

## CRITIQUE
- **Audit Finding #1 is VALID**: The project lacks a structured handshake/session establishment phase. M4 (Key Exchange) happens AFTER M3 (Encryption), but encryption requires a shared key that only exists after key exchange. The milestone order is wrong: key exchange MUST happen before encryption. The correct order should be: TUN/TAP → UDP Transport → Key Exchange (handshake) → Encryption → Routing/NAT.
- **Audit Finding #2 is VALID**: MTU/MSS clamping is critical for VPN tunnels. Encapsulation adds overhead (UDP header 8 bytes, encryption overhead ~28 bytes for AES-GCM, outer IP header 20 bytes), so the TUN interface MTU must be reduced below the path MTU to prevent fragmentation. Without this, packets are silently dropped or fragmented, causing mysterious TCP stalls.
- **M1 mentions 'TUN device persists while file descriptor is open' but doesn't mention IFF_TUN flag**: The ioctl TUNSETIFF call requires specifying IFF_TUN (layer 3) vs IFF_TAP (layer 2). This is a fundamental configuration choice.
- **M2 'NAT traversal basics' as a deliverable is hand-wavy**: Real NAT traversal (STUN, hole punching) is complex. The project should either properly scope this (e.g., 'works when at least one endpoint has a public IP') or add a dedicated milestone.
- **M3 mentions AES-256-GCM but M4 derives keys AFTER**: The milestone ordering is backwards. You can't encrypt without keys. M4 must come before M3.
- **M3 anti-replay window is mentioned in AC but not explained**: A sliding window algorithm (RFC 6479) tracking seen sequence numbers is needed. This is non-trivial to implement correctly.
- **M4 'peer authentication' is vague**: The deliverable says 'verifying remote endpoint identity' but doesn't specify HOW. Without a PKI or pre-shared keys, DH alone is vulnerable to MITM. Pre-shared keys or certificate pinning must be explicitly addressed.
- **M5 'DNS leaks' pitfall is mentioned but not addressed in AC**: DNS requests bypassing the VPN tunnel are a real privacy concern. The AC should include DNS configuration.
- **M5 is Linux-specific**: The entire routing/NAT milestone uses Linux-specific tools (iptables, ip route). This should be acknowledged.
- **Missing keepalive mechanism**: VPN connections through NAT expire if no packets are sent. A keepalive mechanism (periodic empty packets) is needed.
- **Missing graceful error handling**: What happens when the remote endpoint is unreachable? The VPN should detect this and attempt reconnection.

## FIXED YAML
```yaml
id: build-vpn
name: Build Your Own VPN
description: Point-to-point VPN tunnel with TUN interface, encryption, and key exchange
difficulty: expert
estimated_hours: "55-85"
essence: >
  TUN virtual interface manipulation via ioctl system calls, UDP packet encapsulation
  with custom framing protocol, ephemeral Diffie-Hellman key exchange for session
  establishment, AES-GCM authenticated encryption with nonce management and anti-replay
  protection, and routing table configuration for traffic tunneling.
why_important: >
  Building a VPN from scratch teaches the full networking stack from layer 3 interfaces
  through cryptographic protocols to application-layer tunneling, skills directly
  applicable to infrastructure security, network engineering, and understanding
  production VPN implementations like WireGuard and OpenVPN.
learning_outcomes:
  - Implement TUN virtual network devices using ioctl system calls for IP packet capture and injection
  - Design UDP-based transport protocol with packet framing and multiplexed I/O
  - Build Diffie-Hellman key exchange with peer authentication for session establishment
  - Implement AES-GCM authenticated encryption with nonce management and anti-replay protection
  - Configure Linux routing tables and iptables rules for traffic forwarding through the tunnel
  - Handle MTU/MSS clamping to prevent fragmentation in encapsulated tunnels
  - Debug encrypted network traffic using packet analysis tools
skills:
  - Virtual Network Interfaces
  - UDP Socket Programming
  - Symmetric Cryptography
  - Key Exchange Protocols
  - Network Routing
  - Packet Encapsulation
  - Anti-Replay Protection
  - MTU Management
tags:
  - build-from-scratch
  - c
  - encryption
  - expert
  - go
  - networking
  - routing
  - rust
  - tunneling
architecture_doc: architecture-docs/build-vpn/index.md
languages:
  recommended:
    - Go
    - Rust
    - C
  also_possible:
    - Python
resources:
  - name: TUN/TAP Interface Tutorial""
    url: "https://www.kernel.org/doc/Documentation/networking/tuntap.txt"
    type: documentation
  - name: WireGuard Whitepaper""
    url: "https://www.wireguard.com/papers/wireguard.pdf"
    type: paper
  - name: Noise Protocol Framework""
    url: "http://noiseprotocol.org/"
    type: specification
prerequisites:
  - type: skill
    name: Network programming (UDP sockets)
  - type: skill
    name: Cryptography basics (symmetric encryption, hashing)
  - type: skill
    name: Linux networking (ip command, basic iptables)
  - type: skill
    name: C or systems language with raw syscall access
notes: >
  This project is Linux-specific (TUN device, iptables, ip route). macOS has utun
  devices with a different API. Windows requires a TAP-Windows driver. The routing
  and NAT milestone uses Linux-specific tools. At least one endpoint must have a
  public IP address (full NAT traversal with hole punching is out of scope).
milestones:
  - id: build-vpn-m1
    name: TUN Interface
    description: >
      Create and configure a TUN (Layer 3) virtual network device for
      capturing and injecting IP packets.
    acceptance_criteria:
      - TUN device is created by opening /dev/net/tun and calling ioctl TUNSETIFF with IFF_TUN | IFF_NO_PI flags
      - TUN interface appears in 'ip link show' output and can be assigned an IP address and brought up
      - Raw IP packets (no protocol information header due to IFF_NO_PI) are read from the TUN file descriptor when traffic is routed to the TUN interface
      - Writing a valid IP packet to the TUN file descriptor delivers it to the local network stack (verified by ping to the TUN IP address from the same machine receiving ICMP echo reply)
      - Interface MTU is configured (default 1500, will be reduced in later milestones for encapsulation overhead)
      - The TUN device is cleaned up (file descriptor closed) on program exit, and the interface disappears
      - Program runs with appropriate privileges (root or CAP_NET_ADMIN capability)
    pitfalls:
      - Forgetting IFF_NO_PI adds a 4-byte protocol information header to every packet read/written, corrupting IP packet parsing
      - Must run as root or with CAP_NET_ADMIN; otherwise ioctl TUNSETIFF fails with EPERM
      - The TUN interface disappears when the file descriptor is closed; the program must keep it open for the lifetime of the VPN
      - TUN (Layer 3, IP packets) vs TAP (Layer 2, Ethernet frames) is a critical distinction; this project uses TUN
      - Assigning an IP address and bringing the interface up requires additional ioctl calls or shelling out to 'ip addr add' and 'ip link set up'
    concepts:
      - Virtual network interfaces (TUN vs TAP)
      - ioctl system calls for device configuration
      - IP packet structure (IPv4 header)
      - Linux network interface management
    skills:
      - Low-level network programming
      - System call interfaces (ioctl, open, read, write)
      - File descriptor management
      - Linux networking fundamentals
    deliverables:
      - TUN device creation with IFF_TUN | IFF_NO_PI via ioctl
      - IP address and MTU configuration on the TUN interface
      - Packet reading from TUN (capturing outbound IP packets)
      - Packet writing to TUN (injecting inbound IP packets)
      - Verification via ping through TUN interface
    estimated_hours: "6-10"

  - id: build-vpn-m2
    name: UDP Transport Layer
    description: >
      Create a UDP socket for tunneling packets between VPN endpoints,
      with multiplexed I/O between TUN and UDP.
    acceptance_criteria:
      - UDP socket is created and bound to a configurable port for receiving packets from the remote endpoint
      - Packets read from TUN are wrapped in a simple framing header (packet type, length) and sent via UDP to the configured remote endpoint address
      - Packets received from UDP are extracted from the framing header and written to the local TUN interface
      - select/poll/epoll multiplexes the TUN file descriptor and UDP socket for concurrent I/O without blocking on either
      - An unencrypted (plaintext) IP-over-UDP tunnel functions correctly: ping from one endpoint's TUN subnet reaches the other endpoint's TUN subnet
      - Server mode accepts packets from any source address and tracks the most recent source as the peer address (enabling NAT traversal for the case where the client is behind NAT)
      - Keepalive packets are sent periodically (e.g., every 25 seconds) to maintain NAT mappings
    pitfalls:
      - Encapsulation adds overhead: IP header (20 bytes) + UDP header (8 bytes) + framing header + encryption overhead (added later). TUN MTU must be reduced to prevent outer packet fragmentation
      - NAT traversal only works when the server has a public IP; both endpoints behind NAT requires hole punching which is out of scope
      - UDP packets can be reordered, duplicated, or lost; the framing protocol should include sequence numbers for later anti-replay use
      - Blocking on TUN read while a UDP packet is waiting (or vice versa) causes latency spikes; multiplexing is essential
      - Maximum UDP payload should be conservative (~1400 bytes) to avoid IP fragmentation across internet paths
    concepts:
      - UDP tunneling and encapsulation
      - I/O multiplexing (select/poll/epoll)
      - Packet framing protocol design
      - NAT keepalive
    skills:
      - UDP socket programming
      - Asynchronous I/O and event loops
      - Network protocol design
      - Packet framing and serialization
    deliverables:
      - UDP socket creation and binding on configurable port
      - Packet framing header with type and sequence number
      - TUN → UDP encapsulation and UDP → TUN extraction
      - I/O multiplexing between TUN fd and UDP socket
      - Periodic keepalive packet transmission
      - Plaintext tunnel verification via cross-endpoint ping
    estimated_hours: "6-10"

  - id: build-vpn-m3
    name: Key Exchange and Session Establishment
    description: >
      Implement a handshake protocol using Diffie-Hellman key exchange
      with peer authentication to establish session encryption keys.
    acceptance_criteria:
      - Ephemeral key pairs (ECDH using Curve25519 or similar) are generated fresh for each new VPN session
      - Handshake protocol exchanges public keys over UDP and both sides derive an identical shared secret
      - Peer authentication verifies the remote endpoint's identity using pre-shared keys (PSK) or pre-exchanged public key fingerprints to prevent man-in-the-middle attacks
      - HKDF (or equivalent KDF) derives separate encryption keys for client→server and server→client traffic directions from the shared secret
      - Session keys provide perfect forward secrecy: compromising long-term keys does not compromise past session traffic
      - Handshake completes within a reasonable timeout (e.g., 5 seconds) and the tunnel transitions to an 'established' state before data packets flow
      - Handshake failure (authentication failure, timeout) is detected and reported clearly without crashing
      - Key material is securely erased from memory after session keys are derived
    pitfalls:
      - Diffie-Hellman without authentication is vulnerable to MITM; pre-shared keys or public key fingerprints are the minimum viable authentication
      - Weak random number generation for ephemeral keys compromises the entire session; use a CSPRNG (crypto/rand, getrandom(), etc.)
      - Reusing static keys across sessions breaks forward secrecy; always generate ephemeral keys per session
      - Key material left in memory after derivation can be recovered via memory dumps; explicitly zero key buffers after use
      - The handshake must handle packet loss (retransmit handshake messages on timeout) since it runs over unreliable UDP
    concepts:
      - Elliptic-curve Diffie-Hellman key exchange
      - Perfect forward secrecy
      - Key derivation functions (HKDF)
      - Peer authentication with pre-shared keys
    skills:
      - Public key cryptography implementation
      - Secure protocol handshake design
      - Key derivation and management
      - Cryptographic library usage
    deliverables:
      - ECDH key exchange generating ephemeral key pairs and deriving shared secret
      - Peer authentication using pre-shared key or public key fingerprint verification
      - HKDF-based session key derivation producing separate keys per traffic direction
      - Handshake state machine with timeout and retransmission
      - Secure key material erasure after session establishment
    estimated_hours: "8-12"

  - id: build-vpn-m4
    name: Encryption and Anti-Replay
    description: >
      Encrypt tunnel traffic with AES-GCM using the session keys,
      with nonce management and anti-replay protection.
    acceptance_criteria:
      - All data packets are encrypted with AES-256-GCM (or ChaCha20-Poly1305) using the session key derived in M3 before being sent over UDP
      - Decryption verifies the authentication tag; tampered or corrupted packets are silently dropped (not processed)
      - Each packet uses a unique nonce constructed from a monotonically increasing 64-bit counter; nonce reuse with the same key never occurs
      - Anti-replay sliding window (e.g., 64-128 packet window per RFC 6479) rejects packets with duplicate or too-old sequence numbers
      - Encrypted tunnel passes data correctly: ping through VPN tunnel succeeds, and Wireshark shows only encrypted UDP payloads (no plaintext IP packets)
      - TUN interface MTU is reduced to account for encryption overhead (AES-GCM: 12-byte nonce + 16-byte auth tag + 8-byte counter = ~36 bytes + UDP/IP headers), preventing outer packet fragmentation
      - Key rotation: session keys are renegotiated after a configurable data volume (e.g., every 64MB) or time interval (e.g., every 2 hours) by initiating a new handshake
    pitfalls:
      - Nonce reuse with AES-GCM is CATASTROPHIC: it completely breaks confidentiality and authenticity. Use a simple counter that never wraps (renegotiate keys before 2^64 packets)
      - Forgetting to verify the authentication tag allows attackers to inject modified packets; always check the tag before processing decrypted data
      - Anti-replay window that is too small causes legitimate reordered packets to be dropped; too large wastes memory. 64-128 packets is a good default
      - MTU calculation must account for ALL overhead layers: outer IP (20) + UDP (8) + framing (8+) + nonce (12) + auth tag (16) = ~64 bytes minimum; set TUN MTU to path MTU - 64 or lower
      - If the counter reaches a dangerous level (e.g., 2^63), the session MUST be rekeyed before nonce space is exhausted
    concepts:
      - Authenticated encryption (AES-GCM / ChaCha20-Poly1305)
      - Nonce construction and management
      - Anti-replay sliding window
      - MTU/MSS clamping for encapsulated tunnels
    skills:
      - Cryptographic library usage
      - Secure nonce management
      - Sliding window algorithm implementation
      - MTU calculation and configuration
    deliverables:
      - AES-256-GCM encryption of data packets before UDP transmission
      - Authentication tag verification on decryption with silent drop on failure
      - Counter-based nonce management ensuring uniqueness
      - Anti-replay sliding window rejecting duplicate/old sequence numbers
      - TUN MTU reduction accounting for encryption and encapsulation overhead
      - Key rotation triggering renegotiation after volume/time threshold
    estimated_hours: "8-12"

  - id: build-vpn-m5
    name: Routing, NAT, and DNS Configuration
    description: >
      Configure routing tables, NAT masquerading, and DNS settings for
      full VPN tunnel functionality on Linux.
    acceptance_criteria:
      - All client traffic is routed through the VPN tunnel by replacing the default gateway with the TUN interface
      - A host route to the VPN server's public IP via the original gateway is preserved to prevent a routing loop (VPN traffic must not be routed through itself)
      - Server-side NAT masquerade (iptables MASQUERADE rule) allows VPN clients to access the internet through the server's external interface
      - Split tunneling mode routes only specified destination subnets through the VPN, leaving other traffic on the direct path
      - DNS is configured to use a tunnel-reachable DNS server (either the VPN server running a resolver, or a public resolver routed through the tunnel) to prevent DNS leaks
      - IPv6 traffic is either tunneled or explicitly blocked to prevent IPv6 leaks when only IPv4 tunneling is implemented
      - Original routing table and DNS configuration are restored cleanly when the VPN disconnects (normal exit or crash via signal handler)
      - TCP MSS clamping (iptables -j TCPMSS --clamp-mss-to-pmtu) is configured to prevent TCP segments from exceeding the tunnel MTU
    pitfalls:
      - Replacing the default route without preserving the route to the VPN server IP causes a routing loop that kills the VPN connection immediately
      - SSH sessions to the VPN server may break if the route change causes existing connections to be rerouted through the tunnel; add a specific route for management traffic
      - DNS leaks expose browsing activity even when traffic is tunneled; resolv.conf or systemd-resolved must be configured to use a tunnel-safe resolver
      - IPv6 leaks: if the system has IPv6 connectivity and only IPv4 is tunneled, IPv6 traffic bypasses the VPN entirely; block IPv6 or tunnel it
      - Forgetting to restore routes on exit leaves the system with broken networking requiring manual intervention
      - TCP MSS clamping is essential; without it, TCP connections through the tunnel stall because large segments are fragmented and reassembly fails
      - iptables rules persist across VPN restarts and can accumulate if not cleaned up; use unique chain names or clean up on startup
    concepts:
      - IP routing tables and default gateway
      - NAT masquerading with iptables
      - DNS leak prevention
      - TCP MSS clamping
      - Split tunneling
    skills:
      - Linux routing table manipulation
      - iptables firewall configuration
      - DNS configuration management
      - Network debugging and connectivity testing
    deliverables:
      - Default route configuration through TUN interface with VPN server host route preserved
      - NAT masquerade on server forwarding VPN client traffic to internet
      - DNS configuration directing queries through the tunnel
      - IPv6 leak prevention (block or tunnel)
      - TCP MSS clamping via iptables
      - Split tunneling configuration for specified subnets
      - Cleanup handler restoring original routes and DNS on exit
    estimated_hours: "10-15"
```
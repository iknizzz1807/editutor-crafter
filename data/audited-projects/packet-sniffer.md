# AUDIT & FIX: packet-sniffer

## CRITIQUE
- **BPF scope confusion (M5)**: The AC says 'Support BPF filter expressions' which implies either (a) passing filter strings to libpcap's pcap_compile/pcap_setfilter, or (b) implementing a BPF bytecode compiler/evaluator. These are vastly different in complexity. If using libpcap, this is trivial. If using raw sockets, you'd need to compile BPF and attach via setsockopt SO_ATTACH_FILTER, which is an entire sub-project. The AC must be explicit.
- **M2 FCS/CRC**: The deliverable mentions 'Frame validation including minimum size and CRC checks' but captured packets typically do NOT include the FCS (it's stripped by the NIC before delivery to the OS). This AC is misleading.
- **M3 IPv6**: The deliverables mention 'IPv6 header parsing with next header chain support' but no AC requires it, and the milestone title says 'IP Header Parsing' (implying IPv4). Either scope it properly or remove.
- **VLAN tags (802.1Q)**: Mentioned only as a pitfall but not in any AC. 802.1Q tags shift all subsequent header offsets by 4 bytes, which is a critical correctness issue.
- **Ring buffer (M1 deliverable)**: 'Ring buffer for captured packet storage' is a deliverable but no AC requires it. If using libpcap, the library handles buffering internally.
- **Missing**: No AC for packet timestamps (critical for any sniffer).
- **Missing**: No AC for PCAP file reading/writing (only mentioned as a deliverable in M5).
- **Platform specificity**: Raw sockets work differently on Linux (AF_PACKET), macOS (BPF device), and Windows (WinPcap/Npcap). No AC addresses this.

## FIXED YAML
```yaml
id: packet-sniffer
name: Packet Sniffer
description: Network packet capture and protocol header parsing
difficulty: intermediate
estimated_hours: "20-30"
essence: >
  Packet capture from network interfaces using libpcap (or raw sockets),
  binary protocol header parsing across network layers (Ethernet, IPv4,
  TCP/UDP), and BPF-based filtering for selective traffic inspection.
why_important: >
  Building this teaches network protocol internals and binary data
  manipulation essential for network programming, security analysis,
  and debugging distributed systems.
learning_outcomes:
  - Implement packet capture using libpcap or platform-specific raw socket APIs
  - Parse binary Ethernet, IPv4, TCP, and UDP headers using struct unpacking and byte manipulation
  - Apply Berkeley Packet Filter expressions via libpcap for selective packet filtering
  - Handle network byte order (big-endian) conversion to host byte order
  - Build protocol dissectors that parse nested protocol headers based on type fields
  - Implement formatted output displaying packet metadata with timestamps
  - Write and read PCAP files for offline analysis
skills:
  - Raw Socket / libpcap Programming
  - Binary Protocol Parsing
  - Network Packet Analysis
  - BPF Filter Syntax
  - Byte Order Conversion
  - Network Layer Programming
tags:
  - analysis
  - c
  - go
  - intermediate
  - networking
  - packet-capture
  - python
  - raw-sockets
architecture_doc: architecture-docs/packet-sniffer/index.md
languages:
  recommended:
    - C
    - Python
    - Go
  also_possible:
    - Rust
resources:
  - name: Libpcap Programming Tutorial
    url: "https://www.tcpdump.org/pcap.html"
    type: tutorial
  - name: Building a Packet Sniffer from Scratch
    url: "https://aidanvidal.github.io/posts/Packet_Sniffer.html"
    type: tutorial
  - name: Scapy Documentation
    url: "https://scapy.readthedocs.io/"
    type: reference
prerequisites:
  - type: skill
    name: Networking basics (TCP/IP model)
  - type: skill
    name: Binary data parsing
  - type: skill
    name: C structs or Python struct module
milestones:
  - id: packet-sniffer-m1
    name: Packet Capture Setup
    description: Set up packet capture using libpcap (recommended) or platform-specific raw sockets.
    estimated_hours: "4-5"
    concepts:
      - libpcap API
      - Network interfaces
      - Privilege requirements
      - Promiscuous mode
    skills:
      - Network programming
      - System-level I/O operations
      - Working with privileged operations
      - Platform-specific API usage
    acceptance_criteria:
      - List all available network interfaces with their names and descriptions using pcap_findalldevs or equivalent
      - Open a selected interface for live capture using pcap_open_live with configurable snapshot length
      - Enable promiscuous mode to capture all traffic on the interface (not just traffic destined for this host)
      - Capture raw packets in a loop using pcap_loop or pcap_next_ex; print packet length and timestamp for each
      - Handle permission errors gracefully (root/admin required); print clear error message if insufficient privileges
      - Support a "-c COUNT" option to capture exactly COUNT packets then exit
      - Record high-resolution timestamps (microsecond or nanosecond) for each captured packet
    pitfalls:
      - Requires root/admin privileges; handle EPERM gracefully
      - Interface names vary by OS (eth0 on Linux, en0 on macOS); use pcap_findalldevs for portability
      - Virtual/loopback interfaces may not support promiscuous mode or may use different link types
      - Snapshot length (snaplen) too small truncates packets; default to 65535
    deliverables:
      - Interface enumeration and selection
      - Live capture loop with packet count and timestamp display
      - Promiscuous mode configuration
      - Permission error handling

  - id: packet-sniffer-m2
    name: Ethernet and VLAN Parsing
    description: Parse Ethernet frames including 802.1Q VLAN tag handling.
    estimated_hours: "3-4"
    concepts:
      - Ethernet framing
      - MAC addresses
      - EtherType field
      - 802.1Q VLAN tagging
    skills:
      - Binary data parsing with struct unpacking
      - Network byte order (big-endian) conversion
      - Variable-offset header parsing
    acceptance_criteria:
      - Extract destination MAC (bytes 0-5), source MAC (bytes 6-11), and EtherType (bytes 12-13) from Ethernet header
      - Format MAC addresses as colon-separated hex (e.g., aa:bb:cc:dd:ee:ff)
      - Convert EtherType from network byte order (big-endian) to host byte order
      - Detect 802.1Q VLAN tags (EtherType 0x8100); extract VLAN ID and the real EtherType from the next 4 bytes
      - Correctly compute the offset to the network layer payload (14 bytes normally, 18 bytes with VLAN tag)
      - Display parsed Ethernet header information for each captured packet
      - Verify against Wireshark for at least 5 captured packets
    pitfalls:
      - 802.1Q VLAN tags add 4 bytes, shifting all subsequent header offsets; failing to detect them causes mis-parsing
      - NIC hardware typically strips the Ethernet FCS (CRC) before delivering to the OS; do NOT expect FCS in captured data
      - Some interfaces use different link layer types (e.g., Linux cooked capture SLL); check pcap_datalink()
      - Jumbo frames can exceed 1500 bytes payload but this doesn't affect header parsing
    deliverables:
      - Ethernet header parser extracting MACs and EtherType
      - VLAN tag detection and parsing with correct offset adjustment
      - Link layer type checking via pcap_datalink()
      - Formatted Ethernet header display

  - id: packet-sniffer-m3
    name: IPv4 Header Parsing
    description: Parse IPv4 headers to extract addresses, protocol, and handle variable header length.
    estimated_hours: "4-5"
    concepts:
      - IPv4 header structure
      - IHL (Internet Header Length)
      - Protocol numbers
      - IP fragmentation
    skills:
      - IPv4 protocol field parsing
      - Variable-length header handling
      - Network address formatting (dotted decimal)
    acceptance_criteria:
      - Parse IPv4 header only when EtherType is 0x0800; skip other protocols with a log message
      - Extract version (must be 4) and IHL from the first byte; compute header length as IHL * 4 bytes
      - Extract total length, identification, flags, fragment offset, TTL, protocol number, and header checksum
      - Extract source and destination IP addresses (4 bytes each) and format as dotted decimal notation
      - Handle variable header length (IHL > 5) by skipping IP options to find the transport layer payload
      - Verify header checksum by computing the ones-complement sum over the header; flag packets with bad checksums
      - Display parsed IPv4 header fields for each captured packet
    pitfalls:
      - IHL is in 4-byte words, not bytes; multiply by 4 to get actual header length
      - Fragmented packets (fragment offset > 0 or MF flag set) have no transport header in non-first fragments
      - IP options (IHL > 5) are rare but must be skipped correctly to find the transport header
      - Do not assume fixed 20-byte IP header
    deliverables:
      - IPv4 header parser with all standard fields
      - Variable-length header handling via IHL
      - Header checksum verification
      - Source and destination IP formatting
      - Protocol number extraction for transport layer dispatch

  - id: packet-sniffer-m4
    name: TCP/UDP Parsing
    description: Parse TCP and UDP transport headers for port and flag information.
    estimated_hours: "4-5"
    concepts:
      - TCP header structure and flags
      - UDP header structure
      - Port numbers and service identification
    skills:
      - Transport layer protocol parsing
      - TCP flag bit extraction
      - Service identification by port
    acceptance_criteria:
      - Dispatch to TCP parser when IP protocol is 6, UDP parser when protocol is 17; log and skip others
      - "TCP: Extract source port, destination port, sequence number, acknowledgment number, data offset, flags (SYN/ACK/FIN/RST/PSH/URG), window size"
      - "UDP: Extract source port, destination port, length, and checksum"
      - TCP data offset is in 4-byte words; compute actual header length to find payload start
      - Identify well-known services by port number (HTTP/80, HTTPS/443, DNS/53, SSH/22) in output display
      - Calculate and display payload size (total IP length minus IP header minus transport header)
      - Display formatted transport header information alongside Ethernet and IP headers
    pitfalls:
      - TCP data offset field is in 4-byte words (like IHL); multiply by 4
      - Port numbers are unsigned 16-bit big-endian; convert to host byte order
      - Do not attempt to parse TCP payload as transport header in subsequent IP fragments
      - TCP flags are a bitmask in a single byte; extract each flag individually
    deliverables:
      - TCP header parser with flags and sequence numbers
      - UDP header parser with length and checksum
      - Port extraction and well-known service identification
      - Payload size calculation
      - Combined multi-layer packet display

  - id: packet-sniffer-m5
    name: Filtering and PCAP Output
    description: Add BPF capture filters (via libpcap) and PCAP file I/O.
    estimated_hours: "5-6"
    concepts:
      - Berkeley Packet Filter expressions
      - PCAP file format
      - Packet display formatting
    skills:
      - BPF filter syntax and application via libpcap API
      - PCAP file reading and writing
      - High-volume output formatting
    acceptance_criteria:
      - Accept a BPF filter expression string (e.g., "tcp port 80") and apply it using pcap_compile + pcap_setfilter
      - Handle BPF compilation errors with clear error messages including the invalid expression
      - "-w FILE" flag writes captured packets to a PCAP file using pcap_dump for offline analysis
      - "-r FILE" flag reads packets from a PCAP file instead of live capture and processes them through the same parser
      - Display a one-line summary per packet: timestamp, src_mac, dst_mac, src_ip:port -> dst_ip:port, protocol, length, TCP flags
      - "-x" flag shows hex dump of packet payload after the summary line
      - Verify PCAP output files are readable by tcpdump and Wireshark
    pitfalls:
      - "BPF filter expressions are compiled by libpcap; do NOT implement a BPF compiler yourself"
      - High traffic rates can overwhelm display; consider buffered output or summary mode
      - Timestamps in PCAP files have specific format requirements (seconds + microseconds since epoch)
      - When reading from PCAP files, the link-layer type may differ from live capture; check and adapt
    deliverables:
      - BPF filter application via libpcap API
      - PCAP file writer for offline storage
      - PCAP file reader for offline analysis
      - Formatted one-line packet summary display
      - Hex dump output option for payload inspection

```
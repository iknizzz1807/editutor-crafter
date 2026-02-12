# Build Your Own BitTorrent: Design Document


## Overview

This document outlines the design for a complete BitTorrent client that can download and seed files in a peer-to-peer network. The key architectural challenge is coordinating concurrent downloads from multiple peers while managing piece verification, peer state machines, and tracker communication protocols.


> This guide is meant to help you understand the big picture before diving into each milestone. Refer back to it whenever you need context on how components connect.


## Context and Problem Statement

> **Milestone(s):** This section provides foundational context for all milestones (1-4) by establishing the problem space and design challenges.

### The Library Network Analogy

Imagine a vast network of libraries spread across the world, each containing different collections of books. When you want to read a specific book, you don't need to travel to the one library that has it. Instead, you can request it through an **inter-library loan system** where libraries cooperate to share their collections. The library network maintains a **catalog service** that knows which libraries have which books, and when you request a book, the catalog tells you exactly where to find it.

BitTorrent works remarkably similarly to this library network, but instead of books, we're sharing files, and instead of libraries, we have individual computers (called **peers**). Just as libraries maintain catalogs of their collections, BitTorrent uses **torrent files** that contain a catalog of what files are available and how to verify you received the correct content. The **tracker** serves as the central catalog service, maintaining a list of which peers currently have which files available for sharing.

The key insight is that no single library (or peer) needs to have the complete collection. When you want a popular book, multiple libraries might have copies, so you can choose the most convenient one or even get different chapters from different libraries simultaneously. Similarly, when downloading a file via BitTorrent, you can download different **pieces** of the file from multiple peers concurrently, dramatically speeding up the process.

Just as libraries have policies about lending (you must return books, there are limits on how many you can borrow), BitTorrent implements **incentive mechanisms** where peers who share files get priority access to download from others. This creates a self-sustaining ecosystem where sharing benefits everyone.

The analogy breaks down in one crucial way: unlike physical books that can only be in one place at a time, digital files can be perfectly copied. Once you've downloaded a complete file via BitTorrent, you automatically become another "library" in the network, able to serve that file to future requesters. This creates a **viral sharing effect** where popular files become more available over time, not less.

### File Sharing Approaches

The evolution of file sharing represents different architectural approaches to the fundamental problem of distributing content efficiently across a network. Each approach involves trade-offs between performance, reliability, cost, and control.

| Approach | Architecture | Discovery Method | Content Source | Bandwidth Usage | Failure Mode | Example |
|----------|-------------|------------------|----------------|-----------------|--------------|---------|
| **Centralized** | Single server hosts all content | Central directory/search | One server per file | Server pays all bandwidth costs | Single point of failure - server down means no access | Early web downloads, FTP sites |
| **Client-Server CDN** | Multiple geographic servers with replicated content | DNS-based geographic routing | Closest available server | CDN provider pays bandwidth costs | Graceful degradation - some regions may lose access | Netflix, Steam, major websites |
| **Pure P2P** | All peers are equal, no central authority | Distributed hash tables (DHT) | Any peer with content | Distributed across all peers | Network fragmentation - peers may not find each other | Early Gnutella, Freenet |
| **Hybrid P2P (BitTorrent)** | Peers share content, tracker coordinates | Central tracker for peer discovery | Multiple peers simultaneously | Distributed across downloading peers | Tracker failure stops new downloads, but existing swarms continue | BitTorrent, modern P2P protocols |

> **Key Insight**: BitTorrent's hybrid approach combines the reliability of centralized peer discovery with the scalability benefits of distributed content delivery. This architectural decision reflects a fundamental principle: **centralize coordination, distribute execution**.

**Decision: Hybrid P2P Architecture for BitTorrent**
- **Context**: File sharing systems must balance discoverability (finding content and peers) with scalability (handling load as popularity increases)
- **Options Considered**: 
  1. Pure centralized (single server hosts files)
  2. Pure P2P (fully decentralized discovery and content)
  3. Hybrid P2P (centralized tracker, distributed content)
- **Decision**: Hybrid P2P with tracker-based peer discovery and peer-to-peer content transfer
- **Rationale**: Centralized trackers solve the peer discovery problem efficiently while distributed content transfer scales bandwidth costs with demand and creates redundancy
- **Consequences**: Enables rapid peer discovery and coordinated downloading while distributing bandwidth costs, but creates tracker dependency for initial peer discovery

The bandwidth economics are particularly compelling. In a centralized model, if a file becomes popular, the server operator pays exponentially increasing bandwidth costs. With BitTorrent, increased popularity means more peers have the complete file, creating **more upload capacity** in the system. The bandwidth cost scales with the number of people willing to share, not just with demand.

### Core Technical Challenges

BitTorrent solves three fundamental distributed systems problems that make peer-to-peer file sharing practical at scale. Each challenge represents a different aspect of coordinating untrusted participants in a decentralized network.

**Challenge 1: Peer Discovery in a Dynamic Network**

The first challenge is the **bootstrapping problem**: how do you find other peers who have the content you want when there's no central directory of active participants? Unlike a web server with a fixed IP address, BitTorrent peers are typically home computers with dynamic IP addresses that come online and go offline unpredictably.

BitTorrent solves this through the **tracker protocol**. The tracker acts as a rendezvous point where peers announce their availability and discover others. When a peer wants to download a file, it extracts the tracker URL from the torrent file, announces itself to the tracker with its current IP address and port, and receives a list of other peers currently sharing that file.

The tracker doesn't store the actual file content - it only maintains **ephemeral peer lists** that reflect the current state of the swarm. Peers must periodically re-announce to stay in the active peer list, creating a self-cleaning mechanism where offline peers automatically disappear from the system.

| Tracker Interaction | Purpose | Information Exchanged | Frequency |
|---------------------|---------|---------------------|-----------|
| **Initial Announce** | Join swarm, get initial peer list | Peer reports: info_hash, peer_id, IP, port, bytes uploaded/downloaded | Once per torrent |
| **Periodic Announce** | Stay visible, report progress, get updated peers | Updated statistics, receive peers that joined since last announce | Every 15-30 minutes |
| **Event Announce** | Report state changes | 'started', 'completed', or 'stopped' events with current statistics | When events occur |

**Challenge 2: Content Verification in an Untrusted Network**

The second challenge is **content integrity**: how do you verify that the data you receive from untrusted peers hasn't been corrupted or maliciously modified? Traditional file downloads from trusted servers don't face this problem, but in a P2P network, any peer could send you garbage data.

BitTorrent's solution is **cryptographic piece verification**. The torrent file contains SHA-1 hashes of every piece of the target file. When a peer receives a piece from another peer, it immediately computes the SHA-1 hash of the received data and compares it to the expected hash from the torrent file. If the hashes don't match, the piece is discarded and requested again from a different peer.

This creates a **trustless verification system** where peers don't need to trust each other - they only need to trust the cryptographic hashes in the torrent file. The hash verification happens at the **piece level** (typically 256KB to 1MB chunks) rather than the entire file, allowing verification to occur incrementally during download.

| Verification Stage | Data Verified | Hash Source | Action on Mismatch |
|-------------------|---------------|-------------|-------------------|
| **Torrent File** | Info dictionary metadata | SHA-1 of bencoded info dict | Reject torrent as invalid |
| **Individual Pieces** | Each 256KB-1MB piece | Pre-computed SHA-1 in pieces list | Discard piece, request from different peer |
| **Complete File** | Assembled final file | Recompute all piece hashes | File complete and verified |

> **Critical Security Property**: The piece hashing system ensures that even if malicious peers control 99% of the swarm, they cannot corrupt your download as long as at least one honest peer has each piece. The cryptographic verification makes corruption detectable with overwhelming probability.

**Challenge 3: Fair Sharing Incentives**

The third challenge is the **free rider problem**: in a system where anyone can download files, what prevents everyone from downloading without uploading, causing the system to collapse from lack of seeders? This is a classic tragedy of the commons scenario.

BitTorrent addresses this through **tit-for-tat reciprocity** and the **choking algorithm**. Peers track how much data they've uploaded to and downloaded from each connected peer. Each peer "unchokes" (allows downloads from) only a limited number of other peers at any time, prioritizing peers who are uploading data back to them.

The choking mechanism creates **economic incentives** for sharing. Peers who upload get better download performance because other peers prioritize them. Peers who only try to download get "choked" by most peers, severely limiting their download speed. This creates a virtuous cycle where contributing to the network directly improves your own experience.

| Peer Relationship | Upload to Them | Download from Them | Their Incentive | Your Benefit |
|-------------------|----------------|-------------------|----------------|--------------|
| **Mutual Exchange** | High | High | You provide content they want | Fast downloads from them |
| **Optimistic Unchoke** | High | Low (they're new) | Bootstraps new peers into economy | Potential future reciprocation |
| **Free Rider** | Low/Zero | Low (choked) | Strong incentive to start sharing | Preserve bandwidth for contributors |
| **Seed** | High | Zero (they're complete) | Altruistic sharing | Community health, future availability |

The **optimistic unchoke** mechanism provides a crucial bootstrap path for new peers who haven't yet contributed to the swarm. Periodically, peers will unchoke a random peer regardless of reciprocation history, giving newcomers a chance to start downloading and begin contributing.

> **Economic Design Insight**: BitTorrent's incentive system transforms file sharing from a pure public good (where rational actors free ride) into a **reciprocal economy** where contribution directly improves individual outcomes. This aligns individual incentives with system health.

These three challenges - peer discovery, content verification, and fair sharing - represent the core technical problems that any practical P2P file sharing system must solve. BitTorrent's solutions to these problems have proven remarkably durable, with the core protocols remaining largely unchanged for over two decades while handling enormous volumes of global file sharing traffic.

The interdependence of these solutions is crucial to understand. The tracker system enables rapid peer discovery, but peers must implement content verification because they can't trust discovered peers. The verification system enables safe downloading from untrusted sources, but the incentive system ensures there are sources worth downloading from. Each component reinforces the others to create a robust, self-sustaining ecosystem.

![BitTorrent Client Architecture](./diagrams/system-architecture.svg)

### Implementation Guidance

This section provides practical guidance for implementing the BitTorrent client architecture in Go, with complete starter code for infrastructure components and detailed skeletons for core learning components.

**A. Technology Recommendations**

| Component | Simple Option | Advanced Option | Recommendation |
|-----------|---------------|-----------------|----------------|
| **HTTP Client** | `net/http` package | Custom HTTP with connection pooling | Simple - Go's HTTP client handles tracker communication well |
| **TCP Networking** | `net.Conn` with manual protocol | Framework like `github.com/pion/webrtc` | Simple - BitTorrent wire protocol is straightforward TCP |
| **Concurrency** | Goroutines with channels | Worker pool libraries | Simple - Go's built-in concurrency is perfect for P2P |
| **File I/O** | `os.File` with `io` interfaces | Memory-mapped files with `mmap` | Simple - Sequential piece writing doesn't need mmap |
| **Hashing** | `crypto/sha1` package | Hardware-accelerated crypto | Simple - SHA-1 performance is sufficient |
| **Bencode Parsing** | Custom recursive parser | Third-party bencode library | **Custom** - Parsing is core learning objective |

**B. Recommended Module Structure**

```
bittorrent-client/
├── cmd/
│   └── bt-client/
│       └── main.go              ← CLI entry point, argument parsing
├── internal/
│   ├── bencode/
│   │   ├── decoder.go           ← Bencode parser (Milestone 1)
│   │   └── decoder_test.go
│   ├── torrent/
│   │   ├── metainfo.go          ← Torrent file structures
│   │   ├── parser.go            ← Torrent parsing (Milestone 1)
│   │   └── parser_test.go
│   ├── tracker/
│   │   ├── client.go            ← HTTP tracker client (Milestone 2)
│   │   ├── announce.go          ← Announce request/response
│   │   └── tracker_test.go
│   ├── peer/
│   │   ├── connection.go        ← Peer wire protocol (Milestone 3)
│   │   ├── handshake.go         ← Handshake implementation
│   │   ├── messages.go          ← Protocol message types
│   │   └── peer_test.go
│   ├── piece/
│   │   ├── manager.go           ← Piece download coordination (Milestone 4)
│   │   ├── verification.go      ← Hash verification
│   │   └── piece_test.go
│   └── client/
│       ├── client.go            ← Main BitTorrent client orchestration
│       └── client_test.go
├── pkg/                         ← Public APIs (if building library)
├── testdata/                    ← Sample .torrent files for testing
├── go.mod
└── README.md
```

**C. Infrastructure Starter Code**

Here's complete, working infrastructure code that handles non-core concerns:

```go
// internal/torrent/metainfo.go - Complete data structures
package torrent

import (
    "crypto/sha1"
    "time"
)

// MetaInfo represents the complete parsed torrent file
type MetaInfo struct {
    Announce     string    `json:"announce"`
    CreationDate time.Time `json:"creation_date,omitempty"`
    Comment      string    `json:"comment,omitempty"`
    InfoHash     [20]byte  `json:"info_hash"`
    Info         Info      `json:"info"`
}

// Info represents the info dictionary from the torrent file
type Info struct {
    Name        string `json:"name"`
    Length      int64  `json:"length"`           // Single file mode
    PieceLength int64  `json:"piece_length"`
    Pieces      []byte `json:"pieces"`           // SHA-1 hashes concatenated
    Files       []File `json:"files,omitempty"`  // Multi-file mode
}

// File represents a file in multi-file torrent mode
type File struct {
    Length int64    `json:"length"`
    Path   []string `json:"path"`
}

// GetPieceHashes returns individual piece hashes from the concatenated pieces field
func (i *Info) GetPieceHashes() [][20]byte {
    numPieces := len(i.Pieces) / 20
    hashes := make([][20]byte, numPieces)
    for i := 0; i < numPieces; i++ {
        copy(hashes[i][:], i.Pieces[i*20:(i+1)*20])
    }
    return hashes
}

// CalculateInfoHash computes SHA-1 hash of bencoded info dictionary
func CalculateInfoHash(bencoded []byte) [20]byte {
    return sha1.Sum(bencoded)
}
```

```go
// internal/tracker/announce.go - Complete tracker protocol structures
package tracker

import (
    "net/url"
    "strconv"
    "time"
)

// AnnounceRequest represents parameters sent to tracker
type AnnounceRequest struct {
    InfoHash   [20]byte
    PeerID     [20]byte
    Port       int
    Uploaded   int64
    Downloaded int64
    Left       int64
    Event      string // "started", "completed", "stopped", or empty
}

// AnnounceResponse represents tracker's response
type AnnounceResponse struct {
    Interval    int    `bencode:"interval"`
    Complete    int    `bencode:"complete"`
    Incomplete  int    `bencode:"incomplete"`
    Peers       []byte `bencode:"peers"`        // Compact format
    FailureReason string `bencode:"failure reason,omitempty"`
}

// PeerInfo represents a single peer from tracker response
type PeerInfo struct {
    IP   [4]byte
    Port uint16
}

// BuildAnnounceURL constructs the complete tracker announce URL
func (req *AnnounceRequest) BuildAnnounceURL(announceURL string) string {
    base, _ := url.Parse(announceURL)
    params := url.Values{}
    
    params.Set("info_hash", string(req.InfoHash[:]))
    params.Set("peer_id", string(req.PeerID[:]))
    params.Set("port", strconv.Itoa(req.Port))
    params.Set("uploaded", strconv.FormatInt(req.Uploaded, 10))
    params.Set("downloaded", strconv.FormatInt(req.Downloaded, 10))
    params.Set("left", strconv.FormatInt(req.Left, 10))
    params.Set("compact", "1")
    
    if req.Event != "" {
        params.Set("event", req.Event)
    }
    
    base.RawQuery = params.Encode()
    return base.String()
}

// ParseCompactPeers decodes compact peer list from tracker response
func ParseCompactPeers(compactPeers []byte) []PeerInfo {
    numPeers := len(compactPeers) / 6
    peers := make([]PeerInfo, numPeers)
    
    for i := 0; i < numPeers; i++ {
        offset := i * 6
        copy(peers[i].IP[:], compactPeers[offset:offset+4])
        peers[i].Port = uint16(compactPeers[offset+4])<<8 + uint16(compactPeers[offset+5])
    }
    
    return peers
}
```

**D. Core Logic Skeleton Code**

For the main learning objectives, provide detailed skeletons that map to the algorithm steps:

```go
// internal/bencode/decoder.go - Core learning skeleton
package bencode

import (
    "bufio"
    "errors"
    "io"
)

// Decoder handles recursive parsing of bencode data
type Decoder struct {
    reader *bufio.Reader
}

// NewDecoder creates a decoder for the given data stream
func NewDecoder(r io.Reader) *Decoder {
    return &Decoder{reader: bufio.NewReader(r)}
}

// Decode parses the next bencode value and returns it
// This is the main entry point for bencode parsing
func (d *Decoder) Decode() (interface{}, error) {
    // TODO 1: Peek at the next byte to determine the type
    // TODO 2: If byte is 'i', call d.decodeInteger()
    // TODO 3: If byte is 'l', call d.decodeList()
    // TODO 4: If byte is 'd', call d.decodeDictionary()
    // TODO 5: If byte is digit (0-9), call d.decodeString()
    // TODO 6: Return error for any other byte value
    // Hint: Use d.reader.Peek(1) to look at next byte without consuming it
    return nil, errors.New("not implemented")
}

// decodeString parses a bencode string: <length>:<content>
func (d *Decoder) decodeString() ([]byte, error) {
    // TODO 1: Read digits until ':' to get length
    // TODO 2: Convert length string to integer
    // TODO 3: Read exactly 'length' bytes for string content
    // TODO 4: Return the raw bytes (don't convert to string - could be binary)
    // Hint: Use d.reader.ReadBytes(':') to read until delimiter
    return nil, errors.New("not implemented")
}

// decodeInteger parses a bencode integer: i<number>e
func (d *Decoder) decodeInteger() (int64, error) {
    // TODO 1: Read and verify 'i' prefix
    // TODO 2: Read digits (and possible '-') until 'e'
    // TODO 3: Convert to int64 using strconv.ParseInt
    // TODO 4: Read and verify 'e' suffix
    return 0, errors.New("not implemented")
}

// decodeList parses a bencode list: l<item1><item2>...e
func (d *Decoder) decodeList() ([]interface{}, error) {
    // TODO 1: Read and verify 'l' prefix
    // TODO 2: Create empty slice for results
    // TODO 3: Loop until next byte is 'e':
    // TODO 4:   Call d.Decode() recursively for each item
    // TODO 5:   Append result to slice
    // TODO 6: Read and verify 'e' suffix
    // TODO 7: Return completed slice
    return nil, errors.New("not implemented")
}

// decodeDictionary parses a bencode dictionary: d<key1><value1><key2><value2>...e
func (d *Decoder) decodeDictionary() (map[string]interface{}, error) {
    // TODO 1: Read and verify 'd' prefix
    // TODO 2: Create empty map for results
    // TODO 3: Loop until next byte is 'e':
    // TODO 4:   Call d.decodeString() for key (keys are always strings)
    // TODO 5:   Call d.Decode() recursively for value
    // TODO 6:   Store key-value pair in map
    // TODO 7: Read and verify 'e' suffix
    // TODO 8: Return completed map
    // Hint: Convert key bytes to string: string(keyBytes)
    return nil, errors.New("not implemented")
}
```

```go
// internal/client/client.go - Main orchestration skeleton
package client

import (
    "context"
    "sync"
)

// Client coordinates all aspects of BitTorrent downloading
type Client struct {
    metaInfo     *torrent.MetaInfo
    trackerClient *tracker.Client
    peers        map[string]*peer.Connection
    pieceManager *piece.Manager
    
    // Synchronization
    mu           sync.RWMutex
    ctx          context.Context
    cancel       context.CancelFunc
}

// NewClient creates a BitTorrent client for the given torrent
func NewClient(metaInfo *torrent.MetaInfo) *Client {
    ctx, cancel := context.WithCancel(context.Background())
    return &Client{
        metaInfo: metaInfo,
        peers:    make(map[string]*peer.Connection),
        ctx:      ctx,
        cancel:   cancel,
    }
}

// Download orchestrates the complete download process
func (c *Client) Download(outputPath string) error {
    // TODO 1: Initialize piece manager with metaInfo piece data
    // TODO 2: Contact tracker to get initial peer list
    // TODO 3: Connect to multiple peers concurrently (start with 3-5)
    // TODO 4: Start download coordinator goroutine
    // TODO 5: Periodically re-announce to tracker (every 30 minutes)
    // TODO 6: Monitor for completion - all pieces downloaded and verified
    // TODO 7: Write final file and transition to seeding mode
    // TODO 8: Handle graceful shutdown on context cancellation
    return errors.New("not implemented")
}
```

**E. Language-Specific Hints**

Go-specific implementation tips for BitTorrent development:

- **Byte Handling**: Use `[]byte` for all binary data (piece hashes, peer IDs). Never convert to string unless displaying to user.
- **Network Timeouts**: Always set timeouts on network operations: `conn.SetDeadline(time.Now().Add(30*time.Second))`
- **Binary Protocol**: Use `encoding/binary` for reading/writing network byte order: `binary.BigEndian.Uint32(bytes)`
- **Concurrency**: Use `sync.RWMutex` for peer map access. Use channels to coordinate between goroutines.
- **Error Handling**: Wrap errors with context: `fmt.Errorf("failed to parse piece %d: %w", pieceIndex, err)`
- **File I/O**: Use `os.OpenFile` with `O_CREATE|O_WRONLY` flags. Call `file.Sync()` after writing pieces.
- **HTTP Client**: Reuse `http.Client` instance across tracker requests for connection pooling.

**F. Milestone Checkpoints**

After implementing each milestone, verify these behaviors:

**Milestone 1 Checkpoint - Torrent Parsing**:
```bash
go run cmd/bt-client/main.go parse testdata/sample.torrent
# Expected output:
# Announce URL: http://tracker.example.com:8080/announce
# Info Hash: 1234567890abcdef1234567890abcdef12345678
# File Name: example.txt
# File Size: 1048576 bytes
# Piece Length: 262144 bytes
# Number of Pieces: 4
```

**Milestone 2 Checkpoint - Tracker Communication**:
```bash
go run cmd/bt-client/main.go announce testdata/sample.torrent
# Expected output:
# Contacting tracker: http://tracker.example.com:8080/announce
# Received 15 peers:
#   192.168.1.100:6881
#   10.0.0.5:51234
#   [... more peers]
# Interval: 1800 seconds
# Complete: 5, Incomplete: 10
```

**Milestone 3 Checkpoint - Peer Protocol**:
```bash
go run cmd/bt-client/main.go connect testdata/sample.torrent 192.168.1.100:6881
# Expected output:
# Connecting to peer 192.168.1.100:6881
# Handshake successful: peer_id = abcdef1234567890abcdef1234567890abcdef12
# Received bitfield: 11110000 (peer has first 4 pieces)
# Peer unchoked us, ready to request pieces

```


## Goals and Non-Goals

> **Milestone(s):** This section defines the scope for all milestones (1-4) by establishing clear boundaries for what our BitTorrent client will and will not implement.

Building a complete BitTorrent client involves numerous complex features and protocols that have evolved over decades of peer-to-peer development. Without clear scope boundaries, it's easy to get overwhelmed by the sheer breadth of functionality that modern BitTorrent clients support. Think of this like planning a cross-country road trip: you need to decide upfront which cities you'll visit, which scenic routes you'll take, and which detours you'll skip, or you'll end up wandering aimlessly and never reach your destination.

This section establishes three critical categories of requirements: **functional goals** (the core features we must implement), **non-functional goals** (the quality attributes our system must exhibit), and **explicit non-goals** (features we deliberately choose not to implement). This scope definition serves as our north star throughout the implementation process, helping us make consistent decisions about complexity trade-offs and feature prioritization.

The goal-setting process for a BitTorrent client is particularly challenging because the protocol ecosystem includes dozens of optional extensions, performance optimizations, and compatibility quirks developed over 20+ years of evolution. A production BitTorrent client like qBittorrent or Transmission supports hundreds of features. Our learning-focused implementation needs to capture the essential complexity of BitTorrent without getting bogged down in edge cases that obscure the core concepts.

### Functional Goals

Our BitTorrent client will implement the four core pillars of BitTorrent functionality that enable complete participation in a peer-to-peer swarm. These represent the minimum viable feature set for a working BitTorrent client that can successfully download and share files with other peers in the ecosystem.

> **Decision: Core BitTorrent Protocol Implementation**
> - **Context**: BitTorrent encompasses many protocol extensions and optional features developed over two decades
> - **Options Considered**: 
>   1. Implement only basic downloading without seeding
>   2. Implement full BitTorrent v1 protocol with downloading and seeding
>   3. Include modern extensions like DHT and magnet links
> - **Decision**: Implement complete BitTorrent v1 protocol including seeding capability
> - **Rationale**: Seeding is essential for understanding BitTorrent's reciprocity model and makes our client a good network citizen. BitTorrent v1 provides all core concepts without extension complexity.
> - **Consequences**: More implementation complexity than download-only, but creates a complete learning experience and functional client

| Functional Goal | Description | Key Components | Success Criteria |
|-----------------|-------------|----------------|------------------|
| **Torrent File Parsing** | Parse `.torrent` files to extract all metadata needed for downloading | Bencode decoder, metadata extractor, info hash calculator | Successfully parse real torrent files and extract announce URL, file info, piece hashes |
| **Tracker Communication** | Communicate with HTTP trackers to discover peers | HTTP client, announce protocol, peer list parser | Receive valid peer lists from public trackers, handle tracker responses and errors |
| **Peer Wire Protocol** | Implement complete peer-to-peer communication protocol | TCP connection manager, message parser, state machine | Successfully handshake with real BitTorrent clients, exchange messages, download blocks |
| **File Download Management** | Coordinate downloading file pieces from multiple peers | Piece scheduler, download orchestrator, content verifier | Download complete files that match original content hash, handle peer disconnections |
| **Content Verification** | Verify integrity of downloaded data using cryptographic hashes | SHA-1 hasher, piece validator, corruption detector | Reject corrupted pieces, re-request invalid data, ensure file integrity |
| **Upload and Seeding** | Serve file pieces to other peers requesting them | Upload manager, request responder, bandwidth tracker | Successfully seed files to other clients, track upload statistics for tracker reporting |

**Torrent File Parsing** forms the foundation of our client by extracting all metadata required for the download process. The Bencode decoder must handle the four Bencode data types (strings, integers, lists, dictionaries) with proper binary string support. The metadata extractor pulls critical information including the tracker announce URL, file specifications, and piece hash list. Most importantly, the info hash calculation must compute the SHA-1 digest from the exact bencoded bytes of the info dictionary, as this serves as the unique identifier for the torrent across the entire BitTorrent network.

**Tracker Communication** implements the HTTP-based announce protocol that serves as BitTorrent's peer discovery mechanism. Our client must construct properly formatted announce requests with URL-encoded parameters including the info hash, peer ID, port number, and transfer statistics. The tracker responds with a list of peers in compact binary format (6 bytes per peer: 4 bytes IP + 2 bytes port) that we must parse into usable network addresses. The client must also implement the announce interval system, re-contacting the tracker periodically to report progress and discover new peers.

**Peer Wire Protocol** represents the most complex functional requirement, implementing the standardized message-based protocol that BitTorrent peers use to coordinate file transfers. This includes the 68-byte handshake sequence that establishes connections, the length-prefixed message framing system, and the complete set of peer protocol messages (choke, unchoke, interested, have, bitfield, request, piece). The implementation must maintain per-peer state machines tracking choking and interest status, manage request pipelining for optimal throughput, and handle the intricate dance of peer coordination.

**File Download Management** orchestrates the complex process of downloading file pieces from multiple peers simultaneously. This requires implementing piece selection algorithms (starting with random selection, evolving toward rarest-first for efficiency), managing concurrent downloads to avoid requesting the same piece multiple times, and coordinating the assembly of completed pieces into the final output file. The download manager must handle peer disconnections gracefully, redistributing incomplete piece requests to other available peers.

**Content Verification** ensures the integrity of downloaded data by validating each piece against its expected SHA-1 hash from the torrent metadata. When a piece download completes, the client must compute the piece's hash and compare it against the corresponding 20-byte hash from the torrent's piece hash list. Corrupted or invalid pieces must be discarded and re-requested from different peers. This verification system is critical for security and prevents the propagation of corrupted data through the swarm.

**Upload and Seeding** completes the reciprocal nature of BitTorrent by serving file pieces to other peers. When peers send request messages for specific pieces our client possesses, we must respond with the corresponding piece data. This requires tracking which pieces we have available, managing upload bandwidth fairly across requesting peers, and maintaining statistics on bytes uploaded for tracker reporting. Effective seeding behavior is essential for swarm health and BitTorrent's long-term sustainability.

### Non-Functional Goals

Beyond core functionality, our BitTorrent client must meet specific quality attributes that determine its effectiveness and usability in real-world scenarios. These non-functional requirements define how well the system performs its functional goals rather than what it does.

> **Decision: Performance vs. Simplicity Balance**
> - **Context**: BitTorrent clients can be optimized for extreme performance with complex algorithms, or kept simple for educational clarity
> - **Options Considered**:
>   1. Prioritize maximum performance with advanced optimizations
>   2. Prioritize code clarity and educational value
>   3. Balance reasonable performance with maintainable code
> - **Decision**: Target reasonable performance with maintainable, well-documented code
> - **Rationale**: Educational goals require readable code, but unusably slow performance would prevent real-world testing and learning
> - **Consequences**: May not match production client performance, but provides solid foundation for learning and future optimization

| Non-Functional Goal | Requirement | Measurement Criteria | Implementation Strategy |
|---------------------|-------------|---------------------|------------------------|
| **Download Performance** | Achieve reasonable download speeds comparable to other clients | Download 100MB+ files at 70%+ of available bandwidth | Implement request pipelining, concurrent peer connections, efficient piece selection |
| **Memory Efficiency** | Handle large torrents without excessive memory usage | Support 1GB+ torrents with <100MB RAM usage | Stream piece data to disk, avoid loading entire files in memory |
| **Connection Reliability** | Maintain stable connections with multiple peers | Successfully maintain 10+ concurrent peer connections | Implement proper TCP connection management, handle network interruptions |
| **Error Recovery** | Gracefully handle common failure scenarios | Recover from tracker failures, peer disconnections, corrupted data | Comprehensive error handling, retry logic, fallback mechanisms |
| **Resource Fairness** | Share upload bandwidth fairly among requesting peers | No single peer monopolizes upload capacity | Implement upload slot management, fair queuing for requests |
| **Code Maintainability** | Produce readable, well-documented, testable code | All components have unit tests, clear module boundaries | Modular architecture, comprehensive documentation, test coverage |

**Download Performance** ensures our client can achieve reasonable throughput when downloading files from healthy swarms. While we don't need to match the performance of highly optimized production clients like libtorrent, the client must download files fast enough to be practically useful. This requires implementing request pipelining (maintaining multiple outstanding block requests per peer), managing concurrent connections to multiple peers, and using effective piece selection strategies that minimize download time.

**Memory Efficiency** becomes critical when handling large torrents with hundreds or thousands of pieces. Our client must avoid loading entire files into memory, instead streaming piece data directly to disk as it arrives. The piece management system should maintain metadata about piece availability and download progress without storing the actual piece content in RAM. This approach enables handling multi-gigabyte torrents on resource-constrained systems.

**Connection Reliability** addresses the inherent instability of peer-to-peer networks where peers frequently join and leave the swarm. Our client must detect failed connections promptly, clean up resources from disconnected peers, and continuously discover new peers to replace lost connections. The connection management system should handle common network issues like timeouts, connection resets, and temporary network partitions without crashing or stalling downloads.

**Error Recovery** encompasses the broad category of graceful degradation when things go wrong. Tracker servers may become unavailable, requiring fallback strategies or retry logic. Peers may send malformed messages that require defensive parsing. Downloaded pieces may fail hash verification and need re-downloading from different sources. The client must handle these scenarios without human intervention, automatically recovering and continuing the download process.

**Resource Fairness** reflects BitTorrent's fundamental principle of reciprocity – peers that contribute upload bandwidth should receive better download performance. Our client must avoid selfish behavior that would harm swarm health, such as downloading without seeding or monopolizing other peers' upload capacity. This includes implementing reasonable limits on concurrent requests per peer and fairly distributing our upload capacity among requesting peers.

**Code Maintainability** recognizes that this implementation serves educational purposes and may be extended or modified by learners. The codebase must prioritize clarity and comprehensibility over micro-optimizations. Each component should have well-defined responsibilities, clear interfaces, and comprehensive unit tests. Documentation should explain not just what the code does, but why specific design decisions were made.

### Explicit Non-Goals

Defining what we will NOT implement is equally important as defining what we will implement. These explicit non-goals help maintain project scope and prevent feature creep that could derail the core learning objectives. Modern BitTorrent clients implement dozens of protocol extensions and advanced features that, while valuable in production, would obscure the fundamental concepts we're trying to teach.

> **Decision: Exclude Advanced BitTorrent Extensions**
> - **Context**: Modern BitTorrent includes many protocol extensions (DHT, PEX, encryption, etc.) that provide additional functionality
> - **Options Considered**:
>   1. Implement all major extensions for feature completeness
>   2. Implement basic extensions like DHT for modern compatibility
>   3. Focus solely on core BitTorrent v1 protocol
> - **Decision**: Exclude all protocol extensions and advanced features
> - **Rationale**: Extensions add significant complexity without teaching new fundamental concepts. Core protocol provides complete learning experience.
> - **Consequences**: Client won't support magnet links or trackerless operation, but remains focused on essential BitTorrent concepts

| Explicit Non-Goal | Rationale | Complexity Impact | Alternative Approach |
|-------------------|-----------|-------------------|---------------------|
| **Distributed Hash Table (DHT)** | DHT is a complex distributed systems topic that deserves separate study | Would add 500+ lines of complex networking code | Use tracker-based torrents only |
| **Magnet Link Support** | Requires DHT implementation for metadata download | Depends on DHT, adds metadata resolution complexity | Work with traditional `.torrent` files |
| **Protocol Encryption** | Crypto implementation is error-prone and adds little educational value | SSL/TLS integration, key exchange protocols | Rely on network-level security if needed |
| **Multi-Tracker Support** | Tracker failover logic adds complexity without new concepts | Redundant with single-tracker error handling | Use single primary tracker per torrent |
| **Peer Exchange (PEX)** | Peer discovery optimization that doesn't change core concepts | Additional message types and peer management | Rely on tracker for peer discovery |
| **Web Seeding (HTTP/FTP)** | Hybrid P2P/client-server model that complicates architecture | HTTP client integration, different download paths | Pure P2P downloading only |
| **Selective File Download** | File filtering within multi-file torrents | Piece-to-file mapping complexity, partial completion | Download complete torrents only |
| **Bandwidth Limiting** | QoS feature that doesn't impact core protocol understanding | Rate limiting algorithms, traffic shaping | Rely on OS-level bandwidth management |
| **Advanced Piece Selection** | Sophisticated algorithms like endgame mode and smart seeding | Complex heuristics and optimization logic | Use simple rarest-first selection |
| **Resume/Persistence** | Saving and restoring partial download state | Serialization, state management, crash recovery | Restart downloads from beginning |

**Distributed Hash Table (DHT)** represents a completely separate distributed systems concept that enables trackerless operation. While DHT is fascinating technology, implementing it properly requires understanding distributed consensus, network partitions, routing table maintenance, and peer bootstrapping – topics that could fill their own comprehensive tutorial. Including DHT would double the project's complexity while teaching concepts orthogonal to BitTorrent's core file-sharing mechanisms.

**Magnet Link Support** depends heavily on DHT for metadata resolution, since magnet links contain only the info hash rather than complete torrent metadata. Without DHT, magnet links become unusable, making this feature dependent on our DHT non-goal. Traditional `.torrent` files provide all necessary metadata upfront, eliminating the need for complex metadata discovery protocols.

**Protocol Encryption** adds cryptographic complexity that doesn't enhance understanding of BitTorrent's core concepts. Implementing encryption properly requires deep knowledge of cryptographic protocols, key exchange mechanisms, and security best practices – specialized knowledge that's orthogonal to peer-to-peer file sharing concepts. Network-level security (VPN, HTTPS) can provide encryption when needed without complicating the BitTorrent implementation.

**Multi-Tracker Support** allows torrents to specify backup trackers for redundancy, but the failover logic and tracker prioritization algorithms don't teach new concepts beyond single-tracker communication. The error handling patterns for tracker failures apply equally whether dealing with one tracker or ten. Supporting multi-tracker scenarios would add configuration complexity and edge cases without educational benefit.

**Peer Exchange (PEX)** enables peers to share information about other peers in the swarm, providing an additional peer discovery mechanism beyond tracker announcements. While PEX improves swarm connectivity and reduces tracker load, it's essentially an optimization that doesn't change the fundamental peer-to-peer download process. Tracker-based peer discovery provides sufficient peer connectivity for learning purposes.

**Web Seeding** allows torrents to include HTTP or FTP servers as additional download sources, creating a hybrid P2P/client-server architecture. This feature requires implementing HTTP range requests, integrating with existing piece management systems, and handling the different reliability characteristics of web servers versus P2P peers. The added complexity doesn't reinforce core BitTorrent concepts and introduces architectural complications.

**Selective File Download** enables users to choose specific files from multi-file torrents, downloading only desired content. This requires complex piece-to-file mapping logic, partial torrent completion tracking, and modified seeding behavior for incomplete torrents. While useful in practice, selective downloading obscures the fundamental concept of torrents as atomic units of content sharing.

**Bandwidth Limiting** provides quality-of-service features that help users manage network resource consumption. However, implementing effective bandwidth limiting requires understanding rate limiting algorithms, traffic shaping, and network scheduling – topics that don't reinforce BitTorrent protocol concepts. Operating system tools and network-level QoS provide better bandwidth management without complicating the BitTorrent implementation.

**Advanced Piece Selection** algorithms like endgame mode (requesting final pieces from all peers) and smart seeding (prioritizing rare pieces when seeding) can significantly improve download performance and swarm health. However, these optimizations involve complex heuristics and game-theoretic considerations that obscure the basic piece selection concept. Simple rarest-first selection provides adequate performance while remaining easy to understand and implement.

**Resume/Persistence** enables users to stop and restart downloads without losing progress, requiring serialization of download state, piece completion tracking, and crash recovery logic. While extremely valuable for user experience, persistence adds significant implementation complexity around state management and error recovery that doesn't teach new BitTorrent protocol concepts. Starting downloads from scratch simplifies implementation and testing.

> The key insight behind these non-goals is that BitTorrent's core concepts – bencoding, tracker communication, peer protocol, and piece management – provide a complete and challenging learning experience. Advanced features, while valuable in production systems, often optimize or extend these core concepts without introducing fundamentally new ideas. By maintaining strict scope boundaries, we ensure learners master the essential concepts before exploring the rich ecosystem of BitTorrent extensions.

### Implementation Guidance

This section provides concrete technology recommendations and architectural guidance for implementing our scoped BitTorrent client in Go.

#### A. Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| **HTTP Client** | `net/http` with default client | `net/http` with custom transport and timeouts |
| **TCP Networking** | `net.Conn` with basic I/O | `net.Conn` with connection pooling and buffering |
| **Concurrency** | Goroutines with channels | Worker pools with `sync.WaitGroup` coordination |
| **File I/O** | `os.File` with synchronous writes | `os.File` with buffered writes and fsync |
| **Hashing** | `crypto/sha1` standard library | `crypto/sha1` with pre-allocated buffers |
| **Bencode** | Custom recursive parser | Third-party library (not recommended for learning) |
| **Logging** | `log` standard library | Structured logging with `slog` (Go 1.21+) |
| **Testing** | `testing` with table-driven tests | `testing` with property-based testing |
| **Configuration** | Hard-coded constants | `flag` package for command-line options |

The simple options provide everything needed for a functional BitTorrent client while keeping dependencies minimal and code transparent. Advanced options offer better performance and production-readiness but add complexity that may distract from core learning objectives.

#### B. Recommended File/Module Structure

```
bittorrent-client/
├── cmd/
│   └── bittorrent/
│       └── main.go                 ← CLI entry point
├── internal/
│   ├── bencode/
│   │   ├── decoder.go              ← Bencode parsing logic
│   │   └── decoder_test.go
│   ├── torrent/
│   │   ├── metainfo.go            ← Torrent file structures and parsing
│   │   └── metainfo_test.go
│   ├── tracker/
│   │   ├── client.go              ← HTTP tracker communication
│   │   ├── announce.go            ← Announce request/response handling
│   │   └── client_test.go
│   ├── peer/
│   │   ├── connection.go          ← Individual peer connection management
│   │   ├── protocol.go            ← Wire protocol message handling
│   │   ├── handshake.go          ← Connection handshake logic
│   │   └── connection_test.go
│   ├── piece/
│   │   ├── manager.go             ← Piece download coordination
│   │   ├── scheduler.go           ← Piece selection algorithms
│   │   ├── verifier.go           ← Content hash verification
│   │   └── manager_test.go
│   └── client/
│       ├── client.go              ← Main BitTorrent client orchestration
│       └── client_test.go
├── pkg/
│   └── bittorrent/
│       └── types.go               ← Public API types and constants
├── testdata/
│   ├── sample.torrent            ← Test torrent files
│   └── sample_content/           ← Expected file content for tests
├── go.mod
├── go.sum
└── README.md
```

This structure separates concerns cleanly: `cmd/` contains the executable entry point, `internal/` contains implementation packages that map to our architectural components, `pkg/` exposes public APIs for potential library use, and `testdata/` provides test fixtures. Each component gets its own package with focused responsibilities and comprehensive tests.

#### C. Infrastructure Starter Code

**File: `internal/bencode/decoder.go`**
```go
package bencode

import (
    "bufio"
    "errors"
    "fmt"
    "io"
    "strconv"
)

// Common bencode parsing errors
var (
    ErrUnexpectedEOF    = errors.New("unexpected end of input")
    ErrInvalidFormat    = errors.New("invalid bencode format")
    ErrInvalidInteger   = errors.New("invalid integer format")
    ErrInvalidString    = errors.New("invalid string format")
)

// Decoder handles bencode parsing from an input stream
type Decoder struct {
    reader *bufio.Reader
}

// NewDecoder creates a new bencode decoder for the given reader
func NewDecoder(r io.Reader) *Decoder {
    return &Decoder{
        reader: bufio.NewReader(r),
    }
}

// peekByte returns the next byte without consuming it
func (d *Decoder) peekByte() (byte, error) {
    bytes, err := d.reader.Peek(1)
    if err != nil {
        return 0, err
    }
    return bytes[0], nil
}

// readByte consumes and returns the next byte
func (d *Decoder) readByte() (byte, error) {
    return d.reader.ReadByte()
}

// readUntil reads bytes until the specified delimiter
func (d *Decoder) readUntil(delim byte) ([]byte, error) {
    return d.reader.ReadSlice(delim)
}

// Decode parses the next bencode value from the input stream
func (d *Decoder) Decode() (interface{}, error) {
    // TODO: Implement main decode dispatch based on first byte
    // Hint: Look at first character to determine type:
    // '0'-'9': string (length prefix)
    // 'i': integer
    // 'l': list  
    // 'd': dictionary
    return nil, fmt.Errorf("decode not implemented")
}
```

**File: `pkg/bittorrent/types.go`**
```go
package bittorrent

import "time"

// Standard BitTorrent constants
const (
    PIECE_HASH_SIZE    = 20  // SHA-1 produces 20-byte hashes
    COMPACT_PEER_SIZE  = 6   // 4 bytes IP + 2 bytes port
    BLOCK_SIZE         = 16384 // Standard 16KB block size
    MAX_PEERS          = 50  // Reasonable concurrent connection limit
)

// MetaInfo represents the complete parsed torrent file
type MetaInfo struct {
    Announce     string    // Primary tracker announce URL
    CreationDate time.Time // When torrent was created
    Comment      string    // Optional comment
    InfoHash     [20]byte  // SHA-1 hash of info dictionary
    Info         Info      // File and piece information
}

// Info contains the core file and piece information from torrent
type Info struct {
    Name        string  // Suggested filename or directory name
    Length      int64   // Total file size (single-file torrents only)
    PieceLength int64   // Size of each piece except possibly the last
    Pieces      []byte  // Concatenated SHA-1 hashes of all pieces
    Files       []File  // File list (multi-file torrents only)
}

// File represents a single file in a multi-file torrent
type File struct {
    Length int64    // File size in bytes
    Path   []string // Path components for file location
}

// GetPieceHashes splits the concatenated piece hashes into individual hashes
func (i *Info) GetPieceHashes() [][20]byte {
    numPieces := len(i.Pieces) / PIECE_HASH_SIZE
    hashes := make([][20]byte, numPieces)
    
    for i := 0; i < numPieces; i++ {
        copy(hashes[i][:], i.Pieces[i*PIECE_HASH_SIZE:(i+1)*PIECE_HASH_SIZE])
    }
    
    return hashes
}

// PeerInfo represents a peer's network address
type PeerInfo struct {
    IP   [4]byte // IPv4 address
    Port uint16  // TCP port number
}

// String formats peer info as "IP:port"
func (p PeerInfo) String() string {
    return fmt.Sprintf("%d.%d.%d.%d:%d", p.IP[0], p.IP[1], p.IP[2], p.IP[3], p.Port)
}
```

#### D. Core Logic Skeleton Code

**File: `internal/torrent/metainfo.go`**
```go
package torrent

import (
    "crypto/sha1"
    "fmt"
    "time"
    
    "your-project/internal/bencode"
    "your-project/pkg/bittorrent"
)

// ParseFromFile reads and parses a torrent file from disk
func ParseFromFile(filepath string) (*bittorrent.MetaInfo, error) {
    // TODO 1: Open the torrent file for reading
    // TODO 2: Create bencode decoder for the file
    // TODO 3: Decode the root dictionary
    // TODO 4: Extract announce URL from root dict
    // TODO 5: Extract info dictionary from root dict  
    // TODO 6: Calculate info hash from bencoded info dict bytes
    // TODO 7: Parse info dictionary into Info struct
    // TODO 8: Return populated MetaInfo struct
    // Hint: Use CalculateInfoHash() helper function
    return nil, fmt.Errorf("not implemented")
}

// CalculateInfoHash computes SHA-1 hash of bencoded info dictionary
func CalculateInfoHash(bencoded []byte) [20]byte {
    // TODO 1: Create SHA-1 hasher
    // TODO 2: Write bencoded bytes to hasher
    // TODO 3: Get final hash sum as [20]byte array
    // TODO 4: Return the hash
    // Hint: Use crypto/sha1 package
    return [20]byte{}
}

// parseInfo converts bencode dictionary to Info struct
func parseInfo(infoDict map[string]interface{}) (bittorrent.Info, error) {
    // TODO 1: Extract "name" field as string
    // TODO 2: Extract "piece length" field as int64
    // TODO 3: Extract "pieces" field as byte slice
    // TODO 4: Check if single-file torrent (has "length" field)
    // TODO 5: If single-file, extract "length" field
    // TODO 6: If multi-file, extract "files" list and parse each file
    // TODO 7: Validate piece length and pieces alignment
    // TODO 8: Return populated Info struct
    // Hint: Handle both single-file and multi-file torrent formats
    return bittorrent.Info{}, fmt.Errorf("not implemented")
}
```

#### E. Language-Specific Hints

**Go-Specific Implementation Tips:**

- **Bencode Parsing**: Use `bufio.Reader` for efficient byte-by-byte parsing. The `ReadSlice()` method is particularly useful for reading until delimiters.

- **Binary Data Handling**: Bencode strings are binary data, not UTF-8 text. Always work with `[]byte` slices and avoid string conversion unless specifically needed for text fields.

- **Error Handling**: Create specific error types for different parsing failures. Use `errors.New()` for constant errors and `fmt.Errorf()` for contextual errors.

- **HTTP Requests**: Use `http.Get()` for simple tracker requests. For production code, create custom `http.Client` with timeouts: `&http.Client{Timeout: 30 * time.Second}`.

- **Concurrency**: Use goroutines for peer connections and channels for coordination. Pattern: `go handlePeer(conn, pieceManager)` for each peer connection.

- **File I/O**: Use `os.OpenFile()` with `O_CREATE|O_WRONLY` flags for output files. Call `file.Sync()` after writing each piece to ensure data persistence.

- **Network Byte Order**: BitTorrent uses big-endian byte order. Use `binary.BigEndian.Uint32()` and `binary.BigEndian.PutUint32()` for multi-byte integers.

- **Testing**: Create test torrents using existing BitTorrent clients. Keep small test files (< 1MB) for fast test execution.

#### F. Milestone Checkpoints

**After Milestone 1 (Torrent Parsing):**
```bash
# Test command
go test ./internal/bencode/... ./internal/torrent/...

# Expected behavior
# All tests pass
# Can parse real .torrent files 
# Info hash matches other BitTorrent clients
```

**Verification Steps:**
1. Download a small torrent file from a legal source (Linux distributions work well)
2. Parse it with your implementation and extract the info hash
3. Verify the info hash matches what other BitTorrent clients show
4. Check that announce URL and file information are correctly extracted

**After Milestone 2 (Tracker Communication):**
```bash
# Test with actual tracker
go run cmd/bittorrent/main.go announce sample.torrent

# Expected output
# Found X peers from tracker
# Peer list: [IP:port IP:port ...]
# No network errors or parsing failures
```

**After Milestone 3 (Peer Protocol):**
```bash
# Test peer connections
go run cmd/bittorrent/main.go connect sample.torrent

# Expected behavior  
# Successfully handshake with multiple peers
# Receive bitfield and piece messages
# No protocol violations or connection errors
```

**After Milestone 4 (Complete Download):**
```bash
# Full download test
go run cmd/bittorrent/main.go download sample.torrent output.file

# Expected behavior
# Download completes successfully
# Output file matches original content
# Can seed file to other clients
```

#### G. Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| **"Invalid info hash"** | Wrong bytes used for SHA-1 calculation | Compare raw info dict bytes with working client | Use exact bencoded bytes, not re-encoded |
| **"Tracker returns error"** | Malformed announce request | Check URL encoding of info hash and peer ID | Properly URL-encode binary data |
| **"Peers reject handshake"** | Wrong protocol string or info hash | Log handshake bytes, compare with specification | Use exact 19-byte protocol string |
| **"Download stalls"** | Not sending interested/request messages | Log peer message exchange | Send interested after bitfield, request after unchoke |
| **"Piece verification fails"** | Wrong piece boundaries or corruption | Compare piece hashes with torrent metadata | Check piece length calculation and SHA-1 implementation |
| **"Connection timeouts"** | Network issues or wrong peer addresses | Test peer connectivity with telnet/nc | Implement connection timeouts and retry logic |

**Common Implementation Mistakes:**

⚠️ **Pitfall: Re-encoding Info Dictionary**
Many learners calculate the info hash by parsing the info dictionary and then re-encoding it with their own bencode encoder. This fails because bencode allows multiple valid encodings of the same data (different key ordering, etc.). The info hash MUST be calculated from the exact original bytes.

⚠️ **Pitfall: String vs Binary Confusion**
Bencode "strings" are actually binary data. Converting them to Go strings can corrupt binary data like piece hashes or compact peer lists. Always work with `[]byte` slices until you specifically need text representation.

⚠️ **Pitfall: Blocking on Choked Peers**
New implementations often send piece requests immediately after handshake, but peers start in the "choked" state. You must wait for an "unchoke" message before sending requests, or the peer will ignore your requests and potentially disconnect.

This comprehensive scope definition and implementation guidance provides the foundation for building a complete BitTorrent client while maintaining focus on the essential concepts that make BitTorrent work.


## High-Level Architecture

> **Milestone(s):** This section establishes the architectural foundation for all milestones (1-4) by defining the core components and their interactions that will be built incrementally throughout the project.

Building a BitTorrent client is like orchestrating a complex distributed system where multiple specialized components must work together seamlessly. Think of it as organizing a massive library project where you need a **cataloger** to understand what books you're looking for (torrent parsing), a **directory service** to find libraries that have those books (tracker communication), a **librarian network** to manage relationships with other libraries (peer management), and a **book assembly team** to piece together chapters from different sources while verifying they're authentic (piece management).

The architecture follows a clear separation of concerns, where each component has a well-defined responsibility and communicates with others through clean interfaces. This modular design allows us to implement and test each component independently while maintaining the ability to coordinate complex multi-peer downloads. The key architectural insight is that BitTorrent's complexity lies not in any single protocol, but in the careful coordination between multiple concurrent processes that must share state and make decisions based on constantly changing network conditions.

![BitTorrent Client Architecture](./diagrams/system-architecture.svg)

### Component Responsibilities

The BitTorrent client architecture consists of four major components, each with distinct responsibilities and well-defined interfaces. Understanding these boundaries is crucial because BitTorrent's distributed nature means that each component must handle its own failures gracefully while contributing to the overall system's resilience.

| Component | Primary Responsibility | Key Interfaces | External Dependencies | State Managed |
|-----------|----------------------|----------------|---------------------|---------------|
| **TorrentParser** | Decode `.torrent` files and extract metadata required for download coordination | `ParseFromFile(filepath) (*MetaInfo, error)`, `GetPieceHashes() [][20]byte`, `CalculateInfoHash([]byte) [20]byte` | File system for reading torrent files | Parsed torrent metadata, info hash, piece information |
| **TrackerClient** | Communicate with tracker servers to discover peers and report download progress | `Announce(*AnnounceRequest) (*AnnounceResponse, error)`, `BuildAnnounceURL(announceURL) string`, `ParseCompactPeers([]byte) []PeerInfo` | HTTP client for tracker communication | Peer list, announce intervals, upload/download statistics |
| **PeerManager** | Manage connections to multiple peers and implement the BitTorrent wire protocol | `ConnectToPeer(PeerInfo) (*Connection, error)`, `SendHandshake(*Connection) error`, `ReceiveMessage(*Connection) (Message, error)` | TCP networking, peer connections | Connection states, message queues, peer capabilities |
| **PieceManager** | Coordinate piece downloads, verify content integrity, and schedule requests across peers | `RequestPiece(pieceIndex int) error`, `VerifyPiece(pieceIndex int, data []byte) bool`, `GetNextPiece() (int, error)`, `AssembleFile(outputPath string) error` | File system for writing completed pieces | Piece availability, download progress, verification status |

The **TorrentParser** serves as the system's entry point, transforming human-readable torrent files into the structured metadata that drives all subsequent operations. Its most critical responsibility is computing the **info hash** correctly, as this 20-byte identifier serves as the unique key that connects all other components. The parser must handle the intricacies of Bencode format while maintaining exact byte-for-byte fidelity when calculating the SHA-1 hash of the info dictionary.

The **TrackerClient** acts as the system's directory service, maintaining the bridge between our client and the centralized coordination provided by tracker servers. It translates our internal state into the HTTP-based tracker protocol and decodes the compact peer format that trackers use to efficiently transmit lists of available peers. This component must handle tracker failures gracefully and implement proper re-announce scheduling to maintain our presence in the swarm.

> **Key Architectural Insight:** The separation between TrackerClient and PeerManager reflects BitTorrent's hybrid design—centralized peer discovery with decentralized data transfer. This separation allows us to implement different tracker types (HTTP, UDP) without affecting peer communication logic.

The **PeerManager** implements the complex state machines required by the BitTorrent wire protocol. Each peer connection maintains independent state for choking, interest, and message queues, but the PeerManager coordinates these connections to maximize overall throughput. It must handle connection failures, implement proper handshaking, and manage the flow control mechanisms that prevent any single peer from overwhelming the client.

The **PieceManager** orchestrates the actual file reconstruction process, making decisions about which pieces to request from which peers based on availability and rarity. This component embodies the BitTorrent protocol's clever incentive mechanisms—by preferring rare pieces, it ensures that the entire swarm maintains good piece distribution. The PieceManager also handles the critical content verification step, rejecting corrupted data and coordinating retries when hash verification fails.

### Recommended Module Structure

The Go implementation benefits from a clean module structure that mirrors the architectural boundaries while providing clear import paths and testability. The structure follows Go conventions for internal packages while maintaining the flexibility to expose public APIs as the client evolves.

```
bittorrent-client/
├── cmd/
│   └── bittorrent/
│       └── main.go                    ← CLI entry point, orchestrates all components
├── internal/
│   ├── torrent/
│   │   ├── parser.go                  ← MetaInfo struct, ParseFromFile implementation
│   │   ├── bencode.go                 ← Bencode decoder with all four type handlers
│   │   ├── metadata.go                ← Info, File structs and hash calculations  
│   │   └── parser_test.go             ← Unit tests for parsing logic
│   ├── tracker/
│   │   ├── client.go                  ← TrackerClient struct, HTTP announce logic
│   │   ├── protocol.go                ← AnnounceRequest/Response structs
│   │   ├── peers.go                   ← Compact peer parsing, PeerInfo struct
│   │   └── client_test.go             ← Mock tracker tests
│   ├── peer/
│   │   ├── manager.go                 ← PeerManager coordination logic
│   │   ├── connection.go              ← Individual peer connection state machine
│   │   ├── protocol.go                ← Wire protocol message structs
│   │   ├── handshake.go               ← Handshake implementation
│   │   └── manager_test.go            ← Peer protocol tests
│   ├── piece/
│   │   ├── manager.go                 ← PieceManager with selection algorithms
│   │   ├── scheduler.go               ← Rarest-first and endgame logic
│   │   ├── verifier.go                ← SHA-1 verification and retry logic
│   │   ├── assembler.go               ← File reconstruction from pieces
│   │   └── manager_test.go            ← Piece management tests
│   └── client/
│       ├── client.go                  ← Main Client struct coordinating all components
│       ├── download.go                ← Download orchestration logic
│       └── client_test.go             ← Integration tests
├── pkg/                               ← Public APIs (if building a library)
│   └── bittorrent/
│       └── client.go                  ← Public client interface
├── testdata/                          ← Sample torrent files for testing
│   ├── sample.torrent
│   └── multi-file.torrent
├── go.mod                             ← Go module definition
└── README.md                          ← Usage instructions and examples
```

> **Decision: Internal Package Structure**
> - **Context**: Go's `internal/` package convention prevents external imports while allowing clean separation of concerns
> - **Options Considered**: 
>   1. Flat structure with all components in root package
>   2. Public packages exposing all internals
>   3. Internal packages with selective public API exposure
> - **Decision**: Use `internal/` packages with a small public API surface
> - **Rationale**: This approach provides implementation flexibility while preventing external code from depending on internal details that may change as we optimize the BitTorrent implementation
> - **Consequences**: Forces us to design clean interfaces between components and makes testing more systematic, but requires more careful API design for any public library usage

The **cmd/bittorrent/** directory contains the command-line interface that demonstrates how all components work together. This separation allows the core BitTorrent logic to be testable independently of CLI argument parsing and user interaction. The main.go file serves as the integration point where we instantiate all components and coordinate their interactions.

Each **internal/** package corresponds directly to one of our architectural components, with clear responsibility boundaries. The separation between `connection.go` and `manager.go` in the peer package reflects the difference between managing individual peer state machines versus coordinating across multiple peers. Similarly, the piece package separates the scheduling algorithms (`scheduler.go`) from the verification logic (`verifier.go`) because these concerns have different testing requirements and failure modes.

The **testdata/** directory provides sample torrent files that enable integration testing without requiring network access. These files should include various scenarios: single-file torrents, multi-file torrents, and edge cases like very small piece sizes or unusual announce URLs.

### High-Level Data Flow

Understanding how data flows through the BitTorrent client helps clarify the relationships between components and identifies the critical paths where errors can propagate. The data flow follows a clear pipeline from static torrent metadata through dynamic network coordination to final file reconstruction.

The complete data flow can be visualized as four distinct phases, each dominated by a different component but requiring coordination with others:

**Phase 1: Metadata Extraction** - The TorrentParser reads the `.torrent` file from disk and transforms it through several representations. Raw bytes become Bencode tokens, which become structured dictionaries, which finally become strongly-typed `MetaInfo` and `Info` structs. The critical output is the info hash, computed from the exact bencoded bytes of the info dictionary, which becomes the unique identifier used by all subsequent phases.

| Data Transform | Input Format | Output Format | Key Operations | Error Conditions |
|----------------|--------------|---------------|----------------|------------------|
| File → Bencode | Raw bytes from disk | Bencode token stream | `NewDecoder()`, `Decode()` | File not found, invalid Bencode syntax |
| Bencode → Structs | Decoded dictionaries/lists | `MetaInfo`, `Info` structs | Field extraction, type conversion | Missing required fields, wrong types |
| Info → Hash | Bencoded info dictionary | 20-byte SHA-1 hash | `CalculateInfoHash()` | None (SHA-1 is deterministic) |
| Pieces → Hashes | Concatenated piece hashes | Array of `[20]byte` | `GetPieceHashes()` | Invalid piece string length |

**Phase 2: Peer Discovery** - The TrackerClient uses the info hash and file metadata to construct announce requests that discover other clients in the swarm. The tracker responds with a compact list of peer IP addresses and ports, along with timing information that controls how frequently we should re-announce our presence. This phase transforms our isolated client into a participant in the distributed swarm.

| Data Transform | Input Format | Output Format | Key Operations | Coordination Required |
|----------------|--------------|---------------|----------------|----------------------|
| MetaInfo → Request | Torrent metadata | HTTP GET URL | `BuildAnnounceURL()` | None (pure transformation) |
| Request → Response | HTTP request | Bencode response | HTTP client, response parsing | Network I/O, tracker availability |
| Response → Peers | Compact peer data | `[]PeerInfo` structs | `ParseCompactPeers()` | None (parsing is deterministic) |
| Progress → Stats | Download state | Updated request params | Progress tracking | Coordination with PieceManager |

**Phase 3: Connection Establishment** - The PeerManager takes the list of potential peers and establishes BitTorrent protocol connections with a subset of them. Each connection involves a handshake exchange that verifies both clients are working with the same torrent (via info hash comparison) and establishes the message framing that will be used for all subsequent communication. This phase transforms network addresses into active protocol connections.

The peer connection process requires careful coordination between components because the PeerManager needs piece availability information from the PieceManager to make intelligent connection decisions, while the PieceManager needs peer capabilities from the PeerManager to schedule downloads effectively.

| Connection Phase | Data Input | Data Output | Component Interaction | Failure Handling |
|------------------|------------|-------------|----------------------|------------------|
| TCP Connect | IP address, port | TCP socket | Network layer only | Retry with exponential backoff |
| Handshake | Info hash, peer ID | Verified connection | Verify info hash matches | Close connection on mismatch |
| Bitfield Exchange | Local piece state | Peer piece availability | Share with PieceManager | Assume peer has no pieces |
| Message Loop | Protocol messages | Updated peer state | Continuous PeerManager coordination | Connection cleanup and retry |

**Phase 4: Content Transfer** - The PieceManager coordinates the actual download process by deciding which pieces to request from which peers, verifying the integrity of downloaded data, and assembling verified pieces into the final file. This phase involves complex scheduling decisions that balance piece rarity, peer performance, and download completion objectives.

The content transfer phase represents the most complex data flow because it involves continuous bidirectional communication between all components. The PieceManager must track piece availability across all connected peers, make scheduling decisions that optimize for swarm health, and coordinate with the PeerManager to send block requests to appropriate connections.

> **Critical Data Flow Insight:** The info hash serves as the system's primary key, flowing from TorrentParser through TrackerClient to PeerManager as the identifier that ensures all components are coordinating around the same file. Any corruption of this 20-byte value will cause complete system failure.

**Inter-Phase Coordination Points:**

1. **Parser → Tracker**: `MetaInfo` structure provides announce URL, info hash, and file size needed for tracker communication
2. **Tracker → Peer**: `[]PeerInfo` list drives connection establishment, but PeerManager may connect to subset based on configured limits
3. **Peer → Piece**: Bitfield messages from successful handshakes inform piece availability, enabling intelligent scheduling decisions
4. **Piece → Tracker**: Download progress (bytes downloaded, uploaded, remaining) flows back to tracker for announce updates
5. **Piece → Peer**: Block requests and piece completion notifications create continuous bidirectional communication

The data flow design ensures that each component can operate with appropriate autonomy while maintaining the coordination necessary for efficient downloads. Error conditions in any phase can be isolated and handled without bringing down the entire system, though certain failures (like tracker unavailability) may significantly impact performance.

> **Decision: Unidirectional Data Flow with Coordination Channels**
> - **Context**: BitTorrent requires complex coordination between components that operate at different time scales (parsing is one-time, tracker announces are periodic, peer connections are persistent, piece transfers are continuous)
> - **Options Considered**:
>   1. Shared mutable state with locks for coordination
>   2. Message-passing architecture with channels
>   3. Unidirectional flow with explicit coordination points
> - **Decision**: Unidirectional primary data flow with well-defined coordination interfaces
> - **Rationale**: This provides the predictability of functional programming (data flows one direction) while accommodating the necessary coordination through clean interfaces rather than shared mutable state
> - **Consequences**: Each component can be tested independently, but we must carefully design the coordination interfaces to avoid deadlocks or race conditions

### Implementation Guidance

Building the high-level architecture requires careful attention to Go-specific patterns and project organization. The following guidance provides the infrastructure and project structure needed to implement the component-based design effectively.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option | Recommended for Learning |
|-----------|---------------|-----------------|-------------------------|
| HTTP Client | `net/http` standard library | Custom HTTP client with connection pooling | `net/http` (focus on BitTorrent, not HTTP optimization) |
| Concurrency | `sync.Mutex` + goroutines | `sync.RWMutex` + worker pools | Start with `sync.Mutex`, add `RWMutex` for piece availability |
| Networking | `net.Conn` with manual framing | `bufio.Reader/Writer` wrappers | `bufio` for message parsing (handles partial reads) |
| Logging | `log` standard library | Structured logging (logrus/zap) | `log` initially, upgrade to structured when debugging |
| Testing | `testing` standard library | `testify` for assertions | `testing` (learn Go idioms first) |

**Complete Infrastructure Code:**

```go
// internal/client/client.go - Main coordination structure
package client

import (
    "context"
    "fmt"
    "sync"
    "time"

    "bittorrent-client/internal/torrent"
    "bittorrent-client/internal/tracker"
    "bittorrent-client/internal/peer"
    "bittorrent-client/internal/piece"
)

// Client coordinates all BitTorrent operations for a single torrent download
type Client struct {
    metaInfo        *torrent.MetaInfo
    trackerClient   *tracker.Client
    peerManager     *peer.Manager
    pieceManager    *piece.Manager
    
    // Coordination channels
    peerUpdates     chan []tracker.PeerInfo
    progressUpdates chan piece.ProgressUpdate
    
    // Synchronization
    mu              sync.RWMutex
    ctx             context.Context
    cancel          context.CancelFunc
}

// NewClient creates a BitTorrent client for the given torrent
func NewClient(metaInfo *torrent.MetaInfo) *Client {
    ctx, cancel := context.WithCancel(context.Background())
    
    client := &Client{
        metaInfo:        metaInfo,
        trackerClient:   tracker.NewClient(metaInfo.Announce, metaInfo.InfoHash),
        peerManager:     peer.NewManager(metaInfo.InfoHash),
        pieceManager:    piece.NewManager(metaInfo.Info),
        peerUpdates:     make(chan []tracker.PeerInfo, 10),
        progressUpdates: make(chan piece.ProgressUpdate, 100),
        ctx:             ctx,
        cancel:          cancel,
    }
    
    return client
}

// Download orchestrates the complete download process
func (c *Client) Download(outputPath string) error {
    // TODO 1: Start tracker announce loop in separate goroutine
    // TODO 2: Start peer connection management in separate goroutine  
    // TODO 3: Start piece download coordination in separate goroutine
    // TODO 4: Wait for download completion or context cancellation
    // TODO 5: Clean up all goroutines and connections
    panic("implement me")
}

// Close stops all operations and cleans up resources
func (c *Client) Close() error {
    c.cancel()
    // TODO: Close all connections and wait for goroutines to finish
    return nil
}
```

```go
// internal/client/coordination.go - Inter-component coordination helpers
package client

import (
    "context"
    "log"
    "time"

    "bittorrent-client/internal/tracker"
    "bittorrent-client/internal/piece"
)

// startTrackerAnnounces manages periodic communication with tracker
func (c *Client) startTrackerAnnounces() {
    // Initial announce to get peer list
    go func() {
        ticker := time.NewTicker(30 * time.Second) // Default interval
        defer ticker.Stop()
        
        for {
            select {
            case <-c.ctx.Done():
                return
            case <-ticker.C:
                // TODO: Get current progress from PieceManager
                // TODO: Build announce request with current stats
                // TODO: Send announce to tracker
                // TODO: Parse response and send peers to peerUpdates channel
                // TODO: Update ticker interval based on tracker response
            }
        }
    }()
}

// startPeerManagement handles peer connections and message processing
func (c *Client) startPeerManagement() {
    go func() {
        for {
            select {
            case <-c.ctx.Done():
                return
            case newPeers := <-c.peerUpdates:
                // TODO: Connect to new peers (up to MAX_PEERS limit)
                // TODO: Start message processing for each connection
                // TODO: Handle peer disconnections and errors
            }
        }
    }()
}

// startPieceCoordination manages piece requests and assembly
func (c *Client) startPieceCoordination() {
    go func() {
        for {
            select {
            case <-c.ctx.Done():
                return
            case progress := <-c.progressUpdates:
                // TODO: Update piece availability based on peer bitfields
                // TODO: Select next pieces using rarest-first algorithm
                // TODO: Send block requests to appropriate peers
                // TODO: Verify completed pieces and retry on hash failure
                // TODO: Assemble verified pieces into output file
            }
        }
    }()
}
```

**Core Component Skeleton:**

```go
// internal/torrent/parser.go - Core torrent parsing logic
package torrent

import (
    "crypto/sha1"
    "os"
    "time"
)

// MetaInfo represents parsed torrent file metadata
type MetaInfo struct {
    Announce     string
    CreationDate time.Time
    Comment      string
    InfoHash     [20]byte
    Info         Info
}

// Info represents the info dictionary from torrent file
type Info struct {
    Name        string
    Length      int64
    PieceLength int64
    Pieces      []byte
    Files       []File
}

// File represents individual file in multi-file torrent
type File struct {
    Length int64
    Path   []string
}

// ParseFromFile reads and parses a torrent file from disk
func ParseFromFile(filepath string) (*MetaInfo, error) {
    // TODO 1: Open file and read all bytes
    // TODO 2: Create bencode decoder from file data
    // TODO 3: Decode top-level dictionary
    // TODO 4: Extract announce URL and optional fields
    // TODO 5: Extract info dictionary and calculate hash
    // TODO 6: Parse file information from info dict
    panic("implement me")
}

// GetPieceHashes extracts individual piece hashes from concatenated pieces field
func (i *Info) GetPieceHashes() [][20]byte {
    // TODO 1: Verify pieces length is multiple of 20
    // TODO 2: Split pieces into 20-byte chunks
    // TODO 3: Convert each chunk to [20]byte array
    panic("implement me")
}

// CalculateInfoHash computes SHA1 hash of bencoded info dictionary
func CalculateInfoHash(bencoded []byte) [20]byte {
    hash := sha1.Sum(bencoded)
    return hash
}
```

**File Structure Setup Script:**

```bash
#!/bin/bash
# setup-project.sh - Creates the complete project structure

mkdir -p bittorrent-client/{cmd/bittorrent,internal/{torrent,tracker,peer,piece,client},pkg/bittorrent,testdata}

# Create go.mod
cat > bittorrent-client/go.mod << 'EOF'
module bittorrent-client

go 1.21

require (
    // Add dependencies as needed
)
EOF

# Create main.go entry point
cat > bittorrent-client/cmd/bittorrent/main.go << 'EOF'
package main

import (
    "flag"
    "fmt"
    "log"
    "os"

    "bittorrent-client/internal/torrent"
    "bittorrent-client/internal/client"
)

func main() {
    var (
        torrentFile = flag.String("torrent", "", "Path to .torrent file")
        output      = flag.String("output", "", "Output file path")
    )
    flag.Parse()

    if *torrentFile == "" || *output == "" {
        fmt.Fprintf(os.Stderr, "Usage: %s -torrent <file> -output <path>\n", os.Args[0])
        os.Exit(1)
    }

    // Parse torrent file
    metaInfo, err := torrent.ParseFromFile(*torrentFile)
    if err != nil {
        log.Fatalf("Failed to parse torrent: %v", err)
    }

    // Create and run client
    client := client.NewClient(metaInfo)
    defer client.Close()

    if err := client.Download(*output); err != nil {
        log.Fatalf("Download failed: %v", err)
    }

    fmt.Printf("Download completed: %s\n", *output)
}
EOF
```

**Milestone Checkpoint:**

After implementing the high-level architecture:

1. **Structure Verification**: Run `find bittorrent-client -name "*.go" | head -10` - you should see the organized package structure
2. **Compilation Check**: Run `go build ./cmd/bittorrent` - should compile without errors (even with panic statements)
3. **Import Verification**: All internal packages should import cleanly without circular dependencies
4. **Interface Validation**: Each component should have clear interface methods defined (even if not implemented)

**Common Architecture Pitfalls:**

⚠️ **Pitfall: Circular Import Dependencies**
Creating circular imports between packages (e.g., peer imports piece, piece imports peer) will prevent compilation. Solve this by defining shared interfaces in a common package or using dependency injection to break cycles.

⚠️ **Pitfall: Shared Mutable State Without Synchronization** 
Components accessing shared data structures without proper locking will cause race conditions. Use Go's race detector (`go run -race`) to detect these issues during development.

⚠️ **Pitfall: Blocking Operations in Coordination Channels**
Sending to channels without proper buffering or goroutine management can deadlock the entire system. Always use `select` statements with context cancellation for coordination code.

**Debugging Hints:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|-------------|-----------------|-----|
| Import cycle errors | Circular dependencies between packages | Check import statements in each package | Move shared interfaces to separate package |
| Deadlocks during shutdown | Goroutines waiting on channels that never close | Add logging to goroutine lifecycles | Implement proper context cancellation |
| Race condition panics | Shared data access without locks | Run with `-race` flag | Add mutex protection to shared state |
| Memory leaks | Goroutines not terminating | Use `go tool pprof` to check goroutine count | Ensure all goroutines respect context cancellation |


## Data Model

> **Milestone(s):** This section establishes the foundational data structures that will be built across all milestones: torrent metadata parsing (Milestone 1), tracker communication (Milestone 2), peer protocol (Milestone 3), and piece management (Milestone 4).

Building a BitTorrent client requires carefully designed data structures that capture the complex relationships between torrent files, peer connections, piece downloads, and protocol messages. Think of the data model as the **blueprint for a complex construction project** - just as architects must specify every beam, joint, and connection before builders can construct a skyscraper, we must define every struct, field, and relationship before implementing the BitTorrent protocol. Each data structure serves as a contract between different components of our system, ensuring they can communicate effectively and maintain consistent state.

The BitTorrent data model spans four distinct domains: **torrent metadata** (what files we're sharing and how they're structured), **peer state** (who we're connected to and what they can provide), **piece management** (tracking download progress and verification), and **protocol messages** (the language peers use to communicate). These domains interact in sophisticated ways - torrent metadata defines the pieces we need, peer state determines who can provide them, piece management coordinates the downloads, and protocol messages facilitate the actual data transfer.

![Core Data Structures](./diagrams/data-model.svg)

### Torrent Metadata Structures

The torrent metadata structures represent the parsed contents of a `.torrent` file, which serves as the **recipe for reconstructing a file or collection of files** from pieces distributed across the peer-to-peer network. Think of torrent metadata as a **detailed recipe book** - it tells you what ingredients you need (the pieces), how much of each ingredient (piece lengths and hashes), where to find suppliers (tracker URLs), and how to verify you've got the right ingredients (SHA-1 hashes for verification). This metadata is the authoritative source of truth for everything about a torrent.

> **Decision: Hierarchical Metadata Structure**
> - **Context**: Torrent files contain nested information - global metadata (announce URL, creation date) and file-specific information (name, length, pieces). We need to decide how to structure this data in memory.
> - **Options Considered**: Flat structure with all fields at one level, hierarchical structure matching the original Bencode format, separate structures for single vs multi-file torrents
> - **Decision**: Hierarchical structure with `MetaInfo` containing global metadata and an `Info` struct containing file-specific data
> - **Rationale**: Mirrors the original Bencode structure, makes info hash calculation straightforward (we hash just the `Info` portion), and provides clear separation between tracker-related metadata and content-related metadata
> - **Consequences**: Requires two-level access for file information but makes the protocol implementation cleaner and more maintainable

The `MetaInfo` structure represents the top-level torrent file contents and serves as the entry point for all torrent operations. Every BitTorrent client begins by parsing a torrent file into this structure, which then drives all subsequent operations from tracker communication to piece verification.

| Field | Type | Description |
|-------|------|-------------|
| `Announce` | `string` | Primary tracker URL where peers can be found for this torrent |
| `CreationDate` | `time.Time` | When this torrent was created (optional field from torrent file) |
| `Comment` | `string` | Human-readable description of the torrent contents (optional field) |
| `InfoHash` | `[20]byte` | SHA-1 hash of the bencoded info dictionary - the unique identifier for this torrent |
| `Info` | `Info` | The info dictionary containing file structure and piece information |

The `InfoHash` field deserves special attention as it serves as the **unique fingerprint** for the entire torrent. This 20-byte SHA-1 hash is computed from the exact bencoded bytes of the info dictionary and is used everywhere in the BitTorrent protocol - tracker announces include it to identify which torrent they're interested in, peer handshakes include it to verify both peers are talking about the same torrent, and DHT lookups use it as the key. The info hash must be calculated from the original bencoded bytes, not from a re-encoding of the parsed data structure, because even tiny differences in encoding would produce different hashes.

The `Info` structure contains the core file and piece information that defines what we're actually downloading. This structure is hashed to produce the info hash, so its contents are immutable once a torrent is created.

| Field | Type | Description |
|-------|------|-------------|
| `Name` | `string` | Suggested name for the file or top-level directory when saving |
| `Length` | `int64` | Total length in bytes for single-file torrents (zero for multi-file) |
| `PieceLength` | `int64` | Size in bytes of each piece (typically 256KB, 512KB, or 1MB) |
| `Pieces` | `[]byte` | Concatenated SHA-1 hashes of all pieces (20 bytes per piece) |
| `Files` | `[]File` | List of files in multi-file torrents (empty for single-file) |

The `Pieces` field requires careful handling as it contains concatenated 20-byte SHA-1 hashes rather than a slice of individual hashes. For a torrent with 1000 pieces, this field contains exactly 20,000 bytes. The `GetPieceHashes()` method extracts individual hashes from this concatenated format, providing a more convenient interface for piece verification operations.

> **Critical Design Insight**: The `Length` and `Files` fields are mutually exclusive - single-file torrents use `Length` and leave `Files` empty, while multi-file torrents populate `Files` and set `Length` to zero. This design follows the original BitTorrent specification and avoids redundant data storage.

For multi-file torrents, the `File` structure describes each individual file within the torrent. Multi-file torrents are conceptually **flat streams of bytes** that happen to be logically divided into files, similar to a ZIP archive that preserves directory structure.

| Field | Type | Description |
|-------|------|-------------|
| `Length` | `int64` | Size of this specific file in bytes |
| `Path` | `[]string` | Directory path components leading to this file (e.g., ["docs", "readme.txt"]) |

The `Path` field uses a slice of strings rather than a single path string to avoid platform-specific path separator issues. When writing files to disk, the client joins these components using the operating system's appropriate path separator.

### Peer and Connection State

Peer and connection state structures manage the complex relationships and state machines involved in BitTorrent peer communication. Think of peer state management as **managing relationships in a complex social network** - you need to track who you know (discovered peers), who you're actively talking to (connected peers), what each person is interested in (piece availability), and the current status of each relationship (choking and interested states). Each peer connection is essentially an independent state machine that must be carefully coordinated with piece management and download strategies.

> **Decision: Separation of Peer Discovery and Connection State**
> - **Context**: We need to distinguish between peers we've discovered (from tracker or DHT) and peers we're actively connected to. These have different data requirements and lifecycles.
> - **Options Considered**: Single structure for both discovered and connected peers, separate structures for each state, connection state as extension of peer info
> - **Decision**: Separate `PeerInfo` for discovered peers and `Connection` for active connections
> - **Rationale**: Discovered peers only need IP/port information, while active connections need message queues, state machines, and buffers. Separation keeps memory usage reasonable and makes state transitions clear.
> - **Consequences**: Requires mapping between peer info and connections, but provides cleaner abstraction and better resource management

The `PeerInfo` structure represents a peer discovered through tracker announces or DHT lookups. This is the minimal information needed to attempt a connection to a peer.

| Field | Type | Description |
|-------|------|-------------|
| `IP` | `[4]byte` | IPv4 address of the peer (big-endian byte order) |
| `Port` | `uint16` | TCP port number where peer accepts BitTorrent connections |

The compact peer format used by trackers packs this information into exactly 6 bytes per peer - 4 bytes for the IPv4 address followed by 2 bytes for the port in network byte order. The `ParseCompactPeers()` function converts this packed format into `PeerInfo` structures for easier handling.

Active peer connections require significantly more state tracking than simple peer discovery. Each connection maintains multiple pieces of state: the TCP connection itself, protocol state (handshake completion, supported extensions), choking state (four-flag state machine), piece availability (bitfield of what pieces the peer has), and request queues (outstanding block requests).

| Field | Type | Description |
|-------|------|-------------|
| `Conn` | `net.Conn` | Underlying TCP connection to the peer |
| `PeerID` | `[20]byte` | Unique identifier for this peer (received during handshake) |
| `Bitfield` | `[]byte` | Bitmap indicating which pieces this peer has available |
| `AmChoking` | `bool` | Whether we are choking this peer (refusing to upload to them) |
| `AmInterested` | `bool` | Whether we are interested in pieces this peer has |
| `PeerChoking` | `bool` | Whether this peer is choking us (refusing to upload to us) |
| `PeerInterested` | `bool` | Whether this peer is interested in pieces we have |
| `PendingRequests` | `map[int]BlockRequest` | Outstanding block requests we've sent to this peer |
| `RequestQueue` | `chan BlockRequest` | Queue of block requests waiting to be sent |

The four boolean flags (`AmChoking`, `AmInterested`, `PeerChoking`, `PeerInterested`) implement the core BitTorrent choking algorithm. This creates a state machine with 16 possible states, though only certain combinations are meaningful for data transfer.

| Our State | Peer State | Can Download | Can Upload | Typical Action |
|-----------|------------|--------------|------------|----------------|
| Interested, Unchoked | Any | Yes | Varies | Send piece requests |
| Interested, Choked | Any | No | Varies | Wait for unchoke |
| Not Interested | Any | No | Varies | Update interest when pieces change |
| Any | Interested, We Unchoke | Varies | Yes | Send requested pieces |
| Any | Interested, We Choke | Varies | No | Ignore piece requests |
| Any | Not Interested | Varies | No | No upload activity |

The `Bitfield` uses a compact bitmap representation where each bit indicates whether the peer has the corresponding piece. For a torrent with 1000 pieces, the bitfield requires 125 bytes (1000 bits rounded up to the nearest byte boundary). Bit manipulation operations check and set individual pieces within this bitmap.

> **Protocol Design Insight**: The choking mechanism serves as BitTorrent's incentive system - peers upload to those who upload to them, creating a tit-for-tat relationship that encourages participation rather than freeloading. New peers get optimistic unchokes to bootstrap the relationship.

### Piece and Block Management

Piece and block management structures coordinate the complex process of downloading, verifying, and assembling pieces into complete files. Think of piece management as **coordinating a massive jigsaw puzzle** where multiple people are working on different sections simultaneously, pieces must be verified before acceptance, and you need to track which pieces are rarest to prioritize them appropriately. The challenge is coordinating concurrent downloads while avoiding duplicate work and ensuring data integrity.

Pieces and blocks represent different granularities of data transfer. A **piece** is the unit of verification (typically 256KB to 1MB) that can be independently hash-checked, while a **block** is the unit of network transfer (typically 16KB) that fits in a single protocol message. Each piece contains multiple blocks, and blocks must be requested individually from peers.

> **Decision: Separate Piece and Block Granularity**
> - **Context**: BitTorrent uses different sizes for verification (pieces) and network transfer (blocks). We need to decide how to model this relationship.
> - **Options Considered**: Single granularity using pieces only, single granularity using blocks only, separate piece and block concepts with mapping between them
> - **Decision**: Separate piece and block concepts with explicit mapping and state tracking
> - **Rationale**: Matches protocol design where pieces are verified independently but transferred as smaller blocks. Allows request pipelining and partial piece recovery if connections fail.
> - **Consequences**: More complex state management but better network efficiency and failure handling

The `PieceState` structure tracks the download progress and verification status of a single piece. Each piece progresses through several states during download: not started, in progress (with some blocks downloaded), complete but unverified, verified and available, or failed verification.

| Field | Type | Description |
|-------|------|-------------|
| `Index` | `int` | Zero-based index of this piece in the torrent |
| `Length` | `int` | Total size of this piece in bytes (may be smaller for last piece) |
| `Hash` | `[20]byte` | Expected SHA-1 hash for verification of complete piece |
| `Blocks` | `map[int]*Block` | Map of block offset to block state within this piece |
| `State` | `PieceStateEnum` | Current state: NotStarted, InProgress, Complete, Verified, Failed |
| `Priority` | `int` | Download priority (higher numbers downloaded first) |
| `Availability` | `int` | Number of connected peers that have this piece |

The `Blocks` map uses byte offsets as keys, allowing efficient lookup of specific blocks within the piece. Block offsets are typically multiples of `BLOCK_SIZE` (16384 bytes), with the final block in each piece potentially being smaller.

Individual blocks represent the smallest unit of data transfer and request management. Each block tracks its own download state and can be independently requested from different peers.

| Field | Type | Description |
|-------|------|-------------|
| `PieceIndex` | `int` | Which piece this block belongs to |
| `Offset` | `int` | Byte offset within the piece where this block starts |
| `Length` | `int` | Size of this block in bytes (typically 16KB, smaller for last block) |
| `Data` | `[]byte` | The actual downloaded data (nil until downloaded) |
| `State` | `BlockStateEnum` | Current state: NotRequested, Requested, Downloaded |
| `RequestedFrom` | `string` | Peer ID that we requested this block from (for timeout handling) |
| `RequestTime` | `time.Time` | When we sent the request (for timeout detection) |

Block state management requires careful coordination to avoid duplicate requests while ensuring failed requests are retried. The `RequestedFrom` and `RequestTime` fields enable timeout detection - if a block request isn't fulfilled within a reasonable time, it can be re-requested from a different peer.

| Block State | Next States | Trigger | Action |
|-------------|-------------|---------|--------|
| NotRequested | Requested | Peer available and not choked | Send request message |
| Requested | Downloaded | Piece message received | Store data, verify length |
| Requested | NotRequested | Request timeout | Mark for retry with different peer |
| Downloaded | NotRequested | Piece verification failed | Mark all blocks for re-download |

The piece selection and scheduling system coordinates downloads across multiple peers to optimize performance. The `PieceManager` maintains global state about download progress and peer availability.

| Field | Type | Description |
|-------|------|-------------|
| `Pieces` | `map[int]*PieceState` | State of all pieces in the torrent |
| `IncompletePieces` | `[]int` | Pieces that are partially downloaded (for endgame mode) |
| `RequestQueue` | `chan BlockRequest` | Global queue of blocks that need downloading |
| `CompletedPieces` | `map[int]bool` | Quick lookup for pieces that are verified and complete |
| `TotalPieces` | `int` | Total number of pieces in this torrent |
| `TotalLength` | `int64` | Total size of all data in bytes |

> **Scheduling Strategy**: The piece selection algorithm implements **rarest-first scheduling** - pieces held by fewer peers are downloaded first to improve overall swarm health. This prevents situations where common pieces are over-replicated while rare pieces risk being lost if their few holders disconnect.

### Protocol Message Formats

Protocol message structures define the language that BitTorrent peers use to communicate. Think of these messages as **vocabulary in a formal diplomatic protocol** - each message type has a specific format, purpose, and expected response, and both sides must follow the protocol exactly or communication breaks down. The wire protocol uses binary encoding for efficiency, with each message having a specific structure that must be parsed and generated correctly.

The BitTorrent peer wire protocol uses a simple message framing format: each message starts with a 4-byte length prefix (in big-endian byte order), followed by a 1-byte message ID, followed by the message payload. The handshake message is special and doesn't follow this framing format.

> **Decision: Structured Message Types with Union-Style Interface**
> - **Context**: BitTorrent messages have different payloads but share common framing. We need to decide how to represent this variety in a type-safe way.
> - **Options Considered**: Single message struct with optional fields, interface with concrete types for each message, raw byte handling with manual parsing
> - **Decision**: Interface-based design with concrete types for each message category
> - **Rationale**: Provides type safety, makes message handling explicit, and allows for easy extension with new message types
> - **Consequences**: Requires more types but provides better compile-time checking and clearer message handling logic

The handshake message establishes a BitTorrent connection and verifies both peers are talking about the same torrent. This message has a fixed 68-byte format that differs from all other protocol messages.

| Field | Type | Size | Description |
|-------|------|------|-------------|
| `ProtocolLength` | `uint8` | 1 byte | Length of protocol string (always 19 for BitTorrent) |
| `Protocol` | `string` | 19 bytes | Protocol identifier: "BitTorrent protocol" |
| `Reserved` | `[8]byte` | 8 bytes | Reserved for extension flags (usually all zeros) |
| `InfoHash` | `[20]byte` | 20 bytes | SHA-1 hash identifying the torrent |
| `PeerID` | `[20]byte` | 20 bytes | Unique identifier for the sending peer |

After the handshake, all messages follow the standard framing format. The message ID determines how to interpret the payload that follows.

| Message ID | Name | Payload Size | Purpose |
|------------|------|--------------|---------|
| -1 | Keep-alive | 0 bytes | Maintain connection when no other messages needed |
| 0 | Choke | 0 bytes | Sender will not fulfill requests from receiver |
| 1 | Unchoke | 0 bytes | Sender will fulfill requests from receiver |
| 2 | Interested | 0 bytes | Sender wants to download from receiver |
| 3 | Not interested | 0 bytes | Sender doesn't want anything from receiver |
| 4 | Have | 4 bytes | Announces sender has completed a piece |
| 5 | Bitfield | Variable | Announces which pieces sender has (sent after handshake) |
| 6 | Request | 12 bytes | Request a block of data from a piece |
| 7 | Piece | Variable | Deliver requested block data |
| 8 | Cancel | 12 bytes | Cancel a previously sent request |

The `Have` message announces when a peer completes downloading a piece. This allows other peers to update their availability calculations for piece selection.

| Field | Type | Description |
|-------|------|-------------|
| `PieceIndex` | `uint32` | Zero-based index of the completed piece |

The `Bitfield` message is sent immediately after handshake completion to announce which pieces the peer already has. This message uses the same compact bitmap format as the `Bitfield` field in connection state.

| Field | Type | Description |
|-------|------|-------------|
| `Bitfield` | `[]byte` | Bitmap of available pieces (1 bit per piece) |

The `Request` message asks a peer to send a specific block of data. BitTorrent clients typically pipeline multiple requests to keep the network connection busy.

| Field | Type | Description |
|-------|------|-------------|
| `PieceIndex` | `uint32` | Which piece contains the desired block |
| `BlockOffset` | `uint32` | Byte offset within the piece where block starts |
| `BlockLength` | `uint32` | Size of the requested block (typically 16384 bytes) |

The `Piece` message delivers requested block data. This is the core message that actually transfers file content between peers.

| Field | Type | Description |
|-------|------|-------------|
| `PieceIndex` | `uint32` | Which piece this block belongs to |
| `BlockOffset` | `uint32` | Byte offset within the piece |
| `BlockData` | `[]byte` | The actual file data for this block |

> **Protocol Efficiency Note**: Block requests are typically pipelined - clients send multiple request messages before receiving the first piece response. This keeps the network connection saturated and improves transfer speeds, but requires careful tracking of outstanding requests.

The `Cancel` message allows clients to cancel previously sent requests, typically used in endgame mode when the same block is requested from multiple peers to avoid slow connections.

| Field | Type | Description |
|-------|------|-------------|
| `PieceIndex` | `uint32` | Which piece contained the block to cancel |
| `BlockOffset` | `uint32` | Byte offset of the block to cancel |
| `BlockLength` | `uint32` | Size of the block to cancel |

### Common Pitfalls

⚠️ **Pitfall: Info Hash Calculation from Re-encoded Data**
Many developers make the mistake of calculating the info hash by re-encoding the parsed `Info` structure back to Bencode format. This fails because Bencode encoding is not canonical - the same data can be encoded in multiple ways (dictionary key ordering, integer representation). The info hash must be calculated from the original bytes of the info dictionary as it appeared in the torrent file. Store these original bytes during parsing or calculate the hash before parsing the info dictionary contents.

⚠️ **Pitfall: Bitfield Bit Ordering Confusion**
The bitfield uses big-endian bit ordering within each byte, which can be confusing. The first piece (index 0) is represented by the most significant bit of the first byte. When checking if piece N is available, use `bitfield[N/8] & (0x80 >> (N%8))` rather than assuming little-endian bit ordering. Test with a known bitfield pattern to verify your bit manipulation is correct.

⚠️ **Pitfall: Block Boundary Calculations**
The last piece in a torrent is typically smaller than the standard piece length, and the last block in any piece may be smaller than the standard block size. Always calculate block and piece sizes dynamically rather than assuming they're all the same size. Use `min(BLOCK_SIZE, remaining_piece_bytes)` when calculating block lengths to handle these boundary conditions correctly.

⚠️ **Pitfall: Concurrent Access to Peer State**
Peer connections run in separate goroutines and can modify connection state concurrently with the main piece management logic. Protect all shared state (bitfields, pending requests, choking flags) with appropriate synchronization. Use separate mutexes for different aspects of peer state to avoid lock contention while ensuring consistency.

⚠️ **Pitfall: Message Length Validation**
Always validate message lengths before parsing message payloads. A malicious or buggy peer could send a message claiming to be much larger than it actually is, leading to buffer overruns or excessive memory allocation. Check that declared message lengths match expected sizes for fixed-format messages, and impose reasonable limits on variable-length messages like bitfields and piece data.

### Implementation Guidance

The data model implementation requires careful attention to memory layout, serialization formats, and concurrent access patterns. Go's strong typing system helps prevent many common errors, but BitTorrent's binary protocols require explicit handling of byte ordering and bit manipulation.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Serialization | Manual byte manipulation with encoding/binary | Protocol Buffers with custom wire format |
| Concurrency | sync.Mutex for each data structure | sync.RWMutex with fine-grained locking |
| Memory Management | Standard garbage collection | sync.Pool for frequent allocations |
| Bit Operations | Standard bit shifting and masking | Third-party bitset library |

#### Recommended File Structure

The data model spans multiple packages to maintain clean separation of concerns and enable independent testing of each component.

```
internal/
  torrent/
    metadata.go          ← MetaInfo, Info, File structures
    metadata_test.go     ← Torrent parsing tests
  peer/
    connection.go        ← Connection state and peer management
    connection_test.go   ← Peer state machine tests
  piece/
    manager.go          ← PieceState, Block, and scheduling logic
    manager_test.go     ← Piece selection and verification tests
  protocol/
    messages.go         ← All protocol message types
    messages_test.go    ← Message parsing and generation tests
```

#### Complete Infrastructure Code

Here's the complete message framing infrastructure that handles the binary protocol details:

```go
package protocol

import (
    "encoding/binary"
    "fmt"
    "io"
)

const (
    HANDSHAKE_LENGTH = 68
    PROTOCOL_STRING = "BitTorrent protocol"
)

// MessageFramer handles the binary framing of BitTorrent protocol messages
type MessageFramer struct {
    conn io.ReadWriter
}

func NewMessageFramer(conn io.ReadWriter) *MessageFramer {
    return &MessageFramer{conn: conn}
}

// ReadMessage reads a complete framed message from the connection
func (mf *MessageFramer) ReadMessage() (MessageID, []byte, error) {
    // Read 4-byte length prefix
    var length uint32
    err := binary.Read(mf.conn, binary.BigEndian, &length)
    if err != nil {
        return 0, nil, fmt.Errorf("failed to read message length: %w", err)
    }
    
    // Handle keep-alive messages (length = 0)
    if length == 0 {
        return KEEPALIVE, nil, nil
    }
    
    // Read message ID (1 byte)
    var messageID uint8
    err = binary.Read(mf.conn, binary.BigEndian, &messageID)
    if err != nil {
        return 0, nil, fmt.Errorf("failed to read message ID: %w", err)
    }
    
    // Read remaining payload
    payload := make([]byte, length-1) // -1 because we already read message ID
    _, err = io.ReadFull(mf.conn, payload)
    if err != nil {
        return 0, nil, fmt.Errorf("failed to read message payload: %w", err)
    }
    
    return MessageID(messageID), payload, nil
}

// WriteMessage writes a framed message to the connection
func (mf *MessageFramer) WriteMessage(messageID MessageID, payload []byte) error {
    // Calculate total length (message ID + payload)
    length := uint32(1 + len(payload))
    
    // Write length prefix
    err := binary.Write(mf.conn, binary.BigEndian, length)
    if err != nil {
        return fmt.Errorf("failed to write message length: %w", err)
    }
    
    // Write message ID
    err = binary.Write(mf.conn, binary.BigEndian, uint8(messageID))
    if err != nil {
        return fmt.Errorf("failed to write message ID: %w", err)
    }
    
    // Write payload
    _, err = mf.conn.Write(payload)
    if err != nil {
        return fmt.Errorf("failed to write message payload: %w", err)
    }
    
    return nil
}
```

Complete bitfield manipulation utilities:

```go
package peer

import "math"

// BitfieldOps provides utilities for manipulating piece availability bitfields
type BitfieldOps struct {
    bitfield []byte
    numPieces int
}

func NewBitfield(numPieces int) *BitfieldOps {
    byteCount := int(math.Ceil(float64(numPieces) / 8.0))
    return &BitfieldOps{
        bitfield: make([]byte, byteCount),
        numPieces: numPieces,
    }
}

func (bf *BitfieldOps) SetPiece(pieceIndex int) {
    if pieceIndex >= bf.numPieces {
        return
    }
    byteIndex := pieceIndex / 8
    bitOffset := pieceIndex % 8
    bf.bitfield[byteIndex] |= (0x80 >> bitOffset)
}

func (bf *BitfieldOps) HasPiece(pieceIndex int) bool {
    if pieceIndex >= bf.numPieces {
        return false
    }
    byteIndex := pieceIndex / 8
    bitOffset := pieceIndex % 8
    return (bf.bitfield[byteIndex] & (0x80 >> bitOffset)) != 0
}

func (bf *BitfieldOps) CountPieces() int {
    count := 0
    for i := 0; i < bf.numPieces; i++ {
        if bf.HasPiece(i) {
            count++
        }
    }
    return count
}

func (bf *BitfieldOps) ToBytes() []byte {
    result := make([]byte, len(bf.bitfield))
    copy(result, bf.bitfield)
    return result
}
```

#### Core Logic Skeletons

Here are the key data structure definitions with TODO comments for the core logic that learners should implement:

```go
package torrent

import (
    "crypto/sha1"
    "time"
)

// MetaInfo represents the parsed contents of a .torrent file
type MetaInfo struct {
    Announce     string
    CreationDate time.Time
    Comment      string
    InfoHash     [20]byte
    Info         Info
}

// CalculateInfoHash computes the SHA-1 hash of the bencoded info dictionary
func (mi *MetaInfo) CalculateInfoHash(bencodedInfoDict []byte) [20]byte {
    // TODO 1: Use crypto/sha1 to create a new hash instance
    // TODO 2: Write the exact bencoded bytes to the hasher
    // TODO 3: Call Sum() to get the final 20-byte hash
    // TODO 4: Convert the slice to a [20]byte array and return
    // Hint: This must use the original bencoded bytes, not re-encoded data
}

// Info represents the info dictionary from a torrent file
type Info struct {
    Name        string
    Length      int64
    PieceLength int64
    Pieces      []byte
    Files       []File
}

// GetPieceHashes extracts individual 20-byte piece hashes from concatenated pieces field
func (info *Info) GetPieceHashes() [][20]byte {
    // TODO 1: Calculate how many pieces there are (len(Pieces) / 20)
    // TODO 2: Create a slice to hold the individual hashes
    // TODO 3: Loop through the Pieces field in 20-byte chunks
    // TODO 4: Copy each 20-byte chunk into a [20]byte array
    // TODO 5: Append each hash array to the result slice
    // Hint: Use copy() to convert []byte to [20]byte
}

// CalculateTotalLength returns total bytes across all files (for multi-file torrents)
func (info *Info) CalculateTotalLength() int64 {
    // TODO 1: If this is a single-file torrent (Length > 0), return Length
    // TODO 2: Otherwise, iterate through Files slice and sum all file lengths
    // TODO 3: Return the total calculated length
}
```

```go
package peer

import (
    "net"
    "sync"
    "time"
)

// Connection represents an active connection to a BitTorrent peer
type Connection struct {
    conn            net.Conn
    peerID          [20]byte
    bitfield        *BitfieldOps
    amChoking       bool
    amInterested    bool
    peerChoking     bool
    peerInterested  bool
    pendingRequests map[string]*BlockRequest
    mutex           sync.RWMutex
}

// UpdateInterest checks if we should be interested in this peer based on their pieces
func (c *Connection) UpdateInterest(neededPieces []int) {
    // TODO 1: Lock the connection for reading
    // TODO 2: Check if peer has any pieces we need (iterate through neededPieces)
    // TODO 3: If peer has needed pieces and we're not interested, send INTERESTED message
    // TODO 4: If peer has no needed pieces and we are interested, send NOT_INTERESTED
    // TODO 5: Update the amInterested flag accordingly
    // TODO 6: Unlock the connection
    // Hint: Use bitfield.HasPiece() to check peer's available pieces
}

// CanDownload returns true if we can request blocks from this peer
func (c *Connection) CanDownload() bool {
    // TODO 1: Lock for reading
    // TODO 2: Check if we're interested AND peer is not choking us
    // TODO 3: Return the result
    // Hint: Both conditions must be true for downloads to proceed
}
```

```go
package piece

import (
    "crypto/sha1"
    "sync"
)

// Manager coordinates piece downloads and verification across multiple peers
type Manager struct {
    pieces          map[int]*PieceState
    completedPieces map[int]bool
    inProgress      map[int]bool
    totalPieces     int
    pieceLength     int64
    pieceHashes     [][20]byte
    mutex           sync.RWMutex
}

// SelectNextPiece implements rarest-first piece selection strategy
func (pm *Manager) SelectNextPiece(peerBitfield *BitfieldOps) int {
    // TODO 1: Lock for reading to examine piece availability
    // TODO 2: Build a list of pieces we need that this peer has
    // TODO 3: Count how many peers have each candidate piece (availability)
    // TODO 4: Sort candidates by availability (rarest first)
    // TODO 5: Return the rarest piece index, or -1 if no pieces needed
    // Hint: Lower availability count = higher priority for download
}

// VerifyPiece checks if a completed piece matches its expected hash
func (pm *Manager) VerifyPiece(pieceIndex int, data []byte) bool {
    // TODO 1: Get the expected hash for this piece from pieceHashes
    // TODO 2: Calculate SHA-1 hash of the provided data
    // TODO 3: Compare calculated hash with expected hash
    // TODO 4: If hashes match, mark piece as completed and return true
    // TODO 5: If hashes don't match, mark piece for re-download and return false
    // Hint: Use bytes.Equal() to compare hash arrays
}
```

#### Language-Specific Hints

- Use `encoding/binary.BigEndian` for all network byte order conversions in protocol messages
- The `[20]byte` array type is not directly assignable from `[]byte` slices - use `copy()` for conversion
- Implement `String()` methods on your message types for better debugging output
- Use `sync.RWMutex` for data structures that are read frequently but written infrequently
- Consider using `context.Context` for cancellation in long-running download operations
- The `net.Conn` interface provides both reading and writing, but you may need to wrap it with `bufio.Reader` for efficient parsing

#### Milestone Checkpoints

After implementing the data model structures, verify the following behavior:

**Milestone 1 Checkpoint**: Torrent metadata parsing works correctly
- Run `go test ./internal/torrent/...` - all bencode parsing tests should pass
- Parse a real .torrent file and verify the info hash matches other BitTorrent clients
- Check that `GetPieceHashes()` returns the correct number of 20-byte hashes
- Verify multi-file torrents correctly populate the `Files` slice

**Milestone 2 Checkpoint**: Peer state management functions properly
- Run `go test ./internal/peer/...` - connection state tests should pass
- Create a bitfield, set some pieces, and verify `HasPiece()` returns correct results
- Test the four-flag state machine with various choking/interested combinations
- Verify that `CanDownload()` returns true only when interested and unchoked

**Milestone 3 Checkpoint**: Protocol message handling works
- Run `go test ./internal/protocol/...` - message parsing tests should pass
- Create each message type and verify serialization round-trips correctly
- Test message framing with various payload sizes including zero-length messages
- Verify handshake message format matches the 68-byte specification exactly

**Milestone 4 Checkpoint**: Piece management coordinates correctly
- Run `go test ./internal/piece/...` - piece selection and verification tests should pass
- Test piece selection with various peer bitfield configurations
- Verify piece verification correctly accepts good hashes and rejects bad ones
- Check that completed pieces are properly tracked and not re-downloaded


## Torrent File Parsing (Milestone 1)

> **Milestone(s):** This section corresponds to Milestone 1 - Torrent File Parsing, implementing Bencode decoding and torrent metadata extraction with proper info hash calculation.

Torrent file parsing forms the foundation of any BitTorrent client. Before we can connect to trackers, discover peers, or download pieces, we must first understand what files we're trying to download and how they're organized. This milestone involves decoding the Bencode format used by BitTorrent and extracting critical metadata that drives the entire download process.

The parsing process involves three fundamental challenges. First, we must decode the Bencode binary encoding format that stores all torrent metadata in a compact, language-agnostic form. Second, we need to extract specific fields from the decoded data structure to understand file organization, tracker locations, and piece verification data. Third, we must compute the info hash - a cryptographic identifier that uniquely identifies the torrent across the entire BitTorrent network.

![Core Data Structures](./diagrams/data-model.svg)

Understanding torrent file structure is essential because every subsequent operation in our BitTorrent client depends on this metadata. The announce URL tells us where to find other peers, the piece length and piece hashes enable us to verify downloaded data integrity, and the info hash serves as our torrent's unique fingerprint for tracker communication and peer handshakes.

### Bencode: Binary JSON Analogy

Think of Bencode as JSON's binary cousin designed specifically for peer-to-peer applications. Just like JSON provides a structured way to represent nested data (objects, arrays, strings, numbers), Bencode offers the same organizational capabilities but with two key differences that make it perfect for BitTorrent.

First, Bencode handles binary data natively. While JSON struggles with binary content and requires encoding schemes like Base64, Bencode treats byte strings as first-class citizens. This matters enormously in BitTorrent because piece hashes are raw 20-byte SHA-1 digests, peer IDs contain arbitrary binary data, and the protocol frequently exchanges binary payloads. Imagine trying to represent a piece hash like `\x12\x34\x56...` in JSON - you'd need encoding layers that add complexity and overhead.

Second, Bencode produces completely deterministic output. Unlike JSON, where the same data structure can be serialized in multiple valid ways (different key ordering, varied whitespace), Bencode has exactly one canonical representation for any given data structure. This determinism is crucial for computing info hashes, since the SHA-1 digest must be calculated from the exact bytes of the serialized info dictionary.

The mental model to internalize is this: **Bencode is like JSON, but designed for systems that need binary data support and cryptographic hashing**. Just as you might parse JSON to extract a web API response, we parse Bencode to extract torrent metadata. The same concepts apply - nested dictionaries become maps, arrays become lists, and values can be strings, numbers, or nested structures.

Bencode supports exactly four data types, mirroring the essential types found in most structured data formats:

| Bencode Type | JSON Equivalent | Format | Example | Use in Torrents |
|-------------|----------------|---------|---------|-----------------|
| Byte String | String (but binary-safe) | `<length>:<data>` | `4:spam` | File names, piece hashes, tracker URLs |
| Integer | Number | `i<value>e` | `i42e` | File lengths, piece length, creation date |
| List | Array | `l<items>e` | `l4:spam3:egge` | File lists, announce list |
| Dictionary | Object | `d<key><value>e` | `d3:cow3:moo4:spam4:eggse` | Root structure, info dict |

The elegance of Bencode lies in its simplicity. There are no null values, no floating-point numbers, no boolean types - just the minimal set of types needed to represent structured data with binary content. This simplicity makes the parser straightforward to implement while ensuring compatibility across different programming languages and platforms.

### Bencode Decoder Design

The Bencode decoder uses a recursive descent parsing strategy, where each data type has a dedicated parsing method that handles its specific format. This approach mirrors how you might parse a mathematical expression by recognizing operators and recursively evaluating sub-expressions.

The core insight is that Bencode is **self-delimiting** - each encoded value contains enough information to determine where it ends. Strings begin with their length, integers are wrapped with `i` and `e` markers, and container types (lists and dictionaries) use `l`/`d` and `e` delimiters. This means our parser can read a single character to determine the upcoming data type, then delegate to the appropriate parsing method.

![Bencode Parsing State Machine](./diagrams/bencode-parsing.svg)

The parsing state machine operates on a simple principle: examine the next character to determine the value type, then consume exactly the bytes needed for that value. For strings, we read digits until we hit a colon, parse the length, then read exactly that many data bytes. For integers, we consume characters between `i` and `e` markers. For containers, we recursively parse items until we encounter the closing `e`.

> **Decision: Recursive Descent Parser Architecture**
> - **Context**: Need to parse nested Bencode structures containing arbitrary nesting of lists and dictionaries
> - **Options Considered**: Recursive descent parser, table-driven parser, hand-written state machine
> - **Decision**: Recursive descent with method-per-type design
> - **Rationale**: Bencode's simple grammar maps naturally to recursive methods, making the code readable and maintainable. The nesting depth is typically shallow in torrent files, so stack overflow isn't a concern.
> - **Consequences**: Simple implementation and good error reporting, but potential stack overflow on pathological inputs

The `Decoder` struct maintains parsing state through a buffered reader that provides lookahead capabilities for type discrimination. The core `Decode` method acts as a dispatcher, examining the next byte to determine which specialized parsing method to invoke.

| Method | Purpose | Input Format | Returns | Error Cases |
|--------|---------|-------------|---------|-------------|
| `Decode()` | Main entry point, dispatches to type-specific parsers | Any valid Bencode | `interface{}`, `error` | Invalid type prefix, EOF |
| `decodeString()` | Parse length-prefixed byte strings | `<digits>:<data>` | `[]byte`, `error` | Invalid length, insufficient data |
| `decodeInteger()` | Parse signed integers | `i<digits>e` | `int64`, `error` | Invalid digits, missing terminator |
| `decodeList()` | Parse heterogeneous arrays | `l<items>e` | `[]interface{}`, `error` | Malformed items, missing terminator |
| `decodeDictionary()` | Parse string-keyed maps | `d<pairs>e` | `map[string]interface{}`, `error` | Non-string keys, unsorted keys |

The string parsing algorithm demonstrates the careful byte handling required for binary data:

1. Read decimal digits until encountering a colon character
2. Parse the accumulated digits as the string length (validate non-negative)
3. Allocate a byte slice of exactly that length
4. Read exactly that many bytes from the input stream (not text lines)
5. Return the raw byte slice without any encoding interpretation

Integer parsing requires careful handling of the `i` and `e` delimiters while supporting negative values:

1. Verify the current byte is `i` and advance past it
2. Read characters until encountering `e`, accumulating digits and optional negative sign
3. Validate the accumulated string represents a valid integer (no leading zeros except for zero itself)
4. Parse the string as a signed 64-bit integer
5. Consume the closing `e` delimiter

List parsing showcases the recursive nature of the algorithm:

1. Verify the current byte is `l` and advance past it
2. Initialize an empty slice to accumulate parsed items
3. While the next byte is not `e`, recursively call `Decode()` to parse the next item
4. Append each successfully parsed item to the accumulating slice
5. Consume the closing `e` delimiter and return the completed slice

Dictionary parsing adds the complexity of maintaining sorted key order and ensuring all keys are strings:

1. Verify the current byte is `d` and advance past it
2. Initialize an empty map to accumulate key-value pairs
3. While the next byte is not `e`, parse a key-value pair:
   - Parse the key (must be a string, return error if not)
   - Parse the value (recursive call to `Decode()`)
   - Verify key ordering (current key must be lexicographically greater than previous)
   - Insert the pair into the accumulating map
4. Consume the closing `e` delimiter and return the completed map

The key ordering validation is crucial because Bencode requires dictionary keys to appear in sorted order. This ensures deterministic serialization, which is essential for computing consistent info hashes. If we encounter keys out of order, we must reject the input as malformed.

### Torrent Metadata Extraction

Once we've decoded the Bencode structure into native data types, we need to extract the specific fields that define a torrent's characteristics. Think of this process like extracting structured information from a parsed JSON API response - we know the expected schema and need to navigate the nested data to find the values we need.

The root level of every torrent file contains several key fields that serve different purposes in the BitTorrent ecosystem:

| Field | Type | Required | Purpose | Example |
|-------|------|----------|---------|---------|
| `announce` | String | Yes | Primary tracker URL for peer discovery | `http://tracker.example.com:8080/announce` |
| `announce-list` | List of Lists | No | Backup tracker URLs organized by tiers | `[["http://primary.com"], ["http://backup1.com", "http://backup2.com"]]` |
| `info` | Dictionary | Yes | File and piece information (used for info hash) | Contains `name`, `length`, `pieces`, etc. |
| `creation date` | Integer | No | Unix timestamp when torrent was created | `1609459200` |
| `comment` | String | No | Human-readable description | `"Ubuntu 20.04.1 Desktop amd64"` |
| `created by` | String | No | Software that created the torrent | `"mktorrent 1.1"` |

The `info` dictionary contains the core file and piece information that defines what we're downloading:

| Info Field | Type | Required | Purpose | Notes |
|------------|------|----------|---------|-------|
| `name` | String | Yes | Suggested filename or directory name | UTF-8 encoded, used for local file creation |
| `piece length` | Integer | Yes | Size of each piece in bytes (except possibly the last) | Typically powers of 2, between 16KB and 8MB |
| `pieces` | String | Yes | Concatenated SHA-1 hashes of all pieces | Length must be multiple of 20 bytes |
| `length` | Integer | Single-file | Total file size in bytes | Present only for single-file torrents |
| `files` | List | Multi-file | List of file descriptors for multi-file torrents | Each contains `length` and `path` |

The metadata extraction algorithm operates by navigating the decoded Bencode structure and type-checking each expected field:

1. **Validate root structure**: Ensure the decoded result is a dictionary containing required top-level keys
2. **Extract announce URL**: Retrieve the `announce` field as a string and validate it's a well-formed HTTP or UDP URL
3. **Extract info dictionary**: Retrieve the `info` field as a dictionary for further processing
4. **Extract file information**: Determine if this is a single-file or multi-file torrent and extract appropriate file metadata
5. **Extract piece information**: Retrieve piece length and piece hashes for download verification
6. **Extract optional fields**: Safely extract creation date, comment, and other non-critical metadata

> **Decision: Single vs Multi-File Torrent Handling**
> - **Context**: Torrents can contain either a single file (with `length` in info) or multiple files (with `files` array)
> - **Options Considered**: Separate parsing paths for each type, normalize to common representation, support both natively
> - **Decision**: Normalize single-file torrents to the multi-file representation internally
> - **Rationale**: Simplifies downstream code by providing a uniform interface, while preserving all original metadata
> - **Consequences**: Slight overhead for single-file torrents, but dramatically simplified piece and download management logic

The piece hash extraction deserves special attention because it's critical for content verification. The `pieces` field contains concatenated 20-byte SHA-1 hashes, so we must split this byte string into individual hash values:

1. Validate that the pieces string length is a multiple of 20 bytes
2. Calculate the number of pieces as `len(pieces) / 20`
3. Iterate through the string in 20-byte chunks, extracting each piece hash
4. Store the hashes in a slice for indexed access during downloads

For multi-file torrents, we need to process the `files` array to understand the directory structure:

1. Iterate through each file descriptor in the files array
2. Extract the file length as an integer
3. Extract the path as a list of strings representing the directory hierarchy
4. Validate that path components don't contain directory traversal sequences like `..`
5. Accumulate total torrent size by summing individual file lengths

The extraction process must handle missing optional fields gracefully while strictly validating required fields. For optional fields like creation date, we check for their presence before attempting type conversion. For required fields like announce URL, we return an error if the field is missing or has the wrong type.

### Info Hash Calculation

The info hash represents the cryptographic identity of a torrent - it's how trackers, peers, and clients uniquely identify the specific content being shared. Think of it as a fingerprint that's computed from the torrent's file and piece information. This 20-byte SHA-1 digest is derived not from the parsed data structure, but from the exact encoded bytes of the info dictionary as it appears in the torrent file.

This distinction is absolutely critical. We cannot simply re-encode the parsed info dictionary and compute its hash, because Bencode serialization must produce identical output to the original file. Even though Bencode is deterministic, our parser might represent the data differently than the original encoder (for example, using different string types or numeric representations).

> The info hash must be computed from the original Bencode bytes, not from re-serialized parsed data. This ensures consistency across all BitTorrent clients regardless of their internal data representations.

The info hash serves multiple purposes in the BitTorrent ecosystem:

| Use Case | Purpose | Why Identity Matters |
|----------|---------|---------------------|
| Tracker Communication | Identifies which swarm to join | Trackers group peers by info hash |
| Peer Handshakes | Verifies peers are sharing the same content | Prevents mixing incompatible swarms |
| DHT Lookups | Distributed hash table key for peer discovery | DHT stores peer lists keyed by info hash |
| Magnet Links | Content identifier in magnet URIs | Allows torrent-less peer discovery |
| Duplicate Detection | Prevents downloading same content twice | Client can identify already-downloaded torrents |

The algorithm for info hash calculation requires careful handling of the original Bencode stream:

1. **Locate info dictionary boundaries**: Find the start of the info dictionary in the original Bencode data
2. **Extract exact bytes**: Copy the complete info dictionary including its `d` and `e` delimiters
3. **Compute SHA-1 hash**: Apply SHA-1 cryptographic hash function to these exact bytes
4. **Store as 20-byte array**: Preserve the binary hash for use in protocol messages

The boundary location algorithm must parse the Bencode structure while tracking byte positions:

1. Begin parsing at the root dictionary marker `d`
2. Parse key-value pairs, tracking the byte offset of each key
3. When encountering the key `"info"`, record the current byte position
4. Parse the info dictionary value completely, recording the ending byte position
5. Extract the substring from start position to end position inclusive of delimiters

> **Decision: Original Bytes vs Re-encoding for Info Hash**
> - **Context**: Need to compute SHA-1 hash of info dictionary for torrent identification
> - **Options Considered**: Re-encode parsed data, track original bytes during parsing, separate parsing pass for hash calculation
> - **Decision**: Track byte boundaries during initial parse to extract original info bytes
> - **Rationale**: Guarantees hash consistency with torrent creator and other clients, avoids re-encoding complexity
> - **Consequences**: Slightly more complex parser but ensures protocol compliance and interoperability

The implementation must handle the info dictionary as a contiguous byte slice from the original torrent file. This means our parser needs to either track byte positions during parsing or provide a method to locate dictionary boundaries after parsing is complete.

A robust approach involves augmenting our Bencode decoder to track parsing positions:

1. **Enhanced decoder state**: Modify the `Decoder` to track current byte position alongside the buffered reader
2. **Boundary recording**: When parsing the root dictionary, record start and end positions for the info key's value
3. **Byte extraction**: After successful parsing, extract the info dictionary bytes using recorded boundaries
4. **Hash computation**: Apply SHA-1 to the extracted bytes and store the result in our `MetaInfo` structure

The info hash becomes the torrent's primary key throughout the BitTorrent protocol. It appears in tracker announce requests, peer handshake messages, and DHT queries. Every piece of communication about this torrent will reference this 20-byte identifier, making its correct calculation essential for protocol compatibility.

### Common Parsing Pitfalls

Torrent file parsing involves several subtle challenges that frequently trip up implementers. Understanding these pitfalls and their solutions is crucial for building a robust BitTorrent client that works correctly with real-world torrent files.

⚠️ **Pitfall: Treating Bencode Strings as Text**

Many developers assume Bencode strings contain UTF-8 text and attempt to decode them as such. This fails catastrophically with binary data like piece hashes, which contain arbitrary byte values that aren't valid UTF-8 sequences. The pieces field, for example, contains concatenated SHA-1 hashes that include null bytes, control characters, and invalid UTF-8 sequences.

**Why it's wrong**: Bencode byte strings can contain any byte values from 0x00 to 0xFF. Attempting UTF-8 decoding will either fail with encoding errors or silently corrupt the binary data through replacement characters.

**How to fix**: Always handle Bencode strings as raw byte slices (`[]byte` in Go). Only convert to text strings when you're certain the content is textual (like file names) and need to display it to users. For protocol fields like piece hashes and peer IDs, preserve the binary representation.

⚠️ **Pitfall: Incorrect Info Dictionary Boundary Detection**

Computing the info hash requires extracting the exact bytes of the info dictionary from the original torrent file. A common mistake is attempting to find these boundaries through string searching or regex matching, which fails when the info dictionary contains binary data that happens to include the search patterns.

**Why it's wrong**: String-based searches can match content inside binary fields, leading to incorrect boundary detection. For example, searching for the byte sequence `"e"` to find the dictionary end might match a byte inside a piece hash.

**How to fix**: Parse the Bencode structure properly while tracking byte positions. The parser understands the structure and can correctly identify where the info dictionary begins and ends, even when it contains arbitrary binary data.

⚠️ **Pitfall: Ignoring Dictionary Key Ordering Requirements**

Bencode requires dictionary keys to appear in lexicographically sorted order. Some implementations skip this validation, accepting malformed torrent files that would be rejected by other clients. This can lead to info hash mismatches when different clients compute hashes from differently-ordered data.

**Why it's wrong**: The Bencode specification mandates key ordering to ensure deterministic serialization. Accepting out-of-order keys means your client might compute different info hashes than other clients for the same logical content.

**How to fix**: Validate key ordering during dictionary parsing. Maintain the previous key and ensure each new key is lexicographically greater. Reject torrent files with misordered keys as malformed.

⚠️ **Pitfall: Integer Parsing Edge Cases**

Bencode integer parsing has several edge cases that naive implementations handle incorrectly. These include leading zeros (forbidden except for zero itself), empty integer fields, and very large numbers that exceed language-specific integer ranges.

**Why it's wrong**: Incorrect integer parsing can lead to wrong file sizes, piece lengths, or creation dates. More critically, accepting malformed integers that other clients reject can cause protocol incompatibilities.

**How to fix**: Implement strict integer validation:
- Reject integers with leading zeros except for `i0e`
- Reject empty integers like `ie`
- Handle potential overflow for very large file sizes
- Validate that integers are well-formed decimal numbers

⚠️ **Pitfall: Path Traversal in Multi-File Torrents**

Multi-file torrents specify file paths as arrays of path components. Malicious torrent files might include path components like `".."` that could cause files to be written outside the intended download directory, potentially overwriting system files.

**Why it's wrong**: Path traversal attacks can compromise the user's system by overwriting arbitrary files. A torrent containing a file path like `[".", ".", ".", "etc", "passwd"]` could overwrite `/etc/passwd` on Unix systems.

**How to fix**: Validate all path components during metadata extraction:
- Reject path components that are `"."` or `".."`
- Reject absolute paths that start with `/` or drive letters
- Sanitize path components to remove dangerous characters
- Construct file paths within a designated download directory

⚠️ **Pitfall: Piece Hash Count Validation**

The pieces field contains concatenated SHA-1 hashes, so its length must be a multiple of 20 bytes. However, you also need to validate that the number of piece hashes matches the expected count based on total file size and piece length.

**Why it's wrong**: Accepting torrents with incorrect piece counts can lead to downloads that can never complete (missing pieces) or buffer overruns when accessing piece hashes by index.

**How to fix**: Calculate expected piece count and validate consistency:
- Compute expected pieces as `ceil(total_file_size / piece_length)`
- Verify that `len(pieces) / 20 == expected_piece_count`
- Reject torrents where the piece hash count doesn't match the calculated requirement

⚠️ **Pitfall: Memory Exhaustion on Large Torrents**

Very large torrents can contain millions of pieces, each requiring a 20-byte hash. Loading all piece hashes into memory simultaneously might exhaust available RAM for torrents containing terabytes of data.

**Why it's wrong**: Memory exhaustion can crash your client or make the system unresponsive. A 1TB torrent with 256KB pieces requires about 80MB just for piece hashes.

**How to fix**: For very large torrents, consider streaming piece hash access rather than loading everything into memory. Implement lazy loading where piece hashes are read from the torrent file as needed, or use memory-mapped files for efficient access without full loading.

### Implementation Guidance

This subsection provides the concrete Go implementation infrastructure needed to parse torrent files and extract metadata. The focus is on providing complete, working code for supporting components while leaving the core learning objectives as exercises for implementation.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| File I/O | `os.File` with `io.Reader` interface | Memory-mapped files with `golang.org/x/exp/mmap` |
| Hash Computation | `crypto/sha1` standard library | Streaming hash computation for large files |
| Error Handling | Basic `error` returns with wrapped context | Structured error types with error codes |
| Testing | Table-driven tests with `testing` package | Property-based testing with `github.com/leanovate/gopter` |
| Logging | Standard `log` package for debugging | Structured logging with `github.com/sirupsen/logrus` |

#### Recommended File Structure

```
project-root/
├── cmd/
│   └── torrent/
│       └── main.go              ← CLI entry point for testing
├── internal/
│   ├── bencode/
│   │   ├── decoder.go           ← Core bencode parsing logic
│   │   ├── decoder_test.go      ← Unit tests for decoder
│   │   └── types.go             ← Bencode type constants
│   ├── torrent/
│   │   ├── metainfo.go          ← Torrent metadata structures
│   │   ├── parser.go            ← High-level torrent file parsing
│   │   ├── parser_test.go       ← Integration tests with real torrents
│   │   └── infohash.go          ← Info hash calculation utilities
│   └── testdata/
│       ├── single-file.torrent  ← Test torrent files
│       ├── multi-file.torrent
│       └── invalid-*.torrent    ← Malformed files for error testing
└── go.mod
```

#### Infrastructure Starter Code

**File: `internal/bencode/types.go` (Complete - Ready to Use)**

```go
package bencode

import (
	"errors"
	"fmt"
)

// BencodeType represents the four fundamental Bencode data types
type BencodeType int

const (
	TypeString BencodeType = iota
	TypeInteger
	TypeList
	TypeDictionary
)

// Common Bencode parsing errors
var (
	ErrInvalidType       = errors.New("invalid bencode type marker")
	ErrUnexpectedEOF     = errors.New("unexpected end of input")
	ErrInvalidInteger    = errors.New("invalid integer format")
	ErrInvalidLength     = errors.New("invalid string length")
	ErrUnsortedKeys      = errors.New("dictionary keys not in sorted order")
	ErrNonStringKey      = errors.New("dictionary key is not a string")
	ErrInsufficientData  = errors.New("insufficient data for declared length")
)

// ParseError wraps parsing errors with position information
type ParseError struct {
	Offset int
	Err    error
}

func (e *ParseError) Error() string {
	return fmt.Sprintf("bencode parse error at offset %d: %v", e.Offset, e.Err)
}

func (e *ParseError) Unwrap() error {
	return e.Err
}

// IsValidInteger checks if a string represents a valid Bencode integer
func IsValidInteger(s string) bool {
	if len(s) == 0 {
		return false
	}
	
	// Handle negative numbers
	start := 0
	if s[0] == '-' {
		if len(s) == 1 {
			return false // Just "-" is invalid
		}
		start = 1
	}
	
	// Check for leading zeros (forbidden except for "0" itself)
	if s[start] == '0' && len(s) > start+1 {
		return false
	}
	
	// Verify all remaining characters are digits
	for i := start; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	
	return true
}
```

**File: `internal/torrent/metainfo.go` (Complete - Ready to Use)**

```go
package torrent

import (
	"crypto/sha1"
	"time"
)

// MetaInfo represents the complete torrent file metadata
type MetaInfo struct {
	Announce     string     `json:"announce"`
	AnnounceList [][]string `json:"announce_list,omitempty"`
	CreationDate time.Time  `json:"creation_date,omitempty"`
	Comment      string     `json:"comment,omitempty"`
	CreatedBy    string     `json:"created_by,omitempty"`
	InfoHash     [20]byte   `json:"info_hash"`
	Info         Info       `json:"info"`
}

// Info represents the info dictionary containing file and piece data
type Info struct {
	Name        string `json:"name"`
	Length      int64  `json:"length,omitempty"`      // Single-file torrents only
	Files       []File `json:"files,omitempty"`       // Multi-file torrents only
	PieceLength int64  `json:"piece_length"`
	Pieces      []byte `json:"pieces"`                // Concatenated SHA-1 hashes
}

// File represents a single file in a multi-file torrent
type File struct {
	Length int64    `json:"length"`
	Path   []string `json:"path"`
}

// GetPieceHashes extracts individual piece hashes from the concatenated pieces field
func (info *Info) GetPieceHashes() [][20]byte {
	const PIECE_HASH_SIZE = 20
	numPieces := len(info.Pieces) / PIECE_HASH_SIZE
	hashes := make([][20]byte, numPieces)
	
	for i := 0; i < numPieces; i++ {
		start := i * PIECE_HASH_SIZE
		copy(hashes[i][:], info.Pieces[start:start+PIECE_HASH_SIZE])
	}
	
	return hashes
}

// TotalLength calculates the total size of all files in the torrent
func (info *Info) TotalLength() int64 {
	if info.Length > 0 {
		// Single-file torrent
		return info.Length
	}
	
	// Multi-file torrent - sum all file lengths
	total := int64(0)
	for _, file := range info.Files {
		total += file.Length
	}
	return total
}

// ExpectedPieceCount calculates how many pieces this torrent should have
func (info *Info) ExpectedPieceCount() int {
	totalLength := info.TotalLength()
	return int((totalLength + info.PieceLength - 1) / info.PieceLength) // Ceiling division
}

// ValidatePieceHashes checks that the piece hash count matches expected
func (info *Info) ValidatePieceHashes() error {
	const PIECE_HASH_SIZE = 20
	
	if len(info.Pieces)%PIECE_HASH_SIZE != 0 {
		return errors.New("pieces field length is not a multiple of 20")
	}
	
	actualCount := len(info.Pieces) / PIECE_HASH_SIZE
	expectedCount := info.ExpectedPieceCount()
	
	if actualCount != expectedCount {
		return fmt.Errorf("piece hash count mismatch: got %d, expected %d", 
			actualCount, expectedCount)
	}
	
	return nil
}
```

#### Core Logic Skeleton Code

**File: `internal/bencode/decoder.go` (Skeleton - Implement the TODOs)**

```go
package bencode

import (
	"bufio"
	"io"
	"strconv"
)

// Decoder handles parsing of Bencode data streams
type Decoder struct {
	reader *bufio.Reader
	offset int // Track current byte position for error reporting
}

// NewDecoder creates a new Bencode decoder for the given reader
func NewDecoder(r io.Reader) *Decoder {
	return &Decoder{
		reader: bufio.NewReader(r),
		offset: 0,
	}
}

// Decode parses the next Bencode value from the stream
// Returns the parsed value as interface{} - caller must type assert
func (d *Decoder) Decode() (interface{}, error) {
	// TODO 1: Peek at the next byte to determine value type
	// TODO 2: Dispatch to appropriate parsing method based on type:
	//         - '0'-'9': string (starts with length digits)
	//         - 'i': integer
	//         - 'l': list  
	//         - 'd': dictionary
	// TODO 3: Return ParseError with offset information for invalid types
	// Hint: Use d.reader.Peek(1) to look ahead without consuming
	panic("implement me")
}

// decodeString parses a Bencode string in format "length:data"
func (d *Decoder) decodeString() ([]byte, error) {
	// TODO 1: Read decimal digits until hitting ':' character
	// TODO 2: Parse the accumulated digits as string length
	// TODO 3: Validate length is non-negative using IsValidInteger
	// TODO 4: Read exactly that many bytes as the string data
	// TODO 5: Return the raw bytes (don't convert to text)
	// Hint: Use d.readUntil(':') helper and strconv.Atoi for parsing
	panic("implement me")
}

// decodeInteger parses a Bencode integer in format "i<number>e"
func (d *Decoder) decodeInteger() (int64, error) {
	// TODO 1: Verify next byte is 'i' and consume it
	// TODO 2: Read characters until 'e', accumulating the number string
	// TODO 3: Validate integer format using IsValidInteger (no leading zeros)
	// TODO 4: Parse as int64 and handle potential overflow
	// TODO 5: Consume the closing 'e' delimiter
	// Hint: Use d.readUntil('e') and strconv.ParseInt
	panic("implement me")
}

// decodeList parses a Bencode list in format "l<items>e"
func (d *Decoder) decodeList() ([]interface{}, error) {
	// TODO 1: Verify next byte is 'l' and consume it
	// TODO 2: Initialize empty slice for accumulating items
	// TODO 3: Loop while next byte is not 'e':
	//         - Recursively call Decode() for each item
	//         - Append successful results to slice
	// TODO 4: Consume closing 'e' delimiter
	// TODO 5: Return completed slice
	// Hint: Check d.peekByte() != 'e' for loop condition
	panic("implement me")
}

// decodeDictionary parses a Bencode dictionary in format "d<pairs>e"
func (d *Decoder) decodeDictionary() (map[string]interface{}, error) {
	// TODO 1: Verify next byte is 'd' and consume it
	// TODO 2: Initialize empty map and previousKey for sorting validation
	// TODO 3: Loop while next byte is not 'e':
	//         - Parse key (must be string type)
	//         - Validate key is lexicographically > previousKey
	//         - Parse value (recursive Decode() call)
	//         - Insert key-value pair into map
	// TODO 4: Consume closing 'e' delimiter
	// TODO 5: Return completed map
	// Hint: Use string comparison for key ordering validation
	panic("implement me")
}

// Helper methods (Complete - Ready to Use)

func (d *Decoder) readByte() (byte, error) {
	b, err := d.reader.ReadByte()
	if err == nil {
		d.offset++
	}
	return b, err
}

func (d *Decoder) peekByte() (byte, error) {
	bytes, err := d.reader.Peek(1)
	if err != nil {
		return 0, err
	}
	return bytes[0], nil
}

func (d *Decoder) readUntil(delimiter byte) (string, error) {
	var result []byte
	for {
		b, err := d.readByte()
		if err != nil {
			return "", err
		}
		if b == delimiter {
			break
		}
		result = append(result, b)
	}
	return string(result), nil
}
```

**File: `internal/torrent/parser.go` (Skeleton - Implement the TODOs)**

```go
package torrent

import (
	"crypto/sha1"
	"os"
	"time"
	"your-project/internal/bencode"
)

// ParseFromFile reads and parses a torrent file from disk
func ParseFromFile(filepath string) (*MetaInfo, error) {
	// TODO 1: Open the torrent file for reading
	// TODO 2: Create a Bencode decoder for the file
	// TODO 3: Parse the complete Bencode structure
	// TODO 4: Extract metadata using extractMetaInfo
	// TODO 5: Calculate info hash from original file bytes
	// Hint: You'll need to re-read the file to get original info bytes
	panic("implement me")
}

// extractMetaInfo converts decoded Bencode data to MetaInfo struct
func extractMetaInfo(decoded interface{}) (*MetaInfo, error) {
	// TODO 1: Type assert root as map[string]interface{}
	// TODO 2: Extract required 'announce' field as string
	// TODO 3: Extract required 'info' dictionary
	// TODO 4: Extract optional fields (announce-list, creation date, comment)
	// TODO 5: Parse the info dictionary using extractInfo
	// TODO 6: Validate extracted metadata for consistency
	// Hint: Use type assertion with ok pattern: value, ok := data.(string)
	panic("implement me")
}

// extractInfo converts decoded info dictionary to Info struct
func extractInfo(infoDict interface{}) (Info, error) {
	// TODO 1: Type assert as map[string]interface{}
	// TODO 2: Extract required fields: name, piece length, pieces
	// TODO 3: Determine single-file vs multi-file by checking for 'files' field
	// TODO 4: For single-file: extract 'length' field
	// TODO 5: For multi-file: extract 'files' array and process each file
	// TODO 6: Validate piece length and pieces field format
	// TODO 7: Return populated Info struct
	// Hint: Check for 'files' field presence to determine torrent type
	panic("implement me")
}

// CalculateInfoHash computes SHA-1 hash of the info dictionary bytes
func CalculateInfoHash(torrentData []byte) ([20]byte, error) {
	// TODO 1: Parse Bencode to find info dictionary boundaries
	// TODO 2: Extract exact bytes of info dict including 'd' and 'e' delimiters
	// TODO 3: Compute SHA-1 hash of those exact bytes
	// TODO 4: Return as 20-byte array
	// Hint: Need to track byte positions during parsing to extract original bytes
	panic("implement me")
}
```

#### Language-Specific Hints

**Go-Specific Implementation Tips:**

- Use `bufio.Reader` for efficient parsing with lookahead capabilities via `Peek()`
- Handle binary data with `[]byte` slices - never convert to `string` unless displaying to users
- Use `strconv.ParseInt()` with base 10 for integer parsing with overflow detection
- Implement error wrapping with `fmt.Errorf("context: %w", originalError)` for better debugging
- Use type assertions with the two-value form: `value, ok := data.(string)` to avoid panics
- Apply `crypto/sha1.Sum()` for hash calculation - it returns `[20]byte` directly
- Consider using `os.Open()` with `defer file.Close()` for proper resource management

**Common Go Gotchas in This Context:**

- String vs `[]byte` confusion: Bencode "strings" are binary data, use `[]byte`
- Integer overflow: Use `int64` for all numeric fields to handle large files
- Map iteration order: Go maps have random iteration order, but Bencode requires sorted keys during encoding
- Error handling: Always check errors from I/O operations before using results
- Type assertions: Failed type assertions panic unless using the two-value form

#### Milestone Checkpoint

After implementing the core parsing logic, verify your implementation with these tests:

**Test Command:**
```bash
cd internal/bencode && go test -v
cd ../torrent && go test -v
```

**Expected Behavior:**
1. **Bencode decoder tests pass**: All four value types parse correctly, edge cases are handled
2. **Torrent parsing integration test**: Parse a real torrent file and extract metadata
3. **Info hash consistency**: Calculate same info hash as other BitTorrent clients for the same torrent

**Manual Verification Steps:**
1. Download a simple torrent file (like Ubuntu ISO) from a legitimate source
2. Parse it with your implementation: `go run cmd/torrent/main.go parse ubuntu.torrent`
3. Expected output should show:
   - Announce URL (HTTP tracker)
   - File name and size matching the actual ISO
   - Piece count and piece length (typically 512KB to 4MB for large files)
   - Valid 40-character hex info hash
4. Compare info hash with another BitTorrent client - they should match exactly

**Signs Something Is Wrong:**
| Symptom | Likely Cause | How to Debug | Fix |
|---------|--------------|--------------|-----|
| Parse errors on valid torrents | Bencode decoder bugs | Test each type separately | Review string/integer parsing logic |
| Info hash doesn't match other clients | Using re-encoded bytes instead of original | Print hash input bytes | Extract exact original info dict bytes |
| Piece count validation fails | Incorrect total size calculation | Check multi-file length summing | Verify Files vs Length handling |
| Panic on type assertions | Missing error handling | Add two-value type assertions | Use `value, ok := data.(type)` pattern |


## Tracker Communication (Milestone 2)

> **Milestone(s):** This section corresponds to Milestone 2 - Tracker Communication, implementing HTTP tracker protocol for peer discovery with proper URL encoding and response parsing.

After successfully parsing a torrent file and extracting its metadata, our BitTorrent client faces a fundamental challenge: finding other peers who have the same file. Unlike a traditional client-server model where you simply connect to a known server, BitTorrent operates as a decentralized peer-to-peer network. The **tracker** serves as the coordination point that helps peers discover each other, acting as a matchmaker in the distributed file-sharing ecosystem.

### Tracker as Matchmaking Service

Think of a BitTorrent tracker as a sophisticated dating app for files. Just as a dating app helps people with similar interests find each other, a tracker helps peers interested in the same files discover and connect to one another. When you join a dating app, you create a profile describing yourself and what you're looking for. Similarly, when your BitTorrent client contacts a tracker, it announces its identity, what file it wants (identified by the info hash), and its current status (how much it has downloaded, uploaded, and how much is left).

The dating app doesn't store the actual conversations between matched users—it just facilitates the initial introduction. Likewise, the tracker doesn't participate in the actual file transfer. It simply maintains a registry of peers interested in each torrent and provides peer lists when requested. The tracker knows which peers are actively participating in a swarm (the collection of all peers sharing a particular torrent), their network addresses, and their current participation status.

This analogy extends to the periodic check-ins that both systems require. Dating apps might ask you to update your profile or confirm you're still active. Similarly, BitTorrent clients must periodically re-announce to the tracker, reporting their progress and confirming they're still participating in the swarm. This prevents the tracker from serving stale peer information and ensures an accurate view of swarm health.

The tracker protocol is intentionally lightweight and stateless. The tracker doesn't need to remember complex state about each peer—it simply maintains a current snapshot of who's interested in what files and their network coordinates. This simplicity allows trackers to serve thousands of concurrent torrents and handle massive swarms efficiently.

### Announce Request Protocol

The **announce request** is the fundamental communication mechanism between BitTorrent clients and trackers. This HTTP GET request follows a specific format defined by the BitTorrent protocol specification, carrying essential information about the client's current state and requesting a list of peers participating in the same swarm.

The announce URL construction begins with the base tracker URL extracted from the torrent file's `announce` field. To this base URL, we append a query string containing several required parameters that inform the tracker about our client's current status and capabilities.

| Parameter | Type | Description | Encoding Requirements |
|-----------|------|-------------|----------------------|
| `info_hash` | 20-byte binary | SHA-1 hash identifying the torrent | URL-encoded binary data |
| `peer_id` | 20-byte binary | Unique identifier for our client instance | URL-encoded binary data |
| `port` | Integer | TCP port our client listens on for peer connections | Decimal string |
| `uploaded` | Integer | Total bytes uploaded to other peers | Decimal string |
| `downloaded` | Integer | Total bytes downloaded from other peers | Decimal string |
| `left` | Integer | Bytes remaining to complete the download | Decimal string |
| `compact` | Integer | Request compact peer list format (always 1) | Literal "1" |
| `event` | String | Current client state transition | "started", "completed", "stopped", or omitted |

The most critical aspect of announce request construction is proper URL encoding of binary parameters. The `info_hash` and `peer_id` are 20-byte binary values that must be percent-encoded for HTTP transmission. Each byte that is not a unreserved character (alphanumeric, hyphen, underscore, period, or tilde) must be represented as `%XX` where XX is the hexadecimal representation of the byte value.

> **Decision: HTTP GET vs POST for Announce Requests**
> - **Context**: The BitTorrent specification allows both GET and POST methods for announce requests, requiring implementers to choose an approach
> - **Options Considered**: HTTP GET with query parameters, HTTP POST with form data, HTTP POST with request body
> - **Decision**: Use HTTP GET with query parameters
> - **Rationale**: GET requests are simpler to implement, debug, and cache. They're stateless and idempotent by nature, matching the tracker protocol's design. Most existing trackers expect GET requests, ensuring maximum compatibility
> - **Consequences**: URL length limitations could theoretically be an issue with very long announce URLs, but in practice BitTorrent announce requests are well within browser and server URL limits

The `event` parameter deserves special attention as it communicates significant state changes to the tracker. When a client first starts downloading a torrent, it sends `event=started` to announce its entry into the swarm. Upon completing the download, it sends `event=completed` to indicate it's now seeding. When shutting down gracefully, it sends `event=stopped` to allow the tracker to remove it from the peer list immediately rather than waiting for a timeout.

Announce requests also serve as a progress reporting mechanism. The `uploaded`, `downloaded`, and `left` fields allow the tracker to maintain statistics about swarm health and individual peer contribution. Some trackers use this information for ratio enforcement or to provide preferential treatment to peers who contribute more upload bandwidth.

The `compact` parameter requests that the tracker return peer information in a space-efficient binary format rather than a more verbose dictionary format. This compact format reduces bandwidth usage and parsing overhead, making it the preferred approach for production implementations.

Here's the step-by-step process for constructing and sending an announce request:

1. Extract the base announce URL from the torrent's `MetaInfo.Announce` field
2. Generate a unique 20-byte `peer_id` for this client session (commonly using a prefix identifying the client software followed by random bytes)
3. Calculate current progress values: bytes uploaded, downloaded, and remaining
4. URL-encode the binary `info_hash` and `peer_id` values using percent-encoding
5. Construct the query string by concatenating all parameters with proper separators
6. Append the query string to the base URL and send an HTTP GET request
7. Parse the response to extract peer information and tracker guidance

### Peer List Response Parsing

The tracker's response to an announce request contains critical information for peer discovery and swarm coordination. This response is encoded using the same Bencode format used for torrent files, requiring our existing decoder to parse the structured data.

| Response Field | Type | Description | Required |
|----------------|------|-------------|----------|
| `interval` | Integer | Seconds to wait before next announce | Yes |
| `complete` | Integer | Number of peers with complete file (seeders) | No |
| `incomplete` | Integer | Number of peers still downloading (leechers) | No |
| `peers` | Binary String | Compact peer list in 6-byte format | Yes |
| `failure reason` | String | Error message if request failed | Only on failure |

The most important field is the `peers` binary string, which contains a concatenated list of peer network addresses in compact format. Each peer is represented by exactly 6 bytes: 4 bytes for the IPv4 address followed by 2 bytes for the port number, both in network byte order (big-endian).

Parsing the compact peer list requires careful handling of binary data and endianness conversion:

1. Verify the peers field length is a multiple of 6 bytes (each peer requires exactly 6 bytes)
2. Iterate through the binary data in 6-byte chunks
3. Extract bytes 0-3 as the IPv4 address in network byte order
4. Extract bytes 4-5 as the port number in network byte order
5. Convert the port from big-endian to host byte order
6. Create a `PeerInfo` structure containing the IP address and port

The `interval` field specifies how long the client should wait before making the next announce request. This prevents clients from overwhelming the tracker with excessive requests while ensuring the tracker maintains reasonably current swarm information. Typical intervals range from 15 minutes to several hours, with shorter intervals used for more active torrents.

> **Critical Implementation Detail**: The compact peer format uses network byte order (big-endian) for both IP addresses and port numbers. Many programming languages use host byte order internally, requiring explicit conversion using functions like `binary.BigEndian.Uint16()` in Go or `socket.ntohs()` in C. Failure to perform this conversion results in incorrect peer addresses and connection failures.

The `complete` and `incomplete` fields provide valuable swarm health information. `complete` indicates the number of seeders (peers with the complete file), while `incomplete` shows the number of leechers (peers still downloading). This information helps clients assess download viability—a torrent with no seeders cannot be completed, while one with many seeders typically offers faster download speeds.

**Re-announce scheduling** is crucial for maintaining accurate peer lists and swarm participation. The client must track the time of the last announce and schedule the next one based on the tracker's `interval` response. This scheduling should account for network delays and provide some jitter to prevent all clients from announcing simultaneously.

The re-announce process follows this algorithm:

1. Record the timestamp of the successful announce response
2. Extract the `interval` value from the tracker response
3. Schedule the next announce for `current_time + interval + random_jitter`
4. Use a small random jitter (±10% of interval) to distribute load
5. Include updated progress statistics in the next announce request
6. Reset the schedule if any announce request fails, implementing exponential backoff

### Tracker Error Handling

Robust tracker communication requires comprehensive error handling to deal with network failures, server errors, and protocol violations. The distributed nature of BitTorrent means that tracker failures should not prevent file downloads—clients must gracefully degrade and continue operating with cached peer information.

| Failure Mode | Detection Method | Recovery Strategy | Retry Timing |
|--------------|------------------|-------------------|--------------|
| Network timeout | HTTP request timeout | Exponential backoff retry | 30s, 60s, 120s, 300s |
| HTTP error status | 4xx/5xx response codes | Check failure reason, retry with backoff | Based on status code |
| Invalid response format | Bencode parsing failure | Log error, use cached peers | Next scheduled interval |
| Tracker failure reason | "failure reason" field in response | Log reason, retry with backoff | 5min, 15min, 30min |
| DNS resolution failure | Domain name lookup failure | Retry with exponential backoff | 60s, 300s, 900s |
| Connection refused | TCP connection failure | Assume temporary failure, retry | 30s, 120s, 300s |

**Network timeout handling** requires careful balance between responsiveness and resource usage. Tracker requests should have reasonable timeouts (typically 30-60 seconds) to detect unresponsive servers without blocking the client indefinitely. When timeouts occur, implement exponential backoff with jitter to avoid overwhelming recovering servers.

**HTTP error response handling** depends on the specific status code received. 4xx errors typically indicate client-side issues (malformed requests, invalid info hashes) that won't resolve with retries, while 5xx errors suggest server-side problems that may be temporary. The client should distinguish between these cases and adjust retry behavior accordingly.

**Malformed response handling** occurs when the tracker returns data that doesn't conform to the expected Bencode format or contains invalid peer information. Rather than crashing, the client should log the error details and continue operating with previously cached peer information. This allows downloads to continue even when trackers misbehave.

> **Design Principle**: Graceful degradation is fundamental to BitTorrent's robustness. The protocol is designed to continue functioning even when central infrastructure (like trackers) fails. Clients should never become completely non-functional due to tracker issues.

**Exponential backoff implementation** prevents clients from overwhelming failing trackers while still attempting recovery. The backoff algorithm should:

1. Start with a base delay (e.g., 30 seconds) after the first failure
2. Double the delay after each subsequent failure, up to a maximum (e.g., 30 minutes)
3. Add random jitter (±25% of the delay) to prevent thundering herd effects
4. Reset to the base delay after any successful request
5. Continue attempting indefinitely, as trackers may recover after extended outages

**Alternative tracker support** enhances reliability when torrents specify multiple tracker URLs in the `announce-list` field. The client should try trackers in order, falling back to alternatives when the primary tracker fails. This requires tracking the health of each tracker and implementing intelligent failover logic.

The failover algorithm proceeds as follows:

1. Attempt to contact the primary tracker from `announce` field
2. If the primary tracker fails, iterate through `announce-list` tiers
3. Within each tier, try trackers in random order to distribute load
4. Mark trackers as failed after multiple consecutive failures
5. Periodically retry failed trackers to detect when they recover
6. Prefer trackers that have recently provided successful responses

### Implementation Guidance

This subsection provides concrete implementation details for the tracker communication system, including complete working code for HTTP transport and response parsing, plus detailed skeletons for the core tracker protocol logic.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| HTTP Client | `net/http.Client` with custom timeout | `net/http.Client` with retry middleware and connection pooling |
| URL Encoding | `url.QueryEscape()` for individual parameters | Custom percent-encoding for binary data compliance |
| Response Parsing | Reuse existing Bencode decoder | Streaming Bencode parser for large peer lists |
| Concurrency | Single goroutine with timer-based scheduling | Worker pool with priority queue for multiple trackers |
| Error Handling | Simple retry with fixed delays | Exponential backoff with jitter and circuit breaker |

#### Recommended File Structure

```
internal/tracker/
  client.go              ← main tracker client implementation
  client_test.go         ← unit tests for tracker communication
  announce.go            ← announce request/response structures
  peers.go               ← peer list parsing and management
  errors.go              ← tracker-specific error types
cmd/torrent-client/
  main.go                ← integrates tracker client with overall application
```

#### Infrastructure Code: HTTP Client and URL Encoding

```go
// Package tracker provides BitTorrent tracker communication functionality
package tracker

import (
    "context"
    "fmt"
    "io"
    "net/http"
    "net/url"
    "time"
)

// HTTPClient wraps net/http.Client with BitTorrent-specific configuration
type HTTPClient struct {
    client *http.Client
}

// NewHTTPClient creates an HTTP client configured for tracker communication
func NewHTTPClient() *HTTPClient {
    return &HTTPClient{
        client: &http.Client{
            Timeout: 30 * time.Second,
            Transport: &http.Transport{
                MaxIdleConns:        10,
                IdleConnTimeout:     30 * time.Second,
                DisableCompression:  true, // Bencode responses don't compress well
                MaxIdleConnsPerHost: 2,
            },
        },
    }
}

// Get performs an HTTP GET request with proper error handling
func (h *HTTPClient) Get(ctx context.Context, url string) ([]byte, error) {
    req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
    if err != nil {
        return nil, fmt.Errorf("creating request: %w", err)
    }

    // Set User-Agent to identify our BitTorrent client
    req.Header.Set("User-Agent", "go-torrent-client/1.0")

    resp, err := h.client.Do(req)
    if err != nil {
        return nil, fmt.Errorf("executing request: %w", err)
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        return nil, fmt.Errorf("HTTP error: %d %s", resp.StatusCode, resp.Status)
    }

    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return nil, fmt.Errorf("reading response body: %w", err)
    }

    return body, nil
}

// URLEncodeBinary performs percent-encoding for binary data (info_hash and peer_id)
// This is more strict than url.QueryEscape() to ensure BitTorrent compatibility
func URLEncodeBinary(data []byte) string {
    result := make([]byte, 0, len(data)*3) // Worst case: all bytes need encoding
    
    for _, b := range data {
        // Unreserved characters: A-Z, a-z, 0-9, -, _, ., ~
        if (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z') || 
           (b >= '0' && b <= '9') || b == '-' || b == '_' || 
           b == '.' || b == '~' {
            result = append(result, b)
        } else {
            result = append(result, '%')
            result = append(result, "0123456789ABCDEF"[b>>4])
            result = append(result, "0123456789ABCDEF"[b&15])
        }
    }
    
    return string(result)
}

// BackoffScheduler implements exponential backoff for failed requests
type BackoffScheduler struct {
    baseDelay    time.Duration
    maxDelay     time.Duration
    currentDelay time.Duration
    failures     int
}

// NewBackoffScheduler creates a scheduler with sensible defaults for tracker communication
func NewBackoffScheduler() *BackoffScheduler {
    return &BackoffScheduler{
        baseDelay:    30 * time.Second,
        maxDelay:     30 * time.Minute,
        currentDelay: 30 * time.Second,
    }
}

// OnFailure records a failure and returns the next delay duration
func (b *BackoffScheduler) OnFailure() time.Duration {
    b.failures++
    
    // Exponential backoff with jitter
    jitter := time.Duration(float64(b.currentDelay) * 0.1 * (2*rand.Float64() - 1))
    delay := b.currentDelay + jitter
    
    // Double for next time, up to maximum
    b.currentDelay *= 2
    if b.currentDelay > b.maxDelay {
        b.currentDelay = b.maxDelay
    }
    
    return delay
}

// OnSuccess resets the backoff state after a successful request
func (b *BackoffScheduler) OnSuccess() {
    b.failures = 0
    b.currentDelay = b.baseDelay
}
```

#### Core Data Structures

```go
// AnnounceRequest represents the parameters sent to a BitTorrent tracker
type AnnounceRequest struct {
    InfoHash   [20]byte // SHA-1 hash of the torrent's info dictionary
    PeerID     [20]byte // Unique identifier for this client instance
    Port       int      // TCP port this client listens on
    Uploaded   int64    // Total bytes uploaded to other peers
    Downloaded int64    // Total bytes downloaded from other peers
    Left       int64    // Bytes remaining to complete download
    Event      string   // "started", "completed", "stopped", or empty
}

// AnnounceResponse represents the tracker's response to an announce request
type AnnounceResponse struct {
    Interval      int    // Seconds until next announce
    Complete      int    // Number of seeders (peers with complete file)
    Incomplete    int    // Number of leechers (peers downloading)
    Peers         []byte // Compact peer list (6 bytes per peer)
    FailureReason string // Error message if request failed
}

// PeerInfo represents a single peer's network address
type PeerInfo struct {
    IP   [4]byte  // IPv4 address in network byte order
    Port uint16   // TCP port in host byte order
}

// String returns a human-readable representation of the peer
func (p PeerInfo) String() string {
    return fmt.Sprintf("%d.%d.%d.%d:%d", p.IP[0], p.IP[1], p.IP[2], p.IP[3], p.Port)
}

// Client manages communication with BitTorrent trackers
type Client struct {
    httpClient *HTTPClient
    backoff    *BackoffScheduler
    
    // Tracker state
    announceURL   string
    lastAnnounce  time.Time
    nextAnnounce  time.Time
    trackerID     string // Optional tracker-assigned ID
}
```

#### Core Logic Skeleton: Tracker Client Implementation

```go
// NewClient creates a tracker client for the given torrent
func NewClient(announceURL string) *Client {
    return &Client{
        httpClient:  NewHTTPClient(),
        backoff:     NewBackoffScheduler(),
        announceURL: announceURL,
    }
}

// BuildAnnounceURL constructs the complete tracker URL with all required parameters
func (c *Client) BuildAnnounceURL(req AnnounceRequest) string {
    // TODO 1: Parse the base announce URL to separate base from existing query parameters
    // TODO 2: URL-encode the binary info_hash using URLEncodeBinary function
    // TODO 3: URL-encode the binary peer_id using URLEncodeBinary function  
    // TODO 4: Build query parameter string with all required fields
    // TODO 5: Handle optional event parameter (only include if not empty)
    // TODO 6: Include tracker_id if we received one from previous response
    // TODO 7: Add compact=1 to request compact peer format
    // TODO 8: Combine base URL with query parameters
    // Hint: Use url.Values{} to build query parameters safely
    // Hint: Don't use url.QueryEscape for info_hash/peer_id - use URLEncodeBinary
}

// Announce sends an announce request to the tracker and parses the response
func (c *Client) Announce(ctx context.Context, req AnnounceRequest) (*AnnounceResponse, error) {
    // TODO 1: Build the complete announce URL using BuildAnnounceURL
    // TODO 2: Send HTTP GET request using c.httpClient.Get with context
    // TODO 3: Handle HTTP errors and network timeouts appropriately
    // TODO 4: Parse the response body as Bencode using existing decoder
    // TODO 5: Extract announce response fields from Bencode dictionary
    // TODO 6: Check for "failure reason" field and return error if present
    // TODO 7: Validate that required fields (interval, peers) are present
    // TODO 8: Update c.lastAnnounce and c.nextAnnounce timestamps
    // TODO 9: Store tracker_id if provided for future requests
    // TODO 10: Reset backoff scheduler on successful response
    // Hint: Reuse the Bencode decoder from milestone 1
    // Hint: Return tracker errors (failure reason) as a distinct error type
}

// ParseCompactPeers converts the binary peer list into PeerInfo structures
func ParseCompactPeers(compactPeers []byte) ([]PeerInfo, error) {
    // TODO 1: Validate that peer list length is multiple of 6 bytes
    // TODO 2: Calculate number of peers from total length
    // TODO 3: Create slice to hold parsed peer information
    // TODO 4: Iterate through compact peer data in 6-byte chunks
    // TODO 5: Extract 4-byte IP address in network byte order
    // TODO 6: Extract 2-byte port number in network byte order  
    // TODO 7: Convert port from big-endian to host byte order
    // TODO 8: Create PeerInfo struct and add to result slice
    // TODO 9: Return completed peer list
    // Hint: Use binary.BigEndian.Uint16() for port conversion
    // Hint: IP address bytes can be copied directly (already in network order)
}

// ScheduleNextAnnounce determines when the next announce should occur
func (c *Client) ScheduleNextAnnounce(interval int) time.Time {
    // TODO 1: Calculate base next announce time as lastAnnounce + interval
    // TODO 2: Add small random jitter (±10% of interval) to distribute load
    // TODO 3: Ensure next announce is not in the past
    // TODO 4: Update c.nextAnnounce field
    // TODO 5: Return the scheduled time
    // Hint: Use rand.Float64() for jitter calculation
    // Hint: time.Now().Add() for time arithmetic
}

// ShouldAnnounce returns true if it's time to send another announce request
func (c *Client) ShouldAnnounce() bool {
    // TODO 1: Compare current time with c.nextAnnounce
    // TODO 2: Return true if current time is after scheduled announce time
    // TODO 3: Handle case where nextAnnounce is zero (first announce)
}

// OnAnnounceError handles announce request failures with backoff
func (c *Client) OnAnnounceError(err error) time.Duration {
    // TODO 1: Record the failure with backoff scheduler
    // TODO 2: Get the next retry delay from backoff.OnFailure()
    // TODO 3: Update nextAnnounce to current time + retry delay
    // TODO 4: Log the error with appropriate detail level
    // TODO 5: Return the retry delay for caller information
    // Hint: Distinguish between different error types for logging
    // Hint: Network errors vs tracker errors may need different handling
}
```

#### Testing and Validation Checkpoints

After implementing the tracker communication system, verify correct behavior with these checkpoints:

**Unit Test Verification:**
```bash
go test ./internal/tracker/... -v
```

Expected test coverage should include:
- URL encoding of binary data (info_hash and peer_id)
- Compact peer list parsing with various list sizes
- Announce request parameter construction
- Error handling for malformed tracker responses
- Backoff scheduling behavior

**Integration Test with Real Tracker:**
1. Use a test torrent file with a known working tracker
2. Send an announce request with `event=started`
3. Verify you receive a valid peer list in the response
4. Check that the `interval` field contains a reasonable value (300-1800 seconds)
5. Confirm peer IP addresses and ports are correctly parsed

**Manual Verification Steps:**
1. Enable debug logging to see announce URLs being constructed
2. Copy an announce URL and test it manually with `curl` - you should get Bencode response
3. Verify that URL encoding produces valid HTTP URLs (no unescaped binary data)
4. Check that periodic re-announce happens at the correct intervals
5. Test error handling by using an invalid tracker URL

**Common Issues and Debugging:**

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| "Invalid info_hash" error | Incorrect URL encoding of binary data | Use URLEncodeBinary, not url.QueryEscape |
| Empty peer list returned | Torrent has no active peers, or announce parameters incorrect | Test with popular torrent, verify all required parameters |
| Connection timeouts | Network issues or unresponsive tracker | Implement proper timeout handling and retry logic |
| Parse errors on response | Tracker returned non-Bencode data | Add response format validation before parsing |

![Complete Download Flow](./diagrams/download-sequence.svg)

The tracker communication implementation provides the foundation for peer discovery in our BitTorrent client. Once we can successfully retrieve peer lists from trackers, the next milestone will focus on establishing direct connections with those peers using the BitTorrent wire protocol.


## Peer Wire Protocol (Milestone 3)

> **Milestone(s):** This section corresponds to Milestone 3 - Peer Protocol, implementing the BitTorrent peer wire protocol including handshakes, message framing, and state machines.

After establishing communication with the tracker and discovering peers in Milestone 2, we now face the core challenge of the BitTorrent protocol: establishing direct peer-to-peer connections and coordinating the actual transfer of file data. The peer wire protocol is BitTorrent's application-layer communication standard that governs how peers handshake, exchange availability information, negotiate transfers, and maintain fair sharing relationships.

The peer wire protocol operates over TCP connections and implements a sophisticated state machine that balances efficiency with fairness. Unlike simple client-server protocols, BitTorrent peers must simultaneously act as both clients (requesting data) and servers (providing data), while maintaining multiple concurrent connections and enforcing upload reciprocity through choking mechanisms.

### Peer Protocol as Conversation Rules

Think of the BitTorrent peer wire protocol like the formal conversation rules at a diplomatic dinner party. When diplomats from different countries meet, they follow established etiquette to ensure productive communication despite language barriers and cultural differences.

The **handshake** is like the formal introduction where diplomats present their credentials and verify they're attending the same event (same torrent). They exchange names (peer IDs) and confirm they're discussing the same topic (info hash verification).

The **bitfield exchange** is like each diplomat placing their briefcase contents on the table, showing what documents (pieces) they brought to share. This transparency allows everyone to see who has what information, enabling efficient negotiation.

The **choking and interest system** mirrors diplomatic negotiation protocols. A diplomat might be "interested" in another's documents but "choked" (denied access) until they offer something valuable in return. The "unchoke" decision represents granting access based on reciprocal value exchange.

**Message framing** provides the structured communication format, like diplomatic protocols that specify how to format official communications with proper headers, lengths, and content organization to prevent misunderstandings.

This analogy captures the essential challenge: BitTorrent peers must coordinate complex multi-party exchanges while maintaining fairness and preventing exploitation, just as diplomatic protocols enable cooperation between self-interested parties.

### Connection Handshake

The BitTorrent handshake serves as the authentication and compatibility verification phase that must complete successfully before any data transfer can begin. This 68-byte message exchange ensures both peers are speaking the same protocol version and participating in the same torrent swarm.

The handshake message format consists of five distinct components that provide comprehensive connection validation:

| Field | Length | Type | Description |
|-------|--------|------|-------------|
| Protocol Length | 1 byte | uint8 | Always 19 (length of "BitTorrent protocol") |
| Protocol String | 19 bytes | ASCII | Always "BitTorrent protocol" |
| Reserved Bytes | 8 bytes | uint64 | Feature flags (all zeros for basic implementation) |
| Info Hash | 20 bytes | [20]byte | SHA-1 hash identifying the specific torrent |
| Peer ID | 20 bytes | [20]byte | Unique identifier for the connecting peer |

The handshake exchange follows a specific sequence that validates compatibility at each step:

1. **Connection establishment**: The initiating peer opens a TCP connection to the remote peer's IP and port (discovered from tracker).

2. **Handshake transmission**: The initiator immediately sends the complete 68-byte handshake message without waiting for any response.

3. **Protocol validation**: The receiving peer reads the first 20 bytes and verifies the protocol length (19) and protocol string ("BitTorrent protocol") match expected values.

4. **Torrent validation**: The receiving peer extracts the info hash from bytes 28-47 and confirms it matches a torrent they're actively sharing.

5. **Response handshake**: If validation succeeds, the receiving peer sends back its own 68-byte handshake message with the same info hash but its own peer ID.

6. **Mutual validation**: The initiating peer receives and validates the response handshake, confirming the info hash matches and noting the remote peer's ID.

> **Decision: Immediate Handshake Transmission**
> - **Context**: Handshake could wait for connection confirmation or send immediately after TCP establishment
> - **Options Considered**: Wait for explicit ready signal vs. immediate transmission vs. HTTP-style request/response
> - **Decision**: Send handshake immediately after TCP connection establishment
> - **Rationale**: Reduces round-trip latency, matches BitTorrent specification exactly, and simplifies state machine (no waiting state needed)
> - **Consequences**: Enables faster connection establishment but requires careful handling of partial reads during handshake parsing

The info hash validation is the critical security check that prevents peers from different torrents from establishing connections. Since the info hash is computed from the exact bencoded bytes of the torrent's info dictionary, it serves as both a unique identifier and a cryptographic commitment to the specific file contents and structure being shared.

Peer ID validation serves multiple purposes beyond simple identification. Well-formed peer IDs follow conventions that identify the BitTorrent client software and version, enabling peers to implement client-specific optimizations or compatibility workarounds. Our implementation should generate a recognizable peer ID that identifies our client while remaining unique across instances.

The reserved bytes field provides extensibility for future protocol features like encryption, DHT support, or enhanced messaging. For a basic implementation, these bytes should be set to zero, but the field must be preserved during handshake forwarding to maintain protocol compatibility.

**Handshake Error Conditions and Recovery:**

| Error Condition | Detection Method | Recovery Action |
|----------------|------------------|-----------------|
| Wrong protocol string | String comparison mismatch | Close connection immediately |
| Unknown info hash | Info hash not in active torrents | Send handshake with error info hash |
| Connection timeout | No handshake received within 30s | Close connection and retry |
| Partial handshake | TCP connection closed mid-handshake | Log error and attempt reconnection |
| Duplicate peer ID | Same peer ID as local client | Close connection (connecting to self) |

### Message Framing and Parsing

After successful handshake completion, all subsequent communication uses the BitTorrent message framing format, which provides reliable message boundaries over the streaming TCP connection. This framing system must handle variable-length messages while maintaining parsing efficiency and error recovery capabilities.

Every BitTorrent message (except the handshake) follows a standardized format that begins with a length prefix to enable proper message boundary detection:

| Field | Length | Type | Description |
|-------|--------|------|-------------|
| Message Length | 4 bytes | uint32 (big-endian) | Total bytes in message (0 for keep-alive) |
| Message ID | 1 byte | uint8 | Message type identifier (omitted for keep-alive) |
| Payload | Variable | bytes | Message-specific data (length = Message Length - 1) |

The message framing protocol defines nine standard message types that handle all peer communication needs:

| Message ID | Name | Payload Length | Purpose |
|------------|------|----------------|---------|
| - | keep-alive | 0 bytes | Maintain connection (length = 0, no ID) |
| 0 | choke | 0 bytes | Refuse to upload to peer |
| 1 | unchoke | 0 bytes | Allow uploads to peer |
| 2 | interested | 0 bytes | Want to download from peer |
| 3 | not interested | 0 bytes | No longer want to download |
| 4 | have | 4 bytes | Announce piece availability |
| 5 | bitfield | Variable | Announce complete piece availability |
| 6 | request | 12 bytes | Request specific block of data |
| 7 | piece | 9 + block size | Deliver requested block data |
| 8 | cancel | 12 bytes | Cancel previous block request |

The message parsing state machine must handle the streaming nature of TCP while maintaining message boundaries and providing robust error recovery:

![Peer Message Parsing](./diagrams/message-parsing-flow.svg)

**Message Parsing Algorithm:**

1. **Length Reading State**: Read exactly 4 bytes from the TCP connection to get the message length field, handling partial reads by maintaining a read buffer.

2. **Length Validation**: Verify the message length is reasonable (0 for keep-alive, 1-17 for standard messages, larger values only valid for bitfield and piece messages).

3. **Keep-Alive Detection**: If length is 0, this is a keep-alive message. Reset connection timeout and return to length reading state.

4. **Message ID Reading**: Read 1 byte for the message ID, validating it falls within the expected range (0-8 for standard protocol).

5. **Payload Length Calculation**: Calculate payload length as (message length - 1) and validate it matches expected length for the message type.

6. **Payload Reading**: Read exactly the calculated payload bytes, handling partial reads and maintaining progress state.

7. **Message Dispatch**: Parse the complete message according to its type and dispatch to appropriate handler function.

8. **State Reset**: Return to length reading state for the next message.

The parser must handle several challenging scenarios that arise from TCP's streaming nature and network reliability issues:

> **Critical Insight: Partial Read Handling**
> 
> TCP provides a byte stream, not message boundaries. A single `Read()` call might return part of a message, multiple complete messages, or a combination. The parser must accumulate bytes across multiple reads and only dispatch complete messages.

**Message Parsing State Management:**

| Parser State | Data Needed | Next State | Error Conditions |
|-------------|-------------|------------|------------------|
| ReadingLength | 4 bytes | ReadingMessage or KeepAlive | Connection closed, invalid length |
| ReadingMessage | Message Length bytes | ReadingLength | Connection closed, timeout |
| KeepAlive | 0 bytes | ReadingLength | N/A |

The message framer must maintain parsing state across multiple read operations, accumulating partial data until complete messages are available. This requires careful buffer management to avoid memory leaks while handling arbitrarily large messages (piece messages can be 16KB+).

**Endianness Considerations:**

All multi-byte integers in BitTorrent messages use big-endian (network) byte order. The message length, piece indices, block offsets, and block lengths must be converted from network byte order to host byte order during parsing and vice versa during message construction.

⚠️ **Pitfall: Partial Message Processing**
Many implementations incorrectly assume that a single TCP read will return a complete message. This works during testing with local connections but fails under network stress or with slow peers. Always accumulate bytes until you have a complete message before processing.

### Peer State Management

BitTorrent peers maintain a four-flag state machine that governs upload and download permissions between connected peers. This state system implements the core game theory mechanism that incentivizes sharing and prevents freeloading by requiring reciprocal exchange.

The peer state consists of four independent boolean flags that capture the bidirectional nature of BitTorrent relationships:

| State Flag | Direction | Meaning | Initial Value |
|------------|-----------|---------|---------------|
| am_choking | Local → Remote | We refuse to upload to peer | true |
| am_interested | Local → Remote | We want to download from peer | false |
| peer_choking | Remote → Local | Peer refuses to upload to us | true |
| peer_interested | Remote → Local | Peer wants to download from us | false |

These flags combine to determine the actual data transfer permissions and drive the message exchange patterns between peers:

![Peer Connection State Machine](./diagrams/peer-state-machine.svg)

**State Transition Rules:**

The state machine transitions occur in response to specific message types and local decision-making algorithms:

| Current State | Message/Event | New State | Action Taken |
|---------------|---------------|-----------|--------------|
| am_choking=true | Send unchoke | am_choking=false | Allow uploads to peer |
| am_choking=false | Send choke | am_choking=true | Block uploads to peer |
| am_interested=false | Local decision | am_interested=true | Send interested message |
| am_interested=true | Local decision | am_interested=false | Send not interested |
| peer_choking=true | Receive unchoke | peer_choking=false | Can send requests |
| peer_choking=false | Receive choke | peer_choking=true | Cancel pending requests |
| peer_interested=false | Receive interested | peer_interested=true | Consider unchoking |
| peer_interested=true | Receive not interested | peer_interested=false | May choke peer |

The combination of these flags determines the operational mode of the peer relationship:

**Operational Modes:**

| Mode | Conditions | Upload Allowed | Download Allowed | Description |
|------|------------|----------------|------------------|-------------|
| Mutual Exchange | !am_choking && !peer_choking && am_interested && peer_interested | Yes | Yes | Optimal trading state |
| Uploading Only | !am_choking && peer_choking && !am_interested && peer_interested | Yes | No | Seeding to downloader |
| Downloading Only | am_choking && !peer_choking && am_interested && !peer_interested | No | Yes | Receiving from seeder |
| Idle Connection | am_choking && peer_choking | No | No | Maintaining connection |

> **Decision: Four-Flag State Model**
> - **Context**: Need to track bidirectional choking and interest state between peers
> - **Options Considered**: Single combined state enum vs. separate directional flags vs. simplified choke-only model
> - **Decision**: Four independent boolean flags (am_choking, am_interested, peer_choking, peer_interested)
> - **Rationale**: Matches BitTorrent specification exactly, enables independent state transitions, simplifies message handling logic
> - **Consequences**: Requires careful synchronization but provides full protocol compliance and optimal flexibility

**Interest Determination Algorithm:**

The `am_interested` flag should be updated whenever the peer's available pieces change or our local needs change:

1. **Bitfield Analysis**: When receiving a peer's initial bitfield, check if they have any pieces we need.

2. **Have Message Processing**: When peer announces a new piece via `have` message, check if we need that specific piece.

3. **Local Completion**: When we complete a piece, check if we still need any pieces the peer has.

4. **Interest Update**: Send `interested` or `not interested` message only when the interest state actually changes.

**Choking Decision Algorithm:**

The choking decisions implement BitTorrent's incentive mechanism and should balance fairness with performance:

1. **Reciprocity Analysis**: Prefer to unchoke peers who are uploading to us at good rates.

2. **Optimistic Unchoking**: Periodically unchoke a random peer to discover better trading partners.

3. **Seeding Strategy**: When seeding (we have all pieces), unchoke peers with the best download rates to maximize distribution efficiency.

4. **Connection Limits**: Maintain reasonable limits on simultaneously unchoked peers (typically 4-5) to prevent connection overload.

The state machine must handle message ordering carefully, as network delays can cause messages to arrive out of expected sequence. For example, a peer might send a `request` message just before receiving our `choke` message, requiring graceful handling of this race condition.

### Request Pipelining

Request pipelining allows BitTorrent clients to maintain multiple outstanding block requests simultaneously, dramatically improving download throughput by overlapping network round-trip times with data transmission. Without pipelining, the client would wait for each 16KB block to arrive before requesting the next one, severely limiting throughput on high-latency connections.

The pipelining system must carefully balance throughput optimization with resource management and fairness considerations. Too few outstanding requests waste bandwidth due to round-trip delays, while too many requests consume excessive memory and can overwhelm slower peers.

**Request Pipeline Architecture:**

Each peer connection maintains a request pipeline that tracks outstanding requests and manages the flow of new requests based on network conditions and peer capabilities:

| Pipeline Component | Purpose | Typical Size |
|--------------------|---------|--------------|
| Outstanding Requests | Map of sent but not received requests | 5-10 requests |
| Request Queue | Prepared requests waiting to send | 20-50 requests |
| Receive Buffer | Arriving piece data being assembled | 16KB per request |
| Timeout Tracker | Monitors request response times | Per-request timers |

**Request Message Format:**

Each request message specifies exactly which 16KB block is needed from a specific piece:

| Field | Length | Type | Description |
|-------|--------|------|-------------|
| Piece Index | 4 bytes | uint32 | Which piece contains the desired block |
| Block Offset | 4 bytes | uint32 | Byte offset within the piece |
| Block Length | 4 bytes | uint32 | Number of bytes requested (typically 16384) |

The request pipelining algorithm coordinates multiple concurrent block downloads while maintaining proper flow control:

**Pipeline Management Algorithm:**

1. **Pipeline Capacity Check**: Determine how many additional requests can be sent based on current outstanding count and peer capacity.

2. **Block Selection**: Choose the next needed blocks from the piece selection algorithm, prioritizing blocks from pieces that are already partially downloaded.

3. **Request Transmission**: Send request messages for selected blocks, recording each request with timestamp and expected response size.

4. **Response Processing**: When piece messages arrive, match them to outstanding requests, validate block data, and update pipeline state.

5. **Timeout Handling**: Periodically check for requests that have exceeded reasonable response times and re-request from other peers.

6. **Pipeline Refill**: Continuously add new requests to maintain optimal pipeline depth based on network conditions.

> **Decision: Adaptive Pipeline Depth**
> - **Context**: Need to optimize throughput while avoiding peer overload and memory exhaustion
> - **Options Considered**: Fixed pipeline depth vs. adaptive sizing vs. unlimited pipelining
> - **Decision**: Adaptive pipeline sizing based on peer performance and network conditions
> - **Rationale**: Fixed depth wastes opportunities with fast peers and overloads slow peers; adaptive sizing maximizes efficiency
> - **Consequences**: Requires performance monitoring and dynamic adjustment but provides optimal throughput across diverse network conditions

**Pipeline Depth Calculation:**

The optimal pipeline depth depends on the bandwidth-delay product of the connection and the peer's processing capacity:

1. **Bandwidth Estimation**: Monitor the peer's actual data delivery rate over recent time windows.

2. **Latency Measurement**: Track round-trip times between request messages and corresponding piece messages.

3. **Capacity Calculation**: Estimate how many blocks should be in-flight to maintain continuous data flow: `pipeline_depth = (bandwidth * round_trip_time) / block_size`.

4. **Peer Limits**: Respect any peer-advertised limits on concurrent requests and avoid overwhelming slower peers.

5. **Memory Constraints**: Cap pipeline depth based on available memory for buffering outstanding requests.

**Request Timeout and Recovery:**

Network issues and peer problems can cause some requests to be lost or delayed indefinitely. The pipeline must detect these situations and recover gracefully:

| Timeout Condition | Detection Method | Recovery Action |
|-------------------|------------------|-----------------|
| Slow Response | Request age > 2 * average RTT | Mark as slow, reduce pipeline depth |
| Lost Request | Request age > 60 seconds | Cancel and re-request from different peer |
| Peer Disconnect | TCP connection closed | Cancel all requests, redistribute to other peers |
| Invalid Response | Piece data fails validation | Re-request block, potential peer ban |

**Endgame Mode Considerations:**

When downloading the final pieces of a torrent, the standard rarest-first selection may result in only a few peers having the remaining pieces. In this situation, endgame mode modifies the pipelining strategy:

1. **Redundant Requests**: Send requests for the same blocks to multiple peers to avoid stalling on slow responses.

2. **Cancel Propagation**: When a block arrives, immediately send cancel messages to other peers for the same block to avoid wasted bandwidth.

3. **Aggressive Timeouts**: Use shorter timeout periods since completion is prioritized over bandwidth efficiency.

⚠️ **Pitfall: Memory Leaks in Pipeline Management**
Outstanding request tracking must carefully manage memory allocation and cleanup. Failed to cancel requests on peer disconnection leads to memory leaks, while failed to remove completed requests causes unbounded memory growth.

### Common Protocol Pitfalls

The BitTorrent peer wire protocol contains several subtle requirements and edge cases that frequently trip up implementers. These pitfalls often manifest as connection failures, poor performance, or protocol violations that cause other clients to disconnect.

⚠️ **Pitfall: Endianness Confusion**

**Problem**: All multi-byte integers in BitTorrent messages use big-endian (network) byte order, but many developers forget to convert between network and host byte order when parsing or constructing messages.

**Symptoms**: Message parsing fails with apparently random values, piece indices seem impossibly large, or length fields cause buffer overruns.

**Example**: A 4-byte piece index of `0x00000001` (piece 1) appears as `0x01000000` (piece 16777216) on little-endian systems without proper conversion.

**Fix**: Always use proper byte order conversion functions (`binary.BigEndian` in Go) when reading or writing multi-byte fields in protocol messages.

⚠️ **Pitfall: Partial TCP Read Handling**

**Problem**: Assuming that a single TCP `read()` call returns a complete message. TCP is a stream protocol and may deliver data in arbitrary chunks unrelated to message boundaries.

**Symptoms**: Messages appear truncated, parsing fails intermittently (especially under load), or the client works locally but fails with remote peers.

**Example**: A 17KB piece message might arrive as three separate reads of 8KB, 8KB, and 1KB, requiring accumulation across multiple read operations.

**Fix**: Always accumulate bytes until you have a complete message. Maintain parsing state across multiple read operations and only process complete messages.

⚠️ **Pitfall: Blocking on Choked Peers**

**Problem**: Sending request messages to peers who have choked us, or failing to cancel outstanding requests when a peer sends a choke message.

**Symptoms**: Download stalls even though peers are connected, requests time out repeatedly, or peers disconnect due to protocol violations.

**Example**: Peer sends choke message but client continues sending request messages, violating the protocol and causing the peer to close the connection.

**Fix**: Immediately cancel all outstanding requests when receiving a choke message, and only send new requests to peers who are not choking us.

⚠️ **Pitfall: Handshake Validation Ordering**

**Problem**: Performing handshake validation checks in the wrong order or with incorrect byte ranges, leading to connection establishment failures.

**Symptoms**: All incoming connections fail with protocol errors, connections work with some clients but not others, or info hash mismatches are reported.

**Example**: Reading the info hash from bytes 20-39 instead of bytes 28-47, causing validation to fail against random data from the reserved bytes field.

**Fix**: Follow the exact handshake format: 1 byte length + 19 bytes protocol string + 8 bytes reserved + 20 bytes info hash + 20 bytes peer ID.

⚠️ **Pitfall: State Machine Synchronization Errors**

**Problem**: Updating peer state flags inconsistently or failing to synchronize state changes with message transmission, leading to desynchronized state machines between peers.

**Symptoms**: Peers appear to be in impossible states, upload/download permissions don't match message exchange, or connections deadlock.

**Example**: Setting `am_choking = false` locally but forgetting to send the unchoke message, causing state desynchronization.

**Fix**: Always update local state and send corresponding messages atomically. Never change state flags without sending appropriate protocol messages.

⚠️ **Pitfall: Request Pipeline Memory Management**

**Problem**: Failing to properly cleanup outstanding requests when peers disconnect, or not limiting pipeline depth based on available memory.

**Symptoms**: Memory usage grows unboundedly during downloads, request timeouts don't trigger cleanup, or the client crashes with out-of-memory errors.

**Example**: Peer disconnects with 50 outstanding 16KB requests, but the request tracking structures are never cleaned up, leaking 800KB per disconnection.

**Fix**: Implement proper resource cleanup on peer disconnection and enforce reasonable limits on concurrent outstanding requests based on available memory.

⚠️ **Pitfall: Keep-Alive Timing Issues**

**Problem**: Not sending keep-alive messages during idle periods, or failing to reset connection timeouts when receiving keep-alive messages.

**Symptoms**: Connections drop unexpectedly during periods of no data transfer, peers disconnect after exactly 2-3 minutes of inactivity.

**Example**: During endgame phase when few requests are being sent, connections time out because no keep-alive messages are transmitted.

**Fix**: Send keep-alive messages (zero-length messages) every 2 minutes when no other messages have been sent, and reset timeout timers when receiving any message including keep-alives.

### Implementation Guidance

This subsection provides the practical foundation for implementing the BitTorrent peer wire protocol, focusing on TCP connection management, message parsing infrastructure, and state tracking systems that enable robust peer communication.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| TCP Networking | net.Conn with basic read/write | Custom connection pooling with bufio |
| Message Parsing | Manual byte slicing | Protocol buffer style parser |
| State Management | Simple struct with mutexes | Actor model with message channels |
| Concurrency | One goroutine per peer | Worker pool with shared connections |
| Error Handling | Basic error returns | Structured error types with recovery |

**Recommended File Structure:**

The peer protocol implementation should be organized to separate networking concerns from protocol logic:

```
internal/peer/
  connection.go           ← TCP connection management and handshake
  message.go             ← Message parsing and framing
  protocol.go            ← Peer state machine and protocol logic
  pipeline.go            ← Request pipelining and flow control
  connection_test.go     ← Connection and handshake tests
  message_test.go        ← Message parsing tests
  protocol_test.go       ← State machine and integration tests
internal/wire/
  messages.go            ← Message type definitions and constants
  framer.go             ← Message framing utilities
```

**Core Infrastructure - Complete Message Framing System:**

```go
package wire

import (
    "encoding/binary"
    "fmt"
    "io"
)

// Message type constants
const (
    MsgKeepAlive    = -1  // Special case for keep-alive (no ID)
    MsgChoke        = 0
    MsgUnchoke      = 1
    MsgInterested   = 2
    MsgNotInterested = 3
    MsgHave         = 4
    MsgBitfield     = 5
    MsgRequest      = 6
    MsgPiece        = 7
    MsgCancel       = 8
)

const (
    HANDSHAKE_LENGTH = 68
    PROTOCOL_STRING = "BitTorrent protocol"
    BLOCK_SIZE = 16384
    MAX_MESSAGE_LENGTH = BLOCK_SIZE + 13  // piece message with full block
)

// Message represents a parsed BitTorrent protocol message
type Message struct {
    ID      int
    Payload []byte
}

// MessageFramer handles reading and writing length-prefixed messages
type MessageFramer struct {
    conn   io.ReadWriter
    buffer []byte  // Reusable buffer for parsing
}

// NewMessageFramer creates a message framer for the given connection
func NewMessageFramer(conn io.ReadWriter) *MessageFramer {
    return &MessageFramer{
        conn:   conn,
        buffer: make([]byte, MAX_MESSAGE_LENGTH+4), // +4 for length prefix
    }
}

// ReadMessage reads and parses the next message from the connection
func (mf *MessageFramer) ReadMessage() (*Message, error) {
    // Read 4-byte length prefix
    if _, err := io.ReadFull(mf.conn, mf.buffer[:4]); err != nil {
        return nil, fmt.Errorf("reading message length: %w", err)
    }
    
    length := binary.BigEndian.Uint32(mf.buffer[:4])
    
    // Handle keep-alive message (length = 0)
    if length == 0 {
        return &Message{ID: MsgKeepAlive, Payload: nil}, nil
    }
    
    // Validate message length
    if length > MAX_MESSAGE_LENGTH {
        return nil, fmt.Errorf("message too large: %d bytes", length)
    }
    
    // Read message ID + payload
    if _, err := io.ReadFull(mf.conn, mf.buffer[:length]); err != nil {
        return nil, fmt.Errorf("reading message body: %w", err)
    }
    
    messageID := int(mf.buffer[0])
    payload := make([]byte, length-1)
    copy(payload, mf.buffer[1:length])
    
    return &Message{ID: messageID, Payload: payload}, nil
}

// WriteMessage writes a message to the connection with proper framing
func (mf *MessageFramer) WriteMessage(msg *Message) error {
    if msg.ID == MsgKeepAlive {
        // Keep-alive: just write 4 zero bytes
        binary.BigEndian.PutUint32(mf.buffer[:4], 0)
        _, err := mf.conn.Write(mf.buffer[:4])
        return err
    }
    
    messageLength := uint32(len(msg.Payload) + 1) // +1 for message ID
    binary.BigEndian.PutUint32(mf.buffer[:4], messageLength)
    mf.buffer[4] = byte(msg.ID)
    copy(mf.buffer[5:], msg.Payload)
    
    _, err := mf.conn.Write(mf.buffer[:4+messageLength])
    return err
}

// Handshake represents the initial 68-byte handshake message
type Handshake struct {
    InfoHash [20]byte
    PeerID   [20]byte
}

// MarshalBinary serializes handshake to 68-byte wire format
func (h *Handshake) MarshalBinary() []byte {
    buf := make([]byte, HANDSHAKE_LENGTH)
    buf[0] = 19  // Protocol string length
    copy(buf[1:20], PROTOCOL_STRING)
    // buf[20:28] reserved bytes (already zero)
    copy(buf[28:48], h.InfoHash[:])
    copy(buf[48:68], h.PeerID[:])
    return buf
}

// UnmarshalBinary parses 68-byte handshake from wire format
func (h *Handshake) UnmarshalBinary(data []byte) error {
    if len(data) != HANDSHAKE_LENGTH {
        return fmt.Errorf("invalid handshake length: %d", len(data))
    }
    
    if data[0] != 19 {
        return fmt.Errorf("invalid protocol length: %d", data[0])
    }
    
    if string(data[1:20]) != PROTOCOL_STRING {
        return fmt.Errorf("invalid protocol string: %s", data[1:20])
    }
    
    copy(h.InfoHash[:], data[28:48])
    copy(h.PeerID[:], data[48:68])
    return nil
}
```

**Core Infrastructure - Bitfield Operations:**

```go
package peer

// BitfieldOps provides efficient operations on piece availability bitfields
type BitfieldOps struct {
    bitfield  []byte
    numPieces int
}

// NewBitfieldOps creates a bitfield for the specified number of pieces
func NewBitfieldOps(numPieces int) *BitfieldOps {
    byteCount := (numPieces + 7) / 8  // Round up to nearest byte
    return &BitfieldOps{
        bitfield:  make([]byte, byteCount),
        numPieces: numPieces,
    }
}

// SetPiece marks a piece as available
func (bf *BitfieldOps) SetPiece(pieceIndex int) {
    if pieceIndex >= bf.numPieces {
        return
    }
    byteIndex := pieceIndex / 8
    bitIndex := pieceIndex % 8
    bf.bitfield[byteIndex] |= (0x80 >> bitIndex)
}

// HasPiece returns true if the specified piece is available
func (bf *BitfieldOps) HasPiece(pieceIndex int) bool {
    if pieceIndex >= bf.numPieces {
        return false
    }
    byteIndex := pieceIndex / 8
    bitIndex := pieceIndex % 8
    return (bf.bitfield[byteIndex] & (0x80 >> bitIndex)) != 0
}

// CountAvailablePieces returns the number of pieces marked as available
func (bf *BitfieldOps) CountAvailablePieces() int {
    count := 0
    for i := 0; i < bf.numPieces; i++ {
        if bf.HasPiece(i) {
            count++
        }
    }
    return count
}

// Bytes returns the raw bitfield bytes for transmission
func (bf *BitfieldOps) Bytes() []byte {
    result := make([]byte, len(bf.bitfield))
    copy(result, bf.bitfield)
    return result
}
```

**Core Logic Skeleton - Peer Connection Management:**

```go
package peer

import (
    "net"
    "sync"
    "time"
    "context"
)

// Connection represents a single peer connection with full state tracking
type Connection struct {
    conn     net.Conn
    framer   *wire.MessageFramer
    
    // Peer identification
    peerID   [20]byte
    infoHash [20]byte
    
    // State machine flags
    amChoking       bool
    amInterested    bool
    peerChoking     bool
    peerInterested  bool
    
    // Piece availability
    bitfield *BitfieldOps
    
    // Request pipeline
    pendingRequests map[string]*PendingRequest
    maxPipeline     int
    
    // Synchronization
    mutex sync.RWMutex
    
    // Lifecycle
    ctx    context.Context
    cancel context.CancelFunc
}

// PendingRequest tracks an outstanding block request
type PendingRequest struct {
    PieceIndex  int
    Offset      int
    Length      int
    RequestTime time.Time
}

// NewConnection creates a new peer connection
func NewConnection(conn net.Conn, infoHash [20]byte, peerID [20]byte) *Connection {
    ctx, cancel := context.WithCancel(context.Background())
    return &Connection{
        conn:            conn,
        framer:          wire.NewMessageFramer(conn),
        infoHash:        infoHash,
        peerID:          peerID,
        amChoking:       true,   // Start choked
        amInterested:    false,  // Start not interested
        peerChoking:     true,   // Assume peer starts choked
        peerInterested:  false,  // Assume peer starts not interested
        pendingRequests: make(map[string]*PendingRequest),
        maxPipeline:     5,      // Conservative initial pipeline depth
        ctx:             ctx,
        cancel:          cancel,
    }
}

// PerformHandshake executes the BitTorrent handshake protocol
func (c *Connection) PerformHandshake() error {
    // TODO 1: Create handshake message with our info hash and peer ID
    // TODO 2: Send handshake to peer using conn.Write()
    // TODO 3: Read 68-byte response using io.ReadFull()
    // TODO 4: Parse response handshake and validate protocol string
    // TODO 5: Verify info hash matches our torrent
    // TODO 6: Store peer's ID for future reference
    // Hint: Use wire.Handshake struct and its marshal/unmarshal methods
    panic("TODO: implement handshake")
}

// UpdateInterest updates our interested state based on peer's available pieces
func (c *Connection) UpdateInterest(neededPieces []int) error {
    c.mutex.Lock()
    defer c.mutex.Unlock()
    
    // TODO 1: Check if peer has any pieces we need using bitfield.HasPiece()
    // TODO 2: Determine new interested state (true if any needed pieces available)
    // TODO 3: If interest state changed, send appropriate message (interested/not interested)
    // TODO 4: Update amInterested flag to match new state
    // Hint: Only send message if state actually changes to avoid redundant traffic
    panic("TODO: implement interest updating")
}

// SendRequest sends a block request if peer is unchoked and pipeline has capacity
func (c *Connection) SendRequest(pieceIndex, offset, length int) error {
    c.mutex.Lock()
    defer c.mutex.Unlock()
    
    // TODO 1: Check if peer is choking us (peer_choking flag)
    // TODO 2: Check if we have pipeline capacity (len(pendingRequests) < maxPipeline)
    // TODO 3: Create request message with piece index, offset, length
    // TODO 4: Send request message using framer.WriteMessage()
    // TODO 5: Add request to pendingRequests map for tracking
    // TODO 6: Record request time for timeout detection
    // Hint: Use fmt.Sprintf("%d-%d-%d", pieceIndex, offset, length) as request key
    panic("TODO: implement request sending")
}

// ProcessMessage handles incoming messages from the peer
func (c *Connection) ProcessMessage(msg *wire.Message) error {
    switch msg.ID {
    case wire.MsgChoke:
        return c.handleChoke()
    case wire.MsgUnchoke:
        return c.handleUnchoke()
    case wire.MsgInterested:
        return c.handleInterested()
    case wire.MsgNotInterested:
        return c.handleNotInterested()
    case wire.MsgHave:
        return c.handleHave(msg.Payload)
    case wire.MsgBitfield:
        return c.handleBitfield(msg.Payload)
    case wire.MsgRequest:
        return c.handleRequest(msg.Payload)
    case wire.MsgPiece:
        return c.handlePiece(msg.Payload)
    case wire.MsgCancel:
        return c.handleCancel(msg.Payload)
    default:
        return fmt.Errorf("unknown message type: %d", msg.ID)
    }
}

// handleChoke processes incoming choke message
func (c *Connection) handleChoke() error {
    c.mutex.Lock()
    defer c.mutex.Unlock()
    
    // TODO 1: Set peerChoking flag to true
    // TODO 2: Cancel all pending requests (clear pendingRequests map)
    // TODO 3: Notify piece manager that requests were cancelled
    // Hint: Choking means peer won't fulfill our requests, so clean up pipeline
    panic("TODO: implement choke handling")
}

// handleUnchoke processes incoming unchoke message
func (c *Connection) handleUnchoke() error {
    c.mutex.Lock()
    defer c.mutex.Unlock()
    
    // TODO 1: Set peerChoking flag to false
    // TODO 2: Notify piece manager that we can now send requests
    // Hint: Unchoking enables request sending, but don't automatically send requests here
    panic("TODO: implement unchoke handling")
}

// CanDownload returns true if we can request blocks from this peer
func (c *Connection) CanDownload() bool {
    c.mutex.RLock()
    defer c.mutex.RUnlock()
    
    // TODO 1: Check that we're interested in peer's pieces (amInterested)
    // TODO 2: Check that peer is not choking us (peerChoking == false)
    // TODO 3: Return true only if both conditions are met
    panic("TODO: implement download capability check")
}
```

**Language-Specific Hints:**

- **TCP Connection Management**: Use `net.Dial()` for outgoing connections and `net.Listen()` for incoming connections. Set reasonable timeouts with `SetDeadline()` methods.

- **Binary Protocol Parsing**: Use `encoding/binary` package with `binary.BigEndian` for all multi-byte integer conversions. Use `io.ReadFull()` to ensure complete reads.

- **Concurrency**: Use separate goroutines for reading and writing messages. Protect shared state with `sync.RWMutex` - use read locks for queries and write locks for modifications.

- **Error Handling**: Distinguish between recoverable errors (temporary network issues) and fatal errors (protocol violations). Use context cancellation for graceful shutdown.

- **Memory Management**: Reuse message buffers where possible to reduce garbage collection pressure. Use buffer pools for frequently allocated/deallocated structures.

**Milestone Checkpoint:**

After implementing the peer wire protocol, verify the following behavior:

1. **Handshake Verification**: 
   ```bash
   go run cmd/client/main.go handshake <torrent-file> <peer-ip>:<peer-port>
   # Should output: "Handshake successful: peer ID <hex-encoded-peer-id>"
   ```

2. **Message Exchange Test**:
   - Connect to a peer and complete handshake
   - Receive and parse bitfield message
   - Send interested message if peer has needed pieces
   - Verify state transitions occur correctly

3. **Pipeline Test**:
   - Send multiple request messages to an unchoked peer
   - Verify requests are tracked in pending requests map
   - Process incoming piece messages and match to requests
   - Confirm pipeline refills automatically

**Expected Output Examples:**
```
INFO: Connecting to peer 192.168.1.100:6881
INFO: Handshake completed, peer ID: 2d5458333030302d787878787878787878
INFO: Received bitfield: peer has 847/1000 pieces
INFO: Sent interested message (peer has pieces we need)
INFO: Received unchoke message, can now download
INFO: Sent 5 requests, pipeline full
INFO: Received piece 0 block 0 (16384 bytes)
INFO: Pipeline refilled, sent request for piece 0 block 1
```

**Common Implementation Issues:**

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Handshake always fails | Wrong byte offsets in parsing | Use exact offsets: info hash at 28-47, peer ID at 48-67 |
| Messages appear corrupted | Missing endianness conversion | Use binary.BigEndian for all multi-byte fields |
| Download stalls after start | Sending requests to choked peers | Check peerChoking flag before sending requests |
| Memory usage grows continuously | Not cleaning up disconnected peers | Implement proper connection cleanup with context cancellation |
| Connections drop frequently | Missing keep-alive messages | Send keep-alive every 2 minutes during idle periods |


## Piece Management & Seeding (Milestone 4)

> **Milestone(s):** This section corresponds to Milestone 4 - Piece Management & Seeding, implementing piece verification, download scheduling, and upload capabilities for complete BitTorrent functionality.

The final milestone transforms our BitTorrent client from a simple peer protocol implementation into a fully functional file-sharing system. This phase introduces the sophisticated orchestration layer that coordinates concurrent downloads, verifies data integrity, schedules piece requests efficiently, and serves data to other peers. The challenge lies in managing the complex interplay between multiple concurrent peer connections while ensuring data correctness and optimal download performance.

### Piece Management as Jigsaw Puzzle

Think of piece management as coordinating a massive **collaborative jigsaw puzzle** where multiple people work together to assemble a picture, but with several critical constraints. Each person (peer) has different puzzle pieces available, some pieces might be damaged and need replacement, and you want to prioritize the rarest pieces first to ensure the puzzle can be completed even if people leave.

In this analogy, each **piece** represents a fixed-size chunk of the file with a unique verification code (SHA1 hash) that proves it's the correct piece. The **puzzle box** (torrent metadata) tells you exactly what each completed piece should look like. **Multiple helpers** (peers) each have different subsets of pieces, and you need to coordinate requests so that you're not asking the same person for pieces they don't have, while ensuring you get the rarest pieces before common ones.

The key insight is that unlike a physical puzzle, digital pieces can be **copied** rather than moved, so successful completion helps everyone else by making pieces more available. However, pieces can also be **corrupted** in transit, requiring verification against the expected hash before acceptance.

This mental model captures the essential challenges: **availability tracking** (knowing who has which pieces), **request coordination** (avoiding duplicate work), **rarity-based prioritization** (getting scarce pieces first), **integrity verification** (ensuring pieces aren't corrupted), and **reciprocal sharing** (becoming a piece source for others).

> **Decision: Piece-Centric vs Block-Centric Management**
> - **Context**: BitTorrent transfers data in 16KB blocks, but verifies integrity at the piece level (typically 256KB-1MB). We must decide whether our primary abstraction tracks pieces or individual blocks.
> - **Options Considered**: Block-centric tracking (fine-grained but complex), Piece-centric tracking (simpler but less granular), Hybrid approach (piece state with block progress)
> - **Decision**: Piece-centric management with internal block tracking
> - **Rationale**: Pieces are the unit of verification and sharing, making them the natural abstraction. Block-level details are implementation concerns within piece management.
> - **Consequences**: Simpler state management and clear verification boundaries, but requires careful block coordination within pieces to avoid request conflicts.

| Management Approach | Granularity | State Complexity | Verification Model | Request Coordination |
|---------------------|-------------|------------------|--------------------|--------------------|
| Block-Centric | 16KB blocks | High (thousands of states) | Deferred until piece complete | Complex inter-block dependencies |
| Piece-Centric | 256KB-1MB pieces | Medium (hundreds of states) | Natural verification unit | Clear piece-level coordination |
| Hybrid | Pieces with block progress | Medium-High | Piece-level verification | Block coordination within pieces |

### Content Verification

Content verification forms the **trust foundation** of the entire BitTorrent system. Unlike centralized downloads where you trust the server, peer-to-peer networks require cryptographic proof that received data hasn't been corrupted or maliciously altered. Every piece must pass SHA1 hash verification before being considered valid and available for sharing to other peers.

The verification process operates as a **cryptographic gatekeeper** where each completed piece undergoes mandatory hash computation and comparison against the expected value from the torrent metadata. This verification serves multiple purposes: detecting network transmission errors, identifying malicious peers sending corrupt data, ensuring file integrity across the distributed network, and preventing the propagation of corrupted data to other peers.

**Verification Algorithm:**
1. Accumulate all blocks belonging to a piece until the piece is complete (typically 16-64 blocks per piece)
2. Concatenate the block data in correct offset order to reconstruct the complete piece
3. Compute the SHA1 hash of the reconstructed piece data using a cryptographic hash function
4. Compare the computed hash against the expected hash from the torrent's piece list
5. If hashes match, mark the piece as verified and available for sharing; if they don't match, discard the piece data and re-request all blocks
6. Update the local bitfield to reflect piece availability and notify connected peers of the new piece
7. Write the verified piece data to the appropriate file offset on disk

> The critical insight is that verification happens at piece boundaries, not block boundaries. Blocks are network transfer units, but pieces are integrity units. A single corrupted block invalidates the entire piece.

**Hash Verification State Machine:**

| Current State | Event | Next State | Actions Taken |
|---------------|-------|------------|---------------|
| Downloading | All blocks received | Verifying | Concatenate blocks, compute SHA1 |
| Verifying | Hash matches expected | Complete | Write to disk, update bitfield, notify peers |
| Verifying | Hash mismatch | Failed | Discard data, reset block states, ban peer |
| Failed | Retry requested | Downloading | Clear piece state, re-request blocks |
| Complete | Seed request received | Complete | Serve piece data from disk |

⚠️ **Pitfall: Partial Piece Verification**
Many implementations attempt to verify pieces before all blocks are received, or try to verify individual blocks. This is fundamentally wrong because SHA1 hashes are computed over complete pieces. Attempting early verification leads to false negatives and wasted computation. Always wait for complete piece assembly before verification.

⚠️ **Pitfall: Memory vs Disk Trade-offs**
Holding multiple complete pieces in memory during verification can consume significant RAM (megabytes per piece). However, writing unverified data to disk risks file corruption. The solution is careful memory management: verify pieces immediately upon completion and implement memory pressure handling by prioritizing verification of the oldest complete pieces first.

**Corruption Handling Strategy:**

When piece verification fails, the client must implement a sophisticated response that balances network efficiency with peer reputation management. The failure could indicate a transmission error (temporary) or a malicious peer (permanent). Our approach prioritizes data integrity while maintaining reasonable download performance.

| Failure Scenario | Detection Method | Response Strategy | Peer Impact |
|-------------------|------------------|-------------------|-------------|
| Single block corruption | Hash mismatch | Re-request all piece blocks from different peers | Mark suspect peer, don't ban immediately |
| Repeated failures from peer | Multiple hash failures | Ban peer, close connection | Add to peer blacklist |
| Systematic corruption | Hash failures across multiple pieces | Re-request from different peer set | Evaluate peer selection algorithm |
| Network transmission error | Occasional random failures | Retry with exponential backoff | No peer penalty |

### Piece Selection Strategy

Piece selection determines download efficiency and swarm health through sophisticated algorithms that balance individual download speed with overall network resilience. The primary strategy, **rarest-first selection**, prioritizes pieces held by the fewest peers, ensuring that scarce content remains available even as peers leave the swarm.

Think of rarest-first as **digital archaeology** where you prioritize acquiring the most fragile artifacts first. If a piece is only held by one peer, losing that peer means the entire swarm loses access to that piece, potentially making the file incompletable. By prioritizing rare pieces, we improve overall swarm health while ensuring our own download can complete.

**Rarest-First Algorithm:**
1. For each connected peer, maintain a bitfield indicating which pieces they possess
2. For each piece we still need, count how many connected peers possess it (availability count)
3. Sort needed pieces by ascending availability count, breaking ties by piece index
4. When selecting the next piece to request, choose the piece with the lowest availability count that we haven't started downloading
5. If multiple pieces have the same (lowest) availability, prefer pieces closer to the beginning of the file for sequential access benefits
6. Once a piece is selected, immediately begin requesting blocks from peers that have the piece

> The fundamental insight is that rarest-first is an **insurance policy** for the swarm. By ensuring rare pieces get distributed quickly, we prevent scenarios where the swarm becomes incomplete due to peer departure.

**Availability Tracking Data Structures:**

| Data Structure | Purpose | Update Triggers | Performance Characteristics |
|----------------|---------|-----------------|---------------------------|
| Piece availability map | Maps piece index to peer count | Peer connect/disconnect, bitfield/have messages | O(1) lookup, O(n) update on peer change |
| Needed piece priority queue | Sorted list of pieces by rarity | Piece completion, availability changes | O(log n) insertion, O(1) top access |
| Peer piece bitfields | Which pieces each peer has | Initial bitfield, subsequent have messages | O(1) lookup per peer, O(m) storage per peer |
| In-progress piece set | Pieces currently being downloaded | Piece selection, completion, failure | O(1) membership test, prevents duplicate requests |

**Advanced Selection Strategies:**

Beyond basic rarest-first, sophisticated clients implement additional strategies for different phases of the download process:

**Random First Phase:** For the initial pieces, some clients use random selection rather than rarest-first. This provides faster startup by avoiding competition for the same rare pieces among new peers in the swarm. Random first typically applies to the first 4-8 pieces.

**Endgame Mode:** When only a few pieces remain (typically less than 10% of total pieces), switch to endgame mode where all remaining blocks are requested from all available peers. This prevents the download from stalling on slow peers holding the last few pieces. Endgame mode requires careful duplicate request handling to avoid wasting bandwidth.

**Sequential Mode:** For streaming applications, sequential selection downloads pieces in file order rather than rarity order. This enables playback of media files before download completion but can harm swarm health by reducing piece diversity.

**Endgame Mode Algorithm:**
1. Trigger when fewer than `MAX_ENDGAME_PIECES` (typically 10-20) pieces remain incomplete
2. For each remaining block in any incomplete piece, send requests to ALL peers that have the corresponding piece
3. When any peer sends a block, immediately send cancel messages to all other peers that were requested for the same block
4. Continue until all pieces are complete, accepting the bandwidth overhead for guaranteed completion

| Selection Strategy | Phase | Priority Metric | Benefits | Drawbacks |
|--------------------|-------|-----------------|----------|-----------|
| Random First | Startup (first ~5 pieces) | Random selection | Fast startup, reduced competition | Ignores rarity, may hurt swarm |
| Rarest First | Main download | Ascending availability count | Improves swarm health, ensures completion | Can be slow for initial pieces |
| Endgame | Final ~10 pieces | Request from all available peers | Prevents stalling, guaranteed completion | High bandwidth overhead |
| Sequential | Streaming use cases | File offset order | Enables progressive playback | Poor swarm health, inefficient |

⚠️ **Pitfall: Availability Count Staleness**
Availability counts can become stale when peers disconnect without notification or when bitfield updates are missed. Stale counts lead to poor piece selection decisions. The solution is periodic availability recalculation and timeout-based peer connection validation. Implement a background task that recalculates availability counts every 60-120 seconds.

⚠️ **Pitfall: Endgame Mode Triggering**
Triggering endgame mode too early wastes bandwidth through duplicate requests. Triggering too late causes stalling on slow peers. Monitor both the absolute number of remaining pieces AND the download rate. If download rate drops below a threshold AND few pieces remain, consider endgame mode even if the piece count hasn't reached the normal threshold.

![Piece Selection Algorithm](./diagrams/piece-selection-flow.svg)

### Concurrent Download Management

Concurrent download management orchestrates multiple peer connections to maximize throughput while avoiding conflicts and ensuring efficient resource utilization. This involves coordinating piece requests across peers, managing block-level request pipelining, and handling the complex state synchronization required for safe concurrent access to shared data structures.

Think of concurrent download management as conducting a **symphony orchestra** where each musician (peer connection) plays their part (downloads specific pieces) while following the conductor's coordination (piece manager) to create harmonious music (efficient file download). Each musician has different capabilities (bandwidth, piece availability) and may occasionally miss notes (network failures), requiring real-time adaptation and coordination.

The primary challenge is **request coordination** - ensuring that multiple peers don't waste bandwidth downloading the same blocks while maintaining sufficient request pipelining to keep connections busy. This requires sophisticated state tracking and communication between peer connection handlers and the central piece manager.

**Concurrent Download Architecture:**

The system employs a **centralized coordination** model where a single `PieceManager` component orchestrates downloads across multiple `PeerConnection` instances. This design trades some efficiency for simplicity and correctness, avoiding the complex distributed coordination required in fully decentralized approaches.

| Component | Responsibilities | Concurrency Model | State Management |
|-----------|------------------|-------------------|------------------|
| PieceManager | Piece selection, request coordination, verification | Single goroutine with channels | Centralized piece state, availability tracking |
| PeerConnection | Block requests, message handling, peer state | One goroutine per peer | Local peer state, pending requests |
| RequestCoordinator | Block assignment, duplicate prevention | Shared state with mutexes | Block reservation map, timeout tracking |
| VerificationQueue | Hash computation, disk writes | Dedicated worker pool | Verification work queue, completion callbacks |

**Request Coordination Protocol:**

Effective request coordination prevents bandwidth waste while ensuring sufficient parallelism. The protocol operates through a reservation system where blocks are reserved before requests are sent, preventing duplicate downloads.

**Block Request Algorithm:**
1. `PeerConnection` requests work from `PieceManager` by sending a work request message
2. `PieceManager` selects the highest-priority piece that the peer has and we need
3. `PieceManager` identifies unreserved blocks within the selected piece (blocks not currently being downloaded by other peers)
4. Reserve up to `MAX_PIPELINE_DEPTH` blocks for this peer, marking them as "reserved" with peer ID and timestamp
5. Return block list to `PeerConnection` for immediate request transmission
6. `PeerConnection` sends BitTorrent request messages to the peer for each reserved block
7. Track pending requests with timeouts; if no response within `BLOCK_TIMEOUT`, return blocks to unreserved state
8. When block data arrives, forward to `PieceManager` for assembly and potential verification

| Request State | Meaning | Timeout | Transition Triggers |
|---------------|---------|---------|-------------------|
| Available | Block can be requested | N/A | New piece selected, block request timeout |
| Reserved | Block assigned to peer | 30-60 seconds | Block request sent to peer |
| Requested | Request sent to peer | 120-180 seconds | Block data received, peer disconnect |
| Received | Data received, awaiting assembly | N/A | All piece blocks received |

**Synchronization and Locking Strategy:**

Concurrent access to shared piece state requires careful synchronization to prevent race conditions while maintaining performance. The design uses a combination of channels for coordination and fine-grained locking for shared data structures.

> **Decision: Channel-Based vs Mutex-Based Coordination**
> - **Context**: Multiple goroutines need coordinated access to piece state, request queues, and peer availability information.
> - **Options Considered**: Pure channel-based CSP model, Pure mutex-based shared memory, Hybrid approach with channels for coordination and mutexes for data
> - **Decision**: Hybrid approach with channels for major coordination and fine-grained mutexes for shared data structures
> - **Rationale**: Channels provide clean coordination semantics for work distribution, while mutexes offer efficient access to frequently-read data like bitfields and availability maps.
> - **Consequences**: Simpler reasoning about coordination flows but requires discipline to avoid deadlocks between channel operations and mutex acquisition.

**Concurrency Synchronization Points:**

| Shared Resource | Protection Mechanism | Access Pattern | Lock Granularity |
|------------------|---------------------|----------------|------------------|
| Piece availability map | Read-write mutex | Frequent reads, rare writes | Per-piece or global RWMutex |
| Block reservation map | Mutex | Moderate read/write | Per-piece or global mutex |
| Peer bitfields | Read-write mutex | Frequent reads during selection | Per-peer RWMutex |
| Piece completion status | Atomic operations | High-frequency status checks | Lock-free with atomic.Bool |
| Download statistics | Atomic operations | Frequent updates, periodic reads | Lock-free counters |

⚠️ **Pitfall: Lock Ordering and Deadlocks**
When multiple locks must be acquired, inconsistent ordering leads to deadlocks. Establish a clear lock hierarchy: always acquire piece-level locks before peer-level locks, and never hold a peer lock while waiting for a piece lock. Use timeouts on lock acquisition in critical paths to detect potential deadlocks.

⚠️ **Pitfall: Request Pipeline Starvation**
If piece selection is too slow or block reservation contention is high, peer connections can starve for work, leading to underutilized bandwidth. Implement request pipeline depth monitoring and preemptive block reservation to ensure each peer connection always has 3-5 outstanding requests.

![Multi-Peer Download Coordination](./diagrams/concurrent-download.svg)

### Upload and Seeding

Upload and seeding capabilities transform our client from a simple downloader into a full participant in the BitTorrent ecosystem. This involves serving piece data to other peers, implementing fair sharing policies, tracking upload statistics for tracker reporting, and maintaining connections specifically for seeding purposes.

Think of seeding as **running a specialized library** where you've completed collecting a particular book series and now help other collectors find the volumes they need. You need to track which books you have, respond to requests efficiently, maintain fair lending policies (not always serving the same frequent borrower), and keep statistics on how much you've helped the community.

The seeding process requires **role reversal** from the downloading logic - instead of selecting pieces to request, you respond to incoming requests by serving data from your completed files. However, the underlying mechanisms (message handling, connection management, piece verification) remain largely the same.

**Upload Request Handling:**

When acting as a seeder, the client receives `MsgRequest` messages from peers asking for specific blocks. The upload handler must validate requests, check local piece availability, read data from disk, and respond with `MsgPiece` messages containing the requested data.

**Upload Response Algorithm:**
1. Receive `MsgRequest` message containing piece index, block offset, and block length
2. Validate request parameters: piece index within torrent bounds, offset aligned to block boundaries, length not exceeding maximum block size
3. Check local bitfield to confirm we have the requested piece (respond with rejection if not available)
4. Verify peer is not choked - if choked, ignore the request (peer should not have sent it)
5. Read requested data from the appropriate file offset on disk (piece_index × piece_length + block_offset)
6. Construct `MsgPiece` response with piece index, block offset, and data payload
7. Send response to peer and update upload statistics (bytes uploaded, pieces served)
8. Apply bandwidth limiting if configured to prevent overwhelming network connection

| Request Validation | Check | Failure Response | Security Implication |
|-------------------|-------|------------------|---------------------|
| Piece bounds | `0 ≤ piece_index < total_pieces` | Ignore request | Prevents buffer overruns |
| Block offset alignment | `offset % BLOCK_SIZE == 0` | Ignore request | Ensures proper block boundaries |
| Block length limits | `length ≤ BLOCK_SIZE` | Ignore request | Prevents excessive memory allocation |
| Peer choke state | `!peer_choking` | Ignore request | Enforces BitTorrent protocol |
| Local piece availability | `HasPiece(piece_index)` | Ignore or send rejection | Prevents serving unavailable data |

**Choking and Unchoking Policy:**

BitTorrent's choking mechanism implements **tit-for-tat reciprocity** where peers prioritize uploading to those who provide good download speeds in return. This creates incentives for mutual cooperation while preventing pure "leeching" behavior that would harm the swarm.

**Standard Choking Algorithm:**
1. Maintain upload rate statistics for each connected peer over the last 20-30 seconds
2. Every 10 seconds (choking round), recalculate which peers to unchoke based on their download rates to us
3. Unchoke the top 3-4 peers providing the best download rates (reciprocal unchoking)
4. Additionally, unchoke one random peer every 30 seconds (optimistic unchoking) to discover new high-speed peers
5. Send `MsgChoke` to peers being choked and `MsgUnchoke` to peers being unchoked
6. Peers that are choked should not send requests; ignore any requests from choked peers

| Choking Category | Selection Criteria | Update Frequency | Purpose |
|------------------|-------------------|------------------|---------|
| Reciprocal Unchoke | Top 3-4 peers by download rate to us | Every 10 seconds | Reward peers providing good service |
| Optimistic Unchoke | Random peer selection | Every 30 seconds | Discover new high-speed peers |
| Seed Optimistic | Random among interested peers | Every 30 seconds | Distribute pieces when we're seeding |
| Anti-snubbing | Peers not choking us | When snubbed | Maintain reciprocity relationships |

**Seeding-Specific Adaptations:**

When seeding (upload-only mode after download completion), the choking algorithm adapts since there are no download rates to reciprocate. Seeding policies focus on **fairness** and **swarm health** rather than direct reciprocity.

**Seeding Upload Policy:**
1. Prefer peers with the lowest upload rates (helping slower peers)
2. Rotate unchoked peers regularly to ensure fair access across the swarm
3. Prioritize peers requesting rare pieces to improve swarm piece distribution
4. Implement bandwidth limits to prevent seeding from overwhelming the connection
5. Consider peer reputation and protocol compliance when making unchoking decisions

⚠️ **Pitfall: Upload Without Download Tracking**
When seeding, it's tempting to remove download rate tracking since we're not downloading. However, proper reciprocity tracking is essential for effective peer relationships. Maintain download statistics even while seeding - they inform which peers to prioritize and help identify protocol violations.

⚠️ **Pitfall: Disk I/O Performance**
Serving upload requests requires frequent disk reads at random offsets, which can become a performance bottleneck with many concurrent requesters. Implement an LRU piece cache in memory to serve frequently requested pieces without disk access. Monitor disk I/O patterns and consider read-ahead strategies for sequential requests.

**Upload Statistics and Tracker Reporting:**

Accurate upload statistics are essential for tracker communication and monitoring seeding effectiveness. The client must track bytes uploaded, pieces served, and upload rates for both local monitoring and tracker reporting.

| Upload Metric | Tracking Granularity | Reporting Frequency | Purpose |
|---------------|---------------------|-------------------|---------|
| Total bytes uploaded | Global counter | Every tracker announce | Tracker reporting, ratio monitoring |
| Upload rate (current) | 30-second rolling average | Continuous | Choking decisions, bandwidth monitoring |
| Pieces served | Per-piece counters | On piece completion | Swarm health analysis |
| Peer upload rates | Per-peer statistics | 20-second windows | Reciprocity calculations |
| Bandwidth utilization | Connection-level monitoring | Real-time | Bandwidth limiting, QoS |

### Common Piece Management Pitfalls

Piece management introduces complex state coordination and timing challenges that frequently trip up implementations. Understanding these common pitfalls helps avoid subtle bugs that can lead to download failures, data corruption, or poor performance.

⚠️ **Pitfall: Race Conditions in Piece Completion**
Multiple peer connections may simultaneously complete different blocks of the same piece, leading to race conditions during piece assembly and verification. The symptom is pieces being verified multiple times or verification failing due to incomplete piece data.

**Problem:** Two peers send the final blocks of a piece simultaneously. Both connections detect piece completion and attempt verification concurrently, but one connection may start verification before the other's block is incorporated.

**Solution:** Use atomic piece completion detection with compare-and-swap operations. Only allow one goroutine to transition a piece from "downloading" to "verifying" state. Implement piece-level mutexes for assembly operations.

```go
// Example of atomic piece completion
if atomic.CompareAndSwapInt32(&piece.state, StateDownloading, StateVerifying) {
    // Only one goroutine will execute this verification
    go verifyPiece(piece)
}
```

⚠️ **Pitfall: Memory Leaks from Uncompleted Pieces**
Pieces that never complete (due to peer disconnections or failures) can accumulate partial block data in memory indefinitely, leading to memory leaks in long-running clients.

**Problem:** Peers disconnect while downloading pieces, leaving partially completed pieces with allocated block buffers that are never freed because the piece never enters the cleanup path.

**Solution:** Implement periodic piece cleanup that identifies pieces stuck in downloading state for extended periods. Add reference counting for block data and timeout-based cleanup for abandoned pieces.

⚠️ **Pitfall: Inefficient Piece Selection Updates**
Recalculating piece priorities after every peer connection change or piece completion can become a performance bottleneck with large torrents and many peers.

**Problem:** With 1000+ piece torrents and dozens of peers, naively recalculating the entire piece priority queue after each bitfield update results in O(n log n) operations that can consume significant CPU time.

**Solution:** Implement incremental priority updates that only recalculate affected pieces. Use efficient data structures like priority queues that support decrease-key operations for targeted updates.

⚠️ **Pitfall: Verification CPU Blocking**
Computing SHA1 hashes for large pieces (1MB+) in the main goroutine blocks other operations, leading to poor responsiveness and connection timeouts.

**Problem:** SHA1 computation for large pieces can take several milliseconds, during which the piece manager cannot respond to new requests or handle peer messages, causing apparent hangs.

**Solution:** Perform verification in a dedicated worker pool. Use channels to queue verification work and return results asynchronously. Implement verification priority queues to prioritize pieces needed for immediate requests.

⚠️ **Pitfall: Endgame Mode Bandwidth Waste**
Poorly implemented endgame mode can waste enormous bandwidth by continuing to request blocks that have already been received from faster peers.

**Problem:** Endgame mode sends duplicate requests to all available peers but fails to promptly cancel requests when blocks arrive, leading to multiple peers sending the same block data.

**Solution:** Implement aggressive request cancellation in endgame mode. When any block arrives, immediately send cancel messages to all other peers that were sent requests for that block. Track which peers have been sent cancel messages to avoid redundant cancellations.

⚠️ **Pitfall: Disk I/O Concurrency Issues**
Multiple goroutines attempting to write different pieces to the same file simultaneously can lead to corruption or poor performance due to seek thrashing.

**Problem:** Concurrent piece writes to random file offsets cause excessive disk seeking and potential data races if file operations aren't properly synchronized.

**Solution:** Implement a disk I/O coordinator that serializes writes to the same file. Use file-level mutexes or dedicated I/O worker goroutines per file. Consider write coalescing for pieces that map to the same file regions.

| Pitfall Category | Primary Symptom | Detection Method | Prevention Strategy |
|------------------|-----------------|------------------|-------------------|
| Race Conditions | Verification failures, corrupt pieces | Stress testing with many peers | Atomic state transitions, piece-level locking |
| Memory Leaks | Growing memory usage | Memory profiling, leak detection | Timeout-based cleanup, reference counting |
| Performance Issues | High CPU usage, slow response | CPU profiling, latency monitoring | Incremental updates, async operations |
| Protocol Violations | Peer disconnections, bandwidth waste | Network traffic analysis | Strict protocol compliance, proper cancellation |

### Implementation Guidance

This implementation guidance provides concrete Go code for building the piece management and seeding system. The focus is on providing complete infrastructure components and detailed skeletons for the core learning algorithms.

**A. Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Hash Computation | `crypto/sha1` with `io.Copy` | Worker pool with `crypto/sha1` for parallelism |
| Disk I/O | `os.File` with `Seek` and `Read/Write` | Memory-mapped files with `syscall.Mmap` |
| Concurrency Coordination | Channels with `select` statements | `sync.WaitGroup` and `context.Context` trees |
| State Management | Maps with `sync.RWMutex` | Lock-free structures with `sync/atomic` |
| Priority Queues | `container/heap` with custom types | Third-party libraries like `github.com/emirpasic/gods` |

**B. Recommended File Structure:**

```
internal/
  piece/
    manager.go           ← PieceManager and coordination logic
    state.go            ← PieceState and Block data structures  
    verification.go     ← Hash verification worker pool
    selection.go        ← Piece selection algorithms
    upload.go           ← Upload handling and seeding logic
    manager_test.go     ← Unit tests for piece management
  download/
    coordinator.go      ← Multi-peer download coordination
    endgame.go         ← Endgame mode implementation
    statistics.go      ← Upload/download statistics tracking
```

**C. Infrastructure Code - Verification Worker Pool:**

```go
// internal/piece/verification.go
package piece

import (
    "crypto/sha1"
    "fmt"
    "sync"
)

// VerificationRequest represents a piece that needs hash verification
type VerificationRequest struct {
    PieceIndex int
    Data       []byte
    ExpectedHash [20]byte
    Callback   func(pieceIndex int, verified bool)
}

// VerificationPool manages concurrent piece verification
type VerificationPool struct {
    workers    int
    requests   chan VerificationRequest
    wg         sync.WaitGroup
    shutdown   chan struct{}
}

// NewVerificationPool creates a verification worker pool
func NewVerificationPool(workers int) *VerificationPool {
    pool := &VerificationPool{
        workers:  workers,
        requests: make(chan VerificationRequest, workers*2),
        shutdown: make(chan struct{}),
    }
    
    // Start worker goroutines
    for i := 0; i < workers; i++ {
        pool.wg.Add(1)
        go pool.worker()
    }
    
    return pool
}

// SubmitVerification queues a piece for verification
func (p *VerificationPool) SubmitVerification(req VerificationRequest) {
    select {
    case p.requests <- req:
        // Successfully queued
    case <-p.shutdown:
        // Pool is shutting down
        req.Callback(req.PieceIndex, false)
    }
}

// worker processes verification requests
func (p *VerificationPool) worker() {
    defer p.wg.Done()
    
    for {
        select {
        case req := <-p.requests:
            verified := p.verifyPiece(req.Data, req.ExpectedHash)
            req.Callback(req.PieceIndex, verified)
        case <-p.shutdown:
            return
        }
    }
}

// verifyPiece computes SHA1 and compares with expected hash
func (p *VerificationPool) verifyPiece(data []byte, expected [20]byte) bool {
    computed := sha1.Sum(data)
    return computed == expected
}

// Shutdown gracefully stops the verification pool
func (p *VerificationPool) Shutdown() {
    close(p.shutdown)
    p.wg.Wait()
    close(p.requests)
}
```

**D. Infrastructure Code - Priority Queue for Piece Selection:**

```go
// internal/piece/selection.go
package piece

import (
    "container/heap"
    "sync"
)

// PiecePriority represents a piece with its priority for selection
type PiecePriority struct {
    Index        int
    Availability int  // Lower is higher priority (rarest first)
    Priority     int  // Manual priority adjustment
}

// PriorityQueue implements heap.Interface for piece selection
type PriorityQueue []*PiecePriority

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
    // Primary: lowest availability (rarest first)
    if pq[i].Availability != pq[j].Availability {
        return pq[i].Availability < pq[j].Availability
    }
    // Secondary: manual priority
    if pq[i].Priority != pq[j].Priority {
        return pq[i].Priority > pq[j].Priority
    }
    // Tertiary: lowest index (sequential preference)
    return pq[i].Index < pq[j].Index
}

func (pq PriorityQueue) Swap(i, j int) { pq[i], pq[j] = pq[j], pq[i] }

func (pq *PriorityQueue) Push(x interface{}) {
    *pq = append(*pq, x.(*PiecePriority))
}

func (pq *PriorityQueue) Pop() interface{} {
    old := *pq
    n := len(old)
    item := old[n-1]
    *pq = old[0 : n-1]
    return item
}

// PieceSelector manages piece selection with thread-safe priority queue
type PieceSelector struct {
    queue     PriorityQueue
    available map[int]int  // piece index -> availability count
    needed    map[int]bool // pieces we still need
    mutex     sync.RWMutex
}

// NewPieceSelector creates a piece selector for the given torrent
func NewPieceSelector(totalPieces int) *PieceSelector {
    return &PieceSelector{
        queue:     make(PriorityQueue, 0),
        available: make(map[int]int),
        needed:    make(map[int]bool),
    }
}

// UpdateAvailability updates piece availability from peer bitfields
func (ps *PieceSelector) UpdateAvailability(pieceIndex int, available bool) {
    ps.mutex.Lock()
    defer ps.mutex.Unlock()
    
    if available {
        ps.available[pieceIndex]++
    } else {
        ps.available[pieceIndex]--
        if ps.available[pieceIndex] <= 0 {
            delete(ps.available, pieceIndex)
        }
    }
    
    // Rebuild priority queue if needed
    ps.rebuildQueue()
}

// SelectNextPiece returns the highest priority piece we need
func (ps *PieceSelector) SelectNextPiece(peerBitfield *BitfieldOps) int {
    ps.mutex.RLock()
    defer ps.mutex.RUnlock()
    
    // Find highest priority piece that peer has and we need
    for ps.queue.Len() > 0 {
        piece := heap.Pop(&ps.queue).(*PiecePriority)
        
        if ps.needed[piece.Index] && peerBitfield.HasPiece(piece.Index) {
            return piece.Index
        }
    }
    
    return -1 // No suitable piece found
}

// MarkComplete removes a piece from needed set
func (ps *PieceSelector) MarkComplete(pieceIndex int) {
    ps.mutex.Lock()
    defer ps.mutex.Unlock()
    
    delete(ps.needed, pieceIndex)
    ps.rebuildQueue()
}

// rebuildQueue reconstructs priority queue (call with mutex held)
func (ps *PieceSelector) rebuildQueue() {
    ps.queue = ps.queue[:0]
    
    for pieceIndex := range ps.needed {
        availability := ps.available[pieceIndex]
        piece := &PiecePriority{
            Index:        pieceIndex,
            Availability: availability,
            Priority:     0, // Default priority
        }
        heap.Push(&ps.queue, piece)
    }
}
```

**E. Core Logic Skeleton - Piece Manager:**

```go
// internal/piece/manager.go
package piece

import (
    "context"
    "sync"
    "time"
)

// PieceManager coordinates piece downloads, verification, and serving
type PieceManager struct {
    metaInfo        *torrent.MetaInfo
    pieces          map[int]*PieceState
    selector        *PieceSelector
    verificationPool *VerificationPool
    statistics      *Statistics
    mutex           sync.RWMutex
    
    // Channels for coordination
    workRequests    chan WorkRequest
    blockData       chan BlockData
    verificationResults chan VerificationResult
}

// WorkRequest represents a peer asking for work
type WorkRequest struct {
    PeerID    string
    Bitfield  *BitfieldOps
    Response  chan WorkResponse
}

// WorkResponse contains blocks assigned to a peer
type WorkResponse struct {
    Blocks []BlockRequest
    Error  error
}

// BlockRequest represents a specific block to download
type BlockRequest struct {
    PieceIndex int
    Offset     int
    Length     int
}

// NewPieceManager creates a piece manager for the given torrent
func NewPieceManager(metaInfo *torrent.MetaInfo) *PieceManager {
    // TODO 1: Initialize piece state map with one PieceState per piece
    // TODO 2: Create piece selector with total piece count
    // TODO 3: Initialize verification pool with appropriate worker count
    // TODO 4: Set up communication channels with proper buffer sizes
    // TODO 5: Start main coordination goroutine
    // Hint: Use len(metaInfo.Info.GetPieceHashes()) for piece count
    panic("implement NewPieceManager")
}

// RequestWork assigns blocks to a peer for downloading
func (pm *PieceManager) RequestWork(peerID string, peerBitfield *BitfieldOps) ([]BlockRequest, error) {
    // TODO 1: Create WorkRequest with peer information
    // TODO 2: Send request on workRequests channel
    // TODO 3: Wait for response on the response channel
    // TODO 4: Return assigned blocks or error
    // Hint: Use buffered response channel to avoid goroutine leaks
    panic("implement RequestWork")
}

// SubmitBlockData processes received block data
func (pm *PieceManager) SubmitBlockData(peerID string, pieceIndex, offset int, data []byte) error {
    // TODO 1: Validate block parameters (bounds checking)
    // TODO 2: Create BlockData message
    // TODO 3: Send on blockData channel for processing
    // TODO 4: Update download statistics
    // Hint: Validate offset + len(data) doesn't exceed piece bounds
    panic("implement SubmitBlockData")
}

// ServeUploadRequest handles requests from other peers
func (pm *PieceManager) ServeUploadRequest(pieceIndex, offset, length int) ([]byte, error) {
    // TODO 1: Validate request parameters
    // TODO 2: Check if we have the requested piece
    // TODO 3: Read data from disk at appropriate offset
    // TODO 4: Update upload statistics
    // TODO 5: Return requested data
    // Hint: Calculate file offset as piece_index * piece_length + offset
    panic("implement ServeUploadRequest")
}

// coordinator is the main coordination goroutine
func (pm *PieceManager) coordinator(ctx context.Context) {
    for {
        select {
        case workReq := <-pm.workRequests:
            // TODO 1: Use piece selector to find next piece for this peer
            // TODO 2: Find unreserved blocks in the selected piece
            // TODO 3: Reserve blocks for this peer with timeout
            // TODO 4: Create BlockRequest list
            // TODO 5: Send response on workReq.Response channel
            // Hint: Limit blocks per peer to MAX_PIPELINE_DEPTH (5-10)
            
        case blockData := <-pm.blockData:
            // TODO 1: Find the piece state for this block
            // TODO 2: Add block data to piece assembly buffer
            // TODO 3: Check if piece is now complete (all blocks received)
            // TODO 4: If complete, submit for verification
            // TODO 5: Update block reservation status
            // Hint: Use atomic operations for piece completion detection
            
        case verifyResult := <-pm.verificationResults:
            // TODO 1: Check verification result (hash match)
            // TODO 2: If verified, mark piece complete and write to disk
            // TODO 3: If failed, reset piece state for re-download
            // TODO 4: Update bitfield and notify connected peers
            // TODO 5: Update piece selector with completion status
            // Hint: Send Have messages to all connected peers for verified pieces
            
        case <-ctx.Done():
            return
        }
    }
}

// selectBlocksForPeer finds unreserved blocks in a piece for download
func (pm *PieceManager) selectBlocksForPeer(pieceIndex int, maxBlocks int) []BlockRequest {
    // TODO 1: Get piece state for the specified piece
    // TODO 2: Iterate through blocks in the piece
    // TODO 3: Find blocks that are not reserved or completed
    // TODO 4: Reserve found blocks for this peer
    // TODO 5: Create BlockRequest list up to maxBlocks limit
    // Hint: Standard block size is 16KB except for the last block of a piece
    panic("implement selectBlocksForPeer")
}

// assemblePiece combines received blocks into complete piece data
func (pm *PieceManager) assemblePiece(pieceIndex int) []byte {
    // TODO 1: Get piece state and verify all blocks are present
    // TODO 2: Calculate total piece size (may be less than piece_length for last piece)
    // TODO 3: Create output buffer of appropriate size
    // TODO 4: Copy each block's data to correct offset in output buffer
    // TODO 5: Return assembled piece data
    // Hint: Sort blocks by offset before assembly to ensure correct order
    panic("implement assemblePiece")
}
```

**F. Core Logic Skeleton - Endgame Mode:**

```go
// internal/download/endgame.go
package download

import (
    "sync"
    "time"
)

// EndgameManager handles aggressive downloading of final pieces
type EndgameManager struct {
    pieceManager   *piece.PieceManager
    connections    map[string]*peer.Connection
    activeRequests map[BlockKey][]string  // block -> list of peers requested from
    mutex          sync.RWMutex
    enabled        bool
}

// BlockKey uniquely identifies a block
type BlockKey struct {
    PieceIndex int
    Offset     int
}

// NewEndgameManager creates endgame mode coordinator
func NewEndgameManager(pm *piece.PieceManager) *EndgameManager {
    return &EndgameManager{
        pieceManager:   pm,
        connections:    make(map[string]*peer.Connection),
        activeRequests: make(map[BlockKey][]string),
    }
}

// ShouldEnterEndgame determines if endgame mode should be activated
func (em *EndgameManager) ShouldEnterEndgame() bool {
    // TODO 1: Count remaining incomplete pieces
    // TODO 2: Check current download rate trend
    // TODO 3: Calculate percentage of completion
    // TODO 4: Return true if conditions met for endgame
    // Hint: Typical threshold is <20 pieces remaining and >95% complete
    panic("implement ShouldEnterEndgame")
}

// RequestAllRemaining sends requests for all incomplete blocks to all available peers
func (em *EndgameManager) RequestAllRemaining() {
    // TODO 1: Get list of all incomplete pieces
    // TODO 2: For each piece, identify incomplete blocks
    // TODO 3: For each block, send requests to all peers that have the piece
    // TODO 4: Track all outstanding requests for cancellation
    // TODO 5: Set timeouts for endgame requests (shorter than normal)
    // Hint: Only request from peers that are not choking us
    panic("implement RequestAllRemaining")
}

// OnBlockReceived handles block completion in endgame mode
func (em *EndgameManager) OnBlockReceived(pieceIndex, offset int, fromPeer string) {
    // TODO 1: Create BlockKey for the received block
    // TODO 2: Find all other peers that were sent requests for this block
    // TODO 3: Send Cancel messages to those peers immediately
    // TODO 4: Remove block from active requests tracking
    // TODO 5: Update statistics for bandwidth saved by cancellation
    // Hint: Send cancellation even if peer hasn't responded yet
    panic("implement OnBlockReceived")
}
```

**G. Milestone Checkpoint:**

After implementing piece management and seeding:

1. **Basic Verification Test:**
   ```bash
   go test ./internal/piece/... -v
   # Should pass all unit tests for piece management
   ```

2. **Integration Test with Real Torrent:**
   ```bash
   go run cmd/bittorrent/main.go download test.torrent output_file
   # Should successfully download and verify a small torrent
   ```

3. **Expected Behavior Verification:**
   - Piece selection should prioritize rarest pieces first
   - Hash verification should reject corrupted pieces and re-request
   - Multiple peer connections should coordinate without conflicts
   - Seeding should respond to upload requests from other clients
   - Download should enter endgame mode for final pieces

4. **Performance Monitoring:**
   - Monitor piece verification time (should be <100ms per piece)
   - Check memory usage (should not grow unboundedly)
   - Verify concurrent download efficiency (multiple active peers)
   - Measure upload responsiveness (request handling latency)

**H. Debugging Tips:**

| Symptom | Likely Cause | Diagnosis Method | Fix |
|---------|--------------|------------------|-----|
| Download stalls at 99% | Endgame mode not triggered | Check remaining piece count | Implement proper endgame threshold |
| Hash verification failures | Network corruption or malicious peer | Log piece hashes and peer sources | Add peer reputation tracking |
| Memory usage grows constantly | Piece cleanup not working | Profile memory allocation | Implement timeout-based cleanup |
| Poor download speed | Inefficient piece selection | Analyze piece request patterns | Optimize rarest-first algorithm |
| Upload requests ignored | Choking logic errors | Check peer choke/unchoke states | Fix reciprocity algorithm |


## Interactions and Data Flow

> **Milestone(s):** This section integrates all four milestones (1-4) by showing how torrent parsing, tracker communication, peer protocols, and piece management work together during the complete download lifecycle.

Building a BitTorrent client involves orchestrating multiple concurrent subsystems that must work together seamlessly. Think of this like conducting a symphony orchestra where each musician (component) has their own part to play, but the magic happens when they synchronize their timing and harmonize their contributions. The torrent parser provides the sheet music, the tracker acts as the concert program listing all performers, the peer connections are the individual instruments playing their parts, and the piece manager is the conductor ensuring everything comes together into a coherent performance.

This orchestration becomes particularly complex because BitTorrent operates in a highly concurrent, distributed environment where failures are common and coordination must happen without central control. Unlike a traditional client-server application where one component controls the flow, BitTorrent requires careful choreography between autonomous components that each maintain their own state and make independent decisions while contributing to the collective goal of file transfer.

The key architectural challenge is managing shared state and coordination points without introducing race conditions or deadlocks. Each component operates in its own goroutines with its own lifecycle, but they must coordinate access to shared data structures like piece availability maps, peer connection pools, and download progress tracking. This requires careful design of synchronization primitives and message passing patterns.

### Complete Download Lifecycle

The BitTorrent download process follows a well-defined sequence that progresses through distinct phases, each building upon the previous phase's results. Think of this like planning and executing a complex heist where each step must be completed successfully before the next can begin, but once the operation is underway, multiple activities happen simultaneously while maintaining coordination.

**Phase 1: Initialization and Metadata Extraction**

The lifecycle begins when a user provides a torrent file path to the client. The `TorrentParser` immediately takes control and begins the metadata extraction process. This phase is entirely synchronous and must complete successfully before any networking can begin.

The parser creates a `Decoder` instance and begins parsing the bencode structure. As it processes the torrent file, it extracts critical information including the announce URL, file information, piece length, and piece hashes. Most importantly, it calculates the info hash by taking the SHA-1 digest of the exact bencoded bytes of the info dictionary. This info hash becomes the unique identifier that ties together all subsequent operations.

Once parsing completes successfully, the system has a complete `MetaInfo` structure containing all the information needed to begin the download. At this point, the client calculates derived information like the total file size, expected number of pieces, and creates the initial piece state map where every piece is marked as needed and unavailable.

> The info hash calculation during this phase is critical because it serves as the cryptographic fingerprint that ensures all peers are working on exactly the same file. Even a single byte difference in the original torrent would produce a completely different info hash, preventing any coordination.

**Phase 2: Peer Discovery Through Tracker Communication**

With the torrent metadata successfully parsed, the system transitions to peer discovery. The `TrackerClient` takes the lead in this phase, using the announce URL and info hash from the metadata to contact the tracker and obtain a list of peers participating in the swarm.

The tracker client constructs an `AnnounceRequest` containing the info hash, a randomly generated peer ID, the port the client will listen on, and initial statistics (zero bytes uploaded/downloaded, total bytes left equal to file size). This request gets sent as an HTTP GET to the tracker's announce URL with proper URL encoding of binary fields.

When the tracker responds with an `AnnounceResponse`, the system parses the compact peer list to extract IP addresses and ports of available peers. The tracker also provides an announce interval that determines when the client should next contact the tracker to update its status and refresh the peer list.

This phase typically completes quickly (within a few seconds) but establishes an ongoing relationship with the tracker. The client schedules periodic re-announces that will continue throughout the download to report progress and discover new peers joining the swarm.

**Phase 3: Concurrent Peer Connection Establishment**

Once the system has a list of potential peers, it enters the most complex phase where multiple activities happen simultaneously. The `PeerManager` begins establishing TCP connections to multiple peers in parallel, typically limiting itself to around 50 concurrent connections to avoid overwhelming the network or the local system.

For each successful TCP connection, the peer manager creates a `Connection` instance that immediately begins the BitTorrent handshake process. The handshake involves sending a 68-byte message containing the protocol identifier, info hash, and the client's peer ID, then waiting for the peer to respond with their own handshake message.

Successful handshakes are followed immediately by an exchange of bitfield messages where each peer announces which pieces they currently have available. This bitfield information gets stored in each connection's state and becomes crucial for piece selection decisions.

During this phase, connections may fail for various reasons: peers may be unreachable, handshakes may fail due to mismatched info hashes, or peers may disconnect. The system handles these failures gracefully by maintaining a pool of connection attempts and replacing failed connections with new attempts to different peers.

**Phase 4: Coordinated Multi-Peer Downloading**

With multiple peer connections established and bitfield information exchanged, the system enters the core downloading phase where the `PieceManager` coordinates piece requests across all available peers. This is where the orchestration becomes most complex because multiple components must work together while maintaining high throughput.

The piece manager implements a rarest-first selection strategy where it continuously monitors which pieces are available from which peers and prioritizes downloading pieces that are held by the fewest peers. This strategy helps ensure that rare pieces get downloaded early, preventing situations where the download stalls because only one peer has a needed piece and that peer becomes unavailable.

For each peer connection, the piece manager examines the peer's bitfield and selects appropriate pieces to request based on the rarest-first algorithm and the peer's choking state. When a peer is unchoked and interested, the connection begins sending block requests for 16KB chunks within selected pieces, maintaining a pipeline of several outstanding requests to maximize throughput.

As block data arrives from peers, the piece manager buffers the blocks until an entire piece is complete, then immediately verifies the piece by computing its SHA-1 hash and comparing it to the expected hash from the torrent metadata. Successfully verified pieces get written to disk at the correct file offset and the piece is marked as complete in the system's piece map.

This coordination requires careful synchronization because multiple peers may be downloading different pieces simultaneously, pieces must be assembled in the correct order in the output file, and the system must track which blocks have been requested from which peers to avoid duplicate requests and detect when re-requests are needed.

**Phase 5: Endgame and Completion**

As the download nears completion, the system enters endgame mode when only a small number of pieces remain unfinished. During endgame, the piece selection strategy changes from rarest-first to a more aggressive approach where the remaining blocks are requested from all peers that have them, ensuring that the download doesn't stall waiting for a single slow peer.

Once the final piece is downloaded and verified, the system performs a final integrity check of the complete file and transitions into seeding mode where it can serve pieces to other peers in the swarm. The tracker client sends a final announce with the "completed" event to inform the tracker that this peer now has the complete file.

### Inter-Component Communication

The BitTorrent client's components communicate through a carefully designed combination of direct method calls, channel-based message passing, and shared data structures protected by synchronization primitives. Think of this like a modern office building where different departments (components) need to coordinate their work through various communication channels: some use direct phone calls for urgent matters, others send memos through internal mail systems, and some share information through central bulletin boards that everyone can access.

**Direct Method Invocation for Synchronous Operations**

Certain operations require immediate, synchronous responses and use direct method calls between components. These typically involve configuration queries, state checks, and operations that must complete atomically.

The `PieceManager` directly calls methods on `Connection` objects when it needs to send requests or check peer state. When the piece manager determines that a peer should download a specific piece, it calls `SendRequest()` directly on the connection object with the piece index, block offset, and block length. This synchronous call allows the piece manager to immediately know whether the request was successfully queued for transmission.

Similarly, when connections receive incoming messages from peers, they make direct calls to piece manager methods to report received blocks or state changes. The connection calls `SubmitBlockData()` on the piece manager when a piece message arrives, providing the piece index, block offset, and data payload. This direct call allows the connection to immediately know if the block was accepted or if there was an error.

The torrent parser operates entirely through direct method calls since it provides a synchronous service to initialize the other components. The main client code calls `ParseFromFile()` and receives a complete `MetaInfo` structure that it then passes to other components during their initialization.

**Channel-Based Message Passing for Asynchronous Coordination**

For operations that can be handled asynchronously or that involve coordination between multiple goroutines, the system uses Go channels to pass messages and work items between components. This approach prevents blocking and allows components to process work at their own pace while maintaining proper ordering.

The piece manager uses several channels to coordinate its internal operations:

| Channel | Type | Purpose | Producer | Consumer |
|---------|------|---------|----------|----------|
| `workRequests` | `chan WorkRequest` | Assigns download work to peers | PieceManager main goroutine | Peer request handlers |
| `blockData` | `chan BlockData` | Reports received block data | Connection goroutines | PieceManager verification |
| `verificationResults` | `chan VerificationResult` | Reports piece verification outcomes | Verification pool workers | PieceManager state updates |
| `peerEvents` | `chan PeerEvent` | Reports peer connection changes | PeerManager | PieceManager interest updates |

The tracker client uses channels to handle periodic announces without blocking other operations. It runs a background goroutine that waits on a timer channel and sends announce requests at the appropriate intervals, then sends results back through a response channel that other components can monitor.

**Shared Data Structures with Synchronization**

Some information must be accessible to multiple components simultaneously, requiring shared data structures protected by mutexes or other synchronization primitives. These structures serve as the "single source of truth" for critical state that affects coordination decisions.

The piece state map is the most important shared data structure, containing the current status of every piece in the torrent:

| Field | Type | Access Pattern | Protection |
|-------|------|----------------|------------|
| `pieces` | `map[int]*PieceState` | Read: all components, Write: PieceManager only | `sync.RWMutex` |
| `statistics` | `*Statistics` | Read: tracker client, Write: piece manager | `sync.RWMutex` |
| `connections` | `map[string]*Connection` | Read: piece manager, Write: peer manager | `sync.RWMutex` |
| `bitfields` | `map[string]*BitfieldOps` | Read: piece manager, Write: connections | Per-connection mutex |

The piece manager holds the write lock on the piece map when updating piece states after verification, but uses read locks when selecting pieces for download. This allows multiple peer connections to concurrently check piece availability without blocking each other, while ensuring consistency when pieces are completed.

Connection state is managed through per-connection mutexes that protect the choking/interested flags and pending request maps. When a connection receives a choke message from a peer, it acquires its own mutex to update the `peerChoking` flag and clear any pending requests, then notifies the piece manager through a channel message.

**Event-Driven State Updates**

The system uses an event-driven architecture where state changes in one component trigger notifications to other components that need to react to those changes. This decouples components and allows them to operate independently while maintaining system-wide consistency.

When a peer connection completes its handshake and receives a bitfield, it generates a `PeerAvailable` event that gets sent to the piece manager. The piece manager responds by updating its peer availability map and potentially selecting new pieces to download from the newly available peer.

Similarly, when the piece manager completes verification of a downloaded piece, it generates events that notify the tracker client (to update upload/download statistics), the peer connections (to update interest states), and the file writer (to persist the piece data to disk).

The tracker client operates on a timer-driven event model where it generates `AnnounceNeeded` events based on the interval provided by the tracker, then sends `PeerListUpdate` events when announce responses are received with new peer information.

> **Decision: Channel-Based vs Callback-Based Event System**
> - **Context**: Components need to notify each other of state changes without tight coupling
> - **Options Considered**: Direct callbacks, observer pattern, channel-based messaging
> - **Decision**: Channel-based messaging with typed event structures
> - **Rationale**: Channels provide natural backpressure, are type-safe, integrate well with Go's concurrency model, and allow easy testing by mocking channel interactions
> - **Consequences**: Slightly more complex setup but much better testability, debuggability, and resilience to component failures

### Concurrency Coordination

Managing concurrency in a BitTorrent client presents unique challenges because the system must coordinate dozens of independent peer connections while maintaining consistency of shared state and avoiding race conditions. Think of this like managing a busy restaurant kitchen where multiple chefs (peer connections) are working on different dishes (pieces) simultaneously, sharing common ingredients (blocks) and equipment (network resources), while the head chef (piece manager) coordinates the timing so that complete meals (verified pieces) are delivered to customers (written to disk) in the correct order.

The complexity comes from the fact that BitTorrent operations are inherently asynchronous and distributed, with network delays, peer behavior, and piece availability constantly changing. The system must handle these dynamic conditions while ensuring that shared data structures remain consistent and that the download progresses efficiently.

**Goroutine Architecture and Lifecycle Management**

The BitTorrent client uses a structured approach to goroutine management where each major component runs in its own goroutine or goroutine pool, with clearly defined lifecycles and shutdown procedures.

| Component | Goroutines | Lifecycle | Shutdown Signal |
|-----------|------------|-----------|-----------------|
| `TrackerClient` | 1 announce goroutine | Starts with first announce, runs until completion | Context cancellation |
| `PeerManager` | 1 connection manager + N connection handlers | Starts during peer discovery, connection handlers spawn/die with connections | Connection-specific contexts |
| `PieceManager` | 1 coordinator + M verification workers | Starts immediately after metadata parsing, runs until completion | Completion detection |
| `Connection` | 2 per connection (read/write) | Spawned when TCP connection established, dies on disconnect | Connection close |

Each goroutine operates independently but coordinates through the message passing and shared state mechanisms described earlier. The key insight is that goroutines should own their local state and communicate changes rather than directly modifying shared state, reducing the need for fine-grained locking.

The main client goroutine acts as the lifecycle coordinator, starting and stopping component goroutines in the correct order and handling shutdown signals. When the user cancels the download or the download completes, the main goroutine cancels the root context, which cascades cancellation signals to all component contexts.

**Shared State Synchronization Patterns**

The most critical synchronization challenge involves the piece state map, which must be accessed by multiple peer connections simultaneously for piece selection and progress tracking, while being updated by the piece manager when pieces are completed.

The system uses a read-write mutex pattern where the piece manager holds exclusive write access during state updates, but allows concurrent read access for piece selection:

```
Reader Pattern (Piece Selection):
1. Acquire read lock on piece map
2. Examine piece states and peer bitfields
3. Identify candidate pieces for download
4. Release read lock
5. Make download requests (no locks held)

Writer Pattern (Piece Completion):
1. Verify piece hash outside of any locks
2. Acquire write lock on piece map
3. Update piece state to completed
4. Update statistics and availability maps
5. Release write lock
6. Notify other components via channels
```

This pattern minimizes lock contention because piece selection (read operations) happens much more frequently than piece completion (write operations), and the read operations can proceed concurrently without interfering with each other.

Connection state synchronization follows a different pattern because each connection's state is independent. Each `Connection` object has its own mutex that protects its local state (choking flags, pending requests, bitfield), eliminating contention between different peer connections.

**Request Pipeline Coordination**

One of the most complex coordination challenges involves managing the request pipeline where multiple blocks within a piece may be requested from different peers, and the system must track which blocks have been requested, received, and verified while avoiding duplicate requests and handling peer failures.

The piece manager maintains a sophisticated state machine for each piece that tracks individual block states:

| Block State | Meaning | Next States | Coordination Requirements |
|-------------|---------|-------------|--------------------------|
| `Needed` | Block not yet requested | `Requested` | None - initial state |
| `Requested` | Block requested from specific peer | `Received`, `Needed` | Must track timeout and peer identity |
| `Received` | Block data received and buffered | `Verified` | Must wait for all blocks in piece |
| `Verified` | Part of completed, hash-verified piece | Final state | Must coordinate file write |

The coordination challenge comes from the fact that block requests may time out, peers may disconnect, or multiple peers may send the same block. The system handles these scenarios through careful state management:

When a peer connection wants to request blocks, it calls `RequestWork()` on the piece manager with its peer ID and available piece bitfield. The piece manager examines its internal state, selects appropriate blocks that are in the `Needed` state, atomically transitions them to `Requested` with the requesting peer's ID, and returns the list of blocks to request.

If a peer disconnects or a request times out, the piece manager scans for blocks in the `Requested` state associated with that peer and transitions them back to `Needed` so they can be requested from other peers. This recovery process must be atomic to prevent blocks from being lost or double-requested.

**Endgame Mode Coordination**

The endgame mode presents the most complex coordination challenge because the normal piece selection rules change and the system begins requesting the same blocks from multiple peers simultaneously. This requires careful coordination to avoid wasting bandwidth while ensuring the download completes quickly.

The `EndgameManager` monitors download progress and activates when fewer than `MAX_ENDGAME_PIECES` remain incomplete. Once activated, it changes the request strategy to send requests for remaining blocks to all peers that have the containing pieces, then cancels duplicate requests as soon as any peer provides the block.

This coordination requires tracking which peers have been sent requests for which blocks, and when a block is received, immediately sending cancel messages to all other peers that were sent requests for the same block. The timing is critical because cancel messages may not arrive before peers send the duplicate data, so the system must be prepared to receive and discard duplicate blocks.

> **Decision: Optimistic vs Pessimistic Concurrency Control**
> - **Context**: Multiple peers may attempt to download the same pieces simultaneously
> - **Options Considered**: Pessimistic locking (reserve pieces), optimistic conflicts (detect and resolve), hybrid approach
> - **Decision**: Optimistic with conflict detection for normal mode, pessimistic for endgame
> - **Rationale**: Optimistic allows better parallelism when conflicts are rare, but endgame requires careful coordination to avoid waste
> - **Consequences**: More complex state management but better performance and simpler deadlock avoidance

**Graceful Shutdown Coordination**

Properly shutting down a BitTorrent client requires careful coordination to ensure that in-flight operations complete cleanly and that the final state is reported to the tracker. The shutdown process follows a specific sequence that allows components to complete their work before terminating.

When shutdown is initiated (either by user request or download completion), the main goroutine begins an orderly shutdown process. First, it signals the peer manager to stop accepting new connections and begin closing existing connections gracefully. Existing connections are allowed to complete any in-flight piece requests before closing.

The piece manager continues processing incoming block data and verification until all peer connections have closed, ensuring that any blocks received during shutdown are properly handled. Only after all peer connections have terminated does the piece manager shut down its verification workers.

Finally, the tracker client sends a final announce to report the current statistics and indicate that the peer is leaving the swarm. This announce uses the "stopped" event type and helps other peers in the swarm understand that this peer is no longer available.

The entire shutdown process is coordinated through context cancellation that flows from the main goroutine to all component goroutines, with each component responsible for completing its cleanup tasks before the context deadline expires.

![Complete Download Flow](./diagrams/download-sequence.svg)

![BitTorrent Client Architecture](./diagrams/system-architecture.svg)

![Multi-Peer Download Coordination](./diagrams/concurrent-download.svg)

### Implementation Guidance

This section provides the infrastructure and coordination code needed to orchestrate the complete BitTorrent download lifecycle. The target is a clean, production-ready implementation that handles the complex interactions between components.

**Technology Recommendations**

| Coordination Aspect | Simple Approach | Advanced Approach |
|---------------------|-----------------|-------------------|
| Inter-component messaging | Go channels with struct types | Event bus with typed handlers |
| State synchronization | `sync.RWMutex` on shared maps | Lock-free data structures with atomic operations |
| Goroutine coordination | Context-based cancellation | Supervised goroutine trees with restart policies |
| Error propagation | Error return values + logging | Structured error events with recovery strategies |
| Progress tracking | Atomic counters + periodic reports | Real-time metrics with time-series storage |

For a learning implementation, stick with the simple approaches that use Go's built-in concurrency primitives effectively.

**Recommended Module Structure**

```
internal/
  client/
    client.go              ← Main orchestration logic
    lifecycle.go           ← Download lifecycle management
    events.go             ← Inter-component event types
  coordination/
    piece_coordinator.go   ← Piece request coordination
    peer_coordinator.go    ← Peer connection coordination
    state_manager.go       ← Shared state synchronization
  statistics/
    tracker.go            ← Download progress tracking
    metrics.go            ← Performance metrics
```

**Event System Infrastructure (Complete)**

```go
package client

import (
    "context"
    "sync"
    "time"
)

// EventType represents the type of events that flow between components
type EventType int

const (
    EventPeerConnected EventType = iota
    EventPeerDisconnected
    EventPieceCompleted
    EventBlockReceived
    EventTrackerResponse
    EventDownloadComplete
)

// Event represents a coordination event between components
type Event struct {
    Type      EventType
    PeerID    string
    Timestamp time.Time
    Data      interface{}
}

// EventBus coordinates message passing between components
type EventBus struct {
    subscribers map[EventType][]chan Event
    mutex       sync.RWMutex
    ctx         context.Context
    cancel      context.CancelFunc
}

// NewEventBus creates a new event coordination system
func NewEventBus(ctx context.Context) *EventBus {
    busCtx, cancel := context.WithCancel(ctx)
    return &EventBus{
        subscribers: make(map[EventType][]chan Event),
        ctx:         busCtx,
        cancel:      cancel,
    }
}

// Subscribe registers a channel to receive events of specified type
func (eb *EventBus) Subscribe(eventType EventType, bufferSize int) <-chan Event {
    eb.mutex.Lock()
    defer eb.mutex.Unlock()
    
    ch := make(chan Event, bufferSize)
    eb.subscribers[eventType] = append(eb.subscribers[eventType], ch)
    return ch
}

// Publish sends an event to all subscribers of the event type
func (eb *EventBus) Publish(event Event) {
    eb.mutex.RLock()
    subscribers := eb.subscribers[event.Type]
    eb.mutex.RUnlock()
    
    event.Timestamp = time.Now()
    
    for _, ch := range subscribers {
        select {
        case ch <- event:
            // Event delivered successfully
        case <-eb.ctx.Done():
            return
        default:
            // Subscriber's buffer is full, skip this subscriber
            // In production, might want to log this condition
        }
    }
}

// Shutdown closes all subscriber channels and stops the event bus
func (eb *EventBus) Shutdown() {
    eb.cancel()
    
    eb.mutex.Lock()
    defer eb.mutex.Unlock()
    
    for _, channels := range eb.subscribers {
        for _, ch := range channels {
            close(ch)
        }
    }
    eb.subscribers = make(map[EventType][]chan Event)
}
```

**State Synchronization Infrastructure (Complete)**

```go
package coordination

import (
    "sync"
    "time"
)

// SharedState manages synchronized access to global BitTorrent state
type SharedState struct {
    pieces      map[int]*PieceState
    connections map[string]*Connection
    statistics  *Statistics
    
    pieceMutex      sync.RWMutex
    connectionMutex sync.RWMutex
    statsMutex      sync.RWMutex
}

// Statistics tracks download progress and performance metrics
type Statistics struct {
    Downloaded    int64     // Bytes successfully downloaded and verified
    Uploaded      int64     // Bytes uploaded to other peers
    Left          int64     // Bytes remaining to download
    StartTime     time.Time // When download began
    CompletedPieces int     // Number of pieces completed
    TotalPieces     int     // Total pieces in torrent
    ConnectedPeers  int     // Currently connected peer count
}

// NewSharedState creates synchronized state manager
func NewSharedState(totalPieces int, totalLength int64) *SharedState {
    return &SharedState{
        pieces:      make(map[int]*PieceState),
        connections: make(map[string]*Connection),
        statistics: &Statistics{
            Left:        totalLength,
            StartTime:   time.Now(),
            TotalPieces: totalPieces,
        },
    }
}

// GetPieceState safely retrieves piece state for reading
func (ss *SharedState) GetPieceState(pieceIndex int) *PieceState {
    ss.pieceMutex.RLock()
    defer ss.pieceMutex.RUnlock()
    return ss.pieces[pieceIndex]
}

// UpdatePieceState safely updates piece state
func (ss *SharedState) UpdatePieceState(pieceIndex int, state *PieceState) {
    ss.pieceMutex.Lock()
    defer ss.pieceMutex.Unlock()
    ss.pieces[pieceIndex] = state
}

// GetAvailablePieces returns slice of piece indices that need downloading
func (ss *SharedState) GetAvailablePieces() []int {
    ss.pieceMutex.RLock()
    defer ss.pieceMutex.RUnlock()
    
    var needed []int
    for index, piece := range ss.pieces {
        if piece.State == PieceStateNeeded {
            needed = append(needed, index)
        }
    }
    return needed
}

// AddConnection registers a new peer connection
func (ss *SharedState) AddConnection(peerID string, conn *Connection) {
    ss.connectionMutex.Lock()
    defer ss.connectionMutex.Unlock()
    ss.connections[peerID] = conn
    
    ss.statsMutex.Lock()
    ss.statistics.ConnectedPeers++
    ss.statsMutex.Unlock()
}

// RemoveConnection unregisters a peer connection
func (ss *SharedState) RemoveConnection(peerID string) {
    ss.connectionMutex.Lock()
    defer ss.connectionMutex.Unlock()
    delete(ss.connections, peerID)
    
    ss.statsMutex.Lock()
    ss.statistics.ConnectedPeers--
    ss.statsMutex.Unlock()
}

// GetConnections returns all active connections (copy for safety)
func (ss *SharedState) GetConnections() map[string]*Connection {
    ss.connectionMutex.RLock()
    defer ss.connectionMutex.RUnlock()
    
    connections := make(map[string]*Connection)
    for id, conn := range ss.connections {
        connections[id] = conn
    }
    return connections
}

// UpdateStatistics safely updates download progress statistics
func (ss *SharedState) UpdateStatistics(downloaded, uploaded int64, completedPieces int) {
    ss.statsMutex.Lock()
    defer ss.statsMutex.Unlock()
    
    ss.statistics.Downloaded = downloaded
    ss.statistics.Uploaded = uploaded
    ss.statistics.Left = ss.statistics.Left - (downloaded - ss.statistics.Downloaded)
    ss.statistics.CompletedPieces = completedPieces
}

// GetStatistics returns current statistics (copy for safety)
func (ss *SharedState) GetStatistics() Statistics {
    ss.statsMutex.RLock()
    defer ss.statsMutex.RUnlock()
    return *ss.statistics // Return copy
}
```

**Main Client Orchestration (Skeleton)**

```go
package client

import (
    "context"
    "fmt"
    "time"
    
    "internal/coordination"
    "internal/peer"
    "internal/piece"
    "internal/torrent"
    "internal/tracker"
)

// Client orchestrates the complete BitTorrent download process
type Client struct {
    metaInfo     *torrent.MetaInfo
    sharedState  *coordination.SharedState
    eventBus     *EventBus
    
    trackerClient *tracker.Client
    peerManager   *peer.Manager
    pieceManager  *piece.PieceManager
    
    ctx    context.Context
    cancel context.CancelFunc
}

// NewClient creates a BitTorrent client for the specified torrent
func NewClient(torrentPath string) (*Client, error) {
    // TODO 1: Parse torrent file using torrent.ParseFromFile()
    // TODO 2: Calculate total file length and expected piece count
    // TODO 3: Create shared state manager with piece count and file size
    // TODO 4: Create event bus for component coordination
    // TODO 5: Initialize tracker client with metainfo
    // TODO 6: Initialize peer manager with event bus
    // TODO 7: Initialize piece manager with metainfo and shared state
    // TODO 8: Set up context for lifecycle management
}

// Download orchestrates the complete download lifecycle
func (c *Client) Download(outputPath string) error {
    // TODO 1: Start tracker communication to get initial peer list
    // TODO 2: Subscribe to peer connection events from peer manager
    // TODO 3: Subscribe to piece completion events from piece manager
    // TODO 4: Start peer manager to begin connection attempts
    // TODO 5: Start piece manager to coordinate downloads
    // TODO 6: Monitor download progress and handle events
    // TODO 7: Detect completion condition (all pieces verified)
    // TODO 8: Send completion announce to tracker
    // TODO 9: Begin seeding mode (serve pieces to other peers)
    // Hint: Use select statement to handle multiple event channels
    // Hint: Check for context cancellation to support early termination
}

// handlePeerEvent processes peer connection state changes
func (c *Client) handlePeerEvent(event Event) {
    // TODO 1: Extract peer ID and event type from event data
    // TODO 2: Update shared state based on connection/disconnection
    // TODO 3: For new connections, trigger interest state updates
    // TODO 4: For disconnections, clean up pending requests
    // TODO 5: Update connection count statistics
}

// handlePieceEvent processes piece completion and verification events
func (c *Client) handlePieceEvent(event Event) {
    // TODO 1: Extract piece index and verification result from event
    // TODO 2: Update piece state in shared state manager
    // TODO 3: Update download statistics (bytes downloaded, pieces completed)
    // TODO 4: Check if download is complete (all pieces verified)
    // TODO 5: Notify tracker client if significant progress made
    // TODO 6: Update interest states for all peer connections
}

// Shutdown gracefully terminates all client operations
func (c *Client) Shutdown() error {
    // TODO 1: Cancel context to signal shutdown to all components
    // TODO 2: Stop accepting new peer connections
    // TODO 3: Allow in-flight piece requests to complete
    // TODO 4: Send final statistics to tracker with "stopped" event
    // TODO 5: Close all peer connections gracefully
    // TODO 6: Shutdown event bus and clean up resources
    // TODO 7: Wait for all goroutines to terminate with timeout
}

// isDownloadComplete checks if all pieces have been verified
func (c *Client) isDownloadComplete() bool {
    // TODO 1: Get current statistics from shared state
    // TODO 2: Compare completed pieces to total expected pieces
    // TODO 3: Return true only if all pieces completed and verified
}
```

**Milestone Checkpoints**

After implementing the coordination infrastructure, verify these behaviors:

1. **Event Flow Verification**: Create a simple test that publishes events through the event bus and verifies that subscribers receive them in the correct order without blocking.

2. **Concurrent State Access**: Run multiple goroutines that simultaneously read and write piece states through the shared state manager, ensuring no race conditions occur.

3. **Lifecycle Coordination**: Start the full client with a small test torrent and verify that components start up in the correct order and shut down cleanly when cancelled.

4. **Progress Reporting**: Monitor the statistics during a real download to ensure that progress is accurately tracked and reported to the tracker at appropriate intervals.

Expected behavior: The client should successfully coordinate all components, handle peer connections and disconnections gracefully, maintain accurate download progress, and complete downloads without deadlocks or race conditions.

**Common Integration Issues**

⚠️ **Pitfall: Deadlocks in State Access**
When multiple components need to access shared state simultaneously, incorrect lock ordering can cause deadlocks. Always acquire locks in a consistent order: pieces, then connections, then statistics.

⚠️ **Pitfall: Event Channel Blocking**
If event subscribers don't process events quickly enough, the event bus can block publishers. Use buffered channels and handle the case where subscribers can't keep up by dropping events if necessary.

⚠️ **Pitfall: Goroutine Leaks During Shutdown**
Components may not terminate cleanly if they're blocked on channel operations or waiting for network I/O. Always use context cancellation with timeouts to force termination if graceful shutdown fails.

⚠️ **Pitfall: Race Conditions in Statistics**
Download statistics are accessed by multiple components simultaneously. Protect all statistics updates with mutexes and be careful about compound operations that read-modify-write multiple fields.


## Error Handling and Edge Cases

> **Milestone(s):** This section applies to all milestones (1-4) by establishing comprehensive error handling strategies that ensure robustness across torrent parsing, tracker communication, peer protocols, and piece management.

Building a robust BitTorrent client requires anticipating and gracefully handling numerous failure scenarios that arise in distributed peer-to-peer systems. Unlike centralized systems where failures are localized and predictable, BitTorrent clients must operate in an adversarial environment where peers may misbehave, networks partition unpredictably, and data corruption occurs frequently. This section establishes comprehensive error handling strategies that transform a fragile prototype into a production-ready client capable of operating reliably in the chaotic world of peer-to-peer file sharing.

Think of error handling in BitTorrent like designing a ship to navigate treacherous waters. Just as a well-designed vessel must handle storms, equipment failures, and unpredictable conditions while maintaining course toward its destination, a BitTorrent client must gracefully handle network failures, malicious peers, and data corruption while continuing to make progress toward completing the download. The key insight is that failures are not exceptional cases to be avoided, but normal operating conditions that must be anticipated and managed systematically.

### System Failure Modes

Understanding the complete catalog of potential failures is essential for building comprehensive error handling. BitTorrent clients operate in a complex distributed environment where failures can occur at every layer of the system stack, from low-level network connectivity to high-level protocol violations. Each failure mode requires specific detection mechanisms and recovery strategies.

**Network-Level Failures**

Network failures represent the most common category of issues encountered in peer-to-peer systems. These failures can occur at any point in the network stack and often manifest as timeouts, connection resets, or partial data transmission.

| Failure Mode | Detection Mechanism | Symptoms | Recovery Strategy |
|--------------|-------------------|----------|------------------|
| Connection Timeout | Socket connect() timeout | TCP connection never establishes | Exponential backoff retry with peer blacklisting |
| Connection Reset | ECONNRESET error | Existing connection drops unexpectedly | Immediate reconnection attempt with request replay |
| Partial Read | Incomplete message framing | Message length prefix without payload | Buffer incomplete data, retry read with timeout |
| DNS Resolution Failure | Resolver error for tracker hostname | Cannot resolve tracker announce URL | Try backup trackers, fallback to IP addresses |
| Network Partition | Multiple connection failures | All peer connections timeout simultaneously | Pause operations, retry after exponential backoff |
| Bandwidth Exhaustion | Slow transfer rates, timeouts | Block downloads taking excessive time | Reduce concurrent connections, implement rate limiting |

**Tracker-Level Failures**

Tracker failures directly impact peer discovery, which is critical for maintaining a healthy swarm connection. These failures require sophisticated retry logic and fallback mechanisms to ensure the client can continue discovering peers even when primary trackers are unavailable.

| Failure Mode | Detection Mechanism | Symptoms | Recovery Strategy |
|--------------|-------------------|----------|------------------|
| HTTP 4xx Client Error | HTTP response status code | 400-499 status in announce response | Validate request parameters, fix encoding issues |
| HTTP 5xx Server Error | HTTP response status code | 500-599 status in announce response | Exponential backoff retry, try backup trackers |
| Malformed Response | Bencode parsing error | Invalid bencode in tracker response | Skip response, mark tracker as unreliable |
| Missing Required Fields | Field validation error | Response missing 'peers' or 'interval' | Use default values where possible, retry announce |
| Tracker Timeout | HTTP request timeout | No response within configured timeout | Mark tracker as slow, increase timeout for retries |
| Invalid Peer List | Compact peer parsing error | Malformed compact peer data | Skip invalid entries, use successfully parsed peers |

**Peer Protocol Failures**

Peer protocol failures occur during direct communication between BitTorrent clients and can range from simple message format errors to complex state machine violations. These failures require careful state management to prevent protocol deadlocks and ensure fair resource allocation.

| Failure Mode | Detection Mechanism | Symptoms | Recovery Strategy |
|--------------|-------------------|----------|------------------|
| Handshake Mismatch | Info hash comparison | Peer reports different info hash | Immediately disconnect, blacklist peer |
| Protocol Violation | State machine validation | Invalid message for current state | Send protocol error, attempt graceful disconnect |
| Message Corruption | Length/checksum validation | Message length doesn't match payload | Request message retransmission, reset connection |
| Peer Timeout | Keepalive timer expiration | No messages received within keepalive period | Send keepalive, disconnect if no response |
| Request Queue Overflow | Pipeline depth tracking | Too many outstanding requests | Stop sending new requests, wait for completions |
| Choke/Unchoke Cycling | State change frequency monitoring | Rapid choke/unchoke state changes | Implement choke state dampening, reduce request rate |

**Data Integrity Failures**

Data integrity failures are particularly critical in BitTorrent because they can corrupt the downloaded file and waste significant bandwidth. These failures require immediate detection and recovery to prevent propagation of corrupt data.

| Failure Mode | Detection Mechanism | Symptoms | Recovery Strategy |
|--------------|-------------------|----------|------------------|
| Piece Hash Mismatch | SHA1 verification | Downloaded piece hash doesn't match torrent | Discard piece, redownload from different peers |
| Block Data Corruption | Piece assembly validation | Block doesn't fit expected piece structure | Request block retransmission, check peer reliability |
| File System Errors | Write operation failures | Cannot write piece data to disk | Check disk space, permissions, retry with backoff |
| Incomplete Piece Assembly | Block count validation | Missing blocks when piece should be complete | Request missing blocks, verify block tracking |
| Size Mismatch | File length validation | Downloaded file size doesn't match torrent metadata | Verify piece boundaries, check for padding errors |
| Write Permission Errors | File system error codes | Cannot create or modify output files | Check permissions, suggest alternative output paths |

> **Critical Insight**: The key to robust error handling in BitTorrent is understanding that failures cascade through multiple layers. A network timeout doesn't just mean "retry the connection" – it means reassessing piece assignments, updating peer reliability scores, adjusting request pipeline depths, and potentially triggering tracker re-announce to find alternative peers.

**State Consistency Failures**

State consistency failures occur when different components of the BitTorrent client maintain conflicting views of system state. These failures are particularly dangerous because they can lead to subtle bugs that only manifest under specific timing conditions.

| Failure Mode | Detection Mechanism | Symptoms | Recovery Strategy |
|--------------|-------------------|----------|------------------|
| Piece State Divergence | Cross-component validation | Different components report conflicting piece states | Rebuild state from authoritative source (disk) |
| Peer State Desynchronization | Connection state validation | Local peer state doesn't match peer's reported state | Perform handshake renegotiation |
| Request Tracking Mismatch | Pipeline validation | Outstanding requests don't match sent requests | Clear request queue, resend active requests |
| Bitfield Inconsistency | Availability counting errors | Peer availability doesn't match bitfield data | Request fresh bitfield update |
| Statistics Corruption | Counter validation | Downloaded/uploaded counters are inconsistent | Recalculate statistics from piece state |
| Concurrent Modification | Lock validation failures | Multiple threads modify shared state simultaneously | Implement stricter locking, detect race conditions |

### Recovery and Retry Logic

Effective recovery from failures requires sophisticated retry logic that adapts to different types of failures and learns from past experiences. The key insight is that not all failures are equal – some indicate temporary network issues that resolve quickly, while others suggest fundamental problems requiring different recovery strategies.

**Exponential Backoff Strategy**

Exponential backoff prevents overwhelming failed services with repeated requests while providing reasonable recovery times for transient failures. The strategy must be tailored to different failure types because network timeouts require different backoff parameters than tracker server errors.

The `BackoffScheduler` manages retry timing for different types of operations:

| Operation Type | Base Delay | Max Delay | Multiplier | Jitter | Reset Condition |
|---------------|------------|-----------|------------|---------|-----------------|
| Tracker Announce | 30 seconds | 15 minutes | 2.0 | ±25% | Successful announce |
| Peer Connection | 5 seconds | 5 minutes | 1.5 | ±20% | Successful handshake |
| Block Request | 1 second | 30 seconds | 2.0 | ±10% | Successful block receipt |
| File I/O Operation | 100ms | 10 seconds | 3.0 | ±15% | Successful write |
| DNS Resolution | 2 seconds | 2 minutes | 2.0 | ±30% | Successful resolution |

The backoff algorithm follows this progression:
1. Start with the base delay for the operation type
2. After each failure, multiply the current delay by the multiplier
3. Add random jitter to prevent thundering herd effects
4. Cap the delay at the maximum value for the operation type
5. Reset to base delay when the reset condition is met
6. Track consecutive failures to implement circuit breaker patterns

**Circuit Breaker Pattern**

Circuit breakers prevent the client from repeatedly attempting operations that are likely to fail, providing fast failure responses and reducing resource waste. Different components require different circuit breaker configurations based on their failure characteristics and recovery patterns.

| Component | Failure Threshold | Recovery Timeout | Half-Open Requests | Success Threshold |
|-----------|------------------|------------------|-------------------|------------------|
| Tracker Client | 5 consecutive failures | 5 minutes | 1 announce | 2 successes |
| Peer Connection | 3 consecutive handshake failures | 2 minutes | 1 connection | 1 success |
| File Writer | 10 consecutive write failures | 30 seconds | 1 write | 3 successes |
| DNS Resolver | 3 consecutive resolution failures | 1 minute | 1 resolution | 1 success |

The circuit breaker state machine operates as follows:
1. **Closed State**: Normal operation, all requests pass through
2. **Open State**: All requests fail immediately without attempting the operation
3. **Half-Open State**: Allow limited requests to test if the service has recovered
4. **Recovery**: If half-open requests succeed, return to closed state

**Peer Quality Scoring**

Peer quality scoring helps the client prioritize reliable peers and avoid problematic ones. The scoring system tracks multiple metrics to build a comprehensive view of peer reliability and performance.

| Metric | Weight | Positive Factors | Negative Factors | Decay Rate |
|--------|---------|-----------------|------------------|------------|
| Connection Reliability | 0.3 | Successful handshakes | Connection timeouts, resets | 0.95/hour |
| Data Quality | 0.4 | Correct piece hashes | Hash mismatches, corruption | 0.9/hour |
| Transfer Performance | 0.2 | Fast block transfers | Slow responses, timeouts | 0.98/hour |
| Protocol Compliance | 0.1 | Proper message format | Protocol violations | 0.85/hour |

The scoring algorithm maintains a running average for each peer:
1. Initialize all peers with a neutral score (0.5)
2. Update scores based on observed behavior using weighted factors
3. Apply time-based decay to prevent permanent peer blacklisting
4. Use scores to prioritize connection attempts and request assignments
5. Implement minimum score thresholds for different operations

> **Decision: Peer Scoring vs. Simple Blacklisting**
> - **Context**: Need to handle unreliable peers while maintaining adequate peer diversity
> - **Options Considered**: Binary blacklisting, simple success/failure counters, comprehensive scoring system
> - **Decision**: Comprehensive scoring system with time-based decay
> - **Rationale**: Peers may become reliable over time, binary decisions lose valuable peer diversity, scoring provides fine-grained prioritization
> - **Consequences**: More complex implementation but better peer utilization and fault tolerance

**Request Recovery and Replay**

When peer connections fail during active downloads, the client must recover in-flight requests and reassign them to other peers. This process requires careful coordination between the piece manager and peer connections to prevent data loss and duplicate work.

The request recovery process follows these steps:
1. **Failure Detection**: Peer connection detects network failure or timeout
2. **Request Inventory**: Catalog all outstanding requests for the failed peer
3. **State Validation**: Verify which requests were actually in progress
4. **Peer Notification**: Inform piece manager of the connection failure
5. **Request Reassignment**: Piece manager reassigns requests to other available peers
6. **Progress Preservation**: Maintain download progress statistics and piece state
7. **Quality Update**: Update peer quality scores based on the failure

The reassignment algorithm prioritizes requests based on several factors:

| Priority Factor | Weight | Rationale |
|----------------|--------|-----------|
| Piece Rarity | 0.4 | Rare pieces are harder to obtain from alternative peers |
| Request Age | 0.3 | Older requests represent more invested effort |
| Peer Availability | 0.2 | More available peers increase reassignment success probability |
| Block Position | 0.1 | Later blocks in a piece represent more completion progress |

### Protocol Edge Cases

BitTorrent protocol edge cases arise from the interaction between the formal protocol specification and real-world implementation variations. These edge cases often involve timing issues, message ordering problems, and interpretation ambiguities that can cause interoperability failures between different BitTorrent clients.

**Message Ordering and Timing Edge Cases**

The peer wire protocol assumes certain message ordering constraints, but network delays and concurrent processing can violate these assumptions. The client must handle out-of-order messages and timing-dependent protocol states gracefully.

| Edge Case | Scenario | Detection | Recovery Strategy |
|-----------|----------|-----------|------------------|
| Bitfield After Pieces | Peer sends bitfield after piece messages | Message sequence validation | Merge bitfield with existing piece knowledge |
| Request Before Unchoke | Request received while peer should be choked | Choke state validation | Queue request for when peer becomes unchoked |
| Piece for Unrequested Block | Piece message for block we didn't request | Request tracking validation | Accept piece if needed, update request tracking |
| Double Handshake | Multiple handshake messages on same connection | Handshake completion flag | Ignore duplicate handshake, continue normal protocol |
| Stale Keep-Alive | Keep-alive messages after connection should be closed | Connection state tracking | Ignore keep-alive, proceed with connection cleanup |
| Race Condition Choke | Choke message arrives after request sent | Message timestamp comparison | Honor choke, cancel in-flight requests |

**Message Format Edge Cases**

Real-world BitTorrent implementations sometimes generate messages that deviate from the strict protocol specification. The client must be liberal in what it accepts while maintaining security and correctness.

| Edge Case | Violation | Tolerance Strategy | Security Consideration |
|-----------|-----------|-------------------|------------------------|
| Oversized Bitfield | Bitfield longer than expected piece count | Truncate to expected length | Validate against maximum torrent size |
| Invalid Message Length | Length prefix doesn't match payload size | Attempt payload parsing with actual length | Limit maximum message size to prevent DoS |
| Unknown Message Type | Message ID not in protocol specification | Log and ignore unknown messages | Track unknown message frequency per peer |
| Missing Keep-Alive | No keep-alive within expected interval | Extend timeout, send our own keep-alive | Implement maximum silence period |
| Zero-Length Request | Request with zero-byte length | Treat as invalid, send rejection | Prevent infinite loops in request handling |
| Duplicate Have Messages | Multiple have messages for same piece | Ignore duplicates, update availability once | Track message frequency to detect spam |

**State Machine Violations**

Peer state machines can enter invalid states due to message loss, network partitions, or implementation bugs. The client must detect these violations and recover to a consistent state.

| Violation Type | Invalid Transition | Detection Method | Recovery Action |
|---------------|-------------------|------------------|-----------------|
| Request While Choked | Sending request when peer is choking us | Outbound message validation | Buffer request until unchoked |
| Interest Without Need | Claiming interest in peer with no needed pieces | Bitfield intersection check | Send not-interested message |
| Choke Oscillation | Rapid choke/unchoke cycles | State change frequency tracking | Implement choke state dampening |
| Stale Interested State | Remaining interested after obtaining all pieces | Periodic interest validation | Send not-interested and update state |
| Request Pipeline Overflow | Too many outstanding requests | Pipeline depth tracking | Stop sending requests until pipeline drains |
| Bitfield Desync | Peer's have messages inconsistent with bitfield | Cross-reference have messages with bitfield | Request fresh bitfield update |

**Data Validation Edge Cases**

Data validation must handle various forms of corruption and malicious data while maintaining download progress. The challenge is distinguishing between innocent errors and malicious attacks.

| Edge Case | Data Issue | Validation Approach | Response Strategy |
|-----------|-------------|-------------------|------------------|
| Partial Hash Match | Piece data partially correct | Block-level validation where possible | Re-request specific corrupt blocks |
| Size Overflow | Piece larger than expected | Strict size limits before hash calculation | Reject oversized pieces, penalize peer |
| Encoding Issues | Binary data in string fields | Validate data encoding expectations | Accept binary strings, log encoding mismatches |
| Padding Bytes | Extra bytes in file pieces | Check against file boundary expectations | Handle padding according to torrent specification |
| Timestamp Anomalies | Future or ancient timestamp in metadata | Timestamp range validation | Accept valid torrents with suspicious timestamps |
| Unicode Issues | Non-UTF8 strings in text fields | Graceful encoding handling | Attempt common encodings, fallback to byte display |

⚠️ **Pitfall: Treating All Protocol Violations as Attacks**

Many developers implement overly strict protocol validation that disconnects peers for minor infractions. This approach reduces peer diversity and can prevent successful downloads. Instead, implement graduated responses: log minor violations, penalize moderate violations, and disconnect only for severe violations that threaten system integrity.

**Concurrency and Race Condition Edge Cases**

BitTorrent clients are inherently concurrent systems with multiple peer connections, piece management threads, and I/O operations running simultaneously. Race conditions and synchronization issues can lead to subtle bugs that only manifest under specific timing conditions.

| Race Condition | Scenario | Detection | Prevention |
|---------------|----------|-----------|------------|
| Piece Completion Race | Multiple peers complete same piece simultaneously | Duplicate completion detection | Atomic piece state transitions |
| Connection State Race | Peer state changes while processing messages | State consistency validation | Message processing serialization |
| Request Assignment Race | Multiple threads assign same block | Request tracking validation | Centralized request coordination |
| File Write Race | Concurrent writes to same file region | Write operation validation | Serialize writes per file region |
| Statistics Update Race | Concurrent updates to download statistics | Counter inconsistency detection | Atomic counter operations |
| Shutdown Race | Component shutdown during active operations | Operation completion tracking | Graceful shutdown coordination |

The key to preventing race conditions is implementing proper synchronization at the right granularity. Too coarse-grained locking reduces parallelism, while too fine-grained locking increases complexity and deadlock risk.

> **Decision: Fine-Grained vs. Coarse-Grained Locking**
> - **Context**: Need to balance concurrency performance with synchronization complexity
> - **Options Considered**: Single global lock, per-component locks, fine-grained field-level locks
> - **Decision**: Per-component locks with careful lock ordering
> - **Rationale**: Provides good parallelism without excessive complexity, follows lock hierarchy to prevent deadlocks
> - **Consequences**: Enables concurrent operations while maintaining predictable synchronization behavior

### Implementation Guidance

The error handling implementation requires careful coordination between all system components to ensure consistent error recovery behavior. The key insight is building error handling into the system architecture from the beginning rather than adding it as an afterthought.

**Technology Recommendations Table:**

| Component | Simple Option | Advanced Option |
|-----------|--------------|-----------------|
| Error Types | Standard Go errors with error wrapping | Custom error hierarchy with error codes and context |
| Retry Logic | Simple exponential backoff timers | Configurable backoff strategies with circuit breakers |
| Logging | Standard log package with structured fields | Structured logging with error correlation IDs |
| Monitoring | Basic error counters and timing metrics | Comprehensive metrics with error categorization |
| Recovery | Manual retry on specific error conditions | Automatic recovery with pluggable recovery strategies |

**Recommended File/Module Structure:**

```
internal/
  errors/
    types.go              ← Custom error types and error hierarchy
    recovery.go           ← Retry and recovery logic implementations
    circuit.go            ← Circuit breaker pattern implementation
  coordinator/
    errors.go             ← Component-specific error handling
    recovery.go           ← Component recovery procedures
  tracker/
    errors.go             ← Tracker communication error handling
    backoff.go            ← Tracker-specific retry logic
  peer/
    errors.go             ← Peer protocol error handling
    validation.go         ← Message and state validation
  piece/
    errors.go             ← Piece management error handling
    verification.go       ← Data integrity validation
```

**Infrastructure Starter Code:**

```go
// Package errors provides comprehensive error handling infrastructure
package errors

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// ErrorType categorizes different types of errors for appropriate handling
type ErrorType int

const (
    ErrorTypeNetwork ErrorType = iota
    ErrorTypeProtocol
    ErrorTypeData
    ErrorTypeTracker
    ErrorTypeFileSystem
    ErrorTypeConcurrency
)

// BitTorrentError wraps standard errors with additional context
type BitTorrentError struct {
    Type      ErrorType
    Component string
    Operation string
    Err       error
    Context   map[string]interface{}
    Timestamp time.Time
}

func (e *BitTorrentError) Error() string {
    return fmt.Sprintf("[%s:%s] %s: %v", e.Component, e.Operation, e.Type, e.Err)
}

func (e *BitTorrentError) Unwrap() error {
    return e.Err
}

// BackoffScheduler implements exponential backoff with jitter
type BackoffScheduler struct {
    baseDelay     time.Duration
    maxDelay      time.Duration
    multiplier    float64
    jitterPercent int
    currentDelay  time.Duration
    failures      int
    mutex         sync.RWMutex
}

func NewBackoffScheduler(base, max time.Duration, multiplier float64) *BackoffScheduler {
    return &BackoffScheduler{
        baseDelay:     base,
        maxDelay:      max,
        multiplier:    multiplier,
        jitterPercent: 20,
        currentDelay:  base,
    }
}

func (b *BackoffScheduler) NextDelay() time.Duration {
    b.mutex.Lock()
    defer b.mutex.Unlock()
    
    delay := b.currentDelay
    b.failures++
    
    // Calculate next delay with exponential backoff
    next := time.Duration(float64(b.currentDelay) * b.multiplier)
    if next > b.maxDelay {
        next = b.maxDelay
    }
    b.currentDelay = next
    
    // Add jitter to prevent thundering herd
    jitter := time.Duration(int64(delay) * int64(b.jitterPercent) / 100)
    jitterAmount := time.Duration(rand.Int63n(int64(jitter*2))) - jitter
    
    return delay + jitterAmount
}

func (b *BackoffScheduler) Reset() {
    b.mutex.Lock()
    defer b.mutex.Unlock()
    b.currentDelay = b.baseDelay
    b.failures = 0
}

// CircuitBreaker implements circuit breaker pattern for fail-fast behavior
type CircuitBreaker struct {
    name              string
    failureThreshold  int
    recoveryTimeout   time.Duration
    successThreshold  int
    state             CircuitState
    failures          int
    successes         int
    lastFailureTime   time.Time
    mutex             sync.RWMutex
}

type CircuitState int

const (
    CircuitClosed CircuitState = iota
    CircuitOpen
    CircuitHalfOpen
)

func NewCircuitBreaker(name string, failureThreshold int, recoveryTimeout time.Duration) *CircuitBreaker {
    return &CircuitBreaker{
        name:             name,
        failureThreshold: failureThreshold,
        recoveryTimeout:  recoveryTimeout,
        successThreshold: 2,
        state:           CircuitClosed,
    }
}

func (cb *CircuitBreaker) Execute(ctx context.Context, operation func() error) error {
    if !cb.canExecute() {
        return &BitTorrentError{
            Type:      ErrorTypeNetwork,
            Component: "circuit-breaker",
            Operation: cb.name,
            Err:       fmt.Errorf("circuit breaker open"),
            Timestamp: time.Now(),
        }
    }
    
    err := operation()
    cb.recordResult(err)
    return err
}

func (cb *CircuitBreaker) canExecute() bool {
    cb.mutex.RLock()
    defer cb.mutex.RUnlock()
    
    switch cb.state {
    case CircuitClosed:
        return true
    case CircuitOpen:
        return time.Since(cb.lastFailureTime) >= cb.recoveryTimeout
    case CircuitHalfOpen:
        return true
    default:
        return false
    }
}

func (cb *CircuitBreaker) recordResult(err error) {
    cb.mutex.Lock()
    defer cb.mutex.Unlock()
    
    if err != nil {
        cb.failures++
        cb.lastFailureTime = time.Now()
        
        if cb.failures >= cb.failureThreshold {
            cb.state = CircuitOpen
        }
    } else {
        cb.successes++
        
        if cb.state == CircuitHalfOpen && cb.successes >= cb.successThreshold {
            cb.state = CircuitClosed
            cb.failures = 0
            cb.successes = 0
        }
    }
}

// PeerQualityScorer tracks peer reliability and performance metrics
type PeerQualityScorer struct {
    scores map[string]*PeerScore
    mutex  sync.RWMutex
}

type PeerScore struct {
    ConnectionReliability float64
    DataQuality          float64
    TransferPerformance  float64
    ProtocolCompliance   float64
    LastUpdate          time.Time
}

func NewPeerQualityScorer() *PeerQualityScorer {
    return &PeerQualityScorer{
        scores: make(map[string]*PeerScore),
    }
}

func (pqs *PeerQualityScorer) UpdateScore(peerID string, metric string, value float64) {
    pqs.mutex.Lock()
    defer pqs.mutex.Unlock()
    
    score, exists := pqs.scores[peerID]
    if !exists {
        score = &PeerScore{
            ConnectionReliability: 0.5,
            DataQuality:          0.5,
            TransferPerformance:  0.5,
            ProtocolCompliance:   0.5,
        }
        pqs.scores[peerID] = score
    }
    
    // Exponential moving average with time-based decay
    alpha := 0.1
    decay := pqs.calculateDecay(score.LastUpdate)
    
    switch metric {
    case "connection":
        score.ConnectionReliability = score.ConnectionReliability*decay + value*alpha
    case "data":
        score.DataQuality = score.DataQuality*decay + value*alpha
    case "performance":
        score.TransferPerformance = score.TransferPerformance*decay + value*alpha
    case "protocol":
        score.ProtocolCompliance = score.ProtocolCompliance*decay + value*alpha
    }
    
    score.LastUpdate = time.Now()
}

func (pqs *PeerQualityScorer) calculateDecay(lastUpdate time.Time) float64 {
    hoursSince := time.Since(lastUpdate).Hours()
    return math.Pow(0.95, hoursSince) // 5% decay per hour
}
```

**Core Logic Skeleton Code:**

```go
// ErrorHandler coordinates error recovery across all system components
type ErrorHandler struct {
    backoffSchedulers map[string]*BackoffScheduler
    circuitBreakers   map[string]*CircuitBreaker
    qualityScorer     *PeerQualityScorer
    recoveryStrategies map[ErrorType]RecoveryStrategy
    eventBus          *EventBus
    mutex             sync.RWMutex
}

// HandleError processes errors and determines appropriate recovery actions
func (eh *ErrorHandler) HandleError(ctx context.Context, err error) RecoveryAction {
    // TODO 1: Unwrap error to extract BitTorrentError with type and context
    // TODO 2: Determine error severity and classification
    // TODO 3: Check circuit breaker state for the failing component
    // TODO 4: Calculate backoff delay based on error history
    // TODO 5: Select appropriate recovery strategy based on error type
    // TODO 6: Update peer quality scores if error involves specific peer
    // TODO 7: Publish error event to event bus for component coordination
    // TODO 8: Return recovery action with delay and retry parameters
    // Hint: Use type assertion to extract BitTorrentError details
}

// RecoverFromFailure executes recovery procedures for different failure types
func (eh *ErrorHandler) RecoverFromFailure(ctx context.Context, failureType ErrorType, component string) error {
    // TODO 1: Look up recovery strategy for the specific failure type
    // TODO 2: Check if circuit breaker allows recovery attempt
    // TODO 3: Calculate appropriate backoff delay before retry
    // TODO 4: Execute pre-recovery validation and cleanup
    // TODO 5: Attempt recovery operation with timeout and context
    // TODO 6: Update component state based on recovery success/failure
    // TODO 7: Record recovery metrics and update error statistics
    // TODO 8: Reset backoff scheduler on successful recovery
    // Hint: Recovery strategies may involve reconnection, re-announce, or state reset
}

// ValidateMessageIntegrity checks peer protocol messages for correctness
func ValidateMessageIntegrity(msg *Message, expectedType int, maxSize int) error {
    // TODO 1: Validate message is not nil and has required fields
    // TODO 2: Check message type matches expected type or is valid alternative
    // TODO 3: Verify message payload size is within reasonable bounds
    // TODO 4: Validate message-specific payload format and constraints
    // TODO 5: Check for known malformed message patterns from misbehaving clients
    // TODO 6: Return appropriate error type for different validation failures
    // Hint: Different message types have different validation requirements
}

// HandlePieceVerificationFailure manages data integrity failures
func (pm *PieceManager) HandlePieceVerificationFailure(pieceIndex int, peerID string, data []byte) error {
    // TODO 1: Log piece verification failure with hash mismatch details
    // TODO 2: Update peer quality score to reflect data integrity issue
    // TODO 3: Mark piece as incomplete and remove corrupted data
    // TODO 4: Cancel any outstanding requests for this piece from the same peer
    // TODO 5: Add peer to temporary blacklist for this piece
    // TODO 6: Request piece from alternative peers with higher quality scores
    // TODO 7: Update piece availability tracking to reflect failed verification
    // TODO 8: Notify statistics tracker of wasted bandwidth and retry
    // Hint: Consider whether this is first failure or repeated failure from same peer
}

// HandleConnectionFailure manages peer connection loss and recovery
func (pm *PeerManager) HandleConnectionFailure(peerID string, err error) error {
    // TODO 1: Catalog all outstanding requests for the failed peer connection
    // TODO 2: Update connection statistics and peer quality scores
    // TODO 3: Notify piece manager to reassign outstanding requests
    // TODO 4: Clean up connection state and release allocated resources
    // TODO 5: Determine if connection should be retried or peer should be avoided
    // TODO 6: Schedule reconnection attempt with appropriate backoff delay
    // TODO 7: Update peer availability for piece selection algorithms
    // TODO 8: Trigger tracker re-announce if too many connections have failed
    // Hint: Different error types suggest different retry strategies
}
```

**Language-Specific Hints:**

- Use Go's error wrapping with `fmt.Errorf("context: %w", err)` to maintain error chains for debugging
- Implement custom error types with methods like `Temporary() bool` and `Timeout() bool` for classification
- Use `context.WithTimeout()` for operations that need deadline-based failure detection
- Leverage Go's `sync.RWMutex` for protecting shared error state while allowing concurrent reads
- Use `time.NewTicker()` for implementing periodic error recovery attempts
- Implement proper cleanup with `defer` statements to ensure resources are released on error paths

**Milestone Checkpoint:**

After implementing comprehensive error handling:

1. **Test Network Failure Recovery**: Disconnect network during download, verify client recovers gracefully when connectivity returns
2. **Verify Tracker Failover**: Stop primary tracker, confirm client switches to backup trackers automatically
3. **Validate Data Corruption Handling**: Introduce corrupted piece data, verify client detects and recovers
4. **Test Peer Protocol Violations**: Send malformed messages, confirm client handles gracefully without crashing
5. **Monitor Error Metrics**: Verify error counters, backoff timers, and circuit breaker states update correctly
6. **Check Concurrency Safety**: Run under race detector (`go test -race`) to detect synchronization issues

**Debugging Tips:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Client hangs after errors | Deadlock in error recovery | Check goroutine stack traces | Review lock ordering, add timeouts |
| Excessive retry attempts | Backoff not working | Log retry delays and counts | Verify backoff implementation, add circuit breakers |
| Memory leaks during errors | Resources not cleaned up | Monitor memory usage over time | Add proper cleanup in error paths |
| Inconsistent error handling | Missing error checks | Review error propagation | Add error wrapping and consistent handling |
| Performance degradation | Too aggressive error recovery | Profile CPU and network usage | Tune backoff parameters, reduce retry frequency |
| Error message confusion | Poor error context | Examine error message quality | Add structured error information and correlation IDs |

The comprehensive error handling system transforms a fragile BitTorrent prototype into a robust client capable of operating reliably in production environments. By anticipating failures, implementing graduated recovery strategies, and learning from past experiences, the client can maintain download progress even in challenging network conditions with unreliable peers.


## Testing Strategy and Milestones

> **Milestone(s):** This section provides comprehensive testing guidance for all milestones (1-4) by establishing verification strategies, test checkpoints, and debugging approaches that ensure correct implementation across torrent parsing, tracker communication, peer protocols, and piece management.

Building a BitTorrent client involves complex interactions between multiple concurrent systems, making testing both crucial and challenging. Think of testing a BitTorrent client like **quality assurance for a busy restaurant kitchen** — you need to verify that individual stations (bencode parser, tracker client) work correctly in isolation, that the coordination between stations (peer manager talking to piece manager) flows smoothly, and that the entire kitchen can handle a full dinner rush (downloading a complete torrent with multiple peers) without breaking down. Just as a restaurant needs both ingredient quality checks and full service simulations, our BitTorrent client requires both unit tests for individual components and integration tests with real torrents and peer interactions.

The testing strategy must address several unique challenges in P2P systems: network failures can occur at any time, peers may behave unpredictably or maliciously, race conditions emerge from concurrent downloads, and timing-dependent behaviors make tests non-deterministic. Unlike traditional client-server applications where you control one endpoint, BitTorrent testing requires coordinating with external peers and trackers that may be unreliable or unavailable.

Our testing approach follows a **pyramid structure** with three distinct layers, each serving different verification goals and catching different classes of bugs. The foundation consists of fast, deterministic unit tests that verify individual component logic without external dependencies. The middle layer includes integration tests that exercise component interactions using controlled environments and mock peers. The apex contains end-to-end tests that download real torrents from live networks, validating the complete system under realistic conditions.

### Unit Testing Strategy

**Unit testing** for BitTorrent components focuses on verifying the correctness of individual algorithms and data transformations without network dependencies or concurrency complications. Think of unit tests as **testing individual tools in a workshop** — you verify that each hammer swings correctly, each saw cuts accurately, and each measuring tape gives precise readings before attempting to build furniture. Each component must demonstrate correct behavior under various inputs, edge cases, and error conditions in isolation.

The bencode parsing component serves as an excellent example of comprehensive unit testing. Bencode parsing involves recursive data structure handling, binary string processing, and precise byte-level operations that are perfect candidates for deterministic unit tests. The parser must handle well-formed inputs correctly, detect malformed inputs gracefully, and maintain proper state across recursive calls.

| Test Category | Test Cases | Expected Behavior | Validation Method |
|---------------|------------|-------------------|-------------------|
| Valid Bencode Strings | `4:spam`, `0:`, `10:helloworld` | Correct byte array extraction | Compare parsed bytes with expected values |
| Valid Bencode Integers | `i42e`, `i-42e`, `i0e` | Correct integer conversion | Verify parsed integer matches expected value |
| Valid Bencode Lists | `le`, `l4:spami42ee`, nested lists | Proper list structure and element parsing | Recursively validate list contents and types |
| Valid Bencode Dictionaries | `de`, `d4:spami42ee`, nested dicts | Correct key-value mapping and ordering | Verify all keys present with correct values |
| Malformed Input | `4:spa`, `i42`, `l4:spam`, incomplete structures | Appropriate error reporting | Verify specific error types and positions |
| Edge Cases | Empty strings, zero integers, maximum values | Boundary condition handling | Test limits of integer ranges and string lengths |

The torrent metadata extraction component requires tests that verify correct field extraction, info hash calculation, and validation logic. These tests use pre-computed torrent files with known metadata values, allowing precise verification of parsing accuracy.

| Component | Test Focus | Key Test Cases | Verification Strategy |
|-----------|------------|----------------|----------------------|
| `MetaInfo` Parsing | Field extraction accuracy | Single-file torrents, multi-file torrents, optional fields | Compare extracted fields with manually verified values |
| Info Hash Calculation | SHA1 computation precision | Various info dictionaries, edge cases | Pre-compute expected hashes using reference implementations |
| Piece Hash Extraction | Hash array processing | Different piece counts, hash validation | Verify individual piece hashes match expected values |
| URL Validation | Announce URL processing | HTTP/HTTPS URLs, multi-tracker lists | Validate URL parsing and format requirements |

Tracker communication unit tests focus on URL construction, parameter encoding, and response parsing without making actual HTTP requests. Mock HTTP responses allow testing various tracker response formats and error conditions in a controlled environment.

| Test Scenario | Input Data | Expected Output | Error Conditions |
|---------------|------------|-----------------|------------------|
| Announce URL Construction | `AnnounceRequest` with all fields | Properly encoded URL with all parameters | Missing required fields should cause validation errors |
| Binary Data Encoding | Info hash with special bytes | Correct percent-encoding | Invalid byte sequences should be handled gracefully |
| Compact Peer Parsing | 6-byte peer entries | `PeerInfo` structs with correct IP/port | Truncated or invalid peer data should trigger errors |
| Response Validation | Tracker response dictionary | Extracted interval, peer list, counts | Missing required fields should be detected |

The peer wire protocol components benefit enormously from unit testing because the protocol involves precise byte-level message formatting, state machine transitions, and binary data handling. These operations are deterministic and perfect for fast, reliable unit tests.

| Protocol Component | Test Categories | Critical Test Cases | Verification Approach |
|-------------------|-----------------|--------------------|-----------------------|
| `Handshake` Processing | Serialization/deserialization | Valid handshakes, wrong protocol, invalid lengths | Binary comparison with reference implementations |
| `Message` Framing | Length-prefixed parsing | All message types, keep-alive, oversized messages | Verify correct message reconstruction from bytes |
| `BitfieldOps` Operations | Bit manipulation | Set/get operations, edge piece indices, empty bitfields | Mathematical verification of bit operations |
| Peer State Machine | State transitions | All valid transitions, invalid transition attempts | State verification after each message |

⚠️ **Pitfall: Testing with Real Network Dependencies**

A common mistake in BitTorrent testing is writing unit tests that make actual network requests to trackers or attempt to connect to live peers. This approach creates flaky tests that fail due to network conditions, tracker availability, or external peer behavior rather than code bugs. Unit tests must be **completely isolated** from external dependencies.

Instead, use dependency injection to provide mock implementations during testing. The tracker client should accept an HTTP client interface that can be replaced with a mock during testing. Similarly, peer connections should use mock network connections that simulate various network conditions and peer behaviors without requiring actual TCP connections.

The piece management component presents unique unit testing challenges because it involves complex algorithms like rarest-first selection, concurrent request management, and content verification. However, these algorithms can be tested deterministically by providing controlled peer availability data and verifying piece selection decisions.

| Algorithm | Test Focus | Input Scenarios | Expected Behaviors |
|-----------|------------|-----------------|-------------------|
| Rarest-First Selection | Piece prioritization | Various availability distributions | Pieces with lowest availability selected first |
| Block Request Generation | Work distribution | Peer bitfields, pipeline limits | Correct block ranges, no overlapping requests |
| Piece Verification | Content integrity | Valid pieces, corrupted pieces, hash mismatches | Accept valid pieces, reject corrupt pieces with specific errors |
| Endgame Detection | Mode transition | Nearly complete downloads | Endgame activates at correct thresholds |

### Integration Testing

**Integration testing** verifies that BitTorrent components work correctly together, handling the complex interactions between concurrent subsystems without the unpredictability of live network conditions. Think of integration testing as **testing the entire restaurant service flow** — you verify that orders flow correctly from the front-of-house to the kitchen, that timing coordination works between different cooking stations, and that completed dishes reach the correct tables. Integration tests use controlled environments with mock peers and local trackers to simulate realistic BitTorrent interactions.

The key challenge in BitTorrent integration testing is managing the **concurrency and timing dependencies** that emerge when multiple components interact. The tracker client must coordinate with the peer manager to provide fresh peer lists, the peer manager must coordinate with the piece manager to request needed blocks, and the piece manager must coordinate with file I/O to assemble completed pieces. These interactions involve shared state, message passing, and timing-sensitive behaviors that require careful test orchestration.

Integration tests for BitTorrent use several **controlled environment techniques** to provide realistic interactions while maintaining test reliability and speed. Local tracker servers provide predictable peer discovery without external dependencies. Mock peer implementations simulate various peer behaviors including normal operation, slow responses, connection failures, and protocol violations. Dedicated test torrents with known content allow verification of complete download cycles with predictable outcomes.

| Integration Test Level | Components Involved | Test Environment | Validation Focus |
|-----------------------|-------------------|------------------|------------------|
| Parser + Tracker | `TorrentParser`, `TrackerClient` | Local HTTP server | Torrent metadata flows correctly to tracker requests |
| Tracker + Peer Manager | `TrackerClient`, `PeerManager` | Mock tracker with controlled peer lists | Peer discovery and connection establishment |
| Peer + Piece Manager | `PeerManager`, `PieceManager` | Mock peers with controlled bitfields | Block request coordination and piece assembly |
| Complete Download Flow | All components | Local tracker + mock peers + test torrent | End-to-end download with verification |

The **tracker integration tests** verify that the tracker client correctly processes torrent metadata to build announce requests and that the peer manager can establish connections with discovered peers. These tests use a local HTTP server that implements the tracker protocol, allowing precise control over tracker responses and timing.

> **Design Insight: Test Torrent Creation**
> Integration tests require specially crafted torrent files with predictable content and known piece hashes. Create test torrents from small text files with easily verifiable content (like "Hello, World!" repeated to fill exact piece boundaries). This allows tests to verify both the download mechanics and the final file content without requiring large data transfers.

| Test Scenario | Mock Tracker Behavior | Expected Client Response | Validation Points |
|---------------|----------------------|--------------------------|-------------------|
| Successful Announce | Return peer list with 3 mock peers | Establish connections to all 3 peers | Verify connection count, handshake completion |
| Tracker Error Response | Return failure reason | Implement backoff and retry | Verify exponential backoff timing |
| Compact Peer Format | Return binary peer list | Parse IP/port correctly | Verify individual peer addresses |
| Re-announce Timing | Specify announce interval | Re-announce at correct intervals | Verify timing and updated statistics |

**Peer protocol integration tests** focus on the interaction between peer connections and the piece management system. These tests use mock peer implementations that simulate various peer behaviors and network conditions to verify that the peer manager correctly handles the protocol state machine and block transfer logic.

The mock peer implementation provides controlled responses to handshakes, bitfield exchanges, and piece requests. This allows testing various peer behaviors including immediate responses, delayed responses, connection drops, and protocol violations. The piece manager's reaction to these behaviors validates the robustness of the peer coordination logic.

| Mock Peer Behavior | Test Purpose | Expected Client Reaction | Success Criteria |
|-------------------|--------------|-------------------------|------------------|
| Normal Operation | Happy path verification | Complete block downloads | All requested blocks received correctly |
| Slow Responses | Timeout handling | Request retry or peer replacement | Download continues with other peers |
| Connection Drop | Network failure resilience | Graceful connection cleanup | No resource leaks, request redistribution |
| Invalid Messages | Protocol violation handling | Connection termination with error logging | Malformed messages don't crash client |
| Choking Behavior | Upload permission changes | Pause requests when choked, resume when unchoked | Respect peer choking decisions |

**Piece assembly integration tests** verify the coordination between downloading blocks from multiple peers and assembling them into complete, verified pieces. These tests use multiple mock peers, each providing different pieces or blocks, to simulate the parallel downloading that occurs in real BitTorrent swarms.

The test setup involves creating a small test torrent with 3-4 pieces, then using mock peers that each have different piece availability. The integration test verifies that the piece manager correctly coordinates block requests across peers, assembles received blocks into pieces, verifies piece hashes, and writes completed pieces to the output file.

| Test Configuration | Peer Setup | Download Pattern | Validation Focus |
|-------------------|------------|------------------|------------------|
| Sequential Pieces | Each peer has consecutive pieces | Download pieces 0,1,2,3 in order | Basic piece assembly and file writing |
| Interleaved Availability | Peers have overlapping piece sets | Rarest-first selection with multiple sources | Piece selection algorithm and peer coordination |
| Partial Availability | Some pieces only available from one peer | Handle peer bottlenecks | Request distribution and peer management |
| Corrupted Pieces | Mock peer sends invalid piece data | Hash verification and re-request | Content integrity and error recovery |

### Milestone Verification Checkpoints

**Milestone verification checkpoints** provide concrete, measurable criteria for determining whether each implementation phase is working correctly before proceeding to the next phase. Think of these checkpoints as **quality gates in a manufacturing process** — each stage must meet specific standards before the product moves to the next assembly station. These checkpoints catch integration issues early and ensure that each milestone provides a solid foundation for subsequent development.

Each milestone checkpoint includes both **automated verification** through test execution and **manual verification** through observable behavior and output inspection. The automated tests provide rapid feedback on correctness, while manual verification ensures that the system exhibits expected behavior under realistic conditions.

#### Milestone 1: Torrent File Parsing Checkpoints

The first milestone focuses on torrent file parsing and metadata extraction. Successful completion requires accurate bencode decoding, correct field extraction, and proper info hash calculation. The verification checkpoints ensure that parsing works correctly with various torrent file formats and edge cases.

| Checkpoint | Verification Method | Expected Outcome | Troubleshooting |
|------------|-------------------|------------------|-----------------|
| Bencode Decoder | Unit test suite with 50+ test cases | All tests pass, handles all four bencode types | Check for off-by-one errors in length parsing |
| Torrent Metadata | Parse real torrent files from various sources | Extract announce URL, file info, piece data correctly | Verify info dict boundaries using hex editor |
| Info Hash Calculation | Compare with reference implementations | SHA1 hash matches expected values exactly | Ensure bencoding preserves exact byte sequences |
| Edge Case Handling | Test with malformed and unusual torrents | Graceful error reporting with specific error types | Check error message clarity and debugging information |

**Manual Verification for Milestone 1:**
```bash
# Test with a real torrent file
./bittorrent parse ubuntu-20.04.3-desktop-amd64.iso.torrent

Expected Output:
Announce URL: http://torrent.ubuntu.com:6969/announce
Info Hash: a3b5c7e9f1d3e5a7b9c1f3e5a7b9c1d3e5a7b9c1
File Name: ubuntu-20.04.3-desktop-amd64.iso
File Size: 2,785,017,856 bytes
Piece Length: 262,144 bytes
Piece Count: 10,630 pieces
```

⚠️ **Pitfall: Info Hash Calculation Errors**

The most common error in Milestone 1 is incorrect info hash calculation due to improper handling of the bencoded info dictionary boundaries. The info hash must be calculated from the **exact bencoded bytes** of the info dictionary, not from a re-encoded version. Parse the torrent file, locate the start and end positions of the info dictionary in the original bytes, then calculate SHA1 from that exact byte slice.

#### Milestone 2: Tracker Communication Checkpoints

The second milestone implements tracker communication for peer discovery. Successful completion requires proper URL construction, binary data encoding, HTTP request handling, and peer list parsing. The checkpoints verify that the client can successfully communicate with various tracker types and handle different response formats.

| Checkpoint | Verification Method | Expected Outcome | Common Issues |
|------------|-------------------|------------------|---------------|
| Announce URL Construction | Unit tests with various parameter combinations | Correctly formatted URLs with proper encoding | Binary data encoding, parameter ordering |
| HTTP Tracker Communication | Integration test with local tracker | Successful announce and peer list retrieval | URL encoding, content-type headers |
| Compact Peer Parsing | Unit tests with known peer data | Correct IP/port extraction from binary format | Endianness, byte order interpretation |
| Error Handling | Test with various tracker error responses | Appropriate error detection and reporting | HTTP status codes, tracker error messages |

**Manual Verification for Milestone 2:**
```bash
# Test tracker communication
./bittorrent announce ubuntu-20.04.3-desktop-amd64.iso.torrent

Expected Output:
Announcing to: http://torrent.ubuntu.com:6969/announce
Announce Response:
  Interval: 1800 seconds
  Complete: 147 seeders
  Incomplete: 23 leechers
  Peers discovered: 50 peers
  Sample peers:
    91.189.89.123:51413
    185.125.190.39:42000
    ...
```

The tracker communication checkpoint must verify that the client correctly handles both successful announcements and various error conditions. Test with multiple tracker URLs, including trackers that return errors, timeouts, and malformed responses.

#### Milestone 3: Peer Wire Protocol Checkpoints

The third milestone implements the peer wire protocol including handshakes, message framing, and state management. The checkpoints verify correct protocol implementation, message parsing, and peer state coordination.

| Checkpoint | Verification Method | Expected Outcome | Critical Validations |
|------------|-------------------|------------------|---------------------|
| Handshake Protocol | Integration test with mock peers | Successful handshake exchange | Protocol string, info hash verification |
| Message Parsing | Unit tests with all message types | Correct message deserialization | Length framing, message ID handling |
| Bitfield Exchange | Integration test with controlled peers | Accurate piece availability tracking | Bitfield size calculation, bit operations |
| State Machine | Unit tests covering all transitions | Proper choking/interested state management | State transition validation, invalid state handling |
| Request Pipeline | Integration test with mock peer responses | Multiple outstanding requests managed correctly | Pipeline depth limits, request/response correlation |

**Manual Verification for Milestone 3:**
```bash
# Test peer protocol with verbose logging
./bittorrent connect 91.189.89.123:51413 --info-hash=a3b5c7e9f1d3e5a7b9c1f3e5a7b9c1d3e5a7b9c1

Expected Output:
Connecting to peer 91.189.89.123:51413...
Sending handshake...
Received handshake response:
  Protocol: BitTorrent protocol
  Info Hash: a3b5c7e9f1d3e5a7b9c1f3e5a7b9c1d3e5a7b9c1 ✓
  Peer ID: -UT355W-[random bytes]
Received bitfield message: 1,327 bytes
Peer has 8,547 of 10,630 pieces (80.4%)
Sending interested message...
Peer state: not choking, we are interested
```

⚠️ **Pitfall: Message Framing Errors**

Peer protocol message framing errors are common and difficult to debug. Each message starts with a 4-byte big-endian length prefix followed by the message data. Partial reads from TCP connections can split messages across multiple read operations. Implement a message buffer that accumulates bytes until complete messages are available for processing.

#### Milestone 4: Piece Management & Seeding Checkpoints

The final milestone implements piece downloading, verification, and seeding capabilities. The checkpoints verify complete download functionality, content integrity, and upload capabilities.

| Checkpoint | Verification Method | Expected Outcome | Key Metrics |
|------------|-------------------|------------------|-------------|
| Piece Verification | Download test torrent with known content | All pieces pass SHA1 verification | Zero hash verification failures |
| Download Completion | End-to-end test with small torrent | Complete file matches original exactly | File size and content hash verification |
| Rarest-First Selection | Test with controlled peer availability | Pieces selected in order of increasing availability | Selection algorithm correctness |
| Upload Capability | Test with requesting peers | Successfully serve piece data to other peers | Upload statistics and peer satisfaction |
| Concurrent Downloads | Test with multiple simultaneous peers | Efficient bandwidth utilization across peers | Download rate and peer coordination |

**Manual Verification for Milestone 4:**
```bash
# Complete download test
./bittorrent download test-file.torrent --output=/tmp/downloaded

Expected Output:
Parsing torrent file...
Contacting tracker...
Discovered 15 peers, connecting...
Connected to 8 peers
Downloading...
Progress: [██████████████████████████████] 100% (1.2 MB/s)
Download complete: 2,785,017,856 bytes in 38m 42s
Verifying final file hash... ✓
Starting seeding mode...
Serving requests from 3 peers...
Uploaded 45.2 MB to other peers
```

The final checkpoint must verify that the downloaded file is **byte-for-byte identical** to the original. Calculate the SHA1 hash of the completed file and compare it with the expected hash from the torrent metadata. Additionally, test that the client can successfully serve piece requests from other peers, demonstrating complete BitTorrent protocol implementation.

> **Critical Success Criteria**
> The ultimate test of Milestone 4 success is downloading a real torrent file from a live BitTorrent network. Choose a small, popular torrent (like a Linux distribution ISO) that has many seeders. Your client should successfully discover peers, download all pieces, verify content integrity, and produce a file that matches the original exactly. This demonstrates that your implementation is compatible with the broader BitTorrent ecosystem.

### Implementation Guidance

The testing infrastructure for a BitTorrent client requires careful setup to provide reliable, fast feedback while covering the complex interactions between concurrent components. The testing strategy emphasizes **isolation** for unit tests, **controlled environments** for integration tests, and **realistic conditions** for milestone verification.

#### Technology Recommendations

| Testing Component | Simple Option | Advanced Option |
|------------------|---------------|-----------------|
| Unit Testing Framework | Go's built-in testing package (`testing`) | Ginkgo + Gomega for BDD-style tests |
| Mock Generation | Manual mock interfaces | GoMock for automatic mock generation |
| HTTP Mocking | httptest package for mock servers | WireMock for complex HTTP scenarios |
| Binary Test Data | Embedded hex strings in test code | Binary test files with loading utilities |
| Test Assertion Library | Basic Go assert functions | Testify for rich assertion methods |
| Concurrent Testing | Go race detector (`go test -race`) | Stress testing with multiple goroutines |

#### Recommended File Structure

The testing structure mirrors the main code organization while providing comprehensive coverage and easy test execution:

```
project-root/
  cmd/bittorrent/
    main.go
    main_test.go              ← CLI integration tests
  internal/
    torrent/
      parser.go
      parser_test.go          ← Bencode and metadata parsing tests
      testdata/               ← Sample torrent files for testing
        small.torrent
        multi-file.torrent
        malformed.torrent
    tracker/
      client.go
      client_test.go          ← Tracker communication unit tests
      mock_tracker_test.go    ← Local HTTP server for integration tests
    peer/
      connection.go
      connection_test.go      ← Peer protocol unit tests
      mock_peer_test.go       ← Mock peer implementations
      protocol_test.go        ← Message framing and state machine tests
    piece/
      manager.go
      manager_test.go         ← Piece selection and verification tests
      integration_test.go     ← Multi-component integration tests
  test/
    fixtures/                 ← Test data and utilities
      test_torrents/          ← Known good torrents for integration testing
      mock_tracker.go         ← Reusable mock tracker implementation
      mock_peer.go           ← Reusable mock peer implementation
    integration/              ← End-to-end integration tests
      download_test.go        ← Complete download cycle tests
      seeding_test.go         ← Upload and seeding tests
```

#### Infrastructure Starter Code

**Test Torrent Generator** (Complete implementation):

```go
// test/fixtures/torrent_generator.go
package fixtures

import (
    "crypto/sha1"
    "os"
    "path/filepath"
    "time"
    
    "github.com/yourusername/bittorrent/internal/torrent"
)

// TestTorrentConfig defines parameters for generating test torrents
type TestTorrentConfig struct {
    FileName     string
    Content      []byte
    PieceLength  int64
    AnnounceURL  string
}

// GenerateTestTorrent creates a complete torrent file with known content
// for testing download and verification logic
func GenerateTestTorrent(config TestTorrentConfig) (*torrent.MetaInfo, error) {
    // Calculate piece count and generate piece hashes
    pieceCount := (int64(len(config.Content)) + config.PieceLength - 1) / config.PieceLength
    var pieceHashes []byte
    
    for i := int64(0); i < pieceCount; i++ {
        start := i * config.PieceLength
        end := start + config.PieceLength
        if end > int64(len(config.Content)) {
            end = int64(len(config.Content))
        }
        
        hash := sha1.Sum(config.Content[start:end])
        pieceHashes = append(pieceHashes, hash[:]...)
    }
    
    // Build info dictionary
    info := torrent.Info{
        Name:        config.FileName,
        Length:      int64(len(config.Content)),
        PieceLength: config.PieceLength,
        Pieces:      pieceHashes,
    }
    
    // Calculate info hash (would use actual bencode encoding)
    infoHash := sha1.Sum([]byte("mock-info-dict")) // TODO: Implement proper bencoding
    
    metaInfo := &torrent.MetaInfo{
        Announce:     config.AnnounceURL,
        CreationDate: time.Now(),
        Comment:      "Generated for testing",
        CreatedBy:    "BitTorrent Test Suite",
        InfoHash:     infoHash,
        Info:         info,
    }
    
    return metaInfo, nil
}

// WriteTestFile creates the actual file content for a test torrent
func WriteTestFile(path string, content []byte) error {
    if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
        return err
    }
    return os.WriteFile(path, content, 0644)
}

// LoadTestTorrent loads a pre-built test torrent from the fixtures directory
func LoadTestTorrent(name string) (*torrent.MetaInfo, error) {
    path := filepath.Join("testdata", name+".torrent")
    return torrent.ParseFromFile(path)
}
```

**Mock Tracker Implementation** (Complete HTTP server):

```go
// test/fixtures/mock_tracker.go
package fixtures

import (
    "encoding/binary"
    "net"
    "net/http"
    "net/http/httptest"
    "net/url"
    "strconv"
    "time"
)

// MockTracker provides controlled tracker responses for testing
type MockTracker struct {
    server    *httptest.Server
    responses map[string]*TrackerResponse
    requests  []*TrackerRequest
}

type TrackerResponse struct {
    Interval   int
    Complete   int
    Incomplete int
    Peers      []MockPeer
    Error      string
}

type TrackerRequest struct {
    InfoHash   string
    PeerID     string
    Port       int
    Uploaded   int64
    Downloaded int64
    Left       int64
    Event      string
    Timestamp  time.Time
}

type MockPeer struct {
    IP   net.IP
    Port uint16
}

// NewMockTracker creates a local HTTP server that implements tracker protocol
func NewMockTracker() *MockTracker {
    mt := &MockTracker{
        responses: make(map[string]*TrackerResponse),
        requests:  make([]*TrackerRequest, 0),
    }
    
    mux := http.NewServeMux()
    mux.HandleFunc("/announce", mt.handleAnnounce)
    mt.server = httptest.NewServer(mux)
    
    return mt
}

// SetResponse configures the tracker response for a specific info hash
func (mt *MockTracker) SetResponse(infoHash string, response *TrackerResponse) {
    mt.responses[infoHash] = response
}

// GetRequests returns all announce requests received by the tracker
func (mt *MockTracker) GetRequests() []*TrackerRequest {
    return mt.requests
}

// URL returns the mock tracker's announce URL
func (mt *MockTracker) URL() string {
    return mt.server.URL + "/announce"
}

// Close shuts down the mock tracker server
func (mt *MockTracker) Close() {
    mt.server.Close()
}

func (mt *MockTracker) handleAnnounce(w http.ResponseWriter, r *http.Request) {
    // Parse announce request parameters
    query := r.URL.Query()
    
    request := &TrackerRequest{
        InfoHash:   query.Get("info_hash"),
        PeerID:     query.Get("peer_id"),
        Event:      query.Get("event"),
        Timestamp:  time.Now(),
    }
    
    // Parse numeric parameters with error handling
    if port, err := strconv.Atoi(query.Get("port")); err == nil {
        request.Port = port
    }
    if uploaded, err := strconv.ParseInt(query.Get("uploaded"), 10, 64); err == nil {
        request.Uploaded = uploaded
    }
    if downloaded, err := strconv.ParseInt(query.Get("downloaded"), 10, 64); err == nil {
        request.Downloaded = downloaded
    }
    if left, err := strconv.ParseInt(query.Get("left"), 10, 64); err == nil {
        request.Left = left
    }
    
    mt.requests = append(mt.requests, request)
    
    // Look up configured response
    response, exists := mt.responses[request.InfoHash]
    if !exists {
        http.Error(w, "Unknown torrent", http.StatusNotFound)
        return
    }
    
    // Return error response if configured
    if response.Error != "" {
        w.Header().Set("Content-Type", "text/plain")
        w.WriteHeader(http.StatusInternalServerError)
        w.Write([]byte(response.Error))
        return
    }
    
    // Build bencode response (simplified - would use proper bencode encoding)
    compactPeers := make([]byte, len(response.Peers)*6)
    for i, peer := range response.Peers {
        copy(compactPeers[i*6:], peer.IP.To4())
        binary.BigEndian.PutUint16(compactPeers[i*6+4:], peer.Port)
    }
    
    // TODO: Implement proper bencode encoding for production use
    w.Header().Set("Content-Type", "text/plain")
    w.Write([]byte("d8:intervali" + strconv.Itoa(response.Interval) + 
                   "e8:completei" + strconv.Itoa(response.Complete) + 
                   "e10:incompletei" + strconv.Itoa(response.Incomplete) + 
                   "e5:peers" + strconv.Itoa(len(compactPeers)) + ":"))
    w.Write(compactPeers)
    w.Write([]byte("e"))
}

// DefaultPeers returns a standard set of mock peers for testing
func DefaultPeers() []MockPeer {
    return []MockPeer{
        {IP: net.ParseIP("192.168.1.100"), Port: 6881},
        {IP: net.ParseIP("10.0.0.50"), Port: 51413},
        {IP: net.ParseIP("172.16.1.200"), Port: 42000},
    }
}
```

#### Core Testing Logic Skeletons

**Torrent Parsing Test Suite** (Structure with detailed TODOs):

```go
// internal/torrent/parser_test.go
package torrent

import (
    "testing"
    "path/filepath"
)

func TestBencodeDecoding(t *testing.T) {
    // TODO 1: Test valid string decoding - create test cases for various string lengths
    // TODO 2: Test valid integer decoding - include positive, negative, and zero values
    // TODO 3: Test valid list decoding - include empty lists and nested structures
    // TODO 4: Test valid dictionary decoding - include empty dicts and nested structures
    // TODO 5: Test malformed input handling - verify specific error types and positions
    // TODO 6: Test edge cases - empty strings, maximum integer values, deeply nested structures
    // Hint: Use table-driven tests with input/expected output pairs
}

func TestTorrentMetadataExtraction(t *testing.T) {
    // TODO 1: Load test torrent file from testdata directory
    // TODO 2: Parse torrent file using ParseFromFile function
    // TODO 3: Verify announce URL extraction matches expected value
    // TODO 4: Verify file info extraction (name, length, piece length)
    // TODO 5: Verify piece hash extraction and count
    // TODO 6: Test with multi-file torrents and verify file list parsing
    // Hint: Pre-compute expected values using a reference BitTorrent implementation
}

func TestInfoHashCalculation(t *testing.T) {
    // TODO 1: Load torrent file and extract raw bencode bytes
    // TODO 2: Locate info dictionary boundaries in the raw bytes
    // TODO 3: Calculate SHA1 hash of exact info dictionary bytes
    // TODO 4: Compare with expected hash from reference implementation
    // TODO 5: Test with multiple torrent files to ensure consistency
    // TODO 6: Verify that re-encoding doesn't change the hash
    // Hint: Use hex.DecodeString to create expected hash values from known torrents
}
```

**Integration Test Suite** (Structure with coordination logic):

```go
// test/integration/download_test.go
package integration

import (
    "context"
    "testing"
    "time"
    
    "github.com/yourusername/bittorrent/test/fixtures"
)

func TestCompleteDownloadCycle(t *testing.T) {
    // TODO 1: Create test torrent with known content using fixtures.GenerateTestTorrent
    // TODO 2: Start mock tracker with configured peer list using fixtures.NewMockTracker
    // TODO 3: Start mock peers that serve the test torrent content
    // TODO 4: Create BitTorrent client and initiate download
    // TODO 5: Monitor download progress and verify piece-by-piece completion
    // TODO 6: Verify final file content matches original exactly
    // TODO 7: Test seeding mode by having client serve requests from other mock peers
    // Hint: Use timeouts to prevent test hangs, typical integration test takes 10-30 seconds
}

func TestMultiPeerCoordination(t *testing.T) {
    ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
    defer cancel()
    
    // TODO 1: Create test torrent with multiple pieces (4-6 pieces)
    // TODO 2: Configure 3 mock peers with different piece availability patterns
    //         - Peer 1: has pieces 0, 2, 4
    //         - Peer 2: has pieces 1, 3, 5
    //         - Peer 3: has pieces 0, 1, 2 (overlapping)
    // TODO 3: Verify client discovers all peers through mock tracker
    // TODO 4: Monitor piece requests to ensure optimal distribution across peers
    // TODO 5: Verify rarest-first selection algorithm chooses pieces correctly
    // TODO 6: Simulate peer disconnection and verify client handles gracefully
    // TODO 7: Verify download completes successfully using remaining peers
    // Hint: Log all peer interactions to verify correct protocol behavior
}
```

#### Milestone Checkpoint Scripts

**Automated Verification Scripts**:

```bash
#!/bin/bash
# scripts/verify-milestone-1.sh

echo "=== Milestone 1: Torrent File Parsing Verification ==="

# Run unit tests
echo "Running bencode parsing tests..."
go test ./internal/torrent/... -v -run TestBencode

echo "Running torrent metadata extraction tests..."
go test ./internal/torrent/... -v -run TestMetadata

# Test with real torrent file
echo "Testing with real torrent file..."
./bittorrent parse test/fixtures/testdata/small.torrent > /tmp/parse_output.txt

# Verify expected output format
if grep -q "Announce URL:" /tmp/parse_output.txt && \
   grep -q "Info Hash:" /tmp/parse_output.txt && \
   grep -q "File Name:" /tmp/parse_output.txt; then
    echo "✓ Torrent parsing output format correct"
else
    echo "✗ Torrent parsing output format incorrect"
    cat /tmp/parse_output.txt
    exit 1
fi

echo "=== Milestone 1 Verification Complete ==="
```

The testing strategy provides a comprehensive foundation for building a robust BitTorrent client through systematic verification at multiple levels. Unit tests catch algorithmic errors and edge cases early, integration tests verify component interactions in controlled environments, and milestone checkpoints ensure each implementation phase meets quality standards before proceeding. This layered approach builds confidence that the final system will work correctly in the complex, unpredictable environment of real BitTorrent networks.


## Debugging Guide

> **Milestone(s):** This section provides essential debugging guidance for all milestones (1-4) by establishing systematic approaches to diagnose and fix common issues across torrent parsing, tracker communication, peer protocols, and piece management.

Building a BitTorrent client involves multiple interacting systems: binary protocol parsing, concurrent networking, cryptographic verification, and complex state management. When things go wrong, the symptoms can be misleading and the root causes can be buried deep in protocol minutiae or race conditions. This debugging guide provides systematic approaches to diagnose and resolve the most common issues that learners encounter when implementing each milestone.

Think of debugging a BitTorrent client like being a detective investigating a crime scene. Each symptom is a clue that points toward potential causes, but the evidence can be scattered across multiple components. A failed download might be caused by incorrect Bencode parsing, malformed tracker requests, peer protocol violations, or piece verification failures. The key is building a systematic methodology to collect evidence, form hypotheses, and test them methodically.

The debugging process follows a consistent pattern across all components: observe the symptoms, collect diagnostic data, form hypotheses about root causes, test the hypotheses with targeted experiments, and verify the fixes. This section provides specific guidance for each milestone's unique challenges while establishing common debugging practices that apply across the entire system.

### Torrent Parsing Debug Guide

Torrent file parsing forms the foundation of the entire BitTorrent client, and parsing errors can manifest as mysterious failures much later in the download process. The Bencode format, while simple in concept, has several edge cases that can trap unwary implementers. Understanding these pitfalls and their debugging approaches is crucial for building a robust foundation.

#### Bencode Parsing Issues

The most common torrent parsing failures stem from incorrect Bencode decoding, which can produce subtly corrupted data that only manifests when used by other components. These issues require systematic diagnosis starting from the lowest level of byte interpretation.

**Symptom-Cause-Fix Mapping for Bencode Parsing:**

| Symptom | Likely Cause | Diagnostic Steps | Fix |
|---------|--------------|------------------|-----|
| Parser crashes on "invalid character" | Treating binary data as UTF-8 string | Print raw bytes around error offset, check for non-printable characters | Use `[]byte` for all Bencode strings, not Go `string` type |
| Info hash doesn't match tracker expectations | Including extra bytes in info dict boundaries | Log exact byte ranges used for hash calculation, compare with reference implementation | Parse info dict as complete bencoded value, not individual fields |
| Piece count mismatch | Incorrect piece hash extraction from concatenated field | Verify pieces field length is multiple of 20, count extracted hashes | Extract 20-byte chunks sequentially, verify total length |
| Random parsing failures on large files | Buffer overruns or incorrect offset tracking | Add offset logging to every parse step, verify buffer boundaries | Implement proper bounds checking in decoder |
| Nested dictionary corruption | Stack overflow in recursive parsing | Monitor recursion depth, check for circular references | Add recursion depth limits and iterative parsing for deep nesting |

The most insidious Bencode parsing bug involves the treatment of binary strings. Many implementers assume Bencode strings contain text data and convert them to their language's native string type. However, Bencode strings frequently contain binary data (like piece hashes or info dictionaries), and any encoding conversion corrupts this data.

> **Critical Insight**: Bencode strings are binary-safe byte sequences, not text strings. Always parse them as byte arrays and only convert to text when you know the content is actually textual (like file names or announce URLs).

#### Info Hash Calculation Debugging

The info hash serves as the unique identifier for a torrent and must be calculated from the exact bencoded bytes of the info dictionary. Any deviation in these bytes produces a different hash, causing tracker and peer rejections. Info hash calculation errors are particularly difficult to debug because the hash looks "correct" but doesn't match what other clients compute.

**Info Hash Validation Process:**

1. **Isolate the exact info dictionary bytes**: The info hash must be computed from the exact bencoded representation of the info dictionary as it appears in the torrent file, not from a re-encoding of parsed data
2. **Verify byte boundaries**: Log the start and end positions of the info dictionary within the torrent file and ensure no extra bytes are included
3. **Cross-reference with known tools**: Use standard tools like `transmission-show` or online torrent analyzers to get the expected info hash for comparison
4. **Validate the SHA1 calculation**: Ensure you're using SHA1 (not SHA256 or other hash functions) and computing it over the raw bytes
5. **Check for encoding issues**: Verify that no character encoding conversion has occurred on the info dictionary bytes

A common debugging technique is to export the exact bytes used for hash calculation to a file and compute the SHA1 manually using command-line tools to verify your calculation logic.

#### Torrent Metadata Extraction Issues

Once Bencode parsing succeeds, extracting meaningful metadata from the parsed structures presents its own challenges. The BitTorrent specification has evolved over time, leading to multiple ways to represent the same information, and real-world torrent files often contain unexpected variations.

**Metadata Extraction Debug Checklist:**

| Component | Validation Check | Expected Behavior | Common Errors |
|-----------|-----------------|-------------------|---------------|
| Announce URL | URL format validation | Valid HTTP/HTTPS/UDP URL | Missing protocol, malformed URL encoding |
| File Information | Single vs multi-file detection | Length field XOR files array present | Both or neither present, incorrect total size calculation |
| Piece Information | Piece count calculation | `(pieces field length) / 20 == ceiling(total size / piece length)` | Off-by-one errors, integer division rounding |
| Creation Date | Unix timestamp validation | Valid time within reasonable range | Negative timestamps, far-future dates |
| Encoding Fields | Character set handling | UTF-8 for modern torrents, other encodings for legacy | Incorrect encoding assumptions |

The piece count validation deserves special attention because it involves multiple related calculations. The number of piece hashes in the pieces field should match the number of pieces required to cover the total file size given the piece length. Mismatches indicate either parsing errors or corrupted torrent files.

### Network Protocol Debugging

Network protocol debugging requires understanding both the HTTP tracker protocol and the BitTorrent peer wire protocol. Network issues can manifest as connection failures, timeouts, protocol violations, or data corruption. The key to effective network debugging is systematic observation of the actual bytes sent and received, not just the application-level interpretation of those bytes.

#### Tracker Communication Issues

Tracker communication failures are often the first network-related problems encountered during implementation. These issues typically stem from incorrect URL encoding, missing parameters, or mishandling of the tracker response format.

**Tracker Debug Investigation Process:**

1. **Capture the exact HTTP request**: Log the complete URL with all parameters to verify proper construction
2. **Verify URL encoding**: Ensure binary data (like info hash) is percent-encoded correctly for URLs
3. **Check parameter completeness**: Confirm all required parameters (info_hash, peer_id, port, uploaded, downloaded, left) are present
4. **Validate response parsing**: Log the raw tracker response before attempting to parse it
5. **Handle error responses**: Check for error messages in tracker responses before assuming success

The most frequent tracker communication bug involves incorrect URL encoding of the info hash. The 20-byte SHA1 hash contains binary data that must be percent-encoded for inclusion in HTTP URLs, but many implementers use incorrect encoding methods.

**Tracker Communication Debug Table:**

| Symptom | Diagnostic Command | Expected Result | Common Fix |
|---------|-------------------|-----------------|------------|
| "Invalid info_hash" error | `curl -v "tracker_url"` with parameters | HTTP 200 with peer list | Correct URL encoding of binary info hash |
| Empty peer list | Check `left` parameter value | Should be > 0 for incomplete downloads | Set `left` to remaining bytes, not total bytes |
| Tracker timeouts | Network connectivity test to tracker | Successful connection | Implement proper timeout handling and retries |
| Malformed peer data | Hex dump of peers field in response | 6-byte entries (4 bytes IP + 2 bytes port) | Handle compact peer format correctly |
| Authentication failures | Verify announce URL from torrent | URL should match exactly | Use announce URL from torrent, not guessed URL |

#### Peer Wire Protocol Debugging

The BitTorrent peer wire protocol is more complex than HTTP tracker communication, involving stateful connections, message framing, and binary protocol adherence. Peer protocol debugging requires understanding the exact message formats and state machine transitions.

**Peer Protocol Debug Strategy:**

The peer protocol debugging process follows a layered approach, starting from the lowest level (TCP connection) and working up through handshakes, message framing, and application logic.

**Connection Establishment Debug Steps:**

1. **TCP connection verification**: Confirm that TCP connections to peers can be established successfully
2. **Handshake exchange validation**: Verify that both sides of the handshake follow the exact 68-byte format
3. **Protocol string matching**: Ensure the 19-byte protocol identifier matches exactly
4. **Info hash verification**: Confirm that handshake info hashes match the torrent being downloaded
5. **Peer ID exchange**: Validate that peer IDs are exchanged correctly and stored for connection tracking

**Message Framing Debug Process:**

Message framing errors are particularly insidious because they can cause the entire connection to become unsynchronized, leading to cascading parsing failures for subsequent messages.

| Framing Issue | Symptoms | Debug Approach | Solution |
|---------------|----------|----------------|----------|
| Length prefix errors | Messages appear corrupted or truncated | Log raw bytes of length prefix, verify network byte order | Use `binary.BigEndian` for 4-byte length prefix |
| Partial message reads | Parser hangs or fails randomly | Implement complete message buffering before parsing | Read exact message length bytes before processing |
| Message type confusion | Unexpected message type errors | Log message ID byte after length prefix | Verify message ID matches expected protocol values |
| Keep-alive handling | Connection timeouts or unexpected closes | Monitor for zero-length messages | Handle keep-alive messages (length 0) without message ID |
| Buffer boundary errors | Corruption at message boundaries | Validate buffer management between messages | Reset buffers properly between message reads |

#### State Machine Debugging

The peer protocol involves a complex state machine governing when requests can be sent and received. State machine violations can cause peers to disconnect or refuse to serve data.

**Peer State Machine Debug Matrix:**

| Current State | Attempted Action | Expected Behavior | Debug Check |
|---------------|------------------|-------------------|-------------|
| `peer_choking = true` | Send request message | Should be rejected/ignored | Verify choke state before sending requests |
| `am_interested = false` | Expect piece messages | Peer should not send data | Send interested message before expecting data |
| `bitfield` not received | Send requests for pieces | Should fail or be ignored | Wait for bitfield after handshake |
| Connection just established | Send requests immediately | Should fail | Complete handshake and bitfield exchange first |
| Multiple pending requests | Connection suddenly closes | Likely protocol violation | Check request pipelining limits |

### Concurrency and State Issues

BitTorrent clients are inherently concurrent systems, managing multiple peer connections, piece downloads, and verification processes simultaneously. Concurrency bugs are among the most challenging to debug because they often involve race conditions that only manifest under specific timing conditions or high load.

#### Race Condition Identification

Race conditions in BitTorrent clients typically occur around shared data structures like piece state, peer connection management, and download progress tracking. These bugs are notoriously difficult to reproduce and debug because they depend on specific timing of concurrent operations.

**Common Race Condition Patterns:**

| Race Condition Type | Manifestation | Detection Method | Prevention Strategy |
|---------------------|---------------|------------------|---------------------|
| Piece state corruption | Same piece downloaded multiple times or verification failures | Log all piece state changes with timestamps and goroutine IDs | Use mutex protection around piece state updates |
| Connection map corruption | Peer connections appear/disappear randomly | Monitor connection map size and detect unexpected changes | Synchronize all connection map access with read-write locks |
| Download statistics inconsistency | Progress reporting jumps or decreases | Validate statistics consistency at regular intervals | Use atomic operations for counter updates |
| Request duplication | Multiple requests for same block sent to different peers | Log all outgoing requests with peer identification | Coordinate block requests through central manager |
| Verification race | Piece verification runs on partially downloaded data | Check for incomplete pieces entering verification | Ensure atomic piece completion before verification |

#### Deadlock Detection and Prevention

Deadlocks can occur when multiple goroutines wait for locks held by each other, typically involving the piece manager, peer connections, and shared state coordination. Deadlock detection requires understanding the lock acquisition order across different components.

**Deadlock Prevention Strategy:**

1. **Establish lock ordering**: Define a consistent order for acquiring multiple locks across all components
2. **Minimize lock scope**: Hold locks for the shortest possible duration to reduce contention
3. **Use timeouts**: Implement timeout-based lock acquisition to detect potential deadlocks
4. **Monitor lock contention**: Log lock acquisition and release events during debugging
5. **Separate read and write operations**: Use read-write locks to allow concurrent reads when possible

#### State Synchronization Issues

The BitTorrent client maintains complex shared state across multiple concurrent operations, and synchronization failures can lead to inconsistent views of download progress, peer availability, and piece completion status.

**State Synchronization Debug Approach:**

| Component | Shared State | Synchronization Mechanism | Common Issues |
|-----------|--------------|---------------------------|---------------|
| PieceManager | Piece completion status | Read-write mutex on piece map | Pieces marked complete before verification |
| PeerManager | Connection status and capabilities | Connection map mutex | Stale connection references |
| Statistics | Download/upload counters | Atomic operations | Inconsistent progress reporting |
| BitfieldOps | Available piece tracking | Bitfield-level locking | Race between bitfield updates and reads |
| EventBus | Event distribution | Channel-based message passing | Blocked subscribers causing deadlocks |

#### Memory Leak Detection

Long-running BitTorrent clients can develop memory leaks through accumulated goroutines, unclosed connections, or retained data structures. Memory leak debugging requires systematic monitoring of resource usage patterns.

**Memory Leak Investigation Process:**

1. **Monitor goroutine count**: Use `runtime.NumGoroutine()` to detect goroutine leaks
2. **Track connection lifecycle**: Ensure all network connections are properly closed
3. **Validate buffer management**: Check for growing buffers that are never reset
4. **Monitor piece data retention**: Ensure completed pieces don't retain unnecessary data
5. **Check event subscriber cleanup**: Verify event bus subscribers are unregistered properly

### Debugging Tools and Techniques

Effective BitTorrent debugging requires a combination of logging strategies, network monitoring tools, and systematic testing approaches. The distributed and concurrent nature of BitTorrent makes traditional debugging approaches insufficient.

#### Logging Strategy

A comprehensive logging strategy is essential for BitTorrent debugging because many issues only become apparent through analysis of event sequences across multiple components and connections.

**Structured Logging Framework:**

| Log Level | Component | Event Types | Example Information |
|-----------|-----------|-------------|-------------------|
| DEBUG | Bencode Parser | Parse events, offset tracking | "Parsing dictionary at offset 1234, found 5 keys" |
| INFO | Tracker Client | Announce requests/responses | "Announce successful: 25 peers returned, interval 1800s" |
| INFO | Peer Manager | Connection lifecycle | "Connected to peer 192.168.1.5:6881, capabilities: extension protocol" |
| INFO | Piece Manager | Piece completion | "Piece 42 verified successfully, 847/1000 pieces complete" |
| WARN | Protocol Handler | Protocol violations | "Peer 192.168.1.5 sent request while choked, ignoring" |
| ERROR | All Components | Error conditions | "Failed to verify piece 15: hash mismatch, requesting from different peer" |

#### Network Traffic Analysis

Network-level debugging provides ground truth about what data is actually being sent and received, independent of application-level interpretation.

**Network Analysis Tools:**

| Tool | Use Case | Command Example | Interpretation |
|------|---------|-----------------|----------------|
| Wireshark | Complete protocol analysis | GUI-based capture and filtering | Examine exact bytes, timing, TCP behavior |
| tcpdump | Command-line packet capture | `tcpdump -i eth0 -s 0 host tracker.example.com` | Verify tracker communication |
| netstat | Connection state monitoring | `netstat -an \| grep :6881` | Check peer connection status |
| ss | Modern connection analysis | `ss -tuln` | Monitor listening sockets and connections |
| curl | Manual HTTP testing | `curl -v "http://tracker/announce?info_hash=..."` | Test tracker requests manually |

#### State Inspection Techniques

BitTorrent clients maintain complex internal state that changes rapidly during operation. Effective debugging requires the ability to snapshot and analyze this state at key points.

**State Inspection Methods:**

1. **Component state dumps**: Implement methods to export complete component state in human-readable format
2. **Periodic state snapshots**: Capture state at regular intervals to identify trends and anomalies
3. **Event correlation**: Link state changes to specific events across component boundaries  
4. **Consistency validation**: Implement invariant checking to detect state corruption early
5. **Interactive debugging interfaces**: Provide runtime access to internal state for investigation

#### Test Environment Setup

Debugging BitTorrent clients requires controlled test environments that can simulate various network conditions, peer behaviors, and failure scenarios.

**Controlled Testing Environment Components:**

| Component | Purpose | Implementation | Benefits |
|-----------|---------|----------------|----------|
| Mock Tracker | Simulate tracker responses | Local HTTP server with configurable responses | Test tracker error handling, peer list variations |
| Mock Peers | Simulate peer behavior | TCP servers implementing peer protocol | Test protocol edge cases, malformed messages |
| Network Simulation | Control network conditions | Traffic shaping, packet loss injection | Test timeout handling, connection recovery |
| Test Torrents | Known content validation | Generated torrents with predictable data | Verify piece assembly, hash calculation |
| Load Testing | Concurrent connection handling | Multiple simultaneous peer connections | Identify race conditions, resource leaks |

#### Performance Profiling

BitTorrent clients involve significant CPU usage for cryptographic operations, network I/O, and concurrent coordination. Performance profiling helps identify bottlenecks that might not cause functional failures but impact user experience.

**Profiling Strategy:**

1. **CPU profiling**: Identify hot spots in piece verification, protocol parsing, and coordination logic
2. **Memory profiling**: Track allocation patterns and identify potential leaks
3. **Goroutine profiling**: Monitor concurrent operation patterns and identify coordination bottlenecks  
4. **Network I/O profiling**: Measure throughput and identify network-related performance issues
5. **Lock contention analysis**: Identify synchronization bottlenecks that limit concurrency

#### Systematic Bug Reproduction

BitTorrent bugs often involve complex interactions between multiple components, making them difficult to reproduce reliably. Systematic reproduction techniques help isolate the root causes.

**Bug Reproduction Methodology:**

| Step | Approach | Tools | Documentation |
|------|---------|-------|---------------|
| Minimal reproduction case | Reduce to simplest failing scenario | Small test torrents, single peers | Record exact steps and environment |
| Environment isolation | Eliminate external variables | Local network, controlled peers | Document all dependencies |
| Timing manipulation | Control concurrent execution | Artificial delays, single-threaded mode | Identify timing-sensitive bugs |
| Input variation | Systematically vary inputs | Different torrent files, peer behaviors | Map failure boundaries |
| State injection | Start from known problematic states | Save/restore component state | Test specific scenarios |

#### Debug Output Organization

The volume of debug information in a BitTorrent client can be overwhelming without proper organization and filtering strategies.

**Debug Output Management:**

1. **Component-based filtering**: Enable/disable debugging for specific components
2. **Severity-based filtering**: Control output volume through log level selection
3. **Context correlation**: Tag all log messages with operation context (torrent ID, peer ID, piece index)
4. **Timeline reconstruction**: Ensure all debug output includes precise timestamps
5. **Cross-component tracing**: Use correlation IDs to track operations across component boundaries

### Implementation Guidance

This subsection provides practical debugging tools and techniques specifically tailored for Go implementation, along with complete infrastructure code for debugging support.

#### Technology Recommendations Table

| Debugging Aspect | Simple Option | Advanced Option |
|------------------|---------------|-----------------|
| Logging Framework | Standard `log` package with custom formatting | `github.com/sirupsen/logrus` or `go.uber.org/zap` |
| Network Analysis | `tcpdump` command-line tool | Wireshark GUI with BitTorrent protocol dissectors |
| Profiling | Built-in `go tool pprof` with `net/http/pprof` | Continuous profiling with `github.com/pyroscope-io/pyroscope` |
| Test Environment | Manual mock servers | `github.com/jarcoal/httpmock` for HTTP, custom TCP mocks |
| State Inspection | JSON marshal of structs | Custom debug HTTP endpoints with live state |

#### Recommended File Structure

```
project-root/
  cmd/
    bittorrent-debug/
      main.go                 ← debug CLI tool
  internal/
    debug/
      logger.go              ← structured logging infrastructure
      profiler.go            ← performance profiling utilities
      state_inspector.go     ← runtime state inspection
      mock_tracker.go        ← test tracker implementation
      mock_peer.go           ← test peer implementation
    torrent/
      parser_debug.go        ← torrent parsing debug utilities
    tracker/
      client_debug.go        ← tracker communication debugging
    peer/
      connection_debug.go    ← peer protocol debugging
    piece/
      manager_debug.go       ← piece management debugging
  test/
    fixtures/
      test_torrents/         ← known test torrent files
      expected_outputs/      ← expected parsing results
```

#### Complete Debugging Infrastructure Code

**Structured Logging System (`internal/debug/logger.go`):**

```go
package debug

import (
    "context"
    "fmt"
    "io"
    "log"
    "os"
    "runtime"
    "strings"
    "sync"
    "time"
)

// LogLevel represents logging severity levels
type LogLevel int

const (
    DEBUG LogLevel = iota
    INFO
    WARN
    ERROR
)

var levelNames = map[LogLevel]string{
    DEBUG: "DEBUG",
    INFO:  "INFO",
    WARN:  "WARN",
    ERROR: "ERROR",
}

// ComponentLogger provides structured logging with component identification
type ComponentLogger struct {
    component string
    level     LogLevel
    output    io.Writer
    mutex     sync.RWMutex
}

// BitTorrentLogger manages logging across all components
type BitTorrentLogger struct {
    loggers map[string]*ComponentLogger
    mutex   sync.RWMutex
}

var globalLogger = &BitTorrentLogger{
    loggers: make(map[string]*ComponentLogger),
}

// NewComponentLogger creates a logger for a specific component
func NewComponentLogger(component string) *ComponentLogger {
    globalLogger.mutex.Lock()
    defer globalLogger.mutex.Unlock()
    
    if logger, exists := globalLogger.loggers[component]; exists {
        return logger
    }
    
    logger := &ComponentLogger{
        component: component,
        level:     INFO, // Default level
        output:    os.Stdout,
    }
    
    globalLogger.loggers[component] = logger
    return logger
}

// SetLevel configures the minimum log level for this component
func (l *ComponentLogger) SetLevel(level LogLevel) {
    l.mutex.Lock()
    defer l.mutex.Unlock()
    l.level = level
}

// Debug logs debug-level messages with detailed context
func (l *ComponentLogger) Debug(format string, args ...interface{}) {
    l.logf(DEBUG, format, args...)
}

// Info logs informational messages
func (l *ComponentLogger) Info(format string, args ...interface{}) {
    l.logf(INFO, format, args...)
}

// Warn logs warning messages
func (l *ComponentLogger) Warn(format string, args ...interface{}) {
    l.logf(WARN, format, args...)
}

// Error logs error messages
func (l *ComponentLogger) Error(format string, args ...interface{}) {
    l.logf(ERROR, format, args...)
}

func (l *ComponentLogger) logf(level LogLevel, format string, args ...interface{}) {
    l.mutex.RLock()
    if level < l.level {
        l.mutex.RUnlock()
        return
    }
    output := l.output
    component := l.component
    l.mutex.RUnlock()
    
    // Get caller information for debugging
    _, file, line, ok := runtime.Caller(2)
    var caller string
    if ok {
        parts := strings.Split(file, "/")
        if len(parts) > 0 {
            caller = fmt.Sprintf("%s:%d", parts[len(parts)-1], line)
        }
    }
    
    timestamp := time.Now().Format("2006-01-02 15:04:05.000")
    levelName := levelNames[level]
    
    message := fmt.Sprintf(format, args...)
    logLine := fmt.Sprintf("[%s] %s [%s] %s: %s\n", 
        timestamp, levelName, component, caller, message)
    
    fmt.Fprint(output, logLine)
}

// SetGlobalLevel sets the log level for all components
func SetGlobalLevel(level LogLevel) {
    globalLogger.mutex.Lock()
    defer globalLogger.mutex.Unlock()
    
    for _, logger := range globalLogger.loggers {
        logger.SetLevel(level)
    }
}
```

**State Inspector (`internal/debug/state_inspector.go`):**

```go
package debug

import (
    "encoding/json"
    "fmt"
    "net/http"
    "runtime"
    "sync"
    "time"
)

// StateInspector provides runtime inspection of component state
type StateInspector struct {
    components map[string]StateProvider
    mutex      sync.RWMutex
    server     *http.Server
}

// StateProvider interface allows components to expose their state
type StateProvider interface {
    GetState() interface{}
    GetStatistics() map[string]interface{}
}

// SystemStats provides system-level debugging information
type SystemStats struct {
    Goroutines    int           `json:"goroutines"`
    MemoryUsage   MemoryStats   `json:"memory"`
    Uptime        time.Duration `json:"uptime"`
    Timestamp     time.Time     `json:"timestamp"`
}

type MemoryStats struct {
    Allocated   uint64 `json:"allocated"`
    TotalAlloc  uint64 `json:"total_alloc"`
    SystemMem   uint64 `json:"system_mem"`
    NumGC       uint32 `json:"num_gc"`
}

var globalInspector = &StateInspector{
    components: make(map[string]StateProvider),
}

var startTime = time.Now()

// RegisterComponent adds a component for state inspection
func RegisterComponent(name string, provider StateProvider) {
    globalInspector.mutex.Lock()
    defer globalInspector.mutex.Unlock()
    globalInspector.components[name] = provider
}

// StartDebugServer launches HTTP server for state inspection
func StartDebugServer(port int) error {
    mux := http.NewServeMux()
    
    // System state endpoint
    mux.HandleFunc("/debug/system", handleSystemState)
    
    // Component state endpoints
    mux.HandleFunc("/debug/components", handleComponentList)
    mux.HandleFunc("/debug/component/", handleComponentState)
    
    // Memory and goroutine debugging
    mux.HandleFunc("/debug/goroutines", handleGoroutineDebug)
    
    globalInspector.server = &http.Server{
        Addr:    fmt.Sprintf(":%d", port),
        Handler: mux,
    }
    
    return globalInspector.server.ListenAndServe()
}

func handleSystemState(w http.ResponseWriter, r *http.Request) {
    var memStats runtime.MemStats
    runtime.ReadMemStats(&memStats)
    
    stats := SystemStats{
        Goroutines: runtime.NumGoroutine(),
        MemoryUsage: MemoryStats{
            Allocated:  memStats.Alloc,
            TotalAlloc: memStats.TotalAlloc,
            SystemMem:  memStats.Sys,
            NumGC:      memStats.NumGC,
        },
        Uptime:    time.Since(startTime),
        Timestamp: time.Now(),
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(stats)
}

func handleComponentList(w http.ResponseWriter, r *http.Request) {
    globalInspector.mutex.RLock()
    defer globalInspector.mutex.RUnlock()
    
    components := make([]string, 0, len(globalInspector.components))
    for name := range globalInspector.components {
        components = append(components, name)
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]interface{}{
        "components": components,
        "count":      len(components),
    })
}

func handleComponentState(w http.ResponseWriter, r *http.Request) {
    componentName := r.URL.Path[len("/debug/component/"):]
    
    globalInspector.mutex.RLock()
    provider, exists := globalInspector.components[componentName]
    globalInspector.mutex.RUnlock()
    
    if !exists {
        http.Error(w, "Component not found", http.StatusNotFound)
        return
    }
    
    state := map[string]interface{}{
        "component":  componentName,
        "state":      provider.GetState(),
        "statistics": provider.GetStatistics(),
        "timestamp":  time.Now(),
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(state)
}

func handleGoroutineDebug(w http.ResponseWriter, r *http.Request) {
    // This would typically use pprof for detailed goroutine analysis
    count := runtime.NumGoroutine()
    
    response := map[string]interface{}{
        "goroutine_count": count,
        "timestamp":       time.Now(),
        "warning":         "Use 'go tool pprof http://localhost:port/debug/pprof/goroutine' for detailed analysis",
    }
    
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}
```

#### Core Logic Debugging Skeleton

**Bencode Parser Debugging (`internal/torrent/parser_debug.go`):**

```go
package torrent

// DebugDecoder wraps the standard decoder with extensive logging
type DebugDecoder struct {
    *Decoder
    logger *debug.ComponentLogger
}

// NewDebugDecoder creates a decoder with comprehensive debugging
func NewDebugDecoder(r io.Reader) *DebugDecoder {
    return &DebugDecoder{
        Decoder: NewDecoder(r),
        logger:  debug.NewComponentLogger("BencodeParser"),
    }
}

// DebugDecode parses with extensive logging of parse state
func (d *DebugDecoder) DebugDecode() (interface{}, error) {
    // TODO 1: Log initial parser state (offset, buffer size)
    // TODO 2: Call standard Decode() method
    // TODO 3: Log final state and parsed value type
    // TODO 4: If error occurs, log exact position and surrounding bytes
    // TODO 5: For successful parses, validate the parsed structure
    // Hint: Use d.logger.Debug() for detailed parse steps
    panic("implement DebugDecode")
}

// ValidateInfoHash compares calculated hash with expected value
func ValidateInfoHash(metaInfo *MetaInfo, expectedHex string) error {
    // TODO 1: Convert expected hex string to byte array
    // TODO 2: Compare with metaInfo.InfoHash using bytes.Equal
    // TODO 3: If mismatch, log both hashes in hex format for comparison
    // TODO 4: Return descriptive error with both hash values
    // Hint: Use fmt.Sprintf("%x", hash) for hex formatting
    panic("implement ValidateInfoHash")
}
```

#### Milestone Checkpoints

**After Milestone 1 - Torrent Parsing:**
```bash
# Verify Bencode parsing with debug output
go run cmd/debug/main.go parse test/fixtures/ubuntu.torrent

# Expected output should include:
# - All torrent metadata fields correctly extracted
# - Info hash matching standard tools: transmission-show ubuntu.torrent
# - Piece count calculation: total_size/piece_length (rounded up)
# - No parsing errors or warnings

# Manual verification:
transmission-show test/fixtures/ubuntu.torrent | grep "Hash:"
# Compare with your parser output
```

**After Milestone 2 - Tracker Communication:**
```bash
# Test tracker communication with debug logging
go run cmd/debug/main.go tracker test/fixtures/ubuntu.torrent

# Expected behavior:
# - HTTP GET request logged with all parameters
# - Successful response with peer list
# - Compact peer format correctly decoded to IP:port pairs
# - No URL encoding errors

# Manual verification:
curl -v "http://tracker-url/announce?info_hash=..." 
# Should return bencoded response with peer list
```

**After Milestone 3 - Peer Protocol:**
```bash
# Test peer connections with protocol debugging
go run cmd/debug/main.go peer test/fixtures/ubuntu.torrent

# Expected behavior:
# - TCP connections established to multiple peers
# - Handshake exchange completed successfully
# - Bitfield messages received and parsed
# - State machine transitions logged correctly

# Check debug server:
curl http://localhost:8080/debug/component/PeerManager
# Should show connected peers and their states
```

**After Milestone 4 - Complete Download:**
```bash
# Full download with comprehensive debugging
go run cmd/debug/main.go download test/fixtures/ubuntu.torrent output/

# Expected behavior:
# - All pieces downloaded and verified successfully
# - No hash verification failures
# - Multiple concurrent peer connections
# - Final file matches expected checksum

# Verify file integrity:
sha1sum output/ubuntu-file.iso
# Should match published hash
```

#### Debugging Tips Table

| Symptom | Likely Cause | Diagnosis Command | Fix |
|---------|--------------|------------------|-----|
| "Connection refused" to peers | Firewall blocking, wrong port | `telnet peer-ip peer-port` | Check firewall, verify port from tracker |
| Handshake timeout | Protocol mismatch, wrong info hash | Wireshark capture of handshake | Verify 68-byte format, check info hash |
| No piece requests sent | Peer choking, not interested | Debug peer state machine | Send interested message, wait for unchoke |
| Piece verification fails | Data corruption, wrong hash | Compare downloaded vs expected hash | Re-request piece, check hash calculation |
| Download stalls | All peers choking | Monitor choke/unchoke messages | Implement optimistic unchoking |
| Memory leak during download | Goroutines not cleaned up | `curl localhost:8080/debug/system` | Ensure connection cleanup on errors |
| Race condition crashes | Concurrent access to shared state | Run with `go run -race` | Add proper mutex protection |
| Tracker "Invalid request" | Missing/wrong parameters | Log complete URL before request | Check parameter encoding, completeness |


## Future Extensions

> **Milestone(s):** This section builds upon all four milestones (1-4) by outlining advanced BitTorrent features and optimizations that can be added to the core implementation, showing how the current architecture accommodates extensibility.

Building a complete BitTorrent client following the four core milestones creates a solid foundation, but the BitTorrent ecosystem offers many advanced features that can significantly enhance functionality, performance, and user experience. This section explores how our carefully designed architecture can accommodate these extensions without requiring fundamental rewrites.

Think of our current BitTorrent client as a reliable bicycle that gets you from point A to point B. The future extensions described here are like upgrading to an electric bike with GPS navigation, anti-theft systems, and performance monitoring. The core mechanics remain the same - you still pedal, steer, and brake - but the enhanced features make the journey faster, safer, and more enjoyable. Similarly, our core architecture of torrent parsing, tracker communication, peer protocols, and piece management remains intact while we layer on sophisticated enhancements.

The extensibility of our design stems from several architectural decisions made throughout the previous sections. The modular component structure with well-defined interfaces, the event bus for loose coupling between components, the shared state management system, and the pluggable error handling framework all contribute to making extensions straightforward to implement without disrupting existing functionality.

### Advanced Protocol Features

The BitTorrent protocol has evolved significantly since its inception, with numerous BitTorrent Enhancement Proposals (BEPs) adding powerful features for peer discovery, security, and convenience. Our current architecture provides excellent hooks for integrating these advanced protocol features.

#### Distributed Hash Table (DHT) Integration

The most significant limitation of our current tracker-based approach is the single point of failure represented by centralized trackers. **Distributed Hash Table (DHT)** technology, specifically the Kademlia-based Mainline DHT used by BitTorrent, eliminates this dependency by creating a decentralized peer discovery network.

Think of DHT as transforming our library network from one that relies on a central catalog service to a system where every library maintains connections to a few neighboring libraries and can find any book by asking neighbors, who ask their neighbors, creating a web of knowledge that requires no central authority. Each peer becomes both a user and a provider of the peer discovery service.

The DHT protocol involves several key components that integrate naturally with our existing architecture:

| DHT Component | Purpose | Integration Point | Data Structures |
|---------------|---------|------------------|-----------------|
| Node ID | 160-bit identifier for DHT participation | New field in `PeerInfo` | `NodeID [20]byte` |
| Routing Table | Maintains k-closest nodes in each bucket | New `DHTClient` component | `RoutingTable` with buckets |
| Query Engine | Implements find_node and get_peers queries | Extends `TrackerClient` interface | `DHTQuery` and `DHTResponse` types |
| Announce Protocol | Announces presence for info hashes | Parallel to HTTP announces | `DHTAnnounce` message type |

> **Decision: DHT as Tracker Fallback**
> - **Context**: Our current design relies solely on HTTP/UDP trackers which can become unavailable or censored
> - **Options Considered**: 
>   1. Replace tracker system entirely with DHT
>   2. Implement DHT as primary with tracker fallback  
>   3. Implement DHT as tracker fallback with gradual promotion
> - **Decision**: Implement DHT as tracker fallback that can be promoted to primary
> - **Rationale**: This approach maintains backward compatibility while providing resilience, and allows users to choose their preferred discovery method
> - **Consequences**: Requires dual peer discovery pipelines but provides maximum flexibility and fault tolerance

The DHT integration extends our existing `TrackerClient` interface by implementing a `DHTClient` that conforms to the same announce/response pattern. This allows the `PeerManager` to treat DHT and HTTP trackers uniformly, trying DHT when HTTP trackers fail or supplementing HTTP results with DHT peers for better swarm connectivity.

The technical implementation involves UDP message handling for the DHT protocol, maintaining a routing table of nearby nodes, and implementing the iterative query algorithm that contacts progressively closer nodes until finding peers for a specific info hash. The routing table requires periodic maintenance to remove stale nodes and discover new ones, which integrates well with our existing event bus system for coordinating background maintenance tasks.

#### Magnet Link Support

**Magnet links** represent a paradigm shift from torrent files to pure metadata URLs that enable instant sharing without distributing actual torrent files. Think of magnet links as GPS coordinates that lead you to a treasure location - instead of carrying a detailed treasure map (torrent file), you have just enough information to find other treasure hunters who can share their maps with you.

A typical magnet link contains:
- **Info hash**: The 20-byte SHA1 identifier for the torrent
- **Display name**: Human-readable file or torrent name  
- **Tracker URLs**: Optional list of backup trackers
- **File size**: Optional hint about total download size

| Magnet Component | Format | Purpose | Integration |
|------------------|--------|---------|-------------|
| Info Hash | `xt=urn:btih:[40-char-hex]` | Unique torrent identifier | Directly usable by DHT queries |
| Display Name | `dn=[url-encoded-name]` | User interface display | Temporary `MetaInfo.Info.Name` |
| Trackers | `tr=[url-encoded-tracker]` | Fallback peer discovery | Added to `MetaInfo.AnnounceList` |
| Exact Length | `xl=[byte-count]` | Progress tracking hint | Temporary `MetaInfo.Info.Length` |

The magnet link workflow requires significant extensions to our torrent parsing pipeline. Instead of starting with a complete `MetaInfo` structure parsed from a torrent file, magnet links provide only partial metadata. Our client must:

1. **Parse the magnet URL** to extract the info hash and optional metadata
2. **Create a skeleton MetaInfo** with the known information and placeholder values
3. **Query DHT and trackers** using the info hash to find peers
4. **Request metadata from peers** using the Extension Protocol (BEP 9)
5. **Reconstruct the complete MetaInfo** once metadata is received from peers
6. **Validate the reconstructed info hash** matches the original magnet link
7. **Proceed with normal download** using the complete metadata

This workflow integrates naturally with our existing architecture by extending the `ParseFromFile` function to also accept magnet URLs, creating a new `MetadataFetcher` component that requests missing information from peers, and using the event bus to coordinate the transition from partial to complete metadata.

#### Protocol Encryption and Obfuscation

Network-level protocol encryption addresses ISP throttling and deep packet inspection by making BitTorrent traffic appear as generic encrypted data rather than recognizable P2P patterns. **Message Stream Encryption (MSE)**, defined in BEP 3, provides this capability through a negotiated encryption layer.

Think of protocol encryption as having a conversation in a crowded restaurant where network monitors represent eavesdroppers. Without encryption, your BitTorrent conversation is clearly audible to anyone listening. With encryption, your conversation appears as meaningless whispers that reveal neither the topic (file being shared) nor the participants (peer IP addresses in piece requests).

The encryption negotiation occurs during the connection handshake phase, making it a natural extension to our existing `Connection.PerformHandshake()` method:

| Encryption Phase | Purpose | Data Exchanged | Integration Point |
|------------------|---------|----------------|------------------|
| Diffie-Hellman | Establish shared secret | Public keys (96-768 bytes) | Before protocol handshake |
| Method Selection | Choose encryption level | Plaintext/RC4/AES preference | Extends handshake options |
| Verification | Confirm successful setup | Encrypted hash verification | After key establishment |
| Stream Mode | Ongoing message protection | All subsequent messages | Wraps `MessageFramer` |

The implementation requires cryptographic libraries for Diffie-Hellman key exchange and stream ciphers, but integrates cleanly by creating an `EncryptedConnection` wrapper around our existing `Connection` type. The `MessageFramer` operates identically but reads/writes through the encryption layer instead of directly to the TCP connection.

#### Multi-Tracker and Tracker Tiers

Modern torrents often specify multiple trackers organized in tiers for redundancy and load balancing. **Multi-tracker support** ensures download success even when some trackers are unavailable and distributes the announce load across multiple servers.

Think of multi-tracker support as having multiple weather services for planning outdoor activities. Instead of relying on a single meteorologist who might be unavailable or incorrect, you consult several services simultaneously and use the best available information from all sources combined.

The tracker tier system works hierarchically:

| Tier Level | Behavior | Failure Handling | Success Criteria |
|------------|----------|------------------|------------------|
| Tier 0 (Primary) | Try all trackers simultaneously | Move to Tier 1 if all fail | Any tracker responds successfully |
| Tier 1+ (Backup) | Try after primary tier fails | Move to next tier if all fail | Continue with any working tracker |
| Cross-Tier | Periodically retry failed tiers | Promote working trackers | Maintain best available service |

This extends our `TrackerClient` to manage multiple tracker instances, implementing a `TrackerPool` that coordinates announces across tiers and aggregates peer lists from multiple sources. The event bus facilitates coordination between tracker instances, allowing successful responses from any tracker to update the global peer list while failed trackers are relegated to lower priority.

### Performance Optimizations

While our core implementation focuses on correctness and protocol compliance, real-world BitTorrent usage demands sophisticated optimizations for bandwidth efficiency, connection management, and intelligent piece selection that goes far beyond basic rarest-first algorithms.

#### Advanced Piece Selection Strategies

The **rarest-first algorithm** implemented in Milestone 4 provides a solid foundation, but production BitTorrent clients employ much more sophisticated piece selection that adapts to network conditions, peer behavior, and download progress phases.

Think of advanced piece selection as evolving from a simple "buy the least common baseball card first" strategy to a sophisticated trading algorithm that considers card condition, trader reliability, geographic proximity, and market trends. The goal remains completing your collection, but the strategy adapts to complex real-world conditions.

Advanced selection strategies include multiple complementary algorithms:

| Selection Strategy | Use Case | Algorithm | Benefits |
|-------------------|----------|-----------|----------|
| Random First Piece | Cold start with no pieces | Random selection from available | Immediate participation in swarm |
| Strict Priority | Important files first | User-defined file ordering | Critical content downloaded first |
| Availability-Based | Rare piece prioritization | Enhanced rarest-first with scoring | Better swarm health maintenance |
| Locality-Aware | Network topology optimization | Prefer peers with low latency | Reduced bandwidth costs |
| Completion Clustering | Sequential reading support | Group adjacent pieces | Enables streaming playback |

The implementation extends our existing `PieceSelector` with a strategy pattern that dynamically chooses algorithms based on download state:

```
Download Phase → Strategy Selection:
- Bootstrap (0-5% complete): Random first piece to join swarm quickly  
- Ramp-up (5-90% complete): Availability-based rarest-first for efficiency
- Endgame (90-100% complete): Multi-source requesting for final pieces
- Streaming mode: Completion clustering for sequential piece availability
```

**Locality-aware selection** optimizes for network topology by preferring peers that are geographically or topologically close. This reduces internet transit costs and improves download speeds by utilizing high-bandwidth local connections. The implementation maintains peer latency statistics and factors proximity into piece assignment decisions.

**Completion clustering** groups piece requests to enable streaming playback of media files. Instead of downloading pieces purely by rarity, the algorithm identifies the current playback position and ensures a buffer of sequential pieces ahead of the playback cursor. This enables video streaming while the file is still downloading.

#### Bandwidth Management and Traffic Shaping

Effective bandwidth management ensures BitTorrent activity doesn't overwhelm network connections while maximizing throughput within specified limits. **Adaptive bandwidth allocation** monitors network conditions and adjusts transfer rates dynamically.

Think of bandwidth management as managing water flow through a complex irrigation system. You have a fixed amount of water (bandwidth) that must be distributed among many fields (connections) while ensuring no field is flooded (connection overwhelmed) or drought-stricken (connection starved), and the distribution must adapt to changing weather conditions (network congestion).

The bandwidth management system operates at multiple levels:

| Management Level | Scope | Controls | Adaptation Triggers |
|------------------|-------|----------|---------------------|
| Global Limits | Entire client | Total upload/download caps | User configuration |
| Per-Torrent | Individual torrents | Torrent priority weighting | Progress-based adjustment |
| Per-Peer | Individual connections | Fair share allocation | Peer performance metrics |
| Congestion Control | Network conditions | Dynamic rate adjustment | RTT and loss detection |

**Token bucket algorithms** provide smooth rate limiting by allowing brief bursts above the average rate while maintaining long-term compliance with bandwidth limits. Each connection receives tokens at a controlled rate and consumes tokens for data transfer, preventing sustained overconsumption while accommodating natural traffic patterns.

**Congestion detection** monitors round-trip times and packet loss indicators to detect network congestion early and reduce transmission rates before connections timeout. This maintains higher overall throughput by avoiding the dramatic rate reductions that occur when TCP connections detect congestion through timeouts.

The implementation creates a `BandwidthManager` component that integrates with our existing `Connection` objects by intercepting read/write operations and applying rate limiting. The manager maintains token buckets per connection and globally, coordinating through the event bus to redistribute unused bandwidth allocation from idle connections to active transfers.

#### Connection Optimization and Pooling

Efficient connection management maximizes throughput while minimizing resource consumption through intelligent peer selection, connection reuse, and protocol optimizations.

**Connection pooling** maintains pools of established connections that can be quickly reassigned between torrents sharing the same peers. Think of this as maintaining relationships with reliable trading partners across multiple card collections - instead of introducing yourself repeatedly for each new collection, you leverage existing trust relationships for faster transactions.

The connection optimization strategies include:

| Optimization | Mechanism | Benefits | Implementation |
|--------------|-----------|----------|----------------|
| Keep-Alive | Maintain idle connections | Reduced handshake overhead | Extended connection timeouts |
| Multiplexing | Multiple torrents per connection | Better resource utilization | Protocol extension negotiation |
| Fast Extensions | Optimized message flow | Reduced round-trip delays | Fast peer and allowed fast messages |
| Pipelining Tuning | Dynamic pipeline depth | Optimal throughput/latency | RTT-based pipeline adjustment |

**Fast Extension Protocol** (BEP 6) reduces the time from connection establishment to active data transfer by allowing peers to immediately request pieces without waiting for complete bitfield exchange. This is particularly beneficial for peers that reconnect frequently or have high-latency connections.

**Dynamic pipeline tuning** adjusts the number of outstanding block requests based on measured round-trip times and bandwidth-delay product. High-latency connections benefit from deeper pipelines that keep the network path saturated, while low-latency connections require shallower pipelines to maintain responsiveness.

The optimization integrates with our existing `Connection` and `PeerManager` by adding connection lifecycle management, peer performance tracking, and protocol capability negotiation. The event bus coordinates connection reallocation between torrents and communicates performance metrics for optimization decisions.

### User Interface Integration

Our carefully designed core architecture provides excellent separation between the BitTorrent protocol engine and user interface concerns, enabling integration with diverse interface paradigms from command-line tools to sophisticated graphical applications.

#### API Design for GUI Integration

The key to successful GUI integration lies in designing clean APIs that expose BitTorrent functionality through well-defined interfaces while hiding protocol complexity from interface developers.

Think of the API design as creating a car dashboard for BitTorrent - the dashboard shows speed, fuel level, and engine temperature without requiring the driver to understand internal combustion, fuel injection timing, or cooling system pressure. Similarly, GUI developers need download progress, peer counts, and transfer rates without needing to understand Bencode parsing, piece verification, or message framing protocols.

The API architecture follows a layered approach:

| API Layer | Purpose | Interface Style | Data Format |
|-----------|---------|----------------|-------------|
| Core Engine | Protocol implementation | Internal Go interfaces | Native Go types |
| Service Layer | Business logic abstraction | Method calls with callbacks | Structured events |
| Transport Layer | External communication | REST/gRPC/WebSockets | JSON/Protocol Buffers |
| Client SDKs | Language-specific bindings | Native language idioms | Language-native types |

**Event-driven architecture** provides the foundation for responsive user interfaces by exposing BitTorrent state changes through structured events rather than requiring interfaces to poll for status updates. Our existing event bus naturally extends to external consumers:

```
Internal Event → API Event Translation:
- EventPieceCompleted → {type: "progress", torrent: "abc123", percent: 45.2}
- EventPeerConnected → {type: "peer_update", torrent: "abc123", peers: 12}  
- EventDownloadComplete → {type: "complete", torrent: "abc123", path: "/downloads/file.mp4"}
```

**REST API design** provides HTTP-based access for web interfaces and cross-platform compatibility. The API follows RESTful principles while accommodating BitTorrent-specific operations:

| Endpoint | Method | Purpose | Response |
|----------|---------|---------|----------|
| `/torrents` | GET | List all active torrents | Torrent summary array |
| `/torrents` | POST | Add new torrent | Created torrent details |
| `/torrents/{id}` | GET | Detailed torrent status | Complete torrent state |
| `/torrents/{id}/pause` | POST | Pause/resume torrent | Operation result |
| `/torrents/{id}/files` | GET | File list and priorities | File details array |
| `/events` | WebSocket | Real-time updates | Event stream |

#### Real-time Status Updates

Modern BitTorrent interfaces require real-time updates for download progress, peer connectivity, and transfer statistics. **WebSocket-based event streaming** provides efficient real-time communication without the overhead of continuous HTTP polling.

Think of real-time updates as the difference between checking your mailbox every few minutes versus having a notification system that immediately alerts you when mail arrives. WebSocket connections maintain persistent communication channels that deliver updates instantly as they occur within the BitTorrent engine.

The real-time update system streams multiple categories of information:

| Update Category | Frequency | Data Elements | Use Cases |
|----------------|-----------|---------------|-----------|
| Progress Updates | Per piece completion | Percentage, ETA, speed | Progress bars, statistics |
| Peer Information | Connection changes | Count, locations, quality | Network visualization |
| Transfer Metrics | Every few seconds | Rates, totals, ratios | Performance monitoring |
| System Events | As they occur | Errors, completions, alerts | User notifications |

**Update aggregation and throttling** prevents overwhelming clients with excessive messages during high-activity periods. The system batches related updates and applies rate limiting to maintain responsive interfaces without message flooding:

```
Aggregation Strategy:
- Progress updates: Maximum once per second per torrent
- Peer updates: Batch connection changes every 5 seconds  
- Transfer metrics: Smooth using exponential moving averages
- Critical events: Send immediately without throttling
```

The implementation extends our event bus with external subscription capabilities, WebSocket connection management, and update aggregation logic. The system maintains separate update channels for different client types, allowing mobile apps to receive minimal updates while desktop applications get comprehensive information.

#### Web-based Dashboard Implementation

A **web-based dashboard** provides universal access to BitTorrent functionality through standard web browsers, eliminating the need for platform-specific client installation while providing rich interactive capabilities.

Think of the web dashboard as transforming BitTorrent from a desktop application into a web service similar to how email evolved from desktop clients to Gmail - the core functionality remains the same, but web access provides universal availability, easy updates, and consistent cross-platform experience.

The web dashboard architecture separates concerns cleanly:

| Component | Technology | Responsibility | Integration |
|-----------|------------|----------------|-------------|
| Backend API | Go HTTP server | Protocol engine interface | Wraps our core components |
| Frontend SPA | React/Vue/Angular | User interface logic | Consumes REST API |
| Real-time Updates | WebSockets | Live data streaming | Event bus integration |
| Authentication | JWT/OAuth | Access control | Optional security layer |

**Single Page Application (SPA) design** provides responsive user experience through client-side rendering and AJAX communication. The SPA maintains application state locally and synchronizes with the BitTorrent engine through API calls and WebSocket events.

**Responsive design principles** ensure the dashboard works effectively across device sizes from smartphones to desktop monitors. The interface adapts layout, information density, and interaction patterns based on screen size and input capabilities.

Key dashboard features include:

- **Torrent management**: Add, remove, pause, and prioritize torrents
- **Progress monitoring**: Real-time download progress with speed charts
- **Peer visualization**: Geographic and network maps of peer connections  
- **File management**: Individual file priorities and selective downloading
- **Statistics dashboard**: Historical transfer rates and ratio tracking
- **Settings configuration**: Bandwidth limits, port configuration, and preferences

The web dashboard integrates with our core architecture by running the BitTorrent engine as a background service with HTTP API endpoints. The separation allows the protocol engine to operate independently while providing multiple interface options - web dashboard, desktop GUI, mobile app, or command-line interface - all accessing the same underlying functionality.

**Security considerations** for web interfaces include CSRF protection, input validation, and optional authentication systems. Since BitTorrent clients often run on local networks, the security model balances ease of use with protection against unauthorized access.

The implementation creates a new `cmd/web-server/` directory containing the HTTP server, API handlers, and web assets. The server embeds the frontend assets using Go's embed functionality for single-binary deployment while supporting development mode with external asset serving.

> **Key Design Insight**: The extensive use of interfaces and event-driven architecture in our core implementation makes these extensions possible without fundamental architectural changes. Each extension builds upon existing abstractions rather than requiring rewrites, demonstrating the value of thoughtful initial design.

These future extensions transform our functional BitTorrent client into a production-ready system capable of competing with commercial clients. The extensions maintain the educational value of the core implementation while showcasing how proper architecture enables sophisticated functionality through incremental enhancement rather than replacement.

### Implementation Guidance

This section provides practical guidance for implementing the advanced features described above, focusing on architectural integration points and technology choices that build upon our existing Go-based BitTorrent client.

#### Technology Recommendations

| Extension Category | Simple Approach | Advanced Approach |
|--------------------|----------------|-------------------|
| DHT Implementation | Basic Kademlia with UDP sockets | Full BEP 5 with routing table optimization |
| Magnet Link Parsing | Regular expressions + URL parsing | Dedicated parser with validation |
| Protocol Encryption | RC4 stream cipher | AES with multiple cipher support |
| Web API | Standard library HTTP + JSON | Gin/Echo framework + Protocol Buffers |
| Real-time Updates | Server-Sent Events | WebSocket with message queues |
| Frontend Framework | Vanilla JavaScript + HTMX | React/Vue with TypeScript |

#### Recommended Project Structure

The extensions integrate into our existing project structure while maintaining clear separation of concerns:

```
project-root/
  cmd/
    torrent-client/main.go    ← original CLI client
    web-server/main.go        ← new web dashboard server
  internal/
    torrent/                  ← core components (unchanged)
    tracker/                  ← existing tracker client
    peer/                     ← existing peer protocol
    piece/                    ← existing piece management
    dht/                      ← new DHT implementation
      client.go
      routing_table.go
      protocol.go
    magnet/                   ← new magnet link support
      parser.go
      metadata_fetcher.go
    encryption/               ← new protocol encryption
      mse.go
      cipher.go
    api/                      ← new web API
      handlers.go
      websocket.go
      models.go
    dashboard/                ← new web frontend
      static/
      templates/
  web/                        ← frontend assets
    src/
    dist/
  pkg/
    extensions/               ← reusable extension interfaces
```

#### DHT Implementation Starter Code

The DHT implementation extends our existing tracker interface pattern:

```go
// Package dht implements Distributed Hash Table peer discovery
package dht

import (
    "context"
    "net"
    "time"
    "your-project/internal/tracker"
)

// Client implements tracker.Client interface for DHT-based peer discovery
type Client struct {
    nodeID     [20]byte
    conn       *net.UDPConn
    routingTable *RoutingTable
    queries    map[string]*PendingQuery
    eventBus   *EventBus
    ctx        context.Context
    cancel     context.CancelFunc
}

// RoutingTable maintains k-closest nodes for DHT queries
type RoutingTable struct {
    nodeID  [20]byte
    buckets []Bucket
    mutex   sync.RWMutex
}

// Bucket contains up to K nodes with similar node IDs
type Bucket struct {
    nodes     []NodeInfo
    lastSeen  time.Time
    mutex     sync.RWMutex
}

// NodeInfo represents a DHT node
type NodeInfo struct {
    ID       [20]byte
    Addr     *net.UDPAddr
    LastSeen time.Time
}

// NewDHTClient creates a DHT client that implements tracker.Client interface
func NewDHTClient(port int) (*Client, error) {
    // TODO: Initialize UDP socket on specified port
    // TODO: Generate random 160-bit node ID
    // TODO: Create routing table with 160 buckets
    // TODO: Start maintenance goroutines for routing table
    // TODO: Bootstrap from well-known DHT nodes
    return nil, nil
}

// Announce implements tracker.Client.Announce for DHT peer discovery
func (c *Client) Announce(ctx context.Context, req tracker.AnnounceRequest) (*tracker.AnnounceResponse, error) {
    // TODO: Perform get_peers query for req.InfoHash
    // TODO: If no peers found, perform announce_peer to store our info
    // TODO: Aggregate peers from all responding nodes
    // TODO: Return tracker.AnnounceResponse with discovered peers
    // Hint: Use iterative deepening to find closest nodes
    return nil, nil
}
```

#### Magnet Link Support Implementation

Magnet link support extends our torrent parsing pipeline:

```go
// Package magnet handles magnet link parsing and metadata fetching
package magnet

import (
    "context"
    "net/url"
    "your-project/internal/torrent"
    "your-project/internal/peer"
)

// MagnetLink represents parsed magnet URI components
type MagnetLink struct {
    InfoHash    [20]byte
    DisplayName string
    Trackers    []string
    ExactLength int64
}

// MetadataFetcher retrieves missing torrent metadata from peers
type MetadataFetcher struct {
    infoHash     [20]byte
    peerManager  *peer.Manager
    eventBus     *EventBus
    ctx          context.Context
}

// ParseMagnetLink extracts components from magnet URI
func ParseMagnetLink(magnetURI string) (*MagnetLink, error) {
    // TODO: Parse URL and validate magnet: scheme
    // TODO: Extract xt parameter and decode info hash
    // TODO: Extract dn parameter for display name
    // TODO: Extract tr parameters for tracker URLs
    // TODO: Extract xl parameter for exact length
    // Hint: Use net/url.ParseQuery for parameter extraction
    return nil, nil
}

// FetchMetadata retrieves complete torrent metadata from peers
func (f *MetadataFetcher) FetchMetadata() (*torrent.MetaInfo, error) {
    // TODO: Connect to peers using DHT/tracker discovery
    // TODO: Send metadata extension handshake (BEP 9)
    // TODO: Request metadata pieces from multiple peers
    // TODO: Verify metadata integrity against info hash
    // TODO: Construct complete MetaInfo structure
    // Hint: Use Extension Protocol with ut_metadata message type
    return nil, nil
}
```

#### Web API Infrastructure

The web API provides HTTP and WebSocket endpoints for external client integration:

```go
// Package api provides HTTP API for external client integration
package api

import (
    "encoding/json"
    "net/http"
    "github.com/gorilla/websocket"
    "your-project/internal/coordination"
)

// Server provides HTTP API and WebSocket endpoints
type Server struct {
    client    *Client
    eventBus  *EventBus
    upgrader  websocket.Upgrader
    wsClients map[*websocket.Conn]bool
    mutex     sync.RWMutex
}

// TorrentStatus represents API torrent information
type TorrentStatus struct {
    InfoHash     string  `json:"info_hash"`
    Name         string  `json:"name"`
    Size         int64   `json:"size"`
    Downloaded   int64   `json:"downloaded"`
    Uploaded     int64   `json:"uploaded"`
    Progress     float64 `json:"progress"`
    DownloadRate float64 `json:"download_rate"`
    UploadRate   float64 `json:"upload_rate"`
    Peers        int     `json:"peers"`
    Status       string  `json:"status"`
}

// NewAPIServer creates HTTP server with REST and WebSocket endpoints
func NewAPIServer(client *Client, port int) *Server {
    // TODO: Initialize HTTP server with CORS middleware
    // TODO: Register REST endpoint handlers
    // TODO: Configure WebSocket upgrader
    // TODO: Subscribe to client event bus
    // TODO: Start WebSocket broadcast goroutine
    return nil
}

// HandleTorrents provides GET /torrents endpoint
func (s *Server) HandleTorrents(w http.ResponseWriter, r *http.Request) {
    // TODO: Collect status from all active torrents
    // TODO: Transform internal state to API format
    // TODO: Return JSON array of torrent status
    // Hint: Use json.NewEncoder for response formatting
}

// HandleWebSocket upgrades HTTP to WebSocket for real-time updates
func (s *Server) HandleWebSocket(w http.ResponseWriter, r *http.Request) {
    // TODO: Upgrade HTTP connection to WebSocket
    // TODO: Register client for event broadcasts
    // TODO: Handle client disconnection cleanup
    // TODO: Send periodic keep-alive pings
    // Hint: Use gorilla/websocket for protocol handling
}
```

#### Milestone Checkpoints

**DHT Integration Checkpoint:**
After implementing basic DHT support, verify functionality:
- Start client with DHT enabled: `./torrent-client --enable-dht example.torrent`
- Check logs for DHT bootstrap messages and routing table population
- Verify peer discovery works without HTTP trackers
- Expected: "DHT discovered 15 peers for torrent" log messages

**Magnet Link Checkpoint:**  
After implementing magnet link parsing and metadata fetching:
- Add magnet link: `./torrent-client "magnet:?xt=urn:btih:abcd1234..."`
- Verify metadata fetching from peers: watch for "Fetching metadata from peer" logs
- Check that download proceeds normally after metadata retrieval
- Expected: Complete download from magnet link without original torrent file

**Web API Checkpoint:**
After implementing HTTP API and WebSocket updates:
- Start web server: `./web-server --port 8080`
- Test REST endpoints: `curl http://localhost:8080/torrents`
- Connect WebSocket client and verify real-time updates
- Expected: JSON responses and live progress updates via WebSocket

#### Performance Testing and Optimization

Advanced features require performance validation to ensure they enhance rather than degrade the client experience:

| Feature | Performance Metric | Testing Method | Target |
|---------|-------------------|----------------|--------|
| DHT Integration | Query response time | Measure average get_peers latency | < 2 seconds |
| Magnet Links | Metadata fetch time | Time from magnet to first piece | < 30 seconds |
| API Throughput | Requests per second | HTTP load testing with concurrent clients | > 100 RPS |
| WebSocket Updates | Message latency | Measure event to client delivery time | < 100ms |

Use Go's built-in benchmarking for performance testing:

```go
func BenchmarkDHTQuery(b *testing.B) {
    client := setupTestDHTClient()
    infoHash := [20]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        peers, err := client.FindPeers(infoHash)
        if err != nil {
            b.Fatal(err)
        }
        if len(peers) == 0 {
            b.Error("No peers found")
        }
    }
}
```

The future extensions described in this section demonstrate how thoughtful architectural design enables sophisticated functionality through incremental enhancement. Each extension builds upon existing interfaces and patterns, proving that well-designed systems can evolve gracefully to meet new requirements without fundamental rewrites.


## Glossary

> **Milestone(s):** This glossary supports all milestones (1-4) by providing comprehensive definitions of BitTorrent-specific terms, protocols, and technical concepts used throughout the implementation.

This glossary serves as a comprehensive reference for all BitTorrent-specific terminology, protocol concepts, and technical terms used throughout the design document. Understanding these terms is essential for implementing a correct BitTorrent client, as the protocol has many domain-specific concepts that differ from general networking or file transfer protocols.

The terms are organized alphabetically within conceptual categories to help developers quickly locate definitions while working on specific components. Each definition includes not only the meaning but also the context in which the term is used and its relationship to other BitTorrent concepts.

### BitTorrent Protocol Core Concepts

**announce**: An HTTP request sent by a BitTorrent client to a tracker server to report the client's current status (uploaded, downloaded, left) and request a list of other peers sharing the same torrent. The announce request includes the info hash, peer ID, port number, and progress statistics. Announces are sent periodically at intervals specified by the tracker, typically every 30 minutes, and also when the client starts downloading, completes downloading, or stops.

**bencode**: A binary encoding format used exclusively by BitTorrent for serializing structured data in torrent files and some protocol messages. Bencode supports four data types: byte strings (length:content), integers (i<number>e), lists (l<items>e), and dictionaries (d<key><value>e). Unlike JSON, bencode preserves exact byte representation, which is crucial for calculating consistent info hashes across different implementations.

**bitfield**: A compact bitmap data structure where each bit represents whether a peer has a specific piece of the torrent. The bitfield is typically sent immediately after the handshake to inform the remote peer about piece availability. Bitfields allow efficient communication of piece availability using minimal bandwidth - for a torrent with 1000 pieces, the bitfield requires only 125 bytes compared to 4000 bytes if piece numbers were sent individually.

**block**: A 16KB (16,384 byte) subdivision of pieces used as the unit of network transfer between peers. While pieces are the unit of verification (each has a SHA-1 hash), blocks are the unit of network requests to optimize pipeline efficiency and reduce memory usage. A typical piece of 256KB contains 16 blocks. Peers request specific blocks within pieces rather than entire pieces to enable concurrent downloads from multiple peers.

**choking**: A flow control mechanism where a peer refuses to upload data to another peer, even if that peer is interested in downloading. Choking prevents peers from being overwhelmed by too many simultaneous upload requests and implements BitTorrent's reciprocal sharing incentive - peers typically unchoke others who are uploading to them. The choking state is communicated via choke/unchoke messages in the peer wire protocol.

**compact peer format**: A space-efficient binary encoding for peer information in tracker responses, where each peer is represented as 6 bytes: 4 bytes for IPv4 address and 2 bytes for port number in network byte order. This format reduces tracker response size compared to the dictionary format and is now the standard format used by most trackers and clients.

**endgame mode**: An aggressive downloading strategy activated when only a few pieces remain to complete a torrent. In endgame mode, the client requests all remaining blocks from all available peers simultaneously to avoid stalling on slow peers. Once a block is received from any peer, requests to other peers for that block are canceled. This prevents the common scenario where the last few percent of a download becomes very slow.

**handshake**: The initial 68-byte message exchanged between BitTorrent peers to establish a connection. The handshake contains the protocol string ("BitTorrent protocol"), info hash of the torrent, and the sender's peer ID. Both peers must send handshakes, and each must verify that the remote peer's info hash matches the expected torrent. The handshake serves as authentication that both peers are participating in the same torrent swarm.

**info hash**: A 20-byte SHA-1 cryptographic hash of the bencoded info dictionary from a torrent file. The info hash serves as the unique identifier for a torrent across the BitTorrent network - all peers sharing the same torrent will have identical info hashes. The info hash is used in tracker announces, peer handshakes, and DHT operations. Calculating the info hash correctly requires hashing the exact bencoded bytes of the info dictionary, not a re-encoded version.

**message framing**: The protocol mechanism for parsing variable-length messages from a TCP byte stream. BitTorrent messages use a length-prefix framing format: 4 bytes for message length in network byte order, followed by 1 byte for message type, followed by the message payload. Proper message framing prevents protocol desynchronization and enables reliable parsing of message boundaries in the TCP stream.

**peer**: An individual computer or BitTorrent client participating in the sharing of a specific torrent. Peers can be in various states: seeders (have complete file), leechers (downloading), or partial seeders (have some pieces). Each peer is identified by a unique peer ID and can simultaneously connect to multiple other peers to exchange pieces. The collection of all peers sharing a torrent is called a swarm.

**piece**: A fixed-size chunk of the torrent's data that serves as the unit of verification and availability tracking. Each piece has a SHA-1 hash specified in the torrent metadata, allowing peers to verify data integrity after downloading. Piece sizes are typically powers of 2 (64KB, 128KB, 256KB, 512KB) and are chosen to balance verification granularity with metadata overhead. Pieces are subdivided into 16KB blocks for network transfer.

**pipeline**: The practice of maintaining multiple outstanding block requests to a peer simultaneously, rather than waiting for each request to complete before sending the next. Pipelining dramatically improves download throughput by keeping the network connection saturated and reducing the impact of network round-trip time. Most clients maintain 5-10 outstanding requests per peer, though the optimal depth depends on network conditions.

**rarest-first**: A piece selection algorithm that prioritizes downloading pieces that are held by the fewest number of connected peers. This strategy increases the overall availability of pieces in the swarm by ensuring that rare pieces are duplicated quickly, reducing the risk of pieces becoming unavailable if peers leave. Rarest-first is crucial for swarm health but is typically disabled for the first few pieces to enable quick startup.

**seeding**: The process of uploading complete pieces to other peers after finishing the download. Seeders are essential for torrent health as they ensure availability of all pieces. The seeding process involves responding to piece requests from other peers by reading data from the completed file and sending it via piece messages. Many BitTorrent clients implement seeding ratios to encourage users to seed for extended periods.

**swarm**: The complete collection of peers participating in the sharing of a specific torrent, identified by the same info hash. A healthy swarm contains multiple seeders and maintains good piece availability across all participants. Swarm dynamics involve peers joining (downloading), completing (becoming seeders), and leaving, with the tracker coordinating peer discovery within the swarm.

**tracker**: A server that maintains lists of peers participating in torrents and coordinates peer discovery. Trackers do not store or transfer file data - they only facilitate peer connections by providing lists of active peers for each torrent. Clients communicate with trackers via HTTP announce requests at regular intervals to report status and discover new peers. Modern BitTorrent also supports trackerless operation using Distributed Hash Tables (DHT).

**torrent file**: A metadata file with a .torrent extension containing all information needed to download and verify a file or set of files. Torrent files are bencoded dictionaries containing the tracker URL, file information (names, sizes, directory structure), piece size, and SHA-1 hashes of all pieces. The torrent file serves as the authoritative specification for what constitutes the complete, correct version of the shared content.

### Protocol Message Types

**choke message**: A peer wire protocol message (ID 0) sent to inform a remote peer that no data will be uploaded to them, regardless of requests. Choking is BitTorrent's primary flow control and incentive mechanism. A peer typically chokes others when bandwidth is limited or when implementing tit-for-tat reciprocal sharing strategies. The choke state persists until an explicit unchoke message is sent.

**have message**: A peer wire protocol message (ID 4) announcing that the sender has successfully downloaded and verified a specific piece. Have messages enable peers to maintain accurate bitfields of what pieces their connected peers possess. These messages are broadcast to all connected peers immediately after successful piece verification, allowing the swarm to quickly learn about new piece availability.

**interested message**: A peer wire protocol message (ID 2) indicating that the sender wants to download data from the recipient. A peer becomes interested when it needs pieces that the remote peer has available. The interested state is a prerequisite for receiving data - choked but interested peers will be unchoked when bandwidth becomes available, while uninterested peers remain choked regardless of their upload contribution.

**piece message**: A peer wire protocol message (ID 7) containing actual file data in response to a request. Piece messages include the piece index, byte offset within the piece, and the block data. These are typically the largest messages in the protocol, carrying 16KB blocks of content. Piece messages are only sent to unchoked peers and represent the core data transfer mechanism of BitTorrent.

**request message**: A peer wire protocol message (ID 6) asking for a specific 16KB block within a piece. Requests specify the piece index, byte offset within the piece, and block length (usually 16KB). Peers can pipeline multiple requests for efficiency, but must not request from choked peers. Request messages drive the actual data transfer process in BitTorrent.

**unchoke message**: A peer wire protocol message (ID 1) indicating that the sender is willing to upload data to the recipient in response to requests. Unchoking typically occurs as part of reciprocal sharing algorithms - peers unchoke others who are uploading to them. The number of simultaneous unchokes is usually limited (typically 4-8) to prevent upload bandwidth from being spread too thin across many peers.

### Data Structures and Technical Terms

**AnnounceRequest**: A data structure containing all parameters needed for a tracker announce request, including info hash, peer ID, port, uploaded/downloaded/left byte counts, and event type. This structure is serialized into HTTP GET parameters when communicating with trackers. Proper construction of announce requests is critical for tracker communication and peer discovery.

**AnnounceResponse**: A data structure representing the tracker's response to an announce request, containing the announce interval, swarm statistics (complete/incomplete peer counts), and the compact peer list. Trackers may also return failure reasons for invalid requests. Parsing announce responses correctly is essential for extracting peer information and scheduling future announces.

**BitfieldOps**: A data structure and associated operations for managing piece availability bitmaps efficiently. Provides methods to set/clear/test individual bits, count available pieces, and serialize bitfields for transmission. Bitfield operations must handle bit-level manipulation correctly, including proper byte boundary handling for piece counts that aren't multiples of 8.

**Connection**: A data structure representing an active TCP connection to another BitTorrent peer, including socket, message framer, peer state flags (choking/interested), bitfield, and pending request queues. Connection objects manage the complete lifecycle of peer communication, from handshake through message exchange to cleanup. Proper connection state management is crucial for protocol correctness.

**Decoder**: A bencode decoder implementation that can parse the four bencode data types from a byte stream. The decoder uses recursive descent parsing to handle nested structures like lists and dictionaries. Decoder state includes current position in the stream and error handling for malformed bencode data. Correct bencode parsing is fundamental to torrent file processing.

**Info**: A data structure representing the info dictionary from a torrent file, containing file names, lengths, piece length, and concatenated piece hashes. The info dictionary describes the actual content being shared and is used to calculate the info hash that identifies the torrent. All torrent metadata validation and file construction depends on correct Info parsing.

**MetaInfo**: The complete data structure representing a parsed torrent file, including tracker URLs, creation metadata, and the info dictionary. MetaInfo serves as the authoritative source of torrent configuration throughout the client's operation. Proper MetaInfo construction from bencode data enables all subsequent BitTorrent operations.

**PieceState**: A data structure tracking the download progress and verification status of an individual piece, including block completion status, hash verification results, and piece priority. Piece state management coordinates concurrent downloads from multiple peers and ensures data integrity through hash verification.

### Networking and Concurrency Concepts

**circuit breaker**: A software pattern that prevents repeated failures by automatically failing fast when error rates exceed thresholds. In BitTorrent contexts, circuit breakers protect against misbehaving peers or trackers by temporarily avoiding connection attempts after repeated failures, then gradually allowing test connections to detect recovery.

**event bus**: An asynchronous messaging system enabling loose coupling between BitTorrent client components. Components publish events (peer connected, piece completed, tracker response) and subscribe to events they need to handle. Event buses simplify coordination between components like piece managers, peer managers, and user interfaces without tight coupling.

**exponential backoff**: A retry strategy where the delay between retry attempts increases exponentially after each failure, often with random jitter to prevent synchronized retries across multiple clients. BitTorrent clients use exponential backoff for tracker announces, peer connections, and piece requests to avoid overwhelming struggling resources.

**graceful shutdown**: A shutdown procedure that allows in-flight operations to complete before terminating the program. For BitTorrent clients, graceful shutdown includes completing piece verifications, sending final tracker announces, closing peer connections cleanly, and flushing any buffered file writes. Proper shutdown prevents data corruption and maintains good network citizenship.

**message framing**: The protocol mechanism for parsing variable-length messages from TCP byte streams using length prefixes. BitTorrent's framing uses 4-byte length headers followed by message payloads. Correct framing implementation prevents protocol desynchronization and enables reliable message parsing even with partial TCP reads.

**shared state**: Data structures accessible by multiple concurrent goroutines or threads, requiring synchronization mechanisms like mutexes to prevent race conditions. In BitTorrent clients, shared state includes piece availability maps, peer connection lists, and download statistics that multiple components access concurrently.

**state machine**: A computational model defining system behavior through states, transitions, and events. BitTorrent peer connections implement state machines tracking choking/unchoking and interested/uninterested states, with message-driven transitions between states. Proper state machine implementation ensures protocol correctness and prevents invalid state combinations.

### File and Data Management

**piece verification**: The process of computing SHA-1 hashes of downloaded pieces and comparing them against the expected hashes from the torrent metadata. Verification ensures data integrity and detects corruption or malicious peers. Failed verification requires re-downloading the piece from different peers. Verification typically occurs immediately after completing a piece download.

**rarest-first selection**: A piece selection algorithm that prioritizes pieces held by the fewest connected peers, increasing overall swarm health by duplicating rare pieces quickly. The algorithm maintains availability counts for all pieces across connected peers and selects pieces with minimum availability. This strategy prevents pieces from becoming unavailable when peers leave the swarm.

### Advanced BitTorrent Features

**Distributed Hash Table (DHT)**: A decentralized peer discovery mechanism that eliminates dependence on central tracker servers. DHT networks store peer information distributed across participating nodes, allowing torrent clients to find peers without tracker access. DHT implementation involves the Kademlia distributed hash table protocol with peer routing tables and distributed key-value storage.

**magnet link**: A URI scheme for identifying torrents using their info hash rather than torrent files. Magnet links enable instant sharing without distributing torrent files, with the complete metadata fetched from peers after connection. Magnet links typically include the info hash, display name, and tracker URLs, enabling clients to join swarms and download metadata on-demand.

**Message Stream Encryption (MSE)**: A protocol extension providing encryption for BitTorrent communications to prevent ISP throttling and deep packet inspection. MSE encrypts the peer wire protocol after handshake completion while maintaining protocol compatibility. Implementation involves cryptographic key exchange and stream cipher encryption of subsequent messages.

### Error Handling and Quality Assurance

**failure cascade**: A failure mode where one component failure triggers additional failures in dependent components, potentially leading to system-wide problems. BitTorrent clients must implement circuit breakers, timeouts, and isolation mechanisms to prevent tracker failures from affecting peer communications or piece verification failures from disrupting other downloads.

**peer quality scoring**: A system for evaluating and ranking peer reliability based on metrics like connection stability, data transfer rates, protocol compliance, and data integrity. Quality scores influence peer selection decisions, connection priorities, and choking algorithms to optimize overall download performance by preferring high-quality peers.

**protocol violation**: Deviation from the BitTorrent wire protocol specification that can cause interoperability problems or security vulnerabilities. Common violations include incorrect message formatting, invalid state transitions, or sending data to choked peers. Robust clients must detect and handle protocol violations gracefully while maintaining compatibility with compliant peers.

### Testing and Development

**integration testing**: Testing approach that verifies component interactions using controlled environments like mock trackers and test torrents with known content. Integration tests validate complete workflows like announce → peer discovery → handshake → piece download → verification, ensuring all components work together correctly.

**mock tracker**: A test implementation of the tracker protocol using local HTTP servers to provide predictable responses for testing. Mock trackers enable controlled testing of announce requests, error handling, peer list parsing, and retry logic without depending on external tracker infrastructure or network conditions.

**test torrent**: A specially crafted torrent file with known content, piece boundaries, and expected hashes for verification during testing. Test torrents enable automated verification of parsing, downloading, and verification logic by providing predictable inputs and expected outputs for test cases.

**unit testing**: Testing individual components in isolation without external dependencies like network connections or file systems. Unit tests verify specific functionality like bencode parsing, piece hash calculation, bitfield operations, and message serialization using controlled inputs and expected outputs.

### Performance and Optimization

**bandwidth management**: Techniques for controlling upload and download rates to optimize performance and comply with user-configured limits. Implementation involves token bucket algorithms, rate limiting, connection prioritization, and adaptive adjustment based on network conditions and peer performance.

**connection pooling**: Maintaining reusable network connections across multiple torrents or operations to reduce connection establishment overhead. Connection pools must handle connection lifetime management, error recovery, and resource cleanup while maintaining protocol compliance and security boundaries.

**token bucket algorithm**: A rate limiting algorithm that allows controlled bursts while maintaining average rate limits over time. Token buckets accumulate tokens at a fixed rate up to a maximum capacity, with each operation consuming tokens. This enables BitTorrent clients to achieve smooth bandwidth utilization while respecting rate limits.

### User Interface and API

**Single Page Application (SPA)**: A web application architecture where client-side JavaScript handles user interface updates and communicates with server APIs via AJAX. SPA BitTorrent interfaces provide responsive user experiences with real-time torrent status updates, file browsing, and download management without page refreshes.

**WebSocket**: A persistent bidirectional communication protocol enabling real-time updates between web browsers and servers. BitTorrent web interfaces use WebSocket connections to push live status updates, peer statistics, and download progress to user interfaces without polling overhead.

### Security and Network

**DHT bootstrap**: The process of joining a Distributed Hash Table network by connecting to well-known bootstrap nodes that provide initial routing table entries. Bootstrap nodes help new clients discover other DHT participants and begin participating in distributed peer discovery operations.

**metadata fetching**: The process of retrieving complete torrent metadata from peers using the Extension Protocol when only an info hash is available (such as from magnet links). Metadata fetching involves requesting metadata pieces from peers who have the complete torrent information.

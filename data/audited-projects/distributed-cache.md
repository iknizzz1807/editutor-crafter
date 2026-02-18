# AUDIT & FIX: distributed-cache

## CRITIQUE
- **Technical Inaccuracy (Confirmed - O(1) Lookup):** Consistent hash ring lookups are O(log N) using binary search on the sorted ring positions, not O(1). O(1) is only possible with a direct-mapping array of size equal to the hash space (e.g., 2^32), which wastes enormous memory. The AC must be corrected to O(log N).
- **Logical Gap (Confirmed - Thundering Herd):** When a popular cache key expires, hundreds of concurrent requests all miss the cache simultaneously and hit the backend database. This is the thundering herd / cache stampede problem. The project needs request collapsing (singleflight pattern) and/or probabilistic early expiration.
- **Missing: Cache-Aside vs Read-Through vs Write-Through:** The project doesn't define the caching strategy. Is the cache a standalone store? Does it integrate with a backend database? The cache invalidation pattern is completely undefined.
- **M1 Virtual Node Count:** 'Virtual nodes reduce hotspot risk' but doesn't specify how many virtual nodes per physical node. The standard recommendation is 100-300 virtual nodes per physical node for good distribution. This should be a configurable parameter with a documented default.
- **M2 Memory Accounting:** The AC says 'Memory limit enforced by evicting entries when total size exceeds threshold' but doesn't specify how size is measured. Is it key + value byte size? Does it include overhead (hash map entries, linked list pointers)? Accurate memory accounting is surprisingly hard.
- **M3 Gossip Protocol Complexity:** Implementing a full gossip protocol from scratch is a significant sub-project. The milestone underestimates the complexity — SWIM, epidemic broadcast, or even basic push/pull gossip requires careful design.
- **M4 Anti-Entropy Not Explained:** 'Anti-entropy process repairs inconsistent replicas in the background' is stated without any guidance on how (Merkle tree comparison, full key scan, etc.).
- **Missing: Eviction Under Replication:** If a node evicts a key due to memory pressure but the key is replicated on another node, what happens? Reads to the evicting node miss but the key exists elsewhere. The cache should either coordinate eviction across replicas or acknowledge inconsistent eviction.

## FIXED YAML
```yaml
id: distributed-cache
name: Distributed Cache
description: >-
  Build a distributed cache with consistent hashing, LRU eviction,
  cluster communication, replication with quorum, and thundering herd
  protection.
difficulty: advanced
estimated_hours: "30-45"
essence: >-
  Consistent hash ring with virtual nodes for O(log N) key-to-node routing,
  per-node LRU eviction with memory accounting, gossip-based cluster
  membership, configurable replication with quorum reads/writes, and
  singleflight request collapsing to prevent thundering herd on cache
  misses.
why_important: >-
  Building a distributed cache teaches the core challenges of distributed
  systems — data partitioning, replication, consistency vs availability
  trade-offs, and cache-specific challenges like thundering herd — which
  are fundamental to systems engineering at scale.
learning_outcomes:
  - Implement consistent hashing with virtual nodes for balanced O(log N) key routing
  - Design LRU eviction with accurate memory accounting
  - Build cluster membership using gossip protocol or static configuration
  - Implement replication with configurable quorum for consistency control
  - Prevent thundering herd using singleflight / request collapsing
  - Handle node failures with automatic re-routing and replication recovery
  - Debug CAP theorem trade-offs in practice
  - Build inter-node communication for cache routing and replication
skills:
  - Consistent Hashing with Virtual Nodes
  - LRU Cache with Memory Accounting
  - Distributed Systems Design
  - CAP Theorem Trade-offs
  - TCP/gRPC Networking
  - Replication Protocols
  - Thundering Herd Prevention
  - Gossip Protocol Basics
tags:
  - advanced
  - caching
  - distributed
  - eviction
  - go
  - java
  - python
  - replication
  - sharding
architecture_doc: architecture-docs/distributed-cache/index.md
languages:
  recommended:
    - Go
    - Java
    - Rust
  also_possible:
    - Python
    - JavaScript
resources:
  - type: article
    name: Consistent Hashing Explained
    url: https://www.toptal.com/big-data/consistent-hashing
  - type: paper
    name: "Amazon Dynamo Paper"
    url: https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf
  - type: article
    name: "Singleflight Pattern for Cache Stampede"
    url: https://pkg.go.dev/golang.org/x/sync/singleflight
prerequisites:
  - type: skill
    name: Hash tables and linked lists
  - type: skill
    name: TCP networking
  - type: skill
    name: Concurrency (mutex, channels/goroutines)
milestones:
  - id: distributed-cache-m1
    name: Consistent Hash Ring
    description: >-
      Implement consistent hashing with virtual nodes for distributing
      keys across cache nodes. Key lookup is O(log N) where N is the
      total number of virtual nodes on the ring (binary search on
      sorted ring positions).
    acceptance_criteria:
      - "Hash ring maps keys to nodes using a hash function (e.g., MD5, MurmurHash3) with the ring space [0, 2^32)"
      - "Each physical node is represented by V virtual nodes (configurable, default 150) distributed across the ring"
      - "lookup(key) returns the responsible node by finding the first virtual node position ≥ hash(key) via binary search — O(log N) complexity where N = total virtual nodes"
      - Distribution balance: with 3 nodes and 150 vnodes each, key distribution across nodes has < 15% deviation from uniform over 10,000 random keys
      - "add_node(node) adds the node's virtual nodes to the ring and remaps only the keys that now belong to the new node (minimal disruption)"
      - "remove_node(node) removes the node's virtual nodes and remaps affected keys to successor nodes"
      - Benchmark: lookup latency is < 10μs for a ring with 1000 virtual nodes
    pitfalls:
      - Claiming O(1) lookup: ring lookup is O(log N) via binary search, not O(1). Only a direct-mapping array (wasteful) achieves O(1).
      - Too few virtual nodes: with V=1, distribution is extremely uneven. 100-300 vnodes per physical node is the standard recommendation.
      - Poor hash function: using CRC32 or Java's hashCode() produces clustered ring positions. Use a well-distributed hash like MurmurHash3.
      - Not handling the ring wrap-around: if hash(key) > max virtual node position, the key wraps to the first virtual node in the ring.
    concepts:
      - Consistent hashing with hash ring
      - Virtual nodes for load balancing
      - O(log N) binary search on sorted ring
      - Minimal key remapping on membership changes
    skills:
      - Hash function selection and evaluation
      - Sorted data structure with binary search
      - Ring topology implementation
      - Distribution analysis and verification
    deliverables:
      - Hash ring with configurable virtual node count per physical node
      - lookup(key) returning responsible node in O(log N)
      - add_node / remove_node with minimal key remapping
      - Distribution balance test suite
    estimated_hours: "5-7"

  - id: distributed-cache-m2
    name: Cache Node with LRU Eviction and Thundering Herd Protection
    description: >-
      Implement a single cache node with GET, SET, DELETE operations,
      LRU eviction when memory limit is reached, TTL-based expiration,
      and singleflight request collapsing to prevent thundering herd
      on cache misses.
    acceptance_criteria:
      - "GET(key) returns the value if present and not expired, or MISS. Accessing a key promotes it to most-recently-used."
      - "SET(key, value, ttl) stores the entry. If memory limit is exceeded, the least-recently-used entry is evicted before insertion."
      - "DELETE(key) removes the entry immediately"
      - "LRU is O(1) for all operations using a hash map + doubly-linked list"
      - "Memory accounting tracks key_size + value_size + overhead (fixed 64 bytes per entry). Total tracked size does not exceed configured memory_limit."
      - TTL expiration: expired keys are lazily deleted on access AND periodically cleaned by a background sweeper (every 1s, sampling 20 random keys per sweep)
      - Singleflight / request collapsing: when multiple concurrent GET requests miss the same key simultaneously, only ONE fetch to the backend is performed and all waiters receive the result
      - "All operations are thread-safe under 100 concurrent goroutines/threads with no data races"
    pitfalls:
      - Memory accounting errors: not counting key size, or not subtracting evicted entry size, causes the cache to over- or under-fill
      - Thundering herd on popular key expiration: 1000 concurrent requests all miss and hit the backend simultaneously. Singleflight pattern is mandatory.
      - Eager TTL expiration scanning all keys: O(N) full scan is too expensive. Use random sampling (Redis-style) or lazy expiration.
      - Lock contention on the LRU list: every GET requires moving a node in the linked list. Consider sharded LRU (multiple independent LRUs by key hash).
    concepts:
      - LRU cache with O(1) operations
      - Memory accounting and limit enforcement
      - Singleflight / request collapsing pattern
      - Lazy + sampled TTL expiration (Redis-style)
    skills:
      - Doubly-linked list with hash map integration
      - Memory size tracking
      - Singleflight pattern implementation
      - Concurrent data structure design
    deliverables:
      - LRU cache with GET, SET, DELETE in O(1)
      - Memory limit enforcement with accurate size accounting
      - TTL support with lazy + sampled expiration
      - Singleflight deduplication for concurrent cache misses
    estimated_hours: "6-8"

  - id: distributed-cache-m3
    name: Cluster Communication and Request Routing
    description: >-
      Build inter-node communication so that any node can receive a
      request for any key and route it to the correct owner node based
      on the hash ring. Implement node discovery and health checking.
    acceptance_criteria:
      - "Each node exposes an inter-node RPC API (TCP/gRPC/HTTP) for GET, SET, DELETE operations from peer nodes"
      - "When a node receives a client request for a key it doesn't own, it forwards the request to the owning node and returns the response to the client (proxy mode)"
      - Node discovery: static configuration (list of node addresses) at startup. All nodes share the same consistent hash ring.
      - Health checks: each node pings all peers every 5s. Unresponsive nodes (3 consecutive failures) are marked down and removed from the local hash ring.
      - "When a node is removed from the ring, its key range is absorbed by the successor node (keys are lost unless replicated — M4)"
      - "When a node recovers and re-announces, it is re-added to the ring and starts accepting its key range"
      - "Request routing latency overhead (proxy hop) is < 5ms p99 on local network"
    pitfalls:
      - Split brain: node A thinks node B is down, node B thinks node A is down. Both remove each other from their rings, causing inconsistent routing.
      - Stale ring membership: a node's local ring view is out of date, routing requests to the wrong node or a dead node
      - Not handling the 'all nodes down' case: if a node's ring is empty, it should serve what it can from local cache or return 503
      - Full gossip protocol is complex: for a beginner implementation, static configuration with health checks is sufficient. Don't over-engineer.
    concepts:
      - Request routing via hash ring
      - Proxy-mode forwarding
      - Membership management (static or gossip)
      - Health checking and ring updates
    skills:
      - Inter-node RPC implementation
      - Proxy/forwarding logic
      - Membership tracking
      - Health check implementation
    deliverables:
      - Inter-node RPC for cache operations
      - Client request routing to owning node via hash ring
      - Static node discovery with health-check-based removal
      - Ring membership update on node failure/recovery
    estimated_hours: "8-10"

  - id: distributed-cache-m4
    name: Replication and Consistency
    description: >-
      Add data replication for fault tolerance. Each key is replicated
      to N successor nodes on the hash ring. Implement configurable
      read/write quorum for consistency control.
    acceptance_criteria:
      - Configurable replication factor R (default 3): each key is stored on the owning node and R-1 successor nodes on the ring
      - Write quorum W (configurable): a SET is acknowledged after W nodes confirm the write. W=1 is fast but risks data loss; W=R is safe but slow.
      - Read quorum Q (configurable): a GET reads from Q nodes and returns the most recent value. For strong consistency, W + Q > R.
      - Conflict resolution: if read quorum returns divergent values, the most recent value (by timestamp or version counter) is returned and read-repair is triggered
      - Read repair: after a read detects stale replicas, the latest value is written back to the stale nodes asynchronously
      - "When a node fails, reads/writes can still succeed as long as quorum is met with remaining replicas"
      - Integration test: with R=3, W=2, Q=2: kill one node, verify writes and reads still succeed. Kill two nodes, verify writes fail (quorum not met).
    pitfalls:
      - Stale reads with W=1, Q=1: a write goes to one node, a read goes to a different node that doesn't have the write yet. W + Q must > R for consistency.
      - Write conflicts during partition: two clients write the same key to different partition sides. Conflict resolution (LWW, version vector) determines the winner.
      - Eviction inconsistency across replicas: one replica evicts a key due to memory pressure but others still have it. Reads may return the key from non-evicting replicas.
      - Anti-entropy complexity: detecting which keys are inconsistent across replicas requires Merkle trees or full key comparison, which is expensive for large caches.
    concepts:
      - Replication factor and successor selection on hash ring
      - Read/write quorum (W + Q > R for strong consistency)
      - Read repair for eventual consistency
      - Conflict resolution strategies
    skills:
      - Quorum-based read/write implementation
      - Replica management on consistent hash ring
      - Read repair protocol
      - Conflict resolution (LWW or version counter)
    deliverables:
      - Configurable replication factor with successor-based replica placement
      - Write quorum enforcement across replicas
      - Read quorum with conflict detection and resolution
      - Read repair for stale replica synchronization
    estimated_hours: "10-14"
```
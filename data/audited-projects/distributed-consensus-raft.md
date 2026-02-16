# AUDIT & FIX: distributed-consensus-raft

## CRITIQUE
- **Missing linearizable reads**: The audit is absolutely correct. Without Read Index or lease-based reads, a client reading from the leader during a network partition may receive stale data. The leader may have been deposed but doesn't know it yet. This is a fundamental safety issue.
- **Missing Pre-Vote**: Without the Pre-Vote phase (Raft §9.6), a partitioned node increments its term while partitioned, then disrupts the healthy cluster when it rejoins by forcing a new election with its higher term. This is a well-known production issue.
- **Joint Consensus complexity**: The audit correctly notes that single-node-at-a-time membership change is simpler and often preferred. However, Joint Consensus is important to understand. The solution is to implement single-node changes first, then optionally add Joint Consensus.
- **Missing client interaction layer**: No milestone addresses how clients interact with the Raft cluster (e.g., redirect to leader, client session tracking for exactly-once semantics). Raft is useless without a client-facing API.
- **Missing state machine application**: Raft is a log replication protocol, but the milestones never require building an actual state machine (e.g., a key-value store) that applies committed log entries. Without this, there's nothing to test correctness against.
- **'Incremental snapshotting' is vague**: Does this mean copy-on-write, background serialization, or chunked transfer? The term needs definition.
- **Missing deterministic testing/simulation**: Raft implementations are notoriously difficult to test. No mention of deterministic simulation (like FoundationDB-style), Jepsen-like testing, or fault injection.
- **Persistent state requirements are mentioned but not detailed**: 'Persistent term and vote state to survive crashes' — what storage medium? What fsync requirements? What about WAL for log entries?
- **No mention of batching or pipelining for performance**: Real Raft implementations batch log entries and pipeline AppendEntries RPCs. Without this, throughput is limited to 1 entry per RTT.

## FIXED YAML
```yaml
id: distributed-consensus-raft
name: Titan Consensus Engine
description: "Industrial-grade Raft consensus implementation with leader election, log replication, snapshots, membership changes, linearizable reads, and a key-value state machine."
difficulty: expert
estimated_hours: "70-100"
essence: >
  A faithful implementation of the Raft consensus protocol: leader election with
  Pre-Vote, log replication with conflict resolution, snapshot-based log compaction,
  safe membership changes, linearizable reads, and a key-value state machine
  demonstrating the full lifecycle of replicated state.
architecture_doc: architecture-docs/distributed-consensus-raft/index.md
languages:
  recommended:
    - Rust
    - Go
    - Zig
  also_possible:
    - C++
    - Java
resources:
  - type: article
    name: "In Search of an Understandable Consensus Algorithm (Raft paper)"
    url: https://raft.github.io/raft.pdf
  - type: repository
    name: "etcd/raft implementation"
    url: https://github.com/etcd-io/raft
  - type: article
    name: "Raft Refloated: Do We Have Consensus?"
    url: https://www.cl.cam.ac.uk/~ms705/pub/papers/2015-osr-raft.pdf
  - type: tool
    name: "Jepsen distributed systems testing"
    url: https://jepsen.io/
prerequisites:
  - type: skill
    name: Networking (RPC, TCP sockets, or gRPC)
  - type: skill
    name: Persistent storage (file I/O, fsync semantics)
  - type: skill
    name: Concurrency (mutexes, channels, or async)
  - type: skill
    name: Distributed systems concepts (CAP, consensus basics)
skills:
  - Raft consensus protocol
  - Distributed state machine replication
  - Write-ahead logging
  - Snapshot and log compaction
  - Linearizable reads
  - Membership change protocols
  - Deterministic testing and fault injection
tags:
  - build-from-scratch
  - consensus
  - distributed-systems
  - expert
  - go
  - raft
  - replication
  - rust
milestones:
  - id: distributed-consensus-raft-m1
    name: "Leader Election with Pre-Vote"
    description: >
      Implement Raft leader election with randomized timeouts, persistent vote/term
      state, the Pre-Vote extension to prevent term inflation from partitioned nodes,
      and the RPC transport layer.
    acceptance_criteria:
      - "Randomized election timeouts (configurable range, e.g., 150-300ms) prevent split votes; in a 5-node cluster, a leader is elected within 2 election timeout periods after startup in >99% of runs (test over 100 runs)"
      - "Safety: at most one leader exists per term; this is verified by an invariant checker that runs during all tests and asserts no two nodes claim leadership for the same term"
      - "Persistent state: currentTerm, votedFor, and the log are persisted to durable storage (file with fsync) before any RPC response is sent; a node that crashes and restarts does not violate safety"
      - "Pre-Vote phase: a node that has been partitioned does not increment its term during the partition; upon rejoining, it sends Pre-Vote RPCs first; if it cannot win a pre-election (majority would not grant vote), it does not disrupt the existing leader"
      - "RequestVote RPC: correctly implements the Raft §5.2 and §5.4 rules (grant vote only if candidate's log is at least as up-to-date as voter's log)"
      - "AppendEntries RPC (heartbeat only in this milestone): the leader sends periodic heartbeats to prevent followers from starting elections; heartbeat interval is configurable and significantly less than election timeout"
      - "RPC transport layer: nodes communicate via a defined RPC mechanism (gRPC, custom TCP, or in-process channels for testing); transport is abstracted behind an interface to support both real network and deterministic testing"
    pitfalls:
      - "Election timeout too tight: if the election timeout range is too narrow, split votes occur frequently. Ensure the range is at least 2x the heartbeat interval."
      - "fsync omission: writing to a file without fsync means data may be in the OS page cache but not on disk. A power failure loses the data, violating Raft's safety requirement. Always fsync persistent state."
      - "Pre-Vote implementation subtlety: Pre-Vote does NOT increment the term. It's a read-only check. If you increment the term during Pre-Vote, you've defeated the purpose."
      - "RPC ordering assumptions: Raft does NOT require FIFO delivery. RPCs may arrive out of order or be duplicated. Every RPC handler must be idempotent with respect to stale messages."
    concepts:
      - Raft leader election (§5.2)
      - Pre-Vote extension (§9.6)
      - Persistent state requirements (§5.3)
      - RPC design for consensus
    skills:
      - RPC implementation (gRPC or custom)
      - Durable storage with fsync
      - Timer management for election/heartbeat
      - Invariant checking in tests
    deliverables:
      - Leader election with randomized timeouts
      - Pre-Vote phase preventing term inflation
      - Persistent storage for term, vote, and log
      - RequestVote and AppendEntries (heartbeat) RPC handlers
      - Transport abstraction supporting real and test networks
      - Election invariant test (one leader per term)
    estimated_hours: "14-20"

  - id: distributed-consensus-raft-m2
    name: "Log Replication & State Machine"
    description: >
      Implement log replication with conflict resolution, commit index advancement,
      and a key-value state machine that applies committed entries. Include
      batching for throughput.
    acceptance_criteria:
      - "Log matching property: if two logs contain an entry with the same index and term, then (a) the entries are identical and (b) all preceding entries are identical; verified by invariant checks"
      - "Leader appends client commands as new log entries and replicates via AppendEntries RPC to all followers; entries are committed once a majority of nodes have replicated them"
      - "Conflict resolution: when a follower's log conflicts with the leader's (different term at same index), the follower truncates its log from the conflicting entry onward and accepts the leader's entries (Raft §5.3)"
      - "Optimized conflict resolution: on AppendEntries rejection, the follower returns enough information (conflicting term and first index of that term) for the leader to skip back efficiently, not one entry at a time"
      - "State machine: a simple key-value store that applies committed log entries (PUT key value, GET key, DELETE key); GET returns the value as of the latest committed entry"
      - "Batching: the leader batches multiple client commands into a single AppendEntries RPC to amortize RPC overhead; throughput exceeds 1000 commands/second on a 3-node local cluster"
      - "Follower crash recovery: a follower that crashes, loses uncommitted entries, and restarts correctly catches up to the leader's log via AppendEntries RPCs without data loss"
      - "Network partition handling: during a minority partition, the minority cannot commit new entries; when the partition heals, the minority's divergent entries are replaced by the majority leader's entries"
    pitfalls:
      - "Committing entries from previous terms: a leader MUST NOT commit entries from previous terms by counting replicas alone. It must commit a current-term entry first, which implicitly commits all prior entries (Raft §5.4.2). Getting this wrong violates safety."
      - "Log truncation race: if a follower receives an AppendEntries that requires truncation while concurrently applying entries, the state machine may apply a truncated entry. Serialize log operations."
      - "State machine application must be idempotent or exactly-once: if a node crashes after applying but before advancing the lastApplied index, it re-applies the entry on restart. The state machine must handle this."
      - "Naive conflict resolution (back up one entry at a time) causes O(n) RPCs to catch up a follower that was partitioned for a long time. The optimized approach is essential for production."
    concepts:
      - Raft log replication (§5.3)
      - Log matching property
      - Commit rules and safety (§5.4)
      - State machine replication
    skills:
      - Log data structure with persistence
      - Conflict resolution algorithms
      - Key-value state machine implementation
      - Network partition simulation for testing
    deliverables:
      - Log replication via AppendEntries with conflict resolution
      - Optimized nextIndex backtracking
      - Commit index advancement with majority rule
      - Key-value state machine (PUT/GET/DELETE)
      - Batched AppendEntries for throughput
      - Partition and crash recovery tests
    estimated_hours: "16-22"

  - id: distributed-consensus-raft-m3
    name: "Log Compaction & Linearizable Reads"
    description: >
      Implement snapshot-based log compaction to bound log growth, InstallSnapshot
      RPC for slow followers, and linearizable reads to prevent stale data.
    acceptance_criteria:
      - "Snapshotting: when the log exceeds a configurable size (e.g., 10,000 entries), the state machine's current state is serialized to a snapshot file; all log entries up to the snapshot's lastIncludedIndex are discarded"
      - "Snapshot is taken in the background without blocking the main Raft loop; the state machine supports either copy-on-write or consistent-read isolation during snapshot serialization"
      - "InstallSnapshot RPC: when a follower is too far behind for log-based catch-up (required entries already compacted), the leader sends its snapshot via InstallSnapshot; the follower loads the snapshot and resumes normal replication from the snapshot's lastIncludedIndex"
      - "InstallSnapshot handles large snapshots by chunking: snapshots larger than a configured chunk size are sent in multiple RPC calls; the follower reassembles and verifies integrity (checksum)"
      - "Linearizable reads using Read Index: a read request causes the leader to (1) record the current commit index as the read index, (2) confirm it is still leader by exchanging a round of heartbeats with a majority, (3) wait for the state machine to apply up to the read index, then (4) serve the read from the state machine"
      - "Read-only queries do NOT append entries to the log, avoiding log bloat from read-heavy workloads"
      - "Stale read detection: if the leader cannot confirm its leadership (heartbeat timeout), the read returns an error rather than stale data"
    pitfalls:
      - "Snapshot during membership change: if a snapshot is taken while a membership change is in progress, the snapshot must include the current cluster configuration. Otherwise, a node loading the snapshot has the wrong membership."
      - "InstallSnapshot overwriting uncommitted entries: when a follower installs a snapshot, it must discard its entire log up to the snapshot index, including any uncommitted entries. This is correct but feels wrong."
      - "Read Index without heartbeat confirmation is NOT linearizable: a deposed leader that doesn't know it's been replaced will serve stale reads. The heartbeat round is not optional."
      - "Snapshot I/O blocking: writing a multi-GB snapshot synchronously on the main thread stalls the Raft loop, causing election timeouts. Background snapshotting is essential, not optional."
    concepts:
      - Log compaction via snapshots (§7)
      - InstallSnapshot RPC
      - Linearizable reads (§8, Read Index)
      - Background I/O
    skills:
      - State machine serialization
      - Chunked RPC transfer with integrity checks
      - Read Index protocol implementation
      - Background I/O with consistency guarantees
    deliverables:
      - Snapshot creation from state machine with background serialization
      - Log truncation after snapshot
      - InstallSnapshot RPC with chunked transfer
      - Linearizable Read Index implementation
      - Tests verifying no stale reads during leader changes
    estimated_hours: "16-22"

  - id: distributed-consensus-raft-m4
    name: "Membership Changes & Client Interaction"
    description: >
      Implement safe cluster membership changes (single-node-at-a-time, optionally
      Joint Consensus), client request routing, and client session tracking for
      exactly-once semantics.
    acceptance_criteria:
      - "Single-node membership change: nodes can be added or removed one at a time via a configuration change log entry; the Raft safety guarantee holds throughout the transition (verified by invariant tests)"
      - "New node catch-up: before a new node is added to the cluster configuration, it receives log entries as a non-voting learner until it is within a configurable number of entries behind the leader; only then is the configuration change committed"
      - "Disruptive server prevention: a node being removed from the cluster does not trigger elections after receiving the removal configuration entry; it steps down gracefully"
      - "Leader step-down: if the leader removes itself from the cluster, it commits the configuration change, then steps down, allowing a new leader to be elected from the remaining nodes"
      - "(Optional) Joint Consensus: implement the two-phase membership change protocol (C_old,new → C_new) for changing multiple nodes simultaneously; document the additional complexity and when it's necessary vs. single-node changes"
      - "Client request routing: clients that connect to a follower are redirected to the current leader (via leader hint in response); clients retry on a different node if the leader is unreachable"
      - "Client sessions: each client registers a session ID; the server tracks the last applied command per session, preventing duplicate application if a client retries a command that was already committed (exactly-once semantics)"
      - "The complete system (all milestones integrated) passes a chaos test: randomly killing nodes, partitioning the network, and submitting concurrent client requests for 10 minutes produces no safety violations (invariant checker) and all committed writes are eventually visible"
    pitfalls:
      - "Multiple simultaneous membership changes without Joint Consensus: changing two nodes at once with single-node changes can create two disjoint majorities. This is the fundamental reason membership changes are constrained to one-at-a-time."
      - "Adding a node before it's caught up: if a new node joins the quorum immediately, it's a slow node that can't keep up, causing the cluster to stall waiting for its acknowledgments."
      - "Client retry without session tracking: if a client retries a write that was already committed (but the response was lost), the write is applied twice. Session-based deduplication is required for exactly-once."
      - "Stale leader hint: a follower's knowledge of the leader may be out of date. The client must handle the case where the hinted leader is no longer leader."
    concepts:
      - Raft membership changes (§6)
      - Joint Consensus (§6, optional)
      - Client interaction protocol
      - Exactly-once semantics via sessions
    skills:
      - Membership change protocol implementation
      - Learner/non-voting node management
      - Client session tracking
      - Chaos testing and fault injection
    deliverables:
      - Single-node-at-a-time membership change
      - New node catch-up as non-voting learner
      - Leader step-down on self-removal
      - (Optional) Joint Consensus implementation
      - Client routing with leader hints
      - Client session tracking for exactly-once semantics
      - Chaos test harness with invariant checking
    estimated_hours: "18-26"
```
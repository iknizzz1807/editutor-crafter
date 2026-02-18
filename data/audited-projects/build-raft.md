# AUDIT & FIX: build-raft

## CRITIQUE
- **Missing Log Compaction / Snapshotting**: The audit correctly identifies this as a major gap. Without snapshots, the log grows indefinitely, making node restarts take O(entire_history) time and consuming unbounded disk space. The InstallSnapshot RPC is a core part of Raft described in Section 7 of the paper. This is not optional for a serious implementation.
- **Missing Persistence Requirements**: Raft's correctness requires that currentTerm, votedFor, and the log entries are persisted to stable storage before responding to any RPC. The current project has ZERO ACs requiring persistence. If a node crashes and restarts with a fresh state, it can vote for a different candidate in the same term—violating the election safety property. This is a critical correctness bug.
- **Milestone 3 (Safety Properties) is Not a Code Milestone**: The ACs for M3 describe invariants to verify, not code to write. 'Split vote scenario resolves correctly' and 'Newly elected leader contains all committed log entries' are properties that should be TRUE if M1 and M2 are implemented correctly. M3 should be a testing/verification milestone, not presented as a standalone implementation milestone.
- **Cluster Membership Changes is Extremely Advanced**: Joint consensus is Section 6 of the extended Raft paper and is notoriously difficult to implement correctly. Including it without log compaction (which is simpler and more practically necessary) is a questionable priority ordering.
- **Hour Estimates Have Absurd Ranges**: '15-25 hours' for log replication and '18-32 hours' for membership changes suggest the author doesn't know how long these take. The ranges are too wide to be useful.
- **No Client Interaction**: There's no milestone for building a client-facing interface. Without a way to submit commands and read state, the Raft implementation can't be tested as a complete system.
- **'Log up-to-date check' is Mentioned as a Pitfall but Its Implementation is an AC**: AC 4 of M1 requires the log up-to-date check, but it's also listed as a pitfall. This is inconsistent—it can't be both a required implementation and a common mistake.
- **Figure 8 Scenario Mentioned but Not Explained or Required**: M3 pitfall mentions 'Figure 8 scenario' without requiring the learner to implement the fix (leader commits from current term only). This is one of the subtlest parts of Raft and deserves a proper AC.

## FIXED YAML
```yaml
id: build-raft
name: Build Your Own Raft
description: Consensus algorithm with leader election, log replication, persistence, and snapshotting
difficulty: expert
estimated_hours: "80-120"
essence: >
  Leader-based replicated state machine using term-numbered log entries,
  randomized election timeouts, majority quorums, persistent state for
  crash recovery, and log compaction via snapshots to achieve linearizable
  consensus across crash-recovery failures and network partitions.
why_important: >
  Raft is the consensus algorithm behind etcd, CockroachDB, Consul, and
  many production distributed systems. Building it from scratch teaches
  you the deepest principles of distributed consensus, crash recovery,
  and the subtleties that make distributed systems correct.
learning_outcomes:
  - Implement leader election with randomized timeouts, term-based voting, and log up-to-date checks
  - Design log replication with AppendEntries RPC, commit index tracking, and consistency checks
  - Persist currentTerm, votedFor, and log to stable storage for crash recovery
  - Implement snapshot-based log compaction with InstallSnapshot RPC
  - Verify safety properties through invariant testing (election safety, leader completeness, log matching)
  - Build a client-facing interface for submitting commands and reading state
  - Debug network partition scenarios and split-brain prevention
  - Handle the Figure 8 scenario correctly (leader commits only from current term)
skills:
  - Distributed Consensus
  - Leader Election Algorithms
  - Log Replication
  - Persistent State and Crash Recovery
  - Log Compaction (Snapshotting)
  - RPC Communication
  - State Machine Replication
  - Invariant Testing
tags:
  - algorithms
  - build-from-scratch
  - consensus
  - distributed
  - expert
  - go
  - java
  - leader-election
  - log-replication
  - rust
  - state-machine
architecture_doc: architecture-docs/build-raft/index.md
languages:
  recommended:
    - Go
    - Rust
    - Java
  also_possible: []
resources:
  - type: paper
    name: "In Search of an Understandable Consensus Algorithm (Extended Version)"
    url: "https://raft.github.io/raft.pdf"
  - type: tool
    name: "Raft Visualization"
    url: "https://raft.github.io/"
  - type: tutorial
    name: "MIT 6.824 Raft Lab"
    url: "https://pdos.csail.mit.edu/6.824/labs/lab-raft.html"
  - type: paper
    name: "Raft Refloated: Do We Have Consensus?"
    url: "https://www.cl.cam.ac.uk/~ms705/pub/papers/2015-osr-raft.pdf"
prerequisites:
  - type: skill
    name: "Distributed systems basics (consensus, replication, failure models)"
  - type: skill
    name: "Concurrency (mutexes, goroutines/threads, timers)"
  - type: skill
    name: "Networking (RPC or gRPC)"
milestones:
  - id: build-raft-m1
    name: "Leader Election & Persistence"
    description: >
      Implement Raft leader election with terms, randomized election
      timeouts, RequestVote RPC, and the log up-to-date check. Persist
      currentTerm, votedFor, and log to stable storage so nodes survive
      crashes.
    acceptance_criteria:
      - "Each node stores and increments a monotonically increasing currentTerm; all RPCs include the sender's term; a node receiving an RPC with a higher term steps down to follower and updates its term"
      - "Election timeout is randomized within a configurable range (e.g., 150ms-300ms); when the timeout fires without receiving a heartbeat, the node transitions from follower to candidate"
      - "Candidate increments its term, votes for itself, resets its election timer, and sends RequestVote RPCs to all peers in parallel"
      - RequestVote RPC includes candidateId, term, lastLogIndex, and lastLogTerm; receiver grants vote only if: (a) it has not voted in this term or already voted for this candidate, AND (b) the candidate's log is at least as up-to-date (compared by lastLogTerm first, then lastLogIndex)
      - "A candidate that receives votes from a majority of nodes (including itself) transitions to leader and immediately sends empty AppendEntries heartbeats to all peers"
      - "A candidate that receives an AppendEntries RPC from a valid leader (same or higher term) steps down to follower"
      - Persistence: currentTerm, votedFor, and log entries are written to stable storage (disk) BEFORE responding to any RPC; on restart, these values are restored from disk
      - Persistence test: node votes YES in term 5, is killed, restarts; verify it still knows it voted in term 5 and does not vote for a different candidate in term 5
      - Election safety test: in a 5-node cluster, partition into [2,3] groups; verify each partition elects at most one leader; heal partition, verify single leader emerges
      - Split vote test: force 2 candidates to start simultaneously; verify the system resolves within a bounded number of election rounds (due to randomized timeouts)
    pitfalls:
      - "Not persisting votedFor allows a crashed-and-restarted node to vote twice in the same term, violating election safety (two leaders in one term)"
      - "Not persisting currentTerm allows a restarted node to accept stale RPCs from old terms"
      - "Election timeout range too narrow increases split vote probability; too wide increases leader election latency after failure"
      - The log up-to-date check (comparing lastLogTerm, then lastLogIndex) is subtle: a shorter log with a higher last term is more up-to-date than a longer log with a lower last term
      - "Granting votes without checking the log up-to-date property allows election of a leader that is missing committed entries, violating leader completeness"
    concepts:
      - Term-based epoch numbering
      - Randomized election timeouts to break symmetry
      - Log up-to-date comparison for vote granting
      - Persistent state for crash-recovery correctness
      - Majority quorum for leader election
    deliverables:
      - "Node state machine with follower, candidate, and leader states and transitions"
      - "RequestVote RPC with log up-to-date check"
      - "Election timeout with randomization and heartbeat reset"
      - "Persistent state storage (currentTerm, votedFor, log) with fsync"
      - "Persistence recovery on restart"
      - "Election safety test and split vote resolution test"
    estimated_hours: "15-20"

  - id: build-raft-m2
    name: "Log Replication"
    description: >
      Implement log replication from leader to followers using AppendEntries
      RPC. Track commit index and apply committed entries to the state
      machine. Handle the Figure 8 scenario correctly.
    acceptance_criteria:
      - "Leader maintains nextIndex and matchIndex for each follower; nextIndex initialized to leader's last log index + 1; matchIndex initialized to 0"
      - "AppendEntries RPC includes term, leaderId, prevLogIndex, prevLogTerm, entries[], and leaderCommit"
      - "Follower rejects AppendEntries if its log does not contain an entry at prevLogIndex with term == prevLogTerm; leader decrements nextIndex and retries (log backtracking)"
      - On successful AppendEntries response, leader updates matchIndex for that follower and advances commitIndex to the highest index N where: (a) N > commitIndex, (b) a majority of matchIndex[i] >= N, AND (c) log[N].term == currentTerm (Figure 8 safety)
      - Condition (c) is critical: the leader ONLY commits entries from its own current term; entries from previous terms are committed indirectly when a current-term entry is committed after them
      - "Committed entries are applied to the state machine in log order (index 1, 2, 3, ...); lastApplied tracks the highest applied index; no entry is applied twice"
      - Heartbeat: leader sends AppendEntries with empty entries[] at a configurable interval (e.g., 50ms) to maintain authority and prevent elections
      - "Log entries and commit index updates are persisted to stable storage before acknowledging"
      - Replication test: submit 100 commands to the leader; verify all commands are replicated to all followers and applied to all state machines in the same order
      - Follower restart test: kill a follower, submit 10 commands, restart follower; verify it catches up via AppendEntries and applies all 10 commands
      - Leader failure test: kill the leader after 50 commands; verify a new leader is elected and the remaining 50 commands submitted to the new leader are replicated correctly; all 100 commands are applied in the same order on all nodes
    pitfalls:
      - Off-by-one errors in log indexing: Raft log is 1-indexed in the paper; many languages use 0-indexed arrays. Pick a convention and be consistent.
      - "Committing entries from a previous term directly (without the Figure 8 check) allows committed entries to be overwritten, violating safety. Always check log[N].term == currentTerm."
      - "Not persisting log entries before responding to AppendEntries means a crash after responding loses entries the leader thinks are replicated"
      - "Log backtracking by decrementing nextIndex one-at-a-time is correct but slow for long divergent logs; optimize with ConflictTerm/ConflictIndex hints"
      - "Applying entries out of order or skipping entries causes state machine divergence; always apply sequentially from lastApplied + 1"
    concepts:
      - AppendEntries RPC with log consistency check
      - nextIndex/matchIndex tracking per follower
      - Commit index advancement with majority and current-term check
      - Figure 8 scenario and its resolution
      - Log backtracking on consistency failure
    deliverables:
      - "AppendEntries RPC with prevLogIndex/prevLogTerm consistency check"
      - "Leader replication loop with nextIndex/matchIndex tracking"
      - "Commit index advancement with majority check AND current-term-only rule"
      - "State machine application loop applying committed entries in order"
      - "Heartbeat mechanism preventing unnecessary elections"
      - "Log backtracking on follower consistency check failure"
      - "Replication, follower-restart, and leader-failure tests"
    estimated_hours: "20-30"

  - id: build-raft-m3
    name: "Log Compaction (Snapshotting)"
    description: >
      Implement snapshot-based log compaction. Nodes periodically snapshot
      their state machine and discard log entries up to the snapshot index.
      Implement InstallSnapshot RPC for sending snapshots to slow followers.
    acceptance_criteria:
      - "When the log exceeds a configurable size threshold (e.g., 10000 entries), the node takes a snapshot of the current state machine state, recording lastIncludedIndex and lastIncludedTerm"
      - "After snapshotting, log entries up to lastIncludedIndex are discarded; subsequent log indexing accounts for the snapshot offset"
      - "Snapshot is persisted to stable storage; on restart, the node restores the state machine from the latest snapshot and replays log entries after lastIncludedIndex"
      - InstallSnapshot RPC: when the leader's nextIndex for a follower points to a discarded entry (before the leader's snapshot), the leader sends its snapshot instead of AppendEntries
      - Follower receiving InstallSnapshot: if the snapshot covers entries the follower doesn't have, the follower discards its entire log, installs the snapshot as its state, and updates lastIncludedIndex/lastIncludedTerm
      - "If the follower's log extends beyond the snapshot's lastIncludedIndex, it retains the suffix and only discards entries covered by the snapshot"
      - Snapshot test: submit 15000 entries (threshold=10000); verify snapshot is taken; kill and restart a node; verify it recovers from snapshot + remaining log entries
      - Slow follower test: partition a follower, submit enough entries to trigger snapshot on leader, heal partition; verify leader sends InstallSnapshot and follower catches up correctly
    pitfalls:
      - "After log compaction, log index 1 no longer exists in the array; all log indexing must account for the snapshot offset (lastIncludedIndex). This is a pervasive change that touches many functions."
      - "Snapshotting and log truncation must be atomic; truncating the log before the snapshot is persisted risks losing committed state on crash"
      - "InstallSnapshot must handle the case where the follower has already applied some entries that are in the snapshot; avoid re-applying them to the state machine"
      - "Large snapshots can take significant time to transfer; consider chunked transfer for InstallSnapshot (Raft paper Section 7 mentions this)"
    concepts:
      - Snapshot-based log compaction
      - InstallSnapshot RPC
      - Log indexing with snapshot offset
      - State machine recovery from snapshot + log suffix
    deliverables:
      - "Snapshot trigger based on configurable log size threshold"
      - "Snapshot creation serializing state machine state with lastIncludedIndex/Term"
      - "Log truncation discarding entries covered by snapshot"
      - "InstallSnapshot RPC implementation (leader sending, follower receiving)"
      - "Recovery from snapshot on restart"
      - "Snapshot and slow-follower catch-up tests"
    estimated_hours: "15-20"

  - id: build-raft-m4
    name: "Client Interface & Linearizability"
    description: >
      Build a client-facing interface for submitting commands and reading
      state. Implement mechanisms for linearizable reads and duplicate
      command detection.
    acceptance_criteria:
      - "Client submits commands to the leader via RPC; if the node is not the leader, it returns the leader's address (or an error) so the client can redirect"
      - "Command submission returns success only after the command is committed (replicated to majority and applied to state machine); clients retry on timeout"
      - Duplicate command detection: each client has a unique client ID; each command has a monotonically increasing sequence number; the state machine deduplicates commands by (clientId, sequenceNumber) to prevent re-execution on retry
      - Deduplication test: client submits command with seq=5; network drops the response; client retries seq=5; verify the command is applied exactly once
      - Linearizable reads: read-only queries go through the leader; leader confirms it is still the leader by committing a no-op entry (or using ReadIndex optimization) before responding; this prevents stale reads from a deposed leader
      - Stale read test: partition the leader from the majority; the old leader should NOT serve reads after losing leadership; the new leader should serve reads correctly
      - Key-value store state machine: implement a simple key-value store (get, put, delete) as the application state machine to demonstrate the Raft library
    pitfalls:
      - "Without duplicate detection, a client retrying a timed-out command causes the command to be applied twice; this breaks any non-idempotent operation"
      - "A leader that has been partitioned from the majority may still believe it is the leader and serve stale reads; the ReadIndex or heartbeat-check mechanism prevents this"
      - "Client session state (for deduplication) must be part of the replicated state machine state; otherwise, a leader change loses deduplication information"
      - "Not redirecting clients to the current leader causes clients to time out repeatedly against followers"
    concepts:
      - Client command submission and response
      - Duplicate detection with client sessions
      - Linearizable reads via ReadIndex or no-op commit
      - State machine application interface
    deliverables:
      - "Client RPC interface for command submission with leader redirect"
      - "Duplicate detection using (clientId, sequenceNumber) tracked in state machine"
      - "Linearizable read implementation (ReadIndex or no-op commit)"
      - "Key-value store state machine (get, put, delete)"
      - "Deduplication test and stale read prevention test"
    estimated_hours: "12-18"

  - id: build-raft-m5
    name: "Safety Verification & Stress Testing"
    description: >
      Verify Raft safety properties through comprehensive testing including
      network partitions, concurrent failures, and long-running stress
      tests. This milestone validates the correctness of all previous
      milestones.
    acceptance_criteria:
      - Election safety: over 1000 randomized test runs with various partition scenarios, verify no term ever has two leaders simultaneously
      - Leader completeness: after every leader election, verify the new leader's log contains all entries that were committed by any previous leader
      - Log matching: at any point during any test, verify that if two nodes have a log entry with the same index and term, all preceding entries are identical
      - State machine safety: at the end of every test, verify all non-failed nodes have applied exactly the same sequence of commands in the same order
      - Partition test suite: (a) leader isolated from majority, (b) cluster split into [2,3], (c) cascading partitions, (d) partition healed after new leader elected; verify safety in all cases
      - Concurrent failure test: kill 2 nodes in a 5-node cluster, submit commands, restart nodes, verify all commands are committed and applied correctly
      - Stress test: submit 10000 commands with random node kills and network partitions over 10 minutes; verify all committed commands are applied consistently across all surviving nodes
      - No-op on leader election: verify new leader commits a no-op entry from its own term before serving client commands (ensures previous-term entries are committed)
    pitfalls:
      - "Deterministic tests with fixed seeds miss rare race conditions; combine with randomized soak tests that run for extended periods"
      - "Testing only happy paths misses the subtle edge cases (Figure 8, stale leader reads, duplicate commands during partition) that cause real-world failures"
      - "Not testing with realistic network latency and message reordering produces misleadingly optimistic results"
      - "Safety property checks must run as invariant assertions throughout the test, not just at the end; a transient violation that self-heals is still a bug"
    concepts:
      - Raft safety invariants (election safety, leader completeness, log matching, state machine safety)
      - Chaos testing with partitions and crashes
      - Invariant checking during test execution
      - Long-running stress tests for distributed systems
    deliverables:
      - "Invariant checker verifying all four Raft safety properties continuously during tests"
      - "Partition test suite covering leader isolation, split, cascade, and heal scenarios"
      - "Concurrent failure test with node kills and restarts"
      - "Long-running stress test with random faults and safety verification"
      - "Test report summarizing safety property violations (should be zero) and performance metrics"
    estimated_hours: "15-20"
```
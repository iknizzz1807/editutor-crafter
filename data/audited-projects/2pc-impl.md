# AUDIT & FIX: 2pc-impl

## CRITIQUE
- **Logical Ordering Problem**: Milestone 1 demands 'Coordinator process survives kill-9 and recovers correct transaction state from the log on restart' and 'Recovery procedure replays the log and resumes any in-progress two-phase commit protocol rounds'—but the protocol phases haven't been implemented yet. You can't test recovery of a protocol that doesn't exist. This is pedagogically backwards.
- **Missing Participant Uncertainty State**: The audit correctly identifies that there's no AC covering the critical 'in-doubt' period where a participant has voted YES but hasn't received the coordinator's decision. This is THE central problem of 2PC—the participant is blocked and cannot unilaterally decide. The project barely addresses this.
- **'Two-Phase Locking' Listed as a Skill but Never Implemented**: 2PL is a concurrency control mechanism, not part of the 2PC commit protocol. Listing it as a skill is misleading. The project never asks the learner to implement isolation levels or lock managers.
- **Milestone 3 'Atomic broadcast' is Incorrect Terminology**: 2PC does not use atomic broadcast (which is a stronger primitive equivalent to consensus). The coordinator sends point-to-point messages to each participant. Calling this 'atomic broadcast' is technically inaccurate.
- **Presumed Abort Optimization in M4 is Premature**: The 'presumed abort' optimization is mentioned as a deliverable but never has an AC. It's a subtle optimization that reduces log writes by not logging aborts—implementing it without rigorous ACs risks silent correctness bugs.
- **No Network Simulation or Fault Injection Framework**: The project mentions testing 'network partitions' but provides no milestone for building or using a test harness that can simulate these failures. Without this, the learner cannot verify recovery correctness.
- **Idempotent Operations Mentioned in Learning Outcomes but Never Tested**: 'Design idempotent commit/abort operations for retry safety' is a learning outcome but no AC verifies that repeated delivery of COMMIT or ABORT produces the same result.
- **Transaction Isolation Skill Listed but Irrelevant**: 2PC is about atomic commitment, not isolation. There's no read/write conflict handling in this project.

## FIXED YAML
```yaml
id: 2pc-impl
name: Two-Phase Commit
description: Distributed atomic commitment protocol
difficulty: advanced
estimated_hours: "20-30"
essence: >
  Distributed atomic commitment protocol coordinating multiple participants
  through synchronized prepare and commit phases with durable write-ahead
  logging, blocking coordinator-based consensus, and failure recovery
  mechanisms to achieve all-or-nothing transaction semantics across unreliable
  networks.
why_important: >
  Building 2PC teaches the foundational atomic commitment mechanism underlying
  distributed databases and microservices, giving practical experience with
  coordinator-based protocols, the blocking problem, write-ahead logging, and
  failure recovery trade-offs that appear in every distributed system.
learning_outcomes:
  - Implement durable write-ahead logging with fsync guarantees for transaction recovery
  - Design coordinator and participant state machines with explicit state transitions
  - Build the prepare phase with participant voting, lock acquisition, and timeout handling
  - Build the commit/abort phase with durable decision logging before notification
  - Handle the participant uncertainty (in-doubt) state when coordinator is unreachable
  - Implement coordinator crash recovery that re-drives in-progress transactions
  - Implement participant crash recovery that consults logs and queries coordinator
  - Design idempotent commit/abort operations verified through duplicate delivery tests
  - Implement presumed abort optimization to reduce log write overhead
skills:
  - Atomic Commitment Protocols
  - Write-Ahead Logging with Fsync
  - Coordinator/Participant State Machines
  - Crash Recovery Algorithms
  - Distributed Failure Handling
  - Blocking Protocol Analysis
  - Idempotent Operation Design
  - RPC Communication and Timeout Management
tags:
  - advanced
  - atomic-commit
  - coordinator
  - distributed
  - go
  - java
  - python
  - recovery
  - wal
architecture_doc: architecture-docs/2pc-impl/index.md
languages:
  recommended:
    - Go
    - Java
    - Python
  also_possible:
    - Rust
    - Erlang
resources:
  - name: "Bernstein Chapter 7: Distributed Commit"
    url: "https://www.cs.princeton.edu/courses/archive/fall16/cos418/papers/bernstein-ch7.pdf"
    type: paper
  - name: "Consensus Protocols: Two-Phase Commit"
    url: "https://www.the-paper-trail.org/post/2014-10-16-consensus-protocols-two-phase-commit/"
    type: article
  - name: "Mohan et al. - Presumed Abort and Presumed Commit"
    url: "https://dl.acm.org/doi/10.1145/319996.320000"
    type: paper
prerequisites:
  - type: skill
    name: "Distributed systems concepts (consensus, failure models)"
  - type: skill
    name: "Transaction concepts (ACID, atomicity)"
  - type: skill
    name: "File I/O and durability (fsync, write-ahead logging concepts)"
milestones:
  - id: 2pc-impl-m1
    name: "Write-Ahead Log & State Machines"
    description: >
      Implement a durable write-ahead log and define the coordinator and
      participant state machines with explicit transitions. This milestone
      focuses on the foundation—persistence and state model—without yet
      implementing the protocol communication.
    acceptance_criteria:
      - "WAL appends log records containing (tx_id, state, participant_list, timestamp) and calls fsync before returning success"
      - Log records support states: PREPARE_SENT, VOTE_YES, VOTE_NO, COMMIT, ABORT, ACK for both coordinator and participant roles
      - Coordinator state machine defines transitions: INIT -> PREPARE_SENT -> COMMITTED | ABORTED -> DONE, with each transition logged before acting
      - Participant state machine defines transitions: INIT -> PREPARED (voted YES) | ABORTED (voted NO) -> COMMITTED | ABORTED, with each transition logged before responding
      - "Log replay function reads all records and reconstructs the last known state for every transaction, returning a map of tx_id -> state"
      - Crash simulation test: write 5 log records, kill the process after record 3, restart, verify only records 1-3 are recovered (no partial writes)
      - "Log truncation removes records for completed transactions (DONE state) while preserving records for in-progress transactions"
    pitfalls:
      - "Calling fsync on the file descriptor is not enough if the filesystem uses write barriers; on Linux, use fdatasync or O_DSYNC for guaranteed durability"
      - "Partial writes after crash produce corrupted records; use length-prefixed records with CRC checksums to detect and skip incomplete entries during replay"
      - "Truncating the log before all participants have acknowledged completion loses the ability to answer termination queries from recovered participants"
      - "Defining state machines informally leads to impossible transitions; draw the explicit FSM diagram and assert valid transitions in code"
    concepts:
      - Write-ahead logging with fsync durability guarantees
      - CRC-protected length-prefixed log record format
      - Coordinator and participant finite state machines
      - Log replay for crash recovery
    deliverables:
      - "WAL implementation with append, fsync, read-all, and truncate operations"
      - "Log record format with CRC, tx_id, role (coordinator/participant), state, and participant list"
      - "Coordinator state machine with validated transitions and logging at each step"
      - "Participant state machine with validated transitions and logging at each step"
      - "Log replay function reconstructing transaction states after restart"
      - "Crash simulation unit test verifying durability and partial-write detection"
    estimated_hours: "4-5"

  - id: 2pc-impl-m2
    name: "Prepare Phase (Voting)"
    description: >
      Implement the first phase of 2PC: coordinator sends PREPARE to all
      participants, each participant acquires locks and votes YES or NO,
      and the coordinator collects votes with timeout handling.
    acceptance_criteria:
      - "Coordinator logs PREPARE_SENT to WAL, then sends PREPARE RPC to every participant listed in the transaction"
      - "Participant receiving PREPARE acquires all required resource locks; if successful, logs VOTE_YES to its local WAL, then responds YES to coordinator"
      - "Participant that cannot acquire locks (conflict, resource unavailable) logs VOTE_NO and responds NO to coordinator; no locks are held after voting NO"
      - "Participant persists its vote to WAL BEFORE sending the vote response to the coordinator (force-write rule)"
      - "Coordinator aborts the transaction if any participant's vote is not received within a configurable timeout (e.g., 5 seconds)"
      - "After voting YES, participant enters the PREPARED (in-doubt/uncertain) state and MUST NOT unilaterally abort or commit until it receives the coordinator's decision"
      - Test: with 3 participants, participant 2 votes NO; verify coordinator receives the NO vote and proceeds to abort in the commit phase
    pitfalls:
      - "If the participant sends its vote before logging it, a crash after sending means the vote is lost and the coordinator may have already counted it—violating the protocol's safety guarantee"
      - "Setting the vote timeout too low causes spurious aborts under normal network latency; measure baseline RTT and set timeout to at least 3x p99 RTT"
      - "After voting YES, the participant is in the uncertainty window—it cannot time out and unilaterally abort because the coordinator may have already decided COMMIT based on this vote"
      - "Locks acquired during prepare must be held until the final decision (commit or abort) is received; releasing them prematurely violates atomicity"
    concepts:
      - Prepare/voting phase of 2PC
      - Force-write rule (log before send)
      - Participant uncertainty (in-doubt) state
      - Lock acquisition and holding during protocol
    deliverables:
      - "Coordinator PREPARE broadcast sending PREPARE RPC to all participants"
      - Participant vote logic: lock acquisition, WAL write, vote response
      - "Vote collection with configurable timeout and early abort on any NO"
      - "Participant PREPARED state tracking with explicit uncertainty window documentation"
      - "Integration test with 3 participants covering all-YES and mixed-vote scenarios"
    estimated_hours: "4-6"

  - id: 2pc-impl-m3
    name: "Commit/Abort Phase (Decision)"
    description: >
      Implement the second phase: coordinator makes and logs the global
      decision, notifies all participants, collects acknowledgments, and
      participants apply or rollback changes. Ensure idempotent decision
      delivery.
    acceptance_criteria:
      - "If all participants voted YES, coordinator logs COMMIT to WAL; if any voted NO or timed out, coordinator logs ABORT to WAL"
      - "Coordinator logs the decision to WAL BEFORE sending the decision to any participant (this is the commit point—the point of no return)"
      - "Coordinator sends COMMIT or ABORT to each participant and waits for ACK from each"
      - "Participant receiving COMMIT applies changes, releases locks, logs COMMITTED to WAL, and responds ACK"
      - "Participant receiving ABORT rolls back changes, releases locks, logs ABORTED to WAL, and responds ACK"
      - "Coordinator retries decision delivery to participants that do not ACK within timeout, using exponential backoff"
      - Decision delivery is idempotent: if a participant receives a duplicate COMMIT or ABORT, it responds ACK without re-applying changes; verified by test sending COMMIT twice
      - "Transaction completes (coordinator logs DONE) only after all participants have acknowledged"
    pitfalls:
      - "Logging the decision AFTER sending it to participants means a coordinator crash after sending but before logging could lose the decision, creating an inconsistency if some participants committed and the recovered coordinator defaults to abort"
      - "Not retrying decision delivery to non-responding participants means those participants remain blocked in the uncertainty state indefinitely"
      - "Applying changes without idempotency means duplicate COMMIT messages (from retries) cause double application of the transaction"
      - "A participant that crashes after applying COMMIT but before sending ACK will cause the coordinator to keep retrying; the participant must check its WAL on restart and re-send ACK"
    concepts:
      - Commit point (decision record durably logged)
      - Idempotent decision application
      - Reliable delivery with retry and acknowledgment
      - Lock release after decision
    deliverables:
      - Decision logic: COMMIT if all YES, ABORT otherwise; logged to WAL before notification
      - "Decision notification RPC sent to each participant with retry on timeout"
      - "Participant apply/rollback logic triggered by decision, with lock release"
      - "Acknowledgment collection with retry for non-responding participants"
      - Idempotency test: duplicate COMMIT delivery produces single application and ACK
      - Transaction completion: coordinator logs DONE after all ACKs received
    estimated_hours: "4-6"

  - id: 2pc-impl-m4
    name: "Failure Recovery & Blocking Analysis"
    description: >
      Handle coordinator and participant crash recovery using WAL replay.
      Implement the participant termination protocol for querying outcomes
      during the uncertainty window. Analyze and demonstrate the blocking
      nature of 2PC. Implement presumed abort optimization.
    acceptance_criteria:
      - Coordinator crash recovery: on restart, replay WAL; for transactions in PREPARE_SENT state, re-collect votes or abort; for transactions in COMMIT/ABORT state, re-send decision to unacknowledged participants
      - Participant crash recovery: on restart, replay WAL; for transactions in PREPARED state (voted YES, no decision), enter uncertainty state and query coordinator for outcome
      - Participant termination protocol: participant in uncertainty state sends QUERY(tx_id) to coordinator; coordinator responds with COMMIT, ABORT, or UNKNOWN
      - "If coordinator responds UNKNOWN (no decision logged), the participant must remain blocked until the coordinator makes a decision—this is the fundamental blocking property of 2PC"
      - Blocking demonstration test: crash the coordinator after receiving all YES votes but before logging the decision; verify all participants are stuck in uncertainty state and cannot progress
      - Presumed abort optimization: coordinator does not force-write ABORT decisions (only COMMIT); on recovery, any transaction without a COMMIT record is presumed aborted, reducing log writes for abort-heavy workloads
      - Presumed abort correctness test: abort a transaction without logging ABORT, crash coordinator, recover, verify the transaction is correctly presumed aborted and participants are notified
      - End-to-end recovery test: run 10 transactions, crash coordinator at random points, restart, verify all transactions either fully committed or fully aborted across all participants
    pitfalls:
      - "The blocking problem is inherent to 2PC and cannot be fully solved without moving to 3PC or Paxos Commit; the learner must understand this is a fundamental limitation, not a bug"
      - Presumed abort is unsafe if participants do not also follow the protocol: a participant that voted YES and crashes must query the coordinator on recovery, not assume abort
      - "Coordinator single point of failure means the entire protocol halts if the coordinator is permanently lost; document this limitation explicitly"
      - "Network partitions between coordinator and participants look identical to coordinator crashes from the participant's perspective; the termination protocol handles both cases"
      - Recovery loops: a participant that repeatedly queries an unavailable coordinator must use exponential backoff to avoid overwhelming the network
    concepts:
      - Coordinator crash recovery via WAL replay
      - Participant uncertainty state and termination protocol
      - Blocking property of 2PC (fundamental limitation)
      - Presumed abort optimization
      - Single point of failure analysis
    deliverables:
      - Coordinator recovery: WAL replay, re-drive in-progress transactions to completion
      - Participant recovery: WAL replay, query coordinator for unknown outcomes
      - Termination protocol: participant QUERY RPC and coordinator response handler
      - Blocking demonstration: test proving participants cannot progress without coordinator
      - Presumed abort: optimized log writes for abort case with correctness test
      - "End-to-end recovery integration test with random crash injection"
    estimated_hours: "5-8"
```
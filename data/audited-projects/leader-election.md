# AUDIT & FIX: leader-election

## CRITIQUE
- **Logical Gap (Confirmed - Split Brain):** Neither the Bully nor Ring algorithm addresses network partitions. Both assume reliable message delivery. In a partition, each partition can elect its own leader, causing split brain. The project MUST address quorum requirements (N/2 + 1 nodes must participate for a valid election) or at minimum acknowledge this limitation explicitly.
- **Technical Inaccuracy (Confirmed - Ring Algorithm):** The Ring algorithm AC says 'the node with the highest ID becomes leader' but doesn't describe the Election vs Elected message phase. Without this two-phase approach, the election message circulates forever. The initiator must detect when its own ID comes back around, then send a Coordinator/Elected message.
- **Missing Fencing:** No mention of fencing tokens or leader epoch numbers. Without these, a node that was leader, got partitioned, and comes back can still act as leader, causing data corruption. Every leader election system needs monotonically increasing epoch/term numbers.
- **M1 Scope Overload:** M1 deliverables include 'Network partition handler' and 'Node failure detector with suspicion levels' — these are advanced topics crammed into the first milestone alongside basic socket programming. This is way too much for a first step.
- **Pitfall Weakness:** 'Split brain' is listed as a pitfall without explaining *why* it happens (network partition) or *how* to detect/prevent it (quorum, fencing). Pitfalls should be educational.
- **Missing Linearizability Discussion:** Leader election is useless if clients can still talk to the old leader. The project needs to address how clients discover the current leader and reject stale leaders.
- **Missing Testing Guidance:** No mention of how to simulate node failures, network partitions, or message delays for testing. This is critical for a distributed systems project.

## FIXED YAML
```yaml
id: leader-election
name: Leader Election
description: >-
  Implement distributed leader election using the Bully algorithm and
  Ring algorithm, with quorum-based split-brain prevention, fencing
  tokens, and fault injection testing.
difficulty: intermediate
estimated_hours: "15-22"
essence: >-
  Coordinating multiple distributed nodes to agree on a single leader
  through asynchronous message-passing protocols, preventing split-brain
  via quorum requirements and fencing tokens (epoch numbers), and handling
  concurrent elections, network partitions, and node failures.
why_important: >-
  Leader election is fundamental to distributed systems like Kubernetes,
  Kafka, and etcd. Building it teaches you fault-tolerant coordination,
  split-brain prevention, and the critical role of epoch/term numbers
  in ensuring correctness during network partitions.
learning_outcomes:
  - Implement inter-node message passing over TCP with timeout-based failure detection
  - Design the Bully algorithm with priority-based leader selection
  - Design the Ring algorithm with two-phase Election/Coordinator messages
  - Prevent split-brain using quorum requirements (majority of nodes)
  - Implement fencing tokens (epoch numbers) to detect stale leaders
  - Test fault tolerance using simulated node failures and network partitions
  - Debug race conditions in concurrent election scenarios
  - Handle node recovery and re-integration into the cluster
skills:
  - Distributed Coordination
  - Failure Detection (heartbeat + timeout)
  - Quorum-based Split-Brain Prevention
  - Fencing Token / Epoch Management
  - Message Passing over TCP
  - Fault Injection Testing
  - State Machine Design
  - Leader Election Algorithms
tags:
  - bully
  - coordination
  - distributed
  - go
  - intermediate
  - java
  - python
  - ring
  - quorum
architecture_doc: architecture-docs/leader-election/index.md
languages:
  recommended:
    - Go
    - Python
    - Java
  also_possible:
    - Rust
    - Erlang
resources:
  - name: Bully Algorithm - Wikipedia
    url: https://en.wikipedia.org/wiki/Bully_algorithm
    type: article
  - name: Ring Election Algorithm
    url: https://www.cs.colostate.edu/~cs551/CourseNotes/Synchronization/LeijdElect.html
    type: article
  - name: "Designing Data-Intensive Applications - Chapter 8 (Leader Election)"
    url: https://dataintensive.net/
    type: book
prerequisites:
  - type: skill
    name: TCP socket programming
  - type: skill
    name: Concurrency (threads/goroutines)
  - type: skill
    name: Basic distributed systems concepts
milestones:
  - id: leader-election-m1
    name: Node Communication and Failure Detection
    description: >-
      Set up a cluster of nodes that can send point-to-point and broadcast
      messages over TCP. Implement heartbeat-based failure detection with
      configurable timeouts.
    acceptance_criteria:
      - "Each node has a unique numeric ID and a configurable list of peer addresses (static cluster membership)"
      - "Point-to-point send(target_id, message) delivers a serialized message to a specific peer over TCP"
      - "Broadcast(message) sends a message to all known peers (best effort, no guaranteed delivery)"
      - "Each node sends periodic heartbeat messages to all peers at a configurable interval (default 1s)"
      - "A node is marked as suspected-failed if no heartbeat is received within a configurable timeout (default 3s)"
      - "Node failure/recovery events are logged with timestamp, node ID, and event type"
      - "Messages are JSON-serialized with a type field (HEARTBEAT, ELECTION, OK, COORDINATOR) for dispatch"
    pitfalls:
      - "Not handling partial TCP reads: use length-prefixed framing for messages"
      - "Heartbeat timeout too short relative to network latency: causes false failure detection"
      - "Blocking send() calls: if a peer is slow/down, sending to it blocks heartbeats to other peers. Use per-peer send queues or async I/O."
      - "Not handling connection refused when a peer is down: catch the exception and mark the peer as suspected"
    concepts:
      - Point-to-point and broadcast messaging
      - Heartbeat-based failure detection
      - Message type dispatch
      - Suspected vs confirmed failure
    skills:
      - TCP socket programming
      - Message serialization and framing
      - Background heartbeat scheduling
      - Connection error handling
    deliverables:
      - Node process with unique ID and peer address configuration
      - TCP message transport with send(target, msg) and broadcast(msg)
      - Heartbeat sender and receiver with configurable interval
      - Failure detector marking nodes as suspected after timeout
    estimated_hours: "4-5"

  - id: leader-election-m2
    name: Bully Algorithm with Quorum
    description: >-
      Implement the Bully election algorithm where the highest-ID live
      node becomes leader. Add quorum requirement to prevent split-brain
      during network partitions.
    acceptance_criteria:
      - "When a node detects the current leader as failed, it initiates an election by sending ELECTION to all higher-ID nodes"
      - "If a higher-ID node responds with OK within the election timeout (configurable, default 5s), the initiator defers and waits"
      - "If no higher-ID node responds within the timeout, the initiator declares itself leader and broadcasts COORDINATOR to all nodes"
      - "A node receiving a COORDINATOR message accepts the new leader only if the sender's ID is higher than or equal to its own"
      - "Quorum check: an election is only valid if the electing node can communicate with a majority (N/2 + 1) of the cluster. If quorum is not met, the node does NOT declare itself leader and logs a warning."
      - "Each election produces a monotonically increasing epoch number (term). The COORDINATOR message includes the epoch. Nodes reject COORDINATOR messages with an epoch ≤ their current known epoch."
      - "Concurrent elections (two nodes detect failure simultaneously) converge to the same leader within 2 election timeout periods"
    pitfalls:
      - "Split brain without quorum: in a 5-node cluster partitioned 2|3, both sides could elect a leader. The quorum requirement prevents the minority partition from electing."
      - "Missing epoch/term numbers: a former leader that was partitioned and comes back sends stale COORDINATOR messages. Without epoch comparison, other nodes accept the stale leader."
      - "Election storm: if timeout is too short, nodes repeatedly trigger elections. Use exponential backoff between election attempts."
      - "Not handling the case where the new leader crashes immediately after sending COORDINATOR: nodes must re-detect failure and re-elect."
    concepts:
      - Bully algorithm three-phase protocol (ELECTION → OK → COORDINATOR)
      - Quorum requirement for partition tolerance
      - Epoch/term numbers for leader validity
      - Concurrent election convergence
    skills:
      - Election state machine implementation
      - Quorum calculation (N/2 + 1)
      - Epoch tracking and comparison
      - Timeout and backoff management
    deliverables:
      - Bully election initiator triggered by leader failure detection
      - ELECTION, OK, and COORDINATOR message handlers
      - Quorum check before declaring victory
      - Epoch number tracking and validation on COORDINATOR messages
    estimated_hours: "5-6"

  - id: leader-election-m3
    name: Ring Election with Two-Phase Protocol
    description: >-
      Implement the Ring election algorithm where nodes are arranged in a
      logical ring. Election messages collect live node IDs as they
      traverse the ring. Use a two-phase protocol (Election phase →
      Coordinator phase) to prevent infinite loops.
    acceptance_criteria:
      - "Nodes are arranged in a logical ring ordered by ID; each node knows its successor (next higher ID, wrapping around)"
      - "Phase 1 (Election): initiator sends an ELECTION message containing its own ID to its successor. Each node appends its ID and forwards to its successor."
      - "When the ELECTION message returns to the initiator (its ID is already in the list), Phase 1 is complete. The initiator selects the highest ID from the collected list."
      - "Phase 2 (Coordinator): initiator sends a COORDINATOR message with the winner's ID and epoch around the ring. Each node records the new leader and epoch."
      - "If a node's successor is down, it skips to the next live node in the ring (ring repair)"
      - "Quorum: the election is valid only if the number of IDs collected in the ELECTION message is > N/2"
      - "Epoch number is incremented on each election and included in COORDINATOR. Nodes reject stale COORDINATOR messages."
    pitfalls:
      - "Infinite loop without two-phase protocol: if there is no termination condition, the ELECTION message circulates forever"
      - "Ring break with multiple consecutive failures: if two adjacent nodes fail, the ring is broken. The skip logic must handle chains of failures."
      - "Multiple simultaneous elections: two nodes detect failure and start elections concurrently. The protocol must converge (typically by the message with the longer ID list absorbing the shorter one)."
      - "Node rejoin during election: a recovering node receives an ELECTION message with an in-progress election. It should add its ID and forward, not start a new election."
    concepts:
      - Ring topology and successor management
      - Two-phase election protocol (Election → Coordinator)
      - Ring repair by skipping failed successors
      - Election termination detection
    skills:
      - Ring data structure maintenance
      - Two-phase protocol implementation
      - Failure-tolerant message forwarding
      - Concurrent election handling
    deliverables:
      - Ring topology manager with successor lookup and ring repair
      - Phase 1 ELECTION message handler collecting live node IDs
      - Phase 2 COORDINATOR message handler announcing the winner with epoch
      - Ring repair logic skipping failed nodes to maintain message flow
    estimated_hours: "5-6"

  - id: leader-election-m4
    name: Fault Injection Testing
    description: >-
      Write comprehensive tests that simulate node failures, network
      partitions, and concurrent elections to verify correctness.
    acceptance_criteria:
      - "Test: kill the current leader process; verify a new leader is elected within 2 * election_timeout"
      - "Test: partition a 5-node cluster into (2, 3); verify only the majority partition elects a leader; minority partition has no leader"
      - "Test: heal the partition; verify the minority nodes accept the majority's leader (with the correct epoch)"
      - "Test: kill and restart a node; verify it re-integrates and accepts the current leader without triggering a new election"
      - "Test: trigger 3 simultaneous elections from different nodes; verify exactly one leader emerges with a consistent epoch across all nodes"
      - "Test: verify epoch numbers are monotonically increasing across successive elections"
    pitfalls:
      - "Flaky tests from timing dependencies: use controllable/mockable clocks and message delivery"
      - "Not testing the quorum path: many implementations work perfectly until a partition occurs"
      - "Process management in tests: ensure all spawned processes are cleaned up even if the test fails"
    concepts:
      - Fault injection patterns
      - Deterministic testing of distributed systems
      - Partition simulation
      - Test isolation
    skills:
      - Process management in tests
      - Network partition simulation (iptables, in-process message filtering)
      - Distributed system test harness design
      - Assertion design for eventually-consistent properties
    deliverables:
      - Test harness for spawning and killing node processes
      - Network partition simulator (drop messages between node groups)
      - Test suite covering leader failure, partition, heal, rejoin, and concurrent election
      - CI-compatible test runner with timeout and cleanup
    estimated_hours: "3-4"

```
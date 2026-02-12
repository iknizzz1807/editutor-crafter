# Titan Consensus Engine: Design Document


## Overview

Titan implements an industrial-grade Raft consensus algorithm to enable bulletproof distributed systems that maintain data consistency and availability even during network partitions and node failures. The key architectural challenge is achieving strong consistency across a cluster while maintaining availability and handling the complex state transitions required for safe consensus.


> This guide is meant to help you understand the big picture before diving into each milestone. Refer back to it whenever you need context on how components connect.


## Context and Problem Statement

> **Milestone(s):** Foundational understanding for all milestones (1-4)

Building reliable distributed systems is one of the most challenging problems in computer science. When multiple computers must work together to provide a single coherent service, they face fundamental questions that don't exist in single-machine systems: How do they agree on what happened when messages are lost? How do they maintain consistency when the network partitions? How do they continue operating when some machines crash? These questions aren't just academic curiosities—they're the daily reality for any system that spans multiple data centers, regions, or even just multiple servers in a rack.

The consensus problem sits at the heart of these challenges. Consensus algorithms allow a group of distributed nodes to agree on a single value or sequence of operations, even in the presence of failures. This capability is fundamental to building systems that are both consistent and available, from replicated databases and distributed filesystems to blockchain networks and container orchestration platforms.

However, implementing consensus correctly is notoriously difficult. The gap between theoretical algorithms described in academic papers and production-ready systems is enormous. Industrial applications require not just correctness proofs, but also practical considerations: performance under load, graceful degradation during failures, operational simplicity, and the ability to evolve the system over time. Many consensus implementations fail in subtle ways that only emerge under real-world conditions—network partitions that last hours, hardware that fails partially rather than cleanly, or software bugs that corrupt state in ways the original algorithm never anticipated.

### Mental Model: The Democratic Assembly

Before diving into the technical complexities of distributed consensus, it's helpful to understand the core problem through a familiar analogy: **parliamentary democracy**. Imagine a legislative assembly where members must vote on important decisions, but they face communication challenges that mirror those in distributed systems.

In a traditional parliament, members gather in the same room, votes are conducted simultaneously, and results are immediately visible to everyone. This corresponds to a centralized system where all components can communicate instantly and reliably. But what if our parliamentary assembly had to operate under more challenging conditions?

Consider a **distributed parliament** where members are scattered across different cities, connected only by an unreliable postal system. Messages can be delayed, lost, or arrive out of order. Some members might be unreachable for extended periods due to postal strikes or bad weather. Yet the assembly must still make decisions and ensure that all members eventually agree on what was decided.

This scenario captures the essence of distributed consensus:

**The Leadership Challenge**: Just as a parliament needs a Speaker to manage proceedings, distributed systems need leader election. If multiple members simultaneously declare themselves Speaker (split leadership), chaos ensues with conflicting decisions and parliamentary deadlock. The system must ensure that exactly one leader emerges, even when communication is unreliable.

**The Voting Process**: When the Speaker proposes legislation, they must collect votes from members. But unlike a physical parliament where everyone votes simultaneously, our distributed assembly faces timing challenges. The Speaker must handle situations where some members haven't responded (are they deliberating, or did the message never arrive?), responses arrive out of order, or members change their minds after learning about other votes.

**The Record Keeping**: Every decision must be recorded in an official parliamentary log. But in our distributed setting, multiple secretaries in different cities must maintain identical copies of this log. If they receive updates in different orders or some updates are lost, their records will diverge, leading to conflicting interpretations of what was decided. The consensus protocol must ensure all secretaries maintain identical records.

**The Succession Problem**: When a Speaker becomes unreachable, the assembly must elect a new one without creating a power vacuum or, worse, multiple competing leaders. This transition must preserve the integrity of all previous decisions while enabling new business to proceed.

**The Membership Challenge**: As the assembly evolves, new members join and others retire. These membership changes must be carefully coordinated to avoid situations where different subsets of members have different views of who's currently eligible to vote, potentially leading to conflicting decisions from overlapping but different majorities.

This parliamentary analogy illuminates why consensus algorithms like Raft use terms (analogous to legislative sessions), leader election (Speaker selection), log replication (maintaining identical parliamentary records), and careful transition procedures (managing changes in leadership and membership). The fundamental insight is that achieving agreement in an asynchronous, unreliable environment requires sophisticated protocols that handle all the edge cases that would never occur in a centralized system.

> **Key Insight**: Distributed consensus isn't just about getting nodes to agree on a value—it's about maintaining that agreement over time, through failures and changes, while providing a foundation for higher-level applications to build upon.

### The CAP Theorem in Practice

The **CAP theorem**, formulated by Eric Brewer, states that distributed systems can guarantee at most two of three properties: **Consistency** (all nodes see the same data simultaneously), **Availability** (the system remains operational), and **Partition tolerance** (the system continues operating despite network failures). While often cited as a fundamental limitation, the CAP theorem's practical implications are more nuanced and directly influence how we design consensus systems.

In real-world distributed systems, **network partitions are not optional**—they will occur. Cables get cut, switches fail, data centers lose connectivity, and wireless networks experience interference. This reality means that partition tolerance isn't a choice but a requirement, forcing us to choose between consistency and availability when partitions occur.

**The Consistency Choice**: Systems that prioritize consistency will refuse to serve requests during network partitions if doing so might compromise data integrity. Consider a banking system where two data centers lose connectivity. A consistency-focused system will shut down operations rather than risk allowing conflicting transactions that might overdraft accounts or double-spend money. This approach sacrifices availability (the system becomes unavailable during partitions) to maintain consistency (no conflicting data states).

**The Availability Choice**: Systems that prioritize availability will continue serving requests during partitions, even if it means accepting that different parts of the system might temporarily have inconsistent views of the data. A social media platform might choose this approach—if users in different regions can't see each other's latest posts during a partition, the service remains usable, and conflicts can be resolved later when connectivity is restored.

The Raft consensus algorithm, which Titan implements, makes a clear choice: **it prioritizes consistency over availability**. During network partitions, Raft ensures that only the partition containing a majority of nodes can continue processing requests. The minority partition becomes unavailable, refusing to serve requests that might conflict with decisions made by the majority.

This design choice has profound implications for how consensus systems behave in practice:

| Partition Scenario | Raft Behavior | Consistency Impact | Availability Impact |
|-------------------|---------------|-------------------|-------------------|
| Majority partition | Continues normal operation | Maintains consistency | Full availability |
| Minority partition | Becomes read-only or unavailable | Maintains consistency | Reduced availability |
| Even split (rare) | Both sides become unavailable | Maintains consistency | Complete unavailability |

**The Quorum Requirement**: Raft's consistency guarantee relies on the **majority quorum principle**. In a cluster of N nodes, any decision requires agreement from at least ⌊N/2⌋ + 1 nodes. This ensures that two conflicting decisions cannot both achieve majority support, preventing the system from entering inconsistent states.

Consider a 5-node cluster that experiences a network partition dividing it into groups of 3 and 2 nodes. The group with 3 nodes can still form a majority (3 > 5/2), so it continues operating normally. The group with 2 nodes cannot form a majority (2 < 5/2), so it stops accepting writes to prevent conflicts. When the partition heals, the minority group synchronizes with the majority group's state, ensuring consistency is maintained.

**Practical Implications for System Design**: Understanding CAP theorem implications helps architects make informed decisions about system behavior:

1. **Cluster Sizing**: Odd-numbered clusters (3, 5, 7 nodes) are preferred because they can tolerate the same number of failures as the next even-numbered cluster while requiring fewer resources. A 3-node cluster can tolerate 1 failure, same as a 4-node cluster.

2. **Deployment Topology**: Nodes should be distributed across failure domains (racks, data centers, availability zones) to ensure that common infrastructure failures don't eliminate the majority quorum. However, spreading nodes across high-latency links (e.g., cross-region) impacts performance.

3. **Application Behavior**: Applications built on Raft must handle scenarios where write operations become unavailable during minority partitions. This might involve falling back to cached data, degraded functionality, or explicit error messages to users.

4. **Operational Considerations**: Operators must understand that certain maintenance operations (taking down multiple nodes simultaneously) can impact availability, and plan accordingly.

> **Design Principle**: Consensus systems don't eliminate the CAP theorem's constraints—they make explicit, predictable choices about how to handle them. Raft's choice to prioritize consistency makes it suitable for applications where data integrity is more important than continuous availability.

### Consensus Algorithm Landscape

The field of distributed consensus has evolved through decades of research and practical experience, yielding several prominent algorithms with different trade-offs and design philosophies. Understanding this landscape helps contextualize why Raft was chosen for Titan and how it compares to alternatives.

**Paxos: The Theoretical Foundation**: Paxos, introduced by Leslie Lamport, was the first consensus algorithm to be rigorously proven correct in asynchronous networks with node failures. Paxos works through a multi-phase protocol where proposers suggest values, acceptors vote on proposals, and learners discover the chosen value. The algorithm's safety guarantees are mathematically elegant: if a value is chosen, all future attempts to choose will select the same value.

However, Paxos's theoretical elegance comes with practical challenges:

| Paxos Characteristics | Benefits | Drawbacks |
|----------------------|----------|-----------|
| Multi-phase protocol | Strong safety guarantees | Complex to implement correctly |
| Role-based design | Clear separation of concerns | Requires multiple message rounds |
| Flexible leadership | Can operate without stable leader | Performance suffers without leader |
| Academic precision | Well-understood theoretical properties | Gap between theory and practice |

The complexity of implementing Paxos correctly has led to what researchers call the "Paxos complexity problem." While the core algorithm is well-defined, building a complete system requires solving many practical issues that the original paper doesn't address: leader election, log management, failure detection, and performance optimizations. Different implementations make different choices for these issues, leading to systems that are theoretically equivalent but practically quite different.

**PBFT: Byzantine Fault Tolerance**: Practical Byzantine Fault Tolerance (PBFT), developed by Castro and Liskov, addresses a different class of failures than Paxos or Raft. While Paxos and Raft assume that nodes either operate correctly or crash (fail-stop failures), PBFT handles Byzantine failures where nodes might behave arbitrarily—sending conflicting messages, corrupting data, or even acting maliciously.

PBFT's approach to handling Byzantine failures involves:

| PBFT Characteristics | Benefits | Drawbacks |
|---------------------|----------|-----------|
| Byzantine fault tolerance | Handles malicious nodes | Requires 3f+1 nodes to tolerate f failures |
| Three-phase protocol | Strong safety under adversarial conditions | High message complexity (O(n²)) |
| Deterministic view changes | Predictable leader succession | Performance degrades with cluster size |
| Cryptographic authentication | Prevents message forgery | Computational overhead for signatures |

The 3f+1 requirement means PBFT clusters are significantly larger than Raft clusters for equivalent fault tolerance. To tolerate one Byzantine failure, PBFT requires 4 nodes, while Raft needs only 3 nodes to tolerate one crash failure. This difference becomes more pronounced as the desired fault tolerance increases.

**Raft: Understandability as a Design Goal**: Raft was explicitly designed to be more understandable than Paxos while providing equivalent safety and liveness guarantees. Its creators, Diego Ongaro and John Ousterhout, identified understandability as a crucial factor in building reliable systems, arguing that complex algorithms lead to implementation bugs and operational mistakes.

Raft achieves understandability through several design principles:

| Raft Design Principle | Implementation | Benefits |
|----------------------|----------------|----------|
| Strong leader | All log entries flow through leader | Simplifies reasoning about system state |
| Term-based epochs | Each leader has unique term number | Eliminates conflicting leadership |
| Log matching property | Identical logs up to any point | Simplifies consistency reasoning |
| Randomized timeouts | Prevents split votes in elections | Reduces election conflicts |
| Joint consensus | Two-phase membership changes | Safe cluster reconfiguration |

The strong leader approach means that clients only interact with the leader, and all log entries originate from the leader. This eliminates many of the complex scenarios that Paxos must handle, where multiple nodes might simultaneously try to propose values.

**Comparative Analysis**: Each algorithm represents different trade-offs in the design space:

| Algorithm | Failure Model | Fault Tolerance | Message Complexity | Understandability | Practical Adoption |
|-----------|---------------|-----------------|-------------------|-------------------|-------------------|
| Paxos | Crash failures | f+1 nodes for f failures | O(n) per decision | Low | Moderate (Google, etc.) |
| PBFT | Byzantine failures | 3f+1 nodes for f failures | O(n²) per decision | Low | Limited (blockchain) |
| Raft | Crash failures | 2f+1 nodes for f failures | O(n) per decision | High | High (etcd, Consul, etc.) |

> **Decision: Algorithm Selection for Titan**
> - **Context**: Industrial distributed systems need consensus that balances theoretical correctness with practical implementability and operational simplicity.
> - **Options Considered**: 
>   1. Paxos for its theoretical maturity and flexibility
>   2. PBFT for its Byzantine fault tolerance capabilities
>   3. Raft for its emphasis on understandability and practical implementation
> - **Decision**: Implement Raft consensus algorithm
> - **Rationale**: Titan targets industrial use cases where crash failures are the primary concern, implementation correctness is crucial, and operational teams need to understand and debug the system. Raft's emphasis on understandability, combined with its proven track record in production systems like etcd and Consul, makes it the best fit. The vast majority of distributed system failures in practice stem from implementation bugs rather than theoretical algorithm weaknesses, making Raft's implementability advantage decisive.
> - **Consequences**: Titan will handle crash failures but not Byzantine failures. The system will be easier to implement, test, and operate, but will require careful deployment practices to ensure the crash failure model matches the actual deployment environment.

**Evolution and Modern Considerations**: The consensus landscape continues evolving with new algorithms and optimizations:

- **Multi-Raft**: Systems like CockroachDB use multiple Raft groups to scale beyond single cluster limits
- **Flexible Paxos**: Recent work shows that Paxos's quorum requirements can be relaxed in certain scenarios
- **Blockchain Consensus**: Algorithms like PBFT have been adapted for cryptocurrency and blockchain applications
- **Specialized Consensus**: Domain-specific optimizations for particular workloads (read-heavy, write-heavy, etc.)

For Titan's goals of providing an industrial-grade foundation for distributed systems, Raft's combination of proven correctness, implementation simplicity, and operational understandability makes it the optimal choice. The algorithm's widespread adoption in production systems provides additional confidence in its real-world viability.

### Implementation Guidance

Understanding the consensus landscape and fundamental trade-offs provides the foundation for implementing Titan. This section bridges the conceptual understanding developed above with practical implementation considerations.

#### Technology Recommendations

| Component | Simple Option | Advanced Option | Recommendation for Titan |
|-----------|---------------|-----------------|-------------------------|
| Transport | HTTP REST + JSON | gRPC + Protocol Buffers | gRPC for performance and schema evolution |
| Serialization | JSON | Protocol Buffers, MessagePack | Protocol Buffers for efficiency and type safety |
| Storage | File-based logs | Embedded database (SQLite, RocksDB) | File-based logs for simplicity and control |
| Logging | Python logging module | Structured logging (structlog) | Structured logging for operational visibility |
| Testing | unittest | pytest + property-based testing (hypothesis) | pytest with hypothesis for consensus properties |
| Configuration | YAML files | etcd, Consul | YAML files to avoid circular dependencies |

The technology choices reflect the principle of starting simple but planning for scale. Early implementations can use simpler options while the architecture supports migrating to more sophisticated approaches as requirements evolve.

#### Recommended Module Structure

Organizing the codebase from the beginning prevents the common mistake of cramming all consensus logic into a single monolithic file. The structure below supports the milestone-based development approach:

```
titan/
├── __init__.py
├── main.py                     # Entry point and cluster bootstrap
├── core/
│   ├── __init__.py
│   ├── node.py                 # Main RaftNode class and state management
│   ├── types.py                # Core data types (Term, LogEntry, etc.)
│   └── state_machine.py        # State machine interface and implementations
├── consensus/
│   ├── __init__.py
│   ├── election.py             # Leader election logic (Milestone 1)
│   ├── replication.py          # Log replication (Milestone 2)
│   ├── compaction.py           # Log compaction and snapshots (Milestone 3)
│   └── membership.py           # Cluster membership changes (Milestone 4)
├── transport/
│   ├── __init__.py
│   ├── rpc.py                  # RPC message definitions and handling
│   ├── grpc_transport.py       # gRPC transport implementation
│   └── local_transport.py      # In-memory transport for testing
├── storage/
│   ├── __init__.py
│   ├── log.py                  # Persistent log implementation
│   ├── state.py                # Persistent state (term, vote)
│   └── snapshot.py             # Snapshot creation and management
├── test/
│   ├── __init__.py
│   ├── unit/                   # Unit tests for individual components
│   ├── integration/            # Multi-node integration tests
│   └── chaos/                  # Chaos engineering and partition tests
└── examples/
    ├── kv_store.py             # Example key-value store state machine
    └── counter.py              # Simple counter state machine
```

This structure separates concerns cleanly:
- **Core**: Fundamental types and the main node orchestration
- **Consensus**: Milestone-specific consensus logic that can be developed incrementally
- **Transport**: Network communication abstracted behind interfaces
- **Storage**: Persistence layer with clear separation between logs, state, and snapshots
- **Test**: Comprehensive testing at multiple levels
- **Examples**: Concrete state machines that demonstrate how to use Titan

#### Infrastructure Starter Code

Several components are necessary infrastructure but not core to learning consensus algorithms. These complete implementations can be used directly:

**Basic gRPC Transport** (transport/grpc_transport.py):
```python
"""
Complete gRPC transport implementation for Raft RPCs.
This handles the network communication so you can focus on consensus logic.
"""

import asyncio
import grpc
from typing import Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor

from ..core.types import NodeId, RequestVoteRequest, RequestVoteResponse
from ..core.types import AppendEntriesRequest, AppendEntriesResponse
from .rpc import RaftServicer, add_RaftServicer_to_server

class GrpcTransport:
    """
    Production-ready gRPC transport for Raft consensus messages.
    Handles connection management, retries, and failure detection.
    """
    
    def __init__(self, node_id: NodeId, listen_addr: str, peer_addresses: Dict[NodeId, str]):
        self.node_id = node_id
        self.listen_addr = listen_addr
        self.peer_addresses = peer_addresses
        self.peer_stubs: Dict[NodeId, any] = {}
        self.server: Optional[grpc.aio.Server] = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    async def start(self, raft_node) -> None:
        """Start the gRPC server and initialize peer connections."""
        # Server setup
        self.server = grpc.aio.server(self.executor)
        servicer = RaftServicer(raft_node)
        add_RaftServicer_to_server(servicer, self.server)
        
        listen_port = self.listen_addr.split(':')[1]
        self.server.add_insecure_port(f'[::]:{listen_port}')
        await self.server.start()
        
        # Initialize peer connections
        for peer_id, addr in self.peer_addresses.items():
            if peer_id != self.node_id:
                channel = grpc.aio.insecure_channel(addr)
                self.peer_stubs[peer_id] = RaftStub(channel)
    
    async def send_request_vote(self, target: NodeId, request: RequestVoteRequest) -> Optional[RequestVoteResponse]:
        """Send RequestVote RPC to target node with timeout and retry logic."""
        if target not in self.peer_stubs:
            return None
            
        try:
            stub = self.peer_stubs[target]
            response = await asyncio.wait_for(
                stub.RequestVote(request),
                timeout=1.0  # 1 second timeout for vote requests
            )
            return response
        except (grpc.RpcError, asyncio.TimeoutError) as e:
            # Log the error but don't crash - this is expected during network issues
            print(f"RequestVote to {target} failed: {e}")
            return None
    
    async def send_append_entries(self, target: NodeId, request: AppendEntriesRequest) -> Optional[AppendEntriesResponse]:
        """Send AppendEntries RPC to target node with timeout."""
        if target not in self.peer_stubs:
            return None
            
        try:
            stub = self.peer_stubs[target]
            response = await asyncio.wait_for(
                stub.AppendEntries(request),
                timeout=5.0  # Longer timeout for log replication
            )
            return response
        except (grpc.RpcError, asyncio.TimeoutError) as e:
            print(f"AppendEntries to {target} failed: {e}")
            return None
    
    async def stop(self) -> None:
        """Gracefully shutdown the transport."""
        if self.server:
            await self.server.stop(grace=5.0)
        self.executor.shutdown(wait=True)
```

**Persistent State Manager** (storage/state.py):
```python
"""
Complete persistent state implementation.
Handles the durable state required by Raft consensus.
"""

import os
import json
import fcntl
from typing import Optional
from pathlib import Path

from ..core.types import Term, NodeId

class PersistentState:
    """
    Manages the persistent state required by Raft: current term and voted for.
    Uses file-based storage with atomic writes and fsync for durability.
    """
    
    def __init__(self, data_dir: str, node_id: NodeId):
        self.data_dir = Path(data_dir)
        self.node_id = node_id
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.data_dir / f"raft_state_{node_id}.json"
        self._current_term: Term = Term(0)
        self._voted_for: Optional[NodeId] = None
        self._load_state()
    
    def _load_state(self) -> None:
        """Load persistent state from disk, initializing if file doesn't exist."""
        if not self.state_file.exists():
            self._save_state()
            return
            
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
                self._current_term = Term(data.get('current_term', 0))
                self._voted_for = data.get('voted_for')
        except (IOError, json.JSONDecodeError) as e:
            # If we can't read the state file, start fresh
            print(f"Warning: Could not load state file {self.state_file}: {e}")
            print("Starting with fresh state")
            self._current_term = Term(0)
            self._voted_for = None
            self._save_state()
    
    def _save_state(self) -> None:
        """Atomically save state to disk with fsync for durability."""
        state_data = {
            'current_term': self._current_term,
            'voted_for': self._voted_for,
            'node_id': self.node_id
        }
        
        # Atomic write: write to temp file, fsync, then rename
        temp_file = self.state_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
            
            # Atomic rename
            temp_file.rename(self.state_file)
        except IOError as e:
            # Clean up temp file on failure
            if temp_file.exists():
                temp_file.unlink()
            raise RuntimeError(f"Failed to save persistent state: {e}")
    
    @property
    def current_term(self) -> Term:
        """Get the current term."""
        return self._current_term
    
    @property
    def voted_for(self) -> Optional[NodeId]:
        """Get the candidate voted for in current term (if any)."""
        return self._voted_for
    
    def update_term(self, new_term: Term, voted_for: Optional[NodeId] = None) -> None:
        """
        Update the current term and optionally record a vote.
        Automatically saves to disk for durability.
        """
        if new_term < self._current_term:
            raise ValueError(f"Cannot update to older term {new_term} from {self._current_term}")
        
        if new_term > self._current_term:
            # New term resets the vote
            self._current_term = new_term
            self._voted_for = voted_for
        elif new_term == self._current_term and voted_for is not None:
            # Voting in current term
            if self._voted_for is not None and self._voted_for != voted_for:
                raise ValueError(f"Already voted for {self._voted_for} in term {new_term}")
            self._voted_for = voted_for
        
        self._save_state()
```

#### Core Logic Skeleton

The core consensus logic should be implemented by learners following the design patterns established in this document. Here's the skeleton for the main `RaftNode` class:

```python
"""
Core Raft node implementation skeleton.
TODO: Implement the consensus logic following the algorithms described in the design doc.
"""

from enum import Enum
from typing import Optional, Dict, List, Set
import asyncio
import random

from .types import NodeId, Term, LogIndex, LogEntry
from ..storage.state import PersistentState
from ..storage.log import PersistentLog
from ..transport.grpc_transport import GrpcTransport

class NodeState(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate" 
    LEADER = "leader"

class RaftNode:
    """
    Core Raft consensus node implementation.
    Manages state transitions, elections, and log replication.
    """
    
    def __init__(self, node_id: NodeId, cluster_members: Set[NodeId], 
                 data_dir: str, transport: GrpcTransport):
        self.node_id = node_id
        self.cluster_members = cluster_members
        self.transport = transport
        
        # Persistent state (survives crashes)
        self.persistent_state = PersistentState(data_dir, node_id)
        self.log = PersistentLog(data_dir, node_id)
        
        # Volatile state (reset on crash)
        self.state = NodeState.FOLLOWER
        self.current_leader: Optional[NodeId] = None
        self.commit_index = LogIndex(0)
        self.last_applied = LogIndex(0)
        
        # Leader volatile state (valid only when leader)
        self.next_index: Dict[NodeId, LogIndex] = {}
        self.match_index: Dict[NodeId, LogIndex] = {}
        
        # Election timing
        self.election_timeout_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        
    async def start(self) -> None:
        """
        Start the Raft node and begin participating in consensus.
        TODO: Initialize state and start election timeout.
        """
        # TODO 1: Initialize transport and start listening for RPCs
        # TODO 2: Start as follower with election timeout
        # TODO 3: Begin the main consensus loop
        pass
    
    async def _start_election_timeout(self) -> None:
        """
        Start the randomized election timeout.
        TODO: Implement election timeout with randomization to prevent split votes.
        """
        # TODO 1: Generate random timeout between 150ms and 300ms
        # TODO 2: Wait for timeout duration
        # TODO 3: If no leader heartbeat received, start election
        # Hint: Use asyncio.sleep() and handle cancellation when heartbeat arrives
        pass
    
    async def _start_election(self) -> None:
        """
        Begin leader election by transitioning to candidate and requesting votes.
        TODO: Implement the election algorithm from the Election section.
        """
        # TODO 1: Transition to CANDIDATE state
        # TODO 2: Increment current term and vote for self
        # TODO 3: Send RequestVote RPCs to all other nodes
        # TODO 4: Count votes and determine if majority achieved
        # TODO 5: If majority, become leader; otherwise, return to follower
        # Hint: Use asyncio.gather() to send votes concurrently
        pass
    
    async def _become_leader(self) -> None:
        """
        Transition to leader state and begin sending heartbeats.
        TODO: Initialize leader state and start heartbeat process.
        """
        # TODO 1: Set state to LEADER and update current_leader
        # TODO 2: Initialize next_index and match_index for all followers
        # TODO 3: Cancel election timeout task
        # TODO 4: Start periodic heartbeat task
        # TODO 5: Send initial empty AppendEntries (heartbeat) to establish leadership
        pass
    
    async def handle_request_vote(self, request) -> any:
        """
        Handle RequestVote RPC from candidate.
        TODO: Implement vote decision logic based on Raft safety rules.
        """
        # TODO 1: Check if candidate's term is at least as current as ours
        # TODO 2: Check if we haven't already voted in this term
        # TODO 3: Check if candidate's log is at least as up-to-date as ours
        # TODO 4: Grant vote if all conditions met, deny otherwise
        # TODO 5: Update persistent state if granting vote
        # TODO 6: Reset election timeout if we grant vote
        pass
    
    async def handle_append_entries(self, request) -> any:
        """
        Handle AppendEntries RPC from leader.
        TODO: Implement log consistency checks and entry appending.
        """
        # TODO 1: Check if leader's term is at least as current as ours
        # TODO 2: Verify log consistency at prev_log_index
        # TODO 3: Remove conflicting entries if any
        # TODO 4: Append new entries to log
        # TODO 5: Update commit_index based on leader_commit
        # TODO 6: Reset election timeout since we heard from leader
        pass
```

#### Language-Specific Implementation Hints

**Python Asyncio Patterns**: Raft requires careful coordination of concurrent activities (election timeouts, heartbeats, RPC handling). Use these patterns:

- `asyncio.create_task()` for background tasks that should run concurrently
- `asyncio.gather()` for waiting on multiple concurrent operations (like sending votes to all nodes)
- `asyncio.wait_for()` with timeouts for RPC calls that might hang
- Task cancellation (`task.cancel()`) when state transitions invalidate ongoing operations

**Persistent State Management**: The Raft safety properties depend critically on persistent state surviving crashes:

- Always call `fsync()` after writing critical state (term, vote) before responding to RPCs
- Use atomic file operations (write to temp file, fsync, rename) to prevent corruption
- Design your state classes to make it impossible to forget persistence (automatic save on modification)

**Error Handling Philosophy**: Consensus systems must distinguish between expected failures (network issues, node crashes) and unexpected failures (programming bugs):

- Network timeouts and connection failures should be logged but not crash the node
- Programming errors (assertion failures, unexpected state) should crash fast for easier debugging
- Use type hints and validation to catch programming errors early

**Testing Considerations**: Consensus algorithms have subtle correctness properties that are easy to violate:

- Write property-based tests that verify safety invariants (never two leaders in same term)
- Use deterministic testing with controlled network delays and failures
- Test all state transition combinations (follower→candidate→leader, leader→follower, etc.)

#### Milestone Checkpoint: Foundation Understanding

After implementing this foundational understanding:

**What to verify**:
1. You can explain the consensus problem using the parliamentary democracy analogy
2. You understand why Titan chooses consistency over availability during partitions
3. You can articulate why Raft was chosen over Paxos or PBFT for this project

**Expected behavior**:
- The basic project structure is set up with the recommended module organization
- The infrastructure starter code compiles and basic transport tests pass
- You have a clear mental model of how the four milestones build upon each other

**Signs something is wrong**:
- You're overwhelmed by the complexity and don't know where to start (revisit the mental models)
- You want to implement everything at once rather than following the milestone progression
- You're making technology choices that contradict the recommendations without clear rationale

This foundational understanding provides the conceptual framework for implementing the Raft consensus algorithm across the four development milestones.


## Goals and Non-Goals

> **Milestone(s):** All milestones (1-4) - defines the scope and requirements for the entire Titan consensus engine

The first principle of successful software architecture is knowing exactly what you're building and, equally important, what you're *not* building. Think of designing a consensus engine like planning a space mission - you must define precise mission parameters, performance constraints, and acceptable trade-offs before engineering a single component. Without clear boundaries, feature creep and scope expansion will derail even the most technically sound implementation.

Titan's requirements emerge from real-world distributed systems pain points: data corruption during network partitions, split-brain scenarios that compromise safety, and consensus algorithms that work in academic papers but fail under production load. Our goals reflect the harsh realities of industrial distributed systems where network failures are routine, nodes crash unexpectedly, and business logic cannot tolerate data inconsistency.

### Functional Requirements

The core behaviors that Titan must exhibit represent the fundamental guarantees that any system built on top of Raft consensus depends upon. These requirements directly map to the safety and liveness properties that make distributed consensus mathematically sound and practically useful.

**State Machine Replication**: Titan must maintain identical state machines across all nodes in the cluster, with strong guarantees about consistency and ordering. The consensus engine replicates a log of operations, and each node applies these operations to its local state machine in identical order. This is the foundation that enables distributed systems to present a single, consistent view of data to clients.

| Requirement | Specification | Verification Method |
|-------------|---------------|-------------------|
| Log Ordering | All committed entries appear in identical order across all nodes | Compare log sequences after partition recovery |
| State Machine Safety | Applying same log sequence produces identical state on all nodes | Hash comparison of state machine outputs |
| Entry Commitment | Once an entry is committed on any node, it appears on all nodes | Verify commitment propagation during network failures |
| Linearizability | Operations appear to execute atomically at some point between start and completion | Timeline analysis of concurrent operations |

**Leader Election Guarantees**: The system must ensure exactly one leader exists per term, preventing the split-brain scenarios that plague many distributed systems. This requires sophisticated election algorithms that handle network partitions, node failures, and timing edge cases.

The election process must be **deterministic yet randomized** - deterministic in that the same network conditions always produce valid outcomes, but randomized to prevent deadlock scenarios where multiple candidates continuously split the vote. Election timeouts use randomized intervals between `ELECTION_TIMEOUT_MIN` and `ELECTION_TIMEOUT_MAX` to ensure that in most cases, only one node times out and starts an election.

| Election Property | Requirement | Implementation Constraint |
|------------------|------------|--------------------------|
| Single Leader | At most one leader per term | Majority vote requirement |
| Election Safety | Only candidates with up-to-date logs can win | Log comparison in RequestVote |
| Progress Guarantee | Eventually some candidate wins in stable network | Randomized timeouts prevent split votes |
| Term Monotonicity | Terms increase monotonically across all nodes | Persistent term storage |

> **Critical Safety Property**: The fundamental safety guarantee is that if any node commits log entry E at index I, then no node will ever commit a different entry at index I. This property, combined with the log matching property, ensures that all nodes maintain consistent state.

**Persistent State Durability**: Titan must survive node crashes and restarts without compromising safety guarantees. This requires careful management of persistent state - the minimal information that must survive crashes to maintain consensus properties.

The three pieces of persistent state are the current term, the candidate that received this node's vote in the current term (if any), and the complete log of entries. These must be written to stable storage before responding to RPCs or sending votes. The storage system must provide atomic writes and ordering guarantees - partial writes during crashes cannot leave the system in an inconsistent state.

**Network Partition Tolerance**: The system must continue operating correctly during network partitions, maintaining safety even when the cluster splits into multiple segments. Nodes in the minority partition must reject client operations rather than risk data corruption.

During partitions, the majority partition continues processing client requests while the minority partition becomes read-only. When the partition heals, nodes in the minority automatically catch up to the majority's state. This requires sophisticated log reconciliation algorithms that can identify conflicting entries and resolve them safely.

| Partition Scenario | Required Behavior | Safety Mechanism |
|-------------------|------------------|------------------|
| Majority partition | Continue normal operation | Quorum-based decisions |
| Minority partition | Reject writes, allow reads of committed data | Leader cannot achieve majority |
| Partition healing | Minority nodes catch up to majority | Log conflict resolution |
| Repeated partitions | Maintain safety across multiple split/heal cycles | Term-based conflict detection |

**Client Operation Semantics**: Titan must provide clear semantics for client operations, including exactly-once execution guarantees and proper handling of retries during leader changes.

When a client submits an operation, the system guarantees that the operation will be executed exactly once on each node's state machine, even if the client retries due to timeouts or leader failures. This requires duplicate detection mechanisms and careful coordination between the consensus layer and the application state machine.

### Performance and Scale Requirements

Industrial-grade consensus engines must perform well under realistic production workloads while maintaining correctness guarantees. These requirements reflect the operational constraints that determine whether Titan can serve as the foundation for real distributed systems.

**Latency Targets**: Consensus adds inherent latency overhead due to the need for majority coordination. Our targets balance responsiveness with the fundamental constraints of distributed agreement.

| Operation Type | Target Latency (P99) | Rationale |
|---------------|---------------------|-----------|
| Leader Election | < 1 second | Fast recovery from leader failures |
| Entry Commitment | < 100ms | Acceptable for most interactive applications |
| Read Operations | < 10ms | Reads from leader without consensus overhead |
| Follower Catch-up | < 5 seconds for 1000 entries | Reasonable recovery time for short partitions |

The commitment latency target assumes a local area network with sub-millisecond RTT between nodes. Wide-area deployments will require higher latency targets, but the fundamental algorithms remain the same.

**Throughput Requirements**: The system must support realistic workloads without becoming a bottleneck for applications built on top of it.

| Metric | Target | Test Scenario |
|--------|--------|---------------|
| Entry Throughput | 10,000 entries/second | 1KB entries, 5-node cluster, single leader |
| Concurrent Clients | 1,000 active clients | Mixed read/write workload |
| Log Growth | Handle 1M+ entries per node | Long-running systems with compaction |
| Snapshot Creation | < 10 second interruption | During active traffic |

These targets assume modern hardware (SSD storage, gigabit networking) and represent the minimum viable performance for production use. Higher-performance applications may require optimizations like batching, pipelining, or read replicas.

**Cluster Scale Limits**: Raft's performance characteristics change significantly with cluster size due to the need for majority coordination. Our targets reflect practical limits for most use cases.

| Cluster Size | Use Case | Performance Impact |
|-------------|----------|-------------------|
| 3 nodes | Development, small deployments | Optimal performance |
| 5 nodes | Production standard | Minor latency increase |
| 7 nodes | High-availability critical systems | Noticeable latency impact |
| 9+ nodes | Not recommended | Significant performance degradation |

> **Architecture Decision: Cluster Size Optimization**
> - **Context**: Larger clusters provide better fault tolerance but degrade performance due to increased coordination overhead
> - **Options Considered**: Support unlimited cluster sizes, cap at 5 nodes, cap at 9 nodes
> - **Decision**: Support up to 9 nodes with performance warnings beyond 5
> - **Rationale**: 5 nodes tolerates 2 failures which satisfies most use cases; 9 nodes provides extreme fault tolerance for critical systems; beyond 9 nodes, multi-raft approaches are more effective
> - **Consequences**: Simpler implementation and testing; clear guidance for operators; performance remains predictable

**Resource Consumption Limits**: The consensus engine must operate within reasonable resource constraints to avoid overwhelming the systems it runs on.

| Resource | Limit | Monitoring Threshold |
|----------|-------|---------------------|
| Memory Usage | < 1GB per 100K log entries | Alert at 80% of limit |
| Disk I/O | < 100 IOPS for normal operation | Burst to 1000 IOPS during catch-up |
| Network Bandwidth | < 10MB/s sustained | Burst to 100MB/s for snapshots |
| CPU Usage | < 20% of one core | Spike to 50% during elections |

### Explicit Non-Goals

Clearly defining what Titan will *not* do is as important as defining what it will do. These explicit non-goals prevent scope creep and set appropriate expectations for users and developers.

**State Machine Implementation**: Titan provides consensus on the order of operations but does not implement any specific state machine logic. Applications must implement their own state machines that apply committed log entries to update their state.

This separation of concerns allows Titan to serve as a foundation for diverse applications - key-value stores, configuration services, coordination primitives, and custom distributed applications. The consensus engine guarantees that all nodes see the same sequence of operations; the state machine determines what those operations mean.

**Byzantine Fault Tolerance**: Titan assumes a fail-stop failure model where nodes either operate correctly or stop completely. It does not defend against Byzantine faults where nodes may behave arbitrarily or maliciously.

Byzantine fault tolerance requires fundamentally different algorithms (like PBFT) with higher overhead and complexity. For most enterprise use cases, fail-stop assumptions are appropriate when combined with proper security, monitoring, and access controls.

| Failure Type | Handled by Titan? | Alternative Solution |
|--------------|-------------------|---------------------|
| Node crashes | ✅ Yes | Core Raft algorithm |
| Network partitions | ✅ Yes | Majority-based decisions |
| Message delays | ✅ Yes | Timeout-based detection |
| Message corruption | ✅ Yes | Transport layer checksums |
| Malicious nodes | ❌ No | Byzantine fault tolerant consensus |
| Software bugs that cause incorrect behavior | ❌ No | Testing, formal verification, monitoring |

**Multi-Raft Coordination**: Titan implements a single Raft group. It does not provide coordination between multiple Raft groups, transaction semantics across groups, or sharding mechanisms.

Applications that need to scale beyond a single Raft group must implement their own multi-raft coordination, possibly using techniques like two-phase commit or saga patterns. This keeps Titan focused on implementing single-group consensus correctly rather than tackling the additional complexity of distributed transactions.

**Application-Specific Optimizations**: Titan does not include optimizations for specific workloads like read-heavy applications, write-heavy applications, or particular data models.

| Optimization Type | Why Not Included | How Applications Can Address |
|------------------|------------------|----------------------------|
| Read replicas | Adds complexity without universal benefit | Implement at application layer |
| Write batching | Different applications have different batching needs | Application controls entry content |
| Compression | Depends on data characteristics | Compress data before submitting to Titan |
| Encryption | Various key management and algorithm requirements | Encrypt at application or transport layer |

**Operational Tooling**: Titan provides the core consensus algorithm but does not include comprehensive operational tools like monitoring dashboards, automated deployment, or cluster management utilities.

These tools are important for production deployment but vary significantly across environments and organizational needs. Titan provides the necessary instrumentation and APIs for building such tools rather than prescribing specific implementations.

**Cross-Datacenter Replication**: While Titan tolerates network partitions, it is not optimized for wide-area network deployments with high latency between nodes.

Cross-datacenter consensus requires specialized techniques like witness nodes, hierarchical consensus, or eventual consistency models. Titan focuses on providing strong consistency within a single failure domain.

> **Design Principle: Focused Scope**: Titan aims to be an excellent foundation for building distributed systems rather than a complete distributed system platform. This focused scope enables thorough testing, clear documentation, and predictable behavior within the defined boundaries.

**Custom Transport Protocols**: Titan uses standard RPC mechanisms and does not implement custom network protocols optimized for consensus traffic.

While specialized protocols could reduce latency or bandwidth usage, they would significantly increase implementation complexity and reduce interoperability. Standard RPC frameworks provide adequate performance for most use cases while offering better tooling and debugging support.

### Implementation Guidance

The requirements defined above translate into specific implementation constraints and technology choices that guide the development of Titan. This guidance bridges the gap between high-level goals and concrete code.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| RPC Transport | HTTP REST with JSON | gRPC with Protocol Buffers |
| Persistent Storage | JSON files with fsync | Embedded database (SQLite, LevelDB) |
| Networking | Python `requests` library | `asyncio` with custom protocol |
| State Machine Interface | Simple function calls | Plugin architecture with IPC |
| Configuration | YAML files | Dynamic configuration with validation |

**Recommended Project Structure:**
```
titan/
  __init__.py                    ← Package initialization
  node.py                       ← Main RaftNode class
  log.py                        ← Log management and persistence
  rpc.py                        ← Network communication
  storage.py                    ← Persistent state management
  types.py                      ← Core data types and enums
  config.py                     ← Configuration management
  
  tests/
    test_election.py            ← Election algorithm tests
    test_replication.py         ← Log replication tests
    test_compaction.py          ← Snapshot and compaction tests
    test_membership.py          ← Membership change tests
    integration/                ← End-to-end testing
    
  examples/
    kv_store.py                ← Key-value store state machine
    counter.py                 ← Simple counter example
    
  tools/
    cluster_admin.py           ← Cluster management utilities
    log_inspector.py           ← Log analysis and debugging
```

**Core Type Definitions:**
```python
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
import json

class NodeState(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate" 
    LEADER = "leader"

@dataclass
class LogEntry:
    """Individual entry in the replicated log"""
    term: int           # Term when entry was created
    index: int          # Position in log (1-indexed)
    data: bytes         # Application data
    timestamp: float    # Creation timestamp for debugging
    
    def to_dict(self) -> Dict[str, Any]:
        # TODO: Implement serialization for persistence
        pass
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        # TODO: Implement deserialization
        pass

@dataclass
class RequestVoteRequest:
    """RPC request for candidate to request votes"""
    term: int           # Candidate's term
    candidate_id: str   # Candidate requesting vote
    last_log_index: int # Index of candidate's last log entry
    last_log_term: int  # Term of candidate's last log entry

@dataclass  
class RequestVoteResponse:
    """RPC response to vote request"""
    term: int           # Current term, for candidate to update itself
    vote_granted: bool  # True means candidate received vote
```

**Configuration Constants:**
```python
# Election timing constants
ELECTION_TIMEOUT_MIN = 0.15    # 150ms minimum
ELECTION_TIMEOUT_MAX = 0.30    # 300ms maximum  
HEARTBEAT_INTERVAL = 0.05      # 50ms heartbeat interval

# Performance and scale limits
MAX_CLUSTER_SIZE = 9
MAX_ENTRIES_PER_APPEND = 100   # Batch size for efficiency
SNAPSHOT_THRESHOLD = 10000     # Trigger snapshot after this many entries
```

**Requirements Validation Framework:**
```python
class RequirementsValidator:
    """Validates that implementation meets functional requirements"""
    
    def validate_election_safety(self, cluster_state: Dict[str, Any]) -> bool:
        """Verify at most one leader per term"""
        # TODO: Check that no two nodes believe they are leader in same term
        # TODO: Verify leader has majority of votes
        pass
    
    def validate_log_matching(self, node_logs: Dict[str, List[LogEntry]]) -> bool:
        """Verify log matching property across all nodes"""
        # TODO: Compare logs across nodes up to commit index
        # TODO: Ensure identical entries at each index
        pass
    
    def validate_state_machine_safety(self, node_states: Dict[str, Any]) -> bool:
        """Verify state machines converge to same state"""
        # TODO: Compare state machine outputs after applying same log
        pass
```

**Performance Monitoring Integration:**
```python
from dataclasses import dataclass
import time
from typing import Dict

@dataclass
class PerformanceMetrics:
    """Tracks performance against requirements"""
    election_latencies: List[float]
    commit_latencies: List[float]
    throughput_samples: List[int]
    resource_usage: Dict[str, float]
    
    def check_latency_requirements(self) -> Dict[str, bool]:
        """Verify latency targets are met"""
        # TODO: Calculate P99 latencies
        # TODO: Compare against targets in requirements
        pass
    
    def check_throughput_requirements(self) -> bool:
        """Verify throughput targets are met"""
        # TODO: Calculate sustained throughput
        # TODO: Check against 10,000 entries/second target
        pass
```

**Milestone Validation Checkpoints:**

After implementing each milestone, run these validation checks:

| Milestone | Validation Command | Expected Result |
|-----------|-------------------|-----------------|
| 1: Election & Safety | `python -m pytest tests/test_election.py -v` | All election tests pass |
| 2: Log Replication | `python -m pytest tests/test_replication.py -v` | All replication tests pass |
| 3: Log Compaction | `python -m pytest tests/test_compaction.py -v` | All compaction tests pass |
| 4: Membership Changes | `python -m pytest tests/test_membership.py -v` | All membership tests pass |

**Integration Testing Setup:**
```python
import subprocess
import time

def start_test_cluster(num_nodes: int = 3) -> List[subprocess.Popen]:
    """Start a test cluster for integration testing"""
    # TODO: Start multiple node processes
    # TODO: Configure them to communicate with each other
    # TODO: Return process handles for cleanup
    pass

def submit_test_operations(leader_addr: str, operations: List[Dict]) -> List[Dict]:
    """Submit operations to cluster and verify they commit"""
    # TODO: Send operations to leader
    # TODO: Wait for commitment
    # TODO: Verify operations appear on all nodes
    pass
```

This implementation guidance provides the concrete foundation for building Titan while ensuring adherence to the functional, performance, and scope requirements defined above. The key insight is that clear requirements enable focused implementation - developers know exactly what to build and can validate their progress against objective criteria.


## High-Level Architecture

> **Milestone(s):** All milestones (1-4) - establishes the foundational architecture that supports election, replication, compaction, and membership changes

The architecture of a consensus system is fundamentally different from typical application architectures because it must handle the unique challenges of distributed agreement. Think of Titan's architecture like a **diplomatic embassy**: multiple specialized departments (components) work together to maintain relationships with foreign entities (other nodes), keep detailed records of all communications (logs), and ensure that important decisions are made collectively and recorded permanently. Each department has clear responsibilities, but they must coordinate carefully to maintain consistency and handle the inevitable communication failures that occur in international relations.

The Titan consensus engine is built around four core architectural principles: **separation of concerns** between consensus logic and application state, **event-driven coordination** between components to handle asynchronous distributed operations, **persistent state management** to survive crashes and maintain safety guarantees, and **explicit failure handling** at every component boundary. These principles guide every architectural decision and ensure that the system can maintain correctness even when individual components fail or network conditions deteriorate.

### Component Responsibilities

The Titan architecture consists of five primary components that collaborate to implement the Raft consensus protocol. Each component has clearly defined responsibilities and interfaces, allowing for modular implementation and testing while ensuring that the complex interactions required for consensus are handled correctly.

#### Consensus Engine Core

The **Consensus Engine** serves as the central coordinator and state machine for the Raft protocol itself. Think of it as the **prime minister's office** in our diplomatic embassy analogy - it makes the high-level decisions, coordinates between departments, and maintains the overall strategic direction. The Consensus Engine owns the fundamental Raft state transitions, election logic, and protocol orchestration.

| Responsibility | Description | Key Operations |
|---|---|---|
| Node State Management | Maintains current node role (follower/candidate/leader) and handles state transitions | `transition_to_candidate()`, `become_leader()`, `step_down()` |
| Election Coordination | Orchestrates leader election process with randomized timeouts and vote collection | `start_election()`, `collect_votes()`, `handle_vote_response()` |
| Term Management | Tracks current term and ensures monotonic term progression across the cluster | `update_current_term()`, `advance_term()`, `validate_term()` |
| RPC Message Routing | Processes incoming Raft RPCs and coordinates responses | `handle_request_vote()`, `handle_append_entries()`, `handle_install_snapshot()` |
| Heartbeat Management | Maintains leader authority through periodic heartbeats and detects leader failure | `send_heartbeats()`, `reset_election_timer()`, `check_leader_timeout()` |
| Protocol Safety | Enforces Raft safety properties and prevents split-brain scenarios | `validate_log_consistency()`, `check_commit_safety()`, `ensure_single_leader()` |

The Consensus Engine maintains both persistent state that must survive crashes (current term, voted for, log entries) and volatile state that is reconstructed on startup (current role, election timeouts, peer tracking). It coordinates with other components through well-defined interfaces but never directly manipulates storage or network resources - instead, it issues commands to specialized components and reacts to events they generate.

> **Design Insight**: The Consensus Engine is intentionally stateless regarding storage and network operations. This separation allows for easier testing (we can mock storage and network) and ensures that the complex consensus logic can be reasoned about independently of infrastructure concerns.

#### Log Manager

The **Log Manager** handles all aspects of the replicated log, which is the heart of state machine replication. Think of it as the **embassy's records department** - it maintains meticulous records of every decision and proposal, ensures that records are stored safely and can be retrieved even after system failures, and provides fast access to recent records while archiving older ones efficiently.

| Responsibility | Description | Key Operations |
|---|---|---|
| Log Entry Storage | Persists log entries with strong durability guarantees and crash recovery | `append_entries()`, `get_entries()`, `get_last_entry()` |
| Log Consistency | Implements the log matching property and handles conflict resolution | `check_consistency()`, `find_conflict()`, `truncate_conflicting()` |
| Index Management | Maintains log indices, tracks committed entries, and manages log metadata | `get_commit_index()`, `update_commit_index()`, `get_log_length()` |
| Entry Validation | Validates log entries for correctness and enforces ordering constraints | `validate_entry()`, `check_ordering()`, `verify_integrity()` |
| Persistence Interface | Provides crash-safe storage operations with appropriate fsync semantics | `persist_state()`, `recover_state()`, `ensure_durability()` |
| Compaction Coordination | Coordinates with snapshot creation and manages log truncation safely | `create_snapshot_point()`, `truncate_before()`, `get_snapshot_metadata()` |

The Log Manager is responsible for implementing the critical **log matching property**: if two logs contain an entry with the same index and term, then they are identical in all preceding entries. This property is fundamental to Raft's safety guarantees and requires careful coordination between log storage, consistency checking, and conflict resolution.

> **Critical Safety Requirement**: The Log Manager must ensure that once a log entry is acknowledged as written to disk, it will survive any single-node failure. This typically requires fsync operations, but the Log Manager abstracts these details from the Consensus Engine.

#### State Machine Interface

The **State Machine Interface** acts as the bridge between the consensus protocol and the application-specific state machine that clients actually interact with. In our embassy analogy, this is like the **public services counter** - it takes the decisions made by the diplomatic process and translates them into concrete actions that affect the outside world, while also providing a clean interface for citizens (clients) to interact with embassy services.

| Responsibility | Description | Key Operations |
|---|---|---|
| Command Application | Applies committed log entries to the application state machine in order | `apply_command()`, `get_apply_index()`, `ensure_sequential_application()` |
| State Queries | Provides read access to current state machine state with appropriate consistency guarantees | `read_state()`, `get_snapshot()`, `check_consistency()` |
| Snapshot Creation | Creates point-in-time snapshots of state machine state for log compaction | `create_snapshot()`, `get_snapshot_metadata()`, `validate_snapshot()` |
| Snapshot Installation | Restores state machine from snapshot during catch-up or recovery | `install_snapshot()`, `verify_snapshot()`, `clear_state()` |
| Client Interface | Provides external API for client requests with proper consensus integration | `submit_command()`, `query_state()`, `get_leader_info()` |
| Determinism Enforcement | Ensures state machine operations are deterministic and reproducible | `validate_determinism()`, `handle_non_deterministic_ops()`, `ensure_reproducibility()` |

The State Machine Interface must handle the complex timing requirements of consensus systems. Client commands cannot be applied immediately - they must first be replicated to a majority of nodes and committed. Similarly, read operations must be handled carefully to ensure linearizability, potentially requiring interaction with the Consensus Engine to verify leadership status.

#### Network Transport Layer

The **Network Transport Layer** handles all inter-node communication required for the Raft protocol. This is like the **communications department** of our embassy - it manages secure channels to other embassies, handles message delivery and acknowledgment, deals with unreliable communication networks, and provides delivery guarantees that the diplomatic process can rely on.

| Responsibility | Description | Key Operations |
|---|---|---|
| RPC Transport | Implements reliable request-response communication between nodes | `send_request_vote()`, `send_append_entries()`, `send_install_snapshot()` |
| Connection Management | Maintains persistent connections to cluster peers with automatic reconnection | `connect_to_peer()`, `disconnect_peer()`, `get_connection_status()` |
| Message Serialization | Handles encoding/decoding of Raft messages with version compatibility | `serialize_message()`, `deserialize_message()`, `check_version()` |
| Failure Detection | Detects network failures and provides failure notifications to consensus layer | `detect_partition()`, `report_failure()`, `check_peer_health()` |
| Retry and Backoff | Implements intelligent retry logic with exponential backoff for failed requests | `retry_with_backoff()`, `handle_timeout()`, `adjust_retry_policy()` |
| Security Interface | Provides authentication and encryption for inter-node communication | `authenticate_peer()`, `encrypt_message()`, `verify_identity()` |

The Network Transport Layer must handle the **partial failure** nature of distributed systems gracefully. Messages may be lost, delayed, duplicated, or reordered. The transport layer provides **at-least-once delivery** semantics, and the Raft protocol handles deduplication and ordering at the consensus level.

> **Performance Consideration**: The transport layer should support connection pooling and message batching to improve throughput, but these optimizations must not compromise the correctness of the consensus protocol.

#### Storage Subsystem

The **Storage Subsystem** provides persistent, crash-safe storage for all consensus state. This is like the **embassy's vault** - it stores critical documents with multiple layers of protection, ensures that important records survive even catastrophic failures, and provides quick access to frequently needed documents while maintaining long-term archives.

| Responsibility | Description | Key Operations |
|---|---|---|
| Persistent State | Stores Raft persistent state (current term, voted for, log entries) with durability | `save_term()`, `save_vote()`, `persist_log_entry()` |
| Atomic Operations | Provides atomic updates for critical state changes | `atomic_update()`, `begin_transaction()`, `commit_transaction()` |
| Crash Recovery | Recovers consistent state after crashes and validates data integrity | `recover_on_startup()`, `validate_integrity()`, `repair_corruption()` |
| Snapshot Storage | Manages snapshot files with efficient storage and retrieval | `store_snapshot()`, `load_snapshot()`, `list_snapshots()` |
| Performance Optimization | Implements efficient storage patterns for consensus workloads | `batch_writes()`, `optimize_reads()`, `manage_cache()` |
| Storage Abstraction | Provides pluggable storage backends for different deployment environments | `configure_backend()`, `migrate_storage()`, `backup_state()` |

The Storage Subsystem must guarantee that once a write operation returns successfully, the data will survive any single-node failure. This typically requires careful coordination of write operations with fsync calls, but the exact implementation depends on the chosen storage backend (local disk, cloud storage, distributed storage, etc.).

### Component Interaction Patterns

The five core components interact through well-defined patterns that ensure correctness while maintaining modularity. Understanding these interaction patterns is crucial for implementing a robust consensus system.

#### Event-Driven Coordination

Components communicate primarily through **asynchronous events** rather than direct method calls. This pattern prevents deadlocks and allows each component to maintain its own internal consistency while reacting to external events. For example, when the Consensus Engine decides to start an election, it doesn't directly call network methods. Instead, it publishes an "ElectionStarted" event, and the Network Transport Layer subscribes to this event and sends the appropriate RequestVote RPCs.

| Event Type | Publisher | Subscribers | Trigger Conditions |
|---|---|---|---|
| `ElectionStarted` | Consensus Engine | Network Transport | Election timeout expires or leader failure detected |
| `VoteReceived` | Network Transport | Consensus Engine | RequestVote response arrives from peer |
| `LogEntryAppended` | Log Manager | Consensus Engine, State Machine | New entry successfully persisted to log |
| `CommitAdvanced` | Consensus Engine | State Machine, Log Manager | Majority replication confirmed |
| `SnapshotCreated` | State Machine | Log Manager, Storage | Snapshot creation completed |
| `NetworkPartition` | Network Transport | Consensus Engine | Peer connectivity failure detected |

#### Request-Response Flows

For operations that require immediate responses, components use **structured request-response flows** with clear error handling and timeout management. These flows maintain the event-driven architecture while providing synchronous semantics where needed.

| Flow | Initiator | Target | Purpose | Timeout Handling |
|---|---|---|---|---|
| `AppendLogEntry` | Consensus Engine | Log Manager | Persist new log entry | Retry with backoff |
| `QueryState` | Client Interface | State Machine | Read current state | Linearizability check |
| `SendHeartbeat` | Consensus Engine | Network Transport | Maintain leader authority | Mark peer as failed |
| `CreateSnapshot` | Log Manager | State Machine | Trigger snapshot creation | Continue with degraded performance |
| `ValidateEntry` | Log Manager | Consensus Engine | Check entry consistency | Reject inconsistent entry |
| `PersistTerm` | Consensus Engine | Storage | Save term change | Retry until successful |

#### Error Propagation

Errors are propagated through the component hierarchy using **structured error types** that include enough context for proper handling. Components distinguish between **recoverable errors** (temporary network failures, busy resources) and **fatal errors** (data corruption, configuration problems) and handle each category appropriately.

### Recommended Module Structure

The module structure reflects the component architecture while providing clear separation between public interfaces, internal implementation, and external dependencies. This organization supports both development team productivity and system maintainability.

```
titan/
├── cmd/
│   ├── titan-server/           # Server executable
│   │   └── main.py
│   └── titan-client/           # Client tools
│       └── main.py
├── titan/
│   ├── __init__.py
│   ├── consensus/              # Consensus Engine Core
│   │   ├── __init__.py
│   │   ├── engine.py          # Main consensus logic
│   │   ├── election.py        # Election algorithms
│   │   ├── state.py          # Node state management
│   │   └── safety.py         # Safety property enforcement
│   ├── log/                   # Log Manager
│   │   ├── __init__.py
│   │   ├── manager.py         # Log management
│   │   ├── entry.py          # Log entry definitions
│   │   ├── compaction.py     # Snapshot and compaction
│   │   └── persistence.py    # Storage interface
│   ├── statemachine/          # State Machine Interface
│   │   ├── __init__.py
│   │   ├── interface.py       # Abstract state machine
│   │   ├── application.py     # Application logic integration
│   │   └── snapshot.py        # Snapshot management
│   ├── transport/             # Network Transport Layer
│   │   ├── __init__.py
│   │   ├── rpc.py            # RPC implementation
│   │   ├── messages.py       # Message definitions
│   │   ├── connection.py     # Connection management
│   │   └── security.py       # Authentication/encryption
│   ├── storage/               # Storage Subsystem
│   │   ├── __init__.py
│   │   ├── backend.py        # Storage backend interface
│   │   ├── local.py          # Local disk storage
│   │   ├── memory.py         # In-memory storage (testing)
│   │   └── recovery.py       # Crash recovery logic
│   ├── types/                 # Shared type definitions
│   │   ├── __init__.py
│   │   ├── core.py           # Core types (Term, NodeId, etc.)
│   │   ├── messages.py       # RPC message types
│   │   └── errors.py         # Error type hierarchy
│   └── utils/                 # Utility modules
│       ├── __init__.py
│       ├── timing.py          # Timeout and timing utilities
│       ├── config.py          # Configuration management
│       └── logging.py         # Structured logging
├── tests/
│   ├── unit/                  # Unit tests by component
│   ├── integration/           # Integration tests
│   ├── property/              # Property-based tests
│   └── chaos/                 # Chaos engineering tests
└── docs/
    ├── api/                   # API documentation
    └── examples/              # Usage examples
```

#### Module Dependency Rules

To maintain architectural integrity and prevent circular dependencies, modules follow strict dependency rules:

| Module | May Import From | May NOT Import From | Rationale |
|---|---|---|---|
| `consensus/` | `log/`, `transport/`, `storage/`, `types/`, `utils/` | `statemachine/` | Consensus logic should not depend on specific state machine implementations |
| `log/` | `storage/`, `types/`, `utils/` | `consensus/`, `transport/`, `statemachine/` | Log management is a foundational service |
| `statemachine/` | `types/`, `utils/` | `consensus/`, `log/`, `transport/` | State machine should be reusable across different consensus implementations |
| `transport/` | `types/`, `utils/` | `consensus/`, `log/`, `statemachine/` | Transport is infrastructure that serves all components |
| `storage/` | `types/`, `utils/` | All other modules | Storage is the foundational layer |

These dependency rules ensure that components can be tested independently and that the architecture remains modular and extensible.

#### Interface Definitions

Each component exposes its functionality through well-defined interfaces that specify contracts without exposing implementation details:

| Interface | Module | Purpose | Key Methods |
|---|---|---|---|
| `ConsensusEngine` | `consensus/engine.py` | Main consensus protocol implementation | `start()`, `stop()`, `submit_command()`, `handle_rpc()` |
| `LogManager` | `log/manager.py` | Log storage and management | `append_entries()`, `get_entries()`, `create_snapshot()` |
| `StateMachine` | `statemachine/interface.py` | Application state management | `apply_command()`, `create_snapshot()`, `install_snapshot()` |
| `Transport` | `transport/rpc.py` | Inter-node communication | `send_request_vote()`, `send_append_entries()` |
| `StorageBackend` | `storage/backend.py` | Persistent storage operations | `save()`, `load()`, `delete()`, `atomic_update()` |

### Threading and Concurrency Model

Consensus systems must handle multiple concurrent activities: processing client requests, sending heartbeats, handling network messages, and performing background tasks like log compaction. The threading model must ensure safety while providing good performance and avoiding common concurrency pitfalls.

![System Architecture Overview](./diagrams/system-architecture.svg)

#### Single-Writer, Multiple-Reader Pattern

Titan employs a **single-writer, multiple-reader** pattern for critical consensus state. All modifications to core consensus state (current term, node state, log entries) are serialized through a single **consensus thread**, while multiple **worker threads** handle concurrent read operations and background tasks.

| Thread Type | Responsibilities | Access Pattern | Synchronization |
|---|---|---|---|
| Consensus Thread | Processes Raft RPCs, manages elections, coordinates replication | Exclusive write access to consensus state | Owns all state modifications |
| Network Threads | Handle incoming/outgoing network messages | Read-only access to consensus state | Message queues to consensus thread |
| Client Threads | Process client requests and queries | Read-only access through consensus thread | Request queues and response callbacks |
| Background Threads | Log compaction, snapshot creation, metrics collection | Coordinated access through consensus thread | Event-based coordination |

This pattern eliminates most race conditions by ensuring that the complex consensus logic runs in a single thread, while parallelizing the I/O-intensive operations that don't require coordination.

> **Design Rationale**: The single-writer pattern is inspired by successful consensus implementations like etcd and MongoDB. While it may seem to limit parallelism, consensus algorithms are inherently sequential in their core logic, and this pattern dramatically simplifies correctness reasoning.

#### Message Queue Architecture

Components communicate through **lock-free message queues** that provide asynchronous, ordered delivery semantics. Each component maintains input queues for different message types and processes them in priority order.

| Queue Type | Priority | Message Types | Processing Guarantees |
|---|---|---|---|
| High Priority | 1 | Safety-critical RPCs (`RequestVote`, leader `AppendEntries`) | Processed immediately, never dropped |
| Medium Priority | 2 | Client requests, follower `AppendEntries` | Processed promptly, may be throttled under load |
| Low Priority | 3 | Background tasks, metrics, logging | Best-effort processing, may be dropped under pressure |
| Administrative | 0 | Shutdown, configuration changes | Interrupts all other processing |

The consensus thread processes messages in strict priority order, ensuring that safety-critical operations are never delayed by lower-priority work. This queue-based architecture also provides natural **backpressure** - when the consensus thread is overloaded, lower-priority queues fill up, providing feedback to clients and background processes.

#### Lock-Free Data Structures

For frequently accessed read-only data, Titan uses **lock-free data structures** with atomic operations and memory barriers. This allows multiple reader threads to access consensus state without blocking while maintaining strong consistency guarantees.

| Data Structure | Access Pattern | Synchronization Mechanism | Use Case |
|---|---|---|---|
| Current Term | Atomic read/write | `AtomicInteger` with memory barriers | Term validation in RPC handlers |
| Node State | Atomic read/write | `AtomicEnum` with state transitions | Quick state checks |
| Last Log Index | Atomic read/write | `AtomicInteger` with increment operations | Log index management |
| Peer Status | Copy-on-write | Immutable maps with atomic reference updates | Cluster membership tracking |
| Configuration | Copy-on-write | Immutable configuration objects | System configuration access |

These lock-free structures provide **linearizable reads** without blocking writers, crucial for maintaining low latency in the consensus protocol.

#### Error Handling and Recovery

The threading model includes comprehensive error handling that prevents failures in one component from affecting others:

| Error Type | Detection Method | Recovery Action | Thread Impact |
|---|---|---|---|
| Network Failures | Timeout detection, connection errors | Retry with backoff, mark peer as failed | Isolated to network threads |
| Storage Failures | I/O exceptions, corruption detection | Attempt recovery, fall back to read-only mode | Background thread handles recovery |
| Memory Exhaustion | Queue size monitoring, allocation failures | Shed low-priority load, trigger emergency compaction | All threads participate in load shedding |
| Logic Errors | Assertion failures, invariant violations | Log error context, attempt graceful degradation | Consensus thread may restart |
| Deadlock Detection | Thread monitoring, timeout detection | Force thread restart, escalate to process restart | Affected threads only |

#### Architecture Decisions for Concurrency

> **Decision: Single-Writer Consensus Thread**
> - **Context**: Consensus algorithms involve complex state transitions that are difficult to parallelize safely
> - **Options Considered**: 
>   1. Fine-grained locking around individual state variables
>   2. Single-writer thread for all consensus operations
>   3. Actor-model with message passing between stateful actors
> - **Decision**: Single-writer thread with message queues for coordination
> - **Rationale**: Eliminates race conditions in consensus logic, simplifies testing and debugging, and provides predictable performance characteristics
> - **Consequences**: Limits consensus throughput to single-thread performance, but consensus is typically not the bottleneck in distributed systems

> **Decision: Lock-Free Read Access**
> - **Context**: Many operations need to read consensus state without modifying it, and blocking on the consensus thread would hurt performance
> - **Options Considered**:
>   1. Read-write locks around shared state
>   2. Lock-free atomic operations for frequently read data
>   3. Message passing for all read operations
> - **Decision**: Lock-free atomic operations with copy-on-write for complex data
> - **Rationale**: Provides linearizable reads without blocking writers, eliminates lock contention, and scales well with multiple reader threads
> - **Consequences**: Requires careful memory management and increases implementation complexity for complex data structures

#### Common Concurrency Pitfalls

⚠️ **Pitfall: Race Conditions in State Transitions**
The most common concurrency error in consensus implementations is allowing multiple threads to modify consensus state simultaneously. For example, if both a heartbeat timeout and a vote response can trigger state transitions concurrently, the node might enter an invalid state where it thinks it's both a candidate and a follower. **Solution**: Serialize all state modifications through the consensus thread and use atomic compare-and-swap operations for simple state changes.

⚠️ **Pitfall: Deadlock in Cross-Component Calls**
Components that call each other's methods while holding locks can easily create deadlocks. For instance, if the Log Manager calls back into the Consensus Engine while holding a log lock, and the Consensus Engine tries to append to the log while holding a consensus lock, deadlock occurs. **Solution**: Use message queues for all cross-component communication and never make blocking calls while holding locks.

⚠️ **Pitfall: Memory Visibility Issues**
Changes made by the consensus thread may not be immediately visible to other threads due to CPU caching and memory reordering. This can cause reader threads to see stale state even after the consensus thread has updated it. **Solution**: Use proper memory barriers (volatile variables, atomic operations) and ensure that all shared state updates include appropriate synchronization primitives.

⚠️ **Pitfall: Queue Overflow Under Load**
When the system is overloaded, message queues can fill up faster than they can be processed, leading to memory exhaustion or dropped messages. **Solution**: Implement proper backpressure mechanisms, prioritize safety-critical messages, and have emergency load-shedding procedures that maintain system safety while reducing throughput.

### Implementation Guidance

This section provides concrete implementation guidance for building the Titan architecture using Python's asyncio framework, which provides excellent support for the event-driven patterns required by consensus systems.

#### Technology Recommendations

| Component | Simple Option | Advanced Option | Production Considerations |
|---|---|---|---|
| Transport | HTTP REST with `aiohttp` | gRPC with `grpcio-async` | gRPC provides better performance and type safety |
| Serialization | JSON with `json` module | Protocol Buffers with `protobuf` | Protobuf handles versioning and schema evolution |
| Storage | File-based with `aiofiles` | SQLite with `aiosqlite` | SQLite provides ACID transactions and crash recovery |
| Logging | Python `logging` module | Structured logging with `structlog` | Structured logs are essential for debugging distributed systems |
| Configuration | YAML with `pyyaml` | Consul/etcd integration | External configuration supports dynamic reconfiguration |
| Monitoring | Simple metrics with `prometheus_client` | OpenTelemetry integration | Full observability is crucial for production consensus systems |

#### Recommended File Structure

```
titan/
├── titan/
│   ├── __init__.py
│   ├── node.py                 # Main node implementation
│   ├── consensus/
│   │   ├── __init__.py
│   │   ├── engine.py           # Consensus engine implementation
│   │   ├── election.py         # Election logic
│   │   └── safety.py           # Safety property enforcement
│   ├── log/
│   │   ├── __init__.py
│   │   ├── manager.py          # Log manager implementation
│   │   └── entry.py            # Log entry types
│   ├── transport/
│   │   ├── __init__.py
│   │   ├── rpc.py             # RPC transport layer
│   │   └── messages.py         # Message type definitions
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── backend.py          # Storage backend interface
│   │   └── file_storage.py     # File-based storage implementation
│   ├── statemachine/
│   │   ├── __init__.py
│   │   ├── interface.py        # State machine interface
│   │   └── kv_store.py         # Example key-value store implementation
│   └── types/
│       ├── __init__.py
│       ├── core.py             # Core type definitions
│       └── messages.py         # RPC message types
├── tests/
├── examples/
└── setup.py
```

#### Core Type Definitions

```python
# titan/types/core.py
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Dict, Any
import time

# Core identifier types
NodeId = str
Term = int
LogIndex = int

class NodeState(Enum):
    """Raft node states with clear transitions"""
    FOLLOWER = auto()
    CANDIDATE = auto() 
    LEADER = auto()

@dataclass
class LogEntry:
    """Individual entry in the replicated log"""
    term: int
    index: int  
    data: bytes
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize entry for persistence"""
        # TODO: Implement serialization to dictionary
        pass
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        """Deserialize entry from storage"""
        # TODO: Implement deserialization from dictionary  
        pass

# Constants for timing and cluster limits
ELECTION_TIMEOUT_MIN = 0.15  # 150ms minimum election timeout
ELECTION_TIMEOUT_MAX = 0.30  # 300ms maximum election timeout  
HEARTBEAT_INTERVAL = 0.05    # 50ms leader heartbeat interval
MAX_CLUSTER_SIZE = 9         # 9 nodes maximum for performance
SNAPSHOT_THRESHOLD = 10000   # 10000 entries trigger snapshot
```

#### Message Type Definitions

```python
# titan/types/messages.py
from dataclasses import dataclass
from typing import List, Optional
from .core import Term, NodeId, LogIndex, LogEntry

@dataclass
class RequestVoteRequest:
    """Vote request sent by candidates during election"""
    term: int
    candidate_id: str  
    last_log_index: int
    last_log_term: int

@dataclass  
class RequestVoteResponse:
    """Vote response from followers to candidates"""
    term: int
    vote_granted: bool

@dataclass
class AppendEntriesRequest:
    """Log replication request sent by leader"""
    term: int
    leader_id: str
    prev_log_index: int
    prev_log_term: int
    entries: List[LogEntry]
    leader_commit: int

@dataclass
class AppendEntriesResponse:
    """Response to log replication request"""
    term: int
    success: bool
    match_index: Optional[int] = None
    conflict_index: Optional[int] = None
```

#### Consensus Engine Skeleton

```python
# titan/consensus/engine.py
import asyncio
import random
from typing import Dict, Optional, Set
from ..types.core import NodeState, Term, NodeId, ELECTION_TIMEOUT_MIN, ELECTION_TIMEOUT_MAX
from ..types.messages import RequestVoteRequest, RequestVoteResponse

class ConsensusEngine:
    """Core Raft consensus engine implementation"""
    
    def __init__(self, node_id: NodeId, cluster_nodes: Set[NodeId]):
        self.node_id = node_id
        self.cluster_nodes = cluster_nodes
        
        # Persistent state (must survive crashes)
        self.current_term: Term = 0
        self.voted_for: Optional[NodeId] = None
        
        # Volatile state  
        self.state = NodeState.FOLLOWER
        self.current_leader: Optional[NodeId] = None
        self.election_timer: Optional[asyncio.Task] = None
        
        # Leader state
        self.votes_received: Set[NodeId] = set()
        
    async def start(self) -> None:
        """Initialize and start the consensus engine"""
        # TODO 1: Recover persistent state from storage
        # TODO 2: Initialize log manager and transport layer  
        # TODO 3: Start election timer as follower
        # TODO 4: Begin processing message queues
        pass
        
    async def handle_request_vote(self, request: RequestVoteRequest) -> RequestVoteResponse:
        """Process vote request from candidate"""
        # TODO 1: Check if request term is greater than current term
        # TODO 2: If so, update current term and become follower
        # TODO 3: Check if we can vote for this candidate:
        #   - Haven't voted in this term, OR already voted for this candidate
        #   - Candidate's log is at least as up-to-date as ours
        # TODO 4: If voting yes, persist vote and reset election timer
        # TODO 5: Return vote response with current term and decision
        pass
        
    def _generate_election_timeout(self) -> float:
        """Generate randomized election timeout to prevent split votes"""
        # TODO: Return random timeout between ELECTION_TIMEOUT_MIN and ELECTION_TIMEOUT_MAX
        pass
        
    async def _start_election(self) -> None:
        """Initiate leader election as candidate"""
        # TODO 1: Increment current term
        # TODO 2: Transition to candidate state  
        # TODO 3: Vote for self and persist the vote
        # TODO 4: Reset election timer
        # TODO 5: Send RequestVote RPCs to all other nodes
        # TODO 6: Collect votes and become leader if majority achieved
        pass
```

#### Storage Interface Implementation  

```python
# titan/storage/file_storage.py
import os
import json
import fcntl
from typing import Dict, Any, Optional
import aiofiles

class FileStorage:
    """Simple file-based storage with crash safety"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.state_file = os.path.join(data_dir, "raft_state.json")
        os.makedirs(data_dir, exist_ok=True)
        
    async def save_state(self, state: Dict[str, Any]) -> None:
        """Atomically save Raft state to disk"""
        # TODO 1: Write state to temporary file
        # TODO 2: Fsync temporary file to ensure durability  
        # TODO 3: Atomically rename temporary file to final location
        # TODO 4: Handle any I/O errors appropriately
        pass
        
    async def load_state(self) -> Dict[str, Any]:
        """Load Raft state from disk, return empty dict if not found"""
        # TODO 1: Check if state file exists
        # TODO 2: If exists, read and parse JSON content
        # TODO 3: Validate state structure and handle corruption
        # TODO 4: Return empty state dict if file doesn't exist
        pass
        
    async def append_log_entry(self, entry_data: Dict[str, Any]) -> None:
        """Append log entry with crash safety"""
        # TODO 1: Serialize entry to JSON line
        # TODO 2: Append to log file with exclusive access
        # TODO 3: Fsync log file to ensure persistence
        # TODO 4: Handle disk full and other I/O errors
        pass
```

#### Milestone Checkpoints

**Checkpoint 1 - Basic Architecture (After High-Level Architecture)**
- **Verify**: All core modules import without errors
- **Test Command**: `python -c "from titan.consensus.engine import ConsensusEngine; print('Architecture OK')"`
- **Expected Behavior**: Clean import with no import errors or circular dependencies
- **Manual Verification**: Create a ConsensusEngine instance and verify it initializes with correct state

**Checkpoint 2 - Component Integration (After implementing component interfaces)**
- **Verify**: Components can be instantiated and communicate via message queues
- **Test Command**: Run unit tests with `python -m pytest tests/unit/test_architecture.py`
- **Expected Behavior**: All components start successfully and exchange basic messages
- **Manual Verification**: Start a single node and verify it transitions to candidate state after election timeout

**Checkpoint 3 - Threading Model (After implementing concurrency patterns)**  
- **Verify**: Multiple threads handle messages without race conditions
- **Test Command**: Run stress tests with `python -m pytest tests/integration/test_concurrency.py`
- **Expected Behavior**: No deadlocks or race conditions under concurrent load
- **Manual Verification**: Send concurrent client requests while triggering elections

#### Language-Specific Implementation Hints

**AsyncIO Patterns for Consensus**:
- Use `asyncio.Queue` for message passing between components
- Use `asyncio.Event` for coordination between background tasks
- Use `asyncio.create_task()` for concurrent operations that need to be awaited
- Use `asyncio.gather()` when waiting for multiple operations to complete

**Error Handling Best Practices**:
- Create custom exception hierarchy for different failure types
- Use `try/except/finally` with specific exception types, never catch `Exception`
- Log error context including node state, current term, and operation being performed
- Implement circuit breaker patterns for network operations

**Performance Optimization**:  
- Use `asyncio.Semaphore` to limit concurrent operations
- Batch multiple log entries into single disk writes when possible
- Use connection pooling for network operations  
- Implement backpressure by monitoring queue sizes

#### Debugging Tips for Architecture Issues

| Symptom | Likely Cause | How to Diagnose | Fix |
|---|---|---|---|
| Import errors or circular dependencies | Incorrect module dependency structure | Check import statements and dependency graph | Refactor to follow dependency rules |
| Messages not being processed | Queue overflow or thread deadlock | Monitor queue sizes and thread status | Implement proper backpressure and timeout handling |
| Inconsistent state across components | Race conditions in shared state access | Add logging to state transitions and use thread-safe primitives | Serialize state modifications through single thread |
| High memory usage | Message queues growing without bounds | Monitor queue sizes and processing rates | Implement queue limits and load shedding |
| Performance degradation | Excessive locking or blocking operations | Profile thread activity and lock contention | Replace locks with lock-free data structures where possible |


## Data Model and Core Types

> **Milestone(s):** All milestones (1-4) - defines the fundamental data structures used throughout election, replication, compaction, and membership changes

### Mental Model: The Government Record System

Think of the Raft data model like a government's record-keeping system. Each government office (node) maintains identical copies of three types of records: **identity documents** (persistent state like current term and vote), **official journals** (the replicated log of decisions), and **correspondence** (RPC messages between offices). Just as government offices must maintain consistent records and communicate through formal channels to prevent conflicting decisions, Raft nodes use structured data types to ensure consensus safety.

The beauty of this analogy extends to state management: persistent state is like official documents stored in fireproof safes that survive building fires (crashes), while volatile state is like working notes that get recreated when officials return to work. Messages between offices follow strict protocols - just as you can't submit a marriage license without proper signatures and seals, Raft RPCs contain specific fields that enable recipients to verify authenticity and maintain consistency.

This structured approach to data modeling is what makes consensus algorithms work reliably. Without careful attention to what data persists, what gets transmitted, and how state evolves, distributed systems quickly fall into split-brain scenarios or data loss. Every field in every data structure serves a specific purpose in the broader safety and liveness guarantees.

### Raft Node State

The Raft node state is divided into two categories: **persistent state** that must survive crashes and **volatile state** that can be reconstructed on restart. This division is critical for both performance and correctness - persistent state requires expensive disk writes with fsync guarantees, while volatile state enables fast in-memory operations.

#### Persistent State

Persistent state forms the foundation of Raft's safety guarantees. These fields must be written to stable storage and fsynced before responding to RPCs, ensuring that a node's commitments survive crashes. The persistent state acts like a node's "sworn oath" - once written, the node is bound by these values even after restart.

| Field | Type | Description |
|-------|------|-------------|
| `current_term` | `Term` | Latest term server has seen, monotonically increasing |
| `voted_for` | `NodeId` or `None` | Candidate that received vote in current term, or None if no vote cast |
| `log` | `List[LogEntry]` | Complete log of entries; index starts at 1 |

The `current_term` serves as Raft's logical clock, ensuring that nodes can distinguish between "old" and "new" leadership epochs. Every time a node starts an election, it increments its term, creating a globally unique timestamp for that leadership attempt. Terms never decrease and provide the fundamental ordering mechanism for preventing split-brain scenarios.

The `voted_for` field implements Raft's "one vote per term" safety property. Once a node votes for a candidate in a given term, it cannot vote for anyone else in that term, even after a crash and restart. This prevents election cycles where the same node votes for multiple candidates, which could lead to multiple leaders in a single term.

The `log` contains the complete sequence of state machine operations that define the system's history. Unlike the term and vote which are single values, the log is an append-only sequence that grows continuously during normal operation. Each entry contains not just the operation data, but also metadata that enables consistency checking across nodes.

#### Volatile State

Volatile state represents a node's current understanding of cluster state and can be safely reconstructed after crashes. This state changes frequently during normal operation and doesn't require expensive persistence operations.

| Field | Type | Description |
|-------|------|-------------|
| `node_state` | `NodeState` | Current role: FOLLOWER, CANDIDATE, or LEADER |
| `commit_index` | `LogIndex` | Index of highest log entry known to be committed |
| `last_applied` | `LogIndex` | Index of highest log entry applied to state machine |
| `leader_id` | `NodeId` or `None` | Current leader's identifier, None if unknown |
| `election_deadline` | `float` | Timestamp when current election timeout expires |

The `node_state` determines how the node behaves in response to RPCs and timeouts. This is the primary state in Raft's state machine and drives all decision-making logic. Unlike persistent state, a crashed node always restarts as a FOLLOWER, regardless of its pre-crash state.

The `commit_index` and `last_applied` form a crucial pair that manages the pipeline between consensus and state machine application. The `commit_index` represents the highest log entry that has been replicated to a majority of servers (and thus is safe to apply), while `last_applied` tracks how much of that committed log has actually been processed by the local state machine. The invariant `last_applied ≤ commit_index ≤ last_log_index` must always hold.

#### Volatile State on Leaders

Leaders maintain additional volatile state to coordinate replication across followers. This state is recreated and initialized after each leader election, reflecting the leader's responsibility to actively manage cluster state.

| Field | Type | Description |
|-------|------|-------------|
| `next_index` | `Dict[NodeId, LogIndex]` | For each server, index of next log entry to send |
| `match_index` | `Dict[NodeId, LogIndex]` | For each server, index of highest log entry known to be replicated |

The `next_index` array represents the leader's best guess about what log entries each follower needs. Initially set optimistically to the leader's last log index + 1, this value gets decremented when AppendEntries RPCs fail due to log inconsistencies. This implements Raft's "probe and backtrack" approach to finding the point where leader and follower logs match.

The `match_index` array tracks confirmed replication progress for each follower. Unlike `next_index` (which is speculative), `match_index` values are only updated when followers explicitly acknowledge successful replication. The leader uses these values to determine when log entries have been replicated to a majority and can be safely committed.

> **Critical Insight**: The separation between `next_index` and `match_index` enables efficient handling of followers that are far behind. The leader can optimistically send large batches of entries (based on `next_index`) while maintaining accurate tracking of confirmed progress (via `match_index`) for commitment decisions.

### Log Entry Format

The log entry format balances simplicity with the metadata requirements for consistency checking and debugging. Each entry represents a single operation submitted by a client, along with the context needed to ensure proper ordering and replication.

#### Core Log Entry Structure

| Field | Type | Description |
|-------|------|-------------|
| `term` | `Term` | Term when entry was created by leader |
| `index` | `LogIndex` | Position in log, monotonically increasing starting from 1 |
| `data` | `bytes` | Application-specific command or operation payload |
| `timestamp` | `float` | Wall-clock time when entry was created (for debugging/monitoring) |

The `term` field enables the **log matching property** - Raft's guarantee that if two logs contain an entry with the same index and term, then the logs are identical in all preceding entries. This property allows efficient consistency checking by comparing just the term and index of the last entry, rather than checksumming entire log prefixes.

The `index` provides total ordering of operations within the log. Unlike terms (which can have gaps when elections fail), log indices are consecutive integers starting from 1. The combination of term and index uniquely identifies every log entry in the cluster's history, enabling precise conflict detection and resolution.

The `data` field contains the opaque application payload - the actual state machine command submitted by clients. Raft treats this as an arbitrary byte array, maintaining strict separation between consensus (Raft's responsibility) and command interpretation (the application's responsibility). This design enables Raft to provide consensus for any type of state machine, from key-value stores to SQL databases.

The `timestamp` serves operational rather than consensus purposes. It enables debugging timeline reconstruction, performance analysis, and log archival decisions. Importantly, consensus decisions never depend on timestamps to avoid clock synchronization requirements across nodes.

#### Log Metadata and Indexing

Beyond individual entries, the log maintains several derived properties that enable efficient operations:

| Property | Type | Description |
|----------|------|-------------|
| `last_log_index` | `LogIndex` | Index of the most recent log entry |
| `last_log_term` | `Term` | Term of the most recent log entry |
| `log_size` | `int` | Total number of entries in the log |
| `storage_size` | `int` | Approximate storage space consumed by log |

These properties support critical Raft operations without requiring full log scans. For example, election safety depends on comparing `last_log_index` and `last_log_term` values - a candidate can only win if its log is "at least as up-to-date" as any majority of voters. Computing this efficiently requires O(1) access to the last log entry's metadata.

#### Serialization and Storage Format

Log entries must be serialized for persistent storage with careful attention to cross-platform compatibility and forward evolution. The serialization format affects both correctness (ensuring identical data after crash recovery) and performance (minimizing disk I/O overhead).

| Serialization Aspect | Requirement | Rationale |
|----------------------|-------------|-----------|
| Byte Order | Little-endian for integers | Consistent across common architectures |
| Length Encoding | Variable-length prefixes | Efficient storage for different data sizes |
| Checksum | CRC32 per entry | Detect storage corruption early |
| Version Header | Magic bytes + format version | Enable format evolution |

> **Decision: Binary vs. JSON Serialization**
> - **Context**: Log entries are written frequently and must be read during recovery
> - **Options Considered**: JSON (human-readable, slow), Protocol Buffers (efficient, complex), Custom Binary (fast, simple)
> - **Decision**: Custom binary format with length prefixes and checksums
> - **Rationale**: Minimizes disk I/O overhead while maintaining simplicity and corruption detection
> - **Consequences**: Requires careful implementation but delivers optimal performance for consensus-critical path

### RPC Message Formats

Raft's RPC messages form the vocabulary through which nodes coordinate consensus. Each message type serves a specific purpose in the consensus protocol and contains exactly the information needed for recipients to make safety-preserving decisions.

#### RequestVote RPC

The RequestVote RPC implements Raft's leader election mechanism. Candidates send these messages to all other nodes when starting an election, and the responses determine whether the candidate can become leader.

**RequestVoteRequest Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `term` | `Term` | Candidate's current term |
| `candidate_id` | `NodeId` | Candidate requesting vote |
| `last_log_index` | `LogIndex` | Index of candidate's last log entry |
| `last_log_term` | `Term` | Term of candidate's last log entry |

The `term` field enables recipients to detect outdated election attempts and update their own term if they've fallen behind. The candidate's term must be greater than the recipient's current term for the vote to be considered - this prevents "time travel" where old election messages disrupt newer leaders.

The `candidate_id` identifies which node is requesting the vote, enabling the recipient to record its vote choice persistently. This supports Raft's "one vote per term" safety property by allowing nodes to reject duplicate vote requests from different candidates in the same term.

The `last_log_index` and `last_log_term` enable recipients to verify that the candidate's log is sufficiently up-to-date to serve as leader. A candidate's log is considered up-to-date if either: (1) its last entry has a higher term than the voter's, or (2) the terms are equal but the candidate's log is at least as long. This prevents elections of candidates with incomplete logs, preserving committed entries.

**RequestVoteResponse Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `term` | `Term` | Current term of the responding node |
| `vote_granted` | `bool` | True if candidate received vote |

The response `term` enables candidates to detect if they've become outdated during the election process. If the response term is higher than the candidate's term, the candidate must immediately revert to follower state and abandon its election attempt.

The `vote_granted` field communicates the voting decision based on the recipient's evaluation of term freshness, vote availability, and log up-to-date-ness. A `True` value represents a binding commitment - the voter promises not to vote for any other candidate in this term.

#### AppendEntries RPC

The AppendEntries RPC serves dual purposes: heartbeat messages to maintain leader authority and actual log replication to synchronize state across followers. The same message structure handles both cases, with empty entries indicating heartbeats.

**AppendEntriesRequest Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `term` | `Term` | Leader's current term |
| `leader_id` | `NodeId` | Leader identifier for follower redirection |
| `prev_log_index` | `LogIndex` | Index of log entry immediately preceding new entries |
| `prev_log_term` | `Term` | Term of prev_log_index entry |
| `entries` | `List[LogEntry]` | Log entries to store (empty for heartbeat) |
| `leader_commit` | `LogIndex` | Leader's commit_index for follower advancement |

The `term` and `leader_id` establish the sender's authority and help followers identify the current leader for client redirection. Followers reject AppendEntries from nodes with outdated terms, preventing old leaders from disrupting newer leadership.

The `prev_log_index` and `prev_log_term` implement Raft's consistency check mechanism. Before appending new entries, followers verify that their log contains an entry at `prev_log_index` with term `prev_log_term`. This ensures that new entries are only appended when the logs are consistent up to that point, maintaining the log matching property.

The `entries` array contains zero or more log entries to append. Empty arrays indicate heartbeat messages that maintain leader authority without adding new entries. Non-empty arrays represent actual replication attempts that extend the follower's log.

The `leader_commit` field enables followers to advance their commit index safely. Followers set their commit index to `min(leader_commit, index_of_last_new_entry)`, ensuring they don't commit entries they haven't received while staying current with cluster-wide commit progress.

**AppendEntriesResponse Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `term` | `Term` | Current term of the responding follower |
| `success` | `bool` | True if follower accepted the AppendEntries |
| `match_index` | `LogIndex` | Highest index known to match leader's log |
| `conflict_index` | `LogIndex` | Suggested index for leader's next attempt (if success=False) |
| `conflict_term` | `Term` | Term of conflicting entry (if success=False) |

The response `term` enables leaders to detect when they've become outdated, similar to RequestVote responses. Leaders must step down immediately if they receive responses with higher terms.

The `success` field indicates whether the consistency check passed and entries were appended. `True` means the follower's log matched at `prev_log_index` and all provided entries were successfully appended.

The `match_index` field helps leaders track replication progress efficiently. When `success=True`, this indicates the highest index that the leader can confidently mark as replicated to this follower.

The `conflict_index` and `conflict_term` enable fast log backtracking when consistency checks fail. Instead of decrementing `next_index` by one and retrying, leaders can jump back to more promising positions based on the follower's conflicting entry information.

#### InstallSnapshot RPC

The InstallSnapshot RPC handles the case where a follower has fallen so far behind that the leader has already compacted away the log entries needed to bring the follower up-to-date. This RPC transfers a point-in-time snapshot of the state machine directly.

**InstallSnapshotRequest Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `term` | `Term` | Leader's current term |
| `leader_id` | `NodeId` | Leader identifier |
| `last_included_index` | `LogIndex` | Index of last entry included in snapshot |
| `last_included_term` | `Term` | Term of last entry included in snapshot |
| `offset` | `int` | Byte offset where chunk fits into snapshot file |
| `data` | `bytes` | Raw bytes of snapshot chunk starting at offset |
| `done` | `bool` | True if this is the final chunk |

The `term` and `leader_id` serve the same authority establishment purpose as in AppendEntries, ensuring that only legitimate leaders can install snapshots.

The `last_included_index` and `last_included_term` define the snapshot's position in the log timeline. After installing the snapshot, the follower's log will begin at `last_included_index + 1`, with all earlier entries replaced by the snapshot state.

The `offset`, `data`, and `done` fields implement chunked transfer of potentially large snapshots. Large state machines might generate snapshots that exceed reasonable RPC message sizes, so the snapshot is transferred in sequential chunks. The `offset` field ensures chunks are applied in the correct order, while `done=True` signals completion.

**InstallSnapshotResponse Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `term` | `Term` | Current term of the responding follower |

The minimal response reflects the fire-and-forget nature of snapshot installation. Unlike log replication, which requires precise progress tracking, snapshot installation either succeeds completely or fails completely. The leader will retry the entire snapshot on any failure.

> **Architecture Decision: Chunked vs. Streaming Snapshots**
> - **Context**: State machines can grow large, making snapshots exceed practical RPC limits
> - **Options Considered**: Single large RPC (simple, size-limited), Chunked RPCs (complex, scalable), Separate streaming protocol (optimal, very complex)
> - **Decision**: Chunked RPCs with offset-based ordering
> - **Rationale**: Balances implementation complexity with scalability needs while reusing existing RPC infrastructure
> - **Consequences**: Enables large state machines but requires careful chunk ordering and retry logic

### Implementation Guidance

The data model forms the foundation of every Raft operation, so getting these structures right is critical for both correctness and performance. The implementation should prioritize clarity and type safety while optimizing for the consensus protocol's access patterns.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Serialization | JSON with dataclasses | Protocol Buffers with generated code |
| Storage | SQLite with WAL mode | Custom binary format with mmap |
| Validation | Manual field checking | Pydantic models with automatic validation |
| Type Safety | Type hints with mypy | Strict dataclasses with frozen=True |

#### Recommended File Structure

```
titan/
├── core/
│   ├── __init__.py
│   ├── types.py              ← Core type definitions (this section)
│   ├── state.py              ← Node state management
│   ├── log.py                ← Log entry storage and indexing
│   └── messages.py           ← RPC message definitions
├── storage/
│   ├── __init__.py
│   ├── persistent.py         ← Persistent state storage
│   └── log_storage.py        ← Log entry persistence
├── network/
│   ├── __init__.py
│   ├── rpc.py                ← RPC transport layer
│   └── serialization.py     ← Message serialization
└── tests/
    ├── test_types.py         ← Type validation tests
    ├── test_serialization.py ← Persistence tests
    └── fixtures/             ← Test data and mocks
```

#### Infrastructure: Core Type Definitions

```python
# titan/core/types.py
"""
Core Raft data types and enumerations.
Provides the fundamental building blocks for all consensus operations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
import time
import json
from pathlib import Path

# Basic Raft types
NodeId = str
Term = int
LogIndex = int

class NodeState(Enum):
    """Raft node states in the consensus state machine."""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"

# Constants for timeouts and limits
ELECTION_TIMEOUT_MIN = 150  # milliseconds
ELECTION_TIMEOUT_MAX = 300  # milliseconds
HEARTBEAT_INTERVAL = 50     # milliseconds
MAX_CLUSTER_SIZE = 9        # nodes
SNAPSHOT_THRESHOLD = 10000  # entries

@dataclass
class LogEntry:
    """
    A single entry in the replicated log.
    Contains both the application command and metadata for consensus.
    """
    term: Term
    index: LogIndex
    data: bytes
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize entry for persistence."""
        return {
            'term': self.term,
            'index': self.index,
            'data': self.data.hex(),  # Convert bytes to hex string
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        """Deserialize entry from storage."""
        return cls(
            term=data['term'],
            index=data['index'],
            data=bytes.fromhex(data['data']),
            timestamp=data['timestamp']
        )

@dataclass
class PersistentState:
    """
    Raft state that must survive crashes.
    Written to disk before responding to RPCs.
    """
    current_term: Term = 0
    voted_for: Optional[NodeId] = None
    log: List[LogEntry] = field(default_factory=list)
    
    def save_to_file(self, filepath: Path) -> None:
        """Persist state to disk with fsync."""
        data = {
            'current_term': self.current_term,
            'voted_for': self.voted_for,
            'log': [entry.to_dict() for entry in self.log]
        }
        # Write atomically: temp file -> fsync -> rename
        temp_path = filepath.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
            f.flush()
            import os
            os.fsync(f.fileno())
        temp_path.rename(filepath)
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> 'PersistentState':
        """Load state from disk, return empty state if file doesn't exist."""
        if not filepath.exists():
            return cls()
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls(
            current_term=data['current_term'],
            voted_for=data['voted_for'],
            log=[LogEntry.from_dict(entry) for entry in data['log']]
        )
```

#### Infrastructure: RPC Message Types

```python
# titan/core/messages.py
"""
Raft RPC message definitions.
Defines the wire format for all inter-node communication.
"""

from dataclasses import dataclass
from typing import List, Optional
from .types import Term, NodeId, LogIndex, LogEntry

@dataclass
class RequestVoteRequest:
    """Candidate's request for votes during leader election."""
    term: Term
    candidate_id: NodeId
    last_log_index: LogIndex
    last_log_term: Term

@dataclass
class RequestVoteResponse:
    """Voter's response to election request."""
    term: Term
    vote_granted: bool

@dataclass
class AppendEntriesRequest:
    """Leader's log replication and heartbeat message."""
    term: Term
    leader_id: NodeId
    prev_log_index: LogIndex
    prev_log_term: Term
    entries: List[LogEntry]
    leader_commit: LogIndex

@dataclass
class AppendEntriesResponse:
    """Follower's response to log replication."""
    term: Term
    success: bool
    match_index: LogIndex = 0
    conflict_index: LogIndex = 0
    conflict_term: Term = 0

@dataclass
class InstallSnapshotRequest:
    """Leader's snapshot transfer for catching up followers."""
    term: Term
    leader_id: NodeId
    last_included_index: LogIndex
    last_included_term: Term
    offset: int
    data: bytes
    done: bool

@dataclass
class InstallSnapshotResponse:
    """Follower's acknowledgment of snapshot chunk."""
    term: Term

# Message serialization utilities
import json
from typing import Dict, Any, Type, TypeVar

T = TypeVar('T')

def serialize_message(msg: Any) -> bytes:
    """Convert message to JSON bytes for network transmission."""
    # Convert dataclass to dict, handling LogEntry objects specially
    def to_serializable(obj):
        if hasattr(obj, '__dict__'):
            result = obj.__dict__.copy()
            if hasattr(obj, 'entries') and obj.entries:
                result['entries'] = [entry.to_dict() for entry in obj.entries]
            return result
        return obj
    
    data = to_serializable(msg)
    return json.dumps(data).encode('utf-8')

def deserialize_message(data: bytes, msg_type: Type[T]) -> T:
    """Convert JSON bytes back to message object."""
    parsed = json.loads(data.decode('utf-8'))
    
    # Handle LogEntry reconstruction for AppendEntries
    if 'entries' in parsed and parsed['entries']:
        parsed['entries'] = [LogEntry.from_dict(e) for e in parsed['entries']]
    
    return msg_type(**parsed)
```

#### Core Logic: Node State Management

```python
# titan/core/state.py
"""
Raft node state management.
Handles persistent and volatile state with proper update semantics.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import time
import random
from .types import (
    NodeId, Term, LogIndex, NodeState, LogEntry,
    ELECTION_TIMEOUT_MIN, ELECTION_TIMEOUT_MAX, HEARTBEAT_INTERVAL
)

@dataclass
class VolatileState:
    """State that can be reconstructed after crashes."""
    node_state: NodeState = NodeState.FOLLOWER
    commit_index: LogIndex = 0
    last_applied: LogIndex = 0
    leader_id: Optional[NodeId] = None
    election_deadline: float = 0.0
    
    # Leader-only state (reset on leader election)
    next_index: Dict[NodeId, LogIndex] = field(default_factory=dict)
    match_index: Dict[NodeId, LogIndex] = field(default_factory=dict)

class RaftNodeState:
    """
    Complete Raft node state management.
    Coordinates persistent and volatile state with proper safety guarantees.
    """
    
    def __init__(self, node_id: NodeId, cluster_nodes: List[NodeId], 
                 state_file: Path):
        self.node_id = node_id
        self.cluster_nodes = cluster_nodes
        self.state_file = state_file
        
        # Load persistent state from disk
        self.persistent = PersistentState.load_from_file(state_file)
        self.volatile = VolatileState()
        
        # Initialize election timeout
        self._reset_election_timeout()
    
    def update_term(self, term: Term, voted_for: Optional[NodeId] = None) -> None:
        """
        Update persistent term and vote state.
        Must be called before responding to any RPC.
        """
        # TODO 1: Check if new term is greater than current term
        # TODO 2: Update persistent state: current_term and voted_for
        # TODO 3: Reset volatile state to FOLLOWER (new terms start as followers)
        # TODO 4: Save persistent state to disk with fsync
        # TODO 5: Reset election timeout for new term
        pass
    
    def start_election(self) -> None:
        """
        Transition to CANDIDATE state and begin election.
        Increments term and votes for self.
        """
        # TODO 1: Increment current_term
        # TODO 2: Vote for self (set voted_for to self.node_id)
        # TODO 3: Change state to CANDIDATE
        # TODO 4: Reset election timeout
        # TODO 5: Save persistent state (term and vote are now committed)
        # TODO 6: Initialize vote counting (start with 1 vote for self)
        pass
    
    def become_leader(self) -> None:
        """
        Transition to LEADER state after winning election.
        Initialize leader-specific volatile state.
        """
        # TODO 1: Change node_state to LEADER
        # TODO 2: Set leader_id to self.node_id
        # TODO 3: Initialize next_index for all followers (last_log_index + 1)
        # TODO 4: Initialize match_index for all followers (0)
        # TODO 5: Clear election timeout (leaders don't use election timeouts)
        pass
    
    def _reset_election_timeout(self) -> None:
        """Set random election timeout to prevent split votes."""
        timeout_ms = random.randint(ELECTION_TIMEOUT_MIN, ELECTION_TIMEOUT_MAX)
        self.volatile.election_deadline = time.time() + (timeout_ms / 1000.0)
    
    def is_election_timeout(self) -> bool:
        """Check if election timeout has expired."""
        return time.time() > self.volatile.election_deadline
    
    @property
    def last_log_index(self) -> LogIndex:
        """Get index of last log entry (0 if log is empty)."""
        return len(self.persistent.log)
    
    @property
    def last_log_term(self) -> Term:
        """Get term of last log entry (0 if log is empty)."""
        if not self.persistent.log:
            return 0
        return self.persistent.log[-1].term
    
    def append_entries(self, entries: List[LogEntry]) -> None:
        """
        Append new entries to log.
        Does NOT commit entries - that's handled separately.
        """
        # TODO 1: Validate that entries have consecutive indices
        # TODO 2: Set correct term for each entry (current_term)
        # TODO 3: Append entries to persistent.log
        # TODO 4: Save persistent state to disk
        pass
```

#### Milestone Checkpoint

After implementing the data model types:

1. **Type Validation**: Run `python -m pytest tests/test_types.py -v` - should pass all serialization and validation tests
2. **State Persistence**: Verify that `PersistentState.save_to_file()` and `load_from_file()` round-trip correctly
3. **Message Serialization**: Test that RPC messages serialize/deserialize without data loss
4. **Expected Behavior**: Create a `LogEntry`, serialize it, restart your process, deserialize it - all fields should be identical

Signs that something is wrong:
- **Symptom**: Type errors during serialization → **Check**: Ensure all `LogEntry.data` fields are `bytes`, not strings
- **Symptom**: State doesn't persist after restart → **Check**: Verify `fsync()` is called and no exceptions during save
- **Symptom**: Log indices are inconsistent → **Check**: Ensure log indices start at 1 and increment consecutively


## Election and Safety (Milestone 1)

> **Milestone(s):** Milestone 1: Election & Safety - Core Raft leader election and heartbeat mechanism

### Mental Model: Political Elections

Think of Raft elections like a parliamentary democracy with strict constitutional rules designed to prevent chaos. In a healthy democracy, there's always exactly one prime minister at any given time, and leadership transitions follow predictable rules that prevent power vacuums or contested leadership.

In Raft, each **term** is like a political term of office - it's a monotonically increasing number that represents a period of potential leadership. Just as political terms ensure that newer mandates supersede older ones, Raft terms provide a global ordering that prevents nodes from acting on stale information. When a node discovers a higher term, it immediately recognizes the authority of that newer term, just as a politician would recognize a newly elected government.

The **randomized election timeouts** work like staggered campaign announcement deadlines. In human politics, if everyone announced their candidacy at exactly the same moment, chaos would ensue with split votes and unclear mandates. Raft prevents this by having each node wait a random amount of time before starting an election campaign. This natural staggering means that typically one node will become a candidate before others, collect votes quickly, and establish clear leadership before competitors even begin campaigning.

The **vote persistence** mechanism is like constitutional law - once you've cast your vote in a term, that decision is permanently recorded and cannot be changed. This prevents the electoral chaos that would occur if voters could change their minds mid-election. In Raft, once a node votes for a candidate in a particular term, that vote is written to persistent storage and cannot be revoked, ensuring electoral integrity even if the node crashes and restarts.

The **heartbeat system** functions like regular parliamentary sessions - the leader must continuously demonstrate active governance by sending heartbeat messages to all followers. If followers stop receiving these "proof of life" signals, they assume the leader has failed (like a government falling due to inability to govern) and trigger new elections. This ensures that leadership failures are detected quickly and new leadership emerges automatically.

![Raft Node State Machine](./diagrams/raft-state-machine.svg)

### Leader Election Algorithm

The Raft election algorithm implements a democratic process with built-in safeguards against split votes and electoral deadlock. Every node begins life as a **follower**, waiting to receive either heartbeats from an existing leader or vote requests from candidates. This initial state ensures that new nodes don't immediately disrupt stable clusters by launching unnecessary elections.

The election process begins when a follower's **election timeout** expires without receiving communication from a leader. This timeout is randomized between `ELECTION_TIMEOUT_MIN` and `ELECTION_TIMEOUT_MAX` to prevent simultaneous candidacy announcements that would lead to split votes. When the timeout expires, the follower transitions to the **candidate** state and begins the election protocol.

The detailed election algorithm proceeds as follows:

1. **Increment the current term** - The candidate increments its `current_term` counter, representing a bid for leadership in a new political term. This ensures that election attempts are totally ordered across the cluster.

2. **Vote for self** - The candidate immediately casts its vote for itself in the new term and persists both the new term and vote decision to stable storage. This self-vote ensures the candidate has at least one vote and prevents it from voting for competitors in the same term.

3. **Reset election timeout** - A new randomized timeout is set to handle the case where this election fails and another election attempt becomes necessary. This prevents immediate re-election attempts that could interfere with other candidates.

4. **Send RequestVote RPCs in parallel** - The candidate simultaneously sends `RequestVoteRequest` messages to all other nodes in the cluster. Parallel sending ensures the election completes quickly, reducing the window for split votes.

5. **Collect responses and count votes** - As `RequestVoteResponse` messages arrive, the candidate tallies votes granted in its favor. Each positive response represents one node's endorsement of the candidate's leadership bid.

6. **Evaluate election outcome** - The candidate monitors for three possible outcomes:
   - **Victory**: Receiving votes from a strict majority of nodes (more than half the cluster size) results in immediate transition to leader state
   - **Defeat**: Receiving a heartbeat from another node claiming leadership with an equal or higher term results in stepping down to follower
   - **Stalemate**: The election timeout expires without achieving majority support, triggering a new election attempt

7. **Handle victory** - Upon receiving majority support, the candidate transitions to `LEADER` state and immediately begins sending heartbeat messages to all followers to establish authority and prevent new elections.

8. **Handle defeat** - If another node establishes legitimate leadership (indicated by valid heartbeats with current or higher term), the candidate gracefully transitions to `FOLLOWER` and recognizes the new leader's authority.

9. **Handle stalemate** - If no candidate achieves majority support within the election timeout, all candidates return to follower state, increment their terms, and begin new randomized election timeouts. The randomization ensures that subsequent elections are less likely to result in additional stalemates.

The `RequestVoteRequest` message carries four critical pieces of information that enable recipients to make informed voting decisions:

| Field | Type | Purpose |
|-------|------|---------|
| `term` | `Term` | The candidate's current term number, establishing the political period for this election |
| `candidate_id` | `NodeId` | Unique identifier of the requesting candidate for vote tracking |
| `last_log_index` | `LogIndex` | Index of the candidate's last log entry, used for log completeness checking |
| `last_log_term` | `Term` | Term of the candidate's last log entry, used for log recency verification |

Recipients evaluate vote requests using strict criteria designed to maintain cluster safety and prevent data loss:

1. **Term validity check** - The recipient rejects requests from candidates with terms lower than its own current term, preventing outdated leadership claims
2. **Vote availability check** - The recipient grants votes only if it hasn't already voted in the current term, ensuring each node votes at most once per term
3. **Log completeness check** - The recipient compares the candidate's log to its own, granting votes only to candidates whose logs are at least as complete and recent as its own

![Leader Election Sequence](./diagrams/election-sequence.svg)

> **Critical Safety Property**: The election algorithm guarantees that at most one leader can be elected in any given term. This follows from the mathematical fact that two distinct candidates cannot both receive votes from a strict majority of nodes, since majorities must overlap by at least one node.

**Election State Transitions:**

| Current State | Trigger Event | Next State | Actions Taken |
|---------------|---------------|------------|---------------|
| `FOLLOWER` | Election timeout expires | `CANDIDATE` | Increment term, vote for self, send RequestVote RPCs |
| `CANDIDATE` | Receive majority votes | `LEADER` | Send heartbeats to all followers |
| `CANDIDATE` | Receive higher term | `FOLLOWER` | Update term, reset election timeout |
| `CANDIDATE` | Election timeout expires | `CANDIDATE` | Increment term, restart election |
| `LEADER` | Receive higher term | `FOLLOWER` | Step down, update term |
| Any State | Receive valid RPC with higher term | `FOLLOWER` | Update term, reset election timeout |

### Heartbeat and Failure Detection

The heartbeat mechanism serves as the circulatory system of the Raft cluster, continuously proving leader vitality and preventing unnecessary elections in stable clusters. Once elected, a leader must regularly demonstrate its continued operation by sending `AppendEntriesRequest` messages (which may be empty) to all followers at intervals shorter than the typical election timeout.

The leader's heartbeat responsibilities include:

1. **Periodic heartbeat transmission** - Every `HEARTBEAT_INTERVAL` milliseconds, the leader sends `AppendEntriesRequest` messages to all followers, regardless of whether new log entries need replication. This regular communication prevents followers from timing out and starting elections.

2. **Authority assertion** - Each heartbeat contains the leader's current term, allowing followers to verify the leader's legitimacy. Followers receiving heartbeats from leaders with equal or greater terms reset their election timeouts, acknowledging continued leadership.

3. **Cluster health monitoring** - Leaders track response times and success rates for heartbeats to each follower, enabling detection of network issues or follower failures that might require special handling.

4. **Election suppression** - By maintaining regular communication, heartbeats prevent the election timeouts that would otherwise trigger leadership challenges from healthy followers.

The failure detection mechanism operates through timeout-based monitoring on both sides of the leader-follower relationship:

**Follower-side failure detection** works by monitoring the time elapsed since the last valid communication from the leader. Each follower maintains an `election_deadline` timestamp that is reset whenever it receives:
- Valid `AppendEntriesRequest` messages from the current leader
- Valid `RequestVoteRequest` messages from candidates with equal or higher terms
- Any RPC that causes the follower to update its current term

When the current time exceeds the `election_deadline`, the follower concludes that the leader has failed and transitions to candidate state to begin a new election. The randomized nature of election timeouts means that typically only one follower will timeout first, reducing the likelihood of split votes.

**Leader-side failure detection** is more nuanced since leaders don't automatically step down due to communication failures with followers. However, leaders do monitor follower responsiveness for operational reasons:
- Tracking which followers are current with log replication
- Identifying followers that may need `InstallSnapshot` operations due to being too far behind
- Detecting network partitions that might affect the leader's ability to maintain majority support

> **Design Insight**: The asymmetric nature of failure detection reflects the reality that in consensus systems, losing a leader is far more disruptive than losing a follower. Leaders step down only when they discover higher terms, not due to communication failures with followers.

**Heartbeat Message Structure:**

The `AppendEntriesRequest` message serves double duty as both a log replication mechanism and a heartbeat. For pure heartbeat purposes (when no new entries need replication), the message contains:

| Field | Type | Value for Heartbeat | Purpose |
|-------|------|---------------------|---------|
| `term` | `Term` | Leader's current term | Establishes leader authority |
| `leader_id` | `NodeId` | Leader's identifier | Identifies message source |
| `prev_log_index` | `LogIndex` | Index of last known replicated entry | Log consistency checking |
| `prev_log_term` | `Term` | Term of last known replicated entry | Log consistency verification |
| `entries` | `List[LogEntry]` | Empty list `[]` | No new entries for pure heartbeat |
| `leader_commit` | `LogIndex` | Leader's commit index | Informs followers of committed entries |

**Timeout Configuration Trade-offs:**

The choice of timeout values involves balancing election frequency against failure detection speed:

| Timeout Range | Pros | Cons | Best Use Case |
|---------------|------|------|---------------|
| Short (50-100ms) | Fast failure detection, quick recovery | Frequent false elections due to network delays | Low-latency networks, small clusters |
| Medium (150-300ms) | Good balance of speed and stability | Moderate recovery time | General purpose, recommended default |
| Long (500ms+) | Very stable, few false elections | Slow failure detection, poor user experience | High-latency networks, large clusters |

> **Configuration Guidance**: The relationship `HEARTBEAT_INTERVAL < ELECTION_TIMEOUT_MIN < ELECTION_TIMEOUT_MAX` must be maintained, with heartbeat interval typically set to one-third of the minimum election timeout to provide multiple heartbeat opportunities within each election timeout window.

### Architecture Decisions for Elections

The election system involves several critical design decisions that significantly impact cluster behavior, safety, and performance. Each decision represents a careful balance of trade-offs that must be understood for successful implementation.

> **Decision: Randomized Election Timeouts**
> - **Context**: Multiple nodes timing out simultaneously leads to split votes and election failures, reducing cluster availability during leader transitions
> - **Options Considered**: Fixed timeouts, exponential backoff, priority-based election ordering
> - **Decision**: Implement randomized timeouts between `ELECTION_TIMEOUT_MIN` (150ms) and `ELECTION_TIMEOUT_MAX` (300ms)
> - **Rationale**: Randomization naturally staggers candidate emergence, typically allowing one candidate to complete election before others begin. Mathematical analysis shows that with proper timeout ranges, split vote probability drops below 1% for reasonable cluster sizes.
> - **Consequences**: Enables fast, reliable leader election in most cases, but introduces slight unpredictability in election timing. Requires careful timeout range tuning based on network characteristics.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Fixed timeouts | Predictable timing, simple implementation | High split vote probability, poor availability | No |
| Exponential backoff | Reduces split votes over time | Slow convergence, complex tuning | No |
| Randomized timeouts | Low split vote probability, fast elections | Requires range tuning, slight unpredictability | **Yes** |
| Priority-based | Deterministic leader selection | Requires global priority assignment, single point of failure | No |

> **Decision: Term-Based Authority**
> - **Context**: Distributed systems require a mechanism to distinguish current leadership from stale leadership claims, especially after network partitions heal
> - **Options Considered**: Wall-clock timestamps, logical clocks, monotonic terms, leader leases
> - **Decision**: Use monotonically increasing term numbers as the source of leadership authority
> - **Rationale**: Terms provide total ordering of leadership epochs without requiring synchronized clocks. Higher terms automatically supersede lower terms, enabling clean resolution of leadership conflicts after partition healing.
> - **Consequences**: Enables safe leadership transitions and partition tolerance, but requires persistent term storage and careful term comparison logic in all RPC handlers.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Wall-clock timestamps | Human-readable, natural ordering | Requires clock synchronization, vulnerable to clock skew | No |
| Logical clocks | No clock sync required | Complex vector clock maintenance | No |
| Monotonic terms | Simple, total ordering, crash-safe | Requires persistent storage | **Yes** |
| Leader leases | High performance reads | Complex lease renewal, vulnerable to clock issues | No |

> **Decision: Vote Persistence Requirements**
> - **Context**: Node crashes during elections could lead to double-voting if vote decisions aren't durably stored, potentially electing multiple leaders
> - **Options Considered**: In-memory votes only, synchronous disk persistence, asynchronous persistence, majority-witnessed votes
> - **Decision**: Require synchronous persistence of both `current_term` and `voted_for` before responding to vote requests
> - **Rationale**: Vote persistence ensures electoral integrity across crashes. A node that votes for candidate A cannot vote for candidate B in the same term after restarting, preventing double-voting that could lead to split-brain scenarios.
> - **Consequences**: Provides strong safety guarantees and prevents double-voting, but adds disk I/O latency to every vote request, potentially slowing elections in high-latency storage environments.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| In-memory votes | Fast voting, no I/O overhead | Unsafe after crashes, possible double-voting | No |
| Asynchronous persistence | Better performance than sync | Race conditions, potential safety violations | No |
| Synchronous persistence | Strong safety, prevents double-voting | Higher latency, requires reliable storage | **Yes** |
| Majority-witnessed | No local disk required | Complex protocol, requires additional round trips | No |

> **Decision: Leadership Step-Down Triggers**
> - **Context**: Leaders must recognize when their authority is no longer valid to prevent split-brain scenarios and ensure cluster progress
> - **Options Considered**: Heartbeat failure step-down, term-based step-down only, quorum loss step-down, lease-based step-down
> - **Decision**: Leaders step down only upon discovering higher terms, not due to communication failures with followers
> - **Rationale**: Communication failures don't necessarily indicate leadership invalidity - the leader may still have majority support. Term-based step-down ensures safety while avoiding unnecessary leadership changes due to transient network issues.
> - **Consequences**: Provides strong leadership stability and avoids thrashing, but leaders may continue operating during minority partitions until they discover higher terms from the majority partition.

| Option | Pros | Cons | Chosen? |
|--------|------|------|---------|
| Heartbeat failure step-down | Fast detection of isolation | False positives from network glitches | No |
| Term-based step-down only | Stable leadership, safety guaranteed | May continue during minority partition | **Yes** |
| Quorum loss step-down | Immediate minority partition detection | Complex majority detection, potential thrashing | No |
| Lease-based step-down | Bounded leadership during partitions | Requires synchronized clocks, complex renewal | No |

**Persistent State Management:**

The election system requires careful management of persistent state to maintain safety across node restarts. The `PersistentState` structure captures the minimal information that must survive crashes:

| Field | Type | Persistence Requirement | Recovery Behavior |
|-------|------|------------------------|-------------------|
| `current_term` | `Term` | Must fsync before any term-based action | Restore exact term to prevent authority conflicts |
| `voted_for` | `Optional[NodeId]` | Must fsync before vote response | Restore vote to prevent double-voting |
| `log` | `List[LogEntry]` | Must fsync before acknowledging client writes | Restore complete log for consistency |

**Election Performance Characteristics:**

Understanding the performance implications of election design helps with cluster sizing and timeout tuning:

| Cluster Size | Expected Election Time | Split Vote Probability | Recommended Timeout Range |
|--------------|------------------------|------------------------|---------------------------|
| 3 nodes | 150-200ms | < 1% | 150-300ms |
| 5 nodes | 200-250ms | < 2% | 200-400ms |
| 7 nodes | 250-350ms | < 3% | 300-600ms |
| 9 nodes | 300-500ms | < 5% | 400-800ms |

> **Scaling Consideration**: Election time grows with cluster size due to increased message complexity and higher split vote probability. Clusters larger than 9 nodes typically use multi-tier architectures rather than single Raft groups.

### Common Election Pitfalls

Election implementation contains several subtle correctness issues that can lead to safety violations, liveness problems, or degraded performance. Understanding these pitfalls is crucial for building reliable consensus systems.

⚠️ **Pitfall: Term Updates Without Persistence**

A common mistake is updating the in-memory `current_term` without immediately persisting it to stable storage before taking any term-based actions like voting or stepping down. This creates a dangerous race condition where a node might:

1. Receive a `RequestVoteRequest` with term 5 (higher than its current term 3)
2. Update its in-memory term to 5 and send a positive vote response
3. Crash before persisting the new term to disk
4. Restart with term 3 from disk and potentially vote again in term 5 for a different candidate

This double-voting scenario can lead to multiple leaders being elected in the same term, violating Raft's fundamental safety property. The correct approach requires:

- Immediately persist `current_term` and `voted_for` to stable storage using `fsync`
- Only send vote responses after successful persistence
- Always restore term and vote state from persistent storage on startup

Example of the persistence sequence:
1. Receive RPC with higher term
2. Call `update_term(new_term, None)` which performs synchronous disk write
3. Only after successful persistence, send RPC response or take other actions

⚠️ **Pitfall: Insufficient Randomization Range**

Setting election timeout ranges too narrow leads to frequent split votes, while ranges too wide cause slow failure detection. A common error is using ranges like 100-110ms, which provide insufficient randomization to prevent simultaneous candidate emergence.

The timeout range must account for:
- Network round-trip time variability (typically 1-10ms in local networks)
- RPC processing time at recipients (typically 1-5ms)
- Time required to collect majority votes (proportional to cluster size)

Insufficient randomization manifests as:
- Frequent elections with no winner (visible in logs as repeated term increments)
- High cluster CPU usage due to constant election activity  
- Poor client request latency due to unavailable leadership

The fix requires setting `ELECTION_TIMEOUT_MAX` to at least twice `ELECTION_TIMEOUT_MIN`, with the range wide enough to accommodate network jitter. For local networks, 150-300ms works well, while high-latency networks may require 500-1000ms ranges.

⚠️ **Pitfall: Ignoring Log Completeness in Vote Decisions**

A critical safety requirement is that nodes only vote for candidates whose logs are at least as complete and up-to-date as their own. Failing to implement this check can lead to data loss when a candidate with an incomplete log becomes leader and overwrites committed entries.

The log completeness check compares the candidate's last log entry with the recipient's last log entry:

1. **Term comparison**: If the candidate's last log term is higher, the candidate is more up-to-date
2. **Index comparison**: If terms are equal, the candidate with the higher last log index is more up-to-date  
3. **Vote decision**: Only vote for candidates that are at least as up-to-date

Without this check, the following dangerous scenario can occur:
- Node A has log entries [term=1, term=2, term=3] 
- Node B has log entries [term=1, term=2] (missing the term=3 entry)
- Node B starts an election and receives votes from nodes that also lack the term=3 entry
- Node B becomes leader and never replicates the term=3 entry, causing data loss

⚠️ **Pitfall: Race Conditions in State Transitions**

Election state transitions must be atomic to prevent inconsistent behavior where a node believes it's in multiple states simultaneously or takes actions inappropriate for its current state. Common race conditions include:

- **Concurrent election timeout and vote request processing**: A follower's election timeout expires while simultaneously receiving a vote request, potentially causing it to both start its own election and vote for another candidate
- **Concurrent heartbeat and election start**: A candidate receives a heartbeat just as it's transitioning to leader state, causing confusion about current leadership
- **Concurrent term updates**: Multiple RPC handlers updating `current_term` simultaneously without proper synchronization

The solution requires:
- Protecting all state transitions with appropriate locking mechanisms
- Processing RPCs and timeouts in a single-threaded event loop, or
- Using atomic operations and careful ordering of state updates

⚠️ **Pitfall: Improper Election Timeout Reset**

Election timeouts must be reset at the correct times to prevent both unnecessary elections and failure to detect actual leader failures. Common mistakes include:

- **Forgetting to reset timeout on valid RPCs**: Not resetting the election timeout when receiving valid `AppendEntriesRequest` or `RequestVoteRequest` messages, causing followers to start elections even when leadership is stable
- **Resetting timeout on invalid RPCs**: Resetting timeouts when receiving RPCs with stale terms, allowing failed leaders to prevent new elections by continuing to send invalid heartbeats
- **Double timeout resets**: Resetting timeouts multiple times for the same RPC, potentially in different threads, causing race conditions

The correct timeout reset behavior:
- Reset election timeout when receiving `AppendEntriesRequest` with current or higher term from valid leader
- Reset election timeout when receiving `RequestVoteRequest` with higher term (since this indicates potential new leadership)
- Do NOT reset timeout for RPCs with terms lower than the current term
- Reset timeout when transitioning to follower state from any other state

⚠️ **Pitfall: Heartbeat Interval Misconfiguration**

Setting heartbeat intervals too close to election timeouts creates a brittle system where minor network delays trigger false leader elections. The relationship between heartbeat interval and election timeout must provide sufficient safety margin.

Common configuration mistakes:
- Setting heartbeat interval to more than half of minimum election timeout
- Using fixed intervals without accounting for network jitter
- Failing to account for RPC processing time in interval calculations

The recommended configuration maintains `HEARTBEAT_INTERVAL ≤ ELECTION_TIMEOUT_MIN / 3` to ensure multiple heartbeat opportunities within each election timeout window. This provides resilience against:
- Individual heartbeat message loss
- Temporary network delays
- Brief processing delays at followers

⚠️ **Pitfall: Incomplete Persistence Error Handling**

Disk I/O operations required for term and vote persistence can fail due to hardware issues, disk full conditions, or file system errors. Improper handling of these failures can lead to safety violations or cluster unavailability.

Common persistence error patterns:
- **Sending vote responses before confirming disk write**: Responding to vote requests before ensuring the term and vote are durably stored
- **Ignoring fsync failures**: Not checking return codes from sync operations, missing cases where data wasn't actually written to persistent storage  
- **Partial state persistence**: Successfully writing term but failing to write vote, or vice versa, leading to inconsistent recovery state

Proper persistence error handling requires:
- Always check return codes from write and sync operations
- Treat persistence failures as fatal errors that prevent further participation
- Use atomic file operations or write-ahead logging to ensure consistent persistence
- Consider node failure modes and recovery procedures when persistence fails

### Implementation Guidance

This subsection provides concrete implementation support for building the Raft election and safety mechanisms. The focus is on providing complete infrastructure code and detailed skeletons for core election logic.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| RPC Transport | HTTP REST with JSON serialization | gRPC with Protocol Buffers |
| Persistent Storage | JSON files with explicit fsync | SQLite with WAL mode |
| Threading Model | asyncio event loop (single-threaded) | Thread pool with locks |
| Configuration | YAML configuration files | Environment variables with validation |
| Logging | Python logging with structured output | Structured logging with correlation IDs |

**Recommended File Structure:**

```
titan-consensus/
├── main.py                     ← Entry point and CLI
├── config/
│   └── node_config.py         ← Configuration management
├── core/
│   ├── __init__.py
│   ├── node.py                ← Main RaftNode class (core logic skeleton)
│   ├── state.py               ← NodeState enum and state management
│   ├── types.py               ← All data types (RequestVoteRequest, etc.)
│   └── persistence.py         ← Persistent state management (complete)
├── network/
│   ├── __init__.py
│   ├── rpc_server.py          ← HTTP server for incoming RPCs (complete)
│   ├── rpc_client.py          ← HTTP client for outgoing RPCs (complete)
│   └── messages.py            ← Message serialization utilities (complete)
├── tests/
│   ├── __init__.py
│   ├── test_election.py       ← Election-specific tests
│   └── test_integration.py    ← Multi-node integration tests
└── scripts/
    └── cluster_setup.py       ← Helper for starting test clusters
```

**Infrastructure Starter Code:**

**File: `core/types.py` (Complete)**
```python
"""
Core data types for Raft consensus implementation.
All types follow the exact naming conventions from the design document.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum
import time

# Type aliases for clarity and type checking
NodeId = str
Term = int
LogIndex = int

class NodeState(Enum):
    """Node state in the Raft protocol."""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"

@dataclass
class LogEntry:
    """Individual log entry containing client data and metadata."""
    term: Term
    index: LogIndex
    data: bytes
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize entry for persistence or network transmission."""
        return {
            'term': self.term,
            'index': self.index,
            'data': self.data.hex(),  # Convert bytes to hex string for JSON
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        """Deserialize entry from storage or network."""
        return cls(
            term=data['term'],
            index=data['index'],
            data=bytes.fromhex(data['data']),
            timestamp=data['timestamp']
        )

@dataclass
class RequestVoteRequest:
    """RPC request for candidate to solicit votes."""
    term: Term
    candidate_id: NodeId
    last_log_index: LogIndex
    last_log_term: Term
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'term': self.term,
            'candidate_id': self.candidate_id,
            'last_log_index': self.last_log_index,
            'last_log_term': self.last_log_term
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RequestVoteRequest':
        return cls(
            term=data['term'],
            candidate_id=data['candidate_id'],
            last_log_index=data['last_log_index'],
            last_log_term=data['last_log_term']
        )

@dataclass
class RequestVoteResponse:
    """RPC response for vote requests."""
    term: Term
    vote_granted: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'term': self.term,
            'vote_granted': self.vote_granted
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RequestVoteResponse':
        return cls(
            term=data['term'],
            vote_granted=data['vote_granted']
        )

@dataclass
class AppendEntriesRequest:
    """RPC request for log replication and heartbeats."""
    term: Term
    leader_id: NodeId
    prev_log_index: LogIndex
    prev_log_term: Term
    entries: List[LogEntry]
    leader_commit: LogIndex
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'term': self.term,
            'leader_id': self.leader_id,
            'prev_log_index': self.prev_log_index,
            'prev_log_term': self.prev_log_term,
            'entries': [entry.to_dict() for entry in self.entries],
            'leader_commit': self.leader_commit
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppendEntriesRequest':
        return cls(
            term=data['term'],
            leader_id=data['leader_id'],
            prev_log_index=data['prev_log_index'],
            prev_log_term=data['prev_log_term'],
            entries=[LogEntry.from_dict(e) for e in data['entries']],
            leader_commit=data['leader_commit']
        )

@dataclass
class AppendEntriesResponse:
    """RPC response for append entries requests."""
    term: Term
    success: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'term': self.term,
            'success': self.success
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppendEntriesResponse':
        return cls(
            term=data['term'],
            success=data['success']
        )

@dataclass
class PersistentState:
    """State that must survive crashes and be persisted to disk."""
    current_term: Term
    voted_for: Optional[NodeId]
    log: List[LogEntry]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'current_term': self.current_term,
            'voted_for': self.voted_for,
            'log': [entry.to_dict() for entry in self.log]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersistentState':
        return cls(
            current_term=data['current_term'],
            voted_for=data.get('voted_for'),
            log=[LogEntry.from_dict(e) for e in data['log']]
        )

@dataclass
class VolatileState:
    """State that can be reconstructed after crashes."""
    node_state: NodeState
    commit_index: LogIndex
    last_applied: LogIndex
    leader_id: Optional[NodeId]
    election_deadline: float
    
    def __init__(self):
        self.node_state = NodeState.FOLLOWER
        self.commit_index = 0
        self.last_applied = 0
        self.leader_id = None
        self.election_deadline = 0.0

# Configuration constants
ELECTION_TIMEOUT_MIN = 0.150  # 150ms minimum election timeout
ELECTION_TIMEOUT_MAX = 0.300  # 300ms maximum election timeout  
HEARTBEAT_INTERVAL = 0.050    # 50ms leader heartbeat interval
MAX_CLUSTER_SIZE = 9          # 9 nodes maximum recommended cluster size
```

**File: `core/persistence.py` (Complete)**
```python
"""
Persistent state management with atomic writes and crash safety.
Handles the critical requirement that term and vote state survive crashes.
"""
import json
import os
import tempfile
import threading
from typing import Optional
from .types import PersistentState, LogEntry

class StateManager:
    """Manages persistent state with atomic writes and crash safety."""
    
    def __init__(self, data_dir: str):
        """Initialize state manager with data directory."""
        self.data_dir = data_dir
        self.state_file = os.path.join(data_dir, 'raft_state.json')
        self.lock = threading.Lock()  # Protect concurrent access
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
    
    def save_to_file(self, state: PersistentState) -> None:
        """
        Atomically persist state to disk with fsync for durability.
        Critical for election safety - must complete before sending vote responses.
        """
        with self.lock:
            # Write to temporary file first for atomic replacement
            temp_fd, temp_path = tempfile.mkstemp(dir=self.data_dir, suffix='.tmp')
            try:
                with os.fdopen(temp_fd, 'w') as f:
                    json.dump(state.to_dict(), f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk
                
                # Atomic replacement - either old or new file exists, never partial
                os.replace(temp_path, self.state_file)
                
                # Sync directory to ensure filename change is persistent
                dir_fd = os.open(self.data_dir, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
                    
            except Exception as e:
                # Clean up temp file on any error
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise e
    
    def load_from_file(self) -> Optional[PersistentState]:
        """
        Load persistent state from disk.
        Returns None if no state file exists (first startup).
        """
        with self.lock:
            if not os.path.exists(self.state_file):
                return None
            
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                return PersistentState.from_dict(data)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                raise RuntimeError(f"Corrupted state file {self.state_file}: {e}")
    
    def initialize_default_state(self, node_id: str) -> PersistentState:
        """Create and persist initial state for new nodes."""
        initial_state = PersistentState(
            current_term=0,
            voted_for=None,
            log=[]
        )
        self.save_to_file(initial_state)
        return initial_state
```

**File: `network/rpc_server.py` (Complete)**
```python
"""
HTTP-based RPC server for receiving Raft messages.
Provides clean separation between network handling and consensus logic.
"""
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Callable, Dict, Any
import logging

from core.types import (
    RequestVoteRequest, RequestVoteResponse,
    AppendEntriesRequest, AppendEntriesResponse
)

logger = logging.getLogger(__name__)

class RaftRPCHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Raft RPC calls."""
    
    def do_POST(self):
        """Handle incoming RPC requests."""
        try:
            # Parse request path to determine RPC type
            if self.path == '/request_vote':
                self._handle_request_vote()
            elif self.path == '/append_entries':
                self._handle_append_entries()
            else:
                self._send_error(404, "Unknown RPC endpoint")
        except Exception as e:
            logger.error(f"RPC handler error: {e}")
            self._send_error(500, str(e))
    
    def _handle_request_vote(self):
        """Process RequestVote RPC."""
        request_data = self._read_request_body()
        request = RequestVoteRequest.from_dict(request_data)
        
        # Delegate to consensus node for processing
        response = self.server.consensus_node.handle_request_vote(request)
        
        self._send_response(response.to_dict())
    
    def _handle_append_entries(self):
        """Process AppendEntries RPC."""
        request_data = self._read_request_body()
        request = AppendEntriesRequest.from_dict(request_data)
        
        # Delegate to consensus node for processing
        response = self.server.consensus_node.handle_append_entries(request)
        
        self._send_response(response.to_dict())
    
    def _read_request_body(self) -> Dict[str, Any]:
        """Read and parse JSON request body."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        return json.loads(body)
    
    def _send_response(self, data: Dict[str, Any]):
        """Send JSON response."""
        response_json = json.dumps(data)
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response_json)))
        self.end_headers()
        self.wfile.write(response_json.encode('utf-8'))
    
    def _send_error(self, code: int, message: str):
        """Send error response."""
        self.send_response(code)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(message.encode('utf-8'))
    
    def log_message(self, format, *args):
        """Suppress default HTTP server logging."""
        pass

class RaftRPCServer:
    """RPC server for Raft consensus communication."""
    
    def __init__(self, host: str, port: int, consensus_node):
        """Initialize RPC server."""
        self.host = host
        self.port = port
        self.consensus_node = consensus_node
        self.server = None
        self.server_thread = None
    
    def start(self) -> None:
        """Start the RPC server in a background thread."""
        self.server = HTTPServer((self.host, self.port), RaftRPCHandler)
        self.server.consensus_node = self.consensus_node
        
        self.server_thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True
        )
        self.server_thread.start()
        logger.info(f"RPC server started on {self.host}:{self.port}")
    
    def stop(self) -> None:
        """Stop the RPC server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.server_thread:
            self.server_thread.join()
        logger.info("RPC server stopped")
```

**File: `network/rpc_client.py` (Complete)**
```python
"""
HTTP-based RPC client for sending Raft messages.
Handles network failures and timeouts gracefully.
"""
import json
import requests
import logging
from typing import Optional
from requests.exceptions import RequestException, Timeout

from core.types import (
    RequestVoteRequest, RequestVoteResponse,
    AppendEntriesRequest, AppendEntriesResponse
)

logger = logging.getLogger(__name__)

class RaftRPCClient:
    """RPC client for sending Raft messages to other nodes."""
    
    def __init__(self, timeout: float = 1.0):
        """Initialize RPC client with request timeout."""
        self.timeout = timeout
        self.session = requests.Session()
    
    def send_request_vote(self, target_host: str, target_port: int, 
                         request: RequestVoteRequest) -> Optional[RequestVoteResponse]:
        """
        Send RequestVote RPC to target node.
        Returns None if request fails (network error, timeout, etc.).
        """
        url = f"http://{target_host}:{target_port}/request_vote"
        try:
            response = self.session.post(
                url,
                json=request.to_dict(),
                timeout=self.timeout
            )
            response.raise_for_status()
            return RequestVoteResponse.from_dict(response.json())
            
        except (RequestException, Timeout, ValueError) as e:
            logger.warning(f"RequestVote to {target_host}:{target_port} failed: {e}")
            return None
    
    def send_append_entries(self, target_host: str, target_port: int,
                           request: AppendEntriesRequest) -> Optional[AppendEntriesResponse]:
        """
        Send AppendEntries RPC to target node.
        Returns None if request fails (network error, timeout, etc.).
        """
        url = f"http://{target_host}:{target_port}/append_entries"
        try:
            response = self.session.post(
                url,
                json=request.to_dict(),
                timeout=self.timeout
            )
            response.raise_for_status()
            return AppendEntriesResponse.from_dict(response.json())
            
        except (RequestException, Timeout, ValueError) as e:
            logger.warning(f"AppendEntries to {target_host}:{target_port} failed: {e}")
            return None
```

**Core Logic Skeleton Code:**

**File: `core/node.py` (Skeleton with detailed TODOs)**
```python
"""
Main Raft consensus node implementation.
This is the core logic that students should implement following the design document.
"""
import asyncio
import random
import threading
import time
import logging
from typing import Dict, List, Optional, Set

from .types import (
    NodeId, Term, LogIndex, NodeState,
    RequestVoteRequest, RequestVoteResponse,
    AppendEntriesRequest, AppendEntriesResponse,
    PersistentState, VolatileState,
    ELECTION_TIMEOUT_MIN, ELECTION_TIMEOUT_MAX, HEARTBEAT_INTERVAL
)
from .persistence import StateManager
from network.rpc_client import RaftRPCClient

logger = logging.getLogger(__name__)

class RaftNode:
    """
    Main Raft consensus node implementing leader election and safety.
    Students implement the core election logic in the methods below.
    """
    
    def __init__(self, node_id: NodeId, cluster_nodes: Dict[NodeId, tuple], 
                 data_dir: str):
        """
        Initialize Raft node.
        
        Args:
            node_id: Unique identifier for this node
            cluster_nodes: Map of node_id -> (host, port) for all cluster nodes
            data_dir: Directory for persistent state storage
        """
        self.node_id = node_id
        self.cluster_nodes = cluster_nodes
        self.other_nodes = {nid: addr for nid, addr in cluster_nodes.items() 
                           if nid != node_id}
        
        # State management
        self.state_manager = StateManager(data_dir)
        self.persistent_state = None  # Loaded in start()
        self.volatile_state = VolatileState()
        
        # Network communication
        self.rpc_client = RaftRPCClient()
        
        # Threading and timing
        self.lock = threading.RLock()  # Protects all state access
        self.running = False
        self.election_timer = None
        self.heartbeat_timer = None
    
    def start(self) -> None:
        """
        Initialize and start the consensus node.
        Loads persistent state and begins election timeout.
        """
        with self.lock:
            # TODO 1: Load persistent state from disk using state_manager
            # If no state file exists, initialize with default values
            # Hint: Use state_manager.load_from_file() and handle None return
            
            # TODO 2: Initialize volatile state
            # Set node_state to FOLLOWER, reset election timeout
            # Hint: Call _reset_election_timeout() to start timeout monitoring
            
            # TODO 3: Set running flag and start background timers
            # The node should begin participating in the cluster
            
            pass  # Remove this when implementing
    
    def stop(self) -> None:
        """Stop the node and clean up resources."""
        with self.lock:
            self.running = False
            if self.election_timer:
                self.election_timer.cancel()
            if self.heartbeat_timer:
                self.heartbeat_timer.cancel()
    
    def handle_request_vote(self, request: RequestVoteRequest) -> RequestVoteResponse:
        """
        Process RequestVote RPC from candidate.
        Implements the vote decision logic with safety checks.
        """
        with self.lock:
            logger.info(f"Received RequestVote from {request.candidate_id} for term {request.term}")
            
            # TODO 1: Check if request term is higher than current term
            # If so, update current term and convert to follower
            # Hint: Use update_term() method and transition to FOLLOWER state
            
            # TODO 2: Reject vote if request term is lower than current term
            # Return RequestVoteResponse with current term and vote_granted=False
            
            # TODO 3: Check if we have already voted in this term
            # If voted_for is not None and not equal to candidate_id, reject vote
            # This prevents double-voting in the same term
            
            # TODO 4: Implement log completeness check
            # Compare candidate's last log term and index with our own
            # Only vote for candidates whose logs are at least as complete
            # Hint: Use _is_log_more_complete() helper method
            
            # TODO 5: Grant vote and persist decision
            # Update voted_for field and save to disk before responding
            # Reset election timeout since we're participating in valid election
            # Hint: Use update_term() to persist vote and _reset_election_timeout()
            
            # TODO 6: Return appropriate RequestVoteResponse
            # Include current term and whether vote was granted
            
            pass  # Remove this when implementing
    
    def handle_append_entries(self, request: AppendEntriesRequest) -> AppendEntriesResponse:
        """
        Process AppendEntries RPC from leader (heartbeat or log replication).
        For Milestone 1, this primarily handles heartbeats and term updates.
        """
        with self.lock:
            logger.debug(f"Received AppendEntries from {request.leader_id} for term {request.term}")
            
            # TODO 1: Check if request term is higher than current term
            # If so, update current term and convert to follower
            # Hint: Use update_term() and transition to FOLLOWER state
            
            # TODO 2: Reject if request term is lower than current term
            # Return AppendEntriesResponse with current term and success=False
            
            # TODO 3: Valid heartbeat from current leader
            # Reset election timeout to prevent unnecessary elections
            # Update leader_id in volatile state for tracking
            # Hint: Use _reset_election_timeout() and update volatile_state.leader_id
            
            # TODO 4: For Milestone 1, accept all heartbeats (empty entries)
            # Log consistency checking will be implemented in Milestone 2
            # Return success=True for valid heartbeats from current term leader
            
            # TODO 5: Return AppendEntriesResponse
            # Include current term and success status
            
            pass  # Remove this when implementing
    
    def _start_election(self) -> None:
        """
        Transition to candidate state and begin election process.
        Implements the core election algorithm from the design document.
        """
        with self.lock:
            if not self.running:
                return
                
            logger.info(f"Starting election for term {self.persistent_state.current_term + 1}")
            
            # TODO 1: Increment current term and vote for self
            # This represents a bid for leadership in the new term
            # Hint: Use update_term() to persist new term and self-vote
            
            # TODO 2: Transition to CANDIDATE state
            # Update volatile_state.node_state and clear leader_id
            
            # TODO 3: Reset election timeout for potential retry
            # If this election fails, we need another timeout for the next attempt
            # Hint: Use _reset_election_timeout()
            
            # TODO 4: Send RequestVote RPCs to all other nodes in parallel
            # Create RequestVoteRequest with current term and log information
            # Use threading to send requests concurrently for faster elections
            # Hint: Create threads that call _send_vote_request() for each peer
            
            # TODO 5: Start vote counting in separate thread
            # The election outcome depends on collecting majority votes
            # Hint: Start thread that calls _collect_votes() with vote futures
            
            pass  # Remove this when implementing
    
    def _send_vote_request(self, target_node_id: NodeId) -> Optional[RequestVoteResponse]:
        """
        Send RequestVote RPC to a specific node.
        Helper method for parallel vote collection.
        """
        # TODO 1: Get target node address from cluster_nodes
        # Handle case where node is not in cluster configuration
        
        # TODO 2: Create RequestVoteRequest with current state
        # Include current term, candidate_id, and last log information
        # Hint: Use _get_last_log_term() and _get_last_log_index() helpers
        
        # TODO 3: Send RPC using rpc_client
        # Handle network failures gracefully (return None)
        # Hint: Use rpc_client.send_request_vote()
        
        pass  # Remove this when implementing
    
    def _collect_votes(self, vote_responses: List[Optional[RequestVoteResponse]]) -> None:
        """
        Process vote responses and determine election outcome.
        Handles victory, defeat, and stalemate scenarios.
        """
        with self.lock:
            if self.volatile_state.node_state != NodeState.CANDIDATE:
                return  # Election was superseded
            
            # TODO 1: Count granted votes from responses
            # Include self-vote in count, handle None responses (network failures)
            # Only count votes from current term to avoid stale responses
            
            # TODO 2: Check for election victory (majority of cluster)
            # If votes >= (cluster_size + 1) // 2, transition to leader
            # Hint: Use _become_leader() method
            
            # TODO 3: Check for election defeat
            # If any response has higher term, step down to follower
            # Hint: Use update_term() and transition to FOLLOWER
            
            # TODO 4: Handle stalemate (insufficient votes)
            # Return to follower state and wait for next election timeout
            # The randomized timeout will prevent repeated stalemates
            
            pass  # Remove this when implementing
    
    def _become_leader(self) -> None:
        """
        Transition to leader state and begin sending heartbeats.
        Establishes leadership authority in the cluster.
        """
        with self.lock:
            logger.info(f"Became leader for term {self.persistent_state.current_term}")
            
            # TODO 1: Update volatile state for leadership
            # Set node_state to LEADER, update leader_id to self
            
            # TODO 2: Cancel election timeout (leaders don't participate in elections)
            # Clear any pending election timer
            
            # TODO 3: Start sending heartbeats immediately
            # Begin heartbeat timer to maintain leadership authority
            # Hint: Use _start_heartbeat_timer()
            
            pass  # Remove this when implementing
    
    def _send_heartbeats(self) -> None:
        """
        Send heartbeat AppendEntries RPCs to all followers.
        Maintains leadership authority and prevents new elections.
        """
        if self.volatile_state.node_state != NodeState.LEADER:
            return
        
        # TODO 1: Create AppendEntriesRequest for heartbeat
        # Empty entries list, include current term and leader info
        # For Milestone 1, use simple values for prev_log fields
        
        # TODO 2: Send heartbeats to all other nodes in parallel
        # Use threading for concurrent heartbeat delivery
        # Hint: Create threads that call _send_heartbeat() for each peer
        
        # TODO 3: Schedule next heartbeat
        # Continue heartbeat timer while leader
        # Hint: Use _start_heartbeat_timer()
        
        pass  # Remove this when implementing
    
    def update_term(self, new_term: Term, voted_for: Optional[NodeId]) -> None:
        """
        Update persistent term and vote state with atomic disk write.
        Critical for election safety - must complete before any term-based actions.
        """
        # TODO 1: Update persistent state fields
        # Set current_term to new_term and voted_for as specified
        
        # TODO 2: Save to disk with fsync for durability
        # This must complete before sending any RPC responses
        # Hint: Use state_manager.save_to_file()
        
        # TODO 3: Update volatile state if stepping down
        # If new term is higher than current, transition to FOLLOWER
        # Clear leader_id and cancel any pending timers
        
        pass  # Remove this when implementing
    
    def _reset_election_timeout(self) -> None:
        """Reset election timeout with randomized duration."""
        if self.election_timer:
            self.election_timer.cancel()
        
        timeout = random.uniform(ELECTION_TIMEOUT_MIN, ELECTION_TIMEOUT_MAX)
        self.election_timer = threading.Timer(timeout, self._election_timeout)
        self.election_timer.start()
        self.volatile_state.election_deadline = time.time() + timeout
    
    def _start_heartbeat_timer(self) -> None:
        """Start heartbeat timer for leaders."""
        if self.heartbeat_timer:
            self.heartbeat_timer.cancel()
        
        self.heartbeat_timer = threading.Timer(HEARTBEAT_INTERVAL, self._send_heartbeats)
        self.heartbeat_timer.start()
    
    def _election_timeout(self) -> None:
        """Handle election timeout expiration."""
        with self.lock:
            if (self.running and 
                self.volatile_state.node_state in [NodeState.FOLLOWER, NodeState.CANDIDATE]):
                self._start_election()
    
    # Helper methods for log operations (simple implementations for Milestone 1)
    def _get_last_log_index(self) -> LogIndex:
        """Get index of last log entry."""
        return len(self.persistent_state.log)
    
    def _get_last_log_term(self) -> Term:
        """Get term of last log entry."""
        if not self.persistent_state.log:
            return 0
        return self.persistent_state.log[-1].term
    
    def _is_log_more_complete(self, candidate_last_term: Term, 
                             candidate_last_index: LogIndex) -> bool:
        """
        Check if candidate's log is at least as complete as ours.
        Used for vote decision safety checks.
        """
        our_last_term = self._get_last_log_term()
        our_last_index = self._get_last_log_index()
        
        # Candidate is more complete if:
        # 1. Higher last log term, OR
        # 2. Same last log term but higher or equal last log index
        return (candidate_last_term > our_last_term or 
                (candidate_last_term == our_last_term and 
                 candidate_last_index >= our_last_index))
```

**Milestone 1 Checkpoint:**

After implementing the election logic, you should be able to verify the following behaviors:

1. **Start a 3-node cluster:**
   ```bash
   python main.py --node-id=node1 --cluster=node1:8001,node2:8002,node3:8003 --port=8001 &
   python main.py --node-id=node2 --cluster=node1:8001,node2:8002,node3:8003 --port=8002 &
   python main.py --node-id=node3 --cluster=node1:8001,node2:8002,node3:8003 --port=8003 &
   ```

2. **Expected behavior:**
   - One node should become leader within 300ms
   - Other nodes should remain as followers
   - Leader should send heartbeats every 50ms
   - No new elections should occur while leader is healthy

3. **Test leader failure:**
   - Kill the leader process (Ctrl+C)
   - Remaining nodes should detect failure and elect new leader within 600ms
   - New leader should begin sending heartbeats

4. **Signs of correct implementation:**
   - Logs show successful elections with vote collection
   - Only one leader exists at any time
   - Heartbeats prevent unnecessary elections
   - Term numbers increase monotonically

5. **Common debugging steps:**
   - Check that persistent state is saved before vote responses
   - Verify election timeouts are properly randomized
   - Confirm log completeness checks in vote decisions
   - Ensure proper state transitions and lock usage


## Log Replication (Milestone 2)

> **Milestone(s):** Milestone 2: Log Replication - Synchronize state across the cluster with consistency guarantees

### Mental Model: Bookkeeping Ledger

Think of Raft log replication like maintaining a company's accounting ledger across multiple branch offices. Each branch office (node) maintains its own copy of the company's transaction ledger (log), but only the corporate headquarters (leader) can authorize new transactions (log entries). When headquarters processes a new transaction, it sends the entry to all branches with specific instructions: "Add this entry at position 247, but only if your entry at position 246 matches transaction #1023-ABC." This cross-checking ensures that all branches have identical transaction histories - if a branch's ledger doesn't match at position 246, it means some previous transactions were missed or corrupted, so the branch must first reconcile its ledger before adding the new entry.

The brilliance of this system is that headquarters can confidently mark a transaction as "committed" (permanently recorded) once it receives confirmation from a majority of branches. Even if some branches are temporarily disconnected or slow, the company can continue operating as long as most branches stay synchronized. When disconnected branches reconnect, they can catch up by comparing their ledgers with headquarters and receiving any missing transactions.

This mental model captures the essence of Raft's **log matching property** - identical prefixes guarantee consistency - and explains why the leader must verify log consistency before appending new entries. Just as accounting ledgers maintain referential integrity through sequential numbering and cross-references, Raft logs maintain distributed consistency through index-term pairs and prefix matching.

### AppendEntries Protocol

The `AppendEntries` RPC serves as the primary mechanism for log replication and forms the heartbeat system that maintains leader authority. This protocol handles three critical responsibilities: replicating new log entries to followers, providing periodic heartbeats to prevent elections, and performing consistency checks to ensure log matching properties.

The `AppendEntriesRequest` message structure contains all information necessary for both replication and consistency verification:

| Field | Type | Description |
|-------|------|-------------|
| `term` | `Term` | Current term of the leader sending the request |
| `leader_id` | `NodeId` | Identity of the leader node for follower tracking |
| `prev_log_index` | `LogIndex` | Index of log entry immediately preceding new entries |
| `prev_log_term` | `Term` | Term of the entry at prev_log_index position |
| `entries` | `List[LogEntry]` | Log entries to append (empty for heartbeat) |
| `leader_commit` | `LogIndex` | Highest index known to be committed by leader |

The `prev_log_index` and `prev_log_term` fields implement the consistency check that ensures the **log matching property**. Before accepting any new entries, a follower must verify that its log entry at `prev_log_index` has term `prev_log_term`. This check guarantees that the follower's log prefix matches the leader's log up to the insertion point.

The `AppendEntriesResponse` provides the feedback mechanism for the leader to track replication progress and detect inconsistencies:

| Field | Type | Description |
|-------|------|-------------|
| `term` | `Term` | Follower's current term for leader term validation |
| `success` | `bool` | True if consistency check passed and entries were appended |
| `last_log_index` | `LogIndex` | Follower's highest log index after processing request |
| `conflict_index` | `LogIndex` | Index where log conflict was detected (if success=false) |
| `conflict_term` | `Term` | Term of conflicting entry at conflict_index |

The follower's processing algorithm follows these steps:

1. **Term Validation**: If the request term is less than the follower's current term, reject the request immediately and return the follower's term. This prevents outdated leaders from corrupting the log.

2. **Leadership Recognition**: If the request term is greater than or equal to the follower's term, update the follower's term and recognize the sender as the current leader. Reset the election timeout to prevent unnecessary elections.

3. **Log Consistency Check**: Verify that the follower has an entry at `prev_log_index` with term `prev_log_term`. If this check fails, the follower's log diverges from the leader's log and must be repaired.

4. **Conflict Resolution**: If the consistency check fails, truncate the follower's log from the point of divergence and return conflict information to help the leader find the correct insertion point.

5. **Entry Appending**: If the consistency check passes, append the new entries to the follower's log. Replace any existing conflicting entries that may exist due to previous failed leadership attempts.

6. **Commit Index Update**: Update the follower's commit index to min(leader_commit, last_log_index) and apply any newly committed entries to the state machine.

> **Critical Insight**: The AppendEntries protocol achieves both safety and liveness by coupling consistency checks with entry replication. The prev_log_index/prev_log_term check prevents log corruption, while the automatic backtracking mechanism ensures progress even when logs diverge.

When a consistency check fails, the leader employs a **backtracking algorithm** to find the point where the follower's log matches its own:

1. **Conflict Detection**: The follower returns `conflict_index` and `conflict_term` indicating where the mismatch occurred.

2. **Leader Backtrack**: The leader searches backward in its own log to find the last entry for `conflict_term`, or the end of its log if `conflict_term` doesn't exist.

3. **Retry with Earlier Index**: The leader retries the AppendEntries RPC with `prev_log_index` set to the computed backtrack position.

4. **Progressive Repair**: This process repeats until the leader finds a log position where both nodes agree, then proceeds with normal replication.

### Log Matching Property Implementation

The **log matching property** represents the fundamental invariant that ensures safety in Raft consensus. This property states that if two logs contain an entry with the same index and term, then the logs are identical in all preceding entries. This invariant enables Raft to maintain consistency without complex reconciliation protocols.

The property emerges from two key constraints enforced by the Raft protocol:

**Constraint 1: Unique Log Assignment** - A leader creates at most one entry for any given index in its term. When a leader appends an entry to its log, it assigns a unique index based on its current log length. The leader never overwrites entries from its own term.

**Constraint 2: Append-Only Semantics** - Followers accept entries only if they pass the consistency check. The `prev_log_index` and `prev_log_term` verification ensures that new entries append to an identical log prefix.

The implementation of log matching requires careful management of three critical data structures:

| Data Structure | Purpose | Consistency Requirement |
|---------------|---------|------------------------|
| `log` | Ordered sequence of entries | Immutable prefix property |
| `next_index` | Leader's tracking of follower positions | Conservative estimation |
| `match_index` | Leader's confirmation of replicated entries | Authoritative tracking |

The leader maintains per-follower tracking arrays to implement the log matching property:

- `next_index[follower]`: The index of the next log entry to send to each follower. Initially set to leader's last log index + 1.
- `match_index[follower]`: The highest log index known to be replicated on each follower. Initially set to 0.

These tracking structures enable the leader to customize AppendEntries requests for each follower's current position while maintaining the consistency guarantees:

```
Algorithm: Maintain Log Matching Property

1. Initialize next_index[follower] = last_log_index + 1 for all followers
2. Initialize match_index[follower] = 0 for all followers
3. For each AppendEntries to follower:
   a. Set prev_log_index = next_index[follower] - 1
   b. Set prev_log_term = log[prev_log_index].term
   c. Set entries = log[next_index[follower]:]
4. On successful AppendEntries response:
   a. Update match_index[follower] = prev_log_index + len(entries)
   b. Update next_index[follower] = match_index[follower] + 1
5. On failed AppendEntries response (consistency check failed):
   a. Decrement next_index[follower]
   b. Retry AppendEntries with new prev_log_index
6. Continue until consistency is achieved or follower catches up
```

The **conflict optimization** accelerates the backtracking process when followers have divergent logs. Instead of decrementing `next_index` by one entry per retry, the follower provides conflict information to help the leader jump to more promising positions:

| Conflict Scenario | Follower Response | Leader Action |
|-------------------|------------------|---------------|
| Missing Entry | `conflict_index` = follower's log length | Set `next_index` = `conflict_index` |
| Term Mismatch | `conflict_index` = first index of `conflict_term` | Find last entry with `conflict_term` in leader log |
| No Conflict Term | `conflict_term` not in leader log | Set `next_index` = `conflict_index` |

> **Design Insight**: The log matching property transforms a complex distributed consensus problem into a simpler replication problem. By ensuring identical prefixes, Raft eliminates the need for Byzantine agreement protocols or complex reconciliation logic.

### Entry Commitment Protocol

The **commitment protocol** determines when log entries become permanent and safe to apply to the state machine. This protocol balances safety (never losing committed entries) with liveness (making progress despite failures) through majority-based agreement and careful timing of commitment decisions.

An entry becomes **committed** when it has been replicated to a majority of nodes in the cluster. The leader tracks replication progress using the `match_index` array and calculates the highest index that satisfies the majority condition. However, Raft enforces an additional safety constraint: a leader can only commit entries from its current term directly, and entries from previous terms become committed indirectly when a current-term entry is committed.

The commitment algorithm proceeds through these stages:

1. **Entry Creation**: When a client request arrives, the leader appends a new entry with the current term to its local log.

2. **Replication Broadcast**: The leader sends AppendEntries RPCs to all followers containing the new entry and any preceding entries the follower might be missing.

3. **Majority Tracking**: As followers respond with successful AppendEntries acknowledgments, the leader updates the `match_index` for each follower.

4. **Commitment Calculation**: The leader calculates the highest index N where:
   - A majority of nodes have `match_index[node] >= N`
   - The entry at index N has term equal to the leader's current term

5. **Commitment Decision**: If such an index N exists and N > current `commit_index`, the leader updates its `commit_index` to N.

6. **State Machine Application**: The leader applies all entries from `last_applied + 1` to `commit_index` to its state machine.

7. **Commitment Propagation**: Future AppendEntries RPCs inform followers of the new `commit_index`, allowing them to apply the committed entries to their state machines.

The **majority calculation** requires careful handling of cluster membership and failure scenarios:

| Cluster Size | Majority Size | Tolerance | Implication |
|-------------|---------------|-----------|-------------|
| 3 nodes | 2 nodes | 1 failure | Can commit with 1 follower agreement |
| 5 nodes | 3 nodes | 2 failures | Can commit with 2 follower agreements |
| 7 nodes | 4 nodes | 3 failures | Can commit with 3 follower agreements |

> **Decision: Current-Term Commitment Restriction**
> - **Context**: Previous Raft versions allowed leaders to commit entries from previous terms directly, but this created subtle safety violations during leader changes.
> - **Options Considered**: Direct commitment of previous-term entries, indirect commitment only, hybrid approach with additional safety checks
> - **Decision**: Leaders can only commit entries from their current term directly; previous-term entries become committed indirectly
> - **Rationale**: This restriction prevents scenarios where a leader commits an entry that could later be overwritten by a different leader with a more recent log
> - **Consequences**: Slightly increases commit latency after leader changes but guarantees safety under all partition scenarios

The **state machine application** process maintains strict ordering and idempotency requirements:

```
Algorithm: Apply Committed Entries to State Machine

1. Identify newly committed entries: range from last_applied + 1 to commit_index
2. For each entry in sequential order:
   a. Verify entry integrity (checksum, term consistency)
   b. Apply entry operation to state machine
   c. Record operation result for client response
   d. Increment last_applied index
   e. Optionally trigger snapshot creation if log grows too large
3. Send responses to clients for newly committed operations
4. Update persistent state with new last_applied value
```

The commitment protocol handles several edge cases that can arise during network partitions or rapid leader changes:

**Scenario 1: Leader Failure After Replication** - If a leader crashes after replicating an entry to a majority but before updating its `commit_index`, the entry remains uncommitted. The new leader must replicate a new entry from its term to indirectly commit the previous entry.

**Scenario 2: Competing Leaders** - If network partitions create multiple leaders, only the leader with access to a majority can commit entries. When the partition heals, the minority leader's uncommitted entries will be rolled back.

**Scenario 3: Follower Lag** - Slow followers don't prevent commitment as long as a majority responds. Lagging followers catch up through normal AppendEntries processing.

### Architecture Decisions for Replication

The log replication subsystem requires several critical design decisions that affect performance, correctness, and operational characteristics. These decisions must balance conflicting requirements: low latency for client operations, high throughput for bulk data replication, strong consistency guarantees, and tolerance for network failures.

> **Decision: Batching Strategy for AppendEntries**
> - **Context**: Individual AppendEntries RPCs for each client request create excessive network overhead and limit throughput, but batching introduces latency for time-sensitive operations
> - **Options Considered**: No batching (immediate replication), time-based batching (wait up to X ms), size-based batching (wait for N entries), adaptive batching based on load
> - **Decision**: Implement adaptive batching that uses immediate replication under low load and switches to size-based batching (up to 100 entries) under high load
> - **Rationale**: Adaptive batching provides low latency for interactive operations while maximizing throughput during bulk operations; size limit prevents unbounded memory growth
> - **Consequences**: Requires more complex scheduling logic but optimizes for both latency and throughput scenarios; increases code complexity in the replication pipeline

| Batching Option | Latency | Throughput | Implementation Complexity | Memory Usage |
|----------------|---------|------------|--------------------------|--------------|
| No Batching | Lowest | Low | Simple | Constant |
| Time-Based | Medium | Medium | Medium | Bounded |
| Size-Based | Variable | High | Medium | Bounded |
| Adaptive | Optimal | High | Complex | Bounded |

> **Decision: Retry Logic and Backoff Strategy**
> - **Context**: Network failures and temporary follower unavailability require robust retry mechanisms, but aggressive retries can amplify network congestion and create cascade failures
> - **Options Considered**: Fixed interval retry, exponential backoff, linear backoff with jitter, circuit breaker pattern
> - **Decision**: Implement exponential backoff with jitter (base 100ms, max 5s) plus circuit breaker that temporarily removes failed followers from replication
> - **Rationale**: Exponential backoff prevents retry storms while jitter avoids thundering herd; circuit breaker maintains cluster progress when individual nodes fail
> - **Consequences**: Provides resilience against network issues but requires careful tuning of timeout parameters; failed nodes may fall behind and require snapshot transfer

The retry algorithm follows this progression:

```
Algorithm: Exponential Backoff with Circuit Breaker

1. Initial retry: immediate (0ms delay)
2. Subsequent retries: min(base_delay * 2^attempt + random(0, jitter), max_delay)
3. Circuit breaker triggers after consecutive_failures > threshold (default: 5)
4. During circuit breaker: exclude follower from majority calculations
5. Circuit breaker reset: attempt connection every heartbeat_interval
6. Success resets retry count and closes circuit breaker
```

> **Decision: Pipeline Depth for Concurrent Replication**
> - **Context**: Waiting for each AppendEntries RPC to complete before sending the next request creates unnecessary latency, but unlimited pipelining can overwhelm slower followers
> - **Options Considered**: Synchronous replication (no pipelining), fixed pipeline depth, adaptive pipeline based on follower response times
> - **Decision**: Implement fixed pipeline depth of 3 outstanding RPCs per follower with flow control
> - **Rationale**: Modest pipelining improves throughput without overwhelming followers; fixed depth simplifies implementation and reasoning about memory usage
> - **Consequences**: Improves replication throughput by 2-3x but complicates error handling and requires careful tracking of in-flight requests

| Pipeline Depth | Latency Impact | Memory Usage | Error Handling Complexity | Throughput Gain |
|----------------|----------------|--------------|---------------------------|-----------------|
| 1 (Synchronous) | High | Low | Simple | Baseline |
| 3 (Moderate) | Medium | Medium | Medium | 2-3x |
| 10 (Aggressive) | Low | High | Complex | 3-4x |
| Unlimited | Lowest | Unbounded | Very Complex | Diminishing returns |

> **Decision: Log Entry Verification and Integrity**
> - **Context**: Network corruption, storage errors, or implementation bugs can corrupt log entries during replication, leading to divergent state machines
> - **Options Considered**: No verification (trust network/storage), CRC32 checksums, cryptographic hashes, end-to-end integrity with client signatures
> - **Decision**: Implement CRC32 checksums for all log entries with verification on append and during catch-up operations
> - **Rationale**: CRC32 provides sufficient protection against common corruption sources with minimal computational overhead; cryptographic hashes unnecessary for integrity (not security)
> - **Consequences**: Adds 4 bytes per log entry and minor CPU overhead but prevents silent corruption from propagating through the cluster

The integrity verification process integrates with the normal replication flow:

1. **Entry Creation**: Leader calculates CRC32 checksum when appending entry to local log
2. **Replication**: Checksum included in AppendEntries RPC alongside entry data
3. **Follower Verification**: Follower recalculates checksum and rejects entry if mismatch
4. **Storage Verification**: Checksum verified when reading entries from persistent storage
5. **Corruption Recovery**: Corrupted entries trigger log repair through leader replication

### Common Replication Pitfalls

Understanding the common mistakes in log replication implementation helps avoid subtle bugs that can compromise safety or liveness. These pitfalls often arise from misunderstanding the timing requirements, state management complexities, or edge cases in the Raft protocol.

⚠️ **Pitfall: Concurrent AppendEntries Processing**

Many implementations incorrectly handle concurrent AppendEntries RPCs to the same follower, leading to out-of-order processing and log corruption. This typically happens when the leader sends multiple AppendEntries requests in quick succession (pipelining), but the follower processes responses out of order due to network reordering or varying processing times.

**Why it's wrong**: Raft requires strictly sequential log appending. If AppendEntries RPC #2 (covering indices 100-105) completes before RPC #1 (covering indices 95-99), the follower's log becomes inconsistent. The follower might accept entries 100-105 even though entries 95-99 are missing, violating the log matching property.

**How to fix**: Implement per-follower sequence numbers or use a single-threaded queue for AppendEntries processing per follower. The leader should wait for RPC #1 to complete before sending RPC #2, or the follower should buffer out-of-order responses and apply them sequentially.

```
Correct Implementation Pattern:
1. Leader maintains per-follower send queue (FIFO)
2. Send next AppendEntries only after previous RPC completes
3. Alternative: Follower buffers responses and applies in sequence order
```

⚠️ **Pitfall: Premature Commitment Based on Stale Acknowledgments**

Implementers sometimes count acknowledgments from previous AppendEntries RPCs toward commitment of newer entries, leading to safety violations. This occurs when a follower's successful response to an old RPC arrives after newer entries have been replicated, and the leader incorrectly includes this follower in the majority count for the newer entries.

**Why it's wrong**: The follower might not actually have the newer entries, so counting its vote toward commitment can lead to committing entries that aren't replicated to a majority. If the leader crashes before the follower receives the newer entries, the committed entry could be lost.

**How to fix**: Associate each AppendEntries RPC with a unique identifier and only count acknowledgments for the specific entries being committed. Update `match_index` only when receiving acknowledgments for the corresponding AppendEntries RPC.

| Problem | Symptom | Root Cause | Solution |
|---------|---------|------------|----------|
| Stale ACK counting | Entries committed without majority | Mismatched RPC responses | RPC sequence numbers |
| Out-of-order processing | Log gaps or duplicates | Concurrent RPC handling | Sequential processing |
| Race in commitment | Inconsistent commit indices | Shared state mutation | Atomic operations |

⚠️ **Pitfall: Incorrect Handling of Term Updates During Replication**

A common mistake involves continuing with AppendEntries processing even after discovering that the follower has a higher term. This can lead to log corruption when a node that should become a follower continues acting as a leader.

**Why it's wrong**: If a follower responds with a higher term, the sender is no longer a valid leader and must immediately step down. Continuing to process the AppendEntries response or attempting further replication can violate the single-leader-per-term invariant.

**How to fix**: Check the response term immediately upon receiving any RPC response. If the response term is higher than the current term, immediately update the local term, convert to follower state, and abandon all leader activities including the current AppendEntries processing.

⚠️ **Pitfall: Log Truncation Without Proper Conflict Resolution**

Implementations sometimes truncate follower logs incorrectly when conflicts are detected, either truncating too aggressively (losing valid entries) or not truncating enough (leaving conflicting entries).

**Why it's wrong**: Incorrect truncation can either lose entries that should be preserved (causing unnecessary re-replication) or leave conflicting entries that violate log consistency. Both cases can lead to permanent log divergence or safety violations.

**How to fix**: When a conflict is detected at index N, truncate the log to include only entries up to index N-1. This preserves all entries that are known to be consistent while removing all entries that might conflict with the leader's log.

```
Correct Truncation Algorithm:
1. Detect conflict at prev_log_index with different term
2. Find highest index < prev_log_index where terms match
3. Truncate log to include only entries up to that index
4. Reject current AppendEntries and request retry from leader
```

⚠️ **Pitfall: Inconsistent Commit Index Updates**

Many implementations update the commit index incorrectly, either advancing it too aggressively (beyond what's actually replicated) or too conservatively (preventing progress). This often happens when the majority calculation includes failed nodes or excludes responsive nodes.

**Why it's wrong**: Advancing commit index beyond actual majority replication can commit entries that might be lost if the leader fails. Being too conservative prevents clients from seeing committed operations, reducing system availability.

**How to fix**: Calculate the commit index as the highest index N where:
- At least (cluster_size/2 + 1) nodes have match_index >= N
- The entry at index N has the current leader's term
- N > current commit_index

| Error Type | Symptom | Impact | Prevention |
|------------|---------|--------|------------|
| Aggressive commit | Safety violation | Data loss on leader failure | Strict majority verification |
| Conservative commit | Liveness issue | Delayed client responses | Accurate match_index tracking |
| Wrong term check | Safety violation | Commits overridden by new leader | Current-term-only commitment |

⚠️ **Pitfall: Network Partition Handling in Replication**

Implementations often fail to handle network partitions correctly, either continuing to commit in the minority partition or failing to make progress when they should be able to.

**Why it's wrong**: A leader in the minority partition cannot maintain the majority requirement for safe commitment, while a leader in the majority partition should continue operating normally. Mishandling this can lead to split-brain scenarios or unnecessary unavailability.

**How to fix**: Continuously monitor the number of responsive followers. If fewer than majority respond within the heartbeat timeout, stop committing new entries but continue sending heartbeats to detect partition healing. Resume normal operation when majority connectivity is restored.

### Implementation Guidance

**A. Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| RPC Transport | HTTP with JSON requests | gRPC with Protocol Buffers |
| Logging | Python logging with structured format | Structured logging with OpenTelemetry |
| Persistence | Direct file I/O with fsync | Embedded database (SQLite/LevelDB) |
| Concurrency | Threading with locks | Async/await with asyncio |
| Serialization | JSON for human readability | MessagePack for efficiency |

**B. Recommended File Structure:**

```
titan/
  consensus/
    __init__.py
    node.py                 ← main RaftNode class
    replication.py         ← AppendEntries protocol (this milestone)
    log.py                 ← log management utilities
    rpc.py                 ← network communication layer
    state.py               ← persistent and volatile state
    types.py               ← data structure definitions
  storage/
    __init__.py
    log_storage.py         ← persistent log storage
    state_storage.py       ← persistent state storage
  tests/
    test_replication.py    ← replication-specific tests
    test_integration.py    ← end-to-end scenarios
```

**C. Infrastructure Starter Code:**

Complete RPC message definitions:

```python
# types.py - Complete message type definitions
from dataclasses import dataclass
from typing import List, Optional
import time
import json

@dataclass
class LogEntry:
    term: int
    index: int
    data: bytes
    timestamp: float
    
    def to_dict(self) -> dict:
        return {
            'term': self.term,
            'index': self.index,
            'data': self.data.hex(),  # Convert bytes to hex string
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'LogEntry':
        return cls(
            term=data['term'],
            index=data['index'],
            data=bytes.fromhex(data['data']),
            timestamp=data['timestamp']
        )

@dataclass
class AppendEntriesRequest:
    term: int
    leader_id: str
    prev_log_index: int
    prev_log_term: int
    entries: List[LogEntry]
    leader_commit: int
    
    def to_dict(self) -> dict:
        return {
            'term': self.term,
            'leader_id': self.leader_id,
            'prev_log_index': self.prev_log_index,
            'prev_log_term': self.prev_log_term,
            'entries': [entry.to_dict() for entry in self.entries],
            'leader_commit': self.leader_commit
        }

@dataclass
class AppendEntriesResponse:
    term: int
    success: bool
    last_log_index: int
    conflict_index: int
    conflict_term: int
    
    def to_dict(self) -> dict:
        return {
            'term': self.term,
            'success': self.success,
            'last_log_index': self.last_log_index,
            'conflict_index': self.conflict_index,
            'conflict_term': self.conflict_term
        }
```

Complete RPC transport layer:

```python
# rpc.py - Simple HTTP-based RPC transport
import requests
import json
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class RPCTransport:
    def __init__(self, node_id: str, port: int, timeout: float = 1.0):
        self.node_id = node_id
        self.port = port
        self.timeout = timeout
        self.peer_addresses: Dict[str, str] = {}
    
    def register_peer(self, peer_id: str, address: str, port: int):
        """Register the network address for a peer node."""
        self.peer_addresses[peer_id] = f"http://{address}:{port}"
    
    def send_append_entries(self, target: str, request: AppendEntriesRequest) -> Optional[AppendEntriesResponse]:
        """Send AppendEntries RPC to target node."""
        if target not in self.peer_addresses:
            logger.error(f"Unknown peer: {target}")
            return None
        
        url = f"{self.peer_addresses[target]}/append_entries"
        try:
            response = requests.post(
                url,
                json=request.to_dict(),
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            if response.status_code == 200:
                data = response.json()
                return AppendEntriesResponse(
                    term=data['term'],
                    success=data['success'],
                    last_log_index=data['last_log_index'],
                    conflict_index=data['conflict_index'],
                    conflict_term=data['conflict_term']
                )
            else:
                logger.warning(f"AppendEntries to {target} failed: {response.status_code}")
                return None
        except requests.RequestException as e:
            logger.warning(f"Network error sending AppendEntries to {target}: {e}")
            return None
    
    def start_server(self, handler_callback):
        """Start HTTP server to receive RPCs. Uses Flask for simplicity."""
        from flask import Flask, request, jsonify
        
        app = Flask(__name__)
        app.logger.disabled = True  # Disable Flask logging
        
        @app.route('/append_entries', methods=['POST'])
        def append_entries():
            try:
                data = request.get_json()
                # Convert dict back to AppendEntriesRequest
                entries = [LogEntry.from_dict(e) for e in data['entries']]
                req = AppendEntriesRequest(
                    term=data['term'],
                    leader_id=data['leader_id'],
                    prev_log_index=data['prev_log_index'],
                    prev_log_term=data['prev_log_term'],
                    entries=entries,
                    leader_commit=data['leader_commit']
                )
                
                response = handler_callback.handle_append_entries(req)
                return jsonify(response.to_dict())
            except Exception as e:
                logger.error(f"Error handling AppendEntries: {e}")
                return jsonify({'error': str(e)}), 500
        
        # Run server in background thread
        import threading
        server_thread = threading.Thread(
            target=lambda: app.run(host='0.0.0.0', port=self.port, debug=False)
        )
        server_thread.daemon = True
        server_thread.start()
        logger.info(f"RPC server started on port {self.port}")
```

**D. Core Logic Skeleton Code:**

The main replication logic that students need to implement:

```python
# replication.py - Core replication logic to be implemented by students
from typing import Dict, List, Optional, Tuple
import threading
import time
import logging
from .types import *
from .rpc import RPCTransport

logger = logging.getLogger(__name__)

class ReplicationManager:
    """Manages log replication to followers. Students implement the core logic."""
    
    def __init__(self, node_id: str, transport: RPCTransport):
        self.node_id = node_id
        self.transport = transport
        self.next_index: Dict[str, int] = {}    # Next log index to send to each follower
        self.match_index: Dict[str, int] = {}   # Highest log index replicated to each follower
        self.replication_lock = threading.Lock()
        
    def handle_append_entries(self, request: AppendEntriesRequest) -> AppendEntriesResponse:
        """
        Process incoming AppendEntries RPC as a follower.
        
        This is the core of log replication safety. Must implement:
        1. Term validation and leadership recognition
        2. Log consistency checking with prev_log_index/prev_log_term
        3. Conflict detection and resolution
        4. Entry appending and commit index updates
        
        Args:
            request: AppendEntries RPC from leader
            
        Returns:
            Response indicating success/failure and conflict information
        """
        # TODO 1: Validate request term against current term
        # - If request.term < current_term: reject immediately
        # - If request.term >= current_term: update term and recognize leader
        # - Reset election timeout to prevent unnecessary election
        
        # TODO 2: Perform log consistency check
        # - Verify follower has entry at prev_log_index with term prev_log_term
        # - If check fails: return conflict information for leader backtracking
        # - Use conflict optimization: return conflict_index and conflict_term
        
        # TODO 3: Handle conflicting entries
        # - If existing entry conflicts with new entry: truncate log from conflict point
        # - Only truncate when actual conflict detected, not speculatively
        
        # TODO 4: Append new entries to log
        # - Add all entries from request.entries to follower's log
        # - Ensure entries are appended in correct order
        # - Update last_log_index appropriately
        
        # TODO 5: Update commit index and apply entries
        # - Set commit_index = min(request.leader_commit, last_log_index)
        # - Apply newly committed entries to state machine
        # - Update last_applied index after successful application
        
        # TODO 6: Construct and return response
        # - success=True if consistency check passed and entries appended
        # - Include current term and updated last_log_index
        # - Include conflict information if consistency check failed
        
        pass  # Students implement this
    
    def replicate_to_followers(self, followers: List[str], log_entries: List[LogEntry], 
                              commit_index: int) -> bool:
        """
        Replicate log entries to all followers as leader.
        
        Implements the leader's side of log replication:
        1. Determine appropriate prev_log_index for each follower
        2. Send AppendEntries RPC with consistency check
        3. Handle responses and update tracking indices
        4. Retry with backtracking on conflicts
        
        Args:
            followers: List of follower node IDs
            log_entries: Current log state for replication
            commit_index: Current leader's commit index
            
        Returns:
            True if majority of followers acknowledged replication
        """
        # TODO 1: Initialize or validate next_index and match_index for all followers
        # - Set next_index[follower] = last_log_index + 1 for new followers
        # - Ensure match_index[follower] starts at 0 for new followers
        
        # TODO 2: Send AppendEntries RPC to each follower concurrently
        # - Calculate prev_log_index = next_index[follower] - 1
        # - Calculate prev_log_term = log_entries[prev_log_index].term
        # - Include entries from next_index[follower] to end of log
        # - Set leader_commit to current commit_index
        
        # TODO 3: Process AppendEntries responses
        # - If response.term > current_term: step down to follower immediately
        # - If success=True: update match_index and next_index for follower
        # - If success=False: implement backtracking algorithm using conflict info
        
        # TODO 4: Implement conflict resolution backtracking
        # - Use conflict_index and conflict_term for efficient backtracking
        # - Retry AppendEntries with adjusted prev_log_index
        # - Continue until consistency achieved or timeout
        
        # TODO 5: Calculate majority acknowledgment
        # - Count followers where match_index >= target log index
        # - Include leader itself in majority calculation
        # - Return True if majority (cluster_size/2 + 1) achieved
        
        # Hint: Use threading for concurrent RPC sends but protect shared state with locks
        # Hint: Implement exponential backoff for failed RPCs
        
        pass  # Students implement this
    
    def calculate_commit_index(self, log_entries: List[LogEntry], current_term: int, 
                              current_commit: int) -> int:
        """
        Calculate the highest log index that can be safely committed.
        
        Implements Raft's commitment rule:
        - Only commit entries from current term directly
        - Require majority replication before commitment
        - Never decrease commit index
        
        Args:
            log_entries: Current log state
            current_term: Leader's current term
            current_commit: Current commit index
            
        Returns:
            New commit index (>= current_commit)
        """
        # TODO 1: Find highest index N where majority has match_index >= N
        # - Sort match_index values to find median (majority threshold)
        # - Identify highest index replicated to majority
        
        # TODO 2: Verify current-term restriction
        # - Only commit entries with term == current_term
        # - Previous-term entries committed indirectly
        
        # TODO 3: Ensure monotonic increase
        # - New commit index must be >= current_commit
        # - Never move commit index backward
        
        # TODO 4: Validate log bounds
        # - Ensure commit index doesn't exceed log length
        # - Handle empty log case appropriately
        
        pass  # Students implement this
```

**E. Language-Specific Hints:**

- Use `threading.Lock()` to protect shared state like `next_index` and `match_index` during concurrent replication
- Implement RPC timeouts with `requests.post(timeout=1.0)` to handle network failures gracefully
- Use `logging` module with structured format for debugging replication issues: `logger.info("AppendEntries to %s: success=%s", follower_id, response.success)`
- Handle JSON serialization carefully for `bytes` fields - convert to hex strings for network transport
- Use `time.time()` for timestamps and timeout calculations in election and heartbeat logic
- Implement proper error handling for network exceptions: `requests.RequestException`, `json.JSONDecodeError`

**F. Milestone Checkpoint:**

After implementing log replication, verify the following behavior:

1. **Start a 3-node cluster** and verify heartbeat messages:
```bash
python -m titan.node --id=node1 --port=8001 --peers=node2:8002,node3:8003
python -m titan.node --id=node2 --port=8002 --peers=node1:8001,node3:8003
python -m titan.node --id=node3 --port=8003 --peers=node1:8001,node2:8002
```

2. **Verify leader election** and check log replication:
   - Wait for leader election (check logs for "became leader" message)
   - Send client request to leader: `curl -X POST http://localhost:8001/client -d '{"data": "test"}'`
   - Verify entry appears in all node logs
   - Expected: All nodes show same log index and term for the entry

3. **Test replication consistency**:
   - Send multiple requests rapidly to leader
   - Verify all nodes have identical logs: `curl http://localhost:8001/debug/log`
   - Expected: Same sequence of entries with identical terms and indices

4. **Test follower failure recovery**:
   - Kill one follower node (Ctrl+C)
   - Send requests to leader (should continue working with 2/3 nodes)
   - Restart killed follower
   - Expected: Follower catches up and matches leader's log

Signs something is wrong and what to check:
- **Split votes in logs**: Check election timeout randomization
- **Log inconsistencies**: Verify prev_log_index/prev_log_term checking
- **Replication failures**: Check RPC timeouts and network connectivity
- **Commit index not advancing**: Verify majority calculation in commitment protocol


## Log Compaction (Milestone 3)

> **Milestone(s):** Milestone 3: Log Compaction - Manage log growth with snapshots and state machine installs

### Mental Model: Photo Albums

Think of log compaction like organizing your digital photo collection. Over time, you accumulate thousands of photos on your phone, making it slow and consuming storage. To solve this, you periodically create photo albums that capture key memories and delete the individual photos that led to those moments. The album preserves the important state (your memories) while discarding the intermediate steps (all the individual shots you took to get the perfect picture).

In Raft, the **log** is like your photo collection - it grows unbounded as the system processes operations. A **snapshot** is like a photo album - it captures the current state of your application (the important outcome) and allows you to discard the individual log entries (the intermediate operations) that led to that state. When a new node joins or falls behind, instead of replaying thousands of individual operations, you can transfer the snapshot (like sharing a photo album) to quickly bring them up to the current state.

The critical insight is that snapshots compress history while preserving correctness. Just as a photo album captures the essence of a vacation without needing every individual photo, a snapshot captures the current state of the state machine without needing every log entry that created that state.

![Log Compaction Process](./diagrams/compaction-flowchart.svg)

### Incremental Snapshot Creation

Creating snapshots is fundamentally about **state machine extraction** - capturing the current state of your application in a format that can be persisted and later restored. The challenge is doing this safely without blocking normal consensus operations or creating inconsistent snapshots during concurrent modifications.

The **snapshot trigger mechanism** determines when to create snapshots. Titan uses a simple threshold-based approach: when the log grows beyond `SNAPSHOT_THRESHOLD` entries since the last snapshot, the leader initiates snapshot creation. This prevents unbounded log growth while amortizing snapshot overhead across many operations.

| Trigger Condition | Threshold | Rationale |
|-------------------|-----------|-----------|
| Log Entry Count | 10,000 entries | Balances memory usage with snapshot frequency |
| Time-Based | Not used | Log growth rate varies too much across workloads |
| Size-Based | Not implemented | Entry size varies significantly in practice |
| Memory Pressure | Not implemented | Would require complex memory monitoring |

> **Decision: Threshold-Based Snapshot Triggering**
> - **Context**: Need to determine when to create snapshots to manage log growth
> - **Options Considered**: Entry count threshold, time-based intervals, memory pressure detection
> - **Decision**: Use fixed entry count threshold of 10,000 entries
> - **Rationale**: Simple to implement and reason about, provides predictable memory bounds, works well across different workload patterns
> - **Consequences**: May create snapshots too frequently for small entries or too infrequently for large entries, but provides good baseline behavior

The **snapshot creation process** follows these steps to ensure consistency and safety:

1. **State Machine Freeze**: The leader temporarily stops applying new log entries to the state machine to ensure a consistent point-in-time snapshot
2. **State Extraction**: The state machine serializes its current state into a snapshot format, including all application data and metadata
3. **Snapshot Metadata Creation**: Record the `last_included_index` and `last_included_term` that correspond to the last log entry reflected in the snapshot
4. **Atomic Snapshot Installation**: Atomically write the snapshot to persistent storage with its metadata before resuming normal operations
5. **Log Entry Marking**: Mark log entries up to `last_included_index` as eligible for truncation (but don't truncate immediately)
6. **State Machine Resume**: Resume applying new log entries that arrived during snapshot creation

The **incremental snapshot format** balances simplicity with efficiency. Each snapshot is self-contained and includes everything needed to restore the state machine:

| Snapshot Component | Content | Purpose |
|-------------------|---------|---------|
| Snapshot Metadata | `last_included_index`, `last_included_term`, `creation_timestamp` | Identifies what log entries the snapshot represents |
| State Machine Data | Application-specific serialized state | The actual data needed to restore the state machine |
| Checksum | SHA-256 hash of all snapshot data | Detects corruption during storage or transfer |

> The key insight for snapshot consistency is that we must freeze the state machine during extraction. Unlike databases with MVCC, Raft state machines typically don't support point-in-time reads, so we need an explicit consistency point.

**Snapshot storage strategy** uses a simple file-based approach with atomic replacement:

1. **Temporary Snapshot Creation**: Write the new snapshot to a temporary file (`snapshot.tmp`)
2. **Checksum Validation**: Calculate and verify the snapshot checksum before finalizing
3. **Atomic Replacement**: Use atomic file rename (`snapshot.tmp` → `snapshot.current`) to replace the previous snapshot
4. **Old Snapshot Cleanup**: Remove the previous snapshot file after successful replacement

This approach ensures that snapshot creation never leaves the system in an inconsistent state - either the old snapshot exists or the new one does, never a partial or corrupted snapshot.

**Concurrent operation handling** is critical during snapshot creation. While the state machine is frozen, the Raft leader continues accepting new log entries and replicating them to followers. These operations are queued and applied after snapshot creation completes:

| Operation Type | Behavior During Snapshot | Justification |
|----------------|-------------------------|---------------|
| New Client Requests | Accepted and logged normally | Maintains system availability |
| Log Replication | Continues to followers | Ensures cluster consistency |
| State Machine Apply | Paused until snapshot complete | Prevents snapshot inconsistency |
| Heartbeats | Continue normally | Maintains leader authority |

### InstallSnapshot RPC Protocol

The `InstallSnapshot` RPC enables efficient transfer of snapshots to followers who have fallen far behind or to new nodes joining the cluster. Unlike normal log replication which sends incremental entries, snapshot installation provides a "fast forward" mechanism to bring nodes up to a recent state.

**InstallSnapshot message structure** supports chunked transfer to handle large snapshots that exceed network message limits:

| Field | Type | Purpose |
|-------|------|---------|
| `term` | `int` | Current leader's term for safety checks |
| `leader_id` | `str` | Identifies the snapshot source |
| `last_included_index` | `int` | Highest log index represented in snapshot |
| `last_included_term` | `int` | Term of the last included log entry |
| `offset` | `int` | Byte offset of this chunk within the snapshot |
| `data` | `bytes` | Snapshot data chunk (may be partial) |
| `done` | `bool` | True if this is the final chunk |

The **chunked transfer protocol** handles snapshots that are too large for single network messages:

1. **Chunk Size Calculation**: Leader divides snapshot into chunks (typically 64KB each) based on network MTU and message size limits
2. **Sequential Transfer**: Send chunks in order starting from offset 0, waiting for acknowledgment before sending the next chunk
3. **Chunk Assembly**: Follower accumulates chunks in order, rejecting out-of-order or duplicate chunks
4. **Completion Verification**: After the final chunk (marked with `done=true`), follower verifies the complete snapshot checksum
5. **Atomic Installation**: Only install the snapshot if all chunks were received and checksum verification passes

**InstallSnapshot response handling** is simpler than AppendEntries since snapshots represent a complete state rather than incremental changes:

| Response Field | Type | Purpose |
|----------------|------|---------|
| `term` | `int` | Follower's current term for leader validation |

The follower's response processing follows this logic:

1. **Term Validation**: If the follower's term is higher than the request term, reject the snapshot and return the higher term
2. **Chunk Processing**: If term is valid, accept the chunk and send acknowledgment
3. **Installation Completion**: After final chunk, install snapshot and truncate conflicting log entries

**Snapshot installation safety** requires careful handling of the follower's existing log. The snapshot represents a specific point in the log history, so any log entries that conflict with this point must be removed:

1. **Log Truncation Point**: If follower has log entries beyond `last_included_index`, compare terms at that position
2. **Conflict Detection**: If the follower's log entry at `last_included_index` has a different term than `last_included_term`, truncate all entries from that point forward
3. **Snapshot Installation**: Replace the state machine state with the snapshot data
4. **Index Update**: Set `commit_index` and `last_applied` to `last_included_index`

> **Decision: Chunked Snapshot Transfer**
> - **Context**: Snapshots can be much larger than typical network message limits
> - **Options Considered**: Single large message, chunked transfer, separate file transfer protocol
> - **Decision**: Use chunked transfer within existing RPC framework
> - **Rationale**: Reuses existing RPC infrastructure, provides flow control, handles network failures gracefully
> - **Consequences**: Adds complexity to message handling but keeps transfer logic within Raft protocol

**InstallSnapshot trigger conditions** determine when leaders should send snapshots instead of regular AppendEntries:

| Condition | Action | Rationale |
|-----------|--------|-----------|
| Follower needs entry before snapshot | Send InstallSnapshot | Required entries no longer exist in log |
| Follower is very far behind | Send InstallSnapshot | More efficient than sending many log entries |
| New node joining cluster | Send InstallSnapshot | Fastest way to bring node to current state |

The **leader's InstallSnapshot logic** integrates with normal replication:

1. **Gap Detection**: When sending AppendEntries, if the required `prev_log_index` is before the snapshot's `last_included_index`, switch to InstallSnapshot
2. **Snapshot Transmission**: Send snapshot chunks until complete, then resume normal AppendEntries from the snapshot point
3. **Follower State Tracking**: Track each follower's snapshot installation progress separately from their log replication state

### Safe Log Truncation

Log truncation is the final step of compaction - removing old log entries that are no longer needed because they're represented in a snapshot. This must be done carefully to maintain safety invariants and ensure the cluster can continue operating correctly.

**Truncation safety conditions** must all be met before removing log entries:

| Safety Condition | Requirement | Violation Consequence |
|------------------|-------------|----------------------|
| Snapshot Persistence | Snapshot successfully written to disk | Loss of committed state if node restarts |
| Majority Confirmation | Majority of cluster has snapshot or equivalent log entries | Split-brain during leader election |
| Leader Authority | Only current leader performs truncation | Inconsistent truncation across cluster |
| Application Safety | All truncated entries were applied to state machine | Loss of committed operations |

The **truncation algorithm** ensures these safety conditions are met:

1. **Snapshot Verification**: Confirm the snapshot was successfully created and persisted with correct checksums
2. **Follower Status Check**: Verify that a majority of followers have either installed this snapshot or have log entries up to `last_included_index`
3. **Application Confirmation**: Ensure all log entries up to `last_included_index` have been applied to the local state machine
4. **Atomic Log Update**: Remove all log entries with index ≤ `last_included_index` in a single atomic operation
5. **Index Adjustment**: Update internal bookkeeping to account for the new log starting point

**Majority confirmation tracking** prevents premature truncation that could compromise safety:

```
Truncation Decision Matrix:
- Node A (Leader): Has snapshot + can truncate
- Node B (Follower): Has snapshot → Can proceed with truncation  
- Node C (Follower): Has log entries up to snapshot point → Can proceed
- Node D (Follower): Missing recent entries → Must wait for D to catch up
- Node E (Down): Ignore for majority calculation

Majority = 3/5 nodes confirmed → Safe to truncate
```

> The critical safety insight is that we can only truncate log entries if we're certain a majority of the cluster can reconstruct that state either from snapshots or existing log entries. Premature truncation could make it impossible to elect a new leader.

**Log index remapping** handles the complexity of maintaining log semantics after truncation. After removing entries 1-10000, the log now starts at index 10001, but we need to maintain the illusion of a contiguous log:

| Original Index | After Truncation | Access Method |
|----------------|------------------|---------------|
| 1-10000 | Represented in snapshot | State machine query |
| 10001 | First log entry (index 10001) | Direct log access |
| 10002+ | Subsequent entries | Direct log access |

**Truncation implementation** uses atomic file operations to ensure consistency:

1. **Backup Creation**: Create a backup of the current log file before truncation
2. **New Log Creation**: Write a new log file containing only entries after `last_included_index`
3. **Atomic Replacement**: Atomically replace the old log file with the new truncated log
4. **Backup Cleanup**: Remove the backup after successful truncation
5. **Index Update**: Update all internal pointers and indices to reflect the new log structure

**Error handling during truncation** must preserve system safety:

| Error Scenario | Detection | Recovery Action |
|----------------|-----------|----------------|
| Disk full during truncation | Write failure | Abort truncation, keep original log |
| Corruption in new log file | Checksum mismatch | Restore from backup, retry later |
| Process crash during truncation | Startup detection | Complete truncation or restore backup |
| Network partition during status check | Timeout or connection failure | Delay truncation until connectivity restored |

> **Decision: Conservative Truncation Strategy**
> - **Context**: Need to balance log space usage with safety guarantees
> - **Options Considered**: Aggressive truncation after snapshot creation, conservative majority confirmation, periodic background truncation
> - **Decision**: Only truncate after majority confirmation with periodic retry
> - **Rationale**: Prioritizes safety over immediate space reclamation, prevents split-brain scenarios
> - **Consequences**: May use more disk space temporarily but ensures cluster can always elect a valid leader

### Architecture Decisions for Compaction

The design of log compaction involves several critical trade-offs that affect performance, safety, and implementation complexity. Each decision shapes how the system behaves under different workloads and failure scenarios.

> **Decision: Leader-Only Snapshot Creation**
> - **Context**: Need to determine which nodes can create and initiate snapshots
> - **Options Considered**: Leader-only creation, follower-initiated snapshots, distributed snapshot creation
> - **Decision**: Only the current leader creates and distributes snapshots
> - **Rationale**: Simplifies coordination, ensures snapshot consistency, leverages leader's authoritative state
> - **Consequences**: Leader bears additional computational load, but eliminates snapshot coordination complexity

| Snapshot Creation Strategy | Coordination Complexity | Resource Distribution | Consistency Guarantees |
|----------------------------|------------------------|----------------------|----------------------|
| Leader-Only | Low - leader decides when/how | High leader load | Strong - single authority |
| Follower-Initiated | Medium - requires leader approval | Distributed load | Medium - needs validation |
| Distributed | High - requires consensus | Even distribution | Weak - multiple snapshot versions |

> **Decision: Synchronous State Machine Freeze**
> - **Context**: Need to ensure snapshot consistency during state machine extraction
> - **Options Considered**: Synchronous freeze, asynchronous copy-on-write, versioned state machine
> - **Decision**: Temporarily freeze state machine during snapshot creation
> - **Rationale**: Guarantees perfect consistency, simple to implement correctly, bounded impact duration
> - **Consequences**: Brief availability impact during snapshotting, but eliminates complex consistency mechanisms

**Snapshot frequency trade-offs** balance resource usage against operational benefits:

| Frequency Strategy | Disk I/O Impact | Memory Usage | Recovery Speed | Implementation |
|-------------------|----------------|---------------|----------------|----------------|
| Fixed Threshold (10K entries) | Predictable | Bounded | Fast for large gaps | Simple |
| Time-Based Intervals | Variable | Variable | Inconsistent | Medium complexity |
| Adaptive Based on Growth Rate | Optimized | Optimized | Fast | High complexity |

> **Decision: Fixed Entry Count Threshold**
> - **Context**: Determine optimal snapshot frequency strategy
> - **Options Considered**: Fixed count, adaptive rate, memory pressure, time intervals
> - **Decision**: Create snapshot every 10,000 log entries
> - **Rationale**: Provides predictable behavior, simple implementation, works well across workload types
> - **Consequences**: May not be optimal for all workloads but provides reliable baseline performance

**Snapshot format decisions** affect portability, efficiency, and debugging capabilities:

| Format Aspect | Chosen Approach | Alternative | Rationale |
|---------------|-----------------|-------------|-----------|
| Serialization | Application-defined with metadata wrapper | Built-in format (JSON/Protobuf) | Allows optimal app-specific encoding |
| Compression | Optional per-application | Always compressed | Apps can choose based on data characteristics |
| Checksumming | SHA-256 of complete snapshot | Per-chunk checksums | Simpler validation, adequate security |
| Metadata | Separate header section | Embedded in data | Enables fast metadata access without reading entire snapshot |

> **Decision: Chunked Transfer with Flow Control**
> - **Context**: Handle snapshots larger than network message limits
> - **Options Considered**: Single large transfer, chunked with flow control, separate file transfer
> - **Decision**: 64KB chunks with sequential acknowledgment-based flow control
> - **Rationale**: Balances transfer efficiency with memory usage, provides automatic retry capability
> - **Consequences**: Adds message complexity but enables handling arbitrarily large snapshots reliably

**Failure recovery strategies** during compaction operations:

| Failure Point | Recovery Strategy | Data Consistency Impact |
|---------------|------------------|------------------------|
| During snapshot creation | Abort and retry later | No impact - original log intact |
| During chunk transfer | Resume from last successful chunk | No impact - follower validates complete snapshot |
| During log truncation | Complete operation or restore backup | No impact - atomic truncation design |
| During follower installation | Retry InstallSnapshot from beginning | No impact - follower keeps original state until success |

> **Decision: Optimistic Truncation with Validation**
> - **Context**: Determine when it's safe to remove old log entries after snapshotting
> - **Options Considered**: Immediate truncation, majority confirmation, pessimistic delay
> - **Decision**: Truncate only after majority of cluster has snapshot or equivalent log entries
> - **Rationale**: Prevents scenarios where new leader cannot be elected due to missing log history
> - **Consequences**: Uses more disk space temporarily but guarantees cluster availability

### Common Compaction Pitfalls

Log compaction introduces subtle timing and consistency challenges that can lead to data loss, split-brain scenarios, or system unavailability. Understanding these pitfalls helps avoid critical bugs during implementation.

⚠️ **Pitfall: Premature Log Truncation**

The most dangerous compaction error is truncating log entries before ensuring the cluster can safely operate without them. This typically happens when a developer truncates immediately after creating a snapshot, without verifying cluster state.

**What happens**: After truncation, if the current leader fails and a new election begins, the remaining nodes might not have enough log history to participate effectively. In extreme cases, if the snapshot is corrupted or lost, the truncated state becomes unrecoverable.

**Example scenario**: Leader creates snapshot of entries 1-10000 and immediately truncates. Leader then crashes. Remaining followers have entries 1-8000. No node has a complete picture of the state through entry 10000, making it impossible to safely elect a new leader or reconstruct the complete state.

**Prevention**: Always confirm that a majority of cluster nodes have either the snapshot or equivalent log entries before truncating. Implement a confirmation phase where the leader tracks follower progress and only truncates when safety conditions are met.

⚠️ **Pitfall: State Machine Inconsistency During Snapshots**

Creating snapshots while the state machine continues processing operations can capture an inconsistent intermediate state. This is particularly tricky with multi-step operations or when the state machine has complex invariants.

**What happens**: The snapshot might capture a state where some effects of an operation are visible but others aren't, creating a snapshot that represents a state the system never actually experienced.

**Example scenario**: A bank transfer operation moves $100 from Account A to Account B. If snapshotting occurs between debiting A and crediting B, the snapshot shows $100 disappeared from the system entirely.

**Prevention**: Implement proper state machine freezing during snapshot creation. Either pause all state machine modifications or use copy-on-write mechanisms that guarantee point-in-time consistency.

⚠️ **Pitfall: Incomplete Snapshot Transfer Handling**

Failing to properly handle chunked snapshot transfer interruptions can leave followers in inconsistent states or cause infinite retry loops.

**What happens**: Network failures during snapshot transfer can leave followers with partial snapshots. If not handled correctly, followers might apply partial state or leaders might not properly detect and retry failed transfers.

**Example scenario**: InstallSnapshot sends 50 out of 100 chunks, then network fails. Follower has partial data but leader assumes transfer succeeded. Follower can't reconstruct complete state and falls permanently behind.

**Prevention**: Implement robust chunk tracking with checksums and completion verification. Followers should only install snapshots after receiving and validating all chunks. Leaders should track transfer state per follower and retry failed transfers.

⚠️ **Pitfall: Snapshot Metadata Inconsistency**

Mismatching snapshot content with metadata (especially `last_included_index` and `last_included_term`) creates subtle consistency violations that may not manifest immediately.

**What happens**: If snapshot metadata doesn't accurately reflect the log entries represented in the snapshot, future log matching and conflict resolution can make incorrect decisions.

**Example scenario**: Snapshot actually represents state through entry 10000 term 5, but metadata claims it represents entry 10000 term 4. Later log matching decisions based on this metadata can incorrectly accept or reject log entries.

**Prevention**: Generate snapshot metadata atomically with snapshot content. Validate metadata consistency before persisting snapshots. Include metadata validation in snapshot checksum calculations.

⚠️ **Pitfall: Concurrent Snapshot Creation**

Allowing multiple snapshot creation operations to run simultaneously can create race conditions and resource contention that corrupts snapshots or causes system instability.

**What happens**: Multiple threads creating snapshots simultaneously can interfere with each other's state machine access, create conflicting file writes, or generate snapshots with different `last_included_index` values.

**Example scenario**: Two snapshot operations start simultaneously. Both freeze the state machine at slightly different points and create snapshots with the same filename, causing file corruption or inconsistent snapshot selection.

**Prevention**: Use proper locking to ensure only one snapshot creation operation runs at a time. Implement snapshot creation as an atomic state machine operation with clear ownership semantics.

⚠️ **Pitfall: Ignoring Disk Space During Compaction**

Snapshot creation temporarily requires significant additional disk space (original log + snapshot), which can cause disk full conditions that corrupt the operation.

**What happens**: If disk space is exhausted during snapshot creation or log truncation, file operations can fail partially, leaving the system in an inconsistent state with corrupted logs or incomplete snapshots.

**Example scenario**: System has 1GB free space, creates 800MB snapshot successfully, but fails during log truncation when temporary files require additional space. System is left with both full original log and complete snapshot, exceeding available space.

**Prevention**: Check available disk space before starting compaction operations. Reserve sufficient space for temporary files during the operation. Implement cleanup procedures that can recover from partial operations.

⚠️ **Pitfall: Snapshot Installation Without Log Conflict Resolution**

Installing snapshots without properly resolving conflicts with the follower's existing log entries can create inconsistent states or lost operations.

**What happens**: When a follower receives a snapshot, it must properly handle any log entries that conflict with or extend beyond the snapshot's coverage. Incorrect handling can lose committed operations or accept conflicting entries.

**Example scenario**: Follower has log entries 9500-10500. Receives snapshot covering entries 1-10000 term 5. Follower's entry 10000 is in term 4 (conflict), but follower fails to properly truncate conflicting entries 10000-10500, leaving inconsistent log state.

**Prevention**: Implement proper conflict detection when installing snapshots. Truncate all follower log entries that conflict with the snapshot's `last_included_index` and `last_included_term`. Validate log consistency after snapshot installation.

### Implementation Guidance

This implementation provides complete infrastructure for log compaction while leaving core snapshot logic for learners to implement based on their specific state machine requirements.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|----------------|
| Snapshot Storage | File-based with atomic rename | Object storage with versioning |
| Serialization | JSON with custom state machine encoding | Protocol Buffers with schema evolution |
| Checksumming | hashlib.sha256 for complete files | Per-chunk checksums with streaming validation |
| Chunk Transfer | Simple byte arrays in RPC | Streaming with backpressure control |

**Recommended File Structure:**

```
titan_consensus/
  consensus/
    node.py                    ← main RaftNode class
    log_manager.py            ← log storage and replication
    snapshot_manager.py       ← snapshot creation and installation (THIS COMPONENT)
    state_machine.py          ← abstract state machine interface
  storage/
    snapshot_storage.py       ← snapshot persistence utilities
    log_storage.py           ← log file management
  rpc/
    messages.py              ← InstallSnapshotRequest/Response definitions
    transport.py             ← network communication layer
  examples/
    kv_state_machine.py      ← example state machine implementation
  tests/
    test_snapshot_manager.py ← compaction tests with partition simulation
```

**Complete Snapshot Storage Infrastructure:**

```python
import os
import hashlib
import json
import tempfile
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class SnapshotMetadata:
    """Metadata for a state machine snapshot."""
    last_included_index: int
    last_included_term: int
    creation_timestamp: float
    checksum: str
    size_bytes: int

class SnapshotStorage:
    """Handles persistent storage of state machine snapshots with atomic operations."""
    
    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.current_snapshot_path = self.storage_dir / "snapshot.current"
        self.metadata_path = self.storage_dir / "snapshot.metadata"
    
    def save_snapshot(self, snapshot_data: bytes, metadata: SnapshotMetadata) -> None:
        """Atomically save snapshot with metadata and checksum verification."""
        # Verify checksum before saving
        calculated_checksum = hashlib.sha256(snapshot_data).hexdigest()
        if calculated_checksum != metadata.checksum:
            raise ValueError(f"Checksum mismatch: expected {metadata.checksum}, got {calculated_checksum}")
        
        # Write to temporary file first
        with tempfile.NamedTemporaryFile(dir=self.storage_dir, delete=False) as temp_file:
            temp_file.write(snapshot_data)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_path = temp_file.name
        
        # Write metadata to temporary file
        metadata_temp = f"{self.metadata_path}.tmp"
        with open(metadata_temp, 'w') as f:
            json.dump(metadata.__dict__, f)
            f.flush()
            os.fsync(f.fileno())
        
        # Atomic replacement of both files
        os.rename(temp_path, self.current_snapshot_path)
        os.rename(metadata_temp, self.metadata_path)
    
    def load_snapshot(self) -> Tuple[Optional[bytes], Optional[SnapshotMetadata]]:
        """Load current snapshot with metadata validation."""
        try:
            if not self.current_snapshot_path.exists() or not self.metadata_path.exists():
                return None, None
            
            # Load metadata first
            with open(self.metadata_path, 'r') as f:
                metadata_dict = json.load(f)
                metadata = SnapshotMetadata(**metadata_dict)
            
            # Load and verify snapshot data
            with open(self.current_snapshot_path, 'rb') as f:
                snapshot_data = f.read()
            
            # Verify checksum
            calculated_checksum = hashlib.sha256(snapshot_data).hexdigest()
            if calculated_checksum != metadata.checksum:
                raise ValueError("Snapshot checksum validation failed")
            
            return snapshot_data, metadata
        
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            return None, None

class ChunkedTransfer:
    """Handles chunked transfer of large snapshots over RPC."""
    
    CHUNK_SIZE = 64 * 1024  # 64KB chunks
    
    def __init__(self):
        self.active_transfers: Dict[str, Dict] = {}
    
    def start_transfer(self, follower_id: str, snapshot_data: bytes, metadata: SnapshotMetadata) -> str:
        """Initialize chunked transfer for a follower."""
        transfer_id = f"{follower_id}_{metadata.creation_timestamp}"
        self.active_transfers[transfer_id] = {
            'data': snapshot_data,
            'metadata': metadata,
            'next_offset': 0,
            'total_size': len(snapshot_data)
        }
        return transfer_id
    
    def get_next_chunk(self, transfer_id: str) -> Tuple[Optional[bytes], int, bool]:
        """Get next chunk for transfer. Returns (data, offset, is_done)."""
        if transfer_id not in self.active_transfers:
            return None, 0, True
        
        transfer = self.active_transfers[transfer_id]
        start_offset = transfer['next_offset']
        end_offset = min(start_offset + self.CHUNK_SIZE, transfer['total_size'])
        
        chunk_data = transfer['data'][start_offset:end_offset]
        is_done = end_offset >= transfer['total_size']
        
        transfer['next_offset'] = end_offset
        
        if is_done:
            del self.active_transfers[transfer_id]
        
        return chunk_data, start_offset, is_done

class FollowerSnapshotReceiver:
    """Handles receiving and assembling chunked snapshots on follower nodes."""
    
    def __init__(self):
        self.receiving_snapshots: Dict[str, Dict] = {}
    
    def receive_chunk(self, leader_id: str, last_included_index: int, 
                     offset: int, data: bytes, done: bool) -> bool:
        """Receive and assemble snapshot chunk. Returns True if complete."""
        transfer_key = f"{leader_id}_{last_included_index}"
        
        if transfer_key not in self.receiving_snapshots:
            self.receiving_snapshots[transfer_key] = {
                'chunks': {},
                'expected_offset': 0
            }
        
        transfer = self.receiving_snapshots[transfer_key]
        
        # Verify sequential chunk delivery
        if offset != transfer['expected_offset']:
            return False  # Out of order chunk, reject
        
        transfer['chunks'][offset] = data
        transfer['expected_offset'] = offset + len(data)
        
        if done:
            # Assemble complete snapshot
            complete_data = b''.join(
                transfer['chunks'][off] 
                for off in sorted(transfer['chunks'].keys())
            )
            transfer['complete_data'] = complete_data
            return True
        
        return False
    
    def get_complete_snapshot(self, leader_id: str, last_included_index: int) -> Optional[bytes]:
        """Get assembled snapshot data if transfer is complete."""
        transfer_key = f"{leader_id}_{last_included_index}"
        transfer = self.receiving_snapshots.get(transfer_key)
        
        if transfer and 'complete_data' in transfer:
            complete_data = transfer['complete_data']
            del self.receiving_snapshots[transfer_key]
            return complete_data
        
        return None
```

**Abstract State Machine Interface:**

```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class StateMachine(ABC):
    """Abstract interface for application state machines with snapshot support."""
    
    @abstractmethod
    def apply_entry(self, entry_data: bytes) -> Any:
        """Apply a log entry to the state machine and return the result."""
        # TODO: Implement application-specific entry processing
        # This method should update internal state based on the log entry
        # and return any result that should be sent back to the client
        pass
    
    @abstractmethod
    def create_snapshot(self) -> bytes:
        """Create a point-in-time snapshot of current state machine state."""
        # TODO: Serialize current state machine state into bytes
        # This should capture ALL state needed to restore the state machine
        # Consider using JSON, pickle, or custom serialization based on your needs
        # The result must be deterministic - same state produces same bytes
        pass
    
    @abstractmethod
    def restore_from_snapshot(self, snapshot_data: bytes) -> None:
        """Restore state machine state from a snapshot."""
        # TODO: Replace current state machine state with snapshot data
        # This should completely replace the state, not merge with existing state
        # Must handle deserialization errors gracefully
        # After this call, state machine should behave as if all log entries
        # up to the snapshot point had been applied
        pass
    
    @abstractmethod
    def get_state_size(self) -> int:
        """Return approximate size of current state for snapshot decisions."""
        # TODO: Return estimated memory/serialized size of current state
        # Used to help determine when snapshots would be beneficial
        # Doesn't need to be perfectly accurate, just a reasonable estimate
        pass
```

**Core Snapshot Manager Skeleton:**

```python
import time
import threading
from typing import Optional, List
from .storage.snapshot_storage import SnapshotStorage, SnapshotMetadata
from .storage.snapshot_storage import ChunkedTransfer, FollowerSnapshotReceiver
from .rpc.messages import InstallSnapshotRequest, InstallSnapshotResponse

class SnapshotManager:
    """Manages snapshot creation, storage, and installation for Raft log compaction."""
    
    def __init__(self, node_id: str, storage_dir: str, state_machine, log_manager):
        self.node_id = node_id
        self.storage = SnapshotStorage(storage_dir)
        self.state_machine = state_machine
        self.log_manager = log_manager
        self.chunked_transfer = ChunkedTransfer()
        self.snapshot_receiver = FollowerSnapshotReceiver()
        self.snapshot_lock = threading.Lock()
        
        # Load existing snapshot on startup
        self.current_snapshot_data, self.current_snapshot_metadata = self.storage.load_snapshot()
    
    def should_create_snapshot(self) -> bool:
        """Determine if a new snapshot should be created based on log size."""
        # TODO 1: Get current log size from log_manager
        # TODO 2: If no previous snapshot, check if log size > SNAPSHOT_THRESHOLD
        # TODO 3: If previous snapshot exists, check if entries since snapshot > SNAPSHOT_THRESHOLD
        # TODO 4: Return True if snapshot should be created
        # Hint: Use SNAPSHOT_THRESHOLD = 10000 entries as the trigger
        pass
    
    def create_snapshot(self) -> bool:
        """Create a new snapshot of current state machine state."""
        with self.snapshot_lock:
            # TODO 1: Get current log state (last_applied_index, term) from log_manager
            # TODO 2: Freeze state machine to prevent concurrent modifications
            # TODO 3: Extract state machine data using state_machine.create_snapshot()
            # TODO 4: Calculate checksum of snapshot data
            # TODO 5: Create SnapshotMetadata with current timestamp and log position
            # TODO 6: Save snapshot using self.storage.save_snapshot()
            # TODO 7: Update self.current_snapshot_data and self.current_snapshot_metadata
            # TODO 8: Resume state machine operations
            # TODO 9: Return True if successful, False if failed
            # Hint: Wrap in try/except to handle disk errors gracefully
            pass
    
    def install_snapshot_from_leader(self, request: InstallSnapshotRequest) -> InstallSnapshotResponse:
        """Handle InstallSnapshot RPC from leader."""
        # TODO 1: Validate request term against current node term
        # TODO 2: If term is old, return InstallSnapshotResponse with current term
        # TODO 3: Process chunk using self.snapshot_receiver.receive_chunk()
        # TODO 4: If chunk processing fails (out of order), return error response
        # TODO 5: If not final chunk (done=False), return success response
        # TODO 6: If final chunk (done=True), get complete snapshot data
        # TODO 7: Validate complete snapshot checksum
        # TODO 8: Install snapshot using self._install_complete_snapshot()
        # TODO 9: Update log state (commit_index, last_applied) to snapshot position
        # TODO 10: Return success response
        # Hint: Handle each step's errors separately for better debugging
        pass
    
    def send_install_snapshot(self, follower_id: str, rpc_client) -> bool:
        """Send current snapshot to a follower using chunked transfer."""
        if not self.current_snapshot_data:
            return False
            
        # TODO 1: Start chunked transfer using self.chunked_transfer.start_transfer()
        # TODO 2: Get transfer_id for tracking this transfer
        # TODO 3: Loop until all chunks sent:
        # TODO 4:   Get next chunk using self.chunked_transfer.get_next_chunk()
        # TODO 5:   Create InstallSnapshotRequest with chunk data and metadata
        # TODO 6:   Send request using rpc_client.send_install_snapshot()
        # TODO 7:   Check response term - if higher, update node term and return False
        # TODO 8:   If request fails, retry with backoff or abort transfer
        # TODO 9: Return True if all chunks sent successfully
        # Hint: Add timeout and retry logic for network failures
        pass
    
    def _install_complete_snapshot(self, snapshot_data: bytes, metadata: SnapshotMetadata) -> None:
        """Install a complete snapshot, replacing current state machine state."""
        # TODO 1: Validate snapshot checksum matches metadata
        # TODO 2: Create backup of current state machine state (if needed)
        # TODO 3: Call state_machine.restore_from_snapshot(snapshot_data)
        # TODO 4: Update log_manager to reflect new snapshot position
        # TODO 5: Truncate log entries that are now covered by snapshot
        # TODO 6: Save snapshot to storage using self.storage.save_snapshot()
        # TODO 7: Update self.current_snapshot_data and self.current_snapshot_metadata
        # Hint: This should be atomic - either completely succeeds or leaves system unchanged
        pass
    
    def get_snapshot_info(self) -> Optional[Dict]:
        """Get information about current snapshot for status reporting."""
        if not self.current_snapshot_metadata:
            return None
            
        return {
            'last_included_index': self.current_snapshot_metadata.last_included_index,
            'last_included_term': self.current_snapshot_metadata.last_included_term,
            'creation_timestamp': self.current_snapshot_metadata.creation_timestamp,
            'size_bytes': self.current_snapshot_metadata.size_bytes
        }
```

**Milestone Checkpoint:**

After implementing the core snapshot logic, verify functionality with these tests:

```bash
# Test snapshot creation
python -m pytest tests/test_snapshot_manager.py::test_snapshot_creation -v

# Test chunked transfer
python -m pytest tests/test_snapshot_manager.py::test_chunked_transfer -v

# Test snapshot installation
python -m pytest tests/test_snapshot_manager.py::test_snapshot_installation -v
```

**Expected behavior after this milestone:**
- Leader creates snapshots automatically when log exceeds threshold
- Snapshots can be transferred to followers in chunks
- Followers can install snapshots and truncate old log entries
- System continues operating normally during snapshot operations
- Log growth is bounded by snapshot frequency

**Signs of correct implementation:**
- Log files don't grow unbounded over time
- New nodes can catch up quickly using snapshots instead of replaying entire log
- Snapshot creation doesn't block normal consensus operations
- Chunked transfer handles network failures gracefully

**Common debugging issues:**
- **Snapshot creation hangs**: Check if state machine freezing is implemented correctly
- **Transfer failures**: Verify chunk sequencing and checksum validation
- **Installation failures**: Ensure log truncation preserves safety invariants
- **Corruption on restart**: Check that snapshot persistence uses atomic operations


## Membership Changes (Milestone 4)

> **Milestone(s):** Milestone 4: Membership Changes - Safely add or remove nodes from the cluster without downtime

### Mental Model: Committee Restructuring

Think of cluster membership changes like restructuring a corporate board of directors while keeping the company operational. Imagine a 5-person board that needs to add 2 new members and remove 1 existing member. The dangerous approach would be to simply announce "effective immediately, person X is out and persons Y and Z are in." This could create chaos - what if the old member doesn't get the memo and continues voting? What if the new members aren't up to speed on recent decisions? You might end up with competing groups each claiming to be the legitimate board.

The safe approach uses a **transition period** where both the old and new configurations operate together temporarily. During this joint period, any major decision requires approval from both the old majority AND the new majority. This prevents either group from making unilateral decisions. Only after everyone acknowledges the transition can we move to the new configuration exclusively.

In Raft, this is called **joint consensus** - a two-phase protocol where the cluster temporarily operates under both the old membership (C_old) and new membership (C_new) simultaneously. Any decision must achieve quorum in BOTH configurations. This mathematical property guarantees that at no point can two separate groups each believe they have legitimate authority to make decisions.

The key insight is that overlap prevents split-brain scenarios. If we need majorities from both the old and new configurations, then by definition those majorities must share at least one node that can coordinate between them. This shared node acts like a bridge preventing the cluster from fragmenting into independent groups.

### Joint Consensus Protocol

The joint consensus protocol implements atomic membership changes through a carefully orchestrated two-phase transition. The protocol ensures that at no point during the change can multiple leaders exist or conflicting decisions be made.

#### Phase 1: Enter Joint Consensus

The membership change begins when the current leader receives a configuration change request. The leader cannot immediately switch to the new configuration because this could create a split-brain scenario where nodes disagree about who should be in the cluster.

Instead, the leader creates a special **joint configuration log entry** that contains both the old configuration (C_old) and the new configuration (C_new). This entry is replicated to all nodes using the standard log replication protocol. Once this entry is committed, all nodes begin operating under joint consensus rules.

| Joint Consensus Rules | Description | Safety Guarantee |
|---------------------|-------------|------------------|
| Dual Quorum Requirement | Any decision requires majority approval from BOTH C_old AND C_new | Prevents split-brain between old and new groups |
| Leader Election Constraint | A candidate can become leader only if it receives votes from majorities in BOTH configurations | Ensures single leadership during transition |
| Log Commitment Rule | Entries commit only when replicated to majorities in BOTH configurations | Maintains consistency across all nodes |
| Heartbeat Distribution | Leader sends heartbeats to ALL nodes in C_old ∪ C_new | Keeps entire extended cluster synchronized |

The mathematical foundation is straightforward: if a decision requires majorities from both configurations, then any two decisions must involve at least one common node. This common node provides the coordination point that prevents conflicting decisions.

Consider a cluster transitioning from {A, B, C} to {A, B, D, E}. Under joint consensus:
- C_old majority: any 2 of {A, B, C}
- C_new majority: any 3 of {A, B, D, E}
- Joint requirement: must satisfy BOTH conditions

Any valid decision requires nodes from both majorities, ensuring coordination through the shared nodes (A and/or B).

#### Phase 2: Transition to New Configuration

Once the joint configuration entry is committed and all nodes are operating under dual-quorum rules, the leader can safely initiate the second phase. The leader creates a new configuration log entry containing only C_new and replicates it using the joint consensus rules (requiring majorities from both configurations).

When this second entry commits, all nodes switch exclusively to the new configuration C_new. The old configuration C_old is discarded, and normal single-configuration operation resumes.

The two-phase structure provides the critical safety property: the transition from single-old to single-new configuration passes through a joint state where both configurations must agree. This prevents any window where multiple groups could make independent decisions.

![Joint Consensus Membership Change](./diagrams/membership-change-sequence.svg)

#### Configuration Change State Machine

Each node maintains state to track the current membership configuration and any ongoing transitions.

| Configuration State | Description | Quorum Requirements | Next Transitions |
|-------------------|-------------|-------------------|------------------|
| STABLE_OLD | Operating under original configuration C_old only | Majority of C_old | Enter JOINT_CONSENSUS |
| JOINT_CONSENSUS | Operating under both C_old and C_new | Majority of C_old AND majority of C_new | Transition to STABLE_NEW |
| STABLE_NEW | Operating under new configuration C_new only | Majority of C_new | Can initiate new change |
| TRANSITIONING | Processing configuration change request | Current rules apply | Complete current change |

The state transitions are triggered by specific log entries:

1. **C_old,new entry commits** → STABLE_OLD to JOINT_CONSENSUS
2. **C_new entry commits** → JOINT_CONSENSUS to STABLE_NEW

#### Rollback and Error Handling

Joint consensus provides natural rollback capabilities. If the new configuration proves problematic during the joint phase (e.g., new nodes are unresponsive), the leader can initiate a rollback by creating a configuration entry with only C_old. Since this still requires joint consensus approval, it provides a safe return path.

| Rollback Scenario | Detection | Action | Safety Guarantee |
|------------------|-----------|--------|------------------|
| New nodes unresponsive | Heartbeat timeouts from C_new nodes | Leader proposes C_old-only configuration | Joint consensus prevents abandoning responsive old nodes |
| Network partition | Cannot reach C_new majority | Continue with C_old rules until partition heals | Prevents premature commitment to unreachable configuration |
| Leader failure during transition | Election timeout in joint state | New leader elected using joint consensus rules | Maintains transition state across leader changes |

> **Decision: Two-Phase Joint Consensus**
> - **Context**: Membership changes risk creating split-brain scenarios if nodes disagree about cluster composition
> - **Options Considered**: Direct configuration switch, single-phase with pre-vote, two-phase joint consensus
> - **Decision**: Implement two-phase joint consensus as specified in Raft paper
> - **Rationale**: Mathematical guarantee against split-brain through dual-quorum requirements, proven correct in academic literature
> - **Consequences**: Adds complexity but provides strongest safety guarantees, enables safe rollback, handles arbitrary membership changes

### New Node Catch-up Process

Before a new node can participate in consensus decisions, it must be brought up to the current cluster state. Allowing an uninitialized node to immediately participate could disrupt availability - the node would reject all log entries due to missing history, forcing the leader to send large portions of the log repeatedly.

The catch-up process ensures new nodes are productive cluster members before they gain voting rights.

#### Pre-Configuration Catch-up

New nodes undergo a preliminary catch-up phase before the membership change begins. During this phase, the new node connects to the current leader but does not participate in any consensus decisions.

| Catch-up Phase | Node Status | Activities | Safety Considerations |
|----------------|-------------|------------|----------------------|
| Initial Contact | Non-voting observer | Receives InstallSnapshot RPC from leader | Cannot affect consensus decisions |
| Snapshot Installation | Receiving baseline state | Downloads and applies current snapshot | May temporarily lag behind real-time |
| Log Synchronization | Catching up to current | Receives AppendEntries for recent log entries | Still cannot vote or become leader |
| Readiness Verification | Ready for promotion | Leader verifies node is reasonably current | Must be within threshold of current log |

The leader monitors the new node's progress and only initiates the membership change when the node demonstrates it can keep up with the current log replication rate.

#### Catch-up Threshold Determination

The leader uses several metrics to determine when a new node is ready for promotion to voting status:

| Readiness Metric | Threshold | Rationale | Measurement |
|------------------|-----------|-----------|-------------|
| Log Index Gap | ≤ 100 entries behind | Node can catch up quickly during joint consensus | `leader_last_log_index - node_last_log_index` |
| Replication Latency | ≤ 2x normal heartbeat interval | Node responds promptly to replication requests | Average time for AppendEntries response |
| Success Rate | ≥ 95% of recent AppendEntries succeed | Node is stable and reachable | Success ratio over last 20 requests |
| Availability Window | 30 seconds of consistent performance | Node demonstrates sustained reliability | Continuous monitoring period |

These thresholds prevent the common mistake of adding nodes that immediately become bottlenecks.

#### State Transfer Mechanics

The catch-up process leverages the existing snapshot and log replication infrastructure with some modifications for non-voting nodes.

**Snapshot Transfer Process:**
1. Leader identifies the most recent snapshot that encompasses the state machine
2. Leader sends InstallSnapshot RPC chunks to the new node
3. New node validates and applies the snapshot
4. Leader begins sending AppendEntries for post-snapshot log entries
5. New node processes entries but does not participate in commitment decisions

**Optimization for Large States:**
For clusters with large state machines, the catch-up process can be optimized by allowing new nodes to receive snapshots from any current cluster member, not just the leader. This distributes the transfer load and reduces impact on the leader's normal operations.

> **Decision: Pre-Configuration Catch-up**
> - **Context**: New nodes need current state before participating in consensus to avoid disrupting cluster performance
> - **Options Considered**: Immediate addition with degraded performance, catch-up during joint consensus, pre-configuration catch-up
> - **Decision**: Require substantial catch-up before initiating membership change
> - **Rationale**: Prevents new nodes from becoming immediate bottlenecks, ensures they can contribute positively to cluster performance
> - **Consequences**: Adds delay to membership changes but significantly improves success rate and availability

### Disruptive Server Prevention

A **disruptive server** is a node that negatively impacts cluster availability despite being technically functional. This can occur when new nodes have poor network connectivity, insufficient resources, or configuration problems that cause them to be slow or unreliable.

The membership change protocol includes several mechanisms to detect and prevent disruptive servers from degrading cluster performance.

#### Performance Impact Detection

The leader continuously monitors the performance impact of all nodes, with special attention to newly added nodes during their initial operational period.

| Performance Metric | Normal Range | Disruptive Threshold | Detection Window |
|-------------------|--------------|---------------------|------------------|
| AppendEntries Latency | ≤ 50ms average | > 200ms sustained | 10-request moving average |
| Request Success Rate | ≥ 98% | < 90% over window | Last 50 requests |
| Heartbeat Response Time | ≤ 25ms | > 100ms sustained | Last 10 heartbeats |
| Log Application Rate | Keep pace with leader | > 1000 entries behind consistently | Continuous monitoring |

When a node exceeds disruptive thresholds, the leader can initiate remedial actions before the node severely impacts overall cluster performance.

#### Graduated Response to Disruptive Behavior

Rather than immediately removing potentially disruptive servers, the system employs a graduated response that attempts to resolve issues before resorting to membership changes.

**Response Escalation Levels:**

| Level | Trigger | Action | Duration |
|-------|---------|--------|----------|
| 1 - Monitor | Performance degradation detected | Increase monitoring frequency, log warnings | 30 seconds |
| 2 - Isolate | Sustained poor performance | Reduce batching to problematic node, prioritize other nodes | 60 seconds |
| 3 - Warn | Performance remains poor | Send diagnostic RPCs, request self-health check | 30 seconds |
| 4 - Remove | Node confirmed disruptive | Initiate membership change to remove node | As needed |

This graduated approach prevents premature removal of nodes experiencing temporary issues while protecting the cluster from sustained performance degradation.

#### Network Partition Tolerance

Disruptive server detection must distinguish between genuinely problematic nodes and nodes that are victims of network partitions. A node that appears slow might actually be in a different network segment with higher latency but still functioning correctly.

The detection algorithm considers network-wide patterns:

| Scenario | Detection Pattern | Response |
|----------|------------------|----------|
| Single Node Issues | Only one node shows degraded performance | Apply graduated response to specific node |
| Widespread Degradation | Multiple nodes show similar issues | Investigate network-wide problems, delay removal decisions |
| Partition Recovery | Previously slow node suddenly responsive | Reset monitoring metrics, clear warning state |
| Geographic Latency | Consistent but elevated latency from region | Adjust thresholds based on geographic grouping |

#### Automatic Remediation

In some cases, the system can automatically address disruptive server issues without human intervention.

**Self-Healing Mechanisms:**

1. **Configuration Adjustment**: Automatically adjust heartbeat intervals and timeouts for nodes with consistently higher latency
2. **Load Rebalancing**: Reduce the batch sizes sent to slower nodes while maintaining consistency
3. **Diagnostic Information**: Request nodes to report their local health status and resource utilization
4. **Graceful Degradation**: Temporarily reduce the node's participation while allowing it to catch up

These mechanisms often resolve transient issues without requiring membership changes.

> **Decision: Graduated Disruptive Server Response**
> - **Context**: New nodes may have performance issues that impact cluster availability but might be resolvable
> - **Options Considered**: Immediate removal on performance issues, tolerance with degraded performance, graduated response with escalation
> - **Decision**: Implement graduated response with monitoring, isolation, and eventual removal if needed
> - **Rationale**: Balances cluster protection with tolerance for temporary issues, provides opportunity for problem resolution
> - **Consequences**: More complex monitoring logic but better handling of edge cases and transient problems

### Architecture Decisions for Membership

Several key architectural decisions shape how membership changes are implemented and integrated with the rest of the Raft consensus system.

#### Configuration Storage and Persistence

Membership configurations must be stored as part of the replicated log to ensure all nodes have consistent views of cluster composition. However, the way configurations are stored affects recovery, bootstrap, and operational procedures.

> **Decision: Configuration as Special Log Entries**
> - **Context**: Membership information must be replicated and persistent but also quickly accessible for operational decisions
> - **Options Considered**: Separate configuration store, embed in regular log entries, special configuration log entries, hybrid approach
> - **Decision**: Use special log entry types for configuration changes with cached current configuration
> - **Rationale**: Ensures configuration changes follow same consistency guarantees as data changes, provides audit trail
> - **Consequences**: Configuration changes consume log space but gain full Raft consistency guarantees

| Storage Decision | Implementation | Benefits | Trade-offs |
|------------------|----------------|----------|------------|
| Log Entry Storage | Configuration stored as `ConfigurationEntry` type in replicated log | Consistent replication, audit trail, crash recovery | Consumes log space, requires log scanning for bootstrap |
| Current Config Cache | Active configuration cached in volatile state | Fast access for operational decisions | Must be rebuilt from log after restart |
| Configuration Versioning | Each configuration has monotonic version number | Enables easy comparison and ordering | Additional complexity in version management |
| Bootstrap Configuration | Initial configuration stored in separate bootstrap file | Enables cluster startup with known members | Special case handling for initial startup |

#### Concurrent Operations During Membership Changes

The system must define how normal client operations interact with ongoing membership changes. The key question is whether to allow, queue, or reject client requests during the joint consensus phase.

| Concurrent Operation Policy | Benefits | Drawbacks | Chosen |
|---------------------------|----------|-----------|---------|
| Allow All Operations | No service interruption during membership changes | Increased complexity, potential performance impact | ✓ |
| Queue Operations | Simpler implementation, predictable performance | Service interruption, queue management complexity | ✗ |
| Reject Operations | Simplest implementation | Service unavailable during changes | ✗ |

The chosen approach allows normal client operations to continue during membership changes, but they must achieve the dual-quorum requirement during the joint consensus phase. This maintains availability while preserving safety guarantees.

#### Leader Transition During Membership Changes

A critical question is what happens if the leader fails during a membership change. The system must handle leader election under joint consensus rules and potentially continue or abort in-progress membership changes.

**Leader Election During Joint Consensus:**

When operating under joint consensus, leader election requires vote majorities from both C_old and C_new. This ensures that any new leader understands the ongoing transition and can safely continue or complete it.

| Election Scenario | Vote Requirement | Outcome | Handling |
|------------------|------------------|---------|----------|
| Election during JOINT_CONSENSUS | Majority from C_old AND majority from C_new | Single legitimate leader | New leader continues membership change |
| Partial votes from old config | Majority from C_old only | No leader elected | Continue election attempts |
| Partial votes from new config | Majority from C_new only | No leader elected | Continue election attempts |
| Network prevents joint majority | Neither configuration achieves majority | No leader elected | Wait for network healing |

#### Membership Change Atomicity

The two-phase protocol provides strong atomicity guarantees for membership changes, but the implementation must handle various edge cases where the process might be interrupted.

**Atomicity Guarantees:**

| Guarantee | Implementation | Failure Handling |
|-----------|----------------|------------------|
| Single Change at a Time | Reject new membership requests while one is in progress | Queue subsequent requests until current change completes |
| Complete or Rollback | Either fully transition to new config or remain at old config | Use timeout-based rollback if joint consensus phase stalls |
| Persistent State | Membership changes survive leader failures and restarts | New leaders continue in-progress changes from log state |
| Consistent View | All nodes eventually converge on same configuration | Use log replication to ensure configuration consistency |

#### Integration with Log Compaction

Membership changes interact with log compaction in several important ways. Configuration information must remain accessible even after log truncation.

**Configuration Preservation During Compaction:**

1. **Current Configuration**: Always preserved in snapshot metadata
2. **In-Progress Changes**: Joint consensus state included in snapshot if change is ongoing
3. **Historical Audit**: Recent configuration history preserved in log even after compaction
4. **Bootstrap Recovery**: Snapshots contain sufficient information to restart nodes with correct membership

> **Decision: Configuration-Aware Snapshotting**
> - **Context**: Log compaction must preserve essential membership information for proper cluster operation
> - **Options Considered**: Include full configuration history, current configuration only, configuration-aware compaction boundaries
> - **Decision**: Include current configuration and any in-progress change state in snapshot metadata
> - **Rationale**: Ensures nodes can restart and rejoin cluster with correct membership understanding
> - **Consequences**: Slightly larger snapshot metadata but critical for operational correctness

### Common Membership Pitfalls

Membership changes involve complex distributed coordination that can fail in subtle ways. Understanding and avoiding common mistakes is critical for building a robust implementation.

#### Split-Brain Risks and Prevention

⚠️ **Pitfall: Direct Configuration Switch**
The most dangerous mistake is attempting to change cluster membership by directly switching from the old configuration to the new configuration without using joint consensus. This appears simpler but creates dangerous race conditions.

**Why it's wrong:** Consider changing from {A, B, C} to {B, C, D}. If some nodes apply the change before others, you might have:
- Nodes A, B think the cluster is {A, B, C} (majority = 2 nodes)
- Nodes C, D think the cluster is {B, C, D} (majority = 2 nodes)
- Both groups can elect leaders simultaneously: A elected by {A, B}, D elected by {C, D}

**How to fix:** Always use the two-phase joint consensus protocol. During the transition, decisions require majorities from BOTH configurations, preventing split-brain by mathematical impossibility.

⚠️ **Pitfall: Ignoring Network Partitions During Changes**
Starting a membership change when the cluster is already partitioned can lead to conflicting configuration changes being applied in different partitions.

**Why it's wrong:** If the cluster is partitioned and both sides attempt membership changes, you can end up with permanently incompatible configurations that cannot be reconciled.

**How to fix:** Leaders should verify they can communicate with a majority of the current configuration before initiating membership changes. Implement partition detection through heartbeat monitoring.

#### Timing Races and Coordination Issues

⚠️ **Pitfall: Premature New Node Promotion**
Adding nodes to the voting configuration before they've caught up with the current log creates immediate performance problems and can prevent progress.

**Why it's wrong:** New nodes will reject AppendEntries for log indices they haven't seen, forcing the leader to send large amounts of historical data. During joint consensus, this can prevent achieving the required majorities.

**How to fix:** Implement the catch-up process with specific readiness thresholds. Only begin membership changes when new nodes demonstrate they can keep pace with current operations.

| Timing Issue | Symptom | Root Cause | Prevention |
|--------------|---------|------------|------------|
| Slow membership change | Joint consensus phase takes very long | New nodes not ready, causing replication delays | Implement readiness verification before starting change |
| Change rollback loops | Membership change repeatedly fails and rolls back | Network issues or node failures during transition | Add jitter to rollback timing, implement exponential backoff |
| Leader election storms | Frequent elections during membership change | Joint consensus requirements too strict for current network | Adjust election timeouts during transitions |
| Configuration drift | Nodes have different views of membership | Log replication failures during configuration changes | Implement configuration consistency verification |

⚠️ **Pitfall: Concurrent Membership Changes**
Allowing multiple membership changes to proceed simultaneously creates complex interactions that are difficult to reason about and can lead to inconsistent states.

**Why it's wrong:** Joint consensus is designed for transitioning between two configurations. With multiple concurrent changes, you could have overlapping joint states that violate the mathematical properties that ensure safety.

**How to fix:** Implement a single-change-at-a-time policy. Queue subsequent membership change requests until the current change completes or rolls back.

#### Availability Loss During Transitions

⚠️ **Pitfall: Blocking Operations During Membership Changes**
Some implementations unnecessarily block all client operations during membership changes, creating availability windows that defeat the purpose of the consensus system.

**Why it's wrong:** Membership changes can take significant time, especially if new nodes need to catch up or network conditions are poor. Blocking client operations during this time creates unnecessary downtime.

**How to fix:** Allow normal client operations to continue during membership changes. Apply the dual-quorum requirement to client operations during joint consensus, but don't block them entirely.

⚠️ **Pitfall: Inadequate Rollback Mechanisms**
Failing to implement proper rollback for membership changes that encounter problems can leave the cluster in a stuck state where it cannot make progress.

**Why it's wrong:** If a membership change fails (e.g., new nodes become unresponsive), the cluster might remain in joint consensus indefinitely, requiring manual intervention to restore normal operation.

**How to fix:** Implement timeout-based rollback mechanisms. If joint consensus cannot be completed within a reasonable time, automatically initiate rollback to the original configuration.

| Availability Issue | Impact | Detection | Recovery |
|--------------------|--------|-----------|----------|
| Stuck in joint consensus | Reduced throughput due to dual-quorum requirement | Monitor time spent in joint phase | Implement automatic rollback after timeout |
| Unreachable new nodes | Cannot complete membership change | Heartbeat failures from new nodes | Rollback to original configuration |
| Overloaded leader during catch-up | Leader cannot service normal requests | High latency for client operations | Distribute catch-up load or pause additions |
| Resource exhaustion | Cluster performance degrades | Monitor resource utilization metrics | Implement admission control for membership changes |

#### State Inconsistency and Recovery Issues

⚠️ **Pitfall: Configuration Recovery After Crashes**
Failing to properly handle configuration state during node restarts can lead to nodes rejoining with incorrect membership information.

**Why it's wrong:** If a node crashes during a membership change and restarts with stale configuration information, it might not understand the current cluster composition and could disrupt operations.

**How to fix:** Ensure configuration information is properly persisted in the log and that nodes rebuild their current configuration view from the log during startup. Include configuration state in snapshot metadata.

⚠️ **Pitfall: Ignoring Minority Partitions During Recovery**
When a network partition heals, nodes that were in the minority partition might have outdated configuration information that conflicts with changes made by the majority partition.

**Why it's wrong:** Minority partition nodes might reject valid operations or attempt to participate in elections with outdated membership views.

**How to fix:** Implement partition recovery protocols where nodes verify their configuration information against the current cluster state when rejoining after a partition.

### Implementation Guidance

The membership changes implementation builds upon the election, replication, and compaction systems established in previous milestones. This section provides the concrete infrastructure and code patterns needed to safely implement joint consensus.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Configuration Storage | JSON files with atomic write | SQLite with WAL mode |
| Network Monitoring | Ping-based reachability | Full network topology discovery |
| Health Checking | Simple heartbeat responses | Comprehensive node health metrics |
| Change Coordination | Leader-only membership changes | Distributed change proposals with voting |

#### Recommended File Structure

The membership change implementation adds several new components to the existing Raft architecture:

```
titan-consensus/
  src/
    consensus/
      __init__.py
      raft_node.py                 ← extend with membership methods
      election.py                  ← existing election logic
      replication.py               ← existing replication logic
      compaction.py                ← existing compaction logic
      membership.py                ← NEW: membership change coordination
      config_manager.py            ← NEW: configuration persistence and validation
      health_monitor.py            ← NEW: node health and performance monitoring
      joint_consensus.py           ← NEW: joint consensus state machine
    network/
      rpc_client.py               ← extend with membership RPCs
      rpc_server.py               ← extend with membership handlers
    storage/
      log_store.py                ← extend with configuration entries
      snapshot_store.py           ← extend with membership metadata
    tests/
      test_membership.py          ← NEW: comprehensive membership change tests
      test_joint_consensus.py     ← NEW: joint consensus protocol tests
      test_disruptive_servers.py  ← NEW: performance impact tests
```

#### Configuration Management Infrastructure

Complete infrastructure for managing cluster configurations throughout the membership change lifecycle:

```python
# config_manager.py - Complete configuration management system

from enum import Enum
from typing import Set, Optional, Dict, List, Tuple
from dataclasses import dataclass
import json
import time
import threading
import os

class ConfigurationState(Enum):
    STABLE = "stable"
    JOINT = "joint"
    TRANSITIONING = "transitioning"

@dataclass
class ClusterConfiguration:
    """Represents a cluster membership configuration."""
    nodes: Set[str]
    version: int
    timestamp: float
    
    def to_dict(self) -> Dict:
        return {
            'nodes': list(self.nodes),
            'version': self.version,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ClusterConfiguration':
        return cls(
            nodes=set(data['nodes']),
            version=data['version'],
            timestamp=data['timestamp']
        )
    
    def majority_size(self) -> int:
        return len(self.nodes) // 2 + 1
    
    def contains_majority(self, responding_nodes: Set[str]) -> bool:
        intersection = self.nodes.intersection(responding_nodes)
        return len(intersection) >= self.majority_size()

@dataclass
class JointConfiguration:
    """Represents the joint consensus configuration with old and new configs."""
    old_config: ClusterConfiguration
    new_config: ClusterConfiguration
    initiated_at: float
    
    def requires_dual_majority(self, responding_nodes: Set[str]) -> bool:
        return (self.old_config.contains_majority(responding_nodes) and 
                self.new_config.contains_majority(responding_nodes))
    
    def all_nodes(self) -> Set[str]:
        return self.old_config.nodes.union(self.new_config.nodes)

class ConfigurationManager:
    """Manages cluster configuration state and transitions."""
    
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.current_config: Optional[ClusterConfiguration] = None
        self.joint_config: Optional[JointConfiguration] = None
        self.state = ConfigurationState.STABLE
        self.lock = threading.RLock()
        self.version_counter = 0
        
    def initialize_bootstrap_config(self, initial_nodes: Set[str]) -> None:
        """Initialize the cluster with the first configuration."""
        with self.lock:
            self.current_config = ClusterConfiguration(
                nodes=initial_nodes,
                version=1,
                timestamp=time.time()
            )
            self.state = ConfigurationState.STABLE
            self._persist_configuration()
    
    def start_membership_change(self, new_nodes: Set[str]) -> bool:
        """Initiate a membership change to the new configuration."""
        with self.lock:
            if self.state != ConfigurationState.STABLE:
                return False  # Change already in progress
            
            if new_nodes == self.current_config.nodes:
                return True  # No change needed
            
            self.joint_config = JointConfiguration(
                old_config=self.current_config,
                new_config=ClusterConfiguration(
                    nodes=new_nodes,
                    version=self.current_config.version + 1,
                    timestamp=time.time()
                ),
                initiated_at=time.time()
            )
            self.state = ConfigurationState.JOINT
            self._persist_configuration()
            return True
    
    def complete_membership_change(self) -> bool:
        """Complete the membership change by transitioning to new config only."""
        with self.lock:
            if self.state != ConfigurationState.JOINT or not self.joint_config:
                return False
            
            self.current_config = self.joint_config.new_config
            self.joint_config = None
            self.state = ConfigurationState.STABLE
            self._persist_configuration()
            return True
    
    def rollback_membership_change(self) -> bool:
        """Rollback to the old configuration."""
        with self.lock:
            if self.state != ConfigurationState.JOINT or not self.joint_config:
                return False
            
            # Keep old config, discard new config
            self.joint_config = None
            self.state = ConfigurationState.STABLE
            self._persist_configuration()
            return True
    
    def get_voting_nodes(self) -> Set[str]:
        """Get all nodes that should participate in voting."""
        with self.lock:
            if self.state == ConfigurationState.JOINT:
                return self.joint_config.all_nodes()
            return self.current_config.nodes.copy()
    
    def requires_dual_majority(self) -> bool:
        """Check if decisions require majorities from both old and new configs."""
        with self.lock:
            return self.state == ConfigurationState.JOINT
    
    def check_majority(self, responding_nodes: Set[str]) -> bool:
        """Check if responding nodes constitute a valid majority."""
        with self.lock:
            if self.state == ConfigurationState.JOINT:
                return self.joint_config.requires_dual_majority(responding_nodes)
            return self.current_config.contains_majority(responding_nodes)
    
    def _persist_configuration(self) -> None:
        """Atomically persist current configuration to disk."""
        config_data = {
            'state': self.state.value,
            'current_config': self.current_config.to_dict() if self.current_config else None,
            'joint_config': {
                'old_config': self.joint_config.old_config.to_dict(),
                'new_config': self.joint_config.new_config.to_dict(),
                'initiated_at': self.joint_config.initiated_at
            } if self.joint_config else None
        }
        
        # Atomic write using temporary file
        temp_file = f"{self.config_file}.tmp"
        with open(temp_file, 'w') as f:
            json.dump(config_data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        
        os.replace(temp_file, self.config_file)
    
    def load_configuration(self) -> bool:
        """Load configuration from disk after restart."""
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            with self.lock:
                self.state = ConfigurationState(config_data['state'])
                
                if config_data['current_config']:
                    self.current_config = ClusterConfiguration.from_dict(
                        config_data['current_config']
                    )
                
                if config_data['joint_config']:
                    joint_data = config_data['joint_config']
                    self.joint_config = JointConfiguration(
                        old_config=ClusterConfiguration.from_dict(joint_data['old_config']),
                        new_config=ClusterConfiguration.from_dict(joint_data['new_config']),
                        initiated_at=joint_data['initiated_at']
                    )
                
                return True
                
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return False
```

#### Health Monitoring Infrastructure

Complete system for monitoring node health and detecting disruptive servers:

```python
# health_monitor.py - Node health and performance monitoring

from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import threading
import statistics
from collections import deque

@dataclass
class HealthMetrics:
    """Health metrics for a single node."""
    node_id: str
    last_response_time: float
    avg_latency_ms: float
    success_rate: float
    last_log_index: int
    responses_received: int
    last_heartbeat: float
    
    def is_healthy(self) -> bool:
        current_time = time.time()
        return (self.success_rate >= 0.90 and 
                self.avg_latency_ms <= 200 and
                current_time - self.last_heartbeat <= 10.0)
    
    def is_disruptive(self) -> bool:
        current_time = time.time()
        return (self.success_rate < 0.80 or 
                self.avg_latency_ms > 500 or
                current_time - self.last_heartbeat > 15.0)

class NodeHealthMonitor:
    """Monitors health and performance of cluster nodes."""
    
    def __init__(self, monitoring_window: int = 50):
        self.monitoring_window = monitoring_window
        self.node_metrics: Dict[str, HealthMetrics] = {}
        self.latency_history: Dict[str, deque] = {}
        self.success_history: Dict[str, deque] = {}
        self.lock = threading.RLock()
        
    def record_rpc_response(self, node_id: str, success: bool, 
                           latency_ms: float, log_index: int) -> None:
        """Record the result of an RPC call to a node."""
        with self.lock:
            current_time = time.time()
            
            # Initialize tracking for new nodes
            if node_id not in self.latency_history:
                self.latency_history[node_id] = deque(maxlen=self.monitoring_window)
                self.success_history[node_id] = deque(maxlen=self.monitoring_window)
                self.node_metrics[node_id] = HealthMetrics(
                    node_id=node_id,
                    last_response_time=current_time,
                    avg_latency_ms=latency_ms,
                    success_rate=1.0 if success else 0.0,
                    last_log_index=log_index,
                    responses_received=1,
                    last_heartbeat=current_time
                )
            
            # Update history
            self.latency_history[node_id].append(latency_ms)
            self.success_history[node_id].append(1 if success else 0)
            
            # Calculate updated metrics
            avg_latency = statistics.mean(self.latency_history[node_id])
            success_rate = statistics.mean(self.success_history[node_id])
            
            # Update metrics
            metrics = self.node_metrics[node_id]
            metrics.last_response_time = current_time
            metrics.avg_latency_ms = avg_latency
            metrics.success_rate = success_rate
            metrics.last_log_index = log_index
            metrics.responses_received += 1
            
            if success:
                metrics.last_heartbeat = current_time
    
    def get_node_health(self, node_id: str) -> Optional[HealthMetrics]:
        """Get current health metrics for a node."""
        with self.lock:
            return self.node_metrics.get(node_id)
    
    def get_disruptive_nodes(self) -> List[str]:
        """Get list of nodes that are currently disruptive."""
        with self.lock:
            disruptive = []
            for node_id, metrics in self.node_metrics.items():
                if metrics.is_disruptive():
                    disruptive.append(node_id)
            return disruptive
    
    def is_node_ready_for_promotion(self, node_id: str, 
                                   leader_log_index: int) -> bool:
        """Check if a new node is ready to be promoted to voting status."""
        with self.lock:
            metrics = self.node_metrics.get(node_id)
            if not metrics:
                return False
            
            # Check all readiness criteria
            log_gap = leader_log_index - metrics.last_log_index
            has_sufficient_history = metrics.responses_received >= 20
            
            return (metrics.is_healthy() and 
                    log_gap <= 100 and 
                    has_sufficient_history)
    
    def reset_node_metrics(self, node_id: str) -> None:
        """Reset metrics for a node (useful after partition recovery)."""
        with self.lock:
            if node_id in self.node_metrics:
                del self.node_metrics[node_id]
                del self.latency_history[node_id]
                del self.success_history[node_id]
```

#### Joint Consensus State Machine

Core logic skeleton for implementing the joint consensus protocol:

```python
# joint_consensus.py - Joint consensus protocol implementation

from typing import Set, Optional, Dict
import time
import logging

class JointConsensusManager:
    """Manages the joint consensus protocol for membership changes."""
    
    def __init__(self, config_manager, health_monitor, rpc_client):
        self.config_manager = config_manager
        self.health_monitor = health_monitor
        self.rpc_client = rpc_client
        self.logger = logging.getLogger(__name__)
        
        # Timeouts and thresholds
        self.joint_consensus_timeout = 60.0  # seconds
        self.catchup_timeout = 30.0  # seconds
        
    def initiate_membership_change(self, target_nodes: Set[str]) -> bool:
        """Initiate a membership change to the target configuration."""
        # TODO 1: Validate that no membership change is currently in progress
        # TODO 2: Check that current node is the leader
        # TODO 3: Verify new nodes are caught up (if adding nodes)
        # TODO 4: Create joint configuration with old + new membership
        # TODO 5: Replicate joint configuration entry to cluster
        # TODO 6: Monitor joint consensus progress and handle timeouts
        # Hint: Use self.config_manager.start_membership_change(target_nodes)
        pass
    
    def handle_joint_consensus_commit(self, joint_entry) -> None:
        """Handle commitment of joint consensus configuration entry."""
        # TODO 1: Update local configuration manager to joint consensus state
        # TODO 2: Begin applying dual-majority rules for all decisions
        # TODO 3: Start monitoring phase for new node performance
        # TODO 4: Schedule transition to new configuration after verification
        # Hint: All subsequent operations need dual majority until phase 2
        pass
    
    def complete_membership_transition(self) -> bool:
        """Complete transition from joint consensus to new configuration."""
        # TODO 1: Verify all nodes in new configuration are healthy and current
        # TODO 2: Create new configuration entry (C_new only)
        # TODO 3: Replicate new configuration entry using joint consensus rules
        # TODO 4: Update local state to use new configuration only
        # TODO 5: Remove old nodes from monitoring and heartbeat targets
        # Hint: This requires dual majority for the final configuration change
        pass
    
    def check_dual_majority(self, responding_nodes: Set[str]) -> bool:
        """Check if responding nodes satisfy dual majority requirement."""
        # TODO 1: Get current configuration state from config manager
        # TODO 2: If in joint consensus, verify majority in BOTH old and new configs
        # TODO 3: If stable, verify majority in single current config
        # TODO 4: Return true only if appropriate majorities are satisfied
        # Hint: Use self.config_manager.check_majority(responding_nodes)
        pass
    
    def handle_membership_timeout(self) -> None:
        """Handle timeout during joint consensus phase."""
        # TODO 1: Assess why joint consensus is taking too long
        # TODO 2: Check if new nodes are responsive and keeping up
        # TODO 3: If nodes are problematic, initiate rollback to old configuration
        # TODO 4: If network issues, extend timeout and continue monitoring
        # TODO 5: Log timeout reason and actions taken for debugging
        # Hint: Use health monitor to identify specific node issues
        pass
    
    def rollback_membership_change(self) -> bool:
        """Rollback from joint consensus to original configuration."""
        # TODO 1: Create rollback configuration entry (C_old only)
        # TODO 2: Replicate rollback entry using current joint consensus rules
        # TODO 3: Update configuration manager to stable state with old config
        # TODO 4: Stop monitoring new nodes and remove from heartbeat targets
        # TODO 5: Log rollback reason and notify cluster administrators
        # Hint: Rollback still requires dual majority to ensure safety
        pass
    
    def prepare_new_nodes_for_addition(self, new_nodes: Set[str]) -> bool:
        """Ensure new nodes are ready before starting membership change."""
        # TODO 1: For each new node, establish connection and verify reachability
        # TODO 2: Send InstallSnapshot RPC to bring nodes to current state
        # TODO 3: Begin sending AppendEntries to sync recent log entries
        # TODO 4: Monitor catch-up progress using health monitor metrics
        # TODO 5: Only return True when all new nodes meet readiness criteria
        # Hint: Use self.health_monitor.is_node_ready_for_promotion()
        pass
```

#### Membership Change Integration

Integration points for adding membership change support to the existing Raft node:

```python
# Add to raft_node.py - Integration with existing consensus logic

def handle_configuration_change_request(self, new_nodes: Set[str]) -> bool:
    """Handle request to change cluster membership."""
    # TODO 1: Verify this node is currently the leader
    # TODO 2: Validate new configuration is different from current
    # TODO 3: Check no other membership change is in progress
    # TODO 4: Initiate catch-up process for any new nodes
    # TODO 5: Begin joint consensus protocol once nodes are ready
    # Hint: Only leaders can initiate membership changes
    pass

def handle_append_entries_during_joint_consensus(self, request) -> response:
    """Modified AppendEntries handling for joint consensus phase."""
    # TODO 1: Process AppendEntries using existing log matching logic
    # TODO 2: If in joint consensus, apply dual majority rules for commitment
    # TODO 3: Update commit index only when dual majority achieved
    # TODO 4: Send response with success/failure based on joint consensus rules
    # Hint: Extend existing handle_append_entries with membership awareness
    pass

def conduct_election_during_joint_consensus(self) -> None:
    """Modified election process for joint consensus phase."""
    # TODO 1: Send RequestVote RPCs to all nodes in old AND new configurations
    # TODO 2: Collect votes and verify majority from BOTH configurations
    # TODO 3: Only become leader if dual majority votes received
    # TODO 4: If elected, continue any in-progress membership change
    # Hint: Extend existing start_election with dual-majority logic
    pass
```

#### Milestone Checkpoint

After implementing membership changes, verify the system behaves correctly:

**Test Commands:**
```bash
# Start initial 3-node cluster
python -m titan_consensus start --nodes=node1,node2,node3 --node-id=node1

# In separate terminals, start other nodes
python -m titan_consensus start --nodes=node1,node2,node3 --node-id=node2
python -m titan_consensus start --nodes=node1,node2,node3 --node-id=node3

# Test adding a node
curl -X POST http://node1:8080/admin/add-node -d '{"node_id": "node4"}'

# Verify joint consensus phase
curl http://node1:8080/admin/config-status
# Should show: {"state": "joint", "old_nodes": ["node1", "node2", "node3"], "new_nodes": ["node1", "node2", "node3", "node4"]}

# Wait for completion
sleep 10
curl http://node1:8080/admin/config-status
# Should show: {"state": "stable", "nodes": ["node1", "node2", "node3", "node4"]}
```

**Expected Behavior:**
1. Membership change enters joint consensus phase requiring dual majorities
2. New nodes are caught up before gaining voting rights
3. System remains available for client operations during the change
4. Change either completes successfully or rolls back safely
5. No split-brain scenarios occur even with network partitions during changes

**Signs of Problems:**
- Membership changes get stuck in joint consensus indefinitely
- Split-brain with multiple leaders during configuration changes
- Client operations fail unnecessarily during membership changes
- New nodes are promoted too early and disrupt cluster performance
- System cannot recover properly after leader failure during membership change


## Interactions and Data Flow

> **Milestone(s):** All milestones (1-4) - defines how components communicate throughout election, replication, compaction, and membership changes

### Mental Model: Orchestra Communication

Think of Titan's component interactions like a symphony orchestra during a live performance. The conductor (Consensus Engine) coordinates the entire ensemble, sending visual cues and tempo signals to different sections. The musicians (Network Layer, Log Manager, State Machine) each have specialized roles but must communicate precisely to create harmonious output. Sheet music represents the persistent state that survives between performances, while the conductor's gestures are the volatile coordination signals. When the conductor changes (leader election), there's a brief moment of uncertainty before the new conductor establishes their rhythm and the orchestra synchronizes again. Network partitions are like acoustic problems where some musicians can't hear the conductor, forcing sections to make local decisions until communication resumes.

The key insight from this analogy is that distributed consensus requires both **structured communication protocols** (the musical notation everyone understands) and **adaptive failure handling** (what happens when the conductor disappears mid-performance). Every message must be designed to handle partial failures, delays, and ordering issues that would never occur in a single-process system.

### RPC Transport Layer

The RPC transport layer provides the fundamental communication substrate that enables consensus operations across the cluster. This layer abstracts the complexities of network communication while providing the reliability and ordering guarantees that Raft requires for correctness.

> **Decision: Reliable Message Delivery with Failure Detection**
> - **Context**: Consensus algorithms require both message delivery and the ability to detect when delivery has definitively failed
> - **Options Considered**: 
>   1. Fire-and-forget UDP with application-level retries
>   2. TCP with connection pooling and timeout management
>   3. Higher-level RPC frameworks (gRPC, HTTP/2) with built-in retry logic
> - **Decision**: TCP-based RPC with application-controlled timeouts and explicit failure detection
> - **Rationale**: TCP provides connection-oriented reliability while allowing Raft-specific timeout logic for leader election and failure detection
> - **Consequences**: Enables precise control over network timeouts critical for election safety, but requires careful connection management

| Transport Option | Reliability | Timeout Control | Complexity | Chosen? |
|-----------------|-------------|-----------------|------------|---------|
| Raw UDP + App Retries | Manual | Full Control | High | No |
| TCP + Connection Pooling | Built-in | Good Control | Medium | **Yes** |
| gRPC/HTTP2 | Built-in | Limited Control | Low | No |

The transport layer maintains **persistent connections** between cluster nodes to minimize connection establishment overhead during normal operations. Each connection carries a lightweight heartbeat mechanism independent of Raft's own heartbeat protocol - this allows the transport layer to detect network-level failures faster than application-level timeouts.

#### Message Routing and Addressing

The transport layer implements a **node-centric routing model** where each node maintains direct connections to every other node in the cluster. This full-mesh topology ensures that any node can communicate directly with any other node without intermediate routing, which is critical for the timing guarantees that Raft requires.

| Component | Responsibility | Input | Output |
|-----------|---------------|--------|---------|
| Connection Manager | Maintain persistent connections to all peers | NodeId list from cluster configuration | Active TCP connections |
| Message Serializer | Convert RPC objects to wire format | RequestVote, AppendEntries, InstallSnapshot objects | Serialized bytes |
| Failure Detector | Identify network and node failures | Connection events, timeout expiry | Node availability status |
| Retry Controller | Handle message retransmission with backoff | Failed send operations | Retry schedule or permanent failure |

The addressing scheme uses **logical node identifiers** (`NodeId` strings) that map to network addresses through a configuration service. This abstraction allows nodes to change their network addresses without disrupting the consensus protocol, which is essential for deployment flexibility and network reconfiguration.

#### Timeout Management and Failure Detection

Network timeout configuration directly impacts Raft's safety and liveness properties. The transport layer provides **tiered timeout detection** that distinguishes between temporary network congestion and permanent node failures.

| Timeout Type | Value | Purpose | Failure Action |
|--------------|-------|---------|----------------|
| Send Timeout | 100ms | Detect network congestion | Retry with exponential backoff |
| Connection Timeout | 500ms | Detect connection failures | Mark connection as failed, attempt reconnection |
| Node Timeout | 5000ms | Detect node failures | Report node as unavailable to consensus engine |
| Heartbeat Interval | 30s | Maintain connection liveness | Send keep-alive, detect silent failures |

> The critical design principle is that **network timeouts must be significantly shorter than Raft election timeouts** to ensure that network-level failures don't trigger unnecessary leader elections. A network hiccup should not destabilize the consensus protocol.

The failure detector implements **adaptive timeout adjustment** based on observed network conditions. During periods of high latency or packet loss, timeouts are automatically extended to prevent false positive failure detection. This adaptation helps maintain cluster stability during network stress while still detecting genuine failures promptly.

#### Message Ordering and Delivery Guarantees

While TCP provides ordered delivery within a single connection, the transport layer must handle **cross-connection ordering** for messages that affect the same log entries or election terms. The transport layer implements message sequencing that ensures causally related messages are delivered in the correct order.

| Ordering Scenario | Problem | Solution | Implementation |
|------------------|---------|----------|---------------|
| Concurrent AppendEntries | Messages for same log index arrive out of order | Sequence numbers per source-destination pair | Buffer and reorder messages |
| Election Messages | RequestVote arrives before term update | Term-based ordering | Queue messages until term is current |
| Snapshot Installation | Chunks arrive out of sequence | Chunk sequence numbers | Reassemble in order, reject duplicates |
| Configuration Changes | Old config messages after new config | Configuration version stamps | Reject outdated configuration messages |

The transport layer also implements **duplicate detection** using message sequence numbers and delivery acknowledgments. This prevents the same log entry from being processed multiple times if a network retry succeeds after the original message was delayed rather than lost.

### Normal Operation Flow

During stable operation, Titan follows a predictable sequence of interactions that maintains strong consistency while providing reasonable performance for client operations. Understanding this normal flow is crucial because all failure handling is designed to return the system to this stable state.

#### Client Request Processing

The normal operation flow begins when a client submits a request to the cluster. This request must be processed through the consensus protocol to ensure that all nodes apply the same state changes in the same order.

1. **Client Request Reception**: The request arrives at any cluster node, but only the current leader can process write operations. Follower nodes immediately redirect clients to the known leader using the `leader_id` from their `VolatileState`.

2. **Leader Authorization Check**: The leader verifies that it's still the legitimate leader by confirming that it has received heartbeat responses from a majority of followers within the last `HEARTBEAT_INTERVAL`. This prevents split-brain scenarios where a partitioned leader continues accepting requests.

3. **Log Entry Creation**: The leader creates a new `LogEntry` with the client's operation data, assigns it the next sequential `LogIndex`, and stamps it with the current `Term`. The entry is immediately appended to the leader's local log before any network communication.

4. **Persistent State Update**: The leader updates its `PersistentState` by appending the new entry to its log and synchronizing this change to stable storage. This ensures that even if the leader crashes immediately after this point, the operation will not be lost.

5. **Follower Replication Initiation**: The leader sends `AppendEntriesRequest` messages to all followers in parallel. Each request includes the new log entry plus the consistency check information (previous log index and term) required by Raft's log matching property.

6. **Follower Consistency Verification**: Each follower receives the append request and performs the log matching check by comparing the `prev_log_index` and `prev_log_term` values with its own log. If the check passes, the follower appends the entry to its log and acknowledges success.

7. **Majority Confirmation**: The leader waits for acknowledgment from a majority of followers (including itself). Once majority confirmation is received, the entry is considered **committed** and the leader updates its `commit_index`.

8. **State Machine Application**: The leader applies the committed entry to its state machine and updates its `last_applied` index. This step transforms the abstract log entry into concrete state changes that affect the application's behavior.

9. **Client Response**: The leader sends the operation result back to the client, indicating successful completion. The client can now rely on the fact that this operation has been durably committed across a majority of nodes.

10. **Commit Propagation**: On the next heartbeat cycle, the leader informs all followers of the new `commit_index` through the `leader_commit` field in subsequent `AppendEntriesRequest` messages. This allows followers to apply the committed entries to their own state machines.

![Log Replication Sequence](./diagrams/replication-sequence.svg)

#### Heartbeat and Maintenance Operations

Between client requests, the leader maintains cluster health through regular heartbeat operations that serve multiple purposes beyond just asserting leadership.

| Heartbeat Function | Mechanism | Frequency | Purpose |
|-------------------|-----------|-----------|---------|
| Leadership Assertion | Empty `AppendEntriesRequest` | Every `HEARTBEAT_INTERVAL` (50ms) | Prevent follower election timeouts |
| Commit Propagation | `leader_commit` field in heartbeats | Every heartbeat | Inform followers of new commit index |
| Failure Detection | Monitor response patterns | Continuous | Identify slow or failed followers |
| Log Consistency Check | Include prev_log_index/term | Every heartbeat | Detect and repair log inconsistencies |

The heartbeat cycle implements **adaptive batching** where multiple client operations received within a single heartbeat interval are bundled together in a single `AppendEntriesRequest`. This optimization reduces network overhead while maintaining the consistency guarantees required by Raft.

> **Critical Insight**: Heartbeats are not just keep-alive messages - they actively maintain the consistency and safety properties of the cluster. A leader that cannot send heartbeats to a majority of followers must step down to prevent split-brain scenarios.

#### Background Maintenance Tasks

During normal operation, several background processes run continuously to maintain cluster health and performance:

1. **Log Compaction Monitoring**: The system continuously monitors log growth and triggers snapshot creation when the log exceeds `SNAPSHOT_THRESHOLD` entries. This process runs asynchronously to avoid blocking normal operations.

2. **Performance Metric Collection**: Each node maintains `HealthMetrics` for all peers, tracking response times, success rates, and log synchronization status. These metrics inform decisions about cluster reconfiguration and failure detection.

3. **Persistent State Synchronization**: Critical state changes are immediately synchronized to stable storage using `save_to_file()` operations with explicit fsync to ensure durability across crashes.

4. **Configuration Monitoring**: The system continuously monitors for pending configuration changes and manages the two-phase commit process for cluster membership modifications.

### Failure Scenario Flows

Failure handling in distributed consensus is significantly more complex than normal operation because failures can be partial, temporary, or cascading. Titan's failure handling is designed around the principle that **safety is never compromised for availability** - the system will become unavailable rather than risk data inconsistency.

#### Leader Failure Detection and Recovery

Leader failure is the most common failure scenario and the one that most directly impacts client-visible availability. The detection and recovery process must balance rapid failure detection with stability against transient network issues.

**Failure Detection Timeline:**

1. **Heartbeat Deadline Expiry**: Followers detect potential leader failure when they don't receive heartbeats within their randomized election timeout (150-300ms). Each follower independently monitors this deadline using the `election_deadline` field in its `VolatileState`.

2. **Network vs. Node Failure Discrimination**: Before declaring the leader failed, followers attempt to distinguish between network partitions and actual node failures by checking connectivity to other cluster members. If a follower can reach other nodes but not the leader, it's more likely a leader-specific failure.

3. **Candidate State Transition**: After the election timeout expires, a follower transitions to `CANDIDATE` state by incrementing its current term, voting for itself, and updating its persistent state to record this vote.

4. **Election Initiation**: The candidate sends `RequestVoteRequest` messages to all other nodes in parallel, including its current log state (last log index and term) to ensure that only nodes with sufficiently up-to-date logs can become leader.

5. **Vote Collection and Validation**: The candidate waits for responses while maintaining a count of received votes. Votes are only valid if they come from the current term - votes from older terms are ignored as evidence of network delays.

6. **Election Resolution**: The candidate becomes leader if it receives votes from a strict majority of nodes. If it receives votes from only a minority, or if it discovers a higher term from another candidate, it reverts to follower state.

**Recovery Process:**

| Recovery Phase | Duration | Actions Taken | Success Criteria |
|---------------|----------|---------------|------------------|
| Leader Establishment | 1-2 election timeouts | New leader sends initial heartbeats | Majority of followers acknowledge leadership |
| Log Synchronization | Variable (depends on lag) | AppendEntries to bring followers current | All reachable followers match leader log |
| Client Service Resumption | <1 heartbeat interval | Begin accepting client requests | Leader has confirmed majority connectivity |
| Cluster Health Verification | 5-10 heartbeats | Verify all expected nodes responsive | Full cluster operational or failed nodes identified |

> **Key Safety Property**: During leader failure, the cluster becomes temporarily unavailable for writes, but this unavailability preserves the fundamental safety guarantee that committed data is never lost or corrupted.

#### Network Partition Handling

Network partitions represent one of the most challenging failure scenarios because they can split the cluster into multiple groups that each believe they should continue operating independently.

**Partition Detection:**

Network partitions are detected through **quorum loss** rather than direct network monitoring. When a leader cannot maintain heartbeat communication with a majority of followers, it must step down to prevent split-brain scenarios.

1. **Heartbeat Failure Pattern Analysis**: The leader tracks successful heartbeat responses from each follower. When responses fall below majority threshold for consecutive heartbeat cycles, a partition is suspected.

2. **Quorum Verification**: Before stepping down, the leader attempts to contact all followers using alternative communication paths (if available) to confirm that the communication failure represents a genuine partition rather than temporary congestion.

3. **Graceful Leadership Abdication**: If the leader cannot maintain majority contact, it transitions to follower state and stops accepting client requests. This prevents the minority partition from making decisions that conflict with the majority partition.

4. **Minority Partition Dormancy**: Nodes in the minority partition cannot elect a new leader because they lack the quorum required for valid elections. These nodes enter a dormant state where they maintain their logs but don't process client requests.

**Partition Recovery:**

| Recovery Stage | Trigger | Process | Outcome |
|---------------|---------|---------|---------|
| Connectivity Restoration | Network healing | Nodes reestablish connections | Message exchange resumes |
| Term Synchronization | Higher term discovery | Nodes update to highest observed term | All nodes share current term |
| Log Consistency Check | AppendEntries exchanges | Compare log contents for conflicts | Identify any log divergence |
| Conflict Resolution | Log inconsistencies found | Overwrite minority partition logs | Log matching property restored |
| Normal Operation | Consistency verified | Resume client request processing | Full cluster availability |

> **Partition Tolerance Principle**: Raft chooses consistency over availability during partitions. The majority partition can continue operating, but the minority partition must wait for connectivity restoration rather than risk inconsistency.

#### Node Crash and Recovery

Individual node crashes are handled through Raft's built-in recovery mechanisms that leverage persistent state to restore nodes to a consistent state after restart.

**Crash Recovery Process:**

1. **State Recovery from Storage**: The crashed node restarts by loading its `PersistentState` from stable storage using `load_from_file()`. This recovers the node's term, vote history, and log contents up to the point of the last successful write.

2. **Cluster Reintegration**: The recovered node starts in follower state and waits to receive heartbeats from the current leader. If no leader is active, it participates in the next election cycle according to normal election rules.

3. **Log Synchronization**: The current leader detects that the recovered node's log may be out of date and uses the normal `AppendEntriesRequest` protocol to bring the node current. This may involve sending many entries if the node was down for an extended period.

4. **Snapshot-Based Recovery**: If the recovered node's log is so far behind that the leader has already compacted the required entries, the leader uses `InstallSnapshotRequest` to transfer a complete snapshot of the current state.

5. **Operational Status Restoration**: Once the node's log matches the leader's committed entries, it resumes normal operation as a full cluster member capable of participating in elections and consensus decisions.

**Health Monitoring Integration:**

| Monitoring Aspect | Metric Tracked | Threshold | Action Taken |
|-------------------|----------------|-----------|--------------|
| Recovery Progress | Log entries synchronized per second | <100 entries/sec | Alert administrators about slow recovery |
| Recovery Duration | Time from restart to current | >60 seconds | Consider snapshot transfer instead of log replay |
| Resource Usage | CPU/Memory during catch-up | >80% sustained | Throttle recovery to maintain cluster performance |
| Network Impact | Bandwidth used for recovery | >50% of available | Prioritize normal client operations |

#### Cascading Failure Prevention

Titan implements multiple mechanisms to prevent single node failures from cascading into cluster-wide outages.

**Failure Isolation Strategies:**

1. **Resource Throttling**: Nodes under stress automatically throttle their participation in consensus operations rather than timing out and triggering unnecessary elections.

2. **Adaptive Timeouts**: Election timeouts are automatically extended during periods of high cluster load or network latency to prevent spurious leader changes.

3. **Load Shedding**: Overloaded leaders can temporarily reject new client requests while continuing to process already-committed operations, maintaining cluster stability.

4. **Circuit Breaker Pattern**: Nodes that repeatedly fail operations are temporarily excluded from critical path operations while remaining part of the consensus quorum for safety.

⚠️ **Pitfall: Thundering Herd Elections**
When multiple followers simultaneously detect leader failure, they may all transition to candidate state and trigger a series of conflicting elections. Titan prevents this through **randomized election timeouts** where each follower waits a different amount of time before starting an election. This staggered approach ensures that usually only one follower becomes candidate at a time, reducing the likelihood of split votes and election conflicts.

⚠️ **Pitfall: Stale Leader Acceptance**
After a network partition heals, nodes may temporarily accept commands from a leader whose term is outdated. Titan prevents this by requiring every RPC to include the sender's current term, and recipients automatically update their term and revert to follower state if they receive a message from a higher term. This ensures that stale leaders are immediately detected and neutralized.

⚠️ **Pitfall: Snapshot Transfer Blocking**
Large snapshot transfers during node recovery can block normal cluster operations if not properly managed. Titan addresses this by implementing **chunked snapshot transfer** with flow control, where snapshots are sent in small pieces that don't interfere with normal heartbeat and replication operations.

### Implementation Guidance

The implementation of interactions and data flow requires careful attention to concurrency, network programming, and failure handling. The following guidance provides concrete starting points for building a robust consensus system.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| RPC Transport | HTTP/1.1 with JSON over persistent connections | gRPC with Protocol Buffers |
| Connection Management | Simple connection pooling with manual retry logic | Advanced connection management with automatic failover |
| Message Serialization | JSON with manual schema validation | Protocol Buffers with schema evolution |
| Network Failure Detection | TCP socket timeouts with exponential backoff | Advanced health checking with multiple detection methods |
| Threading Model | Thread-per-connection with blocking I/O | Async I/O with event loop and futures |
| Logging and Observability | Standard logging with structured output | Distributed tracing with metrics collection |

#### Recommended File Structure

```
titan/
  core/
    consensus_engine.py          ← Main consensus coordinator
    node_state.py                ← Node state management
    election.py                  ← Election logic from previous milestone
    replication.py               ← Log replication from previous milestone
  transport/
    rpc_transport.py             ← RPC transport layer (this section)
    message_serialization.py     ← Message encoding/decoding
    connection_manager.py        ← Connection pooling and management
    failure_detector.py          ← Network failure detection
  flow/
    request_processor.py         ← Client request handling (this section)
    failure_handler.py           ← Failure scenario coordination (this section)
    heartbeat_manager.py         ← Heartbeat and maintenance operations
  tests/
    test_normal_operations.py    ← Normal flow testing
    test_failure_scenarios.py    ← Failure injection testing
    test_transport_layer.py      ← Network layer testing
```

#### Infrastructure Starter Code

**Complete RPC Transport Implementation:**

```python
import asyncio
import json
import logging
import time
from typing import Dict, Optional, Any, Callable
from dataclasses import asdict
import websockets
from websockets.exceptions import ConnectionClosed, InvalidURI

logger = logging.getLogger(__name__)

class RPCTransport:
    """Complete RPC transport layer with connection pooling and failure detection."""
    
    def __init__(self, node_id: str, listen_port: int, timeout_ms: int = 5000):
        self.node_id = node_id
        self.listen_port = listen_port
        self.timeout_seconds = timeout_ms / 1000.0
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.running = False
        self.server = None
        
    async def start(self) -> None:
        """Start the RPC transport server and begin accepting connections."""
        self.running = True
        self.server = await websockets.serve(
            self._handle_connection,
            "localhost", 
            self.listen_port
        )
        logger.info(f"RPC transport listening on port {self.listen_port}")
    
    async def stop(self) -> None:
        """Gracefully shutdown the transport layer."""
        self.running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        # Close all existing connections
        for conn in self.connections.values():
            await conn.close()
        self.connections.clear()
    
    def register_handler(self, message_type: str, handler: Callable) -> None:
        """Register a handler function for a specific message type."""
        self.message_handlers[message_type] = handler
    
    async def send_rpc(self, target_node: str, target_port: int, 
                      message_type: str, message_data: Any) -> Optional[Any]:
        """Send an RPC message to target node and wait for response."""
        try:
            uri = f"ws://localhost:{target_port}"
            
            async with websockets.connect(uri, timeout=self.timeout_seconds) as websocket:
                # Prepare message
                rpc_message = {
                    "type": message_type,
                    "sender": self.node_id,
                    "data": asdict(message_data) if hasattr(message_data, '__dict__') else message_data,
                    "timestamp": time.time()
                }
                
                # Send message
                await websocket.send(json.dumps(rpc_message))
                
                # Wait for response
                response_raw = await asyncio.wait_for(
                    websocket.recv(), 
                    timeout=self.timeout_seconds
                )
                
                response = json.loads(response_raw)
                return response.get("data")
                
        except asyncio.TimeoutError:
            logger.warning(f"RPC timeout sending {message_type} to {target_node}")
            return None
        except ConnectionClosed:
            logger.warning(f"Connection closed when sending {message_type} to {target_node}")
            return None
        except Exception as e:
            logger.error(f"RPC error sending {message_type} to {target_node}: {e}")
            return None
    
    async def _handle_connection(self, websocket, path):
        """Handle incoming RPC connections and route messages to handlers."""
        try:
            async for message_raw in websocket:
                try:
                    message = json.loads(message_raw)
                    message_type = message.get("type")
                    
                    if message_type in self.message_handlers:
                        handler = self.message_handlers[message_type]
                        response_data = await handler(message.get("data"))
                        
                        # Send response back
                        response = {
                            "type": f"{message_type}_response",
                            "data": response_data,
                            "timestamp": time.time()
                        }
                        await websocket.send(json.dumps(response))
                    else:
                        logger.warning(f"No handler for message type: {message_type}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON received: {e}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    
        except ConnectionClosed:
            logger.info("Client connection closed")
        except Exception as e:
            logger.error(f"Connection handler error: {e}")
```

**Complete Message Serialization:**

```python
import json
from typing import Dict, Any, Type, TypeVar
from dataclasses import asdict, is_dataclass

from titan.data_model import (
    RequestVoteRequest, RequestVoteResponse,
    AppendEntriesRequest, AppendEntriesResponse,
    InstallSnapshotRequest, InstallSnapshotResponse,
    LogEntry
)

T = TypeVar('T')

class MessageSerializer:
    """Handles serialization and deserialization of all Raft RPC messages."""
    
    # Registry of message types to their corresponding classes
    MESSAGE_TYPES = {
        'request_vote': RequestVoteRequest,
        'request_vote_response': RequestVoteResponse,
        'append_entries': AppendEntriesRequest,
        'append_entries_response': AppendEntriesResponse,
        'install_snapshot': InstallSnapshotRequest,
        'install_snapshot_response': InstallSnapshotResponse,
        'log_entry': LogEntry
    }
    
    @staticmethod
    def serialize(obj: Any) -> str:
        """Convert a Raft message object to JSON string."""
        if is_dataclass(obj):
            data = asdict(obj)
            # Add type information for deserialization
            data['__type__'] = obj.__class__.__name__
            return json.dumps(data)
        else:
            return json.dumps(obj)
    
    @staticmethod
    def deserialize(json_str: str, expected_type: Type[T] = None) -> T:
        """Convert JSON string back to Raft message object."""
        data = json.loads(json_str)
        
        if expected_type and is_dataclass(expected_type):
            # Remove type metadata if present
            data.pop('__type__', None)
            return expected_type(**data)
        
        # Auto-detect type from metadata
        type_name = data.get('__type__')
        if type_name:
            for msg_type_name, msg_class in MessageSerializer.MESSAGE_TYPES.items():
                if msg_class.__name__ == type_name:
                    data.pop('__type__')
                    return msg_class(**data)
        
        # Return raw dictionary if no type detected
        return data
    
    @staticmethod
    def serialize_log_entries(entries: list) -> str:
        """Efficiently serialize a list of log entries."""
        serialized_entries = []
        for entry in entries:
            if isinstance(entry, LogEntry):
                serialized_entries.append(asdict(entry))
            else:
                serialized_entries.append(entry)
        return json.dumps(serialized_entries)
    
    @staticmethod
    def deserialize_log_entries(json_str: str) -> list:
        """Efficiently deserialize a list of log entries."""
        entries_data = json.loads(json_str)
        entries = []
        for entry_data in entries_data:
            if isinstance(entry_data, dict):
                entries.append(LogEntry(**entry_data))
            else:
                entries.append(entry_data)
        return entries
```

#### Core Logic Skeleton

**Request Processor (Student Implementation):**

```python
import asyncio
import logging
from typing import Optional, Any
from titan.core.consensus_engine import ConsensusEngine
from titan.transport.rpc_transport import RPCTransport
from titan.data_model import LogEntry, NodeState

logger = logging.getLogger(__name__)

class RequestProcessor:
    """Handles client requests and coordinates their processing through the consensus protocol."""
    
    def __init__(self, consensus_engine: ConsensusEngine, transport: RPCTransport):
        self.consensus = consensus_engine
        self.transport = transport
        self.pending_requests: Dict[str, asyncio.Future] = {}
    
    async def process_client_request(self, operation_data: bytes) -> Optional[Any]:
        """
        Process a client request through the Raft consensus protocol.
        Returns the result of applying the operation, or None if failed.
        """
        # TODO 1: Check if this node is the current leader
        # Hint: Use self.consensus.volatile_state.node_state == NodeState.LEADER
        
        # TODO 2: If not leader, redirect client to known leader
        # Hint: Return error with leader_id from self.consensus.volatile_state.leader_id
        
        # TODO 3: Verify leader hasn't lost quorum recently
        # Hint: Check that majority of followers responded to recent heartbeats
        
        # TODO 4: Create new LogEntry with operation data
        # Hint: Use current term, next log index, current timestamp
        
        # TODO 5: Append entry to leader's local log
        # Hint: Use self.consensus.log_manager.append_entry(entry)
        
        # TODO 6: Persist the log entry to stable storage
        # Hint: Use self.consensus.persistent_state.save_to_file()
        
        # TODO 7: Send AppendEntries RPC to all followers in parallel
        # Hint: Use asyncio.gather() to send to all followers simultaneously
        
        # TODO 8: Wait for majority acknowledgment
        # Hint: Count successful responses, need (cluster_size // 2) + 1
        
        # TODO 9: If majority confirms, mark entry as committed
        # Hint: Update self.consensus.volatile_state.commit_index
        
        # TODO 10: Apply committed entry to state machine
        # Hint: Use self.consensus.state_machine.apply_entry(entry)
        
        # TODO 11: Return operation result to client
        # Hint: Return the result from state machine application
        
        pass
    
    async def handle_append_entries_response(self, follower_id: str, 
                                           response: 'AppendEntriesResponse') -> None:
        """Handle response from follower to AppendEntries RPC."""
        # TODO 1: Check if response is for current term
        # TODO 2: If response indicates success, update follower's match_index
        # TODO 3: If response indicates failure, handle log inconsistency
        # TODO 4: Check if new entries can now be committed
        # TODO 5: Notify any pending client requests that are now committed
        pass
    
    async def redirect_to_leader(self, operation_data: bytes) -> Optional[Any]:
        """Redirect client request to the current known leader."""
        # TODO 1: Get current leader_id from volatile state
        # TODO 2: Look up leader's network address
        # TODO 3: Forward request to leader using RPC transport
        # TODO 4: Return leader's response to client
        pass
```

**Failure Handler (Student Implementation):**

```python
import asyncio
import logging
import time
from typing import Set, Optional
from titan.core.consensus_engine import ConsensusEngine
from titan.data_model import NodeState, NodeId

logger = logging.getLogger(__name__)

class FailureHandler:
    """Coordinates failure detection and recovery across the cluster."""
    
    def __init__(self, consensus_engine: ConsensusEngine):
        self.consensus = consensus_engine
        self.failed_nodes: Set[NodeId] = set()
        self.partition_detected = False
        self.last_majority_contact = time.time()
    
    async def handle_leader_failure(self) -> None:
        """Coordinate response to detected leader failure."""
        # TODO 1: Verify that leader is actually unreachable
        # Hint: Try direct communication with suspected failed leader
        
        # TODO 2: Check if this node should start an election
        # Hint: Wait for randomized election timeout to prevent conflicts
        
        # TODO 3: Transition to CANDIDATE state if timeout expires
        # Hint: Use self.consensus.start_election()
        
        # TODO 4: Send RequestVote RPCs to all other nodes
        # Hint: Include current log state to ensure election safety
        
        # TODO 5: Collect votes and determine election outcome
        # Hint: Need majority votes to become leader
        
        # TODO 6: If elected, transition to LEADER and start heartbeats
        # Hint: Use self.consensus.become_leader()
        
        # TODO 7: If not elected, return to FOLLOWER and wait for new leader
        # Hint: Reset election timeout and wait for heartbeats
        
        pass
    
    async def handle_network_partition(self, reachable_nodes: Set[NodeId]) -> None:
        """Handle detected network partition."""
        # TODO 1: Determine if this partition has quorum
        # Hint: Check if reachable_nodes contains majority of cluster
        
        # TODO 2: If no quorum, step down from leadership (if leader)
        # Hint: Transition to FOLLOWER and stop accepting client requests
        
        # TODO 3: If has quorum, continue operating but mark unreachable nodes
        # Hint: Update self.failed_nodes with unreachable node IDs
        
        # TODO 4: Implement exponential backoff for reconnection attempts
        # Hint: Try to reconnect with increasing delays
        
        # TODO 5: When partition heals, verify log consistency
        # Hint: Use normal AppendEntries protocol to check logs
        
        # TODO 6: Resolve any log conflicts from partition
        # Hint: Minority partition logs must be overwritten
        
        pass
    
    async def handle_node_recovery(self, recovered_node: NodeId) -> None:
        """Handle a previously failed node rejoining the cluster."""
        # TODO 1: Remove node from failed_nodes set
        # TODO 2: Check if recovered node's log is current
        # TODO 3: If log is stale, start catch-up replication
        # TODO 4: If log is very stale, consider snapshot installation
        # TODO 5: Once caught up, resume normal operations with node
        pass
    
    def detect_partition(self) -> bool:
        """Check if network partition is affecting this node."""
        # TODO 1: Count nodes that responded to recent heartbeats
        # TODO 2: Check if response count represents majority
        # TODO 3: Consider network latency and timeout patterns
        # TODO 4: Return True if partition is suspected
        pass
```

#### Milestone Checkpoints

**After implementing RPC Transport:**
- Run `python -m pytest tests/test_transport_layer.py -v`
- Expected: All transport tests pass, connections established successfully
- Manual verification: Start two nodes, send test message between them
- Signs of problems: Connection timeouts, serialization errors, port conflicts

**After implementing Normal Operation Flow:**
- Run `python -m pytest tests/test_normal_operations.py -v`
- Expected: Client requests processed correctly, log replication works
- Manual verification: Send client request to leader, verify log entry on followers
- Signs of problems: Requests hang indefinitely, followers don't receive entries

**After implementing Failure Scenarios:**
- Run `python -m pytest tests/test_failure_scenarios.py -v`
- Expected: Elections work, partitions handled gracefully, recovery succeeds
- Manual verification: Kill leader node, verify new leader elected within election timeout
- Signs of problems: Split-brain scenarios, failed elections, data inconsistency

#### Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Client requests hang | Leader not receiving majority responses | Check follower logs for AppendEntries failures | Verify network connectivity, check follower health |
| Frequent leader elections | Network instability or timeout misconfiguration | Monitor election timeout vs network latency | Increase election timeouts, improve network stability |
| Log inconsistency after partition | Minority partition accepted writes | Compare logs across nodes for conflicts | Implement proper quorum checking before accepting writes |
| RPC timeouts | Network congestion or overloaded nodes | Monitor network latency and node CPU usage | Implement adaptive timeouts, add backpressure |
| Failed node won't rejoin | Snapshot required but transfer failing | Check snapshot size and network capacity | Implement chunked snapshot transfer with retries |


## Error Handling and Edge Cases

> **Milestone(s):** All milestones (1-4) - comprehensive failure handling is critical throughout election, replication, compaction, and membership changes

### Mental Model: Emergency Response System

Think of Titan's error handling like a well-designed emergency response system for a large city. Just as emergency services must prepare for different types of disasters (earthquakes, fires, medical emergencies), a consensus system must anticipate and handle various failure modes. Each type of failure requires specific detection mechanisms (like earthquake sensors or smoke detectors) and coordinated response protocols (like evacuation procedures or medical triage). The key insight is that failures are not exceptional cases to be ignored—they are normal operating conditions that must be handled systematically.

Like emergency responders who practice disaster scenarios regularly, consensus systems must be designed with failure as the default assumption. The system must remain safe even when multiple failures occur simultaneously, just as emergency protocols must work even when communication systems are damaged. The goal is not to prevent all failures (which is impossible), but to detect them quickly and recover gracefully while maintaining critical safety properties.

### Failure Mode Taxonomy

Understanding failure modes in distributed consensus is like understanding different types of medical emergencies—each requires different diagnosis and treatment. Failures in Raft can be categorized along several dimensions: scope (individual node vs. network-wide), duration (transient vs. persistent), and impact (safety-threatening vs. liveness-affecting). This taxonomy helps us design appropriate detection and recovery mechanisms for each category.

**Node-Level Failures** represent problems affecting individual nodes in the cluster. These are the most common failure type and include both crash failures (where the node stops responding entirely) and Byzantine-style behavior (where the node responds incorrectly). While Raft is designed to handle fail-stop behavior, it must also detect and isolate nodes that exhibit corrupt or malicious responses.

| Failure Type | Characteristics | Detection Method | Safety Impact | Liveness Impact |
|--------------|-----------------|------------------|---------------|-----------------|
| Node Crash | Complete loss of responsiveness | Heartbeat timeout | Low (if minority) | Low (if minority) |
| Partial Failure | Responds to some but not all RPCs | Selective RPC failure | Medium | Medium |
| Slow Node | Responds but with high latency | Latency monitoring | Low | High |
| Corrupt State | Sends invalid responses | Response validation | High | Medium |
| Clock Skew | Incorrect timestamp behavior | Time drift detection | Medium | Low |
| Disk Full | Cannot persist state changes | Write operation failures | High | High |
| Memory Exhaustion | Cannot process requests | Resource monitoring | Medium | High |

**Network-Level Failures** affect communication between nodes and are particularly dangerous because they can cause split-brain scenarios. These failures require careful detection to distinguish between node failures and network partitions. The key insight is that network failures are often asymmetric—node A might be able to reach node B, but not vice versa.

| Failure Type | Characteristics | Detection Method | Split-Brain Risk | Recovery Complexity |
|--------------|-----------------|------------------|------------------|---------------------|
| Total Partition | Complete network isolation | Bidirectional timeout | High | High |
| Partial Partition | Some nodes isolated | Partial connectivity | Very High | Very High |
| Message Loss | Intermittent packet drops | Retry pattern analysis | Low | Low |
| Message Delay | High but variable latency | Latency distribution | Medium | Medium |
| Message Corruption | Garbled network packets | Checksum validation | Medium | Low |
| Asymmetric Partition | Unidirectional connectivity | Directional timeout | Very High | Very High |

**Timing Failures** occur when the system's timing assumptions are violated. These are subtle but dangerous because they can cause safety violations without obvious symptoms. Raft's safety depends on timeouts being properly configured relative to network latency and processing delays.

| Failure Type | Root Cause | Symptoms | Safety Risk | Mitigation Strategy |
|--------------|------------|----------|-------------|---------------------|
| Election Timeout Too Short | High network latency | Frequent elections | Medium | Adaptive timeout |
| Heartbeat Too Slow | Leader overload | False leader failure | Low | Leader load shedding |
| Clock Drift | NTP failure | Timestamp skew | Medium | Clock synchronization |
| Process Pause | GC or swap | Apparent node failure | Medium | Pause detection |
| Network Congestion | Bandwidth saturation | Variable latency | Low | Traffic shaping |

**State Corruption Failures** involve inconsistencies in persistent state, which threaten the fundamental safety guarantees of the consensus system. These failures often result from disk corruption, software bugs, or improper shutdown procedures. They are particularly dangerous because they may not be immediately detectable.

| Corruption Type | Location | Detection Method | Recovery Approach | Data Loss Risk |
|-----------------|----------|------------------|-------------------|----------------|
| Log Entry Corruption | Persistent log | Entry validation | Log repair/rebuild | Medium |
| Term/Vote Corruption | Persistent state | State validation | State reset | Low |
| Snapshot Corruption | Snapshot files | Checksum verification | Snapshot rebuild | High |
| Metadata Corruption | File system | File integrity check | Metadata reconstruction | Medium |
| Index Inconsistency | Log indices | Index validation | Log compaction | Low |

> **Critical Insight**: The most dangerous failures are those that violate safety while appearing to maintain liveness. A system that fails loudly is better than one that silently corrupts data.

### Failure Detection Strategies

Effective failure detection in consensus systems is like having a comprehensive medical monitoring system—it must quickly identify problems while avoiding false alarms that could trigger unnecessary interventions. The challenge is that consensus systems operate in an asynchronous environment where it's impossible to distinguish between a slow node and a failed node with certainty. Therefore, detection strategies must be probabilistic and tunable.

**Heartbeat-Based Detection** is the primary mechanism for detecting node and network failures. The leader sends regular heartbeat messages (empty `AppendEntries` requests) to all followers, and followers expect to receive these messages within the election timeout period. However, implementing heartbeat detection correctly requires careful attention to timing and failure modes.

The heartbeat system maintains health metrics for each node to distinguish between different types of failures. A node that consistently fails to respond is likely crashed, while a node with high variance in response times may be experiencing load or network issues. The system tracks multiple metrics to build a comprehensive picture of each node's health.

| Metric | Purpose | Collection Method | Threshold | Action Triggered |
|--------|---------|------------------|-----------|------------------|
| Response Rate | Detect total failure | RPC success/failure ratio | < 50% over 30s | Mark node as failed |
| Response Latency | Detect performance issues | RPC round-trip time | > 3x median | Flag as slow node |
| Last Response Time | Detect recent failure | Timestamp tracking | > 2x election timeout | Suspect failure |
| Consecutive Failures | Detect persistent issues | Failure count | > 5 consecutive | Initiate recovery |
| Log Index Progress | Detect replication lag | Log advancement rate | < 10% of leader rate | Start catch-up |

**Response Validation** ensures that nodes are not only responding but responding correctly. This detects corrupt state, software bugs, and Byzantine behavior. Each RPC response is validated against expected formats and logical consistency rules.

```python
def record_rpc_response(node_id: str, success: bool, latency_ms: float, log_index: int) -> None:
    """Update health metrics after receiving RPC response"""
    # TODO 1: Update response rate moving average
    # TODO 2: Record latency in histogram for percentile calculation  
    # TODO 3: Check for consecutive failure patterns
    # TODO 4: Update last successful response timestamp
    # TODO 5: Validate log index progression for replication monitoring
```

**Network Partition Detection** is more complex because it requires distinguishing between node failures and communication failures. A partition is suspected when a node can reach some peers but not others, or when the pattern of failures suggests network-level issues rather than individual node problems.

| Detection Signal | Interpretation | Confidence Level | Response Strategy |
|------------------|----------------|------------------|-------------------|
| All nodes unreachable | Local network failure | High | Self-isolate |
| Partial reachability | Network partition | Medium | Form sub-cluster |
| Asymmetric failures | Routing problems | Medium | Investigate connectivity |
| Timeout patterns | Congestion vs partition | Low | Increase timeouts |
| Geographic correlation | Data center partition | High | Activate DR procedures |

**Adaptive Timeout Management** adjusts failure detection sensitivity based on observed network conditions. When the network is stable, timeouts can be aggressive to enable fast failure detection. When the network is unstable, timeouts must be conservative to avoid false positives.

> **Decision: Adaptive Election Timeouts**
> - **Context**: Fixed election timeouts cause either slow failure detection (timeouts too long) or false positives (timeouts too short)
> - **Options Considered**: Fixed timeouts, fully adaptive timeouts, bounded adaptive timeouts
> - **Decision**: Bounded adaptive timeouts that adjust within `ELECTION_TIMEOUT_MIN` and `ELECTION_TIMEOUT_MAX`
> - **Rationale**: Provides fast detection in stable networks while preventing false positives during network instability
> - **Consequences**: Requires latency monitoring and timeout adjustment logic, but significantly improves cluster stability

### Recovery Protocols

Recovery in consensus systems is like emergency medical procedures—there are standard protocols for each type of failure, but the specific treatment depends on the severity and context of the problem. The key principle is that recovery must preserve safety properties even if it temporarily affects liveness. It's better to have a temporarily unavailable cluster than a cluster that violates consistency guarantees.

**Node Crash Recovery** is the most straightforward recovery scenario. When a node crashes and restarts, it must reconstruct its volatile state from persistent state and catch up on any log entries it missed while down. The recovery process follows a careful sequence to ensure the node doesn't disrupt the cluster during its recovery.

1. **State Reconstruction Phase**: The node loads its persistent state from disk, including current term, vote record, and log entries. It validates the integrity of this state using checksums and consistency checks.

2. **Cluster Reconnection Phase**: The node contacts other cluster members to determine the current leader and term. If the node's term is stale, it updates to the current term and resets its vote.

3. **Log Catch-up Phase**: If the node's log is behind the leader's log, it receives missing entries through normal `AppendEntries` RPCs or `InstallSnapshot` if it's too far behind.

4. **Service Resumption Phase**: Once the node is caught up to within a reasonable threshold of the leader, it begins participating in normal cluster operations.

| Recovery Step | Validation Required | Failure Handling | Time Limit |
|---------------|-------------------|------------------|------------|
| Load persistent state | File integrity, checksum | Reset to clean state | 30 seconds |
| Contact cluster | Network connectivity | Retry with backoff | 60 seconds |
| Catch up log | Entry consistency | Request snapshot | 300 seconds |
| Resume service | Log currency | Continue catch-up | No limit |

**Network Partition Recovery** is more complex because it requires careful coordination to avoid split-brain scenarios. When a partition heals, the system must determine which side of the partition was legitimate and reconcile any divergent state.

The recovery process uses term numbers as the primary mechanism for determining legitimacy. The side of the partition that achieved higher term numbers had a legitimate leader and quorum, so its state takes precedence. Nodes from the minority partition must discard any uncommitted entries and accept the majority's log.

1. **Partition Detection**: Nodes detect that previously unreachable nodes are now reachable again through heartbeat restoration or explicit connectivity testing.

2. **Term Reconciliation**: Nodes exchange term information to determine which side had legitimate leadership during the partition.

3. **Log Reconciliation**: Nodes with stale logs must truncate any conflicting entries and accept the authoritative log from the legitimate leader.

4. **State Machine Synchronization**: If state machines diverged during the partition, the minority side must restore from the majority's snapshot or replay the correct log entries.

**Leader Failure Recovery** requires electing a new leader and ensuring all committed entries are preserved. The election process naturally handles this recovery, but there are specific steps to ensure no data is lost and the new leader is properly established.

| Recovery Phase | Key Actions | Safety Checks | Completion Criteria |
|----------------|-------------|---------------|-------------------|
| Detect failure | Heartbeat timeout | Confirm with multiple nodes | Consensus on failure |
| Start election | Increment term, request votes | Verify log currency | Majority votes received |
| Establish leadership | Send heartbeats | Confirm follower acceptance | All reachable nodes follow |
| Reconcile logs | Identify committed entries | Validate entry consistency | All followers synchronized |

**Cascade Failure Recovery** handles situations where multiple failures occur simultaneously or in rapid succession. This requires prioritizing recovery actions and managing resource constraints during the recovery process.

The system implements a recovery hierarchy that addresses the most critical failures first: network connectivity, leader establishment, log consistency, and finally performance optimization. Each level must be stabilized before proceeding to the next.

> **Decision: Recovery Prioritization**
> - **Context**: Multiple simultaneous failures can overwhelm recovery systems and cause thrashing
> - **Options Considered**: Parallel recovery, sequential recovery, adaptive prioritization
> - **Decision**: Sequential recovery with fixed priority order: safety first, then liveness
> - **Rationale**: Prevents recovery actions from interfering with each other and ensures safety properties are never compromised for performance
> - **Consequences**: Recovery may be slower during cascade failures, but system remains consistent

### Split-Brain Prevention

Split-brain prevention is the most critical aspect of consensus system safety—it's like ensuring that only one person can hold the nuclear launch codes at any time. A split-brain scenario occurs when multiple nodes believe they are the leader simultaneously, which can lead to divergent state and irreparable data corruption. Preventing this scenario requires multiple layers of protection, each designed to catch failures that earlier layers might miss.

**Quorum-Based Leadership** is the fundamental mechanism that prevents split-brain scenarios. A node can only become leader if it receives votes from a strict majority of the cluster. Since majorities overlap, it's mathematically impossible for two nodes to both receive majority votes in the same term.

However, implementing quorum-based leadership correctly requires careful attention to several subtle issues. The definition of "majority" must be based on the committed cluster configuration, not the currently reachable nodes. A node that becomes isolated must not be able to declare itself leader just because it can't reach the other nodes.

| Cluster Size | Majority Threshold | Max Failures Tolerated | Split-Brain Protection |
|--------------|-------------------|----------------------|----------------------|
| 1 | 1 | 0 | N/A (single point of failure) |
| 3 | 2 | 1 | Strong |
| 5 | 3 | 2 | Strong |
| 7 | 4 | 3 | Strong |
| 9 | 5 | 4 | Strong |

**Term-Based Temporal Ordering** provides a second layer of protection by ensuring that leadership is totally ordered in time. Each leader election increments the term number, and nodes reject messages from leaders with stale terms. This prevents old leaders from continuing to operate after losing leadership.

The term mechanism handles subtle timing issues like network delays and process pauses. Even if a leader is temporarily isolated and then reconnects, it will discover that its term is stale and immediately step down. The system guarantees that leadership transitions are unambiguous and irreversible.

**Vote Persistence** ensures that a node cannot vote for multiple candidates in the same term, even across crashes and restarts. Each node persists its vote before responding to `RequestVote` RPCs, and this vote record is checked on restart to prevent duplicate voting.

```python
def handle_request_vote(self, request: RequestVoteRequest) -> RequestVoteResponse:
    """Process vote request with split-brain prevention checks"""
    # TODO 1: Check if request term is current or newer
    # TODO 2: Verify candidate's log is at least as up-to-date as ours
    # TODO 3: Check if we've already voted in this term
    # TODO 4: Persist vote decision before responding
    # TODO 5: Reset election timeout if granting vote
    # TODO 6: Return vote decision with current term
```

**Leadership Validation** requires followers to actively validate that their leader is legitimate before accepting commands. This catches scenarios where a leader might have been partitioned and lost quorum but continues sending commands to a subset of followers.

Followers perform several checks on each `AppendEntries` request: the leader's term must be current, the log consistency checks must pass, and the commit index progression must be reasonable. If any of these checks fail, the follower rejects the request and may trigger a new election.

| Validation Check | Purpose | Failure Response | Safety Impact |
|------------------|---------|------------------|---------------|
| Term currency | Detect stale leader | Reject with current term | Critical |
| Log consistency | Detect log corruption | Reject with conflict info | Critical |
| Commit progression | Detect invalid commits | Reject and investigate | Critical |
| Leader identity | Detect impersonation | Reject and alert | High |
| Message integrity | Detect corruption | Reject and retry | Medium |

**Lease-Based Extensions** provide additional protection for systems that require stronger consistency guarantees. Under this approach, followers grant the leader a "lease" during which they promise not to elect a new leader. This prevents leadership changes due to transient network issues while maintaining safety.

The lease mechanism requires careful clock management and bounded clock skew between nodes. Leases must be shorter than election timeouts, and followers must refuse to grant votes while their leader lease is active. This provides stronger availability guarantees while maintaining safety.

**Monitoring and Alerting** for split-brain scenarios involves detecting patterns that might indicate multiple leaders or other safety violations. The system continuously monitors for conflicting heartbeats, duplicate leader claims, and log inconsistencies that might indicate a split-brain condition.

| Alert Condition | Severity | Investigation Steps | Immediate Response |
|-----------------|----------|--------------------|--------------------|
| Multiple heartbeat sources | Critical | Check network topology | Isolate conflicting nodes |
| Term regression | Critical | Examine log history | Stop all operations |
| Conflicting commits | Critical | Compare state machines | Initiate emergency recovery |
| Vote count anomaly | High | Audit vote records | Verify cluster membership |
| Leadership churn | Medium | Check network stability | Adjust timeouts |

> **Critical Safety Principle**: When in doubt, choose availability loss over consistency violation. A temporarily unavailable cluster can be recovered, but corrupted data may be irreparable.

⚠️ **Pitfall: False Split-Brain Detection**
Many implementations create false split-brain alerts by confusing normal leadership changes with actual split-brain scenarios. A rapid series of leader elections during network instability is not a split-brain—it's the system working correctly to maintain safety. True split-brain requires simultaneous leaders in the same term, which should be impossible if the system is implemented correctly.

⚠️ **Pitfall: Minority Partition Behavior**
Nodes in a minority partition must not attempt to elect a leader or process client requests, even if they can communicate among themselves. Some implementations incorrectly allow minority partitions to operate under the assumption that they might be the majority. This violates safety and can cause data corruption when the partition heals.

⚠️ **Pitfall: Stale Leader Detection**
A leader that loses contact with a majority of followers must immediately step down, even if it can still reach some followers. Some implementations allow stale leaders to continue operating as long as they have any followers, which can lead to split-brain scenarios when the network heals.

### Implementation Guidance

The error handling system requires robust infrastructure for monitoring, detecting, and recovering from various failure modes. This implementation provides the foundation for building a production-ready consensus system that can handle real-world failure scenarios.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Health Monitoring | Simple timeout tracking | Comprehensive metrics with Prometheus |
| Failure Detection | Basic heartbeat system | ML-based anomaly detection |
| Recovery Coordination | Manual recovery procedures | Automated recovery with circuit breakers |
| Split-Brain Prevention | Term validation only | Lease-based validation with NTP sync |
| Alerting System | Log-based alerts | Real-time monitoring dashboard |

**Recommended Module Structure:**

```
internal/consensus/
├── failure_detector.py      ← Health monitoring and failure detection
├── recovery_manager.py      ← Failure recovery coordination  
├── health_metrics.py        ← Node health tracking and analysis
├── split_brain_guard.py     ← Split-brain prevention mechanisms
├── adaptive_timeouts.py     ← Dynamic timeout adjustment
└── failure_scenarios.py     ← Test failure injection framework
```

**Infrastructure Starter Code:**

```python
import time
import threading
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics

@dataclass
class HealthMetrics:
    """Track health metrics for failure detection"""
    node_id: str
    last_response_time: float = 0.0
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    last_log_index: int = 0
    responses_received: int = 0
    last_heartbeat: float = 0.0
    
    # Internal tracking
    _latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    _successes: deque = field(default_factory=lambda: deque(maxlen=50))
    
    def update_response(self, success: bool, latency_ms: float, log_index: int):
        """Update metrics with new response data"""
        self.last_response_time = time.time()
        self.responses_received += 1
        self.last_log_index = log_index
        
        self._latencies.append(latency_ms)
        self._successes.append(success)
        
        if self._latencies:
            self.avg_latency_ms = statistics.mean(self._latencies)
        
        if self._successes:
            self.success_rate = sum(self._successes) / len(self._successes)
    
    def is_healthy(self) -> bool:
        """Determine if node is healthy based on metrics"""
        now = time.time()
        stale_threshold = 5.0  # 5 seconds
        
        return (
            self.success_rate >= 0.5 and
            now - self.last_response_time < stale_threshold and
            self.avg_latency_ms < 1000  # 1 second max latency
        )

class AdaptiveTimeoutManager:
    """Manages dynamic timeout adjustment based on network conditions"""
    
    def __init__(self):
        self.base_timeout = 150  # milliseconds
        self.max_timeout = 300
        self.min_timeout = 100
        self.adjustment_factor = 1.0
        self._latency_history = deque(maxlen=1000)
        self._lock = threading.RLock()
    
    def record_network_latency(self, latency_ms: float):
        """Record network latency for timeout calculation"""
        with self._lock:
            self._latency_history.append(latency_ms)
            self._recalculate_timeout()
    
    def _recalculate_timeout(self):
        """Recalculate timeout based on recent network performance"""
        if len(self._latency_history) < 10:
            return
        
        # Use 95th percentile latency
        p95_latency = statistics.quantiles(self._latency_history, n=20)[18]
        
        # Timeout should be at least 3x expected latency
        recommended_timeout = max(p95_latency * 3, self.min_timeout)
        recommended_timeout = min(recommended_timeout, self.max_timeout)
        
        # Smooth the adjustment
        self.adjustment_factor = 0.8 * self.adjustment_factor + 0.2 * (recommended_timeout / self.base_timeout)
    
    def get_election_timeout(self) -> float:
        """Get current election timeout in seconds"""
        with self._lock:
            timeout_ms = self.base_timeout * self.adjustment_factor
            timeout_ms = max(self.min_timeout, min(self.max_timeout, timeout_ms))
            return timeout_ms / 1000.0

class SplitBrainGuard:
    """Prevents split-brain scenarios through multiple validation layers"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.current_term = 0
        self.voted_for = None
        self.known_leaders = {}  # term -> leader_id
        self._lock = threading.RLock()
    
    def can_grant_vote(self, term: int, candidate_id: str, candidate_log_term: int, candidate_log_index: int,
                       own_log_term: int, own_log_index: int) -> bool:
        """Determine if vote can be granted while preventing split-brain"""
        with self._lock:
            # Update term if necessary
            if term > self.current_term:
                self.current_term = term
                self.voted_for = None
            
            # Can't vote for older terms
            if term < self.current_term:
                return False
            
            # Already voted in this term
            if self.voted_for is not None and self.voted_for != candidate_id:
                return False
            
            # Candidate log must be at least as up-to-date
            if candidate_log_term < own_log_term:
                return False
            if candidate_log_term == own_log_term and candidate_log_index < own_log_index:
                return False
            
            return True
    
    def grant_vote(self, term: int, candidate_id: str):
        """Grant vote and persist decision"""
        with self._lock:
            self.voted_for = candidate_id
            # TODO: Persist vote to stable storage
    
    def validate_leader(self, term: int, leader_id: str) -> bool:
        """Validate that a leader claim is legitimate"""
        with self._lock:
            if term < self.current_term:
                return False
            
            if term > self.current_term:
                self.current_term = term
                self.voted_for = None
            
            # Check for conflicting leader claims in same term
            if term in self.known_leaders:
                return self.known_leaders[term] == leader_id
            
            self.known_leaders[term] = leader_id
            return True
```

**Core Logic Skeleton Code:**

```python
def detect_partition(self) -> bool:
    """Detect if this node is in a network partition"""
    # TODO 1: Count nodes that have responded recently (within 2x heartbeat interval)
    # TODO 2: Calculate if reachable nodes form a majority of cluster
    # TODO 3: Check for asymmetric connectivity patterns
    # TODO 4: Return True if partition detected
    pass

def handle_leader_failure(self) -> None:
    """Coordinate response to leader failure"""
    # TODO 1: Verify leader is actually unreachable (not just slow)
    # TODO 2: Check if this node should start an election
    # TODO 3: Wait for randomized election timeout
    # TODO 4: Start election if no other candidate emerges
    # TODO 5: Handle case where multiple candidates start simultaneously
    pass

def handle_network_partition(self, unreachable_nodes: Set[str]) -> None:
    """Manage behavior during network partition"""
    # TODO 1: Determine which side of partition this node is on
    # TODO 2: Count reachable nodes vs unreachable nodes
    # TODO 3: If in minority partition, step down and stop accepting requests
    # TODO 4: If in majority partition, continue normal operations
    # TODO 5: Set up monitoring for partition healing
    pass

def recover_from_partition(self, newly_reachable_nodes: List[str]) -> None:
    """Handle partition healing and state reconciliation"""
    # TODO 1: Exchange term information with newly reachable nodes
    # TODO 2: Determine which side had legitimate leadership during partition
    # TODO 3: If this side was minority, accept state from majority side
    # TODO 4: Truncate any conflicting log entries
    # TODO 5: Restore state machine if necessary
    pass

def validate_cluster_consistency(self) -> List[str]:
    """Check for split-brain or consistency violations"""
    # TODO 1: Compare log entries across all reachable nodes
    # TODO 2: Check for conflicting committed entries
    # TODO 3: Verify term progression is monotonic
    # TODO 4: Look for evidence of multiple concurrent leaders
    # TODO 5: Return list of detected violations
    pass
```

**Milestone Checkpoint:**

After implementing error handling and edge case management, verify the system behaves correctly:

1. **Split-Brain Prevention Test**: Start a 5-node cluster, partition it 3-2, verify only one side remains active
2. **Leader Failure Recovery**: Kill the leader node, verify new leader is elected within 2 election timeouts
3. **Partition Healing**: Heal the network partition, verify minority side accepts majority state
4. **Cascade Failure**: Fail 2 nodes simultaneously, verify cluster remains consistent and available
5. **Byzantine Behavior**: Inject corrupted responses from one node, verify it gets isolated

Expected behavior: The cluster should never have multiple leaders simultaneously, should recover from minority failures within seconds, and should maintain data consistency even during complex failure scenarios.

**Debugging Tips:**

| Symptom | Likely Cause | Diagnosis Steps | Fix |
|---------|--------------|-----------------|-----|
| Frequent leader elections | Network instability or timeouts too aggressive | Check election timeout vs network latency | Increase timeouts or improve network |
| Cluster becomes unavailable | Split-brain prevention too conservative | Check quorum calculation and reachability | Fix connectivity or adjust cluster size |
| Data inconsistency | Split-brain occurred | Compare logs across nodes, check term history | Implement proper vote persistence |
| Slow failure detection | Heartbeat intervals too long | Measure actual failure detection time | Decrease heartbeat interval |
| False failure alarms | Timeouts too aggressive for network | Monitor false positive rate | Implement adaptive timeouts |


## Testing Strategy

> **Milestone(s):** All milestones (1-4) - comprehensive testing is essential to verify correctness and robustness throughout election, replication, compaction, and membership changes

### Mental Model: Scientific Experimentation

Think of testing a consensus system like conducting rigorous scientific experiments on a complex ecosystem. Just as biologists don't just observe animals in perfect laboratory conditions, we can't test distributed systems only under ideal circumstances. We need controlled experiments (unit tests), field studies (integration tests), stress testing (chaos engineering), and long-term observation (property-based testing) to understand how the system truly behaves.

The key insight is that consensus systems have **emergent properties** - behaviors that only appear when multiple nodes interact under real-world conditions. A single node might work perfectly in isolation, but the consensus protocol only proves itself when nodes disagree, networks partition, and failures cascade. Our testing strategy must therefore create increasingly realistic conditions that expose these emergent behaviors.

### Mental Model: Quality Assurance Pyramid

Envision testing as a pyramid with four distinct layers, each serving a different purpose. At the base, we have fast unit tests that verify individual components in isolation. Moving up, we have integration tests that verify component interactions. Near the top, we have chaos tests that simulate real-world failures. At the apex, we have long-running property tests that verify fundamental invariants hold over time.

Each layer catches different types of bugs. Unit tests catch logic errors in individual functions. Integration tests catch interface mismatches and protocol violations. Chaos tests catch rare race conditions and cascading failures. Property tests catch subtle correctness violations that only emerge after thousands of operations.

### Milestone Checkpoints

The milestone checkpoint approach ensures that we verify both functional correctness and non-functional properties at each stage of implementation. Each checkpoint includes specific test scenarios, expected behaviors, and performance criteria that must be met before proceeding to the next milestone.

#### Milestone 1: Election & Safety Verification

The election and safety milestone focuses on verifying that leader election works correctly under all conditions and that the fundamental safety property of "at most one leader per term" is never violated.

**Functional Verification Criteria:**

| Test Scenario | Expected Behavior | Verification Method |
|---------------|------------------|-------------------|
| Single node startup | Node becomes leader immediately | Check `node_state == LEADER` and term increments |
| Three node cluster startup | One leader elected within election timeout window | Verify exactly one leader across all nodes |
| Leader failure with two followers | New leader elected within 2x election timeout | Monitor state transitions and term progression |
| Split vote scenario | Election retry with randomized timeouts | Verify eventual leader election despite initial ties |
| Network partition during election | Majority partition elects leader, minority stays follower | Test with 3-node and 5-node clusters |
| Concurrent elections | Only one leader emerges per term | Stress test with simultaneous candidate transitions |

**Safety Property Verification:**

The most critical safety property is that there can never be two leaders in the same term. This property must hold even under the most adverse conditions.

| Safety Property | Test Method | Invariant Check |
|----------------|-------------|----------------|
| Single leader per term | Monitor all nodes continuously during elections | `count(nodes where node_state == LEADER and term == T) <= 1` |
| Vote integrity | Verify each node votes at most once per term | `voted_for` field changes at most once per term increment |
| Term monotonicity | Terms only increase, never decrease | `new_term >= old_term` for all state transitions |
| Election timeout randomization | Verify timeouts are properly randomized | Measure actual timeout values fall within expected range |

**Performance Criteria:**

| Metric | Target | Measurement Method |
|--------|--------|------------------|
| Election latency (3 nodes) | < 500ms under normal conditions | Time from leader failure to new leader heartbeat |
| Election latency (5 nodes) | < 750ms under normal conditions | Account for additional network round trips |
| Split vote recovery time | < 1.5 seconds | Time from split vote to successful election |
| Heartbeat overhead | < 5% of network bandwidth | Measure heartbeat message frequency and size |

**Checkpoint Verification Process:**

1. **Isolation Testing**: Start with single-node tests to verify basic state machine transitions work correctly
2. **Pair Testing**: Test two-node scenarios to verify basic RPC communication and vote exchange
3. **Cluster Testing**: Test 3-node and 5-node clusters under normal conditions
4. **Failure Injection**: Systematically kill leaders and verify election recovery
5. **Stress Testing**: Run elections under high load to verify performance criteria
6. **Extended Runtime**: Run for 24+ hours to catch rare timing bugs

#### Milestone 2: Log Replication Verification

Log replication verification focuses on ensuring that the log matching property is maintained and that committed entries are never lost, even under network partitions and node failures.

**Functional Verification Criteria:**

| Test Scenario | Expected Behavior | Verification Method |
|---------------|------------------|-------------------|
| Normal replication | Client requests replicated to majority before commit | Verify log consistency across nodes |
| Follower crash during replication | Leader retries until follower recovers and catches up | Test log repair and conflict resolution |
| Network partition with minority leader | Minority leader cannot commit new entries | Verify commits require majority acknowledgment |
| Network partition with majority leader | Majority leader continues accepting requests | Verify continued operation with degraded cluster |
| Conflicting log entries from past terms | Follower logs are corrected to match leader | Test log conflict detection and resolution |
| Out-of-order message delivery | Log remains consistent despite network reordering | Use network simulator with message delays |

**Consistency Property Verification:**

The log matching property states that if two logs contain an entry with the same index and term, then the logs are identical in all preceding entries.

| Consistency Property | Test Method | Invariant Check |
|---------------------|-------------|----------------|
| Log matching property | Compare logs across all nodes after operations | For any two nodes A,B: `log[i].term == log[j].term` ⟹ `log[0:i] == log[0:j]` |
| Commit point consistency | Verify committed entries never change | Once `commit_index >= i`, `log[i]` never changes |
| State machine consistency | Apply same log to multiple state machines | All state machines produce identical results |
| Durability guarantee | Committed entries survive node crashes | Restart nodes and verify log contents persist |

**Performance Criteria:**

| Metric | Target | Measurement Method |
|--------|--------|------------------|
| Replication latency (3 nodes) | < 10ms for single entry | Time from client request to commit notification |
| Replication throughput | > 1000 ops/sec with batching | Measure sustained request processing rate |
| Log repair time | < 5 seconds for 1000 entry gap | Time to catch up lagging follower |
| Memory usage growth | Linear with uncommitted entries | Monitor process memory during high load |

#### Milestone 3: Log Compaction Verification

Log compaction verification ensures that snapshots correctly preserve state machine state and that the `InstallSnapshot` protocol safely transfers snapshots to lagging followers.

**Functional Verification Criteria:**

| Test Scenario | Expected Behavior | Verification Method |
|---------------|------------------|-------------------|
| Automatic snapshot creation | Snapshot created when log exceeds threshold | Verify snapshot files and log truncation |
| Snapshot state consistency | Snapshot captures exact state machine state | Compare snapshot restoration with log replay |
| InstallSnapshot to lagging follower | Follower receives and applies snapshot correctly | Verify follower state matches leader after transfer |
| Concurrent snapshots and replication | Normal replication continues during snapshotting | Verify no interference between processes |
| Snapshot failure and recovery | Failed snapshots don't corrupt existing state | Test disk full, network errors during snapshot |
| Log truncation safety | Truncation only occurs after successful snapshot | Verify log entries available until snapshot complete |

**Compaction Property Verification:**

Log compaction must preserve all committed state while reducing log size without affecting system availability.

| Compaction Property | Test Method | Invariant Check |
|-------------------|-------------|----------------|
| State preservation | Compare state before/after compaction | State machine state identical pre/post snapshot |
| Log size reduction | Verify old entries removed after snapshot | Log size decreases after successful compaction |
| Availability maintenance | System accepts requests during compaction | Monitor request success rate during snapshots |
| Incremental transfer | Large snapshots transfer without blocking | Verify chunked transfer doesn't stall operations |

**Performance Criteria:**

| Metric | Target | Measurement Method |
|--------|--------|------------------|
| Snapshot creation time | < 5 seconds for 100MB state | Time from snapshot start to completion |
| Snapshot transfer rate | > 10MB/sec between nodes | Measure InstallSnapshot throughput |
| Availability impact | < 1% increase in request latency | Monitor P99 latency during compaction |
| Storage efficiency | > 90% log size reduction | Compare log sizes before/after compaction |

#### Milestone 4: Membership Changes Verification

Membership change verification focuses on ensuring that joint consensus safely transitions cluster membership without creating split-brain scenarios or losing availability.

**Functional Verification Criteria:**

| Test Scenario | Expected Behavior | Verification Method |
|---------------|------------------|-------------------|
| Add single node to cluster | Node added without affecting existing operations | Verify cluster size increases and quorum adjusts |
| Remove single node from cluster | Node removed gracefully with quorum adjustment | Verify cluster continues with smaller quorum |
| Add multiple nodes simultaneously | Joint consensus manages complex membership change | Test dual majority requirements |
| Membership change during partition | Change proceeds only with dual majority available | Verify safety under network partitions |
| New node catch-up process | New node reaches current state before voting | Monitor log replication to new nodes |
| Membership change rollback | Failed changes revert to original configuration | Test rollback under various failure scenarios |

**Joint Consensus Property Verification:**

Joint consensus must ensure that during membership transitions, decisions require majorities from both old and new configurations.

| Joint Consensus Property | Test Method | Invariant Check |
|-------------------------|-------------|----------------|
| Dual majority requirement | Verify decisions need both old and new majorities | For any decision: `old_votes >= old_majority AND new_votes >= new_majority` |
| Configuration consistency | All nodes agree on current configuration state | Configuration version identical across all nodes |
| Transition atomicity | Membership changes complete fully or not at all | No partial configuration states persist |
| Availability preservation | Cluster remains available during transitions | Monitor request success rate during changes |

**Performance Criteria:**

| Metric | Target | Measurement Method |
|--------|--------|------------------|
| Membership change latency | < 30 seconds for single node addition | Time from change initiation to completion |
| New node catch-up time | < 60 seconds for 10GB log | Time for new node to reach voting readiness |
| Availability impact | < 5% increase in request latency | Monitor performance during membership changes |
| Configuration convergence | < 10 seconds for all nodes to agree | Time for configuration consensus across cluster |

### Property-Based Testing

Property-based testing moves beyond specific test scenarios to verify that fundamental invariants hold across all possible system states and operation sequences. Instead of testing specific cases, we define properties that must always be true and let the testing framework generate thousands of random scenarios to try to violate these properties.

### Mental Model: Mathematical Proofs

Think of property-based testing like mathematical theorem proving. Instead of showing that a theorem holds for specific examples (3+4=7, 5+2=7), we prove that it holds for all possible inputs (commutativity: a+b=b+a for all a,b). Similarly, instead of testing specific election scenarios, we prove that safety properties hold for all possible election sequences.

The power comes from **generative testing** - the framework automatically creates thousands of different scenarios, including edge cases human testers might never think of. When a property violation is found, the framework automatically shrinks the failing case to the minimal example that demonstrates the bug.

#### Core Raft Invariants

These properties must hold regardless of the specific sequence of operations, failures, or timing that occurs in the system.

**Safety Invariants:**

| Property | Formal Definition | Testing Strategy |
|----------|------------------|------------------|
| Election Safety | At most one leader per term across all nodes | Generate random election scenarios with varying timing and failures |
| Leader Append-Only | Leaders never overwrite or delete existing log entries | Generate random leader changes and verify log monotonicity |
| Log Matching | If logs contain entry with same index/term, all preceding entries identical | Generate random replication scenarios and verify log consistency |
| Leader Completeness | If entry committed in term T, it appears in leader logs for all terms > T | Generate scenarios with leadership changes after commits |
| State Machine Safety | If node applies entry at index i, no other node applies different entry at i | Generate concurrent operation scenarios across multiple nodes |

**Liveness Invariants:**

| Property | Formal Definition | Testing Strategy |
|----------|------------------|------------------|
| Election Progress | If majority of nodes can communicate, leader eventually elected | Generate partition scenarios and verify eventual convergence |
| Replication Progress | Client requests eventually commit if majority nodes available | Generate load scenarios with intermittent failures |
| Catch-up Progress | Lagging followers eventually catch up to leader log | Generate scenarios with temporary node isolation |
| Membership Progress | Membership changes eventually complete or rollback | Generate complex membership change scenarios |

#### Generative Test Strategy

Property-based testing uses **generators** that create random but realistic sequences of operations and failures. The key is making the random generation intelligent enough to create meaningful scenarios while covering the vast space of possible system states.

**Operation Generators:**

| Generator Type | Purpose | Example Outputs |
|----------------|---------|----------------|
| Election Sequence Generator | Creates realistic election scenarios | "Node A starts election, B votes yes, C crashes before voting, A becomes leader" |
| Client Request Generator | Generates realistic client workloads | "Write X=1, Read Y, Write Z=X+Y, concurrent reads during writes" |
| Failure Pattern Generator | Creates realistic failure scenarios | "Node crash for 5 seconds, network partition lasting 30 seconds, slow network" |
| Timing Variation Generator | Tests different timing interleavings | "Request A starts, Request B starts, A commits, B commits vs B commits, A commits" |

**Shrinking Strategy:**

When a property violation is detected, the testing framework automatically **shrinks** the failing test case to find the minimal example that demonstrates the bug. This is crucial for debugging complex distributed systems.

| Shrinking Dimension | Strategy | Example |
|---------------------|----------|---------|
| Operation Count | Reduce number of operations while preserving failure | "100 operations cause bug" → "3 operations cause same bug" |
| Cluster Size | Test with smaller clusters if possible | "7-node failure" → "3-node failure with same root cause" |
| Timing Windows | Minimize timing dependencies | "Race over 500ms window" → "Race over 1ms window" |
| Failure Complexity | Simplify failure patterns | "Network partition + 2 node crashes" → "Single node crash" |

#### Implementation Strategy for Property Testing

The property-based testing implementation requires careful design to make the random generation both effective and efficient.

**Test Environment Abstraction:**

| Component | Responsibility | Interface |
|-----------|----------------|-----------|
| Virtual Time | Deterministic timing control | `advance_time(delta)`, `current_time()`, `schedule_event(time, callback)` |
| Virtual Network | Controllable message delivery | `set_partition(nodes)`, `set_delay(src, dst, ms)`, `drop_rate(percent)` |
| Cluster Simulator | Multi-node coordination | `create_cluster(size)`, `kill_node(id)`, `restart_node(id)` |
| State Monitor | Property verification | `check_invariants()`, `record_state_transition()`, `verify_consistency()` |

**Property Test Structure:**

Each property test follows a standard structure that separates the scenario generation from the property verification, making tests both readable and maintainable.

1. **Setup Phase**: Initialize cluster with random configuration (size, initial leader, etc.)
2. **Generation Phase**: Generate random sequence of operations and failures
3. **Execution Phase**: Apply operations to simulated cluster with controlled timing
4. **Verification Phase**: Check that all invariants held throughout execution
5. **Cleanup Phase**: Reset environment for next test iteration

#### Property Test Examples

**Election Safety Property Test:**

The election safety test generates random election scenarios and verifies that at most one leader exists per term.

Test Strategy:
1. Generate random cluster sizes (3, 5, 7 nodes)
2. Generate random failure patterns (node crashes, network partitions)
3. Generate random timing variations (election timeouts, message delays)
4. For each scenario, verify that `len([n for n in nodes if n.state == LEADER and n.term == T]) <= 1`
5. Use state machine model to predict expected behavior and compare with actual

**Log Consistency Property Test:**

The log consistency test verifies that the log matching property holds under all possible replication scenarios.

Test Strategy:
1. Generate random client request sequences (reads, writes, mixed operations)
2. Generate random failure patterns during replication
3. Generate random message delivery orders and delays
4. After each operation, verify log matching property across all nodes
5. Use reference implementation to compute expected final state

### Chaos and Partition Testing

Chaos testing goes beyond property-based testing to simulate the harsh realities of production environments. While property tests verify correctness under controlled randomness, chaos tests verify robustness under realistic failure patterns that mirror real-world distributed system challenges.

### Mental Model: Stress Testing Infrastructure

Think of chaos testing like stress-testing a bridge. Engineers don't just verify that a bridge holds the expected load under perfect conditions - they test it under extreme weather, with varying loads, during earthquakes, and with simulated component failures. Similarly, chaos testing subjects our consensus system to realistic combinations of failures that production systems actually encounter.

The key insight is that **correlated failures** are common in production. When one node fails, it's often because of a systemic issue that affects multiple nodes. When network partitions occur, they often follow patterns based on physical network topology. Chaos testing simulates these realistic failure correlation patterns.

#### Failure Pattern Categories

Real-world distributed systems face predictable categories of failures that occur with different frequencies and characteristics. Our chaos testing must simulate each category with realistic parameters.

**Infrastructure Failures:**

| Failure Type | Characteristics | Simulation Strategy |
|--------------|----------------|-------------------|
| Node Crashes | Sudden process termination, no graceful shutdown | Kill processes with SIGKILL, verify state persistence and recovery |
| Slow Nodes | High CPU/memory usage, delayed message processing | Inject CPU throttling and memory pressure, measure impact on consensus |
| Disk Failures | Write failures, read errors, capacity exhaustion | Simulate disk full conditions and I/O errors during log writes |
| Clock Skew | Unsynchronized system clocks across nodes | Inject time drift and verify timeout behavior remains correct |

**Network Failures:**

| Failure Type | Characteristics | Simulation Strategy |
|--------------|----------------|-------------------|
| Network Partitions | Nodes split into disconnected groups | Use iptables to create realistic partition topologies |
| Packet Loss | Random message drops, varying loss rates | Configure random packet dropping with different loss percentages |
| Network Delays | High latency, jitter, timeout violations | Add variable network delays to test timeout handling |
| Asymmetric Partitions | A can reach B but B cannot reach A | Create unidirectional network failures |

**Cascading Failures:**

| Failure Type | Characteristics | Simulation Strategy |
|--------------|----------------|-------------------|
| Thundering Herd | Multiple nodes start elections simultaneously | Trigger simultaneous timeouts across multiple nodes |
| Resource Exhaustion | Memory/disk pressure spreads across cluster | Inject resource pressure that affects multiple nodes |
| Dependency Failures | External service failures affect multiple nodes | Simulate shared dependency failures (DNS, storage, etc.) |
| Maintenance Cascades | Planned maintenance triggers unexpected failures | Simulate rolling restarts with injected failures |

#### Partition Topology Testing

Network partitions in real systems follow patterns based on physical network topology. Random partitions don't adequately test the scenarios that production systems actually encounter.

**Realistic Partition Patterns:**

| Partition Type | Description | Test Scenario |
|----------------|-------------|---------------|
| Clean Split | Cluster divides into two connected groups | Test 3-node → (2,1) and 5-node → (3,2) splits |
| Isolated Node | Single node loses connectivity to rest | Verify isolated node doesn't disrupt cluster |
| Cascading Split | Partition grows over time | Start with one isolated node, gradually expand partition |
| Partial Mesh | Some connections work, others fail | Create complex connectivity matrix with mixed reachability |
| Bridge Node | One node connects two otherwise partitioned groups | Test scenarios where removing one node causes split |
| Flapping Partition | Partition heals and re-occurs repeatedly | Test rapid partition/healing cycles |

**Partition Testing Strategy:**

| Test Phase | Duration | Purpose | Verification |
|------------|----------|---------|-------------|
| Pre-partition baseline | 30 seconds | Establish normal operation | Verify stable leader, successful client requests |
| Partition creation | Immediate | Simulate network failure | Verify appropriate nodes become unavailable |
| Partition duration | 60-300 seconds | Test behavior during partition | Verify majority continues, minority stops |
| Partition healing | Immediate | Simulate network recovery | Verify cluster reunification and log repair |
| Post-partition verification | 60 seconds | Verify full recovery | Verify all nodes converged to consistent state |

#### Chaos Testing Framework Architecture

The chaos testing framework requires sophisticated infrastructure to create realistic failure scenarios while maintaining observability and reproducibility.

**Chaos Controller Components:**

| Component | Responsibility | Key Methods |
|-----------|----------------|-------------|
| Failure Injector | Creates and manages failure scenarios | `inject_node_failure(node_id, duration)`, `create_partition(group_a, group_b)` |
| Network Controller | Manages network topology and failures | `set_latency(src, dst, ms)`, `set_packet_loss(src, dst, percent)` |
| Resource Controller | Manages CPU, memory, disk resource pressure | `throttle_cpu(node_id, percent)`, `limit_memory(node_id, bytes)` |
| Scenario Generator | Creates realistic failure sequences | `generate_partition_scenario()`, `generate_cascading_failure()` |
| Health Monitor | Tracks system health during chaos | `monitor_consensus_health()`, `verify_safety_properties()` |
| Recovery Verifier | Validates recovery after failures | `verify_cluster_convergence()`, `check_data_integrity()` |

#### Specific Chaos Test Scenarios

**Scenario 1: Leader Isolation Test**

This scenario tests the critical case where the current leader becomes isolated from the majority of the cluster.

Test Sequence:
1. Start 5-node cluster, wait for stable leader election
2. Generate steady client request load (10 requests/second)
3. Identify current leader node
4. Create network partition isolating leader from other 4 nodes
5. Verify majority (4 nodes) elects new leader within election timeout
6. Verify isolated leader steps down to follower state
7. Continue client requests to new leader for 2 minutes
8. Heal partition and verify cluster reconvergence
9. Verify isolated node's log is repaired to match cluster

**Expected Behavior:**
- New leader elected within 300-500ms
- Client requests continue without data loss
- Isolated leader log repaired upon reconnection
- No split-brain scenarios occur

**Scenario 2: Cascading Node Failure Test**

This scenario simulates a realistic failure pattern where multiple nodes fail in sequence, testing the cluster's resilience to correlated failures.

Test Sequence:
1. Start 7-node cluster for better failure tolerance
2. Generate mixed read/write workload
3. Kill one random non-leader node (simulate hardware failure)
4. Wait 30 seconds, kill second random node (simulate cascading failure)
5. Wait 30 seconds, kill current leader (simulate correlated infrastructure failure)
6. Verify remaining 4 nodes elect new leader and continue operations
7. Restart failed nodes one by one with 60-second intervals
8. Verify each node successfully rejoins and catches up

**Expected Behavior:**
- Cluster maintains availability with 4/7 nodes
- Restarted nodes successfully catch up via log replication
- No data corruption or consistency violations
- Performance degrades gracefully with fewer nodes

**Scenario 3: Network Flapping Test**

This scenario tests behavior under unstable network conditions with repeated partitions and healing.

Test Sequence:
1. Start 5-node cluster
2. Generate continuous client workload
3. Create rapid partition/healing cycles:
   - Partition for 10 seconds (3 nodes vs 2 nodes)
   - Heal for 20 seconds
   - Partition for 15 seconds (different split: 4 nodes vs 1 node)
   - Heal for 30 seconds
   - Repeat pattern for 10 minutes
4. Verify cluster remains consistent despite network instability
5. Verify no split-brain conditions occur during transitions

**Expected Behavior:**
- Cluster maintains consistency despite frequent partitions
- Leadership changes appropriately with partition patterns
- No permanent inconsistencies develop
- Client requests eventually succeed despite temporary failures

### Performance and Scale Testing

Performance testing for consensus systems requires understanding the unique characteristics of distributed coordination protocols. Unlike traditional application performance testing, consensus performance is fundamentally limited by network latency and the requirement for majority agreement.

### Mental Model: Orchestra Performance

Think of consensus performance like measuring an orchestra's performance. You can't just measure how fast individual musicians can play - the performance is limited by the need for coordination and synchronization. The conductor (leader) must wait for responses from the musicians (followers) before proceeding to the next movement (operation). Network latency is like the time it takes for sound to travel across a large concert hall.

The key insight is that consensus systems have **coordination overhead** that creates different performance characteristics than single-node systems. The goal isn't to eliminate this overhead (which is impossible) but to minimize it while maintaining correctness guarantees.

#### Performance Characteristic Analysis

Understanding the fundamental performance characteristics helps set realistic expectations and identify optimization opportunities.

**Latency Characteristics:**

| Operation Type | Minimum Latency | Typical Latency | Latency Factors |
|----------------|----------------|----------------|----------------|
| Read (leader local) | ~1ms | 5-10ms | Leader processing overhead |
| Read (linearizable) | 1x network RTT | 10-20ms | Heartbeat confirmation required |
| Write (single entry) | 1x network RTT | 15-30ms | Majority replication required |
| Write (batched) | 1x network RTT | 20-50ms | Amortized over multiple entries |
| Election (no contention) | 1x network RTT | 200-400ms | Vote collection overhead |
| Election (with contention) | 2-3x network RTT | 500-1500ms | Multiple rounds required |

**Throughput Characteristics:**

| Workload Type | Expected Throughput | Limiting Factor | Optimization Strategy |
|---------------|-------------------|----------------|---------------------|
| Write-heavy | 1000-10000 ops/sec | Network bandwidth, leader processing | Batching, pipeline replication |
| Read-heavy | 10000-100000 ops/sec | State machine performance | Read-only replicas, leader stickiness |
| Mixed workload | 5000-50000 ops/sec | Write coordination overhead | Read/write separation |
| Small cluster (3 nodes) | Higher per-node throughput | Less coordination overhead | Fewer network round trips |
| Large cluster (7+ nodes) | Lower per-node throughput | More coordination overhead | Hierarchical consensus |

#### Scalability Testing Methodology

Consensus systems have unique scalability characteristics that require specialized testing approaches. The relationship between cluster size and performance is non-linear and depends on multiple factors.

**Cluster Size Impact Analysis:**

| Cluster Size | Majority Size | Expected Latency Impact | Expected Throughput Impact | Failure Tolerance |
|--------------|---------------|------------------------|---------------------------|------------------|
| 3 nodes | 2 | Baseline | Baseline | 1 node failure |
| 5 nodes | 3 | +20-30% | -10-20% | 2 node failures |
| 7 nodes | 4 | +40-50% | -20-30% | 3 node failures |
| 9 nodes | 5 | +60-80% | -30-40% | 4 node failures |

The scalability testing strategy must account for the trade-off between fault tolerance and performance. Larger clusters provide better fault tolerance but with increased coordination overhead.

**Load Testing Progressive Methodology:**

| Test Phase | Objective | Cluster Configuration | Load Pattern |
|------------|-----------|----------------------|-------------|
| Baseline Single Node | Establish maximum theoretical performance | 1 node (no consensus) | Ramp from 0 to saturation |
| Minimal Consensus | Measure consensus overhead | 3 nodes | Same load pattern as baseline |
| Optimal Size | Find best performance/reliability balance | 5 nodes | Sustained load at target throughput |
| Stress Testing | Identify breaking points | 5 nodes | Exponential load increase until failure |
| Scale Testing | Measure large cluster performance | 7, 9 nodes | Compare throughput and latency degradation |
| Mixed Workload | Test realistic usage patterns | 5 nodes | Mix of reads, writes, elections |

#### Performance Test Scenarios

**Scenario 1: Sustained Throughput Test**

This test measures the maximum sustained throughput the consensus system can handle without degradation.

Test Configuration:
- 5-node cluster with stable leader
- Progressive load increase: 100, 500, 1000, 2000, 5000, 10000 ops/sec
- 60-second measurement period at each load level
- Mix: 70% writes, 30% reads
- 1KB average payload size

Measurement Criteria:

| Metric | Acceptable Threshold | Warning Threshold | Critical Threshold |
|--------|-------------------|------------------|-------------------|
| Average Latency | < 50ms | 50-100ms | > 100ms |
| P99 Latency | < 200ms | 200-500ms | > 500ms |
| Error Rate | < 0.1% | 0.1-1% | > 1% |
| CPU Utilization | < 80% | 80-90% | > 90% |
| Memory Usage | < 80% of available | 80-90% | > 90% |
| Network Utilization | < 70% of bandwidth | 70-85% | > 85% |

**Scenario 2: Latency Sensitivity Test**

This test measures how latency degrades under different load conditions and identifies latency outliers.

Test Configuration:
- 5-node cluster across different network latencies (1ms, 10ms, 50ms, 100ms)
- Fixed moderate load (1000 ops/sec)
- Detailed latency distribution measurement
- 30-minute test duration for statistical significance

Latency Measurement Strategy:

| Percentile | Target Multiplier | Measurement Method |
|------------|------------------|-------------------|
| P50 | 2x network RTT | Median of all operations |
| P90 | 3x network RTT | 90th percentile measurement |
| P99 | 5x network RTT | 99th percentile measurement |
| P99.9 | 10x network RTT | Tail latency analysis |

**Scenario 3: Election Performance Test**

This test measures the performance impact of leader elections under various conditions.

Test Configuration:
- 5-node cluster with triggered leader failures
- Background load during elections (500 ops/sec)
- Systematic testing of different election scenarios
- Measurement of recovery time and impact

Election Scenarios:

| Scenario | Trigger Method | Expected Recovery Time | Performance Impact |
|----------|---------------|----------------------|-------------------|
| Clean leader shutdown | Graceful process stop | < 300ms | Minimal request queuing |
| Leader crash | Process kill -9 | < 500ms | Brief request failures |
| Leader network isolation | Network partition | < 500ms | Requests redirected to new leader |
| Multiple candidate elections | Simultaneous timeouts | < 1000ms | Extended unavailability window |
| Leader election during high load | Kill leader under stress | < 800ms | Request queuing and retries |

#### Performance Optimization Testing

Performance optimization requires systematic testing of different configuration parameters and architectural choices to understand their impact on system performance.

**Configuration Parameter Impact Testing:**

| Parameter | Test Range | Primary Impact | Secondary Impact |
|-----------|------------|---------------|----------------|
| Heartbeat Interval | 10ms - 200ms | Election detection time | Network overhead |
| Election Timeout Range | 100ms - 1000ms | Election stability | Recovery time |
| Batch Size | 1 - 1000 entries | Throughput | Latency |
| Log Sync Frequency | Every write - 100ms intervals | Durability vs performance | Crash recovery time |
| Network Buffer Sizes | 4KB - 1MB | Network efficiency | Memory usage |

**Optimization Strategy Testing:**

| Optimization | Measurement Approach | Expected Benefit | Risk Factors |
|-------------|---------------------|-----------------|--------------|
| Request Batching | Compare single vs batched requests | 2-5x throughput improvement | Increased latency |
| Pipeline Replication | Measure concurrent vs sequential replication | 20-40% latency reduction | Complexity increase |
| Read Optimization | Compare leader reads vs consensus reads | 10x read performance | Reduced consistency |
| Compression | Test different compression algorithms | 30-50% bandwidth reduction | CPU overhead |
| Connection Pooling | Measure connection reuse benefits | 10-20% latency reduction | Connection management complexity |

### Implementation Guidance

This implementation guidance provides the infrastructure and testing tools needed to verify consensus system correctness and performance throughout development.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|----------------|
| Testing Framework | pytest with standard assertions | Hypothesis for property-based testing |
| Load Generation | Custom Python scripts with threading | Locust for distributed load testing |
| Network Simulation | Manual process control (kill -9) | Mininet for realistic network topologies |
| Time Control | time.sleep() and manual coordination | Simulated time with deterministic scheduling |
| Metrics Collection | Print statements and manual timing | Prometheus + Grafana for comprehensive monitoring |
| Failure Injection | Basic process killing | Chaos Monkey or Gremlin for sophisticated failures |

#### Testing Infrastructure Code

**Complete Test Environment Setup:**

```python
"""
Comprehensive testing infrastructure for Titan consensus engine.
Provides controlled environments for property-based testing, chaos testing,
and performance measurement.
"""

import time
import threading
import random
import socket
import subprocess
import json
from typing import Dict, List, Set, Optional, Callable, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import pytest
from unittest.mock import Mock, patch

@dataclass
class TestMetrics:
    """Comprehensive metrics collection for test analysis."""
    start_time: float
    end_time: float
    operations_attempted: int
    operations_succeeded: int
    operations_failed: int
    election_count: int
    leader_changes: int
    average_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    network_messages_sent: int
    network_bytes_sent: int
    
    def success_rate(self) -> float:
        """Calculate operation success rate."""
        if self.operations_attempted == 0:
            return 0.0
        return self.operations_succeeded / self.operations_attempted
    
    def throughput_ops_per_sec(self) -> float:
        """Calculate sustained throughput."""
        duration = self.end_time - self.start_time
        if duration <= 0:
            return 0.0
        return self.operations_succeeded / duration

class VirtualNetwork:
    """Simulates network with controllable failures and delays."""
    
    def __init__(self):
        self.partitions: Set[frozenset] = set()
        self.delays: Dict[tuple, float] = {}
        self.drop_rates: Dict[tuple, float] = {}
        self.message_count = 0
        self.bytes_sent = 0
        
    def create_partition(self, group_a: Set[str], group_b: Set[str]) -> None:
        """Create network partition between two groups."""
        self.partitions.add(frozenset(group_a))
        self.partitions.add(frozenset(group_b))
        
    def heal_partition(self) -> None:
        """Remove all network partitions."""
        self.partitions.clear()
        
    def set_delay(self, src: str, dst: str, delay_ms: float) -> None:
        """Set network delay between two nodes."""
        self.delays[(src, dst)] = delay_ms / 1000.0
        
    def set_drop_rate(self, src: str, dst: str, drop_rate: float) -> None:
        """Set packet drop rate between two nodes."""
        self.drop_rates[(src, dst)] = drop_rate
        
    def can_deliver(self, src: str, dst: str) -> bool:
        """Check if message can be delivered given current partitions."""
        for partition in self.partitions:
            if src in partition and dst not in partition:
                return False
            if dst in partition and src not in partition:
                return False
        return True
        
    def should_drop(self, src: str, dst: str) -> bool:
        """Determine if message should be dropped."""
        drop_rate = self.drop_rates.get((src, dst), 0.0)
        return random.random() < drop_rate
        
    def get_delay(self, src: str, dst: str) -> float:
        """Get network delay between nodes."""
        return self.delays.get((src, dst), 0.0)
        
    def send_message(self, src: str, dst: str, message: bytes) -> bool:
        """Simulate sending message with network conditions."""
        self.message_count += 1
        self.bytes_sent += len(message)
        
        if not self.can_deliver(src, dst):
            return False
            
        if self.should_drop(src, dst):
            return False
            
        delay = self.get_delay(src, dst)
        if delay > 0:
            threading.Timer(delay, lambda: None).start()
            
        return True

class ClusterSimulator:
    """Manages multi-node consensus cluster for testing."""
    
    def __init__(self, cluster_size: int):
        self.cluster_size = cluster_size
        self.nodes: Dict[str, Mock] = {}
        self.virtual_network = VirtualNetwork()
        self.metrics = TestMetrics(
            start_time=0, end_time=0, operations_attempted=0,
            operations_succeeded=0, operations_failed=0,
            election_count=0, leader_changes=0,
            average_latency_ms=0, p99_latency_ms=0, max_latency_ms=0,
            network_messages_sent=0, network_bytes_sent=0
        )
        self.current_leader: Optional[str] = None
        self.operation_latencies: List[float] = []
        
    def create_cluster(self) -> None:
        """Initialize cluster with specified number of nodes."""
        for i in range(self.cluster_size):
            node_id = f"node_{i}"
            # TODO: Replace with actual TitanNode instance
            self.nodes[node_id] = Mock()
            self.nodes[node_id].node_id = node_id
            self.nodes[node_id].current_term = 0
            self.nodes[node_id].node_state = "FOLLOWER"
            
    def start_cluster(self) -> None:
        """Start all nodes and wait for leader election."""
        self.metrics.start_time = time.time()
        # TODO: Call actual start() method on each node
        # TODO: Wait for leader election to complete
        # TODO: Set self.current_leader
        pass
        
    def kill_node(self, node_id: str) -> None:
        """Simulate node crash/failure."""
        if node_id in self.nodes:
            # TODO: Call actual shutdown/cleanup on node
            self.nodes[node_id].node_state = "CRASHED"
            
    def restart_node(self, node_id: str) -> None:
        """Restart a previously killed node."""
        if node_id in self.nodes:
            # TODO: Restart actual node process
            self.nodes[node_id].node_state = "FOLLOWER"
            
    def submit_request(self, data: bytes) -> bool:
        """Submit client request to current leader."""
        self.metrics.operations_attempted += 1
        start_time = time.time()
        
        # TODO: Send request to actual leader node
        # TODO: Wait for response or timeout
        # TODO: Record success/failure and latency
        
        latency = (time.time() - start_time) * 1000  # Convert to ms
        self.operation_latencies.append(latency)
        
        # Mock success for now
        self.metrics.operations_succeeded += 1
        return True
        
    def verify_consistency(self) -> List[str]:
        """Verify all nodes have consistent state."""
        violations = []
        
        # TODO: Compare logs across all nodes
        # TODO: Verify commit indices are consistent
        # TODO: Verify state machine states are identical
        # TODO: Return list of any violations found
        
        return violations
        
    def finalize_metrics(self) -> TestMetrics:
        """Calculate final test metrics."""
        self.metrics.end_time = time.time()
        
        if self.operation_latencies:
            self.metrics.average_latency_ms = sum(self.operation_latencies) / len(self.operation_latencies)
            self.metrics.p99_latency_ms = sorted(self.operation_latencies)[int(len(self.operation_latencies) * 0.99)]
            self.metrics.max_latency_ms = max(self.operation_latencies)
            
        self.metrics.network_messages_sent = self.virtual_network.message_count
        self.metrics.network_bytes_sent = self.virtual_network.bytes_sent
        
        return self.metrics

class PropertyTest:
    """Property-based testing framework for consensus invariants."""
    
    def __init__(self, cluster_size: int = 5):
        self.cluster_size = cluster_size
        self.simulator = ClusterSimulator(cluster_size)
        
    def test_election_safety(self, num_iterations: int = 100) -> None:
        """Test that at most one leader exists per term."""
        for i in range(num_iterations):
            self.simulator.create_cluster()
            self.simulator.start_cluster()
            
            # TODO: Generate random election scenarios
            # TODO: Kill random leaders, create partitions
            # TODO: Verify single leader per term invariant
            # TODO: Record any violations
            
            violations = self.verify_election_safety()
            assert len(violations) == 0, f"Election safety violations: {violations}"
            
    def verify_election_safety(self) -> List[str]:
        """Verify election safety property."""
        violations = []
        term_leaders: Dict[int, List[str]] = {}
        
        for node_id, node in self.simulator.nodes.items():
            if node.node_state == "LEADER":
                term = node.current_term
                if term not in term_leaders:
                    term_leaders[term] = []
                term_leaders[term].append(node_id)
                
        for term, leaders in term_leaders.items():
            if len(leaders) > 1:
                violations.append(f"Term {term} has multiple leaders: {leaders}")
                
        return violations
        
    def test_log_matching_property(self, num_iterations: int = 50) -> None:
        """Test that log matching property holds under all conditions."""
        for i in range(num_iterations):
            self.simulator.create_cluster()
            self.simulator.start_cluster()
            
            # TODO: Generate random request sequences
            # TODO: Inject random failures during replication
            # TODO: Verify log matching property
            
            violations = self.verify_log_matching()
            assert len(violations) == 0, f"Log matching violations: {violations}"
            
    def verify_log_matching(self) -> List[str]:
        """Verify log matching property across all nodes."""
        violations = []
        
        # TODO: Compare logs across all nodes
        # TODO: For each pair of nodes, verify log matching property
        # TODO: If log[i].term == log[j].term at same index, verify all preceding entries match
        
        return violations

class ChaosTestSuite:
    """Comprehensive chaos testing scenarios."""
    
    def __init__(self, cluster_size: int = 5):
        self.cluster_size = cluster_size
        self.simulator = ClusterSimulator(cluster_size)
        
    def test_leader_isolation(self) -> TestMetrics:
        """Test leader isolation and recovery."""
        self.simulator.create_cluster()
        self.simulator.start_cluster()
        
        # Generate steady load
        load_thread = threading.Thread(target=self._generate_load, args=(10, 120))
        load_thread.start()
        
        # Wait for stable operation
        time.sleep(30)
        
        # Isolate current leader
        if self.simulator.current_leader:
            other_nodes = [nid for nid in self.simulator.nodes.keys() 
                          if nid != self.simulator.current_leader]
            self.simulator.virtual_network.create_partition(
                {self.simulator.current_leader}, set(other_nodes)
            )
            
        # Wait for new leader election and continued operation
        time.sleep(60)
        
        # Heal partition
        self.simulator.virtual_network.heal_partition()
        
        # Wait for cluster convergence
        time.sleep(30)
        
        load_thread.join()
        
        # Verify consistency
        violations = self.simulator.verify_consistency()
        assert len(violations) == 0, f"Consistency violations: {violations}"
        
        return self.simulator.finalize_metrics()
        
    def _generate_load(self, ops_per_second: int, duration_seconds: int) -> None:
        """Generate steady background load."""
        start_time = time.time()
        operation_interval = 1.0 / ops_per_second
        
        while time.time() - start_time < duration_seconds:
            request_data = f"request_{random.randint(0, 1000000)}".encode()
            self.simulator.submit_request(request_data)
            time.sleep(operation_interval)

class PerformanceTestSuite:
    """Comprehensive performance testing scenarios."""
    
    def __init__(self):
        self.results: Dict[str, TestMetrics] = {}
        
    def test_sustained_throughput(self, cluster_size: int = 5) -> TestMetrics:
        """Test maximum sustained throughput."""
        simulator = ClusterSimulator(cluster_size)
        simulator.create_cluster()
        simulator.start_cluster()
        
        # Progressive load testing
        load_levels = [100, 500, 1000, 2000, 5000]
        
        for target_ops_per_sec in load_levels:
            print(f"Testing {target_ops_per_sec} ops/sec...")
            
            # Run for 60 seconds at this load level
            self._run_sustained_load(simulator, target_ops_per_sec, 60)
            
            metrics = simulator.finalize_metrics()
            
            # Check if performance is still acceptable
            if metrics.average_latency_ms > 100 or metrics.success_rate() < 0.99:
                print(f"Performance degraded at {target_ops_per_sec} ops/sec")
                break
                
        return simulator.finalize_metrics()
        
    def _run_sustained_load(self, simulator: ClusterSimulator, 
                           ops_per_second: int, duration_seconds: int) -> None:
        """Run sustained load for specified duration."""
        start_time = time.time()
        operation_interval = 1.0 / ops_per_second
        next_operation_time = start_time
        
        while time.time() - start_time < duration_seconds:
            current_time = time.time()
            
            if current_time >= next_operation_time:
                request_data = f"perf_test_{random.randint(0, 1000000)}".encode()
                simulator.submit_request(request_data)
                next_operation_time += operation_interval
            else:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
```

#### Core Testing Skeleton Code

**Property-Based Test Implementation:**

```python
def test_consensus_safety_properties():
    """
    Comprehensive property-based safety testing.
    Uses Hypothesis to generate random scenarios and verify invariants.
    """
    # TODO 1: Set up property test with random cluster configurations
    # TODO 2: Generate random operation sequences (elections, requests, failures)
    # TODO 3: For each scenario, verify all safety invariants hold
    # TODO 4: Use shrinking to find minimal failing cases
    # TODO 5: Record any property violations with full scenario context
    
def verify_election_safety_invariant(cluster_state):
    """
    Verify election safety: at most one leader per term.
    """
    # TODO 1: Extract current term and leader information from all nodes
    # TODO 2: Group nodes by term number
    # TODO 3: For each term, count number of leaders
    # TODO 4: Assert count <= 1 for all terms
    # TODO 5: Return detailed violation information if assertion fails

def verify_log_consistency_invariant(cluster_state):
    """
    Verify log matching property across all nodes.
    """
    # TODO 1: Extract log contents from all nodes
    # TODO 2: For each pair of nodes, compare logs entry by entry
    # TODO 3: If entries at same index have same term, verify all preceding entries match
    # TODO 4: Check that committed entries are identical across all nodes
    # TODO 5: Return list of any consistency violations found

def test_chaos_leader_isolation():
    """
    Test leader isolation scenario with background load.
    """
    # TODO 1: Create 5-node cluster and wait for stable leader
    # TODO 2: Start background client request load (10 ops/sec)
    # TODO 3: After 30 seconds, create network partition isolating leader
    # TODO 4: Verify majority partition elects new leader within election timeout
    # TODO 5: Continue load for 60 seconds, verify requests succeed
    # TODO 6: Heal partition and verify cluster reconvergence
    # TODO 7: Verify no data loss or consistency violations occurred

def test_performance_sustained_throughput():
    """
    Measure maximum sustained throughput under realistic conditions.
    """
    # TODO 1: Create 5-node cluster with stable leader
    # TODO 2: Start with low load (100 ops/sec) and verify baseline performance
    # TODO 3: Progressively increase load: 500, 1000, 2000, 5000 ops/sec
    # TODO 4: At each level, measure latency (avg, P99) and success rate
    # TODO 5: Identify maximum sustainable load (success rate > 99%, latency < 50ms)
    # TODO 6: Record detailed performance metrics for analysis
    # Hint: Use threading to generate consistent load patterns
```

#### Milestone Checkpoints

**Checkpoint 1 - Election Safety Verification:**
```bash
# Run election safety tests
python -m pytest tests/test_election.py::test_election_safety -v

# Expected output:
# ✓ Single node becomes leader immediately
# ✓ Three node cluster elects exactly one leader
# ✓ Leader failure triggers new election within timeout
# ✓ Split vote scenarios eventually converge to single leader
# ✓ Network partitions maintain single leader per majority
```

**Checkpoint 2 - Replication Consistency:**
```bash
# Run log replication tests  
python -m pytest tests/test_replication.py::test_log_consistency -v

# Expected output:
# ✓ Log matching property maintained under normal operation
# ✓ Follower crash and recovery preserves log consistency
# ✓ Network partitions don't create conflicting commits
# ✓ Log conflicts resolved correctly after partition healing
```

**Checkpoint 3 - Compaction Robustness:**
```bash
# Run log compaction tests
python -m pytest tests/test_compaction.py::test_snapshot_safety -v

# Expected output:
# ✓ Snapshots preserve exact state machine state
# ✓ InstallSnapshot correctly transfers state to followers
# ✓ Log truncation only occurs after confirmed snapshot
# ✓ Concurrent operations continue during snapshotting
```

**Checkpoint 4 - Membership Safety:**
```bash
# Run membership change tests
python -m pytest tests/test_membership.py::test_joint_consensus -v

# Expected output:
# ✓ Membership changes require dual majority approval
# ✓ New nodes catch up before participating in consensus
# ✓ Cluster remains available during membership transitions
# ✓ Failed membership changes rollback safely


## Debugging Guide

> **Milestone(s):** All milestones (1-4) - comprehensive debugging is essential throughout election, replication, compaction, and membership changes

### Mental Model: Medical Diagnosis

Think of debugging a consensus system like diagnosing a complex medical condition. Just as a doctor systematically examines symptoms, runs tests, and traces root causes through interconnected body systems, debugging Raft requires understanding how distributed symptoms manifest from underlying system failures. A patient's headache might stem from dehydration, stress, or a serious neurological condition - similarly, a "split vote" symptom might indicate timing issues, network problems, or configuration bugs. The key is building a diagnostic framework that moves systematically from observable symptoms to root causes, using the right tools and techniques at each step.

Unlike debugging single-threaded applications where you can step through code linearly, consensus systems exhibit **emergent properties** - behaviors that only appear when multiple nodes interact. A bug might only manifest when three specific conditions occur simultaneously across different nodes. This makes debugging both more challenging and more critical, since consensus bugs can lead to data loss or split-brain scenarios that compromise system safety.

The debugging approach for Titan follows a structured methodology: first establish what "healthy" behavior looks like for each milestone, then build comprehensive observability into the system, and finally develop systematic procedures for diagnosing and resolving issues when they arise. This section provides that complete diagnostic framework.

### Debugging Tools and Techniques

#### Comprehensive Logging Strategy

The foundation of consensus debugging is **structured logging** that captures the distributed state machine's evolution across all nodes. Unlike application logging that focuses on business events, consensus logging must capture the precise timing and causality of distributed operations.

**Log Level Hierarchy**

The logging system uses a hierarchical approach where each level serves specific debugging purposes:

| Log Level | Purpose | What to Log | When to Use |
|-----------|---------|-------------|------------|
| TRACE | Message-level debugging | Every RPC sent/received with full content | During development and deep debugging |
| DEBUG | State transition debugging | State changes, election events, log operations | Normal development and integration testing |
| INFO | Operational events | Leader elections, membership changes, snapshots | Production monitoring |
| WARN | Recoverable issues | Network timeouts, rejected votes, log conflicts | Production alerting |
| ERROR | Safety violations | Split brain detection, corrupted logs, persistent failures | Immediate production attention |

**Structured Log Format**

Every log entry follows a consistent structure that enables automated analysis and correlation across nodes:

```
[TIMESTAMP] [LEVEL] [NODE_ID] [COMPONENT] [TERM] [EVENT_TYPE] message key1=value1 key2=value2
```

For example:
```
[2024-01-15T10:30:45.123Z] [DEBUG] [node-1] [ELECTION] [term=5] [VOTE_RECEIVED] received vote from node-2 granted=true last_log_index=1247
```

**Component-Specific Logging**

Each major component implements detailed logging that captures its specific responsibilities:

| Component | Key Events to Log | Essential Fields |
|-----------|------------------|------------------|
| Election | Vote requests/responses, timeouts, state transitions | term, candidate_id, vote_granted, election_deadline |
| Replication | AppendEntries RPCs, log conflicts, commit updates | prev_log_index, entries_count, success, conflict_index |
| Compaction | Snapshot creation, InstallSnapshot RPCs, log truncation | snapshot_index, snapshot_size, install_progress |
| Membership | Configuration changes, catch-up progress, dual majorities | old_config, new_config, ready_nodes |

#### State Inspection and Debugging Tools

**Runtime State Dump**

The system provides comprehensive state inspection capabilities that capture the complete distributed state at any moment. This is crucial because consensus bugs often involve subtle inconsistencies that only become apparent when examining multiple nodes simultaneously.

The state dump includes:

| State Category | Information Captured | Format |
|----------------|---------------------|---------|
| Node State | current_term, voted_for, node_state, leader_id | JSON with timestamps |
| Log State | log entries, commit_index, last_applied | Structured log view with checksums |
| Network State | active connections, pending RPCs, failure counts | Connection matrix with latencies |
| Timing State | election_deadline, last_heartbeat, timeout values | Millisecond precision with drift |
| Memory State | volatile state, caches, pending operations | Memory usage with object counts |

**Distributed State Consistency Checker**

Since consensus systems must maintain consistency across multiple nodes, the debugging tools include automated consistency verification:

```python
def validate_cluster_consistency() -> List[str]:
    """
    Verify consistency properties across all nodes in the cluster.
    Returns list of consistency violations found.
    """
    # Implementation details in Implementation Guidance
```

This checker validates critical invariants:

| Invariant | Check Performed | Violation Severity |
|-----------|----------------|-------------------|
| Single Leader | At most one leader per term across all nodes | CRITICAL - indicates split brain |
| Log Matching | Identical entries at same index across committed logs | CRITICAL - data corruption risk |
| Term Monotonicity | Terms never decrease, votes consistent within terms | HIGH - safety violation |
| Commit Safety | Committed entries never change | CRITICAL - durability violation |
| Configuration Consistency | All nodes agree on cluster membership | HIGH - availability risk |

**Performance Profiling Tools**

Consensus systems are sensitive to timing, making performance profiling essential for debugging latency and throughput issues:

| Metric Category | Measurements | Diagnostic Value |
|-----------------|--------------|------------------|
| RPC Latencies | Per-operation timing distribution | Identify network bottlenecks |
| Election Timing | Time to elect leader, split vote frequency | Tune timeout parameters |
| Log Throughput | Entries per second, batch sizes | Optimize replication performance |
| Memory Usage | Log growth, snapshot overhead | Detect memory leaks |
| CPU Profiling | Hot paths, lock contention | Optimize critical sections |

#### Debugging Approaches and Methodologies

**Systematic Root Cause Analysis**

When debugging consensus issues, follow this structured approach:

1. **Symptom Identification**: What observable behavior differs from expected? Document the exact symptoms with timestamps and affected operations.

2. **Scope Determination**: Is this affecting one node, a subset, or the entire cluster? Network partitions often create different symptoms on different sides of the split.

3. **Timeline Reconstruction**: Use structured logs to build a timeline of events leading to the issue. Pay special attention to timing relationships between nodes.

4. **Invariant Checking**: Run consistency checkers to identify which safety or liveness properties are being violated.

5. **Hypothesis Formation**: Based on symptoms and invariant violations, form specific hypotheses about root causes.

6. **Targeted Investigation**: Use appropriate debugging tools to test each hypothesis systematically.

7. **Fix Validation**: After implementing a fix, verify that it resolves the root cause without introducing new issues.

**Deterministic Replay Debugging**

For particularly complex bugs, the system supports deterministic replay of distributed scenarios:

| Replay Component | Captured Information | Replay Capability |
|------------------|---------------------|-------------------|
| Message Ordering | Exact RPC timing and ordering | Reproduce race conditions |
| Random Events | Election timeout values, node IDs | Eliminate non-determinism |
| External Events | Client requests, node failures | Replay exact failure scenarios |
| Network Conditions | Partitions, delays, packet loss | Reproduce network-dependent bugs |

> **Design Insight**: Deterministic replay is invaluable for consensus debugging because many bugs only occur under specific timing conditions that are nearly impossible to reproduce manually. The investment in building replay capability pays off enormously when debugging complex distributed race conditions.

### Common Election Bugs

#### Split Vote and Election Failures

**Split Vote Scenarios**

Split votes occur when no candidate receives a majority, requiring a new election. While occasional split votes are normal, frequent split votes indicate serious timing or configuration issues.

| Split Vote Cause | Symptoms | Root Cause | Fix |
|------------------|----------|------------|-----|
| Synchronized timeouts | Elections start simultaneously across multiple nodes | Insufficient randomization in election timeouts | Increase timeout randomization range |
| Network partitions | Candidates can't reach majority of nodes | Network connectivity issues | Improve failure detection, partition handling |
| Clock skew | Nodes have different views of timeout expiration | System clock synchronization problems | Use monotonic clocks, implement clock skew detection |
| Configuration mismatch | Nodes disagree on cluster membership | Inconsistent configuration deployment | Implement configuration validation and synchronization |

⚠️ **Pitfall: Deterministic Election Timeouts**

A common mistake is using deterministic or poorly randomized election timeouts. If multiple nodes use the same timeout value, they'll start elections simultaneously, leading to persistent split votes. The randomization must be cryptographically secure and use sufficient range (typically 150-300ms spread).

**Symptoms**: Frequent elections with no winner, high election frequency in logs, cluster unable to elect leader.

**Diagnosis**: Check if multiple nodes log election starts at nearly the same time. Look for patterns in timeout values.

**Fix**: Implement proper randomization using the system's cryptographic random number generator, ensure sufficient timeout range based on network latency measurements.

**Term Confusion Problems**

Term confusion occurs when nodes have inconsistent views of the current term, leading to rejected votes and failed elections.

| Term Issue | Detection | Root Cause | Resolution |
|------------|-----------|------------|------------|
| Term regression | Node receives RPC with lower term than current | Persistent state corruption | Restore from backup, investigate storage issues |
| Missing term updates | Node doesn't update term after seeing higher term | Failed persistent state write | Check disk space, file permissions, implement retry logic |
| Vote inconsistency | Node votes for multiple candidates in same term | Race condition in vote handling | Add proper locking around vote state updates |
| Premature term increment | Node increments term without proper cause | Spurious election triggers | Review election timeout and failure detection logic |

⚠️ **Pitfall: Non-Atomic Vote Updates**

Failing to atomically update both the current term and voted_for fields can lead to nodes voting multiple times in the same term, violating election safety.

**Symptoms**: Multiple candidates claiming victory in same term, inconsistent vote counts, election safety violations.

**Diagnosis**: Search logs for multiple vote grants from same node within a single term.

**Fix**: Ensure the `update_term()` method atomically updates both fields with proper persistence guarantees.

**Election Timeout Issues**

Incorrect election timeout handling is a frequent source of election problems:

| Timeout Issue | Symptoms | Debugging Steps | Solution |
|---------------|----------|----------------|----------|
| Too aggressive timeouts | Constant elections, leader churn | Monitor election frequency vs network latency | Increase timeout minimums based on actual network measurements |
| Too conservative timeouts | Slow failure detection, poor availability | Measure actual failure detection time | Decrease timeouts while maintaining split vote prevention |
| Non-monotonic clocks | Spurious timeouts, timing violations | Check for system clock adjustments | Use monotonic clock sources exclusively |
| Timeout reset bugs | Elections don't start when leader fails | Verify timeout reset on heartbeat receipt | Review heartbeat handling logic |

#### Leader Recognition Problems

**Multiple Leader Detection**

The most serious election bug is multiple nodes believing they are leader simultaneously (split brain):

| Split Brain Cause | Detection Method | Prevention | Recovery |
|-------------------|------------------|------------|----------|
| Network partition | Monitor vote counts vs cluster size | Implement proper quorum requirements | Force leader step-down when partition heals |
| Persistent state bugs | Check for multiple nodes with same term as leader | Validate term/vote persistence atomicity | Restart affected nodes, restore from consistent backup |
| Race conditions | Look for simultaneous leader transitions | Add proper synchronization in state transitions | Implement leader lease mechanism |
| Configuration errors | Verify cluster size calculations | Validate configuration deployment | Force re-election with correct configuration |

⚠️ **Pitfall: Incorrect Quorum Calculation**

Calculating majority as `cluster_size / 2` instead of `cluster_size / 2 + 1` can lead to split brain scenarios in even-sized clusters during partitions.

**Symptoms**: Multiple leaders logged simultaneously, conflicting log entries, client operations appearing to succeed but disappearing later.

**Diagnosis**: Check for multiple nodes logging "became leader" with same or overlapping terms.

**Fix**: Implement correct majority calculation: `len(cluster) // 2 + 1` and validate against actual responses received.

**Leader Isolation Issues**

Leaders can become isolated while still believing they hold leadership:

| Isolation Scenario | Detection | Consequences | Resolution |
|--------------------|-----------|--------------|------------|
| Network asymmetry | Leader can send but not receive responses | Leader accepts operations that can't be committed | Implement heartbeat response requirements |
| Partial partition | Leader connected to minority of followers | Split brain risk when majority elects new leader | Monitor follower response rates, step down if insufficient |
| Slow followers | Followers too slow to respond to heartbeats | Premature leader elections | Implement adaptive timeouts based on follower performance |
| Resource exhaustion | Leader overwhelmed, can't send heartbeats | Unnecessary elections due to apparent failure | Monitor leader resource usage, implement backpressure |

### Common Replication Bugs

#### Log Inconsistency Issues

**Log Matching Violations**

The log matching property ensures that if two logs contain an entry with the same index and term, then the logs are identical for all entries up through that index. Violations of this property indicate serious bugs that can lead to data divergence.

| Violation Type | Symptoms | Root Cause | Repair Strategy |
|----------------|----------|------------|-----------------|
| Term mismatch | Same index has different terms on different nodes | Leader crash during replication, incomplete writes | Force follower to adopt leader's log through conflict resolution |
| Content mismatch | Same index/term has different data on different nodes | Data corruption, serialization bugs, storage issues | Identify corrupted node, restore from consistent replica |
| Missing entries | Follower missing committed entries | Network issues during replication, follower crashes | Re-send missing entries through normal AppendEntries |
| Duplicate entries | Follower has duplicate entries at different indices | Retry logic bugs, RPC delivery duplicates | Implement idempotent log append operations |

⚠️ **Pitfall: Insufficient Conflict Detection**

Failing to check both index and term when detecting log conflicts can lead to accepting inconsistent log entries. The conflict detection must verify that the entry at `prev_log_index` has exactly the term specified in `prev_log_term`.

**Symptoms**: Nodes have different data for same log index, application state divergence, inconsistent query results.

**Diagnosis**: Compare log contents across nodes, look for entries with same index but different terms or content.

**Fix**: Implement comprehensive conflict detection that checks both index bounds and term matching, with proper fallback to conflict resolution.

**Commit Index Synchronization**

Commit index represents the highest log entry known to be replicated on a majority of nodes. Bugs in commit index handling can lead to premature commitment or failure to commit safe entries.

| Commit Issue | Detection | Impact | Resolution |
|--------------|-----------|--------|------------|
| Premature commit | Commit index exceeds actual replication | Data loss on leader failure | Implement conservative commit advancement based on confirmed responses |
| Delayed commit | Safe entries not committed promptly | Poor client latency, unnecessary re-replication | Review commit index update logic, ensure timely advancement |
| Inconsistent commit | Followers have different commit indices | Application state divergence | Ensure commit index included in all AppendEntries messages |
| Commit regression | Commit index moves backward | Safety violation, potential data loss | Never allow commit index to decrease, investigate root cause |

#### Network Partition Handling

**Minority Partition Recovery**

When a network partition occurs, nodes in the minority partition cannot elect a leader and must not commit new entries. Proper partition handling ensures that these nodes correctly rejoin the cluster when the partition heals.

| Partition Issue | Symptoms | Debugging | Recovery |
|-----------------|----------|-----------|----------|
| Minority commits | Minority partition commits entries | Check commit logs vs cluster size | Reject commits with insufficient responses |
| Stale leader | Old leader continues operating in minority | Monitor response counts from followers | Implement leader step-down on insufficient heartbeat responses |
| Partition amnesia | Node forgets it was partitioned | Review join/rejoin logic | Implement proper state synchronization on partition heal |
| Conflicting operations | Operations committed on both sides of partition | Compare logs across partition heal | Implement conflict resolution, may require operator intervention |

⚠️ **Pitfall: Trusting Local State During Partitions**

Nodes in a minority partition might continue to believe they can commit operations if they don't properly count responses. Every commit decision must be based on actual responses received, not assumptions about cluster state.

**Symptoms**: Operations appear committed but disappear after partition heal, multiple conflicting entries for same index.

**Diagnosis**: Check commit logic vs response counts, look for commits with insufficient acknowledgments.

**Fix**: Ensure commit decisions require actual majority responses, implement timeout-based failure detection for unresponsive followers.

**Majority Partition Behavior**

The majority partition must continue operating normally while properly handling the temporary unavailability of minority nodes.

| Majority Issue | Detection | Prevention | Handling |
|----------------|-----------|------------|----------|
| Premature timeout | Majority gives up on minority too quickly | Monitor timeout vs network latency | Adjust timeouts for network conditions |
| Resource leakage | Accumulating state for unavailable nodes | Monitor memory usage for pending operations | Implement cleanup for long-disconnected nodes |
| Performance degradation | Majority slows down due to minority timeouts | Measure operation latency distribution | Implement adaptive timeouts and parallel operations |
| State bloat | Logs grow due to minority catch-up requirements | Monitor log size growth vs compaction | Ensure minority nodes can catch up through snapshots |

#### Entry Commitment Protocol Issues

**Majority Replication Verification**

The entry commitment protocol ensures that entries are only considered committed when they have been replicated to a majority of nodes. Bugs in this protocol can lead to lost or inconsistent data.

| Commitment Bug | Symptoms | Root Cause | Fix |
|----------------|----------|------------|-----|
| False majority | Commits with insufficient replication | Incorrect response counting | Implement proper majority calculation and response tracking |
| Replication gaps | Committed entries not on all followers | Network issues, follower crashes | Ensure eventual consistency through catch-up mechanisms |
| Term boundary issues | Entries from previous terms not committed | Leader change during replication | Implement proper term-based commit rules |
| Ordering violations | Entries committed out of order | Race conditions in parallel replication | Ensure commit index advances monotonically |

**Previous Term Entry Handling**

Raft requires special handling for entries from previous terms to ensure safety across leader changes:

| Previous Term Issue | Safety Risk | Detection | Resolution |
|---------------------|-------------|-----------|------------|
| Premature commitment | Committing previous term entries without current term entry | Check term of committed entries | Only commit previous term entries after current term entry is majority replicated |
| Term confusion | Mixing current and previous term commitment rules | Review commitment logic per term | Implement separate handling for current vs previous term entries |
| Leadership validation | New leader commits previous entries incorrectly | Verify leadership establishment | Ensure new leader commits a no-op entry before committing previous term entries |
| State machine application | Applying uncommitted previous term entries | Monitor commit vs apply indices | Never apply entries beyond committed index |

### Common Compaction Bugs

#### Snapshot Timing and Safety Issues

**State Machine Freeze Problems**

During snapshot creation, the state machine must be temporarily frozen to ensure consistency. Bugs in the freeze mechanism can lead to corrupted snapshots or blocked operations.

| Freeze Issue | Symptoms | Root Cause | Resolution |
|--------------|----------|------------|------------|
| Incomplete freeze | Snapshot contains inconsistent state | Concurrent modifications during snapshot | Implement proper read-write locks for state machine access |
| Deadlock | System hangs during snapshot creation | Lock ordering issues, nested locks | Review lock acquisition order, implement timeout-based deadlock detection |
| Extended freeze | Operations blocked for excessive time | Slow snapshot serialization, large state | Implement incremental snapshotting, optimize serialization |
| Partial restoration | Restored state missing recent updates | Applied entries after snapshot started | Ensure snapshot includes all entries up to snapshot index |

⚠️ **Pitfall: Snapshotting Without Proper Coordination**

Creating snapshots while the state machine continues processing can result in snapshots that represent no valid point-in-time state. The snapshot must reflect the exact state after applying all entries up to the snapshot index.

**Symptoms**: Restored nodes have inconsistent state, application invariants violated after restoration.

**Diagnosis**: Compare application state after restoration with other nodes, check for partial or inconsistent updates.

**Fix**: Implement proper state machine coordination that ensures snapshots capture atomic state at a specific log index.

**Log Truncation Safety**

After creating a snapshot, old log entries can be safely removed, but truncation bugs can lead to data loss or inability to catch up followers.

| Truncation Issue | Safety Risk | Detection | Prevention |
|------------------|-------------|-----------|------------|
| Premature truncation | Log entries truncated before snapshot confirmed | Data loss if snapshot corrupted | Only truncate after snapshot persistence confirmed |
| Incomplete truncation | Old entries not removed, log continues growing | Resource exhaustion | Implement complete truncation up to snapshot index |
| Metadata inconsistency | Snapshot metadata doesn't match truncated log | Restoration failures | Ensure atomic update of snapshot metadata and log truncation |
| Follower consideration | Truncation before followers catch up | Followers can't catch up through log replication | Monitor follower progress before truncation |

#### InstallSnapshot RPC Issues

**Chunked Transfer Problems**

Large snapshots must be transferred in chunks to avoid overwhelming the network or RPC layer. Bugs in chunked transfer can lead to corrupted or incomplete snapshot installation.

| Transfer Issue | Symptoms | Root Cause | Resolution |
|----------------|----------|------------|------------|
| Chunk ordering | Snapshot restored incorrectly | Out-of-order chunk delivery | Implement sequence numbers and ordered reassembly |
| Missing chunks | Installation fails with incomplete data | Network drops, RPC failures | Add chunk acknowledgments and retry logic |
| Corruption detection | Corrupted snapshots accepted | No integrity checking | Implement checksums for individual chunks and complete snapshot |
| Memory management | Out of memory during large transfers | Accumulating chunks in memory | Stream chunks to disk, implement backpressure |
| Timeout handling | Transfers fail due to timeouts | Large snapshots, slow networks | Implement appropriate timeouts based on snapshot size |

⚠️ **Pitfall: Blocking on Snapshot Installation**

Installing snapshots can take significant time, and blocking the main consensus loop during installation can cause leader elections and cluster instability.

**Symptoms**: Frequent elections during snapshot installation, cluster unavailability during catch-up.

**Diagnosis**: Check timing correlation between snapshot installation and election events.

**Fix**: Implement asynchronous snapshot installation that doesn't block normal consensus operations, maintain heartbeat responses during installation.

**Concurrent Operations During Installation**

While a follower is installing a snapshot, it must handle concurrent AppendEntries requests appropriately to maintain consistency.

| Concurrency Issue | Problem | Detection | Handling |
|--------------------|---------|-----------|----------|
| Stale AppendEntries | Receiving entries for replaced log | Check entry indices vs snapshot | Reject AppendEntries with indices covered by installing snapshot |
| Installation interruption | New leader interrupts ongoing installation | Snapshot installation fails | Implement cancellation and cleanup for interrupted installations |
| State confusion | Node unsure whether to use log or snapshot | Inconsistent state during transition | Clearly define state machine source during installation |
| Duplicate installation | Multiple concurrent installations | Resource exhaustion, corruption | Implement installation state tracking to prevent duplicates |

#### Snapshot Storage and Persistence

**Atomic Snapshot Storage**

Snapshots must be stored atomically to prevent corruption from crashes during the storage process:

| Storage Issue | Risk | Detection | Prevention |
|---------------|------|-----------|------------|
| Partial writes | Corrupted snapshot files | Checksum validation fails | Write to temporary file, then atomic rename |
| Metadata inconsistency | Snapshot metadata doesn't match content | Restoration validation fails | Store metadata and content atomically |
| Storage failures | Disk full, permission errors | Write operations fail | Check available space, validate permissions before writing |
| Backup corruption | Backup snapshots corrupted | Multiple restoration attempts fail | Implement multiple backup copies with validation |

**Snapshot Validation and Recovery**

Comprehensive validation ensures snapshot integrity and enables recovery from corruption:

| Validation Type | Check Performed | Recovery Action |
|-----------------|----------------|-----------------|
| Checksum validation | Compare stored vs computed checksums | Retry from backup snapshot or rebuild from log |
| Metadata validation | Verify snapshot metadata consistency | Restore metadata from log analysis |
| Content validation | Check snapshot content against known invariants | Rebuild snapshot from log replay |
| Size validation | Verify expected vs actual snapshot size | Detect truncated files, restore from backup |

### Performance Debugging

#### Identifying Bottlenecks

**Coordination Overhead Analysis**

Consensus systems inherently have coordination overhead that can become a bottleneck under high load. Understanding and optimizing this overhead is crucial for performance.

| Bottleneck Type | Symptoms | Measurement | Optimization |
|-----------------|----------|-------------|--------------|
| Serialization overhead | High CPU usage during log operations | Profile serialization time per entry | Optimize serialization format, implement batching |
| Network saturation | High latency, dropped messages | Monitor network bandwidth utilization | Implement message compression, reduce message frequency |
| Disk I/O latency | Slow operation completion | Monitor fsync latency distribution | Optimize storage configuration, implement batching |
| Lock contention | Threads blocked waiting for locks | Profile lock wait times | Reduce lock scope, implement lock-free operations where possible |
| Memory allocation | High GC pressure, allocation rate | Monitor memory allocation patterns | Implement object pooling, reduce allocations |

**Latency Analysis**

Understanding the latency breakdown helps identify optimization opportunities:

| Latency Component | Measurement | Typical Range | Optimization Target |
|-------------------|-------------|---------------|-------------------|
| Network RTT | Round-trip time between nodes | 1-50ms | Network optimization, topology |
| Disk persistence | Time to fsync log entries | 1-10ms | Storage optimization, batching |
| Serialization | Time to encode/decode messages | 0.1-1ms | Format optimization, caching |
| Lock acquisition | Time waiting for exclusive access | 0.01-1ms | Reduce contention, lock-free algorithms |
| State machine application | Time to apply committed entries | 0.1-10ms | Application optimization |

⚠️ **Pitfall: Premature Optimization**

Optimizing the wrong bottleneck can waste significant effort and potentially introduce bugs. Always measure and identify the actual bottleneck before optimizing.

**Symptoms**: Performance improvements don't materialize despite optimization efforts.

**Diagnosis**: Use profiling tools to identify actual time spent in different system components.

**Fix**: Focus optimization efforts on the components that consume the most time in realistic workloads.

#### Optimizing Consensus Performance

**Batching Optimizations**

Batching multiple operations together can significantly improve throughput by amortizing coordination overhead:

| Batching Strategy | Benefits | Implementation Considerations | Trade-offs |
|-------------------|----------|-------------------------------|------------|
| Log entry batching | Reduce disk I/O operations | Implement batch size limits, timeout-based flushing | Increased latency for individual operations |
| Network message batching | Reduce network overhead | Buffer management, message ordering | Memory usage, complexity |
| State machine batching | Reduce application overhead | Transaction boundaries, atomicity | Error handling complexity |
| Response batching | Reduce acknowledgment overhead | Ordering guarantees, failure handling | Increased client latency |

**Adaptive Timeout Tuning**

Consensus performance is highly sensitive to timeout values, which should adapt to actual network conditions:

| Timeout Type | Tuning Strategy | Measurement | Adaptation Algorithm |
|--------------|-----------------|-------------|---------------------|
| Election timeout | Based on network latency distribution | Monitor actual RTT between nodes | Set to 10x mean RTT with minimum safety margin |
| Heartbeat interval | Fraction of election timeout | Monitor heartbeat success rate | Adjust to maintain high success rate while minimizing overhead |
| RPC timeout | Based on operation type and load | Monitor RPC completion times | Use exponential backoff with adaptive maximum |
| Replication timeout | Based on follower performance | Monitor follower response latency | Individual timeouts per follower based on historical performance |

**Throughput Optimization**

Maximizing consensus throughput requires careful balance between batching, parallelism, and coordination overhead:

| Optimization Technique | Throughput Impact | Implementation | Monitoring |
|------------------------|-------------------|----------------|------------|
| Parallel replication | 2-5x improvement | Send AppendEntries to followers concurrently | Monitor response correlation |
| Pipeline replication | 3-10x improvement | Send next batch before previous completes | Track in-flight operations |
| Asynchronous persistence | 2-3x improvement | Overlap disk I/O with network operations | Monitor persistence lag |
| Read optimization | Variable | Implement read-only operations without consensus | Track read vs write ratio |

> **Design Insight**: The key to consensus performance optimization is understanding that consensus is fundamentally about coordinating agreement, not processing data. Most optimizations focus on reducing the coordination overhead while maintaining the essential safety and liveness properties.

**Memory Usage Optimization**

Large-scale consensus deployments must carefully manage memory usage to avoid performance degradation:

| Memory Consumer | Optimization Strategy | Implementation | Monitoring |
|-----------------|----------------------|----------------|------------|
| Log storage | Implement aggressive compaction | Tune snapshot frequency | Monitor log size growth rate |
| Message buffers | Use object pooling | Pre-allocate message objects | Monitor allocation rate |
| State caching | Implement LRU eviction | Cache frequently accessed state | Monitor cache hit rate |
| Network buffers | Tune buffer sizes | Optimize for typical message sizes | Monitor buffer utilization |

#### Performance Testing and Benchmarking

**Load Testing Scenarios**

Comprehensive performance testing requires multiple scenarios that stress different aspects of the consensus system:

| Test Scenario | Purpose | Configuration | Success Criteria |
|---------------|---------|---------------|------------------|
| Sustained throughput | Maximum steady-state performance | Constant load, no failures | Achieve target ops/second for 1+ hours |
| Burst capacity | Handle traffic spikes | Sudden load increase | Handle 10x normal load for 1+ minutes |
| Failure resilience | Performance during failures | Kill nodes during load testing | Maintain >50% throughput during single node failure |
| Network stress | Performance under poor network | Add latency, packet loss | Maintain correctness with 10% packet loss |
| Large cluster scaling | Performance vs cluster size | Test with 3, 5, 7, 9 nodes | Measure coordination overhead scaling |

**Performance Monitoring and Alerting**

Production consensus systems require continuous performance monitoring to detect degradation before it impacts applications:

| Metric | Monitoring | Alert Threshold | Diagnostic Action |
|--------|------------|-----------------|-------------------|
| Operation latency | P99 latency tracking | >3x baseline | Check for network/disk issues |
| Election frequency | Elections per hour | >10 per hour | Investigate timeout tuning |
| Log replication lag | Follower lag behind leader | >1000 entries | Check follower health |
| Disk utilization | Log growth vs compaction | >80% disk usage | Trigger emergency compaction |
| Memory utilization | Heap usage growth | >90% heap usage | Investigate memory leaks |

### Implementation Guidance

This implementation guidance provides the concrete tools and code needed to build comprehensive debugging capabilities into Titan. The focus is on providing complete, production-ready debugging infrastructure that developers can use immediately.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Logging | Python's `logging` module with JSON formatter | Structured logging with `structlog` + centralized aggregation |
| State Inspection | JSON dumps with pretty printing | Interactive debugging with `pdb` + state visualization |
| Performance Profiling | Built-in `cProfile` + `pstats` | `py-spy` for production profiling + flame graphs |
| Network Simulation | Socket delays with `time.sleep()` | `netem` network emulation for realistic conditions |
| Test Framework | `unittest` with custom consensus assertions | `pytest` with property-based testing using `hypothesis` |

#### Recommended File Structure

```
titan/
  debug/
    __init__.py
    logger.py              ← Structured logging configuration
    state_inspector.py     ← Runtime state inspection tools
    consistency_checker.py ← Distributed state validation
    profiler.py           ← Performance profiling utilities
    replay.py             ← Deterministic replay framework
  testing/
    chaos/
      __init__.py
      network_simulator.py ← Network partition/delay simulation
      failure_injector.py  ← Systematic failure injection
    property/
      __init__.py
      invariants.py        ← Consensus invariant checkers
      generators.py        ← Test case generators
  scripts/
    debug_cluster.py      ← Interactive cluster debugging
    analyze_logs.py       ← Log analysis and correlation
    performance_test.py   ← Automated performance testing
```

#### Structured Logging Infrastructure

```python
import logging
import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class LogLevel(Enum):
    TRACE = "TRACE"
    DEBUG = "DEBUG" 
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"

@dataclass
class LogContext:
    """Captures the context for a log entry in the consensus system."""
    node_id: str
    term: int
    component: str
    timestamp: float
    
class ConsensusLogger:
    """
    Structured logger for Raft consensus operations.
    Provides consistent logging format across all components.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.logger = logging.getLogger(f"titan.{node_id}")
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure structured JSON logging format."""
        # TODO 1: Set up JSON formatter that includes node_id, timestamp, level
        # TODO 2: Configure log levels based on environment (DEBUG for dev, INFO for prod)
        # TODO 3: Add file handler for persistent logging
        # TODO 4: Add console handler for development debugging
        pass
    
    def log_election_event(self, event_type: str, term: int, **kwargs) -> None:
        """Log election-related events with consistent structure."""
        # TODO 1: Create LogContext with current term and ELECTION component
        # TODO 2: Add event_type to the log data
        # TODO 3: Include all kwargs as additional fields
        # TODO 4: Emit structured log entry
        pass
    
    def log_replication_event(self, event_type: str, term: int, **kwargs) -> None:
        """Log replication-related events."""
        # TODO 1: Create LogContext with REPLICATION component
        # TODO 2: Include prev_log_index, entries_count if available in kwargs
        # TODO 3: Add success/failure status
        # TODO 4: Emit with appropriate log level based on event_type
        pass
    
    def log_rpc_event(self, rpc_type: str, target: str, success: bool, 
                     latency_ms: float, **kwargs) -> None:
        """Log RPC operations with timing information."""
        # TODO 1: Include RPC timing and success metrics
        # TODO 2: Add target node information
        # TODO 3: Include request/response size if available
        # TODO 4: Log at WARN level if latency exceeds thresholds
        pass

# Global logger registry for multi-node debugging
_loggers: Dict[str, ConsensusLogger] = {}

def get_logger(node_id: str) -> ConsensusLogger:
    """Get or create logger for a specific node."""
    if node_id not in _loggers:
        _loggers[node_id] = ConsensusLogger(node_id)
    return _loggers[node_id]
```

#### State Inspection Tools

```python
from typing import Dict, List, Any
import json
from dataclasses import asdict

class StateInspector:
    """
    Comprehensive state inspection for debugging consensus issues.
    Provides snapshots of distributed state across the cluster.
    """
    
    def __init__(self, cluster_nodes: Dict[str, Any]):
        self.cluster_nodes = cluster_nodes
    
    def dump_cluster_state(self) -> Dict[str, Any]:
        """
        Create comprehensive state dump of entire cluster.
        Returns structured data suitable for analysis.
        """
        # TODO 1: Iterate through all nodes in cluster
        # TODO 2: For each node, capture persistent state (term, voted_for, log)
        # TODO 3: Capture volatile state (node_state, commit_index, leader_id)
        # TODO 4: Include timing information (election_deadline, last_heartbeat)
        # TODO 5: Add network state (connections, pending RPCs)
        # TODO 6: Return as structured dictionary with timestamps
        pass
    
    def validate_cluster_consistency(self) -> List[str]:
        """
        Check consensus invariants across the cluster.
        Returns list of violations found.
        """
        violations = []
        
        # TODO 1: Check election safety (at most one leader per term)
        # TODO 2: Verify log matching property across all nodes
        # TODO 3: Ensure commit index never decreases
        # TODO 4: Validate that committed entries are identical across nodes
        # TODO 5: Check term monotonicity (terms never go backward)
        # TODO 6: Verify configuration consistency across nodes
        # For each violation found, append descriptive message to violations list
        
        return violations
    
    def analyze_election_state(self) -> Dict[str, Any]:
        """Analyze current election state and detect issues."""
        # TODO 1: Identify current leaders and candidates
        # TODO 2: Check for split vote scenarios
        # TODO 3: Analyze timeout distributions and randomization
        # TODO 4: Detect synchronized election starts
        # TODO 5: Return analysis with recommendations
        pass
    
    def check_replication_health(self) -> Dict[str, Any]:
        """Analyze log replication health across followers."""
        # TODO 1: Calculate replication lag for each follower
        # TODO 2: Identify followers with log conflicts
        # TODO 3: Check commit index synchronization
        # TODO 4: Detect network partition effects
        # TODO 5: Return health metrics with alerts
        pass

def create_state_snapshot(node) -> Dict[str, Any]:
    """Create detailed state snapshot for a single node."""
    # TODO 1: Capture PersistentState fields (current_term, voted_for, log)
    # TODO 2: Capture VolatileState fields (node_state, commit_index, etc.)
    # TODO 3: Include timing information with high precision
    # TODO 4: Add memory usage statistics
    # TODO 5: Include recent performance metrics
    # TODO 6: Return as JSON-serializable dictionary
    pass
```

#### Performance Profiling Tools

```python
import time
import statistics
from typing import List, Dict
from dataclasses import dataclass, field
from collections import defaultdict, deque

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for consensus operations."""
    operation_latencies: List[float] = field(default_factory=list)
    rpc_latencies: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    election_count: int = 0
    leader_changes: int = 0
    throughput_ops_per_sec: float = 0.0
    memory_usage_mb: float = 0.0
    
    def add_operation_latency(self, latency_ms: float) -> None:
        """Record latency for a client operation."""
        # TODO 1: Add latency to operation_latencies list
        # TODO 2: Maintain sliding window (last 1000 operations)
        # TODO 3: Update throughput calculation if needed
        pass
    
    def add_rpc_latency(self, rpc_type: str, latency_ms: float) -> None:
        """Record latency for specific RPC type."""
        # TODO 1: Add to appropriate RPC type bucket
        # TODO 2: Maintain sliding window per RPC type
        # TODO 3: Update aggregate statistics
        pass
    
    def get_latency_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentile distribution."""
        # TODO 1: Calculate P50, P95, P99 for operation latencies
        # TODO 2: Include max and mean latencies
        # TODO 3: Return as dictionary for easy reporting
        pass
    
    def calculate_throughput(self, window_seconds: float) -> float:
        """Calculate operations per second over time window."""
        # TODO 1: Count operations in time window
        # TODO 2: Divide by window size to get ops/sec
        # TODO 3: Update internal throughput_ops_per_sec field
        pass

class PerformanceProfiler:
    """Real-time performance profiling for consensus operations."""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.metrics = PerformanceMetrics()
        self.operation_start_times: Dict[str, float] = {}
    
    def start_operation(self, operation_id: str) -> None:
        """Start timing a consensus operation."""
        # TODO 1: Record current timestamp for operation_id
        # TODO 2: Store in operation_start_times dictionary
        pass
    
    def end_operation(self, operation_id: str) -> float:
        """End timing and record latency."""
        # TODO 1: Calculate elapsed time since start_operation
        # TODO 2: Add to metrics using add_operation_latency
        # TODO 3: Clean up operation_start_times entry
        # TODO 4: Return the calculated latency
        pass
    
    def profile_rpc(self, rpc_type: str, target_node: str, rpc_func, *args, **kwargs):
        """Profile an RPC call with timing."""
        # TODO 1: Record start time
        # TODO 2: Call rpc_func with provided arguments
        # TODO 3: Calculate elapsed time
        # TODO 4: Record in metrics with add_rpc_latency
        # TODO 5: Return RPC result
        pass
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        # TODO 1: Calculate latency percentiles
        # TODO 2: Include throughput metrics
        # TODO 3: Add RPC-specific statistics
        # TODO 4: Include election and leader change counts
        # TODO 5: Format as structured report
        pass
```

#### Consistency Checking Framework

```python
from typing import Set, Optional
from collections import Counter

class ConsistencyChecker:
    """Validates consensus invariants across distributed cluster."""
    
    def __init__(self, cluster_state: Dict[str, Any]):
        self.cluster_state = cluster_state
        self.violations = []
    
    def check_election_safety(self) -> List[str]:
        """Verify at most one leader per term."""
        term_leaders = defaultdict(list)
        
        # TODO 1: Iterate through all nodes in cluster_state
        # TODO 2: For each node in LEADER state, record (term, node_id)
        # TODO 3: Check if any term has multiple leaders
        # TODO 4: Return list of violation descriptions
        pass
    
    def check_log_matching(self) -> List[str]:
        """Verify log matching property across all nodes."""
        # TODO 1: For each log index, collect entries from all nodes
        # TODO 2: Check that entries at same index have same term
        # TODO 3: Check that entries with same index+term have same content
        # TODO 4: Verify that if entries match at index N, all entries 0..N-1 also match
        # TODO 5: Return descriptions of any violations found
        pass
    
    def check_commit_safety(self) -> List[str]:
        """Verify that committed entries never change."""
        # TODO 1: Find minimum commit_index across all nodes
        # TODO 2: For all entries up to min_commit_index, verify identical content
        # TODO 3: Check that commit_index never decreases on any node
        # TODO 4: Return any safety violations found
        pass
    
    def check_term_monotonicity(self) -> List[str]:
        """Verify that terms never decrease."""
        # TODO 1: Check each node's term progression over time
        # TODO 2: Verify voted_for consistency within each term
        # TODO 3: Check that term increments are justified (elections)
        # TODO 4: Return any monotonicity violations
        pass
    
    def run_all_checks(self) -> List[str]:
        """Run comprehensive consistency validation."""
        all_violations = []
        all_violations.extend(self.check_election_safety())
        all_violations.extend(self.check_log_matching())
        all_violations.extend(self.check_commit_safety())
        all_violations.extend(self.check_term_monotonicity())
        return all_violations
```

#### Milestone Checkpoints

After implementing each milestone, use these checkpoints to verify correct debugging capability:

**Milestone 1 - Election Debugging:**
```bash
# Run election debugging tests
python -m titan.debug.test_election_debugging
# Expected: Detect split votes, timeout issues, multiple leaders
# Verify: Election safety violations caught, proper term tracking
```

**Milestone 2 - Replication Debugging:**
```bash
# Run replication debugging tests  
python -m titan.debug.test_replication_debugging
# Expected: Detect log inconsistencies, commit issues
# Verify: Log matching violations caught, proper conflict detection
```

**Milestone 3 - Compaction Debugging:**
```bash
# Run compaction debugging tests
python -m titan.debug.test_compaction_debugging  
# Expected: Detect snapshot corruption, truncation issues
# Verify: Snapshot safety validated, proper cleanup detection
```

**Milestone 4 - Membership Debugging:**
```bash
# Run membership debugging tests
python -m titan.debug.test_membership_debugging
# Expected: Detect split-brain during transitions
# Verify: Joint consensus violations caught, proper transition tracking
```

#### Debugging Tips

| Symptom | Likely Cause | Diagnostic Command | Fix |
|---------|--------------|-------------------|-----|
| Frequent elections | Timeout misconfiguration | `analyze_logs.py --election-frequency` | Increase timeout range, check network latency |
| Split brain detected | Partition + configuration bug | `debug_cluster.py --check-leaders` | Force re-election, verify quorum calculation |
| Log divergence | Replication conflict handling | `debug_cluster.py --compare-logs` | Implement proper conflict resolution |
| High latency | Network or disk bottleneck | `performance_test.py --profile` | Optimize batching, check storage performance |
| Memory growth | Log not compacting | `debug_cluster.py --memory-usage` | Tune snapshot frequency, check compaction logic |


## Future Extensions

> **Milestone(s):** Post-completion enhancements building on all milestones (1-4) - describes potential optimizations and scaling opportunities

### Mental Model: Urban Planning

Think of Titan's future extensions like urban planning for a growing city. The current implementation is like a well-designed small town with solid infrastructure - roads (RPC layer), utilities (consensus protocol), and governance (leader election). But as the population grows, we need highways (batching), express transit (pipelining), specialized districts (advanced features), and eventually satellite cities (multi-Raft) to handle the scale while maintaining quality of life (performance and reliability).

Just as urban planners must balance current needs with future growth, the extensions we discuss maintain Titan's core guarantees while dramatically improving performance and scale. Each optimization is like adding infrastructure that enhances the city without requiring residents to change how they live.

The beauty of good distributed systems design, like good urban planning, is that the foundational architecture can support these enhancements without requiring a complete rebuild. The consensus protocol remains the same, but we add express lanes for common operations and new districts for specialized workloads.

### Performance Optimization Opportunities

The current Titan implementation prioritizes correctness and clarity over raw performance. This foundation provides numerous opportunities for optimization that can deliver order-of-magnitude improvements while preserving the strong consistency guarantees that make Raft valuable.

#### Batching and Pipelining Optimizations

The most impactful performance improvements come from reducing coordination overhead through intelligent batching and pipelining. These optimizations transform the consensus engine from handling one operation at a time to processing streams of operations efficiently.

> **Decision: Request Batching Architecture**
> - **Context**: Current implementation processes one client request per `AppendEntriesRequest`, creating high per-operation overhead
> - **Options Considered**: 
>   1. Maintain one-request-per-RPC for simplicity
>   2. Time-based batching (collect requests for fixed intervals)
>   3. Adaptive batching (adjust batch size based on load)
> - **Decision**: Implement adaptive batching with configurable limits
> - **Rationale**: Adaptive batching provides optimal performance under varying load while preventing latency spikes during low-traffic periods
> - **Consequences**: Enables 10-100x throughput improvement with minimal latency impact; requires more complex request tracking and timeout management

**Request Batching Implementation Strategy:**

The leader maintains a request buffer that accumulates client operations before packaging them into a single `AppendEntriesRequest`. The batching algorithm balances throughput optimization with latency constraints through adaptive thresholds.

| Batching Parameter | Conservative Setting | Aggressive Setting | Production Recommendation |
|---|---|---|---|
| Max Batch Size | 10 entries | 100 entries | 50 entries |
| Max Batch Delay | 10ms | 100ms | 25ms |
| Dynamic Sizing | Disabled | Load-based | Enable with 2x scaling |
| Flush Triggers | Time only | Time + size + priority | Time + size + load |

The batching algorithm follows this decision process:

1. Client request arrives at leader and enters the pending request queue
2. Leader checks if immediate flush is required (queue full, high-priority request, or flush timer expired)
3. If not flushing immediately, leader starts or extends the batch timer
4. When flush triggers fire, leader packages all pending requests into a single log entry with batch metadata
5. Normal `AppendEntriesRequest` processing occurs with the batched entry
6. After commit, leader unpacks the batch and responds to individual clients with their operation results

**Pipeline Replication Architecture:**

Pipeline replication allows the leader to send multiple `AppendEntriesRequest` messages to followers without waiting for responses, dramatically reducing latency for sequential operations while maintaining safety guarantees.

| Pipeline Configuration | Safety Impact | Performance Gain | Complexity Cost |
|---|---|---|---|
| Pipeline Depth 1 (current) | Maximum safety | Baseline | Minimal |
| Pipeline Depth 5 | No safety impact | 2-3x latency reduction | Moderate |
| Pipeline Depth 20 | No safety impact | 5-10x latency reduction | High |
| Unlimited Pipeline | No safety impact | Maximum performance | Very high |

The pipeline implementation maintains safety through careful sequence number management and ensures that commit decisions respect the log matching property even with outstanding requests.

#### Network and Serialization Optimizations

Current RPC implementation prioritizes simplicity over efficiency, creating opportunities for substantial performance gains through optimized serialization and transport mechanisms.

> **Decision: Serialization Format Selection**
> - **Context**: JSON serialization is human-readable but inefficient for high-throughput consensus
> - **Options Considered**:
>   1. Keep JSON for simplicity and debuggability
>   2. Protocol Buffers for efficiency with schema evolution
>   3. MessagePack for efficiency with JSON compatibility
> - **Decision**: Protocol Buffers with JSON fallback for debugging
> - **Rationale**: Protocol Buffers provide 5-10x serialization performance improvement and built-in schema evolution while maintaining compatibility
> - **Consequences**: Enables higher throughput; requires proto schema management; complicates debugging without tooling

**Transport Layer Enhancements:**

| Transport Feature | Current State | Enhanced Version | Performance Impact |
|---|---|---|---|
| Connection Management | Per-RPC connections | Persistent connection pools | 50% latency reduction |
| Compression | None | gRPC with gzip | 60% bandwidth reduction |
| Multiplexing | Sequential requests | HTTP/2 streams | 3x concurrent request handling |
| Serialization | JSON | Protocol Buffers | 5x serialization speedup |

The enhanced transport layer maintains backward compatibility while providing opt-in performance improvements that can be enabled per-node based on capability negotiation.

#### Memory and Storage Optimizations

The current implementation loads entire logs into memory and uses simple file-based persistence. Production deployments benefit from memory-efficient log management and optimized storage patterns.

**Memory-Efficient Log Management:**

Instead of keeping all log entries in memory, an optimized implementation maintains a sliding window of recent entries with disk-backed storage for historical data.

| Component | Current Approach | Optimized Approach | Memory Reduction |
|---|---|---|---|
| Log Storage | Full in-memory list | LRU cache + disk backing | 90% for large logs |
| Entry Indexing | Linear search | B-tree index | O(log n) vs O(n) lookup |
| Snapshot Metadata | Full metadata in memory | Lazy-loaded metadata | 80% metadata overhead |
| Term Tracking | Per-entry term storage | Run-length encoded terms | 75% term overhead |

### Advanced Raft Features

Beyond basic consensus, Raft supports several advanced features that dramatically improve read performance and enable specialized use cases while maintaining safety guarantees.

#### ReadIndex Optimization

ReadIndex allows followers to serve consistent reads without going through the leader, reducing read latency and distributing read load across the cluster.

> **Decision: ReadIndex Implementation Strategy**
> - **Context**: All reads currently go through leader, creating bottleneck and single point of failure for read workloads
> - **Options Considered**:
>   1. Maintain leader-only reads for simplicity
>   2. Implement basic ReadIndex for followers
>   3. Implement ReadIndex with lease optimization
> - **Decision**: Implement ReadIndex with optional lease optimization
> - **Rationale**: ReadIndex provides immediate read scalability while lease optimization enables local reads for latency-sensitive applications
> - **Consequences**: Enables linear read scaling; requires careful implementation to maintain linearizability; adds complexity to read path

**ReadIndex Protocol Implementation:**

The ReadIndex protocol ensures that follower reads are consistent by confirming that the follower's log is up-to-date before serving reads.

| ReadIndex Step | Actor | Action | Safety Guarantee |
|---|---|---|---|
| 1. Read Request | Client | Sends read to any node | None yet |
| 2. ReadIndex Request | Follower | Requests commit confirmation from leader | Leader validates its authority |
| 3. Heartbeat Confirmation | Leader | Sends heartbeat to majority | Confirms current leadership |
| 4. Index Response | Leader | Returns current commit index | Guarantees linearizability point |
| 5. Log Verification | Follower | Waits for commit index advancement | Ensures up-to-date state |
| 6. State Machine Read | Follower | Executes read against local state | Linearizable result |

This protocol ensures that reads reflect all committed writes that happened before the read request, maintaining linearizability while distributing read load.

#### Lease-Based Read Optimization

Lease-based reads eliminate the ReadIndex protocol overhead by allowing leaders to serve reads locally during lease periods when they're guaranteed to be the current leader.

| Lease Parameter | Conservative Setting | Aggressive Setting | Production Recommendation |
|---|---|---|---|
| Lease Duration | 100ms | 1000ms | 500ms |
| Lease Renewal | Every heartbeat | Manual renewal | Every 2 heartbeats |
| Clock Skew Buffer | 50ms | 10ms | 25ms |
| Lease Read Fallback | Always ReadIndex | Never fallback | ReadIndex on lease expiry |

**Lease Safety Guarantees:**

The lease mechanism maintains safety through careful clock management and conservative lease expiration. The leader can serve reads locally only when it holds a valid lease, which requires:

1. Successful heartbeat responses from a majority within the lease period
2. Local clock verification that the lease hasn't expired accounting for clock skew
3. No pending configuration changes that might affect leadership validity

#### Client Request Deduplication

Production consensus systems must handle client retry scenarios without creating duplicate operations, requiring request deduplication at the consensus layer.

| Deduplication Strategy | Memory Overhead | Duplicate Detection | Implementation Complexity |
|---|---|---|---|
| No Deduplication | None | None | Minimal |
| Session-Based | Low (per-session) | Perfect within session | Moderate |
| Global Request IDs | Medium (per-request) | Perfect globally | High |
| Hybrid Approach | Low-Medium | Near-perfect | Moderate-High |

**Session-Based Deduplication Design:**

Each client establishes a session with a unique session ID and maintains a sequence number for requests within that session. The consensus engine tracks the last applied sequence number per session and ignores duplicate requests.

| Session Component | Purpose | Storage Location | Persistence Requirements |
|---|---|---|---|
| Session ID | Unique client identifier | Client-generated UUID | Client-persistent |
| Sequence Number | Request ordering within session | Client state | Client-persistent |
| Last Applied | Highest applied sequence per session | Server state machine | Snapshot-included |
| Session Timeout | Cleanup threshold for old sessions | Server configuration | Not persisted |

### Multi-Raft and Sharding

Single Raft groups scale to thousands of operations per second but eventually hit fundamental limits. Multi-Raft architectures scale to millions of operations per second by partitioning data across multiple independent Raft groups.

#### Sharding Architecture Design

Multi-Raft systems partition data across multiple independent Raft groups, each handling a subset of the keyspace or logical partitions.

> **Decision: Sharding Strategy for Multi-Raft**
> - **Context**: Single Raft group limits throughput to ~10K ops/sec; applications need 100K-1M+ ops/sec
> - **Options Considered**:
>   1. Maintain single Raft for simplicity
>   2. Hash-based sharding with static assignment
>   3. Range-based sharding with dynamic rebalancing
>   4. Hybrid hash + range sharding
> - **Decision**: Hash-based sharding with static assignment initially, range-based migration later
> - **Rationale**: Hash sharding provides even distribution and simple implementation; range sharding enables better locality but requires complex rebalancing
> - **Consequences**: Enables 10-100x throughput scaling; requires cross-shard coordination for multi-key operations; adds significant operational complexity

**Multi-Raft Cluster Organization:**

| Component | Single Raft | Multi-Raft | Scaling Factor |
|---|---|---|---|
| Raft Groups | 1 group | N groups (typically 16-256) | Linear with N |
| Keyspace Partitioning | All keys in one group | Keys distributed across groups | Perfect parallelism |
| Cross-Shard Operations | Not applicable | Requires distributed transactions | Added complexity |
| Failure Blast Radius | Entire system | Single shard (1/N of data) | N-fold reduction |

**Shard Assignment and Routing:**

The multi-Raft system requires a routing layer that determines which Raft group handles each key. Hash-based sharding provides simple, deterministic routing:

```
shard_id = hash(key) % num_shards
raft_group = shard_map[shard_id]
```

| Routing Strategy | Pros | Cons | Best Use Case |
|---|---|---|---|
| Client-Side Routing | Low latency, no proxy overhead | Client complexity, harder updates | High-performance applications |
| Proxy-Based Routing | Simple clients, centralized logic | Additional hop, proxy bottleneck | Mixed client environments |
| Embedded Routing | Best performance, integrated logic | Library complexity, language binding | Native applications |

#### Cross-Shard Coordination

Multi-Raft systems require coordination mechanisms for operations that span multiple shards, such as distributed transactions or global secondary indexes.

**Distributed Transaction Protocol:**

Cross-shard transactions use a two-phase commit protocol coordinated by one of the participating Raft groups or a dedicated transaction coordinator.

| Transaction Phase | Coordinator Actions | Participant Actions | Safety Guarantees |
|---|---|---|---|
| Prepare | Send prepare to all shards | Acquire locks, validate, respond | Isolation maintained |
| Decision | Collect votes, decide commit/abort | Wait for decision | Atomicity ensured |
| Commit/Abort | Send decision to all participants | Apply/release based on decision | Durability/consistency maintained |
| Cleanup | Verify completion | Release resources | Resource cleanup |

**Global Secondary Index Management:**

Global indexes span multiple shards and require careful coordination to maintain consistency between primary data and index entries.

| Index Update Strategy | Consistency Level | Performance Impact | Implementation Complexity |
|---|---|---|---|
| Synchronous Updates | Strong consistency | High latency | Moderate |
| Asynchronous Updates | Eventual consistency | Low latency | High (requires conflict resolution) |
| Transactional Updates | Strong consistency | Medium latency | Very high |

#### Operational Considerations for Multi-Raft

Multi-Raft deployments introduce operational complexity that must be carefully managed through automation and monitoring.

**Shard Rebalancing and Migration:**

As data grows unevenly across shards or cluster topology changes, the system may need to migrate data between Raft groups.

| Rebalancing Trigger | Detection Method | Migration Strategy | Downtime Impact |
|---|---|---|---|
| Load Imbalance | Throughput monitoring | Live migration with dual-write | None |
| Storage Imbalance | Disk usage monitoring | Snapshot transfer | Brief read-only period |
| Topology Changes | Node addition/removal | Background resharding | None |
| Manual Rebalancing | Administrator command | Planned migration | Scheduled briefly |

**Multi-Raft Monitoring and Observability:**

Operating multiple Raft groups requires enhanced monitoring that tracks both individual group health and system-wide metrics.

| Monitoring Category | Individual Group Metrics | System-Wide Metrics | Alert Conditions |
|---|---|---|---|
| Performance | RPC latency, throughput | Aggregate throughput, P99 latency | Individual group degradation |
| Health | Leader stability, election frequency | Overall availability, cross-shard success rate | Group unavailability |
| Resource Usage | Memory per group, disk growth | Total resource consumption, imbalance | Resource exhaustion |
| Consistency | Replication lag, commit rates | Cross-shard transaction success | Prolonged inconsistency |

### Implementation Guidance

This section provides concrete guidance for implementing the performance optimizations and advanced features described above.

#### Technology Recommendations

| Component | Simple Option | Advanced Option | Production Recommendation |
|---|---|---|---|
| Serialization | JSON with msgpack fallback | Protocol Buffers | Protocol Buffers with JSON debug mode |
| Transport | HTTP/1.1 with connection pooling | gRPC with HTTP/2 | gRPC with TLS and connection pooling |
| Storage Engine | File-based with memory cache | Embedded database (RocksDB) | RocksDB with custom log format |
| Monitoring | Basic logging | Prometheus + Grafana | OpenTelemetry with multiple exporters |
| Load Balancing | Round-robin client-side | Hardware load balancer | Service mesh (Istio/Linkerd) |

#### Recommended Project Structure for Extensions

```
titan-extensions/
├── cmd/
│   ├── titan-server/         ← Enhanced server with optimizations
│   └── titan-benchmark/      ← Performance testing tools
├── pkg/
│   ├── consensus/           ← Core Raft from milestones 1-4
│   ├── transport/
│   │   ├── grpc/            ← gRPC transport implementation
│   │   └── http/            ← HTTP transport fallback
│   ├── serialization/
│   │   ├── protobuf/        ← Protocol Buffers implementation
│   │   └── json/            ← JSON fallback for debugging
│   ├── optimization/
│   │   ├── batching/        ← Request batching logic
│   │   ├── pipelining/      ← Pipeline replication
│   │   └── readindex/       ← ReadIndex implementation
│   ├── multiraft/
│   │   ├── coordinator/     ← Cross-shard coordination
│   │   ├── sharding/        ← Shard management
│   │   └── routing/         ← Request routing
│   └── monitoring/
│       ├── metrics/         ← Performance metrics collection
│       └── tracing/         ← Distributed tracing
├── proto/                   ← Protocol Buffer definitions
├── benchmarks/              ← Performance test suites
└── examples/
    ├── kv-store/           ← Key-value store using optimizations
    └── multi-shard/        ← Multi-Raft demonstration
```

#### Batching Infrastructure Implementation

**Complete Batching Manager:**

```python
from typing import List, Optional, Callable, Any
import asyncio
import time
from dataclasses import dataclass
from collections import deque
import threading

@dataclass
class BatchedRequest:
    """Individual request within a batch with completion tracking."""
    request_id: str
    data: bytes
    timestamp: float
    completion_future: asyncio.Future
    priority: int = 0  # Higher numbers = higher priority

@dataclass 
class BatchConfiguration:
    """Configuration parameters for adaptive batching."""
    max_batch_size: int = 50
    max_batch_delay_ms: float = 25.0
    enable_adaptive_sizing: bool = True
    adaptive_scaling_factor: float = 2.0
    priority_flush_threshold: int = 10

class AdaptiveBatchManager:
    """Manages request batching with adaptive sizing and priority handling."""
    
    def __init__(self, config: BatchConfiguration, flush_callback: Callable):
        self.config = config
        self.flush_callback = flush_callback
        self.pending_requests: deque[BatchedRequest] = deque()
        self.batch_timer: Optional[asyncio.Handle] = None
        self.lock = threading.Lock()
        self.current_batch_size = config.max_batch_size // 2  # Start conservative
        self.recent_throughput = 0.0
        self.last_flush_time = 0.0
        
    def add_request(self, request_id: str, data: bytes, priority: int = 0) -> asyncio.Future:
        """Add request to batch and return future for completion notification."""
        # TODO 1: Create BatchedRequest with current timestamp and completion future
        # TODO 2: Acquire lock and add to pending_requests deque
        # TODO 3: Check if immediate flush required (priority, size, or other triggers)
        # TODO 4: If not immediate flush, start or reset batch timer
        # TODO 5: Return completion future for caller to await
        pass
        
    def _should_flush_immediately(self) -> bool:
        """Determine if batch should be flushed without waiting for timer."""
        # TODO 1: Check if pending_requests length >= current_batch_size
        # TODO 2: Check if any pending request has priority >= priority_flush_threshold  
        # TODO 3: Check if oldest pending request exceeds max delay
        # TODO 4: Return True if any condition met, False otherwise
        pass
        
    async def _flush_batch(self) -> None:
        """Flush current batch and notify completion futures."""
        # TODO 1: Acquire lock and move all pending_requests to local batch list
        # TODO 2: Cancel existing batch_timer if set
        # TODO 3: If batch is empty, return early
        # TODO 4: Call flush_callback with batch data
        # TODO 5: Set completion results on all request futures
        # TODO 6: Update throughput metrics and adaptive sizing
        pass
        
    def _update_adaptive_sizing(self, batch_size: int, flush_latency: float) -> None:
        """Adjust batch size based on recent performance metrics."""
        # TODO 1: Calculate current throughput (requests per second)
        # TODO 2: Compare to recent_throughput to detect trends
        # TODO 3: Increase batch size if throughput improving and latency acceptable
        # TODO 4: Decrease batch size if latency too high or throughput declining  
        # TODO 5: Clamp batch size to configured min/max bounds
        pass
```

#### ReadIndex Implementation Skeleton

**ReadIndex Protocol Manager:**

```python
from typing import Dict, Optional, Set
import asyncio
import time
from dataclasses import dataclass
from enum import Enum

class ReadIndexState(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed" 
    EXPIRED = "expired"

@dataclass
class ReadIndexRequest:
    """Tracks read index request through confirmation process."""
    read_index: LogIndex
    request_time: float
    confirmation_future: asyncio.Future
    required_confirmations: int
    received_confirmations: Set[NodeId]
    state: ReadIndexState = ReadIndexState.PENDING

class ReadIndexManager:
    """Manages ReadIndex protocol for consistent follower reads."""
    
    def __init__(self, node_id: NodeId, consensus_engine):
        self.node_id = node_id
        self.consensus_engine = consensus_engine
        self.pending_read_indexes: Dict[LogIndex, ReadIndexRequest] = {}
        self.current_lease_expiry: Optional[float] = None
        self.lease_duration_ms: float = 500.0
        self.clock_skew_buffer_ms: float = 25.0
        
    async def request_consistent_read(self) -> LogIndex:
        """Request read index for consistent read on follower."""
        # TODO 1: Check if this node is leader with valid lease
        # TODO 2: If leader with lease, return current commit_index immediately
        # TODO 3: If follower, send ReadIndex request to current leader
        # TODO 4: Wait for confirmation that leader is still active
        # TODO 5: Return confirmed read index for state machine read
        pass
        
    async def handle_read_index_request(self, follower_id: NodeId) -> LogIndex:
        """Handle ReadIndex request from follower (leader only)."""
        # TODO 1: Verify this node is current leader
        # TODO 2: Get current commit_index as the read index
        # TODO 3: Send heartbeat to majority to confirm leadership
        # TODO 4: Wait for majority heartbeat responses
        # TODO 5: Return confirmed read index to follower
        pass
        
    def update_lease(self, heartbeat_responses: Set[NodeId]) -> None:
        """Update leader lease based on heartbeat responses."""
        # TODO 1: Check if received majority heartbeat responses
        # TODO 2: If majority, extend lease by lease_duration_ms
        # TODO 3: Account for clock_skew_buffer_ms in expiry calculation
        # TODO 4: Update current_lease_expiry timestamp
        pass
        
    def has_valid_lease(self) -> bool:
        """Check if current leader lease is still valid."""
        # TODO 1: Check if current_lease_expiry is set
        # TODO 2: Get current time and compare to lease expiry
        # TODO 3: Account for clock_skew_buffer_ms
        # TODO 4: Return True if lease valid, False otherwise
        pass
```

#### Multi-Raft Coordinator Implementation

**Shard Routing and Cross-Shard Coordination:**

```python
from typing import Dict, List, Set, Tuple, Optional
import hashlib
import asyncio
from dataclasses import dataclass
from enum import Enum

class TransactionState(Enum):
    PREPARING = "preparing"
    PREPARED = "prepared" 
    COMMITTING = "committing"
    COMMITTED = "committed"
    ABORTING = "aborting"
    ABORTED = "aborted"

@dataclass
class ShardOperation:
    """Single operation within a cross-shard transaction."""
    shard_id: int
    operation_type: str  # "read", "write", "delete"
    key: str
    value: Optional[bytes]
    expected_version: Optional[int] = None

@dataclass
class CrossShardTransaction:
    """Manages distributed transaction across multiple Raft shards."""
    transaction_id: str
    operations: List[ShardOperation]
    participating_shards: Set[int]
    coordinator_shard: int
    state: TransactionState
    prepare_responses: Dict[int, bool]
    start_time: float
    timeout_seconds: float = 30.0

class MultiRaftCoordinator:
    """Coordinates operations across multiple Raft groups."""
    
    def __init__(self, num_shards: int):
        self.num_shards = num_shards
        self.shard_leaders: Dict[int, NodeId] = {}
        self.raft_groups: Dict[int, Any] = {}  # shard_id -> RaftNode
        self.active_transactions: Dict[str, CrossShardTransaction] = {}
        
    def get_shard_for_key(self, key: str) -> int:
        """Determine which shard handles the given key."""
        # TODO 1: Hash the key using consistent hash function
        # TODO 2: Take modulo num_shards to get shard assignment
        # TODO 3: Return shard_id for routing
        pass
        
    async def execute_single_shard_operation(self, key: str, operation: str, value: Optional[bytes] = None) -> Any:
        """Execute operation on single shard (fast path)."""
        # TODO 1: Determine target shard using get_shard_for_key
        # TODO 2: Get current leader for target shard
        # TODO 3: Send operation directly to shard leader
        # TODO 4: Wait for response and return result
        pass
        
    async def execute_cross_shard_transaction(self, operations: List[ShardOperation]) -> bool:
        """Execute multi-shard transaction using two-phase commit."""
        # TODO 1: Generate unique transaction_id
        # TODO 2: Identify all participating shards from operations
        # TODO 3: Choose coordinator shard (typically first participating shard)
        # TODO 4: Create CrossShardTransaction object
        # TODO 5: Execute prepare phase across all shards
        # TODO 6: Collect prepare votes from all participants
        # TODO 7: Make commit/abort decision based on votes
        # TODO 8: Execute commit/abort phase across all shards
        # TODO 9: Clean up transaction state and return success/failure
        pass
        
    async def _execute_prepare_phase(self, transaction: CrossShardTransaction) -> bool:
        """Execute prepare phase of two-phase commit."""
        # TODO 1: Send prepare requests to all participating shards
        # TODO 2: Each shard validates operations and acquires locks
        # TODO 3: Wait for responses from all shards with timeout
        # TODO 4: Return True if all shards vote commit, False otherwise
        pass
        
    async def _execute_commit_phase(self, transaction: CrossShardTransaction, decision: bool) -> None:
        """Execute commit or abort phase of two-phase commit."""
        # TODO 1: Send commit/abort decision to all participating shards
        # TODO 2: Each shard applies or rolls back based on decision
        # TODO 3: Wait for acknowledgments from all shards
        # TODO 4: Handle any failures during commit phase
        # TODO 5: Clean up transaction state
        pass
        
    def add_shard(self, shard_id: int, raft_node) -> None:
        """Add new shard to multi-raft cluster."""
        # TODO 1: Register raft_node for shard_id
        # TODO 2: Update routing tables
        # TODO 3: Begin shard rebalancing if needed
        pass
        
    def remove_shard(self, shard_id: int) -> None:
        """Remove shard from multi-raft cluster.""" 
        # TODO 1: Drain operations from shard
        # TODO 2: Migrate data to other shards
        # TODO 3: Update routing tables
        # TODO 4: Clean up shard resources
        pass
```

#### Milestone Checkpoint: Performance Extensions

After implementing the performance optimizations, verify the enhancements:

**Batching Verification:**
1. Run benchmark: `python -m titan.benchmarks.batching_test --batch-sizes 1,10,50,100`
2. Expected behavior: Throughput should scale linearly with batch size up to optimal point
3. Verify adaptive sizing: Monitor batch size adjustments under varying load
4. Check latency impact: P99 latency should remain under 2x single-request latency

**ReadIndex Verification:**
1. Deploy 3-node cluster: `python -m titan.examples.readindex_demo`
2. Submit read requests to followers and measure latency
3. Expected behavior: Follower reads should complete without leader involvement after ReadIndex confirmation
4. Verify lease optimization: Leader reads should complete immediately during lease periods

**Multi-Raft Verification:**
1. Start multi-shard cluster: `python -m titan.examples.multiraft_demo --shards 4`
2. Submit single-shard and cross-shard transactions
3. Expected behavior: Single-shard ops complete in ~1 RTT; cross-shard ops complete in ~3 RTTs
4. Verify transaction isolation: Concurrent transactions should not interfere

**Performance Regression Testing:**
- Single-node throughput should improve by 10-100x with batching
- Read latency should decrease by 50-80% with ReadIndex
- Multi-shard throughput should scale linearly with shard count
- Memory usage should remain stable with optimizations enabled

#### Language-Specific Implementation Hints

**Python Performance Considerations:**
- Use `asyncio` for concurrent RPC handling rather than threading
- Implement batching with `collections.deque` for O(1) append/pop
- Use `uvloop` event loop for improved async performance
- Profile with `cProfile` and `py-spy` to identify bottlenecks

**Serialization Optimization:**
- Use `protobuf` Python bindings for message serialization
- Implement message pooling to reduce allocation overhead
- Consider `orjson` for JSON fallback serialization (faster than stdlib)

**Transport Layer Enhancement:**
- Use `grpcio` for HTTP/2 transport with connection pooling
- Implement retry logic with exponential backoff
- Add request/response compression for large payloads
- Use TLS for production deployments

**Memory Management:**
- Implement LRU cache for log entries using `functools.lru_cache`
- Use `__slots__` for dataclasses to reduce memory overhead
- Monitor memory usage with `tracemalloc` during development

These extensions transform Titan from a reference implementation into a production-ready consensus engine capable of handling demanding distributed systems workloads while maintaining the strong consistency guarantees that make Raft valuable.


## Glossary

> **Milestone(s):** All milestones (1-4) - provides comprehensive definitions for all technical terms used throughout election, replication, compaction, and membership changes

### Mental Model: Technical Dictionary

Think of this glossary as a technical dictionary specifically tailored for distributed consensus systems. Just as a medical dictionary defines terms differently than a general dictionary (where "pressure" has specific meaning related to blood flow rather than just "force applied"), this glossary provides precise definitions within the context of Raft consensus and distributed systems. Each term has been carefully chosen to maintain consistency throughout the Titan implementation and aligns with established academic literature while remaining accessible to practitioners.

The glossary serves multiple purposes: it ensures consistent terminology across all documentation and code, provides quick reference during development, clarifies subtle distinctions between similar concepts, and establishes a shared vocabulary for team communication. When debugging complex distributed behaviors, having precise terminology prevents miscommunication that could lead to incorrect analysis.

### Core Consensus Terms

| Term | Definition | Context | Related Terms |
|------|------------|---------|---------------|
| **consensus** | Agreement among distributed nodes on a single value or sequence of values, even in the presence of failures | Fundamental goal of Raft algorithm | quorum, agreement, coordination |
| **quorum** | Majority of nodes (⌊N/2⌋ + 1) required to make binding decisions in a cluster of N nodes | Essential for preventing split-brain scenarios | majority, voting, decision threshold |
| **split-brain** | Dangerous condition where multiple nodes simultaneously believe they are the leader | Primary safety violation Raft prevents | partition, multiple leaders, safety violation |
| **linearizability** | Consistency model where operations appear to execute atomically at some point between start and completion | Strongest consistency guarantee Raft provides | sequential consistency, atomicity, ordering |
| **state machine replication** | Technique where identical state machines on multiple nodes execute same sequence of operations | Core abstraction Raft implements | deterministic execution, replicated state |
| **partition tolerance** | System's ability to continue operating correctly despite network communication failures | One of CAP theorem properties Raft optimizes for | network partition, availability, CAP theorem |

### Raft-Specific Terms

| Term | Definition | Context | Related Terms |
|------|------------|---------|---------------|
| **term** | Logical clock value that increases monotonically and identifies leadership epochs | Fundamental ordering mechanism in Raft | election epoch, leadership period, logical time |
| **log matching property** | Safety guarantee that if two logs have entries at same index with same term, all preceding entries are identical | Critical invariant for consistency | log consistency, safety property, prefix matching |
| **joint consensus** | Two-phase protocol for safely changing cluster membership without losing availability | Enables safe dynamic reconfiguration | membership change, configuration transition, dual majority |
| **dual majority** | Requirement for majorities from both old and new configurations during joint consensus | Prevents split-brain during membership changes | joint configuration, safety during transition |
| **heartbeat** | Periodic `AppendEntries` message with empty entries sent by leader to maintain authority | Prevents unnecessary elections | leadership assertion, failure detection, liveness |
| **election timeout** | Randomized period after which follower becomes candidate if no heartbeat received | Triggers leader election process | failure detection, randomization, liveness |
| **split vote** | Election outcome where no candidate receives majority due to vote distribution | Resolved by randomized timeouts in subsequent elections | election failure, vote fragmentation |

### Node States and Roles

| Term | Definition | Context | Related Terms |
|------|------------|---------|---------------|
| **follower** | Node state where node accepts commands from leader and participates in elections | Default state for most nodes most of the time | passive replica, vote participant |
| **candidate** | Transitional node state during election when requesting votes from peers | Active during leader election process | election participant, vote requester |
| **leader** | Node state with authority to accept client requests and coordinate log replication | Unique per term, coordinates all cluster activity | primary replica, coordinator |
| **persistent state** | Node data that survives crashes and must be stored on stable storage | Critical for safety across failures | durable state, crash recovery |
| **volatile state** | Node data that can be reconstructed after restart and stored in memory | Performance optimization, rebuilt during recovery | transient state, reconstructible data |

### Log Management Terms

| Term | Definition | Context | Related Terms |
|------|------------|---------|---------------|
| **log entry** | Individual record containing client command, term, and index | Basic unit of replication in Raft | command record, replicated operation |
| **log index** | Monotonically increasing position identifier for log entries | Orders operations and enables consistency checks | position, sequence number, ordering |
| **commit index** | Highest log index known to be replicated on majority of nodes | Determines which entries are safe to apply | safety threshold, replication confirmation |
| **log compaction** | Process of removing old log entries using snapshots to prevent unbounded growth | Essential for long-running systems | snapshot creation, space management |
| **snapshot** | Point-in-time capture of complete state machine state | Enables log truncation and fast node recovery | state checkpoint, compacted history |
| **chunked transfer** | Technique for sending large snapshots in small pieces over RPC | Handles network limitations and memory constraints | incremental transfer, streaming |
| **log truncation** | Safe removal of log entries that are covered by snapshots | Reclaims storage space after compaction | space reclamation, history compression |

### Network and Communication Terms

| Term | Definition | Context | Related Terms |
|------|------------|---------|---------------|
| **RPC (Remote Procedure Call)** | Network communication mechanism for inter-node messages | Foundation of all Raft node communication | network protocol, message passing |
| **AppendEntries** | RPC for log replication and heartbeats from leader to followers | Primary replication mechanism | log synchronization, heartbeat |
| **RequestVote** | RPC for candidates to request votes during leader election | Election communication protocol | vote solicitation, election message |
| **InstallSnapshot** | RPC for transferring snapshots from leader to lagging followers | Recovery mechanism for slow nodes | snapshot transfer, catch-up protocol |
| **network partition** | Communication failure that splits cluster into isolated groups | Primary failure mode Raft must handle | split network, communication failure |
| **failure detection** | Process of identifying node crashes or communication failures | Triggers recovery and election processes | liveness monitoring, crash detection |

### Membership and Configuration Terms

| Term | Definition | Context | Related Terms |
|------|------------|---------|---------------|
| **cluster configuration** | Set of nodes that participate in consensus decisions | Defines voting membership and communication targets | membership set, participant list |
| **configuration change** | Process of adding or removing nodes from cluster membership | Enables dynamic scaling and maintenance | membership modification, reconfiguration |
| **catch-up process** | Procedure for bringing new nodes to current state before voting promotion | Prevents performance degradation from slow nodes | node preparation, state synchronization |
| **disruptive server** | Functional node that negatively impacts cluster availability despite correct operation | Typically slow nodes that delay consensus | performance degradation, availability impact |
| **node promotion** | Process of granting voting rights to a new node after catch-up completion | Final step in adding new cluster members | membership activation, voting enablement |

### Safety and Correctness Terms

| Term | Definition | Context | Related Terms |
|------|------------|---------|---------------|
| **safety property** | Guarantee that something bad never happens during system execution | Fundamental correctness requirement | invariant, never property |
| **liveness property** | Guarantee that something good eventually happens during system execution | Progress and availability requirement | eventually property, progress |
| **election safety** | Property ensuring at most one leader exists per term | Core safety guarantee of Raft | single leadership, term uniqueness |
| **leader completeness** | Property ensuring leaders contain all committed entries from previous terms | Prevents data loss during leadership changes | commit preservation, history completeness |
| **state machine safety** | Property ensuring all nodes apply same sequence of operations | Guarantees replica consistency | deterministic execution, consistency |
| **commit safety** | Property ensuring committed entries never change or disappear | Durability guarantee for client operations | immutability, persistence guarantee |

### Testing and Verification Terms

| Term | Definition | Context | Related Terms |
|------|------------|---------|---------------|
| **property-based testing** | Testing approach that verifies invariants and safety properties rather than specific scenarios | Essential for distributed systems validation | invariant checking, generative testing |
| **chaos testing** | Testing methodology that simulates network failures and node crashes | Validates robustness under adverse conditions | fault injection, stress testing |
| **deterministic replay** | Technique for reproducing exact distributed execution scenarios | Critical for debugging distributed races | execution reproduction, debugging aid |
| **milestone checkpoints** | Verification criteria and expected behavior after each implementation milestone | Structured validation approach | progress validation, implementation gates |
| **consistency checking** | Process of validating distributed system invariants across all nodes | Ensures correctness properties hold | invariant validation, correctness verification |

### Performance and Optimization Terms

| Term | Definition | Context | Related Terms |
|------|------------|---------|---------------|
| **batching** | Collecting multiple requests before processing to reduce overhead | Performance optimization technique | request aggregation, throughput optimization |
| **pipelining** | Sending multiple requests without waiting for responses | Reduces latency in request processing | parallel processing, latency optimization |
| **adaptive timeout** | Dynamic timeout adjustment based on observed network conditions | Improves election stability and performance | dynamic configuration, self-tuning |
| **coordination overhead** | Cost of achieving distributed agreement in terms of time and resources | Performance consideration in consensus protocols | agreement cost, distributed overhead |
| **throughput** | Number of operations the consensus system can process per unit time | Key performance metric | ops per second, processing rate |
| **latency** | Time required to complete individual consensus operations | Critical performance characteristic | response time, operation duration |

### Advanced Features Terms

| Term | Definition | Context | Related Terms |
|------|------------|---------|---------------|
| **ReadIndex** | Protocol allowing consistent reads from followers without going through leader | Performance optimization for read-heavy workloads | consistent reads, follower reads |
| **lease-based reads** | Local leader reads during periods of guaranteed leadership | Optimization that trades safety for performance | leader lease, read optimization |
| **multi-Raft** | Architecture using multiple independent Raft groups for horizontal scaling | Scaling technique beyond single consensus group | sharding, horizontal scaling |
| **cross-shard coordination** | Managing operations that span multiple Raft groups | Distributed transaction coordination | multi-group operations, distributed coordination |
| **adaptive batching** | Dynamic adjustment of batch size based on current load conditions | Self-tuning performance optimization | dynamic batching, load-responsive optimization |

### Error and Failure Terms

| Term | Definition | Context | Related Terms |
|------|------------|---------|---------------|
| **cascading failure** | Single failure that triggers multiple additional failures | Dangerous failure mode to prevent | failure propagation, systemic failure |
| **thundering herd** | Problem where multiple nodes simultaneously start elections | Performance issue solved by randomized timeouts | simultaneous election, election storm |
| **term confusion** | Inconsistent views of current term across cluster nodes | Common bug in election implementation | term synchronization, state inconsistency |
| **log inconsistency** | Different entries appearing at same index on different nodes | Violation of log matching property | consistency violation, safety breach |
| **snapshot corruption** | Invalid or incomplete snapshot data that cannot restore state | Data integrity failure in compaction | data corruption, integrity violation |
| **split-brain prevention** | Mechanisms ensuring single leader guarantee is maintained | Core safety mechanism | leadership uniqueness, safety enforcement |

### Data Structure and Implementation Terms

| Term | Definition | Context | Related Terms |
|------|------------|---------|---------------|
| **NodeId** | String identifier uniquely identifying each cluster node | Basic addressing mechanism | node identifier, cluster addressing |
| **Term** | Integer representing election term for logical time ordering | Fundamental ordering mechanism | election epoch, logical clock |
| **LogIndex** | Integer position identifier for log entries | Entry ordering and indexing | position counter, sequence number |
| **LogEntry** | Data structure containing term, index, data, and timestamp | Basic replication unit | command record, operation entry |
| **NodeState** | Enumeration of possible node states (FOLLOWER, CANDIDATE, LEADER) | State machine implementation | role enumeration, state tracking |
| **ClusterConfiguration** | Data structure defining current cluster membership | Configuration management | membership definition, cluster state |
| **SnapshotMetadata** | Data structure containing snapshot metadata and validation information | Snapshot management | checkpoint metadata, snapshot info |

### Constants and Configuration Terms

| Term | Definition | Value | Purpose |
|------|------------|-------|---------|
| **ELECTION_TIMEOUT_MIN** | Minimum election timeout duration | 150ms | Prevents excessive election frequency |
| **ELECTION_TIMEOUT_MAX** | Maximum election timeout duration | 300ms | Ensures reasonable failure detection |
| **HEARTBEAT_INTERVAL** | Frequency of leader heartbeat messages | 50ms | Maintains leadership authority |
| **MAX_CLUSTER_SIZE** | Maximum recommended cluster size | 9 nodes | Limits coordination overhead |
| **SNAPSHOT_THRESHOLD** | Log entries count triggering snapshot creation | 10000 entries | Prevents unbounded log growth |
| **CHUNK_SIZE** | Size of snapshot transfer chunks | 64KB | Balances memory and network efficiency |
| **JOINT_CONSENSUS_TIMEOUT** | Maximum time allowed in joint consensus state | 60 seconds | Prevents indefinite transition states |
| **CATCHUP_TIMEOUT** | Maximum time for new node preparation | 30 seconds | Limits impact on cluster availability |

### Logging and Debugging Terms

| Term | Definition | Context | Related Terms |
|------|------------|---------|---------------|
| **structured logging** | Consistent format for log messages across all components | Enables automated log analysis | formatted logging, parseable logs |
| **log level** | Severity classification for log messages (TRACE, DEBUG, INFO, WARN, ERROR) | Controls logging verbosity | severity level, logging granularity |
| **performance profiling** | Measuring timing and resource usage of consensus operations | Optimization and debugging tool | timing analysis, resource monitoring |
| **state inspection** | Examining internal node state for debugging purposes | Debugging and validation technique | state examination, diagnostic tool |
| **trace correlation** | Linking related log messages across distributed operations | Debugging distributed interactions | operation tracking, distributed tracing |

### Implementation Guidance

Understanding distributed consensus terminology is critical for successful implementation. The terms defined above form the foundation of all technical discussions, code comments, documentation, and debugging sessions. Consistent use of these terms prevents miscommunication and ensures that all team members share the same mental models.

#### Technology Recommendations for Terminology Management

| Aspect | Simple Option | Advanced Option |
|--------|---------------|-----------------|
| Code Documentation | Inline comments with term definitions | Automated documentation with glossary links |
| API Documentation | Manual term explanations in docstrings | OpenAPI with terminology annotations |
| Logging | Hard-coded term usage in log messages | Structured logging with term validation |
| Testing | Manual test naming with terminology | Property-based tests that validate term usage |

#### Recommended File Structure

```
project-root/
  docs/
    glossary.md                    ← this document
    terminology-guide.md           ← usage guidelines
  internal/types/
    consensus_types.go             ← core type definitions
    constants.go                   ← all consensus constants
  internal/logging/
    structured_logger.go           ← terminology-aware logging
  scripts/
    validate_terminology.py       ← check consistent term usage
```

#### Core Type Definitions

```python
from enum import Enum
from typing import Dict, List, Optional, Set, Union
from dataclasses import dataclass

# Core identifier types
NodeId = str
Term = int
LogIndex = int

class NodeState(Enum):
    """Node role in Raft consensus protocol."""
    FOLLOWER = "follower"
    CANDIDATE = "candidate" 
    LEADER = "leader"

class LogLevel(Enum):
    """Logging severity levels for structured logging."""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"

# TODO: Add your consensus-specific enums here
# TODO: Define ConfigurationState enum for membership changes
# TODO: Define ReadIndexState enum for follower reads
# TODO: Define TransactionState enum for cross-shard operations
```

#### Terminology Validation Utilities

```python
import re
from typing import Set

class TerminologyValidator:
    """Validates consistent terminology usage across codebase."""
    
    # Preferred terms mapping deprecated/alternate terms to preferred ones
    PREFERRED_TERMS = {
        "agreement": "consensus",
        "majority": "quorum", 
        "epoch": "term",
        "position": "index",
        "checkpoint": "snapshot",
        "configuration": "cluster_configuration"
    }
    
    def validate_code_terminology(self, file_content: str) -> List[str]:
        """Check code for consistent terminology usage."""
        # TODO: Scan for deprecated terms in comments and variable names
        # TODO: Validate that constants use exact naming conventions
        # TODO: Check that log messages use structured terminology
        # TODO: Report inconsistencies with file and line numbers
        pass
    
    def validate_documentation_terminology(self, doc_content: str) -> List[str]:
        """Check documentation for consistent terminology usage."""
        # TODO: Ensure technical terms match glossary definitions
        # TODO: Validate that analogies are used consistently
        # TODO: Check that cross-references use exact term names
        pass
```

#### Logging Integration

```python
import logging
from datetime import datetime
from typing import Any, Dict

@dataclass
class LogContext:
    """Structured context for consensus logging."""
    node_id: str
    term: int
    component: str
    timestamp: float

class ConsensusLogger:
    """Logger with built-in consensus terminology."""
    
    def __init__(self, node_id: NodeId):
        self.node_id = node_id
        self.logger = logging.getLogger(f"consensus.{node_id}")
    
    def log_election_event(self, event_type: str, term: Term, **kwargs) -> None:
        """Log election events with consistent terminology."""
        # TODO: Structure election event with standard fields
        # TODO: Include term, candidate_id, vote_granted fields
        # TODO: Ensure event_type uses standard election terminology
        # TODO: Add correlation IDs for tracking election sequences
        pass
    
    def log_replication_event(self, event_type: str, term: Term, **kwargs) -> None:
        """Log replication events with consistent terminology."""
        # TODO: Structure replication event with log_index, entries_count
        # TODO: Include leader_id, follower_id for tracking flows
        # TODO: Use standard replication terminology (append, commit, apply)
        # TODO: Track replication latency and success rates
        pass
```

#### Milestone Checkpoints

**Milestone 1 Checkpoint: Election Terminology**
After implementing election logic, verify terminology usage:
- Run `python scripts/validate_terminology.py --milestone=election`
- Check that election logs use terms: "candidate", "follower", "leader", "term", "vote_granted", "split_vote"
- Verify no deprecated terms like "epoch" or "primary" appear in election code
- Confirm election timeout constants use exact naming conventions

**Milestone 2 Checkpoint: Replication Terminology**
After implementing log replication, verify:
- Replication logs use terms: "log_index", "commit_index", "log_matching_property", "append_entries"
- Safety property terminology appears in consistency checks
- No conflicting terms like "position" instead of "index"

**Milestone 3 Checkpoint: Compaction Terminology**
After implementing log compaction, verify:
- Snapshot logs use terms: "snapshot", "log_compaction", "chunked_transfer", "log_truncation"
- No deprecated terms like "checkpoint" or "backup" in compaction code
- InstallSnapshot RPC uses exact field naming from conventions

**Milestone 4 Checkpoint: Membership Terminology**
After implementing membership changes, verify:
- Configuration logs use terms: "joint_consensus", "dual_majority", "cluster_configuration", "disruptive_server"
- No ambiguous terms like "reconfiguration" or "cluster_change"
- Membership transition logs clearly distinguish old vs new configurations

#### Debugging with Terminology

When debugging consensus issues, precise terminology becomes critical:

| Symptom | Terminology Confusion | Correct Analysis |
|---------|----------------------|------------------|
| "Node won't become primary" | Using "primary" instead of "leader" | "Node won't become leader - check election timeout and term advancement" |
| "Log positions don't match" | Using "position" instead of "index" | "Log indices don't match - verify log matching property implementation" |
| "Cluster reconfiguration hangs" | Using "reconfiguration" instead of "membership change" | "Membership change hangs - check joint consensus state and dual majority" |
| "Checkpoints are corrupted" | Using "checkpoint" instead of "snapshot" | "Snapshots are corrupted - verify snapshot metadata and chunked transfer" |

#### Language-Specific Hints

**Python-Specific Terminology Integration:**
- Use `typing.NewType` to create semantic type aliases: `NodeId = NewType('NodeId', str)`
- Leverage `Enum` classes for all state enumerations with descriptive string values
- Use dataclasses with `__post_init__` for term validation
- Implement `__str__` methods that use consistent terminology in debugging output

**Documentation Integration:**
- Include terminology validation in CI/CD pipeline
- Use doc generation tools that link technical terms to glossary entries
- Maintain terminology changelog when evolving definitions
- Create IDE plugins that suggest preferred terms over deprecated alternatives

# AUDIT & FIX: build-kafka

## CRITIQUE
- **Logical Gap (Confirmed - Metadata/Coordinator):** The original has no milestone for cluster metadata management. In real Kafka, ZooKeeper or KRaft manages broker discovery, topic metadata, controller election, and partition leader assignment. Without a coordinator, there's no way to know which broker owns which partition, and leader election has no mechanism to execute. This is architecturally fundamental.
- **Technical Inaccuracy (Confirmed - High Watermark):** The original states the high watermark advances 'after all ISR members acknowledge the offset.' This is imprecise. The leader tracks the fetch positions of all ISR followers and advances the high watermark to the largest offset that is present on ALL ISR replicas. Followers don't 'acknowledge'—they fetch, and the leader observes their fetch offset.
- **M1 Mixing Concerns:** Consumer offset tracking is listed as an AC in M1 (Topic and Partitions) but conceptually belongs in M3 (Consumer Groups). Offset management is a consumer-side concern, not a topic/partition concern.
- **M2 Producer:** 'Idempotent producer assigns sequence numbers' is marked as '(optional)' but is a critical feature for exactly-once semantics. It should at least be a clearly defined stretch goal with its own AC.
- **Missing: Segment Management:** Log segments (rolling, retention, compaction) are not addressed. A partition is not a single file—it's a sequence of segment files with time/size-based rolling. Without this, the log grows unbounded.
- **M4 Replication:** 'Leader election selects a new leader from the current ISR set within seconds'—this assumes ISR is non-empty. What happens when ISR is empty? Unclean leader election (data loss) vs availability tradeoff must be addressed.
- **Estimated Hours:** 60-100 is too wide. Adding metadata coordinator and log segment management pushes to ~110 hours.

## FIXED YAML
```yaml
id: build-kafka
name: Build Your Own Kafka
description: >-
  Distributed append-only log with topic partitioning, producer batching,
  consumer group coordination, leader-follower replication, and cluster
  metadata management.
difficulty: expert
estimated_hours: 110
essence: >-
  Append-only segmented log per partition with deterministic key-based routing,
  a metadata controller managing topic/partition assignments and leader election,
  leader-follower replication with in-sync replica tracking and high watermark
  advancement, and consumer group coordination with partition rebalancing for
  parallel ordered message processing.
why_important: >-
  Building this teaches foundational distributed systems patterns used in
  production data pipelines—log-structured storage, replication consensus,
  consumer coordination, and failure recovery—skills that directly translate
  to designing high-throughput, fault-tolerant backend systems.
learning_outcomes:
  - Implement topic partitioning with segmented append-only log storage
  - Build a metadata controller managing broker registration, topic creation, and leader election
  - Design producer batching with configurable acknowledgment levels (acks=0,1,all)
  - Implement consumer group coordination with partition assignment and rebalancing
  - Build leader-follower replication with ISR tracking and high watermark
  - Implement log segment management with rolling, retention, and compaction
  - Handle producer retries with idempotent sequence numbers
  - Debug network partition scenarios and leader failover
skills:
  - Distributed consensus
  - Log-structured storage
  - Partition replication
  - Consumer coordination
  - Leader election
  - Offset management
  - Binary protocol design
  - Segment management
tags:
  - build-from-scratch
  - distributed
  - expert
  - message-queue
  - partitions
  - replication
  - streaming
architecture_doc: architecture-docs/build-kafka/index.md
languages:
  recommended:
    - Go
    - Java
    - Rust
  also_possible:
    - Scala
    - C++
resources:
  - type: book
    name: "Kafka: The Definitive Guide"
    url: https://www.confluent.io/resources/kafka-the-definitive-guide/
  - type: paper
    name: "Kafka: a Distributed Messaging System"
    url: https://www.microsoft.com/en-us/research/wp-content/uploads/2017/09/Kafka.pdf
  - type: documentation
    name: Apache Kafka Protocol Guide
    url: https://kafka.apache.org/protocol
  - type: documentation
    name: KRaft (KIP-500) Design
    url: https://cwiki.apache.org/confluence/display/KAFKA/KIP-500
prerequisites:
  - type: skill
    name: Networking and TCP programming
  - type: skill
    name: Distributed systems fundamentals
  - type: skill
    name: Binary protocol design
  - type: skill
    name: File I/O and log-structured storage
milestones:
  - id: build-kafka-m1
    name: Log Storage & Partitions
    description: >-
      Implement the core storage layer: topics with multiple partitions, each
      partition backed by a segmented append-only log with index files.
    acceptance_criteria:
      - Topic creation allocates a configurable number of partitions, each initialized as an empty segmented log
      - Each partition is a directory containing log segment files; each segment stores a contiguous range of offsets
      - Messages appended to a partition receive monotonically increasing offsets starting from 0
      - Log segments roll to a new file when the active segment exceeds a configurable size (default 1GB) or age (default 7 days)
      - Offset index file maps message offsets to file positions within the segment for O(1) random access by offset
      - Timestamp index maps timestamps to offsets for time-based lookups
      - Read by offset returns the message at the specified offset, or the next available offset if the exact offset has been compacted/deleted
      - Message format includes key (nullable), value, timestamp, headers (key-value pairs), and offset
    pitfalls:
      - Treating a partition as a single file makes it impossible to implement retention and compaction—segment from day one
      - Offset gaps can occur after compaction; consumers must handle sparse offsets gracefully
      - Null keys are valid—use round-robin or sticky partition assignment, not hash routing
      - Index files must be memory-mapped for fast offset lookups; linear scanning the log is O(n)
      - Segment rolling during active writes must be atomic—no message should span two segments
    concepts:
      - Log segment is a contiguous file containing a range of sequential messages
      - Offset index enables O(1) lookup of a message by its offset within a segment
      - Segment rolling creates a new active segment when the current one reaches capacity
      - Message key determines partition routing (hash-based) for ordering guarantees
    skills:
      - Log-structured storage
      - Segment file management
      - Index file design
      - Memory-mapped file I/O
    deliverables:
      - Topic and partition creation with configurable partition count
      - Segmented append-only log per partition
      - Offset and timestamp index files per segment
      - Segment rolling based on size and age thresholds
      - Message format with key, value, timestamp, headers, and offset
      - Read-by-offset using index for fast lookup
    estimated_hours: 16

  - id: build-kafka-m2
    name: Metadata Controller
    description: >-
      Implement a metadata controller (inspired by KRaft) that manages broker
      registration, topic metadata, partition-to-broker assignment, and
      leader election.
    acceptance_criteria:
      - Brokers register with the controller on startup, reporting their ID, host, port, and available capacity
      - Controller detects broker failures via missed heartbeats (configurable timeout, default 10s) and marks them as dead
      - Topic creation request handled by controller, which assigns partition leaders and replicas across available brokers
      - Partition leader assignment distributes leadership evenly across brokers (verified by measuring leader count per broker)
      - Controller triggers leader election for partitions whose leader broker has failed, selecting a new leader from the ISR
      - Metadata API returns current topic list, partition assignments, and leader locations for client discovery
      - Controller state is persisted to a metadata log for crash recovery (controller can reconstruct state on restart)
    pitfalls:
      - Single controller is a single point of failure—document the limitation and plan for controller failover as a stretch goal
      - Broker heartbeat timeout too aggressive causes false failure detection during GC pauses or network blips
      - Leader assignment without rack-awareness concentrates replicas on the same failure domain
      - Metadata log must be fsynced before acknowledging state changes to prevent loss on controller crash
      - Race condition between broker failure detection and leader election can cause brief unavailability—document the window
    concepts:
      - Metadata controller is the centralized authority for cluster topology
      - Broker registration and heartbeat-based failure detection
      - Partition leader assignment distributes load across brokers
      - Leader election selects a new leader from the in-sync replica set
      - Metadata log provides crash recovery for controller state
    skills:
      - Cluster coordination
      - Failure detection
      - Leader election algorithms
      - Metadata persistence
    deliverables:
      - Broker registration API with heartbeat-based liveness tracking
      - Topic creation with partition-to-broker assignment
      - Leader election triggered on broker failure
      - Metadata API returning topic, partition, and leader information
      - Metadata log for controller crash recovery
      - Dead broker detection and partition reassignment
    estimated_hours: 18

  - id: build-kafka-m3
    name: Producer
    description: >-
      Implement the producer client with partition routing, batching,
      configurable acknowledgments, and retry logic.
    acceptance_criteria:
      - Message key hashing (murmur2 or CRC32) consistently routes messages with the same key to the same partition
      - Null-key messages are distributed across partitions via round-robin or sticky partitioner
      - Batch accumulator groups messages by target partition and flushes when batch size or linger time threshold is reached
      - Configurable acks: 0 (fire-and-forget, no waiting), 1 (leader acknowledges), all (all ISR members acknowledge)
      - Retry logic re-sends failed produce requests with exponential backoff up to configurable max retries (default 3)
      - Idempotent producer mode assigns per-partition sequence numbers; broker deduplicates retries by rejecting out-of-sequence messages
      - Producer discovers partition leaders via metadata API and refreshes on LEADER_NOT_AVAILABLE error
    pitfalls:
      - Batch timeout (linger.ms) too high adds latency; too low defeats batching—tune based on workload
      - Leader failover during produce causes temporary errors; retry logic must refresh metadata and find the new leader
      - Without idempotent mode, retries after network timeout cause duplicate messages (exactly-once requires idempotent producer)
      - Batch compression (snappy, lz4) saves network bandwidth but adds CPU overhead—make it configurable
      - Fire-and-forget (acks=0) provides no delivery guarantee; document the tradeoff clearly
    concepts:
      - Partition routing determines which partition receives each message
      - Batch accumulation amortizes network overhead across multiple messages
      - Acknowledgment levels trade durability guarantees for throughput
      - Idempotent producer uses sequence numbers for exactly-once at-partition level
    skills:
      - Partition routing algorithms
      - Batch buffering and flushing
      - Network protocol design
      - Retry logic with backoff
      - Idempotent message deduplication
    deliverables:
      - Key-based partition routing with consistent hashing
      - Null-key round-robin/sticky partition assignment
      - Batch accumulator with size and time-based flushing
      - Configurable acks (0, 1, all) with appropriate waiting logic
      - Retry with exponential backoff and metadata refresh
      - Idempotent producer mode with sequence number tracking
    estimated_hours: 16

  - id: build-kafka-m4
    name: Consumer Groups
    description: >-
      Implement consumer group protocol with membership management, partition
      assignment, offset commits, and rebalancing.
    acceptance_criteria:
      - Consumer joins a group by sending JoinGroup request to the group coordinator (a designated broker)
      - Group coordinator selects a group leader who computes and distributes partition assignments to all members
      - Partition assignment strategies: range (contiguous partition ranges per consumer) and round-robin (interleaved assignment)
      - Offset commit persists the consumer's last processed offset per partition to the broker (internal __consumer_offsets topic or equivalent)
      - Offset fetch returns the last committed offset for each partition, enabling resume-on-restart without reprocessing
      - Rebalancing triggers when a consumer joins, leaves (explicit or heartbeat timeout), or topic partition count changes
      - During rebalancing, all consumers pause processing, revoke current assignments, and receive new assignments
      - Session timeout (default 10s) and heartbeat interval (default 3s) are configurable
    pitfalls:
      - Rebalance storms occur when consumers restart in rapid succession—use sticky assignment to minimize partition movement
      - Committing offsets before processing causes data loss; committing after processing causes duplicates on failure—document the tradeoff
      - Stuck rebalances from unresponsive consumers block the entire group—enforce max.poll.interval and remove slow consumers
      - Duplicate processing is inherent during rebalancing; consumers must be designed for at-least-once semantics
      - Offset storage must be durable; losing committed offsets causes full reprocessing
    concepts:
      - Consumer group distributes partition ownership across consumers for parallel processing
      - Group coordinator manages membership and triggers rebalancing
      - Partition assignment strategies determine which consumer reads which partitions
      - Offset commit tracks consumer progress for resume-on-restart
      - Rebalancing protocol redistributes partitions on membership changes
    skills:
      - Distributed coordination protocols
      - Membership management
      - Partition assignment algorithms
      - Offset management
      - Failure detection
    deliverables:
      - JoinGroup/SyncGroup/LeaveGroup/Heartbeat protocol messages
      - Group coordinator assigning a leader and managing membership state
      - Range and round-robin partition assignment strategies
      - Offset commit and fetch operations with durable storage
      - Rebalancing triggered by membership changes with partition revocation and reassignment
      - Session timeout and heartbeat configuration
    estimated_hours: 22

  - id: build-kafka-m5
    name: Replication
    description: >-
      Implement leader-follower partition replication with ISR tracking,
      high watermark, and leader failover.
    acceptance_criteria:
      - Each partition has a leader replica (serves reads and writes) and N-1 follower replicas (replicate from leader)
      - Followers continuously fetch new records from the leader using a fetch protocol that reports the follower's current offset
      - Leader tracks each follower's fetch offset; followers whose offset is within the configured lag threshold are in the ISR
      - Followers falling behind the lag threshold are removed from ISR; they rejoin ISR when caught up
      - High watermark (HW) is the largest offset that exists on ALL ISR replicas; the leader computes HW based on ISR fetch positions
      - Consumers can only read up to the high watermark—messages above HW are uncommitted and not visible
      - Leader failure triggers election of a new leader from the current ISR (clean election); unclean election from non-ISR replicas is configurable and off by default
      - acks=all requires ALL ISR members to have replicated the message before the produce is acknowledged
      - min.insync.replicas setting rejects writes when the ISR size drops below the configured threshold
    pitfalls:
      - High watermark is computed by the LEADER based on ISR fetch positions, not by followers acknowledging—the original description was inaccurate
      - ISR shrinking to empty makes the partition unavailable for writes (unless unclean leader election is enabled, accepting data loss)
      - Unclean leader election (electing a non-ISR replica) can lose committed messages—this must be a conscious configuration choice
      - Follower fetch protocol must handle leader changes gracefully by detecting epoch/leader changes and re-fetching from new leader
      - Data loss window: if leader crashes after local write but before ISR replication, messages between HW and leader's log end are lost
    concepts:
      - Leader handles all reads and writes; followers replicate for fault tolerance
      - In-Sync Replica (ISR) set contains followers that are fully caught up with the leader
      - High watermark is the committed offset boundary visible to consumers
      - Clean leader election from ISR guarantees no committed data loss
      - Unclean leader election from non-ISR trades data loss for availability
    skills:
      - Replication protocol design
      - ISR management
      - High watermark computation
      - Leader election
      - Failure recovery
    deliverables:
      - Leader-follower replication with continuous follower fetch
      - ISR tracking based on follower lag threshold
      - High watermark computation and advancement by leader
      - Consumer read restriction to high watermark offset
      - Leader election from ISR on leader failure
      - min.insync.replicas enforcement on produce
      - Configurable unclean leader election (off by default)
    estimated_hours: 26

  - id: build-kafka-m6
    name: Log Retention & Compaction
    description: >-
      Implement log segment lifecycle management with time/size-based retention
      and key-based log compaction.
    acceptance_criteria:
      - Time-based retention deletes log segments older than a configurable retention period (default 7 days)
      - Size-based retention deletes oldest segments when total partition log size exceeds configured limit
      - Retention runs as a background task at configurable intervals without blocking writes
      - Log compaction (for compacted topics) retains only the latest value for each key, removing older duplicates
      - Compaction produces new segments with tombstones (null-value records) for deleted keys, retained for a configurable tombstone retention period
      - Active segment (currently being written to) is never eligible for retention or compaction
      - Consumers reading from offsets within a deleted segment receive an error directing them to the earliest available offset
    pitfalls:
      - Deleting segments that consumers haven't finished reading causes data loss from the consumer's perspective—track consumer lag
      - Log compaction with very large key spaces is memory-intensive (offset map must fit in memory)—use bounded memory with multiple passes
      - Tombstone records must be retained long enough for all consumers to process them; premature deletion causes ghost keys
      - Compaction must not reorder messages within a partition—only remove older duplicates of the same key
      - Retention and compaction running simultaneously can cause excessive I/O—throttle and schedule carefully
    concepts:
      - Time-based retention deletes segments older than configured period
      - Size-based retention caps total log size per partition
      - Log compaction deduplicates by key, retaining only the latest value
      - Tombstone records signal key deletion during compaction
      - Active segment immunity prevents data loss during compaction
    skills:
      - Background task scheduling
      - File lifecycle management
      - Log compaction algorithms
      - Disk space management
    deliverables:
      - Time-based retention deleting expired log segments
      - Size-based retention enforcing per-partition log size limits
      - Background retention task running at configurable intervals
      - Log compaction retaining latest value per key
      - Tombstone handling with configurable retention period
      - Active segment exclusion from retention and compaction
    estimated_hours: 12
```
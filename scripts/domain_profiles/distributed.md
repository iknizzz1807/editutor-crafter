# DOMAIN PROFILE: Distributed Systems
# Applies to: distributed, world-scale
# Projects: build-raft, gossip, saga, service mesh, Spanner-like, DynamoDB-like, serverless, etc.

## Fundamental Tension Type
Distributed impossibility results. CONSISTENCY (every node sees same data) vs AVAILABILITY (every request gets response) vs PARTITION TOLERANCE (works despite network failures). CAP: can't have all three. FLP: no deterministic consensus in async system with one crash.

Secondary: consensus needs 2f+1 nodes, strong consistency requires coordination (adds latency), exactly-once impossible over unreliable networks.

## Three-Level View
- **Level 1 — Single Node**: Local state, local processing
- **Level 2 — Cluster Coordination**: Consensus, replication, leader election, membership
- **Level 3 — Network Reality**: Failures, partitions, latency, reordering, duplication, clock skew, split brain

## Soul Section: "Failure Soul"
- Leader crashes mid-operation?
- Follower crashes, returns 10 min later?
- Network partition — which side continues?
- Message duplicated? Is operation idempotent?
- Messages out of order?
- Blast radius if this fails? Cascading?
- Detection method? Heartbeat? Timeout? Latency?
- Recovery: automatic or manual?

## Alternative Reality Comparisons
etcd (Raft), ZooKeeper (Zab), Consul (gossip+Raft), CockroachDB (distributed SQL), Cassandra (eventual, gossip), Kafka (replicated log), Spanner (TrueTime), DynamoDB (quorum), FoundationDB (deterministic simulation).

## TDD Emphasis
- Message format: MANDATORY — every RPC/message type with all fields
- State machine per role: MANDATORY (leader/follower/candidate + transitions + timeouts)
- Failure mode analysis: MANDATORY — enumerate all failures + expected behavior
- Consistency guarantees: specify exactly (linearizable, sequential, causal, eventual)
- Idempotency: which ops idempotent, duplicate handling
- Timeout/retry: exact values, backoff algorithms
- Memory layout: ONLY for log entries and wire messages
- Cache line: SKIP
- Benchmarks: consensus latency p50/p99, throughput under failures, recovery time
- Simulation tests: partition, leader crash, message loss, clock skew

## Cross-Domain Notes
Borrow from systems-lowlevel when: high-perf networking (epoll, io_uring, kernel bypass).
Borrow from data-storage when: replicated storage layer (WAL, page management).
Borrow from security when: mTLS between nodes, auth for cluster membership.



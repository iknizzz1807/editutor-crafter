# DOMAIN PROFILE: Data Storage & Databases
# Applies to: data-storage
# Projects: build-sqlite, build-redis, build-kafka, B-tree, WAL, vector-db, time-series-db, search-engine, etc.

## Fundamental Tension Type
I/O and durability. SPEED (in-memory, no sync) vs DURABILITY (survive crashes, corruption). Every storage system answers: "How much durability for how much speed?"

Secondary: read-optimized vs write-optimized (B-Tree vs LSM), consistency vs availability (CAP), space vs speed (compression cost).

## Three-Level View
- **Level 1 — Query/API**: GET/PUT/SELECT command
- **Level 2 — Storage Engine**: Buffer pool, WAL, indexes, compaction, tx manager
- **Level 3 — Disk/OS I/O**: fsync, page cache, mmap, SSD vs HDD, direct I/O

## Soul Section: "Durability Soul"
Default focus: crash behavior, WAL, fsync, recovery. But adapt to the project's true core — a vector DB is about similarity search, a search engine is about relevance ranking, a time-series DB is about temporal queries. Let the project's nature guide your depth.

## Alternative Reality Comparisons
SQLite, PostgreSQL, Redis, RocksDB/LevelDB, LMDB, Cassandra, DuckDB, FoundationDB.

## TDD Emphasis
- On-disk format: MANDATORY — byte layout of pages, headers, records
- Memory layout of hot structures: MANDATORY
- WAL record format: MANDATORY if durable
- Crash recovery procedure: step-by-step
- Benchmarks: read/write latency, throughput, recovery time, space amplification

## Cross-Domain Awareness
May need systems knowledge (mmap, fsync, direct I/O), distributed concepts (replication, consensus), or AI/ML concepts (vector similarity, embeddings).



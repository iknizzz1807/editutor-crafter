# AUDIT & FIX: wal-impl

## CRITIQUE
- **CLR (Compensation Log Record) Missing**: The ARIES recovery algorithm requires CLRs to handle crashes during recovery (crash during undo). Without CLRs, if the system crashes while undoing a transaction, the next recovery will try to undo the undo, potentially creating an infinite loop. The original milestones mention ARIES but completely omit this critical mechanism.
- **Steal/No-Force Policy Context Missing**: WAL exists because of the buffer pool's steal policy (dirty pages can be flushed before commit) and no-force policy (dirty pages don't need to be flushed at commit). Without understanding these policies, students don't understand WHY WAL is necessary. The project should at least explain this context.
- **Analysis Phase Missing**: ARIES has THREE phases — Analysis, Redo, Undo. The milestones say 'redo then undo' in M3 but the AC mentions scanning from checkpoint to identify committed/active transactions, which IS the analysis phase. However, it's not explicitly named or structured, and the dirty page table from analysis is not used to optimize redo.
- **No Transaction API**: The milestones implement log records and recovery but there's no transaction begin/commit/abort API that actually writes log records. Without this, the WAL is untestable.
- **Log Record Types Incomplete**: M1 deliverables mention 'redo, undo, checkpoint' record types but omit CLR, begin, commit, and abort record types. ARIES needs all of these.
- **Concurrent Transaction Testing Absent**: M2 mentions concurrent writers but no AC in M3/M4 verifies recovery correctness with multiple interleaved transactions.
- **Group Commit Not Addressed**: Group commit (batching multiple transactions' commits into a single fsync) is the most important WAL performance optimization and is mentioned in M2 concepts but has no AC.
- **Master Record Not Explained**: M4 pitfall mentions 'master record' but doesn't explain it. The master record stores the LSN of the last completed checkpoint, used as the starting point for recovery.

## FIXED YAML
```yaml
id: wal-impl
name: "Write-Ahead Log Implementation"
description: >-
  Implement a write-ahead log with ARIES-style crash recovery including analysis,
  redo, and undo phases with compensation log records (CLRs), fuzzy checkpointing,
  and group commit for transaction durability.
difficulty: advanced
estimated_hours: "20-30"
essence: >-
  Sequential log record append with LSN-tracked durability through fsync,
  supporting a steal/no-force buffer pool policy, followed by ARIES crash
  recovery that reconstructs database state through analysis, redo, and undo
  phases with compensation log records (CLRs) to ensure idempotent recovery
  even after crashes during recovery itself.
why_important: >-
  Building this teaches you the fundamental mechanism behind ACID guarantees in
  production databases (PostgreSQL, MySQL, SQLite, MongoDB). Understanding WAL
  and ARIES recovery is critical for any backend or infrastructure engineer
  working with systems that must not lose data despite crashes.
learning_outcomes:
  - Understand why WAL is necessary (steal/no-force buffer pool policy)
  - Design binary log record formats with LSN tracking, CRC integrity, and multiple record types
  - Implement append-only file writes with fsync for durable persistence
  - Build a transaction API (begin, write, commit, abort) that generates log records
  - Implement ARIES recovery with analysis, redo, and undo phases
  - Implement compensation log records (CLRs) for crash-safe undo operations
  - Design group commit to batch multiple transaction commits per fsync
  - Implement fuzzy checkpoints to bound recovery time
  - Debug crash scenarios and verify recovery correctness
skills:
  - Write-Ahead Logging
  - ARIES Crash Recovery
  - Compensation Log Records
  - Log-Structured Storage
  - File System Durability
  - Binary Protocol Design
  - Concurrency Control
  - Group Commit Optimization
tags:
  - advanced
  - c
  - crash-recovery
  - databases
  - go
  - implementation
  - rust
  - storage
architecture_doc: architecture-docs/wal-impl/index.md
languages:
  recommended:
    - Rust
    - Go
    - C
  also_possible:
    - Python
    - Java
resources:
  - name: "ARIES: A Transaction Recovery Method (Mohan et al.)"
    url: https://cs.stanford.edu/people/chr101/cs345/aries.pdf
    type: paper
  - name: "SQLite WAL Mode"
    url: https://sqlite.org/wal.html
    type: documentation
  - name: "CMU Database Systems - Recovery"
    url: https://15445.courses.cs.cmu.edu/
    type: course
prerequisites:
  - type: skill
    name: File I/O (read, write, seek, fsync)
  - type: skill
    name: Database transaction concepts (ACID, commit, abort)
  - type: skill
    name: Binary serialization/deserialization
milestones:
  - id: wal-impl-m1
    name: "Log Record Format & Transaction API"
    description: >-
      Design and implement log record structure with multiple record types (BEGIN,
      UPDATE, CLR, COMMIT, ABORT, CHECKPOINT), and build a transaction API that
      generates log records for begin, write, commit, and abort operations.
    estimated_hours: "4-6"
    concepts:
      - WAL motivation: steal/no-force buffer pool policy
      - Log record types (BEGIN, UPDATE, COMMIT, ABORT, CLR, CHECKPOINT)
      - LSN (Log Sequence Number) as monotonically increasing record identifier
      - Before-image (undo data) and after-image (redo data)
      - Compensation Log Record (CLR) structure with undoNextLSN pointer
      - CRC checksum for corruption detection
    skills:
      - Binary serialization with variable-length records
      - Record type design with discriminated unions
      - CRC32 checksum computation
      - Transaction state tracking
    acceptance_criteria:
      - Each log record has a header containing LSN (monotonically increasing, unique), record type, transaction ID, previous LSN of the same transaction (prevLSN), and record length
      - Record types include BEGIN (transaction start), UPDATE (data modification with before-image and after-image), COMMIT, ABORT, CLR (compensation record with undoNextLSN), and CHECKPOINT
      - UPDATE records store page ID, offset within page, before-image (old value), and after-image (new value) for undo and redo respectively
      - CLR records store the undoNextLSN field pointing to the next record to undo if the current undo is interrupted by a crash
      - CRC32 checksum is appended to each record; checksum covers the entire record including header
      - Transaction API exposes begin_txn() → txn_id, write(txn_id, page_id, offset, old_val, new_val), commit(txn_id), and abort(txn_id)
      - begin_txn() writes a BEGIN record; write() writes an UPDATE record; commit() writes a COMMIT record; abort() initiates undo and writes CLRs followed by an ABORT record
      - Records are correctly serialized to bytes and deserialized back with byte-perfect round-trip fidelity
    pitfalls:
      - Variable-length before/after images require length-prefixed encoding; fixed-size records waste space but simplify parsing
      - prevLSN chain links all records of the same transaction, enabling efficient undo by following the chain backward
      - CLR's undoNextLSN is NOT the same as prevLSN — undoNextLSN points to the next record to undo (skipping the already-undone record), while prevLSN points to the previous record of the same transaction
      - Forgetting the CRC makes it impossible to detect torn writes or disk corruption during recovery
    deliverables:
      - Log record header structure with LSN, type, txn_id, prevLSN, and length
      - All record types (BEGIN, UPDATE, COMMIT, ABORT, CLR, CHECKPOINT) with appropriate fields
      - CRC32 integrity checksum per record
      - Record serialization and deserialization with round-trip tests
      - Transaction API (begin, write, commit, abort) generating appropriate log records
      - Transaction table tracking active transactions and their lastLSN

  - id: wal-impl-m2
    name: "Log Writer with Group Commit"
    description: >-
      Implement an append-only log writer with fsync-based durability, concurrent
      writer support, and group commit optimization that batches multiple
      transaction commits into a single fsync.
    estimated_hours: "5-7"
    concepts:
      - Append-only sequential writes
      - fsync semantics and durability guarantees
      - Group commit (batching multiple commits per fsync)
      - Concurrent log append with serialized writes
      - Log segment rotation at size threshold
      - Torn write detection and prevention
    skills:
      - File I/O with fsync optimization
      - Concurrent write serialization (mutex or lock-free buffer)
      - Group commit implementation
      - Log segment management
    acceptance_criteria:
      - Log records are appended sequentially to the active log segment file; writes never overwrite existing data
      - COMMIT is not acknowledged to the client until the log record (and all preceding records for that transaction) is flushed to disk via fsync
      - Group commit batches multiple pending commits into a single fsync — when multiple transactions commit concurrently, only one fsync call is made for the batch
      - Group commit benchmark shows at least 5x throughput improvement over per-commit fsync for 100 concurrent committing transactions
      - Concurrent writers (from different transactions) are serialized using a mutex or lock-free log buffer so that records do not interleave within the file
      - Log segment rotation creates a new segment file when the current segment exceeds configurable size threshold (default 64MB); old segments are retained
      - Atomic write guarantee: if the system crashes mid-write, the partial record is detectable by CRC mismatch during recovery and is truncated
      - Write buffer batches small records in memory before flushing, reducing the number of write system calls
    pitfalls:
      - fsync on Linux flushes to disk controller cache, not necessarily to disk platters; O_DSYNC or fdatasync may be more appropriate depending on requirements
      - Group commit requires a wait mechanism — committing transactions must wait for the next fsync batch rather than each calling fsync independently
      - Torn writes (partial record on disk) must be handled during recovery; CRC mismatch on the last record indicates a torn write
      - Log segment rotation must be atomic with the switch — writing to the old segment and the new segment simultaneously corrupts data
      - Write buffer must be flushed on commit — buffering commit records without flushing defeats durability
    deliverables:
      - Sequential log writer appending records to the active segment file
      - fsync-based durability with commit-before-acknowledge guarantee
      - Group commit mechanism batching concurrent commits into single fsync
      - Concurrent writer serialization (mutex or lock-free buffer)
      - Log segment rotation at configurable size threshold
      - Write buffer for batching small records
      - Group commit benchmark comparing per-commit vs batched fsync throughput

  - id: wal-impl-m3
    name: "ARIES Crash Recovery with CLRs"
    description: >-
      Implement ARIES-style crash recovery with three phases: analysis (identify
      committed and active transactions), redo (replay all changes), and undo
      (roll back uncommitted transactions using CLRs for crash-safe undo).
    estimated_hours: "6-9"
    concepts:
      - ARIES three phases: Analysis, Redo, Undo
      - Analysis phase builds transaction table and dirty page table
      - Redo phase replays ALL changes (committed and uncommitted) from redoLSN
      - Undo phase rolls back uncommitted transactions using prevLSN chain
      - CLRs written during undo prevent re-undo on crash during recovery
      - Idempotent redo (comparing pageLSN to record LSN)
    skills:
      - ARIES algorithm implementation
      - Log scanning and state reconstruction
      - CLR generation during undo
      - Idempotent operation design
      - Crash simulation and testing
    acceptance_criteria:
      - Analysis phase: scans log from last checkpoint forward, reconstructs the transaction table (active transactions with status and lastLSN) and dirty page table (pages with uncommitted changes and their recLSN)
      - Redo phase: replays ALL logged changes (both committed and uncommitted) starting from the minimum recLSN in the dirty page table; a change is skipped if the page's pageLSN >= the record's LSN (already applied)
      - Undo phase: rolls back all transactions that were active (uncommitted) at crash time by following each transaction's prevLSN chain backward, applying inverse operations
      - Each undo operation writes a CLR to the log with undoNextLSN pointing to the next record to undo; this ensures that if the system crashes during undo, recovery skips already-undone records
      - Recovery is idempotent — running recovery multiple times produces the same database state (verified by test)
      - Recovery handles crash during undo correctly — if a crash occurs mid-undo, the next recovery uses CLRs to skip already-undone operations
      - Test with 3 concurrent transactions: T1 committed, T2 aborted before crash, T3 active at crash — after recovery, T1's changes are present, T2's and T3's changes are rolled back
      - Corrupted final log record (torn write) is detected by CRC mismatch and truncated before recovery begins
    pitfalls:
      - Redo must replay ALL changes, not just committed ones — uncommitted changes may have been flushed to disk (steal policy) and must be present for undo to work
      - CLRs are critical for crash safety during recovery — without them, a crash during undo causes redo to replay the undone changes, then undo tries to undo them again, potentially looping
      - PageLSN comparison during redo is essential for idempotency; without it, redo applies the same change twice, corrupting data
      - The dirty page table's recLSN determines where redo starts; without it, redo must start from the beginning of the log (slow)
      - Undo order must follow reverse LSN order across all active transactions (using a priority queue of lastLSNs), not undo each transaction independently
    deliverables:
      - Analysis phase reconstructing transaction table and dirty page table from log
      - Redo phase replaying changes from minimum recLSN with pageLSN skip logic
      - Undo phase rolling back active transactions with CLR generation
      - CLR writer ensuring crash-safe undo operations
      - Torn write detector truncating corrupt final record
      - Recovery integration test with committed, aborted, and in-flight transactions
      - Crash-during-recovery test verifying CLR correctness

  - id: wal-impl-m4
    name: "Fuzzy Checkpointing & Log Truncation"
    description: >-
      Implement fuzzy (non-blocking) checkpoints that record the current
      transaction table and dirty page table to the log, enabling faster recovery
      and safe log truncation.
    estimated_hours: "4-6"
    concepts:
      - Fuzzy checkpoint (non-blocking, concurrent with transactions)
      - Checkpoint record content (transaction table, dirty page table)
      - Master record storing last checkpoint LSN
      - Log truncation based on checkpoint position
      - Recovery time bounded by checkpoint interval
    skills:
      - Background checkpoint scheduling
      - Snapshot consistency under concurrent modification
      - Log file lifecycle management
      - Recovery time optimization
    acceptance_criteria:
      - Fuzzy checkpoint writes a CHECKPOINT_BEGIN record, captures snapshots of the transaction table and dirty page table, then writes a CHECKPOINT_END record with the captured data
      - Concurrent transactions continue executing during checkpoint without blocking; the snapshot represents a consistent but potentially slightly stale view
      - Master record (stored in a well-known location, e.g., first page of a master file) stores the LSN of the last completed CHECKPOINT_END record
      - Recovery reads the master record to find the last checkpoint, then starts the analysis phase from that checkpoint's LSN
      - Log segments older than the last checkpoint's minimum recLSN can be safely truncated (deleted); truncation reclaims disk space
      - Log truncation does NOT delete segments that contain records needed for undo of active transactions (i.e., segments referenced by any active transaction's firstLSN)
      - Checkpoint interval is configurable (by time or by number of log records); shorter intervals mean faster recovery but more checkpoint overhead
      - Recovery time test: "without checkpoint, recovery scans entire log; with checkpoint at 50% through the log, recovery scan is approximately halved"
    pitfalls:
      - Checkpoint must capture transaction table and dirty page table atomically (or nearly so); capturing them at different times can cause inconsistencies
      - Master record must be updated atomically after CHECKPOINT_END is written; crash between CHECKPOINT_END and master record update is handled by using the previous checkpoint
      - Log truncation must be conservative — truncating segments needed for undo of long-running transactions causes unrecoverable data loss
      - Frequent checkpoints reduce recovery time but add I/O overhead during normal operation; benchmark the tradeoff
      - The master record file must be separate from the log segments; otherwise, truncating old segments destroys the master record
    deliverables:
      - Fuzzy checkpoint writer producing CHECKPOINT_BEGIN and CHECKPOINT_END records with transaction table and dirty page table snapshots
      - Master record file storing last completed checkpoint LSN
      - Recovery integration reading master record to find checkpoint starting point
      - Log truncation safely removing segments older than checkpoint minimum recLSN
      - Configurable checkpoint interval (time-based or record-count-based)
      - Recovery time benchmark comparing with and without checkpointing
```
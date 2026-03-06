# 🎯 Project Charter: Write-Ahead Log Implementation
## What You Are Building
A production-grade Write-Ahead Log (WAL) system implementing the ARIES crash recovery algorithm. Your implementation will serialize transaction records to an append-only log file with CRC32 integrity, batch multiple transaction commits into single fsync operations via group commit, and perform three-phase crash recovery (Analysis, Redo, Undo) with Compensation Log Records for idempotent recovery even after crashes during recovery itself. The final system supports fuzzy checkpointing to bound recovery time and enables safe log segment truncation.
## Why This Project Exists
Every production database—PostgreSQL, MySQL, SQLite, MongoDB—uses a Write-Ahead Log to guarantee ACID durability. Yet most backend engineers treat WAL as a black box, unaware of the design decisions that make databases resilient to power failures, disk corruption, and crashes mid-operation. Building one from scratch exposes the fundamental tradeoffs: why the steal/no-force buffer pool policy requires both redo and undo capability, how group commit turns a 10ms fsync into sub-millisecond latency per transaction, and why ARIES recovery must replay *uncommitted* changes before rolling them back. This is the systems-level understanding that separates application developers from infrastructure engineers.
## What You Will Be Able to Do When Done
- Design binary log record formats with LSN tracking, CRC integrity, and variable-length fields
- Implement an append-only log writer with fsync-based durability and concurrent writer serialization
- Achieve 5x+ throughput improvement using group commit to batch multiple commits per fsync
- Build the complete ARIES recovery algorithm: Analysis (reconstruct transaction state), Redo (replay all changes with pageLSN idempotency), and Undo (roll back uncommitted transactions)
- Write Compensation Log Records (CLRs) that enable crash-safe undo—even if the system crashes during recovery
- Implement fuzzy checkpointing that captures system state without blocking transactions
- Safely truncate old log segments based on checkpoint boundaries and active transaction ranges
- Debug crash scenarios by understanding exactly what state the log captures and how recovery reconstructs it
## Final Deliverable
~2,500-3,500 lines of C (or equivalent Rust/Go) across 12-15 source files implementing: log record serialization (6 record types), append-only log writer with 64MB segment rotation, group commit with leader/follower protocol, complete ARIES recovery with CLR generation, fuzzy checkpointing with atomic master record updates, and background checkpoint/truncation thread. The system boots in QEMU or on real hardware, recovers correctly from simulated crashes, and demonstrates group commit achieving 5x+ throughput over per-commit fsync.
## Is This Project For You?
**You should start this if you:**
- Are comfortable with C, Rust, or Go (systems programming with manual memory management or explicit ownership)
- Understand basic file I/O (open, read, write, seek, fsync)
- Know database transaction concepts (ACID properties, commit, abort)
- Have experience with binary serialization (packing structs to bytes, endianness)
- Can debug concurrent code with mutexes and condition variables
**Come back after you've learned:**
- C pointers and manual memory management (try "Learn C the Hard Way" first)
- Basic database internals (CMU 15-445 lectures on buffer pools and logging)
- Concurrent programming with pthreads or equivalent
## Estimated Effort
| Phase | Time |
|-------|------|
| Log Record Format & Transaction API | ~4-6 hours |
| Log Writer with Group Commit | ~5-7 hours |
| ARIES Crash Recovery with CLRs | ~6-9 hours |
| Fuzzy Checkpointing & Log Truncation | ~4-6 hours |
| **Total** | **~20-28 hours** |
## Definition of Done
The project is complete when:
- All six record types (BEGIN, UPDATE, COMMIT, ABORT, CLR, CHECKPOINT) serialize/deserialize with byte-perfect round-trip fidelity and CRC32 validation
- Group commit benchmark shows ≥5x throughput improvement over per-commit fsync with 100 concurrent transactions
- Three-phase ARIES recovery correctly handles the test scenario: T1 committed (changes present), T2 aborted before crash (changes rolled back via CLRs), T3 active at crash (changes rolled back during undo)
- Recovery is idempotent: running crash recovery N times produces identical database state
- Crash-during-undo test passes: log contains partial CLRs, recovery follows undo_next_lsn pointers to resume correctly
- Fuzzy checkpoint captures transaction table and dirty page table without blocking concurrent transactions (lock hold <100µs)
- Recovery time with checkpoint at 50% log position is ≤60% of recovery without checkpoint
- Log truncation correctly identifies safe segments: respects both minimum rec_lsn and active transaction first_lsn ranges
- All unit tests pass with 100% success rate, including torn write detection and truncation blocked by long-running transactions

---

# 📚 Before You Read This: Prerequisites & Further Reading
> **Read these first.** The Atlas assumes you are familiar with the foundations below.
> Resources are ordered by when you should encounter them — some before you start, some at specific milestones.
---
## Database Durability Fundamentals
### Read BEFORE starting this project — required foundational knowledge.
**Paper**: Mohan, C., et al. "ARIES: A Transaction Recovery Method Supporting Fine-Granularity Locking and Partial Rollbacks Using Write-Ahead Logging." *ACM TODS* 17, no. 1 (1992): 94-162.
**Why**: This is THE original paper on the ARIES recovery algorithm you'll implement. Every design decision in Milestones 3-4 traces directly to this work.
**When to read**: Before Milestone 1. The paper is dense; read Sections 1-3 for concepts, then reference Sections 4-6 during Milestone 3 implementation.
---
## Steal/No-Force Buffer Pool Policy
### Read BEFORE starting Milestone 1 — explains WHY write-ahead logging exists.
**Best Explanation**: Hellerstein, Joseph, and Stonebraker, Michael, eds. *Readings in Database Systems*, 5th ed. Chapter 6: "Logging and Recovery" (specifically the introduction and the "Shadow Paging vs. WAL" section).
**Why**: The steal/no-force policy is the fundamental tension that makes WAL necessary. Without understanding this, the entire design seems arbitrary.
**When to read**: Before Milestone 1. Re-read the section on "Write-Ahead Logging Protocol" before Milestone 3.
---
## Log Sequence Numbers (LSNs)
### Read AFTER Milestone 1 (Log Record Format) — you'll have context for why LSNs matter.
**Best Explanation**: Chapter 17 of "Database Internals" by Alex Petrov, specifically the "Log Sequence Numbers and Page LSNs" section (pages 328-335).
**Why**: Petrov provides the clearest explanation of how LSNs serve as both identifiers and positions, and how pageLSN enables idempotent recovery.
**When to read**: After completing Milestone 1, before starting Milestone 2. You'll appreciate the LSN-as-offset design pattern much more.
---
## fsync and Storage Durability
### Read BEFORE Milestone 2 (Log Writer) — critical for understanding group commit motivation.
**Paper**: Pillai, Thanumalayan S., et al. "All File Systems Are Not Created Equal: On the Complexity of Crafting Crash-Consistent Applications." *SOSP '17* (2017): 433-448.
**Why**: This paper systematically demonstrates that fsync semantics are surprisingly subtle and often misunderstood. Essential context for why group commit is necessary.
**When to read**: Before Milestone 2. Section 3 on "File System Crash Consistency" is most relevant.
---
## Group Commit and Batching
### Read AFTER Milestone 2 (Log Writer) — you'll have implemented the basic version.
**Spec**: The PostgreSQL documentation on "WAL Configuration" describes the `commit_delay` and `commit_siblings` parameters implementing group commit.
**Why**: PostgreSQL's production implementation shows how the theoretical concept maps to real-world tuning parameters.
**When to read**: After completing Milestone 2. Compare your implementation's batch size threshold approach with PostgreSQL's time-based approach.
---
## Idempotent Operations
### Read BEFORE Milestone 3 (ARIES Recovery) — conceptual foundation for redo phase.
**Best Explanation**: Kleppmann, Martin. *Designing Data-Intensive Applications*. Chapter 9: "Consistency and Consensus" — specifically the section on "Idempotence" (pages 330-332 in most editions).
**Why**: The redo phase must be idempotent — running it multiple times must produce identical results. Kleppmann explains this concept clearly in the context of distributed systems, which directly applies to crash recovery.
**When to read**: Before starting Milestone 3. Return to this concept when implementing the pageLSN skip logic.
---
## Compensation Log Records (CLRs)
### Read DURING Milestone 3 (ARIES Recovery) — specifically during undo phase implementation.
**Paper**: Reference Section 3.3 of the ARIES paper (Mohan et al., 1992) — "Compensation Log Records."
**Why**: The CLR mechanism is subtle and easy to misunderstand. The original paper's explanation is still the clearest. Focus on the distinction between `prev_lsn` and `undo_next_lsn`.
**When to read**: When implementing the undo phase in Milestone 3. Read Section 3.3, then implement, then re-read to verify understanding.
---
## Checkpointing Strategies
### Read BEFORE Milestone 4 (Fuzzy Checkpointing) — explains the design space.
**Best Explanation**: The SQLite documentation on "Write-Ahead Logging" — specifically the "Checkpointing" section.
**Why**: SQLite's WAL mode documentation provides a practical, implementation-focused explanation of checkpointing tradeoffs that complements the ARIES paper's theoretical treatment.
**When to read**: Before Milestone 4. SQLite uses a simpler checkpoint model than ARIES, which helps clarify what ARIES adds.
---
## Recovery Time Analysis
### Read AFTER Milestone 4 — context for why your implementation matters.
**Paper**: Gray, Jim, and Reuter, Andreas. *Transaction Processing: Concepts and Techniques*. Chapter 11: "Recovery Manager" — specifically the "Recovery Time Bounds" discussion.
**Why**: Gray and Reuter provide the theoretical framework for analyzing recovery time as a function of checkpoint interval — the same tradeoffs your Milestone 4 benchmark measures.
**When to read**: After completing Milestone 4 and running your benchmarks. Compare your measured speedup with their theoretical predictions.
---
## Cross-Domain Connections
### Read AFTER completing the project — connects WAL concepts to broader systems.
**Best Explanation**: Kleppmann, Martin. *Designing Data-Intensive Applications*. Chapter 5: "Replication" — specifically the sections on "Implementation of Replication Logs" and "Relational vs. Operation Logs."
**Why**: The WAL you've built appears throughout distributed systems: Kafka's log compaction, event sourcing, Raft's log replication, and distributed consensus. Kleppmann draws these connections explicitly.
**When to read**: After completing all four milestones. This chapter will help you see how the patterns you've learned transfer to distributed databases, message queues, and consensus algorithms.
---
## Reference Implementations
### Code to study DURING implementation — specific files for comparison.
**PostgreSQL WAL**: `src/backend/access/transam/xlog.c` and `src/backend/access/transam/xloginsert.c`
**Why**: Production-quality WAL implementation with group commit, checkpoints, and recovery. Focus on `XLogInsert` for the append path and `PerformRecovery` for ARIES phases.
**When to study**: Reference during Milestones 2-4. Don't read linearly; search for specific functions matching your implementation.
---
**SQLite WAL**: `src/wal.c` and `src/wal.h`
**Why**: Simpler than PostgreSQL but still production-quality. Good reference for the basic append/checkpoint flow without ARIES complexity.
**When to study**: During Milestone 2 for append logic and Milestone 4 for checkpointing. SQLite's checkpoint is simpler than ARIES fuzzy checkpointing but follows the same principles.
---
*Total resources: 13 (2 papers, 1 spec, 2 code references, 4 book chapters, 4 cross-references)*

---

# Write-Ahead Log Implementation

A Write-Ahead Log (WAL) is the fundamental durability mechanism in virtually every production database system. This project implements a complete WAL with ARIES-style crash recovery, teaching you how databases guarantee that committed transactions survive system crashes. You'll build log record formats, an append-only writer with group commit optimization, and the full three-phase ARIES recovery algorithm (Analysis, Redo, Undo) with Compensation Log Records (CLRs) for crash-safe recovery. The system supports fuzzy checkpointing to bound recovery time and enable safe log truncation.



<!-- MS_ID: wal-impl-m1 -->
# Milestone 1: Log Record Format & Transaction API
## The Durability Problem: Why WAL Exists
You're building a database. A client sends a transaction: "Update account A from $100 to $80, update account B from $50 to $70." Your code modifies the in-memory buffer pool, acknowledges success, and the client walks away happy.
Then the power fails.
When the system restarts, what's on disk? Account A might have $80 (the new value), $100 (the old value), or garbage (a partial write). Account B is in the same uncertain state. The client thinks the transfer completed, but you have no way to know.
This is the **durability problem**. Before you can understand the solution (Write-Ahead Logging), you need to understand why it's necessary.

![Steal/No-Force Buffer Pool Policy](./diagrams/diag-steal-noforce-policy.svg)

[[EXPLAIN:steal/no-force-buffer-pool-policy|Steal/No-Force Buffer Pool Policy]]
### The Three Policies That Matter
A buffer pool manager makes two independent decisions that determine durability characteristics:
| Policy | Question | Implication |
|--------|----------|-------------|
| **Steal** | Can uncommitted changes be written to disk before commit? | If YES, we need UNDO capability to roll back aborted transactions |
| **Force** | Must all committed changes be written to disk before commit returns? | If NO, we need REDO capability to replay after crash |
The combination **Steal/No-Force** is what almost every production database uses:
- **Steal = YES**: We want the buffer pool to evict dirty pages (pages with uncommitted changes) when under memory pressure. Blocking eviction until commit would make the database unusable.
- **Force = NO**: Waiting for disk writes before acknowledging commit would make every transaction take 10+ milliseconds (the typical fsync latency on SSDs). We want to acknowledge immediately after logging.
But Steal/No-Force creates a problem: after a crash, the disk might contain uncommitted changes (from stolen pages) and might be missing committed changes (from unforced writes). We need a way to restore consistency.
**The WAL Principle**: Before any page is written to disk, write a log record describing the change to stable storage. After a crash, use the log to redo committed changes and undo uncommitted changes.
## The Log Record: Your Atomic Unit of Durability

![Log Record Binary Format (Microscopic)](./diagrams/diag-log-record-binary-layout.svg)

A log record is the smallest durable unit in your system. Each record describes one atomic modification. But it's not just "what changed" — it's a sophisticated data structure that enables crash recovery.
### The Header: Every Record Has These Fields
```c
// Fixed-size header for every log record (16 bytes)
typedef struct {
    uint64_t lsn;        // Log Sequence Number: unique, monotonically increasing
    uint32_t type;       // Record type: BEGIN, UPDATE, COMMIT, ABORT, CLR, CHECKPOINT
    uint32_t txn_id;     // Transaction this record belongs to
    uint64_t prev_lsn;   // LSN of previous record in same transaction (0 if first)
    uint32_t length;     // Total record length including header and CRC
} LogRecordHeader;
```
[[EXPLAIN:lsn-(log-sequence-number)-semantics|LSN (Log Sequence Number) semantics]]
Let's examine why each field exists:
**LSN (Log Sequence Number)**: This is the record's unique identifier. LSNs are monotonically increasing — each new record gets an LSN higher than all previous records. In practice, LSNs are often byte offsets into the log file, making them both identifiers and positions.
**Type**: Discriminates between record types. We'll cover each type in detail, but for now know that we have six: `BEGIN`, `UPDATE`, `COMMIT`, `ABORT`, `CLR`, and `CHECKPOINT`.
**txn_id**: Every record belongs to a transaction. During recovery, we need to know which records to undo (uncommitted transactions) and which to leave alone (committed ones).
**prev_lsn**: This is the crucial field that many developers miss. Records from the same transaction are linked backward through `prev_lsn`. If transaction 5 has three UPDATE records at LSNs 100, 150, and 200, then:
- Record at LSN 200 has `prev_lsn = 150`
- Record at LSN 150 has `prev_lsn = 100`
- Record at LSN 100 has `prev_lsn = 0` (or the LSN of BEGIN)
This backward chain enables efficient undo — we don't need to scan the entire log to find a transaction's records.
**Length**: We need to know where one record ends and the next begins. This field stores the total record size, including the header and the trailing CRC.
### The Record Types: A Discriminated Union

![Log Record Type Hierarchy](./diagrams/diag-record-type-hierarchy.svg)

After the header, the payload varies by record type. Let's examine each:
#### BEGIN Record
Marks the start of a transaction. Payload is minimal — the header already contains the `txn_id`.
```c
typedef struct {
    LogRecordHeader header;
    // No additional payload needed
    uint32_t crc;  // Covers header
} BeginRecord;
```
Total size: 16 (header) + 0 (payload) + 4 (CRC) = 20 bytes.
#### UPDATE Record
This is the workhorse. Every data modification generates an UPDATE record.
```c
typedef struct {
    LogRecordHeader header;
    // What page was modified
    uint64_t page_id;
    // Where within the page (byte offset)
    uint32_t offset;
    // Before-image: old value (for undo)
    uint32_t old_value_len;
    uint8_t  old_value[];  // Variable-length
    // After-image: new value (for redo)
    uint32_t new_value_len;
    uint8_t  new_value[];  // Variable-length
    // CRC covers all of the above
    uint32_t crc;
} UpdateRecord;
```
The **before-image** is the old value — what the data looked like before this transaction touched it. During recovery's undo phase, we write this back to undo the change.
The **after-image** is the new value — what the transaction wrote. During recovery's redo phase, we write this to replay the change.
**Why store both?** This is the key insight of ARIES. Because of the steal policy, uncommitted changes might already be on disk. If we crash and need to undo, we need the before-image. Because of the no-force policy, committed changes might not be on disk. If we crash and need to redo, we need the after-image.
#### COMMIT Record
Marks successful transaction completion.
```c
typedef struct {
    LogRecordHeader header;
    uint32_t crc;
} CommitRecord;
```
When a COMMIT record is safely on disk (after fsync), the transaction is durable. The client can be acknowledged.
#### ABORT Record
Marks transaction failure. Written after all undo work is complete.
```c
typedef struct {
    LogRecordHeader header;
    uint32_t crc;
} AbortRecord;
```
#### CLR (Compensation Log Record)
This is the sophisticated record that makes crash-safe recovery possible.
```c
typedef struct {
    LogRecordHeader header;
    // The page being undone
    uint64_t page_id;
    uint32_t offset;
    // The inverse operation's data (undo information)
    uint32_t undo_data_len;
    uint8_t  undo_data[];
    // CRITICAL: Points to the NEXT record to undo (not the current one!)
    uint64_t undo_next_lsn;
    uint32_t crc;
} ClrRecord;
```

![CLR Record Structure vs UPDATE Record](./diagrams/diag-clr-structure.svg)

The `undo_next_lsn` field is subtle and crucial. It is NOT the same as `prev_lsn` in the header. Here's the difference:
- `prev_lsn`: The previous record in this transaction's chain (always the UPDATE record being undone)
- `undo_next_lsn`: The next record to undo after this CLR (points to the `prev_lsn` of the undone UPDATE)
Consider transaction T1 with UPDATE records at LSN 100, 150, 200:
1. Abort T1. Start undoing LSN 200.
2. Write CLR at LSN 250 with `undo_next_lsn = 150` (skip 200, undo 150 next)
3. Crash!
4. Recovery: We see T1 was aborted (status in transaction table). We see CLR at 250 with `undo_next_lsn = 150`.
5. We resume undo from LSN 150, not 200. The CLR "remembers" that we already undid 200.
Without CLRs, a crash during undo would cause us to re-undo already-undone operations, potentially corrupting data or causing infinite loops.

![CLR Prevents Recovery Loop](./diagrams/diag-clr-crash-during-undo.svg)

#### CHECKPOINT Record
Captures system state to bound recovery time. We'll cover fuzzy checkpointing in Milestone 4, but the record structure is:
```c
typedef struct {
    LogRecordHeader header;
    // Number of active transactions
    uint32_t num_active_txns;
    // Array of (txn_id, status, last_lsn) for each active transaction
    TxnTableEntry active_txns[];
    // Number of dirty pages in buffer pool
    uint32_t num_dirty_pages;
    // Array of (page_id, rec_lsn) for each dirty page
    DirtyPageEntry dirty_pages[];
    uint32_t crc;
} CheckpointRecord;
```
### CRC32: Detecting Corruption
Every record ends with a CRC32 checksum covering the header and payload. This is your defense against:
1. **Torn writes**: If the system crashes mid-write, the partial record will have a mismatched CRC
2. **Disk corruption**: Silent data corruption is real; CRC catches it
3. **Memory corruption**: Bugs in your code might corrupt records before write; CRC catches this too
```c
#include <zlib.h>  // For crc32()
uint32_t compute_crc(const void* data, size_t len) {
    // CRC32 from zlib: fast, well-tested
    return crc32(0L, (const unsigned char*)data, len);
}
```
During recovery, if the final record has a CRC mismatch, we truncate it — it was a torn write from the crash.
## The prev_lsn Chain: Your Undo Lifeline

![prevLSN Chain Across Transactions](./diagrams/diag-prevlsn-chain.svg)

Let's trace through a concrete example. Transaction T1 performs three updates, then aborts:
| LSN | Type | txn_id | prev_lsn | Description |
|-----|------|--------|----------|-------------|
| 100 | BEGIN | 1 | 0 | Transaction starts |
| 150 | UPDATE | 1 | 100 | Update page 5 |
| 200 | UPDATE | 1 | 150 | Update page 7 |
| 250 | UPDATE | 1 | 200 | Update page 3 |
| 300 | CLR | 1 | 250 | Undo LSN 250, undoNext=200 |
| 350 | CLR | 1 | 300 | Undo LSN 200, undoNext=150 |
| 400 | CLR | 1 | 350 | Undo LSN 150, undoNext=100 |
| 450 | ABORT | 1 | 400 | Transaction fully undone |
Notice the chain: to undo T1, we follow `prev_lsn` backward from the last UPDATE (250 → 200 → 150 → 100). Each CLR records `undo_next_lsn` to skip already-undone work if we crash during abort.
This is why `prev_lsn` must be stored in every record, not just UPDATE records. The chain is the transaction's history, and we need to traverse it backward during undo.
## Serialization: From Structs to Bytes

![Serialization/Deserialization Round-Trip](./diagrams/diag-serialization-roundtrip.svg)

You cannot write C structs directly to disk. Here's why:
1. **Alignment padding**: Compilers insert padding between struct fields for alignment. The padding bytes contain garbage.
2. **Endianness**: Different architectures store multi-byte integers differently (big-endian vs little-endian).
3. **Pointer fields**: You can't serialize pointers — they're meaningless after restart.
4. **Variable-length fields**: C structs can't represent them directly.
You need explicit serialization. Here's the approach:
### Serialization Strategy
1. **Fixed header**: Always serialize in little-endian (canonical choice, matches x86/ARM)
2. **Length-prefix variable fields**: Store length before data
3. **No padding in serialized form**: Pack bytes tightly
4. **CRC at the end**: Covers everything before it
```c
// Serialize an UPDATE record to a byte buffer
// Returns bytes written, or -1 on error
ssize_t serialize_update_record(const UpdateRecord* rec, uint8_t* buf, size_t buf_len) {
    if (!rec || !buf) return -1;
    size_t offset = 0;
    // Helper to write little-endian integers
    auto write_le64 = [&](uint64_t val) {
        if (offset + 8 > buf_len) return false;
        buf[offset++] = val & 0xFF;
        buf[offset++] = (val >> 8) & 0xFF;
        buf[offset++] = (val >> 16) & 0xFF;
        buf[offset++] = (val >> 24) & 0xFF;
        buf[offset++] = (val >> 32) & 0xFF;
        buf[offset++] = (val >> 40) & 0xFF;
        buf[offset++] = (val >> 48) & 0xFF;
        buf[offset++] = (val >> 56) & 0xFF;
        return true;
    };
    auto write_le32 = [&](uint32_t val) {
        if (offset + 4 > buf_len) return false;
        buf[offset++] = val & 0xFF;
        buf[offset++] = (val >> 8) & 0xFF;
        buf[offset++] = (val >> 16) & 0xFF;
        buf[offset++] = (val >> 24) & 0xFF;
        return true;
    };
    // Header
    if (!write_le64(rec->header.lsn)) return -1;
    if (!write_le32(rec->header.type)) return -1;
    if (!write_le32(rec->header.txn_id)) return -1;
    if (!write_le64(rec->header.prev_lsn)) return -1;
    if (!write_le32(rec->header.length)) return -1;
    // UPDATE-specific fields
    if (!write_le64(rec->page_id)) return -1;
    if (!write_le32(rec->offset)) return -1;
    // Before-image
    if (!write_le32(rec->old_value_len)) return -1;
    if (offset + rec->old_value_len > buf_len) return -1;
    memcpy(buf + offset, rec->old_value, rec->old_value_len);
    offset += rec->old_value_len;
    // After-image
    if (!write_le32(rec->new_value_len)) return -1;
    if (offset + rec->new_value_len > buf_len) return -1;
    memcpy(buf + offset, rec->new_value, rec->new_value_len);
    offset += rec->new_value_len;
    // CRC (covers everything up to this point)
    uint32_t crc = compute_crc(buf, offset);
    if (!write_le32(crc)) return -1;
    return offset;
}
```
### Deserialization: Reading Records Back
Deserialization must handle:
1. Reading the fixed header first
2. Using `length` field to know total record size
3. Parsing variable-length fields based on length prefixes
4. Verifying CRC before accepting the record
```c
// Deserialize an UPDATE record from a byte buffer
// Returns bytes consumed, or -1 on error
ssize_t deserialize_update_record(const uint8_t* buf, size_t buf_len, UpdateRecord* rec) {
    if (!buf || !rec || buf_len < HEADER_SIZE) return -1;
    size_t offset = 0;
    // Read header
    rec->header.lsn = read_le64(buf + offset); offset += 8;
    rec->header.type = read_le32(buf + offset); offset += 4;
    rec->header.txn_id = read_le32(buf + offset); offset += 4;
    rec->header.prev_lsn = read_le64(buf + offset); offset += 8;
    rec->header.length = read_le32(buf + offset); offset += 4;
    // Verify we have enough data
    if (buf_len < rec->header.length) return -1;
    // Verify CRC first
    uint32_t stored_crc = read_le32(buf + rec->header.length - 4);
    uint32_t computed_crc = compute_crc(buf, rec->header.length - 4);
    if (stored_crc != computed_crc) {
        return -1;  // CRC mismatch — corruption!
    }
    // Read UPDATE-specific fields
    rec->page_id = read_le64(buf + offset); offset += 8;
    rec->offset = read_le32(buf + offset); offset += 4;
    // Before-image
    rec->old_value_len = read_le32(buf + offset); offset += 4;
    rec->old_value = malloc(rec->old_value_len);
    if (!rec->old_value) return -1;
    memcpy(rec->old_value, buf + offset, rec->old_value_len);
    offset += rec->old_value_len;
    // After-image
    rec->new_value_len = read_le32(buf + offset); offset += 4;
    rec->new_value = malloc(rec->new_value_len);
    if (!rec->new_value) {
        free(rec->old_value);
        return -1;
    }
    memcpy(rec->new_value, buf + offset, rec->new_value_len);
    offset += rec->new_value_len;
    return rec->header.length;  // Return total record size
}
```
### Round-Trip Testing
Every serialization function needs a round-trip test: serialize, deserialize, verify the deserialized record matches the original.
```c
void test_update_roundtrip(void) {
    // Create test record
    uint8_t old_val[] = {0xDE, 0xAD, 0xBE, 0xEF};
    uint8_t new_val[] = {0xCA, 0xFE, 0xBA, 0xBE, 0x00};
    UpdateRecord original = {
        .header = {
            .lsn = 1000,
            .type = RECORD_UPDATE,
            .txn_id = 42,
            .prev_lsn = 500,
            .length = 0  // Computed during serialization
        },
        .page_id = 7,
        .offset = 24,
        .old_value_len = sizeof(old_val),
        .old_value = old_val,
        .new_value_len = sizeof(new_val),
        .new_value = new_val
    };
    // Serialize
    uint8_t buffer[1024];
    ssize_t written = serialize_update_record(&original, buffer, sizeof(buffer));
    assert(written > 0);
    // Deserialize
    UpdateRecord parsed = {0};
    ssize_t consumed = deserialize_update_record(buffer, written, &parsed);
    assert(consumed == written);
    // Verify header
    assert(parsed.header.lsn == original.header.lsn);
    assert(parsed.header.type == original.header.type);
    assert(parsed.header.txn_id == original.header.txn_id);
    assert(parsed.header.prev_lsn == original.header.prev_lsn);
    // Verify payload
    assert(parsed.page_id == original.page_id);
    assert(parsed.offset == original.offset);
    assert(parsed.old_value_len == original.old_value_len);
    assert(memcmp(parsed.old_value, original.old_value, original.old_value_len) == 0);
    assert(parsed.new_value_len == original.new_value_len);
    assert(memcmp(parsed.new_value, original.new_value, original.new_value_len) == 0);
    // Clean up
    free(parsed.old_value);
    free(parsed.new_value);
    printf("Round-trip test passed!\n");
}
```
## Transaction API: The Public Interface

![Transaction State Machine with Log Records](./diagrams/diag-txn-state-machine.svg)

Now that we understand log records, let's build the transaction API that generates them. This is what the rest of your database will call.
### API Design
```c
// Opaque transaction handle
typedef uint64_t TxnId;
// Transaction manager (owns the log)
typedef struct TransactionManager TransactionManager;
// Create a new transaction manager with log file at given path
TransactionManager* txn_manager_create(const char* log_path);
// Destroy the transaction manager (flushes any pending writes)
void txn_manager_destroy(TransactionManager* mgr);
// Begin a new transaction, returns transaction ID
TxnId begin_txn(TransactionManager* mgr);
// Log a write operation (does NOT modify data — just logs it)
// Returns 0 on success, -1 on error
int write_txn(TransactionManager* mgr, TxnId txn_id,
              uint64_t page_id, uint32_t offset,
              const void* old_val, size_t old_len,
              const void* new_val, size_t new_len);
// Commit a transaction (writes COMMIT record, flushes log)
// Returns 0 on success, -1 on error
int commit_txn(TransactionManager* mgr, TxnId txn_id);
// Abort a transaction (writes CLRs for all operations, then ABORT record)
// Returns 0 on success, -1 on error
int abort_txn(TransactionManager* mgr, TxnId txn_id);
```
### Transaction State Tracking
You need to track active transactions and their state:
```c
typedef enum {
    TXN_ACTIVE,
    TXN_COMMITTED,
    TXN_ABORTED
} TxnStatus;
typedef struct {
    TxnId txn_id;
    TxnStatus status;
    uint64_t first_lsn;   // LSN of BEGIN record
    uint64_t last_lsn;    // LSN of most recent record (for prev_lsn chain)
} TransactionEntry;
typedef struct {
    TransactionEntry* entries;
    size_t capacity;
    size_t count;
} TransactionTable;
```
### Transaction Manager Implementation
```c
struct TransactionManager {
    int log_fd;                 // File descriptor for log file
    uint64_t next_lsn;          // Next LSN to assign
    TransactionTable txn_table; // Active transactions
    pthread_mutex_t lock;       // Protects all state
};
TransactionManager* txn_manager_create(const char* log_path) {
    TransactionManager* mgr = calloc(1, sizeof(TransactionManager));
    if (!mgr) return NULL;
    // Open log file for append-only writes
    mgr->log_fd = open(log_path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (mgr->log_fd < 0) {
        free(mgr);
        return NULL;
    }
    mgr->next_lsn = 1;  // LSN 0 is reserved/invalid
    // Initialize transaction table
    mgr->txn_table.capacity = 1024;
    mgr->txn_table.entries = calloc(mgr->txn_table.capacity, sizeof(TransactionEntry));
    if (!mgr->txn_table.entries) {
        close(mgr->log_fd);
        free(mgr);
        return NULL;
    }
    pthread_mutex_init(&mgr->lock, NULL);
    return mgr;
}
```
### begin_txn: Starting a Transaction
```c
TxnId begin_txn(TransactionManager* mgr) {
    pthread_mutex_lock(&mgr->lock);
    // Allocate transaction ID
    TxnId txn_id = mgr->txn_table.count + 1;  // Simple allocation
    // Create BEGIN record
    BeginRecord rec = {
        .header = {
            .lsn = mgr->next_lsn++,
            .type = RECORD_BEGIN,
            .txn_id = txn_id,
            .prev_lsn = 0,
            .length = 20  // Header (16) + CRC (4)
        }
    };
    // Serialize and write
    uint8_t buffer[20];
    ssize_t written = serialize_begin_record(&rec, buffer, sizeof(buffer));
    if (written != 20) {
        pthread_mutex_unlock(&mgr->lock);
        return 0;  // Invalid txn_id on error
    }
    if (write(mgr->log_fd, buffer, written) != written) {
        pthread_mutex_unlock(&mgr->lock);
        return 0;
    }
    // Add to transaction table
    TransactionEntry* entry = &mgr->txn_table.entries[mgr->txn_table.count++];
    entry->txn_id = txn_id;
    entry->status = TXN_ACTIVE;
    entry->first_lsn = rec.header.lsn;
    entry->last_lsn = rec.header.lsn;
    pthread_mutex_unlock(&mgr->lock);
    return txn_id;
}
```
### write_txn: Logging a Modification
```c
int write_txn(TransactionManager* mgr, TxnId txn_id,
              uint64_t page_id, uint32_t offset,
              const void* old_val, size_t old_len,
              const void* new_val, size_t new_len) {
    pthread_mutex_lock(&mgr->lock);
    // Find transaction in table
    TransactionEntry* entry = find_txn(&mgr->txn_table, txn_id);
    if (!entry || entry->status != TXN_ACTIVE) {
        pthread_mutex_unlock(&mgr->lock);
        return -1;
    }
    // Create UPDATE record
    UpdateRecord rec = {
        .header = {
            .lsn = mgr->next_lsn++,
            .type = RECORD_UPDATE,
            .txn_id = txn_id,
            .prev_lsn = entry->last_lsn,  // Link to previous record
            .length = 0  // Computed during serialization
        },
        .page_id = page_id,
        .offset = offset,
        .old_value_len = old_len,
        .old_value = (void*)old_val,
        .new_value_len = new_len,
        .new_value = (void*)new_val
    };
    // Serialize (compute length and CRC)
    uint8_t* buffer = malloc(65536);  // Max record size
    if (!buffer) {
        pthread_mutex_unlock(&mgr->lock);
        return -1;
    }
    ssize_t written = serialize_update_record(&rec, buffer, 65536);
    if (written < 0) {
        free(buffer);
        pthread_mutex_unlock(&mgr->lock);
        return -1;
    }
    // Write to log
    if (write(mgr->log_fd, buffer, written) != written) {
        free(buffer);
        pthread_mutex_unlock(&mgr->lock);
        return -1;
    }
    // Update transaction's last_lsn
    entry->last_lsn = rec.header.lsn;
    free(buffer);
    pthread_mutex_unlock(&mgr->lock);
    return 0;
}
```
### commit_txn: Durability Point
```c
int commit_txn(TransactionManager* mgr, TxnId txn_id) {
    pthread_mutex_lock(&mgr->lock);
    TransactionEntry* entry = find_txn(&mgr->txn_table, txn_id);
    if (!entry || entry->status != TXN_ACTIVE) {
        pthread_mutex_unlock(&mgr->lock);
        return -1;
    }
    // Create COMMIT record
    CommitRecord rec = {
        .header = {
            .lsn = mgr->next_lsn++,
            .type = RECORD_COMMIT,
            .txn_id = txn_id,
            .prev_lsn = entry->last_lsn,
            .length = 20
        }
    };
    // Serialize and write
    uint8_t buffer[20];
    ssize_t written = serialize_commit_record(&rec, buffer, sizeof(buffer));
    if (write(mgr->log_fd, buffer, written) != written) {
        pthread_mutex_unlock(&mgr->lock);
        return -1;
    }
    // CRITICAL: fsync before acknowledging commit
    // This is the durability guarantee
    if (fsync(mgr->log_fd) < 0) {
        pthread_mutex_unlock(&mgr->lock);
        return -1;
    }
    // Mark transaction committed
    entry->status = TXN_COMMITTED;
    entry->last_lsn = rec.header.lsn;
    pthread_mutex_unlock(&mgr->lock);
    return 0;
}
```
The `fsync` call is the heart of durability. Until this call returns, the transaction is not committed — a crash would lose it. After this call returns, the transaction is durable: even if the power fails immediately, recovery will find the COMMIT record and preserve the transaction's effects.
### abort_txn: Undo with CLRs
Abort is more complex. We need to:
1. Find all UPDATE records for this transaction
2. Undo them in reverse order (follow `prev_lsn` chain)
3. Write a CLR for each undo operation
4. Write an ABORT record when done
```c
int abort_txn(TransactionManager* mgr, TxnId txn_id) {
    pthread_mutex_lock(&mgr->lock);
    TransactionEntry* entry = find_txn(&mgr->txn_table, txn_id);
    if (!entry || entry->status != TXN_ACTIVE) {
        pthread_mutex_unlock(&mgr->lock);
        return -1;
    }
    // Mark as aborting (prevents new operations)
    entry->status = TXN_ABORTED;
    // Undo chain: follow prev_lsn backward
    uint64_t current_lsn = entry->last_lsn;
    uint64_t undo_next_lsn = 0;
    // We need to read log records to find UPDATEs
    // For now, assume we have a function to read a record by LSN
    // In practice, this requires scanning or an index
    while (current_lsn > entry->first_lsn) {
        // Read record at current_lsn
        LogRecordHeader header;
        if (read_record_header_at_lsn(mgr, current_lsn, &header) < 0) {
            break;  // Error reading log
        }
        if (header.type == RECORD_UPDATE) {
            UpdateRecord update;
            if (read_update_record_at_lsn(mgr, current_lsn, &update) < 0) {
                break;
            }
            // Write CLR for this undo
            ClrRecord clr = {
                .header = {
                    .lsn = mgr->next_lsn++,
                    .type = RECORD_CLR,
                    .txn_id = txn_id,
                    .prev_lsn = entry->last_lsn,  // Link to last record
                    .length = 0
                },
                .page_id = update.page_id,
                .offset = update.offset,
                .undo_data_len = update.old_value_len,
                .undo_data = update.old_value,  // Undo = write old value
                .undo_next_lsn = update.header.prev_lsn  // Next to undo
            };
            uint8_t* buffer = malloc(65536);
            ssize_t written = serialize_clr_record(&clr, buffer, 65536);
            write(mgr->log_fd, buffer, written);
            free(buffer);
            entry->last_lsn = clr.header.lsn;
            undo_next_lsn = clr.undo_next_lsn;
            free_update_record(&update);
        }
        current_lsn = header.prev_lsn;  // Follow chain backward
    }
    // Write ABORT record
    AbortRecord abort_rec = {
        .header = {
            .lsn = mgr->next_lsn++,
            .type = RECORD_ABORT,
            .txn_id = txn_id,
            .prev_lsn = entry->last_lsn,
            .length = 20
        }
    };
    uint8_t buffer[20];
    ssize_t written = serialize_abort_record(&abort_rec, buffer, sizeof(buffer));
    write(mgr->log_fd, buffer, written);
    fsync(mgr->log_fd);  // Ensure abort is durable
    entry->last_lsn = abort_rec.header.lsn;
    pthread_mutex_unlock(&mgr->lock);
    return 0;
}
```
## The Binary Format in Detail
Let's look at the exact byte layout for each record type. This is what appears on disk.
### Header Format (16 bytes, always present)
```
Offset  Size  Field
------  ----  -----
0       8     lsn (little-endian uint64)
8       4     type (little-endian uint32)
12      4     txn_id (little-endian uint32)
16      8     prev_lsn (little-endian uint64)
24      4     length (little-endian uint32)
28      -     (payload follows)
```
### BEGIN Record (20 bytes total)
```
Offset  Size  Field
------  ----  -----
0       28    header
28      4     crc32
= 32 bytes total? No — let me recalculate.
Actually:
Header = 8 + 4 + 4 + 8 + 4 = 28 bytes
BEGIN has no payload
CRC = 4 bytes
Total = 32 bytes
```
Wait, let me re-examine the header struct I defined:
```c
typedef struct {
    uint64_t lsn;        // 8 bytes
    uint32_t type;       // 4 bytes
    uint32_t txn_id;     // 4 bytes
    uint64_t prev_lsn;   // 8 bytes
    uint32_t length;     // 4 bytes
} LogRecordHeader;       // = 28 bytes
```
So BEGIN record = 28 (header) + 4 (CRC) = 32 bytes.
### UPDATE Record Format
```
Offset  Size        Field
------  ----        -----
0       28          header
28      8           page_id
36      4           offset
40      4           old_value_len
44      old_len     old_value
44+old  4           new_value_len
48+old  new_len     new_value
48+o+n  4           crc32
```
Total = 28 + 8 + 4 + 4 + old_len + 4 + new_len + 4 = 48 + old_len + new_len
### CLR Record Format
```
Offset  Size        Field
------  ----        -----
0       28          header
28      8           page_id
36      4           offset
40      4           undo_data_len
44      undo_len    undo_data
44+undo 8           undo_next_lsn
52+undo 4           crc32
```
Total = 28 + 8 + 4 + 4 + undo_len + 8 + 4 = 56 + undo_len
## Testing Your Implementation
### Unit Tests for Record Serialization
```c
void test_begin_record(void) {
    BeginRecord rec = {
        .header = {
            .lsn = 1,
            .type = RECORD_BEGIN,
            .txn_id = 42,
            .prev_lsn = 0,
            .length = 32
        }
    };
    uint8_t buf[32];
    ssize_t written = serialize_begin_record(&rec, buf, sizeof(buf));
    assert(written == 32);
    // Verify LSN at offset 0 (little-endian)
    assert(buf[0] == 1);
    assert(buf[1] == 0);
    // ... verify other fields
    // Verify CRC (last 4 bytes)
    uint32_t computed = compute_crc(buf, 28);
    uint32_t stored = read_le32(buf + 28);
    assert(computed == stored);
    printf("BEGIN record test passed!\n");
}
void test_variable_length_update(void) {
    uint8_t old_val[] = "hello";
    uint8_t new_val[] = "goodbye";
    UpdateRecord rec = {
        .header = { .lsn = 100, .type = RECORD_UPDATE, .txn_id = 1, .prev_lsn = 50 },
        .page_id = 7,
        .offset = 16,
        .old_value_len = 5,
        .old_value = old_val,
        .new_value_len = 7,
        .new_value = new_val
    };
    uint8_t buf[256];
    ssize_t written = serialize_update_record(&rec, buf, sizeof(buf));
    assert(written == 48 + 5 + 7);  // Header + payload + CRC
    // Deserialize and verify
    UpdateRecord parsed = {0};
    ssize_t consumed = deserialize_update_record(buf, written, &parsed);
    assert(consumed == written);
    assert(parsed.old_value_len == 5);
    assert(memcmp(parsed.old_value, "hello", 5) == 0);
    assert(parsed.new_value_len == 7);
    assert(memcmp(parsed.new_value, "goodbye", 7) == 0);
    free(parsed.old_value);
    free(parsed.new_value);
    printf("Variable-length UPDATE test passed!\n");
}
```
### Integration Test: Full Transaction Lifecycle
```c
void test_transaction_lifecycle(void) {
    // Create manager with temporary log file
    TransactionManager* mgr = txn_manager_create("/tmp/test_log.bin");
    assert(mgr != NULL);
    // Begin transaction
    TxnId txn = begin_txn(mgr);
    assert(txn > 0);
    // Log some writes
    uint8_t old1[] = {0x00, 0x00};
    uint8_t new1[] = {0xDE, 0xAD};
    assert(write_txn(mgr, txn, 1, 0, old1, 2, new1, 2) == 0);
    uint8_t old2[] = {0x11, 0x11};
    uint8_t new2[] = {0xBE, 0xEF};
    assert(write_txn(mgr, txn, 2, 0, old2, 2, new2, 2) == 0);
    // Commit
    assert(commit_txn(mgr, txn) == 0);
    // Verify log file contains expected records
    // (Read file and parse records)
    txn_manager_destroy(mgr);
    printf("Transaction lifecycle test passed!\n");
}
```
### CRC Corruption Detection Test
```c
void test_crc_detects_corruption(void) {
    UpdateRecord rec = { /* ... */ };
    uint8_t buf[256];
    ssize_t written = serialize_update_record(&rec, buf, sizeof(buf));
    // Deserialize should succeed
    UpdateRecord parsed = {0};
    assert(deserialize_update_record(buf, written, &parsed) == written);
    free(parsed.old_value);
    free(parsed.new_value);
    // Corrupt a byte in the middle
    buf[20] ^= 0xFF;
    // Deserialize should now fail (CRC mismatch)
    assert(deserialize_update_record(buf, written, &parsed) < 0);
    printf("CRC corruption detection test passed!\n");
}
```
## Common Pitfalls
### 1. Forgetting to Link prev_lsn
The `prev_lsn` field must point to the previous record of the *same transaction*, not the previous record in the log overall. If transaction T1 has records at LSN 100, 150, 200 and transaction T2 has a record at LSN 175, then T1's record at 200 should have `prev_lsn = 150`, not `prev_lsn = 175`.
### 2. Confusing undo_next_lsn with prev_lsn
In a CLR record:
- `header.prev_lsn` = previous record in this transaction (the UPDATE being undone)
- `undo_next_lsn` = next record to undo (the `prev_lsn` of the UPDATE being undone)
If UPDATE has `prev_lsn = 150`, and we're undoing it with a CLR, the CLR has:
- `prev_lsn` = LSN of last record before this CLR (could be another CLR or the UPDATE itself)
- `undo_next_lsn` = 150 (skip the undone UPDATE, go to 150 next)
### 3. Not CRC-ing the Entire Record
The CRC must cover everything: header, payload, everything except the CRC itself. A common mistake is to CRC only the payload, missing header corruption.
### 4. Memory Leaks in Variable-Length Records
When deserializing, you `malloc` memory for `old_value` and `new_value`. The caller must `free` these. Document this clearly and provide a `free_update_record()` helper.
### 5. Off-By-One in Length Calculations
The `length` field in the header includes:
- The header itself (28 bytes)
- All payload fields
- The CRC (4 bytes)
A 32-byte BEGIN record has `length = 32`, not 28.
## Knowledge Cascade
You've now mastered log record formats. Here's where this knowledge connects:
**→ ARIES Recovery (Milestone 3)**: Every record type you defined plays a role. BEGIN marks transaction start. UPDATE provides before/after images for redo/undo. COMMIT and ABORT determine transaction fate. CLRs enable crash-safe undo. The `prev_lsn` chain is how undo finds work to do.
**→ Buffer Pool Integration (Future)**: The pageLSN stored in each page's header is compared against log record LSNs during redo. If `pageLSN >= record.lsn`, the change is already on disk — skip it. This idempotency is crucial for correctness.
**→ Distributed Consensus (Cross-Domain)**: The techniques you learned — LSNs as monotonic IDs, prev_lsn chains as logical logs, CRC for integrity — appear in Raft's log replication, Kafka's offset tracking, and Paxos's ballot numbers. The patterns repeat.
**→ Network Protocols (Cross-Domain)**: Your binary serialization with length-prefixed fields and trailing checksums is identical to TCP segment format, TLS records, and countless wire protocols. You've learned a fundamental systems technique.
**→ Compiler AST Serialization (Cross-Domain)**: Variable-length records with discriminated unions (the `type` field determining payload structure) is exactly how compilers serialize abstract syntax trees. The same patterns apply.
---

![LSN Tracking: The Connective Tissue](./diagrams/diag-lsn-tracking-everywhere.svg)

---
[[CRITERIA_JSON: {"milestone_id": "wal-impl-m1", "criteria": ["Log record header contains LSN (8-byte little-endian, monotonically increasing), type (4-byte), txn_id (4-byte), prev_lsn (8-byte, links records within same transaction), and length (4-byte, includes header + payload + CRC)", "Six record types implemented: BEGIN (transaction start, 32 bytes), UPDATE (page modification with before/after images), COMMIT (durable commit marker), ABORT (transaction termination after undo), CLR (compensation record with undo_next_lsn field), CHECKPOINT (transaction table and dirty page table snapshot)", "UPDATE records serialize page_id (8-byte), offset (4-byte), old_value_len + old_value (length-prefixed), new_value_len + new_value (length-prefixed)", "CLR records contain undo_next_lsn field that points to the next record to undo (prev_lsn of the UPDATE being undone), enabling crash-safe recovery resumption", "CRC32 checksum computed over header + payload, stored as final 4 bytes of each record, verified during deserialization", "Transaction API: begin_txn() writes BEGIN and returns txn_id, write_txn() writes UPDATE with linked prev_lsn, commit_txn() writes COMMIT and fsyncs, abort_txn() writes CLRs following prev_lsn chain then writes ABORT", "Transaction table tracks active transactions with txn_id, status (ACTIVE/COMMITTED/ABORTED), first_lsn, and last_lsn for prev_lsn linking", "Serialization/deserialization achieves byte-perfect round-trip fidelity for all record types with variable-length fields", "Unit tests verify: BEGIN/UPDATE/CLR serialization format, variable-length field handling, CRC corruption detection, prev_lsn chain correctness, full transaction lifecycle"]}]
<!-- END_MS -->


<!-- MS_ID: wal-impl-m2 -->
<!-- MS_ID: wal-impl-m2 -->
# Milestone 2: Log Writer with Group Commit
## The fsync Problem: Why Your Database is Slow
You built the log record format. You can serialize BEGIN, UPDATE, COMMIT records. You wrote `commit_txn()` to call `fsync()`. You're done, right?
Let's talk about what happens when you actually run this code.
You call `fsync()` on your log file. What does the operating system do?
```
Your process
     │
     ▼
┌─────────────────┐
│  write() syscall │  ~1 microsecond (copies to page cache)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   fsync() call   │  ← This is where time disappears
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│              THE fsync BLACK BOX                     │
│                                                      │
│  1. Flush OS page cache to device    ~0.1ms         │
│  2. Tell disk controller "sync now"  ~0.01ms        │
│  3. Wait for controller to respond   ~???           │
│                                                      │
│     On HDD: Seek (5-10ms) + Rotation (0-8ms)        │
│             = 5-18ms per fsync                       │
│                                                      │
│     On SSD: FTL commit, capacitor drain             │
│             = 0.5-3ms per fsync                      │
│                                                      │
│     On "enterprise SSD" with write cache:            │
│             = 0.01-0.1ms (but is it really durable?) │
└─────────────────────────────────────────────────────┘
```

![fsync Latency: Where Time Goes](./diagrams/diag-fsync-latency-breakdown.svg)


> **🔑 Foundation: fsync vs fdatasync vs O_DSYNC**
> 
> ## What It IS
These are three mechanisms for ensuring data durability on disk in Linux/POSIX systems. They differ in what they guarantee and how they achieve it.
**`fsync(fd)`** — The full guarantee. Ensures both file data *and* metadata (timestamps, file size, permissions, etc.) are written to stable storage. Returns only when the physical disk has confirmed the write.
**`fdatasync(fd)`** — The data-only guarantee. Ensures file *content* is durable, but skips metadata that isn't critical for data integrity. Timestamps? Ignored. File size if it changed? That's synced (needed to know the file ends correctly).
**`O_DSYNC`** (file open flag) — A declarative version of fdatasync. When you open a file with this flag, every `write()` behaves as if followed by `fdatasync()`. The kernel handles it, potentially more efficiently.
## WHY You Need It Right Now
When you write to a file, you're actually writing to:
1. The application's buffer (if buffered)
2. The kernel's page cache
3. The disk's internal cache
4. Finally, the physical platter/NAND cells
A simple `write()` returning success means: "the kernel accepted your data." Crash right after? Data gone.
If you're building:
- **Databases** — You need fsync on transaction logs
- **Message queues** — You need durability guarantees on enqueue
- **Configuration files** — fdatasync is usually sufficient
- **High-throughput logging** — O_DSYNC or periodic fdatasync
Choosing wrong means either corrupted state after crash (too weak) or terrible performance (too strong).
## ONE Key Insight
**Think in terms of "durability scope" vs "durability timing."**
```
fsync      = Data + All Metadata,  On-demand
fdatasync  = Data + Critical Meta, On-demand
O_DSYNC    = Data + Critical Meta, Per-write (automatic)
```
**The performance gap is real:**
- fsync can trigger two separate disk flushes (data, then metadata)
- fdatasync often coalesces to one flush
- O_DSYNC lets the kernel schedule writes optimally, but you lose control over batching
**Mental model:** `fsync` is "I need to survive a power outage with perfect state." `fdatasync` is "I need the *content* to survive; I can rebuild the rest." `O_DSYNC` is "I never want to think about syncing again, just make every write durable."
**Practical rule of thumb:** Use fdatasync for application data. Use fsync only when you're modifying metadata that matters (file size changes, permissions, etc.) or when you don't know what changed. Use O_DSYNC when you want simplicity and every write must survive.

**The numbers you need to know:**
| Storage Type | fsync Latency | Commits/Second (Max) |
|--------------|---------------|---------------------|
| HDD (7200 RPM) | 10-15ms | ~70-100 |
| SATA SSD | 1-3ms | ~300-1000 |
| NVMe SSD | 0.1-1ms | ~1000-10000 |
| RAM | N/A (no fsync) | Millions |
Here's the brutal math: if each commit requires one fsync, and fsync takes 10ms, your database can do **100 transactions per second maximum**. Not per client — total. Every transaction blocks on disk I/O.
This isn't a bug in your code. It's physics.
## The Solution: Amortize the Cost
If fsync costs 10ms, and you can't make it faster, what can you do?
**Do it less often.**
What if, instead of fsync-ing after every commit, you waited a tiny bit and let multiple commits batch together? One fsync for 10 commits means each commit "costs" 1ms instead of 10ms.
This is **group commit**, and it's not an optimization — it's how every production database achieves reasonable throughput. PostgreSQL does it. MySQL does it. SQLite in WAL mode does it. Oracle does it.
The insight: fsync cost is (almost) fixed whether you're syncing 1 record or 100 records. The I/O path has fixed overheads that dominate for small writes.
```
Naive approach: 10 commits = 10 fsyncs = 100ms
Group commit:   10 commits = 1 fsync  = 10ms
```

![Group Commit: Before vs After](./diagrams/diag-group-commit-batching.svg)

But here's the tension: waiting to batch means latency. If you wait 10ms to accumulate commits, every commit takes at least 10ms even if the disk is idle. You're trading latency for throughput.
The art of group commit is balancing this tradeoff: batch enough to amortize fsync cost, but don't wait so long that you hurt latency unnecessarily.
## Append-Only Writes: The Log's Fundamental Shape
Before we implement group commit, let's understand the simpler requirement: **sequential append-only writes**.
A write-ahead log has one crucial property: **you never modify existing data**. You only ever append new records to the end. This isn't just a design choice — it's what "log" means.

![Append-Only Log File Growth](./diagrams/diag-append-only-writes.svg)

Why does this matter?
1. **No seeks**: Appending to the end of a file is a sequential write. No disk head movement (on HDDs), no random I/O patterns. The disk can stream data at full bandwidth.
2. **Simpler recovery**: If you never overwrite, you never have to worry about "which version is correct?" Recovery scans forward from the beginning (or checkpoint), applying records in order.
3. **Concurrency is simpler**: Multiple writers all want to write to the same place (the end). This is a single contention point, but it's a simple one: serialize the appends.
### The LSN-as-Offset Design
A beautiful design pattern: **make LSN equal to byte offset in the log file**.
```c
// LSN 0 is invalid/reserved
// LSN 100 means "byte offset 100 in the log file"
// To read record with LSN 100: seek to offset 100, read
```
This means:
- No separate LSN index needed
- Finding a record by LSN is O(1): just seek
- LSNs are naturally monotonically increasing (file only grows)
```c
typedef struct {
    int log_fd;           // File descriptor for log file
    uint64_t next_lsn;    // Also equals current file offset
} LogManager;
uint64_t allocate_lsn(LogManager* log) {
    uint64_t lsn = log->next_lsn;
    // LSN will be incremented by record size after write
    return lsn;
}
```
## Concurrent Writers: The Serialization Problem
You have multiple threads, each handling a transaction. They all need to write log records. What happens if they write simultaneously?

![Concurrent Writer Serialization](./diagrams/diag-concurrent-writer-serialization.svg)

**Scenario: Two threads append simultaneously**
```
Thread A: write record A (50 bytes)
Thread B: write record B (30 bytes)
Without serialization:
- Thread A seeks to offset 1000, starts writing
- Thread B seeks to offset 1000, starts writing
- Result: corrupted interleaved data
With serialization:
- Thread A grabs lock, writes at offset 1000-1049
- Thread B waits, then writes at offset 1050-1079
- Result: clean sequential records
```
The solution is a mutex around the append operation:
```c
typedef struct {
    int log_fd;
    uint64_t next_lsn;
    pthread_mutex_t append_lock;  // Serializes appends
} LogManager;
int append_record(LogManager* log, const uint8_t* data, size_t len) {
    pthread_mutex_lock(&log->append_lock);
    // Allocate LSN (equals current offset)
    uint64_t record_lsn = log->next_lsn;
    // Write record
    ssize_t written = write(log->log_fd, data, len);
    if (written != (ssize_t)len) {
        pthread_mutex_unlock(&log->append_lock);
        return -1;
    }
    // Update next LSN
    log->next_lsn += len;
    pthread_mutex_unlock(&log->append_lock);
    return record_lsn;
}
```
The lock is held only during the `write()` syscall, which is fast (microseconds) since it just copies to the OS page cache. The expensive `fsync()` happens outside this lock.
## Group Commit: The Protocol
Now we get to the heart of this milestone. How do you batch multiple commits into one fsync?
The key insight: **decouple "record written" from "record durable"**.
A transaction's COMMIT record is written to the log immediately. But we don't fsync right away. Instead, we add the transaction to a "waiting for sync" list. A separate thread (or the first waiter) periodically fsyncs, and all waiting transactions are notified.

![Group Commit Leader/Follower Protocol](./diagrams/diag-group-commit-state-machine.svg)

### The Leader/Follower Model
```c
typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t sync_complete;   // Signaled when fsync done
    uint64_t pending_sync_lsn;      // All records up to this LSN need sync
    int sync_in_progress;           // Is someone currently doing fsync?
    uint64_t last_synced_lsn;       // Last LSN known to be on disk
    int log_fd;
} GroupCommitManager;
```
**Transaction commit flow:**
```c
int commit_with_group_commit(GroupCommitManager* gcm, uint64_t commit_lsn) {
    pthread_mutex_lock(&gcm->lock);
    // Update the maximum LSN needing sync
    if (commit_lsn > gcm->pending_sync_lsn) {
        gcm->pending_sync_lsn = commit_lsn;
    }
    // Check if someone is already doing fsync
    if (gcm->sync_in_progress) {
        // We're a FOLLOWER: wait for leader to complete sync
        uint64_t my_lsn = commit_lsn;
        while (gcm->last_synced_lsn < my_lsn) {
            pthread_cond_wait(&gcm->sync_complete, &gcm->lock);
        }
        pthread_mutex_unlock(&gcm->lock);
        return 0;  // Commit is now durable
    }
    // We're the LEADER: do the fsync
    gcm->sync_in_progress = 1;
    uint64_t sync_target = gcm->pending_sync_lsn;
    pthread_mutex_unlock(&gcm->lock);  // Release lock during fsync
    // Do the expensive fsync (millisecond-scale)
    fsync(gcm->log_fd);
    // Wake up all waiters
    pthread_mutex_lock(&gcm->lock);
    gcm->last_synced_lsn = sync_target;
    gcm->sync_in_progress = 0;
    pthread_cond_broadcast(&gcm->sync_complete);  // Wake all followers
    pthread_mutex_unlock(&gcm->lock);
    return 0;
}
```
**What just happened:**
1. Transaction A commits, writes COMMIT record at LSN 1000
2. Transaction A becomes LEADER, starts fsync
3. Transaction B commits, writes COMMIT record at LSN 1050
4. Transaction B sees sync_in_progress, becomes FOLLOWER, waits on condition variable
5. Transaction C commits, writes COMMIT record at LSN 1100
6. Transaction C also becomes FOLLOWER, waits
7. Leader's fsync completes (covers LSN 1000-1100, all written before fsync)
8. Leader broadcasts to all waiters
9. Transactions A, B, C all return successfully from commit
**One fsync, three commits.**
### The Timing Window
There's a subtle issue: when does the leader decide to sync? If the leader fsyncs immediately, there's no batching. If the leader waits too long, latency suffers.
Common approaches:
1. **Immediate with timeout**: Leader starts fsync immediately, but waits up to N microseconds for more transactions to join. If the timeout expires, fsync whatever's there.
2. **Batch size threshold**: Wait until N transactions are waiting, then fsync.
3. **Hybrid**: Wait for either timeout or batch size, whichever comes first.
```c
// Hybrid approach: wait up to 1ms or 10 transactions
#define GROUP_COMMIT_TIMEOUT_US  1000
#define GROUP_COMMIT_BATCH_SIZE  10
int leader_wait_for_batch(GroupCommitManager* gcm) {
    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_nsec += GROUP_COMMIT_TIMEOUT_US * 1000;
    if (deadline.tv_nsec >= 1000000000) {
        deadline.tv_sec++;
        deadline.tv_nsec -= 1000000000;
    }
    int waiters = 0;
    while (waiters < GROUP_COMMIT_BATCH_SIZE) {
        if (pthread_cond_timedwait(&gcm->batch_ready, &gcm->lock, &deadline) == ETIMEDOUT) {
            break;
        }
        waiters++;
    }
    return waiters;
}
```
### Group Commit Benchmark
The 5x improvement requirement isn't arbitrary. Let's see why it happens.
**Without group commit (per-commit fsync):**
```
100 transactions, each commits separately
Each commit: ~2ms fsync (SSD)
Total time: 100 × 2ms = 200ms
Throughput: 500 commits/second
```
**With group commit (batched fsync):**
```
100 transactions commit concurrently
All batched into single fsync
Each batch: ~2ms fsync + 0.5ms overhead
Assuming 10 batches for 100 transactions
Total time: 10 × 2.5ms = 25ms
Throughput: 4000 commits/second
```
That's an **8x improvement**. The exact number depends on your concurrency level and fsync latency, but 5x is a reasonable minimum target.
```c
void benchmark_group_commit(void) {
    const int NUM_TRANSACTIONS = 100;
    const int NUM_THREADS = 10;
    // Without group commit (fsync per commit)
    uint64_t start = get_time_us();
    run_transactions_no_group_commit(NUM_TRANSACTIONS, NUM_THREADS);
    uint64_t no_group_time = get_time_us() - start;
    // With group commit
    start = get_time_us();
    run_transactions_with_group_commit(NUM_TRANSACTIONS, NUM_THREADS);
    uint64_t group_time = get_time_us() - start;
    double speedup = (double)no_group_time / group_time;
    printf("Without group commit: %lu us\n", no_group_time);
    printf("With group commit:    %lu us\n", group_time);
    printf("Speedup: %.1fx\n", speedup);
    assert(speedup >= 5.0);  // Must achieve at least 5x
}
```
## Torn Write Detection: When Crashes Happen Mid-Write
What happens if the system crashes while writing a log record?

![Torn Write Detection via CRC](./diagrams/diag-torn-write-detection.svg)

```
Record at LSN 1000 (50 bytes):
- Header written: bytes 1000-1027 ✓
- Payload written: bytes 1028-1045 ✓
- CRASH!
- CRC not written: bytes 1046-1049 ✗
```
On recovery, you read the record at LSN 1000. The header says `length = 50`. You read 50 bytes. You compute CRC over bytes 1000-1045. The stored CRC at bytes 1046-1049 is garbage (whatever was on disk before). **CRC mismatch → record is invalid.**
This is **torn write detection**, and your CRC from Milestone 1 already provides it. During recovery:
```c
int scan_log_for_recovery(LogManager* log) {
    uint8_t buffer[MAX_RECORD_SIZE];
    uint64_t offset = 0;
    while (1) {
        // Read header first
        ssize_t n = pread(log->log_fd, buffer, HEADER_SIZE, offset);
        if (n < HEADER_SIZE) {
            // End of file or incomplete header
            break;
        }
        LogRecordHeader header;
        parse_header(buffer, &header);
        // Read full record
        n = pread(log->log_fd, buffer, header.length, offset);
        if (n < header.length) {
            // Incomplete record — torn write
            printf("Torn write detected at offset %lu, truncating\n", offset);
            ftruncate(log->log_fd, offset);  // Truncate file
            break;
        }
        // Verify CRC
        uint32_t stored_crc = read_le32(buffer + header.length - 4);
        uint32_t computed_crc = compute_crc(buffer, header.length - 4);
        if (stored_crc != computed_crc) {
            printf("CRC mismatch at offset %lu, truncating\n", offset);
            ftruncate(log->log_fd, offset);
            break;
        }
        // Record is valid, add to recovery state
        process_record_for_recovery(buffer, header.length);
        offset += header.length;
    }
    return 0;
}
```
### The Atomic Write Myth
Some developers hope that small writes (less than 512 bytes, or 4KB, or some "block size") are atomic. **This is false.**
- The disk sector size (512 bytes or 4KB) doesn't guarantee atomicity on power failure
- The OS page cache can have partial writes
- SSDs can have partial page writes during power loss
**Your only guarantee is: after fsync returns, the data is durable.** Anything written but not fsynced might be partial or missing after a crash. The CRC catches partial writes; the fsync guarantees completeness.
## Write Buffering: Reducing System Calls
Each `write()` syscall has overhead (context switch, kernel entry). For high-throughput workloads, you want to minimize syscalls.
A **write buffer** accumulates records in memory, then flushes them in batches:
```c
typedef struct {
    uint8_t* buffer;
    size_t capacity;
    size_t used;
    uint64_t first_lsn_in_buffer;  // LSN of first buffered record
    uint64_t last_lsn_in_buffer;   // LSN of last buffered record
} WriteBuffer;
typedef struct {
    int log_fd;
    WriteBuffer write_buf;
    pthread_mutex_t lock;
} BufferedLogWriter;
```
### Buffering Strategy
```c
int buffered_append(BufferedLogWriter* writer, const uint8_t* data, size_t len, uint64_t lsn) {
    pthread_mutex_lock(&writer->lock);
    // If buffer is full, flush it first
    if (writer->write_buf.used + len > writer->write_buf.capacity) {
        flush_buffer(writer);
    }
    // Add to buffer
    memcpy(writer->write_buf.buffer + writer->write_buf.used, data, len);
    writer->write_buf.used += len;
    if (writer->write_buf.used == 0) {
        writer->write_buf.first_lsn_in_buffer = lsn;
    }
    writer->write_buf.last_lsn_in_buffer = lsn;
    pthread_mutex_unlock(&writer->lock);
    return 0;
}
int flush_buffer(BufferedLogWriter* writer) {
    if (writer->write_buf.used == 0) return 0;
    ssize_t written = write(writer->log_fd, 
                            writer->write_buf.buffer, 
                            writer->write_buf.used);
    if (written != (ssize_t)writer->write_buf.used) {
        return -1;
    }
    writer->write_buf.used = 0;
    return 0;
}
```
### The Critical Constraint: Commit Must Flush
You can buffer regular writes, but **COMMIT records must flush immediately** (or at least before acknowledging commit):
```c
int buffered_commit(BufferedLogWriter* writer, const uint8_t* commit_record, size_t len, uint64_t lsn) {
    pthread_mutex_lock(&writer->lock);
    // Add COMMIT record to buffer
    memcpy(writer->write_buf.buffer + writer->write_buf.used, commit_record, len);
    writer->write_buf.used += len;
    writer->write_buf.last_lsn_in_buffer = lsn;
    // MUST flush before returning
    // This is where durability is guaranteed
    flush_buffer(writer);
    fsync(writer->log_fd);  // Or let group commit handle this
    pthread_mutex_unlock(&writer->lock);
    return 0;
}
```
If you buffer a COMMIT without flushing, and the system crashes, the transaction is lost — even though the client was told it committed. This is a **durability violation**, the worst kind of database bug.
## Log Segment Rotation: Bounded Files
A log file that grows forever has problems:
- File system limits (max file size)
- Slow operations (stat, seek on huge files)
- No way to reclaim space for old, unneeded logs
The solution: **log segment rotation**. When the current segment reaches a size threshold, close it and start a new one.

![Log Segment Rotation](./diagrams/diag-log-segment-rotation.svg)

```c
#define DEFAULT_SEGMENT_SIZE (64 * 1024 * 1024)  // 64 MB
typedef struct {
    char base_path[256];     // e.g., "/data/log/wal"
    int current_fd;          // FD of active segment
    uint64_t current_size;   // Bytes written to current segment
    uint64_t segment_size_limit;
    uint32_t current_segment_id;
} LogSegmentManager;
int rotate_if_needed(LogSegmentManager* lsm) {
    if (lsm->current_size < lsm->segment_size_limit) {
        return 0;  // No rotation needed
    }
    // Close current segment
    fsync(lsm->current_fd);
    close(lsm->current_fd);
    // Open new segment
    lsm->current_segment_id++;
    char path[512];
    snprintf(path, sizeof(path), "%s.%06u", lsm->base_path, lsm->current_segment_id);
    lsm->current_fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (lsm->current_fd < 0) {
        return -1;
    }
    lsm->current_size = 0;
    return 0;
}
```
### Segment Naming Convention
```
/data/log/wal.000001  [LSN 0 - 67,108,863]
/data/log/wal.000002  [LSN 67,108,864 - 134,217,727]
/data/log/wal.000003  [LSN 134,217,728 - ...]
```
Segment ID in filename makes it easy to:
- List segments in order
- Find which segment contains a given LSN
- Delete old segments during log truncation (Milestone 4)
### LSN to Segment Mapping
With segments, LSN is no longer a simple file offset. You need to track which segment contains which LSN range:
```c
typedef struct {
    uint32_t segment_id;
    uint64_t start_lsn;    // First LSN in this segment
    uint64_t end_lsn;      // Last LSN in this segment (0 if active)
} SegmentInfo;
int find_segment_for_lsn(LogSegmentManager* lsm, uint64_t lsn, SegmentInfo* info) {
    // Scan segment metadata (could be in a separate index file)
    for (uint32_t seg_id = 1; seg_id <= lsm->current_segment_id; seg_id++) {
        if (get_segment_info(seg_id, info) < 0) continue;
        if (lsn >= info->start_lsn && 
            (info->end_lsn == 0 || lsn <= info->end_lsn)) {
            return 0;  // Found
        }
    }
    return -1;  // LSN not found
}
```
## Putting It All Together: The Complete Log Writer
Let's integrate all the pieces: append-only writes, concurrent serialization, group commit, write buffering, and segment rotation.
```c
typedef struct {
    // Segment management
    LogSegmentManager segments;
    // Write buffering
    WriteBuffer write_buffer;
    // Group commit
    GroupCommitManager group_commit;
    // Concurrency
    pthread_mutex_t append_lock;
    // LSN allocation
    uint64_t next_lsn;
} WalWriter;
WalWriter* wal_writer_create(const char* base_path) {
    WalWriter* w = calloc(1, sizeof(WalWriter));
    if (!w) return NULL;
    // Initialize segments
    strncpy(w->segments.base_path, base_path, sizeof(w->segments.base_path) - 1);
    w->segments.segment_size_limit = DEFAULT_SEGMENT_SIZE;
    w->segments.current_segment_id = 1;
    char path[512];
    snprintf(path, sizeof(path), "%s.000001", base_path);
    w->segments.current_fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (w->segments.current_fd < 0) {
        free(w);
        return NULL;
    }
    // Initialize write buffer (1MB)
    w->write_buffer.capacity = 1024 * 1024;
    w->write_buffer.buffer = malloc(w->write_buffer.capacity);
    if (!w->write_buffer.buffer) {
        close(w->segments.current_fd);
        free(w);
        return NULL;
    }
    // Initialize group commit
    pthread_mutex_init(&w->group_commit.lock, NULL);
    pthread_cond_init(&w->group_commit.sync_complete, NULL);
    w->group_commit.log_fd = w->segments.current_fd;
    w->group_commit.last_synced_lsn = 0;
    // Initialize append lock
    pthread_mutex_init(&w->append_lock, NULL);
    w->next_lsn = 1;  // LSN 0 is invalid
    return w;
}
int wal_append(WalWriter* w, const uint8_t* record, size_t len) {
    pthread_mutex_lock(&w->append_lock);
    // Check for rotation before writing
    if (rotate_if_needed(&w->segments) < 0) {
        pthread_mutex_unlock(&w->append_lock);
        return -1;
    }
    // Update group commit's FD if segment rotated
    w->group_commit.log_fd = w->segments.current_fd;
    // Allocate LSN
    uint64_t lsn = w->next_lsn;
    w->next_lsn += len;
    // Buffer the write
    if (w->write_buffer.used + len > w->write_buffer.capacity) {
        // Flush buffer if full
        flush_to_segment(&w->segments, &w->write_buffer);
    }
    memcpy(w->write_buffer.buffer + w->write_buffer.used, record, len);
    w->write_buffer.used += len;
    w->segments.current_size += len;
    pthread_mutex_unlock(&w->append_lock);
    return lsn;
}
int wal_commit_sync(WalWriter* w, uint64_t commit_lsn) {
    // This integrates with group commit
    pthread_mutex_lock(&w->append_lock);
    // Flush buffer to ensure COMMIT record is on disk
    flush_to_segment(&w->segments, &w->write_buffer);
    pthread_mutex_unlock(&w->append_lock);
    // Now do group commit
    return commit_with_group_commit(&w->group_commit, commit_lsn);
}
```
### Transaction API Integration
Now your transaction API from Milestone 1 uses the WAL writer:
```c
int commit_txn(TransactionManager* mgr, TxnId txn_id) {
    pthread_mutex_lock(&mgr->lock);
    TransactionEntry* entry = find_txn(&mgr->txn_table, txn_id);
    if (!entry || entry->status != TXN_ACTIVE) {
        pthread_mutex_unlock(&mgr->lock);
        return -1;
    }
    // Create COMMIT record
    CommitRecord rec = {
        .header = {
            .lsn = 0,  // Assigned by wal_append
            .type = RECORD_COMMIT,
            .txn_id = txn_id,
            .prev_lsn = entry->last_lsn,
        }
    };
    // Serialize
    uint8_t buffer[32];
    ssize_t len = serialize_commit_record(&rec, buffer, sizeof(buffer));
    // Append to WAL (buffered)
    uint64_t commit_lsn = wal_append(mgr->wal, buffer, len);
    if (commit_lsn == 0) {
        pthread_mutex_unlock(&mgr->lock);
        return -1;
    }
    rec.header.lsn = commit_lsn;
    // Group commit sync (blocks until durable)
    if (wal_commit_sync(mgr->wal, commit_lsn) < 0) {
        pthread_mutex_unlock(&mgr->lock);
        return -1;
    }
    // Transaction is now durable
    entry->status = TXN_COMMITTED;
    entry->last_lsn = commit_lsn;
    pthread_mutex_unlock(&mgr->lock);
    return 0;
}
```
## Testing the Implementation
### Test 1: Sequential Append Correctness
```c
void test_sequential_append(void) {
    WalWriter* w = wal_writer_create("/tmp/test_wal");
    assert(w != NULL);
    // Append 100 records
    uint64_t prev_lsn = 0;
    for (int i = 0; i < 100; i++) {
        uint8_t record[64];
        fill_test_record(record, sizeof(record), i);
        uint64_t lsn = wal_append(w, record, sizeof(record));
        assert(lsn > prev_lsn);
        prev_lsn = lsn;
    }
    // Verify file size
    struct stat st;
    stat("/tmp/test_wal.000001", &st);
    assert(st.st_size == 100 * 64);
    wal_writer_destroy(w);
    printf("Sequential append test passed!\n");
}
```
### Test 2: Group Commit Throughput
```c
#define NUM_THREADS 10
#define TXNS_PER_THREAD 10
typedef struct {
    WalWriter* w;
    int thread_id;
    uint64_t total_time;
} ThreadArg;
void* commit_thread(void* arg) {
    ThreadArg* ta = arg;
    uint64_t start = get_time_us();
    for (int i = 0; i < TXNS_PER_THREAD; i++) {
        uint8_t commit_record[32];
        uint64_t lsn = wal_append(ta->w, commit_record, sizeof(commit_record));
        wal_commit_sync(ta->w, lsn);
    }
    ta->total_time = get_time_us() - start;
    return NULL;
}
void test_group_commit_throughput(void) {
    WalWriter* w = wal_writer_create("/tmp/benchmark_wal");
    pthread_t threads[NUM_THREADS];
    ThreadArg args[NUM_THREADS];
    uint64_t start = get_time_us();
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].w = w;
        args[i].thread_id = i;
        pthread_create(&threads[i], NULL, commit_thread, &args[i]);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    uint64_t total_time = get_time_us() - start;
    int total_commits = NUM_THREADS * TXNS_PER_THREAD;
    double commits_per_sec = (double)total_commits * 1000000 / total_time;
    printf("Group commit throughput: %.0f commits/sec\n", commits_per_sec);
    // Compare with non-group commit (run separately)
    // Should be at least 5x faster
    assert(commits_per_sec > 1000);  // Adjust threshold for your hardware
    wal_writer_destroy(w);
}
```
### Test 3: Torn Write Detection
```c
void test_torn_write_detection(void) {
    // Write a valid record
    WalWriter* w = wal_writer_create("/tmp/torn_test_wal");
    uint8_t record[64];
    fill_test_record(record, sizeof(record), 0);
    wal_append(w, record, sizeof(record));
    wal_commit_sync(w, 1);
    // Simulate torn write by corrupting the last record
    int fd = open("/tmp/torn_test_wal.000001", O_RDWR);
    struct stat st;
    fstat(fd, &st);
    // Write garbage at the end (simulating partial write)
    uint8_t garbage[32] = {0};
    pwrite(fd, garbage, sizeof(garbage), st.st_size - 16);
    close(fd);
    // Recovery should detect and truncate
    int recovery_result = wal_recover("/tmp/torn_test_wal");
    assert(recovery_result == 0);
    // File should be truncated at original size - 32
    stat("/tmp/torn_test_wal.000001", &st);
    assert(st.st_size == 64);  // Only the valid record remains
    printf("Torn write detection test passed!\n");
}
```
### Test 4: Segment Rotation
```c
void test_segment_rotation(void) {
    WalWriter* w = wal_writer_create("/tmp/rotation_test_wal");
    w->segments.segment_size_limit = 1024;  // Small limit for testing
    // Write enough to trigger rotation
    uint8_t record[256];
    for (int i = 0; i < 10; i++) {
        wal_append(w, record, sizeof(record));
    }
    // Should have multiple segments
    struct stat st;
    assert(stat("/tmp/rotation_test_wal.000001", &st) == 0);
    assert(stat("/tmp/rotation_test_wal.000002", &st) == 0);
    assert(stat("/tmp/rotation_test_wal.000003", &st) == 0);
    wal_writer_destroy(w);
    printf("Segment rotation test passed!\n");
}
```
## Common Pitfalls
### 1. Forgetting to Update Group Commit FD on Rotation
When segment rotates, the file descriptor changes. If group commit still has the old FD, it fsyncs the wrong file (or a closed FD).
```c
// WRONG: FD never updated after rotation
w->group_commit.log_fd = initial_fd;  // Set once
rotate_segment();  // New FD, but group_commit still has old one
fsync(w->group_commit.log_fd);  // Fsyncing closed FD!
// CORRECT: Update on every rotation
if (rotate_segment(&w->segments) == ROTATED) {
    w->group_commit.log_fd = w->segments.current_fd;
}
```
### 2. Buffering Without Flushing on Commit
The write buffer improves throughput, but COMMIT records must be flushed:
```c
// WRONG: Buffer includes COMMIT but no flush
buffered_append(commit_record);
return success;  // Lie! Commit isn't durable
// CORRECT: Flush before acknowledging commit
buffered_append(commit_record);
flush_buffer();    // Write to OS
fsync();           // Durability
return success;    // Now it's true
```
### 3. Deadlock Between Append Lock and Group Commit Lock
If you hold the append lock while waiting for group commit, you block all other writers:
```c
// WRONG: Holding append lock during fsync
pthread_mutex_lock(&append_lock);
append_record();
commit_with_group_commit();  // Waits for fsync
pthread_mutex_unlock(&append_lock);  // Blocked for milliseconds!
// CORRECT: Release append lock before waiting
pthread_mutex_lock(&append_lock);
uint64_t lsn = append_record();
pthread_mutex_unlock(&append_lock);  // Release early
commit_with_group_commit(lsn);  // Wait outside lock
```
### 4. Losing Records on Rotation
If rotation happens while the write buffer contains records, those records must be flushed to the old segment before switching:
```c
// WRONG: Rotate without flushing buffer
if (size > limit) {
    close(old_fd);
    open(new_fd);
    // Records in buffer are lost!
}
// CORRECT: Flush before rotation
if (size + buffer.used > limit) {
    flush_buffer();  // All records go to old segment
    close(old_fd);
    open(new_fd);
}
```
### 5. Group Commit Starvation
If the leader crashes or hangs, all followers wait forever:
```c
// WRONG: No timeout on condition wait
while (last_synced < my_lsn) {
    pthread_cond_wait(&sync_complete, &lock);  // Forever if leader dies
}
// CORRECT: Use timed wait with fallback
while (last_synced < my_lsn) {
    struct timespec timeout = compute_timeout(1000);  // 1ms
    int rc = pthread_cond_timedwait(&sync_complete, &lock, &timeout);
    if (rc == ETIMEDOUT) {
        // Leader might be dead, become leader ourselves
        if (!sync_in_progress) {
            sync_in_progress = 1;
            pthread_mutex_unlock(&lock);
            fsync(log_fd);
            // ... complete as leader
        }
    }
}
```
## Knowledge Cascade
You've mastered the log writer. Here's where this knowledge connects:
**→ ARIES Recovery (Milestone 3)**: The log writer produces the records that recovery consumes. Every design decision you made — LSN as offset, CRC for torn writes, segment boundaries — matters during recovery. The recovery code will read the segments you wrote, detect torn writes with your CRCs, and reconstruct state from your records.
**→ Checkpointing (Milestone 4)**: Segments you created become the unit of log truncation. Old segments (before the checkpoint) can be deleted. The checkpoint record itself is written through this same log writer, triggering fsync like any other commit.
**→ Kafka (Cross-Domain)**: Kafka segments work identically — bounded files, sequential writes, rotation at size threshold. Kafka's "offset" is your LSN. Kafka's log compaction is your log truncation. You've built the same primitive.
**→ Event Sourcing (Cross-Domain)**: Event stores are append-only logs. Your group commit optimization applies — batching events before persisting. The torn write problem is identical. The segment rotation is identical.
**→ SSD Internals (Cross-Domain)**: SSDs themselves use log-structured writing internally. The FTL (Flash Translation Layer) appends writes to internal logs, never overwriting in place. Group commit at the application level batches writes that become batched again at the SSD level. You're aligning with the hardware's natural behavior.
**→ File System Design (Cross-Domain)**: Log-structured file systems (LFS) apply the same principle at the file system level. They batch all writes (data and metadata) into segments, then write segments sequentially. Your WAL is a mini log-structured file system for transaction records.
---

![WAL System Architecture Map](./diagrams/diag-satellite-wal-system.svg)

---
[[CRITERIA_JSON: {"milestone_id": "wal-impl-m2", "criteria": ["Log records appended sequentially via write() at current file offset; no pread/pwrite with overwrite, no seek to earlier positions; next_lsn equals current file size (LSN-as-offset design)", "COMMIT record durability guarantee: commit_txn() does not return success until COMMIT record and all preceding records for that transaction are persisted via fsync (or batched via group commit fsync)", "Group commit leader/follower protocol: first committing transaction becomes leader and calls fsync; subsequent concurrent committers become followers waiting on condition variable; leader broadcasts when fsync completes, releasing all followers whose LSNs are covered", "Group commit achieves ≥5x throughput improvement benchmarked with 100 concurrent transactions on 10+ threads compared to naive per-commit fsync; benchmark measures total commits completed per second", "Concurrent writer serialization via pthread_mutex around append operation; lock held only during write() syscall (microseconds), not during fsync (milliseconds)", "Log segment rotation triggered when current segment exceeds configurable threshold (default 64MB); creates new segment file with incremented ID; old segments retained until truncation (Milestone 4)", "Atomic write guarantee: partial records from crash detected by CRC32 mismatch during recovery scan; trailing garbage truncated via ftruncate() before recovery proceeds", "Write buffer batches records in memory (configurable size, suggest 1MB) before write() syscall; flush triggered on buffer full or commit; COMMIT records force buffer flush before fsync", "Group commit updates log_fd reference after segment rotation to prevent fsync on stale/closed file descriptor", "Unit tests verify: sequential LSN allocation, segment file creation at size threshold, torn write detection and truncation, concurrent append correctness (no interleaved records)"]}]
<!-- END_MS -->


<!-- MS_ID: wal-impl-m3 -->
# Milestone 3: ARIES Crash Recovery with CLRs
## The Recovery Problem: When Your Database Wakes Up Confused
You built the log writer. You can append records, batch commits with group commit, and detect torn writes. Your database is running smoothly. Transactions commit, data changes, clients are happy.
Then the power fails.
When your database restarts, it faces a fundamental question: **What just happened?**
The disk contains data pages in an uncertain state. Some committed transactions might not be on disk (no-force policy). Some uncommitted transactions might be on disk (steal policy). The log contains a mix of BEGIN, UPDATE, COMMIT, ABORT, and CLR records in no particular order across transactions.
How do you restore the database to a consistent state where:
- All committed transactions are reflected
- No uncommitted transactions are reflected
- The database could resume normal operations immediately?
This is the **crash recovery problem**, and ARIES is the algorithm that solves it.

![ARIES Three-Phase Recovery Overview](./diagrams/diag-aries-three-phases.svg)

### The Naive Model (And Why It's Wrong)
Most developers intuitively think recovery works like this:
1. Replay all committed transactions
2. Skip all uncommitted transactions
3. Done
This model is dangerously wrong. Here's why.
**Scenario**: Transaction T1 updates page P from value A to value B. The buffer pool, under memory pressure, evicts page P to disk (steal policy allowed this). T1 hasn't committed yet. Now the system crashes.
When you restart, page P on disk contains value B (T1's uncommitted change). If you "skip uncommitted transactions," you leave B on disk. But T1 never committed — B should never have persisted. You've just corrupted the database.
**The insight**: You can't just "skip" uncommitted transactions because their changes might already be on disk. You must **actively undo** them. And to undo them, you need their before-images — which means the changes must be present in the log and potentially replayed first.
This is why ARIES has three phases, not two. Let's understand each.
## Phase 1: Analysis — "What Was I Doing?"
The Analysis phase scans the log to reconstruct the state of the system at the moment of crash. It builds two critical tables:

![Analysis Phase: Building Tables](./diagrams/diag-analysis-phase-detail.svg)

### The Transaction Table
```c
typedef struct {
    uint32_t txn_id;
    TxnStatus status;        // ACTIVE, COMMITTED, ABORTED
    uint64_t first_lsn;      // LSN of BEGIN record
    uint64_t last_lsn;       // LSN of most recent record
} TransactionTableEntry;
typedef struct {
    TransactionTableEntry* entries;
    size_t capacity;
    size_t count;
} TransactionTable;
```
The transaction table answers: "Which transactions were in-flight at crash time?"
During analysis, you scan forward through the log:
- **BEGIN record**: Add transaction to table with status=ACTIVE
- **UPDATE/CLR record**: Update the transaction's `last_lsn`
- **COMMIT record**: Mark transaction as COMMITTED
- **ABORT record**: Mark transaction as ABORTED
After the scan, any transaction still marked ACTIVE was in-flight when the crash occurred — these need undo.
### The Dirty Page Table (DPT)
```c
typedef struct {
    uint64_t page_id;
    uint64_t rec_lsn;        // LSN of first record that dirtied this page
} DirtyPageEntry;
typedef struct {
    DirtyPageEntry* entries;
    size_t capacity;
    size_t count;
} DirtyPageTable;
```
The dirty page table answers: "Which pages might have uncommitted changes on disk?"
The `rec_lsn` (recovery LSN) is crucial — it's the LSN of the first log record that modified this page since it was last clean. Redo will start from the **minimum rec_lsn** across all dirty pages.
During analysis:
- **UPDATE/CLR record**: If page not in DPT, add it with `rec_lsn = record.lsn`
- **CHECKPOINT record**: Initialize DPT from checkpoint's snapshot
### Analysis Implementation
```c
int analysis_phase(LogManager* log, uint64_t checkpoint_lsn,
                   TransactionTable* txn_table, DirtyPageTable* dpt) {
    uint64_t current_lsn = checkpoint_lsn;
    uint8_t buffer[MAX_RECORD_SIZE];
    while (1) {
        // Read record at current_lsn
        ssize_t bytes_read = read_record_at_lsn(log, current_lsn, buffer, sizeof(buffer));
        if (bytes_read <= 0) break;  // End of log or error
        LogRecordHeader header;
        parse_header(buffer, &header);
        switch (header.type) {
            case RECORD_BEGIN: {
                // New transaction starts
                TransactionTableEntry* entry = txn_table_add(txn_table, header.txn_id);
                entry->status = TXN_ACTIVE;
                entry->first_lsn = current_lsn;
                entry->last_lsn = current_lsn;
                break;
            }
            case RECORD_UPDATE: {
                UpdateRecord update;
                parse_update_record(buffer, &update);
                // Update transaction's last_lsn
                TransactionTableEntry* entry = txn_table_find(txn_table, header.txn_id);
                if (entry) {
                    entry->last_lsn = current_lsn;
                }
                // Add page to dirty page table if not present
                if (!dpt_find(dpt, update.page_id)) {
                    dpt_add(dpt, update.page_id, current_lsn);
                }
                break;
            }
            case RECORD_CLR: {
                ClrRecord clr;
                parse_clr_record(buffer, &clr);
                // Update transaction's last_lsn
                TransactionTableEntry* entry = txn_table_find(txn_table, header.txn_id);
                if (entry) {
                    entry->last_lsn = current_lsn;
                }
                // Add page to DPT if not present
                if (!dpt_find(dpt, clr.page_id)) {
                    dpt_add(dpt, clr.page_id, current_lsn);
                }
                break;
            }
            case RECORD_COMMIT: {
                TransactionTableEntry* entry = txn_table_find(txn_table, header.txn_id);
                if (entry) {
                    entry->status = TXN_COMMITTED;
                    entry->last_lsn = current_lsn;
                }
                break;
            }
            case RECORD_ABORT: {
                TransactionTableEntry* entry = txn_table_find(txn_table, header.txn_id);
                if (entry) {
                    entry->status = TXN_ABORTED;
                    entry->last_lsn = current_lsn;
                }
                break;
            }
            case RECORD_CHECKPOINT: {
                CheckpointRecord checkpoint;
                parse_checkpoint_record(buffer, &checkpoint);
                // Initialize tables from checkpoint
                for (uint32_t i = 0; i < checkpoint.num_active_txns; i++) {
                    txn_table_add_entry(txn_table, &checkpoint.active_txns[i]);
                }
                for (uint32_t i = 0; i < checkpoint.num_dirty_pages; i++) {
                    dpt_add_entry(dpt, &checkpoint.dirty_pages[i]);
                }
                break;
            }
        }
        current_lsn += header.length;
    }
    return 0;
}
```
### What Analysis Produces
After the Analysis phase:
| Table | Contains | Used For |
|-------|----------|----------|
| Transaction Table | All transactions active at crash (status=ACTIVE) | Undo phase knows what to roll back |
| Dirty Page Table | All pages potentially out-of-sync on disk | Redo phase knows where to start |
## Phase 2: Redo — "Replay Everything"
Here's the counterintuitive truth that trips up most developers: **Redo replays ALL changes, both committed and uncommitted.**

![Why Redo Replays Uncommitted Changes](./diagrams/diag-redo-includes-uncommitted.svg)

Why? Because of the steal policy. Consider:
1. Transaction T1 (uncommitted) updates page P
2. Buffer pool evicts page P to disk (steal policy)
3. System crashes
4. Recovery starts
If Redo skipped T1's change, page P would be... what? It has T1's change on disk! You can't "skip" something that's already there.
**The Redo principle**: Replay ALL changes from the minimum rec_lsn forward. This ensures the database is in the same state it was at crash time — including uncommitted changes. Then Undo will remove the uncommitted ones.
### Redo Starting Point: minimum rec_lsn
The Dirty Page Table tells you where to start. Find the minimum rec_lsn:
```c
uint64_t find_redo_start_lsn(DirtyPageTable* dpt) {
    if (dpt->count == 0) {
        return 0;  // No dirty pages, nothing to redo
    }
    uint64_t min_rec_lsn = UINT64_MAX;
    for (size_t i = 0; i < dpt->count; i++) {
        if (dpt->entries[i].rec_lsn < min_rec_lsn) {
            min_rec_lsn = dpt->entries[i].rec_lsn;
        }
    }
    return min_rec_lsn;
}
```
### The pageLSN Check: Making Redo Idempotent

![Redo Phase: pageLSN Skip Logic](./diagrams/diag-redo-phase-idempotency.svg)

Here's a subtle problem: what if a change was already written to disk before the crash? If you redo it, you apply it twice.
**Solution**: Every page stores a `pageLSN` — the LSN of the most recent log record that modified this page. During Redo:
```c
// Read pageLSN from page on disk
uint64_t disk_page_lsn = read_page_lsn(buffer_pool, page_id);
// Compare to record LSN
if (disk_page_lsn >= record_lsn) {
    // This change is already on disk — skip it
    continue;
}
```
This makes Redo **idempotent** — running it multiple times produces the same result.

> **🔑 Foundation: Idempotent operations**
> 
> ## What It Is
An operation is **idempotent** when executing it multiple times produces the same result as executing it once. Put another way: calling it 1 time, 10 times, or 1000 times leaves the system in identical state.
```python
# Idempotent: result is always 5
def set_value(x):
    state = x
    return state
set_value(5)  # state = 5
set_value(5)  # state = 5 (no change)
set_value(5)  # state = 5 (no change)
# NOT idempotent: result changes each call
def increment():
    state += 1
    return state
increment()  # state = 1
increment()  # state = 2 (changed!)
increment()  # state = 3 (changed again!)
```
In HTTP terms: `GET`, `PUT`, and `DELETE` are idempotent. `POST` typically is not.
## Why You Need It Right Now
When building distributed systems—or any code that might retry on failure—idempotency isn't optional, it's survival.
**The problem:** Networks fail. Timeouts happen. Services crash mid-operation. Without idempotency, a retry that should recover from failure instead corrupts your data:
```python
# Dangerous: double-charging customers
def process_payment(order_id, amount):
    charge_credit_card(amount)  # First call succeeds
    mark_order_paid(order_id)   # Network timeout before this
    # Retry charges the card AGAIN!
```
**The fix:** Design operations so retries are safe:
```python
def process_payment(idempotency_key, amount):
    if payment_exists(idempotency_key):
        return existing_payment(idempotency_key)  # Safe to retry
    charge = charge_credit_card(amount)
    save_payment(idempotency_key, charge)
    return charge
```
## Key Insight
> **Idempotency transforms "at least once" delivery into "exactly once" semantics.**
Most messaging systems and networks guarantee "at least once" delivery—messages may arrive multiple times. Your code must handle this. By making operations idempotent, you absorb duplicates safely without special coordination.
**Mental model:** Think of idempotent operations as "setting state" versus non-idempotent operations as "accumulating state." `x = 5` is idempotent. `x += 1` is not. When in doubt, design for "set" semantics over "accumulate" semantics.

```c
int redo_phase(LogManager* log, BufferPool* pool, DirtyPageTable* dpt) {
    uint64_t redo_start = find_redo_start_lsn(dpt);
    if (redo_start == 0) {
        return 0;  // Nothing to redo
    }
    uint64_t current_lsn = redo_start;
    uint8_t buffer[MAX_RECORD_SIZE];
    while (1) {
        ssize_t bytes_read = read_record_at_lsn(log, current_lsn, buffer, sizeof(buffer));
        if (bytes_read <= 0) break;
        LogRecordHeader header;
        parse_header(buffer, &header);
        // Only UPDATE and CLR records need redo
        if (header.type == RECORD_UPDATE) {
            UpdateRecord update;
            parse_update_record(buffer, &update);
            // Check if page is in dirty page table
            DirtyPageEntry* dpe = dpt_find(dpt, update.page_id);
            if (!dpe) {
                // Page wasn't dirty, skip
                goto next_record;
            }
            // Idempotency check: is this change already on disk?
            uint64_t page_lsn = read_page_lsn(pool, update.page_id);
            if (page_lsn >= current_lsn) {
                // Already applied, skip
                goto next_record;
            }
            // Apply the change: write after-image to page
            apply_update(pool, update.page_id, update.offset,
                        update.new_value, update.new_value_len);
            // Update pageLSN
            set_page_lsn(pool, update.page_id, current_lsn);
        }
        else if (header.type == RECORD_CLR) {
            ClrRecord clr;
            parse_clr_record(buffer, &clr);
            // CLR redo: apply the undo operation
            DirtyPageEntry* dpe = dpt_find(dpt, clr.page_id);
            if (!dpe) goto next_record;
            uint64_t page_lsn = read_page_lsn(pool, clr.page_id);
            if (page_lsn >= current_lsn) goto next_record;
            // CLR contains the undo data (before-image of original update)
            apply_update(pool, clr.page_id, clr.offset,
                        clr.undo_data, clr.undo_data_len);
            set_page_lsn(pool, clr.page_id, current_lsn);
        }
        next_record:
        current_lsn += header.length;
    }
    // Flush all redone pages to disk
    buffer_pool_flush_all(pool);
    return 0;
}
```
### Why Redo Includes Uncommitted Changes
Let's trace through a concrete example:
```
LSN 100: T1 BEGIN
LSN 150: T1 UPDATE page 5 (old=A, new=B)
LSN 200: T2 BEGIN
LSN 250: T2 UPDATE page 7 (old=X, new=Y)
LSN 300: T1 COMMIT
LSN 350: [CRASH]
```
At crash time:
- T1 is committed (COMMIT record at LSN 300)
- T2 is active (no COMMIT record)
What's on disk?
- Page 5 might have B (if buffer pool evicted it) or A (if not)
- Page 7 might have Y or X
Redo replays LSN 150 (T1's change) and LSN 250 (T2's change):
- After Redo: Page 5 = B, Page 7 = Y
- This is exactly the in-memory state at crash time
Now Undo:
- T1 is committed → nothing to undo
- T2 is active → undo LSN 250, restore page 7 to X
Final state: Page 5 = B (T1 committed), Page 7 = X (T2 rolled back). Correct!
**If Redo had skipped T2's change**:
- After Redo: Page 5 = B, Page 7 = X (old value)
- But what if page 7 had Y on disk? (buffer pool evicted it)
- Undo writes X, but... was that the right thing? Page already had Y.
- The before-image in T2's UPDATE record assumes Y is in memory, but what if disk had something else?
- **Chaos**: You can't reliably undo without first ensuring the current state matches what undo expects.
This is why Redo replays ALL changes — to establish a known state that Undo can then work with.
## Phase 3: Undo — "Roll Back the Incomplete"
After Redo, the database is in exactly the state it was at crash time. Now Undo removes the effects of uncommitted transactions.

![Undo Phase: prevLSN Chain Traversal](./diagrams/diag-undo-prevlsn-traversal.svg)

### Finding Transactions to Undo
From the Analysis phase, you have the transaction table. Any transaction with `status == TXN_ACTIVE` needs undo:
```c
typedef struct {
    uint32_t txn_id;
    uint64_t last_lsn;    // Start point for undo
} UndoCandidate;
int find_transactions_to_undo(TransactionTable* txn_table, 
                              UndoCandidate* candidates, 
                              size_t* count) {
    *count = 0;
    for (size_t i = 0; i < txn_table->count; i++) {
        if (txn_table->entries[i].status == TXN_ACTIVE) {
            candidates[*count].txn_id = txn_table->entries[i].txn_id;
            candidates[*count].last_lsn = txn_table->entries[i].last_lsn;
            (*count)++;
        }
    }
    return 0;
}
```
### Undo Order: Global LSN Priority
Here's a subtle requirement: **Undo must process records in reverse LSN order across ALL active transactions, not per-transaction.**
Why? Consider:
```
T1: UPDATE page 5 at LSN 100
T2: UPDATE page 5 at LSN 200
(crash, both active)
```
If you undo T1 first (LSN 100), you restore page 5 to T1's before-image. Then you undo T2 (LSN 200), restoring to T2's before-image. But T2's before-image is T1's after-image, which you just undid! You've now applied the wrong value.
**Correct approach**: Undo in reverse LSN order globally:
1. Undo LSN 200 (T2's update)
2. Undo LSN 100 (T1's update)
This ensures you're always undoing the most recent change first, which has the correct before-image.

![Undo Order: Global LSN Priority Queue](./diagrams/diag-undo-order-priority-queue.svg)

```c
typedef struct {
    uint32_t txn_id;
    uint64_t lsn;          // LSN of record to undo
    uint64_t prev_lsn;     // Next record to undo in this transaction
} UndoWork;
// Priority queue ordered by LSN (highest first)
PriorityQueue undo_queue;
void build_undo_queue(TransactionTable* txn_table, PriorityQueue* queue) {
    for (size_t i = 0; i < txn_table->count; i++) {
        TransactionTableEntry* entry = &txn_table->entries[i];
        if (entry->status == TXN_ACTIVE && entry->last_lsn > entry->first_lsn) {
            UndoWork work = {
                .txn_id = entry->txn_id,
                .lsn = entry->last_lsn,
                .prev_lsn = 0  // Will be filled when we read the record
            };
            pq_push(queue, work);
        }
    }
}
```
### The Undo Loop with CLR Generation
Now the heart of Undo — and the reason CLRs exist:
```c
int undo_phase(LogManager* log, BufferPool* pool, 
               TransactionTable* txn_table, DirtyPageTable* dpt) {
    PriorityQueue undo_queue;
    pq_init(&undo_queue);
    build_undo_queue(txn_table, &undo_queue);
    while (!pq_empty(&undo_queue)) {
        UndoWork work = pq_pop(&undo_queue);  // Highest LSN first
        // Read the record to undo
        uint8_t buffer[MAX_RECORD_SIZE];
        ssize_t bytes = read_record_at_lsn(log, work.lsn, buffer, sizeof(buffer));
        if (bytes <= 0) continue;
        LogRecordHeader header;
        parse_header(buffer, &header);
        // Only UPDATE records need undo
        // (CLRs and BEGIN don't have inverse operations)
        if (header.type == RECORD_UPDATE) {
            UpdateRecord update;
            parse_update_record(buffer, &update);
            // Apply the before-image (undo the change)
            apply_update(pool, update.page_id, update.offset,
                        update.old_value, update.old_value_len);
            // Write a CLR recording this undo
            uint64_t clr_lsn = write_clr(log, 
                                         work.txn_id,
                                         update.page_id,
                                         update.offset,
                                         update.old_value,
                                         update.old_value_len,
                                         header.prev_lsn);  // undo_next_lsn
            // Update pageLSN
            set_page_lsn(pool, update.page_id, clr_lsn);
            // If there's more to undo in this transaction, add to queue
            if (header.prev_lsn > 0) {
                UndoWork next_work = {
                    .txn_id = work.txn_id,
                    .lsn = header.prev_lsn
                };
                pq_push(&undo_queue, next_work);
            } else {
                // Reached BEGIN record, write ABORT record
                write_abort_record(log, work.txn_id);
                TransactionTableEntry* entry = txn_table_find(txn_table, work.txn_id);
                if (entry) entry->status = TXN_ABORTED;
            }
        }
        else if (header.type == RECORD_CLR) {
            // CLR encountered during undo — follow undo_next_lsn
            ClrRecord clr;
            parse_clr_record(buffer, &clr);
            if (clr.undo_next_lsn > 0) {
                UndoWork next_work = {
                    .txn_id = work.txn_id,
                    .lsn = clr.undo_next_lsn
                };
                pq_push(&undo_queue, next_work);
            } else {
                // No more to undo
                write_abort_record(log, work.txn_id);
                TransactionTableEntry* entry = txn_table_find(txn_table, work.txn_id);
                if (entry) entry->status = TXN_ABORTED;
            }
        }
    }
    // Flush all undone pages
    buffer_pool_flush_all(pool);
    // Sync the log (CLRs and ABORT records must be durable)
    fsync(log->fd);
    pq_destroy(&undo_queue);
    return 0;
}
```
### The CLR: Making Undo Crash-Safe
Here's the critical insight. What happens if the system crashes DURING undo?
```
T1 has UPDATEs at LSN 100, 150, 200
Undo starts:
  - Undo LSN 200, write CLR at LSN 300 with undo_next_lsn=150
  - [CRASH]
Recovery runs again:
  - Analysis: T1 is still ACTIVE (no ABORT record)
  - Redo: Replays everything including CLR at 300
  - Undo: Needs to undo T1
```
Without CLRs, undo would try to undo LSN 200 again. But LSN 200 is already undone! The CLR at LSN 300 records "I already undid 200, next undo 150."

![CLR Prevents Recovery Loop](./diagrams/diag-clr-crash-during-undo.svg)

```c
uint64_t write_clr(LogManager* log, uint32_t txn_id,
                   uint64_t page_id, uint32_t offset,
                   const void* undo_data, size_t undo_len,
                   uint64_t undo_next_lsn) {
    ClrRecord clr = {
        .header = {
            .type = RECORD_CLR,
            .txn_id = txn_id,
            .prev_lsn = get_last_lsn(log, txn_id),  // Link to previous record
        },
        .page_id = page_id,
        .offset = offset,
        .undo_data_len = undo_len,
        .undo_data = undo_data,
        .undo_next_lsn = undo_next_lsn  // THE KEY FIELD
    };
    uint8_t buffer[MAX_RECORD_SIZE];
    ssize_t len = serialize_clr_record(&clr, buffer, sizeof(buffer));
    uint64_t clr_lsn = log_append(log, buffer, len);
    return clr_lsn;
}
```
The `undo_next_lsn` field points to the next record to undo if we crash. It's the `prev_lsn` of the UPDATE we just undid.
**Without CLRs**:
- Crash during undo
- Recovery redoes the original UPDATE
- Recovery tries to undo it again
- Potential data corruption or infinite loop
**With CLRs**:
- Crash during undo
- Recovery redoes the CLR (which applies the undo)
- Recovery sees CLR, follows `undo_next_lsn` to skip already-undone work
- Recovery continues from where it left off
This is why CLRs are essential for correct recovery. They make undo **restartable**.
## Torn Write Handling: Before Recovery Starts
Before any phase runs, you must handle torn writes from the crash:
```c
int detect_and_truncate_torn_writes(LogManager* log) {
    uint64_t file_size = get_file_size(log->fd);
    uint64_t scan_offset = 0;
    uint8_t buffer[MAX_RECORD_SIZE];
    // Scan to find the last valid record
    uint64_t last_valid_end = 0;
    while (scan_offset < file_size) {
        // Try to read header
        ssize_t header_bytes = pread(log->fd, buffer, HEADER_SIZE, scan_offset);
        if (header_bytes < HEADER_SIZE) {
            // Incomplete header — torn write
            break;
        }
        LogRecordHeader header;
        parse_header(buffer, &header);
        // Sanity check: length must be reasonable
        if (header.length < HEADER_SIZE + 4 || header.length > MAX_RECORD_SIZE) {
            // Corrupted length field
            break;
        }
        // Try to read full record
        ssize_t record_bytes = pread(log->fd, buffer, header.length, scan_offset);
        if (record_bytes < header.length) {
            // Incomplete record
            break;
        }
        // Verify CRC
        uint32_t stored_crc = read_le32(buffer + header.length - 4);
        uint32_t computed_crc = compute_crc(buffer, header.length - 4);
        if (stored_crc != computed_crc) {
            // CRC mismatch — torn write or corruption
            break;
        }
        // Record is valid
        last_valid_end = scan_offset + header.length;
        scan_offset = last_valid_end;
    }
    // Truncate any garbage at the end
    if (last_valid_end < file_size) {
        printf("Torn write detected: truncating %lu bytes\n", 
               file_size - last_valid_end);
        ftruncate(log->fd, last_valid_end);
    }
    return 0;
}
```
## The Complete Recovery Flow
```c
int crash_recovery(LogManager* log, BufferPool* pool, 
                   uint64_t checkpoint_lsn) {
    // Step 0: Handle torn writes
    detect_and_truncate_torn_writes(log);
    // Step 1: Analysis
    TransactionTable txn_table;
    DirtyPageTable dpt;
    txn_table_init(&txn_table);
    dpt_init(&dpt);
    printf("Starting Analysis phase...\n");
    analysis_phase(log, checkpoint_lsn, &txn_table, &dpt);
    printf("Analysis complete:\n");
    printf("  Active transactions: %zu\n", txn_table.count);
    printf("  Dirty pages: %zu\n", dpt.count);
    // Step 2: Redo
    printf("Starting Redo phase...\n");
    redo_phase(log, pool, &dpt);
    printf("Redo complete\n");
    // Step 3: Undo
    printf("Starting Undo phase...\n");
    undo_phase(log, pool, &txn_table, &dpt);
    printf("Undo complete\n");
    // Cleanup
    txn_table_destroy(&txn_table);
    dpt_destroy(&dpt);
    return 0;
}
```
## Testing Recovery: The Three-Transaction Scenario
The acceptance criteria specify a complex test scenario:
- T1 committed before crash
- T2 aborted before crash (has CLRs in log)
- T3 active at crash (needs undo)

![Three-Transaction Recovery Test](./diagrams/diag-recovery-test-scenario.svg)

```c
void test_three_transaction_recovery(void) {
    // Setup: Create log and buffer pool
    LogManager* log = log_create("/tmp/recovery_test.wal");
    BufferPool* pool = buffer_pool_create("/tmp/recovery_test.db");
    // === BEFORE CRASH ===
    // T1: Committed transaction
    uint32_t t1 = begin_txn(log);
    write_txn(log, t1, 1, 0, "A", 1, "B", 1);  // Page 1: A → B
    commit_txn(log, t1);
    // T2: Aborted transaction (CLRs in log)
    uint32_t t2 = begin_txn(log);
    write_txn(log, t2, 2, 0, "X", 1, "Y", 1);  // Page 2: X → Y
    abort_txn(log, t2);  // Generates CLR, then ABORT record
    // T3: Active transaction (no commit, no abort)
    uint32_t t3 = begin_txn(log);
    write_txn(log, t3, 3, 0, "P", 1, "Q", 1);  // Page 3: P → Q
    // No commit, no abort — still active
    // === SIMULATE CRASH ===
    // In a real test, you'd fork(), crash the child, and have parent verify
    // For simplicity, we'll just clear in-memory state and run recovery
    // Clear transaction table (simulate restart)
    log->txn_table.count = 0;
    // === RECOVERY ===
    crash_recovery(log, pool, 0);  // No checkpoint
    // === VERIFY ===
    // T1's change should be present
    char page1_val = read_page_byte(pool, 1, 0);
    assert(page1_val == 'B');
    // T2's change should be rolled back (Y → X)
    char page2_val = read_page_byte(pool, 2, 0);
    assert(page2_val == 'X');
    // T3's change should be rolled back (Q → P)
    char page3_val = read_page_byte(pool, 3, 0);
    assert(page3_val == 'P');
    printf("Three-transaction recovery test PASSED\n");
    log_destroy(log);
    buffer_pool_destroy(pool);
}
```
## Testing Idempotency: Recovery After Recovery

![Recovery Idempotency Verification](./diagrams/diag-recovery-idempotency-test.svg)

```c
void test_recovery_idempotency(void) {
    LogManager* log = log_create("/tmp/idempotency_test.wal");
    BufferPool* pool = buffer_pool_create("/tmp/idempotency_test.db");
    // Create some state
    uint32_t t1 = begin_txn(log);
    write_txn(log, t1, 1, 0, "A", 1, "B", 1);
    commit_txn(log, t1);
    uint32_t t2 = begin_txn(log);
    write_txn(log, t2, 2, 0, "X", 1, "Y", 1);
    // T2 active (no commit)
    // Run recovery first time
    crash_recovery(log, pool, 0);
    // Capture state
    char state1_page1 = read_page_byte(pool, 1, 0);
    char state1_page2 = read_page_byte(pool, 2, 0);
    // Clear in-memory state again
    log->txn_table.count = 0;
    // Run recovery SECOND time
    crash_recovery(log, pool, 0);
    // Capture state again
    char state2_page1 = read_page_byte(pool, 1, 0);
    char state2_page2 = read_page_byte(pool, 2, 0);
    // States must be identical
    assert(state1_page1 == state2_page1);
    assert(state1_page2 == state2_page2);
    printf("Recovery idempotency test PASSED\n");
    log_destroy(log);
    buffer_pool_destroy(pool);
}
```
## Testing Crash During Undo
```c
void test_crash_during_undo(void) {
    // This test simulates a crash during undo and verifies CLR correctness
    LogManager* log = log_create("/tmp/crash_during_undo_test.wal");
    BufferPool* pool = buffer_pool_create("/tmp/crash_during_undo_test.db");
    // Create transaction with multiple updates
    uint32_t t1 = begin_txn(log);
    write_txn(log, t1, 1, 0, "A", 1, "B", 1);  // LSN 100
    write_txn(log, t1, 1, 1, "C", 1, "D", 1);  // LSN 150
    write_txn(log, t1, 1, 2, "E", 1, "F", 1);  // LSN 200
    // No commit — active at crash
    // Simulate partial undo: manually write a CLR for the first undo
    // (as if we crashed after undoing LSN 200 but before finishing)
    ClrRecord partial_clr = {
        .header = {
            .type = RECORD_CLR,
            .txn_id = t1,
            .prev_lsn = 200,
        },
        .page_id = 1,
        .offset = 2,
        .undo_data = "E",
        .undo_data_len = 1,
        .undo_next_lsn = 150  // Next to undo is LSN 150
    };
    // ... serialize and write CLR ...
    // Now run recovery
    log->txn_table.count = 0;  // Clear state
    crash_recovery(log, pool, 0);
    // Verify: T1 should be fully rolled back
    // Page 1 should have original values A, C, E
    assert(read_page_byte(pool, 1, 0) == 'A');
    assert(read_page_byte(pool, 1, 1) == 'C');
    assert(read_page_byte(pool, 1, 2) == 'E');
    printf("Crash during undo test PASSED\n");
    log_destroy(log);
    buffer_pool_destroy(pool);
}
```
## Common Pitfalls
### 1. Redo Skipping Uncommitted Transactions
The most common mistake is filtering by transaction status during Redo:
```c
// WRONG: Skip uncommitted during redo
if (txn_table_find(&txn_table, header.txn_id)->status != TXN_COMMITTED) {
    continue;  // Skip this record
}
// This is wrong! Uncommitted changes may be on disk and need to be present for undo.
```
Redo must replay ALL changes regardless of transaction status.
### 2. Undo Per-Transaction Instead of Global LSN Order
```c
// WRONG: Undo each transaction independently
for (each active transaction T) {
    lsn = T.last_lsn;
    while (lsn > T.first_lsn) {
        undo_record(lsn);
        lsn = record.prev_lsn;
    }
}
// This is wrong! If two transactions modified the same page, order matters.
```
Use a priority queue ordered by LSN descending across all active transactions.
### 3. Forgetting to Write CLRs
```c
// WRONG: Undo without writing CLR
apply_update(pool, page_id, offset, old_value, old_len);
// No CLR written!
// If we crash now, recovery will re-undo this record, potentially corrupting data.
```
Every undo operation must write a CLR before continuing.
### 4. Ignoring Existing CLRs During Undo
```c
// WRONG: Treat CLR like any other record
if (header.type == RECORD_CLR) {
    // Skip CLRs during undo? No!
    continue;
}
```
CLRs encountered during undo must be followed via `undo_next_lsn`, not skipped.
### 5. Not Flushing Pages After Redo/Undo
```c
// WRONG: Complete recovery without flushing
redo_phase(...);
undo_phase(...);
return;  // Pages not flushed!
// If we crash again, redo will reapply changes that were already applied.
```
Both Redo and Undo must flush dirty pages before completing.
### 6. pageLSN Not Updated During Redo
```c
// WRONG: Apply redo without updating pageLSN
apply_update(pool, page_id, offset, new_value, new_len);
// Forgot: set_page_lsn(pool, page_id, record_lsn);
// Next recovery will redo this record again!
```
Every page modification must update `pageLSN` to the record's LSN.
## Knowledge Cascade
You've now implemented ARIES crash recovery — one of the most sophisticated algorithms in database systems. Here's where this knowledge connects:
**→ Checkpointing (Milestone 4)**: The checkpoint provides the starting LSN for Analysis. Without checkpoints, Analysis must scan from the beginning of the log — potentially gigabytes. Checkpoints bound recovery time.
**→ Distributed Consensus (Cross-Domain)**: The prevLSN chain is identical to Raft's log back-pointers and blockchain's block hashes. Each record points backward, creating an immutable chain. Recovery traverses this chain just like a blockchain verifier traverses block headers.
**→ Event Sourcing (Cross-Domain)**: Event stores face the same recovery problem. Events are logged, the system crashes, and you must replay events to restore state. The same "redo everything" principle applies — you can't selectively replay "committed" events without knowing their effects on state.
**→ Software Transactional Memory (Cross-Domain)**: STM systems log memory modifications and undo on conflict. The CLR concept appears as "undo logs" in STM. A conflict during commit triggers undo, and crashes during undo are handled similarly.
**→ Version Vectors (Cross-Domain)**: The pageLSN comparison for idempotency is the same pattern as version vectors in distributed systems. "If my version is >= your version, skip the update" prevents duplicate message processing and concurrent modification issues.
**→ Git (Cross-Domain)**: The prevLSN chain is exactly git's parent commit pointers. `git log` traverses the chain backward just like undo traverses the prevLSN chain. Git's "revert" is your CLR — a record that undoes a previous change and points to what's next.
**→ Message Queues (Cross-Domain)**: Exactly-once delivery requires idempotent message processing. The same pageLSN comparison pattern appears: "if I've already processed this message ID, skip it." Recovery is fundamentally about idempotency.
---

![Crash Scenarios and WAL Handling](./diagrams/diag-crash-scenarios-matrix.svg)

---
[[CRITERIA_JSON: {"milestone_id": "wal-impl-m3", "criteria": ["Analysis phase scans log from checkpoint_lsn forward, building transaction table (txn_id, status, first_lsn, last_lsn for each active transaction) and dirty page table (page_id, rec_lsn for each potentially-dirty page)", "Redo phase starts from minimum rec_lsn in dirty page table and replays ALL UPDATE and CLR records regardless of transaction status (committed or uncommitted)", "Redo skips records where pageLSN >= record_lsn (idempotency check), ensuring running redo multiple times produces identical page state", "Redo updates pageLSN to record_lsn after applying each change, preventing re-application on subsequent recovery attempts", "Undo phase identifies all transactions with status=ACTIVE in transaction table as needing rollback", "Undo uses global priority queue ordered by LSN descending to process records across all active transactions, not per-transaction sequential undo", "Each undo operation applies the before-image from UPDATE record and writes a CLR with undo_next_lsn pointing to the next record to undo (prev_lsn of the undone UPDATE)", "CLRs encountered during undo are followed via undo_next_lsn field to skip already-undone work, enabling crash-safe recovery resumption", "Recovery is idempotent: running crash_recovery() N times produces identical database state, verified by test comparing page contents after multiple recovery runs", "Torn write detection runs before recovery phases: scans log from beginning, truncates file at last record with valid CRC32 checksum", "Three-transaction test: T1 committed → changes present after recovery; T2 aborted before crash → changes rolled back via CLRs; T3 active at crash → changes rolled back during undo phase", "Crash-during-undo test: log contains partial CLRs from interrupted undo, recovery resumes correctly by following undo_next_lsn pointers in existing CLRs", "Both redo and undo phases flush all modified pages to disk before returning, ensuring recovery state persists across subsequent crashes", "Transaction status transitions: BEGIN adds ACTIVE entry, COMMIT marks COMMITTED, ABORT marks ABORTED, only ACTIVE transactions undergo undo"]}]
<!-- END_MS -->


<!-- MS_ID: wal-impl-m4 -->
<!-- MS_ID: wal-impl-m4 -->
# Milestone 4: Fuzzy Checkpointing & Log Truncation
## The Recovery Time Problem: Your Log is Growing Forever
You've built ARIES crash recovery. Your database can survive any crash, correctly redoing committed transactions and rolling back uncommitted ones. The system is durable.
But there's a problem lurking in production: **recovery time grows linearly with log size.**
Imagine your database has been running for a month. The log file is 50 GB. A crash occurs. How long does recovery take?
```
Analysis: Scan 50 GB of log records
Redo: Replay potentially millions of changes
Undo: Roll back any in-flight transactions
At 500 MB/s disk read speed:
50 GB ÷ 500 MB/s = 100 seconds minimum
Plus processing overhead: 5-10 minutes of downtime
```
Five minutes of downtime after a crash. In a high-availability system, that's unacceptable. But what can you do? The log is the source of truth for recovery.
**The insight**: You don't need the *entire* log to recover. You only need the portion that describes changes to pages that might not be on disk yet. If you could record "everything before this point is guaranteed to be on disk," recovery could start from there instead of the beginning.
This is what **checkpointing** does: it bounds recovery time by recording a known-good starting point.

![Recovery Scan: With vs Without Checkpoint](./diagrams/diag-recovery-scan-without-checkpoint.svg)

## The Naive Checkpoint: Stop the World
The simplest checkpoint approach is intuitive but problematic:
```
1. Stop accepting new transactions
2. Wait for all active transactions to complete
3. Flush all dirty pages to disk
4. Write a checkpoint record
5. Resume accepting transactions
```
This produces a perfect checkpoint — the database is in a known-consistent state, and recovery can start fresh from the checkpoint. But the cost is brutal:
| Database Load | Checkpoint Duration | Impact |
|---------------|---------------------|--------|
| Light (10 TPS) | Seconds | Noticeable pauses |
| Moderate (1000 TPS) | Tens of seconds | Application timeouts |
| Heavy (10000 TPS) | Minutes | Effectively unavailable |
In production, "stop the world" checkpoints are unacceptable. The database must remain available during checkpointing. But how can you capture a consistent snapshot while transactions are actively modifying data?
## Fuzzy Checkpointing: Consistency Without Blocking
The key insight of **fuzzy checkpointing** is that the checkpoint doesn't need to capture a perfectly consistent snapshot. It just needs to capture *enough information* for recovery to figure out what happened.

![Fuzzy Checkpoint: Concurrent with Transactions](./diagrams/diag-fuzzy-checkpoint-timeline.svg)

Here's the revelation: **recovery already handles inconsistency**. The Analysis phase reconstructs transaction state. The Redo phase replays all changes. The Undo phase rolls back incomplete work. A "fuzzy" checkpoint that captures a slightly stale view of the transaction table and dirty page table is perfectly usable — recovery will sort out the details.
### The Fuzzy Checkpoint Protocol
Instead of stopping the world, a fuzzy checkpoint:
1. **Write CHECKPOINT_BEGIN record** — marks the start of the checkpoint
2. **Capture snapshots** of the transaction table and dirty page table (these may be slightly stale by the time they're written)
3. **Write CHECKPOINT_END record** — contains the captured snapshots
4. **Update master record** — points to the CHECKPOINT_END LSN
Crucially, transactions continue executing throughout this process. A transaction might commit *during* the checkpoint, or a new transaction might start. The captured state won't be perfectly accurate, but that's okay — recovery handles it.
```c
typedef enum {
    CHECKPOINT_INACTIVE,
    CHECKPOINT_IN_PROGRESS
} CheckpointState;
typedef struct {
    CheckpointState state;
    uint64_t begin_lsn;         // LSN of CHECKPOINT_BEGIN
    TransactionTable* txn_snapshot;   // Captured transaction table
    DirtyPageTable* dpt_snapshot;     // Captured dirty page table
    pthread_mutex_t lock;
} CheckpointManager;
```
### Why "Fuzzy" Works
Consider what happens if a transaction commits during checkpoint:
```
Time 0: Checkpoint captures transaction table (T1 is ACTIVE)
Time 1: T1 commits (writes COMMIT record)
Time 2: Checkpoint writes CHECKPOINT_END with "T1 is ACTIVE"
Time 3: Checkpoint completes
```
The checkpoint says T1 is active, but T1 actually committed. Is this a problem?
No. During recovery:
1. Analysis scans from checkpoint LSN forward
2. Analysis encounters T1's COMMIT record
3. Analysis updates T1's status to COMMITTED
4. Recovery proceeds correctly
The checkpoint gave a slightly stale starting point, but recovery "caught up" by scanning forward from the checkpoint. The key is that the checkpoint provides a *bound* on how far back recovery needs to scan, not a perfect snapshot.

![Checkpoint Record Payload Structure](./diagrams/diag-checkpoint-record-content.svg)

## The Master Record: Your Bootstrap Location
You have checkpoint records in the log. But how does recovery find them? Scanning the entire log to find the last checkpoint defeats the purpose.
The solution: a **master record** in a well-known location.
```c
#define MASTER_RECORD_PATH "wal.master"
typedef struct {
    uint64_t magic;           // Magic number for validation
    uint64_t version;         // Format version
    uint64_t last_checkpoint_lsn;   // LSN of last CHECKPOINT_END
    uint64_t checkpoint_count;      // Number of checkpoints taken
    uint32_t crc;             // CRC of the above
} MasterRecord;
```
The master record is stored in a separate file (or the first page of a designated file). It's small (one page or less) and updated atomically after each checkpoint completes.

![Master Record: Bootstrap Location](./diagrams/diag-master-record-location.svg)

### Atomic Master Record Updates
The master record must be updated atomically. If the system crashes during the update, recovery must still find a valid (possibly slightly old) checkpoint.
**Approach 1: Double-write**
```c
int update_master_record(const char* path, MasterRecord* rec) {
    char temp_path[256];
    snprintf(temp_path, sizeof(temp_path), "%s.tmp", path);
    // Write to temporary file first
    int fd = open(temp_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return -1;
    // Compute CRC
    rec->crc = compute_crc(rec, offsetof(MasterRecord, crc));
    if (write(fd, rec, sizeof(MasterRecord)) != sizeof(MasterRecord)) {
        close(fd);
        unlink(temp_path);
        return -1;
    }
    fsync(fd);
    close(fd);
    // Atomic rename
    if (rename(temp_path, path) < 0) {
        unlink(temp_path);
        return -1;
    }
    return 0;
}
```
The `rename()` syscall is atomic on POSIX systems. After rename completes, the new master record is visible; before rename, the old one is still there. A crash at any point leaves either the old or new record intact, never garbage.
**Approach 2: In-place with versioning**
```c
typedef struct {
    uint64_t version;         // Incremented on each update
    MasterRecord records[2];  // Two slots, version determines active
} VersionedMasterRecord;
```
Write to the inactive slot, then update the version number. Readers check the version to determine which slot is active. This enables in-place updates without rename, useful on systems where rename is expensive.
## Implementing the Fuzzy Checkpoint
### Checkpoint Trigger
Checkpoints should be triggered automatically based on configurable criteria:
```c
typedef struct {
    uint64_t time_interval_ms;      // Time between checkpoints (0 = disabled)
    uint64_t record_count_interval; // Records between checkpoints (0 = disabled)
    uint64_t min_recovery_lsn_age;  // Only checkpoint if recovery would benefit
} CheckpointConfig;
typedef struct {
    CheckpointConfig config;
    uint64_t last_checkpoint_time;  // Timestamp of last checkpoint
    uint64_t last_checkpoint_lsn;   // LSN of last CHECKPOINT_END
    uint64_t records_since_checkpoint;
} CheckpointScheduler;
int should_checkpoint(CheckpointScheduler* sched, LogManager* log) {
    // Time-based trigger
    if (sched->config.time_interval_ms > 0) {
        uint64_t now = get_time_ms();
        if (now - sched->last_checkpoint_time >= sched->config.time_interval_ms) {
            return 1;
        }
    }
    // Record count trigger
    if (sched->config.record_count_interval > 0) {
        if (sched->records_since_checkpoint >= sched->config.record_count_interval) {
            return 1;
        }
    }
    return 0;
}
```
### The Checkpoint Implementation
```c
int take_fuzzy_checkpoint(CheckpointManager* cm, LogManager* log, 
                          TransactionTable* active_txn_table,
                          DirtyPageTable* active_dpt) {
    pthread_mutex_lock(&cm->lock);
    // Step 1: Write CHECKPOINT_BEGIN
    CheckpointBeginRecord begin_rec = {
        .header = {
            .type = RECORD_CHECKPOINT_BEGIN,
            .txn_id = 0,  // System operation, not a transaction
            .prev_lsn = 0,
        }
    };
    uint64_t begin_lsn = log_append(log, &begin_rec, sizeof(begin_rec));
    cm->begin_lsn = begin_lsn;
    cm->state = CHECKPOINT_IN_PROGRESS;
    // Step 2: Capture snapshots (transactions continue during this)
    TransactionTable* txn_snapshot = capture_transaction_table(active_txn_table);
    DirtyPageTable* dpt_snapshot = capture_dirty_page_table(active_dpt);
    // Step 3: Write CHECKPOINT_END with captured data
    CheckpointEndRecord end_rec = {
        .header = {
            .type = RECORD_CHECKPOINT_END,
            .txn_id = 0,
            .prev_lsn = begin_lsn,
        },
        .num_active_txns = txn_snapshot->count,
        .num_dirty_pages = dpt_snapshot->count,
    };
    // Serialize the checkpoint record with variable-length data
    uint8_t* buffer = malloc(MAX_CHECKPOINT_SIZE);
    size_t offset = serialize_checkpoint_end(&end_rec, buffer, 
                                             txn_snapshot, dpt_snapshot);
    uint64_t end_lsn = log_append(log, buffer, offset);
    free(buffer);
    // Step 4: Sync the log (checkpoint must be durable)
    fsync(log->fd);
    // Step 5: Update master record
    MasterRecord master = {
        .magic = MASTER_MAGIC,
        .version = 1,
        .last_checkpoint_lsn = end_lsn,
        .checkpoint_count = cm->checkpoint_count + 1,
    };
    update_master_record(MASTER_RECORD_PATH, &master);
    cm->state = CHECKPOINT_INACTIVE;
    cm->checkpoint_count++;
    // Clean up snapshots
    free(txn_snapshot);
    free(dpt_snapshot);
    pthread_mutex_unlock(&cm->lock);
    return 0;
}
```
### Capturing Snapshots Under Concurrent Modification
The snapshot capture must be thread-safe. Transactions are actively modifying the tables while we're trying to capture them.
```c
TransactionTable* capture_transaction_table(TransactionTable* active) {
    TransactionTable* snapshot = malloc(sizeof(TransactionTable));
    snapshot->capacity = active->count;
    snapshot->entries = malloc(sizeof(TransactionTableEntry) * snapshot->capacity);
    // Lock briefly to copy
    pthread_mutex_lock(&active->lock);
    snapshot->count = active->count;
    for (size_t i = 0; i < active->count; i++) {
        snapshot->entries[i] = active->entries[i];
    }
    pthread_mutex_unlock(&active->lock);
    return snapshot;
}
```
The lock is held only for the memory copy — microseconds. Transactions are blocked minimally. The snapshot might be slightly stale by the time it's written (a transaction committed or aborted), but that's acceptable.
## Recovery with Checkpoints
The recovery flow now incorporates checkpoint awareness:
```c
int crash_recovery_with_checkpoint(LogManager* log, BufferPool* pool) {
    // Step 0: Handle torn writes
    detect_and_truncate_torn_writes(log);
    // Step 1: Read master record to find checkpoint
    uint64_t checkpoint_lsn = 0;
    MasterRecord master;
    if (read_master_record(MASTER_RECORD_PATH, &master) == 0) {
        if (validate_master_record(&master)) {
            checkpoint_lsn = master.last_checkpoint_lsn;
            printf("Found checkpoint at LSN %lu\n", checkpoint_lsn);
        }
    }
    // Step 2: If checkpoint exists, read it to initialize tables
    TransactionTable txn_table;
    DirtyPageTable dpt;
    txn_table_init(&txn_table);
    dpt_init(&dpt);
    if (checkpoint_lsn > 0) {
        CheckpointEndRecord checkpoint;
        if (read_checkpoint_record(log, checkpoint_lsn, &checkpoint) == 0) {
            // Initialize tables from checkpoint
            for (uint32_t i = 0; i < checkpoint.num_active_txns; i++) {
                txn_table_add(&txn_table, &checkpoint.active_txns[i]);
            }
            for (uint32_t i = 0; i < checkpoint.num_dirty_pages; i++) {
                dpt_add(&dpt, &checkpoint.dirty_pages[i]);
            }
        }
    }
    // Step 3: Analysis phase starts from checkpoint LSN
    printf("Starting Analysis phase from LSN %lu...\n", checkpoint_lsn);
    analysis_phase(log, checkpoint_lsn, &txn_table, &dpt);
    // Step 4: Redo phase (unchanged)
    printf("Starting Redo phase...\n");
    redo_phase(log, pool, &dpt);
    // Step 5: Undo phase (unchanged)
    printf("Starting Undo phase...\n");
    undo_phase(log, pool, &txn_table, &dpt);
    // Cleanup
    txn_table_destroy(&txn_table);
    dpt_destroy(&dpt);
    return 0;
}
```
The key change: **Analysis starts from the checkpoint LSN, not from the beginning of the log**. All records before the checkpoint are irrelevant — their effects are either already on disk or captured in the checkpoint's dirty page table.
## Log Truncation: Reclaiming Disk Space
With checkpointing in place, you can now safely delete old log segments. But which segments can be deleted?

![Log Truncation: What Can Be Deleted](./diagrams/diag-log-truncation-safety.svg)

### The Truncation Safety Rule
A log segment can be safely deleted if and only if:
1. **The segment ends before the checkpoint's minimum recLSN** — All dirty pages from before the checkpoint have been flushed to disk, so redo won't need those records.
2. **The segment is not needed for undo** — No active transaction at checkpoint time had its firstLSN in that segment.
```c
int can_truncate_segment(LogSegmentManager* lsm, 
                         uint32_t segment_id,
                         uint64_t checkpoint_min_rec_lsn,
                         TransactionTable* checkpoint_txn_table) {
    SegmentInfo info;
    if (get_segment_info(lsm, segment_id, &info) < 0) {
        return 0;  // Can't get info, don't truncate
    }
    // Rule 1: Segment must end before checkpoint's min recLSN
    if (info.end_lsn >= checkpoint_min_rec_lsn) {
        return 0;  // Segment might be needed for redo
    }
    // Rule 2: No active transaction's firstLSN in this segment
    for (size_t i = 0; i < checkpoint_txn_table->count; i++) {
        TransactionTableEntry* entry = &checkpoint_txn_table->entries[i];
        if (entry->first_lsn >= info.start_lsn && 
            entry->first_lsn <= info.end_lsn) {
            return 0;  // Segment needed for undo
        }
    }
    return 1;  // Safe to truncate
}
```
### Implementing Truncation
```c
int truncate_old_segments(LogSegmentManager* lsm, 
                          uint64_t checkpoint_min_rec_lsn,
                          TransactionTable* checkpoint_txn_table) {
    int segments_deleted = 0;
    for (uint32_t seg_id = 1; seg_id < lsm->current_segment_id; seg_id++) {
        if (can_truncate_segment(lsm, seg_id, checkpoint_min_rec_lsn, 
                                 checkpoint_txn_table)) {
            char path[512];
            snprintf(path, sizeof(path), "%s.%06u", lsm->base_path, seg_id);
            printf("Truncating segment %u (path: %s)\n", seg_id, path);
            if (unlink(path) == 0) {
                segments_deleted++;
                mark_segment_deleted(lsm, seg_id);
            }
        }
    }
    return segments_deleted;
}
```
### The Long-Running Transaction Problem
There's a subtle issue: what if a transaction stays active for a very long time?
```
T1 starts at LSN 100
... 1 billion log records later ...
T1 is still active
Checkpoint at LSN 1,000,000,100
```
T1's `first_lsn` is 100. For undo to work, you need the records starting from LSN 100. This means **you can never truncate any segment** as long as T1 is active.
This is a real problem in production databases. Solutions include:
1. **Transaction timeout**: Abort transactions that run too long
2. **Savepoints**: Allow partial rollback, enabling truncation
3. **Nested transactions**: Break long transactions into smaller units
4. **Eager undo**: Undo old operations even if transaction is still active (advanced)
For your implementation, document the limitation: long-running transactions prevent log truncation.
## The Checkpoint Overhead Tradeoff
Frequent checkpoints reduce recovery time but add overhead during normal operation. Let's quantify this.

![Checkpoint Interval Tradeoff Curve](./diagrams/diag-checkpoint-interval-tradeoff.svg)

```c
void benchmark_checkpoint_tradeoff(void) {
    // Test scenarios
    struct {
        const char* name;
        uint64_t interval_records;
    } scenarios[] = {
        {"No checkpoint", 0},
        {"Every 1000 records", 1000},
        {"Every 10000 records", 10000},
        {"Every 100000 records", 100000},
    };
    for (int i = 0; i < 4; i++) {
        // Run workload with checkpoint interval
        TransactionManager* mgr = setup_test_database();
        mgr->checkpoint_config.record_count_interval = scenarios[i].interval_records;
        uint64_t start = get_time_us();
        run_workload(mgr, 100000);  // 100k transactions
        uint64_t workload_time = get_time_us() - start;
        // Simulate crash and measure recovery
        uint64_t recovery_start = get_time_us();
        crash_recovery_with_checkpoint(mgr->log, mgr->pool);
        uint64_t recovery_time = get_time_us() - recovery_start;
        printf("Scenario: %s\n", scenarios[i].name);
        printf("  Workload time: %lu ms\n", workload_time / 1000);
        printf("  Recovery time: %lu ms\n", recovery_time / 1000);
        printf("\n");
        cleanup_test_database(mgr);
    }
}
```
Typical results:
| Checkpoint Interval | Workload Overhead | Recovery Time |
|---------------------|-------------------|---------------|
| None | 0% | Full log scan |
| Every 1000 records | ~5-10% | ~1% of log |
| Every 10000 records | ~1-2% | ~10% of log |
| Every 100000 records | ~0.5% | ~50% of log |
The sweet spot depends on your SLA: how much overhead can you tolerate, and how fast must recovery be?
### Recovery Time Bounding
A common production requirement: "Recovery must complete within N seconds." You can derive the checkpoint interval from this:
```c
uint64_t compute_checkpoint_interval_for_recovery_sla(
    uint64_t recovery_sla_ms,
    uint64_t records_per_second_during_recovery,
    uint64_t bytes_per_record
) {
    // How many records can we scan in the SLA time?
    uint64_t max_records = (records_per_second_during_recovery * recovery_sla_ms) / 1000;
    // Checkpoint interval should ensure log never exceeds this
    return max_records / 2;  // Factor of 2 safety margin
}
```
## The Complete Checkpoint System
Let's integrate all the pieces:
```c
typedef struct {
    // Checkpoint management
    CheckpointManager checkpoint;
    CheckpointScheduler scheduler;
    // Log management
    LogManager* log;
    LogSegmentManager* segments;
    // Active tables (shared with transaction processing)
    TransactionTable* active_txn_table;
    DirtyPageTable* active_dpt;
    // Background thread
    pthread_t checkpoint_thread;
    int shutdown_requested;
} CheckpointSystem;
void* checkpoint_thread_func(void* arg) {
    CheckpointSystem* sys = arg;
    while (!sys->shutdown_requested) {
        // Sleep briefly
        usleep(100000);  // 100ms
        // Check if checkpoint is needed
        if (should_checkpoint(&sys->scheduler, sys->log)) {
            printf("Taking checkpoint...\n");
            take_fuzzy_checkpoint(&sys->checkpoint, sys->log,
                                 sys->active_txn_table, sys->active_dpt);
            // Update scheduler
            sys->scheduler.last_checkpoint_time = get_time_ms();
            sys->scheduler.last_checkpoint_lsn = sys->checkpoint.last_end_lsn;
            sys->scheduler.records_since_checkpoint = 0;
            // Attempt truncation
            uint64_t min_rec_lsn = find_min_rec_lsn(sys->active_dpt);
            int deleted = truncate_old_segments(sys->segments, min_rec_lsn,
                                                sys->active_txn_table);
            printf("Truncated %d old segments\n", deleted);
        }
    }
    return NULL;
}
CheckpointSystem* checkpoint_system_create(LogManager* log,
                                           LogSegmentManager* segments,
                                           TransactionTable* txn_table,
                                           DirtyPageTable* dpt,
                                           CheckpointConfig* config) {
    CheckpointSystem* sys = calloc(1, sizeof(CheckpointSystem));
    sys->log = log;
    sys->segments = segments;
    sys->active_txn_table = txn_table;
    sys->active_dpt = dpt;
    sys->scheduler.config = *config;
    sys->scheduler.last_checkpoint_time = get_time_ms();
    pthread_mutex_init(&sys->checkpoint.lock, NULL);
    // Start background thread
    pthread_create(&sys->checkpoint_thread, NULL, checkpoint_thread_func, sys);
    return sys;
}
void checkpoint_system_destroy(CheckpointSystem* sys) {
    sys->shutdown_requested = 1;
    pthread_join(sys->checkpoint_thread, NULL);
    pthread_mutex_destroy(&sys->checkpoint.lock);
    free(sys);
}
```
## Testing the Implementation
### Test 1: Recovery Time Reduction
```c
void test_checkpoint_recovery_time(void) {
    // Setup database
    TransactionManager* mgr = setup_test_database();
    // Write 10000 records WITHOUT checkpoint
    for (int i = 0; i < 10000; i++) {
        uint32_t txn = begin_txn(mgr);
        write_txn(mgr, txn, i % 100, 0, "old", 3, "new", 3);
        commit_txn(mgr, txn);
    }
    // Measure recovery time without checkpoint
    uint64_t start = get_time_us();
    crash_recovery_with_checkpoint(mgr->log, mgr->pool);
    uint64_t recovery_no_checkpoint = get_time_us() - start;
    // Write 10000 more records WITH checkpoint at halfway point
    for (int i = 0; i < 10000; i++) {
        uint32_t txn = begin_txn(mgr);
        write_txn(mgr, txn, i % 100, 0, "old", 3, "new", 3);
        commit_txn(mgr, txn);
        if (i == 5000) {
            // Take checkpoint at 50%
            take_fuzzy_checkpoint(&mgr->checkpoint, mgr->log,
                                 &mgr->txn_table, &mgr->dpt);
        }
    }
    // Measure recovery time with checkpoint
    start = get_time_us();
    crash_recovery_with_checkpoint(mgr->log, mgr->pool);
    uint64_t recovery_with_checkpoint = get_time_us() - start;
    printf("Recovery without checkpoint: %lu us\n", recovery_no_checkpoint);
    printf("Recovery with checkpoint: %lu us\n", recovery_with_checkpoint);
    printf("Speedup: %.2fx\n", 
           (double)recovery_no_checkpoint / recovery_with_checkpoint);
    // With checkpoint at 50%, recovery should be roughly halved
    assert(recovery_with_checkpoint < recovery_no_checkpoint * 0.6);
    cleanup_test_database(mgr);
    printf("Recovery time test PASSED\n");
}
```
### Test 2: Log Truncation Correctness
```c
void test_log_truncation(void) {
    TransactionManager* mgr = setup_test_database();
    mgr->segments->segment_size_limit = 10000;  // Small segments for testing
    // Write enough to create multiple segments
    for (int i = 0; i < 1000; i++) {
        uint32_t txn = begin_txn(mgr);
        write_txn(mgr, txn, i % 10, 0, "old", 3, "new", 3);
        commit_txn(mgr, txn);
    }
    // Take checkpoint
    take_fuzzy_checkpoint(&mgr->checkpoint, mgr->log,
                         &mgr->txn_table, &mgr->dpt);
    // Get segment count before truncation
    int segments_before = count_segments(mgr->segments);
    printf("Segments before truncation: %d\n", segments_before);
    // Truncate old segments
    uint64_t min_rec_lsn = find_min_rec_lsn(&mgr->dpt);
    int deleted = truncate_old_segments(mgr->segments, min_rec_lsn, &mgr->txn_table);
    printf("Segments deleted: %d\n", deleted);
    // Verify recovery still works
    crash_recovery_with_checkpoint(mgr->log, mgr->pool);
    // Verify all committed transactions are present
    for (int i = 0; i < 10; i++) {
        char val[4];
        read_page(mgr->pool, i, 0, val, 3);
        assert(memcmp(val, "new", 3) == 0);
    }
    cleanup_test_database(mgr);
    printf("Log truncation test PASSED\n");
}
```
### Test 3: Long-Running Transaction Blocks Truncation
```c
void test_long_txn_blocks_truncation(void) {
    TransactionManager* mgr = setup_test_database();
    mgr->segments->segment_size_limit = 10000;
    // Start a long-running transaction
    uint32_t long_txn = begin_txn(mgr);
    // Write many records (creates multiple segments)
    for (int i = 0; i < 1000; i++) {
        uint32_t txn = begin_txn(mgr);
        write_txn(mgr, txn, i % 10, 0, "old", 3, "new", 3);
        commit_txn(mgr, txn);
    }
    // Take checkpoint
    take_fuzzy_checkpoint(&mgr->checkpoint, mgr->log,
                         &mgr->txn_table, &mgr->dpt);
    // Attempt truncation
    uint64_t min_rec_lsn = find_min_rec_lsn(&mgr->dpt);
    int deleted = truncate_old_segments(mgr->segments, min_rec_lsn, &mgr->txn_table);
    // No segments should be deleted (long_txn blocks it)
    assert(deleted == 0);
    printf("Long-running transaction correctly blocked truncation\n");
    // Now commit the long transaction
    commit_txn(mgr, long_txn);
    // Take another checkpoint
    take_fuzzy_checkpoint(&mgr->checkpoint, mgr->log,
                         &mgr->txn_table, &mgr->dpt);
    // Now truncation should work
    deleted = truncate_old_segments(mgr->segments, min_rec_lsn, &mgr->txn_table);
    assert(deleted > 0);
    printf("After commit, truncation succeeded: %d segments deleted\n", deleted);
    cleanup_test_database(mgr);
    printf("Long transaction blocking test PASSED\n");
}
```
### Test 4: Master Record Recovery
```c
void test_master_record_recovery(void) {
    TransactionManager* mgr = setup_test_database();
    // Write some data and take checkpoint
    for (int i = 0; i < 100; i++) {
        uint32_t txn = begin_txn(mgr);
        write_txn(mgr, txn, i, 0, "old", 3, "new", 3);
        commit_txn(mgr, txn);
    }
    take_fuzzy_checkpoint(&mgr->checkpoint, mgr->log,
                         &mgr->txn_table, &mgr->dpt);
    // Read master record
    MasterRecord master;
    assert(read_master_record(MASTER_RECORD_PATH, &master) == 0);
    assert(master.last_checkpoint_lsn > 0);
    printf("Master record checkpoint LSN: %lu\n", master.last_checkpoint_lsn);
    // Corrupt the master record (simulate partial write)
    int fd = open(MASTER_RECORD_PATH, O_RDWR);
    uint8_t garbage[16] = {0};
    pwrite(fd, garbage, 16, 0);
    close(fd);
    // Recovery should detect corruption and handle gracefully
    // (fall back to scanning from beginning)
    crash_recovery_with_checkpoint(mgr->log, mgr->pool);
    // Verify data is still correct
    for (int i = 0; i < 100; i++) {
        char val[4];
        read_page(mgr->pool, i, 0, val, 3);
        assert(memcmp(val, "new", 3) == 0);
    }
    cleanup_test_database(mgr);
    printf("Master record recovery test PASSED\n");
}
```
## Common Pitfalls
### 1. Updating Master Record Before Checkpoint is Durable
```c
// WRONG: Update master record before fsync
write_checkpoint_end(log, checkpoint_data);
update_master_record(master_path, checkpoint_lsn);  // Too early!
fsync(log->fd);  // What if crash here?
// CORRECT: fsync first, then update master
write_checkpoint_end(log, checkpoint_data);
fsync(log->fd);  // Checkpoint is now durable
update_master_record(master_path, checkpoint_lsn);
```
If the master record points to a checkpoint that's not on disk, recovery will fail.
### 2. Truncating Segments Needed for Undo
```c
// WRONG: Only check recLSN, ignore active transactions
if (segment.end_lsn < min_rec_lsn) {
    truncate(segment);  // Might delete undo data!
}
// CORRECT: Check both recLSN and firstLSN
if (segment.end_lsn < min_rec_lsn && 
    !has_active_txn_in_segment(segment, txn_table)) {
    truncate(segment);
}
```
### 3. Non-Atomic Master Record Updates
```c
// WRONG: In-place update without atomicity
int fd = open(master_path, O_RDWR);
pwrite(fd, &new_master, sizeof(new_master), 0);  // Crash here = corruption!
close(fd);
// CORRECT: Use rename for atomicity
write_to_temp_file(temp_path, &new_master);
rename(temp_path, master_path);  // Atomic
```
### 4. Checkpoint Starvation Under High Load
If checkpoints take too long under high load, and a new checkpoint is triggered before the previous one completes, you get checkpoint queueing:
```c
// WRONG: Allow overlapping checkpoints
if (should_checkpoint()) {
    take_checkpoint();  // What if previous still running?
}
// CORRECT: Check state first
if (should_checkpoint() && checkpoint.state == CHECKPOINT_INACTIVE) {
    take_checkpoint();
}
```
### 5. Forgetting to Flush Buffer Pool Before Checkpoint
While fuzzy checkpoints don't require a clean buffer pool, flushing dirty pages before checkpoint improves the checkpoint's usefulness:
```c
// BETTER: Flush before checkpoint for better bounds
buffer_pool_flush_all(pool);
take_fuzzy_checkpoint(...);
// Now min_rec_lsn is higher, recovery starts later
```
This is optional but recommended before important checkpoints.
## Knowledge Cascade
You've completed the WAL implementation with checkpointing and log truncation. Here's where this knowledge connects:
**→ MVCC Snapshot Isolation (Cross-Domain)**: The fuzzy snapshot you captured during checkpoint is identical to the snapshot taken by a transaction under MVCC. Both capture a "consistent but slightly stale" view without blocking writers. The transaction table you captured is the same structure an MVCC system uses to determine visibility.
**→ Virtual Machine Live Migration (Cross-Domain)**: VM migration uses the same technique as fuzzy checkpointing. The hypervisor captures a snapshot of VM memory while the VM continues running, then iteratively copies changed pages. The "fuzzy" snapshot enables migration without stopping the VM, just as your fuzzy checkpoint enables checkpointing without stopping transactions.
**→ Kafka Log Compaction (Cross-Domain)**: Kafka's log compaction is your log truncation. Kafka retains the latest message for each key, deleting older versions. Your truncation retains records needed for recovery, deleting older ones. Both systems answer: "what's the minimum log needed to reconstruct state?"
**→ Filesystem Superblocks (Cross-Domain)**: Your master record is a superblock. Filesystems store critical bootstrap information in a well-known location (superblock, MFT, etc.). Recovery starts by reading this location, just as your database recovery reads the master record. The pattern is universal: "bootstrap information at a known address."
**→ Garbage Collection Pause Times (Cross-Domain)**: The checkpoint interval tradeoff is the same as GC pause time tuning. Frequent GC (frequent checkpoints) means short pauses (fast recovery) but higher overhead. Infrequent GC (infrequent checkpoints) means lower overhead but longer pauses (slower recovery). The tuning knob is identical.
**→ Backup Window Optimization (Cross-Domain)**: Fuzzy checkpointing is the principle behind modern backup systems. Instead of stopping the database to take a consistent backup, they capture a fuzzy snapshot (using filesystem snapshots, copy-on-write, etc.) and let the application continue. Recovery from backup is equivalent to your checkpoint-based recovery.
---

![WAL System Architecture Map](./diagrams/diag-satellite-wal-system.svg)

---
[[CRITERIA_JSON: {"milestone_id": "wal-impl-m4", "criteria": ["Fuzzy checkpoint writes CHECKPOINT_BEGIN record at start, captures atomic snapshots of transaction table and dirty page table (via brief mutex lock), then writes CHECKPOINT_END record containing serialized snapshots", "Concurrent transactions continue executing during checkpoint without blocking; transactions may commit, abort, or start between CHECKPOINT_BEGIN and CHECKPOINT_END", "Master record stored in separate file (wal.master) with magic number, version, last_checkpoint_lsn, and CRC32; updated atomically via temp file + rename pattern", "Master record update occurs AFTER CHECKPOINT_END is fsynced to disk; crash before master update leaves previous checkpoint valid", "Recovery reads master record first, validates CRC, and starts Analysis phase from checkpoint_lsn if valid; falls back to log start if master record corrupted or missing", "Log truncation checks two conditions: (1) segment.end_lsn < checkpoint's minimum recLSN, AND (2) no active transaction's firstLSN falls within segment range", "Long-running transactions prevent log truncation; truncation only succeeds when all transactions active at checkpoint time have completed", "Checkpoint interval configurable via time_interval_ms or record_count_interval; default time-based interval of 30 seconds recommended", "Recovery time benchmark: with checkpoint at 50% log position, recovery time is <= 60% of recovery time without checkpoint (measured with 10000 transactions)", "Truncation safety verified by test: delete segments, run recovery, confirm all committed transactions present and correct", "Checkpoint scheduler runs in background thread, checking interval triggers every 100ms, performing checkpoint + truncation when triggered", "Unit tests verify: master record atomic update (crash during write leaves previous valid), truncation blocked by long transaction, recovery with corrupted master record falls back gracefully"]}]
<!-- END_MS -->




# TDD

A production-grade Write-Ahead Log system implementing the ARIES recovery algorithm for ACID-compliant durability. The system provides crash-safe transaction logging with group commit optimization, fuzzy checkpointing for bounded recovery time, and Compensation Log Records (CLRs) for idempotent recovery even after crashes during recovery itself. The design follows the steal/no-force buffer pool policy, enabling maximum concurrency while guaranteeing that committed transactions survive any crash scenario.



<!-- TDD_MOD_ID: wal-impl-m1 -->
# Module: Log Record Format & Transaction API
## Module ID: wal-impl-m1
---
## 1. Module Charter
This module defines the binary format for Write-Ahead Log records and provides a transaction API that generates correctly-linked log records. It implements six record types (BEGIN, UPDATE, COMMIT, ABORT, CLR, CHECKPOINT) with a fixed 28-byte header containing LSN, type, txn_id, prev_lsn, and length fields. The module handles serialization to little-endian byte format with trailing CRC32 checksum and deserialization with corruption detection. The TransactionManager provides begin_txn(), write_txn(), commit_txn(), and abort_txn() operations, maintaining a TransactionTable that tracks active transactions with their first_lsn and last_lsn for prev_lsn chain construction. This module does NOT handle file persistence (Milestone 2), crash recovery (Milestone 3), or checkpoint scheduling (Milestone 4). Invariants: (1) LSNs are monotonically increasing and unique, (2) prev_lsn always points to the previous record of the same transaction (0 for first record), (3) every record ends with a CRC32 covering header + payload, (4) CLR's undo_next_lsn differs from prev_lsn and points to the next record to undo.
---
## 2. File Structure
```
wal/
├── include/
│   ├── wal_types.h          # [1] Core type definitions, constants, error codes
│   ├── wal_record.h         # [2] Record structures and serialization API
│   └── wal_transaction.h    # [3] Transaction manager and transaction table
└── src/
    ├── wal_record.c         # [4] Serialization/deserialization implementation
    └── wal_transaction.c    # [5] Transaction manager implementation
test/
├── test_record_serialization.c  # [6] Unit tests for record formats
└── test_transaction_api.c       # [7] Integration tests for transaction lifecycle
```
---
## 3. Complete Data Model
### 3.1 Core Constants and Types
```c
// wal_types.h
#ifndef WAL_TYPES_H
#define WAL_TYPES_H
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
// ============== CONSTANTS ==============
#define WAL_MAGIC               0x57414C00   // "WAL\0" - for future file header
#define HEADER_SIZE             28           // Fixed header size in bytes
#define CRC_SIZE                4            // CRC32 size in bytes
#define MAX_RECORD_SIZE         65536        // 64KB max record size
#define MAX_TXN_TABLE_SIZE      1024         // Max concurrent transactions
// Record type codes (stored as uint32_t in header)
typedef enum {
    RECORD_INVALID     = 0,
    RECORD_BEGIN       = 1,
    RECORD_UPDATE      = 2,
    RECORD_COMMIT      = 3,
    RECORD_ABORT       = 4,
    RECORD_CLR         = 5,
    RECORD_CHECKPOINT  = 6
} RecordType;
// Transaction status
typedef enum {
    TXN_INVALID   = 0,
    TXN_ACTIVE    = 1,
    TXN_COMMITTED = 2,
    TXN_ABORTED   = 3
} TxnStatus;
// Error codes
typedef enum {
    WAL_OK                   = 0,
    WAL_ERR_SERIALIZATION    = -1,   // Buffer too small, field overflow
    WAL_ERR_DESERIALIZATION  = -2,   // Malformed record
    WAL_ERR_CRC_MISMATCH     = -3,   // Corruption detected
    WAL_ERR_INVALID_TXN      = -4,   // Unknown transaction ID
    WAL_ERR_INVALID_STATE    = -5,   // Operation on committed/aborted txn
    WAL_ERR_INVALID_TYPE     = -6,   // Unknown record type
    WAL_ERR_ALLOC            = -7,   // Memory allocation failure
    WAL_ERR_BUFFER_OVERFLOW  = -8,   // Record exceeds MAX_RECORD_SIZE
    WAL_ERR_NULL_POINTER     = -9    // NULL argument
} WalError;
// ============== TYPE ALIASES ==============
typedef uint64_t Lsn;        // Log Sequence Number
typedef uint32_t TxnId;      // Transaction Identifier
typedef uint64_t PageId;     // Page Identifier
// Invalid/sentinel values
#define INVALID_LSN    ((Lsn)0)
#define INVALID_TXN_ID ((TxnId)0)
#define INVALID_PAGE_ID ((PageId)UINT64_MAX)
#endif // WAL_TYPES_H
```
### 3.2 Log Record Header
Every record begins with this fixed 28-byte header. All multi-byte integers are stored little-endian.
```

![Log Record Header Binary Layout](./diagrams/tdd-diag-m1-01.svg)

```
**Byte Layout:**
| Offset | Size | Field      | Type      | Description                              |
|--------|------|------------|-----------|------------------------------------------|
| 0x00   | 8    | lsn        | uint64_t  | Unique, monotonically increasing ID      |
| 0x08   | 4    | type       | uint32_t  | RecordType enum value                    |
| 0x0C   | 4    | txn_id     | uint32_t  | Transaction this record belongs to       |
| 0x10   | 8    | prev_lsn   | uint64_t  | Previous record in same transaction      |
| 0x18   | 4    | length     | uint32_t  | Total record size (header + payload + CRC)|
```c
// wal_record.h
typedef struct {
    Lsn        lsn;       // Byte offset 0x00-0x07
    RecordType type;      // Byte offset 0x08-0x0B
    TxnId      txn_id;    // Byte offset 0x0C-0x0F
    Lsn        prev_lsn;  // Byte offset 0x10-0x17
    uint32_t   length;    // Byte offset 0x18-0x1B
} LogRecordHeader;        // Total: 28 bytes (0x1C)
_Static_assert(sizeof(LogRecordHeader) == 28, "Header must be exactly 28 bytes");
```
**Field Semantics:**
- **lsn**: Assigned at record creation. Must be > 0 (0 is INVALID_LSN). Serves as both unique identifier and, in M2, byte offset into log file.
- **type**: Must be a valid RecordType enum value. Unknown values cause WAL_ERR_INVALID_TYPE during deserialization.
- **txn_id**: Links record to transaction. For CHECKPOINT records, this is 0 (system operation, not a user transaction).
- **prev_lsn**: Forms backward chain within transaction. First record of transaction has prev_lsn = 0. Enables O(1) undo traversal.
- **length**: Includes HEADER_SIZE (28) + payload size + CRC_SIZE (4). Must be >= 32 (minimum for BEGIN record).
### 3.3 BEGIN Record
Marks transaction start. No variable payload.
```
Byte Layout:
+------------------+------------------+------------------+------------------+
|    Header (28 bytes)                                                  |
+------------------+------------------+------------------+------------------+
|    CRC32 (4 bytes)                                                    |
+------------------+------------------+------------------+------------------+
Total: 32 bytes
```
```c
typedef struct {
    LogRecordHeader header;
    // No payload for BEGIN
} BeginRecord;
#define BEGIN_RECORD_SIZE (HEADER_SIZE + CRC_SIZE)  // 32 bytes
```
### 3.4 UPDATE Record
The workhorse record. Stores page modification with before/after images.
```

![Record Type Hierarchy](./diagrams/tdd-diag-m1-02.svg)

```
**Byte Layout:**
| Offset          | Size        | Field          | Type      | Description                    |
|-----------------|-------------|----------------|-----------|--------------------------------|
| 0x00            | 28          | header         | -         | Fixed header                   |
| 0x1C            | 8           | page_id        | uint64_t  | Page being modified            |
| 0x24            | 4           | offset         | uint32_t  | Byte offset within page        |
| 0x28            | 4           | old_value_len  | uint32_t  | Length of before-image         |
| 0x2C            | old_len     | old_value      | uint8_t[] | Before-image (for undo)        |
| 0x2C+old_len    | 4           | new_value_len  | uint32_t  | Length of after-image          |
| 0x30+old_len    | new_len     | new_value      | uint8_t[] | After-image (for redo)         |
| 0x30+old+new    | 4           | crc32          | uint32_t  | CRC of preceding bytes         |
```c
typedef struct {
    LogRecordHeader header;
    PageId    page_id;        // 8 bytes
    uint32_t  offset;         // 4 bytes
    // Variable-length fields (stored as length-prefixed in serialized form)
    uint32_t  old_value_len;
    uint8_t*  old_value;      // NULL if old_value_len == 0
    uint32_t  new_value_len;
    uint8_t*  new_value;      // NULL if new_value_len == 0
} UpdateRecord;
// Minimum UPDATE size: header(28) + page_id(8) + offset(4) + old_len(4) + new_len(4) + crc(4) = 52 bytes
#define UPDATE_RECORD_MIN_SIZE 52
```
**Memory Ownership:**
- `old_value` and `new_value` are owned by the struct after deserialization
- Caller must call `update_record_destroy()` to free
- For serialization, pointers reference caller-owned memory (not freed)
### 3.5 COMMIT Record
Marks successful transaction completion. Durability is guaranteed when this record is fsynced (M2).
```c
typedef struct {
    LogRecordHeader header;
    // No payload for COMMIT
} CommitRecord;
#define COMMIT_RECORD_SIZE (HEADER_SIZE + CRC_SIZE)  // 32 bytes
```
### 3.6 ABORT Record
Written after all undo work completes. Indicates transaction rolled back.
```c
typedef struct {
    LogRecordHeader header;
    // No payload for ABORT
} AbortRecord;
#define ABORT_RECORD_SIZE (HEADER_SIZE + CRC_SIZE)  // 32 bytes
```
### 3.7 CLR (Compensation Log Record)
Critical for crash-safe undo. Written during undo phase to record progress.
```

![UPDATE Record Binary Layout](./diagrams/tdd-diag-m1-03.svg)

```
**Byte Layout:**
| Offset        | Size        | Field          | Type      | Description                         |
|---------------|-------------|----------------|-----------|-------------------------------------|
| 0x00          | 28          | header         | -         | Fixed header                        |
| 0x1C          | 8           | page_id        | uint64_t  | Page being undone                   |
| 0x24          | 4           | offset         | uint32_t  | Byte offset within page             |
| 0x28          | 4           | undo_data_len  | uint32_t  | Length of undo data                 |
| 0x2C          | undo_len    | undo_data      | uint8_t[] | Data to write (before-image)        |
| 0x2C+undo_len | 8           | undo_next_lsn  | uint64_t  | Next record to undo (NOT prev_lsn!) |
| 0x34+undo_len | 4           | crc32          | uint32_t  | CRC of preceding bytes              |
```c
typedef struct {
    LogRecordHeader header;
    PageId    page_id;
    uint32_t  offset;
    uint32_t  undo_data_len;
    uint8_t*  undo_data;
    Lsn       undo_next_lsn;   // CRITICAL: Points to next record to undo
} ClrRecord;
#define CLR_RECORD_MIN_SIZE (HEADER_SIZE + 8 + 4 + 4 + 8 + CRC_SIZE)  // 56 bytes
```
**CRITICAL DISTINCTION:**
```
{{DIAGRAM:tdd-diag-m1-04}}
```
| Field          | Source          | Meaning                                      |
|----------------|-----------------|----------------------------------------------|
| header.prev_lsn| Last record of txn | Previous record in transaction chain       |
| undo_next_lsn  | UPDATE's prev_lsn | Skip this undone record, undo this next    |
Example: Undoing UPDATE at LSN 200 (prev_lsn=150):
- CLR written at LSN 250
- CLR.header.prev_lsn = 200 (links CLR into transaction chain)
- CLR.undo_next_lsn = 150 (after crash, resume undo from 150, skip 200)
### 3.8 CHECKPOINT Record
Captures transaction table and dirty page table snapshot. Variable-length based on active state.
```c
// Entry in checkpoint's transaction table
typedef struct {
    TxnId     txn_id;
    TxnStatus status;
    Lsn       first_lsn;
    Lsn       last_lsn;
} TxnTableEntry;  // 24 bytes per entry
// Entry in checkpoint's dirty page table
typedef struct {
    PageId    page_id;
    Lsn       rec_lsn;    // First LSN that dirtied this page
} DirtyPageEntry;  // 16 bytes per entry
typedef struct {
    LogRecordHeader header;
    uint32_t  num_active_txns;
    TxnTableEntry* active_txns;   // Array of num_active_txns entries
    uint32_t  num_dirty_pages;
    DirtyPageEntry* dirty_pages;  // Array of num_dirty_pages entries
} CheckpointRecord;
// Serialized layout:
// [header: 28] [num_active_txns: 4] [num_dirty_pages: 4]
// [active_txns: 24 * num_active_txns] [dirty_pages: 16 * num_dirty_pages]
// [crc32: 4]
#define CHECKPOINT_RECORD_MIN_SIZE (HEADER_SIZE + 4 + 4 + CRC_SIZE)  // 40 bytes
#define TXN_ENTRY_SERIALIZED_SIZE 24
#define DIRTY_PAGE_ENTRY_SERIALIZED_SIZE 16
```
### 3.9 Transaction Table
Tracks active transactions for prev_lsn linking.
```c
// wal_transaction.h
typedef struct {
    TxnId     txn_id;
    TxnStatus status;
    Lsn       first_lsn;   // LSN of BEGIN record
    Lsn       last_lsn;    // LSN of most recent record (for prev_lsn linking)
} TransactionEntry;
typedef struct {
    TransactionEntry* entries;
    size_t    capacity;
    size_t    count;
} TransactionTable;
```
### 3.10 Transaction Manager
Orchestrates transaction lifecycle and record generation.
```c
typedef struct TransactionManager TransactionManager;
struct TransactionManager {
    // LSN allocation
    Lsn       next_lsn;          // Next LSN to assign (monotonically increasing)
    // Transaction tracking
    TransactionTable txn_table;  // Active transactions
    // Record callback (for M2 integration, NULL in M1)
    // In M1, we just track state; M2 will plug in actual writing
    int (*write_record_fn)(TransactionManager* mgr, const uint8_t* data, size_t len);
    void*    write_record_ctx;
    // Concurrency (prepared for M2, unused in M1 single-threaded)
    // pthread_mutex_t lock;  // Uncomment in M2
};
// In M1, records are returned to caller rather than written to file
typedef struct {
    uint8_t*  data;
    size_t    len;
    Lsn       lsn;
} SerializedRecord;
```
---
## 4. Interface Contracts
### 4.1 Serialization Functions
```c
// wal_record.h
// ============== CRC COMPUTATION ==============
/**
 * Compute CRC32 checksum over data.
 * Uses zlib's crc32() function internally.
 * 
 * @param data   Pointer to data to checksum
 * @param len    Length of data in bytes
 * @return       CRC32 value
 */
uint32_t wal_compute_crc(const uint8_t* data, size_t len);
// ============== HEADER SERIALIZATION ==============
/**
 * Serialize header to little-endian byte buffer.
 * 
 * @param header  Header to serialize
 * @param buf     Output buffer (must be >= HEADER_SIZE bytes)
 * @return        WAL_OK on success, WAL_ERR_NULL_POINTER if buf is NULL
 */
int wal_serialize_header(const LogRecordHeader* header, uint8_t* buf);
/**
 * Deserialize header from little-endian byte buffer.
 * 
 * @param buf     Input buffer (must be >= HEADER_SIZE bytes)
 * @param header  Output header structure
 * @return        WAL_OK on success
 */
int wal_deserialize_header(const uint8_t* buf, LogRecordHeader* header);
// ============== BEGIN RECORD ==============
/**
 * Serialize BEGIN record.
 * 
 * @param rec     Record to serialize (header fields must be populated)
 * @param buf     Output buffer
 * @param buf_len Buffer capacity
 * @param[out] written Bytes written on success
 * @return        WAL_OK, WAL_ERR_NULL_POINTER, WAL_ERR_BUFFER_OVERFLOW
 * 
 * Postcondition: *written == BEGIN_RECORD_SIZE (32)
 */
int wal_serialize_begin(const BeginRecord* rec, uint8_t* buf, size_t buf_len, size_t* written);
/**
 * Deserialize BEGIN record.
 * 
 * @param buf        Input buffer
 * @param buf_len    Buffer size
 * @param[out] rec   Deserialized record (caller does NOT own any memory)
 * @param[out] consumed Bytes consumed from buffer
 * @return        WAL_OK, WAL_ERR_DESERIALIZATION, WAL_ERR_CRC_MISMATCH
 */
int wal_deserialize_begin(const uint8_t* buf, size_t buf_len, BeginRecord* rec, size_t* consumed);
// ============== UPDATE RECORD ==============
/**
 * Serialize UPDATE record with variable-length values.
 * 
 * @param rec     Record to serialize
 * @param buf     Output buffer
 * @param buf_len Buffer capacity
 * @param[out] written Bytes written on success
 * @return        WAL_OK, WAL_ERR_NULL_POINTER, WAL_ERR_BUFFER_OVERFLOW,
 *                WAL_ERR_SERIALIZATION (if values exceed MAX_RECORD_SIZE)
 * 
 * The serialized format:
 *   [header: 28][page_id: 8][offset: 4][old_len: 4][old_val][new_len: 4][new_val][crc: 4]
 */
int wal_serialize_update(const UpdateRecord* rec, uint8_t* buf, size_t buf_len, size_t* written);
/**
 * Deserialize UPDATE record.
 * 
 * @param buf        Input buffer
 * @param buf_len    Buffer size
 * @param[out] rec   Deserialized record (old_value and new_value are ALLOCATED,
 *                   caller MUST call wal_destroy_update() to free)
 * @param[out] consumed Bytes consumed
 * @return        WAL_OK, WAL_ERR_DESERIALIZATION, WAL_ERR_CRC_MISMATCH,
 *                WAL_ERR_ALLOC (memory allocation failed)
 */
int wal_deserialize_update(const uint8_t* buf, size_t buf_len, UpdateRecord* rec, size_t* consumed);
/**
 * Free memory allocated by deserialization.
 * Safe to call with NULL pointers.
 */
void wal_destroy_update(UpdateRecord* rec);
// ============== COMMIT RECORD ==============
int wal_serialize_commit(const CommitRecord* rec, uint8_t* buf, size_t buf_len, size_t* written);
int wal_deserialize_commit(const uint8_t* buf, size_t buf_len, CommitRecord* rec, size_t* consumed);
// ============== ABORT RECORD ==============
int wal_serialize_abort(const AbortRecord* rec, uint8_t* buf, size_t buf_len, size_t* written);
int wal_deserialize_abort(const uint8_t* buf, size_t buf_len, AbortRecord* rec, size_t* consumed);
// ============== CLR RECORD ==============
/**
 * Serialize CLR record.
 * 
 * @param rec     Record to serialize
 * @param buf     Output buffer
 * @param buf_len Buffer capacity
 * @param[out] written Bytes written
 * @return        WAL_OK or error code
 * 
 * IMPORTANT: undo_next_lsn must be set correctly (see 3.7)
 */
int wal_serialize_clr(const ClrRecord* rec, uint8_t* buf, size_t buf_len, size_t* written);
/**
 * Deserialize CLR record.
 * Caller must call wal_destroy_clr() to free undo_data.
 */
int wal_deserialize_clr(const uint8_t* buf, size_t buf_len, ClrRecord* rec, size_t* consumed);
void wal_destroy_clr(ClrRecord* rec);
// ============== CHECKPOINT RECORD ==============
/**
 * Serialize CHECKPOINT record.
 * 
 * @param rec     Record to serialize (arrays must be populated)
 * @param buf     Output buffer
 * @param buf_len Buffer capacity
 * @param[out] written Bytes written
 * @return        WAL_OK or error code
 */
int wal_serialize_checkpoint(const CheckpointRecord* rec, uint8_t* buf, size_t buf_len, size_t* written);
/**
 * Deserialize CHECKPOINT record.
 * Caller must call wal_destroy_checkpoint() to free arrays.
 */
int wal_deserialize_checkpoint(const uint8_t* buf, size_t buf_len, CheckpointRecord* rec, size_t* consumed);
void wal_destroy_checkpoint(CheckpointRecord* rec);
```
### 4.2 Transaction Table Functions
```c
// wal_transaction.h
/**
 * Initialize transaction table with given capacity.
 * 
 * @param table    Table to initialize
 * @param capacity Initial capacity (will grow if needed)
 * @return         WAL_OK or WAL_ERR_ALLOC
 */
int txn_table_init(TransactionTable* table, size_t capacity);
/**
 * Destroy transaction table, freeing all memory.
 */
void txn_table_destroy(TransactionTable* table);
/**
 * Find transaction entry by ID.
 * 
 * @param table   Table to search
 * @param txn_id  Transaction ID to find
 * @return        Pointer to entry or NULL if not found
 */
TransactionEntry* txn_table_find(TransactionTable* table, TxnId txn_id);
const TransactionEntry* txn_table_find_const(const TransactionTable* table, TxnId txn_id);
/**
 * Add new transaction entry.
 * 
 * @param table   Table to modify
 * @param txn_id  New transaction ID
 * @param status  Initial status
 * @param first_lsn LSN of BEGIN record
 * @return        Pointer to new entry or NULL on allocation failure
 */
TransactionEntry* txn_table_add(TransactionTable* table, TxnId txn_id, TxnStatus status, Lsn first_lsn);
/**
 * Remove transaction entry (after commit/abort completes).
 * 
 * @param table   Table to modify
 * @param txn_id  Transaction ID to remove
 * @return        WAL_OK or WAL_ERR_INVALID_TXN
 */
int txn_table_remove(TransactionTable* table, TxnId txn_id);
```
### 4.3 Transaction Manager Functions
```c
// wal_transaction.h
/**
 * Create transaction manager.
 * 
 * @return        Allocated manager or NULL on failure
 */
TransactionManager* txn_manager_create(void);
/**
 * Destroy transaction manager.
 * Frees all resources including transaction table.
 */
void txn_manager_destroy(TransactionManager* mgr);
/**
 * Begin a new transaction.
 * 
 * @param mgr     Transaction manager
 * @param[out] record Serialized BEGIN record (caller must free record->data)
 * @return        Transaction ID (> 0) or INVALID_TXN_ID on error
 * 
 * Side effects:
 *   - Allocates new txn_id (increments internal counter)
 *   - Adds entry to transaction table with status=TXN_ACTIVE
 *   - Allocates and populates record->data (caller must free)
 *   - Sets record->lsn to assigned LSN
 * 
 * Error conditions:
 *   - NULL pointer -> INVALID_TXN_ID, record unchanged
 *   - Allocation failure -> INVALID_TXN_ID, record unchanged
 */
TxnId txn_begin(TransactionManager* mgr, SerializedRecord* record);
/**
 * Log a write operation within a transaction.
 * 
 * @param mgr       Transaction manager
 * @param txn_id    Transaction ID (must be TXN_ACTIVE)
 * @param page_id   Page being modified
 * @param offset    Byte offset within page
 * @param old_val   Before-image (for undo)
 * @param old_len   Length of old_val (can be 0)
 * @param new_val   After-image (for redo)
 * @param new_len   Length of new_val (can be 0)
 * @param[out] record Serialized UPDATE record (caller must free record->data)
 * @return        WAL_OK or error code
 * 
 * Side effects:
 *   - Updates transaction's last_lsn to new record's LSN
 *   - Links new record via prev_lsn to previous last_lsn
 *   - Allocates record->data (caller must free)
 * 
 * Error conditions:
 *   - NULL pointer -> WAL_ERR_NULL_POINTER
 *   - Invalid txn_id -> WAL_ERR_INVALID_TXN
 *   - Non-ACTIVE transaction -> WAL_ERR_INVALID_STATE
 *   - Record too large -> WAL_ERR_BUFFER_OVERFLOW
 */
int txn_write(TransactionManager* mgr, TxnId txn_id,
              PageId page_id, uint32_t offset,
              const void* old_val, size_t old_len,
              const void* new_val, size_t new_len,
              SerializedRecord* record);
/**
 * Commit a transaction.
 * 
 * @param mgr     Transaction manager
 * @param txn_id  Transaction ID (must be TXN_ACTIVE)
 * @param[out] record Serialized COMMIT record (caller must free)
 * @return        WAL_OK or error code
 * 
 * Side effects:
 *   - Changes transaction status to TXN_COMMITTED
 *   - Updates transaction's last_lsn
 *   - In M2+, will trigger fsync; in M1, just generates record
 * 
 * Error conditions:
 *   - Same as txn_write()
 */
int txn_commit(TransactionManager* mgr, TxnId txn_id, SerializedRecord* record);
/**
 * Abort a transaction.
 * 
 * @param mgr     Transaction manager
 * @param txn_id  Transaction ID (must be TXN_ACTIVE)
 * @param[out] records Array of serialized records (CLRs + ABORT)
 * @param[in,out] num_records Input: array capacity; Output: records generated
 * @return        WAL_OK or error code
 * 
 * Side effects:
 *   - Changes transaction status to TXN_ABORTED
 *   - Generates CLR for each UPDATE (in reverse order via prev_lsn chain)
 *   - Generates ABORT record at end
 * 
 * IMPORTANT: In M1, this function cannot actually read back UPDATE records
 * to generate CLRs (no persistence yet). It returns WAL_ERR_NOT_IMPLEMENTED
 * or requires caller to provide undo information. M3 will implement full abort.
 * 
 * For M1 testing: abort_txn_simple() just writes ABORT record without CLRs.
 */
int txn_abort(TransactionManager* mgr, TxnId txn_id, 
              SerializedRecord* records, size_t* num_records);
/**
 * Simple abort for M1 testing (just writes ABORT record).
 * Does NOT generate CLRs (requires M3 recovery infrastructure).
 */
int txn_abort_simple(TransactionManager* mgr, TxnId txn_id, SerializedRecord* record);
/**
 * Free a serialized record's data buffer.
 */
void serialized_record_destroy(SerializedRecord* record);
```
### 4.4 Helper Functions
```c
// wal_record.h
/**
 * Get human-readable name for record type.
 */
const char* wal_record_type_name(RecordType type);
/**
 * Get human-readable name for transaction status.
 */
const char* wal_txn_status_name(TxnStatus status);
/**
 * Get human-readable name for error code.
 */
const char* wal_error_name(WalError err);
/**
 * Calculate total serialized size for UPDATE record.
 */
size_t wal_update_record_size(uint32_t old_len, uint32_t new_len);
/**
 * Calculate total serialized size for CLR record.
 */
size_t wal_clr_record_size(uint32_t undo_data_len);
/**
 * Calculate total serialized size for CHECKPOINT record.
 */
size_t wal_checkpoint_record_size(uint32_t num_txns, uint32_t num_pages);
/**
 * Validate record header fields.
 * 
 * @param header  Header to validate
 * @return        WAL_OK if valid, error code otherwise
 * 
 * Checks:
 *   - lsn > 0
 *   - type is valid RecordType
 *   - length >= 32 (minimum record size)
 */
int wal_validate_header(const LogRecordHeader* header);
```
---
## 5. Algorithm Specifications
### 5.1 Little-Endian Serialization
All multi-byte integers are serialized little-endian (LSB first). This matches x86-64 and ARM64 native byte order, avoiding conversion overhead on common platforms.
```c
// Write 64-bit little-endian
static inline void write_le64(uint8_t* buf, uint64_t val) {
    buf[0] = (uint8_t)(val & 0xFF);
    buf[1] = (uint8_t)((val >> 8) & 0xFF);
    buf[2] = (uint8_t)((val >> 16) & 0xFF);
    buf[3] = (uint8_t)((val >> 24) & 0xFF);
    buf[4] = (uint8_t)((val >> 32) & 0xFF);
    buf[5] = (uint8_t)((val >> 40) & 0xFF);
    buf[6] = (uint8_t)((val >> 48) & 0xFF);
    buf[7] = (uint8_t)((val >> 56) & 0xFF);
}
// Write 32-bit little-endian
static inline void write_le32(uint8_t* buf, uint32_t val) {
    buf[0] = (uint8_t)(val & 0xFF);
    buf[1] = (uint8_t)((val >> 8) & 0xFF);
    buf[2] = (uint8_t)((val >> 16) & 0xFF);
    buf[3] = (uint8_t)((val >> 24) & 0xFF);
}
// Read 64-bit little-endian
static inline uint64_t read_le64(const uint8_t* buf) {
    return ((uint64_t)buf[0]) |
           ((uint64_t)buf[1] << 8) |
           ((uint64_t)buf[2] << 16) |
           ((uint64_t)buf[3] << 24) |
           ((uint64_t)buf[4] << 32) |
           ((uint64_t)buf[5] << 40) |
           ((uint64_t)buf[6] << 48) |
           ((uint64_t)buf[7] << 56);
}
// Read 32-bit little-endian
static inline uint32_t read_le32(const uint8_t* buf) {
    return ((uint32_t)buf[0]) |
           ((uint32_t)buf[1] << 8) |
           ((uint32_t)buf[2] << 16) |
           ((uint32_t)buf[3] << 24);
}
```
### 5.2 UPDATE Record Serialization Algorithm
```

![prev_lsn Chain Across Transactions](./diagrams/tdd-diag-m1-05.svg)

```
**Procedure serialize_update(rec, buf, buf_len, written):**
1. **Validate inputs**
   - IF rec == NULL OR buf == NULL: RETURN WAL_ERR_NULL_POINTER
   - Calculate required_size = UPDATE_RECORD_MIN_SIZE + rec->old_value_len + rec->new_value_len
   - IF required_size > MAX_RECORD_SIZE: RETURN WAL_ERR_BUFFER_OVERFLOW
   - IF required_size > buf_len: RETURN WAL_ERR_BUFFER_OVERFLOW
2. **Initialize write cursor**
   - offset = 0
3. **Write header (28 bytes)**
   - write_le64(buf + offset, rec->header.lsn); offset += 8
   - write_le32(buf + offset, rec->header.type); offset += 4
   - write_le32(buf + offset, rec->header.txn_id); offset += 4
   - write_le64(buf + offset, rec->header.prev_lsn); offset += 8
   - write_le32(buf + offset, required_size); offset += 4  // length field
4. **Write UPDATE-specific fixed fields (12 bytes)**
   - write_le64(buf + offset, rec->page_id); offset += 8
   - write_le32(buf + offset, rec->offset); offset += 4
5. **Write before-image (length-prefixed)**
   - write_le32(buf + offset, rec->old_value_len); offset += 4
   - IF rec->old_value_len > 0 AND rec->old_value != NULL:
     - memcpy(buf + offset, rec->old_value, rec->old_value_len)
     - offset += rec->old_value_len
6. **Write after-image (length-prefixed)**
   - write_le32(buf + offset, rec->new_value_len); offset += 4
   - IF rec->new_value_len > 0 AND rec->new_value != NULL:
     - memcpy(buf + offset, rec->new_value, rec->new_value_len)
     - offset += rec->new_value_len
7. **Compute and write CRC**
   - crc = wal_compute_crc(buf, offset)
   - write_le32(buf + offset, crc); offset += 4
8. **Return success**
   - *written = offset
   - ASSERT offset == required_size
   - RETURN WAL_OK
### 5.3 UPDATE Record Deserialization Algorithm
**Procedure deserialize_update(buf, buf_len, rec, consumed):**
1. **Validate inputs**
   - IF buf == NULL OR rec == NULL: RETURN WAL_ERR_NULL_POINTER
   - IF buf_len < HEADER_SIZE: RETURN WAL_ERR_DESERIALIZATION
2. **Read and validate header**
   - Parse header using wal_deserialize_header()
   - err = wal_validate_header(&rec->header)
   - IF err != WAL_OK: RETURN err
   - IF rec->header.type != RECORD_UPDATE: RETURN WAL_ERR_INVALID_TYPE
   - IF buf_len < rec->header.length: RETURN WAL_ERR_DESERIALIZATION
3. **Verify CRC**
   - stored_crc = read_le32(buf + rec->header.length - 4)
   - computed_crc = wal_compute_crc(buf, rec->header.length - 4)
   - IF stored_crc != computed_crc: RETURN WAL_ERR_CRC_MISMATCH
4. **Read UPDATE-specific fixed fields**
   - offset = HEADER_SIZE  // 28
   - rec->page_id = read_le64(buf + offset); offset += 8
   - rec->offset = read_le32(buf + offset); offset += 4
5. **Read before-image**
   - rec->old_value_len = read_le32(buf + offset); offset += 4
   - IF rec->old_value_len > 0:
     - rec->old_value = malloc(rec->old_value_len)
     - IF rec->old_value == NULL: RETURN WAL_ERR_ALLOC
     - memcpy(rec->old_value, buf + offset, rec->old_value_len)
     - offset += rec->old_value_len
   - ELSE:
     - rec->old_value = NULL
6. **Read after-image**
   - rec->new_value_len = read_le32(buf + offset); offset += 4
   - IF rec->new_value_len > 0:
     - rec->new_value = malloc(rec->new_value_len)
     - IF rec->new_value == NULL:
       - free(rec->old_value)  // Clean up partial allocation
       - RETURN WAL_ERR_ALLOC
     - memcpy(rec->new_value, buf + offset, rec->new_value_len)
     - offset += rec->new_value_len
   - ELSE:
     - rec->new_value = NULL
7. **Return success**
   - *consumed = rec->header.length
   - RETURN WAL_OK
### 5.4 Transaction Begin Algorithm
**Procedure txn_begin(mgr, record):**
1. **Validate inputs**
   - IF mgr == NULL OR record == NULL: RETURN INVALID_TXN_ID
2. **Allocate transaction ID**
   - txn_id = mgr->next_txn_id++
   - (Simple increment; M2+ may need more sophisticated allocation)
3. **Allocate LSN**
   - lsn = mgr->next_lsn
   - mgr->next_lsn += BEGIN_RECORD_SIZE  // LSN = byte offset
4. **Create BEGIN record**
   - rec.header.lsn = lsn
   - rec.header.type = RECORD_BEGIN
   - rec.header.txn_id = txn_id
   - rec.header.prev_lsn = 0  // First record
   - rec.header.length = BEGIN_RECORD_SIZE
5. **Serialize record**
   - record->data = malloc(BEGIN_RECORD_SIZE)
   - IF record->data == NULL: RETURN INVALID_TXN_ID
   - wal_serialize_begin(&rec, record->data, BEGIN_RECORD_SIZE, &record->len)
   - record->lsn = lsn
6. **Add to transaction table**
   - entry = txn_table_add(&mgr->txn_table, txn_id, TXN_ACTIVE, lsn)
   - IF entry == NULL:
     - free(record->data)
     - RETURN INVALID_TXN_ID
   - entry->last_lsn = lsn
7. **Return transaction ID**
   - RETURN txn_id
### 5.5 Transaction Write Algorithm
```

![Transaction State Machine](./diagrams/tdd-diag-m1-06.svg)

```
**Procedure txn_write(mgr, txn_id, page_id, offset, old_val, old_len, new_val, new_len, record):**
1. **Validate inputs**
   - IF mgr == NULL OR record == NULL: RETURN WAL_ERR_NULL_POINTER
2. **Find transaction**
   - entry = txn_table_find(&mgr->txn_table, txn_id)
   - IF entry == NULL: RETURN WAL_ERR_INVALID_TXN
   - IF entry->status != TXN_ACTIVE: RETURN WAL_ERR_INVALID_STATE
3. **Calculate record size**
   - rec_size = wal_update_record_size(old_len, new_len)
   - IF rec_size > MAX_RECORD_SIZE: RETURN WAL_ERR_BUFFER_OVERFLOW
4. **Allocate LSN**
   - lsn = mgr->next_lsn
   - mgr->next_lsn += rec_size
5. **Create UPDATE record**
   - rec.header.lsn = lsn
   - rec.header.type = RECORD_UPDATE
   - rec.header.txn_id = txn_id
   - rec.header.prev_lsn = entry->last_lsn  // Link to previous
   - rec.header.length = rec_size
   - rec.page_id = page_id
   - rec.offset = offset
   - rec.old_value_len = old_len
   - rec.old_value = (uint8_t*)old_val  // Cast away const for struct
   - rec.new_value_len = new_len
   - rec.new_value = (uint8_t*)new_val
6. **Serialize record**
   - record->data = malloc(rec_size)
   - IF record->data == NULL: RETURN WAL_ERR_ALLOC
   - err = wal_serialize_update(&rec, record->data, rec_size, &record->len)
   - IF err != WAL_OK:
     - free(record->data)
     - RETURN err
   - record->lsn = lsn
7. **Update transaction entry**
   - entry->last_lsn = lsn
8. **Return success**
   - RETURN WAL_OK
### 5.6 prev_lsn Chain Construction
The prev_lsn chain is the critical data structure enabling efficient undo. Each record's prev_lsn points to the previous record of the **same transaction**, not the previous record in the log.
```

![Serialization/Deserialization Round-Trip](./diagrams/tdd-diag-m1-07.svg)

```
**Example:**
```
Time  T1 (txn_id=1)           T2 (txn_id=2)
----  -------------------     -------------------
t0    BEGIN [lsn=100]         
      prev_lsn=0
t1                            BEGIN [lsn=150]
                              prev_lsn=0
t2    UPDATE [lsn=200]        
      prev_lsn=100
t3                            UPDATE [lsn=250]
                              prev_lsn=150
t4    UPDATE [lsn=300]        
      prev_lsn=200
t5                            COMMIT [lsn=350]
                              prev_lsn=250
t6    COMMIT [lsn=400]        
      prev_lsn=300
```
**Chain for T1:** 400 → 300 → 200 → 100 → 0 (BEGIN)
**Chain for T2:** 350 → 250 → 150 → 0 (BEGIN)
To undo T1, follow chain: start at 400, then 300, then 200, then 100 (BEGIN = stop).
---
## 6. Error Handling Matrix
| Error                    | Detected By                          | Recovery Action                           | User-Visible? |
|--------------------------|--------------------------------------|-------------------------------------------|---------------|
| WAL_ERR_NULL_POINTER     | All public functions at entry        | Return error immediately, no side effects | Yes (API)     |
| WAL_ERR_SERIALIZATION    | Serialize functions on field overflow| Return error, no partial output           | Yes (API)     |
| WAL_ERR_DESERIALIZATION  | Deserialize on truncated input       | Return error, rec may be partially filled | Yes (API)     |
| WAL_ERR_CRC_MISMATCH     | Deserialize on CRC verification      | Return error, rec unchanged               | Yes (API)     |
| WAL_ERR_INVALID_TXN      | txn_* on unknown txn_id              | Return error, no state change             | Yes (API)     |
| WAL_ERR_INVALID_STATE    | txn_* on committed/aborted txn       | Return error, no state change             | Yes (API)     |
| WAL_ERR_INVALID_TYPE     | Deserialize on unknown record type   | Return error, rec header filled           | Yes (API)     |
| WAL_ERR_ALLOC            | Any malloc failure                   | Return error, no partial allocation       | Yes (API)     |
| WAL_ERR_BUFFER_OVERFLOW  | Serialize on record > MAX_RECORD_SIZE| Return error, no output                   | Yes (API)     |
**Invariant: No error path leaves the TransactionManager or TransactionTable in an inconsistent state.**
If an error occurs mid-operation:
1. Do not modify transaction table
2. Do not modify next_lsn
3. Free any partially allocated memory
4. Return error code
---
## 7. Implementation Sequence with Checkpoints
### Phase 1: Header and Fixed Record Types (1-2 hours)
**Files to create:** `include/wal_types.h`, `include/wal_record.h`, `src/wal_record.c` (partial)
**Tasks:**
1. Define all constants, enums, and error codes in `wal_types.h`
2. Define `LogRecordHeader` struct with static assert for size
3. Implement `wal_compute_crc()` using zlib
4. Implement `write_le64/32` and `read_le64/32` inline functions
5. Implement `wal_serialize_header()` and `wal_deserialize_header()`
6. Implement `wal_validate_header()`
7. Implement BEGIN record serialize/deserialize
8. Implement COMMIT record serialize/deserialize
9. Implement ABORT record serialize/deserialize
**Checkpoint 1:**
```bash
# Compile and run basic header tests
gcc -c src/wal_record.c -I include -lz
# You should be able to:
# - Serialize/deserialize a header with round-trip fidelity
# - Serialize/deserialize BEGIN, COMMIT, ABORT records
# - Verify CRC catches corruption
./test_header_basics  # Should pass all tests
```
### Phase 2: UPDATE Record with Variable-Length Fields (1-2 hours)
**Files to create:** Continue `src/wal_record.c`
**Tasks:**
1. Define `UpdateRecord` struct
2. Implement `wal_update_record_size()` helper
3. Implement `wal_serialize_update()` with length-prefixed fields
4. Implement `wal_deserialize_update()` with memory allocation
5. Implement `wal_destroy_update()` for cleanup
**Checkpoint 2:**
```bash
# Test variable-length serialization
./test_update_record
# Should pass:
# - Empty values (old_len=0, new_len=0)
# - Small values (1-10 bytes)
# - Large values (1KB+)
# - Round-trip fidelity
# - CRC corruption detection
```
### Phase 3: CLR Record with undo_next_lsn (1 hour)
**Files to create:** Continue `src/wal_record.c`
**Tasks:**
1. Define `ClrRecord` struct
2. Implement `wal_clr_record_size()` helper
3. Implement `wal_serialize_clr()`
4. Implement `wal_deserialize_clr()`
5. Implement `wal_destroy_clr()`
**Checkpoint 3:**
```bash
./test_clr_record
# Should verify:
# - undo_next_lsn is distinct from prev_lsn
# - Round-trip fidelity
# - undo_data properly serialized/deserialized
```
### Phase 4: CHECKPOINT Record Structure (0.5 hours)
**Files to create:** Continue `src/wal_record.c`
**Tasks:**
1. Define `TxnTableEntry` and `DirtyPageEntry` structs
2. Define `CheckpointRecord` struct
3. Implement `wal_checkpoint_record_size()` helper
4. Implement `wal_serialize_checkpoint()` with array serialization
5. Implement `wal_deserialize_checkpoint()` with array allocation
6. Implement `wal_destroy_checkpoint()`
**Checkpoint 4:**
```bash
./test_checkpoint_record
# Should verify:
# - Empty checkpoint (0 txns, 0 pages)
# - Checkpoint with multiple entries
# - Round-trip fidelity
```
### Phase 5: Transaction API and State Tracking (1-2 hours)
**Files to create:** `include/wal_transaction.h`, `src/wal_transaction.c`
**Tasks:**
1. Define `TransactionEntry` and `TransactionTable` structs
2. Implement `txn_table_init()`, `txn_table_destroy()`
3. Implement `txn_table_find()`, `txn_table_add()`, `txn_table_remove()`
4. Define `TransactionManager` struct
5. Implement `txn_manager_create()`, `txn_manager_destroy()`
6. Implement `txn_begin()` with record generation
7. Implement `txn_write()` with prev_lsn linking
8. Implement `txn_commit()` with record generation
9. Implement `txn_abort_simple()` (just ABORT record, no CLRs)
10. Implement `serialized_record_destroy()`
**Checkpoint 5:**
```bash
./test_transaction_api
# Should verify:
# - begin returns valid txn_id
# - write creates UPDATE with correct prev_lsn
# - commit creates COMMIT record
# - Multiple transactions have independent prev_lsn chains
# - Error handling for invalid txn_id, wrong state
```
### Phase 6: Unit Tests and Round-Trip Verification (1-2 hours)
**Files to create:** `test/test_record_serialization.c`, `test/test_transaction_api.c`
**Tasks:**
1. Create comprehensive test suite for each record type
2. Implement round-trip tests for all record types
3. Implement edge case tests (empty values, max sizes)
4. Implement corruption detection tests
5. Create transaction lifecycle integration test
6. Benchmark serialization performance
**Checkpoint 6 (Final):**
```bash
# Run all tests
./run_all_tests
# All tests should pass
# Benchmarks should meet targets:
# - Fixed record serialize: <100ns
# - Variable record serialize: <500ns
# - Fixed record deserialize: <200ns
# - Variable record deserialize: <1µs
```
---
## 8. Test Specification
### 8.1 Header Tests
```c
// test_header_basics.c
void test_header_roundtrip(void) {
    LogRecordHeader original = {
        .lsn = 0x0102030405060708ULL,
        .type = RECORD_UPDATE,
        .txn_id = 0x0A0B0C0D,
        .prev_lsn = 0xF0E0D0C0B0A09080ULL,
        .length = 0x11223344
    };
    uint8_t buf[HEADER_SIZE];
    assert(wal_serialize_header(&original, buf) == WAL_OK);
    LogRecordHeader parsed;
    assert(wal_deserialize_header(buf, &parsed) == WAL_OK);
    assert(parsed.lsn == original.lsn);
    assert(parsed.type == original.type);
    assert(parsed.txn_id == original.txn_id);
    assert(parsed.prev_lsn == original.prev_lsn);
    assert(parsed.length == original.length);
}
void test_header_validation(void) {
    LogRecordHeader h;
    // LSN must be > 0
    h.lsn = 0; h.type = RECORD_BEGIN; h.length = 32;
    assert(wal_validate_header(&h) == WAL_ERR_DESERIALIZATION);
    // Type must be valid
    h.lsn = 1; h.type = 999; h.length = 32;
    assert(wal_validate_header(&h) == WAL_ERR_INVALID_TYPE);
    // Length must be >= 32
    h.type = RECORD_BEGIN; h.length = 31;
    assert(wal_validate_header(&h) == WAL_ERR_DESERIALIZATION);
    // Valid header
    h.length = 32;
    assert(wal_validate_header(&h) == WAL_OK);
}
```
### 8.2 BEGIN Record Tests
```c
void test_begin_record_roundtrip(void) {
    BeginRecord original = {
        .header = {
            .lsn = 100,
            .type = RECORD_BEGIN,
            .txn_id = 42,
            .prev_lsn = 0,
            .length = BEGIN_RECORD_SIZE
        }
    };
    uint8_t buf[BEGIN_RECORD_SIZE];
    size_t written;
    assert(wal_serialize_begin(&original, buf, sizeof(buf), &written) == WAL_OK);
    assert(written == BEGIN_RECORD_SIZE);
    BeginRecord parsed;
    size_t consumed;
    assert(wal_deserialize_begin(buf, sizeof(buf), &parsed, &consumed) == WAL_OK);
    assert(consumed == BEGIN_RECORD_SIZE);
    assert(parsed.header.lsn == original.header.lsn);
    assert(parsed.header.type == RECORD_BEGIN);
    assert(parsed.header.txn_id == 42);
    assert(parsed.header.prev_lsn == 0);
}
void test_begin_crc_verification(void) {
    BeginRecord rec = { .header = { .lsn = 1, .type = RECORD_BEGIN, .txn_id = 1, .prev_lsn = 0 } };
    uint8_t buf[BEGIN_RECORD_SIZE];
    size_t written;
    wal_serialize_begin(&rec, buf, sizeof(buf), &written);
    // Corrupt a byte
    buf[10] ^= 0xFF;
    BeginRecord parsed;
    size_t consumed;
    assert(wal_deserialize_begin(buf, sizeof(buf), &parsed, &consumed) == WAL_ERR_CRC_MISMATCH);
}
```
### 8.3 UPDATE Record Tests
```c
void test_update_record_empty_values(void) {
    UpdateRecord original = {
        .header = { .lsn = 100, .type = RECORD_UPDATE, .txn_id = 1, .prev_lsn = 50 },
        .page_id = 7,
        .offset = 16,
        .old_value_len = 0,
        .old_value = NULL,
        .new_value_len = 0,
        .new_value = NULL
    };
    uint8_t buf[MAX_RECORD_SIZE];
    size_t written;
    assert(wal_serialize_update(&original, buf, sizeof(buf), &written) == WAL_OK);
    assert(written == UPDATE_RECORD_MIN_SIZE);  // 52 bytes
    UpdateRecord parsed = {0};
    size_t consumed;
    assert(wal_deserialize_update(buf, written, &parsed, &consumed) == WAL_OK);
    assert(consumed == written);
    assert(parsed.page_id == 7);
    assert(parsed.offset == 16);
    assert(parsed.old_value_len == 0);
    assert(parsed.old_value == NULL);
    assert(parsed.new_value_len == 0);
    assert(parsed.new_value == NULL);
    wal_destroy_update(&parsed);
}
void test_update_record_with_values(void) {
    uint8_t old_val[] = {0xDE, 0xAD, 0xBE, 0xEF};
    uint8_t new_val[] = {0xCA, 0xFE, 0xBA, 0xBE, 0x00};
    UpdateRecord original = {
        .header = { .lsn = 100, .type = RECORD_UPDATE, .txn_id = 1, .prev_lsn = 50 },
        .page_id = 7,
        .offset = 24,
        .old_value_len = sizeof(old_val),
        .old_value = old_val,
        .new_value_len = sizeof(new_val),
        .new_value = new_val
    };
    uint8_t buf[MAX_RECORD_SIZE];
    size_t written;
    assert(wal_serialize_update(&original, buf, sizeof(buf), &written) == WAL_OK);
    UpdateRecord parsed = {0};
    size_t consumed;
    assert(wal_deserialize_update(buf, written, &parsed, &consumed) == WAL_OK);
    assert(parsed.page_id == original.page_id);
    assert(parsed.offset == original.offset);
    assert(parsed.old_value_len == sizeof(old_val));
    assert(memcmp(parsed.old_value, old_val, sizeof(old_val)) == 0);
    assert(parsed.new_value_len == sizeof(new_val));
    assert(memcmp(parsed.new_value, new_val, sizeof(new_val)) == 0);
    wal_destroy_update(&parsed);
}
void test_update_record_buffer_overflow(void) {
    uint8_t large_value[100000];  // 100KB
    UpdateRecord rec = {
        .header = { .lsn = 1, .type = RECORD_UPDATE, .txn_id = 1, .prev_lsn = 0 },
        .page_id = 1,
        .offset = 0,
        .old_value_len = sizeof(large_value),
        .old_value = large_value,
        .new_value_len = sizeof(large_value),
        .new_value = large_value
    };
    uint8_t buf[MAX_RECORD_SIZE];
    size_t written;
    assert(wal_serialize_update(&rec, buf, sizeof(buf), &written) == WAL_ERR_BUFFER_OVERFLOW);
}
```
### 8.4 CLR Record Tests
```c
void test_clr_undo_next_lsn_distinct_from_prev_lsn(void) {
    uint8_t undo_data[] = {0xAA, 0xBB};
    ClrRecord rec = {
        .header = { .lsn = 300, .type = RECORD_CLR, .txn_id = 1, .prev_lsn = 200 },
        .page_id = 5,
        .offset = 10,
        .undo_data_len = sizeof(undo_data),
        .undo_data = undo_data,
        .undo_next_lsn = 150  // Must be prev_lsn of the UPDATE being undone (200 -> 150)
    };
    uint8_t buf[MAX_RECORD_SIZE];
    size_t written;
    assert(wal_serialize_clr(&rec, buf, sizeof(buf), &written) == WAL_OK);
    ClrRecord parsed = {0};
    size_t consumed;
    assert(wal_deserialize_clr(buf, written, &parsed, &consumed) == WAL_OK);
    // CRITICAL: These must be different!
    assert(parsed.header.prev_lsn == 200);   // Previous record in transaction
    assert(parsed.undo_next_lsn == 150);     // Next record to undo
    assert(parsed.header.prev_lsn != parsed.undo_next_lsn);
    wal_destroy_clr(&parsed);
}
```
### 8.5 Transaction API Tests
```c
void test_transaction_lifecycle(void) {
    TransactionManager* mgr = txn_manager_create();
    assert(mgr != NULL);
    SerializedRecord record;
    // Begin
    TxnId txn = txn_begin(mgr, &record);
    assert(txn != INVALID_TXN_ID);
    assert(record.lsn > 0);
    assert(record.len == BEGIN_RECORD_SIZE);
    serialized_record_destroy(&record);
    // Write
    uint8_t old_val[] = "A";
    uint8_t new_val[] = "B";
    int err = txn_write(mgr, txn, 1, 0, old_val, 1, new_val, 1, &record);
    assert(err == WAL_OK);
    assert(record.lsn > 0);
    // Verify prev_lsn chain
    LogRecordHeader header;
    wal_deserialize_header(record.data, &header);
    assert(header.prev_lsn == 1);  // Points to BEGIN record
    serialized_record_destroy(&record);
    // Commit
    err = txn_commit(mgr, txn, &record);
    assert(err == WAL_OK);
    serialized_record_destroy(&record);
    // Verify transaction is committed
    const TransactionEntry* entry = txn_table_find_const(&mgr->txn_table, txn);
    assert(entry == NULL);  // Removed after commit (or status = COMMITTED)
    txn_manager_destroy(mgr);
}
void test_prev_lsn_chain_multiple_writes(void) {
    TransactionManager* mgr = txn_manager_create();
    SerializedRecord record;
    TxnId txn = txn_begin(mgr, &record);
    Lsn begin_lsn = record.lsn;
    serialized_record_destroy(&record);
    Lsn prev_lsn = begin_lsn;
    for (int i = 0; i < 5; i++) {
        uint8_t val = (uint8_t)i;
        txn_write(mgr, txn, i, 0, &val, 1, &val, 1, &record);
        LogRecordHeader header;
        wal_deserialize_header(record.data, &header);
        assert(header.prev_lsn == prev_lsn);
        prev_lsn = record.lsn;
        serialized_record_destroy(&record);
    }
    txn_commit(mgr, txn, &record);
    serialized_record_destroy(&record);
    txn_manager_destroy(mgr);
}
void test_concurrent_transactions_independent_chains(void) {
    TransactionManager* mgr = txn_manager_create();
    SerializedRecord r1, r2;
    TxnId t1 = txn_begin(mgr, &r1);
    TxnId t2 = txn_begin(mgr, &r2);
    Lsn t1_lsn = r1.lsn;
    Lsn t2_lsn = r2.lsn;
    serialized_record_destroy(&r1);
    serialized_record_destroy(&r2);
    // T1 writes
    uint8_t v1 = 'A';
    txn_write(mgr, t1, 1, 0, &v1, 1, &v1, 1, &r1);
    LogRecordHeader h1;
    wal_deserialize_header(r1.data, &h1);
    assert(h1.prev_lsn == t1_lsn);  // T1's prev_lsn points to T1's BEGIN
    serialized_record_destroy(&r1);
    // T2 writes
    uint8_t v2 = 'B';
    txn_write(mgr, t2, 2, 0, &v2, 1, &v2, 1, &r2);
    LogRecordHeader h2;
    wal_deserialize_header(r2.data, &h2);
    assert(h2.prev_lsn == t2_lsn);  // T2's prev_lsn points to T2's BEGIN, NOT T1's write
    serialized_record_destroy(&r2);
    txn_commit(mgr, t1, &r1);
    txn_commit(mgr, t2, &r2);
    serialized_record_destroy(&r1);
    serialized_record_destroy(&r2);
    txn_manager_destroy(mgr);
}
void test_invalid_transaction_operations(void) {
    TransactionManager* mgr = txn_manager_create();
    SerializedRecord record;
    // Invalid txn_id
    assert(txn_write(mgr, 999, 1, 0, "A", 1, "B", 1, &record) == WAL_ERR_INVALID_TXN);
    assert(txn_commit(mgr, 999, &record) == WAL_ERR_INVALID_TXN);
    // Operation on committed transaction
    TxnId txn = txn_begin(mgr, &record);
    serialized_record_destroy(&record);
    txn_commit(mgr, txn, &record);
    serialized_record_destroy(&record);
    assert(txn_write(mgr, txn, 1, 0, "A", 1, "B", 1, &record) == WAL_ERR_INVALID_STATE);
    assert(txn_commit(mgr, txn, &record) == WAL_ERR_INVALID_STATE);
    txn_manager_destroy(mgr);
}
```
### 8.6 Round-Trip Verification Test
```c
void test_all_record_types_roundtrip(void) {
    // Test each record type with various sizes
    struct {
        RecordType type;
        const char* name;
    } types[] = {
        {RECORD_BEGIN, "BEGIN"},
        {RECORD_UPDATE, "UPDATE"},
        {RECORD_COMMIT, "COMMIT"},
        {RECORD_ABORT, "ABORT"},
        {RECORD_CLR, "CLR"},
        {RECORD_CHECKPOINT, "CHECKPOINT"}
    };
    for (int i = 0; i < 6; i++) {
        printf("Testing %s round-trip...\n", types[i].name);
        // ... create record of each type, serialize, deserialize, verify
    }
    printf("All round-trip tests passed!\n");
}
```
---
## 9. Performance Targets
| Operation                          | Target         | How to Measure                          |
|------------------------------------|----------------|-----------------------------------------|
| BEGIN record serialize             | < 100 ns       | `./benchmark --operation=serialize-begin --iterations=1000000` |
| COMMIT record serialize            | < 100 ns       | `./benchmark --operation=serialize-commit` |
| UPDATE record serialize (64 bytes) | < 500 ns       | `./benchmark --operation=serialize-update --size=64` |
| UPDATE record serialize (4KB)      | < 5 µs         | `./benchmark --operation=serialize-update --size=4096` |
| BEGIN record deserialize           | < 200 ns       | `./benchmark --operation=deserialize-begin` |
| UPDATE record deserialize (64B)    | < 1 µs         | `./benchmark --operation=deserialize-update --size=64` |
| CRC32 computation (1KB)            | < 500 ns       | `./benchmark --operation=crc32 --size=1024` |
| Round-trip fidelity                | 100%           | All test cases pass byte-compare check |
| Memory allocation per deserialize  | 2 allocs max   | Valgrind --track-allocs=yes             |
**Benchmark Template:**
```c
void benchmark_serialize_begin(void) {
    BeginRecord rec = { .header = { .lsn = 1, .type = RECORD_BEGIN, .txn_id = 1, .prev_lsn = 0 } };
    uint8_t buf[BEGIN_RECORD_SIZE];
    size_t written;
    const int ITERATIONS = 1000000;
    uint64_t start = get_time_ns();
    for (int i = 0; i < ITERATIONS; i++) {
        rec.header.lsn = i + 1;
        wal_serialize_begin(&rec, buf, sizeof(buf), &written);
    }
    uint64_t elapsed = get_time_ns() - start;
    printf("BEGIN serialize: %llu ns/op\n", elapsed / ITERATIONS);
}
```
---
## 10. Diagrams Reference
```

![Transaction Table Structure](./diagrams/tdd-diag-m1-08.svg)

```
```
{{DIAGRAM:tdd-diag-m1-09}}
```
```

![Transaction API Call Sequence](./diagrams/tdd-diag-m1-10.svg)

```
---
## 11. State Machine: Transaction Lifecycle
```
States: INVALID -> ACTIVE -> COMMITTED | ABORTED
Transitions:
  txn_begin()     : INVALID -> ACTIVE
  txn_write()     : ACTIVE -> ACTIVE (no state change)
  txn_commit()    : ACTIVE -> COMMITTED
  txn_abort()     : ACTIVE -> ABORTED
Illegal Transitions:
  ACTIVE -> ACTIVE (via begin)     : Error: txn_id already exists
  COMMITTED -> ACTIVE              : Error: cannot modify committed txn
  ABORTED -> ACTIVE                : Error: cannot modify aborted txn
  COMMITTED -> COMMITTED           : Error: already committed
  ABORTED -> ABORTED               : Error: already aborted
```
---
[[CRITERIA_JSON: {"module_id": "wal-impl-m1", "criteria": ["Log record header is exactly 28 bytes with fields: lsn (uint64_t LE, offset 0x00), type (uint32_t LE, offset 0x08), txn_id (uint32_t LE, offset 0x0C), prev_lsn (uint64_t LE, offset 0x10), length (uint32_t LE, offset 0x18)", "Six record types defined with correct type codes: BEGIN=1, UPDATE=2, COMMIT=3, ABORT=4, CLR=5, CHECKPOINT=6", "BEGIN record total size is 32 bytes (28 header + 4 CRC), no variable payload", "UPDATE record serialized format: [header:28][page_id:8][offset:4][old_len:4][old_val:N][new_len:4][new_val:M][crc:4], minimum 52 bytes", "UPDATE record old_value and new_value are length-prefixed, allowing 0-length values", "CLR record contains undo_next_lsn field (uint64_t) distinct from header.prev_lsn; undo_next_lsn points to next record to undo, prev_lsn links CLR into transaction chain", "CLR serialized format: [header:28][page_id:8][offset:4][undo_len:4][undo_data:N][undo_next_lsn:8][crc:4], minimum 56 bytes", "CHECKPOINT record contains num_active_txns (uint32_t), num_dirty_pages (uint32_t), followed by serialized TxnTableEntry array (24 bytes each) and DirtyPageEntry array (16 bytes each)", "CRC32 computed using zlib crc32() function over all bytes preceding CRC field, stored as uint32_t LE in final 4 bytes of record", "TransactionTable tracks txn_id, status (ACTIVE/COMMITTED/ABORTED), first_lsn, last_lsn for each active transaction", "txn_begin() generates BEGIN record with prev_lsn=0, allocates LSN, adds entry to transaction table with status=ACTIVE", "txn_write() generates UPDATE record with prev_lsn set to transaction's current last_lsn, then updates last_lsn to new record's LSN", "txn_commit() generates COMMIT record, changes transaction status to COMMITTED", "prev_lsn chain links records within same transaction only; concurrent transactions have independent chains", "Deserialization allocates memory for variable-length fields (old_value, new_value, undo_data); destroy functions free allocated memory", "Round-trip tests verify byte-perfect serialization/deserialization for all record types including variable-length fields", "Error codes defined for: NULL_POINTER, SERIALIZATION, DESERIALIZATION, CRC_MISMATCH, INVALID_TXN, INVALID_STATE, INVALID_TYPE, ALLOC, BUFFER_OVERFLOW", "Performance targets: fixed record serialize <100ns, variable record serialize <500ns, deserialize with CRC <1µs for 64-byte payloads"]}]
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: wal-impl-m2 -->
<!-- TDD_MOD_ID: wal-impl-m2 -->
# Module: Log Writer with Group Commit
## Module ID: wal-impl-m2
---
## 1. Module Charter
This module implements durable, append-only log writing with the LSN-as-offset design pattern, where each LSN directly maps to a byte position in the log file. It provides concurrent writer serialization via mutex, ensuring multiple threads can safely append records without interleaving. The group commit optimization batches multiple transaction commits into a single fsync call, achieving ≥5x throughput improvement over per-commit fsync. A write buffer reduces syscall overhead by batching small records before writing. Log segment rotation creates new segment files when the current segment exceeds a configurable threshold (default 64MB), enabling bounded file sizes and future truncation. Torn write detection via CRC32 mismatch identifies and truncates partial records from crashes. This module does NOT implement crash recovery (M3), checkpointing (M4), or transaction semantics (M1). Invariants: (1) LSN equals current file offset at time of write, (2) COMMIT records are never acknowledged until fsync completes, (3) no two records share the same LSN, (4) segment rotation is atomic—never writing to two segments simultaneously, (5) torn writes are always detected and truncated before any recovery logic runs.
---
## 2. File Structure
```
wal/
├── include/
│   ├── wal_writer.h         # [1] WalWriter, WriteBuffer, GroupCommitManager APIs
│   └── wal_segment.h        # [2] LogSegmentManager for rotation
└── src/
    ├── wal_writer.c         # [3] Append, sync, group commit implementation
    └── wal_segment.c        # [4] Segment creation, rotation, metadata
test/
├── test_append_correctness.c    # [5] Sequential append, LSN allocation
├── test_group_commit.c          # [6] Throughput benchmark, leader/follower
├── test_concurrent_writers.c    # [7] Multi-threaded append verification
├── test_segment_rotation.c      # [8] Rotation at threshold, segment naming
└── test_torn_write_detection.c  # [9] CRC truncation on corrupt final record
```
---
## 3. Complete Data Model
### 3.1 Core Constants
```c
// wal_writer.h
#ifndef WAL_WRITER_H
#define WAL_WRITER_H
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <pthread.h>
#include "wal_types.h"
// ============== CONSTANTS ==============
#define DEFAULT_SEGMENT_SIZE      (64 * 1024 * 1024)   // 64 MB
#define DEFAULT_WRITE_BUFFER_SIZE (1024 * 1024)        // 1 MB
#define MAX_WRITE_BUFFER_SIZE     (16 * 1024 * 1024)   // 16 MB
#define GROUP_COMMIT_TIMEOUT_US   1000                 // 1ms batch window
#define GROUP_COMMIT_MAX_WAITERS  1024                 // Max concurrent waiters
// Group commit states
typedef enum {
    GC_INACTIVE = 0,      // No sync in progress
    GC_LEADER_ACTIVE = 1, // Leader is performing fsync
    GC_SHUTDOWN = 2       // Writer is shutting down
} GroupCommitState;
// Sync result for waiters
typedef enum {
    GC_SYNC_SUCCESS = 0,
    GC_SYNC_ERROR = 1,
    GC_SYNC_SHUTDOWN = 2
} GroupCommitResult;
#endif // WAL_WRITER_H
```
### 3.2 Write Buffer Structure
Buffers records in memory before issuing write() syscalls. Reduces context switch overhead for high-throughput scenarios.
```

![fsync Latency Breakdown](./diagrams/tdd-diag-m2-01.svg)

```
```c
typedef struct {
    uint8_t*   buffer;           // Allocated buffer memory
    size_t     capacity;         // Total buffer capacity in bytes
    size_t     used;             // Bytes currently in buffer
    Lsn        first_buffered_lsn;  // LSN of first record in buffer
    Lsn        last_buffered_lsn;   // LSN of last record in buffer
    bool       needs_flush;      // True if COMMIT record is buffered
} WriteBuffer;
// Invariants:
// - used <= capacity
// - first_buffered_lsn <= last_buffered_lsn when used > 0
// - needs_flush implies a COMMIT record is in the buffer
```
### 3.3 Log Segment Structure
Manages individual log segment files and their metadata.
```c
typedef struct {
    uint32_t   segment_id;       // Segment number (1, 2, 3, ...)
    int        fd;               // File descriptor (-1 if closed)
    uint64_t   start_lsn;        // LSN of first record in segment
    uint64_t   current_offset;   // Current write offset (= next LSN to assign)
    uint64_t   size_limit;       // Max bytes before rotation
} LogSegment;
// File naming: {base_path}.{segment_id:06d}
// Example: /data/wal.000001, /data/wal.000002
```
### 3.4 Log Segment Manager
Orchestrates segment lifecycle and rotation.
```

![Group Commit Leader/Follower Protocol](./diagrams/tdd-diag-m2-02.svg)

```
```c
typedef struct {
    char       base_path[256];   // Base path for segment files
    LogSegment current;          // Active segment
    uint32_t   segment_count;    // Total segments created
    uint64_t   segment_size_limit;  // Configurable rotation threshold
} LogSegmentManager;
// Invariants:
// - current.fd >= 0 during normal operation
// - segment_count equals the highest segment_id ever created
// - Rotation only occurs at record boundaries (never mid-record)
```
### 3.5 Group Commit Manager
Implements the leader/follower protocol for batching fsync operations.
```

![Group Commit Timeline](./diagrams/tdd-diag-m2-03.svg)

```
```c
typedef struct {
    pthread_mutex_t lock;            // Protects all fields
    pthread_cond_t  sync_complete;   // Signaled when fsync finishes
    pthread_cond_t  leader_ready;    // Signaled when leader starts
    GroupCommitState state;          // Current state
    int              log_fd;         // FD to fsync (updated on rotation)
    // Sync tracking
    uint64_t         pending_sync_lsn;   // Max LSN needing sync
    uint64_t         last_synced_lsn;    // Last LSN known durable
    // Waiter management
    uint32_t         waiter_count;       // Number of waiting followers
    GroupCommitResult last_result;       // Result of last fsync
    int              last_error;         // errno if last_result == GC_SYNC_ERROR
    // Statistics
    uint64_t         total_syncs;        // Total fsync calls made
    uint64_t         total_waiters;      // Total transactions batched
} GroupCommitManager;
// Invariants:
// - state == GC_LEADER_ACTIVE implies exactly one thread is doing fsync
// - waiter_count == 0 implies no followers waiting
// - last_synced_lsn <= pending_sync_lsn always
```
### 3.6 WAL Writer (Main Structure)
Top-level structure coordinating all components.
```c
typedef struct {
    // Segment management
    LogSegmentManager segments;
    // Write buffering
    WriteBuffer       write_buffer;
    // Group commit
    GroupCommitManager group_commit;
    // Concurrency
    pthread_mutex_t   append_lock;     // Serializes record appends
    // LSN allocation (protected by append_lock)
    uint64_t          next_lsn;        // Next LSN to assign (= current file offset)
    // Configuration
    uint64_t          fsync_interval;  // Bytes between automatic fsync (0 = disabled)
    bool              sync_on_write;   // If true, fsync after every write (for testing)
    // State
    bool              is_shutdown;     // True after wal_writer_destroy() called
} WalWriter;
```
### 3.7 Memory Layout: WriteBuffer During Operation
```

![Concurrent Writer Serialization](./diagrams/tdd-diag-m2-04.svg)

```
**Buffer State Example (1MB buffer, 3 records buffered):**
```
Offset    Content                      Size
-------   -------------------------    --------
0x0000    [Record 1: 64 bytes]         64
0x0040    [Record 2: 128 bytes]        128
0x00C0    [Record 3: 32 bytes]         32
0x00E0    [Unused]                     1,048,480
-------   -------------------------    --------
          Total Used: 224 bytes
          Capacity: 1,048,576 bytes (1MB)
```
### 3.8 File Layout: Segment Naming Convention
```
{{DIAGRAM:tdd-diag-m2-05}}
```
**On-Disk Structure:**
```
/data/wal/
├── wal.000001    [LSN 1 - 67,108,863]       ← First segment (64MB)
├── wal.000002    [LSN 67,108,864 - ...]     ← Second segment
└── wal.000003    [Active, growing]          ← Current active segment
```
**LSN to Segment Mapping:**
```c
// Given LSN, find segment file
// segment_id = (lsn / segment_size_limit) + 1
// file_offset = lsn % segment_size_limit
uint32_t lsn_to_segment_id(uint64_t lsn, uint64_t segment_size) {
    return (uint32_t)(lsn / segment_size) + 1;
}
uint64_t lsn_to_file_offset(uint64_t lsn, uint64_t segment_size) {
    return lsn % segment_size;
}
```
### 3.9 Group Commit State Machine
```

![Log Segment Rotation](./diagrams/tdd-diag-m2-06.svg)

```
**States and Transitions:**
```
                    +-----------------+
                    |  GC_INACTIVE    |  <-- Initial state, no sync pending
                    +--------+--------+
                             |
            First committer becomes leader
                             |
                             v
                    +-----------------+
                    | GC_LEADER_ACTIVE|  <-- Leader doing fsync
                    +--------+--------+
                             |
               fsync completes OR error
                             |
                             v
                    +-----------------+
                    |  GC_INACTIVE    |  <-- Back to inactive
                    +-----------------+
                    +-----------------+
                    |  GC_SHUTDOWN    |  <-- Shutdown requested
                    +-----------------+
```
---
## 4. Interface Contracts
### 4.1 Write Buffer Functions
```c
// wal_writer.h
/**
 * Initialize write buffer with given capacity.
 * 
 * @param buf      Buffer to initialize
 * @param capacity Buffer size in bytes
 * @return         WAL_OK or WAL_ERR_ALLOC
 */
int write_buffer_init(WriteBuffer* buf, size_t capacity);
/**
 * Destroy write buffer, freeing memory.
 */
void write_buffer_destroy(WriteBuffer* buf);
/**
 * Add record to buffer.
 * 
 * @param buf      Buffer to modify
 * @param data     Record data to copy
 * @param len      Record length
 * @param lsn      LSN of this record
 * @param is_commit True if this is a COMMIT record (triggers flush requirement)
 * @return         WAL_OK, WAL_ERR_NULL_POINTER, or WAL_ERR_BUFFER_OVERFLOW
 * 
 * Precondition: buf->used + len <= buf->capacity (caller must flush if needed)
 * Postcondition: Record data is copied into buffer, used updated
 */
int write_buffer_append(WriteBuffer* buf, const uint8_t* data, size_t len, 
                        Lsn lsn, bool is_commit);
/**
 * Check if buffer needs flushing.
 * 
 * @return True if buffer contains COMMIT record or is full
 */
bool write_buffer_needs_flush(const WriteBuffer* buf);
/**
 * Clear buffer after successful flush.
 * Does NOT free memory, just resets used count.
 */
void write_buffer_clear(WriteBuffer* buf);
```
### 4.2 Log Segment Functions
```c
// wal_segment.h
/**
 * Initialize segment manager.
 * 
 * @param mgr          Manager to initialize
 * @param base_path    Base path for segment files (e.g., "/data/wal")
 * @param size_limit   Segment rotation threshold in bytes
 * @return             WAL_OK, WAL_ERR_ALLOC, or error from file creation
 */
int segment_manager_init(LogSegmentManager* mgr, const char* base_path, 
                         uint64_t size_limit);
/**
 * Destroy segment manager, closing current segment.
 */
void segment_manager_destroy(LogSegmentManager* mgr);
/**
 * Get current write offset (= next LSN to assign).
 */
uint64_t segment_manager_get_next_lsn(const LogSegmentManager* mgr);
/**
 * Write data to current segment.
 * 
 * @param mgr    Segment manager
 * @param data   Data to write
 * @param len    Length of data
 * @return       WAL_OK or error code
 * 
 * Side effect: May trigger rotation if write would exceed size_limit
 */
int segment_manager_write(LogSegmentManager* mgr, const uint8_t* data, size_t len);
/**
 * Check if rotation is needed after a write.
 * 
 * @return True if current_offset >= size_limit
 */
bool segment_manager_needs_rotation(const LogSegmentManager* mgr);
/**
 * Rotate to a new segment.
 * 
 * @param mgr    Segment manager
 * @return       WAL_OK or error code
 * 
 * Preconditions:
 *   - Current segment is synced (caller's responsibility)
 * Postconditions:
 *   - New segment file created and opened
 *   - current.fd points to new segment
 *   - current.segment_id incremented
 */
int segment_manager_rotate(LogSegmentManager* mgr);
/**
 * Get file descriptor for current segment.
 */
int segment_manager_get_fd(const LogSegmentManager* mgr);
/**
 * Sync current segment to disk.
 */
int segment_manager_sync(LogSegmentManager* mgr);
/**
 * Build segment filename.
 * 
 * @param mgr         Segment manager
 * @param segment_id  Segment number
 * @param[out] path   Output buffer for full path
 * @param path_len    Buffer capacity
 * @return            WAL_OK or WAL_ERR_BUFFER_OVERFLOW
 */
int segment_manager_build_path(const LogSegmentManager* mgr, uint32_t segment_id,
                               char* path, size_t path_len);
```
### 4.3 Group Commit Functions
```c
// wal_writer.h
/**
 * Initialize group commit manager.
 * 
 * @param gcm     Manager to initialize
 * @param log_fd  Initial file descriptor to sync
 * @return        WAL_OK or error from mutex/condvar init
 */
int group_commit_init(GroupCommitManager* gcm, int log_fd);
/**
 * Destroy group commit manager.
 * Wakes all waiters with GC_SYNC_SHUTDOWN.
 */
void group_commit_destroy(GroupCommitManager* gcm);
/**
 * Update the file descriptor to sync (call after segment rotation).
 * 
 * @param gcm     Group commit manager
 * @param new_fd  New file descriptor
 * 
 * Thread-safe: Can be called while waiters are present.
 */
void group_commit_update_fd(GroupCommitManager* gcm, int new_fd);
/**
 * Request sync for a commit LSN.
 * 
 * @param gcm        Group commit manager
 * @param commit_lsn LSN of the COMMIT record to sync
 * @param timeout_us Maximum time to wait for sync (0 = wait forever)
 * @return           GC_SYNC_SUCCESS, GC_SYNC_ERROR, or GC_SYNC_SHUTDOWN
 * 
 * Protocol:
 *   - First caller (when state == GC_INACTIVE) becomes LEADER
 *   - Leader performs fsync, then broadcasts to waiters
 *   - Subsequent callers become FOLLOWERS and wait
 *   - Followers return when their LSN <= last_synced_lsn
 * 
 * Leader behavior:
 *   1. Set state = GC_LEADER_ACTIVE
 *   2. Release lock
 *   3. Call fsync(log_fd)
 *   4. Reacquire lock
 *   5. Update last_synced_lsn = pending_sync_lsn
 *   6. Set state = GC_INACTIVE
 *   7. Broadcast sync_complete
 * 
 * Follower behavior:
 *   1. Increment waiter_count
 *   2. Update pending_sync_lsn = max(pending_sync_lsn, commit_lsn)
 *   3. Wait on sync_complete while last_synced_lsn < commit_lsn
 *   4. Decrement waiter_count
 *   5. Return result
 */
GroupCommitResult group_commit_sync(GroupCommitManager* gcm, Lsn commit_lsn,
                                    uint64_t timeout_us);
/**
 * Get statistics.
 */
uint64_t group_commit_get_total_syncs(const GroupCommitManager* gcm);
uint64_t group_commit_get_total_waiters(const GroupCommitManager* gcm);
double group_commit_get_avg_batch_size(const GroupCommitManager* gcm);
```
### 4.4 WAL Writer Functions
```c
// wal_writer.h
/**
 * Create WAL writer.
 * 
 * @param base_path       Base path for log segments (e.g., "/data/wal")
 * @param segment_size    Segment rotation threshold (0 = use default 64MB)
 * @param buffer_size     Write buffer size (0 = use default 1MB)
 * @return                Allocated writer or NULL on error
 */
WalWriter* wal_writer_create(const char* base_path, 
                             uint64_t segment_size,
                             size_t buffer_size);
/**
 * Destroy WAL writer.
 * 
 * - Flushes write buffer
 * - Syncs current segment
 * - Wakes any group commit waiters with shutdown status
 * - Closes all files
 * - Frees memory
 */
void wal_writer_destroy(WalWriter* writer);
/**
 * Append a log record.
 * 
 * @param writer   WAL writer
 * @param data     Serialized record data
 * @param len      Record length
 * @param[out] lsn Assigned LSN on success
 * @return         WAL_OK or error code
 * 
 * This function:
 *   1. Acquires append_lock
 *   2. Allocates LSN (= current file offset)
 *   3. Adds to write buffer (flushing if needed)
 *   4. Releases append_lock
 *   5. Returns assigned LSN
 * 
 * The record is NOT guaranteed durable until wal_commit_sync() is called.
 */
int wal_append(WalWriter* writer, const uint8_t* data, size_t len, Lsn* lsn);
/**
 * Commit sync: ensure a commit LSN is durable.
 * 
 * @param writer     WAL writer
 * @param commit_lsn LSN of COMMIT record to sync
 * @param timeout_us Max wait time (0 = wait forever)
 * @return           GC_SYNC_SUCCESS, GC_SYNC_ERROR, or GC_SYNC_SHUTDOWN
 * 
 * This function:
 *   1. Flushes write buffer if commit_lsn is in buffer
 *   2. Calls group_commit_sync() to wait for fsync
 *   3. Returns when commit is durable
 * 
 * CRITICAL: This MUST be called before acknowledging commit to client.
 */
GroupCommitResult wal_commit_sync(WalWriter* writer, Lsn commit_lsn,
                                  uint64_t timeout_us);
/**
 * Flush write buffer to disk.
 * 
 * @param writer   WAL writer
 * @return         WAL_OK or error from write()
 * 
 * Called automatically when:
 *   - Buffer is full
 *   - COMMIT record is appended
 *   - Explicit flush requested
 */
int wal_flush_buffer(WalWriter* writer);
/**
 * Force sync of current segment.
 * 
 * @param writer   WAL writer
 * @return         WAL_OK or error from fsync()
 * 
 * Use sparingly - bypasses group commit optimization.
 */
int wal_force_sync(WalWriter* writer);
/**
 * Get current LSN (next to be assigned).
 */
Lsn wal_writer_get_next_lsn(const WalWriter* writer);
/**
 * Get statistics.
 */
typedef struct {
    uint64_t total_records;
    uint64_t total_bytes;
    uint64_t total_syncs;
    uint64_t total_commits_batched;
    uint64_t buffer_flushes;
    uint64_t segment_rotations;
} WalWriterStats;
void wal_writer_get_stats(const WalWriter* writer, WalWriterStats* stats);
```
### 4.5 Torn Write Detection
```c
// wal_writer.h
/**
 * Detect and truncate torn writes from a crash.
 * 
 * @param base_path    Base path for log segments
 * @param[out] last_valid_lsn  LSN of last valid record (0 if log empty)
 * @return             WAL_OK or error code
 * 
 * Algorithm:
 *   1. Open the most recent segment file
 *   2. Scan from beginning, validating each record's CRC
 *   3. On CRC mismatch or truncated record, truncate file at that point
 *   4. Return LSN of last valid record
 * 
 * Must be called BEFORE any recovery logic (M3).
 */
int wal_detect_truncate_torn_writes(const char* base_path, Lsn* last_valid_lsn);
/**
 * Internal: Validate a single record during scan.
 * 
 * @param fd      File descriptor positioned at record start
 * @param[out] lsn LSN of the validated record
 * @return        >0 = record size (valid), 0 = EOF, <0 = error/invalid
 */
ssize_t wal_validate_record(int fd, Lsn* lsn);
```
---
## 5. Algorithm Specifications
### 5.1 LSN-as-Offset Append Algorithm
```

![Write Buffer Structure](./diagrams/tdd-diag-m2-07.svg)

```
**Procedure wal_append(writer, data, len, lsn):**
```
1. VALIDATE INPUTS
   IF writer == NULL OR data == NULL OR lsn == NULL:
       RETURN WAL_ERR_NULL_POINTER
   IF len == 0 OR len > MAX_RECORD_SIZE:
       RETURN WAL_ERR_SERIALIZATION
   IF writer->is_shutdown:
       RETURN WAL_ERR_INVALID_STATE
2. ACQUIRE APPEND LOCK
   pthread_mutex_lock(&writer->append_lock)
3. ALLOCATE LSN
   *lsn = writer->next_lsn
   // LSN equals current file offset - no separate counter needed
4. CHECK BUFFER CAPACITY
   IF writer->write_buffer.used + len > writer->write_buffer.capacity:
       // Buffer full, must flush first
       err = flush_buffer_internal(writer)
       IF err != WAL_OK:
           pthread_mutex_unlock(&writer->append_lock)
           RETURN err
5. ADD TO WRITE BUFFER
   err = write_buffer_append(&writer->write_buffer, data, len, *lsn, false)
   IF err != WAL_OK:
       pthread_mutex_unlock(&writer->append_lock)
       RETURN err
6. UPDATE NEXT LSN
   writer->next_lsn += len
7. CHECK SEGMENT ROTATION
   IF segment_manager_needs_rotation(&writer->segments):
       // Flush before rotating
       flush_buffer_internal(writer)
       // Note: rotation happens after current write is complete
8. RELEASE LOCK AND RETURN
   pthread_mutex_unlock(&writer->append_lock)
   RETURN WAL_OK
```
**Invariant Check:**
- After step 6: `writer->next_lsn == segment_manager_get_next_lsn(&writer->segments) + write_buffer.used`
- LSN uniqueness guaranteed by monotonic increment
### 5.2 Write Buffer Flush Algorithm
**Procedure flush_buffer_internal(writer):**
```
1. CHECK IF BUFFER EMPTY
   IF writer->write_buffer.used == 0:
       RETURN WAL_OK  // Nothing to flush
2. WRITE TO SEGMENT
   err = segment_manager_write(&writer->segments,
                                writer->write_buffer.buffer,
                                writer->write_buffer.used)
   IF err != WAL_OK:
       RETURN err
3. CHECK FOR ROTATION
   IF segment_manager_needs_rotation(&writer->segments):
       // Sync current segment before rotation
       segment_manager_sync(&writer->segments)
       // Rotate to new segment
       err = segment_manager_rotate(&writer->segments)
       IF err != WAL_OK:
           RETURN err
       // Update group commit's FD reference
       group_commit_update_fd(&writer->group_commit,
                              segment_manager_get_fd(&writer->segments))
4. CLEAR BUFFER
   write_buffer_clear(&writer->write_buffer)
5. UPDATE STATISTICS
   writer->stats.buffer_flushes++
6. RETURN SUCCESS
   RETURN WAL_OK
```
### 5.3 Group Commit Leader/Follower Protocol
```

![Torn Write Detection via CRC](./diagrams/tdd-diag-m2-08.svg)

```
**Procedure group_commit_sync(gcm, commit_lsn, timeout_us):**
```
1. ACQUIRE LOCK
   pthread_mutex_lock(&gcm->lock)
2. CHECK FOR SHUTDOWN
   IF gcm->state == GC_SHUTDOWN:
       pthread_mutex_unlock(&gcm->lock)
       RETURN GC_SYNC_SHUTDOWN
3. UPDATE PENDING SYNC LSN
   IF commit_lsn > gcm->pending_sync_lsn:
       gcm->pending_sync_lsn = commit_lsn
4. DETERMINE ROLE
   IF gcm->state == GC_INACTIVE:
       // === LEADER PATH ===
       GOTO leader_path
   ELSE:
       // === FOLLOWER PATH ===
       GOTO follower_path
// ================================
// LEADER PATH
// ================================
leader_path:
   gcm->state = GC_LEADER_ACTIVE
   // Signal any waiters that leader is active (optional optimization)
   pthread_cond_broadcast(&gcm->leader_ready)
   // Get sync target before releasing lock
   sync_target = gcm->pending_sync_lsn
   fd_to_sync = gcm->log_fd
   // Release lock during expensive fsync
   pthread_mutex_unlock(&gcm->lock)
   // Perform fsync (this is the slow part - milliseconds)
   sync_result = fsync(fd_to_sync)
   // Reacquire lock to update state
   pthread_mutex_lock(&gcm->lock)
   IF sync_result < 0:
       // fsync failed
       gcm->last_result = GC_SYNC_ERROR
       gcm->last_error = errno
   ELSE:
       gcm->last_result = GC_SYNC_SUCCESS
       gcm->last_synced_lsn = sync_target
       gcm->total_syncs++
   // Wake all waiters
   gcm->state = GC_INACTIVE
   pthread_cond_broadcast(&gcm->sync_complete)
   // Leader's own commit is now durable
   result = gcm->last_result
   pthread_mutex_unlock(&gcm->lock)
   RETURN result
// ================================
// FOLLOWER PATH
// ================================
follower_path:
   gcm->waiter_count++
   gcm->total_waiters++
   // Wait for sync to complete covering our LSN
   deadline = compute_deadline(timeout_us)
   WHILE gcm->last_synced_lsn < commit_lsn AND gcm->state != GC_SHUTDOWN:
       IF timeout_us > 0:
           rc = pthread_cond_timedwait(&gcm->sync_complete, &gcm->lock, &deadline)
           IF rc == ETIMEDOUT:
               // Timeout - become leader ourselves if no leader active
               IF gcm->state == GC_INACTIVE:
                   gcm->waiter_count--
                   GOTO leader_path
           ELSE IF rc != 0:
               // Error
               gcm->waiter_count--
               pthread_mutex_unlock(&gcm->lock)
               RETURN GC_SYNC_ERROR
       ELSE:
           pthread_cond_wait(&gcm->sync_complete, &gcm->lock)
   // Check result
   IF gcm->state == GC_SHUTDOWN:
       result = GC_SYNC_SHUTDOWN
   ELSE IF gcm->last_synced_lsn >= commit_lsn:
       result = GC_SYNC_SUCCESS
   ELSE:
       result = GC_SYNC_ERROR
   gcm->waiter_count--
   pthread_mutex_unlock(&gcm->lock)
   RETURN result
```
### 5.4 Segment Rotation Algorithm
```

![WalWriter Component Architecture](./diagrams/tdd-diag-m2-09.svg)

```
**Procedure segment_manager_rotate(mgr):**
```
1. SYNC CURRENT SEGMENT
   // Ensure all data is durable before closing
   fsync(mgr->current.fd)
2. CLOSE CURRENT SEGMENT
   close(mgr->current.fd)
   mgr->current.fd = -1
3. INCREMENT SEGMENT ID
   mgr->segment_count++
   new_segment_id = mgr->segment_count
4. BUILD NEW SEGMENT PATH
   err = segment_manager_build_path(mgr, new_segment_id, path, sizeof(path))
   IF err != WAL_OK:
       // Critical error - cannot continue
       // Attempt to reopen old segment
       mgr->segment_count--  // Rollback
       reopen_previous_segment(mgr)
       RETURN WAL_ERR_FILE_CREATE
5. CREATE NEW SEGMENT FILE
   new_fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644)
   IF new_fd < 0:
       mgr->segment_count--  // Rollback
       reopen_previous_segment(mgr)
       RETURN WAL_ERR_FILE_CREATE
6. UPDATE CURRENT SEGMENT STATE
   mgr->current.fd = new_fd
   mgr->current.segment_id = new_segment_id
   mgr->current.start_lsn = mgr->current.current_offset
   mgr->current.current_offset = 0  // Reset for new segment
7. RETURN SUCCESS
   RETURN WAL_OK
```
**Atomicity Guarantee:**
- After step 2, old segment is closed but complete
- After step 5, new segment exists and is ready
- If crash between steps 2 and 5, recovery (M3) can detect incomplete rotation
- No data is ever written to two segments simultaneously
### 5.5 Torn Write Detection Algorithm
```
{{DIAGRAM:tdd-diag-m2-10}}
```
**Procedure wal_detect_truncate_torn_writes(base_path, last_valid_lsn):**
```
1. FIND MOST RECENT SEGMENT
   segment_id = find_latest_segment_id(base_path)
   IF segment_id == 0:
       *last_valid_lsn = 0
       RETURN WAL_OK  // No segments exist
2. OPEN SEGMENT FILE
   build_segment_path(base_path, segment_id, path)
   fd = open(path, O_RDWR)
   IF fd < 0:
       RETURN WAL_ERR_FILE_OPEN
3. GET FILE SIZE
   fstat(fd, &st)
   file_size = st.st_size
   current_offset = 0
4. SCAN ALL RECORDS
   WHILE current_offset < file_size:
       // Try to read header
       bytes_read = pread(fd, header_buf, HEADER_SIZE, current_offset)
       IF bytes_read < HEADER_SIZE:
           // Incomplete header - torn write
           GOTO truncate_at_current_offset
       // Parse header
       parse_header(header_buf, &header)
       // Validate header fields
       IF header.length < MIN_RECORD_SIZE OR header.length > MAX_RECORD_SIZE:
           // Corrupted length - torn write or garbage
           GOTO truncate_at_current_offset
       IF header.lsn == 0 OR header.type == RECORD_INVALID:
           // Invalid header fields
           GOTO truncate_at_current_offset
       // Try to read full record
       IF current_offset + header.length > file_size:
           // Record extends past EOF - torn write
           GOTO truncate_at_current_offset
       // Read full record for CRC verification
       record_buf = allocate(header.length)
       bytes_read = pread(fd, record_buf, header.length, current_offset)
       IF bytes_read < header.length:
           free(record_buf)
           GOTO truncate_at_current_offset
       // Verify CRC
       stored_crc = read_le32(record_buf + header.length - 4)
       computed_crc = wal_compute_crc(record_buf, header.length - 4)
       free(record_buf)
       IF stored_crc != computed_crc:
           // CRC mismatch - torn write or corruption
           GOTO truncate_at_current_offset
       // Record is valid - update last valid LSN and continue
       *last_valid_lsn = header.lsn
       current_offset += header.length
   // All records valid
   close(fd)
   RETURN WAL_OK
truncate_at_current_offset:
   // Truncate file at start of invalid record
   IF current_offset < file_size:
       ftruncate(fd, current_offset)
   close(fd)
   RETURN WAL_OK
```
---
## 6. Error Handling Matrix
| Error                      | Detected By                              | Recovery Action                              | User-Visible? |
|----------------------------|------------------------------------------|----------------------------------------------|---------------|
| WAL_ERR_NULL_POINTER       | All public functions at entry            | Return immediately, no state change          | Yes (API)     |
| WAL_ERR_ALLOC              | malloc in buffer/segment init            | Return error, no partial allocation          | Yes (API)     |
| WAL_ERR_FILE_CREATE        | open() on new segment fails              | Attempt to reopen previous segment, return   | Yes (API)     |
| WAL_ERR_FILE_OPEN          | open() on existing segment fails         | Return error, no state change                | Yes (API)     |
| WAL_ERR_FILE_WRITE         | write() syscall fails                    | Return error, buffer state unchanged         | Yes (API)     |
| WAL_ERR_FILE_SYNC          | fsync() syscall fails                    | Return GC_SYNC_ERROR to all waiters          | Yes (API)     |
| WAL_ERR_BUFFER_OVERFLOW    | Record exceeds buffer capacity           | Flush buffer first, then retry               | No (internal) |
| WAL_ERR_SERIALIZATION      | Invalid record size (0 or > MAX)         | Return error, no append performed            | Yes (API)     |
| WAL_ERR_INVALID_STATE      | Operation on shutdown writer             | Return error, no state change                | Yes (API)     |
| GC_SYNC_ERROR              | fsync() returns error                    | All waiters receive error, log errno         | Yes (API)     |
| GC_SYNC_SHUTDOWN           | Writer destroyed while waiters present   | All waiters receive shutdown status          | Yes (API)     |
| GC_SYNC_TIMEOUT            | Timed wait expires without leader        | Caller may retry or become leader            | Yes (API)     |
**Lock Invariant:** No error path may leave any mutex locked. Every lock acquisition has a corresponding release on all paths (success, error, early return).
**State Consistency:** After any error:
- Write buffer remains in valid state (may contain partial data)
- Segment manager points to valid, open segment
- Group commit manager state is GC_INACTIVE or GC_SHUTDOWN
- next_lsn never decreases
---
## 7. Implementation Sequence with Checkpoints
### Phase 1: Write Buffer (1 hour)
**Files:** `include/wal_writer.h` (partial), `src/wal_writer.c` (partial)
**Tasks:**
1. Define `WriteBuffer` struct
2. Implement `write_buffer_init()`
3. Implement `write_buffer_destroy()`
4. Implement `write_buffer_append()`
5. Implement `write_buffer_needs_flush()`
6. Implement `write_buffer_clear()`
**Checkpoint 1:**
```bash
gcc -c src/wal_writer.c -I include
./test_write_buffer
# Tests should pass:
# - Init/destroy cycle
# - Append single record
# - Append multiple records
# - Buffer full detection
# - COMMIT flag sets needs_flush
```
### Phase 2: Log Segment Manager (1-2 hours)
**Files:** `include/wal_segment.h`, `src/wal_segment.c`
**Tasks:**
1. Define `LogSegment` and `LogSegmentManager` structs
2. Implement `segment_manager_init()`
3. Implement `segment_manager_destroy()`
4. Implement `segment_manager_get_next_lsn()`
5. Implement `segment_manager_write()`
6. Implement `segment_manager_sync()`
7. Implement `segment_manager_build_path()`
8. Implement `segment_manager_needs_rotation()`
9. Implement `segment_manager_rotate()`
**Checkpoint 2:**
```bash
./test_segment_manager
# Tests should pass:
# - Create first segment
# - Write records, verify LSN allocation
# - Rotation at size threshold
# - Segment file naming convention
# - Sync to disk
```
### Phase 3: WAL Writer Append (1 hour)
**Files:** Continue `src/wal_writer.c`
**Tasks:**
1. Define `WalWriter` struct
2. Implement `wal_writer_create()`
3. Implement `wal_writer_destroy()`
4. Implement `wal_append()` (without group commit)
5. Implement `wal_flush_buffer()`
6. Implement `wal_writer_get_next_lsn()`
**Checkpoint 3:**
```bash
./test_append_correctness
# Tests should pass:
# - Single record append
# - Multiple sequential appends
# - LSN monotonic increase
# - Buffer flush on full
# - Segment rotation triggered
# - File contains correct data
```
### Phase 4: Group Commit Manager (2-3 hours)
**Files:** Continue `src/wal_writer.c`
**Tasks:**
1. Define `GroupCommitManager` struct
2. Implement `group_commit_init()`
3. Implement `group_commit_destroy()`
4. Implement `group_commit_update_fd()`
5. Implement `group_commit_sync()` with leader/follower protocol
6. Implement statistics functions
**Checkpoint 4:**
```bash
./test_group_commit_basic
# Tests should pass:
# - Single committer (becomes leader)
# - Two concurrent committers (leader + follower)
# - Multiple followers batched
# - Leader failure returns error to all
# - Shutdown wakes all waiters
```
### Phase 5: Integration - Full Commit Path (1 hour)
**Files:** Continue `src/wal_writer.c`
**Tasks:**
1. Implement `wal_commit_sync()` integrating buffer flush + group commit
2. Wire up segment rotation with group commit FD update
3. Add statistics tracking
**Checkpoint 5:**
```bash
./test_full_commit
# Tests should pass:
# - Commit flow: append -> commit_sync -> return
# - Commit record is durable on disk
# - Multiple concurrent commits batched
# - Rotation during commit handled correctly
```
### Phase 6: Concurrent Writer Testing (1 hour)
**Files:** `test/test_concurrent_writers.c`
**Tasks:**
1. Multi-threaded append test
2. Verify no interleaved records
3. Stress test with many threads
4. Verify LSN uniqueness under concurrency
**Checkpoint 6:**
```bash
./test_concurrent_writers
# Tests should pass:
# - 10 threads, 100 records each
# - All records intact in file
# - No LSN collisions
# - No deadlocks
```
### Phase 7: Torn Write Detection (1 hour)
**Files:** Add to `src/wal_writer.c`
**Tasks:**
1. Implement `wal_validate_record()`
2. Implement `wal_detect_truncate_torn_writes()`
3. Test with simulated torn writes
**Checkpoint 7:**
```bash
./test_torn_write_detection
# Tests should pass:
# - Valid log scans completely
# - Corrupted final record truncated
# - Partial header truncated
# - CRC mismatch truncated
# - Returns correct last_valid_lsn
```
### Phase 8: Benchmarks and Final Testing (1-2 hours)
**Files:** `test/test_group_commit.c` (benchmark), all tests
**Tasks:**
1. Implement group commit throughput benchmark
2. Compare per-commit fsync vs batched
3. Verify ≥5x improvement
4. Run all tests in sequence
5. Memory leak check with Valgrind
**Checkpoint 8 (Final):**
```bash
./run_all_tests
# All tests pass
./benchmark_group_commit
# Output should show:
# Without group commit: X commits/sec
# With group commit: Y commits/sec
# Speedup: Y/X (must be >= 5.0)
valgrind --leak-check=full ./test_all
# All memory freed, no leaks
```
---
## 8. Test Specification
### 8.1 Write Buffer Tests
```c
void test_write_buffer_init_destroy(void) {
    WriteBuffer buf;
    assert(write_buffer_init(&buf, 1024) == WAL_OK);
    assert(buf.buffer != NULL);
    assert(buf.capacity == 1024);
    assert(buf.used == 0);
    write_buffer_destroy(&buf);
    assert(buf.buffer == NULL);
}
void test_write_buffer_append(void) {
    WriteBuffer buf;
    write_buffer_init(&buf, 1024);
    uint8_t data[] = {1, 2, 3, 4, 5};
    assert(write_buffer_append(&buf, data, sizeof(data), 100, false) == WAL_OK);
    assert(buf.used == sizeof(data));
    assert(buf.first_buffered_lsn == 100);
    assert(buf.last_buffered_lsn == 100);
    assert(memcmp(buf.buffer, data, sizeof(data)) == 0);
    write_buffer_destroy(&buf);
}
void test_write_buffer_commit_flag(void) {
    WriteBuffer buf;
    write_buffer_init(&buf, 1024);
    uint8_t data[] = {1, 2, 3};
    write_buffer_append(&buf, data, sizeof(data), 100, false);
    assert(!write_buffer_needs_flush(&buf));
    write_buffer_append(&buf, data, sizeof(data), 200, true);  // COMMIT
    assert(write_buffer_needs_flush(&buf));
    write_buffer_destroy(&buf);
}
void test_write_buffer_overflow(void) {
    WriteBuffer buf;
    write_buffer_init(&buf, 10);
    uint8_t data[20];
    assert(write_buffer_append(&buf, data, sizeof(data), 100, false) 
           == WAL_ERR_BUFFER_OVERFLOW);
    write_buffer_destroy(&buf);
}
```
### 8.2 Segment Manager Tests
```c
void test_segment_creation(void) {
    LogSegmentManager mgr;
    assert(segment_manager_init(&mgr, "/tmp/test_wal", 1024) == WAL_OK);
    // Verify first segment created
    assert(mgr.segment_count == 1);
    assert(mgr.current.fd >= 0);
    assert(mgr.current.segment_id == 1);
    // Verify file exists
    struct stat st;
    assert(stat("/tmp/test_wal.000001", &st) == 0);
    segment_manager_destroy(&mgr);
    unlink("/tmp/test_wal.000001");
}
void test_segment_write_and_lsn(void) {
    LogSegmentManager mgr;
    segment_manager_init(&mgr, "/tmp/test_wal", 1024);
    // Initial LSN
    assert(segment_manager_get_next_lsn(&mgr) == 0);
    // Write record
    uint8_t data[] = {1, 2, 3, 4, 5};
    assert(segment_manager_write(&mgr, data, sizeof(data)) == WAL_OK);
    // LSN advanced
    assert(segment_manager_get_next_lsn(&mgr) == sizeof(data));
    // Verify file contents
    uint8_t read_buf[10];
    int fd = open("/tmp/test_wal.000001", O_RDONLY);
    assert(read(fd, read_buf, sizeof(data)) == sizeof(data));
    assert(memcmp(read_buf, data, sizeof(data)) == 0);
    close(fd);
    segment_manager_destroy(&mgr);
    unlink("/tmp/test_wal.000001");
}
void test_segment_rotation(void) {
    LogSegmentManager mgr;
    segment_manager_init(&mgr, "/tmp/test_wal", 10);  // Very small limit
    // Write to fill segment
    uint8_t data[] = {1, 2, 3, 4, 5};
    segment_manager_write(&mgr, data, sizeof(data));
    // Trigger rotation
    assert(segment_manager_needs_rotation(&mgr) == false);  // 5 < 10
    segment_manager_write(&mgr, data, sizeof(data));
    assert(segment_manager_needs_rotation(&mgr) == true);   // 10 >= 10
    assert(segment_manager_rotate(&mgr) == WAL_OK);
    assert(mgr.segment_count == 2);
    assert(mgr.current.segment_id == 2);
    // Verify new segment file exists
    struct stat st;
    assert(stat("/tmp/test_wal.000002", &st) == 0);
    // Cleanup
    segment_manager_destroy(&mgr);
    unlink("/tmp/test_wal.000001");
    unlink("/tmp/test_wal.000002");
}
```
### 8.3 Group Commit Tests
```c
void test_group_commit_single_leader(void) {
    GroupCommitManager gcm;
    int fd = open("/tmp/test_gc", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    write(fd, "test", 4);
    assert(group_commit_init(&gcm, fd) == WAL_OK);
    // Single committer becomes leader
    GroupCommitResult result = group_commit_sync(&gcm, 100, 0);
    assert(result == GC_SYNC_SUCCESS);
    assert(gcm.last_synced_lsn == 100);
    assert(gcm.total_syncs == 1);
    group_commit_destroy(&gcm);
    close(fd);
    unlink("/tmp/test_gc");
}
void test_group_commit_batching(void) {
    GroupCommitManager gcm;
    int fd = open("/tmp/test_gc", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    group_commit_init(&gcm, fd);
    // Spawn multiple committer threads
    #define NUM_THREADS 5
    pthread_t threads[NUM_THREADS];
    struct {
        GroupCommitManager* gcm;
        Lsn commit_lsn;
        GroupCommitResult result;
    } args[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].gcm = &gcm;
        args[i].commit_lsn = 100 + i * 10;
        pthread_create(&threads[i], NULL, committer_thread, &args[i]);
    }
    // Wait for all to complete
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
        assert(args[i].result == GC_SYNC_SUCCESS);
    }
    // Only one sync should have occurred
    assert(gcm.total_syncs == 1);
    assert(gcm.total_waiters == NUM_THREADS - 1);  // Leader not counted as waiter
    group_commit_destroy(&gcm);
    close(fd);
    unlink("/tmp/test_gc");
}
void* committer_thread(void* arg) {
    struct committer_arg* a = arg;
    a->result = group_commit_sync(a->gcm, a->commit_lsn, 0);
    return NULL;
}
```
### 8.4 Full WAL Writer Tests
```c
void test_wal_append_commit_flow(void) {
    WalWriter* writer = wal_writer_create("/tmp/test_wal", 0, 0);
    assert(writer != NULL);
    // Append a record
    uint8_t record[] = {0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 
                        0xDE, 0xAD, 0xBE, 0xEF};  // Minimal valid record
    Lsn lsn;
    assert(wal_append(writer, record, sizeof(record), &lsn) == WAL_OK);
    assert(lsn == 0);  // First LSN
    // Commit sync
    GroupCommitResult result = wal_commit_sync(writer, lsn + sizeof(record), 0);
    assert(result == GC_SYNC_SUCCESS);
    wal_writer_destroy(writer);
    unlink("/tmp/test_wal.000001");
}
void test_concurrent_appends(void) {
    WalWriter* writer = wal_writer_create("/tmp/test_wal", 0, 0);
    #define NUM_THREADS 10
    #define RECORDS_PER_THREAD 100
    pthread_t threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, append_thread, writer);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    // Verify all records present
    struct stat st;
    stat("/tmp/test_wal.000001", &st);
    assert(st.st_size == NUM_THREADS * RECORDS_PER_THREAD * 32);
    wal_writer_destroy(writer);
    unlink("/tmp/test_wal.000001");
}
```
### 8.5 Torn Write Detection Tests
```c
void test_torn_write_truncation(void) {
    // Create a valid log file
    int fd = open("/tmp/test_torn_wal.000001", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    // Write valid record
    uint8_t valid_record[32] = { /* valid record data */ };
    write(fd, valid_record, sizeof(valid_record));
    // Write partial (torn) record
    uint8_t partial_record[32] = { /* partial data */ };
    write(fd, partial_record, 16);  // Only half written
    close(fd);
    // Detect and truncate
    Lsn last_valid;
    assert(wal_detect_truncate_torn_writes("/tmp/test_torn_wal", &last_valid) == WAL_OK);
    // Verify truncation
    struct stat st;
    stat("/tmp/test_torn_wal.000001", &st);
    assert(st.st_size == 32);  // Only valid record remains
    unlink("/tmp/test_torn_wal.000001");
}
void test_valid_log_unchanged(void) {
    // Create valid log with multiple records
    int fd = open("/tmp/test_valid_wal.000001", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    for (int i = 0; i < 10; i++) {
        uint8_t record[32];
        // Populate with valid record...
        write(fd, record, sizeof(record));
    }
    fsync(fd);
    close(fd);
    // Scan should not truncate
    Lsn last_valid;
    wal_detect_truncate_torn_writes("/tmp/test_valid_wal", &last_valid);
    struct stat st;
    stat("/tmp/test_valid_wal.000001", &st);
    assert(st.st_size == 320);  // All 10 records intact
    unlink("/tmp/test_valid_wal.000001");
}
```
### 8.6 Group Commit Benchmark
```c
void benchmark_group_commit(void) {
    const int NUM_TRANSACTIONS = 100;
    const int NUM_THREADS = 10;
    // === Without group commit (per-commit fsync) ===
    WalWriter* writer = wal_writer_create("/tmp/bench_nogc", 0, 0);
    writer->sync_on_write = true;  // Force fsync on every write
    uint64_t start = get_time_us();
    run_concurrent_commits(writer, NUM_TRANSACTIONS, NUM_THREADS);
    uint64_t no_gc_time = get_time_us() - start;
    wal_writer_destroy(writer);
    // === With group commit ===
    writer = wal_writer_create("/tmp/bench_gc", 0, 0);
    start = get_time_us();
    run_concurrent_commits(writer, NUM_TRANSACTIONS, NUM_THREADS);
    uint64_t gc_time = get_time_us() - start;
    wal_writer_destroy(writer);
    // === Results ===
    double speedup = (double)no_gc_time / gc_time;
    printf("Without group commit: %lu us\n", no_gc_time);
    printf("With group commit:    %lu us\n", gc_time);
    printf("Speedup: %.2fx\n", speedup);
    assert(speedup >= 5.0);  // MUST achieve at least 5x
    // Cleanup
    unlink("/tmp/bench_nogc.000001");
    unlink("/tmp/bench_gc.000001");
}
```
---
## 9. Performance Targets
| Operation                        | Target       | How to Measure                                  |
|----------------------------------|--------------|-------------------------------------------------|
| Record append (buffered)         | < 10 µs      | `./benchmark --op=append --iterations=100000`   |
| Write buffer flush (1MB)         | < 100 µs     | `./benchmark --op=flush --size=1MB`             |
| Segment rotation                 | < 1 ms       | `./benchmark --op=rotate`                       |
| Group commit single              | ~fsync time  | `./benchmark --op=commit_single`                |
| Group commit batch (10 waiters)  | ~fsync time  | `./benchmark --op=commit_batch --waiters=10`    |
| Group commit throughput speedup  | ≥ 5x         | `./benchmark_group_commit`                      |
| Torn write scan (1GB log)        | < 2 s        | `./benchmark --op=torn_scan --size=1GB`         |
| Concurrent append (10 threads)   | No deadlock  | Stress test with timeout                        |
**Benchmark Implementation Template:**
```c
void benchmark_append(void) {
    WalWriter* writer = wal_writer_create("/tmp/bench", 0, 0);
    uint8_t record[64] = {0};
    const int ITERATIONS = 100000;
    uint64_t start = get_time_ns();
    for (int i = 0; i < ITERATIONS; i++) {
        Lsn lsn;
        wal_append(writer, record, sizeof(record), &lsn);
    }
    uint64_t elapsed = get_time_ns() - start;
    printf("Append latency: %llu ns/op\n", elapsed / ITERATIONS);
    wal_writer_destroy(writer);
    unlink("/tmp/bench.000001");
}
```
---
## 10. Concurrency Specification
### 10.1 Lock Ordering
```
append_lock (WalWriter)
    └── group_commit.lock (GroupCommitManager)
NEVER acquire in reverse order - deadlock risk.
```
### 10.2 Lock Hold Durations
| Lock                | Max Hold Time | When Held                              |
|---------------------|---------------|----------------------------------------|
| append_lock         | < 100 µs      | During buffer append, NOT during fsync |
| group_commit.lock   | < 1 µs (leader) or fsync duration (follower waiting) | During state check, NOT during actual fsync |
### 10.3 Thread Safety Guarantees
| Function              | Thread-Safe? | Notes                                        |
|-----------------------|--------------|----------------------------------------------|
| wal_append            | Yes          | Serialized by append_lock                    |
| wal_commit_sync       | Yes          | Uses group commit protocol                   |
| wal_flush_buffer      | Yes          | Called within append_lock                    |
| wal_force_sync        | Yes          | Independent of group commit                  |
| segment_manager_*     | No           | Protected by caller's append_lock            |
| group_commit_*        | Yes          | Own internal lock                            |
### 10.4 Memory Barriers
- `pthread_mutex_lock` and `pthread_mutex_unlock` provide full memory barriers
- `pthread_cond_wait` and `pthread_cond_broadcast` provide full memory barriers
- No additional memory barriers needed for correctness
---
## 11. Diagrams Reference
```

![Segment FD Update on Rotation](./diagrams/tdd-diag-m2-11.svg)

```
```
{{DIAGRAM:tdd-diag-m2-12}}
```
---
## 12. State Machine: Group Commit Protocol
```
States: INACTIVE -> LEADER_ACTIVE -> INACTIVE
        (any state) -> SHUTDOWN
Transitions:
  INACTIVE + first waiter   : INACTIVE -> LEADER_ACTIVE (waiter becomes leader)
  LEADER_ACTIVE + fsync done: LEADER_ACTIVE -> INACTIVE
  (any) + destroy()         : (any) -> SHUTDOWN
Per-Waiter States:
  WAITING -> COMPLETE (when last_synced_lsn >= commit_lsn)
  WAITING -> ERROR (on fsync failure)
  WAITING -> SHUTDOWN (on writer destroy)
```
---
[[CRITERIA_JSON: {"module_id": "wal-impl-m2", "criteria": ["LSN-as-offset design: next_lsn always equals current file offset; after append, next_lsn incremented by record length; no separate LSN counter exists", "Append-only writes: all records written via sequential write() at current file offset; no pread/pwrite to earlier positions; no in-place modification of existing records", "WriteBuffer batches records in memory (default 1MB); flush triggered when buffer full OR when COMMIT record appended; flush issues single write() syscall for all buffered data", "LogSegmentManager creates segment files with naming convention {base_path}.{segment_id:06d}; rotation triggered when current_offset >= segment_size_limit (default 64MB)", "Segment rotation is atomic: (1) fsync current segment, (2) close fd, (3) create new segment file, (4) update current.fd; never writing to two segments simultaneously", "GroupCommitManager implements leader/follower protocol: first committer becomes leader and calls fsync(); subsequent concurrent committers become followers waiting on condition variable", "Leader releases group_commit.lock before calling fsync(); followers wait on pthread_cond_wait; leader broadcasts on sync_complete when fsync finishes", "group_commit_sync() updates pending_sync_lsn = max(pending_sync_lsn, commit_lsn) before determining role; ensures leader syncs all pending commits", "Followers check last_synced_lsn >= commit_lsn after waking; if true, commit is durable and function returns GC_SYNC_SUCCESS", "Group commit benchmark achieves >=5x throughput improvement: 100 concurrent transactions with batched fsync vs 100 sequential per-commit fsync", "Concurrent writer serialization via pthread_mutex (append_lock); lock held only during buffer append (microseconds), released before any fsync", "Append lock protects next_lsn, write_buffer, and segment writes; group_commit.lock protects group commit state independently", "Torn write detection scans log from start: reads header, validates length range, reads full record, verifies CRC32; truncates at first invalid record via ftruncate()", "CRC mismatch on final record indicates torn write from crash; file truncated to end of last valid record; recovery (M3) proceeds from truncated position", "wal_commit_sync() flushes write buffer if commit_lsn is buffered, then calls group_commit_sync(); commit not durable until this function returns GC_SYNC_SUCCESS", "group_commit_update_fd() called after segment rotation to update GroupCommitManager's log_fd reference; prevents fsync on stale/closed fd", "Unit tests verify: sequential LSN allocation matches file offsets, segment rotation at threshold, torn write truncation, concurrent append produces no interleaved records", "Error handling: fsync failure sets last_result=GC_SYNC_ERROR and broadcasts to all waiters; all waiters receive error and must handle retry", "Shutdown safety: group_commit_destroy() sets state=GC_SHUTDOWN and broadcasts; all waiters wake with GC_SYNC_SHUTDOWN result"]}]<!-- END_TDD_MOD -->
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: wal-impl-m3 -->
# Module: ARIES Crash Recovery with CLRs
## Module ID: wal-impl-m3
---
## 1. Module Charter
This module implements the complete ARIES (Algorithm for Recovery and Isolation Exploiting Semantics) crash recovery algorithm with three phases: Analysis, Redo, and Undo. The Analysis phase scans the log from the checkpoint LSN forward, reconstructing the TransactionTable (active transactions with status, first_lsn, last_lsn) and DirtyPageTable (pages with rec_lsn tracking first dirtying LSN). The Redo phase replays ALL changes (both committed and uncommitted) starting from the minimum rec_lsn, using pageLSN comparison for idempotency—skipping records where the page's current pageLSN >= record LSN. The Undo phase rolls back all transactions that were ACTIVE at crash time using a global LSN-ordered priority queue, writing Compensation Log Records (CLRs) for each undo operation with undo_next_lsn pointing to the next record to undo. CLRs enable crash-safe recovery resumption: if the system crashes during undo, the next recovery run follows undo_next_lsn pointers to skip already-undone work. This module does NOT implement checkpoint creation (M4), transaction semantics (M1), or log writing (M2). Invariants: (1) Redo replays ALL UPDATE and CLR records regardless of transaction status, (2) Undo processes records in reverse global LSN order across all active transactions, (3) every undo operation writes a CLR before proceeding, (4) pageLSN is updated after every page modification during redo/undo, (5) recovery is idempotent—running N times produces identical database state.
---
## 2. File Structure
```
wal/
├── include/
│   ├── wal_recovery.h      # [1] RecoveryManager, recovery phases API
│   ├── wal_buffer_pool.h   # [2] BufferPool abstraction for page access
│   └── wal_undo.h          # [3] UndoWork, priority queue, CLR generation
└── src/
    ├── wal_recovery.c      # [4] Analysis, Redo, Undo phase implementations
    ├── wal_buffer_pool.c   # [5] Simple buffer pool for recovery testing
    └── wal_undo.c          # [6] Undo queue management, CLR writer
test/
├── test_analysis_phase.c       # [7] Transaction table and DPT reconstruction
├── test_redo_phase.c           # [8] Replay with pageLSN idempotency
├── test_undo_phase.c           # [9] Priority queue, prev_lsn traversal
├── test_clr_generation.c       # [10] CLR correctness, undo_next_lsn
├── test_recovery_integration.c # [11] Three-transaction scenario
├── test_idempotency.c          # [12] Recovery N times = same state
└── test_crash_during_undo.c    # [13] CLR-based recovery resumption
```
---
## 3. Complete Data Model
### 3.1 Core Constants
```c
// wal_recovery.h
#ifndef WAL_RECOVERY_H
#define WAL_RECOVERY_H
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "wal_types.h"
#include "wal_record.h"
// ============== CONSTANTS ==============
#define MAX_DIRTY_PAGES     65536       // Max pages in dirty page table
#define MAX_ACTIVE_TXNS     4096        // Max concurrent transactions
#define MAX_UNDO_QUEUE_SIZE 16384       // Max pending undo operations
#define PAGE_SIZE           4096        // Standard page size
// Recovery phase states
typedef enum {
    RECOVERY_INACTIVE = 0,
    RECOVERY_ANALYZING = 1,
    RECOVERY_REDOING = 2,
    RECOVERY_UNDOING = 3,
    RECOVERY_COMPLETE = 4,
    RECOVERY_FAILED = 5
} RecoveryState;
// Recovery error codes (extending WalError)
typedef enum {
    RECOVERY_OK = 0,
    RECOVERY_ERR_CORRUPTION = -100,     // Log record unreadable
    RECOVERY_ERR_UNDO_INCOMPLETE = -101,// Cannot find before-image
    RECOVERY_ERR_PAGE_NOT_FOUND = -102, // Page doesn't exist
    RECOVERY_ERR_LOG_READ = -103,       // Error reading log file
    RECOVERY_ERR_ALLOC = -104,          // Memory allocation failure
    RECOVERY_ERR_TXN_TABLE_FULL = -105, // Transaction table overflow
    RECOVERY_ERR_DPT_FULL = -106        // Dirty page table overflow
} RecoveryError;
#endif // WAL_RECOVERY_H
```
### 3.2 Transaction Table (Recovery-Specific)
Tracks transactions encountered during Analysis phase.
```c
typedef struct {
    TxnId     txn_id;         // Transaction identifier
    TxnStatus status;         // ACTIVE, COMMITTED, or ABORTED
    Lsn       first_lsn;      // LSN of BEGIN record (start of chain)
    Lsn       last_lsn;       // LSN of most recent record (end of chain)
} RecoveryTxnEntry;
typedef struct {
    RecoveryTxnEntry* entries;
    size_t    capacity;
    size_t    count;
    // Hash map for O(1) lookup (txn_id -> index)
    uint32_t* hash_table;
    size_t    hash_capacity;
} RecoveryTxnTable;
```
**Field Semantics:**
- `first_lsn`: Used to identify when a transaction's entire history is within the log scan range
- `last_lsn`: Starting point for undo traversal via prev_lsn chain
- `status`: Only ACTIVE transactions at end of Analysis need undo; COMMITTED/ABORTED are skipped
### 3.3 Dirty Page Table (DPT)
Tracks pages that might have uncommitted changes on disk.
```

![ARIES Three-Phase Recovery Overview](./diagrams/tdd-diag-m3-01.svg)

```
```c
typedef struct {
    PageId    page_id;        // Page identifier
    Lsn       rec_lsn;        // Recovery LSN: first LSN that dirtied this page
} DirtyPageEntry;
typedef struct {
    DirtyPageEntry* entries;
    size_t    capacity;
    size_t    count;
    // Hash map for O(1) lookup (page_id -> index)
    uint64_t* page_id_hash;
    size_t    hash_capacity;
} DirtyPageTable;
```
**Field Semantics:**
- `rec_lsn`: The LSN of the first log record that modified this page since it was last clean. Redo starts from the MINIMUM rec_lsn across all dirty pages.
- If a page is not in the DPT, it doesn't need redo (already clean on disk)
### 3.4 Buffer Pool Page Structure
Minimal buffer pool for recovery operations.
```c
// wal_buffer_pool.h
typedef struct {
    PageId    page_id;
    uint8_t   data[PAGE_SIZE];
    Lsn       page_lsn;       // LSN of most recent modification
    bool      is_dirty;       // Has been modified during recovery
    bool      is_valid;       // Data is loaded
    uint32_t  pin_count;      // Reference count
} BufferPage;
typedef struct {
    BufferPage* pages;
    size_t    capacity;
    size_t    count;
    // Hash map for O(1) lookup (page_id -> page*)
    BufferPage** page_hash;
    size_t    hash_capacity;
    // File descriptor for the database file
    int       db_fd;
    char      db_path[256];
} BufferPool;
```
**Memory Layout of BufferPage:**
```

![Analysis Phase: Building Transaction Table](./diagrams/tdd-diag-m3-02.svg)

```
| Offset  | Size    | Field       | Description                          |
|---------|---------|-------------|--------------------------------------|
| 0x0000  | 8       | page_id     | Page identifier                      |
| 0x0008  | 4096    | data        | Page content                         |
| 0x1008  | 8       | page_lsn    | LSN of last modification             |
| 0x1010  | 1       | is_dirty    | Modified flag                        |
| 0x1011  | 1       | is_valid    | Data loaded flag                     |
| 0x1012  | 4       | pin_count   | Reference count                      |
| 0x1016  | 8       | (padding)   | Align to 8 bytes                     |
| Total: 0x1020 (4128 bytes) per page |
```
### 3.5 Undo Work Item
Represents a single record that needs undo.
```c
// wal_undo.h
typedef struct {
    TxnId     txn_id;         // Transaction this undo belongs to
    Lsn       record_lsn;     // LSN of record to undo
    RecordType record_type;   // RECORD_UPDATE (CLR handled specially)
} UndoWorkItem;
```
### 3.6 Undo Priority Queue
Max-heap ordered by LSN (highest LSN first for correct undo order).
```

![Analysis Phase: Building Dirty Page Table](./diagrams/tdd-diag-m3-03.svg)

```
```c
typedef struct {
    UndoWorkItem* items;
    size_t    capacity;
    size_t    count;
} UndoQueue;
// Invariants:
// - Max-heap property: items[i].record_lsn >= items[2i+1].record_lsn
//                         AND items[i].record_lsn >= items[2i+2].record_lsn
// - Items with highest LSN are processed first
```
### 3.7 Recovery Manager
Top-level structure coordinating all recovery phases.
```c
typedef struct {
    // Recovery state
    RecoveryState state;
    // Tables built during Analysis
    RecoveryTxnTable txn_table;
    DirtyPageTable   dpt;
    // Buffer pool for page access
    BufferPool* pool;
    // Log access
    const char* log_base_path;
    int         log_fd;
    uint64_t    log_size;
    // Undo queue
    UndoQueue   undo_queue;
    // Checkpoint information (from M4)
    Lsn         checkpoint_lsn;
    // Statistics
    uint64_t    records_analyzed;
    uint64_t    records_redone;
    uint64_t    records_undone;
    uint64_t    clrs_written;
    // Error tracking
    RecoveryError last_error;
    char      error_message[256];
} RecoveryManager;
```
### 3.8 Page Header Format (On-Disk)
Each page has a header containing pageLSN for idempotency checks.
```

![Why Redo Replays Uncommitted Changes](./diagrams/tdd-diag-m3-04.svg)

```
| Offset  | Size    | Field       | Description                          |
|---------|---------|-------------|--------------------------------------|
| 0x0000  | 8       | page_lsn    | LSN of most recent log record applied|
| 0x0008  | 8       | page_id     | Page identifier (for verification)   |
| 0x0010  | 4       | free_offset | Byte offset of first free byte       |
| 0x0014  | 4       | tuple_count | Number of tuples in page             |
| 0x0018  | 4072    | data        | Page content (tuples)                |
| Total: 4096 bytes (PAGE_SIZE) |
```
### 3.9 Recovery Flow State Diagram
```

![Redo Phase: pageLSN Skip Logic](./diagrams/tdd-diag-m3-05.svg)

```
**Phase Transitions:**
```
INACTIVE -> ANALYZING (start recovery)
ANALYZING -> REDOING (analysis complete)
REDOING -> UNDOING (redo complete)
UNDOING -> COMPLETE (all active txns undone)
(any state) -> FAILED (unrecoverable error)
```
---
## 4. Interface Contracts
### 4.1 Recovery Manager Functions
```c
// wal_recovery.h
/**
 * Initialize recovery manager.
 * 
 * @param mgr        Recovery manager to initialize
 * @param pool       Buffer pool for page access (caller-owned)
 * @param log_path   Base path for log segments (e.g., "/data/wal")
 * @return           RECOVERY_OK or error code
 */
int recovery_manager_init(RecoveryManager* mgr, BufferPool* pool, 
                          const char* log_path);
/**
 * Destroy recovery manager, freeing all resources.
 * Does NOT destroy the buffer pool (caller-owned).
 */
void recovery_manager_destroy(RecoveryManager* mgr);
/**
 * Set checkpoint LSN for recovery starting point.
 * 
 * @param mgr           Recovery manager
 * @param checkpoint_lsn LSN of last CHECKPOINT_END (0 = no checkpoint)
 */
void recovery_set_checkpoint(RecoveryManager* mgr, Lsn checkpoint_lsn);
/**
 * Run complete crash recovery (all three phases).
 * 
 * @param mgr        Recovery manager
 * @return           RECOVERY_OK or error code
 * 
 * Procedure:
 *   1. Detect and truncate torn writes
 *   2. Run Analysis phase
 *   3. Run Redo phase
 *   4. Run Undo phase
 *   5. Flush all dirty pages
 *   6. Sync log (CLRs and ABORT records)
 * 
 * After this function returns RECOVERY_OK:
 *   - All committed transactions are reflected in pages
 *   - All uncommitted transactions are rolled back
 *   - Database is in consistent state
 */
int recovery_run(RecoveryManager* mgr);
/**
 * Get human-readable error message.
 */
const char* recovery_error_message(RecoveryError err);
```
### 4.2 Analysis Phase Functions
```c
/**
 * Run Analysis phase: scan log and build tables.
 * 
 * @param mgr        Recovery manager
 * @return           RECOVERY_OK or error code
 * 
 * Algorithm:
 *   1. Start at checkpoint_lsn (or 0 if no checkpoint)
 *   2. For each record:
 *      - BEGIN: Add transaction to txn_table with ACTIVE status
 *      - UPDATE/CLR: Update txn's last_lsn, add page to DPT if not present
 *      - COMMIT: Mark transaction COMMITTED
 *      - ABORT: Mark transaction ABORTED
 *      - CHECKPOINT: Initialize tables from checkpoint data
 *   3. After scan: All ACTIVE transactions need undo
 * 
 * Postconditions:
 *   - txn_table contains all transactions active at crash time
 *   - dpt contains all potentially dirty pages with rec_lsn
 */
int recovery_analysis_phase(RecoveryManager* mgr);
/**
 * Initialize tables from checkpoint record.
 * 
 * @param mgr        Recovery manager
 * @param checkpoint_lsn LSN of CHECKPOINT_END record
 * @return           RECOVERY_OK or error code
 */
int recovery_init_from_checkpoint(RecoveryManager* mgr, Lsn checkpoint_lsn);
```
### 4.3 Redo Phase Functions
```c
/**
 * Run Redo phase: replay all changes from min rec_lsn.
 * 
 * @param mgr        Recovery manager
 * @return           RECOVERY_OK or error code
 * 
 * Algorithm:
 *   1. Find min_rec_lsn = minimum rec_lsn in DPT (0 if DPT empty)
 *   2. Scan log from min_rec_lsn forward
 *   3. For each UPDATE record:
 *      a. If page not in DPT, skip (page is clean)
 *      b. Read page, get page_lsn
 *      c. If page_lsn >= record_lsn, skip (already applied)
 *      d. Apply after-image to page
 *      e. Update page_lsn = record_lsn
 *   4. For each CLR record:
 *      a. Same logic as UPDATE, but apply undo_data
 *   5. Flush all modified pages to disk
 * 
 * CRITICAL: Redo replays ALL changes, including uncommitted transactions.
 * This is necessary because uncommitted changes may have been flushed to disk
 * (steal policy), and undo needs the current state to match log expectations.
 */
int recovery_redo_phase(RecoveryManager* mgr);
/**
 * Find minimum rec_lsn in dirty page table.
 * 
 * @param dpt        Dirty page table
 * @return           Minimum rec_lsn, or 0 if table empty
 */
Lsn dpt_find_min_rec_lsn(const DirtyPageTable* dpt);
/**
 * Apply an UPDATE record's after-image to a page.
 * 
 * @param pool       Buffer pool
 * @param page_id    Target page
 * @param offset     Byte offset within page
 * @param data       Data to write
 * @param len        Data length
 * @param lsn        LSN of the record (for page_lsn update)
 * @return           RECOVERY_OK or error code
 */
int recovery_apply_redo(BufferPool* pool, PageId page_id, uint32_t offset,
                        const uint8_t* data, uint32_t len, Lsn lsn);
```
### 4.4 Undo Phase Functions
```c
/**
 * Run Undo phase: roll back all active transactions.
 * 
 * @param mgr        Recovery manager
 * @return           RECOVERY_OK or error code
 * 
 * Algorithm:
 *   1. Build undo queue from all ACTIVE transactions
 *   2. While queue not empty:
 *      a. Pop item with highest LSN
 *      b. Read record at that LSN
 *      c. If UPDATE:
 *         - Apply before-image to page
 *         - Write CLR with undo_next_lsn = UPDATE's prev_lsn
 *         - If prev_lsn > first_lsn, push prev_lsn to queue
 *         - Else write ABORT record
 *      d. If CLR:
 *         - Follow undo_next_lsn (skip already-undone work)
 *         - If undo_next_lsn > 0, push to queue
 *         - Else write ABORT record
 *   3. Flush all modified pages
 *   4. Sync log (CLRs and ABORT records must be durable)
 */
int recovery_undo_phase(RecoveryManager* mgr);
/**
 * Build initial undo queue from transaction table.
 * 
 * @param mgr        Recovery manager
 * @return           RECOVERY_OK or error code
 * 
 * Adds last_lsn of each ACTIVE transaction to the queue.
 */
int recovery_build_undo_queue(RecoveryManager* mgr);
/**
 * Process a single undo work item.
 * 
 * @param mgr        Recovery manager
 * @param item       Work item to process
 * @param[out] next_lsn Next LSN to undo (0 if done)
 * @return           RECOVERY_OK or error code
 */
int recovery_process_undo_item(RecoveryManager* mgr, const UndoWorkItem* item,
                               Lsn* next_lsn);
```
### 4.5 CLR Generation Functions
```c
// wal_undo.h
/**
 * Write a CLR record for an undo operation.
 * 
 * @param mgr        Recovery manager
 * @param txn_id     Transaction being undone
 * @param page_id    Page being modified
 * @param offset     Byte offset within page
 * @param undo_data  Before-image (data to restore)
 * @param undo_len   Length of undo_data
 * @param undo_next_lsn Next record to undo (prev_lsn of the UPDATE)
 * @param prev_lsn   Previous record in this transaction
 * @param[out] clr_lsn LSN of the written CLR
 * @return           RECOVERY_OK or error code
 * 
 * The CLR enables crash-safe undo:
 *   - If crash occurs after CLR is written, next recovery sees it
 *   - CLR's undo_next_lsn points to the next record to undo
 *   - Recovery skips already-undone work by following undo_next_lsn
 */
int recovery_write_clr(RecoveryManager* mgr, TxnId txn_id,
                       PageId page_id, uint32_t offset,
                       const uint8_t* undo_data, uint32_t undo_len,
                       Lsn undo_next_lsn, Lsn prev_lsn,
                       Lsn* clr_lsn);
/**
 * Write an ABORT record after undo completes.
 * 
 * @param mgr        Recovery manager
 * @param txn_id     Transaction being aborted
 * @param prev_lsn   Previous record in this transaction
 * @return           RECOVERY_OK or error code
 */
int recovery_write_abort_record(RecoveryManager* mgr, TxnId txn_id, Lsn prev_lsn);
```
### 4.6 Undo Queue Functions
```c
/**
 * Initialize undo queue.
 */
int undo_queue_init(UndoQueue* queue, size_t capacity);
/**
 * Destroy undo queue.
 */
void undo_queue_destroy(UndoQueue* queue);
/**
 * Add work item to queue.
 * Queue maintains max-heap property by LSN.
 */
int undo_queue_push(UndoQueue* queue, const UndoWorkItem* item);
/**
 * Remove and return highest-LSN item.
 * Returns false if queue is empty.
 */
bool undo_queue_pop(UndoQueue* queue, UndoWorkItem* out_item);
/**
 * Check if queue is empty.
 */
bool undo_queue_is_empty(const UndoQueue* queue);
/**
 * Get current size.
 */
size_t undo_queue_size(const UndoQueue* queue);
```
### 4.7 Buffer Pool Functions
```c
// wal_buffer_pool.h
/**
 * Initialize buffer pool.
 * 
 * @param pool       Pool to initialize
 * @param capacity   Maximum number of pages
 * @param db_path    Path to database file
 * @return           RECOVERY_OK or error code
 */
int buffer_pool_init(BufferPool* pool, size_t capacity, const char* db_path);
/**
 * Destroy buffer pool, flushing dirty pages.
 */
void buffer_pool_destroy(BufferPool* pool);
/**
 * Get a page, loading from disk if necessary.
 * 
 * @param pool       Buffer pool
 * @param page_id    Page to get
 * @param[out] page  Pointer to page (DO NOT FREE)
 * @return           RECOVERY_OK or error code
 */
int buffer_pool_get_page(BufferPool* pool, PageId page_id, BufferPage** page);
/**
 * Mark page as dirty.
 */
void buffer_pool_mark_dirty(BufferPool* pool, PageId page_id);
/**
 * Flush all dirty pages to disk.
 */
int buffer_pool_flush_all(BufferPool* pool);
/**
 * Read page_lsn from a page.
 */
Lsn buffer_pool_get_page_lsn(const BufferPage* page);
/**
 * Update page_lsn on a page.
 */
void buffer_pool_set_page_lsn(BufferPage* page, Lsn lsn);
/**
 * Apply data to page at offset.
 * 
 * @param page       Page to modify
 * @param offset     Byte offset within page data
 * @param data       Data to write
 * @param len        Data length
 * @return           RECOVERY_OK or error code
 */
int buffer_page_apply(BufferPage* page, uint32_t offset, 
                      const uint8_t* data, uint32_t len);
```
### 4.8 Transaction Table Functions (Recovery)
```c
/**
 * Initialize recovery transaction table.
 */
int recovery_txn_table_init(RecoveryTxnTable* table, size_t capacity);
/**
 * Destroy recovery transaction table.
 */
void recovery_txn_table_destroy(RecoveryTxnTable* table);
/**
 * Find entry by transaction ID.
 */
RecoveryTxnEntry* recovery_txn_table_find(RecoveryTxnTable* table, TxnId txn_id);
/**
 * Add or update entry.
 */
RecoveryTxnEntry* recovery_txn_table_add(RecoveryTxnTable* table, TxnId txn_id,
                                         TxnStatus status, Lsn first_lsn, Lsn last_lsn);
/**
 * Update last_lsn for existing entry.
 */
int recovery_txn_table_update_last_lsn(RecoveryTxnTable* table, TxnId txn_id, Lsn lsn);
/**
 * Get count of ACTIVE transactions.
 */
size_t recovery_txn_table_count_active(const RecoveryTxnTable* table);
```
### 4.9 Dirty Page Table Functions
```c
/**
 * Initialize dirty page table.
 */
int dpt_init(DirtyPageTable* dpt, size_t capacity);
/**
 * Destroy dirty page table.
 */
void dpt_destroy(DirtyPageTable* dpt);
/**
 * Find entry by page ID.
 */
DirtyPageEntry* dpt_find(DirtyPageTable* dpt, PageId page_id);
/**
 * Add page with rec_lsn (no-op if already present).
 */
int dpt_add(DirtyPageTable* dpt, PageId page_id, Lsn rec_lsn);
/**
 * Get minimum rec_lsn.
 */
Lsn dpt_min_rec_lsn(const DirtyPageTable* dpt);
```
---
## 5. Algorithm Specifications
### 5.1 Complete Recovery Flow
```

![Redo Idempotency Verification](./diagrams/tdd-diag-m3-06.svg)

```
**Procedure recovery_run(mgr):**
```
1. SET STATE
   mgr->state = RECOVERY_ANALYZING
2. DETECT TORN WRITES
   err = wal_detect_truncate_torn_writes(mgr->log_base_path, &last_valid_lsn)
   IF err != WAL_OK:
       mgr->state = RECOVERY_FAILED
       RETURN err
3. RUN ANALYSIS PHASE
   err = recovery_analysis_phase(mgr)
   IF err != RECOVERY_OK:
       mgr->state = RECOVERY_FAILED
       RETURN err
   mgr->state = RECOVERY_REDOING
4. RUN REDO PHASE
   err = recovery_redo_phase(mgr)
   IF err != RECOVERY_OK:
       mgr->state = RECOVERY_FAILED
       RETURN err
   mgr->state = RECOVERY_UNDOING
5. RUN UNDO PHASE
   err = recovery_undo_phase(mgr)
   IF err != RECOVERY_OK:
       mgr->state = RECOVERY_FAILED
       RETURN err
6. FLUSH AND SYNC
   buffer_pool_flush_all(mgr->pool)
   fsync(mgr->log_fd)
7. COMPLETE
   mgr->state = RECOVERY_COMPLETE
   RETURN RECOVERY_OK
```
### 5.2 Analysis Phase Algorithm
```

![Undo Phase: Priority Queue Construction](./diagrams/tdd-diag-m3-07.svg)

```
**Procedure recovery_analysis_phase(mgr):**
```
1. INITIALIZE TABLES
   recovery_txn_table_init(&mgr->txn_table, MAX_ACTIVE_TXNS)
   dpt_init(&mgr->dpt, MAX_DIRTY_PAGES)
2. LOAD CHECKPOINT IF PRESENT
   IF mgr->checkpoint_lsn > 0:
       err = recovery_init_from_checkpoint(mgr, mgr->checkpoint_lsn)
       IF err != RECOVERY_OK:
           RETURN err
3. SET SCAN START
   scan_lsn = mgr->checkpoint_lsn
   IF scan_lsn == 0:
       scan_lsn = 1  // Start from first record
4. OPEN LOG FILE
   // Find the segment containing scan_lsn
   err = open_log_segment_for_lsn(mgr, scan_lsn)
   IF err != RECOVERY_OK:
       RETURN err
5. SCAN LOOP
   WHILE scan_lsn < mgr->log_size:
       // Read record header
       err = read_record_at_lsn(mgr, scan_lsn, buffer, &record_size)
       IF err != RECOVERY_OK:
           // End of log or error
           BREAK
       // Parse header
       parse_header(buffer, &header)
       SWITCH header.type:
           CASE RECORD_BEGIN:
               // New transaction starts
               recovery_txn_table_add(&mgr->txn_table, header.txn_id,
                                      TXN_ACTIVE, scan_lsn, scan_lsn)
           CASE RECORD_UPDATE:
               // Update transaction's last_lsn
               recovery_txn_table_update_last_lsn(&mgr->txn_table, 
                                                   header.txn_id, scan_lsn)
               // Parse UPDATE to get page_id
               parse_update_record(buffer, &update)
               // Add page to DPT if not present
               IF dpt_find(&mgr->dpt, update.page_id) == NULL:
                   dpt_add(&mgr->dpt, update.page_id, scan_lsn)
           CASE RECORD_CLR:
               // Update transaction's last_lsn
               recovery_txn_table_update_last_lsn(&mgr->txn_table,
                                                   header.txn_id, scan_lsn)
               // Parse CLR to get page_id
               parse_clr_record(buffer, &clr)
               // Add page to DPT if not present
               IF dpt_find(&mgr->dpt, clr.page_id) == NULL:
                   dpt_add(&mgr->dpt, clr.page_id, scan_lsn)
           CASE RECORD_COMMIT:
               entry = recovery_txn_table_find(&mgr->txn_table, header.txn_id)
               IF entry != NULL:
                   entry->status = TXN_COMMITTED
                   entry->last_lsn = scan_lsn
           CASE RECORD_ABORT:
               entry = recovery_txn_table_find(&mgr->txn_table, header.txn_id)
               IF entry != NULL:
                   entry->status = TXN_ABORTED
                   entry->last_lsn = scan_lsn
           CASE RECORD_CHECKPOINT:
               // Already loaded from checkpoint_lsn, but update if newer
               parse_checkpoint_record(buffer, &checkpoint)
               // Merge checkpoint tables into current tables
               merge_checkpoint_tables(mgr, &checkpoint)
       mgr->records_analyzed++
       scan_lsn += record_size
6. LOG RESULTS
   num_active = recovery_txn_table_count_active(&mgr->txn_table)
   printf("Analysis complete: %zu active transactions, %zu dirty pages\n",
          num_active, mgr->dpt.count)
7. RETURN SUCCESS
   RETURN RECOVERY_OK
```
### 5.3 Redo Phase Algorithm
```

![Undo Order: Why Global LSN Matters](./diagrams/tdd-diag-m3-08.svg)

```
**Procedure recovery_redo_phase(mgr):**
```
1. FIND REDO START POINT
   min_rec_lsn = dpt_min_rec_lsn(&mgr->dpt)
   IF min_rec_lsn == 0:
       // No dirty pages, nothing to redo
       printf("Redo phase: no dirty pages, skipping\n")
       RETURN RECOVERY_OK
   printf("Redo phase: starting from LSN %lu\n", min_rec_lsn)
2. INITIALIZE SCAN
   scan_lsn = min_rec_lsn
   records_redone = 0
3. SCAN LOOP
   WHILE scan_lsn < mgr->log_size:
       // Read record
       err = read_record_at_lsn(mgr, scan_lsn, buffer, &record_size)
       IF err != RECOVERY_OK:
           BREAK
       parse_header(buffer, &header)
       // Only UPDATE and CLR records need redo
       IF header.type == RECORD_UPDATE:
           err = redo_update_record(mgr, buffer, scan_lsn)
           IF err == RECOVERY_OK:
               records_redone++
           ELSE IF err != RECOVERY_ERR_PAGE_NOT_FOUND:
               // Real error, not just missing page
               RETURN err
       ELSE IF header.type == RECORD_CLR:
           err = redo_clr_record(mgr, buffer, scan_lsn)
           IF err == RECOVERY_OK:
               records_redone++
           ELSE IF err != RECOVERY_ERR_PAGE_NOT_FOUND:
               RETURN err
       scan_lsn += record_size
4. FLUSH MODIFIED PAGES
   buffer_pool_flush_all(mgr->pool)
   printf("Redo phase: %lu records redone\n", records_redone)
   mgr->records_redone = records_redone
5. RETURN SUCCESS
   RETURN RECOVERY_OK
```
**Procedure redo_update_record(mgr, buffer, record_lsn):**
```
1. PARSE RECORD
   err = parse_update_record(buffer, &update)
   IF err != RECOVERY_OK:
       RETURN err
2. CHECK IF PAGE IN DPT
   dpt_entry = dpt_find(&mgr->dpt, update.page_id)
   IF dpt_entry == NULL:
       // Page wasn't dirty at crash, skip
       RETURN RECOVERY_OK
3. GET PAGE
   err = buffer_pool_get_page(mgr->pool, update.page_id, &page)
   IF err != RECOVERY_OK:
       RETURN RECOVERY_ERR_PAGE_NOT_FOUND
4. IDEMPOTENCY CHECK
   page_lsn = buffer_pool_get_page_lsn(page)
   IF page_lsn >= record_lsn:
       // This change is already on disk, skip
       RETURN RECOVERY_OK
5. APPLY AFTER-IMAGE
   err = buffer_page_apply(page, update.offset, 
                           update.new_value, update.new_value_len)
   IF err != RECOVERY_OK:
       RETURN err
6. UPDATE PAGE_LSN
   buffer_pool_set_page_lsn(page, record_lsn)
   buffer_pool_mark_dirty(mgr->pool, update.page_id)
7. RETURN SUCCESS
   RETURN RECOVERY_OK
```
### 5.4 Undo Phase Algorithm
```

![CLR Prevents Recovery Loop](./diagrams/tdd-diag-m3-09.svg)

```
**Procedure recovery_undo_phase(mgr):**
```
1. CHECK FOR ACTIVE TRANSACTIONS
   num_active = recovery_txn_table_count_active(&mgr->txn_table)
   IF num_active == 0:
       printf("Undo phase: no active transactions, skipping\n")
       RETURN RECOVERY_OK
   printf("Undo phase: %zu active transactions to roll back\n", num_active)
2. INITIALIZE UNDO QUEUE
   err = undo_queue_init(&mgr->undo_queue, MAX_UNDO_QUEUE_SIZE)
   IF err != RECOVERY_OK:
       RETURN err
3. BUILD INITIAL QUEUE
   err = recovery_build_undo_queue(mgr)
   IF err != RECOVERY_OK:
       undo_queue_destroy(&mgr->undo_queue)
       RETURN err
4. PROCESS UNDO LOOP
   records_undone = 0
   clrs_written = 0
   WHILE !undo_queue_is_empty(&mgr->undo_queue):
       // Pop highest-LSN item
       undo_queue_pop(&mgr->undo_queue, &work_item)
       // Process the item
       Lsn next_lsn = 0
       err = recovery_process_undo_item(mgr, &work_item, &next_lsn)
       IF err != RECOVERY_OK:
           undo_queue_destroy(&mgr->undo_queue)
           RETURN err
       records_undone++
       // Add next item to queue if any
       IF next_lsn > 0:
           UndoWorkItem next_item = {
               .txn_id = work_item.txn_id,
               .record_lsn = next_lsn,
               .record_type = RECORD_UPDATE  // Will be determined when read
           }
           undo_queue_push(&mgr->undo_queue, &next_item)
5. FLUSH AND SYNC
   buffer_pool_flush_all(mgr->pool)
   fsync(mgr->log_fd)
   printf("Undo phase: %lu records undone, %lu CLRs written\n",
          records_undone, clrs_written)
   mgr->records_undone = records_undone
   mgr->clrs_written = clrs_written
6. CLEANUP
   undo_queue_destroy(&mgr->undo_queue)
7. RETURN SUCCESS
   RETURN RECOVERY_OK
```
**Procedure recovery_build_undo_queue(mgr):**
```
1. ITERATE OVER TRANSACTION TABLE
   FOR each entry IN mgr->txn_table:
       IF entry->status == TXN_ACTIVE AND entry->last_lsn > entry->first_lsn:
           // Add last_lsn to queue (will undo from end of chain)
           UndoWorkItem item = {
               .txn_id = entry->txn_id,
               .record_lsn = entry->last_lsn,
               .record_type = RECORD_UPDATE  // Placeholder
           }
           err = undo_queue_push(&mgr->undo_queue, &item)
           IF err != RECOVERY_OK:
               RETURN err
2. RETURN SUCCESS
   RETURN RECOVERY_OK
```
### 5.5 Process Undo Item Algorithm
```

![CLR Record Structure in Recovery Context](./diagrams/tdd-diag-m3-10.svg)

```
**Procedure recovery_process_undo_item(mgr, item, next_lsn):**
```
1. READ RECORD
   err = read_record_at_lsn(mgr, item->record_lsn, buffer, &record_size)
   IF err != RECOVERY_OK:
       RETURN err
2. PARSE HEADER
   parse_header(buffer, &header)
3. HANDLE BY RECORD TYPE
   SWITCH header.type:
       CASE RECORD_UPDATE:
           // Undo the update
           err = process_undo_update(mgr, buffer, &header, next_lsn)
           RETURN err
       CASE RECORD_CLR:
           // Already-undone work, follow undo_next_lsn
           err = process_undo_clr(mgr, buffer, &header, next_lsn)
           RETURN err
       CASE RECORD_BEGIN:
           // Reached beginning of transaction
           *next_lsn = 0
           // Write ABORT record
           err = recovery_write_abort_record(mgr, header.txn_id, 
                                              item->record_lsn)
           // Mark transaction as aborted
           entry = recovery_txn_table_find(&mgr->txn_table, header.txn_id)
           IF entry != NULL:
               entry->status = TXN_ABORTED
           RETURN err
       DEFAULT:
           // Should not happen
           RETURN RECOVERY_ERR_CORRUPTION
```
**Procedure process_undo_update(mgr, buffer, header, next_lsn):**
```
1. PARSE UPDATE RECORD
   err = parse_update_record(buffer, &update)
   IF err != RECOVERY_OK:
       RETURN err
2. GET PAGE
   err = buffer_pool_get_page(mgr->pool, update.page_id, &page)
   IF err != RECOVERY_OK:
       RETURN RECOVERY_ERR_PAGE_NOT_FOUND
3. APPLY BEFORE-IMAGE (UNDO)
   err = buffer_page_apply(page, update.offset,
                           update.old_value, update.old_value_len)
   IF err != RECOVERY_OK:
       RETURN err
4. UPDATE PAGE_LSN (will be set to CLR's LSN)
   buffer_pool_mark_dirty(mgr->pool, update.page_id)
5. WRITE CLR
   Lsn clr_lsn;
   err = recovery_write_clr(mgr, header->txn_id,
                            update.page_id, update.offset,
                            update.old_value, update.old_value_len,
                            header->prev_lsn,  // undo_next_lsn
                            item->record_lsn,  // prev_lsn for CLR
                            &clr_lsn)
   IF err != RECOVERY_OK:
       RETURN err
6. SET PAGE_LSN TO CLR LSN
   buffer_pool_set_page_lsn(page, clr_lsn)
   mgr->clrs_written++
7. DETERMINE NEXT LSN
   IF header->prev_lsn > 0:
       // More records to undo in this transaction
       *next_lsn = header->prev_lsn
   ELSE:
       // Reached BEGIN, write ABORT
       *next_lsn = 0
       entry = recovery_txn_table_find(&mgr->txn_table, header->txn_id)
       IF entry != NULL:
           recovery_write_abort_record(mgr, header->txn_id, clr_lsn)
           entry->status = TXN_ABORTED
8. RETURN SUCCESS
   RETURN RECOVERY_OK
```
**Procedure process_undo_clr(mgr, buffer, header, next_lsn):**
```
1. PARSE CLR RECORD
   err = parse_clr_record(buffer, &clr)
   IF err != RECOVERY_OK:
       RETURN err
2. CLR INDICATES ALREADY-UNDONE WORK
   // Follow undo_next_lsn to skip already-undone record
   IF clr.undo_next_lsn > 0:
       *next_lsn = clr.undo_next_lsn
   ELSE:
       // No more to undo for this transaction
       *next_lsn = 0
       // Transaction should already be marked aborted
       entry = recovery_txn_table_find(&mgr->txn_table, header->txn_id)
       IF entry != NULL AND entry->status != TXN_ABORTED:
           recovery_write_abort_record(mgr, header->txn_id, 
                                        header->lsn)
           entry->status = TXN_ABORTED
3. RETURN SUCCESS
   RETURN RECOVERY_OK
```
### 5.6 CLR Write Algorithm
**Procedure recovery_write_clr(mgr, txn_id, page_id, offset, undo_data, undo_len, undo_next_lsn, prev_lsn, clr_lsn):**
```
1. ALLOCATE LSN
   *clr_lsn = mgr->log_size  // LSN = current log offset
2. CREATE CLR RECORD
   ClrRecord clr = {
       .header = {
           .lsn = *clr_lsn,
           .type = RECORD_CLR,
           .txn_id = txn_id,
           .prev_lsn = prev_lsn,
           .length = 0  // Computed during serialization
       },
       .page_id = page_id,
       .offset = offset,
       .undo_data_len = undo_len,
       .undo_data = undo_data,
       .undo_next_lsn = undo_next_lsn
   }
3. SERIALIZE
   err = wal_serialize_clr(&clr, buffer, MAX_RECORD_SIZE, &written)
   IF err != WAL_OK:
       RETURN RECOVERY_ERR_CORRUPTION
4. WRITE TO LOG
   lseek(mgr->log_fd, 0, SEEK_END)
   bytes_written = write(mgr->log_fd, buffer, written)
   IF bytes_written != written:
       RETURN RECOVERY_ERR_LOG_READ
5. UPDATE LOG SIZE
   mgr->log_size += written
6. RETURN SUCCESS
   RETURN RECOVERY_OK
```
### 5.7 Undo Queue (Max-Heap) Implementation
```

![Three-Transaction Recovery Test Scenario](./diagrams/tdd-diag-m3-11.svg)

```
**Procedure undo_queue_push(queue, item):**
```
1. CHECK CAPACITY
   IF queue->count >= queue->capacity:
       RETURN RECOVERY_ERR_ALLOC
2. ADD AT END
   index = queue->count
   queue->items[index] = *item
   queue->count++
3. BUBBLE UP
   WHILE index > 0:
       parent = (index - 1) / 2
       IF queue->items[index].record_lsn > queue->items[parent].record_lsn:
           SWAP(queue->items[index], queue->items[parent])
           index = parent
       ELSE:
           BREAK
4. RETURN SUCCESS
   RETURN RECOVERY_OK
```
**Procedure undo_queue_pop(queue, out_item):**
```
1. CHECK EMPTY
   IF queue->count == 0:
       RETURN false
2. GET ROOT (MAX)
   *out_item = queue->items[0]
3. MOVE LAST TO ROOT
   queue->items[0] = queue->items[queue->count - 1]
   queue->count--
4. BUBBLE DOWN
   index = 0
   WHILE true:
       left = 2 * index + 1
       right = 2 * index + 2
       largest = index
       IF left < queue->count AND 
          queue->items[left].record_lsn > queue->items[largest].record_lsn:
           largest = left
       IF right < queue->count AND
          queue->items[right].record_lsn > queue->items[largest].record_lsn:
           largest = right
       IF largest != index:
           SWAP(queue->items[index], queue->items[largest])
           index = largest
       ELSE:
           BREAK
5. RETURN SUCCESS
   RETURN true
```
---
## 6. Error Handling Matrix
| Error                         | Detected By                              | Recovery Action                              | User-Visible? |
|-------------------------------|------------------------------------------|----------------------------------------------|---------------|
| RECOVERY_ERR_CORRUPTION       | CRC mismatch, invalid record type        | Abort recovery, log error, return to caller  | Yes           |
| RECOVERY_ERR_UNDO_INCOMPLETE  | UPDATE with missing before-image         | Abort recovery, page may be corrupted        | Yes           |
| RECOVERY_ERR_PAGE_NOT_FOUND   | Page ID not in buffer pool or file       | Skip redo for this record (page may be new)  | No (internal) |
| RECOVERY_ERR_LOG_READ         | read() syscall failure                   | Abort recovery, check disk                   | Yes           |
| RECOVERY_ERR_ALLOC            | malloc failure                           | Abort recovery, free what we can             | Yes           |
| RECOVERY_ERR_TXN_TABLE_FULL   | Too many concurrent transactions         | Abort recovery, increase MAX_ACTIVE_TXNS     | Yes           |
| RECOVERY_ERR_DPT_FULL         | Too many dirty pages                     | Abort recovery, increase MAX_DIRTY_PAGES     | Yes           |
**State Consistency After Error:**
- On any error, recovery manager state is set to RECOVERY_FAILED
- Buffer pool may contain partially modified pages (not flushed)
- Transaction table and DPT may be partially built
- Log file may have partial CLRs written (will be truncated on next recovery)
**Recovery After Recovery Failure:**
- Run torn write detection again (truncates partial CLRs)
- Start fresh recovery from checkpoint
---
## 7. Implementation Sequence with Checkpoints
### Phase 1: Data Structures (1 hour)
**Files:** `include/wal_recovery.h`, `include/wal_buffer_pool.h`, `include/wal_undo.h`
**Tasks:**
1. Define all constants and enums
2. Define RecoveryTxnTable and RecoveryTxnEntry
3. Define DirtyPageTable and DirtyPageEntry
4. Define BufferPool and BufferPage
5. Define UndoQueue and UndoWorkItem
6. Define RecoveryManager
**Checkpoint 1:**
```bash
gcc -c src/wal_recovery.c -I include
# Should compile without errors
# All structs defined with correct sizes
```
### Phase 2: Buffer Pool (1-2 hours)
**Files:** `src/wal_buffer_pool.c`
**Tasks:**
1. Implement buffer_pool_init/destroy
2. Implement buffer_pool_get_page (with file read)
3. Implement buffer_page_apply
4. Implement page_lsn get/set
5. Implement buffer_pool_flush_all
**Checkpoint 2:**
```bash
./test_buffer_pool
# Tests pass:
# - Create pool, load page from file
# - Modify page, flush, verify on disk
# - page_lsn round-trip
```
### Phase 3: Transaction Table and DPT (1 hour)
**Files:** Continue `src/wal_recovery.c`
**Tasks:**
1. Implement recovery_txn_table_* functions
2. Implement dpt_* functions
3. Implement hash lookups for O(1) access
**Checkpoint 3:**
```bash
./test_tables
# Tests pass:
# - Add/find transactions
# - Add/find dirty pages
# - min_rec_lsn calculation
# - count_active calculation
```
### Phase 4: Analysis Phase (2-3 hours)
**Files:** Continue `src/wal_recovery.c`
**Tasks:**
1. Implement log segment opening for LSN
2. Implement record reading at LSN
3. Implement recovery_init_from_checkpoint
4. Implement recovery_analysis_phase main loop
5. Handle all record types
**Checkpoint 4:**
```bash
./test_analysis_phase
# Tests pass:
# - Scan log, build txn table
# - Scan log, build DPT
# - Checkpoint initialization
# - COMMIT/ABORT status updates
```
### Phase 5: Redo Phase (2-3 hours)
**Files:** Continue `src/wal_recovery.c`
**Tasks:**
1. Implement dpt_min_rec_lsn
2. Implement redo_update_record
3. Implement redo_clr_record
4. Implement recovery_redo_phase main loop
5. Test idempotency (pageLSN >= record LSN skip)
**Checkpoint 5:**
```bash
./test_redo_phase
# Tests pass:
# - Redo from min_rec_lsn
# - pageLSN skip logic
# - After-image applied correctly
# - DPT filtering works
```
### Phase 6: Undo Queue (1 hour)
**Files:** `src/wal_undo.c`
**Tasks:**
1. Implement undo_queue_init/destroy
2. Implement undo_queue_push (max-heap insert)
3. Implement undo_queue_pop (max-heap extract)
4. Test heap property maintenance
**Checkpoint 6:**
```bash
./test_undo_queue
# Tests pass:
# - Push/pop maintains max-heap
# - Items returned in LSN descending order
# - Empty queue behavior
```
### Phase 7: Undo Phase (2-3 hours)
**Files:** Continue `src/wal_recovery.c`, `src/wal_undo.c`
**Tasks:**
1. Implement recovery_build_undo_queue
2. Implement process_undo_update
3. Implement process_undo_clr
4. Implement recovery_write_clr
5. Implement recovery_write_abort_record
6. Implement recovery_undo_phase main loop
**Checkpoint 7:**
```bash
./test_undo_phase
# Tests pass:
# - Undo queue built from active txns
# - Before-image applied correctly
# - CLR written with correct undo_next_lsn
# - ABORT record written at end
```
### Phase 8: Integration (1 hour)
**Files:** Continue `src/wal_recovery.c`
**Tasks:**
1. Implement recovery_manager_init/destroy
2. Implement recovery_run (complete flow)
3. Wire all phases together
4. Add statistics tracking
**Checkpoint 8:**
```bash
./test_recovery_integration
# Tests pass:
# - Full recovery from crash
# - Committed txns preserved
# - Uncommitted txns rolled back
```
### Phase 9: Three-Transaction Test (1 hour)
**Files:** `test/test_recovery_integration.c`
**Tasks:**
1. Create test scenario with T1 committed, T2 aborted, T3 active
2. Simulate crash
3. Run recovery
4. Verify final state
**Checkpoint 9:**
```bash
./test_three_transaction
# Output:
# T1 changes present: PASS
# T2 changes rolled back: PASS
# T3 changes rolled back: PASS
```
### Phase 10: Idempotency and Crash-During-Undo Tests (1-2 hours)
**Files:** `test/test_idempotency.c`, `test/test_crash_during_undo.c`
**Tasks:**
1. Implement idempotency test (run recovery N times)
2. Implement crash-during-undo simulation
3. Verify CLRs enable resumption
4. Performance benchmarks
**Checkpoint 10 (Final):**
```bash
./test_idempotency
# PASS: Recovery produces identical state on repeated runs
./test_crash_during_undo
# PASS: Recovery resumes from CLR undo_next_lsn
./benchmark_recovery
# Analysis: <1ms per 1000 records
# Redo: <5ms per 1000 records
# Undo: <5ms per 1000 records
```
---
## 8. Test Specification
### 8.1 Analysis Phase Tests
```c
void test_analysis_single_transaction(void) {
    // Create log with BEGIN, UPDATE, COMMIT
    create_test_log("/tmp/test_analysis",
        BEGIN(1, LSN=100),
        UPDATE(1, page=5, LSN=150),
        COMMIT(1, LSN=200));
    RecoveryManager mgr;
    recovery_manager_init(&mgr, pool, "/tmp/test_analysis");
    recovery_analysis_phase(&mgr);
    // Transaction should be COMMITTED
    RecoveryTxnEntry* entry = recovery_txn_table_find(&mgr.txn_table, 1);
    assert(entry != NULL);
    assert(entry->status == TXN_COMMITTED);
    assert(entry->first_lsn == 100);
    assert(entry->last_lsn == 200);
    // Page 5 should be in DPT
    DirtyPageEntry* dpe = dpt_find(&mgr.dpt, 5);
    assert(dpe != NULL);
    assert(dpe->rec_lsn == 150);
    recovery_manager_destroy(&mgr);
}
void test_analysis_active_transaction(void) {
    // Create log with BEGIN, UPDATE (no COMMIT)
    create_test_log("/tmp/test_active",
        BEGIN(1, LSN=100),
        UPDATE(1, page=5, LSN=150));
    RecoveryManager mgr;
    recovery_manager_init(&mgr, pool, "/tmp/test_active");
    recovery_analysis_phase(&mgr);
    // Transaction should be ACTIVE
    RecoveryTxnEntry* entry = recovery_txn_table_find(&mgr.txn_table, 1);
    assert(entry != NULL);
    assert(entry->status == TXN_ACTIVE);
    recovery_manager_destroy(&mgr);
}
void test_analysis_multiple_transactions(void) {
    create_test_log("/tmp/test_multi",
        BEGIN(1, LSN=100),
        BEGIN(2, LSN=150),
        UPDATE(1, page=1, LSN=200),
        COMMIT(1, LSN=250),
        UPDATE(2, page=2, LSN=300),
        BEGIN(3, LSN=350),
        UPDATE(3, page=3, LSN=400));
    // T1 committed, T2 active, T3 active
    RecoveryManager mgr;
    recovery_manager_init(&mgr, pool, "/tmp/test_multi");
    recovery_analysis_phase(&mgr);
    assert(recovery_txn_table_count_active(&mgr.txn_table) == 2);
    RecoveryTxnEntry* t1 = recovery_txn_table_find(&mgr.txn_table, 1);
    assert(t1->status == TXN_COMMITTED);
    RecoveryTxnEntry* t2 = recovery_txn_table_find(&mgr.txn_table, 2);
    assert(t2->status == TXN_ACTIVE);
    recovery_manager_destroy(&mgr);
}
```
### 8.2 Redo Phase Tests
```c
void test_redo_applies_after_image(void) {
    // Create page with old value
    create_page("/tmp/test_db", 1, "OLD", 0, 3);
    // Create log with UPDATE
    create_test_log("/tmp/test_redo",
        BEGIN(1, LSN=100),
        UPDATE(1, page=1, offset=0, old="OLD", new="NEW", LSN=150),
        COMMIT(1, LSN=200));
    RecoveryManager mgr;
    recovery_manager_init(&mgr, pool, "/tmp/test_redo");
    recovery_analysis_phase(&mgr);
    recovery_redo_phase(&mgr);
    // Page should have new value
    BufferPage* page;
    buffer_pool_get_page(pool, 1, &page);
    assert(memcmp(page->data, "NEW", 3) == 0);
    assert(page->page_lsn == 150);
    recovery_manager_destroy(&mgr);
}
void test_redo_idempotency(void) {
    // Create page already at LSN 200
    create_page_with_lsn("/tmp/test_idem", 1, "VAL", 0, 3, 200);
    // Log has UPDATE at LSN 150 (older than page)
    create_test_log("/tmp/test_idem_log",
        BEGIN(1, LSN=100),
        UPDATE(1, page=1, offset=0, old="OLD", new="VAL", LSN=150),
        COMMIT(1, LSN=200));
    RecoveryManager mgr;
    recovery_manager_init(&mgr, pool, "/tmp/test_idem_log");
    recovery_analysis_phase(&mgr);
    recovery_redo_phase(&mgr);
    // Page should NOT be modified (page_lsn >= record_lsn)
    BufferPage* page;
    buffer_pool_get_page(pool, 1, &page);
    assert(memcmp(page->data, "VAL", 3) == 0);  // Unchanged
    assert(page->page_lsn == 200);  // Original LSN preserved
    recovery_manager_destroy(&mgr);
}
void test_redo_includes_uncommitted(void) {
    // Page starts empty
    create_page("/tmp/test_uncom", 1, "___", 0, 3);
    // Log has uncommitted UPDATE
    create_test_log("/tmp/test_uncom_log",
        BEGIN(1, LSN=100),
        UPDATE(1, page=1, offset=0, old="___", new="NEW", LSN=150));
    // No COMMIT
    RecoveryManager mgr;
    recovery_manager_init(&mgr, pool, "/tmp/test_uncom_log");
    recovery_analysis_phase(&mgr);
    recovery_redo_phase(&mgr);
    // Page should have NEW (uncommitted change replayed)
    BufferPage* page;
    buffer_pool_get_page(pool, 1, &page);
    assert(memcmp(page->data, "NEW", 3) == 0);
    // Undo phase will roll this back
    recovery_undo_phase(&mgr);
    // Now page should have original value
    assert(memcmp(page->data, "___", 3) == 0);
    recovery_manager_destroy(&mgr);
}
```
### 8.3 Undo Phase Tests
```c
void test_undo_single_transaction(void) {
    create_page("/tmp/test_undo", 1, "NEW", 0, 3);
    create_test_log("/tmp/test_undo_log",
        BEGIN(1, LSN=100),
        UPDATE(1, page=1, offset=0, old="OLD", new="NEW", LSN=150));
    RecoveryManager mgr;
    recovery_manager_init(&mgr, pool, "/tmp/test_undo_log");
    recovery_run(&mgr);
    // Page should be rolled back to OLD
    BufferPage* page;
    buffer_pool_get_page(pool, 1, &page);
    assert(memcmp(page->data, "OLD", 3) == 0);
    // Transaction should be ABORTED
    RecoveryTxnEntry* entry = recovery_txn_table_find(&mgr.txn_table, 1);
    assert(entry->status == TXN_ABORTED);
    recovery_manager_destroy(&mgr);
}
void test_undo_global_lsn_order(void) {
    // Two transactions modify same page
    create_page("/tmp/test_order", 1, "A", 0, 1);
    create_test_log("/tmp/test_order_log",
        BEGIN(1, LSN=100),
        BEGIN(2, LSN=150),
        UPDATE(1, page=1, offset=0, old="A", new="B", LSN=200),
        UPDATE(2, page=1, offset=0, old="B", new="C", LSN=250));
    // Both active, should undo in reverse LSN order
    RecoveryManager mgr;
    recovery_manager_init(&mgr, pool, "/tmp/test_order_log");
    recovery_run(&mgr);
    // Page should be back to "A"
    BufferPage* page;
    buffer_pool_get_page(pool, 1, &page);
    assert(page->data[0] == 'A');
    recovery_manager_destroy(&mgr);
}
void test_undo_writes_clr(void) {
    create_page("/tmp/test_clr", 1, "NEW", 0, 3);
    create_test_log("/tmp/test_clr_log",
        BEGIN(1, LSN=100),
        UPDATE(1, page=1, offset=0, old="OLD", new="NEW", LSN=150));
    RecoveryManager mgr;
    recovery_manager_init(&mgr, pool, "/tmp/test_clr_log");
    recovery_run(&mgr);
    // Log should contain CLR
    assert(mgr.clrs_written == 1);
    // Read CLR and verify undo_next_lsn
    ClrRecord clr;
    read_last_clr_from_log("/tmp/test_clr_log", &clr);
    assert(clr.undo_next_lsn == 100);  // Points to BEGIN
    recovery_manager_destroy(&mgr);
}
```
### 8.4 Three-Transaction Integration Test
```c
void test_three_transaction_scenario(void) {
    // Setup: T1 committed, T2 aborted (has CLRs), T3 active
    create_page("/tmp/test_3txn/p1", 1, "A", 0, 1);
    create_page("/tmp/test_3txn/p2", 2, "X", 0, 1);
    create_page("/tmp/test_3txn/p3", 3, "P", 0, 1);
    create_test_log("/tmp/test_3txn_log",
        // T1: Committed
        BEGIN(1, LSN=100),
        UPDATE(1, page=1, offset=0, old="A", new="B", LSN=150),
        COMMIT(1, LSN=200),
        // T2: Aborted (with CLR)
        BEGIN(2, LSN=250),
        UPDATE(2, page=2, offset=0, old="X", new="Y", LSN=300),
        CLR(2, page=2, offset=0, undo="X", undo_next=250, LSN=350),
        ABORT(2, LSN=400),
        // T3: Active at crash
        BEGIN(3, LSN=450),
        UPDATE(3, page=3, offset=0, old="P", new="Q", LSN=500));
    // Crash here
    RecoveryManager mgr;
    recovery_manager_init(&mgr, pool, "/tmp/test_3txn_log");
    recovery_run(&mgr);
    // Verify final state
    BufferPage* p1, *p2, *p3;
    buffer_pool_get_page(pool, 1, &p1);
    buffer_pool_get_page(pool, 2, &p2);
    buffer_pool_get_page(pool, 3, &p3);
    // T1's change should be present (committed)
    assert(p1->data[0] == 'B');
    // T2's change should be rolled back (aborted)
    assert(p2->data[0] == 'X');
    // T3's change should be rolled back (was active)
    assert(p3->data[0] == 'P');
    recovery_manager_destroy(&mgr);
    printf("Three-transaction test PASSED\n");
}
```
### 8.5 Idempotency Test
```c
void test_recovery_idempotency(void) {
    // Create initial state
    create_page("/tmp/test_idem_db/p1", 1, "A", 0, 1);
    create_page("/tmp/test_idem_db/p2", 2, "X", 0, 1);
    create_test_log("/tmp/test_idem_db_log",
        BEGIN(1, LSN=100),
        UPDATE(1, page=1, offset=0, old="A", new="B", LSN=150),
        COMMIT(1, LSN=200),
        BEGIN(2, LSN=250),
        UPDATE(2, page=2, offset=0, old="X", new="Y", LSN=300));
    // Run recovery first time
    RecoveryManager mgr;
    recovery_manager_init(&mgr, pool, "/tmp/test_idem_db_log");
    recovery_run(&mgr);
    // Capture state
    char state1_p1, state1_p2;
    buffer_pool_get_page(pool, 1, &page);
    state1_p1 = page->data[0];
    buffer_pool_get_page(pool, 2, &page);
    state1_p2 = page->data[0];
    recovery_manager_destroy(&mgr);
    // Run recovery second time
    recovery_manager_init(&mgr, pool, "/tmp/test_idem_db_log");
    recovery_run(&mgr);
    // Capture state again
    char state2_p1, state2_p2;
    buffer_pool_get_page(pool, 1, &page);
    state2_p1 = page->data[0];
    buffer_pool_get_page(pool, 2, &page);
    state2_p2 = page->data[0];
    // States must be identical
    assert(state1_p1 == state2_p1);
    assert(state1_p2 == state2_p2);
    printf("Idempotency test PASSED: state identical after 2 recovery runs\n");
    recovery_manager_destroy(&mgr);
}
```
### 8.6 Crash During Undo Test
```c
void test_crash_during_undo_resumes(void) {
    // Create page
    create_page("/tmp/test_cdu/p1", 1, "A", 0, 1);
    // Log with transaction that needs undo, plus a CLR indicating partial undo
    create_test_log("/tmp/test_cdu_log",
        BEGIN(1, LSN=100),
        UPDATE(1, page=1, offset=0, old="A", new="B", LSN=150),
        UPDATE(1, page=1, offset=1, old="A", new="C", LSN=200),
        // Partial undo: CLR for LSN 200
        CLR(1, page=1, offset=1, undo="A", undo_next=150, LSN=250));
    // Crash here - T1 still active, LSN 150 not undone
    RecoveryManager mgr;
    recovery_manager_init(&mgr, pool, "/tmp/test_cdu_log");
    recovery_run(&mgr);
    // Verify undo resumed from CLR's undo_next_lsn
    BufferPage* page;
    buffer_pool_get_page(pool, 1, &page);
    // Both updates should be undone
    assert(page->data[0] == 'A');
    assert(page->data[1] == 'A');
    // Transaction should be aborted
    RecoveryTxnEntry* entry = recovery_txn_table_find(&mgr.txn_table, 1);
    assert(entry->status == TXN_ABORTED);
    printf("Crash-during-undo test PASSED: recovery resumed from CLR\n");
    recovery_manager_destroy(&mgr);
}
```
---
## 9. Performance Targets
| Operation                        | Target              | How to Measure                                  |
|----------------------------------|---------------------|-------------------------------------------------|
| Analysis phase (1000 records)    | < 1 ms              | `./benchmark_recovery --phase=analysis --records=1000` |
| Redo phase (1000 records)        | < 5 ms (with I/O)   | `./benchmark_recovery --phase=redo --records=1000` |
| Undo phase (1000 records)        | < 5 ms (with I/O)   | `./benchmark_recovery --phase=undo --records=1000` |
| Complete recovery (10000 records)| < 50 ms             | `./benchmark_recovery --records=10000`          |
| CLR write                        | < 100 µs            | `./benchmark_recovery --op=clr_write`           |
| Undo queue push/pop              | < 1 µs              | `./benchmark_recovery --op=queue`               |
| pageLSN check (idempotency)      | < 100 ns            | `./benchmark_recovery --op=page_lsn_check`      |
| Idempotency verification         | 100% match          | Compare page state after N recovery runs        |
**Benchmark Template:**
```c
void benchmark_analysis_phase(void) {
    // Create log with 10000 records
    create_large_test_log("/tmp/bench", 10000);
    RecoveryManager mgr;
    recovery_manager_init(&mgr, pool, "/tmp/bench");
    uint64_t start = get_time_ns();
    recovery_analysis_phase(&mgr);
    uint64_t elapsed = get_time_ns() - start;
    printf("Analysis: %lu records in %lu us (%.2f us/record)\n",
           mgr.records_analyzed,
           elapsed / 1000,
           (double)elapsed / mgr.records_analyzed / 1000);
    recovery_manager_destroy(&mgr);
}
```
---
## 10. Concurrency Specification
### 10.1 Single-Threaded During Recovery
Recovery is inherently single-threaded:
- No concurrent transactions during recovery
- No concurrent page modifications
- Undo queue processing is sequential
### 10.2 Lock Requirements (for future multi-threaded extension)
If recovery is extended for parallelism:
```
Lock Ordering:
  txn_table.lock -> dpt.lock -> undo_queue.lock
Never acquire in reverse order.
```
### 10.3 Memory Barriers
Single-threaded recovery requires no memory barriers. All state transitions are sequential.
---
## 11. Diagrams Reference
```

![Recovery Idempotency Test Flow](./diagrams/tdd-diag-m3-12.svg)

```
```

![Crash During Undo Test](./diagrams/tdd-diag-m3-13.svg)

```
```

![Transaction Table State Transitions During Recovery](./diagrams/tdd-diag-m3-14.svg)

```
```

![Dirty Page Table with rec_lsn](./diagrams/tdd-diag-m3-15.svg)

```
```

![Complete Recovery Flow Integration](./diagrams/tdd-diag-m3-16.svg)

```
---
## 12. State Machine: Recovery Flow
```
States: INACTIVE -> ANALYZING -> REDOING -> UNDOING -> COMPLETE
        (any state) -> FAILED
Transitions:
  INACTIVE + recovery_run()      : INACTIVE -> ANALYZING
  ANALYZING + complete           : ANALYZING -> REDOING
  REDOING + complete             : REDOING -> UNDOING
  UNDOING + complete             : UNDOING -> COMPLETE
  (any) + unrecoverable error    : (any) -> FAILED
Per-Transaction States (in txn_table):
  (not present) -> ACTIVE (on BEGIN)
  ACTIVE -> COMMITTED (on COMMIT record)
  ACTIVE -> ABORTED (on ABORT record or after undo)
  COMMITTED/ABORTED -> (terminal, no further transitions)
```
---
[[CRITERIA_JSON: {"module_id": "wal-impl-m3", "criteria": ["RecoveryTxnTable tracks txn_id, status (ACTIVE/COMMITTED/ABORTED), first_lsn (BEGIN record), last_lsn (most recent record) for each transaction encountered during Analysis phase", "DirtyPageTable tracks page_id and rec_lsn (first LSN that dirtied the page); rec_lsn determines redo starting point via minimum across all entries", "Analysis phase scans log from checkpoint_lsn forward (or from LSN 1 if no checkpoint), building txn_table and DPT by processing BEGIN, UPDATE, CLR, COMMIT, ABORT, CHECKPOINT records", "Analysis handles CHECKPOINT records by initializing/merging transaction table and dirty page table from checkpoint snapshot data", "Redo phase starts from dpt_min_rec_lsn() and replays ALL UPDATE and CLR records regardless of transaction committed status", "Redo idempotency: before applying record, read page_lsn from page; if page_lsn >= record_lsn, skip (change already on disk)", "Redo updates page_lsn to record_lsn after applying each change, preventing re-application on subsequent recovery attempts", "Redo applies UPDATE records by writing new_value (after-image) at specified offset; applies CLR records by writing undo_data", "Undo phase builds UndoQueue (max-heap by LSN) containing last_lsn of each ACTIVE transaction from txn_table", "Undo processes records in global reverse-LSN order using priority queue (max-heap pop returns highest LSN first)", "Each undo of UPDATE record: (1) apply old_value (before-image) to page, (2) write CLR with undo_next_lsn=UPDATE's prev_lsn, (3) update page_lsn to CLR's LSN", "CLR's undo_next_lsn points to prev_lsn of the UPDATE being undone, enabling crash recovery to skip already-undone work", "Undo encounters existing CLR: follow undo_next_lsn to determine next record to undo (skip already-undone UPDATE)", "When undo reaches BEGIN record (prev_lsn=0): write ABORT record, mark transaction status=ABORTED", "Recovery idempotency verified by test: run recovery N times, compare all page contents - must be byte-identical", "Torn write detection runs before recovery phases: scan log, truncate at first record with invalid CRC32", "Three-transaction test: T1 COMMITTED → changes present after recovery; T2 ABORTED (CLRs in log) → changes rolled back; T3 ACTIVE at crash → changes rolled back during undo", "Crash-during-undo test: log contains partial CLRs, recovery follows undo_next_lsn pointers to resume correctly", "Both redo and undo phases call buffer_pool_flush_all() before returning, ensuring all modified pages are persisted", "Transaction status transitions: BEGIN adds ACTIVE entry, COMMIT marks COMMITTED, ABORT marks ABORTED, undo completion writes ABORT and marks ABORTED"]}]
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: wal-impl-m4 -->
# Module: Fuzzy Checkpointing & Log Truncation
## Module ID: wal-impl-m4
---
## 1. Module Charter
This module implements non-blocking fuzzy checkpointing that captures transaction table and dirty page table snapshots while transactions continue executing, bounded recovery time by providing a known starting point for the Analysis phase, and safe log truncation that reclaims disk space by deleting old log segments. The fuzzy checkpoint writes CHECKPOINT_BEGIN, captures atomic snapshots via brief mutex-protected copy, writes CHECKPOINT_END with serialized snapshot data, and updates a master record file atomically via the temp-file-rename pattern. The master record stores the LSN of the last completed CHECKPOINT_END, enabling recovery to bootstrap without scanning the entire log. Log truncation safely removes segments that end before the checkpoint's minimum rec_lsn AND contain no records needed for undo of transactions active at checkpoint time. A background thread schedules checkpoints based on configurable time or record-count intervals. This module does NOT implement transaction semantics (M1), log writing (M2), or recovery phases (M3). Invariants: (1) master record always points to a valid, complete checkpoint (or is invalid/corrupt with CRC mismatch), (2) no segment containing records needed for recovery is ever truncated, (3) checkpoint snapshot capture holds locks for <100µs, (4) long-running transactions prevent truncation of segments containing their firstLSN.
---
## 2. File Structure
```
wal/
├── include/
│   ├── wal_checkpoint.h     # [1] CheckpointManager, CheckpointScheduler APIs
│   ├── wal_master_record.h  # [2] MasterRecord structure and atomic update
│   └── wal_truncate.h       # [3] LogTruncator, segment safety checks
└── src/
    ├── wal_checkpoint.c     # [4] Fuzzy checkpoint implementation
    ├── wal_master_record.c  # [5] Master record read/write with atomicity
    └── wal_truncate.c       # [6] Log segment truncation logic
test/
├── test_checkpoint_fuzzy.c      # [7] Concurrent transactions during checkpoint
├── test_master_record.c         # [8] Atomic update, corruption handling
├── test_log_truncation.c        # [9] Safe vs unsafe segment deletion
├── test_checkpoint_recovery.c   # [10] Recovery time with/without checkpoint
└── test_long_txn_blocking.c     # [11] Long transaction prevents truncation
```
---
## 3. Complete Data Model
### 3.1 Core Constants
```c
// wal_checkpoint.h
#ifndef WAL_CHECKPOINT_H
#define WAL_CHECKPOINT_H
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <pthread.h>
#include "wal_types.h"
#include "wal_record.h"
// ============== CONSTANTS ==============
#define MASTER_RECORD_MAGIC     0x57414C4D   // "WALM" - magic number
#define MASTER_RECORD_VERSION   1
#define MASTER_RECORD_PATH      "wal.master"
#define DEFAULT_CHECKPOINT_INTERVAL_MS    30000   // 30 seconds
#define DEFAULT_CHECKPOINT_RECORD_COUNT   10000   // 10k records
#define MAX_CHECKPOINT_TXN_ENTRIES        4096
#define MAX_CHECKPOINT_DIRTY_PAGES        65536
#define CHECKPOINT_SNAPSHOT_TIMEOUT_US    100     // 100µs max lock hold
// Checkpoint states
typedef enum {
    CHECKPOINT_INACTIVE = 0,     // No checkpoint in progress
    CHECKPOINT_BEGIN_WRITTEN = 1,// CHECKPOINT_BEGIN written, capturing
    CHECKPOINT_END_WRITING = 2,  // Writing CHECKPOINT_END
    CHECKPOINT_SYNCING = 3,      // Syncing log and updating master
    CHECKPOINT_SHUTDOWN = 4      // Checkpoint system shutting down
} CheckpointState;
// Checkpoint trigger types
typedef enum {
    TRIGGER_NONE = 0,
    TRIGGER_TIME = 1,
    TRIGGER_RECORD_COUNT = 2,
    TRIGGER_MANUAL = 3,
    TRIGGER_SHUTDOWN = 4
} CheckpointTrigger;
// Error codes
typedef enum {
    CHECKPOINT_OK = 0,
    CHECKPOINT_ERR_IN_PROGRESS = -200,
    CHECKPOINT_ERR_SNAPSHOT_FAILED = -201,
    CHECKPOINT_ERR_MASTER_WRITE = -202,
    CHECKPOINT_ERR_LOG_WRITE = -203,
    CHECKPOINT_ERR_ALLOC = -204,
    CHECKPOINT_ERR_SHUTDOWN = -205
} CheckpointError;
#endif // WAL_CHECKPOINT_H
```
### 3.2 Master Record Structure
The master record is a small file (one disk block) storing checkpoint bootstrap information. It's the first thing recovery reads.
```
{{DIAGRAM:tdd-diag-m4-01}}
```
**Byte Layout:**
| Offset | Size | Field             | Type      | Description                           |
|--------|------|-------------------|-----------|---------------------------------------|
| 0x00   | 4    | magic             | uint32_t  | MASTER_RECORD_MAGIC (0x57414C4D)      |
| 0x04   | 4    | version           | uint32_t  | Format version (currently 1)          |
| 0x08   | 8    | last_checkpoint_lsn | uint64_t | LSN of last CHECKPOINT_END record     |
| 0x10   | 8    | checkpoint_count  | uint64_t  | Number of checkpoints taken           |
| 0x18   | 8    | create_time       | uint64_t  | Unix timestamp of master record creation |
| 0x20   | 8    | update_time       | uint64_t  | Unix timestamp of last update         |
| 0x28   | 4    | crc32             | uint32_t  | CRC32 of bytes 0x00-0x27              |
| Total: 0x2C (44 bytes, padded to 64 for sector alignment) |
```c
// wal_master_record.h
#ifndef WAL_MASTER_RECORD_H
#define WAL_MASTER_RECORD_H
#include "wal_types.h"
#define MASTER_RECORD_SIZE      64      // Padded to sector size
typedef struct {
    uint32_t magic;            // 0x00-0x03
    uint32_t version;          // 0x04-0x07
    uint64_t last_checkpoint_lsn;  // 0x08-0x0F
    uint64_t checkpoint_count;     // 0x10-0x17
    uint64_t create_time;          // 0x18-0x1F
    uint64_t update_time;          // 0x20-0x27
    uint32_t crc32;                // 0x28-0x2B
    uint8_t  padding[20];          // 0x2C-0x3F (pad to 64 bytes)
} MasterRecord;
_Static_assert(sizeof(MasterRecord) == MASTER_RECORD_SIZE, 
               "MasterRecord must be exactly 64 bytes");
// Error codes for master record operations
typedef enum {
    MASTER_OK = 0,
    MASTER_ERR_FILE_OPEN = -300,
    MASTER_ERR_FILE_WRITE = -301,
    MASTER_ERR_FILE_READ = -302,
    MASTER_ERR_RENAME = -303,
    MASTER_ERR_CRC_MISMATCH = -304,
    MASTER_ERR_INVALID_MAGIC = -305,
    MASTER_ERR_INVALID_VERSION = -306,
    MASTER_ERR_ALLOC = -307
} MasterRecordError;
#endif // WAL_MASTER_RECORD_H
```
### 3.3 Checkpoint Snapshot Structures
Snapshots captured during fuzzy checkpoint, stored in CHECKPOINT_END record.
```c
// wal_checkpoint.h
// Snapshot of a single transaction entry
typedef struct {
    TxnId     txn_id;
    TxnStatus status;
    Lsn       first_lsn;
    Lsn       last_lsn;
} CheckpointTxnEntry;  // 24 bytes
// Snapshot of a single dirty page entry
typedef struct {
    PageId    page_id;
    Lsn       rec_lsn;
} CheckpointDirtyPageEntry;  // 16 bytes
// Complete checkpoint snapshot
typedef struct {
    uint64_t begin_lsn;        // LSN of CHECKPOINT_BEGIN
    uint64_t end_lsn;          // LSN of CHECKPOINT_END (0 until written)
    uint32_t num_active_txns;
    CheckpointTxnEntry* txn_entries;
    uint32_t num_dirty_pages;
    CheckpointDirtyPageEntry* dirty_page_entries;
    uint64_t capture_time_us;  // Timestamp when snapshot was taken
} CheckpointSnapshot;
```
### 3.4 Checkpoint Manager
Orchestrates fuzzy checkpoint lifecycle.
```

![Fuzzy Checkpoint Timeline](./diagrams/tdd-diag-m4-02.svg)

```
```c
typedef struct {
    // State machine
    CheckpointState state;
    pthread_mutex_t state_lock;
    pthread_cond_t  state_change;
    // Current checkpoint data
    CheckpointSnapshot current_snapshot;
    Lsn                current_begin_lsn;
    // Configuration
    uint64_t time_interval_ms;      // 0 = disabled
    uint64_t record_count_interval; // 0 = disabled
    // Statistics
    uint64_t checkpoint_count;
    uint64_t total_checkpoint_time_us;
    uint64_t last_checkpoint_time_us;
    // References to shared state (not owned)
    TransactionTable* active_txn_table;   // From M1 TransactionManager
    DirtyPageTable*   active_dpt;         // From M3 recovery or buffer pool
    // References to log infrastructure (not owned)
    WalWriter*        wal_writer;         // From M2
    const char*       log_base_path;
    // Master record path
    char              master_record_path[256];
    // Error tracking
    CheckpointError   last_error;
    char              error_message[256];
} CheckpointManager;
```
### 3.5 Checkpoint Scheduler
Determines when checkpoints should be triggered.
```c
typedef struct {
    // Configuration
    uint64_t time_interval_ms;
    uint64_t record_count_interval;
    // State tracking
    uint64_t last_checkpoint_time_ms;   // Timestamp of last checkpoint
    uint64_t last_checkpoint_lsn;       // LSN of last CHECKPOINT_END
    uint64_t records_since_checkpoint;  // Records written since last checkpoint
    // Trigger source for last checkpoint
    CheckpointTrigger last_trigger;
} CheckpointScheduler;
```
### 3.6 Log Truncator
Determines which segments can be safely deleted.
```

![Checkpoint Record Payload Structure](./diagrams/tdd-diag-m4-03.svg)

```
```c
// wal_truncate.h
#ifndef WAL_TRUNCATE_H
#define WAL_TRUNCATE_H
#include "wal_types.h"
// Segment metadata for truncation decisions
typedef struct {
    uint32_t segment_id;
    char     file_path[512];
    uint64_t start_lsn;        // LSN of first record in segment
    uint64_t end_lsn;          // LSN of last record (0 if active)
    uint64_t file_size;
    bool     is_active;        // True if this is the current write segment
} SegmentMetadata;
typedef struct {
    const char* base_path;
    SegmentMetadata* segments;
    size_t    segment_count;
    size_t    segment_capacity;
} LogTruncator;
// Truncation result
typedef struct {
    uint32_t segments_deleted;
    uint64_t bytes_freed;
    uint32_t segments_blocked;  // Segments that couldn't be deleted
    uint32_t blocking_txns;     // Transactions blocking truncation
} TruncationResult;
// Error codes
typedef enum {
    TRUNCATE_OK = 0,
    TRUNCATE_ERR_DIR_READ = -400,
    TRUNCATE_ERR_FILE_DELETE = -401,
    TRUNCATE_ERR_SEGMENT_ACTIVE = -402,
    TRUNCATE_ERR_TXN_BLOCKING = -403,
    TRUNCATE_ERR_ALLOC = -404
} TruncateError;
#endif // WAL_TRUNCATE_H
```
### 3.7 Checkpoint System (Background Thread)
Complete system with background checkpoint thread.
```c
typedef struct {
    CheckpointManager   manager;
    CheckpointScheduler scheduler;
    LogTruncator        truncator;
    // Background thread
    pthread_t           checkpoint_thread;
    bool                thread_running;
    bool                shutdown_requested;
    pthread_mutex_t     shutdown_lock;
    pthread_cond_t      shutdown_cond;
    // Sync interval for thread wake-up
    uint64_t            check_interval_ms;
} CheckpointSystem;
```
### 3.8 File Layout: Master Record Location
```

![Master Record: Bootstrap Location](./diagrams/tdd-diag-m4-04.svg)

```
**On-Disk Structure:**
```
/data/wal/
├── wal.master           ← Master record (64 bytes)
├── wal.000001           ← Segment 1
├── wal.000002           ← Segment 2
├── wal.000003           ← Current active segment
└── wal.master.tmp       ← Temporary file during atomic update
```
### 3.9 Checkpoint Timeline: Concurrent with Transactions
```

![Master Record Atomic Update Pattern](./diagrams/tdd-diag-m4-05.svg)

```
**Timeline Example:**
```
Time    Checkpoint Thread              Transaction Thread 1      Transaction Thread 2
----    -------------------            -------------------      -------------------
t0      Write CHECKPOINT_BEGIN
t1      Acquire txn_table lock         (blocked briefly)
t2      Copy txn_table snapshot        (blocked briefly)
t3      Release txn_table lock         BEGIN T3                  UPDATE T2
t4      Acquire dpt lock               COMMIT T3                 (continues)
t5      Copy dpt snapshot              (continues)               (continues)
t6      Release dpt lock               (continues)               COMMIT T2
t7      Write CHECKPOINT_END           (continues)               (continues)
t8      fsync log                      (continues)               (continues)
t9      Update master record           (continues)               (continues)
t10     Checkpoint complete
```
---
## 4. Interface Contracts
### 4.1 Master Record Functions
```c
// wal_master_record.h
/**
 * Initialize a new master record with default values.
 * 
 * @param rec      Record to initialize
 */
void master_record_init(MasterRecord* rec);
/**
 * Read master record from file.
 * 
 * @param path     Path to master record file
 * @param[out] rec Read record
 * @return         MASTER_OK, MASTER_ERR_FILE_READ, MASTER_ERR_CRC_MISMATCH,
 *                 MASTER_ERR_INVALID_MAGIC, MASTER_ERR_INVALID_VERSION
 * 
 * If the file doesn't exist, returns MASTER_ERR_FILE_READ with errno == ENOENT.
 * In this case, caller should treat as "no checkpoint" (recovery from log start).
 */
int master_record_read(const char* path, MasterRecord* rec);
/**
 * Write master record to file atomically.
 * 
 * @param path     Path to master record file
 * @param rec      Record to write
 * @return         MASTER_OK or error code
 * 
 * Algorithm:
 *   1. Serialize record to buffer (with CRC)
 *   2. Write to temporary file (path + ".tmp")
 *   3. fsync temporary file
 *   4. Rename temporary to final path (atomic on POSIX)
 *   5. fsync parent directory (to ensure rename is durable)
 * 
 * The rename() syscall is atomic: after it returns, either the old or new
 * file is visible, never a partial state.
 */
int master_record_write(const char* path, const MasterRecord* rec);
/**
 * Validate master record fields.
 * 
 * @param rec      Record to validate
 * @return         true if valid, false otherwise
 * 
 * Checks:
 *   - magic == MASTER_RECORD_MAGIC
 *   - version <= MASTER_RECORD_VERSION (forward compatible)
 *   - CRC32 matches
 */
bool master_record_validate(const MasterRecord* rec);
/**
 * Compute CRC32 for master record.
 * 
 * @param rec      Record to checksum
 * @return         CRC32 value
 */
uint32_t master_record_compute_crc(const MasterRecord* rec);
```
### 4.2 Checkpoint Manager Functions
```c
// wal_checkpoint.h
/**
 * Initialize checkpoint manager.
 * 
 * @param mgr          Manager to initialize
 * @param wal_writer   WAL writer for writing checkpoint records
 * @param txn_table    Transaction table to snapshot (shared reference)
 * @param dpt          Dirty page table to snapshot (shared reference)
 * @param master_path  Path for master record file
 * @return             CHECKPOINT_OK or error code
 */
int checkpoint_manager_init(CheckpointManager* mgr,
                            WalWriter* wal_writer,
                            TransactionTable* txn_table,
                            DirtyPageTable* dpt,
                            const char* master_path);
/**
 * Destroy checkpoint manager.
 */
void checkpoint_manager_destroy(CheckpointManager* mgr);
/**
 * Take a fuzzy checkpoint.
 * 
 * @param mgr        Checkpoint manager
 * @param trigger    What triggered this checkpoint
 * @param[out] checkpoint_lsn LSN of CHECKPOINT_END record
 * @return           CHECKPOINT_OK or error code
 * 
 * Algorithm:
 *   1. Check state == CHECKPOINT_INACTIVE (return error if not)
 *   2. Set state = CHECKPOINT_BEGIN_WRITTEN
 *   3. Write CHECKPOINT_BEGIN record
 *   4. Capture transaction table snapshot (brief lock)
 *   5. Capture dirty page table snapshot (brief lock)
 *   6. Set state = CHECKPOINT_END_WRITING
 *   7. Write CHECKPOINT_END record with snapshot data
 *   8. Set state = CHECKPOINT_SYNCING
 *   9. Sync log (fsync)
 *   10. Update master record (atomic)
 *   11. Set state = CHECKPOINT_INACTIVE
 * 
 * CRITICAL: Transactions continue during steps 4-10.
 * The snapshot represents a consistent but slightly stale view.
 */
int checkpoint_take(CheckpointManager* mgr, CheckpointTrigger trigger,
                    Lsn* checkpoint_lsn);
/**
 * Check if a checkpoint is currently in progress.
 */
bool checkpoint_is_in_progress(const CheckpointManager* mgr);
/**
 * Get last checkpoint LSN from master record.
 * 
 * @param mgr        Checkpoint manager
 * @param[out] lsn   Last checkpoint LSN (0 if no valid checkpoint)
 * @return           CHECKPOINT_OK or error code
 */
int checkpoint_get_last_lsn(CheckpointManager* mgr, Lsn* lsn);
/**
 * Get statistics.
 */
void checkpoint_get_stats(const CheckpointManager* mgr,
                          uint64_t* count,
                          uint64_t* total_time_us,
                          uint64_t* last_time_us);
```
### 4.3 Checkpoint Scheduler Functions
```c
/**
 * Initialize checkpoint scheduler.
 * 
 * @param scheduler        Scheduler to initialize
 * @param time_interval_ms Time between checkpoints (0 = disabled)
 * @param record_count     Records between checkpoints (0 = disabled)
 */
void checkpoint_scheduler_init(CheckpointScheduler* scheduler,
                               uint64_t time_interval_ms,
                               uint64_t record_count);
/**
 * Check if a checkpoint should be taken.
 * 
 * @param scheduler   Scheduler
 * @param current_lsn Current WAL LSN
 * @param current_time_ms Current timestamp
 * @return            TRIGGER_TIME, TRIGGER_RECORD_COUNT, or TRIGGER_NONE
 */
CheckpointTrigger checkpoint_scheduler_should_checkpoint(
    CheckpointScheduler* scheduler,
    Lsn current_lsn,
    uint64_t current_time_ms);
/**
 * Record that a checkpoint was taken.
 */
void checkpoint_scheduler_record_checkpoint(CheckpointScheduler* scheduler,
                                            Lsn checkpoint_lsn,
                                            uint64_t checkpoint_time_ms);
/**
 * Record that records were written.
 */
void checkpoint_scheduler_record_written(CheckpointScheduler* scheduler,
                                         uint64_t record_count);
```
### 4.4 Log Truncator Functions
```c
// wal_truncate.h
/**
 * Initialize log truncator.
 * 
 * @param truncator  Truncator to initialize
 * @param base_path  Base path for log segments
 * @return           TRUNCATE_OK or error code
 */
int truncator_init(LogTruncator* truncator, const char* base_path);
/**
 * Destroy log truncator.
 */
void truncator_destroy(LogTruncator* truncator);
/**
 * Scan log directory and build segment metadata.
 * 
 * @param truncator  Truncator
 * @return           TRUNCATE_OK or error code
 */
int truncator_scan_segments(LogTruncator* truncator);
/**
 * Check if a segment can be safely truncated.
 * 
 * @param truncator       Truncator
 * @param segment         Segment to check
 * @param checkpoint_min_rec_lsn Minimum rec_lsn from checkpoint
 * @param active_txns     Transactions active at checkpoint time
 * @param num_active_txns Number of active transactions
 * @return                true if safe to truncate, false otherwise
 * 
 * Safety rules:
 *   1. segment.end_lsn < checkpoint_min_rec_lsn (not needed for redo)
 *   2. No active transaction has first_lsn within segment range
 */
bool truncator_can_delete_segment(const LogTruncator* truncator,
                                  const SegmentMetadata* segment,
                                  Lsn checkpoint_min_rec_lsn,
                                  const CheckpointTxnEntry* active_txns,
                                  uint32_t num_active_txns);
/**
 * Truncate old log segments.
 * 
 * @param truncator       Truncator
 * @param checkpoint_snapshot Snapshot from last checkpoint
 * @param[out] result     Truncation result
 * @return                TRUNCATE_OK or error code
 */
int truncator_truncate(LogTruncator* truncator,
                       const CheckpointSnapshot* checkpoint_snapshot,
                       TruncationResult* result);
/**
 * Delete a single segment file.
 * 
 * @param truncator  Truncator
 * @param segment_id Segment to delete
 * @return           TRUNCATE_OK or error code
 */
int truncator_delete_segment(LogTruncator* truncator, uint32_t segment_id);
/**
 * Get segment metadata by ID.
 */
const SegmentMetadata* truncator_find_segment(const LogTruncator* truncator,
                                               uint32_t segment_id);
```
### 4.5 Checkpoint System Functions
```c
// wal_checkpoint.h
/**
 * Create complete checkpoint system with background thread.
 * 
 * @param base_path       Base path for log and master record
 * @param wal_writer      WAL writer
 * @param txn_table       Transaction table
 * @param dpt             Dirty page table
 * @param time_interval   Checkpoint interval in ms (0 = use default)
 * @param record_interval Record count interval (0 = use default)
 * @return                Allocated system or NULL on error
 */
CheckpointSystem* checkpoint_system_create(const char* base_path,
                                           WalWriter* wal_writer,
                                           TransactionTable* txn_table,
                                           DirtyPageTable* dpt,
                                           uint64_t time_interval,
                                           uint64_t record_interval);
/**
 * Destroy checkpoint system.
 * 
 * Signals background thread to shutdown, waits for completion,
 * then frees all resources.
 */
void checkpoint_system_destroy(CheckpointSystem* sys);
/**
 * Force an immediate checkpoint.
 * 
 * @param sys        Checkpoint system
 * @param[out] lsn   LSN of checkpoint (can be NULL)
 * @return           CHECKPOINT_OK or error code
 */
int checkpoint_system_force_checkpoint(CheckpointSystem* sys, Lsn* lsn);
/**
 * Get last checkpoint LSN.
 */
Lsn checkpoint_system_get_last_lsn(CheckpointSystem* sys);
/**
 * Background thread function (internal).
 */
void* checkpoint_thread_func(void* arg);
```
---
## 5. Algorithm Specifications
### 5.1 Fuzzy Checkpoint Algorithm
```

![Log Truncation Safety Regions](./diagrams/tdd-diag-m4-06.svg)

```
**Procedure checkpoint_take(mgr, trigger, checkpoint_lsn):**
```
1. VALIDATE STATE
   pthread_mutex_lock(&mgr->state_lock)
   IF mgr->state != CHECKPOINT_INACTIVE:
       pthread_mutex_unlock(&mgr->state_lock)
       RETURN CHECKPOINT_ERR_IN_PROGRESS
   mgr->state = CHECKPOINT_BEGIN_WRITTEN
   pthread_mutex_unlock(&mgr->state_lock)
2. START TIMING
   start_time = get_time_us()
3. WRITE CHECKPOINT_BEGIN
   begin_record = {
       .header = {
           .type = RECORD_CHECKPOINT_BEGIN,
           .txn_id = 0,  // System operation
       }
   }
   err = wal_append(mgr->wal_writer, begin_record, &mgr->current_begin_lsn)
   IF err != WAL_OK:
       mgr->state = CHECKPOINT_INACTIVE
       RETURN CHECKPOINT_ERR_LOG_WRITE
4. CAPTURE TRANSACTION TABLE SNAPSHOT
   snapshot_start = get_time_us()
   pthread_mutex_lock(&mgr->active_txn_table->lock)
   // Copy transaction entries
   mgr->current_snapshot.num_active_txns = mgr->active_txn_table->count
   ALLOCATE mgr->current_snapshot.txn_entries (count * sizeof(CheckpointTxnEntry))
   FOR i = 0 TO mgr->active_txn_table->count - 1:
       mgr->current_snapshot.txn_entries[i] = convert_to_checkpoint_entry(
           &mgr->active_txn_table->entries[i])
   pthread_mutex_unlock(&mgr->active_txn_table->lock)
   snapshot_duration = get_time_us() - snapshot_start
   ASSERT snapshot_duration < CHECKPOINT_SNAPSHOT_TIMEOUT_US
5. CAPTURE DIRTY PAGE TABLE SNAPSHOT
   snapshot_start = get_time_us()
   pthread_mutex_lock(&mgr->active_dpt->lock)
   mgr->current_snapshot.num_dirty_pages = mgr->active_dpt->count
   ALLOCATE mgr->current_snapshot.dirty_page_entries (count * sizeof(...))
   FOR i = 0 TO mgr->active_dpt->count - 1:
       mgr->current_snapshot.dirty_page_entries[i] = convert_to_checkpoint_entry(
           &mgr->active_dpt->entries[i])
   pthread_mutex_unlock(&mgr->active_dpt->lock)
   snapshot_duration = get_time_us() - snapshot_start
   ASSERT snapshot_duration < CHECKPOINT_SNAPSHOT_TIMEOUT_US
6. UPDATE STATE
   pthread_mutex_lock(&mgr->state_lock)
   mgr->state = CHECKPOINT_END_WRITING
   pthread_mutex_unlock(&mgr->state_lock)
7. WRITE CHECKPOINT_END
   end_record = build_checkpoint_end_record(&mgr->current_snapshot)
   err = wal_append(mgr->wal_writer, end_record, &mgr->current_snapshot.end_lsn)
   IF err != WAL_OK:
       mgr->state = CHECKPOINT_INACTIVE
       free snapshot memory
       RETURN CHECKPOINT_ERR_LOG_WRITE
8. SYNC LOG
   pthread_mutex_lock(&mgr->state_lock)
   mgr->state = CHECKPOINT_SYNCING
   pthread_mutex_unlock(&mgr->state_lock)
   err = wal_force_sync(mgr->wal_writer)
   IF err != WAL_OK:
       mgr->state = CHECKPOINT_INACTIVE
       RETURN CHECKPOINT_ERR_LOG_WRITE
9. UPDATE MASTER RECORD
   master = {
       .magic = MASTER_RECORD_MAGIC,
       .version = MASTER_RECORD_VERSION,
       .last_checkpoint_lsn = mgr->current_snapshot.end_lsn,
       .checkpoint_count = mgr->checkpoint_count + 1,
       .update_time = get_time_unix()
   }
   err = master_record_write(mgr->master_record_path, &master)
   IF err != MASTER_OK:
       // Log is synced, checkpoint is valid, but master record is stale
       // Recovery will use previous checkpoint - acceptable
       LOG_WARNING("Master record update failed")
10. COMPLETE
    pthread_mutex_lock(&mgr->state_lock)
    mgr->state = CHECKPOINT_INACTIVE
    pthread_cond_broadcast(&mgr->state_change)
    pthread_mutex_unlock(&mgr->state_lock)
11. UPDATE STATISTICS
    end_time = get_time_us()
    mgr->checkpoint_count++
    mgr->total_checkpoint_time_us += (end_time - start_time)
    mgr->last_checkpoint_time_us = end_time - start_time
    *checkpoint_lsn = mgr->current_snapshot.end_lsn
12. FREE SNAPSHOT MEMORY
    free(mgr->current_snapshot.txn_entries)
    free(mgr->current_snapshot.dirty_page_entries)
    mgr->current_snapshot.txn_entries = NULL
    mgr->current_snapshot.dirty_page_entries = NULL
13. RETURN SUCCESS
    RETURN CHECKPOINT_OK
```
### 5.2 Master Record Atomic Write Algorithm
```

![Long Transaction Blocking Truncation](./diagrams/tdd-diag-m4-07.svg)

```
**Procedure master_record_write(path, rec):**
```
1. COMPUTE CRC
   rec_copy = *rec
   rec_copy.crc32 = 0
   rec_copy.crc32 = wal_compute_crc(&rec_copy, offsetof(MasterRecord, crc32))
2. BUILD TEMPORARY PATH
   temp_path = path + ".tmp"
3. CREATE TEMPORARY FILE
   temp_fd = open(temp_path, O_WRONLY | O_CREAT | O_TRUNC, 0644)
   IF temp_fd < 0:
       RETURN MASTER_ERR_FILE_OPEN
4. WRITE TO TEMPORARY FILE
   bytes_written = write(temp_fd, &rec_copy, sizeof(MasterRecord))
   IF bytes_written != sizeof(MasterRecord):
       close(temp_fd)
       unlink(temp_path)
       RETURN MASTER_ERR_FILE_WRITE
5. SYNC TEMPORARY FILE
   fsync(temp_fd)
   close(temp_fd)
6. ATOMIC RENAME
   result = rename(temp_path, path)
   IF result < 0:
       unlink(temp_path)
       RETURN MASTER_ERR_RENAME
7. SYNC PARENT DIRECTORY (optional but recommended)
   dir_path = dirname(path)
   dir_fd = open(dir_path, O_RDONLY | O_DIRECTORY)
   IF dir_fd >= 0:
       fsync(dir_fd)
       close(dir_fd)
8. RETURN SUCCESS
   RETURN MASTER_OK
```
**Why this is atomic:**
- `rename()` is atomic on POSIX: it either completes or doesn't happen
- Before rename: old master record is valid
- After rename: new master record is valid
- During rename: filesystem guarantees no intermediate state is visible
### 5.3 Log Truncation Safety Check Algorithm
```

![Checkpoint Interval Tradeoff Curve](./diagrams/tdd-diag-m4-08.svg)

```
**Procedure truncator_can_delete_segment(truncator, segment, checkpoint_min_rec_lsn, active_txns, num_active_txns):**
```
1. CHECK IF SEGMENT IS ACTIVE
   IF segment->is_active:
       // Never truncate the segment we're currently writing to
       RETURN false
2. CHECK REDO REQUIREMENT
   // Segment must end before checkpoint's minimum rec_lsn
   // (all changes in segment are guaranteed to be on disk)
   IF segment->end_lsn >= checkpoint_min_rec_lsn:
       // Segment may contain records needed for redo
       RETURN false
3. CHECK UNDO REQUIREMENT
   // No active transaction should have its firstLSN in this segment
   FOR i = 0 TO num_active_txns - 1:
       txn = &active_txns[i]
       IF txn->first_lsn >= segment->start_lsn AND
          txn->first_lsn <= segment->end_lsn:
           // This transaction's undo chain starts in this segment
           RETURN false
4. SAFE TO TRUNCATE
   RETURN true
```
### 5.4 Complete Truncation Algorithm
**Procedure truncator_truncate(truncator, checkpoint_snapshot, result):**
```
1. SCAN SEGMENTS
   err = truncator_scan_segments(truncator)
   IF err != TRUNCATE_OK:
       RETURN err
2. FIND CHECKPOINT MIN REC LSN
   min_rec_lsn = UINT64_MAX
   FOR i = 0 TO checkpoint_snapshot->num_dirty_pages - 1:
       IF checkpoint_snapshot->dirty_page_entries[i].rec_lsn < min_rec_lsn:
           min_rec_lsn = checkpoint_snapshot->dirty_page_entries[i].rec_lsn
   IF min_rec_lsn == UINT64_MAX:
       min_rec_lsn = 0  // No dirty pages
3. INITIALIZE RESULT
   result->segments_deleted = 0
   result->bytes_freed = 0
   result->segments_blocked = 0
   result->blocking_txns = 0
4. CHECK EACH SEGMENT
   FOR i = 0 TO truncator->segment_count - 1:
       segment = &truncator->segments[i]
       IF truncator_can_delete_segment(truncator, segment, min_rec_lsn,
                                       checkpoint_snapshot->txn_entries,
                                       checkpoint_snapshot->num_active_txns):
           // Delete segment file
           err = truncator_delete_segment(truncator, segment->segment_id)
           IF err == TRUNCATE_OK:
               result->segments_deleted++
               result->bytes_freed += segment->file_size
           ELSE:
               result->segments_blocked++
       ELSE:
           result->segments_blocked++
           // Track blocking transactions
           FOR j = 0 TO checkpoint_snapshot->num_active_txns - 1:
               txn = &checkpoint_snapshot->txn_entries[j]
               IF txn->first_lsn >= segment->start_lsn AND
                  txn->first_lsn <= segment->end_lsn:
                   result->blocking_txns++
5. RETURN SUCCESS
   RETURN TRUNCATE_OK
```
### 5.5 Background Checkpoint Thread Algorithm
```

![Checkpoint State Machine](./diagrams/tdd-diag-m4-09.svg)

```
**Procedure checkpoint_thread_func(sys):**
```
1. INITIALIZATION
   scheduler = &sys->scheduler
   mgr = &sys->manager
2. MAIN LOOP
   WHILE !sys->shutdown_requested:
       // Sleep for check interval
       usleep(sys->check_interval_ms * 1000)
       // Check for shutdown
       IF sys->shutdown_requested:
           BREAK
       // Check if checkpoint needed
       current_time = get_time_ms()
       current_lsn = wal_writer_get_next_lsn(sys->manager.wal_writer)
       trigger = checkpoint_scheduler_should_checkpoint(
           scheduler, current_lsn, current_time)
       IF trigger != TRIGGER_NONE:
           // Take checkpoint
           Lsn checkpoint_lsn;
           err = checkpoint_take(mgr, trigger, &checkpoint_lsn)
           IF err == CHECKPOINT_OK:
               // Update scheduler
               checkpoint_scheduler_record_checkpoint(
                   scheduler, checkpoint_lsn, current_time)
               // Attempt truncation
               TruncationResult trunc_result;
               truncator_truncate(&sys->truncator, &mgr->current_snapshot,
                                  &trunc_result)
               LOG_INFO("Checkpoint complete: LSN=%lu, deleted %u segments",
                        checkpoint_lsn, trunc_result.segments_deleted)
           ELSE:
               LOG_ERROR("Checkpoint failed: %d", err)
3. SHUTDOWN CHECKPOINT
   // Take final checkpoint on shutdown
   IF !sys->shutdown_requested:
       Lsn final_lsn;
       checkpoint_take(mgr, TRIGGER_SHUTDOWN, &final_lsn)
4. EXIT THREAD
   RETURN NULL
```
### 5.6 Recovery Integration: Read Master Record
**Procedure recovery_read_checkpoint_lsn(log_base_path, checkpoint_lsn):**
```
1. BUILD MASTER RECORD PATH
   master_path = log_base_path + "/" + MASTER_RECORD_PATH
   // Or if log_base_path includes filename prefix:
   // master_path = dirname(log_base_path) + "/" + MASTER_RECORD_PATH
2. READ MASTER RECORD
   MasterRecord master;
   err = master_record_read(master_path, &master)
3. HANDLE READ ERRORS
   IF err == MASTER_ERR_FILE_READ AND errno == ENOENT:
       // No master record - first run or deleted
       *checkpoint_lsn = 0
       RETURN RECOVERY_OK
   IF err == MASTER_ERR_CRC_MISMATCH:
       // Corrupted master record
       LOG_WARNING("Master record CRC mismatch, starting from log beginning")
       *checkpoint_lsn = 0
       RETURN RECOVERY_OK
   IF err == MASTER_ERR_INVALID_MAGIC OR err == MASTER_ERR_INVALID_VERSION:
       // Invalid or incompatible master record
       LOG_WARNING("Master record invalid, starting from log beginning")
       *checkpoint_lsn = 0
       RETURN RECOVERY_OK
   IF err != MASTER_OK:
       RETURN RECOVERY_ERR_LOG_READ
4. VALIDATE CHECKPOINT LSN
   IF master.last_checkpoint_lsn == 0:
       // No checkpoint taken yet
       *checkpoint_lsn = 0
       RETURN RECOVERY_OK
5. RETURN CHECKPOINT LSN
   *checkpoint_lsn = master.last_checkpoint_lsn
   RETURN RECOVERY_OK
```
---
## 6. Error Handling Matrix
| Error                          | Detected By                          | Recovery Action                              | User-Visible? |
|--------------------------------|--------------------------------------|----------------------------------------------|---------------|
| CHECKPOINT_ERR_IN_PROGRESS     | checkpoint_take when state != INACTIVE | Return error, caller may retry after wait  | Yes (API)     |
| CHECKPOINT_ERR_SNAPSHOT_FAILED | Mutex lock timeout during capture    | Abort checkpoint, reset state to INACTIVE    | Yes (API)     |
| CHECKPOINT_ERR_MASTER_WRITE    | master_record_write fails            | Checkpoint valid but master stale; log warning | No (log)    |
| CHECKPOINT_ERR_LOG_WRITE       | wal_append or fsync fails            | Abort checkpoint, reset state, return error  | Yes (API)     |
| CHECKPOINT_ERR_ALLOC           | malloc failure                       | Abort checkpoint, free partial, return error | Yes (API)     |
| CHECKPOINT_ERR_SHUTDOWN        | Operation during system shutdown     | Return error, no state change                | Yes (API)     |
| MASTER_ERR_FILE_OPEN           | open() fails on temp file            | Return error, no partial file                | Yes (API)     |
| MASTER_ERR_FILE_WRITE          | write() fails                        | Close and unlink temp file, return error     | Yes (API)     |
| MASTER_ERR_RENAME              | rename() fails                       | Unlink temp file, old master intact, return error | Yes (API) |
| MASTER_ERR_CRC_MISMATCH        | CRC validation on read               | Return error, caller treats as "no checkpoint" | Yes (API)  |
| MASTER_ERR_INVALID_MAGIC       | Magic number check on read           | Return error, caller treats as "no checkpoint" | Yes (API)  |
| TRUNCATE_ERR_DIR_READ          | scandir() or stat() fails            | Return error, no segments deleted            | Yes (API)     |
| TRUNCATE_ERR_FILE_DELETE       | unlink() fails                       | Log error, continue with other segments      | No (log)      |
| TRUNCATE_ERR_SEGMENT_ACTIVE    | Attempt to delete active segment     | Skip segment, no error                       | No (internal) |
| TRUNCATE_ERR_TXN_BLOCKING      | Active txn in segment range          | Skip segment, tracked in result              | No (internal) |
**State Consistency After Error:**
- Checkpoint failure: state reset to INACTIVE, no partial checkpoint visible
- Master record write failure: previous master record intact, checkpoint valid
- Truncation failure: some segments may be deleted, no data loss (segments still exist on disk if needed)
---
## 7. Implementation Sequence with Checkpoints
### Phase 1: Master Record (1 hour)
**Files:** `include/wal_master_record.h`, `src/wal_master_record.c`
**Tasks:**
1. Define MasterRecord struct with static assert for size
2. Implement master_record_init()
3. Implement master_record_compute_crc()
4. Implement master_record_validate()
5. Implement master_record_read()
6. Implement master_record_write() with atomic temp-file-rename
**Checkpoint 1:**
```bash
gcc -c src/wal_master_record.c -I include -lz
./test_master_record_basic
# Tests pass:
# - Init creates valid record
# - Write creates file
# - Read recovers written data
# - CRC mismatch detected
# - Atomic rename works (test with concurrent reader)
```
### Phase 2: Checkpoint Snapshot Capture (1 hour)
**Files:** `include/wal_checkpoint.h` (partial), `src/wal_checkpoint.c` (partial)
**Tasks:**
1. Define CheckpointTxnEntry and CheckpointDirtyPageEntry
2. Define CheckpointSnapshot
3. Implement snapshot capture functions (with timing assertions)
4. Test lock hold duration < 100µs
**Checkpoint 2:**
```bash
./test_snapshot_capture
# Tests pass:
# - Empty tables captured
# - Large tables captured
# - Lock hold duration < 100µs
# - Concurrent modification doesn't corrupt snapshot
```
### Phase 3: Checkpoint Manager State Machine (1-2 hours)
**Files:** Continue `src/wal_checkpoint.c`
**Tasks:**
1. Define CheckpointManager struct
2. Implement checkpoint_manager_init/destroy
3. Implement state transitions
4. Implement CHECKPOINT_BEGIN and CHECKPOINT_END record writing
5. Wire to WalWriter from M2
**Checkpoint 3:**
```bash
./test_checkpoint_basic
# Tests pass:
# - State transitions correct
# - Records written to log
# - CHECKPOINT_IN_PROGRESS error when double-call
```
### Phase 4: Master Record Integration (0.5 hours)
**Files:** Continue `src/wal_checkpoint.c`
**Tasks:**
1. Add master record path to CheckpointManager
2. Call master_record_write after fsync
3. Handle write errors gracefully
**Checkpoint 4:**
```bash
./test_checkpoint_master
# Tests pass:
# - Master record updated after checkpoint
# - Master record contains correct LSN
# - Master record survives crash simulation (kill -9)
```
### Phase 5: Checkpoint Scheduler (0.5 hours)
**Files:** Continue `src/wal_checkpoint.c`
**Tasks:**
1. Define CheckpointScheduler struct
2. Implement time-based trigger
3. Implement record-count trigger
4. Implement should_checkpoint check
**Checkpoint 5:**
```bash
./test_checkpoint_scheduler
# Tests pass:
# - Time trigger fires after interval
# - Record count trigger fires after threshold
# - Both triggers work independently
```
### Phase 6: Log Truncator (1-2 hours)
**Files:** `include/wal_truncate.h`, `src/wal_truncate.c`
**Tasks:**
1. Define SegmentMetadata and LogTruncator
2. Implement truncator_init/destroy
3. Implement truncator_scan_segments (read directory)
4. Implement truncator_can_delete_segment
5. Implement truncator_truncate
6. Implement truncator_delete_segment
**Checkpoint 6:**
```bash
./test_log_truncation
# Tests pass:
# - Scan finds all segments
# - Safe segments identified correctly
# - Unsafe segments not deleted
# - Active transaction blocks truncation
```
### Phase 7: Background Thread (1 hour)
**Files:** Continue `src/wal_checkpoint.c`
**Tasks:**
1. Define CheckpointSystem struct
2. Implement checkpoint_system_create
3. Implement checkpoint_thread_func
4. Implement checkpoint_system_destroy (graceful shutdown)
5. Wire scheduler, manager, and truncator together
**Checkpoint 7:**
```bash
./test_checkpoint_thread
# Tests pass:
# - Thread starts and runs
# - Checkpoints taken automatically
# - Truncation happens after checkpoint
# - Graceful shutdown completes final checkpoint
```
### Phase 8: Recovery Integration (1 hour)
**Files:** Modify `src/wal_recovery.c` from M3
**Tasks:**
1. Add recovery_read_checkpoint_lsn function
2. Modify recovery_run to read master record first
3. Initialize txn_table and dpt from checkpoint if present
4. Start Analysis from checkpoint_lsn instead of 0
**Checkpoint 8:**
```bash
./test_recovery_with_checkpoint
# Tests pass:
# - Recovery reads master record
# - Recovery starts from checkpoint LSN
# - Tables initialized from checkpoint
# - Corrupted master record handled gracefully
```
### Phase 9: Recovery Time Benchmark (1 hour)
**Files:** `test/test_checkpoint_recovery.c`
**Tasks:**
1. Create test with 10000 records
2. Measure recovery time without checkpoint
3. Take checkpoint at 50%
4. Measure recovery time with checkpoint
5. Verify recovery time reduction
**Checkpoint 9 (Final):**
```bash
./benchmark_recovery_time
# Output:
# Recovery without checkpoint: X ms
# Recovery with checkpoint: Y ms
# Speedup: X/Y (must be >= 1.67, i.e., <= 60%)
./run_all_tests
# All tests pass
valgrind --leak-check=full ./test_all
# No memory leaks
```
---
## 8. Test Specification
### 8.1 Master Record Tests
```c
void test_master_record_roundtrip(void) {
    MasterRecord original;
    master_record_init(&original);
    original.last_checkpoint_lsn = 12345;
    original.checkpoint_count = 42;
    original.update_time = get_time_unix();
    // Write
    assert(master_record_write("/tmp/test_master", &original) == MASTER_OK);
    // Read
    MasterRecord read_back;
    assert(master_record_read("/tmp/test_master", &read_back) == MASTER_OK);
    // Verify
    assert(read_back.magic == MASTER_RECORD_MAGIC);
    assert(read_back.version == MASTER_RECORD_VERSION);
    assert(read_back.last_checkpoint_lsn == 12345);
    assert(read_back.checkpoint_count == 42);
    assert(master_record_validate(&read_back) == true);
    unlink("/tmp/test_master");
}
void test_master_record_crc_detection(void) {
    MasterRecord rec;
    master_record_init(&rec);
    master_record_write("/tmp/test_crc", &rec);
    // Corrupt the file
    int fd = open("/tmp/test_crc", O_RDWR);
    pwrite(fd, "X", 1, 8);  // Corrupt last_checkpoint_lsn
    close(fd);
    // Read should fail CRC
    MasterRecord read_back;
    assert(master_record_read("/tmp/test_crc", &read_back) == MASTER_ERR_CRC_MISMATCH);
    unlink("/tmp/test_crc");
}
void test_master_record_atomic_update(void) {
    // Write initial record
    MasterRecord rec;
    master_record_init(&rec);
    rec.last_checkpoint_lsn = 100;
    master_record_write("/tmp/test_atomic", &rec);
    // Simulate crash during update: temp file exists, no rename
    MasterRecord new_rec;
    master_record_init(&new_rec);
    new_rec.last_checkpoint_lsn = 200;
    // Write to temp file manually (simulating partial update)
    int fd = open("/tmp/test_atomic.tmp", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    write(fd, &new_rec, sizeof(new_rec));
    close(fd);
    // Read should still get old record
    MasterRecord read_back;
    assert(master_record_read("/tmp/test_atomic", &read_back) == MASTER_OK);
    assert(read_back.last_checkpoint_lsn == 100);  // Old value
    unlink("/tmp/test_atomic");
    unlink("/tmp/test_atomic.tmp");
}
void test_master_record_missing_file(void) {
    MasterRecord rec;
    int err = master_record_read("/tmp/nonexistent_master", &rec);
    assert(err == MASTER_ERR_FILE_READ);
    assert(errno == ENOENT);
}
```
### 8.2 Checkpoint Manager Tests
```c
void test_checkpoint_basic_flow(void) {
    // Setup
    WalWriter* writer = wal_writer_create("/tmp/test_cp_wal", 0, 0);
    TransactionTable txn_table;
    DirtyPageTable dpt;
    txn_table_init(&txn_table, 100);
    dpt_init(&dpt, 100);
    CheckpointManager mgr;
    checkpoint_manager_init(&mgr, writer, &txn_table, &dpt, "/tmp/test_cp_master");
    // Take checkpoint
    Lsn checkpoint_lsn;
    assert(checkpoint_take(&mgr, TRIGGER_MANUAL, &checkpoint_lsn) == CHECKPOINT_OK);
    assert(checkpoint_lsn > 0);
    // Verify state
    assert(mgr.state == CHECKPOINT_INACTIVE);
    assert(mgr.checkpoint_count == 1);
    // Verify master record
    MasterRecord master;
    assert(master_record_read("/tmp/test_cp_master", &master) == MASTER_OK);
    assert(master.last_checkpoint_lsn == checkpoint_lsn);
    // Cleanup
    checkpoint_manager_destroy(&mgr);
    wal_writer_destroy(writer);
    unlink("/tmp/test_cp_wal.000001");
    unlink("/tmp/test_cp_master");
}
void test_checkpoint_concurrent_transactions(void) {
    // Setup with real transaction manager
    TransactionManager* txn_mgr = setup_test_transaction_manager();
    CheckpointManager cp_mgr;
    // ... initialize ...
    // Start transaction T1
    TxnId t1 = begin_txn(txn_mgr);
    // Take checkpoint (T1 is ACTIVE)
    Lsn checkpoint_lsn;
    checkpoint_take(&cp_mgr, TRIGGER_MANUAL, &checkpoint_lsn);
    // T1 should be in checkpoint snapshot
    MasterRecord master;
    master_record_read("/tmp/test_cp_master", &master);
    // Read checkpoint record
    CheckpointRecord cp_rec;
    read_checkpoint_record(txn_mgr->log, checkpoint_lsn, &cp_rec);
    assert(cp_rec.num_active_txns >= 1);
    bool found_t1 = false;
    for (uint32_t i = 0; i < cp_rec.num_active_txns; i++) {
        if (cp_rec.active_txns[i].txn_id == t1) {
            found_t1 = true;
            assert(cp_rec.active_txns[i].status == TXN_ACTIVE);
        }
    }
    assert(found_t1);
    // Commit T1 after checkpoint
    commit_txn(txn_mgr, t1);
    // Cleanup
    // ...
}
void test_checkpoint_double_take_error(void) {
    CheckpointManager mgr = setup_test_checkpoint_manager();
    Lsn lsn1, lsn2;
    // First should succeed
    assert(checkpoint_take(&mgr, TRIGGER_MANUAL, &lsn1) == CHECKPOINT_OK);
    // Force state to non-INACTIVE (simulating in-progress)
    mgr.state = CHECKPOINT_END_WRITING;
    // Second should fail
    assert(checkpoint_take(&mgr, TRIGGER_MANUAL, &lsn2) == CHECKPOINT_ERR_IN_PROGRESS);
    // Reset and try again
    mgr.state = CHECKPOINT_INACTIVE;
    assert(checkpoint_take(&mgr, TRIGGER_MANUAL, &lsn2) == CHECKPOINT_OK);
    // Cleanup
    // ...
}
```
### 8.3 Log Truncation Tests
```c
void test_truncation_safe_segment(void) {
    // Create multiple segments
    LogTruncator truncator;
    truncator_init(&truncator, "/tmp/test_trunc_wal");
    // Create segment files manually
    create_test_segment("/tmp/test_trunc_wal.000001", 0, 1000);
    create_test_segment("/tmp/test_trunc_wal.000002", 1000, 2000);
    create_test_segment("/tmp/test_trunc_wal.000003", 2000, 3000);  // Active
    truncator_scan_segments(&truncator);
    // Checkpoint at LSN 2500, min_rec_lsn = 1500
    CheckpointSnapshot snapshot = {
        .end_lsn = 2500,
        .num_active_txns = 0,  // No active transactions
        .num_dirty_pages = 1,
        .dirty_page_entries = &(CheckpointDirtyPageEntry){ .rec_lsn = 1500 }
    };
    // Truncate
    TruncationResult result;
    assert(truncator_truncate(&truncator, &snapshot, &result) == TRUNCATE_OK);
    // Segment 1 should be deleted (end_lsn=1000 < min_rec_lsn=1500)
    assert(result.segments_deleted == 1);
    assert(access("/tmp/test_trunc_wal.000001", F_OK) != 0);  // Deleted
    // Segment 2 should NOT be deleted (end_lsn=2000 >= min_rec_lsn=1500)
    assert(access("/tmp/test_trunc_wal.000002", F_OK) == 0);  // Still exists
    // Cleanup
    // ...
}
void test_truncation_blocked_by_active_txn(void) {
    LogTruncator truncator;
    truncator_init(&truncator, "/tmp/test_block_wal");
    create_test_segment("/tmp/test_block_wal.000001", 0, 1000);
    create_test_segment("/tmp/test_block_wal.000002", 1000, 2000);
    truncator_scan_segments(&truncator);
    // Checkpoint with active transaction that started in segment 1
    CheckpointSnapshot snapshot = {
        .end_lsn = 3000,
        .num_active_txns = 1,
        .txn_entries = &(CheckpointTxnEntry){
            .txn_id = 1,
            .status = TXN_ACTIVE,
            .first_lsn = 500  // In segment 1 (0-1000)
        },
        .num_dirty_pages = 1,
        .dirty_page_entries = &(CheckpointDirtyPageEntry){ .rec_lsn = 1500 }
    };
    // Truncate
    TruncationResult result;
    truncator_truncate(&truncator, &snapshot, &result);
    // Segment 1 should NOT be deleted (active txn started there)
    assert(result.segments_deleted == 0);
    assert(result.blocking_txns >= 1);
    assert(access("/tmp/test_block_wal.000001", F_OK) == 0);  // Still exists
    // Cleanup
    // ...
}
void test_truncation_after_txn_commit(void) {
    // Same setup as above, but now transaction is committed
    CheckpointSnapshot snapshot = {
        .end_lsn = 3000,
        .num_active_txns = 0,  // Transaction committed
        .num_dirty_pages = 1,
        .dirty_page_entries = &(CheckpointDirtyPageEntry){ .rec_lsn = 100 }
    };
    // Now segment 1 should be deletable
    // ...
}
```
### 8.4 Recovery Time Benchmark Test
```c
void benchmark_recovery_with_checkpoint(void) {
    const int NUM_RECORDS = 10000;
    // === Without checkpoint ===
    TransactionManager* mgr = setup_test_database("/tmp/bench_no_cp");
    for (int i = 0; i < NUM_RECORDS; i++) {
        TxnId txn = begin_txn(mgr);
        write_txn(mgr, txn, i % 100, 0, "old", 3, "new", 3);
        commit_txn(mgr, txn);
    }
    // Simulate crash and recover
    uint64_t start = get_time_us();
    crash_recovery(mgr->log, mgr->pool);
    uint64_t recovery_no_checkpoint = get_time_us() - start;
    cleanup_test_database(mgr);
    // === With checkpoint at 50% ===
    mgr = setup_test_database("/tmp/bench_with_cp");
    for (int i = 0; i < NUM_RECORDS; i++) {
        TxnId txn = begin_txn(mgr);
        write_txn(mgr, txn, i % 100, 0, "old", 3, "new", 3);
        commit_txn(mgr, txn);
        if (i == NUM_RECORDS / 2) {
            // Take checkpoint at 50%
            checkpoint_take(&mgr->checkpoint, TRIGGER_MANUAL, NULL);
        }
    }
    // Simulate crash and recover
    start = get_time_us();
    crash_recovery(mgr->log, mgr->pool);
    uint64_t recovery_with_checkpoint = get_time_us() - start;
    cleanup_test_database(mgr);
    // === Results ===
    printf("Recovery without checkpoint: %lu us\n", recovery_no_checkpoint);
    printf("Recovery with checkpoint:    %lu us\n", recovery_with_checkpoint);
    double ratio = (double)recovery_with_checkpoint / recovery_no_checkpoint;
    printf("Ratio: %.2f (must be <= 0.60)\n", ratio);
    assert(ratio <= 0.60);  // Must be at most 60% of original time
}
```
### 8.5 Long Transaction Blocking Test
```c
void test_long_transaction_blocks_truncation(void) {
    // Setup with small segments
    TransactionManager* mgr = setup_test_database_with_segment_size(
        "/tmp/test_long_txn", 10000);  // 10KB segments
    CheckpointSystem* cp_sys = checkpoint_system_create(
        "/tmp/test_long_txn", mgr->wal, &mgr->txn_table, &mgr->dpt, 0, 0);
    // Start long-running transaction
    TxnId long_txn = begin_txn(mgr);
    // Write many records (creates multiple segments)
    for (int i = 0; i < 1000; i++) {
        TxnId short_txn = begin_txn(mgr);
        write_txn(mgr, short_txn, i % 10, 0, "old", 3, "new", 3);
        commit_txn(mgr, short_txn);
    }
    // Force checkpoint
    checkpoint_system_force_checkpoint(cp_sys, NULL);
    // Check truncation - should be blocked
    TruncationResult result;
    truncator_truncate(&cp_sys->truncator, &cp_sys->manager.current_snapshot, &result);
    assert(result.segments_deleted == 0);  // Blocked by long_txn
    assert(result.blocking_txns >= 1);
    printf("Long transaction blocked truncation of %u segments\n",
           result.segments_blocked);
    // Now commit long transaction
    commit_txn(mgr, long_txn);
    // Take another checkpoint
    checkpoint_system_force_checkpoint(cp_sys, NULL);
    // Now truncation should work
    truncator_truncate(&cp_sys->truncator, &cp_sys->manager.current_snapshot, &result);
    assert(result.segments_deleted > 0);
    printf("After commit, truncated %u segments\n", result.segments_deleted);
    // Cleanup
    checkpoint_system_destroy(cp_sys);
    cleanup_test_database(mgr);
}
```
---
## 9. Performance Targets
| Operation                          | Target         | How to Measure                                  |
|------------------------------------|----------------|-------------------------------------------------|
| Checkpoint snapshot capture        | < 100 µs       | `./benchmark_checkpoint --op=snapshot`          |
| Master record write (atomic)       | < 1 ms         | `./benchmark_checkpoint --op=master_write`      |
| Complete checkpoint (1000 txns)    | < 10 ms        | `./benchmark_checkpoint --txns=1000`            |
| Checkpoint overhead (throughput)   | < 2%           | Compare TPS with/without checkpointing          |
| Recovery time reduction (50% CP)   | <= 60%         | `./benchmark_recovery_time`                     |
| Segment scan (100 segments)        | < 10 ms        | `./benchmark_truncate --op=scan`                |
| Truncation decision (per segment)  | < 10 µs        | `./benchmark_truncate --op=decision`            |
| Background thread wake overhead    | < 1 µs         | `./benchmark_checkpoint --op=wake`              |
**Benchmark Template:**
```c
void benchmark_checkpoint_snapshot(void) {
    TransactionTable txn_table;
    DirtyPageTable dpt;
    // Populate with 1000 transactions and 10000 dirty pages
    txn_table_init(&txn_table, 1000);
    dpt_init(&dpt, 10000);
    for (int i = 0; i < 1000; i++) {
        txn_table_add(&txn_table, i, TXN_ACTIVE, i * 100, i * 100 + 50);
    }
    for (int i = 0; i < 10000; i++) {
        dpt_add(&dpt, i, i * 10);
    }
    // Measure snapshot capture time
    CheckpointSnapshot snapshot;
    const int ITERATIONS = 100;
    uint64_t total_time = 0;
    for (int i = 0; i < ITERATIONS; i++) {
        uint64_t start = get_time_ns();
        capture_snapshot(&txn_table, &dpt, &snapshot);
        total_time += get_time_ns() - start;
        free(snapshot.txn_entries);
        free(snapshot.dirty_page_entries);
    }
    printf("Snapshot capture: %lu ns avg (%lu txns, %lu pages)\n",
           total_time / ITERATIONS,
           (uint64_t)txn_table.count,
           (uint64_t)dpt.count);
    // Verify under 100µs
    assert(total_time / ITERATIONS < 100000);
}
```
---
## 10. Concurrency Specification
### 10.1 Lock Ordering
```
checkpoint_manager.state_lock
    └── active_txn_table.lock
        └── active_dpt.lock
NEVER acquire in reverse order.
```
### 10.2 Lock Hold Durations
| Lock                       | Max Hold Time | When Held                              |
|----------------------------|---------------|----------------------------------------|
| state_lock                 | < 1 µs        | State check and transition             |
| active_txn_table.lock      | < 100 µs      | Snapshot copy only                     |
| active_dpt.lock            | < 100 µs      | Snapshot copy only                     |
| shutdown_lock              | < 1 ms        | During shutdown sequence               |
### 10.3 Thread Safety Guarantees
| Function                | Thread-Safe? | Notes                                        |
|-------------------------|--------------|----------------------------------------------|
| checkpoint_take         | Yes          | Protected by state_lock, only one at a time  |
| checkpoint_is_in_progress | Yes        | Reads state under lock                       |
| master_record_read      | Yes          | Read-only, no shared state                   |
| master_record_write     | Yes          | Atomic rename, no lock needed                |
| truncator_truncate      | Yes          | No shared mutable state                      |
| checkpoint_system_force_checkpoint | Yes | Signals background thread                  |
### 10.4 Background Thread Safety
- Background thread reads shared state under appropriate locks
- Shutdown uses condition variable to signal completion
- Force checkpoint signals thread via condition variable
---
## 11. State Machine: Checkpoint Lifecycle
```

![Background Checkpoint Thread Flow](./diagrams/tdd-diag-m4-10.svg)

```
**States and Transitions:**
```
States: INACTIVE -> BEGIN_WRITTEN -> END_WRITING -> SYNCING -> INACTIVE
        (any state) -> SHUTDOWN
Transitions:
  INACTIVE + checkpoint_take()     : INACTIVE -> BEGIN_WRITTEN
  BEGIN_WRITTEN + snapshot done    : BEGIN_WRITTEN -> END_WRITING
  END_WRITING + record written     : END_WRITING -> SYNCING
  SYNCING + master updated         : SYNCING -> INACTIVE
  (any) + checkpoint_take() error  : (any) -> INACTIVE (with error)
  (any) + shutdown_requested       : (any) -> SHUTDOWN
Per-Checkpoint Flow:
  1. Write CHECKPOINT_BEGIN
  2. Capture txn_table snapshot (lock < 100µs)
  3. Capture dpt snapshot (lock < 100µs)
  4. Write CHECKPOINT_END
  5. Sync log
  6. Update master record
  7. Complete
```
---
## 12. Diagrams Reference
```

![Recovery Time Benchmark Setup](./diagrams/tdd-diag-m4-11.svg)

```
```
{{DIAGRAM:tdd-diag-m4-12}}
```
```

![Truncation Two-Condition Check](./diagrams/tdd-diag-m4-13.svg)

```
```

![Checkpoint System Component Architecture](./diagrams/tdd-diag-m4-14.svg)

```
```

![Recovery with Checkpoint Bootstrap](./diagrams/tdd-diag-m4-15.svg)

```
---
[[CRITERIA_JSON: {"module_id": "wal-impl-m4", "criteria": ["Fuzzy checkpoint writes CHECKPOINT_BEGIN record, captures atomic snapshots of transaction table and dirty page table via brief mutex lock (<100µs), writes CHECKPOINT_END record containing serialized snapshot data", "Concurrent transactions continue executing during checkpoint without blocking; transactions may commit, abort, or start between CHECKPOINT_BEGIN and CHECKPOINT_END", "MasterRecord struct is exactly 64 bytes with fields: magic (uint32_t, 0x57414C4D), version (uint32_t), last_checkpoint_lsn (uint64_t), checkpoint_count (uint64_t), create_time (uint64_t), update_time (uint64_t), crc32 (uint32_t), padding (20 bytes)", "Master record stored in separate file (wal.master) and updated atomically via temp-file-rename pattern: write to .tmp, fsync, rename to final path", "Master record update occurs AFTER CHECKPOINT_END is fsynced to disk; crash before master update leaves previous checkpoint valid and recoverable", "master_record_read validates magic, version, and CRC32; returns MASTER_ERR_CRC_MISMATCH or MASTER_ERR_INVALID_MAGIC on corruption", "Recovery reads master record first via recovery_read_checkpoint_lsn(); if valid, starts Analysis phase from checkpoint_lsn; if corrupted or missing, falls back to log start (LSN 0)", "CheckpointScheduler triggers checkpoints based on time_interval_ms (default 30000) or record_count_interval (default 10000); checkpoint_scheduler_should_checkpoint() returns TRIGGER_TIME, TRIGGER_RECORD_COUNT, or TRIGGER_NONE", "LogTruncator scans log directory to build SegmentMetadata array with segment_id, file_path, start_lsn, end_lsn, file_size, is_active", "Truncation safety rule 1: segment.end_lsn < checkpoint's minimum rec_lsn (segment not needed for redo)", "Truncation safety rule 2: no active transaction's first_lsn falls within segment range (segment not needed for undo)", "truncator_can_delete_segment() returns true only if both safety rules are satisfied; active segments are never deleted", "TruncationResult tracks segments_deleted, bytes_freed, segments_blocked, blocking_txns", "Long-running transactions prevent log truncation; truncation only succeeds after all transactions active at checkpoint time have completed", "CheckpointSystem runs background thread that wakes every check_interval_ms, checks scheduler, takes checkpoint if triggered, then attempts truncation", "Recovery time benchmark: with checkpoint at 50% log position, recovery time is <= 60% of recovery without checkpoint (measured with 10000 transactions)", "Checkpoint snapshot capture holds txn_table.lock and dpt.lock each for <100µs; total checkpoint time <10ms for 1000 transactions", "Checkpoint overhead during normal operation is <2% of transaction throughput", "checkpoint_take() returns CHECKPOINT_ERR_IN_PROGRESS if checkpoint already in progress (state != CHECKPOINT_INACTIVE)", "Unit tests verify: master record atomic update (temp file + rename), truncation blocked by long transaction, recovery with corrupted master falls back to log start"]}]
<!-- END_TDD_MOD -->


# Project Structure: Write-Ahead Log Implementation
## Directory Tree
```
wal/
├── include/                     # Public header files
│   ├── wal_types.h              # [M1] Core types, constants, error codes
│   ├── wal_record.h             # [M1] Log record structures, serialization API
│   ├── wal_transaction.h        # [M1] Transaction manager, transaction table
│   ├── wal_writer.h             # [M2] WalWriter, WriteBuffer, GroupCommitManager
│   ├── wal_segment.h            # [M2] LogSegmentManager for rotation
│   ├── wal_recovery.h           # [M3] RecoveryManager, recovery phases API
│   ├── wal_buffer_pool.h        # [M3] BufferPool abstraction for page access
│   ├── wal_undo.h               # [M3] UndoWork, priority queue, CLR generation
│   ├── wal_checkpoint.h         # [M4] CheckpointManager, CheckpointScheduler APIs
│   ├── wal_master_record.h      # [M4] MasterRecord structure, atomic update
│   └── wal_truncate.h           # [M4] LogTruncator, segment safety checks
├── src/                         # Implementation files
│   ├── wal_record.c             # [M1] Serialization/deserialization implementation
│   ├── wal_transaction.c        # [M1] Transaction manager implementation
│   ├── wal_writer.c             # [M2] Append, sync, group commit implementation
│   ├── wal_segment.c            # [M2] Segment creation, rotation, metadata
│   ├── wal_recovery.c           # [M3] Analysis, Redo, Undo phase implementations
│   ├── wal_buffer_pool.c        # [M3] Simple buffer pool for recovery testing
│   ├── wal_undo.c               # [M3] Undo queue management, CLR writer
│   ├── wal_checkpoint.c         # [M4] Fuzzy checkpoint implementation
│   ├── wal_master_record.c      # [M4] Master record read/write with atomicity
│   └── wal_truncate.c           # [M4] Log segment truncation logic
test/                            # Test files
├── test_record_serialization.c  # [M1] Unit tests for record formats
├── test_transaction_api.c       # [M1] Integration tests for transaction lifecycle
├── test_append_correctness.c    # [M2] Sequential append, LSN allocation
├── test_group_commit.c          # [M2] Throughput benchmark, leader/follower
├── test_concurrent_writers.c    # [M2] Multi-threaded append verification
├── test_segment_rotation.c      # [M2] Rotation at threshold, segment naming
├── test_torn_write_detection.c  # [M2] CRC truncation on corrupt final record
├── test_analysis_phase.c        # [M3] Transaction table and DPT reconstruction
├── test_redo_phase.c            # [M3] Replay with pageLSN idempotency
├── test_undo_phase.c            # [M3] Priority queue, prev_lsn traversal
├── test_clr_generation.c        # [M3] CLR correctness, undo_next_lsn
├── test_recovery_integration.c  # [M3] Three-transaction scenario
├── test_idempotency.c           # [M3] Recovery N times = same state
├── test_crash_during_undo.c     # [M3] CLR-based recovery resumption
├── test_checkpoint_fuzzy.c      # [M4] Concurrent transactions during checkpoint
├── test_master_record.c         # [M4] Atomic update, corruption handling
├── test_log_truncation.c        # [M4] Safe vs unsafe segment deletion
├── test_checkpoint_recovery.c   # [M4] Recovery time with/without checkpoint
└── test_long_txn_blocking.c     # [M4] Long transaction prevents truncation
Makefile                         # Build system
README.md                        # Project overview
```
## Creation Order
### 1. **Project Setup** (30 min)
   - Create directory structure: `mkdir -p wal/include wal/src test`
   - Create `Makefile` with compilation rules
   - Create `README.md` with project overview
### 2. **Milestone 1: Core Types and Records** (3-4 hours)
   - `include/wal_types.h` — Core type definitions, constants, error codes
   - `include/wal_record.h` — Record structures and serialization API
   - `src/wal_record.c` — Serialization/deserialization implementation
   - `test/test_record_serialization.c` — Unit tests for record formats
### 3. **Milestone 1: Transaction API** (2-3 hours)
   - `include/wal_transaction.h` — Transaction manager and transaction table
   - `src/wal_transaction.c` — Transaction manager implementation
   - `test/test_transaction_api.c` — Integration tests for transaction lifecycle
### 4. **Milestone 2: Write Buffer and Segments** (2-3 hours)
   - `include/wal_writer.h` — WalWriter, WriteBuffer, GroupCommitManager APIs
   - `include/wal_segment.h` — LogSegmentManager for rotation
   - `src/wal_segment.c` — Segment creation, rotation, metadata
### 5. **Milestone 2: WAL Writer and Group Commit** (3-4 hours)
   - `src/wal_writer.c` — Append, sync, group commit implementation
   - `test/test_append_correctness.c` — Sequential append tests
   - `test/test_group_commit.c` — Throughput benchmark
   - `test/test_concurrent_writers.c` — Multi-threaded tests
   - `test/test_segment_rotation.c` — Rotation tests
   - `test/test_torn_write_detection.c` — CRC truncation tests
### 6. **Milestone 3: Recovery Data Structures** (2 hours)
   - `include/wal_recovery.h` — RecoveryManager, recovery phases API
   - `include/wal_buffer_pool.h` — BufferPool abstraction
   - `include/wal_undo.h` — UndoWork, priority queue, CLR generation
   - `src/wal_buffer_pool.c` — Buffer pool implementation
### 7. **Milestone 3: Recovery Phases** (5-6 hours)
   - `src/wal_recovery.c` — Analysis, Redo, Undo implementations
   - `src/wal_undo.c` — Undo queue management, CLR writer
   - `test/test_analysis_phase.c` — Analysis phase tests
   - `test/test_redo_phase.c` — Redo phase tests
   - `test/test_undo_phase.c` — Undo phase tests
### 8. **Milestone 3: Recovery Integration** (2-3 hours)
   - `test/test_clr_generation.c` — CLR correctness tests
   - `test/test_recovery_integration.c` — Three-transaction scenario
   - `test/test_idempotency.c` — Idempotency verification
   - `test/test_crash_during_undo.c` — Crash resumption tests
### 9. **Milestone 4: Checkpointing** (3-4 hours)
   - `include/wal_checkpoint.h` — CheckpointManager, CheckpointScheduler APIs
   - `include/wal_master_record.h` — MasterRecord structure
   - `src/wal_master_record.c` — Master record atomic read/write
   - `src/wal_checkpoint.c` — Fuzzy checkpoint implementation
   - `test/test_checkpoint_fuzzy.c` — Concurrent checkpoint tests
   - `test/test_master_record.c` — Atomic update tests
### 10. **Milestone 4: Log Truncation** (2-3 hours)
   - `include/wal_truncate.h` — LogTruncator, segment safety checks
   - `src/wal_truncate.c` — Log segment truncation logic
   - `test/test_log_truncation.c` — Safe vs unsafe deletion tests
   - `test/test_checkpoint_recovery.c` — Recovery time benchmark
   - `test/test_long_txn_blocking.c` — Long transaction blocking tests
## File Count Summary
| Category | Count |
|----------|-------|
| Header files (include/) | 11 |
| Source files (src/) | 10 |
| Test files (test/) | 19 |
| Build/Docs | 2 |
| **Total files** | **42** |
| **Directories** | **3** |
## Estimated Lines of Code
| Module | Estimated LOC |
|--------|---------------|
| M1: Log Record Format & Transaction API | ~1,500 |
| M2: Log Writer with Group Commit | ~2,000 |
| M3: ARIES Crash Recovery with CLRs | ~2,500 |
| M4: Fuzzy Checkpointing & Log Truncation | ~1,800 |
| Test code | ~3,000 |
| **Total** | **~10,800** |
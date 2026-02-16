# AUDIT & FIX: collaborative-editor

## CRITIQUE
- CRITICAL: The audit correctly identifies that implementing both CRDTs (M1) and OT (M3) as sequential milestones in one editor is architecturally conflicting. In practice, you choose ONE conflict resolution strategy. Having students build both creates confusion about which one actually powers the editor.
- The project should either: (a) focus on CRDTs only (the modern approach used by Yjs, Automerge), (b) focus on OT only (the classic approach used by Google Docs), or (c) clearly frame M3 as a comparative study/alternative implementation with a clear explanation of when to use which.
- M1 AC 'Implement RGA (Replicated Growable Array) for ordered text' is a deliverable description, not a measurable AC. How do you verify correctness? Need convergence tests.
- M1 AC 'Ensure convergence across all replicas after all operations are applied' — how is this measured? Need a specific test scenario (e.g., 3 replicas, each applies 100 concurrent operations, final document state is identical).
- M2 is cursor presence which logically depends on the sync mechanism, but it's unclear whether cursors use the CRDT or a separate channel.
- M3 pitfall says 'T(T(a,b),c) = T(a,T(b,c))' — this is NOT the correct OT property. The transformation properties are TP1: T(op1, op2) ∘ T(op2, op1) gives same document, not associativity. This is a technical inaccuracy.
- M4 (Undo/Redo) is the right final milestone but extremely hard. The AC doesn't specify how undo interacts with the chosen conflict resolution strategy.
- Estimated hours: all milestones at 12.5 hours each = 50 total, but M1 and M3 alone could each take 20+ hours for a correct implementation.
- No milestone for network synchronization layer (WebSocket server, message ordering, reconnection handling). The CRDT/OT is only the data structure; you need a sync protocol.
- No mention of document persistence/loading. How is the document stored and loaded on reconnection?
- The resources are good but the project structure needs fundamental rethinking.

## FIXED YAML
```yaml
id: collaborative-editor
name: Collaborative Editor
description: >-
  Real-time collaborative text editor using CRDTs for conflict-free
  concurrent editing with cursor presence, persistence, and undo/redo.
difficulty: advanced
estimated_hours: 60
essence: >-
  Conflict-free concurrent editing through a sequence CRDT (RGA or YATA)
  providing mathematically guaranteed convergence, real-time operation
  synchronization over WebSocket with causal ordering, cursor presence
  broadcasting, and collaborative undo/redo via inverse operations.
why_important: >-
  Collaborative editing is used in Google Docs, Notion, and Figma.
  Understanding CRDTs teaches distributed systems convergence guarantees,
  causal ordering, and real-time synchronization — skills applicable to any
  real-time collaborative application.
learning_outcomes:
  - Implement a sequence CRDT (RGA) for conflict-free text insertion and deletion
  - Build a WebSocket synchronization layer with causal operation ordering
  - Handle concurrent edits from multiple users with guaranteed convergence
  - Implement cursor presence synchronization across connected editors
  - Design collaborative undo/redo using inverse operations
  - Persist document state and support offline reconnection
skills:
  - CRDT Implementation
  - WebSocket Real-time Sync
  - Causal Ordering
  - Concurrent State Management
  - Conflict Resolution Algorithms
  - Cursor Presence Protocols
tags:
  - advanced
  - algorithms
  - concurrency
  - crdt
  - distributed-systems
  - real-time
  - sync
architecture_doc: architecture-docs/collaborative-editor/index.md
languages:
  recommended:
    - JavaScript
    - TypeScript
  also_possible:
    - Go
    - Rust
resources:
  - name: Yjs CRDT Documentation
    url: https://docs.yjs.dev/
    type: documentation
  - name: CRDT Papers Collection
    url: https://crdt.tech/papers.html
    type: paper
  - name: A Comprehensive Study of CRDTs
    url: https://pages.lip6.fr/Marc.Shapiro/papers/RR-7687.pdf
    type: paper
  - name: "RGA: Replicated Growable Array Paper"
    url: https://hal.inria.fr/inria-00555588/document
    type: paper
prerequisites:
  - type: project
    id: chat-app
    name: Real-time Chat
  - type: skill
    name: WebSocket protocol
  - type: skill
    name: Tree and linked list data structures
milestones:
  - id: collaborative-editor-m1
    name: Sequence CRDT Implementation
    description: >-
      Implement a sequence CRDT (RGA — Replicated Growable Array) for
      conflict-free text insertion and deletion with guaranteed convergence.
    estimated_hours: 15
    concepts:
      - "RGA: each character has a unique ID (site_id, logical_clock) determining total order"
      - "Tombstone deletion: deleted characters are marked, not removed, to maintain ID stability"
      - "Causal ordering: operations are applied only after all causally preceding operations"
      - "Convergence: all replicas applying the same set of operations reach identical state regardless of order"
      - "Position identifiers: unique IDs that remain valid even as concurrent inserts occur at the same position"
    skills:
      - CRDT algorithms
      - Logical clock implementation
      - Linked list / tree data structures
      - Convergence testing
    acceptance_criteria:
      - "Insert operation creates a character with a unique ID (site_id, lamport_timestamp) and positions it after a specified predecessor ID in the sequence"
      - "Delete operation marks a character as a tombstone by its unique ID; the character is hidden from the visible document but retained in the internal structure"
      - "Concurrent inserts at the same position (same predecessor) are deterministically ordered using the total order on IDs (compare lamport_timestamp, break ties with site_id)"
      - "Convergence test: 3 independent replicas each apply 50 random concurrent insert and delete operations; after all operations are exchanged and applied, all 3 replicas produce the identical visible document string"
      - "Operations are commutative: applying operations in any order produces the same final state (verified by randomizing application order in tests)"
      - "Document with 10,000 characters supports insert and delete operations in under 10ms each (O(log n) or better lookup)"
    pitfalls:
      - Lamport timestamp collision when two sites have identical clocks — must use site_id as tiebreaker
      - Tombstones growing unboundedly consuming memory — need garbage collection strategy (addressed later)
      - O(n) linear scan for every operation on large documents — use an indexed structure (tree, skip list)
      - Not enforcing causal ordering causing operations to reference non-existent predecessor IDs
    deliverables:
      - RGA data structure with insert and delete operations
      - Unique ID generation using (site_id, lamport_timestamp) pairs
      - Lamport clock that increments on local operations and merges on remote operations
      - Tombstone-based deletion preserving ID stability
      - Deterministic ordering for concurrent inserts at same position
      - Convergence test suite with multiple replicas and randomized operations
      - Performance-optimized internal structure (tree or skip list) for O(log n) operations

  - id: collaborative-editor-m2
    name: Sync Layer & Network Protocol
    description: >-
      Build the WebSocket-based synchronization layer that broadcasts CRDT
      operations between connected editors with causal ordering, reconnection
      handling, and document persistence.
    estimated_hours: 15
    concepts:
      - "Operation-based sync: broadcast each insert/delete operation to all peers via server relay"
      - "Causal ordering with vector clocks: only apply an operation when all its causal predecessors have been applied"
      - "State sync on reconnection: send missing operations or full document state to reconnecting clients"
      - "Document persistence: periodically snapshot the CRDT state for durability"
    skills:
      - WebSocket server implementation
      - Vector clock causal ordering
      - State synchronization protocols
      - Document persistence and snapshots
    acceptance_criteria:
      - "WebSocket server relays CRDT operations from each client to all other connected clients for the same document"
      - "Operations include a vector clock; a receiving client buffers operations whose causal predecessors have not yet been applied, and applies them in causal order"
      - "On client reconnection, the server sends all operations the client missed (determined by comparing vector clocks) or a full state snapshot if the gap is too large"
      - "Document state is persisted to storage (database or file) at configurable intervals (default every 30 seconds) and on last client disconnect"
      - "A client that disconnects, makes offline edits, and reconnects successfully merges its local operations with the server state without data loss"
      - "Server handles 10 simultaneous editors on the same document with operation broadcast latency under 100ms"
    pitfalls:
      - Not buffering out-of-order operations causes CRDT state corruption
      - Sending full document state on every reconnection instead of just missing operations (wasteful)
      - Not persisting document state — server restart loses all work
      - WebSocket message ordering is guaranteed per-connection but operations from different clients can interleave
      - Missing heartbeat/keep-alive causing silent disconnections
    deliverables:
      - WebSocket server with document room management (join, leave, broadcast)
      - Operation broadcast protocol relaying CRDT ops to all peers
      - Vector clock-based causal ordering buffer on client side
      - Reconnection sync protocol (missing ops or full state snapshot)
      - Document persistence with periodic snapshots
      - Offline edit queue that replays local operations on reconnection
      - Connection health monitoring with heartbeat/ping-pong

  - id: collaborative-editor-m3
    name: Cursor Presence & Awareness
    description: >-
      Synchronize cursor positions and text selections across all connected
      editors with colored indicators and user presence awareness.
    estimated_hours: 12
    concepts:
      - "Cursor position as CRDT ID reference: cursor points to a character ID, not a numeric index"
      - "Selection range as pair of CRDT IDs (anchor, focus)"
      - "Presence protocol: lightweight high-frequency updates separate from document operations"
      - "Cursor transformation: when remote operations insert/delete near cursor, position adjusts automatically"
    skills:
      - WebSocket presence channel
      - UI state synchronization
      - Cursor position mapping between CRDT IDs and editor positions
      - Client-side rendering of remote cursors
    acceptance_criteria:
      - "Each connected user's cursor position is broadcast to all peers and displayed as a colored caret with username label in the editor; updates arrive within 200ms"
      - "Text selections (highlight ranges) are broadcast and displayed as colored overlays with the selecting user's color"
      - "When a remote operation inserts or deletes text before a user's cursor, the cursor position adjusts correctly (maps CRDT ID to current visual position)"
      - "Users who disconnect have their cursor removed from all editors within 5 seconds"
      - "Each user is assigned a unique color from a palette; colors are consistent across all editors for the same user"
      - "Cursor updates are throttled (max 10 updates/second per user) to avoid flooding the WebSocket channel"
    pitfalls:
      - Using numeric index for cursor position instead of CRDT character ID — breaks on concurrent edits
      - Cursor flicker from broadcasting position on every keystroke without throttling
      - Selection becoming inverted (focus < anchor) after remote insert within selection range
      - Not removing disconnected users' cursors causing ghost cursors
      - High-frequency cursor updates consuming excessive bandwidth
    deliverables:
      - Cursor position sharing protocol using CRDT character IDs
      - Text selection range broadcasting (anchor and focus as CRDT IDs)
      - Remote cursor rendering overlay with username label and assigned color
      - Cursor position transformation when document changes shift positions
      - User presence tracking (join, leave, idle detection)
      - Cursor update throttling (rate-limited broadcasting)

  - id: collaborative-editor-m4
    name: Collaborative Undo/Redo
    description: >-
      Implement undo/redo that correctly handles concurrent edits by using
      inverse operations scoped to the local user's action history.
    estimated_hours: 18
    concepts:
      - "Selective undo: each user undoes only their own operations, not everyone's"
      - "Inverse operations: undo of insert is delete of the same character; undo of delete is re-insert"
      - "Undo in CRDT context: inverse operations are themselves CRDT operations that are broadcast and merged"
      - "Operation grouping: consecutive keystrokes are grouped into a single undo unit (word or phrase)"
      - "Redo as undo of undo: maintain a redo stack cleared on new operations"
    skills:
      - Undo/redo stack management
      - Inverse operation computation
      - Operation grouping heuristics
      - State reconciliation
    acceptance_criteria:
      - "Undo reverses only the current user's most recent operation group; other users' operations are unaffected"
      - "Undo of an insert operation generates a delete of the same character(s); undo of a delete re-inserts the deleted character(s) at their original position (using the tombstoned CRDT IDs)"
      - "Undo/redo operations are themselves CRDT operations broadcast to all peers; all replicas converge after undo"
      - "Consecutive character insertions within 500ms are grouped into a single undo unit; pressing undo reverts the entire group"
      - "Redo reapplies the most recently undone operation group; the redo stack is cleared when the user performs a new edit"
      - "Undo works correctly even when concurrent remote operations have interleaved with the user's operations (the undone characters are correctly identified regardless of document mutations)"
      - "Undo stack survives reconnection: after disconnect and reconnect, the user can still undo their pre-disconnect operations"
    pitfalls:
      - Undoing a delete requires storing the deleted content (tombstones help here in CRDTs)
      - Undo stack invalidation on reconnection if not stored with CRDT operation IDs
      - Not grouping rapid keystrokes forces user to undo one character at a time
      - Redo stack not cleared on new edit causing confusing redo behavior
      - Inverse of a grouped operation must be applied atomically (all or nothing)
    deliverables:
      - Per-user undo stack tracking operation groups with CRDT operation references
      - Inverse operation generator (insert->delete, delete->re-insert)
      - Operation grouping by time proximity (configurable threshold)
      - Redo stack with clear-on-new-edit semantics
      - Undo/redo operations broadcast as regular CRDT ops for convergence
      - Undo correctness test with concurrent edits from multiple users

```
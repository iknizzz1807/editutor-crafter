# AUDIT & FIX: chat-app

## CRITIQUE
- **Audit Finding Confirmed (Auth Ordering - CRITICAL):** Authentication is in M4, but the WebSocket server accepts connections from M1. This means M1-M3 operate with completely unauthenticated connections, which is a fundamental security flaw. In production, an unauthenticated WebSocket endpoint is a trivial DoS vector (each connection holds server resources). Auth must come before or immediately after the basic server is set up.
- **Audit Finding Confirmed (Redundant Typing Indicators):** Typing indicators appear as an AC in both M2 ('Typing indicator is broadcast when user starts typing and cleared after timeout') and M4 ('Typing indicator events broadcast to room members showing who is currently typing'). This is redundant and confusing—which milestone actually owns the typing indicator feature?
- **M2/M3 Ordering Issue:** M2 (Message Broadcasting) mentions broadcasting to all connected clients, but M3 (Chat Rooms) introduces room-based messaging. M2's AC says 'Join notification is broadcast to room members when a new user enters the room'—but rooms don't exist until M3. This is a logical ordering error.
- **M3 Premature History Loading:** M3 AC says 'Joining a room loads configurable number of recent messages from history' but message persistence doesn't exist until M4. You cannot load history from a database that hasn't been implemented yet.
- **M4 Scope Overload:** M4 tries to do authentication, message persistence, typing indicators, AND presence tracking in one milestone. This is too much for a single milestone and mixes orthogonal concerns.
- **Missing Reconnection Logic:** The learning outcomes mention 'reconnection logic' but no milestone has an AC for handling client disconnects and reconnects with message catch-up.
- **Missing Message Ordering:** Learning outcomes mention 'message ordering guarantees' but no AC addresses this.
- **Estimated Hours Mismatch:** M1 is 2-3 hours but M4 is 6-8 hours. The total (15-20h) is well below the project estimate of 25-35h. There's a gap.

## FIXED YAML
```yaml
id: chat-app
name: Real-time Chat Application
description: >-
  Build a real-time chat application using WebSockets with authentication,
  room-based messaging, message persistence, and presence tracking.
difficulty: intermediate
estimated_hours: "30-40"
essence: >-
  Persistent full-duplex TCP connections enabling bidirectional message
  streaming between clients and servers, with event-driven broadcasting
  patterns, connection lifecycle management, and distributed state
  synchronization challenges across unreliable network topologies.
why_important: >-
  Real-time communication powers critical modern applications from
  collaborative tools to live trading platforms, teaching you event-driven
  architecture, stateful connection management, and the complexities of
  synchronizing state across unreliable networks—skills directly applicable
  to any system requiring low-latency updates.
learning_outcomes:
  - Implement WebSocket handshake and maintain persistent bidirectional connections
  - Build authentication for WebSocket connections using token validation during the upgrade handshake
  - Design event-driven message broadcasting with room-based multicasting
  - Implement message persistence with database storage and paginated history retrieval
  - Handle connection lifecycle including reconnection logic and message catch-up
  - Build presence tracking to show online/offline status of users
  - Secure WebSocket connections with origin validation and message rate limiting
  - Implement message ordering guarantees using server-assigned sequence numbers
skills:
  - WebSocket Protocol
  - Event-driven Architecture
  - Stateful Connection Management
  - Real-time Broadcasting
  - Token-based Authentication
  - Message Persistence
  - Connection Lifecycle Handling
  - Network Security
tags:
  - intermediate
  - javascript
  - presence
  - real-time
  - typescript
  - websockets
architecture_doc: architecture-docs/chat-app/index.md
languages:
  recommended:
    - JavaScript
    - TypeScript
  also_possible:
    - Go
    - Python
    - Rust
resources:
  - name: Socket.io Chat Tutorial
    url: https://socket.io/get-started/chat
    type: tutorial
  - name: WebSocket API - MDN
    url: https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API
    type: documentation
  - name: "RFC 6455 - The WebSocket Protocol"
    url: https://datatracker.ietf.org/doc/html/rfc6455
    type: documentation
prerequisites:
  - type: skill
    name: JavaScript/Node.js
  - type: skill
    name: Basic HTML/CSS
  - type: skill
    name: Understanding of HTTP
  - type: skill
    name: Database basics
milestones:
  - id: chat-app-m1
    name: WebSocket Server & Authentication
    description: >-
      Set up a WebSocket server that authenticates connections during the
      upgrade handshake and manages connection lifecycle.
    acceptance_criteria:
      - "Server accepts WebSocket upgrade requests on a designated path and establishes persistent connections"
      - "Authentication is enforced during the HTTP upgrade handshake; connections without a valid JWT or session token are rejected with 401 before upgrade completes"
      - "Authenticated connections are tracked with a unique connection ID and the associated user identity"
      - "Connection and disconnection events are logged with user identity and timestamp"
      - "Server handles malformed upgrade requests and invalid messages without crashing or dropping other clients"
      - "Heartbeat ping/pong frames are sent at a configurable interval; connections that miss pong within the timeout are terminated"
    pitfalls:
      - "Accepting WebSocket connections before authentication exposes the server to trivial resource exhaustion DoS attacks"
      - "Memory leaks from not cleaning up closed connections in the connection tracking map"
      - "Not implementing heartbeat/ping-pong allows dead connections to consume server resources indefinitely"
      - "Not validating the Origin header allows cross-site WebSocket hijacking"
    concepts:
      - WebSocket protocol upgrade handshake
      - Token validation during HTTP upgrade
      - Connection lifecycle state machine
      - Heartbeat mechanisms for liveness detection
    skills:
      - WebSocket Server Configuration
      - Authentication During Upgrade
      - Connection Lifecycle Management
      - Event-Driven Server Architecture
    deliverables:
      - WebSocket server with HTTP upgrade handling
      - Auth middleware rejecting unauthenticated upgrade requests
      - Connection registry tracking active authenticated sessions
      - Ping/pong heartbeat mechanism for dead connection detection
    estimated_hours: "6-8"

  - id: chat-app-m2
    name: Room-Based Messaging & Broadcasting
    description: >-
      Implement chat rooms where users can join, leave, and send messages
      that are broadcast only to room members.
    acceptance_criteria:
      - "Users can create new rooms and join existing rooms by sending a join event with the room name"
      - "Messages sent to a room are delivered only to members currently in that room, excluding the sender"
      - "Messages include sender username, text content, server-assigned UTC timestamp, and a monotonic sequence number per room"
      - "Join and leave notifications are broadcast to room members when a user enters or exits a room"
      - "Room listing returns all available rooms with current member counts"
      - "Users are automatically removed from all rooms on disconnect"
      - "Room names are validated and sanitized; invalid names are rejected"
    pitfalls:
      - "Not checking WebSocket readyState before sending causes errors on closed connections"
      - "Invalid JSON messages crashing the server instead of being rejected gracefully"
      - "Not removing users from rooms on disconnect causes phantom users in room member lists"
      - "Empty rooms accumulating indefinitely; implement cleanup for rooms with no members after a timeout"
      - "Missing message validation allowing empty or oversized messages"
    concepts:
      - Room-based pub/sub messaging
      - JSON message protocol design
      - Fan-out broadcasting pattern
      - Message ordering with sequence numbers
    skills:
      - Multi-Room Architecture Design
      - JSON Protocol Design and Validation
      - Message Broadcasting to Subsets
      - Server-Side Message Processing
    deliverables:
      - Room creation, join, and leave operations
      - Room-scoped message broadcasting
      - Join/leave notification events
      - Room listing with member counts
      - Message format with sender, content, timestamp, and sequence number
    estimated_hours: "5-7"

  - id: chat-app-m3
    name: Message Persistence & History
    description: >-
      Store messages in a database and provide paginated history retrieval
      when users join a room or scroll back.
    acceptance_criteria:
      - "All messages are persisted to the database with sender_id, room_id, content, timestamp, and sequence_number fields"
      - "When a user joins a room, the last N messages (configurable, default 50) are loaded from the database and sent to the client"
      - "Older messages are retrievable via cursor-based pagination using the sequence number as the cursor"
      - "Message retrieval queries are indexed and complete within 50ms for rooms with up to 100k messages"
      - "Database schema includes indexes on (room_id, sequence_number) for efficient history queries"
    pitfalls:
      - "Loading entire message history on join causes memory spikes and slow joins for active rooms"
      - "Using offset-based pagination for message history is slow for deep pages; use cursor-based pagination"
      - "Not indexing room_id + timestamp/sequence causes full table scans on history queries"
      - "Race conditions between message persistence and history loading can cause missed messages"
    concepts:
      - Message persistence patterns
      - Cursor-based pagination
      - Database indexing for time-series data
      - Write-ahead vs write-behind persistence
    skills:
      - Database Integration for Chat History
      - Pagination for Large Datasets
      - Query Optimization with Indexes
      - Data Consistency Patterns
    deliverables:
      - Message database schema with indexes on room_id and sequence_number
      - Message persistence writing each broadcast message to the database
      - History loading endpoint returning recent messages on room join
      - Cursor-based pagination for scrolling back through older messages
    estimated_hours: "5-7"

  - id: chat-app-m4
    name: Presence, Typing Indicators & Reconnection
    description: >-
      Implement user presence tracking, typing indicators, and client
      reconnection with message catch-up.
    acceptance_criteria:
      - "Online/offline presence status is broadcast to room members when an authenticated user connects or disconnects"
      - "Presence list for a room returns all currently online members with their status"
      - "Typing indicator is broadcast when a user starts typing; it auto-clears after a configurable timeout (e.g., 3 seconds) of inactivity"
      - "Typing indicators are rate-limited to at most one event per second per user to prevent spam"
      - "Client reconnection after a dropped connection resumes from the last seen sequence number, receiving missed messages"
      - "Reconnection uses exponential backoff with jitter to avoid thundering herd on server recovery"
    pitfalls:
      - "Presence race conditions: user disconnect and reconnect in quick succession can show incorrect status"
      - "Typing indicator spam if not rate-limited; clients can flood the server with typing events"
      - "Thundering herd problem when many clients reconnect simultaneously after a server restart"
      - "Message catch-up loading too many messages; cap the catch-up window and direct client to history API for older messages"
    concepts:
      - Presence tracking with last-seen timestamps
      - Debouncing and rate limiting for typing events
      - Reconnection with exponential backoff and jitter
      - Message catch-up using sequence numbers
    skills:
      - User Presence Tracking
      - Client Reconnection Patterns
      - Rate Limiting for Events
      - State Synchronization
    deliverables:
      - Presence system broadcasting online/offline status changes
      - Typing indicator with debounce timeout and rate limiting
      - Client reconnection logic with exponential backoff
      - Message catch-up delivering missed messages since last sequence number
    estimated_hours: "7-9"
```
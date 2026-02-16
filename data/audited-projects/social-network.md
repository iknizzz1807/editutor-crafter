# AUDIT & FIX: social-network

## CRITIQUE
- M1 AC 'User profile with bio, avatar, links' is not a measurable criterion — it's a feature description. Should specify validation rules, response format, etc.
- M1 has no AC for preventing self-follows, yet lists it as a pitfall. Pitfalls should be things students might miss, but the AC should catch it if it's important.
- M2 AC 'Home feed (posts from followed users)' is an incomplete sentence, not a measurable criterion.
- M2 describes fan-out-on-write but has no AC measuring that fan-out actually works (e.g., a new post appears in all followers' feeds within X seconds). The fan-out mechanism is in deliverables but untested by AC.
- M2 mentions the celebrity problem as a pitfall but the learning outcomes promise 'hybrid optimization for celebrity accounts' — there's no milestone or AC that addresses this.
- M3 AC 'Like counts update in real-time' — this is vague. Real-time how? WebSocket push? Polling? What latency bound?
- M3 mentions 'Share and repost' in AC but it's not in deliverables.
- M4 AC 'Real-time notification delivery pushes events within two seconds' — good, measurable. But no AC for batching similar notifications (e.g., '5 people liked your post') which is listed as a pitfall.
- M5 AC for trending/recommendations is underspecified — no measurable criteria for ranking algorithm quality.
- M6 AC 'Redis caching for feeds reduces database read load by 80 percent' — how is this measured? Needs to specify the benchmark methodology.
- M6 mentions 'Database sharding scheme partitioning data by user ID range' in deliverables but no AC tests it. Sharding is a massive architectural decision with no verification criteria.
- The estimated hours are ranges (e.g., '8-10') which is fine but inconsistent formatting across the project.
- No security considerations anywhere — no AC for authentication, authorization, input sanitization, or rate limiting until M6.
- M1 delivers 'Profile privacy settings' but no AC tests privacy enforcement.

## FIXED YAML
```yaml
id: social-network
name: Social Network
description: >-
  Social network with user profiles, follow graph, activity feed, interactions,
  notifications, search, and performance optimization.
difficulty: advanced
estimated_hours: 70
essence: >-
  Graph-based relationship modeling combined with fan-out write/read hybrid
  architectures for feed generation, real-time bi-directional communication
  through WebSocket connections, and distributed cache coordination to handle
  concurrent social interactions at scale.
why_important: >-
  Building this project teaches the core infrastructure patterns behind
  platforms like Twitter and Instagram — specifically how to architect systems
  that handle complex many-to-many relationships, optimize read-heavy workloads
  through strategic caching, and deliver real-time updates without database
  bottlenecks.
learning_outcomes:
  - Implement fan-out-on-write feed architecture with hybrid optimization for high-follower accounts
  - Design graph-based data models for follower relationships using adjacency lists
  - Build WebSocket-based notification systems with Redis pub/sub for multi-server coordination
  - Implement distributed caching strategies using Redis for timeline storage and cache invalidation
  - Build rate limiting and priority queues to prevent spam and ensure critical notifications deliver first
  - Optimize N+1 query problems in social graphs through denormalization and batch loading
skills:
  - Fan-out Architecture
  - Graph Data Modeling
  - WebSocket Real-time Communication
  - Distributed Caching
  - Message Queue Systems
  - Redis Pub/Sub
  - Database Denormalization
tags:
  - advanced
  - feed
  - followers
  - graph
  - likes
  - notifications
  - real-time
architecture_doc: architecture-docs/social-network/index.md
languages:
  recommended:
    - JavaScript
    - Python
    - Go
  also_possible:
    - Ruby
    - Java
    - Elixir
resources:
  - name: Feed Architecture
    url: https://www.youtube.com/watch?v=QmX2NPkJTKg
    type: video
  - name: System Design Social Network
    url: https://www.youtube.com/results?search_query=system+design+social+network
    type: video
prerequisites:
  - type: skill
    name: Full-stack web development
  - type: skill
    name: Database design
  - type: skill
    name: Caching concepts
  - type: skill
    name: Real-time systems (WebSocket basics)
milestones:
  - id: social-network-m1
    name: User Profiles & Follow System
    description: >-
      Build user profiles with CRUD operations and a follow/follower
      relationship graph with atomic operations and denormalized counts.
    estimated_hours: 10
    concepts:
      - Self-referential many-to-many relationships (follow graph)
      - Denormalized follower/following counts with atomic increment/decrement
      - Count caching with periodic reconciliation
      - Composite indexes for efficient follower list queries
    skills:
      - Database schema design
      - Many-to-many relationships
      - SQL query optimization
      - API endpoint design
    acceptance_criteria:
      - "User profile CRUD: create, read, update, delete with validation on bio (max 500 chars), avatar URL (valid URL format), and links (max 5, valid URLs)"
      - "Follow/unfollow operations atomically create/delete the relationship record AND increment/decrement denormalized follower and following counts in a single transaction"
      - Self-follow attempts return a 400 error with descriptive message
      - "Follower and following lists return cursor-paginated results (max 50 per page) including user ID, username, display name, and avatar"
      - "Denormalized follower and following counts remain accurate after 100+ rapid follow/unfollow operations (verified by comparing count to actual relationship rows)"
      - "Profile privacy settings (public/private) are persisted; private profiles return 403 for follower list requests from non-followers"
      - Authentication is required for all write operations; read operations on public profiles are unauthenticated
    pitfalls:
      - Self-follow allowed if not explicitly checked
      - Count drift from denormalization without transactional consistency
      - N+1 queries when loading follower lists with profile data
      - Missing composite index on (follower_id, followed_id) causing slow lookups
    deliverables:
      - User profile CRUD API (bio, avatar, links, privacy setting)
      - Follow/unfollow endpoints with duplicate-follow prevention and self-follow guard
      - Cursor-paginated follower and following list endpoints
      - Denormalized count columns with atomic transactional updates
      - Authentication middleware protecting write endpoints

  - id: social-network-m2
    name: Posts & Feed (Fan-out on Write)
    description: >-
      Implement post creation and a home feed using fan-out-on-write
      architecture with asynchronous delivery and cursor-based pagination.
    estimated_hours: 12
    concepts:
      - "Fan-out on write: when a user posts, write a reference to every follower's feed"
      - Asynchronous fan-out via background job queue to avoid blocking post creation
      - Cursor-based pagination for stable feed ordering
      - "Hybrid fan-out: fan-out on write for normal users, fan-out on read for high-follower accounts (>10K followers)"
    skills:
      - Asynchronous processing
      - Background job systems
      - Database indexing strategies
      - Cursor pagination
    acceptance_criteria:
      - "Post creation endpoint accepts text (max 5000 chars) and optional image URL; returns the created post within 200ms regardless of follower count"
      - "Fan-out is performed asynchronously: a background job writes the post reference to each follower's feed list within 5 seconds of creation for users with <10K followers"
      - "For users with >=10K followers, the feed is assembled at read time (fan-out on read) to avoid expensive write amplification"
      - "Home feed endpoint returns chronologically ordered posts from followed users using cursor-based pagination (next_cursor token, max 20 posts per page)"
      - "User's own post list returns all authored posts in reverse chronological order with cursor pagination"
      - Feed pagination is stable — inserting new posts does not cause duplicates or skips in paginated results
    pitfalls:
      - Fan-out blocking post creation response time (must be async)
      - "Celebrity problem: fan-out on write for millions of followers is prohibitively slow and expensive"
      - Offset-based pagination performance degrades and causes duplicates/skips as new posts are inserted
      - Not indexing the feed table by (user_id, created_at) causing full table scans
    deliverables:
      - Post creation endpoint with text and image support
      - Background fan-out job that writes post references to follower feeds
      - Hybrid fan-out logic with configurable follower threshold
      - Feed retrieval endpoint with cursor-based pagination
      - User post timeline endpoint

  - id: social-network-m3
    name: Likes, Comments & Interactions
    description: >-
      Add like/unlike, threaded comments, and interaction count tracking with
      race condition prevention.
    estimated_hours: 10
    concepts:
      - "Idempotent like toggle: unique constraint on (user_id, post_id) prevents double-likes"
      - Atomic count updates with database-level constraints
      - Threaded comments using adjacency list (parent_id) or materialized path
      - Optimistic UI updates with server reconciliation
    skills:
      - Transaction management
      - Race condition prevention
      - Atomic operations
      - Nested data structures
    acceptance_criteria:
      - "Like/unlike is idempotent: liking an already-liked post returns success without incrementing count; a unique constraint on (user_id, post_id) prevents duplicates at the database level"
      - "Like count is updated atomically (UPDATE posts SET like_count = like_count + 1) within the same transaction as the like record insert"
      - Comments support text content (max 2000 chars) with author attribution and creation timestamp
      - "Threaded replies are supported with a parent_comment_id field; replies are returned nested under their parent up to 3 levels deep"
      - "Like and comment counts on a post are consistent: count column matches the actual count of related records (verified under concurrent load with 50 simultaneous like requests)"
      - "Notification trigger events are emitted asynchronously for like, comment, and reply actions (consumed by M4)"
    pitfalls:
      - Double-like race condition without unique constraint at database level
      - Using application-level count tracking instead of atomic SQL increment
      - Deeply nested comment trees causing recursive query performance issues
      - Returning all comments without pagination for posts with thousands of comments
    deliverables:
      - Like/unlike toggle with unique constraint and atomic count update
      - Comment CRUD with parent_comment_id for threading
      - Paginated comment retrieval with nested reply loading (max 3 levels)
      - Async event emission for notification pipeline

  - id: social-network-m4
    name: Notifications
    description: >-
      Build a real-time notification system for social events with batching,
      WebSocket delivery, and preference controls.
    estimated_hours: 10
    concepts:
      - Notification inbox pattern with unread count
      - "WebSocket or SSE for real-time push delivery"
      - "Notification batching: '5 people liked your post' instead of 5 individual notifications"
      - Redis pub/sub for multi-server WebSocket coordination
    skills:
      - Push notification architecture
      - Real-time event delivery (WebSocket/SSE)
      - Notification batching and deduplication
      - Redis pub/sub
    acceptance_criteria:
      - "Notifications fire for: new follower, like on user's post, comment on user's post, reply to user's comment"
      - Self-notifications are suppressed (user does not receive notification for their own actions)
      - "Similar notifications within a 5-minute window are batched (e.g., '3 people liked your post' instead of 3 separate notifications)"
      - "Real-time delivery via WebSocket pushes notification events to connected clients within 2 seconds of the triggering action"
      - Unread notification count badge is accurate and updates in real-time when new notifications arrive or are marked as read
      - "Mark-as-read supports single and bulk operations; count never goes below zero"
      - "Per-type notification preferences allow users to opt out of specific notification types (e.g., disable like notifications)"
    pitfalls:
      - Notification spam from rapid repeated actions (batching is essential)
      - Notifying yourself on your own actions
      - Unread count going negative due to race conditions in decrement logic
      - WebSocket connection state management across page navigations
      - Not coordinating WebSocket across multiple server instances (need Redis pub/sub)
    deliverables:
      - Notification type registry (follow, like, comment, reply, mention)
      - Notification delivery pipeline consuming events from M3 and routing to recipient inbox
      - Notification batching logic grouping similar events within time window
      - WebSocket endpoint for real-time notification push with Redis pub/sub for multi-server
      - Read/unread status with bulk mark-as-read
      - Notification preference settings per user per notification type

  - id: social-network-m5
    name: Search & Discovery
    description: >-
      Add user and post search, trending topics, and content discovery
      features.
    estimated_hours: 12
    concepts:
      - Full-text search with trigram or inverted index
      - Trending algorithm based on recent engagement velocity
      - "Content discovery: surfacing popular posts from outside the user's network"
    skills:
      - Full-text search implementation
      - Search indexing
      - Content ranking algorithms
      - Database query performance
    acceptance_criteria:
      - "User search by name or username returns results ranked by relevance for partial prefix matches (e.g., 'joh' matches 'john', 'johanna') with response time <200ms for datasets up to 1M users"
      - "Post search by keyword returns results ranked by a combination of recency and engagement score"
      - "Hashtag extraction from post content indexes hashtags for search and trending calculation"
      - "Trending topics are computed from hashtag and engagement volume over a sliding 24-hour window, updated at least every 5 minutes"
      - "Explore page returns popular posts from outside the user's follow graph, paginated"
      - Search results are paginated with cursor-based pagination (max 20 per page)
    pitfalls:
      - Full-text search without an index (trigram, GIN, or dedicated search engine) is too slow at scale
      - Trending manipulation via spam accounts (need minimum account age or engagement threshold)
      - Filter bubbles in recommendations showing only similar content
      - Not rate-limiting search endpoint allowing abuse
    deliverables:
      - User search endpoint with full-text matching on name and username
      - Post search endpoint with keyword and hashtag matching
      - Hashtag extraction and indexing pipeline
      - Trending topics aggregation (sliding window, minimum engagement threshold)
      - Explore page endpoint surfacing popular content outside user's network

  - id: social-network-m6
    name: Performance & Scaling
    description: >-
      Optimize the system for performance with caching, background processing,
      CDN integration, and load testing.
    estimated_hours: 16
    concepts:
      - "Cache-aside pattern: check cache, on miss load from DB and populate cache"
      - Cache invalidation strategies (TTL, write-through, event-driven)
      - CDN for static media assets
      - Load testing methodology and bottleneck identification
    skills:
      - Redis caching
      - Database query optimization
      - Load testing
      - CDN integration
      - Rate limiting
    acceptance_criteria:
      - "Redis caching for feeds and profiles: cache hit rate >80% under steady-state load, measured by comparing Redis hits to total requests over a 5-minute window"
      - "Feed cache is invalidated within 5 seconds when a new post is created by a followed user"
      - Background job processing handles fan-out and notification delivery asynchronously (no synchronous fan-out in request path)
      - "CDN serves uploaded media with cache-control headers (max-age >= 1 day) and response time <100ms from edge"
      - "Database indexes are verified: EXPLAIN ANALYZE on feed query, follower list query, and search query all show index scans (no sequential scans on tables >10K rows)"
      - "Load testing with a tool (k6, locust, or wrk) confirms the system handles 1000 concurrent users with p95 response time <500ms for feed and profile endpoints"
      - Rate limiting middleware returns 429 for clients exceeding 100 requests/minute
    pitfalls:
      - Cache invalidation complexity causing stale feeds
      - Celebrity user problem causing hot cache keys (use cache sharding or fan-out-on-read)
      - Hot partition on popular post records (likes table)
      - Load testing without realistic data distribution giving misleading results
      - CDN cache not invalidated when media is deleted
    deliverables:
      - Redis cache-aside implementation for feed and profile data with TTL and event-driven invalidation
      - Database index audit with EXPLAIN ANALYZE verification
      - CDN integration for media assets
      - Rate limiting middleware (token bucket or sliding window)
      - Load test scripts and results report with bottleneck analysis
      - Query optimization report showing before/after for critical queries

```
# DOMAIN PROFILE: Web & Application Development
# Applies to: app-dev
# Projects: GraphQL, WebSocket, build-react, build-bundler, collaborative editor, SaaS, etc.

## Fundamental Tension Type
Engineering tradeoffs. SIMPLICITY (ship fast) vs SCALABILITY (handle growth). FLEXIBILITY (any use case) vs PERFORMANCE (optimize common case).

Secondary: real-time vs request-response, SSR vs CSR, monolith vs microservices, abstraction vs control.

## Three-Level View
- **Level 1 — Client/User**: UI interactions, perceived latency
- **Level 2 — Application Server**: Routing, middleware, business logic, caching
- **Level 3 — Infrastructure**: DB, cache, queues, CDN, load balancing

For "Build Your Own" projects (build-react, build-bundler, etc.): look INTO the thing — Public API → Internal Engine → Underlying Primitives.

## Soul Section: "Scale Soul"
Default: what breaks at 10x users, N+1 queries, caching layers, cold cache, connection pools. But adapt — building a framework is about API ergonomics and extension points, building a real-time system is about connection lifecycle and conflict resolution.

## Alternative Reality Comparisons
Express/Fastify, Django/FastAPI, Rails, Spring Boot, Next.js/Nuxt/SvelteKit, React/Vue/Svelte/Solid, Apollo/tRPC, gRPC-Web.

## TDD Emphasis
- API spec: MANDATORY — endpoints, schemas, status codes, errors
- DB schema: MANDATORY — tables, indexes, relationships, migrations
- Auth flow: specify exactly
- WebSocket protocol: if real-time, every message type
- Memory layout: SKIP. Cache line: SKIP. Lock ordering: SKIP.
- Benchmarks: API latency p50/p95/p99, concurrent connections, req/sec

## Cross-Domain Awareness
May need systems knowledge (epoll, TCP, process management for "from scratch" servers), security (auth, CSRF/XSS), distributed concepts (session consistency, cache invalidation), or compiler knowledge (bundler = AST parsing + dependency resolution).




## Artist Examples for This Domain
- **packet_journey**: HTTP Request -> Middleware A -> Controller -> Service -> DB -> Response.
- **state_evolution**: React component lifecycle or Redux/Store state transition.
- **data_walk**: URL parsing or Virtual DOM tree diffing.
- **before_after**: UI state before and after a WebSocket notification.

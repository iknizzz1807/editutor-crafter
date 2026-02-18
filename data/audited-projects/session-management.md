# AUDIT & FIX: session-management

## CRITIQUE
- Critical gap: Session fixation defense (regenerating session ID after login/privilege escalation) is mentioned in M1 pitfalls but NOT in any acceptance criteria. This is the most important session security mechanism and must be a measurable AC.
- Milestone 2 says 'Encrypt or sign cookie values to prevent client-side tampering' but the session ID is an opaque random token — encrypting it provides no security benefit. Signing prevents tampering, but if the session ID is random and looked up server-side, tampering just produces a 'session not found' error. The real purpose of signing is for cookie-based sessions where the entire session state is in the cookie.
- Milestone 2 deliverables list 'HttpOnly flag preventing JavaScript access', 'SameSite attribute restricting cross-origin cookie transmission', and 'Secure cookie configuration with HttpOnly and Secure flags' — these are largely the same thing split into separate deliverables.
- Missing: No mention of session ID entropy requirements (OWASP recommends 128+ bits from CSPRNG). M1 says 'sufficient length' but doesn't specify what's sufficient.
- Missing: CSRF protection is listed in M2 but only as one AC among cookie flags. CSRF deserves more rigorous treatment with specific token generation and validation mechanisms.
- Milestone 3 mentions 'device fingerprinting' but the pitfalls correctly note its unreliability. The milestone should focus on session metadata (IP, user-agent) for anomaly detection, not fingerprinting.
- The project depends on a Redis project but doesn't require HTTP server knowledge, which is actually more fundamental for cookie handling.
- Estimated hours (30 total) seems reasonable but M1 and M2 could be more balanced.

## FIXED YAML
```yaml
id: session-management
name: Session Management System
description: >-
  Implement secure server-side session management with cryptographically
  secure session IDs, session fixation defense, secure cookie transport,
  CSRF protection, distributed storage, and multi-device session control.
difficulty: advanced
estimated_hours: "30-40"
essence: >-
  Cryptographically secure session ID generation (128+ bit entropy from
  CSPRNG), session fixation prevention through ID regeneration on privilege
  escalation, secure cookie attributes (HttpOnly, Secure, SameSite, __Host-
  prefix), distributed session storage with TTL-based expiration, CSRF
  token binding, and concurrent session management across multiple devices.
why_important: >-
  Session management is the foundation of authentication in web applications.
  Building it teaches distributed state management, security hardening
  against fixation/hijacking attacks, and the practical tradeoffs of
  stateful authentication at scale.
learning_outcomes:
  - Generate cryptographically secure session IDs with 128+ bit entropy
  - Implement session fixation prevention by regenerating IDs after login
  - Configure secure cookie attributes (HttpOnly, Secure, SameSite, __Host-)
  - Build distributed session storage with Redis and TTL-based expiration
  - Implement CSRF protection with session-bound anti-forgery tokens
  - Handle idle timeout, absolute timeout, and sliding expiration policies
  - Manage concurrent sessions across multiple devices with selective revocation
  - Detect session anomalies using IP and user-agent metadata tracking
skills:
  - Distributed Systems
  - Cryptographic Security
  - Session Storage Design
  - Cookie Security
  - CSRF Protection
  - Redis/Caching
  - Concurrent State Management
  - Authentication Patterns
tags:
  - advanced
  - authentication
  - cookies
  - distributed
  - security
  - sessions
  - csrf
architecture_doc: architecture-docs/session-management/index.md
languages:
  recommended:
    - Go
    - Python
    - Java
  also_possible: []
resources:
  - name: OWASP Session Management Cheat Sheet""
    url: https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html
    type: documentation
  - name: Redis Session Management""
    url: https://redis.io/solutions/session-management/
    type: documentation
  - name: Session Fixation Prevention""
    url: https://owasp.org/www-community/controls/Session_Fixation_Protection
    type: article
prerequisites:
  - type: project
    id: build-redis
    name: Redis implementation or familiarity
  - type: skill
    name: HTTP protocol and cookie mechanics
milestones:
  - id: session-management-m1
    name: "Secure Session Creation, Storage, and Fixation Defense"
    description: >-
      Implement cryptographically secure session ID generation, server-side
      session storage with expiration, and session fixation prevention
      through ID regeneration.
    acceptance_criteria:
      - >-
        Session IDs are generated using a CSPRNG with at least 128 bits
        (16 bytes) of entropy, encoded as a URL-safe string (hex or
        Base64URL). Measured: no two session IDs collide in 1 million
        generated IDs.
      - >-
        All session data is stored server-side (Redis, database, or
        in-memory store) keyed by session ID. The session cookie contains
        ONLY the session ID, never session data.
      - >-
        Session fixation defense: when a user authenticates (login), the
        old session ID is invalidated and a new session ID is generated.
        Session data is migrated to the new ID. The old ID cannot be
        used to access the authenticated session.
      - >-
        Idle timeout: sessions with no activity for a configurable duration
        (default 30 minutes) are automatically expired and removed from
        storage.
      - >-
        Absolute timeout: sessions are forcibly expired after a maximum
        lifetime (default 24 hours) regardless of activity.
      - >-
        Multiple storage backends are supported (at minimum: in-memory
        for development, Redis for production) behind a common interface.
      - >-
        Expired sessions are automatically cleaned up (Redis TTL or
        periodic sweep) without manual intervention.
    pitfalls:
      - >-
        Session fixation: if the session ID is not regenerated after login,
        an attacker who sets a session ID in the victim's browser (via
        URL parameter, cookie injection, or XSS) gains authenticated
        access when the victim logs in. This is the most critical session
        security bug.
      - >-
        Session ID in URL: never put session IDs in URLs. They leak via
        Referer headers, browser history, access logs, and shared links.
        Use cookies only.
      - >-
        Idle vs. absolute timeout confusion: idle timeout resets on each
        request; absolute timeout does not. Both are needed. A session
        with only idle timeout can last forever with continuous activity.
      - >-
        Race condition: concurrent requests with the same session ID can
        cause read-modify-write conflicts on session data. Use Redis
        transactions or optimistic locking.
    concepts:
      - CSPRNG session ID generation (128+ bit entropy)
      - Session fixation and ID regeneration
      - Idle timeout vs. absolute timeout
      - Server-side session storage patterns
    skills:
      - Session security
      - Distributed storage (Redis)
      - Timeout management
      - Concurrency control
    deliverables:
      - CSPRNG-based session ID generator (128+ bits)
      - Server-side session store with pluggable backends
      - Session fixation defense (ID regeneration on login)
      - Dual timeout policy (idle + absolute)
      - Automatic expired session cleanup
      - Test demonstrating fixation defense (old ID rejected after regeneration)
    estimated_hours: "10-12"

  - id: session-management-m2
    name: "Secure Cookie Transport and CSRF Protection"
    description: >-
      Implement secure cookie configuration with all security attributes,
      and build CSRF protection using session-bound anti-forgery tokens.
    acceptance_criteria:
      - >-
        Session cookie is set with all security flags: HttpOnly (prevents
        JavaScript access), Secure (HTTPS only), SameSite=Lax or Strict
        (prevents cross-site request attachment).
      - >-
        Cookie uses __Host- prefix (e.g., __Host-SessionId) which
        enforces Secure flag, no Domain attribute, and Path=/, preventing
        cookie injection from subdomains.
      - >-
        Cookie Max-Age or Expires is set to match the session absolute
        timeout so the browser discards the cookie when the session expires.
      - >-
        CSRF anti-forgery token is generated per session (at least 128
        bits from CSPRNG) and stored in the session data on the server.
      - >-
        State-changing requests (POST, PUT, DELETE) require the CSRF
        token submitted as a request header (X-CSRF-Token) or hidden
        form field, validated against the session-stored token.
      - >-
        CSRF token validation uses constant-time comparison to prevent
        timing side-channels.
      - >-
        Missing or invalid CSRF token produces a 403 Forbidden response
        with a descriptive error.
      - >-
        Header-based session tokens (Authorization: Bearer <session_id>)
        are supported as a cookie-less alternative for API clients, with
        the same security validation.
    pitfalls:
      - >-
        SameSite=None requires the Secure flag (HTTPS). Setting
        SameSite=None without Secure causes the cookie to be rejected
        by modern browsers.
      - >-
        CSRF tokens must be bound to the session. A global CSRF token
        shared across sessions allows an attacker to use their own valid
        token in a CSRF attack.
      - >-
        Double-submit cookie pattern (CSRF token in both cookie and
        header) is simpler but weaker than synchronizer token pattern.
        The synchronizer pattern (server-stored token) is preferred.
      - >-
        Cookie size limit (~4KB): if session data is stored in the cookie
        (not recommended), it can exceed the limit. Server-side storage
        avoids this entirely.
    concepts:
      - Cookie security attributes (HttpOnly, Secure, SameSite, __Host-)
      - CSRF synchronizer token pattern
      - Cookie vs. header-based session transport
      - Constant-time token comparison
    skills:
      - Cookie security configuration
      - CSRF token generation and validation
      - HTTP security headers
      - API authentication patterns
    deliverables:
      - Secure cookie configuration with all flags and __Host- prefix
      - CSRF token generation and session binding
      - CSRF validation middleware for state-changing requests
      - Header-based session token support for API clients
      - Test suite for CSRF rejection, cookie flags, and token validation
    estimated_hours: "8-10"

  - id: session-management-m3
    name: "Multi-Device Sessions and Anomaly Detection"
    description: >-
      Support multiple concurrent sessions per user with listing,
      selective revocation, session limits, and basic anomaly detection.
    acceptance_criteria:
      - >-
        Each user can have multiple active sessions, each tracked with
        metadata: session ID (hashed for display), creation time, last
        activity time, IP address, and user-agent string.
      - >-
        Session listing API returns all active sessions for the
        authenticated user, showing metadata but never the raw session ID.
      - >-
        Individual session revocation: a user can terminate any specific
        session (except the current one, which uses a separate logout
        endpoint).
      - >-
        Global logout: a user can terminate ALL sessions (including the
        current one), invalidating every session ID associated with the
        user.
      - >-
        Configurable concurrent session limit (default: 5 per user).
        When the limit is exceeded, the oldest session is automatically
        revoked (FIFO eviction).
      - >-
        Basic anomaly detection: if a session's request IP address
        changes significantly (different /16 subnet) or user-agent
        changes, the session is flagged for re-authentication or
        terminated based on policy.
      - >-
        Last activity timestamp is updated on each request, supporting
        the idle timeout from M1.
    pitfalls:
      - >-
        Displaying raw session IDs in the UI: never expose the full
        session ID. Show a hash, truncation, or opaque identifier.
        Exposing the ID enables session hijacking.
      - >-
        Session limit without cleanup: if old sessions aren't properly
        evicted, users get locked out of creating new sessions. Always
        implement automatic eviction.
      - >-
        IP-based anomaly detection false positives: mobile users switch
        IPs frequently (WiFi to cellular). VPN users share IPs. Use
        anomaly detection for flagging, not automatic termination, unless
        the security policy requires it.
      - >-
        Time zone confusion in session metadata: store all timestamps
        in UTC. Display in the user's local timezone on the frontend.
    concepts:
      - Per-user session enumeration
      - Session revocation (single and global)
      - Concurrent session limits and eviction
      - IP and user-agent anomaly detection
    skills:
      - Multi-session tracking
      - Session revocation patterns
      - Anomaly detection heuristics
      - Session metadata management
    deliverables:
      - Per-user session tracking with metadata (IP, user-agent, timestamps)
      - Session listing API (no raw session IDs exposed)
      - Individual and global session revocation
      - Concurrent session limit with FIFO eviction
      - Basic anomaly detection for IP/user-agent changes
      - Test suite for concurrent sessions, revocation, and limit enforcement
    estimated_hours: "10-12"
```
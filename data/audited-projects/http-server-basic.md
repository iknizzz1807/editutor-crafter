# AUDIT & FIX: http-server-basic

## CRITIQUE
- **Huge jump between M1 and M4**: M1 is a sequential TCP server, M4 demands thread pools, non-blocking I/O, and graceful shutdown. There's no intermediate milestone covering shared state, thread-per-connection basics isolated from pooling, or an introduction to synchronization primitives. This will cause students to struggle.
- **Missing HTTP compliance ACs**: No AC for handling `Connection: keep-alive` or `Connection: close` headers. HTTP/1.1 defaults to keep-alive, meaning the server must handle multiple requests per connection. This is a fundamental protocol requirement.
- **No Range request or conditional request support**: The audit correctly identifies this. Static file servers commonly need `If-Modified-Since`, `If-None-Match` (ETags), and `Range` headers for resume-able downloads and caching. While not all are required for 'basic', at least `If-Modified-Since`/304 should be an AC.
- **M2 mentions 'virtual host routing support' via Host header**: This is premature for a basic HTTP server and adds scope without corresponding AC for actually implementing virtual hosts. It should be scoped as 'extract and log the Host header value' not 'virtual host routing support'.
- **M3 directory listing security**: The deliverable says 'Directory listing displaying files when path maps to a directory' but this is a security risk if enabled by default. The AC doesn't mention this should be configurable or disabled by default.
- **M3 path traversal AC is good but implementation guidance is missing**: Students will try `path.startsWith(root)` which can be bypassed with symlinks. The pitfall should mention canonicalizing/realpath before comparison.
- **M4 AC is contradictory**: It asks for BOTH 'thread-per-connection model' AND 'non-blocking with select/poll' in the same milestone. These are fundamentally different architectures. Students should implement one concurrency model thoroughly, not both superficially.
- **No AC for request timeout**: A client that connects but never sends data will hold a thread/fd forever. Timeouts are essential for any production server.
- **Missing AC for HEAD method**: HEAD requests should return headers without body, which is required by HTTP/1.1 spec.

## FIXED YAML
```yaml
id: http-server-basic
name: HTTP Server (Basic)
description: Static file serving HTTP/1.1 server with concurrent connections
difficulty: intermediate
estimated_hours: "15-24"
essence: >
  Socket-level TCP server implementation that parses HTTP/1.1 protocol messages,
  manages file I/O for static content delivery with security constraints, and
  coordinates concurrent client connections through threading or a thread pool.
why_important: >
  Building this establishes foundational knowledge of how web infrastructure works
  beneath frameworks, teaching network programming patterns used across backend services,
  distributed systems, and real-time applications.
learning_outcomes:
  - Implement TCP socket binding, listening, and accept loops with proper error handling
  - Parse HTTP/1.1 request lines and headers per RFC 7230
  - Serve static files with correct MIME types and security constraints against directory traversal
  - Handle concurrent connections using thread-per-connection and thread pool models
  - Implement HTTP/1.1 keep-alive for multiple requests per connection
  - Handle conditional requests with If-Modified-Since and 304 Not Modified responses
  - Implement graceful shutdown completing in-flight requests
skills:
  - TCP/IP Socket Programming
  - HTTP Protocol Implementation
  - Concurrent Programming
  - System I/O Operations
  - Network Debugging
  - Thread Synchronization
  - File System Security
  - Error Handling
tags:
  - c
  - concurrency
  - go
  - intermediate
  - networking
  - protocols
  - request-response
  - rust
  - service
  - static-files
architecture_doc: architecture-docs/http-server-basic/index.md
languages:
  recommended:
    - C
    - Go
    - Rust
  also_possible:
    - Python
    - Java
resources:
  - name: "HTTP/1.1 Specification (RFC 7230)"
    url: https://tools.ietf.org/html/rfc7230
    type: specification
  - name: "Beej's Guide to Network Programming"
    url: https://beej.us/guide/bgnet/
    type: tutorial
prerequisites:
  - type: skill
    name: TCP/IP basics
  - type: skill
    name: Socket programming
  - type: skill
    name: File I/O
milestones:
  - id: http-server-basic-m1
    name: TCP Server & HTTP Response
    description: >
      Create a TCP server that accepts connections sequentially, reads raw bytes from
      the client socket, and sends a hardcoded HTTP/1.1 response. Focus on the socket
      lifecycle: bind, listen, accept, read, write, close.
    acceptance_criteria:
      - Server binds to a configurable port (default 8080) and listens for incoming TCP connections
      - Accept loop handles sequential client connections one at a time
      - Read complete request data from client socket, handling partial reads by looping until CRLF CRLF delimiter is found
      - Send a hardcoded HTTP/1.1 200 OK response with Content-Type, Content-Length, and Date headers, plus a small HTML body
      - Close client socket and free resources after response is fully sent
      - Server survives client disconnection without crashing (handle SIGPIPE or equivalent)
    pitfalls:
      - Byte order issues: port number must use htons() for network byte order on bind
      - Partial reads: recv/read may return fewer bytes than requested; must loop and accumulate until the end-of-headers delimiter (\r\n\r\n) is found
      - Forgetting to close the client file descriptor leaks resources; server will hit FD limits after ~1000 connections
      - On Linux, writing to a closed client socket sends SIGPIPE which kills the server; must ignore SIGPIPE or use MSG_NOSIGNAL
    concepts:
      - Socket system calls: socket, bind, listen, accept
      - TCP connection lifecycle
      - HTTP response format
    skills:
      - Socket programming
      - Network I/O and partial read handling
      - Error handling for system calls
      - Resource management (file descriptor cleanup)
    deliverables:
      - TCP socket creation, binding, and listening
      - Sequential accept loop for client connections
      - Partial-read-safe request reading
      - Hardcoded HTTP response with proper headers
      - Clean client socket closure
    estimated_hours: "2-4"

  - id: http-server-basic-m2
    name: HTTP Request Parsing
    description: >
      Parse the HTTP/1.1 request line and headers into structured data. Handle
      malformed requests with appropriate error responses.
    acceptance_criteria:
      - Parse request line extracting method (GET, HEAD, POST), path, and HTTP version (HTTP/1.1); reject malformed request lines with 400 Bad Request
      - Parse headers into key-value pairs, handling optional whitespace around header values and case-insensitive header names per RFC 7230
      - Handle both CRLF (\r\n) and bare LF (\n) line endings for robustness
      - Extract and log the Host header value (required in HTTP/1.1)
      - Read request body based on Content-Length header when present (for future POST support)
      - Respond with 501 Not Implemented for methods other than GET and HEAD
      - HEAD requests return the same headers as GET but with an empty body
      - Respond with 414 URI Too Long for request paths exceeding 8KB
    pitfalls:
      - Mixing up CRLF and LF line endings causes header parsing to fail on certain clients (e.g., telnet sends LF only)
      - Header names are case-insensitive per HTTP spec but naive string comparison treats them as case-sensitive
      - Buffer overflow risk: unbounded request line or header reading without a length limit allows denial of service
      - Not handling missing Host header (required in HTTP/1.1) silently violates the spec
    concepts:
      - HTTP/1.1 message format (RFC 7230)
      - Request line parsing
      - Header parsing rules
    skills:
      - String parsing with delimiters
      - Protocol implementation
      - Buffer management with size limits
      - Error response generation
    deliverables:
      - Request line parser extracting method, path, and version
      - Header parser producing case-insensitive key-value map
      - Body reader using Content-Length
      - Error responses for malformed requests (400), unsupported methods (501), and oversized URIs (414)
      - HEAD method support returning headers without body
    estimated_hours: "3-4"

  - id: http-server-basic-m3
    name: Static File Serving
    description: >
      Serve static files from a configured document root directory with proper
      Content-Type detection, security validation, and conditional request support.
    acceptance_criteria:
      - Map URL path to filesystem path within a configured document root directory
      - Canonicalize the resolved path (resolve symlinks, normalize ../) and verify it is still within the document root; reject with 403 Forbidden if path escapes root
      - Detect and set Content-Type header based on file extension using a MIME type map supporting at least: .html, .css, .js, .json, .png, .jpg, .gif, .svg, .txt, .pdf
      - Send file contents in response body with correct Content-Length header
      - Return 404 Not Found for files that don't exist, with a simple HTML error page
      - Return 403 Forbidden for paths that resolve outside the document root
      - Support If-Modified-Since header: compare against file modification time and return 304 Not Modified with no body when file is unchanged
      - Set Last-Modified header on all file responses using the file's modification timestamp
      - Serve index.html automatically when path maps to a directory (if index.html exists); otherwise return 403
    pitfalls:
      - Using simple string prefix check for path containment is bypassable with symlinks; must canonicalize (realpath) THEN check containment
      - URL-encoded path traversal: %2e%2e%2f decodes to ../ and must be decoded before path resolution
      - Binary files (images, PDFs) must be read and sent in binary mode; text-mode reads corrupt binary content on some platforms
      - Missing Content-Type defaults to application/octet-stream, which triggers download instead of display in browsers
      - Last-Modified timestamp comparison must handle timezone differences and second-level granularity per HTTP date format
    concepts:
      - Path resolution and canonicalization
      - MIME types and Content-Type
      - Conditional HTTP requests (304 Not Modified)
      - Directory traversal prevention
    skills:
      - File I/O in binary mode
      - Path security validation
      - MIME type resolution
      - HTTP conditional request handling
    deliverables:
      - File serving from document root with path canonicalization
      - MIME type detection for common file types
      - 404 and 403 error pages
      - If-Modified-Since / Last-Modified conditional response support
      - Index file auto-serving for directory paths
    estimated_hours: "4-6"

  - id: http-server-basic-m4
    name: Concurrent Connections
    description: >
      Handle multiple simultaneous client connections using a thread-per-connection
      model with a bounded thread pool. Support HTTP/1.1 keep-alive and graceful shutdown.
    acceptance_criteria:
      - Thread-per-connection model spawns a handler thread for each accepted connection
      - Thread pool with configurable size (default 16) limits maximum concurrent handlers; connections beyond the pool size are queued or rejected with 503 Service Unavailable
      - HTTP/1.1 keep-alive: multiple sequential requests are handled on a single connection without closing; Connection: close header signals the server to close after the response
      - Per-connection read timeout (configurable, default 30 seconds): connections idle longer than the timeout are closed to prevent resource exhaustion
      - Graceful shutdown on SIGTERM/SIGINT: stop accepting new connections, complete all in-flight requests, then exit
      - No file descriptor leaks under sustained load: after 10,000 sequential connections, the server's open FD count returns to baseline
      - Shared state (e.g., connection counter, access log) is protected with mutexes; no data races under concurrent access
    pitfalls:
      - Thread-per-connection without a pool limit allows an attacker to exhaust OS threads/memory with many connections
      - Keep-alive connections without idle timeouts accumulate and exhaust file descriptors
      - Forgetting to join/detach handler threads on shutdown leaks resources or causes segfaults
      - Signal handling in multithreaded programs requires care: signals may be delivered to any thread; use dedicated signal thread or sigwait
      - Accessing shared counters or log files from multiple threads without synchronization causes data races
    concepts:
      - Thread-per-connection concurrency model
      - Thread pooling and work queues
      - HTTP/1.1 persistent connections
      - Graceful shutdown with signal handling
    skills:
      - Thread creation and management
      - Thread pool implementation
      - Mutex/synchronization for shared state
      - Connection timeout management
      - Signal handling in multithreaded contexts
    deliverables:
      - Thread-per-connection accept loop
      - Bounded thread pool with configurable size
      - HTTP/1.1 keep-alive with Connection header support
      - Per-connection idle timeout enforcement
      - Graceful shutdown on SIGTERM/SIGINT
      - Shared state protection with mutexes
    estimated_hours: "5-8"
```
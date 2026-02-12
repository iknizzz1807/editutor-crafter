# HTTP Server: Design Document


## Overview

This system implements a concurrent HTTP/1.1 static file server that accepts TCP connections, parses HTTP requests, and serves files from a document root directory. The key architectural challenge is handling multiple concurrent client connections efficiently while maintaining proper HTTP protocol compliance and security boundaries.


> This guide is meant to help you understand the big picture before diving into each milestone. Refer back to it whenever you need context on how components connect.


## Context and Problem Statement

> **Milestone(s):** Background for all milestones - establishes fundamental concepts needed throughout the project

The HTTP server project presents a fascinating intersection of network programming, protocol implementation, and concurrent system design. While web servers might seem ubiquitous in modern computing, building one from scratch reveals the intricate dance of TCP socket management, HTTP message parsing, file system interaction, and concurrent connection handling. This section establishes the foundational concepts and challenges that make HTTP server implementation both educational and technically demanding.

Understanding these fundamentals is crucial because HTTP servers must simultaneously handle multiple concerns: they operate at the intersection of network protocols (TCP/IP), application protocols (HTTP), operating system resources (file descriptors, threads), and security boundaries (filesystem access control). Each of these domains introduces its own complexity and failure modes, and their interaction creates emergent challenges that require careful architectural consideration.

### The Library Desk Analogy

To build intuition about HTTP server architecture, consider the analogy of a library reference desk during peak hours. The reference desk represents your HTTP server, and library patrons represent client requests arriving over the network.

**The Single Librarian Problem**: Initially, imagine a library with one reference desk staffed by a single librarian. When a patron arrives with a question, the librarian must listen to the complete question, understand what information is being requested, locate the appropriate books or documents, and provide a complete answer before helping the next patron. If the requested information requires searching through archives in the basement, every other patron must wait in line while the librarian retrieves the material. This represents a **sequential, blocking server model** - each request must complete entirely before the next can begin processing.

**The Multi-Desk Solution**: As patron volume increases, the library might add multiple reference desks, each staffed by a different librarian. Now multiple patrons can be served simultaneously, with each librarian handling their own queue. However, this approach requires careful resource management - if all librarians are simultaneously searching the basement archives, the reference area becomes empty and new patrons face long delays. This represents a **thread-per-connection model** where each connection gets dedicated processing resources.

**The Specialist Dispatcher Model**: A more sophisticated approach involves a front desk that quickly categorizes patron requests and routes them to specialists - simple catalog questions go to one area, research requests to another, and document retrieval to a third team. The front desk never blocks on any individual request, instead maintaining awareness of all active requests and efficiently routing work. This represents an **event-driven, asynchronous server model** that can handle many concurrent connections without proportional resource scaling.

**Resource Sharing and Conflicts**: In all models, librarians must access shared resources - the card catalog, the archive keys, the photocopier. When multiple librarians need the same resource simultaneously, coordination becomes essential to prevent conflicts and ensure fair access. This mirrors the challenges of concurrent access to shared server resources like configuration data, log files, and cached content.

**Request Complexity Variations**: Some patron requests are simple ("Where are the restrooms?") while others are complex ("I need all documents related to 19th-century agricultural practices in the Pacific Northwest"). Similarly, HTTP requests vary dramatically - serving a small CSS file requires minimal processing, while serving a large video file involves substantial I/O and bandwidth management. The server architecture must handle both efficiently without allowing complex requests to starve simple ones.

This analogy illuminates several key architectural decisions that HTTP servers must make: How should incoming connections be accepted and queued? Should each connection receive dedicated processing resources or share them? How should the server handle requests of varying complexity? What happens when system resources become constrained?

### Existing Web Server Approaches

Production web servers have evolved different architectural approaches to address the concurrency and performance challenges inherent in HTTP serving. Understanding these approaches provides context for the design decisions in our educational HTTP server implementation.

| Server | Concurrency Model | Connection Handling | Pros | Cons |
|---------|------------------|-------------------|------|------|
| Apache HTTP Server (traditional) | Process-per-connection with pre-forking | Each connection handled by dedicated process | Strong isolation, robust fault tolerance, simple debugging | High memory overhead, limited scalability, process creation costs |
| Apache HTTP Server (modern) | Hybrid with worker MPM | Thread pool with event-driven accept | Better resource efficiency than pure process model | Still more overhead than pure event-driven approaches |
| nginx | Event-driven with worker processes | Single-threaded event loop per CPU core | Extremely high connection capacity, low memory usage | Complex programming model, difficult debugging |
| lighttpd | Event-driven single process | select/poll/epoll-based event loop | Lightweight, efficient for static content | Limited by single process, less robust under high load |

**Apache's Process-per-Connection Legacy**: The original Apache HTTP Server used a process-per-connection model where each incoming HTTP connection was handled by a dedicated process. This approach provided excellent isolation - a crash in one connection handler couldn't affect others - but suffered from severe scalability limitations. Process creation overhead meant that under high connection loads, significant CPU time was spent managing processes rather than serving content. Memory usage scaled linearly with concurrent connections, creating practical limits in the hundreds of simultaneous connections rather than thousands.

**The C10K Problem and Event-Driven Solutions**: The "C10K problem" - handling 10,000 concurrent connections on a single server - exposed the limitations of thread-per-connection and process-per-connection models. Event-driven architectures emerged as a solution, using operating system primitives like `select()`, `poll()`, and `epoll()` to monitor many connections simultaneously within a single thread. nginx popularized this approach, demonstrating that a single server could handle tens of thousands of concurrent connections with minimal memory overhead.

**Hybrid Threading Models**: Modern servers often combine approaches to balance simplicity and performance. A common pattern uses a small number of worker threads (typically one per CPU core) where each worker runs an event loop managing many connections. This provides the scalability benefits of event-driven architecture while maintaining some isolation between workers and enabling utilization of multiple CPU cores.

**Thread Pool with Connection Queuing**: Another common approach maintains a pool of worker threads that pull connection-handling tasks from a shared queue. New connections are accepted by a dedicated acceptor thread and placed in the work queue, where available worker threads can claim them. This model provides good resource control - the thread pool size can be tuned based on system resources - while avoiding the overhead of creating threads per connection.

> **Critical Insight**: The choice of concurrency model fundamentally determines server scalability characteristics. Process-per-connection models excel at isolation and simplicity but scale poorly. Event-driven models achieve excellent scalability but increase implementation complexity and debugging difficulty. Thread pools balance these concerns but introduce coordination complexity.

**Resource Management Implications**: Each concurrency model has different implications for system resource usage. Process-per-connection models consume significant memory per connection (typically several MB per process) but provide strong CPU isolation. Thread-per-connection models reduce memory overhead (typically 8KB-2MB per thread depending on stack size) but share address space. Event-driven models minimize per-connection overhead (typically a few KB per connection state structure) but concentrate processing in fewer threads, potentially creating CPU bottlenecks.

**Error Isolation Trade-offs**: The concurrency model also determines fault isolation characteristics. In process-per-connection models, a crash in request handling affects only that single connection. In thread-per-connection models, a crash can potentially bring down the entire server process. Event-driven models face similar risks, where a bug in connection handling can affect many connections processed by the same event loop.

### Core Technical Challenges

Building an HTTP server involves navigating several categories of technical challenges, each with distinct characteristics and common failure modes. Understanding these challenges upfront helps inform architectural decisions and implementation priorities.

#### Network Programming Complexity

Network programming introduces fundamental challenges that don't exist in traditional single-process applications. The most basic challenge is the **asynchronous and unreliable nature of network communication**. Unlike function calls within a process, network operations can fail in numerous ways: connections can be refused, data can be lost or corrupted, remote hosts can become unreachable, and operations can timeout. These failures can occur at any point during communication, requiring comprehensive error handling throughout the server implementation.

**Socket Lifecycle Management** presents another layer of complexity. Each client connection involves creating a socket file descriptor, binding it to network addresses, managing its state transitions (listening, accepting, reading, writing), and ensuring proper cleanup. File descriptor leaks are a common failure mode - forgetting to close socket file descriptors eventually exhausts the operating system's file descriptor limit, causing the server to refuse new connections even when CPU and memory resources are available.

**Partial Read/Write Operations** create subtle bugs that are difficult to reproduce in testing but common in production. Network operations like `read()` and `write()` are not guaranteed to process all requested data in a single call. A call to `read(fd, buffer, 1024)` might only read 237 bytes if that's all currently available in the kernel's socket buffer. Similarly, `write()` operations might only transmit part of the data if the socket's send buffer becomes full. Robust network code must loop on these operations, tracking progress and resuming from partial completion.

| Network Challenge | Common Failure Mode | Detection Method | Recovery Strategy |
|-------------------|---------------------|------------------|-------------------|
| Connection timeouts | Client connects but never sends data | Set socket timeout options, monitor idle time | Close connection after timeout period |
| Partial reads | HTTP parsing fails on incomplete headers | Track bytes read vs expected, buffer management | Continue reading until complete message received |
| Partial writes | Response truncated, client sees incomplete data | Check `write()` return value vs intended bytes | Loop on writes, track progress, handle EAGAIN |
| File descriptor exhaustion | `accept()` returns -1 with EMFILE/ENFILE | Monitor return values, system fd limits | Close idle connections, implement connection limits |
| Port binding conflicts | Server fails to start, address already in use | `bind()` returns -1 with EADDRINUSE | Use SO_REUSEADDR socket option, check port availability |

**Endianness and Binary Data Handling** introduces platform-specific complexity. Network protocols typically use network byte order (big-endian), while many modern processors use little-endian byte order. Functions like `htons()` (host-to-network-short) and `ntohs()` (network-to-host-short) are essential for converting port numbers and addresses correctly. Failure to handle byte order conversion can cause connections to fail or bind to unexpected ports.

#### HTTP Protocol Parsing Challenges

The HTTP/1.1 protocol appears deceptively simple in examples, but robust parsing requires handling numerous edge cases and protocol variations. **Line Ending Variations** present an immediate challenge - the HTTP specification mandates CRLF (`\r\n`) line endings, but real-world clients sometimes send only LF (`\n`). Parsers must handle both variations gracefully while maintaining protocol compliance.

**Header Parsing Complexity** extends beyond simple key-value extraction. HTTP headers can span multiple lines using continuation syntax, where subsequent lines beginning with whitespace are considered part of the previous header. Header values can contain quoted strings with escape sequences. Header names are case-insensitive, requiring normalization for consistent processing. Some headers like `Set-Cookie` can appear multiple times in a single response, requiring special handling.

```
Example complex header scenarios:
X-Custom-Header: value spans
    multiple lines
Content-Type: text/html; charset="utf-8"
Set-Cookie: session=abc123; Path=/
Set-Cookie: preferences=theme:dark; Expires=Wed, 09 Jun 2023 10:18:14 GMT
```

**Content Length and Transfer Encoding** create parsing state machine complexity. HTTP messages can indicate body length in several ways: explicit `Content-Length` header, chunked transfer encoding, or connection close. Each approach requires different parsing logic and buffer management. Chunked encoding, in particular, requires parsing hex-encoded chunk sizes and handling chunk boundaries correctly.

| HTTP Parsing Challenge | Impact | Solution Approach |
|------------------------|---------|------------------|
| Malformed request lines | Server crash or incorrect routing | Validate components, return 400 Bad Request for invalid format |
| Missing required headers | Protocol violation, client confusion | Check for Host header in HTTP/1.1, return appropriate error codes |
| Oversized headers | Memory exhaustion, DoS vulnerability | Implement header size limits, return 431 Request Header Fields Too Large |
| Invalid method names | Security issues, unexpected behavior | Whitelist valid methods, return 405 Method Not Allowed |
| URL encoding issues | Path traversal, security vulnerabilities | Proper URL decoding with validation |

**Request Body Handling** introduces additional complexity for methods like POST and PUT. The server must read the exact number of bytes specified by the `Content-Length` header - reading too few leaves data in the socket buffer that corrupts subsequent requests, while reading too many can block waiting for data that never arrives. For chunked encoding, the server must parse chunk headers and reconstruct the original message body.

#### Concurrency and Thread Safety

Concurrent connection handling multiplies the complexity of every operation by introducing race conditions, resource contention, and synchronization requirements. **Shared State Management** becomes critical when multiple connections need access to server configuration, cached data, or shared resources like log files. Without proper synchronization, concurrent access can corrupt data structures, leading to crashes or incorrect behavior.

**Resource Contention** occurs when multiple connections compete for limited resources. File descriptor limits, memory allocation, thread pools, and even CPU cache lines become points of contention under high load. The server must implement fair resource allocation policies and graceful degradation when resources become scarce.

**Deadlock Prevention** requires careful ordering of lock acquisition and understanding of dependency relationships between shared resources. A common deadlock scenario occurs when one thread holds a connection lock while waiting for a file system lock, while another thread holds the file system lock while waiting for the connection lock. Breaking such cycles requires consistent lock ordering or timeout-based lock acquisition.

| Concurrency Challenge | Risk | Prevention Strategy |
|-----------------------|------|-------------------|
| Race conditions in request counting | Incorrect statistics, resource leaks | Use atomic operations or proper locking |
| Multiple threads writing to log files | Corrupted log entries, file system errors | Synchronize log writes or use per-thread log buffers |
| Thread pool exhaustion | New connections hang indefinitely | Implement connection limits and queue bounds |
| Memory allocation contention | Performance degradation under load | Use thread-local allocators or lock-free data structures |
| Signal handling in multi-threaded context | Undefined behavior, crashes | Block signals in worker threads, handle in dedicated thread |

**Thread Lifecycle Management** adds operational complexity. Worker threads must be created, assigned work, monitored for health, and cleanly shut down during server restart. Thread creation is expensive, so servers typically use thread pools, but pool sizing requires balancing resource usage against responsiveness. Too few threads create bottlenecks under load, while too many threads can cause excessive context switching overhead.

**Graceful Shutdown Coordination** becomes complex with multiple concurrent connections. The server must stop accepting new connections, complete in-flight requests within a reasonable timeout, release resources cleanly, and coordinate shutdown across all worker threads. This requires careful state management and inter-thread communication.

> **Design Principle**: The complexity of concurrent systems grows exponentially with the number of shared resources and interaction points. Minimizing shared state and designing for isolation reduces both implementation complexity and debugging difficulty.

These technical challenges influence every architectural decision in HTTP server design. The choice of concurrency model affects network programming complexity - event-driven servers must handle partial operations differently than thread-per-connection servers. HTTP parsing requirements influence buffer management strategies and error handling approaches. Understanding these challenges upfront enables making informed trade-offs between simplicity, performance, and robustness throughout the implementation process.

### Implementation Guidance

Understanding the theoretical challenges is essential, but translating them into working code requires practical knowledge of tools, libraries, and development approaches. This guidance bridges the gap between conceptual understanding and actual implementation.

#### Technology Recommendations

The choice of programming language significantly impacts implementation complexity and learning outcomes. Each language provides different levels of abstraction and safety guarantees for network programming:

| Component | Beginner-Friendly Option | Advanced Option |
|-----------|-------------------------|-----------------|
| Socket Programming | C with BSD sockets (manual memory management) | Go with net package (garbage collected, safer) |
| HTTP Parsing | Manual string parsing with explicit state machine | Rust with nom parser combinator library |
| Concurrency | pthread library with manual thread management | Go with goroutines and channels |
| File I/O | POSIX file operations (open/read/write/close) | Rust with async I/O and proper error types |
| Configuration | Command-line arguments with getopt | TOML/YAML configuration files with validation |
| Logging | Printf to stdout/stderr | Structured logging with log levels and rotation |

**C Language Considerations**: C provides the most direct exposure to underlying system concepts, making it excellent for educational purposes. However, C requires manual memory management, explicit error checking, and careful buffer management. Common C pitfalls include buffer overflows, memory leaks, and format string vulnerabilities. The lack of built-in string handling means implementing HTTP parsing requires careful attention to buffer boundaries.

**Go Language Benefits**: Go simplifies many network programming challenges through garbage collection, built-in string types, and comprehensive standard library networking support. The `net/http` package provides high-level abstractions, but implementing from scratch using `net` package sockets still exposes core concepts while providing memory safety. Go's goroutines make concurrent programming more approachable than manual thread management.

**Rust Language Advantages**: Rust provides memory safety without garbage collection and excellent error handling through the `Result` type system. However, Rust's ownership system can create learning obstacles for developers new to the language. The async ecosystem in Rust is powerful but complex for educational purposes.

#### Essential C Socket Programming Foundation

For C implementation, several fundamental patterns appear throughout HTTP server code. Understanding these patterns prevents common mistakes:

**Socket Creation and Error Handling Pattern**:
```c
// Socket creation with proper error handling
int create_server_socket(int port) {
    int sockfd;
    struct sockaddr_in server_addr;
    int opt = 1;
    
    // TODO: Create socket with AF_INET, SOCK_STREAM, 0
    // TODO: Check if socket creation failed (returns -1)
    // TODO: Set SO_REUSEADDR option to avoid "Address already in use" errors
    // TODO: Configure server_addr structure with AF_INET, INADDR_ANY, htons(port)
    // TODO: Bind socket to address, check for errors
    // TODO: Start listening with reasonable backlog (e.g., 128)
    // TODO: Return socket file descriptor
    
    // Error handling: close socket and return -1 on any failure
}
```

**Connection Accept Loop Pattern**:
```c
// Main server loop accepting connections
void server_main_loop(int server_fd) {
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    int client_fd;
    
    while (server_running) {
        // TODO: Accept incoming connection
        // TODO: Handle accept errors (EINTR should retry, others should log and continue)
        // TODO: For thread-per-connection: create thread for handle_client(client_fd)
        // TODO: For sequential: call handle_client(client_fd) directly
        // TODO: Ensure client_fd is closed in all code paths
    }
}
```

**Safe Read/Write Operations**:
```c
// Read complete HTTP request handling partial reads
ssize_t read_complete_request(int sockfd, char *buffer, size_t max_size) {
    size_t total_read = 0;
    ssize_t bytes_read;
    
    while (total_read < max_size - 1) {
        // TODO: Read available data into buffer + total_read
        // TODO: Handle read errors (EAGAIN/EWOULDBLOCK for non-blocking sockets)
        // TODO: Handle connection close (bytes_read == 0)
        // TODO: Update total_read counter
        // TODO: Check for complete HTTP request (look for \r\n\r\n)
        // TODO: Break when complete request found
    }
    
    // TODO: Null-terminate buffer and return total bytes read
}
```

#### HTTP Message Structure Reference

Understanding HTTP message format is crucial for parser implementation:

| HTTP Request Component | Format | Example | Parsing Notes |
|------------------------|---------|---------|---------------|
| Request Line | METHOD URI HTTP/1.1\r\n | GET /index.html HTTP/1.1\r\n | Split on spaces, validate each component |
| Header Lines | Name: Value\r\n | Content-Type: text/html\r\n | Split on first colon, trim whitespace |
| Header End | \r\n | \r\n | Indicates end of headers, body follows |
| Message Body | Raw bytes | File contents or form data | Length specified by Content-Length header |

**Common HTTP Methods to Support**:
- GET: Retrieve resource, no request body
- HEAD: Like GET but response body omitted
- POST: Submit data, request body contains data
- OPTIONS: Query server capabilities
- Return 405 Method Not Allowed for unsupported methods

**Essential Response Status Codes**:
- 200 OK: Successful GET request
- 404 Not Found: Requested file doesn't exist
- 400 Bad Request: Malformed HTTP request
- 405 Method Not Allowed: Unsupported HTTP method
- 500 Internal Server Error: Server-side error occurred

#### File System Security Patterns

Preventing directory traversal attacks requires careful path validation:

```c
// Secure path resolution preventing directory traversal
int validate_and_resolve_path(const char *document_root, const char *request_path, 
                            char *resolved_path, size_t max_path_len) {
    // TODO: Remove leading slash from request_path if present
    // TODO: Decode URL-encoded characters in request_path
    // TODO: Join document_root and request_path safely
    // TODO: Resolve symbolic links and relative paths (realpath())
    // TODO: Verify resolved path starts with document_root
    // TODO: Return error if path traversal detected
    // TODO: Check if resolved path exists and is readable
    // TODO: Copy resolved path to output buffer
}
```

**MIME Type Detection**:
Common file extension to Content-Type mappings:
- .html, .htm → text/html
- .css → text/css
- .js → application/javascript
- .png → image/png
- .jpg, .jpeg → image/jpeg
- .gif → image/gif
- Unknown extensions → application/octet-stream

#### Development and Testing Setup

**Compilation flags for development**:
```bash
# Development build with debugging symbols and warnings
gcc -Wall -Wextra -g -DDEBUG -o httpserver *.c

# Production build with optimizations
gcc -O2 -DNDEBUG -o httpserver *.c
```

**Testing with command-line tools**:
```bash
# Test basic connectivity
telnet localhost 8080

# Send manual HTTP request
printf "GET / HTTP/1.1\r\nHost: localhost\r\n\r\n" | nc localhost 8080

# Test with curl
curl -v http://localhost:8080/

# Test concurrent connections
ab -n 100 -c 10 http://localhost:8080/
```

**Debugging techniques**:
- Use `strace` to monitor system calls: `strace -e trace=network,file ./httpserver`
- Monitor network traffic: `tcpdump -i lo port 8080`
- Check file descriptor usage: `lsof -p $(pidof httpserver)`
- Memory leak detection: `valgrind --leak-check=full ./httpserver`

#### Project Structure Organization

Recommended file organization for maintainable HTTP server implementation:

```
httpserver/
├── src/
│   ├── main.c                 ← Entry point, argument parsing
│   ├── tcp_server.c           ← Socket creation, connection accept loop
│   ├── tcp_server.h           ← TCP server interface definitions
│   ├── http_parser.c          ← HTTP request/response parsing
│   ├── http_parser.h          ← HTTP data structures and functions
│   ├── file_handler.c         ← Static file serving, MIME types
│   ├── file_handler.h         ← File operations interface
│   ├── worker_pool.c          ← Thread pool management (optional)
│   ├── worker_pool.h          ← Concurrency management interface
│   └── utils.c                ← Common utilities (logging, string ops)
├── tests/
│   ├── test_http_parser.c     ← Unit tests for HTTP parsing
│   ├── test_file_handler.c    ← Unit tests for file operations
│   └── integration_test.sh    ← End-to-end server tests
├── www/                       ← Document root for testing
│   ├── index.html
│   ├── styles.css
│   └── images/
├── Makefile                   ← Build configuration
└── README.md                  ← Usage instructions
```

This organization separates concerns clearly, making it easier to implement and test each component independently while maintaining clean interfaces between components.


## Goals and Non-Goals

> **Milestone(s):** Foundation for all milestones - defines scope boundaries and implementation priorities throughout the project

The goals and non-goals section serves as our project compass, establishing clear boundaries between what we will build and what we deliberately exclude. Think of this like **planning a camping trip** - you need to decide what essential gear to pack (functional requirements), what level of comfort you're targeting (non-functional requirements), and what luxury items to leave behind despite being tempting (explicit non-goals). Just as a successful camping trip requires balancing necessity against pack weight, our HTTP server must balance educational value against implementation complexity.

This educational HTTP server prioritizes **deep understanding of core networking and concurrency concepts** over comprehensive HTTP feature coverage. We're building a production-quality foundation that demonstrates proper socket programming, HTTP protocol handling, and concurrent connection management, while deliberately excluding advanced features that would obscure these fundamental lessons.

The scope decisions made here directly impact our architectural choices throughout the project. Each functional requirement drives specific component responsibilities, each non-functional requirement influences our concurrency model selection, and each explicit non-goal prevents scope creep that could derail the learning objectives.

### Functional Requirements

The functional requirements define the **core HTTP/1.1 capabilities** that our server must implement to qualify as a working static file server. These requirements map directly to our four project milestones and establish the minimum viable functionality needed for real-world HTTP communication.

**TCP Connection Management** forms the foundation of our server architecture. The server must bind to a configurable port address and maintain a stable listening socket throughout its lifecycle. The connection acceptance process must handle the complete TCP handshake sequence, managing the transition from listening socket to established client connections. Each accepted connection requires proper resource allocation and cleanup to prevent file descriptor leaks over extended operation periods.

| Requirement | Acceptance Criteria | Milestone |
|-------------|-------------------|-----------|
| Port Binding | Bind to configurable port (default 8080) using `AF_INET` and `SOCK_STREAM` | 1 |
| Connection Acceptance | Accept multiple sequential and concurrent client connections | 1, 4 |
| Socket Resource Management | Properly close client file descriptors and free associated memory | 1 |
| Connection Lifecycle | Handle complete TCP connection establishment and teardown | 1 |

**HTTP Protocol Compliance** ensures our server correctly interprets and responds to HTTP/1.1 messages according to RFC specifications. The request parsing component must handle the three-part HTTP message structure: request line, headers, and optional body. Response generation must include proper status codes, required headers, and correctly formatted message structure.

| Requirement | Acceptance Criteria | Milestone |
|-------------|-------------------|-----------|
| Request Line Parsing | Extract method, path, and HTTP version from first request line | 2 |
| Header Processing | Parse headers into key-value pairs with whitespace normalization | 2 |
| GET Method Support | Handle GET requests with appropriate response generation | 2 |
| HTTP Response Format | Generate responses with status line, headers, and body | 2 |
| Host Header Extraction | Parse Host header for virtual host routing capability | 2 |
| Error Response Generation | Return appropriate 4xx/5xx responses for invalid requests | 2 |

**Static File Serving** capabilities transform URL paths into filesystem operations while maintaining security boundaries. The path resolution system must safely map HTTP request paths to filesystem locations within a configured document root directory. Content delivery includes proper MIME type detection and binary file handling.

| Requirement | Acceptance Criteria | Milestone |
|-------------|-------------------|-----------|
| Document Root Mapping | Map URL paths to filesystem paths within configured root directory | 3 |
| File Content Delivery | Read and serve file contents with correct Content-Length header | 3 |
| MIME Type Detection | Set Content-Type header based on file extension analysis | 3 |
| 404 Not Found Handling | Return 404 responses for non-existent files with proper error page | 3 |
| Directory Traversal Prevention | Reject requests containing `../` sequences or other path escape attempts | 3 |
| Directory Listing | Display directory contents when request path maps to directory | 3 |
| Binary File Support | Correctly serve binary files without content corruption | 3 |

**Concurrent Connection Handling** enables the server to process multiple client requests simultaneously without blocking. The concurrency implementation must prevent one slow client from stalling service to other clients. Resource management becomes critical under concurrent load to prevent memory exhaustion and file descriptor depletion.

| Requirement | Acceptance Criteria | Milestone |
|-------------|-------------------|-----------|
| Thread-per-Connection Model | Create dedicated thread for each client connection | 4 |
| Thread Pool Option | Limit concurrent threads to prevent resource exhaustion | 4 |
| Non-blocking I/O Option | Handle multiple connections using select/poll event-driven model | 4 |
| Graceful Shutdown | Complete in-flight requests before server termination | 4 |
| Connection Limits | Reject new connections when resource limits are reached | 4 |
| Thread Safety | Ensure shared resources are properly synchronized between threads | 4 |

> **Design Insight**: These functional requirements intentionally focus on **depth over breadth**. Rather than implementing dozens of HTTP features superficially, we implement core networking, parsing, file serving, and concurrency concepts thoroughly. This approach ensures learners understand the foundational principles that apply to any networked service, not just HTTP servers.

**Configuration Management** provides runtime flexibility without requiring recompilation. The server must accept configuration parameters for port binding, document root directory, thread pool sizes, and connection limits. Configuration validation ensures the server fails fast with clear error messages rather than exhibiting undefined behavior with invalid settings.

| Configuration Parameter | Type | Default Value | Validation |
|------------------------|------|---------------|------------|
| Listen Port | Integer | 8080 | Range 1-65535, not in use |
| Document Root | String | "./public" | Directory exists and readable |
| Max Threads | Integer | 10 | Range 1-1000 |
| Max Connections | Integer | 100 | Range 1-10000 |
| Request Timeout | Integer | 30 seconds | Range 1-300 seconds |

### Non-Functional Requirements

The non-functional requirements establish **performance and reliability characteristics** appropriate for an educational implementation. These requirements balance realistic expectations with learning objectives, ensuring the server exhibits production-like behavior without requiring production-scale optimization.

**Performance Characteristics** define acceptable response times and throughput under typical educational workloads. The server should handle moderate concurrent load gracefully while providing predictable response latency. These targets ensure the implementation feels responsive during testing and demonstration while remaining achievable with straightforward algorithms.

| Performance Metric | Target Value | Measurement Method | Rationale |
|-------------------|--------------|-------------------|-----------|
| Response Latency | < 100ms for small files | Time between request completion and response start | Human-perceptible responsiveness |
| Concurrent Connections | 50+ simultaneous clients | Load testing with multiple curl instances | Demonstrates effective concurrency |
| File Size Limit | 100MB maximum | Server rejects larger files with 413 response | Prevents memory exhaustion |
| Request Rate | 100+ requests/second | Sustained load testing | Shows proper connection management |
| Memory Usage | < 50MB under typical load | Process monitoring during operation | Ensures resource efficiency |

**Reliability Requirements** focus on **graceful degradation** and predictable failure modes rather than absolute fault tolerance. The server should detect error conditions promptly and respond with appropriate HTTP status codes rather than crashing or hanging indefinitely.

| Reliability Aspect | Requirement | Detection Method | Recovery Action |
|-------------------|-------------|------------------|-----------------|
| Malformed Request Handling | Return 400 Bad Request for parsing failures | HTTP parser validation | Send error response and close connection |
| File System Errors | Return 500 Internal Server Error for I/O failures | File operation return codes | Log error and return HTTP error response |
| Connection Timeout | Close idle connections after timeout period | Timer-based monitoring | Clean up resources and close socket |
| Resource Exhaustion | Reject new connections when limits reached | Connection counting | Return 503 Service Unavailable |
| Thread Safety | No data corruption under concurrent access | Stress testing | Use appropriate synchronization primitives |

> **Decision: Educational Performance Targets**
> - **Context**: Need to balance realistic performance expectations with educational simplicity
> - **Options Considered**: Production-scale targets (1000s RPS), minimal targets (10 RPS), moderate targets (100 RPS)
> - **Decision**: Moderate performance targets requiring proper concurrency but not extreme optimization
> - **Rationale**: High enough to demonstrate the value of concurrent design, low enough to achieve with straightforward implementation
> - **Consequences**: Enables meaningful performance testing while keeping code complexity manageable for learning

**Portability Requirements** ensure the server builds and runs correctly across common development environments. The implementation should use POSIX-standard networking APIs where possible, with clear documentation of any platform-specific requirements.

| Platform Aspect | Requirement | Implementation Strategy |
|------------------|-------------|------------------------|
| Operating System | Support Linux, macOS, and Windows | Use standard socket APIs with platform-specific compilation |
| Compiler | Build with GCC, Clang, and MSVC | Standard C99 code with minimal extensions |
| Dependencies | Minimize external library requirements | Use system libraries and standard library functions |
| Configuration | Support command-line arguments and config files | Standard argument parsing and file I/O |

**Security Requirements** implement **fundamental security practices** without attempting comprehensive security coverage. The focus remains on preventing basic attacks that could compromise the host system while maintaining code clarity for educational purposes.

| Security Aspect | Requirement | Implementation Approach |
|------------------|-------------|------------------------|
| Directory Traversal Prevention | Block `../` path escape attempts | Path validation and canonical path resolution |
| Buffer Overflow Prevention | Bounds checking on all string operations | Fixed-size buffers with length validation |
| Resource Exhaustion Protection | Limit concurrent connections and memory usage | Connection counting and request size limits |
| Input Validation | Reject malformed HTTP requests | Strict parsing with early rejection |

### Explicit Non-Goals

The explicit non-goals prevent **scope creep** by clearly stating advanced HTTP features and optimizations that we deliberately exclude. These exclusions maintain focus on core learning objectives while acknowledging that production HTTP servers require additional capabilities.

**Advanced HTTP Protocol Features** represent significant implementation complexity that would overshadow our core networking and concurrency lessons. While these features are essential for production HTTP servers, they introduce parsing complexity, state management challenges, and protocol intricacies that distract from fundamental concepts.

| Excluded Feature | Complexity Reason | Learning Impact |
|------------------|-------------------|-----------------|
| HTTP/2 and HTTP/3 | Binary framing, multiplexing, stream management | Obscures basic HTTP message structure |
| Chunked Transfer Encoding | Complex parsing state machine, streaming assembly | Complicates request/response handling |
| Content Compression | Compression algorithm integration, negotiation | Adds processing overhead to core flow |
| Range Requests | Partial content parsing, byte range calculations | Distracts from basic file serving |
| WebSocket Protocol | Protocol upgrade negotiation, framing, persistence | Different paradigm from request/response |
| HTTP Caching | Cache validation, ETags, conditional requests | Complex metadata management |

⚠️ **Pitfall: Feature Scope Creep**
Learners often attempt to add these advanced features during implementation, thinking they're "simple additions." However, each feature significantly increases code complexity and introduces new failure modes. For example, chunked transfer encoding requires maintaining parsing state across multiple `read()` calls, handling chunk size parsing, and assembling fragmented data - this complexity obscures the core lesson about HTTP message structure.

**Dynamic Content Generation** capabilities like server-side scripting, template processing, and database integration represent entirely different problem domains. While valuable for web application development, these features shift focus away from systems programming concepts toward application development.

| Excluded Capability | Rationale | Alternative Learning Path |
|---------------------|-----------|-------------------------|
| CGI/FastCGI Integration | Process management and IPC complexity | Separate process management project |
| Server-Side Scripting | Language runtime integration | Web application framework study |
| Template Engine | String processing and substitution logic | Text processing project |
| Database Connectivity | Database protocol and connection pooling | Database systems course |
| Session Management | State persistence and security | Web application security study |

**Production-Scale Optimizations** involve sophisticated algorithms and data structures that, while important for high-performance servers, add implementation complexity without enhancing understanding of core concepts. These optimizations represent performance engineering rather than foundational systems programming.

| Excluded Optimization | Implementation Complexity | Core Concept Impact |
|-----------------------|---------------------------|-------------------|
| Connection Pooling | Connection lifecycle management, pool algorithms | Obscures basic connection handling |
| Epoll/Kqueue Event Systems | Platform-specific APIs, event dispatching | Complicates concurrency model comparison |
| Zero-Copy I/O | Platform-specific sendfile(), memory mapping | Distracts from basic file I/O understanding |
| CPU Affinity Tuning | Scheduler interaction, NUMA awareness | Beyond scope of basic networking |
| Load Balancing | Request distribution algorithms, health checking | Separate distributed systems topic |
| Memory Pool Allocation | Custom memory management, fragmentation handling | Separate memory management study |

> **Decision: Static Files Only**
> - **Context**: Must choose between static file serving versus dynamic content generation
> - **Options Considered**: Static only, CGI support, embedded scripting, full application server
> - **Decision**: Static file serving exclusively
> - **Rationale**: Dynamic content requires process management, security sandboxing, and application-specific logic that overshadows networking fundamentals
> - **Consequences**: Enables focus on HTTP protocol, file I/O, and concurrency without application development complexity

**Enterprise Integration Features** involve external system interactions and operational concerns that extend beyond the core HTTP server implementation. While critical for production deployments, these features represent separate problem domains with their own complexity.

| Integration Feature | Scope Reason | Separate Learning Context |
|--------------------|--------------|-------------------------|
| SSL/TLS Support | Cryptographic protocol complexity | Network security course |
| Authentication Systems | Identity management, credential validation | Security and identity systems |
| Logging and Monitoring | Structured logging, metrics collection, alerting | Operations and observability |
| Configuration Management | Hot reload, distributed configuration | Configuration systems design |
| Health Check Endpoints | Service discovery, load balancer integration | Distributed systems architecture |
| Reverse Proxy Features | Request forwarding, load distribution | Proxy and gateway systems |

**Platform-Specific Optimizations** tie the implementation to particular operating systems or hardware architectures. While these optimizations provide significant performance benefits in production, they reduce code portability and introduce platform-specific complexity that distracts from universal networking concepts.

| Platform Feature | Exclusion Reason |
|------------------|------------------|
| Linux-specific epoll | Reduces portability, complicates event loop |
| Windows IOCP | Platform-specific async I/O model |
| macOS kqueue | BSD-specific event notification |
| Linux sendfile() | Platform-specific zero-copy optimization |
| Memory-mapped files | Complex memory management, platform differences |
| CPU-specific SIMD | Processor-specific optimization |

> **Key Insight**: These explicit non-goals aren't permanent limitations - they represent **learning prioritization**. Each excluded feature could become the focus of a separate advanced project once learners master the foundational concepts. The goal is depth-first learning rather than breadth-first feature coverage.

By maintaining strict boundaries around our implementation scope, we ensure that each line of code written serves the core educational objectives: understanding network programming, HTTP protocol handling, file system interaction, and concurrent request processing. Every feature addition must pass the test: "Does this enhance understanding of fundamental systems programming concepts, or does it distract from them?"

### Implementation Guidance

The implementation approach balances **educational clarity** with production-quality practices, ensuring learners develop good habits while maintaining focus on core concepts.

**A. Technology Recommendations Table:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Socket Programming | POSIX sockets with blocking I/O | Non-blocking sockets with select/poll |
| Threading Model | pthread_create per connection | Thread pool with work queue |
| HTTP Parsing | String manipulation with strtok | State machine parser |
| Configuration | Command-line arguments only | Config file + command-line override |
| Logging | printf to stdout/stderr | Structured logging with levels |
| Build System | Simple Makefile | Autotools or CMake |

**B. Recommended File/Module Structure:**

```
http-server/
├── src/
│   ├── main.c                 ← Entry point and configuration
│   ├── tcp_server.c           ← TCP socket management (Milestone 1)
│   ├── tcp_server.h
│   ├── http_parser.c          ← HTTP request/response parsing (Milestone 2)
│   ├── http_parser.h
│   ├── file_handler.c         ← Static file serving (Milestone 3)
│   ├── file_handler.h
│   ├── concurrency.c          ← Connection management (Milestone 4)
│   ├── concurrency.h
│   └── common.h               ← Shared constants and utilities
├── public/                    ← Default document root
│   ├── index.html
│   └── test.txt
├── tests/                     ← Test scripts and data
├── Makefile
└── README.md
```

**C. Infrastructure Starter Code:**

Complete configuration management system ready for use:

```c
// common.h - Shared definitions and configuration
#include <sys/socket.h>
#include <netinet/in.h>
#include <pthread.h>

#define MAX_REQUEST_SIZE 8192
#define MAX_PATH_LENGTH 1024
#define MAX_RESPONSE_SIZE 16384
#define DEFAULT_PORT 8080
#define DEFAULT_DOC_ROOT "./public"
#define DEFAULT_MAX_THREADS 10

typedef struct {
    int port;
    char document_root[MAX_PATH_LENGTH];
    int max_threads;
    int max_connections;
    int request_timeout_sec;
} ServerConfig;

typedef struct {
    char method[16];
    char path[MAX_PATH_LENGTH];
    char version[16];
    char headers[32][2][256];  // [header_count][key/value][string]
    int header_count;
    char *body;
    size_t body_length;
} HTTPRequest;

typedef struct {
    int status_code;
    char status_text[64];
    char headers[32][2][256];  // [header_count][key/value][string]
    int header_count;
    char *body;
    size_t body_length;
} HTTPResponse;

typedef struct {
    int client_fd;
    struct sockaddr_in client_addr;
    time_t connect_time;
    pthread_t thread_id;
} ConnectionContext;

// Configuration parsing utility - complete implementation
int parse_config(int argc, char *argv[], ServerConfig *config) {
    // Set defaults
    config->port = DEFAULT_PORT;
    strncpy(config->document_root, DEFAULT_DOC_ROOT, MAX_PATH_LENGTH - 1);
    config->max_threads = DEFAULT_MAX_THREADS;
    config->max_connections = 100;
    config->request_timeout_sec = 30;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            config->port = atoi(argv[++i]);
            if (config->port < 1 || config->port > 65535) {
                fprintf(stderr, "Invalid port: %d\n", config->port);
                return -1;
            }
        } else if (strcmp(argv[i], "--root") == 0 && i + 1 < argc) {
            strncpy(config->document_root, argv[++i], MAX_PATH_LENGTH - 1);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            config->max_threads = atoi(argv[++i]);
        }
    }
    
    return 0;
}
```

**D. Core Logic Skeleton Code:**

```c
// tcp_server.c - Core socket operations for learner implementation
#include "tcp_server.h"

/**
 * Creates and configures a server socket bound to the specified port.
 * Sets SO_REUSEADDR to allow rapid server restarts.
 * Returns: socket file descriptor on success, -1 on failure
 */
int create_server_socket(int port) {
    // TODO 1: Create socket using AF_INET and SOCK_STREAM
    // TODO 2: Set SO_REUSEADDR socket option to allow address reuse
    // TODO 3: Configure sockaddr_in structure with INADDR_ANY and port
    // TODO 4: Use htons() to convert port to network byte order
    // TODO 5: Bind socket to address using bind() system call
    // TODO 6: Start listening with backlog of 10 connections using listen()
    // TODO 7: Return socket file descriptor
    // Hint: Check each system call return value and use perror() for errors
}

/**
 * Reads a complete HTTP request from client socket, handling partial reads.
 * Continues reading until complete request is received or error occurs.
 * Returns: number of bytes read on success, -1 on error, 0 on connection close
 */
ssize_t read_complete_request(int sockfd, char *buffer, size_t max_size) {
    // TODO 1: Initialize total bytes read counter
    // TODO 2: Loop until complete request received or error
    // TODO 3: Use recv() to read available data into buffer
    // TODO 4: Handle partial reads by updating buffer position
    // TODO 5: Check for HTTP request end marker (\r\n\r\n)
    // TODO 6: Handle connection close (recv returns 0)
    // TODO 7: Handle errors (recv returns -1) with appropriate errno checking
    // TODO 8: Prevent buffer overflow by checking max_size limit
    // Hint: Use strstr() to find HTTP header end sequence
}

/**
 * Main server loop accepting and dispatching client connections.
 * Runs until interrupted or fatal error occurs.
 */
void server_main_loop(int server_fd) {
    // TODO 1: Initialize connection tracking structures
    // TODO 2: Set up signal handlers for graceful shutdown
    // TODO 3: Main accept() loop - wait for client connections
    // TODO 4: For each accepted connection, create ConnectionContext
    // TODO 5: Dispatch connection to appropriate handler (thread/process/event)
    // TODO 6: Handle accept() errors and temporary failures (EAGAIN, EMFILE)
    // TODO 7: Implement graceful shutdown - complete active requests
    // TODO 8: Clean up all resources before exit
    // Hint: Use accept() in a loop, check errno on failures
}
```

**E. Language-Specific Hints:**

- **Socket Creation**: Use `socket(AF_INET, SOCK_STREAM, 0)` for TCP sockets
- **Address Reuse**: `setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))` prevents "Address already in use" errors
- **Network Byte Order**: Always use `htons()` for port numbers and `htonl()` for IP addresses
- **Partial Reads**: Network `recv()` calls may return fewer bytes than requested - always check return value and loop if needed
- **Error Handling**: Check system call return values; use `perror()` or `strerror(errno)` for descriptive error messages
- **Signal Safety**: Use `sigaction()` instead of `signal()` for reliable signal handling
- **Thread Safety**: Protect shared data structures with `pthread_mutex_t`; use `pthread_create()` for thread-per-connection model

**F. Milestone Checkpoint:**

After implementing the goals and scope definition:
1. **Document Review**: Ensure all team members understand the scope boundaries
2. **Setup Verification**: Confirm development environment can compile and run basic C socket programs
3. **Architecture Validation**: Verify the planned file structure makes sense for your team's workflow
4. **Scope Verification**: List any features you're tempted to add and confirm they belong in the non-goals section

Expected behavior after scope definition:
- Clear understanding of what success looks like for each milestone
- Agreement on which features to implement vs. defer to future projects
- Development environment setup and ready for Milestone 1 implementation
- Architectural decisions documented for future reference

**G. Common Scope Management Pitfalls:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Implementation taking much longer than expected | Feature creep beyond defined scope | Review code against functional requirements | Remove features not in requirements |
| Code becoming extremely complex | Attempting production optimizations | Check against explicit non-goals | Simplify to meet educational objectives |
| Difficulty testing basic functionality | Too many features implemented simultaneously | Verify milestone progression | Focus on one milestone at a time |
| Team disagreement on implementation approach | Unclear architectural decisions | Review decision records | Document additional ADRs as needed |

The key to successful scope management is **disciplined focus** on the learning objectives. When in doubt, choose the simpler approach that most clearly demonstrates the core concept being taught.


## High-Level Architecture

> **Milestone(s):** Foundation for Milestones 1-4 - provides the structural blueprint that guides implementation across TCP server basics, HTTP parsing, file serving, and concurrency management

The high-level architecture section serves as our blueprint for understanding how the HTTP server's components work together to transform raw TCP connections into meaningful HTTP responses. Think of this architecture as a **restaurant's service flow** - just as a restaurant has distinct stations (host desk, kitchen, servers, manager) that coordinate to serve customers, our HTTP server has specialized components that each handle a specific aspect of client requests while maintaining clear boundaries and communication patterns.

### Component Overview

The HTTP server architecture centers around four primary components, each with distinct responsibilities and clear interfaces. This separation follows the **single responsibility principle** - each component has one primary job and does it well, making the system easier to understand, test, and maintain.

![HTTP Server System Overview](./diagrams/system-overview.svg)

The **TCP Server Component** acts as the front door of our system, similar to a hotel's reception desk. It handles all the low-level networking concerns: creating sockets, binding to network addresses, listening for incoming connections, and accepting client requests. This component shields the rest of the system from the complexities of socket programming and network protocol details. Its primary responsibility is managing the **connection lifecycle** - from initial socket creation through connection acceptance to resource cleanup.

The **HTTP Parser Component** serves as the system's translator, converting raw byte streams from TCP connections into structured HTTP message objects. Think of it as a postal worker who takes incoming mail and sorts it into organized bins - the parser takes the raw request bytes and extracts the HTTP method, URL path, headers, and body content into well-defined data structures. This component handles all the intricacies of HTTP protocol parsing, including line ending variations, header formatting, and body content processing.

The **File Handler Component** functions as the system's librarian, responsible for mapping HTTP URL paths to actual files on the filesystem and serving their contents safely. Like a librarian who helps patrons find books while ensuring they don't access restricted sections, this component performs path resolution, security validation, MIME type detection, and file content delivery. It serves as the critical security boundary that prevents directory traversal attacks and ensures clients can only access files within the designated document root.

The **Concurrency Management Component** orchestrates how the server handles multiple simultaneous client connections. Think of it as a restaurant manager who decides whether to assign one waiter per table, maintain a pool of waiters who serve multiple tables, or have a single super-efficient waiter handle all tables. This component implements different concurrency models (thread-per-connection, thread pool, or event-driven) and manages system resources to prevent overload while maintaining responsiveness.

| Component | Primary Responsibility | Key Abstractions | Dependencies |
|-----------|------------------------|------------------|--------------|
| TCP Server | Network connection lifecycle | Socket management, connection acceptance | Operating system sockets |
| HTTP Parser | Protocol message translation | Request/response structures, state machine | TCP byte streams |
| File Handler | Static content delivery | Path resolution, MIME detection | Filesystem, security validation |
| Concurrency Manager | Multi-client coordination | Thread/event management, resource limits | Threading primitives, I/O multiplexing |

Each component maintains clear boundaries through well-defined interfaces. The TCP Server produces raw client connections and byte streams. The HTTP Parser consumes these streams and produces structured `HTTPRequest` objects. The File Handler consumes requests and produces `HTTPResponse` objects with file content. The Concurrency Manager orchestrates these interactions across multiple simultaneous clients.

> **Key Design Insight**: The component separation allows us to test each piece independently and swap implementations without affecting other parts. For example, we can start with a simple thread-per-connection model in the Concurrency Manager and later upgrade to an event-driven model without touching the parsing or file serving logic.

### Recommended File Structure

Organizing code into a logical file structure is crucial for maintainability and understanding. The recommended structure follows C programming best practices, separating interface definitions from implementations and grouping related functionality together.

```
http-server/
├── src/
│   ├── main.c                 ← Entry point, argument parsing, server startup
│   ├── server.h               ← Main server interface and configuration
│   ├── server.c               ← Server initialization and main loop coordination
│   ├── tcp/
│   │   ├── tcp_server.h       ← TCP server component interface
│   │   ├── tcp_server.c       ← Socket creation, binding, acceptance logic
│   │   └── connection.h       ← Connection context and lifecycle definitions
│   ├── http/
│   │   ├── http_parser.h      ← HTTP parsing interface and data structures
│   │   ├── http_parser.c      ← Request/response parsing implementation
│   │   ├── http_request.h     ← HTTPRequest structure and utilities
│   │   ├── http_response.h    ← HTTPResponse structure and utilities
│   │   └── http_status.c      ← HTTP status code definitions and utilities
│   ├── file/
│   │   ├── file_handler.h     ← File serving interface and security functions
│   │   ├── file_handler.c     ← Path resolution and file serving logic
│   │   ├── mime_types.h       ← MIME type detection interface
│   │   └── mime_types.c       ← File extension to MIME type mapping
│   ├── concurrency/
│   │   ├── connection_manager.h  ← Concurrency management interface
│   │   ├── thread_pool.c      ← Thread pool implementation
│   │   ├── event_loop.c       ← Event-driven I/O implementation
│   │   └── worker_thread.c    ← Individual connection handler logic
│   └── util/
│       ├── logger.h           ← Logging interface and macros
│       ├── logger.c           ← Logging implementation
│       ├── buffer.h           ← Dynamic buffer management
│       └── buffer.c           ← Buffer allocation and manipulation
├── config/
│   ├── server.conf            ← Default server configuration
│   └── mime.types             ← MIME type mappings
├── www/                       ← Default document root directory
│   ├── index.html            ← Default homepage
│   ├── 404.html              ← Custom 404 error page
│   └── static/               ← Static assets (CSS, JS, images)
├── tests/
│   ├── unit/                 ← Unit tests for individual components
│   ├── integration/          ← Integration tests for component interaction
│   └── performance/          ← Load testing and benchmarking
├── docs/
│   ├── README.md             ← Project documentation
│   └── API.md                ← Component interface documentation
└── Makefile                  ← Build configuration and targets
```

This structure provides several important benefits for the learning process. The **component isolation** allows students to focus on one aspect of the server at a time - they can work on TCP networking without worrying about HTTP parsing, or implement file serving without getting bogged down in concurrency details. The **header/implementation separation** follows C best practices and makes it easy to understand interfaces before diving into implementation details.

The **progressive complexity** arrangement lets students implement components in dependency order. They start with basic TCP server functionality, then add HTTP parsing, followed by file serving, and finally concurrency management. Each milestone builds naturally on the previous ones without requiring major refactoring.

> **Learning Insight**: The file structure mirrors the component architecture - each major directory corresponds to one of our four main components. This makes it easy to find code related to specific functionality and reinforces the architectural boundaries in the actual codebase.

### Request Processing Flow

The request processing flow describes how our four components collaborate to handle a complete HTTP request-response cycle. Understanding this flow is essential because it shows how the individual components we'll implement in later milestones work together to create a functioning web server.

![Request Processing Sequence](./diagrams/request-processing-flow.svg)

The flow begins when a client establishes a TCP connection to our server. The **TCP Server Component** receives this connection through the standard socket lifecycle: `socket()` creates the server socket, `bind()` attaches it to a specific port, `listen()` marks it as accepting connections, and `accept()` blocks until a client connects. Once a connection arrives, `accept()` returns a new client file descriptor representing the dedicated communication channel with that specific client.

At this point, the **Concurrency Management Component** takes control of the newly accepted connection. Depending on the configured concurrency model, it either spawns a new thread to handle this client, assigns the connection to a worker thread from a pre-allocated pool, or adds the client socket to an event loop for non-blocking processing. This decision affects performance characteristics but doesn't change the fundamental processing steps that follow.

The connection handler begins by reading the HTTP request data from the client socket. This involves calling `read()` or `recv()` repeatedly until a complete HTTP request is received. The challenge here is handling **partial reads** - network operations may return fewer bytes than requested, so the server must buffer incoming data and continue reading until it receives the complete request headers (indicated by a blank line) and any request body.

Once complete request data is available, the **HTTP Parser Component** takes over. It processes the raw bytes according to HTTP/1.1 protocol rules, extracting the request line (method, path, HTTP version), parsing headers into key-value pairs, and handling any request body content. The parser validates the request format and populates an `HTTPRequest` structure with the parsed information. If parsing fails due to malformed input, the parser generates an appropriate HTTP error response (400 Bad Request) without proceeding to file serving.

With a valid `HTTPRequest` in hand, the **File Handler Component** begins the process of mapping the requested URL path to actual file content. This involves several security-critical steps: validating the request path to prevent directory traversal attacks, resolving the path relative to the configured document root, checking file permissions and existence, and determining the appropriate MIME type based on the file extension.

If the requested file exists and is accessible, the File Handler reads its contents and constructs an `HTTPResponse` object with appropriate headers (Content-Type, Content-Length, Last-Modified) and the file data as the response body. If the file doesn't exist, it generates a 404 Not Found response. If security validation fails (such as detecting a directory traversal attempt), it returns a 403 Forbidden response.

The final step involves the connection handler sending the complete HTTP response back to the client. This requires formatting the `HTTPResponse` object into valid HTTP protocol format (status line, headers, blank line, body) and writing all the data to the client socket using `write()` or `send()` operations. Like reading, writing may require multiple system calls to send all response data, especially for large files.

After successful response delivery, the connection handler performs cleanup operations: closing the client file descriptor to free the network connection, releasing any allocated memory for request/response processing, and updating connection statistics or logs. The handler then either terminates (in thread-per-connection model) or returns to process the next connection (in thread pool or event-driven models).

| Processing Stage | Component Responsible | Key Operations | Potential Failures |
|------------------|----------------------|----------------|--------------------|
| Connection Accept | TCP Server | `socket()`, `bind()`, `listen()`, `accept()` | Port already bound, network errors |
| Concurrency Dispatch | Concurrency Manager | Thread creation or event registration | Resource exhaustion, thread limits |
| Request Reading | Connection Handler | `read()` with partial read handling | Client disconnect, timeout, oversized request |
| HTTP Parsing | HTTP Parser | Protocol parsing, structure population | Malformed request, unsupported method |
| Path Resolution | File Handler | Security validation, file system access | Directory traversal, permission denied |
| Response Generation | File Handler | File reading, MIME detection, header creation | File not found, I/O errors |
| Response Transmission | Connection Handler | `write()` with complete data delivery | Client disconnect, network congestion |
| Cleanup | Connection Handler | Resource deallocation, connection closure | File descriptor leaks |

The beauty of this architecture is its **modularity** - each component has a clear input and output interface, making it possible to test and debug each stage independently. When problems occur, the structured flow helps isolate issues to specific components and processing stages.

> **Error Propagation Design**: Errors detected in any component flow back through the system as HTTP error responses rather than crashing the server. This ensures that problems with individual requests don't affect other concurrent connections or server stability.

The request processing flow also demonstrates why our component separation is effective. The TCP Server handles networking complexity, the HTTP Parser manages protocol details, the File Handler deals with security and content delivery, and the Concurrency Manager coordinates multiple simultaneous flows of this same process. Each component can be implemented, tested, and optimized independently while contributing to the overall request-response capability.

This flow repeats continuously as the server processes client requests. The main server loop accepts new connections and dispatches them for processing, while existing connections move through their individual request-response cycles. The architecture supports both simple sequential processing (for learning) and sophisticated concurrent processing (for performance) using the same fundamental flow pattern.

### Implementation Guidance

The implementation guidance bridges the gap between architectural understanding and actual C code. The following recommendations and starter code help structure the development process across all four milestones.

#### Technology and Library Recommendations

| Component | Simple Approach | Production Approach | Recommended for Learning |
|-----------|----------------|-------------------|------------------------|
| TCP Networking | POSIX sockets (`socket.h`, `netinet/in.h`) | POSIX sockets with epoll/kqueue | POSIX sockets |
| HTTP Parsing | Manual string parsing with `strtok()` | Purpose-built state machine parser | State machine parser |
| File I/O | Standard C file operations (`fopen()`, `fread()`) | Memory-mapped files (`mmap()`) | Standard C file operations |
| Concurrency | `pthread` library | Event-driven with `epoll`/`select` | Start with `pthread` |
| Logging | `printf()` to stdout/stderr | Structured logging with levels | `printf()` with log levels |
| Configuration | Command-line arguments | Configuration file parsing | Command-line arguments |

#### Project Structure and Build System

The Makefile should support incremental development, allowing students to build and test individual components:

```makefile
# Makefile - Progressive build targets for milestone development
CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -pthread -g
SRCDIR = src
OBJDIR = obj

# Milestone 1: TCP Server basics
tcp_server: $(OBJDIR)/main.o $(OBJDIR)/tcp_server.o $(OBJDIR)/logger.o
	$(CC) $(CFLAGS) -o $@ $^

# Milestone 2: Add HTTP parsing
http_parser: $(OBJDIR)/main.o $(OBJDIR)/tcp_server.o $(OBJDIR)/http_parser.o $(OBJDIR)/logger.o
	$(CC) $(CFLAGS) -o $@ $^

# Milestone 3: Add file serving
file_server: $(OBJDIR)/main.o $(OBJDIR)/tcp_server.o $(OBJDIR)/http_parser.o $(OBJDIR)/file_handler.o $(OBJDIR)/mime_types.o $(OBJDIR)/logger.o
	$(CC) $(CFLAGS) -o $@ $^

# Milestone 4: Complete server with concurrency
http_server: $(OBJDIR)/main.o $(OBJDIR)/server.o $(OBJDIR)/tcp_server.o $(OBJDIR)/http_parser.o $(OBJDIR)/file_handler.o $(OBJDIR)/connection_manager.o $(OBJDIR)/logger.o
	$(CC) $(CFLAGS) -o $@ $^
```

#### Core Infrastructure Starter Code

Students should focus their learning effort on the four main components rather than boilerplate code. The following infrastructure provides a solid foundation:

**Logger Implementation** (Complete - Ready to Use):
```c
// src/util/logger.h - Logging infrastructure for debugging and monitoring
#ifndef LOGGER_H
#define LOGGER_H

#include <stdio.h>
#include <time.h>

typedef enum {
    LOG_DEBUG = 0,
    LOG_INFO = 1,
    LOG_WARN = 2,
    LOG_ERROR = 3
} log_level_t;

#define LOG_DEBUG_MSG(fmt, ...) log_message(LOG_DEBUG, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_INFO_MSG(fmt, ...) log_message(LOG_INFO, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_WARN_MSG(fmt, ...) log_message(LOG_WARN, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_ERROR_MSG(fmt, ...) log_message(LOG_ERROR, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

void log_message(log_level_t level, const char* file, int line, const char* fmt, ...);
void set_log_level(log_level_t level);

#endif

// src/util/logger.c
#include "logger.h"
#include <stdarg.h>
#include <string.h>

static log_level_t current_level = LOG_INFO;
static const char* level_names[] = {"DEBUG", "INFO", "WARN", "ERROR"};

void set_log_level(log_level_t level) {
    current_level = level;
}

void log_message(log_level_t level, const char* file, int line, const char* fmt, ...) {
    if (level < current_level) return;
    
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    char time_buffer[26];
    strftime(time_buffer, 26, "%Y-%m-%d %H:%M:%S", tm_info);
    
    const char* filename = strrchr(file, '/');
    filename = filename ? filename + 1 : file;
    
    fprintf(stderr, "[%s] %s %s:%d - ", time_buffer, level_names[level], filename, line);
    
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    
    fprintf(stderr, "\n");
    fflush(stderr);
}
```

**Configuration Management** (Complete - Ready to Use):
```c
// src/server.h - Server configuration and initialization
#ifndef SERVER_H
#define SERVER_H

#include <time.h>
#include <pthread.h>
#include <netinet/in.h>

#define MAX_PATH_LENGTH 1024
#define DEFAULT_PORT 8080
#define MAX_REQUEST_SIZE 8192
#define DEFAULT_MAX_THREADS 50
#define DEFAULT_MAX_CONNECTIONS 100
#define DEFAULT_TIMEOUT_SEC 30

typedef struct {
    int port;
    char document_root[MAX_PATH_LENGTH];
    int max_threads;
    int max_connections;
    int request_timeout_sec;
} ServerConfig;

typedef struct {
    int client_fd;
    struct sockaddr_in client_addr;
    time_t connect_time;
    pthread_t thread_id;
} ConnectionContext;

// Configuration initialization and validation
ServerConfig* create_default_config(void);
int parse_command_line_args(int argc, char* argv[], ServerConfig* config);
int validate_config(const ServerConfig* config);
void print_config(const ServerConfig* config);

#endif
```

#### Core Component Skeleton Code

For the main learning components, provide function signatures with detailed TODO comments that map to the architectural design:

**TCP Server Component Skeleton**:
```c
// src/tcp/tcp_server.c - TCP server implementation skeleton
#include "tcp_server.h"
#include "../util/logger.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <errno.h>

int create_server_socket(int port) {
    // TODO 1: Create TCP socket using socket(AF_INET, SOCK_STREAM, 0)
    // TODO 2: Set SO_REUSEADDR option to allow port reuse after restart
    // TODO 3: Configure sockaddr_in structure with AF_INET, port, INADDR_ANY
    // TODO 4: Bind socket to the configured address using bind()
    // TODO 5: Start listening with listen() and appropriate backlog
    // TODO 6: Return server socket file descriptor, or -1 on error
    // Hint: Use htons() to convert port to network byte order
    // Hint: Log each step for debugging - socket creation, binding, listening
    return -1; // Placeholder
}

void server_main_loop(int server_fd) {
    // TODO 1: Initialize infinite loop for connection acceptance
    // TODO 2: Call accept() to wait for incoming client connections
    // TODO 3: Log client connection details (IP address, port)
    // TODO 4: Create ConnectionContext structure for this client
    // TODO 5: Dispatch connection to concurrency manager for processing
    // TODO 6: Handle accept() errors (EINTR, EMFILE, ENFILE)
    // TODO 7: Continue loop after handling connection or error
    // Hint: accept() blocks until client connects - this is expected
    // Hint: Store client address info for logging and security
}
```

**HTTP Parser Component Skeleton**:
```c
// src/http/http_parser.c - HTTP parsing implementation skeleton
#include "http_parser.h"
#include "../util/logger.h"
#include <string.h>
#include <stdlib.h>

ssize_t read_complete_request(int sockfd, char* buffer, size_t max_size) {
    // TODO 1: Initialize buffer position and bytes remaining counter
    // TODO 2: Loop reading from socket until complete request received
    // TODO 3: Handle partial reads - recv() may return fewer bytes than requested
    // TODO 4: Look for "\r\n\r\n" sequence indicating end of headers
    // TODO 5: If Content-Length header present, continue reading body
    // TODO 6: Return total bytes read, or -1 on error
    // Hint: Use recv() with MSG_DONTWAIT for non-blocking reads
    // Hint: Grow buffer if request approaches max_size limit
    return -1; // Placeholder
}

int parse_http_request(const char* raw_request, HTTPRequest* request) {
    // TODO 1: Initialize HTTPRequest structure to zero/empty values
    // TODO 2: Extract method, path, version from first line using strtok()
    // TODO 3: Validate method is supported (GET initially)
    // TODO 4: Parse headers line by line, splitting on ": " delimiter
    // TODO 5: Store headers in request->headers array with count
    // TODO 6: Handle special headers like Content-Length, Host
    // TODO 7: Return 0 on success, HTTP error code on failure
    // Hint: Use strncpy() to prevent buffer overflows
    // Hint: Trim whitespace from header values
    return -1; // Placeholder
}
```

#### Development and Testing Checkpoints

After implementing each milestone, students should verify functionality with these specific tests:

**Milestone 1 Checkpoint**: TCP Server Basics
- Command: `./tcp_server 8080`
- Test: `telnet localhost 8080` should connect without error
- Expected: Server logs "Client connected from 127.0.0.1:XXXXX"
- Test: Send any text, should receive hardcoded HTTP response
- Signs of problems: "Address already in use" (port not released), connection refused (binding failed), server crashes (missing error handling)

**Milestone 2 Checkpoint**: HTTP Request Parsing
- Command: `./http_parser 8080`
- Test: `curl -v http://localhost:8080/test.html`
- Expected: Server logs parsed method=GET, path=/test.html, version=HTTP/1.1
- Test: `curl -H "Custom-Header: test-value" http://localhost:8080/`
- Expected: Server logs the custom header in parsed headers list
- Signs of problems: Parsing fails on normal requests (line ending issues), crashes on malformed input (missing bounds checking)

**Milestone 3 Checkpoint**: Static File Serving
- Setup: Create `www/index.html` and `www/test.txt` in document root
- Test: `curl http://localhost:8080/index.html` should return file contents
- Expected: Correct Content-Type header based on .html extension
- Test: `curl http://localhost:8080/../etc/passwd` should return 403 Forbidden
- Expected: Security validation prevents directory traversal
- Signs of problems: Binary files corrupted (text mode reading), 404 for existing files (path resolution issues)

**Milestone 4 Checkpoint**: Concurrent Connections
- Test: `for i in {1..10}; do curl http://localhost:8080/index.html & done; wait`
- Expected: All 10 requests complete successfully with same response
- Test: Use `ab -n 100 -c 10 http://localhost:8080/index.html` for load testing
- Expected: No connection refused errors, consistent response times
- Signs of problems: Requests hang (thread deadlock), server crashes (race conditions), increasing response times (resource leaks)

#### Common Implementation Pitfalls

⚠️ **Pitfall: Network Byte Order Confusion**
Students often forget to use `htons()` when setting port numbers in `sockaddr_in` structures. This causes the server to bind to unexpected ports (e.g., port 8080 becomes port 36895 in big-endian). Always use `htons()` for ports and `htonl()` for addresses, even though `INADDR_ANY` is already in network byte order.

⚠️ **Pitfall: Partial Read Handling**
Network reads can return fewer bytes than requested, but beginners often assume `recv()` always returns complete data. This causes parsing failures when HTTP requests are split across multiple TCP packets. Always loop on read operations until you have complete data, checking for both partial reads and error conditions.

⚠️ **Pitfall: File Descriptor Leaks**
Forgetting to close client sockets leads to file descriptor exhaustion, especially under concurrent load. The server will eventually fail to accept new connections when the per-process file descriptor limit is reached. Ensure every `accept()` has a corresponding `close()`, even in error paths.

⚠️ **Pitfall: Buffer Overflow in Parsing**
HTTP header values can be arbitrarily long, but fixed-size buffers in `HTTPRequest` structures can overflow. Always use `strncpy()` instead of `strcpy()` and validate input lengths before copying. Consider rejecting requests with headers exceeding reasonable limits.

⚠️ **Pitfall: Directory Traversal Security**
Students often implement simple string concatenation for path resolution (`document_root + request_path`), which allows attacks like `../../../etc/passwd`. Use `realpath()` or manual path validation to ensure resolved paths stay within the document root directory.


## Data Model

> **Milestone(s):** Foundation for Milestones 1-4 - defines the core data structures that represent HTTP messages, server configuration, and connection state throughout TCP server basics, HTTP parsing, file serving, and concurrency management

The data model forms the backbone of our HTTP server implementation, defining how we represent HTTP messages, server configuration, and connection state in memory. Think of these data structures as the **vocabulary** our system uses to communicate internally - just as a library has standardized forms for book requests, patron information, and checkout records, our HTTP server needs standardized structures to represent requests, responses, configuration, and active connections.

![HTTP Data Structures](./diagrams/data-model.svg)

The data model serves several critical purposes in our architecture. First, it provides a **common language** between components - when the HTTP parser extracts a request, it produces an `HTTPRequest` structure that the file handler can immediately understand. Second, it establishes **memory ownership** boundaries, making it clear which component is responsible for allocating and freeing different data. Third, it defines the **security boundaries** where validation must occur, particularly around path handling and request size limits.

### HTTP Message Structures

HTTP messages are the fundamental units of communication in our server. Just as a postal system has standardized envelope formats with sender addresses, recipient addresses, and contents, HTTP has standardized request and response formats that our server must parse and generate correctly.

The `HTTPRequest` structure represents an incoming client request after parsing. This structure captures all the essential information from the HTTP request line and headers, providing a clean interface for other components to access request data without needing to understand HTTP parsing details.

| Field Name | Type | Description |
|------------|------|-------------|
| `method` | `char[16]` | HTTP method (GET, POST, etc.) - fixed size prevents overflow attacks |
| `path` | `char[MAX_PATH_LENGTH]` | Requested URL path - bounds checked during parsing |
| `version` | `char[16]` | HTTP version string (HTTP/1.1) - used for protocol compliance |
| `headers` | `char[MAX_HEADERS][2][256]` | Array of header key-value pairs - [index][0] = key, [index][1] = value |
| `header_count` | `int` | Number of headers parsed - prevents access beyond valid headers |
| `body` | `char*` | Pointer to request body data - dynamically allocated based on Content-Length |
| `body_length` | `size_t` | Size of body in bytes - critical for binary data and memory management |

The fixed-size fields for method, path, and version provide **buffer overflow protection** by enforcing maximum lengths during parsing. The headers array uses a two-dimensional structure where each header occupies one slot with separate key and value strings. This design trades some memory efficiency for simplicity and safety - we can iterate through headers without complex pointer arithmetic.

> **Critical Design Insight**: The `HTTPRequest` structure owns all its string data except the body, which is dynamically allocated. This ownership model ensures that once parsing completes, the request structure is self-contained and can be passed between threads safely.

The `HTTPResponse` structure represents an outgoing server response. This structure mirrors the request format but includes fields specific to response generation, particularly status codes and response bodies that may contain file data.

| Field Name | Type | Description |
|------------|------|-------------|
| `status_code` | `int` | HTTP status code (200, 404, 500, etc.) - used for response line generation |
| `status_text` | `char[64]` | Status reason phrase (OK, Not Found, etc.) - human readable status |
| `headers` | `char[MAX_HEADERS][2][256]` | Response headers - same format as request for consistency |
| `header_count` | `int` | Number of response headers - prevents iteration beyond valid headers |
| `body` | `char*` | Response body content - often points to file data or error page HTML |
| `body_length` | `size_t` | Size of response body - essential for Content-Length header |

The response structure separates `status_code` as an integer from `status_text` as a string. This design allows components to set status codes using constants (200, 404) while the HTTP formatting code handles the textual representation. The body pointer typically points to either file content loaded from disk or statically allocated error page HTML.

> **Decision: Fixed-Size vs Dynamic Headers**
> - **Context**: HTTP headers can vary widely in number and size, and we need to balance memory efficiency with implementation simplicity
> - **Options Considered**:
>   1. Fixed-size array of header structs (chosen)
>   2. Linked list of dynamically allocated headers
>   3. Hash table for O(1) header lookup
> - **Decision**: Fixed-size array with maximum header count
> - **Rationale**: Educational simplicity outweighs memory efficiency. Fixed arrays eliminate complex memory management and pointer chasing. Most HTTP requests have fewer than 20 headers, making the memory overhead acceptable.
> - **Consequences**: Simple iteration and memory management, but wastes memory for requests with few headers. Header count limits prevent resource exhaustion attacks.

### Server Configuration

Server configuration defines the runtime behavior and operational parameters of our HTTP server. Think of configuration as the **operating procedures** for our library - it defines which port to listen on (like which desk to staff), where files are stored (like which building houses the collection), and how many concurrent patrons we can serve (like staffing levels).

The `ServerConfig` structure centralizes all configurable server parameters, making it easy to adjust behavior without recompiling code. This structure is typically populated from command-line arguments, configuration files, or environment variables during server startup.

| Field Name | Type | Description |
|------------|------|-------------|
| `port` | `int` | TCP port number for incoming connections - typically 8080 for development |
| `document_root` | `char[MAX_PATH_LENGTH]` | Filesystem path to serve files from - security boundary for file access |
| `max_threads` | `int` | Maximum number of worker threads in thread pool - prevents resource exhaustion |
| `max_connections` | `int` | Maximum concurrent client connections - limits memory and file descriptor usage |
| `request_timeout_sec` | `int` | Seconds to wait for complete request - prevents slowloris attacks |

The `document_root` field establishes the **security boundary** for file serving. All file access must be validated against this root directory to prevent directory traversal attacks. The path validation logic uses this field as the trusted base path for resolving all requested file paths.

The concurrency control fields (`max_threads`, `max_connections`) provide **resource protection** mechanisms. Without these limits, a malicious client could exhaust server resources by opening thousands of connections or triggering creation of unlimited threads. These limits enable **graceful degradation** under load rather than complete system failure.

> **Decision: Single Configuration Structure vs Component-Specific Config**
> - **Context**: Different server components need different configuration parameters, and we must decide how to organize these settings
> - **Options Considered**:
>   1. Single global configuration structure (chosen)
>   2. Separate configuration structures per component
>   3. Key-value configuration map with string keys
> - **Decision**: Single `ServerConfig` structure with all parameters
> - **Rationale**: Simplifies configuration management and ensures all components see consistent values. Eliminates coordination issues between component-specific configs. Educational code benefits from centralized configuration.
> - **Consequences**: All components depend on the same config structure, which creates coupling but eliminates configuration synchronization bugs. Adding new parameters requires updating the central structure.

The default configuration values provide sensible starting points for development and testing:

| Parameter | Default Value | Rationale |
|-----------|---------------|-----------|
| `port` | `DEFAULT_PORT` (8080) | Non-privileged port that doesn't require root access |
| `document_root` | `./public` | Relative path allows easy setup in any directory |
| `max_threads` | 10 | Sufficient for development testing without resource exhaustion |
| `max_connections` | 100 | Reasonable limit for educational server implementation |
| `request_timeout_sec` | 30 | Long enough for normal requests, short enough to prevent attacks |

### Connection State Management

Connection state tracking manages the lifecycle and context of individual client connections. Think of connection state as the **patron record** system in our library analogy - for each visitor, we track when they arrived, which desk they're assigned to, and what resources they're currently using.

The `ConnectionContext` structure maintains per-connection information throughout the request processing lifecycle. This context gets created when a client connects and destroyed when the connection closes, providing a clean abstraction for connection-specific data.

| Field Name | Type | Description |
|------------|------|-------------|
| `client_fd` | `int` | File descriptor for client socket - used for all I/O operations with this client |
| `client_addr` | `struct sockaddr_in` | Client's IP address and port - useful for logging and access control |
| `connect_time` | `time_t` | Timestamp when connection was established - enables timeout detection |
| `thread_id` | `pthread_t` | Thread handling this connection - used for thread pool management |

The `client_fd` field is the **primary key** for this connection - all socket operations use this file descriptor to identify which client to communicate with. This field must be carefully managed to prevent **file descriptor leaks**, where unclosed descriptors eventually exhaust system resources.

The `client_addr` field captures the network address of the connecting client. This information serves multiple purposes: security logging (tracking which IPs generate errors), access control (blocking malicious IPs), and debugging (identifying problematic clients). The `sockaddr_in` structure contains the client's IP address and port number in network byte order.

> **Decision: Per-Connection Context vs Global Connection Tracking**
> - **Context**: We need to track active connections for resource management and graceful shutdown, and must decide how to organize this tracking
> - **Options Considered**:
>   1. Individual context structures passed to handlers (chosen)
>   2. Global connection registry with lookup by file descriptor
>   3. Thread-local storage for connection data
> - **Decision**: Individual `ConnectionContext` structures passed to request handlers
> - **Rationale**: Provides clear ownership model where each handler owns its connection context. Eliminates global state synchronization issues. Makes testing easier since handlers receive all needed context as parameters.
> - **Consequences**: Connection context must be explicitly passed between functions, but this makes data flow visible and eliminates hidden dependencies on global state.

The connection lifecycle follows a predictable pattern that our data structures support:

1. **Connection Establishment**: TCP server creates `ConnectionContext` with client socket information
2. **Request Processing**: Context gets passed to HTTP parser and file handler for request processing
3. **Response Generation**: Context used for writing response back to client socket
4. **Connection Cleanup**: Context destruction triggers socket closure and resource cleanup

Connection timeout detection uses the `connect_time` field to identify connections that have been idle too long. The timeout logic compares current time against `connect_time + request_timeout_sec` to determine if a connection should be forcibly closed:

```
current_time = time(NULL);
connection_age = current_time - context->connect_time;
if (connection_age > config->request_timeout_sec) {
    // Force connection closure to prevent resource exhaustion
}
```

Thread tracking via `thread_id` enables proper cleanup during server shutdown. The main server can iterate through active connections and signal worker threads to complete current requests before terminating. This field also helps with debugging by associating log messages with specific threads.

> ⚠️ **Pitfall: Connection Context Lifecycle Management**
> 
> The most common mistake is failing to properly clean up connection contexts, leading to memory leaks and file descriptor exhaustion. Every `ConnectionContext` creation must have a corresponding cleanup that closes the client socket and frees any dynamically allocated resources. The cleanup must occur even when errors happen during request processing - use proper error handling patterns to ensure cleanup code always runs.

### Common Data Model Pitfalls

Understanding the common mistakes made when designing and implementing these data structures helps avoid subtle bugs that can be difficult to debug in a networked system.

⚠️ **Pitfall: Buffer Overflow in Fixed-Size Fields**

The fixed-size character arrays in `HTTPRequest` and `HTTPResponse` provide safety but require careful bounds checking during parsing. Failing to validate input lengths before copying into these fields creates classic buffer overflow vulnerabilities. Always use `strncpy` instead of `strcpy` and ensure null termination:

```c
// Incorrect - potential overflow
strcpy(request->method, parsed_method);

// Correct - bounds checking with null termination
strncpy(request->method, parsed_method, sizeof(request->method) - 1);
request->method[sizeof(request->method) - 1] = '\0';
```

⚠️ **Pitfall: Memory Ownership Confusion**

The HTTP structures mix stack-allocated fixed arrays with heap-allocated dynamic content (body fields). This creates confusion about who owns what memory and when to free it. Establish clear ownership rules: the parsing component allocates body memory, the request/response structure owns it during processing, and the cleanup component frees it. Never free body memory from multiple locations.

⚠️ **Pitfall: Network Byte Order in Address Structures**

The `sockaddr_in` structure in `ConnectionContext` stores addresses in network byte order, while application code typically works with host byte order. Forgetting to convert between these formats causes connection failures and incorrect logging. Always use `htons`/`ntohs` for ports and `htonl`/`ntohl` for addresses when converting between network and host byte order.

⚠️ **Pitfall: Thread Safety in Shared Structures**

While individual `ConnectionContext` structures are thread-safe because each connection runs in its own thread, the shared `ServerConfig` structure requires careful handling. Multiple threads may read configuration values concurrently, which is safe, but dynamic configuration updates would require synchronization. For this educational implementation, treat `ServerConfig` as read-only after initialization to avoid concurrency issues.

⚠️ **Pitfall: Resource Cleanup on Early Exit**

Error conditions during request processing can cause early returns from functions, potentially skipping cleanup code. This leads to file descriptor leaks and memory leaks that accumulate over time. Structure your code to use cleanup patterns like `goto error_cleanup` in C or defer statements in other languages to ensure resources are always freed.

The data model provides the foundation for all subsequent components in our HTTP server. These structures define the contracts between components and establish the memory management patterns that prevent common security vulnerabilities and resource leaks. Understanding these structures deeply enables confident implementation of the parsing, file serving, and concurrency components that build upon this foundation.

### Implementation Guidance

The data model implementation focuses on creating robust, secure data structures that prevent common web server vulnerabilities while maintaining simplicity for educational purposes. The following guidance provides complete working code for structure definitions and helper functions.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Memory Management | Manual malloc/free with cleanup functions | Memory pools with automatic cleanup |
| String Handling | Fixed-size buffers with strncpy | Dynamic strings with automatic resizing |
| Configuration | Compile-time constants | Runtime config file parsing |
| Address Handling | Manual network/host byte order conversion | Address utility library wrapper |

#### Recommended File Structure

```
http-server/
  src/
    data_model.h              ← structure definitions (this section)
    data_model.c              ← helper functions and cleanup
    http_parser.h             ← parsing component
    http_parser.c
    file_handler.h            ← file serving component  
    file_handler.c
    tcp_server.h              ← network component
    tcp_server.c
    main.c                    ← entry point
  tests/
    test_data_model.c         ← unit tests for structures
  public/                     ← document root for serving files
    index.html
    style.css
```

#### Complete Data Structure Definitions

**File: `src/data_model.h`**

```c
#ifndef DATA_MODEL_H
#define DATA_MODEL_H

#include <sys/socket.h>
#include <netinet/in.h>
#include <pthread.h>
#include <time.h>
#include <stddef.h>

// Protocol and size constants
#define MAX_REQUEST_SIZE 8192
#define MAX_PATH_LENGTH 1024
#define MAX_HEADERS 32
#define MAX_HEADER_LENGTH 256
#define DEFAULT_PORT 8080

// HTTP request structure - represents parsed client request
typedef struct {
    char method[16];                              // HTTP method (GET, POST, etc.)
    char path[MAX_PATH_LENGTH];                   // URL path component
    char version[16];                             // HTTP version (HTTP/1.1)
    char headers[MAX_HEADERS][2][MAX_HEADER_LENGTH]; // [index][0]=key, [index][1]=value
    int header_count;                             // Number of valid headers
    char* body;                                   // Dynamically allocated body content
    size_t body_length;                          // Size of body in bytes
} HTTPRequest;

// HTTP response structure - represents server response to send
typedef struct {
    int status_code;                              // HTTP status code (200, 404, etc.)
    char status_text[64];                         // Status reason phrase
    char headers[MAX_HEADERS][2][MAX_HEADER_LENGTH]; // Response headers
    int header_count;                             // Number of response headers
    char* body;                                   // Response body content
    size_t body_length;                          // Size of response body
} HTTPResponse;

// Server configuration - runtime parameters
typedef struct {
    int port;                                     // TCP port to listen on
    char document_root[MAX_PATH_LENGTH];          // Root directory for file serving
    int max_threads;                              // Maximum worker threads
    int max_connections;                          // Maximum concurrent connections
    int request_timeout_sec;                      // Request timeout in seconds
} ServerConfig;

// Connection context - per-connection state tracking
typedef struct {
    int client_fd;                                // Client socket file descriptor
    struct sockaddr_in client_addr;               // Client network address
    time_t connect_time;                          // When connection was established
    pthread_t thread_id;                          // Thread handling this connection
} ConnectionContext;

// Helper function declarations
void init_http_request(HTTPRequest* request);
void init_http_response(HTTPResponse* response);
void init_server_config(ServerConfig* config);
void init_connection_context(ConnectionContext* context, int client_fd, struct sockaddr_in* addr);

void cleanup_http_request(HTTPRequest* request);
void cleanup_http_response(HTTPResponse* response);

int add_response_header(HTTPResponse* response, const char* key, const char* value);
const char* get_request_header(const HTTPRequest* request, const char* key);

#endif // DATA_MODEL_H
```

#### Core Helper Functions (Complete Implementation)

**File: `src/data_model.c`**

```c
#include "data_model.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Initialize HTTPRequest structure with safe defaults
void init_http_request(HTTPRequest* request) {
    if (!request) return;
    
    memset(request->method, 0, sizeof(request->method));
    memset(request->path, 0, sizeof(request->path));
    memset(request->version, 0, sizeof(request->version));
    memset(request->headers, 0, sizeof(request->headers));
    request->header_count = 0;
    request->body = NULL;
    request->body_length = 0;
}

// Initialize HTTPResponse structure with safe defaults
void init_http_response(HTTPResponse* response) {
    if (!response) return;
    
    response->status_code = 200;
    strncpy(response->status_text, "OK", sizeof(response->status_text) - 1);
    response->status_text[sizeof(response->status_text) - 1] = '\0';
    memset(response->headers, 0, sizeof(response->headers));
    response->header_count = 0;
    response->body = NULL;
    response->body_length = 0;
}

// Initialize ServerConfig with sensible defaults
void init_server_config(ServerConfig* config) {
    if (!config) return;
    
    config->port = DEFAULT_PORT;
    strncpy(config->document_root, "./public", sizeof(config->document_root) - 1);
    config->document_root[sizeof(config->document_root) - 1] = '\0';
    config->max_threads = 10;
    config->max_connections = 100;
    config->request_timeout_sec = 30;
}

// Initialize connection context with client information
void init_connection_context(ConnectionContext* context, int client_fd, struct sockaddr_in* addr) {
    if (!context) return;
    
    context->client_fd = client_fd;
    if (addr) {
        context->client_addr = *addr;
    } else {
        memset(&context->client_addr, 0, sizeof(context->client_addr));
    }
    context->connect_time = time(NULL);
    context->thread_id = pthread_self();
}

// Clean up dynamically allocated request memory
void cleanup_http_request(HTTPRequest* request) {
    if (!request) return;
    
    if (request->body) {
        free(request->body);
        request->body = NULL;
        request->body_length = 0;
    }
}

// Clean up dynamically allocated response memory
void cleanup_http_response(HTTPResponse* response) {
    if (!response) return;
    
    // Note: Only free body if it was dynamically allocated
    // File content bodies may point to mapped memory
    if (response->body) {
        free(response->body);
        response->body = NULL;
        response->body_length = 0;
    }
}
```

#### Core Logic Skeleton (for learners to complete)

```c
// Add a header to HTTP response - learners implement this
int add_response_header(HTTPResponse* response, const char* key, const char* value) {
    // TODO 1: Check if response and key/value pointers are valid
    // TODO 2: Check if we have space for another header (header_count < MAX_HEADERS)
    // TODO 3: Copy key into headers[header_count][0] using strncpy with bounds checking
    // TODO 4: Copy value into headers[header_count][1] using strncpy with bounds checking
    // TODO 5: Ensure both strings are null-terminated
    // TODO 6: Increment header_count
    // TODO 7: Return 0 on success, -1 on error
    // Hint: Always leave one byte for null terminator when using strncpy
}

// Find header value by key in HTTP request - learners implement this
const char* get_request_header(const HTTPRequest* request, const char* key) {
    // TODO 1: Check if request and key pointers are valid - return NULL if not
    // TODO 2: Loop through request->header_count headers
    // TODO 3: For each header, compare headers[i][0] with key using strcasecmp (case-insensitive)
    // TODO 4: If match found, return pointer to headers[i][1] (the value)
    // TODO 5: If no match found after loop, return NULL
    // Hint: Use strcasecmp for case-insensitive header name comparison
}
```

#### Milestone Checkpoints

**After implementing data structures:**

1. **Compilation Test**: Run `gcc -c src/data_model.c -o data_model.o` - should compile without errors
2. **Initialization Test**: Create simple test program that initializes each structure and prints field values
3. **Memory Test**: Use `valgrind` to check for memory leaks in cleanup functions

**Expected behavior:**
- All structures initialize with safe default values
- Header functions handle bounds checking correctly
- Cleanup functions prevent memory leaks
- No segmentation faults or buffer overflows

**Signs of problems:**
- Compiler warnings about buffer overflows → Check strncpy usage and null termination
- Segfaults during initialization → Check pointer validation in functions
- Memory leaks in valgrind → Ensure cleanup functions free all allocated memory

#### Language-Specific Implementation Hints

**C-Specific Guidelines:**
- Always use `strncpy` instead of `strcpy` for fixed-size buffers
- Set explicit null terminators: `buffer[size-1] = '\0'` after strncpy
- Check all pointer parameters for NULL before dereferencing
- Use `memset` to zero-initialize structures containing arrays
- Free dynamically allocated memory in reverse order of allocation

**Memory Management Patterns:**
- Initialize all structure fields to safe defaults (NULL pointers, zero values)
- Use consistent naming: `init_*` for initialization, `cleanup_*` for deallocation
- Never free the same pointer twice - set to NULL after freeing
- Check malloc return values - handle allocation failures gracefully

**Network Address Handling:**
- Use `htons()` when setting port in sockaddr_in: `addr.sin_port = htons(port)`
- Use `ntohs()` when reading port from sockaddr_in: `port = ntohs(addr.sin_port)`
- Set address family explicitly: `addr.sin_family = AF_INET`
- Use `INADDR_ANY` for server binding: `addr.sin_addr.s_addr = INADDR_ANY`

This implementation foundation provides type-safe, bounds-checked data structures that prevent common web server vulnerabilities while maintaining educational clarity. The helper functions establish consistent patterns for initialization and cleanup that carry forward into more complex components.


## TCP Server Component

> **Milestone(s):** Milestone 1 (TCP Server Basics) - creates the foundational networking layer that accepts and manages TCP connections

### TCP Server Mental Model: Restaurant Host Station Analogy

Think of the TCP server component as the host station at a busy restaurant. Just as a restaurant host manages the front door, greets guests, and assigns them to available tables, our TCP server manages the network "front door," accepts incoming client connections, and prepares them for request processing.

The **server socket** is like the restaurant's front door itself - it's a fixed location where customers know they can arrive. The host doesn't serve food directly; instead, they perform the crucial role of managing arrivals and ensuring each guest gets proper attention. Similarly, our server socket doesn't process HTTP requests directly - it focuses solely on the TCP connection lifecycle.

When a customer approaches the restaurant, the host performs a standard greeting ritual: welcoming them, checking if they have a reservation, and determining where to seat them. In our TCP server, this greeting ritual is the **connection acceptance process**: we accept the incoming connection, validate it's properly formed, and prepare a dedicated communication channel for that specific client.

The host station maintains a guest registry - tracking who's currently dining, when they arrived, and which table they're using. Our TCP server maintains **connection context** - tracking which clients are connected, their network addresses, and the file descriptors representing their communication channels.

Just as a restaurant can only seat a limited number of guests based on table availability and staff capacity, our TCP server must manage **resource limits**. When the restaurant is full, the host might ask new arrivals to wait or politely suggest they try again later. Similarly, when our server reaches its connection limits, it must gracefully handle new connection attempts without crashing or leaving clients hanging indefinitely.

The host station operates independently of the kitchen - guests can arrive and be seated even when the kitchen is temporarily busy. This separation of concerns is crucial in our TCP server design: connection acceptance operates independently of HTTP request processing, allowing us to maintain responsive networking even when file serving or parsing operations take time.

![TCP Connection Lifecycle](./diagrams/connection-lifecycle.svg)

### TCP Server Interface

The TCP server component exposes a clean interface that abstracts the underlying socket programming complexity while providing the necessary control points for configuration and lifecycle management. This interface serves as the contract between the networking layer and the rest of our HTTP server components.

The interface design follows the principle of **progressive disclosure** - simple cases require minimal configuration, while advanced scenarios can access detailed control options. A junior developer can get a basic server running with just a port number, while an experienced developer can fine-tune socket options, connection limits, and error handling behavior.

| Method Name | Parameters | Returns | Description |
|-------------|------------|---------|-------------|
| `create_server_socket` | `port int` | `int` | Creates, configures, and binds server socket to specified port. Returns server file descriptor or -1 on failure. |
| `server_main_loop` | `server_fd int` | `void` | Main connection acceptance loop. Blocks accepting connections until server shutdown signal received. |
| `init_connection_context` | `context ConnectionContext*, client_fd int, addr sockaddr_in` | `void` | Initializes connection context structure with client information and timestamps. |
| `accept_client_connection` | `server_fd int, context ConnectionContext*` | `int` | Accepts single client connection and populates context. Returns client file descriptor or -1. |
| `close_client_connection` | `context ConnectionContext*` | `void` | Properly closes client connection and cleans up associated resources. |
| `set_socket_options` | `sockfd int, options int` | `int` | Configures socket options like SO_REUSEADDR, timeouts. Returns 0 on success. |
| `validate_client_connection` | `context ConnectionContext*` | `int` | Validates incoming connection against rate limits and security policies. Returns 1 if valid. |

The interface separates **socket lifecycle management** from **connection processing**. The `create_server_socket` function handles all the complex socket setup that must happen once at server startup: creating the socket, setting appropriate options, binding to the network interface, and beginning to listen for connections. This separation means the main server loop can focus purely on the accept-process-cleanup cycle without worrying about socket configuration details.

Connection context initialization through `init_connection_context` ensures that every client connection gets proper bookkeeping from the moment it's established. This context tracking becomes crucial later when we implement concurrent connection handling - each thread or event handler needs access to connection-specific state without interfering with other active connections.

The `validate_client_connection` method provides an extension point for implementing security policies, rate limiting, and resource management without cluttering the core acceptance logic. In a production server, this method might check the client's IP address against blacklists, enforce per-IP connection limits, or apply geographic restrictions.

> **Key Design Principle**: The TCP server interface maintains **single responsibility** - it knows how to manage TCP connections but remains agnostic about what data flows over those connections. This separation allows the same TCP foundation to support not just HTTP, but potentially other protocols in future extensions.

### Connection Acceptance Algorithm

The connection acceptance algorithm represents the core networking logic that transforms an abstract "listening socket" into concrete client connections ready for HTTP processing. This algorithm runs continuously while the server is active, making it one of the most critical paths in our entire system.

Understanding this algorithm requires recognizing that **TCP connection establishment** is actually a multi-step handshake between client and server, coordinated by the operating system's networking stack. Our server code participates in this handshake through specific socket system calls, but much of the complexity is handled transparently by the kernel.

Here's the detailed connection acceptance process:

1. **Initialize Server Socket**: Create a socket file descriptor using `socket(AF_INET, SOCK_STREAM, 0)` to establish a TCP communication endpoint. This creates a socket in the **closed state** that exists only in our process memory.

2. **Configure Socket Options**: Apply socket options before binding to prevent common deployment issues. Set `SO_REUSEADDR` to allow rapid server restarts without "Address already in use" errors. Configure timeout options to prevent indefinite blocking on problematic connections.

3. **Bind to Network Address**: Associate the socket with a specific network interface and port using `bind()`. This reserves the port number system-wide and transitions the socket to the **bound state**. The `sockaddr_in` structure specifies IPv4 protocol family, target port in network byte order, and `INADDR_ANY` to accept connections on all available network interfaces.

4. **Begin Listening**: Transition the socket to **listening state** using `listen()` with a backlog parameter that determines how many pending connections the kernel will queue while our server processes current requests. A backlog of 128 provides good balance between memory usage and connection responsiveness.

5. **Enter Acceptance Loop**: Begin the infinite loop that defines our server's operational lifetime. Each iteration handles one complete connection acceptance cycle, from detecting an incoming connection to preparing it for request processing.

6. **Block on Accept Call**: Execute `accept()` system call which blocks the current thread until a client initiates a TCP connection. The kernel handles the three-way TCP handshake automatically - by the time `accept()` returns, the connection is fully established and ready for data exchange.

7. **Extract Client Information**: When `accept()` returns successfully, it provides two critical pieces of information: a new file descriptor representing the established connection, and a `sockaddr_in` structure containing the client's network address and port. This client information enables logging, security validation, and debugging.

8. **Initialize Connection Context**: Populate a `ConnectionContext` structure with the client file descriptor, address information, connection timestamp, and any other per-connection state our server needs to track. This context follows the connection through its entire processing lifecycle.

9. **Validate Connection**: Apply security policies and resource limits to determine if this connection should be processed or rejected. Check the client's IP address against any configured restrictions, verify we haven't exceeded our maximum concurrent connection limit, and apply any rate limiting rules.

10. **Hand Off to Request Processor**: If the connection passes validation, transfer responsibility to the HTTP processing components. In a single-threaded server, this means immediately beginning HTTP parsing. In a multi-threaded server, this means dispatching the connection to a worker thread or adding it to a work queue.

11. **Handle Connection Errors**: If any step in the acceptance process fails, log the error with sufficient detail for debugging, ensure any partially allocated resources are cleaned up, and continue the acceptance loop. Server errors should not prevent processing other valid connections.

12. **Prepare for Next Connection**: Reset any per-connection state, check for shutdown signals, and return to the blocking accept call to handle the next incoming connection.

The algorithm's **error handling strategy** deserves special attention because network programming involves many potential failure modes. Connection acceptance can fail due to temporary resource exhaustion (too many open file descriptors), network interface issues, or malicious connection attempts. The algorithm must distinguish between **recoverable errors** (temporary resource shortage) and **fatal errors** (socket corruption) to maintain server stability.

**Resource management** throughout this algorithm is critical for long-running server stability. Every successful `accept()` call creates a new file descriptor that must eventually be closed. Every `ConnectionContext` structure that gets allocated must eventually be cleaned up. The algorithm includes explicit checkpoints where resource cleanup occurs, preventing the gradual resource leaks that would eventually crash a long-running server.

The algorithm's **blocking behavior** on the accept call has important implications for server architecture. In a single-threaded server, the entire server blocks waiting for connections, which means request processing must complete before new connections can be accepted. This creates a natural backpressure mechanism but limits concurrency. Understanding this blocking behavior is essential for making informed decisions about concurrency models in later milestones.

![Request Processing Sequence](./diagrams/request-processing-flow.svg)

### Architecture Decision Records

The TCP server component involves several fundamental architectural decisions that significantly impact the server's behavior, performance characteristics, and operational requirements. Each decision represents a trade-off between different operational priorities and implementation complexity levels.

> **Decision: Socket Address Configuration**
> - **Context**: Server must bind to a network address that clients can reach, but different deployment scenarios require different binding strategies. Local development needs different configuration than production deployment.
> - **Options Considered**: 
>   1. Bind to localhost (127.0.0.1) only for development safety
>   2. Bind to specific IP address provided via configuration
>   3. Bind to INADDR_ANY (0.0.0.0) accepting connections on all interfaces
> - **Decision**: Bind to `INADDR_ANY` with configurable port
> - **Rationale**: Maximum deployment flexibility allows same code to work in development (localhost access), containerized environments (bridge networks), and production (public interfaces). Security restrictions should be handled by firewalls and network policies rather than application binding logic.
> - **Consequences**: Enables flexible deployment but requires careful firewall configuration in production. Simplifies development and testing workflows.

| Option | Pros | Cons |
|--------|------|------|
| Localhost only | Development safety, prevents accidental exposure | Requires separate production configuration, complicates container networking |
| Specific IP configuration | Precise control, explicit security | Complex configuration management, brittle to infrastructure changes |
| INADDR_ANY | Maximum flexibility, simple configuration | Requires external security controls, potential for misconfiguration |

> **Decision: Socket Option Configuration**
> - **Context**: Default socket behavior often conflicts with server restart requirements and connection handling preferences. Raw sockets have conservative defaults that prioritize data integrity over operational convenience.
> - **Options Considered**:
>   1. Use default socket options for maximum compatibility
>   2. Set SO_REUSEADDR only for restart convenience
>   3. Configure comprehensive socket options including timeouts and buffer sizes
> - **Decision**: Set `SO_REUSEADDR` and basic timeout options, avoid advanced optimizations
> - **Rationale**: `SO_REUSEADDR` solves a critical development pain point (rapid restart failures) with minimal complexity. Advanced socket tuning requires deep networking knowledge and varies significantly across operating systems. Focus on educational value rather than production optimization.
> - **Consequences**: Enables smooth development workflow and basic timeout protection without introducing configuration complexity that obscures learning objectives.

| Option | Pros | Cons |
|--------|------|------|
| Default options | Maximum compatibility, simple code | Restart failures, no timeout protection |
| SO_REUSEADDR only | Solves restart problem, minimal complexity | No protection against hung connections |
| Comprehensive tuning | Production-quality performance | Complex configuration, platform-specific code |

> **Decision: Connection Backlog Size**
> - **Context**: The `listen()` system call requires a backlog parameter that determines how many pending connections the kernel queues while the server processes current requests. This value significantly impacts behavior under load.
> - **Options Considered**:
>   1. Small backlog (5-10) for simple educational scenarios
>   2. Medium backlog (128) balancing memory and responsiveness
>   3. Large backlog (1024+) for maximum connection handling
> - **Decision**: Medium backlog of 128 connections
> - **Rationale**: Small backlogs cause connection rejections during normal load testing with simple tools like curl loops. Large backlogs consume significant kernel memory and can mask concurrency problems by hiding the need for efficient request processing. 128 connections provides sufficient buffering for realistic testing while revealing performance bottlenecks.
> - **Consequences**: Allows meaningful load testing without overwhelming resource usage. Students can observe queue behavior under load without requiring sophisticated testing tools.

| Option | Pros | Cons |
|--------|------|------|
| Small backlog (5-10) | Low memory usage, simple | Connection rejections during normal testing |
| Medium backlog (128) | Good testing flexibility, reasonable memory | Moderate kernel memory usage |
| Large backlog (1024+) | Handles high load | High memory usage, masks performance problems |

> **Decision: Error Handling Strategy**
> - **Context**: Network operations can fail in numerous ways - temporary resource exhaustion, client disconnections, malformed connection attempts, and system-level networking issues. The server must handle these failures without crashing while providing useful debugging information.
> - **Options Considered**:
>   1. Aggressive error checking with server shutdown on any failure
>   2. Graceful degradation continuing operation despite individual connection failures
>   3. Minimal error handling focusing on core functionality
> - **Decision**: Graceful degradation with comprehensive logging
> - **Rationale**: Real servers must remain operational despite individual connection failures - network programming inherently involves partial failures that don't indicate server problems. Aggressive shutdown makes the server fragile during development and testing. Comprehensive logging provides debugging information without operational brittleness.
> - **Consequences**: Server remains stable during testing with imperfect clients (browsers, curl with various options) but requires systematic logging implementation and error classification.

| Option | Pros | Cons |
|--------|------|------|
| Aggressive shutdown | Simple code, fail-fast behavior | Brittle operation, difficult testing |
| Graceful degradation | Stable operation, realistic behavior | Complex error classification |
| Minimal handling | Simple implementation | Difficult debugging, potential resource leaks |

> **Decision: Resource Cleanup Responsibility**
> - **Context**: TCP connections create file descriptors that must be properly closed to prevent resource leaks. Connection contexts may allocate memory that requires cleanup. Clear ownership of cleanup responsibility prevents resource leaks and double-free errors.
> - **Options Considered**:
>   1. TCP server handles all cleanup automatically
>   2. Caller responsible for cleanup using provided cleanup functions
>   3. Hybrid approach with automatic cleanup for errors, manual for success cases
> - **Decision**: Caller responsible for cleanup using provided cleanup functions
> - **Rationale**: Clear ownership semantics prevent resource management confusion. The connection processing components need control over connection lifetime - automatic cleanup could interfere with Keep-Alive implementations or connection reuse. Explicit cleanup functions provide safety without removing control.
> - **Consequences**: Requires discipline in calling cleanup functions but provides clear resource ownership. Enables advanced connection management patterns in future extensions.

| Option | Pros | Cons |
|--------|------|------|
| Automatic cleanup | Simple usage, prevents leaks | Limited control, complicates advanced features |
| Caller responsible | Clear ownership, maximum control | Requires discipline, potential for mistakes |
| Hybrid approach | Balances safety and control | Complex ownership rules, confusing semantics |

### TCP Server Interface Implementation Details

The TCP server interface requires careful attention to **state management** and **error propagation** to ensure reliable operation across diverse deployment scenarios. Each interface method has specific preconditions and postconditions that must be clearly documented and consistently enforced.

The `create_server_socket` function encapsulates the complex socket initialization sequence that must occur exactly once per server lifetime. This function performs operations that cannot be safely retried or undone, making its error handling particularly important. Socket creation can fail due to permission issues (binding to privileged ports), resource exhaustion (too many open file descriptors), or network configuration problems (invalid addresses).

**Socket option configuration** within `create_server_socket` requires platform-specific knowledge but follows a standard pattern. The `SO_REUSEADDR` option allows immediate reuse of the server's port after shutdown, preventing the "Address already in use" errors that commonly frustrate developers during testing. Without this option, the operating system enforces a waiting period (typically 2-4 minutes) before allowing port reuse, which severely impacts development iteration speed.

The `server_main_loop` function implements the server's primary operational mode - an infinite loop accepting connections until explicitly shut down. This function's blocking behavior means it effectively **owns the main thread** in single-threaded server implementations. The loop must handle both normal connection acceptance and graceful shutdown signaling without dropping in-flight connections.

Connection context management through `init_connection_context` establishes the data structures that track each client connection throughout its processing lifecycle. The context includes not just the client file descriptor, but also timing information for timeout enforcement, client address data for logging and security decisions, and space for connection-specific state that accumulates during request processing.

The interface's **error reporting strategy** uses return codes rather than exceptions or global error state, following C language conventions and ensuring that error conditions can be handled immediately at the call site. Negative return values indicate errors, zero indicates success for status-returning functions, and positive values carry meaningful data (like file descriptors).

**Resource ownership semantics** throughout the interface follow a consistent pattern: the caller owns any resources returned by the interface methods and bears responsibility for cleanup. This ownership model prevents the common resource management confusion that occurs when ownership transfers unpredictably between components. The interface provides explicit cleanup functions rather than relying on automatic resource management.

### Connection Acceptance Algorithm Implementation

The connection acceptance algorithm implementation must handle the inherent **asynchronous nature** of network operations while maintaining deterministic server behavior. Network clients can connect, send partial data, and disconnect at any time, creating numerous edge cases that must be handled gracefully without corrupting server state.

The algorithm's **blocking semantics** on the accept call create an important architectural constraint. When `accept()` blocks waiting for connections, the calling thread cannot perform other operations. This blocking behavior is actually beneficial in many scenarios - it provides natural flow control and prevents the server from consuming CPU cycles when no work is available. However, it also means that server shutdown must occur through signal handling or secondary threads, since the main thread remains blocked in the accept call.

**File descriptor management** within the algorithm requires careful attention to avoid resource leaks that would eventually crash the server. Each successful accept operation creates a new file descriptor that represents the client connection. These file descriptors are a limited system resource - most operating systems impose per-process limits between 1,024 and 65,536 open file descriptors. Failing to close client connections eventually exhausts this limit and prevents new connections.

The algorithm must handle **partial connection establishment** scenarios where clients begin the TCP handshake but fail to complete it properly. The `accept()` call returns successfully once the TCP three-way handshake completes, but clients might immediately close the connection or send malformed data. The algorithm treats these as normal occurrences rather than error conditions, logging them for debugging purposes but continuing normal operation.

**Client address extraction** from the `sockaddr_in` structure returned by accept provides valuable debugging and security information. The client's IP address and port number help diagnose connection issues and can be used for implementing security policies like rate limiting or geographic restrictions. However, this address information reflects the immediate network peer, which might be a proxy or load balancer in production deployments rather than the actual end user.

The algorithm's **error classification logic** distinguishes between different types of accept failures to determine appropriate responses. Temporary failures like `EMFILE` (too many open files) or `EAGAIN` (resource temporarily unavailable) indicate resource pressure but don't require server shutdown. Persistent failures like `EBADF` (bad file descriptor) or `EINVAL` (invalid argument) indicate programming errors or socket corruption that require investigation.

**Connection validation** occurs immediately after successful connection establishment, before any data exchange begins. This validation checks that the new connection fits within configured resource limits, that the client address doesn't violate security policies, and that the server has sufficient resources to handle request processing. Connections that fail validation are rejected with appropriate logging but don't impact other active connections.

The algorithm includes **graceful degradation mechanisms** for handling resource pressure without complete service failure. When connection limits are approached, the server can choose to reject new connections with meaningful error responses rather than accepting connections it cannot properly service. This degradation provides better client experience than accepting connections that will eventually timeout due to resource starvation.

**Threading considerations** affect how the connection acceptance algorithm integrates with concurrency models implemented in later milestones. In thread-per-connection models, each accepted connection gets dispatched to a dedicated thread immediately. In event-driven models, accepted connections are added to a monitoring set for asynchronous processing. The algorithm's design supports both approaches through its clear separation of connection establishment and request processing.

### Architecture Decision Records: Advanced Considerations

Several additional architectural decisions significantly impact the TCP server's operational characteristics and integration with other system components. These decisions often involve subtle trade-offs that become apparent only during implementation and testing phases.

> **Decision: Socket Buffer Size Configuration**
> - **Context**: Operating system default socket buffer sizes optimize for general network usage but may not suit HTTP server workloads. HTTP requests vary dramatically in size (from simple GET requests under 200 bytes to large POST requests with file uploads), and buffer sizing affects memory usage and performance.
> - **Options Considered**:
>   1. Use operating system defaults without modification
>   2. Configure explicit send/receive buffer sizes based on expected request sizes
>   3. Dynamic buffer sizing that adapts to observed request patterns
> - **Decision**: Use operating system defaults with monitoring points for future optimization
> - **Rationale**: Operating system defaults represent decades of networking optimization for typical workloads. Premature optimization of socket buffers often provides minimal benefit while adding configuration complexity. Monitoring points allow future measurement-driven optimization without initial complexity.
> - **Consequences**: Simplifies initial implementation and reduces configuration surface area. May require revisiting for high-performance scenarios, but provides solid foundation for educational objectives.

> **Decision: IPv6 Support Strategy**
> - **Context**: Modern networking environments increasingly use IPv6, but IPv6 socket programming adds significant complexity through dual-stack configuration, address family negotiation, and platform-specific behaviors.
> - **Options Considered**:
>   1. IPv4 only using AF_INET sockets exclusively
>   2. IPv6 only using AF_INET6 sockets exclusively  
>   3. Dual-stack supporting both protocols through separate sockets or IPv6 mapped addresses
> - **Decision**: IPv4 only for educational implementation with architecture supporting future IPv6 extension
> - **Rationale**: IPv4 socket programming provides clearer learning path with fewer platform-specific complications. Dual-stack networking involves subtle protocol interactions that distract from core HTTP server concepts. IPv4 remains widely supported and sufficient for learning objectives.
> - **Consequences**: Limits deployment scenarios but maintains focus on core concepts. Architecture design supports adding IPv6 support as advanced extension without requiring complete reimplementation.

> **Decision: Connection Timeout Configuration**
> - **Context**: Client connections can remain idle indefinitely, consuming server resources without productive work. HTTP connections may experience network delays, slow clients, or malicious attempts to exhaust server resources through connection hoarding.
> - **Options Considered**:
>   1. No timeouts relying on client behavior and operating system defaults
>   2. Hard timeouts that forcibly close connections after fixed duration
>   3. Activity-based timeouts that reset when data is exchanged
> - **Decision**: Configurable hard timeouts with reasonable defaults
> - **Rationale**: Educational servers need protection against resource exhaustion during testing with imperfect client code. Hard timeouts provide predictable resource reclamation without complex activity tracking logic. Configurable values allow adaptation to different testing scenarios.
> - **Consequences**: Prevents resource exhaustion during development but may close legitimate slow connections. Provides foundation for implementing more sophisticated timeout strategies in advanced versions.

The timeout decision particularly impacts **testing and debugging workflows** because developers often pause execution in debuggers or step through code slowly while examining server behavior. Aggressive timeouts can interfere with debugging by closing connections while the developer investigates server state. The configurable approach allows disabling timeouts during debugging while maintaining protection during automated testing.

**Platform portability** considerations influence several interface design decisions, particularly around socket option availability and error code interpretation. Windows socket programming uses different error codes than Unix systems, and some socket options have different names or behaviors. The interface design acknowledges these differences without attempting complete abstraction, allowing platform-specific optimizations while maintaining core functionality across operating systems.

### Common TCP Pitfalls

TCP socket programming contains numerous subtle pitfalls that can create intermittent bugs, resource leaks, or security vulnerabilities. Understanding these pitfalls helps developers recognize and avoid common mistakes that can take hours to debug in complex networking code.

⚠️ **Pitfall: Network Byte Order Confusion**

One of the most common mistakes in socket programming involves **byte order conversion** when working with port numbers and IP addresses. The `sockaddr_in` structure expects port numbers in **network byte order** (big-endian), but most development machines use little-endian byte order for local variables.

The symptom appears as binding or connection failures with confusing error messages. A server configured to listen on port 8080 might actually attempt to bind to port 20737 (8080 with bytes swapped) if the `htons()` conversion is forgotten. This creates "Address already in use" errors that seem to occur randomly depending on what other services are running.

The fix requires consistent use of byte order conversion functions: `htons()` for port numbers going into socket structures, and `ntohs()` for extracting port numbers from socket structures. IP addresses require `htonl()` and `ntohl()` conversion, though using `INADDR_ANY` avoids this issue for server binding.

Prevention involves establishing a coding convention where all socket structure assignments use explicit byte order conversion, even when the values appear to work without conversion on the development machine. This consistency prevents subtle bugs when deploying to different architectures.

⚠️ **Pitfall: Ignoring Partial Reads**

Network I/O operations can return **fewer bytes than requested** even when more data is available and the connection remains active. The `read()` system call provides no guarantee about how much data it will return - it might return 1 byte, the full requested amount, or anything in between.

This behavior creates intermittent parsing failures that are difficult to reproduce consistently. An HTTP request header might be split across multiple read operations, causing the parser to see incomplete lines or fragmented header fields. The resulting parsing errors often appear random and platform-dependent.

The solution requires implementing **complete request reading logic** that continues reading until a full HTTP request has been received, rather than assuming a single read operation will capture the entire request. This involves detecting the end of HTTP headers (the empty line containing only CRLF) and reading any message body based on the Content-Length header.

The `read_complete_request()` function addresses this pitfall by encapsulating the multi-read logic needed for reliable HTTP request reception. This function continues reading data until it has assembled a complete request or encounters a genuine error condition.

⚠️ **Pitfall: File Descriptor Leaks**

Every successful `accept()` call creates a new **file descriptor** representing the client connection. These file descriptors must be explicitly closed when connection processing completes, or they accumulate over time until the server exhausts its file descriptor limit and can no longer accept connections.

File descriptor leaks often go unnoticed during short-term testing but become apparent during longer test runs or load testing. The server appears to work correctly initially, then suddenly starts rejecting all new connections with "Too many open files" errors. This failure mode can be particularly confusing because existing connections continue working normally.

The solution requires systematic **resource cleanup** at every exit point from connection processing code. This includes normal completion paths, error conditions, and exception scenarios. Using `ConnectionContext` structures with explicit cleanup functions helps ensure that connection-related resources are properly released.

Prevention involves establishing clear **resource ownership patterns** where each component that acquires resources takes responsibility for releasing them. The TCP server component owns the server socket and is responsible for closing it during shutdown. Connection processing components own client file descriptors and must close them when processing completes.

⚠️ **Pitfall: Signal Handling Interference**

Unix signal handling can interfere with **blocking system calls** like `accept()`, causing them to return with `EINTR` errors when signals are delivered. This creates intermittent connection acceptance failures that appear random and platform-dependent.

The symptom manifests as occasional accept failures during normal operation, particularly when debugging tools or system monitoring generates signals. The server might work perfectly during casual testing but fail intermittently under production-like monitoring conditions.

The fix involves checking for `EINTR` return codes from `accept()` and retrying the operation rather than treating it as a genuine error condition. This **restart logic** should be applied consistently to all blocking system calls that can be interrupted by signals.

More sophisticated signal handling involves masking signals during critical operations or using dedicated signal handling threads, but these approaches add complexity that may not be justified for educational implementations.

⚠️ **Pitfall: Binding to Privileged Ports**

Port numbers below 1024 are considered **privileged ports** that require root permissions for binding on Unix systems. Attempting to bind to port 80 (standard HTTP) or 443 (HTTPS) will fail with permission errors unless the server runs with elevated privileges.

This pitfall often surprises developers who test their server on port 8080 successfully but encounter permission errors when attempting to deploy on standard HTTP ports. The error messages vary across operating systems but typically involve "Permission denied" or "Operation not permitted" responses.

The solution for educational purposes involves using **unprivileged ports** (above 1024) like 8080 or 3000 for development and testing. Production deployment can use port forwarding, reverse proxies, or container networking to map standard ports to unprivileged ports where the server actually listens.

Advanced deployment scenarios might involve **privilege dropping** where the server starts with root permissions to bind privileged ports, then drops to a non-privileged user for request processing. This approach requires careful sequencing of privilege changes and resource allocation.

⚠️ **Pitfall: Address Already in Use Errors**

When a server shuts down, the operating system maintains the socket in a **TIME_WAIT state** for several minutes to handle any delayed packets from recently closed connections. Attempting to restart the server during this period fails with "Address already in use" errors that prevent binding to the same port.

This behavior is particularly frustrating during development when rapid restart cycles are common for testing changes. The server appears to work correctly but becomes inaccessible for several minutes after each shutdown, significantly slowing development iteration.

The `SO_REUSEADDR` socket option allows immediate port reuse by instructing the operating system to bypass the TIME_WAIT restriction. This option must be set before binding the socket to be effective. Setting it after binding has no effect on the current connection.

Care must be taken when using `SO_REUSEADDR` in production environments, as it can allow multiple processes to bind to the same port simultaneously in some configurations, creating unpredictable connection routing behavior.

### Implementation Guidance

The TCP server component provides the networking foundation that all other components build upon. Understanding the socket programming patterns demonstrated here will prove essential when implementing concurrent connection handling in later milestones.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Socket Creation | Standard BSD sockets with basic error checking | BSD sockets with comprehensive option tuning |
| Address Configuration | Hard-coded INADDR_ANY with configurable port | Full address parsing with IPv4/IPv6 detection |
| Error Handling | Printf logging with perror for system errors | Structured logging with error code classification |
| Resource Management | Manual cleanup with explicit function calls | RAII-style cleanup with automatic resource tracking |

#### Recommended File Structure

The TCP server implementation should be organized to separate socket management logic from higher-level server coordination:

```
project-root/
├── src/
│   ├── main.c                    ← server entry point and main loop
│   ├── tcp_server.c              ← TCP server component implementation
│   ├── tcp_server.h              ← TCP server interface definitions
│   ├── server_config.c           ← configuration management
│   └── common.h                  ← shared constants and types
├── include/
│   ├── http_types.h              ← HTTP data structure definitions
│   └── network_utils.h           ← networking utility functions
└── tests/
    ├── tcp_server_test.c         ← unit tests for TCP functionality
    └── integration_test.c        ← end-to-end connection testing
```

#### Infrastructure Starter Code

Here's the complete TCP server foundation that handles socket creation, binding, and basic connection management:

```c
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DEFAULT_PORT 8080
#define MAX_PATH_LENGTH 1024
#define MAX_HEADERS 32
#define MAX_REQUEST_SIZE 8192
#define CONNECTION_BACKLOG 128

// Complete ServerConfig structure with all fields initialized
typedef struct {
    int port;
    char document_root[MAX_PATH_LENGTH];
    int max_threads;
    int max_connections;
    int request_timeout_sec;
} ServerConfig;

// Complete ConnectionContext for tracking client state
typedef struct {
    int client_fd;
    struct sockaddr_in client_addr;
    time_t connect_time;
    pthread_t thread_id;
} ConnectionContext;

// Initialize server configuration with safe defaults
void init_server_config(ServerConfig* config) {
    memset(config, 0, sizeof(ServerConfig));
    config->port = DEFAULT_PORT;
    strncpy(config->document_root, "./public", MAX_PATH_LENGTH - 1);
    config->max_threads = 50;
    config->max_connections = 100;
    config->request_timeout_sec = 30;
}

// Initialize connection context with client information
void init_connection_context(ConnectionContext* context, int client_fd, struct sockaddr_in addr) {
    memset(context, 0, sizeof(ConnectionContext));
    context->client_fd = client_fd;
    context->client_addr = addr;
    context->connect_time = time(NULL);
    context->thread_id = 0;  // Will be set when assigned to thread
}

// Complete socket creation and configuration
int create_server_socket(int port) {
    int server_fd;
    struct sockaddr_in server_addr;
    int reuse_opt = 1;
    
    // Create TCP socket
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("Failed to create server socket");
        return -1;
    }
    
    // Set SO_REUSEADDR to allow rapid server restart
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &reuse_opt, sizeof(reuse_opt)) < 0) {
        perror("Failed to set SO_REUSEADDR");
        close(server_fd);
        return -1;
    }
    
    // Configure server address structure
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;  // Accept on all interfaces
    server_addr.sin_port = htons(port);        // Convert to network byte order
    
    // Bind socket to address
    if (bind(server_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("Failed to bind server socket");
        close(server_fd);
        return -1;
    }
    
    // Begin listening for connections
    if (listen(server_fd, CONNECTION_BACKLOG) < 0) {
        perror("Failed to listen on server socket");
        close(server_fd);
        return -1;
    }
    
    printf("Server listening on port %d\n", port);
    return server_fd;
}

// Accept single client connection with full error handling
int accept_client_connection(int server_fd, ConnectionContext* context) {
    struct sockaddr_in client_addr;
    socklen_t addr_len = sizeof(client_addr);
    int client_fd;
    
    // Accept incoming connection
    client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &addr_len);
    if (client_fd < 0) {
        if (errno == EINTR) {
            // Signal interruption - retry accept
            return 0;  // Indicate retry needed
        }
        perror("Failed to accept client connection");
        return -1;
    }
    
    // Initialize connection context
    init_connection_context(context, client_fd, client_addr);
    
    // Log client connection for debugging
    printf("Accepted connection from %s:%d (fd=%d)\n",
           inet_ntoa(client_addr.sin_addr),
           ntohs(client_addr.sin_port),
           client_fd);
    
    return client_fd;
}

// Properly close client connection and cleanup resources
void close_client_connection(ConnectionContext* context) {
    if (context->client_fd > 0) {
        printf("Closing connection fd=%d\n", context->client_fd);
        close(context->client_fd);
        context->client_fd = -1;
    }
}
```

#### Core Logic Skeleton Code

The main server loop represents the core logic that students should implement themselves. This skeleton provides the structure while requiring students to understand the connection handling flow:

```c
// Main server loop - students implement the connection handling logic
void server_main_loop(int server_fd) {
    ConnectionContext context;
    char request_buffer[MAX_REQUEST_SIZE];
    
    printf("Server entering main loop, waiting for connections...\n");
    
    while (1) {  // Infinite loop until shutdown signal
        // TODO 1: Accept incoming client connection using accept_client_connection()
        // Hint: Handle EINTR return by continuing the loop
        // Hint: Log connection details for debugging
        
        // TODO 2: Read complete HTTP request from client socket
        // Hint: Use read_complete_request() to handle partial reads
        // Hint: Check for read errors and connection closure
        
        // TODO 3: Send hardcoded HTTP response for Milestone 1
        // Hint: Start with "HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\nHello World!\n"
        // Hint: Use write() or send() to transmit response
        
        // TODO 4: Clean up connection resources
        // Hint: Always close client file descriptor
        // Hint: Log connection closure for debugging
        
        // TODO 5: Check for server shutdown signals
        // Hint: Can use global variable set by signal handler
        // Hint: Break from loop to allow graceful shutdown
    }
    
    printf("Server main loop exiting\n");
}

// Complete request reading handling partial reads - students implement the read loop
ssize_t read_complete_request(int sockfd, char* buffer, size_t max_size) {
    // TODO 1: Initialize read tracking variables
    // Hint: Track total bytes read and current buffer position
    
    // TODO 2: Read data in loop until complete request received
    // Hint: HTTP request ends with \r\n\r\n (empty line after headers)
    // Hint: Handle partial reads by continuing to read more data
    
    // TODO 3: Handle read errors and connection closure
    // Hint: read() returns 0 when client closes connection
    // Hint: read() returns -1 on errors, check errno for details
    
    // TODO 4: Validate request doesn't exceed buffer size
    // Hint: Prevent buffer overflow by checking max_size
    // Hint: Return error if request too large for buffer
    
    // TODO 5: Null-terminate buffer and return total bytes read
    // Hint: Ensure buffer is valid C string for parsing
    // Hint: Return total bytes read on success, -1 on error
    
    return -1;  // Students replace with actual implementation
}

// Main function with server initialization and cleanup
int main(int argc, char* argv[]) {
    ServerConfig config;
    int server_fd;
    
    // TODO 1: Initialize server configuration
    // Hint: Use init_server_config() and handle command-line port override
    
    // TODO 2: Create and configure server socket
    // Hint: Use create_server_socket() and check for errors
    
    // TODO 3: Enter main connection handling loop
    // Hint: Use server_main_loop() - this runs until shutdown
    
    // TODO 4: Cleanup server resources on shutdown
    // Hint: Close server file descriptor
    // Hint: Log clean shutdown for debugging
    
    return 0;
}
```

#### Language-Specific Hints

C socket programming requires attention to several platform-specific details and common conventions:

- **Include Headers**: Use `#include <sys/socket.h>`, `#include <netinet/in.h>`, and `#include <arpa/inet.h>` for socket functions. These headers are standard on Unix-like systems but may require additional setup on Windows.

- **Error Checking**: Always check return values from socket functions. Use `perror()` to print meaningful error messages that include the system's description of what went wrong. Socket functions typically return -1 on error and set `errno` to indicate the specific problem.

- **Byte Order**: Use `htons()` for port numbers and `htonl()` for IP addresses when populating socket address structures. These functions convert from host byte order to network byte order automatically, handling platform differences transparently.

- **Address Structures**: Initialize `sockaddr_in` structures with `memset()` before setting individual fields. This ensures that padding bytes are zeroed, preventing subtle bugs from uninitialized memory affecting socket operations.

- **File Descriptor Management**: Treat file descriptors as precious resources that must be explicitly managed. Use consistent patterns for checking validity (fd > 0), closing resources (always check close() return value), and setting invalid markers (fd = -1 after close).

- **Signal Safety**: Be aware that signal delivery can interrupt socket system calls. Production code often blocks signals during critical operations or handles `EINTR` returns consistently across all blocking calls.

#### Milestone Checkpoint

After implementing the TCP server component, verify correct operation through systematic testing:

**Basic Functionality Test**: Start your server and verify it accepts connections properly:
```bash
# Terminal 1: Start your server
./http_server

# Terminal 2: Connect with telnet to verify TCP connection works
telnet localhost 8080
# You should see "Connected to localhost"
# Type anything and press Enter - server should respond with hardcoded HTTP response
# Connection should close automatically after response
```

**Expected Output**: The server should print connection acceptance messages including client IP address and file descriptor numbers. The telnet client should receive your hardcoded HTTP response and the connection should close cleanly.

**Multiple Connection Test**: Verify the server accepts multiple sequential connections:
```bash
# Run several telnet connections in succession
for i in {1..5}; do
  echo "Test request $i" | telnet localhost 8080
done
```

**Expected Behavior**: Each connection should be accepted, receive a response, and close properly. File descriptor numbers should increment for each connection. No "Address already in use" errors should occur.

**Error Condition Testing**: Verify proper error handling by testing edge cases:
- Start server on port already in use (should fail gracefully with clear error message)
- Connect with telnet but send no data (should timeout appropriately)
- Kill server with Ctrl+C and restart immediately (should work due to SO_REUSEADDR)

**Signs of Problems**: 
- "Address already in use" errors on restart indicate missing `SO_REUSEADDR` configuration
- Connections timing out indicate problems with the accept loop or response sending
- Increasing file descriptor numbers without corresponding closes indicate file descriptor leaks
- Server crashes or hangs indicate unhandled error conditions in the main loop

**Debugging Steps**: If tests fail, check these common issues:
1. Verify socket creation returns valid file descriptor (not -1)
2. Confirm `htons()` is used for port number conversion
3. Check that server socket is properly closed on shutdown
4. Verify client file descriptors are closed after each connection
5. Confirm error checking covers all system calls with appropriate perror() messages


## HTTP Parser Component

> **Milestone(s):** Milestone 2 (HTTP Request Parsing) - transforms raw TCP data into structured HTTP message representations that enable proper request processing and response generation

### HTTP Parsing Mental Model: Mail Sorting Analogy

Think of HTTP parsing like working at a post office mail sorting facility. When mail trucks arrive at the facility, they dump large bags of unsorted mail onto conveyor belts. Your job as a mail sorter is to examine each piece of mail, read the addresses, validate the format, and sort everything into the correct bins for delivery.

HTTP parsing works exactly the same way. The TCP server component delivers raw bytes from the network - like those unsorted mail bags. The HTTP parser acts as the mail sorter, examining the incoming byte stream character by character. Just as mail has a specific format (recipient address, return address, postal codes in specific positions), HTTP messages follow the RFC 7230 specification with request lines, headers, and optional message bodies in precise positions.

A mail sorter must handle various challenges: damaged envelopes (malformed requests), missing addresses (invalid HTTP syntax), packages too large for standard processing (requests exceeding size limits), and different mail formats from various countries (different line ending conventions). Similarly, the HTTP parser must gracefully handle malformed input, missing required fields, oversized requests, and protocol variations while maintaining strict adherence to standards.

The sorting process is methodical and stateful. A mail sorter doesn't randomly grab pieces - they process items in order, moving through distinct phases: examining the envelope format, reading the destination address, checking for proper postage, then routing to the appropriate bin. HTTP parsing follows the same methodical approach: first parsing the request line (method, path, version), then processing headers one by one, and finally handling any message body content.

Just as a mail facility has quality control processes to catch problems early and prevent delivery errors, HTTP parsers implement strict validation at each step. A single malformed address doesn't shut down the entire facility - it gets flagged for special handling. Similarly, one bad HTTP request shouldn't crash the server; it should generate an appropriate error response and continue processing other requests.

### Parser Interface

The HTTP parser component exposes a clean, focused API that transforms raw network data into structured `HTTPRequest` objects. The interface design prioritizes safety, clarity, and error handling while maintaining the stateless nature that enables concurrent request processing.

The parser operates through a primary parsing function that accepts raw request data and produces either a valid `HTTPRequest` structure or a specific error condition. This design choice eliminates the complexity of streaming parsers while providing sufficient functionality for the educational goals of this project.

| Method Name | Parameters | Returns | Description |
|-------------|------------|---------|-------------|
| `parse_http_request` | `const char* raw_data`, `size_t data_length`, `HTTPRequest* request` | `int` (status code) | Parses complete HTTP request from raw bytes into structured request object |
| `parse_request_line` | `const char* line`, `HTTPRequest* request` | `int` (status code) | Extracts method, path, and version from HTTP request line |
| `parse_header_line` | `const char* line`, `HTTPRequest* request` | `int` (status code) | Parses single header line and adds to request header collection |
| `parse_message_body` | `const char* body_start`, `size_t content_length`, `HTTPRequest* request` | `int` (status code) | Processes message body content based on Content-Length header |
| `validate_request_format` | `const HTTPRequest* request` | `int` (status code) | Validates parsed request meets HTTP/1.1 requirements |
| `find_header_value` | `const HTTPRequest* request`, `const char* header_name` | `const char*` | Retrieves header value by name (case-insensitive search) |

The parser interface follows a consistent error handling pattern where return values indicate parsing status. Success returns zero, while positive integers represent specific HTTP error conditions that should be returned to the client (400 Bad Request, 414 URI Too Long, etc.). This approach enables the calling code to automatically generate appropriate HTTP error responses without complex error message parsing.

**Parser State and Memory Management**

The parser maintains no internal state between calls, making it inherently thread-safe and suitable for concurrent request processing. Each invocation operates on caller-provided buffers and data structures, eliminating hidden dependencies and resource cleanup complexity.

Memory management follows a clear ownership model. The caller provides the destination `HTTPRequest` structure and retains ownership of all memory. The parser populates fields within the provided structure but does not allocate dynamic memory for basic request processing. For requests with message bodies, the parser sets pointer fields to reference existing buffer locations rather than creating copies.

| Memory Responsibility | Parser Role | Caller Role |
|----------------------|-------------|-------------|
| Input buffer management | Read-only access to raw request data | Allocate sufficient buffer, ensure null termination |
| HTTPRequest structure | Populate fields within provided structure | Allocate structure, call cleanup when finished |
| Header storage | Copy header names and values to fixed arrays | Provide MAX_HEADERS array space in structure |
| Message body handling | Set pointer to body location in input buffer | Ensure buffer remains valid during request processing |
| String null termination | Ensure all string fields are properly terminated | Validate string lengths before use |

**Content-Length and Body Processing**

The parser implements careful Content-Length processing to handle requests with message bodies safely. When a Content-Length header is present, the parser validates that sufficient data exists in the input buffer before setting the body pointer. This prevents buffer overruns while enabling efficient zero-copy body access.

For requests without Content-Length headers, the parser assumes no message body exists, which is appropriate for GET requests and compliant with HTTP/1.1 specifications. The parser does not attempt to implement Transfer-Encoding chunked processing, as this falls outside the scope of a basic static file server.

### HTTP Parsing Algorithm: State Machine for Processing HTTP Message Format

The HTTP parser operates as a finite state machine that processes the incoming request character by character, transitioning through distinct states based on the HTTP message format. This approach provides precise control over parsing while enabling clear error detection and recovery.

![HTTP Parser State Machine](./diagrams/http-parser-state-machine.svg)

The state machine recognizes that HTTP messages have a rigid structure defined by RFC 7230: a request line, zero or more header lines, a blank line separator, and an optional message body. Each component has specific format requirements that must be validated during parsing.

**State Machine Implementation**

| Current State | Input Event | Next State | Action Taken |
|---------------|-------------|------------|--------------|
| PARSER_START | Any character | READING_REQUEST_LINE | Begin accumulating request line characters |
| READING_REQUEST_LINE | Printable character | READING_REQUEST_LINE | Append character to request line buffer |
| READING_REQUEST_LINE | CRLF or LF | PARSING_REQUEST_LINE | Process complete request line, extract method/path/version |
| PARSING_REQUEST_LINE | Processing success | READING_HEADERS | Prepare to process header lines |
| PARSING_REQUEST_LINE | Processing failure | PARSER_ERROR | Set error code for malformed request line |
| READING_HEADERS | Printable character | READING_HEADER_LINE | Begin accumulating current header line |
| READING_HEADER_LINE | Printable character | READING_HEADER_LINE | Append character to header line buffer |
| READING_HEADER_LINE | CRLF or LF | PARSING_HEADER_LINE | Process complete header line |
| PARSING_HEADER_LINE | Valid header format | READING_HEADERS | Add header to request, continue header processing |
| PARSING_HEADER_LINE | Invalid header format | PARSER_ERROR | Set error code for malformed header |
| READING_HEADERS | CRLF or LF (empty line) | CHECKING_BODY | Headers complete, determine if body expected |
| CHECKING_BODY | Content-Length > 0 | READING_BODY | Prepare to read specified number of body bytes |
| CHECKING_BODY | No Content-Length | PARSER_COMPLETE | Request parsing complete (no body) |
| READING_BODY | Sufficient bytes available | PARSER_COMPLETE | Set body pointer, parsing complete |
| READING_BODY | Insufficient bytes | PARSER_ERROR | Set error code for incomplete body |

**Parsing Algorithm Implementation**

The concrete parsing algorithm follows these numbered steps to process HTTP requests systematically:

1. **Initialize parser state** by setting the state machine to PARSER_START and clearing all temporary buffers. Validate that the input buffer contains at least a minimal HTTP request (method + space + path + space + version + CRLF).

2. **Locate request line boundary** by scanning from the beginning of the input buffer to find the first CRLF or LF sequence. This marks the end of the request line and enables safe string processing of the line content.

3. **Extract method field** by finding the first space character in the request line. Copy all characters from the line start to the space into the request method field, ensuring null termination and length validation against the method buffer size.

4. **Extract path field** by finding the second space character in the request line. Copy characters between the first and second spaces into the request path field, performing length validation and URL decoding if necessary.

5. **Extract version field** by copying remaining characters from the second space to the line end into the request version field. Validate that the version matches expected formats like "HTTP/1.1" or "HTTP/1.0".

6. **Process header lines sequentially** by advancing past the request line and examining each subsequent line until encountering an empty line (CRLF CRLF or LF LF). For each header line, locate the colon separator and extract header name and value.

7. **Validate header format** by ensuring each header line contains exactly one colon character, has a non-empty header name, and follows HTTP header naming conventions. Trim whitespace from header values while preserving significant spaces.

8. **Store headers in request structure** by copying header names and values into the request headers array. Implement bounds checking to prevent buffer overflow when the number of headers exceeds MAX_HEADERS.

9. **Determine message body presence** by examining the Content-Length header value. If Content-Length exists and is greater than zero, prepare to read the specified number of bytes as the message body.

10. **Process message body data** by setting the body pointer to reference the appropriate location in the input buffer and updating the body_length field. Validate that sufficient bytes remain in the buffer to satisfy the Content-Length requirement.

**Error Detection and Recovery**

The parsing algorithm implements comprehensive error detection at each stage to catch malformed requests early and provide meaningful error responses. Rather than failing silently or crashing, the parser generates specific HTTP error codes that can be returned to the client.

| Error Condition | Detection Method | HTTP Status Code | Recovery Action |
|-----------------|------------------|------------------|-----------------|
| Missing request line | No CRLF found in first 8192 bytes | 400 Bad Request | Reject request, close connection |
| Invalid method | Method not in allowed set or contains invalid characters | 405 Method Not Allowed | Return error response |
| Malformed URL path | Path contains null bytes or exceeds maximum length | 400 Bad Request | Return error response |
| Unsupported HTTP version | Version not "HTTP/1.0" or "HTTP/1.1" | 505 HTTP Version Not Supported | Return error response |
| Missing header colon | Header line does not contain colon separator | 400 Bad Request | Return error response |
| Too many headers | Header count exceeds MAX_HEADERS | 431 Request Header Fields Too Large | Return error response |
| Invalid Content-Length | Content-Length not a valid positive integer | 400 Bad Request | Return error response |
| Incomplete message body | Available data less than Content-Length | 400 Bad Request | Return error response |

### Architecture Decision Records

**Decision: Synchronous Single-Pass Parsing**
- **Context**: HTTP parsing can be implemented using streaming parsers that process data incrementally, or single-pass parsers that require complete request data. Streaming parsers handle partial data arrival but add state management complexity.
- **Options Considered**: 
  - Streaming parser with internal state buffers
  - Single-pass parser requiring complete request data
  - Hybrid approach with optional streaming support
- **Decision**: Implement single-pass parser requiring complete request data
- **Rationale**: The TCP server component already handles partial reads and assembles complete requests before calling the parser. Single-pass parsing eliminates state management complexity, improves testability, and provides better performance for small-to-medium requests typical in educational scenarios.
- **Consequences**: Simpler implementation and testing, but requires complete request buffering. Maximum request size becomes a hard limit rather than a soft constraint.

| Option | Pros | Cons |
|--------|------|------|
| Streaming parser | Handles very large requests, lower memory usage | Complex state management, harder to test |
| Single-pass parser | Simple implementation, stateless, fast for small requests | Requires complete buffering, memory usage scales with request size |
| Hybrid approach | Flexibility for different use cases | Implementation complexity, unclear API boundaries |

> **Key Design Insight**: Educational HTTP servers benefit more from implementation clarity than handling edge cases like extremely large uploads that rarely occur in static file serving scenarios.

**Decision: Fixed-Size Header Storage**
- **Context**: HTTP headers can be stored in dynamically allocated structures that grow as needed, or in fixed-size arrays with compile-time limits. Dynamic allocation provides unlimited scalability but complicates memory management.
- **Options Considered**:
  - Dynamic header allocation with linked lists or resizable arrays
  - Fixed-size header arrays with MAX_HEADERS limit
  - Hybrid with small fixed buffer and dynamic overflow
- **Decision**: Use fixed-size header arrays with MAX_HEADERS set to 32
- **Rationale**: Static file servers typically receive simple requests with 5-10 headers. A limit of 32 headers provides generous headroom while avoiding dynamic memory management complexity. Fixed-size allocation enables stack-based request structures and eliminates memory leaks.
- **Consequences**: Memory usage is predictable and bounded, implementation remains simple, but requests with excessive headers will be rejected. This aligns with defensive programming practices for server security.

**Decision: Case-Insensitive Header Matching**
- **Context**: HTTP header names are case-insensitive per RFC 7230, but header values are case-sensitive. The parser must decide how to store and match header names while preserving original case for debugging.
- **Options Considered**:
  - Store headers in original case, implement case-insensitive matching functions
  - Normalize all header names to lowercase during parsing
  - Store both original and normalized versions
- **Decision**: Store headers in original case with case-insensitive matching functions
- **Rationale**: Preserving original case aids debugging and maintains fidelity to the original request. Case-insensitive matching functions are straightforward to implement and perform adequately for the small number of headers in typical requests.
- **Consequences**: Header storage uses original case which helps with debugging, but header lookup requires case-insensitive string comparison. Performance impact is negligible for educational use cases.

**Decision: Strict HTTP/1.1 Compliance**
- **Context**: HTTP parsing can be lenient (accepting common deviations from standards) or strict (rejecting any non-compliant input). Lenient parsing improves compatibility but may accept invalid requests.
- **Options Considered**:
  - Lenient parsing that accepts common browser variations
  - Strict parsing that follows RFC 7230 exactly
  - Configurable strictness level
- **Decision**: Implement strict parsing that follows RFC 7230 requirements
- **Rationale**: Educational servers should demonstrate proper protocol implementation. Strict parsing helps students understand HTTP requirements and catches implementation errors in client code. Production servers can be more lenient, but learning servers should prioritize correctness.
- **Consequences**: Some non-compliant clients may be rejected, but this reinforces proper HTTP protocol understanding. Error messages help developers identify and fix client implementation issues.

### Common HTTP Parsing Pitfalls

⚠️ **Pitfall: Line Ending Confusion**
HTTP/1.1 specifies CRLF (\\r\\n) as the line terminator, but many clients send LF (\\n) only, especially during development testing. Parsers that only recognize CRLF will fail to parse requests from curl, telnet, or custom test clients. The parser must accept both CRLF and LF line endings while maintaining strict protocol compliance for production use. Implement line ending detection by checking for \\r\\n first, then falling back to \\n if not found. This approach maintains compatibility without compromising standards compliance.

⚠️ **Pitfall: Header Value Whitespace Handling**
RFC 7230 permits optional whitespace around header values (e.g., "Content-Length: 1234" vs "Content-Length:1234"), but naive parsing may include leading/trailing spaces in header values. This causes string comparison failures when looking for specific header values or parsing numeric headers like Content-Length. Always trim leading and trailing whitespace from header values during parsing, but preserve internal whitespace which may be significant. Use a trimming function that removes spaces and tabs but stops at other whitespace characters.

⚠️ **Pitfall: Buffer Overflow on Long Request Lines**
HTTP requests can contain extremely long URLs or header values that exceed fixed buffer sizes. Parsing without length checking can cause buffer overflows, memory corruption, and security vulnerabilities. Each parsing step must validate input length against available buffer space before copying data. Implement bounds checking by tracking remaining buffer space and comparing against input length before each copy operation. Return HTTP 414 URI Too Long or HTTP 431 Request Header Fields Too Large for oversized components.

⚠️ **Pitfall: Incomplete Content-Length Processing**
When Content-Length header is present, the parser must verify that sufficient data exists in the input buffer before setting the body pointer. Failure to validate available data length can cause the body pointer to reference memory beyond the buffer end, leading to crashes when the file handler attempts to read body content. Always compare Content-Length value against remaining buffer length after header parsing completes. Return HTTP 400 Bad Request if insufficient body data is available.

⚠️ **Pitfall: Method Case Sensitivity Issues**
HTTP methods like GET, POST, PUT are case-sensitive per RFC 7230, but some parsers incorrectly accept lowercase variants like "get" or "post". This creates compatibility issues with strict servers and reinforces incorrect HTTP understanding. Validate that method names match exact case requirements and reject requests with incorrect casing. Provide clear error messages that specify the expected case format to help developers correct client implementations.

⚠️ **Pitfall: Missing Null Termination**
When copying strings from the input buffer into request structure fields, failure to null-terminate the copied strings can cause string functions to read beyond buffer boundaries. This occurs frequently when using functions like `strncpy` that don't guarantee null termination. Always ensure copied strings are null-terminated by either using `snprintf` for copying or manually adding null terminators after `strncpy`. Reserve one byte in each string field for the null terminator.

⚠️ **Pitfall: Header Name Duplicate Handling**
HTTP allows multiple headers with the same name (like multiple Cookie headers), but simple parsers may overwrite earlier values when encountering duplicates. This loses important request information and can break applications that depend on multiple header values. Implement duplicate header handling by either concatenating values with comma separation (per RFC 7230) or storing only the first occurrence with a warning. Document the chosen behavior clearly for API users.

⚠️ **Pitfall: Integer Overflow in Content-Length**
Content-Length header values are parsed as integers, but extremely large values can cause integer overflow in languages with fixed-size integer types. This can result in negative content lengths or wrap-around to small positive values, bypassing security checks. Validate Content-Length values against reasonable maximum sizes before parsing to integer form. Use appropriate integer types that can handle expected maximum request sizes, and reject requests with Content-Length values exceeding implementation limits.

### Implementation Guidance

**A. Technology Recommendations Table:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| String Processing | Standard C string functions (strchr, strncpy, strcmp) | Custom parsing with explicit bounds checking |
| Memory Management | Stack-allocated request structures | Memory pools for zero-allocation parsing |
| Error Handling | Integer return codes with predefined constants | Structured error types with detailed context |
| Header Storage | Fixed arrays with linear search | Hash tables for O(1) header lookup |
| Line Ending Detection | Check for both \\r\\n and \\n patterns | Configurable line ending modes |

**B. Recommended File/Module Structure:**

```
project-root/
  src/
    http_parser.h              ← parser interface declarations
    http_parser.c              ← main parsing implementation
    http_parser_internal.h     ← internal parsing state definitions
    http_utils.c               ← string utilities and helper functions
  test/
    test_http_parser.c         ← comprehensive parser test suite
    test_requests/             ← sample HTTP request files
      valid_get_request.txt
      malformed_header.txt
      missing_version.txt
  include/
    http_types.h               ← HTTPRequest and HTTPResponse definitions
```

**C. Infrastructure Starter Code (COMPLETE, ready to use):**

```c
// http_utils.c - String processing utilities for HTTP parsing
#include <string.h>
#include <ctype.h>
#include <stdlib.h>

// Trims leading and trailing whitespace from a string in-place
char* trim_whitespace(char* str) {
    char* end;
    
    // Trim leading space
    while(isspace((unsigned char)*str)) str++;
    
    if(*str == 0) return str;  // All spaces?
    
    // Trim trailing space
    end = str + strlen(str) - 1;
    while(end > str && isspace((unsigned char)*end)) end--;
    
    // Write new null terminator
    end[1] = '\0';
    
    return str;
}

// Case-insensitive string comparison for header names
int strcasecmp_http(const char* s1, const char* s2) {
    while (*s1 && *s2) {
        char c1 = tolower((unsigned char)*s1);
        char c2 = tolower((unsigned char)*s2);
        if (c1 != c2) return c1 - c2;
        s1++;
        s2++;
    }
    return tolower((unsigned char)*s1) - tolower((unsigned char)*s2);
}

// Find line ending (CRLF or LF) and return pointer to start of next line
const char* find_line_end(const char* start, const char* buffer_end, size_t* line_length) {
    const char* pos = start;
    
    while (pos < buffer_end) {
        if (*pos == '\r' && pos + 1 < buffer_end && *(pos + 1) == '\n') {
            // Found CRLF
            *line_length = pos - start;
            return pos + 2;  // Skip past CRLF
        } else if (*pos == '\n') {
            // Found LF only
            *line_length = pos - start;
            return pos + 1;  // Skip past LF
        }
        pos++;
    }
    
    // No line ending found
    return NULL;
}

// Safely copy string with bounds checking and null termination
int safe_string_copy(char* dest, size_t dest_size, const char* src, size_t src_length) {
    if (src_length >= dest_size) {
        return -1;  // Source too large for destination
    }
    
    strncpy(dest, src, src_length);
    dest[src_length] = '\0';
    return 0;
}
```

**D. Core Logic Skeleton Code (signature + TODOs only):**

```c
// http_parser.c - Main HTTP request parsing implementation

#include "http_parser.h"
#include "http_utils.h"
#include <string.h>
#include <stdlib.h>

// Parses complete HTTP request from raw bytes into structured request object
// Returns 0 on success, HTTP error code on failure
int parse_http_request(const char* raw_data, size_t data_length, HTTPRequest* request) {
    // TODO 1: Initialize request structure with safe defaults using init_http_request()
    // TODO 2: Validate input parameters (non-null pointers, reasonable data_length)
    // TODO 3: Find request line boundary using find_line_end() helper
    // TODO 4: Parse request line (method, path, version) using parse_request_line()
    // TODO 5: Parse headers line by line until empty line found
    // TODO 6: Check for Content-Length header and process body if present
    // TODO 7: Validate complete request format using validate_request_format()
    // Hint: Track current position in buffer to avoid re-parsing same data
    // Hint: Return specific HTTP error codes (400, 414, etc.) for different failures
}

// Extracts method, path, and version from HTTP request line
// Example input: "GET /index.html HTTP/1.1"
int parse_request_line(const char* line, HTTPRequest* request) {
    // TODO 1: Find first space to separate method from path
    // TODO 2: Copy method to request->method with bounds checking
    // TODO 3: Find second space to separate path from version
    // TODO 4: Copy path to request->path with bounds checking
    // TODO 5: Copy version to request->version with bounds checking
    // TODO 6: Validate method is non-empty and contains valid characters
    // TODO 7: Validate path starts with '/' and doesn't contain null bytes
    // TODO 8: Validate version matches "HTTP/1.0" or "HTTP/1.1" format
    // Hint: Use safe_string_copy() helper to prevent buffer overflows
    // Hint: Return 400 for malformed request line, 405 for invalid method
}

// Parses single header line and adds to request header collection
// Example input: "Content-Length: 1234"
int parse_header_line(const char* line, HTTPRequest* request) {
    // TODO 1: Find colon separator between header name and value
    // TODO 2: Validate header name is non-empty and contains valid characters
    // TODO 3: Extract header name (everything before colon)
    // TODO 4: Extract header value (everything after colon)
    // TODO 5: Trim whitespace from header value using trim_whitespace()
    // TODO 6: Check if header_count exceeds MAX_HEADERS limit
    // TODO 7: Store header name and value in request->headers array
    // TODO 8: Increment request->header_count
    // Hint: Header names are case-insensitive but preserve original case
    // Hint: Return 431 if too many headers, 400 for malformed header format
}

// Processes message body content based on Content-Length header
int parse_message_body(const char* body_start, size_t available_bytes, HTTPRequest* request) {
    // TODO 1: Look up Content-Length header using get_request_header()
    // TODO 2: Parse Content-Length value to integer (handle parsing errors)
    // TODO 3: Validate Content-Length is non-negative and reasonable size
    // TODO 4: Check if available_bytes >= content_length
    // TODO 5: Set request->body pointer to body_start location
    // TODO 6: Set request->body_length to parsed content_length value
    // Hint: Content-Length of 0 is valid (empty body)
    // Hint: Return 400 for invalid Content-Length, 413 if body too large
}

// Retrieves header value by name (case-insensitive search)
const char* get_request_header(const HTTPRequest* request, const char* header_name) {
    // TODO 1: Iterate through request->headers array up to header_count
    // TODO 2: Compare each header name using strcasecmp_http() for case-insensitivity
    // TODO 3: Return pointer to header value when match found
    // TODO 4: Return NULL if header not found
    // Hint: Use strcasecmp_http() helper for proper case-insensitive comparison
}
```

**E. Language-Specific Hints:**

- **String Safety**: Use `strncpy()` instead of `strcpy()` to prevent buffer overflows, but remember that `strncpy()` doesn't guarantee null termination. Always manually add null terminators or use `snprintf()` for safer string copying.

- **Memory Layout**: The `HTTPRequest` structure uses fixed-size arrays to avoid dynamic allocation. This means stack allocation is safe and cleanup is automatic, but you must respect buffer boundaries.

- **Header Processing**: HTTP header names are case-insensitive, so "Content-Length", "content-length", and "CONTENT-LENGTH" should all match. Implement case-insensitive comparison using `tolower()` character by character.

- **Integer Parsing**: Use `strtol()` for parsing Content-Length values, and always check the return value and `errno` for parsing errors. Reject negative values and values exceeding reasonable limits.

- **Line Ending Handling**: Real HTTP requests may use CRLF (\\r\\n) or just LF (\\n). Check for CRLF first, then fall back to LF to maintain broad compatibility.

**F. Milestone Checkpoint:**

After implementing the HTTP parser component, verify functionality with these tests:

**Test Command:**
```bash
gcc -o test_parser test_http_parser.c http_parser.c http_utils.c -I../include
./test_parser
```

**Expected Behavior:**
- Parser correctly extracts "GET", "/index.html", "HTTP/1.1" from request line
- Headers like "Host: localhost" are stored as separate name/value pairs
- Case-insensitive header lookup finds "content-length" when searching for "Content-Length"
- Malformed requests return appropriate HTTP error codes (400, 405, 414)
- Parser handles both CRLF and LF line endings without errors

**Manual Testing with Sample Request:**
Create a file `test_request.txt`:
```
GET /test.html HTTP/1.1
Host: localhost:8080
User-Agent: TestClient/1.0
Content-Length: 0

```

The parser should extract method="GET", path="/test.html", version="HTTP/1.1", and three headers. Test with your parsing function to verify correct field population.

**G. Debugging Tips:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Parser crashes on valid requests | Buffer overflow in string copying | Use debugger to check buffer boundaries | Implement proper bounds checking in all string operations |
| Headers contain extra whitespace | Not trimming header values | Print header values with quotes to see whitespace | Apply trim_whitespace() to all header values |
| Case-sensitive header matching fails | Using strcmp instead of strcasecmp | Test with "content-length" vs "Content-Length" | Use case-insensitive comparison for header names |
| Parser accepts malformed requests | Missing validation steps | Test with intentionally bad requests | Add validation for each parsing step |
| Body parsing fails with Content-Length | Integer parsing or bounds checking error | Check Content-Length parsing and available buffer size | Validate Content-Length parsing and buffer bounds |


## File Handler Component

> **Milestone(s):** Milestone 3 (Static File Serving) - maps URL paths to filesystem paths and serves static content with proper security validation and MIME type detection

### File Serving Mental Model: Filing Cabinet Analogy

Think of the file handler component as a corporate filing cabinet system with a security guard. When someone requests a document, they don't walk directly to the filing cabinet - they submit a request to the security guard at the front desk. The security guard performs several critical checks before retrieving any document.

First, the guard validates that the requested document exists within the authorized filing system boundaries. They check the person's access badge (similar to validating the URL path stays within the document root). The guard ensures the request isn't trying to access restricted areas like the executive filing room or the basement archives (preventing directory traversal attacks). Once security validation passes, the guard locates the correct filing cabinet, opens the appropriate drawer, and retrieves the requested document.

The guard also knows what type of document they're handing over - whether it's a text memo, a PDF report, or a spreadsheet - and attaches the appropriate handling instructions (setting the correct Content-Type header). If the document doesn't exist, the guard politely informs the requester with a standardized "Document Not Found" response rather than revealing internal filing system details.

This mental model captures the three essential responsibilities of the file handler: **path security validation**, **file content retrieval**, and **metadata detection**. The security guard represents the validation logic that stands between external requests and the filesystem, ensuring that only authorized access occurs within defined boundaries.

The filing cabinet system also illustrates why **path resolution** must happen in a controlled manner. Just as the security guard follows established procedures for locating documents rather than allowing requesters to wander the filing room, the file handler must resolve paths through a secure, predictable process that prevents unauthorized access to system files or confidential data outside the document root.

### File Handler Interface

The file handler component exposes a clean interface that separates security validation, file operations, and response generation into distinct responsibilities. This separation allows the HTTP parser component to focus on message structure while delegating filesystem concerns to a specialized handler.

| Method Name | Parameters | Returns | Description |
|-------------|------------|---------|-------------|
| `validate_and_resolve_path` | `document_root char*`, `request_path char*`, `resolved_path char*`, `max_path_len size_t` | `int` (0 success, -1 error) | Validates request path against directory traversal attacks and resolves to absolute filesystem path within document root boundaries |
| `detect_mime_type` | `file_path char*`, `mime_type char*`, `max_mime_len size_t` | `int` (0 success, -1 error) | Determines MIME type based on file extension and sets Content-Type header value for HTTP response |
| `serve_file_content` | `resolved_path char*`, `response HTTPResponse*` | `int` (0 success, -1 file not found, -2 read error) | Reads file contents into response body and sets appropriate Content-Length header |
| `handle_directory_request` | `resolved_path char*`, `response HTTPResponse*` | `int` (0 success, -1 error) | Generates directory listing HTML when request path maps to a directory rather than a file |
| `generate_error_response` | `status_code int`, `error_message char*`, `response HTTPResponse*` | `void` | Creates standardized HTTP error response with appropriate status code and minimal error details |
| `check_file_permissions` | `file_path char*` | `int` (0 readable, -1 permission denied) | Verifies that the server process has read permissions for the requested file |

The interface design emphasizes **defensive programming** by requiring callers to provide pre-allocated buffers for path resolution and MIME type detection. This approach prevents memory allocation failures during request processing and ensures predictable memory usage patterns. Buffer sizes are validated through the `max_path_len` and `max_mime_len` parameters, providing protection against buffer overflow attacks.

The return codes follow a consistent pattern where zero indicates success and negative values indicate specific error conditions. This allows the HTTP parser component to translate file handler errors into appropriate HTTP status codes without requiring detailed knowledge of filesystem error semantics.

**Path Validation Interface Details:**

The `validate_and_resolve_path` method performs the most security-critical operation in the entire HTTP server. It accepts a user-provided `request_path` (extracted from the HTTP request URL) and transforms it into a `resolved_path` that points to a real filesystem location. The validation process ensures that the resolved path remains within the `document_root` directory tree, preventing directory traversal attacks that could expose system files or sensitive data.

| Validation Step | Purpose | Example Input | Example Output |
|-----------------|---------|---------------|----------------|
| URL decode | Convert percent-encoded characters | `/docs/file%20name.txt` | `/docs/file name.txt` |
| Path normalization | Remove `.` and `..` components | `/docs/../admin/secret.txt` | `/admin/secret.txt` |
| Root boundary check | Ensure path stays within document root | `/admin/secret.txt` with root `/var/www` | **REJECT** - outside boundary |
| Absolute path construction | Build complete filesystem path | `/index.html` with root `/var/www` | `/var/www/index.html` |

**MIME Type Detection Interface:**

The `detect_mime_type` method provides content type detection based on file extensions. While production web servers often use more sophisticated methods like magic number detection or libmagic integration, the educational implementation focuses on extension-based mapping for simplicity and predictability.

| File Extension | MIME Type | Description |
|----------------|-----------|-------------|
| `.html`, `.htm` | `text/html` | HTML documents |
| `.css` | `text/css` | Cascading Style Sheets |
| `.js` | `application/javascript` | JavaScript files |
| `.json` | `application/json` | JSON data |
| `.txt` | `text/plain` | Plain text files |
| `.png` | `image/png` | PNG images |
| `.jpg`, `.jpeg` | `image/jpeg` | JPEG images |
| `.gif` | `image/gif` | GIF images |
| (unknown) | `application/octet-stream` | Binary data fallback |

### File Serving Algorithm

The file serving algorithm implements a multi-stage process that prioritizes security validation before performing any filesystem operations. This approach ensures that malicious requests are rejected early in the process, minimizing resource consumption and potential attack vectors.

**Stage 1: Request Path Validation**

1. **Extract the path component** from the HTTP request URL, handling query parameters and fragment identifiers by truncating at the first `?` or `#` character
2. **Perform URL decoding** to convert percent-encoded characters back to their original form, handling special characters like spaces (`%20`) and non-ASCII characters
3. **Normalize the path** by resolving `.` (current directory) and `..` (parent directory) components using a stack-based algorithm that processes each path segment
4. **Validate path boundaries** by ensuring the normalized path, when combined with the document root, produces an absolute path that remains within the document root directory tree
5. **Check for null bytes** and other control characters that could indicate path injection attacks or malformed input data
6. **Construct the resolved absolute path** by concatenating the document root with the validated relative path using appropriate path separator characters

**Stage 2: Filesystem Verification**

7. **Check file existence** using the `access()` system call or equivalent to determine if the resolved path corresponds to an actual filesystem entry
8. **Determine entry type** by calling `stat()` to distinguish between regular files, directories, symbolic links, and other filesystem objects
9. **Verify read permissions** to ensure the server process has appropriate access rights to read the requested file or directory contents
10. **Handle symbolic links** by either following them (if policy allows) or treating them as regular files, depending on security configuration

**Stage 3: Content Delivery**

11. **For regular files**: Open the file, determine its size, detect the MIME type based on the file extension, and read the contents into the response body buffer
12. **For directories**: Generate an HTML directory listing showing available files and subdirectories, or serve a default index file (`index.html`) if present
13. **Set response headers** including `Content-Type`, `Content-Length`, and `Last-Modified` based on file metadata and detected content type
14. **Handle large files** by implementing chunked reading to avoid loading entire files into memory, particularly important for binary content like images or videos

> **Design Insight**: The algorithm's security-first approach means that path validation errors result in immediate rejection with a 400 Bad Request response, while filesystem errors (like permission denied or file not found) generate appropriate 4xx error responses. This distinction helps legitimate clients understand whether their request was malformed or simply requested a non-existent resource.

**Error Handling Throughout the Algorithm:**

At each stage, specific error conditions require different responses and cleanup actions. Path validation errors typically indicate malicious requests and should be logged for security monitoring. Filesystem errors might represent temporary conditions (like permission changes) or legitimate requests for non-existent resources.

| Error Condition | Detection Point | HTTP Status | Response Action |
|-----------------|-----------------|-------------|-----------------|
| Invalid URL encoding | Stage 1, Step 2 | 400 Bad Request | Log security warning, minimal error details |
| Directory traversal attempt | Stage 1, Step 4 | 400 Bad Request | Log security alert, block request |
| File not found | Stage 2, Step 7 | 404 Not Found | Standard not found page |
| Permission denied | Stage 2, Step 9 | 403 Forbidden | Access denied message |
| File read error | Stage 3, Step 11 | 500 Internal Server Error | Generic error, log details |

### Architecture Decision Records

> **Decision: Path Security Validation Strategy**
> - **Context**: The file handler must prevent directory traversal attacks while allowing legitimate file access within the document root. Attackers commonly use `../` sequences to access system files outside the intended serving directory.
> - **Options Considered**: 1) Simple string matching to reject `..` patterns, 2) Filesystem-based validation using realpath(), 3) Stack-based path normalization with boundary checking
> - **Decision**: Implement stack-based path normalization combined with absolute path boundary validation
> - **Rationale**: String matching is easily bypassed with encoded characters or alternate representations. Realpath() requires filesystem access and may have race conditions. Stack-based normalization handles all `..` variations while boundary checking prevents any escape from the document root.
> - **Consequences**: Provides robust security against directory traversal attacks but requires more complex implementation than simple string matching. Performance impact is minimal since path validation occurs once per request.

| Option | Security Level | Performance | Implementation Complexity | Chosen? |
|--------|---------------|-------------|---------------------------|---------|
| String matching | Low - easily bypassed | High | Low | No |
| Realpath() validation | Medium - race conditions | Medium | Medium | No |
| Stack-based normalization | High - handles all cases | High | High | **Yes** |

> **Decision: MIME Type Detection Method**
> - **Context**: HTTP responses must include accurate Content-Type headers to enable proper browser rendering and security policies. Different detection methods offer varying accuracy and performance characteristics.
> - **Options Considered**: 1) File extension mapping, 2) Magic number detection from file contents, 3) External library integration (libmagic)
> - **Decision**: Implement file extension mapping with a comprehensive MIME type table
> - **Rationale**: Extension mapping provides predictable, fast results suitable for educational purposes. Magic number detection requires reading file contents and adds complexity. Library integration introduces dependencies and may be overkill for a learning project.
> - **Consequences**: Fast, predictable MIME type detection with minimal resource usage. May incorrectly classify files with wrong extensions, but this is acceptable for educational static file serving.

| Option | Accuracy | Performance | Dependencies | Complexity | Chosen? |
|--------|----------|-------------|--------------|------------|---------|
| Extension mapping | Medium | High | None | Low | **Yes** |
| Magic numbers | High | Medium | None | High | No |
| libmagic integration | Highest | Medium | External library | Medium | No |

> **Decision: Large File Handling Strategy**
> - **Context**: Static file servers must handle files of varying sizes, from small text files to large images or videos. Loading entire files into memory can cause resource exhaustion and poor performance.
> - **Options Considered**: 1) Load complete files into memory, 2) Stream files with fixed-size chunks, 3) Memory-mapped file access
> - **Decision**: Implement streaming with 8KB fixed-size chunks for files larger than 64KB
> - **Rationale**: Complete memory loading causes resource exhaustion with large files. Streaming provides predictable memory usage and good performance. Memory mapping adds complexity and platform-specific code.
> - **Consequences**: Controlled memory usage regardless of file size, but requires more complex response generation logic. Small files still load completely for optimal performance.

| Option | Memory Usage | Performance | Complexity | Scalability | Chosen? |
|--------|--------------|-------------|------------|-------------|---------|
| Complete memory load | Unpredictable | High for small files | Low | Poor | No |
| Fixed-size streaming | Predictable | Good for all sizes | Medium | Excellent | **Yes** |
| Memory mapping | Low | Highest | High | Excellent | No |

> **Decision: Directory Listing Generation**
> - **Context**: When a request maps to a directory rather than a file, the server must decide how to respond. Options include serving default files, generating listings, or returning errors.
> - **Options Considered**: 1) Always return 404 for directory requests, 2) Serve index.html if present otherwise 404, 3) Generate HTML directory listings
> - **Decision**: Serve index.html if present, otherwise generate basic HTML directory listing
> - **Rationale**: This matches conventional web server behavior and provides useful functionality for development and testing. Directory listings help developers verify file placement and organization.
> - **Consequences**: Adds complexity to handle directory traversal and HTML generation, but provides valuable development functionality. Security risk is minimal since listings only show files within the document root.

| Option | Developer Utility | Security Risk | Implementation Complexity | Chosen? |
|--------|------------------|---------------|---------------------------|---------|
| Always 404 | Low | None | Low | No |
| Index.html only | Medium | None | Medium | No |
| Generated listings | High | Low | High | **Yes** |

### Common File Serving Pitfalls

⚠️ **Pitfall: Directory Traversal Vulnerability**

The most critical security vulnerability in static file servers occurs when path validation fails to prevent directory traversal attacks. Attackers use sequences like `../` to escape the document root and access system files like `/etc/passwd` or application configuration files containing sensitive data.

**Why it's dangerous**: A successful directory traversal attack can expose any file readable by the server process, including system configuration, application secrets, or user data stored outside the web root. This represents a complete compromise of server confidentiality.

**Common mistakes**: Implementing path validation with simple string matching (`if (strstr(path, ".."))`) that can be bypassed with URL encoding (`%2e%2e%2f`), Unicode normalization attacks, or alternate path representations. Another mistake is validating the request path but not the resolved absolute path.

**How to fix**: Implement comprehensive path normalization using a stack-based algorithm that processes each path component individually. After normalization, construct the absolute path and verify it begins with the document root path. Always validate the final resolved path, not intermediate representations.

```c
// WRONG - easily bypassed
if (strstr(request_path, "..")) {
    return -1; // reject
}

// CORRECT - normalize then validate final path
char* normalized = normalize_path_components(request_path);
char* absolute = combine_paths(document_root, normalized);
if (!path_starts_with(absolute, document_root)) {
    return -1; // reject - outside boundary
}
```

⚠️ **Pitfall: Binary File Corruption**

Static file servers must handle both text and binary content correctly. A common mistake is treating all files as text and applying transformations like line ending conversion or character encoding changes that corrupt binary data like images, executables, or compressed files.

**Why it's wrong**: Binary file corruption makes images unreadable, breaks downloadable software, and causes unpredictable application behavior. Modern web applications rely heavily on binary assets that must be delivered bit-for-bit identical to their stored versions.

**Detection signs**: Images appear broken in browsers, PDF downloads are corrupted, or binary files have different checksums after serving compared to their original versions stored on disk.

**How to fix**: Always read files in binary mode using appropriate flags (`"rb"` for fopen() or `O_BINARY` for open()). Never apply text transformations to file contents. Set appropriate Content-Type headers to help browsers handle binary content correctly.

⚠️ **Pitfall: Inadequate Error Information Disclosure**

File serving errors require careful balance between helpful debugging information and security. Revealing too much information in error responses helps attackers understand the server's internal structure and identify potential attack vectors.

**Why it's wrong**: Detailed error messages like "Permission denied accessing /var/www/admin/secret.txt" reveal internal filesystem structure and confirm the existence of potentially sensitive directories. This information helps attackers refine their attacks.

**How to fix**: Generate standardized error responses that provide sufficient information for legitimate debugging without revealing internal details. Log detailed error information on the server side for administrator review while sending minimal information to clients.

| Error Condition | Bad Response | Good Response |
|-----------------|--------------|---------------|
| File not found | "File /var/www/docs/secret.txt not found" | "404 Not Found" |
| Permission denied | "Permission denied: /etc/passwd" | "403 Forbidden" |
| Read error | "I/O error reading /home/user/.ssh/id_rsa" | "500 Internal Server Error" |

⚠️ **Pitfall: Inefficient Large File Handling**

Loading entire files into memory before sending responses causes memory exhaustion when serving large files or handling multiple concurrent requests for substantial content. This creates denial of service vulnerabilities and poor performance characteristics.

**Why it's wrong**: A 100MB video file loaded completely into memory for each concurrent request quickly exhausts available RAM. Multiple simultaneous requests for large files can crash the server or trigger out-of-memory kills by the operating system.

**Detection signs**: Memory usage spikes correlating with large file requests, server crashes when serving substantial content, or extremely slow response times for large files due to memory pressure and swapping.

**How to fix**: Implement streaming file delivery using fixed-size buffers (typically 4-8KB). Read chunks from the file and write them directly to the network socket without accumulating the entire file in memory. This provides predictable memory usage regardless of file size.

⚠️ **Pitfall: Race Conditions in File Operations**

File serving involves multiple filesystem operations (stat, open, read) that can create race conditions if files are modified, deleted, or have permissions changed between operations. This can lead to serving partially updated content or crashing due to unexpected file states.

**Why it's wrong**: A file might exist during the stat() check but be deleted before open(), causing the server to crash or return confusing errors. Similarly, files being written by other processes might be served in incomplete states, delivering corrupted content to clients.

**How to fix**: Use atomic file operations where possible and implement proper error handling for all filesystem calls. Consider using file locking for critical operations, though this may impact performance. Design the serving logic to gracefully handle files that become unavailable between checks.

⚠️ **Pitfall: Missing Content-Length Headers**

HTTP responses should include Content-Length headers to enable proper connection handling and download progress indication. Omitting this header forces chunked transfer encoding or connection closing, reducing performance and compatibility.

**Why it's wrong**: Browsers cannot display download progress without Content-Length headers. HTTP/1.1 keep-alive connections may not work properly, forcing new TCP connections for each request and significantly impacting performance.

**How to fix**: Always determine file size using stat() before beginning content delivery and include the Content-Length header in the HTTP response. For chunked file reading, calculate the total size upfront rather than omitting the header.

### Implementation Guidance

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Path manipulation | Manual string operations with bounds checking | Path normalization library (realpath, canonicalize_file_name) |
| MIME detection | Static extension-to-type mapping table | libmagic integration for content-based detection |
| File I/O | Standard C file operations (fopen, fread, fclose) | Memory-mapped files with mmap() for large content |
| Directory listing | Simple opendir/readdir with manual HTML generation | Template engine for customizable directory pages |
| Security validation | Custom path normalization with stack algorithm | chroot() jail for complete filesystem isolation |

#### Recommended File Structure

The file handler component integrates into the HTTP server project as a dedicated module responsible for all filesystem operations and security validation:

```
project-root/
├── src/
│   ├── server.c              ← main server loop and TCP handling
│   ├── http_parser.c         ← HTTP request/response parsing
│   ├── file_handler.c        ← this component - file serving logic
│   ├── file_handler.h        ← public interface definitions
│   └── mime_types.c          ← MIME type mapping table
├── include/
│   ├── http_server.h         ← shared type definitions
│   └── file_handler.h        ← file handler interface
├── config/
│   ├── mime.types            ← MIME type configuration file
│   └── server.conf           ← server configuration
├── www/                      ← default document root
│   ├── index.html           
│   ├── css/
│   └── images/
└── tests/
    ├── test_file_handler.c   ← unit tests for file operations
    └── test_security.c       ← security validation tests
```

This structure separates file handling concerns from network and HTTP parsing logic while providing a clear interface for integration with the overall server architecture.

#### Infrastructure Starter Code

**Complete MIME Type Detection System:**

```c
// mime_types.h
#ifndef MIME_TYPES_H
#define MIME_TYPES_H

typedef struct {
    const char* extension;
    const char* mime_type;
} MimeTypeMapping;

// Initialize MIME type system
int init_mime_types(void);

// Get MIME type by file extension
const char* get_mime_type(const char* file_path);

// Cleanup MIME type resources
void cleanup_mime_types(void);

#endif

// mime_types.c - Complete implementation ready to use
#include "mime_types.h"
#include <string.h>
#include <ctype.h>

static MimeTypeMapping mime_mappings[] = {
    {".html", "text/html"},
    {".htm", "text/html"},
    {".css", "text/css"},
    {".js", "application/javascript"},
    {".json", "application/json"},
    {".txt", "text/plain"},
    {".png", "image/png"},
    {".jpg", "image/jpeg"},
    {".jpeg", "image/jpeg"},
    {".gif", "image/gif"},
    {".svg", "image/svg+xml"},
    {".pdf", "application/pdf"},
    {".zip", "application/zip"},
    {NULL, NULL}  // Sentinel
};

int init_mime_types(void) {
    // MIME system initialized with static table
    return 0;
}

const char* get_mime_type(const char* file_path) {
    if (!file_path) return "application/octet-stream";
    
    // Find last dot in filename
    const char* dot = strrchr(file_path, '.');
    if (!dot) return "application/octet-stream";
    
    // Convert extension to lowercase for comparison
    char ext_lower[32];
    int i = 0;
    while (dot[i] && i < 31) {
        ext_lower[i] = tolower(dot[i]);
        i++;
    }
    ext_lower[i] = '\0';
    
    // Search mapping table
    for (int j = 0; mime_mappings[j].extension; j++) {
        if (strcmp(ext_lower, mime_mappings[j].extension) == 0) {
            return mime_mappings[j].mime_type;
        }
    }
    
    return "application/octet-stream";
}

void cleanup_mime_types(void) {
    // Static table requires no cleanup
}
```

**Complete Path Utility Functions:**

```c
// path_utils.h
#ifndef PATH_UTILS_H
#define PATH_UTILS_H

#include <stddef.h>

// URL decode a path string in-place
int url_decode(char* path);

// Check if path contains directory traversal attempts
int has_directory_traversal(const char* path);

// Normalize path by removing . and .. components
int normalize_path(const char* input, char* output, size_t output_size);

// Check if resolved path is within document root
int is_path_within_root(const char* document_root, const char* resolved_path);

#endif

// path_utils.c - Complete implementation
#include "path_utils.h"
#include <string.h>
#include <ctype.h>
#include <stdlib.h>

int url_decode(char* path) {
    char* read = path;
    char* write = path;
    
    while (*read) {
        if (*read == '%' && read[1] && read[2]) {
            // Convert hex digits to character
            char hex[3] = {read[1], read[2], '\0'};
            char* endptr;
            long value = strtol(hex, &endptr, 16);
            if (*endptr == '\0' && value >= 0 && value <= 255) {
                *write = (char)value;
                read += 3;
            } else {
                *write = *read;
                read++;
            }
        } else if (*read == '+') {
            *write = ' ';
            read++;
        } else {
            *write = *read;
            read++;
        }
        write++;
    }
    *write = '\0';
    return 0;
}

int has_directory_traversal(const char* path) {
    // Look for .. components
    const char* pos = path;
    while ((pos = strstr(pos, "..")) != NULL) {
        // Check if .. is a complete path component
        int is_component = 1;
        if (pos > path && pos[-1] != '/' && pos[-1] != '\\') {
            is_component = 0;
        }
        if (pos[2] != '\0' && pos[2] != '/' && pos[2] != '\\') {
            is_component = 0;
        }
        if (is_component) {
            return 1; // Found directory traversal
        }
        pos += 2;
    }
    return 0;
}

// Additional utility functions with complete implementations...
```

#### Core Logic Skeleton Code

**Main File Handler Interface:**

```c
// file_handler.h - Core interfaces for learner implementation
#include "http_server.h"

// Primary file serving function - learners implement this
int serve_static_file(const HTTPRequest* request, HTTPResponse* response, 
                     const ServerConfig* config) {
    // TODO 1: Extract path from request URL, handling query parameters
    // TODO 2: Validate and resolve path against document root using validate_and_resolve_path()
    // TODO 3: Check if resolved path exists and get file stats using stat()
    // TODO 4: Handle directory requests - look for index.html or generate listing
    // TODO 5: For regular files, detect MIME type and set Content-Type header
    // TODO 6: Read file contents and populate response body
    // TODO 7: Set Content-Length header based on file size
    // TODO 8: Handle all error cases with appropriate HTTP status codes
    // Hint: Use the infrastructure functions for path validation and MIME detection
}

// Path security validation - learners implement core logic
int validate_and_resolve_path(const char* document_root, const char* request_path,
                             char* resolved_path, size_t max_path_len) {
    // TODO 1: Create working copy of request_path for manipulation
    // TODO 2: Perform URL decoding using url_decode() utility function
    // TODO 3: Check for directory traversal patterns using has_directory_traversal()
    // TODO 4: Normalize path components to resolve . and .. using normalize_path()
    // TODO 5: Construct absolute path by combining document_root + normalized_path
    // TODO 6: Verify final path stays within document_root boundaries
    // TODO 7: Check for null bytes or other control characters in path
    // TODO 8: Copy validated path to resolved_path output buffer
    // Hint: Reject immediately if any security check fails
}

// File content reading with streaming support - learners implement
int read_file_content(const char* file_path, HTTPResponse* response) {
    // TODO 1: Open file in binary mode for reading
    // TODO 2: Get file size using fstat() or stat()
    // TODO 3: Allocate response body buffer or set up streaming
    // TODO 4: For small files (< 64KB), read entire content into memory
    // TODO 5: For large files, implement chunked reading with 8KB buffers
    // TODO 6: Set response->body_length to actual bytes read
    // TODO 7: Handle read errors and clean up file descriptor
    // TODO 8: Return appropriate error codes for different failure modes
    // Hint: Always close file descriptor in error cases
}

// Directory listing generation - learners implement
int generate_directory_listing(const char* dir_path, const char* url_path,
                              HTTPResponse* response) {
    // TODO 1: Open directory using opendir()
    // TODO 2: Read directory entries with readdir() in a loop
    // TODO 3: Filter out hidden files (starting with .) if desired
    // TODO 4: Sort entries alphabetically for consistent presentation
    // TODO 5: Generate HTML page with proper DOCTYPE and headers
    // TODO 6: Create table or list showing filenames, sizes, and modification times
    // TODO 7: Make filenames clickable links with proper URL encoding
    // TODO 8: Set Content-Type to text/html and calculate Content-Length
    // Hint: Include parent directory link (..) except for document root
}
```

#### Language-Specific Hints for C Implementation

**File System Operations:**
- Use `stat()` or `fstat()` to get file information including size, type, and permissions
- Open files with `fopen(path, "rb")` to ensure binary mode for all file types
- Check `errno` after failed system calls to determine specific error conditions
- Use `access(path, R_OK)` to verify read permissions before attempting file operations

**Memory Management:**
- Always validate buffer sizes before string operations to prevent buffer overflows
- Use `strncpy()` and `snprintf()` instead of `strcpy()` and `sprintf()` for bounds checking
- Free allocated memory in all code paths, including error handling branches
- Consider using stack-allocated buffers for path manipulation to avoid malloc/free overhead

**Security Considerations:**
- Validate all user input including URL paths, query parameters, and header values
- Use `realpath()` on systems that support it for canonical path resolution
- Be careful with signed/unsigned integer comparisons when checking buffer bounds
- Always null-terminate strings after manipulation to prevent buffer over-reads

**Performance Tips:**
- Cache file statistics for recently accessed files to avoid repeated stat() calls
- Use `sendfile()` on Linux or equivalent zero-copy mechanisms for large file transfers
- Consider `mmap()` for files accessed frequently, but handle mapping failures gracefully
- Implement proper error logging to help diagnose filesystem permission or availability issues

#### Milestone Checkpoint

After implementing the file handler component, verify the functionality with these specific tests:

**Basic File Serving Test:**
```bash
# Start your HTTP server on port 8080
./http_server --port 8080 --docroot ./www

# Test basic file serving
curl -v http://localhost:8080/index.html
# Expected: 200 OK response with HTML content and proper Content-Type header

# Test CSS file serving  
curl -v http://localhost:8080/css/style.css
# Expected: 200 OK with Content-Type: text/css

# Test image serving
curl -v http://localhost:8080/images/logo.png
# Expected: 200 OK with Content-Type: image/png and binary content
```

**Security Validation Test:**
```bash
# Test directory traversal prevention
curl -v "http://localhost:8080/../etc/passwd"
# Expected: 400 Bad Request or 404 Not Found (never file contents)

# Test encoded traversal attempts
curl -v "http://localhost:8080/%2e%2e%2f%2e%2e%2fetc%2fpasswd" 
# Expected: 400 Bad Request (should be caught by URL decoding + validation)

# Test file not found
curl -v http://localhost:8080/nonexistent.html
# Expected: 404 Not Found with standard error page
```

**Directory Handling Test:**
```bash
# Test directory listing
curl -v http://localhost:8080/images/
# Expected: 200 OK with HTML directory listing or index.html if present

# Test root directory
curl -v http://localhost:8080/
# Expected: index.html content or directory listing of document root
```

**Signs of Correct Implementation:**
- All responses include proper `Content-Type` and `Content-Length` headers
- Binary files (images) display correctly in web browsers  
- Directory traversal attempts are rejected with 4xx status codes
- File not found generates 404 responses, not server errors
- Directory requests show listings or serve index files appropriately

**Signs Something is Wrong:**
- Images appear corrupted or broken in browsers → binary file handling issue
- Server crashes on directory traversal attempts → path validation missing
- All files served as `text/plain` → MIME type detection not working
- Memory usage grows continuously → file descriptor or memory leaks
- Slow response times for large files → not implementing streaming properly

![Request Processing Sequence](./diagrams/request-processing-flow.svg)

![File Serving Security Flow](./diagrams/file-serving-flow.svg)


## Concurrency Management Component

> **Milestone(s):** Milestone 4 (Concurrent Connections) - handles multiple simultaneous client connections using different concurrency models to prevent blocking and enable scalable request processing

### Concurrency Mental Model: Restaurant Service Models

Think of an HTTP server handling concurrent connections like a restaurant serving multiple customers simultaneously. Just as restaurants have different service models to handle varying numbers of diners, HTTP servers employ different concurrency strategies to manage multiple client connections efficiently.

The **thread-per-connection model** resembles a high-end restaurant where each table gets a dedicated waiter for the entire meal. When customers arrive, the host (main server thread) seats them and assigns a personal waiter (spawns a new thread) who handles all their needs from start to finish. This provides excellent individual attention - each table gets immediate, personalized service without waiting for other tables to finish. However, this model becomes expensive with many tables since you need to hire more waiters (create more threads), and each waiter consumes resources even when just standing around waiting for the customers to decide what they want (thread overhead during I/O operations).

The **thread pool model** works like a restaurant with a fixed number of waiters who serve multiple tables. When customers arrive, the host seats them and assigns the next available waiter from the existing staff. If all waiters are busy, new customers wait in line until someone becomes free. This prevents the restaurant from hiring unlimited staff during rush hours (prevents thread exhaustion) and maintains consistent service quality. The trade-off is that some customers might wait longer during peak times, but the restaurant operates more efficiently overall.

The **event-driven model** resembles a fast-casual restaurant with a single highly efficient coordinator who manages all orders simultaneously. Instead of assigning dedicated waiters, this coordinator continuously monitors all tables - taking orders when customers are ready, delivering food when the kitchen calls out orders, and processing payments when customers signal they're done. This coordinator never stands idle waiting for one table while others need attention. This model can handle many more customers with fewer staff members, but requires the coordinator to be exceptionally organized and efficient at task-switching.

Each model has distinct trade-offs in terms of resource usage, scalability, complexity, and responsiveness. The choice depends on expected load patterns, available system resources, and performance requirements - just as a restaurant's service model depends on its target clientele, budget constraints, and dining style.

### Concurrency Manager Interface

The concurrency management component provides a unified interface for handling multiple simultaneous client connections regardless of the underlying concurrency model. This abstraction allows the server to switch between different concurrency strategies without changing the core HTTP processing logic.

The primary interface centers around connection dispatching, resource management, and lifecycle control. The concurrency manager accepts new client connections from the TCP server component and routes them to available processing resources according to the chosen concurrency model.

| Method Name | Parameters | Returns | Description |
|-------------|------------|---------|-------------|
| `init_concurrency_manager` | `config: ServerConfig*, model: ConcurrencyModel` | `ConcurrencyManager*` | Initializes concurrency manager with specified model and configuration limits |
| `start_connection_handler` | `manager: ConcurrencyManager*, client_fd: int, client_addr: sockaddr_in` | `int` | Dispatches new client connection to available handler based on concurrency model |
| `wait_for_completion` | `manager: ConcurrencyManager*` | `void` | Blocks until all active connections complete processing (used during shutdown) |
| `get_active_connection_count` | `manager: ConcurrencyManager*` | `int` | Returns number of currently active client connections being processed |
| `shutdown_concurrency_manager` | `manager: ConcurrencyManager*, graceful: int` | `void` | Initiates shutdown sequence, optionally waiting for active connections to complete |
| `cleanup_concurrency_manager` | `manager: ConcurrencyManager*` | `void` | Releases all resources including thread pools, connection tracking structures |

The concurrency manager maintains internal state to track active connections, available resources, and configuration limits. This state enables proper resource management and prevents system overload during high traffic periods.

| Field Name | Type | Description |
|------------|------|-------------|
| `model` | `ConcurrencyModel` | Active concurrency model (thread-per-connection, pool, or event-driven) |
| `max_connections` | `int` | Maximum simultaneous connections allowed before rejecting new clients |
| `max_threads` | `int` | Maximum threads in pool model, ignored for other models |
| `active_connections` | `ConnectionContext*` | Array tracking all currently active client connections |
| `connection_count` | `int` | Current number of active connections being processed |
| `thread_pool` | `pthread_t*` | Thread pool array for pool-based concurrency model |
| `connection_queue` | `ConnectionQueue*` | Queue of pending connections awaiting thread assignment in pool model |
| `shutdown_requested` | `volatile int` | Flag indicating graceful shutdown has been initiated |
| `connection_mutex` | `pthread_mutex_t` | Protects connection tracking data structures from concurrent access |

The interface design separates concurrency concerns from HTTP processing logic. The HTTP parser, file handler, and response generation components remain unchanged regardless of concurrency model. This separation enables testing different concurrency approaches without rewriting core functionality.

Connection lifecycle management follows a consistent pattern across all models. The concurrency manager receives a client file descriptor and address from the TCP server, creates a `ConnectionContext` structure to track the connection state, dispatches the connection to an available processor, and ensures proper cleanup when processing completes.

> **Key Design Insight**: The concurrency manager acts as a resource allocation layer between the TCP server and HTTP processing components. By abstracting concurrency model details, we can optimize for different deployment scenarios (high-throughput vs. low-latency, resource-constrained vs. resource-abundant) without changing the core HTTP server logic.

### Concurrency Model Options

The HTTP server supports three distinct concurrency models, each optimized for different scenarios and resource constraints. Understanding the trade-offs between these models is crucial for selecting the appropriate approach based on expected traffic patterns and system capabilities.

![Concurrency Model Comparison](./diagrams/concurrency-models.svg)

#### Thread-Per-Connection Model

The thread-per-connection model creates a dedicated thread for each incoming client connection. When the TCP server accepts a new connection, the concurrency manager immediately spawns a new thread and passes the client file descriptor to that thread for complete request processing.

This model provides excellent isolation between client connections - a slow or problematic client cannot block processing for other clients since each operates in its own thread. The implementation is straightforward since each thread can use blocking I/O operations without affecting other connections. Error handling is simplified because thread-local failures don't propagate to other client connections.

However, thread-per-connection has significant scalability limitations. Each thread consumes system memory for its stack (typically 8MB on Linux), and thread creation overhead becomes substantial under high connection rates. The operating system's thread scheduler becomes a bottleneck when managing hundreds or thousands of concurrent threads. Many threads spend most of their time blocked on I/O operations, leading to inefficient resource utilization.

The processing algorithm for thread-per-connection follows these steps:

1. Main server thread accepts new client connection from TCP listener
2. Concurrency manager checks current connection count against `max_connections` limit
3. If limit not exceeded, manager calls `pthread_create` to spawn new handler thread
4. New thread receives client file descriptor and address as startup parameters
5. Thread calls `init_connection_context` to establish per-connection state tracking
6. Thread enters complete HTTP request-response processing cycle independently
7. Thread handles request parsing, file serving, and response transmission
8. Upon completion or error, thread calls connection cleanup and terminates
9. Main thread decrements active connection counter when notified of thread completion

This model works best for scenarios with moderate connection concurrency (under 100 simultaneous connections) where simplicity and connection isolation are more important than maximum throughput.

#### Thread Pool Model

The thread pool model pre-creates a fixed number of worker threads that share responsibility for processing client connections. Instead of creating threads dynamically, the concurrency manager maintains a pool of persistent threads that pull work from a shared connection queue.

This approach eliminates thread creation overhead since worker threads are reused across multiple connections. The fixed thread count prevents system resource exhaustion even under heavy load, providing predictable performance characteristics. Thread pool size can be tuned based on available CPU cores and expected workload patterns.

The trade-off is increased complexity in work distribution and resource sharing. Multiple threads must safely access shared data structures, requiring proper synchronization. Connection processing becomes less isolated since threads are reused, potentially allowing state from one connection to affect another if cleanup is incomplete.

The thread pool processing algorithm coordinates work distribution among fixed resources:

1. Server initialization creates `max_threads` worker threads in blocked state
2. Each worker thread enters loop calling `dequeue_connection` on shared work queue
3. Main server thread accepts new client connections as usual
4. Concurrency manager adds new connections to work queue instead of creating threads
5. Next available worker thread dequeues connection and processes complete request-response cycle
6. Worker thread performs connection cleanup and returns to queue waiting state
7. If work queue fills beyond capacity, new connections are rejected with appropriate error
8. During shutdown, manager signals all worker threads to exit after completing current work

Queue management requires careful synchronization to prevent race conditions. The connection queue uses mutex protection and condition variables to coordinate between the main thread adding work and worker threads removing work.

| Queue Operation | Synchronization | Behavior |
|-----------------|-----------------|----------|
| `enqueue_connection` | Acquire queue mutex | Add connection to tail, signal waiting workers |
| `dequeue_connection` | Acquire queue mutex | Remove connection from head, block if empty |
| `queue_full` | Acquire queue mutex | Check if queue size exceeds configured limit |
| `shutdown_queue` | Acquire queue mutex | Mark queue closed, wake all waiting workers |

Thread pool size tuning depends on workload characteristics. CPU-bound workloads benefit from thread count matching CPU core count. I/O-bound workloads (typical for file serving) can benefit from higher thread counts since threads spend time blocked on disk operations. Monitoring connection queue depth and thread utilization helps optimize pool size for specific deployment scenarios.

#### Event-Driven Model

The event-driven model uses a single thread with non-blocking I/O and I/O multiplexing to handle multiple connections simultaneously. Instead of dedicating threads to connections, this model monitors all active file descriptors using `select` or `poll` system calls and processes whichever connections have data ready.

This approach achieves excellent scalability since it eliminates thread overhead entirely. A single event loop can handle thousands of simultaneous connections with minimal memory footprint. Context switching overhead is eliminated since only one thread runs, and the model naturally prioritizes connections with available data over blocked connections.

The complexity trade-off is substantial. All I/O operations must be non-blocking, requiring careful state machine management to handle partial reads and writes. Application logic must be structured to avoid blocking operations, since any blocking call stalls processing for all connections. Error handling becomes more complex since connection failures must be managed within the shared event loop.

The event-driven processing algorithm manages multiple connection states within a single thread:

1. Server initialization sets all client sockets to non-blocking mode using `fcntl`
2. Main event loop calls `select` or `poll` with all active file descriptors
3. System call blocks until at least one file descriptor has data ready
4. Event loop iterates through ready file descriptors to determine required processing
5. For server socket readiness, loop accepts new connection and adds to monitoring set
6. For client socket readiness, loop determines connection state and processes next step
7. Connection processing advances state machine: reading request, parsing, serving file, writing response
8. Partial operations save state and yield control back to event loop for next iteration
9. Completed connections are cleaned up and removed from monitoring set
10. Loop continues until shutdown signal received

Connection state management requires tracking progress through the HTTP request-response cycle:

| Connection State | Next Action | Success Transition | Error Transition |
|------------------|-------------|-------------------|------------------|
| `STATE_READING_REQUEST` | `read_complete_request` non-blocking | `STATE_PARSING_REQUEST` | `STATE_ERROR_CLEANUP` |
| `STATE_PARSING_REQUEST` | `parse_http_request` | `STATE_SERVING_FILE` | `STATE_ERROR_RESPONSE` |
| `STATE_SERVING_FILE` | `serve_static_file` | `STATE_WRITING_RESPONSE` | `STATE_ERROR_RESPONSE` |
| `STATE_WRITING_RESPONSE` | `send` response non-blocking | `STATE_COMPLETE_CLEANUP` | `STATE_ERROR_CLEANUP` |
| `STATE_ERROR_RESPONSE` | Generate and send error | `STATE_COMPLETE_CLEANUP` | `STATE_ERROR_CLEANUP` |

The event-driven model excels for high-concurrency scenarios with many simultaneous connections, particularly when most connections involve small file transfers or when memory usage must be minimized.

### Architecture Decision Records

The concurrency management component involves several critical design decisions that significantly impact the server's scalability, complexity, and resource usage characteristics.

> **Decision: Primary Concurrency Model Selection**
> - **Context**: The server must handle multiple simultaneous client connections efficiently while remaining suitable for educational purposes. Different concurrency models have varying complexity, performance, and resource usage characteristics.
> - **Options Considered**: Thread-per-connection for simplicity, thread pool for balanced scalability, event-driven for maximum performance
> - **Decision**: Implement thread-per-connection as the primary model with thread pool as an advanced option
> - **Rationale**: Thread-per-connection provides the clearest learning progression from single-connection handling to concurrent processing. The blocking I/O model matches earlier milestones and doesn't require restructuring HTTP processing logic. Thread pool can be added incrementally to demonstrate resource management concepts.
> - **Consequences**: Initial implementation will have limited scalability (hundreds vs thousands of connections) but will be easier to debug and understand. Students can observe resource exhaustion directly and understand why more sophisticated models exist.

| Concurrency Model | Learning Value | Implementation Complexity | Scalability | Resource Usage |
|-------------------|----------------|---------------------------|-------------|----------------|
| Thread-per-connection | High - clear progression | Low - familiar blocking I/O | Low - hundreds of connections | High - thread overhead |
| Thread pool | Medium - resource management | Medium - queue synchronization | Medium - thousands of connections | Medium - fixed thread count |
| Event-driven | Low - requires restructuring | High - state machine complexity | High - tens of thousands | Low - single thread |

> **Decision: Connection Limit Enforcement**
> - **Context**: Unbounded connection acceptance can lead to resource exhaustion and system instability. The server needs protection against connection flooding while maintaining acceptable service for legitimate clients.
> - **Options Considered**: No limits (simplest), hard connection limit (reject new connections), connection queuing with timeout
> - **Decision**: Implement hard connection limit with immediate rejection of excess connections
> - **Rationale**: Hard limits provide predictable resource usage and prevent system overload. Immediate rejection gives clear feedback to clients and doesn't consume server resources on queued connections. This approach is easier to implement correctly than timeout-based queuing.
> - **Consequences**: Some legitimate clients may be rejected during traffic spikes, but server stability is guaranteed. Students learn about capacity planning and graceful degradation under load.

> **Decision: Graceful Shutdown Strategy**
> - **Context**: Server shutdown should complete in-flight requests when possible to avoid client errors and data corruption. However, shutdown cannot wait indefinitely for slow or malicious clients.
> - **Options Considered**: Immediate shutdown (close all connections), graceful with timeout (wait limited time), graceful without timeout (wait indefinitely)
> - **Decision**: Graceful shutdown with configurable timeout
> - **Rationale**: Graceful shutdown demonstrates proper resource cleanup and client consideration. Timeout prevents infinite waiting on problematic connections. Configurable timeout allows tuning based on expected request processing time.
> - **Consequences**: Shutdown sequence becomes more complex but more robust. Students learn about lifecycle management and the trade-offs between responsiveness and correctness.

> **Decision: Thread Safety Approach**
> - **Context**: Multiple threads accessing shared server state (configuration, connection counters, logging) require synchronization to prevent data corruption and race conditions.
> - **Options Considered**: No shared state (fully isolated), fine-grained locking (per-data-structure mutexes), coarse-grained locking (single server mutex)
> - **Decision**: Minimal shared state with fine-grained locking for specific data structures
> - **Rationale**: Reducing shared state minimizes synchronization complexity and potential deadlock scenarios. Fine-grained locking allows better concurrency than coarse-grained approaches while remaining understandable for educational purposes.
> - **Consequences**: Each shared data structure requires careful analysis for thread safety requirements. Students learn about concurrent programming principles and common synchronization patterns.

![TCP Connection Lifecycle](./diagrams/connection-lifecycle.svg)

### Common Concurrency Pitfalls

Concurrent programming introduces subtle bugs and resource management issues that are particularly challenging for developers new to multi-threaded systems. Understanding these common pitfalls helps avoid frustrating debugging sessions and system instability.

⚠️ **Pitfall: Thread Resource Leaks**

The most common issue in thread-per-connection implementations is failing to properly clean up thread resources when connections terminate. Each `pthread_create` call allocates system resources that must be explicitly released through `pthread_join` or `pthread_detach`. Failing to do so causes thread handles to accumulate in the kernel, eventually exhausting system thread limits.

The symptom appears as the server initially handling connections correctly, then gradually becoming unable to create new threads. Error messages like "Resource temporarily unavailable" from `pthread_create` indicate thread handle exhaustion. This problem is insidious because it may not manifest during light testing but causes failures after handling hundreds or thousands of connections.

The correct approach is to create threads in detached state using `pthread_detach` immediately after creation, or to use `pthread_attr_setdetachstate` with `PTHREAD_CREATE_DETACHED` when calling `pthread_create`. Detached threads automatically release their resources when they terminate, eliminating the need for explicit joining.

```c
// WRONG: Thread handle leaks
pthread_t thread_id;
pthread_create(&thread_id, NULL, handle_client, client_context);
// Thread handle never cleaned up - resource leak!

// CORRECT: Detached thread cleans up automatically
pthread_t thread_id;
pthread_create(&thread_id, NULL, handle_client, client_context);
pthread_detach(thread_id);  // Thread will clean up when it exits
```

⚠️ **Pitfall: Race Conditions in Connection Counting**

Tracking the number of active connections requires careful synchronization since multiple threads modify the connection counter simultaneously. A common mistake is using non-atomic operations to increment and decrement the counter, leading to race conditions where the count becomes inaccurate.

Inaccurate connection counting can cause the server to accept more connections than intended (potentially causing resource exhaustion) or to reject connections when capacity is actually available (reducing throughput unnecessarily). The race condition typically manifests as connection counts that drift from the true value over time.

Proper connection counting requires mutex protection around all counter modifications or use of atomic operations. The counter must be incremented when connections are accepted and decremented when connections complete, with both operations protected by the same synchronization mechanism.

⚠️ **Pitfall: Deadlock in Shutdown Sequence**

Graceful shutdown sequences can create deadlock scenarios when the main thread waits for worker threads to complete while worker threads wait for shared resources that the main thread holds. This commonly occurs when the main thread holds a mutex while signaling shutdown, and worker threads need that same mutex to complete their cleanup.

The symptom is a server that appears to hang during shutdown - it stops accepting new connections but never fully terminates. Worker threads block waiting for resources, while the main thread blocks waiting for worker threads to finish.

The solution is to carefully order shutdown operations: first stop accepting new connections, then release any shared resources, then signal worker threads to shut down, and finally wait for worker thread completion. Never hold locks while waiting for thread completion.

⚠️ **Pitfall: Buffer Ownership Confusion**

In concurrent servers, multiple threads may reference the same buffer memory for request data, response content, or configuration strings. Confusion about which thread owns (is responsible for freeing) shared buffers leads to either memory leaks or double-free errors.

Memory ownership should be clearly defined and documented. The safest approach is to avoid sharing buffers between threads - each thread should work with its own copy of data. When sharing is necessary for performance reasons, use reference counting or clear ownership transfer protocols.

⚠️ **Pitfall: Non-Reentrant Function Usage**

Many standard library functions are not thread-safe and cannot be called simultaneously from multiple threads. Common examples include `strtok` (use `strtok_r` instead), `gethostbyname` (use `getaddrinfo`), and global variable access in error handling functions like `errno`.

Using non-reentrant functions in multi-threaded code can cause data corruption, incorrect results, or crashes that are difficult to reproduce since they depend on thread scheduling timing. These bugs often appear intermittently and may not manifest during development but cause failures in production.

Always use the reentrant (_r suffix) versions of standard library functions when available, or protect non-reentrant functions with mutexes. Check function documentation for thread safety guarantees before using them in multi-threaded contexts.

⚠️ **Pitfall: Blocking Operations in Event-Driven Model**

When implementing event-driven concurrency, any blocking operation (file I/O, DNS lookups, logging to slow storage) stalls the entire event loop and prevents processing of all other connections. This defeats the primary advantage of the event-driven model and can cause connection timeouts for clients that would otherwise be served successfully.

Symptoms include good performance with few connections that degrades dramatically as connection count increases, or periodic freezes where all connections stop responding simultaneously. These issues often correlate with slow file system operations or network lookups.

Event-driven implementations must use non-blocking I/O for all operations or delegate potentially blocking work to background threads. File operations should use `O_NONBLOCK` flags, and functions like `gethostbyname` should be replaced with asynchronous alternatives or offloaded to worker threads.

### Implementation Guidance

The concurrency management component bridges the gap between single-threaded HTTP processing and production-ready multi-client handling. This implementation guidance provides the necessary infrastructure and patterns to implement robust concurrent connection handling.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Threading | POSIX pthread library with basic create/join | Thread pool with condition variables |
| Synchronization | pthread_mutex for shared data | pthread_rwlock for read-heavy data |
| I/O Multiplexing | select() for basic event handling | epoll (Linux) or kqueue (BSD) for high performance |
| Connection Tracking | Fixed-size arrays with linear search | Dynamic hash table with connection IDs |
| Work Queuing | Simple array with mutex protection | Lock-free queue with atomic operations |

#### Recommended File Structure

The concurrency management component integrates with existing server structure while maintaining clear separation of concerns:

```
http-server/
  src/
    main.c                     ← server startup and shutdown coordination
    tcp_server.c               ← existing TCP connection acceptance
    http_parser.c              ← existing HTTP request parsing  
    file_handler.c             ← existing static file serving
    concurrency_manager.c      ← NEW: connection dispatching and resource management
    concurrency_manager.h      ← NEW: concurrency interface definitions
    connection_context.c       ← NEW: per-connection state management
    connection_context.h       ← NEW: connection tracking structures
    thread_pool.c             ← NEW: worker thread management (optional)
    thread_pool.h             ← NEW: thread pool interface (optional)
  tests/
    test_concurrency.c        ← concurrency-specific test cases
    test_integration.c        ← multi-client integration tests
  config/
    server.conf               ← concurrency tuning parameters
```

This structure keeps concurrency concerns separate from HTTP processing logic while providing clear integration points for connection handling.

#### Infrastructure Starter Code

**Complete Connection Context Management** (connection_context.c):
```c
#include "connection_context.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

void init_connection_context(ConnectionContext* context, int client_fd, sockaddr_in addr) {
    memset(context, 0, sizeof(ConnectionContext));
    context->client_fd = client_fd;
    context->client_addr = addr;
    context->connect_time = time(NULL);
    context->thread_id = pthread_self();
}

void cleanup_connection_context(ConnectionContext* context) {
    if (context->client_fd >= 0) {
        close(context->client_fd);
        context->client_fd = -1;
    }
    // Clear sensitive data
    memset(context, 0, sizeof(ConnectionContext));
}

// Thread-safe connection counter for resource tracking
static int active_connection_count = 0;
static pthread_mutex_t connection_count_mutex = PTHREAD_MUTEX_INITIALIZER;

int increment_connection_count(void) {
    pthread_mutex_lock(&connection_count_mutex);
    int new_count = ++active_connection_count;
    pthread_mutex_unlock(&connection_count_mutex);
    return new_count;
}

int decrement_connection_count(void) {
    pthread_mutex_lock(&connection_count_mutex);
    int new_count = --active_connection_count;
    pthread_mutex_unlock(&connection_count_mutex);
    return new_count;
}

int get_active_connection_count(void) {
    pthread_mutex_lock(&connection_count_mutex);
    int count = active_connection_count;
    pthread_mutex_unlock(&connection_count_mutex);
    return count;
}
```

**Complete Thread-Safe Logging Utility** (logger.c):
```c
#include <stdio.h>
#include <time.h>
#include <pthread.h>
#include <stdarg.h>

static pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;
static FILE* log_file = NULL;

int init_logger(const char* log_filename) {
    pthread_mutex_lock(&log_mutex);
    log_file = fopen(log_filename, "a");
    pthread_mutex_unlock(&log_mutex);
    return (log_file != NULL) ? 0 : -1;
}

void log_message(const char* level, const char* format, ...) {
    pthread_mutex_lock(&log_mutex);
    
    time_t now = time(NULL);
    char time_str[64];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", localtime(&now));
    
    FILE* output = log_file ? log_file : stderr;
    fprintf(output, "[%s] [%s] [Thread %lu] ", time_str, level, pthread_self());
    
    va_list args;
    va_start(args, format);
    vfprintf(output, format, args);
    va_end(args);
    
    fprintf(output, "\n");
    fflush(output);
    
    pthread_mutex_unlock(&log_mutex);
}

void cleanup_logger(void) {
    pthread_mutex_lock(&log_mutex);
    if (log_file) {
        fclose(log_file);
        log_file = NULL;
    }
    pthread_mutex_unlock(&log_mutex);
}
```

#### Core Logic Skeleton Code

**Concurrency Manager Implementation** (concurrency_manager.c):
```c
#include "concurrency_manager.h"
#include "connection_context.h"
#include "http_parser.h"
#include "file_handler.h"
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct {
    ConcurrencyModel model;
    int max_connections;
    int max_threads;
    volatile int shutdown_requested;
    pthread_mutex_t manager_mutex;
} ConcurrencyManager;

// Thread entry point for handling individual client connections
void* handle_client_connection(void* arg) {
    ConnectionContext* context = (ConnectionContext*)arg;
    
    // TODO 1: Initialize HTTP request and response structures using init_http_request/init_http_response
    
    // TODO 2: Read complete HTTP request from client socket using read_complete_request
    // Handle partial reads and network timeouts appropriately
    
    // TODO 3: Parse HTTP request using parse_http_request, handle parsing errors
    
    // TODO 4: Serve static file content using serve_static_file from file_handler component
    
    // TODO 5: Send HTTP response back to client, handle partial writes
    
    // TODO 6: Clean up request/response structures and connection context
    
    // TODO 7: Decrement active connection count and log connection completion
    
    // TODO 8: For detached threads, perform final cleanup and return NULL
    
    return NULL;
}

ConcurrencyManager* init_concurrency_manager(ServerConfig* config, ConcurrencyModel model) {
    // TODO 1: Allocate ConcurrencyManager structure and initialize fields from config
    
    // TODO 2: Initialize mutex for protecting shared manager state
    
    // TODO 3: Set concurrency model and resource limits from ServerConfig
    
    // TODO 4: If model is THREAD_POOL, initialize worker thread pool and work queue
    // Use pthread_create to spawn config->max_threads worker threads
    
    // TODO 5: If model is EVENT_DRIVEN, initialize file descriptor sets and event tracking
    
    // TODO 6: Initialize connection tracking data structures
    
    // TODO 7: Return initialized manager or NULL on failure
}

int start_connection_handler(ConcurrencyManager* manager, int client_fd, sockaddr_in client_addr) {
    // TODO 1: Check if shutdown has been requested, reject new connections if so
    
    // TODO 2: Check current connection count against max_connections limit
    // If limit exceeded, send HTTP 503 Service Unavailable and close client_fd
    
    // TODO 3: Create and initialize ConnectionContext for this client
    // Use init_connection_context with client_fd and client_addr
    
    // TODO 4: Increment active connection count using thread-safe counter
    
    // TODO 5: Dispatch connection based on concurrency model:
    // - THREAD_PER_CONNECTION: Create detached thread with handle_client_connection
    // - THREAD_POOL: Add connection to work queue for worker threads
    // - EVENT_DRIVEN: Add client_fd to monitored file descriptor set
    
    // TODO 6: Log successful connection acceptance with client address
    
    // TODO 7: Return 0 on success, -1 on failure (cleanup connection on failure)
}

void shutdown_concurrency_manager(ConcurrencyManager* manager, int graceful) {
    // TODO 1: Set shutdown_requested flag to prevent new connection acceptance
    
    // TODO 2: If graceful shutdown requested, wait for active connections to complete
    // Use configurable timeout to prevent infinite waiting
    
    // TODO 3: For THREAD_POOL model, signal worker threads to exit and join them
    
    // TODO 4: For EVENT_DRIVEN model, close event loop and cleanup file descriptor sets
    
    // TODO 5: Close any remaining client connections and cleanup resources
    
    // TODO 6: Log shutdown completion with final connection statistics
}
```

#### Language-Specific Hints

**POSIX Thread Management:**
- Use `pthread_create` with `PTHREAD_CREATE_DETACHED` attribute to avoid thread handle leaks
- Always check `pthread_create` return value - thread creation can fail under high load
- Use `pthread_self()` for logging and debugging thread-specific operations
- Mutex initialization with `PTHREAD_MUTEX_INITIALIZER` is simpler than `pthread_mutex_init`

**Socket Configuration for Concurrency:**
- Set `SO_REUSEADDR` on server socket to allow quick restart during development
- Consider `TCP_NODELAY` for reducing latency on small HTTP responses
- Use `fcntl(fd, F_SETFL, O_NONBLOCK)` for event-driven model socket configuration
- Monitor `EMFILE` and `ENFILE` errors which indicate file descriptor exhaustion

**Memory Management in Multithreaded Context:**
- Each thread needs its own request/response buffers to avoid sharing conflicts
- Use `malloc`/`free` carefully - consider memory pools for high-frequency allocation
- Clear sensitive data from buffers before freeing to prevent information leaks
- Consider using `valgrind --tool=helgrind` to detect thread safety issues

#### Milestone Checkpoint

After implementing thread-per-connection concurrency:

**Verification Command:**
```bash
# Terminal 1: Start server
./http_server -p 8080 -r ./www -c 10

# Terminal 2: Test concurrent connections
for i in {1..5}; do
  curl -s http://localhost:8080/test.html &
done
wait
```

**Expected Behavior:**
- Server accepts all 5 simultaneous connections without blocking
- Each connection receives complete HTTP response with correct content
- Server logs show multiple thread IDs handling concurrent requests
- Connection count increases to 5 then decreases back to 0
- No resource leaks or error messages about thread creation failures

**Signs of Problems:**
- Connections hang or timeout: Check for blocking I/O operations
- "Resource temporarily unavailable" errors: Thread cleanup issues or limits exceeded  
- Inconsistent responses: Race conditions in shared data structures
- Server crashes under load: Buffer overflows or memory corruption

**Advanced Testing:**
Use `ab` (Apache Bench) for stress testing: `ab -n 100 -c 10 http://localhost:8080/`
Monitor with `ps -eLf | grep http_server` to observe thread creation patterns


## Interactions and Data Flow

> **Milestone(s):** Integrates all Milestones 1-4 - describes how TCP server, HTTP parser, file handler, and concurrency manager work together to process complete client requests

The interactions and data flow section reveals how our HTTP server's four main components orchestrate together to transform raw TCP connections into complete HTTP responses. Think of this as watching the backstage choreography of a theater production - while audiences see a seamless performance, the magic happens through precise coordination between lighting, sound, props, and actors, each playing their role at exactly the right moment with perfect timing.

Understanding component interactions is crucial because distributed systems fail most often at the boundaries between components, not within individual components themselves. A TCP connection might succeed, HTTP parsing might work perfectly, file serving might execute flawlessly, and concurrency management might handle threading correctly, yet the system could still fail if these components don't communicate properly or if data transformations between them introduce errors.

![HTTP Server System Overview](./diagrams/system-overview.svg)

The interaction patterns we'll explore demonstrate three fundamental distributed systems principles: **separation of concerns** (each component has a single, well-defined responsibility), **loose coupling** (components interact through clean interfaces rather than tight dependencies), and **error isolation** (failures in one component don't cascade uncontrollably to others). These principles become especially important as we scale from handling a single client to managing hundreds of concurrent connections.

### Component Communication Patterns

Think of component communication like a hospital emergency room - when a patient arrives, there's a well-established protocol for how information flows between triage, doctors, nurses, lab technicians, and administrators. Each role has specific responsibilities, clear communication channels, and standardized handoff procedures. No single person needs to understand every other role in detail, but everyone must know exactly how to receive information, what to do with it, and where to send the results.

Our HTTP server follows similar communication patterns, with each component serving as a specialist that receives structured input, performs its specific function, and produces structured output for the next component in the chain. This design enables us to test, debug, and modify components independently while maintaining system-wide coherence.

#### Primary Communication Flow

The primary communication pattern follows a **request-response pipeline** where data flows sequentially through components, with each component adding value and structure to the information stream. Unlike event-driven architectures where components communicate through asynchronous messages, our HTTP server uses **synchronous handoffs** - each component completes its work before passing control to the next component.

| Source Component | Target Component | Data Passed | Communication Method | Error Handling |
|------------------|------------------|-------------|---------------------|----------------|
| TCP Server | Concurrency Manager | `client_fd`, `client_addr` | Function call with `ConnectionContext*` | Returns error code, caller handles cleanup |
| Concurrency Manager | HTTP Parser | `client_fd`, raw request buffer | Thread function parameter | Exception/return code propagates to thread handler |
| HTTP Parser | File Handler | `HTTPRequest*` with parsed fields | Struct pointer parameter | Parser sets error flags, handler checks validity |
| File Handler | HTTP Response | `HTTPResponse*` with content | Struct modification in-place | Handler sets appropriate HTTP status codes |
| HTTP Response | TCP Server | Response bytes, content length | Buffer pointer and size | TCP write errors trigger connection cleanup |

The **ownership transfer model** governs memory management and resource responsibility as data flows between components. When the TCP server accepts a connection, it owns the client file descriptor and must ensure it gets closed. When it hands the connection to the concurrency manager, ownership transfers - the concurrency manager becomes responsible for cleanup. Similarly, when the HTTP parser allocates memory for request body content, it owns that memory until the request processing completes.

#### Secondary Communication Patterns

Beyond the primary request-response pipeline, components use several secondary communication patterns for configuration, monitoring, and resource management:

**Configuration Propagation**: The `ServerConfig` structure flows downward from the main server loop to each component, providing runtime parameters like document root, thread limits, and timeout values. This follows an **immutable configuration** pattern - once the server starts, configuration values don't change, eliminating the need for complex synchronization.

**Resource Status Reporting**: Components report their resource usage back to the concurrency manager through atomic counters and status flags. The concurrency manager tracks active connection counts, thread pool utilization, and memory usage to make scheduling decisions and prevent resource exhaustion.

**Error Signal Propagation**: When components encounter non-recoverable errors, they use a combination of return codes and shared error state to signal problems upstream. This enables graceful degradation - if the file handler can't read a requested file, it generates a 404 response rather than crashing the entire connection.

> **Decision: Synchronous Pipeline vs. Asynchronous Event-Driven Communication**
> - **Context**: Components need to coordinate request processing while maintaining clear boundaries and error handling
> - **Options Considered**: 
>   1. Synchronous function calls with direct data passing
>   2. Asynchronous message queues between components  
>   3. Shared memory with event notifications
> - **Decision**: Synchronous function calls with structured data passing
> - **Rationale**: Synchronous calls provide simpler error handling, easier debugging, and clearer resource ownership semantics for an educational project. Asynchronous approaches add complexity without significant benefits at this scale.
> - **Consequences**: Enables straightforward testing and debugging but limits scalability compared to fully asynchronous designs

| Communication Pattern | Pros | Cons | Used For |
|----------------------|------|------|----------|
| Synchronous Pipeline | Simple error handling, clear ownership, easy debugging | Blocking behavior, limited scalability | Primary request processing |
| Shared Configuration | Consistent settings, no synchronization needed | Requires server restart for changes | Runtime parameters |
| Status Reporting | Real-time resource monitoring, automatic throttling | Additional complexity, potential contention | Resource management |
| Error Propagation | Graceful degradation, localized failure recovery | Complex error code hierarchy | Failure handling |

### Complete Request-Response Cycle

![Request Processing Sequence](./diagrams/request-processing-flow.svg)

The complete request-response cycle represents the journey of a single HTTP request from the moment a TCP connection arrives until the response bytes are sent back to the client and the connection is cleaned up. Think of this like tracking a letter through the postal system - from the moment it's dropped in a mailbox, through sorting facilities, delivery routes, and final delivery, with each step adding information and moving the letter closer to its destination.

Understanding this complete cycle is essential for debugging because most HTTP server issues manifest as problems in the handoffs between stages rather than within individual stages. A request might fail because the TCP server doesn't read enough bytes, because the HTTP parser mishandles headers, because the file handler can't resolve paths, or because the concurrency manager runs out of threads.

#### Stage 1: Connection Establishment and Acceptance

The request-response cycle begins when a client establishes a TCP connection to our server. The TCP server component, running in its main event loop, detects the incoming connection through the `accept()` system call and creates the foundational data structures that will carry information through the entire request processing pipeline.

**Detailed Connection Acceptance Process:**

1. **Socket Event Detection**: The main server loop calls `accept()` on the listening socket `server_fd`, which blocks until a client attempts to connect. When `accept()` returns, it provides a new `client_fd` representing the specific client connection and a `sockaddr_in` structure containing the client's IP address and port.

2. **Connection Context Creation**: The server calls `init_connection_context()` to create a `ConnectionContext` structure that will track this connection's lifecycle. This context includes the client file descriptor, client address information, connection timestamp, and eventually thread identification for debugging purposes.

3. **Resource Availability Check**: Before proceeding with request processing, the server checks whether accepting this connection would exceed configured limits. It queries the concurrency manager's active connection count and compares it against `MAX_CONNECTIONS` from the `ServerConfig`.

4. **Connection Handoff**: If resources are available, the server passes the `ConnectionContext` to the concurrency manager via `start_connection_handler()`. This transfer of ownership is critical - from this point forward, the concurrency manager becomes responsible for the client file descriptor and must ensure it gets closed even if errors occur.

5. **Failure Recovery**: If resource limits are exceeded or if connection handoff fails, the TCP server immediately closes the `client_fd` and logs the failure. This prevents file descriptor leaks and ensures the client receives a connection refused error rather than hanging indefinitely.

**Data Structures at Connection Establishment:**

| Structure | Field | Value | Purpose |
|-----------|-------|-------|---------|
| `ConnectionContext` | `client_fd` | Result of `accept()` call | Socket for client communication |
| `ConnectionContext` | `client_addr` | Client's `sockaddr_in` | Source IP and port for logging |
| `ConnectionContext` | `connect_time` | `time(NULL)` | Connection timestamp for timeout detection |
| `ConnectionContext` | `thread_id` | Initially 0 | Will be set by concurrency manager |

#### Stage 2: Concurrency Dispatching and Resource Allocation

Once the concurrency manager receives the connection context, it must decide how to handle the new connection based on the configured concurrency model. This stage is where our server's scalability characteristics are determined - poor decisions here can lead to resource exhaustion, thread thrashing, or connection starvation.

**Thread-per-Connection Model Processing:**

1. **Thread Creation**: The concurrency manager calls `pthread_create()` to spawn a new thread with `handle_client_connection()` as the entry point and the `ConnectionContext*` as the parameter. The new thread receives a copy of the connection context and begins executing independently.

2. **Resource Tracking**: The manager increments its active connection counter using atomic operations to prevent race conditions. This counter is checked by the TCP server before accepting new connections and is used for resource limit enforcement.

3. **Thread Configuration**: The newly created thread is configured as detached using `pthread_detach()` so that it automatically cleans up its resources when the connection processing completes. This prevents the need for explicit `pthread_join()` calls.

4. **Error Recovery**: If thread creation fails due to system resource limits, the concurrency manager immediately closes the `client_fd`, decrements any counters that were incremented, and returns an error code to the TCP server.

**Thread Pool Model Processing:**

1. **Work Queue Insertion**: Instead of creating a new thread, the concurrency manager calls `enqueue_connection()` to add the connection context to a shared work queue protected by a mutex. Worker threads continuously poll this queue for new work.

2. **Worker Thread Notification**: After enqueuing the connection, the manager signals a condition variable to wake up sleeping worker threads. This ensures that connections don't sit in the queue longer than necessary.

3. **Queue Overflow Handling**: If the work queue is full, indicating that all worker threads are busy and the queue has reached its maximum capacity, the manager can either block waiting for space or immediately reject the connection based on configuration.

**Resource State Tracking:**

| Resource Type | Counter Variable | Limit Check | Cleanup Responsibility |
|---------------|------------------|-------------|------------------------|
| Active Connections | `active_connection_count` | Against `max_connections` | Connection thread |
| Worker Threads | `thread_pool_size` | Against `max_threads` | Concurrency manager |
| Work Queue Entries | `queue->size` | Against `queue->capacity` | Worker thread |
| File Descriptors | System-level tracking | Against `ulimit -n` | Connection thread |

#### Stage 3: HTTP Request Reading and Buffering

Once a worker thread (or dedicated connection thread) begins processing the connection, its first responsibility is reading the complete HTTP request from the client socket. This stage is more complex than it initially appears because TCP provides a byte stream, not message boundaries, and HTTP requests can arrive in multiple TCP segments or be split across several `read()` operations.

**Request Reading Algorithm:**

1. **Buffer Initialization**: The connection handler allocates a request buffer of `MAX_REQUEST_SIZE` bytes (typically 8192) and initializes reading state variables including bytes read so far, buffer remaining, and completion status.

2. **Partial Read Loop**: The handler calls `read_complete_request()` in a loop, which internally uses `recv()` to read bytes from `client_fd`. Each call to `recv()` may return fewer bytes than requested, requiring multiple iterations to read the complete request.

3. **Request Boundary Detection**: As bytes are read, the handler scans for the HTTP request boundary, which is indicated by a blank line (two consecutive CRLF sequences: `\r\n\r\n`). For requests with message bodies, additional reading may be required based on the `Content-Length` header.

4. **Timeout Handling**: Each read operation is performed with a socket timeout to prevent connections from hanging indefinitely. If the timeout expires before a complete request is received, the handler generates a 408 Request Timeout response.

5. **Buffer Overflow Protection**: The handler continuously checks that the total bytes read doesn't exceed `MAX_REQUEST_SIZE`. If the request is too large, it generates a 413 Request Entity Too Large response and closes the connection.

**Request Buffer Management:**

```
Request Buffer Layout:
[HTTP Request Line]\r\n
[Header1: Value1]\r\n  
[Header2: Value2]\r\n
...
[HeaderN: ValueN]\r\n
\r\n
[Message Body (if present)]

Buffer State Variables:
- buffer_start: Points to beginning of allocated buffer
- write_position: Current position for writing new bytes  
- bytes_available: Remaining space in buffer
- request_complete: Boolean indicating full request received
```

The critical challenge in this stage is handling **partial reads** correctly. Network conditions, TCP window sizes, and operating system buffering can cause HTTP requests to arrive in fragments. A robust implementation must accumulate these fragments while detecting request boundaries and preventing buffer overflows.

#### Stage 4: HTTP Request Parsing and Validation

After the complete HTTP request is buffered, the connection handler passes the raw request data to the HTTP parser component. This stage transforms the unstructured byte stream into the structured `HTTPRequest` data structure that subsequent components can process safely.

**Parsing Process Flow:**

1. **Request Structure Initialization**: The handler calls `init_http_request()` to create an `HTTPRequest` structure with safe default values and null pointers. This ensures that cleanup functions work correctly even if parsing fails partway through.

2. **Request Line Parsing**: The parser calls `parse_request_line()` to extract the HTTP method, requested URL path, and protocol version from the first line of the request. This involves finding space characters that separate these three components and validating that each component contains legal characters.

3. **Header Parsing Loop**: The parser enters a loop calling `parse_header_line()` for each subsequent line until it encounters the blank line that separates headers from the message body. Each header line is split on the first colon character, with header names normalized to lowercase and header values trimmed of whitespace.

4. **Message Body Handling**: If the request includes a `Content-Length` header, the parser calls `parse_message_body()` to copy the specified number of bytes into the request's body buffer. This stage must handle cases where the content length is larger than remaining buffer space.

5. **Request Validation**: After parsing completes, the parser performs semantic validation: checking that the HTTP method is supported, that the URL path doesn't contain illegal characters, and that required headers are present.

**Parsing State Validation:**

| Validation Check | Condition | Error Response | Recovery Action |
|------------------|-----------|----------------|-----------------|
| Method Validity | Method in allowed list | 405 Method Not Allowed | Close connection |
| Path Format | No null bytes, valid UTF-8 | 400 Bad Request | Close connection |
| Header Format | Contains colon separator | 400 Bad Request | Skip invalid header |
| Content Length | Numeric, non-negative | 400 Bad Request | Close connection |
| Protocol Version | HTTP/1.0 or HTTP/1.1 | 505 HTTP Version Not Supported | Close connection |

The parsing stage must be extremely robust because it processes untrusted input directly from network clients. Malformed requests, overly long headers, binary content in text fields, and other anomalies are common and must be handled gracefully without crashing the server or enabling security vulnerabilities.

#### Stage 5: File Path Resolution and Security Validation

With a valid `HTTPRequest` structure in hand, the connection handler passes control to the file handler component. The file handler's primary responsibility is translating the HTTP request's URL path into a safe filesystem path within the configured document root, while preventing directory traversal attacks and other path-based security vulnerabilities.

**Path Resolution Algorithm:**

1. **URL Decoding**: The file handler calls `url_decode()` to convert percent-encoded characters in the request path back to their original form. For example, `%20` becomes a space character, and `%2F` becomes a forward slash.

2. **Path Normalization**: The handler calls `normalize_path()` to resolve relative path components like `.` (current directory) and `..` (parent directory). This step is crucial for security because it prevents clients from using sequences like `../../../etc/passwd` to escape the document root.

3. **Security Validation**: The handler calls `validate_and_resolve_path()` to ensure that the normalized path, when combined with the document root, doesn't reference any location outside the configured document root directory.

4. **Filesystem Path Construction**: If validation succeeds, the handler constructs the complete filesystem path by concatenating the document root with the normalized request path.

5. **File Existence and Permission Checks**: The handler uses `stat()` to check whether the resolved path exists and whether the server process has read permissions. It also determines whether the path points to a regular file or a directory.

**Security Validation Details:**

| Security Check | Purpose | Implementation | Failure Response |
|----------------|---------|----------------|------------------|
| Directory Traversal | Prevent access outside document root | Check resolved path starts with document root | 403 Forbidden |
| Symbolic Link Following | Prevent link-based escapes | Use `lstat()` and validate link targets | 403 Forbidden |
| Hidden File Access | Respect Unix hidden file convention | Check for filenames starting with '.' | 404 Not Found |
| Permission Validation | Ensure server can read file | Use `access()` with R_OK flag | 403 Forbidden |
| Path Length Limits | Prevent buffer overflow attacks | Check total path length against limits | 414 URI Too Long |

> The path resolution stage implements the principle of **defense in depth** - multiple independent security checks that each provide protection against different attack vectors. Even if one check fails, the others should still prevent unauthorized access.

#### Stage 6: File Content Reading and Response Generation

Once the file handler has validated the request path and confirmed that the requested resource exists and is accessible, it moves to the content serving phase. This stage reads the file contents, determines appropriate HTTP headers, and constructs the complete HTTP response structure.

**Content Serving Process:**

1. **MIME Type Detection**: The handler calls `detect_mime_type()` to examine the file extension and determine the appropriate `Content-Type` header. This ensures that browsers handle different file types correctly - serving HTML files with `text/html`, images with `image/jpeg`, and so forth.

2. **File Size Calculation**: The handler uses the `stat()` result from the previous stage to determine the file size, which becomes the `Content-Length` header value. This allows clients to display download progress and detect truncated responses.

3. **Response Structure Initialization**: The handler calls `init_http_response()` to create an `HTTPResponse` structure with appropriate default values, then sets the status code to 200 OK for successful requests.

4. **Response Header Population**: The handler calls `add_response_header()` multiple times to populate standard HTTP headers including `Content-Type`, `Content-Length`, `Last-Modified`, and `Server`. Each header addition is bounds-checked to prevent buffer overflows.

5. **File Content Reading**: The handler calls `serve_file_content()` to read the file contents into the response body buffer. For large files, this may involve chunked reading to avoid memory exhaustion.

**Response Generation for Different Resource Types:**

| Resource Type | Detection Method | Response Generation | Special Handling |
|---------------|------------------|---------------------|------------------|
| Regular File | `S_ISREG()` macro on stat result | Read file contents into response body | Binary file detection, streaming for large files |
| Directory | `S_ISDIR()` macro on stat result | Generate HTML directory listing | Sort entries, format sizes, security filtering |
| Missing Resource | `stat()` returns -1 with `ENOENT` | Generate 404 Not Found response | Include helpful error message |
| Permission Denied | `stat()` returns -1 with `EACCES` | Generate 403 Forbidden response | Log access attempt for security monitoring |

**Directory Listing Generation**: When the requested path maps to a directory rather than a file, the handler generates an HTML directory listing by calling `generate_directory_listing()`. This function reads directory entries, sorts them alphabetically, and formats them as clickable links within a basic HTML page structure.

#### Stage 7: HTTP Response Transmission and Connection Cleanup

The final stage of the request-response cycle involves serializing the `HTTPResponse` structure into the HTTP wire format and transmitting it back to the client through the TCP connection. This stage must handle partial writes, connection errors, and proper resource cleanup regardless of whether transmission succeeds or fails.

**Response Transmission Process:**

1. **Response Serialization**: The connection handler constructs the complete HTTP response by formatting the status line, serializing all response headers, and appending the message body. The result is a single buffer containing the complete response in HTTP wire format.

2. **Partial Write Loop**: The handler calls `send()` in a loop to transmit the response buffer to the client. Like reading, writing to TCP sockets can result in partial writes where only some bytes are sent in each operation.

3. **Write Timeout Handling**: Each write operation is performed with a timeout to prevent connections from hanging if the client stops reading data. If a write timeout occurs, the handler logs the event and proceeds to connection cleanup.

4. **Connection State Management**: After successful response transmission, the handler determines whether to keep the connection open for additional requests (HTTP keep-alive) or close it immediately. For this educational server, we typically close connections after each response.

5. **Resource Cleanup**: The handler calls cleanup functions for all allocated resources: `cleanup_http_request()` for the parsed request, `cleanup_http_response()` for the response structure, `close()` for the client file descriptor, and `cleanup_connection_context()` for connection tracking.

**Cleanup Sequence and Error Recovery:**

| Cleanup Stage | Resources Released | Error Handling | Consequences of Failure |
|---------------|-------------------|----------------|-------------------------|
| HTTP Request Cleanup | Request body buffer, header storage | Always performed, errors logged | Memory leak for single connection |
| HTTP Response Cleanup | Response body buffer, header storage | Always performed, errors logged | Memory leak for single connection |
| Socket Closure | Client file descriptor | Check return value, log errors | File descriptor leak |
| Connection Context Cleanup | Context structure, thread resources | Always performed | Memory leak, thread resource leak |
| Concurrency Manager Notification | Active connection count decrement | Always performed | Resource limit tracking error |

> **Critical Insight**: The cleanup sequence must be performed in **reverse dependency order** - resources that depend on others must be cleaned up first. For example, the response body buffer must be freed before the response structure itself, and the socket must be closed before the connection context is cleaned up.

The connection handler implements **exception safety** by using a cleanup pattern that works correctly even if errors occur during response transmission. All resource cleanup operations are designed to be **idempotent** - calling them multiple times or with already-cleaned resources produces safe, predictable results.

### Error Propagation Between Components

Error propagation in distributed systems is like a hospital's emergency response system - when something goes wrong, information about the problem must flow quickly and accurately to the right people who can take appropriate action. A misdiagnosed symptom or a communication failure between departments can turn a manageable situation into a crisis.

![Error Handling and Recovery](./diagrams/error-handling-flow.svg)

In our HTTP server, errors can originate from multiple sources: network problems (connection timeouts, socket errors), parsing failures (malformed HTTP requests, invalid headers), filesystem issues (missing files, permission denials), and concurrency problems (thread creation failures, resource exhaustion). Each type of error requires different handling strategies and generates different types of HTTP responses to clients.

The error propagation design follows the principle of **failing fast and failing safely** - when components detect problems they can't handle locally, they immediately signal the error to calling components rather than attempting incomplete or potentially dangerous recovery. This prevents error conditions from cascading into more serious failures or security vulnerabilities.

#### Error Classification and Response Mapping

Our HTTP server categorizes errors into distinct classes based on their source and severity, with each class mapped to appropriate HTTP status codes and recovery strategies. This classification enables consistent error handling across components and ensures that clients receive meaningful error responses rather than generic failure messages.

**System-Level Error Categories:**

| Error Category | Source Components | HTTP Status Mapping | Recovery Strategy | Example Scenarios |
|----------------|-------------------|---------------------|-------------------|------------------|
| Network Errors | TCP Server, Connection Handler | 500 Internal Server Error | Close connection, log error | Socket creation failure, connection reset |
| Protocol Errors | HTTP Parser | 400 Bad Request, 405 Method Not Allowed | Send error response, close connection | Malformed request line, unsupported method |
| Resource Errors | File Handler | 404 Not Found, 403 Forbidden | Send error response, continue connection | Missing file, permission denied |
| Capacity Errors | Concurrency Manager | 503 Service Unavailable | Reject connection or queue request | Thread pool exhausted, connection limit reached |
| Security Errors | File Handler, HTTP Parser | 403 Forbidden, 400 Bad Request | Send error response, log security event | Directory traversal attempt, oversized request |

**Error Severity Levels:**

The server distinguishes between different severity levels to determine appropriate logging, alerting, and recovery actions:

- **Fatal Errors**: System-level failures that prevent the server from accepting new connections (port binding failures, memory exhaustion)
- **Connection Errors**: Problems that affect individual connections but don't impact overall server operation (client disconnections, parsing failures)  
- **Request Errors**: Issues with specific requests that can be handled with appropriate HTTP error responses (file not found, method not allowed)
- **Warning Conditions**: Unusual but manageable situations that should be logged for monitoring (approaching resource limits, slow client connections)

#### Component Error Interfaces

Each component in our HTTP server implements a consistent error reporting interface that enables upstream components to understand what went wrong and how to respond appropriately. This interface combines traditional C error codes with structured error information that can be used to generate detailed HTTP error responses.

**TCP Server Error Interface:**

| Function | Success Return | Error Return | Error Information | Cleanup Required |
|----------|---------------|---------------|-------------------|------------------|
| `create_server_socket()` | Valid file descriptor (>= 0) | -1 | `errno` set to specific error | None |
| `server_main_loop()` | Does not return | void | Logs fatal errors before exit | Close server socket |
| `read_complete_request()` | Bytes read (> 0) | -1 for errors, 0 for EOF | `errno` or connection state | Close client connection |

**HTTP Parser Error Interface:**

| Function | Success Return | Error Return | Error Information | Cleanup Required |
|----------|---------------|---------------|-------------------|------------------|
| `parse_http_request()` | 0 | Error code constant | Sets error fields in request structure | Call `cleanup_http_request()` |
| `parse_request_line()` | 0 | `PARSE_ERROR_INVALID_METHOD`, `PARSE_ERROR_INVALID_PATH` | Error details in parser state | None |
| `parse_header_line()` | 0 | `PARSE_ERROR_MALFORMED_HEADER` | Line number and content in error state | None |
| `parse_message_body()` | 0 | `PARSE_ERROR_BODY_TOO_LARGE` | Content length and buffer size in error state | Free partial body allocation |

**File Handler Error Interface:**

| Function | Success Return | Error Return | Error Information | Cleanup Required |
|----------|---------------|---------------|-------------------|------------------|
| `validate_and_resolve_path()` | 0 | `PATH_ERROR_TRAVERSAL`, `PATH_ERROR_TOO_LONG` | Path details in error structure | None |
| `serve_file_content()` | 0 | `FILE_ERROR_NOT_FOUND`, `FILE_ERROR_PERMISSION` | File path and system error in response | Close file handle if opened |
| `detect_mime_type()` | 0 | Non-zero for unknown types | Uses default MIME type | None |

> **Decision: Structured Error Codes vs. Exception-Style Error Handling**
> - **Context**: Components need to communicate detailed error information while maintaining C compatibility and performance
> - **Options Considered**:
>   1. Simple integer return codes with global error state
>   2. Structured error objects with detailed context
>   3. Exception-style error handling (not available in C)
> - **Decision**: Structured error codes with component-specific error information
> - **Rationale**: Provides detailed error context for generating appropriate HTTP responses while maintaining C compatibility and avoiding global state that complicates threading
> - **Consequences**: Requires more complex error handling code but enables better error responses and debugging

#### Error Response Generation Pipeline

When any component detects an error condition, it must be transformed into an appropriate HTTP response that provides useful information to the client while avoiding information disclosure that could aid attackers. The error response generation pipeline ensures consistent error formatting and appropriate security filtering.

**Error-to-Response Transformation Process:**

1. **Error Context Collection**: The component detecting the error collects relevant context information including error type, severity, affected resource, and any client-safe details that might help with debugging.

2. **Security Filtering**: The error pipeline calls security filtering functions to remove sensitive information like internal file paths, system error messages that might reveal server configuration, and other implementation details.

3. **HTTP Status Code Selection**: Based on the error category and specific error type, the pipeline selects the most appropriate HTTP status code from the standard set (400 Bad Request, 403 Forbidden, 404 Not Found, 500 Internal Server Error, etc.).

4. **Error Response Construction**: The pipeline calls `generate_error_response()` to create a complete `HTTPResponse` structure with the appropriate status code, standard error headers, and a formatted error message body.

5. **Error Logging**: Before returning the error response to the client, the pipeline logs the complete error details (including sensitive information) to the server's error log for debugging and security monitoring.

**Standard Error Response Format:**

```
HTTP/1.1 [Status Code] [Status Text]
Content-Type: text/html
Content-Length: [Body Length]
Connection: close

<!DOCTYPE html>
<html>
<head><title>[Status Code] [Status Text]</title></head>
<body>
<h1>[Status Code] [Status Text]</h1>
<p>[Client-safe error description]</p>
<hr>
<p>[Server identification]</p>
</body>
</html>
```

**Error Response Examples by Category:**

| Error Source | Status Code | Client Message | Internal Log Message |
|--------------|-------------|----------------|---------------------|
| Path traversal attempt | 403 Forbidden | "Access to the requested resource is forbidden" | "Directory traversal attempt: /app/docs/../../../etc/passwd from 192.168.1.100" |
| File not found | 404 Not Found | "The requested resource could not be found" | "File not found: /var/www/html/missing.html (resolved from /missing.html)" |
| Request too large | 413 Request Entity Too Large | "Request size exceeds server limits" | "Request size 16384 exceeds MAX_REQUEST_SIZE 8192 from 192.168.1.100" |
| Resource exhaustion | 503 Service Unavailable | "Server temporarily unable to handle request" | "Thread pool exhausted: 10/10 threads active, connection queue full" |

#### Error Recovery and Connection State Management

Error recovery in our HTTP server follows different strategies depending on the error severity and the stage of request processing where the error occurred. The goal is to maintain system stability while providing useful feedback to clients and maintaining clear resource cleanup responsibilities.

**Connection State During Error Recovery:**

| Error Stage | Connection State | Recovery Action | Resource Cleanup Required |
|-------------|------------------|-----------------|---------------------------|
| Pre-request (TCP connection) | Connection established but no request received | Send 408 Request Timeout, close connection | Close client socket, free connection context |
| Request parsing | Partial request received, parsing failed | Send 400 Bad Request, close connection | Free request buffers, close socket |
| File handling | Valid request, file operation failed | Send appropriate 4xx/5xx error, keep connection open | Free response buffers, but keep connection |
| Response transmission | Response generated, send failed | Log error, close connection | Free all response resources, close socket |

**Graceful Degradation Strategies:**

When the server encounters resource exhaustion or other capacity-related errors, it implements graceful degradation rather than hard failures:

1. **Connection Queuing**: When the thread pool is full, new connections can be queued for a limited time rather than immediately rejected.

2. **Reduced Functionality**: Under high load, the server might disable non-essential features like directory listings or detailed error messages to conserve resources.

3. **Request Prioritization**: Static file requests for small resources might be prioritized over large file downloads during capacity constraints.

4. **Circuit Breaker Pattern**: If filesystem errors reach a threshold, the server might temporarily return 503 Service Unavailable for all requests to prevent cascading failures.

**Error Propagation Chain Example:**

Consider a request for `/documents/../../../etc/passwd` and how the error propagates through the system:

1. **HTTP Parser**: Successfully parses the request, creates valid `HTTPRequest` structure with path `/documents/../../../etc/passwd`
2. **File Handler**: Calls `normalize_path()` which resolves to `/etc/passwd`, then `validate_and_resolve_path()` detects this is outside document root
3. **Security Error Generation**: File handler creates security error with type `PATH_ERROR_TRAVERSAL` and context information
4. **Error Response Pipeline**: Maps security error to 403 Forbidden status, generates safe client message
5. **Security Logging**: Logs full attack details including client IP, requested path, and resolved path to security log
6. **Connection Continuation**: Returns error response to client but keeps connection open for additional requests

This error propagation chain demonstrates how security errors are handled with appropriate response codes while maintaining detailed internal logging for security monitoring and attack detection.

### Implementation Guidance

The interactions and data flow implementation requires careful orchestration of multiple components working together. This section provides the infrastructure and patterns needed to coordinate TCP server, HTTP parser, file handler, and concurrency manager components effectively.

#### Technology Recommendations

| Component Integration | Simple Option | Advanced Option |
|----------------------|---------------|-----------------|
| Inter-component Communication | Direct function calls with error codes | Message passing with error queues |
| Error Handling | Return codes with errno-style error state | Structured error objects with context |
| Resource Management | Manual cleanup with goto error handling | RAII-style cleanup with cleanup functions |
| Data Flow Monitoring | Simple printf debugging | Structured logging with request tracing |

#### Recommended File Structure

```
http-server/
  src/
    main.c                      ← Main server loop and component coordination
    connection_handler.c         ← Complete request-response cycle implementation
    error_handling.c            ← Error response generation and propagation
    
  include/
    server_types.h              ← All data structure definitions
    component_interfaces.h      ← Function signatures for component communication
    error_codes.h               ← Error type definitions and response mappings
    
  components/
    tcp_server/                 ← TCP server component (from previous sections)
    http_parser/                ← HTTP parser component  
    file_handler/               ← File handler component
    concurrency_manager/        ← Concurrency management component
    
  tests/
    integration_tests.c         ← Full request-response cycle tests
    error_handling_tests.c      ← Error propagation and recovery tests
```

#### Infrastructure Starter Code

**Complete Error Response Generation System** (ready to use):

```c
// error_handling.c - Complete error response infrastructure
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "server_types.h"
#include "error_codes.h"

// Error code to HTTP status mapping
static const struct {
    int error_code;
    int http_status;
    const char* status_text;
    const char* client_message;
} error_mappings[] = {
    {PATH_ERROR_TRAVERSAL, 403, "Forbidden", "Access to the requested resource is forbidden"},
    {PATH_ERROR_TOO_LONG, 414, "URI Too Long", "The requested URI is too long"},
    {FILE_ERROR_NOT_FOUND, 404, "Not Found", "The requested resource could not be found"},
    {FILE_ERROR_PERMISSION, 403, "Forbidden", "Access to the requested resource is forbidden"},
    {PARSE_ERROR_INVALID_METHOD, 405, "Method Not Allowed", "The request method is not supported"},
    {PARSE_ERROR_INVALID_PATH, 400, "Bad Request", "The request path contains invalid characters"},
    {PARSE_ERROR_MALFORMED_HEADER, 400, "Bad Request", "The request contains malformed headers"},
    {PARSE_ERROR_BODY_TOO_LARGE, 413, "Request Entity Too Large", "Request size exceeds server limits"},
    {CAPACITY_ERROR_THREAD_POOL, 503, "Service Unavailable", "Server temporarily unable to handle request"},
    {CAPACITY_ERROR_CONNECTION_LIMIT, 503, "Service Unavailable", "Server connection limit reached"},
    {0, 500, "Internal Server Error", "An internal server error occurred"} // Default
};

void generate_error_response(int error_code, const char* error_context, HTTPResponse* response) {
    // Find appropriate error mapping
    const struct error_mapping* mapping = &error_mappings[0]; // Default to last entry
    for (int i = 0; error_mappings[i].error_code != 0; i++) {
        if (error_mappings[i].error_code == error_code) {
            mapping = &error_mappings[i];
            break;
        }
    }
    
    // Initialize response structure
    init_http_response(response);
    response->status_code = mapping->http_status;
    strncpy(response->status_text, mapping->status_text, sizeof(response->status_text) - 1);
    
    // Add standard headers
    add_response_header(response, "Content-Type", "text/html");
    add_response_header(response, "Connection", "close");
    
    // Generate HTML error page
    char error_body[1024];
    snprintf(error_body, sizeof(error_body),
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head><title>%d %s</title></head>\n"
        "<body>\n"
        "<h1>%d %s</h1>\n"
        "<p>%s</p>\n"
        "<hr>\n"
        "<p>HTTP Server/1.0</p>\n"
        "</body>\n"
        "</html>\n",
        mapping->http_status, mapping->status_text,
        mapping->http_status, mapping->status_text,
        mapping->client_message);
    
    // Set response body
    response->body_length = strlen(error_body);
    response->body = malloc(response->body_length + 1);
    if (response->body) {
        strcpy(response->body, error_body);
        
        // Add Content-Length header
        char content_length[32];
        snprintf(content_length, sizeof(content_length), "%zu", response->body_length);
        add_response_header(response, "Content-Length", content_length);
    }
}

// Security event logging
void log_security_event(const char* event_type, const char* client_ip, 
                       const char* request_path, const char* details) {
    time_t now = time(NULL);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
    
    fprintf(stderr, "[%s] SECURITY %s: client=%s path=%s details=%s\n",
            timestamp, event_type, client_ip, request_path, details);
}

// Request context for error tracking
typedef struct {
    ConnectionContext* connection;
    HTTPRequest* request;
    char client_ip[INET_ADDRSTRLEN];
    time_t start_time;
} RequestContext;

void init_request_context(RequestContext* ctx, ConnectionContext* conn, HTTPRequest* req) {
    ctx->connection = conn;
    ctx->request = req;
    ctx->start_time = time(NULL);
    
    // Convert client address to string for logging
    inet_ntop(AF_INET, &conn->client_addr.sin_addr, ctx->client_ip, sizeof(ctx->client_ip));
}
```

**Complete Connection Lifecycle Management** (ready to use):

```c
// connection_handler.c - Request-response cycle coordination
#include "server_types.h"
#include "component_interfaces.h"
#include "error_codes.h"

// Resource cleanup with error safety
typedef struct {
    HTTPRequest* request;
    HTTPResponse* response; 
    ConnectionContext* connection;
    int client_fd;
    int cleanup_flags;
} ResourceCleanupContext;

#define CLEANUP_REQUEST  (1 << 0)
#define CLEANUP_RESPONSE (1 << 1)
#define CLEANUP_CONNECTION (1 << 2)
#define CLEANUP_SOCKET (1 << 3)

void cleanup_request_resources(ResourceCleanupContext* cleanup) {
    if (cleanup->cleanup_flags & CLEANUP_REQUEST && cleanup->request) {
        cleanup_http_request(cleanup->request);
        free(cleanup->request);
        cleanup->request = NULL;
    }
    
    if (cleanup->cleanup_flags & CLEANUP_RESPONSE && cleanup->response) {
        cleanup_http_response(cleanup->response);
        free(cleanup->response);
        cleanup->response = NULL;
    }
    
    if (cleanup->cleanup_flags & CLEANUP_SOCKET && cleanup->client_fd >= 0) {
        close(cleanup->client_fd);
        cleanup->client_fd = -1;
    }
    
    if (cleanup->cleanup_flags & CLEANUP_CONNECTION && cleanup->connection) {
        cleanup_connection_context(cleanup->connection);
        free(cleanup->connection);
        cleanup->connection = NULL;
    }
}

// Connection state machine for error recovery
typedef enum {
    CONNECTION_READING_REQUEST,
    CONNECTION_PARSING_REQUEST,
    CONNECTION_HANDLING_REQUEST,
    CONNECTION_SENDING_RESPONSE,
    CONNECTION_CLEANUP,
    CONNECTION_ERROR
} ConnectionState;

const char* connection_state_names[] = {
    "READING_REQUEST", "PARSING_REQUEST", "HANDLING_REQUEST",
    "SENDING_RESPONSE", "CLEANUP", "ERROR"
};
```

#### Core Logic Skeleton Code

**Main Request-Response Orchestration** (implement the TODOs):

```c
// Main connection handling thread function
void* handle_client_connection(void* arg) {
    ConnectionContext* conn_ctx = (ConnectionContext*)arg;
    ResourceCleanupContext cleanup = {0};
    RequestContext req_ctx = {0};
    ConnectionState state = CONNECTION_READING_REQUEST;
    int error_code = 0;
    
    // Initialize cleanup context
    cleanup.connection = conn_ctx;
    cleanup.client_fd = conn_ctx->client_fd;
    cleanup.cleanup_flags = CLEANUP_CONNECTION | CLEANUP_SOCKET;
    
    // TODO 1: Set socket timeout options for client_fd
    // Use setsockopt() with SO_RCVTIMEO and SO_SNDTIMEO
    // Set both receive and send timeouts to request_timeout_sec from ServerConfig
    
    // TODO 2: Allocate HTTPRequest and HTTPResponse structures
    // Use malloc() and set cleanup flags appropriately
    // Call init_http_request() and init_http_response() for safe defaults
    
    // TODO 3: Initialize request context for error tracking
    // Call init_request_context() with connection and request pointers
    // This enables detailed logging if errors occur
    
    // Main request processing state machine
    while (state != CONNECTION_CLEANUP && state != CONNECTION_ERROR) {
        switch (state) {
            case CONNECTION_READING_REQUEST: {
                // TODO 4: Read complete HTTP request from socket
                // Call read_complete_request() with appropriate buffer and size limits
                // Handle partial reads by calling in loop until complete request received
                // Set error_code and state = CONNECTION_ERROR if reading fails
                
                if (/* request reading succeeded */) {
                    state = CONNECTION_PARSING_REQUEST;
                } else {
                    // TODO 5: Generate 408 Request Timeout or 400 Bad Request response
                    // Use generate_error_response() with appropriate error code
                    // Set state = CONNECTION_SENDING_RESPONSE to send error back to client
                }
                break;
            }
            
            case CONNECTION_PARSING_REQUEST: {
                // TODO 6: Parse raw request data into HTTPRequest structure
                // Call parse_http_request() with raw buffer and HTTPRequest pointer
                // Handle parsing errors by checking return code and setting error_code
                
                if (/* parsing succeeded */) {
                    state = CONNECTION_HANDLING_REQUEST;
                } else {
                    // TODO 7: Generate appropriate parsing error response (400, 405, etc.)
                    // Use error_code returned from parser to select correct HTTP status
                    // Log security events for suspicious requests (oversized, malformed)
                }
                break;
            }
            
            case CONNECTION_HANDLING_REQUEST: {
                // TODO 8: Process request through file handler
                // Call serve_static_file() with request, response, and server config
                // Handle file serving errors (404, 403) by checking return codes
                
                if (/* file serving succeeded */) {
                    state = CONNECTION_SENDING_RESPONSE;
                } else {
                    // TODO 9: File handler should have populated error response
                    // Verify response structure contains appropriate error status code
                    // Log file access attempts for security monitoring
                }
                break;
            }
            
            case CONNECTION_SENDING_RESPONSE: {
                // TODO 10: Send complete HTTP response back to client
                // Serialize response structure into wire format (status line + headers + body)
                // Handle partial writes by calling send() in loop until all bytes transmitted
                // Set appropriate error codes if transmission fails
                
                state = CONNECTION_CLEANUP;
                break;
            }
        }
    }
    
    // TODO 11: Perform cleanup based on cleanup context flags
    // Call cleanup_request_resources() to release all allocated resources
    // Ensure cleanup happens even if errors occurred during processing
    // Log connection completion with timing and status information
    
    // TODO 12: Notify concurrency manager of connection completion
    // Call decrement_connection_count() to update resource tracking
    // Enable server to accept new connections if limits were reached
    
    return NULL;
}
```

**Error Propagation Integration Points** (implement the TODOs):

```c
// Component error handling integration
int process_request_with_error_handling(HTTPRequest* request, HTTPResponse* response, 
                                       ServerConfig* config, RequestContext* req_ctx) {
    int result = 0;
    
    // TODO 1: Attempt file serving with comprehensive error capture
    // Call serve_static_file() and capture both return code and any error context
    // Check for security violations (directory traversal, permission issues)
    
    if (/* serve_static_file succeeded */) {
        return 0; // Success
    }
    
    // TODO 2: Classify error type and severity
    // Map component-specific error codes to error categories
    // Determine if error requires security logging or just standard error response
    
    switch (/* error classification */) {
        case SECURITY_ERROR:
            // TODO 3: Log security event with full context
            // Include client IP, requested path, attack type, timestamp
            // Use log_security_event() with appropriate event classification
            break;
            
        case RESOURCE_ERROR:
            // TODO 4: Log resource access issue
            // Include file path (sanitized), permission details, filesystem error
            // Check if this indicates a configuration problem
            break;
            
        case CAPACITY_ERROR:
            // TODO 5: Update server health metrics
            // Track resource exhaustion events for monitoring
            // Consider implementing backpressure or rate limiting
            break;
    }
    
    // TODO 6: Generate appropriate error response for client
    // Use generate_error_response() with error code and safe context information
    // Ensure response includes appropriate HTTP status and client-safe message
    
    return result;
}
```

#### Milestone Checkpoints

**After implementing component integration:**
- **Command**: `./http-server 8080 /var/www/html`
- **Test**: `curl -v http://localhost:8080/index.html`
- **Expected**: Complete HTTP response with headers, body content, proper connection handling
- **Debug**: Check that all cleanup functions are called, no file descriptors leaked

**After implementing error handling:**
- **Command**: `curl -v http://localhost:8080/../../../etc/passwd`
- **Expected**: 403 Forbidden response, security event logged, connection remains open
- **Debug**: Verify security logging captures attack attempts with client IP and details

**After implementing complete request-response cycle:**
- **Concurrent Test**: Run multiple `curl` commands simultaneously
- **Expected**: All requests complete successfully, proper resource cleanup
- **Debug**: Monitor with `lsof` to verify no file descriptor leaks under load


## Error Handling and Edge Cases

> **Milestone(s):** All Milestones 1-4 - provides comprehensive error handling strategy that spans TCP connections, HTTP parsing, file system operations, and concurrent request management

Robust error handling forms the backbone of any production-quality HTTP server. Think of error handling as the hospital's emergency response system - when things go wrong, there must be clear protocols for diagnosis, treatment, and recovery that prevent cascading failures from bringing down the entire system. Just as a hospital has different response procedures for minor injuries versus life-threatening emergencies, our HTTP server must classify errors by severity and respond appropriately.

The challenge in HTTP server error handling lies in the distributed nature of failures. Errors can originate from network connectivity issues, malformed client requests, filesystem problems, resource exhaustion, or concurrent access conflicts. Each error source requires different detection mechanisms, recovery strategies, and user-facing responses. The server must maintain service availability for healthy connections while gracefully handling problematic ones.

### Error Categories

Understanding error classification provides the foundation for systematic error handling. Different error categories require different detection methods, recovery strategies, and HTTP status code mappings. This classification system enables consistent error handling across all server components.

**Network Layer Errors** represent the most fundamental category since they affect the basic communication channel between client and server. These errors occur during socket operations, connection management, and data transmission. Network errors often indicate external conditions beyond the server's control, such as client disconnections, network timeouts, or system resource limitations.

**Protocol Layer Errors** emerge during HTTP message parsing and protocol compliance checking. These errors indicate malformed requests, unsupported HTTP features, or protocol violations by clients. Unlike network errors, protocol errors usually result from client implementation problems or malicious requests rather than infrastructure issues.

**Application Layer Errors** occur during request processing, specifically file system operations and path resolution. These errors indicate problems with the requested resources, security violations, or server configuration issues. Application errors often require detailed logging for security monitoring since they may indicate attack attempts.

**Resource Management Errors** arise from system resource exhaustion, memory allocation failures, or concurrency limits being exceeded. These errors indicate server capacity issues and often require immediate load shedding or graceful degradation to maintain service for existing connections.

**Security Errors** represent a special category that spans multiple layers but requires elevated attention due to their potential security implications. Directory traversal attempts, permission violations, and suspicious request patterns fall into this category and trigger additional logging and monitoring.

| Error Category | Typical Causes | Detection Method | Immediate Response | Recovery Strategy |
|---|---|---|---|---|
| Network Layer | Connection drops, socket errors, timeouts | System call return codes, errno values | Close connection, cleanup resources | Retry for transient errors, log persistent issues |
| Protocol Layer | Malformed requests, invalid headers, unsupported methods | Parser state machine errors, validation failures | Generate 4xx error response | Continue processing other connections |
| Application Layer | File not found, permission denied, path resolution failures | File system API errors, security validation | Generate 4xx/5xx error response | Log security events, continue serving |
| Resource Management | Memory exhaustion, thread pool full, file descriptor limits | Resource allocation failures, capacity checks | Reject new connections, return 503 responses | Implement backpressure, graceful degradation |
| Security | Directory traversal, suspicious patterns, access violations | Path validation, pattern detection | Block request, log security event | Update security rules, potential IP blocking |

The error category determines the appropriate logging level, response generation strategy, and resource cleanup requirements. Network layer errors typically require connection termination and resource cleanup but minimal logging since they're often transient. Protocol layer errors generate HTTP error responses and moderate logging for debugging purposes. Application layer errors trigger detailed request logging for troubleshooting. Resource management errors initiate capacity management procedures and high-priority alerts. Security errors activate comprehensive logging and monitoring workflows.

> **Critical Design Insight**: Error classification must happen as early as possible in the request processing pipeline. Early classification enables appropriate resource allocation for error handling and prevents resource waste on requests that will ultimately fail. The classification also determines whether the error represents a recoverable condition or requires connection termination.

### HTTP Error Response Generation

Converting internal system errors into appropriate HTTP status codes and error responses requires a systematic mapping strategy. The HTTP specification provides standardized status codes that communicate error conditions to clients in a universally understood format. However, the mapping between internal error states and HTTP responses must balance informativeness with security considerations.

**Status Code Selection Strategy** follows HTTP semantic conventions while considering security implications. The server must provide enough information for legitimate clients to understand and potentially retry requests, while avoiding information disclosure that could assist attackers. This balance influences both status code selection and error message content.

The `generate_error_response` function serves as the central error response factory, ensuring consistent formatting and appropriate security filtering across all error conditions. This centralized approach prevents inconsistent error handling and ensures security policies apply uniformly.

| Internal Error Type | HTTP Status Code | Status Text | Response Body Strategy | Security Considerations |
|---|---|---|---|---|
| `FILE_ERROR_NOT_FOUND` | 404 | Not Found | Generic "Resource not found" message | Never reveal internal path structure |
| `FILE_ERROR_PERMISSION` | 403 | Forbidden | Generic "Access denied" message | Don't distinguish between non-existent and forbidden |
| `PATH_ERROR_TRAVERSAL` | 400 | Bad Request | Generic "Invalid request" message | Don't reveal security validation details |
| `PATH_ERROR_TOO_LONG` | 414 | Request-URI Too Long | Standard HTTP message | Safe to be specific about length limits |
| `PARSE_ERROR_INVALID_METHOD` | 405 | Method Not Allowed | Include Allow header with supported methods | Include proper Allow header for compliance |
| `PARSE_ERROR_MALFORMED_HEADER` | 400 | Bad Request | Generic "Malformed request" message | Don't reveal parsing implementation details |
| `CAPACITY_ERROR_THREAD_POOL` | 503 | Service Unavailable | Include Retry-After header | Indicate temporary condition to encourage retry |
| Socket read/write errors | 500 | Internal Server Error | Generic server error message | Never expose internal error details |
| Memory allocation failures | 500 | Internal Server Error | Generic server error message | Resource exhaustion details are internal |

The error response generation process follows a structured approach to ensure consistency and security compliance. First, the error classification determines the appropriate HTTP status code family (4xx for client errors, 5xx for server errors). Second, the specific status code selection considers both HTTP semantics and security implications. Third, the response body generation applies security filtering to prevent information disclosure while providing actionable information for legitimate clients.

**Error Response Structure** maintains consistency across all error conditions. Every error response includes standard HTTP headers, properly formatted status lines, and sanitized response bodies. The response structure includes security headers that apply regardless of error type, ensuring consistent security posture.

```c
typedef struct {
    int status_code;           // HTTP status code (400-599 range)
    char status_text[64];      // Standard HTTP reason phrase
    char error_type[32];       // Internal error classification
    char sanitized_message[256]; // Security-filtered user message
    int include_retry_after;   // Whether to include Retry-After header
    int retry_seconds;         // Retry-After header value
} ErrorResponse;
```

The error response generation algorithm ensures consistent processing while allowing customization for specific error types:

1. **Error Classification**: Determine the error category and internal error type from the failure condition
2. **Status Code Mapping**: Apply the status code mapping table to select the appropriate HTTP response code
3. **Message Sanitization**: Filter the error message to remove internal implementation details and potential security information
4. **Header Generation**: Include standard error response headers plus any error-specific headers like Retry-After or Allow
5. **Response Assembly**: Construct the complete HTTP response with proper formatting and content length calculation
6. **Security Header Injection**: Add security-related headers that apply to all error responses
7. **Logging Coordination**: Ensure error logging occurs with appropriate detail level before response transmission

> **Decision: Centralized Error Response Generation**
> - **Context**: Error responses must be consistent across all server components while maintaining security filtering and proper HTTP compliance
> - **Options Considered**: Component-specific error handling, centralized error factory, hybrid approach with component customization
> - **Decision**: Centralized error response factory with configurable message templates
> - **Rationale**: Centralized generation ensures consistent security filtering, HTTP compliance, and response formatting while allowing customization through error type classification
> - **Consequences**: All components must use the central error factory, but this ensures security policies apply uniformly and reduces code duplication

**Security-Aware Error Messaging** prevents information disclosure while providing actionable feedback. The error message generation process applies multiple filtering layers to remove sensitive information that could assist attackers or reveal implementation details. Internal error messages contain full diagnostic information for logging and debugging, while external error messages provide only sanitized, user-appropriate information.

The message sanitization process removes file system paths, internal error codes, memory addresses, configuration details, and other implementation-specific information from user-facing error messages. This filtering happens automatically in the `generate_error_response` function, ensuring consistent security posture regardless of which component detected the error.

![Error Handling and Recovery](./diagrams/error-handling-flow.svg)

### Resource Cleanup Strategies

Proper resource cleanup becomes critical when errors occur, as failed request processing can leave allocated resources in an inconsistent state. Think of resource cleanup like the closing procedures at a restaurant - when something goes wrong during service, there must be clear protocols for clearing tables, returning borrowed items, and resetting the space for the next customer. Without systematic cleanup, resources accumulate and eventually exhaust system capacity.

The challenge in error-time resource cleanup lies in maintaining consistency across multiple resource types while handling partial failures. A single request may involve socket file descriptors, heap-allocated memory, temporary buffers, file handles, thread resources, and connection context structures. When an error occurs partway through processing, some resources may be allocated while others remain uninitialized, requiring careful tracking and conditional cleanup.

**Resource Ownership Tracking** establishes clear responsibility for resource lifecycle management. Each resource must have a single owner responsible for both allocation and deallocation. The `ResourceCleanupContext` structure provides a centralized tracking mechanism that records resource allocation and coordinates cleanup activities across error conditions.

| Resource Type | Allocation Owner | Cleanup Trigger | Cleanup Method | Failure Handling |
|---|---|---|---|---|
| Client Socket FD | TCP Server Component | Connection termination | `close()` system call | Log error, continue cleanup |
| Request Memory | HTTP Parser Component | Request processing completion | `free()` heap memory | Mark as freed, continue |
| Response Memory | File Handler Component | Response transmission completion | `free()` heap memory | Mark as freed, continue |
| File Handles | File Handler Component | File serving completion | `fclose()` file descriptor | Log error, continue cleanup |
| Thread Resources | Concurrency Manager | Thread termination | `pthread_detach()` or `pthread_join()` | Mark thread as orphaned |
| Connection Context | Connection Manager | Connection lifecycle end | `cleanup_connection_context()` | Free partial state |

The `ResourceCleanupContext` structure maintains comprehensive resource tracking throughout request processing. This tracking enables systematic cleanup even when errors occur at any point in the request handling pipeline. The cleanup context uses bit flags to indicate which resources require cleanup, allowing the cleanup process to skip unallocated resources safely.

```c
typedef struct {
    HTTPRequest* request;           // NULL if not allocated
    HTTPResponse* response;         // NULL if not allocated  
    ConnectionContext* connection;  // NULL if not allocated
    int client_fd;                 // -1 if not allocated
    int cleanup_flags;             // Bitmask indicating allocated resources
    time_t error_time;             // When error occurred for timeout tracking
    char error_context[256];       // Description of error location
} ResourceCleanupContext;
```

The cleanup flags use bit manipulation to efficiently track resource allocation status. This approach provides O(1) checking for resource status and enables atomic updates to allocation state. The flag definitions correspond to the major resource categories that require explicit cleanup.

| Cleanup Flag | Resource Indicates | Set When | Cleared When | Used By |
|---|---|---|---|---|
| `CLEANUP_REQUEST` | HTTPRequest allocated | `init_http_request()` succeeds | `cleanup_http_request()` completes | HTTP Parser Component |
| `CLEANUP_RESPONSE` | HTTPResponse allocated | `init_http_response()` succeeds | `cleanup_http_response()` completes | File Handler Component |
| `CLEANUP_CONNECTION` | ConnectionContext allocated | `init_connection_context()` succeeds | `cleanup_connection_context()` completes | Connection Manager |
| `CLEANUP_SOCKET` | Client socket open | `accept()` succeeds | `close()` completes | TCP Server Component |

**Cleanup Sequencing** ensures resources are released in the correct order to avoid dependency violations and potential crashes. Some resources depend on others remaining valid during cleanup, requiring careful ordering of deallocation operations. The cleanup sequence follows the reverse order of allocation, ensuring dependencies remain valid throughout the cleanup process.

The systematic cleanup algorithm processes resources in dependency order while handling partial cleanup failures gracefully:

1. **Error Detection and Context Initialization**: When an error occurs, initialize a `ResourceCleanupContext` structure with the current resource allocation state and error details
2. **Response Resource Cleanup**: If response resources were allocated, call `cleanup_http_response()` to free response body memory and reset response structure
3. **Request Resource Cleanup**: If request resources were allocated, call `cleanup_http_request()` to free request body memory and header storage
4. **File Handle Cleanup**: If file handles remain open from file serving operations, close them and log any cleanup failures
5. **Connection Context Cleanup**: If connection context was allocated, call `cleanup_connection_context()` to free connection-specific memory and reset state
6. **Socket Cleanup**: If the client socket remains open, close the file descriptor and log the connection termination
7. **Thread Resource Cleanup**: If thread-specific resources were allocated, detach or join threads as appropriate for the concurrency model
8. **Cleanup Verification**: Verify all resources marked for cleanup have been properly released and log any cleanup failures for debugging

> **Decision: RAII-Style Cleanup with Explicit Context**
> - **Context**: C lacks automatic resource management, requiring explicit cleanup coordination across error paths
> - **Options Considered**: Manual cleanup at each error site, cleanup callback registration, centralized cleanup context
> - **Decision**: Centralized cleanup context with explicit resource tracking and ordered cleanup
> - **Rationale**: Centralized context ensures consistent cleanup across all error paths while explicit tracking prevents double-free errors and resource leaks
> - **Consequences**: All resource allocation must update the cleanup context, but this ensures systematic cleanup regardless of error location

**Error Recovery vs. Connection Termination** requires careful consideration of error severity and recovery feasibility. Some errors allow request processing to continue with degraded functionality, while others require immediate connection termination to prevent further problems. The decision depends on error type, resource state, and potential impact on other connections.

**Recoverable Error Conditions** allow the connection to remain open for additional requests after appropriate error response generation. Protocol-level errors like malformed headers or unsupported methods can generate error responses without affecting the underlying connection. File system errors like missing files or permission problems also allow connection reuse since they don't indicate connection-level problems.

**Non-Recoverable Error Conditions** require immediate connection termination to prevent cascading failures or security breaches. Network-level errors like socket read failures or connection timeouts indicate unreliable communication channels. Resource exhaustion errors suggest system-wide problems that may affect all connections. Security errors like directory traversal attempts may indicate malicious clients requiring immediate disconnection.

| Error Condition | Recovery Strategy | Connection Fate | Resource Priority | Logging Level |
|---|---|---|---|---|
| Malformed HTTP headers | Generate 400 response | Keep connection open | Normal cleanup | INFO |
| File not found | Generate 404 response | Keep connection open | Normal cleanup | DEBUG |
| Socket read failure | Immediate cleanup | Terminate connection | Emergency cleanup | WARN |
| Memory allocation failure | Generate 503 response | Terminate connection | Emergency cleanup | ERROR |
| Directory traversal attempt | Generate 400 response | Terminate connection | Normal cleanup + security log | SECURITY |
| Thread pool exhaustion | Generate 503 response | Queue for later processing | Delayed cleanup | WARN |

The recovery strategy selection process considers both immediate error handling and long-term system stability. Recoverable errors focus on providing appropriate client feedback while maintaining connection state for future requests. Non-recoverable errors prioritize system stability and security, sacrificing the individual connection to protect overall server health.

**Emergency Cleanup Procedures** handle situations where normal cleanup processes may fail or take too long. When system resources are critically low or security events require immediate response, emergency cleanup bypasses normal cleanup verification and focuses on rapidly releasing critical resources. Emergency cleanup trades thoroughness for speed, accepting some resource leakage to prevent system-wide failure.

The emergency cleanup process prioritizes file descriptors and memory over cleanup verification, since these resources most directly impact system capacity. Connection context and thread resources receive lower priority since their leakage has less immediate system impact. Emergency cleanup includes aggressive timeout handling to prevent cleanup operations from blocking system recovery.

### Common Error Handling Pitfalls

Understanding typical error handling mistakes helps developers avoid subtle bugs that can cause resource leaks, security vulnerabilities, or service disruption. These pitfalls often arise from the complexity of coordinating cleanup across multiple system components and the edge cases that occur under error conditions.

⚠️ **Pitfall: Double-Close of File Descriptors**

One of the most common resource management errors involves closing the same file descriptor multiple times. This occurs when error handling code closes a socket or file handle, but subsequent cleanup code attempts to close the same descriptor again. The second close operation may succeed silently, close an unrelated file descriptor that was allocated after the first close, or generate an error that disrupts cleanup of other resources.

The problem manifests when cleanup code doesn't properly track whether resources have already been released. Multiple code paths may each attempt to close the same descriptor, especially when error handling interrupts normal cleanup sequences. File descriptor reuse by the operating system means the second close may affect a completely different resource, leading to difficult-to-debug corruption.

To prevent double-close errors, always set file descriptors to -1 after closing them and check for -1 before attempting to close. The cleanup functions should verify descriptor validity and update resource tracking immediately after successful close operations.

⚠️ **Pitfall: Incomplete Error Information in Log Messages**

Error log messages often lack sufficient context for effective debugging, making it difficult to reproduce problems or understand their root causes. Generic error messages like "request failed" or "file error" provide little actionable information when troubleshooting server problems. Missing context information like client IP addresses, request paths, or error timing makes it nearly impossible to correlate errors with specific client behaviors or system conditions.

This problem occurs when error handling code focuses on resource cleanup without capturing diagnostic information at the error site. By the time cleanup completes, much of the context that would help identify the error cause has been deallocated or overwritten. Error messages that only include system error codes without application-specific context force debugging efforts to rely on external log correlation.

Effective error logging captures error context at the detection site before cleanup begins. Include client identification, request details, system state, and precise error location in all error log entries. Use structured logging formats that enable automated analysis and correlation across multiple error events.

⚠️ **Pitfall: Resource Cleanup Failures Blocking Shutdown**

Cleanup operations themselves can fail, potentially blocking server shutdown or causing resource accumulation. When cleanup code encounters errors like filesystem permission problems or network timeouts, it may block indefinitely waiting for cleanup operations to complete. This blocking can prevent graceful shutdown and force administrators to kill server processes, potentially causing data loss or corruption.

The problem becomes critical during high error rates when many cleanup operations execute simultaneously. If cleanup operations have unbounded timeouts or lack failure handling, a cascade of cleanup failures can overwhelm system resources and prevent recovery. Cleanup code that retries operations indefinitely can consume excessive CPU and memory resources even when the underlying problems are unrecoverable.

Implement timeout-based cleanup with failure tolerance to prevent cleanup operations from blocking system recovery. Cleanup operations should have strict time limits and continue with partial cleanup when individual operations fail. Log cleanup failures for debugging but don't allow them to prevent overall resource release.

⚠️ **Pitfall: Security Information Disclosure in Error Messages**

Error messages that include internal system information can provide attackers with valuable reconnaissance data about server implementation, filesystem structure, or security mechanisms. Error messages containing file paths, internal error codes, memory addresses, or configuration details help attackers understand system internals and identify potential attack vectors.

This information disclosure often occurs when development-focused error messages reach production environments without proper filtering. Database error messages, filesystem paths, stack traces, and internal function names all provide implementation details that assist in attack planning. Even seemingly innocent information like exact file locations or error timing can reveal system structure to determined attackers.

The solution requires systematic message sanitization that removes internal details while preserving actionable information for legitimate users. Implement separate error message systems for internal logging and external client communication, ensuring sensitive details never reach client-facing error responses.

⚠️ **Pitfall: Memory Leaks During Error Conditions**

Memory allocated during request processing may not be properly freed when errors interrupt normal execution flow. This problem is particularly common in C where manual memory management requires explicit free operations for every malloc. Error conditions that cause early return from functions can skip cleanup code, leading to gradual memory consumption that eventually exhausts system resources.

The issue compounds during high error rates when many requests encounter problems simultaneously. Each leaked allocation is typically small, making the problem difficult to detect during normal testing. However, sustained error conditions can accumulate significant memory usage that impacts overall system performance and stability.

Implement systematic memory tracking using the cleanup context pattern to ensure all allocations are tracked and freed regardless of error conditions. Use tools like valgrind during development to detect memory leaks and verify cleanup code paths execute properly under error conditions.

### Implementation Guidance

The error handling implementation requires careful coordination across all server components to ensure consistent behavior and proper resource management. The following guidance provides complete implementations for infrastructure components and detailed skeletons for core error handling logic.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|---|---|---|
| Error Logging | Standard C printf to stderr | Structured logging with syslog integration |
| Resource Tracking | Manual bit flags in cleanup context | Automated RAII-style wrappers |
| Error Response Generation | Template-based string formatting | JSON-based error response with schema validation |

**Recommended File Structure:**
```
src/
  error_handling.h          ← Error types, cleanup context, function declarations
  error_handling.c          ← Core error handling and cleanup implementation
  http_errors.h            ← HTTP status code mappings and response generation
  http_errors.c            ← HTTP error response implementation
  logging.h                ← Logging interface and security event handling
  logging.c                ← Logging implementation with different output targets
  tests/
    test_error_handling.c   ← Unit tests for error classification and cleanup
    test_http_errors.c      ← Integration tests for error response generation
```

**Complete Error Type Infrastructure:**
```c
// error_handling.h - Complete error type system ready for use

#include <errno.h>
#include <time.h>
#include <pthread.h>

// Error category definitions for systematic classification
typedef enum {
    ERROR_CATEGORY_NETWORK,     // Socket and TCP connection errors
    ERROR_CATEGORY_PROTOCOL,    // HTTP parsing and protocol violations  
    ERROR_CATEGORY_APPLICATION, // File system and request processing errors
    ERROR_CATEGORY_RESOURCE,    // Memory, threads, and capacity errors
    ERROR_CATEGORY_SECURITY     // Security violations and suspicious activity
} ErrorCategory;

// Specific error codes within each category
typedef enum {
    // Network layer errors
    NETWORK_ERROR_CONNECTION_LOST,
    NETWORK_ERROR_SOCKET_READ,
    NETWORK_ERROR_SOCKET_WRITE,
    NETWORK_ERROR_TIMEOUT,
    
    // Protocol layer errors  
    PROTOCOL_ERROR_MALFORMED_REQUEST,
    PROTOCOL_ERROR_INVALID_METHOD,
    PROTOCOL_ERROR_UNSUPPORTED_VERSION,
    PROTOCOL_ERROR_HEADER_TOO_LONG,
    
    // Application layer errors
    APP_ERROR_FILE_NOT_FOUND,
    APP_ERROR_PERMISSION_DENIED,
    APP_ERROR_PATH_TRAVERSAL,
    APP_ERROR_PATH_TOO_LONG,
    
    // Resource management errors
    RESOURCE_ERROR_MEMORY_ALLOCATION,
    RESOURCE_ERROR_THREAD_POOL_FULL,
    RESOURCE_ERROR_FD_EXHAUSTION,
    
    // Security errors
    SECURITY_ERROR_SUSPICIOUS_PATH,
    SECURITY_ERROR_RATE_LIMIT,
    SECURITY_ERROR_BLOCKED_IP
} SpecificErrorCode;

// Complete error information structure
typedef struct {
    ErrorCategory category;           // High-level error classification
    SpecificErrorCode specific_code;  // Detailed error identification
    int system_errno;                 // System error code if applicable
    time_t timestamp;                 // When error occurred
    char client_ip[INET_ADDRSTRLEN];  // Client identification
    char context_info[256];           // Detailed error context
    char internal_message[512];       // Full diagnostic information
    char external_message[256];       // Sanitized user-facing message
} ErrorInfo;

// Resource cleanup context with comprehensive tracking
typedef struct {
    HTTPRequest* request;
    HTTPResponse* response;  
    ConnectionContext* connection;
    int client_fd;
    int cleanup_flags;
    time_t error_time;
    char error_context[256];
    pthread_mutex_t cleanup_mutex;    // Thread safety for cleanup
} ResourceCleanupContext;

// Infrastructure function declarations
ErrorInfo* create_error_info(ErrorCategory category, SpecificErrorCode code, 
                            const char* context);
void log_error_with_context(const ErrorInfo* error);
void log_security_event(const char* event_type, const char* client_ip,
                        const char* details);
int cleanup_request_resources(ResourceCleanupContext* cleanup);
int emergency_cleanup(ResourceCleanupContext* cleanup, int timeout_sec);
```

**Complete HTTP Error Response Infrastructure:**
```c
// http_errors.c - Complete HTTP error response system

#include "http_errors.h"
#include <string.h>
#include <time.h>

// Status code mapping table - complete and ready to use
typedef struct {
    SpecificErrorCode error_code;
    int http_status;
    const char* status_text;
    const char* template_message;
    int include_retry_after;
} ErrorMapping;

static const ErrorMapping error_mappings[] = {
    {APP_ERROR_FILE_NOT_FOUND, 404, "Not Found", "The requested resource was not found", 0},
    {APP_ERROR_PERMISSION_DENIED, 403, "Forbidden", "Access to the requested resource is forbidden", 0},
    {APP_ERROR_PATH_TRAVERSAL, 400, "Bad Request", "The request contains invalid path information", 0},
    {PROTOCOL_ERROR_INVALID_METHOD, 405, "Method Not Allowed", "The request method is not supported", 0},
    {PROTOCOL_ERROR_MALFORMED_REQUEST, 400, "Bad Request", "The request is malformed", 0},
    {RESOURCE_ERROR_THREAD_POOL_FULL, 503, "Service Unavailable", "The server is temporarily overloaded", 1},
    {RESOURCE_ERROR_MEMORY_ALLOCATION, 500, "Internal Server Error", "An internal server error occurred", 0},
    {NETWORK_ERROR_TIMEOUT, 408, "Request Timeout", "The request timed out", 0}
};

// Complete HTTP error response generation - ready to use
void generate_error_response(int status_code, const char* error_message, HTTPResponse* response) {
    // Initialize response structure
    init_http_response(response);
    
    // Set status information
    response->status_code = status_code;
    
    // Find appropriate status text
    const char* status_text = "Unknown Error";
    for (int i = 0; i < sizeof(error_mappings) / sizeof(ErrorMapping); i++) {
        if (error_mappings[i].http_status == status_code) {
            status_text = error_mappings[i].status_text;
            break;
        }
    }
    strncpy(response->status_text, status_text, sizeof(response->status_text) - 1);
    
    // Create HTML error page body
    char html_body[1024];
    snprintf(html_body, sizeof(html_body),
        "<!DOCTYPE html>\n"
        "<html><head><title>%d %s</title></head>\n"
        "<body><h1>%d %s</h1>\n"
        "<p>%s</p>\n"
        "<hr><p>HTTP Server</p></body></html>\n",
        status_code, status_text, status_code, status_text, error_message);
    
    // Allocate and copy response body
    response->body_length = strlen(html_body);
    response->body = malloc(response->body_length + 1);
    if (response->body) {
        strcpy(response->body, html_body);
    }
    
    // Add standard error response headers
    add_response_header(response, "Content-Type", "text/html; charset=utf-8");
    add_response_header(response, "Connection", "close");
    
    // Add Content-Length header
    char content_length[32];
    snprintf(content_length, sizeof(content_length), "%zu", response->body_length);
    add_response_header(response, "Content-Length", content_length);
    
    // Add security headers
    add_response_header(response, "X-Content-Type-Options", "nosniff");
    add_response_header(response, "X-Frame-Options", "DENY");
}
```

**Core Error Handling Skeleton:**
```c
// error_handling.c - Core logic to implement

// Error classification function - implement this
ErrorCategory classify_error(int system_errno, const char* context) {
    // TODO 1: Check for network-related errno values (ECONNRESET, EPIPE, ETIMEDOUT)
    //         Return ERROR_CATEGORY_NETWORK for connection issues
    
    // TODO 2: Check for file system errno values (ENOENT, EACCES, EPERM)  
    //         Return ERROR_CATEGORY_APPLICATION for file access issues
    
    // TODO 3: Check for resource errno values (ENOMEM, EAGAIN, EMFILE)
    //         Return ERROR_CATEGORY_RESOURCE for capacity issues
    
    // TODO 4: Analyze context string for security indicators
    //         Look for ".." patterns, suspicious characters
    //         Return ERROR_CATEGORY_SECURITY for potential attacks
    
    // TODO 5: Default to ERROR_CATEGORY_PROTOCOL for parsing/HTTP issues
    //         This handles cases not covered by system errno
    
    return ERROR_CATEGORY_PROTOCOL; // Placeholder
}

// Complete error information creation - implement this  
ErrorInfo* create_error_info(ErrorCategory category, SpecificErrorCode code, const char* context) {
    // TODO 1: Allocate ErrorInfo structure with malloc, check for NULL return
    
    // TODO 2: Set category, specific_code, and current timestamp
    //         Use time() function for timestamp
    
    // TODO 3: Copy context string to context_info field with bounds checking
    //         Use strncpy to prevent buffer overflow
    
    // TODO 4: Set system_errno to current errno value
    //         Save errno immediately since other calls may modify it
    
    // TODO 5: Generate internal_message with full diagnostic details
    //         Include errno string using strerror(), context, timestamp
    
    // TODO 6: Generate external_message with sanitized information  
    //         Remove internal paths, error codes, implementation details
    //         Use security-safe generic messages for external consumption
    
    return NULL; // Placeholder - return allocated ErrorInfo*
}

// Resource cleanup with error handling - implement this
int cleanup_request_resources(ResourceCleanupContext* cleanup) {
    if (!cleanup) return -1;
    
    int cleanup_errors = 0;
    
    // TODO 1: Lock cleanup mutex for thread safety
    //         Use pthread_mutex_lock with error checking
    
    // TODO 2: Check CLEANUP_RESPONSE flag and call cleanup_http_response()
    //         Clear flag after successful cleanup
    //         Increment cleanup_errors if cleanup fails
    
    // TODO 3: Check CLEANUP_REQUEST flag and call cleanup_http_request()
    //         Clear flag after successful cleanup  
    //         Increment cleanup_errors if cleanup fails
    
    // TODO 4: Check CLEANUP_CONNECTION flag and call cleanup_connection_context()
    //         Clear flag after successful cleanup
    //         Increment cleanup_errors if cleanup fails
    
    // TODO 5: Check CLEANUP_SOCKET flag and close client_fd if valid (>= 0)
    //         Set client_fd to -1 after successful close
    //         Clear CLEANUP_SOCKET flag
    //         Increment cleanup_errors if close fails
    
    // TODO 6: Unlock cleanup mutex
    //         Use pthread_mutex_unlock
    
    // TODO 7: Return 0 for complete success, positive value for partial failures
    //         Log cleanup errors for debugging but don't fail overall cleanup
    
    return cleanup_errors;
}
```

**Security Event Logging Infrastructure:**
```c
// logging.c - Complete security logging system

#include <syslog.h>
#include <stdio.h>

// Initialize logging system - complete implementation
void init_logging_system(const char* server_name) {
    openlog(server_name, LOG_PID | LOG_CONS, LOG_DAEMON);
}

// Security event logging with context - complete implementation  
void log_security_event(const char* event_type, const char* client_ip, 
                       const char* request_path, const char* details) {
    char log_message[1024];
    time_t now = time(NULL);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
    
    snprintf(log_message, sizeof(log_message),
        "SECURITY_EVENT: %s | Time: %s | Client: %s | Path: %s | Details: %s",
        event_type, timestamp, client_ip ? client_ip : "unknown", 
        request_path ? request_path : "none", details ? details : "none");
    
    // Log to syslog for centralized security monitoring
    syslog(LOG_WARNING, "%s", log_message);
    
    // Also log to stderr for development/debugging
    fprintf(stderr, "[SECURITY] %s\n", log_message);
}

// Error logging with full context - complete implementation
void log_error_with_context(const ErrorInfo* error) {
    if (!error) return;
    
    const char* category_names[] = {
        "NETWORK", "PROTOCOL", "APPLICATION", "RESOURCE", "SECURITY"
    };
    
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", 
             localtime(&error->timestamp));
    
    // Determine log level based on category
    int log_level = LOG_INFO;
    switch (error->category) {
        case ERROR_CATEGORY_SECURITY: log_level = LOG_WARNING; break;
        case ERROR_CATEGORY_RESOURCE: log_level = LOG_ERR; break;
        case ERROR_CATEGORY_NETWORK: log_level = LOG_INFO; break;
        default: log_level = LOG_INFO; break;
    }
    
    syslog(log_level, "ERROR: %s | Time: %s | Client: %s | Context: %s | Details: %s",
           category_names[error->category], timestamp, error->client_ip,
           error->context_info, error->internal_message);
}
```

**Milestone Checkpoints:**

After implementing error handling:

1. **Test Error Classification**: Create test requests that trigger each error category. Verify correct status codes:
   ```bash
   # Test file not found
   curl -v http://localhost:8080/nonexistent.html
   # Should return 404 Not Found
   
   # Test directory traversal  
   curl -v http://localhost:8080/../etc/passwd
   # Should return 400 Bad Request and log security event
   
   # Test malformed request
   echo -e "INVALID HTTP REQUEST\r\n\r\n" | nc localhost 8080
   # Should return 400 Bad Request
   ```

2. **Verify Resource Cleanup**: Use process monitoring to confirm no resource leaks:
   ```bash
   # Monitor file descriptor usage
   lsof -p $(pgrep http_server) | wc -l
   # Count should remain stable under error conditions
   
   # Check memory usage under error load
   ps -p $(pgrep http_server) -o pid,vsz,rss
   # Memory should not grow continuously during error tests
   ```

3. **Check Security Logging**: Verify security events appear in logs:
   ```bash
   # Check syslog for security events
   tail -f /var/log/syslog | grep SECURITY_EVENT
   
   # Trigger directory traversal and verify logging
   curl http://localhost:8080/../../../../etc/passwd
   # Should see security event logged with client IP and path
   ```

**Debugging Tips:**

| Symptom | Likely Cause | Diagnosis | Fix |
|---|---|---|---|
| Server crashes on error | Double-free or NULL pointer access | Run with valgrind, check cleanup order | Add NULL checks, fix cleanup sequence |
| Memory usage grows over time | Resource leaks in error paths | Use valgrind --leak-check=full | Implement cleanup context tracking |
| Clients receive generic errors | Error message sanitization too aggressive | Check error mapping table | Add specific but safe error messages |
| Security events not logged | Logging initialization missing | Check syslog configuration | Call init_logging_system() at startup |
| Cleanup hangs during shutdown | Cleanup operations blocking indefinitely | Add timeout monitoring | Implement emergency_cleanup() with timeouts |


## Testing Strategy

> **Milestone(s):** All Milestones 1-4 - provides verification approach for TCP server basics, HTTP request parsing, static file serving, and concurrent connections with specific test scenarios and expected behaviors

The testing strategy for an HTTP server requires a systematic approach that verifies functionality at each development milestone while building confidence in the system's reliability and correctness. Think of testing an HTTP server like quality assurance in a restaurant kitchen - you need to verify each station (prep, cooking, plating) works correctly in isolation, then test how they coordinate during a busy dinner rush. Each milestone represents a different level of complexity, from basic socket operations to handling multiple simultaneous diners (clients) with different dietary requirements (HTTP requests).

The testing approach must balance thorough verification with practical implementation constraints. Unlike production web servers that undergo extensive automated testing suites, this educational HTTP server focuses on milestone-driven verification that builds understanding while ensuring correctness. The testing strategy emphasizes hands-on verification using standard tools that developers commonly use for HTTP server testing and debugging.

### Milestone Verification Checkpoints

Each milestone represents a significant functional capability that must be thoroughly verified before proceeding to the next development phase. The checkpoint approach ensures that fundamental issues are caught early, preventing compound problems that become difficult to diagnose in later milestones. Think of these checkpoints like safety inspections at each floor of a building under construction - you must verify the foundation before adding the framework, and the framework before installing utilities.

#### Milestone 1: TCP Server Basics Verification

The first milestone verification focuses on the fundamental networking capabilities that form the foundation for all HTTP functionality. At this stage, the server should demonstrate proper socket management, connection handling, and basic request-response cycles without any HTTP protocol awareness.

**Socket Creation and Binding Verification:**

The initial test verifies that the server can create and configure a listening socket correctly. Start the server and verify it binds to the specified port without errors. Use the `netstat` command to confirm the server socket is in the LISTEN state:

```bash
netstat -tlnp | grep :8080
```

Expected output should show a TCP socket in LISTEN state bound to port 8080. The server should accept the `SO_REUSEADDR` socket option, allowing rapid restart without encountering "Address already in use" errors. If binding fails, common causes include port conflicts with other services or insufficient privileges for ports below 1024.

**Connection Acceptance Testing:**

Test the server's ability to accept incoming TCP connections by establishing a raw TCP connection using telnet. This test verifies the accept loop and basic connection handling without HTTP protocol concerns:

```bash
telnet localhost 8080
```

Upon successful connection, telnet should report "Connected to localhost" and provide a command prompt. The server should accept the connection without blocking or crashing. Multiple sequential connections should work correctly, demonstrating proper cleanup of previous connections.

**Basic Request-Response Cycle:**

Send a simple string through the telnet connection and verify the server reads the data and responds with a hardcoded HTTP response. At this milestone, the server doesn't need to understand HTTP format - it should read any incoming data and respond with a fixed HTTP-formatted message:

Expected response format:
```
HTTP/1.1 200 OK
Content-Type: text/plain
Content-Length: 13

Hello, World!
```

The server should close the connection after sending the response, and telnet should indicate the connection was closed by the remote host.

**Resource Management Verification:**

Test proper cleanup by establishing multiple sequential connections and monitoring file descriptors. Use the `lsof` command to verify the server doesn't leak file descriptors:

```bash
lsof -p <server_process_id> | grep TCP
```

After each connection closes, the client file descriptor should disappear from the lsof output, indicating proper cleanup.

**Error Handling at TCP Level:**

Test error conditions specific to TCP operations:
- Start two server instances on the same port - the second should fail gracefully with an appropriate error message
- Kill the server process and attempt connection - clients should receive connection refused
- Send data to the server and immediately close the client connection - server should handle the disconnection without crashing

#### Milestone 2: HTTP Request Parsing Verification

The second milestone verification focuses on the server's ability to parse HTTP requests correctly and respond based on the parsed information. This represents a significant complexity increase from basic TCP handling to protocol-aware request processing.

**HTTP Request Line Parsing:**

Test the server's ability to extract method, path, and version from various HTTP request formats. Use curl to send requests with different methods and paths:

```bash
curl -v http://localhost:8080/test/path
curl -v -X POST http://localhost:8080/api/endpoint
curl -v -X HEAD http://localhost:8080/
```

The server should correctly parse each request and potentially log the extracted components. Even if file serving isn't implemented yet, the server should demonstrate it understands the differences between GET, POST, and HEAD requests and can extract the requested path.

**HTTP Header Parsing Testing:**

Send requests with various header configurations to test header parsing robustness:

```bash
curl -H "Host: example.com" -H "User-Agent: TestClient/1.0" http://localhost:8080/
curl -H "Content-Type: application/json" http://localhost:8080/
```

The server should parse headers into key-value pairs and handle common HTTP headers correctly. Test edge cases like headers with extra whitespace around values and headers spanning multiple lines (if supported).

**Malformed Request Handling:**

Test the server's response to malformed HTTP requests using telnet to send malformed data:

- Send incomplete request line: `GET /path` (missing HTTP version)
- Send malformed headers: `InvalidHeader without colon`
- Send requests with invalid methods: `INVALID /path HTTP/1.1`

The server should respond with appropriate HTTP error codes (400 Bad Request) rather than crashing or hanging.

**Content-Length and Body Handling:**

If body parsing is implemented, test POST requests with message bodies:

```bash
curl -X POST -d "test data" -H "Content-Type: text/plain" http://localhost:8080/
```

The server should read the exact number of bytes specified in the Content-Length header and not attempt to read beyond the message body.

#### Milestone 3: Static File Serving Verification

The third milestone verification tests the server's ability to serve files from the filesystem safely and correctly, implementing the core functionality expected from a static web server.

**Basic File Serving:**

Create a test directory structure with various file types:

```
document_root/
  index.html
  style.css
  script.js
  images/
    logo.png
    background.jpg
  documents/
    readme.txt
```

Test file retrieval with curl:

```bash
curl -v http://localhost:8080/index.html
curl -v http://localhost:8080/style.css
curl -v http://localhost:8080/images/logo.png
```

Verify that:
- Files are served with correct Content-Type headers based on file extensions
- Content-Length headers match actual file sizes
- Binary files (images) are served without corruption
- File contents in the response body match the actual file contents exactly

**Directory Listing Testing:**

Test directory requests to verify directory listing functionality:

```bash
curl -v http://localhost:8080/images/
curl -v http://localhost:8080/
```

If directory listings are implemented, verify they include:
- Links to all files and subdirectories
- Proper HTML formatting
- Parent directory navigation (..)
- File size and modification time information

**404 Error Handling:**

Test requests for non-existent files:

```bash
curl -v http://localhost:8080/nonexistent.html
curl -v http://localhost:8080/missing/path/file.txt
```

The server should respond with:
- HTTP 404 Not Found status code
- Appropriate error page content
- Proper Content-Type header for the error response

**Security Validation Testing:**

Test directory traversal attack prevention:

```bash
curl -v http://localhost:8080/../../../etc/passwd
curl -v http://localhost:8080/./../../sensitive_file
curl -v "http://localhost:8080/%2e%2e%2f%2e%2e%2fpasswd"
```

The server should reject these requests with 403 Forbidden or 400 Bad Request responses, never serving files outside the document root directory.

**MIME Type Detection Verification:**

Test various file types to verify correct MIME type detection:

| File Extension | Expected MIME Type | Test Command |
|---|---|---|
| .html | text/html | `curl -I http://localhost:8080/test.html` |
| .css | text/css | `curl -I http://localhost:8080/style.css` |
| .js | application/javascript | `curl -I http://localhost:8080/script.js` |
| .png | image/png | `curl -I http://localhost:8080/image.png` |
| .jpg | image/jpeg | `curl -I http://localhost:8080/photo.jpg` |
| .txt | text/plain | `curl -I http://localhost:8080/readme.txt` |

#### Milestone 4: Concurrent Connections Verification

The fourth milestone verification tests the server's ability to handle multiple simultaneous client connections without blocking or corrupting responses. This represents the most complex testing scenario, requiring coordination of multiple clients and verification of correct concurrent behavior.

**Sequential Connection Handling:**

Before testing true concurrency, verify that multiple sequential connections work correctly:

```bash
for i in {1..10}; do
  curl -s http://localhost:8080/index.html > /dev/null
  echo "Request $i completed"
done
```

All requests should complete successfully without errors or connection failures.

**Basic Concurrent Connection Testing:**

Test multiple simultaneous connections using background processes:

```bash
for i in {1..5}; do
  curl -s http://localhost:8080/index.html > response_$i.html &
done
wait
```

Verify that all response files contain identical, correct content and no responses are corrupted or incomplete due to concurrent access.

**Connection Limit Testing:**

Test the server's behavior when the maximum connection limit is reached:

```bash
# Create connections that stay open briefly
for i in {1..20}; do
  (sleep 2; curl -s http://localhost:8080/) &
done
```

The server should either queue additional connections (thread pool model) or reject excess connections gracefully rather than crashing or hanging.

**Resource Management Under Load:**

Monitor server resource usage during concurrent load:

```bash
# Monitor file descriptors
watch "lsof -p <server_pid> | wc -l"

# Monitor threads (if using threading)
watch "ps -T -p <server_pid>"
```

Verify that:
- File descriptor count returns to baseline after connections close
- Thread count remains within configured limits
- Memory usage doesn't grow unboundedly
- No zombie processes are created

### Testing Tools and Techniques

The testing strategy relies on a combination of standard HTTP client tools, low-level networking utilities, and custom testing scripts to verify functionality across all complexity levels. Each tool serves specific purposes in the verification process, from basic connectivity testing to detailed HTTP protocol validation.

#### Command-Line HTTP Clients

**curl - Primary HTTP Testing Tool:**

curl serves as the primary HTTP testing client due to its comprehensive HTTP protocol support and detailed output capabilities. The verbose flag (`-v`) provides essential debugging information including request headers, response headers, and connection details.

Common curl usage patterns for HTTP server testing:

| Test Purpose | curl Command | Key Flags Explained |
|---|---|---|
| Basic request | `curl -v http://localhost:8080/` | `-v` shows request/response headers |
| HEAD request | `curl -I http://localhost:8080/` | `-I` sends HEAD instead of GET |
| Custom headers | `curl -H "Host: test.com" http://localhost:8080/` | `-H` adds custom header |
| POST request | `curl -X POST -d "data" http://localhost:8080/` | `-X` specifies method, `-d` sends data |
| Follow redirects | `curl -L http://localhost:8080/redirect` | `-L` follows Location headers |
| Timeout control | `curl --max-time 5 http://localhost:8080/` | Prevents hanging on slow responses |
| Save response | `curl -o output.html http://localhost:8080/` | `-o` saves response to file |
| Show timing | `curl -w "@curl-format.txt" http://localhost:8080/` | Custom timing format |

**wget - Alternative HTTP Client:**

wget provides different capabilities useful for testing file download scenarios and recursive retrieval:

```bash
# Download files recursively
wget -r --no-parent http://localhost:8080/

# Test resume capability (if implemented)
wget -c http://localhost:8080/large-file.zip

# Mirror directory structure
wget -m http://localhost:8080/
```

#### Low-Level Network Testing Tools

**telnet - Raw TCP Connection Testing:**

telnet enables direct TCP connection testing without HTTP protocol formatting, essential for testing TCP server basics and debugging connection issues:

```bash
# Basic connection test
telnet localhost 8080

# Test connection timeout
timeout 5 telnet localhost 8080

# Scripted telnet for automated testing
echo -e "GET / HTTP/1.1\r\nHost: localhost\r\n\r\n" | telnet localhost 8080
```

**netcat (nc) - Advanced Network Testing:**

netcat provides more flexibility than telnet for network testing scenarios:

```bash
# Send HTTP request via netcat
echo -e "GET / HTTP/1.1\r\nHost: localhost\r\n\r\n" | nc localhost 8080

# Test UDP (if server supports it)
echo "test" | nc -u localhost 8080

# Port scanning to verify server is listening
nc -z localhost 8080
```

**openssl s_client - HTTPS Testing:**

If HTTPS support is added in future extensions:

```bash
# Test HTTPS connection
openssl s_client -connect localhost:8443 -servername localhost

# Test certificate validation
openssl s_client -connect localhost:8443 -verify_return_error
```

#### Browser Testing Strategies

**Interactive Browser Testing:**

Real browser testing provides essential validation of HTTP server behavior with actual web clients, revealing issues that command-line tools might miss:

1. **Basic Functionality Testing:** Open `http://localhost:8080/` in multiple browsers (Chrome, Firefox, Safari) to verify cross-browser compatibility
2. **Developer Tools Inspection:** Use browser developer tools (F12) to examine:
   - Network tab: Request/response headers, timing, status codes
   - Console tab: JavaScript errors if serving HTML with scripts
   - Sources tab: Verify file contents load correctly
3. **Cache Behavior Testing:** Use Ctrl+F5 (force refresh) to test cache headers and ensure proper cache validation

**Browser-Specific Testing Scenarios:**

| Test Scenario | Browser Action | Expected Behavior |
|---|---|---|
| Favicon request | Load any page | Server should handle `/favicon.ico` gracefully |
| Multiple tabs | Open same URL in multiple tabs | All tabs should load correctly |
| File download | Click download link | File should download without corruption |
| Image loading | Page with multiple images | All images load without broken links |
| Large file handling | Request large files | Download progresses without timeout |

#### Automated Testing Scripts

**Bash Testing Scripts:**

Create reusable test scripts for consistent verification across development iterations:

```bash
#!/bin/bash
# http_server_test.sh - Comprehensive server testing

SERVER_URL="http://localhost:8080"
TEST_DIR="test_files"

# Test basic connectivity
test_connectivity() {
    echo "Testing basic connectivity..."
    if curl -s --max-time 5 "$SERVER_URL/" > /dev/null; then
        echo "✓ Server is reachable"
    else
        echo "✗ Server connectivity failed"
        return 1
    fi
}

# Test file serving
test_file_serving() {
    echo "Testing file serving..."
    for file in "$TEST_DIR"/*.html "$TEST_DIR"/*.css "$TEST_DIR"/*.js; do
        filename=$(basename "$file")
        if curl -s "$SERVER_URL/$filename" | diff - "$file" > /dev/null; then
            echo "✓ $filename served correctly"
        else
            echo "✗ $filename content mismatch"
        fi
    done
}

# Test concurrent connections
test_concurrency() {
    echo "Testing concurrent connections..."
    for i in {1..10}; do
        curl -s "$SERVER_URL/" > "/tmp/response_$i" &
    done
    wait
    
    # Verify all responses are identical
    if diff /tmp/response_1 /tmp/response_2 > /dev/null; then
        echo "✓ Concurrent responses are consistent"
    else
        echo "✗ Concurrent responses differ"
    fi
}

# Run all tests
main() {
    test_connectivity || exit 1
    test_file_serving
    test_concurrency
    echo "Testing completed"
}

main "$@"
```

### Concurrency Testing

Concurrency testing represents the most complex verification challenge, requiring careful coordination of multiple clients and observation of server behavior under simultaneous load. Think of concurrency testing like stress-testing a restaurant during the dinner rush - you need to simulate multiple customers arriving simultaneously, ordering different items, and verify that each customer receives their correct order without delays caused by kitchen coordination issues.

#### Thread-Per-Connection Model Testing

The thread-per-connection concurrency model creates a dedicated thread for each client connection, making it relatively straightforward to test but important to verify proper thread management and resource cleanup.

**Basic Thread Creation Verification:**

Test that the server creates threads correctly for multiple connections:

```bash
# Monitor thread creation in real-time
watch "ps -T -p $(pgrep http_server) | wc -l"

# Create multiple connections and observe thread count
for i in {1..5}; do
    (sleep 10; curl http://localhost:8080/) &
done
```

Expected behavior: Thread count should increase as connections are accepted and decrease as they complete. Total thread count should not exceed the number of active connections plus the main thread.

**Thread Resource Management Testing:**

Verify that threads are properly cleaned up and don't accumulate over time:

```bash
# Baseline thread count
BASELINE=$(ps -T -p $(pgrep http_server) | wc -l)

# Create and complete many connections
for round in {1..10}; do
    for i in {1..5}; do
        curl -s http://localhost:8080/ > /dev/null &
    done
    wait
    sleep 1
done

# Check final thread count
FINAL=$(ps -T -p $(pgrep http_server) | wc -l)

if [ $FINAL -eq $BASELINE ]; then
    echo "✓ Thread cleanup working correctly"
else
    echo "✗ Thread leak detected: $BASELINE -> $FINAL"
fi
```

**Thread Safety Validation:**

Test that concurrent threads don't interfere with each other's request processing:

```bash
# Create files with unique content
mkdir -p test_content
for i in {1..10}; do
    echo "Unique content for file $i - $(date)" > "test_content/file$i.txt"
done

# Request all files simultaneously
for i in {1..10}; do
    curl -s "http://localhost:8080/file$i.txt" > "response$i.txt" &
done
wait

# Verify each response matches its expected content
for i in {1..10}; do
    if diff "test_content/file$i.txt" "response$i.txt" > /dev/null; then
        echo "✓ File $i content correct"
    else
        echo "✗ File $i content corrupted or swapped"
    fi
done
```

#### Thread Pool Model Testing

The thread pool model maintains a fixed number of worker threads that handle connections from a queue, requiring different testing approaches focused on queue management and worker coordination.

**Thread Pool Size Verification:**

Test that the server maintains the configured number of worker threads:

```bash
# Server should maintain exactly N worker threads plus main thread
EXPECTED_THREADS=$((THREAD_POOL_SIZE + 1))
ACTUAL_THREADS=$(ps -T -p $(pgrep http_server) | wc -l)

if [ $ACTUAL_THREADS -eq $EXPECTED_THREADS ]; then
    echo "✓ Thread pool size correct: $ACTUAL_THREADS threads"
else
    echo "✗ Thread pool size incorrect: expected $EXPECTED_THREADS, got $ACTUAL_THREADS"
fi
```

**Queue Overflow Testing:**

Test server behavior when more connections arrive than the thread pool can handle immediately:

```bash
# Create more simultaneous connections than thread pool size
CONNECTIONS=$((THREAD_POOL_SIZE * 3))

echo "Creating $CONNECTIONS simultaneous connections..."
for i in $(seq 1 $CONNECTIONS); do
    (
        # Add delay to keep connections active
        curl -s --max-time 30 "http://localhost:8080/?delay=5" > "pool_response_$i" &
        echo $! > "pool_pid_$i"
    ) &
done

# Wait for all connections
wait

# Verify all connections completed successfully
SUCCESS_COUNT=0
for i in $(seq 1 $CONNECTIONS); do
    if [ -s "pool_response_$i" ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    fi
done

echo "Successful connections: $SUCCESS_COUNT/$CONNECTIONS"
```

**Worker Thread Load Distribution:**

Verify that work is distributed among thread pool workers rather than handled by a single thread:

```bash
# Enable thread ID logging in server (if available)
# Send requests and check logs for thread distribution

for i in {1..20}; do
    curl -s "http://localhost:8080/worker-test-$i" > /dev/null
done

# Analyze server logs to verify multiple thread IDs handled requests
# This requires the server to log thread IDs for verification
```

#### Event-Driven Model Testing

The event-driven (select/poll) model uses a single thread with non-blocking I/O to handle multiple connections, requiring specialized testing to verify proper multiplexing behavior.

**Non-Blocking Behavior Verification:**

Test that the server doesn't block on slow clients:

```bash
# Create a slow client that sends data gradually
{
    echo "GET / HTTP/1.1"
    sleep 5
    echo "Host: localhost"
    sleep 5
    echo ""
    echo ""
} | nc localhost 8080 &

SLOW_PID=$!

# While slow client is active, send fast requests
for i in {1..5}; do
    curl -s http://localhost:8080/ > "fast_response_$i" &
done
wait

# Verify fast requests completed despite slow client
kill $SLOW_PID 2>/dev/null
```

**Connection Multiplexing Testing:**

Verify that a single thread can handle multiple simultaneous connections:

```bash
# Verify server uses only one main thread
THREAD_COUNT=$(ps -T -p $(pgrep http_server) | wc -l)
if [ $THREAD_COUNT -gt 2 ]; then  # Main thread + perhaps one helper
    echo "✗ Event-driven server should not create multiple threads"
else
    echo "✓ Server using event-driven single-thread model"
fi

# Create many simultaneous connections
for i in {1..50}; do
    (echo -e "GET /test$i HTTP/1.1\r\nHost: localhost\r\n\r\n" | nc localhost 8080 > "event_response_$i") &
done
wait

# Count successful responses
SUCCESS_COUNT=$(ls event_response_* 2>/dev/null | wc -l)
echo "Event-driven model handled $SUCCESS_COUNT/50 connections"
```

#### Load Testing and Stress Testing

**Progressive Load Testing:**

Gradually increase the load to identify the server's capacity limits:

```bash
#!/bin/bash
# progressive_load_test.sh

test_load() {
    local connections=$1
    local duration=10
    local success=0
    
    echo "Testing with $connections concurrent connections..."
    
    for i in $(seq 1 $connections); do
        timeout $duration curl -s http://localhost:8080/ > /dev/null && success=$((success + 1)) &
    done
    wait
    
    local success_rate=$((success * 100 / connections))
    echo "Success rate: $success_rate% ($success/$connections)"
    
    return $success_rate
}

# Test increasing loads
for load in 10 25 50 100 200 500; do
    test_load $load
    sleep 5  # Recovery time
done
```

**Memory and Resource Monitoring:**

Monitor server resource usage during concurrent load:

```bash
# Start resource monitoring
{
    while true; do
        echo "$(date): $(ps -o pid,vsz,rss,pcpu,pmem,nlwp -p $(pgrep http_server))"
        sleep 1
    done
} > resource_usage.log &

MONITOR_PID=$!

# Generate load
for i in {1..100}; do
    curl -s http://localhost:8080/large-file.html > /dev/null &
done
wait

# Stop monitoring
kill $MONITOR_PID

# Analyze resource usage
echo "Peak memory usage:"
sort -k3 -nr resource_usage.log | head -1
```

#### Failure Recovery Testing

**Connection Interruption Handling:**

Test server resilience when clients disconnect unexpectedly:

```bash
# Create connections that disconnect abruptly
for i in {1..10}; do
    {
        echo "GET / HTTP/1.1"
        echo "Host: localhost"
        # Disconnect without completing request
    } | timeout 0.1 nc localhost 8080 &
done

# Immediately send normal requests
sleep 0.5
for i in {1..5}; do
    curl -s http://localhost:8080/ > "recovery_test_$i" &
done
wait

# Verify normal requests succeeded despite connection failures
SUCCESS=0
for i in {1..5}; do
    if [ -s "recovery_test_$i" ]; then
        SUCCESS=$((SUCCESS + 1))
    fi
done

echo "Recovery test: $SUCCESS/5 requests succeeded"
```

**Graceful Shutdown Testing:**

Test that the server handles shutdown signals correctly while serving active connections:

```bash
# Start long-running requests
for i in {1..5}; do
    curl -s "http://localhost:8080/?delay=10" > "shutdown_test_$i" &
done

sleep 2  # Let requests start processing

# Send shutdown signal
kill -TERM $(pgrep http_server)

# Wait for requests to complete
wait

# Verify existing requests completed successfully
COMPLETED=0
for i in {1..5}; do
    if [ -s "shutdown_test_$i" ]; then
        COMPLETED=$((COMPLETED + 1))
    fi
done

echo "Graceful shutdown: $COMPLETED/5 requests completed"
```

### Implementation Guidance

The testing strategy implementation requires creating a comprehensive test suite that can be executed at each milestone to verify functionality and catch regressions. The testing approach combines automated scripts with manual verification procedures to ensure thorough coverage.

#### Technology Recommendations

| Testing Component | Simple Option | Advanced Option |
|---|---|---|
| HTTP Client Testing | curl command-line scripts | Python requests library with unittest |
| Load Testing | Bash scripts with background processes | Apache Bench (ab) or wrk load tester |
| Resource Monitoring | ps/lsof command-line tools | Custom monitoring with htop/iotop |
| Automated Testing | Shell scripts with basic assertions | Python pytest with HTTP test fixtures |
| Concurrency Testing | Manual curl with background jobs | Custom multi-threaded test client |

#### Recommended File Structure

```
project-root/
  tests/
    milestone1/
      test_tcp_basics.sh           ← TCP socket functionality
      test_connection_handling.sh  ← Connection accept/close
    milestone2/
      test_http_parsing.sh         ← HTTP request parsing
      test_error_responses.sh      ← Malformed request handling
      sample_requests/             ← Test HTTP request files
    milestone3/
      test_file_serving.sh         ← Static file serving
      test_security.sh             ← Directory traversal tests
      test_content/                ← Test files and directories
        index.html
        style.css
        images/
          test.png
    milestone4/
      test_concurrency.sh          ← Concurrent connection tests
      test_load.sh                 ← Load testing scripts
      test_resource_limits.sh      ← Resource management tests
    tools/
      http_test_client.c           ← Custom test client
      monitor_resources.sh         ← Resource monitoring utilities
      test_runner.sh               ← Master test execution script
  test_results/                    ← Test output and logs
    milestone1_results.txt
    milestone2_results.txt
    resource_usage.log
```

#### Infrastructure Testing Code

**Complete HTTP Test Client Implementation:**

```c
// tools/http_test_client.c - Complete HTTP testing client
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>
#include <pthread.h>

#define MAX_RESPONSE_SIZE 8192
#define MAX_CONCURRENT_CLIENTS 100

typedef struct {
    char* server_host;
    int server_port;
    char* request_path;
    int client_id;
    int* success_count;
    pthread_mutex_t* count_mutex;
} ClientConfig;

// Send HTTP request and receive response
int send_http_request(const char* host, int port, const char* path, char* response, size_t max_response) {
    int sockfd;
    struct sockaddr_in server_addr;
    char request[1024];
    ssize_t bytes_received;
    
    // Create socket
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("socket creation failed");
        return -1;
    }
    
    // Configure server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host, &server_addr.sin_addr) <= 0) {
        perror("invalid address");
        close(sockfd);
        return -1;
    }
    
    // Connect to server
    if (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("connection failed");
        close(sockfd);
        return -1;
    }
    
    // Send HTTP request
    snprintf(request, sizeof(request), 
             "GET %s HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n\r\n", 
             path, host);
    
    if (send(sockfd, request, strlen(request), 0) < 0) {
        perror("send failed");
        close(sockfd);
        return -1;
    }
    
    // Receive response
    bytes_received = recv(sockfd, response, max_response - 1, 0);
    if (bytes_received < 0) {
        perror("recv failed");
        close(sockfd);
        return -1;
    }
    
    response[bytes_received] = '\0';
    close(sockfd);
    return bytes_received;
}

// Thread function for concurrent testing
void* client_thread(void* arg) {
    ClientConfig* config = (ClientConfig*)arg;
    char response[MAX_RESPONSE_SIZE];
    
    int result = send_http_request(config->server_host, config->server_port, 
                                   config->request_path, response, MAX_RESPONSE_SIZE);
    
    if (result > 0) {
        pthread_mutex_lock(config->count_mutex);
        (*config->success_count)++;
        pthread_mutex_unlock(config->count_mutex);
        printf("Client %d: SUCCESS (%d bytes)\n", config->client_id, result);
    } else {
        printf("Client %d: FAILED\n", config->client_id);
    }
    
    return NULL;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <host> <port> <path> [concurrent_clients]\n", argv[0]);
        return 1;
    }
    
    char* host = argv[1];
    int port = atoi(argv[2]);
    char* path = argv[3];
    int concurrent_clients = (argc > 4) ? atoi(argv[4]) : 1;
    
    if (concurrent_clients == 1) {
        // Single request mode
        char response[MAX_RESPONSE_SIZE];
        int result = send_http_request(host, port, path, response, MAX_RESPONSE_SIZE);
        
        if (result > 0) {
            printf("Response received (%d bytes):\n", result);
            printf("%s\n", response);
            return 0;
        } else {
            printf("Request failed\n");
            return 1;
        }
    } else {
        // Concurrent testing mode
        pthread_t threads[MAX_CONCURRENT_CLIENTS];
        ClientConfig configs[MAX_CONCURRENT_CLIENTS];
        int success_count = 0;
        pthread_mutex_t count_mutex = PTHREAD_MUTEX_INITIALIZER;
        
        if (concurrent_clients > MAX_CONCURRENT_CLIENTS) {
            concurrent_clients = MAX_CONCURRENT_CLIENTS;
        }
        
        printf("Starting %d concurrent clients...\n", concurrent_clients);
        
        // Start concurrent clients
        for (int i = 0; i < concurrent_clients; i++) {
            configs[i].server_host = host;
            configs[i].server_port = port;
            configs[i].request_path = path;
            configs[i].client_id = i + 1;
            configs[i].success_count = &success_count;
            configs[i].count_mutex = &count_mutex;
            
            if (pthread_create(&threads[i], NULL, client_thread, &configs[i]) != 0) {
                perror("pthread_create failed");
                return 1;
            }
        }
        
        // Wait for all threads to complete
        for (int i = 0; i < concurrent_clients; i++) {
            pthread_join(threads[i], NULL);
        }
        
        printf("Concurrent test completed: %d/%d clients succeeded\n", 
               success_count, concurrent_clients);
        
        return (success_count == concurrent_clients) ? 0 : 1;
    }
}
```

**Complete Resource Monitoring Script:**

```bash
#!/bin/bash
# tools/monitor_resources.sh - Complete resource monitoring utility

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../test_results"
SERVER_NAME="http_server"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Get server process ID
get_server_pid() {
    pgrep -f "$SERVER_NAME" | head -1
}

# Monitor server resources
monitor_resources() {
    local duration=$1
    local interval=${2:-1}
    local output_file="$LOG_DIR/resource_monitor_$(date +%Y%m%d_%H%M%S).log"
    
    echo "Monitoring server resources for ${duration}s (interval: ${interval}s)"
    echo "Output: $output_file"
    echo "Timestamp,PID,VSZ_KB,RSS_KB,CPU_PCT,MEM_PCT,THREADS,FD_COUNT" > "$output_file"
    
    local start_time=$(date +%s)
    local end_time=$((start_time + duration))
    
    while [ $(date +%s) -lt $end_time ]; do
        local server_pid=$(get_server_pid)
        
        if [ -n "$server_pid" ]; then
            # Get process statistics
            local ps_info=$(ps -o pid,vsz,rss,pcpu,pmem,nlwp -p "$server_pid" --no-headers 2>/dev/null)
            
            if [ -n "$ps_info" ]; then
                # Count file descriptors
                local fd_count=$(ls /proc/$server_pid/fd 2>/dev/null | wc -l)
                
                # Format output
                local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
                echo "$timestamp,$ps_info,$fd_count" | tr -s ' ' ',' >> "$output_file"
            else
                echo "$(date '+%Y-%m-%d %H:%M:%S'),SERVER_NOT_FOUND,,,,,," >> "$output_file"
            fi
        else
            echo "$(date '+%Y-%m-%d %H:%M:%S'),PID_NOT_FOUND,,,,,," >> "$output_file"
        fi
        
        sleep "$interval"
    done
    
    echo "Monitoring completed. Results saved to: $output_file"
    
    # Generate summary
    if [ -s "$output_file" ]; then
        echo "Resource Usage Summary:"
        echo "======================"
        echo "Peak VSZ (KB): $(tail -n +2 "$output_file" | cut -d',' -f3 | sort -nr | head -1)"
        echo "Peak RSS (KB): $(tail -n +2 "$output_file" | cut -d',' -f4 | sort -nr | head -1)"
        echo "Peak Threads: $(tail -n +2 "$output_file" | cut -d',' -f6 | sort -nr | head -1)"
        echo "Peak FDs: $(tail -n +2 "$output_file" | cut -d',' -f7 | sort -nr | head -1)"
    fi
}

# Test connection limits
test_connection_limits() {
    local max_connections=$1
    local server_pid=$(get_server_pid)
    
    if [ -z "$server_pid" ]; then
        echo "Server not running"
        return 1
    fi
    
    echo "Testing connection limits (max: $max_connections)"
    
    # Start monitoring
    monitor_resources 60 1 &
    local monitor_pid=$!
    
    # Create connections gradually
    local success_count=0
    for i in $(seq 1 $max_connections); do
        if timeout 10 "$SCRIPT_DIR/../tools/http_test_client" localhost 8080 / >/dev/null 2>&1; then
            success_count=$((success_count + 1))
            echo -n "."
        else
            echo -n "x"
        fi
        
        # Brief delay between connections
        sleep 0.1
    done
    
    echo ""
    echo "Connection test completed: $success_count/$max_connections successful"
    
    # Stop monitoring
    kill $monitor_pid 2>/dev/null
    wait $monitor_pid 2>/dev/null
}

# Main function
main() {
    case "${1:-help}" in
        "monitor")
            monitor_resources "${2:-30}" "${3:-1}"
            ;;
        "limits")
            test_connection_limits "${2:-100}"
            ;;
        "help"|*)
            echo "Usage: $0 <command> [options]"
            echo "Commands:"
            echo "  monitor <duration> [interval]  - Monitor resources for duration seconds"
            echo "  limits <max_connections>       - Test connection limits"
            echo "  help                          - Show this help"
            ;;
    esac
}

main "$@"
```

#### Core Testing Logic Skeletons

**Master Test Runner:**

```bash
#!/bin/bash
# tests/test_runner.sh - Master test execution script

# TODO 1: Set up test environment - create test directories, compile test tools
# TODO 2: Start server in background and wait for it to be ready
# TODO 3: Execute milestone 1 tests - TCP server basics
# TODO 4: Execute milestone 2 tests - HTTP request parsing  
# TODO 5: Execute milestone 3 tests - static file serving
# TODO 6: Execute milestone 4 tests - concurrent connections
# TODO 7: Run load testing and resource monitoring
# TODO 8: Generate comprehensive test report
# TODO 9: Clean up test environment and stop server
# TODO 10: Return appropriate exit code based on test results

run_milestone_tests() {
    local milestone=$1
    # TODO: Execute all tests for specified milestone
    # TODO: Collect results and update overall test status
    # TODO: Log detailed results to milestone-specific log file
}

setup_test_environment() {
    # TODO: Create necessary test directories and files
    # TODO: Compile custom test clients and utilities
    # TODO: Verify required testing tools are available
}

cleanup_test_environment() {
    # TODO: Remove temporary test files
    # TODO: Kill any remaining background processes
    # TODO: Restore original system state
}

generate_test_report() {
    # TODO: Combine results from all milestone tests
    # TODO: Create HTML or text report with pass/fail summary
    # TODO: Include performance metrics and resource usage
}
```

#### Milestone Checkpoints

**Milestone 1 Verification Script:**

```bash
#!/bin/bash
# tests/milestone1/test_tcp_basics.sh

echo "=== Milestone 1: TCP Server Basics Testing ==="

# Test 1: Server starts and binds to port
test_server_startup() {
    echo "Testing server startup and port binding..."
    
    # TODO: Start server in background
    # TODO: Wait for server to be ready (check port binding)
    # TODO: Verify server is listening on correct port using netstat
    # TODO: Test that server accepts basic TCP connections
    # Expected: Server binds to port 8080 and shows LISTEN state
}

# Test 2: Connection acceptance
test_connection_acceptance() {
    echo "Testing connection acceptance..."
    
    # TODO: Use telnet to connect to server
    # TODO: Verify connection is accepted (telnet shows "Connected")
    # TODO: Send simple string and verify server reads it
    # TODO: Verify server sends hardcoded HTTP response
    # Expected: Connection accepted, response received, connection closed
}
```

**Milestone 4 Concurrent Testing:**

```bash
#!/bin/bash  
# tests/milestone4/test_concurrency.sh

# Test concurrent request handling
test_concurrent_requests() {
    echo "Testing concurrent request handling..."
    
    # TODO: Create multiple background curl processes
    # TODO: Monitor that all requests complete successfully
    # TODO: Verify response content is consistent across all requests
    # TODO: Check that no responses are corrupted or mixed up
    # Expected: All concurrent requests succeed with correct content
}

# Test resource limits
test_resource_limits() {
    echo "Testing resource limits..."
    
    # TODO: Create connections exceeding configured limits
    # TODO: Verify server handles excess connections gracefully
    # TODO: Monitor file descriptor and thread usage
    # TODO: Ensure no resource leaks after connections close
    # Expected: Server enforces limits without crashing
}

```


## Debugging Guide

> **Milestone(s):** All Milestones 1-4 - provides systematic troubleshooting for TCP server basics, HTTP request parsing, static file serving, and concurrent connections with specific diagnostic techniques

### Debugging Mental Model: The Detective's Toolkit

Think of debugging an HTTP server like being a detective investigating a crime scene. Just as detectives use systematic methods to gather evidence, interview witnesses, and piece together what happened, server debugging requires a methodical approach to collecting symptoms, examining evidence, and forming hypotheses. The server's logs are like witness statements, network packet captures are like security footage, and debugger traces are like forensic analysis. A good detective doesn't jump to conclusions but follows the evidence methodically, and a good server debugger applies the same systematic thinking to isolate and resolve issues.

The debugging process for HTTP servers involves three critical phases: symptom identification (what is the observable behavior), evidence collection (gathering diagnostic data from multiple sources), and root cause analysis (correlating evidence to identify the underlying problem). Just as a detective considers multiple theories and tests each one against the evidence, effective server debugging requires considering multiple potential causes and systematically eliminating possibilities through targeted investigation.

### Symptom-Based Diagnosis Table

The following comprehensive diagnosis table organizes common HTTP server issues by observable symptoms, providing a systematic approach to troubleshooting. Each symptom includes multiple potential causes because server problems often manifest in similar ways but have different underlying roots.

| **Symptom** | **Likely Causes** | **Diagnostic Techniques** | **Investigation Commands** |
|-------------|------------------|--------------------------|---------------------------|
| **Server fails to start** | Port already in use; Permission denied for port binding; Invalid configuration values; Missing document root directory | Check port availability; Verify user permissions; Validate configuration; Check filesystem permissions | `netstat -tlnp \| grep 8080`; `sudo lsof -i :8080`; `ls -la /path/to/document/root`; `id` (check user) |
| **Clients cannot connect** | Firewall blocking connections; Server not listening on correct interface; DNS resolution issues; Network connectivity problems | Verify server listening state; Test local vs remote connections; Check firewall rules; Test network path | `ss -tlnp \| grep :8080`; `telnet localhost 8080`; `curl -v http://server:8080/`; `iptables -L` |
| **Connection accepted but no response** | Server hanging in request reading; Infinite loop in request processing; Deadlock in thread synchronization; Process crashed after accept | Monitor server process state; Check thread states; Examine system resources; Look for core dumps | `ps aux \| grep server`; `top -H -p <pid>`; `strace -p <pid>`; `gdb -p <pid>` |
| **Request parsing fails** | Malformed HTTP request format; Line ending issues (CRLF vs LF); Buffer overflow in parsing; Invalid characters in headers | Capture raw request data; Examine line endings; Check buffer boundaries; Validate character encoding | `tcpdump -i lo -A port 8080`; `xxd request.txt`; `strace -e read,write -p <pid>` |
| **File not found errors** | Incorrect path mapping; Directory traversal protection; Case sensitivity issues; Permission denied access | Trace path resolution; Check filesystem permissions; Verify document root mapping; Test file accessibility | `ls -la /document/root/requested/path`; `sudo -u www-data cat /path/to/file`; `realpath /document/root/../etc/passwd` |
| **Slow response times** | Disk I/O bottleneck; Thread pool exhaustion; Memory allocation delays; Network congestion | Monitor system performance; Check thread utilization; Profile memory usage; Measure network latency | `iostat -x 1`; `sar -u 1`; `valgrind --tool=massif ./server`; `iftop` |
| **High CPU usage** | Infinite loop in request processing; Excessive string operations; Inefficient parsing algorithms; Busy-wait conditions | Profile CPU usage; Trace function calls; Examine parsing logic; Check synchronization primitives | `perf top -p <pid>`; `strace -c -p <pid>`; `gdb -p <pid> -ex bt` |
| **Memory leaks** | Missing `cleanup_http_request()` calls; Unreleased connection contexts; Thread resource leaks; File descriptor leaks | Track memory allocation; Monitor file descriptor count; Check cleanup sequences; Analyze heap growth | `valgrind --leak-check=full ./server`; `lsof -p <pid> \| wc -l`; `pmap <pid>` |
| **Concurrent connection issues** | Race conditions in shared data; Thread pool deadlock; Resource limit exceeded; Improper thread synchronization | Test under load; Check thread states; Monitor resource limits; Examine synchronization code | `siege -c 100 -t 30s http://localhost:8080/`; `ulimit -a`; `ps -eLf \| grep server` |
| **Partial responses** | Early connection termination; Buffer size limitations; Network timeout issues; Incomplete file reads | Monitor connection lifecycle; Check buffer sizes; Measure transfer times; Verify file integrity | `curl -v --max-time 30 http://localhost:8080/large.file`; `nc localhost 8080 < request.txt` |
| **Security vulnerabilities** | Directory traversal attacks; Path injection; Buffer overflow; Insufficient input validation | Test path traversal; Inject malformed input; Check boundary conditions; Verify input sanitization | `curl "http://localhost:8080/../../../etc/passwd"`; `python -c "print('A'*10000)" \| nc localhost 8080` |

> **Critical Insight**: Many HTTP server issues present with similar symptoms but have completely different root causes. Systematic evidence collection prevents misdiagnosis and reduces debugging time significantly. Always start with the simplest possible cause before investigating complex scenarios.

### Debugging Tools and Techniques

Effective HTTP server debugging requires a multi-layered approach using different tools to examine behavior at the network, system, and application levels. Each tool provides a different perspective on server operation, and combining insights from multiple tools leads to faster problem resolution.

#### Network-Level Debugging

Network-level debugging examines the raw TCP connections and HTTP message flow between clients and servers. This level reveals issues with connection establishment, protocol compliance, and message formatting that may not be visible at higher abstraction levels.

**Packet Capture and Analysis:**
The `tcpdump` utility captures raw network traffic, allowing examination of actual bytes transmitted between client and server. This technique reveals protocol violations, malformed messages, and connection timing issues that application-level logging might miss.

| **Tool** | **Purpose** | **Key Commands** | **What to Look For** |
|----------|-------------|-----------------|----------------------|
| `tcpdump` | Capture raw network packets | `tcpdump -i lo -A -s 0 port 8080` | Connection establishment, HTTP message format, connection termination |
| `wireshark` | GUI packet analysis | `wireshark -i lo -f "port 8080"` | Protocol violations, timing issues, malformed headers |
| `netcat` | Manual connection testing | `nc localhost 8080` then type HTTP request | Server response to hand-crafted requests |
| `telnet` | Basic connectivity testing | `telnet localhost 8080` | Whether TCP connection can be established |

**Connection State Monitoring:**
Understanding the current state of TCP connections helps identify stuck connections, resource leaks, and capacity issues. Different connection states indicate different phases of the request-response cycle.

```bash
# Monitor active connections and their states
ss -tuln | grep :8080                    # Check if server is listening
ss -tpn | grep :8080                     # Show active connections with process info
netstat -an | grep :8080 | grep ESTABLISHED  # Count active client connections
lsof -i TCP:8080                         # Show processes using port 8080
```

#### System-Level Debugging

System-level debugging examines resource usage, process behavior, and operating system interactions. This perspective reveals issues with memory management, file descriptor usage, and system call patterns that impact server performance and stability.

**Process and Thread Monitoring:**
Understanding how the server process and its threads behave under different conditions helps identify concurrency issues, resource exhaustion, and performance bottlenecks.

| **Tool** | **Purpose** | **Key Commands** | **Diagnostic Information** |
|----------|-------------|-----------------|---------------------------|
| `ps` | Process state monitoring | `ps -eLf | grep server` | Thread count, CPU usage, memory usage |
| `top` | Real-time resource monitoring | `top -H -p <pid>` | Per-thread CPU usage, memory growth |
| `strace` | System call tracing | `strace -p <pid> -e trace=network,file` | Network operations, file access patterns |
| `ltrace` | Library call tracing | `ltrace -p <pid> -e malloc,free` | Memory allocation patterns |
| `pmap` | Memory mapping analysis | `pmap -x <pid>` | Memory layout, shared libraries |

**Resource Limit Investigation:**
System resource limits can cause mysterious server failures that don't generate clear error messages. Checking both soft and hard limits helps identify capacity constraints.

```bash
# Check resource limits for server process
cat /proc/<pid>/limits                   # Current limits for process
ulimit -a                               # Limits for current shell
sysctl fs.file-max                      # System-wide file descriptor limit
cat /proc/sys/net/core/somaxconn        # TCP listen backlog limit
```

#### Application-Level Debugging

Application-level debugging uses debuggers and profiling tools to examine the server's internal state, variable values, and execution flow. This level provides the most detailed view of program behavior but requires understanding of the server's implementation.

**Interactive Debugging with GDB:**
The GNU Debugger allows real-time examination of server state, including variable values, call stacks, and execution flow. This technique is essential for investigating deadlocks, infinite loops, and logic errors.

| **GDB Command** | **Purpose** | **When to Use** |
|-----------------|-------------|-----------------|
| `gdb -p <pid>` | Attach to running server | Server appears hung or behaving unexpectedly |
| `bt` | Show call stack | Identify where server is stuck |
| `thread apply all bt` | Show all thread call stacks | Investigate deadlocks and thread states |
| `p variable_name` | Print variable value | Check request parsing state, connection context |
| `info threads` | List all threads | See thread count and current activity |

**Memory Debugging with Valgrind:**
Valgrind detects memory leaks, buffer overflows, and invalid memory access patterns that can cause crashes or security vulnerabilities. Running the server under Valgrind during testing reveals memory management issues.

```bash
# Run server with comprehensive memory checking
valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         --verbose \
         ./http_server --port 8080 --root /var/www

# Monitor for specific error types
valgrind --tool=memcheck \
         --track-fds=yes \
         ./http_server  # Track file descriptor leaks
```

**Performance Profiling:**
Understanding where the server spends its execution time helps identify performance bottlenecks and optimization opportunities.

| **Tool** | **Purpose** | **Usage** | **Output Analysis** |
|----------|-------------|-----------|-------------------|
| `perf` | CPU profiling | `perf record -p <pid>; perf report` | Function call frequency, CPU hotspots |
| `gprof` | Function profiling | Compile with `-pg`, run server, analyze `gmon.out` | Function execution time, call graph |
| `time` | Basic timing | `time ./http_server` | Overall execution time breakdown |

### Milestone-Specific Issues

Each milestone introduces specific types of problems based on the functionality being implemented. Understanding common issues for each milestone helps focus debugging efforts and provides targeted solutions.

#### Milestone 1: TCP Server Basics Issues

The TCP server foundation presents networking-specific challenges related to socket programming, connection management, and basic protocol handling. These issues often manifest as connection failures or server startup problems.

**Socket Creation and Binding Problems:**

⚠️ **Pitfall: Port Already in Use**
When the server fails to start with "Address already in use" errors, another process is likely using the target port. This commonly happens when a previous server instance didn't terminate cleanly or when system services are using standard ports.

| **Symptom** | **Cause** | **Diagnosis** | **Solution** |
|-------------|-----------|---------------|-------------|
| `bind()` fails with `EADDRINUSE` | Another process using port | `netstat -tlnp \| grep 8080` | Kill existing process or use different port |
| Server starts but clients can't connect | Binding to localhost only | `ss -tlnp \| grep 8080` | Bind to `INADDR_ANY` instead of localhost |
| Permission denied on port binding | Non-root user binding privileged port | `id; echo $PORT` | Use port > 1024 or run with sudo |

⚠️ **Pitfall: Incorrect Socket Options**
Forgetting to set `SO_REUSEADDR` causes server restart failures when connections are in TIME_WAIT state. This is especially problematic during development when restarting the server frequently.

```c
// Missing socket option setup
int server_fd = socket(AF_INET, SOCK_STREAM, 0);
// bind() will fail if previous connections are in TIME_WAIT

// Correct approach with socket options
int server_fd = socket(AF_INET, SOCK_STREAM, 0);
int opt = 1;
setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
```

**Connection Acceptance Issues:**

⚠️ **Pitfall: Blocking Accept Loop**
Using a simple `accept()` loop without proper signal handling or shutdown logic can create servers that cannot be stopped gracefully. The server continues accepting new connections even when shutdown is requested.

| **Problem** | **Symptom** | **Investigation** | **Fix** |
|-------------|-------------|-------------------|---------|
| Server cannot be stopped | Process ignores SIGTERM | `kill -TERM <pid>; ps aux \| grep server` | Add signal handler for graceful shutdown |
| Accept loop consumes 100% CPU | Server spins without blocking | `strace -p <pid>` shows rapid `accept()` calls | Check `accept()` error handling |
| Client connections dropped | `accept()` failing silently | `strace -e accept -p <pid>` | Add error logging for failed accepts |

**File Descriptor Management:**

⚠️ **Pitfall: File Descriptor Leaks**
Failing to close client sockets after processing requests leads to file descriptor exhaustion. This typically manifests as the server stopping accepting new connections after handling a certain number of requests.

```bash
# Monitor file descriptor usage
watch "lsof -p <server_pid> | wc -l"     # Total FDs in use
ls /proc/<pid>/fd/ | wc -l               # Count open file descriptors
cat /proc/<pid>/limits | grep "Max open files"  # Check FD limit
```

#### Milestone 2: HTTP Request Parsing Issues

HTTP parsing introduces complexity around message format compliance, buffer management, and protocol error handling. These issues often manifest as parsing failures or protocol violations.

**Request Line Parsing Problems:**

⚠️ **Pitfall: Line Ending Confusion**
HTTP specifies CRLF (`\r\n`) line endings, but clients sometimes send only LF (`\n`). Parsers that expect exact CRLF sequences fail on otherwise valid requests.

| **Symptom** | **Client Type** | **Line Ending** | **Parser Behavior** |
|-------------|-----------------|-----------------|-------------------|
| Parsing fails on curl requests | curl on Unix | LF only | `find_line_end()` doesn't find CRLF |
| Browser requests fail | Some web browsers | CRLF | Parser works correctly |
| Telnet testing fails | Manual telnet session | Mixed endings | Inconsistent parsing results |

**Buffer Overflow Protection:**

⚠️ **Pitfall: Unbounded Buffer Reads**
Reading request data into fixed-size buffers without bounds checking can cause buffer overflows when clients send large requests or malicious input designed to exploit buffer boundaries.

```c
// Dangerous approach - no bounds checking
char buffer[1024];
read(client_fd, buffer, 2048);  // Can overflow buffer!

// Safe approach with bounds checking
char buffer[MAX_REQUEST_SIZE];
ssize_t bytes_read = read_complete_request(client_fd, buffer, sizeof(buffer));
if (bytes_read >= sizeof(buffer)) {
    // Request too large - send 413 Entity Too Large
}
```

**Header Parsing Edge Cases:**

⚠️ **Pitfall: Header Value Whitespace**
HTTP headers can have leading and trailing whitespace around values that must be trimmed. Failing to handle this correctly causes header matching failures and incorrect behavior.

| **Header Format** | **Raw Value** | **Trimmed Value** | **Impact if Not Trimmed** |
|-------------------|---------------|-------------------|---------------------------|
| `Host: localhost:8080` | `localhost:8080` | `localhost:8080` | No issue |
| `Host:   localhost:8080   ` | `  localhost:8080   ` | `localhost:8080` | Host matching fails |
| `Content-Type: text/html` | ` text/html` | `text/html` | MIME type detection fails |

#### Milestone 3: Static File Serving Issues

File serving introduces security concerns, filesystem interaction complexity, and content type handling challenges. These issues often manifest as security vulnerabilities or incorrect content delivery.

**Path Traversal Security:**

⚠️ **Pitfall: Directory Traversal Attacks**
Failing to properly validate and canonicalize request paths allows attackers to access files outside the document root using `../` sequences. This is a critical security vulnerability.

| **Attack Vector** | **Request Path** | **Resolved Path** | **Security Impact** |
|-------------------|------------------|-------------------|-------------------|
| Basic traversal | `/../../../etc/passwd` | `/etc/passwd` | System file access |
| URL encoded | `/%2E%2E/%2E%2E/etc/passwd` | `/../etc/passwd` | Bypass basic filters |
| Mixed encoding | `/..%2F..%2Fetc%2Fpasswd` | `/../etc/passwd` | Bypass simple validation |

```c
// Proper path validation sequence
int validate_and_resolve_path(const char* document_root, 
                             const char* request_path, 
                             char* resolved_path, 
                             size_t max_path_len) {
    // 1. URL decode the request path
    // 2. Normalize path (remove . and .. components)
    // 3. Prepend document root
    // 4. Canonicalize using realpath()
    // 5. Verify result stays within document root
}
```

**MIME Type Detection Issues:**

⚠️ **Pitfall: Incorrect Content-Type Headers**
Sending files without proper `Content-Type` headers causes browsers to misinterpret content, leading to security issues (MIME sniffing attacks) or display problems.

| **File Extension** | **MIME Type** | **Browser Behavior Without Correct Type** |
|--------------------|---------------|-------------------------------------------|
| `.html` | `text/html` | May display as plain text |
| `.css` | `text/css` | Stylesheet not applied |
| `.js` | `application/javascript` | Script not executed |
| `.pdf` | `application/pdf` | Downloaded instead of displayed |
| `.jpg` | `image/jpeg` | May not display inline |

**File Permission and Access Issues:**

⚠️ **Pitfall: Permission Denied Errors**
Server processes running under restricted user accounts may not have read permissions for files in the document root. This causes 403 Forbidden errors instead of 404 Not Found errors.

```bash
# Check file permissions for server user
sudo -u www-data ls -la /var/www/html/requested-file.txt
sudo -u www-data cat /var/www/html/requested-file.txt

# Fix permission issues
sudo chown -R www-data:www-data /var/www/html/
sudo chmod -R 644 /var/www/html/*.txt
sudo chmod -R 755 /var/www/html/  # Directories need execute permission
```

#### Milestone 4: Concurrent Connections Issues

Concurrency introduces complex issues around thread safety, resource contention, and synchronization. These problems often appear only under load or in specific timing conditions.

**Thread Safety Violations:**

⚠️ **Pitfall: Shared Data Race Conditions**
Multiple threads accessing shared data structures without proper synchronization leads to data corruption, crashes, and unpredictable behavior that only appears under concurrent load.

| **Shared Resource** | **Race Condition** | **Symptom** | **Solution** |
|--------------------|--------------------|-------------|-------------|
| Connection counter | Increment/decrement without mutex | Incorrect active count | Use atomic operations or mutex |
| Configuration data | Modification during request processing | Inconsistent behavior | Make configuration read-only |
| Log files | Concurrent writes | Interleaved log messages | Use per-thread buffers or mutex |
| File handle cache | Cache updates | File corruption | Synchronize cache operations |

**Thread Pool Exhaustion:**

⚠️ **Pitfall: Uncontrolled Thread Creation**
Creating unlimited threads for incoming connections leads to resource exhaustion and system instability under high load. The system runs out of memory or reaches thread limits.

```c
// Dangerous approach - unlimited thread creation
void handle_connection(int server_fd) {
    int client_fd = accept(server_fd, NULL, NULL);
    pthread_t thread;
    pthread_create(&thread, NULL, handle_client_connection, &client_fd);
    pthread_detach(thread);  // Creates unlimited threads!
}

// Safe approach with connection limits
ConcurrencyManager* manager = init_concurrency_manager(config, THREAD_POOL);
if (get_active_connection_count(manager) >= config->max_connections) {
    send_503_service_unavailable(client_fd);
    close(client_fd);
    return;
}
```

**Deadlock Detection:**

⚠️ **Pitfall: Lock Ordering Deadlocks**
Multiple threads acquiring locks in different orders can create deadlock situations where threads wait for each other indefinitely. This causes the server to hang completely.

| **Thread A** | **Thread B** | **Deadlock Scenario** | **Prevention** |
|--------------|--------------|----------------------|----------------|
| Lock mutex1, then mutex2 | Lock mutex2, then mutex1 | Both threads wait forever | Consistent lock ordering |
| Lock connection_mutex, then stats_mutex | Lock stats_mutex, then connection_mutex | Circular wait condition | Lock hierarchy |

**Resource Cleanup Issues:**

⚠️ **Pitfall: Incomplete Thread Cleanup**
Failing to properly clean up thread resources on connection termination leads to memory leaks and resource exhaustion over time. Resources accumulate until the server becomes unstable.

```bash
# Monitor thread resource usage
ps -eLf | grep server | wc -l           # Count total threads
cat /proc/<pid>/status | grep Threads   # Thread count from kernel
ls /proc/<pid>/task/ | wc -l            # Another way to count threads

# Check for resource leaks over time
watch "ps -o pid,ppid,tid,tty,time,cmd -L -p <pid>"
```

### Implementation Guidance

#### Technology Recommendations Table

| **Component** | **Simple Option** | **Advanced Option** |
|---------------|-------------------|-------------------|
| Debugging Tools | `printf()` debugging + `gdb` | `valgrind` + `perf` + automated testing |
| Log Analysis | `grep` + manual inspection | `logrotate` + structured logging |
| Network Testing | `curl` + `telnet` | `siege` + `ab` (Apache Bench) |
| Memory Debugging | `valgrind --leak-check=simple` | `valgrind --leak-check=full` + heap profiling |
| Performance Testing | Single client testing | Multi-threaded load testing |

#### Recommended File Structure

```
project-root/
  debug/
    debug.h                  ← Debug macros and logging utilities
    debug.c                  ← Debug implementation
    test_client.c           ← Simple client for testing
    load_test.c             ← Concurrent load testing tool
  logs/
    server.log              ← Application log output
    access.log              ← HTTP request logging
    error.log               ← Error-specific logging
  tools/
    monitor.sh              ← System resource monitoring script
    test_suite.sh           ← Automated test runner
    debug_server.sh         ← Script to start server with debugging
```

#### Infrastructure Starter Code

**Debug Logging Utility (`debug/debug.h`):**
```c
#ifndef DEBUG_H
#define DEBUG_H

#include <stdio.h>
#include <time.h>
#include <sys/types.h>

// Debug levels
typedef enum {
    DEBUG_ERROR = 0,
    DEBUG_WARN = 1,
    DEBUG_INFO = 2,
    DEBUG_DEBUG = 3,
    DEBUG_TRACE = 4
} DebugLevel;

// Debug configuration
extern DebugLevel current_debug_level;
extern FILE* debug_output;

// Initialize debugging system
void init_debug_system(const char* log_file, DebugLevel level);
void cleanup_debug_system(void);

// Logging macros with file, line, and timestamp
#define DEBUG_LOG(level, fmt, ...) \
    do { \
        if (level <= current_debug_level) { \
            time_t now = time(NULL); \
            struct tm* tm_info = localtime(&now); \
            fprintf(debug_output, "[%04d-%02d-%02d %02d:%02d:%02d] [%s] %s:%d: " fmt "\n", \
                    tm_info->tm_year + 1900, tm_info->tm_mon + 1, tm_info->tm_mday, \
                    tm_info->tm_hour, tm_info->tm_min, tm_info->tm_sec, \
                    level_to_string(level), __FILE__, __LINE__, ##__VA_ARGS__); \
            fflush(debug_output); \
        } \
    } while(0)

#define ERROR(fmt, ...)   DEBUG_LOG(DEBUG_ERROR, fmt, ##__VA_ARGS__)
#define WARN(fmt, ...)    DEBUG_LOG(DEBUG_WARN, fmt, ##__VA_ARGS__)
#define INFO(fmt, ...)    DEBUG_LOG(DEBUG_INFO, fmt, ##__VA_ARGS__)
#define DEBUG(fmt, ...)   DEBUG_LOG(DEBUG_DEBUG, fmt, ##__VA_ARGS__)
#define TRACE(fmt, ...)   DEBUG_LOG(DEBUG_TRACE, fmt, ##__VA_ARGS__)

// Performance timing helpers
typedef struct {
    struct timespec start_time;
    const char* operation_name;
} PerfTimer;

void start_perf_timer(PerfTimer* timer, const char* operation);
void end_perf_timer(PerfTimer* timer);

// Memory debugging helpers
void* debug_malloc(size_t size, const char* file, int line);
void debug_free(void* ptr, const char* file, int line);

#ifdef DEBUG_MEMORY
#define malloc(size) debug_malloc(size, __FILE__, __LINE__)
#define free(ptr) debug_free(ptr, __FILE__, __LINE__)
#endif

// Network debugging helpers
void dump_http_request(const HTTPRequest* request);
void dump_http_response(const HTTPResponse* response);
void dump_connection_context(const ConnectionContext* context);
void log_connection_event(const ConnectionContext* context, const char* event);

// Resource monitoring
void log_resource_usage(pid_t server_pid);
void check_file_descriptor_leaks(pid_t server_pid);

#endif // DEBUG_H
```

**Test Client Utility (`debug/test_client.c`):**
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <time.h>
#include "debug.h"

typedef struct {
    char server_host[256];
    int server_port;
    char request_path[1024];
    int client_id;
    int* success_count;
    pthread_mutex_t* count_mutex;
} ClientConfig;

// Complete test client implementation for concurrent testing
int send_http_request(const char* host, int port, const char* path, 
                     char* response, size_t max_response) {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        ERROR("Failed to create socket: %s", strerror(errno));
        return -1;
    }

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    
    if (inet_pton(AF_INET, host, &server_addr.sin_addr) <= 0) {
        ERROR("Invalid address: %s", host);
        close(sockfd);
        return -1;
    }

    if (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        ERROR("Connection failed: %s", strerror(errno));
        close(sockfd);
        return -1;
    }

    // Send HTTP request
    char request[1024];
    snprintf(request, sizeof(request),
             "GET %s HTTP/1.1\r\n"
             "Host: %s:%d\r\n"
             "Connection: close\r\n"
             "\r\n",
             path, host, port);

    ssize_t sent = send(sockfd, request, strlen(request), 0);
    if (sent < 0) {
        ERROR("Failed to send request: %s", strerror(errno));
        close(sockfd);
        return -1;
    }

    // Receive response
    ssize_t received = recv(sockfd, response, max_response - 1, 0);
    if (received < 0) {
        ERROR("Failed to receive response: %s", strerror(errno));
        close(sockfd);
        return -1;
    }
    response[received] = '\0';

    close(sockfd);
    return 0;
}

void* client_thread(void* arg) {
    ClientConfig* config = (ClientConfig*)arg;
    char response[MAX_RESPONSE_SIZE];
    
    INFO("Client %d starting request to %s:%d%s", 
         config->client_id, config->server_host, config->server_port, config->request_path);

    int result = send_http_request(config->server_host, config->server_port, 
                                  config->request_path, response, sizeof(response));
    
    if (result == 0) {
        // Check for successful HTTP response
        if (strstr(response, "HTTP/1.1 200") != NULL) {
            pthread_mutex_lock(config->count_mutex);
            (*config->success_count)++;
            pthread_mutex_unlock(config->count_mutex);
            INFO("Client %d received successful response", config->client_id);
        } else {
            WARN("Client %d received error response: %.100s", config->client_id, response);
        }
    } else {
        ERROR("Client %d failed to send request", config->client_id);
    }

    return NULL;
}

// Main function for concurrent client testing
int main(int argc, char* argv[]) {
    if (argc != 5) {
        printf("Usage: %s <host> <port> <path> <num_clients>\n", argv[0]);
        return 1;
    }

    init_debug_system("test_client.log", DEBUG_INFO);

    const char* host = argv[1];
    int port = atoi(argv[2]);
    const char* path = argv[3];
    int num_clients = atoi(argv[4]);

    pthread_t threads[MAX_CONCURRENT_CLIENTS];
    ClientConfig configs[MAX_CONCURRENT_CLIENTS];
    int success_count = 0;
    pthread_mutex_t count_mutex = PTHREAD_MUTEX_INITIALIZER;

    if (num_clients > MAX_CONCURRENT_CLIENTS) {
        ERROR("Too many clients requested. Maximum is %d", MAX_CONCURRENT_CLIENTS);
        return 1;
    }

    INFO("Starting concurrent test with %d clients", num_clients);
    time_t start_time = time(NULL);

    // Create client threads
    for (int i = 0; i < num_clients; i++) {
        strncpy(configs[i].server_host, host, sizeof(configs[i].server_host) - 1);
        configs[i].server_port = port;
        strncpy(configs[i].request_path, path, sizeof(configs[i].request_path) - 1);
        configs[i].client_id = i;
        configs[i].success_count = &success_count;
        configs[i].count_mutex = &count_mutex;

        if (pthread_create(&threads[i], NULL, client_thread, &configs[i]) != 0) {
            ERROR("Failed to create client thread %d", i);
            return 1;
        }
    }

    // Wait for all clients to complete
    for (int i = 0; i < num_clients; i++) {
        pthread_join(threads[i], NULL);
    }

    time_t end_time = time(NULL);
    INFO("Test completed: %d/%d requests succeeded in %ld seconds", 
         success_count, num_clients, end_time - start_time);

    cleanup_debug_system();
    return (success_count == num_clients) ? 0 : 1;
}
```

#### Core Logic Skeleton Code

**Debug Integration Points (`debug/debug.c`):**
```c
#include "debug.h"
#include "../http_server.h"  // Include main server headers

// TODO 1: Implement debug level configuration from command line or config file
// TODO 2: Add rotation support for log files to prevent disk space issues
// TODO 3: Implement performance counters for requests per second, average response time
// TODO 4: Add memory leak detection hooks that track malloc/free pairs
// TODO 5: Implement network event logging for connection accept, close, error events

void init_debug_system(const char* log_file, DebugLevel level) {
    // TODO: Open log file with error checking
    // TODO: Set up signal handlers for log rotation (SIGUSR1)
    // TODO: Initialize performance monitoring structures
    // TODO: Set debug level and validate range
}

void log_request_processing(const HTTPRequest* request, const ConnectionContext* context) {
    // TODO 1: Log request line (method, path, version) with client IP
    // TODO 2: Log important headers (Host, Content-Length, User-Agent)
    // TODO 3: Record request timestamp for performance measurement
    // TODO 4: Assign unique request ID for tracking through processing pipeline
    // TODO 5: Log any security-relevant information (potential attacks)
}

void log_response_generation(const HTTPResponse* response, const ConnectionContext* context) {
    // TODO 1: Log response status code and reason phrase
    // TODO 2: Log response size (Content-Length) for bandwidth monitoring
    // TODO 3: Calculate and log request processing time
    // TODO 4: Log any errors that occurred during request processing
    // TODO 5: Update performance counters (total requests, success rate)
}

void debug_connection_state(const ConnectionContext* context, ConnectionState state) {
    // TODO 1: Log state transition with timestamp and connection ID
    // TODO 2: Track connection duration in each state for performance analysis
    // TODO 3: Detect stuck connections that stay in same state too long
    // TODO 4: Log thread ID handling this connection for concurrency debugging
    // TODO 5: Record any error conditions that triggered state changes
}

void monitor_server_health(pid_t server_pid) {
    // TODO 1: Read /proc/<pid>/stat to get CPU usage, memory usage, thread count
    // TODO 2: Count open file descriptors in /proc/<pid>/fd/ directory
    // TODO 3: Check listening socket status and connection queue depth
    // TODO 4: Monitor for memory leaks by tracking heap size growth
    // TODO 5: Alert if resource usage exceeds configurable thresholds
    // Hint: Use /proc filesystem for detailed process information
}
```

#### Language-Specific Hints

**GDB Debugging Techniques:**
- Use `set follow-fork-mode child` to debug worker threads in thread-per-connection model
- Create `.gdbinit` file with custom commands for examining server structures
- Use `thread apply all bt` to see all thread call stacks when investigating deadlocks
- Set conditional breakpoints: `break handle_client_connection if client_fd == 7`

**Valgrind Memory Debugging:**
- Compile with `-g -O0` for better stack traces in Valgrind output
- Use `--track-origins=yes` to see where uninitialized memory came from
- Set `--leak-check=full --show-leak-kinds=all` for comprehensive leak detection
- Use suppressions file to ignore false positives from system libraries

**System Call Tracing:**
- Use `strace -e trace=network,file` to see only networking and file operations
- Add `-f` flag to trace child processes and threads
- Use `-o trace.log` to capture output to file for analysis
- Filter specific operations: `strace -e trace=read,write,accept,close`

#### Milestone Checkpoint

**After Implementing Debug Infrastructure:**
```bash
# Compile server with debug support
gcc -DDEBUG_MEMORY -g -O0 -o http_server *.c -lpthread

# Start server with debug logging
./http_server --debug-level=3 --log-file=debug.log --port=8080

# Test with concurrent clients
./test_client localhost 8080 /index.html 10

# Check for resource leaks
valgrind --leak-check=full ./http_server --port=8081 &
sleep 5
./test_client localhost 8081 / 5
kill %1
```

**Expected Debug Output:**
- Connection accept and close events with timestamps
- Request parsing progress with validation results
- File serving operations with path resolution details
- Thread creation and cleanup events
- Performance timing for each request phase
- Resource usage summaries (memory, file descriptors, threads)

**Signs of Issues:**
- Missing cleanup log entries indicate resource leaks
- Increasing response times suggest performance degradation
- Error logs with stack traces indicate crash conditions
- File descriptor counts that don't decrease indicate leaks


## Future Extensions

> **Milestone(s):** Beyond Milestones 1-4 - explores potential enhancements and architectural considerations for evolving the HTTP server into a production-quality system

The future extensions section serves as our **roadmap for growth** - examining how the current educational HTTP server can evolve into a production-ready system while maintaining its architectural foundations. Think of this like **planning the expansion of a successful small business**: we need to understand which parts of our current structure will scale gracefully, which components will need fundamental redesign, and how to prioritize improvements based on real-world usage patterns.

The beauty of a well-designed educational project lies not just in what it teaches initially, but in how it **grows with the learner**. Our current HTTP server implementation provides a solid foundation of networking, HTTP protocol handling, and concurrency management concepts. However, the gap between an educational server and a production system involves numerous considerations around performance, security, standards compliance, and operational requirements.

This section explores three major categories of extensions: **HTTP protocol enhancements** that expand our standards compliance and feature set, **performance optimizations** that enable the server to handle production-scale loads efficiently, and **security enhancements** that protect against real-world threats. Each category represents a different dimension of system evolution, and understanding their interactions helps guide prioritization decisions.

The architectural decisions we made during the initial implementation - particularly around component separation, data structure design, and error handling patterns - directly impact how easily these extensions can be integrated. Well-designed extensibility points become **force multipliers** for future development, while architectural technical debt creates **friction and complexity** that slows enhancement efforts.

### HTTP Protocol Extensions

The HTTP protocol landscape extends far beyond the basic GET request handling implemented in our educational server. Modern web applications require support for additional HTTP methods, transfer encoding mechanisms, and protocol versions that enable richer client-server interactions and improved performance characteristics.

#### Mental Model: Library Service Expansion

Think of HTTP protocol extensions like **expanding a library's services**. Our current server is like a basic reference desk that can find and deliver books (static files). Adding POST support is like adding a **suggestion box** where patrons can submit new information. Implementing chunked encoding is like allowing **serialized novel delivery** - sending a long book chapter by chapter rather than making patrons wait for the complete volume. Upgrading to HTTP/2 is like **revolutionizing the entire library system** - allowing multiple simultaneous book requests, prioritized delivery, and compressed catalog information.

Each protocol extension builds upon our existing foundation while adding new capabilities that serve different use cases and performance requirements.

#### POST Request Support

Adding POST request support represents the most fundamental HTTP protocol extension, transforming our server from a **read-only content delivery system** into a **bidirectional communication platform**. This enhancement primarily impacts our HTTP parser component and requires new data handling patterns throughout the request processing pipeline.

The POST method differs significantly from GET requests in several key aspects. POST requests typically include substantial **message body content** that must be parsed, validated, and processed according to the specified Content-Type header. Unlike GET requests where all parameters are encoded in the URL path and query string, POST requests can carry **arbitrary payload data** ranging from form submissions to JSON API calls to file uploads.

Our current `HTTPRequest` structure already includes `body` and `body_length` fields designed to accommodate POST request handling, but the parser implementation focuses exclusively on GET request scenarios. Extending POST support requires enhancing the `parse_message_body` function to handle different encoding types and implementing new request processing logic throughout the system.

| POST Implementation Component | Current State | Required Enhancement | Complexity Level |
|-------------------------------|---------------|---------------------|------------------|
| Request Body Parsing | Placeholder implementation | Full Content-Length and Content-Type handling | Medium |
| Form Data Processing | Not implemented | URL-encoded and multipart parsing | High |
| JSON Payload Handling | Not implemented | JSON parsing and validation | Medium |
| File Upload Support | Not implemented | Multipart form processing with file boundaries | High |
| Request Routing | Path-based only | Method-based routing with POST endpoints | Low |
| Response Generation | Static content only | Dynamic response creation based on POST data | Medium |

The most straightforward POST enhancement involves implementing **application/x-www-form-urlencoded** form processing, where the message body contains key-value pairs encoded similar to URL query parameters. This builds directly on existing URL parsing concepts while introducing message body handling patterns.

More sophisticated POST implementations require supporting **multipart/form-data** encoding for file uploads, which introduces complex parsing logic for boundary detection and multiple content sections within a single request. The multipart parsing challenge involves state machine logic similar to our existing HTTP header parsing but with additional complexity around binary data handling and content boundary recognition.

> **Key Design Insight**: POST request support fundamentally changes the server's role from a passive content delivery system to an active data processing system, requiring careful consideration of data validation, storage, and security implications.

**Architecture Decision Record for POST Implementation Strategy:**

> **Decision: Form-First POST Implementation**
> - **Context**: Multiple POST payload types have different complexity levels and use cases, requiring prioritization of implementation order
> - **Options Considered**: 1) JSON-first API server approach, 2) Form-first web application approach, 3) Generic payload processor
> - **Decision**: Implement form-encoded POST requests first, followed by JSON, then multipart file uploads
> - **Rationale**: Form encoding builds naturally on existing URL parsing logic, provides immediate value for web applications, and establishes message body processing patterns for more complex formats
> - **Consequences**: Creates a natural progression from simple to complex POST handling while maintaining compatibility with existing GET-focused architecture

#### Chunked Transfer Encoding

Chunked transfer encoding addresses a fundamental limitation in our current static file serving approach: the requirement to know the complete response size before sending the HTTP headers. This encoding mechanism enables **streaming response delivery** where content can be generated and transmitted incrementally without buffering the entire response in memory.

The current file serving implementation uses the `serve_file_content` function to read complete files into memory and set the Content-Length header before transmission. This approach works well for small static files but creates **memory pressure** and **response latency** issues for large files or dynamically generated content.

Chunked encoding transforms the HTTP response format by removing the Content-Length header and instead sending the message body as a series of **size-prefixed chunks** terminated by a zero-length chunk. Each chunk includes a hexadecimal size indicator followed by the actual data bytes, enabling the client to reconstruct the complete response without knowing the total size in advance.

| Chunked Encoding Component | Implementation Requirements | Integration Points |
|-----------------------------|----------------------------|-------------------|
| Response Header Modification | Replace Content-Length with Transfer-Encoding: chunked | `add_response_header` function |
| Chunk Size Calculation | Determine optimal chunk sizes for different content types | File reading and response generation |
| Chunk Formatting | Hex size prefix + CRLF + data + CRLF formatting | Socket writing functions |
| Stream Processing | Read-process-send pipeline instead of read-all-send | `serve_file_content` redesign |
| Error Handling | Partial chunk transmission and recovery scenarios | Connection cleanup procedures |
| Client Compatibility | HTTP/1.1 compliance and graceful fallback | Request parsing and negotiation |

The implementation challenge centers around **transforming our file serving pipeline** from a buffer-based approach to a streaming approach. Instead of calling `read_file_content` to load an entire file, the chunked implementation requires a `stream_file_content` function that reads fixed-size chunks, formats them according to the chunked encoding specification, and sends them immediately.

Chunked encoding particularly benefits scenarios involving **large static files** (videos, archives, datasets) and **dynamic content generation** (server-side rendering, API responses based on database queries). The memory usage remains constant regardless of response size, and clients can begin processing received data before the complete response arrives.

The error handling implications of chunked encoding extend beyond simple file reading failures. Network interruptions during chunk transmission create **partially transmitted responses** that require careful cleanup and potential retry logic. Unlike fixed-length responses where clients can easily detect incomplete transmissions, chunked responses require proper termination with a zero-length chunk to signal completion.

> **Performance Consideration**: Chunked encoding trades CPU overhead (chunk formatting) for memory efficiency and reduced latency, making it particularly valuable for resource-constrained servers handling large files or high concurrency loads.

#### HTTP/2 Protocol Support

HTTP/2 represents the most significant protocol enhancement, introducing **multiplexed streams**, **header compression**, **server push**, and **binary framing** that fundamentally change client-server communication patterns. This upgrade provides substantial performance benefits but requires extensive architectural modifications throughout our server implementation.

The current HTTP server operates on HTTP/1.1 assumptions: **one request per connection**, **text-based headers**, and **sequential request processing**. HTTP/2 transforms each TCP connection into a **multiplexed stream container** where multiple requests and responses can be interleaved, prioritized, and processed concurrently.

**Key HTTP/2 Features and Implementation Challenges:**

| HTTP/2 Feature | Description | Implementation Complexity | Architectural Impact |
|----------------|-------------|---------------------------|---------------------|
| Binary Framing | Replace text-based protocol with binary frame format | High | Complete parser rewrite |
| Stream Multiplexing | Multiple requests per connection with stream IDs | Very High | Concurrency model redesign |
| Header Compression (HPACK) | Stateful header compression across requests | High | New compression state management |
| Server Push | Proactive resource delivery before client requests | Medium | New response generation patterns |
| Flow Control | Per-stream and connection-level flow control | High | New backpressure handling |
| Stream Prioritization | Client-specified request priority handling | Medium | Scheduler implementation |

The **binary framing layer** represents the foundation of HTTP/2 implementation, requiring a complete replacement of our text-based HTTP parser with a binary frame parser that handles different frame types (DATA, HEADERS, SETTINGS, WINDOW_UPDATE, etc.). This change affects every component that currently processes HTTP messages.

**Stream multiplexing** introduces the most significant architectural challenge: transforming our connection-based concurrency model into a **stream-based model**. Instead of dedicating threads or event handlers to individual connections, HTTP/2 requires managing multiple concurrent streams within each connection while maintaining proper flow control and prioritization.

The HPACK header compression algorithm maintains **connection-specific compression state** that must be synchronized between client and server across all streams within a connection. This stateful compression provides significant bandwidth savings for repeated headers but introduces complexity around state management and error recovery.

**HTTP/2 Architecture Decision Record:**

> **Decision: HTTP/2 as Separate Server Implementation**
> - **Context**: HTTP/2 requires extensive architectural changes that conflict with educational clarity of current HTTP/1.1 implementation
> - **Options Considered**: 1) Extend current server with HTTP/2 support, 2) Create separate HTTP/2 server, 3) Implement HTTP/2 proxy layer
> - **Decision**: Implement HTTP/2 as a separate server that shares file handling and configuration components
> - **Rationale**: Maintains educational clarity of HTTP/1.1 implementation while allowing HTTP/2 exploration without architectural compromises
> - **Consequences**: Requires code duplication for shared functionality but preserves learning progression and enables focused HTTP/2 study

HTTP/2 server push capabilities enable **proactive resource delivery** where the server anticipates client needs and sends resources before explicit requests. For static file servers, this might involve automatically pushing CSS and JavaScript files when serving HTML pages, reducing overall page load times.

The implementation complexity of full HTTP/2 support suggests treating it as a **separate learning project** rather than an extension of the current server. The concepts learned from HTTP/1.1 implementation provide essential background for understanding HTTP/2 benefits and trade-offs, but the implementation approaches differ significantly enough to warrant dedicated focus.

### Performance Optimizations

Performance optimization transforms our educational HTTP server from a **learning demonstration** into a **production-capable system** by addressing efficiency bottlenecks, resource utilization patterns, and scalability limitations inherent in the straightforward implementation approach.

#### Mental Model: Restaurant Efficiency Improvements

Think of performance optimizations like **improving restaurant operations**. Our current server is like a small family restaurant that prepares each order individually from scratch - functional but inefficient under load. Adding caching is like **preparing popular dishes in advance** and keeping them warm. Connection pooling is like **seating arrangements** that minimize table turnover time. Asynchronous I/O is like **optimizing kitchen workflow** so cooks don't wait idle while ingredients are being prepared.

Each optimization addresses specific bottlenecks while maintaining the core functionality and correctness of the original system.

#### Response Caching System

Response caching addresses the fundamental inefficiency of **repeatedly reading and processing identical static files** for multiple client requests. Our current implementation calls `serve_file_content` for every request, resulting in redundant file system operations, memory allocations, and response formatting work.

A well-designed caching system maintains **pre-processed response data** in memory for frequently requested files, eliminating file system access latency and reducing CPU overhead for duplicate requests. The caching implementation requires careful consideration of memory usage, cache invalidation policies, and thread safety for concurrent access patterns.

**Cache Architecture Components:**

| Cache Component | Responsibility | Design Considerations |
|-----------------|---------------|----------------------|
| Cache Storage | In-memory response data storage | Memory limits, eviction policies, data structure efficiency |
| Cache Keys | Unique identifiers for cached responses | File path normalization, query parameter handling |
| Cache Validation | Determining when cached data is stale | File modification time tracking, TTL policies |
| Cache Population | Loading and formatting responses for storage | Background vs. on-demand loading strategies |
| Cache Eviction | Removing entries when memory limits reached | LRU, LFU, or size-based eviction algorithms |
| Thread Safety | Concurrent access protection | Read-write locks, atomic operations |

The cache key generation strategy directly impacts both **correctness** and **performance**. Simple file path keys work for basic static serving but require enhancement for scenarios involving query parameters, range requests, or content negotiation. Cache keys must account for all factors that influence response content while remaining efficient to compute and compare.

**Cache Validation Mechanisms:**

| Validation Approach | Implementation | Pros | Cons |
|---------------------|---------------|------|------|
| File Modification Time | Compare cached mtime with current file mtime | Simple, automatic invalidation | Requires stat() system call per request |
| Time-To-Live (TTL) | Expire entries after fixed time period | Minimal overhead, predictable memory usage | May serve stale content, requires tuning |
| Manual Invalidation | Administrative cache clearing commands | Precise control over freshness | Requires operational procedures |
| Content Hashing | Hash file contents for change detection | Accurate change detection | CPU overhead for large files |

Memory management becomes critical for caching systems serving diverse content types and sizes. **Large file caching** can quickly exhaust available memory, while **many small files** create overhead from cache entry bookkeeping. Effective cache implementations require **adaptive policies** that balance memory usage against cache hit rates.

The thread safety requirements for caching introduce **reader-writer lock patterns** where multiple threads can simultaneously read cached responses but cache updates require exclusive access. High-performance implementations might use **lock-free data structures** or **per-thread cache partitions** to minimize synchronization overhead.

> **Cache Design Insight**: The optimal cache strategy depends heavily on access patterns - a few frequently requested files benefit from simple LRU caching, while many equally popular files require more sophisticated memory management and eviction policies.

**Architecture Decision Record for Cache Implementation:**

> **Decision: Two-Tier Cache with Size-Based Eviction**
> - **Context**: Static file servers exhibit varied access patterns from hot index pages to diverse asset files, requiring different caching strategies
> - **Options Considered**: 1) Simple LRU cache, 2) TTL-based cache, 3) Two-tier cache with hot/warm separation
> - **Decision**: Implement small hot cache for most frequent files plus larger warm cache with size-based eviction
> - **Rationale**: Hot cache ensures best performance for critical files, warm cache handles diverse access patterns without excessive memory usage
> - **Consequences**: Increased implementation complexity but better performance characteristics across different usage patterns

#### Connection Pooling and Keep-Alive

Connection pooling optimizations address the **TCP connection establishment overhead** that becomes significant under high request loads. Our current implementation follows a **connection-per-request** pattern where each HTTP request requires complete TCP handshake, processing, and teardown cycles.

HTTP/1.1 keep-alive functionality enables **connection reuse** where a single TCP connection can handle multiple sequential HTTP requests, eliminating repeated connection establishment costs. This optimization particularly benefits scenarios with **multiple small requests** from the same client, such as web pages loading numerous CSS, JavaScript, and image assets.

**Keep-Alive Implementation Requirements:**

| Keep-Alive Component | Current Implementation | Required Changes |
|---------------------|----------------------|------------------|
| Connection State Tracking | Single request per connection | Multi-request connection lifecycle management |
| Request Parsing | Read once and close | Continuous parsing loop with request boundaries |
| Response Headers | Connection: close default | Connection: keep-alive header management |
| Timeout Handling | Simple request timeout | Idle connection timeout and cleanup |
| Resource Management | Per-request cleanup | Connection-level resource management |
| Error Recovery | Close on any error | Graceful error handling with connection preservation |

The keep-alive implementation transforms our **connection handling loop** from a single request processor into a **multi-request state machine**. After sending each response, the server must determine whether to close the connection or continue listening for additional requests on the same socket.

Connection timeout management becomes more sophisticated with keep-alive support. Instead of simple request processing timeouts, the server must track **idle connection time** and implement policies for closing connections that remain unused beyond acceptable limits. This prevents **connection resource exhaustion** from clients that establish connections but send no additional requests.

**Connection Pooling Architecture:**

| Pool Component | Functionality | Implementation Considerations |
|----------------|---------------|------------------------------|
| Active Connection Registry | Track all open keep-alive connections | Thread-safe connection list management |
| Idle Timeout Monitor | Close connections exceeding idle time limits | Background cleanup thread or event-driven expiration |
| Connection Limits | Enforce maximum concurrent connection counts | Per-client and global connection limits |
| Graceful Shutdown | Complete in-flight requests before closing | Coordinated shutdown across pooled connections |

The connection pooling system must **balance resource utilization** against performance benefits. Keeping too many idle connections consumes file descriptors and memory, while aggressive connection closing eliminates performance advantages. Effective implementations use **adaptive timeout policies** that adjust based on server load and connection usage patterns.

Error handling complexity increases with connection pooling because errors can occur **between requests** rather than just during request processing. Network errors, client disconnections, and protocol violations must be detected and handled gracefully while preserving other active connections within the pool.

> **Performance Trade-off**: Keep-alive connections reduce latency and CPU overhead for multiple requests but increase memory usage and complexity around connection lifecycle management and error recovery.

#### Asynchronous I/O and Event-Driven Processing

Asynchronous I/O represents the most significant architectural optimization, transforming our **thread-based concurrency model** into an **event-driven model** that can handle thousands of concurrent connections with minimal resource overhead. This optimization addresses the fundamental scalability limitation of thread-per-connection architectures.

Our current threading implementation creates **OS threads** for each client connection, leading to substantial memory overhead (typically 2-8MB per thread for stack space) and **context switching costs** as the number of concurrent connections increases. Event-driven architectures use a **single thread with I/O multiplexing** to handle many connections efficiently.

**Event-Driven Architecture Components:**

| Component | Thread-Based Current | Event-Driven Alternative | Benefits |
|-----------|---------------------|-------------------------|----------|
| Connection Management | Thread per connection | Single event loop with connection registry | Reduced memory overhead |
| I/O Operations | Blocking reads and writes | Non-blocking I/O with event notifications | Eliminate thread blocking |
| Request Processing | Sequential within thread | State machine with event callbacks | Handle partial operations efficiently |
| Concurrency Control | Thread synchronization | Single-threaded with cooperative scheduling | Eliminate race conditions |
| Resource Usage | Scales with thread count | Constant regardless of connection count | Improved scalability |

The **event loop** becomes the core architectural component, continuously polling for I/O events across all active connections and dispatching appropriate handlers based on event types (readable, writable, error conditions). This requires transforming our **sequential request processing logic** into a **state machine** that can handle partial operations and resume processing when I/O operations complete.

**State Machine for Event-Driven Request Processing:**

| Connection State | Event Trigger | Next State | Actions Performed |
|------------------|---------------|------------|-------------------|
| `CONNECTION_READING_REQUEST` | Socket readable | `CONNECTION_PARSING_REQUEST` | Read available data, check for complete request |
| `CONNECTION_PARSING_REQUEST` | Request complete | `CONNECTION_HANDLING_REQUEST` | Parse HTTP request, validate headers |
| `CONNECTION_HANDLING_REQUEST` | File I/O complete | `CONNECTION_SENDING_RESPONSE` | Generate response, prepare for transmission |
| `CONNECTION_SENDING_RESPONSE` | Socket writable | `CONNECTION_READING_REQUEST` or cleanup | Send response data, handle partial writes |
| `CONNECTION_ERROR` | Any error condition | Connection cleanup | Log error, close socket, free resources |

The **file I/O handling** presents particular challenges for event-driven architectures because traditional file operations are **inherently blocking**. Reading large files can stall the entire event loop, negating the benefits of asynchronous networking. Advanced implementations require **thread pools for blocking operations** or **asynchronous file I/O** using system-specific interfaces like Linux's `io_uring` or POSIX AIO.

**Non-Blocking I/O Considerations:**

| I/O Operation | Blocking Behavior | Event-Driven Solution |
|---------------|-------------------|----------------------|
| Socket Accept | May block if no connections pending | Use `select`/`poll`/`epoll` to check readiness |
| Socket Read | May block if no data available | Read only when socket is readable, handle partial reads |
| Socket Write | May block if send buffer full | Write only when socket is writable, queue remaining data |
| File Read | Always blocks until data available | Use thread pool or async file I/O interfaces |
| File Stat | May block for network filesystems | Cache stat results or use background threads |

Memory management patterns change significantly in event-driven architectures. Instead of **per-thread stack allocation**, all connection state must be **heap-allocated** and managed explicitly. This requires careful attention to memory leaks and proper cleanup when connections close unexpectedly.

**Architecture Decision Record for Asynchronous I/O:**

> **Decision: Hybrid Event-Driven with Thread Pool for Blocking Operations**
> - **Context**: Pure event-driven architecture conflicts with educational clarity while pure threading doesn't scale effectively
> - **Options Considered**: 1) Pure event-driven with async file I/O, 2) Thread-per-connection with optimizations, 3) Hybrid approach
> - **Decision**: Event-driven networking with dedicated thread pool for file operations
> - **Rationale**: Provides scalability benefits of event-driven I/O while maintaining comprehensible file handling logic
> - **Consequences**: More complex architecture but realistic performance characteristics for production deployment

The debugging and development complexity of event-driven systems significantly exceeds threaded implementations. **State transitions** must be carefully managed, **partial operations** must be handled correctly, and **error conditions** can occur at any point in the state machine. However, the scalability benefits often justify this complexity for production systems.

### Security Enhancements

Security enhancements transform our educational HTTP server from a **trusted environment demonstration** into a **production-hardened system** capable of safely handling untrusted client requests and operating in hostile network environments. These enhancements address authentication, authorization, encryption, and attack prevention mechanisms essential for real-world deployment.

#### Mental Model: Bank Security Systems

Think of security enhancements like **progressively fortifying a bank**. Our current server is like a **friendly neighborhood credit union** - functional and trustworthy within a known community but lacking defenses against sophisticated threats. Adding HTTPS is like **installing bulletproof glass** and **encrypted communication channels**. Implementing authentication is like **requiring ID verification** and **access badges**. Additional security headers are like **alarm systems** and **surveillance cameras** that detect and deter various attack patterns.

Each security layer addresses specific threat categories while maintaining the core functionality of the underlying system.

#### HTTPS and TLS Implementation

HTTPS support addresses the fundamental security vulnerability of **unencrypted communication** between clients and servers. Our current HTTP implementation transmits all data in plaintext, making it vulnerable to **eavesdropping**, **man-in-the-middle attacks**, and **content tampering** by network intermediaries.

TLS (Transport Layer Security) provides **cryptographic protection** through authentication, encryption, and integrity verification. The TLS handshake establishes a secure channel between client and server using certificate-based server authentication and negotiated encryption algorithms.

**TLS Integration Architecture:**

| TLS Component | Integration Point | Implementation Requirements |
|---------------|-------------------|----------------------------|
| Certificate Management | Server startup and configuration | X.509 certificate loading, private key handling, certificate chain validation |
| TLS Handshake | TCP connection acceptance | Protocol version negotiation, cipher suite selection, certificate presentation |
| Encrypted I/O | Socket read/write operations | TLS record layer encryption/decryption, MAC verification |
| Session Management | Connection lifecycle | Session resumption, renegotiation, graceful termination |
| Error Handling | Connection and protocol errors | TLS-specific error codes, certificate validation failures |

The **certificate management system** requires handling X.509 certificates, private keys, and certificate authority chains. Production deployments need **automated certificate renewal** integration with services like Let's Encrypt, certificate validation procedures, and secure private key storage.

**TLS Handshake Process Integration:**

| Handshake Phase | Server Responsibilities | Implementation Considerations |
|-----------------|------------------------|------------------------------|
| ClientHello Processing | Parse supported cipher suites and extensions | Version compatibility, cipher suite selection policies |
| Certificate Presentation | Send server certificate chain | Certificate chain validation, intermediate CA handling |
| Key Exchange | Generate and exchange cryptographic keys | RSA, ECDHE, or other key exchange algorithms |
| Finished Messages | Verify handshake integrity | MAC computation and verification |
| Application Data | Begin encrypted HTTP processing | Seamless transition to normal HTTP request handling |

The **performance implications** of TLS encryption include CPU overhead for cryptographic operations, memory requirements for TLS state, and additional network round-trips for handshake completion. Modern servers use **hardware acceleration** for cryptographic operations and **session resumption** to amortize handshake costs across multiple connections.

**TLS Library Integration Options:**

| TLS Library | Implementation Approach | Pros | Cons |
|-------------|------------------------|------|------|
| OpenSSL | Direct library integration | Full feature support, widely used | Complex API, large dependency |
| mbedTLS | Embedded-focused TLS library | Smaller footprint, cleaner API | Less community support |
| LibreSSL | OpenSSL fork with security focus | Improved security, simplified API | Compatibility considerations |
| Native OS TLS | Platform-specific TLS APIs | OS integration, automatic updates | Platform-specific implementation |

The integration challenge involves **replacing standard socket operations** with TLS-aware equivalents throughout our connection handling code. The `read_complete_request` and response sending functions must use TLS library calls instead of direct socket I/O while maintaining identical semantics for the higher-level HTTP processing logic.

> **Security Consideration**: TLS configuration requires careful attention to cipher suite selection, protocol version support, and certificate validation policies to prevent downgrade attacks and ensure strong cryptographic protection.

**Architecture Decision Record for HTTPS Implementation:**

> **Decision: TLS Termination at HTTP Server Level**
> - **Context**: HTTPS support can be implemented at the HTTP server level or handled by a reverse proxy in front of our server
> - **Options Considered**: 1) Integrate TLS directly into HTTP server, 2) Use reverse proxy for TLS termination, 3) Support both deployment modes
> - **Decision**: Implement TLS integration directly in HTTP server with configuration option to disable for proxy deployments
> - **Rationale**: Educational value of understanding TLS integration while maintaining flexibility for production deployment patterns
> - **Consequences**: Increased complexity but comprehensive understanding of HTTPS implementation details

#### Authentication and Authorization Framework

Authentication and authorization mechanisms enable **identity verification** and **access control** for protected resources, transforming our public file server into a **controlled access system** with user management and permission enforcement capabilities.

Authentication addresses the question **"who is this user?"** through credential verification mechanisms like username/password combinations, API keys, or cryptographic certificates. Authorization answers **"what can this user access?"** through permission systems that control resource access based on verified identity and assigned roles.

**Authentication Mechanisms for HTTP Servers:**

| Authentication Method | Implementation Requirements | Security Characteristics | Use Cases |
|-----------------------|----------------------------|-------------------------|-----------|
| HTTP Basic Authentication | Base64 credential encoding, credential verification | Simple but requires HTTPS for security | Development, simple applications |
| HTTP Digest Authentication | Challenge-response with cryptographic hashes | More secure than basic, complex implementation | Legacy compatibility |
| Bearer Token Authentication | Token generation, validation, and management | Flexible, supports various token formats | API access, modern web applications |
| Certificate-Based Authentication | Client certificate verification during TLS handshake | Strong security, complex PKI requirements | High-security environments |
| Session-Based Authentication | Session token management with server-side storage | Traditional web application pattern | User-facing web applications |

**HTTP Basic Authentication** provides the simplest starting point, requiring minimal protocol changes while demonstrating fundamental authentication concepts. The client sends credentials in an Authorization header, and the server validates them against a user database or configuration file.

**Basic Authentication Implementation Components:**

| Component | Functionality | Integration Points |
|-----------|---------------|-------------------|
| Credential Storage | User database with hashed passwords | Configuration system, user management |
| Header Parsing | Extract and decode Authorization header | HTTP parser component |
| Credential Verification | Compare provided credentials against stored values | Request processing pipeline |
| Challenge Generation | Send WWW-Authenticate header for unauthorized requests | Error response generation |
| Protected Resource Identification | Determine which paths require authentication | File handler component |

The **authorization framework** builds upon authentication to implement **role-based access control** (RBAC) or **access control lists** (ACLs) that define resource-specific permissions. This requires extending our path resolution logic to check user permissions before serving files.

**Authorization Policy Examples:**

| Resource Pattern | Permission Model | Implementation Approach |
|------------------|------------------|------------------------|
| `/public/*` | Open access | No authentication required |
| `/private/*` | Authenticated users only | Require valid authentication |
| `/admin/*` | Administrative users only | Role-based permission check |
| `/user/{username}/*` | Owner access only | Path-based ownership validation |
| `/api/v1/*` | API key authentication | Token-based authentication |

The **session management** aspect of authentication involves maintaining user state across multiple requests. This requires **secure session token generation**, **server-side session storage**, and **session timeout policies** to balance security and usability.

**Session Management Architecture:**

| Session Component | Responsibilities | Security Considerations |
|-------------------|------------------|------------------------|
| Session Token Generation | Create cryptographically secure session identifiers | Use sufficient entropy, avoid predictable patterns |
| Session Storage | Maintain server-side session data | Memory management, persistence options |
| Session Validation | Verify session tokens on each request | Constant-time comparison, timeout enforcement |
| Session Cleanup | Remove expired sessions | Background cleanup processes, memory management |
| Session Security | Protect against session-based attacks | CSRF protection, secure cookie attributes |

> **Authentication Design Principle**: Authentication systems must balance security requirements with usability concerns - overly complex authentication reduces adoption while insufficient security creates vulnerability to compromise.

#### Security Headers and Attack Prevention

Security headers provide **defense-in-depth protection** against common web application attacks by instructing browsers and other clients to enforce additional security policies. These headers complement server-side security measures with **client-side protection mechanisms**.

**Essential Security Headers:**

| Header Name | Protection Purpose | Configuration Considerations |
|-------------|-------------------|----------------------------|
| `Strict-Transport-Security` (HSTS) | Enforce HTTPS connections, prevent protocol downgrade | Max-age policy, subdomain inclusion |
| `Content-Security-Policy` (CSP) | Prevent XSS attacks through content source restrictions | Policy complexity, compatibility requirements |
| `X-Frame-Options` | Prevent clickjacking attacks | Deny vs. SAMEORIGIN policy choice |
| `X-Content-Type-Options` | Prevent MIME-type sniffing attacks | nosniff directive for all content |
| `Referrer-Policy` | Control referrer information disclosure | Privacy vs. functionality balance |
| `Permissions-Policy` | Control browser feature access | Feature-specific policy configuration |

**Content Security Policy (CSP)** represents the most sophisticated security header, enabling fine-grained control over resource loading and script execution. For static file servers, CSP policies must balance security restrictions with legitimate content requirements.

**CSP Policy Examples for Static File Server:**

| Content Type | CSP Directive | Policy Recommendation | Rationale |
|--------------|---------------|----------------------|-----------|
| Scripts | `script-src` | `'self'` for same-origin scripts | Prevent inline scripts and external script injection |
| Stylesheets | `style-src` | `'self' 'unsafe-inline'` if needed | Allow legitimate styles while blocking malicious CSS |
| Images | `img-src` | `'self' data:` | Permit same-origin and data URL images |
| Media | `media-src` | `'self'` | Restrict audio/video sources |
| Fonts | `font-src` | `'self'` | Control font loading sources |
| Default | `default-src` | `'self'` | Fallback policy for unspecified resource types |

**Attack Prevention Mechanisms:**

| Attack Vector | Prevention Approach | Implementation Location | Effectiveness |
|---------------|-------------------|------------------------|---------------|
| Directory Traversal | Path validation and normalization | File handler component | High |
| Request Smuggling | Strict HTTP parsing and validation | HTTP parser component | High |
| Denial of Service | Rate limiting and resource constraints | Connection management | Medium |
| Buffer Overflow | Bounds checking and safe string operations | All input processing | High |
| Injection Attacks | Input validation and sanitization | Request processing pipeline | High |
| Information Disclosure | Error message sanitization | Error handling system | Medium |

**Rate limiting** mechanisms protect against **denial of service attacks** and **resource exhaustion** by limiting the request rate from individual clients or IP addresses. This requires tracking client request patterns and implementing **sliding window** or **token bucket** algorithms for rate enforcement.

**Rate Limiting Implementation Architecture:**

| Rate Limiting Component | Functionality | Design Considerations |
|-------------------------|---------------|----------------------|
| Request Tracking | Monitor requests per client/IP address | Memory usage, cleanup policies |
| Rate Calculation | Compute current request rates | Sliding window vs. fixed window algorithms |
| Limit Enforcement | Block or throttle excessive requests | HTTP 429 response generation |
| Configuration Management | Adjustable rate limits per resource | Per-path limits, global limits |
| Persistence | Optional rate limit state persistence | Recovery after server restart |

**Input validation** and **output encoding** prevent various injection attacks by ensuring that user-provided data cannot be interpreted as executable code or control characters. For HTTP servers, this primarily involves **header validation**, **URL encoding verification**, and **safe error message generation**.

> **Security Design Insight**: Effective security requires defense-in-depth approaches where multiple independent mechanisms provide overlapping protection against different attack vectors and failure modes.

**Architecture Decision Record for Security Implementation Approach:**

> **Decision: Graduated Security Implementation with Configuration Control**
> - **Context**: Security requirements vary significantly between development, staging, and production environments
> - **Options Considered**: 1) Maximum security by default, 2) Minimal security with opt-in features, 3) Configurable security levels
> - **Decision**: Implement tiered security levels (development, staging, production) with clear configuration options
> - **Rationale**: Enables appropriate security for each deployment context while maintaining educational clarity and flexibility
> - **Consequences**: Increased configuration complexity but realistic security posture for different deployment scenarios

The integration of security enhancements requires careful **performance impact assessment** since security operations (encryption, authentication, header processing) add computational overhead to request processing. Production deployments must balance security requirements against performance characteristics and resource utilization constraints.

### Implementation Guidance

The future extensions described above represent significant undertakings that transform our educational HTTP server into a production-ready system. This implementation guidance provides practical approaches for tackling these enhancements systematically while maintaining code quality and educational value.

#### Technology Recommendations

| Extension Category | Simple Option | Advanced Option |
|--------------------|---------------|-----------------|
| POST Request Support | Form-encoded parsing with fixed-size buffers | Streaming parser with configurable size limits |
| Chunked Encoding | Fixed-size chunks with simple formatting | Adaptive chunk sizing based on content type |
| HTTP/2 Implementation | Separate server using existing libraries | Custom implementation for learning |
| Response Caching | In-memory hash table with LRU eviction | Multi-tier cache with persistence options |
| Connection Pooling | Basic keep-alive with fixed timeouts | Adaptive timeout policies with health monitoring |
| Asynchronous I/O | Event-driven with thread pool for file I/O | Pure async with io_uring or IOCP |
| HTTPS Support | OpenSSL integration with basic configuration | Full TLS configuration management |
| Authentication | HTTP Basic with file-based user storage | Token-based with database integration |
| Security Headers | Static header configuration | Dynamic policy generation based on content |

#### Recommended Project Structure for Extensions

```
http-server-extended/
├── cmd/
│   ├── http-server/main.c           ← Original HTTP/1.1 server
│   ├── http2-server/main.c          ← Separate HTTP/2 implementation  
│   └── config-tool/main.c           ← Configuration management utility
├── src/
│   ├── core/                        ← Shared core components
│   │   ├── http_parser.c/h          ← Enhanced parser with POST support
│   │   ├── file_handler.c/h         ← File serving with caching
│   │   ├── connection_manager.c/h   ← Connection pooling and lifecycle
│   │   └── config.c/h               ← Configuration management
│   ├── extensions/                  ← Extension implementations
│   │   ├── cache/                   ← Response caching system
│   │   │   ├── cache_manager.c/h
│   │   │   ├── lru_eviction.c/h
│   │   │   └── cache_stats.c/h
│   │   ├── security/                ← Security enhancements
│   │   │   ├── tls_integration.c/h
│   │   │   ├── auth_handler.c/h
│   │   │   ├── security_headers.c/h
│   │   │   └── rate_limiter.c/h
│   │   ├── performance/             ← Performance optimizations
│   │   │   ├── async_io.c/h
│   │   │   ├── connection_pool.c/h
│   │   │   └── chunked_encoding.c/h
│   │   └── http2/                   ← HTTP/2 implementation
│   │       ├── frame_parser.c/h
│   │       ├── stream_manager.c/h
│   │       ├── hpack_compression.c/h
│   │       └── multiplexer.c/h
│   └── utils/                       ← Shared utilities
│       ├── memory_pool.c/h          ← Memory management
│       ├── thread_pool.c/h          ← Thread pool implementation
│       ├── event_loop.c/h           ← Event-driven I/O support
│       └── crypto_utils.c/h         ← Cryptographic helpers
├── tests/
│   ├── unit/                        ← Component unit tests
│   ├── integration/                 ← Full system integration tests
│   ├── performance/                 ← Load and stress testing
│   └── security/                    ← Security verification tests
├── config/
│   ├── server.conf                  ← Server configuration
│   ├── mime.types                   ← MIME type mappings
│   ├── users.conf                   ← User authentication data
│   └── security.conf                ← Security policy configuration
└── docs/
    ├── extensions/                  ← Extension-specific documentation
    ├── deployment/                  ← Production deployment guides
    └── performance/                 ← Performance tuning guides
```

#### Cache Implementation Starter Code

**Complete Cache Manager Infrastructure:**

```c
// cache_manager.h
#ifndef CACHE_MANAGER_H
#define CACHE_MANAGER_H

#include <stdint.h>
#include <time.h>
#include <pthread.h>

#define MAX_CACHE_ENTRIES 1000
#define MAX_CACHED_FILE_SIZE (1024 * 1024)  // 1MB max per file
#define CACHE_KEY_LENGTH 256

typedef struct CacheEntry {
    char key[CACHE_KEY_LENGTH];
    char* response_data;
    size_t response_length;
    time_t created_time;
    time_t last_access;
    uint64_t access_count;
    struct CacheEntry* next;
    struct CacheEntry* prev;
} CacheEntry;

typedef struct CacheManager {
    CacheEntry* entries[MAX_CACHE_ENTRIES];  // Hash table buckets
    CacheEntry* lru_head;                    // LRU list head
    CacheEntry* lru_tail;                    // LRU list tail
    size_t total_memory;                     // Current memory usage
    size_t max_memory;                       // Memory limit
    uint64_t hits;                           // Cache hit statistics
    uint64_t misses;                         // Cache miss statistics
    pthread_rwlock_t cache_lock;             // Reader-writer lock
} CacheManager;

// Initialize cache manager with memory limits
CacheManager* init_cache_manager(size_t max_memory_bytes);

// Store response in cache with automatic eviction if needed
int cache_store_response(CacheManager* cache, const char* key, 
                        const char* response_data, size_t response_length);

// Retrieve cached response, returns NULL if not found or expired
char* cache_get_response(CacheManager* cache, const char* key, 
                        size_t* response_length);

// Generate cache key from request path and relevant headers
void generate_cache_key(const char* file_path, const char* query_string,
                       char* cache_key, size_t key_length);

// Remove expired entries and enforce memory limits
void cache_cleanup_expired(CacheManager* cache, time_t max_age_seconds);

// Get cache statistics for monitoring
void get_cache_statistics(CacheManager* cache, uint64_t* hits, uint64_t* misses,
                         size_t* memory_used, size_t* entry_count);

// Cleanup and free cache resources
void cleanup_cache_manager(CacheManager* cache);

#endif // CACHE_MANAGER_H
```

#### Authentication Framework Skeleton

**HTTP Basic Authentication Core Logic:**

```c
// Basic authentication handler - learner implements credential verification
int handle_basic_authentication(HTTPRequest* request, HTTPResponse* response) {
    // TODO 1: Extract Authorization header from request
    const char* auth_header = get_request_header(request, "Authorization");
    if (!auth_header) {
        // TODO 2: Generate 401 response with WWW-Authenticate header
        // Hint: Use generate_error_response with status 401
        // Add header: WWW-Authenticate: Basic realm="Restricted Area"
        return 0; // Authentication required
    }
    
    // TODO 3: Verify Authorization header starts with "Basic "
    // TODO 4: Extract and decode base64 credentials
    // Hint: Use base64_decode function to decode credentials
    
    // TODO 5: Split decoded credentials into username:password
    // TODO 6: Verify credentials against user database
    // Hint: Use verify_user_credentials function
    
    // TODO 7: Set authentication context for authorized requests
    // Store username in request context for access control
    
    return 1; // Authentication successful
}

// User credential verification - implement based on storage choice
int verify_user_credentials(const char* username, const char* password) {
    // TODO 1: Load user database (file, memory, etc.)
    // TODO 2: Find user entry by username
    // TODO 3: Compare password hash using secure comparison
    // Hint: Use constant-time comparison to prevent timing attacks
    // TODO 4: Return 1 for valid credentials, 0 for invalid
}
```

#### TLS Integration Framework

**TLS Socket Wrapper for HTTPS Support:**

```c
// tls_socket.h - TLS wrapper around standard sockets
#ifndef TLS_SOCKET_H
#define TLS_SOCKET_H

#include <openssl/ssl.h>
#include <openssl/err.h>

typedef struct TLSContext {
    SSL_CTX* ssl_ctx;
    SSL* ssl;
    int socket_fd;
    int is_server;
} TLSContext;

// Initialize TLS context with certificate and key files
TLSContext* init_tls_server_context(const char* cert_file, const char* key_file);

// Accept TLS connection on server socket
int tls_accept_connection(TLSContext* server_ctx, int client_fd, TLSContext** client_ctx);

// TLS-aware versions of standard socket operations
ssize_t tls_read(TLSContext* ctx, void* buffer, size_t length);
ssize_t tls_write(TLSContext* ctx, const void* buffer, size_t length);
int tls_close(TLSContext* ctx);

// Cleanup TLS resources
void cleanup_tls_context(TLSContext* ctx);

#endif // TLS_SOCKET_H
```

#### Milestone Checkpoints for Extensions

**POST Request Support Milestone:**
1. **Command**: `curl -X POST -d "name=test&value=hello" http://localhost:8080/echo`
2. **Expected Response**: HTTP 200 with request body echoed back
3. **Verification**: Check that Content-Length header is correctly parsed and body data is accessible
4. **Common Issues**: Incomplete body reading, missing Content-Type handling, buffer overflow on large posts

**Caching System Milestone:**
1. **Command**: `ab -n 1000 -c 10 http://localhost:8080/index.html`
2. **Expected Behavior**: Significant performance improvement on repeated requests
3. **Verification**: Cache hit rate > 90% for repeated file access, memory usage within configured limits
4. **Monitoring**: Check cache statistics show hits/misses, verify LRU eviction under memory pressure

**HTTPS Support Milestone:**
1. **Setup**: Generate self-signed certificate: `openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes`
2. **Command**: `curl -k https://localhost:8443/index.html`
3. **Expected Response**: Same content as HTTP version but over encrypted connection
4. **Verification**: Use `openssl s_client -connect localhost:8443` to verify TLS handshake
5. **Common Issues**: Certificate loading failures, cipher suite compatibility, port binding conflicts

#### Performance Testing and Optimization

**Load Testing Framework:**

```bash
#!/bin/bash
# performance_test.sh - Comprehensive server performance testing

echo "Starting HTTP Server Performance Test Suite"

# Test 1: Concurrent connection handling
echo "Test 1: Concurrent Connections"
ab -n 10000 -c 100 http://localhost:8080/index.html

# Test 2: Large file transfer
echo "Test 2: Large File Performance" 
curl -w "@curl-format.txt" http://localhost:8080/large-file.dat

# Test 3: Keep-alive connection reuse
echo "Test 3: Keep-Alive Performance"
ab -k -n 5000 -c 50 http://localhost:8080/small-file.txt

# Test 4: Cache effectiveness
echo "Test 4: Cache Hit Rate"
# Send same request multiple times and measure response time improvement

# Test 5: Memory usage under load
echo "Test 5: Memory Usage Monitoring"
# Monitor server memory usage during high concurrent load
```

#### Security Testing Verification

**Security Audit Checklist:**

| Security Feature | Verification Method | Expected Result |
|------------------|-------------------|-----------------|
| HTTPS Encryption | `nmap --script ssl-enum-ciphers -p 8443 localhost` | Strong cipher suites only |
| Authentication | `curl -u invalid:credentials https://localhost:8443/private/` | 401 Unauthorized response |
| Directory Traversal | `curl https://localhost:8443/../../../etc/passwd` | 400 Bad Request or 403 Forbidden |
| Security Headers | `curl -I https://localhost:8443/` | HSTS, CSP, X-Frame-Options present |
| Rate Limiting | Rapid request script | 429 Too Many Requests after threshold |
| Input Validation | Malformed HTTP requests | Graceful error handling, no crashes |

These extensions transform the educational HTTP server into a comprehensive web server implementation suitable for production deployment while maintaining the architectural clarity that makes the codebase an effective learning tool. Each extension builds upon the foundational concepts established in the original implementation, demonstrating how systems evolve to meet real-world requirements.


## Glossary

> **Milestone(s):** All Milestones 1-4 - provides essential terminology definitions used across TCP server basics, HTTP request parsing, static file serving, and concurrent connections

The glossary serves as your technical compass throughout the HTTP server implementation journey. Think of it as a specialized dictionary that translates between the abstract concepts described in this design document and the concrete technical terminology you'll encounter in networking documentation, HTTP specifications, and systems programming resources. Just as a traveler needs a phrasebook to navigate a foreign country, you need this glossary to navigate the interconnected domains of network programming, HTTP protocol implementation, and concurrent systems design.

This glossary organizes terminology into logical groupings that mirror the architectural components of our HTTP server. Each definition provides not just the meaning but also the contextual significance within our specific implementation. The terminology here bridges the gap between theoretical computer science concepts and practical systems programming, helping you understand not just what each term means but why it matters for building a robust HTTP server.

### Networking and TCP Fundamentals

The networking layer forms the foundation of our HTTP server, requiring precise understanding of how TCP connections operate at the system level.

| Term | Definition | Context in HTTP Server |
|------|------------|------------------------|
| **Address Family** | Protocol family identifier specifying address format and addressing scheme | `AF_INET` specifies IPv4 addressing for our server socket configuration |
| **Bind Operation** | System call associating a socket with a specific local address and port | Our server binds to configured port (default 8080) to accept incoming connections |
| **Client Socket** | Network endpoint representing the client side of a TCP connection | Each accepted connection creates a client socket for bidirectional communication |
| **Connection Establishment** | Three-way handshake process (SYN, SYN-ACK, ACK) creating a TCP connection | Happens automatically when clients connect to our listening server socket |
| **File Descriptor** | Integer handle representing an open file or socket in the operating system | Our server manages multiple file descriptors for server socket and client connections |
| **File Descriptor Leak** | Programming error where file descriptors are not properly closed, causing resource exhaustion | Critical bug pattern in our server - must close client sockets after handling requests |
| **Listen Backlog** | Maximum number of pending connections that can wait for acceptance | Configured during socket setup to handle burst of simultaneous connection attempts |
| **Listen Operation** | System call marking a socket as passive, ready to accept incoming connections | Transitions our server socket from active to passive state for connection acceptance |
| **Network Byte Order** | Big-endian byte ordering used in network protocols for multi-byte values | Port numbers must be converted using `htons()` for proper network transmission |
| **Partial Read** | Network read operation returning fewer bytes than requested due to buffering | Common in TCP - our server must handle incomplete HTTP requests with multiple read calls |
| **Partial Write** | Network write operation sending fewer bytes than requested due to buffering | HTTP responses may require multiple write calls to send complete data |
| **Server Socket** | Network endpoint configured to accept incoming client connections | Primary socket bound to server port, never used for actual data transmission |
| **Socket Address Structure** | Data structure containing address family, port, and IP address information | `sockaddr_in` structure configures server binding and tracks client connection details |
| **Socket Options** | Configuration parameters controlling socket behavior and features | `SO_REUSEADDR` allows immediate port reuse after server restart during development |
| **TCP Connection** | Reliable, ordered, bidirectional communication channel between client and server | Foundation for HTTP request-response exchange with guaranteed delivery and ordering |
| **TCP Connection Lifecycle** | Complete sequence from socket creation through data exchange to connection termination | Our server manages: create → bind → listen → accept → read → write → close |

### HTTP Protocol Terminology

HTTP protocol implementation requires understanding of message structure, parsing challenges, and protocol semantics.

| Term | Definition | Context in HTTP Server |
|------|------------|------------------------|
| **Content-Length Header** | HTTP header specifying the exact size of the message body in bytes | Critical for reading request body completely and setting response body size |
| **Content-Type Header** | HTTP header indicating the media type of the request or response body | Our server sets this based on file extension to help browsers handle served files |
| **CRLF Line Endings** | Carriage Return + Line Feed (`\r\n`) sequence required by HTTP specification | HTTP standard mandates CRLF, but our parser handles both CRLF and LF for robustness |
| **HTTP Header** | Key-value pair providing metadata about the request or response message | Parsed into our header array structure with case-insensitive key matching |
| **HTTP Message Body** | Optional payload data following the headers in an HTTP message | For requests: form data or file uploads; for responses: file content or error pages |
| **HTTP Method** | Verb indicating the desired action to be performed on the identified resource | Our server primarily handles GET requests, with basic support for HEAD and OPTIONS |
| **HTTP Request** | Client message containing method, path, headers, and optional body | Parsed into our `HTTPRequest` structure with validated method, path, and header collection |
| **HTTP Request Line** | First line of HTTP request containing method, path, and protocol version | Format: `GET /index.html HTTP/1.1` - parsed to extract three components |
| **HTTP Response** | Server message containing status code, headers, and optional body content | Generated using our `HTTPResponse` structure with appropriate status and content headers |
| **HTTP Status Code** | Three-digit code indicating the result of the server's attempt to process the request | 200 OK for successful file serving, 404 Not Found for missing files, 500 for errors |
| **HTTP Version** | Protocol version identifier indicating capabilities and message format requirements | Our server supports HTTP/1.1 with basic connection handling and header processing |
| **Host Header** | HTTP/1.1 required header specifying the domain name and port of the server | Used for virtual host routing, though our basic implementation serves single document root |
| **Request Path** | URL path component specifying the requested resource location on the server | Validated for security (no directory traversal) and mapped to filesystem paths |
| **Request Parsing** | Process of converting raw HTTP message bytes into structured data representation | Critical parsing phase handling malformed requests, invalid headers, and protocol violations |
| **Response Generation** | Process of creating properly formatted HTTP response messages from server data | Includes status line construction, header addition, and body content formatting |
| **Status Line** | First line of HTTP response containing protocol version, status code, and reason phrase | Format: `HTTP/1.1 200 OK` - automatically generated based on request processing results |

### File System and Security

Static file serving requires careful handling of filesystem operations and security validation.

| Term | Definition | Context in HTTP Server |
|------|------------|------------------------|
| **Binary File Handling** | Special processing required for non-text files to preserve exact byte content | Our server must avoid text-mode I/O operations that could corrupt binary file data |
| **Directory Listing** | HTML page displaying the contents of a directory when no index file exists | Generated dynamically when request path maps to directory rather than specific file |
| **Directory Traversal** | Security attack using relative path components (`../`) to access files outside document root | Prevented by path normalization and validation before filesystem access |
| **Document Root** | Base directory on the filesystem from which files are served to clients | Security boundary - all served files must reside within this directory tree |
| **File Extension** | Suffix portion of filename used to determine file type and appropriate handling | Used for MIME type detection to set proper Content-Type header in responses |
| **File Permission** | Operating system access control determining read, write, and execute capabilities | Server process must have read permission for all files intended for serving |
| **MIME Type** | Standard identifier indicating the nature and format of file content | Mapped from file extensions to Content-Type headers (e.g., `.html` → `text/html`) |
| **Path Normalization** | Process of removing redundant path components (`.`, `..`) to create canonical paths | Security measure preventing directory traversal attacks through path manipulation |
| **Path Resolution** | Process of mapping URL paths to corresponding filesystem paths within document root | Core file serving operation that must validate security constraints before file access |
| **Path Validation** | Security checks ensuring requested paths don't escape document root boundaries | Combines path normalization with boundary checking to prevent unauthorized file access |
| **Static Content** | Pre-existing files served without server-side processing or dynamic generation | HTML, CSS, JavaScript, images - served directly from filesystem with appropriate headers |
| **URL Decoding** | Process of converting percent-encoded characters in URLs back to original form | Handles encoded characters like `%20` (space) in filenames and paths |

### Concurrency and Threading

Concurrent connection handling requires understanding of different concurrency models and their trade-offs.

| Term | Definition | Context in HTTP Server |
|------|------------|------------------------|
| **Asynchronous I/O** | Non-blocking I/O operations with event notification for completion | Alternative to threading - single thread handles multiple connections with I/O multiplexing |
| **Connection Context** | Per-connection state structure tracking client information and processing status | `ConnectionContext` structure maintaining client socket, address, timing, and thread information |
| **Connection Multiplexing** | Single thread handling multiple simultaneous connections through event-driven programming | Achieved using `select()` or `poll()` to monitor multiple sockets for activity |
| **Connection Pooling** | Reusing TCP connections across multiple HTTP requests to reduce connection overhead | Optimization for keep-alive connections, though not implemented in our basic server |
| **Deadlock** | Circular waiting condition where two or more threads block each other indefinitely | Avoided through consistent lock ordering and avoiding nested lock acquisition |
| **Detached Thread** | Thread that automatically releases its resources when it terminates | Created with `PTHREAD_CREATE_DETACHED` attribute to avoid memory leaks from unjoined threads |
| **Event-Driven Model** | Concurrency approach using single thread with I/O multiplexing to handle multiple connections | Alternative to threading - uses `select()` or `epoll()` to monitor socket events |
| **Graceful Shutdown** | Shutdown process that allows in-flight requests to complete before stopping the server | Prevents client connection errors during server restart by finishing active request processing |
| **Mutex** | Mutual exclusion primitive protecting shared data from concurrent access | `pthread_mutex_t` used to protect connection counters and shared server state |
| **Non-blocking I/O** | I/O operations that return immediately rather than waiting for completion | Socket option `O_NONBLOCK` allows event-driven handling without thread blocking |
| **Race Condition** | Bug where program behavior depends on timing of concurrent operations | Prevented by proper synchronization around shared data like connection counters |
| **Resource Exhaustion** | System running out of threads, memory, or file descriptors under high load | Prevented by connection limits, thread pool sizing, and proper resource cleanup |
| **Thread Pool** | Fixed number of worker threads sharing the workload of processing client connections | Limits resource usage compared to thread-per-connection while maintaining concurrency |
| **Thread Safety** | Property that code functions correctly when accessed by multiple threads simultaneously | Achieved through mutex synchronization and avoiding shared mutable state where possible |
| **Thread-per-connection** | Concurrency model creating dedicated thread for each client connection | Simplest approach but resource-intensive - each connection consumes thread stack memory |

### Error Handling and Debugging

Robust error handling requires understanding of error categories, propagation patterns, and recovery strategies.

| Term | Definition | Context in HTTP Server |
|------|------------|------------------------|
| **Cleanup Sequence** | Ordered process of releasing resources when connections complete or fail | Ensures sockets are closed, memory freed, and threads properly terminated |
| **Error Classification** | Categorizing errors by source (network, protocol, application) and severity level | Helps determine appropriate HTTP status codes and recovery actions |
| **Error Propagation** | Flow of error information between system components to appropriate handlers | Network errors become HTTP 500, file not found becomes HTTP 404 |
| **Graceful Degradation** | System continuing to function with reduced capability under stress or failure | Server remains responsive even when some requests fail or resources are limited |
| **Memory Ownership** | Clear responsibility for allocating and freeing dynamically allocated memory | Prevents memory leaks by defining which component owns and cleans up each allocation |
| **Resource Cleanup** | Process of properly releasing system resources (sockets, files, memory) after use | Critical for preventing resource leaks that would eventually crash the server |
| **Security Event Logging** | Recording security-relevant events (access attempts, blocked requests) for monitoring | Logs directory traversal attempts, permission denials, and suspicious request patterns |

### Performance and Testing

Understanding performance characteristics and testing approaches for validating server behavior.

| Term | Definition | Context in HTTP Server |
|------|------------|------------------------|
| **Cache Eviction** | Removal of entries from cache storage due to memory limits or age | LRU (Least Recently Used) policy removes oldest entries when cache reaches capacity |
| **Cache Hit** | Successful retrieval of requested data from cache storage | Avoids expensive file I/O operations by serving content from memory |
| **Connection Limits** | Maximum number of simultaneous connections the server will accept | Prevents resource exhaustion by rejecting connections beyond configured threshold |
| **Load Testing** | Testing system behavior under expected operational load conditions | Verifies server handles target number of concurrent connections without degradation |
| **Performance Monitoring** | Tracking system metrics like response time, throughput, and resource usage | Measures request processing time, concurrent connection count, and memory usage |
| **Stress Testing** | Testing system limits and failure modes under extreme load | Determines maximum connection capacity and validates graceful failure behavior |

### Advanced Features and Extensions

Terminology for potential future enhancements beyond the basic HTTP server implementation.

| Term | Definition | Context in HTTP Server |
|------|------------|------------------------|
| **Authentication** | Identity verification process for user credentials | HTTP Basic Authentication validates username/password for access control |
| **Authorization** | Access control based on verified user identity | Determines which authenticated users can access specific resources |
| **Certificate Management** | Handling X.509 certificates and private keys for HTTPS | Required for TLS-enabled servers to prove identity to clients |
| **Chunked Encoding** | HTTP transfer encoding for streaming responses without predetermined Content-Length | Allows serving dynamically generated content of unknown size |
| **Keep-Alive** | HTTP connection reuse mechanism for multiple requests over single TCP connection | Reduces connection establishment overhead for clients making multiple requests |
| **Rate Limiting** | Restricting request frequency from clients to prevent abuse | Protects server resources from denial-of-service attacks |
| **Security Headers** | HTTP headers providing client-side security policies | Headers like `X-Content-Type-Options` and `X-Frame-Options` enhance browser security |
| **TLS Handshake** | Cryptographic negotiation process establishing secure HTTPS connection | Establishes encryption keys and validates server certificate before application data |

### System Programming Concepts

Low-level systems programming concepts essential for robust HTTP server implementation.

| Term | Definition | Context in HTTP Server |
|------|------------|------------------------|
| **Buffer Overflow Protection** | Programming techniques preventing writes beyond allocated memory boundaries | Critical for parsing HTTP headers and handling file paths safely |
| **Endianness** | Byte ordering for multi-byte values in memory and network transmission | Network byte order (big-endian) differs from host byte order on little-endian systems |
| **Process Management** | Operating system concepts for creating, monitoring, and terminating processes | Understanding daemon processes and signal handling for production deployment |
| **Signal Handling** | Mechanism for asynchronous notification of events to running processes | SIGTERM and SIGINT handling for graceful shutdown, SIGPIPE handling for broken connections |
| **System Call Interface** | Operating system API for network operations, file I/O, and process management | Socket operations, file operations, and threading primitives used throughout server |

> **Implementation Note**: This glossary provides the foundational vocabulary needed to understand the HTTP server design and implementation. Each term connects to specific code elements, data structures, or algorithms described in the component sections. When encountering unfamiliar terminology in networking documentation or HTTP specifications, refer back to these definitions to maintain context within our specific server architecture.

The terminology here bridges multiple technical domains - from low-level socket programming to high-level HTTP protocol semantics. Understanding these connections helps you see how abstract protocol concepts translate into concrete programming tasks and data structures.

### Implementation Guidance

This section provides practical guidance for understanding and using the terminology throughout your HTTP server implementation.

#### Terminology Usage Patterns

The terminology in this glossary follows specific patterns that map to our implementation structure:

**Component-Specific Terms**: Terms like "TCP Connection Lifecycle" and "HTTP Request Parsing" map directly to component responsibilities. When implementing the TCP Server Component, focus on networking terms. When building the HTTP Parser Component, emphasize protocol terminology.

**Data Structure Mapping**: Many terms correspond directly to fields in our data structures. "HTTP Method" maps to the `method` field in `HTTPRequest`. "Connection Context" is embodied by the `ConnectionContext` structure. "MIME Type" relates to the `MimeTypeMapping` structure.

**Error Category Alignment**: Error-related terminology aligns with our `ErrorCategory` enumeration. "Network" errors include "Partial Read" and "File Descriptor Leak". "Protocol" errors encompass "HTTP Request Parsing" failures and malformed headers.

#### Cross-Reference Guide

| Implementation Area | Key Terms | Primary Data Structures |
|-------------------|-----------|------------------------|
| Socket Programming | Network Byte Order, File Descriptor, TCP Connection Lifecycle | `sockaddr_in`, `ConnectionContext` |
| HTTP Processing | Request Parsing, Status Code, Content-Type Header | `HTTPRequest`, `HTTPResponse` |
| File Operations | Directory Traversal, Path Validation, MIME Type | `ServerConfig`, `MimeTypeMapping` |
| Concurrency | Thread Safety, Mutex, Resource Exhaustion | `ConcurrencyManager`, `ConnectionQueue` |
| Error Handling | Error Classification, Cleanup Sequence, Error Propagation | `ErrorInfo`, `ResourceCleanupContext` |

#### Common Term Confusion

Several terms are frequently confused during implementation:

**"Connection" vs "Socket"**: A TCP connection represents the logical communication channel, while a socket is the programming interface. Our server socket accepts connections, creating client sockets for communication.

**"Request Parsing" vs "Message Parsing"**: Request parsing specifically handles the HTTP request structure, while message parsing could apply to both requests and responses.

**"Thread Pool" vs "Connection Pool"**: Thread pools manage worker threads, while connection pools would manage reusable TCP connections (not implemented in our basic server).

**"Graceful Shutdown" vs "Graceful Degradation"**: Shutdown refers to stopping the server cleanly, while degradation means continuing operation with reduced functionality.

#### Debugging Vocabulary

When troubleshooting issues, these terms help communicate problems precisely:

- **"Partial Read"** indicates incomplete HTTP request reception
- **"Directory Traversal"** signals a security validation failure  
- **"Race Condition"** suggests thread synchronization problems
- **"Resource Exhaustion"** points to connection or thread limit issues
- **"File Descriptor Leak"** means sockets aren't being closed properly

Understanding this vocabulary accelerates both independent debugging and communication with others when seeking help.

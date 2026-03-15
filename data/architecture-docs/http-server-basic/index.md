# 🎯 Project Charter: HTTP Server (Basic)
## What You Are Building
A static-file-serving HTTP/1.1 server built from raw TCP sockets. You will write every layer by hand: a sequential accept loop that handles the socket lifecycle, an RFC 7230-compliant request parser that turns wire bytes into structured data, a five-stage filesystem security pipeline that serves files while blocking directory traversal attacks, and a bounded thread pool with HTTP/1.1 keep-alive and graceful shutdown. By the end, your server will accept concurrent connections, serve real files from a configurable document root, and shut down cleanly on SIGTERM without dropping in-flight requests.
## Why This Project Exists
Most developers interact with web infrastructure entirely through framework abstractions — they have never seen the bytes that flow between a browser and a server, never written a `recv()` loop, and never thought about why `Content-Length` must be exact. Building this from scratch exposes the physical contract underneath every HTTP request: how the OS represents a connection as a file descriptor, how TCP delivers a stream of bytes rather than messages, how a path like `/../etc/passwd` must be neutralized before touching the filesystem, and why a thread pool with a bounded queue is fundamentally different from spawning a thread per connection.
## What You Will Be Able to Do When Done
- Create a TCP server socket with `socket()`, `bind()`, `listen()`, and `accept()` and explain what each syscall does at the kernel level
- Write a partial-read-safe `recv()` loop that accumulates bytes until an HTTP delimiter is found
- Parse HTTP/1.1 request lines and headers per RFC 7230: case-insensitive names, optional whitespace stripping, obsolete-fold unfolding, and bare-LF tolerance
- Implement a five-stage path security pipeline: URL percent-decode → concatenate with document root → `realpath()` canonicalize → containment check → serve
- Block directory traversal via URL encoding (`%2e%2e%2f`), bare `../`, and symlinks inside the document root
- Detect MIME types from file extensions and serve binary files with byte-for-byte integrity
- Implement `If-Modified-Since` / `304 Not Modified` conditional request handling
- Build a bounded thread pool with a circular work queue, mutex/condition-variable synchronization, and a configurable pool size
- Implement HTTP/1.1 persistent connections with per-read idle timeouts enforced via `select()`
- Handle `SIGTERM`/`SIGINT` with an `atomic_int` flag, drain in-flight requests, and exit cleanly
- Protect shared state (connection counter, access log) from data races and verify with ThreadSanitizer
## Final Deliverable
Approximately 1,500–2,500 lines of C across 8 source files: `server.c`, `http_parse.h/c`, `file_server.h/c`, `thread_pool.h/c`, `connection.h/c`, `stats.h/c`. The server binds on a configurable port (default 8080), serves files from a configurable document root, runs a thread pool with configurable size (default 16 workers), and writes an Apache-style access log. It passes `ab -n 10000 -c 100 -k` with zero failed requests, survives ThreadSanitizer with no reported races, and returns its open file descriptor count to baseline after 10,000 sequential connections. It exits with code 0 on `SIGTERM` after completing any in-flight transfers.
## Is This Project For You?
**You should start this if you:**
- Can write and compile C programs with structs, pointers, and manual memory management
- Understand what a file descriptor is and have called `open()`/`read()`/`close()` before
- Know the difference between a process and a thread at a conceptual level
- Are comfortable reading `man` pages and `perror()` output to diagnose syscall failures
- Have seen basic networking concepts (IP address, port, TCP vs. UDP) even if you have never written socket code
**Come back after you've learned:**
- C pointers and pointer arithmetic — you will do `buf + total` and `memchr()` constantly; without this, the partial-read loop will be opaque ([Beej's Guide to C, chapters 1–5](https://beej.us/guide/bgc/))
- How to compile multi-file C projects with `gcc` and a `Makefile` — this project has 8 source files linked together
- Basic Linux command-line tools: `curl`, `nc`, `ss`, `ps`, `kill`, `ab` — all are used in the test scripts
## Estimated Effort
| Phase | Time |
|-------|------|
| Milestone 1: TCP Server & HTTP Response | ~2–4 hours |
| Milestone 2: HTTP Request Parsing | ~3–4 hours |
| Milestone 3: Static File Serving | ~4–6 hours |
| Milestone 4: Concurrent Connections | ~5–8 hours |
| **Total** | **~14–22 hours** |
## Definition of Done
The project is complete when:
- `curl -v http://localhost:8080/` returns `HTTP/1.1 200 OK` with correct `Content-Type`, `Content-Length`, and `Last-Modified` headers, and the response body matches the file on disk byte-for-byte (`md5sum` verified)
- `curl -s "http://localhost:8080/%2e%2e%2f%2e%2e%2fetc/passwd"` and a symlink-escape attempt both return `403 Forbidden`, never `200` or the contents of any file outside the document root
- `ab -n 10000 -c 100 -k http://localhost:8080/index.html` completes with `Failed requests: 0`
- The server's open file descriptor count (`ls /proc/$PID/fd | wc -l`) returns to its baseline value after 10,000 sequential connections — confirming zero FD leaks
- `gcc -fsanitize=thread` build followed by `ab -n 5000 -c 50` produces zero `ThreadSanitizer: data race` reports
- `kill -TERM $SERVER_PID` while a large file download is in progress causes the download to complete successfully (exit code 0, md5sum matches) before the server process exits with code 0

---

# 📚 Before You Read This: Prerequisites & Further Reading
> **Read these first.** The Atlas assumes you are familiar with the foundations below.
> Resources are ordered by when you should encounter them — some before you start, some at specific milestones.
---
## 🔌 TCP & The Socket API
### RFC 793 — Transmission Control Protocol
**Spec.** Postel, J. (1981). *Transmission Control Protocol.* IETF.
- **Read:** Section 1 (Introduction), Section 3.2 (Terminology), Section 3.4 (Establishing a Connection — the three-way handshake), Section 3.5 (Closing a Connection — TIME_WAIT).
- **Why:** The authoritative source for why `recv()` delivers a byte stream, not messages — the single most important concept in Milestone 1.
- **⏱ Read BEFORE starting Milestone 1.** You need the mental model of TCP as a stream before you write the first `recv()` loop.
### *The Linux Programming Interface* — Michael Kerrisk
**Book.** No Starch Press, 2010. Chapter 56 (Sockets: Introduction), Chapter 57 (Sockets: UNIX Domain), Chapter 58 (Sockets: TCP/IP Networks), Chapter 61 (Advanced Socket Topics — covers `SO_REUSEADDR`, `MSG_NOSIGNAL`, `SIGPIPE`).
- **Why:** The single most complete reference for the POSIX socket API on Linux. Chapter 61 explains TIME_WAIT, `SO_REUSEADDR`, and `SIGPIPE` in exactly the depth needed for Milestone 1's pitfall checklist.
- **⏱ Read Chapters 56–58 BEFORE Milestone 1; read Chapter 61 alongside Milestone 1 Phase 4 (when you implement `send_all`).**
### Beej's Guide to Network Programming — Brian "Beej" Hall
**Best Explanation.** https://beej.us/guide/bgnet/ — read *Section 5 (System Calls or Bust)* and *Section 6 (Client-Server Background)*.
- **Why:** The clearest introductory walkthrough of `socket()` → `bind()` → `listen()` → `accept()` that exists online, with annotated C examples that match the M1 code almost exactly.
- **⏱ Read alongside Milestone 1 Phase 1–2 (socket creation and accept loop).**
---
## 📡 HTTP/1.1 Protocol
### RFC 7230 — HTTP/1.1: Message Syntax and Routing
**Spec.** Fielding, R. & Reschke, J. (2014). IETF.
- **Read:** Section 3 (Message Format), Section 3.2 (Header Fields — OWS, case-insensitivity, obs-fold), Section 3.3 (Message Body), Section 5.4 (Host header requirement), Section 6 (Connection Management — persistent connections, keep-alive).
- **Why:** Every parser decision in Milestone 2 — OWS stripping, obs-fold unfolding, bare-LF tolerance, the `Host` header requirement, HTTP/1.1 keep-alive defaults — traces directly to a clause in this document.
- **⏱ Read Section 3 and 3.2 BEFORE Milestone 2 (HTTP parsing). Read Section 6 BEFORE Milestone 4 (keep-alive loop).**
### RFC 7231 — HTTP/1.1: Semantics and Content
**Spec.** Fielding, R. & Reschke, J. (2014). IETF.
- **Read:** Section 4.1 (GET), Section 4.3.2 (HEAD), Section 6 (Response Status Codes — 200, 304, 400, 403, 404, 414, 500, 501, 503), Section 7.1.1.1 (HTTP-date format).
- **Why:** Defines the HEAD method's requirement to return identical headers to GET, the exact HTTP-date format for `Last-Modified`, and the semantics of every status code your server sends.
- **⏱ Read Section 4.3.2 (HEAD) and Section 7.1.1.1 (HTTP-date) BEFORE Milestone 3 (file serving with `Last-Modified`/`If-Modified-Since`).**
### RFC 7232 — HTTP/1.1: Conditional Requests
**Spec.** Fielding, R. & Reschke, J. (2014). IETF.
- **Read:** Section 3.3 (`If-Modified-Since`), Section 4.1 (Evaluation of Preconditions), Section 6 (304 Not Modified).
- **Why:** Defines the exact `<=` comparison semantics for `st.st_mtime` vs. the `If-Modified-Since` timestamp, and the required headers in a 304 response.
- **⏱ Read immediately BEFORE implementing Stage 5 (conditional request check) in Milestone 3.**
---
## 🔒 Filesystem Security
### OWASP Path Traversal Attack
**Best Explanation.** https://owasp.org/www-community/attacks/Path_Traversal — read the full page, especially the "Encoding" and "Bypasses" sections.
- **Why:** Shows real-world examples of every bypass vector described in the M3 Revelation section (`%2e%2e%2f`, null bytes, double encoding, symlinks) with concrete payloads. After reading this, the five-stage security pipeline design is obvious rather than paranoid.
- **⏱ Read BEFORE starting Milestone 3.** The traversal attacks are real; understanding them motivates every design decision in the security pipeline.
### `realpath(3)` and `openat(2)` man pages — Linux man-pages project
**Spec/Code.** `man 3 realpath` and `man 2 openat` on any Linux system.
- **Why:** The `realpath(3)` man page documents the ENOENT/EACCES/ELOOP errno values that drive your 404-vs-403 logic. The `openat(2)` man page with `O_NOFOLLOW` is the reference for the TOCTOU mitigation described in the M3 design decisions table.
- **⏱ Read the `realpath(3)` page BEFORE implementing Stage 3 in Milestone 3.**
---
## ⚙️ Concurrency Primitives
### *Programming with POSIX Threads* — David Butenhof
**Book.** Addison-Wesley, 1997. Chapter 2 (Threads), Chapter 3 (Synchronization — mutex and condition variable), Chapter 5 (Advanced Synchronization — spurious wakeups, lock ordering).
- **Why:** The definitive reference for `pthread_mutex_t`, `pthread_cond_wait` with a `while` loop (not `if`), and the lock-ordering rule to prevent deadlocks. The M4 thread pool design follows the producer-consumer pattern described in Chapter 3 exactly.
- **⏱ Read Chapters 2–3 BEFORE starting Milestone 4.** The spurious wakeup section of Chapter 3 is essential before implementing `worker_thread()`'s `while` loop.**
### *The Art of Multiprocessor Programming* — Herlihy & Shavit
**Book.** Chapter 1 (Introduction — mutual exclusion), Chapter 3 (Concurrent Objects — linearizability), Chapter 10 (Monitors and Blocking Synchronization — condition variables).
- **Why:** Explains WHY condition variables require `while` instead of `if` at the theoretical level, and gives the memory model foundation for why `atomic_int` is required for cross-thread visibility of `shutdown_flag`.
- **⏱ Read Chapter 1 and the relevant sections of Chapter 10 AFTER Milestone 4 Phase 3** (after you've implemented the worker loop and want to understand the correctness argument).
### llhttp — Node.js HTTP Parser (Open Source Reference Implementation)
**Code.** https://github.com/nodejs/llhttp — read `src/llhttp.c` and `src/http.c`.
- **Why:** The production state machine HTTP parser that replaced `http_parser` in Node.js. After writing your own parser in Milestone 2, reading llhttp shows how the same state machine concept scales to a compiler-generated, zero-allocation, streaming design.
- **⏱ Read AFTER completing Milestone 2.** You'll recognize every state transition and understand the design choices llhttp makes differently from yours.
---
## 🧵 Signals & Shutdown
### *Advanced Programming in the UNIX Environment* — Stevens & Rago
**Book.** 3rd edition, Addison-Wesley. Chapter 10 (Signals), Chapter 11 (Threads — signal masking with `pthread_sigmask`).
- **Read:** Chapter 10 Sections 10.1–10.5 (signal concepts, `signal()`, async-signal safety) and the `sigwait()` coverage in Chapter 11.
- **Why:** Explains why `atomic_store` is async-signal-safe (and `printf` is not), and the `sigwait()` pattern for delivering signals synchronously to a dedicated thread — the correct production approach for multithreaded servers.
- **⏱ Read Chapter 10 BEFORE Milestone 4 Phase 7 (graceful shutdown implementation).**
---
## 📊 Performance & I/O
### *Designing Data-Intensive Applications* — Martin Kleppmann
**Book.** O'Reilly, 2017. Chapter 1, Section "Scalability" — specifically the discussion of response time percentiles and load parameters; and Chapter 11, Section "Event-Driven Architectures" (pages 460–465).
- **Why:** Explains why bounded thread pools with explicit backpressure (your 503 + `Retry-After`) are the right model under load, and frames the C10K problem that motivates the event-loop discussion in the M4 Knowledge Cascade.
- **⏱ Read AFTER completing Milestone 4,** when you've implemented the thread pool and want to understand where it sits in the broader landscape of concurrency architectures.
### `sendfile(2)` man page + nginx `sendfile` directive documentation
**Spec/Best Explanation.** `man 2 sendfile` (Linux) + https://nginx.org/en/docs/http/ngx_http_core_module.html#sendfile
- **Why:** Closes the loop on the M3 "Design Decisions" table entry for `sendfile()`. After implementing the `read()` + `send()` loop, the man page shows exactly what one syscall eliminates — the concrete implementation behind "zero-copy" file serving.
- **⏱ Read AFTER completing Milestone 3,** specifically after verifying your binary integrity tests pass.
### The C10K Problem — Dan Kegel
**Best Explanation.** http://www.kegel.com/c10k.html — the original 1999 paper that named the problem.
- **Why:** Provides historical context for why the thread pool model tops out and why `epoll` exists. After implementing your bounded thread pool in Milestone 4, this paper explains the exact performance cliff you would hit at ~10,000 connections and why nginx's architecture is different.
- **⏱ Read AFTER completing Milestone 4** as a "what comes next" framing document.

---

# HTTP Server (Basic)

This project builds a static-file-serving HTTP/1.1 server from raw TCP sockets up through concurrent connection handling. You will write every layer by hand: the socket lifecycle (bind/listen/accept), the HTTP message parser, the filesystem security layer, and the threading model. Nothing is handed to you by a framework — you implement the protocol, the file serving, and the concurrency yourself.

The project exposes the physical reality underneath every web request. When nginx serves a file, it does exactly what you will do here: resolve a socket, parse bytes off the wire into a structured request, canonicalize a filesystem path, read file bytes, format an HTTP response, and manage a pool of threads or event loops to handle many clients simultaneously. By building this yourself you learn not just what happens but why each step exists and what goes wrong when it is skipped.

The four milestones form a dependency chain: TCP socket infrastructure → HTTP protocol parsing → filesystem serving with security → concurrent connection management. Each milestone builds on the previous and introduces a new layer of the hardware/OS/protocol stack. At the end you will have a production-credible server core that handles keep-alive, directory traversal prevention, conditional requests, graceful shutdown, and bounded concurrency.


<!-- MS_ID: http-server-basic-m1 -->
# Milestone 1: TCP Server & HTTP Response
## Where We Are

![Project Atlas: HTTP Server Component Map](./diagrams/diag-l0-satellite-map.svg)

Before you write a single line of HTTP parsing, you need the plumbing that carries every byte to and from a client. This milestone is about that plumbing: the socket lifecycle. You will create a TCP server that accepts connections, reads raw bytes off the wire, and sends a hardcoded HTTP response back. No parsing, no file serving, no threads — just the skeleton that every web server in existence is built on.
By the end of this milestone, you will have a running server you can `curl`. More importantly, you will understand *why* each system call exists and *what goes wrong* when you skip it. That understanding is what separates a server that works on localhost from one that survives a production network.
---
## The Revelation: TCP Is Not a Message Bus
Here is something almost every developer gets wrong the first time they write a network server.
You call `recv()`. You get back a buffer full of bytes. You glance at it, see `GET / HTTP/1.1\r\n`, and think: *great, I have the request.* You parse it and send the response.
This works on your laptop. It works in every test. You ship it. Three weeks later, a user on a mobile network with 200ms latency reports that the server hangs. A load test shows random failures under high request volume. You are confused — the code looks right.
The bug is in an assumption so deep you never thought to question it: that `recv()` delivers a complete request.
**TCP does not deliver messages. TCP delivers a stream of bytes.** The kernel may give you 1 byte or 4000 bytes in a single `recv()` call, depending on factors that have nothing to do with how the sender wrote the data: network fragmentation, Nagle's algorithm buffering small writes together, OS scheduler timing, receive buffer state. A single `write()` on the sender's side can arrive as two, three, or twenty `recv()` calls on your side — or as one call that contains half of one request and the beginning of the next.
[[EXPLAIN:partial-reads-on-stream-sockets-—-tcp-is-a-byte-stream,-not-a-message-protocol;-a-single-recv()-may-return-1-byte-or-n-bytes;-correct-code-always-loops|Partial reads on stream sockets — TCP is a byte stream, not a message protocol; a single recv() may return 1 byte or N bytes; correct code always loops]]
This is the single most important concept in this milestone. Every piece of code you write in this chapter flows from it. The `recv()` loop, the delimiter search, the buffer accumulation strategy — all of it exists because TCP is a stream.

![Partial Read Problem: Why One recv() Is Never Enough](./diagrams/diag-m1-partial-read-loop.svg)

---
## The Socket Lifecycle
Every network connection you will ever write in C passes through the same four-step ceremony. Understand these steps completely once, and every server you ever build will feel familiar.

![TCP Socket Lifecycle: socket() → bind() → listen() → accept() → close()](./diagrams/diag-m1-socket-lifecycle.svg)

### Step 1: `socket()` — Creating the Endpoint
```c
int server_fd = socket(AF_INET, SOCK_STREAM, 0);
if (server_fd < 0) {
    perror("socket");
    exit(EXIT_FAILURE);
}
```
[[EXPLAIN:socket-system-calls-as-file-descriptors-—-socket()-returns-an-int-fd;-all-i/o-uses-the-same-read/write/close-interface|Socket system calls as file descriptors — socket() returns an int FD; all I/O uses the same read/write/close interface]]
`socket()` returns an integer. That integer is a **file descriptor** — the kernel's universal handle for I/O resources. The same integer type used for files (`open()`), pipes (`pipe()`), and terminals is used for network connections. This is not a coincidence; it is the Unix design philosophy: everything is a file. Once you have a socket FD, you can call `read()`, `write()`, and `close()` on it, just as you would on a regular file. Later you will use `recv()` and `send()` instead of `read()`/`write()` because they expose socket-specific flags, but at the kernel level it is the same mechanism.
The arguments to `socket()` specify what kind of connection you want:
- `AF_INET` — IPv4 address family (use `AF_INET6` for IPv6)
- `SOCK_STREAM` — a reliable, ordered byte stream (TCP); contrast with `SOCK_DGRAM` for UDP datagrams
- `0` — let the kernel choose the appropriate protocol for the address family and socket type (it will choose TCP)

![Kernel File Descriptor Table: Listening FD vs. Client FD](./diagrams/diag-m1-fd-table.svg)

### Step 2: `bind()` — Claiming an Address
```c
struct sockaddr_in addr = {0};
addr.sin_family = AF_INET;
addr.sin_addr.s_addr = INADDR_ANY;  // Accept connections on all interfaces
addr.sin_port = htons(8080);        // Port 8080, in network byte order
if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    perror("bind");
    exit(EXIT_FAILURE);
}
```
`bind()` tells the kernel which address and port this socket owns. Without it, the socket exists but has no address — clients have nowhere to connect to.
[[EXPLAIN:network-byte-order-and-htons/htonl-—-cpus-are-little-endian,-network-protocol-is-big-endian;-all-port-and-address-fields-must-be-converted|Network byte order and htons/htonl — CPUs are little-endian, network protocol is big-endian; all port and address fields must be converted]]
Notice `htons(8080)`. The `h` is "host," the `n` is "network," the `s` is "short" (16-bit). Your CPU stores multi-byte numbers with the least significant byte first (little-endian on x86). Network protocols store them with the most significant byte first (big-endian). If you write `addr.sin_port = 8080` without `htons()`, the bytes get swapped and you bind to port 8224 (0x2020) instead of port 8080 (0x1F90). The server starts, `bind()` succeeds, and nothing can connect to it — a bug that is genuinely difficult to diagnose without knowing this context.
`INADDR_ANY` (value `0.0.0.0`) means "accept connections on any network interface." If you have multiple network cards or a loopback interface, the server will accept connections arriving on any of them. For development this is almost always what you want.
#### The SO_REUSEADDR Option
Add this before `bind()`:
```c
int opt = 1;
if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
    perror("setsockopt SO_REUSEADDR");
    exit(EXIT_FAILURE);
}
```
Without `SO_REUSEADDR`, restarting your server within about 60 seconds of stopping it will fail with `bind: Address already in use`. This happens because TCP has a state called TIME_WAIT [[EXPLAIN:tcp-time_wait-state-—-after-a-connection-closes,-the-port-stays-in-time_wait-for-2msl-(~60s)-to-absorb-delayed-packets;-so_reuseaddr-bypasses-this-for-servers|TCP TIME_WAIT state — after a connection closes, the port stays in TIME_WAIT for ~60s to absorb delayed packets; SO_REUSEADDR bypasses this for servers]] — the old socket occupies the port while the kernel waits for any delayed packets to arrive and be discarded. `SO_REUSEADDR` tells the kernel that it is safe for a new socket to claim the same address, which is always true for server sockets (not for the client-side FD that's actually in TIME_WAIT). Every production server sets this option without exception.
### Step 3: `listen()` — Opening for Business
```c
if (listen(server_fd, SOMAXCONN) < 0) {
    perror("listen");
    exit(EXIT_FAILURE);
}
```
`listen()` converts the socket from an active socket (the kind that initiates connections) into a **passive socket** (the kind that accepts them). It also creates two internal queues in the kernel:
- **SYN queue** (also called the incomplete connections queue): holds connections that have received the client's SYN packet but not yet completed the three-way handshake
- **Accept queue** (completed connections queue): holds connections that have completed the handshake and are waiting for your `accept()` call
The second argument is the **backlog** — the maximum size of the accept queue. `SOMAXCONN` uses the system maximum (typically 128 on Linux, tunable via `/proc/sys/net/core/somaxconn`). If the queue fills up, new connection attempts from clients are silently dropped — the client will retry, but a fully saturated server under load will start refusing connections. For this milestone, the default is fine.
### Step 4: `accept()` — Receiving a Client
```c
while (1) {
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
    if (client_fd < 0) {
        perror("accept");
        continue;  // Log and keep running; don't exit the server
    }
    // Handle the client...
    handle_client(client_fd);
    close(client_fd);
}
```
`accept()` **blocks** your process until a client connects. When a client connects, it dequeues the completed connection from the kernel's accept queue and returns a **new file descriptor** — the client FD. This is the most important distinction in the socket API:
- `server_fd` is your **listening socket**. It exists for the lifetime of the server. You never read from it or write to it.
- `client_fd` is a **connected socket**. Each call to `accept()` creates a new one. You read requests from it, write responses to it, and then `close()` it.
Every `client_fd` you accept consumes one entry in your process's file descriptor table. On most Linux systems, a process starts with a limit of 1024 open file descriptors (`ulimit -n`). If you forget to `close(client_fd)` after serving a client, your server leaks one FD per connection. After ~1020 connections (accounting for stdin, stdout, stderr, and `server_fd`), `accept()` returns -1 with `EMFILE: Too many open files` and your server stops accepting new connections. This is a real production failure mode. `close()` every client FD when you are done with it.
---
## The Request Reading Loop
Now you have a `client_fd`. The client is sending you an HTTP request. You need to read it.
Here is the wrong way:
```c
// DON'T DO THIS
char buf[4096];
ssize_t n = recv(client_fd, buf, sizeof(buf), 0);
buf[n] = '\0';
// parse buf as if it's a complete request — WRONG
```
This fails silently on:
- Slow clients that send headers byte by byte
- Large requests whose headers exceed a single TCP segment
- Any network path that fragments packets
Here is the right mental model: you are reading from a water pipe. The water (bytes) arrives continuously. You need to stop reading at a specific point — the end of the HTTP headers, which is signaled by the delimiter `\r\n\r\n` (carriage return, line feed, carriage return, line feed). You must keep reading from the pipe, accumulating bytes into a buffer, until you see that delimiter.
```c
#define REQUEST_BUF_SIZE 8192
// Returns the total number of bytes read, or -1 on error.
// On success, buf contains the full request up to and including \r\n\r\n.
ssize_t read_request(int fd, char *buf, size_t buf_size) {
    size_t total = 0;
    while (total < buf_size - 1) {
        ssize_t n = recv(fd, buf + total, buf_size - 1 - total, 0);
        if (n < 0) {
            perror("recv");
            return -1;
        }
        if (n == 0) {
            // Client closed the connection before sending a complete request
            return -1;
        }
        total += n;
        buf[total] = '\0';  // Keep the buffer null-terminated for safe string ops
        // Search for the end-of-headers delimiter
        if (strstr(buf, "\r\n\r\n") != NULL) {
            return (ssize_t)total;
        }
    }
    // Buffer full without finding delimiter: request too large
    return -1;
}
```
Walk through this carefully:
1. **`recv()` into `buf + total`**: each call appends to where the last one left off. You are building up a contiguous buffer across multiple `recv()` calls.
2. **`n == 0` means EOF**: the client closed its side of the connection. On TCP, a `recv()` that returns 0 means the peer performed an orderly shutdown. Your server must handle this gracefully — it is not an error.
3. **`n < 0` means error**: check `errno`. `EINTR` (interrupted by signal) is often worth retrying; other errors should terminate the connection.
4. **`buf_size - 1 - total`**: you always leave room for the null terminator and never overflow the buffer. This is the guard against a client that sends headers forever to exhaust your memory.
5. **`strstr(buf, "\r\n\r\n")`**: scan the accumulated buffer for the end-of-headers marker. HTTP/1.1 uses CRLF line endings, so headers end with `\r\n\r\n`. Only when you find this delimiter do you have a complete set of headers.


### The Three-Level View of a `recv()` Call
What actually happens when your code calls `recv()`?
**Application Level**: You call `recv(client_fd, buf, 4096, 0)`. Your thread blocks.
**OS/Kernel Level**: The kernel checks the socket's receive buffer (a kernel-managed region of memory, typically 87KB by default on Linux). If the receive buffer has data, the kernel copies bytes from it into your `buf` and returns immediately. If it is empty, the kernel marks your thread as blocked and switches to another thread. When new data arrives on the network interface, the network interrupt handler wakes your thread.
**Hardware Level**: The NIC (Network Interface Card) receives an Ethernet frame, performs a DMA (Direct Memory Access) transfer of the frame data directly into kernel memory without CPU involvement, then raises an interrupt. The interrupt handler updates the socket's receive buffer. If your thread was sleeping, the scheduler marks it runnable.
The critical insight: by the time `recv()` returns, it has copied whatever bytes were available in the kernel receive buffer *at that instant*. The fact that the sender called `send()` with 1400 bytes does not mean those 1400 bytes will arrive at your `recv()` as one unit. TCP may reassemble them from two IP packets, and those packets may arrive milliseconds apart. Between those two arrivals, your `recv()` can return with only the first packet's data.
---
## The HTTP Response
You have read a request (for now, you do not parse it). Time to send a response. An HTTP/1.1 response is plain text bytes with a specific structure.

![Three-Level View: One HTTP Request Across Application / OS / Hardware](./diagrams/diag-cross-layer-three-level-view.svg)

![HTTP/1.1 Response Wire Format: Byte Anatomy](./diagrams/diag-m1-http-response-wire.svg)

```c
static const char *HTTP_RESPONSE =
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: text/html; charset=utf-8\r\n"
    "Content-Length: 27\r\n"
    "Connection: close\r\n"
    "\r\n"
    "<h1>Hello from C server!</h1>";
```
The anatomy:
- **Status line**: `HTTP/1.1 200 OK\r\n` — protocol version, status code, reason phrase
- **Headers**: one per line, `Name: Value\r\n`, case-insensitive names
- **Blank line**: `\r\n` — the mandatory separator between headers and body; the same `\r\n\r\n` your reading loop searches for on the request side
- **Body**: the HTML content; its byte length must exactly match the `Content-Length` header
`Content-Length` must be byte-accurate. If it is too small, the client truncates the body. If it is too large, the client hangs waiting for bytes that never arrive. `Content-Type` tells the browser how to interpret the body — without it, browsers default to `application/octet-stream` and offer a download dialog instead of rendering HTML.
### Sending the Response Safely
Sending the response has the same partial-write problem as receiving has partial-reads. The kernel may not write all bytes in a single `send()` call if the socket's send buffer is full (for large files, this matters a great deal). Use a write loop:
```c
ssize_t send_all(int fd, const char *buf, size_t len) {
    size_t total_sent = 0;
    while (total_sent < len) {
        ssize_t n = send(fd, buf + total_sent, len - total_sent, MSG_NOSIGNAL);
        if (n < 0) {
            perror("send");
            return -1;
        }
        if (n == 0) {
            // Should not happen with blocking sockets, but guard anyway
            return -1;
        }
        total_sent += n;
    }
    return (ssize_t)total_sent;
}
```
Notice `MSG_NOSIGNAL`. This flag is doing critical work.
---
## Surviving Client Disconnection: SIGPIPE

![SIGPIPE: What Happens When You Write to a Closed Connection](./diagrams/diag-m1-sigpipe-flow.svg)

Imagine a client connects, sends a request, and then immediately closes the connection (or the network drops). Before you know the client is gone, you try to `send()` the response. The first `send()` may succeed — the kernel buffers the bytes. The second `send()` discovers the peer has closed and returns `-1` with `errno == EPIPE`. But there is a problem: before `send()` returns, the kernel also **sends `SIGPIPE` to your process**.
By default, `SIGPIPE` **terminates the process**. Your server dies silently on any client disconnection that happens mid-response. This is the same signal that kills a shell pipeline when `head` exits early — write to a closed pipe, get `SIGPIPE`.
You have two choices:
**Option 1: `MSG_NOSIGNAL` flag** (preferred, per-call, Linux-specific):
```c
ssize_t n = send(fd, buf, len, MSG_NOSIGNAL);
// SIGPIPE is suppressed; broken pipe returns -1/EPIPE instead
```
**Option 2: Ignore `SIGPIPE` process-wide** (simpler for a server):
```c
#include <signal.h>
// Call this once at startup, before any sockets are created
signal(SIGPIPE, SIG_IGN);
// All SIGPIPE signals are now ignored; send() returns -1/EPIPE on broken write
```
For a server, `SIG_IGN` is almost always the right choice. You do not want any thread in your process to crash because one client closed their browser tab. After ignoring `SIGPIPE`, check `errno == EPIPE` in your `send()` loop and handle it as a normal "client disconnected" condition.
> On macOS/BSD, `MSG_NOSIGNAL` is not available; use `SO_NOSIGPIPE` socket option or `signal(SIGPIPE, SIG_IGN)` instead.
---
## Putting It Together: The Complete Server
Here is the full minimal server implementing everything above:
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#define DEFAULT_PORT    8080
#define BACKLOG         128
#define REQUEST_BUF_SIZE 8192
static const char *HTTP_RESPONSE =
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: text/html; charset=utf-8\r\n"
    "Content-Length: 27\r\n"
    "Connection: close\r\n"
    "\r\n"
    "<h1>Hello from C server!</h1>";
ssize_t send_all(int fd, const char *buf, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, buf + sent, len - sent, MSG_NOSIGNAL);
        if (n <= 0) return -1;
        sent += n;
    }
    return (ssize_t)sent;
}
ssize_t read_request(int fd, char *buf, size_t buf_size) {
    size_t total = 0;
    while (total < buf_size - 1) {
        ssize_t n = recv(fd, buf + total, buf_size - 1 - total, 0);
        if (n <= 0) return -1;   // 0 = EOF, <0 = error
        total += n;
        buf[total] = '\0';
        if (strstr(buf, "\r\n\r\n") != NULL) return (ssize_t)total;
    }
    return -1;  // Buffer exhausted without end-of-headers
}
void handle_client(int client_fd) {
    char buf[REQUEST_BUF_SIZE];
    ssize_t req_len = read_request(client_fd, buf, sizeof(buf));
    if (req_len < 0) {
        fprintf(stderr, "Failed to read request\n");
        return;
    }
    // Log the first line of the request for debugging
    char *line_end = strstr(buf, "\r\n");
    if (line_end) *line_end = '\0';
    printf("Request: %s\n", buf);
    if (line_end) *line_end = '\r';
    size_t response_len = strlen(HTTP_RESPONSE);
    if (send_all(client_fd, HTTP_RESPONSE, response_len) < 0) {
        fprintf(stderr, "Failed to send response\n");
    }
}
int create_server_socket(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); exit(EXIT_FAILURE); }
    int opt = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    struct sockaddr_in addr = {0};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port);
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        exit(EXIT_FAILURE);
    }
    if (listen(fd, BACKLOG) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }
    return fd;
}
int main(int argc, char *argv[]) {
    int port = DEFAULT_PORT;
    if (argc == 2) port = atoi(argv[1]);
    // Ignore SIGPIPE process-wide; broken writes return EPIPE instead of killing the server
    signal(SIGPIPE, SIG_IGN);
    int server_fd = create_server_socket(port);
    printf("Listening on port %d\n", port);
    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) {
            perror("accept");
            continue;  // Non-fatal: log and keep the server running
        }
        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip));
        printf("Connection from %s:%d\n", client_ip, ntohs(client_addr.sin_port));
        handle_client(client_fd);
        close(client_fd);  // MANDATORY: releases the file descriptor
    }
    close(server_fd);
    return 0;
}
```
### Testing Your Server
Compile and run:
```bash
gcc -Wall -Wextra -o server server.c
./server 8080
```
In another terminal:
```bash
curl -v http://localhost:8080/
```
You should see the full HTTP exchange: curl sending the request headers, your server responding with `200 OK` and the HTML body. The `-v` flag shows every byte exchanged — use it to verify your response headers are correctly formatted.
Test partial-read safety with `telnet` (which sends bytes character-by-character):
```bash
telnet localhost 8080
```
Then type the request manually:
```
GET / HTTP/1.1
Host: localhost
```
(Press Enter twice at the end to send the blank line.) Your read loop should accumulate the bytes and respond correctly.
Test SIGPIPE handling:
```bash
# Send a request and immediately close — nc exits as soon as it receives the response
echo -e "GET / HTTP/1.1\r\nHost: localhost\r\n\r\n" | nc -q 0 localhost 8080
```
Your server should not crash between requests.
---
## The Hardware Soul
Let us look at what happens in the hardware when your server calls `accept()` and `recv()`.
**File Descriptor Table**: `server_fd` and each `client_fd` are entries in your process's file descriptor table, a kernel-managed array of pointers to `file` structs. Each `file` struct points to a `socket` struct which holds the send/receive buffers. At the hardware level, a file descriptor is an index — a 32-bit integer that is an index into this kernel array. The kernel's table for your process is typically cache-hot because the accept loop accesses the same `server_fd` entry repeatedly.
**`accept()` cold vs. hot path**: The first `accept()` call per connection touches the kernel's accept queue, which lives in kernel memory separate from your process. This is a **cold cache miss** — the queue entry was created by the NIC interrupt handler, not your process. By the time `accept()` returns `client_fd`, the socket's receive buffer may already have bytes in it (the client was fast), making the first `recv()` a hot-path kernel-to-user copy.
**`recv()` memory copy**: Every `recv()` performs a copy from kernel socket receive buffer (kernel virtual memory) to your `buf` (user virtual memory). These are different memory regions — the kernel cannot write directly to your stack because kernel code runs with different privilege and different page tables. This copy is your unavoidable baseline cost. For large responses, `sendfile()` (Milestone 3+) eliminates the analogous copy on the send side by keeping data in kernel space.
**Branch prediction**: Your `recv()` loop has a branch on `n <= 0` (error/EOF check) and a branch on `strstr` success. Both are **predictable**: `n > 0` is the common case, and `strstr` returns non-NULL only on the final iteration. Modern CPUs will predict "not found yet" throughout the loop body, paying a misprediction penalty only once per request — negligible.
**Sequential memory access**: `buf + total` writes sequentially into a stack buffer. Sequential writes are the most cache-friendly access pattern possible — the hardware prefetcher will load the next cache line ahead of your write pointer. No cache misses after the first write.
---
## Design Decision: Why a Single Buffer?
In this milestone you accumulate the entire request into one `char buf[8192]`. That is the right choice for now. Let's understand the tradeoff explicitly:
| Approach | Pros | Cons | Used By |
|---|---|---|---|
| **Single fixed buffer (chosen)** ✓ | Simple, stack-allocated, no malloc, cache-local | Max request size limited to buffer size; body not handled | nginx (initial read), this milestone |
| Dynamic growing buffer | No size limit, handles large uploads | `malloc`/`realloc` required, fragmentation risk | curl, libhttp |
| Ring buffer | Efficient for streaming parsers, enables zero-copy parsing | Complex pointer arithmetic, harder to null-terminate | libuv, Nginx (internal) |
8192 bytes (8 KB) is enough for any realistic set of HTTP headers. RFC 7230 does not mandate a limit but 8 KB is the de facto standard — nginx's default is 8 KB per buffer. For request bodies (POST uploads), you will need to read beyond the headers using `Content-Length`, but that is Milestone 2's problem.
---
## Knowledge Cascade — What You Just Unlocked
Understanding the socket lifecycle and partial-read model opens a remarkably wide door.
**1. Every network library has a read buffer.** libuv (Node.js internals), Boost.Asio (C++), Netty (Java), Tokio (Rust) — they all maintain an internal buffer and call your application-level parser only after detecting a message boundary. Now you understand *why*: it is the only correct way to handle a byte stream. When you see `onData(buf)` callbacks in event loop frameworks, you know the framework is doing the `recv()` loop for you. The partial-read problem does not disappear — it gets encapsulated.
**2. File descriptors are the universal handle.** The same FD mechanism that gives you TCP sockets gives you `timerfd_create()` (a timer as an FD), `eventfd()` (an event counter as an FD), `inotify` (filesystem events as an FD), and `io_uring` (a submission/completion ring as FDs). In Milestone 4 you will use `epoll`, which watches a *set of FDs* for readability — and it works on every kind of FD because they all implement the same kernel `poll` interface. File descriptors are the abstraction that makes `select`/`poll`/`epoll` possible.
**3. The `htons()` byte-order conversion you just used is the same problem that causes bugs in every binary protocol.** MySQL's wire protocol, PostgreSQL's protocol, DNS packets, TLS record headers — every field wider than one byte has an endianness. When a database client library fails silently or a DNS resolver returns garbage, byte order is among the first suspects. The muscle memory of "always convert multi-byte fields at protocol boundaries" transfers directly.
**4. SIGPIPE connects sockets to Unix pipes.** The same broken-write signal that kills your server when a client disconnects is the signal that terminates `grep` when `head` exits in `grep pattern bigfile | head -n 10`. Sockets and pipes share the same signal model because they are both implemented using the same kernel abstraction: a pair of file descriptions with a read end and a write end. When you ignore `SIGPIPE` in your server, you are making the same architectural decision that long-running daemons make universally.
**5. FD limits are a production concern from day one.** `ulimit -n 1024` is the default on many Linux systems. This is why `nginx -c nginx.conf` recommends `worker_rlimit_nofile 65535;` and Redis docs say to set `ulimit -n 65536` before starting. Every connection costs one FD. When your server starts returning `EMFILE` under load, it is because you either leaked FDs (forgot `close()`) or hit the OS limit. Both are real, both have killed production services.
---
## Common Pitfalls Checklist
Before moving to Milestone 2, verify your implementation handles all of these:
- [ ] `SO_REUSEADDR` is set before `bind()` — server can restart without 60-second wait
- [ ] `htons()` is used for the port number — correct port binding
- [ ] `SIGPIPE` is ignored or `MSG_NOSIGNAL` is used — server survives client disconnects
- [ ] `recv()` loop continues until `\r\n\r\n` is found — partial-read safe
- [ ] Buffer size limit enforced — no unbounded reads
- [ ] `close(client_fd)` is called after every response — no FD leaks
- [ ] `accept()` errors use `continue`, not `break` — server keeps running after a single bad accept
- [ ] `recv()` returning 0 (client closed) is handled — no null-dereference on empty buffer
- [ ] Response `Content-Length` matches the actual body byte count — clients don't hang
---
<!-- END_MS -->


<!-- MS_ID: http-server-basic-m2 -->
<!-- MS_ID: http-server-basic-m2 -->
# Milestone 2: HTTP Request Parsing
## Where We Are


In Milestone 1 you built the plumbing: a TCP server that accepts a connection, accumulates raw bytes into a buffer until it sees `\r\n\r\n`, and sends a hardcoded response. Your server speaks the socket language fluently. It does not yet speak HTTP.
This milestone is about turning that opaque buffer of bytes into something your code can reason about: a structured `http_request_t` containing a method, a path, a version, and a map of headers. The socket layer handed you a stream. You are now going to impose grammar on it.
---
## The Revelation: HTTP Parsing Is a State Machine, Not a Split
Here is the assumption almost every developer carries into their first HTTP parser:
> "Headers are just lines. Split on newline. Split each line on colon. Done. Five minutes of work."
This assumption is almost right — which makes it dangerous. A parser built on `strtok()` or a sequence of `split()` calls will work correctly on every request a modern browser sends from your laptop. It will silently mishandle requests from mobile proxies, curl with unusual flags, or deliberately crafted test inputs. And in production, those are exactly the inputs you will encounter.
RFC 7230 specifies five distinct challenges that make naive string splitting incorrect. Here they are, in order of the pain they cause when ignored:
**1. Case-insensitive header names.** `Content-Type`, `content-type`, and `CONTENT-TYPE` are the same header. Your `strcmp()` call treats them as three different headers. A simple `strcasecmp()` fix is not enough either — you need a case-insensitive *storage* strategy, not just comparison.
**2. Optional whitespace around the colon.** RFC 7230 Section 3.2 specifies that each header field looks like `field-name ":" OWS field-value OWS`. The OWS ("optional whitespace") is zero or more spaces or tabs before and after the value. A client may send `Content-Length: 42`, `Content-Length:42`, or `Content-Length:   42  `. Your parser must strip that whitespace.
**3. Obsolete line folding (obs-fold).** The original HTTP/1.0 specification allowed a header value to span multiple lines if the continuation line started with a space or tab:
```
Subject: This is
  a folded header value
```
RFC 7230 deprecates this but requires that servers either unfold it (replace `\r\n <whitespace>` with a single space) or reject it with 400 Bad Request. Real clients, particularly older Java HTTP libraries and misconfigured proxies, still send folded headers. If your parser splits blindly on newlines, it will treat the second line as a new header with no name and produce a malformed parse.
**4. Mixed CRLF and bare LF.** The spec requires `\r\n` (carriage return + line feed) as the line terminator. Real clients — particularly `telnet`, certain Python `http.client` versions under edge cases, and some embedded devices — send bare `\n`. A parser that only looks for `\r\n` will fail silently on these clients, treating the entire request as a single unparsed line.
**5. No length prefix — delimiter scanning under size limits.** Unlike binary protocols (which prefix every field with its length), HTTP headers have no length prefix. The end of the header section is signaled by an empty line. You must scan for this delimiter while simultaneously enforcing a maximum buffer size to prevent a client from sending headers forever and exhausting your memory. This interleaving of "scan for delimiter" and "enforce size limit" is inherently stateful.
Put these five together and you realize: a correct HTTP header parser is a small **state machine** that processes bytes one logical unit at a time, with explicit state transitions for each case.

![HTTP Request Parser: State Machine](./diagrams/diag-m2-parser-state-machine.svg)

The revelation is not that HTTP is complex. The revelation is that **every text-based protocol forces you to write a state machine**. JSON parsers, CSS tokenizers, SMTP command parsers, Redis protocol parsers — they all face the same five challenges in different forms. HTTP is your first encounter with this pattern. Build it right here, and you will recognize it instantly the next time.
---
## The Wire Format
Before you can parse, you need to see what you are parsing. Let us look at a complete HTTP/1.1 request at the byte level.

![HTTP/1.1 Request Wire Format: Annotated Byte Map](./diagrams/diag-m2-http-request-anatomy.svg)

```
GET /index.html HTTP/1.1\r\n
Host: localhost:8080\r\n
User-Agent: curl/7.88.1\r\n
Accept: */*\r\n
\r\n
```
Breaking this apart:
- **Request line**: `GET /index.html HTTP/1.1\r\n` — exactly one line containing method, path, and version, separated by single spaces, terminated by CRLF.
- **Header lines**: zero or more `Name: Value\r\n` lines, each containing a colon separator and optional whitespace around the value.
- **Empty line**: `\r\n` — a CRLF on its own line, signaling the end of headers. This is the `\r\n\r\n` your M1 `read_request()` function searched for: the final header line's `\r\n` plus this empty line's `\r\n`.
- **Body**: for requests with a body (POST/PUT), bytes follow the empty line. Their length is specified by the `Content-Length` header.
The request line and each header line are structured differently. Your parser needs separate logic for each.
---
## Designing `http_request_t`
Before writing parsing code, design the data structure you are parsing *into*. This is always the right order: define the output structure, then write code to fill it.
```c
#define MAX_HEADERS    32
#define MAX_PATH_LEN   8192
#define MAX_METHOD_LEN 16
#define MAX_VERSION_LEN 16
typedef struct {
    char name[128];
    char value[1024];
} http_header_t;
typedef struct {
    char       method[MAX_METHOD_LEN];   // "GET", "HEAD", "POST", ...
    char       path[MAX_PATH_LEN];       // "/index.html", "/api/data", ...
    char       version[MAX_VERSION_LEN]; // "HTTP/1.1"
    http_header_t headers[MAX_HEADERS];
    int        header_count;
    const char *body;                    // Points into the raw buffer; NOT a copy
    size_t     body_len;                 // 0 if no Content-Length or body
} http_request_t;
```

![http_request_t Struct Memory Layout](./diagrams/diag-m2-header-map-layout.svg)

Several design decisions are worth examining explicitly:
**Fixed-size header arrays**: `http_header_t headers[MAX_HEADERS]` with `MAX_HEADERS = 32`. Real HTTP requests rarely exceed 20 headers. This avoids dynamic allocation (`malloc`) and keeps the entire `http_request_t` on the stack or in a single allocation. The tradeoff is that requests with more than 32 headers will be truncated — which is a safe failure mode (you log and continue) rather than a memory-growth failure. Apache httpd uses 100 as its default; nginx uses 64. 32 is adequate for a static file server.
**`body` as a pointer into the raw buffer**: The body field is not a copy. It points directly into the `char buf[8192]` that M1's `read_request()` filled. This is a **zero-copy** strategy — no `malloc`, no `memcpy`. The constraint is that the raw buffer must outlive the `http_request_t`. Since both live in `handle_client()`'s stack frame for the same connection, this is safe. Mark it `const char *` to signal "this memory is not yours to modify."
**Header names as fixed-size char arrays**: `char name[128]`. In practice, HTTP header names are short (the longest standard ones, like `Transfer-Encoding`, are 17 characters). 128 bytes is generous. If you encounter a header name longer than 127 characters, it is either a buggy client or a malicious probe — reject with 400 Bad Request.
---
## Parsing the Request Line
The request line is the first line of the HTTP request: `METHOD PATH VERSION\r\n`. Parse it first, before touching headers.
```c
// Returns 0 on success, -1 on error.
// Sets request->method, request->path, request->version.
int parse_request_line(const char *line, size_t line_len,
                       http_request_t *request) {
    // Find the two space delimiters
    const char *first_space = memchr(line, ' ', line_len);
    if (!first_space) return -1;  // No space found → malformed
    size_t method_len = first_space - line;
    if (method_len == 0 || method_len >= MAX_METHOD_LEN) return -1;
    const char *path_start = first_space + 1;
    size_t remaining = line_len - (path_start - line);
    const char *second_space = memchr(path_start, ' ', remaining);
    if (!second_space) return -1;  // No second space → malformed
    size_t path_len = second_space - path_start;
    if (path_len == 0) return -1;  // Empty path → malformed
    // Enforce URI length limit: 414 URI Too Long
    if (path_len >= MAX_PATH_LEN) return -2;  // Caller checks for -2 → send 414
    const char *version_start = second_space + 1;
    size_t version_len = line_len - (version_start - line);
    // Strip trailing \r if present (handles bare \n line endings)
    if (version_len > 0 && version_start[version_len - 1] == '\r') {
        version_len--;
    }
    if (version_len == 0 || version_len >= MAX_VERSION_LEN) return -1;
    // Copy into the struct
    memcpy(request->method,  line,          method_len);
    request->method[method_len] = '\0';
    memcpy(request->path, path_start, path_len);
    request->path[path_len] = '\0';
    memcpy(request->version, version_start, version_len);
    request->version[version_len] = '\0';
    return 0;
}
```
Three things to notice:
**`memchr()` instead of `strchr()`**: `memchr(line, ' ', line_len)` scans exactly `line_len` bytes for a space. `strchr(line, ' ')` scans until it finds a null terminator. Since `line` comes from a network buffer that might not be null-terminated at the right place, `strchr` could read past the end of the line. [[EXPLAIN:memchr-vs-strchr-safety-—-working-with-length-bounded-buffers-from-the-network-requires-using-mem*-functions-not-str*-functions-to-avoid-reading-past-the-valid-data|memchr vs strchr safety — working with length-bounded buffers from the network requires using mem* functions not str* functions to avoid reading past valid data]]
**Return code -2 for 414**: The `parse_request_line` function uses -1 for generic malformation (→ 400 Bad Request) and -2 specifically for oversized paths (→ 414 URI Too Long). The caller inspects the return code and chooses the error response accordingly. You could use an enum for clarity in a larger codebase.
**Stripping trailing `\r`**: When a client sends bare LF line endings (`\n` without `\r`), the version field will still have its `\r` from the original CRLF — wait, actually the opposite: if the client sends `GET / HTTP/1.1\n`, the line will end in `\n` only, so there is no `\r` to strip. But if the client sends `GET / HTTP/1.1\r\n` and your line-splitting logic includes the `\r` in the line content, you need to strip it. The safest approach: always check for and strip a trailing `\r` from any line before using its content.
---
## Parsing Headers: The State Machine
Here is the header parsing function that handles all five RFC challenges. Read through it carefully — each branch corresponds to one of the five challenges from the Revelation section.
```c
// Parses headers from raw_headers (the bytes after the request line).
// Modifies raw_headers by null-terminating each header name and value
// (working in-place on the buffer from read_request()).
// Returns the number of headers parsed, or -1 on error.
int parse_headers(char *raw_headers, size_t raw_len,
                  http_request_t *request) {
    request->header_count = 0;
    char *pos = raw_headers;
    char *end = raw_headers + raw_len;
    while (pos < end && request->header_count < MAX_HEADERS) {
        // Find the end of this line (look for \n, accept with or without preceding \r)
        char *line_end = memchr(pos, '\n', end - pos);
        if (!line_end) break;  // No more complete lines
        // Calculate line content (excluding \n, and \r if present)
        char *line = pos;
        size_t line_len = line_end - pos;
        if (line_len > 0 && line[line_len - 1] == '\r') {
            line_len--;  // Strip \r for CRLF → bare LF normalization
        }
        // Empty line signals end of headers
        if (line_len == 0) {
            // pos after the empty line is where the body starts
            request->body = line_end + 1;
            break;
        }
        // Obs-fold detection: if line starts with space/tab, it's a continuation
        // of the previous header value (RFC 7230 Section 3.2.6).
        if ((line[0] == ' ' || line[0] == '\t') &&
             request->header_count > 0) {
            // Unfold: append a single space + trimmed continuation to previous value
            http_header_t *prev = &request->headers[request->header_count - 1];
            size_t prev_len = strlen(prev->value);
            // Trim leading whitespace from the continuation
            char *cont = line;
            while (cont < line + line_len && (*cont == ' ' || *cont == '\t')) {
                cont++;
            }
            size_t cont_len = (line + line_len) - cont;
            // Append " " + continuation (with space to replace the folding whitespace)
            if (prev_len + 1 + cont_len < sizeof(prev->value) - 1) {
                prev->value[prev_len] = ' ';
                memcpy(prev->value + prev_len + 1, cont, cont_len);
                prev->value[prev_len + 1 + cont_len] = '\0';
            }
            // If it doesn't fit, truncate silently — the value is already useful
            pos = line_end + 1;
            continue;
        }
        // Find the colon separator
        char *colon = memchr(line, ':', line_len);
        if (!colon) {
            // Header line with no colon → malformed; skip or 400
            pos = line_end + 1;
            continue;
        }
        // Extract name: everything before the colon
        size_t name_len = colon - line;
        if (name_len == 0 || name_len >= sizeof(request->headers[0].name)) {
            pos = line_end + 1;
            continue;
        }
        http_header_t *hdr = &request->headers[request->header_count];
        memcpy(hdr->name, line, name_len);
        hdr->name[name_len] = '\0';
        // Normalize header name to lowercase for case-insensitive storage
        for (size_t i = 0; i < name_len; i++) {
            hdr->name[i] = (char)tolower((unsigned char)hdr->name[i]);
        }
        // Extract value: everything after colon, with OWS stripped
        char *value_start = colon + 1;
        char *value_end   = line + line_len;
        // Strip leading OWS
        while (value_start < value_end &&
               (*value_start == ' ' || *value_start == '\t')) {
            value_start++;
        }
        // Strip trailing OWS
        while (value_end > value_start &&
               (*(value_end - 1) == ' ' || *(value_end - 1) == '\t')) {
            value_end--;
        }
        size_t value_len = value_end - value_start;
        if (value_len >= sizeof(hdr->value)) {
            value_len = sizeof(hdr->value) - 1;  // Truncate if too long
        }
        memcpy(hdr->value, value_start, value_len);
        hdr->value[value_len] = '\0';
        request->header_count++;
        pos = line_end + 1;
    }
    return request->header_count;
}
```
Walk through this against the five challenges:
**Challenge 1 — case-insensitive names**: The `tolower()` loop converts every character of the header name to lowercase before storing it. When you later look up `"content-type"`, `"content-length"`, or any other header, you call `strcmp()` with a lowercase key. The storage is normalized at parse time so every subsequent lookup is fast and simple.

![Case-Insensitive Header Normalization: Before/After](./diagrams/diag-m2-case-insensitive-header-lookup.svg)

**Challenge 2 — OWS stripping**: The `while (value_start < value_end && (*value_start == ' ' || ...))` loops advance the start pointer and retreat the end pointer, stripping whitespace from both sides of the value before copying.
**Challenge 3 — obs-fold**: The `if (line[0] == ' ' || line[0] == '\t')` check at the top of the loop detects a continuation line. Instead of creating a new header entry, it appends to the previous one's value with a space separator. This is the unfolding strategy RFC 7230 recommends.
**Challenge 4 — CRLF vs bare LF**: Every `line_end` search uses `memchr(pos, '\n', ...)` — scanning for `\n` only. This handles both `\r\n` (CRLF) and bare `\n` (LF). After finding `\n`, the code strips any trailing `\r` from the line content. This handles both line-ending styles in a single pass.
**Challenge 5 — delimiter scan under size limit**: The `while (pos < end ...)` condition ties the scan directly to the buffer boundary. `end` is `raw_headers + raw_len`, which is bounded by `REQUEST_BUF_SIZE`. You cannot read past it.
---
## Header Lookup
With headers stored as lowercase names, lookup becomes a simple array scan:
```c
const char *get_header(const http_request_t *req, const char *name) {
    // name must already be lowercase
    for (int i = 0; i < req->header_count; i++) {
        if (strcmp(req->headers[i].name, name) == 0) {
            return req->headers[i].value;
        }
    }
    return NULL;  // Header not present
}
```
Usage:
```c
const char *host   = get_header(&req, "host");
const char *ctype  = get_header(&req, "content-type");
const char *clen   = get_header(&req, "content-length");
```
A linear scan through 32 entries costs 32 `strcmp()` calls — a few hundred nanoseconds at most. For a static file server with tens of thousands of requests per second (not millions), this is the right tradeoff. [[EXPLAIN:linear-scan-vs-hash-map-for-small-collections-—-hash-maps-have-overhead-per-lookup-amortized-across-size;-for-n-under-32-linear-scan-is-often-faster-due-to-cache-locality|Linear scan vs hash map for small collections — hash maps have overhead per lookup; for N under 32, linear scan is often faster due to cache locality]]
If you wanted O(1) lookup, you would store headers in a hash map with a case-insensitive hash function. That is what nginx's internal header table does. For this milestone, linear scan is correct.
---
## Validation: Method Checking and Error Responses
After parsing the request line, validate the method. HTTP defines many methods (GET, HEAD, POST, PUT, DELETE, OPTIONS, PATCH, TRACE), but your static file server only supports GET and HEAD. Return 501 Not Implemented for anything else.
```c
typedef enum {
    HTTP_METHOD_GET,
    HTTP_METHOD_HEAD,
    HTTP_METHOD_OTHER
} http_method_t;
http_method_t classify_method(const char *method) {
    if (strcmp(method, "GET")  == 0) return HTTP_METHOD_GET;
    if (strcmp(method, "HEAD") == 0) return HTTP_METHOD_HEAD;
    return HTTP_METHOD_OTHER;
}
```

![Request Validation: Error Response Decision Tree](./diagrams/diag-m2-error-response-decision-tree.svg)

Build a small set of pre-formatted error responses:
```c
#define HTTP_400 \
    "HTTP/1.1 400 Bad Request\r\n" \
    "Content-Type: text/html\r\n" \
    "Content-Length: 49\r\n" \
    "Connection: close\r\n" \
    "\r\n" \
    "<html><body><h1>400 Bad Request</h1></body></html>"
#define HTTP_414 \
    "HTTP/1.1 414 URI Too Long\r\n" \
    "Content-Type: text/html\r\n" \
    "Content-Length: 49\r\n" \
    "Connection: close\r\n" \
    "\r\n" \
    "<html><body><h1>414 URI Too Long</h1></body></html>"
#define HTTP_501 \
    "HTTP/1.1 501 Not Implemented\r\n" \
    "Content-Type: text/html\r\n" \
    "Content-Length: 53\r\n" \
    "Connection: close\r\n" \
    "\r\n" \
    "<html><body><h1>501 Not Implemented</h1></body></html>"
```
> **Important**: Every hardcoded error response string must have a `Content-Length` that exactly matches its body's byte count. Count the bytes carefully — the HTML body in the examples above may not match the `Content-Length` values shown; adjust them to match your actual strings before shipping.
The `Content-Length` values in the macros above are illustrative. You must measure each body: `strlen("<html><body><h1>400 Bad Request</h1></body></html>")` = 49 bytes. Verify with `printf("%zu\n", strlen("..."))`.
---
## The HOST Header: Required in HTTP/1.1
RFC 7230 Section 5.4 states:
> "A client MUST send a Host header field in all HTTP/1.1 request messages."
The `Host` header tells the server which virtual host the client is addressing — essential when a single IP serves multiple domain names. If you ever run nginx with multiple `server_name` blocks, it is the `Host` header that routes the request to the right one.
Your server should validate and log the `Host` header:
```c
void validate_and_log_host(const http_request_t *req) {
    const char *host = get_header(req, "host");
    if (host == NULL) {
        fprintf(stderr, "HTTP/1.1 request missing required Host header\n");
        // For strict compliance, return 400 Bad Request here.
        // For robustness, log and continue.
        return;
    }
    printf("Host: %s\n", host);
}
```
The decision of whether to reject a missing `Host` with 400 Bad Request or to log-and-continue is the **Postel's Law tension**: the spec says reject it, but being robust means serving the request anyway since the handler knows which host it is (there is only one). For a production reverse proxy, reject it. For this learning project, log it and continue — your server is not doing virtual hosting.
[[EXPLAIN:postels-law-be-conservative-in-what-you-send-be-liberal-in-what-you-accept-—-the-robustness-principle-that-shaped-the-web-and-its-security-implications|Postel's Law — "be conservative in what you send, be liberal in what you accept" — the robustness principle that shaped the web and its security implications]]
---
## Reading the Request Body
For GET and HEAD requests, there is no body. For POST (and future PUT), the body follows the empty line, and its length is given by the `Content-Length` header.
Your M1 `read_request()` reads until `\r\n\r\n`. After that delimiter, any additional bytes in the buffer are the start of the body. If the body is larger than the remaining buffer space, you need to read more from the socket.
```c
// Reads the request body into a caller-supplied buffer.
// Returns the number of bytes read, or -1 on error.
// req->body already points at any body bytes captured during header reading.
ssize_t read_body(int client_fd, const http_request_t *req,
                  char *body_buf, size_t body_buf_size) {
    const char *content_length_str = get_header(req, "content-length");
    if (content_length_str == NULL) {
        return 0;  // No body
    }
    long content_length = strtol(content_length_str, NULL, 10);
    if (content_length <= 0) return 0;
    if ((size_t)content_length > body_buf_size) {
        // Body too large for our buffer
        return -1;
    }
    // Copy any body bytes already in the header buffer
    size_t already_have = req->body_len;
    if (already_have > (size_t)content_length) {
        already_have = (size_t)content_length;
    }
    if (already_have > 0) {
        memcpy(body_buf, req->body, already_have);
    }
    // Read remaining bytes from the socket
    size_t needed = (size_t)content_length - already_have;
    size_t total  = already_have;
    while (total < (size_t)content_length) {
        ssize_t n = recv(client_fd, body_buf + total,
                         (size_t)content_length - total, 0);
        if (n <= 0) return -1;
        total += n;
    }
    return (ssize_t)total;
}
```
The body_len field in `http_request_t` needs to be set by your header parsing code. After `parse_headers()` sets `request->body` (pointing into the raw buffer after the empty line), calculate `body_len` as the number of bytes between `request->body` and the end of the data in the buffer:
```c
request->body_len = (raw_buf + total_bytes_read) - request->body;
```
For GET requests in this milestone, `body_len` will be 0. The infrastructure is there for POST in future extensions.
---
## HEAD Method: Headers Without Body
The HEAD method is specified in RFC 7231 Section 4.3.2:
> "The HEAD method is identical to GET except that the server MUST NOT send a message body in the response."
This sounds simple but has a subtle implication: **your response headers for HEAD must be identical to what you would send for GET** — including `Content-Length`, `Content-Type`, and `Last-Modified`. The client uses HEAD to check whether a resource exists and what its size/type would be, without downloading the full content. Caching proxies use HEAD to validate whether a cached entry is still fresh.
The implementation pattern is to write your file-serving logic so that it always computes headers, and only conditionally sends the body:
```c
void handle_request(int client_fd, const http_request_t *req) {
    http_method_t method = classify_method(req->method);
    if (method == HTTP_METHOD_OTHER) {
        send_all(client_fd, HTTP_501, strlen(HTTP_501));
        return;
    }
    // method is GET or HEAD — serve the file
    // serve_file() takes a flag: send_body = (method == HTTP_METHOD_GET)
    serve_file(client_fd, req->path, method == HTTP_METHOD_GET);
}
```
`serve_file()` (implemented in Milestone 3) will always call `fstat()` and build the full response headers. It only calls `write()` for the file contents when `send_body` is true. This is the canonical implementation: **one code path, one flag**.
---
## Putting It Together: The Full Parser
Here is the complete `http_parse.c` module integrating all the pieces:
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "http_parse.h"
// Returns 0 on success, -1 on malformed request, -2 on URI too long.
int parse_request_line(const char *line, size_t line_len,
                       http_request_t *request) {
    const char *first_space = memchr(line, ' ', line_len);
    if (!first_space) return -1;
    size_t method_len = first_space - line;
    if (method_len == 0 || method_len >= MAX_METHOD_LEN) return -1;
    const char *path_start = first_space + 1;
    size_t remaining = line_len - (path_start - line);
    const char *second_space = memchr(path_start, ' ', remaining);
    if (!second_space) return -1;
    size_t path_len = second_space - path_start;
    if (path_len == 0) return -1;
    if (path_len >= MAX_PATH_LEN) return -2;
    const char *version_start = second_space + 1;
    size_t version_len = line_len - (version_start - line);
    if (version_len > 0 && version_start[version_len - 1] == '\r') {
        version_len--;
    }
    if (version_len == 0 || version_len >= MAX_VERSION_LEN) return -1;
    memcpy(request->method,  line,          method_len);  request->method[method_len] = '\0';
    memcpy(request->path,    path_start,    path_len);    request->path[path_len]    = '\0';
    memcpy(request->version, version_start, version_len); request->version[version_len] = '\0';
    return 0;
}
int parse_headers(char *buf, size_t buf_len, http_request_t *request) {
    request->header_count = 0;
    request->body     = NULL;
    request->body_len = 0;
    char *pos = buf;
    char *end = buf + buf_len;
    // Skip the request line (it ends at the first \n)
    char *request_line_end = memchr(pos, '\n', end - pos);
    if (!request_line_end) return -1;
    pos = request_line_end + 1;
    while (pos < end && request->header_count < MAX_HEADERS) {
        char *line_end = memchr(pos, '\n', end - pos);
        if (!line_end) break;
        char *line = pos;
        size_t line_len = line_end - pos;
        if (line_len > 0 && line[line_len - 1] == '\r') {
            line_len--;
        }
        if (line_len == 0) {
            // Empty line: end of headers
            const char *body_start = line_end + 1;
            if (body_start < end) {
                request->body     = body_start;
                request->body_len = (size_t)(end - body_start);
            }
            break;
        }
        if ((line[0] == ' ' || line[0] == '\t') && request->header_count > 0) {
            // obs-fold: unfold into previous header value
            http_header_t *prev = &request->headers[request->header_count - 1];
            size_t prev_len = strlen(prev->value);
            char *cont = line;
            while (cont < line + line_len && (*cont == ' ' || *cont == '\t')) cont++;
            size_t cont_len = (line + line_len) - cont;
            if (prev_len + 1 + cont_len < sizeof(prev->value) - 1) {
                prev->value[prev_len] = ' ';
                memcpy(prev->value + prev_len + 1, cont, cont_len);
                prev->value[prev_len + 1 + cont_len] = '\0';
            }
            pos = line_end + 1;
            continue;
        }
        char *colon = memchr(line, ':', line_len);
        if (!colon) { pos = line_end + 1; continue; }
        size_t name_len = colon - line;
        if (name_len == 0 || name_len >= sizeof(request->headers[0].name)) {
            pos = line_end + 1;
            continue;
        }
        http_header_t *hdr = &request->headers[request->header_count];
        memcpy(hdr->name, line, name_len);
        hdr->name[name_len] = '\0';
        for (size_t i = 0; i < name_len; i++) {
            hdr->name[i] = (char)tolower((unsigned char)hdr->name[i]);
        }
        char *vs = colon + 1;
        char *ve = line + line_len;
        while (vs < ve && (*vs == ' ' || *vs == '\t')) vs++;
        while (ve > vs && (*(ve-1) == ' ' || *(ve-1) == '\t')) ve--;
        size_t value_len = ve - vs;
        if (value_len >= sizeof(hdr->value)) value_len = sizeof(hdr->value) - 1;
        memcpy(hdr->value, vs, value_len);
        hdr->value[value_len] = '\0';
        request->header_count++;
        pos = line_end + 1;
    }
    return request->header_count;
}
// Top-level parse entry point.
// buf: the raw request buffer from read_request().
// buf_len: total bytes in buf.
// Returns 0 on success, negative error code on failure.
// Error codes: -1 = 400 Bad Request, -2 = 414 URI Too Long, -3 = 501 Not Implemented
int http_parse_request(char *buf, size_t buf_len, http_request_t *request) {
    memset(request, 0, sizeof(*request));
    // Find the end of the request line
    char *line_end = memchr(buf, '\n', buf_len);
    if (!line_end) return -1;
    size_t line_len = line_end - buf;
    if (line_len > 0 && buf[line_len - 1] == '\r') line_len--;
    int rc = parse_request_line(buf, line_len, request);
    if (rc == -2) return -2;  // 414
    if (rc != 0)  return -1;  // 400
    // Validate HTTP version (we only speak HTTP/1.1 and HTTP/1.0)
    if (strcmp(request->version, "HTTP/1.1") != 0 &&
        strcmp(request->version, "HTTP/1.0") != 0) {
        return -1;  // 400: unrecognized version
    }
    parse_headers(buf, buf_len, request);
    // Validate method
    if (classify_method(request->method) == HTTP_METHOD_OTHER) return -3;  // 501
    return 0;
}
```
And the updated `handle_client()` in `server.c`:
```c
void handle_client(int client_fd) {
    char buf[REQUEST_BUF_SIZE];
    ssize_t req_len = read_request(client_fd, buf, sizeof(buf));
    if (req_len < 0) {
        send_all(client_fd, HTTP_400, strlen(HTTP_400));
        return;
    }
    http_request_t req;
    int rc = http_parse_request(buf, (size_t)req_len, &req);
    if (rc == -2) {
        send_all(client_fd, HTTP_414, strlen(HTTP_414));
        return;
    }
    if (rc == -3) {
        send_all(client_fd, HTTP_501, strlen(HTTP_501));
        return;
    }
    if (rc != 0) {
        send_all(client_fd, HTTP_400, strlen(HTTP_400));
        return;
    }
    // Log the parsed request
    const char *host = get_header(&req, "host");
    printf("%s %s %s  Host: %s\n",
           req.method, req.path, req.version,
           host ? host : "(none)");
    // For now, send a hardcoded response (file serving comes in M3)
    int send_body = (classify_method(req.method) == HTTP_METHOD_GET);
    if (send_body) {
        send_all(client_fd, HTTP_RESPONSE, strlen(HTTP_RESPONSE));
    } else {
        // HEAD: send only the headers
        send_all(client_fd, HTTP_RESPONSE_HEADERS_ONLY,
                 strlen(HTTP_RESPONSE_HEADERS_ONLY));
    }
}
```
---
## Testing Your Parser
Manual testing with `curl` and `telnet` covers most cases. Here is a systematic test plan:
**Normal GET request:**
```bash
curl -v http://localhost:8080/index.html
```
Expected: server logs `GET /index.html HTTP/1.1  Host: localhost:8080`, returns 200.
**HEAD request (no body):**
```bash
curl -I http://localhost:8080/index.html
```
Expected: curl receives only headers, no body. Your server must return the same headers as GET but suppress the body.
**Oversized URI (should return 414):**
```bash
# Generate a 9000-character path
python3 -c "print('GET /' + 'a'*9000 + ' HTTP/1.1\r\nHost: localhost\r\n\r\n', end='')" | nc localhost 8080
```
Expected: server returns 414 URI Too Long.
**Unsupported method (should return 501):**
```bash
curl -X DELETE http://localhost:8080/file.txt
```
Expected: server returns 501 Not Implemented.
**Malformed request line (should return 400):**
```bash
echo -e "NOTHTTP\r\n\r\n" | nc localhost 8080
```
Expected: server returns 400 Bad Request without crashing.
**Case-insensitive header lookup:**
```bash
curl -H "content-type: text/plain" -H "ACCEPT: application/json" http://localhost:8080/
```
Verify in your logs that headers are stored lowercase regardless of how they were sent.
**Bare LF line endings (robustness test):**
```bash
# Send a request with bare \n instead of \r\n
printf "GET / HTTP/1.1\nHost: localhost\n\n" | nc localhost 8080
```
Expected: server parses and responds correctly.
**`telnet` manual typing:**
```bash
telnet localhost 8080
```
Type each header line and press Enter. Your parser must handle the fact that telnet sends bare LF for each keystroke. This is the classic test for partial-read handling + LF acceptance.
---
## The Hardware Soul
What happens in the hardware when your parser runs?
**Buffer scan — sequential memory access**: The inner loops of `parse_headers()` scan through `buf` linearly with `memchr()`. This is a sequential read through a contiguous 8KB stack buffer. The CPU's hardware prefetcher will detect the sequential pattern and load the next cache line before the scanner reaches it. At 64 bytes per cache line and 8KB total, you touch 128 cache lines — all of them hot after the first pass because the buffer fits in L1 cache (typically 32KB). The entire parse of a normal HTTP request happens entirely in L1 cache. This is as fast as memory access gets.
**`tolower()` in the name normalization loop**: Each `tolower()` call involves a branch (is the character between 'A' and 'Z'?) that is highly predictable: header names are short ASCII strings where most characters are already lowercase. The branch predictor will overwhelmingly predict "already lowercase" and be right most of the time. The misprediction penalty is paid at most once or twice per header name (for the uppercase letters), which is negligible.
**`strcmp()` in method classification**: `strcmp("GET", method)` touches 3–16 bytes of the method string. Both the literal `"GET"` (in read-only data segment) and `req.method` (on the stack) are almost certainly in L1 cache. Three `strcmp()` calls cost under 10 nanoseconds total.
**Stack allocation**: `http_request_t` is about 40KB when you count MAX_HEADERS × (128 + 1024) bytes = 32 × 1152 = 36KB. That pushes it past L1 cache size. If you are allocating `http_request_t` on the stack in `handle_client()`, check with `-Wframe-larger-than=16384` that you are not exceeding the thread stack limit. Consider heap-allocating `http_request_t` or reducing header value buffer sizes.
> ⚠️ **Stack size alert**: With `MAX_HEADERS = 32` and `http_header_t` containing `char name[128]` + `char value[1024]`, the headers array alone is `32 × 1152 = 36,864` bytes. Plus the `char buf[8192]` in `handle_client()`. Your thread may be using 45KB of stack. Default thread stack size is typically 8MB on Linux, so this is safe — but be aware of it. In a thread pool with 64 threads, the headers arrays alone consume 64 × 36KB ≈ 2.3MB of stack across all threads.
---
## Design Decisions
| Approach | Pros | Cons | Used By |
|---|---|---|---|
| **Normalize to lowercase at parse time (chosen)** ✓ | Single normalization point; `strcmp()` works for lookup | Loses original casing (irrelevant for HTTP) | nginx, most HTTP parsers |
| Normalize at lookup time (`strcasecmp`) | Preserves original casing | Every lookup pays the normalization cost | Some small servers |
| Hash map with case-insensitive hash | O(1) lookup | malloc, more code, overkill for ≤32 headers | Production parsers (llhttp) |
| Approach | Pros | Cons | Used By |
|---|---|---|---|
| **Fixed-size header value buffers (chosen)** ✓ | No malloc, predictable memory, safe | Truncates very long values | nginx, Apache |
| Dynamic allocation per value | No truncation | malloc per header, fragmentation | curl, libhttp |
| Single arena allocation | Fast, cache-friendly | More complex pointer arithmetic | jemalloc's parser, llhttp |
---
## Knowledge Cascade — What You Just Unlocked
**1. Case-insensitive comparison shaped standard library design.** The function `strcasecmp()` (and its cousin `strncasecmp()`) was added to POSIX specifically because HTTP, DNS, and MIME type matching all required it. The same requirement appears everywhere text-based protocols meet case: HTML attribute names are case-insensitive, CSS property names are case-insensitive, DNS record type names are case-insensitive, email header names (SMTP/MIME) are case-insensitive. Whenever you design a text protocol, you will face this choice: case-sensitive (simple, fast) or case-insensitive (user-friendly, more complex). HTTP chose case-insensitive for human readability. Redis chose case-insensitive commands for the same reason. TCP/IP port numbers are just integers — no case problem.
**2. State machine parsers are everywhere.** The parser you just wrote — scan bytes, maintain state (which-line-are-we-on, are-we-in-a-header-value, did-we-see-a-colon), transition between states on specific byte patterns — is structurally identical to a JSON parser, a CSS tokenizer, a C compiler's lexer, or a Markdown parser. The `llhttp` library (Node.js's HTTP parser, a replacement for the original `http_parser` from nginx) is a state machine compiled from a description language. The V8 JavaScript engine's scanner that tokenizes JS source code is the same pattern. Once you have written one state machine parser by hand, you will immediately recognize the pattern in every other parser you encounter.
**3. The 8KB URI limit is a security control, not an arbitrary number.** The 414 URI Too Long response exists because an unbounded URI field is a denial-of-service vector: a client that sends a 1GB "path" forces your server to allocate 1GB of memory before it can even look at the first byte of headers. The same pattern appears in every network daemon that parses text: SMTP limits command lines to 512 bytes, Redis limits inline commands to 64KB, nginx defaults to 8KB per header buffer. These are all the same design: **a fixed memory ceiling turns a potential OOM (Out of Memory) attack into a 4xx error**. When you see a 414 or 431 (Request Header Fields Too Large) response in a browser, you are seeing this defense working.
**4. HEAD reveals how CDN cache validation works.** Caching proxies like Varnish, Squid, and CDN edge nodes use HEAD requests to check whether a cached resource is still fresh, without downloading the full body. When your browser connects to a CDN and the CDN needs to verify its cache: it sends `HEAD /file.css HTTP/1.1` to your origin server, receives the `Last-Modified` and `ETag` headers, compares them to what it cached, and either serves the cached version or fetches the full body. This is why your HEAD implementation must return the *same* headers as GET — a CDN that gets different `Content-Length` values from GET vs HEAD would be broken in subtle ways. By implementing HEAD correctly now, you understand the cache validation mechanism that makes CDN caching possible.
**5. Postel's Law created the modern web and its security nightmare.** Your decision to accept bare LF line endings when the spec requires CRLF is an instance of Postel's Law: *be conservative in what you send, be liberal in what you accept*. This principle was written into RFC 793 (TCP) by Jon Postel in 1981. It is why the web survived the chaos of early browser implementations — servers accepted malformed HTML and browsers accepted malformed responses. But this liberalism created a security model where attackers could exploit the gap between "what the spec says" and "what parsers accept." HTTP request smuggling attacks (a class of attacks that can bypass security controls on CDNs and proxies) directly exploit differences in how different parsers interpret ambiguous or slightly-malformed requests. Your bare-LF acceptance is harmless for a static file server. In a security-critical reverse proxy, every deviation from the spec is a potential attack surface.
---
## Common Pitfalls Checklist
Before moving to Milestone 3, verify your implementation handles all of these:
- [ ] Request line parser uses `memchr()`, not `strchr()` — bounds-safe scanning
- [ ] Path length is checked against `MAX_PATH_LEN` before `memcpy()` — 414 returned correctly
- [ ] Header names are lowercased at parse time — all lookups use lowercase keys
- [ ] OWS stripped from both sides of header values — no leading/trailing spaces in stored values
- [ ] Both CRLF and bare LF line endings accepted — `telnet` works
- [ ] obs-fold unfolded into previous header value — folded headers don't create malformed entries
- [ ] `Host` header logged — missing Host generates a warning (or 400 in strict mode)
- [ ] Methods other than GET and HEAD return 501 — not silently ignored
- [ ] HEAD responses contain correct headers but no body — same `Content-Length` as GET
- [ ] Malformed request line (no spaces, empty path, unrecognized version) returns 400
- [ ] `Content-Length` errors in your hardcoded responses checked — body byte count matches header value
- [ ] `http_request_t` size checked — consider heap allocation if stack usage is a concern
---
<!-- END_MS -->


<!-- MS_ID: http-server-basic-m3 -->
<!-- MS_ID: http-server-basic-m3 -->
# Milestone 3: Static File Serving
## Where We Are


In Milestone 1 you built the socket plumbing. In Milestone 2 you taught your server to read HTTP grammar — to turn a stream of bytes into a structured `http_request_t` with a method, path, and headers. Now your server knows *what* the client is asking for. It is time to actually deliver it.
This milestone is about the journey from a URL string like `/images/logo.png` to the bytes of that file arriving in the client's browser — and every security check, metadata query, and protocol negotiation that happens in between. By the end, your server will serve real files from a real directory, detect their types, protect itself from path attacks, and skip transmitting files the client already has.
---
## The Revelation: String Prefix Checks Are Not Security
Here is the assumption almost every developer carries into their first file-serving implementation:
> "I will prepend the document root to the request path. Then I will check that the result still starts with the document root. If it does, I am safe."
In code, it looks like this:
```c
// This is NOT secure. Do not do this.
char full_path[4096];
snprintf(full_path, sizeof(full_path), "%s%s", doc_root, req->path);
if (strncmp(full_path, doc_root, strlen(doc_root)) == 0) {
    // "safe" — open and serve the file
    open(full_path, O_RDONLY);
}
```
This check looks airtight. It is not. Automated vulnerability scanners find and exploit it in under sixty seconds. Here are the three distinct bypass vectors, each requiring a different fix:
**Bypass 1: URL-encoded traversal.** The HTTP spec allows any byte in a URI to be percent-encoded: `.` becomes `%2e`, `/` becomes `%2f`. A request for `GET /%2e%2e%2f%2e%2e%2fetc/passwd HTTP/1.1` arrives in your buffer as the literal string `/%2e%2e%2f%2e%2e%2fetc/passwd`. After your `snprintf`, `full_path` becomes `/var/www/%2e%2e%2f%2e%2e%2fetc/passwd`. The `strncmp` passes — the string starts with `/var/www/`. Then `open()` sees the `%2e%2e` sequences... except it does not. The kernel's `open()` does not decode percent-encoding. So the open fails. But here is the trap: if *you* URL-decode the path before the prefix check (which you must, to serve files correctly), the path becomes `/../../../etc/passwd` before the check, and `strncmp` fails — so far so good. But if you URL-decode *after* the check, the decoded path escapes and the check is useless. The fix: **decode first, check after canonicalization**.
**Bypass 2: Symlinks inside the document root.** Imagine your document root is `/var/www/html`. Someone (a deployment script, a developer shortcut, a compromised file upload) creates a symlink: `ln -s /etc /var/www/html/secrets`. Now a request for `/secrets/passwd` constructs `full_path = /var/www/html/secrets/passwd`. The string check passes — it starts with `/var/www/html`. But when `open()` follows the symlink, it opens `/etc/passwd`. The string `/var/www/html/secrets/passwd` never contained `..` — it was a perfectly innocent-looking path that nevertheless escapes the document root through a symbolic link. The fix: **resolve all symlinks before checking containment**.
**Bypass 3: Double-slash and Unicode normalization.** The string `/var/www/html//../../etc` starts with `/var/www/html/` — your prefix check passes. But the kernel normalizes double slashes and `..` components during path resolution, so `open()` opens `/etc`. Similarly, some filesystems and some locales treat certain Unicode sequences as equivalent to ASCII path separators or `.` characters. The fix: **canonicalize the path — resolve every `..`, every `.`, and every double slash to their physical meaning — before checking containment**.
All three bypasses have the same root cause: **a string comparison on a path is not the same as a filesystem containment check**. The string representation of a path and the actual filesystem object it resolves to can be completely different things. The only correct approach is to call `realpath()`, which asks the kernel to resolve the path to its canonical, unambiguous, symlink-free, `..`-free absolute form. Only after `realpath()` succeeds does a string prefix comparison mean anything.

![Directory Traversal Attack: Three Bypass Vectors Neutralized](./diagrams/diag-m3-traversal-attack-neutralization.svg)

[[EXPLAIN:realpath()-and-filesystem-path-canonicalization-—-what-..-means-at-the-kernel-level-and-how-symlinks-can-escape-a-prefix-check|realpath() and filesystem path canonicalization — what .. means at the kernel level and how symlinks can escape a prefix check]]
---
## The URL-to-Filesystem Pipeline
Every file-serving request travels through a five-stage pipeline. Skipping or reordering any stage opens a vulnerability or a bug.

![URL Path → Filesystem Path Pipeline](./diagrams/diag-m3-url-to-filesystem-mapping.svg)

```
Stage 1: URL Decode
  /%2e%2e%2fpasswd  →  /../passwd
Stage 2: Concatenate with document root
  /var/www + /../passwd  →  /var/www/../passwd
Stage 3: realpath() — canonicalize
  /var/www/../passwd  →  /passwd
Stage 4: Prefix check
  /passwd does NOT start with /var/www → 403 Forbidden
Stage 5: open() and serve
  (only reached if prefix check passes)
```
If you do the prefix check at Stage 2 (before `realpath()`), the path `/var/www/../passwd` passes — it does start with `/var/www`. But after the kernel resolves `..`, it opens `/passwd`. The check at Stage 2 is meaningless. The check must happen at Stage 4.
### Stage 1: URL Decoding
Before concatenating the path with the document root, percent-decode it. A `%` followed by two hex digits should be decoded to the corresponding byte.
```c
// Decodes percent-encoded characters in 'src' into 'dst'.
// dst must be at least as large as src (decoded is never longer than encoded).
// Returns 0 on success, -1 if encoding is malformed.
int url_decode(const char *src, char *dst, size_t dst_size) {
    size_t i = 0, j = 0;
    while (src[i] != '\0' && j < dst_size - 1) {
        if (src[i] == '%') {
            if (!isxdigit((unsigned char)src[i+1]) ||
                !isxdigit((unsigned char)src[i+2])) {
                return -1;  // Malformed percent-encoding → 400 Bad Request
            }
            char hex[3] = { src[i+1], src[i+2], '\0' };
            dst[j++] = (char)strtol(hex, NULL, 16);
            i += 3;
        } else if (src[i] == '+') {
            // '+' in query strings means space; in path segments it is literal '+'
            // For paths, treat '+' as literal
            dst[j++] = '+';
            i++;
        } else {
            dst[j++] = src[i++];
        }
    }
    if (src[i] != '\0') return -1;  // Output buffer exhausted
    dst[j] = '\0';
    return 0;
}
```
Two things to note here: first, after decoding, you must re-validate that the path does not contain null bytes (`\0`). A path like `/%00` decodes to a string containing a null byte, which `strncmp` and `strlen` treat as end-of-string — a source of subtle bugs and bypasses. Reject any decoded path containing a null byte with 400. Second, a `+` in a URL path is literal; only in query strings does `+` mean space. This distinction matters for files with `+` in their names.
```c
// After url_decode(), validate no null bytes snuck in
int contains_null(const char *s, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (s[i] == '\0') return 1;
    }
    return 0;
}
```
### Stage 2: Path Concatenation
Join the document root with the decoded path:
```c
#define DOC_ROOT_MAX 1024
#define FULL_PATH_MAX (DOC_ROOT_MAX + MAX_PATH_LEN + 2)
// Returns 0 on success, -1 on error (path too long).
int build_full_path(const char *doc_root, const char *decoded_path,
                    char *full_path, size_t full_path_size) {
    size_t root_len = strlen(doc_root);
    size_t path_len = strlen(decoded_path);
    // Strip trailing slash from doc_root to avoid double-slash
    while (root_len > 1 && doc_root[root_len - 1] == '/') root_len--;
    // Ensure decoded_path starts with '/'
    if (decoded_path[0] != '/') return -1;
    if (root_len + path_len + 1 >= full_path_size) return -1;
    memcpy(full_path, doc_root, root_len);
    memcpy(full_path + root_len, decoded_path, path_len + 1);  // +1 for '\0'
    return 0;
}
```
### Stage 3: `realpath()` — The Security Keystone
[[EXPLAIN:realpath()-internals-—-the-series-of-lstat()-and-readlink()-calls-it-makes-and-its-ENOENT-behavior-on-nonexistent-paths|realpath() internals — the series of lstat() and readlink() calls it makes, and its ENOENT behavior on nonexistent paths]]
```c
char canonical[PATH_MAX];
if (realpath(full_path, canonical) == NULL) {
    if (errno == ENOENT || errno == ENOTDIR) {
        // File does not exist — send 404
        send_404(client_fd);
        return;
    }
    // Other error (EACCES, ELOOP for circular symlinks, etc.) — send 403
    send_403(client_fd);
    return;
}
```
`realpath()` does several things in one call:
1. Resolves every `..` and `.` component by walking the directory tree.
2. Follows every symlink it encounters, resolving the symlink target recursively.
3. Returns the resulting absolute path with all of these normalized away.
4. Returns `NULL` with `errno` set if the path does not exist (`ENOENT`), a component is not a directory (`ENOTDIR`), or any other filesystem error occurs.
The `ENOENT` return is your 404 signal. If `realpath()` succeeds, the file exists (or the directory exists — you will check separately for the directory case). If it fails with `ENOENT`, the file does not exist.
One critical nuance: `realpath()` with a `NULL` second argument allocates a buffer and returns a pointer to it (POSIX extension, available on Linux and macOS). If you pass a fixed-size buffer as the second argument (as above), you must ensure it is at least `PATH_MAX` bytes (typically 4096 on Linux). Using a fixed buffer is safer in a multithreaded environment because it avoids a `malloc()` that you could forget to `free()`.
### Stage 4: The Prefix Check That Now Actually Works
```c
// After realpath() succeeds, check that canonical is inside doc_root.
// Both paths are now fully resolved — no symlinks, no '..' components.
size_t root_len = strlen(doc_root);
if (strncmp(canonical, doc_root, root_len) != 0 ||
    (canonical[root_len] != '/' && canonical[root_len] != '\0')) {
    send_403(client_fd);
    return;
}
```
The extra check on `canonical[root_len]` is subtle but critical. If your document root is `/var/www/html` and a file resolves to `/var/www/html2/secret.txt`, a naive `strncmp(canonical, doc_root, strlen(doc_root))` passes — both start with `/var/www/html`. But that file is not inside your document root. The check `canonical[root_len] == '/' || canonical[root_len] == '\0'` ensures the match extends to a path separator or end of string, not just a shared prefix of directory names.
---
## Directory Handling: Serving `index.html`
When the resolved path is a directory (the client requested `/`, `/about/`, or `/docs`), you have a choice: list the directory contents, serve an index file, or return 403. A static file server should serve `index.html` if it exists, and return 403 if it does not. Listing directory contents leaks your filesystem structure and is a security concern.

![Directory Path Handling: Index File Auto-Serve Logic](./diagrams/diag-m3-directory-index-logic.svg)

```c
// Check whether the resolved path is a directory.
struct stat st;
if (stat(canonical, &st) < 0) {
    send_404(client_fd);
    return;
}
if (S_ISDIR(st.st_mode)) {
    // Append /index.html and try again
    char index_path[PATH_MAX];
    int n = snprintf(index_path, sizeof(index_path), "%s/index.html", canonical);
    if (n < 0 || (size_t)n >= sizeof(index_path)) {
        send_403(client_fd);
        return;
    }
    // Re-canonicalize (index.html might itself be a symlink)
    if (realpath(index_path, canonical) == NULL) {
        // index.html does not exist or is inaccessible
        send_403(client_fd);
        return;
    }
    // Re-run the prefix check on the new canonical path
    if (strncmp(canonical, doc_root, root_len) != 0 ||
        (canonical[root_len] != '/' && canonical[root_len] != '\0')) {
        send_403(client_fd);
        return;
    }
    // Re-stat the index file
    if (stat(canonical, &st) < 0) {
        send_404(client_fd);
        return;
    }
}
```
Notice that after appending `/index.html`, you call `realpath()` again and re-run the prefix check. This is not paranoia — `index.html` could itself be a symlink that points outside the document root. Every call to `realpath()` is a containment checkpoint.
The order of operations also prevents a TOCTOU (Time-of-Check, Time-of-Use) [[EXPLAIN:TOCTOU-race-condition-—-the-window-between-checking-a-file's-properties-and-using-the-file-where-the-filesystem-can-change|TOCTOU race condition — the window between checking a file's properties and using the file where the filesystem can change]] race: you `stat()` to check whether it is a directory, then `open()` the result. An attacker who can replace the directory with a symlink in the microsecond between your `stat()` and `open()` can bypass your check. For a simple static file server this risk is low, but in production systems this is addressed by using `openat()` with `O_NOFOLLOW` on the final component. For this milestone, the `realpath()` + `stat()` + `open()` sequence is correct and sufficient.
---
## MIME Type Detection

![MIME Type Lookup: Extension Map and Default Fallback](./diagrams/diag-m3-mime-type-dispatch.svg)

The `Content-Type` response header tells the browser how to interpret the response body. This single header controls whether the browser renders HTML, executes JavaScript, displays an image inline, or triggers a file download dialog. A server that gets this wrong breaks the web.
The rule: derive `Content-Type` from the file extension. This is a lookup table:
```c
typedef struct {
    const char *extension;   // Lowercase, with leading dot: ".html"
    const char *mime_type;   // Full MIME type string: "text/html; charset=utf-8"
} mime_entry_t;
static const mime_entry_t MIME_TABLE[] = {
    { ".html",  "text/html; charset=utf-8"       },
    { ".htm",   "text/html; charset=utf-8"       },
    { ".css",   "text/css; charset=utf-8"        },
    { ".js",    "application/javascript"         },
    { ".json",  "application/json"               },
    { ".txt",   "text/plain; charset=utf-8"      },
    { ".xml",   "application/xml"               },
    { ".svg",   "image/svg+xml"                  },
    { ".png",   "image/png"                      },
    { ".jpg",   "image/jpeg"                     },
    { ".jpeg",  "image/jpeg"                     },
    { ".gif",   "image/gif"                      },
    { ".ico",   "image/x-icon"                   },
    { ".pdf",   "application/pdf"                },
    { ".woff",  "font/woff"                      },
    { ".woff2", "font/woff2"                     },
    { ".webp",  "image/webp"                     },
    { NULL,     NULL                             },  // Sentinel
};
const char *get_mime_type(const char *path) {
    const char *dot = strrchr(path, '.');
    if (dot == NULL) return "application/octet-stream";
    // Compare case-insensitively — .HTML and .html are the same
    for (int i = 0; MIME_TABLE[i].extension != NULL; i++) {
        if (strcasecmp(dot, MIME_TABLE[i].extension) == 0) {
            return MIME_TABLE[i].mime_type;
        }
    }
    return "application/octet-stream";  // Unknown extension: trigger download
}
```
`strrchr(path, '.')` finds the *last* dot in the path. This correctly handles files like `archive.tar.gz` — you get `.gz`, not `.tar`. It also handles `dotfiles` like `.gitignore` which have no extension — `strrchr` returns a pointer to the leading `.`, which will not match any entry in the table, so you get `application/octet-stream` (which triggers a download, appropriate for dotfiles).
Why `application/octet-stream` as the default? The browser treats it as raw binary data and offers a download dialog. This is the safe default. If your server returns `text/html` for a JavaScript file, the browser renders it as HTML — which could contain attacker-injected markup. If your server returns `text/plain` for a JavaScript file, browsers refuse to execute it in certain security contexts (notably when loaded as a module with `<script type="module">`). Getting the MIME type wrong is not just a display issue — it can break functionality and create security vulnerabilities.
The `charset=utf-8` suffix on text types tells the browser the character encoding of the text. Without it, the browser uses heuristics to guess the encoding, which can fail on files containing non-ASCII characters. Always include `charset=utf-8` for text MIME types.
[[EXPLAIN:MIME-types-and-Content-Type-negotiation-—-how-browsers-use-Content-Type-to-decide-rendering-vs-download-and-why-text/plain-breaks-JavaScript-modules|MIME types and Content-Type negotiation — how browsers use Content-Type to decide rendering vs. download, and why text/plain breaks JavaScript modules]]
---
## Reading and Sending File Contents

![File Serving Code Path: open() → fstat() → write() Loop](./diagrams/diag-m3-file-serve-flow.svg)

With a validated canonical path and a MIME type, you can now open and serve the file. The key constraint: **always read files in binary mode** — which on Linux/POSIX simply means using `open()`/`read()` rather than `fopen()`/`fread()` in text mode. On POSIX systems there is no text/binary distinction at the OS level; the C standard library's text mode (`fopen("file", "r")` vs `"rb"`) can perform newline translation on Windows but not on Linux. Since your server will send binary content (images, PDFs, fonts), use the POSIX `open()`/`read()`/`close()` calls directly.
[[EXPLAIN:binary-vs-text-mode-file-reading-on-POSIX-—-why-open()-with-O_RDONLY-is-always-binary-and-why-this-matters-for-serving-images-and-PDFs|Binary vs. text mode file reading on POSIX — why open() with O_RDONLY is always binary and why this matters for serving images and PDFs]]
```c
// stat() gives us the file size before opening — needed for Content-Length.
// The stat struct is also used for Last-Modified.
struct stat st;
if (stat(canonical, &st) < 0) {
    send_404(client_fd);
    return;
}
// Open the file for reading
int file_fd = open(canonical, O_RDONLY);
if (file_fd < 0) {
    if (errno == EACCES) {
        send_403(client_fd);
    } else {
        send_404(client_fd);
    }
    return;
}
```
`stat()` before `open()` gives you `st.st_size` (file size in bytes, used for `Content-Length`) and `st.st_mtime` (last modification time, used for `Last-Modified`). You need both before sending the response headers, so getting them via `stat()` first is the right order.
### Building and Sending the Response Headers
```c
// Format the Last-Modified timestamp per RFC 7231 HTTP-date format:
// "Day, DD Mon YYYY HH:MM:SS GMT"
char last_modified[64];
struct tm *gmt = gmtime(&st.st_mtime);
strftime(last_modified, sizeof(last_modified),
         "%a, %d %b %Y %H:%M:%S GMT", gmt);
const char *mime = get_mime_type(canonical);
// Build the response header block
char header_buf[1024];
int header_len = snprintf(header_buf, sizeof(header_buf),
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: %s\r\n"
    "Content-Length: %lld\r\n"
    "Last-Modified: %s\r\n"
    "Connection: close\r\n"
    "\r\n",
    mime,
    (long long)st.st_size,
    last_modified);
if (header_len < 0 || (size_t)header_len >= sizeof(header_buf)) {
    close(file_fd);
    send_500(client_fd);
    return;
}
// Send headers
if (send_all(client_fd, header_buf, (size_t)header_len) < 0) {
    close(file_fd);
    return;
}
```
The `%lld` format for `st.st_size` is important. `off_t` (the type of `st_size`) is a 64-bit signed integer on modern 64-bit Linux (`_FILE_OFFSET_BITS=64`), but on some 32-bit systems it is 32-bit. Casting to `long long` and using `%lld` is portable. A file larger than 2GB would overflow a 32-bit `off_t` — this is why 64-bit file offsets matter for any server that might serve large files.
### Sending the File Body
```c
if (send_body) {  // false for HEAD requests
    char read_buf[65536];  // 64KB read buffer — matches typical NIC MTU aggregation
    ssize_t bytes_read;
    while ((bytes_read = read(file_fd, read_buf, sizeof(read_buf))) > 0) {
        if (send_all(client_fd, read_buf, (size_t)bytes_read) < 0) {
            break;  // Client disconnected mid-transfer
        }
    }
    if (bytes_read < 0) {
        perror("read");  // Disk I/O error — headers already sent, can't send 500
    }
}
close(file_fd);
```
The 64KB read buffer deserves explanation. Reading in large chunks amortizes the overhead of the `read()` system call across more data. A 1-byte read buffer would work correctly but require one `read()` syscall per byte — catastrophic for a 1MB image. A 64KB buffer means one syscall per 65,536 bytes, which is well-matched to how TCP segments are batched in the kernel's send buffer (which defaults to 87KB on Linux). Larger buffers (256KB, 1MB) give diminishing returns because the bottleneck shifts to network throughput.
Note the comment "headers already sent, can't send 500." This is a fundamental constraint of HTTP/1.1 without chunked transfer encoding: once you have started sending the response body, you cannot go back and change the status code. If a disk read fails halfway through serving a 10MB file, the client has already received `HTTP/1.1 200 OK` and 5MB of data. There is no way to retract that. Your options are to close the connection abruptly (the client will see a truncated response) or to use chunked transfer encoding (which allows signaling errors in a trailer). For this milestone, close the connection — the client will observe a partial body and retry or display an error.
---
## Conditional Requests: The If-Modified-Since / 304 Flow

![If-Modified-Since / 304 Not Modified: Full Round-Trip](./diagrams/diag-m3-conditional-request-flow.svg)

When a browser visits a page for the second time, it does not want to download unchanged files again. HTTP/1.1 provides the conditional request mechanism to avoid this redundant transfer. Here is the full round-trip:
**First visit:**
1. Browser sends `GET /style.css HTTP/1.1`
2. Your server responds `200 OK` with `Last-Modified: Mon, 01 Jan 2024 12:00:00 GMT`
3. Browser caches the file and the `Last-Modified` timestamp
**Second visit:**
1. Browser sends `GET /style.css HTTP/1.1` with `If-Modified-Since: Mon, 01 Jan 2024 12:00:00 GMT`
2. Your server checks: has the file changed since that timestamp?
3. If unchanged: respond `304 Not Modified` with no body — saves the entire file transfer
4. If changed: respond `200 OK` with the new file contents and updated `Last-Modified`
The `304 Not Modified` response contains no body but *must* include the same headers the `200` would include: `Content-Type`, `Last-Modified`, and `Content-Length`. The `Content-Length` tells the client the size the body *would* have been — the client already has that many bytes in its cache.
[[EXPLAIN:HTTP-conditional-requests-and-304-Not-Modified-—-how-browser-caches-CDN-edge-caches-and-reverse-proxies-all-use-this-mechanism-to-avoid-retransmitting-unchanged-content|HTTP conditional requests and 304 Not Modified — how browser caches, CDN edge caches, and reverse proxies all use this mechanism to avoid retransmitting unchanged content]]
### Parsing the If-Modified-Since Header
The `If-Modified-Since` header value is an HTTP-date string: `Mon, 01 Jan 2024 12:00:00 GMT`. You need to parse this into a `time_t` to compare it with `st.st_mtime`.
```c
// Parses an HTTP-date string into a time_t.
// Returns (time_t)-1 on parse failure.
time_t parse_http_date(const char *date_str) {
    if (date_str == NULL) return (time_t)-1;
    struct tm tm = {0};
    // strptime parses the string according to the format,
    // filling the struct tm fields.
    char *result = strptime(date_str, "%a, %d %b %Y %H:%M:%S GMT", &tm);
    if (result == NULL) {
        // Try alternate obsolete HTTP/1.0 date format: "Monday, 01-Jan-24 12:00:00 GMT"
        result = strptime(date_str, "%A, %d-%b-%y %H:%M:%S GMT", &tm);
    }
    if (result == NULL) return (time_t)-1;
    tm.tm_isdst = 0;    // HTTP dates are always GMT, no DST
    return timegm(&tm); // Convert UTC struct tm to time_t (Linux/BSD extension)
}
```
`strptime()` is the inverse of `strftime()` — it parses a formatted date string into a `struct tm`. `timegm()` is the UTC-aware version of `mktime()`: it converts a `struct tm` in UTC to a `time_t` without applying local timezone offsets. Using `mktime()` instead of `timegm()` would produce incorrect comparisons on servers not running in UTC. [[EXPLAIN:timegm()-vs-mktime()-and-UTC-timezone-handling-in-HTTP-date-comparison|timegm() vs mktime() and why UTC timezone handling matters for correct If-Modified-Since comparison]]
### The Comparison Logic
```c
time_t if_modified_since = parse_http_date(
    get_header(req, "if-modified-since")
);
// st.st_mtime is the file's last modification time (seconds since epoch, UTC)
// if_modified_since is the client's cached timestamp (seconds since epoch, UTC)
// HTTP spec: "if the selected representation's last modification date
//  is earlier or equal to the date provided in the field value, respond 304."
if (if_modified_since != (time_t)-1 &&
    st.st_mtime <= if_modified_since) {
    // File has not been modified since the client's cached version
    send_304(client_fd, mime, last_modified, st.st_size);
    close(file_fd);
    return;
}
```
The comparison `st.st_mtime <= if_modified_since` uses `<=` (not `<`). The HTTP spec says "earlier *or equal*" — if the file was modified at exactly the same second as the client's cached timestamp, the client's cache is still valid. HTTP dates have one-second granularity, so any modification within the same second is considered "equal."
### Sending the 304 Response
```c
void send_304(int client_fd, const char *mime,
              const char *last_modified, off_t content_length) {
    char buf[512];
    int len = snprintf(buf, sizeof(buf),
        "HTTP/1.1 304 Not Modified\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %lld\r\n"
        "Last-Modified: %s\r\n"
        "Connection: close\r\n"
        "\r\n",
        mime,
        (long long)content_length,
        last_modified);
    if (len > 0 && (size_t)len < sizeof(buf)) {
        send_all(client_fd, buf, (size_t)len);
    }
}
```
The `304` body is empty — no `send_body` call, no file read. You spent `stat()` latency (a kernel call that probably hits the filesystem cache) but saved the entire file read and network transfer. For a 1MB CSS file requested 100 times per second, this optimization eliminates 100MB/s of outbound traffic.
---
## Error Responses: 404 and 403
Your server must return proper HTML error pages, not blank responses. Blank responses confuse browsers and make debugging painful.
```c
#define HTTP_404_BODY "<html><body><h1>404 Not Found</h1>" \
                      "<p>The requested resource was not found.</p>" \
                      "</body></html>"
#define HTTP_403_BODY "<html><body><h1>403 Forbidden</h1>" \
                      "<p>Access to the requested resource is denied.</p>" \
                      "</body></html>"
void send_404(int client_fd) {
    const char *body = HTTP_404_BODY;
    char buf[512];
    int len = snprintf(buf, sizeof(buf),
        "HTTP/1.1 404 Not Found\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n"
        "%s",
        strlen(body), body);
    if (len > 0 && (size_t)len < sizeof(buf)) {
        send_all(client_fd, buf, (size_t)len);
    }
}
void send_403(int client_fd) {
    const char *body = HTTP_403_BODY;
    char buf[512];
    int len = snprintf(buf, sizeof(buf),
        "HTTP/1.1 403 Forbidden\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n"
        "%s",
        strlen(body), body);
    if (len > 0 && (size_t)len < sizeof(buf)) {
        send_all(client_fd, buf, (size_t)len);
    }
}
```
One important distinction: **never reveal in a 404 response whether the file exists**. If a request escapes the document root and you return 404 (because the file at that path does not exist) rather than 403, you have leaked information about the filesystem layout outside the document root. Return 403 for any path that fails the containment check, regardless of whether the underlying path exists. If `realpath()` fails with `ENOENT` for a path that was outside the document root, return 403 anyway — the client should not know whether your `/etc/shadow` file exists.
---
## The Complete `serve_file()` Function
Here is the complete integration, bringing all the pieces together:
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <ctype.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <limits.h>
#include <time.h>
#include "http_parse.h"
// doc_root: the canonical (already realpath'd) document root, e.g. "/var/www/html"
// req: the parsed HTTP request from Milestone 2
// send_body: 1 for GET, 0 for HEAD
void serve_file(int client_fd, const char *doc_root,
                const http_request_t *req, int send_body) {
    // ── Stage 1: URL-decode the path ──────────────────────────────────────
    char decoded_path[MAX_PATH_LEN];
    if (url_decode(req->path, decoded_path, sizeof(decoded_path)) < 0) {
        send_400(client_fd);
        return;
    }
    // Reject null bytes in the decoded path
    if (contains_null(decoded_path, strlen(decoded_path) + 1)) {
        send_400(client_fd);
        return;
    }
    // ── Stage 2: Concatenate with document root ───────────────────────────
    char full_path[PATH_MAX];
    if (build_full_path(doc_root, decoded_path, full_path, sizeof(full_path)) < 0) {
        send_403(client_fd);
        return;
    }
    // ── Stage 3: Canonicalize with realpath() ─────────────────────────────
    char canonical[PATH_MAX];
    if (realpath(full_path, canonical) == NULL) {
        if (errno == ENOENT || errno == ENOTDIR) {
            send_404(client_fd);
        } else {
            send_403(client_fd);
        }
        return;
    }
    // ── Stage 4: Containment check ────────────────────────────────────────
    size_t root_len = strlen(doc_root);
    if (strncmp(canonical, doc_root, root_len) != 0 ||
        (canonical[root_len] != '/' && canonical[root_len] != '\0')) {
        send_403(client_fd);
        return;
    }
    // ── Stage 4b: Handle directory paths ──────────────────────────────────
    struct stat st;
    if (stat(canonical, &st) < 0) {
        send_404(client_fd);
        return;
    }
    if (S_ISDIR(st.st_mode)) {
        char index_path[PATH_MAX];
        int n = snprintf(index_path, sizeof(index_path),
                         "%s/index.html", canonical);
        if (n < 0 || (size_t)n >= sizeof(index_path)) {
            send_403(client_fd);
            return;
        }
        if (realpath(index_path, canonical) == NULL) {
            send_403(client_fd);  // No index.html → 403, not 404
            return;
        }
        // Re-check containment after following the index path
        if (strncmp(canonical, doc_root, root_len) != 0 ||
            (canonical[root_len] != '/' && canonical[root_len] != '\0')) {
            send_403(client_fd);
            return;
        }
        if (stat(canonical, &st) < 0) {
            send_404(client_fd);
            return;
        }
    }
    // ── Stage 5: Conditional request check ───────────────────────────────
    char last_modified[64];
    struct tm *gmt = gmtime(&st.st_mtime);
    strftime(last_modified, sizeof(last_modified),
             "%a, %d %b %Y %H:%M:%S GMT", gmt);
    const char *mime = get_mime_type(canonical);
    time_t ims = parse_http_date(get_header(req, "if-modified-since"));
    if (ims != (time_t)-1 && st.st_mtime <= ims) {
        send_304(client_fd, mime, last_modified, st.st_size);
        return;
    }
    // ── Stage 6: Open the file ────────────────────────────────────────────
    int file_fd = open(canonical, O_RDONLY);
    if (file_fd < 0) {
        if (errno == EACCES) send_403(client_fd);
        else                 send_404(client_fd);
        return;
    }
    // ── Stage 7: Send headers ─────────────────────────────────────────────
    char header_buf[1024];
    int header_len = snprintf(header_buf, sizeof(header_buf),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %lld\r\n"
        "Last-Modified: %s\r\n"
        "Connection: close\r\n"
        "\r\n",
        mime,
        (long long)st.st_size,
        last_modified);
    if (header_len < 0 || (size_t)header_len >= sizeof(header_buf)) {
        close(file_fd);
        send_500(client_fd);
        return;
    }
    if (send_all(client_fd, header_buf, (size_t)header_len) < 0) {
        close(file_fd);
        return;
    }
    // ── Stage 8: Send body (GET only) ─────────────────────────────────────
    if (send_body) {
        char read_buf[65536];
        ssize_t bytes_read;
        while ((bytes_read = read(file_fd, read_buf, sizeof(read_buf))) > 0) {
            if (send_all(client_fd, read_buf, (size_t)bytes_read) < 0) {
                break;  // Client disconnected
            }
        }
    }
    close(file_fd);
}
```
---
## The Hardware Soul
What happens in the hardware when your server calls `realpath()` and then `read()` on a file?
**`realpath()` — a series of kernel roundtrips**: `realpath()` is implemented in libc by making repeated `lstat()` and `readlink()` syscalls for each component of the path. For a path like `/var/www/html/assets/images/logo.png`, it performs roughly: `lstat("/var")`, `lstat("/var/www")`, `lstat("/var/www/html")`, `lstat("/var/www/html/assets")`, `lstat("/var/www/html/assets/images")`, `lstat("/var/www/html/assets/images/logo.png")`. Each `lstat()` is a syscall that crosses the user/kernel boundary, looks up the path component in the parent directory's inode, and returns metadata. Directory entries are cached in the kernel's **dentry cache** (dcache) — a hash table of `(parent_inode, name)` → `child_inode` mappings. For recently-accessed paths, every `lstat()` is a dcache hit and costs around 500–1000 ns. For cold paths (first access after a long idle period), each component may require a disk read for the directory inode — 100µs or more per component.
**`stat()` and the page cache**: After `realpath()` resolves the canonical path, your `stat()` call retrieves the file's inode metadata (size, timestamps, permissions). This data is stored in the **inode cache** (icache), which is part of the kernel's page cache. A hot file (accessed frequently) will have its inode in L3 cache or at most in RAM. A cold inode requires a disk read from the filesystem's inode table — another 100µs on spinning disk, ~100µs on a modern NVMe drive, ~10µs on the fastest SSDs.
**`read()` and the page cache for file contents**: When you call `read(file_fd, buf, 65536)`, the kernel checks whether the file's data pages are in the **page cache** (a RAM-backed cache of disk file contents). If present (cache hit): the kernel copies from page cache into your `buf` — this is a memory-to-memory copy at ~10 GB/s, typically completing in 1–5µs for 64KB. If absent (cache miss): the kernel initiates a disk read, blocks your thread, waits for the disk to deliver the pages (10ms spinning disk, 100µs NVMe), then copies to your buffer.
**Cache line anatomy of the read loop**: Your `char read_buf[65536]` is a 64KB stack-allocated buffer. After the first `read()` fills it, the CPU's L2/L3 caches contain those 64KB. The subsequent `send_all()` call reads that same buffer sequentially — purely cache-hot. The hardware prefetcher detects the sequential access pattern and loads the next cache lines before your send loop reaches them. This is why the read-then-send loop is so efficient: data flows from page cache → your stack buffer → NIC send buffer in a pipeline that is bottlenecked by network throughput, not CPU memory bandwidth.
**Branch prediction in the read loop**: The `while ((bytes_read = read(...)) > 0)` loop has one branch: "is bytes_read positive?" For a 1MB file with 64KB chunks, this branch is taken 15 times and not-taken once. The branch predictor will predict "positive" from the second iteration onward, paying zero misprediction penalty on the common case. The final not-taken prediction costs ~15 cycles — completely amortized across the 1MB transfer.
**Sequential vs. random file access**: Serving static files is a sequential read workload — the gold standard for I/O performance. Sequential reads allow the OS to **readahead**: the kernel detects that you are reading file pages in order and speculatively prefetches the next pages from disk before you request them. Linux's readahead window starts at 128KB and grows up to 512KB for sustained sequential reads. This means that by the time your `read()` loop requests the second 64KB chunk, the kernel has already fetched the next 128KB–512KB from disk. The effective latency of sequential file reads approaches zero for files that fit in the readahead window.
---
## Design Decisions
| Approach | Pros | Cons | Used By |
|---|---|---|---|
| **`realpath()` + prefix check (chosen)** ✓ | Correct security, handles all bypass vectors | Two syscalls before `open()`, ENOENT on missing files | nginx, Apache |
| `open()` with `O_PATH` + `fstat()` chain | One less syscall, FD-based rather than string-based | More complex, not portable to all POSIX systems | Some high-security servers |
| `openat()` with `O_NOFOLLOW` on each component | Eliminates TOCTOU completely | Extremely verbose, requires manual path walking | Security-critical file servers |
| Approach | Pros | Cons | Used By |
|---|---|---|---|
| **Extension → MIME lookup table (chosen)** ✓ | Simple, predictable, no system dependencies | Misses files with no extension, can be wrong | nginx, Apache httpd |
| `libmagic` / file content sniffing | Correct MIME even for extensionless files | Adds dependency, reads file bytes before serving | Some mail servers, Samba |
| OS `/etc/mime.types` file parsing | System-maintained, stays up to date | Startup overhead, not all systems have it | Debian's `mime-support` package |
| Approach | Pros | Cons | Used By |
|---|---|---|---|
| **`read()` + `send()` loop (chosen)** ✓ | Simple, correct, portable | Two copies (disk→kernel→user→kernel→NIC) | Simple servers, learning projects |
| `sendfile()` | One copy (disk→kernel→NIC), lower CPU | Linux-only, fd-to-fd only, no SSL | nginx, lighttpd (HTTP only) |
| `mmap()` + `send()` | Page cache shared across processes | SIGBUS on file truncation mid-send, complex | Redis RDB serving, some CDNs |
The `sendfile()` alternative deserves a mention. Linux's `sendfile(out_fd, in_fd, offset, count)` sends file data directly from the kernel's page cache to the socket's send buffer without copying it into userspace. Your server skips the `char read_buf[65536]` entirely. This is called a "zero-copy" send because no data is copied into user space. For a server handling 10,000 small file requests per second, `sendfile()` measurably reduces CPU usage. For this milestone, the `read()` + `send()` loop is correct and teaches the fundamentals.
---
## Testing Your File Server
Create a test document root:
```bash
mkdir -p /tmp/webroot/css /tmp/webroot/images
echo '<html><body><h1>Hello!</h1></body></html>' > /tmp/webroot/index.html
echo 'body { color: red; }' > /tmp/webroot/css/style.css
cp /usr/share/pixmaps/debian-logo.png /tmp/webroot/images/logo.png 2>/dev/null || \
  dd if=/dev/urandom of=/tmp/webroot/images/test.png bs=1024 count=10
```
**Serve the index:**
```bash
curl -v http://localhost:8080/
# Expect: 200 OK, Content-Type: text/html, HTML body
```
**Serve a CSS file:**
```bash
curl -v http://localhost:8080/css/style.css
# Expect: 200 OK, Content-Type: text/css
```
**Serve a binary file:**
```bash
curl -v http://localhost:8080/images/logo.png -o /tmp/downloaded.png
# Verify: md5sum /tmp/downloaded.png should match original
md5sum /tmp/webroot/images/logo.png /tmp/downloaded.png
```
**Test 404:**
```bash
curl -v http://localhost:8080/does-not-exist.html
# Expect: 404 Not Found with HTML body
```
**Test directory traversal (the critical security test):**
```bash
# URL-encoded traversal
curl -v "http://localhost:8080/%2e%2e%2f%2e%2e%2fetc/passwd"
# Expect: 403 Forbidden
# Classic ../ traversal
curl -v "http://localhost:8080/../../etc/passwd"
# Expect: 403 Forbidden
# Create a symlink and try to follow it
ln -s /etc /tmp/webroot/escape_attempt
curl -v "http://localhost:8080/escape_attempt/passwd"
# Expect: 403 Forbidden
rm /tmp/webroot/escape_attempt
```
**Test conditional requests (If-Modified-Since):**
```bash
# First request — note the Last-Modified header
curl -v http://localhost:8080/index.html 2>&1 | grep "Last-Modified"
# Example output: Last-Modified: Mon, 01 Jan 2024 12:00:00 GMT
# Second request — send back the timestamp
curl -v -H "If-Modified-Since: Mon, 01 Jan 2024 12:00:00 GMT" \
  http://localhost:8080/index.html
# Expect: 304 Not Modified, no body
# Request with an old timestamp — file should be served fresh
curl -v -H "If-Modified-Since: Thu, 01 Jan 1970 00:00:00 GMT" \
  http://localhost:8080/index.html
# Expect: 200 OK with full body
```
**Test HEAD method:**
```bash
curl -I http://localhost:8080/index.html
# Expect: 200 OK headers only, same Content-Length as GET would return
```
**Test Content-Length accuracy for binary files:**
```bash
# Download the file and check its size matches Content-Length header
curl -s http://localhost:8080/images/test.png -o /tmp/served.bin
wc -c /tmp/served.bin
# Must match Content-Length header value exactly
```
---
## Knowledge Cascade — What You Just Unlocked
**1. `realpath()` and the kernel's `namei()` path walk**: Every call to `open()`, `stat()`, or `realpath()` on Linux triggers an internal kernel function called `namei()` (name-to-inode lookup). `namei()` walks the path component by component, looking up each name in the parent directory's hash table, following symlinks as it encounters them, and respecting mount points. The `realpath()` libc call is essentially a userspace reimplementation of the portion of `namei()` that resolves symlinks and `..` components, done via repeated `lstat()` and `readlink()` syscalls. Understanding this explains why deep directory trees have measurable `open()` latency: each additional level adds one `lstat()` roundtrip through the dcache. This is why database systems that create millions of small files (like early Cassandra deployments using one-file-per-SSTable) encounter performance problems — the directory entry scanning cost grows super-linearly. When you see `perf stat -e dTLB-load-misses` spiking on a file-intensive workload, you are watching `namei()` thrashing the TLB as it jumps between inode table pages.
**2. MIME types connect HTTP to email**: The MIME (Multipurpose Internet Mail Extensions) standard was originally defined for email (RFC 2045) to allow non-text content in messages. The same `Content-Type: image/jpeg` header your browser uses to decide whether to display or download a response is used by your email client to decide whether to display or offer to save an email attachment. When your mail client shows you an inline PDF versus a download button, it is reading the same MIME type mechanism you just implemented. The connection goes deeper: the browser's MIME type sniffing behavior (where it second-guesses the server's `Content-Type` based on the first bytes of the file content) was introduced because web servers in the 1990s routinely sent wrong MIME types, and browsers tried to be helpful. This "helpful" behavior became a security vulnerability: an attacker could upload a file with a `.png` extension containing JavaScript, the server would serve it as `image/png`, but the browser would sniff the JS bytes and execute it. The `X-Content-Type-Options: nosniff` header tells the browser to trust the `Content-Type` and never sniff. You now understand exactly what that header is defending against.
**3. Conditional requests are the foundation of CDN invalidation**: The `If-Modified-Since` / `304 Not Modified` mechanism you just implemented is how every CDN edge cache validates its cached content. When Cloudflare or Fastly needs to check whether a cached object is still fresh, it sends a conditional GET with `If-Modified-Since` (or `If-None-Match` for ETags) to the origin server. The origin returns `304` if the object is unchanged, and the CDN serves from cache without fetching the body. This is also how browser caches work: Chrome stores the `Last-Modified` (and `ETag`) from the first response, and sends it back on subsequent visits. Cache-Control headers (`max-age`, `stale-while-revalidate`) control when the browser makes conditional requests vs. serving directly from cache without any network contact. By implementing `Last-Modified` / `304` from scratch, you have demystified what happens when a CDN "invalidates" a cache entry — invalidation forces the next conditional request to return `200` instead of `304` by changing the origin file's modification time or ETag.
**4. The `sendfile()` system call and zero-copy I/O**: The read-send loop you wrote copies data twice: once from kernel page cache into your `char read_buf[]` (a kernel-to-user copy), and once from `read_buf` into the kernel's socket send buffer (a user-to-kernel copy). Linux's `sendfile(out_fd, in_fd, &offset, count)` syscall eliminates the round-trip through user space: data moves from the file's page cache directly to the socket's send buffer inside the kernel. This "zero-copy" technique is what allows nginx to serve static files with CPU usage that is independent of file size. The same `sendfile()` mechanism underlies Java's `FileChannel.transferTo()`, Go's `io.Copy()` (which uses `sendfile()` when both sides are OS files/sockets), and Python's `socket.sendfile()`. When you see benchmark results showing nginx serving 100GB/s of static files with 1% CPU on a high-bandwidth server, `sendfile()` is a major reason why.
**5. Index files and URL-space vs. filesystem-space conventions**: The convention that `GET /` serves `index.html` is a pure convention, not a protocol requirement. Apache's `DirectoryIndex` directive, nginx's `index` directive, and static site generators like Hugo, Jekyll, and Next.js all implement this mapping. But it is a mapping between two different spaces: URL space (where `/about/` is a resource with an identity) and filesystem space (where `about/index.html` is a file with a path). This conceptual gap is why URL rewriting exists — nginx's `try_files $uri $uri/ /index.html;` is a rule that walks the URL-to-filesystem mapping in order: try the exact path, try the directory with index, fall back to a root-level SPA entry point. Your index-file logic is a minimal version of this same idea. Understanding the URL-to-filesystem mapping is the foundation for understanding how single-page applications route requests: a React app served by nginx needs *all* URL paths to map to `index.html`, and only API paths should 404 — which requires a `try_files` rule that your URL-to-filesystem stage implements.
---
## Common Pitfalls Checklist
Before moving to Milestone 4, verify your implementation handles all of these:
- [ ] `url_decode()` is called before path concatenation — percent-encoded traversal (`%2e%2e`) is neutralized
- [ ] Decoded path is checked for null bytes — `%00` cannot truncate your path string
- [ ] `realpath()` is called before the prefix check — symlinks inside doc_root cannot escape
- [ ] Prefix check includes the character *after* the root — `/var/www/html2` cannot match `/var/www/html`
- [ ] Directory paths trigger index.html lookup, not directory listing — no filesystem structure leaked
- [ ] `realpath()` is re-run after appending `/index.html` — the index file itself might be a symlink
- [ ] Containment check is re-run after the index.html realpath — same security guarantee applied
- [ ] `get_mime_type()` uses `strrchr()` for the last dot — `.tar.gz` maps to `.gz`, not `.tar`
- [ ] `application/octet-stream` is the fallback for unknown extensions — triggers browser download dialog
- [ ] `text/*` MIME types include `; charset=utf-8` — no browser encoding guessing for HTML/CSS/JS
- [ ] `Content-Length` matches `st.st_size` exactly — no byte counting by hand
- [ ] `Last-Modified` is formatted in RFC 7231 HTTP-date format with `gmtime()` — always UTC
- [ ] `If-Modified-Since` is parsed with `strptime()` + `timegm()` — not `mktime()` which applies local timezone
- [ ] `304` response includes `Content-Type`, `Content-Length`, and `Last-Modified` — same as `200`
- [ ] `304` response has no body — even for GET requests
- [ ] Files are opened with `O_RDONLY` via POSIX `open()` — binary safe, no newline translation
- [ ] `file_fd` is `close()`d on every exit path — including error paths — no FD leaks
- [ ] 403 is returned for paths outside doc_root even when the target file does not exist — no information leak about external filesystem
---
<!-- END_MS -->


<!-- MS_ID: http-server-basic-m4 -->
<!-- MS_ID: http-server-basic-m4 -->
# Milestone 4: Concurrent Connections
## Where We Are


In Milestone 1 you built the socket plumbing. In Milestone 2 you wrote the HTTP parser. In Milestone 3 you added file serving with path security. Your server works — it can accept a connection, parse the request, serve the file, and close the connection. The problem is that it can only do one of these at a time.
Open `server.c` and look at `main()`. There is a `while(1)` loop that calls `accept()`, then calls `handle_client()`, then loops back. While `handle_client()` is running — reading from the socket, calling `stat()`, reading the file, sending the response — every other client knocking at the port is sitting in the kernel's accept queue, waiting. If a single client has a slow connection and takes 500ms to receive a 1MB file, every other client waits 500ms before your server even reads their first byte.
This milestone changes that. You will teach your server to handle multiple clients simultaneously, to maintain connections across multiple requests, to enforce timeouts, and to shut down cleanly on demand. These are not conveniences — they are the boundary between a toy and a server.
---
## The Revelation: Threads Are Not Free
Here is the assumption almost every developer makes when they first think about concurrent servers:
> "I will just spawn a thread per connection. Threads are cheap, the OS handles scheduling, and I can move on to the interesting parts."
This assumption is correct in the narrow sense that spawning a thread is simpler than building a thread pool. It is catastrophically wrong in the practical sense that a thread-per-connection server is the easiest server in the world to kill with a single laptop and a few lines of Python.
Let us measure what "threads are cheap" actually costs.
**Stack memory**: Every POSIX thread has a stack. The default stack size on Linux is 8 megabytes. This is virtual memory — the kernel does not immediately commit 8MB of RAM for each thread — but virtual address space is not unlimited either, and on 32-bit systems it absolutely is. At 1,000 concurrent connections, you have 8GB of virtual memory consumed by stacks alone. At 10,000 connections (a modest load for a production web server), that is 80GB. On a 64-bit system with 256GB of address space this does not crash you, but you will run out of physical memory for stack pages well before that, and the kernel will start swapping.
**Creation time**: `pthread_create()` takes approximately 10–15 microseconds on a modern Linux system. This is not the per-request overhead — it is a one-time cost per connection. At 10,000 new connections per second, you spend 100–150ms per second just creating threads. That is 10–15% of your server's CPU budget gone to thread creation overhead before you read a single byte.
**Scheduler cost**: The Linux scheduler is designed to be fair among runnable threads. With 10,000 threads competing for 8 CPU cores, each thread gets 1/1,250th of available CPU time. Most of those threads are blocked on `recv()` at any given instant, which is fine — blocked threads do not consume CPU. But when a burst of requests arrives simultaneously, 1,000 threads all become runnable at once. The scheduler must manage context switches between them, each costing approximately 1–10 microseconds of kernel time. A server with 1,000 simultaneous runnable threads spends more time in the kernel scheduler than in your request-handling code.
**The Slowloris attack**: An attacker opens 1,000 connections to your thread-per-connection server and sends one header byte per second on each connection. Each connection is technically "active" — it is not idle, it is sending data — so a naive timeout based on "no data received" would not close it. Your server has 1,000 threads blocked in `read_request()`, each waiting for the next byte that arrives after a 1-second delay. These threads consume stack memory and scheduler slots. Real clients trying to connect find the server overwhelmed not by serving files, but by maintaining connections to an attacker who is sending one byte per second.

![Concurrency Models: Thread-per-Connection vs. Thread Pool](./diagrams/diag-m4-thread-models-comparison.svg)

The fix is not to make threads cheaper — it is to bound how many you create. A **thread pool** creates N threads at startup, where N is configured by the operator based on available resources. Every connection is handed to one of those N threads. If all N threads are busy, new connections wait in a bounded queue. If the queue is also full, new connections are rejected with `503 Service Unavailable`. The attacker can still open 1,000 slow connections, but after filling the pool and queue, new connections are refused in microseconds — the server degrades gracefully rather than collapsing.
> **The insight**: Thread pools with bounded queues are not an optimization of thread-per-connection. They are a fundamentally different contract with the OS: instead of "I will create as many threads as there are connections," the contract is "I will use exactly N threads, always, regardless of load." This contract is what makes the server's behavior predictable under adversarial input.
---
## POSIX Threads: The Primitives
[[EXPLAIN:posix-threads:-pthread_create,-pthread_mutex_lock,-pthread_cond_wait-—-the-primitives-used-in-m4|POSIX threads: pthread_create, pthread_mutex_lock, pthread_cond_wait — the primitives used in M4]]
Before building the thread pool, you need a working mental model of the three POSIX primitives you will use throughout this milestone.
### `pthread_create()` — Spawning a Thread
```c
#include <pthread.h>
// Thread function signature: takes void*, returns void*
void *thread_function(void *arg) {
    int *value = (int *)arg;
    printf("Thread received: %d\n", *value);
    return NULL;
}
pthread_t tid;
int arg = 42;
int rc = pthread_create(&tid, NULL, thread_function, &arg);
if (rc != 0) {
    fprintf(stderr, "pthread_create failed: %s\n", strerror(rc));
    exit(EXIT_FAILURE);
}
```
`pthread_create()` creates a new thread that begins executing `thread_function(arg)` immediately. The `pthread_t` is a handle to the thread — you use it to join or detach. The thread function receives a single `void *` argument, which you cast to whatever type you actually need. The return value of `pthread_create()` is an errno code (not -1), so you check `rc != 0` rather than `rc < 0`.
Two decisions you must make for every thread:
- **Join** (`pthread_join(tid, NULL)`): wait for the thread to finish. The calling thread blocks until the target thread returns. This is correct for threads you need to wait for (like worker threads during shutdown).
- **Detach** (`pthread_detach(tid)` or `pthread_attr_setdetachstate`): tell the kernel to automatically reclaim the thread's resources when it exits, without waiting. Use this for fire-and-forget threads where you will never call `pthread_join()`. A thread that is neither joined nor detached leaks resources — it is the thread equivalent of forgetting `close(fd)`.
### `pthread_mutex_t` — Mutual Exclusion
A mutex (mutual exclusion lock) ensures that only one thread at a time executes a particular section of code. It has exactly two states: locked and unlocked.
```c
pthread_mutex_t counter_lock = PTHREAD_MUTEX_INITIALIZER;
int shared_counter = 0;
// Thread A                          // Thread B
pthread_mutex_lock(&counter_lock);   pthread_mutex_lock(&counter_lock);
shared_counter++;                    // BLOCKS here until Thread A unlocks
pthread_mutex_unlock(&counter_lock); // ... eventually acquires lock
                                     shared_counter++;
                                     pthread_mutex_unlock(&counter_lock);
```
`PTHREAD_MUTEX_INITIALIZER` is a compile-time initializer for static/global mutexes. For dynamically allocated mutexes, use `pthread_mutex_init(&lock, NULL)` and `pthread_mutex_destroy(&lock)` when done.
The contract: every access to shared mutable data must be done under a mutex. "Shared" means more than one thread can access it. "Mutable" means at least one thread writes to it. Reading without a mutex while another thread might be writing is a **data race** — undefined behavior in C that produces results ranging from stale reads to memory corruption to crashes.
### `pthread_cond_t` — Condition Variables
A mutex protects data. A condition variable signals between threads about the *state* of that data. The classic pattern: "wake me up when the queue is no longer empty."
```c
pthread_mutex_t queue_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  queue_cond = PTHREAD_COND_INITIALIZER;
// Consumer (blocks when queue is empty):
pthread_mutex_lock(&queue_lock);
while (queue_is_empty()) {              // WHILE, not if — see spurious wakeup note below
    pthread_cond_wait(&queue_cond, &queue_lock);
    // pthread_cond_wait atomically:
    //   1. Releases queue_lock
    //   2. Blocks this thread
    //   3. When signaled, re-acquires queue_lock before returning
}
item = queue_dequeue();
pthread_mutex_unlock(&queue_lock);
// Producer (wakes up the consumer):
pthread_mutex_lock(&queue_lock);
queue_enqueue(new_item);
pthread_cond_signal(&queue_cond);       // Wake up ONE waiting thread
pthread_mutex_unlock(&queue_lock);
```
The `while` around `pthread_cond_wait()` is not optional. POSIX allows **spurious wakeups** — a thread can return from `pthread_cond_wait()` even if no signal was sent. Using `while` instead of `if` means you re-check the condition every time you wake up, correctly handling both spurious wakeups and cases where multiple threads raced to consume the newly-added item.
`pthread_cond_broadcast()` wakes all waiting threads instead of just one. Use it when multiple consumers should all re-check the condition (e.g., during shutdown, when you want all worker threads to wake up and see the "shutting down" flag).
---
## Model 1: Thread-Per-Connection
Start with the simpler model before building the pool. Thread-per-connection is wrong for production, but it is the right mental model to have before you understand why the pool is better.
```c
#include <pthread.h>
#include <stdlib.h>
#include "http_parse.h"
#include "file_server.h"
typedef struct {
    int     client_fd;
    char    client_ip[INET_ADDRSTRLEN];
    int     client_port;
} connection_t;
void *handle_connection_thread(void *arg) {
    connection_t *conn = (connection_t *)arg;
    printf("[%s:%d] Connection opened\n", conn->client_ip, conn->client_port);
    handle_client(conn->client_fd);  // Your existing handle_client() from M1-M3
    close(conn->client_fd);
    printf("[%s:%d] Connection closed\n", conn->client_ip, conn->client_port);
    free(conn);  // Free the heap-allocated connection_t
    return NULL;
}
// In main():
while (1) {
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
    if (client_fd < 0) {
        perror("accept");
        continue;
    }
    // Allocate on heap — the thread outlives this loop iteration
    connection_t *conn = malloc(sizeof(connection_t));
    if (conn == NULL) {
        fprintf(stderr, "malloc failed: closing connection\n");
        close(client_fd);
        continue;
    }
    conn->client_fd   = client_fd;
    conn->client_port = ntohs(client_addr.sin_port);
    inet_ntop(AF_INET, &client_addr.sin_addr,
              conn->client_ip, sizeof(conn->client_ip));
    pthread_t tid;
    int rc = pthread_create(&tid, NULL, handle_connection_thread, conn);
    if (rc != 0) {
        fprintf(stderr, "pthread_create: %s\n", strerror(rc));
        close(client_fd);
        free(conn);
        continue;
    }
    // Detach: we won't call pthread_join(); the thread cleans up after itself
    pthread_detach(tid);
}
```
Why does `connection_t` need to be heap-allocated? The `accept()` loop runs continuously, overwriting `client_addr` on each iteration. If you pass a pointer to a stack variable (`&client_addr`) to the thread, that variable will be overwritten by the next `accept()` call before the thread reads it. Heap allocation ensures the connection data lives until the thread frees it.
This server works correctly for low connection counts. Try it with `ab -n 1000 -c 10 http://localhost:8080/` (Apache Bench, 1000 requests, 10 concurrent). It handles them fine. Now try `ab -n 10000 -c 1000`. Watch your system with `htop` — you will see 1000 threads, and the server will start slowing down measurably as the scheduler overhead grows. That is the thread pool's entrance cue.
---
## Model 2: The Thread Pool
A thread pool pre-creates N threads at startup. Each thread runs an infinite loop waiting for work. "Work" is a client file descriptor to handle. Work arrives via a shared **work queue** protected by a mutex and a condition variable.

![Thread Pool Data Structures: Work Queue and Synchronization](./diagrams/diag-m4-thread-pool-internals.svg)

### The Work Queue Data Structure
```c
#define THREAD_POOL_SIZE_DEFAULT 16
#define WORK_QUEUE_CAPACITY      1024
typedef struct {
    int  client_fd;
    char client_ip[INET_ADDRSTRLEN];
    int  client_port;
} work_item_t;
typedef struct {
    work_item_t     queue[WORK_QUEUE_CAPACITY]; // Circular buffer
    int             head;        // Index of next item to dequeue
    int             tail;        // Index where next item will be enqueued
    int             count;       // Current number of items in queue
    pthread_mutex_t lock;
    pthread_cond_t  not_empty;   // Signaled when an item is added
    pthread_cond_t  not_full;    // Signaled when an item is removed
    int             shutdown;    // Set to 1 to signal worker threads to exit
    pthread_t       threads[THREAD_POOL_SIZE_DEFAULT];
    int             thread_count;
} thread_pool_t;
```
The queue is a **circular buffer** (also called a ring buffer) — a fixed-size array where `head` and `tail` wrap around using modulo arithmetic. This avoids `memmove()` on every dequeue. [[EXPLAIN:circular-buffer-ring-buffer-—-fixed-size-array-with-head-and-tail-indices-that-wrap-around-using-modulo;-O(1)-enqueue-and-dequeue-without-shifting|Circular buffer (ring buffer) — fixed-size array with head/tail indices that wrap modulo capacity; O(1) enqueue and dequeue without shifting elements]]
Two condition variables instead of one: `not_empty` (workers wait on this when the queue is empty) and `not_full` (the accept loop waits on this if the queue is full, or you can choose to reject with 503 instead). Using two separate condition variables avoids waking the wrong waiter — if you used one cond var, a producer adding an item might wake another producer instead of the waiting consumer.
### Initialization
```c
int thread_pool_init(thread_pool_t *pool, int num_threads) {
    pool->head        = 0;
    pool->tail        = 0;
    pool->count       = 0;
    pool->shutdown    = 0;
    pool->thread_count = num_threads;
    if (pthread_mutex_init(&pool->lock, NULL) != 0) return -1;
    if (pthread_cond_init(&pool->not_empty, NULL) != 0) return -1;
    if (pthread_cond_init(&pool->not_full, NULL) != 0)  return -1;
    for (int i = 0; i < num_threads; i++) {
        int rc = pthread_create(&pool->threads[i], NULL,
                                worker_thread, pool);
        if (rc != 0) {
            fprintf(stderr, "pthread_create worker %d: %s\n", i, strerror(rc));
            // Signal already-created threads to exit, join them, return error
            pool->shutdown = 1;
            pthread_cond_broadcast(&pool->not_empty);
            for (int j = 0; j < i; j++) pthread_join(pool->threads[j], NULL);
            return -1;
        }
    }
    return 0;
}
```
### The Worker Thread
```c
void *worker_thread(void *arg) {
    thread_pool_t *pool = (thread_pool_t *)arg;
    while (1) {
        pthread_mutex_lock(&pool->lock);
        // Wait until there is work OR a shutdown signal
        while (pool->count == 0 && !pool->shutdown) {
            pthread_cond_wait(&pool->not_empty, &pool->lock);
        }
        // Check shutdown AFTER the wait — process remaining items before exiting
        if (pool->shutdown && pool->count == 0) {
            pthread_mutex_unlock(&pool->lock);
            return NULL;
        }
        // Dequeue one item
        work_item_t item = pool->queue[pool->head];
        pool->head = (pool->head + 1) % WORK_QUEUE_CAPACITY;
        pool->count--;
        // Signal the accept loop that a slot is now free
        pthread_cond_signal(&pool->not_full);
        pthread_mutex_unlock(&pool->lock);
        // Handle the connection OUTSIDE the lock — this is the whole point
        handle_client_keep_alive(item.client_fd, item.client_ip, item.client_port);
        close(item.client_fd);
    }
}
```
The most important line in this function is `handle_client_keep_alive()` happening **after** `pthread_mutex_unlock()`. The mutex is held only for the queue manipulation — a few nanoseconds. The actual work of handling the HTTP request, reading the file, and sending the response happens with no lock held. If you kept the lock during `handle_client()`, every worker thread would be serialized, making your thread pool equivalent to a single-threaded server.
### Enqueueing Work: The Accept Loop
```c
// Returns 0 on success, -1 if queue is full (caller should send 503)
int thread_pool_enqueue(thread_pool_t *pool, int client_fd,
                        const char *client_ip, int client_port) {
    pthread_mutex_lock(&pool->lock);
    if (pool->count >= WORK_QUEUE_CAPACITY) {
        pthread_mutex_unlock(&pool->lock);
        return -1;  // Queue full → caller sends 503
    }
    work_item_t *item = &pool->queue[pool->tail];
    item->client_fd   = client_fd;
    item->client_port = client_port;
    strncpy(item->client_ip, client_ip, sizeof(item->client_ip) - 1);
    item->client_ip[sizeof(item->client_ip) - 1] = '\0';
    pool->tail  = (pool->tail + 1) % WORK_QUEUE_CAPACITY;
    pool->count++;
    pthread_cond_signal(&pool->not_empty);  // Wake one worker
    pthread_mutex_unlock(&pool->lock);
    return 0;
}
```
The updated accept loop:
```c
// Accept loop in main()
while (!server_shutdown) {
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
    if (client_fd < 0) {
        if (errno == EINTR) continue;   // Interrupted by signal — loop and check shutdown
        perror("accept");
        continue;
    }
    char client_ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip));
    int client_port = ntohs(client_addr.sin_port);
    if (thread_pool_enqueue(&pool, client_fd, client_ip, client_port) < 0) {
        // Queue full — reject with 503 Service Unavailable
        const char *resp_503 =
            "HTTP/1.1 503 Service Unavailable\r\n"
            "Content-Type: text/html\r\n"
            "Content-Length: 59\r\n"
            "Connection: close\r\n"
            "Retry-After: 1\r\n"
            "\r\n"
            "<html><body><h1>503 Service Unavailable</h1></body></html>";
        send(client_fd, resp_503, strlen(resp_503), MSG_NOSIGNAL);
        close(client_fd);
    }
}
```
Notice `Retry-After: 1` in the 503 response. This header tells the client to wait 1 second before retrying. Browsers and HTTP clients that respect `Retry-After` will back off automatically, reducing the pile-up during overload. This is your first **backpressure** signal — explicitly communicating "slow down" to callers.
> **Thread pool as backpressure**: The bounded queue that triggers 503 when full is your first encounter with backpressure — the mechanism by which a system under load signals to its inputs to slow down or stop. The same pattern appears everywhere a pipeline has a bottleneck: Go channels with fixed capacity (`make(chan int, 16)`) block the sender when full; Kafka consumer lag triggers producer throttling; TCP's receive window shrinks to zero when the receiver cannot keep up, causing the sender to stall. The bounded thread pool queue is the network programming equivalent of a full TCP receive buffer. Understanding this pattern here lets you recognize it instantly anywhere else.
---
## HTTP/1.1 Keep-Alive: One Connection, Many Requests
In the servers you built in Milestones 1–3, every connection was one request followed by one response followed by `close()`. This is called **connection-per-request** or HTTP/1.0 behavior. It is simple and correct, but it is expensive.
Here is why. A TCP connection requires a three-way handshake before the first byte of data can be sent: SYN from client, SYN-ACK from server, ACK from client. On a network with 50ms round-trip time, that handshake costs 75ms (1.5 round-trips) before the HTTP request is even transmitted. For a web page that references 20 CSS files, 30 JavaScript files, and 50 images, HTTP/1.0 behavior would require 100 separate TCP connections — 100 × 75ms = 7.5 seconds just in handshake overhead, before reading a single byte of content.
HTTP/1.1 **keep-alive** (officially called "persistent connections") solves this by keeping the TCP connection open after the response is sent, allowing additional requests to use the same connection. The browser can send `GET /style.css`, `GET /script.js`, and `GET /logo.png` all on the same connection, paying the handshake cost only once.

![HTTP/1.1 Keep-Alive: Single Connection, Multiple Requests](./diagrams/diag-m4-keepalive-connection-loop.svg)

### The Keep-Alive Loop
Replace your single-request `handle_client()` with a loop that processes multiple requests per connection:
```c
#define KEEPALIVE_TIMEOUT_SECONDS 30
#define MAX_REQUESTS_PER_CONNECTION 100  // Prevent runaway connections
void handle_client_keep_alive(int client_fd,
                               const char *client_ip,
                               int client_port) {
    char req_buf[REQUEST_BUF_SIZE];
    int  requests_handled = 0;
    while (requests_handled < MAX_REQUESTS_PER_CONNECTION) {
        // ── Read the next request with timeout ────────────────────────────
        ssize_t req_len = read_request_with_timeout(client_fd, req_buf,
                                                     sizeof(req_buf),
                                                     KEEPALIVE_TIMEOUT_SECONDS);
        if (req_len <= 0) {
            // 0 = client closed connection (orderly shutdown)
            // -1 = timeout or error
            break;
        }
        // ── Parse the request ─────────────────────────────────────────────
        http_request_t req;
        int rc = http_parse_request(req_buf, (size_t)req_len, &req);
        if (rc != 0) {
            const char *err = (rc == -2) ? HTTP_414 : HTTP_400;
            send_all(client_fd, err, strlen(err));
            break;  // Don't continue on a malformed request
        }
        requests_handled++;
        printf("[%s:%d] %s %s\n",
               client_ip, client_port, req.method, req.path);
        // ── Determine keep-alive intent ───────────────────────────────────
        int keep_alive = should_keep_alive(&req);
        // ── Serve the request ─────────────────────────────────────────────
        serve_request(client_fd, &req, keep_alive);
        // ── Close if requested ────────────────────────────────────────────
        if (!keep_alive) {
            break;
        }
    }
}
```
### The Keep-Alive Decision
```c
// Returns 1 if the connection should be kept alive after this response.
int should_keep_alive(const http_request_t *req) {
    // HTTP/1.0 default: close.  HTTP/1.1 default: keep-alive.
    int http11 = (strcmp(req->version, "HTTP/1.1") == 0);
    const char *connection = get_header(req, "connection");
    if (connection != NULL) {
        if (strcasecmp(connection, "close") == 0)      return 0;
        if (strcasecmp(connection, "keep-alive") == 0) return 1;
    }
    // No explicit Connection header — use version default
    return http11 ? 1 : 0;
}
```
This mirrors the RFC 7230 Section 6.3 rule exactly: HTTP/1.1 connections are persistent by default unless `Connection: close` is sent; HTTP/1.0 connections close by default unless `Connection: keep-alive` is sent.
### Propagating Keep-Alive to the Response
Your response headers must include a `Connection` header that reflects your decision:
```c
// In serve_file() / build_response_headers():
const char *connection_header = keep_alive
    ? "Connection: keep-alive\r\n"
    : "Connection: close\r\n";
```
If you send `Connection: keep-alive` but then close the socket, the client will be confused — it sent a second request that you will never answer. If you send `Connection: close` but keep the socket open, the client will close its side and your `read_request()` will see EOF. Consistency between your `Connection` header and your actual behavior is mandatory.
---
## Per-Connection Idle Timeouts
Without timeouts, a client that opens a connection and then goes silent holds a worker thread and a file descriptor indefinitely. With a pool of 16 threads, 16 such clients paralyze your server. This is essentially the Slowloris attack in slow motion.

![Per-Connection Idle Timeout: select() with Timeout](./diagrams/diag-m4-idle-timeout-implementation.svg)

The mechanism: `select()` with a timeout on the client socket before calling `recv()`. If the socket does not become readable within the timeout period, close the connection.
```c
// Reads a complete HTTP request with a per-read timeout.
// Returns total bytes read on success, 0 on clean EOF, -1 on timeout/error.
ssize_t read_request_with_timeout(int fd, char *buf, size_t buf_size,
                                   int timeout_seconds) {
    size_t total = 0;
    while (total < buf_size - 1) {
        // Wait up to timeout_seconds for the socket to be readable
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(fd, &read_fds);
        struct timeval tv;
        tv.tv_sec  = timeout_seconds;
        tv.tv_usec = 0;
        int ready = select(fd + 1, &read_fds, NULL, NULL, &tv);
        if (ready < 0) {
            if (errno == EINTR) continue;  // Interrupted by signal, retry
            return -1;                     // Error
        }
        if (ready == 0) {
            // Timeout: socket was not readable within timeout_seconds
            return -1;
        }
        // Socket is readable — perform the actual read
        ssize_t n = recv(fd, buf + total, buf_size - 1 - total, 0);
        if (n < 0) return -1;
        if (n == 0) return 0;   // Clean EOF
        total += n;
        buf[total] = '\0';
        if (strstr(buf, "\r\n\r\n") != NULL) {
            return (ssize_t)total;
        }
    }
    return -1;  // Buffer exhausted
}
```
`select()` takes a set of file descriptors to monitor and a timeout. It blocks until at least one FD in the set becomes readable, or until the timeout expires. [[EXPLAIN:select()-and-fd_set-—-monitoring-multiple-file-descriptors-for-readability-or-writeability-with-an-optional-timeout;-predecessor-to-poll()-and-epoll()|select() and fd_set — monitoring file descriptors for I/O readiness with an optional timeout; predecessor to poll() and epoll()]]
When `select()` returns 0, the timeout elapsed without any data arriving. You return -1 from `read_request_with_timeout()`, and the caller breaks out of the keep-alive loop and closes the connection. The worker thread is then free to handle the next client.
The timeout applies **per read operation** within a request, not per connection total. This is the right granularity: it allows a slow but active client to send headers byte by byte over 30 seconds (unusual but not malicious), while killing a completely idle connection after 30 seconds.
### SO_RCVTIMEO: The Socket-Level Timeout
An alternative to `select()` is to set a timeout on the socket itself using `setsockopt()`:
```c
struct timeval tv = { .tv_sec = 30, .tv_usec = 0 };
setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
// Now every recv() on this socket will return -1/EAGAIN after 30 seconds of inactivity
```
With `SO_RCVTIMEO` set, `recv()` returns -1 with `errno == EAGAIN` (or `EWOULDBLOCK`) if no data arrives within the timeout. This is simpler than `select()` for the single-socket case. The tradeoff is portability — `SO_RCVTIMEO` behavior is more consistent than it used to be but still has edge cases on some systems. The `select()`-based approach is more explicit and portable.
> **Idle timeouts as DoS defense**: The per-connection timeout you just implemented is identical in concept to nginx's `keepalive_timeout` directive (default: 75 seconds) and Apache's `Timeout` directive (default: 60 seconds). Both exist for the same reason: without them, idle connections accumulate until the server exhausts its file descriptor limit or thread pool. Understanding this connection — that every timeout in production servers is a resource reclamation mechanism — helps you set timeouts correctly. Too short: legitimate slow clients get disconnected. Too long: idle connections accumulate. The 30-second default is a reasonable starting point for an internal server; internet-facing servers often use 15 seconds or less.
---
## Graceful Shutdown
A server that crashes on `Ctrl-C` will close all in-flight connections abruptly. A client that received the first half of a 1MB file download will see a truncated response. Browsers retry, but HTTP clients in scripts or applications may not. Graceful shutdown means: stop accepting new connections, let current requests finish, then exit.

![Graceful Shutdown: SIGTERM to Clean Exit Sequence](./diagrams/diag-m4-graceful-shutdown-sequence.svg)

The mechanism has two parts: catching the signal and propagating the shutdown intent through the server.
### The Signal Handling Problem in Multithreaded Programs

![Signal Delivery in Multithreaded Processes: The Safe Pattern](./diagrams/diag-m4-signal-handling-multithreaded.svg)

Here is the problem: in a multithreaded process, when a signal arrives, the kernel delivers it to **an arbitrary thread** — whichever thread happens to be convenient. This means `SIGTERM` might be delivered to one of your worker threads mid-request. If your signal handler modifies shared state, it runs concurrently with whatever that thread was doing, potentially corrupting data.
The correct pattern is:
1. **Block signals in all worker threads** using `pthread_sigmask()` at thread creation time. This prevents signals from being delivered to worker threads.
2. **Have the main thread or a dedicated signal thread call `sigwait()`** to wait for signals synchronously.
```c
// Call this BEFORE creating worker threads, in main():
// Block SIGTERM and SIGINT in the main thread...
sigset_t signal_mask;
sigemptyset(&signal_mask);
sigaddset(&signal_mask, SIGTERM);
sigaddset(&signal_mask, SIGINT);
// Threads inherit their parent's signal mask.
// By blocking here before creating workers, all worker threads will
// also have SIGTERM/SIGINT blocked.
pthread_sigmask(SIG_BLOCK, &signal_mask, NULL);
```
Then, instead of a signal handler, the main thread waits for the signal synchronously:
```c
// After creating the thread pool and starting the accept loop in a thread:
// Main thread waits for a shutdown signal
int sig;
sigwait(&signal_mask, &sig);  // Blocks until SIGTERM or SIGINT arrives
printf("\nReceived signal %d — initiating graceful shutdown...\n", sig);
// Now initiate shutdown
```
For this project, a simpler approach using an `atomic` flag with `signal()` also works:
```c
#include <stdatomic.h>
static atomic_int server_shutdown = 0;
void signal_handler(int sig) {
    (void)sig;
    server_shutdown = 1;  // Atomic write — safe from signal handler
}
// In main():
signal(SIGTERM, signal_handler);
signal(SIGINT,  signal_handler);
// Accept loop condition:
while (!atomic_load(&server_shutdown)) {
    // ...
}
```
`atomic_int` ensures that the write in the signal handler and the read in the accept loop are not subject to compiler reordering or visibility issues. Without `atomic` or `volatile`, the compiler is free to cache `server_shutdown` in a register and never re-read it from memory — making the loop spin forever even after the signal arrives.
[[EXPLAIN:volatile-vs-atomic-in-signal-handlers-—-why-volatile-is-insufficient-for-cross-thread-visibility-and-atomic_int-is-the-correct-type-for-flags-modified-from-signal-handlers|volatile vs. atomic in signal handlers — why volatile alone is insufficient and atomic_int is the correct type]]
### Shutting Down the Thread Pool
After the accept loop exits, you need to shut down the thread pool cleanly:
```c
void thread_pool_shutdown(thread_pool_t *pool) {
    pthread_mutex_lock(&pool->lock);
    pool->shutdown = 1;
    // Wake ALL workers so they can see the shutdown flag
    pthread_cond_broadcast(&pool->not_empty);
    pthread_mutex_unlock(&pool->lock);
    // Wait for all worker threads to finish their current requests
    for (int i = 0; i < pool->thread_count; i++) {
        pthread_join(pool->threads[i], NULL);
    }
    pthread_mutex_destroy(&pool->lock);
    pthread_cond_destroy(&pool->not_empty);
    pthread_cond_destroy(&pool->not_full);
    printf("All worker threads exited. Server shutdown complete.\n");
}
```
`pthread_cond_broadcast()` wakes all worker threads simultaneously — they all re-check the `while (pool->count == 0 && !pool->shutdown)` condition, see `shutdown == 1`, and return. The `pthread_join()` calls wait for each thread to finish handling its current in-flight request before the main thread proceeds.
The sequence: signal arrives → `server_shutdown` set to 1 → accept loop exits → no new connections accepted → `thread_pool_shutdown()` called → workers finish in-flight requests → workers exit → `pthread_join()` returns → `close(server_fd)` → process exits. Every request in progress at shutdown time completes before the process terminates.
> **Signal handling cross-domain**: The `sigwait()` pattern you just learned — block signals in workers, handle them synchronously in a dedicated thread — is how every major production system handles signals in multithreaded code. Java's `Runtime.addShutdownHook()` runs user-provided threads when a JVM shutdown signal arrives; the JVM internally does what you did: a dedicated signal-handling thread receives `SIGTERM` and triggers the shutdown sequence. Go's `signal.Notify()` sends the signal to a Go channel, where a goroutine processes it — again, synchronous consumption of an asynchronous event. nginx's graceful reload (`nginx -s reload`) uses `SIGHUP` delivered to the master process, which then signals workers via a pipe. Understanding the threading model behind signal handling eliminates an entire class of confusing non-deterministic bugs.
---
## Shared State: The Connection Counter and Access Log
With multiple threads running simultaneously, any data they both touch must be protected. Two common examples: a connection counter (for monitoring) and an access log (for auditing).

![Mutex Contention and Cache-Line False Sharing](./diagrams/diag-m4-mutex-false-sharing.svg)

### The Connection Counter
```c
typedef struct {
    pthread_mutex_t lock;
    int             active_connections;
    int             total_connections;
} server_stats_t;
server_stats_t g_stats = {
    .lock                = PTHREAD_MUTEX_INITIALIZER,
    .active_connections  = 0,
    .total_connections   = 0,
};
void stats_connection_opened(void) {
    pthread_mutex_lock(&g_stats.lock);
    g_stats.active_connections++;
    g_stats.total_connections++;
    pthread_mutex_unlock(&g_stats.lock);
}
void stats_connection_closed(void) {
    pthread_mutex_lock(&g_stats.lock);
    g_stats.active_connections--;
    pthread_mutex_unlock(&g_stats.lock);
}
```
Call `stats_connection_opened()` at the start of `handle_client_keep_alive()` and `stats_connection_closed()` at the end. The mutex ensures that two threads incrementing simultaneously produce the correct result, not the racy result of a non-atomic read-modify-write. [[EXPLAIN:data-race-on-non-atomic-increment-—-why-counter++-is-not-thread-safe-without-a-mutex-or-atomic-operation-even-on-x86|Data race on non-atomic increment — why counter++ is not thread-safe without a mutex or atomic operation, even on x86]]
### Cache-Line False Sharing
There is a subtle performance issue with mutexes and counters that matters when you push for high throughput. Consider this struct:
```c
typedef struct {
    pthread_mutex_t lock;    // 40 bytes on Linux
    int             counter; //  4 bytes
    // ...padding to 64 bytes total...
} stats_t;
```
`pthread_mutex_t` is 40 bytes on Linux. The counter is 4 bytes. Together, they fit in a single 64-byte cache line. Every time any thread acquires the mutex, the hardware writes to the mutex's internal state. Every write to a cache line **invalidates that line in every other CPU core's cache** — this is the MESI cache coherence protocol. [[EXPLAIN:MESI-cache-coherence-protocol-—-how-multi-core-CPUs-keep-their-L1-caches-consistent-using-Modified/Exclusive/Shared/Invalid-states;-why-writing-to-a-shared-cache-line-causes-inter-core-traffic|MESI cache coherence protocol — how multi-core CPUs keep their caches consistent; why writes to a shared cache line cause all other cores to reload that line]]
If 16 threads are competing for `g_stats.lock`, the cache line containing the mutex bounces between cores on every lock/unlock. This is called **false sharing** — threads are not sharing the *data* logically, but they share the *cache line* physically. High-frequency mutex operations can saturate the inter-core interconnect.
For a simple connection counter, the impact is small — you lock briefly, increment, unlock. But the principle matters for hot paths. High-performance servers use one of two approaches:
- **Per-thread counters**: each thread has its own counter, atomically merged when statistics are read. Zero contention during the fast path.
- **`stdatomic` operations**: `atomic_fetch_add(&counter, 1)` on an `atomic_int` uses a single `LOCK ADD` x86 instruction. Faster than a mutex for simple increments, but still causes cache-line bouncing between cores.
For this milestone, the mutex approach is correct and the performance difference is negligible. Know the principle for when it matters.
### The Access Log
An access log shared across threads needs a mutex to prevent interleaved writes:
```c
typedef struct {
    pthread_mutex_t lock;
    FILE           *file;
} access_log_t;
access_log_t g_access_log;
void access_log_init(const char *path) {
    pthread_mutex_init(&g_access_log.lock, NULL);
    g_access_log.file = fopen(path, "a");
    if (g_access_log.file == NULL) {
        perror("access log fopen");
    }
}
void access_log_write(const char *client_ip, int client_port,
                      const char *method, const char *path,
                      int status_code, long long bytes_sent) {
    // Get the current time for the log entry
    time_t now = time(NULL);
    struct tm *gmt = gmtime(&now);
    char time_buf[64];
    strftime(time_buf, sizeof(time_buf), "%d/%b/%Y:%H:%M:%S +0000", gmt);
    pthread_mutex_lock(&g_access_log.lock);
    fprintf(g_access_log.file,
            "%s:%d - - [%s] \"%s %s\" %d %lld\n",
            client_ip, client_port, time_buf,
            method, path, status_code, bytes_sent);
    fflush(g_access_log.file);  // Flush so data survives a crash
    pthread_mutex_unlock(&g_access_log.lock);
}
```
`fflush()` inside the lock ensures that log data is written to the OS buffer before you release the mutex. Without it, a crash between the `fprintf` (which writes to libc's buffer) and the eventual `fflush` (which writes to the kernel's buffer) could lose log entries. For a server, losing log entries during a crash is unacceptable — the crash event is exactly when you most need the log.
---
## Putting It All Together
Here is the complete `main()` that integrates all four components: socket, thread pool, graceful shutdown, and shared state:
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <stdatomic.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <pthread.h>
#include "thread_pool.h"
#include "http_parse.h"
#include "file_server.h"
#include "access_log.h"
#define DEFAULT_PORT        8080
#define DEFAULT_POOL_SIZE   16
#define DEFAULT_DOC_ROOT    "/var/www/html"
static atomic_int server_shutdown = 0;
static thread_pool_t g_pool;
void signal_handler(int sig) {
    (void)sig;
    atomic_store(&server_shutdown, 1);
}
int main(int argc, char *argv[]) {
    int         port      = DEFAULT_PORT;
    int         pool_size = DEFAULT_POOL_SIZE;
    const char *doc_root  = DEFAULT_DOC_ROOT;
    // Simple arg parsing: ./server [port] [pool_size] [doc_root]
    if (argc >= 2) port      = atoi(argv[1]);
    if (argc >= 3) pool_size = atoi(argv[2]);
    if (argc >= 4) doc_root  = argv[3];
    // Ignore SIGPIPE — broken writes return EPIPE, don't kill the process
    signal(SIGPIPE, SIG_IGN);
    // Set up graceful shutdown signal handlers
    signal(SIGTERM, signal_handler);
    signal(SIGINT,  signal_handler);
    // Initialize access log
    access_log_init("access.log");
    // Initialize thread pool
    if (thread_pool_init(&g_pool, pool_size) < 0) {
        fprintf(stderr, "Failed to initialize thread pool\n");
        return EXIT_FAILURE;
    }
    printf("Thread pool: %d workers, queue capacity: %d\n",
           pool_size, WORK_QUEUE_CAPACITY);
    // Create and bind server socket
    int server_fd = create_server_socket(port);
    printf("Listening on port %d, serving from %s\n", port, doc_root);
    // Accept loop
    while (!atomic_load(&server_shutdown)) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd,
                               (struct sockaddr *)&client_addr,
                               &client_len);
        if (client_fd < 0) {
            if (errno == EINTR) continue;   // Signal interrupted accept — check shutdown
            if (errno == EMFILE || errno == ENFILE) {
                // FD limit reached — wait briefly and retry
                fprintf(stderr, "FD limit reached: %s\n", strerror(errno));
                usleep(100000);  // 100ms back-off
                continue;
            }
            perror("accept");
            continue;
        }
        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip));
        int client_port = ntohs(client_addr.sin_port);
        if (thread_pool_enqueue(&g_pool, client_fd, client_ip, client_port) < 0) {
            // Pool queue full — send 503 and close immediately
            const char *resp =
                "HTTP/1.1 503 Service Unavailable\r\n"
                "Content-Length: 0\r\n"
                "Connection: close\r\n"
                "Retry-After: 1\r\n"
                "\r\n";
            send(client_fd, resp, strlen(resp), MSG_NOSIGNAL);
            close(client_fd);
            fprintf(stderr, "Queue full: rejected connection from %s:%d\n",
                    client_ip, client_port);
        }
    }
    printf("Shutdown signal received. Waiting for in-flight requests...\n");
    close(server_fd);               // Stop accepting new connections at OS level
    thread_pool_shutdown(&g_pool);  // Wait for workers to drain
    printf("Graceful shutdown complete.\n");
    return EXIT_SUCCESS;
}
```
Notice the `EMFILE`/`ENFILE` handling in the accept loop. `EMFILE` means this process has hit its per-process FD limit; `ENFILE` means the system-wide FD table is full. Both are recoverable errors if the cause is temporary (a burst of connections all closing soon). Rather than crashing, the server backs off for 100ms and retries — an active connection closing in that window frees an FD slot.
---
## The Hardware Soul
What happens in the hardware when 16 worker threads are actively serving requests?
**Thread context switches**: Each worker thread runs on a physical CPU core. When a worker blocks on `recv()` (waiting for the client to send more data), the Linux scheduler **descheduled** that thread and schedules another runnable thread on the same core. This context switch costs approximately 1–10 microseconds and involves saving the thread's register state (including SSE/AVX registers if the thread used them), switching the TLB's address-space context, and loading the new thread's registers. On a 4-core machine with 16 worker threads, at any given moment some workers are runnable (actively handling requests) and others are blocked (waiting on `recv()`, `read()`, or `send()`). The scheduler keeps cores busy by running the runnable workers. If all 16 workers are simultaneously blocked on I/O, all 4 cores are idle — your CPU is at 0% utilization, waiting for the network.
**Mutex acquire/release and the cache**: When a worker thread calls `pthread_mutex_lock(&pool->lock)`, the hardware executes a compare-and-swap (CAS) instruction (`LOCK CMPXCHG` on x86). This instruction atomically reads the mutex's current state and writes "locked" only if it was "unlocked." The `LOCK` prefix causes the CPU to acquire exclusive ownership of the cache line containing the mutex — it sends a "Request for Ownership" (RFO) message to all other cores, invalidating their copies. Every competing thread that tries to acquire the same mutex causes inter-core cache traffic on the memory bus. For a thread pool queue mutex that is acquired and released at most once per connection dispatch, this overhead is negligible. For a mutex inside a tight inner loop (say, one increment per byte received), this overhead would dominate.
**Memory layout of the thread pool**: `thread_pool_t` contains the circular buffer (`work_item_t queue[1024]`), the head/tail/count integers, the mutex, and the condition variables. At `sizeof(work_item_t)` ≈ 70 bytes and 1024 entries, the queue is about 70KB — larger than L2 cache (typically 256KB–4MB) but smaller than L3 (typically 8–64MB). During a burst of connection arrivals, the producer (accept loop) and consumer (worker threads) both touch queue entries. They access different entries (producer at `tail`, consumers at `head`), so there is no false sharing between producer and consumer accesses — the circular buffer design intentionally keeps them apart in memory.
**`select()` timeout syscall cost**: Each call to `select()` in `read_request_with_timeout()` is a system call. On Linux, the syscall overhead is approximately 100–300 nanoseconds (down from ~1µs before Spectre/Meltdown mitigations reduced syscall performance). For a connection that stays idle for 30 seconds, you make one `select()` call and it blocks the thread in the kernel's wait queue for 30 seconds. The kernel's timer wheel wakes the thread when the timeout expires — zero CPU cost during the wait. The thread consumes no CPU while blocked; only its stack memory (8MB virtual, a few KB physical for the active pages) and the kernel's wait queue entry.
---
## Measuring What You Built
Before declaring this milestone done, measure your server under realistic conditions. These are the numbers you should be able to produce and explain.
**Baseline throughput** (no concurrency bottleneck):
```bash
# Install Apache Bench
apt-get install apache2-utils  # or brew install httpie wrk
# 10,000 requests, 100 concurrent
ab -n 10000 -c 100 http://localhost:8080/index.html
```
Look at:
- Requests per second
- Mean time per request
- Connection Time and Processing Time percentiles
**Keep-alive vs. connection-per-request**:
```bash
# With keep-alive (default for HTTP/1.1)
ab -n 10000 -c 100 -k http://localhost:8080/index.html
# Without keep-alive
ab -n 10000 -c 100 http://localhost:8080/index.html
```
You should see the keep-alive run process significantly more requests per second — the TCP handshake overhead is eliminated.
**FD leak verification** (critical correctness test):
```bash
# Record initial FD count
ls /proc/$SERVER_PID/fd | wc -l
# Run 10,000 sequential connections
ab -n 10000 -c 1 http://localhost:8080/index.html
# Verify FD count returned to baseline
ls /proc/$SERVER_PID/fd | wc -l
```
If the final count is higher than the initial count, you have an FD leak. This is the most important test in this milestone.
**Pool saturation** (triggers 503):
```bash
# Exhaust the pool with slow connections using wrk with a long delay
# Then connect with a fast client and verify 503 response
# (requires wrk with Lua scripts or similar load tool)
```
**Graceful shutdown verification**:
```bash
# Start a large download in the background
curl http://localhost:8080/large_file.bin -o /dev/null &
# Immediately send SIGTERM
kill -TERM $SERVER_PID
# Verify the download completed (not truncated)
wait $!
echo "Exit code: $?"  # Should be 0 (success)
```
---
## Design Decisions


| Concurrency Model | Pros | Cons | Used By |
|---|---|---|---|
| **Thread pool with bounded queue (chosen)** ✓ | Predictable resource use, DoS-resistant, simple to reason about | Head-of-line blocking within pool slots, no I/O multiplexing | nginx (worker processes), Apache MPM Worker, Java thread pool servers |
| Thread-per-connection | Simple code, natural per-request isolation | Unbounded resource use, scheduler overhead at high concurrency | Apache MPM prefork (process-per-connection variant), older Java servers |
| Event loop (epoll + non-blocking I/O) | O(1) connection handling, millions of idle connections | Complex state machines, no natural thread isolation, callback hell | nginx's event model, Node.js, Redis, Go net package |
| Thread pool + epoll (hybrid) | High throughput, bounded threads, handles C10K+ | Most complex, requires non-blocking I/O throughout | Nginx (this is actually what it does), libuv, Netty |
The event loop model deserves a mention because it is how all high-performance production servers actually work at their core. Instead of blocking on `recv()` (which wastes a thread while waiting for data), an event loop uses `epoll_wait()` to wait for *any* of thousands of sockets to become readable, then dispatches the readable ones to handlers. A single thread can manage 100,000 connections because it is never blocked — it only runs when there is data to process. This is why nginx can handle 50,000 concurrent connections on a single process. The thread pool you built here is the conceptually simpler predecessor; once you understand it thoroughly, the event loop model is a natural evolution.
| Keep-Alive Strategy | Pros | Cons | Used By |
|---|---|---|---|
| **Loop until `Connection: close` or timeout (chosen)** ✓ | Amortizes TCP setup cost, correct for HTTP/1.1 | Holds thread/FD longer per client | nginx (keepalive_timeout), Apache |
| Connection-per-request | Simple, natural cleanup, shorter FD hold time | Full TCP handshake per request, slower for multi-asset pages | HTTP/1.0 servers, some simple servers |
| HTTP/2 multiplexing | Multiple streams per connection, no head-of-line blocking at HTTP layer | Binary framing, compression, far more complex implementation | Chrome, Firefox, nginx (with ssl), caddy |
---
## Knowledge Cascade — What You Just Unlocked
**1. Thread pool as universal backpressure**: The bounded queue that rejects with 503 when full is your first hands-on implementation of backpressure — the mechanism by which an overloaded component communicates "slow down" to its upstream. In Go, a channel with fixed capacity (`make(chan int, 16)`) blocks the sender when full — the goroutine equivalent of your queue. In Kafka, consumer lag is a backpressure signal; producers can be configured to block or drop when consumers fall behind. In TCP, the receive window shrinks to zero when the receiver's buffer fills, halting the sender at the protocol level. In your HTTP server, the 503 response tells the client to retry in 1 second. The shapes are different but the invariant is identical: **every pipeline with a bottleneck needs a mechanism to slow its inputs**. Having implemented this in C at the thread level, you will recognize the pattern in every distributed system design.
**2. HTTP/1.1 keep-alive and the road to HTTP/2**: The persistent connections you implemented amortize the TCP three-way handshake (1.5 RTT ≈ 75ms on a 50ms-latency link) across multiple requests. But keep-alive introduces a new problem: **head-of-line blocking**. With one connection and sequential requests, if `GET /large-file.bin` takes 5 seconds to transmit, every subsequent request on that connection (your CSS, JavaScript, images) waits behind it. Browsers responded by opening 6 simultaneous keep-alive connections per server — 6 is not a protocol limit, it is a browser-chosen heuristic to balance parallelism against server load. HTTP/2 solves this differently: it **multiplexes** multiple request-response pairs over a single TCP connection using binary framing, allowing your CSS and JavaScript to download in parallel with the large file without waiting for it to finish. HTTP/3 goes further and replaces TCP with QUIC (a UDP-based protocol) to eliminate TCP's own head-of-line blocking — a single dropped packet in TCP blocks the entire connection, while QUIC's independent streams are not affected by packet loss in other streams. Understanding why keep-alive is not enough — and specifically naming "head-of-line blocking" as the limitation — is the conceptual bridge to HTTP/2.
**3. The event loop model and C10K**: The thread pool you built tops out at around 1,000–10,000 concurrent connections before the OS scheduler overhead and per-thread memory cost degrade performance. The "C10K problem" (handling 10,000 concurrent connections — a paper by Dan Kegel in 1999) is what motivated the shift from thread-per-connection to event-loop architectures. `epoll` (Linux), `kqueue` (BSD/macOS), and `IOCP` (Windows) are the OS primitives that make event loops possible — they watch thousands of sockets simultaneously and notify your code only when a socket has data to process. Node.js, Redis, nginx, and modern async Rust (`tokio`) all use event loops at their core. Your thread pool is the stepping stone: you understand threads, synchronization, and blocking I/O deeply. The event loop model replaces blocking `recv()` with non-blocking sockets + `epoll_wait()`, and replaces one-thread-per-connection with a state machine per connection. The complexity cost is high; the throughput gain at scale is enormous.
**4. POSIX threads and language runtimes**: The `pthread_create`, `pthread_mutex_lock`, and `pthread_cond_wait` calls you wrote directly in C are the primitives that every higher-level threading model compiles down to on Linux. Go's goroutines are green threads multiplexed onto OS threads — the Go scheduler calls `pthread_create` for each OS thread in its pool, then multiplexes thousands of goroutines across those threads. Java's `Thread` object wraps a POSIX thread on Linux. Python's GIL (Global Interpreter Lock) is a `pthread_mutex_t` that allows only one Python bytecode instruction to execute at a time. Rust's `std::thread::spawn` calls `pthread_create`. When a Go goroutine blocks on a channel, the Go scheduler puts the goroutine to sleep and runs another on the same OS thread — the same conceptual operation as your worker thread blocking on `pthread_cond_wait`. You now understand the substrate.
**5. Graceful shutdown as a distributed systems primitive**: The shutdown sequence you implemented — signal arrives, stop accepting, drain in-flight requests, exit — is the fundamental pattern for safe service restarts in distributed systems. Kubernetes sends `SIGTERM` to a pod and waits 30 seconds (`terminationGracePeriodSeconds`) for it to exit before sending `SIGKILL`. That 30-second window exists precisely so the application can drain in-flight requests. Load balancers (nginx upstream, AWS ALB, HAProxy) perform "connection draining" — they stop sending new requests to a server being removed from rotation but wait for existing connections to complete. Your `thread_pool_shutdown()` is the application-side implementation of the same idea. When you deploy a microservice in production and configure its readiness probe and termination handler, you are implementing the same invariant at the platform level that you implemented manually here.
---
## Common Pitfalls Checklist
Before declaring Milestone 4 complete, verify your implementation handles all of these:
- [ ] `pthread_create()` errors are checked and the FD is closed on failure — no FD leaks when thread creation fails
- [ ] Worker threads are either joined (during shutdown) or detached — no "zombie" threads leaking resources
- [ ] `connection_t` or equivalent is heap-allocated, not stack-allocated — no use-after-return when accept loop iterates
- [ ] Work queue enqueue and dequeue are under the same mutex — no concurrent head/tail corruption
- [ ] Worker loop uses `while`, not `if`, around `pthread_cond_wait()` — spurious wakeups handled correctly
- [ ] `pool->shutdown` check happens after the wait, not before — in-flight queue items processed before exit
- [ ] `pthread_cond_broadcast()` used for shutdown, not `pthread_cond_signal()` — all workers wake up
- [ ] `pthread_join()` called for all worker threads during shutdown — main thread waits for drain
- [ ] `server_shutdown` flag is `atomic_int`, not plain `int` — visibility guaranteed across threads
- [ ] `accept()` with `errno == EINTR` uses `continue`, not `break` — signal doesn't kill the accept loop
- [ ] `SO_RCVTIMEO` or `select()`-based timeout is applied to every client socket — idle connections don't hold threads forever
- [ ] Keep-alive loop has a `MAX_REQUESTS_PER_CONNECTION` limit — no single connection can monopolize a worker thread forever
- [ ] `Connection: keep-alive` or `Connection: close` header sent in every response — client knows the connection's fate
- [ ] Access log writes are under a mutex — no interleaved partial writes from concurrent threads
- [ ] `fflush()` called after each log write — data survives a crash
- [ ] 503 response includes `Retry-After` header — cooperative behavior under load
- [ ] `EMFILE` in the accept loop triggers a brief sleep and continue, not exit — recoverable FD exhaustion handled
- [ ] `close(server_fd)` called after accept loop exits — kernel stops queueing new connections immediately
- [ ] FD count verified stable after 10,000 sequential connections — no FD leak
---
<!-- END_MS -->


## System Overview

![System Overview](./diagrams/system-overview.svg)


# TDD

A socket-level HTTP/1.1 static file server built layer by layer: raw TCP socket lifecycle → RFC-compliant request parser → filesystem security pipeline → bounded thread pool with graceful shutdown. Every module exposes the hardware-software contract: bytes on the wire, cache-line cost of mutex acquisition, TLB pressure from realpath(), and scheduler overhead of thread creation. The implementation proceeds as four vertically-stacked modules, each buildable and testable independently.


<!-- TDD_MOD_ID: http-server-basic-m1 -->
# Technical Design Document: TCP Server & HTTP Response
## Module `http-server-basic-m1`
---
## 1. Module Charter
This module creates a single-process, single-threaded TCP server that binds a listening socket, runs a sequential accept loop, reads raw bytes from each client socket until the HTTP end-of-headers delimiter (`\r\n\r\n`) is found, sends a hardcoded HTTP/1.1 200 OK response, and closes the client file descriptor. It is the transport-layer skeleton that every subsequent milestone builds on top of.
This module does **not** parse the HTTP request line, headers, or body. It does **not** serve files from disk. It does **not** spawn threads or handle more than one client at a time. It does **not** implement keep-alive. The only "HTTP knowledge" here is the `\r\n\r\n` end-of-headers sentinel on the read side and a single hardcoded response string on the write side.
**Upstream dependency**: none — this is the foundation layer.
**Downstream dependency**: Milestone 2 will replace the `handle_client()` body with a real parser; the socket lifecycle (`create_server_socket`, accept loop, `close(client_fd)`) survives unchanged into Milestone 4.
**Invariants that must always hold after this module**:
- The server FD is never closed inside the accept loop.
- Every accepted `client_fd` is closed on **every** exit path from `handle_client()`, including error paths.
- `recv()` is always called in a loop; the result of a single call is never treated as a complete request.
- Writing to a closed client socket never kills the process.
---
## 2. File Structure
Create files in this exact order:
```
http-server/
├── 1  server.c          # All implementation for this milestone
├── 2  Makefile          # Build rules
└── 3  test_basic.sh     # Manual test script (Phase 5)
```
No header files are needed in this milestone. Everything is in `server.c`. Milestone 2 will extract a `http_parse.h` / `http_parse.c` pair; do not pre-split.
---
## 3. Complete Data Model
### 3.1 Constants
```c
#define DEFAULT_PORT       8080
#define BACKLOG            128        /* kernel accept-queue depth */
#define REQUEST_BUF_SIZE   8192       /* max bytes accumulated before \r\n\r\n */
```
`BACKLOG = 128` matches Linux's `SOMAXCONN` default. The kernel silently clamps higher values to `SOMAXCONN` anyway. `REQUEST_BUF_SIZE = 8192` matches nginx's default `client_header_buffer_size`; a realistic HTTP/1.1 header set fits comfortably inside 4 KB, giving us 2× headroom.
### 3.2 The Hardcoded Response String
```c
static const char *const HTTP_RESPONSE =
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: text/html; charset=utf-8\r\n"
    "Content-Length: 27\r\n"
    "Connection: close\r\n"
    "\r\n"
    "<h1>Hello from C server!</h1>";
```
**Byte anatomy:**
| Segment | Content | Bytes |
|---|---|---|
| Status line | `HTTP/1.1 200 OK\r\n` | 17 |
| Content-Type header | `Content-Type: text/html; charset=utf-8\r\n` | 41 |
| Content-Length header | `Content-Length: 27\r\n` | 20 |
| Connection header | `Connection: close\r\n` | 19 |
| Header/body separator | `\r\n` | 2 |
| Body | `<h1>Hello from C server!</h1>` | 27 |
| **Total** | | **126** |
**Invariant**: the value in `Content-Length` (27) must equal `strlen("<h1>Hello from C server!</h1>")`. Verify at compile time:
```c
_Static_assert(sizeof("<h1>Hello from C server!</h1>") - 1 == 27,
               "Content-Length does not match body length");
```
`Connection: close` is mandatory here. Without it, HTTP/1.1 clients treat the connection as persistent and block waiting for a second response that never comes.
### 3.3 Memory Layout of the Request Buffer
```
Stack frame of handle_client():
  Offset   Field                  Size
  ------   -----                  ----
  +0       char buf[8192]         8192 bytes  (REQUEST_BUF_SIZE)
  +8192    size_t total           8 bytes
  +8200    ssize_t n              8 bytes
  ...
Total stack usage per call: ~8220 bytes (well within default 8 MB thread stack)
```
The buffer is stack-allocated, not heap-allocated. This means:
- No `malloc` failure path.
- The buffer is automatically freed when `handle_client()` returns.
- It is not shared between calls — each connection gets a fresh zero-state buffer.

![TCP Socket Lifecycle State Machine](./diagrams/tdd-diag-1.svg)

### 3.4 File Descriptor State Machine
Each FD lives in one of three states:
```
[NOT CREATED]
     │  socket()
     ▼
[UNBOUND]       ← server_fd only
     │  bind() + listen()
     ▼
[LISTENING]     ← server_fd only; lives for server lifetime
     │  accept() spawns
     ▼
[CONNECTED]     ← client_fd; created by accept()
     │  read + write
     ▼
[CLOSED]        ← close(client_fd) MANDATORY before loop iteration ends
```
The server FD never enters `[CLOSED]` inside the accept loop. The client FD must reach `[CLOSED]` on every path: normal completion, `read_request` failure, `send_all` failure.

![Kernel File Descriptor Table: server_fd vs. client_fd](./diagrams/tdd-diag-2.svg)

---
## 4. Interface Contracts
### 4.1 `create_server_socket(int port) → int`
```c
int create_server_socket(int port);
```
**Purpose**: create, configure, bind, and listen on a TCP server socket.
**Parameters**:
- `port`: 1–65535. If outside this range, behavior is undefined (caller must validate). The default caller passes `DEFAULT_PORT` (8080).
**Return value**:
- Non-negative `int` (the server file descriptor) on success.
- This function does **not** return on failure — it calls `perror()` and `exit(EXIT_FAILURE)`. Rationale: if we cannot bind the listening socket, the server cannot function; there is nothing to recover to.
**Side effects**:
1. Calls `socket(AF_INET, SOCK_STREAM, 0)`.
2. Sets `SO_REUSEADDR` via `setsockopt()`.
3. Calls `bind()` with `INADDR_ANY` and `htons(port)`.
4. Calls `listen()` with `BACKLOG`.
**Invariants on return**: the returned FD is in the `[LISTENING]` state; `accept()` may be called on it immediately.
**Error behavior**:
| Syscall | On failure | Action |
|---|---|---|
| `socket()` | returns -1 | `perror("socket"); exit(EXIT_FAILURE)` |
| `setsockopt()` | returns -1 | `perror("setsockopt SO_REUSEADDR"); exit(EXIT_FAILURE)` |
| `bind()` | returns -1 | `perror("bind"); exit(EXIT_FAILURE)` |
| `listen()` | returns -1 | `perror("listen"); exit(EXIT_FAILURE)` |
**Key implementation note — `SO_REUSEADDR`**: this option MUST be set before `bind()`. Setting it after has no effect. If omitted, restarting the server within ~60 seconds fails with `EADDRINUSE` due to the TCP `TIME_WAIT` state on the old socket.
**Key implementation note — `htons()`**: `addr.sin_port` is a 16-bit big-endian field. x86 hosts are little-endian. Assigning `port` without `htons()` silently binds to the byte-swapped port number. For port 8080 (0x1F90), the byte-swapped value is 0x901F = 36895. The server starts, `bind()` succeeds, and no client can connect on port 8080. This is the single most common first-timer bug.
### 4.2 `read_request(int fd, char *buf, size_t buf_size) → ssize_t`
```c
ssize_t read_request(int fd, char *buf, size_t buf_size);
```
**Purpose**: accumulate bytes from `fd` into `buf` until either `\r\n\r\n` is found or the buffer is exhausted.
**Parameters**:
- `fd`: a connected client socket FD in the `[CONNECTED]` state.
- `buf`: caller-allocated buffer, at least `buf_size` bytes. Must be writable.
- `buf_size`: total capacity of `buf`, including space for a null terminator. Must be ≥ 5 (to hold at minimum `\r\n\r\n\0`).
**Return value**:
- `> 0`: success. The return value is the total number of bytes written into `buf`. `buf` is null-terminated at position `[return_value]`. The string `\r\n\r\n` is present somewhere within `buf[0..return_value]`.
- `-1`: failure. Causes: `recv()` error, `recv()` returned 0 (client closed connection before sending complete headers), or buffer exhausted without finding `\r\n\r\n`. In all failure cases, the caller must close `fd` and not attempt to use `buf`.
**Postconditions on success**:
- `buf[return_value] == '\0'`
- `strstr(buf, "\r\n\r\n") != NULL`
- `return_value < buf_size` (there is always room for the null terminator)
**Algorithm** (detailed in Section 5.1).
**Error cases**:
| Condition | `recv()` returns | Function returns |
|---|---|---|
| Client closed before `\r\n\r\n` | 0 | -1 |
| Network error / reset | < 0 | -1 |
| Buffer full, no delimiter found | (never called again) | -1 |
| `\r\n\r\n` found | > 0 | total bytes accumulated |
**Thread safety**: not thread-safe; intended for single-threaded use in this milestone.
### 4.3 `send_all(int fd, const char *buf, size_t len) → ssize_t`
```c
ssize_t send_all(int fd, const char *buf, size_t len);
```
**Purpose**: send exactly `len` bytes from `buf` to `fd`, looping until all bytes are sent or an unrecoverable error occurs.
**Parameters**:
- `fd`: a connected client socket FD.
- `buf`: pointer to the data to send. Must be valid for `len` bytes.
- `len`: number of bytes to send. Must be > 0.
**Return value**:
- `(ssize_t)len` on success (all bytes sent).
- `-1` on failure. Causes: `send()` returned -1 (including `EPIPE` from a broken connection), or `send()` returned 0 (should not occur on blocking sockets, guarded anyway).
**`MSG_NOSIGNAL`**: MUST be passed as the flags argument to every `send()` call inside this function. Without it, writing to a connection where the peer has closed their side sends `SIGPIPE` to the process, which by default terminates the process. `MSG_NOSIGNAL` suppresses the signal; the broken-pipe condition is reported as `errno == EPIPE` on the `send()` return value instead.
**Partial write semantics**: the kernel may accept fewer bytes than requested if the socket send buffer is partially full. This is not an error — loop and continue from where the partial write left off.
**Error propagation**: on failure, return -1. Do not call `perror()` inside `send_all()`; let the caller decide whether to log.
### 4.4 `handle_client(int client_fd) → void`
```c
void handle_client(int client_fd);
```
**Purpose**: read one HTTP request from `client_fd`, send the hardcoded response, and return. Does not close `client_fd` — the caller (`main()`) is responsible for `close()`.
**Parameters**:
- `client_fd`: a connected client FD returned by `accept()`.
**Return value**: none.
**Behavior**:
1. Call `read_request(client_fd, buf, sizeof(buf))`.
2. If `read_request` returns ≤ 0: log to `stderr`, return immediately (caller closes FD).
3. Log the first line of the request to `stdout` for debugging.
4. Call `send_all(client_fd, HTTP_RESPONSE, strlen(HTTP_RESPONSE))`.
5. If `send_all` returns < 0: log to `stderr`, return (caller closes FD).
6. Return normally.
**Invariant**: `handle_client()` never closes `client_fd`. This separation of concerns makes it trivial for Milestone 4 to call `handle_client()` from a thread without the thread pool needing special-case logic.
**Logging the request line**: after `read_request()` succeeds, `buf` contains a null-terminated string starting with the request line. Find the first `\r\n` with `strstr(buf, "\r\n")`, temporarily null-terminate there, print, restore the `\r`, and continue. This gives readable logs without mutating the buffer's protocol meaning.
### 4.5 `main(int argc, char *argv[]) → int`
```c
int main(int argc, char *argv[]);
```
**Purpose**: parse the optional port argument, suppress SIGPIPE, create the server socket, run the accept loop.
**Accept loop behavior**:
```
while (1):
    client_fd = accept(server_fd, &client_addr, &client_len)
    if client_fd < 0:
        if errno == EINTR: continue        ← signal interrupted, non-fatal
        if errno == EMFILE: log + sleep(0.1s) + continue  ← FD limit, recoverable
        if errno == ENFILE: log + sleep(0.1s) + continue  ← system FD limit
        perror("accept"); continue         ← other errors: log, keep running
    log client IP:port
    handle_client(client_fd)
    close(client_fd)                       ← ALWAYS, on every path
```
`accept()` failure must **never** break the loop. A single bad `accept()` must not take down the server. The `continue` after every `accept()` error path is load-bearing.
**Port parsing**: `argc == 2` → `port = atoi(argv[1])`. `atoi()` is sufficient here; Milestone 4 may add range validation. No port validation needed for this milestone.
**SIGPIPE suppression at startup**:
```c
signal(SIGPIPE, SIG_IGN);
```
This call must come before any socket operations. It is a process-wide setting; once set, all future `send()`/`write()` calls on broken sockets return -1/EPIPE instead of raising SIGPIPE.
---
## 5. Algorithm Specification
### 5.1 `read_request()` — The Partial-Read Loop
**Input**: connected FD `fd`, writable buffer `buf` of capacity `buf_size`.
**Output**: total bytes accumulated in `buf`, or -1 on failure.
**Step-by-step**:
```
total ← 0
LOOP:
  if total >= buf_size - 1:
    return -1                         ← buffer exhausted, no delimiter found
  n ← recv(fd,
           buf + total,
           buf_size - 1 - total,      ← always leave 1 byte for null terminator
           0)                         ← no flags
  if n < 0:
    if errno == EINTR: CONTINUE LOOP  ← interrupted by signal, retry
    return -1                         ← genuine error (ECONNRESET, etc.)
  if n == 0:
    return -1                         ← peer performed orderly shutdown (TCP FIN)
                                         before sending complete headers
  total ← total + n
  buf[total] ← '\0'                   ← keep buffer null-terminated after each recv()
                                         safe because we reserved buf_size - 1 - total
  if strstr(buf, "\r\n\r\n") ≠ NULL:
    return total                      ← success: complete headers received
  CONTINUE LOOP
```
**Why `buf[total] = '\0'` inside the loop**: `strstr()` is a string function; it reads until it finds `\0`. If `buf` is not null-terminated, `strstr()` reads past the end of valid data, invoking undefined behavior. The null terminator is written at `buf[total]` — the byte immediately after the last `recv()` result — before every `strstr()` call.
**Why `buf_size - 1 - total` as the `recv()` length argument**: this leaves exactly one byte at `buf[total]` available for the null terminator. Without this guard, a buffer-filling response would have `buf[total]` written out of bounds.
**Why check `errno == EINTR`**: signals (including `SIGINT` during development) can interrupt system calls. On Linux, `recv()` returns -1 with `errno == EINTR` when interrupted by a non-fatal signal. Retrying is correct. Treating `EINTR` as a fatal error causes intermittent, hard-to-reproduce failures.
**The CRLF-only delimiter**: the loop searches for `\r\n\r\n`, not `\n\n`. This is correct for HTTP/1.1. `telnet` sends bare `\n`, so the loop will accumulate `telnet` input until a timeout or EOF — but the server will not crash. Milestone 2 adds bare-LF tolerance in the parser; this milestone only needs to not crash.

![Partial Read Problem: TCP Stream vs. Message Assumption](./diagrams/tdd-diag-3.svg)

### 5.2 `send_all()` — The Partial-Write Loop
**Input**: FD `fd`, buffer `buf`, byte count `len`.
**Output**: `(ssize_t)len` on success, -1 on failure.
**Step-by-step**:
```
sent ← 0
LOOP:
  if sent >= len:
    return (ssize_t)len               ← all bytes sent
  n ← send(fd,
           buf + sent,
           len - sent,
           MSG_NOSIGNAL)
  if n < 0:
    return -1                         ← errno is EPIPE (client closed),
                                         ECONNRESET, or other fatal error
  if n == 0:
    return -1                         ← should not occur on blocking socket;
                                         guard against it anyway
  sent ← sent + n
  CONTINUE LOOP
```
**Why partial writes occur**: the kernel's per-socket send buffer has finite capacity (default ~87 KB on Linux). If the send buffer is full — because the client's TCP receive window is full, because the client is reading slowly, or because a large file is being sent — `send()` copies as many bytes as fit and returns fewer than requested. This is not an error. The loop retries until all bytes are accepted by the kernel.
**`MSG_NOSIGNAL` semantics**: this flag is Linux-specific. On macOS/BSD it does not exist; use `SO_NOSIGPIPE` socket option or the process-wide `signal(SIGPIPE, SIG_IGN)` from `main()`. Since this project targets Linux, `MSG_NOSIGNAL` is the per-call mechanism; the process-wide `SIG_IGN` in `main()` provides defense in depth.

![read_request() Algorithm: Accumulation Loop Step-by-Step](./diagrams/tdd-diag-4.svg)

---
## 6. Error Handling Matrix
| Error | Where Detected | Recovery | Logged? | User-Visible? |
|---|---|---|---|---|
| `socket()` fails | `create_server_socket()` | `perror` + `exit(EXIT_FAILURE)` | stderr | Server does not start |
| `setsockopt()` fails | `create_server_socket()` | `perror` + `exit(EXIT_FAILURE)` | stderr | Server does not start |
| `bind()` fails | `create_server_socket()` | `perror` + `exit(EXIT_FAILURE)` | stderr | Server does not start |
| `listen()` fails | `create_server_socket()` | `perror` + `exit(EXIT_FAILURE)` | stderr | Server does not start |
| `accept()` → `EINTR` | `main()` accept loop | `continue` (retry) | No | No |
| `accept()` → `EMFILE` | `main()` accept loop | log + `usleep(100000)` + `continue` | stderr | No (new connections may time out) |
| `accept()` → `ENFILE` | `main()` accept loop | log + `usleep(100000)` + `continue` | stderr | No |
| `accept()` → other | `main()` accept loop | `perror` + `continue` | stderr | No |
| `recv()` → `EINTR` | `read_request()` | `continue` (retry) | No | No |
| `recv()` → 0 (EOF) | `read_request()` | return -1 | No | Client gets no response |
| `recv()` → < 0 | `read_request()` | return -1 | No | Client gets no response |
| Buffer full, no `\r\n\r\n` | `read_request()` | return -1 | stderr in `handle_client` | Client gets no response (FD closed) |
| `send()` → `EPIPE` | `send_all()` | return -1 | stderr in `handle_client` | Client already closed — irrelevant |
| `send()` → other error | `send_all()` | return -1 | stderr in `handle_client` | Client may receive partial response |
| Port byte-order bug | Build-time audit | Use `htons()` | N/A | Server binds wrong port silently |
| FD leak | `close()` omitted | Must `close()` on every path | N/A | `EMFILE` after ~1020 connections |
**Invariant for error handling**: no error path may leave `client_fd` open when `handle_client()` returns. The caller (`main()`) calls `close(client_fd)` unconditionally after `handle_client()` returns — this is the mechanism. `handle_client()` must not call `close()` itself, and must always return (not `exit()`) on errors.
---
## 7. Implementation Sequence with Checkpoints
### Phase 1 — Socket Creation, SO_REUSEADDR, Bind, Listen (0.5–1 h)
Implement `create_server_socket()` in full. Write a `main()` that calls it, prints "Listening on port N", and calls `pause()` (hangs indefinitely).
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#define DEFAULT_PORT  8080
#define BACKLOG       128
int create_server_socket(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); exit(EXIT_FAILURE); }
    int opt = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt SO_REUSEADDR"); exit(EXIT_FAILURE);
    }
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port);        /* <-- htons() is mandatory */
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); exit(EXIT_FAILURE);
    }
    if (listen(fd, BACKLOG) < 0) {
        perror("listen"); exit(EXIT_FAILURE);
    }
    return fd;
}
int main(int argc, char *argv[]) {
    int port = (argc == 2) ? atoi(argv[1]) : DEFAULT_PORT;
    int server_fd = create_server_socket(port);
    printf("Listening on port %d (server_fd=%d)\n", port, server_fd);
    pause();  /* temporary: replaced in Phase 2 */
    close(server_fd);
    return 0;
}
```
**Compile**:
```bash
gcc -Wall -Wextra -o server server.c
```
**Checkpoint 1**: `./server 8080` prints "Listening on port 8080". In another terminal: `ss -tlnp | grep 8080` shows the socket in `LISTEN` state. `Ctrl-C` to kill.
**Verify `htons()` is present**: `grep -n 'sin_port' server.c` must show `htons()` on every assignment.
---
### Phase 2 — Sequential Accept Loop with `close()` on Every Path (0.5 h)
Replace `pause()` with the accept loop. Add a stub `handle_client()`.
```c
void handle_client(int client_fd) {
    /* Phase 2 stub: just log and return */
    (void)client_fd;
    printf("  handle_client() called (stub)\n");
}
int main(int argc, char *argv[]) {
    int port = (argc == 2) ? atoi(argv[1]) : DEFAULT_PORT;
    signal(SIGPIPE, SIG_IGN);                  /* must come before socket ops */
    int server_fd = create_server_socket(port);
    printf("Listening on port %d\n", port);
    while (1) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd,
                               (struct sockaddr *)&client_addr,
                               &client_len);
        if (client_fd < 0) {
            if (errno == EINTR)  { continue; }
            if (errno == EMFILE || errno == ENFILE) {
                fprintf(stderr, "accept: FD limit: %s\n", strerror(errno));
                usleep(100000);
                continue;
            }
            perror("accept");
            continue;        /* never break; never exit */
        }
        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr,
                  client_ip, sizeof(client_ip));
        int client_port = ntohs(client_addr.sin_port);
        printf("Connection from %s:%d (fd=%d)\n",
               client_ip, client_port, client_fd);
        handle_client(client_fd);
        close(client_fd);    /* MANDATORY: on every path */
    }
    close(server_fd);
    return 0;
}
```
**Checkpoint 2**: `./server` + `curl http://localhost:8080/` in another terminal. `curl` hangs (no response sent yet — correct). Your server prints "Connection from 127.0.0.1:XXXXX". `Ctrl-C` on curl. Server keeps running and accepts the next `curl`. No crash.
**FD leak test (Phase 2 baseline)**: `ls /proc/$(pgrep server)/fd | wc -l` — should be 4 (stdin, stdout, stderr, server_fd). Run `for i in $(seq 100); do curl -s http://localhost:8080/ &>/dev/null; done`. FD count returns to 4. If it grows, `close(client_fd)` is missing on some path.
---
### Phase 3 — `read_request()`: Partial-Read Loop with Delimiter Detection (0.5–1 h)
```c
#define REQUEST_BUF_SIZE 8192
ssize_t read_request(int fd, char *buf, size_t buf_size) {
    size_t total = 0;
    while (total < buf_size - 1) {
        ssize_t n = recv(fd,
                         buf + total,
                         buf_size - 1 - total,
                         0);
        if (n < 0) {
            if (errno == EINTR) { continue; }
            return -1;
        }
        if (n == 0) {
            return -1;   /* clean EOF before end-of-headers */
        }
        total += (size_t)n;
        buf[total] = '\0';
        if (strstr(buf, "\r\n\r\n") != NULL) {
            return (ssize_t)total;
        }
    }
    /* buf_size - 1 bytes accumulated, no delimiter: request too large */
    return -1;
}
```
Update `handle_client()` stub to call `read_request()` and log the first request line:
```c
void handle_client(int client_fd) {
    char buf[REQUEST_BUF_SIZE];
    ssize_t req_len = read_request(client_fd, buf, sizeof(buf));
    if (req_len <= 0) {
        fprintf(stderr, "  read_request failed\n");
        return;
    }
    /* Log first line only */
    char *eol = strstr(buf, "\r\n");
    if (eol) { *eol = '\0'; }
    printf("  Request: [%s]\n", buf);
    if (eol) { *eol = '\r'; }
    /* Response: Phase 4 */
}
```
**Checkpoint 3a**: `curl -v http://localhost:8080/` — server prints `Request: [GET / HTTP/1.1]`. `curl` hangs (no response yet). Correct.
**Checkpoint 3b** (partial-read safety): use `telnet localhost 8080`. Type `GET / HTTP/1.1` and press Enter. Type `Host: localhost` and press Enter twice. Server prints the request line. No crash. Telnet sends one line at a time; this exercises the recv() loop with multiple partial reads.
**Checkpoint 3c** (large request rejection): `python3 -c "print('GET /' + 'a'*8200 + ' HTTP/1.1\r\n\r\n', end='')" | nc localhost 8080`. Server returns -1 from `read_request()` (buffer exhausted), logs "read_request failed", closes the FD. No crash.
---
### Phase 4 — Hardcoded HTTP Response, `send_all()`, SIGPIPE Handling (0.5 h)
```c
static const char *const HTTP_RESPONSE =
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: text/html; charset=utf-8\r\n"
    "Content-Length: 27\r\n"
    "Connection: close\r\n"
    "\r\n"
    "<h1>Hello from C server!</h1>";
_Static_assert(sizeof("<h1>Hello from C server!</h1>") - 1 == 27,
               "Content-Length mismatch");
ssize_t send_all(int fd, const char *buf, size_t len) {
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, buf + sent, len - sent, MSG_NOSIGNAL);
        if (n <= 0) {
            return -1;
        }
        sent += (size_t)n;
    }
    return (ssize_t)len;
}
```
Update `handle_client()` final form:
```c
void handle_client(int client_fd) {
    char buf[REQUEST_BUF_SIZE];
    ssize_t req_len = read_request(client_fd, buf, sizeof(buf));
    if (req_len <= 0) {
        fprintf(stderr, "  read_request failed or client disconnected\n");
        return;
    }
    /* Log first line */
    char *eol = strstr(buf, "\r\n");
    if (eol) { *eol = '\0'; }
    printf("  >> %s\n", buf);
    if (eol) { *eol = '\r'; }
    size_t response_len = strlen(HTTP_RESPONSE);
    if (send_all(client_fd, HTTP_RESPONSE, response_len) < 0) {
        fprintf(stderr, "  send_all failed (client may have disconnected)\n");
    }
}
```
**Checkpoint 4a**: `curl -v http://localhost:8080/` shows `HTTP/1.1 200 OK`, all four headers, and `<h1>Hello from C server!</h1>` body. `-v` output must show exactly: `Content-Type: text/html; charset=utf-8`, `Content-Length: 27`, `Connection: close`.
**Checkpoint 4b** (Content-Length verification): `curl -s http://localhost:8080/ | wc -c` must output `27`.
**Checkpoint 4c** (SIGPIPE): `echo -e "GET / HTTP/1.1\r\nHost: localhost\r\n\r\n" | nc -q 0 localhost 8080`. Server logs the request, does not crash. Run 20 times in rapid succession. Server survives all of them.
---
### Phase 5 — Testing (0.5–1 h)
Full test procedure in `test_basic.sh` — see Section 8.

![HTTP/1.1 Response Wire Format: Byte Anatomy](./diagrams/tdd-diag-5.svg)

---
## 8. Test Specification
### 8.1 `create_server_socket()` Tests
**T1.1 — Happy path: binds and listens**
```bash
./server 8080 &
PID=$!
sleep 0.2
ss -tlnp | grep 8080 | grep LISTEN
# Expected: line containing "0.0.0.0:8080" and "LISTEN"
kill $PID
```
**T1.2 — SO_REUSEADDR: fast restart**
```bash
./server 8080 &
PID=$!
sleep 0.2
kill $PID
sleep 0.1   # do NOT wait 60s
./server 8080 &
PID2=$!
sleep 0.2
ss -tlnp | grep 8080 | grep LISTEN  # Must succeed immediately
kill $PID2
```
Failure mode: `bind: Address already in use` within 60 seconds of stopping = `SO_REUSEADDR` missing.
**T1.3 — Port byte order: correct port**
```bash
./server 8080 &
PID=$!
sleep 0.2
ss -tlnp | grep ':8080'   # Must show 8080, NOT 36895 or any other port
kill $PID
```
**T1.4 — Port in use: fails at startup**
```bash
nc -l 8080 &
NC_PID=$!
./server 8080  # Must print "bind: Address already in use" and exit non-zero
kill $NC_PID
```
### 8.2 `read_request()` Tests
**T2.1 — Happy path: single recv() delivers complete headers**
```bash
printf "GET / HTTP/1.1\r\nHost: localhost\r\n\r\n" | nc -q 1 localhost 8080
# Server logs: >> GET / HTTP/1.1
```
**T2.2 — Partial reads: telnet (line-by-line delivery)**
```bash
# Send headers with 100ms delay between lines
(printf "GET / HTTP/1.1\r\n"; sleep 0.1; printf "Host: localhost\r\n"; sleep 0.1; printf "\r\n") \
  | nc -q 1 localhost 8080
# Server must respond with 200 OK, not crash or hang
```
**T2.3 — Client EOF before headers complete**
```bash
printf "GET / HTTP/1.1\r\n" | nc -q 0 localhost 8080
# Server logs "read_request failed". Server continues running.
# Verify: curl http://localhost:8080/ works immediately after.
```
**T2.4 — Buffer exhaustion (oversized request)**
```bash
python3 -c "
import sys
sys.stdout.buffer.write(b'GET /' + b'a'*8200 + b' HTTP/1.1\r\n\r\n')
" | nc -q 1 localhost 8080
# Server logs failure. Does not crash. Next curl works.
```
**T2.5 — Null response on `recv()` == 0**
```bash
# Open connection and immediately close (sends TCP FIN without data)
python3 -c "
import socket, time
s = socket.socket()
s.connect(('localhost', 8080))
s.close()
"
# Server prints "read_request failed". Does not crash.
```
### 8.3 `send_all()` Tests
**T3.1 — Happy path: full response received by curl**
```bash
curl -sv http://localhost:8080/ 2>&1 | grep -E "< (HTTP|Content)"
# Expected:
# < HTTP/1.1 200 OK
# < Content-Type: text/html; charset=utf-8
# < Content-Length: 27
# < Connection: close
```
**T3.2 — Content-Length accuracy**
```bash
BODY=$(curl -s http://localhost:8080/)
echo -n "$BODY" | wc -c
# Expected: 27
```
**T3.3 — SIGPIPE: client closes before response sent**
```bash
for i in $(seq 100); do
  python3 -c "
import socket
s = socket.socket()
s.connect(('localhost', 8080))
s.send(b'GET / HTTP/1.1\r\nHost: localhost\r\n\r\n')
s.close()  # close before server sends response
" &
done
wait
# Server must still be running:
curl -s http://localhost:8080/ | grep -c "Hello"
# Expected: 1
```
### 8.4 `handle_client()` and Accept Loop Tests
**T4.1 — Sequential connections: server handles them all**
```bash
for i in $(seq 50); do
  curl -s http://localhost:8080/ > /dev/null
done
# No errors. Server still running.
```
**T4.2 — FD leak: zero after 10,000 connections**
```bash
SERVER_PID=$(pgrep -n server)
BASELINE=$(ls /proc/$SERVER_PID/fd | wc -l)
ab -n 10000 -c 1 http://localhost:8080/ > /dev/null 2>&1
AFTER=$(ls /proc/$SERVER_PID/fd | wc -l)
echo "Baseline: $BASELINE  After: $AFTER"
# Expected: AFTER == BASELINE (both should be 4)
```
**T4.3 — accept() error recovery: EINTR**
```bash
# Send SIGUSR1 repeatedly while server is accepting (SIGUSR1 does nothing but interrupts)
SERVER_PID=$(pgrep -n server)
for i in $(seq 20); do kill -USR1 $SERVER_PID; sleep 0.05; done &
curl http://localhost:8080/  # Must return 200 OK, server not crashed
```
**T4.4 — Rapid reconnect (SO_REUSEADDR client-side behavior)**
```bash
ab -n 1000 -c 10 http://localhost:8080/
# Expect 0 failed requests. Requests per second > 500.
```
### 8.5 Complete `test_basic.sh`
```bash
#!/bin/bash
set -e
SERVER_BIN=./server
PORT=18080   # non-standard port to avoid conflicts
# Start server
$SERVER_BIN $PORT &
SERVER_PID=$!
sleep 0.3
trap "kill $SERVER_PID 2>/dev/null; exit" EXIT INT TERM
pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; kill $SERVER_PID; exit 1; }
# T1: Basic response
RESPONSE=$(curl -s http://localhost:$PORT/)
echo "$RESPONSE" | grep -q "Hello from C server" && pass "T1: 200 OK body" || fail "T1"
# T2: Content-Length
BODY_LEN=$(curl -s http://localhost:$PORT/ | wc -c | tr -d ' ')
[ "$BODY_LEN" = "27" ] && pass "T2: Content-Length=27" || fail "T2: got $BODY_LEN"
# T3: Headers
HEADERS=$(curl -sI http://localhost:$PORT/)
echo "$HEADERS" | grep -q "Content-Type: text/html" && pass "T3: Content-Type" || fail "T3"
echo "$HEADERS" | grep -q "Connection: close"       && pass "T4: Connection:close" || fail "T4"
# T5: FD leak check
BASELINE=$(ls /proc/$SERVER_PID/fd | wc -l)
for i in $(seq 200); do curl -s http://localhost:$PORT/ > /dev/null; done
AFTER=$(ls /proc/$SERVER_PID/fd | wc -l)
[ "$AFTER" = "$BASELINE" ] && pass "T5: No FD leak" || fail "T5: FD leak $BASELINE -> $AFTER"
# T6: SIGPIPE survival
for i in $(seq 50); do
  python3 -c "
import socket; s=socket.socket(); s.connect(('localhost',$PORT))
s.send(b'GET / HTTP/1.1\r\nHost: localhost\r\n\r\n'); s.close()
" 2>/dev/null
done
ALIVE=$(curl -s http://localhost:$PORT/ | grep -c "Hello")
[ "$ALIVE" = "1" ] && pass "T6: SIGPIPE survived" || fail "T6"
echo "All tests passed."
```
---
## 9. Performance Targets

![SIGPIPE Flow: What Happens Without vs. With MSG_NOSIGNAL](./diagrams/tdd-diag-6.svg)

| Operation | Target | How to Measure |
|---|---|---|
| Accept + recv + send + close round-trip (loopback) | < 1 ms | `ab -n 1000 -c 1 http://localhost:8080/` → "Time per request" < 1 ms |
| Server throughput (sequential, loopback) | > 1,000 req/s | `ab -n 10000 -c 1` → "Requests per second" > 1000 |
| FD count after 10,000 sequential connections | Equal to baseline (4 FDs) | `ls /proc/$PID/fd \| wc -l` before and after |
| Server survival after 1,000 mid-response disconnects | 0 crashes | SIGPIPE test in T3.3; server responds to the 1001st request with 200 |
| recv() loop correctness at any partial-read boundary | 0 failures | `ab -n 5000 -c 20` with simulated jitter (wrk + Lua delay script) |
| Memory per connection | ≤ 8220 bytes stack (no heap alloc) | `valgrind --tool=massif ./server`; confirm 0 malloc calls per request |
**How `ab` reads**:
```
ab -n 10000 -c 1 http://localhost:8080/ 2>&1 | grep -E "Requests per second|Time per request|Failed"
```
- "Failed requests: 0" is mandatory.
- "Requests per second" must be > 1000 on any hardware made after 2015.
- "Time per request" (mean) must be < 1 ms for the single-connection case.

![Three-Level View of a recv() Call](./diagrams/tdd-diag-7.svg)

---
## 10. Hardware Soul

![Module Architecture: M1 Functions and Data Structures](./diagrams/tdd-diag-8.svg)

**`socket()` and the FD table**: the kernel allocates an entry in the process's file descriptor table — an array of `struct file *` pointers. Each `struct file` points to a `struct socket`, which contains the send and receive ring buffers. The FD integer returned by `socket()` is an index into this table. The table itself is in kernel memory, but the index (the FD) is the only handle userspace sees. At `DEFAULT_PORT` traffic levels, the FD table fits entirely in L1 cache (the table for a process with 4 FDs is ~32 bytes of pointers).
**`accept()` cache temperature**: the `server_fd` entry in the FD table is accessed on every accept loop iteration — it stays hot in L1. The new `client_fd` created by `accept()` is cold: the kernel allocates a new `struct file` and `struct socket` from the slab allocator. First access to the client socket's receive buffer will be a cache miss, but the buffer itself is in kernel memory and will be populated by the NIC interrupt handler before `recv()` returns.
**`recv()` kernel-to-user copy**: each `recv()` call crosses the privilege boundary (syscall), copies bytes from the kernel socket receive buffer to `buf` (user stack), and returns. The copy is bounded by the receive buffer occupancy. On loopback, the sender's `send()` fills the receiver's kernel buffer nearly instantaneously; `recv()` almost always returns a large chunk (the full request) rather than a partial read. On real networks, partial reads are common due to packet fragmentation and Nagle's algorithm batching. The loop handles both transparently.
**`send_all()` and the socket send buffer**: the kernel's TCP send buffer defaults to ~87 KB on Linux. Our 126-byte response fits in a single `send()` call. For this milestone, partial writes will never occur in practice. The loop is there because it must be correct for any response size, including Milestone 3's multi-megabyte file responses.
**Sequential memory access in `strstr(buf, "\r\n\r\n")`**: the search scans `buf` linearly from byte 0 to `total`. A typical HTTP request is 200–800 bytes. At 64 bytes per cache line, 4–13 cache lines are scanned. After the first `recv()`, those cache lines are hot in L1. The `strstr()` scan is entirely L1-resident — sub-100 ns cost.
**Branch prediction in the recv() loop**: the loop predicts "not found yet" on every iteration except the last. For a typical request that arrives in one `recv()` call, the loop executes once, `strstr()` finds the delimiter immediately, and the branch is taken. The branch predictor sees an always-taken-then-not-taken pattern; one misprediction penalty (~15 cycles) per request is negligible.
---
## Makefile
```makefile
CC      = gcc
CFLAGS  = -Wall -Wextra -Wpedantic -std=c11 -O2 -g
TARGET  = server
all: $(TARGET)
$(TARGET): server.c
	$(CC) $(CFLAGS) -o $@ $^
clean:
	rm -f $(TARGET)
test: $(TARGET)
	bash test_basic.sh
.PHONY: all clean test
```
**`-std=c11`** is required for `_Static_assert`. `-g` retains debug info for `gdb`/`valgrind`. `-O2` reflects realistic conditions; do not benchmark with `-O0`.
---
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: http-server-basic-m2 -->
# Technical Design Document: HTTP Request Parsing
## Module `http-server-basic-m2`
---
## 1. Module Charter
This module transforms the raw byte buffer produced by `read_request()` (M1) into a structured `http_request_t` that downstream code can inspect without touching wire bytes again. It handles the complete HTTP/1.1 header grammar as defined in RFC 7230: splitting the request line into method, path, and version; parsing header fields with case-insensitive name normalization, optional-whitespace stripping, and obsolete-fold unfolding; tolerating both CRLF and bare-LF line endings; and enforcing size limits that convert denial-of-service inputs into 4xx responses.
This module does **not** serve files from disk, does **not** manage socket file descriptors, does **not** perform URL percent-decoding beyond a stub interface (that work belongs to M3's path-security pipeline), and does **not** spawn or coordinate threads. All parsing state lives entirely in caller-supplied stack buffers; the functions are pure transformations with no global mutable state and no heap allocation.
**Upstream dependency**: M1's `read_request()` delivers a null-terminated `char buf[]` containing at least one `\r\n\r\n` delimiter. The buffer is valid for the duration of the call to `http_parse_request()`.
**Downstream dependency**: M3's `serve_file()` and M4's keep-alive loop both call `http_parse_request()` and inspect the resulting `http_request_t`. The `body` pointer field points into the caller's buffer — the buffer must remain alive as long as the `http_request_t` is in use.
**Invariants that must always hold**:
- Every string field in `http_request_t` (method, path, version, header names, header values) is null-terminated and fits within its declared array.
- All header names stored in `headers[]` are lowercase ASCII; lookups always use lowercase keys.
- `header_count` accurately reflects the number of valid, fully-populated entries in `headers[]`.
- A negative return from `http_parse_request()` leaves `http_request_t` in an unspecified state; the caller must not inspect its fields.
- No function in this module calls `malloc`, `realloc`, or `free`.
---
## 2. File Structure
Create files in this order:
```
http-server/
├── 1  http_parse.h          # struct definitions, constants, function declarations
├── 2  http_parse.c          # all parsing implementation
├── 3  server.c              # updated from M1: includes http_parse.h, calls http_parse_request()
├── 4  Makefile              # updated build rules
└── 5  test_parse.sh         # automated test script (Phase 7)
```
`http_parse.h` is a **new file**. `server.c` from M1 is modified in-place: `handle_client()` gains the parse call and HEAD-vs-GET branching; the socket lifecycle is unchanged.
---
## 3. Complete Data Model
### 3.1 Size Constants
```c
/* http_parse.h */
#define MAX_HEADERS      32     /* max header entries stored; excess silently dropped */
#define MAX_PATH_LEN     8192   /* 8 KB URI limit → 414 if exceeded                  */
#define MAX_METHOD_LEN   16     /* longest standard method is "CONNECT" (7 chars)    */
#define MAX_VERSION_LEN  16     /* "HTTP/1.1" is 8 chars; 16 gives headroom          */
#define MAX_HEADER_NAME  128    /* longest standard header ~40 chars; 128 is generous */
#define MAX_HEADER_VALUE 1024   /* values beyond 1023 are truncated, not rejected     */
```
Rationale for `MAX_HEADERS = 32`: a typical browser request carries 8–15 headers; 32 covers every realistic case while keeping `http_request_t` below 40 KB on the stack.
Rationale for `MAX_PATH_LEN = 8192`: nginx's `client_header_buffer_size` defaults to 8 KB; this project matches that de-facto limit and maps excess to 414 per RFC 7231 §6.5.12.
### 3.2 `http_header_t`
```c
/* http_parse.h */
typedef struct {
    char name[MAX_HEADER_NAME];   /* lowercase ASCII, null-terminated; e.g. "content-type" */
    char value[MAX_HEADER_VALUE]; /* OWS-stripped, obs-fold-unfolded, null-terminated       */
} http_header_t;
```
**Memory layout** (cache-line view, 64 bytes per line):
```
Offset    Field                    Size     Cache lines
0x000     name[0..127]             128 B    lines 0–1   (2 lines)
0x080     value[0..1023]           1024 B   lines 2–17  (16 lines)
Total per http_header_t:           1152 B   18 lines
```
With `MAX_HEADERS = 32`: `32 × 1152 = 36,864` bytes for the headers array alone. This is the dominant cost of `http_request_t`. Keep it in mind when profiling stack usage in M4's thread pool (each worker thread holds one `http_request_t` in its call stack).
### 3.3 `http_request_t`
```c
/* http_parse.h */
typedef struct {
    char          method[MAX_METHOD_LEN];    /* "GET", "HEAD", "POST", …  uppercase */
    char          path[MAX_PATH_LEN];        /* raw URL path, NOT percent-decoded    */
    char          version[MAX_VERSION_LEN];  /* "HTTP/1.1" or "HTTP/1.0"            */
    http_header_t headers[MAX_HEADERS];      /* parsed header entries               */
    int           header_count;              /* 0..MAX_HEADERS                      */
    const char   *body;                      /* pointer into caller's buffer, or NULL */
    size_t        body_len;                  /* bytes from body ptr to end of buffer */
} http_request_t;
```
**Full memory layout**:
```
Offset        Field               Type             Size
0x0000        method[0..15]       char[16]           16 B
0x0010        path[0..8191]       char[8192]       8192 B
0x2010        version[0..15]      char[16]           16 B
0x2020        headers[0..31]      http_header_t[]  36864 B   (32 × 1152)
0xA820        header_count        int                 4 B
0xA824        (padding)                               4 B    (align next ptr)
0xA828        body                const char *        8 B
0xA830        body_len            size_t              8 B
0xA838        end
Total:                                            43064 B  (~42 KB)
```
Stack-allocation warning: `handle_client()` also holds `char buf[8192]` (M1). Combined stack frame per connection ≈ 51 KB. Default Linux thread stack is 8 MB — safe for the thread-per-connection model in M4 with any reasonable pool size.
**`body` semantics**: points directly into the caller's `buf[]` at the first byte after the `\r\n\r\n` delimiter. This is a zero-copy design. The caller must not free or modify `buf[]` while the `http_request_t` is in use. If no bytes follow the delimiter, `body = NULL` and `body_len = 0`.

![HTTP Request Parser State Machine](./diagrams/tdd-diag-9.svg)

### 3.4 `http_method_t`
```c
/* http_parse.h */
typedef enum {
    HTTP_METHOD_GET   = 0,
    HTTP_METHOD_HEAD  = 1,
    HTTP_METHOD_OTHER = 2    /* any method not GET or HEAD → 501 */
} http_method_t;
```
### 3.5 Error Codes Returned by `http_parse_request()`
```c
/* http_parse.h — return values of http_parse_request() */
#define HTTP_PARSE_OK          0    /* success */
#define HTTP_PARSE_BAD_REQUEST -1   /* → send 400 */
#define HTTP_PARSE_URI_TOO_LONG -2  /* → send 414 */
#define HTTP_PARSE_NOT_IMPL    -3   /* → send 501 */
```
### 3.6 Hardcoded Error Response Strings
These live in `http_parse.c` (or a new `responses.h` once M3 needs them too). Every `Content-Length` value must be verified with `_Static_assert` matching `strlen()` of the body literal.
```c
/* http_parse.c */
#define BODY_400 "<html><body><h1>400 Bad Request</h1></body></html>"
#define BODY_414 "<html><body><h1>414 URI Too Long</h1></body></html>"
#define BODY_501 "<html><body><h1>501 Not Implemented</h1></body></html>"
/* Byte counts — verify at compile time */
_Static_assert(sizeof(BODY_400) - 1 == 50, "BODY_400 length mismatch");
_Static_assert(sizeof(BODY_414) - 1 == 50, "BODY_414 length mismatch");
_Static_assert(sizeof(BODY_501) - 1 == 52, "BODY_501 length mismatch");
const char HTTP_400[] =
    "HTTP/1.1 400 Bad Request\r\n"
    "Content-Type: text/html; charset=utf-8\r\n"
    "Content-Length: 50\r\n"
    "Connection: close\r\n"
    "\r\n"
    BODY_400;
const char HTTP_414[] =
    "HTTP/1.1 414 URI Too Long\r\n"
    "Content-Type: text/html; charset=utf-8\r\n"
    "Content-Length: 50\r\n"
    "Connection: close\r\n"
    "\r\n"
    BODY_414;
const char HTTP_501[] =
    "HTTP/1.1 501 Not Implemented\r\n"
    "Content-Type: text/html; charset=utf-8\r\n"
    "Content-Length: 52\r\n"
    "Connection: close\r\n"
    "\r\n"
    BODY_501;
```
> **Action required**: before finalising, run `printf '%s' '<html><body><h1>400 Bad Request</h1></body></html>' | wc -c` for each body string and replace the `Content-Length` values and `_Static_assert` constants with the actual counts. The values above are illustrative. If the `_Static_assert` fires at compile time, the `Content-Length` in the literal is wrong — fix the literal, not the assert.

![http_request_t and http_header_t Memory Layout with Byte Offsets](./diagrams/tdd-diag-10.svg)

---
## 4. Interface Contracts
### 4.1 `http_parse_request()`
```c
/* http_parse.h */
int http_parse_request(char *buf, size_t buf_len, http_request_t *req);
```
**The top-level entry point.** All other functions in this module are called exclusively by this function or by tests.
**Parameters**:
- `buf`: the raw request buffer from `read_request()`. Must be null-terminated. Must be writable (the function adds null terminators in-place at line boundaries during parsing). Must remain valid for the lifetime of `req`.
- `buf_len`: number of valid bytes in `buf`, equal to the return value of `read_request()`. Must be ≥ 4 (minimum valid HTTP request contains at least `\r\n\r\n`). Must not include the null terminator itself.
- `req`: pointer to an `http_request_t` on the caller's stack. Must not be NULL. The function calls `memset(req, 0, sizeof(*req))` before writing any fields, so the caller need not initialize it.
**Return values**:
- `HTTP_PARSE_OK (0)`: `req` is fully populated. `req->method`, `req->path`, `req->version` are all non-empty null-terminated strings. `req->header_count` reflects the number of valid entries in `req->headers[]`.
- `HTTP_PARSE_BAD_REQUEST (-1)`: request line is malformed (no space, empty method, empty path, unrecognized version, or missing `\n` in the buffer). `req` contents are unspecified.
- `HTTP_PARSE_URI_TOO_LONG (-2)`: the path field in the request line is ≥ `MAX_PATH_LEN` bytes. `req` contents are unspecified.
- `HTTP_PARSE_NOT_IMPL (-3)`: request line parses cleanly but `req->method` is neither `"GET"` nor `"HEAD"`. `req` contents are partially populated (method, path, version are set; headers may be parsed).
**Side effects**: writes null bytes at line-end positions inside `buf` during `parse_headers()`. The original CRLF/LF endings are overwritten. Callers that need the original buffer content must copy it before calling `http_parse_request()`.
**Thread safety**: fully re-entrant. No global state is read or written. Multiple threads may call this simultaneously with independent `buf`/`req` pairs.
---
### 4.2 `parse_request_line()`
```c
/* http_parse.h — internal; exposed for unit testing */
int parse_request_line(const char *line, size_t line_len, http_request_t *req);
```
**Parameters**:
- `line`: pointer to the first byte of the request line in `buf`. This is `buf` itself.
- `line_len`: number of bytes in the request line, **not** including the trailing `\n` or `\r\n`. The `\r` (if present) must already be stripped by the caller before passing `line_len`.
- `req`: destination struct. This function writes `req->method`, `req->path`, `req->version`.
**Return values**:
- `0`: success.
- `-1`: malformed (no first space, no second space, empty method, empty path, or `version_len == 0` or `>= MAX_VERSION_LEN`).
- `-2`: path length `>= MAX_PATH_LEN`.
**Preconditions**:
- `line_len > 0`.
- `line` points to at least `line_len` valid readable bytes.
- `req` is not NULL.
**Postconditions on success (return 0)**:
- `req->method` is a null-terminated uppercase string, `strlen(req->method) < MAX_METHOD_LEN`.
- `req->path` is a null-terminated string, `strlen(req->path) < MAX_PATH_LEN`.
- `req->version` is a null-terminated string, `strlen(req->version) < MAX_VERSION_LEN`.
- None of these fields contains a space or a newline character.
**Does not validate** the HTTP version string (that check is in `http_parse_request()`). Does not lower-case the method (HTTP methods are case-sensitive per RFC 7230 §3.1.1 — they are uppercase by spec; do not tolower them).
---
### 4.3 `parse_headers()`
```c
/* http_parse.h — internal; exposed for unit testing */
int parse_headers(char *buf, size_t buf_len, http_request_t *req);
```
**Parameters**:
- `buf`: the full raw request buffer. The function skips the request line by finding the first `\n` and beginning header processing at `pos = first_newline + 1`.
- `buf_len`: total valid bytes in `buf`.
- `req`: destination. This function writes `req->headers[]`, `req->header_count`, `req->body`, `req->body_len`.
**Return value**: the number of headers successfully parsed (0..MAX_HEADERS). Never negative — parsing errors at the individual header level result in that header being skipped, not a function failure. Returns -1 only if the initial request-line skip fails (no `\n` found in `buf`).
**Side effects**: writes `\0` bytes into `buf` at the positions of `\n` characters to enable null-terminated line content. This is the reason `buf` must be writable.
**Postconditions on non-negative return**:
- `req->header_count` equals the return value.
- For every `i` in `0..req->header_count-1`: `req->headers[i].name` is lowercase, non-empty, null-terminated. `req->headers[i].value` is OWS-stripped, obs-fold-unfolded, null-terminated.
- If an empty line (end-of-headers marker) was found: `req->body` points to the byte after the empty line's `\n`, or is NULL if that byte is at or beyond `buf + buf_len`. `req->body_len` is the number of bytes from `req->body` to `buf + buf_len`.
- If no empty line was found (malformed or truncated headers): `req->body = NULL`, `req->body_len = 0`.
---
### 4.4 `get_header()`
```c
/* http_parse.h */
const char *get_header(const http_request_t *req, const char *name);
```
**Parameters**:
- `req`: a successfully parsed request (return value of `http_parse_request()` was 0 or -3).
- `name`: the header name to look up. **Must be lowercase**. The function uses `strcmp()`, not `strcasecmp()`, because names are stored lowercase at parse time.
**Return value**:
- Pointer to the null-terminated value string inside `req->headers[i].value` for the first matching entry.
- `NULL` if no header with that name is present.
**Complexity**: O(n) where n = `req->header_count`. For n ≤ 32 with short name strings, this is faster in practice than a hash map due to cache locality.
**Does not copy**: the returned pointer is valid as long as `req` is valid.
---
### 4.5 `classify_method()`
```c
/* http_parse.h */
http_method_t classify_method(const char *method);
```
**Parameters**: null-terminated method string from `req->method`.
**Return value**:
- `HTTP_METHOD_GET` if `strcmp(method, "GET") == 0`.
- `HTTP_METHOD_HEAD` if `strcmp(method, "HEAD") == 0`.
- `HTTP_METHOD_OTHER` for any other string (including empty string).
**Thread safety**: pure function, no side effects.
---
### 4.6 `url_decode_stub()` (stub only in M2)
```c
/* http_parse.h */
/* Stub: copies src to dst unchanged. M3 replaces with full implementation. */
int url_decode_stub(const char *src, char *dst, size_t dst_size);
```
M2 does not need to decode `%XX` sequences — that is M3's concern. The stub exists so that `serve_file()` in M3 can call `url_decode()` through the same header without changing `http_parse.h`'s interface. Do not implement percent-decoding in this module.
---
## 5. Algorithm Specification
### 5.1 `parse_request_line()` — Detailed Steps

![HTTP/1.1 Request Wire Format: Annotated Byte Map](./diagrams/tdd-diag-11.svg)

```
INPUT:  line       = pointer to first byte of request line
        line_len   = byte count, EXCLUDING trailing \r (if CRLF) and \n
        req        = destination struct
STEP 1: Find first space
  first_space = memchr(line, ' ', line_len)
  if first_space == NULL:
    return -1    ← "GET\r\n" with no spaces: malformed
STEP 2: Extract method
  method_len = first_space - line
  if method_len == 0:             return -1   ← leading space
  if method_len >= MAX_METHOD_LEN: return -1  ← absurdly long token
  memcpy(req->method, line, method_len)
  req->method[method_len] = '\0'
STEP 3: Find second space
  path_start  = first_space + 1
  remaining   = line_len - (path_start - line)
  second_space = memchr(path_start, ' ', remaining)
  if second_space == NULL:
    return -1   ← no version component
STEP 4: Extract path (length guard first)
  path_len = second_space - path_start
  if path_len == 0:              return -1   ← empty path (even "/" has length 1)
  if path_len >= MAX_PATH_LEN:   return -2   ← caller sends 414
  memcpy(req->path, path_start, path_len)
  req->path[path_len] = '\0'
STEP 5: Extract version
  version_start = second_space + 1
  version_len   = line_len - (version_start - line)
  if version_len == 0:              return -1
  if version_len >= MAX_VERSION_LEN: return -1
  memcpy(req->version, version_start, version_len)
  req->version[version_len] = '\0'
STEP 6: return 0
```
**Why `memchr()` not `strchr()`**: `strchr()` scans until `\0`. The `line` pointer points into `buf` which may contain null bytes only where we placed them ourselves (after prior `recv()` calls in the request body area). A `strchr()` scan could stop at one of those nulls before finding the space. `memchr(ptr, ch, len)` scans exactly `len` bytes, never reading past the end of the valid request-line content.
**Why no `strtok()`**: `strtok()` modifies the source string and is not re-entrant. It also collapses multiple consecutive delimiters, masking "double space" anomalies that should produce a 400. `memchr()` + explicit pointer arithmetic is the correct tool.
---
### 5.2 `parse_headers()` — Detailed Steps

![parse_request_line() Algorithm: memchr Space Finding](./diagrams/tdd-diag-12.svg)

The function processes `buf` as a sequence of lines, starting after the request line. Each iteration of the main loop processes exactly one `\n`-terminated line.
```
INPUT:  buf      = full raw request buffer (writable)
        buf_len  = total valid bytes
        req      = destination struct (header_count must be initialized to 0 before entry)
SETUP:
  pos = buf
  end = buf + buf_len
STEP A: Skip the request line
  request_line_end = memchr(pos, '\n', end - pos)
  if request_line_end == NULL: return -1
  pos = request_line_end + 1
MAIN LOOP: while pos < end AND req->header_count < MAX_HEADERS:
  STEP B: Find the end of this line
    line_end = memchr(pos, '\n', end - pos)
    if line_end == NULL: break    ← no more complete lines
  STEP C: Compute logical line length (strip \r if CRLF)
    line     = pos
    line_len = line_end - pos
    if line_len > 0 AND line[line_len - 1] == '\r':
      line_len--                  ← CRLF → strip the \r
  STEP D: Detect empty line (end-of-headers)
    if line_len == 0:
      body_start = line_end + 1
      if body_start < end:
        req->body     = body_start
        req->body_len = (size_t)(end - body_start)
      else:
        req->body     = NULL
        req->body_len = 0
      break                       ← done with headers
  STEP E: Detect obs-fold continuation
    if (line[0] == ' ' OR line[0] == '\t') AND req->header_count > 0:
      prev     = &req->headers[req->header_count - 1]
      prev_len = strlen(prev->value)
      cont     = line
      while cont < line + line_len AND (*cont == ' ' OR *cont == '\t'):
        cont++                    ← skip leading whitespace on continuation
      cont_len = (line + line_len) - cont
      space_needed = prev_len + 1 + cont_len   ← "+1" for joining space
      if space_needed < MAX_HEADER_VALUE - 1:
        prev->value[prev_len]             = ' '
        memcpy(prev->value + prev_len + 1, cont, cont_len)
        prev->value[prev_len + 1 + cont_len] = '\0'
      ← if space_needed overflows, the continuation is silently dropped
      pos = line_end + 1
      continue
    if (line[0] == ' ' OR line[0] == '\t') AND req->header_count == 0:
      ← obs-fold with no preceding header: skip line silently
      pos = line_end + 1
      continue
  STEP F: Find the colon separator
    colon = memchr(line, ':', line_len)
    if colon == NULL:
      pos = line_end + 1
      continue                    ← header line with no colon: skip
  STEP G: Validate and copy the name
    name_len = colon - line
    if name_len == 0 OR name_len >= MAX_HEADER_NAME:
      pos = line_end + 1
      continue                    ← empty or oversized name: skip
    hdr = &req->headers[req->header_count]
    memcpy(hdr->name, line, name_len)
    hdr->name[name_len] = '\0'
  STEP H: Normalize name to lowercase
    for i in 0..name_len-1:
      hdr->name[i] = (char)tolower((unsigned char)hdr->name[i])
    ← cast to unsigned char is MANDATORY: tolower() has UB on signed char
       with values > 127 (e.g., UTF-8 multi-byte header names)
  STEP I: Strip OWS from value and copy
    value_start = colon + 1
    value_end   = line + line_len
    ← strip leading OWS:
    while value_start < value_end AND (*value_start == ' ' OR *value_start == '\t'):
      value_start++
    ← strip trailing OWS:
    while value_end > value_start AND (*(value_end-1) == '' OR *(value_end-1) == '\t'):
      value_end--
    value_len = value_end - value_start
    if value_len >= MAX_HEADER_VALUE:
      value_len = MAX_HEADER_VALUE - 1   ← truncate silently
    memcpy(hdr->value, value_start, value_len)
    hdr->value[value_len] = '\0'
  STEP J: Commit the header entry
    req->header_count++
    pos = line_end + 1
END LOOP
return req->header_count
```
**The `(unsigned char)` cast in `tolower()`**: `tolower()` is defined to accept an `int` that is either `EOF` or representable as `unsigned char`. On platforms where `char` is signed, characters with the high bit set (bytes 0x80–0xFF) are negative when widened to `int`. Passing a negative value to `tolower()` is undefined behavior (and crashes on some implementations). The cast `(unsigned char)` promotes the byte to the range 0–255 before `tolower()` sees it. This is not optional.
**The obs-fold branch (Step E)**: RFC 7230 §3.2.6 defines obs-fold as a line beginning with SP or HT that continues the previous header's value. The correct handling is to replace the `\r\n SP` (or `\r\n HT`) with a single space and append the trimmed continuation text. The function unflods unconditionally; it does not reject obs-fold with 400 (that would be the strict-mode option; for this project, unfolding is sufficient).

![Case-Insensitive Header Normalization: Before/After Buffer Transformation](./diagrams/tdd-diag-13.svg)

---
### 5.3 `http_parse_request()` — Top-Level Orchestration
```
INPUT:  buf     = raw writable request buffer
        buf_len = valid byte count
        req     = destination (zeroed by this function)
STEP 1: Zero-initialize req
  memset(req, 0, sizeof(*req))
STEP 2: Find and measure the request line
  first_nl = memchr(buf, '\n', buf_len)
  if first_nl == NULL: return HTTP_PARSE_BAD_REQUEST
  request_line_len = first_nl - buf
  if request_line_len > 0 AND buf[request_line_len - 1] == '\r':
    request_line_len--   ← strip \r for CRLF normalization
STEP 3: Parse the request line
  rc = parse_request_line(buf, request_line_len, req)
  if rc == -2: return HTTP_PARSE_URI_TOO_LONG
  if rc != 0:  return HTTP_PARSE_BAD_REQUEST
STEP 4: Validate HTTP version
  if strcmp(req->version, "HTTP/1.1") != 0 AND
     strcmp(req->version, "HTTP/1.0") != 0:
    return HTTP_PARSE_BAD_REQUEST
    ← covers "HTTP/2", "HTTP/3", and garbage version strings
STEP 5: Parse headers (always, even for unsupported methods)
  parse_headers(buf, buf_len, req)
  ← parse_headers() returning -1 is non-fatal here; it means the request
     line is the only content. header_count remains 0.
STEP 6: Validate Host header (HTTP/1.1 requirement)
  if strcmp(req->version, "HTTP/1.1") == 0:
    host = get_header(req, "host")
    if host == NULL:
      fprintf(stderr, "HTTP/1.1 request missing required Host header\n")
      ← log-and-continue (not strict rejection) for this project
STEP 7: Classify method
  if classify_method(req->method) == HTTP_METHOD_OTHER:
    return HTTP_PARSE_NOT_IMPL   ← -3; req fields ARE populated
STEP 8: return HTTP_PARSE_OK
```

![obs-fold Unfolding: Before/After Buffer State](./diagrams/tdd-diag-14.svg)

---
### 5.4 `get_header()` — Linear Scan
```
INPUT:  req  = populated http_request_t
        name = lowercase null-terminated search key
LOOP i = 0..req->header_count-1:
  if strcmp(req->headers[i].name, name) == 0:
    return req->headers[i].value    ← pointer into headers array
return NULL
```
No function call overhead beyond `strcmp()`. With `MAX_HEADERS = 32` and average name length ~15 chars, worst-case cost is 32 × ~15 byte comparisons = ~480 byte reads, all from the same `http_request_t` stack allocation. At 64 bytes per cache line, `headers[0..31].name` spans `32 × 2 cache lines = 64 cache lines = 4096 bytes` of cache, all of which will be in L1 after the first miss. Subsequent `get_header()` calls on the same `req` are entirely L1-resident.
---
## 6. Error Handling Matrix
| Error Condition | Detected In | Return / Action | Caller Response | User-Visible |
|---|---|---|---|---|
| No `\n` in buf (truncated) | `http_parse_request()` step 2 | `HTTP_PARSE_BAD_REQUEST` | send `HTTP_400`, close FD | 400 Bad Request |
| No first space in request line | `parse_request_line()` step 1 | `-1` → `HTTP_PARSE_BAD_REQUEST` | send `HTTP_400` | 400 Bad Request |
| Empty method (leading space) | `parse_request_line()` step 2 | `-1` → `HTTP_PARSE_BAD_REQUEST` | send `HTTP_400` | 400 Bad Request |
| No second space (no version) | `parse_request_line()` step 3 | `-1` → `HTTP_PARSE_BAD_REQUEST` | send `HTTP_400` | 400 Bad Request |
| Empty path | `parse_request_line()` step 4 | `-1` → `HTTP_PARSE_BAD_REQUEST` | send `HTTP_400` | 400 Bad Request |
| Path ≥ 8192 bytes | `parse_request_line()` step 4 | `-2` → `HTTP_PARSE_URI_TOO_LONG` | send `HTTP_414` | 414 URI Too Long |
| Version string ≥ 16 bytes | `parse_request_line()` step 5 | `-1` → `HTTP_PARSE_BAD_REQUEST` | send `HTTP_400` | 400 Bad Request |
| Unrecognized version string | `http_parse_request()` step 4 | `HTTP_PARSE_BAD_REQUEST` | send `HTTP_400` | 400 Bad Request |
| Method not GET or HEAD | `http_parse_request()` step 7 | `HTTP_PARSE_NOT_IMPL` | send `HTTP_501` | 501 Not Implemented |
| Header line has no colon | `parse_headers()` step F | skip entry | none | none (header absent) |
| Header name empty or ≥ 128 B | `parse_headers()` step G | skip entry | none | none (header absent) |
| Header value ≥ 1024 B | `parse_headers()` step I | truncate to 1023 bytes | none | truncated value used |
| obs-fold with no prior header | `parse_headers()` step E | skip line | none | none |
| > 32 headers | `parse_headers()` loop condition | stop loop, ignore remainder | none | headers beyond 32 absent |
| Missing Host (HTTP/1.1) | `http_parse_request()` step 6 | log warning, continue | none | none (log only) |
| `parse_headers()` returns -1 | `http_parse_request()` step 5 | continue with `header_count = 0` | treats all headers absent | depends on request |
| `tolower()` on non-ASCII byte without cast | compile-time discipline | UB — prevented by cast | N/A | undefined |
**Invariant**: no error path in any function in this module closes a file descriptor, calls `exit()`, or writes to any global mutable state.
---
## 7. Implementation Sequence with Checkpoints
### Phase 1 — Struct Definitions, Constants, Header File (0.5 h)
Create `http_parse.h` with all `#define` constants, `http_header_t`, `http_request_t`, `http_method_t`, the `HTTP_PARSE_*` return code macros, and function declarations for all public functions. Add include guards.
```c
/* http_parse.h */
#ifndef HTTP_PARSE_H
#define HTTP_PARSE_H
#include <stddef.h>   /* size_t */
#define MAX_HEADERS       32
#define MAX_PATH_LEN      8192
#define MAX_METHOD_LEN    16
#define MAX_VERSION_LEN   16
#define MAX_HEADER_NAME   128
#define MAX_HEADER_VALUE  1024
#define HTTP_PARSE_OK           0
#define HTTP_PARSE_BAD_REQUEST  -1
#define HTTP_PARSE_URI_TOO_LONG -2
#define HTTP_PARSE_NOT_IMPL     -3
typedef struct {
    char name[MAX_HEADER_NAME];
    char value[MAX_HEADER_VALUE];
} http_header_t;
typedef struct {
    char          method[MAX_METHOD_LEN];
    char          path[MAX_PATH_LEN];
    char          version[MAX_VERSION_LEN];
    http_header_t headers[MAX_HEADERS];
    int           header_count;
    const char   *body;
    size_t        body_len;
} http_request_t;
typedef enum {
    HTTP_METHOD_GET   = 0,
    HTTP_METHOD_HEAD  = 1,
    HTTP_METHOD_OTHER = 2
} http_method_t;
int            http_parse_request(char *buf, size_t buf_len, http_request_t *req);
int            parse_request_line(const char *line, size_t line_len, http_request_t *req);
int            parse_headers(char *buf, size_t buf_len, http_request_t *req);
const char    *get_header(const http_request_t *req, const char *name);
http_method_t  classify_method(const char *method);
int            url_decode_stub(const char *src, char *dst, size_t dst_size);
extern const char HTTP_400[];
extern const char HTTP_414[];
extern const char HTTP_501[];
#endif /* HTTP_PARSE_H */
```
Create `http_parse.c` with the includes and a stub body for every function that `return 0`s. Compile: `gcc -Wall -Wextra -std=c11 -c http_parse.c` — zero warnings required.
**Checkpoint 1**: `sizeof(http_request_t)` printed in a small test program. Expected value: ≈ 43064 bytes (may vary slightly due to platform alignment). If wildly different, re-examine field sizes.
```c
/* checkpoint1.c */
#include <stdio.h>
#include "http_parse.h"
int main(void) {
    printf("sizeof(http_header_t)  = %zu\n", sizeof(http_header_t));   /* expect 1152 */
    printf("sizeof(http_request_t) = %zu\n", sizeof(http_request_t));  /* expect ~43064 */
    return 0;
}
```
---
### Phase 2 — `parse_request_line()`: Space Finding, Length Guards, Return Codes (0.5–1 h)
Implement the full algorithm from Section 5.1 in `http_parse.c`. Use `memchr()` exclusively for finding spaces — no `strchr()`, no `strtok()`.
```c
#include <string.h>
#include "http_parse.h"
int parse_request_line(const char *line, size_t line_len, http_request_t *req) {
    /* Step 1 */
    const char *first_space = memchr(line, ' ', line_len);
    if (!first_space) return -1;
    /* Step 2 */
    size_t method_len = (size_t)(first_space - line);
    if (method_len == 0 || method_len >= MAX_METHOD_LEN) return -1;
    memcpy(req->method, line, method_len);
    req->method[method_len] = '\0';
    /* Step 3 */
    const char *path_start = first_space + 1;
    size_t remaining = line_len - (size_t)(path_start - line);
    const char *second_space = memchr(path_start, ' ', remaining);
    if (!second_space) return -1;
    /* Step 4 */
    size_t path_len = (size_t)(second_space - path_start);
    if (path_len == 0) return -1;
    if (path_len >= MAX_PATH_LEN) return -2;
    memcpy(req->path, path_start, path_len);
    req->path[path_len] = '\0';
    /* Step 5 */
    const char *version_start = second_space + 1;
    size_t version_len = line_len - (size_t)(version_start - line);
    if (version_len == 0 || version_len >= MAX_VERSION_LEN) return -1;
    memcpy(req->version, version_start, version_len);
    req->version[version_len] = '\0';
    return 0;
}
```
**Checkpoint 2**: write a small inline test in `main()` of a throwaway `test_phase2.c`:
```c
#include <stdio.h>
#include <string.h>
#include "http_parse.h"
int main(void) {
    http_request_t req;
    memset(&req, 0, sizeof(req));
    /* happy path */
    const char *line = "GET /index.html HTTP/1.1";
    int rc = parse_request_line(line, strlen(line), &req);
    printf("rc=%d method=[%s] path=[%s] version=[%s]\n",
           rc, req.method, req.path, req.version);
    /* expect: rc=0 method=[GET] path=[/index.html] version=[HTTP/1.1] */
    /* 414 trigger */
    char big[9000];
    memset(big, 'a', sizeof(big));
    char line2[9050];
    int n = snprintf(line2, sizeof(line2), "GET /");
    memcpy(line2 + n, big, 8200);
    int n2 = n + 8200;
    memcpy(line2 + n2, " HTTP/1.1", 9);
    rc = parse_request_line(line2, n2 + 9, &req);
    printf("414 test rc=%d (expect -2)\n", rc);
    /* no second space */
    const char *bad = "GET /path";
    rc = parse_request_line(bad, strlen(bad), &req);
    printf("no-version rc=%d (expect -1)\n", rc);
    return 0;
}
```
All three `expect` comments must match.
---
### Phase 3 — `parse_headers()`: Full RFC 7230 Compliance (1–1.5 h)
Implement the full algorithm from Section 5.2. This is the most complex function. Write it in stages:
**Stage 3a**: request-line skip + empty-line detection + body pointer setting. No header parsing yet — just a loop that scans lines and stops at the empty line.
**Stage 3b**: add colon split, name copy, and `tolower()` loop. Test that `Content-Type` stored as `content-type`.
**Stage 3c**: add OWS stripping of value. Test that `Content-Length:   42  ` stores as `42`.
**Stage 3d**: add obs-fold detection and unfolding. Test that a two-line folded value is concatenated with a space.
**Stage 3e**: add bare-LF (no `\r`) tolerance. Test a request with `\n`-only line endings.
```c
#include <ctype.h>
/* full implementation per Section 5.2 algorithm */
int parse_headers(char *buf, size_t buf_len, http_request_t *req) {
    req->header_count = 0;
    req->body         = NULL;
    req->body_len     = 0;
    char *pos = buf;
    char *end = buf + buf_len;
    /* Step A: skip request line */
    char *request_line_end = memchr(pos, '\n', (size_t)(end - pos));
    if (!request_line_end) return -1;
    pos = request_line_end + 1;
    while (pos < end && req->header_count < MAX_HEADERS) {
        /* Step B */
        char *line_end = memchr(pos, '\n', (size_t)(end - pos));
        if (!line_end) break;
        /* Step C */
        char   *line     = pos;
        size_t  line_len = (size_t)(line_end - pos);
        if (line_len > 0 && line[line_len - 1] == '\r') line_len--;
        /* Step D: empty line = end of headers */
        if (line_len == 0) {
            const char *body_start = line_end + 1;
            if (body_start < end) {
                req->body     = body_start;
                req->body_len = (size_t)(end - body_start);
            }
            break;
        }
        /* Step E: obs-fold */
        if (line[0] == ' ' || line[0] == '\t') {
            if (req->header_count > 0) {
                http_header_t *prev = &req->headers[req->header_count - 1];
                size_t prev_len = strlen(prev->value);
                const char *cont = line;
                while (cont < line + line_len && (*cont == ' ' || *cont == '\t')) cont++;
                size_t cont_len = (size_t)((line + line_len) - cont);
                if (prev_len + 1 + cont_len < MAX_HEADER_VALUE - 1) {
                    prev->value[prev_len] = ' ';
                    memcpy(prev->value + prev_len + 1, cont, cont_len);
                    prev->value[prev_len + 1 + cont_len] = '\0';
                }
            }
            pos = line_end + 1;
            continue;
        }
        /* Step F */
        char *colon = memchr(line, ':', line_len);
        if (!colon) { pos = line_end + 1; continue; }
        /* Step G */
        size_t name_len = (size_t)(colon - line);
        if (name_len == 0 || name_len >= MAX_HEADER_NAME) {
            pos = line_end + 1; continue;
        }
        http_header_t *hdr = &req->headers[req->header_count];
        memcpy(hdr->name, line, name_len);
        hdr->name[name_len] = '\0';
        /* Step H */
        for (size_t i = 0; i < name_len; i++) {
            hdr->name[i] = (char)tolower((unsigned char)hdr->name[i]);
        }
        /* Step I */
        char *vs = colon + 1;
        char *ve = line + line_len;
        while (vs < ve && (*vs == ' ' || *vs == '\t'))        vs++;
        while (ve > vs && (*(ve - 1) == ' ' || *(ve - 1) == '\t')) ve--;
        size_t value_len = (size_t)(ve - vs);
        if (value_len >= MAX_HEADER_VALUE) value_len = MAX_HEADER_VALUE - 1;
        memcpy(hdr->value, vs, value_len);
        hdr->value[value_len] = '\0';
        /* Step J */
        req->header_count++;
        pos = line_end + 1;
    }
    return req->header_count;
}
```
**Checkpoint 3**: compile with `-Wall -Wextra -std=c11`; zero warnings. Then run:
```bash
printf "GET / HTTP/1.1\r\nContent-Type:   text/html  \r\nX-Fold: value1\r\n continues\r\n\r\n" \
  | nc -q 1 localhost 8080
# Server (after Phase 4 wires it up) must log:
# method=GET  path=/  version=HTTP/1.1
# header[0]: content-type = text/html
# header[1]: x-fold = value1 continues
```
---
### Phase 4 — `get_header()`, `classify_method()`, `http_parse_request()` Entry Point (0.5 h)
Implement the three remaining functions per Sections 4.4, 4.5, and 5.3. Then update `handle_client()` in `server.c`:
```c
/* server.c — updated handle_client() */
void handle_client(int client_fd) {
    char buf[REQUEST_BUF_SIZE];
    ssize_t req_len = read_request(client_fd, buf, sizeof(buf));
    if (req_len <= 0) {
        send_all(client_fd, HTTP_400, strlen(HTTP_400));
        return;
    }
    http_request_t req;
    int rc = http_parse_request(buf, (size_t)req_len, &req);
    if (rc == HTTP_PARSE_URI_TOO_LONG) {
        send_all(client_fd, HTTP_414, strlen(HTTP_414));
        return;
    }
    if (rc == HTTP_PARSE_NOT_IMPL) {
        send_all(client_fd, HTTP_501, strlen(HTTP_501));
        return;
    }
    if (rc != HTTP_PARSE_OK) {
        send_all(client_fd, HTTP_400, strlen(HTTP_400));
        return;
    }
    const char *host = get_header(&req, "host");
    printf("%s %s %s  Host: %s\n",
           req.method, req.path, req.version,
           host ? host : "(none)");
    /* Phase 6 adds HEAD divergence; for now, always send a hardcoded 200 */
    send_all(client_fd, HTTP_RESPONSE, strlen(HTTP_RESPONSE));
}
```
**Checkpoint 4**: `curl -v http://localhost:8080/` returns 200 OK and server logs `GET / HTTP/1.1  Host: localhost:8080`. `curl -X DELETE http://localhost:8080/` returns 501.
---
### Phase 5 — Error Response Constants with Correct Content-Length (0.5 h)
Complete the `HTTP_400`, `HTTP_414`, `HTTP_501` strings in `http_parse.c` (see Section 3.6). Run the byte-count verification:
```bash
printf '%s' '<html><body><h1>400 Bad Request</h1></body></html>'  | wc -c
printf '%s' '<html><body><h1>414 URI Too Long</h1></body></html>' | wc -c
printf '%s' '<html><body><h1>501 Not Implemented</h1></body></html>' | wc -c
```
Update `Content-Length` values in the string literals to match these counts. Update `_Static_assert` constants to match. Recompile — assert must pass.
**Checkpoint 5**: `curl -sv -X DELETE http://localhost:8080/` output includes `Content-Length: 52` (or whatever your actual body byte count is) and a body of exactly that many bytes. Verify: `curl -s -X DELETE http://localhost:8080/ | wc -c` output equals the `Content-Length` value.
---
### Phase 6 — HEAD Method: Headers-Only Response (0.5 h)
Update `handle_client()` to branch on method:
```c
/* In handle_client(), after http_parse_request() succeeds */
int send_body = (classify_method(req.method) == HTTP_METHOD_GET);
if (send_body) {
    send_all(client_fd, HTTP_RESPONSE, strlen(HTTP_RESPONSE));
} else {
    /* HEAD: send only the status line + headers + blank line, no body.
     * Build a headers-only version of the same response. */
    const char *head_response =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        "Content-Length: 27\r\n"
        "Connection: close\r\n"
        "\r\n";
    send_all(client_fd, head_response, strlen(head_response));
}
```
The `Content-Length` in the HEAD response (27) must match what the GET response would send — not the bytes actually transmitted. This is the RFC requirement.
**Checkpoint 6**: `curl -I http://localhost:8080/` returns headers with `Content-Length: 27` and no body. Verify: `curl -s -I http://localhost:8080/ | wc -c` is larger than 0 (headers were received) but the body is empty.
```bash
# Verify HEAD body is empty
BODY_BYTES=$(curl -s --head http://localhost:8080/ -o /dev/null -w "%{size_download}")
echo "Body bytes: $BODY_BYTES"  # must be 0
```
---
### Phase 7 — Testing (0.5–1 h)
Run `test_parse.sh` (Section 8). All tests must pass before this milestone is considered complete.
---
## 8. Test Specification

![Request Validation: Error Response Decision Tree](./diagrams/tdd-diag-15.svg)

### 8.1 `parse_request_line()` Tests
**T1.1 — Happy path: standard GET**
```
Input:  "GET /index.html HTTP/1.1"
Expect: rc=0, method="GET", path="/index.html", version="HTTP/1.1"
```
**T1.2 — Happy path: root path**
```
Input:  "GET / HTTP/1.1"
Expect: rc=0, method="GET", path="/", version="HTTP/1.1"
```
**T1.3 — Happy path: HEAD method**
```
Input:  "HEAD /css/style.css HTTP/1.0"
Expect: rc=0, method="HEAD", path="/css/style.css", version="HTTP/1.0"
```
**T1.4 — No first space**
```
Input:  "GETHTTP/1.1"
Expect: rc=-1
```
**T1.5 — No second space (no version)**
```
Input:  "GET /path"
Expect: rc=-1
```
**T1.6 — Empty method**
```
Input:  " /path HTTP/1.1"  (leading space)
Expect: rc=-1
```
**T1.7 — Empty path**
```
Input:  "GET  HTTP/1.1"  (two spaces → empty path between them)
Expect: rc=-1
```
**T1.8 — Path exactly at MAX_PATH_LEN - 1 (boundary, should succeed)**
```
Input:  "GET /" + 8190 'a' chars + " HTTP/1.1"  (path_len = 8191)
Expect: rc=0, path length = 8191
```
**T1.9 — Path at MAX_PATH_LEN (414)**
```
Input:  "GET /" + 8191 'a' chars + " HTTP/1.1"  (path_len = 8192)
Expect: rc=-2
```
**T1.10 — CRLF line endings (trailing \r stripped by caller before this function)**
```
Input:  "GET / HTTP/1.1" (no \r — caller already stripped it)
Expect: rc=0, version="HTTP/1.1" (not "HTTP/1.1\r")
```
### 8.2 `parse_headers()` Tests
**T2.1 — Normal headers with CRLF**
```
Input:  "GET / HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n"
Expect: header_count=2
        headers[0].name="host"            headers[0].value="localhost"
        headers[1].name="content-length"  headers[1].value="0"
        body=NULL or body_len=0
```
**T2.2 — Case normalization**
```
Input:  "GET / HTTP/1.1\r\nContent-TYPE: Text/HTML\r\n\r\n"
Expect: headers[0].name="content-type"   headers[0].value="Text/HTML"
Note:  name is lowercased; value case is PRESERVED
```
**T2.3 — OWS stripping on both sides**
```
Input:  "GET / HTTP/1.1\r\nX-Header:   leading and trailing   \r\n\r\n"
Expect: headers[0].value="leading and trailing"
```
**T2.4 — OWS with tabs**
```
Input:  "GET / HTTP/1.1\r\nX-Tab:\there\t\r\n\r\n"
Expect: headers[0].value="here"
```
**T2.5 — obs-fold unfolding**
```
Input:  "GET / HTTP/1.1\r\nSubject: part1\r\n continuation\r\n\r\n"
Expect: headers[0].name="subject"   headers[0].value="part1 continuation"
Note:  leading whitespace on continuation stripped; joined with single space
```
**T2.6 — obs-fold with no prior header (skip, not crash)**
```
Input:  "GET / HTTP/1.1\r\n  orphan continuation\r\nHost: x\r\n\r\n"
Expect: headers[0].name="host"   header_count=1   (orphan line skipped)
```
**T2.7 — Bare LF line endings**
```bash
printf "GET / HTTP/1.1\nHost: localhost\nX-Test: value\n\n" | nc -q 1 localhost 8080
# Server must respond 200 OK, log both headers correctly
```
**T2.8 — Mixed CRLF and bare LF**
```
Input:  "GET / HTTP/1.1\r\nHost: localhost\nX-Mixed: yes\r\n\r\n"
Expect: header_count=2, all values correct
```
**T2.9 — Header line with no colon (skipped)**
```
Input:  "GET / HTTP/1.1\r\nNotAHeader\r\nHost: x\r\n\r\n"
Expect: headers[0].name="host"   header_count=1
```
**T2.10 — 33 headers (MAX_HEADERS + 1): 33rd silently dropped**
```
Input:  request line + 33 unique headers + empty line
Expect: header_count=32   (33rd absent from array)
```
**T2.11 — Body pointer set correctly**
```
Input:  "GET / HTTP/1.1\r\nHost: x\r\n\r\nBODY_DATA"
Expect: req.body points to "BODY_DATA"
        req.body_len = 9
```
**T2.12 — Empty value (colon immediately before OWS/EOL)**
```
Input:  "GET / HTTP/1.1\r\nX-Empty:   \r\n\r\n"
Expect: headers[0].value=""   (empty string after OWS strip)
```
### 8.3 `http_parse_request()` Integration Tests
**T3.1 — Full curl GET round-trip**
```bash
curl -v http://localhost:8080/
# Server log must show: GET / HTTP/1.1  Host: localhost:8080
# Response: 200 OK
```
**T3.2 — HEAD method: headers only, no body**
```bash
BODY=$(curl -s --head http://localhost:8080/)
BODY_SIZE=$(curl -s --head http://localhost:8080/ -o /dev/null -w "%{size_download}")
echo "Body size: $BODY_SIZE"   # must be 0
curl -sI http://localhost:8080/ | grep "Content-Length: 27"  # must match
```
**T3.3 — DELETE returns 501**
```bash
curl -sv -X DELETE http://localhost:8080/ 2>&1 | grep "501 Not Implemented"
```
**T3.4 — POST returns 501**
```bash
curl -sv -X POST -d "data" http://localhost:8080/ 2>&1 | grep "501 Not Implemented"
```
**T3.5 — Oversized URI returns 414**
```bash
python3 -c "
import socket
s = socket.socket()
s.connect(('localhost', 8080))
path = 'a' * 8192
s.send(f'GET /{path} HTTP/1.1\r\nHost: localhost\r\n\r\n'.encode())
resp = s.recv(4096).decode()
print(resp[:30])
s.close()
" 
# Must print: HTTP/1.1 414 URI Too Long
```
**T3.6 — Malformed request line (no spaces) returns 400**
```bash
printf "GARBAGE\r\n\r\n" | nc -q 1 localhost 8080 | head -1
# Must print: HTTP/1.1 400 Bad Request
```
**T3.7 — Unrecognized HTTP version returns 400**
```bash
printf "GET / HTTP/3.0\r\nHost: localhost\r\n\r\n" | nc -q 1 localhost 8080 | head -1
# Must print: HTTP/1.1 400 Bad Request
```
**T3.8 — Missing Host header: server continues (logs warning)**
```bash
printf "GET / HTTP/1.1\r\n\r\n" | nc -q 1 localhost 8080
# Server responds 200 OK (not 400); stderr shows "missing required Host header"
```
**T3.9 — Telnet bare-LF robustness**
```bash
(printf "GET / HTTP/1.1\nHost: localhost\nAccept: */*\n\n"; sleep 1) | nc localhost 8080
# Must receive: HTTP/1.1 200 OK  (not a hang or crash)
```
**T3.10 — Content-Length in error responses is accurate**
```bash
for METHOD in DELETE PUT PATCH OPTIONS; do
  BODY=$(curl -s -X $METHOD http://localhost:8080/)
  HEADERS=$(curl -sI -X $METHOD http://localhost:8080/)
  CL=$(echo "$HEADERS" | grep -i "content-length" | awk '{print $2}' | tr -d '\r')
  ACTUAL=$(echo -n "$BODY" | wc -c | tr -d ' ')
  if [ "$CL" = "$ACTUAL" ]; then
    echo "PASS $METHOD: Content-Length=$CL"
  else
    echo "FAIL $METHOD: Content-Length=$CL vs actual=$ACTUAL"
  fi
done
```
### 8.4 Complete `test_parse.sh`

![M2 Module Architecture: Parser Functions and Data Flow](./diagrams/tdd-diag-16.svg)

```bash
#!/bin/bash
set -e
SERVER_BIN=./server
PORT=18081
$SERVER_BIN $PORT &
SERVER_PID=$!
sleep 0.3
trap "kill $SERVER_PID 2>/dev/null; exit" EXIT INT TERM
pass() { echo "PASS: $1"; }
fail() { echo "FAIL: $1"; kill $SERVER_PID 2>/dev/null; exit 1; }
BASE="http://localhost:$PORT"
# T1: Normal GET
curl -s "$BASE/" | grep -q "Hello from C server" && pass "T1: GET 200" || fail "T1"
# T2: HEAD returns no body
BODY_BYTES=$(curl -s --head "$BASE/" -o /dev/null -w "%{size_download}")
[ "$BODY_BYTES" = "0" ] && pass "T2: HEAD no body" || fail "T2: got $BODY_BYTES bytes"
# T3: HEAD Content-Length matches GET
HEAD_CL=$(curl -sI "$BASE/" | grep -i "content-length" | awk '{print $2}' | tr -d '\r')
GET_BODY=$(curl -s "$BASE/" | wc -c | tr -d ' ')
[ "$HEAD_CL" = "$GET_BODY" ] && pass "T3: HEAD Content-Length matches GET" || fail "T3: $HEAD_CL vs $GET_BODY"
# T4: DELETE returns 501
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "$BASE/")
[ "$STATUS" = "501" ] && pass "T4: DELETE=501" || fail "T4: got $STATUS"
# T5: Oversized URI returns 414
STATUS=$(python3 -c "
import socket, sys
s = socket.socket()
s.connect(('localhost', $PORT))
s.send(b'GET /' + b'a'*8192 + b' HTTP/1.1\r\nHost: localhost\r\n\r\n')
r = s.recv(64).decode()
s.close()
code = r.split()[1] if len(r.split()) > 1 else '0'
print(code)
")
[ "$STATUS" = "414" ] && pass "T5: 414 URI Too Long" || fail "T5: got $STATUS"
# T6: Malformed request returns 400
STATUS=$(printf "BADREQUEST\r\n\r\n" | nc -q 1 localhost $PORT | awk '{print $2}')
[ "$STATUS" = "400" ] && pass "T6: 400 Bad Request" || fail "T6: got $STATUS"
# T7: Bare LF line endings
STATUS=$(printf "GET / HTTP/1.1\nHost: localhost\n\n" | nc -q 1 localhost $PORT | awk '{print $2}')
[ "$STATUS" = "200" ] && pass "T7: bare-LF accepted" || fail "T7: got $STATUS"
# T8: Case-insensitive header storage
curl -s -H "CONTENT-TYPE: application/json" "$BASE/" > /dev/null && pass "T8: case-insensitive header" || fail "T8"
# T9: OWS in header values
STATUS=$(printf "GET / HTTP/1.1\r\nHost:   localhost   \r\n\r\n" | nc -q 1 localhost $PORT | awk '{print $2}')
[ "$STATUS" = "200" ] && pass "T9: OWS in header value" || fail "T9: got $STATUS"
# T10: Error response Content-Length accuracy
for METHOD in DELETE POST; do
  CL=$(curl -sI -X $METHOD "$BASE/" | grep -i "content-length" | awk '{print $2}' | tr -d '\r')
  BODY=$(curl -s -X $METHOD "$BASE/")
  ACTUAL=$(echo -n "$BODY" | wc -c | tr -d ' ')
  [ "$CL" = "$ACTUAL" ] && pass "T10.$METHOD: Content-Length accurate" || fail "T10.$METHOD: CL=$CL actual=$ACTUAL"
done
# T11: Unrecognized version returns 400
STATUS=$(printf "GET / HTTP/9.9\r\nHost: localhost\r\n\r\n" | nc -q 1 localhost $PORT | awk '{print $2}')
[ "$STATUS" = "400" ] && pass "T11: bad version=400" || fail "T11: got $STATUS"
echo ""
echo "All parse tests passed."
```
---
## 9. Performance Targets

![HEAD vs GET Response Data Flow](./diagrams/tdd-diag-17.svg)

| Operation | Target | How to Measure |
|---|---|---|
| Full parse of 2 KB request with 15 headers | < 5 µs | `perf stat -e cycles ./parse_bench` or custom `clock_gettime(CLOCK_MONOTONIC)` loop × 100,000 |
| `get_header()` scan of 32-entry array | < 500 ns | inline benchmark: 1M calls, measure total with `clock_gettime` |
| parse_headers() allocation | 0 bytes heap | `valgrind --tool=massif ./server`; confirm 0 `malloc` calls per request |
| Stack frame size per connection | ≤ 64 KB | `gcc -Wframe-larger-than=65536 http_parse.c server.c` must produce no warnings |
| L1 cache containment | All scanning in L1 | Buffer is 8 KB; L1 is ≥ 32 KB; all `strstr`/`memchr` scans hit L1 after first touch |
| Throughput (sequential loopback) | > 2,000 req/s | `ab -n 20000 -c 1 http://localhost:8080/` → "Requests per second" |
| Memory per connection | 0 bytes heap | `valgrind --leak-check=full ./server` → "0 bytes lost" after 1000 requests |
| 400/501 error responses under load | 0 failed requests | `ab -n 1000 -c 10 -m DELETE http://localhost:8080/` → "Failed requests: 0" |
**How to measure parse latency**:
```c
/* parse_bench.c */
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "http_parse.h"
int main(void) {
    char buf[8192] =
        "GET /index.html HTTP/1.1\r\n"
        "Host: localhost:8080\r\n"
        "User-Agent: bench/1.0\r\n"
        "Accept: text/html,application/xhtml+xml\r\n"
        "Accept-Language: en-US,en;q=0.9\r\n"
        "Accept-Encoding: gzip, deflate\r\n"
        "Connection: keep-alive\r\n"
        "Cache-Control: max-age=0\r\n"
        "\r\n";
    size_t buf_len = strlen(buf);
    struct timespec t0, t1;
    const int N = 100000;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < N; i++) {
        char tmp[8192];
        memcpy(tmp, buf, buf_len + 1);
        http_request_t req;
        http_parse_request(tmp, buf_len, &req);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    long ns = (t1.tv_sec - t0.tv_sec) * 1000000000L + (t1.tv_nsec - t0.tv_nsec);
    printf("%.1f ns/parse (%.1f µs)\n", (double)ns / N, (double)ns / N / 1000.0);
    /* target: < 5000 ns (5 µs) */
    return 0;
}
```
---
## 10. Threat Model
Inputs to this module arrive from untrusted clients over TCP. Every field must be treated as adversarial.
| Attack Vector | Mitigation in This Module |
|---|---|
| Unbounded header name | `name_len >= MAX_HEADER_NAME` check in `parse_headers()` → skip header |
| Unbounded header value | `value_len` capped at `MAX_HEADER_VALUE - 1` before `memcpy` |
| Unbounded URI | `path_len >= MAX_PATH_LEN` in `parse_request_line()` → return -2 → 414 |
| Path with null bytes | M3 responsibility; M2 stores the raw path including `\0` if somehow present; M3 must reject |
| Percent-encoded traversal | M3 responsibility; M2 stores raw path unchanged |
| Extremely long method | `method_len >= MAX_METHOD_LEN` check → -1 |
| obs-fold overflow of value buffer | `prev_len + 1 + cont_len < MAX_HEADER_VALUE - 1` check before `memcpy` |
| > 32 headers (memory exhaustion) | `req->header_count < MAX_HEADERS` loop condition → stop; no realloc |
| Integer overflow in length arithmetic | All lengths are `size_t`; comparisons are `>=` against compile-time constants |
| Missing `Host` header triggering UB | `get_header()` returns `NULL`; caller handles NULL safely |
---
## Makefile Update
```makefile
CC      = gcc
CFLAGS  = -Wall -Wextra -Wpedantic -std=c11 -O2 -g \
          -Wframe-larger-than=65536
TARGET  = server
all: $(TARGET)
$(TARGET): server.c http_parse.c
	$(CC) $(CFLAGS) -o $@ $^
parse_bench: parse_bench.c http_parse.c
	$(CC) $(CFLAGS) -o $@ $^
clean:
	rm -f $(TARGET) parse_bench
test: $(TARGET)
	bash test_parse.sh
bench: parse_bench
	./parse_bench
.PHONY: all clean test bench
```
`-Wframe-larger-than=65536` will fire if any single function's stack frame exceeds 64 KB. The combined `buf[8192]` + `http_request_t` (~43 KB) puts `handle_client()` at ~51 KB — within limit. If you add another large local variable, the warning will alert you before a thread-stack overflow occurs in M4.
---
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: http-server-basic-m3 -->
# Technical Design Document: Static File Serving
## Module `http-server-basic-m3`
---
## 1. Module Charter
This module translates a parsed `http_request_t` (from M2) into a fully-formed HTTP/1.1 file response by executing a five-stage security pipeline: URL percent-decoding, root-relative path concatenation, `realpath()` canonicalization, containment verification, and file streaming. It detects MIME types from file extensions, serves file bytes with correct `Content-Type`, `Content-Length`, and `Last-Modified` headers, auto-serves `index.html` for directory paths, and implements `If-Modified-Since` / `304 Not Modified` conditional request support.
This module does **not** handle TCP socket lifecycle, HTTP request parsing, concurrent connections, chunked transfer encoding, range requests, directory listing, ETag generation, or gzip compression. It receives an already-accepted and already-parsed connection; the caller is responsible for the FD lifecycle.
**Upstream dependency**: M2 delivers a valid `http_request_t` with `path[]` set to the raw (not percent-decoded) request URI path and headers accessible via `get_header()`. The caller supplies a pre-canonicalized `doc_root` string (i.e., `doc_root` itself has already been run through `realpath()` at server startup — never at request time). The caller supplies `client_fd` and `send_body` (1 for GET, 0 for HEAD).
**Downstream dependency**: M4's keep-alive loop calls `serve_file()` once per request/response cycle; `serve_file()` must be fully stateless and re-entrant so that multiple threads may call it simultaneously on different connections.
**Invariants that must always hold**:
- `file_fd` opened inside `serve_file()` is `close()`d on every exit path, including all error paths and the mid-transfer disconnect case.
- A path that resolves outside `doc_root` always produces 403, never 404 — regardless of whether the target file exists. This prevents filesystem enumeration attacks.
- `Content-Length` in every response (200, 304, 403, 404, 500) equals the exact byte count of the response body that follows. Zero-body responses (304, HEAD) carry `Content-Length` reflecting the body size a GET would have returned.
- No function in this module calls `malloc`, `realloc`, or `free`. All buffers are stack-allocated with compile-time-known sizes.
- The pre-canonicalized `doc_root` is never modified; all path operations produce new buffers.
---
## 2. File Structure
Create files in this order:
```
http-server/
├── 1  file_server.h       # structs, constants, all function declarations
├── 2  file_server.c       # full implementation
├── 3  http_parse.h        # unchanged from M2 (dependency)
├── 4  http_parse.c        # unchanged from M2 (dependency)
├── 5  server.c            # updated: handle_client() calls serve_file()
├── 6  Makefile            # updated build rules
└── 7  test_file_server.sh # automated test script (Phase 10)
```
`file_server.h` and `file_server.c` are new files. `server.c` is modified in-place: `handle_client()` replaces the hardcoded response with a call to `serve_file()`. The socket lifecycle and `read_request()` are unchanged.
---
## 3. Complete Data Model
### 3.1 Constants
```c
/* file_server.h */
#define DOC_ROOT_MAX        1024          /* max bytes in doc_root string     */
#define FULL_PATH_MAX       (DOC_ROOT_MAX + MAX_PATH_LEN + 2)
#define FILE_READ_BUF_SIZE  65536         /* 64 KB read buffer; matches NIC MTU aggregation */
#define RESPONSE_HDR_MAX    1024          /* max bytes in the response header block          */
#define HTTP_DATE_LEN       64            /* "Wed, 01 Jan 2025 00:00:00 GMT\0" = 30 bytes;  */
                                          /* 64 gives safe headroom                          */
#define MIME_TYPE_MAX_LEN   64            /* longest MIME string in table                    */
```
`FILE_READ_BUF_SIZE = 65536`: chosen to match the typical NIC receive coalescing window and the Linux TCP default receive buffer size (87 KB). At 64 KB per `read()`, a 1 MB file requires 16 syscalls; a 4 KB file requires 1 syscall. Larger buffers (256 KB, 1 MB) give diminishing returns because the bottleneck shifts to `send_all()` flushing the kernel send buffer.
`RESPONSE_HDR_MAX = 1024`: the longest possible 200 OK header block is approximately:
```
HTTP/1.1 200 OK\r\n                               (15 bytes)
Content-Type: application/octet-stream\r\n        (40 bytes)
Content-Length: 18446744073709551615\r\n           (44 bytes — 20-digit uint64)
Last-Modified: Wed, 01 Jan 2025 00:00:00 GMT\r\n  (46 bytes)
Connection: keep-alive\r\n                         (24 bytes)
\r\n                                               (2 bytes)
```
Total: ~171 bytes. 1024 bytes provides 6× headroom for future headers.
### 3.2 `mime_entry_t` and MIME Table
```c
/* file_server.h */
typedef struct {
    const char *extension;   /* lowercase, leading dot: ".html" */
    const char *mime_type;   /* full MIME string: "text/html; charset=utf-8" */
} mime_entry_t;
```
The table is a static const array terminated by a `{NULL, NULL}` sentinel, defined in `file_server.c`. The extension strings are all lowercase; the lookup uses `strcasecmp()` so that `.HTML`, `.Html`, and `.html` all match.
**Memory layout of `mime_entry_t`** (64-bit, 8-byte pointer alignment):
```
Offset   Field       Type          Size
0x00     extension   const char *  8 bytes
0x08     mime_type   const char *  8 bytes
Total:                             16 bytes  (one cache line fits 4 entries)
```
With 18 entries + sentinel = 19 entries × 16 bytes = 304 bytes total. The entire table fits in 5 cache lines (320 bytes). A linear scan of all 19 entries touches at most 5 cache lines — all L1-resident after the first request.
### 3.3 No New Structs
This module introduces no new structs beyond `mime_entry_t`. All state is passed as parameters or lives in local stack variables within `serve_file()` and its helpers. This is an intentional design choice: `serve_file()` is a pure transformation from `(client_fd, doc_root, http_request_t*, send_body)` → side effects (bytes written to socket).
### 3.4 Stack Frame Inventory for `serve_file()`
```
Variable                     Type              Size
decoded_path[MAX_PATH_LEN]   char[8192]        8192 bytes
full_path[PATH_MAX]          char[4096]        4096 bytes
canonical[PATH_MAX]          char[4096]        4096 bytes
index_path[PATH_MAX]         char[4096]        4096 bytes
last_modified[HTTP_DATE_LEN] char[64]          64   bytes
header_buf[RESPONSE_HDR_MAX] char[1024]        1024 bytes
read_buf[FILE_READ_BUF_SIZE] char[65536]       65536 bytes
st                           struct stat       144  bytes (Linux x86-64)
gmt_result                   struct tm         56   bytes
ims                          time_t            8    bytes
file_fd                      int               4    bytes
(alignment/misc)                               ~100 bytes
                             TOTAL:           ~91 KB
```
**Stack overflow risk**: 91 KB exceeds Linux's default stack size (8 MB) by a large margin? No — 91 KB is well under 8 MB. However, `read_buf[65536]` is the dominant cost. If thread stack sizes are reduced (e.g., custom `pthread_attr_setstacksize` to 128 KB), this will overflow. Document this constraint for M4.
**Mitigation if needed**: move `read_buf` to static storage (not thread-safe) or reduce `FILE_READ_BUF_SIZE`. For M3 (single-threaded), the stack allocation is correct.

![URL Path → Filesystem Path Security Pipeline: 5 Stages](./diagrams/tdd-diag-18.svg)

---
## 4. Interface Contracts
### 4.1 `serve_file()`
```c
/* file_server.h */
void serve_file(int client_fd,
                const char *doc_root,
                const http_request_t *req,
                int send_body);
```
**Parameters**:
- `client_fd`: a connected socket FD in writable state. The callee writes the response to it. Does NOT close it — the caller (M1's accept loop / M4's keep-alive loop) closes it.
- `doc_root`: a pre-canonicalized absolute directory path with no trailing slash. Must have been produced by `realpath()` at server startup. Must be null-terminated. Length < `DOC_ROOT_MAX`. Must remain valid for the duration of this call (it is a pointer into `main()`'s data segment or `argv[]`, both permanent).
- `req`: pointer to a valid `http_request_t` produced by `http_parse_request()` returning `HTTP_PARSE_OK`. `req->path` contains the raw request URI path (percent-encoded, not decoded). `req->headers` is accessible via `get_header()`.
- `send_body`: 1 for GET (send status line + headers + file contents), 0 for HEAD (send status line + headers only).
**Return value**: void. All outcomes — success, 404, 403, 304, 500 — are communicated by writing to `client_fd`. The caller does not need to take any action after `serve_file()` returns beyond closing `client_fd`.
**Guaranteed postconditions**:
- Every `file_fd` opened by this function is closed before it returns.
- `client_fd` is left open (not closed by this function).
- No heap allocation was performed.
**Thread safety**: fully re-entrant. All state is on the stack. `realpath()`, `stat()`, `open()`, `read()`, `gmtime()`, `strftime()` are all thread-safe on Linux with `_REENTRANT` defined.
**Caller integration in `server.c`**:
```c
/* Updated handle_client() in server.c — replaces M2's hardcoded response */
void handle_client(int client_fd) {
    char buf[REQUEST_BUF_SIZE];
    ssize_t req_len = read_request(client_fd, buf, sizeof(buf));
    if (req_len <= 0) {
        send_all(client_fd, HTTP_400, strlen(HTTP_400));
        return;
    }
    http_request_t req;
    int rc = http_parse_request(buf, (size_t)req_len, &req);
    if (rc == HTTP_PARSE_URI_TOO_LONG) { send_all(client_fd, HTTP_414, strlen(HTTP_414)); return; }
    if (rc == HTTP_PARSE_NOT_IMPL)     { send_all(client_fd, HTTP_501, strlen(HTTP_501)); return; }
    if (rc != HTTP_PARSE_OK)           { send_all(client_fd, HTTP_400, strlen(HTTP_400)); return; }
    int send_body = (classify_method(req.method) == HTTP_METHOD_GET);
    serve_file(client_fd, g_doc_root, &req, send_body);
}
```
`g_doc_root` is a global `const char *` set in `main()` after `realpath()` canonicalization of `argv[3]` or the default path.
---
### 4.2 `url_decode()`
```c
/* file_server.h */
int url_decode(const char *src, char *dst, size_t dst_size);
```
**Parameters**:
- `src`: null-terminated percent-encoded path from `req->path`. Length ≤ `MAX_PATH_LEN`.
- `dst`: caller-supplied output buffer, exactly `dst_size` bytes. Must be ≥ `strlen(src) + 1` (decoded is never longer than encoded).
- `dst_size`: byte capacity of `dst`.
**Return values**:
- `0`: success. `dst` contains the decoded path, null-terminated. `strlen(dst) <= strlen(src)`.
- `-1`: malformed percent-encoding. Specifically: `%` not followed by two hex digits. `dst` contents are undefined on failure.
- `-2`: null byte in decoded output. A `%00` sequence decodes to byte 0x00. This is a security-critical rejection — null bytes can truncate path strings and bypass containment checks. `dst` contents are undefined on failure.
- `-3`: `dst_size` exhausted before `src` null terminator was reached (output buffer too small). This should not occur if `dst_size >= strlen(src) + 1`, but the check is mandatory.
**Algorithm**: see Section 5.1.
**Edge cases**:
- `src = "/"` → `dst = "/"`, return 0.
- `src = "/path%20with%20spaces"` → `dst = "/path with spaces"`, return 0.
- `src = "/%2e%2e%2fetc%2fpasswd"` → `dst = "/../etc/passwd"`, return 0. (The `..` traversal is handled by `realpath()` in Stage 3, not here.)
- `src = "/%00secret"` → return -2. The null byte is detected after decoding `%00`.
- `src = "/%GG"` → return -1. `GG` is not valid hex.
- `src = "/%2"` → return -1. Only one hex digit follows `%`.
- `src = "/file+name"` → `dst = "/file+name"`. In path segments, `+` is literal, not space.
---
### 4.3 `build_full_path()`
```c
/* file_server.h */
int build_full_path(const char *doc_root, const char *decoded_path,
                    char *full_path, size_t full_path_size);
```
**Parameters**:
- `doc_root`: pre-canonicalized document root, no trailing slash.
- `decoded_path`: percent-decoded URL path, must start with `/`, null-terminated.
- `full_path`: output buffer, `full_path_size` bytes.
- `full_path_size`: must be ≥ `strlen(doc_root) + strlen(decoded_path) + 1`.
**Return values**:
- `0`: success. `full_path` = `doc_root + decoded_path`, null-terminated.
- `-1`: output would exceed `full_path_size`, or `decoded_path` does not start with `/`.
**Postcondition**: `full_path` contains `doc_root` immediately followed by `decoded_path` (which begins with `/`). No trailing slash is added.
---
### 4.4 `get_mime_type()`
```c
/* file_server.h */
const char *get_mime_type(const char *path);
```
**Parameters**: `path` — the canonical filesystem path (after `realpath()`). Null-terminated.
**Return value**: a pointer to a static const string (from the MIME table). Never NULL. Returns `"application/octet-stream"` for unknown or missing extensions.
**Algorithm**: `strrchr(path, '.')` to find the last dot. If `NULL` or the dot is the last character (no extension), return `"application/octet-stream"`. Otherwise, linear scan the `MIME_TABLE` using `strcasecmp(dot, entry.extension)`.
---
### 4.5 `parse_http_date()`
```c
/* file_server.h */
time_t parse_http_date(const char *date_str);
```
**Parameters**: `date_str` — the value of the `If-Modified-Since` header, or `NULL`.
**Return values**:
- `(time_t)-1`: `date_str` is NULL, or the string does not match any recognized HTTP-date format. Callers treat `(time_t)-1` as "header absent or unparseable" → serve 200 unconditionally.
- Any other `time_t` value: the UTC timestamp represented by `date_str`.
**Recognized formats** (attempted in order):
1. RFC 7231 preferred: `"Wed, 01 Jan 2025 12:00:00 GMT"` — format `"%a, %d %b %Y %H:%M:%S GMT"`.
2. RFC 850 obsolete: `"Wednesday, 01-Jan-25 12:00:00 GMT"` — format `"%A, %d-%b-%y %H:%M:%S GMT"`.
3. ANSI C `asctime()` format: `"Wed Jan  1 12:00:00 2025"` — format `"%a %b %e %H:%M:%S %Y"`.
Must use `timegm()` (not `mktime()`) to convert `struct tm` to `time_t`, because HTTP dates are always UTC and `mktime()` applies the local timezone.
---
### 4.6 `send_404()`, `send_403()`, `send_304()`, `send_500()`
```c
/* file_server.h */
void send_404(int client_fd);
void send_403(int client_fd);
void send_304(int client_fd, const char *mime_type,
              const char *last_modified, off_t content_length,
              int send_body);
void send_500(int client_fd);
```
Each sends a complete, self-contained HTTP response and returns. Does not close `client_fd`. Uses `send_all()` from M1.
`send_304()` parameters: `mime_type` and `last_modified` are the values that would appear in a 200 response. `content_length` is `st.st_size`. `send_body` is always 0 for 304 (no body), but the parameter is kept for consistency; the implementation must never send a body regardless of `send_body`.
---
## 5. Algorithm Specification
### 5.1 `url_decode()` — Percent-Decode with Security Checks

![Directory Traversal: Three Bypass Vectors and Their Neutralization](./diagrams/tdd-diag-19.svg)

```
INPUT:  src       = percent-encoded path (null-terminated)
        dst       = output buffer
        dst_size  = output buffer capacity
OUTPUT: 0 on success; -1 malformed encoding; -2 null byte; -3 buffer full
i ← 0  (src index)
j ← 0  (dst index)
LOOP while src[i] != '\0':
  if j >= dst_size - 1:
    return -3                       ← output buffer full
  if src[i] == '%':
    if src[i+1] == '\0' OR src[i+2] == '\0':
      return -1                     ← truncated %XX sequence
    if NOT isxdigit((unsigned char)src[i+1]):
      return -1                     ← first hex digit invalid
    if NOT isxdigit((unsigned char)src[i+2]):
      return -1                     ← second hex digit invalid
    hex[0] ← src[i+1]
    hex[1] ← src[i+2]
    hex[2] ← '\0'
    decoded_byte ← (char)strtol(hex, NULL, 16)
    if decoded_byte == '\0':
      return -2                     ← null byte in path: security rejection
    dst[j] ← decoded_byte
    i ← i + 3
    j ← j + 1
  else:
    dst[j] ← src[i]
    i ← i + 1
    j ← j + 1
dst[j] ← '\0'
return 0
```
**Critical note on `isxdigit` cast**: `isxdigit()` has undefined behavior on negative `int` values (except `EOF`). `src[i+1]` is a `char`, which is signed on most platforms. Cast every character to `unsigned char` before passing to any `ctype.h` function. This is the same pattern as the `tolower()` cast in M2.
---
### 5.2 `serve_file()` — The Five-Stage Security Pipeline

![url_decode() Algorithm: Percent-Decode State Machine](./diagrams/tdd-diag-20.svg)

The complete orchestrator function, step by step:
```
INPUT:  client_fd, doc_root, req, send_body
──── STAGE 1: URL Decode ──────────────────────────────────────────────────
char decoded_path[MAX_PATH_LEN]
rc ← url_decode(req->path, decoded_path, sizeof(decoded_path))
if rc == -1: send_400(client_fd); return
if rc == -2: send_400(client_fd); return        ← null byte rejection
if rc == -3: send_400(client_fd); return        ← impossible if MAX_PATH_LEN sized
──── STAGE 2: Concatenate with document root ──────────────────────────────
char full_path[PATH_MAX]
rc ← build_full_path(doc_root, decoded_path, full_path, sizeof(full_path))
if rc != 0: send_403(client_fd); return         ← path too long or missing leading /
──── STAGE 3: Canonicalize — realpath() ───────────────────────────────────
char canonical[PATH_MAX]
if realpath(full_path, canonical) == NULL:
  if errno == ENOENT OR errno == ENOTDIR:
    send_404(client_fd); return
  else:                                         ← EACCES, ELOOP, ENAMETOOLONG, etc.
    send_403(client_fd); return
──── STAGE 4: Containment check ────────────────────────────────────────────
size_t root_len ← strlen(doc_root)
if strncmp(canonical, doc_root, root_len) != 0:
  send_403(client_fd); return                   ← does not even start with doc_root
if canonical[root_len] != '/' AND canonical[root_len] != '\0':
  send_403(client_fd); return                   ← prefix match on a longer sibling dir
──── STAGE 4b: Directory detection and index.html append ───────────────────
struct stat st
if stat(canonical, &st) < 0:
  send_404(client_fd); return
if S_ISDIR(st.st_mode):
  char index_path[PATH_MAX]
  n ← snprintf(index_path, sizeof(index_path), "%s/index.html", canonical)
  if n < 0 OR (size_t)n >= sizeof(index_path):
    send_403(client_fd); return
  ← re-canonicalize the index path (index.html itself may be a symlink)
  if realpath(index_path, canonical) == NULL:
    send_403(client_fd); return                 ← index.html absent or inaccessible → 403
  ← re-check containment after following the index symlink
  if strncmp(canonical, doc_root, root_len) != 0 OR
     (canonical[root_len] != '/' AND canonical[root_len] != '\0'):
    send_403(client_fd); return
  ← re-stat the index file
  if stat(canonical, &st) < 0:
    send_404(client_fd); return
──── STAGE 5: Conditional request (If-Modified-Since) ──────────────────────
char last_modified[HTTP_DATE_LEN]
struct tm *gmt ← gmtime(&st.st_mtime)
strftime(last_modified, sizeof(last_modified), "%a, %d %b %Y %H:%M:%S GMT", gmt)
const char *mime ← get_mime_type(canonical)
time_t ims ← parse_http_date(get_header(req, "if-modified-since"))
if ims != (time_t)-1 AND st.st_mtime <= ims:
  send_304(client_fd, mime, last_modified, st.st_size, send_body)
  return                                        ← no file open needed
──── STAGE 6: Open the file ────────────────────────────────────────────────
int file_fd ← open(canonical, O_RDONLY)
if file_fd < 0:
  if errno == EACCES: send_403(client_fd)
  else:               send_404(client_fd)
  return
──── STAGE 7: Build and send response headers ──────────────────────────────
char header_buf[RESPONSE_HDR_MAX]
int header_len ← snprintf(header_buf, sizeof(header_buf),
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: %s\r\n"
    "Content-Length: %lld\r\n"
    "Last-Modified: %s\r\n"
    "Connection: %s\r\n"
    "\r\n",
    mime,
    (long long)st.st_size,
    last_modified,
    "close")                                    ← M4 will pass keep_alive flag here
if header_len < 0 OR (size_t)header_len >= sizeof(header_buf):
  close(file_fd)
  send_500(client_fd)
  return
if send_all(client_fd, header_buf, (size_t)header_len) < 0:
  close(file_fd)
  return                                        ← client disconnected; no error response possible
──── STAGE 8: Stream file body (GET only) ──────────────────────────────────
if send_body:
  char read_buf[FILE_READ_BUF_SIZE]
  ssize_t bytes_read
  while (bytes_read ← read(file_fd, read_buf, sizeof(read_buf))) > 0:
    if send_all(client_fd, read_buf, (size_t)bytes_read) < 0:
      break                                     ← client disconnected; stop sending
  if bytes_read < 0:
    perror("read")                              ← disk I/O error; headers already sent
close(file_fd)                                  ← MANDATORY on every path above
return
```

![realpath() Internals: Kernel dentry/inode Walk](./diagrams/tdd-diag-21.svg)

**Key invariant**: `close(file_fd)` appears exactly once, at the bottom of the function, because all early-return paths either (a) never opened `file_fd` (Stages 1–5 all return before the open), or (b) call `close(file_fd)` explicitly before returning (Stages 7–8 errors). Code review must verify this property.
---
### 5.3 `build_full_path()` — Path Concatenation
```
INPUT:  doc_root, decoded_path, full_path, full_path_size
root_len ← strlen(doc_root)
path_len ← strlen(decoded_path)
if decoded_path[0] != '/':
  return -1                   ← decoded_path must be absolute
← strip trailing slash from doc_root (if any) to avoid double-slash
while root_len > 1 AND doc_root[root_len - 1] == '/':
  root_len--                  ← shrink conceptually; don't modify doc_root
total_len ← root_len + path_len
if total_len + 1 > full_path_size:
  return -1                   ← would overflow output buffer
memcpy(full_path, doc_root, root_len)
memcpy(full_path + root_len, decoded_path, path_len + 1)   ← +1 copies '\0'
return 0
```
---
### 5.4 `get_mime_type()` — Extension Lookup
```
INPUT:  path = canonical filesystem path
dot ← strrchr(path, '.')
if dot == NULL:
  return "application/octet-stream"
if *(dot + 1) == '\0':                     ← file ends with a dot, no extension
  return "application/octet-stream"
for each entry in MIME_TABLE until entry.extension == NULL:
  if strcasecmp(dot, entry.extension) == 0:
    return entry.mime_type
return "application/octet-stream"          ← unknown extension
```
---
### 5.5 `parse_http_date()` — Three-Format HTTP Date Parser
```
INPUT:  date_str (may be NULL)
if date_str == NULL: return (time_t)-1
struct tm tm = {0}
char *result
← Attempt format 1: RFC 7231 preferred
result ← strptime(date_str, "%a, %d %b %Y %H:%M:%S GMT", &tm)
if result != NULL: goto CONVERT
← Attempt format 2: RFC 850 obsolete
memset(&tm, 0, sizeof(tm))
result ← strptime(date_str, "%A, %d-%b-%y %H:%M:%S GMT", &tm)
if result != NULL: goto CONVERT
← Attempt format 3: ANSI C asctime()
memset(&tm, 0, sizeof(tm))
result ← strptime(date_str, "%a %b %e %H:%M:%S %Y", &tm)
if result != NULL: goto CONVERT
return (time_t)-1                          ← no format matched
CONVERT:
  tm.tm_isdst = 0                          ← HTTP dates are always UTC; no DST
  t ← timegm(&tm)
  if t == (time_t)-1: return (time_t)-1   ← timegm() overflow (year > 2038 on 32-bit)
  return t
```
**Why `timegm()` not `mktime()`**: `mktime()` treats `struct tm` as local time and applies the server's timezone offset. A server running in UTC+5 would compute timestamps 5 hours ahead of the actual UTC time, producing incorrect comparisons for every `If-Modified-Since` check. `timegm()` is a Linux/BSD extension (not POSIX standard) that treats `struct tm` as UTC. Define `#define _BSD_SOURCE` or `#define _GNU_SOURCE` before including `<time.h>` to expose it.
---
### 5.6 The MIME Type Table — Full Definition
```c
/* file_server.c — static data, not exported */
static const mime_entry_t MIME_TABLE[] = {
    { ".html",  "text/html; charset=utf-8"        },
    { ".htm",   "text/html; charset=utf-8"        },
    { ".css",   "text/css; charset=utf-8"         },
    { ".js",    "application/javascript"          },
    { ".mjs",   "application/javascript"          },
    { ".json",  "application/json"                },
    { ".txt",   "text/plain; charset=utf-8"       },
    { ".xml",   "application/xml"                 },
    { ".svg",   "image/svg+xml"                   },
    { ".png",   "image/png"                       },
    { ".jpg",   "image/jpeg"                      },
    { ".jpeg",  "image/jpeg"                      },
    { ".gif",   "image/gif"                       },
    { ".ico",   "image/x-icon"                    },
    { ".pdf",   "application/pdf"                 },
    { ".woff",  "font/woff"                       },
    { ".woff2", "font/woff2"                      },
    { ".webp",  "image/webp"                      },
    { NULL,     NULL                              },  /* sentinel */
};
```
The `charset=utf-8` suffix on all `text/*` types is mandatory. Without it, the browser uses heuristics to guess encoding, which fails on UTF-8 files containing multi-byte characters. This is consistent with what every major CDN and web server sends by default.
---
### 5.7 Error Response String Definitions
```c
/* file_server.c */
#define BODY_403 "<html><body><h1>403 Forbidden</h1>" \
                 "<p>Access denied.</p></body></html>"
#define BODY_404 "<html><body><h1>404 Not Found</h1>" \
                 "<p>The requested resource was not found.</p></body></html>"
#define BODY_500 "<html><body><h1>500 Internal Server Error</h1></body></html>"
_Static_assert(sizeof(BODY_403) - 1 == /* COUNT_AT_COMPILE_TIME */, "BODY_403 length wrong");
_Static_assert(sizeof(BODY_404) - 1 == /* COUNT_AT_COMPILE_TIME */, "BODY_404 length wrong");
_Static_assert(sizeof(BODY_500) - 1 == /* COUNT_AT_COMPILE_TIME */, "BODY_500 length wrong");
```
**Action required at implementation time**: run `printf '%s' '<html><body>...' | wc -c` for each body string and fill in the correct byte counts. Replace the `/* COUNT_AT_COMPILE_TIME */` placeholders and set matching `Content-Length` values in the response strings. The `_Static_assert` will catch any mismatch at compile time.
The `send_404()` / `send_403()` / `send_500()` implementations build the header+body using `snprintf` into a local buffer, then call `send_all()`:
```c
void send_404(int client_fd) {
    const char *body = BODY_404;
    char buf[512];
    int len = snprintf(buf, sizeof(buf),
        "HTTP/1.1 404 Not Found\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n"
        "%s",
        strlen(body), body);
    if (len > 0 && (size_t)len < sizeof(buf)) {
        send_all(client_fd, buf, (size_t)len);
    }
}
```
The `send_304()` implementation must never send a body. The `Content-Length` it sends reflects what the file body size *would have been*:
```c
void send_304(int client_fd, const char *mime_type,
              const char *last_modified, off_t content_length,
              int send_body) {
    (void)send_body;   /* always suppress body for 304 */
    char buf[512];
    int len = snprintf(buf, sizeof(buf),
        "HTTP/1.1 304 Not Modified\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %lld\r\n"
        "Last-Modified: %s\r\n"
        "Connection: close\r\n"
        "\r\n",
        mime_type,
        (long long)content_length,
        last_modified);
    if (len > 0 && (size_t)len < sizeof(buf)) {
        send_all(client_fd, buf, (size_t)len);
    }
}
```
---
## 6. Error Handling Matrix

![Containment Prefix Check: Why character-after-root Matters](./diagrams/tdd-diag-22.svg)

| Error Condition | Detected At | HTTP Response | Notes |
|---|---|---|---|
| `url_decode()` returns -1 (malformed `%XX`) | Stage 1 | 400 Bad Request | e.g., `%GG`, `%2` |
| `url_decode()` returns -2 (null byte `%00`) | Stage 1 | 400 Bad Request | Security: null truncates path strings |
| `url_decode()` returns -3 (buffer full) | Stage 1 | 400 Bad Request | Cannot happen if `dst_size = MAX_PATH_LEN` |
| `decoded_path` does not start with `/` | Stage 2 | 403 Forbidden | Malformed path post-decode |
| Path concatenation overflow | Stage 2 | 403 Forbidden | `doc_root` + `decoded_path` > `PATH_MAX` |
| `realpath()` → `ENOENT` or `ENOTDIR` | Stage 3 | 404 Not Found | File/directory does not exist |
| `realpath()` → `EACCES` | Stage 3 | 403 Forbidden | Insufficient read permission |
| `realpath()` → `ELOOP` | Stage 3 | 403 Forbidden | Circular symlink |
| `realpath()` → `ENAMETOOLONG` | Stage 3 | 403 Forbidden | Resolved path too long |
| Canonical path does not start with `doc_root` | Stage 4 | 403 Forbidden | Never 404 — no info leak |
| `canonical[root_len]` is not `/` or `\0` | Stage 4 | 403 Forbidden | Sibling directory prefix match |
| `stat()` fails on canonical path | Stage 4b | 404 Not Found | Race: file deleted between `realpath` and `stat` |
| `S_ISDIR` and `snprintf` for index path overflows | Stage 4b | 403 Forbidden | Path too long to append `/index.html` |
| `realpath()` on index path fails | Stage 4b | 403 Forbidden | `index.html` absent → 403, not 404 |
| Index realpath escapes `doc_root` | Stage 4b | 403 Forbidden | `index.html` is a symlink pointing outside |
| `stat()` on index path fails | Stage 4b | 404 Not Found | index.html deleted between realpath and stat |
| `If-Modified-Since` parse failure | Stage 5 | (ignored, serve 200) | Invalid date format → treat as absent |
| `st.st_mtime <= ims` (file unchanged) | Stage 5 | 304 Not Modified | No body; headers include Content-Length |
| `open()` → `EACCES` | Stage 6 | 403 Forbidden | File exists but unreadable |
| `open()` → any other error | Stage 6 | 404 Not Found | Includes `ENOENT` (race) |
| `snprintf` for response header overflows | Stage 7 | 500 Internal Server Error | `file_fd` closed before sending 500 |
| `send_all()` for headers fails (client disconnected) | Stage 7 | (silent) | `file_fd` closed; no further writes possible |
| `read()` returns -1 mid-transfer | Stage 8 | (silent — headers sent) | Close `file_fd` and connection |
| `send_all()` returns -1 mid-transfer | Stage 8 | (silent) | Break loop; close `file_fd`; caller closes FD |
**Security invariant on 403 vs 404**: a request that resolves outside `doc_root` after `realpath()` must always return 403 — even if the target path does not exist and the filesystem would return `ENOENT` for the escaped path. Returning 404 in that case would reveal that `/etc/shadow_backup` does not exist, which is information about the filesystem outside the document root that the client should not have.
---
## 7. Implementation Sequence with Checkpoints
### Phase 1 — `url_decode()`: Percent-Decode with Malformed-Encoding and Null-Byte Rejection (0.5–1 h)
Create `file_server.h` with all constants, `mime_entry_t`, and function declarations. Create `file_server.c` with stub bodies. Implement `url_decode()` per Section 5.1.
```c
/* Inline unit test — add to main() temporarily */
{
    char out[256];
    assert(url_decode("/index.html", out, sizeof(out)) == 0);
    assert(strcmp(out, "/index.html") == 0);
    assert(url_decode("/%2e%2e%2fetc", out, sizeof(out)) == 0);
    assert(strcmp(out, "/../etc") == 0);
    assert(url_decode("/%00secret", out, sizeof(out)) == -2);
    assert(url_decode("/%GG", out, sizeof(out)) == -1);
    assert(url_decode("/%2", out, sizeof(out)) == -1);
    assert(url_decode("/file+name", out, sizeof(out)) == 0);
    assert(strcmp(out, "/file+name") == 0);
}
```
**Checkpoint 1**: compile with `gcc -Wall -Wextra -std=c11 -o server server.c http_parse.c file_server.c`. All six assertions pass. Zero compiler warnings.
---
### Phase 2 — `build_full_path()`: Root + Decoded Path Concatenation (0.5 h)
Implement `build_full_path()` per Section 5.3.
**Checkpoint 2**:
```c
{
    char out[PATH_MAX];
    assert(build_full_path("/var/www/html", "/index.html", out, sizeof(out)) == 0);
    assert(strcmp(out, "/var/www/html/index.html") == 0);
    /* trailing slash on doc_root stripped */
    assert(build_full_path("/var/www/html/", "/page.html", out, sizeof(out)) == 0);
    assert(strcmp(out, "/var/www/html/page.html") == 0);
    /* missing leading slash on decoded_path */
    assert(build_full_path("/var/www", "noSlash", out, sizeof(out)) == -1);
}
```
---
### Phase 3 — `realpath()` Security Stage + Prefix+Separator Containment Check (0.5–1 h)
Create a minimal `serve_file()` stub that executes Stages 1–4 and sends either 403 or a placeholder 200. Use this to test the security pipeline in isolation.
Set up a test document root:
```bash
mkdir -p /tmp/webroot/subdir
echo '<h1>Test</h1>' > /tmp/webroot/index.html
echo 'body {}' > /tmp/webroot/subdir/style.css
```
**Checkpoint 3** — traversal attacks return 403:
```bash
./server 8080 /tmp/webroot &
# URL-encoded traversal
curl -o /dev/null -w "%{http_code}" "http://localhost:8080/%2e%2e%2f%2e%2e%2fetc/passwd"
# Expected: 403
# Classic traversal
curl -o /dev/null -w "%{http_code}" "http://localhost:8080/../../etc/passwd"
# Expected: 403
# Symlink outside document root
ln -s /etc /tmp/webroot/escape
curl -o /dev/null -w "%{http_code}" "http://localhost:8080/escape/passwd"
# Expected: 403
rm /tmp/webroot/escape
# Sibling directory prefix attack
mkdir -p /tmp/webroot2
echo secret > /tmp/webroot2/secret.txt
curl -o /dev/null -w "%{http_code}" "http://localhost:8080/../webroot2/secret.txt"
# Expected: 403 (after realpath resolves to /tmp/webroot2/secret.txt)
rmdir /tmp/webroot2
```
---
### Phase 4 — Directory Detection → index.html Append → Re-realpath → Re-check (0.5 h)
Add the `S_ISDIR` branch (Stage 4b) to `serve_file()`.
**Checkpoint 4**:
```bash
# Directory path serves index.html
curl -o /dev/null -w "%{http_code}" "http://localhost:8080/"
# Expected: 200
# Directory with no index.html returns 403 (not 404)
mkdir /tmp/webroot/empty_dir
curl -o /dev/null -w "%{http_code}" "http://localhost:8080/empty_dir/"
# Expected: 403
rmdir /tmp/webroot/empty_dir
# Symlinked index.html escaping doc_root returns 403
mkdir /tmp/secret_dir && echo "secret" > /tmp/secret_dir/data.txt
ln -s /tmp/secret_dir/data.txt /tmp/webroot/subdir/index.html
curl -o /dev/null -w "%{http_code}" "http://localhost:8080/subdir/"
# Expected: 403
rm /tmp/webroot/subdir/index.html
rm -rf /tmp/secret_dir
```
---
### Phase 5 — `get_mime_type()` Lookup Table with strrchr and strcasecmp (0.5 h)
Define the full `MIME_TABLE` in `file_server.c`. Implement `get_mime_type()` per Section 5.4.
**Checkpoint 5**:
```c
assert(strcmp(get_mime_type("/path/to/file.html"), "text/html; charset=utf-8") == 0);
assert(strcmp(get_mime_type("/path/to/file.HTML"), "text/html; charset=utf-8") == 0);  /* case-insensitive */
assert(strcmp(get_mime_type("/path/to/file.png"),  "image/png") == 0);
assert(strcmp(get_mime_type("/path/to/noext"),     "application/octet-stream") == 0);
assert(strcmp(get_mime_type("/path/to/file."),     "application/octet-stream") == 0);  /* trailing dot */
assert(strcmp(get_mime_type("/path/to/archive.tar.gz"), "application/octet-stream") == 0);  /* unknown .gz */
```
---
### Phase 6 — stat() for Content-Length and Last-Modified; strftime HTTP-date Formatting (0.5 h)
Add Stage 5 (excluding the 304 check) and Stage 7 (header building) to `serve_file()`. At this point, `serve_file()` sends correct 200 headers but no body yet.
**Checkpoint 6**:
```bash
# Verify all three mandatory headers are present
curl -v "http://localhost:8080/index.html" 2>&1 | grep -E "Content-Type|Content-Length|Last-Modified"
# Expected (example):
# Content-Type: text/html; charset=utf-8
# Content-Length: 14
# Last-Modified: Mon, 01 Jan 2024 12:00:00 GMT
# Verify Content-Length matches actual file size
stat --printf="%s\n" /tmp/webroot/index.html
# Must match the Content-Length header value
```
---
### Phase 7 — parse_http_date() with strptime+timegm; 304 Response Path (0.5–1 h)
Implement `parse_http_date()` per Section 5.5. Add the `If-Modified-Since` check (Stage 5) to `serve_file()`. Implement `send_304()`.
**Checkpoint 7**:
```bash
# First request: capture Last-Modified
LAST_MOD=$(curl -sI "http://localhost:8080/index.html" | grep -i "last-modified" | sed 's/Last-Modified: //' | tr -d '\r')
echo "Last-Modified: $LAST_MOD"
# Second request with exact timestamp: expect 304
STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "If-Modified-Since: $LAST_MOD" \
  "http://localhost:8080/index.html")
echo "Status: $STATUS"   # Must be 304
# Request with old timestamp (1970): expect 200
STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "If-Modified-Since: Thu, 01 Jan 1970 00:00:01 GMT" \
  "http://localhost:8080/index.html")
echo "Status: $STATUS"   # Must be 200
# 304 body must be empty
BODY_BYTES=$(curl -s -o /dev/null -w "%{size_download}" \
  -H "If-Modified-Since: $LAST_MOD" \
  "http://localhost:8080/index.html")
echo "Body bytes: $BODY_BYTES"   # Must be 0
# Invalid If-Modified-Since format: serve 200 (ignore header)
STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "If-Modified-Since: not-a-date" \
  "http://localhost:8080/index.html")
echo "Status: $STATUS"   # Must be 200
```
---
### Phase 8 — open()+read()+send_all() File Streaming Loop with 64KB Buffer; send_body Flag for HEAD (0.5 h)
Add Stages 6 and 8 to `serve_file()`. Implement the read-send loop.
**Checkpoint 8**:
```bash
# Serve a binary file and verify byte-for-byte integrity
cp /bin/ls /tmp/webroot/testbin
curl -s "http://localhost:8080/testbin" -o /tmp/downloaded_bin
md5sum /bin/ls /tmp/downloaded_bin
# Both MD5 sums must be identical
# HEAD returns no body but correct Content-Length
CL_HEAD=$(curl -sI "http://localhost:8080/index.html" | grep -i "content-length" | awk '{print $2}' | tr -d '\r')
CL_GET=$(curl -s "http://localhost:8080/index.html" | wc -c | tr -d ' ')
echo "HEAD Content-Length: $CL_HEAD  GET body bytes: $CL_GET"
# Must be equal
# HEAD body download size is 0
HEAD_DL=$(curl -s --head "http://localhost:8080/index.html" -o /dev/null -w "%{size_download}")
echo "HEAD download bytes: $HEAD_DL"  # Must be 0
rm /tmp/webroot/testbin
```
---
### Phase 9 — Error Response Helpers; All FD Close Paths Verified (0.5 h)
Implement `send_404()`, `send_403()`, `send_500()` per Section 5.7. Verify all error response `Content-Length` values with `_Static_assert`. Audit every return statement in `serve_file()` to confirm `file_fd` is closed before any return that follows an `open()` call.
**Checkpoint 9** — `valgrind` FD audit:
```bash
valgrind --track-fds=yes ./server 8080 /tmp/webroot &
VG_PID=$!
sleep 0.3
# Run 200 requests including errors
for i in $(seq 100); do curl -s "http://localhost:8080/index.html" > /dev/null; done
for i in $(seq 50); do curl -s "http://localhost:8080/nonexistent" > /dev/null; done
for i in $(seq 50); do curl -s "http://localhost:8080/%2e%2e%2fetc/passwd" > /dev/null; done
kill $VG_PID
# valgrind output must show 0 file descriptors still open (beyond stdin/stdout/stderr/server_fd)
```
---
### Phase 10 — Full Test Suite (1–2 h)
Run `test_file_server.sh` per Section 8. All tests must pass.
---
## 8. Test Specification

![Directory Index Auto-Serve Logic: State Transitions](./diagrams/tdd-diag-23.svg)

### 8.1 `url_decode()` Tests
**U1.1 — Plain path (no encoding)**
```
Input:  "/index.html"
Expect: return 0, dst="/index.html"
```
**U1.2 — Single percent-encoded character**
```
Input:  "/file%20name.html"
Expect: return 0, dst="/file name.html"
```
**U1.3 — Encoded path separator (traversal attempt)**
```
Input:  "/%2e%2e%2fetc%2fpasswd"
Expect: return 0, dst="/../etc/passwd"
Note:   decode succeeds; realpath() later rejects the traversal
```
**U1.4 — Null byte rejection**
```
Input:  "/file%00.txt"
Expect: return -2
```
**U1.5 — Malformed hex: non-hex digit**
```
Input:  "/%GG"
Expect: return -1
```
**U1.6 — Malformed hex: truncated sequence**
```
Input:  "/%2"
Expect: return -1
```
**U1.7 — Plus sign is literal in path**
```
Input:  "/file+name.txt"
Expect: return 0, dst="/file+name.txt"
```
**U1.8 — Uppercase hex digits**
```
Input:  "/%2E%2E%2F"
Expect: return 0, dst="/../"
```
**U1.9 — Percent at end of string**
```
Input:  "/file%"
Expect: return -1
```
**U1.10 — Empty path**
```
Input:  "/"
Expect: return 0, dst="/"
```
---
### 8.2 Security Pipeline Tests
**S1 — URL-encoded traversal**
```bash
curl -o /dev/null -w "%{http_code}" "http://localhost:8080/%2e%2e%2fetc/passwd"
# Expected: 403
```
**S2 — Double-encoded traversal** (some clients double-encode)
```bash
curl -o /dev/null -w "%{http_code}" "http://localhost:8080/%252e%252e%252fetc"
# Expected: 404 (the literal string "%2e" is a path that does not exist on disk)
# Note: double-encoding is not a security bypass here because url_decode() only decodes once
```
**S3 — Classic `../` traversal**
```bash
curl -o /dev/null -w "%{http_code}" "http://localhost:8080/../../etc/passwd"
# Expected: 403
```
**S4 — Symlink to outside document root**
```bash
ln -s /etc /tmp/webroot/escape_sym
curl -o /dev/null -w "%{http_code}" "http://localhost:8080/escape_sym/passwd"
# Expected: 403
rm /tmp/webroot/escape_sym
```
**S5 — Sibling directory prefix attack**
```bash
mkdir /tmp/webroot_sibling && echo "secret" > /tmp/webroot_sibling/data
# Request a path that prefix-matches /tmp/webroot but resolves to sibling
curl -o /dev/null -w "%{http_code}" "http://localhost:8080/../webroot_sibling/data"
# Expected: 403
rm -rf /tmp/webroot_sibling
```
**S6 — Null byte in path**
```bash
python3 -c "
import socket
s = socket.socket()
s.connect(('localhost', 8080))
s.send(b'GET /file\x00.txt HTTP/1.1\r\nHost: localhost\r\n\r\n')
print(s.recv(64).decode()[:30])
s.close()
"
# Expected: HTTP/1.1 400 Bad Request
```
**S7 — 403 for path outside root even when target doesn't exist**
```bash
# /tmp/nonexistent_outside_root doesn't exist
curl -o /dev/null -w "%{http_code}" "http://localhost:8080/../../nonexistent_outside_root"
# Expected: 403 (not 404 — no info leak about external filesystem)
```
---
### 8.3 File Serving Tests
**F1 — HTML file served with correct MIME and body**
```bash
RESP=$(curl -sv "http://localhost:8080/index.html" 2>&1)
echo "$RESP" | grep "Content-Type: text/html; charset=utf-8"  # must match
echo "$RESP" | grep "200 OK"
BODY=$(curl -s "http://localhost:8080/index.html")
echo "$BODY" | grep "<h1>"  # body content present
```
**F2 — Binary file integrity (md5sum must match)**
```bash
cp /bin/date /tmp/webroot/testbin
curl -s "http://localhost:8080/testbin" -o /tmp/served_bin
md5sum /bin/date /tmp/served_bin
# Both sums must be identical
ORIG_SIZE=$(stat --printf="%s" /bin/date)
SERVED_SIZE=$(stat --printf="%s" /tmp/served_bin)
[ "$ORIG_SIZE" = "$SERVED_SIZE" ] && echo "PASS: sizes match" || echo "FAIL"
rm /tmp/webroot/testbin /tmp/served_bin
```
**F3 — Content-Length header equals actual body bytes**
```bash
CL=$(curl -sI "http://localhost:8080/index.html" | grep -i "content-length" | awk '{print $2}' | tr -d '\r')
ACTUAL=$(curl -s "http://localhost:8080/index.html" | wc -c | tr -d ' ')
[ "$CL" = "$ACTUAL" ] && echo "PASS: CL=$CL" || echo "FAIL: CL=$CL actual=$ACTUAL"
```
**F4 — 404 for nonexistent file**
```bash
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:8080/does_not_exist.html")
[ "$STATUS" = "404" ] && echo "PASS" || echo "FAIL: $STATUS"
# 404 body must be non-empty HTML
BODY=$(curl -s "http://localhost:8080/does_not_exist.html")
echo "$BODY" | grep -q "404" && echo "PASS: body mentions 404" || echo "FAIL"
```
**F5 — Directory path serves index.html**
```bash
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:8080/")
[ "$STATUS" = "200" ] && echo "PASS" || echo "FAIL: $STATUS"
BODY=$(curl -s "http://localhost:8080/")
# Should be same content as /index.html
DIRECT=$(curl -s "http://localhost:8080/index.html")
[ "$BODY" = "$DIRECT" ] && echo "PASS: / == /index.html" || echo "FAIL"
```
**F6 — Directory with no index.html returns 403**
```bash
mkdir /tmp/webroot/empty_dir
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:8080/empty_dir/")
[ "$STATUS" = "403" ] && echo "PASS" || echo "FAIL: $STATUS"
rmdir /tmp/webroot/empty_dir
```
---
### 8.4 MIME Type Tests
**M1 — Correct MIME types for each extension**
```bash
for EXT_MIME in \
    "html:text/html; charset=utf-8" \
    "css:text/css; charset=utf-8" \
    "js:application/javascript" \
    "json:application/json" \
    "png:image/png" \
    "jpg:image/jpeg" \
    "svg:image/svg+xml" \
    "pdf:application/pdf"; do
  EXT="${EXT_MIME%%:*}"
  EXPECTED="${EXT_MIME#*:}"
  echo "test content" > "/tmp/webroot/test.$EXT"
  CT=$(curl -sI "http://localhost:8080/test.$EXT" | grep -i "content-type" | sed 's/Content-Type: //' | tr -d '\r')
  rm "/tmp/webroot/test.$EXT"
  if [ "$CT" = "$EXPECTED" ]; then
    echo "PASS: .$EXT → $CT"
  else
    echo "FAIL: .$EXT expected '$EXPECTED' got '$CT'"
  fi
done
```
**M2 — Unknown extension returns application/octet-stream**
```bash
echo "data" > /tmp/webroot/file.xyz123
CT=$(curl -sI "http://localhost:8080/file.xyz123" | grep -i "content-type" | sed 's/Content-Type: //' | tr -d '\r')
[ "$CT" = "application/octet-stream" ] && echo "PASS" || echo "FAIL: $CT"
rm /tmp/webroot/file.xyz123
```
**M3 — Case-insensitive extension matching**
```bash
echo "<h1>upper</h1>" > /tmp/webroot/upper.HTML
CT=$(curl -sI "http://localhost:8080/upper.HTML" | grep -i "content-type" | sed 's/Content-Type: //' | tr -d '\r')
[ "$CT" = "text/html; charset=utf-8" ] && echo "PASS" || echo "FAIL: $CT"
rm /tmp/webroot/upper.HTML
```
---
### 8.5 Conditional Request Tests
**C1 — 304 returned when file unchanged**
```bash
LM=$(curl -sI "http://localhost:8080/index.html" | grep -i "last-modified" | cut -d' ' -f2- | tr -d '\r')
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -H "If-Modified-Since: $LM" "http://localhost:8080/index.html")
[ "$STATUS" = "304" ] && echo "PASS" || echo "FAIL: $STATUS"
```
**C2 — 304 has no body**
```bash
LM=$(curl -sI "http://localhost:8080/index.html" | grep -i "last-modified" | cut -d' ' -f2- | tr -d '\r')
DL=$(curl -s -o /dev/null -w "%{size_download}" -H "If-Modified-Since: $LM" "http://localhost:8080/index.html")
[ "$DL" = "0" ] && echo "PASS: no body" || echo "FAIL: got $DL bytes"
```
**C3 — 304 includes Content-Type, Content-Length, Last-Modified**
```bash
LM=$(curl -sI "http://localhost:8080/index.html" | grep -i "last-modified" | cut -d' ' -f2- | tr -d '\r')
HEADERS=$(curl -sI -H "If-Modified-Since: $LM" "http://localhost:8080/index.html")
echo "$HEADERS" | grep -q "Content-Type" && echo "PASS: CT in 304" || echo "FAIL: no CT"
echo "$HEADERS" | grep -q "Content-Length" && echo "PASS: CL in 304" || echo "FAIL: no CL"
echo "$HEADERS" | grep -q "Last-Modified" && echo "PASS: LM in 304" || echo "FAIL: no LM"
```
**C4 — 200 returned for old If-Modified-Since**
```bash
STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "If-Modified-Since: Thu, 01 Jan 1970 00:00:01 GMT" \
  "http://localhost:8080/index.html")
[ "$STATUS" = "200" ] && echo "PASS" || echo "FAIL: $STATUS"
```
**C5 — Invalid date format ignored (serves 200)**
```bash
STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "If-Modified-Since: yesterday" \
  "http://localhost:8080/index.html")
[ "$STATUS" = "200" ] && echo "PASS" || echo "FAIL: $STATUS"
```
**C6 — HEAD with If-Modified-Since returns 304 with no body**
```bash
LM=$(curl -sI "http://localhost:8080/index.html" | grep -i "last-modified" | cut -d' ' -f2- | tr -d '\r')
RESULT=$(curl -s --head -H "If-Modified-Since: $LM" "http://localhost:8080/index.html" -w "\n%{http_code}\n%{size_download}")
echo "$RESULT" | grep "304" && echo "PASS: 304" || echo "FAIL"
DL=$(echo "$RESULT" | tail -1)
[ "$DL" = "0" ] && echo "PASS: no body" || echo "FAIL: $DL"
```
---
### 8.6 FD Leak and Correctness Tests
**L1 — No FD leak after 10,000 requests including errors**
```bash
SERVER_PID=$(pgrep -n server)
BASELINE=$(ls /proc/$SERVER_PID/fd | wc -l)
ab -n 5000 -c 10 "http://localhost:8080/index.html" > /dev/null 2>&1
ab -n 2500 -c 10 "http://localhost:8080/does_not_exist" > /dev/null 2>&1
ab -n 2500 -c 10 "http://localhost:8080/%2e%2e%2fetc" > /dev/null 2>&1
AFTER=$(ls /proc/$SERVER_PID/fd | wc -l)
[ "$AFTER" = "$BASELINE" ] && echo "PASS: no FD leak" || echo "FAIL: $BASELINE → $AFTER"
```
---
### 8.7 Complete `test_file_server.sh`

![MIME Type Lookup Table and Extension Dispatch](./diagrams/tdd-diag-24.svg)

```bash
#!/bin/bash
set -e
PORT=18082
DOC_ROOT=$(mktemp -d)
trap "rm -rf $DOC_ROOT; kill $SERVER_PID 2>/dev/null; exit" EXIT INT TERM
# Set up test fixtures
mkdir -p "$DOC_ROOT/sub"
echo '<html><body><h1>Index</h1></body></html>'  > "$DOC_ROOT/index.html"
echo 'body { color: red; }'                       > "$DOC_ROOT/sub/style.css"
echo '{"key":"value"}'                            > "$DOC_ROOT/data.json"
dd if=/dev/urandom of="$DOC_ROOT/binary.bin" bs=1024 count=64 2>/dev/null
./server $PORT $DOC_ROOT &
SERVER_PID=$!
sleep 0.3
BASE="http://localhost:$PORT"
PASS=0; FAIL=0
check() {
  local desc="$1" expected="$2" actual="$3"
  if [ "$actual" = "$expected" ]; then
    echo "PASS: $desc"; ((PASS++))
  else
    echo "FAIL: $desc (expected '$expected' got '$actual')"; ((FAIL++))
  fi
}
# Basic serving
check "GET index.html → 200" "200" "$(curl -s -o /dev/null -w '%{http_code}' $BASE/)"
check "GET CSS → 200" "200" "$(curl -s -o /dev/null -w '%{http_code}' $BASE/sub/style.css)"
check "GET JSON → 200" "200" "$(curl -s -o /dev/null -w '%{http_code}' $BASE/data.json)"
check "GET 404" "404" "$(curl -s -o /dev/null -w '%{http_code}' $BASE/nosuchfile.txt)"
# MIME types
check "HTML MIME" "text/html; charset=utf-8" \
  "$(curl -sI $BASE/index.html | grep -i content-type | awk '{print $2}' | tr -d '\r')"
check "CSS MIME" "text/css; charset=utf-8" \
  "$(curl -sI $BASE/sub/style.css | grep -i content-type | awk '{print $2}' | tr -d '\r')"
check "JSON MIME" "application/json" \
  "$(curl -sI $BASE/data.json | grep -i content-type | awk '{print $2}' | tr -d '\r')"
# Security
check "URL-encoded traversal → 403" "403" \
  "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/%2e%2e%2fetc/passwd")"
check "Classic traversal → 403" "403" \
  "$(curl -s -o /dev/null -w '%{http_code}' "$BASE/../../etc/passwd")"
# Conditional requests
LM=$(curl -sI $BASE/index.html | grep -i last-modified | cut -d' ' -f2- | tr -d '\r')
check "304 when unchanged" "304" \
  "$(curl -s -o /dev/null -w '%{http_code}' -H "If-Modified-Since: $LM" $BASE/index.html)"
check "304 no body" "0" \
  "$(curl -s -o /dev/null -w '%{size_download}' -H "If-Modified-Since: $LM" $BASE/index.html)"
check "200 with old date" "200" \
  "$(curl -s -o /dev/null -w '%{http_code}' -H "If-Modified-Since: Thu, 01 Jan 1970 00:00:01 GMT" $BASE/index.html)"
# HEAD
check "HEAD no body" "0" \
  "$(curl -s --head $BASE/index.html -o /dev/null -w '%{size_download}')"
HEAD_CL=$(curl -sI $BASE/index.html | grep -i content-length | awk '{print $2}' | tr -d '\r')
GET_SZ=$(curl -s $BASE/index.html | wc -c | tr -d ' ')
check "HEAD CL == GET body" "$GET_SZ" "$HEAD_CL"
# Binary integrity
curl -s $BASE/binary.bin -o /tmp/test_served_bin
ORIG_MD5=$(md5sum "$DOC_ROOT/binary.bin" | awk '{print $1}')
SERV_MD5=$(md5sum /tmp/test_served_bin    | awk '{print $1}')
check "Binary MD5 integrity" "$ORIG_MD5" "$SERV_MD5"
rm -f /tmp/test_served_bin
# FD leak
SERVER_PID_CHECK=$(pgrep -n server)
BASELINE_FD=$(ls /proc/$SERVER_PID_CHECK/fd | wc -l)
for i in $(seq 500); do curl -s $BASE/index.html > /dev/null; done
for i in $(seq 100); do curl -s "$BASE/%2e%2e%2fetc" > /dev/null; done
AFTER_FD=$(ls /proc/$SERVER_PID_CHECK/fd | wc -l)
check "No FD leak" "$BASELINE_FD" "$AFTER_FD"
echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" = "0" ]
```
---
## 9. Performance Targets

![If-Modified-Since / 304 Not Modified: Complete Round-Trip](./diagrams/tdd-diag-25.svg)

| Operation | Target | Measurement Method |
|---|---|---|
| `realpath()` for 3-component path, hot dcache | < 50 µs | `perf stat -e cache-misses ./server`; time `realpath()` calls with `clock_gettime(CLOCK_MONOTONIC)` around Stage 3 |
| `stat()` for hot inode (in kernel page cache) | < 10 µs | Repeat `stat()` on same file in a tight loop; measure with `CLOCK_MONOTONIC` |
| `read()+send_all()` for 64 KB file, loopback | < 500 µs | `ab -n 1000 -c 1` with a 64 KB file; "Time per request" (mean) |
| Throughput for 1 MB file, loopback | ≥ 1 GB/s | `ab -n 100 -c 1` with 1 MB file; compute bytes_transferred / total_time |
| MIME lookup (18-entry linear scan) | < 200 ns | `clock_gettime` wrapper around 1 million `get_mime_type()` calls; total / 1M |
| `url_decode()` for 100-char path | < 1 µs | 1 million calls; total / 1M |
| `parse_http_date()` for RFC 7231 format | < 2 µs | 100K calls with `strptime` target |
| 304 response latency vs 200 (no file read) | ≥ 50% faster than 200 | `ab` comparing a hot 304 path vs a hot 200 path for the same file |
| `serve_file()` heap allocation | 0 bytes | `valgrind --tool=massif`: 0 malloc calls per request |
| FD count stable after 10K requests | Baseline ± 0 | `ls /proc/$PID/fd \| wc -l` before and after `ab -n 10000` |
**Benchmark commands**:
```bash
# Throughput for 1 MB file
dd if=/dev/urandom of=/tmp/webroot/bigfile.bin bs=1024 count=1024
ab -n 500 -c 1 "http://localhost:8080/bigfile.bin" 2>&1 | grep -E "Transfer rate|Requests per"
# MIME lookup microbenchmark (add to separate bench binary)
# parse_http_date microbenchmark
# Both follow the same clock_gettime pattern as M2's parse_bench.c
```
---
## 10. Threat Model

![File Serve Data Flow: open() → stat() → headers → read() → send() Loop](./diagrams/tdd-diag-26.svg)

| Threat | Attack Vector | Mitigation |
|---|---|---|
| Directory traversal via `../` | `req->path` contains `/../` sequences | `realpath()` resolves all `..` before containment check |
| Directory traversal via URL encoding | `%2e%2e%2f` in `req->path` | `url_decode()` applied before path concatenation; decoded path fed to `realpath()` |
| Symlink escape | Symlink inside doc_root points outside | `realpath()` follows all symlinks before containment check |
| Double-encoding bypass | `%252e` → `%2e` → `.` | `url_decode()` decodes once only; the `%` in `%252e` is decoded to `%`; the resulting `%2e` is a literal two-byte string that `realpath()` handles as a filename containing `%` |
| Null byte injection | `%00` in path | `url_decode()` returns -2 on decoded null byte; → 400 |
| Sibling directory prefix match | `/var/www/html2/` matches `/var/www/html` prefix | `canonical[root_len] == '/' OR '\0'` check in Stage 4 |
| index.html symlink escape | `index.html` is a symlink to `/etc/passwd` | `realpath()` re-run on index path; containment re-checked |
| Information leak via 404 on escaped paths | 404 reveals file existence outside doc_root | All escaped paths return 403 regardless of target existence |
| Large file DoS (no timeout) | Client opens connection, requests 10 GB file | M3 does not implement timeouts; M4 handles this via keep-alive idle timeout |
| Disk read during file serving (mid-transfer error) | Storage failure between headers sent and body complete | `read()` error is silent (connection closed); no way to send 500 after headers committed |
---
## 11. Makefile Update
```makefile
CC      = gcc
CFLAGS  = -Wall -Wextra -Wpedantic -std=c11 -O2 -g \
          -Wframe-larger-than=131072 \
          -D_GNU_SOURCE
TARGET  = server
all: $(TARGET)
$(TARGET): server.c http_parse.c file_server.c
	$(CC) $(CFLAGS) -o $@ $^
clean:
	rm -f $(TARGET)
test: $(TARGET)
	bash test_file_server.sh
.PHONY: all clean test
```
`-D_GNU_SOURCE` exposes `timegm()` and `strptime()` on Linux. Without this define, those functions may not be declared in `<time.h>`. `-Wframe-larger-than=131072` allows up to 128 KB stack frames (to accommodate `read_buf[65536]` + other locals ≈ 91 KB) while still catching runaway allocations.
---

![serve_file() Complete State Machine: All Exit Paths](./diagrams/tdd-diag-27.svg)


![Hardware Soul: Page Cache and Read Latency Tiers](./diagrams/tdd-diag-28.svg)

<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: http-server-basic-m4 -->
# Technical Design Document: Concurrent Connections
## Module `http-server-basic-m4`
---
## 1. Module Charter
This module adds true concurrent client handling to the HTTP server by layering two concurrency models — a simple thread-per-connection model for initial implementation, then a bounded thread pool with a circular work queue as the production model — on top of the socket infrastructure (M1), HTTP parser (M2), and file server (M3). It implements HTTP/1.1 persistent connections (keep-alive) with per-read idle timeouts enforced via `select()`, a graceful shutdown mechanism driven by an `atomic_int` flag and `pthread_join` draining, and mutex-protected shared state for a connection counter and access log.
This module does **not** implement `epoll`, `kqueue`, or any non-blocking I/O. It does not implement HTTP/2 multiplexing, chunked transfer encoding, TLS, or connection-level flow control. It does not implement epoll-based event loops, `io_uring`, or coroutines. It does not replace the socket lifecycle (M1), the HTTP parser (M2), or the file server (M3) — it wraps them. All file descriptor creation and closure for client connections remains the responsibility of this module's worker threads (the accept loop opens `client_fd`; the worker closes it after the keep-alive loop exits).
**Upstream dependency**: M3's `serve_file()` is the per-request handler. M2's `http_parse_request()` and M1's `read_request()` are called within the keep-alive loop. The `send_all()` helper from M1 is used for 503 responses. All upstream functions are stateless and re-entrant.
**Downstream dependency**: `main()` in `server.c` integrates everything; no further milestones depend on this module.
**Invariants that must always hold after this module**:
- Every `client_fd` accepted by `accept()` is `close()`d exactly once, by the worker thread that handled it, after the keep-alive loop exits.
- The thread pool always contains exactly `pool->thread_count` live threads from `thread_pool_init()` until `thread_pool_shutdown()` returns.
- The `pool->count` field always equals the actual number of items in the circular buffer: `(pool->tail - pool->head + WORK_QUEUE_CAPACITY) % WORK_QUEUE_CAPACITY` when the lock is held.
- `pool->shutdown == 1` implies no new items will ever be enqueued; workers may drain remaining items then exit.
- `server_shutdown` is written only from `signal_handler()` and read from the accept loop and worker threads; it is `atomic_int` to guarantee cross-thread visibility without a mutex.
- No function in this module calls `exit()` except on unrecoverable startup failures in `thread_pool_init()`.
---
## 2. File Structure
Create files in this order:
```
http-server/
├── 1  thread_pool.h          # work_item_t, thread_pool_t, function declarations
├── 2  thread_pool.c          # full thread pool implementation
├── 3  connection.h           # connection_t, handle_client_keep_alive() declaration
├── 4  connection.c           # keep-alive loop, read_request_with_timeout(), should_keep_alive()
├── 5  stats.h                # server_stats_t, access_log_t declarations
├── 6  stats.c                # stats and access log implementation
├── 7  server.c               # updated main(): signal handling, pool init, accept loop, shutdown
├── 8  http_parse.h           # unchanged from M2
├── 9  http_parse.c           # unchanged from M2
├── 10 file_server.h          # unchanged from M3
├── 11 file_server.c          # unchanged from M3
├── 12 Makefile               # updated: -lpthread, -fsanitize=thread target
└── 13 test_concurrent.sh     # automated test script (Phase 9)
```
`thread_pool.h/c` and `connection.h/c` and `stats.h/c` are all new files. `server.c` is substantially rewritten. The M1–M3 source files are linked unchanged.
---
## 3. Complete Data Model
### 3.1 Constants
```c
/* thread_pool.h */
#define THREAD_POOL_DEFAULT_SIZE    16
#define WORK_QUEUE_CAPACITY         1024    /* power of 2; circular buffer slots */
#define KEEPALIVE_TIMEOUT_SECONDS   30      /* idle timeout per read attempt      */
#define MAX_REQUESTS_PER_CONNECTION 100     /* keep-alive loop hard cap           */
#define ACCESS_LOG_PATH             "access.log"
```
`WORK_QUEUE_CAPACITY = 1024`: power-of-two chosen so that the modulo operation `(idx + 1) % WORK_QUEUE_CAPACITY` can be compiled to a bitmask `(idx + 1) & (WORK_QUEUE_CAPACITY - 1)` by the compiler with `-O2`. At 16 worker threads, the queue provides 64 slots per worker — sufficient buffer for burst traffic spikes without consuming significant memory.
`MAX_REQUESTS_PER_CONNECTION = 100`: prevents a single slow client from monopolizing a worker thread indefinitely. After 100 requests on one connection, the server closes it and the worker is free for other connections.
### 3.2 `work_item_t`
```c
/* thread_pool.h */
typedef struct {
    int  client_fd;                        /* connected socket FD to serve     */
    char client_ip[INET_ADDRSTRLEN];       /* dotted-decimal string, e.g. "192.168.1.1" */
    int  client_port;                      /* host-byte-order port number      */
} work_item_t;
```
**Memory layout** (64-bit Linux, `INET_ADDRSTRLEN = 16`):
```
Offset   Field        Type              Size     Notes
0x00     client_fd    int               4 B      index into kernel FD table
0x04     (padding)    —                 0 B      client_ip[0] at offset 0x04
0x04     client_ip    char[16]          16 B     null-terminated, e.g. "127.0.0.1\0"
0x14     (padding)    —                 0 B      int aligned at 4
0x14     client_port  int               4 B      host byte order
0x18     end
Total:                                  24 B     fits in less than 1 cache line
```
`work_item_t` is value-copied into and out of the circular buffer under the pool lock. At 24 bytes, each copy is 3 × 8-byte stores — negligible cost.
### 3.3 `thread_pool_t`
```c
/* thread_pool.h */
typedef struct {
    work_item_t     queue[WORK_QUEUE_CAPACITY]; /* circular buffer: 24 * 1024 = 24576 B */
    int             head;          /* dequeue index: next item to consume     */
    int             tail;          /* enqueue index: next free slot           */
    int             count;         /* current number of items in queue        */
    pthread_mutex_t lock;          /* protects head, tail, count, shutdown    */
    pthread_cond_t  not_empty;     /* signaled when count goes 0 → 1         */
    pthread_cond_t  not_full;      /* signaled when count goes CAPACITY → CAPACITY-1 */
    int             shutdown;      /* 1 = workers must drain then exit        */
    pthread_t       threads[THREAD_POOL_DEFAULT_SIZE]; /* worker thread handles */
    int             thread_count;  /* actual number of threads created        */
} thread_pool_t;
```
**Memory layout** (cache-line annotated, 64 bytes per line):
```
Offset      Field                      Size         Cache lines
0x0000      queue[0..1023]             24576 B      lines 0–383   (384 lines)
0x6000      head                       4 B          line 384
0x6004      tail                       4 B          line 384
0x6008      count                      4 B          line 384
0x600C      (padding to align mutex)   4 B          line 384
0x6010      lock (pthread_mutex_t)     40 B         lines 384–384 (spans into 385)
0x6038      not_empty (pthread_cond_t) 48 B         lines 384–385
0x6068      not_full  (pthread_cond_t) 48 B         lines 385–386
0x6098      shutdown                   4 B          line 386
0x609C      (padding)                  4 B          line 386
0x60A0      threads[0..15]             128 B        lines 386–387 (16 × 8 bytes)
0x6120      thread_count               4 B          line 388
0x6124      end
Total:                                ~24,740 B    (~24 KB)
```
**Cache-line isolation note**: `head`, `tail`, `count`, and `lock` share a cache line (line 384). Every enqueue and dequeue writes to this line. With one producer (accept loop) and N consumers (workers), this line will bounce between the producer core and whichever consumer core holds it. This is acceptable for a mutex-protected queue — the lock serializes access anyway, so there is no benefit to splitting `head` and `tail` onto separate cache lines as one would do for a lock-free queue.
**Why `count` is redundant but present**: `count` can be derived from `(tail - head + CAPACITY) % CAPACITY`, but computing it in the condition check requires two reads and a modulo. Storing `count` directly makes `while (pool->count == 0)` a single read, reducing the critical section.

![thread_pool_t and work_item_t Memory Layout with Cache Lines](./diagrams/tdd-diag-29.svg)

### 3.4 `connection_t`
```c
/* connection.h */
typedef struct {
    int  client_fd;
    char client_ip[INET_ADDRSTRLEN];
    int  client_port;
    const char *doc_root;           /* pointer into main()'s argv or static storage */
} connection_t;
```
Used only in Phase 1 (thread-per-connection model). Heap-allocated by the accept loop, freed by the thread before returning. The thread pool model (Phase 2+) uses `work_item_t` in the queue and passes `doc_root` as a global.
### 3.5 `server_stats_t`
```c
/* stats.h */
typedef struct {
    pthread_mutex_t lock;
    int             active_connections;   /* currently open connections      */
    long long       total_connections;    /* cumulative since server start   */
    long long       total_requests;       /* cumulative requests served      */
    long long       total_bytes_sent;     /* cumulative response bytes       */
} server_stats_t;
```
**Memory layout**:
```
Offset   Field                 Type              Size
0x00     lock                  pthread_mutex_t   40 B
0x28     active_connections    int               4 B
0x2C     (padding)             —                 4 B
0x30     total_connections     long long         8 B
0x38     total_requests        long long         8 B
0x40     total_bytes_sent      long long         8 B
0x48     end
Total:                                           72 B   (2 cache lines)
```
`lock` occupies the first cache line (lines 0–0). The counters occupy the second cache line. When a thread acquires `lock`, both cache lines are pulled into L1. Updates to all four counters under one lock hold cost one cache line load.
### 3.6 `access_log_t`
```c
/* stats.h */
typedef struct {
    pthread_mutex_t lock;
    FILE           *file;   /* opened with fopen(path, "a") at startup */
} access_log_t;
```
A separate mutex from `server_stats_t`. Holding both simultaneously is not required by any code path, so no lock ordering is needed. If a future change requires holding both, lock `server_stats_t.lock` before `access_log_t.lock` (alphabetical by struct name — a memorable rule).

![Circular Work Queue: Enqueue and Dequeue Operations](./diagrams/tdd-diag-30.svg)

### 3.7 Global Variables
```c
/* server.c */
static volatile sig_atomic_t server_shutdown = 0;  /* written by signal handler */
static atomic_int             shutdown_flag   = 0;  /* read by accept loop and workers */
static thread_pool_t          g_pool;
static server_stats_t         g_stats;
static access_log_t           g_log;
static const char            *g_doc_root;           /* set once in main(), never written again */
```
**Two shutdown flags, two different uses**:
- `server_shutdown` (`volatile sig_atomic_t`): written by `signal_handler()`, which may run on any thread. `sig_atomic_t` guarantees the write is atomic at the hardware level (no torn writes). `volatile` prevents the compiler from caching the read in a register. This is the signal-safe write mechanism.
- `shutdown_flag` (`atomic_int`): read by the accept loop via `atomic_load(&shutdown_flag)`. The accept loop copies `server_shutdown` into `shutdown_flag` on each iteration — an explicit synchronization point that converts the `volatile` signal-handler write into a `_Atomic` value with `memory_order_seq_cst` guarantees for the threaded readers.
Alternatively, use only `atomic_int shutdown_flag` with `atomic_store(&shutdown_flag, 1)` directly from the signal handler — this is safe because `atomic_store` is async-signal-safe on Linux (the store compiles to a single instruction). The two-flag approach is shown here because it makes the signal-to-thread boundary explicit. Either approach is correct; choose one and document it.
---
## 4. Interface Contracts
### 4.1 `thread_pool_init()`
```c
/* thread_pool.h */
int thread_pool_init(thread_pool_t *pool, int num_threads);
```
**Parameters**:
- `pool`: pointer to a zero-initialized `thread_pool_t` on the caller's stack or in static storage. Must not be NULL. The function writes all fields; the caller need not initialize any of them.
- `num_threads`: number of worker threads to create. Must be in range `[1, THREAD_POOL_DEFAULT_SIZE]`. Values outside this range: clamp to `[1, THREAD_POOL_DEFAULT_SIZE]` and log a warning.
**Return values**:
- `0`: success. All `num_threads` threads are running and blocked on `pthread_cond_wait(&pool->not_empty, &pool->lock)`. `pool->thread_count == num_threads`.
- `-1`: failure. `pthread_mutex_init`, `pthread_cond_init`, or any `pthread_create` call failed. On failure, all successfully created threads have been signaled to exit and joined. The mutex and cond vars that were successfully initialized have been destroyed. The pool is in an unusable state; the caller must not call any other pool function.
**Side effects**: calls `pthread_mutex_init`, `pthread_cond_init` (×2), and `num_threads` calls to `pthread_create`. Each `pthread_create` call may take 10–15 µs and allocate 8 MB of virtual address space (thread stack).
**Thread safety**: must be called before any threads can access `pool`. Not re-entrant.
**Error cleanup invariant**: if `pthread_create` fails for the `k`th thread, the function sets `pool->shutdown = 1`, calls `pthread_cond_broadcast(&pool->not_empty)`, then calls `pthread_join` on threads `0..k-1`, then calls `pthread_mutex_destroy` and `pthread_cond_destroy`. The caller receives -1.
---
### 4.2 `thread_pool_enqueue()`
```c
/* thread_pool.h */
int thread_pool_enqueue(thread_pool_t *pool,
                        int client_fd,
                        const char *client_ip,
                        int client_port);
```
**Parameters**:
- `pool`: an initialized pool with `pool->shutdown == 0`. Behavior is undefined if called after `thread_pool_shutdown()`.
- `client_fd`: a connected socket FD. Ownership transfers to the pool: on success, the caller must not close or use `client_fd`. On failure (-1 return), the caller retains ownership and must close it.
- `client_ip`: null-terminated dotted-decimal string. Copied into the `work_item_t`; the caller's buffer need not remain valid after this call returns.
- `client_port`: host-byte-order port number.
**Return values**:
- `0`: success. The item has been enqueued. One worker thread has been signaled.
- `-1`: queue full (`pool->count >= WORK_QUEUE_CAPACITY`). `client_fd` was NOT enqueued; caller must handle it (send 503 and close).
**Concurrency**: acquires `pool->lock`, checks `pool->count`, conditionally enqueues and signals `pool->not_empty`, releases `pool->lock`. The critical section is O(1). Expected hold time: < 500 ns.
**Must not block**: the accept loop calls this while holding no other locks. It must return immediately even when the queue is full (return -1, not block on `not_full`). The `not_full` condition variable exists for potential future use where blocking on a full queue is acceptable; in this module it is not used.
---
### 4.3 `thread_pool_shutdown()`
```c
/* thread_pool.h */
void thread_pool_shutdown(thread_pool_t *pool);
```
**Parameters**: `pool` — an initialized pool. May be called once `pool->shutdown == 0`. Calling twice is undefined behavior.
**Behavior**:
1. Acquires `pool->lock`.
2. Sets `pool->shutdown = 1`.
3. Calls `pthread_cond_broadcast(&pool->not_empty)` — wakes all workers blocked in `pthread_cond_wait`.
4. Releases `pool->lock`.
5. Calls `pthread_join(pool->threads[i], NULL)` for `i = 0..pool->thread_count-1` in order.
**Postconditions**: all worker threads have returned. All in-flight requests (items that were dequeued before shutdown was set) have completed. Items remaining in the queue at shutdown time are NOT processed — they are abandoned. (This is acceptable: the accept loop stops enqueuing before calling `thread_pool_shutdown()`, so in practice the queue drains naturally before shutdown is called.)
**Blocking duration**: blocks until the last in-flight request completes. Maximum block time = `KEEPALIVE_TIMEOUT_SECONDS` (30s) × `pool->thread_count` in the pathological case where every worker is blocked in `read_request_with_timeout()` on an idle client at the moment of shutdown. In practice, the typical shutdown time is the duration of the longest active file transfer.
**Resource cleanup**: calls `pthread_mutex_destroy(&pool->lock)`, `pthread_cond_destroy(&pool->not_empty)`, `pthread_cond_destroy(&pool->not_full)` after all joins complete.
---
### 4.4 `worker_thread()`
```c
/* thread_pool.c — not exported */
static void *worker_thread(void *arg);
```
**Parameters**: `arg` is cast to `thread_pool_t *`. This is the only argument the thread receives; the global `g_doc_root` and `g_stats` are accessed directly.
**Return value**: always `NULL`.
**Loop invariant**: the thread holds `pool->lock` at the top of the loop and at the `pthread_cond_wait` call; it releases the lock before calling `handle_client_keep_alive()` and reacquires it at the top of the next iteration.
**Shutdown semantics**: when `pool->shutdown == 1` AND `pool->count == 0`, the thread releases the lock and returns. If `pool->shutdown == 1` AND `pool->count > 0`, the thread continues to dequeue and handle items — this ensures items enqueued before shutdown was set are fully served.
**Signal mask**: worker threads do NOT alter their signal mask. Signals (SIGTERM, SIGINT) are delivered to an arbitrary thread. If they arrive at a worker, the `signal_handler()` sets the shutdown flag and returns; the worker's current syscall (possibly `recv()` inside `read_request_with_timeout()`) may return `EINTR`, which is handled by retrying `select()` and checking the shutdown flag.
---
### 4.5 `handle_client_keep_alive()`
```c
/* connection.h */
void handle_client_keep_alive(int client_fd,
                               const char *client_ip,
                               int client_port,
                               const char *doc_root);
```
**Parameters**: all four are owned by the caller. `client_fd` is closed by this function before it returns. `doc_root` is read-only.
**Return value**: void. The function closes `client_fd` on all exit paths.
**Loop behavior**: reads one HTTP request, parses it, serves it, then decides whether to loop based on `should_keep_alive()`. Exits the loop and closes `client_fd` when:
- `read_request_with_timeout()` returns ≤ 0 (client closed, timeout, or error).
- `http_parse_request()` returns a non-OK error code (sends appropriate 4xx and breaks).
- `should_keep_alive()` returns 0 (`Connection: close` header or HTTP/1.0 default).
- `requests_handled >= MAX_REQUESTS_PER_CONNECTION`.
- `atomic_load(&shutdown_flag) != 0` (checked at the top of each loop iteration).
**Invariant**: `client_fd` is `close()`d exactly once, at the bottom of this function (after the loop), on every exit path.
---
### 4.6 `read_request_with_timeout()`
```c
/* connection.h */
ssize_t read_request_with_timeout(int fd, char *buf, size_t buf_size,
                                   int timeout_seconds);
```
**Parameters**:
- `fd`: connected client socket, blocking mode.
- `buf`: caller-supplied buffer, at least `buf_size` bytes.
- `buf_size`: buffer capacity including room for null terminator.
- `timeout_seconds`: maximum seconds to wait for data on each `recv()` attempt.
**Return values**:
- `> 0`: total bytes accumulated. `buf` is null-terminated. `strstr(buf, "\r\n\r\n") != NULL`.
- `0`: clean EOF — client sent TCP FIN before completing headers.
- `-1`: timeout expired, `recv()` error, or buffer exhausted without finding `\r\n\r\n`.
**Algorithm**: calls `select(fd + 1, &read_fds, NULL, NULL, &tv)` with `tv.tv_sec = timeout_seconds` before each `recv()`. If `select()` returns 0 (timeout), returns -1. If `select()` returns -1 with `errno == EINTR`, re-constructs the `fd_set` and `timeval` and retries. Uses the same accumulation loop as M1's `read_request()` but with the `select()` guard wrapped around each `recv()`.
**Difference from M1's `read_request()`**: M1's version has no timeout; clients that connect and send nothing block the server forever. This version returns -1 after `timeout_seconds` of inactivity, freeing the worker thread.
**Note on `timeval` reconstruction**: POSIX allows `select()` to modify the `timeval` argument. After an `EINTR` retry, a fresh `timeval` with the original timeout must be constructed — not the modified one. This means the per-EINTR retry resets the full timeout. This is acceptable: `EINTR` is rare (signals during request reading are unusual) and the timeout is for idle connections, not precise timing.
---
### 4.7 `should_keep_alive()`
```c
/* connection.h */
int should_keep_alive(const http_request_t *req);
```
**Parameters**: a fully parsed `http_request_t` (return value of `http_parse_request()` was `HTTP_PARSE_OK`).
**Return values**:
- `1`: keep the connection open after this response.
- `0`: close the connection after this response.
**Logic** (implementing RFC 7230 §6.3):
```
connection_header = get_header(req, "connection")
if connection_header != NULL:
    if strcasecmp(connection_header, "close") == 0:     return 0
    if strcasecmp(connection_header, "keep-alive") == 0: return 1
http11 = (strcmp(req->version, "HTTP/1.1") == 0)
return http11 ? 1 : 0
```
**Thread safety**: pure function, no side effects.
---
### 4.8 `signal_handler()`
```c
/* server.c — internal */
static void signal_handler(int sig);
```
**Parameters**: `sig` — the signal number (SIGTERM or SIGINT). Not inspected; any registered signal triggers the same shutdown sequence.
**Body**:
```c
static void signal_handler(int sig) {
    (void)sig;
    atomic_store(&shutdown_flag, 1);
}
```
**Async-signal-safety**: `atomic_store` is async-signal-safe on Linux (compiles to a single `MOV` or `XCHG` instruction). No `malloc`, `printf`, mutex operations, or other async-signal-unsafe calls may appear here.
**Registration**: called via `signal(SIGTERM, signal_handler)` and `signal(SIGINT, signal_handler)` in `main()` before creating the thread pool. `signal()` is used rather than `sigaction()` for simplicity; in production, `sigaction()` with `SA_RESTART` would be preferable.
---
### 4.9 `stats_connection_opened()`, `stats_connection_closed()`, `stats_request_served()`
```c
/* stats.h */
void stats_connection_opened(server_stats_t *stats);
void stats_connection_closed(server_stats_t *stats);
void stats_request_served(server_stats_t *stats, long long bytes_sent);
```
Each acquires `stats->lock`, modifies the appropriate fields, releases `stats->lock`. Critical section is 2–4 integer operations. Expected hold time: < 200 ns.
**Invariant**: `active_connections` is always ≥ 0. `stats_connection_closed()` must only be called if `stats_connection_opened()` was called for the same connection; otherwise `active_connections` underflows.
---
### 4.10 `access_log_init()`, `access_log_write()`, `access_log_close()`
```c
/* stats.h */
int  access_log_init(access_log_t *log, const char *path);
void access_log_write(access_log_t *log,
                      const char *client_ip, int client_port,
                      const char *method, const char *path,
                      int status_code, long long bytes_sent);
void access_log_close(access_log_t *log);
```
`access_log_init()`: opens `path` with `fopen(path, "a")`. Returns 0 on success, -1 if `fopen` fails. If -1, subsequent `access_log_write()` calls must check `log->file != NULL` and skip silently.
`access_log_write()`: acquires `log->lock`, formats one Apache-style CLF log line using `fprintf()`, calls `fflush(log->file)`, releases `log->lock`. Format:
```
CLIENT_IP:PORT - - [DD/Mon/YYYY:HH:MM:SS +0000] "METHOD PATH" STATUS BYTES\n
```
`fflush()` inside the lock is mandatory: it ensures that on a crash, the last log entry written under the lock is visible in the file. Without `fflush()`, libc's stdio buffer may hold the entry in user-space memory that is lost on crash.
`access_log_close()`: acquires lock, calls `fclose(log->file)`, sets `log->file = NULL`, releases lock, calls `pthread_mutex_destroy(&log->lock)`.
---
## 5. Algorithm Specification
### 5.1 `worker_thread()` — Core Loop

![Thread Pool Worker Loop: Mutex/Cond State Transitions](./diagrams/tdd-diag-31.svg)

```
INPUT:  arg = pointer to thread_pool_t
SETUP:
  pool = (thread_pool_t *)arg
  pthread_mutex_lock(&pool->lock)                   ← acquire lock at top of loop
MAIN LOOP:
  STEP A: Wait for work or shutdown
    while pool->count == 0 AND NOT pool->shutdown:
      pthread_cond_wait(&pool->not_empty, &pool->lock)
      ← atomically: releases lock + blocks; on signal: reacquires lock + returns
      ← use WHILE not IF: handle spurious wakeups
  STEP B: Shutdown check
    if pool->shutdown AND pool->count == 0:
      pthread_mutex_unlock(&pool->lock)
      return NULL                                   ← thread exits cleanly
      ← if pool->shutdown AND pool->count > 0: fall through to dequeue
        (drain remaining items even during shutdown)
  STEP C: Dequeue one item (under lock)
    item = pool->queue[pool->head]                  ← value copy: 24 bytes
    pool->head = (pool->head + 1) % WORK_QUEUE_CAPACITY
    pool->count--
    pthread_cond_signal(&pool->not_full)            ← signal accept loop (if it blocks)
    pthread_mutex_unlock(&pool->lock)               ← RELEASE LOCK before handling
  STEP D: Handle the connection (outside lock)
    stats_connection_opened(&g_stats)
    handle_client_keep_alive(item.client_fd,        ← closes client_fd before returning
                              item.client_ip,
                              item.client_port,
                              g_doc_root)
    stats_connection_closed(&g_stats)
  STEP E: Reacquire lock and loop
    pthread_mutex_lock(&pool->lock)
    GOTO MAIN LOOP
```
**Critical property**: the lock is held for the duration of STEP A and STEP C (queue manipulation only — O(1) operations). The lock is **not** held during STEP D (the entire request/response cycle, which may take hundreds of milliseconds for large files). This is what makes the thread pool actually concurrent — all N workers can be in STEP D simultaneously, serving N clients in parallel.
**Spurious wakeup handling**: POSIX explicitly permits `pthread_cond_wait()` to return without a corresponding `pthread_cond_signal()` or `pthread_cond_broadcast()`. The `while` loop in STEP A re-checks `pool->count == 0` every time it returns from `pthread_cond_wait()`, so spurious wakeups cause a harmless re-check and re-wait.
---
### 5.2 `thread_pool_init()` — Initialization with Error Rollback
```
INPUT:  pool, num_threads
STEP 1: Zero-initialize all fields
  memset(pool, 0, sizeof(*pool))
  pool->thread_count = num_threads
STEP 2: Initialize synchronization primitives
  rc = pthread_mutex_init(&pool->lock, NULL)
  if rc != 0: log_error("pthread_mutex_init", rc); return -1
  rc = pthread_cond_init(&pool->not_empty, NULL)
  if rc != 0:
    pthread_mutex_destroy(&pool->lock)
    log_error("pthread_cond_init not_empty", rc)
    return -1
  rc = pthread_cond_init(&pool->not_full, NULL)
  if rc != 0:
    pthread_cond_destroy(&pool->not_empty)
    pthread_mutex_destroy(&pool->lock)
    log_error("pthread_cond_init not_full", rc)
    return -1
STEP 3: Create worker threads
  for i = 0..num_threads-1:
    rc = pthread_create(&pool->threads[i], NULL, worker_thread, pool)
    if rc != 0:
      log_error("pthread_create", rc)
      ← rollback: signal threads 0..i-1 to exit
      pthread_mutex_lock(&pool->lock)
      pool->shutdown = 1
      pthread_cond_broadcast(&pool->not_empty)
      pthread_mutex_unlock(&pool->lock)
      ← join threads 0..i-1
      for j = 0..i-1:
        pthread_join(pool->threads[j], NULL)
      pthread_cond_destroy(&pool->not_full)
      pthread_cond_destroy(&pool->not_empty)
      pthread_mutex_destroy(&pool->lock)
      return -1
STEP 4: return 0
```

![Thread Pool Architecture: Accept Loop, Queue, Workers](./diagrams/tdd-diag-32.svg)

---
### 5.3 `thread_pool_enqueue()` — Non-Blocking Enqueue
```
INPUT:  pool, client_fd, client_ip, client_port
STEP 1: Acquire lock
  pthread_mutex_lock(&pool->lock)
STEP 2: Check capacity
  if pool->count >= WORK_QUEUE_CAPACITY:
    pthread_mutex_unlock(&pool->lock)
    return -1                              ← caller sends 503, closes client_fd
STEP 3: Write item into circular buffer
  item_ptr = &pool->queue[pool->tail]
  item_ptr->client_fd   = client_fd
  item_ptr->client_port = client_port
  strncpy(item_ptr->client_ip, client_ip, INET_ADDRSTRLEN - 1)
  item_ptr->client_ip[INET_ADDRSTRLEN - 1] = '\0'
  pool->tail  = (pool->tail + 1) % WORK_QUEUE_CAPACITY
  pool->count++
STEP 4: Signal one waiting worker
  pthread_cond_signal(&pool->not_empty)
STEP 5: Release lock and return
  pthread_mutex_unlock(&pool->lock)
  return 0
```
---
### 5.4 `handle_client_keep_alive()` — Per-Connection Request Loop

![Concurrency Models Comparison: Thread-per-Connection vs Thread Pool](./diagrams/tdd-diag-33.svg)

```
INPUT:  client_fd, client_ip, client_port, doc_root
SETUP:
  char buf[REQUEST_BUF_SIZE]        ← 8192 bytes, stack-allocated
  int requests_handled = 0
MAIN LOOP:
  STEP A: Check shutdown flag
    if atomic_load(&shutdown_flag) != 0:
      break                         ← stop accepting new requests on this connection
  STEP B: Read next request with timeout
    req_len = read_request_with_timeout(client_fd, buf, sizeof(buf),
                                        KEEPALIVE_TIMEOUT_SECONDS)
    if req_len == 0:
      break                         ← clean EOF: client closed connection
    if req_len < 0:
      break                         ← timeout or error: close connection
  STEP C: Parse the request
    http_request_t req              ← stack-allocated ~43 KB
    rc = http_parse_request(buf, (size_t)req_len, &req)
    if rc == HTTP_PARSE_URI_TOO_LONG:
      send_all(client_fd, HTTP_414, strlen(HTTP_414))
      break                         ← don't continue on parser error
    if rc == HTTP_PARSE_NOT_IMPL:
      send_all(client_fd, HTTP_501, strlen(HTTP_501))
      break
    if rc != HTTP_PARSE_OK:
      send_all(client_fd, HTTP_400, strlen(HTTP_400))
      break
  STEP D: Determine keep-alive intent BEFORE serving
    int keep_alive = should_keep_alive(&req)
  STEP E: Serve the request
    int send_body = (classify_method(req.method) == HTTP_METHOD_GET)
    serve_file(client_fd, doc_root, &req, send_body)
    ← serve_file() writes the response (headers + body for GET, headers only for HEAD)
    ← It does NOT close client_fd
  STEP F: Log the request
    ← After serve_file(), we don't know the status code from serve_file()
    ← Options: (a) thread-local last_status global, (b) pass status out of serve_file()
    ← For this module: skip status code in log, log method+path only
    access_log_write(&g_log, client_ip, client_port,
                     req.method, req.path, 0, 0)   ← 0 = status unknown in this design
  STEP G: Increment request counter
    requests_handled++
    stats_request_served(&g_stats, 0)
  STEP H: Check loop exit conditions
    if NOT keep_alive:
      break                         ← Connection: close requested
    if requests_handled >= MAX_REQUESTS_PER_CONNECTION:
      break                         ← hard cap: prevent indefinite monopolization
    GOTO MAIN LOOP
CLEANUP:
  close(client_fd)                  ← MANDATORY, on every exit path
  ← loop exits via break → reaches close(client_fd) → function returns
```
**Stack frame total for `handle_client_keep_alive()`**:
```
buf[8192]          8,192 B   (request buffer)
http_request_t     ~43,064 B (parsed request struct, includes headers array)
misc locals        ~200 B
TOTAL:             ~51,456 B (~50 KB)
```
With 16 worker threads each potentially executing `handle_client_keep_alive()`, the combined stack usage is 16 × 50 KB = 800 KB of stack pages (virtual). Physical pages are only allocated on first touch; a request that does not parse all 32 headers will not fault the full `headers[]` array into physical memory.
---
### 5.5 `read_request_with_timeout()` — select()-Gated recv() Loop

![HTTP/1.1 Keep-Alive: Single Connection Multiple Requests Sequence](./diagrams/tdd-diag-34.svg)

```
INPUT:  fd, buf, buf_size, timeout_seconds
SETUP:
  size_t total = 0
MAIN LOOP: while total < buf_size - 1:
  STEP A: Set up select() call
    fd_set read_fds
    FD_ZERO(&read_fds)
    FD_SET(fd, &read_fds)
    struct timeval tv
    tv.tv_sec  = timeout_seconds
    tv.tv_usec = 0
    ← fresh timeval on each iteration and each EINTR retry
  STEP B: Wait for socket readability
    RETRY_SELECT:
    ready = select(fd + 1, &read_fds, NULL, NULL, &tv)
    if ready < 0:
      if errno == EINTR:
        if atomic_load(&shutdown_flag) != 0: return -1  ← shutdown during wait
        ← reconstruct fd_set and tv, retry
        FD_ZERO(&read_fds); FD_SET(fd, &read_fds)
        tv.tv_sec = timeout_seconds; tv.tv_usec = 0
        GOTO RETRY_SELECT
      return -1                          ← genuine select() error
    if ready == 0:
      return -1                          ← timeout: no data for timeout_seconds
  STEP C: Receive data
    n = recv(fd, buf + total, buf_size - 1 - total, 0)
    if n < 0:
      if errno == EINTR: GOTO STEP A    ← reconstruct and retry
      return -1
    if n == 0:
      return 0                          ← clean EOF
    total += n
    buf[total] = '\0'
  STEP D: Check for end-of-headers delimiter
    if strstr(buf, "\r\n\r\n") != NULL:
      return (ssize_t)total
    ← not found: continue loop to read more
return -1                               ← buffer exhausted: request too large
```
**Why reconstruct `timeval` after `EINTR`**: POSIX does not guarantee the value of `tv` after `select()` returns. On Linux, `tv` is modified to reflect remaining time, but relying on this is non-portable and subtly incorrect — if a burst of signals arrives, the remaining timeout could approach zero and cause premature timeout. Reconstructing with the full `timeout_seconds` value resets the idle deadline, which is the correct semantic: "close if silent for 30 consecutive seconds."
---
### 5.6 `thread_pool_shutdown()` — Drain and Join
```
INPUT:  pool
STEP 1: Signal all workers to exit
  pthread_mutex_lock(&pool->lock)
  pool->shutdown = 1
  pthread_cond_broadcast(&pool->not_empty)   ← wake all workers
  pthread_mutex_unlock(&pool->lock)
  ← workers wake, re-check condition, see shutdown==1 AND count==0, exit
  ← OR: workers see shutdown==1 AND count>0, drain items, then exit
STEP 2: Join all worker threads
  for i = 0..pool->thread_count-1:
    pthread_join(pool->threads[i], NULL)     ← blocks until thread i exits
    ← after join: thread i has returned from worker_thread()
    ← any client_fd it was handling has been closed
STEP 3: Destroy synchronization primitives
  pthread_mutex_destroy(&pool->lock)
  pthread_cond_destroy(&pool->not_empty)
  pthread_cond_destroy(&pool->not_full)
STEP 4: return (void)
```

![Per-Connection Idle Timeout: select() Mechanism](./diagrams/tdd-diag-35.svg)

---
### 5.7 Accept Loop with Pool Integration and Graceful Shutdown
```
INPUT:  server_fd (listening), pool, g_doc_root
MAIN LOOP: while atomic_load(&shutdown_flag) == 0:
  STEP A: Accept connection
    struct sockaddr_in client_addr
    socklen_t client_len = sizeof(client_addr)
    client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len)
    if client_fd < 0:
      if errno == EINTR:
        continue                          ← signal interrupted accept, re-check flag
      if errno == EMFILE OR errno == ENFILE:
        fprintf(stderr, "FD limit: %s\n", strerror(errno))
        usleep(100000)                    ← 100ms back-off
        continue
      perror("accept")
      continue                           ← non-fatal: log and keep running
  STEP B: Extract client info
    char client_ip[INET_ADDRSTRLEN]
    inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip))
    int client_port = ntohs(client_addr.sin_port)
  STEP C: Enqueue or reject
    rc = thread_pool_enqueue(&g_pool, client_fd, client_ip, client_port)
    if rc < 0:
      ← Queue full: send 503 and close immediately
      const char *resp_503 = "HTTP/1.1 503 Service Unavailable\r\n"
                             "Content-Length: 0\r\n"
                             "Connection: close\r\n"
                             "Retry-After: 1\r\n"
                             "\r\n"
      send(client_fd, resp_503, strlen(resp_503), MSG_NOSIGNAL)
      close(client_fd)
      fprintf(stderr, "Queue full: rejected %s:%d\n", client_ip, client_port)
POST-LOOP (after shutdown_flag is set):
  close(server_fd)                        ← stop OS from queueing new connections
  thread_pool_shutdown(&pool)             ← drain in-flight, join all workers
  access_log_close(&g_log)
  printf("Graceful shutdown complete.\n")
  return EXIT_SUCCESS
```

![Graceful Shutdown Sequence: SIGTERM to Clean Process Exit](./diagrams/tdd-diag-36.svg)

---
## 6. Error Handling Matrix
| Error | Detected By | Recovery | User-Visible? |
|---|---|---|---|
| `pthread_create()` fails during `thread_pool_init()` | `thread_pool_init()` | Rollback: signal+join created threads, destroy mutexes, return -1; `main()` exits | Server does not start |
| `thread_pool_enqueue()` when queue full | `thread_pool_enqueue()` | Returns -1; accept loop sends 503 + `Retry-After: 1` and closes `client_fd` | 503 Service Unavailable |
| `accept()` returns `EINTR` | accept loop | `continue` — re-check `shutdown_flag` | None |
| `accept()` returns `EMFILE`/`ENFILE` | accept loop | Log, `usleep(100ms)`, `continue` | New connections may time out briefly |
| `accept()` returns other error | accept loop | `perror()` + `continue` | None |
| `read_request_with_timeout()` timeout | `handle_client_keep_alive()` | Break keep-alive loop, `close(client_fd)` | Connection closed silently |
| `read_request_with_timeout()` returns 0 (EOF) | `handle_client_keep_alive()` | Break loop, `close(client_fd)` | None — client already closed |
| `read_request_with_timeout()` returns -1 (error) | `handle_client_keep_alive()` | Break loop, `close(client_fd)` | None |
| `http_parse_request()` returns `HTTP_PARSE_BAD_REQUEST` | `handle_client_keep_alive()` | Send 400, break loop, `close(client_fd)` | 400 Bad Request |
| `http_parse_request()` returns `HTTP_PARSE_URI_TOO_LONG` | `handle_client_keep_alive()` | Send 414, break loop, `close(client_fd)` | 414 URI Too Long |
| `http_parse_request()` returns `HTTP_PARSE_NOT_IMPL` | `handle_client_keep_alive()` | Send 501, break loop, `close(client_fd)` | 501 Not Implemented |
| `serve_file()` error (handled internally) | `serve_file()` | Sends 403/404/500 to client, returns; keep-alive loop continues if `Connection` header allows | 403/404/500 |
| `select()` returns `EINTR` in `read_request_with_timeout()` | `read_request_with_timeout()` | Reconstruct `fd_set`+`timeval`, check `shutdown_flag`, retry | None |
| `select()` returns 0 (timeout) | `read_request_with_timeout()` | Return -1 | Connection closed silently |
| `MAX_REQUESTS_PER_CONNECTION` reached | `handle_client_keep_alive()` | Break loop, `close(client_fd)` | Connection closed (browser reconnects) |
| `pthread_mutex_lock()` fails | Any stats/log function | This is a fatal programming error (corrupted mutex) — call `abort()` | Server crash (logged before abort) |
| `access_log_init()` fails (`fopen` returns NULL) | `main()` | Log warning, continue without access logging | Log file absent |
| `fflush()` fails in `access_log_write()` | `access_log_write()` | Log warning, continue (data may be lost) | Log entries may be lost |
| Spurious `pthread_cond_wait()` wakeup | `worker_thread()` | `while` loop re-checks condition, re-waits | None |
| `pthread_create()` fails in thread-per-connection model (Phase 1) | accept loop | Close `client_fd`, free `connection_t`, log error, continue | Connection dropped silently |
| `shutdown_flag` set during keep-alive loop | `handle_client_keep_alive()` | Break loop at top of next iteration, `close(client_fd)` | In-flight request completes first |
**Data race scenarios and their prevention**:
| Race | Prevention |
|---|---|
| Two threads simultaneously incrementing `g_stats.active_connections` | `g_stats.lock` mutex |
| Two threads writing to `g_log.file` simultaneously | `g_log.lock` mutex |
| Accept loop reading `pool->count` while worker modifies it | `pool->lock` mutex held by both |
| Worker reading `pool->shutdown` without the lock | Written under lock; workers re-read it under the lock inside the `while` condition |
| Signal handler writing `shutdown_flag` while accept loop reads it | `atomic_int` with default `memory_order_seq_cst` |
---
## 7. Implementation Sequence with Checkpoints
### Phase 1 — Thread-per-Connection: `connection_t`, `pthread_create` + Detach (1–1.5 h)
Create `connection.h` and `connection.c`. Implement `handle_client_keep_alive()` as a simple pass-through to M3's `serve_file()` (no keep-alive loop yet — that is Phase 5). Update `server.c`: replace the sequential accept loop with one that heap-allocates `connection_t`, calls `pthread_create()`, and `pthread_detach()`.
```c
/* Phase 1: thread-per-connection accept loop (temporary, replaced in Phase 4+) */
while (!atomic_load(&shutdown_flag)) {
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    int client_fd = accept(server_fd,
                           (struct sockaddr *)&client_addr,
                           &client_len);
    if (client_fd < 0) {
        if (errno == EINTR) continue;
        perror("accept"); continue;
    }
    connection_t *conn = malloc(sizeof(connection_t));
    if (!conn) { close(client_fd); continue; }
    conn->client_fd   = client_fd;
    conn->client_port = ntohs(client_addr.sin_port);
    conn->doc_root    = g_doc_root;
    inet_ntop(AF_INET, &client_addr.sin_addr,
              conn->client_ip, sizeof(conn->client_ip));
    pthread_t tid;
    if (pthread_create(&tid, NULL, connection_thread, conn) != 0) {
        perror("pthread_create");
        close(client_fd);
        free(conn);
        continue;
    }
    pthread_detach(tid);
}
```
```c
/* connection_thread: wraps handle_client_keep_alive() */
static void *connection_thread(void *arg) {
    connection_t *conn = (connection_t *)arg;
    handle_client_keep_alive(conn->client_fd, conn->client_ip,
                              conn->client_port, conn->doc_root);
    free(conn);
    return NULL;
}
```
**Checkpoint 1**: `./server 8080 /tmp/webroot` + `ab -n 1000 -c 10 http://localhost:8080/` → "Failed requests: 0". `htop` shows ~10 threads. `ls /proc/$(pgrep server)/fd | wc -l` returns to baseline after `ab` completes (no FD leak).
---
### Phase 2 — `thread_pool_t` Data Structure: Circular Buffer, Mutex, Two Cond Vars (1 h)
Create `thread_pool.h` with all struct definitions. Create `thread_pool.c` with stub implementations that return 0. Compile the project: `gcc -Wall -Wextra -std=c11 -o server server.c http_parse.c file_server.c thread_pool.c connection.c stats.c -lpthread`.
Verify struct size:
```c
/* Checkpoint 2: compile and run this snippet */
printf("sizeof(work_item_t)  = %zu\n", sizeof(work_item_t));   /* expect 24 */
printf("sizeof(thread_pool_t) = %zu\n", sizeof(thread_pool_t)); /* expect ~24740 */
_Static_assert(WORK_QUEUE_CAPACITY == 1024, "queue capacity");
_Static_assert((WORK_QUEUE_CAPACITY & (WORK_QUEUE_CAPACITY - 1)) == 0,
               "WORK_QUEUE_CAPACITY must be power of 2");
```
**Checkpoint 2**: zero compiler warnings, both `printf` values match expectations, both `_Static_assert` pass.
---
### Phase 3 — `worker_thread()`: pthread_cond_wait with while, Dequeue Under Lock, Handle Outside Lock (1 h)
Implement the full `worker_thread()` per Section 5.1. Implement `thread_pool_init()` per Section 5.2 (including error rollback). Do not yet connect to the accept loop — test in isolation:
```c
/* Phase 3 isolation test in main() */
thread_pool_t pool;
thread_pool_init(&pool, 4);
printf("Pool initialized with 4 threads\n");
sleep(1);
/* Verify: 4 threads are running (htop should show them, blocked on cond_wait) */
thread_pool_shutdown(&pool);
printf("Pool shut down\n");
/* Verify: no threads remain; process exits cleanly */
```
**Checkpoint 3**: `valgrind --tool=helgrind ./server_test` shows no lock-order violations. `htop` shows 4 threads during sleep, 0 extra threads after shutdown. No segfaults.
---
### Phase 4 — `thread_pool_enqueue()`: Return -1 When Full; Accept Loop Sends 503 (0.5 h)
Implement `thread_pool_enqueue()` per Section 5.3. Update `server.c`'s accept loop: replace the thread-per-connection approach with `thread_pool_enqueue()`. Send 503 on queue-full return.
**Checkpoint 4**: `./server 8080 /tmp/webroot` + `ab -n 5000 -c 50 http://localhost:8080/` → "Failed requests: 0" (with 16 workers the queue should not fill on this load). Reduce pool to 1 thread and queue to 4 (temporarily) + 50 concurrent connections → some connections receive `503 Service Unavailable`. Server does not crash.
---
### Phase 5 — `handle_client_keep_alive()`: Multi-Request Loop, `should_keep_alive()`, MAX_REQUESTS Guard (1–1.5 h)
Implement `should_keep_alive()` per Section 4.7. Expand `handle_client_keep_alive()` with the full loop per Section 5.4. Replace the stub implementation from Phase 1.
**Checkpoint 5a** (keep-alive working):
```bash
ab -n 10000 -c 100 -k http://localhost:8080/index.html 2>&1 | grep -E "Keep-Alive|Requests per second|Failed"
# Expected: "Keep-Alive requests: 10000" or equivalent keep-alive metric
# "Failed requests: 0"
# "Requests per second" > baseline from Phase 1 (fewer TCP handshakes)
```
**Checkpoint 5b** (`Connection: close` respected):
```bash
curl -v -H "Connection: close" http://localhost:8080/index.html 2>&1 | grep "Connection:"
# Server response must include "Connection: close"
```
**Checkpoint 5c** (HTTP/1.0 closes by default):
```bash
printf "GET / HTTP/1.0\r\nHost: localhost\r\n\r\n" | nc -q 1 localhost 8080
# Server must respond and close; nc must exit cleanly
```
---
### Phase 6 — `read_request_with_timeout()`: select()-Based Per-Read Deadline, EINTR Retry (0.5–1 h)
Implement `read_request_with_timeout()` per Section 5.5. Replace calls to M1's `read_request()` inside `handle_client_keep_alive()` with `read_request_with_timeout(..., KEEPALIVE_TIMEOUT_SECONDS)`.
**Checkpoint 6a** (timeout fires):
```bash
# Open a connection and send nothing; server should close after 30s
# For testing, temporarily reduce KEEPALIVE_TIMEOUT_SECONDS to 2
time (python3 -c "
import socket, time
s = socket.socket()
s.connect(('localhost', 8080))
time.sleep(10)  # longer than 2s timeout
print('still alive?', s.recv(1))
" 2>&1)
# Expected: s.recv(1) returns b'' (EOF) after ~2 seconds
```
**Checkpoint 6b** (FD count stable after timeout):
```bash
SERVER_PID=$(pgrep -n server)
BASELINE=$(ls /proc/$SERVER_PID/fd | wc -l)
# Open 50 connections and send nothing; wait for timeout
python3 -c "
import socket, time
socks = [socket.socket() for _ in range(50)]
for s in socks: s.connect(('localhost', 8080))
time.sleep(5)  # with 2s timeout, all should be closed by server
" &
sleep 8
AFTER=$(ls /proc/$SERVER_PID/fd | wc -l)
echo "Baseline: $BASELINE  After: $AFTER"
# Expected: AFTER == BASELINE (server closed all idle FDs)
```
---
### Phase 7 — Graceful Shutdown: `atomic_int` Flag, `signal_handler()`, Accept Loop Exit, `thread_pool_shutdown()` with Broadcast + Join (1–1.5 h)
Implement `signal_handler()`. Register it for SIGTERM and SIGINT in `main()`. Add the `atomic_load(&shutdown_flag)` check to the accept loop. Implement `thread_pool_shutdown()` per Section 5.6. Update `main()` post-loop to call `close(server_fd)` then `thread_pool_shutdown()` then `access_log_close()`.
```c
/* server.c — main() signal setup */
signal(SIGPIPE, SIG_IGN);          /* must come first */
signal(SIGTERM, signal_handler);
signal(SIGINT,  signal_handler);
```
**Checkpoint 7a** (SIGTERM causes clean exit):
```bash
./server 8080 /tmp/webroot &
SERVER_PID=$!
sleep 0.5
kill -TERM $SERVER_PID
wait $SERVER_PID
echo "Exit code: $?"   # Must be 0 (EXIT_SUCCESS)
```
**Checkpoint 7b** (in-flight requests complete before exit):
```bash
./server 8080 /tmp/webroot &
SERVER_PID=$!
sleep 0.3
# Start a large download in background
curl -s "http://localhost:8080/bigfile.bin" -o /tmp/bigfile_dl &
CURL_PID=$!
sleep 0.1   # let download start
kill -TERM $SERVER_PID  # send shutdown while download is active
wait $CURL_PID
echo "curl exit: $?"   # Must be 0 (download completed, not truncated)
md5sum /tmp/webroot/bigfile.bin /tmp/bigfile_dl   # Must match
wait $SERVER_PID
echo "server exit: $?"  # Must be 0
```
**Checkpoint 7c** (Ctrl-C works the same as SIGTERM):
```bash
./server 8080 /tmp/webroot
# Press Ctrl-C
# Expected: "Graceful shutdown complete." printed; process exits with code 0
```
---
### Phase 8 — Shared State: `server_stats_t` and `access_log_t` (0.5–1 h)
Create `stats.h` and `stats.c`. Implement all functions per Sections 4.9 and 4.10. Initialize in `main()`. Add `stats_connection_opened/closed()` calls to `worker_thread()`. Add `access_log_write()` call to `handle_client_keep_alive()`.
**Checkpoint 8a** (no data races under ThreadSanitizer):
```bash
gcc -Wall -Wextra -std=c11 -O1 -g -fsanitize=thread \
    -o server_tsan server.c http_parse.c file_server.c thread_pool.c connection.c stats.c \
    -lpthread
./server_tsan 8080 /tmp/webroot &
TSAN_PID=$!
ab -n 2000 -c 20 -k http://localhost:8080/index.html > /dev/null 2>&1
kill -TERM $TSAN_PID
wait $TSAN_PID
# Expected: NO "ThreadSanitizer: data race" lines in output
```
**Checkpoint 8b** (access log entries are non-interleaved):
```bash
ab -n 1000 -c 50 http://localhost:8080/index.html > /dev/null 2>&1
wc -l access.log   # Must be 1000 (one line per request)
# Verify no truncated/interleaved lines:
awk '{if(NF < 5) print "CORRUPT: " NR " " $0}' access.log
# Expected: no output (all lines have >= 5 fields)
```
---
### Phase 9 — Testing (1–2 h)
Run `test_concurrent.sh` per Section 8. All tests must pass.
---
## 8. Test Specification

![Signal Delivery in Multithreaded Process: Safe Pattern](./diagrams/tdd-diag-37.svg)

### 8.1 Thread Pool Tests
**P1.1 — Init with default thread count**
```bash
# Start server, verify 16 worker threads exist
./server 8080 /tmp/webroot &
PID=$!; sleep 0.3
THREADS=$(ls /proc/$PID/task | wc -l)
echo "Threads: $THREADS"  # Expected: 17 (main + 16 workers)
kill $PID
```
**P1.2 — Queue full returns 503**
```bash
# Flood with 2000 concurrent connections (more than queue capacity of 1024)
# Use a tool that can hold connections open
python3 -c "
import socket, time, threading
socks = []
for i in range(200):
    try:
        s = socket.socket()
        s.connect(('localhost', 8080))
        socks.append(s)
    except: pass
time.sleep(2)
# Now try a new connection; should get 503
s2 = socket.socket()
s2.connect(('localhost', 8080))
s2.send(b'GET / HTTP/1.1\r\nHost: localhost\r\n\r\n')
resp = s2.recv(64).decode()
for s in socks: s.close()
s2.close()
print(resp[:30])
" 
```
**P1.3 — `thread_pool_init()` error rollback (simulated)**
This cannot be tested externally without injecting a `pthread_create` failure. Verify via code review: the rollback path in `thread_pool_init()` must call `pthread_join` on the `k` threads created before failure, then destroy mutexes and cond vars. Use AddressSanitizer (`-fsanitize=address`) to confirm no memory leaks in the rollback path.
**P1.4 — Shutdown drains in-flight requests**
```bash
# Verified in Checkpoint 7b above
```
---
### 8.2 Keep-Alive Tests
**K1 — HTTP/1.1 default is keep-alive**
```bash
# curl uses HTTP/1.1 by default; -v should show Connection: keep-alive in response
curl -v http://localhost:8080/index.html 2>&1 | grep -i "connection:"
# Expected: response includes "Connection: keep-alive" (not "close")
```
**K2 — Keep-alive increases throughput vs connection-per-request**
```bash
# With keep-alive (ab default for HTTP/1.1 is keep-alive with -k)
RPS_KA=$(ab -n 10000 -c 10 -k http://localhost:8080/index.html 2>&1 | grep "Requests per second" | awk '{print $4}')
# Without keep-alive
RPS_NO=$(ab -n 10000 -c 10    http://localhost:8080/index.html 2>&1 | grep "Requests per second" | awk '{print $4}')
echo "Keep-alive: $RPS_KA req/s   No keep-alive: $RPS_NO req/s"
# Expected: RPS_KA > RPS_NO (at least 20% faster on loopback)
```
**K3 — `Connection: close` header closes after one request**
```bash
# Send two requests on one connection; second should fail after server closes
python3 -c "
import socket
s = socket.socket()
s.connect(('localhost', 8080))
s.send(b'GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n')
resp = b''
while True:
    chunk = s.recv(4096)
    if not chunk: break
    resp += chunk
print('First response length:', len(resp))
# Now try a second request — server should have closed the connection
s.send(b'GET / HTTP/1.1\r\nHost: localhost\r\n\r\n')
second = s.recv(4096)
print('Second response length:', len(second))  # Must be 0 (connection closed)
s.close()
"
```
**K4 — MAX_REQUESTS_PER_CONNECTION enforced**
```bash
# Send 101 requests on one persistent connection
python3 -c "
import socket
s = socket.socket()
s.connect(('localhost', 8080))
count = 0
req = b'GET / HTTP/1.1\r\nHost: localhost\r\nConnection: keep-alive\r\n\r\n'
while True:
    try:
        s.send(req)
        resp = b''
        while b'\r\n\r\n' not in resp:
            chunk = s.recv(4096)
            if not chunk: 
                print(f'Connection closed after {count} requests')
                break
            resp += chunk
        if not chunk: break
        count += 1
    except BrokenPipeError:
        print(f'Broken pipe after {count} requests')
        break
s.close()
"
# Expected: "Connection closed after 100 requests"
```
**K5 — Idle timeout closes connection**
```bash
# With KEEPALIVE_TIMEOUT_SECONDS temporarily set to 2 for testing
time (python3 -c "
import socket, time
s = socket.socket()
s.connect(('localhost', 8080))
s.send(b'GET / HTTP/1.1\r\nHost: localhost\r\n\r\n')
# Read first response
resp = b''
while b'<h' not in resp: resp += s.recv(4096)
print('Got first response, waiting idle...')
time.sleep(5)   # longer than 2s timeout
data = s.recv(1)
print('Data after idle:', data)   # Must be b'' (server closed)
s.close()
")
# Expected: empty recv after ~2s
```
---
### 8.3 Graceful Shutdown Tests
**G1 — SIGTERM exits with code 0**
```bash
./server 8080 /tmp/webroot &
PID=$!; sleep 0.3
kill -TERM $PID; wait $PID
echo "Exit: $?"   # Must be 0
```
**G2 — SIGINT (Ctrl-C) exits with code 0**
```bash
./server 8080 /tmp/webroot &
PID=$!; sleep 0.3
kill -INT $PID; wait $PID
echo "Exit: $?"   # Must be 0
```
**G3 — In-flight download completes after SIGTERM**
```bash
# Create 2MB test file
dd if=/dev/urandom of=/tmp/webroot/shutdown_test.bin bs=1024 count=2048 2>/dev/null
./server 8080 /tmp/webroot &
PID=$!; sleep 0.3
curl -s http://localhost:8080/shutdown_test.bin -o /tmp/dl_test &
CURL_PID=$!
sleep 0.05   # Let download start but not finish (2MB at localhost ~1GB/s = ~2ms, so this may complete first)
# For a more reliable test, throttle with tc or use a very large file
kill -TERM $PID
wait $CURL_PID; CURL_RC=$?
wait $PID
echo "curl exit: $CURL_RC  server exit: $?"
# Expected: both 0
md5sum /tmp/webroot/shutdown_test.bin /tmp/dl_test
# Must match
```
**G4 — New connections rejected after shutdown signal**
```bash
./server 8080 /tmp/webroot &
PID=$!; sleep 0.3
kill -TERM $PID
sleep 0.1
# Try to connect after server closed server_fd
curl -s --connect-timeout 1 http://localhost:8080/ 2>&1
echo "Exit: $?"   # Must be non-zero (connection refused)
wait $PID
```
---
### 8.4 Shared State / Data Race Tests
**R1 — ThreadSanitizer clean under concurrent load**
```bash
gcc -Wall -std=c11 -O1 -g -fsanitize=thread \
    -o server_tsan server.c http_parse.c file_server.c thread_pool.c connection.c stats.c \
    -lpthread -D_GNU_SOURCE
./server_tsan 8080 /tmp/webroot &
TSAN_PID=$!
sleep 0.3
ab -n 5000 -c 50 -k http://localhost:8080/index.html > /dev/null 2>&1
kill -TERM $TSAN_PID
wait $TSAN_PID 2>&1 | grep -c "ThreadSanitizer"
# Expected: 0 (no TSan reports)
```
**R2 — `active_connections` stays non-negative**
```bash
# After load test, check stats via a signal-triggered dump (add SIGUSR1 handler to print stats)
# Or verify indirectly: check that server processes remain stable under load
ab -n 20000 -c 100 -k http://localhost:8080/index.html
# Expected: "Failed requests: 0" — if stats mutex is broken, server may deadlock and fail requests
```
**R3 — Access log line count matches request count**
```bash
rm -f access.log
ab -n 1000 -c 20 http://localhost:8080/index.html > /dev/null 2>&1
LINES=$(wc -l < access.log)
echo "Log lines: $LINES"   # Must be 1000
```
---
### 8.5 FD Leak Tests
**F1 — Zero FD leak after 10,000 sequential connections**
```bash
SERVER_PID=$(pgrep -n server)
BASELINE=$(ls /proc/$SERVER_PID/fd | wc -l)
ab -n 10000 -c 1 http://localhost:8080/index.html > /dev/null 2>&1
sleep 1   # allow keep-alive connections to time out
AFTER=$(ls /proc/$SERVER_PID/fd | wc -l)
echo "Baseline: $BASELINE  After: $AFTER"
[ "$AFTER" -le "$BASELINE" ] && echo "PASS" || echo "FAIL: FD count grew"
```
**F2 — Zero FD leak after 10,000 concurrent connections with errors**
```bash
SERVER_PID=$(pgrep -n server)
BASELINE=$(ls /proc/$SERVER_PID/fd | wc -l)
ab -n 3000 -c 30 http://localhost:8080/index.html > /dev/null 2>&1
ab -n 3000 -c 30 http://localhost:8080/doesnotexist > /dev/null 2>&1
ab -n 4000 -c 40 "http://localhost:8080/%2e%2e%2fetc" > /dev/null 2>&1
sleep 2
AFTER=$(ls /proc/$SERVER_PID/fd | wc -l)
echo "Baseline: $BASELINE  After: $AFTER"
[ "$AFTER" -le "$((BASELINE + 2))" ] && echo "PASS" || echo "FAIL"
```
---
### 8.6 Complete `test_concurrent.sh`

![Mutex Contention and Cache-Line False Sharing in server_stats_t](./diagrams/tdd-diag-38.svg)

```bash
#!/bin/bash
set -e
PORT=18083
DOC_ROOT=$(mktemp -d)
trap "rm -rf $DOC_ROOT; kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; exit" EXIT INT TERM
# Set up fixtures
echo '<html><body><h1>Test</h1></body></html>' > "$DOC_ROOT/index.html"
dd if=/dev/urandom of="$DOC_ROOT/big.bin" bs=1024 count=512 2>/dev/null   # 512KB
./server $PORT $DOC_ROOT 16 &    # port, doc_root, pool_size
SERVER_PID=$!
sleep 0.5
PASS=0; FAIL=0
check() {
    local desc="$1" expected="$2" actual="$3"
    if [ "$actual" = "$expected" ]; then
        echo "PASS: $desc"; ((PASS++))
    else
        echo "FAIL: $desc (expected='$expected' got='$actual')"; ((FAIL++))
    fi
}
BASE="http://localhost:$PORT"
# Basic concurrent serving
FAILED=$(ab -n 5000 -c 50 -k "$BASE/index.html" 2>&1 | grep "Failed requests" | awk '{print $3}')
check "5000 concurrent requests" "0" "$FAILED"
# Keep-alive improves throughput
RPS_KA=$(ab -n 5000 -c 50 -k "$BASE/index.html" 2>&1 | grep "Requests per second" | awk '{print $4}' | cut -d'.' -f1)
RPS_NO=$(ab -n 5000 -c 50    "$BASE/index.html" 2>&1 | grep "Requests per second" | awk '{print $4}' | cut -d'.' -f1)
# Just verify both are > 0
[ "$RPS_KA" -gt 0 ] && check "Keep-alive serves requests" "true" "true" || check "Keep-alive serves requests" "true" "false"
[ "$RPS_NO" -gt 0 ] && check "No-keepalive serves requests" "true" "true" || check "No-keepalive serves requests" "true" "false"
# Connection: close respected
CONN=$(curl -s -v -H "Connection: close" "$BASE/index.html" 2>&1 | grep -i "< connection:" | tr -d '\r\n')
echo "$CONN" | grep -qi "close" && check "Connection:close in response" "true" "true" || check "Connection:close in response" "true" "false"
# Binary file integrity under concurrency
for i in 1 2 3; do
    curl -s "$BASE/big.bin" -o "/tmp/concurrent_dl_$i.bin" &
done
wait
ORIG_MD5=$(md5sum "$DOC_ROOT/big.bin" | awk '{print $1}')
for i in 1 2 3; do
    DL_MD5=$(md5sum "/tmp/concurrent_dl_$i.bin" | awk '{print $1}')
    check "Binary integrity dl $i" "$ORIG_MD5" "$DL_MD5"
    rm "/tmp/concurrent_dl_$i.bin"
done
# FD leak check
BASELINE_FD=$(ls /proc/$SERVER_PID/fd | wc -l)
ab -n 5000 -c 100 "$BASE/index.html" > /dev/null 2>&1
ab -n 2000 -c 50  "$BASE/nosuchfile" > /dev/null 2>&1
sleep 2
AFTER_FD=$(ls /proc/$SERVER_PID/fd | wc -l)
[ "$AFTER_FD" -le "$((BASELINE_FD + 2))" ] && check "No FD leak" "true" "true" || check "No FD leak" "true" "false ($BASELINE_FD -> $AFTER_FD)"
# Graceful shutdown
curl -s "$BASE/big.bin" -o /tmp/shutdown_dl.bin &
CURL_PID=$!
sleep 0.05
kill -TERM $SERVER_PID
wait $CURL_PID; CURL_RC=$?
wait $SERVER_PID; SERVER_RC=$?
check "Curl completes after SIGTERM" "0" "$CURL_RC"
check "Server exit code 0" "0" "$SERVER_RC"
DL_MD5=$(md5sum /tmp/shutdown_dl.bin | awk '{print $1}')
check "Shutdown download integrity" "$ORIG_MD5" "$DL_MD5"
rm -f /tmp/shutdown_dl.bin
# Restart to avoid trap double-kill
./server $PORT $DOC_ROOT 16 &
SERVER_PID=$!
sleep 0.3
echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" = "0" ]
```
---
## 9. Performance Targets

![Data Race Example: counter++ Without Mutex](./diagrams/tdd-diag-39.svg)

| Operation | Target | Measurement Method |
|---|---|---|
| `thread_pool_init(16 threads)` | < 5 ms total | `clock_gettime(CLOCK_MONOTONIC)` before/after `thread_pool_init()` in `main()` |
| `thread_pool_enqueue()` (queue not full) | < 2 µs | 1M enqueue+dequeue pairs in tight loop with 2 threads; total / 1M |
| `should_keep_alive()` | < 100 ns | 1M calls; `clock_gettime` / 1M |
| `read_request_with_timeout()` — loopback, fresh connection | < 100 µs | `clock_gettime` around `read_request_with_timeout()` for 1000 loopback requests |
| Idle timeout fires within | ≤ timeout_seconds + 1s | Empirical: `time (connect and wait)` |
| Throughput: 16 workers, keep-alive, 1 KB file | ≥ 10,000 req/s | `ab -n 100000 -c 100 -k http://localhost:8080/small.html` |
| Throughput: 16 workers, no keep-alive, 1 KB file | ≥ 3,000 req/s | `ab -n 10000 -c 100 http://localhost:8080/small.html` |
| Throughput improvement: keep-alive vs. no keep-alive | ≥ 2× | Ratio of above two measurements |
| Graceful shutdown drain time (no active requests) | < 100 ms | `time kill -TERM $PID; wait $PID` |
| FD count after 10,000 sequential connections | Baseline ± 0 | `ls /proc/$PID/fd \| wc -l` before and after `ab -n 10000 -c 1` |
| Memory per idle connection (keep-alive, waiting for next request) | 0 heap bytes | `valgrind --tool=massif`; verify `malloc()` not called per-request in steady state |
| ThreadSanitizer false positives | 0 | `gcc -fsanitize=thread` build + `ab -n 5000 -c 50` |
**Benchmark script**:
```bash
# Throughput with keep-alive
ab -n 100000 -c 100 -k http://localhost:8080/index.html 2>&1 \
  | grep -E "Requests per second|Failed|Transfer rate"
# Throughput without keep-alive
ab -n 10000 -c 100 http://localhost:8080/index.html 2>&1 \
  | grep -E "Requests per second|Failed"
# Latency percentiles
ab -n 10000 -c 50 -k http://localhost:8080/index.html 2>&1 \
  | grep -A 10 "Percentage of the requests"
```

![Complete M4 Module Architecture: All Components and Their Interfaces](./diagrams/tdd-diag-40.svg)

---
## 10. Concurrency Specification
### Lock Ordering
To prevent deadlocks, all code that acquires multiple locks simultaneously must acquire them in this order:
```
1. pool->lock
2. g_stats.lock
3. g_log.lock
```
In practice, no current code path holds two of these locks simultaneously. This ordering is documented for future extensions. Violation of this ordering (acquiring `g_stats.lock` while holding `pool->lock`) would be a potential deadlock if another thread acquires them in the reverse order.
### Thread Roles and Their Locks
| Thread | Locks It Acquires | Duration |
|---|---|---|
| Accept loop (main thread) | `pool->lock` | < 500 ns per enqueue |
| Worker thread | `pool->lock`, `g_stats.lock`, `g_log.lock` | `pool->lock`: < 1 µs; others: < 500 ns |
| Signal handler | None (uses `atomic_store`) | < 10 ns |
### Shared Data Access Summary
| Data | Access Pattern | Protection |
|---|---|---|
| `pool->queue[]` | Accept loop writes `tail`; workers read `head` | `pool->lock` mutex |
| `pool->head`, `pool->tail`, `pool->count` | Both read and write | `pool->lock` mutex |
| `pool->shutdown` | Written by `thread_pool_shutdown()` under lock; read by workers under lock | `pool->lock` mutex |
| `shutdown_flag` | Written by signal handler; read by accept loop and workers | `atomic_int` (`_Atomic int`) |
| `g_stats.*` counters | Written and read by worker threads | `g_stats.lock` mutex |
| `g_log.file` | Written by worker threads | `g_log.lock` mutex |
| `g_doc_root` | Written once in `main()` before threads created; read by workers | No lock needed (write-once-read-many, visibility guaranteed by `pthread_create()` which acts as a memory barrier) |
### Spurious Wakeup Policy
All `pthread_cond_wait()` calls in this module use `while` loops:
```c
while (pool->count == 0 && !pool->shutdown) {
    pthread_cond_wait(&pool->not_empty, &pool->lock);
}
```
Never use `if`. Spurious wakeups are guaranteed to be handled safely by this pattern.
---
## Makefile Update
```makefile
CC      = gcc
CFLAGS  = -Wall -Wextra -Wpedantic -std=c11 -O2 -g \
          -Wframe-larger-than=131072 \
          -D_GNU_SOURCE
SRCS    = server.c http_parse.c file_server.c thread_pool.c connection.c stats.c
TARGET  = server
all: $(TARGET)
$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ -lpthread
tsan: $(SRCS)
	$(CC) $(CFLAGS) -O1 -fsanitize=thread -o server_tsan $^ -lpthread
asan: $(SRCS)
	$(CC) $(CFLAGS) -O1 -fsanitize=address,undefined -o server_asan $^ -lpthread
clean:
	rm -f $(TARGET) server_tsan server_asan
test: $(TARGET)
	bash test_concurrent.sh
bench: $(TARGET)
	ab -n 100000 -c 100 -k http://localhost:8080/index.html
.PHONY: all tsan asan clean test bench
```
`-lpthread` must appear **after** all `.c` files in the link command — the linker processes libraries left-to-right and resolves references in order. Placing `-lpthread` before the object files means the linker sees no unresolved `pthread_*` symbols yet and skips the library.
The `tsan` target uses `-O1` (not `-O2`) to reduce the number of optimizations that can confuse TSan's race detector. `-O0` is also acceptable for TSan runs. Never use `-O2 -fsanitize=thread` — aggressive optimizations can produce false positives or mask true races.
---
<!-- END_TDD_MOD -->


# Project Structure: HTTP Server (Basic)
## Directory Tree
```
http-server/
├── server.c                  # M1: socket lifecycle, accept loop, read_request(), send_all(), handle_client(), main()
├── http_parse.h              # M2: http_request_t, http_header_t, http_method_t, parse constants, function declarations
├── http_parse.c              # M2: http_parse_request(), parse_request_line(), parse_headers(), get_header(), classify_method(), url_decode_stub()
├── file_server.h             # M3: mime_entry_t, serve_file() and helper declarations, file-serving constants
├── file_server.c             # M3: serve_file(), url_decode(), build_full_path(), get_mime_type(), parse_http_date(), send_4xx/send_304/send_500 helpers, MIME_TABLE
├── thread_pool.h             # M4: work_item_t, thread_pool_t, thread pool function declarations, pool constants
├── thread_pool.c             # M4: thread_pool_init(), thread_pool_enqueue(), thread_pool_shutdown(), worker_thread()
├── connection.h              # M4: connection_t, handle_client_keep_alive(), read_request_with_timeout(), should_keep_alive() declarations
├── connection.c              # M4: handle_client_keep_alive(), read_request_with_timeout(), should_keep_alive()
├── stats.h                   # M4: server_stats_t, access_log_t, stats/log function declarations
├── stats.c                   # M4: stats_connection_opened/closed(), stats_request_served(), access_log_init/write/close()
├── Makefile                  # M1–M4: build rules, tsan/asan targets, -lpthread, -D_GNU_SOURCE
├── linker.ld                 # (not required — no bare-metal target; omit for Linux userspace)
├── test_basic.sh             # M1: manual/automated test script for socket and response correctness
├── test_parse.sh             # M2: automated test script for HTTP request parsing
├── test_file_server.sh       # M3: automated test script for file serving, security pipeline, conditional requests
├── test_concurrent.sh        # M4: automated test script for concurrency, keep-alive, graceful shutdown
├── parse_bench.c             # M2: microbenchmark binary for http_parse_request() latency
└── README.md                 # Project overview and build/run instructions
```
## Creation Order
1. **Project Setup** (15 min)
   - Create the `http-server/` directory
   - Create empty `Makefile` and `README.md`
2. **M1 — TCP Server & HTTP Response** (2–3 h)
   - `server.c` — implement in phases:
     - Phase 1: `create_server_socket()` with `SO_REUSEADDR`, `bind()`, `listen()`
     - Phase 2: accept loop with `SIGPIPE` suppression and `close(client_fd)`
     - Phase 3: `read_request()` partial-read loop with `\r\n\r\n` delimiter
     - Phase 4: `send_all()` with `MSG_NOSIGNAL`; hardcoded `HTTP_RESPONSE`; `handle_client()`
     - Phase 5: `test_basic.sh`
   - `Makefile` — initial rules: `gcc -Wall -Wextra -std=c11 -O2 -g -o server server.c`
   - `test_basic.sh`
3. **M2 — HTTP Request Parsing** (3–4 h)
   - `http_parse.h` — all structs, constants, error codes, function declarations (with include guards)
   - `http_parse.c` — implement in phases:
     - Phase 1: stub bodies; verify `sizeof(http_request_t)` ≈ 43,064 bytes
     - Phase 2: `parse_request_line()` using `memchr()`
     - Phase 3: `parse_headers()` — line scan, CRLF/LF, obs-fold, OWS strip, `tolower()` cast
     - Phase 4: `get_header()`, `classify_method()`, `http_parse_request()` orchestrator
     - Phase 5: error response constants (`HTTP_400`, `HTTP_414`, `HTTP_501`) with `_Static_assert`
     - Phase 6: HEAD method branch in `handle_client()` inside `server.c`
   - `server.c` — update `handle_client()` to call `http_parse_request()` and route errors
   - `Makefile` — add `http_parse.c` to sources; add `parse_bench` target; add `-Wframe-larger-than=65536`
   - `parse_bench.c`
   - `test_parse.sh`
4. **M3 — Static File Serving** (3–5 h)
   - `file_server.h` — constants (`DOC_ROOT_MAX`, `FILE_READ_BUF_SIZE`, etc.), `mime_entry_t`, all declarations
   - `file_server.c` — implement in phases:
     - Phase 1: `url_decode()` with null-byte and malformed-hex rejection
     - Phase 2: `build_full_path()`
     - Phase 3: `serve_file()` Stages 1–4 — `realpath()` + prefix+separator containment check
     - Phase 4: Stage 4b — directory detection, `/index.html` append, re-`realpath()`, re-check
     - Phase 5: `get_mime_type()` with `MIME_TABLE` (18 entries + sentinel)
     - Phase 6: `stat()` for `Content-Length` / `Last-Modified`; `strftime()` HTTP-date formatting; Stage 7 header build
     - Phase 7: `parse_http_date()` (`strptime()` + `timegm()`); `send_304()`; Stage 5 conditional check
     - Phase 8: `open()` + `read()` + `send_all()` file streaming loop; `send_body` flag for HEAD
     - Phase 9: `send_404()`, `send_403()`, `send_500()` with `_Static_assert` on `Content-Length`
   - `server.c` — update `handle_client()` to call `serve_file()`; add `g_doc_root` global
   - `Makefile` — add `file_server.c`; add `-D_GNU_SOURCE`; bump `-Wframe-larger-than=131072`
   - `test_file_server.sh`
5. **M4 — Concurrent Connections** (4–6 h)
   - `thread_pool.h` — `work_item_t`, `thread_pool_t`, constants, declarations
   - `thread_pool.c` — implement in phases:
     - Phase 2: struct size verification stubs
     - Phase 3: `worker_thread()` with `while`/`pthread_cond_wait`; `thread_pool_init()` with error rollback
     - Phase 4: `thread_pool_enqueue()` (non-blocking, returns -1 when full); `thread_pool_shutdown()`
   - `connection.h` — `connection_t`, function declarations
   - `connection.c` — implement in phases:
     - Phase 1: `handle_client_keep_alive()` as single-request pass-through (thread-per-connection model)
     - Phase 5: full keep-alive loop with `should_keep_alive()` and `MAX_REQUESTS_PER_CONNECTION` guard
     - Phase 6: `read_request_with_timeout()` using `select()` with `EINTR` retry and `shutdown_flag` check
   - `stats.h` — `server_stats_t`, `access_log_t`, function declarations
   - `stats.c` — `stats_connection_opened/closed()`, `stats_request_served()`, `access_log_init/write/close()`
   - `server.c` — rewrite `main()`:
     - Phase 1: thread-per-connection accept loop (temporary)
     - Phase 4: switch to `thread_pool_enqueue()` with 503 on queue-full
     - Phase 7: `signal_handler()`, `atomic_int shutdown_flag`; post-loop `close(server_fd)` + `thread_pool_shutdown()`
     - Phase 8: integrate `g_stats` and `g_log`
   - `Makefile` — add `thread_pool.c connection.c stats.c`; add `-lpthread`; add `tsan` and `asan` targets
   - `test_concurrent.sh`
## File Count Summary
| Category | Count |
|---|---|
| Core source files (`.c`) | 6 (`server.c`, `http_parse.c`, `file_server.c`, `thread_pool.c`, `connection.c`, `stats.c`) |
| Header files (`.h`) | 4 (`http_parse.h`, `file_server.h`, `thread_pool.h`, `connection.h`, `stats.h`) |
| Build / config | 1 (`Makefile`) |
| Test scripts | 4 (`test_basic.sh`, `test_parse.sh`, `test_file_server.sh`, `test_concurrent.sh`) |
| Benchmark source | 1 (`parse_bench.c`) |
| Documentation | 1 (`README.md`) |
| **Total files** | **17** |
| **Directories** | **1** (flat layout — no subdirectories) |
| **Estimated lines of code** | **~2,500–3,200** (excluding test scripts and comments) |
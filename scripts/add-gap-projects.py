#!/usr/bin/env python3
"""Add missing projects for gaps in coverage."""

import yaml
from pathlib import Path

# New projects to fill gaps
GAP_PROJECTS = {
    # Systems domain - add to existing
    "systems": {
        "advanced": [
            {
                "id": "filesystem",
                "name": "Filesystem Implementation",
                "description": "Simple filesystem with inodes, directories, journaling",
                "detailed": True
            },
            {
                "id": "reverse-proxy",
                "name": "Reverse Proxy",
                "description": "Nginx-like proxy with load balancing, caching, SSL termination",
                "detailed": True
            },
        ],
        "expert": [
            {
                "id": "bootloader",
                "name": "x86 Bootloader",
                "description": "Boot from BIOS/UEFI, load kernel, switch to protected mode",
                "detailed": True
            },
            {
                "id": "device-driver",
                "name": "Linux Kernel Module",
                "description": "Character device driver with ioctl, proc interface",
                "detailed": True
            },
            {
                "id": "build-quic",
                "name": "Build Your Own QUIC",
                "description": "QUIC protocol with UDP, TLS 1.3, multiplexing",
                "detailed": True
            },
        ],
    },
    # Security domain
    "security": {
        "intermediate": [
            {
                "id": "sandbox",
                "name": "Process Sandbox",
                "description": "Sandboxing with seccomp, namespaces, capabilities",
                "detailed": True
            },
        ],
        "advanced": [
            {
                "id": "fuzzer",
                "name": "Fuzzing Framework",
                "description": "Coverage-guided fuzzer like AFL with mutation strategies",
                "detailed": True
            },
            {
                "id": "vulnerability-scanner",
                "name": "Vulnerability Scanner",
                "description": "Network/web vulnerability scanner with CVE detection",
                "detailed": True
            },
        ],
    },
    # Specialized domain - performance projects
    "specialized": {
        "intermediate": [
            {
                "id": "profiler",
                "name": "CPU/Memory Profiler",
                "description": "Sampling profiler with flame graphs, memory tracking",
                "detailed": True
            },
        ],
        "advanced": [
            {
                "id": "lock-free-structures",
                "name": "Lock-free Data Structures",
                "description": "Lock-free queue, stack, hashmap with CAS operations",
                "detailed": True
            },
            {
                "id": "simd-library",
                "name": "SIMD Optimization Library",
                "description": "SIMD-accelerated string, math, image operations",
                "detailed": True
            },
            {
                "id": "cache-optimized-structures",
                "name": "Cache-Optimized Data Structures",
                "description": "Cache-oblivious algorithms, B+ trees, memory layouts",
                "detailed": True
            },
        ],
        "expert": [
            {
                "id": "build-vpn",
                "name": "Build Your Own VPN",
                "description": "VPN with TUN/TAP, encryption, key exchange",
                "detailed": True
            },
        ],
    },
}

# Detailed milestones for expert_projects
DETAILED_PROJECTS = {
    "filesystem": {
        "name": "Filesystem Implementation",
        "description": "Build a simple filesystem with inodes, directories, and journaling - understand how data is organized on disk",
        "category": "Systems",
        "difficulty": "advanced",
        "estimated_hours": 60,
        "skills": [
            "Block devices",
            "Inode structures",
            "Directory entries",
            "Journaling",
            "FUSE interface",
            "Caching"
        ],
        "prerequisites": ["File I/O", "Data structures", "C programming"],
        "learning_outcomes": [
            "Understand filesystem internals and on-disk layouts",
            "Implement inode-based file management",
            "Build directory tree operations",
            "Handle crash recovery with journaling"
        ],
        "milestones": [
            {
                "name": "Block Layer",
                "description": "Raw block device read/write operations",
                "skills": ["Block I/O", "Superblock", "Bitmap allocation"],
                "deliverables": [
                    "Block device abstraction",
                    "Superblock with filesystem metadata",
                    "Block bitmap for free space tracking",
                    "Block allocation/deallocation",
                    "Disk image file backend",
                    "Block caching layer"
                ],
                "hints": {
                    "level1": "Basic block device:\n```c\n#define BLOCK_SIZE 4096\n\nstruct block_device {\n    int fd;\n    size_t num_blocks;\n};\n\nint block_read(struct block_device *dev, uint64_t block_num, void *buf) {\n    off_t offset = block_num * BLOCK_SIZE;\n    return pread(dev->fd, buf, BLOCK_SIZE, offset);\n}\n\nint block_write(struct block_device *dev, uint64_t block_num, const void *buf) {\n    off_t offset = block_num * BLOCK_SIZE;\n    return pwrite(dev->fd, buf, BLOCK_SIZE, offset);\n}\n```",
                    "level2": "Superblock structure:\n```c\nstruct superblock {\n    uint32_t magic;           // Filesystem magic number\n    uint32_t block_size;      // Block size in bytes\n    uint64_t total_blocks;    // Total blocks in filesystem\n    uint64_t inode_count;     // Total inodes\n    uint64_t free_blocks;     // Free block count\n    uint64_t free_inodes;     // Free inode count\n    uint64_t block_bitmap;    // Block bitmap start\n    uint64_t inode_bitmap;    // Inode bitmap start\n    uint64_t inode_table;     // Inode table start\n    uint64_t data_blocks;     // Data blocks start\n};\n```",
                    "level3": "Bitmap operations:\n```c\nint bitmap_alloc(uint8_t *bitmap, size_t size) {\n    for (size_t i = 0; i < size; i++) {\n        if (bitmap[i] != 0xFF) {\n            for (int bit = 0; bit < 8; bit++) {\n                if (!(bitmap[i] & (1 << bit))) {\n                    bitmap[i] |= (1 << bit);\n                    return i * 8 + bit;\n                }\n            }\n        }\n    }\n    return -1;  // No free blocks\n}\n\nvoid bitmap_free(uint8_t *bitmap, int index) {\n    bitmap[index / 8] &= ~(1 << (index % 8));\n}\n```"
                }
            },
            {
                "name": "Inode Management",
                "description": "Inode structure for file metadata",
                "skills": ["Inode structure", "Direct/indirect blocks", "Permissions"],
                "deliverables": [
                    "Inode structure with metadata",
                    "Direct block pointers",
                    "Indirect block pointers",
                    "Inode allocation/free",
                    "Inode read/write",
                    "File type and permissions"
                ],
                "hints": {
                    "level1": "Inode structure:\n```c\n#define DIRECT_BLOCKS 12\n\nstruct inode {\n    uint16_t mode;           // File type and permissions\n    uint16_t uid;            // Owner user ID\n    uint16_t gid;            // Owner group ID\n    uint32_t size;           // File size in bytes\n    uint32_t atime;          // Access time\n    uint32_t mtime;          // Modification time\n    uint32_t ctime;          // Change time\n    uint32_t links_count;    // Hard link count\n    uint32_t blocks;         // Number of blocks\n    uint32_t direct[DIRECT_BLOCKS];  // Direct block pointers\n    uint32_t indirect;       // Single indirect\n    uint32_t double_indirect; // Double indirect\n};\n```",
                    "level2": "Get block number for file offset:\n```c\nuint32_t inode_get_block(struct inode *inode, uint32_t file_block) {\n    if (file_block < DIRECT_BLOCKS) {\n        return inode->direct[file_block];\n    }\n    \n    file_block -= DIRECT_BLOCKS;\n    uint32_t ptrs_per_block = BLOCK_SIZE / sizeof(uint32_t);\n    \n    if (file_block < ptrs_per_block) {\n        // Single indirect\n        uint32_t *indirect = read_block(inode->indirect);\n        return indirect[file_block];\n    }\n    \n    file_block -= ptrs_per_block;\n    // Double indirect\n    uint32_t *dbl = read_block(inode->double_indirect);\n    uint32_t *indirect = read_block(dbl[file_block / ptrs_per_block]);\n    return indirect[file_block % ptrs_per_block];\n}\n```",
                    "level3": "Allocate blocks for file growth:\n```c\nint inode_grow(struct inode *inode, size_t new_size) {\n    size_t current_blocks = (inode->size + BLOCK_SIZE - 1) / BLOCK_SIZE;\n    size_t needed_blocks = (new_size + BLOCK_SIZE - 1) / BLOCK_SIZE;\n    \n    for (size_t i = current_blocks; i < needed_blocks; i++) {\n        uint32_t new_block = alloc_block();\n        if (new_block == 0) return -ENOSPC;\n        \n        if (!inode_set_block(inode, i, new_block)) {\n            // May need to allocate indirect blocks\n            if (!alloc_indirect_block(inode, i)) {\n                return -ENOSPC;\n            }\n            inode_set_block(inode, i, new_block);\n        }\n    }\n    inode->size = new_size;\n    return 0;\n}\n```"
                }
            },
            {
                "name": "Directory Operations",
                "description": "Directory entries and path resolution",
                "skills": ["Directory entries", "Path parsing", "Name lookup"],
                "deliverables": [
                    "Directory entry structure",
                    "Add/remove directory entries",
                    "Path to inode resolution",
                    "Create/delete files",
                    "Create/remove directories",
                    "Rename operations"
                ],
                "hints": {
                    "level1": "Directory entry:\n```c\n#define MAX_NAME_LEN 255\n\nstruct dir_entry {\n    uint32_t inode;          // Inode number (0 = deleted)\n    uint16_t rec_len;        // Total entry length\n    uint8_t name_len;        // Name length\n    uint8_t file_type;       // File type\n    char name[MAX_NAME_LEN]; // File name\n};\n\n// File types\n#define FT_UNKNOWN  0\n#define FT_REG_FILE 1\n#define FT_DIR      2\n#define FT_SYMLINK  7\n```",
                    "level2": "Lookup name in directory:\n```c\nstruct inode *dir_lookup(struct inode *dir, const char *name) {\n    size_t offset = 0;\n    while (offset < dir->size) {\n        struct dir_entry *entry = read_dir_entry(dir, offset);\n        \n        if (entry->inode != 0 && \n            entry->name_len == strlen(name) &&\n            strncmp(entry->name, name, entry->name_len) == 0) {\n            return read_inode(entry->inode);\n        }\n        \n        offset += entry->rec_len;\n    }\n    return NULL;  // Not found\n}\n```",
                    "level3": "Path resolution:\n```c\nstruct inode *resolve_path(const char *path) {\n    struct inode *inode = read_inode(ROOT_INODE);\n    \n    char *path_copy = strdup(path);\n    char *token = strtok(path_copy, \"/\");\n    \n    while (token != NULL) {\n        if (!S_ISDIR(inode->mode)) {\n            free(path_copy);\n            return NULL;  // Not a directory\n        }\n        \n        struct inode *next = dir_lookup(inode, token);\n        if (next == NULL) {\n            free(path_copy);\n            return NULL;  // Not found\n        }\n        \n        inode = next;\n        token = strtok(NULL, \"/\");\n    }\n    \n    free(path_copy);\n    return inode;\n}\n```"
                }
            },
            {
                "name": "File Operations",
                "description": "Read, write, truncate files",
                "skills": ["File read/write", "Truncation", "Sparse files"],
                "deliverables": [
                    "Read file data",
                    "Write file data",
                    "Truncate file",
                    "Append to file",
                    "Sparse file support",
                    "File hole handling"
                ],
                "hints": {
                    "level1": "Read file:\n```c\nssize_t file_read(struct inode *inode, void *buf, size_t size, off_t offset) {\n    if (offset >= inode->size) return 0;\n    if (offset + size > inode->size) {\n        size = inode->size - offset;\n    }\n    \n    size_t bytes_read = 0;\n    while (bytes_read < size) {\n        uint32_t block_num = (offset + bytes_read) / BLOCK_SIZE;\n        uint32_t block_offset = (offset + bytes_read) % BLOCK_SIZE;\n        uint32_t to_read = min(BLOCK_SIZE - block_offset, size - bytes_read);\n        \n        uint32_t disk_block = inode_get_block(inode, block_num);\n        char *block_data = read_block(disk_block);\n        memcpy(buf + bytes_read, block_data + block_offset, to_read);\n        \n        bytes_read += to_read;\n    }\n    return bytes_read;\n}\n```",
                    "level2": "Write file:\n```c\nssize_t file_write(struct inode *inode, const void *buf, size_t size, off_t offset) {\n    // Grow file if needed\n    if (offset + size > inode->size) {\n        int err = inode_grow(inode, offset + size);\n        if (err) return err;\n    }\n    \n    size_t bytes_written = 0;\n    while (bytes_written < size) {\n        uint32_t block_num = (offset + bytes_written) / BLOCK_SIZE;\n        uint32_t block_offset = (offset + bytes_written) % BLOCK_SIZE;\n        uint32_t to_write = min(BLOCK_SIZE - block_offset, size - bytes_written);\n        \n        uint32_t disk_block = inode_get_block(inode, block_num);\n        char *block_data = read_block(disk_block);\n        memcpy(block_data + block_offset, buf + bytes_written, to_write);\n        write_block(disk_block, block_data);\n        \n        bytes_written += to_write;\n    }\n    \n    inode->mtime = time(NULL);\n    write_inode(inode);\n    return bytes_written;\n}\n```",
                    "level3": "Truncate file:\n```c\nint file_truncate(struct inode *inode, off_t length) {\n    if (length > inode->size) {\n        // Extend with zeros (sparse)\n        return inode_grow(inode, length);\n    }\n    \n    // Free blocks beyond new size\n    size_t new_blocks = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;\n    size_t old_blocks = (inode->size + BLOCK_SIZE - 1) / BLOCK_SIZE;\n    \n    for (size_t i = new_blocks; i < old_blocks; i++) {\n        uint32_t block = inode_get_block(inode, i);\n        if (block != 0) {\n            free_block(block);\n            inode_set_block(inode, i, 0);\n        }\n    }\n    \n    // Free indirect blocks if needed\n    shrink_indirect_blocks(inode, new_blocks);\n    \n    inode->size = length;\n    inode->mtime = time(NULL);\n    write_inode(inode);\n    return 0;\n}\n```"
                }
            },
            {
                "name": "FUSE Interface",
                "description": "Mount filesystem via FUSE",
                "skills": ["FUSE API", "VFS operations", "Mount/unmount"],
                "deliverables": [
                    "FUSE operation callbacks",
                    "getattr implementation",
                    "readdir implementation",
                    "open/read/write/release",
                    "mkdir/rmdir",
                    "Mount and unmount"
                ],
                "hints": {
                    "level1": "FUSE operations struct:\n```c\n#define FUSE_USE_VERSION 31\n#include <fuse3/fuse.h>\n\nstatic struct fuse_operations myfs_ops = {\n    .getattr  = myfs_getattr,\n    .readdir  = myfs_readdir,\n    .open     = myfs_open,\n    .read     = myfs_read,\n    .write    = myfs_write,\n    .mkdir    = myfs_mkdir,\n    .rmdir    = myfs_rmdir,\n    .unlink   = myfs_unlink,\n    .create   = myfs_create,\n    .truncate = myfs_truncate,\n};\n\nint main(int argc, char *argv[]) {\n    return fuse_main(argc, argv, &myfs_ops, NULL);\n}\n```",
                    "level2": "Implement getattr:\n```c\nstatic int myfs_getattr(const char *path, struct stat *stbuf,\n                        struct fuse_file_info *fi) {\n    memset(stbuf, 0, sizeof(struct stat));\n    \n    struct inode *inode = resolve_path(path);\n    if (!inode) return -ENOENT;\n    \n    stbuf->st_ino = inode->ino;\n    stbuf->st_mode = inode->mode;\n    stbuf->st_nlink = inode->links_count;\n    stbuf->st_uid = inode->uid;\n    stbuf->st_gid = inode->gid;\n    stbuf->st_size = inode->size;\n    stbuf->st_blocks = inode->blocks;\n    stbuf->st_atime = inode->atime;\n    stbuf->st_mtime = inode->mtime;\n    stbuf->st_ctime = inode->ctime;\n    \n    return 0;\n}\n```",
                    "level3": "Implement readdir:\n```c\nstatic int myfs_readdir(const char *path, void *buf, fuse_fill_dir_t filler,\n                        off_t offset, struct fuse_file_info *fi,\n                        enum fuse_readdir_flags flags) {\n    struct inode *dir = resolve_path(path);\n    if (!dir) return -ENOENT;\n    if (!S_ISDIR(dir->mode)) return -ENOTDIR;\n    \n    filler(buf, \".\", NULL, 0, 0);\n    filler(buf, \"..\", NULL, 0, 0);\n    \n    size_t pos = 0;\n    while (pos < dir->size) {\n        struct dir_entry *entry = read_dir_entry(dir, pos);\n        if (entry->inode != 0) {\n            char name[256];\n            strncpy(name, entry->name, entry->name_len);\n            name[entry->name_len] = '\\0';\n            filler(buf, name, NULL, 0, 0);\n        }\n        pos += entry->rec_len;\n    }\n    \n    return 0;\n}\n```"
                }
            }
        ]
    },
    "reverse-proxy": {
        "name": "Reverse Proxy",
        "description": "Build an Nginx-like reverse proxy with load balancing, caching, and SSL termination",
        "category": "Networking",
        "difficulty": "advanced",
        "estimated_hours": 55,
        "skills": [
            "HTTP parsing",
            "Load balancing",
            "Connection pooling",
            "SSL/TLS",
            "Caching",
            "Health checks"
        ],
        "prerequisites": ["HTTP basics", "Networking", "Async I/O"],
        "learning_outcomes": [
            "Understand reverse proxy architecture",
            "Implement load balancing algorithms",
            "Build connection pooling for performance",
            "Handle SSL termination"
        ],
        "milestones": [
            {
                "name": "HTTP Proxy Core",
                "description": "Basic HTTP request forwarding",
                "skills": ["HTTP parsing", "Request forwarding", "Response handling"],
                "deliverables": [
                    "Accept client connections",
                    "Parse HTTP requests",
                    "Forward to upstream",
                    "Return response to client",
                    "Header manipulation",
                    "Error handling"
                ],
                "hints": {
                    "level1": "Basic proxy loop:\n```python\nimport asyncio\n\nasync def handle_client(reader, writer):\n    # Read request\n    request = await read_http_request(reader)\n    \n    # Connect to upstream\n    upstream_reader, upstream_writer = await asyncio.open_connection(\n        'backend.local', 8080\n    )\n    \n    # Forward request\n    upstream_writer.write(request.raw)\n    await upstream_writer.drain()\n    \n    # Read and forward response\n    response = await read_http_response(upstream_reader)\n    writer.write(response.raw)\n    await writer.drain()\n    \n    writer.close()\n    upstream_writer.close()\n```",
                    "level2": "Add headers:\n```python\ndef modify_request(request, client_addr):\n    # Add X-Forwarded headers\n    request.headers['X-Forwarded-For'] = client_addr[0]\n    request.headers['X-Forwarded-Proto'] = 'https' if ssl else 'http'\n    request.headers['X-Real-IP'] = client_addr[0]\n    \n    # Remove hop-by-hop headers\n    for header in ['Connection', 'Keep-Alive', 'Transfer-Encoding']:\n        request.headers.pop(header, None)\n    \n    return request\n```",
                    "level3": "Streaming for large bodies:\n```python\nasync def proxy_body(source, dest, content_length):\n    remaining = content_length\n    while remaining > 0:\n        chunk_size = min(remaining, 64 * 1024)\n        chunk = await source.read(chunk_size)\n        if not chunk:\n            break\n        dest.write(chunk)\n        await dest.drain()\n        remaining -= len(chunk)\n```"
                }
            },
            {
                "name": "Load Balancing",
                "description": "Distribute requests across backends",
                "skills": ["Load balancing algorithms", "Backend selection", "Weighted distribution"],
                "deliverables": [
                    "Round-robin balancing",
                    "Least connections",
                    "Weighted round-robin",
                    "IP hash (sticky sessions)",
                    "Backend configuration",
                    "Dynamic backend updates"
                ],
                "hints": {
                    "level1": "Round-robin:\n```python\nclass RoundRobinBalancer:\n    def __init__(self, backends):\n        self.backends = backends\n        self.index = 0\n    \n    def next(self):\n        backend = self.backends[self.index]\n        self.index = (self.index + 1) % len(self.backends)\n        return backend\n```",
                    "level2": "Least connections:\n```python\nclass LeastConnectionsBalancer:\n    def __init__(self, backends):\n        self.backends = {b: 0 for b in backends}\n    \n    def next(self):\n        backend = min(self.backends, key=self.backends.get)\n        self.backends[backend] += 1\n        return backend\n    \n    def release(self, backend):\n        self.backends[backend] -= 1\n```",
                    "level3": "IP hash for sticky sessions:\n```python\nclass IPHashBalancer:\n    def __init__(self, backends):\n        self.backends = backends\n    \n    def next(self, client_ip):\n        # Consistent hashing\n        hash_val = hash(client_ip)\n        index = hash_val % len(self.backends)\n        return self.backends[index]\n```"
                }
            },
            {
                "name": "Connection Pooling",
                "description": "Reuse connections to backends",
                "skills": ["Connection pools", "Keep-alive", "Pool management"],
                "deliverables": [
                    "Connection pool per backend",
                    "Keep-alive connections",
                    "Pool size limits",
                    "Connection timeout",
                    "Health check integration",
                    "Pool metrics"
                ],
                "hints": {
                    "level1": "Simple connection pool:\n```python\nclass ConnectionPool:\n    def __init__(self, host, port, max_size=10):\n        self.host = host\n        self.port = port\n        self.max_size = max_size\n        self.pool = asyncio.Queue(maxsize=max_size)\n        self.size = 0\n    \n    async def acquire(self):\n        try:\n            return self.pool.get_nowait()\n        except asyncio.QueueEmpty:\n            if self.size < self.max_size:\n                self.size += 1\n                return await self.create_connection()\n            return await self.pool.get()\n    \n    async def release(self, conn):\n        if conn.is_healthy():\n            await self.pool.put(conn)\n        else:\n            self.size -= 1\n            conn.close()\n```",
                    "level2": "Connection wrapper with keep-alive:\n```python\nclass PooledConnection:\n    def __init__(self, reader, writer, pool):\n        self.reader = reader\n        self.writer = writer\n        self.pool = pool\n        self.last_used = time.time()\n    \n    def is_healthy(self):\n        # Check if connection is still alive\n        if time.time() - self.last_used > 60:\n            return False\n        return not self.reader.at_eof()\n    \n    async def __aenter__(self):\n        return self\n    \n    async def __aexit__(self, *args):\n        self.last_used = time.time()\n        await self.pool.release(self)\n```",
                    "level3": "Pool with health checks:\n```python\nclass HealthyConnectionPool(ConnectionPool):\n    async def health_check_loop(self):\n        while True:\n            await asyncio.sleep(30)\n            # Check all idle connections\n            healthy = []\n            while not self.pool.empty():\n                conn = await self.pool.get()\n                if conn.is_healthy():\n                    healthy.append(conn)\n                else:\n                    conn.close()\n                    self.size -= 1\n            for conn in healthy:\n                await self.pool.put(conn)\n```"
                }
            },
            {
                "name": "Caching",
                "description": "Cache responses for performance",
                "skills": ["Cache storage", "Cache invalidation", "Cache headers"],
                "deliverables": [
                    "In-memory cache",
                    "Cache key generation",
                    "TTL and max-age handling",
                    "Cache-Control header parsing",
                    "Conditional requests (ETag, If-Modified-Since)",
                    "Cache bypass rules"
                ],
                "hints": {
                    "level1": "Simple cache:\n```python\nclass ResponseCache:\n    def __init__(self, max_size=1000):\n        self.cache = {}\n        self.max_size = max_size\n    \n    def key(self, request):\n        return f\"{request.method}:{request.host}:{request.path}\"\n    \n    def get(self, request):\n        k = self.key(request)\n        if k in self.cache:\n            entry = self.cache[k]\n            if entry.is_valid():\n                return entry.response\n            del self.cache[k]\n        return None\n    \n    def put(self, request, response, ttl):\n        if len(self.cache) >= self.max_size:\n            self.evict()\n        self.cache[self.key(request)] = CacheEntry(response, ttl)\n```",
                    "level2": "Parse Cache-Control:\n```python\ndef parse_cache_control(response):\n    cc = response.headers.get('Cache-Control', '')\n    directives = {}\n    for part in cc.split(','):\n        part = part.strip()\n        if '=' in part:\n            key, value = part.split('=', 1)\n            directives[key] = value\n        else:\n            directives[part] = True\n    return directives\n\ndef is_cacheable(response):\n    cc = parse_cache_control(response)\n    if cc.get('no-store') or cc.get('private'):\n        return False\n    if response.status_code not in [200, 203, 204, 206, 300, 301, 404, 501]:\n        return False\n    return True\n```",
                    "level3": "Conditional requests:\n```python\nasync def serve_with_cache(request, cache):\n    cached = cache.get(request)\n    \n    if cached:\n        # Check if client has valid cached copy\n        if request.headers.get('If-None-Match') == cached.etag:\n            return Response(status=304)\n        return cached.response\n    \n    # Fetch from upstream\n    response = await fetch_upstream(request)\n    \n    if is_cacheable(response):\n        ttl = get_ttl(response)\n        cache.put(request, response, ttl)\n    \n    return response\n```"
                }
            },
            {
                "name": "SSL Termination",
                "description": "Handle HTTPS connections",
                "skills": ["TLS/SSL", "Certificate management", "SNI"],
                "deliverables": [
                    "SSL context setup",
                    "Certificate loading",
                    "SNI support",
                    "TLS version configuration",
                    "Cipher suite selection",
                    "HTTP to HTTPS redirect"
                ],
                "hints": {
                    "level1": "SSL server setup:\n```python\nimport ssl\n\ndef create_ssl_context(cert_file, key_file):\n    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)\n    ctx.load_cert_chain(cert_file, key_file)\n    ctx.minimum_version = ssl.TLSVersion.TLSv1_2\n    return ctx\n\nasync def start_https_server(host, port, cert, key):\n    ssl_ctx = create_ssl_context(cert, key)\n    server = await asyncio.start_server(\n        handle_client, host, port, ssl=ssl_ctx\n    )\n    return server\n```",
                    "level2": "SNI for multiple domains:\n```python\nclass SNIContext:\n    def __init__(self):\n        self.contexts = {}  # domain -> ssl_context\n    \n    def add_certificate(self, domain, cert, key):\n        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)\n        ctx.load_cert_chain(cert, key)\n        self.contexts[domain] = ctx\n    \n    def sni_callback(self, ssl_obj, server_name, original_ctx):\n        if server_name in self.contexts:\n            ssl_obj.context = self.contexts[server_name]\n\n# Usage\ndefault_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)\ndefault_ctx.sni_callback = sni_handler.sni_callback\n```",
                    "level3": "HTTPS redirect:\n```python\nasync def handle_http(reader, writer):\n    request = await read_http_request(reader)\n    \n    # Redirect to HTTPS\n    host = request.headers.get('Host', 'localhost')\n    location = f'https://{host}{request.path}'\n    \n    response = f'''HTTP/1.1 301 Moved Permanently\r\nLocation: {location}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n'''\n    writer.write(response.encode())\n    await writer.drain()\n    writer.close()\n```"
                }
            }
        ]
    },
    "lock-free-structures": {
        "name": "Lock-free Data Structures",
        "description": "Build lock-free concurrent data structures using atomic operations and CAS",
        "category": "Performance",
        "difficulty": "advanced",
        "estimated_hours": 50,
        "skills": [
            "Atomic operations",
            "Compare-and-swap",
            "Memory ordering",
            "ABA problem",
            "Hazard pointers",
            "Lock-free algorithms"
        ],
        "prerequisites": ["Concurrency basics", "Memory model", "C/C++ or Rust"],
        "learning_outcomes": [
            "Understand lock-free programming principles",
            "Implement CAS-based algorithms",
            "Handle ABA problem and memory reclamation",
            "Build high-performance concurrent structures"
        ],
        "milestones": [
            {
                "name": "Atomic Operations",
                "description": "Understanding atomics and memory ordering",
                "skills": ["Atomics", "Memory ordering", "CAS"],
                "deliverables": [
                    "Atomic load/store",
                    "Compare-and-swap (CAS)",
                    "Fetch-and-add",
                    "Memory ordering (relaxed, acquire, release, seq_cst)",
                    "Compiler and CPU memory barriers",
                    "Atomic reference counting"
                ],
                "hints": {
                    "level1": "Basic CAS loop:\n```c\n#include <stdatomic.h>\n\ntypedef struct {\n    _Atomic int value;\n} AtomicInt;\n\nvoid atomic_increment(AtomicInt *a) {\n    int old = atomic_load(&a->value);\n    while (!atomic_compare_exchange_weak(&a->value, &old, old + 1)) {\n        // CAS failed, old is updated with current value\n        // Loop and retry\n    }\n}\n```",
                    "level2": "Memory ordering:\n```c\n// Producer-consumer with acquire-release\n_Atomic int data;\n_Atomic int flag;\n\n// Producer\nvoid produce(int value) {\n    atomic_store_explicit(&data, value, memory_order_relaxed);\n    atomic_store_explicit(&flag, 1, memory_order_release);  // Release\n}\n\n// Consumer\nint consume() {\n    while (atomic_load_explicit(&flag, memory_order_acquire) == 0);  // Acquire\n    return atomic_load_explicit(&data, memory_order_relaxed);\n}\n```",
                    "level3": "Atomic reference counting:\n```c\ntypedef struct {\n    _Atomic int ref_count;\n    void *data;\n} RefCounted;\n\nvoid retain(RefCounted *obj) {\n    atomic_fetch_add(&obj->ref_count, 1);\n}\n\nvoid release(RefCounted *obj) {\n    if (atomic_fetch_sub(&obj->ref_count, 1) == 1) {\n        // We were the last reference\n        free(obj->data);\n        free(obj);\n    }\n}\n```"
                }
            },
            {
                "name": "Lock-free Stack",
                "description": "Treiber stack implementation",
                "skills": ["Treiber stack", "CAS-based push/pop", "ABA handling"],
                "deliverables": [
                    "Node structure",
                    "Lock-free push",
                    "Lock-free pop",
                    "ABA problem demonstration",
                    "Tagged pointers solution",
                    "Stack traversal"
                ],
                "hints": {
                    "level1": "Treiber stack structure:\n```c\ntypedef struct Node {\n    void *data;\n    struct Node *next;\n} Node;\n\ntypedef struct {\n    _Atomic(Node*) top;\n} LockFreeStack;\n\nvoid push(LockFreeStack *stack, void *data) {\n    Node *new_node = malloc(sizeof(Node));\n    new_node->data = data;\n    \n    Node *old_top = atomic_load(&stack->top);\n    do {\n        new_node->next = old_top;\n    } while (!atomic_compare_exchange_weak(&stack->top, &old_top, new_node));\n}\n```",
                    "level2": "Lock-free pop:\n```c\nvoid *pop(LockFreeStack *stack) {\n    Node *old_top = atomic_load(&stack->top);\n    Node *new_top;\n    \n    do {\n        if (old_top == NULL) {\n            return NULL;  // Stack is empty\n        }\n        new_top = old_top->next;\n    } while (!atomic_compare_exchange_weak(&stack->top, &old_top, new_top));\n    \n    void *data = old_top->data;\n    // WARNING: Can't free old_top here (ABA problem)\n    return data;\n}\n```",
                    "level3": "Tagged pointer to solve ABA:\n```c\ntypedef struct {\n    Node *ptr;\n    uint64_t tag;  // Version counter\n} TaggedPtr;\n\n// Pack pointer and tag into 128-bit value for double-width CAS\n// Or use lower bits of aligned pointer for tag\n\nvoid *pop_safe(LockFreeStack *stack) {\n    TaggedPtr old_top = atomic_load(&stack->top);\n    TaggedPtr new_top;\n    \n    do {\n        if (old_top.ptr == NULL) return NULL;\n        new_top.ptr = old_top.ptr->next;\n        new_top.tag = old_top.tag + 1;  // Increment tag\n    } while (!atomic_compare_exchange_weak(&stack->top, &old_top, new_top));\n    \n    return old_top.ptr->data;\n}\n```"
                }
            },
            {
                "name": "Lock-free Queue",
                "description": "Michael-Scott queue implementation",
                "skills": ["MS queue", "Two-pointer queue", "Helping mechanism"],
                "deliverables": [
                    "Head and tail pointers",
                    "Dummy node initialization",
                    "Lock-free enqueue",
                    "Lock-free dequeue",
                    "Helping (tail update)",
                    "Linearizability"
                ],
                "hints": {
                    "level1": "Queue structure:\n```c\ntypedef struct Node {\n    void *data;\n    _Atomic(struct Node*) next;\n} Node;\n\ntypedef struct {\n    _Atomic(Node*) head;\n    _Atomic(Node*) tail;\n} LockFreeQueue;\n\nvoid init(LockFreeQueue *q) {\n    Node *dummy = malloc(sizeof(Node));\n    dummy->data = NULL;\n    atomic_store(&dummy->next, NULL);\n    atomic_store(&q->head, dummy);\n    atomic_store(&q->tail, dummy);\n}\n```",
                    "level2": "Enqueue with helping:\n```c\nvoid enqueue(LockFreeQueue *q, void *data) {\n    Node *new_node = malloc(sizeof(Node));\n    new_node->data = data;\n    atomic_store(&new_node->next, NULL);\n    \n    while (1) {\n        Node *tail = atomic_load(&q->tail);\n        Node *next = atomic_load(&tail->next);\n        \n        if (tail == atomic_load(&q->tail)) {  // Still consistent?\n            if (next == NULL) {\n                // Tail is really last, try to link new node\n                if (atomic_compare_exchange_weak(&tail->next, &next, new_node)) {\n                    // Success, try to update tail\n                    atomic_compare_exchange_weak(&q->tail, &tail, new_node);\n                    return;\n                }\n            } else {\n                // Tail is behind, help advance it\n                atomic_compare_exchange_weak(&q->tail, &tail, next);\n            }\n        }\n    }\n}\n```",
                    "level3": "Dequeue:\n```c\nvoid *dequeue(LockFreeQueue *q) {\n    while (1) {\n        Node *head = atomic_load(&q->head);\n        Node *tail = atomic_load(&q->tail);\n        Node *next = atomic_load(&head->next);\n        \n        if (head == atomic_load(&q->head)) {\n            if (head == tail) {\n                if (next == NULL) {\n                    return NULL;  // Queue is empty\n                }\n                // Tail is behind, help advance\n                atomic_compare_exchange_weak(&q->tail, &tail, next);\n            } else {\n                // Read data before CAS\n                void *data = next->data;\n                if (atomic_compare_exchange_weak(&q->head, &head, next)) {\n                    // Don't free head (memory reclamation needed)\n                    return data;\n                }\n            }\n        }\n    }\n}\n```"
                }
            },
            {
                "name": "Hazard Pointers",
                "description": "Safe memory reclamation",
                "skills": ["Hazard pointers", "Memory reclamation", "Deferred free"],
                "deliverables": [
                    "Hazard pointer registry",
                    "Protect/release operations",
                    "Retired list",
                    "Scan and reclaim",
                    "Thread-local hazard pointers",
                    "Integration with lock-free structures"
                ],
                "hints": {
                    "level1": "Hazard pointer structure:\n```c\n#define MAX_THREADS 64\n#define HP_PER_THREAD 2\n\ntypedef struct {\n    _Atomic(void*) hp[MAX_THREADS][HP_PER_THREAD];\n    void *retired[MAX_THREADS][1024];  // Per-thread retired list\n    int retired_count[MAX_THREADS];\n} HazardPointerDomain;\n\nvoid hp_protect(HazardPointerDomain *d, int thread_id, int hp_index, void *ptr) {\n    atomic_store(&d->hp[thread_id][hp_index], ptr);\n}\n\nvoid hp_clear(HazardPointerDomain *d, int thread_id, int hp_index) {\n    atomic_store(&d->hp[thread_id][hp_index], NULL);\n}\n```",
                    "level2": "Retire and reclaim:\n```c\nvoid hp_retire(HazardPointerDomain *d, int thread_id, void *ptr) {\n    d->retired[thread_id][d->retired_count[thread_id]++] = ptr;\n    \n    if (d->retired_count[thread_id] >= 100) {\n        hp_scan(d, thread_id);\n    }\n}\n\nvoid hp_scan(HazardPointerDomain *d, int thread_id) {\n    // Collect all hazard pointers\n    void *protected[MAX_THREADS * HP_PER_THREAD];\n    int pcount = 0;\n    for (int t = 0; t < MAX_THREADS; t++) {\n        for (int h = 0; h < HP_PER_THREAD; h++) {\n            void *hp = atomic_load(&d->hp[t][h]);\n            if (hp) protected[pcount++] = hp;\n        }\n    }\n    \n    // Free retired nodes not in protected set\n    // ...\n}\n```",
                    "level3": "Use with stack:\n```c\nvoid *pop_safe(LockFreeStack *stack, HazardPointerDomain *hp, int tid) {\n    while (1) {\n        Node *old_top = atomic_load(&stack->top);\n        if (old_top == NULL) return NULL;\n        \n        // Protect old_top with hazard pointer\n        hp_protect(hp, tid, 0, old_top);\n        \n        // Verify it's still the top\n        if (old_top != atomic_load(&stack->top)) continue;\n        \n        Node *new_top = old_top->next;\n        if (atomic_compare_exchange_weak(&stack->top, &old_top, new_top)) {\n            void *data = old_top->data;\n            hp_clear(hp, tid, 0);\n            hp_retire(hp, tid, old_top);  // Safe deferred free\n            return data;\n        }\n    }\n}\n```"
                }
            },
            {
                "name": "Lock-free Hash Map",
                "description": "Concurrent hash map with lock-free operations",
                "skills": ["Concurrent hashing", "Split-ordered lists", "Resizing"],
                "deliverables": [
                    "Hash bucket array",
                    "Lock-free insert",
                    "Lock-free lookup",
                    "Lock-free delete",
                    "Atomic resize",
                    "Load factor management"
                ],
                "hints": {
                    "level1": "Simple lock-free map structure:\n```c\ntypedef struct Entry {\n    uint64_t key;\n    void *value;\n    _Atomic(struct Entry*) next;\n} Entry;\n\ntypedef struct {\n    _Atomic(Entry*) buckets[INITIAL_SIZE];\n    _Atomic size_t size;\n    _Atomic size_t count;\n} LockFreeHashMap;\n\nuint64_t hash(uint64_t key, size_t size) {\n    return key % size;\n}\n```",
                    "level2": "Lock-free insert:\n```c\nbool insert(LockFreeHashMap *map, uint64_t key, void *value) {\n    Entry *new_entry = malloc(sizeof(Entry));\n    new_entry->key = key;\n    new_entry->value = value;\n    \n    size_t idx = hash(key, atomic_load(&map->size));\n    \n    while (1) {\n        Entry *head = atomic_load(&map->buckets[idx]);\n        \n        // Check if key exists\n        for (Entry *e = head; e != NULL; e = atomic_load(&e->next)) {\n            if (e->key == key) {\n                free(new_entry);\n                return false;  // Already exists\n            }\n        }\n        \n        atomic_store(&new_entry->next, head);\n        if (atomic_compare_exchange_weak(&map->buckets[idx], &head, new_entry)) {\n            atomic_fetch_add(&map->count, 1);\n            return true;\n        }\n    }\n}\n```",
                    "level3": "Lock-free delete with marking:\n```c\n// Use pointer's low bit as \"marked for deletion\" flag\n#define MARKED(p) ((Entry*)((uintptr_t)(p) | 1))\n#define UNMARKED(p) ((Entry*)((uintptr_t)(p) & ~1))\n#define IS_MARKED(p) ((uintptr_t)(p) & 1)\n\nbool delete(LockFreeHashMap *map, uint64_t key) {\n    size_t idx = hash(key, atomic_load(&map->size));\n    \n    while (1) {\n        Entry **prev = &map->buckets[idx];\n        Entry *curr = atomic_load(prev);\n        \n        while (curr != NULL) {\n            Entry *next = atomic_load(&curr->next);\n            \n            if (IS_MARKED(next)) {\n                // Help remove marked node\n                atomic_compare_exchange_weak(prev, &curr, UNMARKED(next));\n                curr = atomic_load(prev);\n            } else if (curr->key == key) {\n                // Mark for deletion\n                if (atomic_compare_exchange_weak(&curr->next, &next, MARKED(next))) {\n                    // Physically remove\n                    atomic_compare_exchange_weak(prev, &curr, next);\n                    return true;\n                }\n            } else {\n                prev = &curr->next;\n                curr = next;\n            }\n        }\n        return false;\n    }\n}\n```"
                }
            }
        ]
    },
    "fuzzer": {
        "name": "Fuzzing Framework",
        "description": "Build a coverage-guided fuzzer like AFL for automated bug finding",
        "category": "Security",
        "difficulty": "advanced",
        "estimated_hours": 55,
        "skills": [
            "Coverage instrumentation",
            "Mutation strategies",
            "Corpus management",
            "Crash detection",
            "Input minimization",
            "Parallel fuzzing"
        ],
        "prerequisites": ["Binary basics", "Testing", "Process management"],
        "learning_outcomes": [
            "Understand fuzzing principles and techniques",
            "Implement coverage-guided mutation",
            "Build crash detection and triage",
            "Design efficient fuzzing campaigns"
        ],
        "milestones": [
            {
                "name": "Target Execution",
                "description": "Execute target program with inputs",
                "skills": ["Process execution", "Timeout handling", "Exit code analysis"],
                "deliverables": [
                    "Fork/exec target",
                    "Stdin/file input modes",
                    "Timeout enforcement",
                    "Crash detection (signals)",
                    "Resource limits",
                    "Exit status collection"
                ],
                "hints": {
                    "level1": "Basic target runner:\n```python\nimport subprocess\nimport signal\n\nclass TargetRunner:\n    def __init__(self, target_cmd, timeout=1.0):\n        self.target_cmd = target_cmd\n        self.timeout = timeout\n    \n    def run(self, input_data):\n        try:\n            result = subprocess.run(\n                self.target_cmd,\n                input=input_data,\n                capture_output=True,\n                timeout=self.timeout\n            )\n            return ExecutionResult(\n                exit_code=result.returncode,\n                stdout=result.stdout,\n                stderr=result.stderr,\n                crashed=result.returncode < 0\n            )\n        except subprocess.TimeoutExpired:\n            return ExecutionResult(timeout=True)\n```",
                    "level2": "Detect crash signals:\n```python\ndef classify_crash(exit_code):\n    if exit_code >= 0:\n        return None\n    \n    sig = -exit_code\n    crash_signals = {\n        signal.SIGSEGV: 'segfault',\n        signal.SIGABRT: 'abort',\n        signal.SIGFPE: 'floating_point',\n        signal.SIGBUS: 'bus_error',\n        signal.SIGILL: 'illegal_instruction',\n    }\n    return crash_signals.get(sig, f'signal_{sig}')\n```",
                    "level3": "Persistent mode (fork server):\n```c\n// In instrumented target\nvoid __afl_forkserver() {\n    while (1) {\n        // Wait for fuzzer\n        int status;\n        read(FORKSRV_FD, &status, 4);\n        \n        pid_t child = fork();\n        if (child == 0) {\n            // Child: close forkserver fds and continue\n            close(FORKSRV_FD);\n            return;\n        }\n        \n        // Parent: report child pid and wait\n        write(FORKSRV_FD + 1, &child, 4);\n        waitpid(child, &status, 0);\n        write(FORKSRV_FD + 1, &status, 4);\n    }\n}\n```"
                }
            },
            {
                "name": "Coverage Tracking",
                "description": "Track code coverage for guidance",
                "skills": ["Instrumentation", "Edge coverage", "Bitmap"],
                "deliverables": [
                    "Coverage bitmap",
                    "Edge/branch coverage",
                    "Compile-time instrumentation",
                    "Coverage comparison",
                    "New coverage detection",
                    "Coverage visualization"
                ],
                "hints": {
                    "level1": "Coverage bitmap:\n```python\nclass CoverageBitmap:\n    def __init__(self, size=65536):\n        self.size = size\n        self.bitmap = bytearray(size)\n        self.virgin = bytearray(b'\\xff' * size)  # Never-seen bits\n    \n    def has_new_coverage(self, exec_bitmap):\n        new_bits = False\n        for i in range(self.size):\n            if exec_bitmap[i] and self.virgin[i]:\n                # New coverage found\n                self.virgin[i] &= ~exec_bitmap[i]\n                self.bitmap[i] |= exec_bitmap[i]\n                new_bits = True\n        return new_bits\n```",
                    "level2": "Compile-time instrumentation (LLVM pass concept):\n```c\n// Injected at each basic block\nuint8_t __afl_area[65536];\nuint32_t __afl_prev_loc;\n\nvoid __afl_trace(uint32_t cur_loc) {\n    // Edge = prev_loc XOR cur_loc\n    __afl_area[cur_loc ^ __afl_prev_loc]++;\n    __afl_prev_loc = cur_loc >> 1;\n}\n\n// Or use compiler flag: clang -fsanitize-coverage=trace-pc-guard\n```",
                    "level3": "Coverage-guided input selection:\n```python\nclass Corpus:\n    def __init__(self):\n        self.inputs = []  # (input_data, coverage_hash)\n    \n    def add_if_interesting(self, input_data, coverage):\n        cov_hash = hash(bytes(coverage))\n        \n        # Check if this coverage is new\n        if self.bitmap.has_new_coverage(coverage):\n            self.inputs.append(CorpusEntry(\n                data=input_data,\n                coverage=coverage,\n                found_at=time.time()\n            ))\n            return True\n        return False\n    \n    def select(self):\n        # Favor smaller inputs and inputs that found new coverage recently\n        weights = [1.0 / (len(e.data) + 1) for e in self.inputs]\n        return random.choices(self.inputs, weights=weights)[0]\n```"
                }
            },
            {
                "name": "Mutation Engine",
                "description": "Generate new test inputs via mutation",
                "skills": ["Bit flips", "Arithmetic", "Dictionary", "Splice"],
                "deliverables": [
                    "Bit flip mutations",
                    "Byte flip mutations",
                    "Arithmetic mutations",
                    "Block operations (insert, delete, overwrite)",
                    "Dictionary-based mutations",
                    "Havoc mode (random mutations)"
                ],
                "hints": {
                    "level1": "Basic mutations:\n```python\nclass Mutator:\n    def bit_flip(self, data, pos):\n        data = bytearray(data)\n        byte_pos = pos // 8\n        bit_pos = pos % 8\n        data[byte_pos] ^= (1 << bit_pos)\n        return bytes(data)\n    \n    def byte_flip(self, data, pos):\n        data = bytearray(data)\n        data[pos] ^= 0xFF\n        return bytes(data)\n    \n    def interesting_value(self, data, pos, size):\n        interesting = [0, 1, -1, 127, 128, 255, 256, 32767, 65535]\n        data = bytearray(data)\n        val = random.choice(interesting)\n        # Write val at pos with given size\n        return bytes(data)\n```",
                    "level2": "Arithmetic and block mutations:\n```python\ndef arithmetic(self, data, pos, size):\n    data = bytearray(data)\n    val = int.from_bytes(data[pos:pos+size], 'little')\n    delta = random.randint(-35, 35)\n    new_val = (val + delta) & ((1 << (size*8)) - 1)\n    data[pos:pos+size] = new_val.to_bytes(size, 'little')\n    return bytes(data)\n\ndef insert_bytes(self, data, pos, count):\n    new_bytes = bytes([random.randint(0, 255) for _ in range(count)])\n    return data[:pos] + new_bytes + data[pos:]\n\ndef delete_bytes(self, data, pos, count):\n    return data[:pos] + data[pos+count:]\n```",
                    "level3": "Havoc mode:\n```python\ndef havoc(self, data):\n    data = bytearray(data)\n    num_mutations = random.randint(1, 16)\n    \n    mutations = [\n        self.bit_flip_random,\n        self.byte_flip_random,\n        self.arithmetic_random,\n        self.overwrite_random,\n        self.insert_random,\n        self.delete_random,\n        self.splice,\n        self.dictionary_insert,\n    ]\n    \n    for _ in range(num_mutations):\n        mutation = random.choice(mutations)\n        data = mutation(data)\n    \n    return bytes(data)\n```"
                }
            },
            {
                "name": "Corpus Management",
                "description": "Manage and minimize test corpus",
                "skills": ["Corpus storage", "Minimization", "Deduplication"],
                "deliverables": [
                    "Corpus storage (files)",
                    "Input minimization",
                    "Corpus distillation",
                    "Crash deduplication",
                    "Coverage-based scoring",
                    "Queue scheduling"
                ],
                "hints": {
                    "level1": "Corpus storage:\n```python\nclass CorpusManager:\n    def __init__(self, corpus_dir):\n        self.corpus_dir = Path(corpus_dir)\n        self.queue_dir = self.corpus_dir / 'queue'\n        self.crashes_dir = self.corpus_dir / 'crashes'\n        self.queue_dir.mkdir(parents=True, exist_ok=True)\n        self.crashes_dir.mkdir(exist_ok=True)\n    \n    def save_input(self, data, coverage_hash):\n        filename = f'id:{self.next_id():06d},cov:{coverage_hash:08x}'\n        path = self.queue_dir / filename\n        path.write_bytes(data)\n        return path\n    \n    def save_crash(self, data, crash_type):\n        sig = hashlib.sha256(data).hexdigest()[:8]\n        filename = f'{crash_type}_{sig}'\n        path = self.crashes_dir / filename\n        path.write_bytes(data)\n```",
                    "level2": "Input minimization:\n```python\ndef minimize(self, data, check_fn):\n    \"\"\"Minimize input while preserving interesting behavior\"\"\"\n    # Binary search for minimum size\n    while len(data) > 1:\n        mid = len(data) // 2\n        \n        # Try first half\n        if check_fn(data[:mid]):\n            data = data[:mid]\n            continue\n        \n        # Try second half\n        if check_fn(data[mid:]):\n            data = data[mid:]\n            continue\n        \n        # Try removing chunks\n        reduced = False\n        chunk_size = max(1, len(data) // 16)\n        for i in range(0, len(data), chunk_size):\n            candidate = data[:i] + data[i+chunk_size:]\n            if check_fn(candidate):\n                data = candidate\n                reduced = True\n                break\n        \n        if not reduced:\n            break\n    \n    return data\n```",
                    "level3": "Crash deduplication:\n```python\nclass CrashDeduplicator:\n    def __init__(self):\n        self.seen_crashes = set()\n    \n    def get_crash_signature(self, crash_info):\n        # Use stack trace hash for deduplication\n        if crash_info.stack_trace:\n            frames = crash_info.stack_trace[:5]  # Top 5 frames\n            return hash(tuple(frames))\n        \n        # Fallback to coverage hash at crash point\n        return hash(bytes(crash_info.coverage[-100:]))\n    \n    def is_unique(self, crash_info):\n        sig = self.get_crash_signature(crash_info)\n        if sig in self.seen_crashes:\n            return False\n        self.seen_crashes.add(sig)\n        return True\n```"
                }
            },
            {
                "name": "Fuzzing Loop",
                "description": "Main fuzzing orchestration",
                "skills": ["Scheduling", "Statistics", "Parallel fuzzing"],
                "deliverables": [
                    "Main fuzzing loop",
                    "Statistics and reporting",
                    "Parallel/distributed fuzzing",
                    "Sync between instances",
                    "Adaptive mutation selection",
                    "Timeout and resource management"
                ],
                "hints": {
                    "level1": "Main fuzzing loop:\n```python\nclass Fuzzer:\n    def run(self):\n        self.load_corpus()\n        \n        while True:\n            # Select input from corpus\n            entry = self.corpus.select()\n            \n            # Mutate\n            for _ in range(self.mutations_per_input):\n                mutated = self.mutator.mutate(entry.data)\n                \n                # Execute\n                result = self.runner.run(mutated)\n                self.stats.total_execs += 1\n                \n                # Check for crash\n                if result.crashed:\n                    self.save_crash(mutated, result)\n                    continue\n                \n                # Check for new coverage\n                if self.corpus.add_if_interesting(mutated, result.coverage):\n                    self.stats.new_paths += 1\n            \n            self.print_stats()\n```",
                    "level2": "Statistics:\n```python\nclass FuzzerStats:\n    def __init__(self):\n        self.start_time = time.time()\n        self.total_execs = 0\n        self.new_paths = 0\n        self.crashes = 0\n        self.timeouts = 0\n    \n    def print_status(self):\n        elapsed = time.time() - self.start_time\n        exec_rate = self.total_execs / elapsed if elapsed > 0 else 0\n        \n        print(f'''\n        Fuzzer Status\n        ─────────────────────────\n        Runtime     : {elapsed:.0f}s\n        Executions  : {self.total_execs}\n        Exec/sec    : {exec_rate:.0f}\n        Corpus size : {self.corpus_size}\n        Crashes     : {self.crashes}\n        Coverage    : {self.coverage_pct:.1f}%\n        ''')\n```",
                    "level3": "Parallel fuzzing with sync:\n```python\nclass ParallelFuzzer:\n    def __init__(self, num_workers, sync_dir):\n        self.num_workers = num_workers\n        self.sync_dir = Path(sync_dir)\n    \n    def run_worker(self, worker_id):\n        fuzzer = Fuzzer(corpus_dir=self.sync_dir / f'worker_{worker_id}')\n        \n        while True:\n            # Run fuzzing iterations\n            fuzzer.fuzz_one()\n            \n            # Periodically sync with other workers\n            if fuzzer.stats.total_execs % 1000 == 0:\n                self.sync_corpus(worker_id)\n    \n    def sync_corpus(self, worker_id):\n        # Import interesting inputs from other workers\n        for other_dir in self.sync_dir.glob('worker_*'):\n            if other_dir.name == f'worker_{worker_id}':\n                continue\n            for input_file in (other_dir / 'queue').glob('*'):\n                self.try_import(input_file)\n```"
                }
            }
        ]
    },
    "profiler": {
        "name": "CPU/Memory Profiler",
        "description": "Build a sampling profiler with flame graphs and memory tracking",
        "category": "Performance",
        "difficulty": "intermediate",
        "estimated_hours": 45,
        "skills": [
            "Stack sampling",
            "Timer signals",
            "Symbol resolution",
            "Flame graphs",
            "Memory tracking",
            "Profile analysis"
        ],
        "prerequisites": ["C programming", "Process internals", "Debug symbols"],
        "learning_outcomes": [
            "Understand sampling-based profiling",
            "Implement stack trace collection",
            "Build flame graph visualization",
            "Design memory allocation tracking"
        ],
        "milestones": [
            {
                "name": "Stack Sampling",
                "description": "Periodically sample call stacks",
                "skills": ["Signal handling", "Stack walking", "Timer setup"],
                "deliverables": [
                    "SIGPROF/ITIMER setup",
                    "Signal handler for sampling",
                    "Stack frame walking",
                    "Sample storage",
                    "Sampling frequency control",
                    "Low-overhead design"
                ],
                "hints": {
                    "level1": "Setup sampling timer:\n```c\n#include <signal.h>\n#include <sys/time.h>\n\nvoid setup_profiler(int frequency_hz) {\n    struct sigaction sa;\n    sa.sa_handler = profile_signal_handler;\n    sa.sa_flags = SA_RESTART;\n    sigaction(SIGPROF, &sa, NULL);\n    \n    struct itimerval timer;\n    timer.it_interval.tv_sec = 0;\n    timer.it_interval.tv_usec = 1000000 / frequency_hz;\n    timer.it_value = timer.it_interval;\n    setitimer(ITIMER_PROF, &timer, NULL);\n}\n```",
                    "level2": "Capture stack in signal handler:\n```c\n#include <execinfo.h>\n\n#define MAX_FRAMES 128\n#define MAX_SAMPLES 100000\n\nvoid *samples[MAX_SAMPLES][MAX_FRAMES];\nint sample_depths[MAX_SAMPLES];\nint sample_count = 0;\n\nvoid profile_signal_handler(int sig) {\n    if (sample_count >= MAX_SAMPLES) return;\n    \n    void *frames[MAX_FRAMES];\n    int depth = backtrace(frames, MAX_FRAMES);\n    \n    memcpy(samples[sample_count], frames, depth * sizeof(void*));\n    sample_depths[sample_count] = depth;\n    sample_count++;\n}\n```",
                    "level3": "Use libunwind for better stacks:\n```c\n#define UNW_LOCAL_ONLY\n#include <libunwind.h>\n\nint capture_stack(void **buffer, int max_depth) {\n    unw_cursor_t cursor;\n    unw_context_t context;\n    \n    unw_getcontext(&context);\n    unw_init_local(&cursor, &context);\n    \n    int depth = 0;\n    while (unw_step(&cursor) > 0 && depth < max_depth) {\n        unw_word_t pc;\n        unw_get_reg(&cursor, UNW_REG_IP, &pc);\n        buffer[depth++] = (void*)pc;\n    }\n    return depth;\n}\n```"
                }
            },
            {
                "name": "Symbol Resolution",
                "description": "Convert addresses to function names",
                "skills": ["Debug symbols", "DWARF", "addr2line"],
                "deliverables": [
                    "Symbol table loading",
                    "Address to function mapping",
                    "Source file and line info",
                    "Demangling C++ names",
                    "Shared library handling",
                    "Symbol caching"
                ],
                "hints": {
                    "level1": "Use dladdr for basic resolution:\n```c\n#include <dlfcn.h>\n\nchar *resolve_symbol(void *addr) {\n    Dl_info info;\n    if (dladdr(addr, &info) && info.dli_sname) {\n        return strdup(info.dli_sname);\n    }\n    // Fallback to address\n    char buf[32];\n    snprintf(buf, sizeof(buf), \"0x%lx\", (unsigned long)addr);\n    return strdup(buf);\n}\n```",
                    "level2": "Use addr2line for source info:\n```python\nimport subprocess\n\nclass SymbolResolver:\n    def __init__(self, binary):\n        self.binary = binary\n        self.cache = {}\n    \n    def resolve(self, address):\n        if address in self.cache:\n            return self.cache[address]\n        \n        result = subprocess.run(\n            ['addr2line', '-f', '-e', self.binary, hex(address)],\n            capture_output=True, text=True\n        )\n        lines = result.stdout.strip().split('\\n')\n        func = lines[0] if lines else '??'\n        location = lines[1] if len(lines) > 1 else '??:0'\n        \n        self.cache[address] = (func, location)\n        return (func, location)\n```",
                    "level3": "Demangle C++ names:\n```python\nimport subprocess\n\ndef demangle(name):\n    if not name.startswith('_Z'):\n        return name\n    \n    result = subprocess.run(\n        ['c++filt', name],\n        capture_output=True, text=True\n    )\n    return result.stdout.strip()\n\n# Or use __cxa_demangle in C\n```"
                }
            },
            {
                "name": "Flame Graph Generation",
                "description": "Visualize profiles as flame graphs",
                "skills": ["Data aggregation", "SVG generation", "Interactive UI"],
                "deliverables": [
                    "Stack aggregation",
                    "Folded stack format",
                    "SVG flame graph",
                    "Color coding",
                    "Zoom and search",
                    "Differential flame graphs"
                ],
                "hints": {
                    "level1": "Generate folded stacks:\n```python\ndef generate_folded_stacks(samples):\n    stacks = {}\n    for sample in samples:\n        # Reverse so caller is first\n        stack_str = ';'.join(reversed(sample.frames))\n        stacks[stack_str] = stacks.get(stack_str, 0) + 1\n    \n    # Output in folded format\n    for stack, count in sorted(stacks.items()):\n        print(f'{stack} {count}')\n\n# Output format:\n# main;foo;bar 42\n# main;foo;baz 17\n```",
                    "level2": "Simple SVG flame graph:\n```python\nclass FlameGraph:\n    def __init__(self, width=1200, row_height=16):\n        self.width = width\n        self.row_height = row_height\n        self.svg_parts = []\n    \n    def generate(self, folded_stacks):\n        total_samples = sum(folded_stacks.values())\n        \n        # Build tree from stacks\n        root = self.build_tree(folded_stacks)\n        \n        # Calculate widths and positions\n        self.layout(root, 0, self.width, 0)\n        \n        # Render to SVG\n        self.render(root)\n        \n        return self.to_svg()\n    \n    def render_frame(self, name, x, width, y, samples):\n        color = self.color_for(name)\n        self.svg_parts.append(f'''\n            <rect x=\"{x}\" y=\"{y}\" width=\"{width}\" height=\"{self.row_height}\" \n                  fill=\"{color}\" />\n            <text x=\"{x+2}\" y=\"{y+12}\">{name}</text>\n        ''')\n```",
                    "level3": "Use Brendan Gregg's flamegraph.pl or d3-flame-graph for interactive visualization."
                }
            },
            {
                "name": "Memory Profiling",
                "description": "Track memory allocations",
                "skills": ["Malloc interception", "Allocation tracking", "Leak detection"],
                "deliverables": [
                    "malloc/free interception",
                    "Allocation size tracking",
                    "Stack trace at allocation",
                    "Memory usage over time",
                    "Leak detection",
                    "Memory flamegraph"
                ],
                "hints": {
                    "level1": "Intercept malloc with LD_PRELOAD:\n```c\n#define _GNU_SOURCE\n#include <dlfcn.h>\n\nstatic void* (*real_malloc)(size_t) = NULL;\nstatic void (*real_free)(void*) = NULL;\n\nvoid __attribute__((constructor)) init() {\n    real_malloc = dlsym(RTLD_NEXT, \"malloc\");\n    real_free = dlsym(RTLD_NEXT, \"free\");\n}\n\nvoid *malloc(size_t size) {\n    void *ptr = real_malloc(size);\n    record_allocation(ptr, size);\n    return ptr;\n}\n\nvoid free(void *ptr) {\n    record_free(ptr);\n    real_free(ptr);\n}\n```",
                    "level2": "Track allocations with stack traces:\n```c\ntypedef struct {\n    void *ptr;\n    size_t size;\n    void *stack[16];\n    int stack_depth;\n} Allocation;\n\n#define MAX_ALLOCATIONS 1000000\nAllocation allocations[MAX_ALLOCATIONS];\nint alloc_count = 0;\n\nvoid record_allocation(void *ptr, size_t size) {\n    Allocation *a = &allocations[alloc_count++];\n    a->ptr = ptr;\n    a->size = size;\n    a->stack_depth = backtrace(a->stack, 16);\n}\n```",
                    "level3": "Leak detection:\n```python\ndef find_leaks(allocations, frees):\n    allocated = {a.ptr: a for a in allocations}\n    \n    for f in frees:\n        if f.ptr in allocated:\n            del allocated[f.ptr]\n    \n    # Remaining allocations are leaks\n    leaks = list(allocated.values())\n    \n    # Group by allocation site\n    by_stack = {}\n    for leak in leaks:\n        stack_key = tuple(leak.stack)\n        if stack_key not in by_stack:\n            by_stack[stack_key] = {'count': 0, 'bytes': 0, 'stack': leak.stack}\n        by_stack[stack_key]['count'] += 1\n        by_stack[stack_key]['bytes'] += leak.size\n    \n    return sorted(by_stack.values(), key=lambda x: -x['bytes'])\n```"
                }
            }
        ]
    },
    "bootloader": {
        "name": "x86 Bootloader",
        "description": "Build a bootloader that loads a kernel from disk and switches to protected mode",
        "category": "Systems",
        "difficulty": "expert",
        "estimated_hours": 45,
        "skills": [
            "x86 assembly",
            "BIOS interrupts",
            "Real mode",
            "Protected mode",
            "GDT setup",
            "Disk I/O"
        ],
        "prerequisites": ["Assembly basics", "Computer architecture"],
        "learning_outcomes": [
            "Understand x86 boot process",
            "Work with BIOS services",
            "Implement mode switching",
            "Load and execute kernel code"
        ],
        "milestones": [
            {
                "name": "Boot Sector",
                "description": "Minimal boot sector that BIOS can load",
                "skills": ["Boot sector format", "BIOS loading", "16-bit assembly"],
                "deliverables": [
                    "512-byte boot sector",
                    "Boot signature (0xAA55)",
                    "Print message using BIOS",
                    "Infinite loop to halt",
                    "Build with NASM",
                    "Test with QEMU"
                ],
                "hints": {
                    "level1": "Minimal boot sector:\n```nasm\n[bits 16]\n[org 0x7c00]\n\nstart:\n    ; Print character using BIOS\n    mov ah, 0x0e    ; BIOS teletype\n    mov al, 'H'\n    int 0x10\n    mov al, 'i'\n    int 0x10\n    \n    jmp $           ; Infinite loop\n\ntimes 510-($-$$) db 0   ; Pad to 510 bytes\ndw 0xaa55               ; Boot signature\n```",
                    "level2": "Build and test:\n```bash\nnasm -f bin boot.asm -o boot.bin\nqemu-system-i386 -fda boot.bin\n```",
                    "level3": "Print string function:\n```nasm\nprint_string:\n    pusha\n    mov ah, 0x0e\n.loop:\n    lodsb           ; Load byte from SI\n    cmp al, 0\n    je .done\n    int 0x10\n    jmp .loop\n.done:\n    popa\n    ret\n\nmsg: db 'Hello from bootloader!', 0\n```"
                }
            },
            {
                "name": "Disk Reading",
                "description": "Load more sectors from disk",
                "skills": ["BIOS disk services", "INT 13h", "CHS addressing"],
                "deliverables": [
                    "Read sectors using INT 13h",
                    "Handle multiple sectors",
                    "Error handling and retry",
                    "Load kernel to memory",
                    "LBA to CHS conversion",
                    "Boot from hard disk"
                ],
                "hints": {
                    "level1": "Read disk sector:\n```nasm\nload_kernel:\n    mov ah, 0x02        ; BIOS read sectors\n    mov al, 15          ; Number of sectors\n    mov ch, 0           ; Cylinder 0\n    mov cl, 2           ; Sector 2 (1-indexed)\n    mov dh, 0           ; Head 0\n    mov dl, [boot_drive] ; Drive number\n    mov bx, 0x1000      ; Destination ES:BX\n    mov es, bx\n    xor bx, bx\n    int 0x13\n    jc disk_error       ; Carry set on error\n    ret\n\nboot_drive: db 0\n```",
                    "level2": "Retry on error:\n```nasm\nload_with_retry:\n    mov cx, 3           ; Retry count\n.retry:\n    push cx\n    call load_sectors\n    pop cx\n    jnc .success\n    \n    ; Reset disk\n    xor ah, ah\n    int 0x13\n    \n    loop .retry\n    jmp disk_error\n.success:\n    ret\n```",
                    "level3": "LBA to CHS:\n```nasm\n; LBA in AX, results in CH=cyl, CL=sect, DH=head\nlba_to_chs:\n    xor dx, dx\n    div word [sectors_per_track]\n    inc dl              ; Sector (1-indexed)\n    mov cl, dl\n    xor dx, dx\n    div word [heads]\n    mov dh, dl          ; Head\n    mov ch, al          ; Cylinder\n    ret\n```"
                }
            },
            {
                "name": "Protected Mode",
                "description": "Switch from 16-bit real mode to 32-bit protected mode",
                "skills": ["GDT", "A20 line", "CR0 register", "Far jump"],
                "deliverables": [
                    "Global Descriptor Table (GDT)",
                    "Enable A20 line",
                    "Set CR0 PE bit",
                    "Far jump to 32-bit code",
                    "Reload segment registers",
                    "Set up stack"
                ],
                "hints": {
                    "level1": "GDT definition:\n```nasm\ngdt_start:\n    dq 0                ; Null descriptor\n\ngdt_code:\n    dw 0xffff           ; Limit\n    dw 0                ; Base (low)\n    db 0                ; Base (mid)\n    db 10011010b        ; Access: present, ring 0, code, readable\n    db 11001111b        ; Flags: 4KB granularity, 32-bit\n    db 0                ; Base (high)\n\ngdt_data:\n    dw 0xffff\n    dw 0\n    db 0\n    db 10010010b        ; Access: present, ring 0, data, writable\n    db 11001111b\n    db 0\n\ngdt_end:\n\ngdt_descriptor:\n    dw gdt_end - gdt_start - 1  ; Size\n    dd gdt_start                 ; Address\n```",
                    "level2": "Switch to protected mode:\n```nasm\nswitch_to_pm:\n    cli                     ; Disable interrupts\n    lgdt [gdt_descriptor]   ; Load GDT\n    \n    ; Enable A20\n    in al, 0x92\n    or al, 2\n    out 0x92, al\n    \n    ; Set PE bit in CR0\n    mov eax, cr0\n    or eax, 1\n    mov cr0, eax\n    \n    ; Far jump to flush pipeline\n    jmp 0x08:protected_mode\n\n[bits 32]\nprotected_mode:\n    ; Reload segment registers\n    mov ax, 0x10        ; Data segment\n    mov ds, ax\n    mov es, ax\n    mov fs, ax\n    mov gs, ax\n    mov ss, ax\n    \n    mov esp, 0x90000    ; Set up stack\n    call kernel_main\n```",
                    "level3": "Jump to loaded kernel:\n```nasm\n[bits 32]\nprotected_mode:\n    mov ax, 0x10\n    mov ds, ax\n    mov ss, ax\n    mov esp, 0x90000\n    \n    ; Kernel was loaded at 0x10000\n    jmp 0x10000\n```"
                }
            }
        ]
    }
}


def main():
    projects_file = Path("data/projects.yaml")

    with open(projects_file, 'r') as f:
        data = yaml.safe_load(f)

    # Add to domains
    for domain in data['domains']:
        domain_id = domain['id']
        if domain_id in GAP_PROJECTS:
            for level, projects in GAP_PROJECTS[domain_id].items():
                if level not in domain.get('projects', {}):
                    domain['projects'][level] = []

                existing_ids = {p['id'] for p in domain['projects'][level]}
                for project in projects:
                    if project['id'] not in existing_ids:
                        domain['projects'][level].append(project)
                        print(f"Added: {project['id']} to {domain_id}/{level}")

    # Add detailed milestones
    if 'expert_projects' not in data:
        data['expert_projects'] = {}

    for project_id, project_data in DETAILED_PROJECTS.items():
        data['expert_projects'][project_id] = project_data
        print(f"Added detailed milestones: {project_id}")

    with open(projects_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

    print(f"\nDone!")


if __name__ == "__main__":
    main()

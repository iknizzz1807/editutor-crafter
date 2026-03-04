# 🎯 Project Charter: Inode-Based Filesystem Implementation

## What You Are Building
A fully functional, inode-based filesystem that runs in userspace via the FUSE (Filesystem in Userspace) interface. You are building the entire stack: from a raw block-device abstraction over a 256MB disk image to a hierarchical directory tree and a write-ahead journaling system. By the end, you will mount your filesystem as a real volume on your Linux or macOS machine and use standard tools like `ls`, `vim`, and `git` to manipulate data stored in your custom binary format.

## Why This Project Exists
Storage is often treated as a magical black box where files simply "exist." By building a filesystem from scratch, you strip away those abstractions to face the hard reality of persistent storage: managing fixed-size blocks, handling fragmentation, and ensuring that a sudden power failure doesn't turn your data into garbage. This project teaches you the core architectural patterns used in Linux's ext4, database storage engines, and high-performance I/O systems.

## What You Will Be Able to Do When Done
- **Format raw storage:** Create a `mkfs` utility that partitions a disk image into superblocks, bitmaps, and inode tables.
- **Implement Indirection:** Use direct, single-indirect, and double-indirect pointers to manage files ranging from a few bytes to 1GB.
- **Navigate Hierarchies:** Write a path-resolution engine that traverses directory entries to find inodes.
- **Manage Sparse Files:** Implement logic that allows files to have massive logical sizes without consuming physical blocks for zero-filled "holes."
- **Handle Concurrency:** Use thread-safe locking to allow multiple processes to read and write to your filesystem simultaneously.
- **Ensure Crash Consistency:** Implement a write-ahead journal (WAL) that can replay committed transactions to recover from a `kill -9` or system crash.

## Final Deliverable
A systems-level codebase (approx. 4,000–6,000 lines of C, Rust, or Go) consisting of:
- `mkfs.myfs`: A tool to initialize a disk image.
- `myfs_daemon`: A FUSE-based driver that mounts the image.
- `libmyfs`: A library containing the core logic for block allocation and journaling.
- A test suite that simulates system crashes and verifies 100% metadata consistency upon recovery.

## Is This Project For You?
**You should start this if you:**
- Are comfortable with manual memory management and pointers (C/Rust style).
- Understand bitwise operations (masking, shifting) for bitmap manipulation.
- Have used standard Unix I/O syscalls (`read`, `write`, `lseek`).

**Come back after you've learned:**
- **Struct Alignment and Padding:** Essential for ensuring your in-memory structures match the bytes on disk.
- **Concurrency Basics:** You will need to use Mutexes to prevent metadata corruption during FUSE callbacks.

## Estimated Effort
| Phase | Time |
|-------|------|
| **Milestone 1:** Block Layer and mkfs | ~12 hours |
| **Milestone 2:** Inode Management | ~12 hours |
| **Milestone 3:** Directory Operations | ~12 hours |
| **Milestone 4:** File Read/Write Operations | ~12 hours |
| **Milestone 5:** FUSE Integration | ~12 hours |
| **Milestone 6:** Journaling and Recovery | ~15 hours |
| **Total** | **~75 hours** |

## Definition of Done
The project is complete when:
- The filesystem can be successfully mounted via FUSE and appears as a standard directory.
- Standard Unix commands (`mkdir`, `touch`, `cp`, `rm`, `ls -laR`) work without errors.
- A 100MB file can be copied into the mount, read back, and verified via `md5sum` to be identical.
- The filesystem survives a "Crash Test": killing the daemon mid-write, restarting it, and having the journal replay to a consistent state with no orphaned blocks.

---

# 📚 Before You Read This: Prerequisites & Further Reading

> **Read these first.** The Atlas assumes you are familiar with the foundations below.
> Resources are ordered by when you should encounter them — some before you start, some at specific milestones.

## 🏗️ Filesystem Foundations & Unix Design

**The UNIX Time-Sharing System**
- **Paper**: Dennis Ritchie and Ken Thompson (1974), *The UNIX Time-Sharing System*.
- **Code**: [xv6-public/fs.h](https://github.com/mit-pdos/xv6-public/blob/master/fs.h) (The `dinode` and `dirent` structures).
- **Best Explanation**: *Operating Systems: Three Easy Pieces* (OSTEP), [Chapter 39: Interlude: Files and Directories](https://ostep.org/file-intro.pdf).
- **Why**: This is the genesis of the inode and the "everything is a file" philosophy that dictates your implementation in Milestones 2 and 3.
- **Pedagogical Timing**: Read **BEFORE starting** the project to understand the mental model you are building toward.

**The Design and Implementation of the 4.3BSD UNIX Operating System**
- **Paper**: McKusick et al. (1984), *A Fast File System for UNIX*.
- **Best Explanation**: [The Fast File System (FFS)](https://ostep.org/file-ffs.pdf) (OSTEP Chapter 41).
- **Why**: The definitive text on why block alignment and locality (discussed in M1) matter for physical disk performance.
- **Pedagogical Timing**: Read **after Milestone 1** to understand why you partitioned the disk into specific regions.

## 💾 Block Storage & Alignment

**Coding for SSDs (Part 2: Architecture and Benchmarking)**
- **Best Explanation**: [Emmanuel Goossaert's Blog](https://codecapsule.com/2014/02/12/coding-for-ssds-part-2-architecture-and-benchmarking/), Section: "Pages and Blocks."
- **Why**: Provides the modern hardware context for the "Hardware Soul Checks" regarding 4KB sector alignment in M1.
- **Pedagogical Timing**: Read **before Milestone 1** to appreciate why we use 4096-byte blocks rather than 512-byte sectors.

## 🌲 Inodes and Indirection

**File System Implementation**
- **Best Explanation**: *Operating Systems: Three Easy Pieces* (OSTEP), [Chapter 40: File System Implementation](https://ostep.org/file-implementation.pdf).
- **Why**: Contains the gold-standard visual diagrams for the multi-level indirect pointer trees you implement in Milestone 2.
- **Pedagogical Timing**: Read **before Milestone 2** to visualize the "Zone" logic before coding the pointer arithmetic.

**Sparse Files and Holes**
- **Spec**: [POSIX.1-2017: lseek()](https://pubs.opengroup.org/onlinepubs/9699919799/functions/lseek.html), specifically the `SEEK_HOLE` and `SEEK_DATA` extensions.
- **Why**: Defines the official behavior for the "Swiss Cheese" file logic implemented in Milestone 4.
- **Pedagogical Timing**: Read **after Milestone 2**, before starting Milestone 4.

## 📂 Directory Structures & Path Resolution

**Ext4 Disk Layout: Directory Indexing (HTrees)**
- **Spec**: [Linux Kernel Documentation: Ext4 Disk Layout](https://www.kernel.org/doc/html/latest/admin-guide/ext4.html#index-nodes), Section: "Directory Indexing."
- **Why**: Explains how real filesystems move beyond the linear scan you implemented in M3 to handle directories with millions of files.
- **Pedagogical Timing**: Read **after Milestone 3** to understand the limitations of O(N) directory lookups.

## 🔌 FUSE (Filesystem in Userspace)

**Libfuse Example: Hello World**
- **Code**: [libfuse/example/hello.c](https://github.com/libfuse/libfuse/blob/master/example/hello.c).
- **Best Explanation**: [FUSE Documentation: How it works](https://www.kernel.org/doc/html/latest/filesystems/fuse.html).
- **Why**: The authoritative reference for the kernel-to-userspace bridge you build in Milestone 5.
- **Pedagogical Timing**: Read **before starting Milestone 5** to understand the callback loop.

## 🛡️ Journaling & Crash Consistency

**Analysis and Evolution of Journaling File Systems**
- **Paper**: Prabhakaran et al. (2005), *Analysis and Evolution of Journaling File Systems*.
- **Best Explanation**: *Operating Systems: Three Easy Pieces* (OSTEP), [Chapter 42: Crash Consistency: FSCK and Journaling](https://ostep.org/file-journaling.pdf).
- **Why**: Clearly articulates the "Window of Vulnerability" and the idempotency requirements for journal replay in Milestone 6.
- **Pedagogical Timing**: Read **before Milestone 6** — this is the most conceptually difficult part of the project.

**SQLite Write-Ahead Logging**
- **Best Explanation**: [SQLite.org: Write-Ahead Log](https://www.sqlite.org/wal.html).
- **Why**: A world-class explanation of how WAL (Write-Ahead Logging) enables concurrent readers and writers, a concept introduced in the M6 "Knowledge Cascade."
- **Pedagogical Timing**: Read **after completing Milestone 6** to see how your journaling logic applies to database engines.

## 🛠️ Performance & Concurrency

**The Linux VFS (Virtual File System) Lock: Pathname Lookup**
- **Best Explanation**: [Neil Brown's "Pathname lookup in the Linux kernel"](https://lwn.net/Articles/649115/) (LWN.net series).
- **Why**: Deep dive into how the kernel handles the concurrency issues you face in M5 (global locks vs. per-inode locks).
- **Pedagogical Timing**: Read **after Milestone 5** if you find your FUSE mount is bottlenecked by the global mutex.

## 🔢 Serialization & Endianness

**The Art of Portable Binary Formats**
- **Best Explanation**: [Rob Pike's "The Byte Order Fallacy"](https://commandcenter.blogspot.com/2012/04/byte-order-fallacy.html).
- **Why**: A masterclass on why you should use shift operations for serialization (M1) rather than relying on C struct padding and native endianness.
- **Pedagogical Timing**: Read **during Milestone 1** when implementing the superblock write.

---

# Filesystem Implementation

Build a complete inode-based filesystem from the ground up, starting with raw block I/O and culminating in a FUSE-mountable filesystem with crash-consistent journaling. This project strips away the abstraction layers between your code and persistent storage, revealing how operating systems transform a bag of bytes on disk into the hierarchical file trees we navigate daily. You'll implement every layer: block allocation with bitmaps, inode metadata with multi-level indirect pointers, directory entry management, file I/O with sparse file support, and write-ahead journaling that survives crashes. By the end, you'll understand why databases, container runtimes, and storage engines all share the same fundamental concerns about block alignment, atomic writes, and recovery semantics.



<!-- MS_ID: filesystem-m1 -->
# Block Layer and mkfs
You're about to build the foundation of a filesystem from scratch. Not a wrapper around an existing filesystem — the actual bytes-on-disk structures that an operating system uses to transform a raw storage device into files and directories.
This milestone is about **structure**: how you partition a bag of bytes into meaningful regions, and how you establish the invariants that every subsequent operation must maintain. Get this wrong, and nothing else works. Get it right, and you've created a stage where the drama of file creation, directory traversal, and crash recovery can unfold.
## The Fundamental Tension: Chaos Demands Structure
Here's what you're up against: a storage device is just a numbered sequence of blocks. Block 0, block 1, block 2, ... block N. That's it. No files, no directories, no permissions, no timestamps — just raw capacity.
Your job is to impose order on this chaos. But here's the catch: **every piece of metadata you create consumes space that could have stored user data**. Every block devoted to tracking free blocks is a block not storing a cat video. Every inode entry is overhead. The structure you design determines the efficiency, reliability, and capabilities of your entire filesystem.
The tension manifests in three dimensions:
1. **Locatability vs. fragmentation**: You need to find things quickly, but files grow and shrink unpredictably
2. **Reliability vs. performance**: You could just write data and hope for the best, but crashes would corrupt everything
3. **Simplicity vs. capability**: A minimal filesystem is easy to implement but frustrating to use
Let's start with the most fundamental abstraction: the block device.
## The Block Device Abstraction

> **🔑 Foundation: Block devices and sector alignment**
> 
> ## What It Is
A **block device** is storage hardware that transfers data in fixed-size chunks called blocks (or sectors). Unlike character devices that stream byte-by-byte, block devices require you to read or write entire blocks at a time.
**Sector alignment** means positioning your data so it starts and ends on block boundaries. The traditional sector size is 512 bytes, but modern drives use 4KB (4096 byte) physical sectors. Misaligned writes can force the drive to read-modify-write across multiple physical sectors, degrading performance and increasing wear.
```
Aligned write (4KB sector):
  |--------|--------|--------|
  0     4096     8192    12288
        [====your data====]     ← Fits perfectly
Misaligned write:
  |--------|--------|--------|
  0     4096     8192    12288
     [====your data====]        ← Spans 2 sectors!
```
## Why You Need It Right Now
When implementing a filesystem, you're directly managing how data lands on disk. Every structure you define — superblocks, inodes, bitmaps, data blocks — must consider alignment:
- **Performance**: Aligned writes are atomic from the device's perspective. Misaligned writes require the device to read two sectors, modify both, and write them back.
- **SSD endurance**: Misaligned writes cause write amplification, eating into the drive's write cycle budget.
- **Atomicity guarantees**: If you're journaling or implementing crash consistency, aligned sector writes are your basic unit of atomicity.
- **Direct I/O requirements**: Using `O_DIRECT` on Linux requires alignment to both the sector size and memory page boundaries.
## Key Insight
**Think of sectors as the storage device's word size.** Just as a CPU reads memory in 4 or 8 byte words, a storage device reads media in 4KB "words." You wouldn't try to read a single byte from RAM — the CPU fetches the whole word and extracts what you need. Similarly, when you write 100 bytes to a sector-aligned offset, the device writes the full 4KB sector.
**The practical rule**: Start every on-disk structure at an offset divisible by your sector size (use 4KB to be safe). If you're writing 32 bytes, you still consume a full sector from the filesystem's perspective.

At the hardware level, storage devices don't deal with individual bytes — they operate in chunks called **blocks** (historically called sectors for magnetic disks). A traditional hard disk has 512-byte sectors. Modern disks and SSDs use 4KB physical sectors. Your filesystem will use a **4KB block size** throughout, which aligns nicely with:
- Modern disk physical sectors (no read-modify-write penalties)
- Memory page size (efficient for caching and mmap)
- Most real-world filesystems (ext4, XFS, btrfs all default to 4KB)
The block device abstraction presents a simple interface:
```c
// Block device operations
#define BLOCK_SIZE 4096  // 4 KB
typedef struct {
    int fd;              // File descriptor for backing file
    uint64_t size;       // Total size in bytes
    uint64_t num_blocks; // Total number of blocks
} BlockDevice;
// Read a single block into the provided buffer
int read_block(BlockDevice* dev, uint64_t block_num, void* buffer);
// Write a single block from the provided buffer
int write_block(BlockDevice* dev, uint64_t block_num, const void* data);
```
Behind this simple interface hides a three-level reality:
**Level 1 — Application (Your Filesystem)**: You request block 42. You neither know nor care where it physically resides.
**Level 2 — OS/Kernel**: The kernel's block layer may cache this request (page cache), merge it with adjacent requests (I/O scheduling), or split it across physical devices (RAID, LVM).
**Level 3 — Hardware**: The storage controller translates logical block addresses to physical locations — cylinder/head/sector for HDDs, flash pages for SSDs. A single 4KB read might trigger multiple NAND flash reads with error correction.
For now, you'll back your block device with a regular file. This is exactly how disk images work, and it's how you'll test your filesystem without needing actual hardware.
```c
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#define BLOCK_SIZE 4096
struct BlockDevice {
    int fd;
    uint64_t size;
    uint64_t num_blocks;
};
BlockDevice* block_device_create(const char* path, uint64_t num_blocks) {
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return NULL;
    // Pre-allocate the file to the desired size
    uint64_t size = num_blocks * BLOCK_SIZE;
    if (ftruncate(fd, size) < 0) {
        close(fd);
        return NULL;
    }
    BlockDevice* dev = malloc(sizeof(BlockDevice));
    if (!dev) {
        close(fd);
        return NULL;
    }
    dev->fd = fd;
    dev->size = size;
    dev->num_blocks = num_blocks;
    return dev;
}
int read_block(BlockDevice* dev, uint64_t block_num, void* buffer) {
    if (block_num >= dev->num_blocks) {
        errno = EINVAL;
        return -1;
    }
    off_t offset = block_num * BLOCK_SIZE;
    if (lseek(dev->fd, offset, SEEK_SET) < 0) {
        return -1;
    }
    ssize_t bytes_read = read(dev->fd, buffer, BLOCK_SIZE);
    if (bytes_read < 0) {
        return -1;
    }
    // If we read fewer bytes (shouldn't happen with pre-allocated file),
    // zero-fill the rest
    if (bytes_read < BLOCK_SIZE) {
        memset((char*)buffer + bytes_read, 0, BLOCK_SIZE - bytes_read);
    }
    return 0;
}
int write_block(BlockDevice* dev, uint64_t block_num, const void* data) {
    if (block_num >= dev->num_blocks) {
        errno = EINVAL;
        return -1;
    }
    off_t offset = block_num * BLOCK_SIZE;
    if (lseek(dev->fd, offset, SEEK_SET) < 0) {
        return -1;
    }
    ssize_t bytes_written = write(dev->fd, data, BLOCK_SIZE);
    if (bytes_written < 0) {
        return -1;
    }
    // Ensure data reaches disk (bypass OS cache for reliability)
    // In production, you might batch these or use fdatasync()
    fsync(dev->fd);
    return 0;
}
void block_device_close(BlockDevice* dev) {
    if (dev) {
        fsync(dev->fd);
        close(dev->fd);
        free(dev);
    }
}
```
**Hardware Soul Check**: Each `read_block` and `write_block` touches at least one 4KB cache line (actually 64 cache lines of 64 bytes each). These are **cold** accesses from the CPU's perspective — the data is coming from disk through the kernel's page cache. A single 4KB read that misses all caches and goes to NVMe storage takes ~25 microseconds. To HDD? ~10 milliseconds. That's a 400× difference. Your filesystem design should minimize block reads by caching metadata aggressively.
## On-Disk Layout: The Blueprint
Now comes the critical design question: **how do you organize your blocks?**
A filesystem needs to track:
- **Free blocks** — which blocks are available for allocation?
- **Free inodes** — which metadata slots are available?
- **File metadata** — who owns this file? How big is it? When was it modified?
- **File data** — the actual contents
- **Recovery information** — how do we survive a crash?

![Filesystem Component Atlas](./diagrams/diag-l0-filesystem-map.svg)

Your on-disk layout divides the block address space into distinct regions:
```
Block 0:           Superblock
Block 1:           Block Bitmap (starts here, may span multiple blocks)
Block 1+N:         Inode Bitmap (starts here, may span multiple blocks)
Block 1+N+M:       Inode Table (fixed-size array of inode structures)
Block 1+N+M+K:     Journal Region (for crash recovery)
Remaining blocks:  Data Blocks (file contents and directory entries)
```
Each region has a specific purpose and must be large enough to serve its function. Let's examine each in detail.
### The Superblock: The Master Record
The superblock is the **entry point** to your filesystem. It lives at block 0 — always. When someone mounts your filesystem, the first thing they read is block 0. If it doesn't contain a valid superblock, the filesystem is either corrupt or not formatted.
```c
#define FS_MAGIC 0x46534D4B  // "FSMK" - filesystem marker
#define FS_VERSION 1
typedef struct {
    uint32_t magic;           // Magic number for identification
    uint32_t version;         // Filesystem version
    uint64_t total_blocks;    // Total blocks in the filesystem
    uint64_t total_inodes;    // Total inodes in the inode table
    uint32_t block_size;      // Block size in bytes (4096)
    uint32_t inode_size;      // Inode size in bytes
    uint64_t block_bitmap_start;  // First block of block bitmap
    uint64_t block_bitmap_blocks; // Number of blocks in block bitmap
    uint64_t inode_bitmap_start;  // First block of inode bitmap
    uint64_t inode_bitmap_blocks; // Number of blocks in inode bitmap
    uint64_t inode_table_start;   // First block of inode table
    uint64_t inode_table_blocks;  // Number of blocks in inode table
    uint64_t journal_start;       // First block of journal
    uint64_t journal_blocks;      // Number of blocks in journal
    uint64_t data_start;          // First data block
    uint64_t free_blocks;         // Current count of free blocks
    uint64_t free_inodes;         // Current count of free inodes
    uint64_t mount_time;          // Last mount time
    uint64_t write_time;          // Last write time
    uint8_t reserved[3928];       // Pad to exactly 4096 bytes
} __attribute__((packed)) Superblock;
```

> **🔑 Foundation: Endianness and serialization**
> 
> ## What It Is
**Endianness** describes how multi-byte values are stored in memory:
- **Big-endian**: Most significant byte first (network byte order). Value `0x12345678` stores as `12 34 56 78`.
- **Little-endian**: Least significant byte first. Value `0x12345678` stores as `78 56 34 12`.
```
Value: 0x00000001 (decimal 1)
Big-endian:    [00][00][00][01]  ← Reads left-to-right naturally
Little-endian: [01][00][00][00]  ← "Small end" comes first
```
**Serialization** is the process of converting in-memory data structures into a byte sequence that can be written to disk or sent over a network. The challenge: your program's internal byte order might not match what's on disk or what another system expects.
## Why You Need It Right Now
Filesystems are **long-lived binary formats**. A filesystem you write today might be mounted on a different architecture years from now. Consider:
- **x86/x64** = little-endian
- **ARM** = bi-endian (usually little in practice)
- **RISC-V** = little-endian
- **PowerPC, SPARC** = big-endian (historically)
If you serialize structs by `fwrite(&my_struct, sizeof(my_struct), 1, file)`, you've baked your CPU's endianness into the disk format. When someone mounts that filesystem on a different architecture, every multi-byte field will be misinterpreted.
**This is why network protocols use big-endian** (network byte order) — it's a universal convention that transcends CPU architecture.
## Key Insight
**Treat your on-disk format as a communication protocol with future systems.** You're not just saving data — you're defining a contract.
The robust pattern:
```c
// Writing (serialize to defined byte order)
uint32_t value = 0x12345678;
write_u32_le(file, value);  // Explicitly write little-endian
// Reading (deserialize from defined byte order)
uint32_t value = read_u32_le(file);  // Explicitly read little-endian
```
Use helper functions that convert to/from a canonical byte order. Most filesystems pick **little-endian** for the on-disk format (ext4, XFS, Btrfs) because x86 dominates, but **big-endian** is traditional for network protocols. What matters is *picking one and being explicit*.
**The mental model**: Your code has two representations of every multi-byte value — the CPU's native format (whatever that may be) and the serialized format (defined by you, unchanging). Conversion happens at the serialization boundary, always.

**Critical Design Decision**: Why pack all this metadata at specific offsets? Because **you need to find it without any other information**. Once you have the superblock, you know where everything else lives. The superblock is the root of all knowledge about your filesystem.
**The Magic Number**: The `magic` field (0x46534D4B) is your filesystem's fingerprint. If block 0 doesn't contain this exact value, the device either isn't formatted or is corrupted. Real filesystems use distinctive magic numbers:
- ext2/3/4: 0xEF53 at offset 0x438
- XFS: "XFSB" at offset 0
- btrfs: "_BHRfS_M" at offset 0x40
### Block Bitmap: Tracking Free Space
The block bitmap is exactly what it sounds like: a giant array of bits, one per data block. Bit = 0 means free. Bit = 1 means allocated.
```c
// How many blocks do we need for the bitmap?
// Each block (4096 bytes) can hold 4096 * 8 = 32768 bits
// Each bit tracks one block
// So one bitmap block can track 32768 blocks
#define BITS_PER_BLOCK (BLOCK_SIZE * 8)
uint64_t calculate_bitmap_blocks(uint64_t total_data_blocks) {
    return (total_data_blocks + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;
}
```
Here's the bitmap allocator — the code that finds free blocks:
```c
typedef struct {
    BlockDevice* dev;
    Superblock* sb;
    void* bitmap_buffer;  // Single-block cache for bitmap operations
} FileSystem;
// Find the first free block, allocate it, and return its number
// Returns 0 on error (block 0 is superblock, never a valid data block)
uint64_t alloc_block(FileSystem* fs) {
    uint64_t bitmap_start = fs->sb->block_bitmap_start;
    uint64_t bitmap_blocks = fs->sb->block_bitmap_blocks;
    // Scan bitmap blocks
    for (uint64_t block_idx = 0; block_idx < bitmap_blocks; block_idx++) {
        uint64_t bitmap_block = bitmap_start + block_idx;
        if (read_block(fs->dev, bitmap_block, fs->bitmap_buffer) < 0) {
            return 0;  // Error
        }
        uint64_t* bits = (uint64_t*)fs->bitmap_buffer;
        // Scan 64-bit chunks for a free bit
        for (int word = 0; word < BLOCK_SIZE / 8; word++) {
            if (bits[word] != ~0ULL) {  // Not all 1s means there's a free bit
                // Find the first 0 bit
                int bit = __builtin_ffsll(~bits[word]) - 1;
                if (bit >= 0) {
                    // Mark it as allocated
                    bits[word] |= (1ULL << bit);
                    // Write back the bitmap block
                    if (write_block(fs->dev, bitmap_block, fs->bitmap_buffer) < 0) {
                        return 0;
                    }
                    // Calculate the actual block number
                    uint64_t block_num = fs->sb->data_start + 
                                         (block_idx * BITS_PER_BLOCK) + 
                                         (word * 64) + bit;
                    // Sanity check
                    if (block_num >= fs->sb->total_blocks) {
                        return 0;  // Out of range
                    }
                    fs->sb->free_blocks--;
                    return block_num;
                }
            }
        }
    }
    return 0;  // No free blocks
}
// Free a previously allocated block
int free_block(FileSystem* fs, uint64_t block_num) {
    if (block_num < fs->sb->data_start || block_num >= fs->sb->total_blocks) {
        errno = EINVAL;
        return -1;  // Not a valid data block
    }
    // Calculate position in bitmap
    uint64_t data_block_idx = block_num - fs->sb->data_start;
    uint64_t bitmap_block_idx = data_block_idx / BITS_PER_BLOCK;
    uint64_t bit_in_block = data_block_idx % BITS_PER_BLOCK;
    if (bitmap_block_idx >= fs->sb->block_bitmap_blocks) {
        errno = EINVAL;
        return -1;
    }
    uint64_t bitmap_block = fs->sb->block_bitmap_start + bitmap_block_idx;
    if (read_block(fs->dev, bitmap_block, fs->bitmap_buffer) < 0) {
        return -1;
    }
    uint64_t* bits = (uint64_t*)fs->bitmap_buffer;
    uint64_t word = bit_in_block / 64;
    uint64_t bit = bit_in_block % 64;
    // Check if already free (double-free detection)
    if (!(bits[word] & (1ULL << bit))) {
        errno = EINVAL;
        return -1;  // Already free!
    }
    // Clear the bit
    bits[word] &= ~(1ULL << bit);
    if (write_block(fs->dev, bitmap_block, fs->bitmap_buffer) < 0) {
        return -1;
    }
    fs->sb->free_blocks++;
    return 0;
}
```
**Hardware Soul Check**: This allocator scans linearly through the bitmap. In the worst case, you're reading multiple 4KB blocks from disk just to find one free bit. On a mostly-full filesystem, this could mean reading dozens of bitmap blocks. Real filesystems use optimizations:
- **Caching**: Keep hot bitmap blocks in memory
- **Hints**: Remember where you last found a free block
- **Buddy allocator**: Group blocks by power-of-2 sizes for faster allocation
### Inode Bitmap: Tracking Metadata Slots
The inode bitmap works identically to the block bitmap, but tracks inode slots instead of data blocks. Each bit represents one position in the inode table.
```c
// Allocate an inode slot (returns inode number, 1-indexed in our design)
uint64_t alloc_inode(FileSystem* fs) {
    uint64_t bitmap_start = fs->sb->inode_bitmap_start;
    uint64_t bitmap_blocks = fs->sb->inode_bitmap_blocks;
    for (uint64_t block_idx = 0; block_idx < bitmap_blocks; block_idx++) {
        uint64_t bitmap_block = bitmap_start + block_idx;
        if (read_block(fs->dev, bitmap_block, fs->bitmap_buffer) < 0) {
            return 0;
        }
        uint64_t* bits = (uint64_t*)fs->bitmap_buffer;
        for (int word = 0; word < BLOCK_SIZE / 8; word++) {
            if (bits[word] != ~0ULL) {
                int bit = __builtin_ffsll(~bits[word]) - 1;
                if (bit >= 0) {
                    bits[word] |= (1ULL << bit);
                    if (write_block(fs->dev, bitmap_block, fs->bitmap_buffer) < 0) {
                        return 0;
                    }
                    uint64_t inode_num = (block_idx * BITS_PER_BLOCK) + 
                                         (word * 64) + bit + 1;  // 1-indexed
                    if (inode_num > fs->sb->total_inodes) {
                        return 0;
                    }
                    fs->sb->free_inodes--;
                    return inode_num;
                }
            }
        }
    }
    return 0;  // No free inodes
}
```
### Inode Table: The Metadata Array
The inode table is a contiguous array of inode structures. Given an inode number N, you can calculate its exact location on disk:
```c
// Inode structure (will be expanded in Milestone 2)
typedef struct {
    uint16_t mode;          // File type and permissions
    uint16_t uid;           // Owner user ID
    uint16_t gid;           // Owner group ID
    uint16_t link_count;    // Number of hard links
    uint64_t size;          // File size in bytes
    uint64_t blocks;        // Number of blocks allocated
    uint64_t atime;         // Access time
    uint64_t mtime;         // Modification time
    uint64_t ctime;         // Change time
    uint64_t direct[12];    // Direct block pointers
    uint64_t indirect;      // Single indirect pointer
    uint64_t double_ind;    // Double indirect pointer
    uint8_t reserved[4000]; // Padding to 4096 bytes
} __attribute__((packed)) Inode;
#define INODE_SIZE sizeof(Inode)  // Should be 4096
#define INODES_PER_BLOCK (BLOCK_SIZE / INODE_SIZE)  // 1
// Calculate block containing inode N
uint64_t inode_to_block(FileSystem* fs, uint64_t inode_num) {
    if (inode_num < 1 || inode_num > fs->sb->total_inodes) {
        return 0;  // Invalid
    }
    // Inode numbers are 1-indexed, but stored in 0-indexed slots
    return fs->sb->inode_table_start + (inode_num - 1) / INODES_PER_BLOCK;
}
// Read an inode from disk
int read_inode(FileSystem* fs, uint64_t inode_num, Inode* inode) {
    uint64_t block_num = inode_to_block(fs, inode_num);
    if (block_num == 0) return -1;
    if (read_block(fs->dev, block_num, fs->bitmap_buffer) < 0) {
        return -1;
    }
    // Copy the inode (for now, we have 1 inode per block)
    memcpy(inode, fs->bitmap_buffer, sizeof(Inode));
    return 0;
}
// Write an inode to disk
int write_inode(FileSystem* fs, uint64_t inode_num, const Inode* inode) {
    uint64_t block_num = inode_to_block(fs, inode_num);
    if (block_num == 0) return -1;
    // For 1 inode per block, just write directly
    memcpy(fs->bitmap_buffer, inode, sizeof(Inode));
    return write_block(fs->dev, block_num, fs->bitmap_buffer);
}
```
**Design Decision**: Why 4096-byte inodes when we only need ~100 bytes of actual data? This is wasteful in terms of space, but it simplifies the implementation for learning purposes. Real filesystems pack inodes more densely:
- ext4: 256-byte inodes, 16 inodes per 4KB block
- XFS: Variable inode size, typically 256 or 512 bytes
You could optimize this later by making `INODE_SIZE` configurable.
## The mkfs Tool: Breathing Life into Empty Space
Now comes the moment of truth. `mkfs` (make filesystem) is the tool that transforms an empty file (or partition) into a filesystem skeleton. It's the creation myth of your filesystem: before mkfs, there is only void; after mkfs, there is structure.
Let's trace through what mkfs must accomplish:
```
1. Parse arguments (image size, block count, etc.)
2. Create/resize the backing file
3. Calculate region sizes and positions
4. Write the superblock
5. Initialize the block bitmap (mark reserved blocks as used)
6. Initialize the inode bitmap (mark reserved inodes as used)
7. Initialize the inode table (zero all inodes)
8. Create the root directory (inode 1)
9. Write the journal region header
10. Verify consistency
```
Here's the complete mkfs implementation:
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
// Default filesystem parameters
#define DEFAULT_TOTAL_BLOCKS  65536   // 256 MB with 4KB blocks
#define DEFAULT_TOTAL_INODES  4096    // ~1 inode per 16 blocks
#define JOURNAL_BLOCKS        1024    // 4 MB journal
FileSystem* fs_format(const char* path, 
                      uint64_t total_blocks,
                      uint64_t total_inodes,
                      uint64_t journal_blocks) {
    // Create the block device
    BlockDevice* dev = block_device_create(path, total_blocks);
    if (!dev) {
        fprintf(stderr, "Failed to create block device: %s\n", strerror(errno));
        return NULL;
    }
    // Allocate filesystem structure
    FileSystem* fs = malloc(sizeof(FileSystem));
    if (!fs) {
        block_device_close(dev);
        return NULL;
    }
    fs->dev = dev;
    fs->sb = malloc(BLOCK_SIZE);
    fs->bitmap_buffer = malloc(BLOCK_SIZE);
    if (!fs->sb || !fs->bitmap_buffer) {
        free(fs->sb);
        free(fs->bitmap_buffer);
        free(fs);
        block_device_close(dev);
        return NULL;
    }
    memset(fs->sb, 0, BLOCK_SIZE);
    // Calculate layout
    uint64_t data_blocks = total_blocks;
    // Block 0: superblock
    uint64_t current_block = 1;
    // Block bitmap: need enough bits for all data blocks
    uint64_t block_bitmap_blocks = (data_blocks + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;
    uint64_t block_bitmap_start = current_block;
    current_block += block_bitmap_blocks;
    // Inode bitmap: need enough bits for all inodes
    uint64_t inode_bitmap_blocks = (total_inodes + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK;
    uint64_t inode_bitmap_start = current_block;
    current_block += inode_bitmap_blocks;
    // Inode table: fixed number of inodes, 1 per block (simplified)
    uint64_t inode_table_blocks = total_inodes;  // 1 inode per 4KB block
    uint64_t inode_table_start = current_block;
    current_block += inode_table_blocks;
    // Journal region
    uint64_t journal_start = current_block;
    current_block += journal_blocks;
    // Data blocks: everything else
    uint64_t data_start = current_block;
    // Verify we have room for data blocks
    if (data_start >= total_blocks) {
        fprintf(stderr, "Error: metadata consumes all blocks, no room for data\n");
        free(fs->sb);
        free(fs->bitmap_buffer);
        free(fs);
        block_device_close(dev);
        return NULL;
    }
    uint64_t actual_data_blocks = total_blocks - data_start;
    // Initialize superblock
    Superblock* sb = fs->sb;
    sb->magic = FS_MAGIC;
    sb->version = FS_VERSION;
    sb->total_blocks = total_blocks;
    sb->total_inodes = total_inodes;
    sb->block_size = BLOCK_SIZE;
    sb->inode_size = INODE_SIZE;
    sb->block_bitmap_start = block_bitmap_start;
    sb->block_bitmap_blocks = block_bitmap_blocks;
    sb->inode_bitmap_start = inode_bitmap_start;
    sb->inode_bitmap_blocks = inode_bitmap_blocks;
    sb->inode_table_start = inode_table_start;
    sb->inode_table_blocks = inode_table_blocks;
    sb->journal_start = journal_start;
    sb->journal_blocks = journal_blocks;
    sb->data_start = data_start;
    sb->free_blocks = actual_data_blocks;  // Will adjust after root dir
    sb->free_inodes = total_inodes - 1;    // Reserve inode 1 for root
    sb->write_time = (uint64_t)time(NULL);
    // Write superblock
    if (write_block(dev, 0, sb) < 0) {
        fprintf(stderr, "Failed to write superblock\n");
        free(fs->sb);
        free(fs->bitmap_buffer);
        free(fs);
        block_device_close(dev);
        return NULL;
    }
    // Initialize block bitmap: mark metadata blocks as used
    memset(fs->bitmap_buffer, 0, BLOCK_SIZE);
    // Mark blocks 0 through (data_start - 1) as used
    // These are: superblock, bitmaps, inode table, journal
    uint64_t metadata_blocks = data_start;
    for (uint64_t i = 0; i < metadata_blocks; i++) {
        uint64_t block_idx = i / BITS_PER_BLOCK;
        uint64_t bit = i % BITS_PER_BLOCK;
        uint64_t word = bit / 64;
        uint64_t bit_in_word = bit % 64;
        // Read the correct bitmap block
        if (i % BITS_PER_BLOCK == 0 && i > 0) {
            // Write previous block and start new one
            write_block(dev, block_bitmap_start + (i / BITS_PER_BLOCK) - 1, 
                       fs->bitmap_buffer);
            memset(fs->bitmap_buffer, 0, BLOCK_SIZE);
        }
        ((uint64_t*)fs->bitmap_buffer)[word] |= (1ULL << bit_in_word);
    }
    // Write the last bitmap block (or first if only one)
    write_block(dev, block_bitmap_start + (metadata_blocks / BITS_PER_BLOCK), 
               fs->bitmap_buffer);
    // Zero the rest of the block bitmap blocks
    memset(fs->bitmap_buffer, 0, BLOCK_SIZE);
    for (uint64_t i = (metadata_blocks + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK; 
         i < block_bitmap_blocks; i++) {
        write_block(dev, block_bitmap_start + i, fs->bitmap_buffer);
    }
    // Initialize inode bitmap: mark inode 1 as used (root directory)
    memset(fs->bitmap_buffer, 0, BLOCK_SIZE);
    ((uint64_t*)fs->bitmap_buffer)[0] = 2;  // Bit 1 is set (inode 1)
    write_block(dev, inode_bitmap_start, fs->bitmap_buffer);
    // Zero remaining inode bitmap blocks
    memset(fs->bitmap_buffer, 0, BLOCK_SIZE);
    for (uint64_t i = 1; i < inode_bitmap_blocks; i++) {
        write_block(dev, inode_bitmap_start + i, fs->bitmap_buffer);
    }
    // Initialize inode table: zero all blocks, then create root
    memset(fs->bitmap_buffer, 0, BLOCK_SIZE);
    for (uint64_t i = 0; i < inode_table_blocks; i++) {
        write_block(dev, inode_table_start + i, fs->bitmap_buffer);
    }
    // Create root directory (inode 1)
    // We'll need to allocate a data block for the directory entries
    uint64_t root_block = data_start;  // First data block
    // Mark it as used in the block bitmap
    memset(fs->bitmap_buffer, 0, BLOCK_SIZE);
    read_block(dev, block_bitmap_start, fs->bitmap_buffer);
    uint64_t data_bit = (root_block - data_start);
    ((uint64_t*)fs->bitmap_buffer)[data_bit / 64] |= (1ULL << (data_bit % 64));
    write_block(dev, block_bitmap_start, fs->bitmap_buffer);
    // Update superblock free count
    sb->free_blocks--;
    write_block(dev, 0, sb);
    // Create root inode
    Inode root_inode;
    memset(&root_inode, 0, sizeof(Inode));
    root_inode.mode = 0755 | 0040000;  // rwxr-xr-x + directory flag
    root_inode.uid = 0;   // root
    root_inode.gid = 0;   // root
    root_inode.link_count = 2;  // . and .. both point to this
    root_inode.size = BLOCK_SIZE;
    root_inode.blocks = 1;
    root_inode.atime = root_inode.mtime = root_inode.ctime = (uint64_t)time(NULL);
    root_inode.direct[0] = root_block;
    write_inode(fs, 1, &root_inode);
    // Create root directory entries (. and ..)
    // Directory entry format (simplified)
    typedef struct {
        uint64_t inode;
        uint16_t rec_len;
        uint8_t name_len;
        uint8_t file_type;
        char name[256];
    } __attribute__((packed)) DirEntry;
    char dir_block[BLOCK_SIZE];
    memset(dir_block, 0, BLOCK_SIZE);
    DirEntry* entry1 = (DirEntry*)dir_block;
    entry1->inode = 1;
    entry1->rec_len = sizeof(DirEntry);
    entry1->name_len = 1;
    entry1->file_type = 2;  // Directory
    entry1->name[0] = '.';
    DirEntry* entry2 = (DirEntry*)(dir_block + sizeof(DirEntry));
    entry2->inode = 1;
    entry2->rec_len = BLOCK_SIZE - 2 * sizeof(DirEntry);  // Rest of block
    entry2->name_len = 2;
    entry2->file_type = 2;  // Directory
    entry2->name[0] = '.';
    entry2->name[1] = '.';
    write_block(dev, root_block, dir_block);
    // Initialize journal region (just zero it for now)
    memset(fs->bitmap_buffer, 0, BLOCK_SIZE);
    for (uint64_t i = 0; i < journal_blocks; i++) {
        write_block(dev, journal_start + i, fs->bitmap_buffer);
    }
    printf("Filesystem created successfully:\n");
    printf("  Total blocks: %lu (%.2f MB)\n", 
           total_blocks, (double)total_blocks * BLOCK_SIZE / (1024 * 1024));
    printf("  Total inodes: %lu\n", total_inodes);
    printf("  Data blocks:  %lu (%.2f MB)\n", 
           actual_data_blocks - 1, 
           (double)(actual_data_blocks - 1) * BLOCK_SIZE / (1024 * 1024));
    printf("  Journal:      %lu blocks (%.2f MB)\n", 
           journal_blocks, (double)journal_blocks * BLOCK_SIZE / (1024 * 1024));
    printf("  Metadata overhead: %.2f%%\n", 
           100.0 * (data_start + 1) / total_blocks);
    return fs;
}
void fs_close(FileSystem* fs) {
    if (fs) {
        // Write superblock one more time
        if (fs->sb && fs->dev) {
            fs->sb->write_time = (uint64_t)time(NULL);
            write_block(fs->dev, 0, fs->sb);
        }
        free(fs->sb);
        free(fs->bitmap_buffer);
        block_device_close(fs->dev);
        free(fs);
    }
}
// mkfs main function
int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <image_file> [size_mb]\n", argv[0]);
        return 1;
    }
    const char* path = argv[1];
    uint64_t size_mb = 256;  // Default 256 MB
    if (argc >= 3) {
        size_mb = strtoull(argv[2], NULL, 10);
        if (size_mb < 1) {
            fprintf(stderr, "Invalid size\n");
            return 1;
        }
    }
    uint64_t total_blocks = (size_mb * 1024 * 1024) / BLOCK_SIZE;
    uint64_t total_inodes = total_blocks / 16;  // 1 inode per 16 blocks
    if (total_inodes < 64) total_inodes = 64;
    uint64_t journal_blocks = total_blocks / 64;  // ~1.5% for journal
    if (journal_blocks < 128) journal_blocks = 128;
    if (journal_blocks > 4096) journal_blocks = 4096;
    FileSystem* fs = fs_format(path, total_blocks, total_inodes, journal_blocks);
    if (!fs) {
        return 1;
    }
    // Verify the filesystem by reading it back
    printf("\nVerifying filesystem...\n");
    Superblock* verify_sb = malloc(BLOCK_SIZE);
    if (read_block(fs->dev, 0, verify_sb) < 0) {
        fprintf(stderr, "Failed to read superblock for verification\n");
        free(verify_sb);
        fs_close(fs);
        return 1;
    }
    if (verify_sb->magic != FS_MAGIC) {
        fprintf(stderr, "Verification failed: magic number mismatch\n");
        free(verify_sb);
        fs_close(fs);
        return 1;
    }
    printf("Superblock verified: magic=0x%08X, version=%u\n", 
           verify_sb->magic, verify_sb->version);
    printf("Root directory at inode 1, data block %lu\n", 
           fs->sb->data_start);
    free(verify_sb);
    fs_close(fs);
    printf("Done.\n");
    return 0;
}
```

![Filesystem Layer Stack](./diagrams/diag-l1-layer-stack.svg)

## The Invariant: Your Contract with the Future
When mkfs completes, your filesystem enters a state that must be preserved by every subsequent operation. This is the **filesystem invariant**:
```
INVARIANT:
1. Block 0 always contains a valid superblock with correct magic number
2. The block bitmap accurately reflects which blocks are allocated
3. The inode bitmap accurately reflects which inodes are allocated
4. Every allocated inode's block pointers point to allocated blocks
5. Every directory entry's inode number refers to an allocated inode
6. The free_blocks count equals the number of 0 bits in the block bitmap
7. The free_inodes count equals the number of 0 bits in the inode bitmap
```
Every file creation, every write, every deletion must maintain these invariants. A single violation — a block marked free but actually containing file data, an inode marked allocated but with garbage contents — corrupts the filesystem.
This is why journaling (which you'll implement in Milestone 6) matters: it ensures that even if a crash occurs mid-operation, recovery can restore the invariants.
## Verification: Trust but Verify
After mkfs, you should be able to read back the filesystem and verify consistency. Here's a verification tool:
```c
int verify_filesystem(FileSystem* fs) {
    int errors = 0;
    // Check superblock
    if (fs->sb->magic != FS_MAGIC) {
        fprintf(stderr, "ERROR: Invalid magic number 0x%08X\n", fs->sb->magic);
        return -1;
    }
    if (fs->sb->version > FS_VERSION) {
        fprintf(stderr, "ERROR: Unsupported version %u\n", fs->sb->version);
        return -1;
    }
    // Count free blocks in bitmap
    uint64_t free_count = 0;
    for (uint64_t i = 0; i < fs->sb->block_bitmap_blocks; i++) {
        read_block(fs->dev, fs->sb->block_bitmap_start + i, fs->bitmap_buffer);
        uint64_t* bits = (uint64_t*)fs->bitmap_buffer;
        for (int word = 0; word < BLOCK_SIZE / 8; word++) {
            // Count 0 bits (free blocks)
            uint64_t free_in_word = __builtin_popcountll(~bits[word]);
            free_count += free_in_word;
        }
    }
    // Adjust for blocks beyond our range in the last bitmap block
    uint64_t total_bits = fs->sb->total_blocks;
    uint64_t excess_bits = (fs->sb->block_bitmap_blocks * BITS_PER_BLOCK) - total_bits;
    free_count -= excess_bits;
    if (free_count != fs->sb->free_blocks) {
        fprintf(stderr, "ERROR: Free block count mismatch: bitmap=%lu, superblock=%lu\n",
                free_count, fs->sb->free_blocks);
        errors++;
    }
    // Verify root directory
    Inode root;
    if (read_inode(fs, 1, &root) < 0) {
        fprintf(stderr, "ERROR: Cannot read root inode\n");
        errors++;
    } else {
        if (!(root.mode & 0040000)) {
            fprintf(stderr, "ERROR: Root inode is not a directory\n");
            errors++;
        }
        if (root.direct[0] < fs->sb->data_start) {
            fprintf(stderr, "ERROR: Root directory block pointer invalid\n");
            errors++;
        }
    }
    return errors;
}
```
## Common Pitfalls (Learn from My Pain)
**Off-by-one in bitmap indexing**: Block 0 is the superblock, not a data block. When you calculate `data_bit = (block_num - data_start)`, verify this produces the right index. I've wasted hours debugging "block 0 is always allocated" bugs.
**Not flushing writes**: The OS caches writes. A crash before `fsync()` loses data. For mkfs, always sync. For normal operations, you'll need a more nuanced strategy (see journaling).
**Magic number typos**: `0x46534D4B` vs `0x46534D4b` — one character difference, hours of debugging. Always define constants and use them consistently.
**Struct padding surprises**: C compilers may insert padding between struct members. The `__attribute__((packed))` directive prevents this, but you should verify with `sizeof()`.
**Endianness**: If you ever move this disk image between architectures (x86 to ARM), the byte order of multi-byte integers will differ. For now, we assume little-endian (x86/ARM), but production filesystems store everything in a fixed byte order (usually little-endian).
## What You've Built
At the end of this milestone, you have:
1. **A block device abstraction** that presents a clean read/write interface over a file
2. **A superblock** at block 0 that describes the entire filesystem layout
3. **A block bitmap** that tracks which data blocks are free or allocated
4. **An inode bitmap** that tracks which inode slots are free or allocated
5. **An inode table** with a root directory already created
6. **A mkfs tool** that can initialize a fresh filesystem image
You can now:
```bash
$ ./mkfs disk.img 100   # Create a 100MB filesystem
$ hexdump -C disk.img | head -20  # Examine the superblock
$ ./verify disk.img     # Verify consistency
```
## Looking Forward: The Knowledge Cascade
The concepts you've mastered here extend far beyond filesystems:
**Database Storage Engines**: PostgreSQL, SQLite, and MySQL all use block-based storage with similar concepts — page headers, free space maps, and fixed-layout pages. The block bitmap is analogous to PostgreSQL's Free Space Map (FSM). If you ever implement a database buffer pool or storage engine, you'll use these exact patterns.
**Memory Allocators**: jemalloc and tcmalloc use bitmap-based allocation for their slab allocators. A "slab" is a contiguous region of memory divided into fixed-size objects, with a bitmap tracking which are allocated. The `alloc_block` function you wrote is essentially `malloc` at a larger scale.
**RAID and Volume Managers**: Linux's mdadm and LVM layer multiple block devices into a single address space, just like your filesystem layers multiple regions. Understanding how LVM calculates extent positions is the same mental model as calculating inode positions.
**Flash Storage Wear Leveling**: SSDs internally maintain a block mapping table (the FTL — Flash Translation Layer) that maps logical block addresses to physical flash pages. When you understand why a filesystem needs a bitmap, you understand why an SSD needs an FTL: both are solving the allocation problem, just at different layers.
**Disk Partitioning**: MBR (Master Boot Record) and GPT (GUID Partition Table) are themselves filesystem-like structures. MBR's partition table lives at offset 0x1BE in sector 0. GPT has a header at LBA 1 and a backup at the last LBA. The principle is identical: fixed locations for critical metadata.
In the next milestone, you'll implement the full inode structure with direct and indirect block pointers, enabling files to grow beyond a few blocks to gigabytes in size. The bitmap allocator you built today will find the blocks; the inode structure will track which blocks belong to which file.
---
<!-- END_MS -->


<!-- MS_ID: filesystem-m2 -->
# Inode Management
You've built the foundation: a block device, a superblock, bitmaps for tracking free space, and a mkfs tool that breathes structure into raw bytes. Now you face the central question of any filesystem: **how do you connect a file's identity to its scattered data blocks?**
The answer lives in the **inode** — the metadata record that bridges the gap between "file" as a concept and "file" as physical storage. This milestone is about building that bridge with the right balance of simplicity and scalability.
## The Fundamental Tension: Files Grow, Pointers Don't
Here's the problem that will occupy your entire design: a file can be 1 byte or 100 gigabytes. But an inode is a **fixed-size structure**. How do you encode a variable-length file's block locations in a fixed number of bytes?
You could give every inode thousands of block pointers — enough to address the largest possible file. But then small files (most files) waste enormous space. A 1KB text file shouldn't require a 64KB inode.
You could store block pointers in a separate, growable data structure — a linked list, perhaps. But now reading a file requires following a chain of pointers, each potentially requiring a disk seek. A 1GB file would need 262,144 linked-list traversals just to find all its blocks.
The solution is **indirection** — a technique you'll recognize from page tables, B-trees, and network routing. Store a few pointers directly for fast access to small files, then use those pointers to reference *other* pointer blocks when files grow large.
The math is beautiful: with just 14 pointer slots in your inode (12 direct + 1 single-indirect + 1 double-indirect), you can address files up to 4 gigabytes. The trick is that most of those pointers live in data blocks you allocate only when needed.
Let's see how.
## The Inode Structure: Metadata and Pointers
An inode contains two categories of information:
1. **File metadata**: Information *about* the file (size, permissions, timestamps)
2. **Block pointers**: Information *locating* the file's data
```c
// File type constants (stored in the high bits of mode)
#define S_IFIFO  0010000  // Named pipe
#define S_IFCHR  0020000  // Character device
#define S_IFDIR  0040000  // Directory
#define S_IFBLK  0060000  // Block device
#define S_IFREG  0100000  // Regular file
#define S_IFLNK  0120000  // Symbolic link
#define S_IFSOCK 0140000  // Socket
// Permission bits (stored in the low 12 bits of mode)
#define S_ISUID  0004000  // Set-user-ID
#define S_ISGID  0002000  // Set-group-ID
#define S_ISVTX  0001000  // Sticky bit
#define S_IRWXU  00700    // Owner permissions
#define S_IRWXG  00070    // Group permissions
#define S_IRWXO  00007    // Other permissions
typedef struct {
    // Metadata fields
    uint16_t mode;          // File type and permissions (see constants above)
    uint16_t uid;           // Owner user ID
    uint16_t gid;           // Owner group ID
    uint16_t link_count;    // Number of hard links to this file
    uint64_t size;          // File size in bytes
    uint64_t blocks;        // Number of 512-byte blocks allocated (stat convention)
    uint64_t atime;         // Access time (seconds since epoch)
    uint64_t mtime;         // Modification time (content changed)
    uint64_t ctime;         // Change time (metadata or content changed)
    // Block pointers
    uint64_t direct[12];    // Direct block pointers (blocks 0-11)
    uint64_t indirect;      // Single-indirect block pointer
    uint64_t double_ind;    // Double-indirect block pointer
    uint64_t triple_ind;    // Triple-indirect (reserved, unused in this design)
    uint8_t reserved[3960]; // Padding to 4096 bytes
} __attribute__((packed)) Inode;
#define INODE_SIZE 4096
#define PTRS_PER_BLOCK (BLOCK_SIZE / sizeof(uint64_t))  // 512 pointers per 4KB block
```
**Hardware Soul Check**: The `mode` field packs two concepts into 16 bits: file type (4 bits) and permissions (12 bits). This isn't just space efficiency — it's how Unix has represented files since the 1970s. The `__attribute__((packed))` directive tells the compiler not to insert padding between fields, ensuring the struct's memory layout exactly matches what we want on disk. Without this, the compiler might insert 2 bytes after `mode` to align `uid` on a 4-byte boundary.
### Understanding the Block Pointer Scheme
The pointer scheme creates three "zones" of file size:
```
┌─────────────────────────────────────────────────────────────────┐
│                        INODE STRUCTURE                          │
├─────────────────────────────────────────────────────────────────┤
│  direct[0]  ─────► Block containing file bytes 0-4095           │
│  direct[1]  ─────► Block containing file bytes 4096-8191        │
│  ...                                                            │
│  direct[11] ─────► Block containing file bytes 45056-49151      │
│                                                                 │
│  indirect    ─────► INDIRECT BLOCK (512 pointers)               │
│                        ├─► Block 49152-53247                    │
│                        ├─► Block 53248-57343                    │
│                        └─► ... (+512 blocks total = 2MB)        │
│                                                                 │
│  double_ind  ─────► DOUBLE-INDIRECT BLOCK (512 indirect ptrs)   │
│                        ├─► Indirect Block 0                     │
│                        │    ├─► Data Block                      │
│                        │    └─► ... (512 data blocks)           │
│                        ├─► Indirect Block 1                     │
│                        └─► ... (512 × 512 = 262144 blocks)      │
└─────────────────────────────────────────────────────────────────┘
```
**Zone 1: Direct Pointers (0 - 48KB)**
The first 12 block pointers are stored directly in the inode. Reading any byte in this range requires exactly one block read — the data block itself. This is the fast path.
**Zone 2: Single-Indirect (48KB - ~4.1MB)**
When a file grows beyond 48KB, we allocate an **indirect block** — a 4KB block containing 512 block pointers (4096 bytes ÷ 8 bytes per pointer). This single block can address 512 × 4KB = 2MB of additional file data.
Reading a byte in this range requires two block reads: first the indirect block, then the data block.
**Zone 3: Double-Indirect (~4.1MB - ~4.2GB)**
For truly large files, we allocate a **double-indirect block** — a block containing 512 pointers to indirect blocks. Each of those indirect blocks points to 512 data blocks. Total capacity: 512 × 512 × 4KB = 1GB.
Reading a byte in this range requires three block reads: double-indirect → indirect → data.
Wait, didn't I say 4GB earlier? Let's recalculate:
- Direct: 12 × 4KB = 48KB
- Single-indirect: 512 × 4KB = 2MB
- Double-indirect: 512 × 512 × 4KB = 1GB
Total: **~1.05GB** with our 4KB block size and 8-byte pointers.
To reach 4GB, you'd need triple-indirect pointers (which we've reserved but not implemented). Alternatively, some filesystems use 4-byte block numbers, fitting 1024 pointers per block instead of 512.
**Design Decision: Why Stop at Double-Indirect?**
| Option | Max File Size | Pointer Reads (worst case) | Complexity | Used By |
|--------|---------------|---------------------------|------------|---------|
| **Double-indirect ✓** | ~1GB | 3 | Moderate | ext2, early Unix |
| Triple-indirect | ~4GB+ | 4 | Higher | ext4 (with extents) |
| Extents | TB range | 2-3 | Different model | ext4, XFS, btrfs |
For this learning project, 1GB files are more than sufficient. Real filesystems use **extents** (contiguous ranges stored as start_block + length) instead of pointer trees, reducing overhead for large contiguous files. But extents require a different mental model — the pointer tree you're building teaches fundamental indirection patterns you'll see in page tables, B-trees, and network routing.
## Calculating Block Offsets
The core operation you'll implement repeatedly: given a file offset (byte position), determine which block contains that byte and how to reach it.
```c
// Block pointer categories
#define DIRECT_BLOCKS       12
#define PTRS_PER_INDIRECT   512   // BLOCK_SIZE / sizeof(uint64_t)
#define BLOCKS_PER_INDIRECT 512   // Same as PTRS_PER_INDIRECT
// File size boundaries
#define DIRECT_MAX_SIZE     (DIRECT_BLOCKS * BLOCK_SIZE)              // 49152 bytes
#define SINGLE_INDIRECT_MAX (DIRECT_MAX_SIZE + PTRS_PER_INDIRECT * BLOCK_SIZE)  // ~2.1MB
#define DOUBLE_INDIRECT_MAX (SINGLE_INDIRECT_MAX + PTRS_PER_INDIRECT * BLOCKS_PER_INDIRECT * BLOCK_SIZE)  // ~1GB
typedef struct {
    int category;        // 0=direct, 1=single-indirect, 2=double-indirect
    int direct_idx;      // Index into direct[] array (if category 0)
    int indirect_idx;    // Index into indirect block (if category 1 or 2)
    int double_idx;      // Index into double-indirect block (if category 2)
    uint64_t block_offset; // Block offset within file (block_num = offset / BLOCK_SIZE)
} BlockLocation;
// Given a file byte offset, calculate where to find the block pointer
BlockLocation locate_block(uint64_t file_offset) {
    BlockLocation loc;
    uint64_t block_offset = file_offset / BLOCK_SIZE;
    loc.block_offset = block_offset;
    if (block_offset < DIRECT_BLOCKS) {
        // Direct block
        loc.category = 0;
        loc.direct_idx = (int)block_offset;
    } else if (block_offset < DIRECT_BLOCKS + PTRS_PER_INDIRECT) {
        // Single-indirect block
        loc.category = 1;
        loc.indirect_idx = (int)(block_offset - DIRECT_BLOCKS);
    } else if (block_offset < DIRECT_BLOCKS + PTRS_PER_INDIRECT + 
                          (uint64_t)PTRS_PER_INDIRECT * BLOCKS_PER_INDIRECT) {
        // Double-indirect block
        loc.category = 2;
        uint64_t remaining = block_offset - DIRECT_BLOCKS - PTRS_PER_INDIRECT;
        loc.double_idx = (int)(remaining / BLOCKS_PER_INDIRECT);
        loc.indirect_idx = (int)(remaining % BLOCKS_PER_INDIRECT);
    } else {
        // Beyond our addressing capability
        loc.category = -1;  // Error: file too large
    }
    return loc;
}
```
**Hardware Soul Check**: This function is pure arithmetic — no disk access, no memory allocation. It runs in nanoseconds. But its output determines whether you'll need 1, 2, or 3 disk reads to get a byte of data. A file with 4MB of random accesses in the double-indirect zone will trigger three times as many disk reads as the same accesses in the direct zone. This is why databases and storage engines obsess over locality — keeping related data in the same block minimizes indirection overhead.
## Reading a Block Through Indirection
Now let's implement the actual block retrieval:
```c
// Read a block pointer from an indirect block
// Returns 0 if the pointer is null (sparse hole) or on error
static uint64_t read_indirect_ptr(FileSystem* fs, uint64_t indirect_block, int index) {
    if (indirect_block == 0) {
        return 0;  // No indirect block allocated = hole
    }
    if (index < 0 || index >= PTRS_PER_INDIRECT) {
        return 0;  // Invalid index
    }
    char buffer[BLOCK_SIZE];
    if (read_block(fs->dev, indirect_block, buffer) < 0) {
        return 0;
    }
    uint64_t* ptrs = (uint64_t*)buffer;
    return ptrs[index];  // 0 means hole, non-zero is a block number
}
// Get the physical block number for a file's logical block offset
// Returns 0 if the block is a hole (unallocated) or on error
uint64_t get_file_block(FileSystem* fs, Inode* inode, uint64_t block_offset) {
    if (block_offset < DIRECT_BLOCKS) {
        // Direct block
        return inode->direct[block_offset];
    }
    block_offset -= DIRECT_BLOCKS;
    if (block_offset < PTRS_PER_INDIRECT) {
        // Single-indirect block
        return read_indirect_ptr(fs, inode->indirect, (int)block_offset);
    }
    block_offset -= PTRS_PER_INDIRECT;
    if (block_offset < (uint64_t)PTRS_PER_INDIRECT * BLOCKS_PER_INDIRECT) {
        // Double-indirect block
        int double_idx = (int)(block_offset / BLOCKS_PER_INDIRECT);
        int indirect_idx = (int)(block_offset % BLOCKS_PER_INDIRECT);
        // First, get the indirect block number from double-indirect
        uint64_t indirect_block = read_indirect_ptr(fs, inode->double_ind, double_idx);
        if (indirect_block == 0) {
            return 0;  // Hole
        }
        // Then, get the data block number from indirect
        return read_indirect_ptr(fs, indirect_block, indirect_idx);
    }
    // Beyond double-indirect range
    return 0;
}
```
Notice the pattern: each level of indirection is a function call that reads a block and returns a pointer. The double-indirect case chains two such lookups. If we implemented triple-indirect, it would chain three.
**The Sparse File Revelation**: A null pointer (value 0) doesn't mean "error" — it means "unallocated." If you read from an unallocated block, the filesystem should return zeros. If you write to an unallocated block, the filesystem should allocate it first. This is how **sparse files** work: a file can have a logical size of 1TB but only allocate blocks where data was actually written.
```c
// Read data from a file at the given offset
// Returns number of bytes read, or -1 on error
int file_read(FileSystem* fs, uint64_t inode_num, uint64_t offset, 
              void* buffer, size_t length) {
    Inode inode;
    if (read_inode(fs, inode_num, &inode) < 0) {
        return -1;
    }
    // Check bounds
    if (offset >= inode.size) {
        return 0;  // Reading past end of file = 0 bytes
    }
    if (offset + length > inode.size) {
        length = inode.size - offset;  // Clamp to file size
    }
    size_t bytes_read = 0;
    char* out = (char*)buffer;
    char block_buffer[BLOCK_SIZE];
    while (bytes_read < length) {
        // Calculate which block and offset within block
        uint64_t file_block = (offset + bytes_read) / BLOCK_SIZE;
        uint64_t block_offset = (offset + bytes_read) % BLOCK_SIZE;
        // Get the physical block number
        uint64_t phys_block = get_file_block(fs, &inode, file_block);
        // How many bytes to read from this block?
        size_t bytes_in_block = BLOCK_SIZE - block_offset;
        size_t bytes_remaining = length - bytes_read;
        size_t to_read = (bytes_in_block < bytes_remaining) ? bytes_in_block : bytes_remaining;
        if (phys_block == 0) {
            // Hole: return zeros
            memset(out + bytes_read, 0, to_read);
        } else {
            // Read the actual block
            if (read_block(fs->dev, phys_block, block_buffer) < 0) {
                return -1;
            }
            memcpy(out + bytes_read, block_buffer + block_offset, to_read);
        }
        bytes_read += to_read;
    }
    // Update access time
    inode.atime = (uint64_t)time(NULL);
    write_inode(fs, inode_num, &inode);
    return (int)bytes_read;
}
```
## Writing and Block Allocation
Writing is where things get interesting. You can't just write to a block — you must first ensure the block exists, allocating it if necessary. And if the file grows into a new indirection zone, you must allocate the indirect blocks too.
```c
// Allocate a new block and return its number
// Returns 0 on failure (block 0 is never a valid data block)
static uint64_t allocate_data_block(FileSystem* fs) {
    return alloc_block(fs);  // From Milestone 1
}
// Write a block pointer into an indirect block, allocating the indirect block if needed
// Returns 0 on success, -1 on error
static int write_indirect_ptr(FileSystem* fs, uint64_t* indirect_block_ptr, 
                               int index, uint64_t data_block) {
    char buffer[BLOCK_SIZE];
    if (*indirect_block_ptr == 0) {
        // Need to allocate the indirect block
        uint64_t new_indirect = allocate_data_block(fs);
        if (new_indirect == 0) {
            return -1;  // No space
        }
        *indirect_block_ptr = new_indirect;
        memset(buffer, 0, BLOCK_SIZE);
    } else {
        if (read_block(fs->dev, *indirect_block_ptr, buffer) < 0) {
            return -1;
        }
    }
    // Write the pointer
    uint64_t* ptrs = (uint64_t*)buffer;
    ptrs[index] = data_block;
    return write_block(fs->dev, *indirect_block_ptr, buffer);
}
// Ensure a block exists at the given file offset, allocating if necessary
// Returns the physical block number, or 0 on error
uint64_t get_or_alloc_block(FileSystem* fs, uint64_t inode_num, 
                             Inode* inode, uint64_t block_offset) {
    // Check direct blocks first
    if (block_offset < DIRECT_BLOCKS) {
        if (inode->direct[block_offset] == 0) {
            uint64_t new_block = allocate_data_block(fs);
            if (new_block == 0) return 0;
            inode->direct[block_offset] = new_block;
            inode->blocks++;  // Track allocated blocks
        }
        return inode->direct[block_offset];
    }
    block_offset -= DIRECT_BLOCKS;
    // Single-indirect
    if (block_offset < PTRS_PER_INDIRECT) {
        int idx = (int)block_offset;
        uint64_t existing = read_indirect_ptr(fs, inode->indirect, idx);
        if (existing == 0) {
            uint64_t new_block = allocate_data_block(fs);
            if (new_block == 0) return 0;
            if (write_indirect_ptr(fs, &inode->indirect, idx, new_block) < 0) {
                free_block(fs, new_block);  // Clean up on failure
                return 0;
            }
            inode->blocks++;
        }
        return read_indirect_ptr(fs, inode->indirect, idx);
    }
    block_offset -= PTRS_PER_INDIRECT;
    // Double-indirect
    if (block_offset < (uint64_t)PTRS_PER_INDIRECT * BLOCKS_PER_INDIRECT) {
        int double_idx = (int)(block_offset / BLOCKS_PER_INDIRECT);
        int indirect_idx = (int)(block_offset % BLOCKS_PER_INDIRECT);
        // Get or allocate the indirect block
        uint64_t indirect_block = read_indirect_ptr(fs, inode->double_ind, double_idx);
        if (indirect_block == 0) {
            // Allocate indirect block
            uint64_t new_indirect = allocate_data_block(fs);
            if (new_indirect == 0) return 0;
            if (write_indirect_ptr(fs, &inode->double_ind, double_idx, new_indirect) < 0) {
                free_block(fs, new_indirect);
                return 0;
            }
            indirect_block = new_indirect;
        }
        // Now get or allocate the data block
        char buffer[BLOCK_SIZE];
        if (read_block(fs->dev, indirect_block, buffer) < 0) return 0;
        uint64_t* ptrs = (uint64_t*)buffer;
        if (ptrs[indirect_idx] == 0) {
            uint64_t new_block = allocate_data_block(fs);
            if (new_block == 0) return 0;
            ptrs[indirect_idx] = new_block;
            if (write_block(fs->dev, indirect_block, buffer) < 0) {
                free_block(fs, new_block);
                return 0;
            }
            inode->blocks++;
        }
        return ptrs[indirect_idx];
    }
    // Beyond double-indirect
    return 0;
}
```
**Hardware Soul Check**: Writing a single byte at file offset 5MB triggers this cascade:
1. Read the double-indirect block (if it exists)
2. Read the appropriate indirect block (if it exists)
3. Write the data to the data block
4. Write the data block
5. If any block was newly allocated, write the indirect or double-indirect block
6. Write the inode (to update size, blocks count, mtime)
That's 2-6 disk writes for one byte! This is why real filesystems buffer writes in memory and batch them. It's also why SSDs have write amplification problems — a small logical write becomes multiple physical writes.
## File Write Implementation
Now the full write operation:
```c
// Write data to a file at the given offset
// Returns number of bytes written, or -1 on error
int file_write(FileSystem* fs, uint64_t inode_num, uint64_t offset,
               const void* data, size_t length) {
    Inode inode;
    if (read_inode(fs, inode_num, &inode) < 0) {
        return -1;
    }
    // For now, we don't support extending files beyond double-indirect range
    uint64_t max_offset = DOUBLE_INDIRECT_MAX;
    if (offset + length > max_offset) {
        length = max_offset - offset;
        if (length == 0) return -1;  // File too large
    }
    size_t bytes_written = 0;
    const char* in = (const char*)data;
    char block_buffer[BLOCK_SIZE];
    while (bytes_written < length) {
        uint64_t file_block = (offset + bytes_written) / BLOCK_SIZE;
        uint64_t block_offset = (offset + bytes_written) % BLOCK_SIZE;
        // Get or allocate the physical block
        uint64_t phys_block = get_or_alloc_block(fs, inode_num, &inode, file_block);
        if (phys_block == 0) {
            // Allocation failed - save what we've written
            break;
        }
        // Read existing block content (for partial writes)
        if (read_block(fs->dev, phys_block, block_buffer) < 0) {
            break;
        }
        // How many bytes to write to this block?
        size_t bytes_in_block = BLOCK_SIZE - block_offset;
        size_t bytes_remaining = length - bytes_written;
        size_t to_write = (bytes_in_block < bytes_remaining) ? bytes_in_block : bytes_remaining;
        // Modify the block
        memcpy(block_buffer + block_offset, in + bytes_written, to_write);
        // Write it back
        if (write_block(fs->dev, phys_block, block_buffer) < 0) {
            break;
        }
        bytes_written += to_write;
    }
    // Update inode metadata
    if (offset + bytes_written > inode.size) {
        inode.size = offset + bytes_written;
    }
    inode.mtime = (uint64_t)time(NULL);
    inode.ctime = inode.mtime;
    // Write the inode back
    if (write_inode(fs, inode_num, &inode) < 0) {
        return -1;
    }
    return (int)bytes_written;
}
```
## Truncation: Growing and Shrinking Files
Truncation is the inverse of allocation. When shrinking a file, you must free all blocks beyond the new size. When growing a file, you update the size but don't allocate blocks (they become holes).
```c
// Free all blocks in an indirect block
static void free_indirect_blocks(FileSystem* fs, uint64_t indirect_block) {
    if (indirect_block == 0) return;
    char buffer[BLOCK_SIZE];
    if (read_block(fs->dev, indirect_block, buffer) < 0) return;
    uint64_t* ptrs = (uint64_t*)buffer;
    for (int i = 0; i < PTRS_PER_INDIRECT; i++) {
        if (ptrs[i] != 0) {
            free_block(fs, ptrs[i]);
        }
    }
    free_block(fs, indirect_block);
}
// Free all blocks in a double-indirect block
static void free_double_indirect_blocks(FileSystem* fs, uint64_t double_ind_block) {
    if (double_ind_block == 0) return;
    char buffer[BLOCK_SIZE];
    if (read_block(fs->dev, double_ind_block, buffer) < 0) return;
    uint64_t* ptrs = (uint64_t*)buffer;
    for (int i = 0; i < PTRS_PER_INDIRECT; i++) {
        if (ptrs[i] != 0) {
            free_indirect_blocks(fs, ptrs[i]);
        }
    }
    free_block(fs, double_ind_block);
}
// Truncate a file to the given size
int file_truncate(FileSystem* fs, uint64_t inode_num, uint64_t new_size) {
    Inode inode;
    if (read_inode(fs, inode_num, &inode) < 0) {
        return -1;
    }
    if (new_size == inode.size) {
        return 0;  // Nothing to do
    }
    if (new_size > inode.size) {
        // Growing: just update size (new regions become holes)
        inode.size = new_size;
        inode.mtime = (uint64_t)time(NULL);
        inode.ctime = inode.mtime;
        return write_inode(fs, inode_num, &inode);
    }
    // Shrinking: need to free blocks
    uint64_t old_blocks = (inode.size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint64_t new_blocks = (new_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Free direct blocks
    for (uint64_t i = new_blocks; i < old_blocks && i < DIRECT_BLOCKS; i++) {
        if (inode.direct[i] != 0) {
            free_block(fs, inode.direct[i]);
            inode.direct[i] = 0;
        }
    }
    // Handle freeing indirect blocks
    if (old_blocks > DIRECT_BLOCKS) {
        // Need to free blocks in the indirect region
        uint64_t indirect_start = DIRECT_BLOCKS;
        uint64_t indirect_end = DIRECT_BLOCKS + PTRS_PER_INDIRECT;
        if (new_blocks <= indirect_start && old_blocks > indirect_start) {
            // Freeing the entire single-indirect region
            if (inode.indirect != 0) {
                char buffer[BLOCK_SIZE];
                read_block(fs->dev, inode.indirect, buffer);
                uint64_t* ptrs = (uint64_t*)buffer;
                for (int i = 0; i < PTRS_PER_INDIRECT; i++) {
                    if (ptrs[i] != 0) {
                        free_block(fs, ptrs[i]);
                    }
                }
                free_block(fs, inode.indirect);
                inode.indirect = 0;
            }
        } else if (new_blocks < indirect_end && old_blocks > indirect_start) {
            // Partially freeing the indirect region
            char buffer[BLOCK_SIZE];
            if (inode.indirect != 0 && read_block(fs->dev, inode.indirect, buffer) == 0) {
                uint64_t* ptrs = (uint64_t*)buffer;
                uint64_t first_to_free = new_blocks > indirect_start ? 
                                          new_blocks - indirect_start : 0;
                for (uint64_t i = first_to_free; i < PTRS_PER_INDIRECT; i++) {
                    if (ptrs[i] != 0) {
                        free_block(fs, ptrs[i]);
                        ptrs[i] = 0;
                    }
                }
                write_block(fs->dev, inode.indirect, buffer);
            }
        }
    }
    // Handle freeing double-indirect blocks
    if (old_blocks > DIRECT_BLOCKS + PTRS_PER_INDIRECT) {
        // If we're truncating below the double-indirect start, free everything
        uint64_t double_start = DIRECT_BLOCKS + PTRS_PER_INDIRECT;
        if (new_blocks <= double_start) {
            if (inode.double_ind != 0) {
                free_double_indirect_blocks(fs, inode.double_ind);
                inode.double_ind = 0;
            }
        }
        // Partial freeing of double-indirect is complex - for simplicity,
        // we'll just free everything past the indirect region
        // A production filesystem would do this more carefully
    }
    // Update inode
    inode.size = new_size;
    inode.blocks = count_allocated_blocks(&inode);  // Helper to recount
    inode.mtime = (uint64_t)time(NULL);
    inode.ctime = inode.mtime;
    return write_inode(fs, inode_num, &inode);
}
```
## Inode Deallocation: Freeing Everything
When deleting a file, you must free all blocks referenced by the inode, including indirect blocks:
```c
// Free all blocks associated with an inode
int inode_free_all_blocks(FileSystem* fs, Inode* inode) {
    // Free direct blocks
    for (int i = 0; i < DIRECT_BLOCKS; i++) {
        if (inode->direct[i] != 0) {
            free_block(fs, inode->direct[i]);
            inode->direct[i] = 0;
        }
    }
    // Free single-indirect and its children
    if (inode->indirect != 0) {
        free_indirect_blocks(fs, inode->indirect);
        inode->indirect = 0;
    }
    // Free double-indirect and its children
    if (inode->double_ind != 0) {
        free_double_indirect_blocks(fs, inode->double_ind);
        inode->double_ind = 0;
    }
    inode->blocks = 0;
    inode->size = 0;
    return 0;
}
// Deallocate an inode entirely
int inode_deallocate(FileSystem* fs, uint64_t inode_num) {
    if (inode_num < 1 || inode_num > fs->sb->total_inodes) {
        errno = EINVAL;
        return -1;
    }
    Inode inode;
    if (read_inode(fs, inode_num, &inode) < 0) {
        return -1;
    }
    // Free all data blocks
    inode_free_all_blocks(fs, &inode);
    // Clear the inode on disk
    memset(&inode, 0, sizeof(Inode));
    write_inode(fs, inode_num, &inode);
    // Free the inode in the bitmap
    // (Implementation depends on your bitmap functions from Milestone 1)
    uint64_t bitmap_block_idx = (inode_num - 1) / BITS_PER_BLOCK;
    uint64_t bit_in_block = (inode_num - 1) % BITS_PER_BLOCK;
    char buffer[BLOCK_SIZE];
    if (read_block(fs->dev, fs->sb->inode_bitmap_start + bitmap_block_idx, buffer) < 0) {
        return -1;
    }
    uint64_t* bits = (uint64_t*)buffer;
    uint64_t word = bit_in_block / 64;
    uint64_t bit = bit_in_block % 64;
    bits[word] &= ~(1ULL << bit);  // Clear the bit
    write_block(fs->dev, fs->sb->inode_bitmap_start + bitmap_block_idx, buffer);
    fs->sb->free_inodes++;
    return 0;
}
```

![Filesystem Layer Stack](./diagrams/diag-l1-layer-stack.svg)

## Timestamp Management
Unix files track three timestamps, each with different update semantics:
```c
// Update timestamps based on operation type
void update_timestamps(Inode* inode, int operation) {
    uint64_t now = (uint64_t)time(NULL);
    switch (operation) {
        case TS_ACCESS:  // File was read
            inode->atime = now;
            break;
        case TS_MODIFY:  // File content was modified
            inode->mtime = now;
            inode->ctime = now;  // ctime also changes
            break;
        case TS_CHANGE:  // Metadata was changed (chmod, chown, etc.)
            inode->ctime = now;
            break;
        case TS_ALL:     // New file created
            inode->atime = now;
            inode->mtime = now;
            inode->ctime = now;
            break;
    }
}
#define TS_ACCESS  1
#define TS_MODIFY  2
#define TS_CHANGE  3
#define TS_ALL     4
```
**Why Three Timestamps?**
- **atime (access time)**: When the file was last read. Backup programs use this to find recently-accessed files. Some systems disable atime updates for performance (mount with `noatime`).
- **mtime (modification time)**: When the file *content* was last changed. This is what `ls -l` shows. It's used by `make` to determine if a file needs recompilation.
- **ctime (change time)**: When the file *metadata* was last changed. This includes permission changes, ownership changes, and link count changes. It's also updated when mtime changes. Note: `ctime` does NOT mean "creation time" — that's a common misconception.
## Link Counting: Hard Links
The `link_count` field tracks how many directory entries point to this inode. A file is only truly deleted when `link_count` reaches zero AND no process has it open.
```c
// Increment link count (for creating hard links)
int inode_link(FileSystem* fs, uint64_t inode_num) {
    Inode inode;
    if (read_inode(fs, inode_num, &inode) < 0) {
        return -1;
    }
    inode.link_count++;
    inode.ctime = (uint64_t)time(NULL);
    return write_inode(fs, inode_num, &inode);
}
// Decrement link count (for unlinking)
// Returns 1 if the file should be deleted (link_count == 0)
int inode_unlink(FileSystem* fs, uint64_t inode_num) {
    Inode inode;
    if (read_inode(fs, inode_num, &inode) < 0) {
        return -1;
    }
    if (inode.link_count > 0) {
        inode.link_count--;
    }
    inode.ctime = (uint64_t)time(NULL);
    write_inode(fs, inode_num, &inode);
    return (inode.link_count == 0) ? 1 : 0;
}
```
## Testing Your Implementation
Here's a test harness to verify your inode operations:
```c
#include <assert.h>
#include <stdio.h>
void test_inode_operations(FileSystem* fs) {
    printf("Testing inode operations...\n");
    // Test 1: Allocate an inode
    uint64_t inode_num = alloc_inode(fs);
    assert(inode_num > 0);
    printf("  ✓ Allocated inode %lu\n", inode_num);
    // Test 2: Initialize and write inode
    Inode inode;
    memset(&inode, 0, sizeof(Inode));
    inode.mode = S_IFREG | 0644;  // Regular file, rw-r--r--
    inode.uid = 1000;
    inode.gid = 1000;
    inode.link_count = 1;
    inode.size = 0;
    update_timestamps(&inode, TS_ALL);
    assert(write_inode(fs, inode_num, &inode) == 0);
    printf("  ✓ Wrote inode metadata\n");
    // Test 3: Read back and verify
    Inode read_back;
    assert(read_inode(fs, inode_num, &read_back) == 0);
    assert(read_back.mode == (S_IFREG | 0644));
    assert(read_back.uid == 1000);
    assert(read_back.link_count == 1);
    printf("  ✓ Read back and verified inode\n");
    // Test 4: Write to direct block
    const char* test_data = "Hello, filesystem!";
    size_t data_len = strlen(test_data);
    int written = file_write(fs, inode_num, 0, test_data, data_len);
    assert(written == (int)data_len);
    printf("  ✓ Wrote %d bytes to file\n", written);
    // Test 5: Read back data
    char buffer[256];
    int bytes_read = file_read(fs, inode_num, 0, buffer, sizeof(buffer));
    assert(bytes_read == (int)data_len);
    assert(memcmp(buffer, test_data, data_len) == 0);
    printf("  ✓ Read back data matches: '%.*s'\n", bytes_read, buffer);
    // Test 6: Verify block was allocated
    assert(read_inode(fs, inode_num, &inode) == 0);
    assert(inode.direct[0] != 0);
    assert(inode.size == data_len);
    printf("  ✓ Block allocated: %lu, file size: %lu\n", 
           inode.direct[0], inode.size);
    // Test 7: Truncate to zero
    assert(file_truncate(fs, inode_num, 0) == 0);
    assert(read_inode(fs, inode_num, &inode) == 0);
    assert(inode.size == 0);
    assert(inode.direct[0] == 0);  // Block should be freed
    printf("  ✓ Truncated file to zero\n");
    // Test 8: Large file (indirect blocks)
    char large_data[BLOCK_SIZE * 15];  // 15 blocks, needs indirect
    memset(large_data, 'X', sizeof(large_data));
    written = file_write(fs, inode_num, 0, large_data, sizeof(large_data));
    assert(written == sizeof(large_data));
    assert(read_inode(fs, inode_num, &inode) == 0);
    assert(inode.indirect != 0);  // Should have allocated indirect block
    printf("  ✓ Wrote large file (15 blocks), indirect block: %lu\n", 
           inode.indirect);
    // Test 9: Deallocate inode
    uint64_t old_block = inode.direct[0];
    assert(inode_deallocate(fs, inode_num) == 0);
    // Verify block was freed (should be allocatable again)
    uint64_t new_block = alloc_block(fs);
    assert(new_block == old_block);  // Should get the same block back
    free_block(fs, new_block);
    printf("  ✓ Deallocated inode, blocks freed\n");
    printf("All inode tests passed!\n");
}
void test_sparse_files(FileSystem* fs) {
    printf("\nTesting sparse files...\n");
    // Create a file
    uint64_t inode_num = alloc_inode(fs);
    Inode inode;
    memset(&inode, 0, sizeof(Inode));
    inode.mode = S_IFREG | 0644;
    inode.uid = 1000;
    inode.gid = 1000;
    inode.link_count = 1;
    update_timestamps(&inode, TS_ALL);
    write_inode(fs, inode_num, &inode);
    // Write at offset 1MB (should create a hole)
    const char* data = "Data at 1MB offset";
    size_t data_len = strlen(data);
    uint64_t offset = 1024 * 1024;  // 1MB
    int written = file_write(fs, inode_num, offset, data, data_len);
    assert(written == (int)data_len);
    // Read from the hole (should be zeros)
    char buffer[BLOCK_SIZE];
    int bytes_read = file_read(fs, inode_num, 0, buffer, BLOCK_SIZE);
    assert(bytes_read == BLOCK_SIZE);
    int all_zeros = 1;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        if (buffer[i] != 0) {
            all_zeros = 0;
            break;
        }
    }
    assert(all_zeros);
    printf("  ✓ Hole reads as zeros\n");
    // Read the actual data
    bytes_read = file_read(fs, inode_num, offset, buffer, data_len);
    assert(bytes_read == (int)data_len);
    assert(memcmp(buffer, data, data_len) == 0);
    printf("  ✓ Data at 1MB offset reads correctly\n");
    // Check file size
    read_inode(fs, inode_num, &inode);
    assert(inode.size == offset + data_len);
    printf("  ✓ File size is %lu (logical size, not allocated size)\n", inode.size);
    // Clean up
    inode_deallocate(fs, inode_num);
    printf("Sparse file tests passed!\n");
}
```
## Common Pitfalls
**Leaking indirect blocks**: When freeing an inode, it's easy to free the direct blocks and forget the indirect ones. Always traverse the entire pointer tree.
**Off-by-one in block calculations**: The transition from direct to indirect happens at block 12, not block 11 or 13. Verify your boundary conditions with the test harness.
**Forgetting to update timestamps**: Every modification must update mtime and ctime. The ctime update on mtime change is particularly easy to miss.
**Double-freeing blocks**: If a crash occurs between freeing a block and clearing its pointer, the bitmap may show the block as free while the inode still references it. On recovery, freeing the inode again would double-free. Journaling (Milestone 6) solves this.
**Not handling partial block writes**: If you write 100 bytes at offset 4050, you need to read the existing block, modify bytes 4050-4095, then write the block back. Writing a full block would corrupt bytes 0-4049.
## What You've Built
At the end of this milestone, you have:
1. **A complete inode structure** with metadata fields and block pointers
2. **Direct block access** for the first 48KB of any file
3. **Single-indirect traversal** for files up to ~2.1MB
4. **Double-indirect traversal** for files up to ~1GB
5. **Block allocation on write** that lazily allocates indirect blocks
6. **Sparse file support** where null pointers read as zeros
7. **Truncation** that properly frees blocks
8. **Complete inode deallocation** that traverses the full pointer tree
You can now:
```bash
$ ./mkfs disk.img 100      # Create filesystem (Milestone 1)
$ ./test_inode disk.img    # Test inode operations
# Tests should pass, showing allocation, read/write, truncation, and deallocation
```
## The Knowledge Cascade
The patterns you've learned extend far beyond filesystems:
**Database B-Trees**: Your indirect block traversal is essentially a 2-level B-tree where the inode is the root, indirect blocks are internal nodes, and data blocks are leaves. Database page management uses the same fan-out calculations — a 16KB page with 8-byte keys can hold ~2000 pointers, creating trees that stay shallow even for billions of records. The next time you see `EXPLAIN` output showing "index depth: 3", you'll know exactly what those levels represent.
**Virtual Memory Page Tables**: Modern CPUs use 4-level page tables: Page Global Directory → Page Upper Directory → Page Middle Directory → Page Table → Page Frame. This is isomorphic to triple-indirect → double-indirect → single-indirect → data block. Both designs solve the same problem: translating a logical address (file offset or virtual address) through a hierarchy of fixed-size pointer arrays. The TLB is the hardware equivalent of caching indirect blocks — both avoid repeated traversals.
**Filesystem Fragmentation**: As files grow and shrink, the blocks they allocate may become scattered across the disk. Consecutive logical blocks (blocks 0, 1, 2, 3) might map to physical blocks 1000, 50000, 2000, 80000. This is why `e4defrag` exists for ext4 — it relocates blocks to restore contiguity, reducing seek overhead on HDDs and improving readahead on SSDs. Understanding indirect blocks explains *why* fragmentation happens and *why* defragmentation is non-trivial (you must update all the pointers).
**Cloud Storage Object Addressing**: Object stores like S3 use a similar indirection pattern internally. Your key ("bucket/object") maps to metadata (the inode equivalent), which maps to actual data locations (blocks spread across storage nodes). The key difference: object stores use consistent hashing instead of fixed-size pointer arrays, allowing infinite scaling at the cost of more complex lookup.
**Sparse Matrix Storage**: Compressed Sparse Row (CSR) format stores matrices by keeping only non-zero values and their column indices. This is exactly what sparse files do — zeros are implicit (holes), non-zeros are explicit (allocated blocks). A 1TB sparse file with 1MB of actual data uses 1MB + overhead, just as a 1M×1M sparse matrix with 1000 non-zeros uses 1000 values + indices.
In the next milestone, you'll build directory structures on top of these inodes, enabling path resolution ("find `/home/user/file.txt`") and the full directory API (`mkdir`, `rmdir`, `readdir`).
---
<!-- END_MS -->


<!-- MS_ID: filesystem-m3 -->
# Directory Operations
You've built the machinery to store and retrieve file data through inodes with direct and indirect block pointers. But there's a glaring gap: **how do you find a file in the first place?**
Users think in terms of paths — `/home/user/documents/letter.txt` — not inode numbers. Your filesystem needs to translate these human-readable paths into the inode numbers your machinery understands. This translation is **directory operations**, and it's where the abstract concept of "hierarchy" becomes concrete.
## The Fundamental Tension: Names vs. Numbers
Here's the problem: **humans need names, but the filesystem only understands numbers.**
When you open a file, you specify a path string. The filesystem needs to resolve that string to an inode number — the actual identifier that lets it find metadata and data blocks. But here's the constraint: paths are hierarchical (nested), variable-length, and human-chosen. Inode numbers are flat, fixed-size, and system-assigned.
You can't store a path-to-inode mapping in a single lookup table because:
1. **Paths are unbounded**: A filesystem can have millions of files with arbitrarily deep paths
2. **Paths change**: Move a directory, and all paths beneath it change
3. **The mapping must be incremental**: You shouldn't need to read the entire filesystem to find one file
The solution is **incremental resolution**: break the path into components, and resolve each component through a chain of lookups. Each directory in the chain maps one component name to the next inode number. The "hierarchy" isn't a single structure — it's an **emergent property** of many small name→inode mappings chained together.
```
Path: /home/user/docs/letter.txt
Resolution chain:
  Root directory (inode 1) → lookup "home" → inode 100
  Directory inode 100      → lookup "user" → inode 200
  Directory inode 200      → lookup "docs" → inode 300
  Directory inode 300      → lookup "letter.txt" → inode 400
Result: /home/user/docs/letter.txt = inode 400
```
Each arrow represents reading a directory's data blocks and scanning for a name. A 5-component path requires 5 directory lookups. This is why deep directory structures are slightly slower — more lookups — but each lookup is independent, so you only read what you need.
## The Revelation: Directories ARE Files
Here's where most developers' mental model breaks:
**A directory is not a container. A directory is a file.**
When you create a directory, you allocate an inode — exactly like creating a regular file. That inode has block pointers — exactly like a regular file. Those blocks contain data — exactly like a regular file.
The only difference is **what the data means**:
- For a regular file: data = user content (text, binary, whatever)
- For a directory: data = an array of directory entries (name → inode mappings)
```c
// A directory entry is just a record in the directory's data blocks
typedef struct {
    uint64_t inode;      // Inode number of the target
    uint16_t rec_len;    // Length of this record (for skipping to next)
    uint8_t  name_len;   // Length of the name
    uint8_t  file_type;  // Type of file (regular, directory, etc.)
    char     name[256];  // Null-terminated filename
} __attribute__((packed)) DirEntry;
```
The "hierarchy" you see in `ls -R` or file explorers is an illusion created by:
1. Each directory containing entries that point to other directories
2. Special entries `.` (self) and `..` (parent) creating bidirectional links
3. Path resolution code that follows the chain
This isn't just an implementation detail — it's the **fundamental design decision** of Unix filesystems. Everything is a file; directories are just files with a specific format. The kernel doesn't have separate "directory storage" — it uses the same inode/block machinery for everything.
**Why This Matters**: When you implement `mkdir`, you're not creating a new kind of object. You're creating a regular inode, marking it as a directory in the mode field, and writing two DirEntry records (`.` and `..`) to its first data block. That's it. The "directory-ness" is entirely in how the data is interpreted.

![Filesystem Component Atlas](./diagrams/diag-l0-filesystem-map.svg)

## Directory Entry Structure: The Name→Inode Map
Let's design the on-disk format for directory entries. Each entry needs to store:
- **inode number**: The target of this mapping
- **name**: The human-readable key
- **metadata for traversal**: Entry length (for scanning), name length, file type
```c
// Directory entry types (stored in file_type field)
#define DT_UNKNOWN   0  // Unknown type
#define DT_REG       1  // Regular file
#define DT_DIR       2  // Directory
#define DT_LNK       3  // Symbolic link
#define DT_BLK       4  // Block device
#define DT_CHR       5  // Character device
#define DT_FIFO      6  // Named pipe
#define DT_SOCK      7  // Socket
// On-disk directory entry structure
// Total size: 280 bytes (with padding for alignment)
typedef struct {
    uint64_t inode;          // 8 bytes: inode number (0 = unused entry)
    uint16_t rec_len;        // 2 bytes: total record length including padding
    uint8_t  name_len;       // 1 byte: length of name (not including null)
    uint8_t  file_type;      // 1 byte: DT_* constant
    char     name[256];      // 256 bytes: name (null-padded)
    uint8_t  _padding[12];   // 12 bytes: pad to 280 bytes total
} __attribute__((packed)) DirEntry;
#define DIR_ENTRY_SIZE 280
#define ENTRIES_PER_BLOCK (BLOCK_SIZE / DIR_ENTRY_SIZE)  // 14 entries per 4KB block
#define MAX_NAME_LEN 255
```
**Hardware Soul Check**: This structure is 280 bytes, which means 14 entries fit in a 4KB block. The `rec_len` field exists because real filesystems (ext4) use variable-length entries — deleting an entry doesn't remove it, just merges it with the next entry by increasing `rec_len`. Our fixed-size entries are simpler but waste space for short names. A 3-character filename like "foo" still takes 280 bytes. Production filesystems pack entries tightly: `rec_len = 8 + name_len + padding_to_4`.
**Alternative Design: Variable-Length Entries**
```c
// How ext4 does it (simplified)
typedef struct {
    uint32_t inode;        // 4 bytes
    uint16_t rec_len;      // 2 bytes: distance to next entry
    uint8_t  name_len;     // 1 byte
    uint8_t  file_type;    // 1 byte
    char     name[];       // Variable length, null-padded to 4-byte boundary
} __attribute__((packed)) Ext4DirEntry;
// A 3-char name takes ~12 bytes, not 280
```
For this project, we'll use fixed-size entries for simplicity. The concepts translate directly to variable-length designs — you just have more complex pointer arithmetic.
## Reading Directory Entries: Scanning for Names
To look up a name in a directory, you read its data blocks and scan each entry:
```c
#include <string.h>
#include <errno.h>
// Find a directory entry by name
// Returns the entry's inode number, or 0 if not found
uint64_t dir_lookup(FileSystem* fs, uint64_t dir_inode_num, const char* name) {
    // Validate: must be a directory
    Inode dir_inode;
    if (read_inode(fs, dir_inode_num, &dir_inode) < 0) {
        return 0;
    }
    if (!(dir_inode.mode & S_IFDIR)) {
        errno = ENOTDIR;
        return 0;
    }
    size_t name_len = strlen(name);
    if (name_len > MAX_NAME_LEN) {
        errno = ENAMETOOLONG;
        return 0;
    }
    // Calculate how many blocks the directory spans
    uint64_t dir_size = dir_inode.size;
    uint64_t num_blocks = (dir_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    char block_buffer[BLOCK_SIZE];
    DirEntry* entries = (DirEntry*)block_buffer;
    // Scan each block
    for (uint64_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        uint64_t phys_block = get_file_block(fs, &dir_inode, block_idx);
        if (phys_block == 0) {
            continue;  // Hole in directory (shouldn't happen)
        }
        if (read_block(fs->dev, phys_block, block_buffer) < 0) {
            return 0;
        }
        // Scan entries in this block
        for (int i = 0; i < ENTRIES_PER_BLOCK; i++) {
            // Skip unused entries (inode == 0)
            if (entries[i].inode == 0) {
                continue;
            }
            // Compare names
            if (entries[i].name_len == name_len &&
                memcmp(entries[i].name, name, name_len) == 0) {
                return entries[i].inode;  // Found!
            }
        }
    }
    errno = ENOENT;
    return 0;  // Not found
}
```
**Hardware Soul Check**: A directory lookup is O(n) in the number of entries — you scan linearly until you find a match. For a directory with 10,000 files, that's reading ~715 blocks (14 entries per block) in the worst case. Real filesystems use hash trees (ext4's `dx_hash`) for large directories, making lookup O(1) on average. The `htree` index stores a hash table in the directory's blocks, trading complexity for speed on large directories.
## Adding Directory Entries: Append and Link
When you add an entry, you append it to the directory's data blocks:
```c
// Add a new entry to a directory
// Returns 0 on success, -1 on error
int dir_add_entry(FileSystem* fs, uint64_t dir_inode_num, 
                   const char* name, uint64_t target_inode, uint8_t file_type) {
    // Validate directory
    Inode dir_inode;
    if (read_inode(fs, dir_inode_num, &dir_inode) < 0) {
        return -1;
    }
    if (!(dir_inode.mode & S_IFDIR)) {
        errno = ENOTDIR;
        return -1;
    }
    // Check for duplicate
    if (dir_lookup(fs, dir_inode_num, name) != 0) {
        errno = EEXIST;
        return -1;  // Name already exists
    }
    size_t name_len = strlen(name);
    if (name_len > MAX_NAME_LEN) {
        errno = ENAMETOOLONG;
        return -1;
    }
    // Find a free slot or append
    char block_buffer[BLOCK_SIZE];
    DirEntry* entries = (DirEntry*)block_buffer;
    uint64_t dir_size = dir_inode.size;
    uint64_t num_blocks = (dir_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks == 0) num_blocks = 1;  // At least one block
    // Scan for free slot (inode == 0)
    for (uint64_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        uint64_t phys_block = get_file_block(fs, &dir_inode, block_idx);
        if (phys_block == 0) {
            // Need to allocate this block
            phys_block = get_or_alloc_block(fs, dir_inode_num, &dir_inode, block_idx);
            if (phys_block == 0) {
                return -1;  // No space
            }
            memset(block_buffer, 0, BLOCK_SIZE);
        } else {
            if (read_block(fs->dev, phys_block, block_buffer) < 0) {
                return -1;
            }
        }
        // Look for free entry in this block
        for (int i = 0; i < ENTRIES_PER_BLOCK; i++) {
            if (entries[i].inode == 0) {
                // Found a free slot!
                entries[i].inode = target_inode;
                entries[i].rec_len = DIR_ENTRY_SIZE;
                entries[i].name_len = (uint8_t)name_len;
                entries[i].file_type = file_type;
                memset(entries[i].name, 0, 256);
                memcpy(entries[i].name, name, name_len);
                // Write back
                if (write_block(fs->dev, phys_block, block_buffer) < 0) {
                    return -1;
                }
                // Update directory size if needed
                uint64_t entry_offset = block_idx * BLOCK_SIZE + i * DIR_ENTRY_SIZE;
                if (entry_offset + DIR_ENTRY_SIZE > dir_inode.size) {
                    dir_inode.size = entry_offset + DIR_ENTRY_SIZE;
                }
                dir_inode.mtime = (uint64_t)time(NULL);
                dir_inode.ctime = dir_inode.mtime;
                write_inode(fs, dir_inode_num, &dir_inode);
                return 0;
            }
        }
    }
    // No free slot found - need to extend directory
    uint64_t new_block_idx = num_blocks;
    uint64_t new_phys_block = get_or_alloc_block(fs, dir_inode_num, &dir_inode, new_block_idx);
    if (new_phys_block == 0) {
        return -1;  // No space
    }
    // Write new entry at start of new block
    memset(block_buffer, 0, BLOCK_SIZE);
    entries[0].inode = target_inode;
    entries[0].rec_len = DIR_ENTRY_SIZE;
    entries[0].name_len = (uint8_t)name_len;
    entries[0].file_type = file_type;
    memcpy(entries[0].name, name, name_len);
    if (write_block(fs->dev, new_phys_block, block_buffer) < 0) {
        return -1;
    }
    // Update directory size
    dir_inode.size = (new_block_idx + 1) * BLOCK_SIZE;
    dir_inode.mtime = (uint64_t)time(NULL);
    dir_inode.ctime = dir_inode.mtime;
    write_inode(fs, dir_inode_num, &dir_inode);
    return 0;
}
```
## Removing Directory Entries: Unlink
Removing an entry clears its slot and decrements the target's link count:
```c
// Remove an entry from a directory
// Returns 0 on success, -1 on error
int dir_remove_entry(FileSystem* fs, uint64_t dir_inode_num, const char* name) {
    // Validate directory
    Inode dir_inode;
    if (read_inode(fs, dir_inode_num, &dir_inode) < 0) {
        return -1;
    }
    if (!(dir_inode.mode & S_IFDIR)) {
        errno = ENOTDIR;
        return -1;
    }
    size_t name_len = strlen(name);
    // Find the entry
    char block_buffer[BLOCK_SIZE];
    DirEntry* entries = (DirEntry*)block_buffer;
    uint64_t dir_size = dir_inode.size;
    uint64_t num_blocks = (dir_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (uint64_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        uint64_t phys_block = get_file_block(fs, &dir_inode, block_idx);
        if (phys_block == 0) continue;
        if (read_block(fs->dev, phys_block, block_buffer) < 0) {
            return -1;
        }
        for (int i = 0; i < ENTRIES_PER_BLOCK; i++) {
            if (entries[i].inode != 0 &&
                entries[i].name_len == name_len &&
                memcmp(entries[i].name, name, name_len) == 0) {
                // Found it! Get target inode before clearing
                uint64_t target_inode_num = entries[i].inode;
                // Clear the entry
                entries[i].inode = 0;
                entries[i].rec_len = 0;
                entries[i].name_len = 0;
                entries[i].file_type = 0;
                memset(entries[i].name, 0, 256);
                if (write_block(fs->dev, phys_block, block_buffer) < 0) {
                    return -1;
                }
                // Update directory timestamps
                dir_inode.mtime = (uint64_t)time(NULL);
                dir_inode.ctime = dir_inode.mtime;
                write_inode(fs, dir_inode_num, &dir_inode);
                // Decrement target's link count
                int should_delete = inode_unlink(fs, target_inode_num);
                if (should_delete) {
                    // Link count reached 0 - free the inode
                    // In a real system, we'd check for open file descriptors first
                    inode_deallocate(fs, target_inode_num);
                }
                return 0;
            }
        }
    }
    errno = ENOENT;
    return -1;  // Entry not found
}
```
**The Link Count Protocol**: When you add an entry, you're creating a **reference** to an inode. The inode's `link_count` tracks how many references exist. When you remove an entry, you decrement `link_count`. Only when it reaches zero (and no process has the file open) can the inode and its blocks be freed.
This is why `rm` doesn't necessarily delete a file's data — if another directory entry (hard link) points to the same inode, the data stays. The inode is only freed when the *last* reference is removed.
## Path Resolution: The Chain of Lookups
Now we can implement full path resolution:
```c
#include <ctype.h>
// Split a path into components
// Returns number of components, or -1 on error
int split_path(const char* path, char components[][MAX_NAME_LEN + 1], int max_components) {
    if (path == NULL || path[0] == '\0') {
        return -1;
    }
    int count = 0;
    const char* start = path;
    // Skip leading slashes
    while (*start == '/') start++;
    while (*start != '\0' && count < max_components) {
        // Find end of component
        const char* end = start;
        while (*end != '\0' && *end != '/') end++;
        int len = end - start;
        if (len > MAX_NAME_LEN) {
            return -1;  // Component too long
        }
        if (len > 0) {
            memcpy(components[count], start, len);
            components[count][len] = '\0';
            count++;
        }
        // Skip slashes
        while (*end == '/') end++;
        start = end;
    }
    return count;
}
// Resolve a path to an inode number
// Returns inode number, or 0 on error
uint64_t path_resolve(FileSystem* fs, const char* path) {
    if (path == NULL || path[0] == '\0') {
        errno = EINVAL;
        return 0;
    }
    // Start from root or current directory
    uint64_t current_inode;
    const char* p = path;
    if (path[0] == '/') {
        current_inode = 1;  // Root inode is always 1
        p++;  // Skip leading slash
    } else {
        // Relative path - would use current working directory
        // For simplicity, we'll use root
        current_inode = 1;
    }
    // Handle empty path or just "/"
    if (*p == '\0') {
        return current_inode;
    }
    // Split into components
    char components[64][MAX_NAME_LEN + 1];
    int num_components = split_path(path, components, 64);
    if (num_components < 0) {
        errno = EINVAL;
        return 0;
    }
    // Resolve each component
    for (int i = 0; i < num_components; i++) {
        char* component = components[i];
        // Handle special entries
        if (strcmp(component, ".") == 0) {
            // Stay in current directory
            continue;
        }
        if (strcmp(component, "..") == 0) {
            // Go to parent directory
            // Look up ".." in current directory
            uint64_t parent = dir_lookup(fs, current_inode, "..");
            if (parent == 0) {
                // At root, ".." points to itself
                continue;
            }
            current_inode = parent;
            continue;
        }
        // Look up component in current directory
        uint64_t next_inode = dir_lookup(fs, current_inode, component);
        if (next_inode == 0) {
            errno = ENOENT;
            return 0;  // Component not found
        }
        // Check if intermediate component is a directory
        if (i < num_components - 1) {
            Inode next;
            if (read_inode(fs, next_inode, &next) < 0) {
                return 0;
            }
            if (!(next.mode & S_IFDIR)) {
                errno = ENOTDIR;
                return 0;  // Intermediate component not a directory
            }
        }
        current_inode = next_inode;
    }
    return current_inode;
}
// Resolve parent directory and extract final component
// Useful for operations like create, mkdir, etc.
uint64_t path_resolve_parent(FileSystem* fs, const char* path, char* last_component) {
    if (path == NULL || path[0] == '\0') {
        errno = EINVAL;
        return 0;
    }
    // Find last component
    const char* last_slash = strrchr(path, '/');
    const char* name;
    const char* parent_path;
    if (last_slash == NULL) {
        // No slash - relative path, parent is "."
        name = path;
        parent_path = ".";
    } else if (last_slash == path) {
        // Slash at beginning - parent is root
        name = last_slash + 1;
        parent_path = "/";
    } else {
        name = last_slash + 1;
        // Need to extract parent path
        static char parent_buf[1024];
        int parent_len = last_slash - path;
        if (parent_len >= (int)sizeof(parent_buf)) {
            errno = ENAMETOOLONG;
            return 0;
        }
        memcpy(parent_buf, path, parent_len);
        parent_buf[parent_len] = '\0';
        parent_path = parent_buf;
    }
    // Check name length
    if (strlen(name) > MAX_NAME_LEN) {
        errno = ENAMETOOLONG;
        return 0;
    }
    // Copy name to output
    strcpy(last_component, name);
    // Resolve parent
    return path_resolve(fs, parent_path);
}
```
**Hardware Soul Check**: Each component in a path requires at least one directory lookup, which means reading at least one block. For a 5-component path, you're doing 5 separate disk reads minimum. This is why modern OSes cache directory entries aggressively — the Linux **dentry cache** (directory entry cache) stores recently-resolved paths in memory, turning subsequent lookups into simple hash table lookups. A cache hit takes nanoseconds; a cache miss takes milliseconds.
## The Special Entries: `.` and `..`
The `.` and `..` entries aren't magic — they're just directory entries like any other:
```c
// Initialize the special entries in a new directory
int dir_init_special_entries(FileSystem* fs, uint64_t dir_inode_num, 
                              uint64_t parent_inode_num) {
    // Add "." entry (points to self)
    if (dir_add_entry(fs, dir_inode_num, ".", dir_inode_num, DT_DIR) < 0) {
        return -1;
    }
    // Add ".." entry (points to parent)
    if (dir_add_entry(fs, dir_inode_num, "..", parent_inode_num, DT_DIR) < 0) {
        // Rollback "." entry
        dir_remove_entry(fs, dir_inode_num, ".");
        return -1;
    }
    return 0;
}
```
**The Root Directory Special Case**: For the root directory, `..` points to itself. This creates a cycle that terminates upward traversal:
```c
// In mkfs, when creating root directory:
dir_add_entry(fs, 1, ".", 1, DT_DIR);   // Root's . points to root
dir_add_entry(fs, 1, "..", 1, DT_DIR);  // Root's .. also points to root
```
This is why `cd /../../../../..` from anywhere eventually stops at root — the `..` lookup returns the same inode.
## mkdir: Creating a New Directory
Creating a directory involves:
1. Allocating a new inode
2. Initializing it as a directory
3. Adding `.` and `..` entries
4. Adding an entry in the parent directory
5. Incrementing the parent's link count (for `..`)
```c
// Create a new directory
// Returns inode number on success, 0 on error
uint64_t fs_mkdir(FileSystem* fs, const char* path) {
    // Resolve parent and get final component
    char name[MAX_NAME_LEN + 1];
    uint64_t parent_inode_num = path_resolve_parent(fs, path, name);
    if (parent_inode_num == 0) {
        return 0;  // Parent doesn't exist
    }
    // Check if name already exists
    if (dir_lookup(fs, parent_inode_num, name) != 0) {
        errno = EEXIST;
        return 0;
    }
    // Verify parent is a directory
    Inode parent_inode;
    if (read_inode(fs, parent_inode_num, &parent_inode) < 0) {
        return 0;
    }
    if (!(parent_inode.mode & S_IFDIR)) {
        errno = ENOTDIR;
        return 0;
    }
    // Allocate new inode
    uint64_t new_inode_num = alloc_inode(fs);
    if (new_inode_num == 0) {
        errno = ENOSPC;
        return 0;  // No free inodes
    }
    // Initialize inode as directory
    Inode new_inode;
    memset(&new_inode, 0, sizeof(Inode));
    new_inode.mode = S_IFDIR | 0755;  // Directory with rwxr-xr-x
    new_inode.uid = 0;   // Would be process's uid
    new_inode.gid = 0;   // Would be process's gid
    new_inode.link_count = 2;  // One from parent's entry, one from "."
    new_inode.size = 0;
    new_inode.blocks = 0;
    new_inode.atime = new_inode.mtime = new_inode.ctime = (uint64_t)time(NULL);
    if (write_inode(fs, new_inode_num, &new_inode) < 0) {
        // Rollback: free the inode
        // (would need to implement inode_bitmap_free)
        return 0;
    }
    // Add . and .. entries
    if (dir_init_special_entries(fs, new_inode_num, parent_inode_num) < 0) {
        return 0;
    }
    // Add entry in parent directory
    if (dir_add_entry(fs, parent_inode_num, name, new_inode_num, DT_DIR) < 0) {
        // Rollback would go here
        return 0;
    }
    // Increment parent's link count (because child's ".." points to it)
    parent_inode.link_count++;
    parent_inode.mtime = (uint64_t)time(NULL);
    parent_inode.ctime = parent_inode.mtime;
    write_inode(fs, parent_inode_num, &parent_inode);
    return new_inode_num;
}
```
**Why link_count = 2 for a new directory?** A directory's link count has a specific meaning:
- Each directory entry in a parent that points to this directory: +1
- The directory's own `.` entry: +1
So a new directory has:
1. The entry in its parent (the name you gave it)
2. Its own `.` entry
Total: 2.
When you add a subdirectory, the parent's link count increases by 1 (for the child's `..` entry). This is why `ls -ld` shows `link_count` as the number of subdirectories + 2 (for `.` and the parent's entry).
## rmdir: Removing a Directory
Removing a directory is only allowed if it's empty (contains only `.` and `..`):
```c
// Check if a directory is empty (contains only . and ..)
int dir_is_empty(FileSystem* fs, uint64_t dir_inode_num) {
    Inode dir_inode;
    if (read_inode(fs, dir_inode_num, &dir_inode) < 0) {
        return -1;
    }
    char block_buffer[BLOCK_SIZE];
    DirEntry* entries = (DirEntry*)block_buffer;
    uint64_t num_blocks = (dir_inode.size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (uint64_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        uint64_t phys_block = get_file_block(fs, &dir_inode, block_idx);
        if (phys_block == 0) continue;
        if (read_block(fs->dev, phys_block, block_buffer) < 0) {
            return -1;
        }
        for (int i = 0; i < ENTRIES_PER_BLOCK; i++) {
            if (entries[i].inode != 0) {
                // Found an entry - check if it's . or ..
                if (strcmp(entries[i].name, ".") != 0 && 
                    strcmp(entries[i].name, "..") != 0) {
                    return 0;  // Directory is not empty
                }
            }
        }
    }
    return 1;  // Directory is empty
}
// Remove a directory
int fs_rmdir(FileSystem* fs, const char* path) {
    // Resolve the directory to remove
    uint64_t dir_inode_num = path_resolve(fs, path);
    if (dir_inode_num == 0) {
        return -1;  // Doesn't exist
    }
    // Verify it's a directory
    Inode dir_inode;
    if (read_inode(fs, dir_inode_num, &dir_inode) < 0) {
        return -1;
    }
    if (!(dir_inode.mode & S_IFDIR)) {
        errno = ENOTDIR;
        return -1;
    }
    // Check if it's empty
    int empty = dir_is_empty(fs, dir_inode_num);
    if (empty < 0) {
        return -1;
    }
    if (empty == 0) {
        errno = ENOTEMPTY;
        return -1;
    }
    // Can't remove root
    if (dir_inode_num == 1) {
        errno = EBUSY;
        return -1;
    }
    // Get parent
    char name[MAX_NAME_LEN + 1];
    uint64_t parent_inode_num = path_resolve_parent(fs, path, name);
    if (parent_inode_num == 0) {
        return -1;
    }
    // Remove entry from parent
    if (dir_remove_entry(fs, parent_inode_num, name) < 0) {
        return -1;
    }
    // Note: dir_remove_entry already decremented link count
    // For a directory, this means the link count went from 2 to 1
    // We need to decrement again for the "." entry
    Inode check_inode;
    read_inode(fs, dir_inode_num, &check_inode);
    if (check_inode.link_count > 0) {
        check_inode.link_count--;
    }
    // If link count is 0, free the inode
    if (check_inode.link_count == 0) {
        inode_deallocate(fs, dir_inode_num);
    } else {
        write_inode(fs, dir_inode_num, &check_inode);
    }
    // Decrement parent's link count (removing child's ".." reference)
    Inode parent_inode;
    if (read_inode(fs, parent_inode_num, &parent_inode) < 0) {
        return -1;
    }
    if (parent_inode.link_count > 0) {
        parent_inode.link_count--;
    }
    parent_inode.mtime = (uint64_t)time(NULL);
    parent_inode.ctime = parent_inode.mtime;
    write_inode(fs, parent_inode_num, &parent_inode);
    return 0;
}
```
## Hard Links: Multiple Names, One Inode
A hard link is just another directory entry pointing to an existing inode:
```c
// Create a hard link
int fs_link(FileSystem* fs, const char* existing_path, const char* new_path) {
    // Resolve existing file
    uint64_t existing_inode_num = path_resolve(fs, existing_path);
    if (existing_inode_num == 0) {
        return -1;  // Source doesn't exist
    }
    // Get existing inode
    Inode existing_inode;
    if (read_inode(fs, existing_inode_num, &existing_inode) < 0) {
        return -1;
    }
    // Can't hard link to directories (would create cycles)
    if (existing_inode.mode & S_IFDIR) {
        errno = EPERM;
        return -1;
    }
    // Resolve parent of new path
    char name[MAX_NAME_LEN + 1];
    uint64_t parent_inode_num = path_resolve_parent(fs, new_path, name);
    if (parent_inode_num == 0) {
        return -1;
    }
    // Check if name already exists
    if (dir_lookup(fs, parent_inode_num, name) != 0) {
        errno = EEXIST;
        return -1;
    }
    // Determine file type
    uint8_t file_type = DT_REG;
    if (existing_inode.mode & S_IFLNK) file_type = DT_LNK;
    // Add entry in parent directory
    if (dir_add_entry(fs, parent_inode_num, name, existing_inode_num, file_type) < 0) {
        return -1;
    }
    // Increment link count
    existing_inode.link_count++;
    existing_inode.ctime = (uint64_t)time(NULL);
    write_inode(fs, existing_inode_num, &existing_inode);
    return 0;
}
```
**Why No Hard Links to Directories?** Allowing hard links to directories would create cycles that break the tree structure. If `/a/b` and `/a/c` both pointed to the same directory inode, you could traverse `/a/b/../c/..` infinitely without ever reaching root. The filesystem tree would become a graph, making operations like `find` and `du` potentially infinite. Symbolic links (which you'll implement later) can point to directories because they're resolved at access time, not stored in the directory structure.
## Symbolic Links: A Peek Ahead
A symbolic link (symlink) is fundamentally different from a hard link:
```c
// Create a symbolic link
int fs_symlink(FileSystem* fs, const char* target_path, const char* link_path) {
    // Resolve parent of link path
    char name[MAX_NAME_LEN + 1];
    uint64_t parent_inode_num = path_resolve_parent(fs, link_path, name);
    if (parent_inode_num == 0) {
        return -1;
    }
    // Check if name already exists
    if (dir_lookup(fs, parent_inode_num, name) != 0) {
        errno = EEXIST;
        return -1;
    }
    // Allocate new inode
    uint64_t new_inode_num = alloc_inode(fs);
    if (new_inode_num == 0) {
        errno = ENOSPC;
        return -1;
    }
    // Initialize inode as symlink
    Inode new_inode;
    memset(&new_inode, 0, sizeof(Inode));
    new_inode.mode = S_IFLNK | 0777;  // Symlinks typically have 777 permissions
    new_inode.uid = 0;
    new_inode.gid = 0;
    new_inode.link_count = 1;
    new_inode.size = strlen(target_path);
    new_inode.atime = new_inode.mtime = new_inode.ctime = (uint64_t)time(NULL);
    // Store target path in inode's blocks (or direct pointers for short paths)
    size_t target_len = strlen(target_path);
    if (target_len < 48) {
        // Short symlink: store in direct pointer space
        memcpy(new_inode.direct, target_path, target_len);
    } else {
        // Long symlink: allocate block and write
        uint64_t block = get_or_alloc_block(fs, new_inode_num, &new_inode, 0);
        if (block == 0) {
            return -1;
        }
        char block_buf[BLOCK_SIZE];
        memset(block_buf, 0, BLOCK_SIZE);
        memcpy(block_buf, target_path, target_len);
        write_block(fs->dev, block, block_buf);
    }
    write_inode(fs, new_inode_num, &new_inode);
    // Add entry in parent directory
    if (dir_add_entry(fs, parent_inode_num, name, new_inode_num, DT_LNK) < 0) {
        return -1;
    }
    return 0;
}
```
**The Key Difference**:
- **Hard link**: Multiple directory entries → same inode → same data blocks. All names are equal. Data persists until *all* links are removed.
- **Symbolic link**: Directory entry → symlink inode → path string → resolve path → target inode. The symlink stores the *path*, not the inode. If the target is deleted, the symlink "dangles" (points to nothing).
```c
// Reading a symlink
int fs_readlink(FileSystem* fs, const char* path, char* buffer, size_t size) {
    uint64_t inode_num = path_resolve(fs, path);
    if (inode_num == 0) {
        return -1;
    }
    Inode inode;
    if (read_inode(fs, inode_num, &inode) < 0) {
        return -1;
    }
    if (!(inode.mode & S_IFLNK)) {
        errno = EINVAL;
        return -1;  // Not a symlink
    }
    size_t link_len = inode.size;
    if (link_len >= size) {
        link_len = size - 1;  // Leave room for null terminator
    }
    // Read target path
    if (link_len < 48) {
        // Short symlink: stored in direct pointers
        memcpy(buffer, inode.direct, link_len);
    } else {
        // Long symlink: stored in data block
        char block_buf[BLOCK_SIZE];
        uint64_t block = get_file_block(fs, &inode, 0);
        if (block == 0) {
            return -1;
        }
        read_block(fs->dev, block, block_buf);
        memcpy(buffer, block_buf, link_len);
    }
    buffer[link_len] = '\0';
    return (int)link_len;
}
```
## Listing Directory Contents: readdir
To implement `ls`, you need to iterate through directory entries:
```c
// Callback function type for directory iteration
typedef void (*dir_iter_callback)(uint64_t inode, const char* name, 
                                   uint8_t type, void* context);
// Iterate over all entries in a directory
int fs_readdir(FileSystem* fs, const char* path, 
               dir_iter_callback callback, void* context) {
    uint64_t dir_inode_num = path_resolve(fs, path);
    if (dir_inode_num == 0) {
        return -1;
    }
    Inode dir_inode;
    if (read_inode(fs, dir_inode_num, &dir_inode) < 0) {
        return -1;
    }
    if (!(dir_inode.mode & S_IFDIR)) {
        errno = ENOTDIR;
        return -1;
    }
    char block_buffer[BLOCK_SIZE];
    DirEntry* entries = (DirEntry*)block_buffer;
    uint64_t num_blocks = (dir_inode.size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (uint64_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        uint64_t phys_block = get_file_block(fs, &dir_inode, block_idx);
        if (phys_block == 0) continue;
        if (read_block(fs->dev, phys_block, block_buffer) < 0) {
            return -1;
        }
        for (int i = 0; i < ENTRIES_PER_BLOCK; i++) {
            if (entries[i].inode != 0) {
                // Call the callback with this entry
                callback(entries[i].inode, entries[i].name, 
                        entries[i].file_type, context);
            }
        }
    }
    // Update access time
    dir_inode.atime = (uint64_t)time(NULL);
    write_inode(fs, dir_inode_num, &dir_inode);
    return 0;
}
```
## Rename: Atomic Move
The `rename` operation moves an entry between directories atomically:
```c
// Rename/move a file or directory
int fs_rename(FileSystem* fs, const char* old_path, const char* new_path) {
    // Resolve old path
    char old_name[MAX_NAME_LEN + 1];
    uint64_t old_parent = path_resolve_parent(fs, old_path, old_name);
    if (old_parent == 0) {
        return -1;
    }
    uint64_t old_inode = dir_lookup(fs, old_parent, old_name);
    if (old_inode == 0) {
        errno = ENOENT;
        return -1;
    }
    // Resolve new path
    char new_name[MAX_NAME_LEN + 1];
    uint64_t new_parent = path_resolve_parent(fs, new_path, new_name);
    if (new_parent == 0) {
        return -1;
    }
    // Check if target exists
    uint64_t existing_inode = dir_lookup(fs, new_parent, new_name);
    if (existing_inode != 0) {
        // Target exists - would need to check if we can replace it
        // For simplicity, we'll fail
        errno = EEXIST;
        return -1;
    }
    // Get inode info to determine file type
    Inode inode;
    if (read_inode(fs, old_inode, &inode) < 0) {
        return -1;
    }
    uint8_t file_type = DT_REG;
    if (inode.mode & S_IFDIR) file_type = DT_DIR;
    else if (inode.mode & S_IFLNK) file_type = DT_LNK;
    // Add entry in new location
    if (dir_add_entry(fs, new_parent, new_name, old_inode, file_type) < 0) {
        return -1;
    }
    // Remove entry from old location
    // Note: This doesn't free the inode because we already added a new reference
    Inode old_parent_inode;
    read_inode(fs, old_parent, &old_parent_inode);
    char block_buffer[BLOCK_SIZE];
    DirEntry* entries = (DirEntry*)block_buffer;
    uint64_t num_blocks = (old_parent_inode.size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (uint64_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        uint64_t phys_block = get_file_block(fs, &old_parent_inode, block_idx);
        if (phys_block == 0) continue;
        read_block(fs->dev, phys_block, block_buffer);
        for (int i = 0; i < ENTRIES_PER_BLOCK; i++) {
            if (entries[i].inode == old_inode) {
                // Found it - clear the entry
                entries[i].inode = 0;
                entries[i].rec_len = 0;
                entries[i].name_len = 0;
                entries[i].file_type = 0;
                memset(entries[i].name, 0, 256);
                write_block(fs->dev, phys_block, block_buffer);
                // Update timestamps
                old_parent_inode.mtime = (uint64_t)time(NULL);
                old_parent_inode.ctime = old_parent_inode.mtime;
                write_inode(fs, old_parent, &old_parent_inode);
                // Handle directory move: update .. entry
                if (file_type == DT_DIR && old_parent != new_parent) {
                    // Update .. in the moved directory to point to new parent
                    char dir_block[BLOCK_SIZE];
                    uint64_t dir_block_num = get_file_block(fs, &inode, 0);
                    if (dir_block_num != 0) {
                        read_block(fs->dev, dir_block_num, dir_block);
                        DirEntry* dir_entries = (DirEntry*)dir_block;
                        // Find .. entry
                        for (int j = 0; j < ENTRIES_PER_BLOCK; j++) {
                            if (dir_entries[j].inode != 0 &&
                                strcmp(dir_entries[j].name, "..") == 0) {
                                // Decrement old parent's link count
                                Inode old_p;
                                read_inode(fs, old_parent, &old_p);
                                if (old_p.link_count > 0) old_p.link_count--;
                                write_inode(fs, old_parent, &old_p);
                                // Update .. to point to new parent
                                dir_entries[j].inode = new_parent;
                                write_block(fs->dev, dir_block_num, dir_block);
                                // Increment new parent's link count
                                Inode new_p;
                                read_inode(fs, new_parent, &new_p);
                                new_p.link_count++;
                                write_inode(fs, new_parent, &new_p);
                                break;
                            }
                        }
                    }
                }
                return 0;
            }
        }
    }
    // This shouldn't happen - we found it earlier
    return -1;
}
```
## Testing Directory Operations
```c
#include <assert.h>
void test_directory_operations(FileSystem* fs) {
    printf("Testing directory operations...\n");
    // Test 1: Create a directory
    uint64_t dir1 = fs_mkdir(fs, "/testdir");
    assert(dir1 > 0);
    printf("  ✓ Created /testdir (inode %lu)\n", dir1);
    // Test 2: Verify it exists
    uint64_t found = path_resolve(fs, "/testdir");
    assert(found == dir1);
    printf("  ✓ Path resolution finds /testdir\n");
    // Test 3: Check . and .. entries
    uint64_t dot = dir_lookup(fs, dir1, ".");
    uint64_t dotdot = dir_lookup(fs, dir1, "..");
    assert(dot == dir1);       // . points to self
    assert(dotdot == 1);       // .. points to root
    printf("  ✓ Special entries: . = %lu, .. = %lu\n", dot, dotdot);
    // Test 4: Create nested directory
    uint64_t dir2 = fs_mkdir(fs, "/testdir/subdir");
    assert(dir2 > 0);
    printf("  ✓ Created /testdir/subdir (inode %lu)\n", dir2);
    // Test 5: Verify parent's link count increased
    Inode dir1_inode;
    read_inode(fs, dir1, &dir1_inode);
    assert(dir1_inode.link_count == 3);  // parent entry + . + subdir's ..
    printf("  ✓ Parent link count is 3 (parent entry + . + child's ..)\n");
    // Test 6: Create a file in the directory
    uint64_t file_inode = alloc_inode(fs);
    Inode file;
    memset(&file, 0, sizeof(Inode));
    file.mode = S_IFREG | 0644;
    file.link_count = 1;
    file.size = 0;
    update_timestamps(&file, TS_ALL);
    write_inode(fs, file_inode, &file);
    dir_add_entry(fs, dir1, "testfile.txt", file_inode, DT_REG);
    printf("  ✓ Created testfile.txt (inode %lu)\n", file_inode);
    // Test 7: Look up the file
    uint64_t found_file = dir_lookup(fs, dir1, "testfile.txt");
    assert(found_file == file_inode);
    printf("  ✓ Lookup found testfile.txt\n");
    // Test 8: Create hard link
    fs_link(fs, "/testdir/testfile.txt", "/testdir/linktofile.txt");
    Inode linked_inode;
    read_inode(fs, file_inode, &linked_inode);
    assert(linked_inode.link_count == 2);
    printf("  ✓ Hard link created, link count = 2\n");
    // Test 9: Can't remove non-empty directory
    int rmdir_result = fs_rmdir(fs, "/testdir");
    assert(rmdir_result < 0 && errno == ENOTEMPTY);
    printf("  ✓ Can't rmdir non-empty directory\n");
    // Test 10: Remove file, then directory
    dir_remove_entry(fs, dir1, "testfile.txt");
    dir_remove_entry(fs, dir1, "linktofile.txt");
    printf("  ✓ Removed files from directory\n");
    // Test 11: Remove subdirectory
    fs_rmdir(fs, "/testdir/subdir");
    found = path_resolve(fs, "/testdir/subdir");
    assert(found == 0);  // Should not exist
    printf("  ✓ Removed /testdir/subdir\n");
    // Test 12: Remove directory
    fs_rmdir(fs, "/testdir");
    found = path_resolve(fs, "/testdir");
    assert(found == 0);
    printf("  ✓ Removed /testdir\n");
    printf("All directory tests passed!\n");
}
```

![Filesystem Layer Stack](./diagrams/diag-l1-layer-stack.svg)

## Common Pitfalls
**Forgetting to update parent link count on mkdir/rmdir**: When you create a directory, the parent's `link_count` must increase by 1 (for the child's `..` entry). When you remove a directory, it must decrease by 1. Forgetting this causes `fsck` to report incorrect link counts.
**Not handling root's `..` correctly**: Root's `..` must point to itself. If you follow `..` from root and get something else, path resolution can loop forever or crash.
**Race conditions in concurrent access**: Multiple threads/processes can modify a directory simultaneously. Without locking, two threads could:
- Both check "name doesn't exist", then both add the same name
- One thread read a directory entry while another removes it
- One thread follow `..` while another removes the directory
Real filesystems use per-inode locks (mutexes) to serialize directory modifications.
**Memory leaks in path resolution**: If you allocate memory for path components, make sure to free it on all error paths. Better yet, use stack-allocated buffers with a maximum component count.
**Not checking for cycles**: While hard links to directories are forbidden, symbolic links can create cycles. Path resolution must track visited symlinks and fail with `ELOOP` if too many are encountered.
## What You've Built
At the end of this milestone, you have:
1. **Directory entry structure** that maps names to inode numbers
2. **add_entry and remove_entry** operations for modifying directories
3. **Path resolution** that traverses the directory tree component by component
4. **Special entries** (`.` and `..`) that create self-reference and parent-reference
5. **mkdir and rmdir** with proper link count management
6. **Hard links** that create multiple names for the same inode
7. **Symbolic links** that store path strings as file data
8. **readdir** for listing directory contents
9. **rename** for atomic move operations
You can now:
```bash
$ ./mkfs disk.img 100
$ ./fs_shell disk.img
fs> mkdir /home
fs> mkdir /home/user
fs> echo "Hello" > /home/user/file.txt
fs> ln /home/user/file.txt /home/user/link.txt
fs> ls /home/user
  file.txt
  link.txt
fs> cat /home/user/link.txt
  Hello
fs> rm /home/user/file.txt
fs> cat /home/user/link.txt
  Hello  # Data still exists - link.txt is still connected
fs> rm /home/user/link.txt
fs> ls /home/user
  (empty)
fs> rmdir /home/user
fs> exit
```
## The Knowledge Cascade
The patterns you've learned extend far beyond filesystems:
**DNS Resolution**: The hierarchical path resolution you implemented is isomorphic to DNS lookups. Resolving `/home/user/docs/file.txt` parallels resolving `docs.user.home.example.com`:
- Start at root (`/` or `.` DNS zone)
- Look up first component (`home` or `home.example.com`)
- Recurse into the returned zone
- Continue until you reach the target
DNS caches correspond directly to the Linux **dentry cache** — both store recently-resolved names to avoid repeated lookups. A DNS cache miss triggers a query to authoritative servers (disk reads); a dentry cache miss triggers directory block reads.
**Garbage Collection Reference Counting**: The link count mechanism you implemented is exactly how Python, PHP, and Swift manage memory. Each object has a reference count:
```python
a = [1, 2, 3]  # refcount = 1
b = a          # refcount = 2 (hard link equivalent)
del a          # refcount = 1 (unlink)
del b          # refcount = 0, object freed (inode deallocated)
```
The difference: filesystems persist reference counts to disk; language runtimes keep them in memory. But the cycle problem is identical — circular references prevent collection. Python's cycle collector is analogous to `fsck` detecting and repairing orphaned inodes.
**Database Indexes**: A directory is essentially a **dense index** mapping string keys (names) to integer values (inode numbers). Database B-tree indexes serve the same purpose for table rows. The key difference: directories are typically scanned linearly, while B-trees use logarithmic search. This is why ext4 switched to HTree indexing for large directories — a 100,000-entry directory is unusable with linear scan but instant with a hash tree.
**Package Managers and Dependency Graphs**: Directories as name→inode maps parallel package registries as name→version maps. When you run `npm install lodash`, npm:
1. Resolves "lodash" in the registry (like `dir_lookup(fs, registry_dir, "lodash")`)
2. Gets a version identifier (like getting an inode number)
3. Resolves that version's dependencies (like following indirect blocks)
"Dependency hell" occurs when different packages require incompatible versions — analogous to trying to create two directory entries with the same name pointing to different inodes. The name can only map to one target.
**Web Routing**: URL path routing in web frameworks is directory resolution with extra steps. When Express.js matches `/users/:id/posts/:postId`, it's doing the same component-by-component traversal, but with pattern matching instead of exact string matching. The router is a "virtual directory" that exists only at runtime.
In the next milestone, you'll implement the full file I/O operations — creating, reading, writing, and truncating files with proper sparse file support. The directory operations you built today provide the namespace; file I/O will provide the data storage.
---
<!-- END_MS -->


<!-- MS_ID: filesystem-m4 -->
# File Read/Write Operations
You've built the machinery to navigate the filesystem: paths resolve to inodes, directories map names to inode numbers, and inodes track block pointers through direct and indirect schemes. Now you face the actual purpose of all that infrastructure: **storing and retrieving user data.**
This milestone is where abstraction meets reality. Users think in terms of "write these bytes at this offset" and "read N bytes starting here." Your filesystem must translate these simple requests into a cascade of block allocations, pointer traversals, partial-block operations, and metadata updates — all while maintaining the invariants that keep data intact across crashes.
## The Fundamental Tension: Files Are Logical, Blocks Are Physical
Here's the core problem: **users operate on byte ranges, but storage operates on blocks.**
When a user writes "Hello, World!" (13 bytes) at offset 4090, they expect those bytes to land at positions 4090-4102 in the file. But your storage device only understands 4KB blocks. You can't write "bytes 4090-4102" — you can only write "block 42" or "block 43."
```
User's mental model:
  File: [byte 0] [byte 1] [byte 2] ... [byte 4090='H'] ... [byte 4102='!'] ...
         └─────────────────────────────────────────────────────────────────┘
                                      13 bytes to write
Storage reality:
  Block 42: [byte 0 ─────── byte 4095]    Block 43: [byte 4096 ─────── byte 8191]
                    └─ write spans both blocks! ─┘
```
This mismatch creates three categories of complexity:
1. **Block boundary crossing**: A 13-byte write at offset 4090 spans two blocks (4090-4095 in block 42, 4096-4102 in block 43). You must read both blocks, modify the relevant portions, and write both back.
2. **Sparse file representation**: A user can write at offset 1GB without ever touching bytes 0 through 1GB-1. Allocating a million blocks for zeros would be wasteful. Instead, you represent the gap as a **hole** — a region where the inode's block pointers are null. Reading from a hole returns zeros without any disk access.
3. **Allocation on demand**: Writing past the end of a file requires allocating new blocks. But allocating a block is expensive (bitmap search + metadata update). You only allocate when actually written, not when the file size is extended.
The tension between logical byte streams and physical block storage is why file I/O is far more complex than it appears. Every operation is a negotiation between what the user wants (bytes) and what the hardware provides (blocks).
## The Revelation: File Size Is Just Metadata
Here's what surprises most developers: **the file size stored in the inode is not a physical constraint — it's a hint.**
When you call `truncate(file, 1_000_000_000)` to make a file 1GB, the filesystem doesn't allocate 256,000 blocks. It just sets `inode.size = 1_000_000_000`. The blocks don't exist yet. They're created lazily, only when you actually write to them.
Conversely, a file with `inode.size = 1_000_000_000` might only have 10 blocks allocated — the rest are holes. This is how sparse files work:
```
A 1GB sparse file with 10KB of actual data:
Logical view:  [0 ......... 1GB]
               ├─ hole ───┤┌─ data ─┐├─ hole ──┤
               0          500MB    500.01MB   1GB
Physical reality:
  - inode.size = 1,000,000,000
  - inode.blocks = 3 (only 3 × 4KB allocated)
  - Blocks allocated at offsets 500MB-500.01MB only
  - Reading anywhere else returns zeros (synthesized)
```

> **🔑 Foundation: Sparse files and holes**
> 
> ## Sparse Files and Holes
### What It IS
A **sparse file** is a file that contains "holes" — regions of zero bytes that don't actually occupy disk space. The filesystem records these holes as logical zeros but doesn't allocate physical blocks for them.
Think of it like Swiss cheese: the file *logically* spans from byte 0 to byte 1 million, but physically, only the "solid" parts take up disk space.
```
A regular file:     [DDDDDDDDDDDDDDDDDDDD]  (20 blocks on disk)
A sparse file:      [DDDD----DDDD----DDDD]  (12 blocks on disk, 8 are holes)
```
Where `D` = data block, `-` = hole (no disk allocation)
**Creating a sparse file:**
```c
int fd = open("sparse.dat", O_WRONLY | O_CREAT, 0644);
write(fd, "HEAD", 4);        // 4 bytes of real data
lseek(fd, 999999, SEEK_CUR); // jump forward 1MB
write(fd, "TAIL", 4);        // 4 more bytes
// Result: ~1MB logical size, ~4KB physical size
```
### WHY You Need It Right Now
Sparse files appear frequently in systems programming:
1. **Disk images** — A 10GB VM image might only have 2GB of actual data; the rest is holes
2. **Databases** — Pre-allocated files where data will grow over time
3. **Log rotation** — Pre-allocating space without immediate disk consumption
4. **Backup tools** — Need to detect holes to avoid backing up gigabytes of zeros
**Critical gotcha:** `ls -l` shows logical size (misleadingly large). Use `du -h` or `stat` to see actual block allocation:
```bash
$ ls -l sparse.dat
-rw-r--r-- 1 user user 1000008 Mar  4 10:00 sparse.dat   # 1MB logical
$ du -h sparse.dat
8.0K    sparse.dat    # Only 8KB on disk!
$ stat sparse.dat
  Size: 1000008     Blocks: 16         # 16 × 512-byte blocks
```
### ONE Key Insight
**Holes are a filesystem-level optimization, not a file format.**
When you *read* a sparse file, the holes look like zeros — your program can't tell the difference between a hole and actual zero bytes written to disk. The distinction only matters for:
- Disk space usage
- Copying/cloning files (should you preserve holes or write real zeros?)
- Network transfer efficiency
The `cp` command's `--sparse=always|auto|never` flag controls this behavior. Many tools destroy sparseness by reading and re-writing zeros, accidentally inflating file size.
**Mental model:** A sparse file is a file with "don't care" regions the filesystem is free to represent as zeros without storing them. It's compression at the allocation layer, not the data layer.

This has profound implications:
- **Reading from a hole** doesn't access disk. The filesystem detects a null block pointer and returns zeros from memory.
- **Writing to a hole** allocates a block first. The filesystem can't write to a block that doesn't exist.
- **Truncation extending a file** creates a hole. No blocks allocated, just metadata updated.
- **Truncation shrinking a file** may free blocks. Blocks beyond the new size are deallocated.
The file size is a logical boundary. The actual storage is determined by which block pointers are non-null.
## Three-Level View: File I/O in Context
Let's trace a file read through all three levels:
**Level 1 — Application (Your Filesystem)**
```c
file_read(fs, inode_num, 5000, buffer, 100);
```
The application requests 100 bytes starting at offset 5000. Your code must:
1. Calculate which block contains offset 5000 (block 1, since 5000 / 4096 = 1)
2. Calculate the offset within that block (5000 % 4096 = 904)
3. Traverse the inode's block pointers to find physical block number
4. Read the block and extract bytes 904-1003
**Level 2 — OS/Kernel**
Your `read_block()` call invokes the kernel via `read()` syscall on the backing file. The kernel:
1. Checks the page cache — is block 1 already in memory?
2. If not, issues a read request to the block layer
3. The block layer may merge this with adjacent requests (I/O scheduling)
4. The request travels to the storage driver
**Level 3 — Hardware**
The storage controller:
1. Translates the logical block address to physical location (SSD: flash page, HDD: sector)
2. Performs the read (SSD: ~25μs, HDD: ~10ms including seek)
3. Transfers data via DMA to the kernel's buffer
4. The kernel copies to your userspace buffer
**Hardware Soul Check**: A single 100-byte read that misses cache triggers a full 4KB block read from storage. On an HDD, that's ~10ms to read 4096 bytes when you only needed 100. This is why read-ahead exists — the kernel speculates you'll want adjacent blocks and reads them proactively. Your filesystem's block-aligned design matches what the hardware wants; the inefficiency is in reading more than requested, not in your design.
## File Creation: Allocating the Inode
Creating a file combines directory operations (from Milestone 3) with inode initialization:
```c
// Create a new regular file
// Returns inode number on success, 0 on error
uint64_t fs_create(FileSystem* fs, const char* path, uint16_t mode) {
    // Resolve parent directory and extract final component
    char name[MAX_NAME_LEN + 1];
    uint64_t parent_inode_num = path_resolve_parent(fs, path, name);
    if (parent_inode_num == 0) {
        return 0;  // Parent doesn't exist
    }
    // Check if name already exists
    if (dir_lookup(fs, parent_inode_num, name) != 0) {
        errno = EEXIST;
        return 0;
    }
    // Verify parent is a directory
    Inode parent_inode;
    if (read_inode(fs, parent_inode_num, &parent_inode) < 0) {
        return 0;
    }
    if (!(parent_inode.mode & S_IFDIR)) {
        errno = ENOTDIR;
        return 0;
    }
    // Allocate new inode
    uint64_t new_inode_num = alloc_inode(fs);
    if (new_inode_num == 0) {
        errno = ENOSPC;
        return 0;  // No free inodes
    }
    // Initialize inode as empty regular file
    Inode new_inode;
    memset(&new_inode, 0, sizeof(Inode));
    new_inode.mode = S_IFREG | (mode & 07777);  // Regular file with requested permissions
    new_inode.uid = 0;   // Would be process's uid in real system
    new_inode.gid = 0;   // Would be process's gid
    new_inode.link_count = 1;
    new_inode.size = 0;
    new_inode.blocks = 0;
    new_inode.atime = new_inode.mtime = new_inode.ctime = (uint64_t)time(NULL);
    if (write_inode(fs, new_inode_num, &new_inode) < 0) {
        // Rollback: free the inode in bitmap
        // (would need to implement free_inode function)
        return 0;
    }
    // Add entry in parent directory
    if (dir_add_entry(fs, parent_inode_num, name, new_inode_num, DT_REG) < 0) {
        // Rollback would go here
        return 0;
    }
    // Update parent's mtime and ctime
    parent_inode.mtime = (uint64_t)time(NULL);
    parent_inode.ctime = parent_inode.mtime;
    write_inode(fs, parent_inode_num, &parent_inode);
    return new_inode_num;
}
```
The newly created file has:
- `size = 0`: No logical bytes yet
- `blocks = 0`: No physical storage allocated
- All block pointers = 0: Every block is a hole
The file exists as a metadata entry only. No storage is consumed until the first write.
## File Read: Mapping Offsets to Blocks
Reading requires translating logical byte offsets to physical block numbers, then extracting the relevant bytes:
```c
// Read data from a file at the given offset
// Returns number of bytes read, or -1 on error
ssize_t fs_read(FileSystem* fs, uint64_t inode_num, uint64_t offset,
                void* buffer, size_t length) {
    Inode inode;
    if (read_inode(fs, inode_num, &inode) < 0) {
        return -1;
    }
    // Check file type
    if (inode.mode & S_IFDIR) {
        errno = EISDIR;
        return -1;
    }
    // Check bounds: reading past end of file returns 0 bytes
    if (offset >= inode.size) {
        return 0;
    }
    // Clamp length to available data
    if (offset + length > inode.size) {
        length = inode.size - offset;
    }
    if (length == 0) {
        return 0;
    }
    size_t bytes_read = 0;
    char* out = (char*)buffer;
    char block_buffer[BLOCK_SIZE];
    while (bytes_read < length) {
        // Calculate which logical block contains this byte
        uint64_t file_block_idx = (offset + bytes_read) / BLOCK_SIZE;
        // Calculate offset within that block
        uint64_t block_offset = (offset + bytes_read) % BLOCK_SIZE;
        // Get the physical block number (may be 0 for hole)
        uint64_t phys_block = get_file_block(fs, &inode, file_block_idx);
        // Calculate how many bytes to read from this block
        size_t bytes_remaining = length - bytes_read;
        size_t bytes_in_block = BLOCK_SIZE - block_offset;
        size_t to_read = (bytes_remaining < bytes_in_block) ? bytes_remaining : bytes_in_block;
        if (phys_block == 0) {
            // Hole: synthesize zeros without disk access
            memset(out + bytes_read, 0, to_read);
        } else {
            // Read the actual block
            if (read_block(fs->dev, phys_block, block_buffer) < 0) {
                return -1;
            }
            memcpy(out + bytes_read, block_buffer + block_offset, to_read);
        }
        bytes_read += to_read;
    }
    // Update access time
    inode.atime = (uint64_t)time(NULL);
    write_inode(fs, inode_num, &inode);
    return (ssize_t)bytes_read;
}
```
**The Hole Detection Logic**: The key insight is `if (phys_block == 0)`. When `get_file_block()` returns 0, it means no block is allocated at that logical position. Rather than failing, we synthesize zeros. This is how sparse files work — holes are implicit, represented by the absence of a block pointer.
**Hardware Soul Check**: Consider reading 1 byte from each of 1000 different 4KB blocks in a sparse file:
- If all are holes: 0 disk reads, just memset operations
- If all are allocated: 1000 disk reads minimum
- If allocated but cached: 0 disk reads (page cache hits)
Sparse files can be dramatically faster to read than dense files because holes bypass I/O entirely.
## Block Boundary Handling: The Read Loop
The read loop handles three cases:
1. **Read entirely within one block**: `to_read` equals the remaining length
2. **Read spans two blocks**: First iteration reads to end of block, second reads from start of next
3. **Read spans many blocks**: Loop continues until all bytes are read
```
Example: Read 8000 bytes starting at offset 3000
Block 0          Block 1          Block 2
[0──────4095]   [4096────8191]   [8192────12287]
     └────────────────────────────────┘
          8000 bytes starting at 3000
Iteration 1: Read bytes 3000-4095 from Block 0 (1096 bytes)
Iteration 2: Read bytes 4096-8191 from Block 1 (4096 bytes)  
Iteration 3: Read bytes 8192-10999 from Block 2 (2808 bytes)
Total: 8000 bytes in 3 block reads
```
The loop naturally handles all cases because each iteration calculates `to_read` based on how much remains and how much space is left in the current block.
## File Write: Allocation on Demand
Writing is more complex than reading because it may require:
1. Allocating new blocks for holes being filled
2. Allocating indirect blocks for file growth
3. Updating the inode size if writing past end
4. Handling partial-block writes (read-modify-write)
```c
// Write data to a file at the given offset
// Returns number of bytes written, or -1 on error
ssize_t fs_write(FileSystem* fs, uint64_t inode_num, uint64_t offset,
                 const void* data, size_t length) {
    Inode inode;
    if (read_inode(fs, inode_num, &inode) < 0) {
        return -1;
    }
    // Check file type
    if (inode.mode & S_IFDIR) {
        errno = EISDIR;
        return -1;
    }
    // Check for maximum file size (beyond double-indirect range)
    uint64_t max_size = DIRECT_MAX_SIZE + 
                        PTRS_PER_INDIRECT * BLOCK_SIZE +
                        PTRS_PER_INDIRECT * PTRS_PER_INDIRECT * BLOCK_SIZE;
    if (offset + length > max_size) {
        errno = EFBIG;
        return -1;
    }
    if (length == 0) {
        return 0;
    }
    size_t bytes_written = 0;
    const char* in = (const char*)data;
    char block_buffer[BLOCK_SIZE];
    bool inode_modified = false;
    while (bytes_written < length) {
        // Calculate which logical block contains this byte
        uint64_t file_block_idx = (offset + bytes_written) / BLOCK_SIZE;
        // Calculate offset within that block
        uint64_t block_offset = (offset + bytes_written) % BLOCK_SIZE;
        // Get or allocate the physical block
        uint64_t phys_block = get_or_alloc_block(fs, inode_num, &inode, file_block_idx);
        if (phys_block == 0) {
            // Allocation failed - stop here
            break;
        }
        // Calculate how many bytes to write to this block
        size_t bytes_remaining = length - bytes_written;
        size_t bytes_in_block = BLOCK_SIZE - block_offset;
        size_t to_write = (bytes_remaining < bytes_in_block) ? bytes_remaining : bytes_in_block;
        // For partial-block writes, we need read-modify-write
        if (block_offset > 0 || to_write < BLOCK_SIZE) {
            // Read existing block content
            if (read_block(fs->dev, phys_block, block_buffer) < 0) {
                break;
            }
            // Modify the relevant portion
            memcpy(block_buffer + block_offset, in + bytes_written, to_write);
            // Write back
            if (write_block(fs->dev, phys_block, block_buffer) < 0) {
                break;
            }
        } else {
            // Full-block write: no need to read first
            if (write_block(fs->dev, phys_block, in + bytes_written) < 0) {
                break;
            }
        }
        bytes_written += to_write;
        inode_modified = true;
    }
    // Update inode metadata
    if (bytes_written > 0) {
        // Extend file size if we wrote past the end
        if (offset + bytes_written > inode.size) {
            inode.size = offset + bytes_written;
        }
        inode.mtime = (uint64_t)time(NULL);
        inode.ctime = inode.mtime;
        write_inode(fs, inode_num, &inode);
    }
    return (ssize_t)bytes_written;
}
```
**The Read-Modify-Write Pattern**: When writing less than a full block, or writing to a non-block-aligned offset, you can't just overwrite. You must:
1. Read the existing block content
2. Modify the relevant bytes
3. Write the entire block back
This is why partial writes are more expensive than full-block writes — they require two I/O operations (read + write) instead of one (write only).

![Filesystem Layer Stack](./diagrams/diag-l1-layer-stack.svg)

## Block Allocation: The get_or_alloc_block Function
The core of write-side allocation is ensuring a block exists before writing to it:
```c
// Get the physical block number for a file's logical block
// If the block doesn't exist (hole), allocate it first
// Returns physical block number, or 0 on error
uint64_t get_or_alloc_block(FileSystem* fs, uint64_t inode_num,
                            Inode* inode, uint64_t file_block_idx) {
    // First check if block already exists
    uint64_t existing = get_file_block(fs, inode, file_block_idx);
    if (existing != 0) {
        return existing;  // Block already allocated
    }
    // Need to allocate a new block
    uint64_t new_block = alloc_block(fs);
    if (new_block == 0) {
        return 0;  // No free blocks
    }
    // Now we need to write the pointer into the inode's block tree
    // This depends on which zone the block falls into
    if (file_block_idx < DIRECT_BLOCKS) {
        // Direct block
        inode->direct[file_block_idx] = new_block;
        inode->blocks++;
        return new_block;
    }
    file_block_idx -= DIRECT_BLOCKS;
    if (file_block_idx < PTRS_PER_INDIRECT) {
        // Single-indirect block
        int idx = (int)file_block_idx;
        // Ensure indirect block exists
        if (inode->indirect == 0) {
            uint64_t indirect_block = alloc_block(fs);
            if (indirect_block == 0) {
                free_block(fs, new_block);
                return 0;
            }
            inode->indirect = indirect_block;
            inode->blocks++;
            // Zero the new indirect block
            char zero_buf[BLOCK_SIZE];
            memset(zero_buf, 0, BLOCK_SIZE);
            write_block(fs->dev, indirect_block, zero_buf);
        }
        // Write pointer into indirect block
        char ind_buf[BLOCK_SIZE];
        if (read_block(fs->dev, inode->indirect, ind_buf) < 0) {
            free_block(fs, new_block);
            return 0;
        }
        uint64_t* ptrs = (uint64_t*)ind_buf;
        ptrs[idx] = new_block;
        write_block(fs->dev, inode->indirect, ind_buf);
        inode->blocks++;
        return new_block;
    }
    file_block_idx -= PTRS_PER_INDIRECT;
    if (file_block_idx < (uint64_t)PTRS_PER_INDIRECT * PTRS_PER_INDIRECT) {
        // Double-indirect block
        int double_idx = (int)(file_block_idx / PTRS_PER_INDIRECT);
        int indirect_idx = (int)(file_block_idx % PTRS_PER_INDIRECT);
        // Ensure double-indirect block exists
        if (inode->double_ind == 0) {
            uint64_t double_block = alloc_block(fs);
            if (double_block == 0) {
                free_block(fs, new_block);
                return 0;
            }
            inode->double_ind = double_block;
            inode->blocks++;
            char zero_buf[BLOCK_SIZE];
            memset(zero_buf, 0, BLOCK_SIZE);
            write_block(fs->dev, double_block, zero_buf);
        }
        // Read double-indirect block
        char double_buf[BLOCK_SIZE];
        if (read_block(fs->dev, inode->double_ind, double_buf) < 0) {
            free_block(fs, new_block);
            return 0;
        }
        uint64_t* double_ptrs = (uint64_t*)double_buf;
        // Ensure the indirect block exists
        if (double_ptrs[double_idx] == 0) {
            uint64_t indirect_block = alloc_block(fs);
            if (indirect_block == 0) {
                free_block(fs, new_block);
                return 0;
            }
            double_ptrs[double_idx] = indirect_block;
            write_block(fs->dev, inode->double_ind, double_buf);
            inode->blocks++;
            char zero_buf[BLOCK_SIZE];
            memset(zero_buf, 0, BLOCK_SIZE);
            write_block(fs->dev, indirect_block, zero_buf);
        }
        // Read indirect block and write data block pointer
        char ind_buf[BLOCK_SIZE];
        uint64_t indirect_block = double_ptrs[double_idx];
        if (read_block(fs->dev, indirect_block, ind_buf) < 0) {
            free_block(fs, new_block);
            return 0;
        }
        uint64_t* ind_ptrs = (uint64_t*)ind_buf;
        ind_ptrs[indirect_idx] = new_block;
        write_block(fs->dev, indirect_block, ind_buf);
        inode->blocks++;
        return new_block;
    }
    // Beyond our addressing capability
    free_block(fs, new_block);
    return 0;
}
```
**The Cascade of Allocation**: Writing a single byte at offset 5MB might trigger:
1. Allocate the double-indirect block (1 block)
2. Allocate an indirect block (1 block)
3. Allocate the data block (1 block)
That's 3 allocations for 1 byte! The filesystem amortizes this by keeping indirect blocks — once allocated, they can address many data blocks.
## Sparse File Creation: Writing at High Offsets
Let's trace what happens when you write at a high offset in an empty file:
```c
// Create a sparse file
uint64_t inode_num = fs_create(fs, "/sparse_test", 0644);
// Write 100 bytes at offset 10MB
char data[100] = "This data lives at 10MB offset in the logical file...";
fs_write(fs, inode_num, 10 * 1024 * 1024, data, 100);
```
What happens internally:
1. `fs_write` calculates the block index: 10MB / 4KB = 2560
2. Block 2560 is in the double-indirect zone (past direct and single-indirect)
3. `get_or_alloc_block` allocates:
   - The double-indirect block (inode->double_ind)
   - An indirect block for entries 2560-3071
   - The actual data block
4. Only 3 blocks allocated, not 2560
5. Reading from offset 0 returns zeros (hole), not disk reads
```c
// Verify sparse behavior
Inode inode;
read_inode(fs, inode_num, &inode);
printf("File size: %lu bytes (logical)\n", inode.size);      // ~10MB
printf("Blocks allocated: %lu (physical)\n", inode.blocks);  // 3
// Read from the hole at offset 0
char buf[100];
fs_read(fs, inode_num, 0, buf, 100);
// buf contains all zeros, no disk I/O for this read
// Read from the actual data at offset 10MB
fs_read(fs, inode_num, 10 * 1024 * 1024, buf, 100);
// buf contains "This data lives...", one disk I/O
```
**Hardware Soul Check**: Sparse files are a performance hack that exploits the hole behavior. A database storing sparse matrices, a VM disk image with empty regions, or a container layer with mostly-identical files — all benefit from not allocating storage for zeros. The trade-off is fragmentation: allocated blocks may be scattered across the disk, causing seek overhead on spinning media.
## File Truncation: Growing and Shrinking
Truncation has two modes with very different implementations:
```c
// Truncate a file to the specified size
int fs_truncate(FileSystem* fs, uint64_t inode_num, uint64_t new_size) {
    Inode inode;
    if (read_inode(fs, inode_num, &inode) < 0) {
        return -1;
    }
    // Check file type
    if (inode.mode & S_IFDIR) {
        errno = EISDIR;
        return -1;
    }
    if (new_size == inode.size) {
        return 0;  // Nothing to do
    }
    if (new_size > inode.size) {
        // Growing: just update size (new regions become holes)
        inode.size = new_size;
        inode.mtime = (uint64_t)time(NULL);
        inode.ctime = inode.mtime;
        return write_inode(fs, inode_num, &inode);
    }
    // Shrinking: need to free blocks beyond new_size
    uint64_t old_size = inode.size;
    uint64_t old_last_block = (old_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint64_t new_last_block = (new_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Free blocks from new_last_block to old_last_block
    for (uint64_t block_idx = new_last_block; block_idx < old_last_block; block_idx++) {
        uint64_t phys_block = get_file_block(fs, &inode, block_idx);
        if (phys_block != 0) {
            free_block(fs, phys_block);
            // Clear the pointer in the inode's block tree
            clear_block_pointer(fs, &inode, block_idx);
            inode.blocks--;
        }
    }
    // Handle partial block at the new end
    if (new_size % BLOCK_SIZE != 0 && new_last_block > 0) {
        // Zero the bytes beyond new_size in the last block
        uint64_t last_block_idx = new_last_block - 1;
        uint64_t phys_block = get_file_block(fs, &inode, last_block_idx);
        if (phys_block != 0) {
            char block_buf[BLOCK_SIZE];
            if (read_block(fs->dev, phys_block, block_buf) == 0) {
                uint64_t offset_in_block = new_size % BLOCK_SIZE;
                memset(block_buf + offset_in_block, 0, BLOCK_SIZE - offset_in_block);
                write_block(fs->dev, phys_block, block_buf);
            }
        }
    }
    // Update inode
    inode.size = new_size;
    inode.mtime = (uint64_t)time(NULL);
    inode.ctime = inode.mtime;
    return write_inode(fs, inode_num, &inode);
}
```
**Growing vs. Shrinking**:
- **Growing** (`new_size > old_size`): Just update the size field. No blocks allocated. The new region is implicitly a hole.
- **Shrinking** (`new_size < old_size`): Free all blocks beyond the new size. This requires traversing the block pointer tree and calling `free_block` for each.
**The Partial Block Problem**: When truncating to a non-block-aligned size, the last block may need partial zeroing. For example, truncating from 10KB to 6KB means:
- Blocks 0 (0-4KB) and 1 (4KB-8KB) remain
- Block 1 needs bytes 6KB-8KB zeroed
- Blocks 2+ are freed
```c
// Helper: Clear a block pointer in the inode's tree
static int clear_block_pointer(FileSystem* fs, Inode* inode, uint64_t block_idx) {
    if (block_idx < DIRECT_BLOCKS) {
        inode->direct[block_idx] = 0;
        return 0;
    }
    block_idx -= DIRECT_BLOCKS;
    if (block_idx < PTRS_PER_INDIRECT) {
        if (inode->indirect == 0) return 0;
        char buf[BLOCK_SIZE];
        if (read_block(fs->dev, inode->indirect, buf) < 0) return -1;
        uint64_t* ptrs = (uint64_t*)buf;
        ptrs[block_idx] = 0;
        write_block(fs->dev, inode->indirect, buf);
        return 0;
    }
    block_idx -= PTRS_PER_INDIRECT;
    if (block_idx < (uint64_t)PTRS_PER_INDIRECT * PTRS_PER_INDIRECT) {
        if (inode->double_ind == 0) return 0;
        int double_idx = (int)(block_idx / PTRS_PER_INDIRECT);
        int indirect_idx = (int)(block_idx % PTRS_PER_INDIRECT);
        char double_buf[BLOCK_SIZE];
        if (read_block(fs->dev, inode->double_ind, double_buf) < 0) return -1;
        uint64_t* double_ptrs = (uint64_t*)double_buf;
        if (double_ptrs[double_idx] == 0) return 0;
        char ind_buf[BLOCK_SIZE];
        if (read_block(fs->dev, double_ptrs[double_idx], ind_buf) < 0) return -1;
        uint64_t* ind_ptrs = (uint64_t*)ind_buf;
        ind_ptrs[indirect_idx] = 0;
        write_block(fs->dev, double_ptrs[double_idx], ind_buf);
        return 0;
    }
    return -1;
}
```
## Freeing Indirect Blocks on Shrink
When truncating below the threshold of an indirect zone, you should free the indirect block itself:
```c
// Extended truncate that frees unused indirect blocks
int fs_truncate_with_indirect_free(FileSystem* fs, uint64_t inode_num, 
                                    uint64_t new_size) {
    Inode inode;
    if (read_inode(fs, inode_num, &inode) < 0) {
        return -1;
    }
    if (new_size >= inode.size) {
        return fs_truncate(fs, inode_num, new_size);
    }
    // Calculate which zones we're truncating into
    uint64_t direct_zone_end = DIRECT_BLOCKS * BLOCK_SIZE;
    uint64_t single_indirect_end = direct_zone_end + 
                                   PTRS_PER_INDIRECT * BLOCK_SIZE;
    // First, do the standard truncation
    int result = fs_truncate(fs, inode_num, new_size);
    if (result < 0) return result;
    // Now free unused indirect blocks
    // If we truncated below the single-indirect zone, free it
    if (new_size < direct_zone_end && inode.indirect != 0) {
        free_block(fs, inode.indirect);
        inode.indirect = 0;
        inode.blocks--;
    }
    // If we truncated below the double-indirect zone, free it and all children
    if (new_size < single_indirect_end && inode.double_ind != 0) {
        // Free all indirect blocks referenced by double_ind
        char double_buf[BLOCK_SIZE];
        if (read_block(fs->dev, inode.double_ind, double_buf) == 0) {
            uint64_t* double_ptrs = (uint64_t*)double_buf;
            for (int i = 0; i < PTRS_PER_INDIRECT; i++) {
                if (double_ptrs[i] != 0) {
                    free_block(fs, double_ptrs[i]);
                    inode.blocks--;
                }
            }
        }
        free_block(fs, inode.double_ind);
        inode.double_ind = 0;
        inode.blocks--;
    }
    write_inode(fs, inode_num, &inode);
    return 0;
}
```
**The Memory Leak Danger**: If you forget to free indirect blocks when truncating, those blocks become orphaned — allocated in the bitmap but unreachable from any inode. This is why `fsck` checks for unreachable blocks.
## Append Operation: Efficient End-of-File Writes
Append is a common pattern — writing to the end of a file without knowing its current size:
```c
// Append data to the end of a file
// Returns number of bytes written, or -1 on error
ssize_t fs_append(FileSystem* fs, uint64_t inode_num, 
                  const void* data, size_t length) {
    Inode inode;
    if (read_inode(fs, inode_num, &inode) < 0) {
        return -1;
    }
    // Append is just a write at the current end
    return fs_write(fs, inode_num, inode.size, data, length);
}
```
**Hardware Soul Check**: In a real filesystem, append is optimized with pre-allocation. When you append to a file, the filesystem may allocate extra blocks speculatively, anticipating more appends. This reduces fragmentation and allocation overhead. The trade-off is internal fragmentation — the file appears larger than its actual data. ext4's "preallocation" and XFS's "allocation groups" both implement this optimization.
## Metadata Updates: When and Why
Every file modification must update metadata:
| Operation | atime | mtime | ctime | size | blocks |
|-----------|-------|-------|-------|------|--------|
| read | ✓ | - | - | - | - |
| write | - | ✓ | ✓ | ✓* | ✓* |
| truncate (grow) | - | ✓ | ✓ | ✓ | - |
| truncate (shrink) | - | ✓ | ✓ | ✓ | ✓ |
*Only if size/blocks actually changed
```c
// Update timestamps based on operation type
typedef enum {
    OP_READ,      // File was read
    OP_WRITE,     // File content was modified
    OP_METADATA   // File metadata was changed
} FileOperation;
void update_file_timestamps(Inode* inode, FileOperation op) {
    uint64_t now = (uint64_t)time(NULL);
    switch (op) {
        case OP_READ:
            inode->atime = now;
            break;
        case OP_WRITE:
            inode->mtime = now;
            inode->ctime = now;  // ctime changes when mtime changes
            break;
        case OP_METADATA:
            inode->ctime = now;
            break;
    }
}
```
**Why ctime Changes on mtime Change**: The `ctime` (change time) tracks *any* modification to the inode, including content changes. It's a superset of mtime. This is useful for backup programs that want to detect any change, not just content changes.
## Testing: Round-Trip Verification
A robust test suite verifies that writes and reads are inverses:
```c
#include <assert.h>
#include <string.h>
#include <stdio.h>
void test_file_read_write(FileSystem* fs) {
    printf("Testing file read/write operations...\n");
    // Test 1: Create and write a small file
    uint64_t inode_num = fs_create(fs, "/testfile.txt", 0644);
    assert(inode_num > 0);
    printf("  ✓ Created /testfile.txt (inode %lu)\n", inode_num);
    const char* test_data = "Hello, Filesystem World!";
    size_t data_len = strlen(test_data);
    ssize_t written = fs_write(fs, inode_num, 0, test_data, data_len);
    assert(written == (ssize_t)data_len);
    printf("  ✓ Wrote %zd bytes\n", written);
    // Test 2: Read back and verify
    char buffer[256];
    ssize_t bytes_read = fs_read(fs, inode_num, 0, buffer, sizeof(buffer));
    assert(bytes_read == (ssize_t)data_len);
    assert(memcmp(buffer, test_data, data_len) == 0);
    printf("  ✓ Read back data matches: '%.*s'\n", (int)bytes_read, buffer);
    // Test 3: Verify metadata
    Inode inode;
    read_inode(fs, inode_num, &inode);
    assert(inode.size == data_len);
    assert(inode.blocks == 1);  // One direct block allocated
    assert(inode.direct[0] != 0);
    printf("  ✓ Metadata correct: size=%lu, blocks=%lu\n", inode.size, inode.blocks);
    // Test 4: Partial overwrite
    const char* new_data = "Goodbye";
    fs_write(fs, inode_num, 0, new_data, strlen(new_data));
    bytes_read = fs_read(fs, inode_num, 0, buffer, sizeof(buffer));
    assert(memcmp(buffer, "Goodbye, Filesystem World!", data_len) == 0);
    printf("  ✓ Partial overwrite works correctly\n");
    // Test 5: Write at offset (creates gap)
    fs_write(fs, inode_num, 100, "MIDDLE", 6);
    // Read from the gap (should be zeros)
    bytes_read = fs_read(fs, inode_num, 50, buffer, 10);
    char zeros[10] = {0};
    assert(memcmp(buffer, zeros, 10) == 0);
    printf("  ✓ Gap reads as zeros\n");
    // Read the written data
    bytes_read = fs_read(fs, inode_num, 100, buffer, 6);
    assert(memcmp(buffer, "MIDDLE", 6) == 0);
    printf("  ✓ Data at offset 100 correct\n");
    // Test 6: Block boundary crossing
    char cross_data[8192];  // 2 blocks
    memset(cross_data, 'X', 8192);
    cross_data[0] = 'A';
    cross_data[4095] = 'B';
    cross_data[4096] = 'C';
    cross_data[8191] = 'D';
    written = fs_write(fs, inode_num, 0, cross_data, 8192);
    assert(written == 8192);
    bytes_read = fs_read(fs, inode_num, 0, buffer, 8192);
    assert(memcmp(buffer, cross_data, 8192) == 0);
    printf("  ✓ Block boundary crossing works\n");
    // Clean up
    // (Would call fs_unlink here)
    printf("File read/write tests passed!\n");
}
void test_sparse_files(FileSystem* fs) {
    printf("\nTesting sparse file operations...\n");
    // Create a file
    uint64_t inode_num = fs_create(fs, "/sparse.dat", 0644);
    assert(inode_num > 0);
    // Write at 1MB offset
    const char* data = "Data at 1MB";
    size_t data_len = strlen(data);
    uint64_t offset = 1024 * 1024;  // 1MB
    ssize_t written = fs_write(fs, inode_num, offset, data, data_len);
    assert(written == (ssize_t)data_len);
    printf("  ✓ Wrote %zd bytes at offset %lu\n", written, offset);
    // Verify file size
    Inode inode;
    read_inode(fs, inode_num, &inode);
    assert(inode.size == offset + data_len);
    printf("  ✓ File size is %lu (logical)\n", inode.size);
    // Count actual blocks (should be far fewer than size suggests)
    printf("  ✓ Blocks allocated: %lu (sparse!)\n", inode.blocks);
    // Read from the hole at offset 0
    char buffer[BLOCK_SIZE];
    ssize_t bytes_read = fs_read(fs, inode_num, 0, buffer, BLOCK_SIZE);
    assert(bytes_read == BLOCK_SIZE);
    // Verify all zeros
    int all_zeros = 1;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        if (buffer[i] != 0) {
            all_zeros = 0;
            break;
        }
    }
    assert(all_zeros);
    printf("  ✓ Hole reads as zeros\n");
    // Read the actual data
    bytes_read = fs_read(fs, inode_num, offset, buffer, data_len);
    assert(bytes_read == (ssize_t)data_len);
    assert(memcmp(buffer, data, data_len) == 0);
    printf("  ✓ Data at 1MB offset correct\n");
    printf("Sparse file tests passed!\n");
}
void test_truncation(FileSystem* fs) {
    printf("\nTesting truncation operations...\n");
    // Create and write a file
    uint64_t inode_num = fs_create(fs, "/truncate_test.dat", 0644);
    char data[16384];  // 4 blocks
    memset(data, 'X', sizeof(data));
    fs_write(fs, inode_num, 0, data, sizeof(data));
    Inode inode;
    read_inode(fs, inode_num, &inode);
    assert(inode.size == 16384);
    assert(inode.blocks == 4);
    printf("  ✓ Created 16KB file (4 blocks)\n");
    // Test 1: Truncate down
    fs_truncate(fs, inode_num, 8192);
    read_inode(fs, inode_num, &inode);
    assert(inode.size == 8192);
    assert(inode.blocks == 2);
    printf("  ✓ Truncated to 8KB (2 blocks)\n");
    // Verify data is still correct
    char buffer[8192];
    fs_read(fs, inode_num, 0, buffer, 8192);
    for (int i = 0; i < 8192; i++) {
        assert(buffer[i] == 'X');
    }
    printf("  ✓ Data preserved after truncate\n");
    // Test 2: Truncate to non-aligned size
    fs_truncate(fs, inode_num, 6000);
    read_inode(fs, inode_num, &inode);
    assert(inode.size == 6000);
    assert(inode.blocks == 2);  // Still 2 blocks (partial last block)
    printf("  ✓ Truncated to 6000 bytes (2 blocks, partial last)\n");
    // Verify partial zeroing
    fs_read(fs, inode_num, 6000, buffer, 100);
    for (int i = 0; i < 100; i++) {
        assert(buffer[i] == 0);  // Beyond new size should be zeros
    }
    printf("  ✓ Partial block zeroed correctly\n");
    // Test 3: Grow via truncate
    fs_truncate(fs, inode_num, 100000);
    read_inode(fs, inode_num, &inode);
    assert(inode.size == 100000);
    // Blocks should NOT increase (growing creates hole)
    printf("  ✓ Grown to 100000 bytes (no new blocks allocated)\n");
    // Verify the new region is a hole
    fs_read(fs, inode_num, 50000, buffer, 100);
    for (int i = 0; i < 100; i++) {
        assert(buffer[i] == 0);
    }
    printf("  ✓ Grown region is a hole (reads as zeros)\n");
    printf("Truncation tests passed!\n");
}
void test_large_file(FileSystem* fs) {
    printf("\nTesting large file operations...\n");
    uint64_t inode_num = fs_create(fs, "/largefile.dat", 0644);
    // Write enough data to require single-indirect blocks
    // 12 direct blocks = 48KB
    // We'll write 100KB to ensure we use indirect blocks
    size_t write_size = 100 * 1024;  // 100KB
    char* data = malloc(write_size);
    for (size_t i = 0; i < write_size; i++) {
        data[i] = (char)(i % 256);
    }
    ssize_t written = fs_write(fs, inode_num, 0, data, write_size);
    assert(written == (ssize_t)write_size);
    printf("  ✓ Wrote 100KB (%zd bytes)\n", written);
    Inode inode;
    read_inode(fs, inode_num, &inode);
    assert(inode.size == write_size);
    assert(inode.indirect != 0);  // Should have allocated indirect block
    printf("  ✓ Indirect block allocated: %lu\n", inode.indirect);
    // Read back and verify
    char* verify = malloc(write_size);
    ssize_t bytes_read = fs_read(fs, inode_num, 0, verify, write_size);
    assert(bytes_read == (ssize_t)write_size);
    assert(memcmp(data, verify, write_size) == 0);
    printf("  ✓ Large file read back correctly\n");
    free(data);
    free(verify);
    printf("Large file tests passed!\n");
}
```

![Filesystem Component Atlas](./diagrams/diag-l0-filesystem-map.svg)

## Common Pitfalls
**Not zero-filling partial blocks**: When writing 100 bytes at offset 4050 in an empty block, you must read the block first (it may contain garbage from a previous file), write bytes 4050-4149, then write the block back. Forgetting the read step corrupts data.
**Forgetting to free indirect blocks on truncate**: When truncating below the indirect zone threshold, you must free the indirect block itself, not just the data blocks it points to. Otherwise, you leak blocks.
**Off-by-one in block calculations**: `block_idx = offset / BLOCK_SIZE` gives the correct block for that offset. But `last_block = (size + BLOCK_SIZE - 1) / BLOCK_SIZE` is the first block *beyond* the file. Be careful with inclusive vs. exclusive bounds.
**Not updating ctime on mtime change**: Every time you update mtime, you must also update ctime. This is easy to forget because ctime feels like "metadata change time" but it actually means "inode change time" — and changing mtime is an inode change.
**Reading past end of file**: `fs_read` should return 0 bytes when offset >= size, not an error. Many implementations incorrectly return -1 with errno = EINVAL.
**Write returning short count**: A write may succeed partially (e.g., disk full mid-write). Your code should handle `written < length` gracefully, not assume all-or-nothing.
## What You've Built
At the end of this milestone, you have:
1. **File creation** that allocates inodes and directory entries
2. **File reading** that maps byte offsets to blocks via inode pointers
3. **Hole detection** that synthesizes zeros for unallocated blocks
4. **File writing** with allocation-on-demand for new blocks
5. **Block allocation** that lazily creates indirect blocks as needed
6. **Sparse file support** where gaps remain as holes
7. **Truncation** that grows via holes and shrinks by freeing blocks
8. **Metadata updates** for atime, mtime, ctime on all operations
9. **Append operation** for efficient end-of-file writes
You can now:
```bash
$ ./mkfs disk.img 100
$ ./fs_shell disk.img
fs> create /test.txt
fs> write /test.txt "Hello, World!"
fs> read /test.txt
  Hello, World!
fs> write_at /test.txt 1000000 "Data at 1MB"
fs> stat /test.txt
  Size: 1000007 bytes
  Blocks: 1 (sparse!)
fs> read_at /test.txt 0 100
  (100 zero bytes)
fs> truncate /test.txt 500
fs> stat /test.txt
  Size: 500 bytes
  Blocks: 1
fs> append /test.txt "More data"
fs> read /test.txt
  (first 500 bytes of original, then "More data")
```
## The Knowledge Cascade
The patterns you've learned extend far beyond filesystems:
**Database Extent Allocation**: PostgreSQL allocates storage in 8KB pages, grouped into 1GB segments. When you INSERT into a table, PostgreSQL finds (or allocates) a page with free space, writes the tuple, and updates the free space map. The page-to-offset mapping you implemented is the same pattern at a different scale. Sparse files in databases enable efficient NULL column storage — NULL values don't consume space in the tuple; they're tracked in a separate null bitmap (exactly analogous to your hole detection).
**Memory-Mapped Files (mmap)**: When you `mmap()` a file, the kernel creates a virtual memory region backed by that file. A page fault on a mapped page triggers the exact same `get_file_block()` logic you wrote — the kernel translates the virtual address to a file offset, finds the physical block, and loads it into a page frame. Holes in files become zero-filled pages in memory. The page cache is the bridge between your filesystem and the virtual memory system.
**Log-Structured Merge Trees (LSM)**: The compaction process in LSM-trees (used by RocksDB, LevelDB, Cassandra) is analogous to file truncation. Old SSTable files contain obsolete data mixed with current data; compaction reads the valid data, writes it to a new file, and frees the old files. The "tombstone" markers in LSM are like directory entries pointing to deleted inodes — they signal that data should be ignored until cleaned up.
**Copy-on-Write Snapshots**: ZFS and Btrfs implement atomic snapshots using CoW. When you modify a block, the filesystem allocates a new block, writes the modified data, and updates the pointer. The old block is preserved for snapshots that still reference it. This is a generalization of your `get_or_alloc_block()` — instead of "allocate if null," it's "always allocate, preserving old." The block pointer becomes a versioning mechanism.
**Video Streaming and HTTP Range Requests**: When a video player requests bytes 10000000-10099999 of a 1GB file (an HTTP byte-range request), the server must:
1. Map the byte range to block numbers (your `file_block_idx` calculation)
2. Read only those blocks (not the whole file)
3. Extract the relevant byte range from each block
4. Stream the result
This is exactly your `fs_read()` function, just exposed over a network protocol. Sparse video files with seek tables stored as holes in the data region optimize streaming — the player can jump to any timestamp without loading intermediate data.
<!-- END_MS -->


<!-- MS_ID: filesystem-m5 -->
# FUSE Integration
You've built every layer of a functioning filesystem: block allocation, inode management, directory operations, and file I/O. Your code can create files, write data, traverse directories, and recover from crashes. But there's one glaring gap: **your filesystem exists only as a library.**
To actually use it, you've been running custom test programs that call your API directly. That's not how real filesystems work. Real filesystems mount into the directory hierarchy, accessible to *any* program — `ls`, `cat`, `vim`, your IDE, backup scripts, everything.
This milestone bridges that gap. You're about to plug your filesystem into the operating system using **FUSE** (Filesystem in Userspace), transforming your code from a library into something that looks and acts like a real mounted volume.
## The Fundamental Tension: Kernel Speed vs. Userspace Flexibility
Here's the conflict at the heart of filesystem design:
**The kernel owns the filesystem interface.** Every program interacts with files through system calls (`open`, `read`, `write`, `close`), and those syscalls are handled by the kernel's VFS (Virtual File System) layer. The kernel expects filesystems to be kernel modules — code running in kernel space with direct access to kernel data structures.
**But kernel development is painful.** A kernel panic crashes the entire system. A bug in your filesystem can corrupt any process's memory. Debugging requires rebooting. Deployment requires root access and kernel rebuilds.
FUSE resolves this tension by **splitting the filesystem in two**:
- **Kernel side**: A small, stable kernel module (`fuse.ko`) that registers with VFS and handles the syscall interface
- **Userspace side**: Your daemon process that implements the actual filesystem logic
The kernel module acts as a translator: VFS calls → `/dev/fuse` → your daemon → response → `/dev/fuse` → VFS returns to the calling process.
```
Traditional kernel filesystem:
  ┌─────────────────────────────────────────────┐
  │              KERNEL SPACE                   │
  │  ┌─────────┐    ┌─────────────────────┐    │
  │  │   VFS   │───▶│  Filesystem Module  │    │
  │  └─────────┘    └─────────────────────┘    │
  └─────────────────────────────────────────────┘
        ▲
        │ syscalls
        ▼
  ┌─────────────────────────────────────────────┐
  │              USER SPACE                     │
  │         Application (cat, ls, etc.)         │
  └─────────────────────────────────────────────┘
FUSE filesystem:
  ┌─────────────────────────────────────────────┐
  │              KERNEL SPACE                   │
  │  ┌─────────┐    ┌─────────────────────┐    │
  │  │   VFS   │───▶│    FUSE Module      │    │
  │  └─────────┘    └──────────┬──────────┘    │
  └────────────────────────────┼────────────────┘
                               │ /dev/fuse
  ┌────────────────────────────┼────────────────┐
  │              USER SPACE    ▼                │
  │                    ┌───────────────────┐    │
  │                    │  Your FUSE Daemon │    │
  │                    │  (this code)      │    │
  │                    └───────────────────┘    │
  └─────────────────────────────────────────────┘
```
**The cost**: Every file operation now crosses the kernel-userspace boundary twice (request and response). Each crossing costs ~1000 CPU cycles for the context switch. A simple `ls` that triggers 50 `getattr` calls now has 50 context switches.
**The benefit**: Your filesystem code runs as a normal userspace process. It can crash without taking down the kernel. You can debug it with `gdb`. You can write it in any language. You can run multiple different filesystems simultaneously without kernel modifications.
## The Revelation: Paths, Not Inodes
Here's what surprises most developers about FUSE: **FUSE callbacks receive paths, not inode numbers.**
When you run `cat /mnt/myfs/a/b/c.txt`, the kernel's VFS layer would normally resolve the path internally, translating each component to an inode number through its dentry cache. But FUSE bypasses this — it passes the *entire path string* to your daemon for each operation.
Your daemon receives this sequence:
1. `getattr("/a/b/c.txt")` — Is this a valid path? What are its permissions?
2. `open("/a/b/c.txt", O_RDONLY)` — Prepare to read this file
3. `read("/a/b/c.txt", offset=0, size=4096)` — Give me the first 4KB
4. `read("/a/b/c.txt", offset=4096, size=4096)` — Give me the next 4KB
5. `release("/a/b/c.txt")` — Done with this file
Each callback must **resolve the path from scratch**. Your `getattr` implementation can't assume that `open` was called first, or that previous calls succeeded. FUSE provides no persistent state between callbacks — you must rebuild context on every operation.
This seems inefficient, and it is. But it's also **flexible**:
- Your filesystem can implement path-based semantics that don't map to inodes at all (e.g., a network filesystem where paths are query parameters)
- You can present a virtual view of data that doesn't exist on disk (e.g., `/proc`-like interfaces)
- You can translate paths on the fly (e.g., case-insensitive filesystems, encrypted filenames)
For your inode-based filesystem, you'll need to call your `path_resolve()` function at the start of every callback. This is where caching becomes critical — a warm dentry cache makes path resolution nearly free.
## The `getattr` Crucible: Performance Meets Correctness
Of all FUSE callbacks, `getattr` is the most frequently called. It's invoked for:
- Every file access (to check permissions)
- Every `ls` (to display file sizes and timestamps)
- Every tab-completion in a shell
- Every `stat()` call from any program
A single `ls -la` in a directory with 100 files triggers 101 `getattr` calls (one for the directory, one for each file). Your `getattr` implementation must be both **fast** and **correct** — any bug here breaks everything.
```c
// FUSE wants a struct stat filled with file attributes
// This is the 64-bit version used by modern FUSE
static int fs_getattr(const char* path, struct stat* stbuf) {
    memset(stbuf, 0, sizeof(struct stat));
    // Resolve the path to an inode
    uint64_t inode_num = path_resolve(fs_global, path);
    if (inode_num == 0) {
        return -ENOENT;  // No such file or directory
    }
    // Read the inode
    Inode inode;
    if (read_inode(fs_global, inode_num, &inode) < 0) {
        return -EIO;  // I/O error
    }
    // Fill in the stat structure
    stbuf->st_ino = inode_num;
    stbuf->st_mode = inode.mode;
    stbuf->st_nlink = inode.link_count;
    stbuf->st_uid = inode.uid;
    stbuf->st_gid = inode.gid;
    stbuf->st_size = inode.size;
    stbuf->st_blocks = inode.blocks;  // In 512-byte units
    stbuf->st_atime = inode.atime;
    stbuf->st_mtime = inode.mtime;
    stbuf->st_ctime = inode.ctime;
    // Block size for I/O operations
    stbuf->st_blksize = BLOCK_SIZE;
    return 0;  // Success
}
```
**Hardware Soul Check**: Every `getattr` that reaches your filesystem triggers:
1. Path resolution: 1-5 directory block reads (one per path component)
2. Inode read: 1 block read
3. Stat structure fill: Pure CPU, nanoseconds
That's potentially 6 disk reads for a single `ls -la` entry. With a cold cache, listing a directory of 100 files could require 600 disk reads. At 10ms per HDD read, that's 6 seconds. This is why real filesystems cache aggressively — the Linux dentry cache keeps resolved paths in memory, turning repeated `getattr` calls into simple hash lookups.
## FUSE Callback Architecture: The Complete Interface
FUSE defines a large callback interface, but you only need to implement the operations your filesystem supports. Here's the complete set for a read-write filesystem:
```c
// FUSE operations structure
static struct fuse_operations fs_oper = {
    .getattr     = fs_getattr,      // Get file attributes
    .readlink    = fs_readlink,     // Read symbolic link target
    .mknod       = fs_mknod,        // Create special file
    .mkdir       = fs_mkdir,        // Create directory
    .unlink      = fs_unlink,       // Remove file
    .rmdir       = fs_rmdir,        // Remove directory
    .symlink     = fs_symlink,      // Create symbolic link
    .rename      = fs_rename,       // Rename/move file
    .link        = fs_link,         // Create hard link
    .chmod       = fs_chmod,        // Change permissions
    .chown       = fs_chown,        // Change ownership
    .truncate    = fs_truncate,     // Change file size
    .open        = fs_open,         // Open file
    .read        = fs_read,         // Read from file
    .write       = fs_write,        // Write to file
    .statfs      = fs_statfs,       // Get filesystem statistics
    .flush       = fs_flush,        // Flush buffered data
    .release     = fs_release,      // Release file (close)
    .fsync       = fs_fsync,        // Sync file to disk
    .opendir     = fs_opendir,      // Open directory for reading
    .readdir     = fs_readdir,      // Read directory entries
    .releasedir  = fs_releasedir,   // Release directory
    .init        = fs_init,         // Initialize filesystem
    .destroy     = fs_destroy,      // Cleanup filesystem
    .create      = fs_create,       // Create and open file
    .utimens     = fs_utimens,      // Change timestamps
};
```
Let's implement each critical callback.
### File Creation and Opening
FUSE has two ways to create files: `mknod` + `open` (separate operations) or `create` (atomic create-and-open). Modern programs use `create` for the common case:
```c
// Create a file and open it (atomic operation)
static int fs_create(const char* path, mode_t mode, struct fuse_file_info* fi) {
    // Extract parent directory and filename
    char name[MAX_NAME_LEN + 1];
    uint64_t parent_inode = path_resolve_parent(fs_global, path, name);
    if (parent_inode == 0) {
        return -ENOENT;
    }
    // Check if file already exists
    uint64_t existing = dir_lookup(fs_global, parent_inode, name);
    if (existing != 0) {
        // If O_EXCL is set, this is an error
        if (fi->flags & O_EXCL) {
            return -EEXIST;
        }
        // Otherwise, just open the existing file
        fi->fh = existing;  // Store inode number in file handle
        return 0;
    }
    // Create the file
    uint64_t new_inode = fs_create_file(fs_global, path, mode & 07777);
    if (new_inode == 0) {
        return -ENOSPC;  // No space or no inodes
    }
    // Store the inode number in the file handle for later operations
    fi->fh = new_inode;
    return 0;
}
// Open an existing file
static int fs_open(const char* path, struct fuse_file_info* fi) {
    uint64_t inode_num = path_resolve(fs_global, path);
    if (inode_num == 0) {
        return -ENOENT;
    }
    Inode inode;
    if (read_inode(fs_global, inode_num, &inode) < 0) {
        return -EIO;
    }
    // Check permissions (simplified - would check against process uid/gid)
    int access_mode = fi->flags & O_ACCMODE;
    if (access_mode != O_RDONLY && (inode.mode & 0222) == 0) {
        return -EACCES;  // Write permission denied
    }
    // Store inode in file handle
    fi->fh = inode_num;
    return 0;
}
```
**The `fuse_file_info` Structure**: The `fi` parameter is FUSE's way of letting you maintain state between `open` and `release`. The `fh` field (file handle) is an arbitrary 64-bit value you can set in `open`/`create` and read back in `read`/`write`/`release`. This avoids repeated path resolution for the same file.
### Reading and Writing
```c
// Read from an open file
static int fs_read(const char* path, char* buf, size_t size, off_t offset,
                   struct fuse_file_info* fi) {
    // Use the file handle if available, otherwise resolve path
    uint64_t inode_num = fi->fh;
    if (inode_num == 0) {
        inode_num = path_resolve(fs_global, path);
        if (inode_num == 0) {
            return -ENOENT;
        }
    }
    ssize_t bytes_read = fs_read_data(fs_global, inode_num, offset, buf, size);
    if (bytes_read < 0) {
        return -EIO;
    }
    return (int)bytes_read;
}
// Write to an open file
static int fs_write(const char* path, const char* buf, size_t size, off_t offset,
                    struct fuse_file_info* fi) {
    uint64_t inode_num = fi->fh;
    if (inode_num == 0) {
        inode_num = path_resolve(fs_global, path);
        if (inode_num == 0) {
            return -ENOENT;
        }
    }
    ssize_t bytes_written = fs_write_data(fs_global, inode_num, offset, buf, size);
    if (bytes_written < 0) {
        return -EIO;
    }
    return (int)bytes_written;
}
// Release (close) a file
static int fs_release(const char* path, struct fuse_file_info* fi) {
    // Nothing special to do - we don't track open files in this design
    // A more sophisticated filesystem would update open count, sync dirty data, etc.
    return 0;
}
```
### Directory Operations
```c
// Create a directory
static int fs_mkdir(const char* path, mode_t mode) {
    uint64_t inode_num = fs_mkdir_dir(fs_global, path);
    if (inode_num == 0) {
        // Check what went wrong
        if (errno == EEXIST) return -EEXIST;
        if (errno == ENOENT) return -ENOENT;
        return -ENOSPC;
    }
    // Set permissions
    Inode inode;
    if (read_inode(fs_global, inode_num, &inode) == 0) {
        inode.mode = S_IFDIR | (mode & 07777);
        write_inode(fs_global, inode_num, &inode);
    }
    return 0;
}
// Read directory entries
// FUSE calls this repeatedly until it returns 0
static int fs_readdir(const char* path, void* buf, fuse_fill_dir_t filler,
                      off_t offset, struct fuse_file_info* fi) {
    uint64_t inode_num = path_resolve(fs_global, path);
    if (inode_num == 0) {
        return -ENOENT;
    }
    Inode inode;
    if (read_inode(fs_global, inode_num, &inode) < 0) {
        return -EIO;
    }
    if (!(inode.mode & S_IFDIR)) {
        return -ENOTDIR;
    }
    // Read directory entries
    // The filler function adds entries to the buffer
    // filler(buf, name, stat, off)
    // Read each block of the directory
    char block_buf[BLOCK_SIZE];
    DirEntry* entries = (DirEntry*)block_buf;
    uint64_t num_blocks = (inode.size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (uint64_t block_idx = 0; block_idx < num_blocks; block_idx++) {
        uint64_t phys_block = get_file_block(fs_global, &inode, block_idx);
        if (phys_block == 0) continue;
        if (read_block(fs_global->dev, phys_block, block_buf) < 0) {
            return -EIO;
        }
        for (int i = 0; i < ENTRIES_PER_BLOCK; i++) {
            if (entries[i].inode != 0) {
                // Add this entry to the directory listing
                struct stat st;
                memset(&st, 0, sizeof(st));
                st.st_ino = entries[i].inode;
                // Set file type in mode
                if (entries[i].file_type == DT_DIR) {
                    st.st_mode = S_IFDIR;
                } else if (entries[i].file_type == DT_LNK) {
                    st.st_mode = S_IFLNK;
                } else {
                    st.st_mode = S_IFREG;
                }
                // filler returns 1 if buffer is full
                if (filler(buf, entries[i].name, &st, 0) != 0) {
                    return 0;  // Buffer full, but not an error
                }
            }
        }
    }
    return 0;
}
// Remove a directory
static int fs_rmdir(const char* path) {
    int result = fs_rmdir_dir(fs_global, path);
    if (result < 0) {
        if (errno == ENOTEMPTY) return -ENOTEMPTY;
        if (errno == ENOENT) return -ENOENT;
        if (errno == ENOTDIR) return -ENOTDIR;
        return -EIO;
    }
    return 0;
}
```
### File Deletion and Renaming
```c
// Remove a file (unlink)
static int fs_unlink(const char* path) {
    // Resolve parent and get filename
    char name[MAX_NAME_LEN + 1];
    uint64_t parent_inode = path_resolve_parent(fs_global, path, name);
    if (parent_inode == 0) {
        return -ENOENT;
    }
    // Check if file exists
    uint64_t file_inode = dir_lookup(fs_global, parent_inode, name);
    if (file_inode == 0) {
        return -ENOENT;
    }
    // Remove the directory entry
    if (dir_remove_entry(fs_global, parent_inode, name) < 0) {
        return -EIO;
    }
    return 0;
}
// Rename/move a file
static int fs_rename(const char* oldpath, const char* newpath) {
    // Resolve old path
    char old_name[MAX_NAME_LEN + 1];
    uint64_t old_parent = path_resolve_parent(fs_global, oldpath, old_name);
    if (old_parent == 0) {
        return -ENOENT;
    }
    uint64_t old_inode = dir_lookup(fs_global, old_parent, old_name);
    if (old_inode == 0) {
        return -ENOENT;
    }
    // Resolve new path
    char new_name[MAX_NAME_LEN + 1];
    uint64_t new_parent = path_resolve_parent(fs_global, newpath, new_name);
    if (new_parent == 0) {
        return -ENOENT;
    }
    // Check if target exists
    uint64_t existing = dir_lookup(fs_global, new_parent, new_name);
    if (existing != 0) {
        // For simplicity, we don't support overwriting
        // A real implementation would check if it's the same file,
        // or if it's an empty directory, etc.
        return -EEXIST;
    }
    // Perform the rename
    if (fs_rename_file(fs_global, oldpath, newpath) < 0) {
        return -EIO;
    }
    return 0;
}
```
### Metadata Operations
```c
// Change file permissions
static int fs_chmod(const char* path, mode_t mode) {
    uint64_t inode_num = path_resolve(fs_global, path);
    if (inode_num == 0) {
        return -ENOENT;
    }
    Inode inode;
    if (read_inode(fs_global, inode_num, &inode) < 0) {
        return -EIO;
    }
    // Preserve file type bits, update permission bits
    inode.mode = (inode.mode & S_IFMT) | (mode & 07777);
    inode.ctime = (uint64_t)time(NULL);
    if (write_inode(fs_global, inode_num, &inode) < 0) {
        return -EIO;
    }
    return 0;
}
// Change file ownership
static int fs_chown(const char* path, uid_t uid, gid_t gid) {
    uint64_t inode_num = path_resolve(fs_global, path);
    if (inode_num == 0) {
        return -ENOENT;
    }
    Inode inode;
    if (read_inode(fs_global, inode_num, &inode) < 0) {
        return -EIO;
    }
    // Only root can change ownership (simplified check)
    // A real implementation would check process capabilities
    if (uid != (uid_t)-1) {
        inode.uid = uid;
    }
    if (gid != (gid_t)-1) {
        inode.gid = gid;
    }
    inode.ctime = (uint64_t)time(NULL);
    if (write_inode(fs_global, inode_num, &inode) < 0) {
        return -EIO;
    }
    return 0;
}
// Truncate a file
static int fs_truncate(const char* path, off_t size) {
    uint64_t inode_num = path_resolve(fs_global, path);
    if (inode_num == 0) {
        return -ENOENT;
    }
    if (fs_truncate_file(fs_global, inode_num, size) < 0) {
        return -EIO;
    }
    return 0;
}
// Change file timestamps
static int fs_utimens(const char* path, const struct timespec ts[2]) {
    uint64_t inode_num = path_resolve(fs_global, path);
    if (inode_num == 0) {
        return -ENOENT;
    }
    Inode inode;
    if (read_inode(fs_global, inode_num, &inode) < 0) {
        return -EIO;
    }
    // ts[0] is atime, ts[1] is mtime
    if (ts[0].tv_nsec != UTIME_NOW && ts[0].tv_nsec != UTIME_OMIT) {
        inode.atime = ts[0].tv_sec;
    }
    if (ts[1].tv_nsec != UTIME_NOW && ts[1].tv_nsec != UTIME_OMIT) {
        inode.mtime = ts[1].tv_sec;
        inode.ctime = (uint64_t)time(NULL);  // ctime updated on mtime change
    }
    if (write_inode(fs_global, inode_num, &inode) < 0) {
        return -EIO;
    }
    return 0;
}
```
### Filesystem Statistics
```c
// Get filesystem statistics (df uses this)
static int fs_statfs(const char* path, struct statvfs* stbuf) {
    memset(stbuf, 0, sizeof(struct statvfs));
    // Filesystem block size
    stbuf->f_bsize = BLOCK_SIZE;
    stbuf->f_frsize = BLOCK_SIZE;  // Fragment size (same as block size)
    // Total blocks in filesystem
    stbuf->f_blocks = fs_global->sb->total_blocks - fs_global->sb->data_start;
    // Free blocks
    stbuf->f_bfree = fs_global->sb->free_blocks;
    stbuf->f_bavail = fs_global->sb->free_blocks;  // Free for non-root
    // Total inodes
    stbuf->f_files = fs_global->sb->total_inodes;
    // Free inodes
    stbuf->f_ffree = fs_global->sb->free_inodes;
    stbuf->f_favail = fs_global->sb->free_inodes;
    // Filesystem ID (could use device number)
    stbuf->f_fsid = 0;
    // Maximum filename length
    stbuf->f_namemax = MAX_NAME_LEN;
    return 0;
}
```
### Initialization and Cleanup
```c
// Initialize the filesystem
static void* fs_init(struct fuse_conn_info* conn) {
    // conn contains connection information
    // We can configure capabilities here
    // Enable big writes (more efficient for large files)
    conn->want |= FUSE_CAP_BIG_WRITES;
    // Return our global filesystem structure
    return fs_global;
}
// Cleanup when filesystem is unmounted
static void fs_destroy(void* private_data) {
    // Flush any cached data
    if (fs_global && fs_global->sb) {
        fs_global->sb->write_time = (uint64_t)time(NULL);
        write_block(fs_global->dev, 0, fs_global->sb);
    }
    // Sync the backing file
    if (fs_global && fs_global->dev) {
        fsync(fs_global->dev->fd);
    }
    // The filesystem structure will be freed by main
}
```
## Concurrency: The Locking Challenge
FUSE is inherently multithreaded by default. Multiple threads handle concurrent requests — one thread processing `readdir` while another handles `write`, another `mkdir`, etc. Without proper synchronization, your filesystem will corrupt data.

![Filesystem Layer Stack](./diagrams/diag-l1-layer-stack.svg)

The critical sections that need protection:
1. **Block allocation**: Two threads could allocate the same block
2. **Inode updates**: Two threads could modify the same inode
3. **Directory modifications**: Adding/removing entries must be atomic
4. **Bitmap updates**: The bitmap must stay consistent with actual allocations
```c
#include <pthread.h>
// Global lock for the entire filesystem
// (A production filesystem would use finer-grained locking)
static pthread_mutex_t fs_lock = PTHREAD_MUTEX_INITIALIZER;
// Helper macros for locking
#define FS_LOCK()   pthread_mutex_lock(&fs_lock)
#define FS_UNLOCK() pthread_mutex_unlock(&fs_lock)
// Thread-safe wrapper for block allocation
uint64_t alloc_block_safe(FileSystem* fs) {
    FS_LOCK();
    uint64_t block = alloc_block(fs);
    FS_UNLOCK();
    return block;
}
// Thread-safe wrapper for inode operations
int read_inode_safe(FileSystem* fs, uint64_t inode_num, Inode* inode) {
    FS_LOCK();
    int result = read_inode(fs, inode_num, inode);
    FS_UNLOCK();
    return result;
}
int write_inode_safe(FileSystem* fs, uint64_t inode_num, const Inode* inode) {
    FS_LOCK();
    int result = write_inode(fs, inode_num, inode);
    FS_UNLOCK();
    return result;
}
```
**A Finer-Grained Approach**: A single global lock is simple but limits parallelism. A better approach uses per-inode locks:
```c
typedef struct {
    pthread_mutex_t inode_locks[MAX_INODES];
    pthread_mutex_t bitmap_lock;
    pthread_mutex_t superblock_lock;
} LockManager;
// Initialize locks
void lock_manager_init(LockManager* lm) {
    for (int i = 0; i < MAX_INODES; i++) {
        pthread_mutex_init(&lm->inode_locks[i], NULL);
    }
    pthread_mutex_init(&lm->bitmap_lock, NULL);
    pthread_mutex_init(&lm->superblock_lock, NULL);
}
// Lock hierarchy to prevent deadlocks:
// 1. bitmap_lock (for allocation)
// 2. inode_lock (for metadata)
// 3. Never hold multiple inode locks simultaneously
//    (or always lock in inode-number order)
```
**Deadlock Prevention Rules**:
1. **Lock ordering**: Always acquire locks in the same order (e.g., bitmap before inode)
2. **No recursive locking**: A thread holding a lock should not try to acquire it again
3. **Lock timeout**: Consider using `pthread_mutex_timedlock` to detect deadlocks
4. **Single-operation semantics**: Don't hold locks across I/O operations
### Thread-Safe FUSE Callbacks
```c
// Thread-safe getattr
static int fs_getattr_safe(const char* path, struct stat* stbuf) {
    FS_LOCK();
    int result = fs_getattr(path, stbuf);
    FS_UNLOCK();
    return result;
}
// Thread-safe create
static int fs_create_safe(const char* path, mode_t mode, struct fuse_file_info* fi) {
    FS_LOCK();
    int result = fs_create(path, mode, fi);
    FS_UNLOCK();
    return result;
}
// Thread-safe write
static int fs_write_safe(const char* path, const char* buf, size_t size,
                         off_t offset, struct fuse_file_info* fi) {
    FS_LOCK();
    int result = fs_write(path, buf, size, offset, fi);
    FS_UNLOCK();
    return result;
}
// Thread-safe readdir
static int fs_readdir_safe(const char* path, void* buf, fuse_fill_dir_t filler,
                           off_t offset, struct fuse_file_info* fi) {
    FS_LOCK();
    int result = fs_readdir(path, buf, filler, offset, fi);
    FS_UNLOCK();
    return result;
}
```
**Hardware Soul Check**: A single global lock means that on a multi-core system, only one CPU can execute filesystem code at a time. With 8 cores, you're using at most 12.5% of your CPU capacity for filesystem operations. This is acceptable for a learning project but would be a serious bottleneck in production. The Linux kernel uses RCU (Read-Copy-Update) and per-CPU data structures to achieve massive parallelism in its filesystem code.
## The Main Program: Putting It All Together
```c
#include <fuse3/fuse.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
// Global filesystem instance (accessible to all callbacks)
FileSystem* fs_global = NULL;
// Print usage information
static void print_usage(const char* progname) {
    fprintf(stderr, "Usage: %s [FUSE options] <disk_image> <mount_point>\n", progname);
    fprintf(stderr, "\nFUSE options:\n");
    fprintf(stderr, "  -f          Foreground (don't daemonize)\n");
    fprintf(stderr, "  -d          Debug mode (implies -f)\n");
    fprintf(stderr, "  -s          Single-threaded\n");
    fprintf(stderr, "  -o allow_other  Allow other users to access\n");
    fprintf(stderr, "\nExample:\n");
    fprintf(stderr, "  %s -f disk.img /mnt/myfs\n", progname);
}
// Parse non-FUSE arguments (our disk image)
static int parse_args(struct fuse_args* args, const char** disk_image) {
    for (int i = 0; i < args->argc; i++) {
        if (args->argv[i][0] != '-') {
            // Non-option argument - assume it's our disk image
            *disk_image = args->argv[i];
            // Remove it from args so FUSE doesn't try to parse it
            if (i < args->argc - 1) {
                memmove(&args->argv[i], &args->argv[i + 1],
                        (args->argc - i - 1) * sizeof(char*));
            }
            args->argc--;
            return 0;
        }
    }
    return -1;  // No disk image found
}
int main(int argc, char* argv[]) {
    const char* disk_image = NULL;
    // Set up FUSE argument parsing
    struct fuse_args args = FUSE_ARGS_INIT(argc, argv);
    // Parse our arguments
    if (parse_args(&args, &disk_image) < 0) {
        print_usage(argv[0]);
        return 1;
    }
    if (disk_image == NULL) {
        print_usage(argv[0]);
        return 1;
    }
    // Open the filesystem
    printf("Opening filesystem: %s\n", disk_image);
    fs_global = fs_open(disk_image);
    if (fs_global == NULL) {
        fprintf(stderr, "Failed to open filesystem: %s\n", strerror(errno));
        return 1;
    }
    // Verify the superblock
    if (fs_global->sb->magic != FS_MAGIC) {
        fprintf(stderr, "Invalid filesystem (bad magic number)\n");
        fs_close(fs_global);
        return 1;
    }
    printf("Filesystem mounted:\n");
    printf("  Total blocks: %lu\n", fs_global->sb->total_blocks);
    printf("  Free blocks:  %lu\n", fs_global->sb->free_blocks);
    printf("  Free inodes:  %lu\n", fs_global->sb->free_inodes);
    // Initialize locks
    pthread_mutex_init(&fs_lock, NULL);
    // Set up FUSE operations
    static struct fuse_operations fs_oper = {
        .getattr     = fs_getattr_safe,
        .readlink    = fs_readlink,
        .mkdir       = fs_mkdir_safe,
        .unlink      = fs_unlink_safe,
        .rmdir       = fs_rmdir_safe,
        .symlink     = fs_symlink,
        .rename      = fs_rename_safe,
        .link        = fs_link,
        .chmod       = fs_chmod_safe,
        .chown       = fs_chown_safe,
        .truncate    = fs_truncate_safe,
        .open        = fs_open_safe,
        .read        = fs_read_safe,
        .write       = fs_write_safe,
        .statfs      = fs_statfs_safe,
        .release     = fs_release,
        .fsync       = fs_fsync,
        .opendir     = fs_opendir,
        .readdir     = fs_readdir_safe,
        .releasedir  = fs_releasedir,
        .init        = fs_init,
        .destroy     = fs_destroy,
        .create      = fs_create_safe,
        .utimens     = fs_utimens_safe,
    };
    // Run FUSE main loop
    printf("Starting FUSE filesystem...\n");
    int ret = fuse_main(args.argc, args.argv, &fs_oper, NULL);
    // Cleanup
    fuse_opt_free_args(&args);
    pthread_mutex_destroy(&fs_lock);
    // fs_close is called by fs_destroy
    return ret;
}
```
## Mounting and Testing
Now you can mount your filesystem and test it with real tools:
```bash
# Compile the FUSE daemon
$ gcc -o fs_fuse fs_fuse.c fs_ops.c block.c inode.c dir.c -lfuse3 -lpthread
# Create a test filesystem image
$ ./mkfs test.img 100
Filesystem created successfully:
  Total blocks: 25600 (100.00 MB)
  ...
# Create a mount point
$ mkdir -p /tmp/myfs
# Mount the filesystem (foreground mode for debugging)
$ ./fs_fuse -f test.img /tmp/myfs
# In another terminal, test it:
$ ls -la /tmp/myfs
total 4
drwxr-xr-x 2 root root 4096 Mar  4 10:00 .
drwxr-xr-x 3 root root 4096 Mar  4 10:00 ..
$ mkdir /tmp/myfs/testdir
$ echo "Hello, FUSE!" > /tmp/myfs/testdir/hello.txt
$ cat /tmp/myfs/testdir/hello.txt
Hello, FUSE!
$ ls -la /tmp/myfs/testdir
total 2
drwxr-xr-x 2 root root root 4096 Mar  4 10:00 .
drwxr-xr-x 3 root root root 4096 Mar  4 10:00 ..
-rw-r--r-- 1 root root 13 Mar  4 10:00 hello.txt
$ cp /etc/passwd /tmp/myfs/
$ diff /etc/passwd /tmp/myfs/passwd
# No output = files are identical
$ df -h /tmp/myfs
Filesystem      Size  Used Avail Use% Mounted on
./fs_fuse       100M   32K  100M   1% /tmp/myfs
```
### Common Issues and Solutions
**Permission Denied**:
```bash
# If you get permission errors as non-root:
$ ./fs_fuse -o allow_other test.img /tmp/myfs
# Note: This requires user_allow_other in /etc/fuse.conf
```
**Transport Endpoint Not Connected**:
```bash
# If the filesystem crashed or was killed improperly:
$ fusermount -u /tmp/myfs
# Or force unmount:
$ sudo umount -l /tmp/myfs
```
**Debug Mode**:
```bash
# Run with debug output to see all FUSE calls:
$ ./fs_fuse -d test.img /tmp/myfs
# This shows every callback invocation with arguments
```
## Testing with Real Workloads
A filesystem isn't tested until real programs use it:
```bash
# Test with a real editor
$ vim /tmp/myfs/test.txt
# Type some content, save, quit
# Test with a compiler
$ cp -r /path/to/small/project /tmp/myfs/
$ cd /tmp/myfs/project
$ make
# Should compile successfully
# Test with tar
$ tar czf /tmp/myfs/backup.tar.gz /tmp/myfs/testdir
$ tar tzf /tmp/myfs/backup.tar.gz
# Should list the contents
# Test concurrent access
$ (for i in {1..100}; do echo "file$i" > /tmp/myfs/concurrent_$i.txt; done) &
$ (for i in {101..200}; do echo "file$i" > /tmp/myfs/concurrent_$i.txt; done) &
$ wait
$ ls /tmp/myfs/concurrent_*.txt | wc -l
200  # All files should exist
# Stress test
$ stress-ng --fs 4 --timeout 60s --fs-ops 10000
```
## Performance Analysis
Let's measure the overhead of FUSE compared to a native filesystem:
```bash
# Sequential write test
$ dd if=/dev/zero of=/tmp/myfs/testfile bs=1M count=10
10+0 records in
10+0 records out
10485760 bytes (10 MB) copied, 0.523 s, 20.0 MB/s
# Same test on native ext4
$ dd if=/dev/zero of=/tmp/testfile bs=1M count=10
10+0 records out
10485760 bytes (10 MB) copied, 0.089 s, 117.8 MB/s
# FUSE overhead: ~6x slower for large sequential writes
```
**Where does the overhead come from?**
1. **Context switches**: 2 per operation (kernel→userspace→kernel)
2. **Path resolution**: FUSE re-resolves paths on every call
3. **No page cache integration**: FUSE has its own caching, not shared with the kernel
4. **Copy overhead**: Data copied from kernel to userspace and back
**Optimization Strategies**:
1. **Enable FUSE caching**: Use `-o entry_timeout=T` and `-o attr_timeout=T`
2. **Big writes**: Use `-o big_writes` to allow larger write chunks
3. **Direct I/O**: Bypass caching for workloads that don't benefit
4. **Writeback caching**: Use `-o writeback_cache` to batch writes
```c
// In fs_init, enable FUSE capabilities:
static void* fs_init(struct fuse_conn_info* conn) {
    // Enable capabilities for better performance
    conn->want |= FUSE_CAP_BIG_WRITES;
    conn->want |= FUSE_CAP_ASYNC_READ;
    conn->want |= FUSE_CAP_WRITEBACK_CACHE;
    // Set maximum write size
    conn->max_write = 128 * 1024;  // 128KB chunks
    return fs_global;
}
```

![Filesystem Component Atlas](./diagrams/diag-l0-filesystem-map.svg)

## Common Pitfalls
**Forgetting to return negative error codes**: FUSE expects negative errno values (e.g., `-ENOENT`), not positive return codes. Returning `ENOENT` instead of `-ENOENT` will be interpreted as success with return value 2.
**Not handling the file handle correctly**: The `fi->fh` field must be set in `open`/`create` and used in `read`/`write`/`release`. Forgetting to set it means every I/O operation will re-resolve the path.
**Returning 0 from readdir without filling buffer**: The `filler` function must be called for each entry. If you return 0 without calling filler, the directory appears empty.
**Incorrect stat structure initialization**: The `struct stat` must be zeroed with `memset` before filling. Uninitialized fields contain garbage that can crash programs.
**Deadlock from nested locking**: If `getattr` is called while holding a lock (e.g., during `readdir`), and `getattr` tries to acquire the same lock, you deadlock. Use recursive mutexes or restructure code to avoid this.
**Not syncing on unmount**: Data written just before unmount may be lost if not flushed. Always sync in `destroy` or `release`.
## What You've Built
At the end of this milestone, you have:
1. **A complete FUSE daemon** that mounts your filesystem as a real volume
2. **All standard VFS callbacks** implemented with proper error codes
3. **Thread-safe operations** with mutex protection for concurrent access
4. **Integration with Unix tools** — `ls`, `cat`, `cp`, `vim`, compilers all work
5. **Proper mount/unmount lifecycle** with data flushing
6. **Performance awareness** — understanding FUSE overhead and optimization
You can now:
```bash
$ ./mkfs disk.img 500
$ mkdir /mnt/myfs
$ ./fs_fuse disk.img /mnt/myfs &
$ cd /mnt/myfs
$ mkdir projects
$ cd projects
$ git clone https://github.com/some/repo.git
$ cd repo
$ make
# Your filesystem is now running real workloads!
```
## The Knowledge Cascade
The patterns you've learned extend far beyond FUSE:
**Network Filesystems (NFS, SMB, 9P)**: These use the same callback architecture as FUSE, just over a network instead of a character device. The VFS layer calls into the NFS client module, which sends RPCs to the server, receives responses, and returns to VFS. The server-side NFS daemon implements callbacks very similar to your FUSE callbacks — `getattr`, `read`, `write`, `readdir`. When you understand FUSE, you understand the core of network filesystems.
**Container Filesystems (overlayfs, UnionFS)**: These layer multiple filesystems with copy-on-write semantics. When you write to a file in an overlay, the filesystem copies it from the read-only lower layer to the writable upper layer, then modifies the copy. FUSE can implement similar layering in userspace — the `write` callback checks if the file exists in the upper layer, copies if not, then writes. Docker's storage drivers are essentially sophisticated FUSE-like callbacks.
**Cloud Storage Gateways (s3fs, goofys)**: These mount cloud object stores (S3, GCS) as local filesystems. The FUSE callbacks translate file operations to HTTP API calls:
- `getattr` → `HEAD /bucket/path`
- `read` → `GET /bucket/path?range=bytes=0-4095`
- `write` → `PUT /bucket/path`
- `readdir` → `GET /bucket?list-type=2&prefix=path/`
The latency difference is dramatic (S3 ~50ms vs local disk ~0.1ms), but the API pattern is identical to what you implemented.
**Version Control Filesystems (GitFS)**: These mount a git repository as a filesystem where branches are directories and commits are snapshots. `ls /mnt/gitfs/branches/` shows all branches. `cat /mnt/gitfs/tags/v1.0/src/main.c` shows the file as it existed at that tag. The FUSE callbacks translate paths to git operations:
- `readdir("/branches")` → `git branch --list`
- `read("/commits/abc123/file.c")` → `git show abc123:file.c`
This is a creative reinterpretation of "file" — the filesystem is a view into git's object database.
**Security Sandboxes (sandboxfs)**: These use FUSE to present a filtered/composed view of the filesystem to sandboxed processes. The `getattr` and `readdir` callbacks check a manifest and only expose allowed paths. Writes can be redirected to a scratch directory. This is how Bazel and other build systems isolate builds — the process sees only what the sandbox allows.
In the next milestone, you'll implement write-ahead journaling to ensure your filesystem survives crashes without corruption. The FUSE integration you built today will let you test crash recovery with real workloads — you can `kill -9` the daemon mid-write, restart it, and verify the journal replays correctly.
---
[[CRITERIA_JSON: {"milestone_id": "filesystem-m5", "criteria": ["FUSE daemon mounts the filesystem image as a regular mount point accessible to all OS programs via standard syscalls", "All required FUSE callbacks implemented with correct semantics: getattr, readdir, lookup (via opendir/readdir), create, open, read, write, release, mkdir, rmdir, unlink, rename, truncate, chmod, utimens, statfs", "Standard Unix tools work correctly: ls -la shows directory contents with accurate sizes/permissions/timestamps; cat reads file contents; cp copies files into the mount; echo redirection creates and writes files; vim and other editors can edit files", "Thread-safe concurrent access: multiple processes can simultaneously read/write/create/delete without corrupting filesystem state, using proper mutex locking around all shared structures (bitmaps, inodes, directories)", "Unmount (via fusermount -u or daemon exit) flushes all pending metadata and data to the disk image via fsync, ensuring no data loss and clean remount", "rename operation atomically moves entries between directories, correctly updating source and parent directory link counts and .. entries for directory moves", "Error handling returns proper negative errno values to FUSE (-ENOENT, -EEXIST, -ENOSPC, -EIO, etc.) allowing applications to receive correct error codes", "File handle (fi->fh) correctly stores inode number in open/create and is used in subsequent read/write/release to avoid redundant path resolution"]}] ]
<!-- END_MS -->


<!-- MS_ID: filesystem-m6 -->
# Write-Ahead Journaling and Crash Recovery
You've built a complete filesystem: blocks are allocated, inodes track files, directories organize names, and FUSE makes it all accessible to real programs. But there's a terrifying question you haven't answered: **what happens when the power goes out?**
Not "what happens if" — **when**. Power failures, kernel panics, someone tripping over a power cord, a process being `kill -9`'d mid-operation. These aren't theoretical concerns. They're inevitable.
This milestone is about surviving that inevitability. You're about to implement **write-ahead journaling** — the technique that lets your filesystem crash at any moment and still recover to a consistent state.
## The Fundamental Tension: Operations Are Not Atomic
Here's the problem that will haunt this entire milestone: **a single logical operation requires multiple physical writes.**
Consider what happens when you create a file:
```
Logical operation: create("/home/user/newfile.txt")
Physical operations required:
1. Allocate an inode (update inode bitmap)
2. Initialize the inode structure
3. Add directory entry to parent
4. Update parent directory's mtime
5. Update superblock free inode count
```
Each of these writes goes to a different block on disk. The disk can only write one block at a time. Between any two writes, a crash can occur.
**The window of vulnerability is enormous.** Let's trace what happens if a crash occurs after each step:
| Crash After | Consequence |
|-------------|-------------|
| Step 1 (bitmap) | Inode marked allocated but never initialized → **orphan inode** |
| Step 2 (inode) | Inode exists but no directory entry points to it → **orphan inode** |
| Step 3 (directory) | Directory entry points to inode, but inode not fully initialized → **corrupt read** |
| Step 4 (parent mtime) | Minor inconsistency (stale timestamp) |
| Step 5 (superblock) | Free inode count wrong → **allocation eventually fails** |
None of these outcomes are acceptable. But here's the deeper problem: **you can't make five separate writes atomic.** The disk doesn't have a "write these five blocks atomically" command. Each 4KB block write is individually atomic, but there's no way to combine them.
This is the fundamental tension: **filesystem operations are logically atomic but physically multi-step.** Your job is to bridge that gap.
## The Revelation: Crash Safety ≠ Data Preservation
Here's what surprises most developers: **crash safety doesn't mean your data survives.**
> **Crash safety means: after a crash, the filesystem metadata is consistent.**
Consistent metadata means:
- Every allocated block is reachable from some inode
- Every allocated inode has a directory entry
- The block bitmap matches actual block usage
- The inode bitmap matches actual inode usage
Crash safety does **NOT** guarantee:
- The last 5 seconds of writes are preserved
- The file you were editing has all your changes
- No data was lost
**Metadata journaling protects structure, not content.** A crash mid-write might lose the last few kilobytes of your document — but the filesystem will mount, your other files will be intact, and `fsck` won't find errors.
This is the trade-off at the heart of journaling. Full data journaling (journaling both metadata and content) is possible but doubles write amplification. Every data block is written twice: once to the journal, once to its final location. Most filesystems choose metadata-only journaling because it's the sweet spot between safety and performance.
## The Solution: Write-Ahead Logging
The technique that solves the multi-write atomicity problem is **write-ahead logging** (WAL), also called journaling.
The core insight is simple but profound: **write down what you're about to do before you do it.**
Here's how it works:
```
WITHOUT JOURNALING:
  1. Write inode bitmap
  2. Write inode
  3. Write directory entry
  4. [CRASH] → System is in inconsistent state
WITH JOURNALING:
  1. Write to journal: "I'm about to allocate inode 42, initialize it, and add entry 'file.txt'"
  2. Write commit record: "The above transaction is complete"
  3. Apply changes to actual filesystem (bitmap, inode, directory)
  4. [CRASH] → On recovery, replay journal → consistent state
```
The journal is a **circular buffer** — a fixed-size region of blocks where transactions are written sequentially. When the journal fills up, it wraps around to the beginning.
**The key invariant**: Data is written to the journal BEFORE being written to the main filesystem. This means on recovery:
- If the journal contains a committed transaction → replay it (it may or may not have been applied to the filesystem)
- If the journal contains an uncommitted transaction → discard it (the operation never completed)
Either way, the filesystem ends up consistent. There's no "partial state" — the operation either fully happened or fully didn't.
## The Journal Structure: A Circular Buffer of Transactions
The journal occupies a fixed set of blocks, configured when the filesystem is created:
```c
// Journal header - lives at the start of the journal region
#define JOURNAL_MAGIC 0x4A524E4C  // "JRNL"
typedef struct {
    uint32_t magic;              // Magic number for validation
    uint32_t version;            // Journal format version
    uint64_t sequence;           // Monotonically increasing transaction ID
    uint64_t head;               // Byte offset where next transaction starts
    uint64_t tail;               // Byte offset of first valid transaction
    uint64_t checksum;           // Checksum of header
    uint8_t  padding[4064];      // Pad to 4096 bytes
} __attribute__((packed)) JournalHeader;
// Journal entry types
#define JE_INODE_ALLOC   1  // Allocate an inode
#define JE_INODE_FREE    2  // Free an inode
#define JE_INODE_UPDATE  3  // Update inode metadata
#define JE_BLOCK_ALLOC   4  // Allocate a block
#define JE_BLOCK_FREE    5  // Free a block
#define JE_DIR_ADD       6  // Add directory entry
#define JE_DIR_REMOVE    7  // Remove directory entry
#define JE_SB_UPDATE     8  // Update superblock
#define JE_COMMIT        9  // Transaction commit marker
#define JE_ABORT        10  // Transaction abort marker
// Individual journal entry header
typedef struct {
    uint16_t type;        // Entry type (JE_*)
    uint16_t flags;       // Flags
    uint32_t length;      // Length of data following this header
    uint64_t sequence;    // Transaction sequence number
    uint32_t checksum;    // Checksum of entry + data
    uint8_t  data[];      // Variable-length entry data
} __attribute__((packed)) JournalEntry;
```
**Hardware Soul Check**: Each journal entry is written sequentially to disk. Sequential writes are the fastest possible disk access pattern because:
- **HDD**: No seek time between writes (heads stay in place)
- **SSD**: Flash pages can be written in order, enabling efficient block erasure
The journal turns random writes (scattered across bitmaps, inode tables, data blocks) into sequential writes (one after another in the journal). This is a significant performance win even ignoring crash safety.
## Transaction Lifecycle: Begin, Log, Commit, Apply
Every filesystem operation that modifies metadata must be wrapped in a transaction:
```c
// Transaction states
typedef enum {
    TXN_INACTIVE,
    TXN_ACTIVE,
    TXN_COMMITTED,
    TXN_APPLIED
} TransactionState;
// Active transaction tracking
typedef struct {
    TransactionState state;
    uint64_t sequence;
    uint64_t start_offset;    // Where this transaction starts in journal
    uint32_t entry_count;
    uint32_t total_size;
    JournalEntry* entries;    // Linked list of logged entries
} Transaction;
// Global journal state
typedef struct {
    BlockDevice* dev;
    uint64_t journal_start;   // First block of journal region
    uint64_t journal_blocks;  // Number of blocks in journal
    JournalHeader header;
    Transaction* active_txn;
    void* buffer;             // Buffer for reading/writing journal blocks
} Journal;
```
### Beginning a Transaction
```c
// Start a new transaction
// Returns transaction handle, or NULL on error
Transaction* journal_begin(Journal* j) {
    if (j->active_txn != NULL) {
        // Already have an active transaction - nested not supported
        errno = EBUSY;
        return NULL;
    }
    Transaction* txn = malloc(sizeof(Transaction));
    if (!txn) return NULL;
    txn->state = TXN_ACTIVE;
    txn->sequence = j->header.sequence++;
    txn->start_offset = j->header.head;
    txn->entry_count = 0;
    txn->total_size = 0;
    txn->entries = NULL;
    j->active_txn = txn;
    // Update header with new sequence number
    journal_write_header(j);
    return txn;
}
```
### Logging Operations
Each operation is logged as a journal entry before being applied:
```c
// Log an operation to the current transaction
int journal_log(Journal* j, uint16_t type, const void* data, uint32_t length) {
    if (j->active_txn == NULL) {
        errno = EINVAL;
        return -1;
    }
    Transaction* txn = j->active_txn;
    // Create the entry
    size_t entry_size = sizeof(JournalEntry) + length;
    JournalEntry* entry = malloc(entry_size);
    if (!entry) return -1;
    entry->type = type;
    entry->flags = 0;
    entry->length = length;
    entry->sequence = txn->sequence;
    memcpy(entry->data, data, length);
    // Calculate checksum
    entry->checksum = calculate_checksum(entry, entry_size);
    // Add to transaction's entry list
    entry->flags |= 0x01;  // Mark as "next" for linked list
    JournalEntry** ptr = &txn->entries;
    while (*ptr) {
        ptr = (JournalEntry**)&(*ptr)->data[(*ptr)->length - sizeof(void*)];
    }
    *ptr = entry;
    txn->entry_count++;
    txn->total_size += entry_size;
    return 0;
}
```
**Logging Specific Operations:**
```c
// Log inode allocation
int journal_log_inode_alloc(Journal* j, uint64_t inode_num) {
    struct {
        uint64_t inode_num;
    } data = { inode_num };
    return journal_log(j, JE_INODE_ALLOC, &data, sizeof(data));
}
// Log inode update (full inode copy)
int journal_log_inode_update(Journal* j, uint64_t inode_num, const Inode* inode) {
    struct {
        uint64_t inode_num;
        Inode inode;
    } data;
    data.inode_num = inode_num;
    memcpy(&data.inode, inode, sizeof(Inode));
    return journal_log(j, JE_INODE_UPDATE, &data, sizeof(data));
}
// Log block allocation
int journal_log_block_alloc(Journal* j, uint64_t block_num) {
    struct {
        uint64_t block_num;
    } data = { block_num };
    return journal_log(j, JE_BLOCK_ALLOC, &data, sizeof(data));
}
// Log block free
int journal_log_block_free(Journal* j, uint64_t block_num) {
    struct {
        uint64_t block_num;
    } data = { block_num };
    return journal_log(j, JE_BLOCK_FREE, &data, sizeof(data));
}
// Log directory entry addition
int journal_log_dir_add(Journal* j, uint64_t dir_inode, 
                        const char* name, uint64_t target_inode) {
    size_t name_len = strlen(name);
    struct {
        uint64_t dir_inode;
        uint64_t target_inode;
        uint8_t name_len;
        char name[255];
    } data;
    data.dir_inode = dir_inode;
    data.target_inode = target_inode;
    data.name_len = name_len;
    memcpy(data.name, name, name_len);
    return journal_log(j, JE_DIR_ADD, &data, 
                       sizeof(data) - 255 + name_len);
}
// Log superblock update
int journal_log_sb_update(Journal* j, const Superblock* sb) {
    return journal_log(j, JE_SB_UPDATE, sb, sizeof(Superblock));
}
```
### Committing a Transaction
The commit is the critical moment — it's what makes a transaction "real":
```c
// Commit the current transaction
// This writes all entries + commit record to the journal
int journal_commit(Journal* j) {
    if (j->active_txn == NULL) {
        errno = EINVAL;
        return -1;
    }
    Transaction* txn = j->active_txn;
    if (txn->state != TXN_ACTIVE) {
        errno = EINVAL;
        return -1;
    }
    // Calculate total size needed
    size_t total_size = txn->total_size + sizeof(JournalEntry);  // + commit record
    // Check if we have room in the journal
    uint64_t available = journal_available_space(j);
    if (available < total_size) {
        // Need to checkpoint first
        if (journal_checkpoint(j) < 0) {
            return -1;
        }
    }
    // Write all entries to the journal
    JournalEntry* entry = txn->entries;
    while (entry) {
        size_t entry_size = sizeof(JournalEntry) + entry->length;
        if (journal_write_entry(j, entry) < 0) {
            // Failed - abort the transaction
            journal_abort(j);
            return -1;
        }
        entry = (JournalEntry*)entry->data[entry->length - sizeof(void*)];
    }
    // Write the commit record
    JournalEntry commit_entry;
    commit_entry.type = JE_COMMIT;
    commit_entry.flags = 0;
    commit_entry.length = 0;
    commit_entry.sequence = txn->sequence;
    commit_entry.checksum = calculate_checksum(&commit_entry, sizeof(JournalEntry));
    if (journal_write_entry(j, &commit_entry) < 0) {
        journal_abort(j);
        return -1;
    }
    // CRITICAL: Flush the journal to disk
    // This ensures the commit record is on stable storage
    fsync(j->dev->fd);
    txn->state = TXN_COMMITTED;
    return 0;
}
```
**The Critical fsync**: The `fsync()` call after writing the commit record is non-negotiable. Without it, the OS may buffer the write, and a crash could lose the commit record. An uncommitted transaction is discarded on recovery, so losing the commit means losing all the work.
### Applying Changes to the Filesystem
After the transaction is committed to the journal, we apply the actual changes:
```c
// Apply a committed transaction to the filesystem
int journal_apply(Journal* j, FileSystem* fs) {
    if (j->active_txn == NULL || 
        j->active_txn->state != TXN_COMMITTED) {
        errno = EINVAL;
        return -1;
    }
    Transaction* txn = j->active_txn;
    // Replay each logged operation
    JournalEntry* entry = txn->entries;
    while (entry) {
        if (apply_entry(fs, entry) < 0) {
            // This shouldn't happen if the transaction was logged correctly
            fprintf(stderr, "Failed to apply journal entry type %d\n", entry->type);
            // Continue anyway - partial application is okay since we'll replay on recovery
        }
        entry = (JournalEntry*)entry->data[entry->length - sizeof(void*)];
    }
    txn->state = TXN_APPLIED;
    return 0;
}
// Apply a single journal entry to the filesystem
static int apply_entry(FileSystem* fs, const JournalEntry* entry) {
    switch (entry->type) {
        case JE_INODE_ALLOC: {
            struct {
                uint64_t inode_num;
            }* data = (void*)entry->data;
            // Mark inode as allocated in bitmap
            return inode_bitmap_set(fs, data->inode_num, 1);
        }
        case JE_INODE_UPDATE: {
            struct {
                uint64_t inode_num;
                Inode inode;
            }* data = (void*)entry->data;
            // Write the inode to disk
            return write_inode(fs, data->inode_num, &data->inode);
        }
        case JE_BLOCK_ALLOC: {
            struct {
                uint64_t block_num;
            }* data = (void*)entry->data;
            // Mark block as allocated in bitmap
            return block_bitmap_set(fs, data->block_num, 1);
        }
        case JE_BLOCK_FREE: {
            struct {
                uint64_t block_num;
            }* data = (void*)entry->data;
            // Mark block as free in bitmap
            return block_bitmap_set(fs, data->block_num, 0);
        }
        case JE_DIR_ADD: {
            struct {
                uint64_t dir_inode;
                uint64_t target_inode;
                uint8_t name_len;
                char name[255];
            }* data = (void*)entry->data;
            // Add directory entry (the actual implementation)
            char name[256];
            memcpy(name, data->name, data->name_len);
            name[data->name_len] = '\0';
            return dir_add_entry_internal(fs, data->dir_inode, 
                                          name, data->target_inode);
        }
        case JE_SB_UPDATE: {
            // Update superblock
            memcpy(fs->sb, entry->data, sizeof(Superblock));
            return write_block(fs->dev, 0, fs->sb);
        }
        default:
            // Unknown entry type - skip
            return 0;
    }
}
```
### Ending a Transaction
```c
// End a transaction and free its resources
void journal_end(Journal* j) {
    if (j->active_txn == NULL) return;
    Transaction* txn = j->active_txn;
    // Free all logged entries
    JournalEntry* entry = txn->entries;
    while (entry) {
        JournalEntry* next = (JournalEntry*)entry->data[entry->length - sizeof(void*)];
        free(entry);
        entry = next;
    }
    free(txn);
    j->active_txn = NULL;
    // Update journal header
    journal_write_header(j);
}
```
### Aborting a Transaction
Sometimes things go wrong before commit:
```c
// Abort the current transaction
int journal_abort(Journal* j) {
    if (j->active_txn == NULL) {
        errno = EINVAL;
        return -1;
    }
    Transaction* txn = j->active_txn;
    // Write an abort record (optional but helpful for debugging)
    JournalEntry abort_entry;
    abort_entry.type = JE_ABORT;
    abort_entry.flags = 0;
    abort_entry.length = 0;
    abort_entry.sequence = txn->sequence;
    abort_entry.checksum = calculate_checksum(&abort_entry, sizeof(JournalEntry));
    journal_write_entry(j, &abort_entry);
    // Free the transaction
    journal_end(j);
    return 0;
}
```
## Wrapping Filesystem Operations in Transactions
Now let's see how actual filesystem operations use the journal:
```c
// Create a file with journaling
uint64_t fs_create_journaled(FileSystem* fs, const char* path, uint16_t mode) {
    Journal* j = fs->journal;
    // Begin transaction
    Transaction* txn = journal_begin(j);
    if (!txn) return 0;
    // Resolve parent directory
    char name[MAX_NAME_LEN + 1];
    uint64_t parent_inode_num = path_resolve_parent(fs, path, name);
    if (parent_inode_num == 0) {
        journal_abort(j);
        return 0;
    }
    // Allocate inode
    uint64_t new_inode_num = alloc_inode(fs);
    if (new_inode_num == 0) {
        journal_abort(j);
        return 0;
    }
    // LOG: inode allocation
    journal_log_inode_alloc(j, new_inode_num);
    // Initialize inode
    Inode new_inode;
    memset(&new_inode, 0, sizeof(Inode));
    new_inode.mode = S_IFREG | (mode & 07777);
    new_inode.uid = 0;
    new_inode.gid = 0;
    new_inode.link_count = 1;
    new_inode.size = 0;
    new_inode.atime = new_inode.mtime = new_inode.ctime = (uint64_t)time(NULL);
    // LOG: inode update
    journal_log_inode_update(j, new_inode_num, &new_inode);
    // LOG: directory entry addition
    journal_log_dir_add(j, parent_inode_num, name, new_inode_num);
    // LOG: superblock update (free inode count changed)
    journal_log_sb_update(j, fs->sb);
    // COMMIT: All operations logged, now commit
    if (journal_commit(j) < 0) {
        journal_abort(j);
        return 0;
    }
    // APPLY: Now actually modify the filesystem
    // These operations must be idempotent - safe to replay
    inode_bitmap_set(fs, new_inode_num, 1);
    write_inode(fs, new_inode_num, &new_inode);
    dir_add_entry_internal(fs, parent_inode_num, name, new_inode_num, DT_REG);
    write_block(fs->dev, 0, fs->sb);
    // END: Transaction complete
    journal_end(j);
    return new_inode_num;
}
```
**The Order Matters**: Notice the sequence:
1. **Begin** transaction
2. **Log** all operations
3. **Commit** to journal (with fsync!)
4. **Apply** changes to filesystem
5. **End** transaction
If a crash occurs before commit, the transaction is incomplete and will be discarded. If a crash occurs after commit but before apply, recovery will replay the transaction. Either way, consistency is preserved.
## Journal Replay: Recovery on Mount
When the filesystem mounts after a crash, it must scan the journal and replay any committed transactions:
```c
// Recover the filesystem by replaying the journal
int journal_recover(Journal* j, FileSystem* fs) {
    printf("Starting journal recovery...\n");
    // Read the journal header
    if (journal_read_header(j) < 0) {
        fprintf(stderr, "Failed to read journal header\n");
        return -1;
    }
    // Validate journal
    if (j->header.magic != JOURNAL_MAGIC) {
        fprintf(stderr, "Invalid journal magic number\n");
        return -1;
    }
    // Scan the journal for committed transactions
    uint64_t offset = j->header.tail;
    uint64_t head = j->header.head;
    int recovered = 0;
    while (offset < head) {
        // Read entry at this offset
        JournalEntry* entry = journal_read_entry(j, offset);
        if (!entry) {
            fprintf(stderr, "Failed to read journal entry at offset %lu\n", offset);
            break;
        }
        // Validate checksum
        size_t entry_size = sizeof(JournalEntry) + entry->length;
        uint32_t checksum = calculate_checksum(entry, entry_size);
        if (checksum != entry->checksum) {
            fprintf(stderr, "Journal entry checksum mismatch at offset %lu\n", offset);
            free(entry);
            break;
        }
        if (entry->type == JE_COMMIT) {
            // Found a commit - replay the transaction
            printf("  Found committed transaction seq=%lu\n", entry->sequence);
            // Replay from transaction start to this commit
            if (replay_transaction(j, fs, entry->sequence) < 0) {
                fprintf(stderr, "  Failed to replay transaction %lu\n", entry->sequence);
            } else {
                recovered++;
                printf("  Replayed transaction %lu successfully\n", entry->sequence);
            }
        }
        offset += entry_size;
        free(entry);
    }
    printf("Journal recovery complete: %d transactions replayed\n", recovered);
    // Clear the journal - all committed transactions have been applied
    journal_clear(j);
    return 0;
}
// Replay a specific transaction
static int replay_transaction(Journal* j, FileSystem* fs, uint64_t sequence) {
    uint64_t offset = j->header.tail;
    while (offset < j->header.head) {
        JournalEntry* entry = journal_read_entry(j, offset);
        if (!entry) return -1;
        size_t entry_size = sizeof(JournalEntry) + entry->length;
        // Stop at commit record for this transaction
        if (entry->type == JE_COMMIT && entry->sequence == sequence) {
            free(entry);
            break;
        }
        // Only replay entries for this transaction
        if (entry->sequence == sequence) {
            if (apply_entry(fs, entry) < 0) {
                fprintf(stderr, "    Failed to apply entry type %d\n", entry->type);
            }
        }
        offset += entry_size;
        free(entry);
    }
    // Sync the filesystem after replay
    fsync(fs->dev->fd);
    return 0;
}
```
**Idempotency is Critical**: The `apply_entry` function must be **idempotent** — safe to call multiple times with the same result. If a transaction was partially applied before a crash, replaying it should still produce the correct final state.
For example, `inode_bitmap_set(fs, inode_num, 1)` sets a bit to 1. Calling it multiple times still results in the bit being 1. This is idempotent.
A non-idempotent operation would be `inode_bitmap_flip(fs, inode_num)` — toggling the bit. Replaying this twice would produce the wrong result.
## Checkpoint: Marking the Journal Clean
Once all committed transactions have been applied to the filesystem, the journal can be cleared:
```c
// Clear the journal after all transactions have been applied
int journal_checkpoint(Journal* j) {
    // Write a new header with head = tail (empty journal)
    j->header.head = j->header.tail;
    j->header.sequence++;  // Advance sequence for next transaction
    if (journal_write_header(j) < 0) {
        return -1;
    }
    // Sync to disk
    fsync(j->dev->fd);
    return 0;
}
// Clear all journal entries
int journal_clear(Journal* j) {
    // Zero out the journal region
    void* zero_buffer = malloc(BLOCK_SIZE);
    if (!zero_buffer) return -1;
    memset(zero_buffer, 0, BLOCK_SIZE);
    for (uint64_t i = 0; i < j->journal_blocks; i++) {
        write_block(j->dev, j->journal_start + i, zero_buffer);
    }
    free(zero_buffer);
    // Reset header
    j->header.head = sizeof(JournalHeader);
    j->header.tail = sizeof(JournalHeader);
    j->header.sequence = 0;
    journal_write_header(j);
    fsync(j->dev->fd);
    return 0;
}
```
The checkpoint operation is crucial for preventing the journal from filling up. Without periodic checkpoints:
1. The journal fills with old transactions
2. New transactions can't be written
3. The filesystem becomes read-only
## The Circular Buffer: Wrap-Around Handling
The journal is a fixed-size circular buffer. When writes reach the end, they wrap to the beginning:
```c
// Calculate available space in the journal
uint64_t journal_available_space(Journal* j) {
    uint64_t journal_size = j->journal_blocks * BLOCK_SIZE;
    if (j->header.head >= j->header.tail) {
        // Simple case: head is ahead of tail
        return journal_size - (j->header.head - j->header.tail) - sizeof(JournalHeader);
    } else {
        // Wrapped: head is behind tail
        return j->header.tail - j->header.head - sizeof(JournalHeader);
    }
}
// Write an entry to the journal, handling wrap-around
int journal_write_entry(Journal* j, const JournalEntry* entry) {
    size_t entry_size = sizeof(JournalEntry) + entry->length;
    uint64_t journal_size = j->journal_blocks * BLOCK_SIZE;
    // Check if entry would wrap around
    uint64_t offset_in_block = j->header.head % BLOCK_SIZE;
    uint64_t remaining_in_block = BLOCK_SIZE - offset_in_block;
    if (entry_size > remaining_in_block) {
        // Entry would cross block boundary - need to handle carefully
        if (entry_size > BLOCK_SIZE) {
            // Entry is larger than a block - not supported
            errno = E2BIG;
            return -1;
        }
        // Pad the rest of this block with zeros and move to next block
        uint64_t current_block = j->journal_start + (j->header.head / BLOCK_SIZE);
        char pad_buffer[BLOCK_SIZE];
        memset(pad_buffer, 0, BLOCK_SIZE);
        // Zero the remainder of the current block
        write_block(j->dev, current_block, pad_buffer);
        // Advance head to start of next block
        j->header.head = ((j->header.head / BLOCK_SIZE) + 1) * BLOCK_SIZE;
        // Check for wrap-around
        if (j->header.head >= journal_size) {
            j->header.head = sizeof(JournalHeader);
        }
    }
    // Now write the entry
    uint64_t block_num = j->journal_start + (j->header.head / BLOCK_SIZE);
    uint64_t block_offset = j->header.head % BLOCK_SIZE;
    // Read the block
    char block_buffer[BLOCK_SIZE];
    if (read_block(j->dev, block_num, block_buffer) < 0) {
        return -1;
    }
    // Write the entry into the block
    memcpy(block_buffer + block_offset, entry, entry_size);
    // Write the block back
    if (write_block(j->dev, block_num, block_buffer) < 0) {
        return -1;
    }
    // Advance head
    j->header.head += entry_size;
    // Handle wrap-around
    if (j->header.head >= journal_size) {
        j->header.head = sizeof(JournalHeader);
    }
    return 0;
}
```

![Filesystem Layer Stack](./diagrams/diag-l1-layer-stack.svg)

## Metadata Journaling vs. Full Data Journaling
There are two journaling modes, with a significant trade-off:
| Mode | What's Journaled | Write Amplification | Safety Level | Used By |
|------|------------------|---------------------|--------------|---------|
| **Metadata-only ✓** | Inodes, bitmaps, directories | 1× (journal only for metadata) | Filesystem consistent, data may be lost | ext4 (default), XFS |
| Full data | Everything including file content | 2× (all data written twice) | Both filesystem and data preserved | ext4 (data=journal),早期日志FS |
```c
// Our implementation uses metadata-only journaling
// Data blocks are written directly, not journaled
ssize_t fs_write_journaled(FileSystem* fs, uint64_t inode_num, 
                           uint64_t offset, const void* data, size_t length) {
    Journal* j = fs->journal;
    // Begin transaction
    Transaction* txn = journal_begin(j);
    if (!txn) return -1;
    // Read current inode
    Inode inode;
    if (read_inode(fs, inode_num, &inode) < 0) {
        journal_abort(j);
        return -1;
    }
    // Perform the write (this allocates blocks, modifies inode)
    ssize_t written = fs_write_internal(fs, inode_num, offset, data, length);
    if (written < 0) {
        journal_abort(j);
        return -1;
    }
    // Re-read the modified inode
    read_inode(fs, inode_num, &inode);
    // LOG: inode update (metadata change only)
    journal_log_inode_update(j, inode_num, &inode);
    // LOG: superblock update (free block count may have changed)
    journal_log_sb_update(j, fs->sb);
    // Note: We do NOT log the actual data being written
    // Data blocks are written directly to their final locations
    // A crash may lose recently written data, but metadata stays consistent
    // COMMIT
    if (journal_commit(j) < 0) {
        journal_abort(j);
        return -1;
    }
    // APPLY (metadata only - data was already written)
    write_inode(fs, inode_num, &inode);
    write_block(fs->dev, 0, fs->sb);
    journal_end(j);
    return written;
}
```
**The Trade-off Explained**: With metadata-only journaling:
- You write 10KB of data → 1 data block write + 1 inode write + journal commit
- A crash might lose the data but the inode will show the correct (old) state
- The filesystem is always mountable and self-consistent
With full data journaling:
- You write 10KB of data → 1 journal write (data) + 1 journal write (commit) + 1 data block write + 1 inode write
- Twice as many writes for data
- A crash preserves both the data and the metadata
Most production filesystems choose metadata-only because the performance cost of full journaling is too high.
## Crash Simulation Testing
The ultimate test of journaling is: does recovery actually work? Let's build a crash simulator:
```c
#include <signal.h>
#include <sys/wait.h>
// Crash simulation: kill the process at a random point during an operation
void test_crash_recovery(FileSystem* fs, const char* disk_image) {
    printf("=== Crash Simulation Test ===\n");
    // Fork a child process to perform operations
    pid_t pid = fork();
    if (pid == 0) {
        // Child process: perform filesystem operations
        // These may be interrupted by SIGKILL at any point
        for (int i = 0; i < 100; i++) {
            char path[256];
            snprintf(path, sizeof(path), "/crash_test_file_%d.txt", i);
            // Create file
            uint64_t inode = fs_create_journaled(fs, path, 0644);
            if (inode == 0) {
                fprintf(stderr, "Failed to create %s\n", path);
                continue;
            }
            // Write some data
            char data[1024];
            memset(data, 'X', sizeof(data));
            fs_write_journaled(fs, inode, 0, data, sizeof(data));
            // Maybe delete it
            if (i % 3 == 0) {
                fs_unlink_journaled(fs, path);
            }
        }
        // If we get here, all operations completed
        exit(0);
    } else {
        // Parent process: kill the child at a random time
        usleep(10000 + (rand() % 100000));  // 10-110ms delay
        kill(pid, SIGKILL);
        waitpid(pid, NULL, 0);
        printf("  Child process killed\n");
        // Now try to recover
        printf("  Attempting recovery...\n");
        // Reopen filesystem (triggers journal replay)
        FileSystem* recovered_fs = fs_open(disk_image);
        if (!recovered_fs) {
            fprintf(stderr, "  FAILED: Could not open filesystem after crash\n");
            return;
        }
        // Verify filesystem consistency
        printf("  Running consistency check...\n");
        int errors = verify_filesystem(recovered_fs);
        if (errors == 0) {
            printf("  SUCCESS: Filesystem is consistent after crash\n");
        } else {
            printf("  FAILED: %d consistency errors found\n", errors);
        }
        // Count files that survived
        int file_count = 0;
        for (int i = 0; i < 100; i++) {
            char path[256];
            snprintf(path, sizeof(path), "/crash_test_file_%d.txt", i);
            if (path_resolve(recovered_fs, path) != 0) {
                file_count++;
            }
        }
        printf("  %d files survived the crash\n", file_count);
        fs_close(recovered_fs);
    }
}
// Comprehensive crash test: test crash at each phase of transaction
void test_crash_at_each_phase(FileSystem* fs, const char* disk_image) {
    printf("\n=== Phase-by-Phase Crash Test ===\n");
    const char* phases[] = {
        "before_begin",
        "after_begin",
        "after_log_inode",
        "after_log_dir",
        "after_commit",
        "after_apply"
    };
    for (int phase = 0; phase < 6; phase++) {
        printf("\nTesting crash at phase: %s\n", phases[phase]);
        // Create a fresh filesystem for this test
        system("rm -f crash_test.img");
        FileSystem* test_fs = fs_format("crash_test.img", 1000, 64, 32);
        // Fork and crash at specific phase
        pid_t pid = fork();
        if (pid == 0) {
            // Child: perform operation with crash point
            test_create_with_crash_point(test_fs, "/testfile.txt", phase);
            exit(0);
        } else {
            waitpid(pid, NULL, 0);
            // Recover and check
            FileSystem* recovered = fs_open("crash_test.img");
            int errors = verify_filesystem(recovered);
            if (errors == 0) {
                printf("  ✓ PASS: Filesystem consistent\n");
                // Check if file exists
                uint64_t inode = path_resolve(recovered, "/testfile.txt");
                if (phase >= 4) {  // After commit
                    printf("  File exists: %s (expected: yes)\n", 
                           inode ? "yes" : "no");
                } else {  // Before commit
                    printf("  File exists: %s (expected: no)\n",
                           inode ? "yes" : "no");
                }
            } else {
                printf("  ✗ FAIL: %d errors\n", errors);
            }
            fs_close(recovered);
            fs_close(test_fs);
        }
    }
}
// Perform create operation with a crash point
void test_create_with_crash_point(FileSystem* fs, const char* path, int crash_phase) {
    Journal* j = fs->journal;
    if (crash_phase == 0) {
        // Crash before begin
        raise(SIGKILL);
    }
    Transaction* txn = journal_begin(j);
    if (crash_phase == 1) {
        // Crash after begin
        raise(SIGKILL);
    }
    uint64_t inode_num = alloc_inode(fs);
    journal_log_inode_alloc(j, inode_num);
    if (crash_phase == 2) {
        raise(SIGKILL);
    }
    Inode inode;
    memset(&inode, 0, sizeof(Inode));
    inode.mode = S_IFREG | 0644;
    inode.link_count = 1;
    journal_log_inode_update(j, inode_num, &inode);
    journal_log_dir_add(j, 1, "testfile.txt", inode_num);
    if (crash_phase == 3) {
        raise(SIGKILL);
    }
    journal_commit(j);
    if (crash_phase == 4) {
        raise(SIGKILL);
    }
    // Apply changes
    inode_bitmap_set(fs, inode_num, 1);
    write_inode(fs, inode_num, &inode);
    dir_add_entry_internal(fs, 1, "testfile.txt", inode_num, DT_REG);
    if (crash_phase == 5) {
        raise(SIGKILL);
    }
    journal_end(j);
}
```
## Integration with FUSE
The FUSE daemon needs to trigger recovery on mount and checkpoint on unmount:
```c
// Modified FUSE init with journal recovery
static void* fs_init_with_journal(struct fuse_conn_info* conn) {
    // Recover journal before accepting requests
    if (fs_global->journal) {
        printf("Checking journal for recovery...\n");
        journal_recover(fs_global->journal, fs_global);
    }
    // Enable capabilities
    conn->want |= FUSE_CAP_BIG_WRITES;
    return fs_global;
}
// Modified FUSE destroy with journal checkpoint
static void fs_destroy_with_journal(void* private_data) {
    if (fs_global) {
        // Checkpoint the journal
        if (fs_global->journal) {
            printf("Checkpointing journal...\n");
            journal_checkpoint(fs_global->journal);
        }
        // Sync everything
        if (fs_global->dev) {
            fsync(fs_global->dev->fd);
        }
    }
}
```
## Common Pitfalls
**Forgetting to fsync the journal**: The commit record must be on stable storage before returning. Without `fsync()`, the OS may buffer the write, and a crash loses the commit.
**Non-idempotent replay operations**: If `apply_entry` can't be called multiple times safely, replay will corrupt the filesystem. Test this explicitly.
**Journal wrap-around without checkpointing**: If the journal fills up, new transactions fail. Implement periodic checkpoints to clear old entries.
**Logging after applying**: The order must be: log → commit → apply. If you apply first and then log, a crash after apply but before commit leaves the filesystem modified but the transaction uncommitted.
**Not logging superblock changes**: The superblock tracks free counts. If these are wrong after recovery, allocation will eventually fail. Always log superblock updates.
**Partial journal entries**: If a crash occurs mid-write to the journal, the entry will have a bad checksum and be discarded. This is correct behavior — but make sure your checksum validation catches it.
## What You've Built
At the end of this milestone, you have:
1. **A journal region** with a header tracking head, tail, and sequence numbers
2. **Transaction lifecycle**: begin, log, commit, apply, end
3. **Journal entries** for each metadata operation type
4. **Commit records** that make transactions durable
5. **Recovery replay** that scans the journal and reapplies committed transactions
6. **Checkpoint mechanism** that clears the journal after all transactions are applied
7. **Crash simulation tests** that verify recovery works
8. **Metadata-only journaling** that protects filesystem structure without full data duplication
You can now:
```bash
$ ./mkfs disk.img 100
$ ./fs_fuse disk.img /mnt/myfs &
$ cp large_file /mnt/myfs/
$ kill -9 $(pgrep fs_fuse)   # Simulate crash
$ ./fs_fuse disk.img /mnt/myfs
Starting journal recovery...
  Found committed transaction seq=42
  Replayed transaction 42 successfully
Journal recovery complete: 1 transactions replayed
$ ls /mnt/myfs
large_file  # File survived the crash
$ fsck disk.img
Filesystem is clean
```
## The Knowledge Cascade
The patterns you've learned extend far beyond filesystems:
**Database Write-Ahead Logs (PostgreSQL, SQLite, MySQL)**: The exact same technique you implemented is how databases achieve ACID durability. PostgreSQL's WAL, SQLite's journal file, and MySQL's redo log all use the same pattern: log modifications before applying them, recover by replaying committed transactions. The main difference is that databases log row-level changes (INSERT, UPDATE, DELETE) while you logged filesystem metadata changes. When you understand journaling, you understand why databases can claim "durable" in ACID — the WAL ensures committed transactions survive crashes.
**Message Queue Durability (Kafka, RabbitMQ)**: Kafka's logs are essentially journals. Each message is written to a sequential log file before being marked as available to consumers. If a broker crashes, it replays the log to restore message state. RabbitMQ's persistent queues use a similar write-ahead pattern. The key insight: durability requires writing to stable storage before acknowledging completion.
**Event Sourcing and CQRS**: The event store in event sourcing IS a journal. Instead of storing current state, you store a sequence of events (state changes). To reconstruct state, you replay all events from the beginning (or from a snapshot). The events are immutable, append-only, and ordered — exactly like journal entries. Your journal replay logic is the same as event sourcing's state reconstruction.
**Distributed Consensus (Raft, Paxos)**: The consensus log is a replicated write-ahead journal. When a leader receives a client request, it:
1. Appends the request to its log (your "journal write")
2. Replicates to followers (your "fsync" equivalent)
3. Commits once majority acknowledge (your "commit record")
4. Applies to state machine (your "apply")
Crash recovery in Raft is identical to journal replay — the follower reads its log and reapplies committed entries. The Raft paper even uses "log" and "journal" interchangeably.
**Version Control Reflogs (Git)**: Git's reflog is a journal of reference updates. Every time you commit, amend, reset, or rebase, Git logs the old and new reference values. If you accidentally `git reset --hard` to the wrong commit, you can recover by consulting the reflog and resetting to a previous state. This is journaling applied to version control metadata.
**Blockchain**: Each block in a blockchain is effectively a journal entry containing a batch of transactions. The chain structure ensures entries can't be modified after being committed. "Replay" in blockchain means re-executing all transactions from genesis to verify state — the same principle as journal recovery, just with cryptographic verification added.
In every case, the core pattern is the same: **write intentions before actions, recover by replaying intentions**. This is the universal pattern for building crash-tolerant systems.
---
[[EXPLAIN:atomic-writes-and-durability|Atomic writes and durability]]
---
<!-- END_MS -->


## System Overview

![System Overview](./diagrams/system-overview.svg)




# TDD

A complete inode-based filesystem with block allocation, directory tree management, FUSE integration, and write-ahead journaling for crash consistency. The design embodies the fundamental tension between logical file abstractions (paths, byte streams) and physical storage reality (blocks, sectors, disk latency). Each layer negotiates with hardware constraints: 4KB block alignment matches SSD pages and memory pages, bitmap allocation trades scan time for constant-time free/used tracking, indirect pointers balance inode size against maximum file size, and journaling transforms random metadata writes into sequential log entries for both crash safety and write performance.



<!-- TDD_MOD_ID: filesystem-m1 -->
# Technical Design Specification: Block Layer and mkfs
## Module Charter
This module implements the foundational layer of the filesystem: a block device abstraction that presents fixed-size (4KB) block I/O operations over a file-backed disk image, and the mkfs formatter that transforms empty space into a structured filesystem. The on-disk layout partitions the block address space into distinct regions—superblock (block 0), block bitmap, inode bitmap, inode table, journal region, and data blocks—with all positions derived from superblock metadata. Bitmap-based allocators track free/used status for both blocks and inodes. This module does NOT implement inode contents (file metadata, indirect pointers), directory operations, file I/O, or journaling logic—those belong to subsequent milestones. The invariant is absolute: after mkfs completes, the filesystem image must be mountable, with a valid superblock at block 0, consistent bitmaps, a root directory inode (number 1) initialized with `.` and `..` entries, and free counts that match actual availability.
---
## File Structure
```
filesystem/
├── 01_block.h           # BlockDevice struct, BLOCK_SIZE constant, function declarations
├── 02_block.c           # block_device_create, read_block, write_block, block_device_close
├── 03_superblock.h      # Superblock struct, FS_MAGIC, FS_VERSION, field documentation
├── 04_superblock.c      # superblock_init, superblock_validate, superblock_write, superblock_read
├── 05_bitmap.h          # Bitmap function declarations, BITS_PER_BLOCK
├── 06_bitmap.c          # block_bitmap_alloc, block_bitmap_free, inode_bitmap_alloc, inode_bitmap_free
├── 07_layout.h          # layout_calculate function, LayoutResult struct
├── 08_layout.c          # Region size calculations from total_blocks
├── 09_mkfs.h            # mkfs function declaration, MkfsOptions struct
├── 10_mkfs.c            # mkfs implementation, root directory creation
├── 11_verify.h          # verify_filesystem declaration
├── 12_verify.c          # Consistency checking implementation
├── 13_mkfs_main.c       # CLI entry point with argument parsing
└── tests/
    ├── test_block.c     # Block device read/write tests
    ├── test_bitmap.c    # Allocation/deallocation tests
    └── test_mkfs.c      # Full mkfs round-trip verification
```
---
## Complete Data Model
### BlockDevice
The block device abstraction wraps a file descriptor and provides block-aligned I/O operations.
```c
// 01_block.h
#include <stdint.h>
#include <stddef.h>
#define BLOCK_SIZE 4096  // 4 KB - matches SSD pages, memory pages, ext4 default
#define BLOCK_SHIFT 12   // log2(BLOCK_SIZE) for fast division
typedef struct {
    int fd;              // File descriptor for backing file (offset 0x00, 4 bytes)
    uint64_t size;       // Total size in bytes (offset 0x08, 8 bytes)
    uint64_t num_blocks; // Total number of blocks (offset 0x10, 8 bytes)
} BlockDevice;           // Total: 24 bytes
// Alignment note: struct fits in 1 cache line (64 bytes) with room to spare
```
**Why each field exists:**
- `fd`: Required for all file operations. Stored to avoid repeated `open()` calls.
- `size`: Needed for bounds checking. Calculated once at creation.
- `num_blocks`: Precomputed for fast validity checks. Avoids division in hot paths.
### Superblock
The superblock is the master record at block 0. It must be readable without any other information.
```c
// 03_superblock.h
#include <stdint.h>
#define FS_MAGIC     0x46534D4B  // "FSMK" - little-endian: 4B 4D 53 46
#define FS_VERSION   1
typedef struct {
    // Identification (offset 0x00, 8 bytes)
    uint32_t magic;              // Must equal FS_MAGIC
    uint32_t version;            // Filesystem version, currently 1
    // Geometry (offset 0x08, 24 bytes)
    uint64_t total_blocks;       // Total blocks in filesystem
    uint64_t total_inodes;       // Total inodes in inode table
    uint32_t block_size;         // Block size in bytes (4096)
    uint32_t inode_size;         // Inode size in bytes (4096 in this design)
    // Block bitmap region (offset 0x20, 16 bytes)
    uint64_t block_bitmap_start;     // First block of block bitmap
    uint64_t block_bitmap_blocks;    // Number of blocks in block bitmap
    // Inode bitmap region (offset 0x30, 16 bytes)
    uint64_t inode_bitmap_start;     // First block of inode bitmap
    uint64_t inode_bitmap_blocks;    // Number of blocks in inode bitmap
    // Inode table region (offset 0x40, 16 bytes)
    uint64_t inode_table_start;      // First block of inode table
    uint64_t inode_table_blocks;     // Number of blocks in inode table
    // Journal region (offset 0x50, 16 bytes)
    uint64_t journal_start;          // First block of journal
    uint64_t journal_blocks;         // Number of blocks in journal
    // Data region (offset 0x60, 8 bytes)
    uint64_t data_start;             // First data block
    // Dynamic counts (offset 0x68, 16 bytes)
    uint64_t free_blocks;            // Current count of free blocks
    uint64_t free_inodes;            // Current count of free inodes
    // Timestamps (offset 0x78, 16 bytes)
    uint64_t mount_time;             // Last mount time (Unix epoch)
    uint64_t write_time;             // Last write time (Unix epoch)
    // Reserved (offset 0x88, 3944 bytes)
    uint8_t reserved[3944];          // Pad to exactly 4096 bytes
    // Checksum (offset 0xFF8, 8 bytes) - last field
    uint64_t checksum;               // CRC64 of all preceding bytes
} __attribute__((packed)) Superblock;  // Total: 4096 bytes = 1 block
// Compile-time verification
static_assert(sizeof(Superblock) == BLOCK_SIZE, "Superblock must be exactly one block");
```
**Byte offset table:**
| Offset | Size | Field | Purpose |
|--------|------|-------|---------|
| 0x00 | 4 | magic | Identify filesystem type |
| 0x04 | 4 | version | Format compatibility |
| 0x08 | 8 | total_blocks | Geometry: total capacity |
| 0x10 | 8 | total_inodes | Geometry: max files |
| 0x18 | 4 | block_size | Geometry: I/O unit |
| 0x1C | 4 | inode_size | Geometry: metadata unit |
| 0x20 | 8 | block_bitmap_start | Region start |
| 0x28 | 8 | block_bitmap_blocks | Region size |
| 0x30 | 8 | inode_bitmap_start | Region start |
| 0x38 | 8 | inode_bitmap_blocks | Region size |
| 0x40 | 8 | inode_table_start | Region start |
| 0x48 | 8 | inode_table_blocks | Region size |
| 0x50 | 8 | journal_start | Region start |
| 0x58 | 8 | journal_blocks | Region size |
| 0x60 | 8 | data_start | Region start |
| 0x68 | 8 | free_blocks | Dynamic count |
| 0x70 | 8 | free_inodes | Dynamic count |
| 0x78 | 8 | mount_time | Last mount |
| 0x80 | 8 | write_time | Last modification |
| 0x88 | 3944 | reserved | Future expansion |
| 0xFF8 | 8 | checksum | Integrity |
**Hardware Soul - Cache Line Analysis:**
- Superblock spans 64 cache lines (4096 / 64)
- First cache line (0x00-0x3F): magic, version, geometry - most frequently read
- Middle cache lines: region pointers - read during allocation
- Last cache line: checksum - validated on mount

![Block Device Architecture](./diagrams/tdd-diag-m1-01.svg)

### LayoutResult
Calculated filesystem layout returned by `layout_calculate`.
```c
// 07_layout.h
typedef struct {
    uint64_t block_bitmap_start;
    uint64_t block_bitmap_blocks;
    uint64_t inode_bitmap_start;
    uint64_t inode_bitmap_blocks;
    uint64_t inode_table_start;
    uint64_t inode_table_blocks;
    uint64_t journal_start;
    uint64_t journal_blocks;
    uint64_t data_start;
    uint64_t data_blocks;      // Number of usable data blocks
    uint64_t metadata_blocks;  // Total blocks consumed by metadata
} LayoutResult;
```
### Directory Entry (for root directory initialization)
```c
// 09_mkfs.c - internal structure
#define DT_DIR 2
#define MAX_NAME_LEN 255
typedef struct {
    uint64_t inode;          // 8 bytes: inode number (0 = unused)
    uint16_t rec_len;        // 2 bytes: record length (280 for this design)
    uint8_t  name_len;       // 1 byte: name length
    uint8_t  file_type;      // 1 byte: DT_* constant
    char     name[256];      // 256 bytes: null-padded name
    uint8_t  _padding[12];   // 12 bytes: align to 280 total
} __attribute__((packed)) DirEntry;
#define DIR_ENTRY_SIZE 280
#define ENTRIES_PER_BLOCK (BLOCK_SIZE / DIR_ENTRY_SIZE)  // 14
```
---
## Interface Contracts
### Block Device Operations
```c
// 01_block.h
/**
 * Create a block device backed by a file.
 * Creates the file if it doesn't exist, truncates if it does.
 * Pre-allocates the file to the required size.
 *
 * @param path      File path for the disk image
 * @param num_blocks Number of blocks to allocate
 * @return BlockDevice* on success, NULL on failure (errno set)
 *
 * Errors:
 *   EACCES  - Permission denied
 *   ENOSPC  - Not enough space on filesystem
 *   EINVAL  - num_blocks == 0
 */
BlockDevice* block_device_create(const char* path, uint64_t num_blocks);
/**
 * Read a single block into the provided buffer.
 *
 * @param dev        BlockDevice from block_device_create
 * @param block_num  Block number to read (0-indexed)
 * @param buffer     Output buffer, must be at least BLOCK_SIZE bytes
 * @return 0 on success, -1 on failure (errno set)
 *
 * Errors:
 *   EINVAL - block_num >= dev->num_blocks
 *   EIO    - Read error from underlying file
 *
 * Postcondition: buffer contains BLOCK_SIZE bytes from block_num,
 *                zero-filled if file ended early
 */
int read_block(BlockDevice* dev, uint64_t block_num, void* buffer);
/**
 * Write a single block from the provided buffer.
 *
 * @param dev        BlockDevice from block_device_create
 * @param block_num  Block number to write (0-indexed)
 * @param data       Data to write, must be at least BLOCK_SIZE bytes
 * @return 0 on success, -1 on failure (errno set)
 *
 * Errors:
 *   EINVAL - block_num >= dev->num_blocks
 *   EIO    - Write error from underlying file
 *
 * Postcondition: block_num contains the data, synced to disk
 */
int write_block(BlockDevice* dev, uint64_t block_num, const void* data);
/**
 * Close and free a block device.
 * Syncs any pending writes before closing.
 *
 * @param dev BlockDevice to close (may be NULL)
 */
void block_device_close(BlockDevice* dev);
```
### Superblock Operations
```c
// 03_superblock.h
/**
 * Initialize a superblock with calculated layout.
 *
 * @param sb            Superblock to initialize
 * @param total_blocks  Total filesystem blocks
 * @param total_inodes  Total inodes to allocate
 * @param layout        Calculated layout from layout_calculate
 *
 * Postcondition: sb has valid magic, version, all region pointers set,
 *                free counts match layout, timestamps set to now
 */
void superblock_init(Superblock* sb, uint64_t total_blocks, 
                     uint64_t total_inodes, const LayoutResult* layout);
/**
 * Validate a superblock read from disk.
 *
 * @param sb Superblock to validate
 * @return 0 if valid, -1 if invalid (errno set)
 *
 * Errors:
 *   EINVAL - magic mismatch
 *   EINVAL - version > FS_VERSION
 *   EINVAL - checksum mismatch
 *   EINVAL - impossible geometry (e.g., data_start >= total_blocks)
 */
int superblock_validate(const Superblock* sb);
/**
 * Calculate CRC64 checksum of superblock.
 * Checksum field itself is treated as zero during calculation.
 *
 * @param sb Superblock to checksum
 * @return CRC64 checksum
 */
uint64_t superblock_checksum(const Superblock* sb);
```
### Bitmap Operations
```c
// 05_bitmap.h
#define BITS_PER_BLOCK (BLOCK_SIZE * 8)  // 32768 bits per 4KB block
/**
 * Allocate a block from the block bitmap.
 * Finds the first free bit, sets it, and returns the corresponding block number.
 *
 * @param dev BlockDevice containing the filesystem
 * @param sb  Superblock with bitmap location info
 * @param bitmap_buffer Temporary buffer of BLOCK_SIZE bytes (caller-allocated)
 * @return Allocated block number, or 0 on failure
 *
 * Errors (in errno):
 *   ENOSPC - No free blocks
 *   EIO   - Read/write error
 *
 * Postcondition: Bit is set in bitmap, sb->free_blocks is decremented
 *                (caller must write superblock)
 *
 * Note: Returns 0 on failure. Block 0 (superblock) is never a valid data block.
 */
uint64_t block_bitmap_alloc(BlockDevice* dev, Superblock* sb, void* bitmap_buffer);
/**
 * Free a block in the block bitmap.
 *
 * @param dev BlockDevice containing the filesystem
 * @param sb  Superblock with bitmap location info
 * @param block_num Block number to free (must be in data region)
 * @param bitmap_buffer Temporary buffer of BLOCK_SIZE bytes
 * @return 0 on success, -1 on failure
 *
 * Errors:
 *   EINVAL - block_num < sb->data_start or >= sb->total_blocks
 *   EINVAL - block_num already free (double-free detection)
 *   EIO   - Read/write error
 */
int block_bitmap_free(BlockDevice* dev, Superblock* sb, uint64_t block_num, 
                      void* bitmap_buffer);
/**
 * Allocate an inode from the inode bitmap.
 *
 * @param dev BlockDevice containing the filesystem
 * @param sb  Superblock with bitmap location info
 * @param bitmap_buffer Temporary buffer of BLOCK_SIZE bytes
 * @return Allocated inode number (1-indexed), or 0 on failure
 *
 * Errors (in errno):
 *   ENOSPC - No free inodes
 *   EIO   - Read/write error
 *
 * Note: Inode numbers are 1-indexed. Inode 0 is never valid.
 */
uint64_t inode_bitmap_alloc(BlockDevice* dev, Superblock* sb, void* bitmap_buffer);
/**
 * Free an inode in the inode bitmap.
 *
 * @param dev BlockDevice containing the filesystem
 * @param sb  Superblock with bitmap location info
 * @param inode_num Inode number to free (1-indexed)
 * @param bitmap_buffer Temporary buffer of BLOCK_SIZE bytes
 * @return 0 on success, -1 on failure
 *
 * Errors:
 *   EINVAL - inode_num < 1 or > sb->total_inodes
 *   EINVAL - inode_num already free
 *   EIO   - Read/write error
 */
int inode_bitmap_free(BlockDevice* dev, Superblock* sb, uint64_t inode_num,
                      void* bitmap_buffer);
```
### Layout Calculation
```c
// 07_layout.h
/**
 * Calculate filesystem layout from parameters.
 *
 * @param total_blocks    Total blocks in filesystem
 * @param total_inodes    Desired inode count
 * @param journal_blocks  Desired journal size in blocks
 * @return LayoutResult with all region positions calculated
 *
 * Algorithm:
 *   1. Block bitmap: ceil(total_blocks / BITS_PER_BLOCK) blocks
 *   2. Inode bitmap: ceil(total_inodes / BITS_PER_BLOCK) blocks
 *   3. Inode table: total_inodes blocks (1 inode per block in this design)
 *   4. Journal: journal_blocks blocks
 *   5. Data: remaining blocks
 *
 * Invariant: data_start + data_blocks <= total_blocks
 */
LayoutResult layout_calculate(uint64_t total_blocks, uint64_t total_inodes,
                              uint64_t journal_blocks);
```
### mkfs Operations
```c
// 09_mkfs.h
typedef struct {
    uint64_t total_blocks;    // 0 = auto-calculate from size_mb
    uint64_t total_inodes;    // 0 = auto-calculate (total_blocks / 16)
    uint64_t journal_blocks;  // 0 = auto-calculate (~1.5% of total)
    uint64_t size_mb;         // Used if total_blocks == 0
} MkfsOptions;
/**
 * Create a filesystem on a disk image.
 *
 * @param path    Path to disk image file (created if doesn't exist)
 * @param options Filesystem parameters
 * @return 0 on success, -1 on failure (errno set)
 *
 * Postcondition:
 *   - File exists with correct size
 *   - Block 0 contains valid superblock
 *   - Block bitmap has metadata blocks marked used
 *   - Inode bitmap has inode 1 marked used
 *   - Inode table has root directory inode initialized
 *   - Data block 0 contains root directory with . and ..
 *   - Journal region is zeroed
 */
int mkfs(const char* path, const MkfsOptions* options);
```
### Verification
```c
// 11_verify.h
/**
 * Verify filesystem consistency.
 *
 * @param path Path to disk image
 * @return 0 if consistent, number of errors found if inconsistent
 *
 * Checks:
 *   1. Superblock magic and version valid
 *   2. Checksum correct
 *   3. Geometry makes sense (data_start < total_blocks)
 *   4. Free block count matches bitmap
 *   5. Free inode count matches bitmap
 *   6. Root directory inode exists and is a directory
 *   7. Root directory has . and .. entries pointing to inode 1
 */
int verify_filesystem(const char* path);
```
---
## Algorithm Specification
### Block Bitmap Allocation
**Input:** BlockDevice `dev`, Superblock `sb`, buffer `bitmap_buffer`
**Output:** Allocated block number (data region), or 0 on failure
**Invariant:** After return, exactly one bit is set that was previously clear
```
ALGORITHM block_bitmap_alloc(dev, sb, bitmap_buffer):
  bitmap_start ← sb.block_bitmap_start
  bitmap_blocks ← sb.block_bitmap_blocks
  data_start ← sb.data_start
  FOR each bitmap_block_idx FROM 0 TO bitmap_blocks - 1:
    bitmap_block ← bitmap_start + bitmap_block_idx
    IF read_block(dev, bitmap_block, bitmap_buffer) ≠ 0:
      RETURN 0  // errno set by read_block
    bits ← cast(bitmap_buffer to uint64_t*)
    // Scan 64-bit words for a free bit
    FOR each word_idx FROM 0 TO (BLOCK_SIZE / 8) - 1:
      IF bits[word_idx] ≠ ~0ULL:  // Not all 1s = has free bit
        // Find first 0 bit using count trailing zeros
        inverted ← ~bits[word_idx]
        bit_pos ← count_trailing_zeros(inverted)  // __builtin_ctzll
        IF bit_pos < 64:
          // Mark as allocated
          bits[word_idx] ← bits[word_idx] OR (1ULL << bit_pos)
          // Write back bitmap block
          IF write_block(dev, bitmap_block, bitmap_buffer) ≠ 0:
            RETURN 0
          // Calculate physical block number
          bit_index ← (bitmap_block_idx * BITS_PER_BLOCK) + 
                      (word_idx * 64) + bit_pos
          phys_block ← data_start + bit_index
          // Validate range
          IF phys_block ≥ sb.total_blocks:
            // Shouldn't happen with correct layout, but fail safe
            bits[word_idx] ← bits[word_idx] AND ~(1ULL << bit_pos)
            write_block(dev, bitmap_block, bitmap_buffer)
            errno ← EINVAL
            RETURN 0
          sb.free_blocks ← sb.free_blocks - 1
          RETURN phys_block
  errno ← ENOSPC
  RETURN 0
```
**Hardware Soul - Branch Prediction:**
- Outer loop: predictable iteration through bitmap blocks
- Inner loop: predictable until free bit found
- `bits[word_idx] ≠ ~0ULL`: 50/50 early in filesystem, predictable later
- `__builtin_ctzll`: compiled to single instruction (TZCNT on x86)
{{DIAGRAM:tdd-diag-m1-02}}
### Block Bitmap Free
**Input:** BlockDevice `dev`, Superblock `sb`, block number `block_num`, buffer
**Output:** 0 on success, -1 on failure
**Invariant:** After return, the bit for `block_num` is clear
```
ALGORITHM block_bitmap_free(dev, sb, block_num, bitmap_buffer):
  data_start ← sb.data_start
  // Validate block is in data region
  IF block_num < data_start OR block_num ≥ sb.total_blocks:
    errno ← EINVAL
    RETURN -1
  // Calculate bitmap position
  data_index ← block_num - data_start
  bitmap_block_idx ← data_index / BITS_PER_BLOCK
  bit_in_bitmap ← data_index % BITS_PER_BLOCK
  // Validate bitmap block index
  IF bitmap_block_idx ≥ sb.block_bitmap_blocks:
    errno ← EINVAL
    RETURN -1
  bitmap_block ← sb.block_bitmap_start + bitmap_block_idx
  // Read bitmap block
  IF read_block(dev, bitmap_block, bitmap_buffer) ≠ 0:
    RETURN -1
  bits ← cast(bitmap_buffer to uint64_t*)
  word_idx ← bit_in_bitmap / 64
  bit_pos ← bit_in_bitmap % 64
  // Check for double-free
  IF (bits[word_idx] AND (1ULL << bit_pos)) = 0:
    errno ← EINVAL  // Already free
    RETURN -1
  // Clear the bit
  bits[word_idx] ← bits[word_idx] AND ~(1ULL << bit_pos)
  // Write back
  IF write_block(dev, bitmap_block, bitmap_buffer) ≠ 0:
    RETURN -1
  sb.free_blocks ← sb.free_blocks + 1
  RETURN 0
```
### Layout Calculation
**Input:** `total_blocks`, `total_inodes`, `journal_blocks`
**Output:** `LayoutResult` with all positions
```
ALGORITHM layout_calculate(total_blocks, total_inodes, journal_blocks):
  result ← zero-initialized LayoutResult
  // Block 0: superblock (implicitly starts at 0)
  current ← 1
  // Block bitmap: track all blocks including metadata
  bits_needed ← total_blocks
  result.block_bitmap_blocks ← ceil(bits_needed / BITS_PER_BLOCK)
  result.block_bitmap_start ← current
  current ← current + result.block_bitmap_blocks
  // Inode bitmap
  result.inode_bitmap_blocks ← ceil(total_inodes / BITS_PER_BLOCK)
  result.inode_bitmap_start ← current
  current ← current + result.inode_bitmap_blocks
  // Inode table (1 inode per block = 4096 bytes per inode)
  result.inode_table_blocks ← total_inodes
  result.inode_table_start ← current
  current ← current + result.inode_table_blocks
  // Journal
  result.journal_blocks ← journal_blocks
  result.journal_start ← current
  current ← current + journal_blocks
  // Data region
  result.data_start ← current
  result.data_blocks ← total_blocks - current
  result.metadata_blocks ← current
  // Validation
  IF result.data_blocks ≤ 0:
    // Error: metadata consumes all space
    RETURN error_result
  RETURN result
```

![Superblock Memory Layout](./diagrams/tdd-diag-m1-03.svg)

### mkfs Implementation
**Input:** `path`, `options`
**Output:** 0 on success, -1 on failure
```
ALGORITHM mkfs(path, options):
  // Resolve auto-calculate options
  IF options.total_blocks = 0:
    total_blocks ← (options.size_mb * 1024 * 1024) / BLOCK_SIZE
  ELSE:
    total_blocks ← options.total_blocks
  IF options.total_inodes = 0:
    total_inodes ← total_blocks / 16  // 1 inode per 16 blocks
    total_inodes ← max(total_inodes, 64)  // Minimum
  ELSE:
    total_inodes ← options.total_inodes
  IF options.journal_blocks = 0:
    journal_blocks ← total_blocks / 64  // ~1.5%
    journal_blocks ← clamp(journal_blocks, 128, 4096)
  ELSE:
    journal_blocks ← options.journal_blocks
  // Calculate layout
  layout ← layout_calculate(total_blocks, total_inodes, journal_blocks)
  IF layout invalid:
    errno ← EINVAL
    RETURN -1
  // Create block device
  dev ← block_device_create(path, total_blocks)
  IF dev = NULL:
    RETURN -1
  // Allocate working buffers
  superblock ← allocate(BLOCK_SIZE)
  bitmap_buffer ← allocate(BLOCK_SIZE)
  zero_buffer ← allocate(BLOCK_SIZE)
  memset(zero_buffer, 0, BLOCK_SIZE)
  // Initialize superblock
  superblock_init(superblock, total_blocks, total_inodes, &layout)
  // Write superblock to block 0
  IF write_block(dev, 0, superblock) ≠ 0:
    GOTO cleanup_error
  // Initialize block bitmap: mark metadata blocks as used
  // Metadata = blocks 0 through (data_start - 1)
  FOR each block FROM 0 TO layout.data_start - 1:
    bitmap_block_idx ← block / BITS_PER_BLOCK
    bit_pos ← block % BITS_PER_BLOCK
    word_idx ← bit_pos / 64
    word_bit ← bit_pos % 64
    // Read correct bitmap block if needed
    IF bitmap_block_idx changed OR first iteration:
      IF NOT first iteration:
        write_block(dev, layout.block_bitmap_start + prev_idx, bitmap_buffer)
      memset(bitmap_buffer, 0, BLOCK_SIZE)
      prev_idx ← bitmap_block_idx
    uint64_t* bits ← cast(bitmap_buffer to uint64_t*)
    bits[word_idx] ← bits[word_idx] OR (1ULL << word_bit)
  // Write final bitmap block
  write_block(dev, layout.block_bitmap_start + prev_idx, bitmap_buffer)
  // Zero remaining block bitmap blocks
  FOR each i FROM prev_idx + 1 TO layout.block_bitmap_blocks - 1:
    write_block(dev, layout.block_bitmap_start + i, zero_buffer)
  // Initialize inode bitmap: mark inode 1 as used (root)
  memset(bitmap_buffer, 0, BLOCK_SIZE)
  uint64_t* bits ← cast(bitmap_buffer to uint64_t*)
  bits[0] ← 0x0000000000000002ULL  // Bit 1 set (inode 1 is used)
  write_block(dev, layout.inode_bitmap_start, bitmap_buffer)
  // Zero remaining inode bitmap blocks
  FOR each i FROM 1 TO layout.inode_bitmap_blocks - 1:
    write_block(dev, layout.inode_bitmap_start + i, zero_buffer)
  // Zero inode table
  FOR each i FROM 0 TO layout.inode_table_blocks - 1:
    write_block(dev, layout.inode_table_start + i, zero_buffer)
  // Create root directory
  // Allocate first data block for root directory entries
  root_data_block ← layout.data_start
  // Mark it in block bitmap
  block_bitmap_mark_used(dev, &layout, root_data_block, bitmap_buffer)
  // Create root inode (inode 1)
  root_inode_buffer ← allocate(BLOCK_SIZE)
  memset(root_inode_buffer, 0, BLOCK_SIZE)
  root_inode ← cast(root_inode_buffer to Inode*)
  root_inode.mode ← S_IFDIR | 0755  // Directory with rwxr-xr-x
  root_inode.uid ← 0
  root_inode.gid ← 0
  root_inode.link_count ← 2  // . and parent entry (itself for root)
  root_inode.size ← BLOCK_SIZE
  root_inode.blocks ← 1
  root_inode.atime ← current_time()
  root_inode.mtime ← current_time()
  root_inode.ctime ← current_time()
  root_inode.direct[0] ← root_data_block
  // Write root inode (inode 1 goes to inode_table_start + 0)
  write_block(dev, layout.inode_table_start, root_inode_buffer)
  // Create root directory entries (. and ..)
  dir_buffer ← allocate(BLOCK_SIZE)
  memset(dir_buffer, 0, BLOCK_SIZE)
  // Entry 1: .
  entry1 ← cast(dir_buffer to DirEntry*)
  entry1.inode ← 1
  entry1.rec_len ← DIR_ENTRY_SIZE
  entry1.name_len ← 1
  entry1.file_type ← DT_DIR
  entry1.name[0] ← '.'
  // Entry 2: ..
  entry2 ← cast(dir_buffer + DIR_ENTRY_SIZE to DirEntry*)
  entry2.inode ← 1  // Root's parent is itself
  entry2.rec_len ← BLOCK_SIZE - DIR_ENTRY_SIZE  // Rest of block
  entry2.name_len ← 2
  entry2.file_type ← DT_DIR
  entry2.name[0] ← '.'
  entry2.name[1] ← '.'
  write_block(dev, root_data_block, dir_buffer)
  // Update superblock with final free count
  superblock.free_blocks ← layout.data_blocks - 1  // Root dir used one
  superblock.free_inodes ← total_inodes - 1  // Root inode used
  superblock.write_time ← current_time()
  superblock.checksum ← superblock_checksum(superblock)
  write_block(dev, 0, superblock)
  // Initialize journal region (zero it)
  FOR each i FROM 0 TO layout.journal_blocks - 1:
    write_block(dev, layout.journal_start + i, zero_buffer)
  // Sync everything to disk
  fsync(dev.fd)
  // Cleanup
  free(superblock)
  free(bitmap_buffer)
  free(zero_buffer)
  free(root_inode_buffer)
  free(dir_buffer)
  block_device_close(dev)
  RETURN 0
cleanup_error:
  free(all buffers)
  block_device_close(dev)
  RETURN -1
```

![Bitmap Block Structure](./diagrams/tdd-diag-m1-04.svg)

---
## Error Handling Matrix
| Error | errno | Detected By | Recovery | User-Visible Message |
|-------|-------|-------------|----------|---------------------|
| File creation failed | EACCES | block_device_create | Return NULL, don't create image | "Permission denied: %s" |
| No space for image | ENOSPC | block_device_create (ftruncate) | Return NULL, remove partial file | "No space for %lu byte image" |
| Invalid block number | EINVAL | read_block/write_block | Return -1, don't modify disk | Internal only (log) |
| Read I/O error | EIO | read_block | Return -1 | "Read error at block %lu" |
| Write I/O error | EIO | write_block | Return -1, state may be inconsistent | "Write error at block %lu" |
| No free blocks | ENOSPC | block_bitmap_alloc | Return 0 | "Filesystem full" |
| Double-free block | EINVAL | block_bitmap_free | Return -1, bitmap unchanged | Internal error (log) |
| No free inodes | ENOSPC | inode_bitmap_alloc | Return 0 | "No free inodes" |
| Invalid superblock magic | EINVAL | superblock_validate | Return -1 | "Not a valid filesystem" |
| Superblock checksum bad | EINVAL | superblock_validate | Return -1 | "Superblock corrupted" |
| Geometry impossible | EINVAL | layout_calculate | Return error result | "Filesystem too small for metadata" |
| Block out of range | EINVAL | block_bitmap_free | Return -1 | Internal error (log) |
---
## Implementation Sequence with Checkpoints
### Phase 1: Block Device Abstraction (2-3 hours)
**Files:** `01_block.h`, `02_block.c`
**Implementation Steps:**
1. Define `BLOCK_SIZE` constant and `BlockDevice` struct
2. Implement `block_device_create`:
   - `open()` with `O_RDWR | O_CREAT | O_TRUNC`
   - `ftruncate()` to pre-allocate size
   - Store `fd`, `size`, `num_blocks`
3. Implement `read_block`:
   - Bounds check `block_num < num_blocks`
   - `lseek()` to `block_num * BLOCK_SIZE`
   - `read()` full block, zero-fill if short
4. Implement `write_block`:
   - Bounds check
   - `lseek()` and `write()`
   - `fsync()` for durability (mkfs phase only)
5. Implement `block_device_close`
**Checkpoint 1:** 
```
$ make test_block
$ ./test_block
Creating 1MB test image...
Writing pattern to block 0: OK
Reading back block 0: OK (pattern matches)
Writing to block 255 (last block): OK
Reading from block 256: FAIL (out of bounds) - expected
All block device tests passed!
```
### Phase 2: Superblock Structure (1-2 hours)
**Files:** `03_superblock.h`, `03_superblock.c`
**Implementation Steps:**
1. Define full `Superblock` struct with `__attribute__((packed))`
2. Add `static_assert` for size == 4096
3. Implement `superblock_init` to populate all fields
4. Implement CRC64 checksum function (use standard polynomial 0x42F0E1EBA9EA3693)
5. Implement `superblock_validate` checking magic, version, checksum
**Checkpoint 2:**
```
$ make test_superblock
$ ./test_superblock
Creating superblock...
  Magic: 0x46534D4B (correct)
  Version: 1
  Total blocks: 25600
  Checksum: 0x...
Validating: OK
Corrupting magic: Validation FAIL (expected)
Corrupting checksum: Validation FAIL (expected)
All superblock tests passed!
```
### Phase 3: Block Bitmap Allocator (2-3 hours)
**Files:** `05_bitmap.h`, `06_bitmap.c`
**Implementation Steps:**
1. Define `BITS_PER_BLOCK` constant
2. Implement `block_bitmap_alloc`:
   - Iterate through bitmap blocks
   - Scan 64-bit words with `__builtin_ffsll` on inverted value
   - Set bit, write back, return block number
3. Implement `block_bitmap_free`:
   - Calculate bitmap position from block number
   - Check for double-free (bit already clear)
   - Clear bit, write back
**Checkpoint 3:**
```
$ make test_bitmap
$ ./test_bitmap
Allocating blocks...
  Block 1: 1024 (first data block) - OK
  Block 2: 1025 - OK
  Block 3: 1026 - OK
Freeing block 1025: OK
Re-allocating: 1025 (got same block back) - OK
Double-free test: FAIL (expected)
Allocating until full: 64738 blocks allocated, then ENOSPC (expected)
All bitmap tests passed!
```
### Phase 4: Inode Bitmap Allocator (1-2 hours)
**Files:** Update `05_bitmap.h`, `06_bitmap.c`
**Implementation Steps:**
1. Implement `inode_bitmap_alloc` (same logic, different region)
2. Implement `inode_bitmap_free`
3. Note: inode numbers are 1-indexed, so bit 0 corresponds to inode 1
**Checkpoint 4:**
```
$ make test_bitmap
$ ./test_bitmap
... block bitmap tests ...
Inode allocation:
  Inode 1: OK
  Inode 2: OK
  ...
  Inode 64: OK
Freeing inode 32: OK
Re-allocating: 32 (got same back) - OK
All inode bitmap tests passed!
```
### Phase 5: Layout Calculation (1-2 hours)
**Files:** `07_layout.h`, `08_layout.c`
**Implementation Steps:**
1. Define `LayoutResult` struct
2. Implement `layout_calculate`:
   - Calculate each region size from inputs
   - Chain regions sequentially starting at block 1
   - Return error if data_blocks <= 0
**Checkpoint 5:**
```
$ make test_layout
$ ./test_layout
Layout for 25600 blocks, 4096 inodes, 1024 journal blocks:
  Block bitmap:    blocks 1-1 (1 block for 25600 bits)
  Inode bitmap:    blocks 2-2 (1 block for 4096 bits)
  Inode table:     blocks 3-4098 (4096 blocks)
  Journal:         blocks 4099-5122 (1024 blocks)
  Data:            blocks 5123-25599 (20477 blocks)
  Metadata overhead: 20.0%
  Validation: OK
Layout for 100 blocks, 10 inodes:
  Data blocks: 88
  Validation: OK
Layout for 10 blocks (too small):
  Validation: FAIL - metadata exceeds space (expected)
All layout tests passed!
```
### Phase 6: mkfs Main Function (2-3 hours)
**Files:** `09_mkfs.h`, `09_mkfs.c`, `13_mkfs_main.c`
**Implementation Steps:**
1. Implement full `mkfs` function per algorithm
2. Create root directory with `.` and `..` entries
3. Implement CLI in `mkfs_main.c` with argument parsing
4. Print summary after successful creation
**Checkpoint 6:**
```
$ make mkfs
$ ./mkfs test.img 100
Creating 100MB filesystem...
Filesystem created successfully:
  Total blocks: 25600 (100.00 MB)
  Total inodes: 1600
  Block bitmap: 1 block
  Inode bitmap: 1 block
  Inode table:  1600 blocks (6.25 MB)
  Journal:      400 blocks (1.56 MB)
  Data blocks:  23598 (92.18 MB)
  Metadata overhead: 7.82%
Root directory created at inode 1
$ ls -la test.img
-rw-r--r-- 1 user user 104857600 Mar  4 10:00 test.img
```
### Phase 7: Verification and Testing (1-2 hours)
**Files:** `11_verify.h`, `12_verify.c`, `tests/test_mkfs.c`
**Implementation Steps:**
1. Implement `verify_filesystem`:
   - Read and validate superblock
   - Count free bits in bitmaps, compare to superblock counts
   - Verify root inode exists and is directory
   - Verify root directory has `.` and `..`
2. Write comprehensive tests
**Checkpoint 7:**
```
$ make verify
$ ./mkfs test.img 100
$ ./verify test.img
Verifying filesystem...
  [OK] Superblock magic: 0x46534D4B
  [OK] Superblock checksum
  [OK] Geometry: data_start < total_blocks
  [OK] Free block count: 23597 (bitmap matches superblock)
  [OK] Free inode count: 1599 (bitmap matches superblock)
  [OK] Root inode (1) exists
  [OK] Root inode is directory
  [OK] Root directory has . entry (inode 1)
  [OK] Root directory has .. entry (inode 1)
Verification complete: 0 errors
$ ./test_mkfs
Running mkfs tests...
  test_small_fs (1MB): OK
  test_medium_fs (100MB): OK
  test_large_fs (1GB): OK
  test_tiny_fs (1MB, minimal inodes): OK
  test_round_trip (create, verify, re-read): OK
All tests passed!
```
---
## Test Specification
### test_block.c
```c
// Test: Create and basic I/O
void test_block_create_and_io() {
    BlockDevice* dev = block_device_create("test_block.img", 256);
    ASSERT(dev != NULL);
    ASSERT(dev->num_blocks == 256);
    ASSERT(dev->size == 256 * BLOCK_SIZE);
    char write_buf[BLOCK_SIZE];
    memset(write_buf, 0xAB, BLOCK_SIZE);
    ASSERT(write_block(dev, 0, write_buf) == 0);
    char read_buf[BLOCK_SIZE];
    ASSERT(read_block(dev, 0, read_buf) == 0);
    ASSERT(memcmp(write_buf, read_buf, BLOCK_SIZE) == 0);
    block_device_close(dev);
    unlink("test_block.img");
}
// Test: Bounds checking
void test_block_bounds() {
    BlockDevice* dev = block_device_create("test_bounds.img", 10);
    char buf[BLOCK_SIZE];
    // Valid
    ASSERT(read_block(dev, 9, buf) == 0);
    // Invalid
    errno = 0;
    ASSERT(read_block(dev, 10, buf) == -1);
    ASSERT(errno == EINVAL);
    errno = 0;
    ASSERT(write_block(dev, 10, buf) == -1);
    ASSERT(errno == EINVAL);
    block_device_close(dev);
    unlink("test_bounds.img");
}
// Test: Last block
void test_block_last_block() {
    BlockDevice* dev = block_device_create("test_last.img", 100);
    char buf[BLOCK_SIZE];
    memset(buf, 'Z', BLOCK_SIZE);
    ASSERT(write_block(dev, 99, buf) == 0);
    memset(buf, 0, BLOCK_SIZE);
    ASSERT(read_block(dev, 99, buf) == 0);
    for (int i = 0; i < BLOCK_SIZE; i++) {
        ASSERT(buf[i] == 'Z');
    }
    block_device_close(dev);
    unlink("test_last.img");
}
```
### test_bitmap.c
```c
// Test: Sequential allocation
void test_bitmap_sequential() {
    BlockDevice* dev = block_device_create("test_seq.img", 10000);
    Superblock sb = create_test_superblock(dev, 10000);
    char buffer[BLOCK_SIZE];
    // Allocate 100 blocks
    uint64_t prev = 0;
    for (int i = 0; i < 100; i++) {
        uint64_t block = block_bitmap_alloc(dev, &sb, buffer);
        ASSERT(block != 0);
        ASSERT(block >= sb.data_start);
        if (prev != 0) ASSERT(block > prev);  // Sequential
        prev = block;
    }
    ASSERT(sb.free_blocks == sb_total_data - 100);
    block_device_close(dev);
    unlink("test_seq.img");
}
// Test: Free and reallocate
void test_bitmap_free_realloc() {
    BlockDevice* dev = block_device_create("test_free.img", 1000);
    Superblock sb = create_test_superblock(dev, 1000);
    char buffer[BLOCK_SIZE];
    uint64_t b1 = block_bitmap_alloc(dev, &sb, buffer);
    uint64_t b2 = block_bitmap_alloc(dev, &sb, buffer);
    uint64_t b3 = block_bitmap_alloc(dev, &sb, buffer);
    // Free middle
    ASSERT(block_bitmap_free(dev, &sb, b2, buffer) == 0);
    // Reallocate - should get same block
    uint64_t b4 = block_bitmap_alloc(dev, &sb, buffer);
    ASSERT(b4 == b2);
    block_device_close(dev);
    unlink("test_free.img");
}
// Test: Double-free detection
void test_bitmap_double_free() {
    BlockDevice* dev = block_device_create("test_dfree.img", 1000);
    Superblock sb = create_test_superblock(dev, 1000);
    char buffer[BLOCK_SIZE];
    uint64_t block = block_bitmap_alloc(dev, &sb, buffer);
    ASSERT(block_bitmap_free(dev, &sb, block, buffer) == 0);
    errno = 0;
    ASSERT(block_bitmap_free(dev, &sb, block, buffer) == -1);
    ASSERT(errno == EINVAL);
    block_device_close(dev);
    unlink("test_dfree.img");
}
// Test: Exhaustion
void test_bitmap_exhaustion() {
    // Create tiny filesystem
    BlockDevice* dev = block_device_create("test_exhaust.img", 100);
    // Layout leaves ~80 data blocks
    Superblock sb = create_test_superblock(dev, 100);
    char buffer[BLOCK_SIZE];
    int count = 0;
    while (block_bitmap_alloc(dev, &sb, buffer) != 0) {
        count++;
    }
    ASSERT(errno == ENOSPC);
    printf("Allocated %d blocks before exhaustion\n", count);
    block_device_close(dev);
    unlink("test_exhaust.img");
}
```
### test_mkfs.c
```c
// Test: Basic mkfs
void test_mkfs_basic() {
    MkfsOptions opts = {.size_mb = 10};
    ASSERT(mkfs("test_basic.img", &opts) == 0);
    // Verify file exists with correct size
    struct stat st;
    ASSERT(stat("test_basic.img", &st) == 0);
    ASSERT(st.st_size == 10 * 1024 * 1024);
    // Verify filesystem
    ASSERT(verify_filesystem("test_basic.img") == 0);
    unlink("test_basic.img");
}
// Test: Round-trip
void test_mkfs_roundtrip() {
    MkfsOptions opts = {.size_mb = 50};
    ASSERT(mkfs("test_roundtrip.img", &opts) == 0);
    // Open and read superblock
    int fd = open("test_roundtrip.img", O_RDONLY);
    Superblock sb;
    read(fd, &sb, sizeof(sb));
    close(fd);
    ASSERT(sb.magic == FS_MAGIC);
    ASSERT(sb.version == FS_VERSION);
    ASSERT(sb.total_blocks == (50 * 1024 * 1024) / BLOCK_SIZE);
    // Verify free counts make sense
    ASSERT(sb.free_blocks > 0);
    ASSERT(sb.free_inodes > 0);
    ASSERT(sb.free_inodes == sb.total_inodes - 1);  // Root used one
    unlink("test_roundtrip.img");
}
// Test: Root directory
void test_mkfs_root_dir() {
    MkfsOptions opts = {.size_mb = 10};
    ASSERT(mkfs("test_root.img", &opts) == 0);
    BlockDevice* dev = block_device_create("test_root.img", 0);
    // Read-only open, size from file
    Superblock sb;
    read_block(dev, 0, &sb);
    // Read root inode
    char inode_buf[BLOCK_SIZE];
    read_block(dev, sb.inode_table_start, inode_buf);
    Inode* root = (Inode*)inode_buf;
    ASSERT((root->mode & S_IFDIR) != 0);
    ASSERT(root->link_count == 2);
    ASSERT(root->size == BLOCK_SIZE);
    ASSERT(root->direct[0] == sb.data_start);
    // Read root directory entries
    char dir_buf[BLOCK_SIZE];
    read_block(dev, root->direct[0], dir_buf);
    DirEntry* entries = (DirEntry*)dir_buf;
    ASSERT(entries[0].inode == 1);
    ASSERT(strcmp(entries[0].name, ".") == 0);
    ASSERT(entries[1].inode == 1);
    ASSERT(strcmp(entries[1].name, "..") == 0);
    block_device_close(dev);
    unlink("test_root.img");
}
// Test: Custom parameters
void test_mkfs_custom_params() {
    MkfsOptions opts = {
        .total_blocks = 50000,
        .total_inodes = 1000,
        .journal_blocks = 500
    };
    ASSERT(mkfs("test_custom.img", &opts) == 0);
    Superblock sb;
    int fd = open("test_custom.img", O_RDONLY);
    read(fd, &sb, sizeof(sb));
    close(fd);
    ASSERT(sb.total_blocks == 50000);
    ASSERT(sb.total_inodes == 1000);
    ASSERT(sb.journal_blocks == 500);
    unlink("test_custom.img");
}
```
---
## Performance Targets
| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| `read_block` (cached) | < 1 μs | `time ./bench_block read 10000` on hot file |
| `read_block` (NVMe) | < 25 μs | Direct I/O, cold cache |
| `read_block` (HDD) | < 10 ms | Direct I/O, random access |
| `write_block` (cached) | < 2 μs | Same as read |
| `block_bitmap_alloc` | < 50 μs | Average over 10000 allocations |
| `inode_bitmap_alloc` | < 50 μs | Same |
| `mkfs 100MB` | < 100 ms | `time ./mkfs test.img 100` |
| `mkfs 1GB` | < 1 s | `time ./mkfs test.img 1024` |
| `verify_filesystem 100MB` | < 200 ms | `time ./verify test.img` |
**Hardware Soul - Performance Notes:**
1. **Sequential writes during mkfs**: The `write_block` calls in mkfs are sequential through the block address space. This is optimal for both HDDs (no seek) and SSDs (efficient page programming). The OS may merge adjacent writes.
2. **Bitmap scan performance**: Scanning for free bits uses `__builtin_ffsll` which compiles to a single instruction. The bottleneck is reading bitmap blocks from disk, not the bit manipulation.
3. **Cache effects**: First `read_block` after `block_device_create` is cold (disk I/O). Subsequent reads to the same block may be served from OS page cache. For mkfs, everything is cold (newly created file).
4. **fsync overhead**: Each `write_block` calls `fsync()`. This is correct for mkfs but slow. A production filesystem would batch writes and sync once.
---
## Diagrams

![Block Allocation Flow](./diagrams/tdd-diag-m1-05.svg)


![mkfs Initialization Sequence](./diagrams/tdd-diag-m1-06.svg)


![Root Directory Entry Layout](./diagrams/tdd-diag-m1-07.svg)


![Filesystem Verification State Machine](./diagrams/tdd-diag-m1-08.svg)


![Bitmap Index Calculation](./diagrams/tdd-diag-m1-09.svg)


![Block Write Data Flow](./diagrams/tdd-diag-m1-10.svg)

---
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: filesystem-m2 -->
# Technical Design Specification: Inode Management
## Module Charter
This module implements the inode structure — the bridge between logical files and physical storage blocks. An inode contains all metadata about a file (type, permissions, ownership, size, timestamps, link count) plus a block pointer scheme enabling files from 0 bytes to ~1GB. The pointer scheme uses 12 direct pointers for small files (first 48KB), one single-indirect pointer for medium files (48KB to ~2MB), and one double-indirect pointer for large files (up to ~1GB). This module provides block offset resolution (translating file byte positions to physical block numbers), lazy block allocation during writes, and complete deallocation including recursive freeing of indirect blocks. It does NOT interpret directory entry formats, implement file data I/O operations, or handle path resolution. The core invariant: every non-zero block pointer in an inode points to an allocated block in the bitmap, and the inode's size field accurately reflects the logical file size regardless of sparse holes.
---
## File Structure
```
filesystem/
├── 14_inode.h           # Inode struct, constants, function declarations
├── 15_inode.c           # read_inode, write_inode, inode serialization
├── 16_block_ptr.h       # BlockLocation struct, zone calculation functions
├── 17_block_ptr.c       # locate_block, get_file_block, read_indirect_ptr
├── 18_inode_alloc.h     # inode_allocate, inode_deallocate declarations
├── 19_inode_alloc.c     # Block allocation on write, recursive free
├── 20_timestamp.h       # Timestamp update constants and functions
├── 21_timestamp.c       # update_timestamps implementation
└── tests/
    ├── test_inode_struct.c   # Inode layout and serialization tests
    ├── test_block_ptr.c      # Zone calculation and pointer resolution
    ├── test_inode_alloc.c    # Allocation and deallocation tests
    └── test_sparse.c         # Sparse file behavior tests
```
---
## Complete Data Model
### Inode Structure
The inode is a fixed 4096-byte record stored in the inode table region. Each inode is self-contained — all metadata and block pointers fit within one block.
```c
// 14_inode.h
#include <stdint.h>
#include <stdbool.h>
// File type constants (stored in high bits of mode)
#define S_IFIFO  0010000  // Named pipe
#define S_IFCHR  0020000  // Character device
#define S_IFDIR  0040000  // Directory
#define S_IFBLK  0060000  // Block device
#define S_IFREG  0100000  // Regular file
#define S_IFLNK  0120000  // Symbolic link
#define S_IFSOCK 0140000  // Socket
// Permission bits (low 12 bits of mode)
#define S_ISUID  0004000  // Set-user-ID
#define S_ISGID  0002000  // Set-group-ID
#define S_ISVTX  0001000  // Sticky bit
#define S_IRWXU  00700    // Owner: rwx
#define S_IRWXG  00070    // Group: rwx
#define S_IRWXO  00007    // Other: rwx
// Block pointer constants
#define DIRECT_POINTERS     12
#define PTRS_PER_BLOCK      512   // BLOCK_SIZE / sizeof(uint64_t)
#define INODE_SIZE          4096
typedef struct {
    // === Metadata fields (offset 0x00 - 0x47, 72 bytes) ===
    // Identification and type (offset 0x00, 2 bytes)
    uint16_t mode;          // File type (high 4 bits) + permissions (low 12 bits)
                            // Example: S_IFREG | 0644 = 0x81A4
    // Ownership (offset 0x02, 4 bytes)
    uint16_t uid;           // Owner user ID (0 = root)
    uint16_t gid;           // Owner group ID (0 = root)
    // Reference counting (offset 0x06, 2 bytes)
    uint16_t link_count;    // Number of directory entries pointing here
                            // File deleted when this reaches 0
                            // Directories: 2 + number of subdirectories
    // Reserved for alignment (offset 0x08, 2 bytes)
    uint16_t _reserved1;
    // Size tracking (offset 0x0A, 16 bytes)
    uint64_t size;          // Logical file size in bytes
    uint64_t blocks;        // Number of 512-byte blocks allocated (stat convention)
                            // Actual 4KB blocks = (blocks * 512) / 4096 = blocks / 8
    // Timestamps (offset 0x1A, 24 bytes)
    uint64_t atime;         // Access time - updated on read
    uint64_t mtime;         // Modification time - updated on write
    uint64_t ctime;         // Change time - updated on ANY metadata change
    // === Block pointers (offset 0x32 - 0x7F, 78 bytes) ===
    // Direct block pointers (offset 0x32, 96 bytes)
    // Points directly to data blocks containing file bytes
    uint64_t direct[DIRECT_POINTERS];  
    // direct[0] → file bytes 0-4095
    // direct[1] → file bytes 4096-8191
    // ...
    // direct[11] → file bytes 45056-49151
    // Total direct capacity: 12 * 4096 = 49,152 bytes (48KB)
    // Single-indirect pointer (offset 0x92, 8 bytes)
    uint64_t indirect;      // Block containing 512 data block pointers
    // Points to a 4KB block that contains uint64_t[512]
    // Each entry points to a data block
    // Capacity: 512 * 4096 = 2,097,152 bytes (2MB)
    // File range: bytes 49152 to 2,146,303
    // Double-indirect pointer (offset 0x9A, 8 bytes)
    uint64_t double_ind;    // Block containing 512 indirect block pointers
    // Points to a 4KB block that contains uint64_t[512]
    // Each entry points to an indirect block
    // Each indirect block points to 512 data blocks
    // Capacity: 512 * 512 * 4096 = 1,073,741,824 bytes (1GB)
    // File range: bytes 2,146,304 to 1,075,888,127
    // Triple-indirect pointer (offset 0xA2, 8 bytes)
    // Reserved for future expansion - not implemented
    uint64_t triple_ind;    // Always 0 in this implementation
    // === Reserved/padding (offset 0xAA - 0xFFF, 3958 bytes) ===
    uint8_t _reserved2[3958];
    // Total: 4096 bytes exactly
} __attribute__((packed)) Inode;
// Compile-time verification
static_assert(sizeof(Inode) == INODE_SIZE, "Inode must be exactly 4096 bytes");
static_assert(offsetof(Inode, direct) == 0x32, "direct array offset mismatch");
static_assert(offsetof(Inode, indirect) == 0x92, "indirect offset mismatch");
static_assert(offsetof(Inode, double_ind) == 0x9A, "double_ind offset mismatch");
```
**Byte Offset Table:**
| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0x00 | 2 | mode | File type + permissions |
| 0x02 | 2 | uid | Owner user ID |
| 0x04 | 2 | gid | Owner group ID |
| 0x06 | 2 | link_count | Hard link count |
| 0x08 | 2 | _reserved1 | Alignment padding |
| 0x0A | 8 | size | Logical file size (bytes) |
| 0x12 | 8 | blocks | 512-byte blocks allocated |
| 0x1A | 8 | atime | Access timestamp |
| 0x22 | 8 | mtime | Modification timestamp |
| 0x2A | 8 | ctime | Change timestamp |
| 0x32 | 96 | direct[12] | Direct block pointers |
| 0x92 | 8 | indirect | Single-indirect pointer |
| 0x9A | 8 | double_ind | Double-indirect pointer |
| 0xA2 | 8 | triple_ind | Reserved (always 0) |
| 0xAA | 3958 | _reserved2 | Future expansion |
| **Total** | **4096** | | |
**Hardware Soul - Cache Line Analysis:**
```
Cache Line 0 (0x00-0x3F): mode, uid, gid, link_count, reserved1, size, blocks, atime (partial)
Cache Line 1 (0x40-0x7F): atime (partial), mtime, ctime, direct[0-3]
Cache Line 2 (0x80-0xBF): direct[4-11], indirect
Cache Line 3 (0xC0-0xFF): double_ind, triple_ind, _reserved2 start
Cache Lines 4-63: _reserved2
Hot cache lines for typical operations:
- Read/Write: Lines 0-2 (metadata + all pointers) = 192 bytes
- getattr: Line 0 only (metadata) = 64 bytes
- Large file access: Lines 0-2 + indirect blocks (separate cache misses)
```
**Why Each Field Exists:**
- **mode**: Encodes file type (required for `stat()`, directory traversal) and permissions (access control)
- **uid/gid**: Ownership for permission checks
- **link_count**: Enables hard links; file only deleted when count reaches 0
- **size**: Logical end-of-file position; needed for read bounds, `stat()`
- **blocks**: Used by `du` and `stat`; tracks actual storage consumption
- **atime/mtime/ctime**: Required by POSIX; used by `make`, backup programs, `ls -l`
- **direct[12]**: Fast path for small files (95%+ of files are <48KB)
- **indirect**: Extends capacity to ~2MB without multiple indirection levels
- **double_ind**: Extends capacity to ~1GB for large files
### BlockLocation Structure
Categorizes a file offset into the appropriate pointer zone and provides indices for traversal.
```c
// 16_block_ptr.h
typedef enum {
    ZONE_DIRECT = 0,        // Direct block pointers
    ZONE_SINGLE_IND = 1,    // Single-indirect zone
    ZONE_DOUBLE_IND = 2,    // Double-indirect zone
    ZONE_BEYOND = -1        // Beyond filesystem capacity
} ZoneType;
typedef struct {
    ZoneType zone;          // Which pointer zone contains this block
    uint64_t block_offset;  // Logical block index within file (offset / 4096)
    // Zone-specific indices
    union {
        struct {
            int direct_idx;         // Index into direct[0..11]
        } direct;
        struct {
            int indirect_idx;       // Index into indirect block (0..511)
        } single_ind;
        struct {
            int double_idx;         // Index into double-indirect block (0..511)
            int indirect_idx;       // Index into indirect block (0..511)
        } double_ind;
    };
} BlockLocation;
// Zone boundary constants (in bytes)
#define DIRECT_ZONE_MAX        (DIRECT_POINTERS * BLOCK_SIZE)  // 49,152 bytes
#define SINGLE_IND_ZONE_MAX    (DIRECT_ZONE_MAX + PTRS_PER_BLOCK * BLOCK_SIZE)  // 2,146,304 bytes
#define DOUBLE_IND_ZONE_MAX    (SINGLE_IND_ZONE_MAX + \
    (uint64_t)PTRS_PER_BLOCK * PTRS_PER_BLOCK * BLOCK_SIZE)  // 1,075,888,128 bytes (~1GB)
```
### Indirect Block Structure
An indirect block is a 4KB block containing 512 block pointers.
```c
// 16_block_ptr.h
typedef struct {
    uint64_t ptrs[PTRS_PER_BLOCK];  // 512 pointers, each to a data block
} IndirectBlock;
// Size: 512 * 8 = 4096 bytes = 1 block
// ptrs[i] = 0 means hole (unallocated)
// ptrs[i] != 0 is a valid data block number
```

![Inode Structure Memory Layout](./diagrams/tdd-diag-m2-01.svg)

---
## Interface Contracts
### Inode Read/Write Operations
```c
// 14_inode.h
/**
 * Read an inode from disk.
 *
 * @param dev        BlockDevice containing the filesystem
 * @param sb         Superblock with inode table location
 * @param inode_num  Inode number to read (1-indexed)
 * @param inode      Output buffer for inode data
 * @return 0 on success, -1 on failure
 *
 * Errors:
 *   EINVAL - inode_num < 1 or > sb->total_inodes
 *   EIO    - Read error from block device
 *
 * Postcondition: inode contains the on-disk inode data
 */
int read_inode(BlockDevice* dev, const Superblock* sb, 
               uint64_t inode_num, Inode* inode);
/**
 * Write an inode to disk.
 *
 * @param dev        BlockDevice containing the filesystem
 * @param sb         Superblock with inode table location
 * @param inode_num  Inode number to write (1-indexed)
 * @param inode      Inode data to write
 * @return 0 on success, -1 on failure
 *
 * Errors:
 *   EINVAL - inode_num < 1 or > sb->total_inodes
 *   EIO    - Write error from block device
 *
 * Postcondition: Inode block on disk contains inode data
 */
int write_inode(BlockDevice* dev, const Superblock* sb,
                uint64_t inode_num, const Inode* inode);
/**
 * Calculate which block contains a given inode.
 *
 * @param sb         Superblock with inode table location
 * @param inode_num  Inode number (1-indexed)
 * @return Physical block number, or 0 if invalid
 *
 * Note: In this design, 1 inode per block, so:
 *   block_num = sb->inode_table_start + (inode_num - 1)
 */
uint64_t inode_to_block(const Superblock* sb, uint64_t inode_num);
/**
 * Initialize a new inode with default values.
 *
 * @param inode      Inode to initialize
 * @param mode       File type and permissions (e.g., S_IFREG | 0644)
 * @param uid        Owner user ID
 * @param gid        Owner group ID
 *
 * Postcondition: All fields zeroed except mode/uid/gid, timestamps set to now,
 *                link_count=1, all block pointers=0
 */
void inode_init(Inode* inode, uint16_t mode, uint16_t uid, uint16_t gid);
```
### Zone Calculation
```c
// 16_block_ptr.h
/**
 * Calculate the block location for a file byte offset.
 *
 * @param file_offset  Byte offset within file
 * @return BlockLocation with zone and indices
 *
 * Example:
 *   offset 0 → ZONE_DIRECT, direct_idx=0
 *   offset 4096 → ZONE_DIRECT, direct_idx=1
 *   offset 49152 → ZONE_SINGLE_IND, indirect_idx=0
 *   offset 2146304 → ZONE_DOUBLE_IND, double_idx=0, indirect_idx=0
 *   offset 1075888128 → ZONE_BEYOND (error)
 */
BlockLocation locate_block(uint64_t file_offset);
/**
 * Check if a file offset falls in a sparse region (hole).
 *
 * @param inode    Inode to check
 * @param offset   Byte offset within file
 * @return true if offset is in a hole, false if allocated or beyond EOF
 */
bool is_hole(const Inode* inode, uint64_t offset);
```
### Block Pointer Resolution
```c
// 17_block_ptr.h
/**
 * Get the physical block number for a file's logical block.
 * Does NOT allocate - returns 0 for holes.
 *
 * @param dev            BlockDevice for indirect block reads
 * @param inode          Inode containing block pointers
 * @param block_offset   Logical block index (0 = first 4KB of file)
 * @param indirect_buf   Temporary buffer for indirect block (BLOCK_SIZE bytes)
 * @return Physical block number, or 0 if hole or error
 *
 * Performance:
 *   - Direct zone: 0 extra I/O (pointer in inode)
 *   - Single-indirect: 1 I/O (read indirect block)
 *   - Double-indirect: 2 I/Os (read double-indirect, then indirect)
 *
 * Note: Returns 0 for both holes and errors. Caller should check
 *       block_offset < inode->size / BLOCK_SIZE to distinguish.
 */
uint64_t get_file_block(BlockDevice* dev, const Inode* inode,
                        uint64_t block_offset, void* indirect_buf);
/**
 * Read a pointer from an indirect block.
 *
 * @param dev             BlockDevice
 * @param indirect_block  Physical block number of indirect block
 * @param index           Index within indirect block (0..511)
 * @param buf             Temporary buffer (BLOCK_SIZE bytes)
 * @return Pointer value (block number), or 0 if invalid
 */
uint64_t read_indirect_ptr(BlockDevice* dev, uint64_t indirect_block,
                           int index, void* buf);
/**
 * Write a pointer to an indirect block.
 *
 * @param dev             BlockDevice
 * @param indirect_block  Physical block number (must be allocated)
 * @param index           Index within indirect block (0..511)
 * @param value           Pointer value to write
 * @param buf             Temporary buffer (BLOCK_SIZE bytes)
 * @return 0 on success, -1 on failure
 */
int write_indirect_ptr(BlockDevice* dev, uint64_t indirect_block,
                       int index, uint64_t value, void* buf);
```
### Block Allocation on Write
```c
// 18_inode_alloc.h
/**
 * Get or allocate a block for a file's logical block.
 * Allocates the block and any necessary indirect blocks.
 *
 * @param fs              FileSystem structure (contains dev, sb)
 * @param inode_num       Inode number being modified
 * @param inode           Inode to update (modified in place)
 * @param block_offset    Logical block index to allocate
 * @param indirect_buf    Temporary buffer (BLOCK_SIZE bytes)
 * @param bitmap_buf      Temporary buffer for bitmap (BLOCK_SIZE bytes)
 * @return Physical block number, or 0 on failure
 *
 * Errors (in errno):
 *   ENOSPC - No free blocks for data or indirect block
 *   EIO   - Read/write error
 *   EINVAL - block_offset beyond double-indirect capacity
 *
 * Side effects:
 *   - May allocate 1-3 blocks (data, indirect, double-indirect)
 *   - Updates inode->blocks count
 *   - Updates inode block pointers
 *   - Caller must write inode to disk
 *
 * Allocation cascade for offset 5MB:
 *   1. Allocate double-indirect block (if first time in this zone)
 *   2. Allocate indirect block (if first time in this subrange)
 *   3. Allocate data block
 */
uint64_t get_or_alloc_block(FileSystem* fs, uint64_t inode_num,
                            Inode* inode, uint64_t block_offset,
                            void* indirect_buf, void* bitmap_buf);
```
### Inode Deallocation
```c
// 18_inode_alloc.h
/**
 * Free all blocks associated with an inode.
 * Recursively frees indirect and double-indirect blocks.
 *
 * @param fs         FileSystem structure
 * @param inode      Inode to clear (modified in place)
 * @param bitmap_buf Temporary buffer for bitmap operations
 * @return 0 on success, -1 on failure
 *
 * Postcondition:
 *   - All data blocks freed
 *   - All indirect blocks freed
 *   - Double-indirect block freed
 *   - inode->blocks = 0
 *   - All block pointers set to 0
 *
 * Note: Does NOT free the inode itself (that's inode_bitmap_free)
 */
int inode_free_all_blocks(FileSystem* fs, Inode* inode, void* bitmap_buf);
/**
 * Free all blocks in an indirect block, then free the indirect block itself.
 *
 * @param fs              FileSystem structure
 * @param indirect_block  Physical block number of indirect block
 * @param bitmap_buf      Temporary buffer
 * @return Number of blocks freed, or -1 on error
 */
int free_indirect_block(FileSystem* fs, uint64_t indirect_block, void* bitmap_buf);
/**
 * Free all blocks in a double-indirect structure.
 *
 * @param fs                FileSystem structure
 * @param double_ind_block  Physical block number of double-indirect block
 * @param indirect_buf      Buffer for reading indirect blocks
 * @param bitmap_buf        Buffer for bitmap operations
 * @return Number of blocks freed, or -1 on error
 */
int free_double_indirect_block(FileSystem* fs, uint64_t double_ind_block,
                               void* indirect_buf, void* bitmap_buf);
```
### Timestamp Management
```c
// 20_timestamp.h
// Timestamp update flags
#define TS_ATIME  (1 << 0)  // Update access time
#define TS_MTIME  (1 << 1)  // Update modification time
#define TS_CTIME  (1 << 2)  // Update change time
// Common combinations
#define TS_READ    TS_ATIME                    // File was read
#define TS_WRITE   (TS_MTIME | TS_CTIME)       // Content modified
#define TS_CREATE  (TS_ATIME | TS_MTIME | TS_CTIME)  // New file
#define TS_META    TS_CTIME                    // Metadata only (chmod, etc.)
/**
 * Update inode timestamps.
 *
 * @param inode  Inode to update
 * @param flags  OR of TS_* constants
 *
 * Semantics:
 *   - atime: Updated when file is read
 *   - mtime: Updated when file content changes
 *   - ctime: Updated when ANY inode field changes (including mtime)
 *
 * Note: Uses time(NULL) for current time. Caller must write inode.
 */
void update_timestamps(Inode* inode, int flags);
/**
 * Get current time as uint64_t Unix timestamp.
 *
 * @return Seconds since Unix epoch
 */
uint64_t get_current_time(void);
```
---
## Algorithm Specification
### locate_block Algorithm
**Input:** `file_offset` - byte position within file
**Output:** `BlockLocation` - zone classification and indices
```
ALGORITHM locate_block(file_offset):
  result ← zero-initialized BlockLocation
  block_offset ← file_offset >> 12  // Division by 4096 (faster than /)
  result.block_offset ← block_offset
  // Zone 1: Direct pointers (blocks 0-11)
  IF block_offset < DIRECT_POINTERS:  // 12
    result.zone ← ZONE_DIRECT
    result.direct.direct_idx ← (int)block_offset
    RETURN result
  // Zone 2: Single-indirect (blocks 12-523)
  block_offset ← block_offset - DIRECT_POINTERS
  IF block_offset < PTRS_PER_BLOCK:  // 512
    result.zone ← ZONE_SINGLE_IND
    result.single_ind.indirect_idx ← (int)block_offset
    RETURN result
  // Zone 3: Double-indirect (blocks 524-262659)
  block_offset ← block_offset - PTRS_PER_BLOCK
  max_double ← (uint64_t)PTRS_PER_BLOCK * PTRS_PER_BLOCK  // 262144
  IF block_offset < max_double:
    result.zone ← ZONE_DOUBLE_IND
    result.double_ind.double_idx ← (int)(block_offset / PTRS_PER_BLOCK)
    result.double_ind.indirect_idx ← (int)(block_offset % PTRS_PER_BLOCK)
    RETURN result
  // Beyond capacity
  result.zone ← ZONE_BEYOND
  RETURN result
```
**Complexity:** O(1) - pure arithmetic, no loops

![Block Pointer Zone Atlas](./diagrams/tdd-diag-m2-02.svg)

### get_file_block Algorithm
**Input:** `dev`, `inode`, `block_offset`, `indirect_buf`
**Output:** Physical block number, or 0 for hole/error
```
ALGORITHM get_file_block(dev, inode, block_offset, indirect_buf):
  loc ← locate_block(block_offset * BLOCK_SIZE)  // Convert to byte offset
  SWITCH loc.zone:
    CASE ZONE_DIRECT:
      // Direct pointer is in inode itself
      RETURN inode->direct[loc.direct.direct_idx]
    CASE ZONE_SINGLE_IND:
      // Check if indirect block exists
      IF inode->indirect = 0:
        RETURN 0  // Hole - no indirect block means entire zone is holes
      // Read pointer from indirect block
      RETURN read_indirect_ptr(dev, inode->indirect, 
                               loc.single_ind.indirect_idx, indirect_buf)
    CASE ZONE_DOUBLE_IND:
      // Check if double-indirect block exists
      IF inode->double_ind = 0:
        RETURN 0  // Hole
      // Step 1: Read double-indirect block, get indirect block number
      double_buf ← indirect_buf  // Reuse buffer
      indirect_block ← read_indirect_ptr(dev, inode->double_ind,
                                         loc.double_ind.double_idx, double_buf)
      IF indirect_block = 0:
        RETURN 0  // Hole - no indirect block at this position
      // Step 2: Read indirect block, get data block number
      // Need separate buffer or careful ordering since we used indirect_buf
      single_buf ← allocate(BLOCK_SIZE)  // Or use second buffer
      result ← read_indirect_ptr(dev, indirect_block,
                                 loc.double_ind.indirect_idx, single_buf)
      free(single_buf)
      RETURN result
    CASE ZONE_BEYOND:
      RETURN 0  // Error: file too large
```
**I/O Count by Zone:**
| Zone | Block Reads | Notes |
|------|-------------|-------|
| Direct | 0 | Pointer in inode |
| Single-indirect | 1 | Read indirect block |
| Double-indirect | 2 | Read double-indirect, then indirect |
**Hardware Soul - Cache/TLB Impact:**
```
Direct zone access:
  - Inode already in memory (cache hit)
  - 0 additional memory accesses
  - Latency: ~0 (if inode cached) or ~1 block read
Single-indirect access:
  - Read indirect block: likely cache miss
  - 1 TLB miss for indirect block page
  - 64 cache line loads (entire 4KB block)
  - Latency: ~25μs (NVMe) or ~10ms (HDD random)
Double-indirect access:
  - 2 cache misses, 2 TLB misses
  - 128 cache line loads total
  - Latency: ~50μs (NVMe) or ~20ms (HDD)
```

![Indirect Block Structure](./diagrams/tdd-diag-m2-03.svg)

### get_or_alloc_block Algorithm
**Input:** `fs`, `inode_num`, `inode`, `block_offset`, `indirect_buf`, `bitmap_buf`
**Output:** Physical block number, or 0 on failure
**Side Effects:** May allocate 1-3 blocks, modifies inode
```
ALGORITHM get_or_alloc_block(fs, inode_num, inode, block_offset, 
                             indirect_buf, bitmap_buf):
  // First, try to get existing block
  existing ← get_file_block(fs->dev, inode, block_offset, indirect_buf)
  IF existing ≠ 0:
    RETURN existing  // Already allocated
  // Need to allocate - determine zone
  loc ← locate_block(block_offset * BLOCK_SIZE)
  SWITCH loc.zone:
    CASE ZONE_DIRECT:
      // Allocate data block
      new_block ← block_bitmap_alloc(fs->dev, fs->sb, bitmap_buf)
      IF new_block = 0:
        errno ← ENOSPC
        RETURN 0
      inode->direct[loc.direct.direct_idx] ← new_block
      inode->blocks ← inode->blocks + 8  // 8 * 512 = 4096
      RETURN new_block
    CASE ZONE_SINGLE_IND:
      RETURN alloc_single_indirect_zone(fs, inode, loc, 
                                        indirect_buf, bitmap_buf)
    CASE ZONE_DOUBLE_IND:
      RETURN alloc_double_indirect_zone(fs, inode, loc,
                                        indirect_buf, bitmap_buf)
    CASE ZONE_BEYOND:
      errno ← EINVAL
      RETURN 0
// Helper for single-indirect allocation
ALGORITHM alloc_single_indirect_zone(fs, inode, loc, indirect_buf, bitmap_buf):
  // Ensure indirect block exists
  IF inode->indirect = 0:
    indirect_block ← block_bitmap_alloc(fs->dev, fs->sb, bitmap_buf)
    IF indirect_block = 0:
      errno ← ENOSPC
      RETURN 0
    // Zero the new indirect block
    memset(indirect_buf, 0, BLOCK_SIZE)
    write_block(fs->dev, indirect_block, indirect_buf)
    inode->indirect ← indirect_block
    inode->blocks ← inode->blocks + 8
  // Allocate data block
  new_block ← block_bitmap_alloc(fs->dev, fs->sb, bitmap_buf)
  IF new_block = 0:
    // Note: indirect block already allocated, not freed
    // This is acceptable - it will be used for future allocations
    errno ← ENOSPC
    RETURN 0
  // Write pointer into indirect block
  write_indirect_ptr(fs->dev, inode->indirect,
                     loc.single_ind.indirect_idx, new_block, indirect_buf)
  inode->blocks ← inode->blocks + 8
  RETURN new_block
// Helper for double-indirect allocation
ALGORITHM alloc_double_indirect_zone(fs, inode, loc, indirect_buf, bitmap_buf):
  double_buf ← allocate(BLOCK_SIZE)
  // Step 1: Ensure double-indirect block exists
  IF inode->double_ind = 0:
    double_block ← block_bitmap_alloc(fs->dev, fs->sb, bitmap_buf)
    IF double_block = 0:
      free(double_buf)
      errno ← ENOSPC
      RETURN 0
    memset(double_buf, 0, BLOCK_SIZE)
    write_block(fs->dev, double_block, double_buf)
    inode->double_ind ← double_block
    inode->blocks ← inode->blocks + 8
  // Step 2: Ensure indirect block exists
  read_block(fs->dev, inode->double_ind, double_buf)
  uint64_t* double_ptrs ← (uint64_t*)double_buf
  IF double_ptrs[loc.double_ind.double_idx] = 0:
    indirect_block ← block_bitmap_alloc(fs->dev, fs->sb, bitmap_buf)
    IF indirect_block = 0:
      free(double_buf)
      errno ← ENOSPC
      RETURN 0
    memset(indirect_buf, 0, BLOCK_SIZE)
    write_block(fs->dev, indirect_block, indirect_buf)
    double_ptrs[loc.double_ind.double_idx] ← indirect_block
    write_block(fs->dev, inode->double_ind, double_buf)
    inode->blocks ← inode->blocks + 8
  ELSE:
    indirect_block ← double_ptrs[loc.double_ind.double_idx]
  // Step 3: Allocate data block
  new_block ← block_bitmap_alloc(fs->dev, fs->sb, bitmap_buf)
  IF new_block = 0:
    free(double_buf)
    errno ← ENOSPC
    RETURN 0
  // Write pointer into indirect block
  read_block(fs->dev, indirect_block, indirect_buf)
  uint64_t* ind_ptrs ← (uint64_t*)indirect_buf
  ind_ptrs[loc.double_ind.indirect_idx] ← new_block
  write_block(fs->dev, indirect_block, indirect_buf)
  inode->blocks ← inode->blocks + 8
  free(double_buf)
  RETURN new_block
```
**Allocation Cascade Example:**
```
Writing 1 byte at offset 5MB in empty file:
5MB = 5,242,880 bytes = block 1280
Block 1280 is in double-indirect zone:
  - Blocks 0-11: direct
  - Blocks 12-523: single-indirect
  - Blocks 524+: double-indirect
  1280 - 524 = 756th block in double-indirect zone
  double_idx = 756 / 512 = 1
  indirect_idx = 756 % 512 = 244
Allocation cascade:
  1. Allocate double-indirect block (block A)
  2. Allocate indirect block for double_idx=1 (block B)
  3. Allocate data block (block C)
Blocks allocated: 3
Actual data: 1 byte
Overhead: 2/3 of allocation
If we then write at offset 5MB + 4KB (block 1281):
  Same double_idx=1, indirect_idx=245
  Only allocate data block (1 block)
  Indirect blocks already exist
```

![Block Location Calculation](./diagrams/tdd-diag-m2-04.svg)

### inode_free_all_blocks Algorithm
**Input:** `fs`, `inode`, `bitmap_buf`
**Output:** 0 on success, -1 on failure
**Postcondition:** All blocks freed, inode->blocks = 0, all pointers = 0
```
ALGORITHM inode_free_all_blocks(fs, inode, bitmap_buf):
  blocks_freed ← 0
  // Step 1: Free direct blocks
  FOR i FROM 0 TO DIRECT_POINTERS - 1:
    IF inode->direct[i] ≠ 0:
      IF block_bitmap_free(fs->dev, fs->sb, inode->direct[i], bitmap_buf) < 0:
        LOG("Warning: failed to free direct block %lu", inode->direct[i])
        // Continue anyway - best effort
      inode->direct[i] ← 0
      blocks_freed ← blocks_freed + 1
  // Step 2: Free single-indirect and its children
  IF inode->indirect ≠ 0:
    freed ← free_indirect_block(fs, inode->indirect, bitmap_buf)
    IF freed > 0:
      blocks_freed ← blocks_freed + freed
    inode->indirect ← 0
  // Step 3: Free double-indirect and its children
  IF inode->double_ind ≠ 0:
    indirect_buf ← allocate(BLOCK_SIZE)
    freed ← free_double_indirect_block(fs, inode->double_ind,
                                       indirect_buf, bitmap_buf)
    free(indirect_buf)
    IF freed > 0:
      blocks_freed ← blocks_freed + freed
    inode->double_ind ← 0
  // Step 4: Clear triple-indirect (should be 0, but be thorough)
  inode->triple_ind ← 0
  // Update inode
  inode->blocks ← 0
  inode->size ← 0
  RETURN 0
ALGORITHM free_indirect_block(fs, indirect_block_num, bitmap_buf):
  blocks_freed ← 0
  buf ← allocate(BLOCK_SIZE)
  IF read_block(fs->dev, indirect_block_num, buf) < 0:
    free(buf)
    RETURN -1
  uint64_t* ptrs ← (uint64_t*)buf
  // Free all data blocks pointed to by this indirect block
  FOR i FROM 0 TO PTRS_PER_BLOCK - 1:
    IF ptrs[i] ≠ 0:
      block_bitmap_free(fs->dev, fs->sb, ptrs[i], bitmap_buf)
      blocks_freed ← blocks_freed + 1
  // Free the indirect block itself
  block_bitmap_free(fs->dev, fs->sb, indirect_block_num, bitmap_buf)
  blocks_freed ← blocks_freed + 1
  free(buf)
  RETURN blocks_freed
ALGORITHM free_double_indirect_block(fs, double_ind_block_num, 
                                     indirect_buf, bitmap_buf):
  blocks_freed ← 0
  double_buf ← allocate(BLOCK_SIZE)
  IF read_block(fs->dev, double_ind_block_num, double_buf) < 0:
    free(double_buf)
    RETURN -1
  uint64_t* double_ptrs ← (uint64_t*)double_buf
  // Free each indirect block and its children
  FOR i FROM 0 TO PTRS_PER_BLOCK - 1:
    IF double_ptrs[i] ≠ 0:
      freed ← free_indirect_block(fs, double_ptrs[i], bitmap_buf)
      IF freed > 0:
        blocks_freed ← blocks_freed + freed
  // Free the double-indirect block itself
  block_bitmap_free(fs->dev, fs->sb, double_ind_block_num, bitmap_buf)
  blocks_freed ← blocks_freed + 1
  free(double_buf)
  RETURN blocks_freed
```
**Complexity Analysis:**
| Inode Type | Max Blocks to Free | Loop Iterations |
|------------|-------------------|-----------------|
| Small file (direct only) | 12 | 12 |
| Medium file (single-ind) | 12 + 512 + 1 | 525 |
| Large file (double-ind) | 12 + 512 + 1 + 512*512 + 512 | 262,659 |
For a full 1GB file, freeing requires:
- 512 indirect blocks to read and process
- 1 double-indirect block to read
- ~262,659 block_bitmap_free calls
- Estimated time: ~10 seconds on HDD, ~0.5 seconds on NVMe
{{DIAGRAM:tdd-diag-m2-05}}
---
## Error Handling Matrix
| Error | errno | Detected By | Recovery | User-Visible Message |
|-------|-------|-------------|----------|---------------------|
| Invalid inode number | EINVAL | read_inode, write_inode | Return -1, no disk access | "Invalid inode number %lu" |
| Block device read error | EIO | read_block (internal) | Return 0 (treat as hole) | "I/O error reading inode %lu" |
| Block device write error | EIO | write_block (internal) | Return -1, state may be partial | "I/O error writing inode %lu" |
| No free blocks for data | ENOSPC | get_or_alloc_block | Return 0, inode unchanged | "Filesystem full" |
| No free blocks for indirect | ENOSPC | get_or_alloc_block | Return 0, partial alloc may exist | "Filesystem full" |
| File too large | EINVAL | locate_block (ZONE_BEYOND) | Return 0, no allocation | "File too large (max 1GB)" |
| Double-free block | EINVAL | block_bitmap_free | Return -1 from helper, logged | Internal error (logged) |
| Indirect block read error | EIO | free_indirect_block | Log warning, continue | "Warning: could not read indirect block" |
**Partial Allocation Recovery:**
When `get_or_alloc_block` fails mid-cascade:
```
Scenario: Allocating at 5MB, double-indirect allocated, indirect allocated, data allocation fails
State after failure:
  - double_ind = X (allocated)
  - indirect block Y pointed to by double_ind[X][1]
  - No data block
Result: File has "orphaned" indirect structure
  - Acceptable: These blocks will be used for subsequent writes
  - Alternative: Rollback all allocations (requires journaling - see M6)
```
---
## Implementation Sequence with Checkpoints
### Phase 1: Inode Structure Definition (1-2 hours)
**Files:** `14_inode.h`, `15_inode.c`
**Implementation Steps:**
1. Define complete `Inode` struct with `__attribute__((packed))`
2. Add `static_assert` for size and field offsets
3. Implement `inode_to_block` helper
4. Implement `read_inode`:
   - Validate inode_num bounds
   - Calculate block number
   - Call `read_block`
   - Copy to output buffer
5. Implement `write_inode`:
   - Validate bounds
   - Calculate block number
   - Copy from input buffer
   - Call `write_block`
6. Implement `inode_init` helper
**Checkpoint 1:**
```
$ make test_inode_struct
$ ./test_inode_struct
Testing inode structure...
  sizeof(Inode) = 4096: OK
  offsetof(mode) = 0: OK
  offsetof(direct[0]) = 50: OK
  offsetof(indirect) = 146: OK
  offsetof(double_ind) = 154: OK
Testing read/write...
  Write inode 1: OK
  Read back inode 1: OK
  Fields match: OK
  Invalid inode 0: -1 (expected)
  Invalid inode 999999: -1 (expected)
All inode struct tests passed!
```
### Phase 2: Zone Boundary Calculation (2-3 hours)
**Files:** `16_block_ptr.h`, `17_block_ptr.c`
**Implementation Steps:**
1. Define `ZoneType` enum and `BlockLocation` struct
2. Define zone boundary constants
3. Implement `locate_block`:
   - Use bit shifts for division by 4096
   - Calculate zone and indices
4. Implement `is_hole` helper (checks if pointer is 0)
**Checkpoint 2:**
```
$ make test_block_ptr
$ ./test_block_ptr
Testing zone calculation...
  offset 0: zone=DIRECT, idx=0: OK
  offset 4095: zone=DIRECT, idx=0: OK
  offset 4096: zone=DIRECT, idx=1: OK
  offset 49151: zone=DIRECT, idx=11: OK
  offset 49152: zone=SINGLE_IND, idx=0: OK
  offset 2097152: zone=SINGLE_IND, idx=511: OK
  offset 2146304: zone=DOUBLE_IND, double=0, indirect=0: OK
  offset 5242880: zone=DOUBLE_IND, double=1, indirect=244: OK
  offset 1075888128: zone=BEYOND: OK
All zone tests passed!
```
### Phase 3: Block Pointer Resolution (3-4 hours)
**Files:** Update `17_block_ptr.c`
**Implementation Steps:**
1. Implement `read_indirect_ptr`:
   - Validate index 0..511
   - Read block, extract pointer at index
2. Implement `write_indirect_ptr`:
   - Read block, modify pointer, write back
3. Implement `get_file_block`:
   - Use `locate_block` to determine zone
   - Handle each zone type
   - Return 0 for holes
**Checkpoint 3:**
```
$ make test_block_ptr
$ ./test_block_ptr
... zone tests ...
Testing pointer resolution (with test image)...
  Direct zone, allocated: block 1234: OK
  Direct zone, hole: 0: OK
  Single-indirect, allocated: block 5678: OK
  Single-indirect, hole: 0: OK
  Single-indirect, no indirect block: 0: OK
  Double-indirect, allocated: block 9012: OK
  Double-indirect, hole in indirect: 0: OK
  Double-indirect, no double-indirect block: 0: OK
All pointer resolution tests passed!
```
### Phase 4: Block Allocation on Write (3-4 hours)
**Files:** `18_inode_alloc.h`, `19_inode_alloc.c`
**Implementation Steps:**
1. Implement `get_or_alloc_block` direct zone case
2. Implement `alloc_single_indirect_zone` helper
3. Implement `alloc_double_indirect_zone` helper
4. Handle allocation failures gracefully
5. Update `inode->blocks` correctly
**Checkpoint 4:**
```
$ make test_inode_alloc
$ ./test_inode_alloc
Testing block allocation...
  Allocate direct[0]: block 1234, inode->blocks=8: OK
  Allocate direct[11]: block 1245, inode->blocks=96: OK
  Allocate block 12 (first single-indirect):
    indirect block allocated: 1300
    data block allocated: 1301
    inode->blocks=112: OK
  Allocate block 523 (last single-indirect): OK
  Allocate block 524 (first double-indirect):
    double-indirect allocated: 1800
    indirect allocated: 1801
    data block allocated: 1802
    inode->blocks=136: OK
  Fill filesystem:
    Allocated 50000 blocks before ENOSPC: OK
All allocation tests passed!
```
### Phase 5: Inode Deallocation (2-3 hours)
**Files:** Update `19_inode_alloc.c`
**Implementation Steps:**
1. Implement `free_indirect_block`
2. Implement `free_double_indirect_block`
3. Implement `inode_free_all_blocks`
4. Test with files of various sizes
**Checkpoint 5:**
```
$ make test_inode_alloc
$ ./test_inode_alloc
... allocation tests ...
Testing deallocation...
  Free small file (3 direct blocks):
    Blocks freed: 3
    inode->blocks=0: OK
    All pointers zeroed: OK
  Free medium file (500 blocks):
    Blocks freed: 502 (500 data + 1 indirect + 1 indirect block itself)
    inode->blocks=0: OK
  Free large file (10000 blocks):
    Blocks freed: 10020 (10000 + 19 indirects + 1 double-ind)
    inode->blocks=0: OK
  Verify blocks returned to bitmap:
    Can re-allocate same number: OK
All deallocation tests passed!
```
### Phase 6: Timestamp Management (1 hour)
**Files:** `20_timestamp.h`, `21_timestamp.c`
**Implementation Steps:**
1. Define TS_* constants
2. Implement `get_current_time`
3. Implement `update_timestamps`
**Checkpoint 6:**
```
$ make test_timestamp
$ ./test_timestamp
Testing timestamps...
  TS_READ: atime updated, mtime/ctime unchanged: OK
  TS_WRITE: mtime and ctime updated, atime unchanged: OK
  TS_CREATE: all three updated: OK
  TS_META: only ctime updated: OK
  Timestamp ordering: atime <= mtime <= ctime (for create): OK
All timestamp tests passed!
```
### Phase 7: Integration Testing (2-3 hours)
**Files:** `tests/test_sparse.c`, comprehensive integration tests
**Implementation Steps:**
1. Test sparse file creation (write at high offset)
2. Test hole detection and reading
3. Test file growth through all zones
4. Test file truncation (will use deallocation)
5. Performance benchmarks
**Checkpoint 7:**
```
$ make test_integration
$ ./test_integration
Testing sparse files...
  Write 100 bytes at offset 1MB:
    File size: 1048576 + 100 = 1048676
    Blocks allocated: 3 (1 double-ind + 1 ind + 1 data): OK
  Read from hole at offset 0:
    Returns zeros: OK
  Read from data at offset 1MB:
    Returns written data: OK
Testing file growth...
  Grow from 0 to 50KB (direct zone): OK
  Grow from 50KB to 2MB (single-ind): OK
  Grow from 2MB to 100MB (double-ind): OK
Testing deallocation after growth...
  Free 100MB file: blocks freed correctly: OK
Performance benchmarks...
  1000 direct block lookups: 0.5ms (0.5μs each): OK
  1000 single-indirect lookups: 25ms (25μs each): OK
  1000 double-indirect lookups: 50ms (50μs each): OK
All integration tests passed!
```

![Sparse File Hole Detection](./diagrams/tdd-diag-m2-06.svg)

---
## Test Specification
### test_inode_struct.c
```c
void test_inode_size_and_layout() {
    ASSERT(sizeof(Inode) == 4096);
    ASSERT(offsetof(Inode, mode) == 0x00);
    ASSERT(offsetof(Inode, uid) == 0x02);
    ASSERT(offsetof(Inode, gid) == 0x04);
    ASSERT(offsetof(Inode, link_count) == 0x06);
    ASSERT(offsetof(Inode, size) == 0x0A);
    ASSERT(offsetof(Inode, blocks) == 0x12);
    ASSERT(offsetof(Inode, atime) == 0x1A);
    ASSERT(offsetof(Inode, mtime) == 0x22);
    ASSERT(offsetof(Inode, ctime) == 0x2A);
    ASSERT(offsetof(Inode, direct) == 0x32);
    ASSERT(offsetof(Inode, indirect) == 0x92);
    ASSERT(offsetof(Inode, double_ind) == 0x9A);
}
void test_inode_read_write() {
    BlockDevice* dev = create_test_device(1000);
    Superblock sb = create_test_superblock(dev, 1000, 100);
    Inode written;
    memset(&written, 0, sizeof(Inode));
    written.mode = S_IFREG | 0644;
    written.uid = 1000;
    written.gid = 1000;
    written.link_count = 1;
    written.size = 12345;
    written.direct[0] = 500;
    ASSERT(write_inode(dev, &sb, 1, &written) == 0);
    Inode read_back;
    ASSERT(read_inode(dev, &sb, 1, &read_back) == 0);
    ASSERT(read_back.mode == written.mode);
    ASSERT(read_back.uid == written.uid);
    ASSERT(read_back.size == written.size);
    ASSERT(read_back.direct[0] == 500);
    block_device_close(dev);
}
void test_inode_bounds_checking() {
    BlockDevice* dev = create_test_device(1000);
    Superblock sb = create_test_superblock(dev, 1000, 100);
    Inode inode;
    errno = 0;
    ASSERT(read_inode(dev, &sb, 0, &inode) == -1);
    ASSERT(errno == EINVAL);
    errno = 0;
    ASSERT(read_inode(dev, &sb, 101, &inode) == -1);
    ASSERT(errno == EINVAL);
    errno = 0;
    ASSERT(write_inode(dev, &sb, 0, &inode) == -1);
    ASSERT(errno == EINVAL);
    block_device_close(dev);
}
void test_inode_init() {
    Inode inode;
    inode_init(&inode, S_IFREG | 0644, 1000, 1000);
    ASSERT(inode.mode == (S_IFREG | 0644));
    ASSERT(inode.uid == 1000);
    ASSERT(inode.gid == 1000);
    ASSERT(inode.link_count == 1);
    ASSERT(inode.size == 0);
    ASSERT(inode.blocks == 0);
    ASSERT(inode.atime > 0);
    ASSERT(inode.mtime == inode.atime);
    ASSERT(inode.ctime == inode.atime);
    for (int i = 0; i < 12; i++) {
        ASSERT(inode.direct[i] == 0);
    }
    ASSERT(inode.indirect == 0);
    ASSERT(inode.double_ind == 0);
}
```
### test_block_ptr.c
```c
void test_locate_block_direct() {
    BlockLocation loc;
    loc = locate_block(0);
    ASSERT(loc.zone == ZONE_DIRECT);
    ASSERT(loc.block_offset == 0);
    ASSERT(loc.direct.direct_idx == 0);
    loc = locate_block(4095);
    ASSERT(loc.zone == ZONE_DIRECT);
    ASSERT(loc.direct.direct_idx == 0);
    loc = locate_block(4096);
    ASSERT(loc.zone == ZONE_DIRECT);
    ASSERT(loc.direct.direct_idx == 1);
    loc = locate_block(49151);  // Last byte of direct zone
    ASSERT(loc.zone == ZONE_DIRECT);
    ASSERT(loc.direct.direct_idx == 11);
}
void test_locate_block_single_indirect() {
    BlockLocation loc;
    loc = locate_block(49152);  // First byte of single-indirect
    ASSERT(loc.zone == ZONE_SINGLE_IND);
    ASSERT(loc.single_ind.indirect_idx == 0);
    loc = locate_block(2097152 + 49151);  // Last byte of single-indirect
    ASSERT(loc.zone == ZONE_SINGLE_IND);
    ASSERT(loc.single_ind.indirect_idx == 511);
}
void test_locate_block_double_indirect() {
    BlockLocation loc;
    loc = locate_block(2146304);  // First byte of double-indirect
    ASSERT(loc.zone == ZONE_DOUBLE_IND);
    ASSERT(loc.double_ind.double_idx == 0);
    ASSERT(loc.double_ind.indirect_idx == 0);
    loc = locate_block(5242880);  // 5MB
    // 5242880 - 2146304 = 3096576 bytes into double-indirect
    // 3096576 / 4096 = 756 blocks
    // double_idx = 756 / 512 = 1
    // indirect_idx = 756 % 512 = 244
    ASSERT(loc.zone == ZONE_DOUBLE_IND);
    ASSERT(loc.double_ind.double_idx == 1);
    ASSERT(loc.double_ind.indirect_idx == 244);
}
void test_locate_block_beyond() {
    BlockLocation loc = locate_block(DOUBLE_IND_ZONE_MAX);
    ASSERT(loc.zone == ZONE_BEYOND);
}
void test_get_file_block_direct() {
    // Create test filesystem
    BlockDevice* dev = create_test_device(10000);
    Superblock sb = create_test_superblock(dev, 10000, 100);
    // Create inode with direct blocks
    Inode inode;
    memset(&inode, 0, sizeof(Inode));
    inode.direct[0] = 5000;
    inode.direct[5] = 5005;
    char buf[BLOCK_SIZE];
    // Get allocated block
    ASSERT(get_file_block(dev, &inode, 0, buf) == 5000);
    ASSERT(get_file_block(dev, &inode, 5, buf) == 5005);
    // Get hole (unallocated)
    ASSERT(get_file_block(dev, &inode, 1, buf) == 0);
    block_device_close(dev);
}
void test_get_file_block_single_indirect() {
    BlockDevice* dev = create_test_device(10000);
    Superblock sb = create_test_superblock(dev, 10000, 100);
    // Create indirect block
    uint64_t indirect_block = 6000;
    char ind_buf[BLOCK_SIZE];
    memset(ind_buf, 0, BLOCK_SIZE);
    uint64_t* ptrs = (uint64_t*)ind_buf;
    ptrs[0] = 7000;
    ptrs[100] = 7100;
    write_block(dev, indirect_block, ind_buf);
    // Create inode
    Inode inode;
    memset(&inode, 0, sizeof(Inode));
    inode.indirect = indirect_block;
    char buf[BLOCK_SIZE];
    // Block 12 = first block in single-indirect zone
    ASSERT(get_file_block(dev, &inode, 12, buf) == 7000);
    ASSERT(get_file_block(dev, &inode, 112, buf) == 7100);
    // Hole
    ASSERT(get_file_block(dev, &inode, 13, buf) == 0);
    block_device_close(dev);
}
```
### test_inode_alloc.c
```c
void test_alloc_direct_zone() {
    FileSystem* fs = create_test_filesystem(10000, 100);
    Inode inode;
    memset(&inode, 0, sizeof(Inode));
    char ind_buf[BLOCK_SIZE], bmp_buf[BLOCK_SIZE];
    // Allocate first direct block
    uint64_t b1 = get_or_alloc_block(fs, 1, &inode, 0, ind_buf, bmp_buf);
    ASSERT(b1 != 0);
    ASSERT(inode.direct[0] == b1);
    ASSERT(inode.blocks == 8);  // 8 * 512 = 4096
    // Allocate another
    uint64_t b2 = get_or_alloc_block(fs, 1, &inode, 5, ind_buf, bmp_buf);
    ASSERT(b2 != 0);
    ASSERT(inode.direct[5] == b2);
    ASSERT(inode.blocks == 16);
    // Re-get same block (should not allocate)
    uint64_t b3 = get_or_alloc_block(fs, 1, &inode, 0, ind_buf, bmp_buf);
    ASSERT(b3 == b1);
    ASSERT(inode.blocks == 16);  // Unchanged
    fs_close(fs);
}
void test_alloc_single_indirect_zone() {
    FileSystem* fs = create_test_filesystem(100000, 1000);
    Inode inode;
    memset(&inode, 0, sizeof(Inode));
    char ind_buf[BLOCK_SIZE], bmp_buf[BLOCK_SIZE];
    // First allocation in single-indirect zone
    uint64_t b1 = get_or_alloc_block(fs, 1, &inode, 12, ind_buf, bmp_buf);
    ASSERT(b1 != 0);
    ASSERT(inode.indirect != 0);  // Indirect block allocated
    ASSERT(inode.blocks >= 16);   // 1 indirect + 1 data
    // Second allocation in same zone (should reuse indirect)
    uint64_t prev_indirect = inode.indirect;
    uint64_t b2 = get_or_alloc_block(fs, 1, &inode, 13, ind_buf, bmp_buf);
    ASSERT(b2 != 0);
    ASSERT(inode.indirect == prev_indirect);  // Same indirect block
    fs_close(fs);
}
void test_alloc_double_indirect_zone() {
    FileSystem* fs = create_test_filesystem(1000000, 10000);
    Inode inode;
    memset(&inode, 0, sizeof(Inode));
    char ind_buf[BLOCK_SIZE], bmp_buf[BLOCK_SIZE];
    // First allocation in double-indirect zone (block 524)
    uint64_t b1 = get_or_alloc_block(fs, 1, &inode, 524, ind_buf, bmp_buf);
    ASSERT(b1 != 0);
    ASSERT(inode.double_ind != 0);
    ASSERT(inode.blocks >= 24);  // 1 double-ind + 1 ind + 1 data
    fs_close(fs);
}
void test_free_all_blocks() {
    FileSystem* fs = create_test_filesystem(100000, 1000);
    Inode inode;
    memset(&inode, 0, sizeof(Inode));
    char ind_buf[BLOCK_SIZE], bmp_buf[BLOCK_SIZE];
    // Allocate blocks in all zones
    for (int i = 0; i < 15; i++) {
        get_or_alloc_block(fs, 1, &inode, i, ind_buf, bmp_buf);
    }
    uint64_t blocks_before = inode.blocks;
    ASSERT(blocks_before > 0);
    // Free all
    inode_free_all_blocks(fs, &inode, bmp_buf);
    ASSERT(inode.blocks == 0);
    ASSERT(inode.size == 0);
    for (int i = 0; i < 12; i++) {
        ASSERT(inode.direct[i] == 0);
    }
    ASSERT(inode.indirect == 0);
    ASSERT(inode.double_ind == 0);
    // Verify blocks returned to bitmap
    uint64_t free_before = fs->sb->free_blocks;
    // Free_all should have incremented free_blocks
    // (This is implicit in block_bitmap_free)
    fs_close(fs);
}
```
### test_sparse.c
```c
void test_sparse_file_creation() {
    FileSystem* fs = create_test_filesystem(100000, 1000);
    // Create inode
    uint64_t inode_num = 1;
    Inode inode;
    inode_init(&inode, S_IFREG | 0644, 0, 0);
    char ind_buf[BLOCK_SIZE], bmp_buf[BLOCK_SIZE];
    // Write at 1MB offset (sparse)
    uint64_t offset_1mb = 1024 * 1024;
    uint64_t block_offset = offset_1mb / BLOCK_SIZE;  // 256
    uint64_t data_block = get_or_alloc_block(fs, inode_num, &inode, 
                                             block_offset, ind_buf, bmp_buf);
    ASSERT(data_block != 0);
    // Set size
    inode.size = offset_1mb + 100;
    // Verify sparse behavior
    // Blocks 0-255 should be holes
    for (uint64_t i = 0; i < block_offset; i++) {
        uint64_t b = get_file_block(fs->dev, &inode, i, ind_buf);
        ASSERT(b == 0);  // Hole
    }
    // Block 256 should be allocated
    ASSERT(get_file_block(fs->dev, &inode, block_offset, ind_buf) == data_block);
    // Check block count (should be small despite large size)
    // 1 double-indirect + 1 indirect + 1 data = 3 blocks
    ASSERT(inode.blocks == 24);  // 3 * 8
    fs_close(fs);
}
void test_hole_detection() {
    Inode inode;
    memset(&inode, 0, sizeof(Inode));
    inode.size = 10000000;  // 10MB
    inode.direct[0] = 1000;  // First block allocated
    // is_hole should return true for unallocated regions
    ASSERT(!is_hole(&inode, 0));      // Block 0 allocated
    ASSERT(is_hole(&inode, 4096));    // Block 1 is hole
    ASSERT(is_hole(&inode, 49152));   // Single-indirect zone, hole
}
```
---
## Performance Targets
| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| `read_inode` | < 25 μs (NVMe) | `time ./bench inode_read 10000` |
| `write_inode` | < 50 μs (NVMe) | `time ./bench inode_write 10000` |
| `locate_block` | < 10 ns | Microbenchmark (pure CPU) |
| `get_file_block` (direct) | < 100 ns | Inode in memory, no I/O |
| `get_file_block` (single-ind) | < 50 μs | 1 block read |
| `get_file_block` (double-ind) | < 100 μs | 2 block reads |
| `get_or_alloc_block` (direct) | < 500 μs | Allocation + write |
| `get_or_alloc_block` (double-ind) | < 5 ms | 3 allocations + 3 writes |
| `inode_free_all_blocks` (1GB file) | < 1 s | Full deallocation |
**Hardware Soul - Latency Breakdown:**
```
get_file_block (double-indirect, cold cache):
  1. locate_block: ~5ns (CPU only, arithmetic)
  2. read_block(double_ind): ~25μs (NVMe) or ~10ms (HDD)
     - 1 TLB miss
     - 64 cache line loads
  3. extract indirect pointer: ~1ns
  4. read_block(indirect): ~25μs (NVMe) or ~10ms (HDD)
     - 1 TLB miss
     - 64 cache line loads
  5. extract data pointer: ~1ns
  Total: ~50μs (NVMe) or ~20ms (HDD)
get_or_alloc_block (double-indirect, first in zone):
  1. get_file_block (returns 0): ~5ns
  2. block_bitmap_alloc (double-ind): ~50μs (scan + write)
  3. write_block (double-ind, zeroed): ~25μs
  4. block_bitmap_alloc (indirect): ~50μs
  5. write_block (indirect, zeroed): ~25μs
  6. block_bitmap_alloc (data): ~50μs
  7. write_indirect_ptr: ~50μs (read + modify + write)
  8. write_indirect_ptr (double): ~50μs
  Total: ~300μs (NVMe), ~120ms (HDD with seeks)
```

![Block Allocation Cascade](./diagrams/tdd-diag-m2-07.svg)

---
## Diagrams

![Inode Deallocation Tree](./diagrams/tdd-diag-m2-08.svg)


![Link Count State Machine](./diagrams/tdd-diag-m2-09.svg)


![File Size vs Block Allocation](./diagrams/tdd-diag-m2-10.svg)

---
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: filesystem-m3 -->
# Technical Design Specification: Directory Operations
## Module Charter
This module implements directory entries as name→inode mappings stored in directory file data blocks, treating directories as special files whose content is an array of `DirEntry` structures. It provides path resolution that traverses the directory tree component-by-component from root (or current working directory), handling special entries `.` (self) and `..` (parent). The module implements directory creation with proper link count management (parent's link count increases for child's `..` entry), directory deletion with emptiness checking, hard link creation that increments link counts, and symbolic link support where the target path is stored as file data. It does NOT implement file content read/write operations (that's M4) or FUSE integration (M5). The core invariants: every directory entry's inode number refers to an allocated inode in the bitmap; every directory's `.` entry points to itself; every directory's `..` entry points to its parent (root's `..` points to itself); a directory's link count equals 2 plus the number of subdirectories (for each child's `..` reference); and no duplicate names exist within a single directory.
---
## File Structure
```
filesystem/
├── 22_direntry.h        # DirEntry struct, DT_* constants, function declarations
├── 23_direntry.c        # Serialization helpers, entry validation
├── 24_dir_lookup.h      # dir_lookup declaration
├── 25_dir_lookup.c      # Directory scanning implementation
├── 26_dir_modify.h      # dir_add_entry, dir_remove_entry declarations
├── 27_dir_modify.c      # Entry add/remove with link count management
├── 28_path.h            # Path parsing structures, path_resolve declaration
├── 29_path.c            # Component splitting, iterative resolution
├── 30_mkdir_rmdir.h     # fs_mkdir, fs_rmdir declarations
├── 31_mkdir_rmdir.c     # Directory create/delete with special entries
├── 32_link.h            # fs_link, fs_symlink declarations
├── 33_link.c            # Hard link and symbolic link implementation
├── 34_rename.h          # fs_rename declaration
├── 35_rename.c          # Atomic rename/move implementation
├── 36_readdir.h         # fs_readdir declaration, callback typedef
├── 37_readdir.c         # Directory listing implementation
└── tests/
    ├── test_direntry.c       # Entry structure and serialization tests
    ├── test_dir_lookup.c     # Lookup and scanning tests
    ├── test_path.c           # Path resolution tests
    ├── test_mkdir_rmdir.c    # Directory lifecycle tests
    ├── test_link.c           # Hard link and symlink tests
    └── test_rename.c         # Rename operation tests
```
---
## Complete Data Model
### DirEntry Structure
The directory entry is the fundamental mapping from name to inode. Each entry occupies 280 bytes, allowing 14 entries per 4KB block.
```c
// 22_direntry.h
#include <stdint.h>
#include <stdbool.h>
// Directory entry file types (stored in file_type field)
#define DT_UNKNOWN   0   // Unknown type
#define DT_REG       1   // Regular file
#define DT_DIR       2   // Directory
#define DT_LNK       3   // Symbolic link
#define DT_BLK       4   // Block device
#define DT_CHR       5   // Character device
#define DT_FIFO      6   // Named pipe
#define DT_SOCK      7   // Socket
// Constants
#define MAX_NAME_LEN     255
#define DIR_ENTRY_SIZE   280
#define ENTRIES_PER_BLOCK (BLOCK_SIZE / DIR_ENTRY_SIZE)  // 14
/**
 * On-disk directory entry structure.
 * Total size: 280 bytes, packed.
 * 
 * A directory is a special file whose data blocks contain
 * an array of these structures. Unused entries have inode = 0.
 * 
 * Memory layout (280 bytes):
 *   0x00-0x07: inode (8 bytes)
 *   0x08-0x09: rec_len (2 bytes)
 *   0x0A:       name_len (1 byte)
 *   0x0B:       file_type (1 byte)
 *   0x0C-0x10B: name (256 bytes, null-padded)
 *   0x10C-0x117: _padding (12 bytes)
 */
typedef struct {
    uint64_t inode;          // 8 bytes: Inode number (0 = unused/deleted entry)
    uint16_t rec_len;        // 2 bytes: Record length (always DIR_ENTRY_SIZE = 280)
                             // Future: variable-length entries would use this for skipping
    uint8_t  name_len;       // 1 byte: Name length (not including null terminator)
    uint8_t  file_type;      // 1 byte: DT_* constant for file type
    char     name[256];      // 256 bytes: Null-terminated filename, padded
    uint8_t  _padding[12];   // 12 bytes: Pad to 280 bytes total
} __attribute__((packed)) DirEntry;
// Compile-time verification
static_assert(sizeof(DirEntry) == DIR_ENTRY_SIZE, "DirEntry must be exactly 280 bytes");
static_assert(ENTRIES_PER_BLOCK == 14, "Should have 14 entries per block");
```
**Byte Offset Table:**
| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0x00 | 8 | inode | Target inode number, 0 = unused slot |
| 0x08 | 2 | rec_len | Record length (280 for fixed-size) |
| 0x0A | 1 | name_len | Actual name length (≤255) |
| 0x0B | 1 | file_type | DT_REG, DT_DIR, DT_LNK, etc. |
| 0x0C | 256 | name | Null-terminated filename |
| 0x10C | 12 | _padding | Alignment to 280 bytes |
| **Total** | **280** | | |
**Hardware Soul - Cache Line Analysis:**
```
Each DirEntry is 280 bytes (4.375 cache lines of 64 bytes).
Entry spans cache lines:
  - Bytes 0-63: inode, rec_len, name_len, file_type, name[0-51]
  - Bytes 64-127: name[52-115]
  - Bytes 128-191: name[116-179]
  - Bytes 192-255: name[180-243]
  - Bytes 256-279: name[244-255], _padding
For name comparison:
  - Short names (≤52 chars): Only first cache line needed
  - Names 53-255: Additional cache line loads required
Sequential scan of 14-entry block:
  - 14 entries × 4.375 cache lines = 61.25 cache lines
  - Effectively the entire 4KB block (64 cache lines)
  - Prefetch-friendly: sequential access pattern
```
**Why Each Field Exists:**
- **inode**: The target of this name→inode mapping. Zero indicates deleted/unused slot.
- **rec_len**: Record length enables variable-length entries in production filesystems. Fixed at 280 here for simplicity.
- **name_len**: Avoids scanning for null terminator during comparison. Enables `memcmp(name, entry->name, name_len)`.
- **file_type**: Allows `readdir` to return file type without reading the inode (optimization for `ls -F`).
- **name[256]**: Supports maximum filename length of 255 characters plus null terminator.
- **_padding[12]**: Aligns struct to 280 bytes, making 14 entries fit exactly in 4096-byte block.
### PathComponent Structure
Used internally for path parsing.
```c
// 28_path.h
#define MAX_PATH_DEPTH    64   // Maximum path components
#define MAX_PATH_LENGTH   4096 // Maximum total path length
typedef struct {
    char name[MAX_NAME_LEN + 1];  // Component name (null-terminated)
    uint8_t length;               // Name length
} PathComponent;
typedef struct {
    PathComponent components[MAX_PATH_DEPTH];
    int count;                    // Number of components
    bool is_absolute;             // True if path starts with /
} ParsedPath;
```
### DirIterCallback Type
Callback for directory iteration.
```c
// 36_readdir.h
/**
 * Callback invoked for each directory entry.
 * 
 * @param inode      Inode number of the entry
 * @param name       Entry name (null-terminated)
 * @param file_type  DT_* constant
 * @param context    User-provided context pointer
 */
typedef void (*DirIterCallback)(uint64_t inode, const char* name, 
                                uint8_t file_type, void* context);
```

![Directory Entry Memory Layout](./diagrams/tdd-diag-m3-01.svg)

---
## Interface Contracts
### Directory Entry Serialization
```c
// 22_direntry.h
/**
 * Initialize a directory entry.
 *
 * @param entry      Entry to initialize
 * @param inode      Target inode number
 * @param name       Entry name (will be copied and null-padded)
 * @param file_type  DT_* constant
 *
 * Precondition: name != NULL, strlen(name) <= MAX_NAME_LEN
 * Postcondition: entry is fully initialized with rec_len = DIR_ENTRY_SIZE
 */
void direntry_init(DirEntry* entry, uint64_t inode, const char* name, 
                   uint8_t file_type);
/**
 * Check if a directory entry is in use.
 *
 * @param entry Entry to check
 * @return true if entry has valid inode, false if unused/deleted
 */
static inline bool direntry_is_used(const DirEntry* entry) {
    return entry->inode != 0;
}
/**
 * Compare a name against a directory entry.
 *
 * @param entry      Directory entry
 * @param name       Name to compare
 * @param name_len   Length of name
 * @return true if names match, false otherwise
 *
 * Uses name_len from entry for early rejection,
 * then memcmp for full comparison.
 */
bool direntry_name_matches(const DirEntry* entry, const char* name, 
                           size_t name_len);
/**
 * Clear a directory entry (mark as unused).
 *
 * @param entry Entry to clear
 *
 * Postcondition: entry->inode = 0, other fields zeroed
 */
void direntry_clear(DirEntry* entry);
```
### Directory Lookup
```c
// 24_dir_lookup.h
/**
 * Look up a name in a directory.
 *
 * @param fs            FileSystem structure
 * @param dir_inode_num Inode number of directory to search
 * @param name          Name to find (null-terminated)
 * @return Inode number of target, or 0 if not found
 *
 * Errors (in errno):
 *   ENOTDIR - dir_inode_num is not a directory
 *   ENAMETOOLONG - strlen(name) > MAX_NAME_LEN
 *   EIO - Read error from block device
 *
 * Algorithm: Sequential scan of all directory data blocks.
 * Complexity: O(n) where n = number of entries in directory.
 *
 * Note: Returns 0 for both "not found" and errors. Check errno
 * to distinguish. errno = 0 means simply "not found".
 */
uint64_t dir_lookup(FileSystem* fs, uint64_t dir_inode_num, const char* name);
/**
 * Look up a name and retrieve its entry details.
 *
 * @param fs            FileSystem structure
 * @param dir_inode_num Inode number of directory to search
 * @param name          Name to find
 * @param out_entry     Output: entry if found (may be NULL)
 * @param out_block_idx Output: block index containing entry (may be NULL)
 * @param out_entry_idx Output: entry index within block (may be NULL)
 * @return Inode number if found, 0 otherwise
 */
uint64_t dir_lookup_ex(FileSystem* fs, uint64_t dir_inode_num, const char* name,
                       DirEntry* out_entry, uint64_t* out_block_idx, 
                       int* out_entry_idx);
```
### Directory Modification
```c
// 26_dir_modify.h
/**
 * Add an entry to a directory.
 *
 * @param fs            FileSystem structure
 * @param dir_inode_num Inode number of parent directory
 * @param name          Entry name
 * @param target_inode  Inode number being linked
 * @param file_type     DT_* constant for target
 * @return 0 on success, -1 on failure
 *
 * Errors:
 *   EEXIST - Name already exists in directory
 *   ENOTDIR - dir_inode_num is not a directory
 *   ENAMETOOLONG - strlen(name) > MAX_NAME_LEN
 *   ENOSPC - No free blocks to extend directory
 *   EIO - Read/write error
 *
 * Postcondition:
 *   - New entry added to directory
 *   - Directory's mtime and ctime updated
 *   - Target inode's link_count unchanged (caller's responsibility)
 *
 * Algorithm: Scan for unused slot, append if none found.
 */
int dir_add_entry(FileSystem* fs, uint64_t dir_inode_num, const char* name,
                  uint64_t target_inode, uint8_t file_type);
/**
 * Remove an entry from a directory.
 *
 * @param fs            FileSystem structure
 * @param dir_inode_num Inode number of parent directory
 * @param name          Entry name to remove
 * @return 0 on success, -1 on failure
 *
 * Errors:
 *   ENOENT - Name not found in directory
 *   ENOTDIR - dir_inode_num is not a directory
 *   EIO - Read/write error
 *
 * Postcondition:
 *   - Entry cleared (inode = 0)
 *   - Directory's mtime and ctime updated
 *   - Target inode's link_count decremented
 *   - If target's link_count reaches 0, inode is freed
 *
 * Note: This handles both unlink (files) and rmdir (directories).
 * For rmdir, caller must verify directory is empty first.
 */
int dir_remove_entry(FileSystem* fs, uint64_t dir_inode_num, const char* name);
/**
 * Internal add entry without link count update.
 * Used by mkdir to avoid double-increment.
 *
 * @return 0 on success, -1 on failure
 */
int dir_add_entry_internal(FileSystem* fs, uint64_t dir_inode_num, 
                           const char* name, uint64_t target_inode, 
                           uint8_t file_type);
```
### Path Resolution
```c
// 28_path.h
/**
 * Parse a path into components.
 *
 * @param path          Path string to parse
 * @param parsed        Output: parsed path structure
 * @return 0 on success, -1 on failure
 *
 * Errors:
 *   EINVAL - path is NULL or empty
 *   ENAMETOOLONG - Component > MAX_NAME_LEN
 *   EINVAL - Too many components (> MAX_PATH_DEPTH)
 *
 * Examples:
 *   "/home/user/file.txt" → ["home", "user", "file.txt"], absolute
 *   "relative/path"       → ["relative", "path"], relative
 *   "/"                   → [], absolute (empty components, root)
 *   ".."                  → [".."], relative
 */
int path_parse(const char* path, ParsedPath* parsed);
/**
 * Resolve a path to an inode number.
 *
 * @param fs    FileSystem structure
 * @param path  Path to resolve (absolute or relative)
 * @return Inode number, or 0 if not found
 *
 * Errors (in errno):
 *   ENOENT - Path component not found
 *   ENOTDIR - Intermediate component is not a directory
 *   ENAMETOOLONG - Component exceeds MAX_NAME_LEN
 *   ELOOP - Too many symbolic link resolutions (future)
 *   EIO - Read error
 *
 * Algorithm:
 *   1. Start at root (inode 1) if absolute, else cwd (inode 1 for now)
 *   2. For each component:
 *      a. "." → stay in current directory
 *      b. ".." → look up ".." in current directory
 *      c. Otherwise → look up component in current directory
 *   3. Return final inode
 *
 * Complexity: O(depth × entries_per_directory)
 */
uint64_t path_resolve(FileSystem* fs, const char* path);
/**
 * Resolve parent directory and extract final component.
 *
 * @param fs             FileSystem structure
 * @param path           Full path
 * @param last_component Output: final component name (must have MAX_NAME_LEN+1 space)
 * @return Parent directory inode number, or 0 on error
 *
 * Example:
 *   path = "/home/user/file.txt"
 *   Returns: inode of "/home/user"
 *   last_component = "file.txt"
 *
 * Use case: create("/a/b/c") needs parent inode of "/a/b" and name "c"
 */
uint64_t path_resolve_parent(FileSystem* fs, const char* path, 
                             char* last_component);
```
### mkdir and rmdir
```c
// 30_mkdir_rmdir.h
/**
 * Create a directory.
 *
 * @param fs    FileSystem structure
 * @param path  Path of directory to create
 * @return New directory inode number, or 0 on failure
 *
 * Errors:
 *   ENOENT - Parent directory doesn't exist
 *   EEXIST - Name already exists
 *   ENOSPC - No free inodes or blocks
 *   EIO - Read/write error
 *
 * Postcondition:
 *   - New directory inode allocated
 *   - Directory contains "." (self) and ".." (parent) entries
 *   - Parent directory has new entry pointing to child
 *   - Parent's link_count incremented (for child's "..")
 *   - Child's link_count = 2 (parent entry + ".")
 */
uint64_t fs_mkdir(FileSystem* fs, const char* path);
/**
 * Remove a directory.
 *
 * @param fs    FileSystem structure
 * @param path  Path of directory to remove
 * @return 0 on success, -1 on failure
 *
 * Errors:
 *   ENOENT - Directory doesn't exist
 *   ENOTDIR - Path is not a directory
 *   ENOTEMPTY - Directory contains entries other than "." and ".."
 *   EBUSY - Attempting to remove root directory
 *   EIO - Read/write error
 *
 * Postcondition:
 *   - Directory entry removed from parent
 *   - Parent's link_count decremented
 *   - Directory inode and data block freed
 */
int fs_rmdir(FileSystem* fs, const char* path);
/**
 * Check if a directory is empty.
 *
 * @param fs            FileSystem structure
 * @param dir_inode_num Directory inode to check
 * @return 1 if empty, 0 if not empty, -1 on error
 *
 * Empty = contains only "." and ".." entries
 */
int dir_is_empty(FileSystem* fs, uint64_t dir_inode_num);
```
### Hard Links and Symbolic Links
```c
// 32_link.h
/**
 * Create a hard link.
 *
 * @param fs            FileSystem structure
 * @param existing_path Path of existing file
 * @param new_path      Path for new link
 * @return 0 on success, -1 on failure
 *
 * Errors:
 *   ENOENT - existing_path doesn't exist
 *   EEXIST - new_path already exists
 *   EPERM - existing_path is a directory (hard links to dirs not allowed)
 *   ENOSPC - No space for new directory entry
 *   EIO - Read/write error
 *
 * Postcondition:
 *   - New directory entry points to same inode
 *   - Inode's link_count incremented
 *   - Both names refer to same data
 */
int fs_link(FileSystem* fs, const char* existing_path, const char* new_path);
/**
 * Create a symbolic link.
 *
 * @param fs            FileSystem structure
 * @param target_path   Path that symlink will point to
 * @param link_path     Path for the symlink itself
 * @return 0 on success, -1 on failure
 *
 * Errors:
 *   ENOENT - Parent of link_path doesn't exist
 *   EEXIST - link_path already exists
 *   ENOSPC - No free inodes or blocks
 *   ENAMETOOLONG - target_path too long
 *   EIO - Read/write error
 *
 * Postcondition:
 *   - New symlink inode allocated with S_IFLNK mode
 *   - Symlink's data contains target_path string
 *   - Directory entry with DT_LNK type
 */
int fs_symlink(FileSystem* fs, const char* target_path, const char* link_path);
/**
 * Read the target of a symbolic link.
 *
 * @param fs        FileSystem structure
 * @param link_path Path to symlink
 * @param buffer    Output buffer for target path
 * @param size      Buffer size
 * @return Number of bytes read, or -1 on failure
 *
 * Errors:
 *   ENOENT - link_path doesn't exist
 *   EINVAL - link_path is not a symlink
 *   EIO - Read error
 */
int fs_readlink(FileSystem* fs, const char* link_path, char* buffer, size_t size);
```
### Rename
```c
// 34_rename.h
/**
 * Rename/move a file or directory.
 *
 * @param fs       FileSystem structure
 * @param oldpath  Current path
 * @param newpath  New path
 * @return 0 on success, -1 on failure
 *
 * Errors:
 *   ENOENT - oldpath doesn't exist
 *   ENOENT - Parent of newpath doesn't exist
 *   EEXIST - newpath already exists (overwrite not supported in this impl)
 *   EINVAL - oldpath is prefix of newpath (can't move into self)
 *   EIO - Read/write error
 *
 * Postcondition:
 *   - Entry removed from old location
 *   - Entry added at new location
 *   - Same inode, same link_count
 *   - For directory moves: .. entry updated if parent changes
 *   - For directory moves: old parent link_count--, new parent link_count++
 */
int fs_rename(FileSystem* fs, const char* oldpath, const char* newpath);
```
### Directory Listing
```c
// 36_readdir.h
/**
 * List directory contents.
 *
 * @param fs       FileSystem structure
 * @param path     Directory path
 * @param callback Function to call for each entry
 * @param context  User context passed to callback
 * @return 0 on success, -1 on failure
 *
 * Errors:
 *   ENOENT - Directory doesn't exist
 *   ENOTDIR - Path is not a directory
 *   EIO - Read error
 *
 * The callback is invoked for each entry including "." and "..".
 * Iteration stops if callback returns non-zero (not implemented).
 */
int fs_readdir(FileSystem* fs, const char* path, 
               DirIterCallback callback, void* context);
```
---
## Algorithm Specification
### dir_lookup Algorithm
**Input:** `fs`, `dir_inode_num`, `name`
**Output:** Inode number of target, or 0 if not found
**Invariant:** Directory inode is not modified
```
ALGORITHM dir_lookup(fs, dir_inode_num, name):
  // Validate inputs
  name_len ← strlen(name)
  IF name_len > MAX_NAME_LEN:
    errno ← ENAMETOOLONG
    RETURN 0
  // Read directory inode
  dir_inode ← read_inode(fs, dir_inode_num)
  IF dir_inode = NULL:
    RETURN 0  // errno set by read_inode
  // Verify it's a directory
  IF (dir_inode.mode & S_IFDIR) = 0:
    errno ← ENOTDIR
    RETURN 0
  // Calculate number of blocks in directory
  dir_size ← dir_inode.size
  num_blocks ← (dir_size + BLOCK_SIZE - 1) / BLOCK_SIZE
  IF num_blocks = 0:
    num_blocks ← 1  // Empty directory still has one block
  // Allocate block buffer
  block_buffer ← allocate(BLOCK_SIZE)
  entries ← cast(block_buffer to DirEntry*)
  errno ← 0  // Clear errno - "not found" is not an error
  // Scan each block
  FOR block_idx FROM 0 TO num_blocks - 1:
    // Get physical block number
    phys_block ← get_file_block(fs, dir_inode, block_idx, block_buffer)
    IF phys_block = 0:
      CONTINUE  // Hole in directory (shouldn't happen, skip)
    // Read directory block
    IF read_block(fs->dev, phys_block, block_buffer) < 0:
      errno ← EIO
      free(block_buffer)
      RETURN 0
    // Scan entries in this block (14 entries per block)
    FOR entry_idx FROM 0 TO ENTRIES_PER_BLOCK - 1:
      entry ← &entries[entry_idx]
      // Skip unused entries
      IF entry.inode = 0:
        CONTINUE
      // Compare names
      IF entry.name_len = name_len AND
         memcmp(entry.name, name, name_len) = 0:
        // Found!
        result ← entry.inode
        free(block_buffer)
        RETURN result
  // Not found
  free(block_buffer)
  RETURN 0  // errno already 0
```
**Complexity:** O(n) where n = number of directory entries. Each entry requires a name comparison.
**Hardware Soul - Access Pattern:**
```
For directory with 140 entries (10 blocks):
  Block reads: 10 (sequential, prefetch-friendly)
  Cache lines: 640 (10 blocks × 64 cache lines)
  Name comparisons: ~70 average (assuming uniform distribution)
Optimization opportunity: Hash-based indexing (ext4 htree)
would reduce lookups to O(1) average, but adds complexity.
```

![Directory as File Concept](./diagrams/tdd-diag-m3-02.svg)

### dir_add_entry Algorithm
**Input:** `fs`, `dir_inode_num`, `name`, `target_inode`, `file_type`
**Output:** 0 on success, -1 on failure
**Postcondition:** New entry added to directory
```
ALGORITHM dir_add_entry(fs, dir_inode_num, name, target_inode, file_type):
  // Validate inputs
  name_len ← strlen(name)
  IF name_len > MAX_NAME_LEN:
    errno ← ENAMETOOLONG
    RETURN -1
  IF name_len = 0:
    errno ← EINVAL
    RETURN -1
  // Verify directory exists and is a directory
  dir_inode ← read_inode(fs, dir_inode_num)
  IF dir_inode = NULL:
    RETURN -1
  IF (dir_inode.mode & S_IFDIR) = 0:
    errno ← ENOTDIR
    RETURN -1
  // Check for duplicate
  existing ← dir_lookup(fs, dir_inode_num, name)
  IF existing ≠ 0:
    errno ← EEXIST
    RETURN -1
  block_buffer ← allocate(BLOCK_SIZE)
  entries ← cast(block_buffer to DirEntry*)
  // Calculate current directory size
  num_blocks ← (dir_inode.size + BLOCK_SIZE - 1) / BLOCK_SIZE
  IF num_blocks = 0:
    num_blocks ← 1
  // Phase 1: Scan for unused slot
  FOR block_idx FROM 0 TO num_blocks - 1:
    phys_block ← get_file_block(fs, dir_inode, block_idx, block_buffer)
    IF phys_block = 0:
      // Need to allocate this block
      phys_block ← get_or_alloc_block(fs, dir_inode_num, dir_inode, 
                                       block_idx, block_buffer, bitmap_buf)
      IF phys_block = 0:
        free(block_buffer)
        RETURN -1  // ENOSPC
      memset(block_buffer, 0, BLOCK_SIZE)
    ELSE:
      IF read_block(fs->dev, phys_block, block_buffer) < 0:
        free(block_buffer)
        RETURN -1
    // Look for free slot
    FOR entry_idx FROM 0 TO ENTRIES_PER_BLOCK - 1:
      IF entries[entry_idx].inode = 0:
        // Found free slot!
        direntry_init(&entries[entry_idx], target_inode, name, file_type)
        // Write block back
        IF write_block(fs->dev, phys_block, block_buffer) < 0:
          free(block_buffer)
          RETURN -1
        // Update directory size if needed
        entry_offset ← block_idx * BLOCK_SIZE + entry_idx * DIR_ENTRY_SIZE
        IF entry_offset + DIR_ENTRY_SIZE > dir_inode.size:
          dir_inode.size ← entry_offset + DIR_ENTRY_SIZE
        // Update timestamps
        dir_inode.mtime ← current_time()
        dir_inode.ctime ← dir_inode.mtime
        write_inode(fs, dir_inode_num, dir_inode)
        free(block_buffer)
        RETURN 0
  // Phase 2: No free slot found - extend directory
  new_block_idx ← num_blocks
  // Allocate new block
  phys_block ← get_or_alloc_block(fs, dir_inode_num, dir_inode,
                                   new_block_idx, block_buffer, bitmap_buf)
  IF phys_block = 0:
    free(block_buffer)
    RETURN -1
  // Initialize new block with entry at position 0
  memset(block_buffer, 0, BLOCK_SIZE)
  direntry_init(&entries[0], target_inode, name, file_type)
  IF write_block(fs->dev, phys_block, block_buffer) < 0:
    free(block_buffer)
    RETURN -1
  // Update directory size
  dir_inode.size ← (new_block_idx + 1) * BLOCK_SIZE
  dir_inode.mtime ← current_time()
  dir_inode.ctime ← dir_inode.mtime
  write_inode(fs, dir_inode_num, dir_inode)
  free(block_buffer)
  RETURN 0
```

![Directory Lookup Scan](./diagrams/tdd-diag-m3-03.svg)

### dir_remove_entry Algorithm
**Input:** `fs`, `dir_inode_num`, `name`
**Output:** 0 on success, -1 on failure
**Postcondition:** Entry cleared, link count updated, inode potentially freed
```
ALGORITHM dir_remove_entry(fs, dir_inode_num, name):
  name_len ← strlen(name)
  // Verify directory
  dir_inode ← read_inode(fs, dir_inode_num)
  IF dir_inode = NULL OR (dir_inode.mode & S_IFDIR) = 0:
    errno ← ENOTDIR
    RETURN -1
  block_buffer ← allocate(BLOCK_SIZE)
  entries ← cast(block_buffer to DirEntry*)
  num_blocks ← (dir_inode.size + BLOCK_SIZE - 1) / BLOCK_SIZE
  // Scan for entry
  FOR block_idx FROM 0 TO num_blocks - 1:
    phys_block ← get_file_block(fs, dir_inode, block_idx, block_buffer)
    IF phys_block = 0:
      CONTINUE
    IF read_block(fs->dev, phys_block, block_buffer) < 0:
      free(block_buffer)
      RETURN -1
    FOR entry_idx FROM 0 TO ENTRIES_PER_BLOCK - 1:
      entry ← &entries[entry_idx]
      IF entry.inode ≠ 0 AND
         entry.name_len = name_len AND
         memcmp(entry.name, name, name_len) = 0:
        // Found! Save target inode before clearing
        target_inode_num ← entry.inode
        // Clear the entry
        direntry_clear(entry)
        // Write block back
        IF write_block(fs->dev, phys_block, block_buffer) < 0:
          free(block_buffer)
          RETURN -1
        // Update directory timestamps
        dir_inode.mtime ← current_time()
        dir_inode.ctime ← dir_inode.mtime
        write_inode(fs, dir_inode_num, dir_inode)
        // Handle target inode's link count
        target_inode ← read_inode(fs, target_inode_num)
        IF target_inode ≠ NULL:
          IF target_inode.link_count > 0:
            target_inode.link_count ← target_inode.link_count - 1
          IF target_inode.link_count = 0:
            // No more references - free the inode
            inode_free_all_blocks(fs, target_inode, bitmap_buf)
            inode_deallocate(fs, target_inode_num)
          ELSE:
            // Still has references - just update
            target_inode.ctime ← current_time()
            write_inode(fs, target_inode_num, target_inode)
        free(block_buffer)
        RETURN 0
  // Not found
  errno ← ENOENT
  free(block_buffer)
  RETURN -1
```
### path_resolve Algorithm
**Input:** `fs`, `path`
**Output:** Inode number, or 0 if not found
**Invariant:** No filesystem modifications
```
ALGORITHM path_resolve(fs, path):
  // Handle empty path
  IF path = NULL OR path[0] = '\0':
    errno ← EINVAL
    RETURN 0
  // Parse path into components
  parsed ← allocate(ParsedPath)
  IF path_parse(path, parsed) < 0:
    free(parsed)
    RETURN 0  // errno set by path_parse
  // Determine starting point
  IF parsed.is_absolute:
    current_inode ← 1  // Root
  ELSE:
    // Relative path - would use cwd in real system
    // For simplicity, use root
    current_inode ← 1
  // Handle "/" case (empty components after parsing)
  IF parsed.count = 0:
    free(parsed)
    RETURN current_inode
  // Resolve each component
  FOR i FROM 0 TO parsed.count - 1:
    component ← parsed.components[i].name
    // Handle special entries
    IF strcmp(component, ".") = 0:
      // Stay in current directory
      CONTINUE
    IF strcmp(component, "..") = 0:
      // Go to parent directory
      parent ← dir_lookup(fs, current_inode, "..")
      IF parent = 0:
        // At root or error - stay at current
        // Root's ".." points to itself
        IF current_inode = 1:
          CONTINUE
        errno ← ENOENT
        free(parsed)
        RETURN 0
      current_inode ← parent
      CONTINUE
    // Regular component - look up in current directory
    next_inode ← dir_lookup(fs, current_inode, component)
    IF next_inode = 0:
      // Component not found
      IF errno = 0:
        errno ← ENOENT
      free(parsed)
      RETURN 0
    // If not the last component, verify it's a directory
    IF i < parsed.count - 1:
      next_inode_data ← read_inode(fs, next_inode)
      IF next_inode_data = NULL:
        free(parsed)
        RETURN 0
      IF (next_inode_data.mode & S_IFDIR) = 0:
        errno ← ENOTDIR
        free(parsed)
        RETURN 0
    current_inode ← next_inode
  free(parsed)
  RETURN current_inode
```
**Path Parse Sub-Algorithm:**
```
ALGORITHM path_parse(path, parsed):
  // Initialize
  memset(parsed, 0, sizeof(ParsedPath))
  parsed->is_absolute ← (path[0] = '/')
  p ← path
  // Skip leading slashes
  WHILE p[0] = '/':
    p ← p + 1
  // Parse components
  WHILE p[0] ≠ '\0' AND parsed->count < MAX_PATH_DEPTH:
    // Find end of component
    start ← p
    WHILE p[0] ≠ '\0' AND p[0] ≠ '/':
      p ← p + 1
    component_len ← p - start
    // Validate length
    IF component_len > MAX_NAME_LEN:
      errno ← ENAMETOOLONG
      RETURN -1
    // Skip empty components (consecutive slashes)
    IF component_len = 0:
      WHILE p[0] = '/':
        p ← p + 1
      CONTINUE
    // Copy component
    memcpy(parsed->components[parsed->count].name, start, component_len)
    parsed->components[parsed->count].name[component_len] ← '\0'
    parsed->components[parsed->count].length ← component_len
    parsed->count ← parsed->count + 1
    // Skip slashes
    WHILE p[0] = '/':
      p ← p + 1
  // Check for overflow
  IF p[0] ≠ '\0':
    errno ← EINVAL  // Too many components
    RETURN -1
  RETURN 0
```

![Path Resolution Chain](./diagrams/tdd-diag-m3-04.svg)

### fs_mkdir Algorithm
**Input:** `fs`, `path`
**Output:** New directory inode number, or 0 on failure
**Postcondition:** Directory created with `.` and `..`, parent link count updated
```
ALGORITHM fs_mkdir(fs, path):
  // Resolve parent and extract final component
  name ← allocate(MAX_NAME_LEN + 1)
  parent_inode_num ← path_resolve_parent(fs, path, name)
  IF parent_inode_num = 0:
    free(name)
    RETURN 0  // errno set
  // Check if name already exists
  IF dir_lookup(fs, parent_inode_num, name) ≠ 0:
    errno ← EEXIST
    free(name)
    RETURN 0
  // Verify parent is a directory
  parent_inode ← read_inode(fs, parent_inode_num)
  IF parent_inode = NULL OR (parent_inode.mode & S_IFDIR) = 0:
    errno ← ENOTDIR
    free(name)
    RETURN 0
  // Allocate new inode
  new_inode_num ← inode_bitmap_alloc(fs->dev, fs->sb, bitmap_buf)
  IF new_inode_num = 0:
    errno ← ENOSPC
    free(name)
    RETURN 0
  // Allocate data block for directory entries
  new_block ← block_bitmap_alloc(fs->dev, fs->sb, bitmap_buf)
  IF new_block = 0:
    // Rollback: free inode
    inode_bitmap_free(fs->dev, fs->sb, new_inode_num, bitmap_buf)
    errno ← ENOSPC
    free(name)
    RETURN 0
  // Initialize new inode as directory
  new_inode ← allocate(Inode)
  inode_init(new_inode, S_IFDIR | 0755, 0, 0)
  new_inode.link_count ← 2  // Parent entry + "."
  new_inode.size ← BLOCK_SIZE
  new_inode.blocks ← 8  // One 4KB block = 8 × 512
  new_inode.direct[0] ← new_block
  write_inode(fs, new_inode_num, new_inode)
  // Initialize directory block with "." and ".."
  dir_buffer ← allocate(BLOCK_SIZE)
  memset(dir_buffer, 0, BLOCK_SIZE)
  entries ← cast(dir_buffer to DirEntry*)
  // Entry 0: "."
  direntry_init(&entries[0], new_inode_num, ".", DT_DIR)
  // Entry 1: ".."
  direntry_init(&entries[1], parent_inode_num, "..", DT_DIR)
  entries[1].rec_len ← BLOCK_SIZE - DIR_ENTRY_SIZE  // Rest of block
  write_block(fs->dev, new_block, dir_buffer)
  // Add entry in parent directory (internal, no link count update)
  dir_add_entry_internal(fs, parent_inode_num, name, new_inode_num, DT_DIR)
  // Increment parent's link count (for child's ".." reference)
  parent_inode.link_count ← parent_inode.link_count + 1
  parent_inode.mtime ← current_time()
  parent_inode.ctime ← parent_inode.mtime
  write_inode(fs, parent_inode_num, parent_inode)
  // Update superblock
  fs->sb->free_inodes ← fs->sb->free_inodes - 1
  fs->sb->free_blocks ← fs->sb->free_blocks - 1
  write_block(fs->dev, 0, fs->sb)
  free(name)
  free(new_inode)
  free(dir_buffer)
  RETURN new_inode_num
```
**Link Count Semantics:**
```
New directory link_count = 2:
  1. Entry in parent directory (the name user specified)
  2. The "." entry inside the directory
Parent's link_count increases by 1:
  - Because child's ".." points to parent
  - Each subdirectory adds 1 to parent's link_count
Example:
  mkdir /home/user/docs
  /home/user's link_count after:
    - "." entry: +1 (implicit, counted as 2 for self)
    - "docs" entry in user: counted as parent entry
    - ".." in docs points to user: +1
  Result: /home/user link_count increases by 1
```

![Special Entries . and ..](./diagrams/tdd-diag-m3-05.svg)

### fs_rmdir Algorithm
**Input:** `fs`, `path`
**Output:** 0 on success, -1 on failure
**Postcondition:** Directory removed, parent link count updated
```
ALGORITHM fs_rmdir(fs, path):
  // Resolve the directory
  dir_inode_num ← path_resolve(fs, path)
  IF dir_inode_num = 0:
    RETURN -1  // errno set
  // Can't remove root
  IF dir_inode_num = 1:
    errno ← EBUSY
    RETURN -1
  // Verify it's a directory
  dir_inode ← read_inode(fs, dir_inode_num)
  IF dir_inode = NULL OR (dir_inode.mode & S_IFDIR) = 0:
    errno ← ENOTDIR
    RETURN -1
  // Check if empty
  empty ← dir_is_empty(fs, dir_inode_num)
  IF empty < 0:
    RETURN -1  // Error
  IF empty = 0:
    errno ← ENOTEMPTY
    RETURN -1
  // Get parent
  name ← allocate(MAX_NAME_LEN + 1)
  parent_inode_num ← path_resolve_parent(fs, path, name)
  IF parent_inode_num = 0:
    free(name)
    RETURN -1
  // Remove entry from parent
  // This decrements dir_inode's link_count
  IF dir_remove_entry(fs, parent_inode_num, name) < 0:
    free(name)
    RETURN -1
  // dir_remove_entry decremented link_count by 1
  // For directories, we need to decrement once more for the "." entry
  dir_inode ← read_inode(fs, dir_inode_num)
  IF dir_inode.link_count > 0:
    dir_inode.link_count ← dir_inode.link_count - 1
  // If link_count is 0, free the inode
  IF dir_inode.link_count = 0:
    // Free data block
    IF dir_inode.direct[0] ≠ 0:
      block_bitmap_free(fs->dev, fs->sb, dir_inode.direct[0], bitmap_buf)
      fs->sb->free_blocks ← fs->sb->free_blocks + 1
    // Free inode
    inode_deallocate(fs, dir_inode_num)
  ELSE:
    // Shouldn't happen for empty directory
    write_inode(fs, dir_inode_num, dir_inode)
  // Decrement parent's link count (removing child's ".." reference)
  parent_inode ← read_inode(fs, parent_inode_num)
  IF parent_inode.link_count > 0:
    parent_inode.link_count ← parent_inode.link_count - 1
  parent_inode.mtime ← current_time()
  parent_inode.ctime ← parent_inode.mtime
  write_inode(fs, parent_inode_num, parent_inode)
  // Update superblock
  write_block(fs->dev, 0, fs->sb)
  free(name)
  RETURN 0
```
### dir_is_empty Algorithm
```
ALGORITHM dir_is_empty(fs, dir_inode_num):
  dir_inode ← read_inode(fs, dir_inode_num)
  IF dir_inode = NULL:
    RETURN -1
  block_buffer ← allocate(BLOCK_SIZE)
  entries ← cast(block_buffer to DirEntry*)
  num_blocks ← (dir_inode.size + BLOCK_SIZE - 1) / BLOCK_SIZE
  FOR block_idx FROM 0 TO num_blocks - 1:
    phys_block ← get_file_block(fs, dir_inode, block_idx, block_buffer)
    IF phys_block = 0:
      CONTINUE
    IF read_block(fs->dev, phys_block, block_buffer) < 0:
      free(block_buffer)
      RETURN -1
    FOR entry_idx FROM 0 TO ENTRIES_PER_BLOCK - 1:
      entry ← &entries[entry_idx]
      IF entry.inode ≠ 0:
        // Found an entry - check if it's "." or ".."
        IF strcmp(entry.name, ".") ≠ 0 AND strcmp(entry.name, "..") ≠ 0:
          // Found a real entry - not empty
          free(block_buffer)
          RETURN 0
  // Only found "." and ".." (or nothing)
  free(block_buffer)
  RETURN 1
```

![mkdir Operation Sequence](./diagrams/tdd-diag-m3-06.svg)

### fs_link Algorithm (Hard Links)
```
ALGORITHM fs_link(fs, existing_path, new_path):
  // Resolve existing file
  existing_inode_num ← path_resolve(fs, existing_path)
  IF existing_inode_num = 0:
    RETURN -1
  // Read existing inode
  existing_inode ← read_inode(fs, existing_inode_num)
  IF existing_inode = NULL:
    RETURN -1
  // Can't hard link to directories
  IF (existing_inode.mode & S_IFDIR) ≠ 0:
    errno ← EPERM
    RETURN -1
  // Resolve parent of new path
  name ← allocate(MAX_NAME_LEN + 1)
  parent_inode_num ← path_resolve_parent(fs, new_path, name)
  IF parent_inode_num = 0:
    free(name)
    RETURN -1
  // Check if name already exists
  IF dir_lookup(fs, parent_inode_num, name) ≠ 0:
    errno ← EEXIST
    free(name)
    RETURN -1
  // Determine file type
  file_type ← DT_REG
  IF (existing_inode.mode & S_IFLNK) ≠ 0:
    file_type ← DT_LNK
  // Add entry in parent directory
  IF dir_add_entry(fs, parent_inode_num, name, existing_inode_num, file_type) < 0:
    free(name)
    RETURN -1
  // Increment link count
  existing_inode.link_count ← existing_inode.link_count + 1
  existing_inode.ctime ← current_time()
  write_inode(fs, existing_inode_num, existing_inode)
  free(name)
  RETURN 0
```
**Why No Hard Links to Directories:**
```
If /a/b and /a/c both pointed to the same directory:
/a/b/../c/.. would create an infinite loop
find /a/b would never terminate
du would count files multiple times
fsck would detect "directory loop"
The filesystem tree must remain a tree, not a graph.
Symlinks to directories are allowed because they're
resolved at access time, not stored in the structure.
```
### fs_symlink Algorithm
```
ALGORITHM fs_symlink(fs, target_path, link_path):
  // Resolve parent of link path
  name ← allocate(MAX_NAME_LEN + 1)
  parent_inode_num ← path_resolve_parent(fs, link_path, name)
  IF parent_inode_num = 0:
    free(name)
    RETURN -1
  // Check if name already exists
  IF dir_lookup(fs, parent_inode_num, name) ≠ 0:
    errno ← EEXIST
    free(name)
    RETURN -1
  // Allocate new inode
  new_inode_num ← inode_bitmap_alloc(fs->dev, fs->sb, bitmap_buf)
  IF new_inode_num = 0:
    errno ← ENOSPC
    free(name)
    RETURN -1
  // Initialize symlink inode
  new_inode ← allocate(Inode)
  inode_init(new_inode, S_IFLNK | 0777, 0, 0)  // Symlinks typically 777
  new_inode.link_count ← 1
  target_len ← strlen(target_path)
  new_inode.size ← target_len
  // Store target path
  IF target_len < 48:
    // Fast path: store in direct pointer space
    // (These would normally be block pointers, but for symlinks
    // we can use the space for the target string)
    memcpy(&new_inode.direct, target_path, target_len)
    new_inode.blocks ← 0
  ELSE:
    // Need to allocate a block for longer paths
    new_block ← block_bitmap_alloc(fs->dev, fs->sb, bitmap_buf)
    IF new_block = 0:
      inode_bitmap_free(fs->dev, fs->sb, new_inode_num, bitmap_buf)
      errno ← ENOSPC
      free(name)
      free(new_inode)
      RETURN -1
    block_buffer ← allocate(BLOCK_SIZE)
    memset(block_buffer, 0, BLOCK_SIZE)
    memcpy(block_buffer, target_path, target_len)
    write_block(fs->dev, new_block, block_buffer)
    new_inode.direct[0] ← new_block
    new_inode.blocks ← 8
    fs->sb->free_blocks ← fs->sb->free_blocks - 1
    free(block_buffer)
  write_inode(fs, new_inode_num, new_inode)
  // Add entry in parent directory
  dir_add_entry(fs, parent_inode_num, name, new_inode_num, DT_LNK)
  // Update superblock
  fs->sb->free_inodes ← fs->sb->free_inodes - 1
  write_block(fs->dev, 0, fs->sb)
  free(name)
  free(new_inode)
  RETURN 0
```
### fs_rename Algorithm
```
ALGORITHM fs_rename(fs, oldpath, newpath):
  // Resolve old path
  old_name ← allocate(MAX_NAME_LEN + 1)
  old_parent ← path_resolve_parent(fs, oldpath, old_name)
  IF old_parent = 0:
    free(old_name)
    RETURN -1
  old_inode ← dir_lookup(fs, old_parent, old_name)
  IF old_inode = 0:
    errno ← ENOENT
    free(old_name)
    RETURN -1
  // Resolve new path
  new_name ← allocate(MAX_NAME_LEN + 1)
  new_parent ← path_resolve_parent(fs, newpath, new_name)
  IF new_parent = 0:
    free(old_name)
    free(new_name)
    RETURN -1
  // Check if target exists
  existing ← dir_lookup(fs, new_parent, new_name)
  IF existing ≠ 0:
    // For simplicity, don't support overwriting
    errno ← EEXIST
    free(old_name)
    free(new_name)
    RETURN -1
  // Get inode info
  inode_data ← read_inode(fs, old_inode)
  IF inode_data = NULL:
    free(old_name)
    free(new_name)
    RETURN -1
  // Determine file type
  file_type ← DT_REG
  IF (inode_data.mode & S_IFDIR) ≠ 0:
    file_type ← DT_DIR
  ELSE IF (inode_data.mode & S_IFLNK) ≠ 0:
    file_type ← DT_LNK
  // Add entry in new location
  IF dir_add_entry_internal(fs, new_parent, new_name, old_inode, file_type) < 0:
    free(old_name)
    free(new_name)
    RETURN -1
  // Remove entry from old location
  // (Must clear manually to avoid link count decrement)
  remove_result ← dir_remove_entry_no_link_decrement(fs, old_parent, old_name)
  IF remove_result < 0:
    // Rollback: remove new entry
    dir_remove_entry(fs, new_parent, new_name)
    free(old_name)
    free(new_name)
    RETURN -1
  // Handle directory move: update .. entry
  IF file_type = DT_DIR AND old_parent ≠ new_parent:
    // Update .. in moved directory
    dir_inode ← read_inode(fs, old_inode)
    IF dir_inode ≠ NULL AND dir_inode.direct[0] ≠ 0:
      dir_block ← allocate(BLOCK_SIZE)
      read_block(fs->dev, dir_inode.direct[0], dir_block)
      entries ← cast(dir_block to DirEntry*)
      // Find .. entry
      FOR i FROM 0 TO ENTRIES_PER_BLOCK - 1:
        IF entries[i].inode ≠ 0 AND strcmp(entries[i].name, "..") = 0:
          // Decrement old parent's link count
          old_parent_inode ← read_inode(fs, old_parent)
          IF old_parent_inode.link_count > 0:
            old_parent_inode.link_count ← old_parent_inode.link_count - 1
          old_parent_inode.ctime ← current_time()
          write_inode(fs, old_parent, old_parent_inode)
          // Update .. to point to new parent
          entries[i].inode ← new_parent
          write_block(fs->dev, dir_inode.direct[0], dir_block)
          // Increment new parent's link count
          new_parent_inode ← read_inode(fs, new_parent)
          new_parent_inode.link_count ← new_parent_inode.link_count + 1
          new_parent_inode.ctime ← current_time()
          write_inode(fs, new_parent, new_parent_inode)
          BREAK
      free(dir_block)
  // Update timestamps on both parents
  update_parent_timestamps(fs, old_parent)
  update_parent_timestamps(fs, new_parent)
  free(old_name)
  free(new_name)
  RETURN 0
```

![Link Count State Transitions](./diagrams/tdd-diag-m3-07.svg)

---
## Error Handling Matrix
| Error | errno | Detected By | Recovery | User-Visible Message |
|-------|-------|-------------|----------|---------------------|
| Name too long | ENAMETOOLONG | dir_lookup, dir_add_entry | Return early, no modifications | "File name too long" |
| Not a directory | ENOTDIR | path_resolve, dir_* | Return error, no modifications | "Not a directory" |
| Name not found | ENOENT | dir_lookup, path_resolve | Return 0, errno=0 (not an error) | (Silent - normal case) |
| Name already exists | EEXIST | dir_add_entry, fs_mkdir | Return error, no modifications | "File exists" |
| Directory not empty | ENOTEMPTY | fs_rmdir | Return error, no modifications | "Directory not empty" |
| Cannot remove root | EBUSY | fs_rmdir | Return error | "Device or resource busy" |
| Hard link to directory | EPERM | fs_link | Return error, no modifications | "Operation not permitted" |
| No free inodes | ENOSPC | fs_mkdir, fs_symlink | Rollback allocations | "No space left on device" |
| No free blocks | ENOSPC | fs_mkdir, fs_symlink | Rollback allocations | "No space left on device" |
| Block read error | EIO | Any read operation | Return error, partial state possible | "Input/output error" |
| Block write error | EIO | Any write operation | Return error, state may be inconsistent | "Input/output error" |
| Invalid path | EINVAL | path_parse | Return error | "Invalid argument" |
| Target exists (rename) | EEXIST | fs_rename | Return error | "File exists" |
---
## Implementation Sequence with Checkpoints
### Phase 1: Directory Entry Structure (1-2 hours)
**Files:** `22_direntry.h`, `23_direntry.c`
**Implementation Steps:**
1. Define `DirEntry` struct with exact byte layout
2. Add `static_assert` for size == 280
3. Implement `direntry_init`
4. Implement `direntry_is_used`
5. Implement `direntry_name_matches`
6. Implement `direntry_clear`
**Checkpoint 1:**
```
$ make test_direntry
$ ./test_direntry
Testing directory entry structure...
  sizeof(DirEntry) = 280: OK
  offsetof(inode) = 0: OK
  offsetof(rec_len) = 8: OK
  offsetof(name_len) = 10: OK
  offsetof(file_type) = 11: OK
  offsetof(name) = 12: OK
Testing direntry_init...
  Entry initialized correctly: OK
  Name null-terminated: OK
  rec_len = 280: OK
Testing name matching...
  "test" matches "test": OK
  "test" != "testing": OK
  "test" != "Test": OK (case-sensitive)
Testing clear...
  Cleared entry: inode = 0: OK
All directory entry tests passed!
```
### Phase 2: Directory Lookup (2-3 hours)
**Files:** `24_dir_lookup.h`, `25_dir_lookup.c`
**Implementation Steps:**
1. Implement basic `dir_lookup` scanning directory blocks
2. Handle `ENOTDIR` and `ENAMETOOLONG` errors
3. Implement sequential scan of entries
4. Implement `dir_lookup_ex` with position information
**Checkpoint 2:**
```
$ make test_dir_lookup
$ ./test_dir_lookup
Testing directory lookup...
  Lookup in empty directory: 0 (not found): OK
  Lookup "." in root: 1: OK
  Lookup ".." in root: 1: OK
  Lookup non-existent name: 0: OK
  Lookup in non-directory: -1, ENOTDIR: OK
  Lookup with name > 255 chars: -1, ENAMETOOLONG: OK
Testing with populated directory...
  Created 10 files: OK
  Lookup each by name: all found: OK
  Lookup non-existent: not found: OK
All lookup tests passed!
```
### Phase 3: Directory Add Entry (2-3 hours)
**Files:** `26_dir_modify.h`, `27_dir_modify.c` (add functions)
**Implementation Steps:**
1. Implement duplicate check
2. Implement free slot scanning
3. Implement directory extension when no free slot
4. Update directory size and timestamps
**Checkpoint 3:**
```
$ make test_dir_modify
$ ./test_dir_modify
Testing directory add entry...
  Add first entry: OK
  Add second entry: OK
  Verify both exist: OK
  Add duplicate name: -1, EEXIST: OK
  Fill directory (14 entries in first block): OK
  Add 15th entry (forces second block): OK
  Verify directory size increased: OK
All add entry tests passed!
```
### Phase 4: Directory Remove Entry (2-3 hours)
**Files:** Update `27_dir_modify.c`
**Implementation Steps:**
1. Implement entry scanning and removal
2. Implement link count decrement
3. Implement inode freeing when link_count = 0
4. Update directory timestamps
**Checkpoint 4:**
```
$ ./test_dir_modify
... add entry tests ...
Testing directory remove entry...
  Remove existing entry: OK
  Verify entry is cleared: OK
  Verify link count decremented: OK
  Remove non-existent: -1, ENOENT: OK
  Remove until link_count = 0:
    Link count reached 0: OK
    Inode freed (bitmap cleared): OK
    Blocks freed: OK
All remove entry tests passed!
```
### Phase 5: Path Resolution (3-4 hours)
**Files:** `28_path.h`, `29_path.c`
**Implementation Steps:**
1. Implement `path_parse` component extraction
2. Handle absolute vs relative paths
3. Implement `path_resolve` iterative lookup
4. Handle `.` and `..` specially
5. Implement `path_resolve_parent`
**Checkpoint 5:**
```
$ make test_path
$ ./test_path
Testing path parsing...
  "/home/user/file" → ["home", "user", "file"]: OK
  "relative/path" → relative: OK
  "/" → empty, absolute: OK
  "///multiple///slashes///" → handled: OK
  ".." → [".."]: OK
  "a/b/c/d/e/f/g/h" (8 deep): OK
Testing path resolution...
  Resolve "/": 1 (root): OK
  Resolve "/.": 1 (root): OK
  Resolve "/..": 1 (root, parent of root): OK
  Resolve "/home" after mkdir: found: OK
  Resolve "/home/user" after nested mkdir: found: OK
  Resolve "/nonexistent": 0, ENOENT: OK
  Resolve "/home/nonexistent": 0, ENOENT: OK
Testing path_resolve_parent...
  Parent of "/a/b/c": inode of "/a/b", name = "c": OK
  Parent of "/file": root inode, name = "file": OK
All path tests passed!
```
### Phase 6: mkdir and rmdir (2-3 hours)
**Files:** `30_mkdir_rmdir.h`, `31_mkdir_rmdir.c`
**Implementation Steps:**
1. Implement `fs_mkdir` with inode and block allocation
2. Create `.` and `..` entries
3. Update parent link count
4. Implement `dir_is_empty`
5. Implement `fs_rmdir` with empty check
6. Update parent link count on rmdir
**Checkpoint 6:**
```
$ make test_mkdir_rmdir
$ ./test_mkdir_rmdir
Testing mkdir...
  mkdir "/testdir": inode 2: OK
  Verify "." entry points to self: OK
  Verify ".." entry points to root: OK
  Verify parent link count = 3: OK
  mkdir "/testdir/subdir": OK
  Verify parent (testdir) link count = 4: OK
  Verify child (subdir) link count = 2: OK
Testing rmdir...
  rmdir non-empty directory: -1, ENOTEMPTY: OK
  Create and remove file in subdir: OK
  rmdir empty subdir: OK
  Verify parent link count decreased: OK
  rmdir root: -1, EBUSY: OK
  rmdir non-existent: -1, ENOENT: OK
All mkdir/rmdir tests passed!
```
### Phase 7: Hard Links and Symlinks (2-3 hours)
**Files:** `32_link.h`, `33_link.c`
**Implementation Steps:**
1. Implement `fs_link` with link count increment
2. Block hard links to directories
3. Implement `fs_symlink` with target storage
4. Implement fast-path for short symlink targets
5. Implement `fs_readlink`
**Checkpoint 7:**
```
$ make test_link
$ ./test_link
Testing hard links...
  Create file "/original": OK
  Link "/link_to_original": OK
  Verify same inode: OK
  Verify link_count = 2: OK
  Read from both: same content: OK
  Unlink original: OK
  Verify link_count = 1: OK
  Verify link still works: OK
  Hard link to directory: -1, EPERM: OK
Testing symbolic links...
  symlink "/link_to_file" -> "/target/file": OK
  Verify symlink exists: OK
  Verify S_IFLNK mode: OK
  readlink returns target: OK
  Symlink to directory: OK (allowed)
  Long symlink (>48 chars): uses data block: OK
All link tests passed!
```
### Phase 8: Rename Operation (2-3 hours)
**Files:** `34_rename.h`, `35_rename.c`
**Implementation Steps:**
1. Implement basic rename within same directory
2. Implement cross-directory rename
3. Handle directory renames (update `..`)
4. Update parent link counts for directory moves
**Checkpoint 8:**
```
$ make test_rename
$ ./test_rename
Testing rename...
  Rename file in same directory: OK
  Verify old name gone: OK
  Verify new name exists: OK
  Verify same inode: OK
  Rename across directories: OK
  Verify parent directories: OK
  Rename directory: OK
  Verify ".." updated: OK
  Verify old parent link count decreased: OK
  Verify new parent link count increased: OK
  Rename to existing name: -1, EEXIST: OK
  Rename non-existent: -1, ENOENT: OK
All rename tests passed!
```
### Phase 9: Integration Testing (2-3 hours)
**Files:** `36_readdir.h`, `37_readdir.c`, integration tests
**Implementation Steps:**
1. Implement `fs_readdir` with callback
2. Write comprehensive integration tests
3. Test edge cases (deep paths, long names, etc.)
4. Test concurrent access patterns
**Checkpoint 9:**
```
$ make test_integration
$ ./test_integration
Running integration tests...
Test: Create nested directory structure
  mkdir /a/b/c/d/e/f: OK
  Verify all directories exist: OK
  Verify link counts: OK
Test: Complex file operations
  Create 100 files: OK
  Create hard links to 50: OK
  Verify link counts: OK
  Delete originals: OK
  Verify links still work: OK
Test: Directory listing
  readdir on root: found all entries: OK
  readdir on nested dir: found entries: OK
Test: Edge cases
  Path with 64 components: OK
  Name with 255 characters: OK
  Empty directory: "." and ".." only: OK
Test: Link count invariants
  mkdir + rmdir: link counts match: OK
  link + unlink: link counts match: OK
All integration tests passed!
```

![rmdir Empty Check](./diagrams/tdd-diag-m3-08.svg)

---
## Test Specification
### test_direntry.c
```c
void test_direntry_size() {
    ASSERT(sizeof(DirEntry) == 280);
    ASSERT(offsetof(DirEntry, inode) == 0);
    ASSERT(offsetof(DirEntry, rec_len) == 8);
    ASSERT(offsetof(DirEntry, name_len) == 10);
    ASSERT(offsetof(DirEntry, file_type) == 11);
    ASSERT(offsetof(DirEntry, name) == 12);
}
void test_direntry_init() {
    DirEntry entry;
    direntry_init(&entry, 42, "testfile.txt", DT_REG);
    ASSERT(entry.inode == 42);
    ASSERT(entry.rec_len == 280);
    ASSERT(entry.name_len == 12);
    ASSERT(entry.file_type == DT_REG);
    ASSERT(strcmp(entry.name, "testfile.txt") == 0);
}
void test_direntry_name_matches() {
    DirEntry entry;
    direntry_init(&entry, 1, "hello", DT_REG);
    ASSERT(direntry_name_matches(&entry, "hello", 5) == true);
    ASSERT(direntry_name_matches(&entry, "hello world", 5) == false);
    ASSERT(direntry_name_matches(&entry, "Hello", 5) == false);
    ASSERT(direntry_name_matches(&entry, "hell", 4) == false);
}
void test_direntry_max_name() {
    char long_name[256];
    memset(long_name, 'a', 255);
    long_name[255] = '\0';
    DirEntry entry;
    direntry_init(&entry, 1, long_name, DT_REG);
    ASSERT(entry.name_len == 255);
    ASSERT(memcmp(entry.name, long_name, 255) == 0);
    ASSERT(entry.name[255] == '\0');
}
```
### test_dir_lookup.c
```c
void test_lookup_empty_directory() {
    FileSystem* fs = create_test_filesystem();
    // Root should have . and ..
    ASSERT(dir_lookup(fs, 1, ".") == 1);
    ASSERT(dir_lookup(fs, 1, "..") == 1);
    ASSERT(dir_lookup(fs, 1, "nonexistent") == 0);
    ASSERT(errno == 0);  // Not found is not an error
}
void test_lookup_after_create() {
    FileSystem* fs = create_test_filesystem();
    // Create a file
    uint64_t file_inode = alloc_inode(fs);
    dir_add_entry(fs, 1, "testfile", file_inode, DT_REG);
    // Should find it
    ASSERT(dir_lookup(fs, 1, "testfile") == file_inode);
    // Should not find others
    ASSERT(dir_lookup(fs, 1, "other") == 0);
}
void test_lookup_not_directory() {
    FileSystem* fs = create_test_filesystem();
    // Create a regular file
    uint64_t file_inode = create_test_file(fs, "/regular");
    // Trying to lookup in a file should fail
    errno = 0;
    ASSERT(dir_lookup(fs, file_inode, "anything") == 0);
    ASSERT(errno == ENOTDIR);
}
```
### test_path.c
```c
void test_parse_absolute_path() {
    ParsedPath parsed;
    ASSERT(path_parse("/home/user/docs", &parsed) == 0);
    ASSERT(parsed.is_absolute == true);
    ASSERT(parsed.count == 3);
    ASSERT(strcmp(parsed.components[0].name, "home") == 0);
    ASSERT(strcmp(parsed.components[1].name, "user") == 0);
    ASSERT(strcmp(parsed.components[2].name, "docs") == 0);
}
void test_parse_relative_path() {
    ParsedPath parsed;
    ASSERT(path_parse("relative/path", &parsed) == 0);
    ASSERT(parsed.is_absolute == false);
    ASSERT(parsed.count == 2);
}
void test_parse_root() {
    ParsedPath parsed;
    ASSERT(path_parse("/", &parsed) == 0);
    ASSERT(parsed.is_absolute == true);
    ASSERT(parsed.count == 0);  // No components
}
void test_parse_special_entries() {
    ParsedPath parsed;
    ASSERT(path_parse(".././..", &parsed) == 0);
    ASSERT(parsed.count == 3);
    ASSERT(strcmp(parsed.components[0].name, "..") == 0);
    ASSERT(strcmp(parsed.components[1].name, ".") == 0);
    ASSERT(strcmp(parsed.components[2].name, "..") == 0);
}
void test_resolve_nested_paths() {
    FileSystem* fs = create_test_filesystem();
    // Create /a/b/c
    fs_mkdir(fs, "/a");
    fs_mkdir(fs, "/a/b");
    uint64_t c_inode = fs_mkdir(fs, "/a/b/c");
    // Resolve various paths
    ASSERT(path_resolve(fs, "/a/b/c") == c_inode);
    ASSERT(path_resolve(fs, "/a/b/c/.") == c_inode);
    ASSERT(path_resolve(fs, "/a/b/c/..") != 0);  // Should be b's inode
    ASSERT(path_resolve(fs, "/a/b/c/../..") != 0);  // Should be a's inode
}
```
### test_mkdir_rmdir.c
```c
void test_mkdir_basic() {
    FileSystem* fs = create_test_filesystem();
    uint64_t dir_inode = fs_mkdir(fs, "/testdir");
    ASSERT(dir_inode > 0);
    // Verify directory exists
    ASSERT(path_resolve(fs, "/testdir") == dir_inode);
    // Verify . and ..
    ASSERT(dir_lookup(fs, dir_inode, ".") == dir_inode);
    ASSERT(dir_lookup(fs, dir_inode, "..") == 1);  // Root
    // Verify link counts
    Inode dir;
    read_inode(fs, dir_inode, &dir);
    ASSERT(dir.link_count == 2);  // Parent entry + "."
    Inode root;
    read_inode(fs, 1, &root);
    ASSERT(root.link_count == 3);  // Original 2 + child's ".."
}
void test_rmdir_non_empty() {
    FileSystem* fs = create_test_filesystem();
    fs_mkdir(fs, "/parent");
    fs_mkdir(fs, "/parent/child");
    // Can't remove non-empty directory
    errno = 0;
    ASSERT(fs_rmdir(fs, "/parent") == -1);
    ASSERT(errno == ENOTEMPTY);
    // Remove child first
    ASSERT(fs_rmdir(fs, "/parent/child") == 0);
    // Now parent can be removed
    ASSERT(fs_rmdir(fs, "/parent") == 0);
}
void test_link_counts_after_rmdir() {
    FileSystem* fs = create_test_filesystem();
    Inode root_before;
    read_inode(fs, 1, &root_before);
    uint64_t initial_link_count = root_before.link_count;
    fs_mkdir(fs, "/testdir");
    fs_rmdir(fs, "/testdir");
    Inode root_after;
    read_inode(fs, 1, &root_after);
    // Link count should return to original
    ASSERT(root_after.link_count == initial_link_count);
}
```
### test_link.c
```c
void test_hard_link_basic() {
    FileSystem* fs = create_test_filesystem();
    // Create original file
    uint64_t orig_inode = create_test_file(fs, "/original");
    write_test_data(fs, orig_inode, "Hello");
    // Create hard link
    ASSERT(fs_link(fs, "/original", "/link") == 0);
    // Verify same inode
    ASSERT(path_resolve(fs, "/link") == orig_inode);
    // Verify link count
    Inode inode;
    read_inode(fs, orig_inode, &inode);
    ASSERT(inode.link_count == 2);
    // Verify same content
    char buf[10];
    read_test_data(fs, orig_inode, buf, 5);
    ASSERT(memcmp(buf, "Hello", 5) == 0);
}
void test_hard_link_unlink() {
    FileSystem* fs = create_test_filesystem();
    uint64_t inode = create_test_file(fs, "/file");
    fs_link(fs, "/file", "/link");
    // Unlink original
    dir_remove_entry(fs, 1, "file");
    // Verify link still works
    Inode inode_data;
    read_inode(fs, inode, &inode_data);
    ASSERT(inode_data.link_count == 1);
    ASSERT(path_resolve(fs, "/link") == inode);
    // Unlink last reference
    dir_remove_entry(fs, 1, "link");
    // Inode should be freed
    ASSERT(path_resolve(fs, "/link") == 0);
}
void test_symlink_basic() {
    FileSystem* fs = create_test_filesystem();
    ASSERT(fs_symlink(fs, "/target/file", "/link") == 0);
    // Verify symlink exists
    uint64_t link_inode = path_resolve(fs, "/link");
    ASSERT(link_inode > 0);
    // Verify it's a symlink
    Inode inode;
    read_inode(fs, link_inode, &inode);
    ASSERT((inode.mode & S_IFLNK) != 0);
    // Read target
    char buf[256];
    int len = fs_readlink(fs, "/link", buf, sizeof(buf));
    ASSERT(len == strlen("/target/file"));
    ASSERT(strcmp(buf, "/target/file") == 0);
}
```
### test_rename.c
```c
void test_rename_same_directory() {
    FileSystem* fs = create_test_filesystem();
    uint64_t inode = create_test_file(fs, "/oldname");
    ASSERT(fs_rename(fs, "/oldname", "/newname") == 0);
    // Old name gone
    ASSERT(path_resolve(fs, "/oldname") == 0);
    // New name exists with same inode
    ASSERT(path_resolve(fs, "/newname") == inode);
}
void test_rename_cross_directory() {
    FileSystem* fs = create_test_filesystem();
    fs_mkdir(fs, "/dir1");
    fs_mkdir(fs, "/dir2");
    uint64_t inode = create_test_file(fs, "/dir1/file");
    ASSERT(fs_rename(fs, "/dir1/file", "/dir2/file") == 0);
    // Old path gone
    ASSERT(path_resolve(fs, "/dir1/file") == 0);
    // New path exists
    ASSERT(path_resolve(fs, "/dir2/file") == inode);
}
void test_rename_directory() {
    FileSystem* fs = create_test_filesystem();
    fs_mkdir(fs, "/olddir");
    fs_mkdir(fs, "/olddir/subdir");
    ASSERT(fs_rename(fs, "/olddir", "/newdir") == 0);
    // Verify new path
    ASSERT(path_resolve(fs, "/newdir") != 0);
    ASSERT(path_resolve(fs, "/newdir/subdir") != 0);
    // Verify .. updated
    uint64_t subdir = path_resolve(fs, "/newdir/subdir");
    ASSERT(dir_lookup(fs, subdir, "..") == path_resolve(fs, "/newdir"));
}
```
---
## Performance Targets
| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| `dir_lookup` (10 entries) | < 100 μs | Microbenchmark |
| `dir_lookup` (100 entries) | < 1 ms | Microbenchmark |
| `dir_lookup` (1000 entries) | < 10 ms | Microbenchmark |
| `dir_add_entry` | < 500 μs | Includes scan + write |
| `dir_remove_entry` | < 500 μs | Includes scan + write |
| `path_resolve` (5 components) | < 2 ms | Typical case |
| `fs_mkdir` | < 1 ms | Includes allocation |
| `fs_rmdir` | < 1 ms | Empty directory |
| `fs_link` | < 500 μs | Hard link creation |
| `fs_symlink` | < 1 ms | Symlink creation |
| `fs_rename` (same dir) | < 500 μs | Simple rename |
| `fs_rename` (cross dir) | < 2 ms | Includes link count updates |
**Hardware Soul - Performance Analysis:**
```
dir_lookup (100 entries, 8 blocks):
  Block reads: 8 (sequential scan)
  Entry comparisons: ~50 average (uniform distribution)
  Cache lines: 512 (8 blocks × 64)
  NVMe latency: 8 × 25μs = 200μs + comparison overhead
  HDD latency: 8 × 10ms = 80ms (worst case, random seeks)
Optimization: Hash-based directory indexing (ext4 htree)
  - Build hash table on first access
  - Subsequent lookups: O(1) average
  - Trade-off: Memory usage, complexity
path_resolve (5 components, 10 entries per directory):
  dir_lookup calls: 5
  Block reads: ~40 (8 per directory, assuming no caching)
  Total latency: ~1ms (NVMe) or ~400ms (HDD)
With dentry cache (future optimization):
  - Cache resolved paths
  - Subsequent resolves: O(1) hash lookup
  - Cache hit: ~100ns
```

![Hard Link vs Symbolic Link](./diagrams/tdd-diag-m3-09.svg)

---
## Diagrams
{{DIAGRAM:tdd-diag-m3-10}}

![Directory Entry Add Flow](./diagrams/tdd-diag-m3-11.svg)


![Path Component Split](./diagrams/tdd-diag-m3-12.svg)

---
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: filesystem-m4 -->
# Technical Design Specification: File Read/Write Operations
## Module Charter
This module implements the core file I/O operations that translate between logical byte streams and physical block storage. It provides file creation (`fs_create`), reading with hole detection (`fs_read`), writing with lazy block allocation (`fs_write`), and truncation with block management (`fs_truncate`). The module bridges the user's view of files as continuous byte sequences with the filesystem's reality of 4KB block-aligned storage, handling block boundary crossing, sparse file holes (unallocated regions that read as zeros), and metadata updates (size, mtime, ctime). It depends on the inode block pointer resolution from M2 (`get_file_block`, `get_or_alloc_block`), directory entry management from M3 (`dir_add_entry`, `path_resolve`), and bitmap allocation from M1. It does NOT implement FUSE callbacks (M5), journaling for crash consistency (M6), or interpret directory entry formats (directories are treated as opaque special files). The core invariants: every byte read from an allocated block returns the exact data written; every byte read from a hole returns zero; the inode's size field always reflects the logical file size; the inode's blocks field always reflects the number of 512-byte units allocated; no block pointer in an inode points to a free block in the bitmap.
---
## File Structure
```
filesystem/
├── 38_file_create.h     # fs_create declaration
├── 39_file_create.c     # File creation implementation
├── 40_file_read.h       # fs_read declaration, FileReadResult struct
├── 41_file_read.c       # File read with hole detection
├── 42_file_write.h      # fs_write declaration, FileWriteResult struct
├── 43_file_write.c      # File write with block allocation
├── 44_file_truncate.h   # fs_truncate declaration
├── 45_file_truncate.c   # Truncate grow/shrink implementation
├── 46_file_append.h     # fs_append declaration
├── 47_file_append.c     # Append operation
├── 48_file_utils.h      # Internal helpers: block_boundary_crosses, etc.
├── 49_file_utils.c      # Utility implementations
└── tests/
    ├── test_file_create.c    # Creation tests
    ├── test_file_read.c      # Read operation tests
    ├── test_file_write.c     # Write operation tests
    ├── test_file_truncate.c  # Truncation tests
    ├── test_sparse.c         # Sparse file behavior tests
    └── test_file_boundary.c  # Block boundary crossing tests
```
---
## Complete Data Model
### FileReadResult Structure
```c
// 40_file_read.h
#include <stdint.h>
#include <stddef.h>
/**
 * Result of a file read operation.
 */
typedef struct {
    ssize_t bytes_read;     // Number of bytes actually read
                            // -1 on error
    int error_code;         // 0 on success, errno value on failure
    bool hit_hole;          // True if any hole was encountered during read
    uint64_t final_offset;  // File offset after read (for future pread use)
} FileReadResult;
```
### FileWriteResult Structure
```c
// 42_file_write.h
#include <stdint.h>
#include <stddef.h>
/**
 * Result of a file write operation.
 */
typedef struct {
    ssize_t bytes_written;      // Number of bytes actually written
                                // -1 on error, may be short on ENOSPC
    int error_code;             // 0 on success, errno value on failure
    uint64_t blocks_allocated;  // Number of new blocks allocated
    uint64_t old_size;          // File size before write
    uint64_t new_size;          // File size after write
    bool extended_file;         // True if write extended beyond old size
} FileWriteResult;
```
### BlockBoundaryInfo Structure
```c
// 48_file_utils.h
/**
 * Information about a block boundary crossing operation.
 * Used for read/write loop calculations.
 */
typedef struct {
    uint64_t file_block_idx;    // Logical block index within file
    uint64_t block_offset;      // Byte offset within the block (0-4095)
    size_t bytes_in_block;      // Bytes available from offset to block end
    bool crosses_boundary;      // True if operation spans multiple blocks
    uint64_t next_block_idx;    // Next block index if crosses_boundary
} BlockBoundaryInfo;
```
### ReadContext Structure (Internal)
```c
// 41_file_read.c - internal
/**
 * Context for a multi-block read operation.
 */
typedef struct {
    const FileSystem* fs;
    const Inode* inode;
    uint64_t current_offset;    // Current file byte offset
    size_t remaining;           // Bytes remaining to read
    char* output_ptr;           // Current position in output buffer
    char block_buffer[BLOCK_SIZE];
    char indirect_buffer[BLOCK_SIZE];
} ReadContext;
```
### WriteContext Structure (Internal)
```c
// 43_file_write.c - internal
/**
 * Context for a multi-block write operation.
 */
typedef struct {
    FileSystem* fs;
    uint64_t inode_num;
    Inode* inode;               // Modified during write
    uint64_t current_offset;    // Current file byte offset
    size_t remaining;           // Bytes remaining to write
    const char* input_ptr;      // Current position in input buffer
    char block_buffer[BLOCK_SIZE];
    char indirect_buffer[BLOCK_SIZE];
    char bitmap_buffer[BLOCK_SIZE];
    uint64_t blocks_allocated;  // Count of newly allocated blocks
} WriteContext;
```
**Byte Offset Analysis for Block Calculations:**
```
Block boundary calculation (offset 5000, length 8000):
  Block 1 (offset 5000-8191):
    block_idx = 5000 / 4096 = 1
    block_offset = 5000 % 4096 = 904
    bytes_in_block = 4096 - 904 = 3192
  Block 2 (offset 8192-12287):
    block_idx = 8192 / 4096 = 2
    block_offset = 0
    bytes_in_block = min(8000 - 3192, 4096) = 4808
  Block 3 (offset 12288-13191):
    block_idx = 12288 / 4096 = 3
    block_offset = 0
    bytes_in_block = 8000 - 3192 - 4808 = 0 (done)
Total: 2 block reads for 8000 bytes starting at offset 5000
```
**Hardware Soul - Cache Line Analysis:**
```
Block buffer (4096 bytes):
  - 64 cache lines of 64 bytes each
  - Sequential access pattern during read/write
  - Prefetcher-friendly: linear scan through buffer
Read-modify-write for partial blocks:
  1. Read block (64 cache line loads)
  2. Modify subset (1-64 cache lines touched)
  3. Write block (64 cache line stores)
  Write amplification: 2× for partial block writes
Full-block write optimization:
  - Skip read phase entirely
  - Direct memcpy from input to block
  - 1× I/O, no write amplification
```

![Read Loop Block Boundary](./diagrams/tdd-diag-m4-01.svg)

---
## Interface Contracts
### File Creation
```c
// 38_file_create.h
#include <stdint.h>
#include "14_inode.h"
/**
 * Create a new regular file.
 *
 * @param fs    FileSystem structure
 * @param path  Full path for the new file
 * @param mode  Permission bits (e.g., 0644), file type will be set to S_IFREG
 * @return New inode number on success, 0 on failure
 *
 * Errors (in errno):
 *   ENOENT  - Parent directory doesn't exist
 *   ENOTDIR - Parent path component is not a directory
 *   EEXIST  - File already exists at path
 *   ENOSPC  - No free inodes
 *   EIO     - Read/write error
 *
 * Postcondition:
 *   - New inode allocated with mode = S_IFREG | (mode & 07777)
 *   - Inode has link_count = 1, size = 0, blocks = 0
 *   - All block pointers are 0 (file is empty)
 *   - Directory entry added in parent
 *   - Parent's mtime and ctime updated
 *   - Timestamps on new file set to current time
 *
 * Note: This creates an EMPTY file. No blocks are allocated
 * until the first write occurs.
 */
uint64_t fs_create(FileSystem* fs, const char* path, uint16_t mode);
```
### File Read
```c
// 40_file_read.h
#include <stdint.h>
#include <stddef.h>
/**
 * Read data from a file at a given offset.
 *
 * @param fs        FileSystem structure
 * @param inode_num Inode number of file to read
 * @param offset    Byte offset within file to start reading
 * @param buffer    Output buffer for read data
 * @param length    Maximum number of bytes to read
 * @return Number of bytes read on success, -1 on failure
 *
 * Errors (in errno):
 *   EISDIR  - inode_num refers to a directory
 *   EINVAL  - offset > SSIZE_MAX (theoretical)
 *   EIO     - Read error from block device
 *
 * Edge cases:
 *   - offset >= file size: Returns 0 bytes (not an error)
 *   - offset + length > file size: Reads to EOF, returns actual bytes
 *   - Reading from hole: Returns zeros without disk I/O
 *   - length = 0: Returns 0 immediately, no side effects
 *
 * Side effects:
 *   - Updates inode->atime to current time
 *   - Writes inode back to disk
 *
 * Complexity:
 *   - O(number of blocks touched)
 *   - Hole reads: O(length / BLOCK_SIZE) memory operations, 0 I/O
 *   - Allocated reads: O(number of blocks) I/O operations
 *
 * Thread safety: Caller must hold lock on inode for consistent results
 */
ssize_t fs_read(FileSystem* fs, uint64_t inode_num, uint64_t offset,
                void* buffer, size_t length);
/**
 * Extended read returning additional metadata.
 *
 * @param fs        FileSystem structure
 * @param inode_num Inode number
 * @param offset    Byte offset
 * @param buffer    Output buffer
 * @param length    Maximum bytes to read
 * @param result    Output: detailed result structure
 * @return Same as fs_read, with result->error_code set
 */
ssize_t fs_read_ex(FileSystem* fs, uint64_t inode_num, uint64_t offset,
                   void* buffer, size_t length, FileReadResult* result);
```
### File Write
```c
// 42_file_write.h
#include <stdint.h>
#include <stddef.h>
/**
 * Write data to a file at a given offset.
 *
 * @param fs        FileSystem structure
 * @param inode_num Inode number of file to write
 * @param offset    Byte offset within file to start writing
 * @param data      Data to write
 * @param length    Number of bytes to write
 * @return Number of bytes written on success, -1 on failure
 *
 * Errors (in errno):
 *   EISDIR  - inode_num refers to a directory
 *   EFBIG   - Write would exceed maximum file size (~1GB for this filesystem)
 *   ENOSPC  - No free blocks for allocation (partial write may have occurred)
 *   EIO     - Write error from block device
 *
 * Edge cases:
 *   - offset > file size: Creates hole between old EOF and offset
 *   - Writing to hole: Allocates block, writes data
 *   - Partial block write: Read-modify-write for non-aligned writes
 *   - length = 0: Returns 0 immediately, no side effects
 *   - ENOSPC mid-write: Returns bytes written so far (short write)
 *
 * Side effects:
 *   - May allocate data blocks (lazy allocation)
 *   - May allocate indirect blocks if entering new zone
 *   - Updates inode->size if offset + length > old size
 *   - Updates inode->mtime and inode->ctime
 *   - Updates inode->blocks count
 *   - Writes inode back to disk
 *   - Updates superblock free_blocks count
 *
 * Allocation cascade for first write at 5MB:
 *   1. Allocate double-indirect block
 *   2. Allocate indirect block
 *   3. Allocate data block
 *   Total: 3 blocks for 1 byte of data
 *
 * Thread safety: Caller must hold exclusive lock on inode
 */
ssize_t fs_write(FileSystem* fs, uint64_t inode_num, uint64_t offset,
                 const void* data, size_t length);
/**
 * Extended write returning allocation details.
 *
 * @param fs        FileSystem structure
 * @param inode_num Inode number
 * @param offset    Byte offset
 * @param data      Data to write
 * @param length    Number of bytes
 * @param result    Output: detailed result structure
 * @return Same as fs_write, with result populated
 */
ssize_t fs_write_ex(FileSystem* fs, uint64_t inode_num, uint64_t offset,
                    const void* data, size_t length, FileWriteResult* result);
```
### File Truncate
```c
// 44_file_truncate.h
#include <stdint.h>
/**
 * Change the size of a file.
 *
 * @param fs        FileSystem structure
 * @param inode_num Inode number of file to truncate
 * @param new_size  New file size in bytes
 * @return 0 on success, -1 on failure
 *
 * Errors (in errno):
 *   EISDIR    - inode_num refers to a directory
 *   EFBIG     - new_size exceeds maximum file size
 *   EIO       - Write error during block freeing
 *
 * Truncate to larger size (new_size > current size):
 *   - Updates inode->size to new_size
 *   - New region becomes a hole (reads as zeros)
 *   - No blocks allocated
 *   - Updates mtime and ctime
 *
 * Truncate to smaller size (new_size < current size):
 *   - Frees all blocks beyond new_size
 *   - Updates block pointers to 0 for freed blocks
 *   - Frees indirect blocks if entire zone is freed
 *   - Updates inode->blocks count
 *   - Partial block at new end: zero-fill beyond new_size
 *   - Updates mtime and ctime
 *
 * Truncate to same size (new_size == current size):
 *   - No changes (early return with success)
 *
 * Postcondition:
 *   - inode->size == new_size
 *   - All blocks beyond new_size are freed
 *   - Partial block at new_size is zero-filled from new_size to block end
 *   - superblock->free_blocks updated
 */
int fs_truncate(FileSystem* fs, uint64_t inode_num, uint64_t new_size);
```
### File Append
```c
// 46_file_append.h
#include <stdint.h>
#include <stddef.h>
/**
 * Append data to the end of a file.
 *
 * @param fs        FileSystem structure
 * @param inode_num Inode number of file
 * @param data      Data to append
 * @param length    Number of bytes to append
 * @return Number of bytes written, or -1 on failure
 *
 * Semantics: Equivalent to fs_write(fs, inode_num, inode.size, data, length)
 *
 * Note: Not atomic with respect to concurrent writers.
 * For concurrent appends, use O_APPEND semantics at FUSE level (M5).
 */
ssize_t fs_append(FileSystem* fs, uint64_t inode_num,
                  const void* data, size_t length);
```
### Utility Functions
```c
// 48_file_utils.h
#include <stdint.h>
#include <stddef.h>
/**
 * Calculate block boundary information for an operation.
 *
 * @param file_offset  Current byte offset within file
 * @param remaining    Bytes remaining in operation
 * @return BlockBoundaryInfo with block indices and counts
 */
BlockBoundaryInfo calculate_block_boundary(uint64_t file_offset, size_t remaining);
/**
 * Check if a block is a hole (unallocated).
 *
 * @param fs            FileSystem structure
 * @param inode         Inode to check
 * @param block_idx     Logical block index within file
 * @param indirect_buf  Temporary buffer for indirect block reads
 * @return true if block is a hole, false if allocated or error
 */
bool is_block_hole(FileSystem* fs, const Inode* inode, uint64_t block_idx,
                   void* indirect_buf);
/**
 * Zero-fill a portion of a block.
 *
 * @param fs            FileSystem structure
 * @param block_num     Physical block number
 * @param start_offset  Byte offset within block to start zeroing
 * @param length        Number of bytes to zero
 * @param block_buf     Temporary buffer (BLOCK_SIZE bytes)
 * @return 0 on success, -1 on failure
 *
 * Used by truncate to zero-fill partial blocks.
 */
int zero_block_range(FileSystem* fs, uint64_t block_num,
                     uint64_t start_offset, size_t length, void* block_buf);
/**
 * Count actual allocated blocks in an inode.
 *
 * @param fs      FileSystem structure
 * @param inode   Inode to count
 * @param ind_buf Temporary buffer
 * @return Number of allocated blocks (not 512-byte units)
 */
uint64_t count_allocated_blocks(FileSystem* fs, const Inode* inode, void* ind_buf);
```
---
## Algorithm Specification
### fs_create Algorithm
**Input:** `fs`, `path`, `mode`
**Output:** New inode number, or 0 on failure
**Postcondition:** Empty file exists with directory entry
```
ALGORITHM fs_create(fs, path, mode):
  // Extract parent directory and filename
  name ← allocate(MAX_NAME_LEN + 1)
  parent_inode_num ← path_resolve_parent(fs, path, name)
  IF parent_inode_num = 0:
    free(name)
    RETURN 0  // errno set by path_resolve_parent
  // Check if name already exists
  IF dir_lookup(fs, parent_inode_num, name) ≠ 0:
    errno ← EEXIST
    free(name)
    RETURN 0
  // Verify parent is a directory
  parent_inode ← read_inode(fs, parent_inode_num)
  IF parent_inode = NULL OR (parent_inode.mode & S_IFDIR) = 0:
    errno ← ENOTDIR
    free(name)
    RETURN 0
  // Allocate new inode
  new_inode_num ← inode_bitmap_alloc(fs->dev, fs->sb, bitmap_buf)
  IF new_inode_num = 0:
    errno ← ENOSPC
    free(name)
    RETURN 0
  // Initialize inode as empty regular file
  new_inode ← allocate(Inode)
  inode_init(new_inode, S_IFREG | (mode & 07777), 0, 0)
  // inode_init sets:
  //   - mode, uid, gid as specified
  //   - link_count = 1
  //   - size = 0
  //   - blocks = 0
  //   - all block pointers = 0
  //   - timestamps to current time
  write_inode(fs, new_inode_num, new_inode)
  // Add directory entry
  IF dir_add_entry(fs, parent_inode_num, name, new_inode_num, DT_REG) < 0:
    // Rollback: free the inode
    inode_bitmap_free(fs->dev, fs->sb, new_inode_num, bitmap_buf)
    free(name)
    free(new_inode)
    RETURN -1
  // Update parent timestamps
  parent_inode.mtime ← current_time()
  parent_inode.ctime ← parent_inode.mtime
  write_inode(fs, parent_inode_num, parent_inode)
  // Update superblock
  fs->sb->free_inodes ← fs->sb->free_inodes - 1
  write_block(fs->dev, 0, fs->sb)
  result ← new_inode_num
  free(name)
  free(new_inode)
  RETURN result
```
**Complexity:** O(directory scan) for duplicate check + O(1) for inode allocation

![Write Read-Modify-Write](./diagrams/tdd-diag-m4-02.svg)

### fs_read Algorithm
**Input:** `fs`, `inode_num`, `offset`, `buffer`, `length`
**Output:** Bytes read, or -1 on error
**Invariant:** File is not modified (except atime)
```
ALGORITHM fs_read(fs, inode_num, offset, buffer, length):
  // Validate and read inode
  inode ← read_inode(fs, inode_num)
  IF inode = NULL:
    RETURN -1
  // Check file type
  IF (inode.mode & S_IFDIR) ≠ 0:
    errno ← EISDIR
    RETURN -1
  // Handle zero-length read
  IF length = 0:
    RETURN 0
  // Handle read past EOF
  IF offset ≥ inode.size:
    RETURN 0  // 0 bytes read, not an error
  // Clamp length to available data
  IF offset + length > inode.size:
    length ← inode.size - offset
  // Initialize read context
  ctx ← ReadContext{
    fs: fs,
    inode: inode,
    current_offset: offset,
    remaining: length,
    output_ptr: buffer
  }
  bytes_read ← 0
  // Main read loop - process one block at a time
  WHILE ctx.remaining > 0:
    // Calculate block information
    block_idx ← ctx.current_offset / BLOCK_SIZE
    block_offset ← ctx.current_offset % BLOCK_SIZE
    bytes_in_block ← BLOCK_SIZE - block_offset
    // Determine how many bytes to read from this block
    to_read ← MIN(bytes_in_block, ctx.remaining)
    // Get physical block number (may be 0 for hole)
    phys_block ← get_file_block(fs->dev, inode, block_idx, ctx.indirect_buffer)
    IF phys_block = 0:
      // Hole: synthesize zeros without disk I/O
      memset(ctx.output_ptr, 0, to_read)
    ELSE:
      // Read the actual block
      IF read_block(fs->dev, phys_block, ctx.block_buffer) < 0:
        // Partial read - return what we have
        IF bytes_read > 0:
          BREAK  // Return partial success
        errno ← EIO
        RETURN -1
      // Copy relevant portion to output
      memcpy(ctx.output_ptr, ctx.block_buffer + block_offset, to_read)
    // Advance context
    ctx.current_offset ← ctx.current_offset + to_read
    ctx.output_ptr ← ctx.output_ptr + to_read
    ctx.remaining ← ctx.remaining - to_read
    bytes_read ← bytes_read + to_read
  // Update access time
  inode.atime ← current_time()
  write_inode(fs, inode_num, inode)
  RETURN bytes_read
```
**Block Boundary Crossing Example:**
```
Read 8000 bytes starting at offset 5000:
File blocks: [block 0][block 1][block 2][block 3]...
                 |        |        |        |
Offsets:      0-4095  4096-8191 8192-12287 ...
Iteration 1:
  current_offset = 5000
  block_idx = 5000 / 4096 = 1
  block_offset = 5000 % 4096 = 904
  bytes_in_block = 4096 - 904 = 3192
  to_read = MIN(3192, 8000) = 3192
  Read bytes 904-4095 from block 1 → 3192 bytes
  remaining = 8000 - 3192 = 4808
Iteration 2:
  current_offset = 5000 + 3192 = 8192
  block_idx = 8192 / 4096 = 2
  block_offset = 0
  bytes_in_block = 4096
  to_read = MIN(4096, 4808) = 4096
  Read entire block 2 → 4096 bytes
  remaining = 4808 - 4096 = 712
Iteration 3:
  current_offset = 8192 + 4096 = 12288
  block_idx = 12288 / 4096 = 3
  block_offset = 0
  bytes_in_block = 4096
  to_read = MIN(4096, 712) = 712
  Read bytes 0-711 from block 3 → 712 bytes
  remaining = 0
Total: 3192 + 4096 + 712 = 8000 bytes in 3 block reads
```
**Hardware Soul - Read Performance:**
```
Cold cache sequential read of 1MB:
  Blocks: 256
  Block reads: 256 (each 4KB)
  Cache lines: 256 × 64 = 16,384
  NVMe: 256 × 25μs = 6.4ms (~160 MB/s)
  HDD sequential: 256 × 0.1ms = 25.6ms (~40 MB/s with readahead)
  HDD random: 256 × 10ms = 2.56s (~0.4 MB/s)
Sparse file read (1MB logical, 1KB actual at end):
  Hole reads: 255 blocks (memset only)
  Actual reads: 1 block
  I/O: 1 × 25μs = 25μs
  memset: 255 × 4096 = ~1MB of memory writes
  Sparse read is 256× faster on NVMe for this case
```

![Sparse File Allocation](./diagrams/tdd-diag-m4-03.svg)

### fs_write Algorithm
**Input:** `fs`, `inode_num`, `offset`, `data`, `length`
**Output:** Bytes written, or -1 on error
**Postcondition:** Data written, blocks allocated, metadata updated
```
ALGORITHM fs_write(fs, inode_num, offset, data, length):
  // Validate and read inode
  inode ← read_inode(fs, inode_num)
  IF inode = NULL:
    RETURN -1
  // Check file type
  IF (inode.mode & S_IFDIR) ≠ 0:
    errno ← EISDIR
    RETURN -1
  // Handle zero-length write
  IF length = 0:
    RETURN 0
  // Check maximum file size (beyond double-indirect range)
  max_size ← DOUBLE_IND_ZONE_MAX
  IF offset + length > max_size:
    // Clamp to max or fail?
    // For simplicity, fail the entire write
    errno ← EFBIG
    RETURN -1
  // Initialize write context
  ctx ← WriteContext{
    fs: fs,
    inode_num: inode_num,
    inode: inode,
    current_offset: offset,
    remaining: length,
    input_ptr: data,
    blocks_allocated: 0
  }
  old_size ← inode.size
  bytes_written ← 0
  // Main write loop
  WHILE ctx.remaining > 0:
    // Calculate block information
    block_idx ← ctx.current_offset / BLOCK_SIZE
    block_offset ← ctx.current_offset % BLOCK_SIZE
    bytes_in_block ← BLOCK_SIZE - block_offset
    to_write ← MIN(bytes_in_block, ctx.remaining)
    // Determine if this is a full-block or partial write
    is_full_block ← (block_offset = 0 AND to_write = BLOCK_SIZE)
    // Get or allocate the physical block
    phys_block ← get_or_alloc_block(fs, inode_num, inode, block_idx,
                                     ctx.indirect_buffer, ctx.bitmap_buffer)
    IF phys_block = 0:
      // Allocation failed - stop here
      // errno is ENOSPC or EIO from get_or_alloc_block
      BREAK
    // Track if this was a new allocation
    // (get_or_alloc_block modifies inode, we can check if blocks increased)
    IF is_full_block:
      // Optimization: Full block write, no read needed
      memcpy(ctx.block_buffer, ctx.input_ptr, BLOCK_SIZE)
    ELSE:
      // Partial write: Read-modify-write
      IF read_block(fs->dev, phys_block, ctx.block_buffer) < 0:
        // Can't read existing data - this is an error
        IF bytes_written > 0:
          BREAK  // Return partial success
        errno ← EIO
        RETURN -1
      // Modify the relevant portion
      memcpy(ctx.block_buffer + block_offset, ctx.input_ptr, to_write)
    // Write the block
    IF write_block(fs->dev, phys_block, ctx.block_buffer) < 0:
      IF bytes_written > 0:
        BREAK  // Return partial success
      errno ← EIO
      RETURN -1
    // Advance context
    ctx.current_offset ← ctx.current_offset + to_write
    ctx.input_ptr ← ctx.input_ptr + to_write
    ctx.remaining ← ctx.remaining - to_write
    bytes_written ← bytes_written + to_write
  // Update inode metadata if any bytes were written
  IF bytes_written > 0:
    // Extend file size if we wrote past the end
    IF offset + bytes_written > inode.size:
      inode.size ← offset + bytes_written
    // Update modification and change times
    inode.mtime ← current_time()
    inode.ctime ← inode.mtime
    // Write inode back to disk
    write_inode(fs, inode_num, inode)
    // Update superblock if blocks were allocated
    // (The free_blocks count is updated by block_bitmap_alloc internally)
    write_block(fs->dev, 0, fs->sb)
  RETURN bytes_written
```
**Write Amplification Analysis:**
```
Partial block write (100 bytes at offset 4050):
  1. Read block containing bytes 4096-8191: 1 I/O
  2. Modify bytes 4050-4149 in buffer: 0 I/O
  3. Write entire block back: 1 I/O
  Total: 2 I/Os for 100 bytes
  Write amplification: 2× (read + write)
Full block write (4096 bytes at offset 8192):
  1. Allocate block (if needed): bitmap operations
  2. Write entire block directly: 1 I/O
  Total: 1 I/O for 4096 bytes
  Write amplification: 1× (write only)
Strategies to minimize write amplification:
  - Buffer small writes and flush as full blocks
  - Use writeback caching (FUSE level)
  - Align writes to block boundaries when possible
```

![Hole Detection Flow](./diagrams/tdd-diag-m4-04.svg)

### fs_truncate Algorithm
**Input:** `fs`, `inode_num`, `new_size`
**Output:** 0 on success, -1 on failure
**Postcondition:** File is exactly new_size bytes
```
ALGORITHM fs_truncate(fs, inode_num, new_size):
  // Read inode
  inode ← read_inode(fs, inode_num)
  IF inode = NULL:
    RETURN -1
  // Check file type
  IF (inode.mode & S_IFDIR) ≠ 0:
    errno ← EISDIR
    RETURN -1
  // Check maximum size
  IF new_size > DOUBLE_IND_ZONE_MAX:
    errno ← EFBIG
    RETURN -1
  // No-op case
  IF new_size = inode.size:
    RETURN 0
  old_size ← inode.size
  IF new_size > old_size:
    // Growing: just update size
    // New region becomes a hole (implicit zeros)
    inode.size ← new_size
    inode.mtime ← current_time()
    inode.ctime ← inode.mtime
    write_inode(fs, inode_num, inode)
    RETURN 0
  // Shrinking: need to free blocks
  old_blocks ← (old_size + BLOCK_SIZE - 1) / BLOCK_SIZE
  new_blocks ← (new_size + BLOCK_SIZE - 1) / BLOCK_SIZE
  indirect_buffer ← allocate(BLOCK_SIZE)
  bitmap_buffer ← allocate(BLOCK_SIZE)
  // Free blocks from new_blocks to old_blocks - 1
  FOR block_idx FROM new_blocks TO old_blocks - 1:
    phys_block ← get_file_block(fs, inode, block_idx, indirect_buffer)
    IF phys_block ≠ 0:
      // Free the block
      block_bitmap_free(fs->dev, fs->sb, phys_block, bitmap_buffer)
      // Clear the pointer in inode
      clear_block_pointer(fs, inode, block_idx, indirect_buffer)
      // Update block count
      inode.blocks ← inode.blocks - 8  // 8 × 512 = 4096
  // Handle partial block at new end
  IF new_size % BLOCK_SIZE ≠ 0 AND new_blocks > 0:
    // Zero bytes from new_size to end of last block
    last_block_idx ← new_blocks - 1
    phys_block ← get_file_block(fs, inode, last_block_idx, indirect_buffer)
    IF phys_block ≠ 0:
      zero_start ← new_size % BLOCK_SIZE
      zero_length ← BLOCK_SIZE - zero_start
      zero_block_range(fs, phys_block, zero_start, zero_length, indirect_buffer)
  // Check if we should free indirect blocks
  // If new_size < direct zone end, free indirect and double-indirect
  IF new_blocks ≤ DIRECT_BLOCKS AND inode.indirect ≠ 0:
    // Free all blocks in indirect block first
    free_indirect_block_contents(fs, inode.indirect, indirect_buffer, bitmap_buffer)
    block_bitmap_free(fs->dev, fs->sb, inode.indirect, bitmap_buffer)
    inode.indirect ← 0
    inode.blocks ← inode.blocks - 8
  IF new_blocks ≤ DIRECT_BLOCKS AND inode.double_ind ≠ 0:
    // Free entire double-indirect structure
    free_double_indirect_structure(fs, inode.double_ind, 
                                    indirect_buffer, bitmap_buffer)
    block_bitmap_free(fs->dev, fs->sb, inode.double_ind, bitmap_buffer)
    inode.double_ind ← 0
    inode.blocks ← inode.blocks - 8
  // Update inode
  inode.size ← new_size
  inode.mtime ← current_time()
  inode.ctime ← inode.mtime
  write_inode(fs, inode_num, inode)
  // Update superblock
  write_block(fs->dev, 0, fs->sb)
  free(indirect_buffer)
  free(bitmap_buffer)
  RETURN 0
// Helper: Clear a block pointer in the inode tree
ALGORITHM clear_block_pointer(fs, inode, block_idx, indirect_buffer):
  IF block_idx < DIRECT_BLOCKS:
    inode.direct[block_idx] ← 0
    RETURN
  block_idx ← block_idx - DIRECT_BLOCKS
  IF block_idx < PTRS_PER_BLOCK:
    IF inode.indirect = 0:
      RETURN
    read_block(fs->dev, inode.indirect, indirect_buffer)
    ptrs ← cast(indirect_buffer to uint64_t*)
    ptrs[block_idx] ← 0
    write_block(fs->dev, inode.indirect, indirect_buffer)
    RETURN
  block_idx ← block_idx - PTRS_PER_BLOCK
  // Double-indirect case
  IF inode.double_ind = 0:
    RETURN
  double_idx ← block_idx / PTRS_PER_BLOCK
  indirect_idx ← block_idx % PTRS_PER_BLOCK
  double_buffer ← allocate(BLOCK_SIZE)
  read_block(fs->dev, inode.double_ind, double_buffer)
  double_ptrs ← cast(double_buffer to uint64_t*)
  indirect_block ← double_ptrs[double_idx]
  IF indirect_block ≠ 0:
    read_block(fs->dev, indirect_block, indirect_buffer)
    ptrs ← cast(indirect_buffer to uint64_t*)
    ptrs[indirect_idx] ← 0
    write_block(fs->dev, indirect_block, indirect_buffer)
  free(double_buffer)
```
**Truncate Examples:**
```
Truncate 100KB file to 6KB:
  old_blocks = ceil(102400 / 4096) = 25
  new_blocks = ceil(6144 / 4096) = 2
  Free blocks 2-24 (23 blocks)
  Zero-fill bytes 6144-8191 in block 1
Truncate 100KB file to 0:
  old_blocks = 25
  new_blocks = 0
  Free all blocks (0-24)
  Clear all block pointers
  Free indirect blocks if allocated
Truncate 0 byte file to 1MB:
  No blocks to free
  Update size to 1048576
  No blocks allocated (hole from 0 to 1MB)
```

![Truncate Shrink Block Free](./diagrams/tdd-diag-m4-05.svg)

### Sparse File Write Algorithm
**Input:** `fs`, `inode_num`, `offset`, `data`, `length` where offset >> current size
**Output:** Bytes written with minimal allocation
```
Example: Write 100 bytes at offset 1MB to empty file
Initial state:
  inode.size = 0
  inode.blocks = 0
  All pointers = 0
After write:
  inode.size = 1048576 + 100 = 1048676
  inode.blocks = 24 (3 blocks × 8)
Block allocation cascade:
  1. get_or_alloc_block for block_idx = 256 (1MB / 4KB)
  2. Block 256 is in double-indirect zone:
     - double_idx = (256 - 12 - 512) / 512 = -268 / 512 = 0
     - Wait, recalculate:
     - Block 256: 0-11 direct, 12-523 single-ind, 524+ double-ind
     - 256 is in single-indirect zone
     - indirect_idx = 256 - 12 = 244
  3. Allocate indirect block (1 block)
  4. Allocate data block (1 block)
  5. Write 100 bytes to data block
Blocks allocated: 2 (indirect + data)
Logical file size: 1MB + 100 bytes
Physical storage: 8KB
Read from offset 0:
  Block 0: hole (returns zeros)
  Block 1: hole (returns zeros)
  ...
  Block 255: hole (returns zeros)
  Block 256: allocated (returns actual data)
```

![File Size vs Allocated Blocks](./diagrams/tdd-diag-m4-06.svg)

---
## Error Handling Matrix
| Error | errno | Detected By | Recovery | User-Visible Message |
|-------|-------|-------------|----------|---------------------|
| File is directory | EISDIR | fs_read, fs_write, fs_truncate | Return -1 immediately, no modifications | "Is a directory" |
| File too large | EFBIG | fs_write, fs_truncate | Return -1, no modifications | "File too large" |
| No space on device | ENOSPC | fs_write (in get_or_alloc_block) | Return bytes written so far (short write) | "No space left on device" |
| I/O error on read | EIO | read_block in fs_read | Return bytes read so far or -1 if nothing read | "Input/output error" |
| I/O error on write | EIO | write_block in fs_write | Return bytes written so far, may have partial data | "Input/output error" |
| Parent not found | ENOENT | fs_create (path_resolve_parent) | Return 0, no inode allocated | "No such file or directory" |
| Parent not directory | ENOTDIR | fs_create | Return 0, no inode allocated | "Not a directory" |
| File exists | EEXIST | fs_create (dir_lookup) | Return 0, no inode allocated | "File exists" |
| Invalid inode | EINVAL | All functions (bitmap check) | Return -1 | Internal error |
| Read past EOF | (none) | fs_read (offset >= size) | Return 0 bytes, not an error | (Silent - normal case) |
| Write creates hole | (none) | fs_write (offset > size) | Normal operation, not an error | (Not visible) |
**Partial Write Handling:**
```
Scenario: Write 10KB at offset 0, only 4KB of space available
Execution:
  1. Write first 4KB block: success
  2. Try to allocate second block: ENOSPC
  3. Return bytes_written = 4096
Application must check:
  if (result < length) {
      // Short write - handle appropriately
      if (errno == ENOSPC) {
          // Clean up or report error
      }
  }
```
---
## Implementation Sequence with Checkpoints
### Phase 1: File Creation (1-2 hours)
**Files:** `38_file_create.h`, `39_file_create.c`
**Implementation Steps:**
1. Implement `fs_create` using path resolution and inode allocation
2. Initialize inode with `S_IFREG` mode
3. Add directory entry using `dir_add_entry`
4. Handle all error cases (parent not found, exists, no space)
5. Update parent timestamps
**Checkpoint 1:**
```
$ make test_file_create
$ ./test_file_create
Testing file creation...
  Create "/testfile.txt": inode 2: OK
  Verify inode mode: S_IFREG | 0644: OK
  Verify size = 0: OK
  Verify blocks = 0: OK
  Create duplicate: 0, EEXIST: OK
  Create with non-existent parent: 0, ENOENT: OK
  Create 100 files: all succeed: OK
  Verify all in directory: OK
All file creation tests passed!
```
### Phase 2: File Read Loop (3-4 hours)
**Files:** `40_file_read.h`, `41_file_read.c`, `48_file_utils.h`, `49_file_utils.c`
**Implementation Steps:**
1. Implement `calculate_block_boundary` helper
2. Implement basic `fs_read` without hole handling
3. Add hole detection (check if `get_file_block` returns 0)
4. Implement zero-fill for holes
5. Handle block boundary crossing
6. Update atime after read
7. Implement `fs_read_ex` variant
**Checkpoint 2:**
```
$ make test_file_read
$ ./test_file_read
Testing file read...
  Read from empty file: 0 bytes: OK
  Read at offset >= size: 0 bytes: OK
  Read with pre-populated data:
    Write 100 bytes, read back: matches: OK
  Read across block boundary:
    Write at 4000, read 200 bytes (crosses boundary): OK
  Read from hole:
    Write at 1MB, read from 0: zeros: OK
    Verify no I/O for hole read: OK (check stats)
  Read past EOF:
    Write 100 bytes, read at offset 200: 0 bytes: OK
All file read tests passed!
```
### Phase 3: File Write Loop (3-4 hours)
**Files:** `42_file_write.h`, `43_file_write.c`
**Implementation Steps:**
1. Implement basic `fs_write` with block allocation
2. Integrate `get_or_alloc_block` from M2
3. Implement full-block optimization (skip read)
4. Implement partial-block read-modify-write
5. Handle block boundary crossing
6. Update size, mtime, ctime after write
7. Implement `fs_write_ex` variant
**Checkpoint 3:**
```
$ make test_file_write
$ ./test_file_write
Testing file write...
  Write to empty file:
    Write "Hello": 5 bytes written: OK
    Verify size = 5: OK
    Verify blocks = 8: OK
  Write at offset:
    Write at 100: extends file: OK
    Verify size = 106: OK
  Partial block write:
    Write 50 bytes at offset 4050: OK
    Verify read-modify-write: OK
  Full block write:
    Write 4096 bytes at offset 8192: OK
    Verify no read occurred (optimization): OK
  Write extends into indirect zone:
    Write 100KB: OK
    Verify indirect block allocated: OK
  Fill filesystem:
    Write until ENOSPC: OK
    Verify short write returned: OK
All file write tests passed!
```
### Phase 4: Block Allocation Integration (2-3 hours)
**Files:** Update `43_file_write.c`
**Implementation Steps:**
1. Test indirect block allocation on zone transition
2. Test double-indirect allocation for large files
3. Verify allocation cascade (3 blocks for first double-indirect write)
4. Test allocation failure handling
5. Update superblock free_blocks correctly
**Checkpoint 4:**
```
$ make test_write_allocation
$ ./test_write_allocation
Testing block allocation during write...
  Write within direct zone (48KB):
    12 blocks allocated: OK
    No indirect block: OK
  Transition to single-indirect:
    Write at 49KB: allocates indirect block: OK
    Total blocks: 13 data + 1 indirect = 14: OK
  Transition to double-indirect:
    Write at 2.1MB: allocates double-ind + indirect + data: OK
    Total new blocks: 3: OK
  Allocation failure:
    Fill disk, verify ENOSPC: OK
    Verify partial write returned: OK
All allocation tests passed!
```
### Phase 5: Truncate Grow (1-2 hours)
**Files:** `44_file_truncate.h`, `45_file_truncate.c` (partial)
**Implementation Steps:**
1. Implement truncate-to-grow case
2. Update size only, no block allocation
3. Verify subsequent read from extended region returns zeros
4. Handle edge case (truncate to same size)
**Checkpoint 5:**
```
$ make test_truncate_grow
$ ./test_truncate_grow
Testing truncate grow...
  Truncate empty file to 1MB:
    size = 1048576: OK
    blocks = 0: OK (no allocation!)
  Read from extended region:
    All zeros: OK
    No I/O performed: OK
  Write to extended region:
    Allocates block at write time: OK
  Truncate to same size:
    No changes: OK
All truncate grow tests passed!
```
### Phase 6: Truncate Shrink (3-4 hours)
**Files:** Update `45_file_truncate.c`
**Implementation Steps:**
1. Implement block freeing for truncate-shrink
2. Clear block pointers in inode
3. Free indirect blocks when entire zone freed
4. Handle partial block zeroing
5. Update superblock free_blocks
**Checkpoint 6:**
```
$ make test_truncate_shrink
$ ./test_truncate_shrink
Testing truncate shrink...
  Create 100KB file, truncate to 10KB:
    size = 10240: OK
    blocks reduced: OK
    Verify blocks freed in bitmap: OK
  Truncate to non-aligned size:
    Truncate to 6000 bytes: OK
    Verify partial block zeroed: OK
  Truncate to 0:
    All blocks freed: OK
    All pointers cleared: OK
    blocks = 0: OK
  Truncate frees indirect blocks:
    Create 3MB file, truncate to 10KB:
      indirect block freed: OK
      double-indirect freed: OK
All truncate shrink tests passed!
```
### Phase 7: Sparse File Handling (2-3 hours)
**Files:** `tests/test_sparse.c`
**Implementation Steps:**
1. Test creating sparse file (write at high offset)
2. Verify hole detection on read
3. Verify lazy allocation on write to hole
4. Test sparse file with multiple holes
5. Verify `du` reports actual block usage
**Checkpoint 7:**
```
$ make test_sparse
$ ./test_sparse
Testing sparse files...
  Create sparse file:
    Write 100 bytes at 10MB: OK
    size = 10485800: OK
    blocks = 24 (3 blocks): OK
    Metadata overhead: ~2.4MB logical, 12KB physical: OK
  Read from hole:
    Read at 0: all zeros: OK
    Read at 5MB: all zeros: OK
    Verify zero I/O for holes: OK
  Write fills hole:
    Write at 5MB: allocates block: OK
    Verify blocks increased: OK
  Multiple holes:
    Write at 1MB, 5MB, 10MB: OK
    Verify 9 blocks allocated (3 data + 6 indirect): OK
  Truncate creates hole:
    Truncate from 10KB to 1MB: OK
    Verify no new blocks: OK
All sparse file tests passed!
```
### Phase 8: Append Operation (1 hour)
**Files:** `46_file_append.h`, `47_file_append.c`
**Implementation Steps:**
1. Implement `fs_append` as write at current size
2. Test sequential appends
3. Verify size grows correctly
**Checkpoint 8:**
```
$ make test_append
$ ./test_append
Testing append...
  Append to empty file:
    Append "Hello": size = 5: OK
  Sequential appends:
    Append " World" 100 times: OK
    size = 600: OK
    Content matches: OK
  Large append:
    Append 1MB: OK
    Verify sequential blocks allocated: OK
All append tests passed!
```
### Phase 9: Integration Testing (2-3 hours)
**Files:** Comprehensive integration tests
**Implementation Steps:**
1. Test round-trip: write, read, verify
2. Test mixed operations: write, truncate, write, read
3. Test edge cases: empty file, 1-byte file, max size
4. Performance benchmarks
5. Stress tests
**Checkpoint 9:**
```
$ make test_integration
$ ./test_integration
Running integration tests...
Test: Round-trip verification
  Write 50KB random data: OK
  Read back: byte-for-byte match: OK
Test: Mixed operations
  Create file, write 10KB, truncate to 5KB: OK
  Append 3KB: size = 8KB: OK
  Read all: matches expected: OK
Test: Edge cases
  Empty file operations: OK
  1-byte file: OK
  Block-aligned file (exactly 4KB): OK
  File spanning indirect zone boundary: OK
Test: Performance
  Sequential read 10MB: 150 MB/s: OK
  Sequential write 10MB: 120 MB/s: OK
  Random read 1000 blocks: 25ms: OK
  Sparse read 1MB hole: 0 I/O: OK
Test: Stress
  1000 create/write/read/delete cycles: OK
  No memory leaks: OK
All integration tests passed!
```

![File Write Data Flow](./diagrams/tdd-diag-m4-07.svg)

---
## Test Specification
### test_file_create.c
```c
void test_create_basic() {
    FileSystem* fs = create_test_filesystem();
    uint64_t inode = fs_create(fs, "/testfile.txt", 0644);
    ASSERT(inode > 0);
    // Verify inode properties
    Inode in;
    read_inode(fs, inode, &in);
    ASSERT((in.mode & S_IFREG) != 0);
    ASSERT((in.mode & 07777) == 0644);
    ASSERT(in.size == 0);
    ASSERT(in.blocks == 0);
    ASSERT(in.link_count == 1);
    // Verify directory entry exists
    ASSERT(dir_lookup(fs, 1, "testfile.txt") == inode);
}
void test_create_duplicate() {
    FileSystem* fs = create_test_filesystem();
    fs_create(fs, "/file.txt", 0644);
    errno = 0;
    uint64_t inode2 = fs_create(fs, "/file.txt", 0644);
    ASSERT(inode2 == 0);
    ASSERT(errno == EEXIST);
}
void test_create_no_parent() {
    FileSystem* fs = create_test_filesystem();
    errno = 0;
    uint64_t inode = fs_create(fs, "/nonexistent/file.txt", 0644);
    ASSERT(inode == 0);
    ASSERT(errno == ENOENT);
}
void test_create_permissions() {
    FileSystem* fs = create_test_filesystem();
    fs_create(fs, "/readonly.txt", 0444);
    fs_create(fs, "/executable.txt", 0755);
    fs_create(fs, "/private.txt", 0600);
    Inode in;
    read_inode(fs, dir_lookup(fs, 1, "readonly.txt"), &in);
    ASSERT((in.mode & 07777) == 0444);
    read_inode(fs, dir_lookup(fs, 1, "executable.txt"), &in);
    ASSERT((in.mode & 07777) == 0755);
}
```
### test_file_read.c
```c
void test_read_empty_file() {
    FileSystem* fs = create_test_filesystem();
    uint64_t inode = fs_create(fs, "/empty.txt", 0644);
    char buffer[100];
    ssize_t n = fs_read(fs, inode, 0, buffer, sizeof(buffer));
    ASSERT(n == 0);  // 0 bytes from empty file
}
void test_read_basic() {
    FileSystem* fs = create_test_filesystem();
    uint64_t inode = fs_create(fs, "/test.txt", 0644);
    // Write some data
    const char* data = "Hello, World!";
    fs_write(fs, inode, 0, data, strlen(data));
    // Read it back
    char buffer[100];
    ssize_t n = fs_read(fs, inode, 0, buffer, sizeof(buffer));
    ASSERT(n == strlen(data));
    ASSERT(memcmp(buffer, data, n) == 0);
}
void test_read_partial() {
    FileSystem* fs = create_test_filesystem();
    uint64_t inode = fs_create(fs, "/test.txt", 0644);
    fs_write(fs, inode, 0, "Hello, World!", 13);
    // Read from middle
    char buffer[10];
    ssize_t n = fs_read(fs, inode, 7, buffer, 5);  // "World"
    ASSERT(n == 5);
    ASSERT(memcmp(buffer, "World", 5) == 0);
}
void test_read_past_eof() {
    FileSystem* fs = create_test_filesystem();
    uint64_t inode = fs_create(fs, "/test.txt", 0644);
    fs_write(fs, inode, 0, "Short", 5);
    // Try to read past end
    char buffer[100];
    ssize_t n = fs_read(fs, inode, 100, buffer, sizeof(buffer));
    ASSERT(n == 0);  // 0 bytes, not an error
}
void test_read_block_boundary() {
    FileSystem* fs = create_test_filesystem();
    uint64_t inode = fs_create(fs, "/test.txt", 0644);
    // Write pattern across boundary
    char pattern[8192];
    for (int i = 0; i < 8192; i++) {
        pattern[i] = (char)(i % 256);
    }
    fs_write(fs, inode, 0, pattern, 8192);
    // Read across boundary
    char buffer[8192];
    ssize_t n = fs_read(fs, inode, 0, buffer, 8192);
    ASSERT(n == 8192);
    ASSERT(memcmp(buffer, pattern, 8192) == 0);
}
void test_read_from_hole() {
    FileSystem* fs = create_test_filesystem();
    uint64_t inode = fs_create(fs, "/sparse.txt", 0644);
    // Write at 1MB, creating hole at 0
    fs_write(fs, inode, 1024 * 1024, "DATA", 4);
    // Read from hole
    char buffer[BLOCK_SIZE];
    ssize_t n = fs_read(fs, inode, 0, buffer, BLOCK_SIZE);
    ASSERT(n == BLOCK_SIZE);
    // Should be all zeros
    for (int i = 0; i < BLOCK_SIZE; i++) {
        ASSERT(buffer[i] == 0);
    }
}
```
### test_file_write.c
```c
void test_write_basic() {
    FileSystem* fs = create_test_filesystem();
    uint64_t inode = fs_create(fs, "/test.txt", 0644);
    const char* data = "Hello, World!";
    ssize_t n = fs_write(fs, inode, 0, data, strlen(data));
    ASSERT(n == strlen(data));
    Inode in;
    read_inode(fs, inode, &in);
    ASSERT(in.size == strlen(data));
    ASSERT(in.blocks == 8);  // One block = 8 × 512
}
void test_write_extends_file() {
    FileSystem* fs = create_test_filesystem();
    uint64_t inode = fs_create(fs, "/test.txt", 0644);
    fs_write(fs, inode, 0, "First", 5);
    fs_write(fs, inode, 100, "Second", 6);
    Inode in;
    read_inode(fs, inode, &in);
    ASSERT(in.size == 106);  // 100 + 6
}
void test_write_partial_block() {
    FileSystem* fs = create_test_filesystem();
    uint64_t inode = fs_create(fs, "/test.txt", 0644);
    // Write full block first
    char full_block[BLOCK_SIZE];
    memset(full_block, 'A', BLOCK_SIZE);
    fs_write(fs, inode, 0, full_block, BLOCK_SIZE);
    // Partial overwrite
    fs_write(fs, inode, 100, "BBBBB", 5);
    // Verify
    char buffer[BLOCK_SIZE];
    fs_read(fs, inode, 0, buffer, BLOCK_SIZE);
    ASSERT(buffer[99] == 'A');
    ASSERT(memcmp(buffer + 100, "BBBBB", 5) == 0);
    ASSERT(buffer[105] == 'A');
}
void test_write_at_high_offset() {
    FileSystem* fs = create_test_filesystem();
    uint64_t inode = fs_create(fs, "/sparse.txt", 0644);
    // Write at 1MB
    uint64_t offset = 1024 * 1024;
    ssize_t n = fs_write(fs, inode, offset, "DATA", 4);
    ASSERT(n == 4);
    Inode in;
    read_inode(fs, inode, &in);
    ASSERT(in.size == offset + 4);
    // Should have allocated some indirect structure
    ASSERT(in.blocks > 8);
}
void test_write_enospc() {
    // Create tiny filesystem
    FileSystem* fs = create_test_filesystem_tiny(100);  // 100 blocks
    uint64_t inode = fs_create(fs, "/fill.txt", 0644);
    // Write until full
    char data[BLOCK_SIZE];
    memset(data, 'X', BLOCK_SIZE);
    ssize_t total = 0;
    ssize_t n;
    while ((n = fs_write(fs, inode, total, data, BLOCK_SIZE)) > 0) {
        total += n;
    }
    // Should have stopped due to ENOSPC
    ASSERT(errno == ENOSPC || errno == 0);
    ASSERT(total > 0);  // Wrote something
    // Verify partial write was saved
    Inode in;
    read_inode(fs, inode, &in);
    ASSERT(in.size == total);
}
```
### test_file_truncate.c
```c
void test_truncate_grow() {
    FileSystem* fs = create_test_filesystem();
    uint64_t inode = fs_create(fs, "/test.txt", 0644);
    fs_write(fs, inode, 0, "Hello", 5);
    // Grow to 1MB
    int ret = fs_truncate(fs, inode, 1024 * 1024);
    ASSERT(ret == 0);
    Inode in;
    read_inode(fs, inode, &in);
    ASSERT(in.size == 1024 * 1024);
    ASSERT(in.blocks == 8);  // Still just 1 block (the original data)
    // Read from new region (should be zeros)
    char buffer[100];
    fs_read(fs, inode, 500000, buffer, 100);
    for (int i = 0; i < 100; i++) {
        ASSERT(buffer[i] == 0);
    }
}
void test_truncate_shrink() {
    FileSystem* fs = create_test_filesystem();
    uint64_t inode = fs_create(fs, "/test.txt", 0644);
    // Write 10KB
    char data[10240];
    memset(data, 'X', sizeof(data));
    fs_write(fs, inode, 0, data, sizeof(data));
    Inode before;
    read_inode(fs, inode, &before);
    uint64_t blocks_before = before.blocks;
    // Shrink to 4KB
    int ret = fs_truncate(fs, inode, 4096);
    ASSERT(ret == 0);
    Inode after;
    read_inode(fs, inode, &after);
    ASSERT(after.size == 4096);
    ASSERT(after.blocks < blocks_before);
    // Verify data preserved
    char buffer[4096];
    fs_read(fs, inode, 0, buffer, 4096);
    for (int i = 0; i < 4096; i++) {
        ASSERT(buffer[i] == 'X');
    }
}
void test_truncate_to_zero() {
    FileSystem* fs = create_test_filesystem();
    uint64_t inode = fs_create(fs, "/test.txt", 0644);
    // Write some data
    char data[10000];
    fs_write(fs, inode, 0, data, sizeof(data));
    // Truncate to zero
    int ret = fs_truncate(fs, inode, 0);
    ASSERT(ret == 0);
    Inode in;
    read_inode(fs, inode, &in);
    ASSERT(in.size == 0);
    ASSERT(in.blocks == 0);
    // All block pointers should be cleared
    for (int i = 0; i < 12; i++) {
        ASSERT(in.direct[i] == 0);
    }
    ASSERT(in.indirect == 0);
    ASSERT(in.double_ind == 0);
}
void test_truncate_partial_block() {
    FileSystem* fs = create_test_filesystem();
    uint64_t inode = fs_create(fs, "/test.txt", 0644);
    // Write full block
    char data[BLOCK_SIZE];
    memset(data, 'A', BLOCK_SIZE);
    fs_write(fs, inode, 0, data, BLOCK_SIZE);
    // Truncate to partial block
    fs_truncate(fs, inode, 1000);
    // Verify partial block zeroed
    char buffer[BLOCK_SIZE];
    fs_read(fs, inode, 0, buffer, BLOCK_SIZE);
    // First 1000 bytes should be 'A'
    for (int i = 0; i < 1000; i++) {
        ASSERT(buffer[i] == 'A');
    }
    // Rest should be zero (though we can't read past size)
}
```
### test_sparse.c
```c
void test_sparse_creation() {
    FileSystem* fs = create_test_filesystem();
    uint64_t inode = fs_create(fs, "/sparse.dat", 0644);
    // Write at 10MB
    uint64_t offset = 10 * 1024 * 1024;
    const char* data = "MARKER";
    fs_write(fs, inode, offset, data, strlen(data));
    Inode in;
    read_inode(fs, inode, &in);
    // Logical size should be ~10MB
    ASSERT(in.size == offset + strlen(data));
    // Physical blocks should be much less
    uint64_t logical_blocks = in.size / BLOCK_SIZE;  // ~2560
    uint64_t physical_blocks = in.blocks / 8;        // ~3
    ASSERT(physical_blocks < logical_blocks / 100);  // <1% allocated
}
void test_sparse_read_hole() {
    FileSystem* fs = create_test_filesystem();
    uint64_t inode = fs_create(fs, "/sparse.dat", 0644);
    // Write at 1MB
    fs_write(fs, inode, 1024 * 1024, "END", 3);
    // Read from hole at 512KB
    char buffer[100];
    ssize_t n = fs_read(fs, inode, 512 * 1024, buffer, 100);
    ASSERT(n == 100);
    // Should be all zeros
    for (int i = 0; i < 100; i++) {
        ASSERT(buffer[i] == 0);
    }
}
void test_sparse_fill_hole() {
    FileSystem* fs = create_test_filesystem();
    uint64_t inode = fs_create(fs, "/sparse.dat", 0644);
    // Create hole by writing at high offset
    fs_write(fs, inode, 1024 * 1024, "END", 3);
    Inode before;
    read_inode(fs, inode, &before);
    uint64_t blocks_before = before.blocks;
    // Fill part of the hole
    fs_write(fs, inode, 512 * 1024, "MIDDLE", 6);
    Inode after;
    read_inode(fs, inode, &after);
    // Should have allocated more blocks
    ASSERT(after.blocks > blocks_before);
    // Verify both regions have correct data
    char buffer[10];
    fs_read(fs, inode, 512 * 1024, buffer, 6);
    ASSERT(memcmp(buffer, "MIDDLE", 6) == 0);
    fs_read(fs, inode, 1024 * 1024, buffer, 3);
    ASSERT(memcmp(buffer, "END", 3) == 0);
}
void test_sparse_multiple_holes() {
    FileSystem* fs = create_test_filesystem();
    uint64_t inode = fs_create(fs, "/sparse.dat", 0644);
    // Write at multiple offsets
    fs_write(fs, inode, 0, "A", 1);              // Block 0
    fs_write(fs, inode, 10 * 1024 * 1024, "B", 1);  // Block 2560
    fs_write(fs, inode, 20 * 1024 * 1024, "C", 1);  // Block 5120
    // Verify all data
    char buf[1];
    fs_read(fs, inode, 0, buf, 1);
    ASSERT(buf[0] == 'A');
    fs_read(fs, inode, 10 * 1024 * 1024, buf, 1);
    ASSERT(buf[0] == 'B');
    fs_read(fs, inode, 20 * 1024 * 1024, buf, 1);
    ASSERT(buf[0] == 'C');
    // Verify holes
    fs_read(fs, inode, 5 * 1024 * 1024, buf, 1);
    ASSERT(buf[0] == 0);
}
```
---
## Performance Targets
| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| `fs_read` (sequential, 1MB) | > 100 MB/s (NVMe) | `time ./bench_read sequential 1048576` |
| `fs_read` (sequential, 1MB, HDD) | > 40 MB/s | Same with HDD backing |
| `fs_read` (random, 1000 blocks) | < 50 ms (NVMe) | `time ./bench_read random 1000` |
| `fs_read` (hole, 1MB) | < 1 ms (memset only) | `time ./bench_read hole 1048576` |
| `fs_write` (sequential, 1MB) | > 80 MB/s (NVMe) | `time ./bench_write sequential 1048576` |
| `fs_write` (partial block) | < 100 μs | `time ./bench_write partial 100` |
| `fs_write` (sparse, 10MB hole + 1 byte) | < 1 ms | `time ./bench_write sparse` |
| `fs_truncate` (grow to 1GB) | < 1 ms (metadata only) | `time ./bench_truncate grow 1073741824` |
| `fs_truncate` (shrink 1GB to 0) | < 100 ms | `time ./bench_truncate shrink 0` |
| `fs_create` | < 500 μs | `time ./bench_create 1` |
| `fs_append` (1KB, 1000 times) | < 500 ms total | `time ./bench_append 1000 1024` |
**Hardware Soul - Latency Breakdown:**
```
fs_read of 4KB block (NVMe, cold cache):
  1. read_inode: ~25 μs (1 block)
  2. get_file_block (direct): ~0 μs (in memory)
  3. read_block: ~25 μs (1 block)
  4. memcpy: ~1 μs
  Total: ~51 μs for 4KB = ~78 MB/s
fs_read of 4KB block (double-indirect, cold cache):
  1. read_inode: ~25 μs
  2. get_file_block double-ind: ~50 μs (2 block reads)
  3. read_block: ~25 μs
  Total: ~100 μs for 4KB = ~40 MB/s
fs_write of 4KB (full block, direct zone):
  1. read_inode: ~25 μs
  2. get_or_alloc_block: ~100 μs (bitmap + allocation)
  3. write_block: ~25 μs
  4. write_inode: ~25 μs
  Total: ~175 μs for 4KB = ~23 MB/s
fs_write of 100 bytes (partial block):
  1. read_inode: ~25 μs
  2. get_or_alloc_block: ~100 μs
  3. read_block (read-modify-write): ~25 μs
  4. memcpy: ~0.1 μs
  5. write_block: ~25 μs
  6. write_inode: ~25 μs
  Total: ~200 μs for 100 bytes = ~0.5 MB/s (write amplification!)
```

![Block Allocation During Write](./diagrams/tdd-diag-m4-08.svg)

---
## Diagrams

![Metadata Update Matrix](./diagrams/tdd-diag-m4-09.svg)


![File I/O Three-Level View](./diagrams/tdd-diag-m4-10.svg)

---
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: filesystem-m5 -->
# Technical Design Specification: FUSE Integration
## Module Charter
This module implements the FUSE (Filesystem in Userspace) daemon that mounts the filesystem as a real mount point accessible to all operating system programs through standard POSIX syscalls. It translates kernel VFS callbacks into filesystem operations by resolving path strings to inodes, executing the requested operations, and returning appropriate error codes. The module provides thread-safe access through mutex locking around all shared structures, handles file handle state management through the `fuse_file_info` structure, and ensures proper mount/unmount lifecycle with data flushing. It does NOT implement the underlying filesystem logic (that's M1-M4), journaling for crash consistency (M6), or interpret directory entry formats beyond what's needed for readdir. The core invariants: every FUSE callback returns a valid negative errno on error; the `fi->fh` field correctly stores inode numbers across open/read/write/release sequences; all operations are serialized through locks to prevent concurrent modification of shared structures; and unmount flushes all pending data to stable storage.
---
## File Structure
```
filesystem/
├── 50_fuse_main.h       # Main program declarations, argument parsing
├── 51_fuse_main.c       # main(), fuse_main integration, signal handling
├── 52_fuse_callbacks.h  # All callback function declarations
├── 53_fuse_getattr.c    # fs_getattr, fs_statfs implementations
├── 54_fuse_dir.c        # fs_mkdir, fs_rmdir, fs_readdir, fs_opendir, fs_releasedir
├── 55_fuse_file.c       # fs_create, fs_open, fs_read, fs_write, fs_release
├── 56_fuse_meta.c       # fs_chmod, fs_chown, fs_truncate, fs_utimens
├── 57_fuse_delete.c     # fs_unlink, fs_rename implementations
├── 58_fuse_symlink.c    # fs_symlink, fs_readlink, fs_link implementations
├── 59_fuse_lock.h       # Locking declarations, LockManager struct
├── 60_fuse_lock.c       # pthread_mutex implementation, lock helpers
└── tests/
    ├── test_fuse_mount.c     # Mount/unmount lifecycle tests
    ├── test_fuse_callbacks.c # Individual callback tests
    ├── test_fuse_concurrent.c # Thread safety tests
    └── test_fuse_real_tools.c # Tests with ls, cat, cp, etc.
```
---
## Complete Data Model
### FuseGlobal Structure
The global state accessible to all FUSE callbacks.
```c
// 50_fuse_main.h
#include <stdint.h>
#include <pthread.h>
#include "03_superblock.h"
#include "14_inode.h"
/**
 * Global filesystem state for FUSE callbacks.
 * Single instance shared across all callback threads.
 */
typedef struct {
    FileSystem*  fs;              // Core filesystem structure
    char*        disk_image_path; // Path to backing disk image
    bool         debug_mode;      // Enable verbose logging
    bool         read_only;       // Mount read-only (no write operations)
    // Locking
    pthread_mutex_t global_lock;  // Coarse-grained lock for all operations
    pthread_mutexattr_t lock_attr;
    // Statistics (for debugging)
    uint64_t      op_count;       // Total operations processed
    uint64_t      error_count;    // Total errors returned
} FuseGlobal;
// Global instance (set in main, accessed by callbacks)
extern FuseGlobal* g_fuse;
// Compile-time configuration
#define FUSE_MAX_WRITE_SIZE  (128 * 1024)  // 128KB max write chunk
#define FUSE_MAX_READ_SIZE   (128 * 1024)  // 128KB max read chunk
```
**Why Each Field Exists:**
- `fs`: The filesystem instance all operations target
- `disk_image_path`: Needed for error messages and potential reopen
- `debug_mode`: Enables per-operation logging for debugging
- `read_only`: Prevents all write operations when true
- `global_lock`: Serializes all filesystem modifications
- `op_count`, `error_count`: Monitoring and debugging metrics
### FuseFileHandle Structure
Extended file handle stored in `fi->fh`.
```c
// 52_fuse_callbacks.h
/**
 * File handle structure stored in fuse_file_info->fh.
 * Converted to/from uint64_t for storage.
 */
typedef struct {
    uint64_t inode_num;     // Inode number of open file
    uint32_t open_flags;    // Original O_* flags from open
    uint32_t sequence;      // Monotonic sequence for debugging
} FuseFileHandle;
#define FH_INODE(fh)     ((FuseFileHandle*)&(fh))->inode_num
#define FH_FLAGS(fh)     ((FuseFileHandle*)&(fh))->open_flags
#define FH_SEQUENCE(fh)  ((FuseFileHandle*)&(fh))->sequence
// Sequence counter for debugging
extern uint32_t g_fh_sequence;
```
**Byte Layout:**
| Offset | Size | Field | Purpose |
|--------|------|-------|---------|
| 0x00 | 8 | inode_num | Identifies the file |
| 0x08 | 4 | open_flags | O_RDONLY, O_WRONLY, O_RDWR, O_APPEND |
| 0x0C | 4 | sequence | Unique ID for debugging |
| **Total** | **16** | | Fits in uint64_t with truncation |
**Note:** For simplicity, we store only the inode number in `fi->fh` (8 bytes). The full `FuseFileHandle` structure is available for future expansion.
### FuseDirHandle Structure
Directory handle for opendir/releasedir.
```c
// 52_fuse_callbacks.h
/**
 * Directory handle for opendir/releasedir.
 * Stored in fi->fh during directory iteration.
 */
typedef struct {
    uint64_t inode_num;     // Directory inode being iterated
    uint64_t current_block; // Current block index (for seekdir/telldir)
    int      entry_index;   // Current entry within block
} FuseDirHandle;
```
### StatfsResult Structure
Result structure for statfs callback.
```c
// 52_fuse_callbacks.h
/**
 * Filesystem statistics returned by statfs.
 * Maps directly to struct statvfs fields.
 */
typedef struct {
    uint64_t block_size;      // Optimal block size for I/O
    uint64_t fragment_size;   // Fragment size (same as block_size)
    uint64_t total_blocks;    // Total blocks in filesystem
    uint64_t free_blocks;     // Free blocks available
    uint64_t avail_blocks;    // Free blocks for non-root
    uint64_t total_files;     // Total file nodes (inodes)
    uint64_t free_files;      // Free file nodes
    uint64_t avail_files;     // Free file nodes for non-root
    uint64_t fsid;            // Filesystem ID
    uint64_t namelen;         // Maximum filename length
} StatfsResult;
```
### LockManager Structure (Future Fine-Grained Locking)
```c
// 59_fuse_lock.h
/**
 * Lock manager for fine-grained inode locking.
 * Current implementation uses single global lock.
 * Future: per-inode locks for better parallelism.
 */
typedef struct {
    pthread_mutex_t global_lock;      // Current: single lock
    // Future expansion:
    // pthread_mutex_t inode_locks[MAX_INODES];
    // pthread_rwlock_t bitmap_lock;
    // pthread_rwlock_t superblock_lock;
} LockManager;
// Lock macros for easy switching between strategies
#define FS_LOCK(g)    pthread_mutex_lock(&(g)->global_lock)
#define FS_UNLOCK(g)  pthread_mutex_unlock(&(g)->global_lock)
#define FS_LOCKED(g)  pthread_mutex_trylock(&(g)->global_lock) == EBUSY
```
**Hardware Soul - Locking Overhead:**
```
pthread_mutex_lock (uncontended):
  - CPU cycles: ~25-50 (atomic operations + branch)
  - Latency: ~25ns on modern x86
  - Cache impact: 1 cache line for mutex (shared/exclusive)
pthread_mutex_lock (contended):
  - CPU cycles: ~500-2000 (context switch to kernel)
  - Latency: ~1-5μs (futex wait/wake)
  - Cache impact: Cache line bounce between cores
FUSE callback with global lock:
  - Context switch in: ~1000 cycles
  - FS_LOCK: ~25 cycles (uncontended)
  - Operation: varies (disk I/O dominated)
  - FS_UNLOCK: ~25 cycles
  - Context switch out: ~1000 cycles
Lock contention scenario:
  - 8 threads, all doing getattr on same directory
  - Only 1 thread in filesystem code at a time
  - Effective parallelism: 12.5% of CPU capacity
  - Mitigation: Fine-grained per-inode locks (future)
```
---
## Interface Contracts
### FUSE Operations Structure
```c
// 52_fuse_callbacks.h
#include <fuse3/fuse.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
/**
 * FUSE operations structure.
 * Maps callback names to implementation functions.
 * Passed to fuse_main() during initialization.
 */
static const struct fuse_operations fs_fuse_oper = {
    // File attributes
    .getattr     = fs_fuse_getattr,
    .statfs      = fs_fuse_statfs,
    // Directory operations
    .mkdir       = fs_fuse_mkdir,
    .rmdir       = fs_fuse_rmdir,
    .opendir     = fs_fuse_opendir,
    .readdir     = fs_fuse_readdir,
    .releasedir  = fs_fuse_releasedir,
    // File operations
    .create      = fs_fuse_create,
    .open        = fs_fuse_open,
    .read        = fs_fuse_read,
    .write       = fs_fuse_write,
    .release     = fs_fuse_release,
    .truncate    = fs_fuse_truncate,
    .ftruncate   = fs_fuse_ftruncate,
    // Metadata operations
    .chmod       = fs_fuse_chmod,
    .chown       = fs_fuse_chown,
    .utimens     = fs_fuse_utimens,
    // Delete/rename
    .unlink      = fs_fuse_unlink,
    .rename      = fs_fuse_rename,
    // Links
    .symlink     = fs_fuse_symlink,
    .readlink    = fs_fuse_readlink,
    .link        = fs_fuse_link,
    // Lifecycle
    .init        = fs_fuse_init,
    .destroy     = fs_fuse_destroy,
    // Flags for capabilities
    .flag_nullpath_ok = 0,     // We need paths for resolution
    .flag_utime_omit_ok = 1,   // Handle UTIME_NOW/UTIME_OMIT
};
```
### fs_fuse_getattr
```c
/**
 * Get file attributes (stat).
 *
 * @param path  File path (absolute from mount point)
 * @param stbuf Output: stat structure to fill
 * @param fi    File info (may be NULL if called without open file)
 * @return 0 on success, -errno on failure
 *
 * FUSE calls this for:
 *   - Every file access (permission check)
 *   - ls -l (to display sizes, timestamps)
 *   - stat() syscall
 *   - Tab completion in shell
 *
 * Performance: This is the MOST FREQUENTLY CALLED callback.
 * A single ls -la in a 100-file directory triggers 101 getattr calls.
 *
 * Errors:
 *   -ENOENT  - Path does not exist
 *   -ENOTDIR - Component of path is not a directory
 *   -EIO     - Read error from block device
 */
static int fs_fuse_getattr(const char* path, struct stat* stbuf,
                           struct fuse_file_info* fi);
```
### fs_fuse_readdir
```c
/**
 * Read directory entries.
 *
 * @param path    Directory path
 * @param buf     Buffer for filler function
 * @param filler  Function to add entries to buffer
 * @param offset  Offset for seeking (usually 0)
 * @param fi      File info from opendir
 * @param flags   FUSE_READDIR_* flags
 * @return 0 on success, -errno on failure
 *
 * The filler function signature:
 *   int filler(void* buf, const char* name,
 *              const struct stat* stbuf, off_t off);
 *
 * Call filler once per directory entry. Return non-zero from filler
 * means buffer is full.
 *
 * MUST include "." and ".." entries.
 *
 * Errors:
 *   -ENOENT  - Directory does not exist
 *   -ENOTDIR - Path is not a directory
 *   -EACCES  - Permission denied
 *   -EIO     - Read error
 */
static int fs_fuse_readdir(const char* path, void* buf,
                           fuse_fill_dir_t filler, off_t offset,
                           struct fuse_file_info* fi,
                           enum fuse_readdir_flags flags);
```
### fs_fuse_create
```c
/**
 * Create and open a file (atomic create+open).
 *
 * @param path  File path to create
 * @param mode  File mode (permissions + type)
 * @param fi    File info (fh will be set to inode number)
 * @return 0 on success, -errno on failure
 *
 * This is the modern way to create files. FUSE calls this for
 * open(path, O_CREAT | O_TRUNC | O_WRONLY, mode).
 *
 * If O_EXCL is set and file exists, return -EEXIST.
 * If file doesn't exist, create it with given mode.
 * Set fi->fh to inode number for subsequent read/write.
 *
 * Errors:
 *   -ENOENT  - Parent directory does not exist
 *   -ENOTDIR - Parent is not a directory
 *   -EEXIST  - File exists (with O_EXCL)
 *   -EACCES  - Permission denied
 *   -ENOSPC  - No space (inodes or blocks)
 *   -EIO     - Write error
 */
static int fs_fuse_create(const char* path, mode_t mode,
                          struct fuse_file_info* fi);
```
### fs_fuse_read
```c
/**
 * Read from an open file.
 *
 * @param path    File path (may be NULL if fi->fh is valid)
 * @param buf     Output buffer for read data
 * @param size    Maximum bytes to read
 * @param offset  File offset to read from
 * @param fi      File info (fh contains inode number)
 * @return Bytes read on success, -errno on failure
 *
 * Use fi->fh to get inode number. Avoid re-resolving path.
 *
 * Reading past EOF: Return 0 (not an error).
 * Reading from hole: Return zeros (sparse file semantics).
 *
 * Errors:
 *   -EISDIR  - Path is a directory
 *   -EACCES  - Permission denied
 *   -EIO     - Read error
 */
static int fs_fuse_read(const char* path, char* buf, size_t size,
                        off_t offset, struct fuse_file_info* fi);
```
### fs_fuse_write
```c
/**
 * Write to an open file.
 *
 * @param path    File path (may be NULL if fi->fh is valid)
 * @param buf     Data to write
 * @param size    Number of bytes to write
 * @param offset  File offset to write to
 * @param fi      File info (fh contains inode number)
 * @return Bytes written on success, -errno on failure
 *
 * Writing past EOF extends the file.
 * Writing to a hole allocates blocks.
 *
 * May return fewer bytes than requested (short write) on ENOSPC.
 *
 * Errors:
 *   -EISDIR  - Path is a directory
 *   -EACCES  - Permission denied (read-only or no write permission)
 *   -EFBIG   - File too large
 *   -ENOSPC  - No space (may write partial)
 *   -EIO     - Write error
 */
static int fs_fuse_write(const char* path, const char* buf, size_t size,
                         off_t offset, struct fuse_file_info* fi);
```
### fs_fuse_unlink
```c
/**
 * Remove a file (unlink).
 *
 * @param path  File path to remove
 * @return 0 on success, -errno on failure
 *
 * Removes directory entry and decrements link count.
 * If link count reaches 0, inode and blocks are freed.
 *
 * Errors:
 *   -ENOENT  - File does not exist
 *   -EISDIR  - Path is a directory (use rmdir)
 *   -EACCES  - Permission denied
 *   -EIO     - Write error
 */
static int fs_fuse_unlink(const char* path);
```
### fs_fuse_rename
```c
/**
 * Rename/move a file or directory.
 *
 * @param oldpath  Current path
 * @param newpath  New path
 * @param flags    RENAME_* flags (usually 0)
 * @return 0 on success, -errno on failure
 *
 * Atomic operation: file always exists at either old or new path.
 *
 * For directory renames:
 *   - Update .. entry in moved directory
 *   - Update parent link counts
 *
 * If newpath exists:
 *   - File: overwrite (if permitted)
 *   - Directory: fail with -EEXIST (must be empty for overwrite)
 *
 * Errors:
 *   -ENOENT  - oldpath does not exist
 *   -ENOTDIR - Parent is not a directory
 *   -EEXIST  - newpath exists (and can't overwrite)
 *   -EINVAL  - Moving into self (oldpath is prefix of newpath)
 *   -EACCES  - Permission denied
 *   -EIO     - Write error
 */
static int fs_fuse_rename(const char* oldpath, const char* newpath,
                          unsigned int flags);
```
### fs_fuse_mkdir
```c
/**
 * Create a directory.
 *
 * @param path  Directory path to create
 * @param mode  Directory permissions
 * @return 0 on success, -errno on failure
 *
 * Creates directory with . and .. entries.
 * Parent's link count increases by 1.
 *
 * Errors:
 *   -ENOENT  - Parent does not exist
 *   -ENOTDIR - Parent is not a directory
 *   -EEXIST  - Name already exists
 *   -EACCES  - Permission denied
 *   -ENOSPC  - No space
 *   -EIO     - Write error
 */
static int fs_fuse_mkdir(const char* path, mode_t mode);
```
### fs_fuse_rmdir
```c
/**
 * Remove a directory.
 *
 * @param path  Directory path to remove
 * @return 0 on success, -errno on failure
 *
 * Directory must be empty (only . and ..).
 * Parent's link count decreases by 1.
 *
 * Errors:
 *   -ENOENT   - Directory does not exist
 *   -ENOTDIR  - Path is not a directory
 *   -ENOTEMPTY - Directory is not empty
 *   -EACCES   - Permission denied
 *   -EBUSY    - Trying to remove root
 *   -EIO      - Write error
 */
static int fs_fuse_rmdir(const char* path);
```
### fs_fuse_truncate
```c
/**
 * Change file size.
 *
 * @param path      File path
 * @param new_size  New size in bytes
 * @param fi        File info (may be NULL)
 * @return 0 on success, -errno on failure
 *
 * Growing: Creates hole (no allocation).
 * Shrinking: Frees blocks beyond new size.
 *
 * Errors:
 *   -EISDIR  - Path is a directory
 *   -EFBIG   - new_size too large
 *   -EACCES  - Permission denied
 *   -EIO     - Write error
 */
static int fs_fuse_truncate(const char* path, off_t new_size,
                            struct fuse_file_info* fi);
```
### fs_fuse_chmod
```c
/**
 * Change file permissions.
 *
 * @param path  File path
 * @param mode  New permission bits
 * @param fi    File info (may be NULL)
 * @return 0 on success, -errno on failure
 *
 * Only changes permission bits, preserves file type.
 *
 * Errors:
 *   -ENOENT  - File does not exist
 *   -EPERM   - Not owner (simplified - we don't check)
 *   -EIO     - Write error
 */
static int fs_fuse_chmod(const char* path, mode_t mode,
                         struct fuse_file_info* fi);
```
### fs_fuse_utimens
```c
/**
 * Change file timestamps.
 *
 * @param path  File path
 * @param ts    Timestamp array [atime, mtime]
 * @param fi    File info (may be NULL)
 * @return 0 on success, -errno on failure
 *
 * ts[0] = atime, ts[1] = mtime
 * tv_nsec may be UTIME_NOW (current time) or UTIME_OMIT (don't change).
 *
 * ctime is always updated to current time.
 *
 * Errors:
 *   -ENOENT  - File does not exist
 *   -EIO     - Write error
 */
static int fs_fuse_utimens(const char* path, const struct timespec ts[2],
                           struct fuse_file_info* fi);
```
### fs_fuse_statfs
```c
/**
 * Get filesystem statistics (df uses this).
 *
 * @param path   Any path in filesystem
 * @param stbuf  Output: statvfs structure
 * @return 0 on success, -errno on failure
 *
 * Fills in block counts, inode counts, limits.
 *
 * Errors:
 *   -EIO  - Read error (superblock)
 */
static int fs_fuse_statfs(const char* path, struct statvfs* stbuf);
```
### fs_fuse_init / fs_fuse_destroy
```c
/**
 * Initialize filesystem (called on mount).
 *
 * @param conn  Connection info (can set capabilities)
 * @return Private data (we return g_fuse)
 *
 * Enable FUSE capabilities here:
 *   - FUSE_CAP_BIG_WRITES: Larger write chunks
 *   - FUSE_CAP_ASYNC_READ: Async reads
 *   - FUSE_CAP_WRITEBACK_CACHE: Buffer writes in kernel
 */
static void* fs_fuse_init(struct fuse_conn_info* conn,
                          struct fuse_config* cfg);
/**
 * Cleanup filesystem (called on unmount).
 *
 * @param private_data  What we returned from init
 *
 * Flush all pending data. Sync disk image.
 * DO NOT free g_fuse here (main does that).
 */
static void fs_fuse_destroy(void* private_data);
```
---
## Algorithm Specification
### fs_fuse_getattr Algorithm
**Input:** `path`, `stbuf`, `fi`
**Output:** 0 on success with stbuf filled, -errno on failure
```
ALGORITHM fs_fuse_getattr(path, stbuf, fi):
  // Clear output structure
  memset(stbuf, 0, sizeof(struct stat))
  // Try to use file handle if available (avoids path resolution)
  inode_num ← 0
  IF fi ≠ NULL AND fi->fh ≠ 0:
    inode_num ← fi->fh
  ELSE:
    // Resolve path to inode
    FS_LOCK(g_fuse)
    inode_num ← path_resolve(g_fuse->fs, path)
    FS_UNLOCK(g_fuse)
    IF inode_num = 0:
      IF errno = 0:
        errno ← ENOENT
      RETURN -errno
  // Read inode
  FS_LOCK(g_fuse)
  result ← read_inode(g_fuse->fs->dev, g_fuse->fs->sb, inode_num, &inode)
  FS_UNLOCK(g_fuse)
  IF result < 0:
    RETURN -EIO
  // Fill stat structure
  stbuf->st_ino     ← inode_num
  stbuf->st_mode    ← inode.mode
  stbuf->st_nlink   ← inode.link_count
  stbuf->st_uid     ← inode.uid
  stbuf->st_gid     ← inode.gid
  stbuf->st_size    ← inode.size
  stbuf->st_blocks  ← inode.blocks  // 512-byte units
  stbuf->st_atime   ← inode.atime
  stbuf->st_mtime   ← inode.mtime
  stbuf->st_ctime   ← inode.ctime
  stbuf->st_blksize ← BLOCK_SIZE
  // Set device ID
  stbuf->st_dev     ← 0  // Would be from stat of mount point
  RETURN 0
```
**Performance Analysis:**
```
fs_fuse_getattr with path resolution:
  1. path_resolve: O(depth × avg_dir_entries)
     - 5-component path with 10-entry directories
     - ~50 dir_lookup calls
     - ~50 × (bitmap + inode reads) = expensive without caching
  2. read_inode: 1 block read
Total: ~51 block reads worst case (cold cache)
Optimization: Dentry cache (future)
  - Cache path → inode mappings
  - Hit rate: ~95% for typical workloads
  - Cache hit: ~1μs vs ~10ms miss
```

![FUSE Architecture](./diagrams/tdd-diag-m5-01.svg)

### fs_fuse_readdir Algorithm
**Input:** `path`, `buf`, `filler`, `offset`, `fi`, `flags`
**Output:** 0 on success, -errno on failure
```
ALGORITHM fs_fuse_readdir(path, buf, filler, offset, fi, flags):
  // Get directory inode
  IF fi ≠ NULL AND fi->fh ≠ 0:
    dir_inode_num ← fi->fh
  ELSE:
    FS_LOCK(g_fuse)
    dir_inode_num ← path_resolve(g_fuse->fs, path)
    FS_UNLOCK(g_fuse)
    IF dir_inode_num = 0:
      RETURN -ENOENT
  // Read directory inode
  FS_LOCK(g_fuse)
  result ← read_inode(g_fuse->fs->dev, g_fuse->fs->sb, dir_inode_num, &dir_inode)
  FS_UNLOCK(g_fuse)
  IF result < 0:
    RETURN -EIO
  // Verify it's a directory
  IF (dir_inode.mode & S_IFDIR) = 0:
    RETURN -ENOTDIR
  // Calculate directory size in blocks
  num_blocks ← (dir_inode.size + BLOCK_SIZE - 1) / BLOCK_SIZE
  IF num_blocks = 0:
    num_blocks ← 1
  // Allocate buffers
  block_buffer ← allocate(BLOCK_SIZE)
  indirect_buffer ← allocate(BLOCK_SIZE)
  // Iterate through directory blocks
  FOR block_idx FROM 0 TO num_blocks - 1:
    // Get physical block
    FS_LOCK(g_fuse)
    phys_block ← get_file_block(g_fuse->fs->dev, &dir_inode, 
                                  block_idx, indirect_buffer)
    FS_UNLOCK(g_fuse)
    IF phys_block = 0:
      CONTINUE  // Hole in directory
    // Read directory block
    FS_LOCK(g_fuse)
    result ← read_block(g_fuse->fs->dev, phys_block, block_buffer)
    FS_UNLOCK(g_fuse)
    IF result < 0:
      free(block_buffer)
      free(indirect_buffer)
      RETURN -EIO
    // Scan entries in block
    entries ← cast(block_buffer to DirEntry*)
    FOR entry_idx FROM 0 TO ENTRIES_PER_BLOCK - 1:
      entry ← &entries[entry_idx]
      // Skip unused entries
      IF entry->inode = 0:
        CONTINUE
      // Prepare minimal stat for filler
      struct stat entry_stat
      memset(&entry_stat, 0, sizeof(entry_stat))
      entry_stat.st_ino ← entry->inode
      // Set file type in mode
      SWITCH entry->file_type:
        CASE DT_DIR:
          entry_stat.st_mode ← S_IFDIR
        CASE DT_LNK:
          entry_stat.st_mode ← S_IFLNK
        DEFAULT:
          entry_stat.st_mode ← S_IFREG
      // Add entry to buffer
      // filler returns 1 if buffer is full
      IF filler(buf, entry->name, &entry_stat, 0, FUSE_FILL_DIR_PLUS) ≠ 0:
        free(block_buffer)
        free(indirect_buffer)
        RETURN 0  // Buffer full, but success so far
  free(block_buffer)
  free(indirect_buffer)
  RETURN 0
```
**Hardware Soul - readdir Performance:**
```
Directory with 140 entries (10 blocks):
  Block reads: 10
  Entry processing: 140 × (memcmp + filler call)
  Filler overhead: ~100ns per entry (memcpy to buffer)
Total: ~10 × 25μs + 140 × 0.1μs = ~264μs (NVMe)
For 10,000 entry directory (715 blocks):
  Block reads: 715
  Processing: ~18ms (NVMe)
  With caching: First readdir slow, subsequent fast
```

![FUSE Callback Flow](./diagrams/tdd-diag-m5-02.svg)

### fs_fuse_create Algorithm
**Input:** `path`, `mode`, `fi`
**Output:** 0 on success with fi->fh set, -errno on failure
```
ALGORITHM fs_fuse_create(path, mode, fi):
  // Check for read-only mount
  IF g_fuse->read_only:
    RETURN -EROFS
  // Extract parent and filename
  name ← allocate(MAX_NAME_LEN + 1)
  FS_LOCK(g_fuse)
  parent_inode_num ← path_resolve_parent(g_fuse->fs, path, name)
  FS_UNLOCK(g_fuse)
  IF parent_inode_num = 0:
    free(name)
    RETURN -ENOENT
  // Check if file already exists
  FS_LOCK(g_fuse)
  existing_inode ← dir_lookup(g_fuse->fs, parent_inode_num, name)
  FS_UNLOCK(g_fuse)
  IF existing_inode ≠ 0:
    // File exists
    IF (fi->flags & O_EXCL) ≠ 0:
      free(name)
      RETURN -EEXIST
    // Open existing file
    fi->fh ← existing_inode
    free(name)
    RETURN 0
  // Create new file
  FS_LOCK(g_fuse)
  new_inode_num ← fs_create(g_fuse->fs, path, mode & 07777)
  FS_UNLOCK(g_fuse)
  IF new_inode_num = 0:
    free(name)
    IF errno = ENOSPC:
      RETURN -ENOSPC
    RETURN -EIO
  // Set file handle
  fi->fh ← new_inode_num
  free(name)
  RETURN 0
```

![File Handle Usage](./diagrams/tdd-diag-m5-03.svg)

### fs_fuse_read Algorithm
**Input:** `path`, `buf`, `size`, `offset`, `fi`
**Output:** Bytes read, or -errno on failure
```
ALGORITHM fs_fuse_read(path, buf, size, offset, fi):
  // Get inode number from file handle
  IF fi ≠ NULL AND fi->fh ≠ 0:
    inode_num ← fi->fh
  ELSE:
    // Fallback to path resolution
    FS_LOCK(g_fuse)
    inode_num ← path_resolve(g_fuse->fs, path)
    FS_UNLOCK(g_fuse)
    IF inode_num = 0:
      RETURN -ENOENT
  // Perform read
  FS_LOCK(g_fuse)
  bytes_read ← fs_read(g_fuse->fs, inode_num, offset, buf, size)
  FS_UNLOCK(g_fuse)
  IF bytes_read < 0:
    RETURN -EIO
  RETURN bytes_read
```
**Note on Locking Granularity:**
The current implementation holds the global lock for the entire read operation. For large reads, this blocks all other operations. A finer-grained approach would:
1. Lock only for path resolution
2. Lock only for inode read
3. Release lock during data block reads
4. Re-acquire for atime update
This would improve parallelism but requires careful state management.
### fs_fuse_write Algorithm
**Input:** `path`, `buf`, `size`, `offset`, `fi`
**Output:** Bytes written, or -errno on failure
```
ALGORITHM fs_fuse_write(path, buf, size, offset, fi):
  // Check read-only
  IF g_fuse->read_only:
    RETURN -EROFS
  // Get inode number
  IF fi ≠ NULL AND fi->fh ≠ 0:
    inode_num ← fi->fh
  ELSE:
    FS_LOCK(g_fuse)
    inode_num ← path_resolve(g_fuse->fs, path)
    FS_UNLOCK(g_fuse)
    IF inode_num = 0:
      RETURN -ENOENT
  // Perform write
  FS_LOCK(g_fuse)
  bytes_written ← fs_write(g_fuse->fs, inode_num, offset, buf, size)
  FS_UNLOCK(g_fuse)
  IF bytes_written < 0:
    IF errno = ENOSPC:
      // May have partial write - but we don't support that yet
      RETURN -ENOSPC
    IF errno = EFBIG:
      RETURN -EFBIG
    RETURN -EIO
  RETURN bytes_written
```
### fs_fuse_rename Algorithm
**Input:** `oldpath`, `newpath`, `flags`
**Output:** 0 on success, -errno on failure
```
ALGORITHM fs_fuse_rename(oldpath, newpath, flags):
  // Check read-only
  IF g_fuse->read_only:
    RETURN -EROFS
  // Resolve old path
  old_name ← allocate(MAX_NAME_LEN + 1)
  FS_LOCK(g_fuse)
  old_parent ← path_resolve_parent(g_fuse->fs, oldpath, old_name)
  FS_UNLOCK(g_fuse)
  IF old_parent = 0:
    free(old_name)
    RETURN -ENOENT
  // Get old inode
  FS_LOCK(g_fuse)
  old_inode ← dir_lookup(g_fuse->fs, old_parent, old_name)
  FS_UNLOCK(g_fuse)
  IF old_inode = 0:
    free(old_name)
    RETURN -ENOENT
  // Resolve new path
  new_name ← allocate(MAX_NAME_LEN + 1)
  FS_LOCK(g_fuse)
  new_parent ← path_resolve_parent(g_fuse->fs, newpath, new_name)
  FS_UNLOCK(g_fuse)
  IF new_parent = 0:
    free(old_name)
    free(new_name)
    RETURN -ENOENT
  // Check for existing target
  FS_LOCK(g_fuse)
  existing ← dir_lookup(g_fuse->fs, new_parent, new_name)
  FS_UNLOCK(g_fuse)
  IF existing ≠ 0:
    // Target exists - check if we can overwrite
    // For simplicity, we don't support overwriting
    free(old_name)
    free(new_name)
    RETURN -EEXIST
  // Perform rename (single lock for atomicity)
  FS_LOCK(g_fuse)
  result ← fs_rename(g_fuse->fs, oldpath, newpath)
  FS_UNLOCK(g_fuse)
  free(old_name)
  free(new_name)
  IF result < 0:
    IF errno = ENOENT:
      RETURN -ENOENT
    RETURN -EIO
  RETURN 0
```

![readdir Filler Callback](./diagrams/tdd-diag-m5-04.svg)

### fs_fuse_init Algorithm
**Input:** `conn`, `cfg`
**Output:** Private data (g_fuse)
```
ALGORITHM fs_fuse_init(conn, cfg):
  // Enable FUSE capabilities for better performance
  // Big writes: Allow larger than 4KB write chunks
  conn->want |= FUSE_CAP_BIG_WRITES
  conn->max_write ← FUSE_MAX_WRITE_SIZE  // 128KB
  // Async reads: Allow kernel to issue multiple reads
  conn->want |= FUSE_CAP_ASYNC_READ
  conn->max_read ← FUSE_MAX_READ_SIZE  // 128KB
  // Writeback cache: Buffer writes in kernel
  // (Requires careful handling of concurrent writes)
  conn->want |= FUSE_CAP_WRITEBACK_CACHE
  // Set other options
  conn->time_gran ← 1  // 1 nanosecond granularity
  // Configure behavior
  cfg->kernel_cache ← 1  // Enable kernel caching
  cfg->entry_timeout ← 1.0   // Cache path lookups for 1 second
  cfg->attr_timeout ← 1.0    // Cache attributes for 1 second
  cfg->negative_timeout ← 1.0  // Cache negative lookups
  // Log startup
  IF g_fuse->debug_mode:
    printf("FUSE initialized with capabilities:\n")
    printf("  max_write: %u\n", conn->max_write)
    printf("  max_read: %u\n", conn->max_read)
    printf("  writeback_cache: enabled\n")
  RETURN g_fuse
```
### fs_fuse_destroy Algorithm
**Input:** `private_data`
**Output:** None
```
ALGORITHM fs_fuse_destroy(private_data):
  IF g_fuse = NULL:
    RETURN
  IF g_fuse->debug_mode:
    printf("FUSE destroy: flushing data\n")
  // Flush superblock
  IF g_fuse->fs ≠ NULL AND g_fuse->fs->sb ≠ NULL:
    FS_LOCK(g_fuse)
    g_fuse->fs->sb->write_time ← current_time()
    write_block(g_fuse->fs->dev, 0, g_fuse->fs->sb)
    FS_UNLOCK(g_fuse)
  // Sync disk image to storage
  IF g_fuse->fs ≠ NULL AND g_fuse->fs->dev ≠ NULL:
    fsync(g_fuse->fs->dev->fd)
  IF g_fuse->debug_mode:
    printf("FUSE destroy: complete\n")
    printf("  Operations processed: %lu\n", g_fuse->op_count)
    printf("  Errors: %lu\n", g_fuse->error_count)
```

![Error Code Mapping](./diagrams/tdd-diag-m5-05.svg)

---
## Error Handling Matrix
| Error | errno | Detected By | Recovery | User-Visible Message |
|-------|-------|-------------|----------|---------------------|
| Path not found | ENOENT | path_resolve, dir_lookup | Return -ENOENT immediately | "No such file or directory" |
| Parent not directory | ENOTDIR | path_resolve | Return -ENOTDIR | "Not a directory" |
| Is a directory | EISDIR | fs_read, fs_write, fs_truncate | Return -EISDIR | "Is a directory" |
| Not a directory | ENOTDIR | fs_readdir, fs_mkdir, fs_rmdir | Return -ENOTDIR | "Not a directory" |
| File exists | EEXIST | fs_create (with O_EXCL), fs_mkdir | Return -EEXIST | "File exists" |
| Directory not empty | ENOTEMPTY | fs_rmdir | Return -ENOTEMPTY | "Directory not empty" |
| Permission denied | EACCES | All write ops (check mode), chmod, chown | Return -EACCES | "Permission denied" |
| Read-only filesystem | EROFS | All write ops | Return -EROFS | "Read-only file system" |
| No space | ENOSPC | fs_create, fs_write | Return -ENOSPC | "No space left on device" |
| File too large | EFBIG | fs_write, fs_truncate | Return -EFBIG | "File too large" |
| Name too long | ENAMETOOLONG | path parsing | Return -ENAMETOOLONG | "File name too long" |
| I/O error | EIO | Any block read/write | Return -EIO | "Input/output error" |
| Operation not permitted | EPERM | fs_link (to directory) | Return -EPERM | "Operation not permitted" |
| Device busy | EBUSY | fs_rmdir (root) | Return -EBUSY | "Device or resource busy" |
| Invalid argument | EINVAL | Various | Return -EINVAL | "Invalid argument" |
**Error Return Pattern:**
```c
// FUSE expects NEGATIVE errno values
// Correct:
return -ENOENT;
// Wrong:
return ENOENT;  // This would be interpreted as success!
```
---
## Implementation Sequence with Checkpoints
### Phase 1: FUSE Operations Structure (1-2 hours)
**Files:** `52_fuse_callbacks.h`
**Implementation Steps:**
1. Include fuse3/fuse.h header
2. Define `fs_fuse_oper` structure with all callback pointers
3. Create stub implementations returning -ENOSYS
4. Verify compilation with libfuse3
**Checkpoint 1:**
```bash
$ pkg-config --cflags --libs fuse3
-I/usr/include/fuse3 -lfuse3 -pthread
$ make fuse_callbacks.o
$ nm fuse_callbacks.o | grep fs_fuse
0000000000000000 D fs_fuse_oper
# All callbacks defined, compiles cleanly
```
### Phase 2: getattr Callback (2-3 hours)
**Files:** `53_fuse_getattr.c`
**Implementation Steps:**
1. Implement `fs_fuse_getattr` with path resolution
2. Fill struct stat from inode
3. Implement `fs_fuse_statfs` from superblock
4. Test with `stat` command
**Checkpoint 2:**
```bash
$ ./fs_fuse -f test.img /mnt/test &
$ stat /mnt/test
  File: /mnt/test
  Size: 4096          Blocks: 8          IO Block: 4096   directory
$ stat /mnt/test/nonexistent
stat: cannot stat '/mnt/test/nonexistent': No such file or directory
$ df -h /mnt/test
Filesystem      Size  Used Avail Use% Mounted on
./fs_fuse       100M  256K  100M   1% /mnt/test
# getattr and statfs working correctly
```
### Phase 3: Directory Callbacks (3-4 hours)
**Files:** `54_fuse_dir.c`
**Implementation Steps:**
1. Implement `fs_fuse_mkdir` using fs_mkdir
2. Implement `fs_fuse_rmdir` with empty check
3. Implement `fs_fuse_opendir` (trivial, store inode in fh)
4. Implement `fs_fuse_readdir` with filler function
5. Implement `fs_fuse_releasedir` (cleanup if needed)
**Checkpoint 3:**
```bash
$ mkdir /mnt/test/dir1
$ mkdir /mnt/test/dir1/subdir
$ ls -la /mnt/test
total 2
drwxr-xr-x 2 root root 4096 Mar  4 10:00 .
drwxr-xr-x 3 root root 4096 Mar  4 10:00 ..
drwxr-xr-x 2 root root 4096 Mar  4 10:00 dir1
$ ls /mnt/test/dir1
subdir
$ rmdir /mnt/test/dir1/subdir
$ rmdir /mnt/test/dir1
$ ls /mnt/test
# Only . and .. (empty)
# All directory operations working
```
### Phase 4: File Callbacks (3-4 hours)
**Files:** `55_fuse_file.c`
**Implementation Steps:**
1. Implement `fs_fuse_create` using fs_create
2. Implement `fs_fuse_open` with permission check
3. Implement `fs_fuse_read` using fs_read
4. Implement `fs_fuse_write` using fs_write
5. Implement `fs_fuse_release` (update atime if needed)
**Checkpoint 4:**
```bash
$ echo "Hello, FUSE!" > /mnt/test/hello.txt
$ cat /mnt/test/hello.txt
Hello, FUSE!
$ dd if=/dev/urandom of=/mnt/test/random.bin bs=4096 count=100
100+0 records in
100+0 records out
409600 bytes (410 kB) copied, 0.05 s, 8.2 MB/s
$ md5sum /mnt/test/random.bin
# Matches source
# File read/write working
```
### Phase 5: Metadata Callbacks (2-3 hours)
**Files:** `56_fuse_meta.c`
**Implementation Steps:**
1. Implement `fs_fuse_chmod` using inode update
2. Implement `fs_fuse_chown` (simplified, no permission check)
3. Implement `fs_fuse_truncate` using fs_truncate
4. Implement `fs_fuse_utimens` with UTIME_NOW/OMIT handling
**Checkpoint 5:**
```bash
$ chmod 644 /mnt/test/hello.txt
$ ls -la /mnt/test/hello.txt
-rw-r--r-- 1 root root 13 Mar  4 10:00 /mnt/test/hello.txt
$ truncate -s 1000000 /mnt/test/hello.txt
$ ls -la /mnt/test/hello.txt
-rw-r--r-- 1 root root 1000000 Mar  4 10:00 /mnt/test/hello.txt
$ touch -d "2020-01-01" /mnt/test/hello.txt
$ stat /mnt/test/hello.txt | grep Modify
Modify: 2020-01-01 00:00:00.000000000 +0000
# Metadata operations working
```
### Phase 6: Delete and Rename (2-3 hours)
**Files:** `57_fuse_delete.c`
**Implementation Steps:**
1. Implement `fs_fuse_unlink` using dir_remove_entry
2. Implement `fs_fuse_rename` using fs_rename
3. Handle directory renames with .. update
4. Test atomic rename
**Checkpoint 6:**
```bash
$ touch /mnt/test/file1.txt
$ mv /mnt/test/file1.txt /mnt/test/file2.txt
$ ls /mnt/test
file2.txt
$ rm /mnt/test/file2.txt
$ ls /mnt/test
# Empty (only . and ..)
$ mkdir /mnt/test/dir
$ touch /mnt/test/dir/file.txt
$ mv /mnt/test/dir /mnt/test/newdir
$ ls /mnt/test/newdir
file.txt
# Delete and rename working
```
### Phase 7: Thread-Safe Wrappers (2-3 hours)
**Files:** `59_fuse_lock.h`, `60_fuse_lock.c`
**Implementation Steps:**
1. Add pthread_mutex_t to FuseGlobal
2. Implement FS_LOCK/FS_UNLOCK macros
3. Wrap all callbacks with lock/unlock
4. Test concurrent access
**Checkpoint 7:**
```bash
# Concurrent file creation
$ for i in {1..100}; do touch /mnt/test/file$i.txt & done
$ wait
$ ls /mnt/test | wc -l
100
# No corruption, all files created
# Concurrent read/write
$ dd if=/dev/zero of=/mnt/test/concurrent.bin bs=1M count=10 &
$ dd if=/mnt/test/concurrent.bin of=/dev/null bs=1M &
$ wait
# No errors, no corruption
# Thread safety verified
```
### Phase 8: Main Program (2-3 hours)
**Files:** `50_fuse_main.h`, `51_fuse_main.c`
**Implementation Steps:**
1. Parse arguments (extract disk image from FUSE args)
2. Open filesystem with fs_open
3. Verify superblock
4. Initialize locks
5. Call fuse_main with operations
6. Cleanup on exit
**Checkpoint 8:**
```bash
$ ./fs_fuse --help
Usage: fs_fuse [FUSE options] <disk_image> <mount_point>
FUSE options:
  -f          Foreground (don't daemonize)
  -d          Debug mode
  -s          Single-threaded
  -o allow_other  Allow other users
$ ./fs_fuse -f test.img /mnt/test &
[1] 12345
Opening filesystem: test.img
Filesystem mounted:
  Total blocks: 25600
  Free blocks:  25500
  Free inodes:  1599
Starting FUSE filesystem...
$ fusermount -u /mnt/test
Checkpointing journal...
FUSE destroy: complete
[1]+  Done                    ./fs_fuse -f test.img /mnt/test
# Main program working
```
### Phase 9: Integration Testing (2-3 hours)
**Files:** `tests/test_fuse_real_tools.c`
**Implementation Steps:**
1. Test with `ls -laR` (recursive listing)
2. Test with `cp -r` (copy directory tree)
3. Test with `vim` (edit file)
4. Test with `gcc` (compile source)
5. Test with `git` (clone repository)
6. Stress test with concurrent operations
**Checkpoint 9:**
```bash
# Copy real directory tree
$ cp -r /usr/include /mnt/test/
$ diff -r /usr/include /mnt/test/include
# No differences
# Compile a program
$ cat > /mnt/test/hello.c << 'EOF'
#include <stdio.h>
int main() { printf("Hello, FUSE!\n"); return 0; }
EOF
$ gcc /mnt/test/hello.c -o /mnt/test/hello
$ /mnt/test/hello
Hello, FUSE!
# Git operations
$ cd /mnt/test
$ git init
$ git config user.email "test@test.com"
$ git config user.name "Test"
$ git add .
$ git commit -m "Initial commit"
[master (root-commit) abc1234] Initial commit
$ git log
# Git working on FUSE filesystem
# Stress test
$ stress-ng --fs 4 --timeout 60s --fs-ops 1000
# No errors, filesystem still consistent
$ ./verify test.img
Verification complete: 0 errors
# All integration tests passed
```
{{DIAGRAM:tdd-diag-m5-06}}
---
## Test Specification
### test_fuse_mount.c
```c
void test_mount_unmount() {
    // Create test filesystem
    system("./mkfs test_mount.img 10");
    // Mount
    pid_t pid = fork();
    if (pid == 0) {
        execl("./fs_fuse", "fs_fuse", "-f", "test_mount.img", "/mnt/test_mount", NULL);
        exit(1);
    }
    // Wait for mount
    sleep(1);
    // Verify mount
    ASSERT(access("/mnt/test_mount", F_OK) == 0);
    // Create file
    int fd = open("/mnt/test_mount/test.txt", O_CREAT | O_WRONLY, 0644);
    ASSERT(fd >= 0);
    write(fd, "test", 4);
    close(fd);
    // Unmount
    system("fusermount -u /mnt/test_mount");
    waitpid(pid, NULL, 0);
    // Verify file persisted
    system("./fs_fuse -f test_mount.img /mnt/test_mount &");
    sleep(1);
    char buf[10];
    fd = open("/mnt/test_mount/test.txt", O_RDONLY);
    ASSERT(fd >= 0);
    ASSERT(read(fd, buf, 4) == 4);
    ASSERT(memcmp(buf, "test", 4) == 0);
    close(fd);
    system("fusermount -u /mnt/test_mount");
    unlink("test_mount.img");
}
```
### test_fuse_callbacks.c
```c
void test_getattr_root() {
    struct stat st;
    ASSERT(stat("/mnt/test", &st) == 0);
    ASSERT(S_ISDIR(st.st_mode));
    ASSERT(st.st_ino == 1);
    ASSERT(st.st_nlink >= 2);
}
void test_getattr_nonexistent() {
    struct stat st;
    ASSERT(stat("/mnt/test/nonexistent", &st) == -1);
    ASSERT(errno == ENOENT);
}
void test_readdir_root() {
    DIR* dir = opendir("/mnt/test");
    ASSERT(dir != NULL);
    int count = 0;
    bool has_dot = false, has_dotdot = false;
    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) {
        count++;
        if (strcmp(ent->d_name, ".") == 0) has_dot = true;
        if (strcmp(ent->d_name, "..") == 0) has_dotdot = true;
    }
    closedir(dir);
    ASSERT(has_dot && has_dotdot);
}
void test_create_and_read() {
    const char* data = "Hello, World!";
    // Create file
    int fd = open("/mnt/test/read_test.txt", O_CREAT | O_WRONLY | O_TRUNC, 0644);
    ASSERT(fd >= 0);
    ASSERT(write(fd, data, strlen(data)) == strlen(data));
    close(fd);
    // Read back
    char buf[100];
    fd = open("/mnt/test/read_test.txt", O_RDONLY);
    ASSERT(fd >= 0);
    ASSERT(read(fd, buf, sizeof(buf)) == strlen(data));
    ASSERT(memcmp(buf, data, strlen(data)) == 0);
    close(fd);
}
void test_write_extend() {
    int fd = open("/mnt/test/extend.txt", O_CREAT | O_WRONLY, 0644);
    ASSERT(fd >= 0);
    // Write at offset 1000
    ASSERT(pwrite(fd, "END", 3, 1000) == 3);
    close(fd);
    // Verify size
    struct stat st;
    ASSERT(stat("/mnt/test/extend.txt", &st) == 0);
    ASSERT(st.st_size == 1003);
}
void test_truncate() {
    // Create file with data
    int fd = open("/mnt/test/trunc.txt", O_CREAT | O_WRONLY, 0644);
    write(fd, "1234567890", 10);
    close(fd);
    // Truncate to 5
    ASSERT(truncate("/mnt/test/trunc.txt", 5) == 0);
    // Verify size
    struct stat st;
    ASSERT(stat("/mnt/test/trunc.txt", &st) == 0);
    ASSERT(st.st_size == 5);
    // Verify content
    char buf[10];
    fd = open("/mnt/test/trunc.txt", O_RDONLY);
    ASSERT(read(fd, buf, 5) == 5);
    ASSERT(memcmp(buf, "12345", 5) == 0);
    close(fd);
}
void test_rename() {
    // Create file
    int fd = open("/mnt/test/oldname.txt", O_CREAT | O_WRONLY, 0644);
    write(fd, "data", 4);
    close(fd);
    // Rename
    ASSERT(rename("/mnt/test/oldname.txt", "/mnt/test/newname.txt") == 0);
    // Verify old doesn't exist
    ASSERT(access("/mnt/test/oldname.txt", F_OK) == -1);
    ASSERT(errno == ENOENT);
    // Verify new exists with same content
    char buf[10];
    fd = open("/mnt/test/newname.txt", O_RDONLY);
    ASSERT(read(fd, buf, 4) == 4);
    ASSERT(memcmp(buf, "data", 4) == 0);
    close(fd);
}
```
### test_fuse_concurrent.c
```c
void test_concurrent_create() {
    // 10 processes creating 10 files each
    for (int p = 0; p < 10; p++) {
        if (fork() == 0) {
            for (int i = 0; i < 10; i++) {
                char path[100];
                snprintf(path, sizeof(path), "/mnt/test/proc%d_file%d.txt", p, i);
                int fd = open(path, O_CREAT | O_WRONLY, 0644);
                if (fd >= 0) {
                    write(fd, "x", 1);
                    close(fd);
                }
            }
            exit(0);
        }
    }
    // Wait for all
    for (int p = 0; p < 10; p++) {
        wait(NULL);
    }
    // Verify all 100 files exist
    int count = 0;
    DIR* dir = opendir("/mnt/test");
    struct dirent* ent;
    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] != '.') count++;
    }
    closedir(dir);
    ASSERT(count == 100);
}
void test_concurrent_read_write() {
    // Create file
    int fd = open("/mnt/test/concurrent.txt", O_CREAT | O_WRONLY, 0644);
    char data[1024];
    memset(data, 'A', sizeof(data));
    write(fd, data, sizeof(data));
    close(fd);
    // Reader and writer processes
    pid_t reader = fork();
    if (reader == 0) {
        for (int i = 0; i < 100; i++) {
            char buf[1024];
            int fd = open("/mnt/test/concurrent.txt", O_RDONLY);
            read(fd, buf, sizeof(buf));
            close(fd);
        }
        exit(0);
    }
    pid_t writer = fork();
    if (writer == 0) {
        for (int i = 0; i < 100; i++) {
            int fd = open("/mnt/test/concurrent.txt", O_WRONLY);
            pwrite(fd, "B", 1, i % 1024);
            close(fd);
        }
        exit(0);
    }
    waitpid(reader, NULL, 0);
    waitpid(writer, NULL, 0);
    // Verify file still valid
    struct stat st;
    ASSERT(stat("/mnt/test/concurrent.txt", &st) == 0);
    ASSERT(st.st_size == 1024);
}
```
### test_fuse_real_tools.c
```c
void test_with_ls() {
    system("mkdir -p /mnt/test/ls_test/subdir");
    system("touch /mnt/test/ls_test/file1.txt");
    system("touch /mnt/test/ls_test/subdir/file2.txt");
    // ls -la should work
    ASSERT(system("ls -la /mnt/test/ls_test > /dev/null") == 0);
    ASSERT(system("ls -la /mnt/test/ls_test/subdir > /dev/null") == 0);
    // Recursive ls
    ASSERT(system("ls -laR /mnt/test/ls_test > /dev/null") == 0);
}
void test_with_cat_cp() {
    // Create file with content
    system("echo 'Hello, FUSE World!' > /mnt/test/cat_test.txt");
    // Cat should work
    ASSERT(system("cat /mnt/test/cat_test.txt > /dev/null") == 0);
    // Cp should work
    ASSERT(system("cp /mnt/test/cat_test.txt /mnt/test/cat_copy.txt") == 0);
    // Diff should show no difference
    ASSERT(system("diff /mnt/test/cat_test.txt /mnt/test/cat_copy.txt") == 0);
}
void test_with_vim() {
    // Create file
    system("touch /mnt/test/vim_test.txt");
    // Vim in ex mode (non-interactive)
    ASSERT(system("vim -es '+normal iHello from Vim' '+wq' /mnt/test/vim_test.txt") == 0);
    // Verify content
    char buf[100];
    int fd = open("/mnt/test/vim_test.txt", O_RDONLY);
    read(fd, buf, sizeof(buf));
    close(fd);
    ASSERT(strstr(buf, "Hello from Vim") != NULL);
}
void test_with_gcc() {
    // Create C source
    system("cat > /mnt/test/hello.c << 'EOF'\n"
           "#include <stdio.h>\n"
           "int main() { printf(\"Hello from FUSE!\\n\"); return 0; }\n"
           "EOF");
    // Compile
    ASSERT(system("gcc /mnt/test/hello.c -o /mnt/test/hello") == 0);
    // Run
    ASSERT(system("/mnt/test/hello | grep -q 'Hello from FUSE'") == 0);
}
```
---
## Performance Targets
| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| `fs_fuse_getattr` (cached) | < 100 μs | `time stat /mnt/test/file` (100 iterations) |
| `fs_fuse_getattr` (uncached) | < 10 ms | After dropping caches, single stat |
| `fs_fuse_readdir` (100 entries) | < 10 ms | `time ls -la /mnt/test/big_dir` |
| `fs_fuse_read` (4KB) | < 200 μs | `time dd if=/mnt/test/file of=/dev/null bs=4K count=1` |
| `fs_fuse_read` (1MB sequential) | > 100 MB/s | `time dd if=/mnt/test/big of=/dev/null bs=1M` |
| `fs_fuse_write` (4KB) | < 500 μs | `time dd if=/dev/zero of=/mnt/test/file bs=4K count=1` |
| `fs_fuse_write` (1MB sequential) | > 80 MB/s | `time dd if=/dev/zero of=/mnt/test/file bs=1M` |
| `fs_fuse_create` | < 1 ms | `time touch /mnt/test/newfile` |
| `fs_fuse_unlink` | < 1 ms | `time rm /mnt/test/file` |
| `fs_fuse_rename` | < 1 ms | `time mv /mnt/test/old /mnt/test/new` |
| FUSE overhead vs native | < 6× | Compare dd speeds on FUSE vs ext4 |
**Hardware Soul - FUSE Overhead Analysis:**
```
Native ext4 read (4KB):
  - Syscall: ~100 cycles
  - VFS lookup (cached): ~500 cycles
  - Page cache hit: ~50 cycles
  - Total: ~650 cycles (~0.3 μs)
FUSE read (4KB):
  - Syscall: ~100 cycles
  - VFS → FUSE module: ~200 cycles
  - Context switch to userspace: ~1000 cycles
  - Path resolution: varies (0 if fi->fh valid)
  - fs_read: filesystem overhead
  - Context switch back: ~1000 cycles
  - Total: ~5000+ cycles (~2.5 μs) minimum
Overhead factor: 5000 / 650 ≈ 8×
Optimizations:
  - Use fi->fh to avoid path resolution: saves ~2000 cycles
  - Enable FUSE_CAP_WRITEBACK_CACHE: batches writes
  - Enable FUSE_CAP_SPLICE_MOVE: zero-copy I/O
  - Increase max_write/max_read: fewer round trips
```

![Mount Lifecycle State Machine](./diagrams/tdd-diag-m5-07.svg)

---
## Diagrams

![Concurrent Access Scenario](./diagrams/tdd-diag-m5-08.svg)

{{DIAGRAM:tdd-diag-m5-09}}

![FUSE Performance Overhead](./diagrams/tdd-diag-m5-10.svg)

---
[[CRITERIA_JSON: {"milestone_id": "filesystem-m5", "criteria": ["FUSE daemon mounts the filesystem image as a regular mount point accessible to all OS programs via standard syscalls", "All required FUSE callbacks implemented with correct semantics: getattr, readdir, lookup (via opendir/readdir), create, open, read, write, release, mkdir, rmdir, unlink, rename, truncate, chmod, utimens, statfs", "Standard Unix tools work correctly: ls -la shows directory contents with accurate sizes/permissions/timestamps; cat reads file contents; cp copies files into the mount; echo redirection creates and writes files; vim and other editors can edit files", "Thread-safe concurrent access: multiple processes can simultaneously read/write/create/delete without corrupting filesystem state, using proper mutex locking around all shared structures (bitmaps, inodes, directories)", "Unmount (via fusermount -u or daemon exit) flushes all pending metadata and data to the disk image via fsync, ensuring no data loss and clean remount", "rename operation atomically moves entries between directories, correctly updating source and parent directory link counts and .. entries for directory moves", "Error handling returns proper negative errno values to FUSE (-ENOENT, -EEXIST, -ENOSPC, -EIO, etc.) allowing applications to receive correct error codes", "File handle (fi->fh) correctly stores inode number in open/create and is used in subsequent read/write/release to avoid redundant path resolution"]}] ]
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: filesystem-m6 -->
# Technical Design Specification: Write-Ahead Journaling and Crash Recovery
## Module Charter
This module implements a write-ahead journal (WAL) that guarantees filesystem metadata consistency across crashes. All metadata operations—inode allocation/deallocation, block allocation/deallocation, directory entry modifications, and superblock updates—are logged to a sequential circular buffer before being applied to the primary filesystem structures. On mount after a crash, the journal is scanned for committed transactions, which are replayed idempotently to restore consistency; uncommitted transactions are discarded. The module uses metadata-only journaling (data blocks are written directly without journaling), providing crash safety for filesystem structure while avoiding the 2× write amplification of full data journaling. It does NOT implement full data journaling, checksum-based error correction (only detection), or distributed consensus. The core invariants: every committed transaction's operations are either fully applied or fully not applied (never partial); after recovery, the filesystem's bitmaps match actual allocation state; and the journal's circular buffer wrap-around never overwrites uncommitted transactions.
---
## File Structure
```
filesystem/
├── 61_journal.h         # JournalHeader, JournalEntry, Transaction structs, constants
├── 62_journal.c         # Journal lifecycle, header I/O, buffer management
├── 63_journal_txn.h     # Transaction begin/end/commit/abort declarations
├── 64_journal_txn.c     # Transaction state machine implementation
├── 65_journal_log.h     # journal_log declaration for all entry types
├── 66_journal_log.c     # Entry serialization with checksum calculation
├── 67_journal_apply.h   # apply_entry declarations, idempotency helpers
├── 68_journal_apply.c   # Entry application to filesystem structures
├── 69_journal_recover.h # journal_recover, journal_checkpoint declarations
├── 70_journal_recover.c # Recovery scan, replay, checkpoint implementation
├── 71_journal_wrap.h    # Wrapped filesystem operations declarations
├── 72_journal_wrap.c    # fs_create_journaled, fs_write_journaled, etc.
└── tests/
    ├── test_journal_header.c  # Header read/write/checksum tests
    ├── test_journal_txn.c     # Transaction lifecycle tests
    ├── test_journal_log.c     # Entry logging tests
    ├── test_journal_apply.c   # Entry application tests
    ├── test_journal_recover.c # Recovery replay tests
    └── test_crash_simulation.c # Fork/kill/verify crash tests
```
---
## Complete Data Model
### JournalHeader Structure
The journal header occupies the first block of the journal region and tracks the circular buffer state.
```c
// 61_journal.h
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include "03_superblock.h"
#include "14_inode.h"
#include "22_direntry.h"
#define JOURNAL_MAGIC       0x4A524E4C  // "JRNL" - little-endian: 4C 4E 52 4A
#define JOURNAL_VERSION     1
#define JOURNAL_HEADER_SIZE BLOCK_SIZE
/**
 * Journal header - lives at the first block of journal region.
 * Tracks circular buffer state for crash recovery.
 * 
 * Memory layout (4096 bytes):
 *   0x000-0x003: magic (4 bytes)
 *   0x004-0x007: version (4 bytes)
 *   0x008-0x00F: sequence (8 bytes) - monotonically increasing transaction ID
 *   0x010-0x017: head (8 bytes) - byte offset where next entry will be written
 *   0x018-0x01F: tail (8 bytes) - byte offset of first valid committed transaction
 *   0x020-0x027: committed_txn_count (8 bytes) - number of committed transactions
 *   0x028-0x02F: checkpoint_sequence (8 bytes) - sequence at last checkpoint
 *   0x030-0x037: checksum (8 bytes) - CRC64 of all preceding bytes
 *   0x038-0xFFF: reserved (4024 bytes) - future expansion
 */
typedef struct {
    uint32_t magic;              // 0x00: Must equal JOURNAL_MAGIC
    uint32_t version;            // 0x04: Journal format version
    uint64_t sequence;           // 0x08: Next transaction sequence number
    uint64_t head;               // 0x10: Write position (byte offset from journal start)
    uint64_t tail;               // 0x18: First valid transaction position
    uint64_t committed_txn_count;// 0x20: Total committed transactions in journal
    uint64_t checkpoint_sequence;// 0x28: Sequence number at last checkpoint
    uint64_t checksum;           // 0x30: CRC64 of header (this field treated as 0)
    uint8_t  reserved[4024];     // 0x38: Future expansion, zero-padded
} __attribute__((packed)) JournalHeader;
static_assert(sizeof(JournalHeader) == BLOCK_SIZE, 
              "JournalHeader must be exactly one block");
```
**Byte Offset Table:**
| Offset | Size | Field | Purpose |
|--------|------|-------|---------|
| 0x00 | 4 | magic | Validate this is a journal region |
| 0x04 | 4 | version | Format compatibility check |
| 0x08 | 8 | sequence | Monotonic transaction ID, survives crashes |
| 0x10 | 8 | head | Next write position in circular buffer |
| 0x18 | 8 | tail | Start of oldest committed transaction |
| 0x20 | 8 | committed_txn_count | Number of committed txns in journal |
| 0x28 | 8 | checkpoint_sequence | Last successfully checkpointed txn |
| 0x30 | 8 | checksum | Detect header corruption |
| 0x38 | 4024 | reserved | Future expansion |
**Why Each Field Exists:**
- **magic/version**: Validate we're reading a journal, not garbage or wrong version
- **sequence**: Enables replay ordering and duplicate detection
- **head/tail**: Circular buffer management; tail advances on checkpoint, head on write
- **committed_txn_count**: Quick check for recovery need (0 = clean)
- **checkpoint_sequence**: Know which transactions have been applied to main filesystem
### JournalEntry Structure
Each journal entry is a variable-length record describing a single operation.
```c
// 61_journal.h
// Journal entry types
#define JE_INVALID          0   // Invalid/unused entry
#define JE_INODE_ALLOC      1   // Allocate an inode
#define JE_INODE_FREE       2   // Free an inode
#define JE_INODE_UPDATE     3   // Update inode metadata
#define JE_BLOCK_ALLOC      4   // Allocate a data block
#define JE_BLOCK_FREE       5   // Free a data block
#define JE_DIR_ADD          6   // Add directory entry
#define JE_DIR_REMOVE       7   // Remove directory entry
#define JE_SB_UPDATE        8   // Update superblock
#define JE_COMMIT           9   // Transaction commit marker
#define JE_ABORT           10   // Transaction abort marker
#define JE_PADDING         11   // Padding to block boundary
/**
 * Journal entry header.
 * Variable-length: header + type-specific data.
 * 
 * Memory layout:
 *   0x00-0x01: type (2 bytes)
 *   0x02-0x03: flags (2 bytes)
 *   0x04-0x07: length (4 bytes) - length of data following this header
 *   0x08-0x0F: sequence (8 bytes) - transaction this entry belongs to
 *   0x10-0x13: checksum (4 bytes) - CRC32 of header + data
 *   0x14+: data (variable) - type-specific payload
 */
typedef struct {
    uint16_t type;        // 0x00: JE_* constant
    uint16_t flags;       // 0x02: Entry flags (reserved, currently 0)
    uint32_t length;      // 0x04: Length of data payload
    uint64_t sequence;    // 0x08: Transaction sequence number
    uint32_t checksum;    // 0x10: CRC32 of this header + data
    uint8_t  data[];      // 0x14: Variable-length type-specific data
} __attribute__((packed)) JournalEntry;
#define JOURNAL_ENTRY_HEADER_SIZE 20  // sizeof(JournalEntry) without data
```
**Entry Data Payloads by Type:**
```c
// JE_INODE_ALLOC data (8 bytes)
typedef struct {
    uint64_t inode_num;   // Inode number allocated
} JEInodeAllocData;
// JE_INODE_FREE data (8 bytes)
typedef struct {
    uint64_t inode_num;   // Inode number freed
} JEInodeFreeData;
// JE_INODE_UPDATE data (4104 bytes)
typedef struct {
    uint64_t inode_num;   // Inode being updated
    Inode    inode;       // Full inode content (4096 bytes)
} JEInodeUpdateData;
// JE_BLOCK_ALLOC data (8 bytes)
typedef struct {
    uint64_t block_num;   // Block number allocated
} JEBlockAllocData;
// JE_BLOCK_FREE data (8 bytes)
typedef struct {
    uint64_t block_num;   // Block number freed
} JEBlockFreeData;
// JE_DIR_ADD data (variable, ~280 bytes max)
typedef struct {
    uint64_t dir_inode;   // Directory inode number
    uint64_t target_inode;// Target inode number
    uint8_t  name_len;    // Length of name
    uint8_t  file_type;   // DT_* constant
    char     name[255];   // Entry name (null-padded)
} JEDirAddData;
// JE_DIR_REMOVE data (variable, ~280 bytes max)
typedef struct {
    uint64_t dir_inode;   // Directory inode number
    uint8_t  name_len;    // Length of name
    char     name[255];   // Entry name to remove
} JEDirRemoveData;
// JE_SB_UPDATE data (4096 bytes)
typedef struct {
    Superblock sb;        // Full superblock content
} JESbUpdateData;
// JE_COMMIT data (0 bytes - header only)
// JE_ABORT data (0 bytes - header only)
// JE_PADDING data (variable - zeros to fill block)
```
**Hardware Soul - Entry Layout Analysis:**
```
JE_INODE_UPDATE entry (largest):
  Header: 20 bytes
  Data: 8 (inode_num) + 4096 (Inode) = 4104 bytes
  Total: 4124 bytes (spans 2 blocks if misaligned)
Entry alignment requirement:
  - Each entry must start at a valid offset
  - If entry would cross block boundary, pad with JE_PADDING
  - Padding entry: header (20 bytes) + zero fill to block end
Journal write pattern:
  - Sequential writes (optimal for HDD seek, SSD page programming)
  - Each transaction: multiple entries written sequentially
  - Commit entry written last, then fsync
  - Wrap-around: may break sequentiality, but rare with adequate journal size
```
### Transaction Structure
Tracks an in-progress transaction.
```c
// 61_journal.h
typedef enum {
    TXN_INACTIVE = 0,    // No active transaction
    TXN_ACTIVE,          // Transaction started, entries being logged
    TXN_COMMITTING,      // Writing commit record
    TXN_COMMITTED,       // Commit record written, not yet applied
    TXN_APPLYING,        // Applying entries to filesystem
    TXN_APPLIED,         // All entries applied successfully
    TXN_ABORTED          // Transaction was aborted
} TransactionState;
/**
 * Active transaction tracking.
 * One transaction active at a time (single-threaded journal).
 */
typedef struct {
    TransactionState state;
    uint64_t sequence;       // This transaction's sequence number
    uint64_t start_offset;   // Where this txn starts in journal (byte offset)
    uint32_t entry_count;    // Number of entries logged
    uint32_t total_size;     // Total bytes written (for buffer management)
    // Linked list of logged entries (in-memory only)
    JournalEntry* first_entry;
    JournalEntry* last_entry;
} Transaction;
```
### Journal Structure
Global journal state.
```c
// 61_journal.h
/**
 * Journal instance.
 * Manages the circular buffer and active transaction.
 */
typedef struct {
    BlockDevice* dev;            // Block device containing the journal
    uint64_t journal_start;      // First block of journal region
    uint64_t journal_blocks;     // Number of blocks in journal
    uint64_t journal_size;       // Total journal size in bytes
    // In-memory header (synced to disk on changes)
    JournalHeader header;
    bool header_dirty;           // Header needs write to disk
    // Active transaction
    Transaction* active_txn;
    // Buffers for I/O
    void* block_buffer;          // BLOCK_SIZE bytes for block reads/writes
    // Locking
    pthread_mutex_t lock;
    // Statistics
    uint64_t txns_committed;
    uint64_t txns_aborted;
    uint64_t entries_logged;
} Journal;
// Calculate available space in journal (bytes)
static inline uint64_t journal_available_space(const Journal* j) {
    if (j->header.head >= j->header.tail) {
        // Simple case: head ahead of tail
        return j->journal_size - (j->header.head - j->header.tail) - BLOCK_SIZE;
    } else {
        // Wrapped: head behind tail
        return j->header.tail - j->header.head - BLOCK_SIZE;
    }
}
```
**Circular Buffer Visualization:**
```
Journal region (simplified, showing byte offsets):
Initial state (empty journal):
  tail = head = BLOCK_SIZE (after header)
  [HEADER][..........................................]
           ^tail ^head
After 3 transactions (no wrap):
  tail = BLOCK_SIZE, head = 50000
  [HEADER][TTTTTTTTTTTTTTTTT.........................]
           ^tail                 ^head
           |-- committed transactions --|
After many transactions (wrapped):
  tail = 100000, head = 50000
  [HEADER]..........[TTTTTTTTTTTTTTTTT][TTTTTTTTTTTTT]
                   ^head              ^tail
                   |<-- new writes    |-- old data to checkpoint
Head catching up to tail (need checkpoint):
  tail = 95000, head = 94000
  [HEADER][TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT..]
                                    ^head ^tail
                                    (only 1000 bytes free!)
```

![Journal Region Layout](./diagrams/tdd-diag-m6-01.svg)

---
## Interface Contracts
### Journal Lifecycle
```c
// 62_journal.c
/**
 * Initialize a journal from an existing journal region.
 * Called during filesystem mount.
 *
 * @param dev            Block device containing the journal
 * @param journal_start  First block of journal region
 * @param journal_blocks Number of blocks in journal
 * @return Journal* on success, NULL on failure
 *
 * Reads header from disk, validates magic and checksum.
 * Does NOT perform recovery (call journal_recover separately).
 *
 * Errors (in errno):
 *   EINVAL - Invalid journal_start or journal_blocks
 *   EIO    - Failed to read journal header
 *   EINVAL - Magic number mismatch
 *   EINVAL - Checksum mismatch
 */
Journal* journal_open(BlockDevice* dev, uint64_t journal_start, 
                      uint64_t journal_blocks);
/**
 * Initialize a new journal region (called by mkfs).
 *
 * @param dev            Block device
 * @param journal_start  First block of journal region
 * @param journal_blocks Number of blocks for journal
 * @return 0 on success, -1 on failure
 *
 * Writes zeroed journal region with valid header.
 * All journal blocks are initialized to zero.
 */
int journal_init(BlockDevice* dev, uint64_t journal_start, 
                 uint64_t journal_blocks);
/**
 * Close a journal, flushing header if dirty.
 *
 * @param j Journal to close (may be NULL)
 */
void journal_close(Journal* j);
```
### Header I/O
```c
// 62_journal.c
/**
 * Read journal header from disk.
 *
 * @param j Journal instance
 * @return 0 on success, -1 on failure
 *
 * Validates magic and checksum after reading.
 */
int journal_read_header(Journal* j);
/**
 * Write journal header to disk.
 *
 * @param j Journal instance
 * @return 0 on success, -1 on failure
 *
 * Calculates checksum before writing.
 * Uses write_block (which does its own fsync).
 */
int journal_write_header(Journal* j);
/**
 * Calculate CRC64 checksum of journal header.
 *
 * @param header Header to checksum (checksum field treated as 0)
 * @return CRC64 value
 */
uint64_t journal_header_checksum(const JournalHeader* header);
```
### Transaction Lifecycle
```c
// 63_journal_txn.h
/**
 * Begin a new transaction.
 *
 * @param j Journal instance
 * @return Transaction* on success, NULL on failure
 *
 * Only one transaction can be active at a time.
 * Allocates Transaction structure and assigns sequence number.
 *
 * Errors (in errno):
 *   EBUSY - A transaction is already active
 *   ENOSPC - Journal is full (need checkpoint)
 */
Transaction* journal_begin(Journal* j);
/**
 * Commit the active transaction.
 *
 * @param j Journal instance
 * @return 0 on success, -1 on failure
 *
 * Writes commit entry to journal, then fsyncs.
 * After commit, the transaction can be applied.
 *
 * This is the CRITICAL durability point. After this returns,
 * the transaction will survive a crash.
 */
int journal_commit(Journal* j);
/**
 * Abort the active transaction.
 *
 * @param j Journal instance
 * @return 0 on success, -1 on failure
 *
 * Writes abort entry to journal (optional, for debugging).
 * Discards all logged entries without applying.
 */
int journal_abort(Journal* j);
/**
 * End the active transaction (after apply or abort).
 *
 * @param j Journal instance
 *
 * Frees transaction resources, updates journal header.
 * Called automatically by journal_apply if successful.
 */
void journal_end(Journal* j);
/**
 * Apply the committed transaction to the filesystem.
 *
 * @param j  Journal instance
 * @param fs FileSystem instance
 * @return 0 on success, -1 on failure
 *
 * Replays each logged entry by calling apply_entry.
 * Entries must be idempotent (safe to apply multiple times).
 */
int journal_apply(Journal* j, FileSystem* fs);
```
### Entry Logging
```c
// 65_journal_log.h
/**
 * Log an operation to the active transaction.
 *
 * @param j      Journal instance (must have active transaction)
 * @param type   Entry type (JE_* constant)
 * @param data   Entry-specific data
 * @param length Length of data
 * @return 0 on success, -1 on failure
 *
 * Calculates checksum, creates JournalEntry, adds to transaction.
 * Entry is in-memory only until commit.
 *
 * Errors (in errno):
 *   EINVAL - No active transaction
 *   EINVAL - Invalid type
 *   ENOSPC - Journal would overflow (shouldn't happen with checkpoint)
 */
int journal_log(Journal* j, uint16_t type, const void* data, uint32_t length);
// Type-specific logging helpers
int journal_log_inode_alloc(Journal* j, uint64_t inode_num);
int journal_log_inode_free(Journal* j, uint64_t inode_num);
int journal_log_inode_update(Journal* j, uint64_t inode_num, const Inode* inode);
int journal_log_block_alloc(Journal* j, uint64_t block_num);
int journal_log_block_free(Journal* j, uint64_t block_num);
int journal_log_dir_add(Journal* j, uint64_t dir_inode, 
                        const char* name, uint64_t target_inode, uint8_t file_type);
int journal_log_dir_remove(Journal* j, uint64_t dir_inode, const char* name);
int journal_log_sb_update(Journal* j, const Superblock* sb);
```
### Entry Application
```c
// 67_journal_apply.h
/**
 * Apply a single journal entry to the filesystem.
 *
 * @param fs    FileSystem instance
 * @param entry Journal entry to apply
 * @return 0 on success, -1 on failure
 *
 * CRITICAL: Operations MUST be idempotent!
 * An entry may be applied multiple times (once during normal
 * operation, again during recovery). The result must be the same.
 *
 * Example: Setting a bit in the bitmap is idempotent.
 *          Toggling a bit is NOT idempotent.
 */
int apply_entry(FileSystem* fs, const JournalEntry* entry);
/**
 * Apply all entries for a specific transaction.
 *
 * @param j        Journal instance
 * @param fs       FileSystem instance
 * @param sequence Transaction sequence number
 * @return 0 on success, -1 on failure
 */
int apply_transaction(Journal* j, FileSystem* fs, uint64_t sequence);
```
### Recovery and Checkpoint
```c
// 69_journal_recover.h
/**
 * Recover the filesystem by replaying the journal.
 *
 * @param j  Journal instance
 * @param fs FileSystem instance
 * @return Number of transactions replayed, or -1 on failure
 *
 * Algorithm:
 *   1. Read journal header, validate
 *   2. Scan from tail to head
 *   3. For each committed transaction (has JE_COMMIT):
 *      - Replay all entries
 *   4. For each uncommitted transaction (no JE_COMMIT):
 *      - Discard (log warning)
 *   5. Call journal_checkpoint to clear
 *
 * Called during filesystem mount.
 */
int journal_recover(Journal* j, FileSystem* fs);
/**
 * Checkpoint the journal.
 *
 * @param j Journal instance
 * @return 0 on success, -1 on failure
 *
 * After all committed transactions are applied to the filesystem,
 * the journal can be cleared. This:
 *   1. Sets tail = head
 *   2. Increments checkpoint_sequence
 *   3. Writes header
 *   4. Zeros the journal region (optional, for security)
 *
 * Called automatically by journal_recover after replay.
 * Should be called periodically during normal operation.
 */
int journal_checkpoint(Journal* j);
/**
 * Scan the journal for transactions.
 *
 * @param j           Journal instance
 * @param callback    Function called for each transaction found
 * @param user_data   Passed to callback
 * @return Number of transactions scanned, or -1 on error
 *
 * Callback receives: sequence number, committed (bool), user_data
 */
typedef void (*journal_scan_callback)(uint64_t sequence, bool committed, 
                                       void* user_data);
int journal_scan(Journal* j, journal_scan_callback callback, void* user_data);
```
### Wrapped Filesystem Operations
```c
// 71_journal_wrap.h
/**
 * Create a file with journaling.
 *
 * @param fs    FileSystem instance (must have journal)
 * @param path  Path for new file
 * @param mode  Permission bits
 * @return New inode number, or 0 on failure
 *
 * Wrapped in a transaction:
 *   1. journal_begin()
 *   2. journal_log_inode_alloc()
 *   3. journal_log_inode_update()
 *   4. journal_log_dir_add()
 *   5. journal_log_sb_update()
 *   6. journal_commit()
 *   7. journal_apply()
 *   8. journal_end()
 */
uint64_t fs_create_journaled(FileSystem* fs, const char* path, uint16_t mode);
/**
 * Write to a file with journaling.
 *
 * @param fs        FileSystem instance
 * @param inode_num Inode to write
 * @param offset    Byte offset
 * @param data      Data to write
 * @param length    Data length
 * @return Bytes written, or -1 on failure
 *
 * Only metadata changes are journaled (inode update, sb update).
 * Data blocks are written directly.
 */
ssize_t fs_write_journaled(FileSystem* fs, uint64_t inode_num, 
                           uint64_t offset, const void* data, size_t length);
/**
 * Create a directory with journaling.
 *
 * @param fs    FileSystem instance
 * @param path  Directory path
 * @return New inode number, or 0 on failure
 */
uint64_t fs_mkdir_journaled(FileSystem* fs, const char* path);
/**
 * Remove a directory with journaling.
 *
 * @param fs    FileSystem instance
 * @param path  Directory path
 * @return 0 on success, -1 on failure
 */
int fs_rmdir_journaled(FileSystem* fs, const char* path);
/**
 * Unlink a file with journaling.
 *
 * @param fs    FileSystem instance
 * @param path  File path
 * @return 0 on success, -1 on failure
 */
int fs_unlink_journaled(FileSystem* fs, const char* path);
/**
 * Rename with journaling.
 *
 * @param fs       FileSystem instance
 * @param oldpath  Current path
 * @param newpath  New path
 * @return 0 on success, -1 on failure
 */
int fs_rename_journaled(FileSystem* fs, const char* oldpath, const char* newpath);
```
---
## Algorithm Specification
### journal_begin Algorithm
**Input:** Journal `j`
**Output:** Transaction pointer, or NULL on failure
**Postcondition:** j->active_txn is set, j->header.sequence incremented
```
ALGORITHM journal_begin(j):
  // Check for existing transaction
  IF j->active_txn ≠ NULL:
    errno ← EBUSY
    RETURN NULL
  // Check for journal space (estimate)
  // A typical transaction is ~5000 bytes
  IF journal_available_space(j) < 10000:
    // Try checkpoint first
    IF journal_checkpoint(j) < 0:
      errno ← ENOSPC
      RETURN NULL
    IF journal_available_space(j) < 10000:
      errno ← ENOSPC
      RETURN NULL
  // Allocate transaction structure
  txn ← malloc(sizeof(Transaction))
  IF txn = NULL:
    errno ← ENOMEM
    RETURN NULL
  // Initialize transaction
  memset(txn, 0, sizeof(Transaction))
  txn->state ← TXN_ACTIVE
  txn->sequence ← j->header.sequence
  txn->start_offset ← j->header.head
  txn->entry_count ← 0
  txn->total_size ← 0
  txn->first_entry ← NULL
  txn->last_entry ← NULL
  // Increment sequence for next transaction
  j->header.sequence ← j->header.sequence + 1
  j->header_dirty ← true
  // Set active transaction
  j->active_txn ← txn
  RETURN txn
```
**Hardware Soul - Sequence Number:**
```
Sequence numbers are 64-bit, starting at 1.
At 1 million transactions per second:
  2^64 / 10^6 / 86400 / 365 ≈ 584,000 years to overflow
Sequence stored in header, survives crashes.
Used for:
  - Detecting incomplete transactions during recovery
  - Ensuring replay order matches commit order
  - Checkpoint tracking
```
### journal_log Algorithm
**Input:** Journal `j`, type, data, length
**Output:** 0 on success, -1 on failure
**Postcondition:** Entry added to transaction's entry list
```
ALGORITHM journal_log(j, type, data, length):
  // Validate
  IF j->active_txn = NULL:
    errno ← EINVAL
    RETURN -1
  IF type = JE_INVALID OR type > JE_PADDING:
    errno ← EINVAL
    RETURN -1
  txn ← j->active_txn
  // Calculate entry size
  entry_size ← JOURNAL_ENTRY_HEADER_SIZE + length
  // Allocate entry
  entry ← malloc(entry_size)
  IF entry = NULL:
    errno ← ENOMEM
    RETURN -1
  // Fill header
  entry->type ← type
  entry->flags ← 0
  entry->length ← length
  entry->sequence ← txn->sequence
  // Copy data
  IF length > 0 AND data ≠ NULL:
    memcpy(entry->data, data, length)
  // Calculate checksum (header + data)
  entry->checksum ← crc32(0, entry, entry_size)
  // Add to transaction's linked list
  IF txn->first_entry = NULL:
    txn->first_entry ← entry
    txn->last_entry ← entry
  ELSE:
    // Link previous last entry to this one
    // We store the "next" pointer in the last 8 bytes of entry->data
    // This is safe because entry->data has variable length
    // (For simplicity, we use a separate allocation for the list)
    // Actually, let's use a simpler approach: array of entries
    // Reallocate to append (simpler than linked list for this use case)
    // ... (implementation detail)
    // For this spec, use linked list with next pointer stored separately
    // Add to end of list
    txn->last_entry ← entry
  txn->entry_count ← txn->entry_count + 1
  txn->total_size ← txn->total_size + entry_size
  j->entries_logged ← j->entries_logged + 1
  RETURN 0
```
**Checksum Calculation:**
```c
// CRC32 implementation (standard polynomial 0xEDB88320)
uint32_t crc32(uint32_t crc, const void* data, size_t length) {
    const uint8_t* bytes = (const uint8_t*)data;
    static uint32_t table[256];
    static bool table_init = false;
    if (!table_init) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int j = 0; j < 8; j++) {
                c = (c & 1) ? (0xEDB88320 ^ (c >> 1)) : (c >> 1);
            }
            table[i] = c;
        }
        table_init = true;
    }
    crc = crc ^ 0xFFFFFFFF;
    for (size_t i = 0; i < length; i++) {
        crc = table[(crc ^ bytes[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFF;
}
```
### journal_commit Algorithm
**Input:** Journal `j`
**Output:** 0 on success, -1 on failure
**Postcondition:** Transaction committed to disk (durable)
```
ALGORITHM journal_commit(j):
  IF j->active_txn = NULL:
    errno ← EINVAL
    RETURN -1
  txn ← j->active_txn
  IF txn->state ≠ TXN_ACTIVE:
    errno ← EINVAL
    RETURN -1
  txn->state ← TXN_COMMITTING
  // Calculate total size needed
  total_size ← txn->total_size + JOURNAL_ENTRY_HEADER_SIZE  // + commit entry
  // Check for journal space
  IF journal_available_space(j) < total_size:
    // This shouldn't happen if begin checked, but handle it
    journal_abort(j)
    errno ← ENOSPC
    RETURN -1
  // Write all entries to journal
  entry ← txn->first_entry
  WHILE entry ≠ NULL:
    IF journal_write_entry(j, entry) < 0:
      journal_abort(j)
      RETURN -1
    entry ← entry->next  // (linked list traversal)
  // Write commit entry
  JournalEntry commit_entry
  commit_entry.type ← JE_COMMIT
  commit_entry.flags ← 0
  commit_entry.length ← 0
  commit_entry.sequence ← txn->sequence
  commit_entry.checksum ← crc32(0, &commit_entry, JOURNAL_ENTRY_HEADER_SIZE)
  IF journal_write_entry(j, &commit_entry) < 0:
    journal_abort(j)
    RETURN -1
  // CRITICAL: fsync the journal to disk
  // This ensures the commit record is on stable storage
  IF fsync(j->dev->fd) < 0:
    journal_abort(j)
    errno ← EIO
    RETURN -1
  // Update header
  j->header.committed_txn_count ← j->header.committed_txn_count + 1
  j->header_dirty ← true
  txn->state ← TXN_COMMITTED
  j->txns_committed ← j->txns_committed + 1
  RETURN 0
```
**Hardware Soul - The fsync Barrier:**
```
The fsync after writing the commit entry is THE critical point.
Without fsync:
  - OS buffers the write
  - Crash loses the commit record
  - Recovery discards the transaction (correct but lost work)
With fsync:
  - Write goes to disk write cache
  - Disk acknowledges to OS
  - Crash preserves the commit record
  - Recovery replays the transaction
fsync cost:
  - HDD: ~10ms (rotational latency + seek)
  - SATA SSD: ~1ms (NAND programming)
  - NVMe SSD: ~100μs (multiple queues, no seek)
Each committed transaction costs one fsync.
Batching multiple operations in one transaction amortizes this cost.
```
{{DIAGRAM:tdd-diag-m6-02}}
### journal_write_entry Algorithm
**Input:** Journal `j`, entry to write
**Output:** 0 on success, -1 on failure
```
ALGORITHM journal_write_entry(j, entry):
  entry_size ← JOURNAL_ENTRY_HEADER_SIZE + entry->length
  // Check for wrap-around
  current_block ← j->journal_start + (j->header.head / BLOCK_SIZE)
  offset_in_block ← j->header.head % BLOCK_SIZE
  remaining_in_block ← BLOCK_SIZE - offset_in_block
  IF entry_size > remaining_in_block:
    // Entry would cross block boundary
    IF entry_size > BLOCK_SIZE:
      // Entry is larger than a block (shouldn't happen with our design)
      errno ← E2BIG
      RETURN -1
    // Write padding entry to fill rest of block
    IF remaining_in_block >= JOURNAL_ENTRY_HEADER_SIZE:
      JournalEntry padding
      padding.type ← JE_PADDING
      padding.flags ← 0
      padding.length ← remaining_in_block - JOURNAL_ENTRY_HEADER_SIZE
      padding.sequence ← 0
      padding.checksum ← crc32(0, &padding, JOURNAL_ENTRY_HEADER_SIZE)
      // Read block, write padding, write back
      IF journal_write_entry_to_block(j, current_block, offset_in_block, 
                                       &padding, JOURNAL_ENTRY_HEADER_SIZE) < 0:
        RETURN -1
    // Advance head to next block
    j->header.head ← ((j->header.head / BLOCK_SIZE) + 1) * BLOCK_SIZE
    // Handle wrap-around
    IF j->header.head >= j->journal_size:
      j->header.head ← BLOCK_SIZE  // Start after header
  // Write the entry
  current_block ← j->journal_start + (j->header.head / BLOCK_SIZE)
  offset_in_block ← j->header.head % BLOCK_SIZE
  IF journal_write_entry_to_block(j, current_block, offset_in_block,
                                   entry, entry_size) < 0:
    RETURN -1
  // Advance head
  j->header.head ← j->header.head + entry_size
  // Handle wrap-around
  IF j->header.head >= j->journal_size:
    j->header.head ← BLOCK_SIZE
  RETURN 0
ALGORITHM journal_write_entry_to_block(j, block_num, offset, entry, size):
  // Read block (read-modify-write)
  IF read_block(j->dev, block_num, j->block_buffer) < 0:
    RETURN -1
  // Copy entry into block
  memcpy((char*)j->block_buffer + offset, entry, size)
  // Write block back
  IF write_block(j->dev, block_num, j->block_buffer) < 0:
    RETURN -1
  RETURN 0
```

![Journal Entry Types](./diagrams/tdd-diag-m6-03.svg)

### journal_apply Algorithm
**Input:** Journal `j`, FileSystem `fs`
**Output:** 0 on success, -1 on failure
```
ALGORITHM journal_apply(j, fs):
  IF j->active_txn = NULL OR j->active_txn->state ≠ TXN_COMMITTED:
    errno ← EINVAL
    RETURN -1
  txn ← j->active_txn
  txn->state ← TXN_APPLYING
  // Apply each entry
  entry ← txn->first_entry
  applied_count ← 0
  WHILE entry ≠ NULL:
    IF apply_entry(fs, entry) < 0:
      // Log error but continue - entry may have been partially applied
      // Since entries are idempotent, replay on recovery will complete
      LOG("Warning: failed to apply entry type %d", entry->type)
    applied_count ← applied_count + 1
    entry ← entry->next
  txn->state ← TXN_APPLIED
  // End the transaction (free resources)
  journal_end(j)
  RETURN 0
```
### apply_entry Algorithm
**Input:** FileSystem `fs`, entry to apply
**Output:** 0 on success, -1 on failure
**Invariant:** Operation is idempotent
```
ALGORITHM apply_entry(fs, entry):
  SWITCH entry->type:
    CASE JE_INODE_ALLOC:
      data ← (JEInodeAllocData*)entry->data
      // Mark inode as allocated in bitmap (idempotent: setting a set bit is no-op)
      RETURN inode_bitmap_set(fs, data->inode_num, 1)
    CASE JE_INODE_FREE:
      data ← (JEInodeFreeData*)entry->data
      // Mark inode as free in bitmap
      RETURN inode_bitmap_set(fs, data->inode_num, 0)
    CASE JE_INODE_UPDATE:
      data ← (JEInodeUpdateData*)entry->data
      // Write entire inode (idempotent: overwriting with same data is no-op)
      RETURN write_inode(fs, data->inode_num, &data->inode)
    CASE JE_BLOCK_ALLOC:
      data ← (JEBlockAllocData*)entry->data
      // Mark block as allocated
      RETURN block_bitmap_set(fs, data->block_num, 1)
    CASE JE_BLOCK_FREE:
      data ← (JEBlockFreeData*)entry->data
      // Mark block as free
      RETURN block_bitmap_set(fs, data->block_num, 0)
    CASE JE_DIR_ADD:
      data ← (JEDirAddData*)entry->data
      // Extract name
      char name[256]
      memcpy(name, data->name, data->name_len)
      name[data->name_len] ← '\0'
      // Add entry (may fail if exists - that's ok, means already applied)
      result ← dir_add_entry_internal(fs, data->dir_inode, name,
                                       data->target_inode, data->file_type)
      IF result < 0 AND errno ≠ EEXIST:
        RETURN -1
      RETURN 0
    CASE JE_DIR_REMOVE:
      data ← (JEDirRemoveData*)entry->data
      char name[256]
      memcpy(name, data->name, data->name_len)
      name[data->name_len] ← '\0'
      // Remove entry (may fail if not found - that's ok)
      result ← dir_remove_entry_internal(fs, data->dir_inode, name)
      IF result < 0 AND errno ≠ ENOENT:
        RETURN -1
      RETURN 0
    CASE JE_SB_UPDATE:
      data ← (JESbUpdateData*)entry->data
      // Write superblock
      memcpy(fs->sb, &data->sb, sizeof(Superblock))
      RETURN write_block(fs->dev, 0, fs->sb)
    CASE JE_COMMIT:
    CASE JE_ABORT:
    CASE JE_PADDING:
      // No action needed
      RETURN 0
    DEFAULT:
      LOG("Unknown journal entry type: %d", entry->type)
      RETURN 0  // Ignore unknown entries
```
**Idempotency Verification:**
```
JE_INODE_ALLOC: inode_bitmap_set(n, 1)
  - First apply: bit 0 → 1, returns 0
  - Second apply: bit 1 → 1, returns 0 (no change)
  - ✓ Idempotent
JE_BLOCK_ALLOC: block_bitmap_set(n, 1)
  - Same as above
  - ✓ Idempotent
JE_INODE_UPDATE: write_inode(n, &inode)
  - First apply: writes inode content
  - Second apply: overwrites with same content
  - ✓ Idempotent (assuming same data)
JE_DIR_ADD: dir_add_entry_internal(...)
  - First apply: adds entry
  - Second apply: fails with EEXIST, we return 0
  - ✓ Idempotent (with error handling)
JE_DIR_REMOVE: dir_remove_entry_internal(...)
  - First apply: removes entry
  - Second apply: fails with ENOENT, we return 0
  - ✓ Idempotent (with error handling)
```

![Transaction Lifecycle](./diagrams/tdd-diag-m6-04.svg)

### journal_recover Algorithm
**Input:** Journal `j`, FileSystem `fs`
**Output:** Number of transactions replayed, or -1 on failure
```
ALGORITHM journal_recover(j, fs):
  LOG("Starting journal recovery...")
  // Read and validate header
  IF journal_read_header(j) < 0:
    LOG("Failed to read journal header")
    RETURN -1
  // Check if journal is clean (no committed transactions)
  IF j->header.committed_txn_count = 0:
    LOG("Journal is clean, no recovery needed")
    RETURN 0
  LOG("Journal has %lu committed transactions", j->header.committed_txn_count)
  // Scan journal for transactions
  offset ← j->header.tail
  head ← j->header.head
  transactions_replayed ← 0
  // Build transaction map
  txn_map ← hashmap_create()  // sequence → {committed, entries[]}
  WHILE offset < head:
    // Read entry at offset
    entry ← journal_read_entry_at(j, offset)
    IF entry = NULL:
      LOG("Failed to read entry at offset %lu", offset)
      BREAK
    // Validate checksum
    entry_size ← JOURNAL_ENTRY_HEADER_SIZE + entry->length
    calc_checksum ← crc32(0, entry, entry_size)
    IF calc_checksum ≠ entry->checksum:
      LOG("Checksum mismatch at offset %lu, stopping scan", offset)
      free(entry)
      BREAK
    seq ← entry->sequence
    // Add entry to transaction map
    IF entry->type = JE_COMMIT:
      txn_map[seq].committed ← true
    ELSE IF entry->type = JE_ABORT:
      txn_map[seq].committed ← false
    ELSE IF entry->type ≠ JE_PADDING:
      txn_map[seq].entries.append(entry)
    offset ← offset + entry_size
    free(entry)
  // Replay committed transactions
  FOR each (seq, txn_info) IN txn_map ORDERED BY seq:
    IF txn_info.committed:
      LOG("Replaying committed transaction seq=%lu (%d entries)",
          seq, txn_info.entries.count)
      FOR each entry IN txn_info.entries:
        IF apply_entry(fs, entry) < 0:
          LOG("  Warning: failed to apply entry type %d", entry->type)
      transactions_replayed ← transactions_replayed + 1
    ELSE:
      LOG("Discarding uncommitted/aborted transaction seq=%lu", seq)
  LOG("Journal recovery complete: %d transactions replayed", 
      transactions_replayed)
  // Checkpoint the journal
  IF journal_checkpoint(j) < 0:
    LOG("Warning: checkpoint failed")
  RETURN transactions_replayed
```

![Transaction Commit Sequence](./diagrams/tdd-diag-m6-05.svg)

### journal_checkpoint Algorithm
**Input:** Journal `j`
**Output:** 0 on success, -1 on failure
```
ALGORITHM journal_checkpoint(j):
  LOG("Checkpointing journal...")
  // Update header
  j->header.tail ← j->header.head  // All transactions before head are applied
  j->header.committed_txn_count ← 0
  j->header.checkpoint_sequence ← j->header.sequence - 1
  j->header_dirty ← true
  // Write header
  IF journal_write_header(j) < 0:
    LOG("Failed to write journal header during checkpoint")
    RETURN -1
  // Optionally zero the journal region (security/cleanliness)
  // This is optional but helps with debugging
  memset(j->block_buffer, 0, BLOCK_SIZE)
  FOR block_idx FROM 1 TO j->journal_blocks - 1:
    write_block(j->dev, j->journal_start + block_idx, j->block_buffer)
  // Sync to disk
  fsync(j->dev->fd)
  LOG("Checkpoint complete: journal cleared")
  RETURN 0
```
### fs_create_journaled Algorithm
**Input:** FileSystem `fs`, path, mode
**Output:** New inode number, or 0 on failure
```
ALGORITHM fs_create_journaled(fs, path, mode):
  j ← fs->journal
  // Begin transaction
  txn ← journal_begin(j)
  IF txn = NULL:
    RETURN 0
  // Resolve parent directory
  name ← extract_filename(path)
  parent_inode ← path_resolve_parent(fs, path, name)
  IF parent_inode = 0:
    journal_abort(j)
    RETURN 0
  // Check for existing file
  IF dir_lookup(fs, parent_inode, name) ≠ 0:
    errno ← EEXIST
    journal_abort(j)
    RETURN 0
  // Allocate inode
  new_inode ← inode_bitmap_alloc(fs)
  IF new_inode = 0:
    journal_abort(j)
    errno ← ENOSPC
    RETURN 0
  // LOG: inode allocation
  journal_log_inode_alloc(j, new_inode)
  // Initialize inode
  Inode inode
  inode_init(&inode, S_IFREG | (mode & 07777), 0, 0)
  // LOG: inode update
  journal_log_inode_update(j, new_inode, &inode)
  // LOG: directory entry add
  journal_log_dir_add(j, parent_inode, name, new_inode, DT_REG)
  // LOG: superblock update
  journal_log_sb_update(j, fs->sb)
  // COMMIT
  IF journal_commit(j) < 0:
    journal_abort(j)
    RETURN 0
  // APPLY: Now actually modify filesystem
  inode_bitmap_set(fs, new_inode, 1)
  write_inode(fs, new_inode, &inode)
  dir_add_entry_internal(fs, parent_inode, name, new_inode, DT_REG)
  write_block(fs->dev, 0, fs->sb)
  // END transaction
  journal_end(j)
  RETURN new_inode
```
**The Critical Order:**
```
1. LOG all operations
2. COMMIT (with fsync)
3. APPLY to filesystem
If crash before COMMIT:
  - Recovery finds no commit record
  - Transaction discarded
  - Filesystem unchanged
If crash after COMMIT but before APPLY:
  - Recovery finds commit record
  - Transaction replayed
  - Filesystem reaches correct state
If crash during APPLY:
  - Some operations may have been applied
  - Recovery replays ALL operations
  - Idempotency ensures correct final state
```

![Journal Write-Ahead Protocol](./diagrams/tdd-diag-m6-06.svg)

---
## Error Handling Matrix
| Error | errno | Detected By | Recovery | User-Visible Message |
|-------|-------|-------------|----------|---------------------|
| Journal full | ENOSPC | journal_begin | Try checkpoint, then fail | "Journal full, try again" |
| Transaction already active | EBUSY | journal_begin | Return NULL | Internal error (log) |
| Invalid journal header | EINVAL | journal_read_header | Fail mount | "Journal corrupted, run fsck" |
| Checksum mismatch (header) | EINVAL | journal_read_header | Fail mount | "Journal corrupted, run fsck" |
| Checksum mismatch (entry) | EINVAL | journal_recover | Stop scan, replay what we have | "Partial journal recovery" |
| Write error during commit | EIO | journal_commit | Abort transaction | "I/O error, transaction lost" |
| fsync error | EIO | journal_commit | Abort transaction | "I/O error, transaction lost" |
| Apply entry failed | (varies) | apply_entry | Log warning, continue | "Warning: recovery incomplete" |
| Checkpoint failed | EIO | journal_checkpoint | Log warning, continue | "Checkpoint failed" |
| No active transaction | EINVAL | journal_log, commit | Return -1 | Internal error (log) |
| Entry too large | E2BIG | journal_write_entry | Abort transaction | Internal error (log) |
**Crash Scenarios:**
| Crash Point | Recovery Result | Filesystem State |
|-------------|-----------------|------------------|
| Before journal_begin | No transaction | Unchanged |
| After journal_begin, before commit | No commit record → discard | Unchanged |
| During commit (partial commit record) | Checksum fails → discard | Unchanged |
| After commit, before apply | Replay transaction | Correct |
| During apply | Replay transaction (idempotent) | Correct |
---
## Implementation Sequence with Checkpoints
### Phase 1: Journal Structures (2-3 hours)
**Files:** `61_journal.h`
**Implementation Steps:**
1. Define JournalHeader with all fields
2. Define JournalEntry with header and data union
3. Define Transaction and Journal structures
4. Add static_assert for sizes
5. Define all JE_* constants
**Checkpoint 1:**
```bash
$ make journal.o
$ nm journal.o | grep "T journal"
# No functions yet, just struct definitions
$ ./test_journal_structs
sizeof(JournalHeader) = 4096: OK
sizeof(JournalEntry) = 20 (header only): OK
offsetof(JournalHeader, magic) = 0: OK
offsetof(JournalHeader, checksum) = 48: OK
All struct layout tests passed!
```
### Phase 2: Journal Header I/O (1-2 hours)
**Files:** `62_journal.c`
**Implementation Steps:**
1. Implement CRC64 checksum function
2. Implement journal_read_header with validation
3. Implement journal_write_header with checksum calculation
4. Implement journal_open and journal_close
5. Implement journal_init for new filesystems
**Checkpoint 2:**
```bash
$ make test_journal_header
$ ./test_journal_header
Creating new journal: OK
Reading header: OK
  magic: 0x4A524E4B (correct): OK
  version: 1: OK
  sequence: 1: OK
  checksum: valid: OK
Corrupting magic: validation fails: OK
Corrupting checksum: validation fails: OK
All header tests passed!
```
### Phase 3: Transaction Begin/End (2-3 hours)
**Files:** `63_journal_txn.h`, `64_journal_txn.c`
**Implementation Steps:**
1. Implement journal_begin with transaction allocation
2. Implement journal_end with resource cleanup
3. Implement sequence number management
4. Handle journal full condition
**Checkpoint 3:**
```bash
$ make test_journal_txn
$ ./test_journal_txn
Begin transaction: seq=1: OK
Begin second transaction: fails with EBUSY: OK
End transaction: OK
Begin after end: seq=2: OK
Fill journal (100 small transactions): OK
Next begin: fails with ENOSPC: OK
Checkpoint: OK
Begin after checkpoint: seq=102: OK
All transaction lifecycle tests passed!
```
### Phase 4: Operation Logging (3-4 hours)
**Files:** `65_journal_log.h`, `66_journal_log.c`
**Implementation Steps:**
1. Implement CRC32 checksum function
2. Implement journal_log generic function
3. Implement type-specific helpers (inode_alloc, block_alloc, etc.)
4. Implement entry linked list management
5. Calculate and store checksums
**Checkpoint 4:**
```bash
$ make test_journal_log
$ ./test_journal_log
Log inode_alloc(42): entry_size=28, checksum valid: OK
Log inode_update(42, ...): entry_size=4124, checksum valid: OK
Log dir_add(1, "test", 43): entry_size=288, checksum valid: OK
Log 100 entries: total_size=50000: OK
Verify checksums: all valid: OK
All logging tests passed!
```
### Phase 5: Transaction Commit (2-3 hours)
**Files:** Update `64_journal_txn.c`
**Implementation Steps:**
1. Implement journal_write_entry with block boundary handling
2. Implement padding entry generation
3. Implement journal_commit with commit record
4. Implement fsync after commit
5. Update header on commit
**Checkpoint 5:**
```bash
$ make test_journal_commit
$ ./test_journal_commit
Begin, log 10 entries, commit: OK
Verify commit entry in journal: OK
Verify fsync was called: OK (check with strace)
Read journal entries back: 10 + commit: OK
All commit tests passed!
```
### Phase 6: Entry Application (3-4 hours)
**Files:** `67_journal_apply.h`, `68_journal_apply.c`
**Implementation Steps:**
1. Implement apply_entry for each entry type
2. Implement idempotent bitmap operations
3. Implement idempotent inode writes
4. Implement idempotent directory operations
5. Test idempotency (apply twice)
**Checkpoint 6:**
```bash
$ make test_journal_apply
$ ./test_journal_apply
Apply inode_alloc(50): bitmap bit set: OK
Apply again (idempotent): bitmap unchanged: OK
Apply inode_update(50, ...): inode written: OK
Apply again: same content: OK
Apply dir_add(1, "test", 50): entry added: OK
Apply again: EEXIST handled: OK
All application tests passed!
```
### Phase 7: Journal Replay (3-4 hours)
**Files:** `69_journal_recover.h`, `70_journal_recover.c`
**Implementation Steps:**
1. Implement journal_read_entry_at
2. Implement journal_scan
3. Implement journal_recover with transaction map
4. Handle committed vs uncommitted transactions
5. Test with various crash scenarios
**Checkpoint 7:**
```bash
$ make test_journal_recover
$ ./test_journal_recover
Create 5 committed transactions: OK
Create 1 uncommitted transaction: OK
Recover: 5 transactions replayed: OK
Verify uncommitted transaction discarded: OK
Verify filesystem state correct: OK
All recovery tests passed!
```
### Phase 8: Checkpoint (1-2 hours)
**Files:** Update `70_journal_recover.c`
**Implementation Steps:**
1. Implement journal_checkpoint
2. Clear journal after checkpoint
3. Update header
4. Test checkpoint clears journal
**Checkpoint 8:**
```bash
$ make test_journal_checkpoint
$ ./test_journal_checkpoint
Create and commit 10 transactions: OK
Checkpoint: OK
Verify journal cleared: OK
Verify header updated: OK
Begin new transaction after checkpoint: OK
All checkpoint tests passed!
```
### Phase 9: Wrapped Filesystem Operations (3-4 hours)
**Files:** `71_journal_wrap.h`, `72_journal_wrap.c`
**Implementation Steps:**
1. Implement fs_create_journaled
2. Implement fs_write_journaled
3. Implement fs_mkdir_journaled
4. Implement fs_rmdir_journaled
5. Implement fs_unlink_journaled
6. Test each wrapped operation
**Checkpoint 9:**
```bash
$ make test_journal_wrap
$ ./test_journal_wrap
fs_create_journaled("/test.txt"): inode 2: OK
Verify journal entries: inode_alloc, inode_update, dir_add, sb_update: OK
fs_write_journaled(2, 0, "Hello", 5): 5 bytes: OK
Verify journal: inode_update, sb_update: OK
fs_mkdir_journaled("/dir"): OK
All wrapped operation tests passed!
```
### Phase 10: Crash Simulation Testing (2-3 hours)
**Files:** `tests/test_crash_simulation.c`
**Implementation Steps:**
1. Implement fork/kill test harness
2. Test crash at various points (before commit, after commit, during apply)
3. Verify recovery produces consistent state
4. Test multiple crash/recover cycles
**Checkpoint 10:**
```bash
$ make test_crash_simulation
$ ./test_crash_simulation
Test: crash before commit
  Fork child, kill mid-log: OK
  Recover: transaction discarded: OK
  Filesystem consistent: OK
Test: crash after commit
  Fork child, kill after fsync: OK
  Recover: transaction replayed: OK
  File exists: OK
Test: crash during apply
  Fork child, kill mid-apply: OK
  Recover: idempotent replay: OK
  Filesystem consistent: OK
Test: 100 crash/recover cycles
  All produced consistent state: OK
All crash simulation tests passed!
```
### Phase 11: FUSE Integration (1-2 hours)
**Files:** Update `51_fuse_main.c`, `60_fuse_lock.c`
**Implementation Steps:**
1. Add journal recovery in fs_fuse_init
2. Add journal checkpoint in fs_fuse_destroy
3. Replace fs_create with fs_create_journaled in callbacks
4. Replace other operations with journaled versions
5. Test with real workloads
**Checkpoint 11:**
```bash
$ ./fs_fuse -f test.img /mnt/test &
Recovering journal...
  2 transactions replayed
Journal recovery complete
Starting FUSE filesystem...
$ echo "test" > /mnt/test/file.txt
$ kill -9 $(pgrep fs_fuse)
$ ./fs_fuse -f test.img /mnt/test &
Recovering journal...
  1 transaction replayed
Journal recovery complete
$ cat /mnt/test/file.txt
test
# Data survived the crash!
$ fusermount -u /mnt/test
Checkpointing journal...
# Clean unmount with checkpoint
```

![Journal Replay Algorithm](./diagrams/tdd-diag-m6-07.svg)

---
## Test Specification
### test_journal_header.c
```c
void test_header_size() {
    ASSERT(sizeof(JournalHeader) == BLOCK_SIZE);
    ASSERT(offsetof(JournalHeader, magic) == 0);
    ASSERT(offsetof(JournalHeader, sequence) == 8);
    ASSERT(offsetof(JournalHeader, head) == 16);
    ASSERT(offsetof(JournalHeader, tail) == 24);
    ASSERT(offsetof(JournalHeader, checksum) == 48);
}
void test_header_checksum() {
    JournalHeader h;
    memset(&h, 0, sizeof(h));
    h.magic = JOURNAL_MAGIC;
    h.version = JOURNAL_VERSION;
    h.sequence = 1;
    h.head = BLOCK_SIZE;
    h.tail = BLOCK_SIZE;
    uint64_t checksum = journal_header_checksum(&h);
    ASSERT(checksum != 0);
    // Modify and verify checksum changes
    h.sequence = 2;
    ASSERT(journal_header_checksum(&h) != checksum);
}
void test_header_read_write() {
    BlockDevice* dev = create_test_device(1000);
    // Initialize journal
    ASSERT(journal_init(dev, 10, 100) == 0);
    // Open and read header
    Journal* j = journal_open(dev, 10, 100);
    ASSERT(j != NULL);
    ASSERT(j->header.magic == JOURNAL_MAGIC);
    ASSERT(j->header.version == JOURNAL_VERSION);
    // Modify and write
    j->header.sequence = 42;
    ASSERT(journal_write_header(j) == 0);
    // Reopen and verify
    journal_close(j);
    j = journal_open(dev, 10, 100);
    ASSERT(j->header.sequence == 42);
    journal_close(j);
    block_device_close(dev);
}
```
### test_journal_txn.c
```c
void test_txn_begin_end() {
    Journal* j = create_test_journal(1000);
    Transaction* txn = journal_begin(j);
    ASSERT(txn != NULL);
    ASSERT(txn->state == TXN_ACTIVE);
    ASSERT(txn->sequence == 1);
    // Can't begin another
    errno = 0;
    ASSERT(journal_begin(j) == NULL);
    ASSERT(errno == EBUSY);
    // End and begin again
    journal_end(j);
    txn = journal_begin(j);
    ASSERT(txn != NULL);
    ASSERT(txn->sequence == 2);
    journal_end(j);
    journal_close(j);
}
void test_txn_journal_full() {
    Journal* j = create_test_journal(10);  // Tiny journal
    // Fill with transactions
    for (int i = 0; i < 100; i++) {
        Transaction* txn = journal_begin(j);
        if (txn == NULL) {
            ASSERT(errno == ENOSPC);
            break;
        }
        // Log a large entry
        char data[4000];
        journal_log(j, JE_INODE_UPDATE, data, sizeof(data));
        journal_commit(j);
        journal_apply(j, create_test_filesystem());
    }
    // Checkpoint should free space
    ASSERT(journal_checkpoint(j) == 0);
    // Now should be able to begin again
    Transaction* txn = journal_begin(j);
    ASSERT(txn != NULL);
    journal_end(j);
    journal_close(j);
}
```
### test_journal_log.c
```c
void test_log_inode_alloc() {
    Journal* j = create_test_journal(100);
    Transaction* txn = journal_begin(j);
    JEInodeAllocData data = {.inode_num = 42};
    ASSERT(journal_log(j, JE_INODE_ALLOC, &data, sizeof(data)) == 0);
    ASSERT(txn->entry_count == 1);
    ASSERT(txn->total_size == JOURNAL_ENTRY_HEADER_SIZE + sizeof(data));
    // Verify entry
    JournalEntry* entry = txn->first_entry;
    ASSERT(entry->type == JE_INODE_ALLOC);
    ASSERT(entry->length == sizeof(data));
    ASSERT(entry->sequence == txn->sequence);
    // Verify checksum
    uint32_t calc = crc32(0, entry, JOURNAL_ENTRY_HEADER_SIZE + entry->length);
    ASSERT(entry->checksum == calc);
    journal_abort(j);
    journal_close(j);
}
void test_log_inode_update() {
    Journal* j = create_test_journal(100);
    Transaction* txn = journal_begin(j);
    JEInodeUpdateData data;
    data.inode_num = 5;
    memset(&data.inode, 'X', sizeof(Inode));
    ASSERT(journal_log(j, JE_INODE_UPDATE, &data, sizeof(data)) == 0);
    ASSERT(txn->total_size == JOURNAL_ENTRY_HEADER_SIZE + sizeof(data));
    journal_abort(j);
    journal_close(j);
}
```
### test_journal_apply.c
```c
void test_apply_inode_alloc_idempotent() {
    FileSystem* fs = create_test_filesystem();
    // Create entry
    JEInodeAllocData data = {.inode_num = 50};
    JournalEntry entry;
    entry.type = JE_INODE_ALLOC;
    entry.length = sizeof(data);
    entry.sequence = 1;
    memcpy(entry.data, &data, sizeof(data));
    entry.checksum = crc32(0, &entry, sizeof(entry.header) + sizeof(data));
    // Apply once
    ASSERT(apply_entry(fs, &entry) == 0);
    ASSERT(inode_bitmap_is_set(fs, 50) == true);
    // Apply again (idempotent)
    ASSERT(apply_entry(fs, &entry) == 0);
    ASSERT(inode_bitmap_is_set(fs, 50) == true);
    fs_close(fs);
}
void test_apply_dir_add_idempotent() {
    FileSystem* fs = create_test_filesystem();
    // Create entry
    JEDirAddData data;
    data.dir_inode = 1;
    data.target_inode = 50;
    data.name_len = 4;
    data.file_type = DT_REG;
    memcpy(data.name, "test", 4);
    JournalEntry entry;
    entry.type = JE_DIR_ADD;
    entry.length = sizeof(data);
    entry.sequence = 1;
    memcpy(entry.data, &data, sizeof(data));
    entry.checksum = crc32(0, &entry, sizeof(entry.header) + sizeof(data));
    // Apply once
    ASSERT(apply_entry(fs, &entry) == 0);
    ASSERT(dir_lookup(fs, 1, "test") == 50);
    // Apply again (should handle EEXIST)
    ASSERT(apply_entry(fs, &entry) == 0);
    ASSERT(dir_lookup(fs, 1, "test") == 50);
    fs_close(fs);
}
```
### test_crash_simulation.c
```c
void test_crash_before_commit() {
    const char* img = "crash_test.img";
    create_test_filesystem_image(img, 100);
    // Fork child that will be killed
    pid_t pid = fork();
    if (pid == 0) {
        // Child: begin transaction, log, but don't commit
        FileSystem* fs = fs_open(img);
        Journal* j = fs->journal;
        Transaction* txn = journal_begin(j);
        JEInodeAllocData data = {.inode_num = 100};
        journal_log(j, JE_INODE_ALLOC, &data, sizeof(data));
        // Sleep to ensure parent can kill us
        sleep(10);
        // Shouldn't reach here
        journal_commit(j);
        exit(0);
    }
    // Parent: kill child quickly
    usleep(100000);  // 100ms
    kill(pid, SIGKILL);
    waitpid(pid, NULL, 0);
    // Recover
    FileSystem* fs = fs_open(img);
    ASSERT(fs != NULL);
    // Inode should NOT be allocated
    ASSERT(inode_bitmap_is_set(fs, 100) == false);
    fs_close(fs);
    unlink(img);
}
void test_crash_after_commit() {
    const char* img = "crash_test.img";
    create_test_filesystem_image(img, 100);
    pid_t pid = fork();
    if (pid == 0) {
        FileSystem* fs = fs_open(img);
        // Create a file (journaled)
        uint64_t inode = fs_create_journaled(fs, "/survivor.txt", 0644);
        // Write data
        fs_write_journaled(fs, inode, 0, "I survived!", 11);
        // Sync to ensure commit
        fsync(fs->dev->fd);
        // Now sleep - parent will kill us before we can do anything else
        sleep(10);
        exit(0);
    }
    usleep(200000);  // 200ms to ensure commit
    kill(pid, SIGKILL);
    waitpid(pid, NULL, 0);
    // Recover
    FileSystem* fs = fs_open(img);
    ASSERT(fs != NULL);
    // File should exist
    uint64_t inode = path_resolve(fs, "/survivor.txt");
    ASSERT(inode != 0);
    // Data should be correct
    char buf[20];
    ssize_t n = fs_read(fs, inode, 0, buf, sizeof(buf));
    ASSERT(n == 11);
    ASSERT(memcmp(buf, "I survived!", 11) == 0);
    fs_close(fs);
    unlink(img);
}
void test_multiple_crash_cycles() {
    const char* img = "crash_cycles.img";
    create_test_filesystem_image(img, 100);
    for (int cycle = 0; cycle < 10; cycle++) {
        pid_t pid = fork();
        if (pid == 0) {
            FileSystem* fs = fs_open(img);
            // Create file
            char path[50];
            snprintf(path, sizeof(path), "/file_%d.txt", cycle);
            fs_create_journaled(fs, path, 0644);
            usleep(50000);  // 50ms
            exit(0);
        }
        // Kill at random time
        usleep(rand() % 100000);
        kill(pid, SIGKILL);
        waitpid(pid, NULL, 0);
        // Recover and verify
        FileSystem* fs = fs_open(img);
        ASSERT(verify_filesystem(fs) == 0);
        fs_close(fs);
    }
    unlink(img);
}
```
---
## Performance Targets
| Operation | Target | Measurement Method |
|-----------|--------|-------------------|
| journal_begin | < 1 μs | Microbenchmark, no I/O |
| journal_log (per entry) | < 500 ns | Microbenchmark, in-memory |
| journal_commit (with fsync) | < 10 ms (HDD), < 1 ms (SSD), < 100 μs (NVMe) | Time from commit call to return |
| apply_entry (inode_update) | < 100 μs | Includes inode write |
| journal_recover (10 transactions) | < 100 ms | Time from mount to ready |
| journal_checkpoint | < 50 ms | Time to clear journal |
| fs_create_journaled | < 15 ms (HDD), < 2 ms (SSD) | Full transaction lifecycle |
| Metadata write amplification | 1.0× | Compare journaled vs non-journaled metadata writes |
**Hardware Soul - Performance Analysis:**
```
Transaction commit latency breakdown (HDD):
  1. journal_log (5 entries): 5 × 500ns = 2.5 μs
  2. journal_write_entry (5 entries + commit): 
     6 × (seek + rotational + transfer) = 6 × 10ms = 60ms worst case
     (Sequential: 6 × 0.5ms = 3ms)
  3. fsync: ~10ms
  Total: ~13-73ms (sequential vs random)
Transaction commit latency breakdown (NVMe):
  1. journal_log: 2.5 μs
  2. journal_write_entry: 6 × 25μs = 150μs
  3. fsync: ~100μs
  Total: ~250μs
NVMe is ~50-300× faster than HDD for journaling!
Optimization: Batch writes
  - Collect all entries in memory
  - Single sequential write for all entries + commit
  - Single fsync
  - Reduces from 6 I/Os to 1-2 I/Os
```

![Circular Buffer Wrap-Around](./diagrams/tdd-diag-m6-08.svg)

---
## Diagrams
{{DIAGRAM:tdd-diag-m6-09}}

![Metadata vs Full Data Journaling](./diagrams/tdd-diag-m6-10.svg)


![File Create with Journaling](./diagrams/tdd-diag-m6-11.svg)

{{DIAGRAM:tdd-diag-m6-12}}
{{DIAGRAM:tdd-diag-m6-13}}

![Journal Entry Checksum](./diagrams/tdd-diag-m6-14.svg)


![Recovery State Machine](./diagrams/tdd-diag-m6-15.svg)

---
[[CRITERIA_JSON: {"module_id": "filesystem-m6", "criteria": ["Journal region occupies a fixed set of blocks configured in superblock with sequential wrap-around storage for journal entries", "Each filesystem metadata operation is wrapped in a transaction with begin, journal writes, commit, apply to data structures, and end lifecycle", "Transaction commit writes a commit record to the journal followed by fsync to ensure durability before returning success", "Recovery on mount scans journal for committed transactions and replays their operations while discarding uncommitted transactions", "Metadata journaling mode journals only metadata changes (inode updates, bitmap changes, directory entries) while data blocks are written directly", "Crash simulation test kills filesystem process mid-operation, remounts, and verifies recovery produces consistent filesystem with no orphaned inodes or leaked blocks", "Journal checkpoint marks journal as clean after all committed transactions are applied so it doesn't replay stale entries", "All journal entry application operations are idempotent allowing safe replay of transactions that may have been partially applied before crash"]}] ]
<!-- END_TDD_MOD -->


# Project Structure: Filesystem Implementation

## Directory Tree

```
project-root/
├── filesystem/              # Core Implementation Files
│   ├── 01_block.h           # (M1) BlockDevice abstraction and constants
│   ├── 02_block.c           # (M1) Block I/O (read/write/create)
│   ├── 03_superblock.h      # (M1) Superblock disk structure
│   ├── 04_superblock.c      # (M1) Superblock init/validation/checksum
│   ├── 05_bitmap.h          # (M1) Bitmap allocation declarations
│   ├── 06_bitmap.c          # (M1) Block/Inode bitmap management
│   ├── 07_layout.h          # (M1) FS region calculation logic
│   ├── 08_layout.c          # (M1) Layout sizing and positioning
│   ├── 09_mkfs.h            # (M1) Filesystem formatter definitions
│   ├── 10_mkfs.c            # (M1) mkfs logic and root initialization
│   ├── 11_verify.h          # (M1) Consistency checker declarations
│   ├── 12_verify.c          # (M1) Implementation of verify_filesystem
│   ├── 13_mkfs_main.c       # (M1) CLI entry point for formatter
│   ├── 14_inode.h           # (M2) Inode structure and constants
│   ├── 15_inode.c           # (M2) Inode read/write/serialization
│   ├── 16_block_ptr.h       # (M2) Multi-level pointer resolution types
│   ├── 17_block_ptr.c       # (M2) Indirect block traversal logic
│   ├── 18_inode_alloc.h     # (M2) Inode block allocation/freeing
│   ├── 19_inode_alloc.c     # (M2) Recursive indirect block management
│   ├── 20_timestamp.h       # (M2) Timestamp update flags/types
│   ├── 21_timestamp.c       # (M2) POSIX atime/mtime/ctime logic
│   ├── 22_direntry.h        # (M3) DirEntry structure (name->inode)
│   ├── 23_direntry.c        # (M3) Directory entry serialization
│   ├── 24_dir_lookup.h      # (M3) Directory search declarations
│   ├── 25_dir_lookup.c      # (M3) Name scanning within blocks
│   ├── 26_dir_modify.h      # (M3) Add/Remove entry declarations
│   ├── 27_dir_modify.c      # (M3) Directory content manipulation
│   ├── 28_path.h            # (M3) Path parsing and resolution types
│   ├── 29_path.c            # (M3) Iterative path resolution logic
│   ├── 30_mkdir_rmdir.h     # (M3) Directory lifecycle declarations
│   ├── 31_mkdir_rmdir.c     # (M3) Directory creation/deletion logic
│   ├── 32_link.h            # (M3) Hard/Symbolic link declarations
│   ├── 33_link.c            # (M3) Link and symlink implementation
│   ├── 34_rename.h          # (M3) Rename operation declarations
│   ├── 35_rename.c          # (M3) Atomic move/rename implementation
│   ├── 36_readdir.h         # (M3) Directory iteration interfaces
│   ├── 37_readdir.c         # (M3) Directory listing implementation
│   ├── 38_file_create.h     # (M4) File creation declarations
│   ├── 39_file_create.c     # (M4) fs_create implementation
│   ├── 40_file_read.h       # (M4) File read result structures
│   ├── 41_file_read.c       # (M4) Block-aware read with holes
│   ├── 42_file_write.h      # (M4) File write result structures
│   ├── 43_file_write.c      # (M4) Block allocation during writes
│   ├── 44_file_truncate.h   # (M4) Truncation/sizing declarations
│   ├── 45_file_truncate.c   # (M4) Block freeing and zero-filling
│   ├── 46_file_append.h     # (M4) Append operation declarations
│   ├── 47_file_append.c     # (M4) Append via write logic
│   ├── 48_file_utils.h      # (M4) Block boundary helpers
│   ├── 49_file_utils.c      # (M4) Internal I/O utility logic
│   ├── 50_fuse_main.h       # (M5) FUSE global state and types
│   ├── 51_fuse_main.c       # (M5) FUSE daemon entry point
│   ├── 52_fuse_callbacks.h  # (M5) FUSE operation vector setup
│   ├── 53_fuse_getattr.c    # (M5) stat and statfs FUSE implementation
│   ├── 54_fuse_dir.c        # (M5) Directory FUSE callbacks
│   ├── 55_fuse_file.c       # (M5) File I/O FUSE callbacks
│   ├── 56_fuse_meta.c       # (M5) Mode/ownership FUSE callbacks
│   ├── 57_fuse_delete.c     # (M5) Unlink and rename FUSE callbacks
│   ├── 58_fuse_symlink.c    # (M5) Link/Symlink FUSE callbacks
│   ├── 59_fuse_lock.h       # (M5) Thread-safety declarations
│   ├── 60_fuse_lock.c       # (M5) Mutex-based concurrency management
│   ├── 61_journal.h         # (M6) Journal structures and constants
│   ├── 62_journal.c         # (M6) Journal region and header I/O
│   ├── 63_journal_txn.h     # (M6) Transaction lifecycle declarations
│   ├── 64_journal_txn.c     # (M6) Begin/Commit/Abort state machine
│   ├── 65_journal_log.h     # (M6) Entry logging helpers
│   ├── 66_journal_log.c     # (M6) Checksummed log entry serialization
│   ├── 67_journal_apply.h   # (M6) Log replay and idempotency helpers
│   ├── 68_journal_apply.c   # (M6) Applying log to physical blocks
│   ├── 69_journal_recover.h # (M6) Mount-time recovery declarations
│   ├── 70_journal_recover.c # (M6) Scanning and replaying transactions
│   ├── 71_journal_wrap.h    # (M6) Atomic operation wrappers
│   └── 72_journal_wrap.c    # (M6) Journaled create/write/delete
├── tests/                   # Automated Test Suite
│   ├── test_block.c         # (M1) Block device unit tests
│   ├── test_bitmap.c        # (M1) Allocator unit tests
│   ├── test_mkfs.c          # (M1) Formatter integration tests
│   ├── test_inode_struct.c  # (M2) Inode layout/read/write tests
│   ├── test_block_ptr.c     # (M2) Indirection resolution tests
│   ├── test_inode_alloc.c   # (M2) Allocation/deallocation tests
│   ├── test_sparse.c        # (M2/M4) Hole detection and handling
│   ├── test_direntry.c      # (M3) Dir structure tests
│   ├── test_dir_lookup.c    # (M3) Directory scanning tests
│   ├── test_path.c          # (M3) Path resolution unit tests
│   ├── test_mkdir_rmdir.c   # (M3) Directory lifecycle tests
│   ├── test_link.c          # (M3) Hard link and symlink tests
│   ├── test_rename.c        # (M3) Atomicity and move tests
│   ├── test_file_create.c   # (M4) File creation tests
│   ├── test_file_read.c     # (M4) Reading data streams
│   ├── test_file_write.c    # (M4) Writing data streams
│   ├── test_file_truncate.c # (M4) File resizing tests
│   ├── test_file_boundary.c # (M4) Block edge cases
│   ├── test_fuse_mount.c    # (M5) Mount/unmount lifecycle tests
│   ├── test_fuse_callbacks.c# (M5) Callback correctness tests
│   ├── test_fuse_concurrent.c# (M5) Thread safety stress tests
│   ├── test_fuse_real_tools.c# (M5) cat/ls/cp/gcc integration tests
│   ├── test_journal_header.c# (M6) Journal integrity tests
│   ├── test_journal_txn.c   # (M6) Txn lifecycle unit tests
│   ├── test_journal_log.c   # (M6) Serialization/checksum tests
│   ├── test_journal_apply.c # (M6) Idempotency verification
│   ├── test_journal_recover.c# (M6) Replay logic tests
│   └── test_crash_simulation.c# (M6) Kill/Recover stress tests
├── Makefile                 # Build system (All)
├── .gitignore               # Ignore disk images and binaries
└── README.md                # Project documentation
```

## Creation Order

1.  **Block Foundation (M1)**
    *   Create `01_block.h` through `04_superblock.c`.
    *   Implement `05_bitmap.c` and `07_layout.c`.
    *   Build the `mkfs` utility (`09_mkfs.c`, `13_mkfs_main.c`) to create disk images.

2.  **Metadata Management (M2)**
    *   Implement `14_inode.c` for inode disk access.
    *   Implement `17_block_ptr.c` to handle the indirect pointer logic.
    *   Build `19_inode_alloc.c` to manage block allocation within files.

3.  **Naming & Hierarchy (M3)**
    *   Implement `23_direntry.c` and `25_dir_lookup.c`.
    *   Implement `29_path.c` for translating strings like `/a/b/c` to inodes.
    *   Implement `31_mkdir_rmdir.c` and `35_rename.c`.

4.  **Byte Stream I/O (M4)**
    *   Implement `41_file_read.c` and `43_file_write.c` with support for block crossing.
    *   Implement `45_file_truncate.c` for file resizing.
    *   Build `49_file_utils.c` for common I/O helpers.

5.  **OS Integration (M5)**
    *   Implement `51_fuse_main.c` and `52_fuse_callbacks.h`.
    *   Map existing operations to FUSE callbacks (`53_fuse_getattr.c` through `58_fuse_symlink.c`).
    *   Implement thread safety in `60_fuse_lock.c`.

6.  **Reliability Layer (M6)**
    *   Build the journal buffer management (`62_journal.c`).
    *   Implement transaction logging (`64_journal_txn.c`, `66_journal_log.c`).
    *   Implement recovery replay (`70_journal_recover.c`).
    *   Wrap standard operations in transactions (`72_journal_wrap.c`).

## File Count Summary
- **Total Source Files:** 72
- **Total Test Files:** 29
- **Total Directories:** 2 (excluding root)
- **Estimated Lines of Code:** ~4,500 - 6,000 LOC (including tests)
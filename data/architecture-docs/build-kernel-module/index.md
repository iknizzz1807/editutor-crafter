# 🎯 Project Charter: Linux Kernel Module — Character Device Driver
## What You Are Building
A production-quality Linux character device driver implemented as a loadable kernel module. The driver exposes `/dev/mydevice` to userspace, accepts `read()`, `write()`, `ioctl()`, and `poll()` system calls, stores data in a kernel-allocated FIFO buffer, exposes runtime statistics through `/proc/mydevice`, and handles concurrent access from multiple processes without data corruption, deadlocks, or unkillable processes.
## Why This Project Exists
The kernel-userspace boundary is the most consequential interface in every Linux system, yet most developers treat it as an opaque black box. Building a driver from scratch forces you to confront what "everything is a file" actually means in silicon: function pointer dispatch tables, hardware-enforced memory separation, CPU privilege levels, and the precise contract that makes `read()` block, `poll()` wake, and `Ctrl+C` terminate—or permanently destroy—a sleeping process.
## What You Will Be Able to Do When Done
- Write and load a kernel module that survives `insmod`/`rmmod` without panicking the machine
- Register a character device with dynamic major number allocation and automatic `/dev/` node creation via udev
- Implement `file_operations` handlers that safely transfer data across the kernel-userspace boundary using `copy_to_user`/`copy_from_user`
- Design a structured `ioctl` interface with encoded command numbers, shared kernel/userspace headers, and correct `-ENOTTY` validation
- Create `/proc` filesystem entries with `seq_file` that expose live driver statistics
- Protect shared kernel state with `mutex_lock_interruptible` so concurrent processes never corrupt each other's data
- Implement blocking I/O using `wait_event_interruptible` and `wake_up_interruptible` so sleeping processes wake correctly when data arrives
- Handle `O_NONBLOCK` with proper `-EAGAIN` semantics and implement `.poll` so your device integrates with `select()`/`poll()`/`epoll()`
- Propagate `-ERESTARTSYS` correctly so signals (including `SIGKILL`) always terminate blocked processes
- Debug kernel code using `printk`, `dmesg`, `strace`, and oops analysis
## Final Deliverable
Approximately 500 lines of C across two source files (`mydevice.c`, `mydevice.h`) plus a Kbuild `Makefile`, a userspace test program (`test_mydevice.c`), and a Python stress test (`concurrent_test.py`). The module loads in under 100ms, creates `/dev/mydevice` and `/proc/mydevice` automatically, passes a checksum-verified concurrent stress test with 4 simultaneous writers and 4 simultaneous readers producing zero data corruption, and unloads cleanly with `rmmod`. A verification script (`verify_m4.sh`) runs all acceptance checks with a single command.
## Is This Project For You?
**You should start this if you:**
- Can write C with pointers, structs, pointer arithmetic, and manual memory management without looking up syntax
- Understand what a process, file descriptor, system call, and virtual address space are at a conceptual level
- Have written signal handlers in C and understand `SA_RESTART`, `EINTR`, and the signal-safe function list (the `signal-handler` prerequisite project covers this)
- Are comfortable working in a Linux terminal and understand `make`, `grep`, `dmesg`, and basic shell scripting
- Can reboot a virtual machine without panic — you will crash the kernel at least once
**Come back after you've learned:**
- C pointers and manual memory management — if `char *p = malloc(n); p[n] = 0;` doesn't immediately read as a bug, learn C first
- The `signal-handler` prerequisite project — blocking I/O in the kernel requires understanding `EINTR`/`SA_RESTART` from the userspace side first
- Basic Linux process model — if you don't know what a file descriptor is or how `fork()`/`exec()` work, study [The Linux Programming Interface](https://man7.org/tlpi/) (chapters 1–5) first
## Estimated Effort
| Phase | Time |
|-------|------|
| Milestone 1: Hello World Kernel Module | ~3 hours |
| Milestone 2: Character Device Driver | ~7 hours |
| Milestone 3: ioctl and /proc Interface | ~7 hours |
| Milestone 4: Concurrent Access, Blocking I/O, and Poll | ~9 hours |
| **Total** | **~26 hours** |
## Definition of Done
The project is complete when:
- `sudo bash verify_m4.sh` runs to completion with zero `FAIL` lines and exits with code 0
- `sudo python3 concurrent_test.py` outputs `PASS: all received messages have valid checksums` with `Corrupted: 0` and `Errors: 0` under 4 concurrent writers and 4 concurrent readers
- A process blocked in `read()` on the device is terminated within one scheduler quantum by `kill -INT $PID` and does not appear in `ps` output afterward (no unkillable `D`-state processes)
- `cat /dev/mydevice` (blocking mode, empty buffer) hangs until a concurrent `echo "data" > /dev/mydevice` is run, then prints the data and exits cleanly
- `sudo rmmod mydevice` succeeds with exit code 0 after all file descriptors are closed, and `dmesg` shows no kernel oops, BUG, or null pointer dereference entries throughout the entire test session

---

# 📚 Before You Read This: Prerequisites & Further Reading
> **Read these first.** The Atlas assumes you are familiar with the foundations below.
> Resources are ordered by when you should encounter them — some before you start, some at specific milestones.
---
## 🧠 Before You Start: Required Foundations
### 1. Virtual Memory and Process Address Spaces
**Best Explanation:** *Operating Systems: Three Easy Pieces* (Arpaci-Dusseau & Arpaci-Dusseau) — **Part II: Virtualization, Chapters 12–16** (Address Spaces, Address Translation, Segmentation, Free Space, Paging). Free at ostep.org.
**Why:** The Atlas repeatedly uses concepts like "kernel space vs. userspace," page faults, and TLB costs. Without a concrete model of how virtual memory works, the explanations of `copy_from_user`, SMAP, and the address space split will be abstract noise. Read this before writing a single line.
**Pedagogical Timing:** **BEFORE starting the project.** The very first Atlas section on kernel vs. userspace address space assumes this fluency.
---
### 2. ELF Object Files and Dynamic Linking
**Best Explanation:** *Linkers and Loaders* (John Levine, 2000) — **Chapter 1** (Linking and Loading) and **Chapter 10** (Dynamic Linking and Loading). Available free at iecc.com/linker.
**Why:** Every milestone in this project produces a `.ko` file — a partially-linked ELF relocatable object. Understanding how symbols are resolved at load time explains why `MODULE_LICENSE("GPL")` gates symbol access, why vermagic must match, and what `insmod` actually does when it calls `finit_module()`.
**Pedagogical Timing:** **BEFORE Milestone 1.** The "Hardware Soul" section of M1 walks through the `insmod` chain step-by-step; this gives you the vocabulary to follow it.
---
### 3. The Linux Kernel Module Interface — The Official Guide
**Code:** Linux kernel source — `samples/hello/` for the minimal module template; `include/linux/module.h` for `MODULE_LICENSE`, `module_init`, `module_exit` macro definitions.
**URL:** https://elixir.bootlin.com/linux/latest/source/include/linux/module.h
**Best Explanation:** *Linux Kernel Development* (Robert Love, 3rd ed.) — **Chapter 16: Modules** (pp. 341–360). Covers `module_init`/`module_exit`, `MODULE_*` macros, `EXPORT_SYMBOL_GPL`, and the vermagic check.
**Why:** This is the gold standard concise reference for exactly the material in Milestone 1 — written by a former kernel developer who worked on the scheduler.
**Pedagogical Timing:** **BEFORE Milestone 1.** Read before implementing `hello.c`.
---
## 🔌 Milestone 2: Character Devices and VFS
### 4. The VFS Layer and `file_operations`
**Paper:** David Kleikamp et al., "The Linux Virtual File System" (from *The Architecture of Open Source Applications*, Vol. 1, 2011). Available free at aosabook.org, Chapter 4.
**Code:** Linux kernel source — `fs/char_dev.c` (the character device registration layer, ~500 lines). Specifically: `cdev_add()`, `chrdev_open()`, and how the VFS dispatches to `file_operations`. https://elixir.bootlin.com/linux/latest/source/fs/char_dev.c
**Why:** Reading the actual 500-line implementation of character device dispatch makes the M2 Atlas material concrete rather than incantatory. You will see exactly how your `mydevice_fops` pointer gets stored and retrieved.
**Pedagogical Timing:** **Read after completing Milestone 1, before implementing Milestone 2.** You need the module infrastructure in place before the dispatch chain makes sense.
---
### 5. `copy_to_user` / `copy_from_user` and SMAP/SMEP
**Paper:** "SMEP: What is it, and how is it used?" — Rafal Wojtczuk & Joanna Rutkowska, Invisible Things Lab, 2011. *Attacking Intel BIOS* (adjacent paper by same authors gives full context for hardware enforcement).
**Code:** Linux kernel source — `arch/x86/lib/usercopy_64.S` — the actual assembly implementation of `copy_to_user` on x86_64, including the `STAC`/`CLAC` instructions and the exception table entries. https://elixir.bootlin.com/linux/latest/source/arch/x86/lib/usercopy_64.S
**Why:** The Atlas explains *what* these functions do; the assembly source shows *how* — the `STAC`/`CLAC` dance, the `rep movsb`, and the `.fixup` section that handles mid-copy page faults. Reading ~80 lines of annotated assembly makes the M2 safety model fully transparent.
**Pedagogical Timing:** **During Milestone 2** — after you write your first `copy_from_user` call and wonder what actually happens on SMAP-enabled hardware.
---
### 6. Linux Device Drivers, 3rd Edition — The Canonical Reference
**Code:** All code examples from the book are at https://github.com/jedfur/ldd3-examples (updated for modern kernels).
**Best Explanation:** *Linux Device Drivers, 3rd Edition* (Corbet, Rubini, Kroah-Hartman) — **Chapter 3: Char Drivers** (pp. 42–86). Covers `file_operations`, `cdev`, major/minor numbers, `copy_to/from_user`, `f_pos` tracking, and the `open`/`release` contract. Free at lwn.net/Kernel/LDD3.
**Why:** This is *the* authoritative reference for kernel driver development, written by the maintainers of the kernel's driver subsystem. Chapter 3 maps almost one-to-one with Milestone 2.
**Pedagogical Timing:** **Read Chapter 3 before Milestone 2, Chapter 6 before Milestone 4.** The book's chapter structure mirrors the project's milestone structure almost exactly.
---
## 🎛️ Milestone 3: ioctl and /proc
### 7. ioctl Command Number Encoding
**Spec:** Linux kernel documentation — `Documentation/userspace-api/ioctl/ioctl-number.rst` — the official registry of `_IOC_TYPE` magic bytes and the structured encoding convention.
**URL:** https://www.kernel.org/doc/html/latest/userspace-api/ioctl/ioctl-number.html
**Code:** `include/uapi/asm-generic/ioctl.h` — the 40-line definition of `_IO`, `_IOR`, `_IOW`, `_IOWR` and the bit-field packing.
https://elixir.bootlin.com/linux/latest/source/include/uapi/asm-generic/ioctl.h
**Why:** Reading the 40-line macro definition once makes the direction-bit confusion (`_IOC_READ` vs. `_IOW`) permanently clear. The official registry prevents magic-byte collisions with real drivers.
**Pedagogical Timing:** **Before implementing `mydevice_ioctl`** in Milestone 3. Read the spec first, then implement — not the reverse.
---
### 8. seq_file Interface for /proc
**Best Explanation:** LWN.net — "Driver porting: The seq_file interface" (Jonathan Corbet, 2003). https://lwn.net/Articles/22355/ — still the definitive tutorial, written by the seq_file author.
**Code:** Linux kernel source — `fs/seq_file.c` (~600 lines, the complete seq_file implementation). Particularly `seq_read()` showing how it buffers `show()` output across partial reads. https://elixir.bootlin.com/linux/latest/source/fs/seq_file.c
**Why:** The LWN article explains *why* the old `proc_read` API breaks for output larger than 4KB and precisely how `single_open` + `seq_printf` solves it. Reading the original author's explanation takes 15 minutes and prevents the most common `/proc` implementation bugs.
**Pedagogical Timing:** **Before implementing `/proc/mydevice`** in Milestone 3.
---
### 9. The Linux UAPI Header System
**Best Explanation:** LWN.net — "Reorganizing the kernel's user-space API headers" (Jonathan Corbet, 2012). https://lwn.net/Articles/507794/ — explains why `include/uapi/` exists and how the kernel enforces the kernel/userspace header contract.
**Why:** Your `mydevice.h` is a micro-version of the UAPI system. Understanding the design rationale explains why the shared header must avoid kernel-only types (`__user`, `struct file`, etc.) and why field ordering in `struct mydevice_status` is an ABI commitment.
**Pedagogical Timing:** **During Milestone 3**, after writing `mydevice.h` — read this to understand the production-scale version of what you just built.
---
## 🔒 Milestone 4: Concurrency, Blocking I/O, and Poll
### 10. Linux Kernel Synchronization Primitives
**Best Explanation:** *Linux Kernel Development* (Robert Love, 3rd ed.) — **Chapter 9: An Introduction to Kernel Synchronization** and **Chapter 10: Kernel Synchronization Methods** (pp. 161–210). The clearest explanation of spinlocks vs. mutexes, the "can you sleep?" rule, and `atomic_t` in print.
**Why:** Love's explanation of why interrupt context cannot sleep — and the hardware/software reason behind it — is the single clearest treatment of the topic that exists in book form. The "can this code path sleep?" heuristic from Chapter 9 is the mental model the Atlas builds on.
**Pedagogical Timing:** **BEFORE Milestone 4.** Read Chapters 9–10 before writing any mutex or wait queue code.
---
### 11. Wait Queues and the Condition-Recheck Pattern
**Code:** Linux kernel source — `include/linux/wait.h` — the complete `wait_event_interruptible` macro definition (~40 lines showing the prepare_to_wait/schedule/finish_wait loop). https://elixir.bootlin.com/linux/latest/source/include/linux/wait.h
**Best Explanation:** *Linux Device Drivers, 3rd Edition* — **Chapter 6: Advanced Char Driver Operations**, section "Blocking I/O" (pp. 153–168). Contains the clearest worked example of a producer-consumer wait queue with the mandatory condition recheck explained line-by-line.
**Why:** Reading the actual macro expansion of `wait_event_interruptible` makes the thundering-herd recheck and the `-ERESTARTSYS` path permanently clear. The LDD3 section provides the producer-consumer worked example.
**Pedagogical Timing:** **Before implementing blocking read/write** in Milestone 4 (Phase 3 in the TDD). Read both together.
---
### 12. poll/select/epoll — Kernel-Side Mechanics
**Best Explanation:** LWN.net — "Scalable I/O Event Notification Mechanisms" (Davide Libenzi, 2001). https://lwn.net/Articles/5744/ — the original epoll design rationale, explaining exactly why `poll_wait` must subscribe before returning the mask.
**Code:** Linux kernel source — `fs/select.c`, function `do_pollfd()` (~30 lines) — how the kernel calls your `.poll` function pointer and interprets the returned mask. https://elixir.bootlin.com/linux/latest/source/fs/select.c
**Why:** The Libenzi article is the primary source explaining why `poll()` is O(n) and `epoll` is O(1) per event — and why your four-line `.poll` implementation works for both. Reading it makes the Atlas's "From your .poll to nginx's epoll" connection literal rather than metaphorical.
**Pedagogical Timing:** **Before implementing `mydevice_poll`** in Milestone 4 (Phase 6 in the TDD).
---
### 13. `SA_RESTART` and the `-ERESTARTSYS` Contract
**Spec:** POSIX.1-2017 — `sigaction(2)` manual page, `SA_RESTART` flag description. Available at `man 2 sigaction` or https://pubs.opengroup.org/onlinepubs/9699919799/functions/sigaction.html
**Best Explanation:** *The Linux Programming Interface* (Michael Kerrisk, 2010) — **Chapter 21: Signals: Signal Handlers**, section 21.5 "Interrupted System Calls and errno EINTR" (pp. 439–445). The definitive treatment of when syscalls are restarted, when they return `EINTR`, and the role of `SA_RESTART`.
**Why:** The Atlas explains `-ERESTARTSYS` from the kernel side; Kerrisk explains it from the userspace side. Reading both gives you the complete picture of why returning `-ERESTARTSYS` (not 0, not `-EINTR`) is the contract that makes `SA_RESTART` work.
**Pedagogical Timing:** **During Milestone 4**, after encountering the `-ERESTARTSYS` pattern in the blocking read implementation. Read *after* writing the wait queue code so you have concrete context.
---
### 14. GFP Flags and the Slab Allocator
**Best Explanation:** *Linux Kernel Development* (Robert Love, 3rd ed.) — **Chapter 12: Memory Management**, sections on the Slab Allocator and GFP flags (pp. 236–252).
**Code:** Linux kernel source — `mm/slub.c`, function `kmem_cache_alloc_node()` — the SLUB allocator's main allocation path, showing how `GFP_KERNEL` vs. `GFP_ATOMIC` controls the allocation behavior under memory pressure. https://elixir.bootlin.com/linux/latest/source/mm/slub.c
**Why:** The "can this context sleep?" question that governs `GFP_KERNEL` vs. `GFP_ATOMIC` is the same question that governs mutex vs. spinlock. Love's Chapter 12 unifies both under a single mental model: "sleeping allocation = sleeping lock = process context only."
**Pedagogical Timing:** **Before Milestone 2** (first `kzalloc` call) and revisit **before Milestone 4** (the RESIZE ioctl places `kzalloc` outside the mutex deliberately — this chapter explains why).
---
## 📖 Reference: Always Available
### 15. The Kernel Documentation Hub
**Spec/Reference:** https://www.kernel.org/doc/html/ — the complete generated kernel documentation, including:
- `Documentation/kernel-hacking/hacking.rst` — "Unreliable Guide to Hacking the Linux Kernel"
- `Documentation/process/coding-style.rst` — kernel coding style (tabs, goto patterns)
- `Documentation/core-api/` — synchronization, memory allocation, and data structure references
**Why:** This is the primary source. When the Atlas references a kernel subsystem behavior, this documentation is the ground truth. Bookmark it; you will return to it throughout all four milestones.
**Pedagogical Timing:** **Ongoing reference throughout all milestones.** Do not read linearly — use it as a lookup reference when the Atlas references a specific behavior.

---

# Linux Kernel Module: Character Device Driver

This project builds a complete Linux character device driver from scratch, progressing from a minimal loadable kernel module to a fully concurrent, poll-capable device with ioctl control and /proc introspection. You will cross the kernel-userspace boundary repeatedly, learning why direct pointer dereference is forbidden, how the kernel's VFS layer dispatches file operations to your code, and how synchronization primitives differ fundamentally from their userspace counterparts.

The project is structured as four milestones that mirror how real device drivers are developed: first prove you can load code into the kernel safely, then implement the data path (read/write), then add the control path (ioctl) and observability (/proc), and finally handle the hardest problem—concurrent access from multiple processes without corruption, deadlocks, or unkillable processes.

By the end, you'll have a working /dev/ device that multiple processes can open, read, write, poll, and control simultaneously—the same architecture underlying /dev/null, /dev/random, and real hardware drivers.



<!-- MS_ID: build-kernel-module-m1 -->
# Milestone 1: Hello World Kernel Module
## Before You Write a Single Line
Stop and consider what you're about to do. When you write a userspace program and it crashes, the operating system catches the fall: the process dies, memory is freed, file descriptors close, and everything else on the machine continues unharmed. You've been writing code inside a protective bubble your entire career as a programmer.
That bubble ends here.
When you load a kernel module and it crashes, the *entire machine* crashes with it. There is no safety net, no exception handler, no segfault-and-continue. A NULL pointer dereference in your module produces a kernel panic — the screen locks up, the filesystem may not flush, and the machine needs a hard reboot. You are writing code that runs with the same authority and in the same address space as the Linux scheduler, the memory allocator, and every other piece of infrastructure the system depends on.
This isn't meant to frighten you. It's meant to calibrate you. The techniques in this milestone — the specific Makefile format, the module metadata, the logging API — aren't bureaucratic ceremony. They're the minimum viable harness that keeps you from destroying your environment while you learn. By the end of this chapter, you'll understand exactly *why* each piece exists.

![Module Lifecycle: insmod → init → running → rmmod → exit](./diagrams/diag-m1-module-lifecycle.svg)

---
## The Revelation: What a Module Actually Is
Here's the mental model most people start with: *a kernel module is like a shared library — the kernel loads it into a separate area, runs its code, and if it crashes, only the module dies.*
This model is completely wrong, and the correction is the most important insight in this milestone.
**A kernel module does not run in its own process.** When you run `insmod mymodule.ko`, the `insmod` process makes a system call (`init_module` or `finit_module`), and the kernel copies your module's code and data into **kernel virtual address space** — the same address space used by every other piece of kernel code. Your module's `init` function runs in the context of the `insmod` process, yes, but it runs with kernel privilege, using kernel stack, inside kernel memory. When `init` returns, your code is permanently woven into the running kernel. Your functions live alongside `schedule()`, `kmalloc()`, and the TCP stack.

> **🔑 Foundation: Kernel vs userspace address spaces**
> 
> ## Kernel vs. Userspace Address Spaces
**What it IS**
Every process running on a Linux system sees a *virtual* address space — a private map of memory addresses that the CPU translates into real physical RAM. This virtual space is split into two distinct regions:
- **Userspace**: The lower portion of the address range, owned by the running application. Your browser, shell, and Python scripts all live here. Each process gets its *own* isolated slice, so one process cannot directly read or write another's memory.
- **Kernel space**: The upper portion, reserved exclusively for the kernel. It is mapped into *every* process's address space, but it is protected by hardware privilege levels (CPU rings). Userspace code that tries to access kernel addresses triggers a fault and is killed.
On a typical 64-bit x86 Linux system, the split is roughly: processes get the lower 128 TB of virtual address space; the kernel occupies a separate high region. On 32-bit systems the classic split was 3 GB userspace / 1 GB kernel (the `3G/1G` split), which is why embedded and older systems sometimes struggled with memory pressure.
When a userspace program needs a kernel service — reading a file, allocating memory, sending a packet — it crosses this boundary via a **system call**. The CPU switches privilege level (ring 3 → ring 0), executes kernel code, then returns to userspace. This crossing is intentional and controlled.
**WHY you need it right now**
Kernel modules run *in kernel space*, with full ring 0 privileges. There is no memory protection between your module and the rest of the kernel. A bad pointer dereference in your module doesn't just crash your program — it can corrupt kernel data structures and panic the whole system. Understanding the boundary explains:
- Why you cannot simply `printf()` from a module (no C standard library in kernel space).
- Why accessing a userspace pointer directly from kernel code is dangerous and requires special functions like `copy_from_user()` / `copy_to_user()`.
- Why a kernel bug can take down the entire machine, not just one process.
**Key insight**
> **Kernel code has no safety net.** In userspace, the kernel protects processes from each other and from themselves. In kernel space, *you* are the kernel — there is nothing below you to catch mistakes. This is why kernel programming demands a different level of discipline than application programming.

This has a direct consequence you need to feel before you write code:
```
Your module code: part of the kernel
Your module data: part of the kernel
A pointer bug in your module: kills the kernel
A logic error in your module: can corrupt any kernel data structure
```
There is no process boundary, no memory protection, no isolation layer. The kernel and your module share one fate.

![Kernel vs Userspace Address Space Layout](./diagrams/diag-m1-kernel-address-space.svg)

This is also why `MODULE_LICENSE("GPL")` is not bureaucracy. The kernel uses it as a **gate** controlling which symbols your code can call. The Linux kernel exports two categories of symbols:
- **Public symbols** (`EXPORT_SYMBOL`): available to all modules regardless of license
- **GPL-only symbols** (`EXPORT_SYMBOL_GPL`): only available to modules that declare a GPL-compatible license
The vast majority of interesting kernel API — wait queues, `proc_create`, many device subsystems — is exported GPL-only. Omit the license declaration, and the linker will refuse to resolve those symbols at load time. You won't get a runtime error; you'll get a build error when trying to link the module against the kernel. Beyond access to symbols, a non-GPL or missing license declaration "taints" the kernel — meaning kernel developers will not investigate any bug reports from a machine running your module, and some security features disable themselves on a tainted kernel.

> **🔑 Foundation: What kernel taint means**
> 
> ## What Kernel Taint Means
**What it IS**
A "tainted" kernel is one that has loaded code or experienced an event that the kernel developers consider potentially compromising to its integrity. The kernel tracks this state in a bitmask stored in the global variable `tainted_mask`. Each bit represents a specific reason for tainting.
Common taint flags include:
| Flag | Meaning |
|---|---|
| `P` | Proprietary (non-GPL) module loaded |
| `O` | Out-of-tree (unofficial) module loaded |
| `E` | Unsigned module loaded |
| `W` | A warning was triggered |
| `C` | Staging driver loaded |
| `X` | Externally-built module (distro-specific) |
You can inspect taint status with:
```bash
cat /proc/sys/kernel/tainted   # numeric bitmask
dmesg | grep -i taint
```
When a module sets taint flag `O` (out-of-tree) or `P` (proprietary), the kernel prints a message like:
```
module: loading out-of-tree module taints kernel.
```
**WHY you need it right now**
This matters for two immediate reasons:
1. **Bug reports**: If your kernel is tainted and you hit a bug, kernel developers may reject your report. They cannot know whether your out-of-tree module caused or masked the issue. When filing upstream bugs, always reproduce with a clean, untainted kernel.
2. **Security features**: Some security subsystems — notably **UEFI Secure Boot** and **lockdown mode** — actively check taint status or enforce restrictions to prevent untrusted code from running in kernel space. In lockdown mode, even loading unsigned modules may be blocked entirely, or certain kernel interfaces (like `/dev/mem`) are restricted. Your module development workflow may need to account for this (e.g., disabling Secure Boot in development VMs, or setting up module signing).
**Key insight**
> Taint is not a punishment — it's a **diagnostic signal**. It tells anyone inspecting a crash or bug report "something non-standard was loaded here." As a module developer, your modules will taint the kernel during development, and that's fine. Just never file a upstream kernel bug report while tainted, and always understand *why* your kernel is tainted before shipping anything to production.

---
## The Environment: What You're Compiling Against
Before writing code, understand what you're compiling against and why it's unusual.
A normal userspace program compiles against libc headers that are stable across kernel versions. A kernel module compiles against the **headers of the specific kernel it will run in**. The kernel has no stable ABI between versions. Internal structures, function signatures, and even whether certain functions exist can change between 5.15 and 6.1. If you compile a module against 5.15 headers and try to load it into a 6.1 kernel, the load will fail — the kernel embeds a "magic number" (`vermagic`) in every `.ko` file and checks it against its own version at load time.
This means:
```bash
# Find your running kernel version
uname -r
# Example output: 6.8.0-45-generic
# The headers you need
ls /usr/src/linux-headers-$(uname -r)/
```
If those headers aren't installed:
```bash
# Ubuntu/Debian
sudo apt install linux-headers-$(uname -r)
# Fedora/RHEL
sudo dnf install kernel-devel-$(uname -r)
```
### The Kbuild System
You cannot compile a kernel module with a plain `gcc` invocation. The kernel has its own build system, called **Kbuild**, and modules must be built through it.

> **🔑 Foundation: Kbuild system and out-of-tree module compilation**
> 
> ## Kbuild and Out-of-Tree Module Compilation
**What it IS**
**Kbuild** is the Linux kernel's build system — the infrastructure of Makefiles, scripts, and conventions that compiles the kernel itself and any code that needs to run inside it. It is not a generic build system; it is deeply aware of kernel configuration (`Kconfig`), compiler flags, architecture quirks, and kernel ABI requirements.
Kernel modules can be built in two ways:
- **In-tree**: The module's source lives inside the kernel source tree and is built as part of the full kernel build. This is how mainline drivers work.
- **Out-of-tree (external)**: The module's source lives *outside* the kernel source tree. Kbuild is invoked from your module's own directory, pointing back at the kernel's build infrastructure. This is the standard approach for driver development, third-party modules, and learning projects.
A minimal out-of-tree module `Makefile` looks like this:
```makefile
obj-m += hello.o
all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean
```
Breaking this down:
- `obj-m += hello.o` — tells Kbuild to build `hello.c` as a loadable module (`.ko` file). `obj-y` would mean "compile into the kernel image itself."
- `-C /lib/modules/$(uname -r)/build` — changes into the kernel build directory for the *currently running kernel*. This symlink points to the kernel headers/build artifacts installed by your distro (e.g., `linux-headers-$(uname -r)` on Debian/Ubuntu).
- `M=$(PWD)` — tells Kbuild to come back and build the module sources in your current directory.
The result is a `hello.ko` file — a kernel object that can be loaded with `insmod` or `modprobe`.
**A critical dependency**: your module must be compiled against the *exact* kernel version it will run on. The kernel enforces this via a **vermagic** string embedded in every `.ko` file. Trying to load a module built for kernel `6.6.0` into a `6.8.2` kernel will fail:
```
insmod: ERROR: could not insert module hello.ko: Invalid module format
```
This is why distros ship `linux-headers` packages tied to specific kernel versions.
**WHY you need it right now**
Every module you write in this project will be built out-of-tree using exactly this pattern. Understanding the Makefile structure means you can:
- Extend it to compile multi-file modules (`obj-m += mymod.o` with `mymod-objs := file1.o file2.o`).
- Pass extra compiler flags (`ccflags-y := -DDEBUG`).
- Target a different kernel version by changing the `-C` path (useful when cross-compiling or building for a test VM).
- Understand what "build directory" vs "source directory" means in error messages.
**Key insight**
> Your module's `Makefile` doesn't actually *do* the building — it just **delegates** to the kernel's own build system. You are borrowing Kbuild's compiler flags, configuration, and toolchain so your module is binary-compatible with the running kernel. Think of it as saying: "Hey kernel build system, please compile this extra file using all the same rules you used for yourself."


![Kbuild Compilation Flow: Source → .ko](./diagrams/diag-m1-kbuild-flow.svg)

Here is the minimal `Makefile` for an out-of-tree module. **The indentation on the `make` command must use a real tab character, not spaces** — this is a Make requirement, not a kernel requirement:
```makefile
# Kbuild target: tells the Kbuild system which .o files to link into your .ko
obj-m += hello.o
# KDIR: path to the running kernel's build directory
# The ?= means "use this if not set externally"
KDIR ?= /lib/modules/$(shell uname -r)/build
# PWD: current directory — where your source lives
PWD := $(shell pwd)
all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules
clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
```
When you run `make`, here's what happens:
1. `make` invokes the kernel's build system at `$(KDIR)` (passing it control with `-C`)
2. The kernel build system reads `M=$(PWD)` and finds your `Makefile` again
3. It sees `obj-m += hello.o` and knows to compile `hello.c` into `hello.o` then link into `hello.ko`
4. The kernel build system applies all the right compiler flags, include paths, and kernel-specific options automatically
The file produced is `hello.ko` — a kernel object. It's an ELF file with some kernel-specific sections added, including the `vermagic` string that must match the running kernel.
> **Why ELF?** Your `.ko` file is a partially-linked ELF relocatable object. It contains references to kernel symbols (`printk`, `module_param`, etc.) that are resolved at load time by the kernel's own module loader — the same concept as a dynamic linker, but running inside the kernel instead of before `main()`. This is why your module can call `printk` without linking against any library: the symbol resolves to the running kernel's copy of `printk` at `insmod` time.
---
## Writing the Module
Now you're ready to write code. Here's the complete minimal module, followed by a line-by-line explanation:
```c
// hello.c — Minimal loadable kernel module
// Required for all kernel modules: provides MODULE_LICENSE, module_init, module_exit
#include <linux/module.h>
// Provides the kernel's printk logging function
#include <linux/kernel.h>
// Provides module_param() and related macros
#include <linux/moduleparam.h>
// Provides __init and __exit annotations
#include <linux/init.h>
// Module metadata — mandatory
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name <you@example.com>");
MODULE_DESCRIPTION("Minimal hello world kernel module");
MODULE_VERSION("0.1");
// Module parameter: integer with default value 4096
// 0444 = permissions on the sysfs file: owner/group/other can read, none can write
static int buffer_size = 4096;
module_param(buffer_size, int, 0444);
MODULE_PARM_DESC(buffer_size, "Size of the internal buffer in bytes (default: 4096)");
// __init: marks this function as initialization code
// The kernel can discard this memory after init completes (saves RAM)
// module_init() registers this as the entry point
static int __init hello_init(void)
{
    // KERN_INFO is the log level — explained below
    printk(KERN_INFO "hello: module loaded, buffer_size=%d\n", buffer_size);
    return 0;  // 0 = success; negative errno = failure, insmod will report error
}
// __exit: marks this function as exit code
// module_exit() registers this as the cleanup entry point
static void __exit hello_exit(void)
{
    printk(KERN_INFO "hello: module unloaded\n");
    // No return value — rmmod cannot be stopped at this point
}
// Register the init and exit functions with the kernel module infrastructure
module_init(hello_init);
module_exit(hello_exit);
```
### The Init/Exit Contract
`module_init` and `module_exit` are macros that register your functions with the kernel's module infrastructure. They don't call your functions directly — they store the function pointer in a specific ELF section that the kernel reads when loading/unloading the module.
**`hello_init` return value is critical:**
- Return `0`: module loads successfully
- Return a negative errno (e.g., `-ENOMEM`, `-ENODEV`): `insmod` fails, the module is not loaded, and the error is reported to the user
This means your init function must clean up any partially-initialized state before returning an error. If you allocate memory halfway through init and then hit an error, you must free that memory before returning the error code — there's no automatic rollback.
**`hello_exit` has no return value** because it cannot be stopped. Once `rmmod` begins unloading a module, the kernel is committed. Your exit function must complete cleanup unconditionally. If cleanup were to fail halfway, the kernel would be in an inconsistent state with no way to recover.
### The `__init` and `__exit` Annotations
These are not just documentation. `__init` places the function in the `.init.text` ELF section. After the module's `init` function runs successfully, the kernel can — and does — **free the pages containing `.init.text`** to reclaim memory. This is why you cannot call an `__init` function from elsewhere in your module after initialization completes: those pages may no longer exist.
Similarly, `__exit` places code in `.exit.text`. If a module is compiled into the kernel directly (not as a loadable module), the kernel knows the exit function will never be called, and it can discard that code at compile time.
---
## Kernel Logging: printk
Every `printf` instinct you have will try to activate. Suppress it. There is no `printf` in the kernel. There is `printk`.

> **🔑 Foundation: printk log levels and dmesg**
> 
> ## printk, Log Levels, and dmesg
**What it IS**
In userspace you use `printf()` to print output. In kernel space, there is no standard I/O, no terminal attached, no `stdout`. Instead, the kernel provides **`printk()`** — the kernel's logging function. Messages written with `printk()` go into an in-memory ring buffer managed by the kernel, and from there they can be read by userspace tools.
**`dmesg`** is the primary tool for reading that ring buffer:
```bash
dmesg          # dump everything
dmesg -w       # follow in real time (like tail -f)
dmesg -T       # show human-readable timestamps
dmesg --level=err,warn  # filter by level
```
On systemd systems, `journalctl -k` is equivalent and adds persistent storage across reboots.
### Log Levels
Every `printk()` call should include a **log level** macro prepended to the format string. These levels let the kernel and log daemons filter, route, and display messages appropriately:
| Macro | Level Number | Meaning / When to use |
|---|---|---|
| `KERN_EMERG` | 0 | System is unusable; imminent crash |
| `KERN_ALERT` | 1 | Action must be taken immediately |
| `KERN_CRIT` | 2 | Critical conditions |
| `KERN_ERR` | 3 | Error conditions — something went wrong |
| `KERN_WARNING` | 4 | Warning — potential problem, not fatal |
| `KERN_NOTICE` | 5 | Normal but significant event |
| `KERN_INFO` | 6 | Informational messages |
| `KERN_DEBUG` | 7 | Debug-level messages |
Usage looks like this:
```c
printk(KERN_INFO "mymodule: loaded successfully\n");
printk(KERN_ERR "mymodule: failed to allocate buffer\n");
printk(KERN_DEBUG "mymodule: value of x = %d\n", x);
```
Modern kernels also provide convenience wrappers that automatically prepend the module name:
```c
pr_info("loaded successfully\n");   // equivalent to printk(KERN_INFO "...")
pr_err("failed to allocate\n");
pr_debug("x = %d\n", x);           // compiled out unless DEBUG is defined
pr_warn("unusual condition\n");
```
These are preferred in modern kernel code — shorter, cleaner, and `pr_debug()` is a no-op unless you compile with `DEBUG` defined or enable dynamic debug.
### The console log level
The kernel has a configurable **console log level** (`/proc/sys/kernel/printk`). Messages at a *lower number* (higher severity) than this threshold are printed immediately to the system console in addition to the ring buffer. Messages at or above the threshold go only to the ring buffer (readable via `dmesg`). By default this is typically level 4 (WARNING), so `KERN_DEBUG` and `KERN_INFO` messages appear in `dmesg` but not on screen during boot.
**WHY you need it right now**
`printk` is your **primary debugging tool** during module development. There is no `gdb` attached, no interactive debugger stepping through kernel code (unless you've set up KGDB, which is a whole project in itself). Your development feedback loop will be:
1. Write code.
2. `make` → `insmod` → trigger the code path.
3. `dmesg | tail -20` — read what your module reported.
4. `rmmod` → iterate.
Using the right log levels matters: spam `KERN_ERR` for routine info and your module becomes impossible to monitor in production. Use `pr_debug()` for verbose diagnostic messages so they can be enabled dynamically without recompiling.
**Key insight**
> Think of `printk` as writing to an in-memory logbook, not to a screen. The **ring buffer is finite** (typically 512 KB to a few MB depending on kernel config). On a busy system or after a long uptime, old messages get overwritten. If your module fires a message at panic time, it may be gone by the time you check. For critical debugging around crashes, consider `KERN_EMERG` or tools like `kdump` — but for normal development, `pr_info` / `pr_debug` + `dmesg -w` is your best friend.


![printk Log Levels and dmesg/Console Routing](./diagrams/diag-m1-printk-levels.svg)

The mechanics are simple:
```c
// Format: printk(LOG_LEVEL "format string", args...);
// Note: the log level is a string LITERAL concatenated with your format string
// This is NOT a function call with two arguments — it's C string literal concatenation
printk(KERN_INFO "hello: value is %d\n", value);
// Same as:
printk(KERN_INFO "hello: value is %d\n", value);
// Compiles to:
printk("\001" "6" "hello: value is %d\n", value);
//      ^KERN_SOH ^6 = KERN_INFO level number
```
The log levels from most to least severe:
| Macro | Number | When to use |
|-------|--------|-------------|
| `KERN_EMERG` | 0 | System is unusable |
| `KERN_ALERT` | 1 | Action must be taken immediately |
| `KERN_CRIT` | 2 | Critical condition |
| `KERN_ERR` | 3 | Error condition — use for driver errors |
| `KERN_WARNING` | 4 | Warning — use for recoverable issues |
| `KERN_NOTICE` | 5 | Normal but significant |
| `KERN_INFO` | 6 | Informational — use for status messages |
| `KERN_DEBUG` | 7 | Debug — use for verbose diagnostics |
There's also the convenience macro `pr_info("hello: loaded\n")` which automatically prepends `KERN_INFO`. Modern kernel code uses `pr_info`, `pr_err`, `pr_debug`, etc. They also prepend the module name automatically if `pr_fmt` is defined. For this milestone, `printk(KERN_INFO ...)` is explicit and clear.
**Reading the log:**
```bash
# Read the kernel ring buffer (recent messages)
dmesg
# Follow in real time (like tail -f for kernel messages)
dmesg -w
# Filter for your module
dmesg | grep hello
# Show timestamps (useful for correlating with insmod/rmmod)
dmesg -T | grep hello
```
The kernel ring buffer is a circular buffer — older messages are dropped when it fills. On a quiet development machine this rarely matters, but on a loaded production system you might miss messages. For this milestone, `dmesg | grep hello` immediately after `insmod` is sufficient.
---
## Module Parameters
Module parameters let userspace configure your module at load time without recompiling. They're also exposed via sysfs for runtime inspection and (optionally) modification.

![Module Parameters: Load-time → sysfs → Runtime](./diagrams/diag-m1-module-param-sysfs.svg)

The declaration has three parts:
```c
// 1. Declare the C variable with a default value
static int buffer_size = 4096;
// 2. Register it as a module parameter
//    module_param(variable_name, type, permissions)
//    Types: bool, charp (char pointer), int, uint, long, ulong
//    Permissions: sysfs file permissions (0444 = read-only, 0644 = read-write)
module_param(buffer_size, int, 0444);
// 3. Document it (shows up in modinfo output)
MODULE_PARM_DESC(buffer_size, "Size of internal buffer in bytes (default: 4096)");
```
**Load-time configuration:**
```bash
# Load with default value
sudo insmod hello.ko
# Load with custom value
sudo insmod hello.ko buffer_size=8192
# The module receives buffer_size=8192 in its init function
```
**Runtime inspection via sysfs:**
```bash
# After loading the module:
cat /sys/module/hello/parameters/buffer_size
# Output: 4096  (or whatever was set at load time)
```
**Permission choices:**
```c
// 0 — parameter is not exposed in sysfs at all (load-time only, can't inspect)
module_param(buffer_size, int, 0);
// 0444 — sysfs file is readable by owner, group, and others; not writable
module_param(buffer_size, int, 0444);
// 0644 — sysfs file is readable by all, writable by owner (root)
// Use this for parameters that should be tunable at runtime
module_param(buffer_size, int, 0644);
```
> ⚠️ **Security note**: Use `0444` unless you specifically need runtime modification. A `0666` sysfs parameter lets any unprivileged user modify kernel module state — a security hole. Parameters that control security-sensitive behavior should use `0400` or `0` to prevent even root from changing them after load (though root can always use `echo` with proper permissions).
For an `int` parameter exposed with `0644`, writing to the sysfs file changes the in-memory value of `buffer_size` while the module is running. Your module code reads `buffer_size` directly — there's no notification callback. If you need to react to parameter changes (e.g., reallocate a buffer), you need a param ops struct with a `set` callback, which is beyond this milestone.
---
## Module Metadata
The four metadata macros appear at the top of every kernel module. They store strings in special ELF sections that `modinfo` reads without loading the module into the kernel:
```c
MODULE_LICENSE("GPL");
// Other valid strings: "GPL v2", "GPL and additional rights", "Dual BSD/GPL"
// "Proprietary" taints the kernel; "" or missing also taints
MODULE_AUTHOR("Ada Lovelace <ada@example.com>");
// Multiple authors: call MODULE_AUTHOR() multiple times
MODULE_DESCRIPTION("Character device driver for tutorial purposes");
MODULE_VERSION("1.0.0");
// No enforced format; commonly "major.minor.patch"
```
Inspect without loading:
```bash
modinfo hello.ko
# Output:
# filename:       /path/to/hello.ko
# version:        0.1
# description:    Minimal hello world kernel module
# author:         Your Name <you@example.com>
# license:        GPL
# srcversion:     A1B2C3D4E5F6...
# depends:
# retpoline:      Y
# name:           hello
# vermagic:       6.8.0-45-generic SMP preempt mod_unload modversions
```
The `vermagic` line is what the kernel checks at load time. It must match `uname -r` exactly, plus the kernel configuration flags. If you compile on one machine and try to load on another with a different kernel, this check fails with `ERROR: could not insert module hello.ko: Invalid module format`.
---
## The Complete Working Module
Here's the full module with all pieces integrated, ready to compile and load:
```c
// hello.c — Complete Milestone 1 kernel module
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/moduleparam.h>
#include <linux/init.h>
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name <you@example.com>");
MODULE_DESCRIPTION("Hello world kernel module with parameter");
MODULE_VERSION("0.1");
// Buffer size parameter: readable from sysfs, not writable at runtime
static int buffer_size = 4096;
module_param(buffer_size, int, 0444);
MODULE_PARM_DESC(buffer_size, "Internal buffer size in bytes (default: 4096)");
static int __init hello_init(void)
{
    // Validate the parameter — never trust input, even from privileged users
    if (buffer_size <= 0) {
        printk(KERN_ERR "hello: invalid buffer_size %d, must be positive\n",
               buffer_size);
        return -EINVAL;  // Invalid argument — insmod will fail
    }
    if (buffer_size > 1024 * 1024) {  // Sanity-check: max 1MB
        printk(KERN_WARNING "hello: buffer_size %d is suspiciously large, "
               "clamping to 1048576\n", buffer_size);
        buffer_size = 1024 * 1024;
    }
    printk(KERN_INFO "hello: module loaded successfully, buffer_size=%d\n",
           buffer_size);
    return 0;
}
static void __exit hello_exit(void)
{
    printk(KERN_INFO "hello: module unloaded\n");
}
module_init(hello_init);
module_exit(hello_exit);
```
And the `Makefile`:
```makefile
# Kbuild instruction: build hello.c into hello.ko
obj-m += hello.o
# Kernel headers location for the running kernel
KDIR ?= /lib/modules/$(shell uname -r)/build
# Source directory
PWD := $(shell pwd)
# Default target: build the module
all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules
# Remove build artifacts
clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
```
---
## Build and Verify
```bash
# 1. Compile (add -Werror to fail on warnings)
make EXTRA_CFLAGS="-Werror"
# Expected output — the cc line is the compiler, the LD line links the module:
# CC [M]  /path/to/hello.o
# MODPOST /path/to/hello.ko
# LD [M]  /path/to/hello.ko
# 2. Inspect the .ko before loading
modinfo hello.ko
# Verify: license, author, description, version, vermagic all present
# 3. Load the module
sudo insmod hello.ko
# 4. Check the kernel log
dmesg | tail -5
# Should show: hello: module loaded successfully, buffer_size=4096
# 5. Verify the module is loaded
lsmod | grep hello
# Should show: hello    <size>    0
# 6. Inspect the parameter
cat /sys/module/hello/parameters/buffer_size
# Output: 4096
# 7. Unload the module
sudo rmmod hello
# 8. Verify the exit message
dmesg | tail -3
# Should show: hello: module unloaded
# Full test with custom parameter:
sudo insmod hello.ko buffer_size=8192
dmesg | grep hello
# Should show: hello: module loaded successfully, buffer_size=8192
cat /sys/module/hello/parameters/buffer_size
# Output: 8192
sudo rmmod hello
```
### Common Errors and What They Mean
```bash
# Error: could not insert module hello.ko: Invalid module format
# Cause: vermagic mismatch — you compiled against different kernel headers
# Fix: ensure linux-headers-$(uname -r) is installed and KDIR points to it
# Error: could not insert module hello.ko: Unknown symbol in module
# Cause: calling GPL-only function without MODULE_LICENSE("GPL")
# Fix: add MODULE_LICENSE("GPL")
# Error: insmod: ERROR: could not insert module hello.ko: Operation not permitted
# Cause: running insmod without sudo
# Fix: use sudo
# Warning during compile: implicit declaration of function 'printk'
# Cause: missing #include <linux/kernel.h>
# Fix: add the include; never ignore warnings with -Werror
```
---
## Hardware Soul: What Happens When You Call `insmod`
Let's look at what the hardware and kernel actually do during `insmod hello.ko`, because understanding this chain explains why modules feel the way they do:
1. **Userspace reads the file**: `insmod` calls `open()` and `read()` on `hello.ko`, bringing the ELF data into userspace memory. Standard filesystem I/O, cache-friendly if the file is warm.
2. **Syscall boundary**: `insmod` calls `finit_module(fd, params, flags)` or `init_module(addr, len, params)`. This crosses from userspace to kernel — a privilege level change. The TLB doesn't flush (same physical memory), but the CPU switches from ring 3 to ring 0.
3. **Module allocation**: The kernel calls `vmalloc()` to allocate virtual memory in kernel space for the module's text and data. `vmalloc` allocates non-contiguous physical pages and maps them into contiguous kernel virtual address space. This involves TLB writes.
4. **Symbol resolution**: The kernel's module loader walks the ELF relocation table and resolves each undefined symbol (like `printk`) to the running kernel's address. This is a small, sequential scan — cache-friendly.
5. **Memory protection**: `.text` sections are marked read-only and executable. `.data` sections are read-write but not executable. This is enforced via page table attributes (no-execute bit, write-protect bit). A single function to update page tables — a few TLB invalidations.
6. **`init` function call**: The kernel calls your `hello_init()` function. This runs in process context (the `insmod` process is still running) but at ring 0. Your `printk` call writes to the kernel ring buffer (a circular buffer in kernel memory — sequential write, cache-hot).
7. **Init section reclaim**: If `__init` is used, the kernel frees the pages containing init code after the function returns. More TLB invalidations, page table updates.
**Memory footprint**: A minimal module like this consumes roughly:
- ~4KB for the `.text` section (your code)
- ~512 bytes for `.data`/`.bss` (your `buffer_size` variable and module metadata)
- Several KB for kernel bookkeeping (module struct, symbol tables)
Total: under 20KB for this hello-world module. Real drivers with complex logic are still typically under 100KB.
---
## Knowledge Cascade: What You've Unlocked
Understanding kernel modules opens a surprising number of doors:
**1. ELF and Dynamic Linking (cross-domain: compilers/linkers)**
The way `insmod` resolves undefined symbols in your `.ko` file is *exactly* the same concept as `ld.so` resolving symbols in a shared library — except it happens inside the kernel without a dynamic linker. Understanding the module loader gives you the "aha!" for how `dlopen`, `LD_PRELOAD`, and dynamic linking work at the ELF level. Both processes walk relocation tables and patch addresses into the code.
**2. sysfs as a Universal Kernel-Userspace Interface**
Your module parameters appear in `/sys/module/hello/parameters/`. This same sysfs mechanism is used throughout the kernel: `/sys/class/` for device classes, `/sys/block/` for storage devices, `/sys/bus/` for hardware buses. Every `cat /sys/class/net/eth0/speed` you've ever run works through the same pseudo-filesystem infrastructure as your `buffer_size` parameter. Understanding module params is understanding sysfs.
**3. The Kernel Address Space Layout**
Knowing that your module lives above `0xffff800000000000` on x86_64 is the prerequisite for understanding why `copy_to_user` / `copy_from_user` exist (Milestone 2's central concept). Userspace pointers like `0x7fff...` are simply *invalid addresses* in kernel context — there's no mapping for them in the kernel's page tables. The functions don't exist to be polite; they exist because there is no other way to safely cross the address space boundary.
**4. Tainted Kernel and Debugging**
When you see a kernel oops (panic output) in the wild and it says `Tainted: P` or `Tainted: G`, you now know exactly what that means. A `P` means a proprietary module was loaded; `G` means all modules are GPL. Kernel developers use this to immediately assess whether a bug report is worth investigating. Understanding taint explains why open-source firmware and GPL modules are not just philosophy — they're the prerequisite for getting kernel developers to help you when things go wrong.
**5. Forward: Everything That Follows**
The `hello_init` / `hello_exit` lifecycle you've just built is the scaffold for every subsequent milestone. In Milestone 2, `hello_init` will allocate a buffer and register a character device. In Milestone 3, it will create a `/proc` entry. In Milestone 4, it will initialize a mutex and wait queue. The init/exit contract — allocate resources on init, release them unconditionally on exit — is the single most important pattern in kernel driver development. You'll use it hundreds of times.
---
## Milestone Checklist
Before moving to Milestone 2, verify each of these manually:
```bash
# 1. Clean build with -Werror produces no warnings
make EXTRA_CFLAGS="-Werror" 2>&1 | grep -E "warning|error"
# Expected: no output
# 2. modinfo shows all metadata
modinfo hello.ko | grep -E "license|author|description|version"
# Expected: all four fields present
# 3. insmod succeeds and shows init message
sudo insmod hello.ko && dmesg | tail -3
# Expected: "hello: module loaded successfully, buffer_size=4096"
# 4. lsmod shows the module
lsmod | grep hello
# Expected: hello listed with usage count 0
# 5. sysfs parameter visible
cat /sys/module/hello/parameters/buffer_size
# Expected: 4096
# 6. Custom parameter works
sudo rmmod hello
sudo insmod hello.ko buffer_size=16384
cat /sys/module/hello/parameters/buffer_size
# Expected: 16384
# 7. rmmod shows exit message
sudo rmmod hello && dmesg | tail -2
# Expected: "hello: module unloaded"
# 8. Invalid parameter returns error
sudo insmod hello.ko buffer_size=-1
# Expected: insmod fails, dmesg shows KERN_ERR message
```
---
<!-- END_MS -->


<!-- MS_ID: build-kernel-module-m2 -->
<!-- MS_ID: build-kernel-module-m2 -->
# Milestone 2: Character Device Driver
## The Moment You've Been Building Toward
In Milestone 1, you wrote code that runs in the kernel. That was the hardest conceptual jump—crossing from userspace into ring 0. But a module that only prints to `dmesg` and exits isn't useful. The whole point of a kernel module is to *expose something to userspace*: a device, an interface, a service.
That's what this milestone is about. You're going to create `/dev/mydevice`—a real file that userspace programs can `open()`, `read()`, `write()`, and `close()`. When a program calls `write(fd, "hello", 5)` on your device, your kernel code runs. When it calls `read(fd, buf, 100)`, your kernel code fills that buffer. You're not just loading code into the kernel anymore—you're *building an interface* that bridges two worlds.
Before you write a single line, you need to understand what that bridge actually is and why it's built the way it is.

![VFS Dispatch: open('/dev/mydevice') → Your file_operations](./diagrams/diag-m2-vfs-dispatch.svg)

---
## The Revelation: Why `memcpy` Would Be a Security Hole
Here's the misconception that trips up every programmer encountering kernel code for the first time.
When userspace calls `write(fd, buf, 100)`, the userspace pointer `buf` contains a virtual address—something like `0x7ffc1a238000`. You might think: the kernel has access to all memory in the system, so it can just `memcpy` from that address into its kernel buffer. It's all RAM. Why the complexity?
Here's why that mental model is catastrophically wrong.
**First, the security problem.** A userspace program can hand the kernel *any* pointer it wants. It could pass `0xffff888000000000`—an address deep inside kernel memory, pointing at the page allocator's internal data structures. A naive kernel `memcpy` from that address would read kernel secrets directly into a buffer that the userspace program can then inspect. Alternatively, a write operation using that forged address could overwrite critical kernel structures. This is the classic category of kernel exploit: tricking the kernel into reading or writing arbitrary kernel memory through a corrupted userspace pointer.
**Second, the hardware problem.** Modern CPUs implement **SMAP** (Supervisor Mode Access Prevention) and **SMEP** (Supervisor Mode Execution Prevention). SMAP means that if kernel code (running in ring 0) attempts to directly read or write a userspace virtual address, the CPU itself generates a fault—not a software check, a hardware fault. The CPU enforces the boundary in silicon. If you try to `memcpy` from a userspace pointer in your kernel code on a SMAP-enabled CPU, the machine crashes immediately.

> **🔑 Foundation: SMAP/SMEP CPU hardware enforcement of kernel-userspace memory separation**
> 
> ## SMAP/SMEP: Hardware-Enforced Memory Separation
### What It Is
Modern x86 CPUs provide two hardware mechanisms that enforce strict boundaries between kernel and userspace memory:
**SMEP (Supervisor Mode Execution Prevention)** — When enabled, the CPU *refuses to execute* any code located in userspace pages while running in kernel mode (ring 0). If the kernel's instruction pointer ever lands on a userspace address, the CPU raises an immediate fault. This exists to stop attackers from placing shellcode in userspace and tricking the kernel into jumping to it.
**SMAP (Supervisor Mode Access Prevention)** — Stronger than SMEP, this prevents the kernel from *reading or writing* userspace memory at all, unless it explicitly unlocks access using the `stac`/`clac` instructions (which temporarily clear the AC flag in RFLAGS). Any accidental or malicious dereference of a userspace pointer in kernel context triggers a fault.
Both features are enabled by default on any modern Linux system with a supported CPU (Intel Broadwell and later, AMD roughly equivalent generations). You can verify this in `/proc/cpuinfo` by looking for `smep` and `smap` in the flags.
### Why You Need This Right Now
When writing kernel code that interacts with userspace — processing syscall arguments, handling ioctl buffers, copying data back and forth — you **cannot simply dereference a userspace pointer**. This isn't just a policy rule; the hardware will physically fault if you try. Code like this:
```c
/* WRONG — SMAP will fault here */
int val = *(int __user *)user_ptr;
```
...will crash your kernel. You *must* use the designated accessor functions:
```c
/* CORRECT */
int val;
if (copy_from_user(&val, user_ptr, sizeof(val)))
    return -EFAULT;
```
`copy_from_user()` and `copy_to_user()` internally bracket the access with `stac`/`clac` to temporarily permit the crossing, then immediately re-arm the protection. They also validate the address range. There is no legitimate shortcut around them.
### Key Mental Model
> **Userspace pointers are not pointers in kernel context — they are *addresses to be translated*, not memory you own.**
Think of SMAP/SMEP as a locked door between two rooms. The kernel lives in one room, userspace in the other. `copy_from_user()` is the official door with proper locks and ID checks. Dereferencing a raw userspace pointer is like walking through the wall — the hardware stops you cold. When you see a `__user` annotation in kernel code, it's a compile-time reminder that the annotated pointer requires door-protocol to access, never direct dereference.

**Third, the page fault problem.** Userspace memory isn't always in RAM. The virtual address `buf` might be valid (it belongs to the process), but the physical page it maps to might currently be swapped out to disk. In userspace, this is handled transparently: your code accesses the page, the CPU raises a page fault, the OS swaps the page in, and execution resumes. In kernel code, if you directly dereference a userspace pointer and that page is swapped out, you get a page fault *inside the kernel*, which is far more complicated to handle safely—and in many kernel contexts, sleeping (which swap-in requires) isn't allowed.
`copy_from_user()` and `copy_to_user()` solve all three problems simultaneously. They're not a politeness convention. They're the *only correct way* to touch userspace memory from kernel code.
[[EXPLAIN:the-copy_to_user/copy_from_user-contract|The copy_to_user/copy_from_user contract]]
---
## How the VFS Routes `/dev/mydevice` to Your Code
Before writing the device driver, you need to understand the dispatch chain that connects a userspace `open("/dev/mydevice", O_RDWR)` to your C function. This chain is the VFS (Virtual File System), and understanding it is one of the most transferable pieces of knowledge in all of systems programming.
[[EXPLAIN:major/minor-numbers-and-vfs-dispatch|Major/minor numbers and VFS dispatch]]
When the kernel boots, it maintains a global registry of character device drivers, keyed by **major number**. When you call `alloc_chrdev_region()`, you're asking the kernel to assign you a major number—a unique identifier that says "driver with this number handles these devices."
The **minor number** is yours to interpret. A driver managing 4 physical disks might register major=253 with minor numbers 0–3. When the kernel receives `open()` for `/dev/sda` (major=8, minor=0) vs `/dev/sdb` (major=8, minor=16), the same driver code runs for both—it uses the minor number to select which physical disk to talk to. In your project, you'll use a single minor number (0) because you have one device instance.

![Major/Minor Number Registry and Device Identification](./diagrams/diag-m2-major-minor-registry.svg)

The `/dev/mydevice` file you'll create is a **device special file**. Unlike regular files, it doesn't store data on disk. Its inode contains two things that matter: a flag marking it as a character device, and the major/minor number pair. When the kernel opens this file, it:
1. Reads the major/minor from the inode
2. Looks up the registered driver for that major number
3. Finds the `cdev` (character device) structure you registered
4. Retrieves your `file_operations` struct from the `cdev`
5. Calls your `.open` function pointer
From that point forward, every system call on that file descriptor dispatches through your `file_operations`. The `read()` syscall calls your `.read`. The `write()` syscall calls your `.write`. The `close()` syscall calls your `.release`. The VFS is a pure function-pointer dispatch table, and `file_operations` is that table.
This pattern—a struct of function pointers representing an interface—is the same mechanism used for pipes, sockets, `/proc` files, and `/sys` files. Once you understand `file_operations`, you understand *all* Linux I/O at the architectural level.

![file_operations Struct Layout and Function Pointer Dispatch](./diagrams/diag-m2-file-operations-struct.svg)

---
## The Three Registration Steps
Getting from "kernel module loaded" to "userspace can open /dev/mydevice" requires three distinct steps. Each step can fail independently, and a correct driver handles failure at each point with proper cleanup.

![Automatic /dev/ Node Creation: class_create → device_create → udev](./diagrams/diag-m2-devnode-creation.svg)

### Step 1: Reserve a Major/Minor Number
```c
#include <linux/fs.h>
static dev_t dev_num;  // Will hold our major:minor
// alloc_chrdev_region(result, first_minor, count, name)
// result: where to store the assigned dev_t
// first_minor: starting minor number (usually 0)
// count: how many consecutive minors to reserve
// name: appears in /proc/devices
ret = alloc_chrdev_region(&dev_num, 0, 1, "mydevice");
if (ret < 0) {
    pr_err("mydevice: failed to allocate major number: %d\n", ret);
    return ret;
}
// Extract major and minor for logging
pr_info("mydevice: registered with major=%d minor=%d\n",
        MAJOR(dev_num), MINOR(dev_num));
```
After this call, `/proc/devices` will show your driver name next to its major number:
```
Character devices:
  1 mem
  4 /dev/vc/0
...
251 mydevice
```
Why `alloc_chrdev_region` instead of `register_chrdev_region` with a hardcoded number? Because hardcoded major numbers conflict with other drivers. Numbers 1–199 are assigned by the kernel community; numbers 200–299 are for experimental drivers. Your development module might collide with another driver if you hardcode 250. Dynamic allocation picks a free number automatically.
### Step 2: Register Your Character Device
```c
#include <linux/cdev.h>
static struct cdev my_cdev;
// Initialize the cdev structure with your file_operations
cdev_init(&my_cdev, &mydevice_fops);  // fops defined later
my_cdev.owner = THIS_MODULE;
// Register the cdev with the kernel
ret = cdev_add(&my_cdev, dev_num, 1);
if (ret < 0) {
    pr_err("mydevice: failed to add cdev: %d\n", ret);
    unregister_chrdev_region(dev_num, 1);  // undo step 1
    return ret;
}
```
`cdev_add` is the moment your driver becomes *reachable* through the major/minor number. After this call, if userspace somehow opened a device file with your major/minor (via manual `mknod`), your file_operations would be invoked. The `/dev/` node doesn't exist yet—but the routing would work if it did.
### Step 3: Create the `/dev/` Node via udev
This is where the magic happens—or rather, where the kernel's device model does the work so you don't have to.
```c
#include <linux/device.h>
static struct class *mydevice_class;
static struct device *mydevice_device;
// Create a device class — appears in /sys/class/
mydevice_class = class_create(THIS_MODULE, "mydevice");
if (IS_ERR(mydevice_class)) {
    pr_err("mydevice: failed to create class\n");
    cdev_del(&my_cdev);
    unregister_chrdev_region(dev_num, 1);
    return PTR_ERR(mydevice_class);
}
// Create a device within that class — triggers udev
mydevice_device = device_create(mydevice_class, NULL, dev_num,
                                 NULL, "mydevice");
if (IS_ERR(mydevice_device)) {
    pr_err("mydevice: failed to create device\n");
    class_destroy(mydevice_class);
    cdev_del(&my_cdev);
    unregister_chrdev_region(dev_num, 1);
    return PTR_ERR(mydevice_device);
}
```
When `device_create` runs, the kernel creates entries in `/sys/class/mydevice/mydevice/`. The **udev** daemon (running in userspace) watches sysfs for these events via netlink socket. When it sees the new device, it reads the major/minor from sysfs and calls `mknod /dev/mydevice c <major> <minor>` automatically. Within milliseconds of `device_create` returning, `/dev/mydevice` exists.
This is the chain: your kernel code → sysfs entry → udev event → `/dev/` node. You never call `mknod` yourself. This is the modern approach; older drivers required manual `mknod` after `insmod`, which was error-prone and didn't persist across reboots.
> ⚠️ **`IS_ERR` and `PTR_ERR`**: Kernel functions that return pointers signal errors by returning a specially-crafted invalid pointer (a value in the range `-4096` to `-1` cast to a pointer). Never check these with `== NULL`. Use `IS_ERR(ptr)` to detect errors and `PTR_ERR(ptr)` to extract the errno. Treating an error pointer as a valid pointer and dereferencing it causes a kernel crash.
---
## The Kernel Buffer
Your device needs memory to store data. In Milestone 1, your module had no state. This module has one central piece of state: a buffer that userspace writes to and reads from.
```c
#include <linux/slab.h>  // for kmalloc/kfree
static char *kernel_buffer;
static size_t buffer_size = 4096;      // configurable via module_param
static size_t buffer_data_len = 0;     // how many bytes are currently stored
// In module_init, after parameter validation:
kernel_buffer = kzalloc(buffer_size, GFP_KERNEL);
if (!kernel_buffer) {
    pr_err("mydevice: failed to allocate %zu bytes\n", buffer_size);
    // clean up registrations, then:
    return -ENOMEM;
}
```
`kzalloc` allocates `buffer_size` bytes from the kernel's slab allocator and **zeroes the memory** before returning. This is important: uninitialized kernel memory could contain sensitive data from previous allocations (passwords, keys, other processes' data). Always use `kzalloc` for buffers that will be exposed to userspace, or explicitly zero with `memset` after `kmalloc`.

> **🔑 Foundation: kmalloc GFP flags: GFP_KERNEL vs GFP_ATOMIC and when to use each**
> 
> ## kmalloc GFP Flags: GFP_KERNEL vs GFP_ATOMIC
### What It Is
`kmalloc()` is the kernel's primary general-purpose allocator for small-to-medium memory objects (analogous to `malloc()` in userspace). Its signature is:
```c
void *kmalloc(size_t size, gfp_t flags);
```
The second argument — the **GFP (Get Free Pages) flags** — tells the allocator *how it's allowed to behave* when memory isn't immediately available. This is not a hint; it's a hard contract about what the allocator may and may not do. The two flags you'll use most are:
**`GFP_KERNEL`** — The allocator can do whatever it needs to satisfy the request. It can sleep (block the current process), trigger memory reclaim, wait for pages to be freed, swap pages out, or run background compaction. This is the "polite, patient" allocation. It has the highest chance of success. Use it whenever your code runs in a process context (syscall handlers, work queue callbacks, etc.) where sleeping is acceptable.
**`GFP_ATOMIC`** — The allocator must return *immediately*, no sleeping allowed. It draws from a reserved emergency memory pool and returns `NULL` if that pool is exhausted rather than waiting. Use this when sleeping is physically impossible: interrupt handlers, softirqs, tasklets, spinlock-held regions, or any other non-preemptible atomic context.
There are compound flags built on these (e.g., `GFP_KERNEL | __GFP_ZERO` to zero the allocation, `GFP_NOWAIT` for non-atomic-but-don't-sleep), but `GFP_KERNEL` and `GFP_ATOMIC` are the foundational choice.
### Why You Need This Right Now
Choosing the wrong flag has real consequences:
- Using `GFP_KERNEL` in an interrupt handler or while holding a spinlock **will trigger a BUG() or deadlock**. The allocator may try to sleep, the scheduler will complain, and at best you get a kernel warning; at worst a hang. The kernel explicitly checks for this and will oops if you violate it.
- Using `GFP_ATOMIC` everywhere "to be safe" is incorrect: it depletes the emergency pool (making the system fragile under memory pressure), and the allocation is *more likely to fail*, meaning you'll need to handle `NULL` returns in more places. It's not a conservative choice — it's a wasteful one.
The pattern to internalize: **before you call `kmalloc`, ask yourself "can the current execution context sleep?"**
```c
/* In a syscall handler — process context, can sleep */
buf = kmalloc(size, GFP_KERNEL);
/* In an IRQ handler or while holding a spinlock — cannot sleep */
buf = kmalloc(size, GFP_ATOMIC);
if (!buf)
    return; /* must handle failure — pool may be empty */
```
### Key Mental Model
> **GFP flags are not about the memory itself — they describe the *execution context* you're calling from.**
`GFP_KERNEL` means "I am in a context that can sleep." `GFP_ATOMIC` means "I am in a context that cannot sleep." The allocator reads these as constraints on its own behavior, not as memory type preferences. If you know whether your code can block, you know which flag to use — no further analysis required.


![kmalloc Internals: Slab Allocator and GFP Flags](./diagrams/diag-m2-kmalloc-slab.svg)

`GFP_KERNEL` means "this allocation can sleep if memory is scarce." The slab allocator may need to reclaim pages, which involves I/O and sleeping. This is fine in `module_init` (process context, no locks held). In interrupt context or while holding a spinlock, you'd use `GFP_ATOMIC` instead—which never sleeps but can fail more often.
In `module_exit`, you must free this buffer:
```c
kfree(kernel_buffer);  // safe to call with NULL pointer (no-op)
```
The `kfree(NULL)` case is intentionally a no-op in the kernel, which simplifies cleanup paths where the buffer might not have been allocated yet.
---
## The `file_operations` Struct
This struct is the heart of your driver. It maps Linux system call names to your C functions:
```c
#include <linux/fs.h>
static int     mydevice_open(struct inode *inode, struct file *filp);
static int     mydevice_release(struct inode *inode, struct file *filp);
static ssize_t mydevice_read(struct file *filp, char __user *buf,
                              size_t count, loff_t *f_pos);
static ssize_t mydevice_write(struct file *filp, const char __user *buf,
                               size_t count, loff_t *f_pos);
static const struct file_operations mydevice_fops = {
    .owner   = THIS_MODULE,
    .open    = mydevice_open,
    .release = mydevice_release,
    .read    = mydevice_read,
    .write   = mydevice_write,
};
```
Every field you don't set defaults to `NULL`. The VFS checks for `NULL` before calling—if `.llseek` is NULL, `lseek()` on your device uses a default implementation. If `.ioctl` is NULL (or rather `.unlocked_ioctl` in modern kernels), ioctl returns `-ENOTTY`. You only implement what you need.
The `__user` annotation on `buf` parameters is crucial. It's not a type modifier that changes compilation—it's a **semantic tag** for the sparse static analysis tool. It marks pointers as "this comes from userspace, do not dereference directly." Compiling with `sparse` or `smatch` will warn if you use a `__user` pointer without `copy_from_user`/`copy_to_user`. In production kernel code, ignoring these warnings is a security audit failure.
---
## Implementing `open` and `release`
These are the simplest handlers, but they establish important patterns:
```c
// Tracks how many file descriptors are currently open on this device
static atomic_t open_count = ATOMIC_INIT(0);
static int mydevice_open(struct inode *inode, struct file *filp)
{
    atomic_inc(&open_count);
    pr_info("mydevice: opened (count now %d)\n", atomic_read(&open_count));
    // try_module_get prevents rmmod while device is open
    if (!try_module_get(THIS_MODULE))
        return -ENODEV;
    return 0;  // success
}
static int mydevice_release(struct inode *inode, struct file *filp)
{
    atomic_dec(&open_count);
    pr_info("mydevice: released (count now %d)\n", atomic_read(&open_count));
    module_put(THIS_MODULE);
    return 0;
}
```
Two things to notice here. First, `atomic_t` instead of a plain `int` for the counter. Multiple processes can open your device simultaneously. If two processes call `open()` concurrently and both read, increment, and write a plain `int`, you have a race condition. `atomic_inc` is a single indivisible CPU instruction (on x86: `lock xadd`), safe without a mutex.

> **🔑 Foundation: Kernel atomic_t and why plain int isn't safe for shared counters**
> 
> ## Kernel atomic_t: Why Plain int Fails for Shared Counters
### What It Is
In the kernel, `atomic_t` is a type (wrapping a plain `int`) combined with a set of operations that guarantee **read-modify-write sequences are indivisible** — they complete entirely or not at all, with no intermediate state visible to other CPUs.
```c
atomic_t refcount;
atomic_set(&refcount, 0);
atomic_inc(&refcount);           /* increment, no return */
int val = atomic_read(&refcount);
int new = atomic_inc_return(&refcount); /* increment and return new value */
bool was_last = atomic_dec_and_test(&refcount); /* dec, return true if now 0 */
```
For 64-bit values, there's `atomic64_t`. For bitmask operations, `set_bit()`/`clear_bit()`/`test_and_set_bit()` provide equivalent atomicity on `unsigned long` bitmaps.
These functions compile down to CPU instructions with explicit lock prefixes (on x86, `LOCK ADD`, `LOCK XADD`, etc.) that assert the memory bus during the operation, preventing any other CPU from modifying the same cache line in between.
### Why a Plain `int` Breaks`
The classic mental model of `counter++` as a single operation is wrong at the machine level. It compiles to three distinct steps:
```
1. LOAD:  read counter from memory into register
2. ADD:   increment the register
3. STORE: write the register back to memory
```
On a multi-core system, two CPUs can interleave these steps:
```
CPU 0: LOAD  counter → reg0 (reads 5)
CPU 1: LOAD  counter → reg1 (reads 5)
CPU 0: ADD   reg0 = 6
CPU 1: ADD   reg1 = 6
CPU 0: STORE counter = 6
CPU 1: STORE counter = 6   ← overwrites CPU 0's write
```
Both CPUs incremented, but the counter only went from 5 to 6, not 7. This is a **lost update**, and it can happen in the kernel for reference counts, statistics counters, sequence numbers, and any shared state updated from multiple CPUs or interrupt contexts.
It gets worse: the C compiler and CPU hardware both reorder and optimize memory accesses. A `volatile int` prevents some compiler reordering but does *nothing* about CPU out-of-order execution or cache coherency. `volatile` is not a synchronization mechanism in the kernel — it's essentially banned for use as a substitute for proper atomics.
### Why You Need This Right Now
If your driver or subsystem maintains any counter that can be touched from multiple CPUs simultaneously — a reference count, an in-flight request counter, a statistics field, a flag updated in both process context and an interrupt handler — you need `atomic_t`. The bugs introduced by using plain `int` are:
- **Intermittent**: they only manifest under load, with specific CPU scheduling timing.
- **Silent**: the counter just silently gets wrong values — no crash, no warning.
- **Catastrophic for reference counts**: a reference count that hits zero when it shouldn't will free memory still in use; one that never reaches zero leaks memory forever.
A concrete kernel example: the page cache uses `atomic_t` for page reference counts. If those were plain `int`, concurrent file reads on different CPUs would silently corrupt the count, leading to use-after-free.
### Key Mental Model
> **"Atomic" means the hardware guarantees no other CPU sees a half-finished operation. `atomic_t` is not about preventing interrupts or holding locks — it's about making a single read-modify-write operation physically indivisible on the memory bus.**
Use `atomic_t` for counters and flags that are *only* updated with simple increments, decrements, or bit operations, and where you don't need to protect multi-step logic. For anything more complex (updating two related variables consistently, protecting a data structure), you still need a lock — `atomic_t` alone won't save you.

Second, `try_module_get` / `module_put`. This reference-counts your module. If `open_count > 0` when someone runs `rmmod`, the kernel refuses to unload—returning "Device or resource busy." Without this, you could have a process with an open file descriptor to your device while the module's code is unloaded, then the process calls `read()`, the VFS dispatches to your now-unloaded `.read` function pointer, and the machine crashes. `try_module_get` prevents that.
> **`inode` vs `filp`**: The `inode` represents the file itself (persistent, on disk or in-memory)—it holds the device's major/minor number and is shared across all opens. The `file` (`filp`) represents *one open file descriptor*—it holds the current position (`f_pos`), the open flags (`O_RDONLY`, `O_NONBLOCK`, etc.), and private data for this specific open. Multiple processes opening the same device get different `filp` pointers, each with their own `f_pos`, but they all share the same `inode`.
---
## Implementing `write`: From Userspace Into the Kernel
```c
static ssize_t mydevice_write(struct file *filp, const char __user *buf,
                               size_t count, loff_t *f_pos)
{
    size_t to_copy;
    unsigned long not_copied;
    // Prevent writing more than the buffer can hold
    if (count > buffer_size)
        to_copy = buffer_size;
    else
        to_copy = count;
    // Zero the buffer before writing new data
    // This ensures stale data from previous writes doesn't linger
    memset(kernel_buffer, 0, buffer_size);
    // THE CRITICAL CALL: copy bytes from userspace into kernel memory
    not_copied = copy_from_user(kernel_buffer, buf, to_copy);
    if (not_copied != 0) {
        pr_err("mydevice: copy_from_user failed, %lu bytes not copied\n",
               not_copied);
        return -EFAULT;
    }
    buffer_data_len = to_copy;
    pr_info("mydevice: wrote %zu bytes\n", to_copy);
    // Return the number of bytes actually written
    // If we silently truncated, return to_copy (not count)
    // Returning less than count tells the caller "retry the rest"
    return (ssize_t)to_copy;
}
```
**The return value contract for write** is precise and matters:
- Return a positive number: bytes successfully written. The caller may retry the remainder.
- Return 0: nothing written, but not an error (unusual for write, more common in read).
- Return a negative errno: error occurred.
When you return `to_copy` instead of `count` (because you truncated), the `write()` syscall returns that smaller number to userspace. A well-behaved caller like the shell's `echo` will see the partial write and not retry. If you needed to signal an error about truncation, you'd return `-ENOSPC` (no space left on device) or `-EFBIG` (file too large).

![copy_from_user / copy_to_user: What Actually Happens](./diagrams/diag-m2-copy-user-mechanics.svg)

### What `copy_from_user` Actually Does
Let's be precise about the sequence of operations inside `copy_from_user(dst, src, n)` on x86_64:
1. **Access validation**: The kernel checks that `src` through `src + n` falls within the calling process's valid address space (below `TASK_SIZE_MAX`). This catches forged kernel-space pointers immediately.
2. **SMAP disable**: On SMAP-enabled CPUs, the kernel executes `STAC` (Set AC flag)—a single privileged instruction that temporarily allows the CPU to access userspace memory. This is a hardware permission, not a software lock.
3. **Fault-safe copy**: The kernel copies bytes using a special code path registered in the **exception table**. If a page fault occurs mid-copy (because a userspace page is swapped out), the exception handler *doesn't* crash the kernel—it allows the copy to continue after the swap-in, or terminates the copy and returns the number of bytes that weren't copied.
4. **SMAP re-enable**: `CLAC` (Clear AC flag) re-enables SMAP protection. Kernel code is once again prohibited from touching userspace memory.
5. **Return**: Returns 0 on full success, or the number of bytes that couldn't be copied.
This is why `copy_from_user` is non-trivial. It's not `memcpy` with extra steps. It's a carefully orchestrated sequence involving CPU flags, hardware fault handling, and process address space validation.
> **The return value trap**: `copy_from_user` returns the number of bytes **NOT** copied—the residual, not the amount successfully transferred. If you copy 100 bytes and 3 fail, it returns 3 (not 97). Many new kernel developers flip this logic and end up with a bug that silently ignores partial copies. The idiom is: `not_copied = copy_from_user(...); if (not_copied) return -EFAULT;`
---
## Implementing `read`: From the Kernel Into Userspace
The `read` handler is more subtle than `write` because it needs to track position—how much of the buffer has already been sent to userspace—and correctly signal EOF.
```c
static ssize_t mydevice_read(struct file *filp, char __user *buf,
                              size_t count, loff_t *f_pos)
{
    size_t available;
    size_t to_copy;
    unsigned long not_copied;
    // How many bytes remain unread?
    if (*f_pos >= buffer_data_len) {
        // EOF: all data has been read
        return 0;
    }
    available = buffer_data_len - (size_t)*f_pos;
    to_copy   = (count < available) ? count : available;
    // Copy from kernel buffer at current position into userspace
    not_copied = copy_to_user(buf, kernel_buffer + *f_pos, to_copy);
    if (not_copied != 0) {
        pr_err("mydevice: copy_to_user failed, %lu bytes not copied\n",
               not_copied);
        return -EFAULT;
    }
    // Advance the file position
    *f_pos += to_copy;
    pr_info("mydevice: read %zu bytes, f_pos now %lld\n", to_copy, *f_pos);
    return (ssize_t)to_copy;
}
```

![Data Walk: echo 'hello' > /dev/mydevice && cat /dev/mydevice](./diagrams/diag-m2-read-write-data-walk.svg)

### The `f_pos` Mechanism and Why `cat` Terminates
`f_pos` is a pointer to the file position, stored in the `struct file` for this open file descriptor. Each `read()` call receives the *current* position and is responsible for updating it to reflect how much was consumed.
The read loop that `cat` and most programs use looks like this (simplified):
```c
// This is what 'cat' effectively does in a loop:
while ((n = read(fd, buf, sizeof(buf))) > 0) {
    write(STDOUT_FILENO, buf, n);
}
// Loop terminates when read() returns 0 (EOF) or negative (error)
```
Your driver must return `0` when there's nothing left to read. If you return a positive number forever (say, always returning the full buffer regardless of position), `cat` will loop infinitely—printing your buffer over and over. This is the single most common bug in first-time kernel drivers.
The `f_pos` update also enables re-entrant reads. When `cat` calls `read()` and the kernel can only satisfy 4096 bytes at a time (a common buffer size), `cat` calls `read()` again. The next call arrives with `*f_pos = 4096`. Your driver resumes from that position. The caller doesn't have to track anything—the file descriptor remembers where it left off.
This is the exact mechanism behind `lseek()` and `pread()`/`pwrite()`. Those system calls manipulate or bypass `f_pos`. Understanding `f_pos` in a driver is understanding how seeking works at the OS level.
---
## The Complete Driver
Here's the complete, integrated driver combining everything above. Study it as a whole before testing it:
```c
// mydevice.c — Complete Milestone 2 character device driver
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>          // file_operations, alloc_chrdev_region
#include <linux/cdev.h>        // cdev_init, cdev_add
#include <linux/device.h>      // class_create, device_create
#include <linux/slab.h>        // kzalloc, kfree
#include <linux/uaccess.h>     // copy_to_user, copy_from_user
#include <linux/atomic.h>      // atomic_t
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name <you@example.com>");
MODULE_DESCRIPTION("Character device driver - Milestone 2");
MODULE_VERSION("0.2");
/* ─── Module parameters ─── */
static size_t buffer_size = 4096;
module_param(buffer_size, ulong, 0444);
MODULE_PARM_DESC(buffer_size, "Size of device buffer in bytes (default: 4096)");
/* ─── Global driver state ─── */
static dev_t          dev_num;           // major:minor pair
static struct cdev    my_cdev;           // kernel cdev structure
static struct class  *mydevice_class;    // sysfs class
static struct device *mydevice_device;   // sysfs device (triggers udev)
static char          *kernel_buffer;     // the actual data buffer
static size_t         buffer_data_len;   // bytes currently stored
static atomic_t       open_count = ATOMIC_INIT(0);
/* ─── Forward declarations ─── */
static int     mydevice_open   (struct inode *, struct file *);
static int     mydevice_release(struct inode *, struct file *);
static ssize_t mydevice_read   (struct file *, char __user *, size_t, loff_t *);
static ssize_t mydevice_write  (struct file *, const char __user *, size_t, loff_t *);
/* ─── file_operations dispatch table ─── */
static const struct file_operations mydevice_fops = {
    .owner   = THIS_MODULE,
    .open    = mydevice_open,
    .release = mydevice_release,
    .read    = mydevice_read,
    .write   = mydevice_write,
};
/* ─── open ─── */
static int mydevice_open(struct inode *inode, struct file *filp)
{
    if (!try_module_get(THIS_MODULE))
        return -ENODEV;
    atomic_inc(&open_count);
    pr_info("mydevice: open() called, open_count=%d\n",
            atomic_read(&open_count));
    return 0;
}
/* ─── release ─── */
static int mydevice_release(struct inode *inode, struct file *filp)
{
    atomic_dec(&open_count);
    pr_info("mydevice: release() called, open_count=%d\n",
            atomic_read(&open_count));
    module_put(THIS_MODULE);
    return 0;
}
/* ─── read ─── */
static ssize_t mydevice_read(struct file *filp, char __user *buf,
                              size_t count, loff_t *f_pos)
{
    size_t available, to_copy;
    unsigned long not_copied;
    if (*f_pos >= (loff_t)buffer_data_len)
        return 0;  /* EOF */
    available = buffer_data_len - (size_t)*f_pos;
    to_copy   = (count < available) ? count : available;
    not_copied = copy_to_user(buf, kernel_buffer + *f_pos, to_copy);
    if (not_copied) {
        pr_err("mydevice: copy_to_user: %lu bytes not copied\n", not_copied);
        return -EFAULT;
    }
    *f_pos += (loff_t)to_copy;
    pr_info("mydevice: read %zu bytes (f_pos=%lld)\n", to_copy, *f_pos);
    return (ssize_t)to_copy;
}
/* ─── write ─── */
static ssize_t mydevice_write(struct file *filp, const char __user *buf,
                               size_t count, loff_t *f_pos)
{
    size_t to_copy;
    unsigned long not_copied;
    /* Clamp to buffer capacity */
    to_copy = (count > buffer_size) ? buffer_size : count;
    /* Clear stale data before overwriting */
    memset(kernel_buffer, 0, buffer_size);
    not_copied = copy_from_user(kernel_buffer, buf, to_copy);
    if (not_copied) {
        pr_err("mydevice: copy_from_user: %lu bytes not copied\n", not_copied);
        return -EFAULT;
    }
    buffer_data_len = to_copy;
    /* Reset read position so the next read starts from the beginning */
    *f_pos = 0;
    pr_info("mydevice: wrote %zu bytes\n", to_copy);
    return (ssize_t)to_copy;
}
/* ─── module_init ─── */
static int __init mydevice_init(void)
{
    int ret;
    /* Validate parameter */
    if (buffer_size == 0 || buffer_size > 1024 * 1024) {
        pr_err("mydevice: invalid buffer_size %zu\n", buffer_size);
        return -EINVAL;
    }
    /* 1. Allocate data buffer */
    kernel_buffer = kzalloc(buffer_size, GFP_KERNEL);
    if (!kernel_buffer) {
        pr_err("mydevice: kzalloc(%zu) failed\n", buffer_size);
        return -ENOMEM;
    }
    buffer_data_len = 0;
    /* 2. Allocate a major/minor number */
    ret = alloc_chrdev_region(&dev_num, 0, 1, "mydevice");
    if (ret < 0) {
        pr_err("mydevice: alloc_chrdev_region failed: %d\n", ret);
        goto err_free_buf;
    }
    pr_info("mydevice: major=%d minor=%d\n", MAJOR(dev_num), MINOR(dev_num));
    /* 3. Initialize and register the cdev */
    cdev_init(&my_cdev, &mydevice_fops);
    my_cdev.owner = THIS_MODULE;
    ret = cdev_add(&my_cdev, dev_num, 1);
    if (ret < 0) {
        pr_err("mydevice: cdev_add failed: %d\n", ret);
        goto err_unreg_region;
    }
    /* 4. Create a device class (sysfs entry: /sys/class/mydevice/) */
    mydevice_class = class_create(THIS_MODULE, "mydevice");
    if (IS_ERR(mydevice_class)) {
        ret = PTR_ERR(mydevice_class);
        pr_err("mydevice: class_create failed: %d\n", ret);
        goto err_del_cdev;
    }
    /* 5. Create the device (triggers udev → /dev/mydevice) */
    mydevice_device = device_create(mydevice_class, NULL, dev_num,
                                     NULL, "mydevice");
    if (IS_ERR(mydevice_device)) {
        ret = PTR_ERR(mydevice_device);
        pr_err("mydevice: device_create failed: %d\n", ret);
        goto err_destroy_class;
    }
    pr_info("mydevice: initialized (buffer=%zu bytes), /dev/mydevice ready\n",
            buffer_size);
    return 0;
/* ─── Error unwinding (reverse order of initialization) ─── */
err_destroy_class:
    class_destroy(mydevice_class);
err_del_cdev:
    cdev_del(&my_cdev);
err_unreg_region:
    unregister_chrdev_region(dev_num, 1);
err_free_buf:
    kfree(kernel_buffer);
    return ret;
}
/* ─── module_exit ─── */
static void __exit mydevice_exit(void)
{
    device_destroy(mydevice_class, dev_num);
    class_destroy(mydevice_class);
    cdev_del(&my_cdev);
    unregister_chrdev_region(dev_num, 1);
    kfree(kernel_buffer);
    pr_info("mydevice: module unloaded\n");
}
module_init(mydevice_init);
module_exit(mydevice_exit);
```
### The Goto Error Pattern
The `goto` labels in `mydevice_init` implement **reverse-order unwinding** of initialization steps. This is the canonical kernel error handling pattern—not a bad practice to be avoided, but the *correct* and *expected* idiom in kernel code.
The rule is simple: every successful initialization step must be undone in its exact reverse order if a later step fails. If you allocate a buffer (step 1), register a region (step 2), add a cdev (step 3), and step 4 fails, your cleanup must: undo step 3, undo step 2, undo step 1. The `goto` chain implements this reverse walk precisely. Real kernel drivers in `drivers/` use this exact pattern extensively.
> **Do not use** `goto` to jump forward or into the middle of code. This pattern only works cleanly when all `goto` targets are at the end of the function and laid out in reverse initialization order. It's a well-understood convention; a kernel reviewer will recognize and expect it.
---
## The Makefile
The Kbuild Makefile from Milestone 1 works unchanged—just update the module name:
```makefile
obj-m += mydevice.o
KDIR ?= /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)
all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules
clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
```
---
## Build and Verify
```bash
# 1. Compile
make
# 2. Load the module
sudo insmod mydevice.ko
# 3. Verify the device node exists
ls -l /dev/mydevice
# crw------- 1 root root 251, 0 ... /dev/mydevice
# The 'c' prefix means character device; 251 is the major, 0 the minor
# 4. Check /proc/devices for your driver name
grep mydevice /proc/devices
# 251 mydevice
# 5. Verify the sysfs class entry
ls /sys/class/mydevice/
# mydevice/
# 6. THE CORE TEST: write then read
echo "hello kernel" | sudo tee /dev/mydevice
sudo cat /dev/mydevice
# Expected output: hello kernel
# 7. Check dmesg for your printk messages
dmesg | grep mydevice
# Should show: open, write, release (from echo/tee), then open, read, release (from cat)
# 8. Test with binary data
echo -n "ABCDEFGH" | sudo tee /dev/mydevice > /dev/null
sudo cat /dev/mydevice | xxd | head
# Expected: 41 42 43 44 45 46 47 48  ABCDEFGH
# 9. Test buffer capacity limit
python3 -c "print('x' * 8192, end='')" | sudo tee /dev/mydevice > /dev/null
# write() returns 4096 (buffer_size), not 8192 — tee may show partial write
sudo wc -c /dev/mydevice
# Note: wc on a device reads it entirely; should show 4096
# 10. Unload cleanly
sudo rmmod mydevice
ls /dev/mydevice
# ls: cannot access '/dev/mydevice': No such file or directory
# udev removes the node when device_destroy() is called
```
### Common Errors at This Milestone
```bash
# "Operation not permitted" on read/write
# Cause: /dev/mydevice is owned by root with mode 600
# Fix: sudo for testing; for persistent permissions, add a udev rule:
echo 'KERNEL=="mydevice", MODE="0666"' | sudo tee /etc/udev/rules.d/99-mydevice.rules
sudo udevadm control --reload
# cat loops infinitely without terminating
# Cause: read() never returns 0; check your f_pos >= buffer_data_len condition
# "Bad address" error from copy_to_user / copy_from_user
# Cause: passing a kernel pointer where a userspace pointer is expected (or vice versa)
# Check: make sure the __user pointer comes directly from the syscall argument
# "Device or resource busy" on rmmod
# Cause: a process still has /dev/mydevice open (check: lsof /dev/mydevice)
# module_put/try_module_get are working correctly if you see this
# dmesg shows open/release but no read/write
# Cause: permissions issue — the process can open but fails the read/write syscall
```
---
## Hardware Soul: Cache Behavior of Your Driver
Let's trace the hardware-level story of `echo "hello" > /dev/mydevice`:
**The write path:**
1. The shell forks a child process, which calls `write(fd, "hello\n", 6)` — a syscall crossing ring 3 → ring 0.
2. The kernel dispatches to `mydevice_write`. Your function stack frame lives on the kernel stack (8KB per thread on x86_64, mapped in kernel address space). Cache: **cold** on first access, **hot** after the function prolog runs.
3. `copy_from_user` copies 6 bytes from the shell's address space (`0x7fff...`) into `kernel_buffer`. The `kernel_buffer` pointer points into a slab-allocated page. This page is likely **cache-cold** if this is the first write after load. The 6-byte copy touches a single cache line (64 bytes). Cost: one L1/L2 miss on first access (~10 cycles), subsequent touches are hot.
4. `memset(kernel_buffer, 0, 4096)` before the copy writes 4096 bytes — 64 cache lines. This is a sequential write: prefetch-friendly, but still 64 cache line allocations. On a first write, this causes 64 cache misses (cold). The CPU's hardware prefetcher recognizes the sequential pattern and begins prefetching ahead.
**The read path:**
1. `cat` opens the device, calls `read(fd, buf, 4096)`.
2. The kernel dispatches to `mydevice_read`. `kernel_buffer` pages are now **cache-hot** from the recent write. The 6-byte `copy_to_user` touches one cache line — fast (~4 cycles L1 hit).
3. `f_pos` advances to 6. Next `read()` call: `*f_pos >= buffer_data_len` → return 0. `cat` sees EOF and exits.
4. TLB cost: `copy_to_user` crosses address spaces (kernel → userspace). The virtual-to-physical mapping for the userspace `buf` must be in the TLB. If `cat` just started, its pages may not be TLB-warm yet — one TLB miss before the copy, then fast.
**Memory layout of driver globals** (at 8-byte alignment, approximate):
```
kernel_buffer    → 8 bytes (pointer)
buffer_size      → 8 bytes (size_t)  
buffer_data_len  → 8 bytes (size_t)
open_count       → 4 bytes (atomic_t)
dev_num          → 4 bytes (dev_t)
my_cdev          → ~64 bytes (struct cdev with embedded kobject)
mydevice_class   → 8 bytes (pointer)
mydevice_device  → 8 bytes (pointer)
```
The four hot-path variables (`kernel_buffer`, `buffer_data_len`, `open_count`, `f_pos`) are all accessed on every read/write. If they fit in a single cache line (which they do, as globals in `.bss`), the hot path benefits from spatial locality. When one is fetched, all four arrive in the same 64-byte cache line. This is a trivial example of **data structure layout for cache efficiency**—a critical concern in high-throughput drivers.
---
## Knowledge Cascade: What You've Unlocked
Understanding character devices and `file_operations` opens a surprising web of connections.
**1. All of Linux I/O Speaks the Same Language (cross-domain: OS design)**
The `file_operations` struct you just implemented is the same interface used by *every* file-like object in Linux. `/proc` files? Same struct. Named pipes (FIFOs)? Same struct. Unix domain sockets? Same struct. The network stack exports itself through `socket_file_ops`. The device mapper (the technology behind LVM and Docker's storage) exposes block devices through a file_operations analogue. Understanding your tiny driver's dispatch table gives you the conceptual key to understanding how every I/O subsystem in Linux is organized. The kernel's I/O model is radically consistent: everything is a file, and files dispatch through function pointer tables.
**2. SMAP/SMEP: Software Ideas Become Silicon (cross-domain: CPU architecture)**
The SMAP/SMEP hardware protections you encountered in the `copy_from_user` explanation are part of a broader trend: security boundaries that used to be enforced purely in software getting baked into CPU silicon. The same philosophy produced:
- **NX/XD bits** (no-execute): pages marked non-executable in page tables; the CPU faults if you try to execute code there. This killed simple stack buffer overflow exploits.
- **SMEP**: kernel can't execute code at userspace addresses. Killed ret2usr attacks.
- **SMAP**: kernel can't *read* userspace addresses without explicit unlock. Killed a whole class of kernel data-leak exploits.
- **MPX** (now deprecated), **PKU** (protection keys), **CET** (control-flow enforcement). 
Each of these is the hardware answer to a class of software exploits. `copy_from_user`'s `STAC`/`CLAC` dance is you working with, not around, the CPU's security model.
**3. Page Faults as Control Flow (cross-domain: OS virtual memory)**
The fact that `copy_from_user` can trigger a page fault mid-copy—and the kernel handles it gracefully—is a window into how the kernel uses page faults as a general-purpose mechanism. Memory-mapped files (`mmap`) work by deliberately not loading file data into memory at mmap time. The first access to each page triggers a page fault; the fault handler loads the data from disk, maps the physical page, and returns. `copy_on_write` for `fork()` works the same way: child and parent share pages marked read-only; the first write triggers a fault that creates a private copy. Page faults aren't error conditions in these cases—they're scheduled work items, dispatched by the CPU on demand. Understanding that `copy_from_user` participates in this same mechanism means you understand the full page fault dispatch chain.
**4. `f_pos` and Seekable vs. Non-Seekable Devices**
The `f_pos` tracking you implemented is why regular files are seekable but some device files aren't. A device like `/dev/random` returns `-ESPIPE` from `lseek()` because position is meaningless—there's no underlying data to seek within. Real block devices (hard drives) *are* seekable: `f_pos` maps to a byte offset on the device. Your implementation is a seekable character device—userspace can `lseek()` to reposition within the buffer. In Milestone 4, you'll implement a stream-oriented device (pipe semantics) where `f_pos` stops being the right model. The contrast between these two semantics is the difference between `/dev/sda` and `/dev/pipe0`—same dispatch mechanism, fundamentally different position semantics.
**5. Forward: The Read/Write Foundation Enables Everything After**
Every feature in Milestones 3 and 4 builds directly on what you built here. The ioctl handler (Milestone 3) is just another function pointer in `file_operations`. The `/proc` introspection lives alongside your device but uses the same VFS dispatch. The mutex and wait queue (Milestone 4) protect `kernel_buffer` and `buffer_data_len`—the same variables you track now. The `O_NONBLOCK` path checks `filp->f_flags & O_NONBLOCK`—the same `filp` pointer you receive in every handler. The architecture you've established is complete; Milestones 3 and 4 add features, not foundations.
---
## Milestone Checklist
Verify each criterion before proceeding:
```bash
# 1. Major/minor allocated dynamically and visible in /proc/devices
grep mydevice /proc/devices
# Expected: "<major_number> mydevice" on one line
# 2. /dev/mydevice exists after insmod (automatic via udev)
ls -l /dev/mydevice
# Expected: crw------- (or similar) with major matching /proc/devices
# 3. file_operations: all four handlers compile and link
# (verified by successful insmod with no "Unknown symbol" errors)
# 4. kmalloc allocation — verify it's working
sudo insmod mydevice.ko buffer_size=0
# Expected: insmod fails with "invalid buffer_size" error in dmesg
# 5. write from userspace works
echo -n "test data" | sudo tee /dev/mydevice
dmesg | grep "wrote 9 bytes"
# Expected: log line showing 9 bytes written
# 6. read to userspace works, returns EOF
sudo cat /dev/mydevice
# Expected: "test data" — and cat terminates (doesn't loop)
# 7. f_pos tracking: partial reads advance position
# (This requires a small test program or dd)
sudo dd if=/dev/mydevice bs=4 count=1 2>/dev/null | xxd
# Expected: first 4 bytes of "test data"
# 8. echo/cat round-trip
echo "hello kernel world" | sudo tee /dev/mydevice && sudo cat /dev/mydevice
# Expected: "hello kernel world" echoed back exactly
# 9. open_count tracking in dmesg
sudo dmesg | grep -E "open_count=[0-9]+"
# Expected: open_count=1 when opened, back to 0 after release
# 10. Clean unload removes /dev/mydevice
sudo rmmod mydevice
ls /dev/mydevice 2>&1 | grep "No such file"
# Expected: device node removed by udev
```
---
<!-- END_MS -->


<!-- MS_ID: build-kernel-module-m3 -->
# Milestone 3: ioctl and /proc Interface
## The Moment the "Everything Is a File" Abstraction Breaks
Unix's most elegant insight is that everything—files, pipes, sockets, devices—speaks the same language: `open`, `read`, `write`, `close`. You just built that language for your character device. But now consider what a userspace program needs to do that doesn't fit this model:
- Resize the kernel buffer from 4KB to 64KB
- Query how many bytes are currently stored and how many opens have occurred
- Clear the buffer without reading it first
None of these are *data transfers*. They're *control operations*—commands that change the device's behavior or query its internal state. The `read`/`write` model breaks down here. You could abuse it (write a special command string that your driver interprets as a resize request), but that's fragile, untyped, and indistinguishable from legitimate data.
This is exactly why `ioctl` exists. It's the kernel's escape hatch—a system call that says "I need to do something to this file descriptor that doesn't fit read or write." Every time you've called `ioctl(fd, TIOCGWINSZ, &ws)` to get a terminal's window size, or `ioctl(fd, FIONREAD, &n)` to check how many bytes are waiting, or used `ioctl` to configure a network interface—you were using this escape hatch.

![ioctl Dispatch: Userspace ioctl() → unlocked_ioctl Handler](./diagrams/diag-m3-ioctl-dispatch-flow.svg)

In this milestone, you'll add two major features to your driver. First, an `ioctl` interface with three commands: resize the buffer, clear it, and query its status. Second, a `/proc/mydevice` entry that exposes runtime statistics—bytes used, open count, read/write call counts—readable at any time by any process with a simple `cat`.
These two features together give your driver a complete interface: `read`/`write` for data, `ioctl` for control, `/proc` for observation.
---
## The Revelation: ioctl Command Numbers Are Not Arbitrary
Here's the misconception you probably arrived with: an ioctl command number is just an integer you pick. You define `#define MY_RESIZE_CMD 1`, `#define MY_CLEAR_CMD 2`, the kernel dispatches based on the file descriptor, and your driver receives whichever number userspace passes. You pick any numbers you want.
This model is wrong in a subtle but important way, and fixing it will give you insight into how the kernel actually dispatches ioctl calls—and why collisions between drivers are a real, historical problem.
An ioctl command number is not an arbitrary integer. It is a **32-bit structured value** with four fields packed together:
```
Bits 31-30: direction  (2 bits)  — none, read, write, read+write
Bits 29-16: size       (14 bits) — size of the argument being transferred
Bits 15-8:  type       (8 bits)  — magic number identifying the driver/subsystem
Bits 7-0:   number     (8 bits)  — sequential command number within this driver
```
[[EXPLAIN:ioctl-command-number-encoding-(_iow/_ior/_iowr)|ioctl command number encoding (_IOW/_IOR/_IOWR)]]

![ioctl Command Number Bit Layout: _IOW/_IOR/_IOWR Macros](./diagrams/diag-m3-ioctl-command-encoding.svg)

The magic number (bits 15-8) is the key insight. It's an 8-bit value—typically an ASCII character—that identifies your subsystem. If you use `'M'` as your magic number, you're saying "all commands with this magic byte belong to my driver." If another driver also chooses `'M'`, and userspace accidentally calls ioctl on the wrong device with one of your command numbers, the other driver's handler receives a command it doesn't understand. With command validation (returning `-ENOTTY` for unknown commands), this is caught. Without it, you could execute the wrong driver's command on the wrong device—the classic driver collision bug.
The Linux kernel maintains an (incomplete, but maintained) registry of magic numbers in `Documentation/userspace-api/ioctl/ioctl-number.rst`. For production drivers, you'd check this file and pick an unregistered magic. For your development driver, something like `'k'` (0x6B) is a reasonable choice for a learning kernel module.
The macros that build these structured command numbers:
```c
_IO(type, nr)           /* no argument */
_IOW(type, nr, data_t)  /* write: userspace sends data to kernel */
_IOR(type, nr, data_t)  /* read:  kernel sends data to userspace */
_IOWR(type, nr, data_t) /* read+write: bidirectional */
```
The naming convention for direction is from userspace's perspective: `_IOW` means userspace **writes** data to the kernel (like `write()`), `_IOR` means userspace **reads** data from the kernel (like `read()`). The `data_t` argument is the *type* of the argument, not a value—the macro uses `sizeof(data_t)` to populate the size bits.
Why encode the size? Two reasons. First, `strace` uses it: `strace` decodes ioctl calls by reading the size field from the command number to know how many bytes to display as the argument—without recompiling or knowing anything about your driver. Second, some architectures use the size and direction bits to automatically handle compat ioctl (translating between 32-bit and 64-bit userspace).
> **Structured binary interfaces (cross-domain: network protocols)** — ioctl command encoding is the same principle as TCP header bit fields or Protocol Buffers wire format: packing multiple fields into a fixed-width integer for efficient dispatch and self-description. The TCP urgent pointer field, the IP protocol field, and Ethernet ethertype all use this same "encode type + metadata in a fixed-width value" approach. Once you see this pattern in ioctl, you'll recognize it everywhere binary interfaces need to be extensible without out-of-band metadata.
---
## The Shared Header: One File, Two Worlds
Before writing any driver code, you need to design the shared header. This is the contract between your kernel module and any userspace program that talks to it. The header must compile correctly in both contexts—and the two contexts have different rules.

![Shared Header: One Header, Two Worlds (Kernel and Userspace)](./diagrams/diag-m3-shared-header.svg)

```c
/* mydevice.h — shared between kernel module and userspace programs */
/* This guard ensures the file is never included twice */
#ifndef MYDEVICE_H
#define MYDEVICE_H
/*
 * Kernel headers define __user, __kernel, etc.
 * Userspace doesn't have linux/types.h available.
 * Use __u32, __u64 types (available in both contexts via asm/types.h)
 * or plain C types that are guaranteed to be the same size.
 */
#include <linux/ioctl.h>   /* provides _IOW/_IOR/_IOWR in kernel context */
/*
 * In userspace, linux/ioctl.h is available via sys/ioctl.h.
 * The #include above will work from kernel builds.
 * In userspace builds, include this header AFTER #include <sys/ioctl.h>
 * or use the conditional below.
 */
/*
 * Magic number — identifies this driver's ioctl command set.
 * 'k' chosen for "kernel module". In production, verify uniqueness
 * against Documentation/userspace-api/ioctl/ioctl-number.rst
 */
#define MYDEVICE_IOC_MAGIC  'k'
/*
 * Structure for passing status information from kernel to userspace.
 * Use fixed-size types to ensure consistent layout on any architecture.
 */
struct mydevice_status {
    unsigned long buffer_size;   /* total buffer capacity in bytes */
    unsigned long bytes_used;    /* bytes currently stored */
    unsigned int  open_count;    /* number of currently open file descriptors */
    unsigned long read_count;    /* total read() calls since module load */
    unsigned long write_count;   /* total write() calls since module load */
};
/*
 * ioctl command definitions.
 * Convention: MYDEVICE_IOC_<VERB> for commands that take no argument,
 *             MYDEVICE_IOC_<NOUN> for commands that transfer a struct.
 *
 * Command 0: Resize the buffer
 *   Direction: userspace writes new size (unsigned long) to kernel
 *   _IOW: Write from userspace perspective
 */
#define MYDEVICE_IOC_RESIZE   _IOW(MYDEVICE_IOC_MAGIC, 0, unsigned long)
/*
 * Command 1: Clear the buffer (no argument needed)
 *   Direction: none — no data transfer
 *   _IO: No data
 */
#define MYDEVICE_IOC_CLEAR    _IO(MYDEVICE_IOC_MAGIC,  1)
/*
 * Command 2: Query device status
 *   Direction: kernel sends struct mydevice_status to userspace
 *   _IOR: Read from userspace perspective
 */
#define MYDEVICE_IOC_STATUS   _IOR(MYDEVICE_IOC_MAGIC, 2, struct mydevice_status)
/*
 * Maximum valid ioctl number for range-checking.
 * Used with _IOC_NR() to validate command numbers in the handler.
 */
#define MYDEVICE_IOC_MAXNR    2
#endif /* MYDEVICE_H */
```
Notice the fixed-size types in `struct mydevice_status`. When sharing a struct across kernel and userspace, you need to be careful about:
- **Type size**: `unsigned long` is 8 bytes on 64-bit kernels but 4 bytes on 32-bit. If a 32-bit userspace program talks to a 64-bit kernel through your ioctl, the struct layout mismatches. For a learning driver, `unsigned long` is fine—your test program and kernel share the same ABI. For production drivers, you'd use `__u64`/`__u32` from `<linux/types.h>` (kernel) or `<stdint.h>` (userspace) to guarantee identical layout.
- **Padding**: C compilers insert padding between struct fields to satisfy alignment requirements. The same struct compiled with different compilers (or different optimization flags) could have different padding. Kernel drivers typically use `__attribute__((packed))` sparingly (it forces unaligned accesses, which are slow or illegal on some architectures) and instead design structs with natural alignment: order fields from largest to smallest type.
> **Header sharing between kernel and userspace — the same pattern used by Linux UAPI headers**. The `include/uapi/` directory in the Linux kernel tree exists precisely to solve this problem at scale. Every syscall argument struct, every ioctl definition, every constant that userspace needs to call into the kernel lives in `include/uapi/`. The build system copies these files to `/usr/include/linux/` where userspace programs can include them. This is how `<linux/if.h>`, `<linux/fs.h>`, and `<linux/bpf.h>` work from userspace. Your single shared header is a micro-version of the same architecture. Android's Binder IPC uses the same principle: a shared header defines the Binder protocol structs used by both the kernel driver and the libbinder userspace library.
---
## Implementing `unlocked_ioctl`
Now let's add the ioctl handler to your driver. The function signature:
```c
static long mydevice_ioctl(struct file *filp, unsigned int cmd, unsigned long arg);
```
Note `unlocked_ioctl`, not `ioctl`. The old `.ioctl` field required the kernel's Big Kernel Lock (BKL) before calling your handler—BKL is gone since Linux 2.6.39. All modern drivers use `.unlocked_ioctl`, which calls your handler without any global lock. You're responsible for your own synchronization (which you'll add properly in Milestone 4).
The `cmd` argument is the full encoded command number (the value of `MYDEVICE_IOC_RESIZE`, etc.). The `arg` argument carries either a direct integer value or a userspace pointer, depending on the command.
Add the handler to your `file_operations`:
```c
static const struct file_operations mydevice_fops = {
    .owner          = THIS_MODULE,
    .open           = mydevice_open,
    .release        = mydevice_release,
    .read           = mydevice_read,
    .write          = mydevice_write,
    .unlocked_ioctl = mydevice_ioctl,   /* ADD THIS */
};
```
Now the handler itself:
```c
static long mydevice_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    int ret = 0;
    /*
     * First layer of validation: check that the magic number matches.
     * _IOC_TYPE() extracts bits 15-8 from the command number.
     * If the magic doesn't match, this command wasn't meant for us.
     */
    if (_IOC_TYPE(cmd) != MYDEVICE_IOC_MAGIC)
        return -ENOTTY;  /* "not a typewriter" — traditional error for wrong ioctl */
    /*
     * Second layer: check the command number is within our valid range.
     * _IOC_NR() extracts bits 7-0 (the sequential command number).
     */
    if (_IOC_NR(cmd) > MYDEVICE_IOC_MAXNR)
        return -ENOTTY;
    /*
     * Third layer: validate the userspace pointer for commands that transfer data.
     * access_ok() checks that the address range [arg, arg+size) is a valid
     * userspace address — it does NOT check if the memory is currently mapped
     * (page faults during copy_from/to_user handle that), but it catches
     * obviously wrong addresses like NULL or kernel-space addresses.
     *
     * _IOC_SIZE() extracts bits 29-16 (the argument size).
     * _IOC_READ/_IOC_WRITE are direction flags from the kernel's perspective:
     *   _IOC_READ means kernel reads from userspace (i.e., _IOW from user's view)
     *   _IOC_WRITE means kernel writes to userspace (i.e., _IOR from user's view)
     */
    if (_IOC_DIR(cmd) & _IOC_READ) {
        /* Kernel will read from userspace — verify user pointer is writable by user */
        if (!access_ok((void __user *)arg, _IOC_SIZE(cmd)))
            return -EFAULT;
    }
    if (_IOC_DIR(cmd) & _IOC_WRITE) {
        /* Kernel will write to userspace — verify user pointer is readable by user */
        if (!access_ok((void __user *)arg, _IOC_SIZE(cmd)))
            return -EFAULT;
    }
    switch (cmd) {
    case MYDEVICE_IOC_RESIZE:
        ret = mydevice_ioctl_resize(arg);
        break;
    case MYDEVICE_IOC_CLEAR:
        ret = mydevice_ioctl_clear();
        break;
    case MYDEVICE_IOC_STATUS:
        ret = mydevice_ioctl_status((struct mydevice_status __user *)arg);
        break;
    default:
        /* Should be unreachable due to MAXNR check above, but be defensive */
        return -ENOTTY;
    }
    return ret;
}
```
> **`-ENOTTY` is the correct error for unknown ioctls.** Despite its historical name ("Inappropriate ioctl for device"), this is the standard errno for "this file descriptor doesn't support this ioctl command." Programs like `strace` and shell scripts that probe for ioctl support expect `-ENOTTY` for unsupported commands. Returning `-EINVAL` would technically work but violates the convention—and confuses callers that need to distinguish "bad command" from "bad argument."
### The direction confusion: `_IOC_READ` vs `_IOW`
The direction bit naming is genuinely confusing and trips up everyone. Let's be explicit:
| From userspace's view | From kernel's view | Meaning |
|---|---|---|
| `_IOW(type, nr, t)` | `_IOC_READ` set | User **w**rites to kernel; kernel **read**s from user |
| `_IOR(type, nr, t)` | `_IOC_WRITE` set | User **r**eads from kernel; kernel **write**s to user |
| `_IOWR(type, nr, t)` | Both set | Bidirectional |
The macro names (`_IOW`, `_IOR`) are from the *userspace application's* perspective. The `_IOC_READ`/`_IOC_WRITE` flag bits are from the *kernel's* perspective. They're inverses of each other. When validating a `_IOW` command (user writes to kernel, kernel reads from user), you check `_IOC_DIR(cmd) & _IOC_READ`.
---
## Implementing the Three ioctl Commands
### Command 1: Buffer Resize
```c
static int mydevice_ioctl_resize(unsigned long arg)
{
    char *new_buf;
    unsigned long new_size;
    /* Copy the new size from userspace */
    if (copy_from_user(&new_size, (unsigned long __user *)arg, sizeof(new_size)))
        return -EFAULT;
    /* Validate the requested size */
    if (new_size == 0) {
        pr_err("mydevice: resize: size cannot be zero\n");
        return -EINVAL;
    }
    if (new_size > 1024 * 1024) {  /* 1MB cap — arbitrary but sensible */
        pr_err("mydevice: resize: size %lu exceeds 1MB limit\n", new_size);
        return -EINVAL;
    }
    /* Allocate the new buffer BEFORE freeing the old one */
    new_buf = kzalloc(new_size, GFP_KERNEL);
    if (!new_buf) {
        pr_err("mydevice: resize: kzalloc(%lu) failed\n", new_size);
        return -ENOMEM;
    }
    /*
     * Copy existing data into the new buffer, up to the smaller of
     * (new_size, buffer_data_len). If shrinking and data doesn't fit,
     * truncate — data is lost. The caller is responsible for handling this.
     */
    if (buffer_data_len > 0) {
        size_t copy_len = (buffer_data_len < new_size) ? buffer_data_len : new_size;
        memcpy(new_buf, kernel_buffer, copy_len);
        buffer_data_len = copy_len;
    }
    /* Swap in the new buffer */
    kfree(kernel_buffer);
    kernel_buffer = new_buf;
    buffer_size   = new_size;
    pr_info("mydevice: buffer resized to %lu bytes\n", new_size);
    return 0;
}
```

![Buffer Resize via ioctl: Edge Cases and Data Integrity](./diagrams/diag-m3-ioctl-resize-edge-cases.svg)

The buffer resize implementation hides a critical ordering requirement: **allocate new memory before freeing old memory**. If you freed first and then `kzalloc` failed, you'd have no buffer at all—a kernel module with a null `kernel_buffer` that subsequent reads/writes would crash on. The safe pattern is always: allocate → copy → swap → free old. This is the same principle as the copy-on-write mechanism: never destroy what you have until you have a working replacement.
When the new size is *smaller* than the current data, you truncate. The acceptance criterion says "either truncate or return `-EBUSY`." Both are valid. Truncating is simpler and matches how `ftruncate(2)` works on regular files—it shrinks the data, not just the capacity. Returning `-EBUSY` is safer for callers that need to know their data was preserved. The right choice depends on your device's semantics. For this milestone, truncation is fine.
### Command 2: Buffer Clear
```c
static int mydevice_ioctl_clear(void)
{
    memset(kernel_buffer, 0, buffer_size);
    buffer_data_len = 0;
    pr_info("mydevice: buffer cleared\n");
    return 0;
}
```
This command takes no argument (`_IO`, not `_IOW`/`_IOR`). The `arg` parameter in the ioctl handler is undefined for no-argument commands—**never dereference it**. Userspace passes 0 by convention, but you can't depend on that.
### Command 3: Status Query
```c
static int mydevice_ioctl_status(struct mydevice_status __user *user_status)
{
    struct mydevice_status status;
    /*
     * Build the status struct in kernel memory first.
     * Never build it directly at the userspace pointer.
     */
    status.buffer_size  = buffer_size;
    status.bytes_used   = buffer_data_len;
    status.open_count   = atomic_read(&open_count);
    status.read_count   = atomic_read(&read_count);   /* new counter */
    status.write_count  = atomic_read(&write_count);  /* new counter */
    /* Copy the complete struct to userspace in one operation */
    if (copy_to_user(user_status, &status, sizeof(status)))
        return -EFAULT;
    return 0;
}
```
The pattern of building the struct in kernel memory and then copying it once is deliberate. You could theoretically copy field by field, but that's more `copy_to_user` calls, more chances for partial failure, and harder to reason about atomicity. A single `copy_to_user` of a complete struct is the standard approach.
To track read and write counts, add two atomics to your module globals and increment them in the respective handlers:
```c
static atomic_t read_count  = ATOMIC_INIT(0);
static atomic_t write_count = ATOMIC_INIT(0);
/* In mydevice_read(), before returning: */
atomic_inc(&read_count);
/* In mydevice_write(), before returning: */
atomic_inc(&write_count);
```
Using `atomic_t` for these statistics counters is correct—multiple CPUs can be in `mydevice_read` simultaneously (one per open file descriptor) and would race on a plain `int`. Atomics give you correct counts without a mutex (which would serialize all reads, tanking parallelism). The trade-off: you might see a read and a write updating their counts interleaved in a way that looks inconsistent in a status snapshot—one process increments `write_count` between when you read `bytes_used` and `write_count`. For statistics counters (as opposed to safety-critical state), this is acceptable.
---
## `/proc` Introspection: Why seq_file Exists
Now for the second half of this milestone: a `/proc` entry that lets anyone run `cat /proc/mydevice` and see your driver's current state.
The first instinct for implementing a `/proc` read function is this:
```c
/* NAIVE APPROACH — DO NOT USE */
static int mydevice_proc_read(char *buf, char **start, off_t offset,
                               int count, int *eof, void *data)
{
    int len = sprintf(buf, "buffer_size: %zu\nbytes_used: %zu\n",
                      buffer_size, buffer_data_len);
    *eof = 1;
    return len;
}
```
This approach has a fatal flaw. The `buf` the kernel provides is limited in size (typically 4096 bytes). If your driver's status output exceeds that, the data is silently truncated. Worse: the `offset` handling for partial reads (when `cat` reads the file in chunks) is error-prone and has historically been implemented incorrectly in thousands of kernel drivers. The old `proc_read` API is deprecated for exactly this reason.
The correct approach is `seq_file`.
[[EXPLAIN:seq_file-interface-for-/proc|seq_file interface for /proc]]

![seq_file: Why Simple proc_read Is Broken for Large Output](./diagrams/diag-m3-seq-file-mechanics.svg)

`seq_file` solves the problem by turning your "generate all output at once" callback into an **iterator**. You implement four functions:
- `start(seq, pos)` — called at the beginning; returns an iterator cookie or NULL if done
- `next(seq, v, pos)` — advances the iterator; returns next cookie or NULL if done
- `stop(seq, v)` — called when iteration is finished (cleanup)
- `show(seq, v)` — called for each item; writes output with `seq_printf`
For a driver with a single block of statistics (not a list), you use the "single" helper that wraps seq_file and collapses start/next/stop into trivial implementations. This is the right tool when your entire /proc output is one logical chunk.
```c
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
static struct proc_dir_entry *proc_entry;
/*
 * seq_show is called once to generate all output for a simple proc file.
 * seq_printf writes to the internal seq_file buffer, which handles
 * fragmentation across multiple kernel pages automatically.
 */
static int mydevice_proc_show(struct seq_file *m, void *v)
{
    seq_printf(m, "=== mydevice status ===\n");
    seq_printf(m, "buffer_size:  %zu bytes\n",  buffer_size);
    seq_printf(m, "bytes_used:   %zu bytes\n",  buffer_data_len);
    seq_printf(m, "open_count:   %d\n",          atomic_read(&open_count));
    seq_printf(m, "read_count:   %lu\n",         (unsigned long)atomic_read(&read_count));
    seq_printf(m, "write_count:  %lu\n",         (unsigned long)atomic_read(&write_count));
    return 0;
}
/*
 * proc_open is called when userspace opens /proc/mydevice.
 * single_open() is a helper that sets up a seq_file backed by our show function.
 * It handles all the internal seq_file machinery for single-output files.
 */
static int mydevice_proc_open(struct inode *inode, struct file *filp)
{
    return single_open(filp, mydevice_proc_show, NULL);
}
/*
 * The file_operations for our /proc entry.
 * seq_read, seq_lseek, and single_release are all provided by the seq_file
 * infrastructure — you don't implement them yourself.
 */
static const struct proc_ops mydevice_proc_ops = {
    .proc_open    = mydevice_proc_open,
    .proc_read    = seq_read,
    .proc_lseek   = seq_lseek,
    .proc_release = single_release,
};
```
Note `struct proc_ops` not `struct file_operations` for `/proc` entries—this is a kernel 5.6+ change. On older kernels (< 5.6), you'd use `struct file_operations` directly. The `proc_ops` struct has a smaller memory footprint since it doesn't include fields that /proc files never use (like `mmap`).
> **`/proc` as a virtual filesystem** — when you `cat /proc/meminfo`, the kernel runs C code to generate that output on the fly. There's no file on any disk. `/proc` is a **virtual filesystem** (type `proc` in `/proc/mounts`): a filesystem implementation where reads execute kernel functions rather than fetching data from storage. The same principle applies to `/sys` (sysfs), `/dev/pts` (devpts for pseudoterminals), and `debugfs`. Once you understand that `/proc/mydevice` is just a function call wearing a file's clothing, the entire `/proc` filesystem becomes transparent: `/proc/interrupts` is a C function that walks the IRQ table; `/proc/net/dev` is a function that iterates network interfaces; `/proc/pid/maps` is a function that walks a process's VMA list. They're all seq_file iterators over kernel data structures.

![/proc/mydevice Output and Statistics Tracking](./diagrams/diag-m3-proc-output-example.svg)

### Creating and Destroying the /proc Entry
In `module_init`:
```c
/* Create /proc/mydevice */
proc_entry = proc_create("mydevice", 0444, NULL, &mydevice_proc_ops);
if (!proc_entry) {
    pr_err("mydevice: proc_create failed\n");
    /* proc_create returns NULL on failure, not an ERR_PTR */
    ret = -ENOMEM;
    goto err_destroy_device;  /* unwind in reverse init order */
}
```
The `0444` permissions mean: owner, group, and world can read, nobody can write. Since `/proc/mydevice` is an observation interface, write access doesn't make sense—writes would need to call `write` on a proc file, which requires a separate `.proc_write` implementation.
The third argument `NULL` is the parent directory—`NULL` means the root of `/proc`. If you wanted `/proc/driver/mydevice`, you'd create the intermediate directory first with `proc_mkdir`.
In `module_exit` (before other cleanup):
```c
proc_remove(proc_entry);
```
`proc_remove` handles both files and directories recursively. Call it before removing the cdev and device class—if the proc file is removed after the device, a concurrent `cat /proc/mydevice` could access freed driver state.
---
## The Complete Updated Driver
Here's the full driver integrating ioctl and /proc with the existing read/write/open/release structure. The new additions are marked with `/* NEW */` comments:
```c
/* mydevice.c — Milestone 3: ioctl + /proc */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/atomic.h>
#include <linux/proc_fs.h>    /* NEW: proc_create, proc_remove */
#include <linux/seq_file.h>   /* NEW: seq_file, seq_printf, single_open */
#include "mydevice.h"          /* NEW: shared ioctl definitions */
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name <you@example.com>");
MODULE_DESCRIPTION("Character device with ioctl + /proc — Milestone 3");
MODULE_VERSION("0.3");
/* ─── Module parameters ─── */
static size_t buffer_size = 4096;
module_param(buffer_size, ulong, 0444);
MODULE_PARM_DESC(buffer_size, "Initial buffer size in bytes (default: 4096)");
/* ─── Global driver state ─── */
static dev_t          dev_num;
static struct cdev    my_cdev;
static struct class  *mydevice_class;
static struct device *mydevice_device;
static struct proc_dir_entry *proc_entry;  /* NEW */
static char   *kernel_buffer;
static size_t  buffer_data_len;
static atomic_t open_count  = ATOMIC_INIT(0);
static atomic_t read_count  = ATOMIC_INIT(0);   /* NEW */
static atomic_t write_count = ATOMIC_INIT(0);   /* NEW */
/* ─── Forward declarations ─── */
static int     mydevice_open(struct inode *, struct file *);
static int     mydevice_release(struct inode *, struct file *);
static ssize_t mydevice_read(struct file *, char __user *, size_t, loff_t *);
static ssize_t mydevice_write(struct file *, const char __user *, size_t, loff_t *);
static long    mydevice_ioctl(struct file *, unsigned int, unsigned long); /* NEW */
/* ─── ioctl helper implementations ─── NEW */
static int mydevice_ioctl_resize(unsigned long arg)
{
    char *new_buf;
    unsigned long new_size;
    if (copy_from_user(&new_size, (unsigned long __user *)arg, sizeof(new_size)))
        return -EFAULT;
    if (new_size == 0 || new_size > 1024 * 1024)
        return -EINVAL;
    new_buf = kzalloc(new_size, GFP_KERNEL);
    if (!new_buf)
        return -ENOMEM;
    if (buffer_data_len > 0) {
        size_t copy_len = (buffer_data_len < new_size) ? buffer_data_len : new_size;
        memcpy(new_buf, kernel_buffer, copy_len);
        buffer_data_len = copy_len;
    }
    kfree(kernel_buffer);
    kernel_buffer = new_buf;
    buffer_size   = new_size;
    pr_info("mydevice: buffer resized to %lu bytes\n", new_size);
    return 0;
}
static int mydevice_ioctl_clear(void)
{
    memset(kernel_buffer, 0, buffer_size);
    buffer_data_len = 0;
    pr_info("mydevice: buffer cleared\n");
    return 0;
}
static int mydevice_ioctl_status(struct mydevice_status __user *user_status)
{
    struct mydevice_status status;
    status.buffer_size  = buffer_size;
    status.bytes_used   = buffer_data_len;
    status.open_count   = (unsigned int)atomic_read(&open_count);
    status.read_count   = (unsigned long)atomic_read(&read_count);
    status.write_count  = (unsigned long)atomic_read(&write_count);
    if (copy_to_user(user_status, &status, sizeof(status)))
        return -EFAULT;
    return 0;
}
/* ─── unlocked_ioctl handler ─── NEW */
static long mydevice_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    if (_IOC_TYPE(cmd) != MYDEVICE_IOC_MAGIC)
        return -ENOTTY;
    if (_IOC_NR(cmd) > MYDEVICE_IOC_MAXNR)
        return -ENOTTY;
    if (_IOC_DIR(cmd) & _IOC_READ) {
        if (!access_ok((void __user *)arg, _IOC_SIZE(cmd)))
            return -EFAULT;
    }
    if (_IOC_DIR(cmd) & _IOC_WRITE) {
        if (!access_ok((void __user *)arg, _IOC_SIZE(cmd)))
            return -EFAULT;
    }
    switch (cmd) {
    case MYDEVICE_IOC_RESIZE:
        return mydevice_ioctl_resize(arg);
    case MYDEVICE_IOC_CLEAR:
        return mydevice_ioctl_clear();
    case MYDEVICE_IOC_STATUS:
        return mydevice_ioctl_status((struct mydevice_status __user *)arg);
    default:
        return -ENOTTY;
    }
}
/* ─── /proc handlers ─── NEW */
static int mydevice_proc_show(struct seq_file *m, void *v)
{
    seq_printf(m, "buffer_size:  %zu\n",  buffer_size);
    seq_printf(m, "bytes_used:   %zu\n",  buffer_data_len);
    seq_printf(m, "open_count:   %d\n",   atomic_read(&open_count));
    seq_printf(m, "read_count:   %d\n",   atomic_read(&read_count));
    seq_printf(m, "write_count:  %d\n",   atomic_read(&write_count));
    return 0;
}
static int mydevice_proc_open(struct inode *inode, struct file *filp)
{
    return single_open(filp, mydevice_proc_show, NULL);
}
static const struct proc_ops mydevice_proc_ops = {
    .proc_open    = mydevice_proc_open,
    .proc_read    = seq_read,
    .proc_lseek   = seq_lseek,
    .proc_release = single_release,
};
/* ─── file_operations (updated) ─── */
static const struct file_operations mydevice_fops = {
    .owner          = THIS_MODULE,
    .open           = mydevice_open,
    .release        = mydevice_release,
    .read           = mydevice_read,
    .write          = mydevice_write,
    .unlocked_ioctl = mydevice_ioctl,   /* NEW */
};
/* ─── open / release (unchanged) ─── */
static int mydevice_open(struct inode *inode, struct file *filp)
{
    if (!try_module_get(THIS_MODULE))
        return -ENODEV;
    atomic_inc(&open_count);
    pr_info("mydevice: open() open_count=%d\n", atomic_read(&open_count));
    return 0;
}
static int mydevice_release(struct inode *inode, struct file *filp)
{
    atomic_dec(&open_count);
    pr_info("mydevice: release() open_count=%d\n", atomic_read(&open_count));
    module_put(THIS_MODULE);
    return 0;
}
/* ─── read (updated: increment read_count) ─── */
static ssize_t mydevice_read(struct file *filp, char __user *buf,
                              size_t count, loff_t *f_pos)
{
    size_t available, to_copy;
    unsigned long not_copied;
    if (*f_pos >= (loff_t)buffer_data_len)
        return 0;
    available = buffer_data_len - (size_t)*f_pos;
    to_copy   = (count < available) ? count : available;
    not_copied = copy_to_user(buf, kernel_buffer + *f_pos, to_copy);
    if (not_copied)
        return -EFAULT;
    *f_pos += (loff_t)to_copy;
    atomic_inc(&read_count);   /* NEW */
    pr_info("mydevice: read %zu bytes\n", to_copy);
    return (ssize_t)to_copy;
}
/* ─── write (updated: increment write_count) ─── */
static ssize_t mydevice_write(struct file *filp, const char __user *buf,
                               size_t count, loff_t *f_pos)
{
    size_t to_copy;
    unsigned long not_copied;
    to_copy = (count > buffer_size) ? buffer_size : count;
    memset(kernel_buffer, 0, buffer_size);
    not_copied = copy_from_user(kernel_buffer, buf, to_copy);
    if (not_copied)
        return -EFAULT;
    buffer_data_len = to_copy;
    *f_pos = 0;
    atomic_inc(&write_count);  /* NEW */
    pr_info("mydevice: wrote %zu bytes\n", to_copy);
    return (ssize_t)to_copy;
}
/* ─── module_init ─── */
static int __init mydevice_init(void)
{
    int ret;
    if (buffer_size == 0 || buffer_size > 1024 * 1024) {
        pr_err("mydevice: invalid buffer_size %zu\n", buffer_size);
        return -EINVAL;
    }
    kernel_buffer = kzalloc(buffer_size, GFP_KERNEL);
    if (!kernel_buffer)
        return -ENOMEM;
    ret = alloc_chrdev_region(&dev_num, 0, 1, "mydevice");
    if (ret < 0)
        goto err_free_buf;
    cdev_init(&my_cdev, &mydevice_fops);
    my_cdev.owner = THIS_MODULE;
    ret = cdev_add(&my_cdev, dev_num, 1);
    if (ret < 0)
        goto err_unreg_region;
    mydevice_class = class_create(THIS_MODULE, "mydevice");
    if (IS_ERR(mydevice_class)) {
        ret = PTR_ERR(mydevice_class);
        goto err_del_cdev;
    }
    mydevice_device = device_create(mydevice_class, NULL, dev_num, NULL, "mydevice");
    if (IS_ERR(mydevice_device)) {
        ret = PTR_ERR(mydevice_device);
        goto err_destroy_class;
    }
    /* NEW: create /proc/mydevice */
    proc_entry = proc_create("mydevice", 0444, NULL, &mydevice_proc_ops);
    if (!proc_entry) {
        ret = -ENOMEM;
        goto err_destroy_device;
    }
    pr_info("mydevice: initialized, /dev/mydevice and /proc/mydevice ready\n");
    return 0;
err_destroy_device:
    device_destroy(mydevice_class, dev_num);
err_destroy_class:
    class_destroy(mydevice_class);
err_del_cdev:
    cdev_del(&my_cdev);
err_unreg_region:
    unregister_chrdev_region(dev_num, 1);
err_free_buf:
    kfree(kernel_buffer);
    return ret;
}
/* ─── module_exit ─── */
static void __exit mydevice_exit(void)
{
    proc_remove(proc_entry);           /* NEW: remove proc first */
    device_destroy(mydevice_class, dev_num);
    class_destroy(mydevice_class);
    cdev_del(&my_cdev);
    unregister_chrdev_region(dev_num, 1);
    kfree(kernel_buffer);
    pr_info("mydevice: unloaded\n");
}
module_init(mydevice_init);
module_exit(mydevice_exit);
```
The Makefile needs a small update to include the header file in the right place:
```makefile
obj-m += mydevice.o
KDIR ?= /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)
# Ensure the current directory is in the include path so mydevice.h is found
ccflags-y := -I$(PWD)
all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules
clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
```
---
## The Userspace Test Program
The acceptance criteria require a userspace test program that exercises all three ioctl commands and reads the /proc entry. This program includes your shared header—exactly the same header the kernel module uses:
```c
/* test_mydevice.c — Userspace test program for Milestone 3 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
/*
 * Include the same header used by the kernel module.
 * In userspace, linux/ioctl.h (included by mydevice.h) is available
 * via /usr/include/linux/ioctl.h — installed by linux-headers or libc-dev.
 */
#include "mydevice.h"
#define DEVICE_PATH "/dev/mydevice"
#define PROC_PATH   "/proc/mydevice"
static void print_proc_entry(void)
{
    FILE *f;
    char line[256];
    printf("\n--- /proc/mydevice ---\n");
    f = fopen(PROC_PATH, "r");
    if (!f) {
        perror("fopen /proc/mydevice");
        return;
    }
    while (fgets(line, sizeof(line), f))
        fputs(line, stdout);
    fclose(f);
    printf("----------------------\n");
}
int main(void)
{
    int fd;
    int ret;
    char buf[64];
    /* Open the device */
    fd = open(DEVICE_PATH, O_RDWR);
    if (fd < 0) {
        perror("open " DEVICE_PATH);
        fprintf(stderr, "Did you run: sudo insmod mydevice.ko ?\n");
        return EXIT_FAILURE;
    }
    printf("Opened %s successfully\n", DEVICE_PATH);
    /* Write some data first */
    const char *msg = "hello ioctl world";
    ret = write(fd, msg, strlen(msg));
    if (ret < 0) {
        perror("write");
        close(fd);
        return EXIT_FAILURE;
    }
    printf("Wrote %d bytes: \"%s\"\n", ret, msg);
    print_proc_entry();
    /* Test 1: STATUS query */
    printf("\n[TEST 1] MYDEVICE_IOC_STATUS\n");
    struct mydevice_status status;
    ret = ioctl(fd, MYDEVICE_IOC_STATUS, &status);
    if (ret < 0) {
        perror("ioctl STATUS");
        close(fd);
        return EXIT_FAILURE;
    }
    printf("  buffer_size:  %lu\n", status.buffer_size);
    printf("  bytes_used:   %lu\n", status.bytes_used);
    printf("  open_count:   %u\n",  status.open_count);
    printf("  read_count:   %lu\n", status.read_count);
    printf("  write_count:  %lu\n", status.write_count);
    /* Test 2: CLEAR */
    printf("\n[TEST 2] MYDEVICE_IOC_CLEAR\n");
    ret = ioctl(fd, MYDEVICE_IOC_CLEAR);
    if (ret < 0) {
        perror("ioctl CLEAR");
        close(fd);
        return EXIT_FAILURE;
    }
    printf("  Buffer cleared\n");
    /* Verify clear: read should return 0 bytes (EOF immediately) */
    lseek(fd, 0, SEEK_SET);  /* reset f_pos — or re-open */
    ssize_t n = read(fd, buf, sizeof(buf));
    printf("  Read after clear: %zd bytes (expected 0)\n", n);
    /* Test 3: RESIZE */
    printf("\n[TEST 3] MYDEVICE_IOC_RESIZE to 8192 bytes\n");
    unsigned long new_size = 8192;
    ret = ioctl(fd, MYDEVICE_IOC_RESIZE, &new_size);
    if (ret < 0) {
        perror("ioctl RESIZE");
        close(fd);
        return EXIT_FAILURE;
    }
    printf("  Resize succeeded\n");
    /* Verify resize via status */
    ret = ioctl(fd, MYDEVICE_IOC_STATUS, &status);
    if (ret == 0)
        printf("  New buffer_size: %lu (expected 8192)\n", status.buffer_size);
    /* Test 4: Unknown ioctl returns -ENOTTY */
    printf("\n[TEST 4] Unknown ioctl command\n");
    ret = ioctl(fd, _IO('k', 99));  /* valid magic, out-of-range nr */
    if (ret < 0 && errno == ENOTTY)
        printf("  Correctly returned ENOTTY (%d)\n", errno);
    else
        printf("  UNEXPECTED: ret=%d errno=%d\n", ret, errno);
    /* Test 5: Wrong magic number */
    printf("\n[TEST 5] Wrong magic number ioctl\n");
    ret = ioctl(fd, _IO('Z', 0));   /* wrong magic */
    if (ret < 0 && errno == ENOTTY)
        printf("  Correctly returned ENOTTY for wrong magic\n");
    else
        printf("  UNEXPECTED: ret=%d errno=%d\n", ret, errno);
    print_proc_entry();
    close(fd);
    printf("\nAll tests complete.\n");
    return EXIT_SUCCESS;
}
```
Build the test program with a simple Makefile target or directly:
```bash
gcc -Wall -Wextra -I. -o test_mydevice test_mydevice.c
```
The `-I.` flag ensures `mydevice.h` is found in the current directory. Note: userspace compilation uses `gcc`, not the kernel build system. The test program is a plain userspace binary.
---
## Build and Verify
```bash
# 1. Compile the module
make
# 2. Load it
sudo insmod mydevice.ko
# 3. Grant access to the device (or run test as root)
sudo chmod 666 /dev/mydevice
# 4. Verify /proc entry exists
cat /proc/mydevice
# Expected output:
# buffer_size:  4096
# bytes_used:   0
# open_count:   0
# read_count:   0
# write_count:  0
# 5. Write some data and re-check
echo "test" > /dev/mydevice
cat /proc/mydevice
# write_count should now be 1, bytes_used should be 5 ("test\n")
# 6. Run the test program
gcc -Wall -I. -o test_mydevice test_mydevice.c
sudo ./test_mydevice
# 7. Verify ioctl commands via strace (shows decoded command numbers!)
sudo strace -e ioctl ./test_mydevice 2>&1 | grep ioctl
# strace decodes the command number using the encoded size/direction/type bits
# You'll see something like:
# ioctl(3, _IOR('k', 2, [104]), ...) = 0
# 8. Verify /proc survives multiple reads (seq_file handles partial reads)
# Read in 1-byte chunks to stress seq_file partial-read handling:
dd if=/proc/mydevice bs=1 2>/dev/null | cat
# 9. Unload cleanly
sudo rmmod mydevice
# Verify /proc entry is gone:
cat /proc/mydevice 2>&1
# Expected: cat: /proc/mydevice: No such file or directory
```
### Common Errors at This Milestone
```bash
# "error: unknown type name 'proc_ops'"
# Cause: kernel < 5.6; use 'file_operations' instead of 'proc_ops' for /proc entries
# Fix: use #if LINUX_VERSION_CODE >= KERNEL_VERSION(5,6,0) conditional
# "ioctl: Inappropriate ioctl for device" (-ENOTTY)
# If this happens on YOUR valid commands:
# Check that mydevice.h is included by BOTH the kernel module and the test program
# Check that the magic number matches in both files
# Use: printf("cmd=0x%08x expected=0x%08x\n", cmd, MYDEVICE_IOC_RESIZE) to debug
# "/proc/mydevice: No such file or directory" immediately after insmod
# Cause: proc_create returned NULL; check dmesg for error messages
# Most common cause: name collision (another /proc/mydevice exists)
# "Bad address" on ioctl STATUS
# Cause: passing a kernel-space struct pointer as the ioctl argument
# Fix: the struct mydevice_status must be a local variable in userspace code,
#      not allocated with kernel functions
# Test program compiles but segfaults
# Likely cause: struct mydevice_status layout mismatch between
# kernel module and userspace compilation
# Verify: sizeof(struct mydevice_status) is the same in both contexts
```
---
## Hardware Soul: ioctl Through the Lens of Cache and Dispatch
Let's trace what happens at the hardware level when userspace calls `ioctl(fd, MYDEVICE_IOC_STATUS, &status)`.
**The dispatch chain:**
1. Userspace executes `syscall` instruction (Linux x86_64 ABI, syscall number 16 = `ioctl`). This is a CPU ring transition: ring 3 → ring 0, one pipeline flush, TLB not flushed (same process, same address space).
2. The kernel's `do_vfs_ioctl` function runs. It reads `cmd` (the 32-bit encoded command number). For ioctls the kernel recognizes itself (like `FIONREAD`, `FIOCLEX`), it handles them here. For device-specific commands, it calls `vfs_ioctl`, which dispatches to `file->f_op->unlocked_ioctl`—your function. This is a single indirect function call through the `file_operations` pointer. The `file_operations` struct is `const` (read-only after module load) and likely hot in cache if the device was recently used.
3. Your `mydevice_ioctl` runs. The `_IOC_TYPE(cmd)` operation is two bit-shift instructions—nanoseconds. The `access_ok` check on `arg` is a range comparison against `TASK_SIZE_MAX`—also nanoseconds, no memory access needed (just register arithmetic).
4. `mydevice_ioctl_status` assembles `struct mydevice_status` on the kernel stack. Four `atomic_read` calls—each one a memory load with an implicit `mfence` (memory fence) on x86, ensuring the counter values are coherent across CPUs. Stack memory is likely hot in L1 (your kernel stack was just in use during the ring transition).
5. `copy_to_user` copies 40 bytes (approximate size of `struct mydevice_status`) from kernel stack to the userspace pointer `arg`. SMAP: `STAC` instruction enables userspace access, `rep movsq` copies data, `CLAC` re-enables SMAP. The destination page is the userspace stack or a local variable in the test program—likely warm in TLB (it was just touched before the syscall).
6. Ring 3 return. The `status` struct is now in userspace memory.
**Total memory touches per `MYDEVICE_IOC_STATUS`:**
- `file_operations` pointer in `struct file`: 1 load (8 bytes, likely L1 hot)
- `open_count`, `read_count`, `write_count` atomics: 3 loads from `.bss` (likely L2, might share a cache line with `buffer_size` and `buffer_data_len`)
- `struct mydevice_status` assembly: 5 stores to kernel stack (L1 hot)
- `copy_to_user`: 5 stores (40 bytes) to userspace page (L1 or L2)
Estimated latency: **2–5 µs** on a modern CPU with warm caches. The dominant cost is the syscall boundary crossing, not the data manipulation. This is why high-frequency control paths (network drivers, high-performance storage) avoid ioctl and prefer memory-mapped ring buffers—zero syscall overhead per operation.
---
## Knowledge Cascade: What You've Unlocked
### The ioctl Escape Hatch and Why Modern Interfaces Avoid It
`ioctl` is powerful but has real problems. Commands are opaque integers—you can't discover what commands a device supports without reading its header file. There's no versioning mechanism; adding a field to `struct mydevice_status` changes the struct size, changes the encoded command number (because size is in the bits), and breaks all existing callers. There's no introspection; you can't ask a device "what ioctls do you support?"
These limitations have pushed Linux toward alternatives:
- **sysfs attributes**: each attribute is a separate file in `/sys/`, readable with `cat`, writable with `echo`. Self-documenting (the filename is the attribute name). Used extensively by device drivers and the PCI subsystem.
- **netlink sockets**: structured messages with attribute-value encoding (similar to TLV in protocols). Used by `iproute2`, Netfilter, and most modern kernel-userspace communication.
- **eBPF**: programs that run in the kernel, eliminating the syscall boundary for observability entirely.
Understanding ioctl's limitations explains why every new kernel subsystem since ~2010 has used netlink or sysfs instead. ioctl survives for hardware devices (where the interface is defined by the hardware spec, not Linux convention) and for performance-critical paths where the alternative overhead matters.
### seq_file and the Iterator Pattern Across Languages
seq_file's `start`/`next`/`stop`/`show` callbacks are the C implementation of the iterator pattern—the same abstraction Python implements as `__iter__`/`__next__`, Rust as the `Iterator` trait, Java as `Iterator<T>`. The problem being solved is identical: how do you generate a potentially-large sequence of items without knowing in advance how many will be consumed or how large a buffer the consumer has?
The iterator pattern answers this by separating *production* (your code, `show()`) from *consumption* (the consumer's buffer size). The seq_file infrastructure handles the mismatch: it calls your iterator, buffers output, delivers it in chunks matching whatever `read()` size the consumer requests. The same principle underlies `io::Read` in Rust's async streams, Python generators, and Java's `Stream` API. Once you see it here, you see it everywhere.
### /proc and the Virtual Filesystem Design Pattern
You just created a file that contains no data. It executes code when read. This seems like an edge case, but it's actually a central design pattern in Linux:
| Path | What "reading" does |
|---|---|
| `/proc/meminfo` | Walks `pgdat` list, sums free/used pages |
| `/proc/net/dev` | Iterates network interfaces, reads stats |
| `/proc/interrupts` | Walks IRQ descriptor table per-CPU |
| `/sys/class/net/eth0/speed` | Calls driver's `ethtool_ops.get_link_ksettings` |
| `/proc/mydevice` | Runs your `mydevice_proc_show` function |
Every entry in `/proc` and `/sys` is a virtual file—a C function wearing a filesystem entry's clothing. This design pattern is called **file as interface**: you get all of the Unix tooling (`cat`, `grep`, `watch`, shell scripts, Python `open()`) for free, without any custom protocol. When you see a metric in `/proc/interrupts`, you know *exactly* how to add your own: implement a seq_file iterator, register with `proc_create`. The mental model transfers completely.
### The Header Sharing Pattern at Scale: Linux UAPI
The `mydevice.h` you created is a micro-version of Linux's UAPI (User API) header system. Every syscall argument, every ioctl struct that crosses the kernel-userspace boundary, every constant userspace needs to communicate with the kernel, lives in `include/uapi/linux/` in the kernel tree. The kernel build system copies these verbatim to `/usr/include/linux/`. When you write `#include <linux/bpf.h>` in a userspace BPF program, you're reading a file that was generated from kernel source—the same file the kernel itself compiled against.
Android's Binder IPC uses the same approach: a shared header defines the Binder wire protocol, and both `drivers/android/binder.c` and `frameworks/native/libs/binder/` include it. FUSE does the same: `include/uapi/linux/fuse.h` is the contract between the kernel's FUSE driver and userspace FUSE servers like libfuse.
The lesson: **the shared header is an ABI contract**. Once it's shipped to userspace, changing it breaks existing programs. This is why Linux is extraordinarily conservative about modifying UAPI headers, and why adding a field to `struct mydevice_status` must always be done by adding new fields at the *end*—never inserting in the middle, never changing existing field types.
---
## Milestone Checklist
```bash
# 1. Module compiles with -Werror, no warnings
make EXTRA_CFLAGS="-Werror" 2>&1 | grep -E "warning|error"
# Expected: no output
# 2. unlocked_ioctl registered (verify via file_operations)
sudo insmod mydevice.ko
sudo cat /sys/kernel/debug/cdev/<major>  # or check via test program
# 3. All three ioctl commands work
sudo ./test_mydevice
# Expected: all TEST blocks show success
# 4. MYDEVICE_IOC_RESIZE correctly updates buffer_size
ioctl_status=$(sudo ./test_mydevice 2>&1 | grep "New buffer_size")
echo "$ioctl_status" | grep 8192
# Expected: "New buffer_size: 8192"
# 5. MYDEVICE_IOC_CLEAR zeroes buffer and resets bytes_used to 0
# (verified by test_mydevice: read after clear returns 0 bytes)
# 6. MYDEVICE_IOC_STATUS returns accurate counts
# write once, read status, verify write_count=1
# 7. Unknown ioctl returns -ENOTTY (not -EINVAL or success)
sudo ./test_mydevice 2>&1 | grep "ENOTTY"
# Expected: two ENOTTY lines (wrong nr, wrong magic)
# 8. /proc/mydevice exists and shows all five fields
cat /proc/mydevice | grep -E "buffer_size|bytes_used|open_count|read_count|write_count"
# Expected: all five fields present
# 9. /proc/mydevice handles partial reads without corruption
dd if=/proc/mydevice bs=1 2>/dev/null | wc -c
# Expected: some positive byte count (not 0, not error)
# 10. Shared header compiles in both kernel and userspace without errors
# (verified: insmod succeeds AND gcc test_mydevice.c succeeds)
# 11. Clean unload removes /proc/mydevice
sudo rmmod mydevice
cat /proc/mydevice 2>&1 | grep "No such file"
# Expected: No such file or directory
# 12. ioctl command numbers encode direction/size/magic correctly
# verify with strace
sudo strace -e ioctl ./test_mydevice 2>&1 | grep "MYDEVICE\|_IOR\|_IOW\|_IO("
```
---
<!-- END_MS -->


<!-- MS_ID: build-kernel-module-m4 -->
# Milestone 4: Concurrent Access, Blocking I/O, and Poll Support
## The Moment Everything You Knew About Concurrency Stops Being Enough
You've been writing concurrent code before this milestone. Maybe you've used `pthread_mutex_lock`, `std::mutex`, `sync.Mutex`, or similar. The mental model you've built: identify the critical section, wrap it in a lock, unlock when done, and the problem is solved. It works in userspace. It's what professionals do.
That model isn't wrong. But in the kernel, it's *incomplete* in three ways that will cause catastrophic failures if you don't understand them:
**Problem 1: There are two kinds of code, and they can't both use a mutex.** Userspace programs run in a single execution model—processes and threads, all of which can sleep. The kernel runs in two distinct execution contexts: process context (code running on behalf of a process, which *can* sleep) and interrupt context (code triggered by hardware interrupts, which *cannot* sleep under any circumstances). A mutex in interrupt context is a kernel panic waiting to happen. You need to know which context you're in before you choose a synchronization primitive.
**Problem 2: Signals don't just interrupt your music app—they interrupt your sleeping kernel code.** When a process calls `read()` and your driver puts it to sleep waiting for data, that process is inside the kernel. If the user presses Ctrl+C while waiting, a signal arrives. If you don't handle this correctly, the process cannot be killed. It becomes unkillable—not just unresponsive, but *literally immune to SIGKILL*. Understanding the `-ERESTARTSYS` contract is the difference between a well-behaved driver and a system administration nightmare.
**Problem 3: Wakeup doesn't mean the data is still there.** When multiple readers are waiting for data and a writer adds one item, all readers wake up simultaneously. Only one of them will find data—the rest need to go back to sleep. If you don't recheck the condition after waking up, you have a class of bugs that manifest as data corruption or infinite spinning. The `wait_event_interruptible` macro handles this correctly; understanding *why* it uses a loop is essential for writing any manual wait logic.
These three problems are the substance of this milestone. By the end, you'll have a driver that handles concurrent access correctly, lets processes sleep without becoming zombies, and integrates with Linux's event notification infrastructure—the same infrastructure that powers nginx, Node.js's event loop, and io_uring.

![The Corruption: What Happens Without a Mutex](./diagrams/diag-m4-concurrency-problem.svg)

---
## The State You Need to Protect
Before choosing a synchronization primitive, precisely identify what you're protecting and from whom.
Your driver has three pieces of mutable shared state:
```c
static char   *kernel_buffer;      /* the actual bytes */
static size_t  buffer_data_len;    /* how many bytes are valid */
static size_t  buffer_size;        /* total capacity (changes on resize) */
```
These three variables form an *invariant*: `buffer_data_len <= buffer_size`, and bytes at indices `[0, buffer_data_len)` in `kernel_buffer` contain valid data. If two processes modify these simultaneously without synchronization, the invariant breaks. The read handler might read `buffer_data_len = 50` while the write handler is midway through setting it to `10` after a clear—you'd copy 50 bytes when only 10 are valid, reading garbage from kernel memory.
Who can reach this state?
- **Reader processes**: calling `read()` on the device file
- **Writer processes**: calling `write()` on the device file  
- **ioctl callers**: calling `MYDEVICE_IOC_RESIZE` or `MYDEVICE_IOC_CLEAR`
All of these are system call handlers. System call handlers run in **process context**—the calling process is context-switched in, running your kernel code on its behalf, and it *can sleep*. This means you can use a mutex.
> The stat tracking variables (`read_count`, `write_count`, `open_count`) are `atomic_t` and don't need mutex protection—atomic operations are inherently safe for simple counters. The mutex protects the buffer invariant, not every variable in your module.

![Mutex vs Spinlock: When to Use Which](./diagrams/diag-m4-mutex-vs-spinlock.svg)

---
## Kernel Synchronization: Two Tools, One Rule
[[EXPLAIN:kernel-synchronization:-mutex-vs-spinlock,-process-context-vs-interrupt-context|Kernel synchronization: mutex vs spinlock, process context vs interrupt context]]
The single rule governing kernel synchronization primitive selection:
**If your critical section can sleep (calls `copy_from_user`, calls `kmalloc(GFP_KERNEL)`, waits for a condition): use a mutex.**  
**If your critical section cannot sleep (runs in interrupt context, holds a spinlock already): use a spinlock.**
Your character device handlers (`read`, `write`, `ioctl`) all run in process context. They can sleep. You use a mutex.
### Setting Up the Mutex
Add a mutex to your module globals:
```c
#include <linux/mutex.h>
static DEFINE_MUTEX(mydevice_mutex);
```
`DEFINE_MUTEX` is a macro that declares and statically initializes a `struct mutex` in one step. For dynamically allocated mutexes (e.g., inside a struct), use `mutex_init(&m)`. Never use a mutex before initializing it—the resulting behavior is undefined and typically results in a kernel oops.
The mutex API:
```c
mutex_lock(&mydevice_mutex);      /* acquire, sleep if already held */
mutex_unlock(&mydevice_mutex);    /* release */
/* Returns 0 on success, -EINTR if interrupted by a signal */
int ret = mutex_lock_interruptible(&mydevice_mutex);
```
`mutex_lock_interruptible` is the version you want in most driver code. If a signal arrives while waiting for the mutex, it returns `-EINTR` instead of sleeping indefinitely. This matters: if a process is waiting on your mutex and gets a SIGKILL, you want it to escape. `mutex_lock` would ignore the signal and keep waiting—creating exactly the unkillable process situation.
### The Critical Section Pattern
Every read and write handler gets the same structure:
```c
static ssize_t mydevice_write(struct file *filp, const char __user *buf,
                               size_t count, loff_t *f_pos)
{
    ssize_t ret;
    if (mutex_lock_interruptible(&mydevice_mutex))
        return -ERESTARTSYS;
    /* --- critical section: buffer_data_len and kernel_buffer are ours --- */
    /* ... actual work ... */
    /* --- end critical section --- */
    mutex_unlock(&mydevice_mutex);
    return ret;
}
```
Notice `-ERESTARTSYS` as the return value when interrupted. This isn't `-EINTR`. The distinction matters enormously.
> **`-ERESTARTSYS` vs `-EINTR`**: `-ERESTARTSYS` is an internal kernel signal that tells the syscall layer "this system call was interrupted—check if it should be automatically restarted." The kernel's signal handling code sees `-ERESTARTSYS` and, depending on whether the process registered its signal handler with `SA_RESTART`, either converts it to `-EINTR` (visible to userspace) or restarts the system call transparently. `-EINTR` goes directly to userspace unchanged. By returning `-ERESTARTSYS` from your handler, you correctly participate in the `SA_RESTART` contract that your prerequisite signal handler project introduced.
---
## Wait Queues and Blocking I/O
The mutex solves data corruption. But it doesn't solve a fundamental usability problem: what should `read()` do when the buffer is empty?
Option A: Return 0 immediately. But 0 from `read()` means EOF—the file is done. A process calling `read()` in a loop would terminate, thinking the stream ended, when actually data just hasn't arrived yet.
Option B: Return an error like `-EAGAIN`. This is the right answer for `O_NONBLOCK` mode. But for blocking mode (the default), the conventional behavior of all Unix I/O—pipes, sockets, terminals—is to *wait* until data is available.
Option C: Spin in a loop checking the buffer. This wastes a full CPU core polling an empty buffer. Unacceptable.
Option D: Put the process to sleep until a writer adds data. The writer then wakes up all sleeping readers. This is exactly how pipes and sockets work, and it's what you'll implement.
The kernel mechanism for this is a **wait queue**.
[[EXPLAIN:wait-queues-and-the-condition-recheck-pattern|Wait queues and the condition-recheck pattern]]

![Wait Queue: Block → Sleep → Wakeup → Recheck → Proceed](./diagrams/diag-m4-wait-queue-lifecycle.svg)

### Declaring and Initializing the Wait Queue
```c
#include <linux/wait.h>
static DECLARE_WAIT_QUEUE_HEAD(mydevice_read_queue);
static DECLARE_WAIT_QUEUE_HEAD(mydevice_write_queue);
```
You need two wait queues: one for readers waiting for data (readers sleep here when buffer is empty), and one for writers waiting for space (writers sleep here when buffer is full). Having separate queues means a write wakeup only disturbs sleeping readers, not sleeping writers—which would be pointless.
### The `wait_event_interruptible` Macro
This macro is the workhorse of blocking I/O in kernel drivers. Its signature:
```c
wait_event_interruptible(queue, condition);
```
Where `queue` is a `wait_queue_head_t` and `condition` is a C expression that evaluates to true when the sleeping process should wake up.
**What this macro actually does** (expanding the implementation):
```c
/* Conceptually: */
while (!(condition)) {
    add_to_wait_queue(&queue, current_process);
    set_current_state(TASK_INTERRUPTIBLE);
    if (signal_pending(current)) {
        remove_from_wait_queue(&queue, current_process);
        return -ERESTARTSYS;
    }
    schedule();  /* yield the CPU — we go to sleep here */
    remove_from_wait_queue(&queue, current_process);
    set_current_state(TASK_RUNNING);
    /* Loop back: recheck condition */
}
```
Three critical elements:
**1. The loop around the condition check.** After `schedule()` returns (meaning the process was woken up), the macro *doesn't assume the condition is true*. It loops back and checks again. This handles **spurious wakeups** (the kernel can occasionally wake a process without a real wakeup call) and **thundering herd** (10 readers all wake up when 1 item is added; 9 of them must go back to sleep). Without this recheck, you'd have 9 readers consuming invalid data.
**2. `TASK_INTERRUPTIBLE` state.** Setting the task state to interruptible means signals can wake the process. This is what makes Ctrl+C work while blocked in `read()`. If you used `TASK_UNINTERRUPTIBLE` instead (via `wait_event` without the `_interruptible` suffix), the process couldn't be killed. Uninterruptible sleep is the "D" state in `ps` output—the dreaded "disk sleep" state seen when a disk controller hangs.
**3. `-ERESTARTSYS` on signal.** If `signal_pending(current)` is true when the process is woken, the macro returns `-ERESTARTSYS` instead of looping. The calling code must propagate this immediately.
### The `-ERESTARTSYS` Chain

![-ERESTARTSYS: Signals Meet Blocking I/O](./diagrams/diag-m4-erestartsys-signal-flow.svg)

The `-ERESTARTSYS` return value is a promise to the kernel:
```
Your driver returns -ERESTARTSYS
    → syscall layer sees it
    → checks if SA_RESTART is set for this signal
    → if yes: restart the syscall (the process doesn't see anything)
    → if no:  convert to -EINTR, return to userspace
    → userspace code sees errno = EINTR and handles it
```
If you *ignore* `-ERESTARTSYS` and return 0 or positive bytes instead:
```
Signal arrives while read() is blocked
    → wait_event_interruptible returns -ERESTARTSYS
    → your code ignores it, returns 0 (EOF)
    → userspace thinks file ended
    → Ctrl+C appears to work (signal handler runs)
    → but the process is confused about its read state
```
Worse: if you convert `-ERESTARTSYS` to returning a partial byte count, you can corrupt the caller's data stream.
The correct pattern is absolute:
```c
if (wait_event_interruptible(mydevice_read_queue, buffer_data_len > 0))
    return -ERESTARTSYS;  /* signal received: propagate immediately */
/* Only reach here if buffer_data_len > 0 */
```
---
## The Complete Read Handler with Blocking
Here's the full blocking read implementation, integrating the mutex and wait queue:
```c
static ssize_t mydevice_read(struct file *filp, char __user *buf,
                              size_t count, loff_t *f_pos)
{
    ssize_t ret;
    size_t  to_copy;
    unsigned long not_copied;
    /*
     * If O_NONBLOCK is set, check data availability without sleeping.
     * filp->f_flags carries the open flags including O_NONBLOCK.
     */
    if (filp->f_flags & O_NONBLOCK) {
        if (buffer_data_len == 0)
            return -EAGAIN;
        /* Data is available: fall through to read it */
    } else {
        /*
         * Blocking mode: sleep until data arrives.
         *
         * NOTE: we check buffer_data_len WITHOUT holding the mutex.
         * This is safe because:
         * 1. wait_event_interruptible internally rechecks under the right
         *    memory ordering guarantees.
         * 2. The worst case of a false positive (we wake up and then
         *    discover under the mutex that data is gone) is handled by
         *    the inner mutex + recheck below.
         *
         * DO NOT hold the mutex while sleeping — that would deadlock
         * any writer trying to add data (it can't acquire the mutex
         * to call wake_up_interruptible).
         */
        if (wait_event_interruptible(mydevice_read_queue, buffer_data_len > 0))
            return -ERESTARTSYS;
    }
    /*
     * Acquire the mutex to safely access buffer state.
     * mutex_lock_interruptible: if interrupted by signal while waiting
     * for the mutex itself, propagate -ERESTARTSYS.
     */
    if (mutex_lock_interruptible(&mydevice_mutex))
        return -ERESTARTSYS;
    /*
     * Recheck condition under the mutex.
     * A concurrent reader may have consumed the data between when we
     * woke up and when we acquired the lock. This is the mandatory
     * condition recheck after any wait.
     */
    if (buffer_data_len == 0) {
        /*
         * Data is gone — another reader got it first.
         * For O_NONBLOCK, return -EAGAIN.
         * For blocking mode, ideally we'd loop back to sleep,
         * but to keep the code simple and illustrative, return -EAGAIN.
         * A production driver would restructure this as an outer loop.
         */
        mutex_unlock(&mydevice_mutex);
        return (filp->f_flags & O_NONBLOCK) ? -EAGAIN : -EAGAIN;
    }
    /* How many bytes to copy? */
    to_copy = (count < buffer_data_len) ? count : buffer_data_len;
    not_copied = copy_to_user(buf, kernel_buffer, to_copy);
    if (not_copied) {
        mutex_unlock(&mydevice_mutex);
        return -EFAULT;
    }
    /*
     * Shift remaining data to the front of the buffer.
     * This implements a simple FIFO: consumed bytes are removed,
     * unconsumed bytes stay at the beginning.
     *
     * Alternative: circular buffer (better performance but more complex).
     * For this milestone, memmove is correct and clear.
     */
    buffer_data_len -= to_copy;
    if (buffer_data_len > 0)
        memmove(kernel_buffer, kernel_buffer + to_copy, buffer_data_len);
    /*
     * Notify writers that space is now available.
     * wake_up_interruptible only wakes processes in TASK_INTERRUPTIBLE
     * state — it won't disturb processes in uninterruptible sleep.
     */
    wake_up_interruptible(&mydevice_write_queue);
    atomic_inc(&read_count);
    ret = (ssize_t)to_copy;
    mutex_unlock(&mydevice_mutex);
    pr_info("mydevice: read %zu bytes, %zu remaining\n", to_copy, buffer_data_len);
    return ret;
}
```
The critical insight here is the **lock/wait ordering**. Notice:
1. We check the condition (`buffer_data_len > 0`) *without* the mutex in `wait_event_interruptible`
2. We acquire the mutex *after* waking up
3. We recheck the condition *under* the mutex
Why don't we hold the mutex while sleeping? Because if we did:
```
Reader holds mutex, checks buffer_data_len == 0, goes to sleep (still holds mutex)
Writer tries to acquire mutex to add data → DEADLOCK
```
The writer can never add data because it can't get the mutex. The reader never wakes up because no writer can signal it. Both processes are stuck forever. This is **the classic deadlock pattern** in producer-consumer designs, and the fix is always: release the lock before sleeping.
---
## The Complete Write Handler with Wakeup
The write handler is the producer. It must:
1. Block if the buffer is full (in blocking mode) or return `-EAGAIN` (in non-blocking mode)
2. Add data to the buffer
3. Wake up sleeping readers

![O_NONBLOCK: Two Personalities of the Same read()](./diagrams/diag-m4-nonblock-vs-blocking.svg)

```c
static ssize_t mydevice_write(struct file *filp, const char __user *buf,
                               size_t count, loff_t *f_pos)
{
    ssize_t ret;
    size_t  space_available;
    size_t  to_copy;
    unsigned long not_copied;
    if (count == 0)
        return 0;
    /* Check if there's space, potentially blocking */
    if (filp->f_flags & O_NONBLOCK) {
        /* Non-blocking: is there ANY space? */
        if (buffer_data_len >= buffer_size)
            return -EAGAIN;
    } else {
        /* Blocking: sleep until space is available */
        if (wait_event_interruptible(mydevice_write_queue,
                                      buffer_data_len < buffer_size))
            return -ERESTARTSYS;
    }
    if (mutex_lock_interruptible(&mydevice_mutex))
        return -ERESTARTSYS;
    /* Recheck under mutex: buffer state may have changed while acquiring lock */
    if (buffer_data_len >= buffer_size) {
        mutex_unlock(&mydevice_mutex);
        return (filp->f_flags & O_NONBLOCK) ? -EAGAIN : -EAGAIN;
    }
    /* How much space is available? */
    space_available = buffer_size - buffer_data_len;
    to_copy = (count < space_available) ? count : space_available;
    /* Copy new data into the buffer, appending after existing data */
    not_copied = copy_from_user(kernel_buffer + buffer_data_len, buf, to_copy);
    if (not_copied) {
        mutex_unlock(&mydevice_mutex);
        return -EFAULT;
    }
    buffer_data_len += to_copy;
    /*
     * Wake up all readers waiting for data.
     * Called BEFORE releasing the mutex: the readers will wake up,
     * try to acquire the mutex, and block until we release it.
     * This is safe and correct — they won't proceed until we're done.
     *
     * wake_up_interruptible wakes ALL waiters on the queue.
     * Only the one that successfully rechecks the condition first
     * will proceed; the rest will re-evaluate and potentially re-sleep.
     */
    wake_up_interruptible(&mydevice_read_queue);
    atomic_inc(&write_count);
    ret = (ssize_t)to_copy;
    mutex_unlock(&mydevice_mutex);
    pr_info("mydevice: wrote %zu bytes, buffer now has %zu bytes\n",
            to_copy, buffer_data_len);
    return ret;
}
```
The data model here has changed from Milestone 3. Instead of overwriting the entire buffer on each write, you now append to it. This is the **pipe semantics** model: data accumulates and is consumed in FIFO order. This is the right model for a concurrent device where readers and writers can operate independently.
The `memmove` in the read handler (shifting remaining data to the front) is simple but O(n) in the amount of remaining data. For a production driver with high throughput requirements, you'd use a **circular buffer** (also called a ring buffer): maintain `head` and `tail` indices, wrap around at `buffer_size`. A circular buffer avoids copying entirely—reads advance `tail`, writes advance `head`. But a circular buffer requires more complex index arithmetic and boundary handling. For this milestone, the `memmove` approach is correct and clear.
> 🔭 **Deep Dive**: Circular buffer design for kernel drivers is covered in detail in *Linux Device Drivers, 3rd Edition* (Corbet, Rubini, Kroah-Hartman), Chapter 6, "Advanced Char Driver Operations." The kernel itself uses circular buffers extensively—the printk ring buffer, the network socket receive buffer (`sk_buff` lists), and the kernel event ring in `io_uring` all use variants of this pattern.
---
## O_NONBLOCK: Two Personalities of the Same Function
The `O_NONBLOCK` flag fundamentally changes the contract of blocking operations. When set at `open()` time (or set later via `fcntl(fd, F_SETFL, O_NONBLOCK)`), the calling process is declaring: "I refuse to block. If you can't immediately satisfy my request, tell me to try again."
The conventional errno is `-EAGAIN` (also spelled `-EWOULDBLOCK` on some systems—they're the same value on Linux). This is the currency of the entire async I/O ecosystem.
You access `O_NONBLOCK` through `filp->f_flags`:
```c
if (filp->f_flags & O_NONBLOCK) {
    /* Non-blocking semantics: return immediately */
    return -EAGAIN;
}
/* Blocking semantics: sleep */
```
This check is done before acquiring any lock and before sleeping. The flow is:
```
read() called with O_NONBLOCK
    → check buffer_data_len
    → if 0: return -EAGAIN immediately
    → if > 0: acquire mutex, read data, return bytes
read() called in blocking mode
    → check buffer_data_len
    → if 0: sleep via wait_event_interruptible
    → wake when writer adds data
    → acquire mutex, read data, return bytes
```
`-EAGAIN` is not an error in the traditional sense—it's a protocol. Programs that use non-blocking I/O expect it and handle it explicitly:
```c
/* Typical non-blocking read loop in userspace */
while (1) {
    ssize_t n = read(fd, buf, sizeof(buf));
    if (n > 0) {
        process(buf, n);
    } else if (n == 0) {
        break;  /* EOF */
    } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
        /* No data right now — come back later via poll/select/epoll */
        wait_for_event(fd);
    } else {
        perror("read");
        break;
    }
}
```
This loop is the skeleton of every event-driven server. nginx, Node.js, Redis—they all run variants of this pattern. Understanding `-EAGAIN` at the kernel driver level means you understand why these systems work the way they do.
---
## Implementing `.poll`: The Bridge to select/poll/epoll
With blocking I/O and `O_NONBLOCK` in place, you have a functional driver for most use cases. But there's a third mode: **multiplexed I/O**, where a single thread monitors many file descriptors simultaneously and acts on whichever becomes ready first.
This is the domain of `select(2)`, `poll(2)`, and `epoll(7)`. These system calls let userspace say: "Tell me when any of these file descriptors have data to read, space to write, or an error to report." They're the foundation of every scalable I/O framework.
[[EXPLAIN:poll/select-kernel-side-mechanics|poll/select kernel-side mechanics]]

![.poll Implementation: poll_wait + Mask Return](./diagrams/diag-m4-poll-mechanics.svg)

For your driver to participate in `poll`/`select`/`epoll`, you implement the `.poll` file operation:
```c
#include <linux/poll.h>
static __poll_t mydevice_poll(struct file *filp, poll_table *wait)
{
    __poll_t mask = 0;
    /*
     * poll_wait() registers this file descriptor with the wait queue.
     * When the kernel's poll machinery needs to know "tell me when
     * this fd changes state", it calls poll_wait to subscribe.
     *
     * poll_wait does NOT block. It records that this poll/select/epoll
     * call is interested in wakeups from mydevice_read_queue and
     * mydevice_write_queue. When wake_up_interruptible is called on
     * either queue, the poll infrastructure re-invokes .poll to
     * check the current mask.
     */
    poll_wait(filp, &mydevice_read_queue,  wait);
    poll_wait(filp, &mydevice_write_queue, wait);
    /*
     * Return the CURRENT readiness mask. This is sampled without
     * the mutex for two reasons:
     * 1. poll() is supposed to be a cheap "what's ready right now" check
     * 2. Locking here would deadlock with writers calling wake_up while
     *    holding the mutex in some scenarios
     *
     * A race where we return POLLIN but data disappears before read()
     * is handled by read()'s own mutex + recheck pattern — returning
     * a spurious POLLIN is safe; the subsequent read() will return
     * -EAGAIN and the caller knows to retry.
     */
    if (buffer_data_len > 0)
        mask |= POLLIN | POLLRDNORM;   /* readable: data available */
    if (buffer_data_len < buffer_size)
        mask |= POLLOUT | POLLWRNORM;  /* writable: space available */
    return mask;
}
```
And register it in `file_operations`:
```c
static const struct file_operations mydevice_fops = {
    .owner          = THIS_MODULE,
    .open           = mydevice_open,
    .release        = mydevice_release,
    .read           = mydevice_read,
    .write          = mydevice_write,
    .unlocked_ioctl = mydevice_ioctl,
    .poll           = mydevice_poll,   /* NEW */
};
```
### The Two-Phase Nature of `.poll`
The `.poll` function has two distinct jobs that happen in the same call:
**Phase 1: Subscribe** (`poll_wait`). Register interest in being notified when state changes. The `poll_table *wait` argument is the subscription mechanism—`poll_wait` adds your wait queues to the set of queues that the poll infrastructure is monitoring. When `wake_up_interruptible` fires on any of these queues, the kernel re-polls your device.
**Phase 2: Report** (the mask). Return a bitmask of what's ready *right now*. This is sampled immediately, without waiting. The mask bits:
| Bit | Meaning |
|-----|---------|
| `POLLIN` | Data available to read |
| `POLLRDNORM` | Normal priority data to read (same as POLLIN for most drivers) |
| `POLLOUT` | Space available to write |
| `POLLWRNORM` | Normal priority space to write (same as POLLOUT for most drivers) |
| `POLLERR` | Error condition (always reported even if not requested) |
| `POLLHUP` | Hangup — other end of connection closed |
**A common mistake**: returning the correct mask bits but forgetting to call `poll_wait`. If you omit `poll_wait`, the subscription never happens. `select()`/`poll()` will call your `.poll` function once, get the current mask, and if the fd isn't ready, it will never re-check it—the call blocks forever or times out, regardless of future wakeups. The function call must always do both.
**Another common mistake**: calling only `poll_wait` but returning 0 always. `select()`/`poll()` will correctly subscribe, get woken up on state changes, and then call your `.poll` again—which returns 0—and conclude nothing is ready. This causes busy-spinning: the poll machinery wakes up repeatedly but always finds nothing ready. Return the current mask accurately.

![From Your .poll to nginx's epoll: The Full Stack](./diagrams/diag-m4-poll-epoll-connection.svg)

### The Poll-to-epoll Connection
Here's the insight that connects your four-line `.poll` function to the entire event-driven networking world:
`select()`, `poll()`, and `epoll()` all ultimately call your `.poll` file operation. The difference is in the *machinery* they use to collect results:
- `select()`/`poll()`: call `.poll` on every monitored fd on each system call invocation. O(n) per call. Fine for small fd counts; breaks down at thousands of connections.
- `epoll()`: uses a different mechanism. On `epoll_ctl(EPOLL_CTL_ADD, fd, ...)`, it calls `.poll` once to register interest (the `poll_wait` phase). Subsequent state changes trigger the kernel to notify the relevant epoll instance directly, without re-scanning all fds. O(1) per event.
Your `.poll` implementation works correctly for all three. The `poll_table *wait` argument carries the subscription mechanism; `poll_wait` does the right thing whether it's called from the select path, poll path, or epoll path. You write one function; the kernel's VFS layer makes it work for all three callers.
This is why nginx can handle 50,000 simultaneous connections on a single thread. Each connection's socket has a `.poll` function (the TCP socket's poll). When data arrives, the network stack calls `wake_up_interruptible` on the socket's wait queue. The epoll infrastructure wakes up once, collects all ready events, and hands them to nginx. No scanning, no polling—pure event delivery.
> The connection from your driver's `.poll` to nginx's event loop is not metaphorical—it's the same kernel code path, just with a different `file_operations` table. Understanding your four-line function means you understand the mechanical foundation of every high-performance I/O framework on Linux.
---
## The Updated ioctl Handlers Under Mutex
Don't forget that `MYDEVICE_IOC_RESIZE` and `MYDEVICE_IOC_CLEAR` also touch the buffer. They need the mutex too:
```c
static int mydevice_ioctl_resize(unsigned long arg)
{
    char *new_buf;
    unsigned long new_size;
    int ret = 0;
    if (copy_from_user(&new_size, (unsigned long __user *)arg, sizeof(new_size)))
        return -EFAULT;
    if (new_size == 0 || new_size > 1024 * 1024)
        return -EINVAL;
    new_buf = kzalloc(new_size, GFP_KERNEL);
    if (!new_buf)
        return -ENOMEM;
    if (mutex_lock_interruptible(&mydevice_mutex)) {
        kfree(new_buf);  /* don't leak the allocation */
        return -ERESTARTSYS;
    }
    if (buffer_data_len > 0) {
        size_t copy_len = (buffer_data_len < new_size) ? buffer_data_len : new_size;
        memcpy(new_buf, kernel_buffer, copy_len);
        buffer_data_len = copy_len;
    } else {
        buffer_data_len = 0;
    }
    kfree(kernel_buffer);
    kernel_buffer = new_buf;
    buffer_size   = new_size;
    /*
     * After resize, there may be new write space — wake writers.
     * Or new read data (truncation removed some) — handled by callers
     * noticing buffer_data_len changed.
     */
    wake_up_interruptible(&mydevice_write_queue);
    wake_up_interruptible(&mydevice_read_queue);
    mutex_unlock(&mydevice_mutex);
    pr_info("mydevice: buffer resized to %lu bytes\n", new_size);
    return ret;
}
static int mydevice_ioctl_clear(void)
{
    if (mutex_lock_interruptible(&mydevice_mutex))
        return -ERESTARTSYS;
    memset(kernel_buffer, 0, buffer_size);
    buffer_data_len = 0;
    /* Clearing created space — wake any blocked writers */
    wake_up_interruptible(&mydevice_write_queue);
    mutex_unlock(&mydevice_mutex);
    pr_info("mydevice: buffer cleared\n");
    return 0;
}
```
Note the `kfree(new_buf)` before returning `-ERESTARTSYS` in the resize handler. You allocated the new buffer *before* acquiring the mutex (you want the potentially-sleeping `kzalloc` outside the critical section). If acquiring the mutex fails, you must free the allocation. Forgetting this leaks kernel memory—it doesn't crash immediately, but a kernel with a memory leak is a time bomb.

![Lock + Wait Queue Interaction: Read and Write Critical Sections](./diagrams/diag-m4-lock-ordering-read-write.svg)

---
## The Complete Milestone 4 Driver
Here is the complete, integrated driver. Read it as a coherent whole—the individual pieces have been explained above; now see how they compose:
```c
/* mydevice.c — Milestone 4: Concurrent access, blocking I/O, poll */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/atomic.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/mutex.h>       /* NEW: mutex_lock, mutex_unlock */
#include <linux/wait.h>        /* NEW: wait_queue_head_t, wait_event_interruptible */
#include <linux/poll.h>        /* NEW: poll_table, poll_wait, POLLIN etc. */
#include <linux/sched/signal.h> /* NEW: signal_pending */
#include "mydevice.h"
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name <you@example.com>");
MODULE_DESCRIPTION("Character device with concurrent access — Milestone 4");
MODULE_VERSION("0.4");
/* ─── Module parameters ─── */
static size_t buffer_size = 4096;
module_param(buffer_size, ulong, 0444);
MODULE_PARM_DESC(buffer_size, "Buffer size in bytes (default: 4096)");
/* ─── Global driver state ─── */
static dev_t          dev_num;
static struct cdev    my_cdev;
static struct class  *mydevice_class;
static struct device *mydevice_device;
static struct proc_dir_entry *proc_entry;
static char   *kernel_buffer;
static size_t  buffer_data_len;
static atomic_t open_count  = ATOMIC_INIT(0);
static atomic_t read_count  = ATOMIC_INIT(0);
static atomic_t write_count = ATOMIC_INIT(0);
/* ─── Synchronization primitives ─── */
static DEFINE_MUTEX(mydevice_mutex);
static DECLARE_WAIT_QUEUE_HEAD(mydevice_read_queue);
static DECLARE_WAIT_QUEUE_HEAD(mydevice_write_queue);
/* ─── Forward declarations ─── */
static int      mydevice_open(struct inode *, struct file *);
static int      mydevice_release(struct inode *, struct file *);
static ssize_t  mydevice_read(struct file *, char __user *, size_t, loff_t *);
static ssize_t  mydevice_write(struct file *, const char __user *, size_t, loff_t *);
static long     mydevice_ioctl(struct file *, unsigned int, unsigned long);
static __poll_t mydevice_poll(struct file *, poll_table *);
/* ─── file_operations ─── */
static const struct file_operations mydevice_fops = {
    .owner          = THIS_MODULE,
    .open           = mydevice_open,
    .release        = mydevice_release,
    .read           = mydevice_read,
    .write          = mydevice_write,
    .unlocked_ioctl = mydevice_ioctl,
    .poll           = mydevice_poll,
};
/* ─── open ─── */
static int mydevice_open(struct inode *inode, struct file *filp)
{
    if (!try_module_get(THIS_MODULE))
        return -ENODEV;
    atomic_inc(&open_count);
    pr_info("mydevice: open() open_count=%d\n", atomic_read(&open_count));
    return 0;
}
/* ─── release ─── */
static int mydevice_release(struct inode *inode, struct file *filp)
{
    atomic_dec(&open_count);
    pr_info("mydevice: release() open_count=%d\n", atomic_read(&open_count));
    module_put(THIS_MODULE);
    return 0;
}
/* ─── read ─── */
static ssize_t mydevice_read(struct file *filp, char __user *buf,
                              size_t count, loff_t *f_pos)
{
    ssize_t ret;
    size_t  to_copy;
    unsigned long not_copied;
    if (count == 0)
        return 0;
    if (filp->f_flags & O_NONBLOCK) {
        if (buffer_data_len == 0)
            return -EAGAIN;
    } else {
        if (wait_event_interruptible(mydevice_read_queue, buffer_data_len > 0))
            return -ERESTARTSYS;
    }
    if (mutex_lock_interruptible(&mydevice_mutex))
        return -ERESTARTSYS;
    /* Recheck: another reader may have consumed the data */
    if (buffer_data_len == 0) {
        mutex_unlock(&mydevice_mutex);
        return (filp->f_flags & O_NONBLOCK) ? -EAGAIN : -EAGAIN;
    }
    to_copy = (count < buffer_data_len) ? count : buffer_data_len;
    not_copied = copy_to_user(buf, kernel_buffer, to_copy);
    if (not_copied) {
        mutex_unlock(&mydevice_mutex);
        return -EFAULT;
    }
    buffer_data_len -= to_copy;
    if (buffer_data_len > 0)
        memmove(kernel_buffer, kernel_buffer + to_copy, buffer_data_len);
    wake_up_interruptible(&mydevice_write_queue);
    atomic_inc(&read_count);
    ret = (ssize_t)to_copy;
    mutex_unlock(&mydevice_mutex);
    return ret;
}
/* ─── write ─── */
static ssize_t mydevice_write(struct file *filp, const char __user *buf,
                               size_t count, loff_t *f_pos)
{
    ssize_t ret;
    size_t  space_available;
    size_t  to_copy;
    unsigned long not_copied;
    if (count == 0)
        return 0;
    if (filp->f_flags & O_NONBLOCK) {
        if (buffer_data_len >= buffer_size)
            return -EAGAIN;
    } else {
        if (wait_event_interruptible(mydevice_write_queue,
                                      buffer_data_len < buffer_size))
            return -ERESTARTSYS;
    }
    if (mutex_lock_interruptible(&mydevice_mutex))
        return -ERESTARTSYS;
    /* Recheck under mutex */
    if (buffer_data_len >= buffer_size) {
        mutex_unlock(&mydevice_mutex);
        return (filp->f_flags & O_NONBLOCK) ? -EAGAIN : -EAGAIN;
    }
    space_available = buffer_size - buffer_data_len;
    to_copy = (count < space_available) ? count : space_available;
    not_copied = copy_from_user(kernel_buffer + buffer_data_len, buf, to_copy);
    if (not_copied) {
        mutex_unlock(&mydevice_mutex);
        return -EFAULT;
    }
    buffer_data_len += to_copy;
    wake_up_interruptible(&mydevice_read_queue);
    atomic_inc(&write_count);
    ret = (ssize_t)to_copy;
    mutex_unlock(&mydevice_mutex);
    return ret;
}
/* ─── poll ─── */
static __poll_t mydevice_poll(struct file *filp, poll_table *wait)
{
    __poll_t mask = 0;
    poll_wait(filp, &mydevice_read_queue,  wait);
    poll_wait(filp, &mydevice_write_queue, wait);
    if (buffer_data_len > 0)
        mask |= POLLIN | POLLRDNORM;
    if (buffer_data_len < buffer_size)
        mask |= POLLOUT | POLLWRNORM;
    return mask;
}
/* ─── ioctl helpers ─── */
static int mydevice_ioctl_resize(unsigned long arg)
{
    char *new_buf;
    unsigned long new_size;
    if (copy_from_user(&new_size, (unsigned long __user *)arg, sizeof(new_size)))
        return -EFAULT;
    if (new_size == 0 || new_size > 1024 * 1024)
        return -EINVAL;
    new_buf = kzalloc(new_size, GFP_KERNEL);
    if (!new_buf)
        return -ENOMEM;
    if (mutex_lock_interruptible(&mydevice_mutex)) {
        kfree(new_buf);
        return -ERESTARTSYS;
    }
    if (buffer_data_len > 0) {
        size_t copy_len = (buffer_data_len < new_size) ? buffer_data_len : new_size;
        memcpy(new_buf, kernel_buffer, copy_len);
        buffer_data_len = copy_len;
    }
    kfree(kernel_buffer);
    kernel_buffer = new_buf;
    buffer_size   = new_size;
    wake_up_interruptible(&mydevice_write_queue);
    wake_up_interruptible(&mydevice_read_queue);
    mutex_unlock(&mydevice_mutex);
    pr_info("mydevice: resized to %lu bytes\n", new_size);
    return 0;
}
static int mydevice_ioctl_clear(void)
{
    if (mutex_lock_interruptible(&mydevice_mutex))
        return -ERESTARTSYS;
    memset(kernel_buffer, 0, buffer_size);
    buffer_data_len = 0;
    wake_up_interruptible(&mydevice_write_queue);
    mutex_unlock(&mydevice_mutex);
    pr_info("mydevice: buffer cleared\n");
    return 0;
}
static int mydevice_ioctl_status(struct mydevice_status __user *user_status)
{
    struct mydevice_status status;
    if (mutex_lock_interruptible(&mydevice_mutex))
        return -ERESTARTSYS;
    status.buffer_size  = buffer_size;
    status.bytes_used   = buffer_data_len;
    status.open_count   = (unsigned int)atomic_read(&open_count);
    status.read_count   = (unsigned long)atomic_read(&read_count);
    status.write_count  = (unsigned long)atomic_read(&write_count);
    mutex_unlock(&mydevice_mutex);
    if (copy_to_user(user_status, &status, sizeof(status)))
        return -EFAULT;
    return 0;
}
static long mydevice_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    if (_IOC_TYPE(cmd) != MYDEVICE_IOC_MAGIC)
        return -ENOTTY;
    if (_IOC_NR(cmd) > MYDEVICE_IOC_MAXNR)
        return -ENOTTY;
    if ((_IOC_DIR(cmd) & _IOC_READ) || (_IOC_DIR(cmd) & _IOC_WRITE)) {
        if (!access_ok((void __user *)arg, _IOC_SIZE(cmd)))
            return -EFAULT;
    }
    switch (cmd) {
    case MYDEVICE_IOC_RESIZE:
        return mydevice_ioctl_resize(arg);
    case MYDEVICE_IOC_CLEAR:
        return mydevice_ioctl_clear();
    case MYDEVICE_IOC_STATUS:
        return mydevice_ioctl_status((struct mydevice_status __user *)arg);
    default:
        return -ENOTTY;
    }
}
/* ─── /proc ─── */
static int mydevice_proc_show(struct seq_file *m, void *v)
{
    seq_printf(m, "buffer_size:  %zu\n",  buffer_size);
    seq_printf(m, "bytes_used:   %zu\n",  buffer_data_len);
    seq_printf(m, "open_count:   %d\n",   atomic_read(&open_count));
    seq_printf(m, "read_count:   %d\n",   atomic_read(&read_count));
    seq_printf(m, "write_count:  %d\n",   atomic_read(&write_count));
    return 0;
}
static int mydevice_proc_open(struct inode *inode, struct file *filp)
{
    return single_open(filp, mydevice_proc_show, NULL);
}
static const struct proc_ops mydevice_proc_ops = {
    .proc_open    = mydevice_proc_open,
    .proc_read    = seq_read,
    .proc_lseek   = seq_lseek,
    .proc_release = single_release,
};
/* ─── module_init ─── */
static int __init mydevice_init(void)
{
    int ret;
    if (buffer_size == 0 || buffer_size > 1024 * 1024) {
        pr_err("mydevice: invalid buffer_size %zu\n", buffer_size);
        return -EINVAL;
    }
    kernel_buffer = kzalloc(buffer_size, GFP_KERNEL);
    if (!kernel_buffer)
        return -ENOMEM;
    ret = alloc_chrdev_region(&dev_num, 0, 1, "mydevice");
    if (ret < 0)
        goto err_free_buf;
    cdev_init(&my_cdev, &mydevice_fops);
    my_cdev.owner = THIS_MODULE;
    ret = cdev_add(&my_cdev, dev_num, 1);
    if (ret < 0)
        goto err_unreg_region;
    mydevice_class = class_create(THIS_MODULE, "mydevice");
    if (IS_ERR(mydevice_class)) {
        ret = PTR_ERR(mydevice_class);
        goto err_del_cdev;
    }
    mydevice_device = device_create(mydevice_class, NULL, dev_num, NULL, "mydevice");
    if (IS_ERR(mydevice_device)) {
        ret = PTR_ERR(mydevice_device);
        goto err_destroy_class;
    }
    proc_entry = proc_create("mydevice", 0444, NULL, &mydevice_proc_ops);
    if (!proc_entry) {
        ret = -ENOMEM;
        goto err_destroy_device;
    }
    pr_info("mydevice: initialized with mutex + wait queues + poll support\n");
    return 0;
err_destroy_device:
    device_destroy(mydevice_class, dev_num);
err_destroy_class:
    class_destroy(mydevice_class);
err_del_cdev:
    cdev_del(&my_cdev);
err_unreg_region:
    unregister_chrdev_region(dev_num, 1);
err_free_buf:
    kfree(kernel_buffer);
    return ret;
}
/* ─── module_exit ─── */
static void __exit mydevice_exit(void)
{
    proc_remove(proc_entry);
    device_destroy(mydevice_class, dev_num);
    class_destroy(mydevice_class);
    cdev_del(&my_cdev);
    unregister_chrdev_region(dev_num, 1);
    kfree(kernel_buffer);
    pr_info("mydevice: unloaded\n");
}
module_init(mydevice_init);
module_exit(mydevice_exit);
```
---
## The Stress Test
The stress test is the proof. A driver that works for one reader and one writer under carefully controlled conditions might still corrupt data under concurrent load. You need to demonstrate correctness under adversarial concurrency.

![Stress Test: 4 Writers + 4 Readers Concurrent Verification](./diagrams/diag-m4-stress-test-architecture.svg)

The strategy: use checksums. Each writer sends a known pattern with a known checksum. Each reader accumulates data and verifies the checksum. If the mutex is working correctly, every byte that enters the buffer exits exactly once, and no bytes are mixed between writers' messages.
### stress_test.sh
```bash
#!/bin/bash
# stress_test.sh — Concurrent reader/writer stress test for mydevice
# Usage: sudo bash stress_test.sh
DEVICE="/dev/mydevice"
TMPDIR=$(mktemp -d)
WRITERS=4
READERS=4
MESSAGES_PER_WRITER=100
MSG_SIZE=256   # bytes per message — fits well within a 4KB buffer
# Cleanup function
cleanup() {
    kill 0 2>/dev/null
    rm -rf "$TMPDIR"
    echo "Stress test cleanup complete."
}
trap cleanup EXIT
if [ ! -c "$DEVICE" ]; then
    echo "ERROR: $DEVICE not found. Run: sudo insmod mydevice.ko"
    exit 1
fi
echo "=== mydevice Stress Test ==="
echo "Writers: $WRITERS  Readers: $READERS"
echo "Messages per writer: $MESSAGES_PER_WRITER  Message size: $MSG_SIZE bytes"
echo ""
# Writer function: writes MESSAGE_PER_WRITER fixed-size messages
# Each message is MSG_SIZE bytes of a repeating character, followed by
# its SHA256 checksum on the next line
writer_func() {
    local id=$1
    local char=$(printf "\\$(printf '%03o' $((65 + id)))")  # 'A' + id
    local sent=0
    for i in $(seq 1 $MESSAGES_PER_WRITER); do
        # Build a fixed-size message: MSG_SIZE bytes of character char
        local msg=$(python3 -c "print('$char' * $MSG_SIZE, end='')")
        local checksum=$(echo -n "$msg" | sha256sum | awk '{print $1}')
        # Write message, retry on EAGAIN (non-blocking not set, so shouldn't happen)
        if echo -n "$msg" > "$DEVICE" 2>/dev/null; then
            echo "$checksum" >> "$TMPDIR/writer_${id}_checksums.txt"
            sent=$((sent + 1))
        else
            echo "Writer $id: write failed on message $i" >&2
        fi
        # Small random sleep to create realistic interleaving
        sleep 0.$(( RANDOM % 10 ))
    done
    echo "Writer $id: sent $sent/$MESSAGES_PER_WRITER messages"
}
# Reader function: reads messages and verifies their checksums
reader_func() {
    local id=$1
    local received=0
    local verified=0
    local failed=0
    local deadline=$((SECONDS + 30))  # 30 second timeout
    while [ $SECONDS -lt $deadline ]; do
        # Read a message from the device
        local data
        data=$(dd if="$DEVICE" bs=$MSG_SIZE count=1 2>/dev/null)
        local status=$?
        if [ $status -ne 0 ] || [ -z "$data" ]; then
            # No data available, brief pause
            sleep 0.1
            continue
        fi
        received=$((received + 1))
        local actual_checksum=$(echo -n "$data" | sha256sum | awk '{print $1}')
        # Verify that this checksum was produced by one of the writers
        # by searching all writer checksum files
        if grep -qF "$actual_checksum" "$TMPDIR"/writer_*_checksums.txt 2>/dev/null; then
            verified=$((verified + 1))
        else
            failed=$((failed + 1))
            echo "Reader $id: CHECKSUM MISMATCH on message $received" >&2
            echo "  Data (first 20 bytes): $(echo -n "$data" | head -c 20)" >&2
            echo "  Computed checksum: $actual_checksum" >&2
        fi
    done
    echo "Reader $id: received=$received verified=$verified failed=$failed"
    echo "$failed" > "$TMPDIR/reader_${id}_failures.txt"
}
# Start readers in background
echo "Starting $READERS readers..."
for i in $(seq 0 $((READERS-1))); do
    reader_func $i &
done
# Brief pause to let readers start and block on empty buffer
sleep 0.5
# Start writers in background
echo "Starting $WRITERS writers..."
for i in $(seq 0 $((WRITERS-1))); do
    writer_func $i &
done
# Wait for all writers to finish
wait $(jobs -p | head -$WRITERS) 2>/dev/null
echo "All writers complete."
# Wait for readers to drain the buffer (give them a few extra seconds)
sleep 3
echo "Stopping readers..."
kill %1 %2 %3 %4 2>/dev/null
wait 2>/dev/null
# Collect results
echo ""
echo "=== Results ==="
total_failures=0
for i in $(seq 0 $((READERS-1))); do
    if [ -f "$TMPDIR/reader_${i}_failures.txt" ]; then
        failures=$(cat "$TMPDIR/reader_${i}_failures.txt")
        total_failures=$((total_failures + failures))
    fi
done
if [ $total_failures -eq 0 ]; then
    echo "PASS: No checksum failures detected."
    echo "Data integrity verified under concurrent access."
else
    echo "FAIL: $total_failures checksum failures detected!"
    echo "Data corruption occurred. Check your mutex implementation."
    exit 1
fi
# Check for kernel oops
if dmesg | grep -iE "oops|BUG:|kernel panic|null pointer" | grep -v "^--$" > /dev/null 2>&1; then
    echo "WARNING: Potential kernel oops detected in dmesg. Review with 'dmesg | tail -50'."
else
    echo "No kernel oops detected."
fi
echo ""
echo "Stress test complete."
```
### A Simpler Concurrent Test with Python
For more precise control and better checksum verification, a Python test is cleaner:
```python
#!/usr/bin/env python3
"""
concurrent_test.py — Precise concurrent stress test for mydevice
Verifies data integrity under concurrent reads and writes.
Usage: sudo python3 concurrent_test.py
"""
import hashlib
import os
import threading
import time
import queue
import sys
DEVICE = "/dev/mydevice"
NUM_WRITERS = 4
NUM_READERS = 4
MESSAGES_PER_WRITER = 50
MESSAGE_SIZE = 128  # bytes
sent_checksums = queue.Queue()
received_checksums = []
checksum_lock = threading.Lock()
stop_event = threading.Event()
errors = []
error_lock = threading.Lock()
def writer(writer_id: int):
    """Write fixed-size messages to the device, recording their checksums."""
    char = chr(ord('A') + writer_id).encode()
    message = char * MESSAGE_SIZE
    checksum = hashlib.sha256(message).hexdigest()
    try:
        fd = os.open(DEVICE, os.O_WRONLY)
    except OSError as e:
        with error_lock:
            errors.append(f"Writer {writer_id}: open failed: {e}")
        return
    for i in range(MESSAGES_PER_WRITER):
        try:
            written = os.write(fd, message)
            if written == MESSAGE_SIZE:
                sent_checksums.put(checksum)
            else:
                with error_lock:
                    errors.append(f"Writer {writer_id}: partial write {written}/{MESSAGE_SIZE}")
        except OSError as e:
            if e.errno == 11:  # EAGAIN
                time.sleep(0.01)
                i -= 1  # retry
                continue
            with error_lock:
                errors.append(f"Writer {writer_id}: write error: {e}")
        time.sleep(0.001 * (writer_id + 1))  # stagger writes
    os.close(fd)
    print(f"Writer {writer_id}: sent {MESSAGES_PER_WRITER} messages")
def reader(reader_id: int):
    """Read messages from the device and record their checksums."""
    try:
        fd = os.open(DEVICE, os.O_RDONLY)
    except OSError as e:
        with error_lock:
            errors.append(f"Reader {reader_id}: open failed: {e}")
        return
    received = 0
    while not stop_event.is_set():
        try:
            data = os.read(fd, MESSAGE_SIZE)
            if data:
                checksum = hashlib.sha256(data).hexdigest()
                with checksum_lock:
                    received_checksums.append(checksum)
                received += 1
        except OSError as e:
            if e.errno == 11:  # EAGAIN — buffer empty, try again
                time.sleep(0.005)
                continue
            if e.errno == 4:   # EINTR — signal, retry
                continue
            break
    os.close(fd)
    print(f"Reader {reader_id}: received {received} messages")
def main():
    if not os.path.exists(DEVICE):
        print(f"ERROR: {DEVICE} not found. Run: sudo insmod mydevice.ko")
        sys.exit(1)
    print(f"Stress test: {NUM_WRITERS} writers × {MESSAGES_PER_WRITER} messages, "
          f"{NUM_READERS} readers, message_size={MESSAGE_SIZE} bytes")
    # Start readers first — they'll block on empty buffer
    reader_threads = [threading.Thread(target=reader, args=(i,), daemon=True)
                      for i in range(NUM_READERS)]
    for t in reader_threads:
        t.start()
    time.sleep(0.2)  # let readers settle
    # Start writers
    writer_threads = [threading.Thread(target=writer, args=(i,))
                      for i in range(NUM_WRITERS)]
    for t in writer_threads:
        t.start()
    # Wait for writers to finish
    for t in writer_threads:
        t.join()
    print("All writers done. Waiting for readers to drain buffer...")
    time.sleep(2)
    stop_event.set()
    for t in reader_threads:
        t.join(timeout=3)
    # Verify
    print("\n=== Verification ===")
    expected = set()
    while not sent_checksums.empty():
        expected.add(sent_checksums.get_nowait())
    received_set = set(received_checksums)
    print(f"Expected unique checksums: {len(expected)}")
    print(f"Received unique checksums: {len(received_set)}")
    print(f"Total messages received:   {len(received_checksums)}")
    corrupted = [c for c in received_checksums if c not in expected]
    if corrupted:
        print(f"FAIL: {len(corrupted)} CORRUPTED messages (unknown checksums)!")
        for c in corrupted[:5]:
            print(f"  Unknown: {c}")
        sys.exit(1)
    elif errors:
        print(f"FAIL: {len(errors)} errors occurred:")
        for e in errors[:5]:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("PASS: All received messages have valid checksums.")
        print("Data integrity confirmed under concurrent access.")
if __name__ == "__main__":
    main()
```
Run it:
```bash
sudo python3 concurrent_test.py
```
A correctly implemented driver will output `PASS`. A driver with a missing mutex will eventually output `FAIL` with corrupted checksums—though the failure may not appear on every run. Race conditions are probabilistic; the stress test makes them reliably observable.
---
## Build and Verify
```bash
# 1. Compile with -Werror
make EXTRA_CFLAGS="-Werror"
# 2. Load the module
sudo insmod mydevice.ko
sudo chmod 666 /dev/mydevice
# 3. Verify blocking behavior: read blocks until data arrives
# In terminal 1:
cat /dev/mydevice
# (hangs — good! blocking on empty buffer)
# In terminal 2:
echo "hello" > /dev/mydevice
# Terminal 1 should immediately print "hello" and return
# 4. Verify O_NONBLOCK behavior
python3 -c "
import os, errno
fd = os.open('/dev/mydevice', os.O_RDONLY | os.O_NONBLOCK)
try:
    data = os.read(fd, 100)
    print(f'Read {len(data)} bytes: {data}')
except OSError as e:
    if e.errno == errno.EAGAIN:
        print('Correctly got EAGAIN on empty buffer with O_NONBLOCK')
    else:
        print(f'Unexpected error: {e}')
os.close(fd)
"
# 5. Verify signal handling: read blocks, then Ctrl+C terminates cleanly
cat /dev/mydevice &
CAT_PID=$!
sleep 0.5
kill -INT $CAT_PID  # send SIGINT (same as Ctrl+C)
wait $CAT_PID
echo "cat exit code: $?"
# Should exit cleanly, not hang
# 6. Verify poll support
python3 -c "
import select, os
fd = os.open('/dev/mydevice', os.O_RDONLY | os.O_NONBLOCK)
# Poll with 100ms timeout — should return immediately (no data)
r, w, x = select.select([fd], [], [], 0.1)
print('Before write — readable:', len(r) > 0)  # Expected: False
# Write from another fd
wfd = os.open('/dev/mydevice', os.O_WRONLY)
os.write(wfd, b'poll test data')
os.close(wfd)
# Poll again — should show readable
r, w, x = select.select([fd], [], [], 0.1)
print('After write  — readable:', len(r) > 0)   # Expected: True
os.close(fd)
"
# 7. Run the stress test
sudo bash stress_test.sh
# Or:
sudo python3 concurrent_test.py
# 8. Verify no oops during or after stress test
dmesg | tail -20 | grep -iE "oops|BUG:|null pointer" || echo "No kernel oops"
# 9. Confirm rmmod works after stress test (module_put balance)
sudo rmmod mydevice
lsmod | grep mydevice || echo "Module cleanly unloaded"
```
### Common Errors at This Milestone
```bash
# Process becomes unkillable (D state in ps output)
# Cause: wait_event instead of wait_event_interruptible, OR
#        not propagating -ERESTARTSYS from mutex_lock_interruptible
# Fix: search your code for 'wait_event(' without '_interruptible'
#      and for 'mutex_lock(' without '_interruptible'
# Readers never get data despite writers writing
# Cause: wake_up_interruptible not called after write, OR
#        condition in wait_event_interruptible is wrong
# Verify: add pr_info("mydevice: calling wake_up") in write handler
# poll() always returns immediately with data = 0
# Cause: not calling poll_wait() before returning mask
# Fix: ensure poll_wait() is called with BOTH wait queues
# Checksum failures in stress test
# Cause: missing mutex around read or write, or buffer corruption
# Debug: reduce to 2 writers/readers and add pr_info for each operation
# "BUG: sleeping function called from invalid context"
# Cause: mutex_lock called from interrupt context or with spinlock held
# This won't happen in your char device handlers (they're process context)
# but would happen if you tried to call mutex_lock from a timer callback
# rmmod fails with "Device or resource busy"
# Cause: a reader process is blocking inside wait_event_interruptible
#        when rmmod is attempted
# Fix: the module refcount (try_module_get/module_put) correctly prevents
#      this — rmmod will refuse until all fds are closed
```
---
## Hardware Soul: Cache and CPU Behavior Under Concurrent Access
Let's trace what the hardware sees during a concurrent read and write on two CPUs.
**The false sharing scenario (avoided by your mutex):**
Without the mutex, CPU 0 (writer) and CPU 1 (reader) both access `buffer_data_len`. If `buffer_data_len` (8 bytes) and `kernel_buffer` pointer (8 bytes) are adjacent in `.bss`, they share a 64-byte cache line. When CPU 0 writes `buffer_data_len`, the cache coherency protocol (MESI on x86) **invalidates that cache line on CPU 1**. CPU 1 must re-fetch the entire 64-byte line from L3 or main memory before it can read `buffer_data_len`. This isn't a bug—this is the correct behavior of cache coherency. But it means that *even with atomics*, concurrent access to adjacent variables causes cache line bouncing between CPUs.
With the mutex:
1. CPU 0 acquires the mutex. Internally, `mutex_lock` uses an atomic compare-and-swap (`cmpxchg`) to set the mutex owner. This is a **locked instruction** with a memory barrier—all CPU 0's prior writes become visible to other CPUs before the lock completes.
2. CPU 0 writes `buffer_data_len` and calls `wake_up_interruptible`. The wake_up includes an **implicit memory barrier** (`smp_mb`): any CPU woken by the wait queue is guaranteed to see all writes CPU 0 made before calling wake_up.
3. CPU 0 releases the mutex. `mutex_unlock` contains another memory barrier—ensures `buffer_data_len`'s new value is flushed to L3 before the unlock completes.
4. CPU 1 (reader, woken from wait queue) acquires the mutex. This cmpxchg on the mutex field causes CPU 1 to fetch the cache line containing the mutex from L3. Now CPU 1 reads `buffer_data_len`—guaranteed to see CPU 0's write because the mutex's barrier sequence ordered the visibility.
**Memory bandwidth cost of your mutex + wait queue:**
Each mutex lock/unlock pair involves:
- 1 `cmpxchg` (locked compare-and-exchange): ~20-40 cycles in the uncontended case
- 1-2 memory barriers: ~10-15 cycles
- Total overhead: ~30-60 cycles per lock/unlock pair on modern x86
For a 4KB write of actual data, `copy_from_user` touches 64 cache lines (4096 / 64). The dominant cost is cache fills, not the mutex itself. Your synchronization overhead is negligible compared to the data transfer cost—the right primitive for the right problem.
**The thundering herd and wake_up_interruptible:**
When 4 readers are sleeping and a writer calls `wake_up_interruptible(&mydevice_read_queue)`, all 4 readers become runnable simultaneously. The scheduler must now schedule all 4 on available CPUs. All 4 try to acquire the mutex. Three of them will immediately fail (the mutex is held by whoever won the race) and go back to sleeping—but now they're sleeping on the mutex, not the wait queue. This is a minor inefficiency: for very high-contention scenarios, `wake_up_interruptible` vs `wake_up` vs targeted wakeup patterns matter for performance. For 4 readers, it's negligible.
---
## Knowledge Cascade: What You've Unlocked
### Process Context vs Interrupt Context: The Foundation of RTOS Design
The distinction between "can sleep" and "cannot sleep" is not peculiar to Linux—it is the central organizing principle of all real-time operating system (RTOS) design. In FreeRTOS, VxWorks, and QNX, interrupt service routines (ISRs) face exactly the same constraint: no blocking, no sleeping, no heap allocation with a general-purpose allocator, no functions that might wait for a resource. The solution in every RTOS is also the same: ISRs defer work to a task/thread that *can* sleep. In Linux, this is the "bottom half" mechanism—interrupt handlers are split into a fast top half (runs in interrupt context, cannot sleep) and a slow bottom half (runs in process context via workqueues or tasklets, can sleep).
If you move into embedded systems, device driver work, or any safety-critical software domain, this process/interrupt context dichotomy will be the first thing senior engineers ask about. You now understand it at a deep enough level to teach it.
### Producer-Consumer and the Universality of Wait Queues
The write-wakes-reader pattern you implemented is the kernel's version of a **condition variable** from POSIX threading (`pthread_cond_wait` / `pthread_cond_signal`). The conceptual structure is identical:
| Kernel | POSIX pthreads | Go | Rust |
|--------|-----------------|----|----|
| `wait_event_interruptible(queue, cond)` | `pthread_cond_wait(&cond, &mutex)` | `<-channel` | `receiver.recv()` |
| `wake_up_interruptible(&queue)` | `pthread_cond_signal(&cond)` | `channel <- val` | `sender.send(val)` |
| `DEFINE_MUTEX` | `pthread_mutex_t` | `sync.Mutex` | `std::sync::Mutex` |
| Condition recheck loop | Mandatory in pthreads | Built into channel semantics | Built into lock |
Go channels, Rust's `mpsc`, and Java's `BlockingQueue` are all higher-level abstractions over this exact pattern. The "wait for condition, recheck after waking" loop that `wait_event_interruptible` implements internally is the same loop you'd write manually with `pthread_cond_wait`. Now you understand *why* the condition must be rechecked: it's not a language-specific quirk—it's a fundamental property of wakeup semantics in any system where the scheduler can introduce spurious wakeups or multiple waiters.
### O_NONBLOCK and the Entire Async I/O Ecosystem
Understanding `-EAGAIN` at the kernel driver level is the key that unlocks comprehension of every async I/O framework you'll ever use:
- **Non-blocking sockets**: `connect()` on a non-blocking TCP socket returns `-EINPROGRESS` (a variant). `send()`/`recv()` return `-EAGAIN` when the buffer is full/empty. Your driver does the exact same thing.
- **epoll's edge-triggered mode**: an fd in edge-triggered epoll only notifies when the state *changes* from unready to ready. If you drain the socket buffer incompletely and go back to `epoll_wait`, you never get another notification. The fix: keep reading until you get `-EAGAIN`, then go back to epoll. You now know why.
- **io_uring**: the kernel-side operations in io_uring ultimately go through the same file_operations `.read`/`.write` callbacks as a regular `read()`/`write()` syscall. The difference is that io_uring batches submissions and completions through a shared ring buffer, eliminating individual syscall overhead. The `-EAGAIN` handling in io_uring's kernel code is identical to what you've written.
- **Node.js's libuv**: at the bottom of Node's event loop is a platform-specific I/O multiplexer (epoll on Linux). Every non-blocking file descriptor in a Node program is registered with epoll. The `poll_wait` + mask return in your `.poll` function is the mechanism that tells libuv "this fd is ready." You are now looking at the foundation underneath JavaScript's async/await.
### Signal-Safe Kernel Programming and SA_RESTART
The `-ERESTARTSYS` contract you implemented connects directly back to your prerequisite `signal-handler` project. Recall: when you register a signal handler with `sigaction`, the `SA_RESTART` flag controls what happens if a slow syscall (one that can block indefinitely, like `read()` on a pipe) is interrupted:
- With `SA_RESTART`: the kernel restarts the syscall automatically. The process doesn't know it was interrupted. This is only possible because your kernel handler returned `-ERESTARTSYS` (not `-EINTR`), allowing the kernel to restart.
- Without `SA_RESTART`: the syscall returns to userspace with `errno = EINTR`. The userspace code must check for `EINTR` and retry.
When you see `EINTR` in `strace` output on a `read()` or `write()` call, you're seeing the userspace-visible manifestation of a `-ERESTARTSYS` return from a driver handler that was interrupted by a signal. Your driver produces exactly these `EINTR`-visible returns. The entire `SA_RESTART` design pattern—which affects every Unix program that handles signals—is predicated on kernel handlers returning `-ERESTARTSYS` correctly.
---
## Milestone Checklist
Before considering this milestone complete, verify each criterion:
```bash
# 1. Mutex is in place — no data corruption under concurrent access
sudo python3 concurrent_test.py  # Must output PASS
# 2. Blocking read works — verify with two terminals
# Terminal 1: cat /dev/mydevice  (should hang)
# Terminal 2: echo "test" > /dev/mydevice  (Terminal 1 should print "test")
# 3. -ERESTARTSYS is propagated — Ctrl+C works on blocked read
cat /dev/mydevice &
sleep 0.3
kill -INT $!
wait $!
echo "Exit code: $?"  # Should be non-zero but process should be DEAD (not stuck)
# 4. O_NONBLOCK returns -EAGAIN on empty buffer (not 0, not blocking)
python3 -c "
import os, errno
fd = os.open('/dev/mydevice', os.O_RDONLY | os.O_NONBLOCK)
try:
    os.read(fd, 100)
    print('ERROR: should have raised EAGAIN')
except OSError as e:
    print('PASS' if e.errno == errno.EAGAIN else f'FAIL: got {e.errno}')
os.close(fd)
"
# 5. O_NONBLOCK on write returns -EAGAIN when buffer is full
# Fill buffer first, then try non-blocking write
python3 -c "
import os, errno
# Fill the buffer
fd = os.open('/dev/mydevice', os.O_WRONLY)
os.write(fd, b'X' * 4096)  # fill 4KB buffer
os.close(fd)
# Try non-blocking write
fd = os.open('/dev/mydevice', os.O_WRONLY | os.O_NONBLOCK)
try:
    os.write(fd, b'more data')
    print('ERROR: should have raised EAGAIN')
except OSError as e:
    print('PASS' if e.errno == errno.EAGAIN else f'FAIL: got {e.errno}')
os.close(fd)
"
# 6. Poll returns POLLIN when data available
python3 -c "
import select, os
wfd = os.open('/dev/mydevice', os.O_WRONLY)
os.write(wfd, b'poll test')
os.close(wfd)
rfd = os.open('/dev/mydevice', os.O_RDONLY | os.O_NONBLOCK)
r, w, x = select.select([rfd], [], [], 1.0)
print('PASS' if rfd in r else 'FAIL')
os.close(rfd)
"
# 7. Poll returns POLLOUT when space available
python3 -c "
import select, os
fd = os.open('/dev/mydevice', os.O_WRONLY | os.O_NONBLOCK)
r, w, x = select.select([], [fd], [], 1.0)
print('PASS (writable)' if fd in w else 'FAIL (not writable)')
os.close(fd)
"
# 8. Wake up fires on write — blocked reader wakes within 100ms
python3 -c "
import os, threading, time
ready = threading.Event()
woke_at = [None]
def reader():
    fd = os.open('/dev/mydevice', os.O_RDONLY)
    ready.set()
    data = os.read(fd, 100)
    woke_at[0] = time.time()
    os.close(fd)
t = threading.Thread(target=reader)
t.start()
ready.wait()
start = time.time()
time.sleep(0.05)
fd = os.open('/dev/mydevice', os.O_WRONLY)
os.write(fd, b'wake!')
os.close(fd)
t.join()
latency_ms = (woke_at[0] - start) * 1000
print(f'Wakeup latency: {latency_ms:.1f}ms')
print('PASS' if latency_ms < 200 else 'FAIL (too slow)')
"
# 9. Stress test with 4 readers + 4 writers
sudo bash stress_test.sh  # OR:
sudo python3 concurrent_test.py
# 10. No kernel oops after all tests
dmesg | grep -cE "oops|BUG:|kernel panic" && echo "FAIL: oops detected" || echo "PASS: no oops"
# 11. rmmod succeeds cleanly (all fds closed)
sudo rmmod mydevice
echo "rmmod exit: $?"  # Should be 0
```
---
<!-- END_MS -->




# TDD

A four-milestone progression from bare kernel module to a fully concurrent, poll-capable character device driver. Each module builds directly on the previous: M1 establishes the init/exit scaffold and Kbuild toolchain; M2 adds the VFS dispatch table and kernel-userspace memory boundary; M3 layers ioctl control and /proc introspection; M4 hardens concurrent access with mutex, wait queues, and poll. The architecture mirrors production kernel drivers in drivers/ — goto error unwinding, atomic_t for statistics, mutex for buffer invariants, wait_event_interruptible for blocking I/O, and seq_file for /proc. The final driver is a working pipe-semantics character device exercised by a checksum-verified concurrent stress test.



<!-- TDD_MOD_ID: build-kernel-module-m1 -->
# Technical Design Specification: Hello World Kernel Module (`build-kernel-module-m1`)
---
## 1. Module Charter
This module establishes the complete out-of-tree kernel module development toolchain and implements a minimal loadable kernel module (`hello.ko`) with proper init/exit lifecycle, ELF metadata, printk logging, and a single `buffer_size` integer parameter exposed via sysfs. The module prints a KERN_INFO message on load (reflecting the parameter value) and a KERN_INFO message on unload; no other side effects occur.
This module does **not** implement any character device, file_operations, buffer allocation, /proc entry, ioctl, wait queue, mutex, or userspace data transfer. Those belong to Milestone 2 and later. The `buffer_size` parameter is validated and stored but no memory is allocated from it—its sole purpose here is to demonstrate the module_param/sysfs binding and load-time configuration.
Upstream dependency: the running kernel's header package (`linux-headers-$(uname -r)`) must be installed before any compilation step. Downstream: every subsequent milestone (`build-kernel-module-m2` through `build-kernel-module-m4`) inherits the Kbuild Makefile pattern and module metadata established here.
**Invariants that must hold at all times:**
- `buffer_size` is in the range `[1, 1048576]` whenever `module_init` returns 0. If the parameter is out of range, `module_init` returns a negative errno and `insmod` fails—the module never reaches running state.
- `MODULE_LICENSE("GPL")` is declared exactly once. Its absence taints the kernel and blocks access to `EXPORT_SYMBOL_GPL` symbols required by all later milestones.
- The vermagic string embedded in `hello.ko` matches `uname -r` on the target machine. Mismatches are caught at `insmod` time, not at compile time.
---
## 2. File Structure
Create files in this exact order:
```
hello/                          ← working directory (1)
├── Makefile                    ← (2) Kbuild delegation makefile
└── hello.c                     ← (3) kernel module source
```
There are no subdirectories, header files, or additional source files for this milestone. The build system produces these artifacts (do not create manually):
```
hello/
├── hello.ko                    ← loadable kernel object (output)
├── hello.o                     ← intermediate object
├── hello.mod.c                 ← generated module glue code
├── hello.mod.o                 ← compiled glue
├── Module.symvers              ← symbol export table (empty for this module)
└── modules.order               ← build-system tracking file
```
---
## 3. Complete Data Model
### 3.1 Module Parameter
```c
/* hello.c — global parameter variable */
static int buffer_size = 4096;
```
| Field | Type | Default | Range | Sysfs Permission | Purpose |
|---|---|---|---|---|---|
| `buffer_size` | `int` | `4096` | `[1, 1048576]` | `0444` (r--r--r--) | Demonstrates module_param; will become the actual buffer allocation size in M2 |
**Why `int` and not `size_t`:** `module_param` supports a fixed set of types: `bool`, `invbool`, `charp`, `short`, `ushort`, `int`, `uint`, `long`, `ulong`, `hexint`. There is no `size_t` type token. Use `int` here. In M2, the live buffer allocation uses `size_t` internally after copying from this parameter.
**Why `0444` and not `0644`:** `buffer_size` has no runtime effect in M1 (no buffer is allocated). Allowing unprivileged writes via `0644` would let any user set it to a negative value, creating false expectations. Read-only sysfs visibility is the correct choice when no runtime setter exists. If `0` were used instead, the parameter would be invisible in sysfs—defeating the acceptance criterion.
### 3.2 ELF Metadata Sections
The following macros emit strings into named ELF sections inside `hello.ko`. They are not C variables—they are linker directives. `modinfo` reads them directly from the ELF binary without loading the module into the kernel.
| Macro | ELF Section | Content | Required |
|---|---|---|---|
| `MODULE_LICENSE("GPL")` | `.modinfo` | `"license=GPL"` | **Yes** — gates EXPORT_SYMBOL_GPL access |
| `MODULE_AUTHOR(...)` | `.modinfo` | `"author=..."` | Recommended |
| `MODULE_DESCRIPTION(...)` | `.modinfo` | `"description=..."` | Recommended |
| `MODULE_VERSION(...)` | `.modinfo` | `"version=..."` | Recommended |
The `vermagic` string is injected automatically by Kbuild during compilation. It encodes the kernel version, SMP flag, preemption model, and module versioning hash. You do not write this manually.
### 3.3 Init/Exit Function Attributes
| Annotation | ELF Section | Linker Effect |
|---|---|---|
| `__init` | `.init.text` | Pages in this section are freed by the kernel after `module_init` returns 0. Calling an `__init`-annotated function after init is undefined behavior (pages may be gone). |
| `__exit` | `.exit.text` | Pages in this section are discarded at compile time when the module is compiled directly into the kernel (`obj-y`). For loadable modules (`obj-m`), this section is retained normally. |
---
## 4. Interface Contracts
### 4.1 `hello_init(void) → int`
**Signature:**
```c
static int __init hello_init(void);
```
**Registered via:** `module_init(hello_init)` — stores the function pointer in the `.initcall` ELF section. The kernel's module loader calls it during `insmod`/`finit_module` syscall processing.
**Preconditions:**
- `buffer_size` has been written by the module parameter subsystem. This happens before `hello_init` is called. The value is whatever was passed as `buffer_size=N` on the `insmod` command line, or `4096` if omitted.
- No locks are held. No interrupts are disabled. The function runs in process context (the `insmod` process), so sleeping is permitted (though this function does not sleep).
**Algorithm:**
1. Validate `buffer_size`. If `buffer_size <= 0`: emit `printk(KERN_ERR ...)` with the offending value, return `-EINVAL`.
2. Validate upper bound. If `buffer_size > 1048576`: emit `printk(KERN_WARNING ...)`, clamp `buffer_size = 1048576`.
3. Emit `printk(KERN_INFO "hello: module loaded, buffer_size=%d\n", buffer_size)`.
4. Return `0`.
**Return values:**
| Return | Meaning | Effect |
|---|---|---|
| `0` | Success | Module reaches running state; sysfs and dmesg entries visible |
| `-EINVAL` | `buffer_size` out of range | `insmod` prints error, module not loaded, no cleanup needed (nothing was allocated) |
**Side effects:**
- Writes one record to the kernel ring buffer via `printk`.
- No memory allocation, no device registration, no file creation.
**Edge cases:**
- `buffer_size = 0`: fails with `-EINVAL` (boundary: `<= 0` check catches zero).
- `buffer_size = -1`: fails with `-EINVAL`.
- `buffer_size = 1`: valid minimum, loads successfully.
- `buffer_size = 1048576`: valid maximum, loads successfully.
- `buffer_size = 1048577`: clamped to `1048576` with a KERN_WARNING, loads successfully.
- No `buffer_size=N` on command line: `buffer_size` retains default `4096`, loads successfully.
### 4.2 `hello_exit(void) → void`
**Signature:**
```c
static void __exit hello_exit(void);
```
**Registered via:** `module_exit(hello_exit)`.
**Preconditions:**
- The module is in running state. The kernel guarantees this—`hello_exit` is only called by `rmmod`/`delete_module` when the module's reference count is zero and no users are active.
- No return value. The function cannot fail. rmmod does not check a return value.
**Algorithm:**
1. Emit `printk(KERN_INFO "hello: module unloaded\n")`.
2. Return.
**Side effects:** One printk write to the kernel ring buffer.
**What NOT to do:** Do not call `kfree(NULL)`, do not attempt to unregister anything—nothing was registered in `hello_init`. An empty exit function (except for printk) is correct and complete.
---
## 5. Algorithm Specification
### 5.1 Kbuild Out-of-Tree Compilation
The `make` invocation delegates to the kernel's build system. The sequence:
```
make (in hello/)
  → MAKE -C /lib/modules/$(uname -r)/build M=$(PWD) modules
    → Kbuild reads M=$(PWD)/Makefile
    → sees: obj-m += hello.o
    → compiles hello.c with kernel CFLAGS:
         -D__KERNEL__ -DMODULE -I<kernel-headers>/include ...
    → produces hello.o (ELF relocatable)
    → runs modpost: generates hello.mod.c, validates exports
    → compiles hello.mod.c → hello.mod.o
    → links hello.o + hello.mod.o → hello.ko
```
The `obj-m` assignment is the switch that tells Kbuild "this is a loadable module." Using `obj-y` instead would mean "compile into the kernel image directly"—this is never correct for out-of-tree development.
### 5.2 Module Load Sequence (insmod internals)
Understanding this sequence is necessary to debug load failures:
```
insmod hello.ko
  1. open("hello.ko") → read ELF into userspace buffer
  2. finit_module(fd, params, 0) syscall
  3. Kernel: copy ELF from userspace
  4. Kernel: check vermagic string against running kernel's own magic
     → MISMATCH: return -ENOEXEC ("Invalid module format")
  5. Kernel: vmalloc() space for .text + .data + .bss
  6. Kernel: resolve undefined symbols (printk, module_param_ops, etc.)
     via the kernel's exported symbol table
     → SYMBOL NOT FOUND: return -ENOENT ("Unknown symbol in module")
     → SYMBOL_GPL but no MODULE_LICENSE("GPL"): return -ENOENT
  7. Kernel: apply ELF relocations (patch function calls to resolved addresses)
  8. Kernel: process module parameters from params string ("buffer_size=8192")
     → writes 8192 into buffer_size variable
  9. Kernel: call hello_init()
     → returns -EINVAL: abort, free vmalloc'd memory, return error to insmod
     → returns 0: module is live
 10. Kernel: expose /sys/module/hello/ tree including parameters/buffer_size
```
Step 4 is why you must install `linux-headers-$(uname -r)`—the vermagic in your `.ko` must match the running kernel exactly.
Step 8 happens **before** `hello_init` is called. This means `buffer_size` already contains the user-supplied value when your validation code runs.
### 5.3 Parameter Validation Algorithm
```c
static int __init hello_init(void)
{
    /* Step 1: reject non-positive values */
    if (buffer_size <= 0) {
        printk(KERN_ERR "hello: invalid buffer_size %d, must be > 0\n",
               buffer_size);
        return -EINVAL;
    }
    /* Step 2: clamp oversized values with a warning */
    if (buffer_size > 1024 * 1024) {
        printk(KERN_WARNING
               "hello: buffer_size %d exceeds 1MB, clamping to 1048576\n",
               buffer_size);
        buffer_size = 1024 * 1024;
    }
    /* Step 3: log successful load */
    printk(KERN_INFO "hello: module loaded successfully, buffer_size=%d\n",
           buffer_size);
    return 0;
}
```
**Why clamp instead of reject for oversized values:** The acceptance criteria require the module to load with any reasonable positive value. Clamping is defensive but permissive. If the project specification required strict rejection, the condition would be `> 1048576 → return -EINVAL`. Either is correct; clamping with a warning is chosen here because it demonstrates KERN_WARNING and is user-friendlier.
---
## 6. Error Handling Matrix
| Error Condition | Detected By | Kernel/Insmod Response | User-Visible? | Recovery |
|---|---|---|---|---|
| `buffer_size <= 0` at load time | `hello_init` validation | `init` returns `-EINVAL`; module not loaded | Yes: `insmod: ERROR: could not insert module hello.ko: Invalid argument` + dmesg KERN_ERR | Re-run `insmod hello.ko buffer_size=4096` |
| `buffer_size > 1048576` at load time | `hello_init` validation | KERN_WARNING emitted, value clamped, module loads | Partial: dmesg warning visible; insmod succeeds | No action needed; module operational |
| Missing `linux-headers-$(uname -r)` | `make` (Kbuild cannot find kernel build dir) | Build fails: `make: /lib/modules/.../build: No such file or directory` | Yes: make exits non-zero | `sudo apt install linux-headers-$(uname -r)` |
| vermagic mismatch | Kernel module loader (step 4 above) | `insmod` fails: `Invalid module format` | Yes: insmod error message | Recompile against correct headers: `make clean && make` |
| Missing `MODULE_LICENSE("GPL")` | Kernel module loader (step 6 above) | Linker fails to resolve `EXPORT_SYMBOL_GPL` symbols at modpost | Yes: build error listing unresolved symbols (may not appear in M1 since M1 uses no GPL-only symbols, but kernel taints and prints warning) | Add `MODULE_LICENSE("GPL");` to source |
| `insmod` run without `sudo` | Kernel capability check (`CAP_SYS_MODULE`) | `insmod: ERROR: could not insert module hello.ko: Operation not permitted` | Yes | Re-run with `sudo` |
| Compile warning with `-Werror` | GCC | Build fails | Yes: compiler output | Fix the warning before proceeding |
**Note on M1 and GPL symbols:** The M1 module calls only `printk` and uses `module_param`, both of which are exported via plain `EXPORT_SYMBOL` (not `EXPORT_SYMBOL_GPL`). Omitting `MODULE_LICENSE("GPL")` will not cause a build failure in M1 specifically, but will taint the kernel with the `P` flag and will cause failures in M2+ which use GPL-only symbols. Include the license declaration regardless.
---
## 7. Implementation Sequence with Checkpoints
### Phase 1: Kbuild Makefile (0.5–1 hour)
Create `hello/Makefile` with exact content:
```makefile
# Kbuild instruction: build hello.c as a loadable module (.ko)
obj-m += hello.o
# KDIR: kernel build directory for the currently-running kernel.
# Override with: make KDIR=/path/to/other/kernel/build
KDIR ?= /lib/modules/$(shell uname -r)/build
# PWD: directory containing this Makefile and hello.c
PWD := $(shell pwd)
all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules
clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
```
**Critical formatting requirement:** The lines beginning with `$(MAKE)` must be indented with a **single TAB character** (ASCII 0x09), not spaces. Make will error with `Makefile:N: *** missing separator. Stop.` if spaces are used. Configure your editor to insert literal tabs in Makefiles.
**Checkpoint 1:** Run `make` from the `hello/` directory with an empty `hello.c` (just `// placeholder`). The build will fail with a compiler error about missing content, but you should see Kbuild machinery activate:
```
make -C /lib/modules/6.x.x-generic/build M=/path/to/hello modules
make[1]: Entering directory '/usr/src/linux-headers-6.x.x-generic'
```
If instead you see `No such file or directory` for the kernel build path, install headers first.
### Phase 2: Module Metadata and Init/Exit (0.5–1 hour)
Create `hello/hello.c`:
```c
// hello.c — Milestone 1: minimal kernel module with metadata and lifecycle
#include <linux/module.h>      /* MODULE_LICENSE, module_init, module_exit */
#include <linux/kernel.h>      /* printk, KERN_INFO, KERN_ERR, KERN_WARNING */
#include <linux/init.h>        /* __init, __exit */
#include <linux/moduleparam.h> /* module_param, MODULE_PARM_DESC */
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name <you@example.com>");
MODULE_DESCRIPTION("Minimal hello world kernel module with parameter");
MODULE_VERSION("0.1");
static int buffer_size = 4096;
module_param(buffer_size, int, 0444);
MODULE_PARM_DESC(buffer_size, "Internal buffer size in bytes (default: 4096, max: 1048576)");
static int __init hello_init(void)
{
	if (buffer_size <= 0) {
		printk(KERN_ERR "hello: invalid buffer_size %d, must be > 0\n",
		       buffer_size);
		return -EINVAL;
	}
	if (buffer_size > 1024 * 1024) {
		printk(KERN_WARNING
		       "hello: buffer_size %d exceeds 1MB, clamping to 1048576\n",
		       buffer_size);
		buffer_size = 1024 * 1024;
	}
	printk(KERN_INFO "hello: module loaded successfully, buffer_size=%d\n",
	       buffer_size);
	return 0;
}
static void __exit hello_exit(void)
{
	printk(KERN_INFO "hello: module unloaded\n");
}
module_init(hello_init);
module_exit(hello_exit);
```
**Indentation:** Linux kernel coding style mandates **tabs**, not spaces. The `if` bodies above use one tab. Configure your editor accordingly.
**Checkpoint 2:** Run `make EXTRA_CFLAGS="-Werror"`. Expected output ends with:
```
  CC [M]  /path/to/hello/hello.o
  MODPOST /path/to/hello/hello.ko
  CC [M]  /path/to/hello/hello.mod.o
  LD [M]  /path/to/hello/hello.ko
```
Zero warnings, zero errors. Verify `hello.ko` exists: `ls -lh hello.ko`.
Run `modinfo hello.ko` and verify all four metadata fields appear:
```
filename:    /path/to/hello/hello.ko
version:     0.1
description: Minimal hello world kernel module with parameter
author:      Your Name <you@example.com>
license:     GPL
...
vermagic:    6.x.x-generic SMP preempt mod_unload modversions
```
### Phase 3: Load/Unload and Sysfs Verification (0.5–1 hour)
```bash
# Load with default parameter
sudo insmod hello.ko
# Verify init message
dmesg | tail -5
# Must contain: hello: module loaded successfully, buffer_size=4096
# Verify module is listed
lsmod | grep hello
# Must show: hello    <size>    0
# Verify sysfs parameter file exists and contains correct value
cat /sys/module/hello/parameters/buffer_size
# Must output: 4096
# Unload
sudo rmmod hello
# Verify exit message
dmesg | tail -3
# Must contain: hello: module unloaded
# Load with custom parameter
sudo insmod hello.ko buffer_size=8192
dmesg | tail -3
# Must contain: hello: module loaded successfully, buffer_size=8192
cat /sys/module/hello/parameters/buffer_size
# Must output: 8192
sudo rmmod hello
```
**Checkpoint 3:** All six commands above produce expected output. The sysfs file exists and shows the correct value both at default and with a custom parameter.
### Phase 4: Verification Script (0.5–1 hour)
Create `hello/verify.sh`:
```bash
#!/bin/bash
# verify.sh — Acceptance test for Milestone 1
# Usage: sudo bash verify.sh
# Exit code: 0 = all pass, 1 = at least one failure
set -euo pipefail
MODULE_NAME="hello"
MODULE_FILE="./hello.ko"
PASS=0
FAIL=0
pass() { echo "  PASS: $1"; PASS=$((PASS+1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL+1)); }
# Ensure we start clean
if lsmod | grep -q "^${MODULE_NAME} "; then
    echo "Removing existing module..."
    rmmod "${MODULE_NAME}"
fi
echo ""
echo "=== Milestone 1 Verification ==="
echo ""
echo "--- Phase 1: Build artifact ---"
if [ -f "${MODULE_FILE}" ]; then
    pass "hello.ko exists"
else
    fail "hello.ko not found — run 'make' first"
    exit 1
fi
echo ""
echo "--- Phase 2: modinfo metadata ---"
LICENSE=$(modinfo "${MODULE_FILE}" 2>/dev/null | grep '^license:' | awk '{print $2}')
AUTHOR=$(modinfo "${MODULE_FILE}" 2>/dev/null | grep '^author:')
DESC=$(modinfo "${MODULE_FILE}" 2>/dev/null | grep '^description:')
VERSION=$(modinfo "${MODULE_FILE}" 2>/dev/null | grep '^version:')
[ "${LICENSE}" = "GPL" ] && pass "LICENSE=GPL" || fail "LICENSE missing or not GPL (got: '${LICENSE}')"
[ -n "${AUTHOR}" ]       && pass "AUTHOR present" || fail "AUTHOR missing"
[ -n "${DESC}" ]         && pass "DESCRIPTION present" || fail "DESCRIPTION missing"
[ -n "${VERSION}" ]      && pass "VERSION present" || fail "VERSION missing"
echo ""
echo "--- Phase 3: Default load ---"
insmod "${MODULE_FILE}"
sleep 0.2
DMESG_LOAD=$(dmesg | grep "hello: module loaded" | tail -1)
[ -n "${DMESG_LOAD}" ] && pass "init printk appears in dmesg" || fail "init printk not found in dmesg"
echo "${DMESG_LOAD}" | grep -q "buffer_size=4096" \
    && pass "default buffer_size=4096 in dmesg" \
    || fail "buffer_size=4096 not found in init message"
lsmod | grep -q "^${MODULE_NAME} " \
    && pass "module appears in lsmod" \
    || fail "module not in lsmod"
SYSFS_VAL=$(cat "/sys/module/${MODULE_NAME}/parameters/buffer_size" 2>/dev/null)
[ "${SYSFS_VAL}" = "4096" ] \
    && pass "sysfs buffer_size = 4096" \
    || fail "sysfs buffer_size unexpected: '${SYSFS_VAL}'"
# Verify sysfs file permissions (0444 = 100444 in octal stat output)
PERM=$(stat -c "%a" "/sys/module/${MODULE_NAME}/parameters/buffer_size" 2>/dev/null)
[ "${PERM}" = "444" ] \
    && pass "sysfs file permission = 0444" \
    || fail "sysfs file permission unexpected: '${PERM}'"
echo ""
echo "--- Phase 4: Unload ---"
rmmod "${MODULE_NAME}"
sleep 0.2
DMESG_EXIT=$(dmesg | grep "hello: module unloaded" | tail -1)
[ -n "${DMESG_EXIT}" ] && pass "exit printk appears in dmesg" || fail "exit printk not found in dmesg"
lsmod | grep -q "^${MODULE_NAME} " \
    && fail "module still appears in lsmod after rmmod" \
    || pass "module removed from lsmod"
echo ""
echo "--- Phase 5: Custom parameter ---"
insmod "${MODULE_FILE}" buffer_size=16384
sleep 0.2
SYSFS_VAL=$(cat "/sys/module/${MODULE_NAME}/parameters/buffer_size" 2>/dev/null)
[ "${SYSFS_VAL}" = "16384" ] \
    && pass "sysfs buffer_size = 16384 after custom load" \
    || fail "sysfs buffer_size unexpected: '${SYSFS_VAL}'"
dmesg | grep "hello: module loaded" | tail -1 | grep -q "buffer_size=16384" \
    && pass "custom buffer_size=16384 in dmesg" \
    || fail "custom buffer_size=16384 not found in init message"
rmmod "${MODULE_NAME}"
echo ""
echo "--- Phase 6: Invalid parameter ---"
set +e
insmod "${MODULE_FILE}" buffer_size=-1 2>/dev/null
RC=$?
set -e
[ "${RC}" -ne 0 ] \
    && pass "insmod with buffer_size=-1 fails (exit ${RC})" \
    || fail "insmod with buffer_size=-1 should fail but succeeded"
dmesg | grep "hello: invalid buffer_size" | tail -1 | grep -q "\-1" \
    && pass "KERN_ERR message for invalid parameter in dmesg" \
    || fail "KERN_ERR message for invalid parameter not found"
# Confirm module is not loaded after failed init
lsmod | grep -q "^${MODULE_NAME} " \
    && fail "module loaded despite invalid parameter" \
    || pass "module not loaded after init failure"
echo ""
echo "================================"
echo "Results: ${PASS} passed, ${FAIL} failed"
echo "================================"
[ "${FAIL}" -eq 0 ] && exit 0 || exit 1
```
```bash
chmod +x verify.sh
sudo bash verify.sh
```
**Checkpoint 4:** All lines print `PASS`. Zero `FAIL` lines. Exit code 0.

![Module Lifecycle: insmod → init → running → rmmod → exit](./diagrams/tdd-diag-1.svg)

The diagram above shows the Kbuild compilation flow from `hello.c` through Kbuild invocation to the final `hello.ko` ELF object, including the modpost step that injects vermagic.

![Kbuild Compilation Flow: hello.c → hello.ko](./diagrams/tdd-diag-2.svg)

The diagram above shows the `insmod` → kernel → `hello_init` → sysfs sequence, with the parameter subsystem writing `buffer_size` before `init` is called.

![Kernel vs Userspace Virtual Address Space Layout (x86-64)](./diagrams/tdd-diag-3.svg)

The diagram above shows the printk ring buffer routing: KERN_INFO goes to dmesg ring buffer; KERN_ERR additionally triggers console output if the console log level permits.

![module_param: Load-time → Parameter Subsystem → sysfs → Runtime](./diagrams/tdd-diag-4.svg)

The diagram above shows the module parameter binding: `module_param(buffer_size, int, 0444)` connects the C variable to the insmod command-line parser and to the `/sys/module/hello/parameters/buffer_size` sysfs attribute file.

![printk Log Levels: KERN_* → Ring Buffer → dmesg / Console Routing](./diagrams/tdd-diag-5.svg)

The diagram above shows `__init` and `__exit` ELF section placement and the kernel's memory reclaim of `.init.text` pages after `hello_init` returns 0.
---
## 8. Test Specification
### 8.1 `hello_init` — Happy Path
| Test | Setup | Expected Outcome | Verification Command |
|---|---|---|---|
| Default load | `sudo insmod hello.ko` | Returns 0; dmesg contains `buffer_size=4096`; lsmod shows `hello` | `dmesg \| grep "hello: module loaded"` |
| Custom parameter | `sudo insmod hello.ko buffer_size=8192` | Returns 0; dmesg contains `buffer_size=8192`; sysfs shows 8192 | `cat /sys/module/hello/parameters/buffer_size` |
| Minimum value | `sudo insmod hello.ko buffer_size=1` | Returns 0; loads successfully | `lsmod \| grep hello` |
| Maximum value | `sudo insmod hello.ko buffer_size=1048576` | Returns 0; loads; sysfs shows 1048576 | `cat /sys/module/hello/parameters/buffer_size` |
### 8.2 `hello_init` — Edge Cases
| Test | Setup | Expected Outcome | Verification Command |
|---|---|---|---|
| Over-limit clamp | `sudo insmod hello.ko buffer_size=2000000` | Returns 0; KERN_WARNING in dmesg; sysfs shows 1048576 | `dmesg \| grep "clamping"` |
| Boundary over-limit | `sudo insmod hello.ko buffer_size=1048577` | Returns 0; KERN_WARNING; sysfs shows 1048576 | `cat /sys/module/hello/parameters/buffer_size` |
### 8.3 `hello_init` — Failure Cases
| Test | Setup | Expected Outcome | Verification Command |
|---|---|---|---|
| Zero value | `sudo insmod hello.ko buffer_size=0` | Returns `-EINVAL`; insmod fails; KERN_ERR in dmesg | `echo $?` after insmod (non-zero) |
| Negative value | `sudo insmod hello.ko buffer_size=-1` | Returns `-EINVAL`; insmod fails | `lsmod \| grep hello` (empty) |
| Large negative | `sudo insmod hello.ko buffer_size=-999999` | Returns `-EINVAL`; insmod fails | `dmesg \| grep "invalid buffer_size"` |
### 8.4 `hello_exit` — Happy Path
| Test | Setup | Expected Outcome | Verification Command |
|---|---|---|---|
| Clean unload | Load then `sudo rmmod hello` | KERN_INFO "module unloaded" in dmesg; not in lsmod | `dmesg \| grep "module unloaded"` |
### 8.5 `modinfo` — Metadata Tests
| Test | Command | Expected Output |
|---|---|---|
| License present | `modinfo hello.ko \| grep license` | `license: GPL` |
| Author present | `modinfo hello.ko \| grep author` | Non-empty author line |
| Description present | `modinfo hello.ko \| grep description` | Non-empty description line |
| Version present | `modinfo hello.ko \| grep version` | Non-empty version line |
| vermagic matches | `modinfo hello.ko \| grep vermagic` | Contains `$(uname -r)` |
| parm documented | `modinfo hello.ko \| grep parm` | `buffer_size:Internal buffer size...` |
### 8.6 Build Quality Test
```bash
# Must produce zero output (no warnings when -Werror is set)
make clean
make EXTRA_CFLAGS="-Werror" 2>&1 | grep -E "warning:|error:" | grep -v "^make"
# Expected: empty output
echo "Exit: $?"
# Expected: Exit: 0
```
### 8.7 Sysfs Permission Test
```bash
# 0444: any user can read, no user can write
cat /sys/module/hello/parameters/buffer_size  # succeeds as non-root
echo 999 > /sys/module/hello/parameters/buffer_size  # fails: Permission denied
```
---
## 9. Performance Targets
| Operation | Target | How to Measure |
|---|---|---|
| `insmod` total latency | < 50 ms | `time sudo insmod hello.ko` — wall clock |
| `hello_init` execution time | < 1 ms | `dmesg -T` timestamps before/after; typically ~100 µs |
| `printk` write to ring buffer | < 1 µs per call | `ktime_get_ns()` bracketing in init (remove before submission) |
| `.ko` file size | < 20 KB | `ls -lh hello.ko` |
| Module memory footprint | < 20 KB | `cat /proc/modules \| grep hello` — third column shows size in bytes |
| `make` build time | < 10 s | `time make` — dominated by Kbuild overhead, not source size |
| `modinfo` execution time | < 100 ms | `time modinfo hello.ko` — reads ELF without loading |
**Memory footprint breakdown for reference:**
- `.text` (init + exit function code): ~200 bytes
- `.data`/`.bss` (`buffer_size` variable): 4 bytes + alignment
- `.modinfo` section (metadata strings): ~200 bytes
- Kernel bookkeeping (`struct module`): ~2–4 KB allocated by kernel on load
- Symbol table retained by kernel: ~1 KB
Total loaded footprint: approximately 4–6 KB, well within the 20 KB target.
---
## 10. Concurrency Specification
M1 has no concurrency concerns. `buffer_size` is written exactly once by the kernel's module parameter subsystem before `hello_init` is called, and is read once inside `hello_init`. After init completes, `buffer_size` is effectively read-only (sysfs permissions are `0444`—no runtime writer). `hello_exit` does not access `buffer_size`.
No synchronization primitives are required for this milestone. Do not add mutexes, spinlocks, or atomics—they add complexity with no benefit and will confuse the reader of the code.
---
## 11. Common Pitfalls Reference
These are the most frequent mistakes made at this milestone, compiled for direct reference during implementation:
**Pitfall 1: Spaces instead of tabs in Makefile.**
Symptom: `Makefile:5: *** missing separator. Stop.`
Fix: The lines with `$(MAKE)` commands must begin with a literal tab (0x09). Use `:set list` in vim to see whitespace, or `cat -A Makefile | grep '^I'` to verify tabs.
**Pitfall 2: Wrong MODULE_LICENSE string.**
`"GPL"`, `"GPL v2"`, and `"GPL and additional rights"` all work. `"Dual BSD/GPL"` and `"Dual MIT/GPL"` work. `"Proprietary"` sets the taint bit. An empty string or no declaration at all taints the kernel. The exact string `"GPL"` is the simplest and most common choice.
**Pitfall 3: printk without log level.**
`printk("hello: loaded\n")` is valid C but uses `KERN_DEFAULT`, which may be filtered by the console log level and is not visible in `dmesg --level=info`. Always specify the log level explicitly: `printk(KERN_INFO "hello: loaded\n")`. Note that the log level string and the format string are adjacent string literals that the C preprocessor concatenates—there is no comma between them.
**Pitfall 4: Forgetting `\n` in printk format string.**
Unlike `printf`, `printk` does not add newlines automatically. Each message that should appear on its own line in dmesg output must end with `\n`.
**Pitfall 5: Using `ulong` type token in `module_param` for a variable declared `int`.**
`module_param(buffer_size, int, 0444)` — the second argument must match the C type of the variable. Using `ulong` with an `int` variable causes a type mismatch warning (which becomes an error with `-Werror`).
**Pitfall 6: Module metadata macros placed after function definitions.**
Conventionally, all `MODULE_*` macros appear at the top of the file after includes, before function definitions. While technically legal anywhere at file scope, the conventional placement is required by kernel coding style and expected by reviewers.
**Pitfall 7: Calling `module_init` or `module_exit` before defining the referenced function.**
C does not require forward declarations for functions used only in macro arguments, but defining the functions before the `module_init`/`module_exit` macros avoids any potential compiler confusion. The convention is: declare, define, then register.
---
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: build-kernel-module-m2 -->
# Technical Design Specification: Character Device Driver (`build-kernel-module-m2`)
---
## 1. Module Charter
This module implements a complete loadable kernel character device driver that exposes `/dev/mydevice` to userspace. It registers a dynamically allocated major/minor number pair, creates a device class and sysfs device node that udev converts to a `/dev/` entry automatically, and installs a `file_operations` dispatch table mapping `open(2)`, `read(2)`, `write(2)`, and `close(2)` syscalls to kernel handler functions. Data flows through a single `kzalloc`-backed kernel buffer of configurable size; `write()` copies bytes from userspace into this buffer using `copy_from_user`; `read()` copies bytes back to userspace using `copy_to_user` and advances `f_pos` so that a second read returns EOF (preventing infinite `cat` loops). An `atomic_t open_count` tracks concurrent opens and decrements on release; `try_module_get`/`module_put` prevents `rmmod` while any file descriptor is open.
This module does **NOT** implement ioctl, `/proc` entries, mutex protection of the buffer, wait queues, blocking I/O, O_NONBLOCK handling, or poll/select support. The buffer is unprotected against concurrent access — two simultaneous writers will race on `buffer_data_len` and `kernel_buffer`. This is the explicit unsolved problem that Milestone 4 addresses. The `buffer_size` module parameter is accepted as a `ulong` and validated in `mydevice_init`; no memory allocation occurs until `mydevice_init` runs.
**Upstream dependency:** `build-kernel-module-m1` — the Kbuild Makefile pattern, module metadata macros, printk discipline, and parameter validation pattern are all inherited. The `KDIR` variable and TAB-indented recipe format from M1's Makefile must be preserved exactly.
**Downstream dependency:** `build-kernel-module-m3` adds `unlocked_ioctl` and a `/proc` entry on top of the `file_operations` struct built here. `build-kernel-module-m4` adds `DEFINE_MUTEX`, `DECLARE_WAIT_QUEUE_HEAD`, and `.poll` — all of which require the buffer state variables (`kernel_buffer`, `buffer_data_len`, `buffer_size`) defined in this module.
**Invariants that must hold at all times:**
- After `mydevice_init` returns 0: `kernel_buffer != NULL`, `buffer_data_len == 0`, `buffer_size` is in `[1, 1048576]`, `dev_num` holds a valid allocated `dev_t`, `my_cdev` is registered, `mydevice_class` is non-NULL and non-ERR, `mydevice_device` is non-NULL and non-ERR, `/dev/mydevice` exists as a character special file.
- After `mydevice_exit` completes: all five resources (device, class, cdev, region, buffer) are freed/unregistered; `/dev/mydevice` is removed by udev; `kernel_buffer` is freed.
- `buffer_data_len <= buffer_size` at all times (enforced by write truncation).
- `open_count >= 0` at all times (atomic increment before decrement, `try_module_get` guards open path).
---
## 2. File Structure
Create files in this exact order:
```
mydevice/                         ← (1) working directory
├── Makefile                      ← (2) Kbuild delegation makefile
└── mydevice.c                    ← (3) kernel module source (sole .c file)
```
Build artifacts produced by `make` (do not create manually):
```
mydevice/
├── mydevice.ko                   ← loadable kernel object (primary output)
├── mydevice.o                    ← intermediate compiled object
├── mydevice.mod.c                ← Kbuild-generated module glue
├── mydevice.mod.o                ← compiled glue
├── Module.symvers                ← symbol export table (empty — no exports)
└── modules.order                 ← Kbuild tracking file
```
Verification artifacts (created manually during phase 6):
```
mydevice/
└── verify.sh                     ← (4) acceptance test script
```
There are no header files, subdirectories, or additional source files in this milestone. The shared `mydevice.h` header for ioctl definitions is introduced in M3.
---
## 3. Complete Data Model
### 3.1 Global Driver State Variables
All variables are declared `static` at file scope. They are initialized in `mydevice_init` and cleaned up in `mydevice_exit`. None are exported.
```c
/* Major:minor pair, allocated dynamically by alloc_chrdev_region */
static dev_t          dev_num;
/* Kernel character device handle, registered with the VFS cdev subsystem */
static struct cdev    my_cdev;
/* sysfs device class — creates /sys/class/mydevice/ */
static struct class  *mydevice_class;
/* sysfs device instance — triggers udev to create /dev/mydevice */
static struct device *mydevice_device;
/* Slab-allocated data buffer; size = buffer_size bytes; zeroed on alloc */
static char          *kernel_buffer;
/* Number of valid bytes currently stored in kernel_buffer ([0, buffer_size]) */
static size_t         buffer_data_len;
/* Total capacity of kernel_buffer; set from module_param, immutable after init */
static size_t         buffer_size = 4096;
/* Number of currently open file descriptors; safe for concurrent access */
static atomic_t       open_count = ATOMIC_INIT(0);
```
**Memory layout of globals in `.bss`/`.data` (approximate, 64-bit, 8-byte alignment):**
| Variable | Type | Size (bytes) | Notes |
|---|---|---|---|
| `dev_num` | `dev_t` (`unsigned int`) | 4 | Packed major:minor |
| `my_cdev` | `struct cdev` | ~128 | Contains embedded `struct kobject` |
| `mydevice_class` | pointer | 8 | NULL until `class_create` succeeds |
| `mydevice_device` | pointer | 8 | NULL until `device_create` succeeds |
| `kernel_buffer` | pointer | 8 | NULL until `kzalloc` succeeds |
| `buffer_data_len` | `size_t` | 8 | Zeroed initially |
| `buffer_size` | `size_t` | 8 | Default 4096 |
| `open_count` | `atomic_t` (wraps `int`) | 4 | `ATOMIC_INIT(0)` |
**Cache line analysis:** `kernel_buffer`, `buffer_data_len`, and `buffer_size` are accessed together on every `read()` and `write()` call. If the compiler places them consecutively in `.bss`, all three fit in a single 64-byte cache line, yielding spatial locality for the hot path. `open_count` is accessed on every `open()` and `release()` — it may share a cache line with the above, which is acceptable at this milestone (M4's mutex introduces explicit cache-line discipline).
### 3.2 Module Parameter
```c
static size_t buffer_size = 4096;
module_param(buffer_size, ulong, 0444);
MODULE_PARM_DESC(buffer_size, "Size of device buffer in bytes (default: 4096, range: 1–1048576)");
```
**Why `ulong` and not `int`:** `size_t` has no `module_param` type token. `ulong` matches `size_t` on 64-bit platforms and prevents negative values from being accepted by the parameter parser. Using `int` would allow `buffer_size=-1` to slip through the parser without error (signed overflow). Using `ulong` rejects negative inputs at the parameter parsing stage.
**Why `0444`:** The buffer size is an initialization-time parameter only; no runtime resize mechanism exists until M3's ioctl. Read-only sysfs exposure is correct. `0` would hide the parameter from sysfs, defeating the acceptance criterion. `0644` would allow unprivileged users to attempt to change it at runtime with no effect (since the buffer is already allocated), creating a misleading interface.
### 3.3 `struct cdev` — Character Device Handle
`struct cdev` is a kernel type defined in `<linux/cdev.h>`. You do not define it; you declare one statically and initialize it with `cdev_init()`.
| Field (internal) | What it stores | Your interaction |
|---|---|---|
| `ops` | Pointer to your `file_operations` | Set by `cdev_init(&my_cdev, &mydevice_fops)` |
| `owner` | `THIS_MODULE` pointer | Set manually: `my_cdev.owner = THIS_MODULE` |
| `dev` | `dev_t` of first device | Set by `cdev_add` |
| `count` | Number of minors | Set by `cdev_add` |
| `kobj` | Embedded kobject for reference counting | Managed internally |
### 3.4 `struct file_operations` Dispatch Table
```c
static const struct file_operations mydevice_fops = {
    .owner   = THIS_MODULE,
    .open    = mydevice_open,
    .release = mydevice_release,
    .read    = mydevice_read,
    .write   = mydevice_write,
};
```
**Every unset field defaults to `NULL`.** The VFS checks for `NULL` before dispatching. Unimplemented operations return sensible defaults:
- `.llseek = NULL` → uses `default_llseek`, which manipulates `f_pos` generically.
- `.unlocked_ioctl = NULL` → `ioctl(2)` returns `-ENOTTY`.
- `.poll = NULL` → `poll(2)` and `select(2)` always report the fd as ready (not ideal, but safe).
- `.mmap = NULL` → `mmap(2)` returns `-ENODEV`.
**Why `const`:** The `file_operations` struct is never modified after initialization. `const` communicates this intent and allows the linker to place it in read-only `.rodata`, preventing accidental corruption.
### 3.5 `struct file` Fields Used by This Driver
You receive a `struct file *filp` in every handler. You use exactly two fields:
| Field | Type | Used In | Purpose |
|---|---|---|---|
| `f_pos` | `loff_t *` | `read`, `write` | Per-fd byte offset; tracks how much has been read |
| `f_flags` | `unsigned int` | Not used in M2 | Would contain `O_NONBLOCK`; checked in M4 |
**`f_pos` is per-file-descriptor, not per-inode.** Multiple processes opening `/dev/mydevice` each get their own `struct file` with an independent `f_pos`. Two simultaneous readers may both read from offset 0 — the race on `buffer_data_len` and `kernel_buffer` contents is the concurrency problem deferred to M4.
### 3.6 `dev_t` Encoding
`dev_t` is a 32-bit value on Linux (defined as `unsigned int`). It packs major and minor numbers:
```
Bits 31–20: major (12 bits, upper portion)
Bits 19–8:  minor (12 bits, upper portion)
Bits  7–0:  minor (8 bits, lower portion)
```
In practice, use the provided macros exclusively:
```c
MAJOR(dev_num)         /* extract major number */
MINOR(dev_num)         /* extract minor number */
MKDEV(major, minor)    /* construct dev_t */
```
`alloc_chrdev_region` fills `dev_num` with both major and minor encoded. You pass `dev_num` unchanged to `cdev_add`, `device_create`, and `device_destroy`.

![VFS Dispatch Chain: open('/dev/mydevice') → file_operations](./diagrams/tdd-diag-6.svg)

---
## 4. Interface Contracts
### 4.1 `mydevice_open(inode, filp) → int`
```c
static int mydevice_open(struct inode *inode, struct file *filp);
```
**Context:** Process context. Called by VFS `do_open()` after the file descriptor is allocated but before the fd number is returned to userspace. Can sleep (though this implementation does not sleep).
**Parameters:**
- `inode`: Pointer to the inode for `/dev/mydevice`. Contains the `dev_t` in `inode->i_rdev`. Do not modify. Valid for the lifetime of the file.
- `filp`: Pointer to the newly allocated `struct file` for this open. `filp->f_pos` is 0 at this point. `filp->private_data` is NULL; may be set for per-fd state (not needed in M2). Valid until `mydevice_release` returns.
**Algorithm:**
1. Call `try_module_get(THIS_MODULE)`. If it returns 0 (false), the module is being unloaded — return `-ENODEV`.
2. Call `atomic_inc(&open_count)`.
3. Call `pr_info("mydevice: open() called, open_count=%d\n", atomic_read(&open_count))`.
4. Return `0`.
**Return values:**
| Return | Condition |
|---|---|
| `0` | Success; VFS returns fd to userspace |
| `-ENODEV` | `try_module_get` failed (module being unloaded) |
**Post-condition:** If returns 0, the module's reference count is incremented; `open_count` is one greater than before the call.
**Edge cases:**
- Two processes call `open()` simultaneously: `atomic_inc` is safe. `try_module_get` uses an internal spinlock and is race-free. Both opens succeed.
- `open()` called after `rmmod` starts: `try_module_get` returns 0 (module reference count is zero or dropping); `-ENODEV` is returned correctly.
### 4.2 `mydevice_release(inode, filp) → int`
```c
static int mydevice_release(struct inode *inode, struct file *filp);
```
**Context:** Process context. Called by VFS when the last reference to this `struct file` is dropped (i.e., when `close(fd)` is called and no `dup()`-derived copies remain open). Guaranteed to be called exactly once per successful `open()`.
**Algorithm:**
1. Call `atomic_dec(&open_count)`.
2. Call `pr_info("mydevice: release() called, open_count=%d\n", atomic_read(&open_count))`.
3. Call `module_put(THIS_MODULE)`.
4. Return `0`.
**Return values:** Always `0`. The VFS ignores the return value of `release` (it cannot propagate errors to `close(2)` after kernel 2.4), but the function signature requires `int`.
**Post-condition:** Module reference count is decremented by one. If this was the last open fd, `rmmod` may now proceed.
**Why `module_put` in release rather than on error paths in open:** `try_module_get` increments the refcount on success. `module_put` must be called exactly once for each successful `try_module_get`. Since `open` increments on success and `release` is called exactly once per successful `open`, the pairing is correct.
### 4.3 `mydevice_write(filp, buf, count, f_pos) → ssize_t`
```c
static ssize_t mydevice_write(struct file *filp, const char __user *buf,
                               size_t count, loff_t *f_pos);
```
**Context:** Process context. `buf` carries the `__user` annotation — it is a userspace virtual address. Do not dereference directly.
**Parameters:**
- `filp`: The open file descriptor struct. Not examined in M2 (O_NONBLOCK check deferred to M4).
- `buf`: Userspace pointer to source data. Must be treated as potentially invalid, hostile, or pointing to swapped-out memory. Access only via `copy_from_user`.
- `count`: Number of bytes the caller wants to write. May be 0. May exceed `buffer_size`.
- `f_pos`: Pointer to the file position. Updated by this function. Conventionally write resets this to 0 so subsequent reads start from the beginning (this driver overwrites the buffer on each write rather than appending).
**Algorithm:**
1. If `count == 0`: return `0` immediately.
2. Compute `to_copy = (count > buffer_size) ? buffer_size : count`. This silently caps the write at buffer capacity.
3. Call `memset(kernel_buffer, 0, buffer_size)`. This zeroes the entire buffer before the new write. Purpose: prevents stale data from a longer previous write from being visible after a shorter new write.
4. Call `not_copied = copy_from_user(kernel_buffer, buf, to_copy)`.
5. If `not_copied != 0`: `pr_err("mydevice: copy_from_user failed, %lu bytes not copied\n", not_copied)`, return `-EFAULT`.
6. Set `buffer_data_len = to_copy`.
7. Set `*f_pos = 0`. Rationale: this driver's read model is "read from position 0 after each write." Resetting f_pos here means the same fd can immediately `read()` and get the just-written data from the beginning. This matches the echo/cat acceptance test: `echo` writes, then `cat` (which opens a new fd, so f_pos starts at 0 anyway) reads.
8. `pr_info("mydevice: wrote %zu bytes\n", to_copy)`.
9. Return `(ssize_t)to_copy`.
**Return values:**
| Return | Condition |
|---|---|
| `> 0` | Bytes successfully written (may be less than `count` if truncated) |
| `0` | `count` was 0 |
| `-EFAULT` | `copy_from_user` returned non-zero (partial copy or bad userspace address) |
**Critical return-value semantics:** Returning `to_copy` when `to_copy < count` tells the caller that a partial write occurred. `write(2)` documentation specifies that the caller should retry the unwritten remainder. The shell's `echo` command ignores partial write returns (it writes in one shot). For programs that handle partial writes correctly (e.g., `dd`), this truncation is visible. This behavior is acceptable at M2; M3 ioctl resize allows the buffer to be enlarged.
**copy_from_user return value trap:** `copy_from_user` returns the number of bytes **NOT** copied (the residual), not the bytes successfully copied. `not_copied == 0` means complete success. `not_copied > 0` means partial failure — return `-EFAULT` in all non-zero cases.
### 4.4 `mydevice_read(filp, buf, count, f_pos) → ssize_t`
```c
static ssize_t mydevice_read(struct file *filp, char __user *buf,
                              size_t count, loff_t *f_pos);
```
**Context:** Process context. `buf` is a userspace pointer.
**Parameters:**
- `filp`: Open file descriptor. Not examined in M2.
- `buf`: Userspace destination buffer. Must be accessed only via `copy_to_user`.
- `count`: Maximum bytes the caller will accept. May be 0.
- `f_pos`: Current read position. This driver uses it as a byte offset into `kernel_buffer`.
**Algorithm:**
1. If `count == 0`: return `0`.
2. If `*f_pos >= (loff_t)buffer_data_len`: return `0` (EOF signal).
3. Compute `available = buffer_data_len - (size_t)*f_pos`.
4. Compute `to_copy = (count < available) ? count : available`.
5. Call `not_copied = copy_to_user(buf, kernel_buffer + *f_pos, to_copy)`.
6. If `not_copied != 0`: `pr_err("mydevice: copy_to_user failed, %lu bytes not copied\n", not_copied)`, return `-EFAULT`.
7. `*f_pos += (loff_t)to_copy`.
8. `pr_info("mydevice: read %zu bytes, f_pos now %lld\n", to_copy, *f_pos)`.
9. Return `(ssize_t)to_copy`.
**Return values:**
| Return | Condition |
|---|---|
| `> 0` | Bytes successfully copied to userspace |
| `0` | EOF: `*f_pos >= buffer_data_len`, or `count == 0` |
| `-EFAULT` | `copy_to_user` returned non-zero |
**The EOF invariant:** Returning `0` when `*f_pos >= buffer_data_len` is the mechanism that terminates `cat`'s read loop. If this check is omitted or the condition is wrong, `cat` calls `read()` forever. This is the most common bug at this milestone. **Verify with an explicit test before proceeding to the next phase.**
**f_pos advance:** `*f_pos += to_copy` is the essential step. Without it, every call to `read()` would re-read from the same offset. With it, multiple `read()` calls on the same fd stream through the buffer correctly, enabling `dd if=/dev/mydevice bs=4 count=1` to read only the first 4 bytes and `dd if=/dev/mydevice bs=4 count=1 skip=1` to read bytes 4–7.

![Major/Minor Number Registry and /dev/ Node Lifetime](./diagrams/tdd-diag-7.svg)

### 4.5 `mydevice_init(void) → int`
```c
static int __init mydevice_init(void);
```
**Context:** Process context (the `insmod` process). Can sleep — `kzalloc(GFP_KERNEL)` may sleep to reclaim pages.
**Full algorithm with goto error unwinding — see Section 5.**
**Return values:**
| Return | Condition |
|---|---|
| `0` | All 5 initialization steps succeeded |
| `-EINVAL` | `buffer_size` out of valid range |
| `-ENOMEM` | `kzalloc` failed |
| `< 0` | `alloc_chrdev_region` error (negative errno) |
| `< 0` | `cdev_add` error (negative errno) |
| `< 0` (via `PTR_ERR`) | `class_create` or `device_create` returned `ERR_PTR` |
### 4.6 `mydevice_exit(void) → void`
```c
static void __exit mydevice_exit(void);
```
**Context:** Process context (the `rmmod` process). **Cannot fail.** All cleanup must complete unconditionally.
**Full algorithm — see Section 5.2.**
---
## 5. Algorithm Specification
### 5.1 `mydevice_init` — Five-Step Initialization with Goto Unwind
The initialization follows a strict ordered sequence. Each step may fail independently. On failure, all previously completed steps must be undone in reverse order. The `goto` pattern is the canonical Linux kernel idiom for this — not a code smell, the expected pattern in `drivers/`.
```
Step 1: Validate buffer_size
Step 2: Allocate kernel buffer (kzalloc)         → err_free_buf
Step 3: Allocate major/minor (alloc_chrdev_region) → err_unreg_region
Step 4: Register cdev (cdev_init + cdev_add)     → err_del_cdev
Step 5: Create device class (class_create)       → err_destroy_class
Step 6: Create device instance (device_create)  → err_destroy_device
```
The goto labels form a chain where each label undoes exactly one step and falls through to the label below it:
```
err_destroy_device: → undo step 6
err_destroy_class:  → undo step 5
err_del_cdev:       → undo step 4
err_unreg_region:   → undo step 3
err_free_buf:       → undo step 2
(return ret)
```
**Detailed procedure:**
```
1. VALIDATE buffer_size:
   if (buffer_size == 0 || buffer_size > 1048576):
       pr_err("mydevice: invalid buffer_size %zu\n", buffer_size)
       return -EINVAL
   // No cleanup needed: nothing allocated yet
2. ALLOCATE BUFFER:
   kernel_buffer = kzalloc(buffer_size, GFP_KERNEL)
   if (!kernel_buffer):
       pr_err("mydevice: kzalloc(%zu) failed\n", buffer_size)
       return -ENOMEM
   buffer_data_len = 0
   // Invariant: kernel_buffer != NULL, buffer_data_len == 0
3. ALLOCATE MAJOR/MINOR:
   ret = alloc_chrdev_region(&dev_num, 0, 1, "mydevice")
   if (ret < 0):
       pr_err("mydevice: alloc_chrdev_region failed: %d\n", ret)
       goto err_free_buf
   pr_info("mydevice: major=%d minor=%d\n", MAJOR(dev_num), MINOR(dev_num))
   // Invariant: dev_num is valid, visible in /proc/devices
4. INITIALIZE AND REGISTER CDEV:
   cdev_init(&my_cdev, &mydevice_fops)
   my_cdev.owner = THIS_MODULE
   ret = cdev_add(&my_cdev, dev_num, 1)
   if (ret < 0):
       pr_err("mydevice: cdev_add failed: %d\n", ret)
       goto err_unreg_region
   // Invariant: VFS can now route open() on dev_num to mydevice_fops
   // (no /dev node yet, but routing is live)
5. CREATE DEVICE CLASS:
   mydevice_class = class_create(THIS_MODULE, "mydevice")
   if (IS_ERR(mydevice_class)):
       ret = PTR_ERR(mydevice_class)
       pr_err("mydevice: class_create failed: %d\n", ret)
       goto err_del_cdev
   // Invariant: /sys/class/mydevice/ exists
6. CREATE DEVICE INSTANCE:
   mydevice_device = device_create(mydevice_class, NULL, dev_num, NULL, "mydevice")
   if (IS_ERR(mydevice_device)):
       ret = PTR_ERR(mydevice_device)
       pr_err("mydevice: device_create failed: %d\n", ret)
       goto err_destroy_class
   // Invariant: udev has been triggered; /dev/mydevice is being created
   // (may take a few ms for udev to respond)
pr_info("mydevice: initialized, buffer=%zu bytes\n", buffer_size)
return 0
err_destroy_class:
    class_destroy(mydevice_class)
err_del_cdev:
    cdev_del(&my_cdev)
err_unreg_region:
    unregister_chrdev_region(dev_num, 1)
err_free_buf:
    kfree(kernel_buffer)
    return ret
```
**Why allocate buffer before registering the device:** If the buffer allocation fails after device registration, userspace might observe the `/dev/` node briefly before it disappears, causing confusing ENODEV errors. By allocating the buffer first (before any kernel registration), a buffer allocation failure is invisible to userspace entirely.
**Why step 4 (`cdev_add`) makes the device live immediately:** After `cdev_add`, if someone manually creates a device node with the correct major/minor via `mknod`, your `file_operations` would be invoked. The `/dev/` node does not exist yet (step 6 hasn't run), so this is theoretical — but it means the window between steps 4 and 6 is a partially-live state. Keep this window small; do not perform slow operations between them.
### 5.2 `mydevice_exit` — Reverse-Order Cleanup
```
mydevice_exit():
    device_destroy(mydevice_class, dev_num)
    // Triggers udev to remove /dev/mydevice (async; udev runs in userspace)
    class_destroy(mydevice_class)
    // Removes /sys/class/mydevice/
    cdev_del(&my_cdev)
    // Unregisters from VFS; no new opens possible on dev_num
    unregister_chrdev_region(dev_num, 1)
    // Releases the major number back to the pool
    kfree(kernel_buffer)
    // Returns buffer to slab allocator
    pr_info("mydevice: module unloaded\n")
```
**Order matters:** `cdev_del` must come before `unregister_chrdev_region`. `device_destroy` must come before `class_destroy` (a class with registered devices cannot be destroyed). `kfree` is safe to call last because no new `read()`/`write()` calls can arrive after `cdev_del` completes — the VFS routing is broken. The `try_module_get`/`module_put` mechanism guarantees no handler is executing when `mydevice_exit` begins, because `rmmod` checks the module reference count before calling `exit`.
### 5.3 `copy_from_user` Internal Sequence (x86_64, SMAP-enabled CPU)
Understanding this sequence is necessary for writing correct error handling:
```
copy_from_user(dst_kernel, src_user, n):
    1. access_ok(src_user, n):
       - Checks src_user + n <= TASK_SIZE_MAX (userspace boundary)
       - Returns false if src_user is a kernel address or NULL
       - On failure: copy_from_user returns n (full residual, 0 bytes copied)
    2. STAC instruction (Set AC flag in RFLAGS)
       - Hardware: CPU now permits ring-0 access to ring-3 pages
       - Prevents SMAP fault for the duration of the copy
    3. Exception-table-guarded memcpy loop:
       - Copies bytes from src_user to dst_kernel
       - If a page fault occurs mid-copy (page swapped out):
           → page fault handler checks exception table
           → swap-in occurs, copy resumes
           → if swap-in fails, copy terminates early
           → residual = bytes not copied, stored in %rax
    4. CLAC instruction (Clear AC flag in RFLAGS)
       - Hardware: SMAP protection re-enabled
    5. Return residual (bytes NOT copied):
       - 0 = complete success
       - > 0 = partial or total failure
```
**The `__user` annotation obligation:** Functions that receive `const char __user *buf` must never pass this pointer to any function that dereferences it directly (e.g., `strlen`, `memcpy`, `strncpy`). Even if SMAP is disabled on the test machine (some VMs disable it), the annotation is a compile-time contract enforced by `sparse` and `smatch`. Write `copy_from_user`-correct code from the start.
### 5.4 copy_to_user Internal Sequence
Symmetric to `copy_from_user` with reversed direction:
```
copy_to_user(dst_user, src_kernel, n):
    1. access_ok(dst_user, n): validates userspace address range
    2. STAC: enable userspace memory access
    3. Exception-table-guarded copy from kernel → userspace page
       - Page faults on dst_user cause swap-in (userspace page may be swapped)
       - Copy completes or terminates early with residual count
    4. CLAC: re-enable SMAP
    5. Return residual (bytes NOT copied)
```

![file_operations Struct: Field Layout and Dispatch](./diagrams/tdd-diag-8.svg)

### 5.5 EOF Detection Logic for Read
The EOF contract is binary and non-negotiable:
```
if (*f_pos >= (loff_t)buffer_data_len):
    return 0   ← EOF: caller must stop reading
```
**What "EOF" means to callers:**
- `cat`: stops the read loop, closes fd, prints output
- `read(2)` in C: caller checks `n == 0`, breaks loop
- `dd`: terminates after the block that returned 0
**What happens without EOF:**
- `cat` calls `read()` in a tight loop forever
- Each call returns 0 bytes (if the check is: `if available == 0, return 0 not -1`)  
Wait — returning `0` IS EOF. The confusion: if `*f_pos` never advances (because the advance is omitted), `read()` returns the same bytes repeatedly. If `f_pos` is advanced but the EOF check is absent, `read()` returns 0 bytes after the data is consumed but still returns 0-byte "success" forever — and the caller's loop condition `while (n > 0)` terminates correctly. The real danger is returning a **positive** value after data is exhausted (e.g., returning some garbage bytes from beyond `buffer_data_len`). The check at step 2 of `mydevice_read` prevents this.
---
## 6. Error Handling Matrix
| Error Condition | Detected By | Response | User-Visible | Recovery |
|---|---|---|---|---|
| `buffer_size == 0` | `mydevice_init` validation | Return `-EINVAL`; no allocation/registration | `insmod` fails: "Invalid argument" + dmesg KERN_ERR | Re-run `insmod` with valid `buffer_size` |
| `buffer_size > 1048576` | `mydevice_init` validation | Return `-EINVAL`; no allocation/registration | `insmod` fails: "Invalid argument" | Re-run with `buffer_size <= 1048576` |
| `kzalloc` returns NULL | `!kernel_buffer` check after `kzalloc` | Return `-ENOMEM`; no unwind needed (nothing registered) | `insmod` fails: "Cannot allocate memory" | Free memory, retry with smaller `buffer_size` |
| `alloc_chrdev_region` fails | `ret < 0` check | `goto err_free_buf`; `kfree(kernel_buffer)` | `insmod` fails; KERN_ERR in dmesg | Check `dmesg` for error code; retry |
| `cdev_add` fails | `ret < 0` check | `goto err_unreg_region`; undo region + buffer | `insmod` fails | Unusual; indicates kernel internal error |
| `class_create` returns ERR_PTR | `IS_ERR(mydevice_class)` | `ret = PTR_ERR(...)`; `goto err_del_cdev` | `insmod` fails | Check if `/sys/class/mydevice` already exists (stale from crashed previous load) |
| `device_create` returns ERR_PTR | `IS_ERR(mydevice_device)` | `ret = PTR_ERR(...)`; `goto err_destroy_class` | `insmod` fails | Verify udev is running; check dmesg |
| `copy_from_user` partial/failure | `not_copied != 0` after call | Return `-EFAULT`; `buffer_data_len` NOT updated (buffer in partially-zeroed but valid state from `memset`) | `write(2)` returns `-1`, `errno = EFAULT` | Userspace bug: passing invalid pointer |
| `copy_to_user` partial/failure | `not_copied != 0` after call | Return `-EFAULT`; `*f_pos` NOT advanced (position not corrupted) | `read(2)` returns `-1`, `errno = EFAULT` | Userspace bug: passing invalid/readonly pointer |
| `try_module_get` fails | Return of `0` from `try_module_get` | Return `-ENODEV` from `open()` | `open(2)` returns `-1`, `errno = ENODEV` | Normal: module is being unloaded |
| Write count > buffer_size | Comparison `count > buffer_size` | Silent truncation: `to_copy = buffer_size` | `write(2)` returns `to_copy` (less than `count`) | Userspace must retry remaining bytes if needed |
| Read from empty buffer | `*f_pos >= buffer_data_len` | Return `0` (EOF) | `read(2)` returns `0` | Normal: no data has been written yet |
| `rmmod` with open fds | `try_module_get`/`module_put` refcount | `rmmod` fails: "Device or resource busy" | User sees `rmmod` error message | Close all fds (`lsof /dev/mydevice`), then retry |
**Invariant after any error path:** `kernel_buffer`, `dev_num`, `my_cdev`, `mydevice_class`, `mydevice_device` — all are either in a fully consistent state (all initialized) or in a fully cleaned-up state (all freed/unregistered). No partial initialization survives a failed `mydevice_init`.
**The `IS_ERR` / `PTR_ERR` contract:** Kernel functions that return pointers signal errors by returning `ERR_PTR(errno)` — a value in the range `(-4096, 0)` cast to a pointer. On x86_64, valid kernel pointers are above `0xffff800000000000`; values near zero are never valid kernel addresses. `IS_ERR(ptr)` checks `(unsigned long)ptr > (unsigned long)-MAX_ERRNO`. **Never check these pointers for NULL.** `class_create` may return a non-NULL invalid pointer; checking `!= NULL` would silently proceed with a corrupted pointer, causing a crash when it's later dereferenced.

![copy_from_user / copy_to_user: SMAP Sequence and Address Validation](./diagrams/tdd-diag-9.svg)

---
## 7. Implementation Sequence with Checkpoints
### Phase 1: `alloc_chrdev_region` + `cdev_init` + `cdev_add` (1–2 hours)
Create `mydevice/Makefile`:
```makefile
obj-m += mydevice.o
KDIR ?= /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)
# Include current dir so future header files are found
ccflags-y := -I$(PWD)
all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules
clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
```
**TAB characters are mandatory** on the `$(MAKE)` lines.
Create `mydevice/mydevice.c` with the skeleton:
```c
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/fs.h>        /* alloc_chrdev_region, unregister_chrdev_region */
#include <linux/cdev.h>      /* cdev_init, cdev_add, cdev_del, struct cdev */
#include <linux/device.h>    /* class_create, device_create, class_destroy, device_destroy */
#include <linux/slab.h>      /* kzalloc, kfree */
#include <linux/uaccess.h>   /* copy_to_user, copy_from_user */
#include <linux/atomic.h>    /* atomic_t, atomic_inc, atomic_dec, atomic_read */
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name <you@example.com>");
MODULE_DESCRIPTION("Character device driver — Milestone 2");
MODULE_VERSION("0.2");
static size_t buffer_size = 4096;
module_param(buffer_size, ulong, 0444);
MODULE_PARM_DESC(buffer_size, "Device buffer size in bytes (default: 4096, max: 1048576)");
static dev_t          dev_num;
static struct cdev    my_cdev;
static struct class  *mydevice_class;
static struct device *mydevice_device;
static char          *kernel_buffer;
static size_t         buffer_data_len;
static atomic_t       open_count = ATOMIC_INIT(0);
/* Forward declarations */
static int     mydevice_open   (struct inode *inode, struct file *filp);
static int     mydevice_release(struct inode *inode, struct file *filp);
static ssize_t mydevice_read   (struct file *filp, char __user *buf,
                                 size_t count, loff_t *f_pos);
static ssize_t mydevice_write  (struct file *filp, const char __user *buf,
                                 size_t count, loff_t *f_pos);
static const struct file_operations mydevice_fops = {
	.owner   = THIS_MODULE,
	.open    = mydevice_open,
	.release = mydevice_release,
	.read    = mydevice_read,
	.write   = mydevice_write,
};
/* Stub implementations — replaced in subsequent phases */
static int mydevice_open(struct inode *inode, struct file *filp) { return 0; }
static int mydevice_release(struct inode *inode, struct file *filp) { return 0; }
static ssize_t mydevice_read(struct file *filp, char __user *buf,
                              size_t count, loff_t *f_pos) { return 0; }
static ssize_t mydevice_write(struct file *filp, const char __user *buf,
                               size_t count, loff_t *f_pos) { return (ssize_t)count; }
static int __init mydevice_init(void)
{
	int ret;
	if (buffer_size == 0 || buffer_size > 1048576) {
		pr_err("mydevice: invalid buffer_size %zu\n", buffer_size);
		return -EINVAL;
	}
	kernel_buffer = kzalloc(buffer_size, GFP_KERNEL);
	if (!kernel_buffer) {
		pr_err("mydevice: kzalloc(%zu) failed\n", buffer_size);
		return -ENOMEM;
	}
	buffer_data_len = 0;
	ret = alloc_chrdev_region(&dev_num, 0, 1, "mydevice");
	if (ret < 0) {
		pr_err("mydevice: alloc_chrdev_region failed: %d\n", ret);
		goto err_free_buf;
	}
	pr_info("mydevice: major=%d minor=%d\n", MAJOR(dev_num), MINOR(dev_num));
	cdev_init(&my_cdev, &mydevice_fops);
	my_cdev.owner = THIS_MODULE;
	ret = cdev_add(&my_cdev, dev_num, 1);
	if (ret < 0) {
		pr_err("mydevice: cdev_add failed: %d\n", ret);
		goto err_unreg_region;
	}
	pr_info("mydevice: cdev registered (no /dev node yet)\n");
	return 0;
err_unreg_region:
	unregister_chrdev_region(dev_num, 1);
err_free_buf:
	kfree(kernel_buffer);
	return ret;
}
static void __exit mydevice_exit(void)
{
	cdev_del(&my_cdev);
	unregister_chrdev_region(dev_num, 1);
	kfree(kernel_buffer);
	pr_info("mydevice: module unloaded\n");
}
module_init(mydevice_init);
module_exit(mydevice_exit);
```
**Checkpoint 1:** Run `make EXTRA_CFLAGS="-Werror"`. Expect zero warnings. Run `sudo insmod mydevice.ko`. Check:
```bash
dmesg | grep mydevice
# Must show: "major=NNN minor=0" and "cdev registered"
grep mydevice /proc/devices
# Must show: "NNN mydevice" where NNN is the allocated major
sudo rmmod mydevice
```
`/dev/mydevice` does NOT exist yet — this is correct. The cdev is registered (VFS routing live) but no udev trigger has fired.
### Phase 2: `class_create` + `device_create` — `/dev/mydevice` via udev (0.5–1 hour)
Extend `mydevice_init` to add steps 5 and 6 after `cdev_add`. Extend `mydevice_exit` to call `device_destroy` and `class_destroy` before `cdev_del`. Add the appropriate goto labels.
The complete `mydevice_init` and `mydevice_exit` at this point match the algorithm in Section 5.1 exactly.
**Checkpoint 2:**
```bash
sudo insmod mydevice.ko
ls -l /dev/mydevice
# Must show: crw------- 1 root root NNN, 0 <date> /dev/mydevice
# 'c' = character device; NNN = your major number
ls /sys/class/mydevice/
# Must show: mydevice/
grep mydevice /proc/devices
# Must show: NNN mydevice
sudo rmmod mydevice
ls /dev/mydevice 2>&1
# Must show: ls: cannot access '/dev/mydevice': No such file or directory
```
If `/dev/mydevice` doesn't appear within 1 second of `insmod`, udev may not be running. Check: `systemctl status udev`. On minimal containers, run `sudo mknod /dev/mydevice c $(grep mydevice /proc/devices | awk '{print $1}') 0` manually as a workaround, but in a standard Linux environment udev handles this automatically.
### Phase 3: kmalloc Buffer + open/release with atomic_t (0.5–1 hour)
Replace the stub `mydevice_open` and `mydevice_release` with the implementations from Section 4.1 and 4.2.
**Checkpoint 3:**
```bash
sudo insmod mydevice.ko
# Open and immediately close:
sudo bash -c 'exec 3>/dev/mydevice; exec 3>&-'
dmesg | tail -4
# Must show:
#   mydevice: open() called, open_count=1
#   mydevice: release() called, open_count=0
# Verify module refcount prevents rmmod while open:
sudo bash -c 'exec 3>/dev/mydevice; sudo rmmod mydevice'
# rmmod must fail: "rmmod: ERROR: Module mydevice is in use"
# Close the fd first:
# (the subshell exits, closing fd 3)
sudo rmmod mydevice  # must succeed now
```
### Phase 4: write with copy_from_user + read with f_pos and EOF (1.5–2 hours)
Replace stub `mydevice_write` and `mydevice_read` with the implementations from Sections 4.3 and 4.4.
**Implementation order within this phase:**
1. Implement `mydevice_write` completely.
2. Test write in isolation: `echo "test" | sudo tee /dev/mydevice`. Verify dmesg shows "wrote N bytes." Do NOT test read yet.
3. Implement `mydevice_read` completely.
4. Test the echo/cat round-trip (Checkpoint 4 below).
**Checkpoint 4 — the core acceptance test:**
```bash
sudo insmod mydevice.ko
sudo chmod 666 /dev/mydevice   # for easier testing; production uses udev rules
# Test 1: basic round-trip
echo "hello kernel" | tee /dev/mydevice
cat /dev/mydevice
# Must output: hello kernel
# Must terminate (not loop)
# Test 2: binary data integrity
echo -n "ABCDEFGH" | tee /dev/mydevice > /dev/null
cat /dev/mydevice | xxd | head -1
# Must show: 41424344 45464748  ABCDEFGH
# Test 3: f_pos tracking with dd
echo -n "0123456789" | tee /dev/mydevice > /dev/null
dd if=/dev/mydevice bs=4 count=1 2>/dev/null | xxd
# Must show: 30313233  (bytes 0-3: "0123")
# Note: dd opens a new fd each time, f_pos starts at 0 for each open
# Test 4: EOF after full read
echo -n "abc" | tee /dev/mydevice > /dev/null
python3 -c "
fd = open('/dev/mydevice', 'rb')
d1 = fd.read(2)  # reads 'ab'
d2 = fd.read(2)  # reads 'c' (1 byte left)
d3 = fd.read(2)  # reads 0 bytes (EOF)
print(f'd1={d1} d2={d2} d3={d3}')
assert d3 == b'', 'EOF not signaled!'
print('EOF test PASSED')
fd.close()
"
# Test 5: buffer capacity clamping
python3 -c "print('x' * 8192, end='')" | tee /dev/mydevice > /dev/null
wc -c /dev/mydevice
# cat reads all; wc -c should show 4096 (clamped to buffer_size)
# Note: wc -c on /dev/mydevice reads until EOF — valid test
sudo rmmod mydevice
```
### Phase 5: Complete goto Error Unwind + module_exit Reverse Cleanup (0.5–1 hour)
Verify the complete goto chain is in place (matches Section 5.1 exactly). Verify `mydevice_exit` cleanup order matches Section 5.2 exactly.
**Checkpoint 5 — forced allocation failure test (simulate ENOMEM):**
```bash
# Test invalid buffer_size paths:
sudo insmod mydevice.ko buffer_size=0
echo "Exit: $?"  # Must be non-zero
dmesg | tail -2   # Must show: "invalid buffer_size 0"
sudo insmod mydevice.ko buffer_size=2000000
echo "Exit: $?"  # Must be non-zero
dmesg | tail -2   # Must show: "invalid buffer_size 2000000"
```
To test `kzalloc` failure, you would need to inject memory pressure (e.g., `stress-ng --vm 1 --vm-bytes 95%`). For this milestone, verifying the code path exists via code review is sufficient — the `!kernel_buffer` check and `return -ENOMEM` must be present.
### Phase 6: Verification Script (1–1.5 hours)
Create `mydevice/verify.sh`:
```bash
#!/bin/bash
# verify.sh — Milestone 2 acceptance test
# Usage: sudo bash verify.sh
set -euo pipefail
MODULE="mydevice"
DEVICE="/dev/mydevice"
PASS=0
FAIL=0
pass() { echo "  PASS: $1"; PASS=$((PASS+1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL+1)); }
# Clean state
lsmod | grep -q "^${MODULE} " && rmmod "${MODULE}" 2>/dev/null || true
echo "=== Milestone 2 Verification ==="
echo ""
echo "--- Build ---"
[ -f "./mydevice.ko" ] && pass "mydevice.ko exists" || { fail "mydevice.ko missing"; exit 1; }
echo ""
echo "--- Load and /dev node ---"
insmod ./mydevice.ko
sleep 0.3
ls "${DEVICE}" > /dev/null 2>&1 && pass "/dev/mydevice exists" || fail "/dev/mydevice not created"
stat -c "%F" "${DEVICE}" | grep -q "character" && pass "/dev/mydevice is character device" || fail "not a character device"
MAJOR_NUM=$(grep mydevice /proc/devices | awk '{print $1}')
[ -n "${MAJOR_NUM}" ] && pass "/proc/devices shows mydevice (major=${MAJOR_NUM})" || fail "/proc/devices entry missing"
ls /sys/class/mydevice/ > /dev/null 2>&1 && pass "/sys/class/mydevice/ exists" || fail "sysfs class missing"
chmod 666 "${DEVICE}"
echo ""
echo "--- open/release lifecycle ---"
BEFORE=$(dmesg | wc -l)
exec 3>"${DEVICE}"; exec 3>&-
sleep 0.1
AFTER_MSG=$(dmesg | tail -5)
echo "${AFTER_MSG}" | grep -q "open_count=1" && pass "open increments open_count" || fail "open_count not incremented"
echo "${AFTER_MSG}" | grep -q "open_count=0" && pass "release decrements open_count" || fail "open_count not decremented"
echo ""
echo "--- write ---"
echo -n "hello milestone2" | tee "${DEVICE}" > /dev/null
dmesg | tail -3 | grep -q "wrote 16 bytes" && pass "write: 16 bytes written" || fail "write count mismatch in dmesg"
echo ""
echo "--- read with EOF ---"
DATA=$(cat "${DEVICE}")
[ "${DATA}" = "hello milestone2" ] && pass "read: data matches written content" || fail "read: data mismatch: '${DATA}'"
# cat must terminate — if it doesn't, this script hangs; timeout handles that
OUT=$(timeout 2 cat "${DEVICE}" 2>&1)
[ $? -eq 0 ] && pass "cat terminates (EOF signaled correctly)" || fail "cat hung or errored"
echo ""
echo "--- f_pos tracking ---"
echo -n "0123456789" | tee "${DEVICE}" > /dev/null
BYTES=$(dd if="${DEVICE}" bs=4 count=1 2>/dev/null | wc -c)
[ "${BYTES}" = "4" ] && pass "dd bs=4 count=1 reads exactly 4 bytes" || fail "dd read ${BYTES} bytes, expected 4"
echo ""
echo "--- buffer capacity clamping ---"
python3 -c "import sys; sys.stdout.buffer.write(b'X' * 8192)" | tee "${DEVICE}" > /dev/null || true
RESULT=$(cat "${DEVICE}" | wc -c)
[ "${RESULT}" = "4096" ] && pass "8192-byte write clamped to 4096 (buffer_size)" || fail "expected 4096 bytes, got ${RESULT}"
echo ""
echo "--- binary data integrity ---"
printf '\x01\x02\x03\x04\xFF\xFE\xFD' | tee "${DEVICE}" > /dev/null
HEX=$(cat "${DEVICE}" | xxd -p | head -c 14)
[ "${HEX}" = "01020304fffefd" ] && pass "binary data round-trip correct" || fail "binary data corrupted: '${HEX}'"
echo ""
echo "--- module_put prevents rmmod while open ---"
exec 3>"${DEVICE}"
set +e
rmmod "${MODULE}" 2>/dev/null
RC=$?
set -e
exec 3>&-
[ "${RC}" -ne 0 ] && pass "rmmod fails while fd is open (module_put working)" || fail "rmmod succeeded with open fd (module_put missing)"
echo ""
echo "--- custom buffer_size parameter ---"
rmmod "${MODULE}" 2>/dev/null || true
insmod ./mydevice.ko buffer_size=512
sleep 0.2
ls "${DEVICE}" > /dev/null 2>&1 && pass "/dev/mydevice created with buffer_size=512" || fail "/dev/mydevice missing with custom buffer_size"
chmod 666 "${DEVICE}"
python3 -c "import sys; sys.stdout.buffer.write(b'Y' * 600)" | tee "${DEVICE}" > /dev/null || true
COUNT=$(cat "${DEVICE}" | wc -c)
[ "${COUNT}" = "512" ] && pass "600-byte write clamped to 512 (custom buffer_size)" || fail "expected 512, got ${COUNT}"
echo ""
echo "--- clean unload ---"
rmmod "${MODULE}"
sleep 0.2
ls "${DEVICE}" 2>&1 | grep -q "No such file" && pass "/dev/mydevice removed after rmmod" || fail "/dev/mydevice still exists after rmmod"
lsmod | grep -q "^${MODULE} " && fail "module still in lsmod" || pass "module not in lsmod after rmmod"
echo ""
echo "=============================="
echo "Results: ${PASS} passed, ${FAIL} failed"
echo "=============================="
[ "${FAIL}" -eq 0 ] && exit 0 || exit 1
```
**Checkpoint 6:** `sudo bash verify.sh` — all lines print `PASS`. Zero `FAIL` lines. Exit code 0.

![f_pos Tracking and EOF Contract: echo/cat Round-Trip](./diagrams/tdd-diag-10.svg)

---
## 8. Test Specification
### 8.1 `mydevice_open` — Happy Path
| Test | Setup | Expected | Verification |
|---|---|---|---|
| Single open | `exec 3>/dev/mydevice` | Returns 0; `open_count=1` in dmesg | `dmesg \| grep open_count=1` |
| Multiple sequential opens | Open, close, open again | Each open: count increments; each close: count decrements | dmesg sequence |
| Concurrent opens (2 fds) | `exec 3>/dev/mydevice; exec 4>/dev/mydevice` | Both succeed; `open_count=2` | `dmesg \| grep open_count=2` |
### 8.2 `mydevice_open` — Edge Cases
| Test | Setup | Expected | Verification |
|---|---|---|---|
| open with O_RDONLY | `cat /dev/mydevice` | Succeeds (no write restriction) | Exit 0 |
| open with O_WRONLY | `echo x > /dev/mydevice` | Succeeds | dmesg open message |
| open with O_RDWR | `exec 3<>/dev/mydevice` | Succeeds | dmesg open message |
### 8.3 `mydevice_release` — Happy Path
| Test | Setup | Expected | Verification |
|---|---|---|---|
| Close after open | `exec 3>/dev/mydevice; exec 3>&-` | `open_count=0` in dmesg | `dmesg \| grep open_count=0` |
| rmmod after all closes | Close all fds then `rmmod` | rmmod succeeds | `lsmod \| grep mydevice` returns empty |
### 8.4 `mydevice_write` — Happy Path
| Test | Setup | Expected | Verification |
|---|---|---|---|
| ASCII string | `echo -n "hello" > /dev/mydevice` | Returns 5; dmesg "wrote 5 bytes" | `dmesg \| grep "wrote 5 bytes"` |
| Full buffer write | Write exactly `buffer_size` bytes | Returns `buffer_size`; dmesg shows same count | `wc -c < /dev/mydevice` after read |
| Zero-byte write | `dd if=/dev/null of=/dev/mydevice` | Returns 0 immediately | dmesg shows "wrote 0 bytes" OR no message (count==0 early return) |
| Binary data | `printf '\x00\x01\xFF' > /dev/mydevice` | Returns 3; round-trip preserves bytes | `cat /dev/mydevice \| xxd` |
### 8.5 `mydevice_write` — Truncation Edge Case
| Test | Setup | Expected | Verification |
|---|---|---|---|
| Over-capacity write | `python3 -c "print('x'*8192,end='')" > /dev/mydevice` | Returns 4096 (buffer_size default); read returns 4096 bytes | `cat /dev/mydevice \| wc -c` outputs `4096` |
| Write exactly buffer_size + 1 | Write 4097 bytes | Returns 4096 | `cat \| wc -c` outputs `4096` |
### 8.6 `mydevice_read` — Happy Path
| Test | Setup | Expected | Verification |
|---|---|---|---|
| Read after write | Write "hello", then `cat` | "hello" appears; cat terminates | `cat /dev/mydevice` output |
| Partial read (dd) | Write "0123456789", `dd bs=4 count=1` | First 4 bytes ("0123") returned | `dd \| xxd` |
| Sequential partial reads | Write 10 bytes, read 4, read 4, read 4 | 4, 4, 2, 0 bytes returned (EOF on 4th) | Python test with explicit `fd.read()` |
| Read from empty buffer | No prior write; `cat /dev/mydevice` | Immediate EOF (0 bytes); cat terminates | Exit code 0; no output |
### 8.7 `mydevice_read` — EOF Correctness (Critical)
```bash
# This test must pass — infinite loop here means read is broken
python3 -c "
import os, time
fd = os.open('/dev/mydevice', os.O_RDONLY)
start = time.time()
reads = 0
while True:
    data = os.read(fd, 4096)
    reads += 1
    if len(data) == 0:
        break
    if time.time() - start > 2.0:
        print('FAIL: read loop did not terminate in 2 seconds')
        os.close(fd)
        exit(1)
print(f'PASS: terminated after {reads} read call(s)')
os.close(fd)
"
```
### 8.8 `mydevice_write` / `mydevice_read` — EFAULT Handling
```bash
# Generate EFAULT by passing a kernel address as the userspace buffer
# This requires a small C test program since Python cannot construct invalid addresses:
cat > /tmp/test_efault.c << 'EOF'
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
int main(void) {
    int fd = open("/dev/mydevice", O_RDWR);
    if (fd < 0) { perror("open"); return 1; }
    /* NULL is an invalid userspace pointer — copy_from_user will fail */
    ssize_t ret = write(fd, NULL, 10);
    if (ret < 0 && errno == EFAULT)
        printf("PASS: write with NULL returns EFAULT\n");
    else
        printf("FAIL: expected EFAULT, got ret=%zd errno=%d (%s)\n",
               ret, errno, strerror(errno));
    /* Write some valid data first, then read with NULL */
    write(fd, "hello", 5);
    ret = read(fd, NULL, 5);
    if (ret < 0 && errno == EFAULT)
        printf("PASS: read with NULL returns EFAULT\n");
    else
        printf("FAIL: expected EFAULT, got ret=%zd errno=%d (%s)\n",
               ret, errno, strerror(errno));
    close(fd);
    return 0;
}
EOF
gcc -o /tmp/test_efault /tmp/test_efault.c
sudo /tmp/test_efault
```
### 8.9 `mydevice_init` — Registration Correctness
| Test | Verification Command | Expected |
|---|---|---|
| Major number allocated | `grep mydevice /proc/devices` | One line with major and "mydevice" |
| Major is dynamic (not hardcoded) | Load twice (rmmod between), compare majors | May differ — dynamic allocation |
| `/dev/mydevice` is character device | `stat -c "%F" /dev/mydevice` | "character special file" |
| `/sys/class/mydevice/mydevice/` exists | `ls /sys/class/mydevice/mydevice/` | Directory with `dev`, `uevent`, etc. |
| Device major/minor in sysfs | `cat /sys/class/mydevice/mydevice/dev` | `MAJOR:MINOR` matching `/proc/devices` |
### 8.10 `mydevice_exit` — Cleanup Correctness
| Test | Verification Command | Expected |
|---|---|---|
| `/dev/mydevice` removed | `ls /dev/mydevice 2>&1` | "No such file or directory" |
| `/proc/devices` entry removed | `grep mydevice /proc/devices` | Empty |
| `/sys/class/mydevice/` removed | `ls /sys/class/mydevice/ 2>&1` | "No such file or directory" |
| Module not in lsmod | `lsmod \| grep mydevice` | Empty |
| dmesg exit message | `dmesg \| tail -2` | "mydevice: module unloaded" |

![module_init Goto Error Unwinding: Reverse-Order Cleanup](./diagrams/tdd-diag-11.svg)

---
## 9. Performance Targets
| Operation | Target | How to Measure |
|---|---|---|
| `open()` latency | < 2 µs | `strace -T cat /dev/mydevice 2>&1 \| grep open` — look at syscall duration in brackets |
| `close()` latency | < 2 µs | `strace -T` on the close syscall |
| `write()` 4KB | < 10 µs | `dd if=/dev/zero of=/dev/mydevice bs=4096 count=1 2>&1` — compute: elapsed / count |
| `read()` 4KB | < 10 µs | `dd if=/dev/mydevice of=/dev/null bs=4096 count=1 2>&1` after filling buffer |
| `insmod` total | < 100 ms | `time sudo insmod mydevice.ko` |
| `rmmod` total | < 50 ms | `time sudo rmmod mydevice` |
| Module `.ko` size | < 25 KB | `ls -lh mydevice.ko` |
| `kernel_buffer` allocation | `buffer_size` bytes exactly | slab allocator rounds up to power-of-2 >= buffer_size |
| Module loaded memory footprint | < `buffer_size` + 8 KB overhead | `cat /proc/modules \| grep mydevice \| awk '{print $2}'` for module struct size |
**Detailed latency breakdown for `write()` 4KB:**
| Sub-operation | Estimated Cost |
|---|---|
| Syscall boundary crossing (ring 3→0) | ~100 ns |
| VFS dispatch through `file_operations` | ~20 ns (pointer dereference, L1 cache hit) |
| `memset(kernel_buffer, 0, 4096)` | ~1–2 µs (64 cache lines × ~15 ns each, first write cold) |
| `copy_from_user` 4KB (warm cache) | ~500 ns (STAC + rep movsq + CLAC) |
| `pr_info` write to ring buffer | ~200 ns |
| Return + ring 0→3 | ~100 ns |
| **Total** | **~3–5 µs** |
**Memory allocation model:**
The slab allocator (`SLUB` on modern kernels) services `kzalloc(buffer_size, GFP_KERNEL)` requests by:
- For `buffer_size <= 4096`: using a pre-existing slab cache for that size class (e.g., `kmalloc-4096`)
- For `buffer_size > 4096`: using a power-of-2 rounded-up slab cache or falling back to page allocator
- Always zeroing the memory before returning (the `z` in `kzalloc`)
Memory is physically contiguous within a slab page but is not necessarily pinned (the kernel may swap it in pathological cases, handled transparently by `copy_to_user`'s exception table).
---
## 10. Concurrency Specification
**M2 explicitly does not add mutex protection.** This section documents the known races and their consequences so the implementer understands why M4 is necessary.
### Known Race: Concurrent Writers
```
CPU 0 (writer A):                    CPU 1 (writer B):
to_copy_A = min(5, buffer_size)      to_copy_B = min(5, buffer_size)
memset(kernel_buffer, 0, 4096)       memset(kernel_buffer, 0, 4096) ← zeroes A's partial work
copy_from_user(kernel_buffer, "hello", 5)
                                     copy_from_user(kernel_buffer, "world", 5)
buffer_data_len = 5                  buffer_data_len = 5
```
Result: `kernel_buffer` contains "world" (B wins), `buffer_data_len = 5`. Consistent but A's data is lost. In a pathological interleaving, `buffer_data_len` could be set to 5 by A while B has already written partial data — a reader sees 5 bytes of garbage. This is the motivation for M4's mutex.
### Known Race: Writer + Reader
```
CPU 0 (reader):                      CPU 1 (writer):
reads buffer_data_len = 0            — not yet written —
→ returns EOF (0)                    copies_from_user → buffer_data_len = 5
```
Result: reader sees empty buffer even though data was written concurrently. The reader returns EOF prematurely. Fixed in M4 by the wait queue mechanism.
### Safe: `open_count` with `atomic_t`
`atomic_t` operations (`atomic_inc`, `atomic_dec`, `atomic_read`) compile to `LOCK XADD`, `LOCK XADD`, and plain `MOV` respectively on x86_64. They are inherently safe for concurrent modification without a mutex. No additional synchronization is needed for `open_count`.
### Safe: `try_module_get` / `module_put`
These functions use an internal spinlock to manipulate the module's `refcnt` field. Safe for concurrent calls. The `rmmod`-while-open scenario is correctly handled: `rmmod` checks `refcnt == 0` before calling `mydevice_exit`; `try_module_get` in `open()` increments refcnt; `module_put` in `release()` decrements it.

![kmalloc Slab Allocator: GFP_KERNEL vs GFP_ATOMIC Decision Tree](./diagrams/tdd-diag-12.svg)

---
## 11. Common Pitfalls Reference
**Pitfall 1: Checking `class_create` result with `== NULL`.**
`class_create` returns `ERR_PTR(errno)` on failure, not NULL. `if (!mydevice_class)` will miss the error and dereference a poisoned pointer. Always: `if (IS_ERR(mydevice_class))`.
**Pitfall 2: `copy_from_user` / `copy_to_user` return value inversion.**
Both return bytes **not** copied (residual). `if (copy_from_user(...) != 0)` means failure. Many developers mistakenly check `if (copy_from_user(...) == 0)` and think they're handling the failure case. The naming is counterintuitive but correct: 0 residual = full success.
**Pitfall 3: Missing EOF return.**
`cat` loops forever. The check `if (*f_pos >= (loff_t)buffer_data_len) return 0;` must appear before any data copy. If it's missing or uses `>` instead of `>=`, reads past the data end return stale bytes forever.
**Pitfall 4: Not resetting `*f_pos = 0` in `mydevice_write`.**
Without this, a second write on the same fd will have `*f_pos` pointing into the middle of the new data on the next read. The read will start partway through the buffer, miss the beginning, and EOF correctly but return incomplete data.
**Pitfall 5: Wrong `module_param` type token for `size_t`.**
`module_param(buffer_size, size_t, 0444)` does not compile — `size_t` is not a recognized type token. Use `ulong` for a `size_t` parameter. Using `uint` on a 64-bit system truncates values > 2^32.
**Pitfall 6: Calling `cdev_del` before `device_destroy` and `class_destroy` in `mydevice_exit`.**
If you remove the cdev first, a concurrent `open()` via the still-existing `/dev/mydevice` node will find no cdev registered and return `-ENXIO`. The correct order ensures the udev trigger fires (`device_destroy`) before the routing is removed (`cdev_del`). During the brief window between `device_destroy` and `cdev_del`, new opens on the now-gone `/dev/` node are prevented by udev removing the file.
**Pitfall 7: Forgetting `my_cdev.owner = THIS_MODULE` after `cdev_init`.**
`cdev_init` does not set `owner`. Without it, the module's reference count is not correctly tied to the cdev's lifetime, and `rmmod` may succeed while the device is in use on some kernel versions.
**Pitfall 8: Using `ssize_t` cast incorrectly.**
`return to_copy` where `to_copy` is `size_t` compiles without warning on 64-bit but would silently truncate on 32-bit if `to_copy > INT_MAX`. The explicit cast `return (ssize_t)to_copy` documents intent. Since `to_copy <= buffer_size <= 1048576 < INT_MAX`, the cast is always safe.
**Pitfall 9: `memset` zeroing the entire buffer before `copy_from_user` is essential.**
Without `memset`, a long write followed by a short write leaves stale bytes from the long write visible after the short write. Example: write "ABCDEFGH" (8 bytes), then write "XY" (2 bytes). Without memset, `cat` returns "XYCDEFGH". With memset, `cat` returns "XY" — the buffer is cleanly overwritten.
**Pitfall 10: Direct dereference of `const char __user *buf` in `mydevice_write`.**
Even a single `buf[0]` dereference will trigger a SMAP fault on modern hardware. Any code review tool running `sparse` will emit `warning: dereference of noderef expression`. Treat `__user` pointers as opaque tokens that must only be passed to `copy_from_user`/`copy_to_user`.
---
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: build-kernel-module-m3 -->
# Technical Design Specification: ioctl and /proc Interface (`build-kernel-module-m3`)
---
## 1. Module Charter
This module extends the M2 character device driver with two major additions: an `unlocked_ioctl` handler exposing three commands (`MYDEVICE_IOC_RESIZE`, `MYDEVICE_IOC_CLEAR`, `MYDEVICE_IOC_STATUS`) and a `/proc/mydevice` entry backed by seq_file that reports five runtime statistics. The ioctl interface follows the kernel's structured 32-bit command encoding (`_IOW`/`_IOR`/`_IO` macros with magic byte `'k'`), with a shared `mydevice.h` header that compiles correctly in both kernel and userspace contexts. The `/proc` entry uses `proc_create` with `proc_ops` and `single_open`/`seq_printf` — not the deprecated raw `proc_read` API.
This module does **NOT** add mutex protection, wait queues, blocking I/O, `O_NONBLOCK` support, or `.poll` implementation. The buffer state variables (`kernel_buffer`, `buffer_data_len`, `buffer_size`) remain unprotected against concurrent access from multiple processes; that is explicitly deferred to M4. The `RESIZE` command allocates the new buffer before freeing the old (allocate-copy-swap-free), which is safe against allocation failure but not against concurrent reads or writes racing during the swap.
**Upstream dependency:** `build-kernel-module-m2` — the full `file_operations` dispatch table, `alloc_chrdev_region`, `cdev_init`/`cdev_add`, `class_create`/`device_create`, and `kzalloc` buffer infrastructure are all inherited unchanged. The goto error-unwind chain from M2 is extended with one new label.
**Downstream dependency:** `build-kernel-module-m4` wraps every ioctl handler body and every `read`/`write` handler body in `mutex_lock_interruptible`/`mutex_unlock`. The function signatures, buffer variables, and `atomic_t` counter names introduced here are referenced unchanged by M4.
**Invariants that must hold at all times:**
- After any ioctl call returns, `kernel_buffer != NULL`, `buffer_data_len <= buffer_size`, and the `kernel_buffer` pointer is valid (not freed).
- `read_count` and `write_count` are non-negative and monotonically non-decreasing.
- Any ioctl command with `_IOC_TYPE(cmd) != MYDEVICE_IOC_MAGIC` returns `-ENOTTY` without modifying any state.
- `/proc/mydevice` exists for the entire duration the module is loaded and is removed before any other cleanup in `mydevice_exit`.
- `MYDEVICE_IOC_RESIZE` with `new_size == 0` or `new_size > 1048576` returns `-EINVAL` and leaves `kernel_buffer`, `buffer_size`, and `buffer_data_len` unmodified.
- If `kzalloc` fails during `RESIZE`, `-ENOMEM` is returned and the old buffer is preserved intact.
---
## 2. File Structure
Create files in this exact order:
```
mydevice/                         ← existing from M2 (1)
├── Makefile                      ← (2) update: add ccflags-y := -I$(PWD) if not present
├── mydevice.h                    ← (3) NEW: shared kernel/userspace ioctl header
├── mydevice.c                    ← (4) update: add ioctl + /proc to M2 source
└── test_mydevice.c               ← (5) NEW: userspace test program
```
Build artifacts produced by `make` (unchanged from M2):
```
mydevice/
├── mydevice.ko
├── mydevice.o
├── mydevice.mod.c
├── mydevice.mod.o
├── Module.symvers
└── modules.order
```
Userspace test binary (built with `gcc`, not Kbuild):
```
mydevice/
└── test_mydevice                 ← (6) compiled from test_mydevice.c
```
Verification script:
```
mydevice/
└── verify.sh                     ← (7) acceptance test script
```
---
## 3. Complete Data Model
### 3.1 ioctl Command Number Encoding
Every ioctl command is a 32-bit integer with four packed fields. The kernel provides macros to construct and decompose these values:
```
Bits 31–30 (2):  direction  — _IOC_NONE(0), _IOC_WRITE(1), _IOC_READ(2), _IOC_RW(3)
Bits 29–16 (14): size       — sizeof(argument type), 0 for _IO commands
Bits 15–8  (8):  type       — magic byte identifying the driver ('k' = 0x6B)
Bits  7–0  (8):  number     — sequential command index within this driver (0, 1, 2, ...)
```
**Direction naming — the inversion that confuses everyone:**
| Macro | `_IOC_DIR` bits | Data movement | User's perspective |
|---|---|---|---|
| `_IO(type, nr)` | `_IOC_NONE` (0) | None | No argument |
| `_IOW(type, nr, t)` | `_IOC_WRITE` (1) | User → Kernel | User **w**rites to kernel |
| `_IOR(type, nr, t)` | `_IOC_READ` (2) | Kernel → User | User **r**eads from kernel |
| `_IOWR(type, nr, t)` | `_IOC_RW` (3) | Bidirectional | User reads and writes |
From the **kernel's** perspective, `_IOC_WRITE` means the kernel reads from userspace; `_IOC_READ` means the kernel writes to userspace. This is the reverse of the user-visible names. The `access_ok()` validation in the ioctl handler must use the kernel's perspective:
```c
/* When _IOC_DIR(cmd) & _IOC_READ: kernel writes to user → validate user ptr is writable */
/* When _IOC_DIR(cmd) & _IOC_WRITE: kernel reads from user → validate user ptr is readable */
```
On Linux ≥ 5.0, `access_ok()` takes only two arguments (pointer, size); the direction argument was removed. The two-argument form is correct for all supported kernels:
```c
if (!access_ok((void __user *)arg, _IOC_SIZE(cmd)))
    return -EFAULT;
```
### 3.2 `struct mydevice_status` — Kernel/Userspace Contract
```c
struct mydevice_status {
    unsigned long buffer_size;   /* offset 0:  total buffer capacity in bytes */
    unsigned long bytes_used;    /* offset 8:  bytes currently stored */
    unsigned int  open_count;    /* offset 16: currently open file descriptors */
    unsigned long read_count;    /* offset 24: total read() calls since load */
    unsigned long write_count;   /* offset 32: total write() calls since load */
};
/* Total: 40 bytes on 64-bit, 20 bytes on 32-bit (unsigned long = 4 bytes) */
```
**Memory layout on 64-bit x86_64 (the target platform):**
| Field | Type | Size (bytes) | Offset | Notes |
|---|---|---|---|---|
| `buffer_size` | `unsigned long` | 8 | 0x00 | Matches `size_t buffer_size` in driver |
| `bytes_used` | `unsigned long` | 8 | 0x08 | Matches `size_t buffer_data_len` in driver |
| `open_count` | `unsigned int` | 4 | 0x10 | `atomic_read(&open_count)` cast to `unsigned int` |
| `(padding)` | — | 4 | 0x14 | Compiler-inserted padding before next `unsigned long` |
| `read_count` | `unsigned long` | 8 | 0x18 | `atomic_read(&read_count)` |
| `write_count` | `unsigned long` | 8 | 0x20 | `atomic_read(&write_count)` |
| **Total** | | **40** | | |
**Why `unsigned long` and not `uint64_t`:** This is a learning driver for single-machine use where the kernel and test program share the same ABI. `unsigned long` is `size_t`-compatible on 64-bit Linux. Production drivers would use `__u64`/`__u32` from `<linux/types.h>` (kernel) / `<stdint.h>` (userspace) to guarantee layout stability across 32/64-bit combinations. The padding between `open_count` and `read_count` means `sizeof(struct mydevice_status) == 40` on 64-bit — both the kernel module and the test program must agree on this value. They will, because both include the same `mydevice.h`.
### 3.3 ioctl Command Numbers
```c
#define MYDEVICE_IOC_MAGIC  'k'          /* 0x6B — magic byte for this driver */
#define MYDEVICE_IOC_RESIZE  _IOW(MYDEVICE_IOC_MAGIC, 0, unsigned long)
#define MYDEVICE_IOC_CLEAR   _IO (MYDEVICE_IOC_MAGIC, 1)
#define MYDEVICE_IOC_STATUS  _IOR(MYDEVICE_IOC_MAGIC, 2, struct mydevice_status)
#define MYDEVICE_IOC_MAXNR   2           /* highest valid command number */
```
**Encoded values (64-bit kernel, `sizeof(unsigned long) == 8`, `sizeof(struct mydevice_status) == 40`):**
| Constant | Encoding | Hex value |
|---|---|---|
| `MYDEVICE_IOC_RESIZE` | `_IOW('k', 0, unsigned long)` = `(1<<30)\|(8<<16)\|('k'<<8)\|0` | `0x406B6B00` → actually `0x40086B00` |
| `MYDEVICE_IOC_CLEAR` | `_IO('k', 1)` = `(0<<30)\|(0<<16)\|('k'<<8)\|1` | `0x00006B01` |
| `MYDEVICE_IOC_STATUS` | `_IOR('k', 2, struct mydevice_status)` = `(2<<30)\|(40<<16)\|('k'<<8)\|2` | `0x80286B02` |
The actual hex values are computed by the preprocessor at compile time using `sizeof`. Do not hardcode them — always use the macro names.
### 3.4 New `atomic_t` Counters
Two new module-level atomics track I/O call counts for the STATUS query and /proc display:
```c
static atomic_t read_count  = ATOMIC_INIT(0);   /* incremented in mydevice_read */
static atomic_t write_count = ATOMIC_INIT(0);   /* incremented in mydevice_write */
```
These are in addition to the `open_count` atomic from M2. All three use `atomic_t` for safety: multiple CPUs can be executing `mydevice_read` or `mydevice_write` concurrently (one per open fd), and concurrent `atomic_inc` calls are safe whereas concurrent `int++` is not.
### 3.5 `/proc` Entry Handle
```c
static struct proc_dir_entry *proc_entry;   /* returned by proc_create; NULL until init */
```
`proc_entry` is used only in `mydevice_exit` to call `proc_remove`. It must be non-NULL when `proc_remove` is called. The goto unwind chain in `mydevice_init` ensures `proc_remove` is only called if `proc_create` succeeded.
### 3.6 Updated Global State Summary
Inheriting all variables from M2, the complete module state after M3:
| Variable | Type | Initial Value | Protected By | Modified By |
|---|---|---|---|---|
| `dev_num` | `dev_t` | 0 | — (write-once in init) | `mydevice_init` |
| `my_cdev` | `struct cdev` | zeroed | — | `mydevice_init` |
| `mydevice_class` | `struct class *` | NULL | — | `mydevice_init` |
| `mydevice_device` | `struct device *` | NULL | — | `mydevice_init` |
| `proc_entry` | `struct proc_dir_entry *` | NULL | — | `mydevice_init` |
| `kernel_buffer` | `char *` | NULL | **None (M3)** | init, write, ioctl RESIZE |
| `buffer_data_len` | `size_t` | 0 | **None (M3)** | write, ioctl CLEAR/RESIZE |
| `buffer_size` | `size_t` | 4096 | **None (M3)** | init, ioctl RESIZE |
| `open_count` | `atomic_t` | 0 | atomic ops | open, release |
| `read_count` | `atomic_t` | 0 | atomic ops | mydevice_read |
| `write_count` | `atomic_t` | 0 | atomic ops | mydevice_write |

![ioctl Command Number 32-bit Encoding: _IOW/_IOR/_IO Bit Layout](./diagrams/tdd-diag-13.svg)

---
## 4. Interface Contracts
### 4.1 `mydevice.h` — Shared Header
```c
/* mydevice.h — shared between kernel module and userspace programs
 * Include from kernel:    #include "mydevice.h" (via ccflags-y := -I$(PWD))
 * Include from userspace: #include "mydevice.h" (after #include <sys/ioctl.h>)
 */
#ifndef MYDEVICE_H
#define MYDEVICE_H
/*
 * linux/ioctl.h provides _IO/_IOW/_IOR/_IOWR in kernel context.
 * In userspace, sys/ioctl.h pulls in linux/ioctl.h via the system include path.
 * The include below works in both contexts.
 */
#include <linux/ioctl.h>
#define MYDEVICE_IOC_MAGIC  'k'
struct mydevice_status {
    unsigned long buffer_size;
    unsigned long bytes_used;
    unsigned int  open_count;
    unsigned long read_count;
    unsigned long write_count;
};
#define MYDEVICE_IOC_RESIZE  _IOW(MYDEVICE_IOC_MAGIC, 0, unsigned long)
#define MYDEVICE_IOC_CLEAR   _IO (MYDEVICE_IOC_MAGIC, 1)
#define MYDEVICE_IOC_STATUS  _IOR(MYDEVICE_IOC_MAGIC, 2, struct mydevice_status)
#define MYDEVICE_IOC_MAXNR   2
#endif /* MYDEVICE_H */
```
**Header rules:**
- No `__user` annotation on struct fields — `__user` is a kernel-only annotation. The struct is defined in terms of plain C types so userspace compilers accept it without modification.
- No kernel-only includes (`linux/fs.h`, `linux/slab.h`, etc.) inside this header. Only `linux/ioctl.h` is included, which is safe in userspace via `/usr/include/linux/ioctl.h`.
- Userspace test program must include `<sys/ioctl.h>` **before** `"mydevice.h"` or must ensure `linux/ioctl.h` is reachable through the system include path. On Debian/Ubuntu, `linux-libc-dev` provides this.
### 4.2 `mydevice_ioctl(filp, cmd, arg) → long`
```c
static long mydevice_ioctl(struct file *filp, unsigned int cmd, unsigned long arg);
```
**Registered as:** `.unlocked_ioctl = mydevice_ioctl` in `mydevice_fops`.
**Context:** Process context. No locks held on entry. Can sleep (calls `kzalloc(GFP_KERNEL)` in the RESIZE path). `arg` is either a direct integer value or a userspace pointer — the interpretation depends on the command.
**Parameters:**
- `filp`: Open file descriptor. Not examined in M3 (no `O_NONBLOCK` check yet).
- `cmd`: 32-bit encoded ioctl command number. May be any value — validation is the first task.
- `arg`: For `_IO` commands: undefined/0, do not dereference. For `_IOW`/`_IOR`/`_IOWR` commands: userspace pointer. Cast to `(void __user *)` before passing to `access_ok` or `copy_from/to_user`.
**Validation sequence (must occur in this order):**
```
1. if (_IOC_TYPE(cmd) != MYDEVICE_IOC_MAGIC) → return -ENOTTY
2. if (_IOC_NR(cmd) > MYDEVICE_IOC_MAXNR)   → return -ENOTTY
3. if (_IOC_DIR(cmd) & (_IOC_READ | _IOC_WRITE)):
       if (!access_ok((void __user *)arg, _IOC_SIZE(cmd))) → return -EFAULT
4. switch(cmd) → dispatch to helper
5. default: → return -ENOTTY  (unreachable after step 2, but defensive)
```
**Return values:**
| Return | Condition |
|---|---|
| `0` | Command executed successfully |
| `-ENOTTY` | Wrong magic number, out-of-range command number, or unrecognized command |
| `-EFAULT` | `access_ok` failed (kernel-space address or NULL for a pointer argument) |
| `-EFAULT` | `copy_from_user` or `copy_to_user` returned non-zero |
| `-EINVAL` | `RESIZE` with `new_size == 0` or `new_size > 1048576` |
| `-ENOMEM` | `RESIZE` `kzalloc` failed |
**No state modification occurs before validation completes.** Steps 1–3 are read-only.
### 4.3 `mydevice_ioctl_resize(arg) → int`
```c
static int mydevice_ioctl_resize(unsigned long arg);
```
**Called from:** `mydevice_ioctl` case `MYDEVICE_IOC_RESIZE`.
**Parameters:** `arg` is a userspace pointer to `unsigned long new_size`. Not yet validated beyond `access_ok` in the caller — this function performs `copy_from_user`.
**Algorithm (detailed in Section 5.1).**
**Return values:**
| Return | Condition |
|---|---|
| `0` | Buffer resized; `kernel_buffer`, `buffer_size`, `buffer_data_len` updated |
| `-EFAULT` | `copy_from_user` failed (bad pointer) |
| `-EINVAL` | `new_size == 0` or `new_size > 1048576` |
| `-ENOMEM` | `kzalloc(new_size, GFP_KERNEL)` returned NULL; old buffer preserved |
**Post-condition on success:** `buffer_size == new_size`, `kernel_buffer` points to a freshly-allocated `new_size`-byte zeroed buffer, `buffer_data_len <= new_size` (truncated if new_size < old buffer_data_len, preserved otherwise).
### 4.4 `mydevice_ioctl_clear(void) → int`
```c
static int mydevice_ioctl_clear(void);
```
**Called from:** `mydevice_ioctl` case `MYDEVICE_IOC_CLEAR`. No argument — the ioctl `arg` value is ignored; this function takes no parameters.
**Algorithm:**
1. `memset(kernel_buffer, 0, buffer_size)` — zeroes all bytes.
2. `buffer_data_len = 0` — marks buffer as empty.
3. `pr_info("mydevice: buffer cleared\n")`.
4. Return `0`.
**Return values:** Always `0`. No failure path exists: `kernel_buffer` is guaranteed non-NULL after successful `mydevice_init`, and `memset` on a valid kernel pointer cannot fail.
**Post-condition:** `buffer_data_len == 0`. A subsequent `read()` returns EOF immediately (0 bytes). A subsequent `write()` starts filling from the beginning.
### 4.5 `mydevice_ioctl_status(user_status) → int`
```c
static int mydevice_ioctl_status(struct mydevice_status __user *user_status);
```
**Called from:** `mydevice_ioctl` case `MYDEVICE_IOC_STATUS`.
**Parameters:** `user_status` is a userspace pointer to a `struct mydevice_status` that the kernel will fill. Validated by `access_ok` in the caller.
**Algorithm:**
1. Declare `struct mydevice_status status` on the kernel stack.
2. Populate fields:
   - `status.buffer_size  = buffer_size;`
   - `status.bytes_used   = buffer_data_len;`
   - `status.open_count   = (unsigned int)atomic_read(&open_count);`
   - `status.read_count   = (unsigned long)atomic_read(&read_count);`
   - `status.write_count  = (unsigned long)atomic_read(&write_count);`
3. `if (copy_to_user(user_status, &status, sizeof(status))) return -EFAULT;`
4. Return `0`.
**Why assemble on kernel stack first:** Never write fields one-by-one to a userspace pointer. A partial `copy_to_user` failure midway through individual field copies leaves the userspace struct in a torn state. A single `copy_to_user` of the complete struct is atomic from the userspace program's perspective (either the whole copy succeeds or it fails — the kernel does not partially update).
**Return values:**
| Return | Condition |
|---|---|
| `0` | All fields copied to userspace |
| `-EFAULT` | `copy_to_user` returned non-zero |
### 4.6 `mydevice_proc_show(m, v) → int`
```c
static int mydevice_proc_show(struct seq_file *m, void *v);
```
**Called from:** `single_open` machinery when userspace reads `/proc/mydevice`. The `v` parameter is always the private data passed to `single_open` (NULL in this implementation).
**Output format (exact — tests rely on field names):**
```
buffer_size:  <N>\n
bytes_used:   <N>\n
open_count:   <N>\n
read_count:   <N>\n
write_count:  <N>\n
```
**Algorithm:**
1. `seq_printf(m, "buffer_size:  %zu\n",  buffer_size);`
2. `seq_printf(m, "bytes_used:   %zu\n",  buffer_data_len);`
3. `seq_printf(m, "open_count:   %d\n",   atomic_read(&open_count));`
4. `seq_printf(m, "read_count:   %d\n",   atomic_read(&read_count));`
5. `seq_printf(m, "write_count:  %d\n",   atomic_read(&write_count));`
6. Return `0`.
**`seq_printf` return value:** Returns the number of bytes written, or a negative value if the internal buffer is full. Do NOT check the return value of individual `seq_printf` calls. The seq_file infrastructure handles buffer management across partial reads — if the buffer fills during a `seq_printf`, the data is buffered in a new page allocation and delivered on the next `read()` call. This is the entire reason seq_file exists.
### 4.7 `mydevice_proc_open(inode, filp) → int`
```c
static int mydevice_proc_open(struct inode *inode, struct file *filp);
```
**Algorithm:**
1. Return `single_open(filp, mydevice_proc_show, NULL)`.
`single_open` allocates a `struct seq_file`, attaches the `show` callback, and stores it in `filp->private_data`. The `NULL` third argument is the private data pointer passed as `v` to `mydevice_proc_show`.
**Return values:** `0` on success, `-ENOMEM` if `single_open`'s internal allocation fails. These propagate directly from `single_open`.

![ioctl Dispatch Flow: Userspace ioctl() → Validation → Switch → Handler](./diagrams/tdd-diag-14.svg)

---
## 5. Algorithm Specification
### 5.1 `MYDEVICE_IOC_RESIZE` — Allocate-Copy-Swap-Free
This is the most complex algorithm in this milestone. The invariant to preserve: `kernel_buffer` is always non-NULL after `mydevice_init` returns 0. A failed resize must leave the buffer intact.
```
mydevice_ioctl_resize(arg):
    STEP 1: READ NEW SIZE FROM USERSPACE
        unsigned long new_size
        not_copied = copy_from_user(&new_size, (unsigned long __user *)arg,
                                     sizeof(new_size))
        if (not_copied != 0):
            return -EFAULT
        // new_size is now a kernel-local copy; arg is no longer accessed
    STEP 2: VALIDATE NEW SIZE
        if (new_size == 0):
            pr_err("mydevice: resize: new_size cannot be zero\n")
            return -EINVAL
            // Invariant preserved: kernel_buffer unchanged
        if (new_size > 1048576):
            pr_err("mydevice: resize: new_size %lu exceeds 1MB limit\n", new_size)
            return -EINVAL
            // Invariant preserved: kernel_buffer unchanged
    STEP 3: ALLOCATE NEW BUFFER (before freeing old)
        char *new_buf = kzalloc(new_size, GFP_KERNEL)
        if (!new_buf):
            pr_err("mydevice: resize: kzalloc(%lu) failed\n", new_size)
            return -ENOMEM
            // Invariant preserved: kernel_buffer still points to old valid buffer
    STEP 4: COPY EXISTING DATA (if any)
        if (buffer_data_len > 0):
            size_t copy_len = (buffer_data_len < new_size) ?
                               buffer_data_len : new_size
            memcpy(new_buf, kernel_buffer, copy_len)
            buffer_data_len = copy_len
            // If new_size < old buffer_data_len: data is truncated
            // If new_size >= old buffer_data_len: all data preserved
        else:
            buffer_data_len = 0
            // nothing to copy
    STEP 5: SWAP AND FREE
        kfree(kernel_buffer)   // free old buffer
        kernel_buffer = new_buf // install new buffer
        buffer_size   = new_size
        // Invariant restored: kernel_buffer non-NULL, buffer_size == new_size,
        //                     buffer_data_len <= buffer_size
    STEP 6: LOG AND RETURN
        pr_info("mydevice: buffer resized to %lu bytes\n", new_size)
        return 0
```
**Critical ordering:** `kzalloc` (step 3) must complete successfully BEFORE `kfree` (step 5). This is the allocate-first safety pattern. Rationale: if `kfree` ran first and `kzalloc` then failed, `kernel_buffer` would be a freed (dangling) pointer — any subsequent access crashes the kernel. By allocating first, a `kzalloc` failure leaves the original buffer fully intact.
**Truncation semantics:** When `new_size < buffer_data_len`, the data beyond `new_size` bytes is silently dropped. The caller is responsible for draining the buffer before shrinking it if data preservation is required. This matches `ftruncate(2)` semantics on regular files.

![Shared Header: One mydevice.h Compiled in Two Worlds](./diagrams/tdd-diag-15.svg)

### 5.2 ioctl Dispatch Validation Sequence
The three-layer validation in `mydevice_ioctl` must occur in this exact order:
```
Layer 1: MAGIC NUMBER CHECK
    _IOC_TYPE(cmd) extracts bits 15-8
    If != 'k' (0x6B): return -ENOTTY immediately
    Purpose: rejects commands from other drivers routed here by mistake
             (should never happen with correct fd, but defensive programming)
Layer 2: COMMAND NUMBER RANGE CHECK
    _IOC_NR(cmd) extracts bits 7-0
    If > MYDEVICE_IOC_MAXNR (2): return -ENOTTY immediately
    Purpose: catches future expansion bugs; a command number of 99
             with the right magic would otherwise reach the switch's default
Layer 3: USERSPACE POINTER VALIDATION
    Only for commands with a non-NONE direction:
    if (_IOC_DIR(cmd) & _IOC_READ):
        // Kernel will write to userspace: validate the ptr is in user address space
        if (!access_ok((void __user *)arg, _IOC_SIZE(cmd))): return -EFAULT
    if (_IOC_DIR(cmd) & _IOC_WRITE):
        // Kernel will read from userspace: same validation
        if (!access_ok((void __user *)arg, _IOC_SIZE(cmd))): return -EFAULT
    Note: _IOC_READ and _IOC_WRITE are NOT mutually exclusive.
    For _IOWR commands, both flags are set and both checks run.
    For _IO commands (_IOC_DIR == 0), neither block executes. arg is NOT accessed.
Layer 4: SWITCH DISPATCH
    switch (cmd):
        case MYDEVICE_IOC_RESIZE: return mydevice_ioctl_resize(arg)
        case MYDEVICE_IOC_CLEAR:  return mydevice_ioctl_clear()
        case MYDEVICE_IOC_STATUS: return mydevice_ioctl_status(
                                      (struct mydevice_status __user *)arg)
        default: return -ENOTTY   ← unreachable, but required
```
**Why `access_ok` is not sufficient and `copy_from/to_user` is still needed:** `access_ok` validates that the address *range* falls within the userspace address space boundary. It does NOT validate that the pages are currently mapped or present. A page in the valid userspace range could be swapped out, `mmap`'d to a file, or unmapped entirely. `copy_from_user`/`copy_to_user` handle page faults during the copy (via the exception table mechanism) and return a non-zero residual if the copy fails. Both checks are required.
### 5.3 seq_file Initialization and Lifecycle
```
proc_create("mydevice", 0444, NULL, &mydevice_proc_ops):
    ├── Creates /proc/mydevice inode with permission 0444
    ├── NULL parent = root of /proc (i.e., /proc/mydevice, not /proc/driver/mydevice)
    ├── Stores &mydevice_proc_ops as the file_operations for this /proc file
    └── Returns struct proc_dir_entry* or NULL on failure
When userspace reads /proc/mydevice:
    open("/proc/mydevice", O_RDONLY)
        → VFS calls mydevice_proc_ops.proc_open = mydevice_proc_open
        → single_open(filp, mydevice_proc_show, NULL)
            → allocates struct seq_file, stores in filp->private_data
            → stores mydevice_proc_show as the iterator
        → returns fd to userspace
    read(fd, buf, N)
        → VFS calls mydevice_proc_ops.proc_read = seq_read
        → seq_read checks if internal buffer is populated
        → if not: calls mydevice_proc_show(seq_file, NULL)
            → seq_printf writes "buffer_size: ...\n" etc. into internal pages
        → copies up to N bytes from internal buffer to userspace
        → if more data remains: updates offset for next read
        → if all data delivered: sets EOF flag
    read(fd, buf, N)  [second call, after full first read]
        → seq_read sees EOF flag set
        → returns 0 to userspace → cat terminates
    close(fd)
        → VFS calls mydevice_proc_ops.proc_release = single_release
        → single_release frees the struct seq_file and internal buffers
```
**Why seq_read / seq_lseek / single_release are not implemented by you:** These are provided by the seq_file infrastructure (`<linux/seq_file.h>`). You implement only `mydevice_proc_show` (the data generator) and `mydevice_proc_open` (which calls `single_open`). The infrastructure handles all fragmentation, partial-read offset management, and cleanup.
### 5.4 `mydevice_init` Extended Goto Chain
M3 adds one new initialization step and one new goto label to the M2 chain:
```
STEP 1: Validate buffer_size              → (no label, early return on fail)
STEP 2: kzalloc kernel_buffer             → err_free_buf
STEP 3: alloc_chrdev_region               → err_unreg_region
STEP 4: cdev_init + cdev_add              → err_del_cdev
STEP 5: class_create                      → err_destroy_class
STEP 6: device_create                     → err_destroy_device
STEP 7: proc_create  [NEW]                → err_remove_proc [NEW = err_destroy_device]
Success path: pr_info + return 0
Unwind chain (reverse order):
err_destroy_device:   device_destroy(mydevice_class, dev_num)
err_destroy_class:    class_destroy(mydevice_class)
err_del_cdev:         cdev_del(&my_cdev)
err_unreg_region:     unregister_chrdev_region(dev_num, 1)
err_free_buf:         kfree(kernel_buffer)
                      return ret
```
Note: `proc_create` failure goto is `err_destroy_device` — it triggers `device_destroy`, falls through to `class_destroy`, falls through to `cdev_del`, etc. A separate `err_remove_proc` label is not needed because `proc_remove` is only called if `proc_create` succeeded, which means we never reach the unwind from `proc_create`'s own failure. The goto from step 7's failure goes to `err_destroy_device` to undo steps 1–6.
`mydevice_exit` cleanup order (MUST be this exact order):
```
1. proc_remove(proc_entry)              ← first: stop new /proc reads
2. device_destroy(mydevice_class, dev_num)
3. class_destroy(mydevice_class)
4. cdev_del(&my_cdev)
5. unregister_chrdev_region(dev_num, 1)
6. kfree(kernel_buffer)
7. pr_info("mydevice: module unloaded\n")
```
**Why `proc_remove` is first:** If a process is mid-read of `/proc/mydevice` when `mydevice_exit` starts, `proc_remove` waits for the reader to finish (the proc infrastructure holds a reference). Only after `proc_remove` returns is it safe to free `kernel_buffer` and unregister the device. Removing the proc entry before the device ensures no new `/proc` readers can start accessing state that's about to be freed.
{{DIAGRAM:tdd-diag-16}}
---
## 6. Error Handling Matrix
| Error | Detected By | Module State After | User-Visible | Recovery |
|---|---|---|---|---|
| `_IOC_TYPE(cmd) != 'k'` | `mydevice_ioctl` layer 1 | Unchanged | `ioctl()` returns -1, `errno = ENOTTY` | Pass correct ioctl command number (from `mydevice.h`) |
| `_IOC_NR(cmd) > 2` | `mydevice_ioctl` layer 2 | Unchanged | `ioctl()` returns -1, `errno = ENOTTY` | Pass valid command (RESIZE=0, CLEAR=1, STATUS=2) |
| `access_ok` fails on `arg` | `mydevice_ioctl` layer 3 | Unchanged | `ioctl()` returns -1, `errno = EFAULT` | Pass valid userspace pointer; NULL and kernel addresses fail |
| `copy_from_user` fails in RESIZE | `mydevice_ioctl_resize` | Unchanged (new_size not read) | `errno = EFAULT` | Pass valid userspace `unsigned long *` |
| `new_size == 0` | `mydevice_ioctl_resize` step 2 | Unchanged | `errno = EINVAL` | Pass `new_size >= 1` |
| `new_size > 1048576` | `mydevice_ioctl_resize` step 2 | Unchanged | `errno = EINVAL` | Pass `new_size <= 1048576` |
| `kzalloc` fails in RESIZE | `!new_buf` check after kzalloc | Unchanged (old buffer preserved) | `errno = ENOMEM` | Free memory; retry with smaller size |
| `copy_to_user` fails in STATUS | `copy_to_user` return check | Unchanged (status struct on stack, not partially written to user) | `errno = EFAULT` | Pass valid writable userspace `struct mydevice_status *` |
| `proc_create` returns NULL | `!proc_entry` check in `mydevice_init` | Device created (steps 1–6 succeeded); goto `err_destroy_device` unwinds all | `insmod` fails: "Cannot allocate memory" | Check if `/proc/mydevice` already exists (stale entry from crashed prev load); reboot or manually remove |
| `CLEAR` on empty buffer | No check needed | `buffer_data_len` stays 0, `memset` zeros already-zero buffer | Returns 0 (success) | Not an error |
| RESIZE to same size | No check; treated as normal resize | New buffer allocated, data copied, old freed | Returns 0 (minor inefficiency) | Not an error; simply reallocates |
| RESIZE smaller than `buffer_data_len` | Step 4 truncates: `copy_len = new_size` | `buffer_data_len = new_size`; excess data lost | Returns 0 | Drain buffer before shrinking if data preservation needed |
| ioctl called with `O_RDONLY` fd | Not checked (M3) | — | Succeeds; ioctl doesn't enforce read/write direction | Expected behavior; ioctl permission is separate from file open mode |
| `/proc/mydevice` read while RESIZE in progress | No mutex (M3); race condition | May read stale `buffer_size` or `buffer_data_len` during swap | Stale data in `/proc` output | Add mutex in M4; acceptable in M3 |
**Invariant after every error return from `mydevice_ioctl`:** `kernel_buffer != NULL`, `buffer_data_len <= buffer_size`, and the old buffer contents are unchanged from before the call.
---
## 7. Implementation Sequence with Checkpoints
### Phase 1: `mydevice.h` — Shared Header (0.5–1 hour)
Create `mydevice/mydevice.h` with the exact content from Section 4.1.
Verify `ccflags-y := -I$(PWD)` is present in `Makefile`. If not, add it:
```makefile
obj-m += mydevice.o
KDIR ?= /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)
ccflags-y := -I$(PWD)    # ensures mydevice.h is found by kernel build
all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules
clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
```
Add `#include "mydevice.h"` to `mydevice.c` (after other includes).
Add the two new atomics to `mydevice.c` globals:
```c
static atomic_t read_count  = ATOMIC_INIT(0);
static atomic_t write_count = ATOMIC_INIT(0);
```
Add `atomic_inc(&read_count)` to `mydevice_read` (before `return (ssize_t)to_copy`).  
Add `atomic_inc(&write_count)` to `mydevice_write` (before `return (ssize_t)to_copy`).
**Checkpoint 1:**
```bash
make EXTRA_CFLAGS="-Werror"
# Expected: zero warnings, zero errors
# Verify header is found:
grep -n "mydevice.h" mydevice.c   # must show the #include line
# Verify atomics are present:
grep -n "read_count\|write_count" mydevice.c | head -10
sudo insmod mydevice.ko
# Write and read to confirm atomics increment:
echo "test" > /dev/mydevice
cat /dev/mydevice
sudo rmmod mydevice
```
### Phase 2: `unlocked_ioctl` Skeleton — Validation + Dispatch (0.5–1 hour)
Add `#include <linux/uaccess.h>` (already present from M2) and ensure the ioctl-related macros compile. Add `.unlocked_ioctl = mydevice_ioctl` to `mydevice_fops`.
Implement `mydevice_ioctl` with the three-layer validation and a switch that calls stub helpers returning `-ENOTTY` temporarily:
```c
static long mydevice_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    if (_IOC_TYPE(cmd) != MYDEVICE_IOC_MAGIC)
        return -ENOTTY;
    if (_IOC_NR(cmd) > MYDEVICE_IOC_MAXNR)
        return -ENOTTY;
    if (_IOC_DIR(cmd) & _IOC_READ) {
        if (!access_ok((void __user *)arg, _IOC_SIZE(cmd)))
            return -EFAULT;
    }
    if (_IOC_DIR(cmd) & _IOC_WRITE) {
        if (!access_ok((void __user *)arg, _IOC_SIZE(cmd)))
            return -EFAULT;
    }
    switch (cmd) {
    case MYDEVICE_IOC_RESIZE:
        return -ENOTTY;   /* stub */
    case MYDEVICE_IOC_CLEAR:
        return -ENOTTY;   /* stub */
    case MYDEVICE_IOC_STATUS:
        return -ENOTTY;   /* stub */
    default:
        return -ENOTTY;
    }
}
```
**Checkpoint 2:**
```bash
make EXTRA_CFLAGS="-Werror"
sudo insmod mydevice.ko
sudo chmod 666 /dev/mydevice
# Test validation layer: wrong magic
python3 -c "
import fcntl, os, errno
fd = os.open('/dev/mydevice', os.O_RDWR)
try:
    fcntl.ioctl(fd, 0x00005A00)  # magic='Z', nr=0 — wrong magic
    print('FAIL: should have returned ENOTTY')
except OSError as e:
    print('PASS' if e.errno == errno.ENOTTY else f'FAIL: got {e.errno}')
os.close(fd)
"
# Test validation layer: right magic, out-of-range nr
python3 -c "
import fcntl, os, errno
fd = os.open('/dev/mydevice', os.O_RDWR)
try:
    fcntl.ioctl(fd, 0x00006B63)  # magic='k', nr=99
    print('FAIL: should have returned ENOTTY')
except OSError as e:
    print('PASS' if e.errno == errno.ENOTTY else f'FAIL: got {e.errno}')
os.close(fd)
"
sudo rmmod mydevice
```
### Phase 3: RESIZE and CLEAR Commands (1–1.5 hours)
Implement `mydevice_ioctl_resize` per the algorithm in Section 5.1. Implement `mydevice_ioctl_clear` per Section 4.4. Replace the stubs in the switch.
**Checkpoint 3:**
```bash
make EXTRA_CFLAGS="-Werror"
sudo insmod mydevice.ko
sudo chmod 666 /dev/mydevice
# Test CLEAR
echo "hello" > /dev/mydevice
python3 -c "
import fcntl, os, errno
from mydevice_consts import MYDEVICE_IOC_CLEAR
fd = os.open('/dev/mydevice', os.O_RDWR)
ret = fcntl.ioctl(fd, MYDEVICE_IOC_CLEAR)
print(f'CLEAR returned: {ret}')   # expect 0
# Read after clear: should be EOF immediately
data = os.read(fd, 100)
print(f'Read after clear: {len(data)} bytes (expected 0)')
os.close(fd)
"
# Test RESIZE to 8192 bytes
python3 -c "
import fcntl, os, struct, errno
fd = os.open('/dev/mydevice', os.O_RDWR)
new_size_buf = struct.pack('Q', 8192)  # unsigned long, 8 bytes, little-endian
import ctypes
new_size = ctypes.c_ulong(8192)
# _IOW('k', 0, unsigned long) = compute or hardcode for testing:
MYDEVICE_IOC_RESIZE = (1 << 30) | (8 << 16) | (ord('k') << 8) | 0
ret = fcntl.ioctl(fd, MYDEVICE_IOC_RESIZE, new_size)
print(f'RESIZE returned: {ret}')  # expect 0
os.close(fd)
"
dmesg | tail -5
# Must show: "buffer resized to 8192 bytes"
# Test RESIZE with invalid sizes:
python3 -c "
import fcntl, os, struct, errno
fd = os.open('/dev/mydevice', os.O_RDWR)
MYDEVICE_IOC_RESIZE = (1 << 30) | (8 << 16) | (ord('k') << 8) | 0
# Test size=0
try:
    fcntl.ioctl(fd, MYDEVICE_IOC_RESIZE, struct.pack('Q', 0))
    print('FAIL: size=0 should return EINVAL')
except OSError as e:
    print('PASS size=0' if e.errno == errno.EINVAL else f'FAIL: got {e.errno}')
# Test size > 1MB
try:
    fcntl.ioctl(fd, MYDEVICE_IOC_RESIZE, struct.pack('Q', 2000000))
    print('FAIL: size>1MB should return EINVAL')
except OSError as e:
    print('PASS size>1MB' if e.errno == errno.EINVAL else f'FAIL: got {e.errno}')
os.close(fd)
"
sudo rmmod mydevice
```
### Phase 4: STATUS Command + Counter Verification (0.5–1 hour)
Implement `mydevice_ioctl_status` per Section 4.5. Replace the STATUS stub. Verify that `read_count` and `write_count` (added in Phase 1) are correctly reported.
**Checkpoint 4:**
```bash
make EXTRA_CFLAGS="-Werror"
sudo insmod mydevice.ko
sudo chmod 666 /dev/mydevice
python3 -c "
import fcntl, os, struct
# Compute STATUS command: _IOR('k', 2, struct mydevice_status)
# struct mydevice_status = ulong, ulong, uint, [4 pad], ulong, ulong = 40 bytes
MYDEVICE_IOC_STATUS = (2 << 30) | (40 << 16) | (ord('k') << 8) | 2
fd = os.open('/dev/mydevice', os.O_RDWR)
# Write 5 bytes
os.write(fd, b'hello')
# Read them back
os.lseek(fd, 0, os.SEEK_SET)
os.read(fd, 5)
# Query status
status_buf = bytearray(40)
fcntl.ioctl(fd, MYDEVICE_IOC_STATUS, status_buf)
# Parse: ulong(8), ulong(8), uint(4), pad(4), ulong(8), ulong(8)
import struct as st
buffer_size, bytes_used, open_count = st.unpack_from('<QQI', status_buf, 0)
read_count, write_count = st.unpack_from('<QQ', status_buf, 24)
print(f'buffer_size={buffer_size} (expected 4096)')
print(f'bytes_used={bytes_used} (expected 5)')
print(f'open_count={open_count} (expected 1)')
print(f'read_count={read_count} (expected 1)')
print(f'write_count={write_count} (expected 1)')
assert buffer_size == 4096
assert bytes_used == 5
assert open_count == 1
assert read_count == 1
assert write_count == 1
print('STATUS test PASSED')
os.close(fd)
"
sudo rmmod mydevice
```
### Phase 5: `/proc/mydevice` via seq_file (1–1.5 hours)
Add includes to `mydevice.c`:
```c
#include <linux/proc_fs.h>    /* proc_create, proc_remove, proc_ops */
#include <linux/seq_file.h>   /* seq_file, seq_printf, single_open, single_release */
```
Add `static struct proc_dir_entry *proc_entry;` to globals (initialized to NULL implicitly).
Implement `mydevice_proc_show`, `mydevice_proc_open`, and `mydevice_proc_ops` per Sections 4.6, 4.7.
```c
static const struct proc_ops mydevice_proc_ops = {
    .proc_open    = mydevice_proc_open,
    .proc_read    = seq_read,
    .proc_lseek   = seq_lseek,
    .proc_release = single_release,
};
```
**`struct proc_ops` vs `struct file_operations` for `/proc`:** Kernel ≥ 5.6 requires `struct proc_ops` for proc files. Kernel < 5.6 uses `struct file_operations`. If cross-version compatibility is needed:
```c
#include <linux/version.h>
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 6, 0)
static const struct proc_ops mydevice_proc_ops = { ... };
#else
static const struct file_operations mydevice_proc_ops = {
    .owner   = THIS_MODULE,
    .open    = mydevice_proc_open,
    .read    = seq_read,
    .llseek  = seq_lseek,
    .release = single_release,
};
#endif
```
For a modern Ubuntu/Fedora system (kernel ≥ 5.6), use `struct proc_ops` directly.
Add to `mydevice_init` (after `device_create` succeeds):
```c
proc_entry = proc_create("mydevice", 0444, NULL, &mydevice_proc_ops);
if (!proc_entry) {
    pr_err("mydevice: proc_create failed\n");
    ret = -ENOMEM;
    goto err_destroy_device;
}
```
Add `proc_remove(proc_entry);` as the **first line** of `mydevice_exit`.
**Checkpoint 5:**
```bash
make EXTRA_CFLAGS="-Werror"
sudo insmod mydevice.ko
sudo chmod 666 /dev/mydevice
# Verify /proc entry exists
ls /proc/mydevice
# Must not error
cat /proc/mydevice
# Must output all 5 fields
echo "test data" > /dev/mydevice
cat /proc/mydevice
# bytes_used must be 10 ("test data\n" = 10 bytes)
# write_count must be 1
# Test partial read handling (seq_file robustness):
dd if=/proc/mydevice bs=1 2>/dev/null | cat
# Must output all 5 lines without corruption
# Test multiple sequential reads:
for i in 1 2 3; do cat /proc/mydevice; echo "---"; done
# Each cat must show consistent output
sudo rmmod mydevice
# Verify /proc entry is gone:
cat /proc/mydevice 2>&1 | grep "No such file"
# Must match
```
### Phase 6: Userspace Test Program (1–2 hours)
Create `mydevice/test_mydevice.c`:
```c
/* test_mydevice.c — Milestone 3 userspace acceptance test */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include "mydevice.h"
#define DEVICE_PATH "/dev/mydevice"
#define PROC_PATH   "/proc/mydevice"
static int pass_count = 0;
static int fail_count = 0;
static void test_pass(const char *name)
{
    printf("  PASS: %s\n", name);
    pass_count++;
}
static void test_fail(const char *name, const char *reason)
{
    printf("  FAIL: %s — %s\n", name, reason);
    fail_count++;
}
static void print_proc(void)
{
    FILE *f;
    char line[256];
    printf("\n  [/proc/mydevice]\n");
    f = fopen(PROC_PATH, "r");
    if (!f) {
        printf("  (could not open /proc/mydevice: %s)\n", strerror(errno));
        return;
    }
    while (fgets(line, sizeof(line), f))
        printf("    %s", line);
    fclose(f);
    printf("\n");
}
int main(void)
{
    int fd, ret;
    printf("=== Milestone 3 Userspace Test ===\n\n");
    fd = open(DEVICE_PATH, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "ERROR: cannot open %s: %s\n", DEVICE_PATH, strerror(errno));
        fprintf(stderr, "Run: sudo insmod mydevice.ko && sudo chmod 666 /dev/mydevice\n");
        return EXIT_FAILURE;
    }
    printf("Opened %s (fd=%d)\n", DEVICE_PATH, fd);
    /* --- Write initial data for counter tests --- */
    const char *msg = "hello ioctl";
    ret = write(fd, msg, strlen(msg));
    if (ret == (int)strlen(msg))
        test_pass("initial write succeeds");
    else
        test_fail("initial write", "wrong byte count");
    print_proc();
    /* --- TEST 1: MYDEVICE_IOC_STATUS --- */
    printf("[TEST 1] MYDEVICE_IOC_STATUS\n");
    {
        struct mydevice_status status;
        ret = ioctl(fd, MYDEVICE_IOC_STATUS, &status);
        if (ret == 0)
            test_pass("STATUS ioctl returns 0");
        else
            test_fail("STATUS ioctl", strerror(errno));
        if (status.buffer_size == 4096)
            test_pass("STATUS.buffer_size == 4096");
        else {
            char buf[64];
            snprintf(buf, sizeof(buf), "got %lu", status.buffer_size);
            test_fail("STATUS.buffer_size", buf);
        }
        if (status.bytes_used == strlen(msg)) {
            test_pass("STATUS.bytes_used == strlen(msg)");
        } else {
            char buf[64];
            snprintf(buf, sizeof(buf), "got %lu, expected %zu", status.bytes_used, strlen(msg));
            test_fail("STATUS.bytes_used", buf);
        }
        if (status.open_count >= 1)
            test_pass("STATUS.open_count >= 1");
        else
            test_fail("STATUS.open_count", "should be at least 1");
        if (status.write_count >= 1)
            test_pass("STATUS.write_count >= 1 after write");
        else
            test_fail("STATUS.write_count", "should be >= 1 after write");
    }
    /* --- TEST 2: MYDEVICE_IOC_CLEAR --- */
    printf("\n[TEST 2] MYDEVICE_IOC_CLEAR\n");
    {
        ret = ioctl(fd, MYDEVICE_IOC_CLEAR);
        if (ret == 0)
            test_pass("CLEAR ioctl returns 0");
        else
            test_fail("CLEAR ioctl", strerror(errno));
        /* After clear: lseek to 0 and read should return 0 bytes */
        lseek(fd, 0, SEEK_SET);
        char buf[64];
        ssize_t n = read(fd, buf, sizeof(buf));
        if (n == 0)
            test_pass("read after CLEAR returns 0 (EOF)");
        else {
            char msg2[64];
            snprintf(msg2, sizeof(msg2), "got %zd bytes", n);
            test_fail("read after CLEAR", msg2);
        }
        /* STATUS should show bytes_used == 0 */
        struct mydevice_status status;
        ioctl(fd, MYDEVICE_IOC_STATUS, &status);
        if (status.bytes_used == 0)
            test_pass("STATUS.bytes_used == 0 after CLEAR");
        else {
            char msg2[64];
            snprintf(msg2, sizeof(msg2), "got %lu", status.bytes_used);
            test_fail("STATUS.bytes_used after CLEAR", msg2);
        }
    }
    /* --- TEST 3: MYDEVICE_IOC_RESIZE --- */
    printf("\n[TEST 3] MYDEVICE_IOC_RESIZE to 8192\n");
    {
        unsigned long new_size = 8192;
        ret = ioctl(fd, MYDEVICE_IOC_RESIZE, &new_size);
        if (ret == 0)
            test_pass("RESIZE to 8192 returns 0");
        else
            test_fail("RESIZE to 8192", strerror(errno));
        struct mydevice_status status;
        ioctl(fd, MYDEVICE_IOC_STATUS, &status);
        if (status.buffer_size == 8192)
            test_pass("STATUS.buffer_size == 8192 after RESIZE");
        else {
            char buf[64];
            snprintf(buf, sizeof(buf), "got %lu", status.buffer_size);
            test_fail("STATUS.buffer_size after RESIZE", buf);
        }
        /* Test RESIZE with data: write 100 bytes, shrink to 50, verify truncation */
        write(fd, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", 100);
        unsigned long small_size = 50;
        ret = ioctl(fd, MYDEVICE_IOC_RESIZE, &small_size);
        if (ret == 0)
            test_pass("RESIZE to 50 (truncation) returns 0");
        else
            test_fail("RESIZE to 50", strerror(errno));
        ioctl(fd, MYDEVICE_IOC_STATUS, &status);
        if (status.bytes_used == 50)
            test_pass("bytes_used == 50 after truncating RESIZE");
        else {
            char buf[64];
            snprintf(buf, sizeof(buf), "got %lu", status.bytes_used);
            test_fail("bytes_used after truncating RESIZE", buf);
        }
    }
    /* --- TEST 4: RESIZE invalid sizes --- */
    printf("\n[TEST 4] RESIZE invalid sizes\n");
    {
        unsigned long zero = 0;
        ret = ioctl(fd, MYDEVICE_IOC_RESIZE, &zero);
        if (ret == -1 && errno == EINVAL)
            test_pass("RESIZE size=0 returns EINVAL");
        else
            test_fail("RESIZE size=0", "expected EINVAL");
        unsigned long too_big = 2000000;
        ret = ioctl(fd, MYDEVICE_IOC_RESIZE, &too_big);
        if (ret == -1 && errno == EINVAL)
            test_pass("RESIZE size=2000000 returns EINVAL");
        else
            test_fail("RESIZE size=2000000", "expected EINVAL");
    }
    /* --- TEST 5: Unknown ioctl (wrong nr, right magic) → ENOTTY --- */
    printf("\n[TEST 5] Unknown ioctl → ENOTTY\n");
    {
        ret = ioctl(fd, _IO(MYDEVICE_IOC_MAGIC, 99));
        if (ret == -1 && errno == ENOTTY)
            test_pass("_IO('k', 99) returns ENOTTY");
        else
            test_fail("_IO('k', 99)", "expected ENOTTY");
    }
    /* --- TEST 6: Wrong magic number → ENOTTY --- */
    printf("\n[TEST 6] Wrong magic number → ENOTTY\n");
    {
        ret = ioctl(fd, _IO('Z', 0));
        if (ret == -1 && errno == ENOTTY)
            test_pass("_IO('Z', 0) returns ENOTTY");
        else
            test_fail("_IO('Z', 0)", "expected ENOTTY");
    }
    /* --- TEST 7: /proc entry fields --- */
    printf("\n[TEST 7] /proc/mydevice field verification\n");
    {
        FILE *f = fopen(PROC_PATH, "r");
        if (!f) {
            test_fail("/proc/mydevice open", strerror(errno));
        } else {
            char content[512] = {0};
            fread(content, 1, sizeof(content) - 1, f);
            fclose(f);
            const char *fields[] = {"buffer_size:", "bytes_used:", "open_count:",
                                    "read_count:", "write_count:"};
            for (int i = 0; i < 5; i++) {
                if (strstr(content, fields[i]))
                    test_pass(fields[i]);
                else
                    test_fail("/proc field missing", fields[i]);
            }
        }
    }
    print_proc();
    close(fd);
    printf("\n=== Results: %d passed, %d failed ===\n", pass_count, fail_count);
    return (fail_count == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
```
Build the test program:
```bash
gcc -Wall -Wextra -I. -o test_mydevice test_mydevice.c
```
**Checkpoint 6:**
```bash
make EXTRA_CFLAGS="-Werror"
gcc -Wall -Wextra -I. -o test_mydevice test_mydevice.c
# Must compile with zero warnings
sudo insmod mydevice.ko
sudo chmod 666 /dev/mydevice
sudo ./test_mydevice
# All lines must show PASS
# Final output: "Results: N passed, 0 failed"
sudo rmmod mydevice
```

![seq_file Mechanics: Why proc_read Breaks for Large Output](./diagrams/tdd-diag-17.svg)

---
## 8. Test Specification
### 8.1 ioctl Validation Layer Tests
| Test | Command | Expected | Verification |
|---|---|---|---|
| Wrong magic (type='Z') | `ioctl(fd, _IO('Z', 0))` | `-1`, `errno = ENOTTY` | Test 6 in test program |
| Wrong magic (type=0x00) | `ioctl(fd, 0x00000000)` | `-1`, `errno = ENOTTY` | Python inline test |
| Right magic, nr=3 (> MAXNR) | `ioctl(fd, _IO('k', 3))` | `-1`, `errno = ENOTTY` | Test 5 in test program |
| Right magic, nr=99 | `ioctl(fd, _IO('k', 99))` | `-1`, `errno = ENOTTY` | Test 5 in test program |
| NULL pointer for RESIZE | `ioctl(fd, MYDEVICE_IOC_RESIZE, NULL)` | `-1`, `errno = EFAULT` | `access_ok(NULL, 8)` fails |
| Kernel address for STATUS | `ioctl(fd, MYDEVICE_IOC_STATUS, 0xffff888000000000UL)` | `-1`, `errno = EFAULT` | Requires C test; Python can't construct kernel addrs |
| CLEAR with non-zero arg | `ioctl(fd, MYDEVICE_IOC_CLEAR, 12345)` | `0` (success, arg ignored) | Correct: `_IO` means no arg |
### 8.2 `MYDEVICE_IOC_RESIZE` — Happy Path
| Test | Setup | Expected | Verification |
|---|---|---|---|
| Resize to 8192 | Load with default 4096; resize to 8192 | Returns 0; STATUS shows buffer_size=8192 | Test 3 in test program |
| Resize to 1 (minimum) | `new_size = 1` | Returns 0; STATUS shows buffer_size=1 | Direct ioctl call |
| Resize to 1048576 (maximum) | `new_size = 1048576` | Returns 0; STATUS shows buffer_size=1048576 | Direct ioctl call |
| Resize with data, no truncation | Write 10 bytes, resize to 100 | Returns 0; bytes_used=10; buffer_size=100 | STATUS after resize |
| Resize with data, truncation | Write 100 bytes, resize to 50 | Returns 0; bytes_used=50; buffer_size=50 | Test 3 (truncation case) |
| Write after resize (larger) | Resize to 8192; write 5000 bytes | Write returns 5000; bytes_used=5000 | echo + cat verification |
### 8.3 `MYDEVICE_IOC_RESIZE` — Failure Cases
| Test | Setup | Expected | State After |
|---|---|---|---|
| `new_size = 0` | Direct ioctl | `-EINVAL` | buffer_size unchanged |
| `new_size = 1048577` | Direct ioctl | `-EINVAL` | buffer_size unchanged |
| `new_size = ULONG_MAX` | `new_size = ~0UL` | `-EINVAL` | buffer_size unchanged |
| NULL pointer | `ioctl(fd, MYDEVICE_IOC_RESIZE, NULL)` | `-EFAULT` (access_ok) | buffer_size unchanged |
### 8.4 `MYDEVICE_IOC_CLEAR` — All Cases
| Test | Setup | Expected | Verification |
|---|---|---|---|
| Clear non-empty buffer | Write "hello", CLEAR | Returns 0; read returns 0 bytes | Test 2 |
| Clear empty buffer | No prior write, CLEAR | Returns 0; read returns 0 bytes | Idempotent |
| Data after clear | Write, CLEAR, write "world" | cat shows "world" only | Round-trip test |
### 8.5 `MYDEVICE_IOC_STATUS` — Counter Accuracy
| Test | Setup | Expected | Verification |
|---|---|---|---|
| Initial counters | Load module, STATUS immediately | read_count=0, write_count=0 | Immediate post-load query |
| After 1 write | Write "abc", STATUS | write_count=1, bytes_used=3 | Test 1 |
| After 1 read | Write "abc", read all, STATUS | read_count=1 | Needs separate STATUS call |
| After CLEAR | Write, CLEAR, STATUS | bytes_used=0; write_count includes prior writes | Test 2 |
| After RESIZE | Resize to 8192, STATUS | buffer_size=8192 | Test 3 |
| Multiple opens | Two fds open simultaneously, STATUS | open_count=2 | Requires two simultaneous opens |
### 8.6 `/proc/mydevice` Tests
| Test | Command | Expected |
|---|---|---|
| File exists | `ls /proc/mydevice` | Exit 0 |
| Permission 0444 | `stat -c "%a" /proc/mydevice` | `444` |
| All 5 fields present | `cat /proc/mydevice \| grep -c ":"` | `5` |
| buffer_size field | `grep "buffer_size" /proc/mydevice` | `buffer_size:  4096` |
| bytes_used after write | Write "hello", `grep bytes_used /proc/mydevice` | `bytes_used:   6` (with newline) |
| write_count increments | Write twice, check | `write_count:  2` |
| Partial read (1 byte) | `dd if=/proc/mydevice bs=1 2>/dev/null \| wc -c` | > 0, all data intact |
| Entry removed after rmmod | `cat /proc/mydevice` after `rmmod` | "No such file or directory" |
| Multiple sequential reads | `for i in 1 2 3; do cat /proc/mydevice; done` | Consistent output each time |
### 8.7 Build Quality
```bash
# Zero warnings with -Werror in kernel module:
make clean && make EXTRA_CFLAGS="-Werror" 2>&1 | grep -E "^.*warning:|^.*error:" | grep -v "^make"
# Expected: empty
# Zero warnings in test program:
gcc -Wall -Wextra -I. -o test_mydevice test_mydevice.c 2>&1 | grep -E "warning:|error:"
# Expected: empty
# Sparse analysis (optional but recommended):
make C=1 EXTRA_CFLAGS="-Werror" 2>&1 | grep -v "^make"
# Expected: no __user dereference warnings
```
### 8.8 `modinfo` Metadata
```bash
modinfo mydevice.ko | grep -E "^license:|^version:"
# license: GPL
# version: 0.3
```
### 8.9 Cleanup Correctness After M3
```bash
sudo insmod mydevice.ko
# Verify all resources created:
ls /proc/mydevice /dev/mydevice /sys/class/mydevice/
sudo rmmod mydevice
# Verify all resources removed:
ls /proc/mydevice 2>&1 | grep "No such file"
ls /dev/mydevice 2>&1 | grep "No such file"
ls /sys/class/mydevice/ 2>&1 | grep "No such file"
```
---
## 9. Performance Targets
| Operation | Target | How to Measure |
|---|---|---|
| `MYDEVICE_IOC_STATUS` latency | 2–5 µs | `strace -T ./test_mydevice 2>&1 \| grep "ioctl.*STATUS"` — value in brackets |
| `MYDEVICE_IOC_CLEAR` latency (4KB buffer) | < 5 µs | `strace -T` on CLEAR ioctl; dominated by `memset` of 64 cache lines |
| `MYDEVICE_IOC_RESIZE` latency (4KB→8KB) | < 20 µs | `strace -T` on RESIZE; includes one `kzalloc`, one `memcpy`, one `kfree` |
| ioctl validation overhead (ENOTTY path) | < 100 ns | 3 integer comparisons, register-only, no memory access |
| `cat /proc/mydevice` latency | < 1 ms | `time cat /proc/mydevice` — wall clock |
| `/proc` single read call (seq_read) | < 50 µs | `strace -T cat /proc/mydevice 2>&1 \| grep read` |
| `insmod mydevice.ko` (M3) | < 150 ms | `time sudo insmod mydevice.ko` |
| `rmmod mydevice` (M3) | < 50 ms | `time sudo rmmod mydevice` |
| `mydevice.ko` file size | < 30 KB | `ls -lh mydevice.ko` |
| `sizeof(struct mydevice_status)` | 40 bytes (64-bit) | `python3 -c "import struct; print(struct.calcsize('QQIxxxxQQ'))"` |
**`MYDEVICE_IOC_STATUS` latency breakdown:**
| Sub-operation | Estimated Cost |
|---|---|
| Syscall boundary (ring 3→0) | ~100 ns |
| VFS ioctl dispatch chain | ~50 ns (2 pointer indirections) |
| `_IOC_TYPE`/`_IOC_NR` validation | ~5 ns (bit operations, registers) |
| `access_ok` check | ~10 ns (range comparison, no memory access) |
| 3× `atomic_read` for counters | ~30 ns (3 MOV instructions; `.bss` likely L1 hot) |
| 2× direct variable reads (`buffer_size`, `buffer_data_len`) | ~10 ns |
| Struct assembly on kernel stack | ~5 ns (5 stores, L1 hot) |
| `copy_to_user` 40 bytes | ~200 ns (STAC + store 40 bytes + CLAC) |
| Return + ring 0→3 | ~100 ns |
| **Total** | **~510 ns – 2 µs** |
**`cat /proc/mydevice` latency breakdown:**
| Sub-operation | Estimated Cost |
|---|---|
| `open()` syscall → `mydevice_proc_open` → `single_open` | ~5–10 µs (single `kmalloc` for seq_file struct) |
| `read()` → `seq_read` → `mydevice_proc_show` → 5× `seq_printf` | ~10–20 µs (writes ~80 bytes of formatted ASCII) |
| `copy_to_user` ~80 bytes to `cat`'s buffer | ~200 ns |
| Second `read()` → EOF path | ~1 µs |
| `close()` → `single_release` → `kfree` | ~2 µs |
| **Total** | **~20–50 µs** |
---
## 10. Concurrency Specification
M3 inherits M2's explicitly-unprotected concurrent access model. This section documents the known races introduced by the new ioctl commands and the `atomic_t` guarantees for counters.
### 10.1 RESIZE Race with Concurrent read/write
```
CPU 0 (mydevice_read):             CPU 1 (RESIZE ioctl):
reads buffer_data_len = 100        new_buf = kzalloc(50, GFP_KERNEL)
                                   memcpy(new_buf, kernel_buffer, 50)
                                   buffer_data_len = 50
                                   kfree(kernel_buffer)     ← OLD BUFFER FREED
copy_to_user(buf, kernel_buffer    ← USE-AFTER-FREE CRASH
              + f_pos, to_copy)
```
This race is a **use-after-free vulnerability** that can crash the kernel. It is **intentionally deferred to M4**. The acceptance criteria for M3 do not include concurrent stress testing — that test belongs to M4. For single-process testing (one process at a time), M3 is safe.
**Do not attempt to add partial locking here.** Half-implemented locking is worse than no locking — it creates a false sense of safety. M4 adds the complete mutex solution.
### 10.2 CLEAR Race with Concurrent read
```
CPU 0 (mydevice_read):             CPU 1 (MYDEVICE_IOC_CLEAR):
reads buffer_data_len = 50         memset(kernel_buffer, 0, buffer_size)
available = 50 - f_pos             buffer_data_len = 0
copy_to_user(buf, kernel_buffer    ← reads zeroed data (correct memory, wrong value)
              + f_pos, available)
```
Outcome: the reader gets all-zero bytes instead of the actual data. The memory access is valid (the buffer pointer is still good), so no crash occurs — but the data is wrong. This is a data correctness issue, not a safety issue. Acceptable at M3.
### 10.3 `atomic_t` Guarantees for `read_count` and `write_count`
`atomic_inc(&read_count)` compiles to `LOCK XADD [mem], 1` on x86_64. This is a single indivisible bus-locked operation. Multiple CPUs simultaneously executing `mydevice_read` can each call `atomic_inc` without races. The final value of `read_count` after N concurrent reads correctly reflects N, without lost updates. No additional synchronization is needed for these statistics counters.
The snapshot in `mydevice_ioctl_status` reads `buffer_size`, `buffer_data_len`, and the atomic counters sequentially. Without a mutex, these reads are not atomic with respect to each other — a write can occur between reading `bytes_used` and reading `write_count`, producing a STATUS snapshot that is internally inconsistent (e.g., `write_count` is 1 but `bytes_used` is 0 because the CLEAR ran in between). This inconsistency is visible but harmless — it does not cause crashes or memory corruption. The M4 mutex makes STATUS snapshots consistent.

![/proc/mydevice Architecture: proc_ops → seq_file → Kernel Stats](./diagrams/tdd-diag-18.svg)

---
## 11. Common Pitfalls Reference
**Pitfall 1: `_IOC_READ`/`_IOC_WRITE` direction confusion.**
`_IOW` (user writes to kernel) sets `_IOC_WRITE` in the command bits. But `_IOC_WRITE` means the kernel **reads** from user. For `access_ok` validation: check `_IOC_DIR(cmd) & _IOC_READ` to validate pointer for `_IOR` commands; check `_IOC_DIR(cmd) & _IOC_WRITE` for `_IOW` commands. The naming is from the kernel's perspective, opposite to the macro names.
**Pitfall 2: Using `struct file_operations` instead of `struct proc_ops` for proc files on kernel ≥ 5.6.**
`proc_create` on modern kernels expects `const struct proc_ops *`. Using `struct file_operations` causes a compiler type mismatch error. The field names differ: `proc_read` not `read`, `proc_lseek` not `llseek`, `proc_release` not `release`.
**Pitfall 3: Checking `proc_create` return value with `IS_ERR()` instead of `!= NULL`.**
`proc_create` returns NULL on failure, not `ERR_PTR(errno)`. Using `IS_ERR(proc_entry)` will not catch the NULL failure case — `IS_ERR(NULL)` returns false. Always check: `if (!proc_entry)`.
**Pitfall 4: Not deferring `/proc` cleanup (`proc_remove`) until `mydevice_exit` starts.**
If `proc_remove` is called after `kfree(kernel_buffer)`, a concurrent `cat /proc/mydevice` reads `mydevice_proc_show`, which accesses `buffer_data_len` and `buffer_size` — both potentially garbage after `kfree`. `proc_remove` blocks until all in-progress reads complete and prevents new reads. Call it **first** in `mydevice_exit`.
**Pitfall 5: `copy_from_user` for RESIZE receives `unsigned long arg` not `unsigned long *`.**
The ioctl handler receives `arg` as `unsigned long` (not a pointer). For `_IOW` commands, `arg` IS a userspace pointer (the userspace address containing the value). Cast to `(unsigned long __user *)` and pass to `copy_from_user`. Do not pass `arg` directly as the source — it is an address, not the value. The pattern: `copy_from_user(&new_size, (unsigned long __user *)arg, sizeof(new_size))`.
**Pitfall 6: Forgetting to `kfree(new_buf)` when `mutex_lock_interruptible` (M4) or validation fails after allocation in RESIZE.**
In M3, the allocation is at step 3 and validation is at step 2, so validation always precedes allocation — no leak on EINVAL. However, the general rule: if you allocate before acquiring a lock or before a step that can fail, you must free on all failure paths after the allocation. This becomes critical in M4.
**Pitfall 7: Including kernel-only headers in `mydevice.h`.**
`mydevice.h` is compiled by both the kernel build system and `gcc` for userspace. Headers like `<linux/fs.h>`, `<linux/slab.h>`, `<linux/atomic.h>` are kernel-only and will fail to compile in userspace. Only `<linux/ioctl.h>` (available as a UAPI header) may be included. All other kernel headers belong in `mydevice.c` only.
**Pitfall 8: `module_param` type mismatch with `buffer_size` declared as `size_t`.**
`module_param(buffer_size, ulong, 0444)` — after M3's RESIZE changes `buffer_size` dynamically, it must remain `size_t buffer_size`. The `ulong` module_param type token is compatible with `size_t` on 64-bit Linux. Using `uint` would truncate on resize to sizes > 4GB (theoretical but incorrect).
**Pitfall 9: Not validating `access_ok` for `_IOWR` commands (both bits set).**
If you implement a future `_IOWR` command and only check one direction, a malicious userspace caller could pass a kernel-space address for the unchecked direction. Always check both `_IOC_READ` and `_IOC_WRITE` bits independently in the validation loop.
**Pitfall 10: `seq_printf` return value checking.**
Do NOT check the return value of individual `seq_printf` calls with `if (seq_printf(...) < 0) return -EIO`. The seq_file infrastructure handles buffer overflow internally by allocating more pages. Checking `seq_printf` return values and returning early from `show()` breaks the iterator contract and causes truncated output. Always let `seq_printf` run to completion and return 0 from the `show` function.
---
<!-- END_TDD_MOD -->


<!-- TDD_MOD_ID: build-kernel-module-m4 -->
# Technical Design Specification: Concurrent Access, Blocking I/O, and Poll Support (`build-kernel-module-m4`)
---
## 1. Module Charter
This module completes the character device driver by adding production-quality concurrent access control, blocking I/O with signal-safe sleep semantics, O_NONBLOCK support, and poll/select/epoll integration. The central additions are: a single `DEFINE_MUTEX` protecting the buffer invariant (`kernel_buffer`, `buffer_data_len`, `buffer_size`) in all read/write/ioctl paths; two `DECLARE_WAIT_QUEUE_HEAD` instances enabling readers to sleep when the buffer is empty and writers to sleep when the buffer is full; `wait_event_interruptible` with mandatory `-ERESTARTSYS` propagation so that signals (including SIGKILL) are never silently swallowed; `wake_up_interruptible` called by writers to unblock sleeping readers and by readers to unblock sleeping writers; a `.poll` file operation with `poll_wait` subscription plus an accurate readiness mask; and a FIFO buffer discipline (append-on-write, `memmove`-consume-on-read) replacing the M2/M3 overwrite model. All ioctl handlers that touch buffer state (`RESIZE`, `CLEAR`) are wrapped in `mutex_lock_interruptible`/`mutex_unlock`. A checksum-verified stress test with 4 concurrent writers and 4 concurrent readers provides the acceptance proof.
This module does **NOT** introduce spinlocks, tasklets, workqueues, timers, interrupt handlers, or any non-process-context code — the driver remains entirely process-context, making mutex (sleeping lock) the correct and only required primitive. It does NOT convert to a circular buffer; `memmove` after read is O(remaining_bytes) and acceptable for a learning driver. It does NOT add `fsync`, `mmap`, `fasync`, or `compat_ioctl`. The `/proc` entry introduced in M3 is carried forward unchanged; its `mydevice_proc_show` reads `buffer_data_len` and `buffer_size` without the mutex — a deliberate tradeoff: the `/proc` snapshot may be momentarily stale but is never unsafe (no memory corruption possible from a stale size read).
**Upstream dependency:** `build-kernel-module-m3` — the complete `file_operations` struct, `unlocked_ioctl`, `/proc` via seq_file, `atomic_t` counters (`read_count`, `write_count`, `open_count`), shared `mydevice.h`, and goto error-unwind chain are all inherited unchanged.
**Downstream dependency:** None. This is the final milestone. The driver is feature-complete after this module.
**Invariants that must hold at all times:**
- The mutex is never held while calling `wait_event_interruptible`. Violation causes deadlock: the waiting thread holds the lock, writers cannot acquire it to add data, the thread never wakes.
- `buffer_data_len <= buffer_size` at all times; enforced by write truncation inside the critical section.
- `kernel_buffer != NULL` at all times after `mydevice_init` returns 0; enforced by all code paths that modify the pointer.
- Every `mutex_lock_interruptible` that returns 0 is paired with exactly one `mutex_unlock`. Every code path that fails `mutex_lock_interruptible` returns `-ERESTARTSYS` without reaching `mutex_unlock`.
- `wake_up_interruptible(&mydevice_read_queue)` is called inside the critical section (after buffer state is updated) and the mutex is released after the wake. Readers woken by this call will block on the mutex, then recheck the condition.
- `-ERESTARTSYS` is never converted to 0, a positive byte count, or any other errno before being returned to the VFS layer.
---
## 2. File Structure
All files are in the existing `mydevice/` directory from M3. No new source files are created; `mydevice.c` is updated in place.
```
mydevice/                             ← existing working directory
├── Makefile                          ← unchanged from M3
├── mydevice.h                        ← unchanged from M3
├── mydevice.c                        ← UPDATED: all M4 changes go here
├── test_mydevice.c                   ← unchanged from M3 (M4 adds new test binary)
├── concurrent_test.py                ← NEW: Python checksum stress test
└── verify_m4.sh                      ← NEW: M4 acceptance test script
```
Build artifacts (unchanged):
```
mydevice/
├── mydevice.ko                       ← rebuilt by make after M4 changes
├── mydevice.o
├── mydevice.mod.c
├── mydevice.mod.o
├── Module.symvers
└── modules.order
```
---
## 3. Complete Data Model
### 3.1 New Synchronization Primitives — Global Declarations
```c
#include <linux/mutex.h>               /* DEFINE_MUTEX, mutex_lock_interruptible,
                                          mutex_unlock, struct mutex */
#include <linux/wait.h>                /* DECLARE_WAIT_QUEUE_HEAD, wait_queue_head_t,
                                          wait_event_interruptible, wake_up_interruptible */
#include <linux/poll.h>                /* poll_table, poll_wait, POLLIN, POLLOUT,
                                          POLLRDNORM, POLLWRNORM, __poll_t */
#include <linux/sched/signal.h>        /* signal_pending — needed on some kernel versions */
/* Single mutex protecting the buffer invariant */
static DEFINE_MUTEX(mydevice_mutex);
/* Wait queue for readers: readers sleep here when buffer_data_len == 0 */
static DECLARE_WAIT_QUEUE_HEAD(mydevice_read_queue);
/* Wait queue for writers: writers sleep here when buffer_data_len >= buffer_size */
static DECLARE_WAIT_QUEUE_HEAD(mydevice_write_queue);
```
**Memory layout of synchronization primitives (x86_64, approximate):**
| Variable | Type | Size (bytes) | Offset in .bss | Notes |
|---|---|---|---|---|
| `mydevice_mutex` | `struct mutex` | 32 | after atomic counters | Contains owner task pointer + wait list |
| `mydevice_read_queue` | `wait_queue_head_t` | 24 | after mutex | Contains spinlock + list head for waiting tasks |
| `mydevice_write_queue` | `wait_queue_head_t` | 24 | after read_queue | Same layout as read queue |
**`struct mutex` internal layout (kernel 6.x, x86_64):**
```
offset  0: atomic_long_t owner  (8 bytes) — 0 = unlocked, pointer = owner task
offset  8: spinlock_t    wait_lock (4 bytes) — protects the wait_list
offset 12: (4 bytes padding)
offset 16: struct list_head wait_list (16 bytes) — queue of blocked tasks
```
Total: 32 bytes. Fits in one cache line (64 bytes) with room for adjacent data.
**`wait_queue_head_t` internal layout:**
```
offset  0: spinlock_t    lock      (4 bytes) — protects the head list
offset  4: (4 bytes padding)
offset  8: struct list_head head   (16 bytes) — list of wait_queue_entry_t
```
Total: 24 bytes. Two wait queues fit in one 64-byte cache line together.
**Cache line analysis for hot-path globals:**
The buffer state variables (`kernel_buffer` [8B], `buffer_data_len` [8B], `buffer_size` [8B]) and the wait queue heads (48B total) together span approximately 72 bytes — crossing one cache line boundary. The mutex (32B) may share a line with `buffer_size`. On the hot path (read/write under contention), the scheduler brings these into L1 as a unit. For a learning driver this layout is acceptable; a production driver would use `__cacheline_aligned` on the mutex to prevent false sharing with read-only fields.
### 3.2 Updated Buffer Model — FIFO Semantics
M4 changes the buffer from "overwrite on write" (M2/M3) to **FIFO append semantics**:
```
buffer state: [consumed data] ... [valid data: indices 0..buffer_data_len) ] ... [free space]
write(): appends to kernel_buffer[buffer_data_len], increments buffer_data_len
read():  copies from kernel_buffer[0..to_copy), then memmoves remaining data:
           memmove(kernel_buffer, kernel_buffer + to_copy, buffer_data_len - to_copy)
           buffer_data_len -= to_copy
```
The `f_pos` field is **no longer used** for position tracking. Under FIFO semantics, `f_pos` in the `struct file` becomes meaningless — reads always consume from the head of the queue. The handler ignores `*f_pos` for position and the parameter is used only because the `file_operations` read signature requires it.
**Why FIFO and not the M2/M3 overwrite model:** Concurrent producers and consumers require a stream model, not a "single latest value" model. If writer A writes "hello" and writer B writes "world", a reader must see one complete message, not a torn mix. FIFO with mutex serialization delivers messages atomically. The overwrite model is only correct for single-producer single-consumer "latest value" semantics (like a sensor reading device).
**FIFO buffer state after a sequence of operations:**
```
Initial:       kernel_buffer = [........] buffer_data_len = 0  buffer_size = 8
write("AB"):   kernel_buffer = [AB......] buffer_data_len = 2
write("CD"):   kernel_buffer = [ABCD....] buffer_data_len = 4
read(3 bytes): copies "ABC", memmoves "D" to index 0
               kernel_buffer = [D.......] buffer_data_len = 1
write("EF"):   kernel_buffer = [DEF.....] buffer_data_len = 3
read(3 bytes): copies "DEF", buffer_data_len = 0
```
### 3.3 Complete Updated Global State Summary
| Variable | Type | Initial Value | Protected By | Modified By in M4 |
|---|---|---|---|---|
| `dev_num` | `dev_t` | 0 | write-once | `mydevice_init` only |
| `my_cdev` | `struct cdev` | zeroed | write-once | `mydevice_init` only |
| `mydevice_class` | `struct class *` | NULL | write-once | `mydevice_init` only |
| `mydevice_device` | `struct device *` | NULL | write-once | `mydevice_init` only |
| `proc_entry` | `struct proc_dir_entry *` | NULL | write-once | `mydevice_init` only |
| `kernel_buffer` | `char *` | NULL | **mydevice_mutex** | read/write/ioctl RESIZE |
| `buffer_data_len` | `size_t` | 0 | **mydevice_mutex** | read/write/ioctl CLEAR/RESIZE |
| `buffer_size` | `size_t` | 4096 | **mydevice_mutex** | ioctl RESIZE |
| `open_count` | `atomic_t` | 0 | atomic ops | open/release (unchanged) |
| `read_count` | `atomic_t` | 0 | atomic ops | mydevice_read (unchanged) |
| `write_count` | `atomic_t` | 0 | atomic ops | mydevice_write (unchanged) |
| `mydevice_mutex` | `struct mutex` | unlocked | self | all buffer-touching paths |
| `mydevice_read_queue` | `wait_queue_head_t` | empty | internal spinlock | read (sleeps), write (wakes) |
| `mydevice_write_queue` | `wait_queue_head_t` | empty | internal spinlock | write (sleeps), read (wakes) |

![The Corruption: Lost Update Without Mutex on buffer_data_len](./diagrams/tdd-diag-19.svg)

---
## 4. Interface Contracts
### 4.1 `mydevice_read(filp, buf, count, f_pos) → ssize_t`
```c
static ssize_t mydevice_read(struct file *filp, char __user *buf,
                              size_t count, loff_t *f_pos);
```
**Context:** Process context. Called by VFS `vfs_read()`. May sleep (in blocking mode). `buf` is a userspace pointer carrying `__user` annotation — never dereference directly.
**Parameters:**
- `filp`: Open file descriptor. `filp->f_flags & O_NONBLOCK` is checked before sleeping.
- `buf`: Userspace destination. Accessed only via `copy_to_user`.
- `count`: Maximum bytes caller will accept. May be 0.
- `f_pos`: Ignored for position under FIFO semantics. Not updated.
**Full algorithm — see Section 5.1.**
**Return values:**
| Return | Condition |
|---|---|
| `> 0` | Bytes successfully consumed from FIFO and copied to userspace |
| `0` | `count == 0` |
| `-EAGAIN` | O_NONBLOCK set and buffer is empty; or mutex contention resolved but data gone (spurious wakeup recovery) |
| `-ERESTARTSYS` | Signal received during `wait_event_interruptible` or `mutex_lock_interruptible` |
| `-EFAULT` | `copy_to_user` returned non-zero |
**Post-conditions on success:** `buffer_data_len` is reduced by `to_copy`; `kernel_buffer[0..buffer_data_len)` contains the remaining unconsumed data (shifted to front via `memmove`); `mydevice_write_queue` has been woken (space is now available); mutex is released; `read_count` is incremented.
**Critical: `-ERESTARTSYS` must never be converted to another value.** Return it directly. The VFS syscall layer intercepts it and either restarts the syscall (if `SA_RESTART` is set on the signal handler) or converts it to `-EINTR` for userspace. If your handler returns `0` instead of `-ERESTARTSYS` after a signal, the process silently loses track of an interrupted operation.
### 4.2 `mydevice_write(filp, buf, count, f_pos) → ssize_t`
```c
static ssize_t mydevice_write(struct file *filp, const char __user *buf,
                               size_t count, loff_t *f_pos);
```
**Context:** Process context. May sleep (in blocking mode).
**Parameters:** Same structure as read. `buf` is userspace source. `f_pos` ignored for FIFO position.
**Full algorithm — see Section 5.2.**
**Return values:**
| Return | Condition |
|---|---|
| `> 0` | Bytes appended to FIFO |
| `0` | `count == 0` |
| `-EAGAIN` | O_NONBLOCK set and buffer is full |
| `-ERESTARTSYS` | Signal received during wait or mutex lock |
| `-EFAULT` | `copy_from_user` returned non-zero |
**Post-conditions on success:** `buffer_data_len` increased by `to_copy`; new data appended at `kernel_buffer[old_buffer_data_len .. old_buffer_data_len + to_copy)`; `mydevice_read_queue` has been woken (data is now available); mutex released; `write_count` incremented.
### 4.3 `mydevice_poll(filp, wait) → __poll_t`
```c
static __poll_t mydevice_poll(struct file *filp, poll_table *wait);
```
**Context:** Process context. Called by `do_sys_poll` / `ep_poll` etc. **MUST NOT sleep and MUST NOT block.** This function is called in a non-blocking context even when the caller is epoll.
**Parameters:**
- `filp`: Open file descriptor.
- `wait`: Opaque struct managed by the poll infrastructure. Pass to `poll_wait`; do not examine directly.
**Algorithm:**
1. Call `poll_wait(filp, &mydevice_read_queue, wait)`.
2. Call `poll_wait(filp, &mydevice_write_queue, wait)`.
3. Declare `__poll_t mask = 0`.
4. If `buffer_data_len > 0`: `mask |= POLLIN | POLLRDNORM`.
5. If `buffer_data_len < buffer_size`: `mask |= POLLOUT | POLLWRNORM`.
6. Return `mask`.
**Why both `poll_wait` calls before the mask check:** `poll_wait` registers this fd with both wait queues. If the mask is returned as non-zero immediately, the poll call returns without ever sleeping — but the registration still happened for the next poll call that finds the fd not ready. If the mask is zero (nothing ready), the poll infrastructure puts the calling process to sleep; any `wake_up_interruptible` on either registered queue will re-invoke `.poll` to re-check.
**Why buffer_data_len is read WITHOUT the mutex:** The mask is a snapshot. A stale read (seeing 0 when data exists) causes a brief extra sleep — harmless. A stale read (seeing data when buffer was just cleared) causes `poll` to return `POLLIN`, which causes the caller to invoke `read()`, which will find no data under the mutex and return `-EAGAIN` — also harmless. Acquiring the mutex in `.poll` would risk deadlock in edge cases where the poll machinery calls `.poll` while the write path is waking readers under the mutex. The standard Linux driver pattern reads shared state without locking in `.poll`.
**Return values:** A bitmask of zero or more of `POLLIN | POLLRDNORM | POLLOUT | POLLWRNORM`. Never returns negative. Never sleeps.
**`POLLERR` and `POLLHUP`:** Not returned by this driver. These are for error conditions (e.g., a broken pipe) or hangup (e.g., one end of a connection closed). A character buffer device has neither concept.
### 4.4 `mydevice_ioctl_resize(arg) → int` (updated for mutex)
```c
static int mydevice_ioctl_resize(unsigned long arg);
```
**Changes from M3:** The buffer swap (steps 4–5 in M3's algorithm) is now wrapped in `mutex_lock_interruptible` / `mutex_unlock`. The `kzalloc` allocation (step 3) occurs **before** acquiring the mutex, because `kzalloc(GFP_KERNEL)` may sleep and sleeping while holding a mutex is correct but sleeping while *trying to acquire* a mutex is also correct — the key constraint is that `kzalloc` must not be called while holding the mutex if there is any chance of deadlock. In this driver there is no deadlock risk from calling `kzalloc` under the mutex (no code path allocates and then tries to acquire `mydevice_mutex`), but the convention of doing heavyweight allocations outside the critical section is followed for clarity.
**Updated algorithm — see Section 5.3.**
**New failure path:** If `mutex_lock_interruptible` returns non-zero after `kzalloc` has succeeded, the newly-allocated `new_buf` must be freed before returning `-ERESTARTSYS`. Failing to do so leaks kernel memory. This is the most common bug introduced when adding mutex protection to existing allocation code.
### 4.5 `mydevice_ioctl_clear(void) → int` (updated for mutex)
```c
static int mydevice_ioctl_clear(void);
```
**Changes from M3:** Entire body wrapped in mutex. After clearing, `wake_up_interruptible(&mydevice_write_queue)` is called to unblock any writers that were sleeping because the buffer was full — clearing has freed space.
**Algorithm:**
1. `if (mutex_lock_interruptible(&mydevice_mutex)) return -ERESTARTSYS;`
2. `memset(kernel_buffer, 0, buffer_size);`
3. `buffer_data_len = 0;`
4. `wake_up_interruptible(&mydevice_write_queue);`
5. `mutex_unlock(&mydevice_mutex);`
6. `pr_info("mydevice: buffer cleared\n");`
7. Return `0`.

![Mutex vs Spinlock Decision: Execution Context Matrix](./diagrams/tdd-diag-20.svg)

---
## 5. Algorithm Specification
### 5.1 `mydevice_read` — Complete Algorithm with Blocking, O_NONBLOCK, and Mutex
```
mydevice_read(filp, buf, count, f_pos):
STEP 0: TRIVIAL EARLY RETURNS
    if (count == 0): return 0
STEP 1: BLOCKING/NONBLOCKING GATE
    if (filp->f_flags & O_NONBLOCK):
        // Non-blocking: check WITHOUT mutex first (fast path)
        if (buffer_data_len == 0):
            return -EAGAIN
        // Data appears available; proceed to acquire mutex and recheck
    else:
        // Blocking mode: sleep until data is available
        // CRITICAL: do NOT hold mydevice_mutex here.
        // If we held the mutex, writers cannot acquire it to add data.
        // This would be an immediate deadlock.
        ret = wait_event_interruptible(mydevice_read_queue,
                                        buffer_data_len > 0)
        if (ret != 0):
            // ret == -ERESTARTSYS: signal received while sleeping
            // Propagate immediately. Do NOT return 0 or -EINTR here.
            return -ERESTARTSYS
STEP 2: ACQUIRE MUTEX
    if (mutex_lock_interruptible(&mydevice_mutex) != 0):
        // Signal arrived while waiting for the mutex itself
        return -ERESTARTSYS
    // --- CRITICAL SECTION BEGINS ---
STEP 3: RECHECK CONDITION UNDER MUTEX
    // Mandatory: another reader may have consumed the data between
    // when wait_event_interruptible returned and when we got the mutex.
    // This is the thundering herd scenario:
    //   - N readers sleeping, all wake when 1 item is written
    //   - First reader acquires mutex, consumes data, releases mutex
    //   - Remaining N-1 readers get mutex, find buffer_data_len == 0
    //   - They must return -EAGAIN (or re-sleep; see note below)
    if (buffer_data_len == 0):
        mutex_unlock(&mydevice_mutex)
        // For O_NONBLOCK this is the definitive answer.
        // For blocking mode, this is a spurious wakeup recovery:
        //   returning -EAGAIN is correct but not ideal — a fully
        //   correct blocking implementation would loop back to step 1.
        //   The simple approach used here (return -EAGAIN) is correct
        //   in the sense that POSIX allows read() to return EAGAIN even
        //   in blocking mode if no data is immediately available after
        //   waking up. Userspace should retry. For the stress test,
        //   callers handle EAGAIN correctly.
        return -EAGAIN
STEP 4: COMPUTE BYTES TO COPY
    to_copy = (count < buffer_data_len) ? count : buffer_data_len
STEP 5: COPY TO USERSPACE
    not_copied = copy_to_user(buf, kernel_buffer, to_copy)
    if (not_copied != 0):
        mutex_unlock(&mydevice_mutex)
        return -EFAULT
STEP 6: CONSUME FROM FIFO (memmove)
    buffer_data_len -= to_copy
    if (buffer_data_len > 0):
        memmove(kernel_buffer, kernel_buffer + to_copy, buffer_data_len)
    // kernel_buffer[0..buffer_data_len) now contains remaining data
STEP 7: WAKE WRITERS (space is available)
    wake_up_interruptible(&mydevice_write_queue)
    // Called inside critical section: writers wake up, try mutex,
    // block until we release it, then proceed with correct state.
STEP 8: UPDATE STATISTICS
    atomic_inc(&read_count)
    ret = (ssize_t)to_copy
    // --- CRITICAL SECTION ENDS ---
    mutex_unlock(&mydevice_mutex)
STEP 9: LOG AND RETURN
    pr_info("mydevice: read %zu bytes, %zu remaining\n",
            to_copy, buffer_data_len)
    return ret
```
**Why `wake_up_interruptible` is called INSIDE the critical section (Step 7, before `mutex_unlock`):**
This is a deliberate choice. The alternative is calling it after `mutex_unlock`. Both are correct in terms of correctness (no data race introduced), but calling before unlock has a subtle advantage: the woken writer will immediately try to acquire the mutex and block — preventing the pathological case where the OS schedules many other threads between the unlock and the wake, during which another writer could re-fill the buffer, causing the newly-woken writer to find no space. In practice for this driver it makes no difference, but the pre-unlock pattern is the standard convention in Linux kernel code.

![Wait Queue Lifecycle: Block → Sleep → Wakeup → Recheck → Proceed](./diagrams/tdd-diag-21.svg)

### 5.2 `mydevice_write` — Complete Algorithm with Blocking, O_NONBLOCK, and Mutex
```
mydevice_write(filp, buf, count, f_pos):
STEP 0: TRIVIAL EARLY RETURN
    if (count == 0): return 0
STEP 1: BLOCKING/NONBLOCKING GATE
    if (filp->f_flags & O_NONBLOCK):
        if (buffer_data_len >= buffer_size):
            return -EAGAIN
        // Space appears available; proceed to mutex + recheck
    else:
        // Blocking mode: sleep until space is available
        // DO NOT hold mydevice_mutex while sleeping.
        ret = wait_event_interruptible(mydevice_write_queue,
                                        buffer_data_len < buffer_size)
        if (ret != 0):
            return -ERESTARTSYS
STEP 2: ACQUIRE MUTEX
    if (mutex_lock_interruptible(&mydevice_mutex) != 0):
        return -ERESTARTSYS
    // --- CRITICAL SECTION BEGINS ---
STEP 3: RECHECK CONDITION UNDER MUTEX
    if (buffer_data_len >= buffer_size):
        mutex_unlock(&mydevice_mutex)
        return -EAGAIN
STEP 4: COMPUTE SPACE AND BYTES TO COPY
    space_available = buffer_size - buffer_data_len
    to_copy = (count < space_available) ? count : space_available
    // to_copy is always >= 1 here (buffer not full, count > 0)
STEP 5: COPY FROM USERSPACE
    not_copied = copy_from_user(kernel_buffer + buffer_data_len,
                                 buf, to_copy)
    if (not_copied != 0):
        mutex_unlock(&mydevice_mutex)
        return -EFAULT
STEP 6: UPDATE BUFFER
    buffer_data_len += to_copy
STEP 7: WAKE READERS (data is available)
    wake_up_interruptible(&mydevice_read_queue)
STEP 8: UPDATE STATISTICS
    atomic_inc(&write_count)
    ret = (ssize_t)to_copy
    // --- CRITICAL SECTION ENDS ---
    mutex_unlock(&mydevice_mutex)
STEP 9: LOG AND RETURN
    pr_info("mydevice: wrote %zu bytes, buffer now %zu/%zu\n",
            to_copy, buffer_data_len, buffer_size)
    return ret
```

![Lock/Wait Ordering: Read Handler Critical Section with Deadlock Prevention](./diagrams/tdd-diag-22.svg)

### 5.3 `mydevice_ioctl_resize` — Updated Algorithm with Mutex
```
mydevice_ioctl_resize(arg):
STEP 1: READ NEW SIZE FROM USERSPACE (outside mutex — I/O can fault)
    unsigned long new_size
    if (copy_from_user(&new_size, (unsigned long __user *)arg,
                        sizeof(new_size)) != 0):
        return -EFAULT
STEP 2: VALIDATE (outside mutex — no shared state read, pure arithmetic)
    if (new_size == 0 || new_size > 1048576):
        return -EINVAL
STEP 3: ALLOCATE NEW BUFFER (outside mutex — kzalloc may sleep)
    char *new_buf = kzalloc(new_size, GFP_KERNEL)
    if (!new_buf):
        return -ENOMEM
STEP 4: ACQUIRE MUTEX
    if (mutex_lock_interruptible(&mydevice_mutex) != 0):
        kfree(new_buf)    ← CRITICAL: prevent memory leak
        return -ERESTARTSYS
    // --- CRITICAL SECTION BEGINS ---
STEP 5: COPY EXISTING DATA
    if (buffer_data_len > 0):
        size_t copy_len = (buffer_data_len < new_size) ?
                           buffer_data_len : new_size
        memcpy(new_buf, kernel_buffer, copy_len)
        buffer_data_len = copy_len
    else:
        buffer_data_len = 0
STEP 6: SWAP BUFFER
    kfree(kernel_buffer)
    kernel_buffer = new_buf
    buffer_size   = new_size
STEP 7: WAKE BOTH QUEUES
    // New size may have created space (if enlarged) → wake writers
    // New size may have truncated data (if shrunk) → wake readers
    // (readers will find truncated data; writers will find new space)
    wake_up_interruptible(&mydevice_write_queue)
    wake_up_interruptible(&mydevice_read_queue)
    // --- CRITICAL SECTION ENDS ---
    mutex_unlock(&mydevice_mutex)
STEP 8: LOG AND RETURN
    pr_info("mydevice: buffer resized to %lu bytes\n", new_size)
    return 0
```
**Why Step 3 (kzalloc) precedes Step 4 (mutex acquisition):**
`kzalloc(GFP_KERNEL)` may sleep to reclaim pages. While sleeping is legal while holding a mutex (it's a sleeping lock by design), placing the allocation inside the critical section means no reads or writes can proceed for the duration of the allocation — potentially hundreds of microseconds. By allocating first, the critical section is reduced to a few memcpy + pointer swaps: microseconds, not hundreds. The tradeoff: if a signal arrives between Step 3 and Step 4, the new buffer must be freed in Step 4's error path.

![-ERESTARTSYS Signal Flow: From Kernel Handler to Userspace errno](./diagrams/tdd-diag-23.svg)

### 5.4 `wait_event_interruptible` Internal Mechanics
Understanding this macro is mandatory for correct usage. It expands to approximately:
```c
/* Conceptual expansion of:
 * wait_event_interruptible(mydevice_read_queue, buffer_data_len > 0)
 */
{
    int __ret = 0;
    /* Fast path: if condition is already true, skip the sleep entirely */
    if (!(buffer_data_len > 0)) {
        DEFINE_WAIT(__wait);
        for (;;) {
            /* Add ourselves to the wait queue and set state INTERRUPTIBLE */
            prepare_to_wait(&mydevice_read_queue, &__wait,
                            TASK_INTERRUPTIBLE);
            /* Check condition AFTER setting state:
             * prevents the race where wake_up fires between
             * the condition check and setting INTERRUPTIBLE */
            if (buffer_data_len > 0)
                break;
            /* Check for pending signals BEFORE sleeping */
            if (signal_pending(current)) {
                __ret = -ERESTARTSYS;
                break;
            }
            /* Yield the CPU: we are now sleeping */
            schedule();
            /* We have been woken. Loop back: re-check condition.
             * This handles BOTH spurious wakeups AND thundering herd:
             * - spurious: condition still false, go back to sleep
             * - thundering herd: condition false (another reader got it),
             *   go back to sleep OR break with condition false (caller rechecks)
             */
        }
        finish_wait(&mydevice_read_queue, &__wait);
    }
    __ret;  /* evaluates to 0 (success) or -ERESTARTSYS */
}
```
**The condition-between-set-and-sleep ordering:** The `prepare_to_wait` + condition check + `schedule()` sequence is carefully ordered. If `wake_up_interruptible` fires between `prepare_to_wait` and `schedule()`, the task is already in `TASK_INTERRUPTIBLE` state in the wait queue — the wake_up sets it back to `TASK_RUNNING`, and when `schedule()` runs, the scheduler immediately returns (the task is runnable). This prevents the "missed wakeup" bug: data is written, wake_up fires, reader sets INTERRUPTIBLE state, reader checks condition (false because it read stale value), reader calls schedule() and sleeps forever. The kernel's implementation avoids this with memory barriers that guarantee the condition check sees the updated value.
{{DIAGRAM:tdd-diag-24}}
### 5.5 `-ERESTARTSYS` Propagation Chain
```
1. Signal delivered to process P (e.g., SIGINT from Ctrl+C)
2. P is sleeping in wait_event_interruptible inside mydevice_read
3. Kernel's signal delivery path calls try_to_wake_up(P)
4. P's state changes from TASK_INTERRUPTIBLE to TASK_RUNNING
5. P's schedule() call returns
6. wait_event_interruptible's loop: signal_pending(current) is TRUE
7. wait_event_interruptible returns -ERESTARTSYS
8. mydevice_read checks: (ret != 0) → returns -ERESTARTSYS to VFS
9. VFS syscall layer (do_syscall_64 / entry_SYSCALL_64) receives -ERESTARTSYS
10. Checks signal handler registration for SIGINT:
    - If SA_RESTART set: converts -ERESTARTSYS to restart the syscall
      → mydevice_read is called again from the beginning
    - If SA_RESTART not set: converts -ERESTARTSYS to -EINTR
      → userspace sees read() return -1 with errno == EINTR
11. Userspace signal handler for SIGINT runs
12. If errno == EINTR: well-behaved programs retry the read
```
**What happens if you return 0 instead of -ERESTARTSYS:**
The process's signal handler still runs (signals are delivered regardless), but the `read()` syscall returns 0 to userspace — which looks like EOF. The program thinks the file ended. If it's in a read loop, it exits. This is a subtle, hard-to-debug behavior: `cat /dev/mydevice &; kill -INT $!` would cause `cat` to silently exit with no output, making it look like the device produced no data.
**What happens if you return -EINTR instead of -ERESTARTSYS:**
Programs using `SA_RESTART` expect the syscall to be automatically restarted. By returning -EINTR, you bypass the restart mechanism — the program sees `errno == EINTR` every time even if it registered `SA_RESTART`. This breaks programs that rely on transparent signal handling (many POSIX programs do).

![.poll Implementation: poll_wait Subscription + Readiness Mask Return](./diagrams/tdd-diag-25.svg)

### 5.6 memmove After FIFO Read — Correctness and Performance
```c
/* After copying to_copy bytes from kernel_buffer[0..to_copy) to userspace: */
buffer_data_len -= to_copy;
if (buffer_data_len > 0)
    memmove(kernel_buffer, kernel_buffer + to_copy, buffer_data_len);
```
**Why `memmove` and not `memcpy`:** The source (`kernel_buffer + to_copy`) and destination (`kernel_buffer`) overlap when `to_copy < buffer_data_len / 2`. `memcpy` with overlapping regions is undefined behavior in C. `memmove` is defined for overlapping regions; the kernel's implementation uses forward copy when `dst < src` (which is always the case here — `dst = kernel_buffer` < `src = kernel_buffer + to_copy`), so there is no actual performance difference for this direction. Use `memmove` unconditionally for correctness.
**Performance analysis:**
- `buffer_data_len` after the copy is the number of bytes shifted.
- Each byte requires one load + one store; the CPU's `rep movsb` / `rep movsd` microcode handles the cache-line-aligned bulk efficiently.
- For a 4KB buffer with 1 byte consumed: shifts 4095 bytes = 64 cache lines ≈ 4–10 µs (L1 hot).
- For a 4KB buffer with 4KB consumed: `buffer_data_len == 0`, `memmove` is not called.
- Amortized across normal usage (reading chunks of similar size to writing chunks): the shift cost is proportional to the unconsumed tail, which is typically small.
**Circular buffer alternative (not implemented):** A circular buffer with `head` and `tail` indices eliminates the shift entirely: `tail = (tail + to_copy) % buffer_size`. At the cost of more complex `copy_to_user` (two calls when the readable region wraps around the buffer end) and more complex `copy_from_user` (same). For a production driver with >10MB/s throughput, implement the circular buffer. For this learning driver, `memmove` is correct and clear.
---
## 6. Error Handling Matrix
| Error | Detected By | Module State After | User-Visible | Must Free? |
|---|---|---|---|---|
| Signal during `wait_event_interruptible` in read | `wait_event_interruptible` returns `-ERESTARTSYS` | Unchanged (mutex not held, buffer untouched) | `read()` returns -1, `errno=EINTR` (or syscall restarts) | No allocations made |
| Signal during `mutex_lock_interruptible` in read | Return value `!= 0` | Unchanged (mutex not acquired) | `read()` returns -1, `errno=EINTR` | No allocations made |
| `copy_to_user` fails in read | `not_copied != 0` | Mutex unlocked before return; `buffer_data_len` NOT decremented (data preserved in buffer); `memmove` NOT called | `read()` returns -1, `errno=EFAULT` | Mutex (released) |
| O_NONBLOCK read on empty buffer | `filp->f_flags & O_NONBLOCK` + `buffer_data_len == 0` | Unchanged | `read()` returns -1, `errno=EAGAIN` | None |
| Spurious wakeup — no data under mutex | `buffer_data_len == 0` after acquiring mutex | Mutex released before return | `read()` returns -1, `errno=EAGAIN`; caller retries | Mutex (released) |
| Signal during `wait_event_interruptible` in write | Return `-ERESTARTSYS` | Unchanged | `write()` returns -1, `errno=EINTR` | None |
| Signal during `mutex_lock_interruptible` in write | Return `-ERESTARTSYS` | Unchanged | `write()` returns -1, `errno=EINTR` | None |
| `copy_from_user` fails in write | `not_copied != 0` | Mutex released; `buffer_data_len` NOT incremented; `kernel_buffer` contains garbage at `[old_buffer_data_len .. old_buffer_data_len + to_copy)` — **mitigate with memset or partial accounting** | `write()` returns -1, `errno=EFAULT` | Mutex |
| O_NONBLOCK write on full buffer | `filp->f_flags & O_NONBLOCK` + `buffer_data_len >= buffer_size` | Unchanged | `write()` returns -1, `errno=EAGAIN` | None |
| Signal during `mutex_lock_interruptible` in RESIZE (after kzalloc) | Return `-ERESTARTSYS` | `new_buf` must be freed before return; old buffer preserved | `ioctl()` returns -1, `errno=EINTR` | `new_buf` (kfree) |
| `kzalloc` fails in RESIZE | `!new_buf` after `kzalloc` | Old buffer preserved intact | `ioctl()` returns -1, `errno=ENOMEM` | None |
| Signal during `mutex_lock_interruptible` in CLEAR | Return `-ERESTARTSYS` | Unchanged | `ioctl()` returns -1, `errno=EINTR` | None |
| `poll` called on empty buffer | `buffer_data_len == 0` in `.poll` | Mask has no `POLLIN`; process sleeps in poll until woken | `poll()` / `select()` blocks until timeout or data arrives | None |
| `poll` called on full buffer | `buffer_data_len >= buffer_size` | Mask has no `POLLOUT` | `poll()` / `select()` blocks for write readiness | None |
| `rmmod` while readers sleeping in wait queue | `try_module_get` / `module_put` prevents rmmod | rmmod returns `EBUSY` | User sees "Device or resource busy" | None — correct behavior |
**copy_from_user partial failure note:** If `copy_from_user` returns `not_copied > 0` after copying some bytes successfully, the bytes already copied into `kernel_buffer + buffer_data_len` are garbage that partially overwrites that region. The safe handling is to return `-EFAULT` without updating `buffer_data_len` — the kernel slab memory is logically "dirty" in `[old_len .. old_len + (to_copy - not_copied))` but since `buffer_data_len` was not advanced, no reader will ever access those bytes. A production driver would `memset` the region to zero after a failed copy, but for this driver it is sufficient to not advance `buffer_data_len`.

![FIFO Buffer State: write-append + read-consume + memmove](./diagrams/tdd-diag-26.svg)

---
## 7. Implementation Sequence with Checkpoints
### Phase 1: DEFINE_MUTEX + Mutex in read/write (1–1.5 hours)
Add to `mydevice.c` includes:
```c
#include <linux/mutex.h>
#include <linux/wait.h>
#include <linux/poll.h>
```
Add to globals:
```c
static DEFINE_MUTEX(mydevice_mutex);
static DECLARE_WAIT_QUEUE_HEAD(mydevice_read_queue);
static DECLARE_WAIT_QUEUE_HEAD(mydevice_write_queue);
```
**Change the buffer model to FIFO append.** Update `mydevice_write` to append at `kernel_buffer + buffer_data_len` instead of writing from index 0. Remove the `memset(kernel_buffer, 0, buffer_size)` that preceded the write. Remove `*f_pos = 0` from write. Update `mydevice_read` to always read from `kernel_buffer[0]` (not `kernel_buffer + *f_pos`), and add the `memmove` after consuming bytes. Remove the `*f_pos` advance.
Wrap the body of `mydevice_read` (after count==0 check) in `mutex_lock_interruptible` / `mutex_unlock`. Same for `mydevice_write`. **No wait queues yet — just mutex.** For now: if the buffer is empty in read, return `0` (EOF). If the buffer is full in write, return `-ENOSPC` temporarily. These return values will be replaced in Phase 3.
**Checkpoint 1:**
```bash
make EXTRA_CFLAGS="-Werror"
# Must compile with zero warnings
sudo insmod mydevice.ko
sudo chmod 666 /dev/mydevice
# FIFO test: write then read in correct order
echo -n "hello" > /dev/mydevice
echo -n "world" >> /dev/mydevice  # append (shell reopens fd, so new write appends to FIFO)
cat /dev/mydevice
# Expected: "helloworld" on one read (or "hello" then "world" in two reads)
# Mutex test: verify no kernel oops in dmesg
dmesg | grep -E "BUG:|oops|null pointer" && echo "FAIL" || echo "PASS: no oops"
sudo rmmod mydevice
```
### Phase 2: FIFO Buffer — Verify Append and memmove (0.5 hours)
Write a focused Python test to verify the FIFO semantics:
```bash
sudo insmod mydevice.ko
sudo chmod 666 /dev/mydevice
python3 -c "
import os
fd = os.open('/dev/mydevice', os.O_RDWR)
os.write(fd, b'AAAA')
os.write(fd, b'BBBB')
data = os.read(fd, 4)
print('First read:', data)    # Expected: b'AAAA'
data = os.read(fd, 4)
print('Second read:', data)   # Expected: b'BBBB'
data = os.read(fd, 4)
print('Third read:', data)    # Expected: b'' (empty = 0 bytes, EOF/EAGAIN)
os.close(fd)
"
sudo rmmod mydevice
```
**Checkpoint 2:** First read returns `b'AAAA'`, second returns `b'BBBB'`, third returns `b''` or `-EAGAIN`. FIFO ordering confirmed.
### Phase 3: wait_event_interruptible for Blocking Read (1–2 hours)
Replace the temporary `return 0` (when buffer is empty) in `mydevice_read` with the full wait_event_interruptible pattern from Section 5.1. Add `wake_up_interruptible(&mydevice_write_queue)` after the memmove inside the critical section.
**Checkpoint 3:**
```bash
sudo insmod mydevice.ko
sudo chmod 666 /dev/mydevice
# Terminal 1: cat should HANG (blocking on empty buffer)
timeout 3 cat /dev/mydevice &
CAT_PID=$!
sleep 0.5
# Terminal 2: write data — cat should wake up
echo -n "wake_test" > /dev/mydevice
# Wait for cat to terminate
wait $CAT_PID
echo "cat exit: $?"
dmesg | tail -5 | grep -E "read|wrote"
# Expected: cat printed "wake_test" and exited cleanly
# Signal test: verify Ctrl+C (SIGINT) terminates blocked cat
timeout 3 cat /dev/mydevice &
CAT_PID=$!
sleep 0.3
kill -INT $CAT_PID
wait $CAT_PID
echo "cat exit after SIGINT: $?"
# Expected: cat exits with non-zero; NOT stuck in D state
ps aux | grep "cat /dev/mydevice" | grep -v grep && echo "FAIL: still running" || echo "PASS: terminated"
sudo rmmod mydevice
```
### Phase 4: Blocking Write + wake_up_interruptible on Write (1–1.5 hours)
Replace the temporary `-ENOSPC` (when buffer is full) in `mydevice_write` with `wait_event_interruptible(mydevice_write_queue, buffer_data_len < buffer_size)`. Ensure `wake_up_interruptible(&mydevice_read_queue)` is called after incrementing `buffer_data_len` inside the critical section.
**Checkpoint 4:**
```bash
sudo insmod mydevice.ko buffer_size=16
sudo chmod 666 /dev/mydevice
# Fill the buffer to capacity
python3 -c "
import os, time, threading
fd = os.open('/dev/mydevice', os.O_WRONLY)
# Write exactly 16 bytes (fills buffer_size=16)
os.write(fd, b'A' * 16)
print('Buffer filled')
# Spawn a thread that will block trying to write more
def blocked_write():
    print('Blocked write starting...')
    n = os.write(fd, b'B' * 8)  # should block until reader consumes some
    print(f'Blocked write completed: {n} bytes')
t = threading.Thread(target=blocked_write)
t.start()
time.sleep(0.5)  # give the write time to block
print('Draining buffer...')
rfd = os.open('/dev/mydevice', os.O_RDONLY)
data = os.read(rfd, 8)  # consume 8 bytes → writer should wake up
print(f'Read {len(data)} bytes')
t.join(timeout=1.0)
print('PASS' if not t.is_alive() else 'FAIL: writer still blocked')
os.close(rfd)
os.close(fd)
"
sudo rmmod mydevice
```
### Phase 5: O_NONBLOCK — -EAGAIN on Empty/Full Buffer (0.5–1 hour)
Add the `filp->f_flags & O_NONBLOCK` check at the beginning of both `mydevice_read` and `mydevice_write` (before `wait_event_interruptible`). If O_NONBLOCK and buffer empty (read) or buffer full (write): return `-EAGAIN` immediately without sleeping.
**Checkpoint 5:**
```bash
sudo insmod mydevice.ko
sudo chmod 666 /dev/mydevice
python3 -c "
import os, errno
# Test O_NONBLOCK read on empty buffer
fd = os.open('/dev/mydevice', os.O_RDONLY | os.O_NONBLOCK)
try:
    os.read(fd, 100)
    print('FAIL: should have raised EAGAIN')
except OSError as e:
    print('PASS' if e.errno == errno.EAGAIN else f'FAIL: errno={e.errno}')
os.close(fd)
# Fill buffer then test O_NONBLOCK write
wfd = os.open('/dev/mydevice', os.O_WRONLY)
os.write(wfd, b'X' * 4096)
os.close(wfd)
wfd2 = os.open('/dev/mydevice', os.O_WRONLY | os.O_NONBLOCK)
try:
    os.write(wfd2, b'Y')
    print('FAIL: buffer full, should have raised EAGAIN')
except OSError as e:
    print('PASS' if e.errno == errno.EAGAIN else f'FAIL: errno={e.errno}')
os.close(wfd2)
"
sudo rmmod mydevice
```
### Phase 6: .poll File Operation (0.5–1 hour)
Add `mydevice_poll` function and register `.poll = mydevice_poll` in `mydevice_fops`.
```c
static __poll_t mydevice_poll(struct file *filp, poll_table *wait)
{
	__poll_t mask = 0;
	poll_wait(filp, &mydevice_read_queue,  wait);
	poll_wait(filp, &mydevice_write_queue, wait);
	if (buffer_data_len > 0)
		mask |= POLLIN | POLLRDNORM;
	if (buffer_data_len < buffer_size)
		mask |= POLLOUT | POLLWRNORM;
	return mask;
}
```
**Checkpoint 6:**
```bash
sudo insmod mydevice.ko
sudo chmod 666 /dev/mydevice
python3 -c "
import select, os, time
# Test: empty buffer → not readable
rfd = os.open('/dev/mydevice', os.O_RDONLY | os.O_NONBLOCK)
r, w, x = select.select([rfd], [], [], 0.05)
print('Empty: readable =', len(r) > 0, '(expected False)')
# Write data then test readable
wfd = os.open('/dev/mydevice', os.O_WRONLY)
os.write(wfd, b'poll_data')
os.close(wfd)
r, w, x = select.select([rfd], [], [], 0.1)
print('After write: readable =', len(r) > 0, '(expected True)')
# Test writable
wfd2 = os.open('/dev/mydevice', os.O_WRONLY | os.O_NONBLOCK)
r, w, x = select.select([], [wfd2], [], 0.05)
print('Has space: writable =', len(w) > 0, '(expected True)')
os.close(rfd)
os.close(wfd2)
"
# Test wake-up via poll: reader polls, writer adds data, poll returns
python3 -c "
import select, os, time, threading
rfd = os.open('/dev/mydevice', os.O_RDONLY | os.O_NONBLOCK)
woke_at = [None]
def poller():
    r, w, x = select.select([rfd], [], [], 2.0)
    woke_at[0] = time.time()
    print('poll woke:', len(r) > 0)
t = threading.Thread(target=poller)
t.start()
start = time.time()
time.sleep(0.1)
wfd = os.open('/dev/mydevice', os.O_WRONLY)
os.write(wfd, b'wakedata')
os.close(wfd)
t.join()
print(f'Wakeup latency: {(woke_at[0]-start)*1000:.1f}ms (target < 200ms)')
os.close(rfd)
"
sudo rmmod mydevice
```
### Phase 7: Mutex in ioctl RESIZE and CLEAR (0.5–1 hour)
Update `mydevice_ioctl_resize` per Section 5.3 (move `mutex_lock_interruptible` after `kzalloc`, add `kfree(new_buf)` before returning `-ERESTARTSYS`). Update `mydevice_ioctl_clear` per Section 4.5.
**Checkpoint 7:**
```bash
sudo insmod mydevice.ko
sudo chmod 666 /dev/mydevice
# Verify RESIZE under write load
python3 -c "
import os, threading, time
wfd = os.open('/dev/mydevice', os.O_WRONLY)
os.write(wfd, b'preserve_this')
os.close(wfd)
# RESIZE to 8192: data should be preserved
import struct, fcntl
MYDEVICE_IOC_RESIZE = (1 << 30) | (8 << 16) | (ord('k') << 8) | 0
fd = os.open('/dev/mydevice', os.O_RDWR)
fcntl.ioctl(fd, MYDEVICE_IOC_RESIZE, struct.pack('Q', 8192))
data = os.read(fd, 100)
print('Data after RESIZE:', data, '(expected b\"preserve_this\")')
os.close(fd)
"
# Verify CLEAR wakes blocked writers
sudo rmmod mydevice
```
### Phase 8: Stress Test and verify_m4.sh (1–2 hours)
Create `mydevice/concurrent_test.py`:
```python
#!/usr/bin/env python3
"""
concurrent_test.py - Checksum-verified concurrent stress test for mydevice (M4)
Usage: sudo python3 concurrent_test.py
"""
import hashlib
import os
import threading
import time
import queue
import sys
import errno
DEVICE      = "/dev/mydevice"
NUM_WRITERS = 4
NUM_READERS = 4
MESSAGES_PER_WRITER = 100
MESSAGE_SIZE        = 64   # bytes per message — well within 4KB buffer
sent_checksums     = queue.Queue()
received_checksums = []
checksum_lock      = threading.Lock()
stop_event         = threading.Event()
errors             = []
error_lock         = threading.Lock()
def writer(writer_id: int):
    char  = bytes([ord('A') + writer_id])
    msg   = char * MESSAGE_SIZE
    cksum = hashlib.sha256(msg).hexdigest()
    try:
        fd = os.open(DEVICE, os.O_WRONLY)
    except OSError as e:
        with error_lock: errors.append(f"W{writer_id} open: {e}")
        return
    sent = 0
    for _ in range(MESSAGES_PER_WRITER):
        retries = 0
        while True:
            try:
                n = os.write(fd, msg)
                if n == MESSAGE_SIZE:
                    sent_checksums.put(cksum)
                    sent += 1
                else:
                    with error_lock:
                        errors.append(f"W{writer_id}: partial write {n}/{MESSAGE_SIZE}")
                break
            except OSError as e:
                if e.errno in (errno.EAGAIN, errno.EINTR):
                    retries += 1
                    if retries > 1000:
                        with error_lock: errors.append(f"W{writer_id}: too many retries")
                        break
                    time.sleep(0.001)
                else:
                    with error_lock: errors.append(f"W{writer_id}: write error {e}")
                    break
        time.sleep(0.0005 * (writer_id + 1))
    os.close(fd)
    print(f"  Writer {writer_id}: sent {sent}/{MESSAGES_PER_WRITER}")
def reader(reader_id: int):
    try:
        fd = os.open(DEVICE, os.O_RDONLY)
    except OSError as e:
        with error_lock: errors.append(f"R{reader_id} open: {e}")
        return
    received = 0
    while not stop_event.is_set():
        try:
            data = os.read(fd, MESSAGE_SIZE)
            if len(data) == MESSAGE_SIZE:
                cksum = hashlib.sha256(data).hexdigest()
                with checksum_lock:
                    received_checksums.append(cksum)
                received += 1
            elif len(data) > 0:
                # Partial read — unusual for blocking mode but handle it
                with error_lock:
                    errors.append(f"R{reader_id}: partial read {len(data)} bytes")
        except OSError as e:
            if e.errno in (errno.EAGAIN, errno.EINTR, errno.ERESTART):
                time.sleep(0.001)
            else:
                break
    os.close(fd)
    print(f"  Reader {reader_id}: received {received}")
def main():
    if not os.path.exists(DEVICE):
        print(f"ERROR: {DEVICE} not found")
        sys.exit(1)
    if os.geteuid() != 0:
        print("WARNING: not running as root; device may be unreadable")
    total_msgs = NUM_WRITERS * MESSAGES_PER_WRITER
    print(f"Stress test: {NUM_WRITERS}W × {MESSAGES_PER_WRITER} msgs × "
          f"{MESSAGE_SIZE}B = {total_msgs} messages total")
    print(f"Readers: {NUM_READERS} | Buffer size: 4096 bytes (default)")
    reader_threads = [
        threading.Thread(target=reader, args=(i,), daemon=True)
        for i in range(NUM_READERS)
    ]
    for t in reader_threads: t.start()
    time.sleep(0.2)  # let readers settle in wait queue
    writer_threads = [
        threading.Thread(target=writer, args=(i,))
        for i in range(NUM_WRITERS)
    ]
    for t in writer_threads: t.start()
    for t in writer_threads: t.join()
    print("  All writers finished. Waiting for buffer drain...")
    time.sleep(3)
    stop_event.set()
    for t in reader_threads: t.join(timeout=3)
    # Collect expected checksums
    expected = {}
    while not sent_checksums.empty():
        c = sent_checksums.get_nowait()
        expected[c] = expected.get(c, 0) + 1
    corrupted = [c for c in received_checksums if c not in expected]
    total_recv = len(received_checksums)
    print(f"\n=== Results ===")
    print(f"  Messages sent:     {sum(expected.values())}")
    print(f"  Messages received: {total_recv}")
    print(f"  Corrupted:         {len(corrupted)}")
    print(f"  Errors:            {len(errors)}")
    for e in errors[:5]: print(f"    {e}")
    if corrupted:
        print("FAIL: data corruption detected")
        sys.exit(1)
    elif len(errors) > 0:
        print(f"FAIL: {len(errors)} errors")
        sys.exit(1)
    else:
        print("PASS: all received messages have valid checksums")
        print("Data integrity confirmed under concurrent access.")
if __name__ == "__main__":
    main()
```
Create `mydevice/verify_m4.sh`:
```bash
#!/bin/bash
# verify_m4.sh — Milestone 4 acceptance test
# Usage: sudo bash verify_m4.sh
set -euo pipefail
MODULE="mydevice"
DEVICE="/dev/mydevice"
PASS=0; FAIL=0
pass() { echo "  PASS: $1"; PASS=$((PASS+1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL+1)); }
lsmod | grep -q "^${MODULE} " && rmmod "${MODULE}" 2>/dev/null || true
echo "=== Milestone 4 Verification ==="
# Build
echo "--- Build ---"
make EXTRA_CFLAGS="-Werror" 2>&1 | grep -E "warning:|error:" | grep -v "^make" \
    && fail "build warnings/errors" || pass "build: zero warnings with -Werror"
[ -f ./mydevice.ko ] && pass "mydevice.ko exists" || { fail "mydevice.ko missing"; exit 1; }
insmod ./mydevice.ko
sleep 0.2
chmod 666 "${DEVICE}"
# Mutex: concurrent writes don't corrupt
echo "--- Mutex Protection ---"
python3 concurrent_test.py && pass "stress test PASS" || fail "stress test FAIL"
# Blocking read
echo "--- Blocking I/O ---"
timeout 3 cat "${DEVICE}" &
CAT_PID=$!
sleep 0.3
echo -n "wakeup_data" > "${DEVICE}"
wait $CAT_PID
[ $? -eq 0 ] && pass "blocking read wakes on write" || fail "blocking read failed"
# SIGINT terminates blocked reader
timeout 3 cat "${DEVICE}" &
CAT_PID=$!
sleep 0.3
kill -INT $CAT_PID 2>/dev/null
wait $CAT_PID 2>/dev/null || true
ps aux | grep -v grep | grep -q "cat ${DEVICE}" \
    && fail "SIGINT did not terminate blocked reader" \
    || pass "SIGINT terminates blocked reader (-ERESTARTSYS working)"
# O_NONBLOCK
echo "--- O_NONBLOCK ---"
python3 -c "
import os, errno, sys
fd = os.open('${DEVICE}', os.O_RDONLY | os.O_NONBLOCK)
try:
    os.read(fd, 100)
    sys.exit(1)
except OSError as e:
    sys.exit(0 if e.errno == errno.EAGAIN else 2)
" && pass "O_NONBLOCK read on empty → EAGAIN" || fail "O_NONBLOCK read wrong errno"
python3 -c "
import os, errno, sys, struct, fcntl
MYDEVICE_IOC_RESIZE = (1 << 30) | (8 << 16) | (ord('k') << 8) | 0
fd = os.open('${DEVICE}', os.O_RDWR)
fcntl.ioctl(fd, MYDEVICE_IOC_RESIZE, struct.pack('Q', 16))
os.write(fd, b'X' * 16)
wfd = os.open('${DEVICE}', os.O_WRONLY | os.O_NONBLOCK)
try:
    os.write(wfd, b'Y')
    sys.exit(1)
except OSError as e:
    sys.exit(0 if e.errno == errno.EAGAIN else 2)
" && pass "O_NONBLOCK write on full → EAGAIN" || fail "O_NONBLOCK write wrong errno"
rmmod "${MODULE}"; insmod ./mydevice.ko; sleep 0.1; chmod 666 "${DEVICE}"
# Poll
echo "--- Poll Support ---"
python3 -c "
import select, os, sys
fd = os.open('${DEVICE}', os.O_RDONLY | os.O_NONBLOCK)
r, w, x = select.select([fd], [], [], 0.05)
ok = len(r) == 0
wfd = os.open('${DEVICE}', os.O_WRONLY)
os.write(wfd, b'poll_check')
os.close(wfd)
r, w, x = select.select([fd], [], [], 0.1)
ok = ok and len(r) > 0
os.close(fd)
sys.exit(0 if ok else 1)
" && pass "poll: empty→not-readable, data→readable" || fail "poll mask incorrect"
python3 -c "
import select, os, sys
fd = os.open('${DEVICE}', os.O_WRONLY | os.O_NONBLOCK)
r, w, x = select.select([], [fd], [], 0.05)
os.close(fd)
sys.exit(0 if len(w) > 0 else 1)
" && pass "poll: has-space → writable" || fail "poll POLLOUT incorrect"
# Poll wakeup latency
python3 -c "
import select, os, time, threading
rfd = os.open('${DEVICE}', os.O_RDONLY | os.O_NONBLOCK)
woke = [None]
def poller():
    r, w, x = select.select([rfd], [], [], 2.0)
    woke[0] = time.time()
t = threading.Thread(target=poller); t.start()
start = time.time(); time.sleep(0.05)
wfd = os.open('${DEVICE}', os.O_WRONLY); os.write(wfd, b'lat'); os.close(wfd)
t.join(2.0)
latency_ms = (woke[0] - start) * 1000 if woke[0] else 9999
os.close(rfd)
import sys; sys.exit(0 if latency_ms < 200 else 1)
" && pass "poll wakeup latency < 200ms" || fail "poll wakeup too slow"
echo ""
rmmod "${MODULE}"
dmesg | grep -cE "BUG:|oops|null pointer" > /dev/null 2>&1 \
    && fail "kernel oops detected in dmesg" \
    || pass "no kernel oops"
echo ""
echo "=============================="
echo "Results: ${PASS} passed, ${FAIL} failed"
echo "=============================="
[ "${FAIL}" -eq 0 ] && exit 0 || exit 1
```
**Checkpoint 8:**
```bash
make EXTRA_CFLAGS="-Werror"
sudo bash verify_m4.sh
# All lines: PASS
# concurrent_test.py: "PASS: all received messages have valid checksums"
```

![poll/epoll Stack: From .poll to nginx Event Loop](./diagrams/tdd-diag-27.svg)

---
## 8. Test Specification
### 8.1 `mydevice_read` — Blocking Mode
| Test | Setup | Expected | Verification |
|---|---|---|---|
| Block on empty buffer | Open device (no prior write), blocking read | `read()` hangs until data arrives | Two-terminal test in Checkpoint 3 |
| Wake up when data written | Blocking reader + concurrent writer | Reader returns written bytes within one scheduler quantum | `concurrent_test.py` + `verify_m4.sh` |
| Read exactly available bytes | Write 10 bytes, read with count=100 | Returns 10 bytes; next read blocks (FIFO empty) | Python: `os.read(fd, 100)` returns `b'...'` 10-byte result |
| Partial read drains correctly | Write 8 bytes, read 4 + read 4 | First read: 4 bytes; second read: 4 bytes; third read: blocks | Python sequential reads |
| Signal interrupts blocking read | `cat /dev/mydevice &; kill -INT $PID` | Process exits; not stuck in D state | `verify_m4.sh` SIGINT test |
| `SA_RESTART` transparent restart | Handler registered with `SA_RESTART`, signal fires, data arrives later | Process eventually gets data without seeing `EINTR` | C test with `sigaction(SA_RESTART)` |
### 8.2 `mydevice_read` — O_NONBLOCK Mode
| Test | Setup | Expected | Verification |
|---|---|---|---|
| Empty buffer → EAGAIN | `O_RDONLY | O_NONBLOCK`, no data | Returns `-1`, `errno == EAGAIN` immediately | Checkpoint 5, `verify_m4.sh` |
| Data available → reads immediately | `O_RDONLY | O_NONBLOCK`, data present | Returns data without blocking | Python: write then nonblocking read |
| EAGAIN after consuming all data | Write 4 bytes, read 4 bytes (non-blocking), read again | Second read: `-EAGAIN` | Sequential nonblocking reads |
### 8.3 `mydevice_write` — Blocking Mode
| Test | Setup | Expected | Verification |
|---|---|---|---|
| Block on full buffer | Fill 4096-byte buffer, blocking write | `write()` hangs until reader consumes | Checkpoint 4 |
| Wake up when reader drains | Blocked writer + concurrent reader | Writer proceeds within scheduler quantum | `concurrent_test.py` |
| Partial write on nearly-full buffer | 4090 bytes in buffer, write 20 bytes | Returns 6 (space available); remaining 14 would require blocking or retry | Python: verify return value |
### 8.4 `mydevice_write` — O_NONBLOCK Mode
| Test | Setup | Expected | Verification |
|---|---|---|---|
| Full buffer → EAGAIN | Fill buffer, `O_WRONLY | O_NONBLOCK`, write 1 byte | Returns `-1`, `errno == EAGAIN` immediately | `verify_m4.sh` |
| Space available → writes immediately | Empty device, `O_WRONLY | O_NONBLOCK` | Returns bytes written without blocking | Checkpoint 5 |
### 8.5 `mydevice_poll` — Readiness Mask Correctness
| Test | Buffer State | Expected Mask | Verification |
|---|---|---|---|
| Empty buffer | `buffer_data_len == 0` | `POLLOUT | POLLWRNORM` only | `select.select([fd], [], [], 0.05)`: fd not in readable set |
| Data available | `buffer_data_len > 0` | `POLLIN | POLLRDNORM | POLLOUT | POLLWRNORM` | After writing: `select.select([fd], [], [], 0.05)`: fd in readable set |
| Full buffer | `buffer_data_len == buffer_size` | `POLLIN | POLLRDNORM` only | Fill buffer: `select.select([], [fd], [], 0.05)`: fd not in writable set |
| Wakeup via write | Reader in `select()`, writer writes data | `select()` returns within scheduler quantum | `verify_m4.sh` poll wakeup test |
| Both `poll_wait` called | Single `.poll` call | Both read and write queues subscribed | Verified by: write-then-clear cycle where poll correctly tracks both transitions |
### 8.6 `mydevice_poll` — Both `poll_wait` Calls Required
```bash
python3 -c "
import select, os, time, threading
# Test: subscribe via poll, then write data, then verify poll fires
rfd = os.open('/dev/mydevice', os.O_RDONLY | os.O_NONBLOCK)
fired = [False]
def watcher():
    r, w, x = select.select([rfd], [], [], 1.0)
    fired[0] = len(r) > 0
t = threading.Thread(target=watcher); t.start()
time.sleep(0.1)
wfd = os.open('/dev/mydevice', os.O_WRONLY)
os.write(wfd, b'notify_test'); os.close(wfd)
t.join(1.0)
print('PASS' if fired[0] else 'FAIL: poll did not fire')
os.close(rfd)
"
```
### 8.7 `mydevice_ioctl_resize` — Mutex Safety Under Concurrent I/O
| Test | Setup | Expected | Verification |
|---|---|---|---|
| RESIZE while readers blocked | Start blocking reader, then RESIZE | RESIZE succeeds; reader either gets data or EAGAIN; no crash | Python: reader thread + ioctl RESIZE |
| Data preserved on grow | Write 10 bytes, RESIZE to 8192 | STATUS shows bytes_used=10 after resize | test_mydevice (M3 test still valid) |
| Signal during RESIZE (mutex wait) | Write to fill buffer, block a writer, RESIZE from another thread | RESIZE returns ERESTARTSYS if signaled; no memory leak | Difficult to trigger deterministically; code review for `kfree(new_buf)` path |
### 8.8 Concurrent Stress Test — Checksum Verification
```bash
# Primary acceptance test
sudo python3 concurrent_test.py
# Must output: "PASS: all received messages have valid checksums"
# Must output: "Corrupted: 0"
# Must output: "Errors: 0"
```
**What a corrupted message looks like:** If two writers' data is interleaved (missing mutex), a reader receives a 64-byte block containing some bytes from writer A and some from writer B. The SHA-256 of this mixed block will not match any known-good checksum from the `sent_checksums` queue — it is flagged as corrupted.
### 8.9 Build Quality
```bash
make clean
make EXTRA_CFLAGS="-Werror" 2>&1 | grep -E "warning:|error:" | grep -v "^make"
# Expected: empty (zero output = zero warnings)
# Verify no __user dereferences (sparse analysis, if installed)
which sparse 2>/dev/null && \
    make C=1 2>&1 | grep -i "noderef\|user\|context" || echo "sparse not installed"
```
### 8.10 `rmmod` After Stress Test
```bash
sudo python3 concurrent_test.py
# Wait for test to complete (all fds closed by test)
sudo rmmod mydevice
echo "rmmod exit: $?"   # Must be 0
lsmod | grep mydevice   # Must be empty
dmesg | tail -2 | grep "module unloaded"  # Must appear
```
{{DIAGRAM:tdd-diag-28}}
---
## 9. Performance Targets
| Operation | Target | How to Measure |
|---|---|---|
| Blocking read wakeup latency | < 200 µs | `verify_m4.sh` poll wakeup test (Python `time.time()` bracketing); scheduler quantum is ~4ms on desktop, wakeup typically < 200µs |
| Uncontended `mutex_lock` + `mutex_unlock` | 30–60 cycles (~10–20 ns) | `perf stat -e cycles sudo python3 -c "import os; [os.write(fd,b'x') and os.read(fd,1) for _ in range(10000)]"` |
| `mydevice_read` hot path (mutex acquired, data available, 64 bytes) | < 5 µs | `strace -T cat /dev/mydevice 2>&1 \| grep read` — value in brackets |
| `mydevice_write` hot path (mutex acquired, space available, 64 bytes) | < 5 µs | `strace -T ./write_test 2>&1 \| grep write` |
| `mydevice_poll` (no blocking, buffer has data) | < 1 µs | `strace -T python3 -c "import select,os; select.select([fd],[],[],0)"` |
| `wake_up_interruptible` (wakes 1 sleeping reader) | < 10 µs (including scheduler decision) | Measured by `concurrent_test.py` implicit timing |
| Stress test throughput (4W + 4R, 64-byte messages) | > 50,000 messages/second | `time sudo python3 concurrent_test.py` ÷ 400 messages |
| `memmove` 4KB tail after 1-byte read | < 5 µs | 64 cache lines × ~70 ns/line = ~4.5 µs |
| `concurrent_test.py` total runtime | < 30 seconds | `time sudo python3 concurrent_test.py` |
| Module `.ko` file size (M4) | < 35 KB | `ls -lh mydevice.ko` |
**Latency budget for one `write(fd, msg, 64)` → reader wakeup → `read()` returns sequence:**
| Sub-operation | Estimated Cost |
|---|---|
| `write()` syscall entry | ~100 ns |
| `wait_event_interruptible` fast path (no sleep, condition already true) | ~50 ns |
| `mutex_lock_interruptible` uncontended | ~20 ns |
| `copy_from_user` 64 bytes | ~200 ns |
| `buffer_data_len += 64` | ~5 ns |
| `wake_up_interruptible` (sets reader TASK_RUNNING) | ~500 ns |
| `mutex_unlock` | ~20 ns |
| `write()` syscall return | ~100 ns |
| Scheduler context switch to reader | ~3–50 µs (scheduler quantum + cache warmup) |
| Reader `mutex_lock` (may briefly contend if writer not yet unlocked) | ~20–500 ns |
| `copy_to_user` 64 bytes | ~200 ns |
| `memmove` (0 bytes if fully consumed) | ~0 ns |
| `mutex_unlock` in reader | ~20 ns |
| `read()` syscall return | ~100 ns |
| **Total write-to-read latency** | **~4–55 µs** |
The dominant variable is the scheduler context switch (3–50 µs). On a non-preemptive desktop kernel with 4ms HZ, worst-case latency is ~4ms. With a `PREEMPT` kernel, worst case drops to ~100 µs.

![Stress Test Architecture: 4 Writers + 4 Readers + Checksum Verification](./diagrams/tdd-diag-29.svg)

---
## 10. Concurrency Specification
### 10.1 Lock Ordering
There is exactly one lock in this driver: `mydevice_mutex`. There are no nested lock acquisitions. Lock ordering analysis: N/A (single lock — deadlock from lock ordering is impossible).
The only deadlock risk is the **mutex-while-sleeping** scenario: calling `wait_event_interruptible` while holding `mydevice_mutex`. This is prevented by the explicit ordering in all handlers:
```
CORRECT order:
    wait_event_interruptible(queue, condition)   ← outside mutex
    mutex_lock_interruptible(&mydevice_mutex)    ← acquire after waking
    ... critical section ...
    wake_up_interruptible(other_queue)           ← inside critical section (safe)
    mutex_unlock(&mydevice_mutex)
WRONG order (DEADLOCK):
    mutex_lock_interruptible(&mydevice_mutex)
    wait_event_interruptible(queue, condition)   ← sleeping while holding mutex
    // DEADLOCK: writer cannot acquire mutex to add data → reader never wakes
```
### 10.2 What `mydevice_mutex` Protects
The mutex protects the **buffer invariant**: the tuple `(kernel_buffer, buffer_data_len, buffer_size)` is consistent when viewed outside the critical section. Specifically:
- `kernel_buffer` points to a valid, allocated slab block of `buffer_size` bytes
- `buffer_data_len <= buffer_size`
- bytes at indices `[0, buffer_data_len)` contain valid, unconsumed FIFO data
Operations that modify any element of this tuple must hold the mutex for the entire modification. Operations that only read the tuple for atomic decisions (like `poll`) may read without the mutex, accepting the risk of a momentarily stale snapshot.
### 10.3 What the Mutex Does NOT Protect
- `open_count`, `read_count`, `write_count`: `atomic_t`, no mutex needed
- `dev_num`, `my_cdev`, `mydevice_class`, `mydevice_device`, `proc_entry`: write-once in `mydevice_init`, read-only thereafter
- `buffer_data_len` reads in `mydevice_poll`: deliberately unprotected; see Section 4.3
### 10.4 Wait Queue Invariants
**`mydevice_read_queue` semantics:**
- A process sleeps on this queue when `buffer_data_len == 0` (and O_NONBLOCK is not set).
- `wake_up_interruptible(&mydevice_read_queue)` is called by:
  - `mydevice_write` after incrementing `buffer_data_len`
  - `mydevice_ioctl_resize` after the buffer swap (resize may preserve data)
- Any process woken from this queue must recheck `buffer_data_len > 0` under the mutex. The condition may be false due to thundering herd.
**`mydevice_write_queue` semantics:**
- A process sleeps on this queue when `buffer_data_len >= buffer_size` (and O_NONBLOCK is not set).
- `wake_up_interruptible(&mydevice_write_queue)` is called by:
  - `mydevice_read` after decrementing `buffer_data_len`
  - `mydevice_ioctl_clear` after setting `buffer_data_len = 0`
  - `mydevice_ioctl_resize` after the buffer swap (larger buffer = more space)
- Any process woken from this queue must recheck `buffer_data_len < buffer_size` under the mutex.
### 10.5 Thundering Herd Analysis
With 4 readers sleeping and 1 writer writing 64 bytes:
1. Writer acquires mutex, appends 64 bytes, calls `wake_up_interruptible(&mydevice_read_queue)`.
2. All 4 readers transition from `TASK_INTERRUPTIBLE` to `TASK_RUNNING`.
3. Scheduler runs reader 0 first. Reader 0 acquires mutex, reads 64 bytes, releases mutex.
4. Readers 1, 2, 3 each acquire mutex, find `buffer_data_len == 0`, release mutex, return `-EAGAIN`.
5. Readers 1, 2, 3 retry (caller's loop) and re-enter `wait_event_interruptible`.
The cost: 3 extra mutex lock/unlock cycles (~180 ns each) and 3 extra `return -EAGAIN` calls. For 4 readers and typical write patterns, this is negligible. For 1000 readers, `wake_up_interruptible` would use `wake_up_interruptible_nr(queue, 1)` to wake only the first reader — not implemented in this driver.
### 10.6 Memory Ordering Guarantees
The kernel's mutex implementation includes full memory barriers (`smp_mb()`) at lock and unlock boundaries. This guarantees:
- All writes to `kernel_buffer`, `buffer_data_len`, `buffer_size` done inside the critical section are **visible to all CPUs** before `mutex_unlock` returns.
- Any CPU that acquires the mutex after a `mutex_unlock` will see all updates made under the lock.
`wake_up_interruptible` includes an implicit `smp_mb__after_spinlock()` that ensures the woken reader sees the written data. This is why the reader does not need an explicit memory barrier before reading `buffer_data_len` after waking — the wake_up/wake sequence provides the required ordering.

![Cache Line Behavior: Driver Globals Layout and False Sharing Analysis](./diagrams/tdd-diag-30.svg)

---
## 11. Common Pitfalls Reference
**Pitfall 1: Sleeping while holding the mutex.**
Symptoms: System hangs. No output from blocked processes. `cat /proc/lockdep` (if enabled) shows the deadlock. All I/O to the device freezes.
Fix: `wait_event_interruptible` must be called BEFORE `mutex_lock_interruptible`. Check all read/write handlers — the wait precedes the lock.
**Pitfall 2: Not freeing `new_buf` on `mutex_lock_interruptible` failure in RESIZE.**
Symptoms: Kernel memory leak. `/proc/slabinfo` shows `kmalloc-*` count growing with each signal-interrupted resize. Eventually OOM.
Fix: `if (mutex_lock_interruptible(&mydevice_mutex)) { kfree(new_buf); return -ERESTARTSYS; }` — the `kfree` before the `return` is mandatory.
**Pitfall 3: Returning 0 or positive from `-ERESTARTSYS` path.**
Symptoms: Ctrl+C on a blocked reader appears to work (signal handler runs) but the `read()` call returned 0, confusing the calling program (e.g., `cat` prints nothing and exits thinking the file ended).
Fix: Any code path that receives `-ERESTARTSYS` from `wait_event_interruptible` or `mutex_lock_interruptible` must `return -ERESTARTSYS` immediately with no other action.
**Pitfall 4: Calling `poll_wait` but not returning the mask (or vice versa).**
Symptoms:
- Only `poll_wait`, no mask: `select()` subscribes correctly, gets woken up, calls `.poll` again, sees no mask returned, immediately falls back to sleep, loops indefinitely.
- Only mask, no `poll_wait`: first `poll()` call returns correctly (if data is available), but if fd is not ready, `select()` sleeps forever — no wakeup subscription was made.
Fix: Both `poll_wait()` calls AND the mask return are required in every `.poll` invocation.
**Pitfall 5: Returning POLLIN when buffer is empty.**
Symptoms: Tight busy-loop in userspace epoll. `top` shows 100% CPU in the polling process.
Fix: `if (buffer_data_len > 0) mask |= POLLIN | POLLRDNORM;` — the condition must be checked. Do not unconditionally set `POLLIN`.
**Pitfall 6: Not calling `wake_up_interruptible` after modifying buffer state.**
Symptoms: Processes sleep in `wait_event_interruptible` forever, even after data is added (read) or consumed (write). `ps aux` shows `D` state (uninterruptible sleep) — or more commonly, `S` state (interruptible sleep) that just never wakes.
Fix: Every code path that changes `buffer_data_len` must call the appropriate `wake_up_interruptible`. Read changes buffer_data_len → wake writers. Write changes buffer_data_len → wake readers. CLEAR changes buffer_data_len → wake writers. RESIZE changes buffer_size and buffer_data_len → wake both.
**Pitfall 7: Using `mutex_lock` instead of `mutex_lock_interruptible`.**
Symptoms: Processes blocked waiting for the mutex cannot be killed. SIGKILL is ignored. Process stays in `D` state. The machine may require a reboot if many such processes accumulate.
Fix: Always use `mutex_lock_interruptible` in file operation handlers. The only legitimate use of `mutex_lock` (uninterruptible) is in contexts where signals cannot be delivered (workqueues, specific kernel threads), which does not apply to this driver.
**Pitfall 8: Updating `buffer_data_len` before `copy_from_user` succeeds.**
Symptoms: A partially-written or garbage region of `kernel_buffer` is visible to readers.
Fix: Call `copy_from_user` into `kernel_buffer + buffer_data_len` first; only increment `buffer_data_len` if `not_copied == 0`.
**Pitfall 9: `memmove` with `memcpy`.**
Symptoms: Occasional data corruption when the consumed region is small relative to the remaining data (overlapping source/destination). Non-deterministic; hard to reproduce.
Fix: Always use `memmove` for the FIFO shift. The compiler may optimize `memmove` to `memcpy` when it can prove non-overlap, so there is no performance cost to using `memmove` unconditionally.
**Pitfall 10: Not testing with `buffer_size=` smaller than `MESSAGE_SIZE` in stress test.**
Symptoms: Test passes because no writer ever fills the buffer to capacity, so the blocking-write code path is never exercised.
Fix: `sudo insmod mydevice.ko buffer_size=64` (smaller than `MESSAGE_SIZE=64*NUM_WRITERS`) and re-run `concurrent_test.py`. All code paths (blocking read, blocking write, EAGAIN, wakeup) should be exercised.
---
<!-- END_TDD_MOD -->


# Project Structure: Linux Kernel Module
## Directory Tree
```
linux-kernel-module/
├── hello/                              # M1: Hello World Module
│   ├── Makefile                        # M1: Kbuild delegation makefile
│   ├── hello.c                         # M1: Module source (init/exit/param)
│   └── verify.sh                       # M1: Acceptance test script
│
└── mydevice/                           # M2–M4: Character Device Driver
    ├── Makefile                        # M2–M4: Kbuild makefile (ccflags-y := -I$(PWD))
    ├── mydevice.h                      # M3–M4: Shared kernel/userspace ioctl header
    ├── mydevice.c                      # M2–M4: Driver source (all features accumulate here)
    ├── test_mydevice.c                 # M3: Userspace ioctl test program (C)
    ├── concurrent_test.py              # M4: Checksum-verified concurrent stress test
    ├── verify.sh                       # M2: Acceptance test script (M2 features)
    └── verify_m4.sh                    # M4: Acceptance test script (M4 features)
```
### Build Artifacts (produced by `make`, do not create manually)
```
hello/
├── hello.ko                            # M1: Loadable kernel object (primary output)
├── hello.o                             # M1: Intermediate compiled object
├── hello.mod.c                         # M1: Kbuild-generated module glue
├── hello.mod.o                         # M1: Compiled glue
├── Module.symvers                      # M1: Symbol export table
└── modules.order                       # M1: Kbuild tracking file
mydevice/
├── mydevice.ko                         # M2–M4: Loadable kernel object (primary output)
├── mydevice.o                          # M2–M4: Intermediate compiled object
├── mydevice.mod.c                      # M2–M4: Kbuild-generated module glue
├── mydevice.mod.o                      # M2–M4: Compiled glue
├── Module.symvers                      # M2–M4: Symbol export table
├── modules.order                       # M2–M4: Kbuild tracking file
└── test_mydevice                       # M3: Compiled userspace test binary (gcc output)
```
---
## Creation Order
1. **Project Setup** (15 min)
   - Create the two working directories
   ```bash
   mkdir -p linux-kernel-module/hello
   mkdir -p linux-kernel-module/mydevice
   ```
2. **M1 — Kbuild Makefile** (30 min)
   - `hello/Makefile` — `obj-m += hello.o`, `KDIR`, `all`/`clean` targets with TAB-indented recipes
3. **M1 — Module Source** (30–60 min)
   - `hello/hello.c` — `MODULE_LICENSE("GPL")`, metadata macros, `buffer_size` param, `hello_init`, `hello_exit`
4. **M1 — Verification Script** (30 min)
   - `hello/verify.sh` — tests metadata, default load, custom param, sysfs permissions, invalid param rejection
5. **M2 — Kbuild Makefile** (15 min)
   - `mydevice/Makefile` — same pattern as M1, add `ccflags-y := -I$(PWD)` for future header inclusion
6. **M2 — Driver Source (skeleton → full)** (3–5 hours, incremental)
   - `mydevice/mydevice.c` — build in phases:
     - Phase 1: globals, `file_operations` stubs, `alloc_chrdev_region` + `cdev_init`/`cdev_add`
     - Phase 2: `class_create` + `device_create` → `/dev/mydevice` via udev
     - Phase 3: `mydevice_open`/`mydevice_release` with `atomic_t open_count`, `try_module_get`
     - Phase 4: `mydevice_write` (`copy_from_user`) + `mydevice_read` (`copy_to_user`, `f_pos`, EOF)
     - Phase 5: complete goto error-unwind chain, `mydevice_exit` reverse cleanup
7. **M2 — Verification Script** (60–90 min)
   - `mydevice/verify.sh` — tests `/dev/` node creation, open/release lifecycle, write/read round-trip, EOF, buffer clamping, binary data, `module_put` refcount, custom `buffer_size`
8. **M3 — Shared Header** (30 min)
   - `mydevice/mydevice.h` — `MYDEVICE_IOC_MAGIC`, `struct mydevice_status`, `MYDEVICE_IOC_RESIZE`/`CLEAR`/`STATUS` macros, include guard
9. **M3 — Driver Source Updates** (2–3 hours, incremental)
   - Update `mydevice/mydevice.c`:
     - Add `#include "mydevice.h"`, `read_count`/`write_count` atomics
     - Add `atomic_inc` calls in `mydevice_read` and `mydevice_write`
     - Add `.unlocked_ioctl = mydevice_ioctl` to `mydevice_fops`
     - Implement `mydevice_ioctl` (3-layer validation + switch)
     - Implement `mydevice_ioctl_resize`, `mydevice_ioctl_clear`, `mydevice_ioctl_status`
     - Add `#include <linux/proc_fs.h>` + `<linux/seq_file.h>`
     - Implement `mydevice_proc_show`, `mydevice_proc_open`, `mydevice_proc_ops`
     - Add `proc_create` to `mydevice_init` (after `device_create`)
     - Add `proc_remove` as first line of `mydevice_exit`
10. **M3 — Userspace Test Program** (60–90 min)
    - `mydevice/test_mydevice.c` — exercises STATUS/CLEAR/RESIZE ioctls, wrong magic/nr validation, `/proc/mydevice` field check
    - Build: `gcc -Wall -Wextra -I. -o test_mydevice test_mydevice.c`
11. **M4 — Driver Source Updates** (3–5 hours, incremental)
    - Update `mydevice/mydevice.c`:
      - Add `#include <linux/mutex.h>`, `<linux/wait.h>`, `<linux/poll.h>`
      - Add `DEFINE_MUTEX(mydevice_mutex)`, `DECLARE_WAIT_QUEUE_HEAD` for read and write queues
      - Change buffer model to FIFO append (remove `memset`-before-write, remove `*f_pos` tracking)
      - Wrap `mydevice_read` in mutex + `wait_event_interruptible` + `O_NONBLOCK` + `memmove` consume
      - Wrap `mydevice_write` in mutex + `wait_event_interruptible` + `O_NONBLOCK` + append
      - Add `wake_up_interruptible` calls in read (wake writers) and write (wake readers)
      - Add `mydevice_poll` with `poll_wait` + readiness mask
      - Register `.poll = mydevice_poll` in `mydevice_fops`
      - Wrap `mydevice_ioctl_resize` and `mydevice_ioctl_clear` in mutex
12. **M4 — Concurrent Stress Test** (60–90 min)
    - `mydevice/concurrent_test.py` — 4 writers × 4 readers, SHA-256 checksum verification, data integrity proof
13. **M4 — Verification Script** (60 min)
    - `mydevice/verify_m4.sh` — build check, stress test, blocking wakeup, SIGINT termination, O_NONBLOCK EAGAIN, poll readiness mask, poll wakeup latency, no kernel oops
---
## File Count Summary
| Category | Count |
|---|---|
| Source files (hand-written) | 8 |
| Scripts | 3 |
| Directories | 3 (root + `hello/` + `mydevice/`) |
| Build artifacts (generated) | 12 |
| **Total hand-written files** | **11** |
**Estimated lines of code (hand-written only):**
| File | ~Lines |
|---|---|
| `hello/Makefile` | 12 |
| `hello/hello.c` | 50 |
| `hello/verify.sh` | 110 |
| `mydevice/Makefile` | 13 |
| `mydevice/mydevice.h` | 40 |
| `mydevice/mydevice.c` (M4 final) | 380 |
| `mydevice/test_mydevice.c` | 160 |
| `mydevice/concurrent_test.py` | 130 |
| `mydevice/verify.sh` | 120 |
| `mydevice/verify_m4.sh` | 100 |
| **Total** | **~1,115** |
# AUDIT & FIX: sandbox

## CRITIQUE
- **Critical Ordering Flaw**: Milestone 1 mentions user namespace mapping in deliverables but doesn't list CLONE_NEWUSER in acceptance criteria. User namespaces are the ONLY way to run unprivileged containers, and their absence means the entire project implicitly requires root, which contradicts modern container design.
- **User Namespace Ordering**: CLONE_NEWUSER must be created FIRST (or simultaneously via clone()) because it grants the fake root (UID 0) inside the namespace needed to create other namespaces without real root. The project doesn't address this critical sequencing.
- **Cgroups v1 vs v2 Ambiguity**: The project vaguely says 'cgroups' without specifying version. Cgroups v2 (unified hierarchy) is now default on modern kernels (5.8+). v1 uses per-controller hierarchies. The filesystem paths, interfaces, and delegation models are fundamentally different. Students will waste hours on the wrong version.
- **Missing cgroup delegation for unprivileged use**: If user namespaces are used, cgroup v2 delegation (via /sys/fs/cgroup subtree) must be configured. This is not mentioned.
- **Milestone 5 (Capability Dropping) should come BEFORE seccomp**: Capabilities should be dropped before installing seccomp filters because seccomp filters are irrevocable and the order of privilege reduction matters. The current ordering (seccomp at M3, caps at M5) is suboptimal for defense-in-depth.
- **Missing Milestone**: No integration/combined sandbox milestone. The milestones are isolated; there's no acceptance criteria for combining ALL layers into a working sandbox.
- **Pitfall quality is shallow**: 'Forgetting SIGCHLD flag' is not actionable. Which call? clone() requires SIGCHLD or the parent won't be notified of child death via wait(). 'Need root/CAP_SYS_ADMIN' is not a pitfall—it's a prerequisite that should be solved by user namespaces.
- **pivot_root vs chroot**: The project mentions both but doesn't emphasize that chroot is trivially escapable (chroot + chdir + chroot again) and pivot_root is the only secure option. This is a critical security distinction.
- **No mention of seccomp SECCOMP_RET_LOG or SECCOMP_RET_TRACE**: Only SECCOMP_RET_KILL is mentioned. For debugging, SECCOMP_RET_LOG is essential.
- **Missing /proc/pid/ns verification**: Acceptance criteria should require verifying namespace isolation via /proc/self/ns/ symlinks, not just /proc entries.

## FIXED YAML
```yaml
id: sandbox
name: Process Sandbox
description: >-
  Kernel-level process isolation using Linux namespaces, seccomp-BPF,
  capability dropping, and cgroups v2 resource controls.
difficulty: intermediate
estimated_hours: "30-45"
essence: >-
  Kernel-level process isolation through coordinated application of user
  namespace unprivileged containment, PID/mount/network/UTS namespace resource
  virtualization, Berkeley Packet Filter system call interception,
  capability-based privilege reduction, and cgroups v2 resource quotas to
  create unprivileged execution environments that prevent system compromise.
why_important: >-
  Building this teaches you the foundational security mechanisms underlying
  containers, sandboxes, and virtualization technologies used in production
  systems like Docker, Kubernetes, and browser sandboxes—skills critical for
  security engineering and systems programming roles.
learning_outcomes:
  - Create user namespaces to enable unprivileged container creation without real root
  - Implement namespace isolation for PID, network, mount, and UTS resource virtualization
  - Build a minimal root filesystem using pivot_root for secure filesystem isolation
  - Drop Linux capabilities systematically to achieve least-privilege execution
  - Design and apply seccomp-BPF filters to whitelist safe system calls and block dangerous operations
  - Configure cgroups v2 controllers to enforce hard limits on CPU time, memory usage, and I/O bandwidth
  - Combine multiple isolation layers in correct order to implement defense-in-depth security architecture
  - Debug permission denied errors and capability requirements in restricted environments
skills:
  - Linux User Namespaces
  - Linux Namespaces (PID, Mount, Network, UTS)
  - Seccomp-BPF Filtering
  - Cgroups v2 Resource Control
  - Capability Management
  - Filesystem Isolation (pivot_root)
  - System Call Analysis
  - Defense in Depth
  - Privilege Separation
tags:
  - c
  - go
  - intermediate
  - isolation
  - rust
  - seccomp
  - security
  - syscall-filtering
  - namespaces
  - cgroups-v2
architecture_doc: architecture-docs/sandbox/index.md
languages:
  recommended:
    - C
    - Rust
    - Go
  also_possible:
    - Python (with ctypes)
resources:
  - name: Linux Namespaces
    url: https://man7.org/linux/man-pages/man7/namespaces.7.html
    type: documentation
  - name: Seccomp BPF
    url: https://www.kernel.org/doc/html/latest/userspace-api/seccomp_filter.html
    type: documentation
  - name: User Namespaces
    url: https://man7.org/linux/man-pages/man7/user_namespaces.7.html
    type: documentation
  - name: Cgroups v2
    url: https://docs.kernel.org/admin-guide/cgroup-v2.html
    type: documentation
prerequisites:
  - type: skill
    name: Linux system calls (clone, unshare, mount, pivot_root)
  - type: skill
    name: Process management (fork, exec, wait)
  - type: skill
    name: Basic understanding of Linux security model
  - type: skill
    name: C programming
milestones:
  - id: sandbox-m1
    name: User Namespace & Process Namespaces
    description: >-
      Create a user namespace FIRST to enable unprivileged namespace creation,
      then create PID, mount, network, and UTS namespaces for full isolation.
    acceptance_criteria:
      - Create new user namespace via clone(CLONE_NEWUSER) or unshare(CLONE_NEWUSER) as the FIRST namespace operation
      - Write uid_map and gid_map to map host UID/GID to root (UID 0) inside the user namespace
      - Write 'deny' to /proc/self/setgroups before writing gid_map (required by kernel since Linux 3.19)
      - Create new PID namespace where sandboxed process sees itself as PID 1 via /proc/self/status
      - Create new mount namespace providing isolated filesystem view independent of host mounts
      - Create new network namespace with only loopback interface (verified via ip link show)
      - Create new UTS namespace with isolated hostname (verified via hostname command inside sandbox)
      - Verify all namespace isolation by comparing /proc/self/ns/ symlink inodes between host and sandbox
      - Demonstrate that the entire sandbox setup works WITHOUT real root privileges (unprivileged user)
    pitfalls:
      - User namespace MUST be created first; it provides fake CAP_SYS_ADMIN needed for other namespaces
      - Forgetting SIGCHLD flag in clone() causes parent wait() to never return for child termination
      - Must mount /proc inside the new PID namespace or ps/top will show host processes
      - uid_map write fails if /proc/self/setgroups is not set to 'deny' first (security restriction)
      - Some distributions disable unprivileged user namespaces via sysctl kernel.unprivileged_userns_clone=0
      - Network namespace starts with NO interfaces; even loopback must be explicitly brought up
    concepts:
      - User namespace UID/GID mapping
      - Linux namespace creation ordering
      - clone() vs unshare() tradeoffs
      - /proc/self/ns/ namespace identification
    skills:
      - System programming in C/Rust/Go
      - Linux namespace APIs (clone, unshare, setns)
      - UID/GID mapping for user namespaces
      - Process isolation and containerization fundamentals
      - Debugging isolated processes with strace/nsenter
    deliverables:
      - User namespace creation with UID/GID mapping to provide fake root inside sandbox
      - PID namespace isolation giving sandboxed process its own PID 1
      - Network namespace isolation with only loopback interface
      - Mount namespace setup providing isolated filesystem view
      - UTS namespace isolation with custom hostname
      - Verification script comparing /proc/self/ns/ inodes between host and sandbox
    estimated_hours: "6-9"

  - id: sandbox-m2
    name: Filesystem Isolation with pivot_root
    description: >-
      Create a minimal root filesystem and use pivot_root (NOT chroot) to
      securely change the root directory, preventing container escape.
    acceptance_criteria:
      - Create minimal root filesystem containing only required binaries, libraries, and /etc files
      - Use ldd or equivalent to identify and copy all shared library dependencies into the rootfs
      - Use pivot_root (not chroot) to atomically swap root and old root mount points
      - Unmount old root after pivot_root to prevent access to host filesystem
      - Mount /proc as type proc inside sandbox (required for PID namespace to work correctly)
      - Mount minimal /dev with only null, zero, urandom, and tty via mknod or bind mount from devtmpfs
      - Mount /sys read-only or not at all to prevent sysfs-based information leaks
      - Make root filesystem read-only via mount with MS_RDONLY flag
      - Provide writable /tmp via tmpfs mount with size limit (e.g., 64MB)
      - Verify parent host filesystem is completely inaccessible from sandbox (ls / shows only sandbox contents)
      - Verify that /proc/1/root inside sandbox does NOT expose host filesystem
    pitfalls:
      - chroot is trivially escapable (chroot + fchdir to saved fd + chroot again); NEVER use chroot for security
      - pivot_root requires new root and old root to be on different mount points; use bind mount to self first
      - Forgetting to unmount old root after pivot_root leaves entire host filesystem accessible
      - Missing /dev/null or /dev/urandom causes many programs to crash or hang
      - Library dependencies (glibc NSS plugins, locale files) are easily missed by ldd
      - Static linking avoids library dependency issues but increases binary size significantly
      - MS_REC flag needed with MS_PRIVATE to prevent mount event propagation to host
    concepts:
      - pivot_root vs chroot security model
      - Mount propagation (shared, private, slave)
      - Minimal rootfs construction
      - Bind mounts and tmpfs
    skills:
      - Filesystem operations (mount, pivot_root, umount)
      - Building minimal container rootfs
      - Managing mount propagation types
      - Library dependency resolution (ldd, static linking)
    deliverables:
      - Minimal rootfs builder script that copies binaries and resolved library dependencies
      - pivot_root implementation atomically swapping filesystem root
      - Old root unmount ensuring host filesystem is completely removed from namespace
      - Read-only root filesystem preventing writes to system directories
      - Tmpfs mounts providing writable scratch space at /tmp with configurable size limit
      - Minimal /dev setup with only essential device nodes
    estimated_hours: "5-8"

  - id: sandbox-m3
    name: Capability Dropping & Privilege Reduction
    description: >-
      Drop Linux capabilities to the minimum required set and lock down
      privilege escalation paths BEFORE applying seccomp filters.
    acceptance_criteria:
      - Enumerate all current process capabilities (effective, permitted, inheritable, ambient, bounding) before dropping
      - Drop ALL capabilities from the bounding set except a configurable minimum required set
      - Clear all ambient capabilities to prevent them from being inherited across execve()
      - Set PR_SET_NO_NEW_PRIVS via prctl() to block future privilege escalation through setuid/setgid binaries
      - Verify capabilities are correctly dropped by reading /proc/self/status CapEff, CapPrm, CapBnd lines
      - Confirm privileged operations fail after dropping: mount() returns EPERM, network interface config fails
      - Demonstrate that setuid binaries inside sandbox do NOT gain elevated capabilities
    pitfalls:
      - Capability dropping order matters: drop bounding set first, then clear ambient, then drop permitted/effective
      - If you need to call setgid()/setuid() to switch to unprivileged user, do it BEFORE dropping CAP_SETUID/CAP_SETGID
      - Ambient capabilities survive execve() and can re-grant dropped caps; MUST be explicitly cleared
      - Some basic operations need capabilities: e.g., mounting proc needs CAP_SYS_ADMIN inside user namespace
      - PR_SET_NO_NEW_PRIVS is irreversible and inherited by all children; set it at the right time
      - capget()/capset() use a versioned header; using wrong version silently fails on some kernels
    concepts:
      - Linux capability sets (effective, permitted, inheritable, ambient, bounding)
      - Principle of least privilege
      - Ambient capability inheritance model
      - PR_SET_NO_NEW_PRIVS flag
    skills:
      - Linux capability system (capset, capget, prctl)
      - Privilege separation techniques
      - Secure process initialization ordering
      - Reading /proc/self/status for capability verification
    deliverables:
      - Capability enumeration utility reading and displaying all five capability sets
      - Bounding set restriction dropping all non-essential capabilities
      - Ambient capability clearing preventing inheritance across execve()
      - No-new-privileges flag application blocking setuid escalation
      - Verification utility confirming all capability sets match expected minimal configuration
      - Integration test proving privileged operations (mount, network config, raw sockets) fail
    estimated_hours: "4-6"

  - id: sandbox-m4
    name: Seccomp-BPF System Call Filtering
    description: >-
      Design and apply seccomp-BPF filters to whitelist only the system calls
      needed by the sandboxed workload, killing the process on violation.
    acceptance_criteria:
      - Create seccomp filter using BPF program syntax (struct sock_filter array or libseccomp)
      - Implement WHITELIST approach allowing only explicitly listed system calls (default deny)
      - Default action is SECCOMP_RET_KILL_PROCESS for any system call not in the whitelist
      - Filter checks architecture field (AUDIT_ARCH_X86_64) to prevent architecture-confusion attacks on x86-64
      - Whitelist includes all system calls required by the target workload (use strace to enumerate)
      - Implement argument-level filtering for sensitive calls (e.g., restrict open() flags, socket() domains)
      - Verify PR_SET_NO_NEW_PRIVS is set BEFORE installing seccomp filter (kernel requirement)
      - Test that blocked system calls (ptrace, mount, reboot, kexec_load) terminate the process immediately
      - Support SECCOMP_RET_LOG mode for debugging to log violations without killing
    pitfalls:
      - MUST set PR_SET_NO_NEW_PRIVS before prctl(PR_SET_SECCOMP) or the call fails with EACCES
      - glibc internally uses many syscalls (e.g., futex, clock_gettime, mprotect); missing them causes mysterious crashes
      - System call numbers differ between architectures (x86-64 vs i386 vs ARM64); always check AUDIT_ARCH
      - x86-64 kernels allow 32-bit syscalls via int 0x80; filter MUST check architecture to prevent bypass
      - Seccomp filters are irrevocable once installed; cannot add more allowed syscalls later
      - BPF programs have a maximum instruction count (4096); complex filters may need restructuring
      - SECCOMP_RET_KILL kills the thread, not the process; use SECCOMP_RET_KILL_PROCESS (Linux 4.14+)
    concepts:
      - Seccomp-BPF filter programming
      - System call whitelisting vs blacklisting (whitelist is the only secure approach)
      - Architecture-confusion attacks
      - BPF instruction set and program limits
    skills:
      - Berkeley Packet Filter (BPF) programming
      - System call enumeration and tracing (strace)
      - Seccomp filter design, testing, and debugging
      - Architecture-aware security filtering
    deliverables:
      - Seccomp-BPF filter program implementing default-deny whitelist policy
      - Architecture check preventing 32-bit syscall bypass on 64-bit kernels
      - System call whitelist generated from strace analysis of target workload
      - Argument-level filtering restricting parameters of sensitive allowed calls
      - Debug mode using SECCOMP_RET_LOG to identify missing required syscalls
      - Integration test proving dangerous syscalls (ptrace, mount, reboot) cause process termination
    estimated_hours: "6-10"

  - id: sandbox-m5
    name: Resource Limits with Cgroups v2
    description: >-
      Use cgroups v2 (unified hierarchy) to limit CPU, memory, PID count,
      and I/O resources for sandboxed processes.
    acceptance_criteria:
      - Verify system is running cgroups v2 by checking /sys/fs/cgroup/cgroup.controllers exists
      - Create dedicated cgroup subtree under a delegated directory for the sandboxed process
      - Enable required controllers (cpu, memory, io, pids) by writing to cgroup.subtree_control
      - Set memory.max to configurable threshold (e.g., 67108864 for 64MB) and verify OOM kill triggers
      - Set cpu.max to configurable quota (e.g., '10000 100000' for 10% of one core) and verify throttling
      - Set pids.max to limit maximum number of processes/threads (e.g., 64) and verify fork bomb protection
      - Set io.max to limit disk throughput (e.g., 'MAJ:MIN rbps=1048576 wbps=1048576' for 1MB/s)
      - Write sandboxed process PID to cgroup.procs to place it under resource control
      - Verify resource limits by running stress tests that exceed each configured limit
      - Clean up cgroup subtree on sandbox exit (rmdir the cgroup directory after all processes exit)
    pitfalls:
      - Cgroups v1 (per-controller hierarchy) and v2 (unified hierarchy) have completely different APIs; this project targets v2 ONLY
      - Controllers must be enabled in PARENT cgroup's subtree_control before they appear in child
      - Cgroup delegation for unprivileged users requires specific ownership/permissions on cgroup directory
      - memory.max=0 doesn't mean unlimited; use 'max' string for no limit
      - io.max requires device major:minor numbers, not device paths
      - Cannot delete a cgroup directory while it still has processes; must wait or migrate them first
      - cpu.max format is 'QUOTA PERIOD' in microseconds; getting the format wrong silently fails
      - Some cloud/container environments don't expose all cgroup controllers
    concepts:
      - Cgroups v2 unified hierarchy
      - Controller enablement and delegation
      - Resource limit enforcement (memory OOM, CPU throttling, PID limits)
      - Cgroup lifecycle management
    skills:
      - Cgroups v2 filesystem interface
      - Resource limit configuration and tuning
      - CPU quota and memory limit enforcement
      - Cgroup delegation for unprivileged use
    deliverables:
      - Cgroup v2 subtree creation with required controller enablement
      - Memory limit (memory.max) restricting maximum resident memory with OOM kill verification
      - CPU limit (cpu.max) throttling processor time with measurable performance impact
      - PID limit (pids.max) capping maximum processes to prevent fork bombs
      - I/O bandwidth limit (io.max) throttling disk throughput per device
      - Cgroup cleanup logic removing subtree after sandbox process exits
    estimated_hours: "5-8"

  - id: sandbox-m6
    name: Integrated Sandbox & Defense in Depth
    description: >-
      Combine all isolation layers into a single unified sandbox executor
      with correct ordering and verify the combined security posture.
    acceptance_criteria:
      - Sandbox executor applies all isolation layers in correct security order: user namespace → other namespaces → pivot_root → capability drop → no_new_privs → seccomp → cgroups → exec target
      - Single command-line interface accepts target binary path, resource limits, and allowed syscall list as arguments
      - Sandboxed process cannot access host filesystem, host network, host PID tree, or host hostname
      - Sandboxed process cannot escalate privileges via any mechanism (setuid, capabilities, ptrace)
      - Sandboxed process is killed when exceeding any configured resource limit (memory, CPU, PIDs)
      - Sandboxed process is killed immediately on any system call not in the whitelist
      - Run a realistic workload (e.g., compile a small C program, run a Python script) inside the sandbox successfully
      - Demonstrate defense-in-depth by showing that disabling any single layer does NOT grant full escape
      - Log all security-relevant events (seccomp violations, OOM kills, capability denials) to stderr
    pitfalls:
      - Ordering is critical: seccomp must come AFTER capability drop and no_new_privs, but BEFORE exec
      - The child process after clone() runs in a partially-initialized environment; minimize work before full setup
      - Error handling in the child (after clone, before exec) is tricky; use a pipe to communicate errors to parent
      - Some workloads require capabilities you've dropped or syscalls you've blocked; iterative tuning is required
      - Testing combined isolation requires deliberate escape attempts, not just positive tests
    concepts:
      - Defense in depth security architecture
      - Security layer ordering and composition
      - Privilege reduction lifecycle
      - Container runtime design patterns
    skills:
      - Systems integration and orchestration
      - Security testing and escape attempt verification
      - Error handling in forked child processes
      - Command-line interface design
    deliverables:
      - Unified sandbox executor combining all isolation layers with correct ordering
      - CLI interface accepting target binary, resource limits, and syscall whitelist configuration
      - Security verification test suite attempting escape via filesystem, network, privilege escalation, and resource abuse
      - Event logging for security violations and resource limit enforcement
      - Documentation of isolation layer ordering rationale and security properties
    estimated_hours: "5-7"
```
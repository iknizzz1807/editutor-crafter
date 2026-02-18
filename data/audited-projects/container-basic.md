# AUDIT & FIX: container-basic

## CRITIQUE
- CRITICAL: Missing User Namespace milestone. User namespaces are essential for rootless containers, a major security feature in modern container runtimes (Podman, rootless Docker). Without this, the project only teaches the "run as root" path which is increasingly deprecated in production.
- Mount Namespace AC #3 says 'Pivot root switches container to new filesystem root directory' but fails to mention the prerequisite that the new root must itself be a mount point. pivot_root(2) requires 'new_root' to be a mount point; the standard pattern is `mount --bind newroot newroot` before calling pivot_root. Without this, pivot_root will fail with EINVAL.
- Milestone 2 doesn't mention unmounting the old root after pivot_root, which is critical — leaving the old root accessible defeats filesystem isolation entirely.
- No mention of UTS namespace (hostname isolation), which is trivial to implement and essential for container identity.
- No mention of IPC namespace, which is important for shared memory isolation.
- Cgroups milestone mentions 'v1 vs v2' in pitfalls but doesn't require the student to handle both or at least detect which is available. Modern systems (systemd-based) use cgroups v2 exclusively.
- The project claims 'advanced' difficulty but is 15-25 hours, which is very light for proper namespace + cgroup + networking implementation.
- Network namespace milestone doesn't mention DNS resolution (/etc/resolv.conf), which is always the first thing that breaks.
- No mention of seccomp filters or capability dropping, which are mentioned in learning outcomes ('security boundaries using namespace capabilities') but absent from milestones.
- PID namespace milestone pitfall 'Stack direction' is cryptic and misleading — the real issue with clone() is providing a properly allocated stack with the correct alignment, not 'direction' per se (though clone does require passing the top of the stack on x86).

## FIXED YAML
```yaml
id: container-basic
name: "Container (Basic)"
description: "Linux namespace isolation, cgroups resource limits, and rootless container support"
difficulty: advanced
estimated_hours: "25-40"
essence: >
  Kernel-level process and resource isolation using namespace syscalls (clone, unshare)
  for PID, mount, network, UTS, and user namespace separation, combined with cgroup
  controllers to partition CPU, memory, and process resources, and pivot_root for
  filesystem isolation — all without hardware virtualization.
why_important: >
  Building containers from scratch demystifies Docker/Kubernetes internals and teaches
  fundamental Linux kernel primitives used in production container orchestration,
  making you effective at debugging container issues, understanding security
  boundaries, and optimizing containerized deployments.
learning_outcomes:
  - Implement process isolation using PID namespaces with clone() or unshare() syscalls
  - Configure mount namespaces with bind-mount + pivot_root to isolate filesystem views
  - Create network namespaces and virtual ethernet pairs for container networking
  - Apply cgroup controllers to enforce CPU, memory, and process resource limits
  - Implement user namespace mapping for rootless (unprivileged) container execution
  - Debug namespace-related issues by understanding /proc filesystem mechanics
  - Build a minimal container runtime handling process lifecycle and cleanup
  - Understand container escape vulnerabilities and mitigation strategies
skills:
  - Linux Kernel APIs
  - System Programming
  - Process Isolation
  - Resource Management
  - Network Virtualization
  - Filesystem Manipulation
  - Low-Level Debugging
  - Container Security
tags:
  - advanced
  - c
  - cgroups
  - go
  - multi-tenancy
  - namespaces
  - rootfs
  - rust
  - user-namespaces
architecture_doc: architecture-docs/container-basic/index.md
languages:
  recommended:
    - C
    - Go
    - Rust
  also_possible:
    - Python
resources:
  - name: "Linux Namespaces"
    url: "https://man7.org/linux/man-pages/man7/namespaces.7.html"
    type: documentation
  - name: "Containers from Scratch"
    url: "https://ericchiang.github.io/post/containers-from-scratch/"
    type: article
  - name: "User Namespaces man page"
    url: "https://man7.org/linux/man-pages/man7/user_namespaces.7.html"
    type: documentation
  - name: "pivot_root(2) man page"
    url: "https://man7.org/linux/man-pages/man2/pivot_root.2.html"
    type: documentation
prerequisites:
  - type: skill
    name: "Linux system calls (clone, unshare, mount)"
  - type: skill
    name: "Process management (fork, exec, wait)"
  - type: skill
    name: "Filesystem basics (mount, chroot)"
milestones:
  - id: container-basic-m1
    name: "PID and UTS Namespace Isolation"
    description: >
      Isolate process tree with PID namespace and hostname with UTS namespace.
    acceptance_criteria:
      - Create new PID namespace using clone(CLONE_NEWPID) or unshare(CLONE_NEWPID); child process sees itself as PID 1
      - "Parent process observes the child's real host PID via the return value of clone() or waitpid()"
      - Container init (PID 1) properly reaps zombie child processes using waitpid(-1, ..., WNOHANG) in a loop
      - UTS namespace (CLONE_NEWUTS) isolates hostname; sethostname() inside container does not affect host
      - Verify isolation by reading /proc/self/status and comparing NSpid field from inside vs outside the namespace
    pitfalls:
      - "clone() requires a properly allocated and aligned stack passed as the child's stack pointer; on x86-64 pass the TOP of the stack (stack grows downward)"
      - PID 1 in a namespace has init responsibilities: it must reap orphaned children or they become zombies that can never be collected
      - "unshare(CLONE_NEWPID) affects children, not the calling process itself; the caller must fork() after unshare for PID 1 behavior"
      - "Forgetting CLONE_NEWUTS means hostname changes inside the container leak to the host"
    concepts:
      - PID namespaces
      - UTS namespaces
      - clone() and unshare() syscalls
      - Init process responsibilities
    skills:
      - System call programming
      - Process lifecycle management
      - Low-level Linux programming
      - Error handling for syscalls
    deliverables:
      - PID namespace creation using clone() or unshare() + fork()
      - UTS namespace creation with independent hostname
      - Process isolation verification comparing inner and outer PIDs
      - Init process (PID 1) zombie reaping loop
    estimated_hours: "3-5"

  - id: container-basic-m2
    name: "Mount Namespace and Filesystem Isolation"
    description: >
      Isolate filesystem mounts using mount namespace with pivot_root
      for complete root filesystem replacement.
    acceptance_criteria:
      - New mount namespace (CLONE_NEWNS) isolates mount/unmount operations from the host
      - Mount propagation is set to private (mount --make-rprivate /) preventing mount events from leaking to host
      - New root directory is bind-mounted onto itself (mount --bind newroot newroot) making it a mount point suitable for pivot_root
      - pivot_root() atomically swaps the root filesystem; old root is mounted at a subdirectory inside the new root
      - Old root is unmounted (umount2 with MNT_DETACH) after pivot_root, making host filesystem completely inaccessible
      - Essential pseudo-filesystems (/proc, /sys, /dev) are mounted inside the container with appropriate flags (e.g., proc with nosuid,noexec)
      - Container processes cannot access any host filesystem path after pivot_root + old root unmount
    pitfalls:
      - "pivot_root(2) requires new_root to be a mount point; without the bind-mount-to-self trick, it fails with EINVAL"
      - "Forgetting MS_REC | MS_PRIVATE on the root mount propagation allows mount events to leak bidirectionally"
      - "Not unmounting the old root after pivot_root leaves the entire host filesystem accessible from within the container"
      - "Mounting /proc without the PID namespace means the container sees host processes"
      - "Device nodes in /dev require special handling; bind-mount only the minimum set needed (null, zero, urandom)"
    concepts:
      - Mount namespaces and propagation types
      - pivot_root vs chroot (pivot_root is stronger)
      - Pseudo-filesystem mounting (/proc, /sys, /dev)
    skills:
      - Filesystem mount operations
      - Root filesystem configuration
      - System call debugging
      - Security-conscious mount flags
    deliverables:
      - Mount namespace creation with private propagation
      - Bind-mount-to-self for new root preparation
      - pivot_root() call swapping container root filesystem
      - Old root unmount ensuring host filesystem inaccessibility
      - /proc, /sys, /dev mounting inside the container namespace
      - Minimal rootfs directory structure verification
    estimated_hours: "4-6"

  - id: container-basic-m3
    name: "Network Namespace and Container Networking"
    description: >
      Isolate network stack and establish connectivity via veth pairs and a bridge.
    acceptance_criteria:
      - New network namespace (CLONE_NEWNET) provides isolated interfaces, routing table, and iptables rules
      - veth pair is created with one end in the container namespace and one end on the host
      - Host-side veth end is attached to a Linux bridge (e.g., ctr0); bridge has an IP address serving as the container gateway
      - Container-side veth has an IP address assigned and a default route pointing to the bridge IP
      - Loopback interface inside the container is brought up (ip link set lo up)
      - "Container can reach external networks via NAT (iptables MASQUERADE on the host's outbound interface)"
      - DNS resolution works inside the container via a bind-mounted or generated /etc/resolv.conf
    pitfalls:
      - "Creating the veth pair before the network namespace exists; the pair must be created then one end moved with 'ip link set <veth> netns <pid>'"
      - "Forgetting to bring up the loopback interface inside the container; many applications depend on localhost"
      - "NAT/MASQUERADE rules require IP forwarding enabled on the host (sysctl net.ipv4.ip_forward=1)"
      - "DNS resolution fails silently without /etc/resolv.conf inside the container; always configure it"
      - "Not cleaning up veth pairs and bridge on container exit, leaking network resources"
    concepts:
      - Network namespaces
      - Virtual ethernet (veth) pairs
      - Linux bridge networking
      - NAT and iptables
    skills:
      - Network interface configuration
      - Virtual device management
      - iptables rule management
      - Network troubleshooting
    deliverables:
      - Network namespace creation for isolated network stack
      - veth pair setup with one end moved into container namespace
      - Linux bridge configuration connecting container veth endpoints
      - IP address and route configuration on both ends
      - NAT/MASQUERADE rules for outbound internet access
      - DNS configuration inside the container
    estimated_hours: "5-7"

  - id: container-basic-m4
    name: "Cgroups Resource Limits"
    description: >
      Limit CPU, memory, and process count using cgroups v2 (with v1 fallback awareness).
    acceptance_criteria:
      - Detect whether the system uses cgroups v1 or v2 by checking /sys/fs/cgroup/cgroup.controllers existence
      - Create a cgroup for the container and write the container PID to cgroup.procs before exec
      - Set memory.max (v2) or memory.limit_in_bytes (v1) and verify OOM killer terminates the process when exceeded
      - "Set cpu.max (v2) with quota and period (e.g., '50000 100000' for 50% CPU) and verify CPU throttling under load"
      - Set pids.max to cap the number of processes; verify fork bomb is contained
      - Clean up the cgroup hierarchy (remove cgroup directory) on container exit to prevent resource leaks
      - Report resource usage (memory.current, cpu.stat) for observability
    pitfalls:
      - "cgroups v2 uses a unified hierarchy with different file names than v1; code must handle the active version"
      - "Writing to cgroup files requires the container PID to already exist; write to cgroup.procs after clone/fork but before exec"
      - "memory.max does not account for kernel memory by default in v2; kernel memory is included automatically"
      - Cleanup order matters: all processes must exit before the cgroup directory can be removed (rmdir)
      - "Not enabling the memory and pids controllers in the parent cgroup's cgroup.subtree_control prevents child cgroups from using them"
    concepts:
      - cgroups v2 unified hierarchy
      - Resource controllers (memory, cpu, pids)
      - OOM killer behavior
    skills:
      - Resource monitoring and profiling
      - Control group filesystem manipulation
      - System resource management
      - Version detection and compatibility
    deliverables:
      - cgroup version detection (v1 vs v2)
      - cgroup creation and process assignment
      - Memory limit configuration with OOM verification
      - CPU limit configuration with quota/period throttling
      - Process count limit preventing fork bombs
      - cgroup cleanup on container exit
    estimated_hours: "4-6"

  - id: container-basic-m5
    name: "User Namespace and Rootless Containers"
    description: >
      Implement user namespace mapping to enable unprivileged (rootless)
      container execution.
    acceptance_criteria:
      - User namespace (CLONE_NEWUSER) is created allowing an unprivileged user to appear as root (UID 0) inside the container
      - UID/GID mapping is written to /proc/<pid>/uid_map and /proc/<pid>/gid_map before the container process calls exec
      - "'deny' is written to /proc/<pid>/setgroups before writing gid_map (required by kernel for unprivileged user namespace creation)"
      - Container process running as mapped UID 0 can perform mount operations and other privileged operations within its namespaces
      - Outside the container, the process is observed running as the original unprivileged UID
      - All previous milestones (PID, mount, network, cgroups) function correctly when combined with user namespace
    pitfalls:
      - "uid_map and gid_map must be written exactly once by the parent process; writing from within the namespace or writing twice fails"
      - "setgroups must be set to 'deny' before writing gid_map for unprivileged users (kernel security requirement since Linux 3.19)"
      - "User namespace must be created first (or simultaneously with CLONE_NEWUSER | CLONE_NEWPID | ...) for other namespaces to work without root"
      - "Network namespace setup with veth pairs still requires host-side root privileges; rootless networking typically uses slirp4netns or pasta"
      - "Some cgroup operations require delegation setup (systemd cgroup delegation) for rootless cgroup control"
    concepts:
      - User namespaces and UID/GID mapping
      - Rootless container security model
      - Capability-based privilege within namespaces
    skills:
      - UID/GID mapping configuration
      - Privilege escalation understanding
      - Security boundary verification
      - Rootless networking alternatives
    deliverables:
      - User namespace creation with CLONE_NEWUSER
      - UID/GID map writer configuring identity mapping (e.g., 0 1000 1)
      - setgroups deny configuration for unprivileged gid_map
      - Integration test running full container stack as unprivileged user
      - Documentation of limitations (networking, cgroup delegation)
    estimated_hours: "4-6"
```
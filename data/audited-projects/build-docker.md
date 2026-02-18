# AUDIT & FIX: build-docker

## CRITIQUE
- SIGNIFICANT OVERLAP with container-basic project. The first three milestones (namespaces, cgroups, filesystem isolation) are virtually identical. If both projects exist on the platform, the dependency should be explicit: build-docker should require container-basic as a prerequisite, or milestones should reference it.
- Missing milestone for container metadata handling. Docker's value proposition over raw namespaces is the IMAGE SPECIFICATION: parsing Dockerfile-derived config for ENTRYPOINT, CMD, ENV, WORKDIR, EXPOSE, USER, VOLUMES. Without this, it's just 'container-basic with OverlayFS.'
- M1 deliverables mention 'Network namespace setup' but the description says 'PID, UTS, mount' only. Network namespace has its own milestone (M5). Inconsistency.
- M3 mentions both chroot and pivot_root but doesn't clearly state that chroot is insecure and pivot_root should be preferred. chroot is trivially escapable without a mount namespace.
- M4 (OverlayFS) doesn't mention whiteout files, which are how OverlayFS handles file deletions in lower layers. Without understanding whiteouts, layer behavior is mysterious.
- M6 says 'Image pull downloads all layers' but doesn't mention authentication for private registries, which is essential for real Docker Hub usage.
- M6 mentions 'Container lifecycle management with start, stop, and remove' but there's no mention of how stop works (sending SIGTERM, waiting, then SIGKILL).
- No mention of seccomp or AppArmor profiles, which are standard container security boundaries.
- The estimated hours (30-50) is too low for an 'expert' project that includes OCI image pulling and full networking.

## FIXED YAML
```yaml
id: build-docker
name: "Build Your Own Docker"
description: "Container runtime with namespaces, cgroups, OverlayFS, OCI images, and container metadata"
difficulty: expert
estimated_hours: "45-70"
essence: >
  Kernel-level process sandboxing through namespace isolation (PID, mount, network,
  UTS, user), cgroup-enforced resource boundaries, copy-on-write filesystem layering
  with OverlayFS, OCI image format parsing with layer extraction, and container
  metadata application (entrypoint, env, workdir) to create lightweight execution
  environments that share the host kernel.
why_important: >
  Building a container runtime demystifies the core technology behind Docker and
  Kubernetes, teaching low-level Linux kernel interfaces essential for systems
  programming, infrastructure engineering, and understanding production containerized
  environments at a fundamental level.
learning_outcomes:
  - Implement process isolation using Linux namespaces (PID, UTS, mount, network, user)
  - Configure cgroup hierarchies to enforce CPU, memory, and process limits
  - Build filesystem isolation using pivot_root with OverlayFS layering
  - Parse OCI image manifests and extract filesystem layers
  - Apply container metadata (ENTRYPOINT, CMD, ENV, WORKDIR, USER) from image config
  - Create virtual network interfaces with veth pairs and bridge networking
  - Implement container lifecycle (create, start, stop, remove)
  - Debug low-level Linux kernel interactions and syscall failures
skills:
  - Linux Kernel APIs
  - Namespace Isolation
  - Cgroup Resource Management
  - OverlayFS and Union Filesystems
  - OCI Image Specification
  - Container Networking
  - Container Lifecycle Management
  - Low-Level Systems Design
tags:
  - build-from-scratch
  - c
  - cgroups
  - containers
  - expert
  - go
  - isolation
  - namespaces
  - oci
  - rust
architecture_doc: architecture-docs/build-docker/index.md
languages:
  recommended:
    - Go
    - Rust
    - C
  also_possible: []
resources:
  - name: "Containers from Scratch"
    url: "https://ericchiang.github.io/post/containers-from-scratch/"
    type: article
  - name: "Containers from Scratch (Video) - Liz Rice"
    url: "https://www.youtube.com/watch?v=8fi7uSYlOdc"
    type: video
  - name: "Linux Namespaces man page"
    url: "https://man7.org/linux/man-pages/man7/namespaces.7.html"
    type: documentation
  - name: "OCI Runtime Specification"
    url: "https://github.com/opencontainers/runtime-spec"
    type: documentation
  - name: "OCI Image Specification"
    url: "https://github.com/opencontainers/image-spec"
    type: documentation
prerequisites:
  - type: project
    name: "container-basic (or equivalent namespace/cgroup experience)"
  - type: skill
    name: "Linux system administration"
  - type: skill
    name: "Process management (fork, exec)"
  - type: skill
    name: "Filesystem concepts (mount, pivot_root)"
  - type: skill
    name: "Basic networking"
milestones:
  - id: build-docker-m1
    name: "Namespace Isolation (PID, UTS, Mount, User)"
    description: >
      Isolate a process using Linux namespaces. Create PID, UTS, mount, and
      user namespaces for full process isolation without root.
    acceptance_criteria:
      - Process runs inside its own PID namespace and sees itself as PID 1 with no visibility of host processes
      - UTS namespace allows setting a container-specific hostname without affecting the host
      - Mount namespace isolates mount/unmount operations; mount propagation set to private
      - User namespace maps host UID to container UID 0, allowing unprivileged container creation (write uid_map/gid_map with deny on setgroups)
      - All namespaces are created in a single clone() call with CLONE_NEWPID|CLONE_NEWUTS|CLONE_NEWNS|CLONE_NEWUSER flags
      - Verify isolation: /proc/self/status NSpid shows PID 1 inside, host PID outside; hostname differs; mount list differs
    pitfalls:
      - "clone() stack allocation and alignment for the child; pass top of stack on x86-64"
      - "User namespace uid_map/gid_map must be written by the parent before the child calls exec; setgroups must be denied first"
      - "PID 1 in namespace must reap zombies; install a SIGCHLD handler or use prctl(PR_SET_CHILD_SUBREAPER)"
      - "unshare(CLONE_NEWPID) affects children, not the caller; must fork after unshare"
    concepts:
      - Linux namespaces (PID, UTS, mount, user)
      - clone() and unshare() syscalls
      - UID/GID mapping for rootless execution
    skills:
      - System programming
      - Process management
      - Privilege management
    deliverables:
      - Multi-namespace process creation with clone()
      - User namespace with UID/GID mapping
      - UTS namespace with hostname isolation
      - PID namespace with PID 1 init behavior
      - Mount namespace with private propagation
    estimated_hours: "4-6"

  - id: build-docker-m2
    name: "Resource Limits (cgroups)"
    description: >
      Limit container resources using cgroups v2 (with v1 awareness).
    acceptance_criteria:
      - Detect cgroups version (v1 vs v2) and use the appropriate filesystem paths
      - Create a dedicated cgroup for the container; write PID to cgroup.procs before exec
      - Memory limit (memory.max in v2) is enforced; OOM killer terminates the container process when exceeded
      - CPU limit (cpu.max with quota/period in v2) restricts CPU usage; verified under load
      - PID limit (pids.max) prevents fork bombs inside the container
      - Cgroup is cleaned up (directory removed) after all container processes exit
      - For v2: ensure required controllers are enabled in parent's cgroup.subtree_control
    pitfalls:
      - "cgroups v2 requires controllers to be delegated via subtree_control in the parent cgroup"
      - "memory.max in v2 does not have a separate kmem limit; kernel memory is included automatically"
      - "Cleanup requires all processes to be dead before rmdir on the cgroup directory succeeds"
      - "Writing to cgroup files after the process has already exec'd may race; write before exec in the setup phase"
    concepts:
      - cgroups v2 unified hierarchy
      - Resource controllers
      - OOM killer
    skills:
      - Resource management
      - Kernel interface programming
      - Version detection
    deliverables:
      - cgroup version detection
      - cgroup creation and PID assignment
      - Memory, CPU, and PID limit configuration
      - cgroup cleanup on container exit
    estimated_hours: "3-5"

  - id: build-docker-m3
    name: "Filesystem Isolation (pivot_root)"
    description: >
      Give container its own root filesystem using pivot_root with proper
      old-root cleanup.
    acceptance_criteria:
      - New root is bind-mounted onto itself (mount --bind newroot newroot) making it a valid mount point for pivot_root
      - "pivot_root() atomically swaps the container's root; old root is mounted at a subdirectory (e.g., /oldroot)"
      - Old root is unmounted with umount2(MNT_DETACH) making the host filesystem completely inaccessible from within
      - /proc is mounted inside the container (mount -t proc proc /proc) with nosuid,noexec,nodev flags
      - /dev is minimally populated with null, zero, urandom, tty devices (bind-mount from host or mknod)
      - Container works with any extracted rootfs tarball (Alpine, Ubuntu, BusyBox)
      - Host filesystem is verified unreachable: ls /oldroot fails, no host paths visible in /proc/mounts
    pitfalls:
      - "pivot_root fails with EINVAL if new_root is not a mount point; the bind-mount-to-self trick is required"
      - "chroot is escapable without mount namespace; always use pivot_root inside a mount namespace"
      - "Forgetting to mount /proc means ps, top, and /proc/self/* don't work inside the container"
      - "Not unmounting old root is a container escape vector"
      - "Missing /dev/null causes many programs to fail silently"
    concepts:
      - pivot_root vs chroot security
      - Root filesystem preparation
      - Minimal /dev population
    skills:
      - Filesystem operations
      - Mount management
      - Security hardening
    deliverables:
      - Rootfs extraction from tarball
      - Bind-mount-to-self preparation
      - pivot_root execution with old root cleanup
      - /proc, /sys, /dev mounting inside container
    estimated_hours: "3-5"

  - id: build-docker-m4
    name: "Layered Filesystem (OverlayFS)"
    description: >
      Implement OverlayFS layering for efficient image storage with
      copy-on-write semantics.
    acceptance_criteria:
      - OverlayFS mount merges multiple read-only lower directories into a single unified view with a writable upper directory
      - Mount command uses: mount -t overlay overlay -o lowerdir=l3:l2:l1,upperdir=upper,workdir=work merged
      - Write operations to files from lower layers trigger copy-up to the upper layer; lower layers remain unmodified
      - File deletion in the container creates whiteout entries (character device 0/0) in the upper layer, hiding the lower-layer file
      - Common image layers are shared between multiple containers; each container has its own upper (writable) layer
      - Container start creates a fresh upper layer; container remove deletes only the upper layer
      - Disk space usage is verified: two containers sharing 5 layers consume space for 5 layers + 2 upper layers, not 2 × 5 layers
    pitfalls:
      - "OverlayFS workdir must be on the same filesystem as upperdir; using a different filesystem causes mount failure"
      - "Directory rename in OverlayFS triggers a full copy-up of all contents; this is expensive and surprising"
      - "Whiteout files (mknod c 0 0) are OverlayFS-specific; they must be cleaned up properly"
      - "Not all filesystem features work on OverlayFS (e.g., inotify on lower-layer files may not trigger)"
      - lowerdir order matters: rightmost is the bottom layer, leftmost is the top
    concepts:
      - Union/overlay filesystems
      - Copy-on-write semantics
      - Whiteout files for deletion
      - Layer deduplication
    skills:
      - OverlayFS configuration
      - Storage optimization
      - Layer management
    deliverables:
      - OverlayFS mount with lower/upper/work directory setup
      - Layer stacking from multiple read-only directories
      - Copy-on-write verification test
      - Whiteout file detection and explanation
      - Layer sharing between multiple containers
    estimated_hours: "4-6"

  - id: build-docker-m5
    name: "Container Networking"
    description: >
      Set up network namespace with veth pairs, bridge, and NAT for container connectivity.
    acceptance_criteria:
      - Network namespace (CLONE_NEWNET) provides isolated interfaces, routing, and iptables
      - veth pair created; one end moved into container namespace, other end attached to a host bridge (e.g., docker0)
      - "Bridge has an IP address serving as the container's default gateway"
      - Container has a unique IP address on the bridge subnet; default route points to bridge IP
      - Loopback interface inside container is brought up
      - NAT masquerade rule on host enables outbound internet from container
      - Port forwarding (DNAT) maps a host port to a container port for inbound access
      - DNS resolution inside container works (/etc/resolv.conf is configured)
      - Two containers on the same bridge can communicate via their IP addresses
    pitfalls:
      - "veth pair must be created on the host, then one end moved into the container's netns with 'ip link set <veth> netns <pid>'"
      - IP forwarding must be enabled on host: sysctl net.ipv4.ip_forward=1
      - "iptables MASQUERADE rule must specify the correct outbound interface"
      - DNS: copy /etc/resolv.conf from host or generate one pointing to a known DNS server
      - "Not cleaning up veth pairs, bridge, and iptables rules on container removal leaks resources"
    concepts:
      - Network namespaces
      - veth pairs and bridges
      - NAT and port forwarding
    skills:
      - Network configuration
      - iptables management
      - Virtual device lifecycle
    deliverables:
      - veth pair creation and namespace assignment
      - Bridge setup and IP configuration
      - NAT masquerade for outbound connectivity
      - Port forwarding (DNAT) for inbound access
      - DNS configuration inside container
      - Inter-container communication verification
    estimated_hours: "5-8"

  - id: build-docker-m6
    name: "OCI Image Pulling and Container Metadata"
    description: >
      Pull OCI images from a registry, parse manifests and config, extract layers,
      and apply container metadata (ENTRYPOINT, CMD, ENV, WORKDIR, USER).
    acceptance_criteria:
      - Image pull fetches the manifest from a Docker registry (e.g., registry-1.docker.io) using the Docker Registry HTTP API v2
      - OCI manifest is parsed to extract layer digests (in order), config digest, and media types
      - Image config JSON is parsed to extract: Entrypoint, Cmd, Env, WorkingDir, User, ExposedPorts, Volumes
      - Layers are downloaded by digest, verified by SHA256 hash, and extracted as tar.gz archives in order
      - Extracted layers are stored in a content-addressable directory (named by digest) for deduplication
      - Container run applies metadata: ENV variables are set in the child process environment; WORKDIR is chdir'd to before exec; USER changes the effective UID/GID; ENTRYPOINT + CMD form the executed command
      - "If user specifies a command on the CLI (e.g., 'run ubuntu /bin/bash'), it overrides CMD but ENTRYPOINT is preserved (Docker semantics)"
      - Image layers are stacked as OverlayFS lower directories in the correct order
    pitfalls:
      - Docker Hub requires auth token exchange (GET /token?service=registry.docker.io&scope=repository: library/alpine:pull) before pulling
      - "Manifest schema v2 vs OCI manifest; handle both media types"
      - Content-addressable storage: layers are identified by sha256 digest, not by name; dedup relies on exact digest matching
      - ENTRYPOINT + CMD interaction: ENTRYPOINT is the executable, CMD provides default arguments; CLI args override CMD only
      - "USER directive requires resolving the username to UID via /etc/passwd inside the rootfs, not the host"
    concepts:
      - OCI image specification
      - Docker Registry HTTP API v2
      - Container metadata (Entrypoint, Cmd, Env, WorkingDir, User)
      - Content-addressable storage
    skills:
      - HTTP API client
      - JSON parsing
      - Image layer management
      - Container configuration
    deliverables:
      - Registry API client with auth token exchange
      - Manifest parser extracting layer digests and config
      - Image config parser extracting metadata fields
      - Layer downloader with SHA256 verification
      - Content-addressable layer storage
      - Container run command applying ENV, WORKDIR, USER, ENTRYPOINT+CMD
    estimated_hours: "8-12"

  - id: build-docker-m7
    name: "Container Lifecycle (start, stop, remove)"
    description: >
      Implement full container lifecycle management with proper signal handling
      and resource cleanup.
    acceptance_criteria:
      - "'create' prepares all container state (rootfs, cgroup, network) without starting the process"
      - "'start' launches the container process within all configured namespaces and cgroups"
      - "'stop' sends SIGTERM to the container's PID 1; waits up to a configurable timeout (default 10s); sends SIGKILL if still running"
      - "'remove' deletes the container's upper OverlayFS layer, cgroup, veth pair, and any port forwarding rules"
      - "Container state is tracked (created, running, stopped, removed) and queryable via a 'ps' or 'list' command"
      - "'exec' runs a new command inside an existing running container's namespaces (using nsenter or setns)"
      - "Exit code of the container's main process is captured and reported"
    pitfalls:
      - "SIGTERM must be sent to the container's PID 1 from the host PID namespace (the host-visible PID), not PID 1 directly"
      - "If PID 1 inside the container doesn't handle SIGTERM, it won't die (PID 1 is protected from signals it doesn't handle)"
      - "nsenter/setns for 'exec' requires opening /proc/<pid>/ns/* file descriptors for each namespace"
      - Cleanup order: stop processes → remove cgroup → unmount overlay → remove network → delete state files
    concepts:
      - Container lifecycle state machine
      - Graceful shutdown (SIGTERM → SIGKILL)
      - nsenter for exec-into-container
    skills:
      - Process lifecycle management
      - Signal handling
      - Resource cleanup
      - State machine design
    deliverables:
      - Container create preparing all resources
      - Container start launching the isolated process
      - Container stop with SIGTERM → timeout → SIGKILL
      - Container remove with full resource cleanup
      - Container list/ps showing state of all containers
      - "Container exec entering running container's namespaces"
    estimated_hours: "6-8"
```
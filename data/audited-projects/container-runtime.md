# AUDIT & FIX: container-runtime

## CRITIQUE
- **Missing pivot_root/chroot**: The project talks about mount namespaces and overlay filesystems but never mentions the actual mechanism (pivot_root or chroot) that confines the process to the container's root filesystem. Without this, the container process can still see the host filesystem. This is a critical security gap.
- **Missing Image Management**: The overlay filesystem milestone assumes image layers already exist on disk, but never explains how they got there. A real container runtime needs to pull OCI images (from a registry), verify digests, and unpack tarballs into layers. Without this, the learner has to manually create layers, which isn't how containers work.
- **Java is Not a Practical Language Choice**: The project requires direct syscall manipulation (clone, unshare, mount, pivot_root, setns, cgroup filesystem writes). Java's JNI overhead makes this extremely painful. Python with ctypes is marginal but possible. Go and Rust are the only practical choices. Listing Java as 'recommended' is misleading.
- **Cgroups v1 vs v2 Not Addressed Systematically**: The pitfall mentions 'Cgroups v1 vs v2 have different interfaces' but the ACs only target v2. Many production systems still run v1 or hybrid. The project should explicitly state it targets cgroups v2 unified hierarchy and require the learner to detect and fail gracefully on v1.
- **Security is Underaddressed**: There's no mention of seccomp profiles, capability dropping, or apparmor/SELinux. A container that only uses namespaces without restricting capabilities is trivially escapable. While a full security hardening milestone may be out of scope, at minimum capabilities should be dropped.
- **50 Hours with Uniform 12.5-Hour Milestones is Suspicious**: Network namespace setup with veth/bridge/NAT is significantly harder than PID namespace creation. The estimates should reflect this.
- **No Container Lifecycle Management**: There's no create/start/stop/delete lifecycle. The milestones implement isolated components but never integrate them into a runtime that can manage container state.

## FIXED YAML
```yaml
id: container-runtime
name: Container Runtime
description: OCI-compliant container execution environment
difficulty: advanced
estimated_hours: "50-70"
essence: >
  Kernel-level process isolation through Linux namespace manipulation,
  filesystem confinement via pivot_root into overlay-mounted image layers,
  hierarchical resource constraint enforcement via cgroup v2 controllers,
  and network isolation with veth/bridge networking—assembled into a
  container lifecycle (create, start, stop, delete).
why_important: >
  Understanding containers at the kernel level is essential for debugging
  container issues, optimizing performance, securing deployments, and
  understanding what Docker and Kubernetes actually do under the hood.
learning_outcomes:
  - Use Linux namespaces (PID, mount, network, UTS, user) for process isolation
  - Implement pivot_root to confine a process to a container root filesystem
  - Pull and unpack OCI image tarballs into layered filesystem directories
  - Build overlay filesystem mounts combining image layers with a writable layer
  - Implement cgroups v2 for CPU, memory, and I/O resource limits
  - Set up bridge networking with veth pairs and NAT port forwarding
  - Drop unnecessary Linux capabilities for security hardening
  - Build container lifecycle management (create, start, exec, stop, delete)
skills:
  - Linux Namespaces (PID, mount, network, UTS, user)
  - pivot_root and Filesystem Confinement
  - OCI Image Specification
  - Overlay Filesystem (overlayfs)
  - Cgroups v2 Resource Control
  - Bridge Networking and NAT
  - Linux Capabilities
  - Container Lifecycle Management
tags:
  - advanced
  - containers
  - images
  - layers
  - linux
  - oci
  - runc
  - systems
architecture_doc: architecture-docs/container-runtime/index.md
languages:
  recommended:
    - Go
    - Rust
  also_possible:
    - C
    - Python (with ctypes)
resources:
  - name: "Linux Namespaces Manual"
    url: "https://man7.org/linux/man-pages/man7/namespaces.7.html"
    type: documentation
  - name: "pivot_root Manual"
    url: "https://man7.org/linux/man-pages/man2/pivot_root.2.html"
    type: documentation
  - name: "Overlay Filesystem Documentation"
    url: "https://docs.kernel.org/filesystems/overlayfs.html"
    type: documentation
  - name: "Build Your Own Container (Go)"
    url: "https://www.infoq.com/articles/build-a-container-golang/"
    type: tutorial
  - name: "Cgroups v2 Manual Page"
    url: "https://man7.org/linux/man-pages/man7/cgroups.7.html"
    type: documentation
  - name: "OCI Image Specification"
    url: "https://github.com/opencontainers/image-spec"
    type: documentation
prerequisites:
  - type: project
    id: container-basic
    name: "Container (Basic)"
  - type: skill
    name: "Linux system programming (syscalls, file descriptors)"
  - type: skill
    name: "Linux namespaces and cgroups concepts"
milestones:
  - id: container-runtime-m1
    name: "Namespaces & Filesystem Confinement"
    description: >
      Create isolated process environments using Linux namespaces (PID,
      mount, UTS, user) and confine the process to a container root
      filesystem using pivot_root. Mount essential pseudo-filesystems
      (/proc, /sys, /dev) inside the container.
    acceptance_criteria:
      - Create new PID namespace: "process inside container sees itself as PID 1 (verified by running 'cat /proc/self/status' inside container)"
      - Create new mount namespace: mounts inside container are not visible from the host (verified by mounting tmpfs inside container and checking host /proc/mounts)
      - Create new UTS namespace: "container can set its own hostname without affecting the host (verified with 'hostname' command inside and outside)"
      - Create new user namespace with UID/GID mapping: container root (UID 0) maps to an unprivileged host UID (verified by checking /proc/self/uid_map inside container)
      - pivot_root: after namespace creation, mount the container rootfs as a new mount point, call pivot_root to swap root, and unmount the old root; the process can no longer access the host filesystem (verified by attempting to read host-specific files like /etc/hostname and confirming they show container values)
      - "Mount /proc inside the container (mount -t proc proc /proc) so process tools (ps, top) work correctly inside the container"
      - "Mount minimal /dev with /dev/null, /dev/zero, /dev/random, /dev/urandom as bind mounts or device nodes"
      - "Drop all Linux capabilities except a minimal set (CAP_NET_BIND_SERVICE, CAP_KILL, CAP_CHOWN) after namespace setup; verify with capsh --print inside container"
    pitfalls:
      - "pivot_root requires the new root to be a mount point (not just a directory); you must bind-mount the rootfs directory onto itself before calling pivot_root"
      - "Forgetting to mount /proc inside the new PID namespace breaks ps, top, and /proc/self/* — most process tools become useless"
      - "User namespace UID mapping must be written to /proc/PID/uid_map BEFORE the process tries to do anything as root inside the namespace; order matters"
      - "Not unmounting the old root after pivot_root leaves the host filesystem accessible inside the container — this is a container escape vulnerability"
      - "Dropping capabilities too early (before pivot_root and mount operations) prevents the container setup from completing; drop capabilities as the LAST setup step before exec'ing the container entrypoint"
    concepts:
      - Linux namespace types and their isolation boundaries
      - pivot_root vs chroot (pivot_root is more secure as it fully detaches the old root)
      - User namespace UID/GID mapping for rootless containers
      - Linux capabilities and the principle of least privilege
      - Pseudo-filesystem mounts (/proc, /dev, /sys)
    deliverables:
      - "Namespace creation using clone() or unshare() for PID, mount, UTS, and user namespaces"
      - "pivot_root implementation swapping container rootfs as the new root"
      - "Old root cleanup unmounting host filesystem from container view"
      - "/proc and /dev mounting inside container namespace"
      - "Capability dropping to minimal set after setup"
      - "Verification tests confirming isolation for each namespace type"
    estimated_hours: "10-14"

  - id: container-runtime-m2
    name: "OCI Image Management & Overlay Filesystem"
    description: >
      Pull OCI image tarballs from a registry (or load from local archive),
      unpack layers into content-addressable storage, and mount them as an
      overlayfs with a writable upper layer. Clean up on container removal.
    acceptance_criteria:
      - "Pull an OCI image manifest and layer blobs from a container registry (e.g., Docker Hub) via the OCI Distribution API (HTTP-based); authenticate with token-based auth for public images"
      - "Alternatively, load an OCI image from a local tar archive (e.g., exported via 'docker save' or 'skopeo copy')"
      - Verify layer integrity: "each downloaded layer blob's SHA256 digest matches the digest listed in the image manifest; reject corrupted layers"
      - "Unpack each layer tarball into a content-addressable directory (e.g., /var/lib/runtime/layers/<sha256>/)"
      - "Mount overlayfs combining all image layers (read-only lower dirs in correct stacking order) with a per-container writable upper dir and work dir"
      - Verify copy-on-write: modify a file inside the container; confirm the modification exists only in the upper dir, not in any lower layer
      - Layer caching: if a layer with the same digest already exists locally, skip download and reuse the cached layer
      - Cleanup on container removal: unmount overlayfs, delete the writable upper and work dirs; shared read-only layers are retained for reuse
      - "Work directory is on the same filesystem as the upper directory (overlayfs requirement); verify this or fail with a clear error"
    pitfalls:
      - "Overlayfs requires kernel 3.18+; older kernels silently fail or produce incorrect behavior. Check kernel version at startup."
      - "The work directory MUST be on the same filesystem as the upper directory and MUST be empty; violating either causes mount failure with a confusing error"
      - "Hardlinks across overlayfs layers cause unexpected behavior (POSIX inode semantics break); this is a known overlayfs limitation"
      - Layer ordering matters: the OCI manifest lists layers bottom-to-top; overlayfs lowerdir parameter lists them left-to-right (topmost first). Getting this wrong produces a broken filesystem.
      - "Not verifying layer digests allows supply-chain attacks where a compromised registry serves modified layers"
    concepts:
      - OCI Image Specification (manifest, config, layers)
      - OCI Distribution API for pulling images
      - Content-addressable storage for layer deduplication
      - Overlayfs mount semantics (lowerdir, upperdir, workdir)
      - Copy-on-write filesystem behavior
    deliverables:
      - Image puller: download manifest and layer blobs from OCI registry with digest verification
      - Layer unpacker: extract tar layers into content-addressable storage directories
      - Overlayfs mounter: combine layers and writable dir into container rootfs mount
      - Layer cache: skip download for locally-available layers
      - Cleanup function: unmount overlayfs and remove per-container writable state
    estimated_hours: "10-14"

  - id: container-runtime-m3
    name: "Resource Limits with Cgroups v2"
    description: >
      Implement CPU, memory, and I/O resource limits using the cgroups v2
      unified hierarchy. Monitor resource usage and handle OOM events.
    acceptance_criteria:
      - "Detect cgroups version at startup; if cgroups v2 unified hierarchy is not available, fail with a clear error message (do not silently fall back to v1)"
      - "Create a cgroup for each container under /sys/fs/cgroup/<runtime-name>/<container-id>/"
      - "Set memory.max to a configurable hard limit; set memory.high to a configurable soft limit that triggers throttling before OOM"
      - "Set cpu.max with configurable quota and period (e.g., '50000 100000' for 50% CPU); verify the container process cannot exceed the configured CPU percentage under sustained load (stress test)"
      - "Set io.max for configurable I/O bandwidth limits on specified block devices"
      - "Add the container process to the cgroup by writing its PID to cgroup.procs"
      - "Monitor resource usage by reading memory.current, cpu.stat, and io.stat; expose these as container metrics"
      - OOM handling: detect OOM kill events by monitoring memory.events (oom_kill counter); log the event and optionally restart the container based on configuration
      - Cleanup: remove the cgroup directory on container deletion; verify no processes remain in the cgroup before removal
    pitfalls:
      - "Cannot remove a cgroup directory while processes are still inside it; EBUSY is returned. Kill all processes first."
      - "Memory limit without swap limit (memory.swap.max) allows the container to use unlimited swap, effectively bypassing the memory limit. Set memory.swap.max = 0 or a bounded value."
      - "cpu.max quota of 0 means no limit (not 'no CPU'); don't confuse this with setting no value"
      - "Writing to cgroup files requires the process to have write access; in rootless mode, cgroup delegation must be configured via systemd"
    concepts:
      - Cgroups v2 unified hierarchy and controller delegation
      - Memory controller (memory.max, memory.high, memory.events)
      - CPU controller (cpu.max with quota/period)
      - I/O controller (io.max for bandwidth throttling)
      - OOM kill detection and handling
    deliverables:
      - "Cgroups v2 detection and validation at startup"
      - "Per-container cgroup creation with configurable memory, CPU, and I/O limits"
      - "Process attachment to cgroup via cgroup.procs"
      - "Resource monitoring reading memory.current, cpu.stat, io.stat"
      - "OOM event detection via memory.events monitoring"
      - "Cgroup cleanup on container deletion"
    estimated_hours: "10-14"

  - id: container-runtime-m4
    name: "Container Networking"
    description: >
      Implement bridge networking for containers using veth pairs, a Linux
      bridge, IP address assignment, and NAT-based port forwarding.
    acceptance_criteria:
      - "Create a Linux bridge (e.g., 'runtime0') with a configurable subnet (e.g., 172.20.0.0/16); assign the bridge IP as the gateway"
      - "For each container, create a veth pair; move one end into the container's network namespace; attach the other end to the bridge"
      - "Assign a unique IP address from the bridge subnet to the container's veth interface; configure the default route to point to the bridge gateway IP"
      - "Enable IP forwarding (net.ipv4.ip_forward = 1) on the host; add iptables MASQUERADE rule for the bridge subnet to enable outbound internet access from containers"
      - Port mapping: add iptables DNAT rules to forward traffic from host_port to container_ip:container_port; verified by curling host:host_port from outside and receiving response from container
      - Inter-container communication: two containers on the same bridge can communicate via their container IPs without NAT (verified by ping between containers)
      - DNS resolution: "write a resolv.conf inside the container with configurable nameserver (default: host's nameserver); container can resolve external domains"
      - Cleanup: on container removal, delete veth pair, remove iptables rules (DNAT and MASQUERADE for specific container), and release IP address back to the pool
    pitfalls:
      - "Deleting the host-side veth automatically deletes the container-side veth; but if the container is deleted first, the host-side veth may orphan. Always clean up explicitly."
      - "iptables rules persist after container death if not explicitly removed; leaked rules accumulate and cause port conflicts or security holes"
      - "MTU mismatch between bridge and container veth causes packet fragmentation and performance degradation; set consistent MTU across all interfaces"
      - IP address pool exhaustion: if the subnet is too small for the number of containers, allocation fails. Use at least a /16 subnet.
      - "Not enabling ip_forward on the host means container traffic to external networks is silently dropped"
    concepts:
      - Virtual Ethernet (veth) pairs and network namespaces
      - Linux bridge networking and packet forwarding
      - NAT (MASQUERADE) for outbound and DNAT for inbound port mapping
      - IP address management (IPAM) from a subnet pool
      - DNS resolution configuration inside containers
    deliverables:
      - "Bridge creation and configuration with gateway IP"
      - "Veth pair creation with one end in container namespace, one on bridge"
      - "IP address allocation from configurable subnet pool"
      - "iptables MASQUERADE for outbound NAT and DNAT for port forwarding"
      - "Inter-container connectivity test via bridge"
      - "Cleanup function removing veth, iptables rules, and releasing IP on container deletion"
    estimated_hours: "10-14"

  - id: container-runtime-m5
    name: "Container Lifecycle & CLI"
    description: >
      Integrate all components into a container lifecycle manager with
      create, start, exec, stop, and delete operations, exposed via a
      CLI tool.
    acceptance_criteria:
      - CLI command 'runtime create <image> --name <name> --memory <limit> --cpu <quota> --port <host: "container>' creates a container: pulls/caches image, unpacks layers, sets up overlayfs, creates cgroup, configures networking, but does NOT start the process"
      - "CLI command 'runtime start <name>' starts the container process inside its namespaces, attached to its cgroup, with pivot_root into the overlayfs rootfs"
      - "CLI command 'runtime exec <name> <cmd>' executes a new process inside an existing running container's namespaces (using setns syscall)"
      - "CLI command 'runtime stop <name>' sends SIGTERM to the container's PID 1, waits for configurable grace period, then SIGKILL if still running"
      - CLI command 'runtime delete <name>' removes the container: unmounts overlayfs, removes cgroup, cleans up networking, deletes writable layer
      - "CLI command 'runtime list' shows all containers with name, status (created/running/stopped), image, PID, IP, and created timestamp"
      - "Container state is persisted to disk (JSON file per container) so the runtime survives restarts and can manage previously created containers"
      - End-to-end test: create a container from an alpine image, start it running a web server, curl it via port mapping, exec a command inside it, stop it, delete it; all operations succeed
    pitfalls:
      - "Container PID 1 must handle SIGTERM properly; if the entrypoint ignores SIGTERM, the grace period expires and SIGKILL is used, which doesn't allow graceful shutdown"
      - "setns for exec requires the target namespaces' file descriptors in /proc/<pid>/ns/; if the container PID 1 has exited, exec fails"
      - "State file corruption (e.g., crash during write) causes the runtime to lose track of containers; use atomic write (write to temp file, then rename)"
      - Zombie processes: "if PID 1 inside the container doesn't reap children, zombie processes accumulate. Consider a minimal init process (like tini) as PID 1."
    concepts:
      - Container lifecycle state machine (created -> running -> stopped -> deleted)
      - Process management (fork, exec, signal handling)
      - setns for joining existing namespaces
      - Persistent state management for container metadata
      - Integration of namespaces, cgroups, overlayfs, and networking
    deliverables:
      - "CLI tool with create, start, exec, stop, delete, and list commands"
      - "Container state persistence (JSON file per container with metadata)"
      - Lifecycle integration: create chains image pull + overlayfs + cgroup + network setup
      - "exec implementation using setns to join running container namespaces"
      - "Graceful stop with SIGTERM + grace period + SIGKILL"
      - "End-to-end integration test covering full lifecycle"
    estimated_hours: "10-14"
```
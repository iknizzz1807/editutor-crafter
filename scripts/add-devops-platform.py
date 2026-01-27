#!/usr/bin/env python3
"""
Add DevOps & Platform Engineering projects to the curriculum.
Focus on CI/CD, containers, infrastructure as code, service mesh.
"""

import yaml
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir, '..', 'data', 'projects.yaml')

devops_projects = {
    "ci-cd-pipeline": {
        "name": "CI/CD Pipeline",
        "description": "Build a continuous integration and deployment pipeline that handles build, test, artifact management, and deployment automation.",
        "why_important": "CI/CD is the backbone of modern software delivery. Understanding pipeline internals makes you better at debugging and optimizing delivery workflows.",
        "difficulty": "intermediate",
        "tags": ["devops", "automation", "infrastructure"],
        "estimated_hours": 35,
        "prerequisites": ["shell"],
        "learning_outcomes": [
            "Design multi-stage pipeline workflows",
            "Implement build and test automation",
            "Handle artifact versioning and storage",
            "Automate deployment with rollback support"
        ],
        "milestones": [
            {
                "name": "Pipeline Definition Parser",
                "description": "Parse YAML pipeline definitions with stages, jobs, steps, and dependencies between them.",
                "hints": {
                    "level1": "Use YAML to define stages with ordered steps.",
                    "level2": "Build a DAG from job dependencies for execution order.",
                    "level3": """```python
from dataclasses import dataclass, field
from typing import Optional
import yaml
from enum import Enum

class StepType(str, Enum):
    SCRIPT = "script"
    CHECKOUT = "checkout"
    ARTIFACT_UPLOAD = "artifact_upload"
    ARTIFACT_DOWNLOAD = "artifact_download"
    DEPLOY = "deploy"

@dataclass
class Step:
    name: str
    type: StepType
    script: Optional[str] = None
    working_dir: str = "."
    env: dict = field(default_factory=dict)
    timeout: int = 600
    continue_on_error: bool = False

@dataclass
class Job:
    name: str
    steps: list[Step]
    needs: list[str] = field(default_factory=list)
    runs_on: str = "default"
    env: dict = field(default_factory=dict)
    condition: Optional[str] = None

@dataclass
class Stage:
    name: str
    jobs: list[Job]

@dataclass
class Pipeline:
    name: str
    stages: list[Stage]
    triggers: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)

class PipelineParser:
    def parse(self, yaml_content: str) -> Pipeline:
        data = yaml.safe_load(yaml_content)

        stages = []
        for stage_data in data.get('stages', []):
            jobs = []
            for job_name, job_data in stage_data.get('jobs', {}).items():
                steps = [
                    Step(
                        name=s.get('name', f'step-{i}'),
                        type=StepType(s.get('type', 'script')),
                        script=s.get('script'),
                        working_dir=s.get('working_dir', '.'),
                        env=s.get('env', {}),
                        timeout=s.get('timeout', 600),
                        continue_on_error=s.get('continue_on_error', False)
                    )
                    for i, s in enumerate(job_data.get('steps', []))
                ]
                jobs.append(Job(
                    name=job_name,
                    steps=steps,
                    needs=job_data.get('needs', []),
                    runs_on=job_data.get('runs_on', 'default'),
                    env=job_data.get('env', {}),
                    condition=job_data.get('if')
                ))
            stages.append(Stage(name=stage_data['name'], jobs=jobs))

        return Pipeline(
            name=data.get('name', 'pipeline'),
            stages=stages,
            triggers=data.get('triggers', {}),
            env=data.get('env', {})
        )

    def build_execution_graph(self, pipeline: Pipeline) -> dict:
        \"\"\"Build DAG for job execution order.\"\"\"
        graph = {}
        for stage in pipeline.stages:
            for job in stage.jobs:
                graph[job.name] = {
                    'job': job,
                    'dependencies': job.needs,
                    'stage': stage.name
                }
        return graph

    def topological_sort(self, graph: dict) -> list[str]:
        \"\"\"Return jobs in valid execution order.\"\"\"
        visited = set()
        order = []

        def visit(name):
            if name in visited:
                return
            visited.add(name)
            for dep in graph[name]['dependencies']:
                visit(dep)
            order.append(name)

        for name in graph:
            visit(name)
        return order
```"""
                },
                "pitfalls": [
                    "Circular dependencies cause infinite loop",
                    "Missing dependency reference crashes at runtime",
                    "YAML anchors and aliases need special handling"
                ]
            },
            {
                "name": "Job Executor",
                "description": "Execute jobs in isolated environments with proper logging, timeout handling, and exit code propagation.",
                "hints": {
                    "level1": "Run each step in subprocess, capture output.",
                    "level2": "Implement timeout and cancellation handling.",
                    "level3": """```python
import asyncio
import subprocess
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"

@dataclass
class StepResult:
    name: str
    status: JobStatus
    exit_code: int
    stdout: str
    stderr: str
    duration: float
    started_at: datetime
    finished_at: datetime

@dataclass
class JobResult:
    name: str
    status: JobStatus
    step_results: list[StepResult]
    duration: float

class JobExecutor:
    def __init__(self, workspace: str):
        self.workspace = workspace
        self._cancelled = False

    async def execute(self, job: Job, context: dict) -> JobResult:
        results = []
        job_start = datetime.utcnow()
        status = JobStatus.SUCCESS

        # Check condition
        if job.condition and not self._evaluate_condition(job.condition, context):
            return JobResult(job.name, JobStatus.SKIPPED, [], 0)

        # Merge environment
        env = {**os.environ, **context.get('env', {}), **job.env}

        for step in job.steps:
            if self._cancelled:
                status = JobStatus.CANCELLED
                break

            step_env = {**env, **step.env}
            result = await self._execute_step(step, step_env)
            results.append(result)

            if result.status == JobStatus.FAILED and not step.continue_on_error:
                status = JobStatus.FAILED
                break

        duration = (datetime.utcnow() - job_start).total_seconds()
        return JobResult(job.name, status, results, duration)

    async def _execute_step(self, step: Step, env: dict) -> StepResult:
        started = datetime.utcnow()

        try:
            if step.type == StepType.SCRIPT:
                result = await self._run_script(step.script, env, step.timeout, step.working_dir)
            elif step.type == StepType.CHECKOUT:
                result = await self._checkout(env)
            else:
                result = (0, "", "")

            exit_code, stdout, stderr = result
            status = JobStatus.SUCCESS if exit_code == 0 else JobStatus.FAILED

        except asyncio.TimeoutError:
            exit_code, stdout, stderr = -1, "", "Step timed out"
            status = JobStatus.FAILED
        except Exception as e:
            exit_code, stdout, stderr = -1, "", str(e)
            status = JobStatus.FAILED

        finished = datetime.utcnow()
        return StepResult(
            name=step.name,
            status=status,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration=(finished - started).total_seconds(),
            started_at=started,
            finished_at=finished
        )

    async def _run_script(self, script: str, env: dict,
                         timeout: int, working_dir: str) -> tuple:
        work_path = os.path.join(self.workspace, working_dir)

        proc = await asyncio.create_subprocess_shell(
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=work_path
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            return proc.returncode, stdout.decode(), stderr.decode()
        except asyncio.TimeoutError:
            proc.kill()
            raise

    def cancel(self):
        self._cancelled = True
```"""
                },
                "pitfalls": [
                    "Zombie processes when parent killed without cleanup",
                    "Environment variable injection security risk",
                    "Large output causes memory exhaustion"
                ]
            },
            {
                "name": "Artifact Management",
                "description": "Implement artifact upload, download, and caching between pipeline stages with versioning.",
                "hints": {
                    "level1": "Store artifacts with run ID and path in object storage.",
                    "level2": "Implement cache keys based on file hashes for dependency caching.",
                    "level3": """```python
import hashlib
import os
import tarfile
import io
from dataclasses import dataclass
from typing import Optional

@dataclass
class Artifact:
    name: str
    path: str
    run_id: str
    job_name: str
    size: int
    checksum: str

class ArtifactManager:
    def __init__(self, storage, cache_storage):
        self.storage = storage  # S3, GCS, etc.
        self.cache = cache_storage

    async def upload(self, run_id: str, job_name: str,
                    name: str, local_path: str) -> Artifact:
        \"\"\"Upload artifact from local path.\"\"\"
        # Create tarball
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
            tar.add(local_path, arcname=os.path.basename(local_path))

        tar_data = tar_buffer.getvalue()
        checksum = hashlib.sha256(tar_data).hexdigest()

        # Upload to storage
        storage_path = f"artifacts/{run_id}/{job_name}/{name}.tar.gz"
        await self.storage.put(storage_path, tar_data)

        return Artifact(
            name=name,
            path=storage_path,
            run_id=run_id,
            job_name=job_name,
            size=len(tar_data),
            checksum=checksum
        )

    async def download(self, artifact: Artifact, local_path: str):
        \"\"\"Download and extract artifact.\"\"\"
        data = await self.storage.get(artifact.path)

        # Verify checksum
        if hashlib.sha256(data).hexdigest() != artifact.checksum:
            raise ValueError("Artifact checksum mismatch")

        tar_buffer = io.BytesIO(data)
        with tarfile.open(fileobj=tar_buffer, mode='r:gz') as tar:
            tar.extractall(local_path)

    async def cache_save(self, key: str, paths: list[str]):
        \"\"\"Save paths to cache with key.\"\"\"
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
            for path in paths:
                if os.path.exists(path):
                    tar.add(path)

        cache_path = f"cache/{key}.tar.gz"
        await self.cache.put(cache_path, tar_buffer.getvalue())

    async def cache_restore(self, key: str, fallback_keys: list[str] = None) -> bool:
        \"\"\"Restore from cache, try fallback keys if primary misses.\"\"\"
        keys_to_try = [key] + (fallback_keys or [])

        for k in keys_to_try:
            cache_path = f"cache/{k}.tar.gz"
            data = await self.cache.get(cache_path)
            if data:
                tar_buffer = io.BytesIO(data)
                with tarfile.open(fileobj=tar_buffer, mode='r:gz') as tar:
                    tar.extractall('.')
                return True
        return False

    def compute_cache_key(self, prefix: str, files: list[str]) -> str:
        \"\"\"Compute cache key from file contents.\"\"\"
        hasher = hashlib.sha256()
        hasher.update(prefix.encode())

        for file_path in sorted(files):
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    hasher.update(f.read())

        return hasher.hexdigest()[:16]
```"""
                },
                "pitfalls": [
                    "Symlinks in tarball can escape extraction directory",
                    "Cache key collision overwrites unrelated cache",
                    "Large artifacts exhaust memory without streaming"
                ]
            },
            {
                "name": "Deployment Strategies",
                "description": "Implement blue-green and canary deployment strategies with health checks and automatic rollback.",
                "hints": {
                    "level1": "Blue-green: switch traffic between two identical environments.",
                    "level2": "Canary: gradually shift traffic percentage to new version.",
                    "level3": """```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import asyncio

class DeploymentStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class DeploymentConfig:
    service_name: str
    version: str
    replicas: int
    health_check_path: str = "/health"
    health_check_interval: int = 5
    health_check_timeout: int = 30

class DeploymentStrategy(ABC):
    @abstractmethod
    async def deploy(self, config: DeploymentConfig) -> DeploymentStatus:
        pass

    @abstractmethod
    async def rollback(self, config: DeploymentConfig):
        pass

class BlueGreenDeployment(DeploymentStrategy):
    def __init__(self, infrastructure, load_balancer):
        self.infra = infrastructure
        self.lb = load_balancer

    async def deploy(self, config: DeploymentConfig) -> DeploymentStatus:
        service = config.service_name

        # Determine current (blue) and new (green) environments
        current = await self.lb.get_active_environment(service)
        new_env = "green" if current == "blue" else "blue"

        try:
            # Deploy to inactive environment
            await self.infra.deploy(
                f"{service}-{new_env}",
                config.version,
                config.replicas
            )

            # Wait for health checks
            healthy = await self._wait_healthy(
                f"{service}-{new_env}",
                config.health_check_path,
                config.health_check_timeout
            )

            if not healthy:
                await self.infra.destroy(f"{service}-{new_env}")
                return DeploymentStatus.FAILED

            # Switch traffic
            await self.lb.switch_traffic(service, new_env)

            # Keep old environment for quick rollback
            # (destroy after confirmation period)

            return DeploymentStatus.SUCCESS

        except Exception as e:
            await self.rollback(config)
            return DeploymentStatus.FAILED

    async def rollback(self, config: DeploymentConfig):
        service = config.service_name
        current = await self.lb.get_active_environment(service)
        previous = "green" if current == "blue" else "blue"
        await self.lb.switch_traffic(service, previous)

    async def _wait_healthy(self, env: str, path: str, timeout: int) -> bool:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if await self.infra.health_check(env, path):
                return True
            await asyncio.sleep(5)
        return False

class CanaryDeployment(DeploymentStrategy):
    def __init__(self, infrastructure, load_balancer, metrics):
        self.infra = infrastructure
        self.lb = load_balancer
        self.metrics = metrics
        self.stages = [5, 25, 50, 75, 100]  # Traffic percentages
        self.stage_duration = 300  # 5 minutes per stage

    async def deploy(self, config: DeploymentConfig) -> DeploymentStatus:
        service = config.service_name
        canary = f"{service}-canary"

        try:
            # Deploy canary with minimal replicas
            await self.infra.deploy(canary, config.version, 1)

            # Wait for initial health
            if not await self._wait_healthy(canary, config.health_check_path, 60):
                await self.infra.destroy(canary)
                return DeploymentStatus.FAILED

            # Progressive traffic shift
            for percentage in self.stages:
                await self.lb.set_canary_weight(service, percentage)

                # Monitor for anomalies
                healthy = await self._monitor_stage(
                    service, canary, self.stage_duration
                )

                if not healthy:
                    await self._auto_rollback(service, canary)
                    return DeploymentStatus.ROLLED_BACK

            # Canary successful - promote to production
            await self.infra.scale(canary, config.replicas)
            await self.lb.promote_canary(service)
            await self.infra.destroy(f"{service}-stable")

            return DeploymentStatus.SUCCESS

        except Exception:
            await self._auto_rollback(service, canary)
            return DeploymentStatus.FAILED

    async def _monitor_stage(self, service: str, canary: str,
                            duration: int) -> bool:
        \"\"\"Monitor canary health during traffic stage.\"\"\"
        end_time = asyncio.get_event_loop().time() + duration

        while asyncio.get_event_loop().time() < end_time:
            # Check error rate
            canary_errors = await self.metrics.get_error_rate(canary)
            stable_errors = await self.metrics.get_error_rate(f"{service}-stable")

            # Canary error rate significantly higher = problem
            if canary_errors > stable_errors * 1.5:
                return False

            # Check latency
            canary_p99 = await self.metrics.get_latency_p99(canary)
            stable_p99 = await self.metrics.get_latency_p99(f"{service}-stable")

            if canary_p99 > stable_p99 * 2:
                return False

            await asyncio.sleep(30)

        return True

    async def _auto_rollback(self, service: str, canary: str):
        await self.lb.set_canary_weight(service, 0)
        await self.infra.destroy(canary)

    async def rollback(self, config: DeploymentConfig):
        await self._auto_rollback(config.service_name, f"{config.service_name}-canary")
```"""
                },
                "pitfalls": [
                    "Database migrations incompatible between versions",
                    "Session affinity breaks during traffic switch",
                    "Metrics lag causes delayed rollback decision"
                ]
            }
        ]
    },

    "container-runtime": {
        "name": "Container Runtime",
        "description": "Build a minimal container runtime using Linux namespaces, cgroups, and overlay filesystem for process isolation.",
        "why_important": "Understanding containers at the kernel level helps debug container issues, optimize performance, and secure deployments.",
        "difficulty": "advanced",
        "tags": ["containers", "linux", "systems"],
        "estimated_hours": 50,
        "prerequisites": ["shell"],
        "learning_outcomes": [
            "Use Linux namespaces for isolation",
            "Implement cgroups for resource limits",
            "Build overlay filesystem for images",
            "Handle container networking"
        ],
        "milestones": [
            {
                "name": "Process Isolation with Namespaces",
                "description": "Create isolated process environment using PID, mount, network, UTS, and user namespaces.",
                "hints": {
                    "level1": "Use clone() with CLONE_NEWPID, CLONE_NEWNS flags.",
                    "level2": "Set up new root filesystem with pivot_root or chroot.",
                    "level3": """```c
#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mount.h>
#include <sys/syscall.h>

#define STACK_SIZE (1024 * 1024)

static char child_stack[STACK_SIZE];

typedef struct {
    char *rootfs;
    char *hostname;
    char **argv;
} container_config;

static int pivot_root(const char *new_root, const char *put_old) {
    return syscall(SYS_pivot_root, new_root, put_old);
}

static int setup_root(const char *rootfs) {
    // Mount rootfs as private
    if (mount(NULL, "/", NULL, MS_REC | MS_PRIVATE, NULL) < 0) {
        perror("mount private");
        return -1;
    }

    // Bind mount new root
    if (mount(rootfs, rootfs, NULL, MS_BIND | MS_REC, NULL) < 0) {
        perror("mount bind rootfs");
        return -1;
    }

    // Create put_old directory
    char put_old[256];
    snprintf(put_old, sizeof(put_old), "%s/.pivot_root", rootfs);
    mkdir(put_old, 0700);

    // Pivot root
    if (pivot_root(rootfs, put_old) < 0) {
        perror("pivot_root");
        return -1;
    }

    // Change to new root
    if (chdir("/") < 0) {
        perror("chdir");
        return -1;
    }

    // Unmount old root
    if (umount2("/.pivot_root", MNT_DETACH) < 0) {
        perror("umount old root");
        return -1;
    }
    rmdir("/.pivot_root");

    return 0;
}

static int setup_mounts(void) {
    // Mount proc
    if (mount("proc", "/proc", "proc", 0, NULL) < 0) {
        perror("mount proc");
        return -1;
    }

    // Mount sysfs
    if (mount("sysfs", "/sys", "sysfs", 0, NULL) < 0) {
        perror("mount sysfs");
        return -1;
    }

    // Mount tmpfs for /tmp
    if (mount("tmpfs", "/tmp", "tmpfs", 0, NULL) < 0) {
        perror("mount tmpfs");
        return -1;
    }

    return 0;
}

static int child_fn(void *arg) {
    container_config *config = (container_config *)arg;

    // Set hostname
    if (sethostname(config->hostname, strlen(config->hostname)) < 0) {
        perror("sethostname");
        return 1;
    }

    // Setup root filesystem
    if (setup_root(config->rootfs) < 0) {
        return 1;
    }

    // Setup mounts
    if (setup_mounts() < 0) {
        return 1;
    }

    // Execute command
    execvp(config->argv[0], config->argv);
    perror("execvp");
    return 1;
}

int run_container(container_config *config) {
    int flags = CLONE_NEWPID |  // New PID namespace
                CLONE_NEWNS |   // New mount namespace
                CLONE_NEWUTS |  // New UTS namespace (hostname)
                CLONE_NEWNET |  // New network namespace
                CLONE_NEWIPC |  // New IPC namespace
                SIGCHLD;

    pid_t pid = clone(child_fn, child_stack + STACK_SIZE, flags, config);
    if (pid < 0) {
        perror("clone");
        return -1;
    }

    // Wait for child
    int status;
    waitpid(pid, &status, 0);
    return WEXITSTATUS(status);
}
```"""
                },
                "pitfalls": [
                    "pivot_root fails if new root is not a mount point",
                    "Forgetting to mount /proc breaks process tools",
                    "User namespace UID mapping required for unprivileged containers"
                ]
            },
            {
                "name": "Resource Limits with Cgroups",
                "description": "Implement CPU, memory, and I/O limits using cgroups v2 for container resource control.",
                "hints": {
                    "level1": "Write to cgroup filesystem to set limits.",
                    "level2": "Create cgroup hierarchy per container, add process to group.",
                    "level3": """```python
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ResourceLimits:
    memory_limit: int = 0      # bytes, 0 = unlimited
    memory_swap: int = 0       # bytes
    cpu_shares: int = 1024     # relative weight
    cpu_quota: int = 0         # microseconds per period
    cpu_period: int = 100000   # microseconds
    pids_max: int = 0          # max processes

class CgroupManager:
    def __init__(self, cgroup_root: str = "/sys/fs/cgroup"):
        self.root = Path(cgroup_root)

    def create(self, container_id: str, limits: ResourceLimits) -> str:
        \"\"\"Create cgroup for container and apply limits.\"\"\"
        cgroup_path = self.root / "containers" / container_id
        cgroup_path.mkdir(parents=True, exist_ok=True)

        # Enable controllers for this cgroup
        self._enable_controllers(cgroup_path.parent)

        # Apply memory limits
        if limits.memory_limit > 0:
            (cgroup_path / "memory.max").write_text(str(limits.memory_limit))
            if limits.memory_swap >= 0:
                swap_max = limits.memory_limit + limits.memory_swap
                (cgroup_path / "memory.swap.max").write_text(str(swap_max))

        # Apply CPU limits
        if limits.cpu_quota > 0:
            cpu_max = f"{limits.cpu_quota} {limits.cpu_period}"
            (cgroup_path / "cpu.max").write_text(cpu_max)

        (cgroup_path / "cpu.weight").write_text(str(limits.cpu_shares))

        # Apply PID limits
        if limits.pids_max > 0:
            (cgroup_path / "pids.max").write_text(str(limits.pids_max))

        return str(cgroup_path)

    def add_process(self, cgroup_path: str, pid: int):
        \"\"\"Add process to cgroup.\"\"\"
        procs_file = Path(cgroup_path) / "cgroup.procs"
        procs_file.write_text(str(pid))

    def get_stats(self, container_id: str) -> dict:
        \"\"\"Get resource usage statistics.\"\"\"
        cgroup_path = self.root / "containers" / container_id

        stats = {}

        # Memory stats
        memory_current = cgroup_path / "memory.current"
        if memory_current.exists():
            stats['memory_usage'] = int(memory_current.read_text().strip())

        memory_stat = cgroup_path / "memory.stat"
        if memory_stat.exists():
            for line in memory_stat.read_text().splitlines():
                key, value = line.split()
                stats[f'memory_{key}'] = int(value)

        # CPU stats
        cpu_stat = cgroup_path / "cpu.stat"
        if cpu_stat.exists():
            for line in cpu_stat.read_text().splitlines():
                key, value = line.split()
                stats[f'cpu_{key}'] = int(value)

        # PID stats
        pids_current = cgroup_path / "pids.current"
        if pids_current.exists():
            stats['pids_current'] = int(pids_current.read_text().strip())

        return stats

    def destroy(self, container_id: str):
        \"\"\"Remove cgroup (processes must be terminated first).\"\"\"
        cgroup_path = self.root / "containers" / container_id

        # Kill any remaining processes
        procs_file = cgroup_path / "cgroup.procs"
        if procs_file.exists():
            pids = procs_file.read_text().strip().split()
            for pid in pids:
                try:
                    os.kill(int(pid), 9)
                except ProcessLookupError:
                    pass

        # Remove cgroup directory
        cgroup_path.rmdir()

    def _enable_controllers(self, parent_path: Path):
        \"\"\"Enable controllers in parent cgroup.\"\"\"
        subtree_control = parent_path / "cgroup.subtree_control"
        if subtree_control.exists():
            current = subtree_control.read_text()
            needed = "+cpu +memory +pids +io"
            for controller in needed.split():
                if controller[1:] not in current:
                    try:
                        subtree_control.write_text(controller)
                    except PermissionError:
                        pass  # Controller might not be available
```"""
                },
                "pitfalls": [
                    "Cgroups v1 vs v2 have different interfaces",
                    "Can't remove cgroup with active processes",
                    "Memory limit without swap limit allows OOM escape"
                ]
            },
            {
                "name": "Overlay Filesystem",
                "description": "Implement image layering using overlayfs for copy-on-write filesystem support.",
                "hints": {
                    "level1": "Stack read-only layers with one writable upper layer.",
                    "level2": "Handle whiteout files for deletions in upper layer.",
                    "level3": """```python
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import json

@dataclass
class ImageLayer:
    id: str
    parent: str = None
    diff_path: str = ""

@dataclass
class Image:
    id: str
    layers: list[ImageLayer] = field(default_factory=list)
    config: dict = field(default_factory=dict)

class OverlayManager:
    def __init__(self, storage_root: str):
        self.root = Path(storage_root)
        self.layers_dir = self.root / "layers"
        self.images_dir = self.root / "images"
        self.containers_dir = self.root / "containers"

        for d in [self.layers_dir, self.images_dir, self.containers_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def create_layer(self, layer_id: str, parent_id: str = None) -> ImageLayer:
        \"\"\"Create a new image layer.\"\"\"
        layer_path = self.layers_dir / layer_id
        layer_path.mkdir(exist_ok=True)
        (layer_path / "diff").mkdir(exist_ok=True)

        return ImageLayer(
            id=layer_id,
            parent=parent_id,
            diff_path=str(layer_path / "diff")
        )

    def prepare_container(self, container_id: str, image: Image) -> str:
        \"\"\"Prepare overlay mount for container.\"\"\"
        container_path = self.containers_dir / container_id
        container_path.mkdir(exist_ok=True)

        # Create container-specific directories
        upper_dir = container_path / "upper"
        work_dir = container_path / "work"
        merged_dir = container_path / "merged"

        for d in [upper_dir, work_dir, merged_dir]:
            d.mkdir(exist_ok=True)

        # Build lower directories (image layers in order)
        lower_dirs = []
        for layer in reversed(image.layers):
            lower_dirs.append(layer.diff_path)

        return self._mount_overlay(
            lower_dirs=lower_dirs,
            upper_dir=str(upper_dir),
            work_dir=str(work_dir),
            merged_dir=str(merged_dir)
        )

    def _mount_overlay(self, lower_dirs: list[str], upper_dir: str,
                      work_dir: str, merged_dir: str) -> str:
        \"\"\"Mount overlayfs.\"\"\"
        lower = ":".join(lower_dirs)
        options = f"lowerdir={lower},upperdir={upper_dir},workdir={work_dir}"

        subprocess.run([
            "mount", "-t", "overlay", "overlay",
            "-o", options, merged_dir
        ], check=True)

        return merged_dir

    def commit_container(self, container_id: str, new_image_id: str) -> Image:
        \"\"\"Create new image from container's upper layer.\"\"\"
        container_path = self.containers_dir / container_id
        upper_dir = container_path / "upper"

        # Create new layer from upper
        new_layer = self.create_layer(new_image_id)
        shutil.copytree(upper_dir, new_layer.diff_path, dirs_exist_ok=True)

        # Process whiteouts (files starting with .wh.)
        self._process_whiteouts(Path(new_layer.diff_path))

        return Image(id=new_image_id, layers=[new_layer])

    def _process_whiteouts(self, layer_path: Path):
        \"\"\"Handle overlayfs whiteout files.\"\"\"
        for root, dirs, files in os.walk(layer_path):
            for f in files:
                if f.startswith('.wh.'):
                    whiteout_path = Path(root) / f
                    # Convert to OCI whiteout format or mark for deletion
                    original_name = f[4:]  # Remove .wh. prefix
                    whiteout_path.rename(Path(root) / f".wh.{original_name}")

    def cleanup_container(self, container_id: str):
        \"\"\"Unmount and remove container filesystem.\"\"\"
        container_path = self.containers_dir / container_id
        merged_dir = container_path / "merged"

        # Unmount overlay
        subprocess.run(["umount", str(merged_dir)], check=False)

        # Remove container directory
        shutil.rmtree(container_path, ignore_errors=True)
```"""
                },
                "pitfalls": [
                    "Overlayfs requires specific kernel version (3.18+)",
                    "Work directory must be empty and same filesystem as upper",
                    "Hardlinks across layers cause unexpected behavior"
                ]
            },
            {
                "name": "Container Networking",
                "description": "Implement bridge networking for containers with port mapping and inter-container communication.",
                "hints": {
                    "level1": "Create veth pairs, attach one end to bridge, one to container.",
                    "level2": "Use iptables for NAT and port forwarding.",
                    "level3": """```python
import subprocess
import ipaddress
from dataclasses import dataclass
from typing import Optional

@dataclass
class NetworkConfig:
    bridge_name: str = "container0"
    bridge_ip: str = "172.17.0.1/16"
    subnet: str = "172.17.0.0/16"

@dataclass
class ContainerNetwork:
    container_id: str
    ip_address: str
    mac_address: str
    veth_host: str
    veth_container: str
    port_mappings: dict = None  # {host_port: container_port}

class NetworkManager:
    def __init__(self, config: NetworkConfig = None):
        self.config = config or NetworkConfig()
        self.subnet = ipaddress.ip_network(self.config.subnet)
        self.allocated_ips = set()
        self._setup_bridge()

    def _setup_bridge(self):
        \"\"\"Create network bridge if not exists.\"\"\"
        bridge = self.config.bridge_name

        # Create bridge
        subprocess.run([
            "ip", "link", "add", bridge, "type", "bridge"
        ], check=False)

        # Set bridge IP
        subprocess.run([
            "ip", "addr", "add", self.config.bridge_ip, "dev", bridge
        ], check=False)

        # Bring up bridge
        subprocess.run(["ip", "link", "set", bridge, "up"], check=True)

        # Enable IP forwarding
        with open("/proc/sys/net/ipv4/ip_forward", "w") as f:
            f.write("1")

        # Setup NAT for outbound traffic
        subprocess.run([
            "iptables", "-t", "nat", "-A", "POSTROUTING",
            "-s", self.config.subnet, "-j", "MASQUERADE"
        ], check=False)

    def connect(self, container_id: str, pid: int,
               port_mappings: dict = None) -> ContainerNetwork:
        \"\"\"Connect container to bridge network.\"\"\"
        # Allocate IP
        ip = self._allocate_ip()

        # Create veth pair
        veth_host = f"veth{container_id[:8]}"
        veth_container = "eth0"

        subprocess.run([
            "ip", "link", "add", veth_host, "type", "veth",
            "peer", "name", veth_container
        ], check=True)

        # Attach host end to bridge
        subprocess.run([
            "ip", "link", "set", veth_host, "master", self.config.bridge_name
        ], check=True)
        subprocess.run(["ip", "link", "set", veth_host, "up"], check=True)

        # Move container end to container's network namespace
        subprocess.run([
            "ip", "link", "set", veth_container, "netns", str(pid)
        ], check=True)

        # Configure container interface (run in container's netns)
        subprocess.run([
            "nsenter", "-t", str(pid), "-n",
            "ip", "addr", "add", f"{ip}/16", "dev", veth_container
        ], check=True)

        subprocess.run([
            "nsenter", "-t", str(pid), "-n",
            "ip", "link", "set", veth_container, "up"
        ], check=True)

        subprocess.run([
            "nsenter", "-t", str(pid), "-n",
            "ip", "link", "set", "lo", "up"
        ], check=True)

        # Set default route
        gateway = self.config.bridge_ip.split('/')[0]
        subprocess.run([
            "nsenter", "-t", str(pid), "-n",
            "ip", "route", "add", "default", "via", gateway
        ], check=True)

        # Setup port mappings
        if port_mappings:
            for host_port, container_port in port_mappings.items():
                self._add_port_mapping(ip, host_port, container_port)

        return ContainerNetwork(
            container_id=container_id,
            ip_address=str(ip),
            mac_address=self._get_mac(veth_host),
            veth_host=veth_host,
            veth_container=veth_container,
            port_mappings=port_mappings
        )

    def _allocate_ip(self) -> ipaddress.IPv4Address:
        for ip in self.subnet.hosts():
            if ip not in self.allocated_ips and str(ip) != self.config.bridge_ip.split('/')[0]:
                self.allocated_ips.add(ip)
                return ip
        raise RuntimeError("No available IP addresses")

    def _add_port_mapping(self, container_ip: str, host_port: int,
                         container_port: int):
        subprocess.run([
            "iptables", "-t", "nat", "-A", "PREROUTING",
            "-p", "tcp", "--dport", str(host_port),
            "-j", "DNAT", "--to-destination", f"{container_ip}:{container_port}"
        ], check=True)

    def disconnect(self, network: ContainerNetwork):
        \"\"\"Remove container from network.\"\"\"
        # Remove port mappings
        if network.port_mappings:
            for host_port, container_port in network.port_mappings.items():
                subprocess.run([
                    "iptables", "-t", "nat", "-D", "PREROUTING",
                    "-p", "tcp", "--dport", str(host_port),
                    "-j", "DNAT", "--to-destination",
                    f"{network.ip_address}:{container_port}"
                ], check=False)

        # Delete veth pair
        subprocess.run([
            "ip", "link", "delete", network.veth_host
        ], check=False)

        # Release IP
        ip = ipaddress.ip_address(network.ip_address)
        self.allocated_ips.discard(ip)

    def _get_mac(self, interface: str) -> str:
        result = subprocess.run(
            ["cat", f"/sys/class/net/{interface}/address"],
            capture_output=True, text=True
        )
        return result.stdout.strip()
```"""
                },
                "pitfalls": [
                    "Container loses network if host veth deleted before container end",
                    "iptables rules persist after container death",
                    "MTU mismatch causes packet fragmentation issues"
                ]
            }
        ]
    },

    "service-mesh": {
        "name": "Service Mesh Sidecar",
        "description": "Build a service mesh sidecar proxy handling service discovery, load balancing, circuit breaking, and mTLS.",
        "why_important": "Service meshes like Istio and Linkerd are essential for microservices. Understanding the proxy layer helps with debugging and optimization.",
        "difficulty": "advanced",
        "tags": ["microservices", "networking", "distributed-systems"],
        "estimated_hours": 55,
        "prerequisites": ["api-gateway", "circuit-breaker"],
        "learning_outcomes": [
            "Implement transparent traffic interception",
            "Build service discovery integration",
            "Handle mTLS between services",
            "Implement advanced load balancing"
        ],
        "milestones": [
            {
                "name": "Traffic Interception",
                "description": "Intercept inbound and outbound traffic transparently using iptables redirect rules.",
                "hints": {
                    "level1": "Use iptables REDIRECT to send traffic to proxy port.",
                    "level2": "Preserve original destination using SO_ORIGINAL_DST.",
                    "level3": """```python
import socket
import struct
import subprocess
from dataclasses import dataclass

SO_ORIGINAL_DST = 80

@dataclass
class InterceptConfig:
    inbound_port: int = 15001
    outbound_port: int = 15006
    proxy_uid: int = 1337
    exclude_ports: list = None

class TrafficInterceptor:
    def __init__(self, config: InterceptConfig = None):
        self.config = config or InterceptConfig()

    def setup_iptables(self):
        \"\"\"Setup iptables rules for traffic interception.\"\"\"
        # Create PROXY_REDIRECT chain
        subprocess.run([
            "iptables", "-t", "nat", "-N", "PROXY_REDIRECT"
        ], check=False)

        # Redirect to proxy
        subprocess.run([
            "iptables", "-t", "nat", "-A", "PROXY_REDIRECT",
            "-p", "tcp", "-j", "REDIRECT", "--to-port", str(self.config.outbound_port)
        ], check=True)

        # Create PROXY_OUTPUT chain for outbound
        subprocess.run([
            "iptables", "-t", "nat", "-N", "PROXY_OUTPUT"
        ], check=False)

        # Exclude proxy's own traffic (prevent loops)
        subprocess.run([
            "iptables", "-t", "nat", "-A", "PROXY_OUTPUT",
            "-m", "owner", "--uid-owner", str(self.config.proxy_uid),
            "-j", "RETURN"
        ], check=True)

        # Exclude localhost
        subprocess.run([
            "iptables", "-t", "nat", "-A", "PROXY_OUTPUT",
            "-d", "127.0.0.1/32", "-j", "RETURN"
        ], check=True)

        # Exclude specific ports
        for port in (self.config.exclude_ports or []):
            subprocess.run([
                "iptables", "-t", "nat", "-A", "PROXY_OUTPUT",
                "-p", "tcp", "--dport", str(port), "-j", "RETURN"
            ], check=True)

        # Send everything else to proxy
        subprocess.run([
            "iptables", "-t", "nat", "-A", "PROXY_OUTPUT",
            "-j", "PROXY_REDIRECT"
        ], check=True)

        # Apply to OUTPUT chain
        subprocess.run([
            "iptables", "-t", "nat", "-A", "OUTPUT",
            "-p", "tcp", "-j", "PROXY_OUTPUT"
        ], check=True)

        # Setup inbound interception
        subprocess.run([
            "iptables", "-t", "nat", "-A", "PREROUTING",
            "-p", "tcp", "-j", "REDIRECT", "--to-port", str(self.config.inbound_port)
        ], check=True)

    def get_original_dest(self, client_socket) -> tuple[str, int]:
        \"\"\"Get original destination from redirected socket.\"\"\"
        # Get original destination using SO_ORIGINAL_DST
        dst = client_socket.getsockopt(socket.SOL_IP, SO_ORIGINAL_DST, 16)

        # Parse sockaddr_in structure
        port = struct.unpack('>H', dst[2:4])[0]
        ip = socket.inet_ntoa(dst[4:8])

        return ip, port

    def cleanup(self):
        \"\"\"Remove iptables rules.\"\"\"
        subprocess.run([
            "iptables", "-t", "nat", "-D", "OUTPUT", "-p", "tcp", "-j", "PROXY_OUTPUT"
        ], check=False)
        subprocess.run([
            "iptables", "-t", "nat", "-F", "PROXY_OUTPUT"
        ], check=False)
        subprocess.run([
            "iptables", "-t", "nat", "-X", "PROXY_OUTPUT"
        ], check=False)
        subprocess.run([
            "iptables", "-t", "nat", "-F", "PROXY_REDIRECT"
        ], check=False)
        subprocess.run([
            "iptables", "-t", "nat", "-X", "PROXY_REDIRECT"
        ], check=False)
```"""
                },
                "pitfalls": [
                    "Redirect loop if proxy traffic not excluded",
                    "SO_ORIGINAL_DST not available on all systems",
                    "IPv6 requires separate ip6tables rules"
                ]
            },
            {
                "name": "Service Discovery Integration",
                "description": "Integrate with service discovery (Consul, Kubernetes) to resolve service names to endpoints.",
                "hints": {
                    "level1": "Watch service registry for endpoint changes.",
                    "level2": "Cache endpoints locally, handle stale entries.",
                    "level3": """```python
import asyncio
import aiohttp
from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod

@dataclass
class Endpoint:
    address: str
    port: int
    weight: int = 100
    healthy: bool = True
    metadata: dict = field(default_factory=dict)

@dataclass
class Service:
    name: str
    endpoints: list[Endpoint] = field(default_factory=list)
    version: str = ""

class ServiceDiscovery(ABC):
    @abstractmethod
    async def get_service(self, name: str) -> Optional[Service]:
        pass

    @abstractmethod
    async def watch(self, name: str, callback):
        pass

class KubernetesDiscovery(ServiceDiscovery):
    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self.base_url = "https://kubernetes.default.svc"
        self._cache: dict[str, Service] = {}
        self._watchers: dict[str, asyncio.Task] = {}

    async def get_service(self, name: str) -> Optional[Service]:
        if name in self._cache:
            return self._cache[name]

        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/api/v1/namespaces/{self.namespace}/endpoints/{name}"
            headers = {"Authorization": f"Bearer {self._get_token()}"}

            async with session.get(url, headers=headers, ssl=False) as resp:
                if resp.status != 200:
                    return None

                data = await resp.json()
                service = self._parse_endpoints(name, data)
                self._cache[name] = service
                return service

    async def watch(self, name: str, callback):
        \"\"\"Watch for endpoint changes.\"\"\"
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/api/v1/namespaces/{self.namespace}/endpoints"
            params = {"watch": "true", "fieldSelector": f"metadata.name={name}"}
            headers = {"Authorization": f"Bearer {self._get_token()}"}

            async with session.get(url, params=params, headers=headers, ssl=False) as resp:
                async for line in resp.content:
                    if line:
                        import json
                        event = json.loads(line)
                        service = self._parse_endpoints(name, event['object'])
                        self._cache[name] = service
                        await callback(service)

    def _parse_endpoints(self, name: str, data: dict) -> Service:
        endpoints = []
        for subset in data.get('subsets', []):
            addresses = subset.get('addresses', [])
            ports = subset.get('ports', [])

            for addr in addresses:
                for port in ports:
                    endpoints.append(Endpoint(
                        address=addr['ip'],
                        port=port['port'],
                        metadata={
                            'node': addr.get('nodeName'),
                            'pod': addr.get('targetRef', {}).get('name')
                        }
                    ))

        return Service(name=name, endpoints=endpoints)

    def _get_token(self) -> str:
        with open('/var/run/secrets/kubernetes.io/serviceaccount/token') as f:
            return f.read()

class ServiceRegistry:
    def __init__(self, discovery: ServiceDiscovery):
        self.discovery = discovery
        self.services: dict[str, Service] = {}
        self._watch_tasks: dict[str, asyncio.Task] = {}

    async def resolve(self, service_name: str) -> list[Endpoint]:
        \"\"\"Resolve service name to healthy endpoints.\"\"\"
        if service_name not in self.services:
            service = await self.discovery.get_service(service_name)
            if service:
                self.services[service_name] = service
                self._start_watch(service_name)

        service = self.services.get(service_name)
        if not service:
            return []

        return [ep for ep in service.endpoints if ep.healthy]

    def _start_watch(self, service_name: str):
        if service_name not in self._watch_tasks:
            self._watch_tasks[service_name] = asyncio.create_task(
                self.discovery.watch(service_name, self._on_update)
            )

    async def _on_update(self, service: Service):
        self.services[service.name] = service
```"""
                },
                "pitfalls": [
                    "Watch connection drops require reconnection logic",
                    "Stale cache serves dead endpoints",
                    "DNS caching conflicts with dynamic discovery"
                ]
            },
            {
                "name": "mTLS and Certificate Management",
                "description": "Implement mutual TLS between services with automatic certificate rotation.",
                "hints": {
                    "level1": "Each service needs its own certificate signed by mesh CA.",
                    "level2": "Rotate certificates before expiry, handle during active connections.",
                    "level3": """```python
import ssl
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import subprocess

@dataclass
class Certificate:
    cert_path: str
    key_path: str
    ca_path: str
    expiry: datetime
    service_name: str

class CertificateManager:
    def __init__(self, service_name: str, cert_dir: str = "/etc/certs"):
        self.service_name = service_name
        self.cert_dir = Path(cert_dir)
        self.current_cert: Certificate = None
        self._rotation_task = None
        self.rotation_threshold = timedelta(hours=1)

    async def initialize(self):
        \"\"\"Load or request initial certificate.\"\"\"
        cert_path = self.cert_dir / "cert.pem"
        key_path = self.cert_dir / "key.pem"
        ca_path = self.cert_dir / "ca.pem"

        if not cert_path.exists():
            await self._request_certificate()

        self.current_cert = Certificate(
            cert_path=str(cert_path),
            key_path=str(key_path),
            ca_path=str(ca_path),
            expiry=self._get_cert_expiry(cert_path),
            service_name=self.service_name
        )

        self._rotation_task = asyncio.create_task(self._rotation_loop())

    def get_ssl_context(self, server: bool = False) -> ssl.SSLContext:
        \"\"\"Get SSL context for mTLS connections.\"\"\"
        if server:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        else:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

        # Load certificate and key
        ctx.load_cert_chain(
            self.current_cert.cert_path,
            self.current_cert.key_path
        )

        # Load CA for peer verification
        ctx.load_verify_locations(self.current_cert.ca_path)

        # Require client certificate (mTLS)
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.check_hostname = not server

        return ctx

    async def _rotation_loop(self):
        \"\"\"Check and rotate certificates before expiry.\"\"\"
        while True:
            time_to_expiry = self.current_cert.expiry - datetime.utcnow()

            if time_to_expiry < self.rotation_threshold:
                await self._rotate_certificate()

            # Check every minute
            await asyncio.sleep(60)

    async def _rotate_certificate(self):
        \"\"\"Request new certificate and hot-swap.\"\"\"
        # Request new certificate
        new_cert_path = self.cert_dir / "cert-new.pem"
        new_key_path = self.cert_dir / "key-new.pem"

        await self._request_certificate(
            cert_path=new_cert_path,
            key_path=new_key_path
        )

        # Atomic swap
        import shutil
        shutil.move(new_cert_path, self.cert_dir / "cert.pem")
        shutil.move(new_key_path, self.cert_dir / "key.pem")

        # Update current cert
        self.current_cert.expiry = self._get_cert_expiry(self.cert_dir / "cert.pem")

    async def _request_certificate(self, cert_path: Path = None,
                                   key_path: Path = None):
        \"\"\"Request certificate from mesh CA (e.g., SPIFFE/SPIRE).\"\"\"
        cert_path = cert_path or self.cert_dir / "cert.pem"
        key_path = key_path or self.cert_dir / "key.pem"

        # Generate CSR
        subprocess.run([
            "openssl", "req", "-new", "-newkey", "rsa:2048",
            "-nodes", "-keyout", str(key_path),
            "-out", "/tmp/csr.pem",
            "-subj", f"/CN={self.service_name}"
        ], check=True)

        # In real implementation, send CSR to CA API
        # For now, self-sign (development only)
        subprocess.run([
            "openssl", "x509", "-req",
            "-in", "/tmp/csr.pem",
            "-CA", str(self.cert_dir / "ca.pem"),
            "-CAkey", str(self.cert_dir / "ca-key.pem"),
            "-CAcreateserial",
            "-out", str(cert_path),
            "-days", "1"
        ], check=True)

    def _get_cert_expiry(self, cert_path: Path) -> datetime:
        result = subprocess.run([
            "openssl", "x509", "-enddate", "-noout", "-in", str(cert_path)
        ], capture_output=True, text=True)

        # Parse: notAfter=Jan  1 00:00:00 2024 GMT
        date_str = result.stdout.strip().split('=')[1]
        return datetime.strptime(date_str, "%b %d %H:%M:%S %Y %Z")

def verify_peer_identity(ssl_socket, expected_service: str) -> bool:
    \"\"\"Verify peer certificate matches expected service identity.\"\"\"
    cert = ssl_socket.getpeercert()

    # Check Common Name
    subject = dict(x[0] for x in cert['subject'])
    cn = subject.get('commonName', '')

    # Check SAN (Subject Alternative Names)
    san = cert.get('subjectAltName', [])
    dns_names = [name for type_, name in san if type_ == 'DNS']
    uri_names = [name for type_, name in san if type_ == 'URI']

    # Verify identity (SPIFFE format: spiffe://trust-domain/service-name)
    expected_spiffe = f"spiffe://mesh.local/{expected_service}"

    return (cn == expected_service or
            expected_service in dns_names or
            expected_spiffe in uri_names)
```"""
                },
                "pitfalls": [
                    "Certificate rotation during active requests causes failures",
                    "Clock skew makes valid certificates appear expired",
                    "Missing SAN in certificate breaks modern TLS verification"
                ]
            },
            {
                "name": "Load Balancing Algorithms",
                "description": "Implement advanced load balancing including round-robin, least connections, weighted, and consistent hashing.",
                "hints": {
                    "level1": "Round-robin rotates through endpoints sequentially.",
                    "level2": "Least connections requires tracking active connection count.",
                    "level3": """```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import bisect
import random
import threading

@dataclass
class EndpointStats:
    address: str
    active_connections: int = 0
    total_requests: int = 0
    failures: int = 0
    avg_latency_ms: float = 0
    weight: int = 100

class LoadBalancer(ABC):
    @abstractmethod
    def select(self, endpoints: list[EndpointStats], key: str = None) -> EndpointStats:
        pass

class RoundRobinBalancer(LoadBalancer):
    def __init__(self):
        self._index = 0
        self._lock = threading.Lock()

    def select(self, endpoints: list[EndpointStats], key: str = None) -> EndpointStats:
        if not endpoints:
            return None

        with self._lock:
            endpoint = endpoints[self._index % len(endpoints)]
            self._index += 1
            return endpoint

class WeightedRoundRobinBalancer(LoadBalancer):
    def __init__(self):
        self._current_weight = 0
        self._index = 0
        self._lock = threading.Lock()

    def select(self, endpoints: list[EndpointStats], key: str = None) -> EndpointStats:
        if not endpoints:
            return None

        with self._lock:
            max_weight = max(ep.weight for ep in endpoints)
            gcd_weight = self._gcd_of_weights([ep.weight for ep in endpoints])

            while True:
                self._index = (self._index + 1) % len(endpoints)
                if self._index == 0:
                    self._current_weight -= gcd_weight
                    if self._current_weight <= 0:
                        self._current_weight = max_weight

                if endpoints[self._index].weight >= self._current_weight:
                    return endpoints[self._index]

    def _gcd_of_weights(self, weights: list[int]) -> int:
        from math import gcd
        from functools import reduce
        return reduce(gcd, weights)

class LeastConnectionsBalancer(LoadBalancer):
    def select(self, endpoints: list[EndpointStats], key: str = None) -> EndpointStats:
        if not endpoints:
            return None

        # Select endpoint with fewest active connections
        # Weighted by: connections / weight
        return min(
            endpoints,
            key=lambda ep: ep.active_connections / max(ep.weight, 1)
        )

class ConsistentHashBalancer(LoadBalancer):
    def __init__(self, replicas: int = 150):
        self.replicas = replicas
        self._ring: list[tuple[int, str]] = []
        self._endpoints: dict[str, EndpointStats] = {}

    def update_endpoints(self, endpoints: list[EndpointStats]):
        \"\"\"Rebuild hash ring when endpoints change.\"\"\"
        self._ring = []
        self._endpoints = {ep.address: ep for ep in endpoints}

        for ep in endpoints:
            for i in range(self.replicas):
                key = f"{ep.address}:{i}"
                hash_val = self._hash(key)
                bisect.insort(self._ring, (hash_val, ep.address))

    def select(self, endpoints: list[EndpointStats], key: str = None) -> EndpointStats:
        if not self._ring:
            self.update_endpoints(endpoints)

        if not key:
            key = str(random.random())

        hash_val = self._hash(key)

        # Binary search for first hash >= key hash
        idx = bisect.bisect_left(self._ring, (hash_val, ""))
        if idx >= len(self._ring):
            idx = 0

        address = self._ring[idx][1]
        return self._endpoints.get(address)

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

class P2CBalancer(LoadBalancer):
    \"\"\"Power of Two Choices - picks best of 2 random endpoints.\"\"\"

    def select(self, endpoints: list[EndpointStats], key: str = None) -> EndpointStats:
        if not endpoints:
            return None
        if len(endpoints) == 1:
            return endpoints[0]

        # Pick two random endpoints
        a, b = random.sample(endpoints, 2)

        # Return one with fewer connections (weighted)
        score_a = a.active_connections / max(a.weight, 1)
        score_b = b.active_connections / max(b.weight, 1)

        return a if score_a <= score_b else b

class AdaptiveBalancer(LoadBalancer):
    \"\"\"Balancer that considers latency and error rates.\"\"\"

    def select(self, endpoints: list[EndpointStats], key: str = None) -> EndpointStats:
        if not endpoints:
            return None

        # Score based on multiple factors
        def score(ep: EndpointStats) -> float:
            # Lower is better
            latency_factor = ep.avg_latency_ms / 100  # Normalize
            error_rate = ep.failures / max(ep.total_requests, 1)
            connection_factor = ep.active_connections / max(ep.weight, 1)

            return (
                latency_factor * 0.3 +
                error_rate * 0.5 +
                connection_factor * 0.2
            )

        return min(endpoints, key=score)
```"""
                },
                "pitfalls": [
                    "Consistent hashing needs ring rebuild on endpoint change",
                    "Least connections can thundering herd to recovered endpoint",
                    "Weight=0 causes division by zero"
                ]
            }
        ]
    },

    "infrastructure-as-code": {
        "name": "Infrastructure as Code Engine",
        "description": "Build an IaC engine that parses declarative configs, manages state, computes diffs, and applies changes to infrastructure.",
        "why_important": "Understanding IaC internals (like Terraform) helps debug state issues, write better modules, and build custom providers.",
        "difficulty": "advanced",
        "tags": ["devops", "infrastructure", "automation"],
        "estimated_hours": 50,
        "prerequisites": ["ci-cd-pipeline"],
        "learning_outcomes": [
            "Design declarative configuration language",
            "Implement state management and locking",
            "Build dependency graph for resource ordering",
            "Handle provider abstraction for multi-cloud"
        ],
        "milestones": [
            {
                "name": "Configuration Parser",
                "description": "Parse HCL-like configuration files with resources, variables, outputs, and module references.",
                "hints": {
                    "level1": "Define grammar for resource blocks with attributes.",
                    "level2": "Handle variable interpolation and references.",
                    "level3": """```python
from dataclasses import dataclass, field
from typing import Any, Optional
import re

@dataclass
class Variable:
    name: str
    type: str = "string"
    default: Any = None
    description: str = ""

@dataclass
class Resource:
    type: str
    name: str
    attributes: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    count: int = 1
    provider: str = None

@dataclass
class Output:
    name: str
    value: Any
    description: str = ""

@dataclass
class Module:
    name: str
    source: str
    variables: dict = field(default_factory=dict)

@dataclass
class Configuration:
    variables: dict[str, Variable] = field(default_factory=dict)
    resources: dict[str, Resource] = field(default_factory=dict)
    outputs: dict[str, Output] = field(default_factory=dict)
    modules: dict[str, Module] = field(default_factory=dict)
    providers: dict[str, dict] = field(default_factory=dict)

class ConfigParser:
    def __init__(self):
        self.config = Configuration()
        self._var_pattern = re.compile(r'\\$\\{([^}]+)\\}')

    def parse(self, content: str) -> Configuration:
        \"\"\"Parse HCL-like configuration.\"\"\"
        import hcl2
        import json

        # Parse HCL to dict
        data = hcl2.loads(content)

        # Parse variables
        for var_block in data.get('variable', []):
            for name, attrs in var_block.items():
                self.config.variables[name] = Variable(
                    name=name,
                    type=attrs.get('type', 'string'),
                    default=attrs.get('default'),
                    description=attrs.get('description', '')
                )

        # Parse resources
        for res_block in data.get('resource', []):
            for res_type, resources in res_block.items():
                for res_name, attrs in resources.items():
                    resource_id = f"{res_type}.{res_name}"
                    self.config.resources[resource_id] = Resource(
                        type=res_type,
                        name=res_name,
                        attributes=attrs,
                        depends_on=attrs.pop('depends_on', []),
                        count=attrs.pop('count', 1),
                        provider=attrs.pop('provider', None)
                    )

        # Parse outputs
        for out_block in data.get('output', []):
            for name, attrs in out_block.items():
                self.config.outputs[name] = Output(
                    name=name,
                    value=attrs.get('value'),
                    description=attrs.get('description', '')
                )

        # Parse modules
        for mod_block in data.get('module', []):
            for name, attrs in mod_block.items():
                self.config.modules[name] = Module(
                    name=name,
                    source=attrs.pop('source'),
                    variables=attrs
                )

        return self.config

    def resolve_references(self, value: Any, context: dict) -> Any:
        \"\"\"Resolve variable and resource references.\"\"\"
        if isinstance(value, str):
            return self._resolve_string(value, context)
        elif isinstance(value, list):
            return [self.resolve_references(v, context) for v in value]
        elif isinstance(value, dict):
            return {k: self.resolve_references(v, context) for k, v in value.items()}
        return value

    def _resolve_string(self, value: str, context: dict) -> Any:
        \"\"\"Resolve interpolations in string value.\"\"\"
        def replacer(match):
            ref = match.group(1)
            return str(self._resolve_reference(ref, context))

        # Check if entire value is a single reference
        if value.startswith('${') and value.endswith('}'):
            ref = value[2:-1]
            return self._resolve_reference(ref, context)

        return self._var_pattern.sub(replacer, value)

    def _resolve_reference(self, ref: str, context: dict) -> Any:
        \"\"\"Resolve a single reference like var.name or aws_instance.web.id\"\"\"
        parts = ref.split('.')

        if parts[0] == 'var':
            return context.get('variables', {}).get(parts[1])
        elif parts[0] == 'local':
            return context.get('locals', {}).get(parts[1])
        elif parts[0] == 'module':
            return context.get('modules', {}).get(parts[1], {}).get(parts[2])
        else:
            # Resource reference
            resource_id = f"{parts[0]}.{parts[1]}"
            resource = context.get('resources', {}).get(resource_id, {})
            if len(parts) > 2:
                return resource.get(parts[2])
            return resource

        return None
```"""
                },
                "pitfalls": [
                    "Circular references cause infinite resolution loop",
                    "Interpolation in count creates chicken-egg problem",
                    "Module source paths need normalization"
                ]
            },
            {
                "name": "State Management",
                "description": "Implement state file tracking of deployed resources with locking for concurrent access.",
                "hints": {
                    "level1": "Store resource IDs and attributes after creation.",
                    "level2": "Use file or remote locking to prevent concurrent modifications.",
                    "level3": """```python
import json
import hashlib
import fcntl
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import asyncio

@dataclass
class ResourceState:
    type: str
    name: str
    id: str
    attributes: dict
    dependencies: list[str] = field(default_factory=list)
    provider: str = ""

@dataclass
class State:
    version: int = 1
    serial: int = 0
    lineage: str = ""
    resources: dict[str, ResourceState] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)

class StateManager:
    def __init__(self, state_path: str = "terraform.tfstate"):
        self.state_path = Path(state_path)
        self.lock_path = Path(f"{state_path}.lock")
        self.state: State = None
        self._lock_fd = None

    def load(self) -> State:
        \"\"\"Load state from file.\"\"\"
        if self.state_path.exists():
            with open(self.state_path) as f:
                data = json.load(f)
                self.state = State(
                    version=data.get('version', 1),
                    serial=data.get('serial', 0),
                    lineage=data.get('lineage', ''),
                    resources={
                        k: ResourceState(**v)
                        for k, v in data.get('resources', {}).items()
                    },
                    outputs=data.get('outputs', {})
                )
        else:
            import uuid
            self.state = State(lineage=str(uuid.uuid4()))

        return self.state

    def save(self):
        \"\"\"Save state to file.\"\"\"
        self.state.serial += 1

        data = {
            'version': self.state.version,
            'serial': self.state.serial,
            'lineage': self.state.lineage,
            'resources': {
                k: asdict(v) for k, v in self.state.resources.items()
            },
            'outputs': self.state.outputs
        }

        # Write to temp file first
        temp_path = self.state_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Atomic rename
        temp_path.rename(self.state_path)

    def lock(self, timeout: float = 60) -> bool:
        \"\"\"Acquire state lock.\"\"\"
        self._lock_fd = open(self.lock_path, 'w')

        start = datetime.now()
        while True:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Write lock info
                lock_info = {
                    'locked_at': datetime.utcnow().isoformat(),
                    'pid': os.getpid(),
                    'hostname': socket.gethostname()
                }
                self._lock_fd.write(json.dumps(lock_info))
                self._lock_fd.flush()
                return True
            except BlockingIOError:
                if (datetime.now() - start).total_seconds() > timeout:
                    return False
                import time
                time.sleep(1)

    def unlock(self):
        \"\"\"Release state lock.\"\"\"
        if self._lock_fd:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            self._lock_fd.close()
            self._lock_fd = None
            self.lock_path.unlink(missing_ok=True)

    def get_resource(self, resource_id: str) -> Optional[ResourceState]:
        return self.state.resources.get(resource_id)

    def set_resource(self, resource_id: str, resource: ResourceState):
        self.state.resources[resource_id] = resource

    def remove_resource(self, resource_id: str):
        self.state.resources.pop(resource_id, None)

    def compute_checksum(self) -> str:
        \"\"\"Compute state checksum for change detection.\"\"\"
        data = json.dumps(asdict(self.state), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

class RemoteStateBackend:
    \"\"\"S3 backend for remote state storage.\"\"\"

    def __init__(self, bucket: str, key: str, region: str):
        self.bucket = bucket
        self.key = key
        self.region = region
        self.lock_table = "terraform-locks"  # DynamoDB table

    async def load(self) -> State:
        import aioboto3
        session = aioboto3.Session()

        async with session.client('s3', region_name=self.region) as s3:
            try:
                response = await s3.get_object(Bucket=self.bucket, Key=self.key)
                data = json.loads(await response['Body'].read())
                return State(**data)
            except s3.exceptions.NoSuchKey:
                return State(lineage=str(uuid.uuid4()))

    async def save(self, state: State):
        import aioboto3
        session = aioboto3.Session()

        async with session.client('s3', region_name=self.region) as s3:
            await s3.put_object(
                Bucket=self.bucket,
                Key=self.key,
                Body=json.dumps(asdict(state), indent=2),
                ContentType='application/json'
            )

    async def lock(self, lock_id: str) -> bool:
        import aioboto3
        session = aioboto3.Session()

        async with session.client('dynamodb', region_name=self.region) as ddb:
            try:
                await ddb.put_item(
                    TableName=self.lock_table,
                    Item={
                        'LockID': {'S': f"{self.bucket}/{self.key}"},
                        'Info': {'S': json.dumps({'id': lock_id})}
                    },
                    ConditionExpression='attribute_not_exists(LockID)'
                )
                return True
            except ddb.exceptions.ConditionalCheckFailedException:
                return False
```"""
                },
                "pitfalls": [
                    "State corruption on partial write (use atomic rename)",
                    "Stale lock from crashed process blocks everyone",
                    "Remote state race condition between read and write"
                ]
            },
            {
                "name": "Dependency Graph & Planning",
                "description": "Build resource dependency graph and generate execution plan with create, update, delete operations.",
                "hints": {
                    "level1": "Build DAG from explicit depends_on and implicit references.",
                    "level2": "Compare desired state vs current state to determine actions.",
                    "level3": """```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import graphlib

class Action(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    REPLACE = "replace"  # delete + create
    NO_OP = "no-op"

@dataclass
class ResourceChange:
    resource_id: str
    action: Action
    before: dict = None
    after: dict = None
    reason: str = ""

@dataclass
class Plan:
    changes: list[ResourceChange] = field(default_factory=list)
    outputs: dict = field(default_factory=dict)

class DependencyGraph:
    def __init__(self):
        self.graph: dict[str, set[str]] = {}

    def add_resource(self, resource_id: str, depends_on: list[str] = None):
        self.graph[resource_id] = set(depends_on or [])

    def add_dependency(self, resource_id: str, depends_on: str):
        self.graph.setdefault(resource_id, set()).add(depends_on)

    def get_order(self) -> list[str]:
        \"\"\"Return resources in dependency order.\"\"\"
        ts = graphlib.TopologicalSorter(self.graph)
        return list(ts.static_order())

    def get_reverse_order(self) -> list[str]:
        \"\"\"Return resources in reverse order (for deletion).\"\"\"
        return list(reversed(self.get_order()))

    def get_dependents(self, resource_id: str) -> set[str]:
        \"\"\"Get all resources that depend on this one.\"\"\"
        dependents = set()
        for res_id, deps in self.graph.items():
            if resource_id in deps:
                dependents.add(res_id)
        return dependents

class Planner:
    def __init__(self, config: Configuration, state: State):
        self.config = config
        self.state = state
        self.graph = DependencyGraph()

    def plan(self) -> Plan:
        \"\"\"Generate execution plan.\"\"\"
        plan = Plan()

        # Build dependency graph
        self._build_graph()

        # Get desired resource IDs
        desired_ids = set(self.config.resources.keys())
        current_ids = set(self.state.resources.keys())

        # Resources to create
        to_create = desired_ids - current_ids
        # Resources to delete
        to_delete = current_ids - desired_ids
        # Resources to potentially update
        to_check = desired_ids & current_ids

        # Process deletions (reverse dependency order)
        for resource_id in self.graph.get_reverse_order():
            if resource_id in to_delete:
                plan.changes.append(ResourceChange(
                    resource_id=resource_id,
                    action=Action.DELETE,
                    before=self.state.resources[resource_id].attributes
                ))

        # Process creates and updates (dependency order)
        for resource_id in self.graph.get_order():
            if resource_id in to_create:
                plan.changes.append(ResourceChange(
                    resource_id=resource_id,
                    action=Action.CREATE,
                    after=self.config.resources[resource_id].attributes
                ))
            elif resource_id in to_check:
                change = self._diff_resource(resource_id)
                if change:
                    plan.changes.append(change)

        return plan

    def _build_graph(self):
        \"\"\"Build dependency graph from configuration.\"\"\"
        for resource_id, resource in self.config.resources.items():
            # Explicit dependencies
            self.graph.add_resource(resource_id, resource.depends_on)

            # Implicit dependencies from references
            refs = self._find_references(resource.attributes)
            for ref in refs:
                if ref in self.config.resources:
                    self.graph.add_dependency(resource_id, ref)

    def _find_references(self, value: Any, refs: set = None) -> set[str]:
        \"\"\"Find resource references in attributes.\"\"\"
        refs = refs or set()

        if isinstance(value, str):
            import re
            pattern = r'\\$\\{([a-z_]+\\.[a-z_][a-z0-9_]*)'
            for match in re.finditer(pattern, value):
                refs.add(match.group(1))
        elif isinstance(value, list):
            for v in value:
                self._find_references(v, refs)
        elif isinstance(value, dict):
            for v in value.values():
                self._find_references(v, refs)

        return refs

    def _diff_resource(self, resource_id: str) -> Optional[ResourceChange]:
        \"\"\"Compare desired vs current state for a resource.\"\"\"
        desired = self.config.resources[resource_id]
        current = self.state.resources[resource_id]

        # Check if attributes changed
        if self._attributes_changed(desired.attributes, current.attributes):
            # Check if change requires replacement
            if self._requires_replacement(desired.type, desired.attributes, current.attributes):
                return ResourceChange(
                    resource_id=resource_id,
                    action=Action.REPLACE,
                    before=current.attributes,
                    after=desired.attributes,
                    reason="Force replacement attribute changed"
                )
            else:
                return ResourceChange(
                    resource_id=resource_id,
                    action=Action.UPDATE,
                    before=current.attributes,
                    after=desired.attributes
                )

        return None

    def _attributes_changed(self, desired: dict, current: dict) -> bool:
        \"\"\"Check if attributes have changed.\"\"\"
        # Simplified comparison - real implementation needs deep diff
        return desired != current

    def _requires_replacement(self, res_type: str, desired: dict, current: dict) -> bool:
        \"\"\"Check if change requires resource replacement.\"\"\"
        # Provider defines which attributes force replacement
        force_new_attrs = {
            'aws_instance': ['ami', 'instance_type'],
            'aws_vpc': ['cidr_block'],
        }

        attrs = force_new_attrs.get(res_type, [])
        for attr in attrs:
            if desired.get(attr) != current.get(attr):
                return True
        return False
```"""
                },
                "pitfalls": [
                    "Cycle in dependencies causes topological sort failure",
                    "Implicit dependency detection misses some patterns",
                    "Replace action must delete before create if unique constraint"
                ]
            },
            {
                "name": "Provider Abstraction",
                "description": "Build provider interface for abstracting cloud APIs with resource CRUD operations.",
                "hints": {
                    "level1": "Define standard interface for create, read, update, delete.",
                    "level2": "Handle resource schema validation and type coercion.",
                    "level3": """```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum

class AttributeType(Enum):
    STRING = "string"
    NUMBER = "number"
    BOOL = "bool"
    LIST = "list"
    MAP = "map"

@dataclass
class AttributeSchema:
    name: str
    type: AttributeType
    required: bool = False
    computed: bool = False  # Set by provider, not user
    force_new: bool = False  # Change requires replacement
    default: Any = None
    description: str = ""

@dataclass
class ResourceSchema:
    type_name: str
    attributes: dict[str, AttributeSchema] = field(default_factory=dict)
    create_timeout: int = 300
    update_timeout: int = 300
    delete_timeout: int = 300

@dataclass
class ResourceData:
    id: str = ""
    attributes: dict = field(default_factory=dict)

class Provider(ABC):
    @abstractmethod
    def get_schema(self) -> dict[str, ResourceSchema]:
        \"\"\"Return schemas for all resource types.\"\"\"
        pass

    @abstractmethod
    async def configure(self, config: dict):
        \"\"\"Configure provider with credentials and settings.\"\"\"
        pass

    @abstractmethod
    async def create(self, resource_type: str, config: dict) -> ResourceData:
        \"\"\"Create a new resource.\"\"\"
        pass

    @abstractmethod
    async def read(self, resource_type: str, id: str) -> Optional[ResourceData]:
        \"\"\"Read current state of resource.\"\"\"
        pass

    @abstractmethod
    async def update(self, resource_type: str, id: str,
                    old_config: dict, new_config: dict) -> ResourceData:
        \"\"\"Update existing resource.\"\"\"
        pass

    @abstractmethod
    async def delete(self, resource_type: str, id: str):
        \"\"\"Delete resource.\"\"\"
        pass

class AWSProvider(Provider):
    def __init__(self):
        self.session = None
        self._schemas = self._define_schemas()

    def _define_schemas(self) -> dict[str, ResourceSchema]:
        return {
            'aws_instance': ResourceSchema(
                type_name='aws_instance',
                attributes={
                    'ami': AttributeSchema('ami', AttributeType.STRING, required=True, force_new=True),
                    'instance_type': AttributeSchema('instance_type', AttributeType.STRING, required=True),
                    'tags': AttributeSchema('tags', AttributeType.MAP),
                    'id': AttributeSchema('id', AttributeType.STRING, computed=True),
                    'public_ip': AttributeSchema('public_ip', AttributeType.STRING, computed=True),
                    'private_ip': AttributeSchema('private_ip', AttributeType.STRING, computed=True),
                }
            ),
            'aws_vpc': ResourceSchema(
                type_name='aws_vpc',
                attributes={
                    'cidr_block': AttributeSchema('cidr_block', AttributeType.STRING, required=True, force_new=True),
                    'tags': AttributeSchema('tags', AttributeType.MAP),
                    'id': AttributeSchema('id', AttributeType.STRING, computed=True),
                }
            ),
        }

    def get_schema(self) -> dict[str, ResourceSchema]:
        return self._schemas

    async def configure(self, config: dict):
        import aioboto3
        self.session = aioboto3.Session(
            aws_access_key_id=config.get('access_key'),
            aws_secret_access_key=config.get('secret_key'),
            region_name=config.get('region', 'us-east-1')
        )

    async def create(self, resource_type: str, config: dict) -> ResourceData:
        if resource_type == 'aws_instance':
            return await self._create_instance(config)
        elif resource_type == 'aws_vpc':
            return await self._create_vpc(config)
        raise ValueError(f"Unknown resource type: {resource_type}")

    async def _create_instance(self, config: dict) -> ResourceData:
        async with self.session.client('ec2') as ec2:
            response = await ec2.run_instances(
                ImageId=config['ami'],
                InstanceType=config['instance_type'],
                MinCount=1,
                MaxCount=1,
                TagSpecifications=[{
                    'ResourceType': 'instance',
                    'Tags': [{'Key': k, 'Value': v} for k, v in config.get('tags', {}).items()]
                }] if config.get('tags') else []
            )

            instance = response['Instances'][0]

            # Wait for running state
            waiter = ec2.get_waiter('instance_running')
            await waiter.wait(InstanceIds=[instance['InstanceId']])

            # Get updated info
            desc = await ec2.describe_instances(InstanceIds=[instance['InstanceId']])
            instance = desc['Reservations'][0]['Instances'][0]

            return ResourceData(
                id=instance['InstanceId'],
                attributes={
                    'ami': instance['ImageId'],
                    'instance_type': instance['InstanceType'],
                    'public_ip': instance.get('PublicIpAddress', ''),
                    'private_ip': instance.get('PrivateIpAddress', ''),
                    'tags': {t['Key']: t['Value'] for t in instance.get('Tags', [])}
                }
            )

    async def read(self, resource_type: str, id: str) -> Optional[ResourceData]:
        if resource_type == 'aws_instance':
            async with self.session.client('ec2') as ec2:
                try:
                    response = await ec2.describe_instances(InstanceIds=[id])
                    if not response['Reservations']:
                        return None
                    instance = response['Reservations'][0]['Instances'][0]
                    return ResourceData(
                        id=id,
                        attributes={
                            'ami': instance['ImageId'],
                            'instance_type': instance['InstanceType'],
                            'public_ip': instance.get('PublicIpAddress', ''),
                            'private_ip': instance.get('PrivateIpAddress', ''),
                        }
                    )
                except Exception:
                    return None
        return None

    async def delete(self, resource_type: str, id: str):
        if resource_type == 'aws_instance':
            async with self.session.client('ec2') as ec2:
                await ec2.terminate_instances(InstanceIds=[id])
                waiter = ec2.get_waiter('instance_terminated')
                await waiter.wait(InstanceIds=[id])
```"""
                },
                "pitfalls": [
                    "API rate limits require retry with backoff",
                    "Eventual consistency means read after create may fail",
                    "Resource stuck in pending state needs timeout handling"
                ]
            }
        ]
    }
}

# Load and update YAML
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)

expert_projects = data.get('expert_projects', {})

for project_id, project_data in devops_projects.items():
    if project_id not in expert_projects:
        expert_projects[project_id] = project_data
        print(f"Added: {project_id}")
    else:
        print(f"Skipped (exists): {project_id}")

# Save
with open(yaml_path, 'w') as f:
    yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False, width=120)

print(f"\nTotal expert_projects: {len(expert_projects)}")

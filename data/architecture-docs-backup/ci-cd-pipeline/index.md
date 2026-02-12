# CI/CD Pipeline: Design Document


## Overview

This system orchestrates automated software delivery by parsing YAML pipeline definitions, executing jobs in isolated environments, managing artifacts between stages, and implementing deployment strategies like blue-green and canary deployments. The key architectural challenge is coordinating distributed execution while maintaining isolation, reliability, and the ability to recover from failures at any stage.


> This guide is meant to help you understand the big picture before diving into each milestone. Refer back to it whenever you need context on how components connect.


## Context and Problem Statement

> **Milestone(s):** This section establishes the foundational understanding that drives all subsequent milestones (1-4).

### Software Delivery Challenges

Think of software delivery like running a factory assembly line. In a traditional factory, each product moves through multiple stations—cutting, welding, painting, quality inspection, packaging—before reaching the customer. Each station has specific requirements: the right tools, trained operators, quality checks, and handoff procedures to the next station. When done manually, a single mistake at any station can halt the entire line, defective products slip through without proper inspection, and scaling up requires proportionally more human oversight and coordination.

Software delivery faces identical challenges, but with even greater complexity. **Manual deployment processes** represent the cottage industry approach to software manufacturing—artisanal, error-prone, and fundamentally unscalable. Every code change must traverse multiple environments (development, testing, staging, production), undergo various validation procedures (unit tests, integration tests, security scans, performance benchmarks), and coordinate with multiple systems (databases, load balancers, monitoring tools, rollback mechanisms).

The **human element** introduces systematic failure points that compound as systems grow. Consider a typical manual deployment scenario: a developer finishes a feature, manually builds the application on their local machine, uploads artifacts to a shared server, SSH into multiple production servers, stops existing services, replaces binaries, updates configuration files, restarts services, and manually verifies the deployment succeeded. Each step requires context switching, introduces opportunities for typos or missed steps, and creates a single point of failure where one person's mistake can bring down the entire system.

**Environment drift** represents another critical failure mode in manual processes. Development environments inevitably diverge from production environments due to different operating system versions, library updates, configuration variations, and accumulated changes over time. A deployment that works perfectly in development may fail catastrophically in production due to subtle environmental differences that only surface under load or with production data patterns.

**Coordination complexity** grows exponentially with team size and deployment frequency. Manual deployments require careful scheduling to avoid conflicts—two teams cannot safely deploy simultaneously without risking service disruption. This coordination overhead forces teams into infrequent, large-batch deployments that bundle many changes together, making it nearly impossible to identify the root cause when problems occur. The result is a deployment process that becomes increasingly risky and stressful as the organization grows.

> The fundamental insight driving CI/CD adoption is that software delivery problems are not primarily technical—they are process and coordination problems that require systematic automation to solve at scale.

**Automation challenges** emerge immediately when teams attempt to systematize their delivery processes. The first challenge is **pipeline definition complexity**. Teams need a way to express their build, test, and deployment procedures as code, but these procedures often involve complex dependencies, conditional logic, parallel execution, and integration with multiple external systems. Creating a pipeline definition language that is both expressive enough to handle real-world scenarios and simple enough for developers to understand and maintain requires careful balance between flexibility and simplicity.

**Execution environment isolation** presents the second major challenge. Each step in a pipeline may require different tools, runtime versions, dependencies, or system configurations. Without proper isolation, pipeline steps can interfere with each other—a test that installs a dependency can break a subsequent build step, or leftover processes from a failed job can consume resources and cause mysterious failures in later executions. Achieving true isolation while maintaining reasonable performance and resource utilization requires sophisticated containerization and resource management.

**Artifact management** introduces storage and coordination challenges that don't exist in manual processes. Pipeline stages must pass build outputs, test reports, container images, and deployment packages between steps that may execute on different machines at different times. These artifacts must be versioned, validated for integrity, stored efficiently, and cleaned up automatically to prevent storage exhaustion. The challenge intensifies when artifacts become large (gigabyte-scale container images or datasets) and must be transferred efficiently without overwhelming network resources.

**State management and failure recovery** represent the most complex automation challenges. Unlike manual processes where humans can assess situations and adapt their approach, automated pipelines must handle every possible failure mode explicitly. Network timeouts, resource exhaustion, flaky tests, deployment target unavailability, and partial failures all require specific detection and recovery strategies. The system must maintain enough state to enable intelligent retry behavior while avoiding the complexity of distributed state management that could introduce additional failure modes.

| Challenge Category | Manual Process Problem | Automation Complexity | Example Failure Mode |
|-------------------|----------------------|---------------------|-------------------|
| Process Consistency | Human error, forgotten steps | Pipeline definition parsing, validation | Developer forgets to run security scan, vulnerability reaches production |
| Environment Management | Configuration drift between environments | Container orchestration, dependency isolation | Library version mismatch causes production crash |
| Coordination | Schedule conflicts between teams | Parallel execution, resource contention | Two deployments modify same database schema simultaneously |
| Artifact Handling | Manual file transfers, version confusion | Storage backends, integrity verification | Wrong artifact version deployed due to naming collision |
| Failure Recovery | Ad-hoc troubleshooting, inconsistent recovery | Comprehensive error handling, state reconciliation | Partial deployment leaves system in inconsistent state |

### Existing CI/CD Solutions

The CI/CD landscape offers several established solutions, each representing different architectural philosophies and trade-off decisions. Understanding these existing approaches illuminates the design space and helps identify opportunities for improvement or specialization.

**Jenkins** pioneered the modern CI/CD space and represents the "universal automation server" approach. Jenkins treats CI/CD as a general-purpose job scheduling and execution problem. Its plugin ecosystem contains thousands of integrations, enabling teams to automate virtually any workflow. Jenkins uses a **master-agent architecture** where a central master node coordinates work distribution to multiple agent nodes that execute jobs. Pipeline definitions use either the older XML-based job configuration or the newer Pipeline-as-Code approach with Groovy-based Jenkinsfiles.

Jenkins' plugin architecture creates both its greatest strength and its most significant weakness. The plugin ecosystem enables incredible flexibility—teams can integrate with any tool, technology stack, or workflow pattern. However, plugin management becomes a significant operational burden as organizations scale. Plugin updates can introduce breaking changes, security vulnerabilities in plugins create attack vectors, and the interdependencies between plugins create a complex web of compatibility requirements that must be managed carefully.

The **Groovy-based pipeline syntax** in Jenkins provides full programming language expressiveness, allowing teams to implement complex conditional logic, loops, and dynamic pipeline generation. However, this expressiveness comes with a steep learning curve and the potential for pipelines to become unmaintainable pieces of embedded Groovy code. Junior developers often struggle with the Jenkins pipeline syntax, particularly when debugging failures or understanding complex pipeline logic written by others.

**GitLab CI** represents the "integrated platform" approach, where CI/CD functionality is tightly coupled with source code management, issue tracking, and deployment environments. GitLab pipelines are defined using YAML files committed directly to the repository, creating a strong connection between code changes and their associated build and deployment procedures. GitLab uses a **runner-based architecture** where lightweight runners (which can be Docker containers, virtual machines, or bare metal servers) execute individual jobs.

GitLab's **integrated approach** eliminates many integration complexities that plague other solutions. Since GitLab controls the entire development lifecycle, it can provide seamless experiences like automatically triggering pipelines on merge requests, displaying pipeline status directly in the code review interface, and correlating deployment issues with specific commits and authors. This integration reduces context switching and provides developers with a unified interface for their entire workflow.

The **YAML-based pipeline definition** in GitLab strikes a balance between expressiveness and simplicity. YAML is more approachable for developers than Groovy scripting, but still supports complex features like matrix builds, conditional execution, and artifact passing between stages. However, YAML's limitations become apparent for highly dynamic pipelines that need to make runtime decisions about their execution path—teams often find themselves pushing against YAML's declarative constraints when implementing sophisticated deployment logic.

**GitHub Actions** represents the "marketplace-driven" approach, where the platform provides basic workflow orchestration and relies on a community marketplace of pre-built actions to implement specific functionality. GitHub Actions uses an **event-driven architecture** where workflows are triggered by repository events (pushes, pull requests, releases) or external webhooks. Individual steps in workflows can either run shell commands directly or invoke actions from the marketplace.

The **marketplace ecosystem** in GitHub Actions creates powerful composability—teams can assemble complex workflows by combining battle-tested actions rather than implementing everything from scratch. Popular actions for common tasks (building Docker images, deploying to cloud platforms, running security scans) are maintained by their respective communities and benefit from widespread usage and feedback. This approach reduces the maintenance burden on individual teams while providing access to sophisticated functionality.

However, the **marketplace dependency** introduces supply chain risks and version management challenges. Teams must carefully evaluate the security and reliability of third-party actions, monitor for updates and security patches, and handle breaking changes when action maintainers update their implementations. The action marketplace also creates a potential single point of failure—if a critical action becomes unmaintained or compromised, it can impact thousands of dependent workflows.

> **Decision: Pipeline Definition Approach**
> - **Context**: Teams need a way to express complex build, test, and deployment workflows that can be version-controlled, reviewed, and maintained alongside application code.
> - **Options Considered**: 
>   - Programmatic (Jenkins Groovy): Full programming language expressiveness
>   - Declarative YAML (GitLab CI): Balance of simplicity and capability  
>   - Marketplace composition (GitHub Actions): Reusable components with community support
> - **Decision**: Declarative YAML with structured validation and dependency graph construction
> - **Rationale**: YAML provides sufficient expressiveness for most CI/CD use cases while remaining approachable for junior developers. Unlike programmatic approaches, YAML pipelines are easier to analyze statically for security issues, resource requirements, and dependency relationships. Unlike marketplace approaches, YAML definitions don't introduce external dependencies or supply chain risks.
> - **Consequences**: Teams get predictable, analyzable pipelines that can be validated before execution. However, extremely dynamic use cases may require multiple pipeline files or external tooling to generate YAML definitions.

| Solution | Architecture Pattern | Definition Format | Plugin/Extension Model | Primary Strength | Primary Weakness |
|----------|---------------------|------------------|----------------------|-----------------|------------------|
| Jenkins | Master-agent with persistent state | Groovy DSL or XML | Thousands of plugins with complex dependencies | Universal flexibility, mature ecosystem | Operational complexity, plugin management overhead |
| GitLab CI | Runner-based with integrated platform | YAML with repository integration | Built-in features with limited extensibility | Seamless integration, unified workflow | Platform lock-in, limited customization |
| GitHub Actions | Event-driven with marketplace | YAML with action composition | Community marketplace of reusable actions | Rich ecosystem, low maintenance burden | Supply chain risks, marketplace dependency |
| Azure DevOps | Service-based with cloud integration | YAML or visual designer | Extensions plus built-in integrations | Enterprise features, Microsoft ecosystem integration | Complexity overhead, learning curve |

**Execution model differences** reveal fundamental architectural trade-offs in CI/CD design. Jenkins' **persistent agent model** provides excellent performance for teams with predictable workloads—agents can cache dependencies, maintain warm JVMs, and reuse expensive setup operations across multiple pipeline executions. However, persistent agents create potential security risks (credential leakage between jobs), resource contention issues (one job can consume resources needed by another), and environment drift problems (accumulated state from previous jobs affecting current executions).

GitLab CI and GitHub Actions favor **ephemeral execution environments** where each job runs in a fresh container or virtual machine that is destroyed after completion. This approach provides strong isolation guarantees and eliminates many categories of environment-related bugs, but at the cost of slower execution times due to repeated environment setup and inability to cache expensive operations between runs.

**Resource management** strategies also vary significantly between platforms. Jenkins requires teams to manually provision and maintain agent infrastructure, providing complete control but significant operational overhead. GitLab offers both self-managed runners (similar to Jenkins agents) and hosted runners (where GitLab manages the execution infrastructure). GitHub Actions primarily relies on hosted runners with limited self-hosted options, reducing operational burden but potentially creating resource limitations for compute-intensive workloads.

> The choice between persistent and ephemeral execution environments represents a fundamental trade-off between performance and isolation that influences the entire CI/CD pipeline architecture.

**Integration patterns** highlight how different solutions handle the challenge of coordinating with external systems. Jenkins' plugin architecture creates tight coupling between the CI/CD platform and external tools—plugins often implement complex logic for interacting with deployment targets, artifact repositories, and monitoring systems. This tight coupling enables sophisticated integrations but creates maintenance overhead and upgrade complexity.

GitLab CI and GitHub Actions favor **loose coupling** through standardized APIs and webhook integrations. External systems notify the CI/CD platform of relevant events, and pipeline steps interact with external systems through their public APIs rather than platform-specific plugins. This approach reduces maintenance overhead but may require more manual integration work for complex use cases.

The **emerging patterns** in modern CI/CD solutions point toward several architectural trends that inform our design decisions. **GitOps workflows** are becoming increasingly popular, where CI pipelines publish artifacts and deployment configurations to Git repositories, and separate deployment agents pull changes and apply them to target environments. **Progressive delivery techniques** like canary deployments and feature flags are moving from specialized tools into core CI/CD platform capabilities. **Security integration** is shifting left, with vulnerability scanning, compliance checking, and secret management becoming first-class pipeline concerns rather than afterthoughts.

### Implementation Guidance

**A. Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|--------------|----------------|
| YAML Parsing | PyYAML with basic validation | Cerberus or Pydantic for schema validation |
| Container Runtime | Docker CLI via subprocess | Docker SDK for Python |
| File Storage | Local filesystem with os.path | Object storage (S3-compatible) with boto3 |
| Process Management | subprocess.Popen with threading | asyncio for concurrent execution |
| Logging | Python logging module | Structured logging with structlog |
| Configuration | Environment variables + YAML | Configuration management with Dynaconf |

**B. Recommended File Structure:**
```
ci-cd-pipeline/
├── src/
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── parser.py          # YAML pipeline parsing (Milestone 1)
│   │   ├── executor.py        # Job execution engine (Milestone 2) 
│   │   ├── artifacts.py       # Artifact management (Milestone 3)
│   │   └── deployment.py      # Deployment strategies (Milestone 4)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── pipeline.py        # Core data structures
│   │   └── artifacts.py       # Artifact models
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── base.py           # Storage interface
│   │   └── filesystem.py     # File system implementation
│   └── utils/
│       ├── __init__.py
│       ├── logging.py        # Logging configuration
│       └── docker_client.py  # Docker integration helpers
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── examples/
│   ├── simple-pipeline.yml
│   ├── complex-pipeline.yml
│   └── deployment-strategies.yml
├── docs/
├── requirements.txt
└── setup.py
```

**C. Infrastructure Starter Code:**

**Docker Client Helper** (`src/utils/docker_client.py`):
```python
"""Docker integration utilities for pipeline execution."""
import docker
import logging
import time
from typing import Dict, List, Optional, Tuple, Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DockerExecutionError(Exception):
    """Raised when Docker container execution fails."""
    def __init__(self, message: str, exit_code: int, logs: str):
        super().__init__(message)
        self.exit_code = exit_code
        self.logs = logs

class DockerClient:
    """Simplified Docker client for pipeline job execution."""
    
    def __init__(self):
        self.client = docker.from_env()
    
    def run_command(self, image: str, command: List[str], environment: Dict[str, str] = None,
                   working_dir: str = "/workspace", volumes: Dict[str, str] = None,
                   timeout: int = 3600) -> Tuple[int, str]:
        """
        Run a command in a Docker container and return exit code and logs.
        
        Args:
            image: Docker image to use
            command: Command and arguments to execute
            environment: Environment variables to set
            working_dir: Working directory inside container
            volumes: Host:container volume mounts
            timeout: Execution timeout in seconds
            
        Returns:
            Tuple of (exit_code, combined_logs)
            
        Raises:
            DockerExecutionError: If execution fails or times out
        """
        container = None
        try:
            container = self.client.containers.run(
                image=image,
                command=command,
                environment=environment or {},
                working_dir=working_dir,
                volumes=volumes or {},
                detach=True,
                remove=True,
                stdout=True,
                stderr=True
            )
            
            # Wait for completion with timeout
            result = container.wait(timeout=timeout)
            exit_code = result['StatusCode']
            
            # Get logs
            logs = container.logs(stdout=True, stderr=True).decode('utf-8')
            
            return exit_code, logs
            
        except docker.errors.ContainerError as e:
            raise DockerExecutionError(f"Container execution failed: {e}", e.exit_status, str(e))
        except Exception as e:
            if container:
                try:
                    container.remove(force=True)
                except:
                    pass
            raise DockerExecutionError(f"Docker execution error: {e}", -1, str(e))
    
    @contextmanager
    def streaming_run(self, image: str, command: List[str], **kwargs) -> Iterator[Tuple[object, Iterator[str]]]:
        """
        Run container with streaming log output.
        
        Yields:
            Tuple of (container, log_stream)
        """
        container = self.client.containers.run(
            image=image,
            command=command,
            detach=True,
            **kwargs
        )
        
        try:
            log_stream = container.logs(stream=True, follow=True)
            yield container, log_stream
        finally:
            try:
                container.remove(force=True)
            except:
                pass
```

**Logging Configuration** (`src/utils/logging.py`):
```python
"""Centralized logging configuration for the pipeline system."""
import logging
import sys
from datetime import datetime
from typing import Optional

class PipelineFormatter(logging.Formatter):
    """Custom formatter that includes pipeline context."""
    
    def format(self, record):
        # Add timestamp
        record.timestamp = datetime.utcnow().isoformat()
        
        # Add pipeline context if available
        if hasattr(record, 'pipeline_id'):
            record.pipeline_context = f"[{record.pipeline_id}]"
        else:
            record.pipeline_context = ""
        
        if hasattr(record, 'job_id'):
            record.job_context = f"[{record.job_id}]"
        else:
            record.job_context = ""
        
        return super().format(record)

def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Configure logging for the pipeline system."""
    
    # Create formatter
    formatter = PipelineFormatter(
        fmt='%(timestamp)s [%(levelname)s] %(pipeline_context)s%(job_context)s %(name)s: %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger('docker').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
```

**D. Core Logic Skeleton:**

**Pipeline Context Manager** (`src/models/pipeline.py`):
```python
"""Core pipeline data structures and execution context."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from datetime import datetime

class JobStatus(Enum):
    """Possible states for job execution."""
    PENDING = "pending"
    RUNNING = "running" 
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"

@dataclass
class PipelineStep:
    """Individual step within a pipeline job."""
    name: str
    script: List[str] = field(default_factory=list)
    image: str = "ubuntu:20.04"
    environment: Dict[str, str] = field(default_factory=dict)
    timeout: int = 3600
    retry_count: int = 0
    
    # TODO: Add validation for required fields
    # TODO: Add method to resolve environment variables
    # TODO: Add method to validate script commands

@dataclass
class PipelineJob:
    """A job represents a unit of work with multiple steps."""
    name: str
    steps: List[PipelineStep]
    depends_on: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    
    # Runtime state
    status: JobStatus = JobStatus.PENDING
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    logs: List[str] = field(default_factory=list)
    
    def validate_dependencies(self, available_jobs: Set[str]) -> List[str]:
        """
        Validate that all job dependencies exist.
        
        Returns:
            List of missing dependencies
        """
        # TODO: Check each dependency in self.depends_on exists in available_jobs
        # TODO: Return list of dependencies that don't exist
        # TODO: Consider adding cycle detection here
        pass

class PipelineDefinition:
    """Complete pipeline definition parsed from YAML."""
    
    def __init__(self, name: str, jobs: List[PipelineJob], 
                 global_env: Dict[str, str] = None):
        self.name = name
        self.jobs = {job.name: job for job in jobs}
        self.global_env = global_env or {}
        self.created_at = datetime.utcnow()
        
        # TODO: Validate no duplicate job names
        # TODO: Validate all dependencies exist  
        # TODO: Build dependency graph for topological sorting
    
    def get_execution_order(self) -> List[List[str]]:
        """
        Return jobs in topological order for execution.
        
        Returns:
            List of job name lists, where each inner list can be executed in parallel
        """
        # TODO: Implement topological sort based on job dependencies
        # TODO: Group independent jobs together for parallel execution
        # TODO: Handle case where no valid ordering exists (circular dependencies)
        pass
    
    def get_job_dependencies(self, job_name: str) -> List[str]:
        """Get direct dependencies for a specific job."""
        # TODO: Return the depends_on list for the specified job
        # TODO: Handle case where job doesn't exist
        pass
```

**E. Language-Specific Hints:**

- **YAML Parsing**: Use `yaml.safe_load()` to prevent code execution vulnerabilities. Always validate the structure before processing.
- **Process Management**: Use `subprocess.Popen()` with proper timeout handling via `communicate(timeout=...)`. Never use `shell=True` with user input.
- **Docker Integration**: The Docker SDK handles connection pooling automatically. Use `docker.from_env()` to respect Docker environment variables.
- **File Handling**: Always use context managers (`with open(...)`) for file operations. Use `os.path.join()` for cross-platform path handling.
- **Environment Variables**: Use `os.environ.get(key, default)` for safe environment variable access. Never expose secrets in logs.
- **Error Handling**: Create custom exception classes for different failure modes. Use `try/except/finally` blocks to ensure cleanup occurs.

**F. Common Security Pitfalls:**

⚠️ **Pitfall: YAML Deserialization Attacks**
Using `yaml.load()` instead of `yaml.safe_load()` can execute arbitrary Python code embedded in YAML files. Malicious pipeline definitions could compromise the CI/CD system.

**Fix**: Always use `yaml.safe_load()` and validate the structure against a known schema before processing.

⚠️ **Pitfall: Command Injection in Shell Commands**
Concatenating user input directly into shell commands enables command injection attacks through malicious branch names, commit messages, or environment variables.

**Fix**: Use parameterized command construction with list-based arguments instead of shell strings. Validate all user input against allowed patterns.

⚠️ **Pitfall: Secret Exposure in Logs**
Environment variables containing secrets can accidentally appear in pipeline logs, making them visible to anyone with log access.

**Fix**: Implement secret masking in logging output and avoid echoing environment variables in shell scripts.


## Goals and Non-Goals

> **Milestone(s):** This section establishes the scope and boundaries that guide all subsequent milestones (1-4).

Think of defining goals and non-goals as drawing the blueprint boundaries for a construction project. Just as an architect must decide whether a building will have a basement, how many floors it will have, and what materials to use before breaking ground, we must establish clear boundaries for our CI/CD pipeline system before diving into implementation details. Without these boundaries, feature creep will turn a focused intermediate-level project into an enterprise-grade monster that takes years to complete.

The goals and non-goals serve as our north star throughout development. When faced with implementation decisions, we can ask: "Does this align with our stated goals?" and "Are we accidentally implementing something we explicitly decided not to build?" This disciplined approach prevents the common trap of over-engineering that plagues many CI/CD implementations.

### Functional Goals

These represent the core capabilities our CI/CD pipeline must deliver to be considered successful. Each goal directly maps to one or more milestones and defines measurable outcomes that validate our implementation.

**Pipeline Orchestration and Execution**

Our system must parse YAML pipeline definitions and execute them reliably. Think of this like a symphony conductor reading a musical score and coordinating dozens of musicians to produce harmonious output. The conductor doesn't play every instrument but ensures each musician knows when to play, what to play, and how their part fits into the whole composition.

The pipeline orchestration encompasses several critical capabilities:

| Capability | Description | Success Criteria |
|------------|-------------|------------------|
| YAML Parsing | Convert human-readable pipeline definitions into executable job graphs | Parser handles complex YAML with anchors, aliases, and nested structures |
| Dependency Resolution | Build directed acyclic graphs from job dependencies | Topological sort produces correct execution order for parallel jobs |
| Variable Substitution | Replace placeholders with environment-specific values | Variables resolve correctly across job boundaries and inheritance chains |
| Conditional Execution | Support if/when expressions for dynamic job execution | Jobs execute only when conditions evaluate to true |
| Parallel Execution | Run independent jobs simultaneously to reduce total pipeline time | Jobs without dependencies execute concurrently without interference |

> **Decision: YAML as Pipeline Definition Language**
> - **Context**: Need human-readable format for defining complex pipelines with stages, jobs, dependencies, and conditions
> - **Options Considered**: JSON (machine-friendly but verbose), YAML (human-friendly with good tooling), Custom DSL (maximum flexibility but learning curve)
> - **Decision**: YAML with strict schema validation
> - **Rationale**: YAML strikes the optimal balance between human readability and machine parseability. It supports advanced features like anchors and aliases for reducing duplication, has excellent tooling support, and is familiar to developers from Docker Compose and Kubernetes
> - **Consequences**: Must handle YAML parsing edge cases carefully, especially around indentation sensitivity and type coercion

**Isolated Job Execution**

Every job must execute in a completely isolated environment to prevent interference between jobs and ensure reproducible builds. Think of this like hotel rooms in a high-rise building—each guest has their own private space with identical amenities, but they cannot access other rooms or affect other guests' experiences.

Container isolation provides several essential guarantees:

| Isolation Aspect | Implementation | Benefit |
|------------------|----------------|---------|
| Process Isolation | Each job runs in separate Docker container | Jobs cannot interfere with each other's processes |
| Filesystem Isolation | Container gets clean filesystem from base image | No leftover files from previous jobs affect current execution |
| Network Isolation | Containers have isolated network namespaces | Jobs cannot accidentally connect to services from other jobs |
| Resource Limits | CPU and memory limits enforced by container runtime | One job cannot starve others of system resources |
| Environment Isolation | Each container gets its own environment variables | Secrets and configuration cannot leak between jobs |

The job executor must handle several critical scenarios that commonly cause CI/CD pipeline failures:

1. **Timeout Management**: Jobs that hang indefinitely will consume resources and block pipeline completion. Our executor implements configurable timeouts with graceful termination followed by force-kill if necessary.

2. **Retry Logic**: Network glitches, temporary service unavailability, or race conditions can cause jobs to fail sporadically. The executor implements exponential backoff retry with configurable maximum attempts.

3. **Log Streaming**: Real-time log output provides essential feedback for debugging failed builds and monitoring long-running jobs. Logs must be streamed without buffering delays while handling large output volumes efficiently.

4. **Exit Code Propagation**: The container's exit code determines job success or failure. The executor must capture and correctly interpret exit codes, distinguishing between different failure modes.

**Artifact Management and Transfer**

Build artifacts represent the valuable outputs of our pipeline—compiled binaries, test reports, deployment packages, and documentation. Think of artifacts like packages in a postal system: they must be labeled clearly, stored securely, transferred reliably, and delivered to the right recipients at the right time.

Our artifact management system provides these essential capabilities:

| Feature | Implementation Strategy | Quality Assurance |
|---------|------------------------|-------------------|
| Upload After Job Completion | Jobs declare artifacts in YAML; executor uploads specified paths | Checksum calculation ensures upload integrity |
| Download Before Job Execution | Dependent jobs automatically receive artifacts they declare as dependencies | Checksum verification detects corruption during transfer |
| Versioning and Retention | Artifacts tagged with pipeline ID, job name, and timestamp | Retention policy prevents storage exhaustion |
| Large File Handling | Streaming upload/download prevents memory exhaustion | Progress reporting for transfers over configurable threshold |
| Path Traversal Protection | Artifact paths validated to prevent directory traversal attacks | Whitelist approach allows only safe path patterns |

> **Decision: Filesystem-Based Artifact Storage**
> - **Context**: Need reliable, fast artifact storage that supports large files and doesn't require external dependencies
> - **Options Considered**: Cloud storage (S3/GCS - scalable but adds dependency), Database storage (simple but poor for large files), Filesystem storage (simple and fast but limited scalability)
> - **Decision**: Filesystem storage with content-addressable organization
> - **Rationale**: For intermediate-level project, filesystem storage eliminates external dependencies while providing excellent performance. Content-addressable storage (using SHA-256 hashes) enables deduplication and integrity verification
> - **Consequences**: Limited to single-machine deployments initially, but architecture supports pluggable storage backends for future cloud integration

**Deployment Strategy Implementation**

Modern deployment strategies go far beyond "upload files and restart the service." Think of deployment strategies like different ways to renovate a busy restaurant: you might close entirely and reopen when done (blue-green), renovate one section at a time while staying open (rolling), or serve a few customers the new menu while most get the old one (canary).

Our system implements three fundamental deployment patterns:

| Strategy | Use Case | Risk Profile | Rollback Speed |
|----------|----------|--------------|----------------|
| Rolling Deployment | Regular updates with moderate risk tolerance | Medium - some instances always running old version during transition | Medium - must update each instance individually |
| Blue-Green Deployment | Critical updates requiring atomic switches | Low - new version fully validated before traffic switch | Fast - single DNS/load balancer change |
| Canary Deployment | High-risk updates requiring gradual validation | Very Low - limited blast radius during validation | Fast - immediate traffic shift back to stable version |

Each deployment strategy requires sophisticated health checking and traffic management:

1. **Health Check Protocols**: The system must verify application health using HTTP endpoints, TCP connections, or custom scripts. Health checks run continuously during deployment transitions.

2. **Traffic Routing**: Load balancer configuration changes orchestrate traffic flow between old and new versions. The system must handle different load balancer types and configuration formats.

3. **Rollback Automation**: When health checks fail or error rates exceed thresholds, automatic rollback restores the previous version without human intervention.

4. **Manual Approval Gates**: Critical deployments require human approval before proceeding. The system pauses execution and waits for authorized approval via API or web interface.

### Non-Functional Goals

Non-functional goals define the quality attributes our system must exhibit—how fast it should be, how reliable, how secure, and how easy to operate. These goals are often more challenging to achieve than functional goals because they require careful attention to architecture, implementation details, and operational concerns.

**Performance and Scalability**

Our CI/CD pipeline must handle realistic workloads without becoming a bottleneck in the development process. Think of performance requirements like traffic capacity for a highway—the system must handle typical daily traffic smoothly while degrading gracefully during peak usage periods.

| Performance Metric | Target | Measurement Method | Impact of Missing Target |
|-------------------|--------|-------------------|-------------------------|
| Pipeline Parse Time | < 2 seconds for complex YAML files (100+ jobs) | Time from file read to executable job graph | Slow feedback delays development workflow |
| Job Startup Time | < 30 seconds from job trigger to first log output | Time from executor receiving job to container start | Long queues develop during busy periods |
| Artifact Transfer Speed | > 50 MB/s for large artifacts (network permitting) | Throughput measurement during upload/download | Build times dominated by artifact transfer |
| Concurrent Job Limit | 10+ simultaneous jobs on standard developer machine | Resource utilization monitoring during parallel execution | Pipelines serialize unnecessarily, increasing total time |
| Memory Efficiency | < 100MB base memory usage plus 50MB per active job | Memory profiling during various load scenarios | System becomes unusable on resource-constrained environments |

> **Decision: Asynchronous Job Execution Model**
> - **Context**: Need to support multiple simultaneous jobs without blocking pipeline progress
> - **Options Considered**: Synchronous execution (simple but slow), Thread-based concurrency (complex state management), Asynchronous execution with event loops (scalable and efficient)
> - **Decision**: Asynchronous execution using Python's asyncio for I/O operations and threading for CPU-bound tasks
> - **Rationale**: Most CI/CD operations are I/O-bound (network requests, file operations, container management). Asyncio provides excellent I/O concurrency without the complexity of manual thread management, while allowing thread pools for CPU-intensive work like YAML parsing
> - **Consequences**: Requires careful handling of async/await patterns throughout codebase, but enables much better resource utilization and responsiveness

**Reliability and Fault Tolerance**

CI/CD pipelines run in environments where failures are inevitable—network partitions, disk space exhaustion, container runtime issues, and external service outages. Our system must be resilient to these failures and provide clear diagnostics when things go wrong.

The reliability strategy encompasses several layers of defense:

| Failure Category | Detection Method | Recovery Strategy | User Impact |
|------------------|------------------|-------------------|-------------|
| Container Runtime Failures | Docker API error responses | Automatic retry with exponential backoff | Transparent - job continues after brief delay |
| Network Connectivity Issues | HTTP timeouts, DNS resolution failures | Retry with different endpoints if available | Job marked as failed after configured retry attempts |
| Disk Space Exhaustion | Filesystem write errors, artifact upload failures | Clean up temporary files, warn about low disk space | Pipeline fails with clear error message |
| YAML Parsing Errors | Schema validation failures, syntax errors | Detailed error messages with line numbers | Pipeline rejected at parse time with actionable feedback |
| Job Timeout Exceeded | Process monitoring, elapsed time tracking | Graceful termination followed by force-kill | Job marked as failed with timeout indication |

**Security and Isolation**

CI/CD pipelines execute arbitrary user code and handle sensitive information like deployment credentials and API keys. Think of security like the security measures at an airport—multiple layers of protection ensure that threats are detected and contained before they can cause damage.

Our security model implements defense in depth:

| Security Layer | Protection Mechanism | Attack Prevention |
|----------------|---------------------|-------------------|
| Container Isolation | Docker security features, non-root user execution | Prevents job-to-job interference and host system access |
| Secret Management | Environment variable injection, no secret persistence | Prevents credential leakage in logs or artifacts |
| Path Traversal Prevention | Input validation, chroot-style containment | Prevents access to files outside designated areas |
| Resource Limits | CPU/memory cgroups, execution timeouts | Prevents resource exhaustion attacks |
| Input Validation | YAML schema enforcement, command sanitization | Prevents injection attacks and malformed input processing |

⚠️ **Pitfall: Inadequate Container Security**
Many CI/CD implementations run containers with excessive privileges or mount sensitive host directories. This creates massive security holes where malicious code can escape the container and compromise the host system. Our implementation uses minimal container privileges, read-only root filesystems where possible, and carefully controlled volume mounts.

**Operational Simplicity**

A CI/CD system that requires a team of specialists to operate defeats the purpose of automation. Our system must be straightforward to install, configure, monitor, and troubleshoot. Think of operational simplicity like the design of a good kitchen appliance—it should work reliably with minimal maintenance and provide clear feedback when something needs attention.

| Operational Aspect | Simplicity Goal | Implementation Strategy |
|-------------------|----------------|------------------------|
| Installation | Single command setup on common platforms | Docker Compose deployment with sensible defaults |
| Configuration | Zero-config default behavior with optional customization | Convention-over-configuration approach |
| Monitoring | Built-in health endpoints and structured logging | Prometheus metrics and JSON-structured logs |
| Troubleshooting | Self-diagnosing error messages with remediation hints | Error categorization with specific fix recommendations |
| Backup/Recovery | Stateless design with externalized persistence | All state in easily-backed-up directories |

### Explicit Non-Goals

Clearly defining what we will NOT build is just as important as defining what we will build. These non-goals prevent feature creep and maintain focus on the core learning objectives. Each non-goal includes rationale for why it's excluded and how the architecture could accommodate it in future iterations.

**Advanced Security Features**

We explicitly exclude enterprise-grade security features that would significantly increase implementation complexity without proportional learning value for an intermediate-level project.

| Excluded Security Feature | Rationale for Exclusion | Future Extension Path |
|---------------------------|------------------------|----------------------|
| RBAC (Role-Based Access Control) | Requires user management system, authentication integration, and complex permission modeling | Plugin architecture supports future RBAC modules |
| Secret Encryption at Rest | Requires key management, encryption/decryption workflows, and secure key storage | Current environment variable approach can be enhanced with encrypted storage backend |
| Audit Logging | Requires persistent audit trail, compliance reporting, and log integrity verification | Structured logging foundation supports future audit log enhancement |
| Network Security Policies | Requires firewall integration, network segmentation, and traffic inspection | Container network isolation provides foundation for advanced networking |
| Code Signing Verification | Requires PKI integration, certificate management, and signature validation workflows | Artifact management system can be extended with signature verification |

> **Design Insight**: Security features often have exponential complexity curves. Basic container isolation provides 80% of the security value with 20% of the implementation effort. Advanced features like RBAC and audit logging require sophisticated infrastructure that would dominate the project scope.

**Distributed Execution and High Availability**

Our system targets single-machine deployment with the potential for future distributed scaling. Distributed systems introduce complexity around consensus, leader election, network partitions, and state synchronization that would overshadow the core CI/CD learning objectives.

| Distributed Feature | Complexity Introduced | Current Alternative |
|--------------------|----------------------|-------------------|
| Multi-Node Job Distribution | Leader election, job scheduling across nodes, failure detection | Single node with high concurrency handles typical development team workloads |
| Shared State Synchronization | Distributed consensus, conflict resolution, split-brain prevention | Local filesystem state with clear extension points for distributed storage |
| Network Partition Handling | CAP theorem trade-offs, partition detection, automatic failover | Single node eliminates network partition scenarios |
| Load Balancing | Health checking, traffic distribution, node discovery | Single endpoint simplifies deployment and reduces operational complexity |

**Advanced Pipeline Features**

Several advanced pipeline features common in enterprise CI/CD platforms are explicitly excluded to maintain project scope while ensuring solid fundamentals.

| Advanced Feature | Exclusion Rationale | Learning Value Trade-off |
|------------------|-------------------|------------------------|
| Pipeline Templates and Inheritance | Requires complex template resolution, inheritance chains, and override semantics | Focus remains on core execution engine rather than templating complexity |
| Matrix Builds | Requires job multiplication, parameter permutation, and result aggregation | Single-job execution mastery more valuable than build matrix complexity |
| Conditional Stage Execution | Requires complex expression evaluation and stage dependency modification | Job-level conditions provide sufficient learning about conditional execution |
| Pipeline Triggers Beyond Git Webhooks | Requires integration with multiple external systems and event handling | Git webhook integration demonstrates core trigger concepts |
| Advanced Caching Strategies | Requires cache key computation, dependency tracking, and invalidation policies | Basic artifact management teaches fundamental concepts |

⚠️ **Pitfall: Feature Creep Temptation**
During implementation, it's tempting to add "just one more feature" that seems simple but actually introduces significant complexity. For example, adding matrix builds requires multiplying every job by every parameter combination, managing result aggregation, and handling partial failures—turning a simple executor into a complex job scheduling system.

**Enterprise Integration and Compliance**

Enterprise environments require integration with existing systems and compliance with regulatory requirements that add substantial implementation burden without core learning value.

| Integration Category | Excluded Capabilities | Architectural Accommodation |
|--------------------|----------------------|----------------------------|
| Identity Providers | LDAP, Active Directory, SAML, OAuth integration | Plugin architecture allows future authentication modules |
| Monitoring Systems | Prometheus exporters, Grafana dashboards, APM integration | Structured logging and metrics endpoints support monitoring integration |
| Ticketing Systems | JIRA integration, ServiceNow workflows, approval systems | Webhook system can trigger external integrations |
| Compliance Reporting | SOX compliance, audit trails, data retention policies | Event logging foundation supports compliance extensions |
| Enterprise Storage | SAN integration, enterprise backup systems, retention policies | Pluggable storage backend accommodates enterprise storage systems |

> **Decision: Plugin Architecture for Future Extensions**
> - **Context**: Need to balance current simplicity with future extensibility for enterprise features
> - **Options Considered**: Monolithic design (simple now but hard to extend), Full plugin system (complex upfront), Hook-based extension points (balanced approach)
> - **Decision**: Define clear extension points with hook-based interfaces
> - **Rationale**: Hook-based design adds minimal current complexity while providing clear paths for future enhancements. Extension points are defined by interfaces that can be implemented by plugins
> - **Consequences**: Requires disciplined interface design and documentation, but enables future feature additions without core architecture changes

### Implementation Guidance

The goals and non-goals establish the foundation for all subsequent implementation decisions. This guidance helps translate high-level objectives into concrete development practices.

**A. Priority Framework for Implementation Decisions**

When facing implementation choices throughout the project, use this decision framework based on our stated goals:

| Decision Factor | Priority Level | Evaluation Questions |
|-----------------|----------------|---------------------|
| Core Functionality | Highest | Does this directly support YAML parsing, job execution, artifact management, or deployment strategies? |
| Learning Value | High | Will implementing this teach valuable CI/CD concepts that transfer to other systems? |
| Operational Simplicity | Medium | Does this make the system easier to install, configure, and troubleshoot? |
| Future Extensibility | Low | Does this create clean extension points without over-engineering current functionality? |

**B. Scope Validation Checklist**

Use this checklist during each milestone to ensure implementation stays within defined boundaries:

**Functional Scope Check:**
- [ ] Does each new function directly support one of the four stated functional goals?
- [ ] Are we implementing the minimum viable version of each capability?
- [ ] Have we resisted the urge to add "helpful" features not mentioned in goals?
- [ ] Does the implementation complexity match the intermediate difficulty target?

**Non-Functional Scope Check:**
- [ ] Are performance targets realistic for single-machine deployment?
- [ ] Does error handling provide clear, actionable feedback without over-engineering?
- [ ] Are security measures appropriate for development/testing environments?
- [ ] Can a single developer operate this system without specialized training?

**Non-Goal Boundary Check:**
- [ ] Are we avoiding enterprise security features that would dominate implementation time?
- [ ] Are we staying focused on single-machine deployment without distributed complexity?
- [ ] Are we implementing core pipeline features before advanced templating or matrix builds?
- [ ] Are we providing extension points without building full enterprise integrations?

**C. Goal-Driven Development Process**

Structure development activities around goal validation rather than pure feature completion:

1. **Milestone Planning**: Each milestone should directly advance one or more functional goals while respecting non-functional constraints and avoiding non-goals.

2. **Implementation Reviews**: Regular checks against goals and non-goals prevent scope creep during development. Ask: "Does this change advance our stated objectives?"

3. **Testing Strategy**: Test cases should validate goal achievement rather than just code coverage. Focus on scenarios that demonstrate successful goal fulfillment.

4. **Documentation Focus**: Document how each component contributes to goal achievement and how architectural decisions respect our stated boundaries.

**D. Extension Point Planning**

While avoiding over-engineering, identify clean extension points that align with potential future goals:

```
Extension Categories:
├── Authentication/Authorization
│   ├── Current: Environment variable secrets
│   └── Future: Plugin interface for auth providers
├── Storage Backends  
│   ├── Current: Filesystem artifact storage
│   └── Future: Cloud storage implementations
├── Deployment Targets
│   ├── Current: Generic deployment strategies
│   └── Future: Platform-specific deployment modules
└── Monitoring Integration
    ├── Current: Structured logging and basic metrics
    └── Future: Monitoring system adapters
```

**E. Milestone Checkpoint Alignment**

Each milestone checkpoint should validate progress toward stated goals:

**Milestone 1 Checkpoint**: Validate YAML parsing and dependency resolution goals
- Parse complex pipeline YAML with jobs, dependencies, and variables
- Generate correct execution order through topological sorting
- Handle parsing errors with clear, actionable error messages

**Milestone 2 Checkpoint**: Validate isolated job execution goals
- Execute jobs in separate Docker containers with proper isolation
- Stream job logs in real-time with timeout and retry handling
- Demonstrate concurrent job execution without interference

**Milestone 3 Checkpoint**: Validate artifact management goals
- Upload, store, and download artifacts between pipeline stages
- Verify artifact integrity with checksum validation
- Implement retention policy that prevents storage exhaustion

**Milestone 4 Checkpoint**: Validate deployment strategy goals
- Implement rolling, blue-green, and canary deployment patterns
- Demonstrate health checking and automatic rollback capabilities
- Show manual approval gates that pause deployment progression

This goal-driven approach ensures that every implementation decision contributes to the project's learning objectives while maintaining realistic scope for an intermediate-level CI/CD pipeline system.


## High-Level Architecture

> **Milestone(s):** This section establishes the foundational system structure that supports all subsequent milestones (1-4).

Think of the CI/CD pipeline system as an orchestrated assembly line in a modern manufacturing plant. Just as an assembly line has specialized workstations that each perform specific operations on products moving through the line, our pipeline system has four main components that work together to transform source code into deployed applications. The **Pipeline Definition Parser** acts like the production planning department, interpreting work orders (YAML files) and determining the sequence of operations. The **Job Executor** functions as the factory floor, where actual work happens in isolated workstations (Docker containers). The **Artifact Manager** serves as the warehouse system, storing intermediate products and ensuring they're available when needed downstream. Finally, the **Deployment Engine** operates like the shipping department, using different strategies to deliver finished products to customers while minimizing disruption.

### Component Overview

The CI/CD pipeline system consists of four primary components, each with distinct responsibilities and clear interfaces. Understanding these components individually and their interactions is crucial for building a maintainable and scalable pipeline system.

> **Decision: Four-Component Architecture**
> - **Context**: CI/CD pipelines involve parsing configurations, executing jobs, managing artifacts, and deploying applications - fundamentally different concerns that could be monolithic or separated
> - **Options Considered**: 
>   1. Monolithic pipeline runner with all functionality in one component
>   2. Two-tier split (parser+executor, artifact+deployment)
>   3. Four-component separation by responsibility
> - **Decision**: Four-component architecture with clear separation of concerns
> - **Rationale**: Each component has distinct failure modes, scaling requirements, and testing needs. The parser needs YAML processing capabilities, the executor needs container orchestration, artifact management needs storage optimization, and deployment needs traffic management. Separating these allows independent evolution and testing.
> - **Consequences**: More interfaces to maintain but better testability, scalability, and maintainability. Each component can be developed by different team members and scaled independently based on workload.

| Component | Architecture Decision | Option Comparison |
|-----------|----------------------|-------------------|
| Monolithic Runner | Single process handling all pipeline operations | Simple deployment, shared state, difficult testing |
| Two-Tier Split | Parser+Executor paired with Artifact+Deployment | Moderate complexity, some coupling remains |
| **Four-Component** | **Separate parser, executor, artifact manager, deployment engine** | **Clear boundaries, independent scaling, easier testing** |

#### Pipeline Definition Parser

The **Pipeline Definition Parser** serves as the system's brain, responsible for interpreting YAML pipeline definitions and creating executable execution plans. Think of it as a compiler that transforms human-readable configuration into machine-executable instructions, complete with dependency resolution and optimization.

The parser's primary responsibility is transforming declarative YAML pipeline definitions into executable `PipelineDefinition` objects with resolved dependencies and validated configuration. It handles the complex task of dependency graph construction, ensuring that jobs execute in the correct order while maximizing parallelism opportunities.

| Responsibility | Description | Input | Output |
|----------------|-------------|--------|--------|
| YAML Parsing | Convert raw YAML text into structured data | Pipeline YAML file | Parsed configuration tree |
| Schema Validation | Ensure configuration matches expected structure | Parsed configuration | Validation errors or success |
| Dependency Resolution | Build execution graph from job dependencies | Job dependency declarations | Directed acyclic graph (DAG) |
| Variable Substitution | Replace variables with actual values | Template expressions, environment variables | Resolved configuration |
| Execution Planning | Generate optimal execution order | Dependency graph | Topologically sorted job groups |

The parser maintains the invariant that all output `PipelineDefinition` objects are valid and executable - no downstream component should encounter malformed configuration. This design principle pushes complexity into the parser but simplifies all other components significantly.

#### Job Executor

The **Job Executor** functions as the pipeline's muscle, taking validated job definitions and executing them in isolated environments. Like a foreman managing workers in separate booths, it ensures each job runs in a clean, controlled environment without interference from other jobs.

The executor's core responsibility is reliable job execution with proper isolation, logging, and error handling. It abstracts away the complexity of container management while providing consistent execution semantics across different job types and requirements.

| Responsibility | Description | Input | Output |
|----------------|-------------|--------|--------|
| Container Management | Create and manage Docker containers for job isolation | Job specifications | Running containers |
| Script Execution | Run shell commands and scripts within containers | `PipelineStep` definitions | Exit codes and output |
| Environment Setup | Configure job environment variables and working directories | Environment configuration | Configured execution context |
| Log Streaming | Capture and forward real-time execution logs | Container stdout/stderr | Structured log events |
| Resource Management | Apply CPU, memory, and timeout limits | Resource specifications | Resource-constrained execution |
| Cleanup | Remove containers and temporary resources after execution | Completed jobs | Clean system state |

> **Critical Design Principle**: The executor treats each job as completely independent - no shared state exists between jobs except through explicit artifact dependencies. This isolation enables reliable parallel execution and simplifies debugging by eliminating hidden dependencies.

#### Artifact Manager

The **Artifact Manager** serves as the pipeline's memory system, handling the storage, retrieval, and lifecycle management of build artifacts. Think of it as a sophisticated library system that not only stores books (artifacts) but also tracks their relationships, ensures their integrity, and automatically removes outdated materials.

The artifact manager bridges the gap between independent job executions by providing a reliable mechanism for jobs to share outputs. It ensures data integrity through checksums and manages storage resources through configurable retention policies.

| Responsibility | Description | Input | Output |
|----------------|-------------|--------|--------|
| Artifact Upload | Store job outputs with metadata and checksums | File paths, artifact metadata | Storage identifiers |
| Artifact Download | Retrieve stored artifacts for dependent jobs | Artifact identifiers | Local file paths |
| Integrity Verification | Validate artifact checksums during transfer | Stored checksums, downloaded files | Validation results |
| Storage Organization | Organize artifacts by pipeline, job, and version | Artifact metadata | Structured storage layout |
| Retention Management | Clean up expired artifacts based on policies | Retention policies, artifact ages | Storage space reclamation |
| Deduplication | Avoid storing identical artifacts multiple times | Artifact content hashes | Space-efficient storage |

The artifact manager implements content-addressable storage, where artifacts are identified by their content hash rather than just their name. This approach enables automatic deduplication and provides strong integrity guarantees.

#### Deployment Engine

The **Deployment Engine** acts as the pipeline's delivery mechanism, implementing various strategies for releasing applications to production environments. Like a logistics coordinator choosing between express delivery, standard shipping, or gradual rollout based on package importance and risk tolerance, the deployment engine selects and executes appropriate deployment strategies.

The deployment engine encapsulates the complexity of modern deployment patterns while providing consistent interfaces for health checking, traffic management, and rollback procedures. It transforms simple "deploy this version" commands into sophisticated multi-step processes that minimize risk and downtime.

| Responsibility | Description | Input | Output |
|----------------|-------------|--------|--------|
| Strategy Selection | Choose appropriate deployment strategy | Deployment configuration, risk tolerance | Selected deployment plan |
| Health Monitoring | Verify application health during deployment | Health check endpoints, success criteria | Health status reports |
| Traffic Management | Control traffic routing during deployments | Load balancer configuration, traffic rules | Updated routing rules |
| Rollback Coordination | Revert to previous version when deployment fails | Failure detection, previous version metadata | Restored previous state |
| Environment Management | Manage multiple deployment environments | Environment specifications | Configured deployment targets |
| Approval Workflows | Handle manual approval gates in deployment process | Approval policies, user permissions | Approval status and progression |

> **Design Insight**: The deployment engine treats each deployment strategy as a state machine with well-defined transitions, rollback points, and health verification steps. This state-machine approach enables consistent behavior across different deployment patterns and simplifies testing of complex deployment scenarios.

### Data Flow

Understanding how data and control flow through the CI/CD pipeline system is essential for grasping the system's behavior and debugging issues when they arise. The data flow follows a clear progression from static configuration to dynamic execution, with well-defined transformation points and persistent state.

![CI/CD Pipeline System Architecture](./diagrams/system-architecture.svg)

Think of the data flow as a river system with four major processing stations. Raw YAML configuration enters like water from mountain springs, gets processed and purified at each station, and emerges as deployed applications. Each station adds value while maintaining the flow's integrity and tracking its progress.

#### Configuration to Execution Flow

The transformation from static YAML configuration to running deployment follows a clear pipeline with distinct phases and handoff points. Each phase validates its inputs and produces well-defined outputs that serve as inputs to the next phase.

| Phase | Input Data | Processing Component | Output Data | Validation Criteria |
|-------|------------|---------------------|-------------|-------------------|
| Definition Parsing | Raw YAML text, environment variables | Pipeline Definition Parser | `PipelineDefinition` object | Schema compliance, dependency validity |
| Execution Planning | `PipelineDefinition` object | Pipeline Definition Parser | Ordered job groups | No circular dependencies, all references resolved |
| Job Execution | `PipelineJob` specifications | Job Executor | Job results, output logs | All steps completed, exit codes captured |
| Artifact Production | Job outputs, file paths | Artifact Manager | Artifact references | Checksums verified, metadata complete |
| Deployment Preparation | Deployment artifacts, configuration | Deployment Engine | Deployment plan | Health checks defined, rollback plan ready |
| Production Release | Deployment plan, approval status | Deployment Engine | Deployed application | Health verified, metrics within thresholds |

The flow includes several feedback loops where later stages can influence earlier decisions. For example, deployment health check failures trigger rollback procedures that may restart portions of the execution pipeline.

#### Artifact Transfer Flow

Artifacts represent the primary mechanism for data sharing between independent job executions. The artifact transfer flow ensures reliable data movement while maintaining isolation between jobs and providing audit trails for debugging.

1. **Artifact Creation**: When a job completes successfully, the Job Executor identifies output files and directories specified in the job's artifact configuration. It calculates content hashes for integrity verification and packages files with metadata.

2. **Upload Processing**: The Artifact Manager receives upload requests containing file paths, metadata, and calculated hashes. It stores files in content-addressable storage using hash-based naming to enable deduplication and creates index entries linking logical artifact names to physical storage locations.

3. **Dependency Resolution**: Before starting dependent jobs, the Job Executor queries the Artifact Manager for required artifacts. The manager verifies that all requested artifacts exist and are available for download, returning storage locations and expected checksums.

4. **Download and Verification**: The Job Executor downloads required artifacts to the job's execution environment, verifies checksums against expected values, and extracts files to appropriate locations. Any checksum mismatches trigger retry logic or job failure.

5. **Lifecycle Management**: The Artifact Manager continuously monitors stored artifacts against retention policies, removing expired items and updating index entries. This background process prevents unlimited storage growth while preserving artifacts needed for active pipelines.

> **Critical Insight**: Artifact transfer is designed to be idempotent - downloading the same artifact multiple times produces identical results. This property enables safe retry logic and simplifies reasoning about pipeline behavior during failures and recoveries.

#### State Synchronization Flow

The pipeline system maintains consistent state across all components through explicit synchronization points and shared data stores. Unlike systems that rely on implicit state sharing, our design makes all state transitions explicit and auditable.

| Synchronization Point | Participating Components | Shared State | Consistency Mechanism |
|-----------------------|-------------------------|--------------|----------------------|
| Pipeline Start | Parser, Executor | `PipelineDefinition` | Parser validates completeness before handoff |
| Job Completion | Executor, Artifact Manager | Job status, artifact availability | Atomic status update with artifact registration |
| Deployment Trigger | Artifact Manager, Deployment Engine | Artifact readiness, deployment authorization | Artifact verification before deployment start |
| Health Check Updates | Deployment Engine, external monitors | Application health status | Periodic health polling with timeout handling |
| Pipeline Completion | All components | Final pipeline status | Distributed status aggregation |

The synchronization design avoids distributed consensus algorithms by establishing clear component ownership of different state domains. The Pipeline Definition Parser owns configuration state, the Job Executor owns execution state, the Artifact Manager owns storage state, and the Deployment Engine owns deployment state.

### Recommended File Structure

A well-organized file structure enables maintainable development and clear separation of concerns. The recommended structure reflects the component architecture while providing convenient development workflows and testing strategies.

> **Decision: Domain-Driven File Organization**
> - **Context**: Python projects can organize code by technical layers (models, views, controllers) or by business domains (parsing, execution, artifacts)
> - **Options Considered**:
>   1. Technical layer organization (models/, services/, utils/)
>   2. Component-based organization (parser/, executor/, artifacts/, deployment/)
>   3. Hybrid approach with shared utilities
> - **Decision**: Component-based organization with shared infrastructure
> - **Rationale**: Each component has distinct dependencies, testing requirements, and development workflows. Developers working on job execution don't need to understand YAML parsing internals. Component-based organization makes the codebase easier to navigate and enables parallel development.
> - **Consequences**: Some code duplication in utilities, but clearer boundaries and easier testing. New developers can focus on one component without understanding the entire system.

```
ci_cd_pipeline/
├── main.py                          # Entry point and CLI interface
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project configuration and build settings
├── README.md                        # Project documentation and setup instructions
│
├── src/                             # Source code root
│   ├── __init__.py
│   │
│   ├── common/                      # Shared utilities and base classes
│   │   ├── __init__.py
│   │   ├── models.py                # Core data structures (PipelineDefinition, PipelineJob, etc.)
│   │   ├── logging.py               # PipelineFormatter and logging setup
│   │   ├── exceptions.py            # Custom exception classes
│   │   └── config.py                # Configuration management and defaults
│   │
│   ├── parser/                      # Pipeline Definition Parser (Milestone 1)
│   │   ├── __init__.py
│   │   ├── yaml_parser.py           # YAML parsing and validation
│   │   ├── dependency_graph.py      # DAG construction and topological sorting
│   │   ├── variable_substitution.py # Environment variable resolution
│   │   └── schema.py                # YAML schema definitions
│   │
│   ├── executor/                    # Job Executor (Milestone 2)
│   │   ├── __init__.py
│   │   ├── docker_client.py         # DockerClient wrapper
│   │   ├── job_runner.py            # Main job execution logic
│   │   ├── container_manager.py     # Container lifecycle management
│   │   └── log_streamer.py          # Real-time log streaming
│   │
│   ├── artifacts/                   # Artifact Manager (Milestone 3)
│   │   ├── __init__.py
│   │   ├── storage.py               # File storage backend
│   │   ├── artifact_manager.py      # Upload/download coordination
│   │   ├── integrity.py             # Checksum verification
│   │   └── retention.py             # Cleanup and retention policies
│   │
│   └── deployment/                  # Deployment Engine (Milestone 4)
│       ├── __init__.py
│       ├── strategies/              # Deployment strategy implementations
│       │   ├── __init__.py
│       │   ├── rolling.py           # Rolling deployment
│       │   ├── blue_green.py        # Blue-green deployment
│       │   └── canary.py            # Canary deployment
│       ├── health_checks.py         # Health monitoring
│       └── traffic_manager.py       # Load balancer integration
│
├── tests/                           # Test suite organization
│   ├── __init__.py
│   ├── unit/                        # Unit tests for individual components
│   │   ├── test_parser/
│   │   ├── test_executor/
│   │   ├── test_artifacts/
│   │   └── test_deployment/
│   ├── integration/                 # Integration tests between components
│   │   ├── test_pipeline_flow.py
│   │   ├── test_artifact_transfer.py
│   │   └── test_deployment_strategies.py
│   └── fixtures/                    # Test data and mock configurations
│       ├── sample_pipelines/
│       ├── mock_artifacts/
│       └── test_environments/
│
├── examples/                        # Example pipeline configurations
│   ├── simple_pipeline.yml
│   ├── multi_stage_pipeline.yml
│   └── deployment_strategies.yml
│
├── docker/                         # Docker-related files
│   ├── Dockerfile                  # Main application container
│   └── test-images/                # Custom test containers
│       ├── python-test/
│       └── node-test/
│
└── docs/                           # Additional documentation
    ├── api.md                      # API documentation
    ├── configuration.md            # Pipeline configuration reference
    └── deployment.md               # Deployment guide
```

#### Component Module Organization

Each component follows a consistent internal organization that promotes maintainable development and testing. The structure separates core logic from external dependencies and provides clear entry points for different functionality.

| Module Type | Purpose | Examples | Dependencies |
|-------------|---------|----------|--------------|
| Core Logic | Business logic and algorithms | `dependency_graph.py`, `job_runner.py` | Only common models and utilities |
| External Adapters | Integration with external systems | `docker_client.py`, `storage.py` | External libraries (docker, filesystem) |
| Internal APIs | Component interfaces and coordination | `artifact_manager.py`, `strategies/__init__.py` | Core logic modules |
| Configuration | Component-specific configuration and schema | `schema.py`, component `__init__.py` files | Common configuration utilities |

This organization enables dependency injection for testing and allows components to evolve their external integrations without affecting core business logic.

#### Development Workflow Support

The file structure supports common development workflows through consistent patterns and tooling integration. Each component can be developed, tested, and deployed independently while maintaining integration with the overall system.

Development scripts and tooling configuration files support the component structure:

| Workflow | Supporting Files | Purpose |
|----------|-----------------|---------|
| Local Development | `main.py`, `requirements.txt` | Single-command pipeline execution |
| Component Testing | `tests/unit/test_*/` | Isolated component testing |
| Integration Testing | `tests/integration/` | End-to-end workflow validation |
| Example Validation | `examples/*.yml` | Real-world configuration testing |
| Container Testing | `docker/test-images/` | Execution environment validation |

> **Development Best Practice**: Each component directory includes an `__init__.py` file that exports the component's primary interface classes and functions. This design enables simple imports like `from src.parser import PipelineParser` while hiding internal implementation details.

#### Testing Strategy Alignment

The file structure directly supports the testing strategy by providing clear boundaries between testable units and realistic integration scenarios. Unit tests focus on individual modules within components, while integration tests exercise component interfaces and data flow.

**⚠️ Common Pitfalls in File Organization:**

- **Circular Import Dependencies**: Placing shared utilities in component directories can create import cycles. The `common/` directory prevents this by providing a shared foundation that all components can import without creating cycles.

- **Overly Deep Nesting**: Creating deep directory hierarchies like `src/executor/container/docker/client.py` makes imports unwieldy and suggests over-engineering. Keep nesting to 2-3 levels maximum.

- **Mixed Concerns in Modules**: Putting both Docker integration and job scheduling logic in the same file makes testing difficult. Separate external adapters from core business logic.

- **Test-Production Structure Mismatch**: Organizing tests differently from source code makes it difficult to locate relevant tests. Mirror the source structure in the test directory.

### Implementation Guidance

This implementation guidance provides concrete technology choices and starter code to support the four-component architecture. The recommendations balance simplicity for learning with realistic patterns used in production CI/CD systems.

#### Technology Recommendations

| Component | Simple Option | Advanced Option | Rationale |
|-----------|---------------|-----------------|-----------|
| YAML Parsing | PyYAML + Cerberus validation | pydantic with custom validators | PyYAML is standard library-level stable, Cerberus provides clear schema validation |
| Container Runtime | Docker SDK for Python | Kubernetes Python client | Docker provides simpler local development, K8s adds production complexity |
| Storage Backend | Local filesystem + JSON metadata | MinIO S3-compatible storage | Filesystem enables easy debugging and testing |
| Health Checking | HTTP requests library | Custom protocol implementations | HTTP covers 90% of real-world health check patterns |
| Configuration | Environment variables + YAML | Consul/etcd distributed config | Simple config reduces operational complexity |
| Logging | Python standard logging + structured JSON | ELK stack integration | Standard logging provides sufficient observability for learning |

#### Core Data Structures

The following data structures form the foundation of component communication and should be implemented in `src/common/models.py`:

```python
"""
Core data models for CI/CD pipeline system.

These models define the interfaces between components and ensure
consistent data representation throughout the system.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


class JobStatus(Enum):
    """Execution status for pipeline jobs."""
    PENDING = "pending"
    RUNNING = "running" 
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


@dataclass
class PipelineStep:
    """Individual execution step within a pipeline job."""
    name: str
    script: List[str]
    image: str = "ubuntu:20.04"  # DEFAULT_IMAGE
    environment: Dict[str, str] = field(default_factory=dict)
    timeout: int = 3600  # DEFAULT_TIMEOUT in seconds
    retry_count: int = 0
    working_directory: str = "/workspace"
    

@dataclass  
class PipelineJob:
    """Complete job definition with metadata and execution state."""
    name: str
    steps: List[PipelineStep]
    depends_on: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    
    # Execution state (populated during runtime)
    status: JobStatus = JobStatus.PENDING
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    logs: List[str] = field(default_factory=list)
    
    def validate_dependencies(self, available_jobs: List[str]) -> List[str]:
        """Validate that all job dependencies exist in the pipeline definition."""
        # TODO: Implement dependency validation
        pass
    
    def get_execution_order(self) -> List[List[str]]:
        """Return topologically sorted job groups for parallel execution."""
        # TODO: Implement topological sorting
        pass


@dataclass
class PipelineDefinition:
    """Complete pipeline configuration with global settings."""
    name: str
    jobs: Dict[str, PipelineJob]
    global_env: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_job_by_name(self, job_name: str) -> Optional[PipelineJob]:
        """Retrieve job by name with None if not found."""
        return self.jobs.get(job_name)
```

#### Component Interface Skeletons

Each component should implement a clear interface that other components can depend on. These skeleton classes define the contracts without implementation details:

```python
"""
Component interface definitions for dependency injection and testing.

src/parser/__init__.py
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, List
from src.common.models import PipelineDefinition

class PipelineParser(ABC):
    """Interface for parsing YAML pipeline definitions."""
    
    @abstractmethod
    def parse_pipeline(self, yaml_content: str, variables: Dict[str, str]) -> PipelineDefinition:
        """Parse YAML pipeline definition with variable substitution."""
        # TODO: Implement YAML parsing with PyYAML
        # TODO: Validate schema using Cerberus
        # TODO: Resolve variable references
        # TODO: Build dependency graph and validate for cycles
        pass


"""
src/executor/__init__.py  
"""

class JobExecutor(ABC):
    """Interface for executing pipeline jobs in isolated environments."""
    
    @abstractmethod
    def run_command(self, image: str, command: List[str], environment: Dict[str, str], 
                   working_dir: str, volumes: Dict[str, str], timeout: int) -> Tuple[int, str]:
        """Execute command in Docker container and return exit code with output."""
        # TODO: Create Docker container with specified image
        # TODO: Mount volumes and set environment variables  
        # TODO: Execute command with timeout handling
        # TODO: Capture stdout/stderr and return with exit code
        pass
    
    @abstractmethod
    def streaming_run(self, image: str, command: List[str], **kwargs) -> Iterator[str]:
        """Execute command with real-time log streaming."""
        # TODO: Start container and attach to stdout/stderr streams
        # TODO: Yield log lines as they become available
        # TODO: Handle container cleanup on completion or timeout
        pass


"""
src/artifacts/__init__.py
"""

class ArtifactManager(ABC):
    """Interface for artifact storage and retrieval."""
    
    @abstractmethod 
    def upload_artifact(self, artifact_path: str, pipeline_id: str, job_name: str) -> str:
        """Upload artifact and return storage identifier."""
        # TODO: Calculate content hash for deduplication
        # TODO: Store file in content-addressable storage
        # TODO: Create metadata index entry
        # TODO: Return artifact identifier for downloads
        pass
    
    @abstractmethod
    def download_artifact(self, artifact_id: str, destination_path: str) -> bool:
        """Download artifact to specified location with integrity verification."""
        # TODO: Retrieve artifact from storage using identifier
        # TODO: Verify checksum against stored hash
        # TODO: Extract to destination path
        # TODO: Return success status
        pass
```

#### Infrastructure Utilities

These utility functions provide complete implementations for common infrastructure tasks, allowing learners to focus on core CI/CD logic:

```python
"""
Infrastructure utilities for CI/CD pipeline components.

src/common/logging.py
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any

class PipelineFormatter(logging.Formatter):
    """Custom formatter that adds pipeline context to log messages."""
    
    def __init__(self, pipeline_id: str = None, job_name: str = None):
        super().__init__()
        self.pipeline_id = pipeline_id
        self.job_name = job_name
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with pipeline context as structured JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'pipeline_id': self.pipeline_id,
            'job_name': self.job_name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        return json.dumps(log_entry)

def setup_logging(level: str = "INFO", log_file: str = None, 
                  pipeline_id: str = None, job_name: str = None) -> logging.Logger:
    """Configure pipeline logging with structured output."""
    logger = logging.getLogger('ci_cd_pipeline')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    formatter = PipelineFormatter(pipeline_id, job_name)
    
    # Console handler for real-time monitoring
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler for persistent logs
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


"""
Docker integration utilities.

src/executor/docker_client.py
"""

import docker
from typing import Dict, List, Tuple, Iterator
from src.common.models import DEFAULT_TIMEOUT, DEFAULT_IMAGE

class DockerClient:
    """Wrapper around Docker SDK with CI/CD-specific functionality."""
    
    def __init__(self):
        self.client = docker.from_env()
    
    def run_container(self, image: str, command: List[str], 
                     environment: Dict[str, str] = None,
                     working_dir: str = "/workspace",
                     volumes: Dict[str, str] = None,
                     timeout: int = DEFAULT_TIMEOUT) -> Tuple[int, str]:
        """Run container and return exit code with combined output."""
        try:
            container = self.client.containers.run(
                image=image,
                command=command,
                environment=environment or {},
                working_dir=working_dir,
                volumes=volumes or {},
                detach=True,
                remove=False  # Keep for log retrieval
            )
            
            # Wait for completion with timeout
            result = container.wait(timeout=timeout)
            exit_code = result['StatusCode']
            
            # Get combined logs
            logs = container.logs(stdout=True, stderr=True).decode('utf-8')
            
            # Cleanup
            container.remove()
            
            return exit_code, logs
            
        except docker.errors.ContainerError as e:
            return e.exit_status, str(e)
        except Exception as e:
            return 1, f"Docker execution failed: {str(e)}"
    
    def stream_container_logs(self, image: str, command: List[str], **kwargs) -> Iterator[str]:
        """Stream container logs in real-time."""
        try:
            container = self.client.containers.run(
                image=image,
                command=command,
                detach=True,
                **kwargs
            )
            
            # Stream logs as they become available
            for log_line in container.logs(stream=True, follow=True):
                yield log_line.decode('utf-8').rstrip()
            
            # Wait for completion and cleanup
            container.wait()
            container.remove()
            
        except Exception as e:
            yield f"ERROR: {str(e)}"
```

#### Milestone Checkpoints

After implementing each component, verify correct functionality using these checkpoints:

**Milestone 1 Checkpoint - Pipeline Parser:**
```bash
# Test basic YAML parsing
python -m pytest tests/unit/test_parser/ -v

# Validate sample pipeline
python -c "
from src.parser import PipelineParser
parser = PipelineParser()
pipeline = parser.parse_pipeline(open('examples/simple_pipeline.yml').read(), {})
print(f'Parsed {len(pipeline.jobs)} jobs')
print(f'Execution order: {pipeline.get_execution_order()}')
"
```

**Expected Output:** Parser successfully creates `PipelineDefinition` with correct job dependency ordering and variable resolution.

**Milestone 2 Checkpoint - Job Executor:**
```bash
# Test container execution
python -c "
from src.executor import JobExecutor
executor = JobExecutor()
exit_code, output = executor.run_command('ubuntu:20.04', ['echo', 'Hello CI/CD'], {}, '/tmp', {}, 30)
print(f'Exit code: {exit_code}')
print(f'Output: {output}')
"
```

**Expected Output:** Container executes successfully with exit code 0 and "Hello CI/CD" output.

**Common Issues and Fixes:**

| Symptom | Likely Cause | Diagnosis | Fix |
|---------|--------------|-----------|-----|
| Import errors between components | Circular dependencies or incorrect paths | Check import statements and PYTHONPATH | Restructure imports through common module |
| Docker connection failures | Docker daemon not running or permissions | `docker ps` command test | Start Docker daemon, check user permissions |
| YAML parsing errors | Schema validation too strict | Print validation error details | Adjust Cerberus schema or fix YAML format |
| Memory exhaustion during testing | Large test fixtures or infinite loops | Monitor memory usage during tests | Use smaller test data, add timeouts |


## Data Model

> **Milestone(s):** This section supports all milestones (1-4) by defining the core data structures used throughout the CI/CD pipeline system.

Think of the data model as the **common language** that all components of the CI/CD pipeline speak. Just as a manufacturing assembly line needs standardized part specifications and work orders that every station can understand, our pipeline needs consistent data structures that represent everything from the initial YAML definition to the final deployment status. Without this shared vocabulary, the Pipeline Parser couldn't communicate job requirements to the Job Executor, and the Artifact Manager couldn't tell the Deployment Engine what files are available.

The data model serves three critical purposes in our system. First, it provides **structural consistency** - ensuring that a job definition means the same thing whether it's being parsed from YAML or executed in a container. Second, it enables **state tracking** - allowing us to monitor and control the pipeline's progress from initial parsing through final deployment. Third, it facilitates **data flow** - defining how information and artifacts move between pipeline stages without losing context or metadata.

Our data model consists of three interconnected layers that mirror the pipeline's execution flow. The **Pipeline Definition Model** captures the declarative structure from YAML files, defining what should happen. The **Execution Model** tracks the dynamic runtime state, recording what is happening. The **Artifact Model** manages the persistent outputs, preserving what has happened. These models work together to provide complete visibility and control over the entire CI/CD process.

![Pipeline Data Model](./diagrams/data-model.svg)

### Pipeline Definition Model

The Pipeline Definition Model represents the static structure parsed from YAML configuration files. Think of this as the **architectural blueprint** for a building - it defines all the rooms (jobs), their connections (dependencies), and the materials needed (artifacts), but doesn't track whether construction has started or which rooms are complete.

At the foundation of our definition model lies the `PipelineDefinition` structure, which serves as the root container for an entire pipeline specification. This structure captures not just the jobs and their relationships, but also the global context that applies across the entire pipeline execution.

| Field | Type | Description |
|-------|------|-------------|
| name | str | Human-readable pipeline identifier used in logs and UI |
| jobs | Dict[str, PipelineJob] | Map of job names to their definitions, enabling lookup by dependency references |
| global_env | Dict[str, str] | Environment variables inherited by all jobs unless overridden at job level |
| created_at | datetime | Timestamp when pipeline definition was parsed, used for auditing and cleanup |

> **Design Insight**: The choice to use a dictionary for jobs rather than a list enables O(1) dependency resolution during graph construction, which becomes critical for large pipelines with complex dependency networks.

Each pipeline consists of multiple jobs, represented by the `PipelineJob` structure. Jobs are the fundamental units of work in our pipeline, analogous to **individual assembly stations** in a manufacturing line. Each station (job) has specific tasks to perform (steps), requires certain inputs (dependencies and artifacts), and produces outputs for downstream stations.

| Field | Type | Description |
|-------|------|-------------|
| name | str | Unique identifier for the job within the pipeline, used in dependency declarations |
| steps | List[PipelineStep] | Ordered sequence of operations to execute, processed sequentially within the job |
| depends_on | List[str] | Names of jobs that must complete successfully before this job can start |
| artifacts | Dict[str, Any] | Specification of files to collect after job completion, keyed by artifact name |
| environment | Dict[str, str] | Job-specific environment variables that supplement or override global settings |
| status | JobStatus | Current execution state of the job, updated by the Job Executor during runtime |
| started_at | datetime | Timestamp when job execution began, used for timeout calculations and reporting |
| finished_at | datetime | Timestamp when job completed (success or failure), used for duration metrics |
| logs | List[str] | Accumulated log messages from all steps in the job, preserved for debugging |

The granular unit of execution within each job is the `PipelineStep`, representing individual commands or operations. Think of steps as the **specific tools and actions** used at each assembly station - each tool (step) has particular requirements for how it should be used and what environment it needs.

| Field | Type | Description |
|-------|------|-------------|
| name | str | Descriptive label for the step, displayed in logs and execution traces |
| script | List[str] | Shell commands to execute in sequence, with each string representing one command line |
| image | str | Docker image name and tag providing the execution environment for the step |
| environment | Dict[str, str] | Step-level environment variables that override job and global settings |
| timeout | int | Maximum execution time in seconds before the step is terminated |
| retry_count | int | Number of automatic retry attempts after initial failure before marking step as failed |

The job execution lifecycle is tracked using the `JobStatus` enumeration, which provides a comprehensive state machine for monitoring progress and handling failures.

| Status | Description | Valid Transitions |
|--------|-------------|-------------------|
| PENDING | Job is waiting for dependencies or resources | → RUNNING, CANCELLED |
| RUNNING | Job is currently executing steps | → SUCCESS, FAILED, CANCELLED, RETRY |
| SUCCESS | Job completed all steps successfully | Terminal state |
| FAILED | Job failed and exceeded retry limit | Terminal state |
| CANCELLED | Job was manually cancelled or pipeline was aborted | Terminal state |
| RETRY | Job failed but has remaining retry attempts | → RUNNING, FAILED |

> **Decision: Separate Step and Job Status Tracking**
> - **Context**: We need to track execution state at both the job and step levels for debugging and retry logic
> - **Options Considered**: Single combined status, separate step status enum, string-based status fields
> - **Decision**: Job-level status enum with step details in logs
> - **Rationale**: Job status drives pipeline orchestration decisions, while step details are primarily for debugging. Separate concerns reduce complexity in the execution engine.
> - **Consequences**: Enables fine-grained retry logic at job level while maintaining simple orchestration state

### Execution Model

The Execution Model captures the dynamic runtime state of pipeline execution, transforming the static definition into a living, breathing system. Think of this as the **factory floor control system** that tracks which assembly lines are running, what's currently being built, and where bottlenecks or failures are occurring.

While the Pipeline Definition Model describes what should happen, the Execution Model records what is happening and what has happened. This distinction is crucial for implementing features like real-time monitoring, failure recovery, and resource management. The execution model must handle the inherent concurrency of pipeline execution while maintaining data consistency and enabling accurate progress reporting.

The execution model builds upon the definition structures but enhances them with runtime-specific information. The `PipelineJob` structure serves double duty, containing both the static definition fields and the dynamic execution state. This design decision ensures that job definitions remain connected to their execution context, simplifying state management and reducing the risk of inconsistencies.

During execution, jobs transition through their lifecycle states according to dependency constraints and resource availability. The execution engine maintains several critical data structures to orchestrate this process effectively.

| Data Structure | Purpose | Key Operations |
|----------------|---------|----------------|
| Dependency Graph | Tracks job relationships and execution order | Topological sort, predecessor checking |
| Execution Queue | Manages jobs ready for execution | Enqueue eligible jobs, dequeue for execution |
| Active Jobs | Monitors currently running jobs | Track resource usage, timeout detection |
| Completion Registry | Records finished jobs and their outcomes | Dependency satisfaction checking, artifact availability |

The job dependency graph is perhaps the most critical runtime structure, enabling the execution engine to determine which jobs can run in parallel and which must wait for predecessors. The graph is constructed during pipeline initialization by analyzing the `depends_on` relationships in job definitions.

> **Design Insight**: The dependency graph is validated at parse time to detect circular dependencies, but the execution engine must also handle dynamic failures that can create unreachable jobs when their dependencies fail.

**Job Execution State Transitions:**

1. **Initialization Phase**: All jobs start in `PENDING` status as the execution engine builds the dependency graph
2. **Eligibility Detection**: Jobs with satisfied dependencies (all predecessors in `SUCCESS` state) become eligible for execution
3. **Resource Allocation**: Eligible jobs are queued for execution subject to resource limits (concurrent job limits, available executors)
4. **Execution Start**: Job transitions to `RUNNING` and begins executing its first step
5. **Step Processing**: Each step executes in sequence, with failures potentially triggering retry logic
6. **Completion Handling**: Job transitions to terminal state (`SUCCESS`, `FAILED`, or `CANCELLED`) and triggers dependent job eligibility checks

The execution model must also handle several complex scenarios that don't exist in the static definition:

**Concurrent Execution Management**: Multiple jobs may execute simultaneously, requiring thread-safe access to shared state like the dependency graph and completion registry. The execution engine uses fine-grained locking to allow maximum parallelism while preventing race conditions.

**Failure Propagation**: When a job fails, all jobs that depend on it (directly or transitively) must be marked as unrunnable. This requires traversing the dependency graph to identify the failure's impact scope.

**Resource Contention**: Jobs may compete for limited resources like execution slots or shared artifacts. The execution model includes resource allocation tracking to prevent oversubscription and ensure fair scheduling.

**Dynamic State Queries**: External systems need to query pipeline state for monitoring and control purposes. The execution model supports queries like "show all running jobs," "list jobs waiting for artifacts," and "calculate estimated completion time."

### Artifact Model

The Artifact Model manages the persistent outputs produced by pipeline jobs and consumed by subsequent stages. Think of artifacts as the **work-in-progress inventory** in a manufacturing system - each assembly station produces components that later stations need, and the inventory system must track what's available, where it's stored, and when it expires.

Artifacts serve as the primary mechanism for data flow between pipeline jobs, enabling complex multi-stage processes like building source code in one job and deploying the resulting binaries in another. The artifact model must handle several challenging requirements: ensuring data integrity across transfers, managing storage efficiently, providing fast access for dependent jobs, and implementing retention policies to prevent unbounded storage growth.

The artifact model centers around content-addressable storage, where each artifact is identified by a cryptographic hash of its contents. This approach provides several critical benefits: automatic deduplication when multiple jobs produce identical outputs, integrity verification to detect corruption, and cache-friendly storage that can optimize repeated downloads.

| Field | Type | Description |
|-------|------|-------------|
| id | str | Content hash (SHA-256) serving as unique identifier and integrity check |
| name | str | Human-readable name specified in job artifact configuration |
| job_name | str | Name of the job that produced this artifact, used for traceability |
| pipeline_name | str | Pipeline that created the artifact, enabling cross-pipeline artifact sharing |
| size_bytes | int | Total size in bytes, used for storage quota management and transfer optimization |
| created_at | datetime | Timestamp when artifact was uploaded, used for retention policy enforcement |
| expires_at | datetime | Calculated expiration time based on retention policy, used for cleanup scheduling |
| checksum | str | SHA-256 hash of artifact contents, used for integrity verification during downloads |
| storage_path | str | Internal path where artifact is stored, abstracted from consumers |
| metadata | Dict[str, str] | Additional key-value pairs for artifact classification and discovery |

**Artifact Lifecycle Management:**

The artifact lifecycle consists of four distinct phases, each with specific responsibilities and failure modes:

1. **Creation Phase**: A job step completes and identifies files to be preserved as artifacts
2. **Upload Phase**: Files are packaged, checksummed, and transferred to artifact storage
3. **Storage Phase**: Artifacts are persisted with metadata and indexed for discovery
4. **Consumption Phase**: Dependent jobs download and verify artifacts for their own processing
5. **Retention Phase**: Expired artifacts are identified and removed according to retention policies

> **Decision: Content-Addressable Storage with Separate Metadata**
> - **Context**: Need to store artifacts efficiently while enabling fast lookups by name and preventing corruption
> - **Options Considered**: File-based storage with naming conventions, database storage with blob fields, content-addressable storage with metadata index
> - **Decision**: Content-addressable storage using SHA-256 hashes as identifiers, with separate metadata index
> - **Rationale**: Content addressing provides automatic deduplication and integrity verification. Separate metadata enables efficient queries without scanning storage.
> - **Consequences**: Requires additional complexity for metadata management but provides strong guarantees for data integrity and storage efficiency

**Storage Organization:**

The artifact storage system organizes files in a hierarchical structure that balances performance, scalability, and maintainability:

```
artifact-storage/
├── objects/
│   ├── ab/
│   │   └── cdef1234... (content-addressed files)
│   ├── cd/
│   │   └── ef567890...
├── metadata/
│   ├── by-name/
│   │   └── build-output -> ../objects/ab/cdef1234...
│   ├── by-pipeline/
│   │   └── main-pipeline/
│   └── retention/
│       └── 2024-01-15.json (artifacts expiring today)
```

This organization enables several optimizations: the two-level directory structure in `objects/` prevents file system performance degradation with large numbers of files, symbolic links in metadata directories provide O(1) lookups without duplicating storage, and retention files enable efficient cleanup batch processing.

**Integrity Verification:**

Every artifact operation includes integrity verification to detect corruption, tampering, or transfer errors. The verification process operates at multiple levels:

- **Upload Verification**: Compute SHA-256 hash during upload and compare against expected hash from job output
- **Storage Verification**: Periodic background scanning computes hashes of stored files and compares against metadata
- **Download Verification**: Recompute hash after download and compare against stored checksum before extraction
- **Cross-Reference Verification**: Ensure metadata entries have corresponding storage files and vice versa

**Retention Policy Implementation:**

The artifact retention system automatically manages storage lifecycle according to configurable policies. Retention policies can be based on multiple criteria:

| Policy Type | Configuration | Implementation |
|-------------|---------------|----------------|
| Age-based | `max_age_days: 30` | Delete artifacts older than specified days |
| Size-based | `max_total_size: 100GB` | Delete oldest artifacts when storage exceeds limit |
| Count-based | `max_artifacts_per_pipeline: 50` | Keep only N most recent artifacts per pipeline |
| Usage-based | `delete_after_days_unused: 7` | Delete artifacts not accessed within time window |

The retention system runs as a background process that periodically scans artifact metadata and identifies candidates for deletion. Before deleting artifacts, the system verifies that no currently running jobs depend on them, preventing race conditions that could cause job failures.

> **Common Pitfalls in Artifact Management:**

⚠️ **Pitfall: Race Conditions During Cleanup**
When implementing retention policies, developers often forget to check for active downloads before deleting artifacts. This can cause job failures when artifacts are deleted between dependency checking and actual download. The solution is to use reference counting or file locking to prevent deletion of artifacts that are currently being accessed.

⚠️ **Pitfall: Insufficient Error Handling During Upload**
Network failures or disk space exhaustion during artifact upload can leave the system in an inconsistent state with partial uploads or missing metadata. Always implement atomic upload operations using temporary files and rename operations, with cleanup of partial uploads on failure.

⚠️ **Pitfall: Memory Exhaustion with Large Artifacts**
Attempting to load entire artifacts into memory for checksum computation or transfer can exhaust system memory with large files. Use streaming I/O with fixed-size buffers to process artifacts of arbitrary size without memory growth.

### Implementation Guidance

This section provides concrete implementation patterns and starter code for the core data structures that form the foundation of the CI/CD pipeline system.

**A. Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Data Serialization | JSON with built-in library | Protocol Buffers with schema evolution |
| State Persistence | File-based JSON storage | SQLite with WAL mode for ACID properties |
| Artifact Storage | Local filesystem with directories | Object storage (S3-compatible) with metadata DB |
| Configuration Parsing | PyYAML with manual validation | Pydantic with automatic schema validation |
| Date/Time Handling | Built-in datetime module | Arrow library for timezone-aware operations |

**B. Recommended File Structure:**

```
ci_cd_pipeline/
├── models/
│   ├── __init__.py
│   ├── pipeline.py          ← Pipeline definition structures
│   ├── execution.py         ← Runtime execution state
│   ├── artifacts.py         ← Artifact management structures
│   └── validators.py        ← Data validation utilities
├── storage/
│   ├── __init__.py
│   ├── artifact_store.py    ← Artifact storage backend
│   └── metadata_store.py    ← Pipeline metadata persistence
└── tests/
    ├── test_models.py       ← Data model unit tests
    └── fixtures/
        └── sample_pipeline.yml
```

**C. Infrastructure Starter Code:**

**File: `models/__init__.py`**

```python
"""
Core data models for CI/CD pipeline system.
Provides type definitions and validation for pipeline definitions,
execution state, and artifact management.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional
import json
import hashlib
from pathlib import Path

# Core enumerations used throughout the system
class JobStatus(Enum):
    """Represents the current state of a job in the execution pipeline."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"

# Constants for default values
DEFAULT_TIMEOUT = 3600  # seconds
DEFAULT_IMAGE = "ubuntu:20.04"

def generate_artifact_id(content: bytes) -> str:
    """Generate content-addressable identifier for artifacts."""
    return hashlib.sha256(content).hexdigest()

def validate_job_name(name: str) -> bool:
    """Validate job name follows naming conventions (alphanumeric + hyphens)."""
    import re
    return bool(re.match(r'^[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]$', name))

class ValidationError(Exception):
    """Raised when data model validation fails."""
    pass
```

**File: `models/pipeline.py`**

```python
"""
Pipeline definition data structures representing static YAML configuration.
These models capture the declarative structure of CI/CD pipelines.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from . import JobStatus, DEFAULT_TIMEOUT, DEFAULT_IMAGE, ValidationError

@dataclass
class PipelineStep:
    """Represents a single execution step within a pipeline job."""
    name: str
    script: List[str]
    image: str = DEFAULT_IMAGE
    environment: Dict[str, str] = field(default_factory=dict)
    timeout: int = DEFAULT_TIMEOUT
    retry_count: int = 0

    def __post_init__(self):
        """Validate step configuration after initialization."""
        if not self.name or not self.name.strip():
            raise ValidationError("Step name cannot be empty")
        if not self.script:
            raise ValidationError("Step must have at least one script command")
        if self.timeout <= 0:
            raise ValidationError("Step timeout must be positive")
        if self.retry_count < 0:
            raise ValidationError("Retry count cannot be negative")

@dataclass
class PipelineJob:
    """Represents a job within a pipeline containing multiple steps."""
    name: str
    steps: List[PipelineStep]
    depends_on: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    logs: List[str] = field(default_factory=list)

    def validate_dependencies(self, available_jobs: List[str]) -> List[str]:
        """Check that all job dependencies exist in the pipeline."""
        # TODO: Implement dependency validation
        # TODO: Check each dependency in depends_on exists in available_jobs
        # TODO: Return list of missing dependencies
        # TODO: Raise ValidationError if any dependencies are missing
        pass

    def get_execution_order(self) -> List[List[str]]:
        """Return topological sort of job dependencies for execution planning."""
        # TODO: Implement topological sort algorithm
        # TODO: Build adjacency list from job dependencies
        # TODO: Use Kahn's algorithm or DFS-based topological sort
        # TODO: Return list of job name lists, where each inner list can execute in parallel
        pass

@dataclass
class PipelineDefinition:
    """Root container for a complete pipeline specification."""
    name: str
    jobs: Dict[str, PipelineJob]
    global_env: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate pipeline definition consistency."""
        if not self.name or not self.name.strip():
            raise ValidationError("Pipeline name cannot be empty")
        if not self.jobs:
            raise ValidationError("Pipeline must contain at least one job")
        
        # Validate all job dependencies exist
        job_names = list(self.jobs.keys())
        for job in self.jobs.values():
            missing_deps = job.validate_dependencies(job_names)
            if missing_deps:
                raise ValidationError(f"Job {job.name} depends on non-existent jobs: {missing_deps}")
```

**D. Core Logic Skeleton Code:**

**File: `models/execution.py`**

```python
"""
Runtime execution state management for pipeline jobs.
Handles state transitions and execution tracking.
"""
from typing import Dict, List, Set, Optional
from datetime import datetime, timedelta
from . import JobStatus
from .pipeline import PipelineJob, PipelineDefinition

class ExecutionState:
    """Manages runtime state of pipeline execution."""
    
    def __init__(self, pipeline: PipelineDefinition):
        self.pipeline = pipeline
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.reverse_graph: Dict[str, Set[str]] = {}
        self.completed_jobs: Set[str] = set()
        self.failed_jobs: Set[str] = set()
        
    def build_dependency_graph(self) -> None:
        """Construct dependency graph from job definitions."""
        # TODO: Initialize empty adjacency lists for all jobs
        # TODO: For each job, add edges from dependencies to job
        # TODO: Build reverse graph for efficient dependent lookup
        # TODO: Validate graph is acyclic (no circular dependencies)
        pass
    
    def get_ready_jobs(self) -> List[str]:
        """Return list of jobs ready for execution (dependencies satisfied)."""
        # TODO: Find jobs in PENDING status
        # TODO: Check if all dependencies are in completed_jobs set
        # TODO: Exclude jobs whose dependencies failed
        # TODO: Return list of job names ready to run
        pass
    
    def mark_job_completed(self, job_name: str, success: bool) -> List[str]:
        """Mark job as completed and return newly eligible jobs."""
        # TODO: Update job status based on success parameter
        # TODO: Add to completed_jobs or failed_jobs set
        # TODO: If failed, mark all transitively dependent jobs as unrunnable
        # TODO: Return list of newly eligible jobs for execution
        pass
    
    def get_execution_summary(self) -> Dict[str, int]:
        """Return summary of job states for monitoring."""
        # TODO: Count jobs in each status
        # TODO: Return dictionary with status counts
        # TODO: Include estimated completion time based on remaining work
        pass
```

**E. Language-Specific Hints:**

- Use `dataclasses` for clean data structure definitions with automatic `__init__` and `__repr__` methods
- Leverage `typing` module for clear type hints that improve code documentation and IDE support
- Use `datetime.now()` for timestamps but consider timezone handling in production systems
- Consider `pydantic` for advanced validation features like custom validators and automatic type coercion
- Use `pathlib.Path` instead of string manipulation for file system operations
- Implement `__post_init__` in dataclasses for validation that requires access to all fields

**F. Milestone Checkpoint:**

After implementing the data model structures, verify correct implementation:

**Test Command:**
```bash
python -m pytest tests/test_models.py -v
```

**Expected Behavior:**
- All data structures can be instantiated with valid parameters
- Validation properly rejects invalid configurations (empty names, negative timeouts, missing dependencies)
- Job dependency validation correctly identifies missing dependencies
- Pipeline definition can be serialized to JSON and deserialized without data loss
- Status transitions follow the defined state machine rules

**Manual Verification:**
```python
# Create a simple pipeline and verify structure
pipeline = PipelineDefinition(
    name="test-pipeline",
    jobs={
        "build": PipelineJob(
            name="build",
            steps=[PipelineStep(name="compile", script=["make build"])]
        ),
        "test": PipelineJob(
            name="test", 
            steps=[PipelineStep(name="run-tests", script=["make test"])],
            depends_on=["build"]
        )
    }
)
print(f"Created pipeline with {len(pipeline.jobs)} jobs")
```

**Signs of Problems:**
- Import errors indicate missing dependencies or circular imports
- Validation errors during normal instantiation suggest overly strict validation
- Missing fields in printed representations indicate incomplete dataclass definitions
- Memory usage growing during repeated instantiation suggests missing cleanup in validation


## Pipeline Definition Parser (Milestone 1)

> **Milestone(s):** Milestone 1: Pipeline Definition Parser

Think of the Pipeline Definition Parser as the **architect's blueprint reader** for our CI/CD system. Just as a construction foreman needs to interpret architectural drawings to understand what to build, when to build it, and in what order, our pipeline parser takes YAML pipeline definitions and transforms them into actionable execution plans. The parser doesn't just read the YAML—it validates the structural integrity (like checking if the blueprint is physically possible), resolves all the variable references (like substituting specific measurements for generic placeholders), and constructs a dependency graph that tells us the optimal order to execute jobs (like determining you can't install the roof before building the walls).

![Job Dependency Graph](./diagrams/dependency-graph.svg)

The pipeline parser serves as the **critical foundation** for our entire CI/CD system. Without accurate parsing and validation, downstream components would receive malformed instructions, leading to runtime failures that are much harder to debug than upfront validation errors. The dependency graph construction is particularly crucial because it enables parallel execution—identifying which jobs can run simultaneously versus which must wait for predecessors to complete.

![Pipeline Execution Sequence](./diagrams/execution-sequence.svg)

### YAML Parsing and Validation

The YAML parsing and validation subsystem acts as the **gatekeeper** that ensures only well-formed, semantically correct pipeline definitions enter our system. Think of it as a **quality control inspector** in a manufacturing plant—it catches defects early when they're cheap to fix, rather than allowing them to propagate through the entire system where they become expensive failures.

The parsing process operates in three distinct phases: structural parsing, schema validation, and semantic validation. Each phase catches different categories of errors and provides increasingly specific feedback to help developers fix their pipeline definitions quickly.

#### Structural Parsing Phase

The structural parsing phase handles the raw YAML document and transforms it into a structured data representation. This phase must handle YAML-specific features like anchors, aliases, multi-document files, and complex data types while detecting syntax errors.

| YAML Feature | Purpose | Validation Required |
|--------------|---------|-------------------|
| Anchors (&name) | Define reusable YAML fragments | Ensure anchor is defined before use |
| Aliases (*name) | Reference previously defined anchors | Validate referenced anchor exists |
| Multi-line strings | Handle script blocks and long values | Preserve indentation and line breaks |
| Complex types | Support nested objects and arrays | Validate proper nesting and structure |
| Comments | Document pipeline configuration | Strip comments but preserve line numbers for error reporting |

The parser must maintain **source location information** throughout the parsing process. When validation errors occur later, developers need to know exactly which line and column in their YAML file contains the problem. This requires building a mapping from parsed data structures back to their original source locations.

> **Critical Design Insight**: Never discard source location information during parsing. Every parsed element should retain a reference to its original position in the YAML file. This investment in bookkeeping pays enormous dividends when developers need to fix validation errors.

#### Schema Validation Phase

Schema validation ensures the parsed YAML conforms to our expected pipeline definition structure. This phase verifies required fields are present, field types are correct, and values fall within acceptable ranges.

**Pipeline Definition Schema Structure:**

| Field Name | Type | Required | Description | Default Value |
|------------|------|----------|-------------|---------------|
| name | string | Yes | Human-readable pipeline identifier | N/A |
| jobs | Dict[str, PipelineJob] | Yes | Map of job names to job definitions | N/A |
| global_env | Dict[str, str] | No | Environment variables for all jobs | {} |
| timeout | int | No | Maximum pipeline execution time (seconds) | 7200 |
| retry_policy | Dict[str, Any] | No | Default retry configuration for all jobs | {"max_attempts": 1} |
| artifacts | Dict[str, Any] | No | Global artifact configuration | {} |
| triggers | List[Dict[str, Any]] | No | Pipeline trigger conditions | [] |

**Job Definition Schema Structure:**

| Field Name | Type | Required | Description | Constraints |
|------------|------|----------|-------------|-------------|
| name | string | Yes | Unique job identifier within pipeline | Must match key in jobs dict |
| steps | List[PipelineStep] | Yes | Ordered list of execution steps | Minimum 1 step |
| depends_on | List[str] | No | List of job names this job depends on | All referenced jobs must exist |
| artifacts | Dict[str, Any] | No | Job-specific artifact configuration | Overrides global artifacts |
| environment | Dict[str, str] | No | Job-specific environment variables | Merged with global_env |
| timeout | int | No | Job timeout in seconds | Range: 1-3600 |
| retry_count | int | No | Number of retry attempts on failure | Range: 0-5 |
| conditional | string | No | Expression determining if job should run | Valid expression syntax |

**Step Definition Schema Structure:**

| Field Name | Type | Required | Description | Constraints |
|------------|------|----------|-------------|-------------|
| name | string | Yes | Step identifier for logging and reporting | Unique within job |
| script | List[str] | Conditional | Shell commands to execute | Required if 'action' not specified |
| action | string | Conditional | Predefined action type | Required if 'script' not specified |
| image | string | No | Docker image for step execution | Valid image reference |
| environment | Dict[str, str] | No | Step-specific environment variables | Merged with job environment |
| working_directory | string | No | Directory for step execution | Valid path within container |
| timeout | int | No | Step timeout in seconds | Range: 1-1800 |
| retry_count | int | No | Step-specific retry attempts | Range: 0-3 |
| continue_on_failure | bool | No | Whether to continue if step fails | Default: false |

The schema validator must provide **precise error messages** that include the field path, expected type, actual value, and suggested corrections. Poor error messages are one of the biggest sources of developer frustration with CI/CD systems.

#### Semantic Validation Phase

Semantic validation goes beyond structure to ensure the pipeline definition makes logical sense. This phase catches errors like circular dependencies, references to non-existent jobs, and resource conflicts that would cause runtime failures.

**Semantic Validation Rules:**

| Validation Rule | Description | Error Handling |
|-----------------|-------------|----------------|
| Job name uniqueness | No duplicate job names within pipeline | List all duplicate names found |
| Dependency existence | All jobs in depends_on lists must exist | Report missing job names and suggest corrections |
| Circular dependencies | Dependency graph must be acyclic | Identify and report the circular path |
| Resource limits | Total resource requests don't exceed limits | Report current usage vs limits |
| Image availability | Referenced Docker images are accessible | Validate image references without pulling |
| Variable references | All variable references can be resolved | List undefined variables |
| Conditional expressions | All conditional expressions are syntactically valid | Parse and validate expression syntax |

> **Decision: Fail-Fast Validation Strategy**
> - **Context**: We could validate pipeline definitions at parse time or defer validation until execution
> - **Options Considered**: 
>   1. Parse-time validation (fail-fast)
>   2. Runtime validation (fail-late)  
>   3. Hybrid approach with warnings
> - **Decision**: Comprehensive parse-time validation with immediate failure
> - **Rationale**: Catching errors early provides faster feedback loops for developers and prevents resource waste on doomed pipeline executions
> - **Consequences**: Requires more sophisticated parsing logic but dramatically improves developer experience and system reliability

#### Error Reporting and Recovery

The validation system must provide actionable error messages that help developers fix their pipeline definitions quickly. Generic error messages like "invalid YAML" waste developer time and create frustration.

**Error Message Structure:**

| Component | Purpose | Example |
|-----------|---------|---------|
| Location | Exact position in source file | Line 23, Column 15 |
| Context | Surrounding YAML structure | In job "build-frontend", step "run-tests" |
| Problem | What validation rule was violated | Missing required field "script" |
| Solution | Specific fix recommendation | Add "script" field with list of shell commands |
| Reference | Link to documentation | See: pipeline-syntax.md#step-definitions |

**Common Validation Errors and Solutions:**

| Error Pattern | Developer Intent | Suggested Fix |
|---------------|------------------|---------------|
| Missing "script" in step | Forgot to specify commands | Add script: ["echo 'hello'"] |
| Invalid job name in depends_on | Typo in dependency reference | Check spelling, list available job names |
| Circular dependency detected | Logical error in job ordering | Show dependency chain, suggest reordering |
| Invalid Docker image reference | Malformed image name | Show correct format: registry/repo:tag |
| Undefined variable reference | Variable not declared | List available variables, check spelling |

The error recovery mechanism should attempt to **continue validation** even after finding errors, collecting as many problems as possible in a single pass. This prevents the frustrating "fix one error, run again, find the next error" cycle.

### Dependency Graph Construction

The dependency graph construction subsystem transforms the flat list of jobs with their dependency declarations into a **Directed Acyclic Graph (DAG)** that enables optimal parallel execution. Think of this process as creating a **project management timeline** from a list of tasks—we need to identify which tasks can start immediately, which must wait for prerequisites, and how to maximize parallelism while respecting dependencies.

The dependency graph serves multiple purposes beyond just execution ordering. It enables impact analysis (which jobs are affected if this job fails), resource optimization (can we run these jobs on the same worker), and progress tracking (how much of the pipeline is complete).

#### Graph Data Structure Design

The dependency graph uses an **adjacency list representation** optimized for the operations we need to perform frequently: finding ready jobs, marking jobs complete, and detecting completion cascades.

**Dependency Graph Internal Structure:**

| Component | Type | Purpose | Key Operations |
|-----------|------|---------|----------------|
| nodes | Dict[str, GraphNode] | Job metadata and state | O(1) lookup by job name |
| incoming_edges | Dict[str, Set[str]] | Jobs this job depends on | O(1) dependency check |
| outgoing_edges | Dict[str, Set[str]] | Jobs that depend on this job | O(1) dependent enumeration |
| ready_queue | List[str] | Jobs ready for execution | O(1) pop, O(n) recompute |
| execution_order | List[List[str]] | Topologically sorted job batches | Pre-computed for visualization |

**GraphNode Data Structure:**

| Field Name | Type | Purpose | Example Value |
|------------|------|---------|---------------|
| job_name | str | Unique job identifier | "build-frontend" |
| status | JobStatus | Current execution state | PENDING, RUNNING, SUCCESS |
| dependencies_remaining | int | Count of incomplete dependencies | 3 |
| dependents | Set[str] | Jobs waiting for this job | {"deploy-staging", "run-e2e-tests"} |
| estimated_duration | int | Historical runtime estimate (seconds) | 180 |
| resource_requirements | Dict[str, Any] | CPU, memory, disk requirements | {"cpu": 2, "memory": "4GB"} |

The graph maintains **dependency counts** for each job, which enables O(1) determination of job readiness. When a job completes, we decrement the dependency count for all its dependents and add any that reach zero to the ready queue.

> **Critical Performance Insight**: Maintaining dependency counts separately from the edge lists enables constant-time readiness checking. Without this optimization, we'd need to traverse all incoming edges for every job on every completion event, creating O(E×V) complexity instead of O(V).

#### Topological Sorting Algorithm

The topological sorting algorithm determines a valid execution order for all jobs while maximizing parallelism opportunities. Our implementation uses **Kahn's algorithm** with modifications to group jobs by execution "levels" rather than producing a single linear order.

**Modified Kahn's Algorithm Steps:**

1. **Initialize dependency counts**: For each job, count the number of jobs it depends on directly
2. **Identify initial ready jobs**: All jobs with zero dependencies form the first execution level  
3. **Process execution level**: Remove all jobs in current level from the graph
4. **Update dependency counts**: For each removed job, decrement dependency count of all its dependents
5. **Form next level**: All jobs that now have zero dependencies form the next execution level
6. **Repeat until graph is empty**: Continue until all jobs are processed or a cycle is detected
7. **Detect cycles**: If jobs remain but no job has zero dependencies, report circular dependency

**Execution Level Formation:**

| Level | Jobs | Parallelism Opportunity | Resource Considerations |
|-------|------|------------------------|------------------------|
| 0 | Jobs with no dependencies | Full parallelism possible | Limited by worker capacity |
| 1 | Jobs depending only on Level 0 | Parallelism within level | May reuse resources from completed Level 0 jobs |  
| 2 | Jobs depending on Levels 0-1 | Decreasing parallelism | Resource allocation becomes more constrained |
| N | Final deployment jobs | Typically sequential | Often requires exclusive resource access |

The algorithm produces an **execution plan** that maximizes parallelism while respecting all dependency constraints. This plan can be visualized for developers and used by the execution engine to optimize resource allocation.

#### Circular Dependency Detection

Circular dependency detection is crucial for providing actionable error messages to developers. When cycles exist, we need to identify not just that a cycle exists, but **which specific jobs form the cycle** and suggest how to break it.

**Cycle Detection Algorithm:**

1. **Run modified topological sort**: Attempt normal topological sorting process
2. **Identify remaining jobs**: After processing, jobs with non-zero dependency counts form cycles
3. **Trace cycle paths**: Use depth-first search from each remaining job to find actual cycles
4. **Report shortest cycle**: Present the simplest cycle to the developer for easier understanding
5. **Suggest cycle breaking**: Identify dependency edges that could be removed to break the cycle

**Cycle Reporting Format:**

| Component | Example | Purpose |
|-----------|---------|---------|
| Cycle identification | "build-frontend → test-frontend → build-frontend" | Show complete dependency loop |
| Involved jobs | ["build-frontend", "test-frontend"] | List all jobs in cycle for batch editing |
| Suggested fixes | Remove dependency: test-frontend → build-frontend | Provide actionable resolution steps |
| Alternative approaches | Split build-frontend into build + package jobs | Suggest architectural improvements |

> **Decision: Comprehensive Cycle Detection**
> - **Context**: Cycles could be detected with simple boolean result or detailed path analysis
> - **Options Considered**:
>   1. Boolean cycle detection (fast, minimal info)
>   2. Single cycle path reporting (moderate complexity)
>   3. All cycle enumeration (comprehensive, slower)
> - **Decision**: Single shortest cycle with detailed reporting
> - **Rationale**: Developers need specific actionable information to fix cycles, but reporting all cycles creates information overload
> - **Consequences**: Requires more sophisticated detection algorithm but provides much better developer experience

#### Dependency Validation

Dependency validation ensures all job references are valid and the resulting graph structure supports efficient execution. This validation catches common developer errors like typos in job names and helps optimize pipeline performance.

**Dependency Validation Checks:**

| Validation Type | Check Description | Error Response |
|----------------|-------------------|----------------|
| Reference existence | All jobs in depends_on exist in pipeline | List available job names, suggest corrections |
| Self-dependency | Job doesn't depend on itself | Remove self-reference automatically |
| Redundant dependencies | Job doesn't declare transitive dependencies | Warn about redundancy, suggest cleanup |
| Resource conflicts | Dependent jobs have compatible resource needs | Warn about resource bottlenecks |
| Timeline feasibility | Dependency chain can complete within timeout | Estimate critical path duration |

**Advanced Dependency Analysis:**

The system can perform sophisticated analysis of the dependency structure to optimize execution and warn about potential problems:

| Analysis Type | Purpose | Output |
|---------------|---------|--------|
| Critical path | Identify longest dependency chain | Highlight jobs that most affect total runtime |
| Parallelism factor | Calculate maximum theoretical speedup | Guide resource allocation decisions |
| Resource hotspots | Find resource contention points | Suggest job scheduling optimizations |
| Failure impact | Determine blast radius of job failures | Prioritize reliability investments |

### Variable Substitution

The variable substitution subsystem handles the **parameterization** of pipeline definitions, allowing the same pipeline structure to be used across different environments, branches, and contexts. Think of variable substitution as a **template engine** for CI/CD pipelines—it takes a generic pipeline definition with placeholders and produces a fully specified pipeline for a particular execution context.

Variable substitution operates in multiple phases and handles several types of variables with different scoping rules and resolution priorities. The system must handle complex scenarios like nested variable references, conditional substitutions, and default value fallbacks while maintaining security boundaries.

#### Variable Types and Scoping

The variable system supports multiple variable types with different sources, scopes, and resolution priorities. Understanding the variable hierarchy is crucial for predictable pipeline behavior.

**Variable Source Hierarchy (highest to lowest priority):**

| Priority | Variable Source | Scope | Example | Override Behavior |
|----------|----------------|-------|---------|------------------|
| 1 | Runtime parameters | Pipeline execution | `--var branch=feature-123` | Overrides all other sources |
| 2 | Job-level environment | Single job | `environment: {DEBUG: true}` | Overrides global variables |
| 3 | Pipeline global_env | Entire pipeline | `global_env: {NODE_ENV: production}` | Default for all jobs |
| 4 | System environment | CI/CD system | `CI=true`, `BUILD_NUMBER=42` | Provided by execution environment |
| 5 | Default values | Variable definition | `${VAR_NAME:-default_value}` | Fallback when variable undefined |

**Variable Scope Resolution Rules:**

| Variable Reference | Resolution Context | Lookup Order | Example Resolution |
|-------------------|-------------------|--------------|-------------------|
| `${BUILD_NUMBER}` | Step execution | Job env → Global env → System env → Error | `42` |
| `${secrets.API_KEY}` | Secure context only | Secret store → Error | `sk_live_abc123` |
| `${matrix.python_version}` | Matrix job expansion | Matrix parameters → Error | `3.9` |
| `${job.build.status}` | Cross-job reference | Completed job state → Error | `SUCCESS` |

The scoping system prevents variable leakage between jobs while allowing controlled sharing of common configuration. Each job execution receives a **merged environment** that combines variables from all applicable scopes.

#### Variable Syntax and Parsing

The variable substitution syntax supports multiple formats to handle different use cases: simple substitutions, default values, conditional expressions, and complex transformations.

**Supported Variable Syntax Patterns:**

| Syntax Pattern | Purpose | Example | Resolved Value |
|----------------|---------|---------|----------------|
| `${VAR_NAME}` | Simple substitution | `${BRANCH_NAME}` | `main` |
| `${VAR_NAME:-default}` | Default value fallback | `${TIMEOUT:-300}` | `300` (if TIMEOUT undefined) |
| `${VAR_NAME:+alternate}` | Conditional substitution | `${DEBUG:+--verbose}` | `--verbose` (if DEBUG defined) |
| `${VAR_NAME/pattern/replacement}` | Pattern replacement | `${BRANCH_NAME/\//-}` | Replace slashes with dashes |
| `${{expression}}` | Complex expression | `${{env.STAGE == 'prod'}}` | `true` or `false` |

**Variable Reference Parser Components:**

| Component | Responsibility | Error Handling |
|-----------|----------------|----------------|
| Lexer | Tokenize variable syntax | Report unterminated references |
| Parser | Build expression AST | Validate expression structure |
| Resolver | Look up variable values | Handle undefined variables gracefully |
| Evaluator | Execute expressions | Catch and report evaluation errors |
| Formatter | Apply string transformations | Validate transformation patterns |

The parser must handle **nested variable references** where the value of one variable contains references to other variables. This requires careful ordering of resolution to avoid infinite loops while supporting legitimate use cases.

> **Decision: Recursive Variable Resolution with Cycle Detection**
> - **Context**: Variable values can contain references to other variables, creating resolution chains
> - **Options Considered**:
>   1. Single-pass resolution (simple, doesn't support nesting)
>   2. Multi-pass resolution (supports nesting, risk of infinite loops)
>   3. Recursive resolution with cycle detection (complex but robust)
> - **Decision**: Recursive resolution with explicit cycle detection and depth limits
> - **Rationale**: Developers expect variable nesting to work, but infinite loops must be prevented
> - **Consequences**: More complex resolution logic but supports advanced use cases safely

#### Expression Evaluation Engine

The expression evaluation engine handles complex variable substitutions that go beyond simple string replacement. This includes conditional logic, arithmetic operations, string manipulations, and function calls.

**Expression Types and Capabilities:**

| Expression Type | Use Case | Example | Result Type |
|----------------|----------|---------|-------------|
| Boolean conditions | Conditional job execution | `env.STAGE == 'production'` | boolean |
| String operations | Dynamic naming | `'deploy-' + env.BRANCH_NAME` | string |
| Arithmetic | Resource calculations | `env.WORKER_COUNT * 2` | number |
| Array operations | Dynamic job matrices | `env.TEST_TARGETS.split(',')` | array |
| Object access | Complex data structures | `job.build.artifacts.count` | any |

**Built-in Functions:**

| Function | Purpose | Signature | Example Usage |
|----------|---------|-----------|---------------|
| `contains(haystack, needle)` | String/array membership | `contains(env.FEATURES, 'experimental')` | Check feature flags |
| `startsWith(string, prefix)` | String prefix matching | `startsWith(env.BRANCH, 'release/')` | Branch classification |
| `join(array, separator)` | Array to string conversion | `join(matrix.versions, ' ')` | Command line building |
| `format(template, ...args)` | String formatting | `format('v{}.{}.{}', major, minor, patch)` | Version string creation |
| `fromJSON(jsonString)` | JSON parsing | `fromJSON(env.BUILD_METADATA)` | Parse complex data |

**Expression Evaluation Security:**

Variable expression evaluation must be **sandboxed** to prevent code injection and information disclosure. The expression evaluator runs in a restricted context with limited access to system resources.

| Security Boundary | Restriction | Enforcement |
|-------------------|-------------|-------------|
| Function access | Only whitelisted functions | Runtime function call filtering |
| File system | No file system access | Remove all file I/O functions |
| Network access | No network operations | Block all network-related functions |
| Process execution | No command execution | Prevent system() and exec() calls |
| Memory limits | Expression evaluation timeout | Kill evaluation after 5 seconds |

#### Variable Validation and Type Checking

The variable system performs validation to catch common errors and ensure pipeline definitions are robust across different execution contexts.

**Variable Validation Checks:**

| Validation Type | Check Description | Error Response |
|----------------|-------------------|----------------|
| Reference validation | All variable references can be resolved | List undefined variables with suggestions |
| Type consistency | Variable usage matches expected type | Report type mismatches with context |
| Required variables | Critical variables have values | Error on missing required variables |
| Format validation | Variable values match expected patterns | Validate against regex patterns |
| Security constraints | No sensitive data in logs | Redact secure variables in output |

**Type Inference and Checking:**

The system performs **static type analysis** on variable expressions to catch type errors before pipeline execution begins.

| Variable Context | Expected Type | Validation Method | Example Check |
|------------------|---------------|-------------------|---------------|
| timeout fields | integer | Parse as number | `timeout: ${TIMEOUT}` expects integer value |
| boolean conditions | boolean | Evaluate expression type | `if: ${ENABLE_TESTS}` expects boolean result |
| array operations | array | Check collection methods | `matrix: ${TEST_MATRIX}` expects array value |
| string concatenation | string | Check operand types | `name: prefix-${SUFFIX}` expects string suffix |

### Implementation Guidance

The Pipeline Definition Parser forms the foundation of our CI/CD system, requiring careful attention to error handling, performance, and maintainability. This implementation guidance provides production-ready components and detailed skeletons for the core parsing logic.

#### Technology Recommendations

| Component | Simple Option | Advanced Option | Recommendation |
|-----------|---------------|-----------------|----------------|
| YAML Parser | PyYAML with safe_load | ruamel.yaml with source tracking | ruamel.yaml for better error reporting |
| Schema Validation | JSON Schema with jsonschema | Custom validator with detailed errors | Custom validator for CI/CD-specific rules |
| Graph Operations | NetworkX library | Custom graph implementation | NetworkX for prototyping, custom for production |
| Expression Engine | Simple string.Template | Jinja2 or custom parser | Custom parser for security and performance |
| Logging | Python logging module | Structured logging with JSON | Structured logging for better debugging |

#### Recommended File Structure

```
pipeline_parser/
├── __init__.py                    # Package exports
├── yaml_parser.py                 # YAML parsing and validation
├── schema_validator.py            # Pipeline definition schema validation
├── dependency_graph.py            # DAG construction and analysis
├── variable_substitution.py       # Variable resolution and expression evaluation
├── exceptions.py                  # Custom exception types
├── models.py                      # Data model definitions
└── tests/
    ├── test_yaml_parser.py
    ├── test_schema_validator.py
    ├── test_dependency_graph.py
    ├── test_variable_substitution.py
    └── fixtures/
        ├── valid_pipelines/
        ├── invalid_pipelines/
        └── test_data.py
```

#### Infrastructure Starter Code

**exceptions.py - Complete Exception Hierarchy:**

```python
"""Custom exceptions for pipeline parsing operations."""

from typing import List, Dict, Any, Optional


class PipelineParserError(Exception):
    """Base exception for all pipeline parsing errors."""
    
    def __init__(self, message: str, source_location: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.source_location = source_location or {}
        
    def __str__(self) -> str:
        if self.source_location:
            line = self.source_location.get('line', 'unknown')
            column = self.source_location.get('column', 'unknown')
            return f"Line {line}, Column {column}: {super().__str__()}"
        return super().__str__()


class ValidationError(PipelineParserError):
    """Exception for data model validation failures."""
    
    def __init__(self, field_path: str, message: str, 
                 source_location: Optional[Dict[str, Any]] = None,
                 suggestions: Optional[List[str]] = None):
        full_message = f"Validation error in '{field_path}': {message}"
        if suggestions:
            full_message += f"\nSuggestions: {', '.join(suggestions)}"
        super().__init__(full_message, source_location)
        self.field_path = field_path
        self.suggestions = suggestions or []


class CircularDependencyError(PipelineParserError):
    """Exception for circular dependency detection."""
    
    def __init__(self, cycle_path: List[str], 
                 source_location: Optional[Dict[str, Any]] = None):
        cycle_str = " → ".join(cycle_path + [cycle_path[0]])
        message = f"Circular dependency detected: {cycle_str}"
        super().__init__(message, source_location)
        self.cycle_path = cycle_path


class VariableResolutionError(PipelineParserError):
    """Exception for variable substitution failures."""
    
    def __init__(self, variable_name: str, context: str, 
                 available_variables: Optional[List[str]] = None,
                 source_location: Optional[Dict[str, Any]] = None):
        message = f"Cannot resolve variable '${{{variable_name}}}' in {context}"
        if available_variables:
            message += f"\nAvailable variables: {', '.join(available_variables)}"
        super().__init__(message, source_location)
        self.variable_name = variable_name
        self.available_variables = available_variables or []
```

**models.py - Complete Data Model Definitions:**

```python
"""Data models for pipeline definitions and execution state."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field


class JobStatus(Enum):
    """Job execution status enumeration."""
    PENDING = "pending"
    RUNNING = "running" 
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


@dataclass
class PipelineStep:
    """Individual step within a pipeline job."""
    name: str
    script: List[str]
    image: str = "ubuntu:20.04"
    environment: Dict[str, str] = field(default_factory=dict)
    timeout: int = 1800  # 30 minutes default
    retry_count: int = 0
    working_directory: str = "/workspace"
    continue_on_failure: bool = False
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("Step name cannot be empty")
        if not self.script:
            raise ValueError("Step script cannot be empty")


@dataclass  
class PipelineJob:
    """Job definition containing multiple steps."""
    name: str
    steps: List[PipelineStep]
    depends_on: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    logs: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("Job name cannot be empty")
        if not self.steps:
            raise ValueError("Job must have at least one step")


@dataclass
class PipelineDefinition:
    """Complete pipeline definition."""
    name: str
    jobs: Dict[str, PipelineJob]
    global_env: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    timeout: int = 7200  # 2 hours default
    
    def __post_init__(self):
        if not self.name:
            raise ValueError("Pipeline name cannot be empty")
        if not self.jobs:
            raise ValueError("Pipeline must have at least one job")


@dataclass
class ExecutionState:
    """Manages runtime pipeline execution state."""
    pipeline: PipelineDefinition
    job_states: Dict[str, JobStatus] = field(default_factory=dict)
    execution_start: Optional[datetime] = None
    execution_end: Optional[datetime] = None
    
    def __post_init__(self):
        # Initialize job states
        for job_name in self.pipeline.jobs:
            self.job_states[job_name] = JobStatus.PENDING
```

#### Core Logic Skeleton Code

**yaml_parser.py - YAML Parsing Core:**

```python
"""YAML pipeline definition parser with source location tracking."""

import yaml
from typing import Dict, Any, List, Optional
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from .exceptions import PipelineParserError, ValidationError
from .models import PipelineDefinition, PipelineJob, PipelineStep


class SourceAwareYAMLParser:
    """YAML parser that maintains source location information."""
    
    def __init__(self):
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        # TODO: Configure YAML parser for source location tracking
        
    def parse_pipeline_yaml(self, yaml_content: str) -> Dict[str, Any]:
        """Parse YAML content into structured data with source locations.
        
        Args:
            yaml_content: Raw YAML string content
            
        Returns:
            Parsed YAML data with source location annotations
            
        Raises:
            PipelineParserError: If YAML syntax is invalid
        """
        # TODO 1: Parse YAML content using ruamel.yaml parser
        # TODO 2: Extract source location information for each node
        # TODO 3: Handle YAML syntax errors with precise error locations
        # TODO 4: Process YAML anchors and aliases correctly
        # TODO 5: Return structured data with location metadata
        # Hint: Use self.yaml.load() and check for CommentedMap types
        pass
        
    def validate_yaml_structure(self, data: Dict[str, Any]) -> None:
        """Validate basic YAML structure before schema validation.
        
        Args:
            data: Parsed YAML data
            
        Raises:
            ValidationError: If basic structure requirements are not met
        """
        # TODO 1: Check for required top-level keys ('name', 'jobs')
        # TODO 2: Validate that 'jobs' is a dictionary
        # TODO 3: Check that each job has required fields
        # TODO 4: Validate job names follow naming conventions
        # TODO 5: Ensure no reserved keywords are used as job names
        # Hint: Use validate_job_name() function for name validation
        pass


def validate_job_name(name: str) -> bool:
    """Validate job naming conventions.
    
    Args:
        name: Job name to validate
        
    Returns:
        True if name is valid, False otherwise
    """
    # TODO 1: Check name is not empty
    # TODO 2: Validate name contains only alphanumeric chars, hyphens, underscores
    # TODO 3: Ensure name doesn't start with hyphen or underscore  
    # TODO 4: Check name length is reasonable (3-50 chars)
    # TODO 5: Verify name is not a reserved keyword
    # Hint: Use regex pattern: ^[a-zA-Z][a-zA-Z0-9_-]*$
    pass
```

**dependency_graph.py - Dependency Graph Construction:**

```python
"""Dependency graph construction and topological sorting."""

from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict, deque

from .exceptions import CircularDependencyError, ValidationError
from .models import PipelineDefinition, PipelineJob


class DependencyGraph:
    """Manages job dependency relationships and execution ordering."""
    
    def __init__(self, pipeline: PipelineDefinition):
        self.pipeline = pipeline
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.incoming_edges: Dict[str, Set[str]] = defaultdict(set)
        self.outgoing_edges: Dict[str, Set[str]] = defaultdict(set)
        self.ready_queue: List[str] = []
        
    def build_dependency_graph(self) -> None:
        """Construct job dependency graph from pipeline definition.
        
        Raises:
            ValidationError: If job dependencies are invalid
            CircularDependencyError: If circular dependencies exist
        """
        # TODO 1: Initialize graph nodes for each job
        # TODO 2: Validate all job dependencies reference existing jobs
        # TODO 3: Build incoming and outgoing edge sets
        # TODO 4: Detect and report circular dependencies
        # TODO 5: Initialize ready queue with jobs that have no dependencies
        # Hint: Use validate_dependencies() to check references exist
        pass
        
    def validate_dependencies(self, available_jobs: Set[str]) -> List[str]:
        """Check job dependencies exist and return any missing references.
        
        Args:
            available_jobs: Set of all job names in pipeline
            
        Returns:
            List of missing job references (empty if all valid)
        """
        # TODO 1: Iterate through all jobs and their depends_on lists
        # TODO 2: Check each dependency exists in available_jobs set
        # TODO 3: Collect any missing references for error reporting
        # TODO 4: Return list of missing references (or empty list)
        # Hint: Use set operations for efficient membership testing
        pass
        
    def get_execution_order(self) -> List[List[str]]:
        """Return topological sort of jobs grouped by execution level.
        
        Returns:
            List of job name lists, each representing a parallel execution level
            
        Raises:
            CircularDependencyError: If circular dependencies prevent ordering
        """
        # TODO 1: Implement modified Kahn's algorithm
        # TODO 2: Process jobs level by level rather than one at a time  
        # TODO 3: For each level, find all jobs with zero remaining dependencies
        # TODO 4: Remove processed jobs and update dependency counts
        # TODO 5: Continue until all jobs processed or cycle detected
        # Hint: Use a copy of incoming_edges to avoid modifying original graph
        pass
        
    def get_ready_jobs(self) -> List[str]:
        """Return jobs ready for execution (no pending dependencies).
        
        Returns:
            List of job names ready to execute
        """
        # TODO 1: Return current contents of ready_queue
        # TODO 2: Ensure ready_queue is kept up-to-date by other methods
        # Hint: This should be a simple accessor method
        pass
        
    def mark_job_completed(self, job_name: str, success: bool) -> List[str]:
        """Mark job as completed and return newly eligible jobs.
        
        Args:
            job_name: Name of completed job
            success: Whether job completed successfully
            
        Returns:
            List of job names that are now ready to execute
        """
        # TODO 1: Remove completed job from graph
        # TODO 2: For each job that depended on this job, decrement dependency count
        # TODO 3: Add jobs with zero remaining dependencies to ready queue
        # TODO 4: If job failed, handle failure propagation based on policy
        # TODO 5: Return list of newly ready jobs
        # Hint: Update both ready_queue and return the newly ready jobs
        pass
        
    def detect_cycles(self) -> Optional[List[str]]:
        """Detect circular dependencies and return shortest cycle path.
        
        Returns:
            List of job names forming a cycle, or None if no cycles exist
        """
        # TODO 1: Use depth-first search to detect back edges
        # TODO 2: When back edge found, trace path to identify cycle
        # TODO 3: Return shortest cycle for clearest error reporting
        # TODO 4: Return None if no cycles detected
        # Hint: Use recursive DFS with visited/visiting state tracking
        pass
```

**variable_substitution.py - Variable Resolution Engine:**

```python
"""Variable substitution and expression evaluation."""

import re
import os
from typing import Dict, Any, List, Optional, Union
from string import Template

from .exceptions import VariableResolutionError


class VariableSubstitution:
    """Handles variable resolution and expression evaluation."""
    
    def __init__(self, global_env: Dict[str, str], system_env: Optional[Dict[str, str]] = None):
        self.global_env = global_env
        self.system_env = system_env or dict(os.environ)
        self.variable_pattern = re.compile(r'\$\{([^}]+)\}')
        
    def resolve_variables(self, text: str, job_env: Optional[Dict[str, str]] = None) -> str:
        """Resolve all variable references in text string.
        
        Args:
            text: String containing variable references
            job_env: Job-specific environment variables
            
        Returns:
            String with all variables resolved
            
        Raises:
            VariableResolutionError: If any variable cannot be resolved
        """
        # TODO 1: Find all variable references using regex pattern
        # TODO 2: For each reference, resolve value using variable hierarchy
        # TODO 3: Handle default values (${VAR:-default}) and conditional substitution
        # TODO 4: Detect and prevent infinite recursion in variable resolution
        # TODO 5: Replace all resolved variables in text and return result
        # Hint: Use self.variable_pattern.findall() to extract variable names
        pass
        
    def resolve_single_variable(self, var_name: str, job_env: Optional[Dict[str, str]] = None) -> str:
        """Resolve a single variable name to its value.
        
        Args:
            var_name: Variable name (without ${} syntax)
            job_env: Job-specific environment variables
            
        Returns:
            Resolved variable value
            
        Raises:
            VariableResolutionError: If variable cannot be resolved
        """
        # TODO 1: Parse variable name for default value syntax (VAR:-default)
        # TODO 2: Check job_env first (highest priority)
        # TODO 3: Check global_env next
        # TODO 4: Check system_env last
        # TODO 5: Use default value if provided, otherwise raise error
        # Hint: Variable hierarchy is job_env > global_env > system_env > default
        pass
        
    def validate_variable_references(self, text: str, available_vars: Dict[str, str]) -> List[str]:
        """Validate all variable references can be resolved.
        
        Args:
            text: String to validate
            available_vars: Combined variable context for validation
            
        Returns:
            List of undefined variable names (empty if all valid)
        """
        # TODO 1: Extract all variable references from text
        # TODO 2: For each reference, check if it can be resolved
        # TODO 3: Handle default value syntax - variables with defaults are always valid
        # TODO 4: Return list of undefined variables for error reporting
        # Hint: Variables with default values (${VAR:-default}) are always resolvable
        pass
        
    def get_available_variables(self, job_env: Optional[Dict[str, str]] = None) -> List[str]:
        """Get list of all available variable names in current context.
        
        Args:
            job_env: Job-specific environment variables
            
        Returns:
            Sorted list of all available variable names
        """
        # TODO 1: Combine all variable sources (job_env, global_env, system_env)
        # TODO 2: Return sorted list of variable names for error messages
        # Hint: Use set() to deduplicate before sorting
        pass
```

#### Milestone Checkpoint

After implementing the Pipeline Definition Parser, verify correct functionality with these checkpoints:

**Test Command:**
```bash
python -m pytest pipeline_parser/tests/ -v
```

**Expected Test Results:**
- YAML parsing tests: 15+ test cases covering valid/invalid syntax
- Schema validation tests: 20+ test cases for required fields, type checking
- Dependency graph tests: 10+ test cases for DAG construction, cycle detection  
- Variable substitution tests: 12+ test cases for resolution, error handling

**Manual Verification:**
1. **Valid Pipeline Parsing**: Create a sample YAML pipeline with multiple jobs and dependencies
2. **Error Reporting**: Test with invalid YAML and verify error messages include line numbers
3. **Dependency Resolution**: Verify complex dependency graphs produce correct execution order
4. **Variable Substitution**: Test variable resolution with environment overrides and defaults

**Success Indicators:**
- Parser correctly handles all YAML syntax features (anchors, aliases, multi-line strings)
- Validation errors include precise source locations and helpful suggestions
- Dependency graph correctly identifies parallel execution opportunities
- Variable substitution works with nested references and complex expressions
- All error messages are actionable and include context for debugging

**Common Issues and Debugging:**
- **Parser hangs**: Check for infinite loops in variable resolution
- **Missing source locations**: Verify ruamel.yaml configuration preserves location data
- **Incorrect dependency order**: Debug topological sort with small test cases
- **Variable resolution fails**: Check variable hierarchy precedence rules


## Job Executor (Milestone 2)

> **Milestone(s):** Milestone 2: Job Executor

Think of the Job Executor as a **skilled foreman managing a construction crew**. Just as a foreman takes architectural blueprints and coordinates individual workers to build a structure safely and efficiently, the Job Executor takes the parsed pipeline definition and orchestrates the actual execution of jobs in isolated environments. The foreman ensures each worker has the right tools, workspace, and materials while monitoring progress, handling delays, and maintaining safety protocols. Similarly, the Job Executor provides each job with the correct Docker environment, manages resource allocation, streams real-time progress updates, and handles failures with retry logic.

![Pipeline Execution Sequence](./diagrams/execution-sequence.svg)

The Job Executor serves as the **execution orchestrator** that bridges the gap between the static pipeline definition created by the Parser and the dynamic runtime environment where actual work happens. While the Pipeline Parser (Milestone 1) focused on understanding what needs to be done, the Job Executor focuses on actually doing it safely, reliably, and observably.

### Container Isolation

Container isolation forms the **security perimeter** and **reproducibility foundation** of our CI/CD pipeline execution. Think of container isolation like **individual clean rooms in a semiconductor manufacturing facility**. Each clean room provides a controlled, sterile environment where specific manufacturing steps can occur without contamination from external factors or interference between concurrent processes. Just as each clean room has precisely controlled temperature, humidity, and particle levels, each container provides a job with exactly the operating system, libraries, runtime versions, and environment variables it needs.

The fundamental challenge that container isolation solves is **environment drift** - the problem where code that works in development fails in production due to subtle differences in system configuration, installed packages, or environment variables. Traditional CI/CD systems that run jobs directly on shared hosts suffer from what we call "ghost dependencies" - jobs that accidentally rely on tools, libraries, or configuration left behind by previous jobs. Container isolation eliminates ghost dependencies by providing each job with a **clean slate environment** that exactly matches the specified Docker image.

> **Decision: Docker as Primary Isolation Mechanism**
> - **Context**: Jobs need isolated execution environments to prevent interference and ensure reproducibility across different hosts
> - **Options Considered**: 
>   1. Process-level isolation with chroot and cgroups
>   2. Virtual machines for each job
>   3. Docker containers
> - **Decision**: Docker containers with configurable resource limits
> - **Rationale**: Docker provides the optimal balance of isolation strength, resource efficiency, and ecosystem compatibility. VMs provide stronger isolation but consume excessive resources and have slow startup times. Process-level isolation is resource-efficient but complex to implement securely and lacks the ecosystem of pre-built images.
> - **Consequences**: Jobs can specify any Docker image, enabling language-specific toolchains. Container startup overhead (1-3 seconds) is acceptable for most CI/CD workloads. Security depends on Docker daemon configuration and kernel capabilities.

| Container Isolation Feature | Purpose | Implementation Strategy |
|------------------------------|---------|------------------------|
| **Image Specification** | Define exact runtime environment per job | Each `PipelineStep` specifies `image` field with Docker image name and tag |
| **Environment Variable Injection** | Provide job-specific configuration and secrets | Merge job-level and global environment variables, inject into container |
| **Working Directory Control** | Set consistent starting directory for commands | Mount workspace volume and set container working directory |
| **Resource Limits** | Prevent jobs from consuming excessive CPU/memory | Configure Docker cgroups limits based on job requirements |
| **Network Isolation** | Control outbound connectivity and service access | Use Docker networks to restrict or allow external communication |
| **Volume Mounting** | Share workspace and artifacts between host and container | Mount workspace directory for source code and artifact exchange |

The container lifecycle follows a **prepare-execute-cleanup** pattern that ensures both security and resource efficiency. During the preparation phase, the Job Executor validates the specified Docker image exists locally or can be pulled from a registry. If the image requires authentication (private registry), the executor uses stored credentials while ensuring sensitive information never appears in logs. The preparation phase also constructs the complete environment variable map by merging global pipeline variables, job-specific variables, and step-level overrides, with step-level variables taking precedence to allow fine-grained control.

The execution phase creates a new container instance with carefully configured isolation boundaries. The executor mounts the workspace directory to provide access to source code and enable artifact collection, but uses read-only mounts for system directories to prevent accidental or malicious modification of the container filesystem. Resource limits are applied through Docker's cgroups integration, setting memory limits to prevent out-of-memory conditions from affecting other jobs and CPU limits to ensure fair resource sharing in multi-tenant environments.

> The key insight for secure container execution is that isolation is a multi-layered defense strategy. No single mechanism provides complete protection, so we combine Docker's namespace isolation, cgroups resource limits, read-only filesystem mounts, and restricted network access to create defense in depth.

The cleanup phase ensures that containers don't accumulate over time and consume system resources. The executor implements a **container lifecycle management** strategy that removes containers immediately after job completion, regardless of success or failure. For debugging purposes, the executor optionally preserves failed containers for a configurable retention period, allowing developers to inspect the exact state that led to failure.

**Container Security Considerations:**

| Security Concern | Risk Level | Mitigation Strategy |
|------------------|------------|-------------------|
| **Container Escape** | High | Use recent Docker version with user namespaces enabled |
| **Privileged Execution** | High | Never run containers in privileged mode; drop unnecessary capabilities |
| **Secret Exposure** | Medium | Inject secrets as environment variables, never in command arguments or logs |
| **Resource Exhaustion** | Medium | Apply memory and CPU limits; implement job timeout enforcement |
| **Network Access** | Low | Use custom Docker networks to control outbound connectivity |

The executor maintains a **container registry cache** to improve performance and reduce external dependencies. When a job requests a Docker image, the executor first checks if the image exists locally with the correct tag. For performance-critical environments, the executor can be configured to pre-pull commonly used images during system startup, reducing job execution latency.

### Logging and Monitoring

Real-time logging and monitoring provide the **observability foundation** that makes CI/CD pipeline execution transparent and debuggable. Think of the logging system as a **mission control center** for space operations - just as mission control needs real-time telemetry from spacecraft to monitor system health, make decisions, and respond to anomalies, the CI/CD pipeline needs comprehensive logging to track job progress, diagnose failures, and provide feedback to developers.

The fundamental challenge in CI/CD logging is capturing and organizing information from **multiple concurrent execution contexts** while maintaining the ability to trace individual job execution paths. Unlike traditional application logging where a single process generates a sequential log stream, CI/CD pipelines execute multiple jobs simultaneously, each potentially running multiple steps in containers that may start and stop rapidly. The logging system must coordinate these multiple information streams into a coherent, searchable, and actionable format.

![Job Execution State Machine](./diagrams/job-state-machine.svg)

Our logging architecture implements a **hierarchical structured logging** approach that captures information at multiple levels of granularity. At the highest level, pipeline-wide events track overall execution progress, job scheduling decisions, and resource allocation. At the job level, logging captures job state transitions, environment setup, and aggregate execution results. At the step level, logging streams real-time output from container execution while maintaining context about which job and pipeline generated the output.

> **Decision: Structured Logging with Context Propagation**
> - **Context**: Need to correlate log messages across multiple concurrent jobs and steps while enabling efficient searching and filtering
> - **Options Considered**:
>   1. Plain text logs with manual parsing
>   2. JSON structured logging with context fields
>   3. Distributed tracing with OpenTelemetry
> - **Decision**: JSON structured logging with hierarchical context propagation
> - **Rationale**: JSON logging provides machine-readable structure for automated analysis while remaining human-readable. Context propagation ensures every log message includes pipeline ID, job name, and step name for correlation. Distributed tracing adds complexity that exceeds current requirements.
> - **Consequences**: Log messages are larger due to repeated context fields, but this enables powerful filtering and correlation capabilities. Integration with log aggregation systems is straightforward.

| Logging Component | Responsibility | Data Captured |
|-------------------|---------------|---------------|
| **Pipeline Logger** | Track overall pipeline execution lifecycle | Pipeline start/end, job scheduling decisions, resource allocation, final status summary |
| **Job Logger** | Monitor individual job progress and state changes | Job queued/started/completed, environment setup, artifact operations, retry attempts |
| **Step Logger** | Stream real-time output from container execution | Command stdout/stderr, exit codes, execution duration, resource usage |
| **System Logger** | Capture executor infrastructure events | Docker operations, volume mounts, network setup, container lifecycle |

The **real-time log streaming** mechanism addresses one of the most critical user experience aspects of CI/CD pipelines - providing immediate feedback about job progress. Developers need to see command output as it happens, not after job completion, to identify issues quickly and make decisions about whether to cancel long-running jobs. The streaming implementation uses a **publisher-subscriber pattern** where step execution publishes log lines to a message bus, and multiple subscribers can consume the stream for different purposes - real-time display in web interfaces, persistent storage for historical analysis, and alerting systems for failure detection.

The step-level logging implementation captures both **standard output streams** (stdout and stderr) from container execution while maintaining strict separation to preserve the semantic meaning of each stream. Standard output typically contains the actual results or progress information from build tools, while standard error contains diagnostic information and error messages. The logger preserves this distinction while adding metadata about the source job and step, enabling sophisticated filtering and analysis.

**Log Message Structure:**

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `timestamp` | ISO 8601 | When the log message was generated | `2024-01-15T10:30:25.123Z` |
| `level` | String | Log level (INFO, WARN, ERROR, DEBUG) | `INFO` |
| `pipeline_id` | String | Unique identifier for the pipeline execution | `build-456-main-abc123` |
| `job_name` | String | Name of the job generating the message | `test-backend` |
| `step_name` | String | Name of the step (if applicable) | `run-unit-tests` |
| `message` | String | Human-readable log message | `Running pytest with coverage` |
| `stream` | String | Output stream source (stdout, stderr, system) | `stdout` |
| `container_id` | String | Docker container ID (for step-level logs) | `a1b2c3d4e5f6` |

The logging system implements **buffering and batching strategies** to balance real-time responsiveness with system performance. Individual log lines are immediately available for streaming to user interfaces, but persistent storage operations are batched to reduce I/O overhead. The system uses a **dual-buffer approach** where one buffer collects incoming log lines while another buffer is being flushed to storage, ensuring continuous operation without blocking.

Log retention and archival follow a **tiered storage strategy** that balances accessibility with storage costs. Recent logs (last 30 days) are stored in high-performance storage for immediate access and searching. Older logs are compressed and moved to cheaper archival storage while maintaining the ability to retrieve them for long-term analysis or compliance requirements.

**Monitoring and Alerting Integration:**

The logging system provides **structured metrics** that enable automated monitoring and alerting. Key metrics include job execution duration, failure rates, resource utilization, and queue depths. These metrics are exposed in Prometheus format for integration with standard monitoring stacks.

| Metric Type | Metric Name | Description | Labels |
|-------------|-------------|-------------|--------|
| **Counter** | `jobs_total` | Total number of jobs executed | `status={success,failed,cancelled}`, `job_name` |
| **Histogram** | `job_duration_seconds` | Job execution time distribution | `job_name`, `pipeline_name` |
| **Gauge** | `active_jobs` | Number of currently executing jobs | `job_name` |
| **Counter** | `container_operations_total` | Docker operations performed | `operation={create,start,stop,remove}` |

### Timeout and Retry Logic

Timeout and retry logic provide the **resilience mechanisms** that handle the inherent unreliability of distributed systems and external dependencies. Think of timeout and retry logic as the **circuit breakers and backup generators** in an electrical grid - they detect when components are not responding normally and automatically switch to alternative approaches or safe shutdown procedures to prevent cascading failures.

The fundamental challenge in CI/CD timeout and retry logic is distinguishing between **transient failures** that should be retried and **permanent failures** that indicate actual problems with code, configuration, or infrastructure. A network timeout when downloading dependencies might resolve on retry, but a compilation error due to syntax problems will fail consistently regardless of retry attempts. The timeout and retry system must make intelligent decisions about when to persist and when to give up.

Our timeout implementation operates at **multiple hierarchical levels** to provide fine-grained control while maintaining reasonable defaults. At the pipeline level, an overall timeout prevents pipelines from running indefinitely due to hung jobs or infinite loops. At the job level, timeouts ensure that jobs with stuck steps don't block other jobs indefinitely. At the step level, timeouts handle individual commands that may hang due to waiting for user input, network timeouts, or infinite loops.

> **Decision: Hierarchical Timeout Architecture**
> - **Context**: Different components have different timeout requirements, and timeouts should be configurable but have sensible defaults
> - **Options Considered**:
>   1. Single global timeout for entire pipeline
>   2. Per-job timeouts only
>   3. Hierarchical timeouts (pipeline > job > step)
> - **Decision**: Hierarchical timeouts with inheritance and override capability
> - **Rationale**: Hierarchical timeouts provide maximum flexibility while maintaining simplicity. Step timeouts handle immediate command hangs, job timeouts handle aggregated step execution, and pipeline timeouts provide ultimate safety. Lower-level timeouts can be more aggressive than higher-level timeouts.
> - **Consequences**: More complex timeout management logic, but enables precise control over execution bounds and prevents resource waste from hung executions.

| Timeout Level | Default Value | Purpose | Override Behavior |
|---------------|---------------|---------|-------------------|
| **Pipeline Timeout** | 4 hours | Prevent indefinite pipeline execution | Specified in `PipelineDefinition.timeout` field |
| **Job Timeout** | 1 hour | Prevent individual jobs from running too long | Calculated as sum of step timeouts + overhead |
| **Step Timeout** | 30 minutes | Handle hung commands and infinite loops | Specified in `PipelineStep.timeout` field |

The timeout implementation uses **signal-based termination** with graceful degradation to ensure containers can clean up resources properly. When a timeout occurs, the executor first sends a SIGTERM signal to the container process, allowing it 30 seconds to shut down gracefully. If the process doesn't exit within the grace period, the executor sends SIGKILL to force termination and immediately removes the container to prevent resource leaks.

Retry logic implements an **exponential backoff strategy** with jitter to prevent thundering herd problems when multiple jobs retry simultaneously. The base retry delay starts at 5 seconds and doubles with each attempt, up to a maximum delay of 2 minutes. Random jitter (±25%) is added to spread retry attempts across time and reduce system load spikes.

**Retry Decision Logic:**

The retry system categorizes failures into **retryable** and **non-retryable** categories based on exit codes, error messages, and execution context. This classification prevents wasting time and resources on failures that won't improve with retry attempts.

| Failure Category | Retry Behavior | Examples | Detection Method |
|------------------|---------------|----------|------------------|
| **Transient Infrastructure** | Retry with backoff | Network timeouts, Docker daemon errors, disk full | Exit code analysis, error message patterns |
| **Transient Dependencies** | Retry with backoff | Package download failures, service unavailable | HTTP status codes, network error types |
| **Permanent Code Issues** | No retry | Compilation errors, test failures, syntax errors | Exit codes 1-2, specific tool error patterns |
| **Configuration Errors** | No retry | Missing environment variables, invalid parameters | Exit code 127, configuration validation failures |
| **Resource Exhaustion** | Retry after delay | Out of memory, disk space | System error codes, resource monitoring |

The retry mechanism maintains **detailed retry history** that helps developers understand why jobs failed and what recovery attempts were made. This history is essential for debugging intermittent issues and tuning retry policies for specific job types.

| Retry History Field | Type | Purpose |
|---------------------|------|---------|
| `attempt_number` | Integer | Which retry attempt (1-based) |
| `started_at` | Timestamp | When this attempt began |
| `finished_at` | Timestamp | When this attempt completed |
| `exit_code` | Integer | Container exit code |
| `failure_reason` | String | Categorized failure type |
| `retry_delay` | Duration | How long before next attempt |

**Advanced Retry Strategies:**

For sophisticated CI/CD workflows, the retry system supports **conditional retry policies** based on job characteristics and failure patterns. Long-running integration tests might have different retry policies than quick unit tests, and jobs that interact with external services might have more aggressive retry policies than jobs that only process local files.

The system implements **retry budget limiting** to prevent infinite retry loops while still allowing reasonable recovery attempts. Each job has a maximum retry count (default: 3) and a maximum total retry time (default: 30 minutes) to ensure that even with exponential backoff, jobs eventually fail definitively rather than consuming resources indefinitely.

> The critical insight for effective retry logic is that retries should be fast for likely-transient failures and slower for likely-permanent failures. Network timeouts should retry quickly, but compilation failures should have longer delays to avoid overwhelming build infrastructure with repeated identical work.

**⚠️ Pitfall: Retry Amplification**
A common mistake is implementing retry logic that amplifies load on already-struggling external services. When a dependency service is experiencing high load, immediate retries from all CI/CD jobs can worsen the situation. The solution is implementing **circuit breaker patterns** where repeated failures to the same external service temporarily disable retries for all jobs, allowing the service to recover.

**⚠️ Pitfall: Inconsistent Retry State**
Another pitfall is losing retry state when the Job Executor process restarts, causing jobs to lose track of how many retry attempts have been made. The solution is **persisting retry state** in the same storage system used for job status, ensuring retry policies are enforced consistently across executor restarts.

### Implementation Guidance

This subsection provides concrete implementation patterns and starter code to help you build the Job Executor component effectively.

**A. Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| **Container Runtime** | Docker CLI via subprocess calls | Docker Python SDK with async support |
| **Logging Framework** | Python `logging` with JSON formatter | Structured logging with `structlog` |
| **Process Management** | `subprocess.Popen` with timeout | `asyncio.subprocess` for concurrent execution |
| **Configuration** | YAML config files with `PyYAML` | Environment-based config with `pydantic` |
| **Metrics Collection** | Simple counters in memory | Prometheus client library |

**B. Recommended File Structure:**

```
ci_cd_pipeline/
  executor/
    __init__.py
    job_executor.py              ← main Job Executor implementation
    docker_client.py             ← Docker integration wrapper
    logging_config.py            ← structured logging setup
    retry_policies.py            ← timeout and retry logic
    container_manager.py         ← container lifecycle management
    execution_state.py           ← job execution state tracking
  tests/
    test_job_executor.py         ← executor integration tests
    test_docker_client.py        ← container execution tests
    test_retry_logic.py          ← timeout and retry tests
```

**C. Infrastructure Starter Code:**

Complete Docker client wrapper with error handling and logging:

```python
"""Docker integration wrapper for secure container execution."""

import docker
import logging
import time
from typing import Dict, List, Tuple, Iterator, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DockerClient:
    """Wrapper around Docker client with CI/CD-specific functionality."""
    
    def __init__(self, base_url: str = "unix://var/run/docker.sock"):
        self.client = docker.DockerClient(base_url=base_url)
        self._validate_docker_connection()
    
    def _validate_docker_connection(self) -> None:
        """Verify Docker daemon is accessible and responsive."""
        try:
            self.client.ping()
            logger.info("Docker daemon connection established")
        except docker.errors.DockerException as e:
            logger.error(f"Cannot connect to Docker daemon: {e}")
            raise RuntimeError(f"Docker daemon unavailable: {e}")
    
    def pull_image(self, image: str, timeout: int = 300) -> bool:
        """Pull Docker image with timeout and retry logic."""
        try:
            logger.info(f"Pulling Docker image: {image}")
            self.client.images.pull(image, timeout=timeout)
            logger.info(f"Successfully pulled image: {image}")
            return True
        except docker.errors.ImageNotFound:
            logger.error(f"Image not found in registry: {image}")
            return False
        except docker.errors.APIError as e:
            logger.error(f"Docker API error pulling {image}: {e}")
            return False
    
    def image_exists_locally(self, image: str) -> bool:
        """Check if image exists in local Docker cache."""
        try:
            self.client.images.get(image)
            return True
        except docker.errors.ImageNotFound:
            return False
    
    @contextmanager
    def create_container(self, image: str, command: List[str], 
                        environment: Dict[str, str] = None,
                        working_dir: str = "/workspace",
                        volumes: Dict[str, str] = None,
                        memory_limit: str = "512m",
                        cpu_limit: str = "1.0"):
        """Context manager for container lifecycle with automatic cleanup."""
        container = None
        try:
            container_config = {
                'image': image,
                'command': command,
                'environment': environment or {},
                'working_dir': working_dir,
                'volumes': volumes or {},
                'mem_limit': memory_limit,
                'cpu_period': 100000,
                'cpu_quota': int(float(cpu_limit) * 100000),
                'detach': True,
                'remove': False,  # Manual cleanup for debugging
            }
            
            logger.debug(f"Creating container with config: {container_config}")
            container = self.client.containers.create(**container_config)
            logger.info(f"Created container {container.short_id} from image {image}")
            
            yield container
            
        except docker.errors.ImageNotFound:
            logger.error(f"Image not found: {image}")
            raise RuntimeError(f"Container image not available: {image}")
        except docker.errors.APIError as e:
            logger.error(f"Docker API error creating container: {e}")
            raise RuntimeError(f"Container creation failed: {e}")
        finally:
            if container:
                try:
                    container.remove(force=True)
                    logger.debug(f"Removed container {container.short_id}")
                except docker.errors.APIError as e:
                    logger.warning(f"Failed to remove container {container.short_id}: {e}")

def run_command(image: str, command: List[str], environment: Dict[str, str] = None,
                working_dir: str = "/workspace", volumes: Dict[str, str] = None,
                timeout: int = 1800) -> Tuple[int, str]:
    """Execute command in Docker container and return exit code and output."""
    client = DockerClient()
    
    # Ensure image is available locally
    if not client.image_exists_locally(image):
        if not client.pull_image(image):
            raise RuntimeError(f"Unable to obtain Docker image: {image}")
    
    with client.create_container(image, command, environment, working_dir, volumes) as container:
        try:
            container.start()
            logger.info(f"Started container {container.short_id} executing: {' '.join(command)}")
            
            # Wait for completion with timeout
            result = container.wait(timeout=timeout)
            exit_code = result['StatusCode']
            
            # Collect output
            output = container.logs(stdout=True, stderr=True).decode('utf-8')
            
            logger.info(f"Container {container.short_id} completed with exit code {exit_code}")
            return exit_code, output
            
        except docker.errors.APIError as e:
            logger.error(f"Container execution error: {e}")
            raise RuntimeError(f"Command execution failed: {e}")
```

Complete structured logging configuration:

```python
"""Structured logging configuration for CI/CD pipeline execution."""

import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional

class PipelineFormatter(logging.Formatter):
    """Custom formatter that adds pipeline context to log messages."""
    
    def __init__(self, pipeline_id: str = None, job_name: str = None):
        super().__init__()
        self.pipeline_id = pipeline_id
        self.job_name = job_name
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON with pipeline context."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name,
            'pipeline_id': getattr(record, 'pipeline_id', self.pipeline_id),
            'job_name': getattr(record, 'job_name', self.job_name),
            'step_name': getattr(record, 'step_name', None),
            'container_id': getattr(record, 'container_id', None),
            'stream': getattr(record, 'stream', 'system'),
        }
        
        # Remove None values for cleaner output
        log_data = {k: v for k, v in log_data.items() if v is not None}
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)

def setup_logging(level: str = "INFO", log_file: Optional[str] = None, 
                 pipeline_id: str = None, job_name: str = None) -> None:
    """Configure structured logging for pipeline execution."""
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter with pipeline context
    formatter = PipelineFormatter(pipeline_id=pipeline_id, job_name=job_name)
    
    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    
    handlers = [console_handler]
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        format='',  # Formatter handles all formatting
    )
    
    # Reduce noise from external libraries
    logging.getLogger('docker').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
```

**D. Core Logic Skeleton Code:**

Main Job Executor with execution orchestration:

```python
"""Core Job Executor implementation with container isolation and retry logic."""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Iterator
from enum import Enum
from dataclasses import dataclass, field

from .docker_client import DockerClient, run_command
from .retry_policies import RetryPolicy, ExponentialBackoffRetry
from .logging_config import setup_logging

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"

@dataclass
class PipelineStep:
    name: str
    script: List[str]
    image: str = "ubuntu:20.04"
    environment: Dict[str, str] = field(default_factory=dict)
    timeout: int = 1800  # 30 minutes default
    retry_count: int = 3
    working_directory: str = "/workspace"
    continue_on_failure: bool = False

@dataclass
class PipelineJob:
    name: str
    steps: List[PipelineStep]
    depends_on: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    logs: List[str] = field(default_factory=list)

class JobExecutor:
    """Executes pipeline jobs in isolated Docker containers with retry logic."""
    
    def __init__(self, workspace_dir: str = "/tmp/ci_workspace", 
                 docker_client: DockerClient = None):
        self.workspace_dir = workspace_dir
        self.docker_client = docker_client or DockerClient()
        self.active_jobs: Dict[str, threading.Thread] = {}
        self.job_states: Dict[str, JobStatus] = {}
        self.retry_policies: Dict[str, RetryPolicy] = {}
        
        # TODO 1: Initialize workspace directory with proper permissions
        # TODO 2: Set up job execution thread pool for concurrent job processing
        # TODO 3: Configure default retry policy for transient failures
        
    def execute_job(self, job: PipelineJob, global_env: Dict[str, str] = None) -> bool:
        """Execute a single job with all its steps in sequence.
        
        Returns True if job completed successfully, False otherwise.
        """
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        
        logger.info(f"Starting job execution: {job.name}", extra={
            'job_name': job.name,
            'step_count': len(job.steps)
        })
        
        try:
            # TODO 4: Merge global environment with job-specific environment
            # TODO 5: Create job-specific workspace directory
            # TODO 6: Execute each step in sequence, stopping on first failure (unless continue_on_failure=True)
            # TODO 7: Collect step outputs and update job logs
            # TODO 8: Handle step timeouts and retry logic
            # TODO 9: Clean up job workspace and temporary files
            
            job.status = JobStatus.SUCCESS
            return True
            
        except Exception as e:
            logger.error(f"Job execution failed: {job.name}", extra={
                'job_name': job.name,
                'error': str(e)
            })
            job.status = JobStatus.FAILED
            return False
            
        finally:
            job.finished_at = datetime.utcnow()
            duration = (job.finished_at - job.started_at).total_seconds()
            logger.info(f"Job completed: {job.name}", extra={
                'job_name': job.name,
                'status': job.status.value,
                'duration_seconds': duration
            })
    
    def execute_step(self, step: PipelineStep, job_env: Dict[str, str], 
                    workspace_path: str) -> Tuple[bool, str]:
        """Execute a single pipeline step in isolated container.
        
        Returns (success, output) tuple.
        """
        logger.info(f"Executing step: {step.name}", extra={
            'step_name': step.name,
            'image': step.image,
            'timeout': step.timeout
        })
        
        # TODO 10: Validate step configuration (image, script, environment)
        # TODO 11: Prepare container volumes mapping workspace directory
        # TODO 12: Execute step script in container with timeout
        # TODO 13: Handle container execution failures and categorize for retry
        # TODO 14: Capture and stream real-time output from container
        # TODO 15: Apply retry logic for transient failures
        
        pass
    
    def streaming_run(self, image: str, command: List[str], **kwargs) -> Iterator[str]:
        """Execute container command with real-time log streaming.
        
        Yields log lines as they are produced by the container.
        """
        # TODO 16: Create container with streaming output configuration
        # TODO 17: Start container and attach to stdout/stderr streams
        # TODO 18: Yield log lines in real-time while container runs
        # TODO 19: Handle container termination and cleanup
        # TODO 20: Propagate container exit code to caller
        
        pass
    
    def should_retry_failure(self, exit_code: int, output: str, attempt: int) -> bool:
        """Determine if a step failure should be retried based on failure characteristics."""
        
        # TODO 21: Analyze exit code to categorize failure type
        # TODO 22: Check output for transient error patterns (network, resource)
        # TODO 23: Consider attempt count against retry limits
        # TODO 24: Apply exponential backoff delay calculation
        
        pass
    
    def cancel_job(self, job_name: str) -> bool:
        """Cancel a running job and clean up its resources."""
        
        # TODO 25: Mark job status as CANCELLED
        # TODO 26: Terminate running containers for this job
        # TODO 27: Clean up job workspace and temporary files
        # TODO 28: Notify any waiting dependencies that job was cancelled
        
        pass

DEFAULT_TIMEOUT = 3600  # 1 hour default timeout
DEFAULT_IMAGE = "ubuntu:20.04"
```

Simple retry policy implementation:

```python
"""Retry policies for handling transient failures in job execution."""

import time
import random
import logging
from abc import ABC, abstractmethod
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)

class FailureCategory(Enum):
    TRANSIENT_INFRASTRUCTURE = "transient_infrastructure"
    TRANSIENT_DEPENDENCIES = "transient_dependencies" 
    PERMANENT_CODE_ISSUES = "permanent_code_issues"
    CONFIGURATION_ERRORS = "configuration_errors"
    RESOURCE_EXHAUSTION = "resource_exhaustion"

class RetryPolicy(ABC):
    """Abstract base class for job retry policies."""
    
    @abstractmethod
    def should_retry(self, attempt: int, exit_code: int, output: str) -> bool:
        """Determine if failure should be retried."""
        pass
    
    @abstractmethod
    def get_retry_delay(self, attempt: int) -> float:
        """Calculate delay before next retry attempt."""
        pass

class ExponentialBackoffRetry(RetryPolicy):
    """Exponential backoff retry policy with jitter."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 5.0, 
                 max_delay: float = 120.0, jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
    
    def should_retry(self, attempt: int, exit_code: int, output: str) -> bool:
        """Determine retry based on attempt count and failure category."""
        if attempt >= self.max_retries:
            return False
        
        failure_category = self._categorize_failure(exit_code, output)
        
        # TODO 29: Implement failure categorization logic
        # TODO 30: Return retry decision based on failure type
        
        return failure_category in [
            FailureCategory.TRANSIENT_INFRASTRUCTURE,
            FailureCategory.TRANSIENT_DEPENDENCIES,
            FailureCategory.RESOURCE_EXHAUSTION
        ]
    
    def get_retry_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with optional jitter."""
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        
        if self.jitter:
            # Add ±25% jitter to prevent thundering herd
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(delay, 1.0)  # Minimum 1 second delay
    
    def _categorize_failure(self, exit_code: int, output: str) -> FailureCategory:
        """Categorize failure type based on exit code and output analysis."""
        
        # TODO 31: Analyze exit codes for standard failure patterns
        # TODO 32: Search output for network timeout indicators
        # TODO 33: Detect resource exhaustion (OOM, disk full) patterns
        # TODO 34: Identify configuration errors vs code issues
        
        pass
```

**E. Language-Specific Hints:**

- **Docker Integration**: Use the `docker` Python library instead of subprocess calls for better error handling and container management
- **Threading Safety**: Use `threading.Lock` around shared state modifications in the job executor
- **Signal Handling**: Implement proper signal handlers (SIGTERM, SIGINT) for graceful shutdown of running jobs
- **Resource Management**: Use context managers (`with` statements) for container lifecycle and file operations
- **Environment Variables**: Use `os.environ.copy()` as base and update with job-specific variables to avoid pollution
- **Timeout Implementation**: Use `signal.alarm()` on Unix systems or `threading.Timer` for cross-platform timeout handling

**F. Milestone Checkpoint:**

After implementing the Job Executor, verify correct functionality:

**Unit Tests to Run:**
```bash
python -m pytest tests/test_job_executor.py -v
python -m pytest tests/test_docker_client.py -v  
python -m pytest tests/test_retry_logic.py -v
```

**Manual Integration Test:**
1. Create a simple pipeline definition with 2-3 steps
2. Run: `python -m executor.job_executor --pipeline=test_pipeline.yaml`
3. Verify you see real-time log output from each step
4. Check that failed steps retry according to policy
5. Confirm containers are cleaned up after job completion

**Expected Behavior:**
- Jobs execute in isolated containers with specified images
- Real-time log streaming shows command output as it happens  
- Failed steps retry with exponential backoff delays
- Timeout enforcement terminates long-running steps
- Container resources are cleaned up after job completion

**Signs Something is Wrong:**
- **Containers accumulate over time**: Check container cleanup logic in finally blocks
- **Jobs hang indefinitely**: Verify timeout implementation and signal handling
- **Retry loops never stop**: Check retry policy logic and maximum attempt limits
- **Environment variables not available**: Verify environment merging and container injection
- **Logs are delayed or missing**: Check real-time streaming implementation and buffering


## Artifact Management (Milestone 3)

> **Milestone(s):** Milestone 3: Artifact Management

Think of Artifact Management as the **logistics and warehousing system** for our CI/CD factory. Just as a manufacturing plant needs to store, track, and move parts between assembly stations, our pipeline needs to handle build outputs (artifacts) that flow between jobs. When the "compile" job produces a binary, the "test" job needs that exact same binary. When the "package" job creates a Docker image, the "deploy" job must receive the identical image without corruption or confusion.

The challenge is more complex than simple file copying. We need content-addressable storage (like a warehouse where items are stored by their DNA fingerprint), integrity verification (ensuring packages aren't damaged in transit), efficient transfer of large artifacts (streaming rather than loading everything into memory), and automatic cleanup (preventing the warehouse from overflowing with obsolete inventory). This milestone transforms our pipeline from isolated job execution into a connected assembly line where outputs seamlessly become inputs.

![Artifact Flow](./diagrams/artifact-flow.svg)

The artifact management system serves as the **connective tissue** between pipeline jobs, enabling complex workflows where early stages produce assets consumed by later stages. Without robust artifact handling, our pipeline would be like an assembly line where workers throw parts over the wall and hope they land in the right place.

### Storage Backend

Think of the storage backend as a **high-tech warehouse management system** that organizes inventory using molecular fingerprints rather than traditional shelf locations. Each artifact gets stored based on its content hash (SHA-256), ensuring that identical files are automatically deduplicated and corruption is immediately detectable. This content-addressable approach means two jobs that produce identical outputs share the same storage space, and we can verify file integrity by simply recalculating the hash.

The storage backend implements a **layered architecture** that separates logical artifact operations from physical storage details. At the top layer, jobs interact with artifacts using pipeline-centric names like `build-artifacts` or `test-reports`. Below that, the artifact manager translates these logical names into content hashes. At the bottom layer, the storage backend organizes files using a hash-based directory structure that distributes load and enables efficient lookups.

> **Decision: Content-Addressable Storage with Hash-Based Organization**
> - **Context**: Need to store and retrieve artifacts between pipeline jobs with integrity guarantees and efficient deduplication
> - **Options Considered**: 
>   1. Simple file paths with job/build naming
>   2. Database-backed metadata with blob storage
>   3. Content-addressable storage using SHA-256 hashes
> - **Decision**: Content-addressable storage using SHA-256 hashes for file organization
> - **Rationale**: Hash-based storage provides automatic deduplication, built-in integrity verification, and immutable references that prevent accidental overwrites. The hash serves as both storage key and integrity checksum.
> - **Consequences**: Enables efficient storage utilization and robust integrity checking, but requires mapping layer between logical artifact names and content hashes

| Storage Approach | Pros | Cons | Chosen? |
|-----------------|------|------|---------|
| Path-based naming | Simple implementation, human-readable paths | No deduplication, vulnerable to corruption, naming conflicts | No |
| Database metadata + blobs | Flexible queries, rich metadata | Complex architecture, potential consistency issues | No |
| Content-addressable (SHA-256) | Automatic deduplication, integrity verification, immutable | Requires mapping layer, less human-readable | **Yes** |

The **directory organization** follows a fan-out structure that prevents any single directory from becoming overwhelmed with files. Using the first few characters of the SHA-256 hash, we create a nested directory structure: `/artifacts/ab/cd/abcd1234...`. This approach, borrowed from Git's object storage, ensures good performance even with millions of artifacts by distributing files across many directories.

**Artifact Storage Data Structures:**

| Field Name | Type | Description |
|------------|------|-------------|
| `content_hash` | `str` | SHA-256 hash of artifact content, used as storage key |
| `original_name` | `str` | Original filename provided by uploading job |
| `size_bytes` | `int` | File size in bytes for storage planning and validation |
| `mime_type` | `str` | MIME type detected from content for proper handling |
| `created_at` | `datetime` | Timestamp when artifact was first uploaded |
| `last_accessed` | `datetime` | Most recent access time for retention policy decisions |
| `reference_count` | `int` | Number of jobs currently referencing this artifact |
| `compression` | `str` | Compression algorithm used (gzip, none) for storage efficiency |
| `metadata` | `Dict[str, str]` | Custom key-value pairs for job-specific artifact information |

**Storage Backend Interface:**

| Method Name | Parameters | Returns | Description |
|-------------|------------|---------|-------------|
| `store_artifact` | `content: bytes, metadata: Dict` | `str` (content hash) | Store artifact content and return content-addressable hash |
| `retrieve_artifact` | `content_hash: str` | `bytes` | Retrieve artifact content by hash, raise if not found |
| `artifact_exists` | `content_hash: str` | `bool` | Check if artifact exists without retrieving content |
| `get_artifact_info` | `content_hash: str` | `ArtifactInfo` | Get metadata about stored artifact |
| `list_artifacts` | `created_after: datetime, limit: int` | `List[str]` | List artifact hashes for retention policy processing |
| `delete_artifact` | `content_hash: str` | `bool` | Remove artifact from storage, returns success status |
| `get_storage_stats` | `None` | `Dict[str, int]` | Return storage usage statistics for monitoring |

The storage backend implements **streaming operations** to handle large artifacts efficiently. Rather than loading entire files into memory, it processes artifacts in configurable chunks (default 64KB). This approach enables the pipeline to handle multi-gigabyte Docker images or large binary assets without exhausting system memory.

> The critical insight for artifact storage is treating content hashes as **immutable identifiers**. Once an artifact is stored with a particular hash, that mapping never changes. This immutability enables aggressive caching, parallel access, and simplified reasoning about artifact lifecycle.

**File System Layout Example:**

```
artifacts/
├── ab/
│   └── cd/
│       └── abcd1234ef567890...  (full hash as filename)
├── fe/
│   └── dc/
│       └── fedc9876ba543210...
└── metadata/
    ├── abcd1234ef567890.json   (artifact metadata)
    └── fedc9876ba543210.json
```

The storage backend maintains **separate metadata files** alongside content files. While the content hash ensures integrity of the artifact data, the metadata file contains additional information like original filename, creation timestamp, and custom job-provided metadata. This separation allows efficient metadata queries without accessing large artifact content.

**Artifact Upload Process:**

1. **Content Hashing**: Calculate SHA-256 hash while streaming content to temporary file
2. **Deduplication Check**: Query storage backend to see if hash already exists
3. **Atomic Storage**: Move temporary file to final hash-based location using atomic filesystem operation
4. **Metadata Creation**: Write artifact metadata to separate JSON file with same hash-based naming
5. **Reference Tracking**: Update artifact reference count for garbage collection purposes
6. **Cleanup**: Remove temporary files and update storage statistics

**Artifact Download Process:**

1. **Hash Resolution**: Map logical artifact name to content hash using job artifact registry
2. **Existence Verification**: Confirm artifact exists in storage before starting transfer
3. **Streaming Transfer**: Stream content from storage to requesting job in fixed-size chunks
4. **Integrity Verification**: Recalculate hash during transfer and verify against expected value
5. **Access Tracking**: Update last_accessed timestamp for retention policy decisions
6. **Error Handling**: Retry with exponential backoff if transfer fails or integrity check fails

### Integrity Verification

Think of integrity verification as a **chain of custody system** for digital evidence. Just as legal evidence requires documentation proving it hasn't been tampered with, artifacts moving through our pipeline need cryptographic proof they haven't been corrupted or modified. We implement this through checksum validation at every transfer point, creating an unbroken chain of verification from upload to consumption.

The integrity verification system operates on the principle of **defense in depth**, implementing multiple layers of protection against corruption. At the storage layer, we use SHA-256 hashes as both content identifiers and integrity checksums. During transfer, we implement streaming hash calculation to detect corruption without requiring full file buffering. At the consumption layer, jobs verify artifact integrity before beginning execution.

> **Decision: SHA-256 for Content Addressing and Integrity**
> - **Context**: Need robust integrity verification that scales to large artifacts and provides collision resistance
> - **Options Considered**:
>   1. CRC32 checksums for basic error detection
>   2. MD5 hashes for moderate integrity checking
>   3. SHA-256 hashes for cryptographic integrity
> - **Decision**: SHA-256 for all content addressing and integrity verification
> - **Rationale**: SHA-256 provides cryptographic collision resistance, is widely supported, and serves dual purpose as content identifier and integrity check. Performance impact is acceptable for CI/CD workloads.
> - **Consequences**: Strong integrity guarantees and future-proof security, but higher computational cost than simpler checksums

| Integrity Method | Collision Resistance | Performance | Security Level | Chosen? |
|------------------|---------------------|-------------|----------------|---------|
| CRC32 | Low (32-bit) | Excellent | Error detection only | No |
| MD5 | Moderate (but broken) | Good | Deprecated due to vulnerabilities | No |
| SHA-256 | Cryptographically strong | Good | Industry standard | **Yes** |

**Integrity Verification Points:**

The integrity verification system implements checkpoints at every stage where artifacts move or transform:

1. **Upload Verification**: Hash calculated during initial upload, stored as content identifier
2. **Storage Verification**: Periodic integrity scans verify stored artifacts match their hashes
3. **Transfer Verification**: Hash recalculated during download to detect transmission errors
4. **Consumption Verification**: Jobs verify artifact integrity before processing begins
5. **Retention Verification**: Artifacts verified before deletion to prevent corruption from masking as valid cleanup

**Checksum Validation Data Structures:**

| Field Name | Type | Description |
|------------|------|-------------|
| `expected_hash` | `str` | SHA-256 hash expected for this artifact |
| `calculated_hash` | `str` | Hash calculated during verification operation |
| `verification_time` | `datetime` | When integrity check was performed |
| `verification_result` | `VerificationResult` | VALID, CORRUPTED, or MISSING status |
| `error_details` | `str` | Description of integrity failure if verification failed |
| `retry_count` | `int` | Number of verification attempts for this artifact |
| `file_size` | `int` | Expected file size for additional validation |
| `chunk_size` | `int` | Size of chunks used for streaming verification |

**Verification Result States:**

| Current State | Event | Next State | Action Taken |
|---------------|-------|------------|-------------|
| PENDING | Start verification | VERIFYING | Begin streaming hash calculation |
| VERIFYING | Hash matches | VALID | Mark artifact as verified, update access time |
| VERIFYING | Hash mismatch | CORRUPTED | Log error, attempt redownload if available |
| VERIFYING | File missing | MISSING | Log error, fail artifact resolution |
| CORRUPTED | Retry requested | VERIFYING | Attempt fresh download and reverification |
| CORRUPTED | Max retries exceeded | FAILED | Mark artifact as permanently unavailable |

**Streaming Integrity Verification Algorithm:**

1. **Initialize Hash Context**: Create SHA-256 hasher and prepare for streaming input
2. **Open Artifact Stream**: Begin reading artifact content in configurable chunks (64KB default)
3. **Process Chunks**: For each chunk, update hash context and forward content to consumer
4. **Finalize Hash**: Complete hash calculation and convert to hexadecimal string
5. **Compare Hashes**: Verify calculated hash matches expected content hash
6. **Handle Mismatch**: If hashes differ, log detailed error and trigger retry or failure
7. **Update Metadata**: Record verification result and timestamp for monitoring

The streaming approach enables **zero-copy integrity verification** for large artifacts. Rather than reading the entire file into memory, calculating a hash, then reading it again for consumption, we calculate the hash while streaming content to the consuming job. This approach halves I/O operations and enables verification of artifacts larger than available memory.

> The key insight for integrity verification is that **corruption can occur anywhere** in the storage and transfer pipeline. Network transmission can flip bits, disk hardware can fail silently, and software bugs can truncate files. Only end-to-end verification provides confidence that jobs receive exactly the artifacts their dependencies produced.

**Corruption Detection and Recovery:**

When integrity verification detects corruption, the system implements a **graduated response strategy**:

1. **Immediate Retry**: First failure triggers immediate redownload attempt, as temporary network errors are common
2. **Exponential Backoff**: Subsequent failures use exponential backoff to avoid overwhelming failing storage systems
3. **Alternative Sources**: If multiple copies exist (from different pipeline runs), attempt download from alternative source
4. **Graceful Degradation**: After max retries, fail the job gracefully with detailed error message rather than hanging indefinitely
5. **Monitoring Integration**: Log corruption events for ops team investigation of underlying storage issues

**Common Integrity Pitfalls:**

⚠️ **Pitfall: Time-of-Check vs Time-of-Use**
Verifying an artifact's integrity and then using it later creates a window where corruption could occur. Instead, verify integrity during streaming transfer to the consumer, ensuring no gap between verification and use.

⚠️ **Pitfall: Partial Read Corruption**
Network timeouts can cause partial reads that appear successful but contain truncated data. Always verify the expected file size matches actual bytes received before beginning integrity verification.

⚠️ **Pitfall: Hash Collision Assumptions**
While SHA-256 collisions are computationally infeasible, software bugs can cause hash calculation errors. Include file size verification as an additional integrity check that catches common hash calculation mistakes.

### Retention Policy

Think of retention policy as the **inventory management system** for our artifact warehouse. Just as physical warehouses need systematic cleanup to prevent overflow, our artifact storage requires automated policies that remove old artifacts while preserving everything currently needed. The challenge is determining "currently needed" in a system where artifacts might be referenced by running pipelines, cached for future builds, or required for debugging recent failures.

The retention policy system implements **lifecycle-based management** that considers multiple factors when deciding artifact fate. Age is the primary factor (artifacts older than 30 days are candidates for removal), but we also consider reference counting (artifacts still referenced by active jobs are protected), access patterns (recently downloaded artifacts get grace periods), and storage pressure (aggressive cleanup during storage exhaustion).

> **Decision: Multi-Factor Retention with Configurable Policies**
> - **Context**: Need automated artifact cleanup that balances storage efficiency with operational needs for debugging and caching
> - **Options Considered**:
>   1. Simple age-based deletion after fixed time period
>   2. LRU cache with size-based eviction
>   3. Multi-factor policy considering age, references, and access patterns
> - **Decision**: Multi-factor retention policy with configurable rules for different artifact types
> - **Rationale**: CI/CD artifacts have diverse lifecycle needs - some are immediately disposable, others need long retention for compliance. Multi-factor approach provides flexibility while preventing both storage overflow and premature cleanup.
> - **Consequences**: More complex retention logic and configuration, but better matches real-world CI/CD requirements

| Retention Approach | Simplicity | Flexibility | Safety | Chosen? |
|-------------------|------------|-------------|---------|---------|
| Fixed age-based | High | Low | Medium (might delete needed artifacts) | No |
| LRU cache | Medium | Medium | Medium (size-only consideration) | No |
| Multi-factor policies | Low | High | High (considers multiple usage signals) | **Yes** |

**Retention Policy Data Structures:**

| Field Name | Type | Description |
|------------|------|-------------|
| `policy_name` | `str` | Human-readable identifier for this retention policy |
| `max_age_days` | `int` | Maximum age in days before artifact becomes eligible for deletion |
| `grace_period_hours` | `int` | Protection period after last access before deletion eligibility |
| `size_threshold_gb` | `float` | Storage size threshold that triggers more aggressive cleanup |
| `min_reference_count` | `int` | Minimum references required to protect artifact from age-based deletion |
| `artifact_pattern` | `str` | Regex pattern matching artifacts subject to this policy |
| `priority` | `int` | Policy priority for overlapping patterns (higher numbers win) |
| `enabled` | `bool` | Whether this policy is currently active |

**Retention Decision Matrix:**

| Artifact Age | Reference Count | Last Access | Storage Pressure | Decision |
|--------------|----------------|-------------|-----------------|----------|
| < 7 days | Any | Any | Any | Keep (recent artifacts protected) |
| 7-30 days | > 0 | < 24h ago | Low | Keep (actively referenced) |
| 7-30 days | 0 | > 72h ago | Low | Delete (unused and aging) |
| > 30 days | > 0 | < 7 days ago | Low | Keep (old but recently accessed) |
| > 30 days | 0 | > 7 days ago | Any | Delete (old and unused) |
| Any | Any | Any | High | Aggressive cleanup (storage pressure) |

**Retention Policy Evaluation Algorithm:**

1. **Enumerate Candidates**: Query artifact storage for all artifacts older than minimum retention age
2. **Load Reference Counts**: Check artifact registry for current job references to each candidate
3. **Apply Policy Rules**: Evaluate each retention policy against candidate artifacts in priority order
4. **Calculate Deletion Set**: Build list of artifacts marked for deletion by active policies
5. **Safety Verification**: Double-check that deletion candidates have zero active references
6. **Execute Cleanup**: Delete artifacts and their metadata files, updating storage statistics
7. **Audit Logging**: Record retention decisions for compliance and debugging purposes

The retention system implements **gradual cleanup** rather than batch processing to avoid overwhelming the storage system. During each retention cycle (typically every 4 hours), the system processes a limited number of artifacts (default 1000) and spreads deletions across multiple storage locations to minimize I/O impact.

**Reference Counting for Safety:**

The retention policy system maintains **active reference counts** to prevent deletion of artifacts still needed by running or queued jobs. When a job declares artifact dependencies, the system increments reference counts. When jobs complete or are cancelled, reference counts are decremented. Only artifacts with zero references become eligible for age-based cleanup.

| Reference Event | Reference Count Change | Artifact Protection |
|-----------------|----------------------|-------------------|
| Job queued with dependency | +1 | Protected from deletion |
| Job starts execution | No change | Continues protection |
| Job completes successfully | -1 | May become eligible for cleanup |
| Job fails or cancelled | -1 | May become eligible for cleanup |
| Manual artifact pin | +1 (permanent) | Protected until manual unpin |

**Storage Pressure Response:**

When storage utilization exceeds configured thresholds, the retention system activates **emergency cleanup modes** with more aggressive policies:

1. **Yellow Alert (80% full)**: Reduce retention periods by 25%, increase cleanup frequency
2. **Orange Alert (90% full)**: Reduce retention periods by 50%, enable size-based eviction
3. **Red Alert (95% full)**: Emergency cleanup of all unreferenced artifacts regardless of age
4. **Critical Alert (98% full)**: Stop accepting new artifacts until space is reclaimed

**Retention Policy Configuration:**

The retention system supports **policy hierarchies** where multiple policies can apply to the same artifact. Policies are evaluated in priority order, and the first matching policy determines the artifact's fate. This enables sophisticated rules like "keep test artifacts for 7 days, but keep release artifacts for 90 days."

```yaml
retention_policies:
  - name: "release_artifacts"
    pattern: "release-v.*"
    max_age_days: 90
    grace_period_hours: 168  # 7 days
    priority: 100
    
  - name: "test_artifacts" 
    pattern: "test-.*"
    max_age_days: 7
    grace_period_hours: 24
    priority: 50
    
  - name: "default_policy"
    pattern: ".*"
    max_age_days: 30
    grace_period_hours: 72
    priority: 1
```

**Retention Monitoring and Compliance:**

The retention system provides **detailed audit trails** for compliance and debugging purposes:

| Audit Event | Recorded Information | Purpose |
|-------------|---------------------|---------|
| Policy evaluation | Artifact hash, policy matched, decision, timestamp | Debugging retention decisions |
| Artifact deletion | Artifact hash, size, age, reference count, policy | Compliance reporting |
| Reference changes | Job ID, artifact hash, reference delta, timestamp | Tracking artifact lifecycle |
| Storage alerts | Utilization level, cleanup actions taken | Operational monitoring |

> The retention policy system embodies the principle of **progressive cleanup** - artifacts flow through increasingly aggressive deletion criteria as they age and lose relevance. This mirrors how humans naturally organize information, keeping recent items easily accessible while gradually archiving or discarding older materials.

**Common Retention Pitfalls:**

⚠️ **Pitfall: Race Conditions with Job Startup**
A job might start just as its dependency artifacts are being deleted. Implement reference counting with proper locking to ensure jobs can "claim" artifacts before retention policies can delete them.

⚠️ **Pitfall: Thundering Herd Storage Pressure**
When storage fills up, multiple retention policy runs might trigger simultaneously, overwhelming the storage system with deletion requests. Use distributed locking to ensure only one retention process runs at a time.

⚠️ **Pitfall: Compliance Violations**
Some artifacts (release builds, security scans) may need longer retention for regulatory compliance. Make sure policy configuration supports compliance requirements and provides audit trails for retained artifacts.

### Implementation Guidance

**A. Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|----------------|
| Storage Backend | Local filesystem with hash directories | Object storage (S3/GCS) with metadata database |
| Integrity Verification | Python hashlib with file streaming | Hardware-accelerated SHA-256 with parallel verification |
| Retention Scheduling | Simple cron job with file scanning | Event-driven cleanup with storage monitoring |
| Metadata Storage | JSON files alongside artifacts | SQLite database for queryable metadata |

**B. Recommended File Structure:**

```
project-root/
  src/
    artifact_manager/
      __init__.py
      storage/
        __init__.py
        backend.py          ← Storage backend implementation
        hash_utils.py       ← SHA-256 utilities and streaming helpers
      integrity/
        __init__.py  
        verifier.py         ← Integrity verification logic
        corruption_handler.py ← Recovery from corruption detection
      retention/
        __init__.py
        policy_engine.py    ← Retention policy evaluation
        cleanup_scheduler.py ← Automated cleanup orchestration
      models/
        __init__.py
        artifact.py         ← Artifact data structures
        policy.py          ← Retention policy models
      exceptions.py         ← Artifact-specific exceptions
    tests/
      test_artifact_manager/
        test_storage_backend.py
        test_integrity_verification.py
        test_retention_policy.py
        fixtures/
          sample_artifacts/   ← Test artifacts for verification
```

**C. Infrastructure Starter Code:**

```python
# src/artifact_manager/storage/hash_utils.py
import hashlib
import os
from typing import Iterator, Tuple

def generate_artifact_id(content: bytes) -> str:
    """Generate SHA-256 hash for content-addressable storage."""
    return hashlib.sha256(content).hexdigest()

def streaming_hash_calculator(file_path: str, chunk_size: int = 65536) -> Iterator[Tuple[bytes, str]]:
    """
    Stream file content while calculating hash.
    Yields (chunk, current_hash) tuples for each chunk.
    Final hash is available in last tuple.
    """
    hasher = hashlib.sha256()
    
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
            yield chunk, hasher.hexdigest()

def create_hash_directory_path(content_hash: str, base_dir: str) -> str:
    """Create nested directory path from hash: base_dir/ab/cd/abcd1234..."""
    return os.path.join(base_dir, content_hash[:2], content_hash[2:4], content_hash)

# src/artifact_manager/models/artifact.py
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional
from enum import Enum

class VerificationResult(Enum):
    PENDING = "pending"
    VERIFYING = "verifying"  
    VALID = "valid"
    CORRUPTED = "corrupted"
    MISSING = "missing"
    FAILED = "failed"

@dataclass
class ArtifactInfo:
    content_hash: str
    original_name: str
    size_bytes: int
    mime_type: str
    created_at: datetime
    last_accessed: datetime
    reference_count: int
    compression: str
    metadata: Dict[str, str]
    
    def to_dict(self) -> Dict:
        """Serialize artifact info for JSON storage."""
        return {
            'content_hash': self.content_hash,
            'original_name': self.original_name,
            'size_bytes': self.size_bytes,
            'mime_type': self.mime_type,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'reference_count': self.reference_count,
            'compression': self.compression,
            'metadata': self.metadata
        }

@dataclass
class RetentionPolicy:
    policy_name: str
    max_age_days: int
    grace_period_hours: int
    size_threshold_gb: float
    min_reference_count: int
    artifact_pattern: str
    priority: int
    enabled: bool

# src/artifact_manager/exceptions.py
class ArtifactManagerError(Exception):
    """Base exception for artifact management operations."""
    pass

class ArtifactNotFoundError(ArtifactManagerError):
    """Raised when requested artifact does not exist."""
    pass

class IntegrityVerificationError(ArtifactManagerError):
    """Raised when artifact fails integrity verification."""
    pass

class StorageQuotaExceededError(ArtifactManagerError):
    """Raised when storage backend cannot accept new artifacts."""
    pass

class CorruptionDetectedError(ArtifactManagerError):
    """Raised when artifact corruption is detected during verification."""
    pass
```

**D. Core Logic Skeleton Code:**

```python
# src/artifact_manager/storage/backend.py
import os
import json
import shutil
from typing import Dict, List, Optional
from ..models.artifact import ArtifactInfo, VerificationResult
from ..exceptions import *
from .hash_utils import generate_artifact_id, create_hash_directory_path

class ArtifactStorageBackend:
    """Content-addressable storage backend for CI/CD artifacts."""
    
    def __init__(self, base_directory: str, max_storage_gb: float = 100.0):
        self.base_directory = base_directory
        self.artifacts_dir = os.path.join(base_directory, "artifacts")
        self.metadata_dir = os.path.join(base_directory, "metadata") 
        self.max_storage_bytes = max_storage_gb * 1024 * 1024 * 1024
        
        # Create directory structure
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
    
    def store_artifact(self, content: bytes, metadata: Dict) -> str:
        """
        Store artifact content and return content-addressable hash.
        Implements atomic storage with deduplication.
        """
        # TODO 1: Calculate SHA-256 hash of content using generate_artifact_id()
        # TODO 2: Check if artifact already exists using artifact_exists()
        # TODO 3: If exists, increment reference count and return existing hash
        # TODO 4: Check storage quota using _check_storage_quota()
        # TODO 5: Create hash-based directory path using create_hash_directory_path()
        # TODO 6: Write content to temporary file first, then atomic move to final location
        # TODO 7: Create ArtifactInfo object with provided metadata
        # TODO 8: Write metadata to JSON file in metadata directory
        # TODO 9: Return content hash
        # Hint: Use tempfile.NamedTemporaryFile for atomic writes
        pass
    
    def retrieve_artifact(self, content_hash: str) -> bytes:
        """
        Retrieve artifact content by hash.
        Updates last_accessed timestamp during retrieval.
        """
        # TODO 1: Validate content_hash format (64 char hexadecimal)
        # TODO 2: Build file path using create_hash_directory_path()
        # TODO 3: Check if file exists, raise ArtifactNotFoundError if missing
        # TODO 4: Read and return file content
        # TODO 5: Update last_accessed timestamp in metadata
        # Hint: Use os.path.exists() for existence check
        pass
    
    def artifact_exists(self, content_hash: str) -> bool:
        """Check if artifact exists without retrieving content."""
        # TODO 1: Build file path using create_hash_directory_path()
        # TODO 2: Return os.path.exists() result for the file
        pass
    
    def get_artifact_info(self, content_hash: str) -> ArtifactInfo:
        """Get metadata about stored artifact."""
        # TODO 1: Build metadata file path in metadata directory
        # TODO 2: Check if metadata file exists
        # TODO 3: Load and parse JSON metadata
        # TODO 4: Create ArtifactInfo object from loaded data
        # TODO 5: Handle datetime parsing for created_at and last_accessed
        # Hint: Use datetime.fromisoformat() for parsing timestamps
        pass
    
    def list_artifacts(self, created_after: datetime = None, limit: int = 1000) -> List[str]:
        """List artifact hashes for retention policy processing."""
        # TODO 1: Walk through metadata directory to find all .json files
        # TODO 2: For each metadata file, parse creation timestamp
        # TODO 3: Filter by created_after if provided
        # TODO 4: Sort by creation time and apply limit
        # TODO 5: Return list of content hashes
        # Hint: Use os.walk() to traverse metadata directory
        pass
    
    def delete_artifact(self, content_hash: str) -> bool:
        """Remove artifact from storage, returns success status."""
        # TODO 1: Check reference count in metadata, refuse deletion if > 0
        # TODO 2: Build paths for both content file and metadata file
        # TODO 3: Remove content file from artifacts directory
        # TODO 4: Remove metadata file from metadata directory  
        # TODO 5: Remove empty parent directories to keep structure clean
        # TODO 6: Return True if successful, False if artifact didn't exist
        # Hint: Use os.rmdir() to remove empty directories
        pass
    
    def get_storage_stats(self) -> Dict[str, int]:
        """Return storage usage statistics for monitoring."""
        # TODO 1: Walk artifacts directory and sum file sizes
        # TODO 2: Count total number of artifacts
        # TODO 3: Calculate percentage of quota used
        # TODO 4: Return dictionary with total_bytes, artifact_count, quota_used_percent
        # Hint: Use os.path.getsize() for file sizes
        pass

# src/artifact_manager/integrity/verifier.py
class IntegrityVerifier:
    """Handles artifact integrity verification with streaming support."""
    
    def verify_artifact_integrity(self, file_path: str, expected_hash: str, 
                                 chunk_size: int = 65536) -> VerificationResult:
        """
        Verify artifact integrity using streaming hash calculation.
        Returns verification result with detailed error information.
        """
        # TODO 1: Check if file exists, return MISSING if not found
        # TODO 2: Initialize SHA-256 hasher for streaming calculation
        # TODO 3: Open file and read in chunks, updating hasher
        # TODO 4: Finalize hash calculation and convert to hexadecimal
        # TODO 5: Compare calculated hash with expected_hash
        # TODO 6: Return VALID if matches, CORRUPTED if different
        # TODO 7: Handle file I/O errors and return appropriate status
        # Hint: Use hashlib.sha256() for streaming hash calculation
        pass
    
    def verify_during_transfer(self, source_path: str, destination_path: str, 
                             expected_hash: str) -> bool:
        """
        Verify integrity while copying file from source to destination.
        Enables zero-copy verification during artifact transfer.
        """
        # TODO 1: Open source file for reading and destination for writing
        # TODO 2: Initialize SHA-256 hasher for streaming calculation
        # TODO 3: Read chunks from source, update hasher, write to destination
        # TODO 4: After transfer complete, finalize hash and compare
        # TODO 5: If hash mismatch, remove corrupted destination file
        # TODO 6: Return True if transfer successful and hash valid
        # Hint: Use shutil.copyfileobj for efficient copying with verification
        pass

# src/artifact_manager/retention/policy_engine.py  
class RetentionPolicyEngine:
    """Evaluates retention policies and manages artifact cleanup."""
    
    def __init__(self, storage_backend: ArtifactStorageBackend):
        self.storage = storage_backend
        self.policies: List[RetentionPolicy] = []
    
    def evaluate_retention_policies(self, max_artifacts: int = 1000) -> List[str]:
        """
        Evaluate retention policies and return list of artifacts to delete.
        Processes limited number of artifacts to avoid overwhelming storage.
        """
        # TODO 1: Get list of artifacts from storage backend with age limit
        # TODO 2: Load current reference counts for all candidate artifacts  
        # TODO 3: For each artifact, evaluate against all enabled policies in priority order
        # TODO 4: Build list of artifacts marked for deletion by policies
        # TODO 5: Double-check that deletion candidates have zero active references
        # TODO 6: Return list of content hashes safe for deletion
        # Hint: Sort policies by priority (descending) for correct evaluation order
        pass
    
    def execute_cleanup(self, deletion_candidates: List[str]) -> Dict[str, int]:
        """
        Execute artifact deletion and return cleanup statistics.
        Implements gradual cleanup to avoid overwhelming storage system.
        """
        # TODO 1: Initialize counters for successful/failed deletions
        # TODO 2: For each candidate, attempt deletion via storage backend
        # TODO 3: Log deletion results for audit trail
        # TODO 4: Update storage statistics after cleanup
        # TODO 5: Return dictionary with cleanup results and metrics
        # Hint: Add delays between deletions to avoid I/O pressure
        pass
```

**E. Language-Specific Hints:**

- **File I/O**: Use `pathlib.Path` for cross-platform path handling and `tempfile` for atomic file operations
- **Streaming**: Use `shutil.copyfileobj()` with custom buffer size for efficient large file transfers
- **Hashing**: `hashlib.sha256()` supports streaming updates - call `update()` multiple times, then `hexdigest()`
- **JSON Metadata**: Use `json.dump()`/`json.load()` for metadata persistence, handle datetime serialization manually
- **Directory Operations**: Use `os.makedirs(exist_ok=True)` for safe directory creation
- **Atomic Operations**: Write to temporary file first, then `os.rename()` for atomic replacement on POSIX systems

**F. Milestone Checkpoint:**

After implementing artifact management functionality:

1. **Test Basic Storage**: 
   ```bash
   python -m pytest tests/test_artifact_manager/test_storage_backend.py -v
   ```
   Expected: All storage operations (store, retrieve, delete) pass with proper hash calculation

2. **Verify Integrity Checking**:
   ```bash
   python -c "
   from src.artifact_manager.integrity.verifier import IntegrityVerifier
   verifier = IntegrityVerifier()
   # Should return VALID for correct file
   print(verifier.verify_artifact_integrity('test_file.txt', 'expected_hash'))
   "
   ```
   Expected: Integrity verification correctly identifies valid vs corrupted files

3. **Test Retention Policy**:
   ```bash
   python -c "
   from src.artifact_manager.retention.policy_engine import RetentionPolicyEngine
   engine = RetentionPolicyEngine(storage_backend)
   candidates = engine.evaluate_retention_policies()
   print(f'Found {len(candidates)} artifacts eligible for cleanup')
   "
   ```
   Expected: Retention evaluation completes without errors and returns reasonable candidate count

4. **Manual Integration Test**:
   - Store several test artifacts with different ages and reference counts
   - Verify artifacts can be retrieved with correct integrity
   - Run retention policy and confirm only appropriate artifacts are marked for deletion
   - Check that storage statistics reflect actual disk usage

Signs of problems:
- **Hash mismatches**: Check file I/O for proper binary mode and complete reads
- **Storage quota errors**: Verify quota calculations include all artifact and metadata files  
- **Retention too aggressive**: Check reference counting logic and policy evaluation order
- **Performance issues**: Profile chunk sizes for hashing and implement streaming properly


## Deployment Strategies (Milestone 4)

> **Milestone(s):** Milestone 4: Deployment Strategies

Think of deployment strategies as **air traffic control systems** for software releases. Just as airports use different approaches to land aircraft safely—some planes arrive during clear weather and can land directly, others need special patterns during storms, and emergency situations require immediate runway clearing—deployment strategies provide structured approaches for safely transitioning software from development to production environments.

![Deployment Strategy Comparison](./diagrams/deployment-strategies.svg)

The fundamental challenge in deployment is managing the **state transition risk**. Unlike simply copying files, deployments involve coordinating multiple moving parts: application servers, load balancers, databases, caches, and external integrations. Each deployment strategy represents a different risk/speed trade-off, much like how pilots choose different landing approaches based on weather conditions, runway availability, and aircraft capabilities.

### Rolling Deployment: Incremental Updates with Health Check Validation

Rolling deployment operates like a **relay race handoff**—each runner (instance) passes the baton (traffic) to the next only after confirming they're ready to run. The system updates one instance at a time, validates it's healthy, then moves to the next instance. This approach minimizes service disruption by maintaining capacity throughout the deployment process.

The core principle behind rolling deployment is **incremental risk management**. Instead of updating all instances simultaneously (which could cause total service failure), rolling deployment limits the blast radius to a single instance. If that instance fails health checks, the deployment stops automatically, leaving the majority of instances running the previous version.

Rolling deployment maintains service availability through **capacity preservation**. If you have five instances handling traffic, the deployment process updates one instance while the other four continue serving requests. The load balancer automatically routes traffic away from the updating instance, then gradually includes it back once health checks pass. This creates a seamless user experience where individual requests might hit different versions temporarily, but the service remains responsive.

> **Decision: Rolling Deployment Instance Selection Strategy**
> - **Context**: Rolling deployments need to decide which instances to update in what order, considering load distribution and failure isolation
> - **Options Considered**: Random selection, round-robin order, load-aware selection
> - **Decision**: Load-aware selection with configurable batch sizing
> - **Rationale**: Random selection can create uneven load distribution during deployment. Round-robin is predictable but doesn't account for instance capacity differences. Load-aware selection ensures capacity is maintained optimally throughout the deployment
> - **Consequences**: Requires monitoring integration to assess instance load, but provides better performance and reliability during deployments

The rolling deployment process follows a **state machine pattern** where each instance transitions through defined states with validation gates:

| Current State | Event | Next State | Actions Taken |
|---------------|-------|------------|---------------|
| RUNNING | deployment_start | DRAINING | Remove from load balancer, wait for active connections to complete |
| DRAINING | connections_drained | UPDATING | Deploy new version, start application |
| UPDATING | application_started | HEALTH_CHECK | Begin health check validation sequence |
| HEALTH_CHECK | health_check_pass | READY | Add back to load balancer with new version |
| HEALTH_CHECK | health_check_fail | ROLLBACK | Restart with previous version, re-run health checks |
| READY | next_instance_selected | RUNNING | Continue deployment to next instance |
| ROLLBACK | rollback_complete | RUNNING | Instance restored to previous version |

Health check validation during rolling deployment requires **multi-dimensional assessment**. A simple "is the process running" check is insufficient—the system must verify functional correctness, performance characteristics, and integration health. This typically involves HTTP endpoint checks, database connectivity validation, and dependency service reachability tests.

> The critical insight for rolling deployment is that **partial deployments are normal operational state**. During the deployment window, different instances run different versions simultaneously. The application architecture must handle version heterogeneity gracefully, especially for inter-service communication and shared state access.

**Rolling Deployment Configuration:**

| Parameter | Type | Description | Default Value |
|-----------|------|-------------|---------------|
| batch_size | int | Number of instances to update simultaneously | 1 |
| health_check_timeout | int | Maximum seconds to wait for health check success | 300 |
| drain_timeout | int | Maximum seconds to wait for connection draining | 120 |
| rollback_on_failure | bool | Automatically rollback failed instances | true |
| max_unavailable | int | Maximum instances allowed to be unavailable | 0 |
| health_check_interval | int | Seconds between health check attempts | 10 |
| success_threshold | int | Consecutive successful health checks required | 3 |
| failure_threshold | int | Consecutive failed health checks before rollback | 2 |

### Blue-Green Deployment: Atomic Traffic Switching Between Environment Versions

Blue-green deployment functions like a **theater with two identical stages**. While one stage (blue) performs for the live audience, the other stage (green) prepares the next show with full dress rehearsals. When the new show is ready and validated, the theater instantly redirects the audience to the green stage, making the transition appear seamless and instantaneous.

The fundamental advantage of blue-green deployment is **atomic switchover**. Unlike rolling deployment's gradual transition, blue-green maintains two complete, identical environments. The switch happens at the load balancer level in a single operation, meaning users experience either the old version or the new version—never a mixed state. This eliminates version compatibility concerns during deployment windows.

Blue-green deployment provides **instant rollback capability**. If issues are discovered after the switch, reverting involves simply flipping the load balancer back to the previous environment. This rollback operation takes seconds rather than minutes or hours required for redeployment, making blue-green ideal for applications with strict availability requirements.

However, blue-green deployment demands **double infrastructure capacity**. Running two complete environments simultaneously requires twice the compute, storage, and network resources. This cost factor makes blue-green most suitable for critical applications where the operational benefits justify the infrastructure investment.

> **Decision: Blue-Green Environment Synchronization Strategy**
> - **Context**: Blue-green deployment requires keeping two environments synchronized for stateful components like databases while allowing version differences for application tiers
> - **Options Considered**: Shared database with application-level compatibility, database replication with delayed synchronization, separate databases with data migration
> - **Decision**: Shared database with backward-compatible schema changes and feature flags
> - **Rationale**: Database replication introduces complexity and potential data inconsistencies. Separate databases require complex data migration during switchover. Shared database with compatibility constraints is simpler and more reliable
> - **Consequences**: Requires database schema to be backward-compatible and may need feature flags to handle behavior differences between versions

The blue-green deployment process orchestrates **environment preparation, validation, and traffic switching**:

1. **Environment Preparation**: The inactive environment (green) receives the new application version while blue continues serving production traffic
2. **Smoke Testing**: Automated tests run against green environment using production-like data to validate basic functionality
3. **Integration Validation**: Green environment connects to shared production services (databases, caches) to verify compatibility
4. **Performance Validation**: Load testing ensures green environment can handle expected traffic patterns
5. **Canary Validation**: Small percentage of real traffic routes to green environment for final validation
6. **Traffic Switch**: Load balancer configuration updates to route all traffic to green environment
7. **Monitoring Period**: Enhanced monitoring watches for issues during initial post-switch period
8. **Environment Cleanup**: Blue environment either preserved for rollback or repurposed for next deployment cycle

**Blue-Green Deployment State Machine:**

| Current State | Event | Next State | Actions Taken |
|---------------|-------|------------|---------------|
| BLUE_ACTIVE | deployment_start | PREPARING_GREEN | Deploy new version to green environment |
| PREPARING_GREEN | deployment_complete | VALIDATING_GREEN | Execute smoke tests and integration checks |
| VALIDATING_GREEN | validation_pass | READY_TO_SWITCH | Prepare load balancer configuration change |
| VALIDATING_GREEN | validation_fail | DEPLOYMENT_FAILED | Clean up green environment, abort deployment |
| READY_TO_SWITCH | switch_approved | SWITCHING | Update load balancer to route traffic to green |
| SWITCHING | switch_complete | GREEN_ACTIVE | Monitor green environment, mark blue as standby |
| GREEN_ACTIVE | rollback_triggered | SWITCHING_BACK | Revert load balancer to route traffic to blue |
| SWITCHING_BACK | rollback_complete | BLUE_ACTIVE | Resume monitoring blue environment |

Blue-green deployment health checks operate at **multiple validation layers**. Unlike rolling deployment's focus on individual instance health, blue-green validation must confirm entire environment readiness before traffic switch:

| Validation Layer | Check Type | Purpose | Failure Action |
|------------------|------------|---------|----------------|
| Infrastructure | Resource availability, network connectivity | Ensure environment can handle load | Abort deployment |
| Application | Service startup, configuration loading | Verify application runs correctly | Redeploy with fixes |
| Integration | Database connections, external APIs | Confirm service dependencies | Check configuration |
| Performance | Response times, throughput capacity | Validate performance requirements | Tune or abort |
| Business Logic | Critical user flows, data processing | Ensure feature correctness | Fix bugs or abort |

### Canary Deployment: Gradual Traffic Shifting with Monitoring and Rollback

Canary deployment works like a **mine safety system**—just as miners used canary birds to detect dangerous gases before exposing the full crew, canary deployment exposes a small subset of users to new software versions to detect problems before full rollout. The system gradually increases the percentage of traffic seeing the new version while continuously monitoring for issues.

The power of canary deployment lies in **progressive validation under real load**. Unlike staging environments with synthetic traffic, canary deployment uses actual user behavior to validate software changes. This real-world validation catches issues that testing environments often miss, such as performance problems under specific usage patterns, edge cases in user workflows, or integration issues with external services.

Canary deployment implements **automatic risk mitigation** through continuous monitoring and threshold-based rollback. The system establishes success criteria (error rates, response times, business metrics) and automatically reverts if metrics deteriorate beyond acceptable thresholds. This automated monitoring acts as a safety net, limiting the impact of problematic releases.

The graduated rollout approach allows **data-driven deployment decisions**. Instead of binary deploy/rollback choices, canary deployment provides granular control over exposure levels. Teams can pause at any percentage, gather additional data, and make informed decisions about proceeding or reverting based on observed system behavior.

> **Decision: Canary Traffic Splitting Strategy**
> - **Context**: Canary deployment needs to route specific traffic percentages to new versions while maintaining session consistency and avoiding bias in user selection
> - **Options Considered**: Random percentage routing, session-sticky routing, geographic-based routing, user-attribute routing
> - **Decision**: Random percentage routing with session stickiness and configurable user attributes
> - **Rationale**: Random routing provides unbiased sampling. Geographic routing can introduce regional bias. User-attribute routing allows targeted testing but requires complex configuration. Session stickiness prevents confusing users with version switching mid-session
> - **Consequences**: Requires session management integration but provides better user experience and more reliable metrics collection

**Canary Deployment Traffic Distribution:**

| Stage | Traffic Percentage | Duration | Success Criteria | Rollback Triggers |
|-------|-------------------|----------|------------------|-------------------|
| Initial Canary | 5% | 15 minutes | Error rate < 0.1%, P99 latency < baseline + 10% | Error rate > 0.5% or latency > baseline + 50% |
| Small Canary | 15% | 30 minutes | Business metrics within 95% confidence interval | Sustained metric degradation > 5 minutes |
| Medium Canary | 35% | 45 minutes | User satisfaction scores stable | Customer complaints increase > 2x baseline |
| Large Canary | 65% | 60 minutes | All metrics within normal ranges | Any critical business metric regression |
| Full Rollout | 100% | Ongoing | Deployment considered successful | Standard monitoring and alerting |

The canary deployment process implements **graduated exposure with continuous monitoring**:

1. **Baseline Establishment**: Collect current system metrics to establish comparison baseline for new version
2. **Initial Canary (5%)**: Route small traffic percentage to new version, monitor basic health metrics
3. **Metric Analysis**: Compare canary metrics against baseline, looking for regressions in key indicators
4. **Progressive Rollout**: Gradually increase traffic percentage through predefined stages if metrics remain healthy
5. **Business Validation**: Monitor business-specific metrics (conversion rates, revenue, user engagement) alongside technical metrics
6. **Automated Rollback**: Trigger automatic rollback if any metric exceeds configured threshold
7. **Manual Gates**: Provide manual approval points for teams to review data before proceeding to next stage
8. **Completion**: Mark deployment successful when 100% traffic serves new version without issues

**Canary Monitoring Metrics:**

| Metric Category | Metric Name | Baseline Period | Threshold Type | Rollback Trigger |
|----------------|-------------|-----------------|----------------|------------------|
| Availability | Error Rate | 7 days | Relative | >150% of baseline |
| Performance | P99 Response Time | 7 days | Absolute | >500ms or >120% baseline |
| Performance | P95 Response Time | 7 days | Relative | >110% of baseline |
| Business | Conversion Rate | 14 days | Statistical | Outside 95% confidence interval |
| Business | Revenue Per User | 14 days | Relative | <95% of baseline |
| Infrastructure | CPU Utilization | 7 days | Absolute | >80% sustained |
| Infrastructure | Memory Usage | 7 days | Relative | >120% of baseline |
| User Experience | Page Load Time | 7 days | Percentile | P95 >3 seconds |

Canary deployment requires **sophisticated traffic routing** that can split requests based on configurable rules while maintaining consistency:

| Routing Strategy | Description | Use Case | Implementation Complexity |
|------------------|-------------|----------|---------------------------|
| Random Percentage | Random traffic distribution based on percentage | General deployments | Low |
| User ID Hash | Consistent routing based on user identifier hash | User experience consistency | Medium |
| Geographic | Route traffic based on user location | Regional rollouts | Medium |
| Device Type | Route based on client device characteristics | Mobile/desktop feature differences | Medium |
| Feature Flag | Route based on user's feature flag configuration | A/B testing integration | High |
| Request Header | Route based on custom request headers | Internal testing and validation | Low |

> The crucial insight for canary deployment is that **monitoring quality determines deployment safety**. A canary deployment is only as good as its ability to detect problems quickly and accurately. Poor monitoring leads to either false positives (unnecessary rollbacks) or false negatives (problems that go undetected until full rollout).

### Common Pitfalls

⚠️ **Pitfall: Database Schema Incompatibility During Blue-Green**
Teams often deploy application changes that require database schema modifications without ensuring backward compatibility. When the blue environment still connects to the same database as the green environment, schema changes can break the blue environment immediately after deployment.

**Why it's wrong**: Blue-green deployment assumes both environments can operate simultaneously against the same shared state. Non-backward-compatible schema changes violate this assumption.

**How to fix**: Implement a three-phase schema change process: (1) Deploy schema additions that don't break existing code, (2) Deploy application changes that use new schema, (3) Clean up unused schema elements in a subsequent deployment.

⚠️ **Pitfall: Insufficient Health Check Coverage in Rolling Deployment**
Many teams implement basic "is the service responding" health checks but miss critical functionality validation. An instance passes health checks but fails to process actual business logic correctly, leading to degraded service quality during deployment.

**Why it's wrong**: Rolling deployment relies entirely on health checks to determine when an instance is ready for traffic. Inadequate health checks allow broken instances to receive user requests.

**How to fix**: Implement layered health checks that validate: basic service startup, database connectivity, external service integration, and critical business functionality. Include synthetic transaction validation that exercises key user workflows.

⚠️ **Pitfall: Canary Metrics Selection Bias**
Teams often select metrics that are easy to measure (response time, error rate) but miss business-critical indicators that better reflect user experience. This leads to deployments that appear technically successful but actually degrade user satisfaction or business outcomes.

**Why it's wrong**: Technical metrics don't always correlate with business impact. A deployment might maintain good response times while introducing subtle bugs that affect user workflows or revenue generation.

**How to fix**: Include business metrics alongside technical metrics: conversion rates, user engagement, revenue per session, feature adoption rates. Establish baseline measurements and statistical significance thresholds for business metrics.

⚠️ **Pitfall: Session Affinity Breaking During Traffic Shifts**
During canary or blue-green deployments, user sessions can break when traffic routing changes mid-session. Users experience logout, lost shopping carts, or application state reset, creating poor user experience even when deployment is technically successful.

**Why it's wrong**: Modern applications rely on session state for user experience continuity. Arbitrary traffic routing changes can distribute user requests across different application versions or instances that don't share session state.

**How to fix**: Implement session-aware traffic routing that ensures user sessions stick to the same application version throughout their duration. Use external session storage (Redis, database) that both versions can access, or implement graceful session migration between versions.

⚠️ **Pitfall: Inadequate Rollback Testing**
Teams thoroughly test forward deployment processes but rarely validate rollback procedures under realistic conditions. When problems occur in production, rollback mechanisms fail or take much longer than expected, extending service disruption.

**Why it's wrong**: Rollback is a critical failure recovery mechanism that must work reliably under stress. Untested rollback procedures often have hidden dependencies or timing issues that only surface during actual incidents.

**How to fix**: Regularly test rollback procedures as part of deployment pipeline validation. Include rollback testing in disaster recovery exercises. Measure and optimize rollback time to meet recovery time objectives (RTO).

### Implementation Guidance

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Load Balancer | HAProxy with config reload | AWS ALB/ELB with API updates |
| Health Checks | HTTP endpoint polling | Consul health checks with multiple validators |
| Metrics Collection | Prometheus with custom metrics | DataDog or New Relic with business metrics |
| Traffic Routing | Nginx with weighted upstream | Istio service mesh with traffic splitting |
| Deployment Orchestration | Shell scripts with curl | Kubernetes deployments with Argo Rollouts |
| State Management | JSON files with file locking | etcd or Consul for distributed coordination |

**Recommended File Structure:**
```
ci_cd_pipeline/
  deployment/
    __init__.py
    strategy.py              ← Abstract base classes and strategy interface
    rolling.py               ← Rolling deployment implementation
    blue_green.py           ← Blue-green deployment implementation  
    canary.py               ← Canary deployment implementation
    health_checks.py        ← Health check validators and runners
    traffic_router.py       ← Load balancer integration and traffic management
    metrics_collector.py    ← Deployment metrics and monitoring
    rollback_manager.py     ← Rollback coordination and automation
  config/
    deployment_strategies.yaml ← Strategy configuration templates
  tests/
    test_deployment_strategies.py
    integration/
      test_deployment_flow.py
```

**Infrastructure Starter Code:**

```python
# deployment/health_checks.py
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class HealthCheckResult(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    TIMEOUT = "timeout"
    ERROR = "error"

@dataclass
class HealthCheckConfig:
    endpoint: str
    timeout: int
    interval: int
    success_threshold: int
    failure_threshold: int
    expected_status: int = 200
    expected_body: Optional[str] = None

class HealthChecker:
    def __init__(self, config: HealthCheckConfig):
        self.config = config
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.logger = logging.getLogger(__name__)
    
    async def check_health(self, instance_url: str) -> Tuple[HealthCheckResult, str]:
        """Execute health check against instance and return result with details."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{instance_url}{self.config.endpoint}",
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    if response.status != self.config.expected_status:
                        return HealthCheckResult.UNHEALTHY, f"Expected status {self.config.expected_status}, got {response.status}"
                    
                    if self.config.expected_body:
                        body = await response.text()
                        if self.config.expected_body not in body:
                            return HealthCheckResult.UNHEALTHY, f"Expected body content '{self.config.expected_body}' not found"
                    
                    return HealthCheckResult.HEALTHY, "Health check passed"
                    
        except asyncio.TimeoutError:
            return HealthCheckResult.TIMEOUT, f"Health check timed out after {self.config.timeout}s"
        except Exception as e:
            return HealthCheckResult.ERROR, f"Health check error: {str(e)}"
    
    async def wait_for_healthy(self, instance_url: str, max_attempts: int = 30) -> bool:
        """Wait for instance to become healthy with consecutive success threshold."""
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        
        for attempt in range(max_attempts):
            result, message = await self.check_health(instance_url)
            
            if result == HealthCheckResult.HEALTHY:
                self.consecutive_successes += 1
                self.consecutive_failures = 0
                
                if self.consecutive_successes >= self.config.success_threshold:
                    self.logger.info(f"Instance {instance_url} healthy after {attempt + 1} attempts")
                    return True
            else:
                self.consecutive_failures += 1
                self.consecutive_successes = 0
                
                if self.consecutive_failures >= self.config.failure_threshold:
                    self.logger.error(f"Instance {instance_url} failed health check: {message}")
                    return False
                    
                self.logger.warning(f"Health check attempt {attempt + 1}: {message}")
                await asyncio.sleep(self.config.interval)
        
        return False

# deployment/traffic_router.py
import json
import subprocess
import tempfile
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class TrafficTarget:
    name: str
    url: str
    weight: int
    health_check_url: str

class HAProxyTrafficRouter:
    def __init__(self, config_path: str, stats_socket: str):
        self.config_path = config_path
        self.stats_socket = stats_socket
        self.logger = logging.getLogger(__name__)
    
    def update_traffic_weights(self, targets: List[TrafficTarget]) -> bool:
        """Update HAProxy configuration with new traffic weights."""
        try:
            # Generate new HAProxy configuration
            config_content = self._generate_haproxy_config(targets)
            
            # Write to temporary file for atomic update
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as tmp_file:
                tmp_file.write(config_content)
                tmp_config_path = tmp_file.name
            
            # Validate configuration
            result = subprocess.run(
                ['haproxy', '-c', '-f', tmp_config_path],
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                self.logger.error(f"HAProxy config validation failed: {result.stderr}")
                return False
            
            # Atomic configuration update with graceful reload
            subprocess.run(['mv', tmp_config_path, self.config_path])
            reload_result = subprocess.run(
                ['systemctl', 'reload', 'haproxy'],
                capture_output=True
            )
            
            return reload_result.returncode == 0
            
        except Exception as e:
            self.logger.error(f"Traffic weight update failed: {e}")
            return False
    
    def _generate_haproxy_config(self, targets: List[TrafficTarget]) -> str:
        """Generate HAProxy configuration for traffic targets."""
        backend_servers = []
        for target in targets:
            backend_servers.append(
                f"    server {target.name} {target.url} weight {target.weight} "
                f"check port 8080 inter 5s"
            )
        
        return f"""
global
    daemon
    stats socket {self.stats_socket} mode 660 level admin

defaults
    mode http
    timeout connect 5000ms
    timeout client 50000ms
    timeout server 50000ms

frontend main
    bind *:80
    default_backend app_servers

backend app_servers
{chr(10).join(backend_servers)}
"""

# deployment/metrics_collector.py
import time
import statistics
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

@dataclass
class MetricSample:
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)

class MetricsCollector:
    def __init__(self, retention_hours: int = 24):
        self.metrics: Dict[str, List[MetricSample]] = {}
        self.retention_hours = retention_hours
        self.logger = logging.getLogger(__name__)
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric sample with timestamp and optional tags."""
        if name not in self.metrics:
            self.metrics[name] = []
        
        sample = MetricSample(
            timestamp=datetime.utcnow(),
            value=value,
            tags=tags or {}
        )
        
        self.metrics[name].append(sample)
        self._cleanup_old_metrics()
    
    def get_metric_stats(self, name: str, duration_minutes: int = 10) -> Optional[Dict]:
        """Calculate statistics for a metric over the specified duration."""
        if name not in self.metrics:
            return None
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=duration_minutes)
        recent_samples = [
            sample for sample in self.metrics[name]
            if sample.timestamp >= cutoff_time
        ]
        
        if not recent_samples:
            return None
        
        values = [sample.value for sample in recent_samples]
        return {
            'count': len(values),
            'avg': statistics.mean(values),
            'median': statistics.median(values),
            'p95': self._percentile(values, 0.95),
            'p99': self._percentile(values, 0.99),
            'min': min(values),
            'max': max(values),
            'latest': values[-1]
        }
    
    def compare_metrics(self, name: str, baseline_minutes: int = 60, 
                       current_minutes: int = 10) -> Optional[Dict]:
        """Compare current metric performance against baseline period."""
        baseline_stats = self.get_metric_stats(name, baseline_minutes)
        current_stats = self.get_metric_stats(name, current_minutes)
        
        if not baseline_stats or not current_stats:
            return None
        
        return {
            'baseline_avg': baseline_stats['avg'],
            'current_avg': current_stats['avg'],
            'avg_change_pct': ((current_stats['avg'] - baseline_stats['avg']) / 
                              baseline_stats['avg'] * 100),
            'p99_change_pct': ((current_stats['p99'] - baseline_stats['p99']) / 
                              baseline_stats['p99'] * 100),
            'regression_detected': current_stats['avg'] > baseline_stats['avg'] * 1.1
        }
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value from sorted list."""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def _cleanup_old_metrics(self):
        """Remove metric samples older than retention period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)
        
        for metric_name in self.metrics:
            self.metrics[metric_name] = [
                sample for sample in self.metrics[metric_name]
                if sample.timestamp >= cutoff_time
            ]
```

**Core Logic Skeleton Code:**

```python
# deployment/strategy.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class DeploymentStatus(Enum):
    PENDING = "pending"
    PREPARING = "preparing"
    DEPLOYING = "deploying"
    VALIDATING = "validating"
    COMPLETE = "complete"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"

@dataclass
class DeploymentTarget:
    name: str
    url: str
    version: str
    status: DeploymentStatus
    health_check_url: str
    metadata: Dict[str, Any]

class DeploymentStrategy(ABC):
    """Abstract base class for deployment strategies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.status = DeploymentStatus.PENDING
        self.targets: List[DeploymentTarget] = []
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker(self._build_health_config())
    
    @abstractmethod
    async def deploy(self, version: str, targets: List[str]) -> bool:
        """Execute deployment strategy for given version and targets."""
        # TODO 1: Validate deployment prerequisites (target availability, version compatibility)
        # TODO 2: Initialize deployment tracking and metrics collection
        # TODO 3: Execute strategy-specific deployment logic
        # TODO 4: Monitor deployment progress and handle failures
        # TODO 5: Complete deployment or trigger rollback on failure
        pass
    
    @abstractmethod
    async def rollback(self) -> bool:
        """Rollback deployment to previous version."""
        # TODO 1: Identify previous stable version for each target
        # TODO 2: Execute strategy-specific rollback procedure
        # TODO 3: Validate rollback success through health checks
        # TODO 4: Update deployment status and clean up resources
        pass
    
    @abstractmethod
    def get_deployment_progress(self) -> Dict[str, Any]:
        """Return current deployment progress and status."""
        # TODO 1: Calculate overall deployment completion percentage
        # TODO 2: Collect per-target status and health information
        # TODO 3: Include relevant metrics and performance data
        # TODO 4: Return structured progress report
        pass

# deployment/rolling.py
class RollingDeployment(DeploymentStrategy):
    """Rolling deployment with incremental instance updates."""
    
    async def deploy(self, version: str, targets: List[str]) -> bool:
        """Execute rolling deployment across target instances."""
        # TODO 1: Validate all targets are healthy before starting deployment
        # TODO 2: Calculate batch size based on configuration and total target count
        # TODO 3: For each batch: drain traffic, deploy new version, run health checks
        # TODO 4: If batch health checks pass, proceed to next batch
        # TODO 5: If batch health checks fail, rollback batch and abort deployment
        # TODO 6: Monitor deployment metrics throughout process
        # TODO 7: Complete deployment when all batches successfully updated
        # Hint: Use self.config['batch_size'] for simultaneous update count
        # Hint: Implement connection draining with configurable timeout
        pass
    
    async def _deploy_batch(self, batch_targets: List[DeploymentTarget], version: str) -> bool:
        """Deploy new version to a batch of targets."""
        # TODO 1: Remove batch targets from load balancer rotation
        # TODO 2: Wait for active connections to drain (respect drain_timeout)
        # TODO 3: Deploy new version to each target in batch
        # TODO 4: Wait for application startup on all batch targets
        # TODO 5: Execute health check validation for entire batch
        # TODO 6: If healthy, add targets back to load balancer
        # TODO 7: If unhealthy, rollback batch and return failure
        pass
    
    async def rollback(self) -> bool:
        """Rollback rolling deployment by reversing update process."""
        # TODO 1: Stop any in-progress batch deployments
        # TODO 2: Identify targets that were successfully updated during deployment
        # TODO 3: Apply rolling rollback process in reverse order
        # TODO 4: For each batch: drain, deploy previous version, validate health
        # TODO 5: Monitor rollback progress and handle failures
        # TODO 6: Mark deployment as rolled back when all targets restored
        pass

# deployment/blue_green.py
class BlueGreenDeployment(DeploymentStrategy):
    """Blue-green deployment with atomic environment switching."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.blue_environment: List[DeploymentTarget] = []
        self.green_environment: List[DeploymentTarget] = []
        self.active_environment = "blue"
    
    async def deploy(self, version: str, targets: List[str]) -> bool:
        """Execute blue-green deployment with environment preparation and switching."""
        # TODO 1: Identify currently active environment (blue/green)
        # TODO 2: Prepare inactive environment with new version
        # TODO 3: Execute comprehensive validation against inactive environment
        # TODO 4: Run smoke tests and integration tests on inactive environment
        # TODO 5: Execute canary validation with small traffic percentage
        # TODO 6: Perform atomic traffic switch to inactive environment
        # TODO 7: Monitor post-switch metrics for immediate issues
        # TODO 8: Mark previous environment as standby for potential rollback
        # Hint: Use traffic router to gradually shift small percentage before full switch
        # Hint: Implement validation timeout to prevent hanging deployments
        pass
    
    async def _prepare_environment(self, environment_targets: List[DeploymentTarget], 
                                 version: str) -> bool:
        """Prepare inactive environment with new application version."""
        # TODO 1: Deploy new version to all targets in inactive environment
        # TODO 2: Wait for all application instances to start successfully
        # TODO 3: Execute smoke tests to validate basic functionality
        # TODO 4: Test database connectivity and external service integration
        # TODO 5: Run performance validation to ensure environment can handle load
        # TODO 6: Return success only if all validation steps pass
        pass
    
    async def _switch_traffic(self, target_environment: str) -> bool:
        """Atomically switch traffic to target environment."""
        # TODO 1: Update load balancer configuration to route to target environment
        # TODO 2: Verify traffic routing change took effect
        # TODO 3: Monitor immediate post-switch metrics for obvious failures
        # TODO 4: Update active environment tracking
        # TODO 5: Return success if switch completed without immediate issues
        pass
    
    async def rollback(self) -> bool:
        """Rollback blue-green deployment by switching back to previous environment."""
        # TODO 1: Identify previous active environment for rollback target
        # TODO 2: Validate previous environment is still healthy and available
        # TODO 3: Execute atomic traffic switch back to previous environment
        # TODO 4: Monitor rollback success through health checks and metrics
        # TODO 5: Mark deployment as rolled back and update environment status
        pass

# deployment/canary.py
class CanaryDeployment(DeploymentStrategy):
    """Canary deployment with graduated traffic shifting and monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.traffic_stages = config.get('traffic_stages', [5, 15, 35, 65, 100])
        self.stage_duration = config.get('stage_duration_minutes', 15)
        self.current_stage = 0
        self.canary_targets: List[DeploymentTarget] = []
        self.baseline_targets: List[DeploymentTarget] = []
    
    async def deploy(self, version: str, targets: List[str]) -> bool:
        """Execute canary deployment with graduated traffic shifting."""
        # TODO 1: Split targets into canary group and baseline group
        # TODO 2: Deploy new version to canary targets only
        # TODO 3: Establish baseline metrics from current production traffic
        # TODO 4: For each traffic stage: update routing, monitor metrics, validate success
        # TODO 5: If stage validation fails, automatically trigger rollback
        # TODO 6: If all stages succeed, complete deployment to 100% traffic
        # TODO 7: Monitor post-deployment metrics for extended period
        # Hint: Use self.traffic_stages to determine percentage progression
        # Hint: Implement automatic rollback triggers based on metric thresholds
        pass
    
    async def _execute_canary_stage(self, traffic_percentage: int) -> bool:
        """Execute single canary stage with specified traffic percentage."""
        # TODO 1: Update traffic routing to send percentage to canary targets
        # TODO 2: Wait for stage duration to collect sufficient metric samples
        # TODO 3: Collect and compare metrics between canary and baseline traffic
        # TODO 4: Evaluate success criteria: error rates, response times, business metrics
        # TODO 5: If metrics within acceptable thresholds, return success
        # TODO 6: If metrics exceed thresholds, trigger automatic rollback
        # Hint: Use metrics_collector.compare_metrics() for statistical validation
        pass
    
    async def _validate_canary_metrics(self, canary_metrics: Dict, 
                                     baseline_metrics: Dict) -> bool:
        """Validate canary metrics against baseline with statistical significance."""
        # TODO 1: Compare error rates between canary and baseline traffic
        # TODO 2: Validate response time percentiles are within acceptable ranges
        # TODO 3: Check business metrics (conversion, revenue) for regressions
        # TODO 4: Apply statistical significance testing for metric differences
        # TODO 5: Return True only if all metrics pass validation thresholds
        # Hint: Consider both absolute thresholds and relative percentage changes
        # Hint: Require minimum sample size for statistical significance
        pass
    
    async def rollback(self) -> bool:
        """Rollback canary deployment by reverting traffic to baseline."""
        # TODO 1: Immediately route 100% traffic back to baseline targets
        # TODO 2: Stop new version deployment on canary targets
        # TODO 3: Optionally redeploy previous version to canary targets
        # TODO 4: Monitor rollback metrics to ensure service restoration
        # TODO 5: Clean up canary deployment state and mark as rolled back
        pass
```

**Language-Specific Hints:**

- **Async/Await**: Use `asyncio` for concurrent health checks and metric collection across multiple deployment targets
- **Process Management**: Use `subprocess.run()` for load balancer configuration updates and external tool integration
- **File Operations**: Use `tempfile.NamedTemporaryFile()` for atomic configuration file updates
- **HTTP Clients**: Use `aiohttp` for non-blocking health check requests and API calls
- **Configuration**: Use `dataclasses` for type-safe configuration management
- **Logging**: Use structured logging with deployment context for troubleshooting
- **Error Handling**: Implement specific exception types for different failure modes (health check timeout, traffic routing failure, metric validation failure)

**Milestone Checkpoint:**

After implementing deployment strategies, verify correct behavior:

1. **Rolling Deployment Test**: `python -m pytest tests/test_rolling_deployment.py -v`
   - Expected: All tests pass showing incremental instance updates with health validation
   - Manual verification: Deploy to 3 test instances, observe one-by-one updates in logs
   - Check: Failed instance should trigger automatic rollback of entire deployment

2. **Blue-Green Deployment Test**: `python -m pytest tests/test_blue_green_deployment.py -v`
   - Expected: Atomic traffic switching between environments with zero downtime
   - Manual verification: Monitor traffic routing during switch, should see instant changeover
   - Check: Rollback should restore traffic to previous environment within seconds

3. **Canary Deployment Test**: `python -m pytest tests/test_canary_deployment.py -v`
   - Expected: Graduated traffic shifting with automatic rollback on metric regression
   - Manual verification: Observe traffic percentages increasing through canary stages
   - Check: Simulate high error rate in canary, should trigger automatic rollback

4. **Integration Test**: Run full deployment pipeline with artifact from Milestone 3
   - Command: `python scripts/test_deployment_integration.py --strategy rolling --artifact test-app-v1.2.3.tar.gz`
   - Expected: Complete deployment flow from artifact download through health validation
   - Signs of problems: Deployment hangs, health checks never pass, traffic routing fails


## Interactions and Data Flow

> **Milestone(s):** This section demonstrates how all four milestones (1-4) work together by detailing the complete data flow from YAML parsing through deployment.

Think of the interactions and data flow as the **choreographed dance** of our CI/CD factory. Just as a ballet performance appears seamless to the audience but requires precise coordination between dancers, musicians, lighting, and stage crew, our CI/CD pipeline orchestrates multiple components working in harmony. Each component has its moment to perform while supporting the overall production. The Pipeline Parser sets the stage, the Job Executor performs the main acts, the Artifact Manager handles the props and costumes between scenes, and the Deployment Engine delivers the final performance to the audience.

Understanding these interactions is crucial because **the system's reliability depends not just on individual components working correctly, but on their coordination during handoffs**. Most CI/CD failures occur not within components but at the boundaries between them - when artifacts are corrupted during transfer, when deployment strategies receive incomplete information, or when job dependencies create deadlocks.

![Pipeline Execution Sequence](./diagrams/execution-sequence.svg)

### Pipeline Execution Flow

The pipeline execution flow represents the **main sequence** from receiving a pipeline definition to completing all jobs. Think of this as the **master conductor's score** that coordinates all the musicians in our CI/CD orchestra.

#### Initialization Phase

The execution flow begins when a trigger event occurs - typically a git push, pull request, or manual pipeline start. The system must bootstrap itself from a cold state into active execution.

| Step | Component | Action | Data Transformed | Success Criteria |
|------|-----------|--------|------------------|------------------|
| 1 | Pipeline Parser | Read YAML definition | Raw YAML → `PipelineDefinition` | Valid syntax, no circular dependencies |
| 2 | Pipeline Parser | Validate schema | `PipelineDefinition` → Validated structure | All required fields present, types correct |
| 3 | Pipeline Parser | Build dependency graph | Job definitions → DAG representation | Topological sort succeeds |
| 4 | Pipeline Parser | Resolve variables | Template expressions → Concrete values | All variable references resolved |
| 5 | Execution State | Initialize tracking | Pipeline definition → `ExecutionState` | All jobs marked `PENDING` |

The initialization phase performs several critical validations that prevent runtime failures:

> **Key Insight**: Variable resolution happens once during initialization, not repeatedly during execution. This ensures consistent behavior and catches template errors early, but means dynamic variables (like build numbers) must be computed before parsing begins.

**Variable Resolution Process**: The parser walks through all job definitions and step scripts, identifying template expressions like `${BUILD_NUMBER}` or `${JOB.previous-job.artifacts.binary}`. It maintains a resolution context that includes global environment variables, pipeline parameters, and artifact references from completed jobs. Unresolved references cause immediate pipeline failure with specific error messages identifying the missing variable and its location.

**Dependency Graph Construction**: The parser uses the `build_dependency_graph()` method to create a directed acyclic graph where nodes represent jobs and edges represent dependencies. It performs depth-first search to detect circular dependencies, throwing `CircularDependencyError` with the complete cycle path. The resulting DAG enables parallel execution while respecting ordering constraints.

#### Execution Coordination Phase

Once initialization succeeds, the system transitions to active execution. The `ExecutionState` component acts as the central coordinator, managing job lifecycle and dependency satisfaction.

| Job State Transitions | Trigger Event | Next State | Actions Performed |
|----------------------|---------------|------------|-------------------|
| `PENDING` → `RUNNING` | All dependencies completed successfully | `RUNNING` | Reserve execution slot, create workspace |
| `RUNNING` → `SUCCESS` | All steps completed with exit code 0 | `SUCCESS` | Upload artifacts, notify dependents |
| `RUNNING` → `FAILED` | Step failed and retry exhausted | `FAILED` | Cancel dependents, preserve logs |
| `RUNNING` → `RETRY` | Step failed but retries remaining | `RETRY` | Clean workspace, increment attempt counter |
| `FAILED` → `CANCELLED` | Manual intervention or timeout | `CANCELLED` | Release resources, mark pipeline failed |

The coordination phase implements a **reactive scheduling algorithm**: rather than pre-computing an execution schedule, it continuously evaluates which jobs are eligible to run based on current system state. This approach handles failures gracefully because it doesn't assume a predetermined execution path.

**Ready Job Detection Algorithm**:
1. The `get_ready_jobs()` method scans all jobs in `PENDING` state
2. For each pending job, it checks that all dependencies are in `SUCCESS` state
3. It verifies that required artifacts from dependencies are available in storage
4. It confirms that execution capacity is available (respecting concurrency limits)
5. Jobs meeting all criteria transition to `RUNNING` and begin execution

**Dependency Satisfaction**: When a job completes successfully, `mark_job_completed()` performs several coordinated actions:
- Updates the job's status and timestamps
- Triggers artifact upload for any declared outputs
- Scans all pending jobs to identify newly eligible candidates  
- Emits progress events for monitoring and UI updates
- Handles failure propagation if the job failed

#### Step Execution Sequence

Within each job, individual steps execute sequentially according to their definition order. The Job Executor maintains strict isolation between steps while providing consistent environment setup.

The step execution sequence follows this detailed protocol:

1. **Workspace Preparation**: The executor creates an isolated workspace directory and mounts it as a volume in the execution container. This workspace persists across all steps in the job, allowing them to share intermediate files.

2. **Environment Assembly**: The executor merges variables from multiple sources in precedence order: global pipeline environment, job-specific environment, and step-specific environment. Later sources override earlier ones for the same variable name.

3. **Container Initialization**: Using `DockerClient.run_command()`, the executor starts a container with the specified image, working directory set to the workspace mount, and merged environment variables injected.

4. **Script Execution**: The step's script commands execute in sequence within the container. The executor captures both stdout and stderr streams, forwarding them to the pipeline logs in real-time using `streaming_run()`.

5. **Exit Code Evaluation**: When the container terminates, the executor examines the exit code. Zero indicates success; any non-zero value indicates failure and triggers retry logic if configured.

6. **Cleanup and Continuation**: The executor removes the container but preserves the workspace for subsequent steps. If `continue_on_failure` is true, step failure doesn't abort the job.

**Timeout Handling**: Each step runs under a configurable timeout managed by the executor. When a timeout expires, the executor sends SIGTERM to the container, waits for a grace period, then forces termination with SIGKILL. The step is marked as failed with a timeout-specific error message.

**Retry Logic Implementation**: Failed steps with `retry_count > 0` enter the retry subsystem. The `ExponentialBackoffRetry` policy calculates delay periods that double with each attempt, preventing thundering herd problems. The `should_retry_failure()` method examines exit codes and error output to distinguish transient failures (network timeouts, resource exhaustion) from permanent failures (compilation errors, test failures).

### Artifact Transfer Flow

The artifact transfer flow manages the **supply chain** of build outputs between pipeline stages. Think of this as the **logistics network** that ensures the right materials reach the right workers at the right time, with quality guarantees throughout the journey.

![Artifact Transfer Flow](./diagrams/artifact-flow.svg)

Artifacts represent the **contract** between pipeline stages - they are the promises that completed jobs make to future jobs about what they have produced. This contract includes not just the file contents but also integrity guarantees, metadata annotations, and lifecycle management.

#### Upload Process

Artifact upload occurs immediately after successful job completion, before the job's status transitions to `SUCCESS`. This timing ensures that dependent jobs never see a completed status without available artifacts.

**Upload Sequence**:

1. **Content Discovery**: The Job Executor scans the workspace for files matching the artifact patterns declared in the job definition. Patterns support glob syntax like `build/artifacts/*.jar` or `test-results/**/*.xml`.

2. **Content Hashing**: For each discovered file, `generate_artifact_id()` computes a SHA-256 hash of the file contents using streaming I/O to handle large artifacts without memory exhaustion. This hash becomes the artifact's permanent identifier in content-addressable storage.

3. **Metadata Collection**: The system extracts metadata including original filename, file size, MIME type detection, creation timestamp, and any custom annotations from the job definition.

4. **Storage Transaction**: The `store_artifact()` method performs an atomic operation: write the file content to the storage backend using the hash as the key, then write the metadata record. If either operation fails, both are rolled back.

5. **Reference Recording**: The system updates the job's artifact registry with mappings from logical artifact names (as declared in the YAML) to content hashes. This enables dependent jobs to request artifacts by name rather than hash.

**Deduplication Benefits**: Content-addressable storage automatically deduplicates identical artifacts across jobs and pipelines. When two jobs produce identical output files, only one copy is stored, with both jobs referencing the same hash. This dramatically reduces storage requirements for artifacts like common dependencies or test fixtures.

**Integrity Guarantees**: The upload process creates an **immutable artifact** - once stored, the content cannot be modified without changing its hash (and thus its identity). This property enables caching optimizations and ensures that artifact references remain valid throughout the pipeline lifecycle.

#### Download Process

Artifact download occurs during job initialization, before any steps execute. The Job Executor pre-fetches all declared dependencies to ensure they're locally available throughout job execution.

**Download Sequence**:

1. **Dependency Resolution**: The executor examines the job's `depends_on` list and resolves each dependency to determine what artifacts it should have produced. This information comes from the dependency job's artifact declarations in the pipeline definition.

2. **Hash Lookup**: For each required artifact, the executor queries the artifact registry to map logical names to content hashes. Missing mappings indicate that a dependency job failed to produce expected outputs.

3. **Availability Check**: Before attempting download, `artifact_exists()` verifies that the content hash exists in storage. This check prevents expensive download attempts for missing artifacts.

4. **Content Retrieval**: `retrieve_artifact()` streams the content from storage into the job's workspace, recreating the original file structure. Large artifacts are handled with streaming I/O to avoid memory pressure.

5. **Integrity Verification**: After download, `verify_artifact_integrity()` recalculates the content hash and compares it to the expected value. Hash mismatches indicate corruption and trigger automatic retry from alternative storage replicas if available.

6. **Workspace Integration**: Downloaded artifacts are placed in the workspace according to the job's artifact mapping configuration, making them available to all subsequent steps at predictable paths.

**Failure Handling**: Download failures are treated as hard errors that prevent job execution, since jobs cannot proceed without their declared dependencies. The system distinguishes between temporary failures (storage service unavailable) and permanent failures (artifact not found) to guide retry behavior.

**Caching Optimization**: The download process implements workspace caching - if a job runs multiple times with identical dependencies, previously downloaded artifacts can be reused without re-fetching from storage, dramatically improving pipeline performance for incremental builds.

#### Storage Backend Operations

The storage backend manages the physical organization of artifact data with hash-based directory structures that provide efficient lookup and enumeration capabilities.

**Storage Layout**:
```
artifacts/
  ab/cd/ef/abcdef123456...  ← content file (first 6 chars become directory path)
  ab/cd/ef/abcdef123456....meta  ← metadata file
  .retention/
    candidates-2024-01-15.txt  ← retention processing workspace
  .locks/
    abcdef123456.lock  ← upload coordination locks
```

**Content-Addressable Benefits**: This organization provides several advantages:
- **Automatic deduplication**: Identical content always maps to the same path
- **Integrity verification**: Path mismatch immediately indicates corruption
- **Efficient enumeration**: Directory structure enables efficient retention processing
- **Atomic operations**: File system operations provide natural transaction boundaries

**Concurrent Access Handling**: Multiple jobs may attempt to upload identical content simultaneously. The storage backend uses file locking to coordinate these operations: the first uploader wins and completes the storage operation, while subsequent uploaders detect the existing content and skip redundant work.

**Metadata Management**: Each content file has a corresponding metadata file containing the `ArtifactInfo` structure serialized as JSON. This separation allows fast metadata queries without reading large content files, enabling efficient retention policy evaluation.

### Deployment Flow

The deployment flow represents the **final mile** of software delivery - the critical transition from build artifacts to running production systems. Think of deployment as the **precision landing** of a spacecraft: all the preparation, testing, and validation culminates in a carefully orchestrated sequence that must execute flawlessly under real-world conditions.

Each deployment strategy implements a different risk-versus-speed trade-off, like choosing between a gentle glider landing (rolling deployment), an instant teleportation (blue-green), or a cautious reconnaissance mission (canary).

#### Rolling Deployment Flow

Rolling deployment updates instances incrementally, validating each change before proceeding. This strategy minimizes resource requirements while providing gradual risk exposure.

**Rolling Deployment Sequence**:

1. **Pre-deployment Validation**: The deployment engine verifies that the new version's artifacts are available and that all target instances are healthy before beginning. This prevents starting a deployment that cannot complete successfully.

2. **Instance Selection**: The strategy selects the first instance for update using a deterministic algorithm (typically oldest-first) to ensure consistent behavior across deployments. Instance selection considers current health status and avoids updating instances that are already serving elevated traffic.

3. **Health Check Baseline**: Before updating any instance, the system records baseline health metrics and response times. These baselines enable automated detection of performance regressions during deployment.

4. **Single Instance Update**: The selected instance is taken out of the load balancer rotation, updated with the new version, and restarted. The `check_health()` method continuously monitors the instance's readiness endpoints.

5. **Health Validation**: `wait_for_healthy()` polls the updated instance until it passes all health checks or times out. Health validation includes not just basic connectivity but also application-specific readiness indicators.

6. **Traffic Restoration**: Once health checks pass, the instance rejoins the load balancer rotation. The system monitors for elevated error rates or response time degradation that might indicate problems invisible to health checks.

7. **Iteration Control**: If the instance update succeeds, the process repeats with the next instance. If it fails, the deployment halts and triggers rollback procedures for any instances already updated.

**Failure Detection and Rollback**: Rolling deployments continuously monitor application metrics during the update process. The `compare_metrics()` function evaluates error rates, response times, and custom application metrics against baseline values. Significant degradation triggers automatic rollback by reverting updated instances to the previous version.

**Coordination with Load Balancers**: Rolling deployments require careful coordination with load balancing infrastructure. The `update_traffic_weights()` method temporarily removes updating instances from rotation, preventing user traffic from reaching partially updated or restarting services.

#### Blue-Green Deployment Flow

Blue-green deployment maintains two complete production environments, enabling instant traffic switching with immediate rollback capability.

**Blue-Green Environment Management**:

The deployment engine maintains metadata about both environments, tracking which is currently active (serving production traffic) and which is staging (available for updates).

| Environment State | Traffic Weight | Update Allowed | Health Check Frequency |
|-------------------|----------------|----------------|------------------------|
| Active Production | 100% | No | Every 30 seconds |
| Staging Ready | 0% | Yes | Every 10 seconds |
| Switching | 50%/50% (briefly) | No | Every 5 seconds |
| Failed Staging | 0% | Yes | Every 60 seconds |

**Blue-Green Deployment Sequence**:

1. **Environment Preparation**: The deployment engine identifies the inactive environment (green if blue is active, or vice versa) and verifies it has sufficient capacity and correct configuration for the new version.

2. **Full Deployment**: The entire application stack deploys to the inactive environment simultaneously. Unlike rolling deployment, this creates a complete parallel environment rather than updating instances incrementally.

3. **Comprehensive Testing**: The inactive environment undergoes extensive testing including smoke tests, integration tests, and synthetic transaction validation. This testing uses production-equivalent data and load patterns.

4. **Atomic Traffic Switch**: After successful validation, `update_traffic_weights()` instantly redirects all production traffic from the active environment to the newly deployed environment. This switch typically occurs at the load balancer level.

5. **Monitoring Window**: The deployment engine monitors the newly active environment for a configured duration (typically 10-15 minutes), watching for issues that manifest only under real production load.

6. **Environment Role Swap**: If monitoring confirms successful deployment, the environments swap roles: the previously inactive environment becomes the new production, while the old production becomes the new staging environment available for the next deployment.

**Atomic Switchover Implementation**: Blue-green deployment's key advantage is atomic switchover - users experience either the old version or the new version, never a mixture. This requires sophisticated load balancer integration that can redirect traffic instantly without connection disruption.

**Database Migration Challenges**: Blue-green deployments face unique challenges with database changes. The deployment engine must coordinate database migrations that are compatible with both versions during the switch window, or use database-per-environment strategies that add significant infrastructure complexity.

#### Canary Deployment Flow

Canary deployment gradually shifts traffic to a new version while continuously monitoring metrics to detect problems before they affect all users.

**Progressive Traffic Shifting**:

Canary deployment implements a multi-stage process where traffic gradually increases to the new version based on confidence levels derived from observed metrics.

| Canary Stage | Traffic Percentage | Duration | Success Criteria |
|--------------|-------------------|----------|------------------|
| Initial | 5% | 10 minutes | Error rate < baseline + 10% |
| Expansion | 25% | 15 minutes | Response time < baseline + 15% |
| Majority | 50% | 20 minutes | Custom metrics within thresholds |
| Completion | 100% | Ongoing | Sustained normal operation |

**Canary Deployment Sequence**:

1. **Canary Environment Setup**: The deployment engine creates a small number of instances running the new version alongside the existing production instances. Canary instances typically represent 5-10% of total capacity.

2. **Initial Traffic Routing**: Load balancer configuration routes a small percentage of production traffic to canary instances using weighted routing rules. User assignment to canary instances can be random or based on criteria like user cohorts or geographic regions.

3. **Metric Collection**: The system begins intensive monitoring of both canary and production instances, collecting metrics including error rates, response times, resource utilization, and application-specific indicators.

4. **Automated Analysis**: `compare_metrics()` continuously evaluates canary performance against production baselines. The analysis uses statistical methods to distinguish normal variance from significant degradation.

5. **Progressive Scaling**: If canary metrics remain within acceptable thresholds, the deployment engine gradually increases canary capacity while reducing production instance count, maintaining constant total capacity.

6. **Traffic Weight Adjustment**: `update_traffic_weights()` progressively shifts more traffic to canary instances, following a predetermined schedule or confidence-based triggers. Each traffic increase triggers a new monitoring period.

7. **Completion or Rollback**: If the canary successfully serves 100% of traffic for a sustained period, the deployment completes. If any stage shows metric degradation, automatic rollback immediately routes all traffic back to the original version.

**Statistical Confidence**: Canary deployments rely on statistical analysis to distinguish real problems from random variation. The deployment engine implements confidence intervals and hypothesis testing to avoid false positives that would cause unnecessary rollbacks.

**User Experience Consistency**: Advanced canary implementations maintain user session affinity to ensure individual users don't switch between versions mid-session, which could cause inconsistent behavior or data corruption.

> **Critical Design Insight**: The deployment flow's sophistication directly impacts system reliability and user experience. Simple deployments introduce risk through sudden changes, while sophisticated strategies like canary deployments provide safety at the cost of complexity and deployment duration.

### Implementation Guidance

The interactions and data flow require careful orchestration between components with proper error handling and state management throughout the pipeline lifecycle.

#### Technology Recommendations

| Component Integration | Simple Option | Advanced Option |
|----------------------|---------------|-----------------|
| Event Communication | Direct method calls with callback interfaces | Message queue (Redis Pub/Sub) with event schemas |
| State Persistence | In-memory dictionaries with periodic JSON dumps | SQLite database with transaction support |
| Progress Monitoring | File-based status updates with polling | WebSocket connections for real-time updates |
| Error Propagation | Exception raising with structured error types | Result types with error context chains |

#### Recommended File Structure

```
pipeline-orchestrator/
  src/
    orchestrator/
      __init__.py
      coordinator.py          ← main orchestration logic
      execution_state.py      ← pipeline state management
      event_dispatcher.py     ← component communication
    flows/
      __init__.py
      pipeline_flow.py        ← pipeline execution coordination
      artifact_flow.py        ← artifact transfer orchestration
      deployment_flow.py      ← deployment strategy coordination
    integration/
      __init__.py
      component_registry.py   ← component lifecycle management
      health_monitor.py       ← cross-component health checking
  tests/
    integration/
      test_pipeline_flow.py
      test_artifact_flow.py
      test_deployment_flow.py
```

#### Core Orchestration Infrastructure

```python
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
from contextlib import asynccontextmanager

class FlowState(Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSING = "pausing"
    PAUSED = "paused"
    RESUMING = "resuming"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class FlowEvent:
    """Event representing a state change or significant occurrence in pipeline flow."""
    event_type: str
    component: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""

class EventDispatcher:
    """Coordinates communication between pipeline components."""
    
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}
        self._event_history: List[FlowEvent] = []
    
    def subscribe(self, event_type: str, callback: Callable[[FlowEvent], None]):
        """Register callback for specific event type."""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)
    
    def emit(self, event: FlowEvent):
        """Dispatch event to all registered listeners."""
        self._event_history.append(event)
        listeners = self._listeners.get(event.event_type, [])
        for callback in listeners:
            try:
                callback(event)
            except Exception as e:
                logging.error(f"Event callback failed: {e}")
    
    def get_event_history(self, since: Optional[datetime] = None) -> List[FlowEvent]:
        """Retrieve event history for debugging and monitoring."""
        if since is None:
            return self._event_history.copy()
        return [e for e in self._event_history if e.timestamp >= since]

class ComponentHealthMonitor:
    """Monitors health of pipeline components and coordinates recovery."""
    
    def __init__(self, event_dispatcher: EventDispatcher):
        self.event_dispatcher = event_dispatcher
        self.component_status: Dict[str, bool] = {}
        self.last_health_check: Dict[str, datetime] = {}
    
    async def check_component_health(self, component_name: str, health_check: Callable[[], bool]) -> bool:
        """Execute health check for component and update status."""
        try:
            is_healthy = health_check()
            self.component_status[component_name] = is_healthy
            self.last_health_check[component_name] = datetime.now()
            
            if not is_healthy:
                self.event_dispatcher.emit(FlowEvent(
                    event_type="component.unhealthy",
                    component=component_name,
                    timestamp=datetime.now(),
                    data={"status": "unhealthy"}
                ))
            
            return is_healthy
        except Exception as e:
            logging.error(f"Health check failed for {component_name}: {e}")
            return False
    
    def get_system_health(self) -> Dict[str, Any]:
        """Return overall system health status."""
        total_components = len(self.component_status)
        healthy_components = sum(1 for status in self.component_status.values() if status)
        
        return {
            "overall_healthy": healthy_components == total_components,
            "component_count": total_components,
            "healthy_count": healthy_components,
            "component_status": self.component_status.copy(),
            "last_check_times": self.last_health_check.copy()
        }
```

#### Pipeline Flow Coordination Skeleton

```python
class PipelineFlowCoordinator:
    """Orchestrates the complete pipeline execution flow from YAML parsing to completion."""
    
    def __init__(self, parser, executor, artifact_manager, event_dispatcher):
        self.parser = parser
        self.executor = executor
        self.artifact_manager = artifact_manager
        self.event_dispatcher = event_dispatcher
        self.execution_state: Optional[ExecutionState] = None
        self.flow_state = FlowState.INITIALIZING
    
    async def execute_pipeline(self, yaml_content: str, pipeline_params: Dict[str, Any]) -> bool:
        """Execute complete pipeline flow from definition to completion."""
        # TODO 1: Parse YAML definition and validate structure
        # TODO 2: Build dependency graph and check for cycles
        # TODO 3: Initialize ExecutionState with all jobs marked PENDING
        # TODO 4: Enter main execution loop, processing ready jobs
        # TODO 5: Handle job completion events and update dependency satisfaction
        # TODO 6: Coordinate artifact uploads after successful job completion
        # TODO 7: Monitor for pipeline completion or failure conditions
        # TODO 8: Clean up resources and emit final pipeline status event
        pass
    
    async def _process_ready_jobs(self) -> List[str]:
        """Identify and start execution of jobs whose dependencies are satisfied."""
        # TODO 1: Query ExecutionState for jobs in PENDING status
        # TODO 2: Check dependency satisfaction for each pending job
        # TODO 3: Verify required artifacts are available in storage
        # TODO 4: Submit eligible jobs to executor with proper isolation
        # TODO 5: Update job status to RUNNING and emit progression events
        # TODO 6: Return list of job names that were started
        pass
    
    def _handle_job_completion(self, job_name: str, success: bool, artifacts: Dict[str, str]):
        """Process job completion and trigger dependent job evaluation."""
        # TODO 1: Update job status in ExecutionState (SUCCESS or FAILED)
        # TODO 2: If successful, coordinate artifact upload to storage
        # TODO 3: Record job completion timestamp and execution summary
        # TODO 4: Evaluate newly eligible dependent jobs
        # TODO 5: Emit job completion event with artifacts and timing info
        # TODO 6: Check if pipeline is complete (all jobs finished)
        pass
    
    async def pause_pipeline(self) -> bool:
        """Gracefully pause pipeline execution, allowing running jobs to complete."""
        # TODO 1: Set flow state to PAUSING to prevent new job starts
        # TODO 2: Wait for currently running jobs to complete or timeout
        # TODO 3: Preserve ExecutionState for later resumption
        # TODO 4: Emit pipeline paused event with current progress
        pass
```

#### Artifact Flow Management Skeleton

```python
class ArtifactFlowManager:
    """Manages artifact transfer flow between pipeline jobs."""
    
    def __init__(self, artifact_manager, event_dispatcher):
        self.artifact_manager = artifact_manager
        self.event_dispatcher = event_dispatcher
        self.transfer_tracking: Dict[str, Dict[str, Any]] = {}
    
    async def coordinate_artifact_upload(self, job_name: str, workspace_path: str, artifact_specs: Dict[str, str]) -> Dict[str, str]:
        """Coordinate uploading job artifacts to storage with integrity verification."""
        # TODO 1: Scan workspace for files matching artifact patterns
        # TODO 2: Calculate content hashes for all discovered artifacts
        # TODO 3: Check if identical artifacts already exist (deduplication)
        # TODO 4: Upload new artifacts to storage backend
        # TODO 5: Record artifact registry mappings (name -> hash)
        # TODO 6: Emit artifact upload events for monitoring
        # TODO 7: Return mapping of artifact names to storage identifiers
        pass
    
    async def coordinate_artifact_download(self, job_name: str, workspace_path: str, required_artifacts: List[str]) -> bool:
        """Download required artifacts to job workspace with integrity verification."""
        # TODO 1: Resolve artifact names to content hashes via registry
        # TODO 2: Check artifact availability in storage backend
        # TODO 3: Download artifacts using streaming I/O for large files
        # TODO 4: Verify integrity using content hash validation
        # TODO 5: Place artifacts in workspace at expected paths
        # TODO 6: Update artifact access timestamps for retention
        # TODO 7: Emit download completion events with transfer stats
        pass
    
    def track_artifact_lineage(self, artifact_hash: str, producer_job: str, consumer_jobs: List[str]):
        """Record artifact lineage for debugging and retention decisions."""
        # TODO 1: Create lineage record linking artifact to producing job
        # TODO 2: Record all consuming jobs for reference counting
        # TODO 3: Update artifact metadata with lineage information
        # TODO 4: Enable lineage queries for debugging failed dependencies
        pass
```

#### Deployment Flow Orchestrator Skeleton

```python
class DeploymentFlowOrchestrator:
    """Orchestrates deployment strategies with health monitoring and rollback."""
    
    def __init__(self, deployment_strategies: Dict[str, DeploymentStrategy], event_dispatcher):
        self.strategies = deployment_strategies
        self.event_dispatcher = event_dispatcher
        self.active_deployments: Dict[str, Any] = {}
    
    async def execute_deployment(self, strategy_name: str, version: str, targets: List[DeploymentTarget]) -> bool:
        """Execute deployment using specified strategy with monitoring and rollback."""
        # TODO 1: Validate deployment parameters and artifact availability
        # TODO 2: Select appropriate deployment strategy implementation
        # TODO 3: Record baseline metrics before deployment begins
        # TODO 4: Execute deployment strategy with progress monitoring
        # TODO 5: Continuously evaluate health metrics during deployment
        # TODO 6: Trigger automatic rollback if metrics exceed thresholds
        # TODO 7: Emit deployment completion or failure events
        pass
    
    async def _monitor_deployment_health(self, deployment_id: str, strategy: DeploymentStrategy) -> bool:
        """Monitor deployment health and trigger rollback if necessary."""
        # TODO 1: Collect current metrics from all deployment targets
        # TODO 2: Compare metrics against baseline using statistical analysis
        # TODO 3: Check application-specific health indicators
        # TODO 4: Evaluate user-defined success criteria and thresholds
        # TODO 5: Make rollback decision based on metric degradation
        # TODO 6: Coordinate rollback execution if health checks fail
        pass
    
    def _coordinate_traffic_management(self, targets: List[TrafficTarget]) -> bool:
        """Coordinate load balancer updates for traffic routing changes."""
        # TODO 1: Validate traffic weight assignments sum to 100%
        # TODO 2: Update load balancer configuration atomically
        # TODO 3: Verify traffic routing changes took effect
        # TODO 4: Monitor for connection disruption during switches
        # TODO 5: Provide rollback capability for traffic changes
        pass
```

#### Integration Testing Framework

```python
import pytest
from unittest.mock import Mock, AsyncMock
import tempfile
import os

class IntegrationTestHarness:
    """Test harness for validating component interactions and data flow."""
    
    def __init__(self):
        self.temp_dirs = []
        self.mock_components = {}
    
    def setup_test_environment(self):
        """Create isolated test environment with temporary storage."""
        workspace = tempfile.mkdtemp(prefix="pipeline_test_")
        self.temp_dirs.append(workspace)
        return workspace
    
    def create_mock_components(self):
        """Create mock components with realistic behavior for integration testing."""
        return {
            'parser': Mock(),
            'executor': AsyncMock(),
            'artifact_manager': Mock(),
            'event_dispatcher': Mock()
        }
    
    def cleanup(self):
        """Clean up test environment and temporary files."""
        for temp_dir in self.temp_dirs:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def integration_harness():
    """Pytest fixture providing integration test harness."""
    harness = IntegrationTestHarness()
    yield harness
    harness.cleanup()

# Example integration test
async def test_complete_pipeline_flow(integration_harness):
    """Test complete pipeline flow from YAML parsing to artifact management."""
    # TODO: Implement comprehensive integration test that:
    # 1. Parses a realistic YAML pipeline definition
    # 2. Executes jobs with artifact dependencies
    # 3. Validates artifact transfer between jobs
    # 4. Verifies proper cleanup and state management
    pass
```

#### Milestone Checkpoints

**Pipeline Execution Flow Checkpoint**:
After implementing the pipeline coordination:
```bash
python -m pytest tests/integration/test_pipeline_flow.py -v
```

Expected behavior: Pipeline should parse YAML, execute jobs in dependency order, handle failures gracefully, and maintain consistent state throughout execution. Test with a 3-job pipeline where job2 depends on job1, and job3 depends on job2.

**Artifact Transfer Flow Checkpoint**:
Verify artifact management integration:
```bash
python -m pytest tests/integration/test_artifact_flow.py -v
```

Expected behavior: Jobs should successfully upload artifacts after completion, dependent jobs should download required artifacts before execution, and integrity verification should detect any corruption during transfer.

**Deployment Flow Checkpoint**:
Test deployment strategy coordination:
```bash
python -m pytest tests/integration/test_deployment_flow.py -v
```

Expected behavior: Deployment strategies should execute with proper health monitoring, traffic management should coordinate load balancer updates, and rollback should trigger automatically when health thresholds are exceeded.

#### Debugging Tips

| Symptom | Likely Cause | Diagnosis | Fix |
|---------|--------------|-----------|-----|
| Pipeline hangs indefinitely | Circular dependency or missing job | Check dependency graph validation, examine job status | Implement cycle detection, validate job references |
| Artifact download fails | Hash mismatch or storage corruption | Check integrity verification logs, validate storage | Implement retry with alternative sources, verify checksums |
| Deployment rollback fails | Health check timeout or traffic management error | Examine load balancer logs, check metric collection | Add fallback rollback strategies, improve health check reliability |
| Jobs start before dependencies complete | Race condition in state management | Check ExecutionState updates, verify event ordering | Add proper synchronization, implement state machine validation |
| Memory exhaustion during artifact transfer | Large files loaded entirely into memory | Monitor memory usage during transfers, check streaming implementation | Implement streaming I/O, add memory limits |


## Error Handling and Edge Cases

> **Milestone(s):** This section provides comprehensive failure analysis and recovery mechanisms that apply across all milestones (1-4), ensuring robust error handling throughout the CI/CD pipeline system.

Think of error handling in a CI/CD pipeline as the **emergency response system** for a modern hospital. Just as a hospital has protocols for every possible medical emergency - cardiac arrest, natural disasters, power outages, equipment failures - our CI/CD pipeline needs comprehensive failure detection and recovery procedures for every component. The hospital's emergency protocols don't just react to problems; they prevent cascading failures, maintain essential services during crises, and have clear escalation paths from automatic responses to manual intervention. Similarly, our pipeline must gracefully handle everything from simple network hiccups to catastrophic infrastructure failures while maintaining system integrity and providing clear diagnostics.

The key insight is that failure handling is not an afterthought - it's a core architectural concern that shapes every design decision. Just as hospital emergency protocols influence the placement of emergency exits, backup power systems, and communication networks, our error handling strategy influences component boundaries, state management, logging architecture, and recovery mechanisms. A well-designed failure handling system turns unpredictable chaos into manageable, recoverable situations with clear resolution paths.

![Job Execution State Machine](./diagrams/job-state-machine.svg)

![Deployment State Machine](./diagrams/deployment-state-machine.svg)

### Failure Mode Analysis

The failure mode analysis serves as our **comprehensive incident playbook** - a systematic catalog of everything that can go wrong, organized by component and impact severity. This analysis follows a structured approach: identifying the failure condition, assessing its blast radius (scope of impact), understanding the detection signals, and mapping the recovery requirements. By cataloging failures systematically, we transform unpredictable emergencies into well-understood scenarios with proven response procedures.

> **Decision: Categorized Failure Classification System**
> - **Context**: CI/CD pipelines have diverse failure modes ranging from transient network issues to permanent infrastructure failures, each requiring different detection and recovery strategies
> - **Options Considered**: Single global error type, component-specific error hierarchies, severity-based classification system
> - **Decision**: Implement categorized failure classification with both component boundaries and severity levels
> - **Rationale**: Component boundaries enable specialized recovery logic while severity levels guide escalation policies and resource allocation decisions
> - **Consequences**: Enables automated recovery for transient failures while escalating persistent issues to manual intervention with appropriate urgency

The failure classification system uses a multi-dimensional approach that considers both the component origin and the failure characteristics:

| Failure Category | Scope | Recovery Strategy | Escalation Threshold |
|------------------|--------|-------------------|---------------------|
| `TRANSIENT` | Single operation | Automatic retry with backoff | 3 consecutive failures |
| `RESOURCE` | Component resources | Resource cleanup + restart | Resource exhaustion |
| `CONFIGURATION` | Component setup | Manual intervention required | Immediate |
| `DEPENDENCY` | External service | Circuit breaker + fallback | Service unavailable |
| `CORRUPTION` | Data integrity | Restore from backup | Data loss detected |
| `CATASTROPHIC` | System-wide | Emergency procedures | System unavailable |

#### Pipeline Parser Failures

The Pipeline Definition Parser faces failures that can render the entire pipeline inoperable, making robust validation and clear error reporting critical for operational success.

**YAML Parsing Failures**: Malformed YAML syntax breaks the fundamental contract between users and the system. These failures occur when developers commit invalid YAML structure, use unsupported syntax features, or introduce encoding issues. The parser must distinguish between recoverable syntax issues (missing quotes, indentation errors) and structural problems (circular references, invalid schema violations) to provide actionable error messages.

| Failure Mode | Trigger Condition | Impact | Detection Method | Recovery Action |
|--------------|------------------|---------|------------------|-----------------|
| Invalid YAML syntax | Malformed indentation, quotes, structure | Pipeline cannot start | YAML parser exception with line/column | Reject with syntax error location |
| Schema validation failure | Missing required fields, wrong types | Pipeline definition unusable | Schema validator error | Reject with field-specific errors |
| Circular dependency | Job A depends on B, B depends on A | Infinite execution loop | Topological sort algorithm failure | Reject with dependency cycle path |
| Variable resolution failure | Reference to undefined variable | Step execution impossible | Variable resolver exception | Reject with undefined variable names |
| Resource limit violation | Pipeline exceeds memory/CPU limits | System instability | Resource monitor alerts | Reject with resource requirement details |

**Dependency Graph Construction Failures**: The dependency graph is the execution backbone of the pipeline. Construction failures occur when job dependencies reference non-existent jobs, create circular dependencies, or violate execution constraints. These failures must be detected during parsing to prevent runtime deadlocks or infinite loops.

The circular dependency detection algorithm uses a **depth-first search with cycle detection** approach. During graph traversal, we maintain a "visiting" set to track the current path and a "visited" set for completed nodes. When we encounter a node already in the "visiting" set, we've found a cycle and can trace the exact dependency path causing the problem.

**Variable Substitution Failures**: Variable resolution failures create particularly challenging debugging scenarios because they often manifest during job execution rather than parsing. The variable resolver must handle nested references, conditional expressions, and scoping rules while providing clear error messages that help users understand complex substitution chains.

| Variable Error Type | Example | Detection Point | Error Message Format |
|--------------------|---------|-----------------|---------------------|
| Undefined variable | `${MISSING_VAR}` | Resolution phase | "Variable 'MISSING_VAR' not found in job 'build' scope" |
| Circular reference | `A=${B}, B=${A}` | Recursive resolution | "Circular variable reference: A → B → A" |
| Type mismatch | `timeout: ${STRING_VAR}` | Type validation | "Expected integer for timeout, got string: '${STRING_VAR}'" |
| Nested resolution failure | `${VAR_${SUFFIX}}` | Complex expression | "Cannot resolve nested variable reference: VAR_${SUFFIX}" |

#### Job Executor Failures

The Job Executor operates in the most volatile environment, interacting with external containers, network resources, and unpredictable user code. Failures here can cascade throughout the pipeline and require sophisticated isolation and recovery mechanisms.

**Container Execution Failures**: Docker container failures represent the most common and varied failure class. These range from simple exit code failures (user code bugs, test failures) to infrastructure problems (Docker daemon unavailable, image pull failures, resource exhaustion). The executor must distinguish between retryable failures (network timeouts, temporary resource constraints) and permanent failures (invalid image references, permission issues).

| Container Failure Type | Root Cause | Retry Strategy | Recovery Action |
|------------------------|------------|----------------|-----------------|
| Image pull failure | Network timeout, invalid image | Exponential backoff, 3 attempts | Switch to cached image or fail job |
| Container start failure | Resource limits, permission denied | No retry | Fail job with resource/permission error |
| Process exit code > 0 | User code error, test failure | Based on `retry_count` setting | Retry with clean container or fail |
| Container killed (OOM) | Memory limit exceeded | Increase memory, retry once | Fail with resource limit explanation |
| Docker daemon unavailable | Infrastructure failure | Circuit breaker pattern | Queue job for later execution |
| Volume mount failure | Path permissions, disk space | No retry | Fail job with filesystem error details |

**Resource Management Failures**: Resource exhaustion failures create cascading problems across multiple jobs. When disk space, memory, or CPU resources become scarce, the executor must implement graceful degradation strategies that prioritize critical jobs while providing clear feedback about resource constraints.

The **resource monitoring system** tracks utilization across multiple dimensions and implements graduated response strategies based on utilization thresholds:

1. **Green Zone (0-70% utilization)**: Normal operation with full parallel execution
2. **Yellow Zone (70-85% utilization)**: Reduce parallel job count, implement queuing
3. **Red Zone (85-95% utilization)**: Emergency throttling, pause non-critical jobs
4. **Critical Zone (95%+ utilization)**: Stop new job starts, begin resource cleanup

**Log Management Failures**: Log capture and streaming failures can hide critical debugging information just when it's needed most. The logging system must handle scenarios where job output exceeds memory limits, network connections fail during log streaming, or disk space runs out during log storage.

| Log Failure Scenario | Impact | Detection | Mitigation |
|----------------------|---------|-----------|------------|
| Log buffer overflow | Lost debugging information | Buffer size monitoring | Implement log rotation, increase buffer |
| Network streaming failure | Real-time monitoring lost | Connection timeout | Cache logs locally, retry transmission |
| Disk space exhaustion | Cannot store job outputs | Disk usage alerts | Compress old logs, emergency cleanup |
| Concurrent log corruption | Garbled multi-job output | Checksum validation | Use atomic log operations, job isolation |

#### Artifact Management Failures

Artifact Management failures can break the entire pipeline workflow by preventing jobs from accessing required dependencies or storing critical outputs. These failures often have delayed impact, making detection and recovery particularly challenging.

**Storage Backend Failures**: The storage backend represents a single point of failure for artifact persistence. Storage failures range from temporary network issues to permanent disk failures, requiring different recovery strategies based on failure persistence and scope.

| Storage Failure Type | Symptoms | Detection Method | Recovery Strategy |
|---------------------|----------|------------------|------------------|
| Disk space exhaustion | Upload operations fail | Storage usage monitoring | Emergency cleanup, compress artifacts |
| File system corruption | Checksum verification fails | Integrity verification | Restore from backup, mark corrupted |
| Network partition | Intermittent access failures | Connection timeout patterns | Retry with backoff, local caching |
| Permission issues | Access denied errors | File operation exceptions | Fix permissions, escalate to admin |
| Hardware failure | Complete storage unavailability | Health check failures | Failover to backup storage |

**Integrity Verification Failures**: Corruption detection during artifact transfer or storage creates complex recovery scenarios. The system must distinguish between transient corruption (network bit errors, temporary storage issues) and persistent corruption (disk failures, software bugs) to choose appropriate recovery actions.

The **streaming integrity verification** process calculates checksums during data transfer to detect corruption without loading entire artifacts into memory:

1. Initialize hash calculator (SHA-256) at transfer start
2. Process data in `DEFAULT_CHUNK_SIZE` chunks during transfer
3. Update running hash with each chunk
4. Compare final hash against expected value
5. On mismatch, retry transfer once, then mark as corrupted
6. Log corruption details for forensic analysis

**Retention Policy Failures**: Retention policy execution failures can lead to storage exhaustion or premature deletion of critical artifacts. The retention system must handle scenarios where deletion operations fail, policy evaluation encounters corrupted metadata, or competing processes create race conditions during cleanup.

| Retention Failure Mode | Cause | Impact | Recovery Method |
|------------------------|-------|---------|-----------------|
| Deletion operation failure | File locked, permission denied | Storage not reclaimed | Retry deletion, escalate to manual cleanup |
| Policy evaluation error | Corrupted metadata, missing fields | Retention decisions incorrect | Rebuild metadata, conservative retention |
| Race condition during cleanup | Multiple processes delete same artifact | Inconsistent state | Use atomic operations, distributed locks |
| Reference counting corruption | Lost reference tracking | Premature deletion risk | Audit reference counts, rebuild from usage logs |

#### Deployment Strategy Failures

Deployment failures have the highest business impact because they directly affect production systems and user-facing services. Deployment strategy failures must be detected rapidly and recovered automatically to minimize service disruption.

**Health Check Failures**: Health check failures can indicate genuine service problems or transient monitoring issues. The deployment system must implement sophisticated health validation that distinguishes between different failure types and responds appropriately to avoid false positives that trigger unnecessary rollbacks.

| Health Check Failure Type | Characteristics | Rollback Decision | Recovery Action |
|---------------------------|----------------|-------------------|-----------------|
| Service startup delay | Temporary failure, improving trend | Wait for threshold | Extend health check timeout |
| Configuration error | Consistent failure, error pattern | Immediate rollback | Revert to previous configuration |
| Resource contention | Intermittent failure, resource alerts | Conditional rollback | Scale resources, retry deployment |
| External dependency failure | Failures outside application | Pause deployment | Wait for dependency recovery |
| Network partition | Infrastructure connectivity issues | Hold current state | Restore network connectivity |

**Traffic Routing Failures**: Load balancer and traffic routing failures during deployment can create split-brain scenarios where different users see different application versions inconsistently. The routing system must maintain atomic switchover capabilities and detect partial routing failures.

**Blue-Green Deployment Specific Failures**: Blue-green deployments face unique failure modes related to environment synchronization and atomic switchover. Environment drift between blue and green instances can cause subtle failures that only manifest after traffic switching.

| Blue-Green Failure Mode | Root Cause | Prevention | Recovery |
|-------------------------|-------------|------------|----------|
| Environment drift | Configuration differences between blue/green | Automated environment validation | Rebuild environment from template |
| Database migration failure | Schema changes incompatible | Database migration testing | Rollback schema, restore data |
| Atomic switchover failure | Load balancer configuration error | Health check before switchover | Emergency traffic restoration |
| Resource exhaustion | Both environments running simultaneously | Resource capacity planning | Emergency resource scaling |

**Canary Deployment Specific Failures**: Canary deployments require sophisticated metrics monitoring and decision-making algorithms. Failures often involve subtle performance degradation or increased error rates that must be detected through statistical analysis rather than binary health checks.

The **canary analysis system** implements multiple detection mechanisms:
- **Error rate comparison**: Statistical significance testing between canary and baseline error rates
- **Performance regression detection**: Response time percentile comparison with confidence intervals  
- **Business metrics impact**: Revenue, conversion rate, or custom business metric analysis
- **Infrastructure metrics**: Resource utilization patterns and efficiency comparisons

### Detection Strategies

Detection strategies serve as our **early warning radar system**, identifying problems before they cascade into system-wide failures. Effective detection combines multiple signal sources - metrics, logs, health checks, and user feedback - to provide comprehensive visibility into system health. The key principle is **layered detection** where multiple independent systems can identify the same failure class, ensuring that critical issues are never missed due to single detector failures.

> **Decision: Multi-Layer Detection Architecture**
> - **Context**: CI/CD pipeline failures can manifest in multiple ways - performance degradation, error rate increases, resource exhaustion - requiring different detection mechanisms
> - **Options Considered**: Single centralized monitoring system, component-specific detection, layered detection with correlation
> - **Decision**: Implement layered detection architecture with automatic correlation and escalation
> - **Rationale**: Layered detection prevents single points of failure in monitoring while correlation reduces alert noise and identifies root causes faster
> - **Consequences**: Requires more complex monitoring infrastructure but provides comprehensive failure detection with reduced false positives

#### Health Check Systems

The health check framework implements **multi-dimensional readiness assessment** that evaluates component health across different aspects: functional correctness, performance characteristics, resource availability, and dependency status. This comprehensive approach prevents false positives where a component appears healthy on simple binary checks but exhibits subtle performance degradation.

| Health Check Dimension | Evaluation Criteria | Success Threshold | Warning Threshold | Failure Threshold |
|------------------------|-------------------|-------------------|-------------------|-------------------|
| **Functional Health** | API endpoints respond correctly | 100% success rate | 99% success rate | <95% success rate |
| **Performance Health** | Response time within limits | <100ms p95 | <200ms p95 | >500ms p95 |
| **Resource Health** | CPU/Memory within limits | <70% utilization | <85% utilization | >95% utilization |
| **Dependency Health** | External services available | All dependencies up | 1 dependency degraded | Critical dependency down |

**Synthetic Health Checks**: Synthetic checks simulate real user workflows to detect problems that simple endpoint checks might miss. For the CI/CD pipeline, synthetic checks involve running minimal test pipelines that exercise all major system components and measure end-to-end performance.

The **synthetic pipeline approach** creates lightweight test jobs that:
1. Parse a minimal YAML pipeline definition
2. Execute a simple container-based job
3. Upload and download a test artifact
4. Perform a mock deployment to a test environment
5. Measure total execution time and flag performance degradation

**Dependency Health Monitoring**: The pipeline system depends on external services like Docker daemon, artifact storage, and deployment targets. Dependency monitoring must distinguish between temporary service degradation and permanent failures to avoid unnecessary system shutdowns.

| Dependency Type | Health Check Method | Check Interval | Failure Threshold | Recovery Detection |
|----------------|-------------------|----------------|-------------------|-------------------|
| Docker Daemon | Container list API call | 30 seconds | 3 consecutive failures | Successful API response |
| Artifact Storage | Write/read test file | 60 seconds | 5 consecutive failures | Successful file operations |
| Database | Connection pool test | 15 seconds | 2 consecutive failures | Connection successful |
| External APIs | Health endpoint ping | 45 seconds | 4 consecutive failures | Response code 200 |

#### Metrics and Alerting

The metrics collection system implements **comprehensive pipeline observability** that captures both technical metrics (execution times, resource usage, error rates) and business metrics (deployment frequency, lead time, failure recovery time). Metrics collection uses a **pull-based architecture** where the monitoring system actively queries components for current status, ensuring metrics are available even when components are under stress.

**Pipeline Execution Metrics**: These metrics track the core pipeline functionality and help identify performance bottlenecks, capacity issues, and quality trends.

| Metric Category | Metric Name | Unit | Alerting Threshold | Business Impact |
|----------------|-------------|------|-------------------|-----------------|
| **Throughput** | `pipelines_started_total` | count/hour | <10/hour during business hours | Development velocity |
| **Performance** | `job_execution_duration_seconds` | seconds | p95 > 300s | Developer feedback time |
| **Reliability** | `job_failure_rate` | percentage | >5% over 1 hour | Code quality confidence |
| **Capacity** | `concurrent_jobs_running` | count | >80% of max capacity | Resource bottlenecks |
| **Quality** | `test_pass_rate` | percentage | <95% over 4 hours | Product stability |

**Infrastructure Metrics**: Infrastructure metrics monitor the underlying platform health and resource utilization patterns that affect pipeline performance.

| Infrastructure Component | Key Metrics | Normal Range | Alert Conditions |
|-------------------------|-------------|--------------|------------------|
| **Docker Engine** | Container start time, image pull duration | <10s start, <30s pull | >60s operations |
| **Storage System** | Disk usage, IOPS, throughput | <80% usage, <1000 IOPS | >90% usage, >5000 IOPS |
| **Network** | Bandwidth utilization, packet loss | <50% bandwidth, 0% loss | >80% bandwidth, >0.1% loss |
| **Compute Resources** | CPU usage, memory usage, load average | <70% CPU, <80% memory | >90% CPU, >95% memory |

**Alerting Rules and Escalation**: The alerting system implements **graduated escalation** that starts with automated remediation, escalates to on-call engineers, and eventually involves management for prolonged outages.

Alerting rules use **statistical significance testing** to reduce false positives from normal operational variance:

1. **Baseline Establishment**: Calculate rolling 7-day baseline for each metric
2. **Anomaly Detection**: Flag deviations >2 standard deviations from baseline
3. **Confirmation Period**: Require anomaly persistence for alert threshold duration
4. **Context Enrichment**: Include related metrics, recent changes, and suggested actions
5. **Escalation Timing**: Automatic escalation if not acknowledged within SLA timeframes

#### Log Analysis and Correlation

Log analysis serves as the **forensic investigation system** that provides detailed context for failures detected by metrics and health checks. The logging architecture implements structured logging with consistent format, correlation IDs, and automated pattern detection to enable rapid root cause analysis during incidents.

**Structured Logging Format**: All components use consistent JSON-structured logging that enables automated parsing and correlation across distributed pipeline execution.

| Log Field | Purpose | Example Value | Correlation Use |
|-----------|---------|---------------|-----------------|
| `timestamp` | Precise timing | `2024-01-15T14:30:45.123Z` | Timeline reconstruction |
| `level` | Severity classification | `ERROR`, `WARN`, `INFO` | Priority filtering |
| `component` | Source identification | `job-executor`, `artifact-manager` | Component-specific analysis |
| `correlation_id` | Request tracking | `pipe-abc123-job-build` | End-to-end tracing |
| `job_id` | Job context | `build-frontend-v1.2.3` | Job-specific debugging |
| `message` | Human-readable description | `Container failed with exit code 1` | Error understanding |
| `metadata` | Additional context | `{"image": "node:16", "duration": 45}` | Detailed analysis |

**Pattern Recognition and Anomaly Detection**: The log analysis system implements automated pattern recognition that identifies common failure signatures and emerging anomaly patterns before they become widespread problems.

**Common Error Patterns**: The system maintains a database of known error signatures that enable rapid classification and automated response for common issues.

| Error Pattern | Log Signature | Automatic Response | Manual Escalation |
|---------------|---------------|-------------------|-------------------|
| **Image Pull Failure** | `"error": "pull access denied"` | Retry with authentication refresh | >3 consecutive failures |
| **Resource Exhaustion** | `"killed": "OOMKilled"` | Increase job memory limits | Resource limit exceeded |
| **Network Timeout** | `"error": "context deadline exceeded"` | Retry with exponential backoff | Network partition detected |
| **Permission Denied** | `"error": "permission denied"` | No automatic retry | Immediate manual review |

**Cross-Component Correlation**: Log correlation uses the `correlation_id` field to trace requests across multiple system components, enabling comprehensive failure analysis that follows the complete execution path.

The **correlation analysis algorithm** works as follows:
1. Extract correlation ID from error log entry
2. Query all components for logs with matching correlation ID
3. Sort timeline by timestamp to reconstruct execution sequence  
4. Identify first failure point and downstream cascading effects
5. Generate correlation report with root cause analysis
6. Attach relevant metrics and health check data for context

### Recovery Mechanisms

Recovery mechanisms represent our **emergency response protocols**, transforming detected failures into concrete remediation actions that restore system functionality. Recovery strategies follow a **graduated response model** that starts with lightweight automatic recovery, escalates through increasingly aggressive interventions, and ultimately falls back to manual procedures when automated recovery fails. The key insight is that different failure types require fundamentally different recovery approaches - transient failures need retry logic, resource exhaustion needs cleanup and scaling, while corruption requires data restoration procedures.

> **Decision: Automated Recovery with Manual Fallback Architecture**
> - **Context**: CI/CD pipeline failures range from simple transient issues to complex systemic problems requiring human judgment and domain expertise
> - **Options Considered**: Fully manual recovery, completely automated recovery, hybrid automated-to-manual escalation
> - **Decision**: Implement automated recovery for well-understood failure classes with clear escalation to manual procedures
> - **Rationale**: Automated recovery handles 80% of routine failures faster than humans while preserving human oversight for complex scenarios requiring business judgment
> - **Consequences**: Requires sophisticated failure classification and automated recovery testing, but dramatically reduces mean time to recovery (MTTR) for common issues

#### Automatic Recovery Strategies

Automatic recovery strategies handle the **routine operational failures** that occur predictably in distributed systems - network timeouts, temporary resource constraints, transient service unavailability. These strategies must be designed conservatively to avoid making problems worse through overly aggressive recovery attempts.

**Retry Logic with Exponential Backoff**: The retry system implements intelligent backoff strategies that balance rapid recovery with system stability. Simple fixed-interval retries can overwhelm struggling services, while exponential backoff gives services time to recover while providing rapid retry for truly transient issues.

| Failure Type | Initial Retry Delay | Backoff Multiplier | Max Retries | Max Delay | Circuit Breaker |
|-------------|-------------------|-------------------|-------------|-----------|-----------------|
| **Network Timeout** | 1 second | 2x | 5 retries | 32 seconds | After 10 failures |
| **Resource Unavailable** | 5 seconds | 1.5x | 3 retries | 17 seconds | After 5 failures |
| **Service Temporarily Down** | 10 seconds | 2x | 4 retries | 80 seconds | After 8 failures |
| **Rate Limit Exceeded** | 60 seconds | 1.2x | 6 retries | 120 seconds | After 3 failures |

The **exponential backoff algorithm** with jitter prevents the "thundering herd" problem where many clients retry simultaneously:

1. Calculate base delay: `initial_delay * (backoff_multiplier ^ attempt_number)`
2. Add randomized jitter: `base_delay * (0.5 + random(0, 0.5))`
3. Cap at maximum delay: `min(calculated_delay, max_delay)`
4. Execute delay before retry attempt
5. Track consecutive failures for circuit breaker evaluation

**Circuit Breaker Pattern**: Circuit breakers prevent cascading failures by temporarily disabling retry attempts when services are clearly unavailable, allowing failing services time to recover without additional load.

| Circuit State | Trigger Condition | Behavior | State Transition |
|--------------|------------------|----------|------------------|
| **CLOSED** | Normal operation | All requests pass through | Open after failure threshold |
| **OPEN** | Failure rate exceeded | All requests fail fast | Half-open after timeout |
| **HALF_OPEN** | Testing recovery | Limited requests allowed | Close on success, open on failure |

**Resource Cleanup and Recovery**: Resource exhaustion failures require active cleanup procedures that reclaim system resources while preserving critical data and ongoing operations.

The **graduated resource cleanup strategy** implements increasingly aggressive cleanup levels based on resource pressure:

1. **Level 1 - Soft Cleanup**: Remove expired cache entries, compress old logs, cleanup temporary files
2. **Level 2 - Aggressive Cleanup**: Cancel queued non-critical jobs, terminate idle containers
3. **Level 3 - Emergency Cleanup**: Pause new job starts, terminate long-running jobs with warnings
4. **Level 4 - System Protection**: Force-kill resource-intensive processes, emergency shutdown procedures

**Container Recovery Procedures**: Container execution failures require careful recovery that preserves job state while providing clean execution environments for retry attempts.

| Container Failure Scenario | Recovery Procedure | State Preservation | Retry Conditions |
|---------------------------|------------------|------------------|------------------|
| **Exit Code > 0** | Clean container restart | Preserve workspace volume | Retry count not exceeded |
| **OOM Killed** | Restart with increased memory | Preserve workspace volume | Memory limit increase possible |
| **Container Hang** | Force kill and restart | Lose in-memory state | Timeout threshold exceeded |
| **Image Pull Failure** | Retry pull, fallback to cached | No state affected | Network connectivity available |
| **Docker Daemon Down** | Queue job for later execution | Preserve all job state | Docker daemon healthy |

#### Manual Intervention Procedures

Manual intervention procedures provide the **human expertise layer** for complex failures that require business judgment, domain knowledge, or coordination across multiple systems. These procedures must be clearly documented with step-by-step instructions that can be followed under stress during production incidents.

**Escalation Triggers and Procedures**: The escalation system automatically transitions from automated recovery to manual procedures based on failure persistence, impact scope, and recovery success rates.

| Escalation Level | Trigger Conditions | Response Team | Maximum Response Time |
|-----------------|------------------|---------------|----------------------|
| **Level 1 - Automated** | Single component failure, retries available | Automated systems only | Immediate |
| **Level 2 - On-Call** | Multiple retry failures, service degradation | On-call engineer | 15 minutes |
| **Level 3 - Team Lead** | Multiple component failures, user impact | Team lead + on-call | 30 minutes |
| **Level 4 - Management** | System-wide outage, business impact | Engineering management | 1 hour |

**Incident Response Runbooks**: Runbooks provide step-by-step procedures for common failure scenarios that require manual intervention. Each runbook includes diagnosis steps, recovery procedures, and rollback plans.

**Pipeline Recovery Runbook Example - Database Connection Failure**:

1. **Immediate Assessment** (2 minutes):
   - Check database health monitoring dashboard
   - Verify network connectivity from pipeline servers  
   - Review recent configuration changes in change log
   - Assess current job queue backup and user impact

2. **Diagnosis Phase** (5 minutes):
   - Test direct database connection from pipeline servers
   - Check database server resource utilization (CPU, memory, connections)
   - Review database server logs for error patterns
   - Validate connection pool configuration matches database limits

3. **Recovery Attempts** (10 minutes):
   - Restart connection pool to clear corrupted connections
   - If network issue detected, coordinate with infrastructure team
   - If database overload detected, implement read-only mode for non-critical operations
   - Scale database resources if capacity issue confirmed

4. **Rollback Plan** (if recovery fails):
   - Enable maintenance mode to prevent new pipeline starts
   - Preserve current job queue state for later processing
   - Communicate status to development teams with estimated recovery time
   - Escalate to Level 3 if recovery not achieved within 30 minutes

**Data Recovery Procedures**: Data corruption or loss requires careful recovery procedures that balance speed with data integrity. Recovery procedures must account for both artifact corruption and pipeline state corruption.

| Data Recovery Scenario | Recovery Procedure | Data Loss Risk | Recovery Time Estimate |
|------------------------|------------------|----------------|------------------------|
| **Artifact Corruption** | Restore from backup, rebuild if necessary | Minimal - artifacts rebuildable | 15-60 minutes |
| **Pipeline State Loss** | Restart affected pipelines from beginning | Moderate - work in progress lost | 30-120 minutes |
| **Configuration Corruption** | Restore from git configuration repository | Minimal - configuration versioned | 5-15 minutes |
| **Database Corruption** | Restore from database backup | High - recent pipeline history lost | 60-240 minutes |

**Emergency Shutdown Procedures**: System-wide failures may require coordinated shutdown to prevent data corruption or cascading failures across dependent systems.

**Emergency Shutdown Sequence**:
1. **Graceful Phase** (5 minutes): Stop accepting new pipelines, allow current jobs to complete
2. **Warning Phase** (2 minutes): Send termination warnings to running jobs, begin cleanup
3. **Forced Phase** (1 minute): Force-kill remaining processes, flush logs and state
4. **Verification Phase**: Confirm all processes terminated, storage systems consistent
5. **Communication**: Notify stakeholders of shutdown completion and estimated recovery time

#### State Recovery and Consistency

State recovery ensures that pipeline execution can resume correctly after failures without losing work or creating inconsistent system state. The state management system implements **write-ahead logging** patterns that record intended operations before executing them, enabling recovery by replaying the operation log.

**Pipeline Execution State Recovery**: Pipeline state includes job execution status, dependency relationships, artifact references, and resource allocations. State recovery must handle scenarios where failures occur at any point during pipeline execution.

| Pipeline State Component | Recovery Method | Consistency Guarantee | Recovery Time |
|-------------------------|----------------|----------------------|---------------|
| **Job Execution Status** | Write-ahead log replay | Exactly-once execution | 1-5 minutes |
| **Artifact References** | Content-addressable verification | Integrity validated | 5-15 minutes |
| **Resource Allocations** | Idempotent resource cleanup | No resource leaks | 2-10 minutes |
| **Dependency Graph** | Rebuild from pipeline definition | Mathematically consistent | <1 minute |

**Job State Machine Recovery**: Each job follows a state machine with defined transitions. Recovery procedures must ensure jobs resume in consistent states without skipping required steps or repeating completed work.

The **job state recovery algorithm** works as follows:
1. Read job state from persistent storage (last known state)
2. Verify state consistency with actual system resources (containers, files)
3. If inconsistency detected, determine safe recovery state
4. If job was `RUNNING`, check for container existence and health
5. If container missing but job marked running, transition to `FAILED` with recovery attempt
6. If container exists and healthy, allow job to continue
7. Update job state log with recovery action taken

**Artifact Consistency Recovery**: Artifact consistency failures can break pipeline workflows by providing corrupted or missing dependencies to downstream jobs. Recovery procedures must verify artifact integrity and rebuild missing or corrupted artifacts when possible.

**Artifact Recovery Decision Matrix**:

| Artifact State | Integrity Status | Recovery Action | Fallback Strategy |
|---------------|-----------------|----------------|-------------------|
| **Present + Valid** | Checksum verified | No action required | N/A |
| **Present + Corrupted** | Checksum mismatch | Delete and rebuild | Use cached backup if available |
| **Missing + Rebuildable** | Not found | Restart producer job | Use last known good version |
| **Missing + Critical** | Not found | Manual restoration required | Escalate to manual recovery |

The consistency recovery system implements **eventual consistency** guarantees where temporary inconsistencies are acceptable as long as the system converges to a consistent state within bounded time periods.

### Implementation Guidance

The error handling implementation requires sophisticated coordination between detection, recovery, and escalation systems. The architecture emphasizes **observable failure modes** where every failure provides sufficient context for automated classification and recovery decisions.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|--------------|-----------------|
| **Error Classification** | Python enum with basic categorization | Hierarchical exception system with metadata |
| **Retry Logic** | Simple backoff with max attempts | Circuit breaker with statistics tracking |
| **Health Checks** | HTTP endpoint polling | Multi-dimensional health with metrics |
| **Log Correlation** | Correlation ID in log messages | Distributed tracing with OpenTelemetry |
| **Metrics Collection** | Basic counters and gauges | Time-series database with alerting rules |
| **State Recovery** | File-based state persistence | Write-ahead log with transaction support |

#### Recommended File Structure

```
pipeline-system/
├── error_handling/
│   ├── __init__.py
│   ├── exceptions.py          ← Custom exception hierarchy
│   ├── failure_detector.py    ← Multi-layer detection system
│   ├── recovery_manager.py    ← Automated recovery coordination
│   ├── circuit_breaker.py     ← Circuit breaker implementation
│   ├── retry_policies.py      ← Retry strategy implementations
│   ├── health_checks.py       ← Component health monitoring
│   ├── metrics_collector.py   ← System metrics and alerting
│   └── incident_manager.py    ← Manual escalation procedures
├── monitoring/
│   ├── log_correlation.py     ← Cross-component log analysis
│   ├── anomaly_detection.py   ← Pattern recognition system
│   └── alerting_rules.py      ← Metric-based alert conditions
└── recovery/
    ├── state_manager.py       ← Pipeline state persistence
    ├── artifact_recovery.py   ← Artifact consistency recovery
    └── emergency_procedures.py ← Runbook automation
```

#### Infrastructure Starter Code

**Exception Hierarchy with Metadata**:

```python
from enum import Enum
from datetime import datetime
from typing import Dict, Any, Optional, List
import traceback

class FailureCategory(Enum):
    TRANSIENT = "transient"
    RESOURCE = "resource" 
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    CORRUPTION = "corruption"
    CATASTROPHIC = "catastrophic"

class PipelineError(Exception):
    """Base exception for all pipeline errors with rich metadata."""
    
    def __init__(
        self, 
        message: str, 
        category: FailureCategory,
        component: str,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        recovery_hint: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.component = component
        self.correlation_id = correlation_id
        self.metadata = metadata or {}
        self.recovery_hint = recovery_hint
        self.timestamp = datetime.utcnow()
        self.stack_trace = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to structured format for logging/alerting."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "component": self.component,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
            "recovery_hint": self.recovery_hint,
            "timestamp": self.timestamp.isoformat(),
            "stack_trace": self.stack_trace
        }

class ValidationError(PipelineError):
    """Pipeline definition validation failures."""
    
    def __init__(self, message: str, field_path: str, **kwargs):
        super().__init__(
            message, 
            FailureCategory.CONFIGURATION, 
            "pipeline-parser",
            **kwargs
        )
        self.field_path = field_path

class CircularDependencyError(PipelineError):
    """Circular dependency in job graph."""
    
    def __init__(self, cycle_path: List[str], **kwargs):
        message = f"Circular dependency detected: {' → '.join(cycle_path)}"
        super().__init__(
            message,
            FailureCategory.CONFIGURATION,
            "pipeline-parser", 
            **kwargs
        )
        self.cycle_path = cycle_path

class ArtifactManagerError(PipelineError):
    """Base class for artifact management errors."""
    
    def __init__(self, message: str, category: FailureCategory, **kwargs):
        super().__init__(message, category, "artifact-manager", **kwargs)

class IntegrityVerificationError(ArtifactManagerError):
    """Artifact integrity verification failed."""
    
    def __init__(self, content_hash: str, expected_hash: str, **kwargs):
        message = f"Integrity check failed for {content_hash[:16]}... (expected {expected_hash[:16]}...)"
        super().__init__(
            message, 
            FailureCategory.CORRUPTION,
            recovery_hint="Delete corrupted artifact and rebuild from source",
            **kwargs
        )
        self.content_hash = content_hash
        self.expected_hash = expected_hash
```

**Circuit Breaker Implementation**:

```python
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Any, Dict
import threading
import time
import logging

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open" 
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: int = 60
    expected_exception: type = Exception

class CircuitBreaker:
    """Circuit breaker implementation with automatic state transitions."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.lock = threading.Lock()
        self.logger = logging.getLogger(f"circuit_breaker.{name}")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker protection."""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    raise PipelineError(
                        f"Circuit breaker {self.name} is OPEN",
                        FailureCategory.DEPENDENCY,
                        "circuit-breaker",
                        recovery_hint=f"Wait {self.config.timeout_seconds}s for automatic reset"
                    )
            
            # Execute the protected function
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.config.expected_exception as e:
                self._on_failure()
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        elapsed = datetime.utcnow() - self.last_failure_time
        return elapsed > timedelta(seconds=self.config.timeout_seconds)
    
    def _transition_to_half_open(self):
        """Transition to half-open state for testing recovery."""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.logger.info(f"Circuit breaker {self.name} transitioned to HALF_OPEN")
    
    def _on_success(self):
        """Handle successful function execution."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0  # Reset failure count on success
    
    def _on_failure(self):
        """Handle failed function execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._transition_to_open()
        elif self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
    
    def _transition_to_closed(self):
        """Transition to closed (normal) state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.logger.info(f"Circuit breaker {self.name} transitioned to CLOSED")
    
    def _transition_to_open(self):
        """Transition to open (failure) state."""
        self.state = CircuitState.OPEN
        self.logger.warning(f"Circuit breaker {self.name} transitioned to OPEN")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None
        }
```

#### Core Logic Skeleton Code

**Failure Detection and Recovery Manager**:

```python
class FailureDetector:
    """Multi-layer failure detection with automatic classification."""
    
    def __init__(self, health_checkers: Dict[str, 'HealthChecker']):
        # TODO 1: Initialize health checkers for each component
        # TODO 2: Set up metrics collection for failure rate tracking
        # TODO 3: Configure log pattern matchers for known error signatures
        # TODO 4: Initialize circuit breakers for external dependencies
        pass
    
    def detect_failures(self) -> List[Dict[str, Any]]:
        """Detect current system failures across all components."""
        # TODO 1: Run health checks on all registered components
        # TODO 2: Analyze recent metrics for anomaly patterns
        # TODO 3: Scan logs for error signatures and correlation patterns
        # TODO 4: Check circuit breaker states for dependency failures
        # TODO 5: Classify detected failures by category and severity
        # TODO 6: Generate failure reports with recovery recommendations
        # Hint: Use parallel execution for health checks to reduce detection time
        pass

class RecoveryManager:
    """Coordinates automated recovery and manual escalation procedures."""
    
    def __init__(self, retry_policies: Dict[FailureCategory, 'RetryPolicy']):
        # TODO 1: Configure retry policies for each failure category
        # TODO 2: Initialize state recovery managers for pipeline components
        # TODO 3: Set up escalation rules and notification channels
        # TODO 4: Load emergency runbook procedures from configuration
        pass
    
    def handle_failure(self, failure_report: Dict[str, Any]) -> bool:
        """Execute appropriate recovery strategy for detected failure."""
        # TODO 1: Extract failure category and component from report
        # TODO 2: Check if automatic recovery is appropriate for this failure type
        # TODO 3: If automatic recovery eligible, execute retry with appropriate policy
        # TODO 4: If recovery succeeds, update metrics and log success
        # TODO 5: If recovery fails or not eligible, escalate to manual procedures
        # TODO 6: Track recovery attempts and success rates for optimization
        # Hint: Use correlation_id to track recovery across multiple attempts
        return False  # Placeholder
    
    def escalate_to_manual(self, failure_report: Dict[str, Any]):
        """Escalate failure to manual intervention procedures."""
        # TODO 1: Determine escalation level based on failure impact and persistence
        # TODO 2: Generate comprehensive incident report with diagnosis information
        # TODO 3: Notify appropriate response team based on escalation level
        # TODO 4: Create incident tracking record for resolution monitoring
        # TODO 5: Execute immediate containment procedures to prevent spread
        # TODO 6: Provide runbook guidance for manual recovery procedures
        pass

class ComponentHealthMonitor:
    """Monitors individual component health with multi-dimensional checks."""
    
    def check_component_health(self, component_name: str) -> Dict[str, Any]:
        """Execute comprehensive health check for specified component."""
        # TODO 1: Run functional health check (API endpoints, basic operations)
        # TODO 2: Check performance health (response times, throughput)
        # TODO 3: Verify resource health (CPU, memory, disk usage)
        # TODO 4: Test dependency health (external services, database connections)
        # TODO 5: Calculate overall health score from individual dimension scores
        # TODO 6: Generate health report with specific failure details if unhealthy
        # Hint: Use timeout and circuit breaker for external dependency checks
        pass
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health across all monitored components."""
        # TODO 1: Collect health status from all registered components
        # TODO 2: Calculate system-wide health metrics and trends
        # TODO 3: Identify components with degraded performance
        # TODO 4: Check for systemic issues affecting multiple components
        # TODO 5: Generate system health dashboard data
        # TODO 6: Include capacity utilization and growth trend analysis
        pass
```

#### Milestone Checkpoints

**Error Handling Integration Test**:
- **Command**: `python -m pytest tests/error_handling/ -v --tb=short`
- **Expected Output**: All failure scenarios tested with appropriate recovery actions
- **Manual Verification**: Trigger each failure type and verify detection + recovery
- **Success Indicators**: 
  - Circuit breakers trigger correctly under load
  - Retry policies respect backoff timing
  - Health checks detect component degradation
  - Manual escalation procedures execute with proper notification

**Failure Injection Testing**:
- **Command**: `python scripts/chaos_testing.py --scenario network_partition --duration 5m`
- **Expected Behavior**: System detects partition, switches to degraded mode, recovers automatically
- **Recovery Verification**: All components return to healthy state within recovery SLA
- **Alert Validation**: Appropriate alerts fired and cleared during test sequence


## Testing Strategy

> **Milestone(s):** This section provides comprehensive testing approaches and validation checkpoints across all milestones (1-4), ensuring each component works correctly in isolation and as part of the complete CI/CD pipeline system.

Think of the testing strategy as a **quality assurance assembly line** that runs parallel to our CI/CD pipeline development. Just as an automobile factory has quality checkpoints at every stage—from individual parts inspection to final road tests—our CI/CD pipeline needs multiple layers of validation to ensure reliability. Each component must work flawlessly in isolation before we can trust the entire system to handle production deployments.

The testing strategy operates on three distinct levels, each serving a different purpose. Unit testing validates individual components like ensuring each gear works perfectly before installing it in the transmission. Integration testing verifies that components communicate correctly, similar to testing that the engine, transmission, and wheels work together smoothly. Milestone checkpoints provide concrete validation criteria that confirm we've successfully implemented each major capability, like test-driving the completed vehicle to verify it meets all safety and performance requirements.

A comprehensive testing approach becomes critical in CI/CD systems because failures can cascade through the entire software delivery pipeline. A bug in pipeline parsing might not surface until hours later during deployment, making root cause analysis extremely difficult. By implementing thorough testing at each level, we catch problems early when they're easier to diagnose and fix, rather than dealing with mysterious production failures that require forensic debugging.

### Unit Testing

Unit testing for CI/CD pipeline components focuses on validating individual functions and methods in complete isolation from external dependencies. Think of unit testing as **quality control for individual machine parts**—each component must meet exact specifications before assembly into the larger system. This isolation is achieved through dependency injection and mocking, allowing us to test complex logic without requiring Docker containers, file systems, or network connections.

The unit testing strategy emphasizes **behavioral verification** over implementation details. Instead of testing internal data structures, we focus on verifying that each component produces correct outputs given specific inputs and maintains proper state transitions. This approach ensures tests remain valuable even when internal implementation changes, while catching regressions that affect external behavior.

#### Pipeline Parser Unit Tests

The `PipelineDefinition` parser requires comprehensive unit testing to handle the complexity of YAML processing, dependency validation, and variable substitution. These tests must verify both successful parsing scenarios and comprehensive error handling with descriptive error messages.

| Test Category | Test Case | Input | Expected Output | Purpose |
|---------------|-----------|--------|-----------------|---------|
| Basic Parsing | Valid minimal pipeline | `name: test, jobs: {build: {steps: [{name: compile, script: [make]}]}}` | `PipelineDefinition` with correct structure | Verify basic YAML parsing |
| Complex Dependencies | Multi-job with dependencies | Jobs with `depends_on` relationships | Correct dependency graph with topological ordering | Validate dependency resolution |
| Variable Resolution | Pipeline with variables | YAML with `${ENV_VAR}` and `${job.artifact}` references | Resolved text with substituted values | Test variable substitution engine |
| Circular Dependencies | Invalid dependency cycle | Job A depends on B, B depends on A | `CircularDependencyError` with cycle path | Detect and report cycles |
| Missing Dependencies | Reference to non-existent job | Job depends on `non-existent-job` | `ValidationError` with specific missing dependency | Validate dependency existence |
| Malformed YAML | Syntax errors | Invalid YAML syntax | `PipelineParserError` with line/column info | Handle parsing failures gracefully |
| Invalid Job Schema | Missing required fields | Job without required `steps` field | `ValidationError` with field path | Enforce schema validation |
| Variable Resolution Errors | Undefined variable reference | Reference to `${UNDEFINED_VAR}` | `VariableResolutionError` with variable name | Handle missing variables |

The dependency graph construction testing requires careful validation of edge cases that could cause runtime failures. Tests must verify that the topological sort produces a valid execution order even with complex dependency patterns, and that parallel execution groups are correctly identified.

```markdown
**Dependency Graph Test Matrix:**

| Graph Structure | Test Scenario | Expected Parallel Groups | Validation Focus |
|-----------------|---------------|-------------------------|------------------|
| Linear Chain | A → B → C → D | [[A], [B], [C], [D]] | Sequential execution |
| Diamond Pattern | A → B,C → D | [[A], [B,C], [D]] | Parallel convergence |
| Multiple Roots | A,B → C,D → E | [[A,B], [C,D], [E]] | Multiple starting points |
| Complex DAG | Mixed dependencies | Correct parallel groups | Real-world complexity |
| Single Job | No dependencies | [[single-job]] | Minimal case |
```

Variable substitution testing must handle the complexity of nested variable references, environment variable precedence, and expression evaluation. The test cases should cover all variable scoping rules and ensure that resolution errors provide sufficient context for debugging.

| Variable Type | Test Input | Context Variables | Expected Result | Error Case |
|---------------|------------|------------------|-----------------|------------|
| Simple Environment | `${HOME}` | `HOME=/home/user` | `/home/user` | `HOME` undefined |
| Job Reference | `${build.artifact}` | `build.artifact=app.jar` | `app.jar` | Job not found |
| Nested Reference | `${base}/${file}` | `base=/tmp, file=test.log` | `/tmp/test.log` | Either undefined |
| Pipeline Parameter | `${pipeline.version}` | `version=1.2.3` | `1.2.3` | Parameter missing |
| Complex Expression | `${env}-${version}` | `env=prod, version=1.0` | `prod-1.0` | Partial resolution |

#### Job Executor Unit Tests

The `PipelineJob` executor testing focuses on isolation from Docker runtime while validating the complex state machine transitions and retry logic. Mock implementations of `DockerClient` allow testing without actual containers while preserving the behavioral contracts.

The executor testing strategy emphasizes **state transition validation** by tracking job status changes and ensuring proper event sequences. Each test verifies that `JobStatus` transitions follow the defined state machine and that appropriate metadata (timestamps, logs, retry counts) gets recorded correctly.

| Test Focus | Mock Components | Test Scenarios | Validation Points |
|------------|-----------------|----------------|-------------------|
| Step Execution | `DockerClient.run_command` | Command success/failure | Exit codes, output capture |
| Retry Logic | Step failures with different exit codes | Exponential backoff timing | Retry count, delay calculation |
| Timeout Handling | Long-running commands | Process termination | Cleanup, timeout detection |
| State Transitions | Job lifecycle events | Status progression | State consistency, timestamps |
| Environment Setup | Variable injection | Env var propagation | Variable availability in steps |
| Error Propagation | Various failure modes | Exception handling | Error categorization, recovery hints |

The retry policy testing requires careful validation of the `ExponentialBackoffRetry` implementation to ensure it handles edge cases like maximum retry limits and backoff cap values correctly.

```markdown
**Retry Policy Test Matrix:**

| Failure Type | Exit Code | Attempt Number | Expected Delay | Should Retry |
|--------------|-----------|----------------|----------------|--------------|
| Transient Network | 1 | 1 | 1 second | Yes |
| Transient Network | 1 | 2 | 2 seconds | Yes |
| Transient Network | 1 | 3 | 4 seconds | Yes |
| Configuration Error | 127 | 1 | N/A | No |
| Permission Denied | 126 | 1 | N/A | No |
| Max Retries Reached | 1 | 5 | N/A | No |
```

#### Artifact Manager Unit Tests

The `ArtifactInfo` and storage backend testing requires careful handling of file system operations without creating actual files. Mock file system implementations allow testing content-addressable storage logic, integrity verification, and retention policies in isolation.

Artifact management testing emphasizes **data integrity validation** throughout the storage lifecycle. Tests must verify that content hashes are calculated correctly, that retrieval produces identical content to what was stored, and that retention policies operate safely without data loss.

| Test Category | Mock Components | Test Data | Validation Focus |
|---------------|-----------------|-----------|------------------|
| Content Addressing | File system operations | Various file sizes/types | Hash calculation accuracy |
| Integrity Verification | Checksum calculations | Corrupted file scenarios | Corruption detection |
| Retention Policies | Storage statistics | Various artifact ages | Cleanup decision logic |
| Metadata Management | Artifact database | Reference counting | Metadata consistency |
| Streaming Operations | Large file handling | Multi-GB test data | Memory usage efficiency |
| Concurrent Access | Multiple upload/download | Race conditions | Thread safety |

The retention policy testing requires simulation of realistic storage conditions with varying artifact ages, access patterns, and storage pressure scenarios.

```markdown
**Retention Policy Test Scenarios:**

| Policy Rule | Artifacts | Storage Pressure | Expected Action |
|-------------|-----------|------------------|-----------------|
| Age > 30 days | 45-day-old artifacts | Normal | Delete eligible |
| References = 0 | Unused artifacts | High | Aggressive cleanup |
| Size > 1GB | Large artifacts | Critical | Priority deletion |
| Pattern match | `*.tmp` files | Any | Pattern-based cleanup |
```

#### Deployment Strategy Unit Tests

The deployment strategy testing isolates the decision-making logic from actual load balancer interactions and health check execution. Mock implementations of traffic routing and health monitoring allow testing complex deployment orchestration without infrastructure dependencies.

Deployment testing focuses on **strategy validation** by verifying that each deployment pattern (rolling, blue-green, canary) follows its defined algorithm and responds correctly to health check failures. The tests must validate decision points where deployments pause, rollback, or continue based on health metrics.

| Strategy Type | Mock Components | Failure Scenarios | Decision Validation |
|---------------|-----------------|-------------------|---------------------|
| Rolling Deployment | Health checks, instance management | Individual instance failures | Rollback timing |
| Blue-Green | Traffic routing, environment management | Environment preparation failures | Atomic switching |
| Canary Deployment | Metrics collection, traffic splitting | Performance degradation | Traffic rollback |
| Health Validation | Service endpoints | Various failure modes | Retry vs abort decisions |

### Integration Testing

Integration testing validates that CI/CD pipeline components communicate correctly and handle realistic workflows end-to-end. Think of integration testing as **systems engineering validation**—verifying that individually perfect components work together seamlessly when combined into the complete system. Unlike unit tests, integration tests use real dependencies like Docker containers, file systems, and network communications to expose interface mismatches and timing issues.

The integration testing strategy focuses on **realistic workflow simulation** using representative pipeline definitions, actual build artifacts, and real deployment scenarios. These tests execute complete user workflows from YAML parsing through deployment, validating that data flows correctly between components and that error conditions propagate appropriately through the system.

Integration tests operate with **controlled infrastructure** that provides predictable behavior while exercising real system components. This includes containerized test services, temporary file systems, and mock deployment targets that respond realistically to health checks and traffic routing commands.

#### Component Communication Testing

The integration test harness validates communication between the four major components by executing realistic pipeline scenarios that exercise all interaction patterns. These tests use `IntegrationTestHarness` to provide controlled infrastructure while maintaining realistic component behavior.

| Integration Test | Components Involved | Data Flow | Success Criteria |
|------------------|--------------------|-----------|-----------------| 
| Parse and Execute | Parser → Executor | `PipelineDefinition` → Job execution | Jobs execute in dependency order |
| Artifact Transfer | Executor → Artifact Manager → Executor | Build outputs → Storage → Download | Artifacts available to dependent jobs |
| Deploy Pipeline | All four components | Complete workflow | Successful deployment execution |
| Error Propagation | All components | Failure scenarios | Proper error handling and reporting |
| Parallel Execution | Parser → Executor | Concurrent job execution | Parallel jobs complete successfully |

The component communication tests verify that `EventDispatcher` correctly routes messages between components and that event ordering remains consistent even under high load. These tests stress-test the event system with rapid pipeline executions and verify that component health monitoring accurately reflects system state.

```markdown
**Event Communication Test Matrix:**

| Event Type | Producer | Consumer | Timing Constraint | Failure Mode |
|------------|----------|----------|-------------------|--------------|
| Job Started | Job Executor | Flow Coordinator | < 100ms | Event loss |
| Artifact Uploaded | Artifact Manager | Dependent Jobs | < 500ms | Missing notification |
| Health Check Failed | Deployment Engine | Recovery Manager | < 1s | Delayed response |
| Pipeline Completed | Flow Coordinator | All components | < 200ms | Incomplete cleanup |
```

#### End-to-End Workflow Testing

End-to-end testing executes complete pipeline workflows using realistic YAML definitions and validates that the entire system produces correct outcomes. These tests use actual Docker containers for job execution, real file system storage for artifacts, and mock deployment targets that simulate production infrastructure behavior.

The workflow testing strategy covers representative pipeline patterns commonly found in real CI/CD systems, including complex dependency graphs, large artifact transfers, and various deployment strategies. Each test validates both successful execution paths and error recovery scenarios.

| Workflow Type | Pipeline Structure | Artifacts | Deployment | Success Criteria |
|---------------|-------------------|-----------|------------|------------------|
| Simple Build | Linear job chain | Small binary | Rolling deployment | Complete deployment |
| Complex Build | Diamond dependency | Multiple artifacts | Blue-green deployment | Zero-downtime switch |
| Microservice | Parallel services | Container images | Canary deployment | Gradual traffic shift |
| Library Build | Test matrix | Documentation + packages | No deployment | Artifact publication |
| Monorepo | Conditional execution | Selective artifacts | Multi-target deployment | Correct target selection |

The end-to-end tests validate **cross-component data consistency** by verifying that artifact references, job dependencies, and deployment states remain coherent throughout pipeline execution. These tests specifically look for race conditions and timing issues that only appear when components interact under realistic load.

#### Real Infrastructure Testing

Integration testing includes scenarios that use actual infrastructure components to validate system behavior under realistic conditions. These tests execute against real Docker daemon, actual file systems, and simulated deployment targets that respond with realistic timing and failure patterns.

Real infrastructure testing focuses on **operational validation** by exercising components under conditions that closely match production usage. This includes testing with large artifacts that stress storage systems, long-running jobs that test timeout handling, and deployment scenarios that exercise health check logic with realistic response times.

| Infrastructure Component | Test Scenarios | Stress Conditions | Validation Focus |
|-------------------------|-----------------|-------------------|------------------|
| Docker Integration | Container lifecycle | Image pulls, resource limits | Process isolation, cleanup |
| File System Storage | Large artifact handling | Disk space pressure | Storage efficiency, cleanup |
| Network Communication | Service health checks | Network latency, failures | Timeout handling, retries |
| Process Management | Long-running jobs | Resource exhaustion | Resource cleanup, monitoring |

The infrastructure testing validates **failure mode handling** by intentionally triggering realistic failure conditions and verifying that the system detects problems quickly and recovers gracefully. These tests ensure that the CI/CD pipeline remains robust when deployed in real environments with imperfect infrastructure.

> **Decision: Real vs Mock Infrastructure for Integration Testing**
> - **Context**: Integration tests need to validate component interactions while maintaining test reliability and execution speed
> - **Options Considered**: Pure mocking, real infrastructure, hybrid approach
> - **Decision**: Hybrid approach with real Docker/filesystem and mocked deployment targets
> - **Rationale**: Real Docker/filesystem tests expose actual integration issues while mocked deployments provide controlled test conditions and avoid infrastructure requirements
> - **Consequences**: Tests catch real-world problems while maintaining execution speed and reliability

### Milestone Checkpoints

Milestone checkpoints provide concrete validation criteria that confirm successful implementation of each major capability. Think of checkpoints as **quality gates** that must be passed before proceeding to the next development phase—similar to safety inspections during construction that verify structural integrity before adding the next floor.

Each checkpoint defines specific behavior that must be demonstrated, including both programmatic tests that can be automated and manual verification steps that confirm the system works as intended. The checkpoints progressively build complexity, ensuring that earlier capabilities remain functional as new features are added.

#### Milestone 1: Pipeline Definition Parser Checkpoint

The parser checkpoint validates that YAML pipeline definitions are correctly transformed into executable dependency graphs with proper error handling and variable resolution. This checkpoint establishes the foundation that all subsequent milestones depend upon.

**Automated Test Validation:**
1. Execute the parser test suite with `python -m pytest tests/test_pipeline_parser.py -v`
2. Verify all dependency graph tests pass, including circular dependency detection
3. Run variable substitution tests with various environment configurations
4. Validate error handling tests produce descriptive error messages

**Manual Verification Steps:**
1. Create a sample pipeline YAML with complex dependencies and variables
2. Parse the pipeline and inspect the generated dependency graph structure
3. Verify that `get_execution_order()` returns correct parallel execution groups
4. Test error scenarios with malformed YAML and verify helpful error messages

| Verification Point | Command/Action | Expected Result | Troubleshooting |
|--------------------|----------------|-----------------|-----------------|
| Basic Parsing | `parser.parse_yaml(simple_pipeline.yaml)` | Valid `PipelineDefinition` object | Check YAML syntax, schema validation |
| Dependency Resolution | `parser.build_dependency_graph()` | DAG with correct topological order | Verify job names match, no typos |
| Variable Substitution | Parse pipeline with `${HOME}` reference | Resolved to actual home directory | Check environment variable availability |
| Error Handling | Parse invalid YAML | Descriptive `ValidationError` | Ensure error includes line numbers |

**Success Criteria Checklist:**
- [ ] Parser handles valid YAML files without errors
- [ ] Dependency graph construction completes successfully
- [ ] Circular dependency detection works and reports cycle paths
- [ ] Variable substitution resolves all references correctly
- [ ] Error messages are specific and include context information
- [ ] Complex dependency patterns produce correct execution order

#### Milestone 2: Job Executor Checkpoint

The executor checkpoint validates that pipeline jobs execute correctly in isolated containers with proper logging, timeout handling, and state management. This checkpoint builds upon the parser foundation and enables reliable job execution.

**Automated Test Validation:**
1. Run executor tests with `python -m pytest tests/test_job_executor.py -v`
2. Execute container integration tests that use real Docker daemon
3. Validate retry policy tests with simulated failures
4. Test timeout handling with long-running mock commands

**Manual Verification Steps:**
1. Execute a simple job that runs a shell command in a container
2. Monitor real-time log output and verify it captures stdout/stderr
3. Test job cancellation and verify proper cleanup
4. Validate retry behavior by intentionally failing a job step

| Verification Point | Command/Action | Expected Result | Troubleshooting |
|--------------------|----------------|-----------------|-----------------|
| Container Execution | Execute job with `echo "Hello World"` | Output captured in logs | Check Docker daemon status |
| Environment Variables | Job with `echo $TEST_VAR` | Variable value in output | Verify variable injection |
| Timeout Handling | Job with `sleep 10` and 5s timeout | Job terminated after 5 seconds | Check timeout implementation |
| Retry Logic | Job that fails twice then succeeds | 3 execution attempts total | Verify retry policy configuration |
| State Transitions | Monitor job status during execution | Correct state progression | Check state machine logic |

**Success Criteria Checklist:**
- [ ] Jobs execute successfully in Docker containers
- [ ] Real-time log streaming works correctly
- [ ] Timeout handling terminates jobs and cleans up containers
- [ ] Retry logic follows exponential backoff policy
- [ ] Job state transitions follow defined state machine
- [ ] Environment variable injection works for all job steps

#### Milestone 3: Artifact Management Checkpoint

The artifact management checkpoint validates that build outputs are correctly stored, retrieved, and managed with integrity verification and retention policies. This checkpoint enables data flow between pipeline stages.

**Automated Test Validation:**
1. Run artifact tests with `python -m pytest tests/test_artifact_manager.py -v`
2. Execute storage integration tests with real file system
3. Validate integrity verification with checksum calculations
4. Test retention policy execution with simulated storage pressure

**Manual Verification Steps:**
1. Upload a test artifact and verify it's stored with correct content hash
2. Retrieve the artifact from storage and verify content integrity
3. Test retention policy execution and verify correct artifacts are cleaned up
4. Validate artifact metadata tracking and reference counting

| Verification Point | Command/Action | Expected Result | Troubleshooting |
|--------------------|----------------|-----------------|-----------------|
| Artifact Upload | Store test file via `store_artifact()` | Returns content-addressable hash | Check storage directory permissions |
| Artifact Download | Retrieve using returned hash | Identical file content | Verify hash calculation logic |
| Integrity Verification | Verify artifact with known checksum | `VerificationResult.VALID` | Check checksum algorithm |
| Retention Policy | Execute cleanup with test artifacts | Correct artifacts deleted | Verify policy rule evaluation |
| Large File Handling | Store/retrieve 100MB file | Successful without memory issues | Check streaming implementation |

**Success Criteria Checklist:**
- [ ] Artifacts are stored with content-addressable hashing
- [ ] Retrieved artifacts are identical to uploaded content
- [ ] Integrity verification detects corruption correctly
- [ ] Retention policies execute safely without data loss
- [ ] Large artifacts are handled efficiently with streaming
- [ ] Metadata tracking maintains accurate reference counts

#### Milestone 4: Deployment Strategies Checkpoint

The deployment strategy checkpoint validates that rolling, blue-green, and canary deployments execute correctly with health checks and automatic rollback capabilities. This checkpoint completes the full CI/CD pipeline functionality.

**Automated Test Validation:**
1. Run deployment tests with `python -m pytest tests/test_deployment_strategies.py -v`
2. Execute strategy integration tests with mock deployment targets
3. Validate health check logic with simulated service responses
4. Test rollback scenarios with various failure conditions

**Manual Verification Steps:**
1. Execute a rolling deployment against mock targets and verify incremental updates
2. Test blue-green deployment and verify atomic traffic switching
3. Execute canary deployment and verify gradual traffic shifting
4. Trigger rollback scenarios and verify automatic recovery

| Verification Point | Command/Action | Expected Result | Troubleshooting |
|--------------------|----------------|-----------------|-----------------|
| Rolling Deployment | Deploy to 3 mock instances | Instances updated one at a time | Check health check configuration |
| Blue-Green Switch | Execute blue-green deployment | Atomic traffic switch | Verify traffic routing logic |
| Canary Traffic | Start canary with 10% traffic | Correct traffic distribution | Check load balancer integration |
| Health Check Failure | Simulate unhealthy instance | Deployment rollback triggered | Verify failure detection logic |
| Manual Approval | Pause deployment for approval | Deployment waits for manual trigger | Check approval gate implementation |

**Success Criteria Checklist:**
- [ ] Rolling deployments update instances incrementally with health validation
- [ ] Blue-green deployments switch traffic atomically after health verification
- [ ] Canary deployments shift traffic gradually with monitoring
- [ ] Health check failures trigger automatic rollback
- [ ] Manual approval gates pause deployments correctly
- [ ] Rollback procedures restore previous version successfully

### Implementation Guidance

#### Technology Recommendations

| Component | Simple Option | Advanced Option | Notes |
|-----------|---------------|-----------------|-------|
| Testing Framework | `pytest` with fixtures | `pytest` with `pytest-asyncio` for async | Use fixtures for test infrastructure |
| Test Isolation | `unittest.mock` for mocking | `testcontainers` for real services | Mock for unit tests, containers for integration |
| File System Testing | `tempfile` with cleanup | `pytest-tmp-path` with automatic cleanup | Temporary directories prevent test pollution |
| Container Testing | Mock `DockerClient` | Real Docker with cleanup | Mock for unit tests, real for integration |
| Test Data Management | JSON fixtures | `pytest-datadir` for complex data | Use datadir for realistic pipeline definitions |
| Test Orchestration | `pytest` discovery | `tox` for multi-environment testing | Tox for testing multiple Python versions |

#### Recommended File Structure

```
ci_cd_pipeline/
├── tests/
│   ├── unit/                          ← Isolated component tests
│   │   ├── test_pipeline_parser.py    ← Parser unit tests
│   │   ├── test_job_executor.py       ← Executor unit tests  
│   │   ├── test_artifact_manager.py   ← Artifact unit tests
│   │   └── test_deployment_strategies.py ← Deployment unit tests
│   ├── integration/                   ← Component interaction tests
│   │   ├── test_component_communication.py ← Event/message flow
│   │   ├── test_end_to_end_workflows.py    ← Complete pipelines
│   │   └── test_infrastructure.py          ← Real infrastructure
│   ├── fixtures/                      ← Test data and utilities
│   │   ├── pipeline_definitions/      ← Sample YAML files
│   │   ├── test_artifacts/            ← Sample build outputs
│   │   └── mock_services/             ← Test service implementations
│   ├── conftest.py                    ← Pytest configuration and fixtures
│   └── test_helpers/                  ← Testing utility functions
│       ├── mock_docker_client.py      ← Docker mocking utilities
│       ├── test_artifact_storage.py   ← Storage mocking
│       └── deployment_mocks.py        ← Deployment target mocks
├── src/ci_cd_pipeline/
│   └── [main application code...]
└── pytest.ini                        ← Pytest configuration
```

#### Test Infrastructure Starter Code

**Complete Test Configuration (`conftest.py`):**

```python
"""Pytest configuration and shared fixtures for CI/CD pipeline testing."""
import tempfile
import shutil
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, Generator

from ci_cd_pipeline.core.models import PipelineDefinition, JobStatus
from ci_cd_pipeline.executor.docker_client import DockerClient
from ci_cd_pipeline.artifact.artifact_manager import ArtifactManager
from ci_cd_pipeline.deployment.strategies import DeploymentStrategy

@pytest.fixture
def temp_storage_dir() -> Generator[Path, None, None]:
    """Provide temporary directory for artifact storage testing."""
    temp_dir = tempfile.mkdtemp(prefix="ci_cd_test_")
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def mock_docker_client() -> Mock:
    """Provide mock Docker client for isolated testing."""
    mock_client = Mock(spec=DockerClient)
    mock_client.run_command.return_value = (0, "Success output")
    mock_client.streaming_run.return_value = iter(["Success output\n"])
    return mock_client

@pytest.fixture
def sample_pipeline_definition() -> PipelineDefinition:
    """Provide sample pipeline definition for testing."""
    from datetime import datetime
    return PipelineDefinition(
        name="test-pipeline",
        jobs={
            "build": PipelineJob(
                name="build",
                steps=[
                    PipelineStep(
                        name="compile",
                        script=["make", "build"],
                        image="ubuntu:20.04",
                        environment={"BUILD_TYPE": "release"},
                        timeout=300,
                        retry_count=2,
                        working_directory="/workspace",
                        continue_on_failure=False
                    )
                ],
                depends_on=[],
                artifacts={"binary": "build/app"},
                environment={"STAGE": "build"},
                status=JobStatus.PENDING,
                started_at=None,
                finished_at=None,
                logs=[]
            )
        },
        global_env={"PROJECT": "test"},
        created_at=datetime.now(),
        timeout=3600
    )

@pytest.fixture
def integration_test_harness(temp_storage_dir: Path) -> 'IntegrationTestHarness':
    """Provide complete test harness for integration testing."""
    return IntegrationTestHarness(storage_dir=temp_storage_dir)

class IntegrationTestHarness:
    """Test infrastructure for integration testing with controlled components."""
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.mock_deployment_targets = []
        self.event_history = []
        
    def create_mock_deployment_target(self, name: str, healthy: bool = True) -> Dict[str, Any]:
        """Create mock deployment target that responds to health checks."""
        target = {
            "name": name,
            "url": f"http://mock-{name}.test",
            "healthy": healthy,
            "response_time": 0.1 if healthy else 5.0
        }
        self.mock_deployment_targets.append(target)
        return target
        
    def simulate_health_check(self, target_url: str) -> tuple[bool, float]:
        """Simulate health check response for mock targets."""
        for target in self.mock_deployment_targets:
            if target["url"] == target_url:
                return target["healthy"], target["response_time"]
        return False, 10.0  # Unknown target is unhealthy
        
    def record_event(self, event_type: str, component: str, data: Dict[str, Any]):
        """Record events for integration test validation."""
        self.event_history.append({
            "type": event_type,
            "component": component,
            "data": data,
            "timestamp": pytest.current_time()
        })
```

**Mock Docker Client Implementation:**

```python
"""Mock Docker client for testing job executor without containers."""
from typing import Dict, List, Tuple, Iterator
from unittest.mock import Mock
import time
import threading

class MockDockerClient:
    """Mock Docker client that simulates container execution for testing."""
    
    def __init__(self):
        self.executed_commands = []
        self.failure_commands = set()  # Commands that should fail
        self.slow_commands = set()     # Commands that should be slow
        
    def run_command(
        self, 
        image: str, 
        command: List[str], 
        environment: Dict[str, str] = None,
        working_dir: str = "/workspace",
        volumes: Dict[str, str] = None,
        timeout: int = 3600
    ) -> Tuple[int, str]:
        """Mock command execution with configurable success/failure."""
        # TODO 1: Record executed command for test validation
        # TODO 2: Check if command should fail based on configuration
        # TODO 3: Simulate execution delay for slow commands
        # TODO 4: Return appropriate exit code and output
        pass
        
    def streaming_run(
        self,
        image: str,
        command: List[str],
        **kwargs
    ) -> Iterator[str]:
        """Mock streaming command execution."""
        # TODO 1: Execute command using run_command
        # TODO 2: Split output into lines for streaming
        # TODO 3: Yield each line with realistic timing
        # TODO 4: Handle timeout scenarios appropriately
        pass
        
    def configure_failure(self, command_pattern: str):
        """Configure specific commands to fail for testing."""
        self.failure_commands.add(command_pattern)
        
    def configure_slow_execution(self, command_pattern: str):
        """Configure specific commands to execute slowly for timeout testing."""
        self.slow_commands.add(command_pattern)
```

#### Core Testing Skeleton Code

**Pipeline Parser Test Skeleton:**

```python
"""Unit tests for pipeline definition parser."""
import pytest
from ci_cd_pipeline.parser.pipeline_parser import PipelineParser
from ci_cd_pipeline.core.exceptions import (
    ValidationError, 
    CircularDependencyError,
    VariableResolutionError
)

class TestPipelineParser:
    """Test suite for pipeline definition parsing and validation."""
    
    @pytest.fixture
    def parser(self):
        """Provide parser instance for testing."""
        return PipelineParser()
    
    def test_parse_valid_pipeline(self, parser):
        """Test parsing of valid pipeline definition."""
        yaml_content = """
        name: test-pipeline
        jobs:
          build:
            steps:
              - name: compile
                script: ["make", "build"]
        """
        # TODO 1: Parse YAML content into PipelineDefinition
        # TODO 2: Verify pipeline name is correctly extracted
        # TODO 3: Verify job structure is properly parsed
        # TODO 4: Verify steps are correctly structured
        # Hint: Use parser.parse_yaml(yaml_content)
        
    def test_detect_circular_dependencies(self, parser):
        """Test detection of circular job dependencies."""
        yaml_content = """
        name: circular-pipeline
        jobs:
          job_a:
            depends_on: [job_b]
            steps: [{name: step1, script: [echo "a"]}]
          job_b:
            depends_on: [job_a]
            steps: [{name: step2, script: [echo "b"]}]
        """
        # TODO 1: Parse pipeline with circular dependencies
        # TODO 2: Attempt to build dependency graph
        # TODO 3: Verify CircularDependencyError is raised
        # TODO 4: Check that error includes cycle path information
        # Hint: Use pytest.raises() to catch expected exception
        
    def test_variable_substitution(self, parser):
        """Test variable resolution in pipeline definitions."""
        yaml_content = """
        name: variable-pipeline
        jobs:
          build:
            environment:
              OUTPUT_DIR: "${WORKSPACE}/build"
            steps:
              - name: build
                script: ["make", "OUTPUT=${OUTPUT_DIR}"]
        """
        # TODO 1: Set up environment variables for test
        # TODO 2: Parse pipeline with variable references
        # TODO 3: Verify variables are resolved in job environment
        # TODO 4: Verify variables are resolved in step scripts
        # Hint: Use monkeypatch to set environment variables
```

**Job Executor Test Skeleton:**

```python
"""Integration tests for job executor with real containers."""
import pytest
from ci_cd_pipeline.executor.job_executor import JobExecutor
from ci_cd_pipeline.core.models import PipelineJob, PipelineStep, JobStatus

class TestJobExecutorIntegration:
    """Integration tests for job execution with Docker containers."""
    
    @pytest.fixture
    def executor(self, mock_docker_client):
        """Provide job executor with mock Docker client."""
        return JobExecutor(docker_client=mock_docker_client)
    
    def test_successful_job_execution(self, executor, sample_pipeline_definition):
        """Test successful execution of pipeline job."""
        job = sample_pipeline_definition.jobs["build"]
        
        # TODO 1: Execute job using executor.execute_job()
        # TODO 2: Verify job completes successfully (returns True)
        # TODO 3: Check job status transitions to JobStatus.SUCCESS
        # TODO 4: Verify step logs are captured correctly
        # TODO 5: Verify job timing metadata is recorded
        # Hint: Mock successful docker responses before execution
        
    def test_job_timeout_handling(self, executor):
        """Test job execution timeout and cleanup."""
        long_running_job = PipelineJob(
            name="timeout-test",
            steps=[
                PipelineStep(
                    name="long-step",
                    script=["sleep", "10"],
                    image="ubuntu:20.04",
                    timeout=2,  # 2 second timeout
                    retry_count=0,
                    working_directory="/workspace",
                    continue_on_failure=False
                )
            ],
            depends_on=[],
            artifacts={},
            environment={},
            status=JobStatus.PENDING,
            logs=[]
        )
        
        # TODO 1: Configure mock to simulate slow execution
        # TODO 2: Execute job and measure execution time
        # TODO 3: Verify job times out after configured duration
        # TODO 4: Verify job status transitions to JobStatus.FAILED
        # TODO 5: Verify cleanup operations are performed
        # Hint: Use time.time() to measure actual execution duration
        
    def test_retry_policy_execution(self, executor, mock_docker_client):
        """Test exponential backoff retry policy."""
        failing_job = PipelineJob(
            name="retry-test",
            steps=[
                PipelineStep(
                    name="flaky-step",
                    script=["exit", "1"],  # Always fails
                    image="ubuntu:20.04",
                    retry_count=3,
                    timeout=30,
                    working_directory="/workspace",
                    continue_on_failure=False
                )
            ],
            depends_on=[],
            artifacts={},
            environment={},
            status=JobStatus.PENDING,
            logs=[]
        )
        
        # TODO 1: Configure mock to fail consistently
        # TODO 2: Execute job and track retry attempts
        # TODO 3: Verify correct number of retries (3 attempts)
        # TODO 4: Verify exponential backoff delays
        # TODO 5: Verify final status is JobStatus.FAILED
        # Hint: Use mock.call_count to verify retry attempts
```

#### Milestone Checkpoint Validation Scripts

**Milestone 1 Validation Script (`validate_milestone_1.py`):**

```python
"""Validation script for Milestone 1: Pipeline Definition Parser."""
import sys
from pathlib import Path
from ci_cd_pipeline.parser.pipeline_parser import PipelineParser

def validate_milestone_1() -> bool:
    """Execute comprehensive validation of pipeline parser implementation."""
    parser = PipelineParser()
    validation_passed = True
    
    print("🔍 Validating Milestone 1: Pipeline Definition Parser")
    print("=" * 60)
    
    # TODO 1: Test basic YAML parsing with sample pipeline
    # TODO 2: Test dependency graph construction and topological sort
    # TODO 3: Test variable substitution with environment variables
    # TODO 4: Test error handling with malformed YAML
    # TODO 5: Test circular dependency detection
    # TODO 6: Report validation results with specific success/failure details
    
    # Return overall validation status
    return validation_passed

if __name__ == "__main__":
    success = validate_milestone_1()
    sys.exit(0 if success else 1)
```

**Milestone Integration Validation (`validate_integration.py`):**

```python
"""End-to-end validation script for complete CI/CD pipeline."""
import tempfile
import yaml
from pathlib import Path
from ci_cd_pipeline.pipeline_coordinator import PipelineCoordinator

def validate_complete_pipeline() -> bool:
    """Execute end-to-end pipeline validation with all components."""
    
    # Sample pipeline that exercises all components
    test_pipeline = {
        "name": "integration-test-pipeline",
        "jobs": {
            "build": {
                "steps": [
                    {
                        "name": "compile",
                        "script": ["echo 'Building application'", "touch app.jar"],
                        "image": "ubuntu:20.04"
                    }
                ],
                "artifacts": {"application": "app.jar"}
            },
            "test": {
                "depends_on": ["build"],
                "steps": [
                    {
                        "name": "unit-tests",
                        "script": ["echo 'Running tests'", "test -f app.jar"],
                        "image": "ubuntu:20.04"
                    }
                ]
            },
            "deploy": {
                "depends_on": ["test"],
                "steps": [
                    {
                        "name": "rolling-deploy",
                        "script": ["echo 'Deploying application'"],
                        "image": "ubuntu:20.04"
                    }
                ]
            }
        }
    }
    
    print("🚀 Validating Complete CI/CD Pipeline")
    print("=" * 50)
    
    # TODO 1: Create temporary pipeline definition file
    # TODO 2: Initialize pipeline coordinator with all components
    # TODO 3: Execute complete pipeline and monitor progress
    # TODO 4: Verify jobs execute in correct dependency order
    # TODO 5: Verify artifacts are transferred between jobs
    # TODO 6: Verify deployment executes successfully
    # TODO 7: Report detailed validation results
    
    return True  # Replace with actual validation logic

if __name__ == "__main__":
    import sys
    success = validate_complete_pipeline()
    print(f"✅ Integration validation: {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
```

#### Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Parser tests fail with "KeyError" | Missing required fields in test YAML | Check YAML structure against schema | Add missing fields to test data |
| Mock Docker client not called | Test not using injected mock | Verify mock is passed to component | Use dependency injection properly |
| Integration tests timeout | Real Docker operations are slow | Check Docker daemon status | Use shorter timeouts for tests |
| Artifact tests fail with "Permission denied" | Test cleanup issues | Check file permissions in temp directory | Use pytest fixtures for cleanup |
| Variable substitution fails | Environment variables not set | Print available variables in test | Use monkeypatch to set test vars |
| Dependency graph invalid | Circular dependency in test data | Visualize dependency graph | Remove circular references |
| Health checks always fail | Mock targets not configured | Check mock target setup | Configure healthy mock responses |
| Tests pass individually but fail in suite | Test isolation problems | Look for shared state between tests | Use fresh fixtures for each test |


## Debugging Guide

> **Milestone(s):** This section provides essential debugging techniques and troubleshooting procedures that apply across all milestones (1-4), helping developers quickly identify and resolve issues during implementation.

Think of debugging a CI/CD pipeline as being a **detective investigating a crime scene**. Unlike a simple program that runs and exits, a CI/CD pipeline involves multiple moving parts—YAML parsing, container execution, file transfers, and deployment orchestration—all working together across time and space. When something goes wrong, the evidence is scattered across log files, container states, filesystem artifacts, and network connections. A systematic approach to gathering clues, forming hypotheses, and testing theories is essential for quickly identifying root causes and implementing fixes.

The complexity of CI/CD debugging stems from the distributed nature of pipeline execution. A single pipeline run might involve parsing YAML on the coordinator node, executing jobs in ephemeral Docker containers, transferring artifacts between storage systems, and deploying to remote environments. Each component has its own failure modes, logging mechanisms, and recovery procedures. Understanding how to navigate this complexity systematically can mean the difference between a 5-minute fix and a 3-hour troubleshooting session.

### Common Implementation Bugs

The following table catalogs the most frequently encountered bugs during CI/CD pipeline development, organized by the symptoms developers typically observe first. Each entry provides the underlying cause, diagnostic techniques, and specific remediation steps.

| Symptom | Root Cause | Diagnostic Steps | Fix |
|---------|------------|------------------|-----|
| Pipeline hangs after parsing | Circular dependency in job graph not detected | Check `build_dependency_graph()` logs, visualize dependency graph with `get_execution_order()` | Add circular dependency detection in `validate_dependencies()`, return `CircularDependencyError` with `cycle_path` |
| Job starts but never completes | Docker container exits but `DockerClient.run_command()` doesn't return | Check `docker ps -a` for container exit codes, examine Docker daemon logs | Add timeout handling in `execute_step()`, use `docker.errors.ContainerError` exception handling |
| Variable substitution fails silently | `resolve_variables()` returns original text with `${VAR}` unchanged | Enable debug logging in variable resolution, check `get_available_variables()` output | Validate all variable references in `validate_variable_references()` before job execution |
| Artifact upload succeeds but download fails | File permissions or path traversal in `store_artifact()` | Check filesystem permissions on artifact storage directory, verify `generate_artifact_id()` output | Use absolute paths in artifact storage, validate content hash matches between upload and download |
| Container environment missing variables | `environment` dict not passed to `DockerClient.run_command()` | Print environment dict before container creation, check container's `/proc/1/environ` | Merge global and job-level environments correctly in `execute_job()` |
| Pipeline reports success but job actually failed | Exit code checking logic inverted in `execute_step()` | Check actual container exit codes vs reported job status | Fix boolean logic: `exit_code == 0` indicates success, not failure |
| Deployment strategy starts but never progresses | Health check URL returns 404, blocking `wait_for_healthy()` | Test health check URLs manually with curl, check `HealthCheckResult` values | Configure correct health check endpoints, add retry logic for temporary unavailability |
| Logs truncated or missing | Output buffer overflow in `streaming_run()` without proper streaming | Check log file sizes, examine memory usage during job execution | Implement proper log streaming with `Iterator` pattern, avoid buffering large outputs |
| Race condition in parallel jobs | Multiple jobs accessing same workspace simultaneously | Check for shared workspace paths, examine file locking errors | Use unique workspace directories per job: `/tmp/pipeline-{job_name}-{timestamp}` |
| Artifact cleanup deletes active artifacts | `RetentionPolicy` evaluation doesn't check `reference_count` | Verify `ArtifactInfo.reference_count` values, check `evaluate_retention_policies()` logic | Add reference count validation, implement grace period before deletion |
| Docker image pull fails intermittently | Network timeouts or registry rate limiting | Check Docker registry connectivity, examine pull error messages | Add exponential backoff retry for image pulls, implement local image caching |
| Deployment rollback hangs indefinitely | Previous deployment version no longer exists | Check deployment history, verify rollback target availability | Implement deployment version tracking, validate rollback targets before execution |
| Memory leak during long pipeline runs | Artifact content cached in memory without cleanup | Monitor memory usage over time, check `ArtifactManager` cache size | Implement streaming artifact transfers, add memory pressure-based cache eviction |
| Job dependency satisfaction logic incorrect | `mark_job_completed()` doesn't update dependency graph correctly | Trace job completion events, verify `get_ready_jobs()` returns correct jobs | Fix dependency counting logic, ensure completed jobs are properly marked in graph |
| Pipeline state corruption after crash | `ExecutionState` not persisted correctly across restarts | Check state persistence files, verify crash recovery logic | Implement write-ahead logging for state changes, add state validation on startup |

### Debugging Techniques

Effective CI/CD pipeline debugging requires a multi-layered approach that combines traditional software debugging with container orchestration and distributed systems techniques. The key is to establish observability at every layer of the system, from YAML parsing through final deployment.

#### Structured Logging Strategy

Think of structured logging as **creating a flight recorder** for your pipeline. Every significant event should be logged with sufficient context to reconstruct what happened without access to the original system state. The `PipelineFormatter` should include correlation IDs that link related events across different components.

| Log Level | Use Case | Example Events | Required Fields |
|-----------|----------|----------------|-----------------|
| DEBUG | Detailed execution flow | Variable resolution steps, container command construction | `correlation_id`, `component`, `operation`, `details` |
| INFO | Major state transitions | Job started/completed, artifact upload/download | `job_name`, `timestamp`, `status`, `duration` |
| WARN | Recoverable errors | Retry attempts, health check failures | `error_type`, `retry_count`, `next_attempt` |
| ERROR | Unrecoverable failures | Job failures, deployment rollbacks | `error_message`, `stack_trace`, `recovery_hint` |

Configure logging early in your debugging session using `setup_logging()` with appropriate verbosity. Enable debug-level logging for the specific component you're investigating to avoid overwhelming output from other parts of the system.

#### Container Debugging Workflow

Container-related issues require special debugging techniques because the execution environment is ephemeral and isolated. The `DockerClient` wrapper should provide comprehensive introspection capabilities.

**Step-by-step container debugging process:**

1. **Capture Container State**: Use `docker inspect <container-id>` to examine the container's configuration, including environment variables, volume mounts, and network settings
2. **Examine Exit Codes**: Check the container's exit code and correlate with standard meanings (0=success, 1-255=various failures)  
3. **Review Container Logs**: Use `docker logs <container-id>` to see both stdout and stderr from the failed execution
4. **Interactive Debugging**: Launch a new container with the same image and environment using `docker run -it --rm <image> /bin/bash` to reproduce issues manually
5. **Volume Mount Inspection**: Verify that workspace directories and artifact paths are correctly mounted and accessible within the container
6. **Network Connectivity**: Test network access from within containers, especially for artifact downloads and deployment targets

#### State Machine Debugging

Pipeline execution follows complex state machines for jobs, deployments, and overall pipeline flow. Understanding the current state and valid transitions is crucial for diagnosing stuck or incorrectly behaving pipelines.

| Component | State Query Method | Transition Log Events | Recovery Actions |
|-----------|-------------------|----------------------|------------------|
| Job Execution | `job.status` field in `PipelineJob` | Job status changes in pipeline logs | Cancel job, retry with clean state |
| Deployment | `get_deployment_progress()` return values | Deployment state transitions | Trigger rollback, manual intervention |
| Pipeline Flow | `ExecutionState.get_execution_summary()` | Pipeline coordinator logs | Pause pipeline, resume from checkpoint |
| Artifact Management | `get_artifact_info(content_hash)` | Artifact lifecycle events | Re-upload artifacts, verify checksums |

#### Dependency Graph Visualization

Complex job dependencies can create subtle bugs that are difficult to understand from logs alone. Create visual representations of the dependency graph to identify issues with parallel execution or dependency satisfaction logic.

```
Use the following debugging commands to export dependency information:
- get_execution_order() returns topological sort for manual verification
- validate_dependencies() identifies missing job references
- get_ready_jobs() shows which jobs can execute now
- build_dependency_graph() logs dependency relationships during construction
```

#### Artifact Transfer Debugging

Artifact-related failures often manifest as missing files, corruption, or permission issues. The key is to trace artifacts through their complete lifecycle from upload to consumption.

**Artifact debugging checklist:**

1. **Verify Upload**: Check that `store_artifact()` returns a valid content hash and the artifact exists in storage
2. **Validate Checksums**: Use `verify_artifact_integrity()` to confirm the stored artifact matches expected content  
3. **Check Permissions**: Ensure the artifact storage directory has correct read/write permissions for the pipeline process
4. **Trace Downloads**: Verify `retrieve_artifact()` downloads to the correct workspace location with proper permissions
5. **Monitor Storage**: Use `get_storage_stats()` to check available space and detect storage quota issues

#### Performance Profiling

Pipeline performance issues can stem from inefficient algorithms, resource contention, or network bottlenecks. Systematic profiling helps identify bottlenecks before they impact production deployments.

| Performance Issue | Measurement Technique | Common Causes | Solutions |
|-------------------|----------------------|---------------|-----------|
| Slow YAML parsing | Time `parse_yaml()` execution | Large pipeline files, complex variable substitution | Cache parsed pipelines, optimize regex patterns |
| Job execution delays | Container creation time tracking | Docker image pulls, resource limits | Pre-pull images, increase CPU/memory limits |
| Artifact transfer slowness | Measure upload/download throughput | Network bandwidth, disk I/O | Enable compression, parallel transfers |
| Deployment timeout | Health check response times | Service startup latency, database connections | Increase timeout thresholds, optimize startup |

### Troubleshooting Checklist

When facing a CI/CD pipeline issue, follow this systematic checklist to quickly identify the problem area and gather necessary diagnostic information. This process moves from broad system checks to specific component investigation.

#### Phase 1: System-Level Verification

Before diving into specific component debugging, verify that the underlying infrastructure is healthy and properly configured.

**Infrastructure Health Check:**

1. **Verify Docker Daemon**: Test Docker connectivity with `docker version` and `docker info`
2. **Check System Resources**: Examine CPU, memory, and disk space availability on the pipeline host
3. **Validate File Permissions**: Ensure the pipeline process has read/write access to workspace and artifact directories  
4. **Test Network Connectivity**: Verify connectivity to container registries, deployment targets, and external dependencies
5. **Review System Logs**: Check system logs for hardware errors, resource exhaustion, or security policy violations

**Configuration Validation:**

1. **Pipeline Definition Syntax**: Validate YAML syntax with a standard YAML parser before pipeline-specific validation
2. **Environment Variables**: Verify all required environment variables are set with expected values
3. **Path Resolution**: Check that all file paths in the pipeline definition resolve to accessible locations
4. **Dependency Availability**: Confirm all required Docker images, external services, and deployment targets are reachable

#### Phase 2: Component-Specific Diagnosis

Once system-level issues are ruled out, focus on the specific pipeline component exhibiting problems. Use the component's specific debugging interfaces and log analysis.

**Pipeline Parser Issues:**

1. **Enable Parser Debug Logging**: Set logging level to DEBUG for the parser component
2. **Validate YAML Structure**: Use `parse_yaml()` with error handling to identify syntax or schema violations
3. **Check Variable Resolution**: Test `resolve_variables()` with sample variable expressions to verify substitution logic
4. **Examine Dependency Graph**: Use `build_dependency_graph()` and verify the output makes logical sense
5. **Test Edge Cases**: Try the parser with minimal, maximal, and malformed pipeline definitions

**Job Execution Problems:**

1. **Container Environment**: Inspect the Docker container's environment variables and volume mounts
2. **Command Execution**: Test the exact command that's failing by running it manually in the same container image
3. **Resource Limits**: Check if jobs are hitting CPU, memory, or timeout limits during execution
4. **Exit Code Analysis**: Map container exit codes to specific failure modes (OOM kill=137, timeout=124, etc.)
5. **Workspace State**: Verify the job workspace contains expected files before and after execution

**Artifact Management Failures:**

1. **Storage Backend Health**: Verify the artifact storage system is accessible and has sufficient space
2. **Checksum Verification**: Use `verify_artifact_integrity()` to confirm stored artifacts haven't been corrupted
3. **Path Resolution**: Check that artifact upload and download paths resolve correctly within job workspaces
4. **Retention Policy**: Verify retention policies aren't deleting artifacts that are still needed
5. **Transfer Logging**: Enable detailed logging for upload and download operations to trace transfer failures

**Deployment Strategy Issues:**

1. **Health Check Validation**: Test deployment health check URLs manually to verify they return expected responses
2. **Traffic Routing**: Verify load balancer configuration allows traffic to reach new deployment instances
3. **Version Tracking**: Check that deployment version information is correctly maintained and accessible for rollback
4. **Resource Provisioning**: Confirm deployment targets have sufficient resources for the new application version
5. **Network Policy**: Verify network security policies allow communication between deployment components

#### Phase 3: Data Flow Tracing

If component-specific debugging doesn't reveal the issue, trace data flow through the entire pipeline to identify where the breakdown occurs.

**End-to-End Pipeline Tracing:**

1. **Correlation ID Tracking**: Follow a single pipeline execution using its correlation ID across all log files and components
2. **State Transition Verification**: Verify each job progresses through expected state transitions (PENDING → RUNNING → SUCCESS/FAILED)
3. **Artifact Lineage**: Trace artifacts from production through consumption to verify the complete transfer chain
4. **Event Sequence Analysis**: Review the event timeline to identify unexpected ordering or missing events
5. **Component Communication**: Verify messages and data are correctly passed between pipeline components

**Performance Bottleneck Identification:**

1. **Timing Analysis**: Measure time spent in each pipeline phase to identify performance bottlenecks
2. **Resource Utilization**: Monitor CPU, memory, and I/O usage throughout pipeline execution
3. **Parallelism Verification**: Confirm that jobs intended to run in parallel are actually executing simultaneously
4. **Queue Analysis**: Check for job queuing delays or resource contention issues
5. **External Dependency Timing**: Measure latency for external service calls, image pulls, and network transfers

### Implementation Guidance

The debugging infrastructure should be built into the CI/CD pipeline from the beginning, not added as an afterthought. Comprehensive observability and diagnostic capabilities are essential for maintaining a production CI/CD system.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Logging Framework | Python `logging` module with file handlers | Structured logging with `structlog` + JSON formatters |
| Container Inspection | Direct Docker CLI commands via `subprocess` | Docker Python SDK with comprehensive error handling |
| State Persistence | JSON files for pipeline state | SQLite database for queryable state history |
| Performance Monitoring | Simple timing decorators | Prometheus metrics with Grafana dashboards |
| Log Aggregation | Local log files with rotation | Centralized logging with ELK stack |

#### Debugging Infrastructure Code

**Comprehensive logging setup with correlation ID support:**

```python
import logging
import structlog
import json
from datetime import datetime
from typing import Dict, Any, Optional

class PipelineFormatter(logging.Formatter):
    """Custom formatter that includes pipeline context in all log messages."""
    
    def __init__(self, include_correlation_id=True, include_component=True):
        self.include_correlation_id = include_correlation_id
        self.include_component = include_component
        super().__init__()
    
    def format(self, record):
        # Add standard pipeline fields to every log record
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = getattr(logging, '_current_correlation_id', 'unknown')
        if not hasattr(record, 'component'):
            record.component = getattr(logging, '_current_component', 'system')
        
        # Create structured log entry
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'component': record.component,
            'correlation_id': record.correlation_id,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add any extra fields passed to the logger
        for key, value in record.__dict__.items():
            if key not in log_data and not key.startswith('_'):
                log_data[key] = value
        
        return json.dumps(log_data)

def setup_logging(level=logging.INFO, log_file=None, correlation_id=None, component=None):
    """Configure pipeline logging with structured output and context."""
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = PipelineFormatter()
    
    # Console handler for interactive debugging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for persistent logs
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set global context
    if correlation_id:
        logging._current_correlation_id = correlation_id
    if component:
        logging._current_component = component

def get_logger(component_name: str, correlation_id: Optional[str] = None):
    """Get a logger with automatic component tagging."""
    logger = logging.getLogger(component_name)
    
    # Add context to all messages from this logger
    old_makeRecord = logger.makeRecord
    def makeRecord(name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None):
        if extra is None:
            extra = {}
        extra['component'] = component_name
        if correlation_id:
            extra['correlation_id'] = correlation_id
        return old_makeRecord(name, level, fn, lno, msg, args, exc_info, func, extra, sinfo)
    
    logger.makeRecord = makeRecord
    return logger
```

**Container debugging utilities:**

```python
import docker
import json
import subprocess
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ContainerDebugInfo:
    """Complete container debugging information."""
    container_id: str
    image: str
    exit_code: int
    stdout: str
    stderr: str
    environment: Dict[str, str]
    mounts: List[Dict[str, str]]
    network_settings: Dict[str, Any]
    resource_usage: Dict[str, Any]

class ContainerDebugger:
    """Utilities for debugging Docker container execution issues."""
    
    def __init__(self, docker_client: docker.DockerClient):
        self.docker_client = docker_client
        self.logger = get_logger('container_debugger')
    
    def capture_container_state(self, container_id: str) -> ContainerDebugInfo:
        """Capture complete state of a container for debugging."""
        try:
            container = self.docker_client.containers.get(container_id)
            
            # Get container inspection data
            inspect_data = container.attrs
            
            # Extract key debugging information
            debug_info = ContainerDebugInfo(
                container_id=container_id,
                image=inspect_data['Config']['Image'],
                exit_code=inspect_data['State']['ExitCode'],
                stdout=container.logs(stdout=True, stderr=False).decode('utf-8', errors='replace'),
                stderr=container.logs(stdout=False, stderr=True).decode('utf-8', errors='replace'),
                environment=dict(env.split('=', 1) for env in inspect_data['Config']['Env']),
                mounts=inspect_data['Mounts'],
                network_settings=inspect_data['NetworkSettings'],
                resource_usage=inspect_data['HostConfig']
            )
            
            self.logger.info(f"Captured debug info for container {container_id}",
                           extra={'exit_code': debug_info.exit_code})
            
            return debug_info
            
        except docker.errors.NotFound:
            self.logger.error(f"Container {container_id} not found")
            raise
        except Exception as e:
            self.logger.error(f"Failed to capture container state: {e}")
            raise
    
    def reproduce_container_environment(self, debug_info: ContainerDebugInfo, interactive=True) -> str:
        """Launch interactive container with same environment for debugging."""
        # Build docker run command
        cmd_parts = ['docker', 'run']
        
        if interactive:
            cmd_parts.extend(['-it', '--rm'])
        
        # Add environment variables
        for key, value in debug_info.environment.items():
            cmd_parts.extend(['-e', f"{key}={value}"])
        
        # Add volume mounts
        for mount in debug_info.mounts:
            if mount['Type'] == 'bind':
                cmd_parts.extend(['-v', f"{mount['Source']}:{mount['Destination']}"])
        
        # Add image and shell
        cmd_parts.append(debug_info.image)
        if interactive:
            cmd_parts.append('/bin/bash')
        
        cmd_string = ' '.join(cmd_parts)
        self.logger.info(f"Reproduction command: {cmd_string}")
        
        return cmd_string
    
    def analyze_exit_code(self, exit_code: int) -> Dict[str, str]:
        """Analyze container exit code and provide debugging hints."""
        exit_code_meanings = {
            0: "Success - container completed normally",
            1: "General errors - check application logs for details",
            2: "Misuse of shell builtins - invalid command syntax",
            125: "Docker daemon error - container failed to run",
            126: "Container command not executable - check permissions",
            127: "Container command not found - check PATH and command spelling",
            128: "Invalid exit argument - exit code out of range",
            130: "Script terminated by Ctrl+C (SIGINT)",
            137: "Process killed by SIGKILL - likely OOM or timeout",
            139: "Segmentation fault - application crash",
            143: "Process terminated by SIGTERM - graceful shutdown"
        }
        
        analysis = {
            'exit_code': str(exit_code),
            'meaning': exit_code_meanings.get(exit_code, f"Unknown exit code {exit_code}"),
            'category': self._categorize_exit_code(exit_code),
            'debugging_hint': self._get_debugging_hint(exit_code)
        }
        
        return analysis
    
    def _categorize_exit_code(self, exit_code: int) -> str:
        """Categorize exit codes for systematic debugging."""
        if exit_code == 0:
            return 'success'
        elif 1 <= exit_code <= 2:
            return 'application_error'
        elif 125 <= exit_code <= 128:
            return 'docker_error'
        elif exit_code in [130, 137, 143]:
            return 'signal_termination'
        elif exit_code == 139:
            return 'crash'
        else:
            return 'unknown'
    
    def _get_debugging_hint(self, exit_code: int) -> str:
        """Provide specific debugging suggestions based on exit code."""
        hints = {
            1: "Check application logs and error output for specific failure details",
            125: "Verify Docker daemon is running and container image exists",
            126: "Check file permissions on scripts and executables",
            127: "Verify command exists in container image and PATH is correct",
            137: "Check memory limits and container resource usage",
            139: "Enable core dumps and check for null pointer dereferences"
        }
        return hints.get(exit_code, "Review container logs and application documentation")
```

**State machine debugging utilities:**

```python
from enum import Enum
from typing import Dict, List, Set, Optional
from datetime import datetime, timedelta
import json

class StateTransitionTracker:
    """Track and validate state machine transitions for debugging."""
    
    def __init__(self, valid_transitions: Dict[str, Set[str]]):
        self.valid_transitions = valid_transitions
        self.transition_history = []
        self.current_states = {}
        self.logger = get_logger('state_tracker')
    
    def record_transition(self, entity_id: str, from_state: str, to_state: str, 
                         timestamp: Optional[datetime] = None):
        """Record a state transition and validate it's legal."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Validate transition is legal
        if from_state not in self.valid_transitions:
            self.logger.warning(f"Unknown source state: {from_state}")
        elif to_state not in self.valid_transitions[from_state]:
            self.logger.error(f"Invalid transition from {from_state} to {to_state}",
                            extra={'entity_id': entity_id, 'invalid_transition': True})
        
        # Record transition
        transition = {
            'entity_id': entity_id,
            'from_state': from_state,
            'to_state': to_state,
            'timestamp': timestamp.isoformat(),
            'valid': to_state in self.valid_transitions.get(from_state, set())
        }
        
        self.transition_history.append(transition)
        self.current_states[entity_id] = to_state
        
        self.logger.info(f"State transition: {entity_id} {from_state} -> {to_state}",
                        extra=transition)
    
    def get_stuck_entities(self, timeout_minutes: int = 30) -> List[Dict[str, Any]]:
        """Find entities that haven't transitioned recently."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=timeout_minutes)
        stuck_entities = []
        
        for entity_id, current_state in self.current_states.items():
            # Find most recent transition for this entity
            recent_transitions = [t for t in self.transition_history 
                                if t['entity_id'] == entity_id]
            
            if not recent_transitions:
                continue
            
            last_transition = max(recent_transitions, 
                                key=lambda t: datetime.fromisoformat(t['timestamp']))
            last_time = datetime.fromisoformat(last_transition['timestamp'])
            
            if last_time < cutoff_time:
                stuck_entities.append({
                    'entity_id': entity_id,
                    'current_state': current_state,
                    'last_transition': last_transition['timestamp'],
                    'minutes_stuck': int((datetime.utcnow() - last_time).total_seconds() / 60)
                })
        
        return stuck_entities
    
    def export_transition_graph(self) -> Dict[str, Any]:
        """Export transition history for visualization tools."""
        return {
            'valid_transitions': {k: list(v) for k, v in self.valid_transitions.items()},
            'current_states': self.current_states,
            'transition_history': self.transition_history,
            'export_timestamp': datetime.utcnow().isoformat()
        }

# Job state machine configuration
JOB_STATE_TRANSITIONS = {
    'PENDING': {'RUNNING', 'CANCELLED'},
    'RUNNING': {'SUCCESS', 'FAILED', 'CANCELLED', 'RETRY'},
    'RETRY': {'RUNNING', 'FAILED', 'CANCELLED'},
    'SUCCESS': {'CANCELLED'},  # Allow cancellation of completed jobs for cleanup
    'FAILED': {'RETRY', 'CANCELLED'},
    'CANCELLED': set()  # Terminal state
}

# Global state tracker instance
job_state_tracker = StateTransitionTracker(JOB_STATE_TRANSITIONS)
```

#### Milestone Checkpoint: Debugging Infrastructure

After implementing the debugging infrastructure, verify it works correctly:

**Test Commands:**
```bash
# Test structured logging
python -m pytest tests/test_debugging.py::test_structured_logging -v

# Test container debugging
python -c "from debugging import ContainerDebugger; print('Debug infrastructure loaded')"

# Test state tracking
python -c "from debugging import job_state_tracker; job_state_tracker.record_transition('job1', 'PENDING', 'RUNNING')"
```

**Expected Behavior:**
- Log messages should include correlation IDs, timestamps, and component names
- Container debugging should capture complete environment and mount information
- State transitions should be validated and invalid transitions should trigger warnings
- Stuck entity detection should identify jobs that haven't progressed in the expected timeframe

**Signs of Correct Implementation:**
- JSON-formatted log output with all required fields
- Container reproduction commands that can be run manually for debugging
- State transition validation that catches invalid transitions
- Performance timing data for identifying bottlenecks


## Future Extensions

> **Milestone(s):** This section outlines potential enhancements that build upon the foundation established in all four milestones (1-4), providing a roadmap for evolving the CI/CD pipeline into an enterprise-grade platform.

Think of the future extensions as **evolutionary branches** from our current CI/CD pipeline system. Just as a thriving city starts with basic infrastructure (roads, utilities, buildings) and then grows to include advanced systems (smart traffic management, environmental monitoring, emergency response networks), our pipeline system has established the fundamental capabilities and can now evolve toward more sophisticated enterprise features.

The current architecture provides a solid foundation that naturally supports these extensions through its modular design, event-driven communication patterns, and abstracted interfaces. Each extension category addresses different scaling challenges that organizations encounter as their CI/CD usage grows from small teams to enterprise-wide deployment orchestration.

### Scalability Improvements: Distributed Execution and Horizontal Scaling

Think of distributed execution as transforming our single construction foreman (Job Executor) into a **construction management company** with multiple project managers, each capable of overseeing different job sites simultaneously. Just as a construction company can take on multiple building projects by distributing work across teams while maintaining centralized planning and quality control, distributed pipeline execution allows us to handle multiple pipelines and jobs concurrently across a cluster of worker nodes.

The current Job Executor architecture already provides the foundation for distributed execution through its container-based isolation model. Each job step runs in an isolated Docker container, which means the actual execution environment is already portable and can be moved between different host machines without modification. The challenge lies in coordinating this distributed work while maintaining the reliability, artifact management, and deployment capabilities we've built.

> **Key Insight**: The event-driven architecture established in our current system becomes crucial for distributed execution. The `EventDispatcher` and `FlowEvent` structures provide the communication backbone needed to coordinate work across multiple nodes while maintaining visibility into overall pipeline progress.

#### Distributed Job Scheduling Architecture

The distributed scheduling system extends our current `PipelineFlowCoordinator` into a cluster-aware orchestrator that can dispatch jobs to available worker nodes based on resource requirements, node capabilities, and current load distribution.

**Distributed Scheduling Components:**

| Component | Responsibility | Current Foundation | Extension Required |
|-----------|---------------|-------------------|-------------------|
| `ClusterCoordinator` | Manages worker node registration and health | `ComponentHealthMonitor` | Add node discovery and load balancing |
| `JobScheduler` | Assigns jobs to optimal worker nodes | `PipelineFlowCoordinator.get_ready_jobs()` | Add resource-aware scheduling algorithm |
| `WorkerNode` | Executes assigned jobs and reports status | `JobExecutor` execution logic | Add cluster communication and heartbeat |
| `ResourceManager` | Tracks resource usage across cluster | Current resource limits per job | Add cluster-wide resource accounting |
| `WorkStealingBalancer` | Redistributes work from overloaded nodes | Not present | New component for dynamic load balancing |

The distributed scheduler operates through a multi-phase process that builds on our existing dependency resolution:

1. **Cluster State Discovery**: The `ClusterCoordinator` maintains a real-time view of all available worker nodes, their current resource utilization (CPU, memory, storage), and their capabilities (supported container runtimes, available tools, network access patterns).

2. **Resource-Aware Job Assignment**: When the `JobScheduler` receives ready jobs from the dependency graph, it evaluates each job's resource requirements (specified in the `PipelineJob.environment` metadata) against available worker capacity and assigns jobs to nodes that can execute them efficiently.

3. **Work Distribution Protocol**: Jobs are distributed using a pull-based model where worker nodes request work when they have available capacity, preventing overloading and allowing for dynamic load balancing as cluster conditions change.

4. **Progress Coordination**: All job status updates, log streams, and completion events flow through the existing `EventDispatcher` system, ensuring the central coordinator maintains complete visibility into distributed execution progress.

5. **Failure Handling and Rescheduling**: When worker nodes fail or become unreachable, the `ClusterCoordinator` detects the failure through missed heartbeats and automatically reschedules any incomplete jobs to healthy nodes.

> **Decision: Pull-Based Work Distribution**
> - **Context**: Distributed job scheduling requires a mechanism for assigning work to worker nodes in a cluster
> - **Options Considered**: 
>   - Push model: Coordinator pushes jobs to workers
>   - Pull model: Workers request jobs from coordinator
>   - Hybrid model: Push with worker feedback
> - **Decision**: Pull-based work distribution with worker capacity reporting
> - **Rationale**: Pull model prevents overloading workers, handles network partitions gracefully, and allows workers to self-regulate based on local resource conditions. Workers know their own capacity better than a central coordinator.
> - **Consequences**: Requires heartbeat mechanism for worker health monitoring and adds latency for work assignment, but provides better fault tolerance and load distribution.

**Distributed Scheduling State Management:**

| State Information | Storage Location | Consistency Model | Recovery Mechanism |
|------------------|------------------|-------------------|-------------------|
| Job assignments | Central coordinator with write-ahead log | Strong consistency | Replay assignment log on coordinator restart |
| Worker heartbeats | In-memory with periodic persistence | Eventually consistent | Workers re-register on coordinator restart |
| Resource utilization | Worker-reported metrics | Eventually consistent | Workers report current state on reconnection |
| Job completion status | Event log replicated to coordinator | Strong consistency | Event replay for missed completion notifications |

#### Horizontal Scaling Patterns

The horizontal scaling implementation leverages Kubernetes-style orchestration patterns while maintaining compatibility with our existing artifact management and deployment systems.

**Auto-Scaling Triggers:**

| Metric | Threshold | Scale-Out Action | Scale-In Action | Cooldown Period |
|--------|-----------|------------------|-----------------|-----------------|
| Average CPU utilization | > 80% for 5 minutes | Add 25% more worker nodes | N/A | 10 minutes |
| Queue depth | > 50 pending jobs | Add worker nodes to reduce queue to < 20 | N/A | 5 minutes |
| Worker node utilization | < 20% for 30 minutes | N/A | Remove underutilized nodes | 60 minutes |
| Job completion latency | > 150% of historical average | Add worker nodes with relevant capabilities | N/A | 15 minutes |

The auto-scaling system integrates with our existing `ComponentHealthMonitor` to make scaling decisions based on comprehensive system health metrics rather than simple resource thresholds. This prevents thrashing during temporary load spikes and ensures scaling decisions consider the overall pipeline ecosystem health.

**Resource Pool Management:**

Think of resource pools as **specialized construction crews** within our distributed system. Just as a construction company might have separate teams for electrical work, plumbing, and general construction, pipeline clusters can maintain specialized worker pools for different types of computational work.

| Pool Type | Specialization | Resource Configuration | Typical Workloads |
|-----------|---------------|----------------------|------------------|
| `cpu-intensive` | High-performance computing | 16+ CPU cores, moderate memory | Compilation, testing, static analysis |
| `memory-intensive` | Large working sets | 64+ GB RAM, moderate CPU | Docker image builds, large test suites |
| `io-intensive` | Storage and network operations | High-bandwidth storage, network | Artifact transfers, database operations |
| `gpu-accelerated` | Machine learning workloads | GPU resources, ML frameworks | Model training, inference pipelines |
| `secure-isolated` | Sensitive workloads | Enhanced security controls | Production deployments, security scanning |

Worker nodes can belong to multiple pools based on their capabilities, and the job scheduler considers pool membership when making assignment decisions. This allows organizations to optimize resource utilization while ensuring jobs run on appropriate infrastructure.

#### Data Consistency in Distributed Systems

The distributed execution system maintains data consistency through a combination of event sourcing (building on our `FlowEvent` system) and distributed state reconciliation.

**Consistency Guarantees:**

| Data Type | Consistency Level | Mechanism | Trade-off |
|-----------|------------------|-----------|-----------|
| Job dependency resolution | Strong consistency | Single coordinator with WAL | Coordinator becomes bottleneck |
| Artifact availability | Eventual consistency | Content-addressable storage with replication | Temporary inconsistency possible |
| Worker health status | Eventual consistency | Heartbeat with configurable timeout | False positives during network partitions |
| Pipeline execution progress | Strong consistency for critical events | Event sourcing with ordered delivery | Higher latency for status updates |

The system uses **vector clocks** to order events across the distributed cluster, ensuring that even if network partitions occur, the system can maintain a consistent view of pipeline execution progress when connectivity is restored.

**Distributed Artifact Management:**

The existing `ArtifactManager` extends naturally to support distributed storage through a content-addressable approach where artifacts are replicated across multiple storage nodes based on configured replication policies.

| Replication Strategy | Use Case | Consistency Trade-off | Performance Impact |
|---------------------|----------|----------------------|-------------------|
| `immediate-replication` | Critical artifacts | Strong consistency, higher latency | 2-3x slower uploads |
| `lazy-replication` | Large artifacts | Eventual consistency | Background replication overhead |
| `geo-replication` | Multi-region deployments | Eventual consistency across regions | Network bandwidth usage |
| `ephemeral-local` | Temporary build artifacts | No replication | Fastest but no durability |

### Security Enhancements: Secret Management and Access Controls

Think of security enhancements as transforming our CI/CD pipeline from a **trusted workshop environment** into a **high-security manufacturing facility**. Just as a secure facility implements multiple layers of access control (badge readers, biometric scanners, compartmentalized access areas), our pipeline security extensions create multiple defensive layers that protect sensitive data, control access to resources, and maintain audit trails for compliance requirements.

The current pipeline architecture provides several security foundations that we can build upon: container isolation limits the blast radius of security issues, the event-driven architecture enables comprehensive audit logging, and the modular design allows security controls to be inserted at component boundaries without disrupting core functionality.

> **Key Security Principle**: Defense in depth requires security controls at every layer of the system - from network access to container runtime to secret handling. No single security mechanism should be a single point of failure.

#### Comprehensive Secret Management System

The secret management system extends beyond simple credential storage to provide a complete lifecycle management solution for sensitive data throughout the pipeline execution process.

**Secret Lifecycle Management:**

| Lifecycle Stage | Security Controls | Implementation | Audit Requirements |
|----------------|------------------|---------------|-------------------|
| **Creation** | Multi-person approval, entropy validation | Integration with HashiCorp Vault or AWS Secrets Manager | Log creation with approver identity |
| **Storage** | Encryption at rest, access logging | AES-256 encryption with hardware security module | Log all access attempts with timestamps |
| **Distribution** | Least-privilege access, time-bounded tokens | Short-lived credentials with automatic rotation | Log credential issuance and expiration |
| **Usage** | Runtime isolation, secure injection | Environment variable injection into containers | Log which jobs accessed which secrets |
| **Rotation** | Automated rotation, zero-downtime updates | Gradual rollout with fallback to previous version | Log rotation events and success/failure |
| **Revocation** | Immediate invalidation, impact analysis | Coordinate with all active pipeline executions | Log revocation reason and affected jobs |

The secret management system integrates with our existing `PipelineJob` execution through a secure injection mechanism that never stores secrets in logs or persists them to disk on worker nodes.

**Secret Injection Architecture:**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Secret Store  │────│  Secret Broker   │────│   Job Executor      │
│   (Vault/HSM)   │    │  (per worker)    │    │   (container)       │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
         │                        │                         │
         │ 1. Request credentials │                         │
         ├────────────────────────┤                         │
         │ 2. Time-limited token  │                         │
         │←───────────────────────┤                         │
         │                        │ 3. Inject into container│
         │                        ├─────────────────────────┤
         │                        │ 4. Automatic cleanup    │
         │                        │←────────────────────────┤
```

The `SecretBroker` component runs on each worker node and acts as a trusted intermediary between the central secret store and individual job containers. This design prevents secrets from being transmitted over the network to multiple worker nodes simultaneously and ensures that secrets are automatically cleaned up when jobs complete.

**Access Control Integration:**

| Integration Point | Control Mechanism | Implementation | Policy Examples |
|------------------|-------------------|---------------|-----------------|
| Pipeline definition | RBAC on pipeline file access | Git repository permissions | Only platform team can modify deployment pipelines |
| Job execution | Role-based job assignment | JWT tokens with role claims | Developers can run test jobs but not production deployments |
| Artifact access | Fine-grained artifact permissions | Content-based access control | Security artifacts only accessible to security team |
| Deployment targets | Environment-specific permissions | Target-aware authorization | Production deployments require additional approval |

#### Zero-Trust Network Architecture

The zero-trust model assumes that network connectivity does not imply authorization and requires authentication and authorization for every system interaction.

**Network Security Controls:**

| Network Layer | Security Mechanism | Implementation | Threat Mitigation |
|---------------|-------------------|---------------|-------------------|
| **Container-to-container** | Mutual TLS authentication | Service mesh (Istio/Linkerd) integration | Man-in-the-middle attacks, eavesdropping |
| **Worker-to-coordinator** | Certificate-based authentication | PKI with automatic certificate rotation | Rogue worker registration |
| **External API calls** | Egress filtering and monitoring | Network policy enforcement | Data exfiltration, supply chain attacks |
| **Artifact transfers** | End-to-end encryption | ChaCha20-Poly1305 with ephemeral keys | Artifact tampering during transfer |

The zero-trust implementation builds on our existing container isolation by adding network-level security controls that verify the identity and authorization of every communication, even within the supposedly trusted cluster environment.

**Supply Chain Security:**

Think of supply chain security as **quality control for all imported materials** in our construction analogy. Just as a construction company verifies that imported steel meets safety standards and comes from approved suppliers, our pipeline must verify that all external dependencies (container images, packages, tools) meet security standards and haven't been compromised.

| Supply Chain Risk | Detection Mechanism | Mitigation Strategy | Implementation |
|-------------------|--------------------|--------------------|---------------|
| **Malicious container images** | Image vulnerability scanning | Approved base image catalog | Integration with Twistlock/Aqua Security |
| **Compromised dependencies** | Package signature verification | Dependency pinning with checksum validation | Integration with Sonatype Nexus/JFrog Artifactory |
| **Insider threats** | Behavioral analysis | Anomaly detection on pipeline usage patterns | ML-based analysis of user behavior |
| **Code injection** | Static analysis | Mandatory security scanning gates | Integration with SonarQube/Checkmarx |

#### Compliance and Audit Framework

The compliance framework transforms our existing `FlowEvent` system into a comprehensive audit trail that meets regulatory requirements for industries like finance, healthcare, and government contracting.

**Regulatory Compliance Mapping:**

| Regulation | Requirement | Pipeline Implementation | Audit Evidence |
|------------|-------------|------------------------|----------------|
| **SOX (Sarbanes-Oxley)** | Change control documentation | Approval workflows with digital signatures | Immutable audit log of all changes and approvals |
| **HIPAA** | Access logging and encryption | Comprehensive access logs with PHI handling controls | Detailed logs of who accessed what data when |
| **PCI DSS** | Secure handling of payment data | Encrypted artifact storage with access controls | Cryptographic proof of secure data handling |
| **FedRAMP** | Continuous monitoring | Real-time security monitoring and alerting | Continuous compliance posture reporting |

The audit framework automatically generates compliance reports by analyzing the event stream and producing evidence packages that auditors can review without accessing the actual pipeline infrastructure.

**Immutable Audit Log Structure:**

| Log Entry Field | Purpose | Data Type | Example Value |
|----------------|---------|-----------|---------------|
| `event_id` | Unique event identifier | UUID | `550e8400-e29b-41d4-a716-446655440000` |
| `timestamp` | When event occurred | ISO 8601 timestamp | `2024-01-15T14:30:00.000Z` |
| `actor_id` | Who performed the action | User/system identifier | `user:john.doe@company.com` |
| `action` | What was done | Enumerated action type | `PIPELINE_EXECUTED` |
| `resource` | What was affected | Resource identifier | `pipeline:production-deployment` |
| `context` | Additional context | JSON object | `{"approval_id": "12345", "environment": "prod"}` |
| `integrity_hash` | Tamper detection | SHA-256 hash | `abc123...` |
| `signature` | Non-repudiation | Digital signature | `def456...` |

### Monitoring and Observability: Metrics Collection and Alerting Systems

Think of monitoring and observability as the **comprehensive instrumentation dashboard** for our CI/CD pipeline system. Just as a modern aircraft cockpit provides pilots with detailed real-time information about every system (engine performance, fuel levels, weather conditions, navigation status), our observability extensions give operations teams complete visibility into pipeline health, performance trends, resource utilization, and business impact metrics.

The current pipeline architecture already provides foundational observability through the `EventDispatcher` system and structured logging, but enterprise-scale operations require more sophisticated monitoring capabilities that can correlate events across distributed systems, predict failures before they occur, and provide business stakeholders with insights into deployment velocity and quality trends.

> **Observability Principle**: The three pillars of observability - metrics (what happened), logs (detailed context), and traces (how requests flow through the system) - must work together to provide complete system visibility. Each pillar provides different insights, but together they enable rapid problem diagnosis and system optimization.

#### Comprehensive Metrics Collection Framework

The metrics collection framework extends our existing pipeline execution tracking into a multi-dimensional monitoring system that captures technical performance metrics, business outcomes, and user experience indicators.

**Pipeline Performance Metrics:**

| Metric Category | Specific Metrics | Collection Method | Business Impact |
|----------------|------------------|-------------------|-----------------|
| **Throughput** | Jobs per hour, pipelines per day, artifact transfer rate | Event counting with time windows | Development team velocity |
| **Latency** | Job execution time, queue wait time, deployment duration | Timing instrumentation in job executor | Time to market for features |
| **Reliability** | Success rate, retry frequency, rollback percentage | Status tracking across all pipeline executions | Product quality and stability |
| **Resource Utilization** | CPU/memory usage, storage consumption, network bandwidth | Container runtime metrics and system monitoring | Infrastructure cost optimization |
| **User Experience** | Developer wait time, feedback loop duration, self-service success rate | End-to-end timing from developer perspective | Developer productivity and satisfaction |

The metrics collection system integrates with our existing `ComponentHealthMonitor` to provide multi-layered health assessment that considers not just individual component status but also system-wide performance trends and capacity planning indicators.

**Real-Time Metrics Architecture:**

```
┌──────────────────┐    ┌────────────────┐    ┌─────────────────────┐
│   Pipeline       │────│   Metrics      │────│   Time Series       │
│   Components     │    │   Collector    │    │   Database          │
└──────────────────┘    └────────────────┘    └─────────────────────┘
         │                        │                         │
         │ Emit structured        │ Aggregate and           │ Store with
         │ metric events          │ enrich metrics          │ retention policy
         │                        │                         │
         ▼                        ▼                         ▼
┌──────────────────┐    ┌────────────────┐    ┌─────────────────────┐
│   Application    │    │   Infrastructure│    │   Business          │
│   Metrics        │    │   Metrics       │    │   Intelligence      │
└──────────────────┘    └────────────────┘    └─────────────────────┘
```

The metrics collection system uses a push-based model where pipeline components emit structured metric events that are aggregated by collectors and stored in a time-series database (such as Prometheus or InfluxDB) for long-term analysis and alerting.

**Custom Metrics for Pipeline Domain:**

| Metric Name | Type | Labels | Description | Alerting Threshold |
|-------------|------|--------|-------------|-------------------|
| `pipeline_job_duration_seconds` | Histogram | pipeline, job, status | Time taken for job execution | >95th percentile of historical average |
| `pipeline_queue_depth` | Gauge | worker_pool | Number of jobs waiting for execution | >50 jobs for >10 minutes |
| `artifact_transfer_bytes_total` | Counter | source, destination, status | Total bytes transferred between stages | N/A (used for capacity planning) |
| `deployment_success_rate` | Gauge | environment, strategy | Percentage of successful deployments | <95% over 24-hour window |
| `security_scan_violations` | Counter | severity, scanner | Security issues found in pipeline | Any critical severity violation |

#### Distributed Tracing for Pipeline Execution

Distributed tracing extends our `FlowEvent` system to track individual pipeline executions as they flow through the distributed system, providing detailed timing and dependency information that enables root cause analysis of performance issues.

**Trace Instrumentation Points:**

| Component | Instrumentation | Span Information | Parent-Child Relationships |
|-----------|----------------|------------------|---------------------------|
| `PipelineFlowCoordinator` | Pipeline execution lifecycle | Pipeline ID, user, trigger | Root span for entire execution |
| `JobExecutor` | Individual job execution | Job name, worker node, resources | Child of pipeline span |
| `ArtifactManager` | Artifact operations | Artifact hash, size, operation type | Child of job spans |
| `DeploymentEngine` | Deployment operations | Strategy, environment, targets | Child of pipeline span |
| `SecretBroker` | Secret retrieval | Secret name, requesting job | Child of job spans |

The distributed tracing system uses OpenTelemetry standards to ensure compatibility with existing observability tools and enables correlation between pipeline events and external system behavior (such as cloud provider API calls or database operations).

**Trace Analysis Capabilities:**

Think of distributed traces as **detailed flight recordings** for each pipeline execution. Just as aircraft black boxes record every system interaction during a flight, distributed traces capture every component interaction during pipeline execution, enabling post-mortem analysis of both successful and failed operations.

| Analysis Type | Use Case | Implementation | Business Value |
|--------------|----------|---------------|----------------|
| **Critical Path Analysis** | Identify bottlenecks in pipeline execution | Calculate longest dependency chain through trace spans | Optimize pipeline performance |
| **Resource Utilization Correlation** | Link performance issues to resource constraints | Correlate trace timing with infrastructure metrics | Right-size infrastructure |
| **Error Propagation Analysis** | Understand how failures cascade through system | Analyze error spans and their downstream effects | Improve fault tolerance |
| **Dependency Impact Assessment** | Measure impact of external service latency | Track external API calls within pipeline traces | Optimize third-party integrations |

#### Intelligent Alerting and Anomaly Detection

The alerting system goes beyond simple threshold-based alerts to provide intelligent anomaly detection that learns normal pipeline behavior patterns and alerts on deviations that may indicate problems before they impact users.

**Multi-Level Alerting Strategy:**

| Alert Level | Trigger Condition | Response Time | Escalation Path | Example Scenario |
|-------------|------------------|---------------|-----------------|------------------|
| **Info** | Trend deviation | 4 hours | Log only | Pipeline execution time increased by 20% |
| **Warning** | Service degradation | 30 minutes | Team notification | Queue depth above normal but not critical |
| **Critical** | Service failure | 5 minutes | On-call engineer | Pipeline completely unable to execute jobs |
| **Emergency** | Security incident | Immediate | Security team + management | Unauthorized access detected |

The alerting system integrates machine learning algorithms to establish baseline behavior patterns for each pipeline and job type, reducing false positive alerts while ensuring that genuine anomalies are detected quickly.

**Anomaly Detection Algorithms:**

| Algorithm | Use Case | Data Requirements | False Positive Rate |
|-----------|----------|------------------|-------------------|
| **Statistical Process Control** | Detect execution time anomalies | 30 days of historical data | <5% with proper tuning |
| **Isolation Forest** | Identify unusual resource usage patterns | Multi-dimensional metrics | <3% for well-defined workloads |
| **Time Series Forecasting** | Predict capacity exhaustion | Continuous metrics with seasonal patterns | <2% for stable workloads |
| **Behavioral Analysis** | Detect unusual user activity patterns | User interaction logs | <1% after learning period |

#### Business Intelligence and Operational Dashboards

The dashboard system provides different views for different organizational roles, from developer self-service interfaces to executive-level business intelligence reports that connect pipeline performance to business outcomes.

**Role-Based Dashboard Views:**

| Role | Primary Metrics | Update Frequency | Key Insights |
|------|----------------|------------------|--------------|
| **Developer** | Personal pipeline status, job failure reasons | Real-time | What's blocking my work? |
| **Team Lead** | Team velocity, quality trends, resource usage | Hourly | How is my team performing? |
| **Platform Engineer** | System reliability, capacity utilization, cost optimization | Daily | How healthy is the platform? |
| **Engineering Manager** | Cross-team metrics, delivery predictability | Weekly | Are we meeting commitments? |
| **Executive** | Business impact, cost efficiency, competitive advantage | Monthly | What's the ROI of our CI/CD investment? |

The business intelligence system correlates pipeline metrics with business outcomes (feature delivery velocity, defect rates, customer satisfaction) to demonstrate the value of CI/CD investments and guide infrastructure optimization decisions.

**Advanced Dashboard Features:**

| Feature | Implementation | User Benefit | Technical Requirements |
|---------|---------------|--------------|----------------------|
| **Predictive Analysis** | Machine learning models on historical data | Anticipate capacity needs | Time-series database with sufficient history |
| **Root Cause Analysis** | Automated correlation of failures with system changes | Faster incident resolution | Event correlation across all system components |
| **Cost Attribution** | Resource usage tracking by team/project | Optimize infrastructure spending | Detailed resource metering and tagging |
| **Compliance Reporting** | Automated audit report generation | Reduce manual compliance work | Integration with audit log system |

### Implementation Guidance

The future extensions build upon the existing pipeline architecture through careful interface design and incremental enhancement rather than fundamental rewrites. This approach allows organizations to adopt these capabilities gradually based on their specific scaling challenges and requirements.

#### Technology Recommendations

| Extension Category | Simple Option | Advanced Option | Integration Complexity |
|--------------------|---------------|----------------|----------------------|
| **Distributed Execution** | Docker Swarm with simple work queue | Kubernetes with custom operators | Medium - requires cluster management |
| **Secret Management** | HashiCorp Vault community edition | Enterprise HSM with FIPS compliance | High - requires security review |
| **Monitoring** | Prometheus + Grafana | DataDog/New Relic enterprise platform | Low - metrics collection interfaces exist |
| **Tracing** | Jaeger with basic configuration | Distributed tracing with APM integration | Medium - requires instrumentation changes |

#### Migration Strategy for Distributed Execution

The migration from single-node to distributed execution follows a phased approach that maintains compatibility with existing pipelines while gradually introducing cluster capabilities.

**Phase 1: Cluster-Aware Components**
Extend existing components with cluster communication capabilities while maintaining single-node operation as the default mode.

**Enhanced JobExecutor Interface:**

| Method | Current Signature | Distributed Extension | Migration Note |
|--------|------------------|----------------------|----------------|
| `execute_job` | `execute_job(job, global_env) -> bool` | `execute_job(job, global_env, worker_context=None) -> bool` | `worker_context=None` maintains backward compatibility |
| `get_execution_order` | `get_execution_order() -> List[List[str]]` | `get_execution_order(resource_constraints=None) -> List[List[str]]` | Resource-aware scheduling optional |

**Phase 2: Worker Node Registration**
Implement worker node discovery and registration without changing job execution logic.

```python
# Worker node registration protocol
class WorkerRegistration:
    def __init__(self, node_id: str, capabilities: Dict[str, Any]):
        self.node_id = node_id
        self.capabilities = capabilities
        self.heartbeat_interval = 30  # seconds
        self.last_heartbeat = None
    
    def register_with_coordinator(self, coordinator_endpoint: str) -> bool:
        # TODO: Send registration request with node capabilities
        # TODO: Establish heartbeat communication channel
        # TODO: Wait for assignment acknowledgment
        pass
```

**Phase 3: Gradual Job Distribution**
Begin distributing jobs to worker nodes while maintaining single-node execution for critical pipelines.

#### Security Enhancement Implementation

The security enhancements integrate with existing component interfaces through secure injection patterns that don't require changes to job execution logic.

**Secret Management Integration Points:**

| Integration Point | Current Implementation | Security Extension | Code Changes Required |
|------------------|----------------------|-------------------|----------------------|
| Environment variables | Direct injection in `execute_step` | Secret broker proxy | Minimal - add secret resolution layer |
| Artifact access | Direct file system access | Encrypted storage with access controls | Medium - requires storage backend changes |
| External API calls | Direct HTTP requests from jobs | Egress filtering and monitoring | Low - network policy configuration |

**Secure Secret Injection Example:**

```python
# Enhanced environment setup with secret management
def setup_secure_environment(step: PipelineStep, job_env: Dict[str, str]) -> Dict[str, str]:
    """Setup job environment with secure secret injection."""
    secure_env = job_env.copy()
    
    # TODO: Identify secret references in environment variables
    # TODO: Request short-lived credentials from SecretBroker
    # TODO: Inject credentials with automatic cleanup registration
    # TODO: Return environment with secrets resolved but not logged
    
    # Hint: Use pattern matching to identify ${SECRET:name} references
    # Hint: Coordinate with SecretBroker for credential lifecycle management
    pass
```

#### Monitoring Integration Architecture

The monitoring extensions integrate through event emission patterns that don't disrupt existing component functionality.

**Metrics Collection Integration:**

```python
# Enhanced component with metrics emission
class MonitoredJobExecutor(JobExecutor):
    def __init__(self, metrics_collector: MetricsCollector = None):
        super().__init__()
        self.metrics = metrics_collector or NullMetricsCollector()
    
    def execute_job(self, job: PipelineJob, global_env: Dict[str, str]) -> bool:
        """Execute job with comprehensive metrics collection."""
        start_time = time.time()
        
        # TODO: Emit job start metric with labels (pipeline, job name, worker)
        # TODO: Execute job using parent implementation
        # TODO: Calculate execution duration and resource usage
        # TODO: Emit completion metrics with success/failure status
        # TODO: Return execution result
        
        # Hint: Use context managers for automatic metric cleanup
        # Hint: Include resource usage metrics from container runtime
        pass
```

**Milestone Checkpoint for Extensions:**

After implementing each extension category, verify the following capabilities:

1. **Distributed Execution**: Deploy a multi-node cluster and verify that jobs distribute across workers while maintaining artifact flow and deployment capabilities
2. **Security Enhancements**: Execute a pipeline that uses secrets and verify that credentials never appear in logs or persist on worker nodes
3. **Monitoring**: Generate load on the system and verify that metrics, traces, and alerts provide actionable insights for operations teams

The extension architecture maintains backward compatibility while providing clear upgrade paths for organizations ready to adopt enterprise-scale capabilities.


## Glossary

> **Milestone(s):** This section provides essential terminology and definitions that support understanding across all milestones (1-4), serving as a comprehensive reference for CI/CD concepts and technical vocabulary.

Think of this glossary as the **common language dictionary** for our CI/CD system. Just as software engineers need to share precise vocabulary to collaborate effectively, this glossary ensures that every term has a clear, unambiguous definition. When debugging a failed deployment or discussing system architecture, everyone can reference these definitions to avoid miscommunication and build shared understanding.

### Core CI/CD Terminology

The foundation of our CI/CD system rests on several key concepts that form the vocabulary for all subsequent discussions. These terms represent the essential building blocks that every team member must understand to work effectively with the pipeline system.

| Term | Definition | Context |
|------|------------|---------|
| **Pipeline Definition** | YAML structure describing stages, jobs and dependencies that defines the complete automation workflow from source code to deployment | Core input format that drives all pipeline execution |
| **Job Dependency Graph** | DAG representation of job execution order that ensures correct sequencing and enables parallel execution where possible | Mathematical foundation for determining execution order in Milestone 1 |
| **Container Isolation** | Using Docker for secure job execution that provides consistent environments and prevents interference between jobs | Security and reproducibility mechanism in Milestone 2 |
| **Artifact Management** | Storage and transfer of build outputs between pipeline stages with integrity verification and lifecycle management | Bridge between jobs that enables data flow in Milestone 3 |
| **Deployment Strategy** | Method for releasing software that balances speed, safety, and rollback capabilities | Risk management approach for production releases in Milestone 4 |
| **Content-Addressable Storage** | Storage using content hashes as identifiers that enables deduplication and integrity verification | Fundamental storage pattern that prevents corruption and saves space |
| **Execution State** | Runtime tracking of pipeline progress that maintains current status and enables recovery after failures | State machine that coordinates all pipeline activities |
| **Variable Substitution** | Template engine for pipeline parameterization that enables reusable pipeline definitions across environments | Configuration management that reduces duplication and errors |

### State Management and Lifecycle

State management forms the backbone of reliable pipeline execution. Every component in our system maintains state machines that track progress, detect failures, and coordinate recovery. Understanding these states and their transitions is crucial for debugging issues and ensuring system reliability.

| Term | Definition | Usage |
|------|------------|-------|
| **State Machine** | Formal model of job status transitions that defines valid state changes and prevents invalid operations | Ensures consistent behavior and enables reliable error handling |
| **Flow State** | Pipeline execution lifecycle state that tracks overall progress from initialization to completion | High-level coordination mechanism in `FlowState` enum |
| **Job Status** | Current execution status of individual jobs using `JobStatus` enum values | Detailed tracking within `PipelineJob` for monitoring and dependencies |
| **Deployment Status** | Current state of deployment operations using `DeploymentStatus` enum | Progress tracking for complex deployment strategies |
| **Circuit State** | Circuit breaker state using `CircuitState` enum to prevent cascading failures | Fault tolerance mechanism that stops retrying failed operations |
| **Verification Result** | Artifact integrity check status using `VerificationResult` enum | Data quality assurance during artifact transfer |

> **Design Insight**: State machines aren't just academic concepts—they're practical tools for building reliable systems. When a job transitions from RUNNING to FAILED, the state machine ensures that dependent jobs don't start, logs are properly closed, and cleanup occurs. Without formal state management, systems become unpredictable and hard to debug.

### Dependency Management and Execution Order

The complexity of modern software builds requires sophisticated dependency management that can handle parallel execution while respecting ordering constraints. Our system uses graph theory algorithms to solve these scheduling problems automatically.

| Term | Definition | Technical Detail |
|------|------------|------------------|
| **Topological Sorting** | Algorithm for execution order that produces a linear ordering of jobs respecting all dependency constraints | Implemented in `get_execution_order()` method using Kahn's algorithm |
| **Circular Dependency Detection** | Finding and reporting dependency cycles that would cause infinite waiting between jobs | Prevents deadlocks by validating DAG structure during parsing |
| **Variable Scoping** | Hierarchy for resolving variable references with global, pipeline, and job-level precedence | Enables configuration inheritance while allowing specific overrides |
| **Expression Evaluation** | Processing complex variable substitutions including conditionals and function calls | Powers dynamic pipeline behavior based on runtime conditions |
| **Dependency Resolution** | Process of determining which artifacts and jobs a given job requires before execution | Critical for artifact download and workspace preparation |

### Error Handling and Recovery Patterns

Robust error handling distinguishes production-ready systems from prototypes. Our CI/CD system implements multiple layers of error detection, categorization, and recovery to handle the inevitable failures in distributed systems.

| Term | Definition | Implementation |
|------|------------|----------------|
| **Failure Category** | Classification of errors using `FailureCategory` enum for appropriate response selection | Enables automated recovery by matching failure type to recovery strategy |
| **Exponential Backoff** | Retry delay strategy doubling delay between attempts to avoid overwhelming failing services | Implemented in `ExponentialBackoffRetry` class with jitter and maximum delay |
| **Circuit Breaker** | Pattern preventing cascading failures by disabling retries after threshold is exceeded | `CircuitBreaker` class tracks failure rates and temporarily stops calling failing services |
| **Thundering Herd** | Problem when many processes retry simultaneously, overwhelming recovering services | Mitigated through jitter in retry timing and circuit breaker coordination |
| **Grace Period** | Time for clean shutdown before force termination allowing processes to save state | Configured in deployment strategies for safe instance replacement |
| **Graduated Response** | Escalating recovery strategies based on failure persistence from automatic to manual intervention | Implemented in `RecoveryManager` with escalation thresholds |

### Storage and Artifact Management

Efficient storage management becomes critical as CI/CD systems scale to handle large codebases and frequent builds. Our artifact management system implements content-addressable storage with sophisticated retention policies to balance performance, reliability, and cost.

| Term | Definition | Technical Implementation |
|------|------------|-------------------------|
| **Storage Backend** | File system organization layer that manages physical artifact storage with hash-based directories | Abstracts storage details and enables future cloud storage integration |
| **Reference Counting** | Tracking how many jobs currently depend on an artifact to prevent premature deletion | Maintained in `ArtifactInfo.reference_count` field with atomic updates |
| **Streaming Verification** | Calculating integrity checksums while transferring data to avoid memory exhaustion | Implemented in `verify_during_transfer()` with configurable chunk sizes |
| **Multi-Factor Retention** | Retention decisions based on multiple criteria including age, references, and access patterns | `RetentionPolicy` supports complex rules for different artifact types |
| **Artifact Lifecycle** | Creation, upload, storage, consumption, and retention phases of build outputs | Complete workflow from `store_artifact()` through `delete_artifact()` |
| **Integrity Verification** | Checksum-based validation using SHA-256 hashes to detect corruption during storage and transfer | Prevents silent data corruption that could compromise builds |

> **Critical Insight**: Content-addressable storage isn't just about deduplication—it fundamentally changes how we think about data integrity. When two jobs produce identical artifacts, they get the same hash. If we later discover corruption in that artifact, we know exactly which builds were affected. This makes forensic analysis much easier.

### Deployment and Traffic Management

Modern deployment strategies require sophisticated traffic management capabilities that can gradually shift load while monitoring system health. Our deployment system implements multiple strategies with automatic rollback capabilities.

| Term | Definition | Strategy Details |
|------|------------|------------------|
| **Rolling Deployment** | Incremental updates with health validation that replaces instances one at a time | Minimizes service disruption but provides slower rollback |
| **Blue-Green Deployment** | Atomic traffic switching between environments after complete environment preparation | Enables instant rollback but requires double resource allocation |
| **Canary Deployment** | Gradual traffic shifting with monitoring that incrementally validates new versions under real load | Provides early problem detection but requires sophisticated monitoring |
| **Health Check Validation** | Multi-dimensional readiness assessment using `HealthCheckConfig` with configurable thresholds | Ensures instances are truly ready before receiving traffic |
| **Traffic Routing** | Load balancer configuration managed through `update_traffic_weights()` calls | Abstracts infrastructure differences across cloud providers |
| **Atomic Switchover** | Instant traffic redirection that changes all user requests simultaneously to new version | Critical for blue-green deployments to avoid mixed version states |
| **Progressive Validation** | Graduated exposure under real load with automatic rollback triggers | Canary strategy that increases traffic based on success metrics |
| **Blast Radius** | Scope of impact when deployment issues occur, controlled through deployment strategy selection | Risk management consideration for deployment planning |
| **Environment Drift** | Differences between blue and green environments that can cause deployment failures | Mitigated through infrastructure-as-code and configuration management |

### Monitoring and Observability

Effective monitoring provides the feedback loops necessary for automated decision-making in deployment strategies and failure recovery. Our system implements comprehensive observability with structured logging and metrics collection.

| Term | Definition | Implementation Detail |
|------|------------|----------------------|
| **Structured Logging** | JSON-formatted log output with correlation IDs and component context | Enables automated log analysis and distributed request tracing |
| **Correlation ID** | Unique identifier linking related events across components for request tracing | Propagated through all system calls to enable end-to-end visibility |
| **Event Communication** | Component coordination mechanism using publish-subscribe patterns | `EventDispatcher` enables loose coupling between system components |
| **Behavioral Analysis** | Monitoring user activity patterns for security threats and performance optimization | Advanced feature for detecting anomalous pipeline usage |
| **Anomaly Detection** | Machine learning-based identification of unusual system behavior | Future extension for predictive failure detection |
| **Business Intelligence** | Connecting technical metrics to business outcomes for stakeholder reporting | Provides ROI metrics for CI/CD system investment |

### Security and Compliance

Security considerations permeate every aspect of CI/CD system design, from container isolation to secret management. Our system implements defense-in-depth strategies to protect against various threat vectors.

| Term | Definition | Security Implication |
|------|------------|---------------------|
| **Supply Chain Risk** | Security risks from third-party dependencies in build processes | Mitigated through dependency scanning and container image validation |
| **Zero-Trust Network** | Security model requiring authentication for all system interactions | Future architecture where no component trusts others by default |
| **Secret Lifecycle Management** | Comprehensive handling of sensitive data from creation to revocation | Prevents credential exposure through automated rotation and access controls |
| **Container Scanning** | Automated vulnerability detection in Docker images before deployment | Prevents deployment of images with known security vulnerabilities |
| **Access Control** | Fine-grained permissions for pipeline operations and artifact access | Role-based system controlling who can trigger deployments and access artifacts |
| **Audit Trail** | Complete record of all system actions for compliance and forensic analysis | Immutable log of all pipeline executions, deployments, and configuration changes |

### Distributed Systems Concepts

As CI/CD systems scale, they inevitably become distributed systems with all the associated complexity. Understanding these concepts is essential for building reliable, scalable pipeline infrastructure.

| Term | Definition | Distributed Systems Context |
|------|------------|----------------------------|
| **Distributed Execution** | Coordinating pipeline work across multiple worker nodes for horizontal scalability | Implemented through `ClusterCoordinator` and `WorkerNode` components |
| **Horizontal Scaling** | Adding more worker nodes to increase system capacity rather than upgrading existing hardware | Enables cost-effective scaling for large development teams |
| **Vector Clocks** | Distributed system technique for ordering events across nodes without synchronized time | Advanced feature for coordinating distributed pipeline execution |
| **Eventual Consistency** | Temporary inconsistencies acceptable with bounded convergence time | Trade-off between performance and consistency in distributed artifact storage |
| **Write-Ahead Logging** | Record operations before execution for recovery after failures | Ensures pipeline state can be recovered even after coordinator crashes |
| **Consensus Protocol** | Algorithm for distributed nodes to agree on system state | Required for distributed deployment coordination and failover |

### Performance and Scalability

Performance optimization becomes critical as CI/CD systems handle increasingly complex builds and larger development teams. Our system implements multiple strategies for managing resource utilization and scaling bottlenecks.

| Term | Definition | Performance Impact |
|------|------------|-------------------|
| **Resource Manager** | Tracks resource usage across cluster to optimize job placement | Prevents resource contention and maximizes hardware utilization |
| **Work Stealing Balancer** | Redistributes work from overloaded nodes to maintain optimal throughput | `WorkStealingBalancer` automatically rebalances work across cluster |
| **Streaming I/O** | Processing data in chunks to handle large files without memory exhaustion | Critical for artifact transfer and log processing at scale |
| **Content Deduplication** | Sharing identical artifacts across multiple pipelines to reduce storage costs | Content-addressable storage automatically deduplicates common dependencies |
| **Cache Invalidation** | Strategies for determining when cached artifacts are no longer valid | Balances build speed with correctness when dependencies change |
| **Backpressure** | Mechanisms for slowing down producers when consumers cannot keep up | Prevents memory exhaustion and system instability under high load |

### Testing and Quality Assurance

Comprehensive testing strategies ensure that our CI/CD system itself is reliable and can be safely evolved. Our testing approach combines multiple levels of validation with automated verification.

| Term | Definition | Testing Context |
|------|------------|-----------------|
| **Unit Testing** | Testing components in isolation with mocked dependencies | Validates individual component behavior without external dependencies |
| **Integration Testing** | Testing component interactions and end-to-end workflows | Verifies that components work correctly together |
| **End-to-End Testing** | Complete workflow validation from YAML parsing through deployment | Ensures entire pipeline system functions correctly |
| **Test Infrastructure** | Controlled environment for testing with mock services and data | `IntegrationTestHarness` provides consistent testing environment |
| **Mock Components** | Simulated dependencies for testing that behave predictably | Enables testing error scenarios and edge cases |
| **Milestone Checkpoints** | Validation criteria for implementation progress with specific acceptance tests | Ensures each milestone delivers working functionality |
| **Chaos Engineering** | Deliberately introducing failures to test system resilience | Advanced testing technique for validating error handling and recovery |

### Debugging and Troubleshooting

Effective debugging capabilities are essential for maintaining CI/CD systems in production. Our system provides comprehensive debugging tools and diagnostic information to quickly identify and resolve issues.

| Term | Definition | Debugging Application |
|------|------------|----------------------|
| **Container Debugging** | Systematic approach to diagnosing Docker container execution issues | `ContainerDebugger` captures complete container state for analysis |
| **State Transition Tracking** | Monitoring and validating state machine transitions for debugging | `StateTransitionTracker` identifies stuck or invalid state changes |
| **Exit Code Analysis** | Mapping container exit codes to specific failure categories | Automated diagnosis of common container execution problems |
| **Distributed Tracing** | Tracking request flow through distributed system components | Future capability for debugging complex distributed pipeline executions |
| **Performance Profiling** | Identifying bottlenecks and resource usage patterns | Tools for optimizing pipeline execution performance |
| **Log Aggregation** | Collecting and correlating logs from multiple components | Centralized view of system behavior across all components |

> **Debugging Philosophy**: The best debugging tools are those that help you understand what the system was *trying* to do, not just what it actually did. Our debugging infrastructure captures intent (from pipeline definitions) alongside execution traces, making it much easier to spot where reality diverged from expectations.

### Advanced Features and Extensions

While not part of the core implementation, understanding these advanced concepts helps with system evolution and integration with enterprise environments.

| Term | Definition | Future Applicability |
|------|------------|---------------------|
| **Multi-Tenancy** | Isolating multiple teams or projects within shared infrastructure | Required for enterprise deployments with multiple development teams |
| **Federation** | Connecting multiple CI/CD clusters for geographic distribution | Enables global development teams with regional infrastructure |
| **Policy Engine** | Rule-based system for enforcing organizational standards and compliance | Automates security and quality gates without manual oversight |
| **Cost Optimization** | Automatic resource scaling and job scheduling to minimize infrastructure costs | Critical for cloud deployments with usage-based pricing |
| **Disaster Recovery** | Procedures and automation for recovering from major system failures | Business continuity planning for mission-critical deployment pipelines |
| **Compliance Framework** | Systematic approach to meeting regulatory requirements like SOC2 or GDPR | Essential for organizations in regulated industries |

### Data Structures and Types

Understanding the key data structures provides the foundation for implementing and extending the CI/CD system. These types form the core vocabulary for all code discussions.

| Type Name | Purpose | Key Fields |
|-----------|---------|------------|
| **PipelineDefinition** | Top-level container for complete pipeline specification | name, jobs, global_env, created_at, timeout |
| **PipelineJob** | Individual job within pipeline with execution state | name, steps, depends_on, artifacts, environment, status, logs |
| **PipelineStep** | Single execution unit within job | name, script, image, environment, timeout, retry_count, working_directory |
| **ArtifactInfo** | Metadata for stored build artifacts | content_hash, original_name, size_bytes, created_at, reference_count |
| **DeploymentTarget** | Destination environment for deployments | name, url, version, status, health_check_url, metadata |
| **RetentionPolicy** | Rules for artifact lifecycle management | policy_name, max_age_days, size_threshold_gb, artifact_pattern |
| **PipelineError** | Structured error information with recovery hints | message, category, component, correlation_id, recovery_hint |
| **ExecutionState** | Runtime pipeline coordination state | Manages job scheduling and dependency resolution |
| **FlowEvent** | System event for component coordination | event_type, component, timestamp, data, correlation_id |

### Implementation Constants and Configuration

System configuration through well-defined constants enables tuning and customization while maintaining reasonable defaults for typical deployments.

| Constant | Value/Purpose | Configuration Context |
|----------|---------------|----------------------|
| **DEFAULT_TIMEOUT** | 3600 seconds | Maximum execution time for jobs without explicit timeout |
| **DEFAULT_IMAGE** | ubuntu:20.04 | Container image used when step doesn't specify image |
| **DEFAULT_CHUNK_SIZE** | 65536 bytes | Buffer size for streaming artifact transfers |
| **MAX_STORAGE_GB** | Configurable storage quota | Prevents runaway storage usage from retention policy failures |
| **DEFAULT_HEARTBEAT_INTERVAL** | 30 seconds | Worker node health check frequency |
| **JOB_STATE_TRANSITIONS** | Valid state change matrix | Defines legal job status transitions for state machine validation |

This glossary serves as the authoritative reference for all terminology used throughout the CI/CD pipeline system. When in doubt about the precise meaning of any concept, refer back to these definitions to ensure consistent understanding across the development team.

### Implementation Guidance

Understanding CI/CD terminology is crucial for effective system implementation and team communication. This guidance provides practical approaches for building and maintaining the vocabulary knowledge needed for successful pipeline development.

**A. Terminology Management Recommendations:**

| Approach | Simple Option | Advanced Option |
|----------|---------------|-----------------|
| Documentation | Markdown glossary with search | Wiki with cross-references and examples |
| Code Comments | Inline term definitions | Generated documentation with term linking |
| Team Onboarding | Glossary review sessions | Interactive terminology quizzes |
| Consistency Checking | Manual code review | Automated terminology validation in CI |

**B. Recommended Documentation Structure:**

```
docs/
  glossary.md                 ← this comprehensive reference
  quick-reference.md          ← essential terms for daily use  
  architecture-decisions/     ← ADRs with terminology context
    001-state-machine-design.md
    002-artifact-storage.md
  troubleshooting/           ← problem-solution guides
    common-issues.md         ← uses consistent terminology
    debugging-checklist.md   ← references glossary terms
```

**C. Terminology Validation Tools:**

```python
# terminology_validator.py - validates consistent term usage across codebase
import re
from typing import Dict, List, Set

class TerminologyValidator:
    """Validates consistent usage of CI/CD terminology across documentation and code."""
    
    def __init__(self, glossary_file: str):
        self.approved_terms = self._load_glossary(glossary_file)
        self.deprecated_terms = {
            'build pipeline': 'pipeline definition',
            'job queue': 'execution state', 
            'artifact store': 'artifact management',
            'deploy script': 'deployment strategy'
        }
    
    def validate_document(self, filepath: str) -> List[str]:
        """Check document for terminology consistency issues."""
        # TODO: Parse document and identify CI/CD terms
        # TODO: Check against approved terms list  
        # TODO: Flag deprecated term usage
        # TODO: Suggest replacements for non-standard terms
        pass
    
    def generate_term_report(self, codebase_path: str) -> Dict[str, int]:
        """Generate usage frequency report for all terms."""
        # TODO: Scan all files in codebase
        # TODO: Count occurrences of each approved term
        # TODO: Identify most/least used terminology
        # TODO: Flag potential terminology drift
        pass

def setup_terminology_checking():
    """Configure pre-commit hooks for terminology validation."""
    # TODO: Install git pre-commit hook
    # TODO: Configure validation rules
    # TODO: Set up automated reports
    pass
```

**D. Team Communication Guidelines:**

```python
# communication_standards.py - guidelines for consistent team communication
class CommunicationStandards:
    """Standards for consistent terminology usage in team communication."""
    
    REQUIRED_CONTEXT = {
        'pipeline failure': ['job name', 'failure category', 'correlation ID'],
        'deployment issue': ['strategy type', 'target environment', 'health check status'], 
        'artifact problem': ['content hash', 'integrity status', 'reference count'],
        'performance concern': ['component name', 'metric type', 'threshold values']
    }
    
    ESCALATION_KEYWORDS = {
        'URGENT': ['data corruption', 'security breach', 'production down'],
        'HIGH': ['deployment failed', 'build broken', 'performance degraded'],
        'MEDIUM': ['test failure', 'warning threshold', 'capacity planning'],
        'LOW': ['documentation update', 'optimization opportunity', 'question']
    }
    
    def format_issue_report(self, issue_type: str, details: Dict) -> str:
        """Format issue report with consistent terminology."""
        # TODO: Validate issue_type against approved terms
        # TODO: Ensure required context is provided
        # TODO: Add correlation ID for tracking
        # TODO: Format using standard template
        pass
```

**E. Milestone Checkpoint - Terminology Mastery:**

After studying this glossary, team members should demonstrate understanding through:

1. **Terminology Quiz**: Answer questions about key concepts without referring to glossary
2. **Code Review Exercise**: Identify and correct inconsistent terminology in sample code
3. **Documentation Writing**: Write a technical explanation using only approved terminology
4. **Troubleshooting Scenario**: Diagnose a pipeline issue using precise vocabulary

Expected outcomes:
- Can distinguish between similar terms (e.g., "pipeline definition" vs "execution state")
- Uses consistent vocabulary in code comments and documentation
- Asks clarifying questions when encountering ambiguous terminology
- Contributes to terminology evolution through formal proposal process

**F. Common Terminology Pitfalls:**

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Mixing abstraction levels | Calling `PipelineJob` a "container" | Use precise type names from data model |
| Inventing synonyms | "Artifact cache" instead of "artifact management" | Stick to glossary terms even if verbose |
| Vague error descriptions | "The pipeline broke" | Use specific failure categories and component names |
| Inconsistent state naming | "Job done" vs "job completed" vs "job finished" | Use exact `JobStatus` enum values |

The glossary is a living document that should evolve as the system grows. All terminology changes should go through the same review process as code changes, ensuring that the shared vocabulary remains accurate and useful for the entire development team.

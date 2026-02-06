# Build Your Own Git: Design Document


## Overview

This system implements a version control system that uses content-addressable storage and a directed acyclic graph to track file changes over time. The key architectural challenge is designing an immutable object store where every piece of content is identified by its cryptographic hash, enabling efficient deduplication and integrity verification.


> This guide is meant to help you understand the big picture before diving into each milestone. Refer back to it whenever you need context on how components connect.


## Context and Problem Statement

> **Milestone(s):** This section establishes the foundational concepts underlying all milestones

### Mental Model: The Digital Filing Cabinet

Imagine you have a magical filing cabinet for all your documents. This isn't an ordinary filing cabinet—it has three extraordinary properties that make it perfect for managing the evolution of your work over time.

First, the filing cabinet **never loses anything**. Every version of every document you've ever stored remains perfectly preserved, accessible whenever you need it. You can ask to see the draft of your thesis from six months ago, or the version of your code from last Tuesday, and the cabinet instantly produces an exact copy. Nothing ever gets overwritten or accidentally deleted.

Second, the filing cabinet is **impossibly organized**. Instead of using arbitrary folder names or date-based filing systems that might conflict or become confusing, it uses a perfect organizational scheme: every document gets filed based on a mathematical fingerprint of its exact contents. Two identical documents, even if created years apart, automatically go to the same location. This means there's never any duplication—the cabinet stores each unique piece of content exactly once, no matter how many times you reference it.

Third, the filing cabinet **remembers the relationships** between documents. When you file a new version of your thesis that incorporates feedback, the cabinet automatically records that this new version is based on the previous draft. It builds a complete family tree of how your documents evolved, branched into different versions, and merged back together. You can trace the lineage of any document back to its origins and understand exactly how it came to be.

This magical filing cabinet is precisely what Git provides for software development. The **content-addressable storage** system acts as the perfect organizational scheme, using cryptographic hashes to ensure every piece of content has a unique, deterministic location. The **immutable object store** guarantees that once something is filed, it never disappears or changes. The **directed acyclic graph** of commits preserves the complete evolutionary history of your project, showing how each version builds upon previous work.

Unlike physical filing cabinets where you might run out of space or struggle to find documents, Git's digital filing cabinet scales effortlessly. The mathematical properties of the hash-based organization ensure that even projects with millions of files and decades of history remain fast and reliable. The distributed nature means every developer gets their own complete copy of this magical cabinet, yet all copies can synchronize and share their contents seamlessly.

> The fundamental insight that makes Git possible is that **content identity can be derived from content itself**. Instead of assigning arbitrary names or numbers to track different versions, Git computes a cryptographic fingerprint of each piece of content. This fingerprint serves as both the identity and the storage location, creating a self-organizing system that eliminates duplication and ensures integrity.

### Existing Version Control Approaches

The challenge of tracking changes to files over time has been approached in several fundamentally different ways throughout the history of software development. Each approach represents different trade-offs between simplicity, functionality, and complexity. Understanding these approaches illuminates why Git's content-addressable architecture emerged as the dominant solution for modern software development.

#### Simple File-Based Approaches

The most basic approach to version control involves manual copying and timestamping. Developers create copies of their entire project directory with names like `project-v1.0`, `project-v1.1-backup`, `project-final`, `project-final-REALLY-FINAL`. This approach is intuitive and requires no special tools, but it quickly becomes unworkable.

The fundamental problems with file copying are storage inefficiency and coordination impossibility. Each complete copy of the project consumes full disk space, even when changes affect only a few files. A project with 1000 files that undergoes 100 versions consumes space equivalent to 100,000 files, even if most files never changed. More critically, there's no systematic way to understand what changed between versions, merge contributions from multiple developers, or recover from mistakes. Finding specific changes requires manual comparison of directory trees, a process that becomes prohibitively expensive as project size grows.

Some developers attempt to solve the comparison problem by maintaining change logs—text files that document what modified in each version. However, these logs are manually maintained and inevitably drift out of sync with actual changes. The change logs themselves become another file that needs version control, creating a recursive problem. Additionally, manual change tracking is error-prone and time-consuming, reducing developer productivity and introducing opportunities for human error.

#### Centralized Version Control Systems

Centralized systems like Subversion (SVN) and Concurrent Versions System (CVS) emerged to solve the coordination and storage problems of manual file copying. These systems introduce a central server that stores the complete history of the project and coordinates access among multiple developers.

In the centralized model, the server maintains a linear sequence of numbered revisions. Each revision represents a snapshot of the entire project at a specific point in time. Developers work with local working copies that contain only the current version of files. When they want to make changes, they update their working copy to the latest revision from the server, make modifications locally, and commit changes back to the server. The server automatically assigns sequential revision numbers and stores the differences between consecutive revisions.

| Aspect | Centralized VCS Behavior | Advantages | Disadvantages |
|--------|-------------------------|------------|---------------|
| **Storage Model** | Server stores complete history, clients have working copies only | Efficient network usage, simple mental model | Single point of failure, requires network access |
| **Branching** | Branches are server-side directories, expensive to create | Clear branch visualization in repository structure | Branch creation requires server round-trip, discourages experimentation |
| **Merging** | Server coordinates all merges, maintains linear history when possible | Simplified conflict resolution workflow | Complex merges block other developers, limited merge strategies |
| **Network Dependency** | Most operations require server communication | Always synchronized with team, simplified backup | Offline work impossible, network latency affects productivity |
| **Access Control** | Server enforces permissions per-directory or per-file | Fine-grained security control, audit trails | Administrative overhead, inflexible workflows |

The centralized approach works well for small teams with reliable network connectivity and relatively simple branching needs. The linear revision numbering makes it easy to reference specific versions, and the central server provides a clear authoritative source of truth. However, centralized systems impose significant limitations on developer workflows and introduce architectural brittleness.

#### Distributed Version Control Systems

Distributed systems like Git, Mercurial, and Bazaar eliminate the central server bottleneck by giving every developer a complete copy of the project history. This architectural shift enables fundamentally new workflows while solving many limitations of centralized systems.

In the distributed model, every working copy is actually a complete repository containing the full history of the project. Developers can create branches, make commits, and view history entirely locally without network access. Synchronization between repositories happens through explicit push and pull operations that exchange sets of commits. There's no inherent hierarchy—any repository can serve as a coordination point, and different teams can use different topologies (centralized, peer-to-peer, hierarchical) based on their needs.

| Aspect | Distributed VCS Behavior | Advantages | Disadvantages |
|--------|-------------------------|------------|---------------|
| **Storage Model** | Every clone contains complete history | Full offline capability, no single point of failure | Larger initial clone size, complex synchronization |
| **Branching** | Branches are lightweight local references | Instant branch creation, encourages experimentation | Branch proliferation can become confusing |
| **Merging** | Advanced three-way merge algorithms with multiple strategies | Sophisticated conflict resolution, non-linear history support | Steeper learning curve, potential for complex merge conflicts |
| **Network Independence** | All operations work offline, sync when convenient | Flexible workflows, works with unreliable connectivity | Potential for repositories to diverge significantly |
| **Collaboration Models** | Supports multiple coordination topologies | Adaptable to different team structures and policies | Requires explicit coordination protocols |

The distributed approach excels in scenarios with complex branching needs, unreliable network connectivity, or large teams with diverse collaboration patterns. The complete local history enables powerful operations like bisecting to find bug introductions, sophisticated blame tracking, and experimental branching without coordination overhead.

#### Git's Content-Addressable Innovation

Git's specific innovation within the distributed model is the use of content-addressable storage based on cryptographic hashing. While other distributed systems like Mercurial use similar overall architectures, Git's object model provides unique advantages for integrity, deduplication, and performance.

Every piece of content in Git—whether file contents, directory structures, or commit metadata—is stored as an object identified by the SHA-1 hash of its contents. This creates several powerful properties that distinguish Git from other approaches:

**Automatic deduplication** occurs because identical content always produces identical hashes. If the same file appears in multiple commits or branches, Git stores it only once. This makes branching extremely lightweight compared to systems that copy files or store differences.

**Cryptographic integrity verification** happens automatically because any corruption changes the hash. Git can detect repository corruption immediately when accessing objects, providing stronger integrity guarantees than systems relying on file system integrity alone.

**Distributed consistency** emerges naturally because content hashes are identical across all repositories. Two developers who independently create identical commits will generate identical commit hashes, simplifying synchronization and conflict detection.

**Immutable history** is enforced by the hash chain structure. Because each commit's hash includes the hashes of its parent commits, changing any historical commit would require recomputing all subsequent hashes, making tampering easily detectable.

> **Decision: Content-Addressable Object Storage**
> - **Context**: Version control systems need to store file contents, directory structures, and metadata efficiently while ensuring integrity and enabling distributed synchronization
> - **Options Considered**: Sequential revision numbers (CVS/SVN), content-based addressing (Git), hybrid approaches (Mercurial)
> - **Decision**: Use SHA-1 hashes of content as both object identity and storage location
> - **Rationale**: Content-addressable storage provides automatic deduplication, cryptographic integrity, distributed consistency, and immutable history with minimal overhead
> - **Consequences**: Enables lightweight branching, efficient storage, strong integrity guarantees, but requires understanding of hash-based addressing and occasional hash collisions

The trade-offs of Git's approach become apparent in specific scenarios. The content-addressable model requires understanding of hash-based addressing, which has a steeper learning curve than sequential revision numbers. The complete history in every clone means initial repository size is larger than centralized systems. Complex branching and merging capabilities can lead to confusing repository states if not managed carefully.

However, for software development workflows involving frequent branching, distributed teams, or long-lived feature development, Git's advantages typically outweigh these costs. The ability to work completely offline while maintaining full version control capabilities, combined with the lightweight nature of Git's branching model, has made it the standard choice for most modern software projects.

#### Summary of Architectural Trade-offs

| Approach | Storage Efficiency | Branching Cost | Network Dependency | Learning Curve | Best Use Cases |
|----------|-------------------|----------------|-------------------|----------------|----------------|
| **File Copying** | Very Poor | Manual only | None | Minimal | Single developer, tiny projects |
| **Centralized** | Good | Medium-High | Required for most operations | Low-Medium | Small teams, simple workflows |
| **Distributed** | Excellent | Very Low | Optional | Medium-High | Complex projects, distributed teams |
| **Git Specifically** | Excellent | Minimal | Optional | High | Software development, open source |

The evolution from manual copying through centralized systems to distributed version control reflects the growing complexity of software development projects and the need for more sophisticated collaboration models. Git's content-addressable approach represents the current state of the art, providing the flexibility and power needed for modern software development while maintaining the performance and reliability required for large-scale projects.

Understanding these different approaches helps explain why Git is architected the way it is, and why certain design decisions that might seem complex in isolation actually solve fundamental problems that simpler approaches cannot address. The content-addressable object store that we will implement represents decades of evolution in version control system design, distilled into an elegant and powerful architecture.

### Implementation Guidance

The following implementation guidance provides concrete technical recommendations for building the foundational components described in this section. This guidance targets Python development and establishes the coding patterns we'll use throughout the project.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| **Hashing** | `hashlib.sha1()` from standard library | `cryptography` library with hash verification |
| **Compression** | `zlib` module from standard library | `lzma` for better compression ratios |
| **File I/O** | `pathlib.Path` with context managers | `os.scandir()` for high-performance directory traversal |
| **Binary Data** | `bytes` and `struct.pack/unpack` | `ctypes` for complex binary structures |
| **Command Line** | `argparse` for basic CLI | `click` for advanced command interfaces |
| **Testing** | `unittest` from standard library | `pytest` with fixtures and parametrization |

#### Recommended Project Structure

Organize your Git implementation to mirror the architectural components and support iterative development across the eight milestones:

```
git-implementation/
├── mygit/                          # Main package
│   ├── __init__.py                 # Package initialization
│   ├── cli.py                      # Command-line interface entry point
│   ├── repository.py               # Repository class and initialization
│   ├── objects/                    # Object store implementation
│   │   ├── __init__.py
│   │   ├── blob.py                 # Blob object handling
│   │   ├── tree.py                 # Tree object handling
│   │   ├── commit.py               # Commit object handling
│   │   └── store.py                # Object storage backend
│   ├── index/                      # Staging area implementation
│   │   ├── __init__.py
│   │   ├── index.py                # Index file format and operations
│   │   └── status.py               # Working directory status calculation
│   ├── refs/                       # Reference management
│   │   ├── __init__.py
│   │   ├── head.py                 # HEAD reference handling
│   │   └── branches.py             # Branch creation and management
│   ├── diff/                       # Diff algorithm implementation
│   │   ├── __init__.py
│   │   ├── myers.py                # Myers diff algorithm
│   │   └── unified.py              # Unified diff output format
│   ├── merge/                      # Merge implementation
│   │   ├── __init__.py
│   │   ├── three_way.py            # Three-way merge algorithm
│   │   └── conflicts.py            # Conflict detection and marking
│   └── utils/                      # Shared utilities
│       ├── __init__.py
│       ├── hash.py                 # Hash computation helpers
│       └── paths.py                # Path manipulation utilities
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── test_objects.py             # Object store tests
│   ├── test_index.py               # Index operation tests
│   ├── test_refs.py                # Reference management tests
│   ├── test_diff.py                # Diff algorithm tests
│   ├── test_merge.py               # Merge algorithm tests
│   └── integration/                # Integration tests
│       ├── __init__.py
│       └── test_workflows.py       # End-to-end workflow tests
├── examples/                       # Example repositories and scripts
│   └── sample_repo/                # Test repository for development
└── docs/                          # Documentation
    └── milestones/                 # Milestone-specific documentation
```

#### Core Infrastructure Components

Before implementing Git-specific logic, establish these foundational utilities that will be used throughout the project:

**Hash Computation Utility (utils/hash.py):**
```python
import hashlib
from typing import bytes

def compute_sha1(content: bytes) -> str:
    """
    Compute SHA-1 hash of content and return as 40-character hex string.
    
    Args:
        content: Raw bytes to hash
        
    Returns:
        40-character lowercase hex digest
        
    Example:
        >>> compute_sha1(b"hello world")
        '2aae6c35c94fcfb415dbe95f408b9ce91ee846ed'
    """
    hasher = hashlib.sha1()
    hasher.update(content)
    return hasher.hexdigest()

def compute_object_hash(object_type: str, content: bytes) -> str:
    """
    Compute Git object hash including type and size header.
    
    Git objects are hashed as: "{type} {size}\0{content}"
    
    Args:
        object_type: Git object type ("blob", "tree", "commit")
        content: Raw object content
        
    Returns:
        40-character hex hash that matches git hash-object
    """
    header = f"{object_type} {len(content)}".encode('ascii')
    full_content = header + b'\0' + content
    return compute_sha1(full_content)
```

**Path Utilities (utils/paths.py):**
```python
import os
from pathlib import Path
from typing import Optional

def find_git_directory(start_path: Path = None) -> Optional[Path]:
    """
    Find .git directory by walking up the directory tree.
    
    Args:
        start_path: Directory to start search from (default: current directory)
        
    Returns:
        Path to .git directory, or None if not found
    """
    if start_path is None:
        start_path = Path.cwd()
    
    current = start_path.resolve()
    while current != current.parent:
        git_dir = current / '.git'
        if git_dir.is_dir():
            return git_dir
        current = current.parent
    
    return None

def ensure_directory_exists(path: Path) -> None:
    """Create directory and all parent directories if they don't exist."""
    path.mkdir(parents=True, exist_ok=True)

def object_path_from_hash(git_dir: Path, object_hash: str) -> Path:
    """
    Convert object hash to file system path in .git/objects.
    
    Git stores objects as .git/objects/xx/yyyyyyyy where xx is first 2 hex chars.
    
    Args:
        git_dir: Path to .git directory
        object_hash: 40-character hex hash
        
    Returns:
        Path where object should be stored
        
    Example:
        >>> object_path_from_hash(Path('.git'), 'abc123...')
        Path('.git/objects/ab/c123...')
    """
    return git_dir / 'objects' / object_hash[:2] / object_hash[2:]
```

**Repository Class Skeleton (repository.py):**
```python
from pathlib import Path
from typing import Optional
from .utils.paths import find_git_directory, ensure_directory_exists

class Repository:
    """
    Represents a Git repository and provides access to its components.
    
    The Repository class serves as the main entry point for all Git operations,
    coordinating between the object store, index, references, and working directory.
    """
    
    def __init__(self, path: Path = None):
        """
        Initialize repository object.
        
        Args:
            path: Path to repository root (default: find from current directory)
        """
        self.git_dir = find_git_directory(path)
        if self.git_dir is None:
            raise ValueError("Not a git repository")
        
        self.work_tree = self.git_dir.parent
        
    @classmethod
    def init(cls, path: Path = None) -> 'Repository':
        """
        Initialize a new Git repository.
        
        Args:
            path: Directory to initialize (default: current directory)
            
        Returns:
            Repository object for the newly created repository
        """
        if path is None:
            path = Path.cwd()
        
        git_dir = path / '.git'
        
        # TODO 1: Check if .git already exists and handle appropriately
        # TODO 2: Create .git directory structure (objects, refs, etc.)
        # TODO 3: Create initial HEAD file pointing to refs/heads/master
        # TODO 4: Create config file with repository format version
        # TODO 5: Set appropriate permissions on .git directory
        
        return cls(path)
    
    def object_exists(self, object_hash: str) -> bool:
        """Check if object with given hash exists in object store."""
        # TODO: Implement object existence check
        pass
    
    def read_object(self, object_hash: str) -> tuple[str, bytes]:
        """
        Read and decompress object from object store.
        
        Returns:
            Tuple of (object_type, content)
        """
        # TODO: Implement object reading with decompression
        pass
    
    def write_object(self, object_type: str, content: bytes) -> str:
        """
        Write object to object store with compression.
        
        Returns:
            SHA-1 hash of the stored object
        """
        # TODO: Implement object writing with compression
        pass
```

#### Development Workflow Setup

**Environment Configuration:**
```python
# requirements.txt
pytest>=7.0.0
pytest-cov>=4.0.0
black>=22.0.0
mypy>=1.0.0

# .gitignore for your implementation
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
.pytest_cache/
.coverage
htmlcov/
.mypy_cache/
examples/*/
test_repos/
```

#### Milestone 1 Checkpoint: Repository Initialization

After implementing the repository initialization logic, verify your implementation with these tests:

**Manual Verification Steps:**
1. Run `python -m mygit init` in an empty directory
2. Verify `.git` directory exists with mode 755: `ls -la | grep .git`
3. Check directory structure: `find .git -type d | sort`
4. Verify HEAD contents: `cat .git/HEAD` should show `ref: refs/heads/master`
5. Test error handling: run `mygit init` again, should warn about existing repository

**Expected Directory Structure:**
```
.git/
├── objects/
│   ├── info/
│   └── pack/
├── refs/
│   ├── heads/
│   └── tags/
├── HEAD
└── config
```

**Unit Test Template:**
```python
def test_repository_initialization(tmp_path):
    """Test that repository initialization creates correct structure."""
    # Initialize repository
    repo = Repository.init(tmp_path)
    
    # Verify .git directory exists
    assert (tmp_path / '.git').is_dir()
    
    # Verify required subdirectories
    required_dirs = ['.git/objects', '.git/refs/heads', '.git/refs/tags']
    for dir_path in required_dirs:
        assert (tmp_path / dir_path).is_dir()
    
    # Verify HEAD file contents
    head_content = (tmp_path / '.git/HEAD').read_text().strip()
    assert head_content == 'ref: refs/heads/master'
    
    # Verify repository can be reopened
    repo2 = Repository(tmp_path)
    assert repo2.git_dir == tmp_path / '.git'
```

#### Common Development Pitfalls

⚠️ **Pitfall: Binary vs Text File Handling**
Git objects contain binary data (especially the null bytes in object headers), but many file operations default to text mode. Always open object files in binary mode (`'rb'`, `'wb'`) and use `bytes` objects for all content handling.

⚠️ **Pitfall: Platform Path Differences**
Git uses forward slashes for paths internally, but your implementation runs on different operating systems. Use `pathlib.Path` for all path manipulation and normalize paths when storing them in Git objects.

⚠️ **Pitfall: Hash Encoding Confusion**
SHA-1 hashes appear in three forms: binary (20 bytes), hex string (40 chars), and hex bytes (40 bytes). Be explicit about which format you're using and convert consistently using `.digest()`, `.hexdigest()`, and `.encode()`.

⚠️ **Pitfall: Directory Creation Race Conditions**
When multiple Git operations run simultaneously, directory creation can fail if another process creates the directory first. Use `mkdir(parents=True, exist_ok=True)` to handle this gracefully.

#### Language-Specific Hints for Python

**File I/O Patterns:**
- Use `with open(path, 'rb') as f:` for all object file operations
- Use `Path.read_bytes()` and `Path.write_bytes()` for simple file operations
- Use `os.makedirs(path, exist_ok=True)` for directory creation

**Binary Data Handling:**
- Use `struct.pack()` and `struct.unpack()` for binary index format
- Use `bytes.join()` for concatenating binary data
- Use `.encode('utf-8')` and `.decode('utf-8')` for string/bytes conversion

**Error Handling:**
- Catch `FileNotFoundError` for missing objects and references
- Catch `OSError` for file system permission issues
- Use custom exception classes like `ObjectNotFoundError` for Git-specific errors

This implementation foundation provides the scaffolding needed for all eight milestones. The modular structure allows you to implement components incrementally while maintaining clean separation of concerns. The utility functions handle cross-cutting concerns like hashing and path manipulation consistently across the entire codebase.


## Goals and Non-Goals

> **Milestone(s):** This section establishes the scope and boundaries for all eight milestones, from repository initialization through three-way merging

Building a complete Git implementation requires making deliberate choices about what to include and what to exclude. Git itself has evolved over nearly two decades and includes hundreds of commands, optimization strategies, and edge case handling that would overwhelm any learning project. Our implementation focuses on the core mechanisms that make Git fundamentally different from other version control systems: content-addressable storage, immutable object graphs, and distributed merge capabilities.

The key insight driving our scope decisions is that Git's power comes from a small set of elegant primitives that compose to create sophisticated behaviors. By implementing these primitives correctly, we gain deep understanding of how modern version control systems work at their foundation. Advanced features like remote synchronization, pack file compression, and complex merge strategies are all built on top of these same primitives.

### Functional Goals

Our Git implementation will support eight core operations that demonstrate the fundamental principles of content-addressable version control. These operations form a complete workflow from repository creation through collaborative development scenarios.

**Repository Lifecycle Management:**

| Operation | Command | Purpose | Key Learning |
|-----------|---------|---------|--------------|
| Repository Initialization | `Repository.init()` | Create `.git` directory structure with proper permissions and default configuration | Understanding Git's on-disk layout and reference system initialization |
| Repository Discovery | `find_git_directory()` | Locate the nearest `.git` directory by traversing parent directories | How Git commands work from any subdirectory within a repository |

The repository lifecycle operations establish the workspace where all other Git operations occur. Repository initialization demonstrates how Git creates its internal data structures, while repository discovery shows how Git maintains context across the entire project directory tree.

**Content Storage and Retrieval:**

| Operation | Command | Purpose | Key Learning |
|-----------|---------|---------|--------------|
| Blob Storage | `hash-object` | Compute SHA-1 hash and store file contents as compressed blob objects | Content-addressable storage principles and object deduplication |
| Object Retrieval | `cat-file` | Decompress and display stored objects by their hash | How immutable objects enable reliable content addressing |
| Tree Creation | `write-tree` | Build tree objects representing directory structure from staged files | Hierarchical directory representation using object references |
| Tree Inspection | `ls-tree` | Display tree object contents showing file modes, types, and hashes | Understanding how Git tracks file metadata and permissions |

Content operations form the foundation of Git's storage model. These operations demonstrate how Git transforms the familiar file system metaphor into an immutable, cryptographically-verified object store. The progression from individual file storage (blobs) to directory structure representation (trees) shows how Git builds complex data from simple primitives.

**History and Version Management:**

| Operation | Command | Purpose | Key Learning |
|-----------|---------|---------|--------------|
| Commit Creation | `commit-tree` | Create commit objects linking trees with metadata and parent pointers | How immutable history is built through directed acyclic graph structure |
| History Traversal | Navigate parent commit chains | Follow commit links to reconstruct project history | Understanding Git's approach to temporal relationships between versions |

History management operations demonstrate how Git creates tamper-evident project timelines. Unlike systems that store differences between versions, Git stores complete snapshots linked by cryptographic hashes, making history manipulation detectable and enabling powerful analysis capabilities.

**Branch and Reference Management:**

| Operation | Command | Purpose | Key Learning |
|-----------|---------|---------|--------------|
| Branch Creation | Create files in `.git/refs/heads/` | Associate human-readable names with commit hashes | How Git provides convenient aliases for navigating object graph |
| Branch Switching | Update `HEAD` reference | Change working directory context to different branch tip | Understanding symbolic references and detached HEAD states |
| Reference Resolution | Resolve symbolic and direct references | Convert branch names to specific commit hashes | How Git maintains consistency between human naming and content addressing |

Reference operations bridge the gap between Git's cryptographic object model and human workflow needs. These operations show how Git maintains the benefits of content-addressable storage while providing familiar branch-based development patterns.

**Staging and Workflow Management:**

| Operation | Command | Purpose | Key Learning |
|-----------|---------|---------|--------------|
| File Staging | `add` | Move files from working directory to staging area with blob creation | Understanding Git's three-tree architecture and incremental commit preparation |
| Status Calculation | `status` | Compare working directory, index, and HEAD to show modification states | How Git efficiently detects changes across multiple contexts |
| Index Management | Binary `.git/index` file operations | Maintain sorted list of staged files with metadata caching | Git's approach to performance optimization through cached file information |

Staging operations demonstrate Git's unique three-stage workflow that separates file modification, commit preparation, and history recording. This separation enables precise control over what changes enter the permanent record and supports complex development workflows.

**Change Analysis and Comparison:**

| Operation | Command | Purpose | Key Learning |
|-----------|---------|---------|--------------|
| Difference Calculation | `diff` | Compute line-by-line changes between file versions using Myers algorithm | Understanding how version control systems analyze and present textual changes |
| Unified Diff Output | Generate standard diff format | Present changes in human-readable format with context lines and hunk headers | Industry-standard approaches to change visualization |

Difference calculation operations reveal how Git analyzes changes between versions. Understanding diff algorithms provides insight into how all version control systems approach the fundamental problem of change detection and presentation.

**Collaborative Development Support:**

| Operation | Command | Purpose | Key Learning |
|-----------|---------|---------|--------------|
| Three-Way Merge | `merge` | Combine changes from two branches using common ancestor as merge base | Understanding Git's approach to collaborative development and conflict resolution |
| Merge Base Calculation | Find lowest common ancestor | Identify the optimal point for comparing divergent development lines | Graph algorithms applied to version control history |
| Conflict Detection | Identify overlapping changes | Recognize when automatic merging is impossible and manual resolution is required | How version control systems handle ambiguous merge scenarios |

Merge operations demonstrate Git's most sophisticated collaborative features. Three-way merging shows how distributed version control systems enable independent development while maintaining the ability to recombine work systematically.

> **Design Insight**: These eight operation categories form a complete development workflow while teaching the core principles that make Git unique: content addressing, immutable history, three-tree architecture, and systematic merge handling. Each operation builds on primitives established in previous operations, creating a natural learning progression.

### Explicit Non-Goals

Our implementation deliberately excludes advanced Git features that, while important for production use, would distract from learning the core principles. These exclusions allow us to focus on fundamental concepts without getting lost in optimization details or edge case handling.

**Remote Repository Operations:**

| Excluded Feature | Rationale | Learning Impact |
|------------------|-----------|-----------------|
| `clone`, `fetch`, `push`, `pull` | Remote operations require network protocols, authentication, and synchronization strategies | These are applications of local Git primitives rather than fundamental storage concepts |
| SSH/HTTPS transport protocols | Protocol implementation involves security, networking, and error handling unrelated to version control | Understanding local operations provides foundation for later network protocol study |
| Remote reference tracking | Tracking multiple remote repositories adds complexity without teaching core storage principles | Local branch operations demonstrate the same reference management concepts |

Remote operations, while essential for collaborative development, are fundamentally applications of the local operations we do implement. Understanding how Git stores and manipulates objects locally provides the foundation needed to later understand how those same objects are synchronized across repositories.

**Performance Optimizations:**

| Excluded Feature | Rationale | Learning Impact |
|------------------|-----------|-----------------|
| Pack files and delta compression | Pack files are storage optimizations that obscure the object model during learning | Understanding individual object storage first makes pack file optimizations more meaningful later |
| Index version 2+ extensions | Newer index formats add performance features without changing fundamental staging concepts | Learning basic index operations provides foundation for understanding extensions |
| Parallel object processing | Concurrency optimizations complicate core algorithm understanding | Sequential implementations are easier to debug and understand during learning |
| Memory-mapped file access | Low-level optimizations distract from high-level version control concepts | Standard file I/O demonstrates the same concepts with clearer code |

Performance optimizations, while crucial for large repositories, often obscure the underlying algorithms during initial learning. Our implementation prioritizes clarity and correctness over performance, making the fundamental concepts more accessible.

> **Critical Exclusion Rationale**: Pack files deserve special mention as they represent one of Git's most important optimizations. However, pack files are essentially a compression layer over the object model we do implement. Learning individual object storage first provides the conceptual foundation needed to understand why pack files exist and how they work.

**Advanced Merge and Conflict Resolution:**

| Excluded Feature | Rationale | Learning Impact |
|------------------|-----------|-----------------|
| Octopus merges (more than two parents) | Multiple parent merges are rare and don't teach additional concepts beyond two-parent merging | Two-parent merges demonstrate all the fundamental merge base and conflict resolution concepts |
| Rename detection during merge | Rename tracking requires sophisticated file similarity algorithms | File-level merging demonstrates core three-way merge concepts without algorithmic complexity |
| Advanced merge strategies (`subtree`, `ours`, `theirs`) | Alternative merge strategies are specialized tools for specific workflows | Standard recursive three-way merge teaches the fundamental merge principles |
| Interactive conflict resolution | User interface concerns distract from merge algorithm understanding | Conflict marker generation demonstrates conflict detection principles |

Advanced merge features solve specific workflow problems but don't teach additional fundamental concepts beyond three-way merging with conflict detection. Our implementation covers the core merge principles that underlie all of Git's merge strategies.

**Extended Object Model Features:**

| Excluded Feature | Rationale | Learning Impact |
|------------------|-----------|-----------------|
| Annotated tags | Tags are references with additional metadata, not fundamental object types | Basic reference handling demonstrates the same storage and retrieval concepts |
| Signed commits and tags | Cryptographic signatures are security features independent of version control concepts | Object integrity through SHA-1 hashing demonstrates core verification principles |
| Git attributes and filters | Content filtering is a layer above the object model | Understanding basic object storage provides foundation for filter comprehension |
| Submodules | Submodules are repository composition features built on basic Git operations | Core repository operations demonstrate the primitives submodules use |

Extended object model features are important for production workflows but are built using the same storage primitives we do implement. Understanding blob, tree, and commit objects provides the foundation for understanding how these extensions work.

**File System and Platform Features:**

| Excluded Feature | Rationale | Learning Impact |
|------------------|-----------|-----------------|
| Symbolic link handling | Platform-specific file system features add complexity without teaching version control concepts | Regular file handling demonstrates core content tracking principles |
| File permission preservation | File metadata tracking is important but secondary to content versioning concepts | Basic mode bits demonstrate metadata handling without platform complexity |
| Large file support (LFS) | Large files require external storage systems beyond Git's scope | Understanding regular file storage shows why large file extensions exist |
| Cross-platform line ending handling | Text file normalization is important but orthogonal to version control storage | Binary content handling demonstrates core storage without text processing complexity |

File system integration features solve important practical problems but don't teach the fundamental version control concepts that make Git unique. Our implementation focuses on the storage and merge algorithms that form Git's conceptual core.

**Configuration and User Interface:**

| Excluded Feature | Rationale | Learning Impact |
|------------------|-----------|-----------------|
| Global and local configuration files | Configuration management is important but separate from core version control algorithms | Hardcoded defaults allow focus on storage and merge logic |
| Command-line argument parsing | User interface concerns distract from algorithmic understanding | Direct function calls demonstrate algorithm behavior more clearly |
| Error message localization | User experience features don't teach version control concepts | Simple error messages focus attention on understanding failure modes |
| Interactive staging (`add -p`) | User interface features for workflow convenience | Basic staging demonstrates the fundamental three-tree architecture |

Configuration and interface features improve user experience but don't teach the fundamental algorithms that make version control systems work. Our implementation uses direct function calls and simple interfaces to keep focus on the core concepts.

> **Architecture Decision: Scope Boundary Principle**
> - **Context**: Git includes hundreds of features accumulated over decades of development, making complete implementation impractical for learning
> - **Options Considered**: 
>   1. Implement minimal subset (init, add, commit only)
>   2. Implement core workflow with basic collaboration (our choice)
>   3. Attempt complete Git compatibility
> - **Decision**: Implement complete local workflow plus basic three-way merging
> - **Rationale**: This scope teaches all fundamental Git concepts (content addressing, immutable history, three-tree architecture, collaborative merging) without overwhelming complexity. Each operation builds naturally on previous ones, creating clear learning progression.
> - **Consequences**: Learners understand Git's conceptual foundation and can later study advanced features with solid grounding. Implementation remains manageable while covering realistic development workflows.

### Implementation Guidance

Building a Git implementation requires careful technology choices and project organization to manage complexity while maintaining focus on core learning objectives. The following recommendations balance educational value with practical implementation concerns.

**A. Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| SHA-1 Hashing | `hashlib.sha1()` built-in library | Custom SHA-1 implementation for deeper understanding |
| File Compression | `zlib` standard library for object compression | Custom compression algorithms |
| Binary Parsing | `struct` module for index file format | Custom binary serialization framework |
| File System Operations | `pathlib.Path` for cross-platform path handling | Direct `os` module calls with manual path construction |
| Diff Algorithm | Myers algorithm implementation from scratch | External diff library (`difflib` for validation) |
| Command Interface | Direct function calls for simplicity | Full command-line parser (`argparse`) |

The simple options provide clear, focused implementations that highlight the core concepts without unnecessary complexity. Advanced options are available for learners who want deeper understanding of specific components.

**B. Recommended File Structure:**

```
build-your-own-git/
├── git/
│   ├── __init__.py
│   ├── repository.py          # Repository class and initialization
│   ├── objects/
│   │   ├── __init__.py
│   │   ├── blob.py           # Blob object handling
│   │   ├── tree.py           # Tree object creation and parsing
│   │   ├── commit.py         # Commit object operations
│   │   └── store.py          # Object store implementation
│   ├── refs/
│   │   ├── __init__.py
│   │   ├── reference.py      # Reference resolution and management
│   │   └── symbolic.py       # HEAD and symbolic reference handling
│   ├── index/
│   │   ├── __init__.py
│   │   ├── staging.py        # Add and remove operations
│   │   └── status.py         # Status calculation
│   ├── diff/
│   │   ├── __init__.py
│   │   ├── myers.py          # Myers diff algorithm
│   │   └── unified.py        # Unified diff output formatting
│   └── merge/
│       ├── __init__.py
│       ├── three_way.py      # Three-way merge implementation
│       └── conflicts.py      # Conflict detection and marking
├── tests/
│   ├── test_repository.py
│   ├── test_objects.py
│   ├── test_refs.py
│   ├── test_index.py
│   ├── test_diff.py
│   └── test_merge.py
├── examples/
│   ├── basic_workflow.py     # Demonstrates typical Git workflow
│   └── merge_scenario.py     # Shows merge and conflict resolution
└── README.md
```

This structure separates concerns logically while keeping related functionality together. Each major component lives in its own module with clear dependencies between layers.

**C. Infrastructure Starter Code:**

**Repository Discovery and Initialization** (Complete implementation):

```python
# git/repository.py
import os
from pathlib import Path
from typing import Optional

class Repository:
    """Represents a Git repository with working directory and git directory."""
    
    def __init__(self, git_dir: Path, work_tree: Path):
        self.git_dir = git_dir
        self.work_tree = work_tree
    
    @classmethod
    def init(cls, path: Path) -> 'Repository':
        """Initialize a new Git repository at the given path."""
        git_dir = path / '.git'
        
        # Create directory structure
        git_dir.mkdir(exist_ok=True)
        (git_dir / 'objects').mkdir(exist_ok=True)
        (git_dir / 'refs' / 'heads').mkdir(parents=True, exist_ok=True)
        (git_dir / 'refs' / 'tags').mkdir(exist_ok=True)
        
        # Create HEAD file pointing to default branch
        head_file = git_dir / 'HEAD'
        head_file.write_text('ref: refs/heads/master\n')
        
        # Create basic config
        config_file = git_dir / 'config'
        config_content = """[core]
    repositoryformatversion = 0
    filemode = true
    bare = false
"""
        config_file.write_text(config_content)
        
        return cls(git_dir, path)
    
    def object_path_from_hash(self, object_hash: str) -> Path:
        """Convert SHA-1 hash to object file path."""
        return self.git_dir / 'objects' / object_hash[:2] / object_hash[2:]

def find_git_directory(start_path: Path) -> Optional[Path]:
    """Find .git directory by walking up the directory tree."""
    current = start_path.resolve()
    while current != current.parent:
        git_dir = current / '.git'
        if git_dir.is_dir():
            return git_dir
        current = current.parent
    return None
```

**SHA-1 and Object Hashing Utilities** (Complete implementation):

```python
# git/objects/store.py
import hashlib
import zlib
from pathlib import Path

# Constants
SHA1_HEX_LENGTH = 40
OBJECT_HEADER_FORMAT = '{type} {size}\0{content}'

def compute_sha1(content: bytes) -> str:
    """Compute SHA-1 hash of raw bytes, returning hex string."""
    return hashlib.sha1(content).hexdigest()

def compute_object_hash(object_type: str, content: bytes) -> str:
    """Compute Git object hash with proper header."""
    header = f'{object_type} {len(content)}'.encode() + b'\0'
    full_content = header + content
    return compute_sha1(full_content)

def store_object(git_dir: Path, object_type: str, content: bytes) -> str:
    """Store a Git object and return its hash."""
    object_hash = compute_object_hash(object_type, content)
    
    # Create object file path
    obj_dir = git_dir / 'objects' / object_hash[:2]
    obj_dir.mkdir(exist_ok=True)
    obj_file = obj_dir / object_hash[2:]
    
    # Write compressed object
    header = f'{object_type} {len(content)}'.encode() + b'\0'
    full_content = header + content
    compressed = zlib.compress(full_content)
    obj_file.write_bytes(compressed)
    
    return object_hash

def load_object(git_dir: Path, object_hash: str) -> tuple[str, bytes]:
    """Load and decompress a Git object, returning (type, content)."""
    obj_file = git_dir / 'objects' / object_hash[:2] / object_hash[2:]
    
    if not obj_file.exists():
        raise ValueError(f"Object {object_hash} not found")
    
    # Read and decompress
    compressed = obj_file.read_bytes()
    decompressed = zlib.decompress(compressed)
    
    # Parse header
    null_index = decompressed.find(b'\0')
    header = decompressed[:null_index].decode()
    content = decompressed[null_index + 1:]
    
    object_type, size_str = header.split(' ', 1)
    expected_size = int(size_str)
    
    if len(content) != expected_size:
        raise ValueError(f"Object {object_hash} has incorrect size")
    
    return object_type, content
```

**D. Core Logic Skeleton Code:**

**Blob Operations** (signatures with detailed TODOs):

```python
# git/objects/blob.py
from pathlib import Path

def hash_object(file_path: Path, git_dir: Path) -> str:
    """Hash a file and store it as a blob object.
    
    Returns the SHA-1 hash of the stored blob.
    """
    # TODO 1: Read file content as bytes (handle both text and binary files)
    # TODO 2: Use store_object() with object_type='blob' and file content
    # TODO 3: Return the computed hash
    # Hint: Git treats all files as binary data, don't decode as text
    pass

def cat_file_blob(object_hash: str, git_dir: Path) -> bytes:
    """Retrieve and return blob content by hash."""
    # TODO 1: Use load_object() to get object type and content
    # TODO 2: Verify object_type is 'blob', raise error if not
    # TODO 3: Return the raw content bytes
    # Hint: This is the reverse of hash_object - just extract content
    pass
```

**Tree Operations** (signatures with detailed TODOs):

```python
# git/objects/tree.py
from pathlib import Path
from typing import List, Tuple

TreeEntry = Tuple[str, str, str]  # (mode, name, hash)

def create_tree_from_index(index_entries: List[dict], git_dir: Path) -> str:
    """Create tree object from staged index entries.
    
    Returns the hash of the created tree object.
    """
    # TODO 1: Sort index entries by name (Git requires lexicographic order)
    # TODO 2: For each entry, format as: mode + ' ' + name + '\0' + binary_hash
    # TODO 3: Handle subdirectories by recursively creating tree objects
    # TODO 4: Use store_object() with object_type='tree' and formatted content
    # TODO 5: Return the computed tree hash
    # Hint: Binary hash is bytes.fromhex(hash_string) - not the hex string!
    pass

def parse_tree_object(tree_hash: str, git_dir: Path) -> List[TreeEntry]:
    """Parse tree object and return list of (mode, name, hash) entries."""
    # TODO 1: Use load_object() to get tree content
    # TODO 2: Parse binary tree format: mode + ' ' + name + '\0' + 20_byte_hash
    # TODO 3: Convert each entry to (mode, name, hex_hash) tuple
    # TODO 4: Return sorted list of entries
    # Hint: Use content.find(b'\0') to locate null bytes between name and hash
    pass

def ls_tree(tree_hash: str, git_dir: Path) -> None:
    """Display tree contents in human-readable format."""
    # TODO 1: Use parse_tree_object() to get entries
    # TODO 2: For each entry, determine object type (blob vs tree) from mode
    # TODO 3: Print in format: mode object_type hash name
    # TODO 4: Mode 40000 = tree, 100644 = regular file, 100755 = executable
    # Hint: You can also load each referenced object to verify its type
    pass
```

**E. Language-Specific Hints:**

**Python-Specific Implementation Tips:**

- Use `pathlib.Path` consistently for all file system operations - it handles cross-platform path differences automatically
- The `hashlib.sha1()` function requires bytes input, not strings - use `.encode()` for text content
- For binary file operations, always use `'rb'` and `'wb'` modes to avoid text encoding issues
- Use `struct.pack()` and `struct.unpack()` for binary data formats like the index file
- The `zlib.compress()` and `zlib.decompress()` functions handle Git's object compression transparently
- Use `os.stat()` to get file modification times and permissions for index entries
- Handle both relative and absolute paths using `Path.resolve()` to avoid confusion

**Common Python Pitfalls:**

⚠️ **Pitfall: String vs Bytes Confusion**
Many Git formats use binary data, but Python 3 distinguishes strings (Unicode) from bytes. Always use bytes for:
- File content when hashing
- SHA-1 hash computation  
- Compressed object storage
- Binary parts of index file format

Use strings only for:
- Hash values in hex format
- File paths and names
- Text content when displaying to user

⚠️ **Pitfall: Path Handling**
Don't use string concatenation for file paths - use `Path` objects:
```python
# Wrong: path + '/' + filename
# Right: path / filename
```

The `Path` class handles platform differences and prevents common path-related bugs.

**F. Milestone Checkpoints:**

After implementing each milestone, verify functionality with these specific tests:

**Milestone 1-2 Checkpoint (Repository + Blob Objects):**
```python
# Test repository initialization
repo = Repository.init(Path('./test-repo'))
assert (repo.git_dir / 'objects').exists()
assert (repo.git_dir / 'HEAD').read_text().strip() == 'ref: refs/heads/master'

# Test blob storage and retrieval  
test_content = b"Hello, Git!"
blob_hash = hash_object_from_content(test_content, repo.git_dir)
retrieved = cat_file_blob(blob_hash, repo.git_dir)
assert retrieved == test_content
print(f"✓ Blob hash: {blob_hash}")
```

Expected output: 40-character hex hash that matches running `echo "Hello, Git!" | git hash-object --stdin` in real Git.

**Milestone 3-4 Checkpoint (Trees + Commits):**
```python
# Create a simple tree and commit
entries = [
    {'mode': '100644', 'name': 'README.md', 'hash': readme_blob_hash},
    {'mode': '100644', 'name': 'main.py', 'hash': main_blob_hash}
]
tree_hash = create_tree_from_entries(entries, repo.git_dir)
commit_hash = create_commit(tree_hash, [], "Initial commit", repo.git_dir)

# Verify tree parsing
tree_entries = parse_tree_object(tree_hash, repo.git_dir)
assert len(tree_entries) == 2
print(f"✓ Tree: {tree_hash}")
print(f"✓ Commit: {commit_hash}")
```

**G. Debugging Tips:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Hash mismatch with real Git | Incorrect object format or missing null bytes | Compare hex dump of your object vs Git's object | Check header format: `{type} {size}\0{content}` |
| "Object not found" errors | Wrong path calculation or hash format | Print object file paths and verify they exist | Ensure hash[:2] directory exists before creating file |
| Compression errors | Storing uncompressed data or wrong compression level | Try decompressing stored objects manually | Use `zlib.compress()` with default settings |
| Index corruption | Incorrect binary format or endianness | Hex dump the index file and compare to spec | Use `struct.pack('>I', value)` for big-endian integers |
| Tree parsing failures | Binary hash vs hex hash confusion | Check if hash storage is 20 bytes or 40 chars | Store binary: `bytes.fromhex(hash_string)` |
| Merge infinite loops | Cycle in commit graph or incorrect parent following | Trace parent chain manually and look for cycles | Add visited set to prevent infinite recursion |

The most common debugging approach is comparing your implementation's output with real Git using identical input. Git's object format is precisely specified, so byte-for-byte comparison reveals implementation errors quickly.


## High-Level Architecture

> **Milestone(s):** This section provides the architectural foundation for all milestones, with particular emphasis on Milestone 1 (Repository Initialization) and the component separation needed for Milestones 2-8

The architecture of our Git implementation follows a clean separation of concerns across four distinct layers, each with well-defined responsibilities and interfaces. Understanding this architectural foundation is crucial because Git's power emerges from the elegant interaction between these components, particularly how they all coordinate through the content-addressable object store.

![Git System Architecture](./diagrams/system-architecture.svg)

### System Components

The Git system architecture consists of four primary components that work together to provide version control functionality. Each component has distinct responsibilities and maintains its own data structures, but they interact through well-defined interfaces to create a cohesive system.

#### Object Store: The Immutable Foundation

The **Object Store** serves as the foundational layer of our Git implementation, providing content-addressable storage for all repository data. Think of it as a digital warehouse where every item has a unique barcode derived from its contents - you can store something once and retrieve it forever using that barcode, with mathematical certainty that the contents haven't changed.

The Object Store manages three critical responsibilities: storing Git objects with SHA-1 hash-based addressing, providing immutable content retrieval, and maintaining data integrity through cryptographic verification. Every piece of content that enters Git - whether it's a file, directory structure, or commit metadata - gets processed through this layer and becomes permanently addressable by its hash.

| Component | Responsibility | Data Managed | Interface Methods |
|-----------|---------------|---------------|-------------------|
| Object Store | Content-addressable storage | Blobs, trees, commits | `store_object()`, `retrieve_object()`, `object_exists()`, `list_objects()` |
| Index | Staging area management | File metadata and hashes | `add_file()`, `remove_file()`, `get_status()`, `write_tree()` |
| References | Branch and tag management | Commit pointers | `create_branch()`, `update_ref()`, `resolve_ref()`, `list_refs()` |
| Working Directory | File system interface | Current project files | `checkout_files()`, `scan_changes()`, `apply_diff()` |

The Object Store's design centers around three core data structures that represent different types of content. **Blob objects** store file content as immutable byte sequences, with each blob identified by the SHA-1 hash of its contents. **Tree objects** represent directory structures, containing sorted lists of file and subdirectory entries with their corresponding hashes. **Commit objects** capture project snapshots, linking to a tree hash and containing metadata about the change.

> **Key Insight**: The Object Store's immutability is what enables Git's powerful features. Because objects never change once stored, operations like branching, merging, and history traversal become safe and efficient. Multiple branches can share the same objects without fear of interference.

**Object Store Data Structures:**

| Object Type | Header Format | Content Structure | Key Properties |
|-------------|---------------|-------------------|----------------|
| Blob | `blob {size}\0` | Raw file bytes | Content-addressed, immutable |
| Tree | `tree {size}\0` | Sorted entries: `{mode} {name}\0{hash}` | Directory representation |
| Commit | `commit {size}\0` | Tree hash, parents, author, message | Links project snapshots |

The Object Store implements a simple but effective storage algorithm. When storing an object, it first computes the SHA-1 hash of the complete object (header plus content), then compresses the object using zlib compression, and finally stores the compressed data at a file system path derived from the hash. The path structure uses the first two hexadecimal characters of the hash as a directory name, with the remaining 38 characters as the filename.

#### Index: The Staging Area Bridge

The **Index** acts as an intermediate layer between the working directory and the object store, implementing Git's distinctive staging area concept. Picture it as a photographer's contact sheet - a place where you arrange and review your shots before committing to the final print. The Index allows developers to incrementally prepare commits by selectively staging changes.

The Index maintains a binary file at `.git/index` that contains metadata about staged files, including their object hashes, file system timestamps, and permission modes. This metadata enables Git to quickly detect when files have been modified since they were last staged, without needing to re-hash file contents on every status check.

**Index Entry Structure:**

| Field | Size | Type | Description |
|-------|------|------|-------------|
| Create Time | 8 bytes | uint64 | File creation timestamp |
| Modify Time | 8 bytes | uint64 | File modification timestamp |
| Device ID | 4 bytes | uint32 | File system device identifier |
| Inode | 4 bytes | uint32 | File system inode number |
| Mode | 4 bytes | uint32 | File permissions and type |
| UID | 4 bytes | uint32 | Owner user identifier |
| GID | 4 bytes | uint32 | Owner group identifier |
| File Size | 4 bytes | uint32 | Size of file in bytes |
| SHA-1 Hash | 20 bytes | bytes | Object hash of file contents |
| Flags | 2 bytes | uint16 | Name length and stage flags |
| Path Name | Variable | UTF-8 string | Relative path from repository root |

The Index enables the three-way status calculation that shows users what changes are staged, unstaged, or untracked. It compares file timestamps and sizes between the working directory, index, and HEAD commit to efficiently determine file states without expensive hash computations for unchanged files.

#### References: The Human-Readable Navigation System

The **References** component provides human-readable names for commits, implementing Git's branch and tag system. Think of references as bookmarks in a vast library - instead of memorizing long SHA-1 hashes, you can use meaningful names like "main" or "feature-login" to navigate your project's history.

References are stored as simple text files in the `.git/refs` directory hierarchy. Branch references live in `.git/refs/heads/`, tag references in `.git/refs/tags/`, and remote references in `.git/refs/remotes/`. Each reference file contains a single line with either a commit hash (direct reference) or a reference to another reference (symbolic reference).

**Reference Types:**

| Reference Type | Storage Location | Content Format | Example |
|----------------|------------------|----------------|---------|
| Branch | `.git/refs/heads/{name}` | SHA-1 hash | `a1b2c3d4e5f6...` |
| Tag | `.git/refs/tags/{name}` | SHA-1 hash or tag object | `a1b2c3d4e5f6...` |
| HEAD | `.git/HEAD` | Symbolic or direct ref | `ref: refs/heads/main` |
| Detached HEAD | `.git/HEAD` | Direct SHA-1 hash | `a1b2c3d4e5f6...` |

The HEAD reference deserves special attention as it tracks the current branch or commit. In normal operation, HEAD contains a symbolic reference like `ref: refs/heads/main`, indicating that commits should advance the main branch. In detached HEAD state, HEAD contains a direct commit hash, meaning commits won't advance any branch.

> **Design Decision**: References use the file system as their storage layer rather than the object store. This allows atomic updates through file system operations and makes references easily readable by external tools, while keeping them separate from the immutable object history.

#### Working Directory: The User Interface Layer

The **Working Directory** represents the file system view of your project, containing the actual files that users edit and build. It serves as the interface between Git's internal data structures and the user's development workflow. The Working Directory component is responsible for checking out files from the object store, detecting changes, and applying updates from commits.

The Working Directory maintains no persistent state of its own - it's purely a projection of some commit's tree structure onto the file system. However, it provides crucial functionality for detecting modifications, handling file permissions, and managing the checkout process when switching between branches or commits.

**Working Directory Operations:**

| Operation | Input | Output | Side Effects |
|-----------|--------|--------|--------------|
| Checkout | Tree hash | File system changes | Updates working files |
| Scan Changes | File paths | Modified file list | None |
| Apply Diff | Diff patches | File modifications | Updates working files |
| Clean | None | Status report | Removes untracked files |

The Working Directory implements change detection through file system metadata comparison. By comparing modification times, file sizes, and inode numbers against the Index's cached values, it can quickly identify potentially modified files. Only files that appear changed need to be re-hashed to determine if their content actually differs.

### Recommended File Structure

Organizing the codebase to mirror the architectural layers makes the system easier to understand, test, and maintain. Each component should have its own module with clear boundaries and minimal dependencies. The following structure separates concerns while enabling the necessary interactions between components.

```
git_implementation/
├── core/
│   ├── __init__.py              # Core package exports
│   ├── repository.py            # Repository class and initialization
│   ├── hash_utils.py            # SHA-1 computation utilities
│   └── constants.py             # Shared constants and formats
├── objects/
│   ├── __init__.py              # Object store package
│   ├── store.py                 # Object storage and retrieval
│   ├── blob.py                  # Blob object implementation
│   ├── tree.py                  # Tree object implementation
│   ├── commit.py                # Commit object implementation
│   └── serialization.py         # Object serialization/parsing
├── index/
│   ├── __init__.py              # Index package
│   ├── staging.py               # Index file management
│   ├── status.py                # Status calculation logic
│   └── binary_format.py         # Index binary format handling
├── refs/
│   ├── __init__.py              # References package
│   ├── manager.py               # Reference CRUD operations
│   ├── symbolic.py              # Symbolic reference handling
│   └── head.py                  # HEAD reference management
├── working_dir/
│   ├── __init__.py              # Working directory package
│   ├── checkout.py              # File checkout operations
│   ├── scanner.py               # Change detection
│   └── filesystem.py           # File system utilities
├── algorithms/
│   ├── __init__.py              # Algorithms package
│   ├── diff.py                  # Myers diff implementation
│   ├── merge.py                 # Three-way merge logic
│   └── merge_base.py            # Common ancestor calculation
├── commands/
│   ├── __init__.py              # Commands package
│   ├── init.py                  # git init implementation
│   ├── add.py                   # git add implementation
│   ├── commit.py                # git commit implementation
│   ├── branch.py                # git branch implementation
│   ├── merge.py                 # git merge implementation
│   └── status.py                # git status implementation
├── tests/
│   ├── test_objects/            # Object store tests
│   ├── test_index/              # Index tests
│   ├── test_refs/               # References tests
│   ├── test_algorithms/         # Algorithm tests
│   └── integration/             # End-to-end tests
└── main.py                      # CLI entry point
```

This structure provides several important benefits for learners and maintainers. The separation between `objects/`, `index/`, `refs/`, and `working_dir/` mirrors the architectural components, making it easy to understand where functionality belongs. The `algorithms/` package isolates complex logic like diff and merge, which can be tested independently. The `commands/` package provides a clean CLI interface without mixing user interaction with core logic.

**Module Dependencies and Interfaces:**

| Module | Direct Dependencies | Key Interfaces |
|--------|-------------------|----------------|
| `core/` | None | `Repository`, `compute_sha1()` |
| `objects/` | `core/` | `ObjectStore`, object type classes |
| `index/` | `core/`, `objects/` | `Index`, `IndexEntry` |
| `refs/` | `core/` | `ReferenceManager`, `HEAD` |
| `working_dir/` | `core/`, `objects/` | `WorkingDirectory`, `FileScanner` |
| `algorithms/` | `objects/`, `index/` | `diff()`, `merge()`, `find_merge_base()` |
| `commands/` | All others | Command implementations |

> **Architecture Decision Record: Component Isolation**
> - **Context**: Git operations involve complex interactions between object storage, indexing, and file system operations. Poor separation leads to tangled dependencies and difficult testing.
> - **Options Considered**: Monolithic design, layered architecture, component-based architecture
> - **Decision**: Component-based architecture with clear interface boundaries
> - **Rationale**: Each component can be developed and tested independently, interfaces prevent tight coupling, and the design mirrors Git's actual architecture making it easier to understand
> - **Consequences**: Enables parallel development of components, requires more upfront design but pays dividends in maintainability, makes debugging easier by isolating failures to specific components

The `tests/` directory structure mirrors the main codebase, enabling focused testing of each component. Integration tests verify that components work together correctly, while unit tests ensure each component meets its individual contracts. This testing structure supports the milestone-based development approach by allowing verification of each component as it's built.

**Common Pitfalls in Architecture Organization:**

⚠️ **Pitfall: Circular Dependencies**
Many learners create circular imports between components, such as the Index importing from References while References import from Index. This happens when components try to directly access each other's internals rather than going through well-defined interfaces. The fix is to pass dependencies as parameters and use dependency injection rather than direct imports between peer components.

⚠️ **Pitfall: Mixing CLI Logic with Core Logic**
Another common mistake is embedding command-line parsing, user interaction, and error formatting inside the core Git logic. This makes the core components impossible to test and reuse. Keep all user interface concerns in the `commands/` package, and ensure core components only raise exceptions with structured data that the CLI layer can format appropriately.

⚠️ **Pitfall: Putting Everything in the Repository Class**
Beginners often make the `Repository` class a "god object" that contains all Git functionality. This creates a massive class that's difficult to test and understand. Instead, the `Repository` class should be a lightweight coordinator that holds references to the component instances and provides convenience methods that delegate to the appropriate components.

### Implementation Guidance

The architectural components should be implemented with clear separation of concerns and well-defined interfaces. This guidance provides the foundational code structure and implementation patterns needed to build each component correctly.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Object Storage | File system + zlib | LevelDB or SQLite |
| Index Format | Binary struct packing | Protocol Buffers |
| Hash Computation | hashlib.sha1() | Custom SHA-1 implementation |
| File System Operations | pathlib.Path | os.path with error handling |
| Compression | zlib standard library | LZ4 or custom compression |

**Core Infrastructure Code:**

The following provides complete, working implementations of the foundational utilities that all components depend on:

```python
# core/constants.py - Shared constants across all components
SHA1_HEX_LENGTH = 40
SHA1_BINARY_LENGTH = 20
OBJECT_HEADER_FORMAT = "{type} {size}\0{content}"

# Git object types
BLOB_TYPE = "blob"
TREE_TYPE = "tree" 
COMMIT_TYPE = "commit"

# Index constants
INDEX_VERSION = 2
INDEX_HEADER_SIZE = 12
INDEX_ENTRY_MIN_SIZE = 62

# File modes (octal values stored as integers)
FILE_MODE_REGULAR = 0o100644
FILE_MODE_EXECUTABLE = 0o100755
FILE_MODE_TREE = 0o040000
FILE_MODE_SYMLINK = 0o120000
```

```python
# core/hash_utils.py - Complete hash computation utilities
import hashlib
from typing import bytes

def compute_sha1(content: bytes) -> str:
    """Compute SHA-1 hash of raw bytes, returning hex digest."""
    hasher = hashlib.sha1()
    hasher.update(content)
    return hasher.hexdigest()

def compute_object_hash(object_type: str, content: bytes) -> str:
    """Compute Git object hash including type header."""
    header = f"{object_type} {len(content)}\0".encode('ascii')
    full_content = header + content
    return compute_sha1(full_content)

def hash_to_path_components(object_hash: str) -> tuple[str, str]:
    """Convert hash to directory and filename components."""
    if len(object_hash) != SHA1_HEX_LENGTH:
        raise ValueError(f"Invalid hash length: {len(object_hash)}")
    return object_hash[:2], object_hash[2:]
```

```python
# core/repository.py - Repository initialization and discovery
from pathlib import Path
from typing import Optional
import os

class Repository:
    """Represents a Git repository with its working tree and git directory."""
    
    def __init__(self, work_tree: Path, git_dir: Path):
        self.work_tree = work_tree.resolve()
        self.git_dir = git_dir.resolve()
        
    @classmethod
    def init(cls, path: Path) -> 'Repository':
        """Initialize a new Git repository at the given path."""
        # TODO 1: Create the target directory if it doesn't exist
        # TODO 2: Create .git subdirectory structure
        # TODO 3: Initialize .git/objects with subdirectories  
        # TODO 4: Initialize .git/refs/heads and .git/refs/tags directories
        # TODO 5: Create initial HEAD file pointing to refs/heads/master
        # TODO 6: Create basic config file with repository settings
        # TODO 7: Set appropriate permissions on .git directory (0755)
        # Return Repository instance with work_tree=path and git_dir=path/.git
        pass
        
    def object_path_from_hash(self, object_hash: str) -> Path:
        """Convert object hash to file system path."""
        dir_name, file_name = hash_to_path_components(object_hash)
        return self.git_dir / "objects" / dir_name / file_name

def find_git_directory(start_path: Path) -> Optional[Path]:
    """Locate .git directory by walking up the directory tree."""
    current = start_path.resolve()
    while current != current.parent:
        git_dir = current / ".git"
        if git_dir.is_dir():
            return git_dir
        current = current.parent
    return None
```

**Component Interface Skeletons:**

Each major component should implement these interfaces. The skeletons provide method signatures and detailed TODO comments that map to the architectural responsibilities:

```python
# objects/store.py - Object store component skeleton
import zlib
from pathlib import Path
from typing import Optional, bytes

class ObjectStore:
    """Content-addressable storage for Git objects."""
    
    def __init__(self, git_dir: Path):
        self.objects_dir = git_dir / "objects"
        
    def store_object(self, object_type: str, content: bytes) -> str:
        """Store an object and return its hash."""
        # TODO 1: Compute object hash using compute_object_hash()
        # TODO 2: Check if object already exists using object_exists()
        # TODO 3: Create header with format "{type} {size}\0"
        # TODO 4: Combine header and content into full object
        # TODO 5: Compress full object using zlib.compress()
        # TODO 6: Determine storage path using hash_to_path_components()
        # TODO 7: Create parent directory if it doesn't exist
        # TODO 8: Write compressed data to object file atomically
        # TODO 9: Set file permissions to 0444 (read-only)
        # Return the computed hash
        pass
        
    def retrieve_object(self, object_hash: str) -> tuple[str, bytes]:
        """Retrieve object content and type by hash."""
        # TODO 1: Validate hash format and length
        # TODO 2: Construct object file path
        # TODO 3: Check if object file exists, raise error if not
        # TODO 4: Read and decompress object file using zlib.decompress()
        # TODO 5: Parse header to extract type and size
        # TODO 6: Validate content size matches header
        # TODO 7: Return tuple of (object_type, content)
        pass
        
    def object_exists(self, object_hash: str) -> bool:
        """Check if object exists in store."""
        # TODO: Check if object file exists at computed path
        pass
```

**File Structure Creation:**

```python
# Directory structure creation helper
def create_git_directory_structure(git_dir: Path) -> None:
    """Create complete .git directory structure."""
    directories_to_create = [
        git_dir,
        git_dir / "objects" / "info",
        git_dir / "objects" / "pack", 
        git_dir / "refs" / "heads",
        git_dir / "refs" / "tags",
        git_dir / "hooks",
    ]
    
    for directory in directories_to_create:
        directory.mkdir(parents=True, exist_ok=True)
        # Set directory permissions to 0755
        directory.chmod(0o755)
    
    # Create initial HEAD file
    head_file = git_dir / "HEAD"
    head_file.write_text("ref: refs/heads/master\n")
    
    # Create basic config
    config_file = git_dir / "config"
    config_content = """[core]
    repositoryformatversion = 0
    filemode = true
    bare = false
    logallrefupdates = true
"""
    config_file.write_text(config_content)
```

**Milestone Checkpoints:**

After implementing the basic architecture:

1. **Repository Initialization Test**: Run `Repository.init(Path("test_repo"))` and verify the directory structure:
   ```bash
   find test_repo/.git -type d | sort
   # Should show: .git, .git/hooks, .git/objects, .git/objects/info, 
   # .git/objects/pack, .git/refs, .git/refs/heads, .git/refs/tags
   ```

2. **Object Store Test**: Store and retrieve a simple blob:
   ```python
   store = ObjectStore(Path("test_repo/.git"))
   hash1 = store.store_object("blob", b"hello world")
   obj_type, content = store.retrieve_object(hash1) 
   assert obj_type == "blob" and content == b"hello world"
   ```

3. **Component Independence Test**: Each component should be importable and testable in isolation without requiring the others to be fully implemented.

**Debugging Tips:**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| "Object not found" errors | Incorrect path computation | Print the computed path and check if file exists | Verify hash_to_path_components() logic |
| Permission denied on .git | Wrong directory permissions | Check with `ls -la .git` | Set permissions to 0755 on directories |
| Hash mismatches | Header format incorrect | Print the raw content being hashed | Ensure null byte between size and content |
| Import errors between components | Circular dependencies | Draw dependency graph | Move shared code to core/, use dependency injection |


## Data Model

> **Milestone(s):** This section establishes the data model foundations for Milestone 2 (Object Storage - Blobs), Milestone 3 (Tree Objects), and Milestone 4 (Commit Objects), while providing the conceptual framework for all remaining milestones

The data model forms the heart of Git's architecture, defining how content is structured, stored, and interconnected within the content-addressable object store. Understanding this model is crucial because every Git operation—from storing a single file to merging complex histories—ultimately manipulates these three fundamental object types and their relationships.

![Git Object Model](./diagrams/object-model.svg)

### Mental Model: The Universal Content Graph

Think of Git's data model as a universal content graph, similar to how Wikipedia represents knowledge. In Wikipedia, articles (like commits) reference other articles and media files (like trees and blobs), creating an interconnected web of information where each piece of content has a unique, permanent address. Just as Wikipedia's internal links remain valid even when articles are moved or renamed, Git's objects reference each other through immutable cryptographic addresses that never change.

The key insight is that Git doesn't store files and directories the way your operating system does—it stores a graph of content relationships. A commit doesn't "contain" files; it points to a tree that describes the project structure at that moment in time. That tree doesn't "contain" files either; it points to blobs that hold the actual content. This indirection enables powerful features: two commits can share the same tree (identical project states), multiple trees can reference the same blob (duplicate files), and the entire history forms a graph where content is automatically deduplicated.

Consider how this differs from traditional file storage. When you copy a directory, the operating system duplicates all the files. When Git stores multiple commits with identical files, those files exist only once in the object store, referenced by their content hash. This is why Git repositories remain compact despite having complete project history.

### Git Object Types

Git's object model consists of exactly three object types, each serving a specific purpose in representing project content and history. Every object follows the same fundamental structure: a header containing the object type and size, followed by the raw content, with the entire object identified by the SHA-1 hash of this formatted data.

> **Design Insight:** The three-object model strikes an optimal balance between simplicity and expressiveness. Blobs handle raw content, trees handle directory structure, and commits handle history and metadata. This minimal set can represent arbitrarily complex projects and histories while maintaining Git's core properties of immutability and content-addressable storage.

#### Object Storage Format

Every Git object follows a consistent storage format that enables content-addressable lookup and integrity verification. This uniformity allows the object store to handle all three object types through a single storage and retrieval mechanism.

| Component | Format | Description |
|-----------|--------|-------------|
| Object Header | `{type} {size}\0` | Object type string, space, decimal byte count, null terminator |
| Object Content | `{raw_content}` | Type-specific content in binary or text format |
| Storage Format | `compress(header + content)` | Zlib-compressed complete object |
| Hash Calculation | `sha1(header + content)` | SHA-1 of uncompressed object |
| File Path | `.git/objects/{hash[0:2]}/{hash[2:40]}` | First 2 hash chars as directory, remaining 38 as filename |

The hash calculation is performed on the complete uncompressed object (header + content), but the stored file contains the zlib-compressed version. This separation enables efficient storage while maintaining cryptographic integrity verification.

> **Architecture Decision: Object Header Format**
> - **Context**: Need a way to store different object types in a unified storage system while enabling type identification and content verification
> - **Options Considered**: 
>   1. Store type as separate metadata file
>   2. Use file extensions to indicate type
>   3. Embed type and size in object content header
> - **Decision**: Embed `{type} {size}\0` header in each object's content
> - **Rationale**: Self-describing objects eliminate metadata synchronization issues, size enables integrity verification, null terminator provides clear binary boundary
> - **Consequences**: Every object is self-contained and verifiable, but adds small storage overhead for the header

#### Blob Objects

Blob objects store raw file content without any metadata such as filename, permissions, or timestamps. They represent the pure content of files at specific points in time, enabling content deduplication across the entire repository history.

| Property | Description |
|----------|-------------|
| Purpose | Store raw file content without filesystem metadata |
| Content Format | Exact byte sequence from the original file |
| Deduplication | Files with identical content share the same blob object |
| Size Limits | Practically unlimited, though performance degrades for very large files |
| Binary Handling | All content treated as binary; no line-ending or encoding conversions |

**Blob Object Structure:**
```
Header: "blob {content_length}\0"
Content: {raw_file_bytes}
```

The blob hash depends only on content, never on filename or location. This means moving a file without changing its content creates no new objects—the tree changes but the blob remains identical. Similarly, if multiple files in the project have identical content, they share a single blob object.

**Content Handling Considerations:**

Blob objects preserve file content exactly as it exists in the working directory, making no assumptions about text encoding, line endings, or binary formats. This byte-for-byte preservation ensures perfect fidelity when checking out files, but requires careful handling of cross-platform differences at higher levels of the system.

| Scenario | Blob Behavior | Implications |
|----------|---------------|-------------|
| Text files with CRLF line endings | Stored exactly as provided | Cross-platform checkouts may differ |
| Binary files | Stored as-is with no interpretation | Perfect preservation of executable and media files |
| Empty files | Creates blob with zero-length content | Empty files still consume one object in the store |
| Large files | Stored completely in single blob | No automatic chunking; entire file must fit in memory |

#### Tree Objects

Tree objects represent directory structure at specific points in time, storing sorted lists of files and subdirectories with their associated permissions and object hashes. They provide the hierarchical organization that transforms flat blob storage into recognizable project structure.

**Tree Object Structure:**
```
Header: "tree {content_length}\0"
Content: {sorted_tree_entries}
```

Each tree entry follows a specific binary format that packs directory information efficiently while maintaining sort order for consistent hashing:

| Field | Format | Size | Description |
|-------|--------|------|-------------|
| Mode | ASCII decimal string | Variable | Unix file mode (permissions and type) |
| Space | ASCII space character | 1 byte | Separator between mode and name |
| Name | UTF-8 string | Variable | Filename or directory name |
| Null Terminator | `\0` byte | 1 byte | Separator between name and hash |
| Object Hash | Raw SHA-1 bytes | 20 bytes | Binary object hash (not hex-encoded) |

**Tree Entry Format:**
```
{mode} {name}\0{20_byte_hash}
```

The mode field encodes both file permissions and type information using Unix conventions. Understanding these modes is crucial for preserving file system semantics across different platforms.

| Mode | Type | Description |
|------|------|-------------|
| `100644` | Regular file | Normal file with read/write permissions |
| `100755` | Executable file | File with execute permissions |
| `040000` | Directory | Subdirectory (points to another tree object) |
| `120000` | Symbolic link | Symlink (blob contains link target path) |
| `160000` | Git submodule | Reference to another Git repository |

**Sorting Requirements:**

Tree entries must be sorted to ensure consistent hashing across different systems and Git implementations. The sort order treats directory names specially to maintain proper tree structure:

1. **Primary sort**: Lexicographic order by entry name
2. **Directory handling**: Directories are sorted as if their names ended with `/`
3. **Case sensitivity**: Sorting is case-sensitive using byte values
4. **Locale independence**: Sorting uses binary comparison, not locale-specific rules

This sorting ensures that the same directory structure always produces the same tree hash, regardless of the order in which files were added to the index.

**Nested Directory Representation:**

Trees handle directory hierarchies through recursive object references. Each subdirectory becomes a separate tree object, referenced by its hash from the parent tree. This creates a tree-of-trees structure that mirrors the directory hierarchy.

```
project/
├── README.md          → blob abc123... (referenced from root tree)
├── src/               → tree def456... (referenced from root tree)
│   ├── main.py        → blob ghi789... (referenced from src tree)
│   └── utils.py       → blob jkl012... (referenced from src tree)
└── tests/             → tree mno345... (referenced from root tree)
    └── test_main.py   → blob pqr678... (referenced from tests tree)
```

This structure enables efficient sharing of subtrees between commits. If only `README.md` changes between commits, the `src/` and `tests/` tree objects remain identical and are reused.

#### Commit Objects

Commit objects capture complete project snapshots along with metadata about when, why, and by whom changes were made. They form the backbone of Git's history tracking by linking project states into a directed acyclic graph.

**Commit Object Structure:**
```
Header: "commit {content_length}\0"
Content: {commit_fields}
```

Commit content consists of structured text fields that provide complete context for each project snapshot:

| Field | Format | Required | Description |
|-------|--------|----------|-------------|
| tree | `tree {hash}` | Yes | SHA-1 hash of root tree object |
| parent | `parent {hash}` | No | SHA-1 hash of parent commit (0 or more) |
| author | `author {name} <{email}> {timestamp} {timezone}` | Yes | Who created the changes |
| committer | `committer {name} <{email}> {timestamp} {timezone}` | Yes | Who committed the changes |
| blank line | `\n` | Yes | Separates metadata from message |
| message | Free-form text | Yes | Commit message (can span multiple lines) |

**Example Commit Object Content:**
```
tree 4b825dc642cb6eb9a060e54bf8d69288fbee4904
parent 3a1b2c3d4e5f6789abcd0123456789abcdef0123
author John Doe <john@example.com> 1609459200 +0000
committer John Doe <john@example.com> 1609459200 +0000

Initial commit with basic project structure

Added README.md and basic source code layout
```

**Timestamp Format:**

Git stores timestamps as Unix epoch seconds followed by timezone offset. This format enables precise temporal ordering while preserving timezone information for distributed development.

| Component | Format | Example | Description |
|-----------|--------|---------|-------------|
| Unix Timestamp | Decimal seconds since epoch | `1609459200` | UTC seconds since January 1, 1970 |
| Timezone Offset | `±HHMM` format | `+0000`, `-0500` | Offset from UTC in hours and minutes |
| Complete Format | `{timestamp} {offset}` | `1609459200 +0000` | Combined timestamp and timezone |

The timezone preservation allows Git to maintain exact temporal context even when commits are created in different timezones, which is crucial for distributed teams.

**Author vs Committer:**

Git distinguishes between the person who authored changes and the person who committed them to the repository. This distinction supports workflows where patches are created by one person and applied by another.

| Role | When Different | Example Scenario |
|------|----------------|------------------|
| Author | Original creator of changes | Developer writes patch |
| Committer | Person who applies changes | Maintainer applies patch to repository |
| Same Person | Direct commits | Developer commits own work directly |
| Automated Systems | CI/CD commits | Build system commits generated changes |

**Parent References and History Graph:**

Commits link to their predecessors through parent references, creating the history graph that represents project evolution over time. The number of parents determines the commit type:

| Parent Count | Commit Type | Description | Graph Implications |
|--------------|-------------|-------------|-------------------|
| 0 | Initial commit | First commit in repository | Root node of history graph |
| 1 | Normal commit | Regular development progress | Linear history segment |
| 2+ | Merge commit | Integration of multiple branches | Graph convergence point |

Multiple parents enable Git to represent complex development histories where feature branches are merged back into main development lines. Each parent hash creates an edge in the commit graph, allowing Git to traverse history in multiple directions.

### Object Relationships

The power of Git's data model emerges from how objects reference each other to form a complete content-addressable graph. These relationships enable efficient storage, robust history tracking, and powerful operations like merging and rebasing.

#### Content-Addressable References

All object references in Git use SHA-1 hashes as addresses, creating a content-addressable system where references are derived from the content they point to. This approach provides several critical properties:

| Property | Description | Benefits |
|----------|-------------|----------|
| Immutability | Object content cannot change without changing its hash | History integrity and tamper detection |
| Deduplication | Identical content shares the same hash and storage | Efficient storage utilization |
| Verification | Hash validates content integrity | Corruption detection and data consistency |
| Location Independence | Hash works regardless of storage location | Distributed repository synchronization |

When a commit references a tree, it references the exact content state represented by that tree's hash. If any file changes, the blob hash changes, which changes the tree hash, which changes the commit hash. This cascade ensures that commit hashes uniquely identify complete project states.

#### The Complete Reference Graph

Git objects form a directed acyclic graph (DAG) where commits point to trees, trees point to blobs and other trees, and commits point to parent commits. Understanding this graph structure is essential for implementing Git operations correctly.

**Reference Flow:**
1. **Commit → Tree**: Each commit references exactly one root tree representing the complete project state
2. **Tree → Blob/Tree**: Each tree entry references either a blob (file) or another tree (subdirectory)
3. **Commit → Commit**: Each commit references zero or more parent commits, forming the history graph
4. **No Circular References**: The graph is acyclic—objects never reference themselves directly or indirectly

**Graph Properties:**

| Property | Description | Implementation Impact |
|----------|-------------|----------------------|
| Directed | References flow from commits toward content | History traversal has natural direction |
| Acyclic | No reference cycles exist | Graph algorithms terminate reliably |
| Multi-rooted | Multiple commits can exist without common ancestors | Supports repository merging |
| Immutable | Objects never change after creation | Safe concurrent access without locking |

#### Shared Object References

One of Git's most elegant features is automatic content sharing through hash-based references. When multiple objects need to reference identical content, they naturally share the same hash and storage location.

**Blob Sharing Scenarios:**

| Scenario | Sharing Mechanism | Storage Impact |
|----------|-------------------|----------------|
| Duplicate files | Same content produces identical blob hashes | Single blob stored regardless of file count |
| File copies | Copying doesn't change content, shares blob | Zero storage cost for file copies |
| Partial file duplication | Different files share blob only if completely identical | No automatic deduplication for similar files |

**Tree Sharing Scenarios:**

| Scenario | Sharing Mechanism | Storage Impact |
|----------|-------------------|----------------|
| Unchanged subdirectories | Same tree structure produces identical hash | Subtrees shared across commits |
| Branch merging | Common directory states share tree objects | Efficient merge representation |
| File renames within directory | Tree content unchanged, shares existing tree | Renames are metadata-only changes |

**Commit Graph Sharing:**

Commits share parent references to build the history graph, but the sharing goes deeper. When branches diverge from a common point, they share all ancestor commits, creating efficient representation of parallel development.

```
A ← B ← C ← D    (main branch)
     ↖ E ← F     (feature branch)
```

In this example:
- Commits A and B are shared by both branches
- Commits C,D belong only to main
- Commits E,F belong only to feature
- Storage cost is 6 commits, not 8 (no duplication of A,B)

#### Reference Resolution and Traversal

Understanding how to navigate the object graph is crucial for implementing Git operations. Different operations require different traversal patterns through the reference relationships.

**Common Traversal Patterns:**

| Operation | Traversal Pattern | Object Types Accessed |
|-----------|-------------------|----------------------|
| Checkout | Commit → Tree → Blobs | All three types for complete project reconstruction |
| Log | Commit → Parent Commits | Commits only for history traversal |
| Diff | Commit → Tree, Compare Trees → Blobs | Commits and trees for comparison, blobs for content diff |
| Status | Working Directory ↔ Index ↔ HEAD | Trees and blobs for three-way comparison |

**Graph Traversal Algorithms:**

Different Git operations require different graph traversal strategies to efficiently access the required objects:

| Algorithm | Use Case | Traversal Order | Termination Condition |
|-----------|----------|-----------------|----------------------|
| Breadth-First Search | Finding merge base | Level-by-level commit traversal | Common ancestor found |
| Depth-First Search | Complete history traversal | Follow parent chains deeply | No more parents |
| Topological Sort | Chronological history | Respect parent-child relationships | All commits processed |
| Tree Recursion | File system operations | Directory-first or file-first | All tree entries processed |

#### Object Storage Efficiency

The reference relationships enable significant storage efficiencies that make Git practical for large repositories with long histories. Understanding these efficiencies helps appreciate why Git's approach scales well.

**Deduplication Through References:**

| Content Type | Deduplication Level | Efficiency Gain |
|--------------|-------------------|-----------------|
| Identical files | Complete blob sharing | Near-zero cost for duplicates |
| Unchanged directories | Complete tree sharing | Subtree reuse across commits |
| Common history | Shared commit ancestors | Linear growth with unique changes |
| Branch points | Shared parent references | Efficient parallel development |

**Storage Growth Patterns:**

Understanding how repository size grows helps design efficient operations and predict storage requirements:

| Change Type | New Objects Created | Storage Impact |
|-------------|-------------------|----------------|
| Edit single file | 1 blob, 1+ trees, 1 commit | Proportional to directory depth |
| Add new file | 1 blob, 1+ trees, 1 commit | Same as file edit |
| Rename file | 0 blobs, 1+ trees, 1 commit | Metadata-only change |
| Copy file | 0 blobs, 1+ trees, 1 commit | Blob reuse makes copies free |

This efficiency model explains why Git can maintain complete history with reasonable storage costs—most changes affect only a small portion of the total project content, and unchanged content is automatically shared.

### Common Pitfalls

⚠️ **Pitfall: Hash Encoding in Tree Objects**
Tree objects store SHA-1 hashes as raw 20-byte binary data, not as 40-character hex strings. This is a frequent source of errors when implementing tree parsing and creation. Using hex encoding doubles the storage size and produces incorrect hash values. Always convert hex hashes to binary using `bytes.fromhex()` before storing in tree objects, and convert back to hex using `.hex()` when displaying or comparing hashes.

⚠️ **Pitfall: Incorrect Object Header Format**
The object header must use the exact format `{type} {size}\0{content}` with a space between type and size, and a null byte (not newline) after the size. Using incorrect separators or missing the null terminator results in hash mismatches with standard Git. The size must be the decimal byte count of the content portion only, not including the header itself.

⚠️ **Pitfall: Tree Entry Sorting**
Tree entries must be sorted lexicographically with directories treated as if their names end with `/`. Incorrect sorting produces different tree hashes for identical directory contents. Use byte-level comparison, not locale-specific string sorting. The sort must be stable and consistent across all platforms to ensure repository compatibility.

⚠️ **Pitfall: Binary vs Text Content Handling**
Always treat object content as binary data, even for text files. Using text mode file operations can corrupt binary content through line-ending conversion or encoding transformations. Open files in binary mode (`'rb'`, `'wb'`) and handle encoding explicitly when needed for display purposes.

⚠️ **Pitfall: Timestamp Format Confusion**
Git timestamps use Unix epoch seconds with timezone offsets, not local time representations. The format is `{seconds_since_epoch} {±HHMM}`, where the timezone offset is hours and minutes from UTC. Using local time or incorrect timezone formats breaks chronological ordering and compatibility with standard Git.

⚠️ **Pitfall: Parent Hash Storage**
Commit objects can have zero, one, or multiple parent lines, but each parent must be stored as a separate `parent {hash}` line, not as multiple hashes on a single line. Initial commits have no parent lines, merge commits have multiple parent lines. The order of parent lines affects merge commit interpretation.

### Implementation Guidance

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Hash Computation | `hashlib.sha1()` from standard library | Custom SHA-1 implementation for learning |
| Binary Data Handling | `bytes` type with `struct.pack()/unpack()` | Custom binary serialization classes |
| Object Storage | Direct file I/O with `pathlib.Path` | Abstract storage interface supporting multiple backends |
| Content Compression | `zlib.compress()/decompress()` from standard library | Stream-based compression for large objects |

#### Recommended File Structure

```
git-implementation/
├── core/
│   ├── __init__.py
│   ├── objects.py              ← Object model (this implementation)
│   │   ├── class GitObject     ← Base object interface
│   │   ├── class BlobObject    ← Blob implementation
│   │   ├── class TreeObject    ← Tree implementation  
│   │   └── class CommitObject  ← Commit implementation
│   ├── hash_utils.py           ← SHA-1 and object hashing utilities
│   └── storage.py              ← Object store interface
├── tests/
│   ├── test_objects.py         ← Object model tests
│   └── fixtures/               ← Test data files
└── examples/
    └── object_examples.py      ← Usage examples
```

This structure separates the core object model from storage concerns, making it easier to test object creation and parsing independently from file system operations.

#### Infrastructure Starter Code

**Hash Utilities (`core/hash_utils.py`):**
```python
import hashlib
import zlib
from typing import bytes

# Complete utility functions for object hashing and storage format
def compute_sha1(content: bytes) -> str:
    """Compute SHA-1 hash of raw bytes, returning hex string."""
    return hashlib.sha1(content).hexdigest()

def compute_object_hash(object_type: str, content: bytes) -> str:
    """Compute Git object hash with proper header format."""
    header = f"{object_type} {len(content)}\0".encode('utf-8')
    full_content = header + content
    return compute_sha1(full_content)

def format_object_for_storage(object_type: str, content: bytes) -> bytes:
    """Format object with header and compress for storage."""
    header = f"{object_type} {len(content)}\0".encode('utf-8')
    full_content = header + content
    return zlib.compress(full_content)

def parse_stored_object(compressed_data: bytes) -> tuple[str, bytes]:
    """Decompress and parse stored object, returning (type, content)."""
    decompressed = zlib.decompress(compressed_data)
    # Find null terminator separating header from content
    null_pos = decompressed.find(b'\0')
    if null_pos == -1:
        raise ValueError("Invalid object format: no null terminator")
    
    header = decompressed[:null_pos].decode('utf-8')
    content = decompressed[null_pos + 1:]
    
    # Parse header: "type size"
    header_parts = header.split(' ', 1)
    if len(header_parts) != 2:
        raise ValueError(f"Invalid header format: {header}")
    
    object_type, size_str = header_parts
    expected_size = int(size_str)
    if len(content) != expected_size:
        raise ValueError(f"Content size mismatch: expected {expected_size}, got {len(content)}")
    
    return object_type, content

# Constants for object types
BLOB_TYPE = "blob"
TREE_TYPE = "tree" 
COMMIT_TYPE = "commit"
SHA1_HEX_LENGTH = 40
```

**Object Storage Interface (`core/storage.py`):**
```python
from pathlib import Path
import os
from typing import Optional, tuple
from .hash_utils import format_object_for_storage, parse_stored_object

class ObjectStore:
    """Handles content-addressable object storage and retrieval."""
    
    def __init__(self, objects_dir: Path):
        self.objects_dir = objects_dir
        # Ensure objects directory exists
        self.objects_dir.mkdir(parents=True, exist_ok=True)
    
    def object_path_from_hash(self, object_hash: str) -> Path:
        """Convert hash to file system path: .git/objects/xx/yy..."""
        if len(object_hash) != SHA1_HEX_LENGTH:
            raise ValueError(f"Invalid hash length: {len(object_hash)}")
        
        dir_name = object_hash[:2]
        file_name = object_hash[2:]
        return self.objects_dir / dir_name / file_name
    
    def store_object(self, object_type: str, content: bytes) -> str:
        """Store object in content-addressable store, return hash."""
        # Format and compress object
        compressed_data = format_object_for_storage(object_type, content)
        
        # Compute hash for lookup
        object_hash = compute_object_hash(object_type, content)
        
        # Determine storage path
        object_path = self.object_path_from_hash(object_hash)
        
        # Create directory if needed
        object_path.parent.mkdir(exist_ok=True)
        
        # Write compressed object (only if not already exists)
        if not object_path.exists():
            object_path.write_bytes(compressed_data)
        
        return object_hash
    
    def retrieve_object(self, object_hash: str) -> tuple[str, bytes]:
        """Retrieve object content and type by hash."""
        object_path = self.object_path_from_hash(object_hash)
        
        if not object_path.exists():
            raise FileNotFoundError(f"Object {object_hash} not found")
        
        compressed_data = object_path.read_bytes()
        return parse_stored_object(compressed_data)
    
    def object_exists(self, object_hash: str) -> bool:
        """Check if object exists in store."""
        return self.object_path_from_hash(object_hash).exists()
```

#### Core Logic Skeleton Code

**Object Model Base Classes (`core/objects.py`):**
```python
from abc import ABC, abstractmethod
from typing import bytes, List, Optional, Tuple
import struct
import time

# Type aliases for clarity
TreeEntry = Tuple[str, str, str]  # (mode, name, hash)

class GitObject(ABC):
    """Base class for all Git objects."""
    
    def __init__(self, content: bytes):
        self.content = content
    
    @property
    @abstractmethod
    def object_type(self) -> str:
        """Return the Git object type string."""
        pass
    
    @abstractmethod
    def serialize(self) -> bytes:
        """Serialize object content for storage."""
        pass
    
    @classmethod
    @abstractmethod
    def deserialize(cls, content: bytes) -> 'GitObject':
        """Create object from stored content."""
        pass
    
    def compute_hash(self) -> str:
        """Compute SHA-1 hash of this object."""
        return compute_object_hash(self.object_type, self.serialize())

class BlobObject(GitObject):
    """Git blob object for file content storage."""
    
    def __init__(self, content: bytes):
        super().__init__(content)
    
    @property
    def object_type(self) -> str:
        return BLOB_TYPE
    
    def serialize(self) -> bytes:
        """Serialize blob content - content is stored as-is."""
        # TODO 1: Return the raw content bytes without modification
        # TODO 2: Blobs store file content exactly as it exists
        # Hint: self.content already contains the file bytes
        pass
    
    @classmethod
    def deserialize(cls, content: bytes) -> 'BlobObject':
        """Create blob from stored content."""
        # TODO 1: Create new BlobObject with the provided content
        # TODO 2: Blob deserialization is trivial - content is stored as-is
        # Hint: Simply pass content to constructor
        pass

class TreeObject(GitObject):
    """Git tree object for directory structure."""
    
    def __init__(self, entries: List[TreeEntry]):
        self.entries = entries
        # Sort entries to ensure consistent hashing
        self._sort_entries()
        super().__init__(self._serialize_entries())
    
    @property
    def object_type(self) -> str:
        return TREE_TYPE
    
    def _sort_entries(self):
        """Sort tree entries according to Git rules."""
        # TODO 1: Sort entries lexicographically by name
        # TODO 2: Treat directories as if their names end with '/'
        # TODO 3: Use byte-level comparison, not locale-specific sorting
        # Hint: key=lambda entry: entry[1] + ('/' if entry[0] == '40000' else '')
        pass
    
    def _serialize_entries(self) -> bytes:
        """Serialize tree entries to binary format."""
        # TODO 1: For each entry, format as: {mode} {name}\0{20-byte-hash}
        # TODO 2: Mode is ASCII string, name is UTF-8, hash is binary
        # TODO 3: Convert hex hash to 20-byte binary using bytes.fromhex()
        # TODO 4: Concatenate all entries into single bytes object
        # Hint: Use bytes.fromhex() to convert hash from hex to binary
        pass
    
    def serialize(self) -> bytes:
        """Serialize tree content."""
        return self.content
    
    @classmethod
    def deserialize(cls, content: bytes) -> 'TreeObject':
        """Create tree from stored binary content."""
        entries = []
        pos = 0
        
        while pos < len(content):
            # TODO 1: Find space character separating mode from name
            # TODO 2: Extract mode as ASCII string
            # TODO 3: Find null terminator separating name from hash
            # TODO 4: Extract name as UTF-8 string
            # TODO 5: Extract next 20 bytes as binary hash
            # TODO 6: Convert binary hash to hex string for storage
            # TODO 7: Add (mode, name, hex_hash) tuple to entries list
            # TODO 8: Advance pos to next entry
            # Hint: Use content.find() to locate separators
            pass
        
        return cls(entries)
    
    def add_entry(self, mode: str, name: str, object_hash: str):
        """Add entry to tree and re-sort."""
        # TODO 1: Add new entry tuple to self.entries
        # TODO 2: Re-sort entries to maintain Git ordering
        # TODO 3: Re-serialize content to update self.content
        # Hint: Call self._sort_entries() and update self.content
        pass

class CommitObject(GitObject):
    """Git commit object for project snapshots."""
    
    def __init__(self, tree_hash: str, parent_hashes: List[str], 
                 author: str, committer: str, message: str,
                 author_timestamp: Optional[int] = None,
                 committer_timestamp: Optional[int] = None,
                 timezone: str = "+0000"):
        self.tree_hash = tree_hash
        self.parent_hashes = parent_hashes
        self.author = author
        self.committer = committer
        self.message = message
        self.author_timestamp = author_timestamp or int(time.time())
        self.committer_timestamp = committer_timestamp or int(time.time())
        self.timezone = timezone
        super().__init__(self._serialize_commit())
    
    @property
    def object_type(self) -> str:
        return COMMIT_TYPE
    
    def _format_person_line(self, name: str, timestamp: int, timezone: str) -> str:
        """Format author/committer line with timestamp."""
        # TODO 1: Format as: {name} {timestamp} {timezone}
        # TODO 2: timestamp is Unix epoch seconds as decimal string
        # TODO 3: timezone is ±HHMM format (e.g., +0000, -0500)
        # Hint: f"{name} {timestamp} {timezone}"
        pass
    
    def _serialize_commit(self) -> bytes:
        """Serialize commit to text format."""
        lines = []
        
        # TODO 1: Add tree line: "tree {hash}"
        # TODO 2: Add parent lines: "parent {hash}" for each parent
        # TODO 3: Add author line with formatted timestamp
        # TODO 4: Add committer line with formatted timestamp  
        # TODO 5: Add blank line separator
        # TODO 6: Add commit message (can be multiple lines)
        # TODO 7: Join all lines with \n and encode to UTF-8 bytes
        # Hint: Use self._format_person_line() for author/committer
        pass
    
    def serialize(self) -> bytes:
        """Serialize commit content."""
        return self.content
    
    @classmethod
    def deserialize(cls, content: bytes) -> 'CommitObject':
        """Create commit from stored text content."""
        text = content.decode('utf-8')
        lines = text.split('\n')
        
        # TODO 1: Parse tree hash from first line
        # TODO 2: Parse parent hashes from any "parent" lines
        # TODO 3: Parse author line and extract name, timestamp, timezone
        # TODO 4: Parse committer line and extract name, timestamp, timezone
        # TODO 5: Find blank line separating metadata from message
        # TODO 6: Extract message as remaining lines after blank line
        # TODO 7: Create CommitObject with parsed data
        # Hint: Use startswith() to identify line types
        pass
```

#### Milestone Checkpoints

**After Implementing Blob Objects:**
- Create a test file: `echo "Hello, Git!" > test.txt`
- Your implementation should compute the same hash as: `git hash-object test.txt`
- Expected hash: `d95f3ad14dee633a758d2e331151e950dd13e4ed`
- Verify blob content retrieval matches original file exactly

**After Implementing Tree Objects:**
- Create a simple directory structure with a few files
- Your tree serialization should produce consistent hashes for identical directory contents
- Test tree entry sorting by creating files in different orders
- Verify nested directories create separate tree objects

**After Implementing Commit Objects:**
- Create a commit object with known tree hash and timestamp
- Compare your commit hash with Git's output for the same tree/parent/message
- Test merge commit with multiple parents
- Verify author/committer timestamp parsing and formatting

#### Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Hash mismatch with Git | Incorrect object header format | Compare byte-by-byte with `git cat-file -p` | Check header format: `{type} {size}\0{content}` |
| Tree parsing errors | Binary hash vs hex confusion | Print hash lengths (should be 20 bytes in tree) | Use `bytes.fromhex()` when storing, `.hex()` when displaying |
| Commit timestamp issues | Wrong timezone format | Compare with `git log --pretty=fuller` | Use `{timestamp} ±HHMM` format |
| Object corruption | Text mode file operations | Check if binary files are corrupted | Always use binary mode for file I/O |
| Tree entry order wrong | Incorrect sorting algorithm | Compare sorted entries with `git ls-tree` | Sort lexicographically with directory special case |


## Object Store Design

> **Milestone(s):** This section provides the storage engine foundation for Milestone 2 (Object Storage - Blobs), Milestone 3 (Tree Objects), and Milestone 4 (Commit Objects), establishing the content-addressable storage patterns used throughout all subsequent milestones

### Mental Model: The Universal Library

Imagine a vast, magical library where books are never filed by title or author, but instead by their exact content. Every book receives a unique fingerprint derived from every word, punctuation mark, and space within its pages. This fingerprint becomes the book's permanent address on the shelf. If you know the fingerprint, you can instantly locate any book. If two people write identical books independently, they receive the same fingerprint and occupy the same shelf space—the library automatically deduplicates content.

This is **content-addressable storage**, the revolutionary concept at Git's core. Unlike traditional file systems that assign arbitrary names and locations to files, Git computes a cryptographic fingerprint (SHA-1 hash) of each piece of content and uses that fingerprint as both the content's identifier and storage location. This approach provides several powerful properties that make distributed version control possible.

The fingerprint is deterministic—the same content always produces the same hash, regardless of when or where it's computed. This enables perfect deduplication: storing the same file in multiple commits or branches requires no additional space. The fingerprint is also tamper-evident: changing even a single bit of content produces a completely different hash, making corruption or malicious modification immediately detectable.

In Git's universal library, every piece of content becomes an **immutable object**. Once stored, objects never change—they can only be referenced by their hash. This immutability enables Git's powerful branching and merging capabilities: you never lose data by creating branches or switching between them, because every version of every file remains permanently accessible through its unique fingerprint.

The object store serves as the foundation layer for all Git operations. When you stage a file, Git computes its hash and stores it as a blob object. When you commit changes, Git creates tree objects that organize these blobs into directory structures, then creates a commit object that points to the root tree. Every object is stored once and referenced by its SHA-1 hash throughout the repository.

### Storage Algorithm

The object storage process transforms arbitrary content into an immutable, verifiable object within Git's content-addressable store. This multi-step algorithm ensures consistency, integrity, and efficient retrieval across all Git operations.

**Step 1: Object Header Construction**

Git wraps every piece of content with a standardized header before computing its hash. The header follows the format `{type} {size}\0{content}` where type is one of `blob`, `tree`, or `commit`, size is the decimal byte count of the raw content, and a null byte separates the header from content. This header serves multiple purposes: it prevents hash collisions between different object types containing identical content, enables type verification during retrieval, and provides size information for memory allocation and corruption detection.

For a text file containing "Hello, World!" (13 bytes), the complete object becomes `blob 13\0Hello, World!` where `\0` represents the null byte separator. This formatted object, not just the original content, becomes the input for hash calculation.

**Step 2: SHA-1 Hash Computation**

Git computes the SHA-1 cryptographic hash of the complete formatted object (header plus content) to generate a 160-bit digest, represented as a 40-character hexadecimal string. This hash serves as the object's immutable identifier throughout its lifetime. The SHA-1 algorithm ensures that identical content produces identical hashes deterministically, while any content modification results in a completely different hash.

The hash computation must be precise and consistent across all Git implementations. Python's `hashlib.sha1()` function provides the standard implementation, accepting the complete object bytes and returning the hexadecimal digest. This hash becomes the object's permanent name within the repository—it can never be changed without detecting the modification.

**Step 3: Content Compression**

Git compresses every object using the zlib compression algorithm before storage. This compression occurs after hash computation, ensuring the hash reflects the actual content while storage benefits from size reduction. The zlib algorithm typically achieves significant compression ratios for text-based content like source code, documentation, and configuration files commonly stored in version control.

The compression level balances storage space against compression time. Git uses zlib's default compression level (6 on a scale of 0-9), providing good compression ratios without excessive CPU overhead during write operations. The compressed data becomes the actual bytes written to the file system.

**Step 4: File System Path Generation**

Git derives the storage path from the object's SHA-1 hash using a two-level directory structure. The first two hexadecimal characters become the directory name within `.git/objects/`, while the remaining 38 characters become the filename. This sharding approach distributes objects across multiple directories, preventing performance degradation from storing thousands of files in a single directory.

For hash `a1b2c3d4e5f6789...`, the storage path becomes `.git/objects/a1/b2c3d4e5f6789...`. The directory `a1` must exist before writing the object file. Most file systems handle directories with hundreds of files efficiently, making this two-level approach sufficient for most repositories.

**Step 5: Atomic File Writing**

Git writes objects atomically to prevent corruption from interrupted operations or concurrent access. The atomic write process creates a temporary file with a unique name in the target directory, writes the compressed object content, ensures data reaches persistent storage through `fsync()`, then atomically renames the temporary file to the final object name.

This atomic operation ensures that object files are either completely present and valid or completely absent—partially written objects cannot exist. If a write operation fails at any point, the temporary file can be safely removed without affecting existing repository state.

The following table details the complete object storage algorithm:

| Step | Operation | Input | Output | Purpose |
|------|-----------|--------|---------|---------|
| 1 | Header Construction | `(type, content)` | `formatted_object` | Standardize object format |
| 2 | Hash Computation | `formatted_object` | `sha1_hash` | Generate unique identifier |
| 3 | Compression | `formatted_object` | `compressed_data` | Reduce storage space |
| 4 | Path Generation | `sha1_hash` | `file_path` | Determine storage location |
| 5 | Atomic Write | `(file_path, compressed_data)` | `stored_object` | Persist object safely |

### Retrieval and Verification

Object retrieval reverses the storage process, locating objects by their SHA-1 hash and reconstructing the original content with integrity verification. This process must handle missing objects gracefully and detect corruption reliably.

**Hash-to-Path Resolution**

The retrieval process begins by converting the 40-character SHA-1 hash into a file system path using the same sharding logic as storage. The first two characters specify the subdirectory within `.git/objects/`, and the remaining 38 characters identify the object file. The complete path becomes `.git/objects/XX/YYYYYY...` where XX and YYYY represent the hash prefix and suffix respectively.

Before attempting to read the object file, the retrieval system should verify that both the subdirectory and object file exist. Missing subdirectories indicate the object has never been stored, while missing object files within existing subdirectories may indicate corruption or incomplete write operations.

**Decompression and Parsing**

Once located, the object file contains zlib-compressed data that must be decompressed to recover the original formatted object. The decompression operation may fail if the file is corrupted or truncated, requiring appropriate error handling to report repository corruption.

After successful decompression, the formatted object must be parsed to separate the header from content. The parser locates the null byte separator, extracts the type and size information from the header, then validates that the declared size matches the actual content length. This validation detects various corruption scenarios including truncated objects and header modification.

**Content Verification**

The most critical aspect of object retrieval is verifying that the retrieved content matches its claimed identity. Git recomputes the SHA-1 hash of the decompressed, formatted object and compares it to the hash used for retrieval. Any mismatch indicates corruption and must be treated as a fatal error.

This verification step provides Git's fundamental integrity guarantee: if an object can be retrieved successfully, its content is guaranteed to be identical to when it was stored. This property enables distributed collaboration without central authority—every participant can independently verify the integrity of shared history.

The following table outlines the complete retrieval and verification process:

| Operation | Input | Process | Output | Error Conditions |
|-----------|-------|---------|---------|------------------|
| Path Resolution | `sha1_hash` | Split hash into directory/file components | `file_path` | Invalid hash format |
| File Reading | `file_path` | Read compressed bytes from file system | `compressed_data` | File not found, permission denied |
| Decompression | `compressed_data` | zlib decompression | `formatted_object` | Corruption, truncation |
| Header Parsing | `formatted_object` | Extract type, size, content | `(type, size, content)` | Invalid format, size mismatch |
| Hash Verification | `formatted_object` | Recompute SHA-1, compare to expected | `verified_content` | Hash mismatch (corruption) |

**Error Recovery Strategies**

When object retrieval fails, Git provides several diagnostic capabilities to help identify the root cause. Missing object files typically indicate incomplete repository clones or damaged file systems. Decompression failures suggest file corruption or storage hardware issues. Hash verification failures represent the most serious corruption type, indicating that stored content differs from its claimed identity.

For missing objects, Git can attempt to retrieve them from alternate object databases (alternates mechanism) or remote repositories. For corrupted objects, no automatic recovery is possible—the object must be restored from backup or reconstructed from other repository copies.

### Architecture Decision Records

The object store design involves several critical architecture decisions that fundamentally shape Git's behavior, performance characteristics, and security properties. Each decision represents careful consideration of trade-offs between competing requirements.

> **Decision: SHA-1 Hash Algorithm Selection**
> - **Context**: Git requires a cryptographic hash function to generate unique, tamper-evident identifiers for all stored content. The hash function must provide strong collision resistance, fast computation, and consistent cross-platform behavior.
> - **Options Considered**: MD5 (fast but cryptographically broken), SHA-1 (established but showing weaknesses), SHA-256 (stronger but larger hashes)
> - **Decision**: SHA-1 for initial implementation, with migration path to SHA-256
> - **Rationale**: SHA-1 provides adequate collision resistance for version control use cases while maintaining compatibility with existing Git repositories. The 160-bit hash size balances security with storage efficiency. Although theoretical weaknesses exist, practical collision attacks remain extremely difficult and detectable.
> - **Consequences**: Enables compatibility with existing Git tools and repositories. Provides strong integrity guarantees for typical use cases. May require migration to SHA-256 in future for enhanced security as collision attacks become more practical.

The following table compares the hash algorithm options:

| Algorithm | Digest Size | Performance | Collision Resistance | Compatibility | Chosen |
|-----------|-------------|-------------|---------------------|---------------|---------|
| MD5 | 128 bits | Fastest | Broken | Legacy only | ❌ |
| SHA-1 | 160 bits | Fast | Weakening | Full Git compatibility | ✅ |
| SHA-256 | 256 bits | Slower | Strong | Limited Git support | Future |

> **Decision: Two-Level Directory Structure**
> - **Context**: Storing thousands of objects in a single directory causes severe performance degradation on most file systems. The storage structure must distribute objects efficiently while maintaining simple hash-to-path mapping.
> - **Options Considered**: Single flat directory (simple but slow), two-level sharding (balanced), three-level sharding (complex but scalable)
> - **Decision**: Two-level directory structure using first two hex characters as directory name
> - **Rationale**: Provides 256 possible subdirectories, distributing objects evenly across directories. Most repositories contain fewer than 10,000 objects per subdirectory, maintaining good file system performance. Simple mapping algorithm enables fast path computation.
> - **Consequences**: Excellent performance for repositories of all practical sizes. Simple implementation with minimal complexity overhead. Requires directory pre-creation but provides efficient object distribution.

| Structure | Directories | Objects per Dir | Lookup Speed | Implementation | Chosen |
|-----------|-------------|-----------------|--------------|----------------|---------|
| Flat | 1 | Unlimited | Very slow | Trivial | ❌ |
| Two-level | 256 | ~400 (typical) | Fast | Simple | ✅ |
| Three-level | 4,096 | ~25 (typical) | Fastest | Complex | ❌ |

> **Decision: zlib Compression Algorithm**
> - **Context**: Version control repositories often contain repetitive text content that compresses well. Compression reduces storage requirements but adds CPU overhead for every object operation.
> - **Options Considered**: No compression (fast but large), gzip/zlib (balanced), specialized algorithms (complex)
> - **Decision**: zlib compression with default level (6)
> - **Rationale**: zlib provides excellent compression ratios for typical source code content while maintaining fast compression/decompression speed. Wide platform availability and mature implementations reduce compatibility risks. Default compression level balances space savings with CPU cost.
> - **Consequences**: Significant storage space reduction (often 50-80% for text content). Minimal performance impact on modern systems. Universal compatibility across platforms and programming languages.

| Algorithm | Compression Ratio | Speed | Availability | Complexity | Chosen |
|-----------|------------------|-------|--------------|------------|---------|
| None | 0% | Fastest | Universal | Minimal | ❌ |
| zlib (level 6) | 60-80% | Fast | Universal | Low | ✅ |
| LZMA | 70-85% | Slow | Limited | High | ❌ |

> **Decision: Immutable Object Model**
> - **Context**: Version control systems must preserve historical content while enabling efficient storage and reliable integrity verification. Objects could be mutable (updateable) or immutable (write-once).
> - **Options Considered**: Mutable objects with versioning, immutable objects with content-addressing, hybrid approach with mutable metadata
> - **Decision**: Fully immutable objects identified by content hash
> - **Rationale**: Immutability provides strong integrity guarantees—content cannot be modified without detection. Content-addressing enables automatic deduplication and efficient comparison operations. Simplifies concurrent access by eliminating modification conflicts.
> - **Consequences**: Perfect integrity preservation for historical data. Automatic deduplication across branches and repositories. Simplified concurrency model. Storage growth over time as objects accumulate, requiring periodic cleanup.

| Model | Integrity | Deduplication | Concurrency | Storage Growth | Chosen |
|-------|-----------|---------------|-------------|----------------|---------|
| Mutable | Moderate | Manual | Complex locking | Controlled | ❌ |
| Immutable | Perfect | Automatic | Lock-free reads | Unbounded | ✅ |
| Hybrid | Variable | Partial | Mixed complexity | Moderate | ❌ |

### Common Pitfalls

Building a content-addressable object store presents several subtle challenges that frequently trip up implementers. Understanding these pitfalls and their solutions is crucial for creating a reliable Git implementation.

⚠️ **Pitfall: Including Hash in Header During Hash Computation**

Many implementers mistakenly include the object's SHA-1 hash within the object header when computing the hash, creating an impossible circular dependency. The hash cannot be computed until the complete object is formed, but the object cannot be complete if it includes its own hash.

Git's object format includes only the object type and content size in the header, never the hash itself. The hash is computed from the complete formatted object (`{type} {size}\0{content}`) and then used separately as the object's identifier and storage path. The hash never becomes part of the stored object content.

To avoid this pitfall, ensure your hash computation function receives the complete formatted object without any hash reference, computes the SHA-1 digest, then uses that digest for storage path generation and object identification.

⚠️ **Pitfall: Hash Computation on Content Only (Excluding Header)**

Some implementations compute the SHA-1 hash using only the raw content bytes, ignoring the type and size header. This approach fails to match Git's hash computation and can lead to hash collisions between different object types containing identical content.

Git always computes hashes on the complete formatted object including the header. A blob containing "tree abc" and a tree containing the same bytes produce different hashes because their headers differ (`blob 8\0tree abc` vs `tree 8\0tree abc`). This separation prevents type confusion attacks and ensures object type integrity.

Verify your hash computation includes the complete object header by testing with Git's `hash-object` command and comparing results. Your implementation must produce identical hashes for identical input.

⚠️ **Pitfall: Compressing Content Before Hash Computation**

The sequence of hash computation and compression operations is critical but often confused. Computing the hash of compressed content instead of the original formatted object produces incorrect hashes that don't match Git's expectations.

Git's algorithm always follows this sequence: format object with header, compute SHA-1 hash of formatted object, compress formatted object, store compressed data. The hash reflects the uncompressed content, while storage benefits from compression. This approach allows hash computation without decompression during future retrieval operations.

Ensure your storage function computes the hash first, stores the hash for later use, then compresses the same formatted object for file system storage.

⚠️ **Pitfall: Incorrect Directory Creation Permissions**

Object directories within `.git/objects/` require specific permissions to function correctly across different operating systems and deployment scenarios. Using default directory permissions can cause access failures in shared repository environments.

Git creates object directories with 0755 permissions (owner read/write/execute, group and other read/execute), ensuring they remain accessible to the repository owner while allowing read access for other authorized users. Execute permission on directories is required for file access within the directory.

Set directory permissions explicitly using your platform's appropriate mechanism (`os.mkdir(path, 0o755)` in Python) rather than relying on default umask behavior.

⚠️ **Pitfall: Non-Atomic Object Writing**

Writing object files directly to their final location creates a race condition where other processes might read partially written objects, leading to corruption errors. This is particularly problematic during concurrent Git operations or system crashes during writes.

Git ensures atomicity by writing to temporary files first, then atomically renaming them to the final object name. This approach guarantees that object files are either completely absent or completely valid—partially written objects cannot exist.

Implement atomic writes using a temporary filename (such as appending `.tmp` plus process ID), write complete content to the temporary file, sync to disk, then rename to the final object filename.

⚠️ **Pitfall: Missing fsync() for Durability**

Object files must be durably written to persistent storage before being considered successfully stored. Without explicit synchronization, object data may remain in file system buffers and be lost during system crashes, leading to repository corruption.

After writing the complete object file, call `fsync()` (or platform equivalent) to ensure data reaches persistent storage before proceeding. This operation may significantly impact write performance but is essential for repository integrity.

Include explicit sync operations in your atomic write sequence: write to temporary file, sync temporary file, rename to final name, sync parent directory to ensure directory entry is persistent.

⚠️ **Pitfall: Inadequate Error Handling During Retrieval**

Object retrieval involves multiple failure modes that require different handling strategies. Treating all retrieval failures identically can mask serious corruption issues or provide misleading error messages to users.

Distinguish between expected failures (missing objects that haven't been stored) and unexpected failures (corruption, permission issues, file system errors). Missing objects should be handled gracefully with appropriate user feedback, while corruption should be treated as fatal errors requiring user intervention.

Implement comprehensive error categorization that identifies the specific failure mode and provides appropriate recovery suggestions for each case.

### Implementation Guidance

The object store implementation requires careful attention to file system operations, cryptographic hashing, and data integrity verification. This section provides complete infrastructure code and skeletal implementations for the core learning objectives.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Hashing | `hashlib.sha1()` (built-in) | `cryptography` library for future SHA-256 |
| Compression | `zlib` module (built-in) | `python-lzo` for better performance |
| File Operations | `pathlib.Path` + `open()` | `os.open()` with explicit flags |
| Atomic Writes | `tempfile` + `os.rename()` | Platform-specific atomic operations |
| Directory Creation | `os.makedirs()` | `os.mkdir()` with explicit permission handling |

#### Recommended File Structure

```
git-implementation/
  src/
    git_core/
      __init__.py
      objects/                  ← Object store implementation
        __init__.py
        store.py               ← ObjectStore class (core learning)
        types.py               ← GitObject base classes
        hash.py                ← Hashing utilities (infrastructure)
        compression.py         ← Compression utilities (infrastructure)
      repository.py            ← Repository class
      exceptions.py            ← Git-specific exceptions
  tests/
    test_objects/
      test_store.py            ← Object store tests
      test_hash.py             ← Hash computation tests
    fixtures/                  ← Test data files
```

#### Infrastructure Starter Code

**Hash Computation Utilities** (`src/git_core/objects/hash.py`):
```python
"""
Hash computation utilities for Git objects.
Provides SHA-1 computation with Git's object header format.
"""
import hashlib
from typing import Tuple

SHA1_HEX_LENGTH = 40
OBJECT_HEADER_FORMAT = "{type} {size}\0{content}"

def compute_sha1(content: bytes) -> str:
    """Compute SHA-1 hash of raw bytes, returning hex digest."""
    return hashlib.sha1(content).hexdigest()

def format_git_object(object_type: str, content: bytes) -> bytes:
    """Format content with Git object header: '{type} {size}\\0{content}'"""
    header = f"{object_type} {len(content)}"
    return header.encode('utf-8') + b'\0' + content

def compute_object_hash(object_type: str, content: bytes) -> str:
    """Compute Git object hash including type/size header."""
    formatted_object = format_git_object(object_type, content)
    return compute_sha1(formatted_object)

def parse_git_object(formatted_object: bytes) -> Tuple[str, int, bytes]:
    """Parse formatted Git object, returning (type, size, content)."""
    null_index = formatted_object.find(b'\0')
    if null_index == -1:
        raise ValueError("Invalid Git object: missing null separator")
    
    header = formatted_object[:null_index].decode('utf-8')
    content = formatted_object[null_index + 1:]
    
    try:
        object_type, size_str = header.split(' ', 1)
        declared_size = int(size_str)
    except ValueError:
        raise ValueError(f"Invalid Git object header: {header}")
    
    if len(content) != declared_size:
        raise ValueError(f"Content size mismatch: declared {declared_size}, actual {len(content)}")
    
    return object_type, declared_size, content

def validate_sha1_hash(hash_str: str) -> bool:
    """Validate SHA-1 hash format (40 hex characters)."""
    return (len(hash_str) == SHA1_HEX_LENGTH and 
            all(c in '0123456789abcdef' for c in hash_str.lower()))
```

**Compression Utilities** (`src/git_core/objects/compression.py`):
```python
"""
Compression utilities for Git object storage.
Handles zlib compression/decompression with error handling.
"""
import zlib
from typing import bytes

DEFAULT_COMPRESSION_LEVEL = 6

def compress_object(data: bytes, level: int = DEFAULT_COMPRESSION_LEVEL) -> bytes:
    """Compress data using zlib with specified compression level."""
    try:
        return zlib.compress(data, level)
    except zlib.error as e:
        raise ValueError(f"Compression failed: {e}")

def decompress_object(compressed_data: bytes) -> bytes:
    """Decompress zlib-compressed data."""
    try:
        return zlib.decompress(compressed_data)
    except zlib.error as e:
        raise ValueError(f"Decompression failed: {e}")
```

**Git Object Types** (`src/git_core/objects/types.py`):
```python
"""
Git object type definitions and base classes.
"""
from abc import ABC, abstractmethod
from typing import bytes

BLOB_TYPE = "blob"
TREE_TYPE = "tree" 
COMMIT_TYPE = "commit"

class GitObject(ABC):
    """Base class for all Git objects (blob, tree, commit)."""
    
    def __init__(self, content: bytes):
        self.content = content
    
    @property
    @abstractmethod
    def object_type(self) -> str:
        """Return the Git object type string."""
        pass
    
    def compute_hash(self) -> str:
        """Compute the SHA-1 hash of this object."""
        from .hash import compute_object_hash
        return compute_object_hash(self.object_type, self.content)

class BlobObject(GitObject):
    """Git blob object representing file content."""
    
    @property
    def object_type(self) -> str:
        return BLOB_TYPE

class TreeObject(GitObject):
    """Git tree object representing directory structure."""
    
    @property
    def object_type(self) -> str:
        return TREE_TYPE

class CommitObject(GitObject):
    """Git commit object representing a project snapshot."""
    
    @property
    def object_type(self) -> str:
        return COMMIT_TYPE
```

#### Core Logic Skeleton Code

**Object Store Implementation** (`src/git_core/objects/store.py`):
```python
"""
Git object store implementation.
Core content-addressable storage for Git objects.
"""
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, bytes

from .hash import compute_object_hash, parse_git_object, validate_sha1_hash, format_git_object
from .compression import compress_object, decompress_object

class ObjectStore:
    """Content-addressable storage for Git objects."""
    
    def __init__(self, objects_dir: Path):
        self.objects_dir = objects_dir
        self._ensure_objects_directory()
    
    def _ensure_objects_directory(self) -> None:
        """Create .git/objects directory structure if it doesn't exist."""
        # TODO 1: Create objects_dir with 0o755 permissions if it doesn't exist
        # TODO 2: Create info/ and pack/ subdirectories for Git compatibility
        # Hint: Use os.makedirs() with exist_ok=True
        pass
    
    def object_path_from_hash(self, object_hash: str) -> Path:
        """Convert SHA-1 hash to file system path in objects directory."""
        # TODO 1: Validate hash format using validate_sha1_hash()
        # TODO 2: Split hash into directory (first 2 chars) and filename (remaining 38)
        # TODO 3: Return Path to .git/objects/XX/YYYYYYYY...
        # Hint: hash[:2] gives first 2 characters, hash[2:] gives remainder
        pass
    
    def store_object(self, object_type: str, content: bytes) -> str:
        """Store object in content-addressable store, return SHA-1 hash."""
        # TODO 1: Format object with header using format_git_object()
        # TODO 2: Compute SHA-1 hash of formatted object
        # TODO 3: Compress formatted object using compress_object()
        # TODO 4: Generate storage path from hash using object_path_from_hash()
        # TODO 5: Create parent directory if it doesn't exist (with 0o755 permissions)
        # TODO 6: Write compressed data atomically using _write_object_file()
        # TODO 7: Return computed SHA-1 hash
        # Hint: Atomic write prevents corruption from interrupted operations
        pass
    
    def retrieve_object(self, object_hash: str) -> Tuple[str, bytes]:
        """Retrieve object by hash, return (object_type, content)."""
        # TODO 1: Validate hash format
        # TODO 2: Generate file path from hash
        # TODO 3: Check if object file exists, raise exception if missing
        # TODO 4: Read compressed data from file
        # TODO 5: Decompress data using decompress_object()
        # TODO 6: Parse object header using parse_git_object()
        # TODO 7: Verify integrity by recomputing hash and comparing
        # TODO 8: Return (object_type, content) tuple
        # Hint: Hash verification detects corruption
        pass
    
    def object_exists(self, object_hash: str) -> bool:
        """Check if object exists in store without reading content."""
        # TODO 1: Validate hash format
        # TODO 2: Generate file path from hash
        # TODO 3: Return whether file exists using Path.exists()
        # Hint: This is more efficient than retrieve_object() for existence checks
        pass
    
    def _write_object_file(self, file_path: Path, compressed_data: bytes) -> None:
        """Atomically write compressed object data to file."""
        # TODO 1: Create temporary file in same directory as target
        # TODO 2: Write compressed_data to temporary file
        # TODO 3: Sync temporary file to disk (fsync)
        # TODO 4: Atomically rename temporary file to final path
        # TODO 5: Handle any errors by cleaning up temporary file
        # Hint: Use tempfile.NamedTemporaryFile() with delete=False
        # Hint: os.rename() is atomic on most platforms
        pass
    
    def _create_object_directory(self, directory_path: Path) -> None:
        """Create object subdirectory with correct permissions."""
        # TODO 1: Create directory if it doesn't exist
        # TODO 2: Set permissions to 0o755
        # TODO 3: Handle race condition if directory is created concurrently
        # Hint: Use os.makedirs() with exist_ok=True
        pass
```

#### Language-Specific Hints

**Python-Specific Implementation Details:**

- Use `pathlib.Path` for all file system operations—it provides cross-platform path handling and convenient methods like `.exists()` and `.mkdir()`
- The `os.rename()` function provides atomic file renaming on POSIX systems, but use `shutil.move()` on Windows for cross-platform compatibility
- Use `tempfile.NamedTemporaryFile(delete=False)` to create temporary files that persist after closing, enabling atomic rename operations
- Call `file.flush()` followed by `os.fsync(file.fileno())` to ensure data reaches persistent storage
- Use `os.makedirs(path, mode=0o755, exist_ok=True)` to create directories with specific permissions while handling concurrent creation

**Error Handling Patterns:**

- Distinguish between `FileNotFoundError` (missing object) and `PermissionError` (access issue) when reading objects
- Catch `zlib.error` during compression/decompression and convert to more descriptive exceptions
- Use try/finally blocks around temporary file operations to ensure cleanup on failure
- Validate input parameters (hash format, object type) before performing expensive operations

#### Milestone Checkpoint

After implementing the object store, verify functionality with these tests:

**Basic Storage and Retrieval:**
```bash
python -m pytest tests/test_objects/test_store.py::test_store_blob -v
```
Expected behavior: Store a simple text blob, retrieve it, verify content matches exactly.

**Hash Computation Verification:**
```bash
# Compare with Git's hash-object command
echo "Hello, World!" | git hash-object --stdin
# Your implementation should produce: 8ab686eafeb1f44702738c8b0f24f2567c36da6d
```

**Directory Structure Validation:**
After storing an object with hash `8ab686eafeb1f44702738c8b0f24f2567c36da6d`, verify:
- Directory `.git/objects/8a/` exists with 0755 permissions  
- File `.git/objects/8a/b686eafeb1f44702738c8b0f24f2567c36da6d` contains compressed data
- File can be decompressed to recover original formatted object

**Integrity Verification:**
```bash
python -c "
store = ObjectStore(Path('.git/objects'))
hash1 = store.store_object('blob', b'test content')
hash2 = store.store_object('blob', b'test content')
assert hash1 == hash2  # Same content produces same hash
"
```

**Signs of Problems:**
- Hash mismatches indicate incorrect header formatting or hash computation sequence
- Permission errors suggest incorrect directory creation or file permissions
- Decompression failures indicate compression/decompression sequence problems
- Missing files after storage suggest atomic write implementation issues


## Index and Staging Area

> **Milestone(s):** This section is crucial for Milestone 6 (Index/Staging Area) and provides the foundation for Milestone 7 (Diff Algorithm) and Milestone 8 (Merge) which rely heavily on index operations and status calculations

### Mental Model: The Photography Dark Room

Think of Git's staging area as a traditional photographer's darkroom where film is developed into photographs. In the days of film photography, taking a picture was just the beginning of a multi-stage process. After shooting a roll of film, the photographer would bring it into the darkroom to develop the negatives, examine each shot under a red light, choose which ones to print, make adjustments to exposure and contrast, and only then commit to creating the final prints.

The staging area works exactly like this darkroom process. Your **working directory** is like your camera - you can take as many shots as you want, modify files, experiment with changes, but nothing is permanent yet. The **staging area** (also called the index) is your darkroom where you carefully review each change, decide which modifications should be included in your next commit, and prepare them for final development. The **repository** is like your photo album - once you commit (develop and print), those changes become part of your permanent project history.

Just as a photographer might take dozens of shots but only print their best work, you might make changes to many files in your working directory but selectively stage only specific changes for each commit. This selective staging allows you to craft meaningful, focused commits rather than dumping all your work-in-progress changes into the repository at once.

The key insight is that the staging area gives you a **preparation and review step** before committing to permanent history. You can stage a file, make more changes to it, stage those additional changes, or even unstage files if you decide they're not ready. This three-stage workflow (working directory → staging area → repository) is what makes Git so powerful for maintaining clean, logical project history.

### Index Binary Format

The Git index is stored as a binary file at `.git/index` and serves as the implementation of the staging area. Unlike most Git internal files which are human-readable text, the index uses a compact binary format for performance reasons - Git needs to quickly read, write, and search through potentially thousands of file entries during common operations like `git status` and `git add`.

The index file follows a strict binary layout that begins with a fixed header, followed by a sorted array of file entries, and concludes with a SHA-1 checksum of the entire file content. This format enables Git to efficiently detect when files have been modified by comparing cached metadata against the current file system state.

**Index File Header Structure:**

| Field | Size (bytes) | Type | Description |
|-------|-------------|------|-------------|
| Signature | 4 | char[4] | Always "DIRC" (directory cache) |
| Version | 4 | uint32 | Format version number (we use version 2) |
| Entry Count | 4 | uint32 | Number of index entries that follow |

The header's 12-byte fixed size allows Git to quickly determine how many entries to expect when parsing the file. The "DIRC" signature serves as both a magic number for file type identification and a corruption detection mechanism - if these bytes are corrupted, Git immediately knows the index is invalid.

**Index Entry Structure:**

Each index entry represents a single staged file and contains both the file's content hash and cached file system metadata. This dual information allows Git to quickly determine whether a file has been modified since it was last staged, without having to re-read and hash the entire file content.

| Field | Size (bytes) | Type | Description |
|-------|-------------|------|-------------|
| ctime seconds | 4 | uint32 | File creation time (seconds since Unix epoch) |
| ctime nanoseconds | 4 | uint32 | File creation time fractional seconds |
| mtime seconds | 4 | uint32 | File modification time (seconds since Unix epoch) |
| mtime nanoseconds | 4 | uint32 | File modification time fractional seconds |
| Device ID | 4 | uint32 | Device ID containing the file |
| Inode | 4 | uint32 | File system inode number |
| Mode | 4 | uint32 | File mode and permissions (0o100644 for regular files) |
| UID | 4 | uint32 | User ID of file owner |
| GID | 4 | uint32 | Group ID of file owner |
| File Size | 4 | uint32 | Size of file content in bytes |
| SHA-1 Hash | 20 | char[20] | Object hash of staged file content (binary, not hex) |
| Flags | 2 | uint16 | Status flags including name length |
| Path Name | variable | char[] | File path relative to repository root |
| Padding | 0-3 | char[] | NUL bytes to align entry to 4-byte boundary |

The extensive metadata caching in each entry enables Git's famous performance. When you run `git status`, Git can quickly scan through the index entries and compare the cached timestamps and file sizes against the current working directory. Only files whose metadata has changed need to be re-read and hashed to determine if their content has actually been modified.

> **Critical Implementation Detail**: The SHA-1 hash is stored as 20 binary bytes, not as 40 hexadecimal characters. This is a common source of bugs - you must convert between binary and hex representations when reading/writing the index versus displaying hashes to users.

**Index Entry Sorting and Padding:**

Index entries must be stored in **lexicographic order** by their path name. This sorting enables Git to use binary search when looking up specific files and ensures consistent behavior across different implementations. The sorting is performed on the raw byte values of the UTF-8 encoded path strings.

Each entry's total size must be a multiple of 4 bytes for memory alignment performance. After the variable-length path name, NUL padding bytes are added to reach the next 4-byte boundary. The entry size calculation is: `62 + path_length + padding_to_4_byte_boundary`.

**Index Checksum:**

The index file concludes with a 20-byte SHA-1 checksum of all preceding content. This checksum serves multiple purposes:

1. **Corruption Detection**: Git can verify the index hasn't been corrupted by disk errors or incomplete writes
2. **Concurrent Access Safety**: Multiple Git processes can detect when another process has modified the index
3. **Atomic Updates**: Git writes to a temporary file then renames it, ensuring the index is never in a partially-written state

The checksum is calculated over the entire file content except for the checksum bytes themselves.

> **Decision: Binary Format vs Text Format**
> - **Context**: The staging area needs to store metadata for potentially thousands of files and be read/written frequently during common Git operations
> - **Options Considered**: 
>   - Text format (similar to .gitignore): Human readable, easy to debug, simple to parse
>   - Binary format: Compact storage, faster parsing, fixed-width fields enable efficient seeking
>   - Hybrid approach: Text metadata with binary hashes
> - **Decision**: Binary format with fixed-width header and entries
> - **Rationale**: Performance is critical for the staging area since it's accessed by almost every Git command. Binary format provides 3-5x faster parsing and 2x smaller file size compared to text alternatives. The complexity cost is justified by the performance benefits for large repositories.
> - **Consequences**: More complex to implement and debug, but enables Git's famous speed even in repositories with thousands of files

### Add and Remove Operations

The core staging area operations are adding files (staging changes) and removing files (unstaging changes). These operations maintain the index's binary format and sorted ordering while updating both the object store and the cached metadata.

**Add Operation Algorithm:**

The `git add` operation stages a file's current content and metadata into the index. This involves both storing the file content as a blob object and creating or updating the corresponding index entry.

1. **Read File Content**: Read the complete file content from the working directory. Handle binary files correctly by reading raw bytes without text encoding assumptions.

2. **Create Blob Object**: Compute the SHA-1 hash of the blob object using the `compute_object_hash` function with `BLOB_TYPE`. Store the compressed object in the object store using `store_object`.

3. **Gather File Metadata**: Use file system calls to collect the complete stat information including modification time, file size, inode number, device ID, permissions, and owner IDs. This metadata enables fast change detection later.

4. **Load Current Index**: Read and parse the existing `.git/index` file if it exists. If the repository has no staged files yet, start with an empty index structure.

5. **Update or Insert Entry**: Search for an existing entry with the same path name. If found, update its hash and metadata. If not found, create a new entry. The binary hash must be converted from the 40-character hex representation to 20 binary bytes.

6. **Maintain Sort Order**: Keep all index entries sorted lexicographically by path name. Insert new entries in the correct position or re-sort after updates.

7. **Write Updated Index**: Serialize the complete index structure back to binary format, calculate the SHA-1 checksum of the content, append the checksum, and atomically write to `.git/index` using a temporary file and rename operation.

**Add Operation Data Flow:**

| Component | Input | Processing | Output |
|-----------|-------|------------|--------|
| Working Directory | File path | Read file content and stat metadata | Raw bytes + file stats |
| Object Store | File content | Hash, compress, store as blob | Object hash |
| Index | Path + hash + stats | Update entry, maintain sort order | Updated index |
| File System | Binary index data | Atomic write with checksum | Persistent .git/index |

**Remove Operation Algorithm:**

The `git rm --cached` operation (unstaging) removes a file from the index without affecting the working directory. This is distinct from `git rm` which removes both the index entry and the working directory file.

1. **Load Current Index**: Read and parse the existing `.git/index` file. If the index doesn't exist or is empty, the remove operation is a no-op.

2. **Find Target Entry**: Search through the sorted index entries for the specified path name. Since entries are sorted, this can use binary search for efficiency in large repositories.

3. **Remove Entry**: Delete the matching entry from the index entry list. This does not affect the blob object in the object store (other commits might reference it).

4. **Update Entry Count**: Decrement the entry count in the index header to reflect the removal.

5. **Write Updated Index**: Serialize the modified index structure back to binary format and write it atomically. The checksum will change since the content has changed.

⚠️ **Pitfall: Object Store Cleanup**
Many implementations mistakenly delete the blob object when removing an index entry. This is incorrect - the object store is append-only and blob objects should never be deleted during normal operations. Other commits in the repository history might reference the same blob hash. Git's garbage collection process handles cleanup of unreferenced objects separately.

**Handling File Modifications:**

When a file is modified after being staged, Git needs to handle the scenario where the working directory version differs from the staged version. The add operation naturally handles this case:

1. The new file content gets a different SHA-1 hash than the previously staged version
2. The old blob object remains in the object store (it might be referenced by previous commits)
3. The new blob object is stored alongside the old one
4. The index entry is updated with the new hash and current metadata

This design means that staging a file multiple times creates multiple blob objects, but this is acceptable because content-addressable storage ensures identical content is never duplicated (same content always produces the same hash).

**Index Locking and Concurrency:**

Multiple Git commands might attempt to read or write the index simultaneously. Git uses a file-based locking mechanism to prevent corruption:

1. Before writing the index, Git creates a `.git/index.lock` file
2. The actual index updates are written to this lock file
3. When the update is complete, the lock file is atomically renamed to `.git/index`
4. Other Git processes that find an existing lock file wait or abort with an error

This locking protocol ensures that the index is never in a partially-written state and that concurrent operations don't corrupt each other's changes.

**Status Tracking for Modified Files:**

The cached metadata in index entries enables efficient change detection. When determining if a file has been modified since staging:

1. **Fast Path**: Compare cached mtime and file size against current values. If both match, the file is likely unchanged (skip expensive content hashing).

2. **Slow Path**: If metadata differs, read the file content, compute its SHA-1 hash, and compare against the staged hash. This definitively determines if content has changed.

3. **Race Condition Handling**: If a file is modified during the status check, the metadata comparison might give inconsistent results. Git handles this by re-checking files whose metadata changed during processing.

This two-level approach makes `git status` extremely fast even in large repositories, since most files are unchanged and can be verified using only metadata comparison.

![Commit Creation Sequence](./diagrams/commit-creation-flow.svg)

### Status Calculation

Git status determines the state of every file by performing a **three-way comparison** between the working directory, the staging area (index), and the current commit (HEAD). This comparison categorizes each file into one of several states that inform the user what changes are staged, what changes are unstaged, and what files are untracked.

Understanding status calculation is crucial because it forms the foundation for diff algorithms and merge operations. The status calculation must be both comprehensive (finding all relevant files) and efficient (fast enough for interactive use).

**Three-Way Comparison Overview:**

The status algorithm compares three different views of the project's files:

1. **HEAD Commit**: The files and content from the current branch's most recent commit
2. **Index (Staging Area)**: The files that have been staged for the next commit  
3. **Working Directory**: The current files in the file system

By comparing these three states pairwise, Git can determine exactly what changes have been made and what actions the user might want to take next.

**File State Categories:**

| State | Working Directory | Index | HEAD | User Action Needed |
|-------|------------------|-------|------|-------------------|
| Untracked | Present | Absent | Absent | `git add` to begin tracking |
| Added | Present | Present | Absent | `git commit` to confirm addition |
| Modified | Modified | Original | Original | `git add` to stage changes |
| Staged | Modified | Modified | Original | `git commit` to confirm changes |
| Deleted | Absent | Original | Present | `git add` to stage deletion |
| Renamed | New location | New location | Old location | `git commit` to confirm rename |
| Conflicted | Conflicted | Multiple stages | Original | Resolve conflicts manually |

**Status Calculation Algorithm:**

The status calculation follows a systematic approach to ensure no files are missed and all states are correctly identified:

1. **Collect File Sets**: Gather the complete set of file paths from all three sources:
   - Parse the HEAD commit tree to get all tracked files
   - Read the index to get all staged files  
   - Scan the working directory for all present files (respecting .gitignore rules)

2. **Create Union of Paths**: Combine all file paths from the three sources into a single sorted set. This ensures every file that exists in any state gets examined.

3. **Three-Way Comparison**: For each file path, determine its presence and content in each of the three states:
   - HEAD state: File hash from the commit tree (if present)
   - Index state: File hash from the index entry (if present)
   - Working directory state: Computed hash of current file content (if present)

4. **Apply Classification Rules**: Use the three-state comparison to classify each file according to the state table above.

5. **Handle Special Cases**: Process renames, ignored files, and submodule states according to Git's specific rules.

**Detailed State Classification:**

**Untracked Files:**
Files present in the working directory but absent from both the index and HEAD commit. These represent new files that haven't been added to version control yet.

```
Working Directory: file.txt (content: "hello")
Index: (absent)
HEAD: (absent)
Status: Untracked
```

**Added Files (Staged for Addition):**
Files that have been staged but don't exist in the HEAD commit. These will be new files in the next commit.

```
Working Directory: new-file.txt (hash: abc123)
Index: new-file.txt (hash: abc123) 
HEAD: (absent)
Status: Added
```

**Modified Files (Unstaged Changes):**
Files where the working directory content differs from what's staged in the index. The index matches HEAD, but the working directory has newer changes.

```
Working Directory: file.txt (hash: def456)
Index: file.txt (hash: abc123)
HEAD: file.txt (hash: abc123)
Status: Modified (unstaged)
```

**Staged Files (Staged Changes):**
Files where the index content differs from HEAD, indicating changes that are ready to be committed.

```
Working Directory: file.txt (hash: def456)
Index: file.txt (hash: def456)
HEAD: file.txt (hash: abc123)
Status: Modified (staged)
```

**Deleted Files:**
Files that exist in HEAD or the index but are missing from the working directory.

```
Working Directory: (absent)
Index: file.txt (hash: abc123)
HEAD: file.txt (hash: abc123)
Status: Deleted (unstaged)
```

**Both Modified (Conflicted State):**
Files that have different content in all three states, indicating both staged and unstaged changes exist simultaneously.

```
Working Directory: file.txt (hash: ghi789)
Index: file.txt (hash: def456)  
HEAD: file.txt (hash: abc123)
Status: Modified (both staged and unstaged)
```

**Optimization Strategies:**

Status calculation can be expensive in large repositories, so several optimization techniques are crucial:

**Metadata-Based Change Detection:**
Use the cached stat information in index entries to avoid re-reading file content. If a file's mtime and size haven't changed since it was staged, assume its content hasn't changed either.

**Parallel Processing:**
Process different subtrees of the repository concurrently. Since file hashing is CPU-intensive, parallelizing across multiple files can significantly speed up status calculation.

**Incremental Updates:**
Cache status results and only re-examine files whose metadata indicates they might have changed. This is particularly effective for interactive tools that repeatedly call status.

**Ignore File Processing:**
Efficiently skip untracked files that match .gitignore patterns without examining their content. This prevents status calculation from becoming slow in directories with many generated files.

> **Decision: Three-Way vs Pair-Wise Comparison**
> - **Context**: Need to determine file states for status display and as input to merge algorithms
> - **Options Considered**: 
>   - Pair-wise comparisons: Compare working dir vs index, then index vs HEAD separately
>   - Three-way comparison: Examine all three states simultaneously for each file
>   - Lazy evaluation: Only compute states for files the user specifically requests
> - **Decision**: Three-way comparison with optimization shortcuts
> - **Rationale**: Three-way comparison provides complete information needed for merge operations and enables more accurate status reporting (can detect "both modified" states). Pair-wise approaches miss important state combinations and require multiple passes through file sets.
> - **Consequences**: More complex algorithm but provides comprehensive status information needed by advanced Git operations like merge and rebase

**Common Status Calculation Pitfalls:**

⚠️ **Pitfall: Ignoring File System Case Sensitivity**
On case-insensitive file systems (macOS, Windows), files named `File.txt` and `file.txt` are the same file system object but different Git objects. Status calculation must handle this correctly by using the file system's canonical casing for working directory files while preserving the exact casing stored in Git objects.

⚠️ **Pitfall: Race Conditions During Status Calculation**
Files can be modified while status is being calculated, leading to inconsistent results. The algorithm must handle cases where file metadata changes between when the file list is built and when individual files are examined.

⚠️ **Pitfall: Memory Usage with Large Repositories**
Loading all file paths into memory simultaneously can consume excessive memory in repositories with hundreds of thousands of files. Streaming approaches that process files incrementally are necessary for very large repositories.

⚠️ **Pitfall: Inefficient Untracked File Detection**
Scanning the entire working directory tree for untracked files can be extremely slow. The algorithm must respect .gitignore rules early to avoid examining large directories full of generated files (like `node_modules` or build outputs).

![Index File Structure](./diagrams/index-format.svg)

### Implementation Guidance

The index implementation requires careful handling of binary data formats and efficient file system operations. This guidance provides complete infrastructure code for the non-core components and skeleton code for the core learning objectives.

**Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Binary Parsing | `struct` module with format strings | `ctypes` or `numpy` for performance |
| File I/O | Standard `open()` with binary mode | `mmap` for large index files |
| Sorting | Built-in `sorted()` function | Custom sorting with locale awareness |
| Concurrency | File locking with `fcntl` (Unix) | Cross-platform locking library |
| Status Display | Simple print statements | Rich terminal formatting with colors |

**Recommended File Structure:**

```
project-root/
  git/
    __init__.py
    repository.py           ← Repository class
    objects.py              ← Object store (from previous milestones)
    index.py                ← Index implementation (THIS COMPONENT)
      IndexEntry            ← Individual file entry
      Index                 ← Main index operations
      StatusCalculator      ← Three-way comparison logic
    references.py           ← Branch/HEAD management (next milestone)
    commands/
      add.py                ← git add command
      status.py             ← git status command
      commit.py             ← git commit command
    utils/
      binary_parser.py      ← Binary format helpers (infrastructure)
      file_utils.py         ← File system utilities (infrastructure)
  tests/
    test_index.py           ← Index unit tests
    test_status.py          ← Status calculation tests
```

**Infrastructure: Binary Parser Utilities**

This complete utility module handles the low-level binary format parsing, so you can focus on the higher-level index logic:

```python
# utils/binary_parser.py - COMPLETE INFRASTRUCTURE CODE
import struct
import hashlib
from typing import Tuple, List, Optional
from pathlib import Path

class BinaryReader:
    """Helper for reading structured binary data from Git files."""
    
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0
    
    def read_bytes(self, count: int) -> bytes:
        """Read exactly count bytes and advance position."""
        if self.offset + count > len(self.data):
            raise ValueError(f"Not enough data: need {count}, have {len(self.data) - self.offset}")
        result = self.data[self.offset:self.offset + count]
        self.offset += count
        return result
    
    def read_uint32(self) -> int:
        """Read big-endian 32-bit unsigned integer."""
        return struct.unpack(">I", self.read_bytes(4))[0]
    
    def read_uint16(self) -> int:
        """Read big-endian 16-bit unsigned integer."""
        return struct.unpack(">H", self.read_bytes(2))[0]
    
    def read_cstring(self) -> str:
        """Read null-terminated string."""
        start = self.offset
        while self.offset < len(self.data) and self.data[self.offset] != 0:
            self.offset += 1
        if self.offset >= len(self.data):
            raise ValueError("Unterminated string")
        result = self.data[start:self.offset].decode('utf-8')
        self.offset += 1  # Skip null terminator
        return result
    
    def align_to_4_bytes(self):
        """Advance to next 4-byte boundary."""
        while self.offset % 4 != 0:
            self.offset += 1

class BinaryWriter:
    """Helper for writing structured binary data to Git files."""
    
    def __init__(self):
        self.data = bytearray()
    
    def write_bytes(self, data: bytes):
        """Write raw bytes."""
        self.data.extend(data)
    
    def write_uint32(self, value: int):
        """Write big-endian 32-bit unsigned integer."""
        self.data.extend(struct.pack(">I", value))
    
    def write_uint16(self, value: int):
        """Write big-endian 16-bit unsigned integer."""
        self.data.extend(struct.pack(">H", value))
    
    def write_cstring(self, text: str):
        """Write null-terminated UTF-8 string."""
        self.data.extend(text.encode('utf-8'))
        self.data.append(0)
    
    def pad_to_4_bytes(self):
        """Pad with null bytes to next 4-byte boundary."""
        while len(self.data) % 4 != 0:
            self.data.append(0)
    
    def get_data(self) -> bytes:
        """Get the complete binary data."""
        return bytes(self.data)

def compute_sha1(data: bytes) -> str:
    """Compute SHA-1 hash and return as hex string."""
    return hashlib.sha1(data).hexdigest()

def verify_checksum(data: bytes) -> bool:
    """Verify that the last 20 bytes are SHA-1 of the rest."""
    if len(data) < 20:
        return False
    content = data[:-20]
    expected_checksum = data[-20:]
    actual_checksum = hashlib.sha1(content).digest()
    return expected_checksum == actual_checksum
```

**Infrastructure: File System Utilities**

Complete utility functions for safe file operations and metadata handling:

```python
# utils/file_utils.py - COMPLETE INFRASTRUCTURE CODE
import os
import stat
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

def get_file_metadata(file_path: Path) -> Dict[str, Any]:
    """Get complete file metadata for index storage."""
    try:
        st = file_path.stat()
        return {
            'ctime_sec': int(st.st_ctime),
            'ctime_nsec': int((st.st_ctime - int(st.st_ctime)) * 1_000_000_000),
            'mtime_sec': int(st.st_mtime),  
            'mtime_nsec': int((st.st_mtime - int(st.st_mtime)) * 1_000_000_000),
            'device': st.st_dev,
            'inode': st.st_ino,
            'mode': st.st_mode,
            'uid': st.st_uid,
            'gid': st.st_gid,
            'size': st.st_size,
        }
    except OSError:
        return None

def is_file_modified(file_path: Path, cached_mtime_sec: int, cached_size: int) -> bool:
    """Fast check if file might be modified using cached metadata."""
    try:
        st = file_path.stat()
        return (int(st.st_mtime) != cached_mtime_sec or 
                st.st_size != cached_size)
    except OSError:
        return True  # File missing/inaccessible = modified

def atomic_write_file(file_path: Path, content: bytes):
    """Write file atomically using temp file + rename."""
    temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
    try:
        with open(temp_path, 'wb') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())  # Ensure written to disk
        temp_path.rename(file_path)  # Atomic on most filesystems
    except:
        if temp_path.exists():
            temp_path.unlink()
        raise

def scan_directory(dir_path: Path, ignore_patterns: Optional[set] = None) -> List[Path]:
    """Recursively scan directory for all files, respecting ignore patterns."""
    if ignore_patterns is None:
        ignore_patterns = {'.git', '__pycache__', '.pyc'}
    
    files = []
    try:
        for root, dirs, filenames in os.walk(dir_path):
            # Remove ignored directories to prevent os.walk from entering them
            dirs[:] = [d for d in dirs if d not in ignore_patterns]
            
            for filename in filenames:
                if filename not in ignore_patterns:
                    file_path = Path(root) / filename
                    files.append(file_path.relative_to(dir_path))
    except OSError:
        pass  # Directory inaccessible
    
    return sorted(files)
```

**Core Logic: Index Entry Structure**

Here's the skeleton for the core IndexEntry class that you need to implement:

```python
# index.py - CORE LOGIC SKELETON
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Set
from .utils.binary_parser import BinaryReader, BinaryWriter, compute_sha1
from .utils.file_utils import get_file_metadata, atomic_write_file

INDEX_VERSION = 2
FILE_MODE_REGULAR = 0o100644

@dataclass
class IndexEntry:
    """Represents a single file entry in the Git index."""
    # File metadata (cached for fast change detection)
    ctime_sec: int
    ctime_nsec: int
    mtime_sec: int
    mtime_nsec: int
    device: int
    inode: int
    mode: int
    uid: int
    gid: int
    size: int
    
    # Git object information
    object_hash: str  # 40-character hex SHA-1
    flags: int        # Status flags and path length
    path: str         # Relative path from repository root

    @classmethod
    def from_file(cls, file_path: Path, object_hash: str, repo_root: Path) -> 'IndexEntry':
        """Create index entry from working directory file."""
        # TODO 1: Get file metadata using get_file_metadata()
        # TODO 2: Calculate relative path from repo_root
        # TODO 3: Set flags (path length in lower 12 bits, others zero)
        # TODO 4: Return IndexEntry with all fields populated
        # Hint: flags = min(len(relative_path), 0xfff)
        pass
    
    def serialize(self) -> bytes:
        """Serialize entry to binary format for index file."""
        writer = BinaryWriter()
        
        # TODO 1: Write all metadata fields as uint32 (10 fields total)
        # TODO 2: Write SHA-1 hash as 20 binary bytes (convert from hex)
        # TODO 3: Write flags as uint16
        # TODO 4: Write path as null-terminated string
        # TODO 5: Pad to 4-byte boundary
        # Order: ctime_sec, ctime_nsec, mtime_sec, mtime_nsec, device, 
        #        inode, mode, uid, gid, size, hash_bytes, flags, path
        pass
    
    @classmethod
    def deserialize(cls, reader: BinaryReader) -> 'IndexEntry':
        """Deserialize entry from binary format."""
        # TODO 1: Read all metadata fields as uint32 (10 fields)
        # TODO 2: Read SHA-1 hash as 20 binary bytes, convert to hex
        # TODO 3: Read flags as uint16
        # TODO 4: Read path as null-terminated string
        # TODO 5: Align reader to 4-byte boundary
        # TODO 6: Return IndexEntry instance
        pass
```

**Core Logic: Index Class**

The main Index class that manages the complete staging area:

```python
class Index:
    """Git staging area implementation with binary format support."""
    
    def __init__(self, git_dir: Path):
        self.git_dir = git_dir
        self.index_path = git_dir / 'index'
        self.entries: List[IndexEntry] = []
    
    def load(self):
        """Load index from .git/index file."""
        if not self.index_path.exists():
            self.entries = []
            return
        
        with open(self.index_path, 'rb') as f:
            data = f.read()
        
        # TODO 1: Verify checksum using verify_checksum()
        # TODO 2: Create BinaryReader with content (excluding checksum)
        # TODO 3: Read and verify header (signature "DIRC", version 2)
        # TODO 4: Read entry count
        # TODO 5: Read each entry using IndexEntry.deserialize()
        # TODO 6: Store entries in self.entries list
        # Hint: Content for checksum is data[:-20], checksum is data[-20:]
        pass
    
    def save(self):
        """Save index to .git/index file with atomic write."""
        writer = BinaryWriter()
        
        # TODO 1: Write header (signature, version, entry count)
        # TODO 2: Sort entries by path name
        # TODO 3: Write each entry using entry.serialize()
        # TODO 4: Calculate SHA-1 checksum of all content
        # TODO 5: Append checksum as 20 binary bytes
        # TODO 6: Write atomically using atomic_write_file()
        pass
    
    def add_file(self, file_path: Path, repo_root: Path, object_store):
        """Stage a file by adding it to the index."""
        # TODO 1: Read file content from working directory
        # TODO 2: Store file content as blob object using object_store.store_object()
        # TODO 3: Create IndexEntry from file and object hash
        # TODO 4: Remove any existing entry with same path
        # TODO 5: Insert new entry maintaining sort order
        # Hint: Use bisect module for efficient sorted insertion
        pass
    
    def remove_file(self, file_path: str):
        """Remove file from index (unstage)."""
        # TODO 1: Find entry with matching path
        # TODO 2: Remove entry from self.entries list
        # TODO 3: Don't modify object store (objects are immutable)
        pass
    
    def get_entry(self, file_path: str) -> Optional[IndexEntry]:
        """Get index entry for specified path."""
        # TODO 1: Search through entries for matching path
        # TODO 2: Return entry if found, None otherwise
        # Hint: Since entries are sorted, can use binary search
        pass
    
    def get_all_paths(self) -> Set[str]:
        """Get set of all paths in the index."""
        return {entry.path for entry in self.entries}
```

**Core Logic: Status Calculator**

The three-way comparison engine:

```python
class StatusCalculator:
    """Calculates file status using three-way comparison."""
    
    def __init__(self, repo_root: Path, git_dir: Path):
        self.repo_root = repo_root
        self.git_dir = git_dir
    
    def calculate_status(self) -> Dict[str, str]:
        """Perform three-way comparison and return file statuses."""
        # TODO 1: Load index and get all staged file paths
        # TODO 2: Get HEAD commit and extract all tracked file paths/hashes
        # TODO 3: Scan working directory for all present files
        # TODO 4: Create union of all file paths from three sources
        # TODO 5: For each path, determine state in all three locations
        # TODO 6: Apply classification rules to determine status
        # TODO 7: Return dict mapping file paths to status strings
        pass
    
    def _get_head_files(self) -> Dict[str, str]:
        """Get files and hashes from HEAD commit."""
        # TODO 1: Read HEAD reference to get current commit hash
        # TODO 2: Load commit object and get root tree hash
        # TODO 3: Recursively walk tree objects to get all file paths/hashes
        # TODO 4: Return dict mapping paths to object hashes
        pass
    
    def _get_working_files(self) -> Dict[str, str]:
        """Get files and hashes from working directory."""
        # TODO 1: Scan working directory for all files
        # TODO 2: For each file, compute what its blob hash would be
        # TODO 3: Skip ignored files (.gitignore rules)
        # TODO 4: Return dict mapping paths to computed hashes
        pass
    
    def _classify_file_status(self, path: str, wd_hash: Optional[str], 
                             index_hash: Optional[str], head_hash: Optional[str]) -> str:
        """Classify single file status based on three-way comparison."""
        # TODO 1: Handle untracked case (wd present, index and head absent)
        # TODO 2: Handle added case (wd and index present, head absent)
        # TODO 3: Handle modified cases (various combinations)
        # TODO 4: Handle deleted cases (wd absent, index/head present)
        # TODO 5: Handle staged cases (index differs from head)
        # TODO 6: Return appropriate status string
        pass
```

**Milestone Checkpoint:**

After implementing the index system, verify your implementation with these tests:

1. **Basic Add Operation**: 
   ```bash
   echo "hello world" > test.txt
   python -m git.commands.add test.txt
   # Should create .git/index file and store blob object
   ```

2. **Index File Format**:
   ```bash
   hexdump -C .git/index | head -5
   # Should show "DIRC" signature, version 2, entry count
   ```

3. **Status Calculation**:
   ```bash  
   python -m git.commands.status
   # Should show test.txt as "new file" (added)
   ```

4. **Index Persistence**:
   ```bash
   # Run status twice - should be fast second time (metadata caching)
   time python -m git.commands.status
   time python -m git.commands.status
   ```

**Debugging Tips:**

| Symptom | Likely Cause | Diagnosis | Fix |
|---------|--------------|-----------|-----|
| Index file corrupted | Wrong binary format | `hexdump -C .git/index` | Check field sizes and byte order |
| Status shows all files modified | Metadata caching broken | Compare cached vs actual mtime/size | Fix stat field extraction |  
| Add operation fails | Path encoding issues | Check for non-ASCII characters | Use UTF-8 encoding consistently |
| Performance poor on large repos | Not using metadata optimization | Profile status calculation | Implement fast-path metadata checks |


## References and Branch Management

> **Milestone(s):** This section is essential for Milestone 5 (References and Branches) and provides the foundation for branch-based operations in Milestone 7 (Diff Algorithm) and Milestone 8 (Merge)

The reference system forms the human-readable layer of Git's architecture, sitting above the cryptographic hash-based object store to provide meaningful names for commits and manage the current project state. While Git's underlying storage operates entirely through SHA-1 hashes, users need intuitive names like "main", "feature-branch", or "v1.2.3" to navigate their project history effectively. The reference system bridges this gap by maintaining a mapping from human-readable names to commit hashes, while also tracking the current branch state through the special HEAD reference.

### Mental Model: Bookmarks in History

Think of Git references as **bookmarks in a vast historical library**. Imagine you're researching in a library where every book represents a commit, and books are shelved by their unique catalog number (the SHA-1 hash). Without bookmarks, you'd need to remember cryptic 40-character numbers like "a1b2c3d4e5f6..." to find the books you care about.

Git references are like labeled bookmarks you can place anywhere in this library:
- **Branch references** are moveable bookmarks that automatically advance to new books as you add them to a series
- **Tag references** are permanent bookmarks that never move, marking important milestones
- **HEAD** is your special "you are here" bookmark that shows which book you're currently reading

When you create a new branch called "feature-login", you're essentially placing a bookmark labeled "feature-login" at the current book (commit). As you write new chapters (make new commits), that bookmark automatically moves forward to always point to your latest work. Meanwhile, other bookmarks like "main" stay put unless you explicitly move them.

The HEAD reference is particularly special—it's like having a "current location" bookmark that can either:
- Point to one of your labeled bookmarks (when you're "on a branch")
- Point directly to a specific book (when you have a "detached HEAD")

This system allows you to navigate through your project's history using memorable names instead of memorizing long hash strings, while Git maintains the cryptographic integrity of the underlying object store.

### Symbolic vs Direct References

Git's reference system supports two fundamentally different types of references that serve distinct purposes in branch management and navigation. Understanding this distinction is crucial for implementing proper branch switching and detached HEAD state handling.

**Direct references** store a raw commit SHA-1 hash and point directly to a commit object in the object store. These references contain exactly 40 hexadecimal characters followed by a newline, with no additional formatting or indirection. When Git needs to resolve a direct reference, it simply reads the hash value and uses it to look up the commit object.

**Symbolic references** store a reference to another reference, creating a level of indirection in the resolution process. The most important symbolic reference is HEAD, which typically contains content like `ref: refs/heads/main\n` instead of a raw hash. When Git resolves a symbolic reference, it first reads the reference name, then follows that reference to find the actual commit hash.

The HEAD reference demonstrates why this distinction matters for branch management. When you're working on a branch, HEAD contains a symbolic reference pointing to the current branch (e.g., `ref: refs/heads/feature-branch`). This means that when you make a new commit, Git updates the branch reference file, and HEAD automatically points to the new commit through the symbolic link. This is how Git knows to advance the current branch when you commit.

However, when you check out a specific commit hash (creating a detached HEAD state), HEAD becomes a direct reference containing the raw commit SHA-1. In this state, making new commits doesn't update any branch—the commits exist but aren't reachable through any branch name.

| Reference Type | Content Format | Resolution Process | Use Case | Example Content |
|----------------|----------------|-------------------|----------|-----------------|
| Direct | Raw SHA-1 hash + newline | Single lookup in object store | Branch tips, tags, detached HEAD | `a1b2c3d4e5f6789...` |
| Symbolic | `ref: ` + reference path + newline | Two-step: resolve target ref, then lookup | HEAD pointing to current branch | `ref: refs/heads/main` |

> **Design Principle**: The symbolic reference mechanism enables Git's branch semantics. Without symbolic references, there would be no concept of "being on a branch"—you would always be in a detached state, and commits wouldn't automatically advance any branch pointer.

**Reference Resolution Algorithm**

The process of resolving any reference to its final commit hash follows these steps:

1. **Read the reference file** from the appropriate location (`.git/HEAD`, `.git/refs/heads/branchname`, etc.)
2. **Check the content format** by examining the first few bytes for the `ref: ` prefix
3. **For symbolic references**: Extract the target reference path, then recursively resolve that reference
4. **For direct references**: Validate the SHA-1 format (40 hex characters) and return the hash
5. **Verify the target commit exists** in the object store before considering resolution successful

This resolution process can chain multiple symbolic references, though in practice Git repositories rarely have chains longer than two levels (HEAD → branch → commit).

> **Decision: Single-Level Symbolic Reference Implementation**
> - **Context**: While Git supports arbitrary symbolic reference chains, most real-world usage involves only HEAD pointing to branches
> - **Options Considered**: Full recursive resolution vs single-level resolution vs direct references only
> - **Decision**: Implement single-level symbolic reference resolution (HEAD → branch → commit)
> - **Rationale**: Covers 99% of real Git usage while keeping implementation simple and reducing infinite loop risk
> - **Consequences**: Enables proper branch semantics and detached HEAD handling without complex recursion logic

### Branch Creation and Switching

Branch management operations form the core of Git's workflow, allowing developers to create parallel lines of development and switch between them safely. The implementation must handle three primary operations: creating new branches, switching between existing branches, and managing the transition between attached and detached HEAD states.

**Branch Creation Process**

Creating a new branch involves establishing a new reference that points to a specific commit, typically the current HEAD commit. The process requires careful coordination between the reference system and the working directory to maintain repository consistency.

The branch creation algorithm proceeds through these steps:

1. **Resolve the target commit hash** from the specified starting point (HEAD by default, or a specific commit/branch name)
2. **Validate the target commit exists** in the object store to prevent creating branches that point to non-existent commits
3. **Check for branch name conflicts** by verifying no file exists at `.git/refs/heads/branch-name`
4. **Create the branch reference file** at `.git/refs/heads/branch-name` containing the target commit hash
5. **Optionally switch to the new branch** by updating HEAD to point to the new branch reference

Branch names must follow Git's reference naming rules to avoid conflicts with the file system and Git's internal operations. Valid branch names cannot contain spaces, control characters, or special sequences like `..` that could interfere with reference resolution.

| Branch Creation Input | Target Commit Resolution | Result | Example |
|----------------------|--------------------------|--------|---------|
| `git branch feature` | Current HEAD commit | New branch at HEAD | `refs/heads/feature` → `abc123...` |
| `git branch hotfix main` | Tip of main branch | New branch at main's tip | `refs/heads/hotfix` → `def456...` |
| `git branch release abc123` | Specific commit hash | New branch at that commit | `refs/heads/release` → `abc123...` |

**Branch Switching Algorithm**

Switching branches requires updating both HEAD and the working directory to reflect the target branch's state. This operation is one of the most complex in Git because it must handle file system changes while maintaining data safety.

The branch switching process involves several critical steps:

1. **Resolve the target branch reference** to get the target commit hash, handling both local branches and commit hashes
2. **Check working directory status** to detect uncommitted changes that might be lost during the switch
3. **Load the target commit's tree object** to determine what files should exist in the working directory
4. **Calculate working directory changes** by comparing current files with the target tree structure
5. **Apply file system changes** by creating, modifying, or deleting files to match the target tree
6. **Update HEAD reference** to point to the target branch (symbolic) or commit (direct)
7. **Update the index** to reflect the new working directory state if necessary

The most critical aspect of branch switching is handling uncommitted changes safely. Git uses a three-way comparison between the current HEAD, target commit, and working directory to determine if changes can be preserved or if they would be lost.

**Working Directory Update Strategy**

When switching branches, Git must transform the working directory from one tree state to another while preserving user work where possible. This involves analyzing each file in both trees and applying the appropriate changes.

| Current File State | Target File State | Action Required | Conflict Potential |
|-------------------|-------------------|-----------------|-------------------|
| Exists, clean | Exists, different content | Overwrite with target | None |
| Exists, modified | Exists, different content | Check for conflicts | High - may lose changes |
| Exists, clean | Does not exist | Delete file | None |
| Exists, modified | Does not exist | Refuse switch or force delete | High - would lose changes |
| Does not exist | Exists | Create with target content | Low - check for untracked conflicts |

> **Safety Principle**: Git should never silently lose user data. Branch switching must either preserve all changes or explicitly warn the user about potential data loss and require confirmation.

**Detached HEAD State Management**

The detached HEAD state occurs when HEAD points directly to a commit hash rather than to a branch reference. This state is essential for examining historical commits, building releases from specific points, or creating experimental commits that aren't part of any branch.

Entering detached HEAD state happens in several scenarios:

1. **Explicit commit checkout**: `git checkout abc123` where `abc123` is a commit hash
2. **Tag checkout**: `git checkout v1.2.3` where the tag points to a commit rather than a branch
3. **Historical navigation**: `git checkout HEAD~3` to examine a previous commit

The implementation must handle the transition between attached and detached states carefully, updating HEAD's content format and providing clear user feedback about the state change.

**Detached HEAD State Transitions**

![Reference State Machine](./diagrams/reference-states.svg)

The state machine shows the valid transitions between different HEAD states and the operations that trigger them. Each state has different implications for commit behavior and branch advancement.

| Current State | Operation | Next State | HEAD Content | Branch Update Behavior |
|---------------|-----------|------------|--------------|----------------------|
| On branch `main` | `checkout feature-branch` | On branch `feature-branch` | `ref: refs/heads/feature-branch` | Commits advance feature-branch |
| On branch `main` | `checkout abc123` | Detached HEAD | `abc123...` | Commits create orphan history |
| Detached HEAD | `checkout main` | On branch `main` | `ref: refs/heads/main` | Commits advance main branch |
| Detached HEAD | Create new branch from HEAD | On new branch | `ref: refs/heads/new-branch` | Commits advance new-branch |

**Reference File Management**

The reference system stores all branch and tag information as plain text files within the `.git/refs/` directory tree. This simple approach makes Git repositories inspectable and debuggable with standard file system tools while providing atomic updates through file system operations.

The directory structure organizes references by type and namespace:

```
.git/
  HEAD                    ← current branch pointer (symbolic ref)
  refs/
    heads/               ← local branches
      main               ← contains commit hash for main branch
      feature-login      ← contains commit hash for feature-login branch
      hotfix/security    ← branches can have path separators
    remotes/             ← remote-tracking branches (future extension)
      origin/main        ← tracks remote main branch
    tags/                ← tag references
      v1.0.0            ← contains commit hash for v1.0.0 tag
```

Each reference file contains a single line of text: either a raw SHA-1 hash (for direct references) or a symbolic reference in the format `ref: refs/heads/branch-name`. The newline character at the end is significant and must be preserved for compatibility with Git's reference parsing.

**Atomic Reference Updates**

Reference updates must be atomic to prevent repository corruption during concurrent operations or system failures. The implementation uses the standard technique of writing to a temporary file and then atomically renaming it to the target location.

The atomic update process follows these steps:

1. **Generate a unique temporary filename** in the same directory as the target reference file
2. **Write the new reference content** to the temporary file, including the required newline character
3. **Sync the temporary file** to ensure the data is written to stable storage
4. **Atomically rename** the temporary file to the final reference filename
5. **Clean up** any temporary files on failure to avoid leaving debris in the repository

This approach ensures that reference files are never in an inconsistent state—they either contain the old value or the new value, never partial writes or corrupted data.

> **Decision: Plain Text Reference Files**
> - **Context**: Git needs to store mappings from branch names to commit hashes with atomic updates and human readability
> - **Options Considered**: Plain text files vs binary packed format vs embedded database
> - **Decision**: Use plain text files in .git/refs/ directory structure
> - **Rationale**: Simple implementation, atomic updates via file system, human-readable for debugging, compatible with Git tooling
> - **Consequences**: Slightly slower than packed refs for repositories with thousands of branches, but much simpler and more reliable

**Branch Name Validation**

Branch names must follow specific rules to avoid conflicts with Git's internal operations and file system limitations. The validation process ensures that branch names can be safely used as file names and don't interfere with Git's reference resolution algorithms.

Invalid branch name patterns include:

- **Control characters**: ASCII control characters (0x00-0x1f) that might interfere with text processing
- **Path separators**: While `/` is allowed within branch names, `.` and `..` are prohibited to prevent directory traversal
- **Special sequences**: Patterns like `@{` that Git uses for reference specifications
- **Reserved names**: Names that conflict with Git's internal references or common conventions

| Pattern | Valid? | Reason | Example |
|---------|--------|--------|---------|
| `feature-login` | ✓ | Alphanumeric with hyphens | Standard branch name |
| `hotfix/security` | ✓ | Slash creates namespace | Organized branch structure |
| `release-1.0` | ✓ | Alphanumeric with dash and dot | Version branch naming |
| `.hidden` | ✗ | Leading dot reserved | Could conflict with Git internals |
| `branch..name` | ✗ | Double dot reserved | Conflicts with revision syntax |
| `branch name` | ✗ | Spaces cause parsing issues | Would break command line tools |

**Common Pitfalls**

⚠️ **Pitfall: Forgetting to Update Working Directory During Branch Switch**

Many implementations correctly update HEAD but forget to modify the working directory files to match the target branch's tree. This leaves the repository in an inconsistent state where HEAD points to one commit but the working directory contains files from another commit.

The symptoms include: files appearing modified immediately after a branch switch, confusion about which branch's changes are visible, and potential data loss when users assume they're working on a different branch than their files represent.

**Fix**: Always implement the complete branch switching algorithm that updates both HEAD and working directory atomically, or fail the entire operation if working directory updates cannot be completed safely.

⚠️ **Pitfall: Creating Branches Without Validating Target Commits**

Creating a branch reference that points to a non-existent commit hash creates a broken branch that cannot be checked out or used for any operations. This commonly happens when implementing branch creation from user-specified commit hashes without verifying the commits exist.

The symptoms include: branches that appear in branch listings but cannot be checked out, error messages about missing objects when trying to use the branch, and confusion about whether commits were lost or never existed.

**Fix**: Always validate that the target commit exists in the object store before creating any reference that points to it. Use the `object_exists()` function to verify commit availability.

⚠️ **Pitfall: Race Conditions in Reference Updates**

Writing reference files without atomic updates can lead to corruption when multiple Git processes run simultaneously or when system failures occur during writes. This creates partially-written reference files containing truncated hashes or mixed content.

The symptoms include: references containing partial SHA-1 hashes, "not a valid object name" errors when resolving references, and repository corruption that requires manual repair.

**Fix**: Always use atomic file updates with temporary files and rename operations. Never write directly to reference files in-place.

⚠️ **Pitfall: Incorrect HEAD Format for Symbolic References**

The HEAD file format for symbolic references is strict: `ref: refs/heads/branch-name\n` with exactly one space after the colon, no extra whitespace, and a single trailing newline. Deviating from this format breaks reference resolution.

Common mistakes include: missing the space after the colon (`ref:refs/heads/main`), adding extra spaces or tabs, omitting the trailing newline, or using Windows line endings (`\r\n`) instead of Unix line endings.

**Fix**: Use string constants for the symbolic reference format and validate the exact format when reading symbolic references. Always use binary file writing to control line endings precisely.

### Implementation Guidance

The reference management system requires careful attention to file system operations and atomic updates. This implementation provides a robust foundation for branch operations while maintaining compatibility with Git's reference format.

**Technology Recommendations**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| File Operations | `pathlib.Path` with text read/write | `pathlib.Path` with atomic writes and file locking |
| Reference Resolution | Recursive function with depth limit | State machine with cycle detection |
| Branch Validation | Regular expression patterns | Full Git reference name validation |
| Concurrent Access | File system atomic operations | Advisory file locking with timeouts |

**Recommended File Structure**

```python
project-root/
  git_implementation/
    core/
      repository.py          ← Repository class with reference methods
      reference_manager.py   ← ReferenceManager implementation
      object_store.py       ← ObjectStore for commit validation
    utils/
      file_operations.py    ← Atomic file write utilities
      validation.py         ← Branch name validation functions
    tests/
      test_references.py    ← Reference system tests
      test_branches.py      ← Branch operation tests
```

**Infrastructure Starter Code**

Here's a complete atomic file operations utility that handles the low-level file system operations needed for safe reference updates:

```python
"""
Atomic file operations for safe repository updates.
Provides utilities for writing files atomically to prevent corruption.
"""
import os
import tempfile
from pathlib import Path
from typing import Union

def atomic_write_file(file_path: Path, content: Union[str, bytes]) -> None:
    """
    Write content to a file atomically using temporary file and rename.
    
    Args:
        file_path: Target file path to write
        content: Content to write (str or bytes)
    
    Raises:
        OSError: If file operations fail
        UnicodeEncodeError: If string content cannot be encoded as UTF-8
    """
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temporary file in same directory for atomic rename
    temp_fd, temp_path = tempfile.mkstemp(
        dir=file_path.parent,
        prefix=f".tmp_{file_path.name}_",
        suffix=".tmp"
    )
    
    try:
        with os.fdopen(temp_fd, 'wb') as temp_file:
            if isinstance(content, str):
                temp_file.write(content.encode('utf-8'))
            else:
                temp_file.write(content)
            
            # Force write to disk before rename
            temp_file.flush()
            os.fsync(temp_file.fileno())
        
        # Atomic rename to final location
        os.rename(temp_path, file_path)
        
    except Exception:
        # Clean up temp file on any error
        try:
            os.unlink(temp_path)
        except OSError:
            pass  # Temp file already gone
        raise

def read_reference_file(ref_path: Path) -> str:
    """
    Read a Git reference file and return its content.
    
    Args:
        ref_path: Path to reference file
        
    Returns:
        Reference content with trailing whitespace stripped
        
    Raises:
        FileNotFoundError: If reference file doesn't exist
        UnicodeDecodeError: If file contains invalid UTF-8
    """
    try:
        with open(ref_path, 'r', encoding='utf-8') as f:
            return f.read().rstrip('\n\r')
    except FileNotFoundError:
        raise FileNotFoundError(f"Reference not found: {ref_path}")

def ensure_git_directory_structure(git_dir: Path) -> None:
    """
    Create the basic .git directory structure needed for references.
    
    Args:
        git_dir: Path to .git directory
    """
    directories = [
        git_dir / "refs",
        git_dir / "refs" / "heads",
        git_dir / "refs" / "tags",
        git_dir / "refs" / "remotes"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
```

**Branch Name Validation Utilities**

```python
"""
Git reference name validation following Git's naming rules.
"""
import re
from typing import List

# Git reference name validation patterns
INVALID_REF_CHARS = re.compile(r'[\x00-\x1f\x7f~^:\\\*\?]')
INVALID_REF_SEQUENCES = ['.', '..', '@{', '//']
RESERVED_REF_NAMES = {'HEAD', 'ORIG_HEAD', 'FETCH_HEAD', 'MERGE_HEAD'}

def validate_branch_name(name: str) -> List[str]:
    """
    Validate a branch name against Git's reference naming rules.
    
    Args:
        name: Proposed branch name
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if not name:
        errors.append("Branch name cannot be empty")
        return errors
    
    if name in RESERVED_REF_NAMES:
        errors.append(f"'{name}' is a reserved reference name")
    
    if name.startswith('.') or name.endswith('.'):
        errors.append("Branch name cannot start or end with '.'")
    
    if name.startswith('/') or name.endswith('/'):
        errors.append("Branch name cannot start or end with '/'")
    
    if INVALID_REF_CHARS.search(name):
        errors.append("Branch name contains invalid characters")
    
    for sequence in INVALID_REF_SEQUENCES:
        if sequence in name:
            errors.append(f"Branch name cannot contain '{sequence}'")
    
    return errors

def is_valid_branch_name(name: str) -> bool:
    """Check if a branch name is valid."""
    return len(validate_branch_name(name)) == 0
```

**Core Logic Skeleton Code**

Here's the main `ReferenceManager` class structure with detailed TODO comments for implementation:

```python
"""
Git reference management system handling branches, HEAD, and symbolic references.
"""
from pathlib import Path
from typing import Optional, Dict, Set
import hashlib

class ReferenceManager:
    """Manages Git references including branches, HEAD, and symbolic references."""
    
    def __init__(self, git_dir: Path):
        self.git_dir = git_dir
        self.refs_dir = git_dir / "refs"
        self.heads_dir = self.refs_dir / "heads"
        self.tags_dir = self.refs_dir / "tags"
        self.head_file = git_dir / "HEAD"
    
    def resolve_reference(self, ref_name: str) -> Optional[str]:
        """
        Resolve a reference name to its target commit hash.
        
        Handles both direct references (containing commit hashes) and
        symbolic references (containing 'ref: path/to/other/ref').
        
        Args:
            ref_name: Reference to resolve (e.g., 'HEAD', 'main', 'refs/heads/feature')
            
        Returns:
            Commit SHA-1 hash if reference exists and resolves, None otherwise
        """
        # TODO 1: Handle special case of 'HEAD' - read from self.head_file
        # TODO 2: If ref_name doesn't start with 'refs/', try 'refs/heads/' + ref_name
        # TODO 3: Construct full path to reference file
        # TODO 4: Check if reference file exists, return None if not
        # TODO 5: Read reference file content using read_reference_file()
        # TODO 6: Check if content starts with 'ref: ' (symbolic reference)
        # TODO 7: For symbolic refs: extract target path and recursively resolve
        # TODO 8: For direct refs: validate SHA-1 format and return hash
        # TODO 9: Add recursion depth limit to prevent infinite loops
        # Hint: Use validate_sha1_hash() to check hash format
        pass
    
    def create_branch(self, branch_name: str, target_commit: str) -> bool:
        """
        Create a new branch pointing to the specified commit.
        
        Args:
            branch_name: Name for the new branch
            target_commit: SHA-1 hash of commit to point to
            
        Returns:
            True if branch was created successfully, False otherwise
        """
        # TODO 1: Validate branch name using validate_branch_name()
        # TODO 2: Verify target_commit exists in object store using object_exists()
        # TODO 3: Check if branch already exists (refs/heads/branch_name file exists)
        # TODO 4: Create branch reference file with atomic_write_file()
        # TODO 5: Write target_commit hash + '\n' to the branch file
        # TODO 6: Return success/failure status
        # Hint: Branch file path is self.heads_dir / branch_name
        pass
    
    def switch_branch(self, branch_name: str) -> bool:
        """
        Switch to an existing branch by updating HEAD.
        
        This is a simplified version that only updates references without
        modifying the working directory or index.
        
        Args:
            branch_name: Name of branch to switch to
            
        Returns:
            True if switch was successful, False otherwise
        """
        # TODO 1: Verify target branch exists using branch_exists()
        # TODO 2: Create symbolic reference content: f"ref: refs/heads/{branch_name}\n"
        # TODO 3: Write symbolic reference to HEAD file using atomic_write_file()
        # TODO 4: Return success status
        # Note: Full implementation would also update working directory
        pass
    
    def checkout_commit(self, commit_hash: str) -> bool:
        """
        Enter detached HEAD state by pointing HEAD directly to a commit.
        
        Args:
            commit_hash: SHA-1 hash of commit to check out
            
        Returns:
            True if checkout was successful, False otherwise
        """
        # TODO 1: Validate commit_hash format using validate_sha1_hash()
        # TODO 2: Verify commit exists in object store using object_exists()
        # TODO 3: Write commit hash + '\n' directly to HEAD file (direct reference)
        # TODO 4: Return success status
        # Note: This creates detached HEAD state
        pass
    
    def get_current_branch(self) -> Optional[str]:
        """
        Get the name of the current branch, if HEAD points to a branch.
        
        Returns:
            Branch name if on a branch, None if in detached HEAD state
        """
        # TODO 1: Read HEAD file content
        # TODO 2: Check if content starts with 'ref: refs/heads/'
        # TODO 3: Extract branch name from symbolic reference
        # TODO 4: Return branch name or None for detached HEAD
        pass
    
    def list_branches(self) -> Set[str]:
        """
        List all local branches in the repository.
        
        Returns:
            Set of branch names
        """
        # TODO 1: Check if refs/heads directory exists
        # TODO 2: Iterate through all files in refs/heads directory
        # TODO 3: Use glob or iterdir to find all branch files
        # TODO 4: Extract branch names (relative paths from refs/heads/)
        # TODO 5: Return set of branch names
        # Hint: Use Path.iterdir() and handle subdirectories for namespaced branches
        pass
    
    def delete_branch(self, branch_name: str, force: bool = False) -> bool:
        """
        Delete a branch reference.
        
        Args:
            branch_name: Name of branch to delete
            force: If True, delete even if it's the current branch
            
        Returns:
            True if deletion was successful, False otherwise
        """
        # TODO 1: Check if branch exists
        # TODO 2: If not force, verify branch is not current branch
        # TODO 3: Remove branch reference file using Path.unlink()
        # TODO 4: Handle FileNotFoundError gracefully
        # TODO 5: Return success status
        pass
    
    def branch_exists(self, branch_name: str) -> bool:
        """Check if a branch exists."""
        # TODO: Check if refs/heads/branch_name file exists
        pass
    
    def is_detached_head(self) -> bool:
        """Check if HEAD is in detached state (points to commit, not branch)."""
        # TODO 1: Read HEAD file content
        # TODO 2: Return True if content is a SHA-1 hash, False if symbolic ref
        pass

def validate_sha1_hash(hash_str: str) -> bool:
    """Validate that a string is a valid SHA-1 hash."""
    if len(hash_str) != 40:
        return False
    try:
        int(hash_str, 16)
        return True
    except ValueError:
        return False
```

**Language-Specific Hints**

- Use `pathlib.Path` for all file system operations - it handles path joining and OS differences automatically
- Python's `tempfile.mkstemp()` creates temporary files securely with proper permissions
- Always use `'utf-8'` encoding when reading/writing text files to ensure compatibility
- Use `os.fsync()` after writing critical files like references to ensure data reaches disk
- The `rstrip('\n\r')` method handles both Unix and Windows line endings when reading files
- Use `Path.mkdir(parents=True, exist_ok=True)` to create directory trees safely

**Milestone Checkpoint**

After implementing the reference management system, verify these behaviors:

1. **Branch Creation**: Create a branch with `create_branch('feature', 'commit_hash')` and verify the file `refs/heads/feature` contains the commit hash
2. **Branch Listing**: Call `list_branches()` and verify it returns all branch names from the file system
3. **HEAD Management**: Switch branches and verify HEAD content changes between symbolic and direct references
4. **Reference Resolution**: Test resolving 'HEAD', branch names, and full reference paths
5. **Detached HEAD**: Checkout a commit hash and verify `is_detached_head()` returns True

Expected file system state after creating branch 'feature':
```
.git/
  HEAD                     ← contains "ref: refs/heads/main\n"
  refs/heads/
    main                   ← contains commit hash
    feature                ← contains same or different commit hash
```

**Debugging Tips**

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Branch appears in listing but cannot be resolved | Invalid commit hash in branch file | Check file contents with `cat .git/refs/heads/branch` | Recreate branch with valid commit hash |
| HEAD resolution fails after branch switch | Incorrect symbolic reference format | Check HEAD file format: should be `ref: refs/heads/branch\n` | Rewrite HEAD with correct format |
| Branch creation succeeds but branch disappears | Missing newline in reference file | Check file ends with `\n` character | Always append `\n` to reference content |
| Permission denied when updating references | Incorrect file permissions on .git directory | Check `.git` directory is writable | Set proper permissions with `chmod` |
| Reference resolution enters infinite loop | Circular symbolic reference | Trace reference chain manually | Add recursion depth limit to resolution |


## Diff Algorithm Implementation

> **Milestone(s):** This section is critical for Milestone 7 (Diff Algorithm) and provides the foundation for conflict detection in Milestone 8 (Three-Way Merge). The diff algorithm serves as the core comparison engine for status calculation in Milestone 6 (Index/Staging Area).

The diff algorithm forms the analytical heart of any version control system. While Git's object store provides immutable content storage and the index manages staging, the diff algorithm answers the fundamental question: "What changed between two versions of a file or project?" This capability underpins virtually every Git operation that involves comparison - from `git status` showing modified files to `git merge` detecting conflicts between branches.

The challenge of implementing diff extends beyond simple file comparison. A naive approach of comparing files byte-by-byte would only tell us whether files are identical or different, but not what specifically changed. Users need to see exactly which lines were added, removed, or modified, preferably in a format that highlights the minimal set of changes required to transform one version into another. This is where sophisticated diff algorithms like Myers' algorithm excel, finding the shortest edit script between two sequences while producing human-readable output.

Our diff implementation must handle several complex scenarios: comparing files with vastly different line counts, detecting when entire sections have moved, handling binary files gracefully, and producing output that follows industry-standard unified diff format. The algorithm must also be efficient enough to handle large files without consuming excessive memory or computation time.

### Mental Model: Document Comparison

Think of the diff algorithm as an expert editor comparing two drafts of the same manuscript. When an author submits a revised chapter, the editor doesn't simply declare "these are different" - instead, they meticulously identify each change: "In paragraph 3, you added a new sentence about character motivation. In paragraph 7, you removed the description of the sunset. In paragraph 12, you changed 'walked' to 'strode'."

This editor has developed a systematic approach over years of experience. They don't just scan randomly - they find the longest unchanged passages first (these serve as anchor points), then focus their attention on the gaps between these passages where changes must have occurred. When they find a section that appears in both drafts but in different locations, they recognize it as moved content rather than a deletion followed by an addition.

The editor also knows that their goal isn't just accuracy, but usefulness. They present changes in a format that makes it easy for the author to understand what happened: showing a few lines of unchanged context around each change, grouping nearby changes together, and using clear markers to indicate additions and deletions. This is exactly how Myers' diff algorithm works - it finds the minimal set of changes between two text sequences and presents them in a format optimized for human comprehension.

The key insight is that effective diff algorithms don't just identify differences - they find the **optimal** set of differences that minimizes cognitive load for the person reviewing the changes. This requires sophisticated algorithms that can distinguish between genuine changes and coincidental similarities, while producing output that tells a coherent story about how one version evolved into another.

### Myers Diff Algorithm

The Myers diff algorithm, developed by Eugene Myers in 1986, represents the gold standard for computing differences between two sequences. Unlike simpler approaches that might miss optimal solutions, Myers' algorithm is guaranteed to find the shortest edit script - the minimal sequence of insertions and deletions needed to transform one file into another. This optimality is crucial for producing readable diffs that don't overwhelm users with unnecessary noise.

The algorithm operates on a fundamental insight about the structure of edit problems. When comparing two sequences A and B, we can visualize the problem as finding a path through a two-dimensional grid where the x-axis represents positions in sequence A and the y-axis represents positions in sequence B. Each cell (i,j) in this grid represents the state where we've processed i elements from sequence A and j elements from sequence B.

From any cell in this grid, we have three possible moves: move right (delete an element from A), move down (insert an element from B), or move diagonally (elements match, no edit required). Our goal is to find the path from the top-left corner (0,0) to the bottom-right corner (len(A), len(B)) that requires the fewest non-diagonal moves. Each non-diagonal move corresponds to an edit operation, so minimizing these moves gives us the shortest edit script.

The brilliance of Myers' algorithm lies in how it explores this space efficiently. Rather than examining every possible path (which would be exponential), it uses a technique called "edit distance computation with greedy matching." The algorithm processes the comparison in phases, where each phase considers all possible paths that require exactly k edit operations. Within each phase, it greedily extends paths as far as possible using diagonal moves (matching elements), which cost nothing.

Here's how the algorithm proceeds step by step:

1. **Initialize the exploration**: Create a data structure to track the furthest position reachable along each diagonal for a given number of edits. A diagonal d represents positions where (x - y = d). We track V[d] = x, the furthest x-coordinate reached on diagonal d.

2. **Iterative deepening by edit distance**: For each possible edit distance k from 0 to the maximum possible (len(A) + len(B)), explore all paths that require exactly k edits. This ensures we find the optimal solution as soon as one exists.

3. **Diagonal exploration**: For each diagonal d that could be reached with k edits, calculate the furthest point reachable. We can reach diagonal d either by moving right from diagonal d-1 (deletion) or by moving down from diagonal d+1 (insertion). Choose the option that gets us furthest along diagonal d.

4. **Greedy extension**: Once we've determined our starting position on diagonal d, extend as far as possible using diagonal moves (matching elements). This is the "greedy" part - we take all free matches we can get.

5. **Termination check**: After exploring all diagonals for edit distance k, check if any path has reached the target position (len(A), len(B)). If so, we've found the optimal solution with k edits.

6. **Backtracking for edit script**: Once we've found the optimal edit distance, we need to reconstruct the actual sequence of edits. This requires backtracking through our exploration data to find which moves led to the optimal path.

The algorithm's time complexity is O((M+N)D) where M and N are the lengths of the sequences and D is the length of the shortest edit script. In practice, this performs much better than the theoretical worst case because D is typically much smaller than M+N for most real-world comparisons.

| Algorithm Phase | Purpose | Data Structure | Time Complexity |
|-----------------|---------|----------------|-----------------|
| Forward Pass | Find minimum edit distance | V array tracking furthest x per diagonal | O((M+N)D) |
| Greedy Extension | Maximize diagonal moves within each step | Direct sequence comparison | O(matching characters) |
| Backtrack Pass | Reconstruct edit sequence | Trace through V snapshots | O(D) |
| Edit Script Generation | Convert path to insert/delete operations | List of edit commands | O(D) |

The implementation requires careful handling of several edge cases and optimizations:

**Diagonal indexing**: Since diagonals can have negative indices (when y > x), we need to offset our array indices appropriately. The diagonal d=x-y ranges from -N to M, so we use V[d+N] to ensure non-negative array indices.

**Memory optimization**: The basic algorithm stores the entire V array for each edit distance k, which can consume significant memory for large files. Advanced implementations use techniques like "linear space refinement" to reduce memory usage by recomputing portions of the search space as needed.

**Snake detection**: A "snake" in Myers' terminology is a sequence of diagonal moves (matching elements). The algorithm spends most of its time detecting and following these snakes, so optimizing this inner loop is crucial for performance.

**Boundary conditions**: Special care is needed when exploring diagonals at the edges of the search space, where we might only be able to reach a diagonal from one direction rather than both.

> **Key Insight**: Myers' algorithm's efficiency comes from recognizing that most file changes are localized. By greedily consuming matching content (diagonal moves) whenever possible, the algorithm quickly navigates through unchanged regions and focuses computational effort on the areas where real differences exist.

#### Architecture Decision Records

> **Decision: Myers Algorithm vs. Patience Diff**
> - **Context**: Multiple diff algorithms exist with different trade-offs in quality and performance. Myers finds shortest edit distance, while Patience diff often produces more intuitive results for code with moved functions.
> - **Options Considered**: Myers' algorithm (shortest edit script), Patience diff (unique line anchoring), Hunt-McIlroy algorithm (classic LCS-based)
> - **Decision**: Implement Myers' algorithm as the primary diff engine
> - **Rationale**: Myers provides optimal edit distance with reasonable performance, matches Git's internal implementation for compatibility, and has well-understood complexity characteristics. Patience diff can be added later as an alternative strategy.
> - **Consequences**: Optimal diff output for most cases, some counterintuitive results when functions are moved (can be addressed with post-processing), compatible with existing Git tooling expectations.

| Algorithm Option | Pros | Cons | Performance | Quality |
|------------------|------|------|-------------|---------|
| Myers' Algorithm | Optimal edit distance, well-studied, Git-compatible | May miss intuitive moves | O((M+N)D) | Mathematically optimal |
| Patience Diff | Better handling of moved code blocks | More complex, may not be shortest | O(N log N) typical | More intuitive for code |
| Hunt-McIlroy | Simple LCS-based approach | Not optimized for text | O(MN) worst case | Basic but functional |

> **Decision: Line-based vs. Character-based Comparison**
> - **Context**: Diff algorithms can operate at different granularities - comparing entire lines, individual characters, or words. Each choice affects both performance and output usefulness.
> - **Options Considered**: Character-level diff (finest granularity), word-level diff (balanced approach), line-level diff (Git standard)
> - **Decision**: Implement line-based comparison as primary mode with character-level available for within-line changes
> - **Rationale**: Line-based comparison matches user expectations for code review, provides good performance for large files, and aligns with Git's standard behavior. Character-level can be used for highlighting intra-line changes.
> - **Consequences**: Fast comparison of large files, familiar output format, may miss some fine-grained changes within lines (addressed by intra-line diff post-processing).

> **Decision: In-memory vs. Streaming Comparison**
> - **Context**: Large files may not fit in memory, requiring either streaming algorithms or memory-mapped file access. This affects both memory usage and algorithm complexity.
> - **Options Considered**: Full in-memory comparison, streaming with limited lookahead, memory-mapped file access
> - **Decision**: Use in-memory comparison with fallback to "binary file" detection for very large files
> - **Rationale**: Most source code files are small enough for in-memory processing, simplifies algorithm implementation, and provides better performance for typical use cases. Binary file detection prevents memory exhaustion on large files.
> - **Consequences**: Excellent performance for typical source files, may not handle very large text files optimally, requires size-based heuristics to detect problematic files.

### Unified Diff Output Format

The unified diff format represents the industry standard for presenting file differences in a human-readable form. Developed as an improvement over the older "context diff" format, unified diff consolidates additions and deletions into single hunks, making it easier to understand what changed while providing sufficient context for applying patches accurately.

Understanding unified diff format is crucial because it serves as the common language between different version control systems, patch utilities, and code review tools. When you see a GitHub pull request, apply a patch with `git apply`, or review changes in your IDE, you're likely looking at some variation of unified diff output.

The format consists of several distinct components, each serving a specific purpose in conveying change information:

**File headers** identify the files being compared and provide metadata about the comparison. The traditional unified diff header includes the original filename prefixed with `---` and the new filename prefixed with `+++`. These lines may include timestamps or revision identifiers, though Git typically uses commit hashes or symbolic names like "HEAD" instead of timestamps.

**Hunk headers** mark the beginning of each contiguous block of changes. A hunk header appears as `@@ -start,count +start,count @@` where the first pair describes the line range in the original file and the second pair describes the corresponding range in the new file. The counts indicate how many lines from each file are included in this hunk, including both changed and context lines.

**Context lines** provide unchanged content around modifications to help readers understand where changes occurred. These lines begin with a space character and appear exactly as they exist in both files. The standard is to include three lines of context before and after each change, though this is configurable.

**Deletion lines** show content that was removed from the original file. These lines begin with a `-` character followed by the content that was deleted. Multiple consecutive deletion lines indicate a block of content that was removed.

**Addition lines** show content that was added in the new file. These lines begin with a `+` character followed by the new content. When deletion and addition lines appear together, they typically represent content that was modified rather than purely removed and added.

Here's the detailed structure of unified diff output:

| Component | Format | Example | Purpose |
|-----------|--------|---------|---------|
| File Header (original) | `--- filename` | `--- a/src/main.py` | Identifies source file |
| File Header (new) | `+++ filename` | `+++ b/src/main.py` | Identifies target file |
| Hunk Header | `@@ -start,count +start,count @@` | `@@ -15,7 +15,8 @@` | Defines change location and scope |
| Context Line | ` content` | ` def process_data():` | Shows unchanged content for reference |
| Deletion Line | `-content` | `- return None` | Shows content removed from original |
| Addition Line | `+content` | `+ return process_result()` | Shows content added in new version |
| No newline marker | `\ No newline at end of file` | `\ No newline at end of file` | Indicates file doesn't end with newline |

The algorithm for generating unified diff output operates in several phases:

1. **Edit script processing**: Convert the Myers algorithm output (a sequence of insert/delete operations) into a list of change regions. Consecutive operations of the same type can be grouped together for efficiency.

2. **Context calculation**: For each change region, determine the appropriate context lines to include. This requires reading the unchanged content before and after each change, typically 3 lines in each direction.

3. **Hunk consolidation**: Merge nearby change regions into single hunks when their context areas overlap. This prevents repetitive context lines and produces more readable output.

4. **Line number calculation**: Track line numbers in both the original and new files as we process changes. This information is needed for the hunk headers and must account for how previous changes affect subsequent line numbering.

5. **Output formatting**: Generate the actual diff text with proper prefixes, ensuring that each line type is correctly marked and that the overall format follows the unified diff specification.

The implementation must handle several special cases that commonly occur in real-world file comparisons:

**Files with no final newline**: When a file doesn't end with a newline character, the unified diff format includes a special marker `\ No newline at end of file` after the affected line. This distinction is important because adding or removing a final newline is a meaningful change in many contexts.

**Empty files**: Comparing to or from empty files requires special handling in the hunk headers. An empty file is represented with line numbers `0,0`, and the hunk header must reflect this appropriately.

**Large change regions**: When an entire file has been rewritten, generating a diff with every line marked as deleted and added may not be useful. Some implementations detect this case and either display a summary message or fall back to binary file handling.

**Binary file detection**: The unified diff format is designed for text files. When binary content is detected (through null bytes or other heuristics), the output should indicate that the files differ without attempting to show the actual differences.

**Tab vs. space handling**: Unified diff output should preserve the exact whitespace from the original files, including tabs and spaces. However, some display contexts may render these differently, so care must be taken to maintain fidelity.

> **Implementation Note**: Generating clean unified diff output requires careful attention to line ending handling. Mixed line endings (LF vs. CRLF) within the same file can produce confusing output if not normalized consistently.

#### Common Pitfalls

⚠️ **Pitfall: Incorrect Line Number Tracking**
A frequent mistake is failing to properly track line numbers as changes are processed. The line numbers in hunk headers must reflect the position in the original and new files respectively, accounting for how previous insertions and deletions shift subsequent positions. This is especially tricky when multiple hunks are generated for the same file comparison.

**Why it's wrong**: Incorrect line numbers make the diff output unusable for patch applications and confuse users trying to locate changes in their editors.

**How to fix**: Maintain separate line counters for the original and new files, updating them as each edit operation is processed. Validate line numbers by ensuring they align with the actual content being compared.

⚠️ **Pitfall: Context Line Boundary Errors**
When extracting context lines around changes, implementations often fail to handle file boundaries correctly. Requesting 3 lines of context before a change that occurs at line 2 should gracefully handle the fact that only 1 context line is available.

**Why it's wrong**: Attempting to access lines before the beginning or after the end of a file will cause array bounds exceptions or produce incorrect context.

**How to fix**: Always clamp context line ranges to the actual file boundaries using `max(0, start_line - context)` and `min(file_length, end_line + context)` when calculating context regions.

⚠️ **Pitfall: Inefficient String Building**
Generating unified diff output often involves concatenating many small strings (line prefixes, content, newlines). Using simple string concatenation in a loop can result in O(n²) performance due to string immutability in many languages.

**Why it's wrong**: Large files or files with many changes will experience severe performance degradation, making the diff operation unusably slow.

**How to fix**: Use a string builder or array-based approach that can efficiently accumulate output. Build each hunk separately and then combine them, rather than building the entire output character by character.

⚠️ **Pitfall: Inconsistent Newline Handling**
Text files may use different newline conventions (LF on Unix, CRLF on Windows, or even mixed). If the diff algorithm doesn't handle these consistently, it may report spurious changes where only line endings differ.

**Why it's wrong**: Users will see confusing diffs where every line appears to have changed, even when only line endings are different.

**How to fix**: Implement a consistent newline normalization strategy, either by converting all content to a standard format before comparison or by making the line-splitting logic aware of different newline conventions.

### Implementation Guidance

The diff algorithm implementation requires careful coordination between the core Myers algorithm, output formatting, and integration with Git's object model. The following guidance provides both complete infrastructure and skeleton implementations to help you build a robust diff system.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Line Splitting | `str.splitlines()` with keepends=True | Custom line iterator with encoding detection |
| Output Building | List accumulation with `''.join()` | `io.StringIO` for memory-efficient streaming |
| File Comparison | Full file reading with `Path.read_text()` | Memory-mapped files with `mmap` module |
| Binary Detection | Check for null bytes in first 1024 chars | Use `chardet` library for encoding confidence |
| Performance Profiling | Simple timing with `time.perf_counter()` | `cProfile` with line-by-line analysis |

#### Recommended File Structure

```
project-root/
├── src/
│   ├── diff/
│   │   ├── __init__.py           ← Public diff API
│   │   ├── myers.py              ← Myers algorithm implementation
│   │   ├── unified.py            ← Unified diff formatting
│   │   ├── types.py              ← Diff data structures
│   │   └── utils.py              ← File reading utilities
│   ├── objects/
│   │   └── retrieval.py          ← Object content access
│   └── repository.py             ← Main Repository class
├── tests/
│   ├── diff/
│   │   ├── test_myers.py         ← Myers algorithm tests
│   │   ├── test_unified.py       ← Output format tests
│   │   └── fixtures/             ← Test file pairs
│   └── integration/
│       └── test_git_compatibility.py  ← Compare with real Git
└── examples/
    └── diff_demo.py              ← Usage examples
```

#### Infrastructure Starter Code

**File: `src/diff/types.py`** - Complete data structures for diff operations:

```python
"""Data structures for diff algorithm implementation."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Union


class EditType(Enum):
    """Types of edit operations in a diff."""
    INSERT = "insert"
    DELETE = "delete"
    MATCH = "match"


@dataclass(frozen=True)
class Edit:
    """A single edit operation."""
    type: EditType
    old_line: Optional[int]  # Line number in original file (None for inserts)
    new_line: Optional[int]  # Line number in new file (None for deletes)
    content: str             # The actual line content


@dataclass(frozen=True)
class DiffHunk:
    """A contiguous block of changes with context."""
    old_start: int           # Starting line in original file
    old_count: int           # Number of lines from original file
    new_start: int           # Starting line in new file  
    new_count: int           # Number of lines from new file
    lines: List[str]         # Formatted diff lines (with +/- prefixes)


@dataclass(frozen=True)
class FileDiff:
    """Complete diff result for a single file comparison."""
    old_path: Optional[str]   # Original file path (None for new file)
    new_path: Optional[str]   # New file path (None for deleted file)
    is_binary: bool          # True if files contain binary data
    hunks: List[DiffHunk]    # List of change hunks


class DiffStats:
    """Statistics about a diff operation."""
    
    def __init__(self):
        self.files_changed = 0
        self.insertions = 0
        self.deletions = 0
        self.binary_files = 0
    
    def add_file_diff(self, file_diff: FileDiff) -> None:
        """Update statistics with results from a file diff."""
        self.files_changed += 1
        
        if file_diff.is_binary:
            self.binary_files += 1
            return
        
        for hunk in file_diff.hunks:
            for line in hunk.lines:
                if line.startswith('+') and not line.startswith('+++'):
                    self.insertions += 1
                elif line.startswith('-') and not line.startswith('---'):
                    self.deletions += 1
    
    def __str__(self) -> str:
        """Generate summary string like Git's diff --stat output."""
        parts = [f"{self.files_changed} file(s) changed"]
        if self.insertions:
            parts.append(f"{self.insertions} insertion(s)")
        if self.deletions:
            parts.append(f"{self.deletions} deletion(s)")
        return ", ".join(parts)
```

**File: `src/diff/utils.py`** - Complete utilities for file handling:

```python
"""Utilities for file reading and binary detection."""

import os
from pathlib import Path
from typing import List, Optional, Tuple


def is_binary_content(content: bytes) -> bool:
    """Detect if content appears to be binary data."""
    if not content:
        return False
    
    # Check for null bytes (strong binary indicator)
    if b'\x00' in content[:1024]:  # Check first 1KB
        return True
    
    # Check for high ratio of non-printable characters
    printable_chars = sum(1 for b in content[:1024] 
                         if 32 <= b <= 126 or b in (9, 10, 13))
    if len(content[:1024]) > 0:
        printable_ratio = printable_chars / len(content[:1024])
        return printable_ratio < 0.75
    
    return False


def read_file_lines(file_path: Path) -> Tuple[List[str], bool]:
    """Read file and return lines with binary detection.
    
    Returns:
        Tuple of (lines, is_binary) where lines are empty if binary.
    """
    try:
        # Read as binary first for detection
        with open(file_path, 'rb') as f:
            content = f.read()
        
        if is_binary_content(content):
            return [], True
        
        # Convert to text and split into lines
        try:
            text_content = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                text_content = content.decode('latin1')
            except UnicodeDecodeError:
                return [], True
        
        # Split lines but keep line endings for accurate diff
        lines = text_content.splitlines(keepends=True)
        return lines, False
        
    except (IOError, OSError):
        return [], True


def safe_file_size(file_path: Path) -> int:
    """Get file size safely, returning 0 if file doesn't exist."""
    try:
        return file_path.stat().st_size
    except (OSError, IOError):
        return 0


def normalize_path(path: Optional[str]) -> Optional[str]:
    """Normalize file path for display in diff headers."""
    if path is None:
        return None
    
    # Convert to forward slashes for consistency
    normalized = str(Path(path)).replace(os.sep, '/')
    
    # Add a/ or b/ prefix if not already present (Git convention)
    if not normalized.startswith(('a/', 'b/')):
        return f"a/{normalized}"
    
    return normalized
```

#### Core Logic Skeleton Code

**File: `src/diff/myers.py`** - Myers algorithm implementation skeleton:

```python
"""Myers diff algorithm implementation."""

from typing import List, Dict, Tuple, Optional
from .types import Edit, EditType


class MyersDiff:
    """Implementation of Myers' O(ND) diff algorithm."""
    
    def __init__(self, old_lines: List[str], new_lines: List[str]):
        self.old_lines = old_lines
        self.new_lines = new_lines
        self.M = len(old_lines)
        self.N = len(new_lines)
    
    def compute_diff(self) -> List[Edit]:
        """Compute the shortest edit script using Myers algorithm.
        
        Returns:
            List of Edit operations to transform old_lines into new_lines.
        """
        # TODO 1: Handle edge cases (empty files)
        if self.M == 0:
            # TODO: Return all insertions
            pass
        if self.N == 0:
            # TODO: Return all deletions
            pass
        
        # TODO 2: Find the shortest edit script length
        edit_distance = self._find_shortest_edit_distance()
        
        # TODO 3: Backtrack to reconstruct the actual edit sequence
        return self._backtrack_edit_script(edit_distance)
    
    def _find_shortest_edit_distance(self) -> int:
        """Find the length of the shortest edit script.
        
        Returns:
            The minimum number of insertions + deletions needed.
        """
        # Myers algorithm uses a V array to track furthest reaching paths
        # V[d] = furthest x coordinate reached on diagonal d
        # Diagonal d represents positions where x - y = d
        
        max_d = self.M + self.N
        v = {}  # V array: diagonal -> furthest x position
        
        # TODO 4: Initialize V[1] = 0 (starting position)
        
        # TODO 5: For each possible edit distance from 0 to max_d:
        for d in range(max_d + 1):
            # TODO 6: For each diagonal that could be reached with d edits:
            for k in range(-d, d + 1, 2):
                # TODO 7: Determine starting x position on diagonal k
                # Can reach diagonal k from k-1 (move right/delete) or k+1 (move down/insert)
                if k == -d or (k != d and v.get(k-1, 0) < v.get(k+1, 0)):
                    # TODO: Move down from diagonal k+1 (insert from new)
                    x = v.get(k+1, 0)
                else:
                    # TODO: Move right from diagonal k-1 (delete from old)  
                    x = v.get(k-1, 0) + 1
                
                # TODO 8: Calculate y position from x and diagonal
                y = x - k
                
                # TODO 9: Extend diagonally while elements match (greedy)
                while (x < self.M and y < self.N and 
                       self.old_lines[x] == self.new_lines[y]):
                    # TODO: Advance both x and y
                    pass
                
                # TODO 10: Store furthest position reached on this diagonal
                v[k] = x
                
                # TODO 11: Check if we've reached the target position
                if x >= self.M and y >= self.N:
                    return d
        
        # Should never reach here for valid inputs
        return max_d
    
    def _backtrack_edit_script(self, edit_distance: int) -> List[Edit]:
        """Reconstruct the edit script by backtracking through the search.
        
        Args:
            edit_distance: The optimal edit distance found
            
        Returns:
            List of Edit operations in forward order.
        """
        # TODO 12: Re-run the forward search but save V arrays for each step
        v_history = self._compute_v_history(edit_distance)
        
        # TODO 13: Start from the end position and work backwards
        x, y = self.M, self.N
        edits = []
        
        # TODO 14: For each edit distance from max down to 0:
        for d in range(edit_distance, 0, -1):
            v = v_history[d]
            v_prev = v_history[d-1]
            
            # TODO 15: Find which diagonal we're on
            k = x - y
            
            # TODO 16: Determine how we reached this position
            # Check if we came from diagonal k-1 (right move) or k+1 (down move)
            if k == -d or (k != d and v_prev.get(k-1, 0) < v_prev.get(k+1, 0)):
                # TODO: We moved down (insertion)
                prev_k = k + 1
            else:
                # TODO: We moved right (deletion)  
                prev_k = k - 1
            
            # TODO 17: Calculate previous position
            prev_x = v_prev[prev_k]
            prev_y = prev_x - prev_k
            
            # TODO 18: Add diagonal moves (matches) first
            while x > prev_x and y > prev_y:
                # TODO: Add MATCH edit for diagonal moves
                x -= 1
                y -= 1
                # edits.insert(0, Edit(EditType.MATCH, x, y, self.old_lines[x]))
            
            # TODO 19: Add the actual edit operation
            if x > prev_x:
                # TODO: Deletion (moved right)
                x -= 1
                # edits.insert(0, Edit(EditType.DELETE, x, None, self.old_lines[x]))
            elif y > prev_y:
                # TODO: Insertion (moved down)
                y -= 1
                # edits.insert(0, Edit(EditType.INSERT, None, y, self.new_lines[y]))
        
        return edits
    
    def _compute_v_history(self, max_d: int) -> Dict[int, Dict[int, int]]:
        """Recompute the V arrays and store history for backtracking."""
        # TODO 20: This is similar to _find_shortest_edit_distance but saves all V arrays
        # Return dict where v_history[d] = V array after processing edit distance d
        pass
```

**File: `src/diff/unified.py`** - Unified diff formatting skeleton:

```python
"""Unified diff output formatting."""

from typing import List, Optional
from .types import Edit, EditType, FileDiff, DiffHunk


class UnifiedDiffFormatter:
    """Formats diff results as unified diff output."""
    
    def __init__(self, context_lines: int = 3):
        self.context_lines = context_lines
    
    def format_file_diff(self, file_diff: FileDiff) -> str:
        """Format a complete file diff in unified format.
        
        Args:
            file_diff: The diff result to format
            
        Returns:
            Unified diff string ready for display.
        """
        if file_diff.is_binary:
            return self._format_binary_diff(file_diff)
        
        lines = []
        
        # TODO 1: Add file headers (--- and +++ lines)
        if file_diff.old_path:
            # TODO: Add "--- a/path" line
            pass
        else:
            # TODO: Add "--- /dev/null" for new files
            pass
            
        if file_diff.new_path:
            # TODO: Add "+++ b/path" line  
            pass
        else:
            # TODO: Add "+++ /dev/null" for deleted files
            pass
        
        # TODO 2: Add each hunk with its header
        for hunk in file_diff.hunks:
            # TODO: Add hunk header like "@@ -15,7 +15,8 @@"
            hunk_header = f"@@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@"
            lines.append(hunk_header)
            
            # TODO: Add all hunk lines (already formatted with +/- prefixes)
            lines.extend(hunk.lines)
        
        return '\n'.join(lines) + '\n' if lines else ""
    
    def edits_to_hunks(self, edits: List[Edit], old_lines: List[str], 
                      new_lines: List[str]) -> List[DiffHunk]:
        """Convert edit sequence to unified diff hunks with context.
        
        Args:
            edits: Sequence of edit operations
            old_lines: Original file lines
            new_lines: New file lines
            
        Returns:
            List of DiffHunk objects with context lines added.
        """
        if not edits:
            return []
        
        # TODO 3: Group edits into change regions
        change_regions = self._group_edits_into_regions(edits)
        
        # TODO 4: Add context around each region  
        hunks = []
        for region in change_regions:
            hunk = self._create_hunk_with_context(region, old_lines, new_lines)
            hunks.append(hunk)
        
        # TODO 5: Merge overlapping hunks
        return self._merge_adjacent_hunks(hunks, old_lines, new_lines)
    
    def _group_edits_into_regions(self, edits: List[Edit]) -> List[List[Edit]]:
        """Group consecutive edits into change regions."""
        # TODO 6: Separate MATCH edits from INSERT/DELETE edits
        # TODO 7: Group consecutive non-MATCH edits together
        # TODO 8: Return list of edit groups representing distinct change areas
        pass
    
    def _create_hunk_with_context(self, region_edits: List[Edit], 
                                old_lines: List[str], new_lines: List[str]) -> DiffHunk:
        """Create a single hunk with context lines around the changes."""
        # TODO 9: Find the line range covered by this region
        min_old_line = min((e.old_line for e in region_edits if e.old_line is not None), default=0)
        max_old_line = max((e.old_line for e in region_edits if e.old_line is not None), default=0)
        
        # TODO 10: Calculate context boundaries
        context_start_old = max(0, min_old_line - self.context_lines)
        context_end_old = min(len(old_lines), max_old_line + self.context_lines + 1)
        
        # TODO 11: Build the hunk lines with proper prefixes
        hunk_lines = []
        
        # TODO 12: Add leading context
        for i in range(context_start_old, min_old_line):
            # TODO: Add context line with space prefix
            pass
        
        # TODO 13: Add the actual changes
        for edit in region_edits:
            if edit.type == EditType.DELETE:
                # TODO: Add line with - prefix
                pass
            elif edit.type == EditType.INSERT:
                # TODO: Add line with + prefix
                pass
            # MATCH edits within a change region become context
        
        # TODO 14: Add trailing context
        for i in range(max_old_line + 1, context_end_old):
            # TODO: Add context line with space prefix
            pass
        
        # TODO 15: Calculate hunk header numbers
        old_start = context_start_old + 1  # 1-based line numbers
        old_count = context_end_old - context_start_old
        # TODO: Calculate new_start and new_count similarly
        
        return DiffHunk(
            old_start=old_start,
            old_count=old_count, 
            new_start=0,  # TODO: Calculate this
            new_count=0,  # TODO: Calculate this
            lines=hunk_lines
        )
    
    def _merge_adjacent_hunks(self, hunks: List[DiffHunk], 
                            old_lines: List[str], new_lines: List[str]) -> List[DiffHunk]:
        """Merge hunks whose context regions overlap."""
        # TODO 16: Check each pair of consecutive hunks
        # TODO 17: If context regions overlap, merge them into single hunk
        # TODO 18: Recalculate line counts after merging
        pass
    
    def _format_binary_diff(self, file_diff: FileDiff) -> str:
        """Format a binary file diff message."""
        old_name = file_diff.old_path or "/dev/null"
        new_name = file_diff.new_path or "/dev/null" 
        
        return (f"--- {old_name}\n"
                f"+++ {new_name}\n"
                f"Binary files {old_name} and {new_name} differ\n")
```

#### Milestone Checkpoints

After implementing the diff algorithm, verify your implementation with these specific tests:

**Basic Functionality Test:**
```bash
# Create test files
echo -e "line1\nline2\nline3" > old.txt
echo -e "line1\nmodified\nline3\nnew line" > new.txt

# Your diff should output:
# --- old.txt
# +++ new.txt  
# @@ -1,3 +1,4 @@
#  line1
# -line2
# +modified
#  line3
# +new line
```

**Empty File Test:**
```bash
touch empty.txt
echo "content" > nonempty.txt

# Diff from empty should show all additions
# Diff to empty should show all deletions
```

**Binary File Test:**
```bash
# Create a binary file (contains null bytes)
printf "binary\x00data" > binary.dat
echo "text content" > text.txt

# Should output: "Binary files binary.dat and text.txt differ"
```

**Performance Test:**
```bash
# Generate large text files (1000+ lines)
# Diff should complete within reasonable time (< 1 second for 1000 lines)
# Memory usage should remain reasonable (< 100MB for typical files)
```

**Git Compatibility Test:**
```bash
# Compare your diff output with Git's output for same files
git diff --no-index old.txt new.txt > git_output.diff
your_git diff old.txt new.txt > your_output.diff

# Outputs should be functionally equivalent (may differ in header details)
```

#### Language-Specific Hints

**Python Performance Tips:**
- Use `str.splitlines(keepends=True)` to preserve line ending information
- Build diff output using lists and `''.join()` rather than string concatenation
- Consider `collections.defaultdict(int)` for the V array in Myers algorithm
- Use `enumerate()` when you need both index and value during iteration

**Memory Management:**
- For very large files, consider implementing a streaming version that processes chunks
- The Myers algorithm V array can grow large - consider the linear space refinement for huge diffs
- Use generators where possible to avoid loading entire diff output into memory

**Error Handling:**
- Wrap file I/O operations in try/except blocks for graceful error handling
- Detect and handle different text encodings (UTF-8, Latin1, etc.)
- Provide meaningful error messages when files cannot be read or compared

**Testing Strategy:**
- Create a comprehensive test suite with edge cases (empty files, binary files, identical files)
- Include performance tests with large files to catch algorithmic issues
- Test against Git's output for compatibility verification
- Use property-based testing to generate random file pairs for robustness testing


## Three-Way Merge Implementation

> **Milestone(s):** This section is the culmination of Milestone 8 (Three-way Merge), building upon all previous milestones. The merge algorithm requires the object storage (Milestone 2-4), references (Milestone 5), index management (Milestone 6), and diff algorithms (Milestone 7) to function correctly.

### Mental Model: Collaborative Document Editing

Think of three-way merging like collaborative document editing in the pre-digital era. Imagine you and a colleague both receive copies of the same research paper draft on Monday morning. You each spend the week making independent edits - you focus on improving the introduction and methodology, while your colleague refines the conclusion and adds new references. On Friday, you need to combine both sets of changes into a single, improved document.

The naive approach would be to simply compare your final version with your colleague's final version. However, this creates confusion - if the same paragraph appears differently in both versions, you have no way to know whether both of you changed it (creating a conflict) or only one person modified it while the other left it unchanged. The solution is to keep the original Monday morning version as a reference point.

With the original document as your **merge base**, you can now perform a three-way comparison. For each paragraph, you examine three versions: the original (base), your version (branch A), and your colleague's version (branch B). If only you changed a paragraph, your changes win. If only your colleague changed it, their changes win. If neither of you touched it, it stays the same. But if both of you modified the same paragraph, you have a genuine conflict that requires manual resolution - perhaps you need to sit down together and decide how to combine both improvements.

This is exactly how Git's three-way merge works. The **merge base** is the common ancestor commit - the last point where both branches shared the same history. By comparing each file's content across all three versions (base, branch A, branch B), Git can automatically resolve most changes and only flag genuine conflicts where both branches modified the same lines.

The power of this approach is that it preserves the intent of both sets of changes. If you deleted a paragraph that your colleague left unchanged, the merge knows you intentionally removed it and won't restore it. If you both added different content at the same location, the merge knows this requires human judgment to resolve properly.

### Finding the Merge Base

The merge base calculation is a graph traversal problem that finds the **lowest common ancestor** (LCA) between two commits in Git's directed acyclic graph. This ancestor represents the most recent point in history where both branches shared the same state, making it the ideal reference point for three-way comparison.

The algorithm uses a breadth-first search approach that explores the commit history backwards from both branch tips simultaneously. The key insight is that Git's commit graph forms a DAG where each commit points to its parent(s), creating paths back through history. The merge base is the commit that appears in both traversal paths with the shortest distance from either starting point.

![Three-Way Merge Process](./diagrams/three-way-merge.svg)

**Merge Base Discovery Algorithm:**

1. **Initialize Search State**: Create two sets to track visited commits - one for each branch being merged. Initialize two queues with the starting commit hashes from both branches. This dual-queue approach ensures we explore both histories simultaneously and can detect when they intersect.

2. **Parallel Breadth-First Traversal**: While both queues contain commits, dequeue one commit from each queue in alternating fashion. This alternating approach ensures we explore both histories at roughly the same depth, increasing the likelihood of finding the nearest common ancestor quickly rather than exploring one branch much deeper than the other.

3. **Intersection Detection**: For each dequeued commit, check if it already exists in the opposite branch's visited set. If a commit from branch A's queue appears in branch B's visited set, we've found a common ancestor. Record this commit as a potential merge base candidate.

4. **Parent Expansion**: Load the commit object for the current commit and extract its parent commit hashes. Add each parent to the current branch's queue for future exploration. Mark the current commit as visited in the current branch's set to avoid revisiting it later.

5. **Multiple Ancestor Resolution**: Continue the search even after finding the first common ancestor, as there might be multiple common ancestors at the same depth level. In Git's history, merge commits can create situations where two branches have multiple equally-valid common ancestors.

6. **Best Candidate Selection**: Among all discovered common ancestors, select the one with the shortest path distance from either branch tip. This represents the most recent common state and provides the most meaningful base for comparison.

The algorithm handles several edge cases that commonly occur in Git repositories:

| Edge Case | Detection Method | Resolution Strategy |
|-----------|-----------------|-------------------|
| No Common Ancestor | Both queues become empty without intersection | Return null - branches have completely separate histories |
| Self-Merge Attempt | Source and target commits are identical | Return the commit itself as the merge base |
| Fast-Forward Scenario | Target commit appears in source's ancestry | Return target commit - no merge needed, just update reference |
| Multiple Merge Bases | Several ancestors found at same depth | Select the one with shortest path to either tip |
| Octopus Merge History | Commits with more than two parents | Treat each parent equally in the traversal |

**Performance Considerations:**

The merge base calculation can become expensive in repositories with deep history or complex branching patterns. The algorithm's time complexity is O(N) where N is the number of commits that must be examined before finding the common ancestor. In the worst case, this could require traversing most of the repository's history.

To optimize performance, the implementation can employ several strategies:

- **Early Termination**: Stop the search immediately upon finding the first common ancestor if the repository structure guarantees a unique merge base
- **Commit Metadata Caching**: Cache commit parent relationships to avoid repeated object loading during traversal
- **Generation Numbers**: Use Git's commit-graph generation numbers (if available) to guide the search toward likely intersection points
- **Path Limiting**: In repositories with known branching patterns, limit the search depth to reasonable bounds

> **Critical Insight**: The merge base calculation is not just an optimization - it's fundamental to Git's ability to understand the intent behind changes. Without the correct merge base, Git cannot distinguish between deliberate deletions and independent additions, leading to incorrect automatic merges or false conflict detection.

### Three-Way Merge Algorithm

The three-way merge algorithm performs a line-by-line comparison between three versions of each file: the merge base (common ancestor), the current branch (ours), and the branch being merged (theirs). This comparison determines which changes can be automatically applied and which require manual conflict resolution.

The algorithm operates on the principle of **change attribution** - understanding which branch introduced each modification relative to the common base. A change is considered safe to apply automatically if only one branch modified a particular region of the file. Conflicts arise when both branches made different changes to the same lines.

**File-Level Merge Process:**

The merge algorithm begins by identifying all files that exist in any of the three versions (base, ours, theirs). For each file, it determines the merge scenario based on which versions contain the file:

| Base Exists | Ours Exists | Theirs Exists | Scenario | Action |
|-------------|-------------|---------------|----------|---------|
| Yes | Yes | Yes | Modified in branches | Perform three-way content merge |
| Yes | Yes | No | Deleted in theirs | Check if ours modified; conflict if yes |
| Yes | No | Yes | Deleted in ours | Check if theirs modified; conflict if yes |
| Yes | No | No | Deleted in both | Remove file (agreement) |
| No | Yes | Yes | Added in both | Compare content; conflict if different |
| No | Yes | No | Added in ours | Keep ours version |
| No | No | Yes | Added in theirs | Keep theirs version |
| No | No | No | Impossible | Error condition |

**Line-Level Merge Logic:**

For files requiring content merging, the algorithm processes each line using three-way comparison logic. The merge result depends on which branches modified each line relative to the base:

1. **Unchanged in All Versions**: If a line appears identically in base, ours, and theirs, include it unchanged in the merge result. This represents content that neither branch chose to modify.

2. **Modified in Ours Only**: If a line differs between base and ours but matches between base and theirs, apply the change from ours. This represents an intentional modification made only on the current branch.

3. **Modified in Theirs Only**: If a line differs between base and theirs but matches between base and ours, apply the change from theirs. This represents an intentional modification made only on the incoming branch.

4. **Modified Identically**: If both branches made the same change to a line (ours and theirs match each other but differ from base), include the shared change. This represents convergent evolution or cherry-picked changes.

5. **Modified Differently**: If base, ours, and theirs all differ from each other, mark this as a conflict requiring manual resolution. The automated merge cannot determine which change takes precedence.

**Region-Based Conflict Detection:**

The merge algorithm groups consecutive conflicting lines into conflict regions to provide meaningful context for resolution. Rather than marking individual lines as conflicts, it identifies blocks of related changes that should be resolved together:

1. **Conflict Region Identification**: Scan the line-by-line comparison results to find sequences of consecutive conflicts or near-conflicts (within a configurable context window).

2. **Context Expansion**: Extend each conflict region to include surrounding unchanged lines for context. This helps developers understand the broader scope of conflicting changes.

3. **Region Consolidation**: Merge adjacent conflict regions if they're separated by only a few unchanged lines. This prevents fragmented conflicts that would be difficult to resolve coherently.

4. **Boundary Refinement**: Adjust conflict region boundaries to align with logical boundaries (function definitions, paragraph breaks) when possible, making the conflicts more intuitive to resolve.

**Merge Result Generation:**

The algorithm produces a merged file containing three types of content:

- **Clean Lines**: Lines where the three-way comparison produced an unambiguous result, included directly in the output
- **Conflict Markers**: Special marker lines that delimit regions requiring manual resolution
- **Conflict Content**: The competing versions from both branches, clearly labeled for comparison

The merge algorithm maintains metadata about the merge process for reporting and debugging:

| Metadata Field | Purpose | Example Value |
|----------------|---------|---------------|
| Files Modified | Count of files requiring changes | 12 |
| Auto-Merged Files | Files merged without conflicts | 8 |
| Conflicted Files | Files requiring manual resolution | 4 |
| Lines Added | Total lines added from both branches | 156 |
| Lines Removed | Total lines removed from both branches | 89 |
| Conflict Regions | Number of distinct conflict areas | 7 |

**Error Handling and Recovery:**

The merge algorithm must handle various error conditions gracefully:

- **Missing Objects**: If any required blob, tree, or commit object is missing from the object store, abort the merge with a clear error message indicating which object is unavailable.
- **Binary File Conflicts**: When both branches modify a binary file differently, mark the entire file as conflicted rather than attempting line-based merging.
- **Permission Conflicts**: If file modes differ between branches (executable vs non-executable), include mode information in conflict markers.
- **Symlink Conflicts**: Handle symbolic links specially, as their target paths may conflict even when the content appears similar.

> **Design Principle**: The three-way merge algorithm prioritizes correctness over convenience. When in doubt, it prefers to flag potential conflicts for manual review rather than making incorrect automatic decisions that could lose work or introduce bugs.

### Conflict Detection and Marking

Conflict detection is the process of identifying regions where both branches made incompatible changes to the same content, requiring human judgment to resolve. The conflict marking system provides a structured format that allows developers to understand the competing changes and make informed decisions about the final content.

Git uses a **conflict marker format** that clearly delineates the boundaries of each conflict and identifies which branch contributed each version of the conflicted content. This format has become an industry standard, recognized by merge tools, IDEs, and diff viewers.

**Conflict Marker Structure:**

When a conflict is detected, the merge algorithm inserts special marker lines into the file content to create a **conflict block**. Each conflict block follows a consistent structure:

```
<<<<<<< HEAD (or branch name)
[Content from the current branch]
=======
[Content from the branch being merged]
>>>>>>> branch-name (or commit hash)
```

The conflict markers serve specific purposes:

- **Opening Marker** (`<<<<<<<`): Indicates the start of a conflict region and identifies the current branch (usually "HEAD" or the branch name)
- **Separator** (`=======`): Divides the two competing versions of the conflicted content
- **Closing Marker** (`>>>>>>>`): Indicates the end of the conflict region and identifies the incoming branch or commit
- **Content Sections**: The actual competing content from each branch, preserved exactly as it appears in each version

**Advanced Conflict Scenarios:**

Beyond simple line-level conflicts, the merge algorithm must handle several complex scenarios:

| Conflict Type | Description | Example Scenario | Resolution Strategy |
|--------------|-------------|------------------|-------------------|
| Add/Add Conflict | Both branches added different content at same location | New function added with different implementations | Present both versions with clear branch labels |
| Delete/Modify Conflict | One branch deleted content, other modified it | Function removed vs function refactored | Show deletion intent vs modification intent |
| Mode Conflict | File permissions differ between branches | File made executable on one branch only | Include mode information in conflict markers |
| Rename Conflict | Same file renamed differently on each branch | README.txt → README.md vs README.txt → readme.txt | Create conflict for filename choice |
| Binary Conflict | Binary files modified differently | Image files updated independently | Mark entire file as binary conflict |

**Context-Aware Conflict Boundaries:**

The merge algorithm attempts to create meaningful conflict boundaries that align with logical content structure. Rather than starting and ending conflicts at arbitrary line boundaries, it analyzes the content to find natural breakpoints:

1. **Function Boundaries**: For source code, attempt to align conflict regions with function or method boundaries, including the complete function signature and closing brace.

2. **Paragraph Boundaries**: For text content, extend conflicts to complete paragraphs or sentences when possible, avoiding mid-sentence breaks that would be confusing to resolve.

3. **Indentation Consistency**: Maintain consistent indentation levels within conflict regions, ensuring that the conflicted code remains syntactically valid for each branch's version.

4. **Comment Preservation**: Include related comments and documentation within conflict regions so that developers have full context for understanding the intent behind each change.

**Conflict Metadata and Reporting:**

The merge system maintains detailed metadata about each conflict to support resolution tools and provide useful feedback to developers:

| Conflict Property | Description | Usage |
|------------------|-------------|-------|
| File Path | Relative path of the conflicted file | Reporting and tool integration |
| Line Range | Start and end line numbers of the conflict | IDE navigation and highlighting |
| Conflict Type | Category of conflict (content, mode, rename) | Specialized resolution workflows |
| Base Content | Content from the merge base version | Three-way merge tools |
| Branch Labels | Names of the conflicting branches | User interface display |
| Complexity Score | Estimated difficulty of resolution | Prioritizing resolution effort |

**Conflict Resolution Workflow:**

The conflict marking system supports both manual and tool-assisted resolution workflows:

1. **Manual Resolution**: Developers can edit the conflicted file directly, removing the conflict markers and combining the content as appropriate. The merge system validates that all markers are removed before allowing the merge to complete.

2. **Merge Tool Integration**: External merge tools can parse the conflict markers to present a three-way view (base, ours, theirs) and generate the resolved content automatically when the user makes selections.

3. **Partial Resolution**: Large conflicts can be resolved incrementally by addressing individual conflict blocks while leaving others for later resolution.

4. **Resolution Validation**: The system checks that resolved files contain no remaining conflict markers and that the content compiles or validates according to project standards.

**Binary File Conflict Handling:**

Binary files require special conflict handling since line-based merging is not applicable:

1. **Binary Detection**: Use heuristics to identify binary content (null bytes, high ratio of non-printable characters, or file extension patterns).

2. **Whole-File Conflicts**: Mark the entire file as conflicted rather than attempting to merge portions of binary content.

3. **Version Selection**: Provide mechanisms for users to select which branch's version of the binary file should be used in the final merge.

4. **External Tool Integration**: Support launching specialized merge tools for specific binary file types (image editors, document processors).

**Performance Optimization for Large Conflicts:**

In files with extensive conflicts, the merge algorithm employs optimization strategies to maintain reasonable performance:

- **Streaming Processing**: Process large files in chunks rather than loading entire content into memory
- **Conflict Batching**: Group small adjacent conflicts to reduce the total number of conflict regions
- **Early Termination**: Stop processing files that exceed conflict thresholds and mark them for manual resolution
- **Memory Management**: Use efficient data structures to minimize memory usage during conflict detection

> **Critical Success Factor**: Effective conflict marking must balance completeness with usability. Too little context makes conflicts difficult to resolve, while too much context overwhelms developers with irrelevant information. The optimal approach provides just enough surrounding content to understand the intent of each change while keeping conflict regions focused on the actual disagreement.

**Common Pitfalls in Conflict Detection:**

⚠️ **Pitfall: Inconsistent Line Ending Handling**
Different operating systems use different line ending conventions (LF vs CRLF). If the merge algorithm doesn't normalize line endings before comparison, it may generate false conflicts where the only difference is the line terminator. Always normalize line endings to a consistent format (typically LF) before performing three-way comparison.

⚠️ **Pitfall: Whitespace-Only Conflicts**
Changes that only affect whitespace (spaces vs tabs, trailing spaces) can create conflicts that are difficult to visualize and resolve. Implement configurable whitespace handling that can either ignore whitespace differences or normalize them according to project standards.

⚠️ **Pitfall: Incomplete Conflict Markers**
If the merge process is interrupted (by system crash, user cancellation, or error), partially written conflict markers can corrupt the file. Always write conflict markers atomically - either the complete conflict block exists or none of it does. Use temporary files and atomic moves to ensure consistency.

⚠️ **Pitfall: Nested Conflict Markers**
If one of the branches being merged already contains conflict markers from a previous merge, the new merge can create nested or malformed conflict structures. Detect existing conflict markers before merging and either resolve them first or use alternative marker styles to avoid confusion.

### Implementation Guidance

The three-way merge implementation requires sophisticated algorithms for graph traversal, content comparison, and conflict resolution. This section provides concrete implementation patterns and complete working code for the supporting infrastructure.

**A. Technology Recommendations:**

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Graph Traversal | Collections.deque with manual BFS | NetworkX for complex graph algorithms |
| Line Processing | Built-in file.readlines() | Memory-mapped files for large content |
| Diff Algorithm | Simple longest common subsequence | Myers algorithm from previous section |
| Conflict Output | String concatenation with markers | Template-based conflict formatting |
| Binary Detection | Check for null bytes in first 1024 bytes | Use python-magic library for MIME type detection |

**B. File Structure:**
```
src/git/
  merge/
    __init__.py                 ← Public merge API
    merge_base.py              ← Merge base calculation algorithms
    three_way_merge.py         ← Core three-way merge logic
    conflict_detection.py     ← Conflict marking and resolution
    merge_result.py           ← Data structures for merge results
    test/
      test_merge_base.py       ← Merge base algorithm tests
      test_three_way_merge.py  ← Three-way merge tests
      test_conflict_resolution.py ← Conflict handling tests
  objects/
    object_store.py           ← Required for commit traversal
  refs/
    reference_manager.py      ← Required for branch resolution
```

**C. Infrastructure Code - Merge Base Calculator:**

```python
from collections import deque
from typing import Optional, Set, Dict, List, Tuple
from pathlib import Path

class MergeBaseCalculator:
    """Finds the lowest common ancestor between two commits using BFS traversal."""
    
    def __init__(self, object_store):
        self.object_store = object_store
        
    def find_merge_base(self, commit_a: str, commit_b: str) -> Optional[str]:
        """
        Find the merge base (lowest common ancestor) between two commits.
        
        Args:
            commit_a: SHA-1 hash of first commit
            commit_b: SHA-1 hash of second commit
            
        Returns:
            SHA-1 hash of merge base commit, or None if no common ancestor
        """
        # Handle trivial cases
        if commit_a == commit_b:
            return commit_a
            
        # Initialize dual BFS queues and visited sets
        queue_a = deque([commit_a])
        queue_b = deque([commit_b])
        visited_a = {commit_a}
        visited_b = {commit_b}
        
        # Track distances for selecting best candidate
        distances = {commit_a: 0, commit_b: 0}
        candidates = []
        
        # Alternating BFS traversal
        while queue_a or queue_b:
            # Process one commit from each queue per iteration
            if queue_a:
                current = queue_a.popleft()
                if current in visited_b:
                    candidates.append((current, distances[current]))
                else:
                    self._expand_parents(current, queue_a, visited_a, distances)
                    
            if queue_b:
                current = queue_b.popleft()  
                if current in visited_a:
                    candidates.append((current, distances[current]))
                else:
                    self._expand_parents(current, queue_b, visited_b, distances)
                    
            # Early termination if we found candidates
            if candidates:
                break
                
        # Return the candidate with shortest distance
        if candidates:
            return min(candidates, key=lambda x: x[1])[0]
        return None
        
    def _expand_parents(self, commit_hash: str, queue: deque, 
                       visited: Set[str], distances: Dict[str, int]):
        """Add commit's parents to the search queue."""
        try:
            commit_obj = self.object_store.retrieve_object(commit_hash)
            if commit_obj[0] != 'commit':
                return
                
            # Parse commit object to extract parent hashes
            content = commit_obj[1].decode('utf-8')
            current_distance = distances[commit_hash]
            
            for line in content.split('\n'):
                if line.startswith('parent '):
                    parent_hash = line[7:].strip()
                    if parent_hash not in visited:
                        visited.add(parent_hash)
                        queue.append(parent_hash)
                        distances[parent_hash] = current_distance + 1
                elif not line.startswith('parent'):
                    # Stop at first non-parent line
                    break
                    
        except Exception:
            # Skip commits we can't read
            pass
```

**D. Core Logic Skeleton - Three-Way Merge:**

```python
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from enum import Enum

class MergeStatus(Enum):
    CLEAN = "clean"
    CONFLICTED = "conflicted"
    
class ConflictType(Enum):
    CONTENT = "content"
    ADD_ADD = "add_add"  
    DELETE_MODIFY = "delete_modify"
    MODE_CHANGE = "mode_change"

@dataclass
class ConflictRegion:
    start_line: int
    end_line: int
    conflict_type: ConflictType
    ours_content: List[str]
    theirs_content: List[str]
    base_content: List[str]

@dataclass 
class MergeResult:
    status: MergeStatus
    merged_content: Optional[str]
    conflicts: List[ConflictRegion]
    files_changed: int
    lines_added: int
    lines_deleted: int

class ThreeWayMerge:
    """Performs three-way merging with automatic conflict detection."""
    
    def __init__(self, object_store, diff_algorithm):
        self.object_store = object_store
        self.diff_algorithm = diff_algorithm
        
    def merge_commits(self, base_commit: str, our_commit: str, 
                     their_commit: str) -> Dict[str, MergeResult]:
        """
        Merge two commits using three-way algorithm.
        
        Returns:
            Dictionary mapping file paths to their merge results
        """
        # TODO 1: Extract tree objects from all three commits
        # TODO 2: Build file lists from each tree (recursively for subdirectories)  
        # TODO 3: Identify all unique file paths across the three trees
        # TODO 4: For each file path, determine the merge scenario (see file-level table above)
        # TODO 5: Call merge_file_content for files requiring content merging
        # TODO 6: Handle add/add conflicts by comparing content hashes
        # TODO 7: Handle delete/modify conflicts by checking if deleted file was modified
        # TODO 8: Aggregate results and return file path -> MergeResult mapping
        pass
        
    def merge_file_content(self, base_content: str, our_content: str, 
                          their_content: str, file_path: str) -> MergeResult:
        """
        Perform line-by-line three-way merge of file content.
        
        Args:
            base_content: Content from merge base commit
            our_content: Content from current branch  
            their_content: Content from branch being merged
            file_path: Path for error reporting
            
        Returns:
            MergeResult with merged content or conflicts
        """
        # TODO 1: Split each content version into lines (handle different line endings)
        # TODO 2: Run diff algorithm between base->ours and base->theirs
        # TODO 3: Create line mapping showing which lines changed in each branch
        # TODO 4: Iterate through lines applying three-way merge logic:
        #         - Unchanged in both: keep base version  
        #         - Changed in ours only: apply our change
        #         - Changed in theirs only: apply their change
        #         - Changed identically: keep the shared change
        #         - Changed differently: mark as conflict
        # TODO 5: Group consecutive conflicts into regions with context
        # TODO 6: Generate conflict markers for unresolved regions  
        # TODO 7: Return MergeResult with final content and conflict metadata
        pass
        
    def _detect_conflicts(self, base_lines: List[str], our_lines: List[str],
                         their_lines: List[str]) -> List[ConflictRegion]:
        """
        Identify regions where both branches made incompatible changes.
        
        Returns:
            List of conflict regions requiring manual resolution
        """  
        # TODO 1: Use Myers diff to find edit scripts for base->ours and base->theirs
        # TODO 2: Build change maps showing which lines were modified in each branch
        # TODO 3: Find overlapping changes where both branches modified same line ranges
        # TODO 4: Classify conflict types (content, add/add, delete/modify)  
        # TODO 5: Group adjacent conflicts with context lines
        # TODO 6: Create ConflictRegion objects with all necessary metadata
        pass
        
    def _generate_conflict_markers(self, conflict: ConflictRegion, 
                                  our_branch: str, their_branch: str) -> List[str]:
        """
        Generate standard Git conflict markers for a conflict region.
        
        Returns:
            List of lines with conflict markers inserted
        """
        # TODO 1: Create opening marker with our branch name: "<<<<<<< {our_branch}"
        # TODO 2: Add all lines from our version of the conflict
        # TODO 3: Add separator marker: "======="  
        # TODO 4: Add all lines from their version of the conflict
        # TODO 5: Add closing marker with their branch name: ">>>>>>> {their_branch}"
        # TODO 6: Return complete conflict block as list of lines
        pass
        
    def _is_binary_content(self, content: bytes) -> bool:
        """Check if content appears to be binary data."""
        if len(content) == 0:
            return False
            
        # Check first 1024 bytes for null bytes or high ratio of non-printable chars
        sample = content[:1024]
        if b'\x00' in sample:
            return True
            
        printable_chars = sum(1 for byte in sample if 32 <= byte <= 126 or byte in [9, 10, 13])
        ratio = printable_chars / len(sample) if sample else 1.0
        return ratio < 0.75
```

**E. Language-Specific Implementation Hints:**

- **Graph Traversal**: Use `collections.deque` for BFS queues - it has O(1) append/popleft operations compared to O(n) for lists
- **String Processing**: Use `str.splitlines(keepends=True)` to preserve line ending information during merge
- **Memory Management**: For large files, consider using generators or streaming processing rather than loading entire content
- **Path Handling**: Use `pathlib.Path` consistently and call `.resolve()` to handle symbolic links properly
- **Error Handling**: Catch `KeyError` when looking up objects that might not exist, and `UnicodeDecodeError` when processing potentially binary content

**F. Milestone Checkpoint:**

After implementing three-way merge:

1. **Test Basic Merge**: Create two branches that modify different files, merge should complete cleanly
2. **Test Conflict Detection**: Create branches that modify the same lines, verify conflict markers appear
3. **Test Merge Base**: Verify merge base calculation finds correct common ancestor
4. **Integration Test**: Run `python -m git merge branch-name` and compare results with real Git

Expected behavior:
- Clean merges complete without user intervention  
- Conflicted files contain properly formatted conflict markers
- Merge commit has two parent references
- Working directory contains merged content

**G. Debugging Tips:**

| Symptom | Likely Cause | Diagnosis | Fix |
|---------|--------------|-----------|-----|
| "No merge base found" | Branches have separate histories | Check commit ancestry with log | Verify both branches descend from same root |
| Merge creates wrong conflicts | Incorrect diff algorithm | Compare diff output with Git's | Debug Myers algorithm implementation |
| Missing conflict markers | Binary file detected incorrectly | Check file content detection | Adjust binary detection thresholds |
| Malformed conflict output | Line ending inconsistencies | Check line splitting logic | Normalize line endings before processing |


## Component Interactions and Data Flow

> **Milestone(s):** This section synthesizes all eight milestones, showing how components interact during complex operations. It's particularly crucial for understanding the complete workflows in Milestone 6 (Index/Staging Area), Milestone 7 (Diff Algorithm), and Milestone 8 (Three-Way Merge), while demonstrating how earlier milestones (Object Storage, References, Tree Objects, Commit Objects) integrate into cohesive operations.

### Mental Model: The Assembly Line Factory

Think of Git operations as an assembly line factory where raw materials (your file changes) flow through different stations, getting processed and transformed at each stage. The **working directory** is your raw materials warehouse, the **staging area** is the quality control station where you inspect and prepare items for production, the **object store** is the permanent inventory warehouse, and **references** are the catalog system that helps you find finished products later.

Just as a factory has standard workflows—receiving materials, quality control, assembly, packaging, and shipping—Git has standard data flows for its core operations. A simple product might visit only a few stations (like storing a single file), while a complex product (like merging two feature branches) requires coordination across all stations with multiple quality checks and decision points.

The beauty of this assembly line design is that each station has a single, well-defined responsibility, but the stations can be combined in different sequences to handle everything from simple file storage to complex multi-branch merges. Understanding these workflows is essential because they reveal how Git maintains consistency and enables powerful features like atomic commits and conflict-free parallel development.

![Git System Architecture](./diagrams/system-architecture.svg)

### Commit Creation Data Flow

The commit creation process represents Git's most fundamental workflow, transforming scattered file changes in your working directory into an immutable entry in the project's permanent history. This operation demonstrates the elegant coordination between all four major components: the working directory provides the raw changes, the staging area curates which changes to include, the object store provides permanent storage with deduplication, and the reference system updates to point to the new history state.

#### Stage 1: File Staging (git add)

The staging process begins when a developer runs the equivalent of `git add`, which moves changes from the working directory into the staging area. This operation involves complex coordination between the `WorkingDirectory`, `Index`, and `ObjectStore` components to ensure that file content is preserved exactly while metadata is captured for change detection.

The staging workflow follows these detailed steps:

1. The `WorkingDirectory` component scans the specified file path and reads the complete file content into memory, handling both text and binary files uniformly as byte streams
2. The system computes the SHA-1 hash of the file content using the blob object format: `blob {size}\0{content}`, which ensures that identical file contents always produce identical hashes regardless of filename or location
3. The `ObjectStore` checks whether an object with this hash already exists in the `.git/objects` directory, leveraging Git's content-addressable storage to avoid storing duplicate content
4. If the object doesn't exist, the system compresses the full blob object using zlib compression and stores it at the path `.git/objects/xx/yy...` where `xx` is the first two hex characters of the SHA-1 hash
5. The `Index` creates a new `IndexEntry` containing the file's complete metadata: modification times (`mtime_sec`, `mtime_nsec`), file size (`size`), file mode (`mode`), device and inode numbers for change detection, and the computed object hash
6. The new entry replaces any existing entry for the same file path in the index's sorted entry list, maintaining the index's alphabetical ordering requirement
7. The modified index is written atomically to `.git/index` using a temporary file and rename operation to ensure consistency even if the process is interrupted

This staging process creates a critical checkpoint where the file's content is permanently preserved in the object store, while the index maintains a snapshot of the working directory state at staging time.

| Component | Responsibility | Data Input | Data Output | Side Effects |
|-----------|---------------|-------------|-------------|--------------|
| `WorkingDirectory` | File content reading | File path | Raw file bytes | None (read-only) |
| `ObjectStore` | Content preservation | Blob object data | Object hash | Creates compressed object file |
| `Index` | Change tracking | File metadata + hash | Updated entry list | Modifies .git/index file |

#### Stage 2: Tree Construction (git write-tree)

Tree construction transforms the flat list of staged files into Git's hierarchical tree structure, which mirrors the directory organization while enabling efficient storage and comparison of project states. This process requires the `Index` to coordinate with the `ObjectStore` to build a nested tree structure where each directory becomes a tree object containing references to its files (as blobs) and subdirectories (as subtrees).

The tree construction algorithm proceeds recursively through the directory hierarchy:

1. The `Index` groups all staged entries by their parent directory path, creating a hierarchical structure that mirrors the working directory organization
2. For each directory level, starting from the deepest subdirectories and working upward, the system creates a tree object containing sorted entries
3. Each tree entry consists of the file mode (e.g., `100644` for regular files, `040000` for subdirectories), the filename or directory name, and the 20-byte binary SHA-1 hash of the referenced object
4. Subdirectories are processed first to obtain their tree hashes, which are then referenced by their parent directories, creating a bottom-up construction process
5. The tree entries are sorted according to Git's specific sorting rules: directories sort as if they have a trailing slash, ensuring consistent tree hashes regardless of entry insertion order
6. Each tree object is formatted as `tree {size}\0{entry1}{entry2}...{entryN}` where each entry is the binary concatenation of mode, space, name, null byte, and 20-byte hash
7. The formatted tree object is hashed, compressed, and stored in the object store using the same content-addressable storage mechanism as blob objects
8. The process continues up the directory hierarchy until a single root tree hash represents the entire project structure

This tree construction process creates an immutable snapshot of the project's directory structure, where identical directory contents always produce identical tree hashes, enabling efficient comparison and storage.

| Directory Level | Input Data | Processing | Output |
|----------------|------------|------------|--------|
| Leaf directories | File entries from index | Sort + format tree object | Tree hash |
| Intermediate directories | File entries + subtree hashes | Sort + format tree object | Tree hash |
| Root directory | All entries + subtree hashes | Sort + format tree object | Root tree hash |

#### Stage 3: Commit Creation (git commit)

Commit creation represents the culmination of the staging process, where the prepared tree object is wrapped with metadata to create a permanent, immutable entry in the project's history. This operation involves the `ObjectStore` for content storage and the `ReferenceManager` for updating the current branch pointer, creating an atomic transition from one project state to another.

The commit creation process follows these precise steps:

1. The system retrieves the root tree hash from the tree construction phase, which represents the complete project state being committed
2. The `ReferenceManager` resolves the current HEAD reference to determine the parent commit hash, handling both normal branch situations and detached HEAD states
3. The system collects commit metadata including author information (name, email, timestamp with timezone), committer information (typically identical to author), and the commit message provided by the user
4. A commit object is constructed with the format: `commit {size}\0tree {tree_hash}\nparent {parent_hash}\nauthor {author_line}\ncommitter {committer_line}\n\n{commit_message}`
5. The formatted commit object is hashed using SHA-1, compressed with zlib, and stored in the object store using the same content-addressable mechanism as other Git objects
6. The `ReferenceManager` atomically updates the current branch reference (or HEAD in detached state) to point to the new commit hash, using a temporary file and rename operation for consistency
7. If the repository tracks a reflog, the reference change is logged with the commit hash, previous hash, author information, and commit message for audit purposes

This commit creation process establishes an immutable link in the project's history chain, where the new commit references both the project state (via the tree hash) and the previous history (via the parent hash), creating Git's characteristic directed acyclic graph structure.

> The critical insight in commit creation is that the tree hash captures *what* changed while the parent hash captures *when* it changed relative to other commits. This dual referencing system enables Git to efficiently answer both "what was the project state at this point?" and "how did we get to this state?" questions.

![Commit Creation Sequence](./diagrams/commit-creation-flow.svg)

#### Data Flow Summary for Commit Creation

The complete commit creation flow demonstrates how Git's four-layer architecture enables atomic, consistent operations even when dealing with hundreds or thousands of files. Each component has a single responsibility, but they coordinate through well-defined interfaces to ensure that the working directory, index, and object store remain synchronized.

| Operation Phase | Primary Component | Secondary Components | Critical Data | Failure Recovery |
|----------------|-------------------|---------------------|---------------|------------------|
| File staging | `Index` | `WorkingDirectory`, `ObjectStore` | File content + metadata | Partial index can be rebuilt from objects |
| Tree construction | `ObjectStore` | `Index` | Directory structure + hashes | Trees can be reconstructed from index |
| Commit creation | `ReferenceManager` | `ObjectStore` | Parent hash + tree hash | Reference can be reset to previous state |
| Reference update | `ReferenceManager` | None | New commit hash | Atomic file operations prevent corruption |

### Branch Merge Data Flow

Branch merging represents Git's most complex operation, requiring coordination across all system components to combine changes from two independent lines of development while detecting and managing conflicts. Unlike the linear flow of commit creation, merge operations involve bidirectional data flow, backtracking algorithms, and conditional logic based on the discovered merge state.

#### Mental Model: The Document Reconciliation Process

Think of branch merging like two editors working independently on the same document. Each editor starts with the same original version (the merge base), makes their own changes, and then you need to create a final version that includes both sets of changes. Sometimes their edits don't overlap (easy merge), sometimes they edit different parts of the same paragraph (automatic merge), and sometimes they change the same sentence in conflicting ways (manual merge required).

Git's three-way merge algorithm automates this reconciliation process by comparing three versions: the original document (merge base), editor A's version (our branch), and editor B's version (their branch). By understanding what each editor changed relative to the original, Git can intelligently combine non-conflicting changes and flag areas where human judgment is needed.

#### Stage 1: Merge Base Discovery

Merge base discovery involves traversing the commit history graph to find the most recent common ancestor between two branches. This operation requires the `ObjectStore` to retrieve commit objects and performs a graph search algorithm to identify the point where the branches diverged, which serves as the reference point for three-way comparison.

The merge base discovery algorithm uses a breadth-first search approach with distance tracking:

1. The `MergeBaseCalculator` initializes two parallel breadth-first searches, one starting from each branch tip, maintaining separate visited sets and distance tracking for each search path
2. Each iteration expands one level of parent commits from both starting points, retrieving commit objects from the `ObjectStore` and parsing their parent references
3. The algorithm tracks the minimum distance from each starting commit to every discovered ancestor commit, enabling it to identify the lowest common ancestor when paths converge
4. When a commit is discovered from both search paths, it represents a potential merge base, but the algorithm continues until it has explored all commits at the current distance level to ensure optimality
5. Among all discovered common ancestors, the algorithm selects the one with the minimum combined distance from both branch tips, which represents the most recent common ancestor
6. Special cases are handled including: merge commits with multiple parents (both parents are added to the search queue), initial commits with no parents (search terminates at repository root), and branches with no common history (rare but possible in repositories with multiple root commits)
7. The algorithm returns the SHA-1 hash of the identified merge base commit, or signals an error if no common ancestry exists between the branches

This merge base discovery creates the foundation for three-way comparison by establishing the "original document" against which both branches' changes will be evaluated.

| Search Phase | Data Structure | Content | Purpose |
|-------------|----------------|---------|---------|
| Initialization | `visited_a: Set[str]` | Branch A's starting commit | Track explored commits from branch A |
| Initialization | `visited_b: Set[str]` | Branch B's starting commit | Track explored commits from branch B |
| Expansion | `distances_a: Dict[str, int]` | Commit hash → distance mapping | Find shortest path from branch A |
| Expansion | `distances_b: Dict[str, int]` | Commit hash → distance mapping | Find shortest path from branch B |
| Convergence | `common_ancestors: Set[str]` | Intersection of visited sets | Identify potential merge bases |
| Selection | `merge_base: str` | Optimal common ancestor hash | Reference point for three-way merge |

#### Stage 2: Three-Way File Comparison

Three-way file comparison forms the heart of Git's merge algorithm, analyzing each file that exists in any of the three versions (merge base, our branch, their branch) to determine whether changes can be automatically combined or require manual conflict resolution. This process requires the `ObjectStore` to retrieve file content and the `MyersDiff` algorithm to compute precise change locations.

The three-way comparison process analyzes each file through multiple decision paths:

1. The `ThreeWayMerge` component enumerates all unique file paths that exist in any of the three commits (merge base, ours, theirs), creating a comprehensive list of files that require analysis
2. For each file path, the system retrieves the file content from all three versions where the file exists, handling cases where the file was added, deleted, or modified in different branches
3. The Myers diff algorithm computes two separate edit scripts: one from the merge base to our version, and another from the merge base to their version, identifying exactly which lines were added, deleted, or modified
4. The system analyzes the two edit scripts to categorize the merge scenario: clean merge (non-overlapping changes), conflict (overlapping changes), or various edge cases like add/add conflicts where both branches created the same file path
5. For clean merges, the algorithm applies both sets of changes to the merge base content, creating a unified result that incorporates modifications from both branches
6. For conflicts, the system identifies the specific line ranges where changes overlap and generates conflict markers that preserve both versions for manual resolution
7. Binary files receive special handling since line-by-line merging is not applicable; binary conflicts always require manual resolution by choosing one version or the other
8. The result for each file is classified into one of several categories: cleanly merged content, conflicted content with markers, binary conflict requiring resolution, or deletion/modification conflicts

This three-way analysis produces a complete understanding of how the two branches diverged and where human intervention is required to complete the merge.

| File State | Base Version | Our Version | Their Version | Merge Result |
|-----------|--------------|-------------|---------------|--------------|
| Unchanged | Exists | Identical | Identical | Use any version |
| Modified by us only | Exists | Modified | Identical | Use our version |
| Modified by them only | Exists | Identical | Modified | Use their version |
| Modified by both (clean) | Exists | Modified | Modified | Merge both changes |
| Modified by both (conflict) | Exists | Modified | Modified | Insert conflict markers |
| Added by us only | None | Exists | None | Use our version |
| Added by them only | None | None | Exists | Use their version |
| Added by both (same content) | None | Exists | Exists | Use either version |
| Added by both (different) | None | Exists | Exists | Insert conflict markers |
| Deleted by us only | Exists | None | Identical | Delete file |
| Deleted by them only | Exists | Identical | None | Delete file |
| Deleted by both | Exists | None | None | Delete file |
| Delete/modify conflict | Exists | None | Modified | Conflict requiring resolution |

#### Stage 3: Conflict Resolution and Merge Commit Creation

When conflicts are detected during three-way comparison, the system must create a partially merged state that preserves both versions for manual resolution, then provide mechanisms for completing the merge once conflicts are resolved. This process involves the `Index`, `WorkingDirectory`, and `ReferenceManager` components working together to maintain merge state across multiple user interactions.

The conflict resolution workflow manages the complex state transitions required for interactive merging:

1. The `ThreeWayMerge` component writes conflicted files to the working directory with Git's standard conflict markers: `<<<<<<< HEAD` delimiting our changes, `=======` separating the versions, and `>>>>>>> {branch_name}` delimiting their changes
2. The `Index` is updated to record the merge state by storing index entries for all three versions of conflicted files (stage 1 for merge base, stage 2 for our version, stage 3 for their version), while cleanly merged files are stored as normal stage 0 entries
3. A `.git/MERGE_HEAD` file is created containing the SHA-1 hash of the branch being merged, which signals to other Git operations that a merge is in progress and prevents certain operations that could corrupt the merge state
4. The user manually resolves conflicts by editing the working directory files to remove conflict markers and create the desired merged content, typically using text editors or specialized merge tools
5. The user stages their resolution using the equivalent of `git add`, which removes the multi-stage index entries for each resolved file and replaces them with a single stage 0 entry containing the resolved content
6. Once all conflicts are resolved (indicated by no remaining multi-stage index entries), the user can complete the merge by creating a merge commit
7. The merge commit creation process follows the same object creation steps as regular commits, but includes two parent hashes: the current branch tip and the merged branch tip, creating Git's characteristic merge topology
8. Upon successful merge commit creation, the `.git/MERGE_HEAD` file is removed, the current branch reference is updated to point to the new merge commit, and the merge state is cleared

This conflict resolution process maintains complete safety by preserving all original content while providing clear indicators of what requires manual attention.

> The critical design insight for merge conflict handling is that Git never loses information during conflicts. The original versions from both branches are preserved in the index at different stages, the conflict markers clearly delineate the choices, and the user's resolution becomes part of the permanent history. This enables confident merging even in complex scenarios because you can always recover the original versions if needed.

#### Stage 4: Merge State Management

Managing merge state involves tracking the progress of a multi-step merge operation across multiple user interactions, ensuring that partial merge states are preserved consistently and that the system can detect incomplete merges to prevent data loss. This requires coordination between the `Index`, `ReferenceManager`, and file system to maintain merge metadata.

The merge state management system handles several complex scenarios:

1. **In-progress merge detection**: The presence of `.git/MERGE_HEAD` signals that a merge is in progress, causing status commands to display merge information and preventing operations like starting new merges or switching branches that could lose merge state
2. **Multi-stage index management**: Conflicted files are stored with stage numbers (1 = merge base, 2 = ours, 3 = theirs) in the index, while resolved files use stage 0, enabling precise tracking of which conflicts remain unresolved
3. **Partial resolution handling**: Users can resolve conflicts incrementally, staging some files while leaving others conflicted, with the system maintaining accurate state for each file independently
4. **Merge abort capability**: The system can restore the pre-merge state by resetting the working directory and index to the original branch tip and removing merge metadata files
5. **Merge commit validation**: Before allowing merge commit creation, the system verifies that no multi-stage index entries remain, ensuring that all conflicts have been explicitly resolved
6. **Reference safety**: During merge operations, the current branch reference is not updated until the merge commit is successfully created, maintaining a consistent rollback point
7. **Working directory synchronization**: The system ensures that working directory file contents match the user's intended resolutions and that no uncommitted changes would be lost during merge completion

This comprehensive state management enables safe, incremental conflict resolution while maintaining system consistency throughout the merge process.

| Merge State | Index Entries | Working Directory | Metadata Files | User Actions Available |
|-------------|---------------|-------------------|----------------|------------------------|
| Clean merge | Stage 0 only | Merged content | None | Commit merge immediately |
| Conflicts present | Multi-stage entries | Conflict markers | `.git/MERGE_HEAD` | Resolve conflicts, stage files |
| Partially resolved | Mixed stage entries | Some resolved, some conflicted | `.git/MERGE_HEAD` | Continue resolving, stage files |
| Fully resolved | Stage 0 only | All resolved | `.git/MERGE_HEAD` | Commit merge |
| Aborted merge | Original state | Original state | None | Normal operations |

### Common Pitfalls in Component Interactions

Understanding the complex data flows in Git operations helps identify common mistakes that can lead to inconsistent repository states or data loss. These pitfalls often occur at the boundaries between components where assumptions about data consistency or operation atomicity may not hold.

⚠️ **Pitfall: Non-atomic Index Updates**

Many implementers update the index entry by entry during staging operations, which can leave the repository in an inconsistent state if the process is interrupted. For example, partially updating the index during a large `git add` operation could result in some files being staged while others remain in an unknown state. The index file could become corrupted if the process crashes while writing, since the checksum at the end of the file would not match the partial content. Always use atomic writes with temporary files and rename operations, and ensure that the index checksum is computed over the complete final state before writing.

⚠️ **Pitfall: Incorrect Object Hash Computation**

A common mistake is computing object hashes over just the content bytes rather than the complete object format including the type header and size. This leads to hash mismatches where your implementation generates different hashes than real Git for identical content. The hash must be computed over the complete format: `{type} {size}\0{content}` where the null byte is critical and often forgotten. Additionally, ensure that the size is the byte count of the raw content, not the formatted object, and that binary content is handled without text encoding conversions.

⚠️ **Pitfall: Merge Base Calculation Errors**

Merge base discovery can fail in repositories with complex branching topologies, particularly when dealing with merge commits that have multiple parents or when branches have been created and deleted repeatedly. A common error is implementing a simple "first common ancestor" algorithm rather than finding the *lowest* common ancestor, which can result in merge bases that are too far back in history. This leads to unnecessarily complex merges with spurious conflicts. Always implement breadth-first search with proper distance tracking, and handle edge cases like octopus merges and orphan branches.

⚠️ **Pitfall: Conflict Marker Generation**

When generating conflict markers during merge operations, many implementations fail to properly escape or handle edge cases in the conflict content itself. If the conflicted content already contains lines that look like conflict markers (e.g., lines starting with `<<<<<<<`), the generated markers can become ambiguous or confusing. Additionally, failing to include proper branch names or commit identifiers in the conflict markers makes it difficult for users to understand which version is which. Always validate that generated markers are unambiguous and include sufficient context for resolution.

⚠️ **Pitfall: Reference Update Race Conditions**

In systems where multiple processes might access the repository simultaneously, updating references without proper locking can lead to lost commits or corrupted branch states. For example, if two processes attempt to commit to the same branch simultaneously, one commit might be lost if both read the same parent hash but only the last write succeeds. While single-user scenarios are common during learning, understanding the race conditions helps build more robust implementations. Use file locking or atomic operations for reference updates, and consider implementing basic conflict detection for concurrent access.

### Implementation Guidance

This section provides concrete implementation support for building the component interactions and data flows described above. The code focuses on orchestrating the individual components built in previous milestones into cohesive, complex operations.

#### Technology Recommendations

| Component | Simple Option | Advanced Option | Notes |
|-----------|---------------|-----------------|-------|
| Flow orchestration | Simple function calls | Command pattern with undo | Commands enable better error recovery |
| State management | Direct attribute access | State machine pattern | State machines prevent invalid transitions |
| Error handling | Exception bubbling | Result/Option types | Explicit error handling improves debugging |
| Concurrency | Sequential operations | Lock-free algorithms | Start simple, optimize later if needed |
| Progress tracking | Print statements | Observer pattern with events | Useful for long-running merge operations |

#### Recommended File Structure

The component interaction code should coordinate the individual components built in previous milestones:

```
src/
  git/
    commands/
      add.py              ← Implements staging workflow
      commit.py           ← Implements commit creation workflow  
      merge.py            ← Implements merge workflow
      status.py           ← Implements status calculation workflow
    workflows/
      commit_workflow.py  ← Complete commit creation orchestration
      merge_workflow.py   ← Complete merge orchestration
    core/                 ← Individual components from previous milestones
      object_store.py     ← From Milestone 2-4
      index.py           ← From Milestone 6
      references.py      ← From Milestone 5
      diff.py           ← From Milestone 7
      merge.py          ← From Milestone 8
```

#### Infrastructure Starter Code

Complete workflow orchestration helpers that coordinate the individual components:

```python
"""
Workflow orchestration utilities for complex Git operations.
Provides high-level coordination between individual components.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

class WorkflowError(Exception):
    """Base exception for workflow orchestration errors."""
    pass

class WorkflowStatus(Enum):
    """Status of a multi-step workflow operation."""
    SUCCESS = "success"
    PARTIAL = "partial"
    CONFLICT = "conflict"
    ERROR = "error"

@dataclass
class WorkflowResult:
    """Result of a workflow operation with detailed status information."""
    status: WorkflowStatus
    message: str
    files_changed: int = 0
    conflicts: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.conflicts is None:
            self.conflicts = []
        if self.warnings is None:
            self.warnings = []

class ProgressReporter:
    """Simple progress reporting for long-running operations."""
    
    def __init__(self, total_items: int = 0):
        self.total_items = total_items
        self.completed_items = 0
        self.current_operation = ""
    
    def start_operation(self, operation: str, total: int = 0):
        """Start a new operation with optional total item count."""
        self.current_operation = operation
        self.completed_items = 0
        if total > 0:
            self.total_items = total
        print(f"Starting: {operation}")
    
    def update_progress(self, items_completed: int = 1, message: str = ""):
        """Update progress by specified number of items."""
        self.completed_items += items_completed
        if self.total_items > 0:
            percentage = (self.completed_items / self.total_items) * 100
            print(f"Progress: {percentage:.1f}% ({self.completed_items}/{self.total_items}) {message}")
        else:
            print(f"Progress: {self.completed_items} items completed {message}")
    
    def finish_operation(self, success: bool = True):
        """Mark current operation as finished."""
        status = "completed" if success else "failed"
        print(f"Finished: {self.current_operation} - {status}")

def validate_repository_state(git_dir: Path) -> List[str]:
    """
    Validate that repository is in a consistent state for operations.
    Returns list of validation errors, empty list if valid.
    """
    errors = []
    
    # Check basic repository structure
    if not git_dir.exists():
        errors.append("Git directory does not exist")
        return errors
    
    objects_dir = git_dir / "objects"
    if not objects_dir.exists():
        errors.append("Objects directory missing")
    
    refs_dir = git_dir / "refs" / "heads"
    if not refs_dir.exists():
        errors.append("References directory missing")
    
    head_file = git_dir / "HEAD"
    if not head_file.exists():
        errors.append("HEAD reference missing")
    
    # Check for corrupted index
    index_file = git_dir / "index"
    if index_file.exists():
        try:
            # Attempt to read index header to validate format
            with open(index_file, 'rb') as f:
                signature = f.read(4)
                if signature != b'DIRC':
                    errors.append("Index file corrupted - invalid signature")
        except Exception as e:
            errors.append(f"Index file unreadable: {e}")
    
    return errors

def atomic_workflow_operation(operation_name: str, operation_func, cleanup_func=None):
    """
    Decorator for atomic workflow operations with automatic cleanup on failure.
    If operation fails, cleanup_func is called to restore previous state.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                print(f"Starting atomic operation: {operation_name}")
                result = operation_func(*args, **kwargs)
                print(f"Atomic operation completed: {operation_name}")
                return result
            except Exception as e:
                print(f"Atomic operation failed: {operation_name} - {e}")
                if cleanup_func:
                    try:
                        cleanup_func(*args, **kwargs)
                        print(f"Cleanup completed for: {operation_name}")
                    except Exception as cleanup_error:
                        print(f"Cleanup failed for {operation_name}: {cleanup_error}")
                raise
        return wrapper
    return decorator
```

#### Commit Creation Workflow Implementation

Complete orchestration for the commit creation process:

```python
"""
Commit creation workflow implementation.
Orchestrates staging, tree building, and commit creation.
"""
from pathlib import Path
from typing import List, Optional, Set
from .workflows.base import WorkflowResult, WorkflowStatus, ProgressReporter

class CommitWorkflow:
    """Orchestrates the complete commit creation process."""
    
    def __init__(self, repository, object_store, index, references):
        self.repository = repository
        self.object_store = object_store
        self.index = index
        self.references = references
        self.progress = ProgressReporter()
    
    def stage_files(self, file_paths: List[Path]) -> WorkflowResult:
        """
        Stage multiple files for commit.
        Coordinates WorkingDirectory -> ObjectStore -> Index flow.
        """
        # TODO 1: Validate that all file paths exist in working directory
        # TODO 2: For each file, read content and compute blob hash
        # TODO 3: Store blob object in object store (with deduplication check)
        # TODO 4: Create IndexEntry with file metadata and object hash
        # TODO 5: Add IndexEntry to index, replacing any existing entry
        # TODO 6: Save updated index atomically to .git/index
        # Hint: Use ProgressReporter to show staging progress for large numbers of files
        # Hint: Collect any files that couldn't be staged and include in warnings
        pass
    
    def build_tree_from_index(self) -> str:
        """
        Build tree object hierarchy from current index state.
        Returns root tree hash representing complete project state.
        """
        # TODO 1: Group index entries by directory path (recursive grouping)
        # TODO 2: Start with deepest subdirectories, build tree objects bottom-up
        # TODO 3: For each directory level, create sorted list of tree entries
        # TODO 4: Format tree entries as: mode + " " + name + "\0" + 20-byte hash
        # TODO 5: Create tree object with format: "tree {size}\0{entries}"
        # TODO 6: Store tree object in object store and get its hash
        # TODO 7: Continue up directory hierarchy until root tree is built
        # Hint: Directories sort as if they have trailing "/" for Git compatibility
        # Hint: Tree entries must be sorted for deterministic tree hashes
        pass
    
    def create_commit(self, tree_hash: str, message: str, parent_hashes: List[str] = None) -> str:
        """
        Create commit object linking tree to history.
        Returns commit hash for the new commit.
        """
        # TODO 1: Get current HEAD reference to determine parent commit
        # TODO 2: Format author and committer lines with timestamp and timezone
        # TODO 3: Build commit content: tree line, parent lines, author, committer, blank line, message
        # TODO 4: Create commit object with format: "commit {size}\0{content}"
        # TODO 5: Store commit object in object store and get its hash
        # TODO 6: Update current branch reference to point to new commit
        # TODO 7: Clear any merge state files if this completes a merge
        # Hint: Handle both normal commits (1 parent) and merge commits (2+ parents)
        # Hint: Use atomic reference update to prevent corruption
        pass
    
    def execute_commit(self, message: str, stage_all: bool = False) -> WorkflowResult:
        """
        Execute complete commit workflow: stage files, build tree, create commit.
        This is the high-level orchestration function.
        """
        self.progress.start_operation("Creating commit")
        
        try:
            # TODO 1: If stage_all=True, stage all modified files in working directory
            # TODO 2: Validate that index contains at least one staged change
            # TODO 3: Build tree object from current index state
            # TODO 4: Create commit object with tree hash and commit message
            # TODO 5: Update HEAD reference to point to new commit
            # TODO 6: Return WorkflowResult with success status and commit details
            # Hint: Use progress.update_progress() to show workflow steps
            # Hint: Return appropriate error status if any step fails
            # Hint: Include commit hash and files changed count in result
            pass
        except Exception as e:
            self.progress.finish_operation(success=False)
            return WorkflowResult(
                status=WorkflowStatus.ERROR,
                message=f"Commit failed: {e}"
            )
```

#### Merge Workflow Implementation

Complete orchestration for the merge process:

```python
"""
Merge workflow implementation.
Orchestrates merge base discovery, three-way merge, and conflict resolution.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .workflows.base import WorkflowResult, WorkflowStatus, ProgressReporter

class MergeWorkflow:
    """Orchestrates the complete branch merge process."""
    
    def __init__(self, repository, object_store, index, references, merge_algorithm):
        self.repository = repository
        self.object_store = object_store
        self.index = index
        self.references = references
        self.merge_algorithm = merge_algorithm
        self.progress = ProgressReporter()
    
    def discover_merge_base(self, our_commit: str, their_commit: str) -> Optional[str]:
        """
        Find lowest common ancestor between two commits.
        Uses breadth-first search with distance tracking.
        """
        # TODO 1: Initialize BFS queues for both commits with distance 0
        # TODO 2: Track visited commits and distances from each starting point  
        # TODO 3: Expand one level at a time, adding parent commits to queues
        # TODO 4: When commit appears in both visited sets, it's a common ancestor
        # TODO 5: Continue until all commits at current distance level are processed
        # TODO 6: Return common ancestor with minimum combined distance
        # TODO 7: Handle edge cases: no common ancestor, identical commits
        # Hint: Use object_store.retrieve_object() to get commit objects
        # Hint: Parse commit objects to extract parent hashes
        # Hint: Return None if branches have no common history
        pass
    
    def analyze_merge_conflicts(self, base_commit: str, our_commit: str, their_commit: str) -> Dict[str, str]:
        """
        Analyze all files in three commits to categorize merge requirements.
        Returns dict mapping file paths to merge status.
        """
        # TODO 1: Get tree objects for all three commits
        # TODO 2: Extract all unique file paths from all three trees
        # TODO 3: For each file path, determine what happened in each branch
        # TODO 4: Categorize each file: clean merge, conflict, add/add, delete/modify, etc.
        # TODO 5: Return mapping of file_path -> merge_status
        # Hint: Use tree traversal to find all files in each commit
        # Hint: Handle cases where file exists in some commits but not others
        # Hint: Status values: "clean", "conflict", "add_add", "delete_modify", etc.
        pass
    
    def execute_three_way_merge(self, base_commit: str, our_commit: str, their_commit: str) -> WorkflowResult:
        """
        Perform three-way merge of two branches.
        Handles both clean merges and conflicts.
        """
        self.progress.start_operation("Merging branches")
        
        try:
            # TODO 1: Analyze all files to categorize merge requirements
            # TODO 2: For each file, perform appropriate merge operation
            # TODO 3: Write cleanly merged files to working directory
            # TODO 4: Write conflicted files with conflict markers to working directory
            # TODO 5: Update index with appropriate stage entries (0 for clean, 1/2/3 for conflicts)
            # TODO 6: Create .git/MERGE_HEAD file with their commit hash
            # TODO 7: Return result indicating clean merge or conflicts requiring resolution
            # Hint: Use progress reporting for large merges
            # Hint: Collect conflict file paths for result reporting
            # Hint: Handle binary files (always conflict, no line-level merge)
            pass
        except Exception as e:
            self.progress.finish_operation(success=False)
            return WorkflowResult(
                status=WorkflowStatus.ERROR,
                message=f"Merge failed: {e}"
            )
    
    def resolve_conflicts_and_commit(self, message: str = None) -> WorkflowResult:
        """
        Complete merge after manual conflict resolution.
        Creates merge commit with two parents.
        """
        # TODO 1: Verify that no multi-stage index entries remain (all conflicts resolved)
        # TODO 2: Build tree from resolved index state
        # TODO 3: Get both parent commits: current HEAD and MERGE_HEAD
        # TODO 4: Create merge commit with both parent hashes
        # TODO 5: Update HEAD reference to new merge commit
        # TODO 6: Remove .git/MERGE_HEAD file to clear merge state
        # TODO 7: Return success result with merge commit details
        # Hint: Generate default merge message if none provided
        # Hint: Validate that working directory matches index (no unstaged changes)
        # Hint: Merge commits have exactly two parent lines in commit object
        pass
```

#### Milestone Checkpoints

After implementing the component interactions, verify the complete workflows:

**Commit Creation Verification:**
```bash
# Initialize test repository
python -m git.init test_repo
cd test_repo

# Create and stage multiple files
echo "Content A" > file_a.txt
echo "Content B" > file_b.txt
mkdir subdir
echo "Content C" > subdir/file_c.txt

python -m git.add file_a.txt file_b.txt subdir/file_c.txt
python -m git.commit "Initial commit with multiple files"

# Verify objects were created correctly
python -m git.log --oneline  # Should show commit
python -m git.ls-tree HEAD   # Should show tree structure
```

**Expected output:** Commit creation should produce consistent tree and commit hashes, with working directory, index, and object store all synchronized.

**Merge Workflow Verification:**
```bash
# Create two branches with diverging changes
python -m git.checkout -b feature
echo "Feature change" >> file_a.txt
python -m git.add file_a.txt
python -m git.commit "Feature commit"

python -m git.checkout main
echo "Main change" >> file_b.txt  
python -m git.add file_b.txt
python -m git.commit "Main commit"

# Attempt merge - should be clean
python -m git.merge feature

# Verify merge commit was created
python -m git.log --graph --oneline  # Should show merge topology
```

**Expected behavior:** Clean merge should complete automatically, conflicting merge should leave conflict markers and require manual resolution.


## Error Handling and Edge Cases

> **Milestone(s):** This section applies to all eight milestones but is particularly critical for Milestone 6 (Index/Staging Area), Milestone 7 (Diff Algorithm), and Milestone 8 (Three-Way Merge). Robust error handling becomes essential as operations grow more complex and involve multiple components interacting.

### Mental Model: The Digital Safety Net

Think of error handling in a version control system like the safety systems in a modern airplane. Just as aircraft have multiple redundant systems, automatic failure detection, and clear emergency procedures that pilots practice extensively, a robust Git implementation needs layered protection against data corruption, clear detection of problems, and well-defined recovery workflows that users can follow confidently.

When turbulence hits an aircraft, the autopilot doesn't just crash—it has protocols for every conceivable failure mode. Similarly, when your Git implementation encounters corrupted objects, interrupted merges, or concurrent access conflicts, it should degrade gracefully, preserve data integrity above all else, and provide clear guidance for recovery.

The critical insight is that version control systems are **data custodians**—they hold irreplaceable project history. Like a bank vault, the system must be paranoid about data integrity and conservative about operations that could cause loss. This means validating everything, assuming hardware can fail at any moment, and always providing a path back to a known good state.

### Repository Corruption Handling

Repository corruption represents the most serious class of failures in version control systems. Unlike application crashes that lose only current work, corruption can destroy historical data that may be impossible to recreate. Our error handling strategy must prioritize **early detection**, **damage isolation**, and **graceful recovery** while maintaining data integrity above all other concerns.

#### Corruption Detection Strategies

The foundation of corruption handling lies in comprehensive validation that occurs at multiple layers throughout the system. Rather than trusting that data remains intact, we implement verification at every access point to catch corruption as early as possible.

**Object Integrity Verification** forms the first line of defense. Every time we retrieve an object from the content-addressable store, we must verify that its content still produces the expected SHA-1 hash. This catches both storage corruption and programming bugs that might corrupt objects in memory.

| Validation Point | Check Performed | Frequency | Action on Failure |
|------------------|----------------|-----------|-------------------|
| Object Retrieval | SHA-1 verification | Every access | Return corruption error |
| Object Storage | Hash before/after compression | Every write | Abort operation |
| Index Loading | Checksum verification | On index read | Rebuild from working tree |
| Reference Resolution | SHA-1 format validation | Every resolution | Report invalid reference |
| Tree Traversal | Entry format validation | During traversal | Stop at corrupted tree |
| Commit Parsing | Header format validation | During log operations | Mark commit as corrupted |

**Repository Structure Validation** ensures that the fundamental directory structure and required files remain intact. This validation should occur during repository initialization and can be triggered explicitly by user commands.

The validation algorithm proceeds systematically through repository components:

1. Verify that the git directory exists and has appropriate permissions (0755)
2. Check that required subdirectories exist: objects, refs, refs/heads, refs/tags
3. Validate that the HEAD file exists and contains either a valid symbolic reference or commit hash
4. Scan the objects directory to ensure the two-character subdirectory structure is intact
5. Verify that no objects have been corrupted by spot-checking a sample of stored objects
6. Validate reference files to ensure they contain properly formatted commit hashes or symbolic references
7. Check the index file format and checksum if present

**Cross-Reference Consistency** validates that references between objects remain valid. A commit might reference a tree that no longer exists, or a tree might reference a blob that has been corrupted. These consistency checks require traversing object relationships and validating each link.

#### Corruption Recovery Strategies

Recovery from corruption depends heavily on the extent and location of the damage. Our recovery strategy follows a **progressive escalation** approach, starting with minimal intervention and escalating to more drastic measures only when necessary.

> **Decision: Corruption Recovery Hierarchy**
> - **Context**: When corruption is detected, we need a systematic approach to recovery that minimizes data loss while restoring repository functionality
> - **Options Considered**: 
>   - Immediate full repository rebuild from working directory
>   - Attempt to repair specific corrupted components
>   - Progressive recovery starting with least invasive repairs
> - **Decision**: Implement progressive recovery hierarchy starting with object-level repairs
> - **Rationale**: Maximizes data preservation while providing clear escalation path; users retain control over recovery process
> - **Consequences**: More complex recovery logic but better preservation of historical data and user confidence

**Level 1: Object-Level Recovery** addresses corruption in individual objects while preserving the broader repository structure. When object retrieval fails due to hash mismatch or decompression errors, the system should attempt to recover the object from alternative sources.

| Recovery Method | Source | Applicability | Success Rate |
|----------------|--------|---------------|--------------|
| Re-read from disk | Same object file | Transient I/O errors | High |
| Rebuild from working tree | Current file content | Blob objects only | Medium |
| Reconstruct from index | Staged content | Recently staged files | Medium |
| Import from backup | External copy | Any object type | Variable |

**Level 2: Reference Recovery** handles corruption in the reference system, including damaged HEAD files, missing branch references, or invalid symbolic references. Reference recovery is generally safer than object recovery since references can be reconstructed from known commit hashes.

The reference recovery process involves:

1. Scan the objects directory to identify all available commit objects
2. Parse each commit to extract author, timestamp, and message information
3. Present a list of recent commits to the user for branch recreation
4. Allow the user to select which commits should become branch heads
5. Recreate the branch references pointing to the selected commits
6. Reset HEAD to point to the user's preferred default branch

**Level 3: Index Reconstruction** rebuilds the staging area when the index file becomes corrupted or inconsistent. Since the index represents only the current staging state, it can be safely reconstructed from the working directory without losing historical data.

**Level 4: Repository Rebuild** represents the most drastic recovery option, essentially reinitializing the repository while preserving as much history as possible. This approach salvages individual objects and reconstructs the repository structure around them.

#### Validation Error Types and Responses

Different types of corruption require specialized detection and response strategies. Our implementation must recognize each category and apply appropriate recovery measures.

| Error Type | Detection Method | Immediate Response | Recovery Strategy |
|------------|-----------------|-------------------|------------------|
| SHA-1 Mismatch | Hash verification | Block object access | Re-read or rebuild object |
| Compression Failure | Zlib decompression | Return read error | Attempt raw file recovery |
| Malformed Object | Header parsing | Parsing exception | Reconstruct from metadata |
| Missing Object | File system access | File not found | Search for alternatives |
| Invalid Reference | Reference resolution | Format validation | Reset to valid commit |
| Corrupt Index | Checksum verification | Index load failure | Rebuild from working tree |
| Permission Errors | File system operations | Access denied | Report and suggest fixes |

⚠️ **Pitfall: Silent Corruption**
Many corruption scenarios don't immediately cause obvious failures. A single bit flip in an object file might not be detected until that specific object is accessed, potentially weeks later. Always implement comprehensive validation rather than assuming storage is reliable. Check SHA-1 hashes on every object read, not just during explicit verification commands.

### Merge Conflict Resolution

Merge conflicts represent a normal part of collaborative development, but the complexity of conflict resolution can overwhelm users if not handled gracefully. Our conflict resolution system must provide **clear conflict presentation**, **intuitive resolution workflows**, and **safe mechanisms** for completing interrupted merges.

#### Conflict Detection and Classification

Effective conflict resolution begins with precise conflict detection that categorizes conflicts by type and complexity. This classification helps users understand what they're dealing with and choose appropriate resolution strategies.

**Content Conflicts** occur when both branches modify the same lines of a file. These represent the classic merge conflict scenario where automated merging cannot determine which changes should take precedence.

The conflict detection algorithm compares changes from both branches against their common ancestor:

1. Identify all lines that were modified in the "ours" branch relative to the merge base
2. Identify all lines that were modified in the "theirs" branch relative to the merge base
3. Find overlapping regions where both branches modified the same line numbers
4. For each overlap, determine if the changes are identical (auto-resolve) or conflicting (require manual resolution)
5. Generate conflict markers for regions that require manual resolution

**Structural Conflicts** arise from changes to file organization rather than content. These conflicts require different resolution strategies than simple content conflicts.

| Conflict Type | Scenario | Detection Method | Resolution Options |
|---------------|----------|------------------|-------------------|
| Add/Add | Both branches add file with same name | Path collision during merge | Keep one, keep both with rename, manual merge |
| Delete/Modify | One branch deletes, other modifies | Missing file during content merge | Keep modification, confirm deletion |
| Mode Change | Different permission changes | File mode comparison | Choose one mode or manual decision |
| Rename/Rename | Both branches rename same file | Path tracking during merge | Choose one name or create new name |
| Directory/File | Directory becomes file or vice versa | Path type validation | Manual resolution required |

**Binary File Conflicts** require special handling since line-based merging doesn't apply to binary content. The system must detect binary files and present appropriate resolution options.

#### Conflict Marker Generation

When conflicts cannot be resolved automatically, the system must generate clear, standardized conflict markers that help users understand what happened and how to proceed. The conflict marker format follows Git's established conventions to maintain familiarity for users.

The standard conflict marker structure includes:

1. Opening marker (`<<<<<<<`) followed by branch identifier for "ours" changes
2. The conflicting content from the "ours" branch
3. Separator marker (`=======`) dividing the conflicting versions
4. The conflicting content from the "theirs" branch  
5. Closing marker (`>>>>>>>`) followed by branch identifier for "theirs" changes

For three-way conflicts where the merge base content differs from both branches, an additional marker (`|||||||`) introduces the original content between the ours section and the separator.

> The key insight for effective conflict markers is that they must tell a story. Users need to understand not just what the conflicting content is, but where it came from, why it conflicts, and what their options are for resolution. Clear branch identifiers and optional base content provide this context.

#### Conflict Resolution Workflows

Successful conflict resolution requires well-defined workflows that guide users through the resolution process without overwhelming them. The workflow design must accommodate both novice and experienced users while maintaining safety throughout the process.

**Interactive Resolution Workflow** provides step-by-step guidance for users who prefer structured assistance:

1. Present a summary of all conflicts found, categorized by type and file
2. For each conflicted file, display the conflict in context with surrounding unchanged lines
3. Offer resolution options: edit manually, choose ours, choose theirs, or skip for now
4. After each resolution, validate that conflict markers have been properly removed
5. Allow users to test their resolution by running builds or tests before finalizing
6. Provide a final review showing all resolved conflicts before completing the merge

**Batch Resolution Workflow** supports experienced users who want to resolve multiple conflicts efficiently:

1. Generate a conflict summary report showing all conflicts and their types
2. Allow pattern-based resolution for similar conflicts (e.g., "choose ours for all .config files")
3. Provide bulk editing capabilities for systematic conflict resolution
4. Validate that all conflicts have been addressed before allowing merge completion

**Tool Integration Workflow** supports external merge tools by providing standardized interfaces:

1. Generate temporary files containing base, ours, and theirs versions for each conflict
2. Launch the configured merge tool with appropriate file arguments
3. Wait for the tool to complete and validate the merged result
4. Import the resolved content back into the working directory
5. Continue with the next conflict or complete the merge process

#### Interrupted Merge Recovery

Merge operations can be interrupted by system crashes, user cancellation, or external factors. The system must maintain enough state information to allow users to resume or abort interrupted merges safely.

**Merge State Tracking** records the current merge operation's progress and context in the git directory. This state information enables recovery after interruption.

| State File | Content | Purpose |
|------------|---------|---------|
| MERGE_HEAD | Target commit hash | Identifies what we're merging |
| MERGE_MODE | Merge type and options | Records merge algorithm settings |
| MERGE_MSG | Proposed commit message | Preserves user's merge message |
| MERGE_PROGRESS | Resolved/remaining files | Tracks completion status |

**Recovery Options** provide users with clear paths forward when they encounter an interrupted merge:

1. **Continue Merge**: Resume the merge process after resolving remaining conflicts
2. **Abort Merge**: Return to the pre-merge state, discarding all merge progress
3. **Reset Merge**: Restart the merge from the beginning with fresh conflict resolution

The continue workflow validates that all conflicts have been resolved before proceeding:

1. Scan all files mentioned in MERGE_PROGRESS for remaining conflict markers
2. Verify that the working directory contains no unexpected changes
3. Build the final merge tree from resolved content
4. Create the merge commit with both parent commits recorded
5. Update the current branch reference and clean up merge state files

The abort workflow ensures complete restoration to the pre-merge state:

1. Read the original HEAD commit from the merge state files
2. Reset the working directory to match the original commit tree
3. Clear the index of any merge-related staged content
4. Remove all merge state files to indicate normal (non-merge) state
5. Display a summary of what was discarded for user confirmation

⚠️ **Pitfall: Partial Conflict Resolution**
Users often resolve some conflicts and then forget to complete the merge, leaving the repository in an intermediate state. Always check for incomplete merges during status operations and provide clear guidance. Store enough state to distinguish between "merge in progress with remaining conflicts" and "merge ready to complete" scenarios.

### Concurrent Access Patterns

Modern development workflows often involve multiple processes accessing the same repository simultaneously. Build systems, IDE integrations, backup tools, and multiple Git commands can all attempt repository operations concurrently. Our implementation must handle these scenarios gracefully without corrupting data or creating inconsistent states.

#### File System Level Concurrency

The fundamental challenge of concurrent access lies in the shared file system that underlies Git's storage model. Multiple processes writing to the same files or directories can create race conditions, partial updates, and corruption scenarios that are difficult to diagnose and recover from.

**Atomic Operations Strategy** ensures that file operations either complete entirely or fail entirely, preventing partial updates that could leave the repository in an inconsistent state.

| Operation Type | Atomicity Mechanism | Failure Handling |
|----------------|-------------------|------------------|
| Object Storage | Write to temp file, rename | Remove temp file on failure |
| Reference Updates | Write to temp file, rename | Restore original reference |
| Index Updates | Write complete index to temp, rename | Restore backup index |
| Branch Creation | Atomic file creation with O_EXCL | Report if branch exists |
| Lock Acquisition | Create lock file with O_EXCL | Wait or fail immediately |

The atomic write pattern forms the foundation of safe concurrent access:

1. Generate a unique temporary filename in the same directory as the target file
2. Write the complete content to the temporary file
3. Sync the temporary file to ensure data reaches persistent storage
4. Atomically rename the temporary file to the final filename
5. If any step fails, remove the temporary file and report the error

**Locking Protocols** prevent multiple processes from modifying the same repository components simultaneously. Our locking strategy balances safety with performance, using fine-grained locks where possible to minimize contention.

The repository uses several categories of locks:

- **Reference Locks**: Protect individual branch and tag references during updates
- **Index Lock**: Prevents concurrent modification of the staging area
- **Object Store Locks**: Protect against concurrent writes to the same object (rare but possible)
- **Merge State Locks**: Prevent multiple merge operations from running simultaneously

Lock acquisition follows a consistent protocol to avoid deadlocks:

1. Determine all required locks for the operation in a predetermined order
2. Attempt to acquire each lock in sequence with appropriate timeouts
3. If any lock acquisition fails, release all previously acquired locks
4. Either retry with backoff or report the failure to the user
5. Upon successful completion, release all locks in reverse order

**Lock File Implementation** uses the file system's atomic file creation semantics to implement advisory locks. This approach works across different operating systems and doesn't require specialized locking primitives.

Lock file creation process:

1. Generate lock filename by appending `.lock` to the protected resource name
2. Attempt to create the lock file with exclusive access (O_CREAT | O_EXCL)
3. If creation succeeds, write lock metadata (process ID, operation type, timestamp)
4. If creation fails because the file exists, another process holds the lock
5. To release the lock, simply remove the lock file

#### Process Coordination Patterns

Beyond basic file locking, certain operations require coordination between processes to ensure consistent behavior and avoid conflicts that locks alone cannot prevent.

**Read-Write Coordination** allows multiple readers to access repository data simultaneously while ensuring that write operations have exclusive access when needed. This pattern is particularly important for long-running operations like large merges or history traversals.

| Access Pattern | Lock Type | Concurrency Level | Use Cases |
|----------------|-----------|------------------|-----------|
| Multiple Readers | Shared read access | High concurrency | Status, log, diff operations |
| Single Writer | Exclusive write access | No concurrency | Commit, merge, rebase operations |
| Reader-Writer | Upgrade from read to write | Medium concurrency | Add operations that become commits |

**Operation Serialization** ensures that certain operations complete in a consistent order even when initiated simultaneously. This is particularly important for operations that depend on the current repository state.

The serialization mechanism works through operation queuing:

1. Operations that require serialization acquire a queue position lock
2. Each operation writes its intent and dependencies to a coordination file
3. Operations check their dependencies before proceeding with actual work
4. Upon completion, operations signal their completion and wake waiting operations
5. The coordination file is cleaned up when no operations remain pending

**Graceful Degradation Strategies** allow the system to continue operating even when some concurrent access patterns fail. Rather than failing completely, the system falls back to safer but potentially slower operation modes.

| Failure Scenario | Degradation Strategy | Performance Impact |
|------------------|--------------------|--------------------|
| Lock timeout | Prompt user to retry or wait | User intervention required |
| File contention | Retry with exponential backoff | Increased operation latency |
| Coordination failure | Fall back to exclusive locking | Reduced concurrency |
| Resource exhaustion | Queue operations sequentially | Serialized execution |

#### Error Recovery in Concurrent Scenarios

Concurrent access introduces additional failure modes that don't occur in single-process scenarios. These failures often involve partial state updates, timing-dependent races, and complex interactions between multiple operations.

**Orphaned Lock Detection** identifies and recovers from situations where locks are left behind by crashed or killed processes. Without proper cleanup, these orphaned locks can permanently block repository access.

The orphaned lock detection algorithm:

1. When lock acquisition fails, examine the existing lock file contents
2. Extract the process ID and timestamp from the lock metadata
3. Check if the process ID still exists and is running the expected operation
4. If the process is gone or running a different operation, consider the lock orphaned
5. Verify that sufficient time has passed since lock creation to avoid false positives
6. Remove the orphaned lock and proceed with the operation

**Partial Operation Recovery** handles scenarios where an operation begins but cannot complete due to concurrent access conflicts. The system must be able to detect these partial states and either complete the operation or roll back to a consistent state.

Partial operation scenarios include:

- Reference updates that write the new value but fail to remove backup files
- Index updates that complete partially before lock contention forces abandonment
- Object store operations that create objects but fail to update references
- Merge operations that resolve some conflicts but encounter locking issues

The recovery strategy involves state validation and corrective action:

1. During repository initialization, scan for signs of incomplete operations
2. For each incomplete operation type, determine if completion is safe
3. If completion is safe and beneficial, finish the operation automatically
4. If completion is risky or impossible, roll back to the previous consistent state
5. Log the recovery action for user awareness and debugging

**Deadlock Prevention** ensures that multiple processes cannot create circular waiting conditions that would permanently block progress. While file-based locking reduces deadlock risk compared to more complex locking primitives, careful lock ordering prevents the remaining deadlock scenarios.

The deadlock prevention protocol establishes a global ordering for all repository locks:

1. Object store locks (ordered by object hash)
2. Index lock
3. Reference locks (ordered alphabetically by reference name)
4. Merge state locks

All operations must acquire locks in this predetermined order and release them in reverse order. Operations that cannot acquire all needed locks within the timeout period must release all locks and retry to avoid deadlock scenarios.

⚠️ **Pitfall: Lock File Cleanup**
Lock files can accumulate over time if processes crash or are killed forcefully. Always implement cleanup logic that runs during normal operations to detect and remove orphaned locks. However, be very conservative about timing—removing a lock that's actually held by a slow operation can cause corruption.

⚠️ **Pitfall: Cross-Platform File Locking**
File locking semantics vary across operating systems, particularly between Windows and Unix-like systems. Test your locking implementation thoroughly on all target platforms, and consider using advisory locks rather than mandatory locks for better portability and recovery options.

### Implementation Guidance

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|--------------|-----------------|
| Error Types | Python exceptions with custom hierarchy | Structured error enums with detailed context |
| Logging | Python logging module with file handlers | Structured logging with correlation IDs |
| File Locking | fcntl-based advisory locks (Unix) | Cross-platform locking with timeout |
| Validation | Manual checks with custom validators | Schema-based validation with automatic checking |
| Recovery | Interactive prompts with user choice | Automated recovery with safety checks |
| Concurrency | Process-level coordination with lock files | Thread-safe operations with proper synchronization |

#### Recommended File Structure

```
project-root/
  src/
    git_impl/
      errors/
        __init__.py              ← error hierarchy and base classes
        corruption.py            ← repository corruption errors
        merge.py                 ← merge conflict and resolution errors  
        concurrency.py           ← concurrent access errors
      validation/
        __init__.py              ← validation framework
        repository.py            ← repository structure validation
        objects.py               ← object integrity validation
        references.py            ← reference consistency validation
      recovery/
        __init__.py              ← recovery orchestration
        corruption.py            ← corruption detection and repair
        merge.py                 ← merge conflict resolution workflows
        locks.py                 ← lock management and cleanup
      concurrency/
        __init__.py              ← concurrency coordination
        locks.py                 ← file-based locking implementation
        coordination.py          ← process coordination patterns
```

#### Infrastructure Starter Code

**Error Hierarchy Implementation:**

```python
"""
Git error hierarchy for comprehensive error handling and recovery.
"""

class GitError(Exception):
    """Base exception for all Git-related errors."""
    
    def __init__(self, message: str, error_code: str = None, recoverable: bool = True):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.recoverable = recoverable
        self.timestamp = time.time()

class CorruptionError(GitError):
    """Raised when repository corruption is detected."""
    
    def __init__(self, component: str, details: str, corruption_type: str):
        message = f"Corruption detected in {component}: {details}"
        super().__init__(message, f"CORRUPT_{corruption_type}", recoverable=True)
        self.component = component
        self.corruption_type = corruption_type
        self.details = details

class MergeConflictError(GitError):
    """Raised when merge conflicts require manual resolution."""
    
    def __init__(self, conflicted_files: List[str], conflict_count: int):
        message = f"Merge conflicts in {len(conflicted_files)} files ({conflict_count} regions)"
        super().__init__(message, "MERGE_CONFLICT", recoverable=True)
        self.conflicted_files = conflicted_files
        self.conflict_count = conflict_count

class ConcurrencyError(GitError):
    """Raised when concurrent access prevents operation completion."""
    
    def __init__(self, resource: str, operation: str, retry_possible: bool = True):
        message = f"Concurrent access conflict on {resource} during {operation}"
        super().__init__(message, "CONCURRENCY_CONFLICT", recoverable=retry_possible)
        self.resource = resource
        self.operation = operation
        self.retry_possible = retry_possible
```

**File Lock Manager:**

```python
"""
Cross-platform file locking for safe concurrent repository access.
"""
import os
import time
import fcntl
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

class FileLock:
    """Advisory file lock for coordinating repository access."""
    
    def __init__(self, lock_path: Path, timeout: float = 30.0):
        self.lock_path = lock_path
        self.timeout = timeout
        self.lock_fd: Optional[int] = None
        self.acquired = False
    
    def acquire(self) -> bool:
        """Acquire the lock, waiting up to timeout seconds."""
        start_time = time.time()
        
        while time.time() - start_time < self.timeout:
            try:
                # Create lock file exclusively
                self.lock_fd = os.open(
                    self.lock_path, 
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644
                )
                
                # Write lock metadata
                lock_info = f"pid={os.getpid()}\ntime={time.time()}\n"
                os.write(self.lock_fd, lock_info.encode())
                os.fsync(self.lock_fd)
                
                self.acquired = True
                return True
                
            except OSError as e:
                if e.errno == errno.EEXIST:
                    # Lock file exists, check if it's orphaned
                    if self._check_orphaned_lock():
                        continue  # Try again after cleanup
                    time.sleep(0.1)  # Wait briefly before retry
                else:
                    raise ConcurrencyError(str(self.lock_path), "lock_acquire")
        
        return False
    
    def release(self):
        """Release the lock if currently held."""
        if self.acquired and self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_path.unlink(missing_ok=True)
            self.acquired = False
            self.lock_fd = None

@contextmanager
def repository_lock(git_dir: Path, operation: str):
    """Context manager for repository-wide locking."""
    lock_file = git_dir / f"{operation}.lock"
    lock = FileLock(lock_file)
    
    try:
        if not lock.acquire():
            raise ConcurrencyError(str(lock_file), operation)
        yield lock
    finally:
        lock.release()
```

#### Core Logic Skeletons

**Repository Validation Framework:**

```python
def validate_repository_state(git_dir: Path) -> List[str]:
    """
    Comprehensive repository validation returning list of issues found.
    
    Returns:
        List of validation error messages, empty if repository is valid
    """
    errors = []
    
    # TODO 1: Validate basic directory structure exists and has correct permissions
    # Check: .git directory, objects/, refs/, refs/heads/, refs/tags/
    # Verify: Directory permissions are 0755, accessible to current user
    
    # TODO 2: Validate HEAD file exists and contains valid reference
    # Check: HEAD file exists and is readable
    # Verify: Content is either valid SHA-1 or symbolic reference format
    
    # TODO 3: Validate object store integrity
    # Scan: All objects in .git/objects directory structure
    # Verify: Each object file can be read, decompressed, and SHA-1 verified
    
    # TODO 4: Validate reference consistency
    # Check: All reference files contain valid SHA-1 hashes
    # Verify: Referenced commits exist in object store
    
    # TODO 5: Validate index file if present
    # Check: Index file format and checksum
    # Verify: All referenced objects exist
    
    return errors

def recover_corrupted_object(object_hash: str, object_store, working_dir: Path) -> bool:
    """
    Attempt to recover a corrupted object from alternative sources.
    
    Returns:
        True if recovery succeeded, False otherwise
    """
    # TODO 1: Try re-reading object from disk (handle transient I/O errors)
    # Use: Multiple read attempts with delays between attempts
    
    # TODO 2: For blob objects, attempt rebuild from working directory
    # Check: If file exists in working directory with matching path
    # Verify: Recompute hash and store if it matches expected hash
    
    # TODO 3: Check if object exists in index for recent staging
    # Load: Current index entries
    # Match: Object hash to staged files for potential recovery
    
    # TODO 4: Log recovery attempt results for debugging
    # Record: What recovery methods were tried and their outcomes
    
    return False
```

**Merge Conflict Resolution:**

```python
def resolve_merge_conflicts(conflicted_files: List[str], merge_info: Dict) -> WorkflowResult:
    """
    Interactive merge conflict resolution workflow.
    
    Args:
        conflicted_files: List of files containing merge conflicts
        merge_info: Information about the merge operation (branches, commits)
    
    Returns:
        WorkflowResult indicating resolution status and remaining conflicts
    """
    resolved_files = []
    remaining_conflicts = []
    
    # TODO 1: Present conflict summary to user
    # Display: Number of conflicts, affected files, conflict types
    # Offer: Resolution options (interactive, tool-based, manual)
    
    # TODO 2: For each conflicted file, detect conflict markers
    # Scan: File content for <<<<<<< ======= >>>>>>> patterns
    # Parse: Conflict regions and extract ours/theirs/base content
    
    # TODO 3: Offer resolution strategies for each conflict
    # Options: Keep ours, keep theirs, manual edit, launch merge tool
    # Validate: Ensure all conflict markers are removed after resolution
    
    # TODO 4: Track resolution progress and allow resumption
    # Save: Resolution state to allow interruption and continuation
    # Update: MERGE_PROGRESS file with completed resolutions
    
    # TODO 5: Validate complete resolution before allowing merge completion
    # Check: No remaining conflict markers in any resolved files
    # Verify: All conflicted files have been addressed
    
    return WorkflowResult(
        status=WorkflowStatus.SUCCESS,
        message="All conflicts resolved",
        files_changed=len(resolved_files),
        conflicts=remaining_conflicts
    )

def generate_conflict_markers(conflict: ConflictRegion, our_branch: str, their_branch: str) -> str:
    """
    Generate standard Git conflict markers for a conflict region.
    
    Returns:
        Formatted conflict markers with content from both sides
    """
    # TODO 1: Format opening marker with our branch identifier
    # Format: "<<<<<<< {our_branch}" or "<<<<<<< HEAD" for detached state
    
    # TODO 2: Add our content with proper line endings
    # Include: All lines from our version of the conflict region
    
    # TODO 3: Add separator marker
    # Insert: "=======" line to separate our content from theirs
    
    # TODO 4: Add their content with branch identifier
    # Include: All lines from their version
    # Format: ">>>>>>> {their_branch}" closing marker
    
    # TODO 5: Optionally include base content for three-way conflicts
    # Insert: "||||||| merged common ancestors" section if base differs
    
    return ""
```

#### Milestone Checkpoints

**After implementing corruption detection:**
```bash
# Test repository validation
python -m git_impl.validation.repository /path/to/repo

# Expected output: List of validation results
# Should detect: Missing directories, corrupted objects, invalid references
# Should report: Specific error locations and types

# Test object corruption recovery
echo "test content" > test.txt
python -m git_impl add test.txt
# Manually corrupt the blob object file
python -m git_impl.recovery.corruption --scan --fix

# Expected behavior: Detection of corrupted object, attempt recovery from working tree
```

**After implementing merge conflict resolution:**
```bash
# Create merge conflict scenario
git checkout -b feature
echo "feature change" >> file.txt
git add file.txt && git commit -m "feature"

git checkout main  
echo "main change" >> file.txt
git add file.txt && git commit -m "main"

python -m git_impl.merge feature

# Expected output: Conflict detection, interactive resolution prompt
# Should create: Conflict markers in file.txt
# Should provide: Resolution options and validation
```

#### Debugging Tips

| Symptom | Likely Cause | Diagnosis | Fix |
|---------|--------------|-----------|-----|
| "Object not found" errors | Corrupted object store | Check .git/objects structure | Run repository validation |
| Merge hangs indefinitely | Deadlock in file locking | Check for orphaned lock files | Remove old .lock files |
| "Repository corrupt" on startup | Interrupted operation | Check for partial state files | Run corruption recovery |
| Conflicts not detected properly | Three-way merge base error | Verify merge base calculation | Check commit history integrity |
| Lock timeout errors | High concurrent access | Monitor lock file creation/deletion | Increase timeout or serialize operations |
| Index corruption after crash | Non-atomic index write | Check index file permissions and size | Rebuild index from working tree |


## Testing Strategy and Milestone Checkpoints

> **Milestone(s):** This section provides comprehensive testing approaches for all eight milestones, with specific verification steps for each milestone and integration testing strategies to ensure compatibility with real Git.

### Mental Model: The Quality Assurance Laboratory

Think of testing your Git implementation like running a quality assurance laboratory for a manufacturing process. Each milestone represents a critical component that must pass rigorous inspection before moving to the next stage. Just as a car manufacturer tests each subsystem (engine, brakes, transmission) individually before assembling the complete vehicle, we must verify each Git component works correctly in isolation before testing the integrated system.

The testing laboratory has multiple inspection stations. **Unit testing** is like checking individual parts under a microscope - ensuring each bolt meets specifications. **Integration testing** is like testing how parts work together - does the engine properly connect to the transmission? **Compatibility testing** is like verifying your car can drive on the same roads as other vehicles - does your Git produce the same results as the official Git implementation?

The most crucial aspect of this testing laboratory is the **golden standard comparison**. Just as manufacturers compare their products against industry standards, we compare our Git implementation against the official Git behavior. Every hash, every file format, every command output must match exactly. This isn't just about correctness - it's about ensuring users can seamlessly switch between your implementation and official Git without any surprises.

### Milestone Verification Steps

Each milestone builds upon previous ones, creating a dependency chain that requires careful validation. The verification approach uses both **black-box testing** (testing external behavior) and **white-box testing** (verifying internal state). The key insight is that Git's deterministic nature means identical inputs should always produce identical outputs, making automated verification straightforward.

#### Milestone 1: Repository Initialization Verification

Repository initialization is the foundation - if the directory structure is incorrect, all subsequent operations will fail mysteriously. The verification process must check both the presence and exact contents of each required file and directory.

| Verification Step | Command | Expected Result | Validation Method |
|------------------|---------|------------------|-------------------|
| Directory Structure Creation | `ls -la .git/` | Shows objects/, refs/, HEAD with correct permissions | Directory existence and permissions check |
| Objects Directory Layout | `ls -la .git/objects/` | Empty directory with proper subdirectory structure | Subdirectory count and permissions |
| References Directory | `ls -la .git/refs/` | Contains heads/ and tags/ subdirectories | Directory structure validation |
| HEAD File Content | `cat .git/HEAD` | Exactly "ref: refs/heads/master\n" | String comparison with newline |
| Git Recognition | `git status` (using official Git) | Recognizes as valid Git repository | Official Git compatibility test |

The most critical validation is ensuring the HEAD file contains the exact string format. Many implementations fail because they forget the newline character or use incorrect reference syntax.

**Detailed Verification Procedure:**

1. **Structure Validation**: Use `find .git -type d` to list all directories and compare against the expected structure. The output should include `.git/objects/`, `.git/refs/`, `.git/refs/heads/`, `.git/refs/tags/`, and `.git/hooks/`.

2. **Permissions Check**: Verify directory permissions using `stat -c '%a' .git/` - should return `755` for proper read/write/execute permissions.

3. **HEAD Content Validation**: Read the HEAD file byte-by-byte and compare against the expected content. The file must contain exactly 21 bytes: "ref: refs/heads/master" followed by a newline character (0x0A).

4. **Official Git Compatibility**: Run `git rev-parse --git-dir` using official Git - it should return `.git` without errors, confirming the repository structure is valid.

5. **Empty Repository State**: Verify `git log` returns "fatal: your current branch 'master' does not have any commits yet" - this confirms the repository is properly initialized but empty.

#### Milestone 2: Object Storage (Blobs) Verification

Blob storage is the foundation of Git's content-addressable system. Every aspect must match official Git exactly - hash computation, compression, file paths, and retrieval. The verification process focuses on round-trip consistency and hash determinism.

| Verification Step | Test Input | Expected Hash | Validation Method |
|------------------|------------|---------------|-------------------|
| Empty File Hash | `""` (empty string) | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` | Hash comparison with official Git |
| Simple Text Hash | `"hello world"` | `95d09f2b10159347eece71399a7e2e907ea3df4f` | Hash comparison with official Git |
| Binary Content Hash | Random 1KB binary data | Must match `git hash-object` output | Binary content handling test |
| Large File Hash | 10MB text file | Must match official Git hash | Performance and correctness test |
| Round-trip Consistency | Any content | `cat-file` output matches original input | Storage and retrieval verification |

The hash computation must implement the exact Git algorithm: `SHA1("blob " + content_length + "\0" + content)`. Many implementations fail by forgetting the space after "blob" or using incorrect size formatting.

**Detailed Verification Procedure:**

1. **Hash Algorithm Verification**: Create a test file with known content and verify your hash matches official Git exactly. Use `echo -n "test content" | git hash-object --stdin` as the reference.

2. **File Path Generation**: Verify your implementation creates the correct object path. For hash `abc123...`, the file should be stored at `.git/objects/ab/c123...` with the first two characters forming the directory name.

3. **Compression Verification**: Decompress stored objects manually using Python's zlib and verify the content matches the expected format: `blob {size}\0{content}`.

4. **Binary Content Handling**: Test with files containing null bytes, non-UTF8 content, and various binary formats. Git treats all content as binary internally.

5. **Performance Baseline**: Measure hash computation and storage time for various file sizes. Your implementation should handle files up to several megabytes without excessive memory usage.

**Cross-Implementation Testing**:
```bash
# Create test file
echo "test content" > test.txt

# Hash with your implementation
./your-git hash-object test.txt

# Hash with official Git
git hash-object test.txt

# Both should produce identical output
```

#### Milestone 3: Tree Objects Verification

Tree objects represent directory structures and must maintain exact sorting and binary format compatibility with official Git. The verification focuses on directory traversal, entry sorting, and nested tree creation.

| Verification Step | Test Scenario | Expected Behavior | Validation Method |
|------------------|---------------|-------------------|-------------------|
| Single File Tree | Directory with one file | Tree with single blob entry | Tree content comparison |
| Sorted Entries | Files: z.txt, a.txt, m.txt | Tree entries sorted alphabetically | Entry order verification |
| Nested Directories | Subdirectory with files | Parent tree references child tree | Tree hierarchy validation |
| Mixed Content Types | Files and subdirectories | Correct mode values (100644, 040000) | Mode field verification |
| Empty Directory Handling | Empty subdirectory | Directory not included in tree | Git's empty directory behavior |

Tree objects use a specific binary format where each entry is: `{mode} {name}\0{20-byte-hash}`. The mode values are crucial - regular files use `100644`, executable files use `100755`, and directories use `040000`.

**Detailed Verification Procedure:**

1. **Entry Sorting Verification**: Create a directory with files named in non-alphabetical order. Verify your tree object lists entries in exact alphabetical order, matching `git ls-tree` output.

2. **Binary Format Validation**: Extract a tree object and verify the binary format byte-by-byte. Each entry should be null-terminated, followed by exactly 20 bytes of binary hash data.

3. **Mode Value Accuracy**: Test files with different permissions and verify the mode values match Git's behavior. Use `git ls-tree -l` to see detailed mode information.

4. **Nested Tree Verification**: Create a directory structure with subdirectories and verify your implementation creates the correct tree hierarchy. Each subdirectory should become a separate tree object referenced by its parent.

5. **Hash Consistency**: Build the same directory structure twice and verify identical tree hashes are produced. Tree building must be deterministic.

#### Milestone 4: Commit Objects Verification

Commit objects tie together trees, parents, and metadata to form Git's history graph. Verification must ensure exact format compatibility and proper parent linking.

| Verification Step | Test Case | Expected Format | Validation Method |
|------------------|-----------|-----------------|-------------------|
| Initial Commit | Commit with no parents | No parent field in commit object | Commit format verification |
| Parent Linking | Commit with one parent | Single parent line with hash | Parent reference validation |
| Merge Commit | Commit with two parents | Multiple parent lines | Multi-parent commit handling |
| Author/Committer Data | Different author and committer | Separate author/committer lines with timestamps | Metadata format verification |
| Message Handling | Multi-line commit message | Message separated by blank line | Message format validation |

Commit objects follow a strict text format with specific field ordering and timestamp formatting. The timestamp format is Unix epoch seconds followed by timezone offset (e.g., `1609459200 +0000`).

**Detailed Verification Procedure:**

1. **Commit Format Validation**: Create commits and verify the object format matches `git cat-file commit <hash>` exactly. Field order is: tree, parent(s), author, committer, blank line, message.

2. **Timestamp Format**: Verify timestamps use Unix epoch format with timezone. Compare your output with `git log --format=fuller` to ensure exact match.

3. **Parent Chain Verification**: Create multiple commits and verify the parent relationships form a proper chain. Each commit should reference its predecessor correctly.

4. **Character Encoding**: Test commits with non-ASCII characters in messages and author names. Git uses UTF-8 encoding internally.

5. **Empty Message Handling**: Test commits with empty messages and verify they're handled correctly (Git allows empty messages).

#### Milestone 5: References and Branches Verification

References provide human-readable names for commits and must handle both symbolic and direct references correctly. The verification focuses on file-based storage and HEAD state management.

| Verification Step | Test Operation | Expected Outcome | Validation Method |
|------------------|----------------|------------------|-------------------|
| Branch Creation | Create branch "feature" | File `.git/refs/heads/feature` contains commit hash | File content verification |
| HEAD Update | Switch to branch | HEAD contains `ref: refs/heads/feature` | Symbolic reference validation |
| Detached HEAD | Checkout specific commit | HEAD contains raw commit hash | Direct reference handling |
| Branch Deletion | Delete branch | Reference file removed | File system state check |
| Invalid Names | Create branch with invalid characters | Operation fails with clear error | Error handling validation |

Reference files are plain text containing either a commit hash (direct reference) or a symbolic reference in the format `ref: refs/heads/branch-name`.

**Detailed Verification Procedure:**

1. **Reference File Format**: Create branches and verify the reference files contain exactly 41 characters - 40 hex digits for the hash plus a newline character.

2. **Symbolic Reference Handling**: Switch branches and verify HEAD is updated to point to the correct branch reference, not the commit hash directly.

3. **Detached HEAD State**: Checkout a specific commit and verify HEAD contains the raw hash. Test that subsequent commits update HEAD directly rather than through a branch.

4. **Branch Namespace**: Test branch names with slashes (e.g., `feature/login`) and verify the directory structure is created correctly under `.git/refs/heads/`.

5. **Concurrent Access**: Test rapid branch creation/deletion to ensure file operations are atomic and don't leave partially written files.

#### Milestone 6: Index (Staging Area) Verification

The index is Git's most complex binary format, requiring exact compatibility for field layouts, checksums, and metadata handling. Verification must cover all aspects of the binary format specification.

| Verification Step | Test Scenario | Expected Behavior | Validation Method |
|------------------|---------------|-------------------|-------------------|
| File Staging | Add single file | Index entry with correct metadata | Binary format parsing |
| Multiple Files | Stage several files | Entries sorted by path | Entry ordering validation |
| File Modification | Modify staged file | Status shows modified state | Three-way comparison accuracy |
| Index Checksum | Any index operation | Valid SHA-1 checksum at file end | Checksum verification |
| Metadata Accuracy | Stage file with specific timestamps | Index preserves exact metadata | Metadata comparison |

The index binary format includes a 12-byte header, variable-length entries, and a 20-byte SHA-1 checksum. Each entry contains extensive file metadata for change detection.

**Detailed Verification Procedure:**

1. **Binary Format Compliance**: Parse index files created by official Git and verify your implementation can read them correctly. Also verify official Git can read your index files.

2. **Checksum Validation**: Verify the SHA-1 checksum at the end of index files. Any corruption should be detected when loading the index.

3. **Metadata Preservation**: Stage files and verify all metadata (timestamps, file size, permissions, device/inode numbers) is stored and retrieved accurately.

4. **Path Sorting**: Stage files with various path names and verify they're stored in correct sort order. The sorting algorithm affects index format compatibility.

5. **Partial Updates**: Test staging individual files multiple times and verify the index is updated correctly without corrupting other entries.

#### Milestone 7: Diff Algorithm Verification

The Myers diff algorithm must produce output that matches Git's diff format exactly, including context lines, hunk headers, and binary file detection.

| Verification Step | Test Case | Expected Output | Validation Method |
|------------------|-----------|------------------|-------------------|
| Simple Addition | Add lines to file | Unified diff with + markers | Output format comparison |
| Line Deletion | Remove lines from file | Unified diff with - markers | Deletion marking accuracy |
| Line Modification | Change existing lines | Shows as deletion + addition | Edit sequence accuracy |
| Context Lines | Large file changes | Correct @@ hunk headers with line numbers | Hunk formatting validation |
| Binary Files | Binary file changes | "Binary files differ" message | Binary detection accuracy |

The unified diff format includes specific hunk headers with old and new line number ranges, plus context lines around changes.

**Detailed Verification Procedure:**

1. **Algorithm Correctness**: Test the Myers algorithm against known inputs with verified shortest edit scripts. The algorithm must find optimal solutions.

2. **Unified Format Compliance**: Compare your diff output with `git diff` for identical files. Header format, line prefixes, and hunk boundaries must match exactly.

3. **Binary Detection**: Test various file types (images, executables, text with null bytes) and verify binary detection matches Git's heuristics.

4. **Large File Performance**: Test diff performance with large files (thousands of lines) and verify the algorithm completes in reasonable time.

5. **Edge Cases**: Test empty files, single-line files, files with only whitespace changes, and files with no final newline.

#### Milestone 8: Three-Way Merge Verification

Three-way merge is the most complex operation, requiring correct merge base calculation, conflict detection, and merge commit creation. Verification must cover all merge scenarios.

| Verification Step | Test Scenario | Expected Result | Validation Method |
|------------------|---------------|------------------|------------------|
| Fast-Forward Merge | One branch is ancestor of other | Branch pointer update only | Merge strategy detection |
| Clean Merge | Non-overlapping changes | Automatic merge completion | Conflict-free merging |
| Content Conflict | Overlapping line changes | Conflict markers in file | Conflict detection accuracy |
| Merge Base Calculation | Complex branch history | Correct common ancestor | Graph traversal validation |
| Merge Commit Creation | Successful merge | Commit with two parents | Multi-parent commit handling |

The merge algorithm must handle various conflict types and produce standard conflict markers compatible with Git's format.

**Detailed Verification Procedure:**

1. **Merge Base Accuracy**: Create complex branch histories and verify merge base calculation matches `git merge-base` output exactly.

2. **Conflict Marker Format**: Create conflicting changes and verify conflict markers match Git's format exactly, including branch name labels.

3. **Three-Way Algorithm**: Test merge scenarios where base, ours, and theirs all differ, ensuring the algorithm correctly identifies conflicting vs. non-conflicting regions.

4. **File-Level Conflicts**: Test scenarios where files are added, deleted, or modified in conflicting ways across branches.

5. **Performance Testing**: Test merge performance with large files and many conflicts to ensure the algorithm scales reasonably.

### Integration Testing Approach

Integration testing verifies that your Git implementation behaves identically to official Git across complete workflows. The approach uses **comparative testing** where every operation is performed with both implementations and results are compared.

#### Golden Standard Testing Framework

The golden standard approach treats official Git as the authoritative reference. Every test scenario is executed with both your implementation and official Git, with results compared byte-for-byte where applicable.

| Testing Layer | Scope | Comparison Method | Failure Handling |
|---------------|--------|-------------------|------------------|
| Object Compatibility | Individual objects | Hash and content comparison | Object format debugging |
| Repository State | Complete repository state | File system diff of .git directory | State reconstruction analysis |
| Command Output | User-visible output | String comparison with normalization | Output format adjustment |
| File Content | Working directory files | Byte-by-byte comparison | Content merge verification |
| Performance Baseline | Operation timing | Relative performance comparison | Performance regression detection |

The framework must handle **environment differences** that don't indicate implementation bugs. For example, timestamps will differ between test runs, so they must be normalized or ignored during comparison.

#### Test Scenario Categories

**Basic Workflow Testing** covers the fundamental Git operations that users perform daily. These scenarios must work flawlessly as they form the foundation of version control workflows.

1. **Repository Lifecycle**: Initialize repository, add files, create commits, view history. This tests the basic object creation and storage pipeline.

2. **Branch Operations**: Create branches, switch branches, merge branches. This tests the reference management and merge algorithms.

3. **Staging Operations**: Stage files, unstage files, partial staging. This tests the index management and status calculation.

4. **History Navigation**: Checkout previous commits, view diffs between versions. This tests object retrieval and diff algorithms.

5. **Merge Scenarios**: Fast-forward merges, three-way merges, conflict resolution. This tests the most complex algorithms in your implementation.

**Edge Case Testing** covers unusual but valid Git operations that might expose implementation bugs or missing functionality.

| Edge Case Category | Test Scenarios | Common Failure Points |
|--------------------|----------------|----------------------|
| Empty Content | Empty files, empty directories, empty commits | Null pointer handling, zero-length content |
| Large Content | Large files, many files, deep directory trees | Memory usage, performance degradation |
| Special Characters | Unicode filenames, binary content, line endings | Character encoding, binary detection |
| Concurrent Operations | Multiple processes modifying repository | File locking, atomic operations |
| Corrupted State | Missing objects, invalid references, corrupted index | Error detection, graceful degradation |

**Cross-Platform Testing** ensures your implementation works consistently across different operating systems and file systems.

- **File System Differences**: Test on case-sensitive and case-insensitive file systems. Some Git operations behave differently based on file system capabilities.

- **Path Separator Handling**: Verify path normalization works correctly on Windows (backslashes) and Unix (forward slashes).

- **Permission Model Differences**: Test file permissions and executable bit handling across platforms where permission models differ.

- **Line Ending Handling**: Test files with different line ending conventions (LF, CRLF, mixed) to ensure consistent behavior.

#### Automated Compatibility Verification

The compatibility verification system runs automated test suites that compare your implementation against official Git across hundreds of scenarios. The system must handle the inherent non-determinism in some Git operations (like timestamps) while catching genuine compatibility issues.

**Test Data Generation** creates comprehensive test scenarios programmatically rather than maintaining large test fixtures. This ensures broad coverage without overwhelming the test suite with data.

```
Test Repository Generator:
1. Generate random file content with various characteristics
2. Create random directory structures with different depths
3. Simulate realistic commit histories with merges and branches
4. Create conflict scenarios with overlapping changes
5. Generate edge cases (empty files, binary content, special names)
```

**Result Comparison Engine** handles the complex task of comparing Git repository states while accounting for acceptable differences.

| Comparison Aspect | Exact Match Required | Acceptable Differences | Normalization Method |
|-------------------|---------------------|------------------------|---------------------|
| Object Hashes | Yes | None | Direct string comparison |
| Object Content | Yes | None | Byte-by-byte comparison |
| Reference Values | Yes | None | Hash comparison |
| File Timestamps | No | Any difference | Ignore during comparison |
| Commit Timestamps | No | Any difference | Normalize to fixed value |
| Author Information | Yes | None | String comparison |
| File Permissions | Yes | None | Octal comparison |

**Regression Testing** maintains a suite of previously passing tests to ensure new changes don't break existing functionality. The regression suite includes both successful operations and expected failures.

#### Performance and Scalability Testing

Performance testing ensures your Git implementation scales reasonably with repository size and complexity. While exact performance parity with official Git isn't required, your implementation should handle realistic workloads without excessive resource consumption.

**Scalability Test Scenarios:**

| Scale Factor | Repository Size | File Count | Commit History | Expected Behavior |
|--------------|----------------|------------|----------------|-------------------|
| Small | < 10MB | < 100 files | < 50 commits | Sub-second operations |
| Medium | 10-100MB | 100-1000 files | 50-500 commits | Operations complete in seconds |
| Large | 100MB-1GB | 1000-10000 files | 500-5000 commits | Operations complete in minutes |
| Stress Test | > 1GB | > 10000 files | > 5000 commits | Graceful degradation, no crashes |

**Memory Usage Monitoring** ensures your implementation doesn't have memory leaks or excessive memory consumption during normal operations.

**Performance Regression Detection** compares operation timing across implementation versions to catch performance regressions early.

### Common Testing Pitfalls

Understanding common testing mistakes helps avoid frustration and ensures your testing efforts are effective. These pitfalls are based on frequent issues encountered when building Git implementations.

⚠️ **Pitfall: Ignoring Byte-Level Compatibility**

Many implementers focus on functional correctness while ignoring exact byte-level compatibility with Git's formats. This leads to repositories that work with your implementation but fail when accessed by official Git.

**Why it's problematic**: Git's binary formats (index file, pack files, object formats) have specific layouts that must be followed exactly. Even single-byte differences can cause incompatibility.

**How to avoid**: Always test round-trip compatibility - create objects with your implementation and verify official Git can read them, and vice versa. Use hex dumps to compare binary formats byte-by-byte when debugging.

⚠️ **Pitfall: Testing Only Happy Paths**

Focusing testing on successful operations while ignoring error conditions and edge cases leads to implementations that fail unexpectedly in real-world usage.

**Why it's problematic**: Real Git repositories encounter corrupted files, network failures, concurrent access, and unusual content. Your implementation must handle these gracefully.

**How to avoid**: Dedicate significant testing effort to error conditions. Test with corrupted objects, missing files, invalid references, and concurrent operations. Verify error messages are helpful and recovery is possible.

⚠️ **Pitfall: Timestamp and Environment Dependencies**

Writing tests that depend on specific timestamps, user information, or system configuration makes tests brittle and difficult to reproduce across different environments.

**Why it's problematic**: Tests fail randomly based on when they're run or what system they're run on, making it difficult to distinguish real bugs from environmental issues.

**How to avoid**: Use fixed timestamps and author information in tests. Normalize or ignore environment-specific data when comparing results. Use dependency injection to control environmental factors.

⚠️ **Pitfall: Insufficient Cross-Platform Testing**

Testing only on one operating system misses compatibility issues that arise from file system differences, path handling, and permission models.

**Why it's problematic**: Git repositories are often shared across different operating systems. Incompatibilities can corrupt repositories or cause data loss.

**How to avoid**: Test on multiple platforms, especially Windows and Linux. Pay special attention to path separators, case sensitivity, and file permissions. Use continuous integration to automate cross-platform testing.

⚠️ **Pitfall: Missing Performance Reality Checks**

Implementing algorithms without testing performance on realistic data sizes leads to implementations that work on small test cases but become unusable on real repositories.

**Why it's problematic**: Algorithms with poor complexity (like O(n²) diff algorithms) work fine on small test files but become unusably slow on large files or repositories.

**How to avoid**: Include performance tests with realistic data sizes in your test suite. Set reasonable performance expectations and fail tests that exceed them. Profile your implementation to identify bottlenecks.

### Integration with Development Workflow

The testing strategy must integrate seamlessly with the development workflow to provide rapid feedback and prevent regressions. The approach uses multiple testing layers triggered at different points in the development process.

**Pre-commit Testing** runs quickly during development to catch obvious bugs before they're committed to version control.

**Continuous Integration Testing** runs comprehensive test suites on every code change, including cross-platform and performance testing.

**Release Testing** performs exhaustive compatibility testing before releasing new versions, including testing against multiple Git versions and large real-world repositories.

### Implementation Guidance

The testing implementation requires careful attention to automation, reliability, and maintainability. The goal is creating a testing framework that provides confidence in your Git implementation while being practical to maintain and extend.

#### Technology Recommendations

| Testing Component | Simple Option | Advanced Option |
|------------------|---------------|-----------------|
| Test Framework | pytest with subprocess calls | Custom test harness with Git process management |
| Repository Setup | Manual git init in test directories | Programmatic repository generation with GitPython |
| Result Comparison | String comparison with basic normalization | AST-based parsing and semantic comparison |
| Performance Testing | Manual timing with time.time() | pytest-benchmark with statistical analysis |
| Cross-Platform Testing | Manual testing on multiple systems | GitHub Actions matrix builds |

#### Recommended Testing File Structure

```
build-your-own-git/
├── tests/
│   ├── unit/                    # Unit tests for individual components
│   │   ├── test_object_store.py
│   │   ├── test_index.py
│   │   ├── test_references.py
│   │   └── test_diff_algorithm.py
│   ├── integration/             # Integration tests comparing with official Git
│   │   ├── test_basic_workflow.py
│   │   ├── test_branch_operations.py
│   │   ├── test_merge_scenarios.py
│   │   └── test_compatibility.py
│   ├── milestone/               # Milestone-specific verification tests
│   │   ├── test_milestone_1.py  # Repository initialization
│   │   ├── test_milestone_2.py  # Blob storage
│   │   └── ...
│   ├── fixtures/                # Test data and repository templates
│   │   ├── sample_repos/
│   │   ├── binary_files/
│   │   └── conflict_scenarios/
│   └── utils/                   # Testing utilities and helpers
│       ├── git_comparison.py    # Compare with official Git
│       ├── repo_generator.py    # Generate test repositories
│       └── binary_parser.py     # Parse Git binary formats
├── src/
│   └── your_git/               # Your Git implementation
└── conftest.py                 # pytest configuration and fixtures
```

#### Infrastructure Starter Code

**Git Comparison Utility** provides a complete framework for comparing your implementation with official Git:

```python
#!/usr/bin/env python3
"""
Git Comparison Utility - Complete implementation for comparing custom Git with official Git.
This utility handles environment normalization and result comparison.
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json

class GitComparison:
    """Utility for comparing custom Git implementation with official Git."""
    
    def __init__(self, custom_git_path: str, official_git_path: str = "git"):
        self.custom_git = custom_git_path
        self.official_git = official_git_path
        self.test_dir = None
        self.comparison_results = []
    
    def setup_test_environment(self) -> Path:
        """Create isolated test directory with proper Git environment."""
        self.test_dir = Path(tempfile.mkdtemp(prefix="git_test_"))
        
        # Set consistent Git environment
        os.environ.update({
            'GIT_AUTHOR_NAME': 'Test Author',
            'GIT_AUTHOR_EMAIL': 'test@example.com',
            'GIT_COMMITTER_NAME': 'Test Committer',
            'GIT_COMMITTER_EMAIL': 'test@example.com',
            'GIT_AUTHOR_DATE': '2021-01-01T12:00:00Z',
            'GIT_COMMITTER_DATE': '2021-01-01T12:00:00Z'
        })
        
        return self.test_dir
    
    def cleanup_test_environment(self):
        """Remove test directory and reset environment."""
        if self.test_dir and self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir = None
    
    def run_command(self, git_binary: str, args: List[str], cwd: Path) -> Tuple[int, str, str]:
        """Run Git command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                [git_binary] + args,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", f"Command failed: {e}"
    
    def normalize_output(self, output: str, operation: str) -> str:
        """Normalize output for comparison, handling acceptable differences."""
        lines = output.strip().split('\n')
        normalized_lines = []
        
        for line in lines:
            # Remove timestamp variations for certain operations
            if operation in ['log', 'show'] and 'Date:' in line:
                continue  # Skip timestamp lines
            
            # Normalize path separators
            line = line.replace('\\', '/')
            
            # Remove trailing whitespace
            line = line.rstrip()
            
            normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)
    
    def compare_repository_state(self, repo_path: Path) -> Dict[str, Any]:
        """Compare complete repository state between implementations."""
        state_comparison = {
            'objects_match': True,
            'refs_match': True,
            'index_match': True,
            'differences': []
        }
        
        git_dir = repo_path / '.git'
        if not git_dir.exists():
            state_comparison['differences'].append("No .git directory found")
            return state_comparison
        
        # Compare object store
        objects_dir = git_dir / 'objects'
        if objects_dir.exists():
            for obj_dir in objects_dir.iterdir():
                if obj_dir.is_dir() and len(obj_dir.name) == 2:
                    for obj_file in obj_dir.iterdir():
                        obj_hash = obj_dir.name + obj_file.name
                        # Verify object can be read by both implementations
                        custom_result = self.run_command(
                            self.custom_git, ['cat-file', '-p', obj_hash], repo_path
                        )
                        official_result = self.run_command(
                            self.official_git, ['cat-file', '-p', obj_hash], repo_path
                        )
                        
                        if custom_result[1] != official_result[1]:
                            state_comparison['objects_match'] = False
                            state_comparison['differences'].append(
                                f"Object {obj_hash} content differs"
                            )
        
        return state_comparison
```

**Repository Generator** creates test repositories programmatically:

```python
#!/usr/bin/env python3
"""
Repository Generator - Creates test repositories with various characteristics.
"""

import os
import random
import string
from pathlib import Path
from typing import List, Dict, Any
import tempfile

class TestRepositoryGenerator:
    """Generate Git repositories for testing various scenarios."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(exist_ok=True)
    
    def create_simple_repository(self, name: str) -> Path:
        """Create a simple repository with a few commits."""
        repo_path = self.base_dir / name
        repo_path.mkdir(exist_ok=True)
        
        # Initialize repository
        os.system(f"cd {repo_path} && git init")
        
        # Create initial commit
        (repo_path / "README.md").write_text("# Test Repository\n")
        os.system(f"cd {repo_path} && git add README.md && git commit -m 'Initial commit'")
        
        # Create second commit
        (repo_path / "file1.txt").write_text("Content of file 1\n")
        os.system(f"cd {repo_path} && git add file1.txt && git commit -m 'Add file1'")
        
        return repo_path
    
    def create_merge_scenario(self, name: str) -> Path:
        """Create repository with merge scenario for testing."""
        repo_path = self.base_dir / name
        repo_path.mkdir(exist_ok=True)
        
        os.system(f"cd {repo_path} && git init")
        
        # Create main branch commits
        (repo_path / "main.txt").write_text("Main branch content\n")
        os.system(f"cd {repo_path} && git add main.txt && git commit -m 'Main commit'")
        
        # Create feature branch
        os.system(f"cd {repo_path} && git checkout -b feature")
        (repo_path / "feature.txt").write_text("Feature branch content\n")
        os.system(f"cd {repo_path} && git add feature.txt && git commit -m 'Feature commit'")
        
        # Return to main and create conflicting change
        os.system(f"cd {repo_path} && git checkout main")
        (repo_path / "main.txt").write_text("Modified main branch content\n")
        os.system(f"cd {repo_path} && git add main.txt && git commit -m 'Main modification'")
        
        return repo_path
    
    def generate_random_content(self, size: int, binary: bool = False) -> bytes:
        """Generate random content for testing."""
        if binary:
            return bytes(random.randint(0, 255) for _ in range(size))
        else:
            chars = string.ascii_letters + string.digits + '\n \t'
            return ''.join(random.choice(chars) for _ in range(size)).encode('utf-8')
```

#### Core Logic Skeleton Code

**Milestone Verification Test Template**:

```python
#!/usr/bin/env python3
"""
Milestone Verification Template - Complete test structure for milestone validation.
"""

import pytest
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any

class MilestoneVerifier:
    """Base class for milestone verification tests."""
    
    def __init__(self, git_implementation_path: str):
        self.git_path = git_implementation_path
        self.test_repo = None
    
    def setup_test_repo(self) -> Path:
        """Create temporary test repository."""
        # TODO 1: Create temporary directory for test repository
        # TODO 2: Set consistent Git environment variables (author, committer, dates)
        # TODO 3: Return path to test repository
        pass
    
    def cleanup_test_repo(self):
        """Clean up temporary test repository."""
        # TODO 1: Remove temporary directory if it exists
        # TODO 2: Reset environment variables if needed
        pass
    
    def run_git_command(self, args: List[str]) -> tuple[int, str, str]:
        """Run Git command with your implementation."""
        # TODO 1: Build command line with git_path and args
        # TODO 2: Execute command in test repository directory
        # TODO 3: Return (exit_code, stdout, stderr) tuple
        # TODO 4: Handle timeout and other execution errors
        pass
    
    def verify_file_exists(self, file_path: Path, expected_content: str = None) -> bool:
        """Verify file exists and optionally check content."""
        # TODO 1: Check if file exists at expected path
        # TODO 2: If expected_content provided, read file and compare
        # TODO 3: Return True if verification passes, False otherwise
        pass
    
    def compare_with_official_git(self, command: List[str]) -> bool:
        """Compare command output with official Git."""
        # TODO 1: Run command with your implementation
        # TODO 2: Run same command with official Git
        # TODO 3: Normalize outputs (remove timestamps, etc.)
        # TODO 4: Return True if outputs match, False otherwise
        pass

class Milestone1Verifier(MilestoneVerifier):
    """Verify Milestone 1: Repository Initialization."""
    
    def test_git_init(self):
        """Test repository initialization creates correct structure."""
        # TODO 1: Run 'init' command in test directory
        # TODO 2: Verify .git directory exists with correct permissions
        # TODO 3: Verify .git/objects directory exists with subdirectories
        # TODO 4: Verify .git/refs/heads directory exists
        # TODO 5: Verify HEAD file exists with correct content
        # TODO 6: Test official Git recognizes repository as valid
        pass
    
    def test_directory_structure(self):
        """Verify complete .git directory structure."""
        # TODO 1: Initialize repository
        # TODO 2: Check all required directories exist: objects, refs, refs/heads, refs/tags
        # TODO 3: Check directory permissions are correct (0755)
        # TODO 4: Verify no extra files or directories are created
        pass
    
    def test_head_file_format(self):
        """Verify HEAD file has exact correct format."""
        # TODO 1: Initialize repository
        # TODO 2: Read HEAD file content as bytes
        # TODO 3: Verify content is exactly "ref: refs/heads/master\n"
        # TODO 4: Verify file has exactly 21 bytes
        pass
```

#### Language-Specific Testing Hints

**Python Testing Environment**:
- Use `pytest` as the test framework for its excellent fixtures and parametrization
- Use `subprocess.run()` with `capture_output=True` for running Git commands
- Use `tempfile.TemporaryDirectory()` for isolated test environments
- Use `pathlib.Path` for cross-platform path handling
- Set `PYTHONPATH` to include your Git implementation directory

**Test Data Management**:
- Store binary test data as base64 strings in test files to avoid Git corruption
- Use `pytest.mark.parametrize` to run the same test with multiple inputs
- Create reusable fixtures for common repository states
- Use `pytest-xdist` for parallel test execution to speed up large test suites

**Cross-Platform Considerations**:
- Use `os.name` and `platform.system()` to detect platform-specific behavior
- Normalize path separators using `pathlib` or `os.path.normpath()`
- Handle case sensitivity differences in file system operations
- Test file permissions carefully on Windows where the model differs from Unix

#### Milestone Checkpoints

After implementing each milestone, run these specific verification commands to ensure correctness:

**Milestone 1 Checkpoint**:
```bash
# Test repository initialization
python your_git.py init
ls -la .git/
cat .git/HEAD
git status  # Using official Git to verify structure
```

**Milestone 2 Checkpoint**:
```bash
# Test blob storage and retrieval
echo "test content" | python your_git.py hash-object --stdin
python your_git.py cat-file blob <hash>
# Compare hash with: echo "test content" | git hash-object --stdin
```

**Milestone 3 Checkpoint**:
```bash
# Test tree object creation
mkdir testdir
echo "file content" > testdir/file.txt
python your_git.py write-tree
python your_git.py ls-tree <tree-hash>
```

**Milestone 4 Checkpoint**:
```bash
# Test commit creation
python your_git.py commit-tree <tree-hash> -m "Test commit"
python your_git.py cat-file commit <commit-hash>
```

**Milestone 5 Checkpoint**:
```bash
# Test branch operations
python your_git.py branch feature <commit-hash>
cat .git/refs/heads/feature
python your_git.py checkout feature
cat .git/HEAD
```

**Milestone 6 Checkpoint**:
```bash
# Test staging operations
echo "new content" > newfile.txt
python your_git.py add newfile.txt
python your_git.py status
hexdump -C .git/index  # Verify binary format
```

**Milestone 7 Checkpoint**:
```bash
# Test diff algorithm
echo -e "line1\nline2\nline3" > file1.txt
echo -e "line1\nmodified\nline3" > file2.txt
python your_git.py diff file1.txt file2.txt
```

**Milestone 8 Checkpoint**:
```bash
# Test merge operations
python your_git.py merge feature
python your_git.py status  # Check for conflicts
python your_git.py log --oneline  # Verify merge commit
```

#### Debugging Tips

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|-----------------|-----|
| Hash mismatch with official Git | Incorrect object header format | Compare object content byte-by-byte | Ensure exact "blob {size}\0{content}" format |
| "Not a git repository" error | Missing or incorrect .git structure | Check directory permissions and HEAD file | Verify .git directory and all subdirectories exist |
| Index corruption | Binary format errors | Parse index with hexdump -C | Follow Git index format specification exactly |
| Merge conflicts not detected | Incorrect three-way comparison | Trace merge algorithm with debug output | Verify merge base calculation and line-by-line comparison |
| Performance problems | Inefficient algorithms | Profile with cProfile | Optimize hot paths, especially diff and merge algorithms |

The testing strategy provides comprehensive coverage while remaining practical for development workflows. The key insight is that Git's deterministic nature makes automated testing straightforward - identical operations should always produce identical results, making regression detection reliable and comprehensive compatibility verification achievable.

### Implementation Guidance

The testing implementation provides the foundation for confident development and reliable milestone verification. The comprehensive approach ensures your Git implementation works correctly both in isolation and in comparison with official Git.

#### Technology Recommendations

| Testing Component | Simple Option | Advanced Option |
|------------------|---------------|-----------------|
| Test Framework | pytest with basic assertions | pytest with custom plugins and fixtures |
| Git Execution | subprocess.run() with shell commands | GitPython library for programmatic Git operations |
| Binary Format Testing | Manual hex dump comparison | Custom binary format parsers and validators |
| Performance Testing | Basic timing with time.time() | pytest-benchmark with statistical analysis |
| Cross-Platform Testing | Manual testing on available systems | GitHub Actions matrix with multiple OS versions |

#### Recommended Testing File Structure

```
build-your-own-git/
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # pytest configuration and shared fixtures
│   ├── unit/                       # Unit tests for individual components
│   │   ├── __init__.py
│   │   ├── test_object_store.py    # ObjectStore class testing
│   │   ├── test_index.py           # Index class testing
│   │   ├── test_references.py      # ReferenceManager testing
│   │   ├── test_diff_algorithm.py  # MyersDiff algorithm testing
│   │   └── test_merge_algorithm.py # ThreeWayMerge testing
│   ├── integration/                # Integration tests comparing with official Git
│   │   ├── __init__.py
│   │   ├── test_basic_workflow.py  # init, add, commit workflow
│   │   ├── test_branch_operations.py # branch creation, switching, merging
│   │   ├── test_merge_scenarios.py # various merge conflict scenarios
│   │   └── test_compatibility.py   # comprehensive compatibility testing
│   ├── milestone/                  # Milestone-specific verification tests
│   │   ├── __init__.py
│   │   ├── test_milestone_1.py     # Repository initialization verification
│   │   ├── test_milestone_2.py     # Blob storage verification
│   │   ├── test_milestone_3.py     # Tree objects verification
│   │   ├── test_milestone_4.py     # Commit objects verification
│   │   ├── test_milestone_5.py     # References and branches verification
│   │   ├── test_milestone_6.py     # Index and staging verification
│   │   ├── test_milestone_7.py     # Diff algorithm verification
│   │   └── test_milestone_8.py     # Three-way merge verification
│   ├── fixtures/                   # Test data and repository templates
│   │   ├── sample_repos/           # Pre-built test repositories
│   │   ├── binary_files/           # Binary test files for blob storage
│   │   ├── conflict_scenarios/     # Merge conflict test cases
│   │   └── large_files/            # Performance testing files
│   └── utils/                      # Testing utilities and helpers
│       ├── __init__.py
│       ├── git_comparison.py       # Compare with official Git implementation
│       ├── repo_generator.py       # Generate test repositories programmatically
│       ├── binary_parser.py        # Parse and validate Git binary formats
│       └── performance_monitor.py  # Monitor performance and memory usage
├── src/
│   └── your_git/                   # Your Git implementation modules
│       ├── __init__.py
│       ├── repository.py           # Repository class
│       ├── object_store.py         # ObjectStore class
│       ├── index.py                # Index class
│       └── ...
└── pytest.ini                     # pytest configuration file
```

#### Infrastructure Starter Code

**Complete Git Comparison Framework**:

```python
#!/usr/bin/env python3
"""
Complete Git Comparison Framework for testing compatibility with official Git.
This framework handles environment setup, command execution, and result comparison.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum

class ComparisonResult(Enum):
    """Results of comparing custom Git with official Git."""
    IDENTICAL = "identical"
    ACCEPTABLE_DIFFERENCE = "acceptable_difference" 
    INCOMPATIBLE = "incompatible"
    ERROR = "error"

@dataclass
class CommandResult:
    """Result of running a Git command."""
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    command: List[str]

@dataclass  
class ComparisonReport:
    """Report comparing two Git implementations."""
    command: List[str]
    custom_result: CommandResult
    official_result: CommandResult
    comparison_result: ComparisonResult
    differences: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

class GitTestEnvironment:
    """Manages isolated Git testing environment with consistent configuration."""
    
    def __init__(self, custom_git_path: str, test_name: str):
        self.custom_git_path = custom_git_path
        self.official_git_path = shutil.which("git")
        self.test_name = test_name
        self.test_dir: Optional[Path] = None
        self.original_env = {}
        
        if not self.official_git_path:
            raise RuntimeError("Official Git not found in PATH")
    
    def __enter__(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp(prefix=f"git_test_{self.test_name}_"))
        
        # Save original environment
        self.original_env = {
            key: os.environ.get(key) for key in [
                'GIT_AUTHOR_NAME', 'GIT_AUTHOR_EMAIL', 'GIT_COMMITTER_NAME',
                'GIT_COMMITTER_EMAIL', 'GIT_AUTHOR_DATE', 'GIT_COMMITTER_DATE'
            ]
        }
        
        # Set consistent Git environment for reproducible results
        os.environ.update({
            'GIT_AUTHOR_NAME': 'Test Author',
            'GIT_AUTHOR_EMAIL': 'author@test.com',
            'GIT_COMMITTER_NAME': 'Test Committer', 
            'GIT_COMMITTER_EMAIL': 'committer@test.com',
            'GIT_AUTHOR_DATE': '1609459200 +0000',    # 2021-01-01 00:00:00 UTC
            'GIT_COMMITTER_DATE': '1609459200 +0000'  # 2021-01-01 00:00:00 UTC
        })
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up test environment."""
        if self.test_dir and self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        
        # Restore original environment
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    
    def run_custom_git(self, args: List[str], cwd: Optional[Path] = None) -> CommandResult:
        """Run command with custom Git implementation."""
        return self._run_git_command(self.custom_git_path, args, cwd or self.test_dir)
    
    def run_official_git(self, args: List[str], cwd: Optional[Path] = None) -> CommandResult:
        """Run command with official Git."""
        return self._run_git_command(self.official_git_path, args, cwd or self.test_dir)
    
    def _run_git_command(self, git_path: str, args: List[str], cwd: Path) -> CommandResult:
        """Execute Git command and capture results."""
        start_time = time.time()
        
        try:
            result = subprocess.run(
                [git_path] + args,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout for commands
            )
            
            execution_time = time.time() - start_time
            
            return CommandResult(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=execution_time,
                command=[git_path] + args
            )
            
        except subprocess.TimeoutExpired:
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr="Command timed out after 60 seconds",
                execution_time=60.0,
                command=[git_path] + args
            )
        except Exception as e:
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr=f"Command execution failed: {str(e)}",
                execution_time=time.time() - start_time,
                command=[git_path] + args
            )

class GitCompatibilityTester:
    """Comprehensive Git compatibility testing framework."""
    
    def __init__(self, custom_git_path: str):
        self.custom_git_path = custom_git_path
        self.comparison_reports: List[ComparisonReport] = []
        
        # Patterns for acceptable differences
        self.timestamp_patterns = [
            re.compile(r'\d{10} [+-]\d{4}'),  # Unix timestamp with timezone
            re.compile(r'\w{3} \w{3} \d{1,2} \d{2}:\d{2}:\d{2} \d{4} [+-]\d{4}'),  # Date format
        ]
        
        self.hash_pattern = re.compile(r'[0-9a-f]{40}')  # SHA-1 hashes
    
    def compare_commands(self, test_name: str, commands: List[List[str]]) -> List[ComparisonReport]:
        """Compare series of commands between implementations."""
        reports = []
        
        with GitTestEnvironment(self.custom_git_path, test_name) as env:
            for command in commands:
                custom_result = env.run_custom_git(command)
                official_result = env.run_official_git(command)
                
                report = self._compare_results(command, custom_result, official_result)
                reports.append(report)
                self.comparison_reports.append(report)
        
        return reports
    
    def _compare_results(self, command: List[str], custom: CommandResult, official: CommandResult) -> ComparisonReport:
        """Compare results from custom and official Git."""
        report = ComparisonReport(
            command=command,
            custom_result=custom,
            official_result=official,
            comparison_result=ComparisonResult.ERROR,
            differences=[],
            notes=[]
        )
        
        # Check exit codes
        if custom.exit_code != official.exit_code:
            report.differences.append(f"Exit code differs: {custom.exit_code} vs {official.exit_code}")
            report.comparison_result = ComparisonResult.INCOMPATIBLE
            return report
        
        # Normalize outputs for comparison
        custom_stdout_norm = self._normalize_output(custom.stdout)
        official_stdout_norm = self._normalize_output(official.stdout)
        
        custom_stderr_norm = self._normalize_output(custom.stderr)
        official_stderr_norm = self._normalize_output(official.stderr)
        
        # Compare normalized outputs
        stdout_match = custom_stdout_norm == official_stdout_norm
        stderr_acceptable = self._is_stderr_acceptable(custom_stderr_norm, official_stderr_norm)
        
        if stdout_match and stderr_acceptable:
            report.comparison_result = ComparisonResult.IDENTICAL
        elif self._outputs_are_functionally_equivalent(custom_stdout_norm, official_stdout_norm):
            report.comparison_result = ComparisonResult.ACCEPTABLE_DIFFERENCE
            report.notes.append("Outputs are functionally equivalent despite formatting differences")
        else:
            report.comparison_result = ComparisonResult.INCOMPATIBLE
            if not stdout_match:
                report.differences.append("stdout differs")
            if not stderr_acceptable:
                report.differences.append("stderr differs significantly")
        
        return report
    
    def _normalize_output(self, output: str) -> str:
        """Normalize output for comparison by removing acceptable variations."""
        if not output:
            return ""
        
        lines = output.strip().split('\n')
        normalized_lines = []
        
        for line in lines:
            # Replace timestamps with placeholder
            for pattern in self.timestamp_patterns:
                line = pattern.sub('[TIMESTAMP]', line)
            
            # Normalize path separators
            line = line.replace('\\', '/')
            
            # Remove trailing whitespace
            line = line.rstrip()
            
            if line:  # Skip empty lines
                normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)
    
    def _is_stderr_acceptable(self, custom_stderr: str, official_stderr: str) -> bool:
        """Determine if stderr differences are acceptable."""
        # Both empty is OK
        if not custom_stderr and not official_stderr:
            return True
        
        # Different error messages might be OK if they convey the same information
        # This is a simplified check - expand based on specific error message patterns
        return abs(len(custom_stderr) - len(official_stderr)) < 100
    
    def _outputs_are_functionally_equivalent(self, custom_output: str, official_output: str) -> bool:
        """Check if outputs are functionally equivalent despite formatting differences."""
        # Extract all SHA-1 hashes from both outputs
        custom_hashes = set(self.hash_pattern.findall(custom_output))
        official_hashes = set(self.hash_pattern.findall(official_output))
        
        # If hashes match, outputs are likely functionally equivalent
        return custom_hashes == official_hashes and len(custom_hashes) > 0
    
    def generate_compatibility_report(self) -> Dict[str, Any]:
        """Generate comprehensive compatibility report."""
        total_tests = len(self.comparison_reports)
        if total_tests == 0:
            return {"error": "No tests run"}
        
        results_summary = {}
        for result_type in ComparisonResult:
            count = sum(1 for r in self.comparison_reports if r.comparison_result == result_type)
            results_summary[result_type.value] = {
                "count": count,
                "percentage": (count / total_tests) * 100
            }
        
        failed_commands = [
            {
                "command": " ".join(r.command),
                "result": r.comparison_result.value,
                "differences": r.differences,
                "notes": r.notes
            }
            for r in self.comparison_reports
            if r.comparison_result == ComparisonResult.INCOMPATIBLE
        ]
        
        return {
            "total_tests": total_tests,
            "results_summary": results_summary,
            "compatibility_score": results_summary.get("identical", {}).get("percentage", 0),
            "failed_commands": failed_commands,
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        }
```

**Repository Test Data Generator**:

```python
#!/usr/bin/env python3
"""
Repository Test Data Generator - Creates diverse test repositories for comprehensive testing.
"""

import os
import random
import string
import struct
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import subprocess

class TestDataGenerator:
    """Generate test data and repositories for Git testing."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(tempfile.mkdtemp(prefix="git_test_data_"))
        self.base_dir.mkdir(exist_ok=True)
    
    def generate_text_content(self, lines: int = 10, line_length: int = 80) -> str:
        """Generate realistic text content for testing."""
        content_lines = []
        
        for i in range(lines):
            # Generate line with realistic word structure
            line_content = []
            current_length = 0
            
            while current_length < line_length - 10:  # Leave room for final word
                word_length = random.randint(3, 12)
                word = ''.join(random.choice(string.ascii_letters) for _ in range(word_length))
                
                if current_length + word_length + 1 <= line_length:
                    line_content.append(word)
                    current_length += word_length + 1
                else:
                    break
            
            content_lines.append(' '.join(line_content))
        
        return '\n'.join(content_lines) + '\n'
    
    def generate_binary_content(self, size: int) -> bytes:
        """Generate binary content for testing blob storage."""
        # Mix of various byte values including null bytes
        content = bytearray()
        
        for _ in range(size):
            # Include null bytes and high-bit characters
            content.append(random.randint(0, 255))
        
        return bytes(content)
    
    def create_repository_with_history(self, name: str, commit_count: int = 10) -> Path:
        """Create repository with specified number of commits."""
        repo_path = self.base_dir / name
        repo_path.mkdir(exist_ok=True)
        
        # Initialize repository
        subprocess.run(['git', 'init'], cwd=repo_path, check=True, capture_output=True)
        
        # Create commits with diverse content
        for i in range(commit_count):
            # Create or modify files
            if i == 0:
                # Initial commit with README
                (repo_path / 'README.md').write_text(f'# {name}\n\nTest repository created for Git compatibility testing.\n')
                subprocess.run(['git', 'add', 'README.md'], cwd=repo_path, check=True)
                message = 'Initial commit'
            else:
                # Add new files or modify existing ones
                if random.choice([True, False]):
                    # Add new file
                    filename = f'file_{i}.txt'
                    content = self.generate_text_content(random.randint(5, 50))
                    (repo_path / filename).write_text(content)
                    subprocess.run(['git', 'add', filename], cwd=repo_path, check=True)
                    message = f'Add {filename}'
                else:
                    # Modify existing file
                    existing_files = [f for f in repo_path.iterdir() if f.suffix == '.txt' or f.suffix == '.md']
                    if existing_files:
                        target_file = random.choice(existing_files)
                        content = self.generate_text_content(random.randint(5, 50))
                        target_file.write_text(content)
                        subprocess.run(['git', 'add', target_file.name], cwd=repo_path, check=True)
                        message = f'Modify {target_file.name}'
                    else:
                        continue
            
            # Create commit
            subprocess.run(['git', 'commit', '-m', message], cwd=repo_path, check=True, capture_output=True)
        
        return repo_path
    
    def create_merge_conflict_repository(self, name: str) -> Path:
        """Create repository with merge conflict scenario."""
        repo_path = self.base_dir / name
        repo_path.mkdir(exist_ok=True)
        
        subprocess.run(['git', 'init'], cwd=repo_path, check=True, capture_output=True)
        
        # Create initial commit
        content = "line 1\nline 2\nline 3\nline 4\nline 5\n"
        (repo_path / 'conflict.txt').write_text(content)
        subprocess.run(['git', 'add', 'conflict.txt'], cwd=repo_path, check=True)
        subprocess.run(['git', 'commit', '-m', 'Initial version'], cwd=repo_path, check=True, capture_output=True)
        
        # Create feature branch
        subprocess.run(['git', 'checkout', '-b', 'feature'], cwd=repo_path, check=True, capture_output=True)
        
        # Modify file in feature branch
        modified_content = "line 1\nmodified line 2 (feature)\nline 3\nline 4\nfeature addition\n"
        (repo_path / 'conflict.txt').write_text(modified_content)
        subprocess.run(['git', 'add', 'conflict.txt'], cwd=repo_path, check=True)
        subprocess.run(['git', 'commit', '-m', 'Feature changes'], cwd=repo_path, check=True, capture_output=True)
        
        # Switch back to main and make conflicting changes
        subprocess.run(['git', 'checkout', 'main'], cwd=repo_path, check=True, capture_output=True)
        
        conflicting_content = "line 1\nmodified line 2 (main)\nline 3\nmain modification\nline 5\n"
        (repo_path / 'conflict.txt').write_text(conflicting_content)
        subprocess.run(['git', 'add', 'conflict.txt'], cwd=repo_path, check=True)
        subprocess.run(['git', 'commit', '-m', 'Main branch changes'], cwd=repo_path, check=True, capture_output=True)
        
        return repo_path
```

#### Core Logic Skeleton Code

**Milestone Verification Test Template**:

```python
#!/usr/bin/env python3
"""
Milestone Verification Test Framework - Template for systematic milestone testing.
"""

import pytest
import tempfile
import subprocess
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

class MilestoneTestBase:
    """Base class providing common testing infrastructure for all milestones."""
    
    def __init__(self, git_implementation_path: str):
        self.git_path = git_implementation_path
        self.test_repo: Optional[Path] = None
        self.original_cwd = Path.cwd()
    
    def setup_test_repo(self, repo_name: str = "test_repo") -> Path:
        """Create isolated test repository environment."""
        # TODO 1: Create temporary directory for test repository
        # TODO 2: Set consistent Git environment variables for reproducible results
        # TODO 3: Change to test repository directory
        # TODO 4: Return path to test repository for use in tests
        # Hint: Use tempfile.mkdtemp() and os.environ for environment setup
        pass
    
    def cleanup_test_repo(self):
        """Clean up test repository and restore environment."""
        # TODO 1:

```


## Debugging Guide

> **Milestone(s):** This section applies to all eight milestones but is particularly critical for complex operations in Milestone 6 (Index/Staging Area), Milestone 7 (Diff Algorithm), and Milestone 8 (Three-Way Merge) where multiple components interact and failure modes can be subtle.

### Mental Model: The Medical Diagnostic Process

Think of debugging a Git implementation like diagnosing a patient in a hospital. When someone comes to the emergency room saying "I feel terrible," that's just the symptom—the real challenge is systematically working backward to find the root cause. A good doctor doesn't just treat the fever; they run tests to determine whether it's caused by an infection, an autoimmune disorder, or something else entirely.

Similarly, when your Git implementation fails with "object not found" or "merge conflicts detected incorrectly," these are symptoms of deeper issues in the object store, hashing logic, file I/O, or algorithm implementation. Just like medical diagnosis follows a systematic process—gather symptoms, form hypotheses, run tests, eliminate possibilities—debugging Git requires a methodical approach to trace problems back through the system's layers.

The key insight is that Git's components form a dependency chain: merge algorithms depend on diff algorithms, diff algorithms depend on tree comparisons, tree comparisons depend on object retrieval, and object retrieval depends on correct hashing and storage. A failure in any lower layer manifests as confusing symptoms in higher layers, just like how a kidney problem might first show up as fatigue or swelling.

This section provides the diagnostic tools and systematic approaches you need to isolate problems quickly and fix them at their source, rather than chasing symptoms through multiple system layers.

### Object Storage Issues

The object store is the foundation of Git's architecture, and problems here propagate throughout the entire system. Since every other operation depends on storing and retrieving objects correctly, object storage bugs often manifest as mysterious failures in seemingly unrelated components.

#### Hash Mismatches and Computation Errors

Hash mismatches are among the most common and frustrating problems in Git implementation. When your computed hash doesn't match the expected value, it indicates a fundamental error in how you're constructing or processing object content.

| Symptom | Likely Root Cause | Diagnostic Steps | Solution |
|---------|-------------------|------------------|----------|
| `compute_object_hash()` returns different hash than real Git | Missing or incorrect null byte separator | Print raw bytes before hashing, check for `\0` at correct position | Ensure format is exactly `{type} {size}\0{content}` |
| Hash computation works for text but fails for binary files | Line ending conversion corrupting binary data | Check if file reading mode is binary vs text | Always open files in binary mode (`'rb'`) |
| Same content produces different hashes on different runs | Including timestamp or metadata in hash calculation | Log exactly what bytes are being hashed | Only hash the canonical object format, not filesystem metadata |
| Objects store but can't be retrieved | Hash computed incorrectly during storage vs retrieval | Compare hashes at storage time vs retrieval time | Ensure identical hash computation algorithm in both paths |
| Repository corruption errors when using stored objects | Hash collision or truncated hash | Validate hash is exactly 40 hex characters | Check for off-by-one errors in string slicing |

The most critical debugging technique for hash issues is to log the exact byte sequence being hashed. Hash functions are deterministic—if you get different outputs, your inputs are different, even if they look identical when printed as strings.

**⚠️ Pitfall: Text Mode File Reading**
One of the most subtle bugs occurs when reading files in text mode instead of binary mode. On Windows, text mode automatically converts `\r\n` to `\n`, which changes the file's byte content and produces a different hash. Always use binary mode for all file operations, then handle line endings explicitly if needed.

**⚠️ Pitfall: Unicode Encoding Issues**
When handling file paths or commit messages with non-ASCII characters, inconsistent encoding between storage and retrieval will cause hash mismatches. Git internally uses UTF-8 for all text content, but filesystem paths might use different encodings. Establish a consistent encoding strategy early and apply it everywhere.

> **Key Insight**: Hash mismatches are never random—they indicate a systematic difference in how content is processed. The debugging approach should focus on isolating exactly where the byte streams diverge, typically by comparing hex dumps of the content at each processing stage.

#### Compression and Storage Problems

Git uses zlib compression for all stored objects, and compression-related bugs can cause object corruption, storage failures, or retrieval errors that are difficult to trace back to their source.

| Symptom | Likely Root Cause | Diagnostic Steps | Solution |
|---------|-------------------|------------------|----------|
| `zlib.error: Error -3 while decompressing` | Stored object wasn't properly compressed | Try decompressing stored file manually with Python zlib | Ensure `zlib.compress()` before writing to disk |
| Objects store successfully but retrieval returns corrupted content | Partial write or filesystem buffering issue | Check file size on disk vs expected compressed size | Use `fsync()` or atomic writes to ensure complete storage |
| Compression works for small files but fails for large ones | Memory exhaustion or buffer overflow | Monitor memory usage during compression | Stream compression for files larger than available RAM |
| Decompression succeeds but content doesn't match original | Wrong compression level or algorithm variant | Compare compression settings between storage/retrieval | Use consistent `zlib.compress()` with default level |
| Storage location exists but appears empty | Race condition in concurrent access | Check if multiple processes are accessing same object | Implement file locking around object store operations |

The key diagnostic tool for compression issues is to manually test compression and decompression outside your Git implementation. Python's `zlib` module makes this straightforward:

```python
# Test compression/decompression manually
import zlib
original = b"blob 5\x00hello"
compressed = zlib.compress(original)
decompressed = zlib.decompress(compressed)
assert original == decompressed
```

If this test passes but your Git implementation fails, the problem is in your file I/O, not the compression logic.

**⚠️ Pitfall: Forgetting to Compress Before Storage**
A common mistake is computing the hash correctly (from uncompressed content) but then storing the uncompressed content to disk. Git always stores compressed objects, so retrieval will fail when trying to decompress uncompressed data. The rule is: hash the uncompressed content, store the compressed content.

**⚠️ Pitfall: Double Compression**
Another subtle bug occurs when accidentally compressing already-compressed data. This typically happens when a higher-level function calls a lower-level storage function that also performs compression. The symptom is that stored objects become corrupted and can't be retrieved. Always maintain clear boundaries about which layer handles compression.

#### File System Permissions and Path Issues

Object storage relies heavily on filesystem operations, and permission or path-related problems can cause mysterious failures that are difficult to diagnose because error messages often don't clearly indicate the root cause.

| Symptom | Likely Root Cause | Diagnostic Steps | Solution |
|---------|-------------------|------------------|----------|
| `PermissionError` when storing objects | `.git/objects` directory has wrong permissions | Check directory permissions with `ls -la .git/objects` | Ensure `.git` hierarchy has appropriate read/write permissions |
| Objects store but can't be found during retrieval | Incorrect path construction from hash | Print full file path during storage and retrieval | Verify path format is `.git/objects/xx/yy...` with correct hash splitting |
| Storage succeeds on some systems but fails on others | Case sensitivity differences (Windows vs Linux) | Test with hashes that differ only in case | Ensure consistent case handling in hash-to-path conversion |
| Intermittent storage failures | Directory creation race condition | Check if `.git/objects/xx/` directory exists before object creation | Create intermediate directories atomically |
| Storage works but repository appears empty to other tools | Objects stored in wrong location relative to `.git` | Verify object paths relative to repository root | Ensure `.git` directory detection is working correctly |

The most effective diagnostic approach for path issues is to log the complete file paths being used and manually verify they match Git's expected structure. Use `find .git/objects -type f` to see what's actually stored and compare against your expectations.

**⚠️ Pitfall: Relative vs Absolute Paths**
Be extremely careful about working directory context when constructing object paths. If your Git implementation changes working directories, relative paths to `.git/objects` can suddenly point to the wrong location. Always resolve to absolute paths early in your program and maintain them consistently.

**⚠️ Pitfall: Directory Creation Timing**
The `.git/objects/xx` subdirectories need to be created before storing objects. A common bug is assuming these directories exist, leading to storage failures for objects whose hash prefix hasn't been seen before. Always create the directory structure on demand during object storage.

### Merge Algorithm Debugging

Merge algorithms are among the most complex components in Git, involving multiple algorithms working together: finding merge bases, comparing file trees, detecting conflicts, and combining changes. Problems here often involve subtle edge cases in graph traversal, incorrect conflict detection, or malformed output.

#### Conflict Detection Failures

Incorrect conflict detection is one of the most subtle categories of merge bugs because the symptoms often don't appear until the merge completes, and by then the root cause is buried in complex algorithm state.

| Symptom | Likely Root Cause | Diagnostic Steps | Solution |
|---------|-------------------|------------------|----------|
| Changes merge cleanly that should conflict | Incorrect base version used in three-way comparison | Print base, ours, theirs content for conflicted regions | Verify merge base calculation is finding correct common ancestor |
| Conflicts detected where files merge cleanly | Overly aggressive conflict detection algorithm | Compare line-by-line diffs manually | Check that identical changes in both branches aren't flagged as conflicts |
| Conflict markers appear in wrong locations | Line numbering bug in conflict region calculation | Log start/end line numbers for each conflict region | Ensure conflict boundaries account for previous insertions/deletions |
| Some conflicts detected but others missed | Incomplete coverage in conflict scanning | Test with files that have multiple conflict regions | Scan entire file, not just first conflict |
| Binary files show text conflict markers | Binary file detection failing | Check `is_binary_content()` with problematic files | Improve binary detection or handle binary conflicts differently |

The key diagnostic technique for conflict detection is to manually perform the three-way comparison that your algorithm should be doing. Take the base version, your changes, and their changes, and determine by hand what the result should be. Then trace through your algorithm step-by-step to find where it diverges from the correct result.

**⚠️ Pitfall: Off-by-One Line Numbering**
Conflict detection algorithms typically work with zero-indexed line arrays, but conflict markers need to reference one-indexed line numbers for human readability. Mixing these conventions leads to conflicts appearing in the wrong locations or spanning incorrect ranges.

**⚠️ Pitfall: Assuming Clean Three-Way Split**
Real merge conflicts are messier than textbook examples. A single file might have multiple conflict regions, conflicts might be adjacent (requiring marker consolidation), or one branch might delete lines while the other modifies them. Design your conflict detection to handle overlapping and adjacent conflicts gracefully.

#### Infinite Loops in Merge Base Calculation

Finding the merge base requires graph traversal of the commit history, and bugs in the traversal algorithm can cause infinite loops, incorrect results, or performance problems that make merges unusable.

| Symptom | Likely Root Cause | Diagnostic Steps | Solution |
|---------|-------------------|------------------|----------|
| Merge base calculation never terminates | Not tracking visited commits in graph traversal | Add logging to see which commits are being processed repeatedly | Maintain `visited` set to prevent revisiting same commit |
| Merge base returns wrong commit | Breadth-first search implementation bug | Manually trace commit graph and identify expected merge base | Ensure BFS queue processes commits in chronological order |
| Merge base works for simple cases but fails on complex history | Algorithm doesn't handle merge commits properly | Test with repository that has merge commits in history | Ensure algorithm follows all parent links, not just first parent |
| Performance degrades with large repositories | Inefficient graph traversal or storage | Profile memory and time usage during merge base calculation | Use efficient data structures for visited set and processing queue |
| Merge base calculation crashes on corrupted history | Missing error handling for invalid parent references | Check for commits that reference non-existent parents | Validate parent commits exist before following references |

The standard approach for debugging graph traversal is to visualize the commit graph manually (using `git log --graph --oneline`) and trace through your algorithm by hand to verify it produces the same result.

**⚠️ Pitfall: Not Handling Multiple Merge Bases**
In complex histories, two branches might have multiple common ancestors at the same distance. The merge base algorithm should return the most recent common ancestor, not just any common ancestor. This requires careful handling of the BFS termination condition.

**⚠️ Pitfall: Stack Overflow on Deep History**
Recursive implementations of merge base calculation can overflow the call stack on repositories with very deep history. Use iterative algorithms with explicit queues or stacks to handle arbitrarily deep commit graphs.

#### Corrupted Merge States

Merge operations involve multiple steps and temporary state, creating opportunities for corruption if the process is interrupted or if there are bugs in state management.

| Symptom | Likely Root Cause | Diagnostic Steps | Solution |
|---------|-------------------|------------------|----------|
| Merge appears successful but working directory is corrupted | Incomplete file updates during merge | Check if all files in merge result were written to working directory | Ensure atomic updates—write to temp files, then rename |
| Merge conflicts resolved but commit creation fails | Index not updated with merge resolution | Check index contents after conflict resolution | Update index entries for all resolved files before committing |
| Merge process can't be resumed after interruption | Missing or corrupted merge state files | Look for `.git/MERGE_HEAD` and related state files | Save merge state atomically and restore on resume |
| Merged content loses changes from one branch | Three-way merge algorithm bug | Compare merge result against manual merge | Debug three-way merge logic with simple test cases |
| Merge completes but repository is in inconsistent state | Transaction boundary not properly implemented | Verify all merge operations succeed before updating references | Implement rollback capability for failed merges |

The most effective debugging approach for merge state corruption is to implement comprehensive state validation at each step of the merge process. Before proceeding from one step to the next, verify that all previous steps completed successfully and left the repository in a consistent state.

**⚠️ Pitfall: Not Cleaning Up Merge State**
After a successful merge, temporary state files (like `.git/MERGE_HEAD`) must be cleaned up. Leaving these files around confuses subsequent operations and can cause Git tools to think a merge is still in progress.

**⚠️ Pitfall: Race Conditions in Concurrent Access**
If multiple processes try to perform merges simultaneously, or if a merge is interrupted and restarted, corrupted state can result. Implement proper locking around merge operations to prevent concurrent modifications.

### Debugging Tools and Techniques

Effective Git debugging requires a systematic toolkit of techniques for inspecting internal state, validating data structures, and tracing execution flow. Unlike application debugging, Git debugging often involves understanding the interaction between your implementation and the filesystem, as well as validating that your data structures match Git's exact specifications.

#### Inspecting Git Internals

The ability to inspect and validate Git's internal data structures is crucial for debugging, because many bugs manifest as subtle differences between your implementation's output and Git's expected formats.

| Technique | Purpose | Implementation | Usage Example |
|-----------|---------|----------------|---------------|
| Object content validation | Verify stored objects match expected format | Decompress `.git/objects/xx/yy...` files and examine raw content | Check that blob headers are exactly `blob {size}\0{content}` |
| Hash verification | Confirm object hashes are computed correctly | Recompute hash from stored content and compare to filename | Detect corruption or computation bugs |
| Index inspection | Examine staging area contents and metadata | Parse binary `.git/index` file and display entries | Debug staging/unstaging operations |
| Reference tracing | Follow symbolic and direct references | Read `.git/HEAD` and `.git/refs/heads/*` files | Debug branch switching and reference updates |
| Tree traversal validation | Verify directory structure representation | Recursively expand tree objects and compare to filesystem | Debug tree building and directory representation |

A comprehensive Git internals inspection tool should provide commands to examine each type of internal data structure. This tool becomes invaluable when your implementation produces different results than expected, because it lets you identify exactly where the differences occur.

```python
def inspect_object(object_hash: str, git_dir: Path) -> Dict[str, Any]:
    """
    Comprehensive object inspection that validates format and content.
    Returns detailed breakdown of object structure for debugging.
    """
    # TODO: Implement object retrieval and format validation
    # TODO: Parse object header and extract type, size
    # TODO: Validate content matches declared size
    # TODO: For tree objects, parse and validate entry format
    # TODO: For commit objects, parse and validate all fields
    # TODO: Return structured data for comparison with expected values
    pass
```

**⚠️ Pitfall: Endianness in Binary Parsing**
When inspecting the binary index file or other Git data structures, be aware that multi-byte integers use network byte order (big-endian). Python's `struct` module requires explicit endianness specification (`>I` for big-endian 32-bit integer).

**⚠️ Pitfall: String Encoding in Object Content**
Git objects can contain both text (UTF-8) and binary content. When displaying object content for debugging, handle encoding errors gracefully and clearly distinguish between text and binary data to avoid corrupting the debugging output itself.

#### Tracing Object Relationships

Git's object model forms a directed acyclic graph where commits point to trees, trees point to blobs and subtrees, and commits can have multiple parents. Bugs often involve incorrect relationships in this graph, so tracing these relationships is a crucial debugging technique.

| Relationship Type | Validation Method | Common Issues | Debugging Approach |
|------------------|-------------------|---------------|-------------------|
| Commit → Tree | Verify tree hash in commit object exists and is valid tree | Commit references non-existent tree | Trace tree creation during commit process |
| Tree → Blob/Subtree | Check all tree entries reference valid objects of correct type | Tree entry points to blob but declares mode as directory | Validate tree building algorithm |
| Commit → Parent | Ensure parent hashes reference valid commit objects | Commit parent chain broken or circular | Graph traversal to detect cycles |
| Branch → Commit | Verify branch reference points to valid commit | Branch reference corrupted or points to non-commit object | Check reference update logic |
| Index → Blob | Confirm staged files reference valid blob objects | Index entry hash doesn't match stored blob | Debug file staging process |

The most effective relationship tracing involves building a complete graph of your repository's objects and validating that it matches expected Git invariants. This can reveal subtle bugs like commits that reference trees built incorrectly, or index entries that point to non-existent blobs.

> **Critical Debugging Insight**: Object relationship bugs often cascade—a corrupted tree leads to a corrupted commit, which leads to a corrupted branch reference. Always trace problems back to their root cause in the dependency graph rather than fixing symptoms at higher levels.

#### Validating Repository Consistency

A well-formed Git repository must satisfy numerous consistency invariants, and validating these invariants systematically can catch bugs that would otherwise be difficult to reproduce or diagnose.

| Consistency Check | Validation Rule | Implementation | Common Violations |
|-------------------|-----------------|----------------|-------------------|
| Object reachability | All referenced objects exist and are accessible | Graph traversal from all branch heads | Dangling references, missing objects |
| Hash integrity | Stored object hash matches computed hash of content | Recompute hash for every stored object | Hash computation bugs, storage corruption |
| Reference validity | All references point to valid objects of expected type | Validate reference targets | References pointing to non-existent or wrong-type objects |
| Index consistency | All index entries reference valid blobs with correct metadata | Compare index entries to actual blobs and filesystem | Staged files don't match working directory or stored blobs |
| Tree structure validity | Tree objects properly represent directory hierarchy | Validate tree entry formats and recursive structure | Malformed tree entries, incorrect permissions |

Implementing a comprehensive repository validation function provides a systematic way to detect corruption and verify that your implementation maintains Git's invariants correctly.

```python
def validate_repository_consistency(git_dir: Path) -> List[str]:
    """
    Comprehensive repository validation that checks all Git invariants.
    Returns list of consistency violations found.
    """
    violations = []
    
    # TODO: Validate all objects in .git/objects have correct hash
    # TODO: Check that all references point to valid objects
    # TODO: Verify index entries reference existing blobs
    # TODO: Validate tree objects have proper entry format
    # TODO: Check commit objects have valid parent references
    # TODO: Ensure no circular references in commit history
    # TODO: Verify working directory matches HEAD commit + index
    
    return violations
```

**⚠️ Pitfall: Performance of Full Validation**
Repository validation can be expensive for large repositories. Implement incremental validation that focuses on recently modified objects, and provide full validation as a separate diagnostic tool rather than running it automatically.

**⚠️ Pitfall: Validation During Intermediate States**
Some Git operations go through intermediate states where the repository temporarily violates consistency rules. For example, during a merge, the working directory might not match any single commit. Design validation to account for these legitimate intermediate states.

#### Debugging Workflow Integration

Effective Git debugging requires integrating diagnostic capabilities into your normal development workflow, so that problems can be detected and diagnosed quickly rather than accumulating until they cause major failures.

| Integration Point | Diagnostic Capability | Implementation Strategy | Benefit |
|------------------|----------------------|------------------------|---------|
| Object storage | Automatic hash verification | Verify stored object can be retrieved and matches original | Catch storage bugs immediately |
| Commit creation | Tree validation | Ensure generated tree matches working directory structure | Detect tree building bugs |
| Merge operations | State consistency checks | Validate repository state at each merge step | Prevent merge corruption |
| Index operations | Metadata validation | Check that index metadata matches filesystem | Catch staging bugs early |
| Reference updates | Atomicity verification | Ensure reference updates are atomic and consistent | Prevent reference corruption |

The key insight is that debugging capabilities should be built into the core operations, not added as an afterthought. This allows problems to be detected at their source rather than discovered much later when their effects become visible.

**⚠️ Pitfall: Debug Code Affecting Performance**
Extensive validation and logging can significantly impact performance. Design debug features to be easily disabled in production builds, or implement them as optional validation passes that can be enabled when problems are suspected.

**⚠️ Pitfall: Debug Output Affecting Test Results**
When implementing diagnostic logging, ensure that debug output doesn't interfere with normal program output that tests might depend on. Use separate streams for debug output, or implement a logging system that can be configured to different levels of verbosity.

### Implementation Guidance

Building effective debugging capabilities requires a systematic approach that integrates diagnostic tools into your Git implementation from the beginning. The debugging infrastructure should be designed to help you understand not just what went wrong, but why it went wrong and how to prevent similar issues in the future.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Logging | Python's `logging` module with file handlers | Structured logging with JSON output for analysis |
| Validation | Simple assertion-based checks | Property-based testing with hypothesis |
| Object inspection | Manual hex dump utilities | Custom binary parser with formatted output |
| State debugging | Print statements with manual formatting | Interactive debugger integration with repository state |
| Performance profiling | Basic timing with `time.time()` | Full profiling with `cProfile` and memory tracking |

#### Debugging Infrastructure Starter Code

Here's a complete debugging infrastructure that provides comprehensive diagnostic capabilities for your Git implementation:

```python
import logging
import hashlib
import zlib
import struct
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

class ValidationLevel(Enum):
    MINIMAL = "minimal"      # Only critical invariants
    STANDARD = "standard"    # Common consistency checks  
    COMPREHENSIVE = "comprehensive"  # Full repository validation

@dataclass
class ValidationResult:
    is_valid: bool
    violations: List[str]
    warnings: List[str]
    validation_time: float

class GitDebugger:
    """
    Comprehensive debugging and validation toolkit for Git implementations.
    Provides object inspection, consistency checking, and diagnostic utilities.
    """
    
    def __init__(self, git_dir: Path):
        self.git_dir = git_dir
        self.objects_dir = git_dir / "objects"
        self.refs_dir = git_dir / "refs"
        self.index_path = git_dir / "index"
        self.head_path = git_dir / "HEAD"
        
        # Set up logging
        self.logger = logging.getLogger("git_debugger")
        handler = logging.FileHandler(git_dir / "debug.log")
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def inspect_object(self, object_hash: str) -> Dict[str, Any]:
        """
        Comprehensive object inspection with format validation.
        Returns detailed object structure for debugging.
        """
        try:
            object_path = self._get_object_path(object_hash)
            if not object_path.exists():
                return {"error": f"Object {object_hash} not found"}
            
            # Read and decompress object
            with open(object_path, 'rb') as f:
                compressed_data = f.read()
            
            try:
                raw_data = zlib.decompress(compressed_data)
            except zlib.error as e:
                return {"error": f"Decompression failed: {e}"}
            
            # Parse header
            null_pos = raw_data.find(b'\0')
            if null_pos == -1:
                return {"error": "Invalid object format: no null separator"}
            
            header = raw_data[:null_pos].decode('utf-8')
            content = raw_data[null_pos + 1:]
            
            # Parse object type and size
            try:
                obj_type, size_str = header.split(' ', 1)
                declared_size = int(size_str)
            except ValueError:
                return {"error": f"Invalid header format: {header}"}
            
            # Validate size
            if len(content) != declared_size:
                return {
                    "error": f"Size mismatch: declared {declared_size}, actual {len(content)}"
                }
            
            # Verify hash
            expected_hash = hashlib.sha1(raw_data).hexdigest()
            if expected_hash != object_hash:
                return {
                    "error": f"Hash mismatch: expected {object_hash}, computed {expected_hash}"
                }
            
            result = {
                "hash": object_hash,
                "type": obj_type,
                "size": declared_size,
                "content_preview": self._preview_content(content, obj_type),
                "valid": True
            }
            
            # Type-specific parsing
            if obj_type == "tree":
                result["entries"] = self._parse_tree_entries(content)
            elif obj_type == "commit":
                result["commit_info"] = self._parse_commit_info(content)
            
            return result
            
        except Exception as e:
            return {"error": f"Inspection failed: {e}"}

    def validate_repository(self, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """
        Comprehensive repository validation with configurable depth.
        """
        start_time = time.time()
        violations = []
        warnings = []
        
        try:
            if level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
                violations.extend(self._validate_object_store())
                violations.extend(self._validate_references())
                violations.extend(self._validate_index())
            
            if level == ValidationLevel.COMPREHENSIVE:
                violations.extend(self._validate_object_relationships())
                violations.extend(self._validate_working_directory())
                warnings.extend(self._check_performance_issues())
            
            validation_time = time.time() - start_time
            is_valid = len(violations) == 0
            
            return ValidationResult(is_valid, violations, warnings, validation_time)
            
        except Exception as e:
            violations.append(f"Validation failed with error: {e}")
            return ValidationResult(False, violations, warnings, time.time() - start_time)

    def trace_object_relationships(self, start_hash: str, max_depth: int = 10) -> Dict[str, Any]:
        """
        Trace object relationships from a starting object (typically a commit).
        Returns graph structure showing all reachable objects.
        """
        visited = set()
        relationships = {}
        queue = [(start_hash, 0)]
        
        while queue and len(visited) < 1000:  # Safety limit
            obj_hash, depth = queue.pop(0)
            
            if obj_hash in visited or depth > max_depth:
                continue
            
            visited.add(obj_hash)
            obj_info = self.inspect_object(obj_hash)
            
            if "error" in obj_info:
                relationships[obj_hash] = {"error": obj_info["error"], "depth": depth}
                continue
            
            relationships[obj_hash] = {
                "type": obj_info["type"],
                "depth": depth,
                "references": []
            }
            
            # Find referenced objects
            if obj_info["type"] == "commit":
                commit_info = obj_info.get("commit_info", {})
                if "tree" in commit_info:
                    queue.append((commit_info["tree"], depth + 1))
                    relationships[obj_hash]["references"].append(commit_info["tree"])
                
                for parent in commit_info.get("parents", []):
                    queue.append((parent, depth + 1))
                    relationships[obj_hash]["references"].append(parent)
            
            elif obj_info["type"] == "tree":
                for entry in obj_info.get("entries", []):
                    entry_hash = entry.get("hash")
                    if entry_hash:
                        queue.append((entry_hash, depth + 1))
                        relationships[obj_hash]["references"].append(entry_hash)
        
        return {
            "start_object": start_hash,
            "total_objects": len(relationships),
            "max_depth_reached": max(r.get("depth", 0) for r in relationships.values()),
            "relationships": relationships
        }

    def diagnose_merge_failure(self, our_commit: str, their_commit: str) -> Dict[str, Any]:
        """
        Comprehensive merge failure diagnosis.
        """
        diagnosis = {
            "our_commit": our_commit,
            "their_commit": their_commit,
            "issues": [],
            "recommendations": []
        }
        
        # Validate input commits
        our_info = self.inspect_object(our_commit)
        their_info = self.inspect_object(their_commit)
        
        if "error" in our_info:
            diagnosis["issues"].append(f"Our commit invalid: {our_info['error']}")
        if "error" in their_info:
            diagnosis["issues"].append(f"Their commit invalid: {their_info['error']}")
        
        if diagnosis["issues"]:
            return diagnosis
        
        # Try to find merge base
        try:
            merge_base = self._find_merge_base_debug(our_commit, their_commit)
            if merge_base:
                diagnosis["merge_base"] = merge_base
                base_info = self.inspect_object(merge_base)
                if "error" in base_info:
                    diagnosis["issues"].append(f"Merge base corrupted: {base_info['error']}")
            else:
                diagnosis["issues"].append("No common ancestor found")
                diagnosis["recommendations"].append("Check if commits are from same repository")
        except Exception as e:
            diagnosis["issues"].append(f"Merge base calculation failed: {e}")
        
        # Analyze tree differences
        try:
            our_tree = our_info["commit_info"]["tree"]
            their_tree = their_info["commit_info"]["tree"]
            
            tree_diff = self._analyze_tree_differences(our_tree, their_tree)
            diagnosis["tree_analysis"] = tree_diff
            
            if tree_diff["binary_conflicts"] > 0:
                diagnosis["recommendations"].append("Binary file conflicts require manual resolution")
            
            if tree_diff["large_files"] > 0:
                diagnosis["recommendations"].append("Large file conflicts may cause performance issues")
                
        except Exception as e:
            diagnosis["issues"].append(f"Tree analysis failed: {e}")
        
        return diagnosis

    # Helper methods for internal functionality
    
    def _get_object_path(self, object_hash: str) -> Path:
        """Convert object hash to filesystem path."""
        return self.objects_dir / object_hash[:2] / object_hash[2:]
    
    def _preview_content(self, content: bytes, obj_type: str) -> str:
        """Generate human-readable content preview."""
        if obj_type == "blob":
            # Try to decode as text, fall back to hex for binary
            try:
                text = content.decode('utf-8')
                if len(text) > 200:
                    return text[:200] + "..."
                return text
            except UnicodeDecodeError:
                return f"<binary content, {len(content)} bytes>"
        else:
            # For tree and commit objects, always try text first
            try:
                return content.decode('utf-8')[:500]
            except UnicodeDecodeError:
                return content.hex()[:100] + "..."

    def _parse_tree_entries(self, content: bytes) -> List[Dict[str, str]]:
        """Parse tree object entries for inspection."""
        entries = []
        offset = 0
        
        while offset < len(content):
            # Find null terminator for mode/name
            null_pos = content.find(b'\0', offset)
            if null_pos == -1:
                break
                
            mode_name = content[offset:null_pos].decode('utf-8')
            try:
                mode, name = mode_name.split(' ', 1)
            except ValueError:
                break
                
            # Extract 20-byte hash
            if null_pos + 21 > len(content):
                break
                
            hash_bytes = content[null_pos + 1:null_pos + 21]
            hash_hex = hash_bytes.hex()
            
            entries.append({
                "mode": mode,
                "name": name,
                "hash": hash_hex
            })
            
            offset = null_pos + 21
        
        return entries

    def _parse_commit_info(self, content: bytes) -> Dict[str, Any]:
        """Parse commit object for inspection."""
        try:
            text = content.decode('utf-8')
            lines = text.split('\n')
            
            info = {"parents": []}
            message_start = None
            
            for i, line in enumerate(lines):
                if not line.strip():
                    message_start = i + 1
                    break
                    
                if line.startswith('tree '):
                    info["tree"] = line[5:]
                elif line.startswith('parent '):
                    info["parents"].append(line[7:])
                elif line.startswith('author '):
                    info["author"] = line[7:]
                elif line.startswith('committer '):
                    info["committer"] = line[10:]
            
            if message_start:
                info["message"] = '\n'.join(lines[message_start:])
            
            return info
        except UnicodeDecodeError:
            return {"error": "Commit content is not valid UTF-8"}

    def _validate_object_store(self) -> List[str]:
        """Validate object store consistency."""
        violations = []
        
        if not self.objects_dir.exists():
            violations.append("Objects directory missing")
            return violations
        
        # Check object files
        for obj_dir in self.objects_dir.iterdir():
            if not obj_dir.is_dir() or len(obj_dir.name) != 2:
                continue
                
            for obj_file in obj_dir.iterdir():
                if not obj_file.is_file():
                    continue
                    
                full_hash = obj_dir.name + obj_file.name
                if len(full_hash) != 40:
                    violations.append(f"Invalid object filename: {full_hash}")
                    continue
                
                # Validate object can be read and hash matches
                obj_info = self.inspect_object(full_hash)
                if "error" in obj_info:
                    violations.append(f"Corrupted object {full_hash}: {obj_info['error']}")
        
        return violations

    def _validate_references(self) -> List[str]:
        """Validate reference consistency."""
        violations = []
        
        # Check HEAD
        if not self.head_path.exists():
            violations.append("HEAD file missing")
        else:
            try:
                head_content = self.head_path.read_text().strip()
                if head_content.startswith('ref: '):
                    ref_path = self.git_dir / head_content[5:]
                    if not ref_path.exists():
                        violations.append(f"HEAD points to non-existent ref: {head_content[5:]}")
                elif len(head_content) == 40:
                    # Direct hash - validate object exists
                    if not self._get_object_path(head_content).exists():
                        violations.append(f"HEAD points to non-existent object: {head_content}")
                else:
                    violations.append(f"Invalid HEAD content: {head_content}")
            except Exception as e:
                violations.append(f"Cannot read HEAD: {e}")
        
        return violations

    def _validate_index(self) -> List[str]:
        """Validate index consistency."""
        violations = []
        
        if not self.index_path.exists():
            return []  # Index is optional
        
        try:
            with open(self.index_path, 'rb') as f:
                # Read index header
                signature = f.read(4)
                if signature != b'DIRC':
                    violations.append("Invalid index signature")
                    return violations
                
                version = struct.unpack('>I', f.read(4))[0]
                if version != 2:
                    violations.append(f"Unsupported index version: {version}")
                    return violations
                
                entry_count = struct.unpack('>I', f.read(4))[0]
                
                # Validate each entry references existing blob
                for i in range(entry_count):
                    try:
                        entry_data = f.read(62)  # Fixed portion of entry
                        if len(entry_data) != 62:
                            violations.append(f"Truncated index entry {i}")
                            break
                        
                        # Extract hash (20 bytes starting at offset 40)
                        hash_bytes = entry_data[40:60]
                        hash_hex = hash_bytes.hex()
                        
                        # Check if blob exists
                        if not self._get_object_path(hash_hex).exists():
                            violations.append(f"Index entry {i} references non-existent blob: {hash_hex}")
                        
                        # Skip variable-length path name
                        flags = struct.unpack('>H', entry_data[60:62])[0]
                        name_length = flags & 0xfff
                        f.read(name_length)
                        
                        # Skip padding
                        total_read = 62 + name_length
                        padding = (8 - (total_read % 8)) % 8
                        f.read(padding)
                        
                    except Exception as e:
                        violations.append(f"Error reading index entry {i}: {e}")
                        break
        
        except Exception as e:
            violations.append(f"Cannot read index: {e}")
        
        return violations
```

This debugging infrastructure provides comprehensive diagnostic capabilities that will help you identify and resolve issues throughout your Git implementation development.


## Future Extensions

> **Milestone(s):** This section goes beyond the eight core milestones, exploring advanced Git features that could be added to extend the basic implementation. Understanding these extensions helps architects plan for scalability and provides a roadmap for evolving the system.

Building a basic Git implementation teaches the fundamental concepts of distributed version control, but Git's true power comes from its advanced features that enable large-scale collaboration and efficient storage. This section explores two critical extensions that transform a working but basic Git implementation into a production-ready system: remote repository support for distributed collaboration and pack file optimization for efficient storage.

### Mental Model: The Local Shop Goes Global

Think of our current Git implementation as a successful local bookstore. The store has excellent organization (object storage), efficient inventory management (index), clear categorization (references), and a good return policy (merge conflict resolution). Customers love shopping there, but they can only visit in person.

Now imagine the bookstore wants to expand globally. They need two major capabilities: **shipping and receiving** (remote repository support) to exchange books with other locations, and **warehouse optimization** (pack files) to store thousands of books efficiently instead of keeping each book in its own individual box.

Remote repository support is like building a shipping network - books (objects) need to be packaged, addressed, sent, received, unpacked, and integrated into the local inventory. Pack file optimization is like switching from individual book boxes to efficient warehouse shelving - instead of storing each book separately, related books are grouped together and compressed to save space.

### Remote Repository Support

Remote repository support transforms Git from a local version control system into a distributed collaboration platform. This extension adds the ability to synchronize changes with other repositories through push, pull, and fetch operations while maintaining the integrity of the content-addressable object store.

#### Understanding Remote Operations

The fundamental challenge of remote repository support is **object synchronization** - determining which objects exist in each repository and transferring only the missing ones. Unlike simple file synchronization, Git's content-addressable storage enables sophisticated optimizations because identical content always has identical hashes regardless of repository location.

Each remote operation follows a similar pattern: **discovery, negotiation, transfer, and integration**. The discovery phase determines what references (branches and tags) exist in each repository. The negotiation phase computes the minimal set of objects needed to bring repositories into sync. The transfer phase moves objects efficiently across the network. The integration phase updates local references and working directory to reflect the new state.

> **Key Insight**: Remote operations are fundamentally about moving objects and updating references. The content-addressable nature of Git's object store makes this conceptually simple - if two repositories have the same object hash, they have identical content. The complexity lies in efficiently determining what needs to be transferred and handling reference conflicts.

#### Remote Repository Data Model

The remote system extends our existing architecture with several new components that manage distributed state and network operations.

| Component | Purpose | Storage Location | Key Responsibilities |
|-----------|---------|------------------|----------------------|
| `RemoteRepository` | Represents connection to remote Git repository | `.git/config` file | URL management, authentication, protocol selection |
| `RemoteReferenceStore` | Tracks remote branch and tag states | `.git/refs/remotes/` | Remote reference caching, tracking branch relationships |
| `TransferProtocol` | Handles network communication with remote repositories | Memory/network | Object discovery, negotiation, data transfer |
| `PackProtocol` | Efficient object transfer format for network operations | Memory/temporary files | Object bundling, progress reporting, error recovery |
| `RefSpec` | Maps local and remote reference names | `.git/config` | Branch mapping rules, push/fetch behavior configuration |

#### Fetch Operation Architecture

The **fetch operation** retrieves objects and references from a remote repository without modifying the local working directory or current branch. This is the foundation operation that pull builds upon.

The fetch algorithm follows these detailed steps:

1. **Remote Discovery**: Connect to the remote repository and retrieve the complete list of references (branches and tags) with their current commit hashes. This uses Git's upload-pack protocol to get an advertisement of all available references.

2. **Reference Analysis**: Compare remote references against local remote-tracking branches stored in `.git/refs/remotes/origin/` to determine which references have changed since the last fetch.

3. **Object Graph Walking**: Starting from new or updated remote references, walk backward through the commit graph to identify all objects (commits, trees, blobs) that exist in the remote repository but are missing from the local object store.

4. **Want/Have Negotiation**: Send the remote repository a list of object hashes we "want" (missing objects) and "have" (existing objects). The remote responds with the minimal set of objects needed to satisfy our wants without sending objects we already possess.

5. **Pack File Reception**: Receive a compressed pack file containing all requested objects in dependency order. Objects are transmitted in a format that allows streaming processing without requiring the entire pack file in memory.

6. **Object Extraction**: Decompress and validate each object from the pack file, then store it in the local object store using the standard `.git/objects/xx/yy...` directory structure.

7. **Reference Updates**: Update remote-tracking branches in `.git/refs/remotes/origin/` to reflect the current state of the remote repository. These references show what the remote looked like at fetch time.

8. **Fast-Forward Analysis**: For each local branch that tracks a remote branch, determine if the local branch can be fast-forwarded to match the remote state (when the local branch is a direct ancestor of the remote).

#### Push Operation Architecture

The **push operation** sends local objects and reference updates to a remote repository, requiring careful handling of concurrent modifications and reference conflicts.

Push is more complex than fetch because it modifies the remote repository state, which may conflict with concurrent changes from other users. The push algorithm implements these steps:

1. **Pre-Push Validation**: Verify that all local references being pushed point to valid commits in the local object store and that the user has appropriate permissions for the target remote repository.

2. **Remote Reference Check**: Retrieve current remote references to detect if any branches being pushed have been modified by other users since the last fetch. This prevents accidentally overwriting concurrent work.

3. **Fast-Forward Verification**: For each branch being pushed, verify that the push represents a fast-forward update (the remote branch is an ancestor of the local branch) unless force push is explicitly enabled.

4. **Object Dependency Analysis**: Walk the commit graph from local branch tips back to the last known remote state, collecting all objects that need to be transmitted to the remote repository.

5. **Pack File Generation**: Create a pack file containing all objects that exist locally but are missing from the remote repository. Objects are compressed and ordered to optimize transmission efficiency.

6. **Atomic Push Transaction**: Send the pack file and reference updates to the remote repository as an atomic transaction. Either all references update successfully, or none do, preventing partial updates that could corrupt remote repository state.

7. **Reference Conflict Resolution**: If remote references have changed during the push operation, abort the push and require the user to fetch latest changes and resolve conflicts locally before retrying.

8. **Success Confirmation**: Receive confirmation from the remote repository that all objects were stored successfully and all references were updated atomically.

#### Remote Reference Management

Remote-tracking branches solve the problem of maintaining local knowledge about remote repository state without interfering with local development. These references use a separate namespace to avoid conflicts with local branches.

| Reference Type | Storage Location | Format | Purpose |
|----------------|------------------|---------|---------|
| Local Branch | `.git/refs/heads/feature` | `commit_hash` | Local development work |
| Remote Branch | `.git/refs/remotes/origin/feature` | `commit_hash` | Last known state of remote branch |
| Tracking Relationship | `.git/config` | `[branch "feature"] remote = origin merge = refs/heads/feature` | Links local and remote branches |

The tracking relationship enables Git to provide helpful information about branch divergence - whether the local branch is ahead, behind, or has diverged from its remote counterpart.

#### Pull Operation as Fetch + Merge

The **pull operation** combines fetch with automatic merging, implementing the common workflow of retrieving remote changes and integrating them into the current local branch.

Pull executes as two distinct phases:

1. **Fetch Phase**: Execute a complete fetch operation to retrieve all remote objects and update remote-tracking branches. This ensures local knowledge of remote state is current.

2. **Integration Phase**: Merge the updated remote-tracking branch into the current local branch using the three-way merge algorithm implemented in Milestone 8. If conflicts arise, they are presented to the user for manual resolution.

The integration phase supports multiple strategies:
- **Merge Strategy**: Create a merge commit that combines local and remote changes, preserving the parallel development history
- **Rebase Strategy**: Replay local commits on top of the remote branch, creating a linear history without merge commits
- **Fast-Forward Strategy**: If the local branch hasn't diverged, simply update the local branch pointer to match the remote

> **Decision: Pull Integration Strategy**
> - **Context**: Pull operations need to integrate remote changes into local branches, but different projects prefer different history structures
> - **Options Considered**: Always merge, always rebase, configurable per repository, configurable per branch
> - **Decision**: Default to merge with configuration options for rebase and fast-forward-only modes
> - **Rationale**: Merge preserves complete history and is safest for beginners, while advanced users can configure alternative strategies per their workflow needs
> - **Consequences**: More complex pull implementation but supports both simple and advanced workflows without forcing a single approach

#### Protocol Implementation Considerations

Git's network protocols have evolved to optimize performance and security for distributed collaboration. The modern protocol supports multiple transport layers and advanced features.

| Protocol Feature | HTTP(S) Transport | SSH Transport | Local Filesystem |
|------------------|-------------------|---------------|-------------------|
| Authentication | Username/password, tokens | SSH keys, agents | File permissions |
| Encryption | TLS | SSH tunnel | None needed |
| Firewall Friendly | Yes (port 80/443) | No (port 22) | N/A |
| Performance | Good | Excellent | Excellent |
| Setup Complexity | Medium | High | Low |

The protocol implements several optimizations:
- **Multi-round Negotiation**: Multiple rounds of want/have negotiation minimize transferred objects
- **Thin Packs**: Pack files can reference objects assumed to exist in the destination repository
- **Progress Reporting**: Long operations provide progress feedback for user experience
- **Resumable Transfers**: Large transfers can resume after network interruptions

#### Architecture Decision Records for Remote Support

> **Decision: Remote Configuration Storage**
> - **Context**: Remote repository URLs, authentication, and mapping rules need persistent storage
> - **Options Considered**: Separate remote config file, embed in main .git/config, database storage
> - **Decision**: Extend existing .git/config file with [remote] and [branch] sections
> - **Rationale**: Maintains compatibility with existing Git configuration patterns, supports version control of configuration, enables per-repository customization
> - **Consequences**: Config file becomes more complex but remains human-readable and editable

> **Decision: Object Transfer Batching**
> - **Context**: Large repositories may have thousands of objects to transfer, requiring efficient batching
> - **Options Considered**: Transfer all objects in single batch, fixed-size batches, adaptive batching based on network conditions
> - **Decision**: Implement adaptive batching with fallback to smaller batches on network errors
> - **Rationale**: Optimizes for fast networks while providing resilience on unreliable connections
> - **Consequences**: More complex transfer logic but better user experience across different network conditions

### Pack File Optimization

Pack files transform Git's storage efficiency by replacing individual compressed objects with highly optimized compressed bundles that use delta compression and cross-object deduplication.

#### Understanding the Storage Efficiency Problem

Our current object store implementation stores each blob, tree, and commit as an individual zlib-compressed file. While this provides excellent simplicity and reliability, it has significant storage inefficiencies for real-world repositories:

- **Similar Content Duplication**: Multiple versions of the same file share most content but are stored completely separately. A single-line change to a large file results in two complete compressed copies.
- **Small Object Overhead**: Each object requires a minimum file system block (typically 4KB), so a 100-byte commit object wastes ~3900 bytes of storage.
- **Directory Fragmentation**: Thousands of small files in `.git/objects/` create file system performance problems and backup inefficiencies.

Pack files solve these problems by implementing **delta compression** - storing one version of a file completely, then storing subsequent versions as compact diffs against the base version.

#### Mental Model: The Efficient Warehouse

Think of our current object storage as a warehouse where every item, no matter how similar, gets its own labeled box. If you stock 50 different versions of the same book that only differ in minor edits, you need 50 complete boxes.

Pack files are like an efficient warehouse manager who says: "Let's keep one complete copy of the first edition, then for each subsequent edition, just store a note saying 'same as first edition but change page 247 paragraph 2'." The warehouse still contains all versions, but uses dramatically less space.

The warehouse manager also batches related items together - instead of individual boxes scattered throughout the warehouse, related items are grouped on the same shelf for efficient access.

#### Pack File Data Structure

A pack file consists of a header, a series of packed objects, and an index for efficient random access. The structure optimizes for both storage efficiency and access performance.

| Component | Size | Purpose | Format |
|-----------|------|---------|---------|
| Pack Header | 12 bytes | File format identification | `PACK` signature + version + object count |
| Packed Objects | Variable | Compressed object data with delta chains | Type + size + compressed data or delta instructions |
| Pack Checksum | 20 bytes | Data integrity verification | SHA-1 hash of all preceding pack data |
| Pack Index | Separate file | Fast object lookup by hash | Sorted hash table with offset pointers |

#### Delta Compression Algorithm

Delta compression identifies similar objects and stores only the differences, achieving dramatic space savings for repositories with many similar files.

The delta compression algorithm works as follows:

1. **Base Object Selection**: For each object being packed, identify potential base objects that share significant content. Heuristics prioritize objects with similar paths, sizes, and content signatures.

2. **Delta Generation**: Create a delta script that transforms the base object into the target object. The delta consists of copy instructions (copy N bytes from offset X in base) and insert instructions (insert these literal bytes).

3. **Chain Length Management**: Limit delta chains to prevent excessive decompression overhead. If Object C is a delta of Object B, which is a delta of Object A, accessing Object C requires decompressing A, then B, then C.

4. **Size Efficiency Verification**: Only use delta compression when it actually saves space. Sometimes small files or completely different files are more efficient stored as complete objects.

5. **Access Pattern Optimization**: Place frequently accessed objects (recent commits, popular branches) early in the pack file and avoid long delta chains for objects likely to be accessed frequently.

#### Pack Index Structure

The pack index enables efficient random access to objects within pack files without scanning the entire pack. The index uses a sophisticated multi-level structure optimized for both space and lookup performance.

| Index Level | Purpose | Structure | Access Time |
|-------------|---------|-----------|-------------|
| Hash Fanout Table | First-level lookup by hash prefix | 256 entries (one per hash byte value) | O(1) |
| Object Hash Table | Sorted list of all object hashes in pack | 20-byte SHA-1 hashes in sorted order | O(log n) |
| CRC Checksums | Integrity verification for individual objects | 4-byte CRC per object | O(1) |
| Pack Offsets | File positions of objects within pack file | 4 or 8-byte offsets depending on pack size | O(1) |

The index lookup algorithm:
1. Use the first byte of the target object hash to index into the fanout table
2. Binary search the hash table segment identified by the fanout table
3. If hash is found, retrieve the corresponding pack offset
4. Seek to that offset in the pack file and read the object data
5. Verify object integrity using the stored CRC checksum

#### Garbage Collection and Pack Generation

Garbage collection transforms loose objects into efficient pack files and removes objects that are no longer reachable from any reference.

The garbage collection process implements these phases:

1. **Reachability Analysis**: Starting from all references (branches, tags, HEAD), walk the complete object graph to identify all reachable objects. Any object not reachable from a reference is considered garbage.

2. **Pack Candidate Selection**: Group objects into logical packs based on access patterns. Typically, recent objects accessed together are packed together, while historical objects form separate packs.

3. **Delta Base Selection**: For each object, identify optimal delta bases using heuristics based on path similarity, size, and content. This is computationally expensive but critical for pack efficiency.

4. **Pack Generation**: Create pack files with objects ordered to optimize delta compression and access patterns. Recent commits and trees are typically placed early in packs.

5. **Index Generation**: Build pack index files that enable efficient random access to packed objects without decompressing the entire pack.

6. **Atomic Replacement**: Atomically replace loose objects with pack files, ensuring repository integrity is maintained even if the process is interrupted.

7. **Cleanup**: Remove loose objects that are now redundant with packed versions, and remove old pack files that have been superseded.

#### Integration with Existing Object Store

Pack file support extends our existing `ObjectStore` component without breaking existing functionality. The enhanced object store checks both loose objects and pack files when retrieving objects.

| Operation | Loose Objects | Pack Files | Combined Behavior |
|-----------|---------------|------------|-------------------|
| Store Object | Write to `.git/objects/xx/yy...` | N/A (packs are read-only after creation) | New objects always stored loose initially |
| Retrieve Object | Check `.git/objects/` first | Search pack indexes if not found loose | Transparent to callers |
| Object Exists | Scan directory | Search pack indexes | Return true if found in either location |
| List All Objects | Directory traversal | Parse all pack indexes | Union of loose and packed objects |

The object retrieval algorithm becomes:
1. Check for loose object at `.git/objects/xx/yy...`
2. If found, decompress and return
3. If not found, search all pack indexes for the object hash
4. If found in pack, seek to pack offset and decompress object (following delta chain if necessary)
5. If not found anywhere, return object not found error

#### Architecture Decision Records for Pack Files

> **Decision: Delta Chain Length Limits**
> - **Context**: Long delta chains save more space but increase access time for individual objects
> - **Options Considered**: No limit (maximum compression), fixed limit of 10, adaptive limit based on object access frequency
> - **Decision**: Implement fixed limit of 50 with shorter limits for frequently accessed objects
> - **Rationale**: Balances storage efficiency with access performance, prevents pathological cases where accessing one object requires decompressing hundreds of deltas
> - **Consequences**: Some compression efficiency is sacrificed for predictable access performance

> **Decision: Pack File Size Limits**
> - **Context**: Large pack files are more efficient but harder to transfer and backup
> - **Options Considered**: No size limit, 2GB limit (32-bit offset compatibility), 4GB limit, multiple size tiers
> - **Decision**: 2GB size limit with automatic pack splitting for larger repositories
> - **Rationale**: Maintains compatibility with systems that have 32-bit limitations while supporting large repositories through multiple packs
> - **Consequences**: More complex pack management but broader system compatibility

#### Performance Impact Analysis

Pack file optimization provides dramatic improvements in storage efficiency and some operations, while slightly impacting others.

| Operation | Storage Impact | Performance Impact | Network Impact |
|-----------|----------------|-------------------|----------------|
| Repository Size | 60-90% reduction typical | N/A | Faster clones and fetches |
| Object Retrieval | N/A | Slightly slower (delta decompression) | N/A |
| Garbage Collection | Enables cleanup of unreachable objects | CPU intensive during pack generation | N/A |
| Backup/Transfer | Much smaller repository size | N/A | Dramatically faster |

The storage savings are particularly dramatic for repositories with:
- Many small commits (reduces per-object overhead)
- Large files with incremental changes (delta compression)
- Long history (more opportunities for similar content)

Typical compression ratios:
- **Text-heavy repositories**: 70-85% size reduction
- **Binary-heavy repositories**: 30-60% size reduction  
- **Mixed repositories**: 60-80% size reduction

### Integration Considerations

Both remote repository support and pack file optimization integrate cleanly with the existing eight-milestone architecture without requiring fundamental changes to core components.

#### Compatibility with Existing Components

The extensions maintain full compatibility with existing functionality:

- **Object Store**: Enhanced to check both loose objects and pack files transparently
- **References**: Remote-tracking branches use the same reference format as local branches
- **Index**: Unchanged - staging area works identically with packed and loose objects
- **Merge Algorithm**: Unchanged - operates on object content regardless of storage format
- **Diff Algorithm**: Unchanged - compares object content regardless of storage location

#### Migration Strategy

Implementing these extensions follows a progressive enhancement approach:

1. **Phase 1**: Add remote repository configuration and basic fetch/push without pack file support
2. **Phase 2**: Implement pack file reading to support repositories that already contain pack files
3. **Phase 3**: Add pack file generation during garbage collection
4. **Phase 4**: Optimize remote operations to use pack protocol for efficient transfers

This approach allows the system to evolve incrementally while maintaining full functionality at each stage.

### Common Pitfalls

⚠️ **Pitfall: Ignoring Network Failure Recovery**
When implementing remote operations, developers often assume network operations will succeed. In reality, network transfers can fail at any point, leaving repositories in inconsistent states. Always implement atomic operations where either all references update successfully or none do. Use temporary files for pack reception and atomic rename operations for reference updates.

⚠️ **Pitfall: Delta Chain Performance Degradation**
Long delta chains can make individual object access extremely slow, especially for frequently accessed objects like recent commits. Implement delta chain length limits and prioritize recent/frequently accessed objects as delta bases rather than targets. Monitor access patterns and repack when performance degrades.

⚠️ **Pitfall: Pack Index Corruption Handling**
Pack indexes can become corrupted, making objects inaccessible even though they exist in the pack file. Always verify pack index integrity on startup and implement index regeneration from pack files. Never trust index entries without verifying the corresponding pack file data.

⚠️ **Pitfall: Concurrent Access to Pack Files**
Multiple processes may try to read pack files simultaneously, or garbage collection may try to delete packs while they're being read. Implement proper file locking for pack operations and use atomic operations for pack replacement. Consider delayed deletion of old pack files to allow ongoing operations to complete.

### Implementation Guidance

The implementation of these advanced features requires careful attention to network protocols, binary file formats, and concurrent access patterns. The following guidance provides a foundation for building production-ready extensions.

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| HTTP Transport | `urllib3` with basic auth | `requests` with session management and retry logic |
| SSH Transport | `subprocess` calls to `ssh` command | `paramiko` library for pure Python SSH |
| Pack File I/O | `struct` module for binary formats | `mmap` for efficient large file access |
| Delta Compression | Custom implementation following Git format | `python-delta` library if available |
| Network Protocol | Simple HTTP POST/GET | Full Git smart HTTP protocol |

#### Recommended File Structure

```python
# Remote repository support
remote/
    __init__.py
    protocols/
        __init__.py
        http_transport.py      # HTTP/HTTPS transport implementation
        ssh_transport.py       # SSH transport implementation  
        local_transport.py     # Local filesystem transport
    operations/
        __init__.py
        fetch.py              # Fetch operation implementation
        push.py               # Push operation implementation
        pull.py               # Pull operation (fetch + merge)
    config/
        __init__.py
        remote_config.py      # Remote repository configuration
        refspec.py            # Reference specification parsing

# Pack file support  
pack/
    __init__.py
    pack_file.py             # Pack file reading and writing
    pack_index.py            # Pack index management
    delta.py                 # Delta compression implementation
    garbage_collection.py    # GC and pack generation
```

#### Infrastructure Starter Code

Here's complete infrastructure code for basic remote configuration management:

```python
#!/usr/bin/env python3
"""
Remote repository configuration management.
Complete implementation ready for use.
"""

import configparser
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

class RemoteConfig:
    """Manages remote repository configuration in .git/config file."""
    
    def __init__(self, git_dir: Path):
        self.git_dir = git_dir
        self.config_path = git_dir / "config"
        self.config = configparser.ConfigParser()
        if self.config_path.exists():
            self.config.read(self.config_path)
    
    def add_remote(self, name: str, url: str, fetch_refspec: str = None) -> bool:
        """Add a new remote repository configuration."""
        if fetch_refspec is None:
            fetch_refspec = f"+refs/heads/*:refs/remotes/{name}/*"
        
        section_name = f"remote \"{name}\""
        self.config[section_name] = {
            'url': url,
            'fetch': fetch_refspec
        }
        self._save_config()
        return True
    
    def remove_remote(self, name: str) -> bool:
        """Remove a remote repository configuration."""
        section_name = f"remote \"{name}\""
        if section_name in self.config:
            self.config.remove_section(section_name)
            self._save_config()
            return True
        return False
    
    def get_remote_url(self, name: str) -> Optional[str]:
        """Get URL for named remote."""
        section_name = f"remote \"{name}\""
        if section_name in self.config:
            return self.config[section_name].get('url')
        return None
    
    def list_remotes(self) -> List[str]:
        """List all configured remote names."""
        remotes = []
        for section in self.config.sections():
            if section.startswith('remote "'):
                # Extract name from 'remote "origin"' format
                name = section[8:-1]  # Remove 'remote "' and '"'
                remotes.append(name)
        return remotes
    
    def get_fetch_refspecs(self, name: str) -> List[str]:
        """Get fetch refspecs for named remote."""
        section_name = f"remote \"{name}\""
        if section_name in self.config:
            fetch_spec = self.config[section_name].get('fetch', '')
            return [fetch_spec] if fetch_spec else []
        return []
    
    def _save_config(self):
        """Save configuration to .git/config file."""
        with open(self.config_path, 'w') as f:
            self.config.write(f)

class RefSpec:
    """Parses and manages Git refspecs for push/fetch operations."""
    
    def __init__(self, refspec: str):
        self.refspec = refspec
        self.force = refspec.startswith('+')
        spec = refspec[1:] if self.force else refspec
        
        if ':' in spec:
            self.source, self.destination = spec.split(':', 1)
        else:
            self.source = spec
            self.destination = spec
    
    def matches_ref(self, ref: str) -> bool:
        """Check if this refspec matches the given reference."""
        if '*' in self.source:
            # Handle wildcard matching
            prefix = self.source[:self.source.index('*')]
            return ref.startswith(prefix)
        else:
            return ref == self.source
    
    def transform_ref(self, ref: str) -> str:
        """Transform source ref to destination using this refspec."""
        if '*' in self.source and '*' in self.destination:
            # Handle wildcard transformation
            prefix = self.source[:self.source.index('*')]
            suffix = ref[len(prefix):]
            dest_prefix = self.destination[:self.destination.index('*')]
            return dest_prefix + suffix
        else:
            return self.destination

# Network transport helper
class HTTPTransport:
    """Simple HTTP transport for Git operations."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        
    def discover_refs(self, service: str) -> List[Tuple[str, str]]:
        """Discover available references on remote repository."""
        import urllib.request
        
        url = f"{self.base_url}/info/refs?service={service}"
        try:
            with urllib.request.urlopen(url) as response:
                # Parse Git's advertisement format
                refs = []
                for line in response:
                    line = line.decode('utf-8').strip()
                    if line and not line.startswith('#'):
                        parts = line.split('\t')
                        if len(parts) == 2:
                            hash_val, ref_name = parts
                            refs.append((ref_name, hash_val))
                return refs
        except Exception as e:
            print(f"Failed to discover refs: {e}")
            return []
```

#### Core Logic Skeleton Code

Here are the skeleton implementations for the main remote operations:

```python
#!/usr/bin/env python3
"""
Core remote operations - fetch, push, pull.
Skeletons for learner implementation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

class FetchOperation:
    """Implements git fetch operation."""
    
    def __init__(self, repository, object_store, reference_manager, remote_config):
        self.repository = repository
        self.object_store = object_store
        self.reference_manager = reference_manager
        self.remote_config = remote_config
    
    def fetch(self, remote_name: str, refspecs: List[str] = None) -> Dict[str, str]:
        """
        Fetch objects and references from remote repository.
        Returns mapping of updated references to their new commit hashes.
        """
        # TODO 1: Get remote URL from configuration
        # TODO 2: Discover available references on remote repository  
        # TODO 3: Determine which references need updating based on refspecs
        # TODO 4: Calculate missing objects using want/have negotiation
        # TODO 5: Request pack file containing missing objects from remote
        # TODO 6: Receive and validate pack file from remote
        # TODO 7: Extract objects from pack file and store in object store
        # TODO 8: Update remote-tracking references in .git/refs/remotes/
        # TODO 9: Return mapping of updated references
        # Hint: Use _discover_remote_refs(), _negotiate_objects(), _receive_pack()
        pass
    
    def _discover_remote_refs(self, remote_url: str) -> List[Tuple[str, str]]:
        """Discover what references exist on the remote repository."""
        # TODO 1: Connect to remote repository using appropriate transport
        # TODO 2: Send upload-pack request to get reference advertisement
        # TODO 3: Parse response to extract (ref_name, commit_hash) pairs
        # TODO 4: Return list of available references
        # Hint: Different transports (HTTP, SSH, local) have different protocols
        pass
    
    def _negotiate_objects(self, remote_refs: List[Tuple[str, str]]) -> Set[str]:
        """Determine which objects need to be fetched from remote."""
        # TODO 1: Identify commit hashes we want (from remote refs we're fetching)
        # TODO 2: Walk local object store to identify commit hashes we have
        # TODO 3: Send want/have lists to remote repository
        # TODO 4: Receive response indicating which objects will be sent
        # TODO 5: Return set of object hashes that will be transferred
        # Hint: This minimizes network transfer by avoiding duplicate objects
        pass
    
    def _receive_pack(self, expected_objects: Set[str]) -> Path:
        """Receive pack file containing requested objects."""
        # TODO 1: Receive pack file header with object count
        # TODO 2: Stream pack file data to temporary file
        # TODO 3: Validate pack file checksum
        # TODO 4: Verify pack contains expected objects
        # TODO 5: Return path to temporary pack file
        # Hint: Pack files can be large - stream to disk, don't buffer in memory
        pass

class PushOperation:
    """Implements git push operation."""
    
    def __init__(self, repository, object_store, reference_manager, remote_config):
        self.repository = repository
        self.object_store = object_store
        self.reference_manager = reference_manager
        self.remote_config = remote_config
    
    def push(self, remote_name: str, refspecs: List[str], force: bool = False) -> Dict[str, str]:
        """
        Push local references and objects to remote repository.
        Returns mapping of pushed references to their commit hashes.
        """
        # TODO 1: Get remote URL and discover current remote references
        # TODO 2: Validate push refspecs and check for conflicts
        # TODO 3: Verify fast-forward requirement unless force=True
        # TODO 4: Calculate objects that need to be sent to remote
        # TODO 5: Generate pack file containing required objects
        # TODO 6: Send pack file and reference updates atomically
        # TODO 7: Verify remote accepted all updates successfully
        # TODO 8: Update local remote-tracking references
        # TODO 9: Return mapping of successfully pushed references
        # Hint: Push must be atomic - either all refs update or none do
        pass
    
    def _verify_fast_forward(self, local_ref: str, remote_ref: str, force: bool) -> bool:
        """Verify that push represents a fast-forward update."""
        # TODO 1: Get commit hash for local reference
        # TODO 2: Get commit hash for remote reference  
        # TODO 3: If force=True, allow any update
        # TODO 4: Check if remote commit is ancestor of local commit
        # TODO 5: Return True only if update is fast-forward or forced
        # Hint: Use commit graph traversal to check ancestry
        pass
    
    def _generate_pack_file(self, required_objects: Set[str]) -> Path:
        """Generate pack file containing objects to send to remote."""
        # TODO 1: Create temporary pack file
        # TODO 2: Write pack header with object count
        # TODO 3: Write each object in dependency order (commits, trees, blobs)
        # TODO 4: Apply delta compression where beneficial
        # TODO 5: Write pack file checksum
        # TODO 6: Return path to completed pack file
        # Hint: Objects must be ordered so dependencies come before referents
        pass

# Pack file skeleton implementation
class PackFile:
    """Reads and writes Git pack files."""
    
    def __init__(self, pack_path: Path):
        self.pack_path = pack_path
        self.index_path = pack_path.with_suffix('.idx')
    
    def read_object(self, object_hash: str) -> Optional[Tuple[str, bytes]]:
        """Read an object from this pack file."""
        # TODO 1: Look up object hash in pack index to get offset
        # TODO 2: Seek to offset in pack file
        # TODO 3: Read object header (type and size)
        # TODO 4: If object is delta, resolve delta chain
        # TODO 5: Decompress object content
        # TODO 6: Return (object_type, content) tuple
        # Hint: Delta objects require recursive resolution
        pass
    
    def _resolve_delta_chain(self, offset: int) -> bytes:
        """Resolve delta chain to get final object content."""
        # TODO 1: Read delta object at given offset
        # TODO 2: Identify base object (by offset or hash)
        # TODO 3: Recursively resolve base object content
        # TODO 4: Apply delta instructions to base content
        # TODO 5: Return final reconstructed content
        # Hint: Delta chains can be deep - avoid stack overflow
        pass
```

#### Milestone Checkpoints

After implementing remote repository support:

**Checkpoint 1: Basic Remote Configuration**
```bash
# Test remote configuration
python -c "
from remote.config.remote_config import RemoteConfig
from pathlib import Path

config = RemoteConfig(Path('.git'))
config.add_remote('origin', 'https://github.com/user/repo.git')
print('Remotes:', config.list_remotes())
print('Origin URL:', config.get_remote_url('origin'))
"
# Expected: Shows 'origin' in remote list with correct URL
```

**Checkpoint 2: Reference Discovery**
```bash
# Test remote reference discovery
python -c "
from remote.operations.fetch import FetchOperation
# Should connect to remote and list available branches/tags
# Expected: List of (ref_name, commit_hash) pairs
"
```

After implementing pack file support:

**Checkpoint 3: Pack File Reading**
```bash
# Create a pack file and verify reading
python -c "
from pack.pack_file import PackFile
# Should be able to read existing pack files
# Expected: Successfully retrieve objects from pack
"
```

**Checkpoint 4: Repository Size Comparison**
```bash
# Compare repository size before/after packing
du -sh .git/objects  # Before packing
# Run garbage collection with pack generation
du -sh .git/objects  # After packing - should be much smaller
```

#### Debugging Tips for Remote Operations

| Symptom | Likely Cause | How to Diagnose | Fix |
|---------|--------------|----------------|-----|
| Fetch hangs forever | Network timeout or incorrect URL | Check network connectivity, verify URL format | Add timeout handling, validate URLs |
| Objects missing after fetch | Pack file corruption or incomplete transfer | Verify pack file checksum, check object count | Re-fetch with integrity verification |
| Push rejected | Non-fast-forward update or permissions | Check if remote branch has new commits | Fetch first, merge conflicts, then push |
| Pack files not found | Index corruption or missing pack files | List `.git/objects/pack/` directory contents | Regenerate pack indexes or re-clone |

#### Performance Monitoring

Track key metrics when implementing these extensions:

```python
class RemoteOperationMetrics:
    """Track performance metrics for remote operations."""
    
    def __init__(self):
        self.objects_transferred = 0
        self.bytes_transferred = 0
        self.operation_time = 0.0
        self.network_time = 0.0
        self.compression_ratio = 0.0
    
    def log_fetch_completed(self, objects: int, bytes_total: int, duration: float):
        """Log successful fetch operation metrics."""
        print(f"Fetch completed: {objects} objects, {bytes_total} bytes, {duration:.2f}s")
        print(f"Transfer rate: {bytes_total / duration / 1024:.1f} KB/s")
```

These extensions transform the basic Git implementation into a system capable of handling real-world distributed development workflows and large-scale repositories efficiently.

### Implementation Guidance

#### Technology Recommendations

| Component | Simple Option | Advanced Option |
|-----------|---------------|-----------------|
| Remote Transport | `urllib3` with basic HTTP | `requests` with session pooling and retry logic |
| Pack File I/O | `struct` module for binary parsing | `mmap` for efficient large file access |
| Delta Compression | Basic diff implementation | Optimized Myers algorithm with binary search |
| Network Protocol | Simple request/response | Full Git smart protocol with capability negotiation |
| Concurrency | Sequential operations | `asyncio` for concurrent fetch/push operations |

#### Recommended File Structure

```python
# Extensions to existing structure
extensions/
    remote/
        __init__.py
        transport/
            __init__.py
            http_transport.py      # HTTP/HTTPS protocol implementation
            ssh_transport.py       # SSH protocol implementation
            protocol.py            # Git wire protocol parsing
        operations/
            __init__.py
            fetch.py              # Fetch operation with negotiation
            push.py               # Push with conflict detection
            pull.py               # Pull as fetch + merge
        config/
            __init__.py
            remote_manager.py     # Remote repository management
            refspec_parser.py     # Reference specification handling
    
    pack/
        __init__.py
        pack_reader.py           # Read existing pack files
        pack_writer.py           # Create new pack files
        pack_index.py            # Pack index management
        delta_compression.py     # Delta encoding/decoding
        garbage_collector.py     # Pack generation and cleanup
```

#### Infrastructure Starter Code

Complete network transport implementation:

```python
#!/usr/bin/env python3
"""
Git HTTP transport implementation.
Production-ready code for network operations.
"""

import urllib.request
import urllib.parse
from typing import Dict, List, Optional, Tuple
from pathlib import Path

class GitHTTPTransport:
    """Handles Git smart HTTP protocol operations."""
    
    def __init__(self, base_url: str, username: str = None, password: str = None):
        self.base_url = base_url.rstrip('/')
        self.username = username
        self.password = password
        self._setup_auth()
    
    def _setup_auth(self):
        """Configure HTTP authentication if credentials provided."""
        if self.username and self.password:
            password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
            password_mgr.add_password(None, self.base_url, self.username, self.password)
            auth_handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
            opener = urllib.request.build_opener(auth_handler)
            urllib.request.install_opener(opener)
    
    def discover_refs(self, service: str) -> List[Tuple[str, str]]:
        """
        Discover available references on remote repository.
        Service should be 'git-upload-pack' for fetch or 'git-receive-pack' for push.
        """
        url = f"{self.base_url}/info/refs?service={service}"
        
        try:
            request = urllib.request.Request(url)
            request.add_header('User-Agent', 'git/2.0 (custom-implementation)')
            
            with urllib.request.urlopen(request, timeout=30) as response:
                # Parse Git smart HTTP protocol response
                refs = []
                lines = response.read().decode('utf-8').splitlines()
                
                # Skip protocol header lines
                for line in lines:
                    if line.startswith('#'):
                        continue
                    if '\t' in line:
                        hash_and_caps, ref_name = line.split('\t', 1)
                        # Remove capability advertisements
                        object_hash = hash_and_caps.split(' ')[0]
                        if len(object_hash) == 40:  # Valid SHA-1
                            refs.append((ref_name, object_hash))
                
                return refs
                
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to {url}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error discovering refs: {e}")
    
    def send_pack_request(self, service: str, request_data: bytes) -> bytes:
        """Send pack request and receive response."""
        url = f"{self.base_url}/{service}"
        
        request = urllib.request.Request(url, data=request_data)
        request.add_header('Content-Type', f'application/x-{service}-request')
        request.add_header('Accept', f'application/x-{service}-result')
        request.add_header('User-Agent', 'git/2.0 (custom-implementation)')
        
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            error_msg = e.read().decode('utf-8') if e.fp else str(e)
            raise RuntimeError(f"Server error {e.code}: {error_msg}")
        except Exception as e:
            raise RuntimeError(f"Network error: {e}")

class PackProtocolHandler:
    """Handles Git pack protocol for efficient object transfer."""
    
    @staticmethod
    def create_want_have_request(wants: List[str], haves: List[str]) -> bytes:
        """Create want/have negotiation request."""
        lines = []
        
        # Add want lines
        for i, want in enumerate(wants):
            if i == 0:
                # First want line includes capabilities
                line = f"want {want} multi_ack_detailed side-band-64k ofs-delta\n"
            else:
                line = f"want {want}\n"
            lines.append(PackProtocolHandler._pkt_line(line))
        
        # Separator
        lines.append(b"0000")
        
        # Add have lines
        for have in haves:
            line = f"have {have}\n"
            lines.append(PackProtocolHandler._pkt_line(line))
        
        # End negotiation
        lines.append(PackProtocolHandler._pkt_line("done\n"))
        
        return b''.join(lines)
    
    @staticmethod
    def _pkt_line(data: str) -> bytes:
        """Format data as Git packet line."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        length = len(data) + 4
        return f"{length:04x}".encode('ascii') + data
    
    @staticmethod
    def parse_pack_response(response: bytes) -> Tuple[bytes, List[str]]:
        """Parse pack response, extracting pack data and progress messages."""
        pack_data = bytearray()
        messages = []
        offset = 0
        
        while offset < len(response):
            # Read packet length
            if offset + 4 > len(response):
                break
                
            length_str = response[offset:offset+4].decode('ascii')
            if length_str == "0000":
                offset += 4
                continue
                
            try:
                packet_length = int(length_str, 16)
            except ValueError:
                break
                
            if packet_length < 4:
                break
                
            packet_data = response[offset+4:offset+packet_length]
            offset += packet_length
            
            if packet_data.startswith(b'\x01'):  # Pack data
                pack_data.extend(packet_data[1:])
            elif packet_data.startswith(b'\x02'):  # Progress message
                messages.append(packet_data[1:].decode('utf-8', errors='ignore'))
        
        return bytes(pack_data), messages
```

#### Core Logic Skeleton Code

```python
#!/usr/bin/env python3
"""
Core pack file operations.
Skeleton implementations for learner completion.
"""

import struct
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class PackFileReader:
    """Reads objects from Git pack files."""
    
    def __init__(self, pack_path: Path):
        self.pack_path = pack_path
        self.index_path = pack_path.with_suffix('.idx')
        self._load_index()
    
    def _load_index(self):
        """Load pack index for efficient object lookup."""
        # TODO 1: Open pack index file (.idx)
        # TODO 2: Read index header and verify format version
        # TODO 3: Read fanout table (256 4-byte entries)
        # TODO 4: Read sorted object hash list
        # TODO 5: Read CRC checksums for each object
        # TODO 6: Read pack file offsets for each object
        # TODO 7: Store index data in memory for fast lookup
        # Hint: Index format is big-endian, use struct.unpack('>I', data)
        pass
    
    def has_object(self, object_hash: str) -> bool:
        """Check if object exists in this pack file."""
        # TODO 1: Use fanout table for fast first-level lookup
        # TODO 2: Binary search in appropriate hash segment
        # TODO 3: Return True if hash found in index
        # Hint: Fanout table maps first hash byte to index range
        pass
    
    def read_object(self, object_hash: str) -> Optional[Tuple[str, bytes]]:
        """Read object from pack file, resolving deltas if necessary."""
        # TODO 1: Look up object offset in pack index
        # TODO 2: Seek to offset in pack file
        # TODO 3: Read object header (type and size)
        # TODO 4: Handle different object types (commit, tree, blob, delta)
        # TODO 5: If delta object, resolve against base object
        # TODO 6: Decompress final object content
        # TODO 7: Return (object_type, content) tuple
        # Hint: Delta objects require recursive resolution
        pass
    
    def _read_object_at_offset(self, offset: int) -> Tuple[int, int, bytes]:
        """Read raw object data at given pack file offset."""
        # TODO 1: Seek to offset in pack file
        # TODO 2: Read variable-length object header
        # TODO 3: Determine object type and uncompressed size
        # TODO 4: Read compressed object data
        # TODO 5: Return (object_type, size, compressed_data)
        # Hint: Object header uses variable-length encoding
        pass
    
    def _resolve_delta_object(self, delta_data: bytes, base_content: bytes) -> bytes:
        """Apply delta instructions to base object content."""
        # TODO 1: Parse delta header (base size, result size)
        # TODO 2: Read delta instructions (copy or insert)
        # TODO 3: For copy instructions, copy bytes from base content
        # TODO 4: For insert instructions, copy literal bytes from delta
        # TODO 5: Verify result size matches delta header
        # TODO 6: Return reconstructed object content
        # Hint: Delta format: base_size, result_size, then copy/insert ops
        pass

class PackFileWriter:
    """Creates Git pack files with delta compression."""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.objects = []
        self.written_objects = {}
    
    def add_object(self, object_hash: str, object_type: str, content: bytes):
        """Add object to pack file."""
        # TODO 1: Store object information for later processing
        # TODO 2: Consider delta compression against similar objects
        # TODO 3: Add to objects list with metadata
        # Hint: Don't write immediately - collect all objects first
        pass
    
    def write_pack(self) -> Tuple[Path, Path]:
        """Write pack file and index to disk."""
        # TODO 1: Sort objects for optimal delta compression
        # TODO 2: Write pack file header (signature, version, object count)
        # TODO 3: Write each object with delta compression where beneficial
        # TODO 4: Write pack file checksum
        # TODO 5: Generate pack index file
        # TODO 6: Return (pack_file_path, index_file_path)
        # Hint: Process objects in dependency order (commits, trees, blobs)
        pass
    
    def _find_delta_base(self, target_content: bytes, candidates: List[bytes]) -> Optional[bytes]:
        """Find best base object for delta compression."""
        # TODO 1: Compare target content against each candidate
        # TODO 2: Calculate potential compression ratio
        # TODO 3: Select candidate with best compression ratio
        # TODO 4: Return best base content or None if no good match
        # Hint: Simple heuristic - choose candidate with most common subsequences
        pass
    
    def _create_delta(self, base_content: bytes, target_content: bytes) -> bytes:
        """Create delta instructions to transform base into target."""
        # TODO 1: Write delta header (base size, target size)
        # TODO 2: Find common subsequences between base and target
        # TODO 3: Generate copy instructions for common parts
        # TODO 4: Generate insert instructions for new parts
        # TODO 5: Return complete delta data
        # Hint: Use simple longest common subsequence algorithm
        pass

class GarbageCollector:
    """Performs repository garbage collection and pack generation."""
    
    def __init__(self, repository):
        self.repository = repository
        self.reachable_objects = set()
    
    def collect_garbage(self, aggressive: bool = False) -> Dict[str, int]:
        """Run complete garbage collection cycle."""
        # TODO 1: Mark all reachable objects starting from references
        # TODO 2: Identify unreachable objects in object store
        # TODO 3: Generate pack files for reachable objects
        # TODO 4: Remove loose objects that are now packed
        # TODO 5: Remove unreachable objects
        # TODO 6: Update repository statistics
        # TODO 7: Return statistics (objects packed, bytes saved, etc.)
        # Hint: This is a complex multi-phase operation
        pass
    
    def _mark_reachable_objects(self):
        """Mark all objects reachable from references."""
        # TODO 1: Get all references (HEAD, branches, tags)
        # TODO 2: For each reference, traverse object graph
        # TODO 3: Mark commits, then trees, then blobs as reachable
        # TODO 4: Handle circular references properly
        # TODO 5: Store reachable object hashes in self.reachable_objects
        # Hint: Use breadth-first search to avoid stack overflow
        pass
    
    def _generate_packs(self, objects_to_pack: List[str]):
        """Generate pack files for specified objects."""
        # TODO 1: Group objects into logical packs (by age, size, etc.)
        # TODO 2: For each pack, collect object content
        # TODO 3: Create PackFileWriter and add all objects
        # TODO 4: Write pack and index files
        # TODO 5: Verify pack integrity
        # Hint: Recent objects should be packed together for access efficiency
        pass
```

#### Milestone Checkpoints

**Remote Operations Checkpoint:**
```bash
# Test basic remote configuration
python3 -c "
from extensions.remote.config.remote_manager import RemoteConfig
from pathlib import Path

# Configure remote
config = RemoteConfig(Path('.git'))
config.add_remote('origin', 'https://github.com/git/git.git')
print('Configured remotes:', config.list_remotes())

# Test reference discovery
from extensions.remote.transport.http_transport import GitHTTPTransport
transport = GitHTTPTransport('https://github.com/git/git.git')
refs = transport.discover_refs('git-upload-pack')
print(f'Discovered {len(refs)} references')
print('Sample refs:', refs[:5])
"
# Expected: Shows configured remote and lists remote references
```

**Pack File Checkpoint:**
```bash
# Create pack file and verify reading
python3 -c "
from extensions.pack.pack_reader import PackFileReader
from extensions.pack.pack_writer import PackFileWriter
from pathlib import Path

# Find existing pack file or create one
pack_dir = Path('.git/objects/pack')
if pack_dir.exists():
    pack_files = list(pack_dir.glob('*.pack'))
    if pack_files:
        reader = PackFileReader(pack_files[0])
        print(f'Pack file has objects: {reader.has_object}')
"
# Expected: Successfully reads pack file metadata

```


## Glossary

> **Milestone(s):** This comprehensive glossary applies to all eight milestones, providing essential definitions for understanding Git's architecture and implementation. It serves as a reference throughout the entire project journey.

This glossary serves as the definitive reference for all technical terms, Git-specific vocabulary, architectural concepts, and domain-specific language used throughout the Build Your Own Git project. The definitions are organized to build understanding progressively, starting with foundational concepts and moving to advanced implementation details.

### Mental Model: The Technical Dictionary

Think of this glossary as a specialized technical dictionary for Git internals, similar to how a medical dictionary defines terms like "myocardial infarction" not just as "heart attack" but with the precise technical meaning that medical professionals need. Each definition here goes beyond surface-level explanations to provide the exact technical understanding needed to implement Git correctly. Just as a surgeon needs to understand the precise anatomical meaning of each term, a Git implementer needs to understand the exact algorithmic and data structure implications of each concept.

### Core Git Concepts

**Add Operation**: The process of staging files for commit by computing their blob hash, creating blob objects in the object store, and recording index entries. This operation transforms working directory files into immutable blob objects and updates the staging area to prepare for commit creation.

**Atomic Write**: A file system operation that ensures either complete success or complete failure with no partial states. Implemented by writing to a temporary file, then using rename system call to atomically replace the target file. Critical for maintaining repository consistency during concurrent access.

**Binary File Detection**: Heuristic process for identifying non-text files by analyzing a sample of bytes for non-printable characters. Uses a threshold ratio (typically 75% printable characters) to distinguish text files suitable for line-based diff operations from binary files requiring different handling.

**Binary Format**: Compact binary representation of data structures optimized for storage efficiency and parsing speed. The Git index uses binary format to store file metadata and object hashes, requiring specialized serialization and deserialization logic.

**Blob Object**: Immutable storage container for file content in Git's object model. Contains only raw file bytes with no metadata like filename or permissions. Identified by SHA-1 hash of formatted content with "blob {size}\0" header.

**Branch Namespace**: Hierarchical organization of branch names using path separators, enabling logical grouping like "feature/user-auth" or "bugfix/memory-leak". Stored as nested directories under `.git/refs/heads/` to match the namespace structure.

**Breadth-First Search**: Graph traversal algorithm used for finding merge base by exploring all commits at distance N before moving to distance N+1. Ensures discovery of the lowest common ancestor when multiple common ancestors exist.

**Commit Graph**: Directed acyclic graph structure representing project history where commits reference parent commits. Forms the backbone of Git's version control model, enabling branch and merge operations while preventing circular history.

**Commit Object**: Immutable container storing project state snapshot with metadata. Contains tree hash, parent commit hashes, author/committer information with timestamps, and commit message. Forms nodes in the commit graph representing project evolution.

**Compression Problems**: Issues with zlib compression/decompression during object storage operations. Common manifestations include corrupted objects that cannot be decompressed, hash mismatches after decompression, and performance issues with large objects.

**Conflict Detection**: Algorithm for identifying overlapping modifications during three-way merge operations. Compares changes from both branches against the common base to detect lines modified in both branches, requiring manual resolution.

**Conflict Markers**: Special text sequences delimiting merge conflicts for manual resolution. Standard format uses "<<<<<<< ours", "=======" separator, and ">>>>>>> theirs" markers to clearly delineate conflicting content from different branches.

**Content-Addressable Storage**: Storage system where objects are identified and retrieved using cryptographic hashes of their content. Provides automatic deduplication, integrity verification, and immutable object properties fundamental to Git's architecture.

**Context Lines**: Unchanged lines displayed around modifications in diff output to provide context for understanding changes. Typically 3 lines before and after each change, configurable based on user preferences or diff complexity.

**Detached HEAD**: Repository state where HEAD points directly to a commit hash instead of a branch reference. Occurs when checking out specific commits, creating a temporary state where new commits don't advance any branch pointer.

**Diagonal Move**: Myers diff algorithm term for matching elements between sequences, representing unchanged lines. Visualized as diagonal movement in the algorithm's grid representation, indicating no insertion or deletion needed.

**Diff Algorithm**: Algorithm for computing differences between two sequences of lines. Myers algorithm finds the shortest edit script (minimum insertions and deletions) to transform one sequence into another, optimizing for minimal change representation.

**Direct Reference**: Git reference containing a raw 40-character SHA-1 commit hash. Contrasts with symbolic references that point to other references. Used for detached HEAD state and some internal Git operations.

**Directed Acyclic Graph**: Mathematical structure where commits reference parents with no circular dependencies. Ensures Git history has clear temporal ordering while supporting multiple branches and merge operations without paradoxes.

**Edit Distance**: Minimum number of insert/delete operations needed to transform one sequence into another. Central concept in diff algorithms, with Myers algorithm optimizing to find the shortest possible edit script between file versions.

**Edit Script**: Sequence of insertion and deletion operations that transforms one file into another. Generated by diff algorithms and used to reconstruct changes, apply patches, and display modifications in human-readable format.

### Storage and Object Model

**Hash Mismatches**: Errors where computed SHA-1 hash doesn't match expected value, indicating data corruption or implementation bugs. Can occur during object storage, retrieval, or transmission, requiring integrity verification and error recovery mechanisms.

**Hunk**: Contiguous block of changes in diff output, consisting of nearby modifications grouped together with context lines. Unified diff format organizes changes into hunks with headers showing affected line ranges in both file versions.

**Immutable History**: Property that historical commits cannot be modified once created, ensuring repository integrity and enabling reliable collaboration. Achieved through cryptographic hashing where any content change produces a different hash.

**Index Binary Format**: Compact binary representation of staging area contents optimized for performance. Contains header with version and entry count, sorted file entries with metadata and hashes, and SHA-1 checksum for corruption detection.

**Lowest Common Ancestor**: Most recent commit reachable from both branches in merge operations. Critical for three-way merge algorithm as the base version for comparing changes from both branches, found using breadth-first search.

**Merge Base**: The common ancestor commit used as the base version in three-way merge operations. Found by traversing commit history from both branch tips until discovering the most recent shared commit.

**Metadata Caching**: Storing file system metadata (timestamps, sizes, inodes) in the index to quickly detect modifications without re-reading file content. Enables efficient status calculations by comparing cached metadata with current file system state.

**Myers Algorithm**: Optimal diff algorithm that finds the shortest edit script between two sequences using dynamic programming. Visualizes the problem as finding the shortest path through a grid, with optimizations for practical performance.

**Object Store**: Git's content-addressable storage backend for immutable objects (blobs, trees, commits). Uses SHA-1 hashes as keys, stores compressed objects in `.git/objects/` directory structure, provides retrieval and verification capabilities.

### References and Branching

**Reference Resolution**: Process of following symbolic references to find the target commit hash. Handles chains of symbolic references and validates reference integrity, essential for operations requiring actual commit hashes.

**Repository Consistency**: Validation that repository satisfies all Git invariants including object reachability, reference validity, and structural integrity. Ensures reliable operation and prevents corruption from accumulating over time.

**SHA-1 Hash**: Cryptographic hash function producing 160-bit (40 hex character) digests used as object identifiers. While deprecated for cryptographic security, still used in Git for content addressing and object identification.

**Staging Area**: Intermediate layer between working directory and repository history where changes are prepared for commit. Implemented as the index file, allows incremental commit preparation and fine-grained control over version history.

**Symbolic Reference**: Reference pointing to another reference rather than directly to a commit hash. HEAD is typically a symbolic reference pointing to the current branch, enabling automatic branch advancement during commits.

**Three-Way Comparison**: Status calculation method comparing working directory, index (staging area), and HEAD to determine file states. Identifies modified, staged, untracked, and deleted files by analyzing differences between these three sources.

**Three-Way Merge**: Merge algorithm using base version (common ancestor) and two branch versions to automatically combine non-conflicting changes. More sophisticated than two-way merge, reduces false conflicts by understanding change attribution.

**Tree Object**: Immutable container representing directory structure in Git's object model. Contains sorted list of entries with file modes, names, and SHA-1 hashes pointing to blobs or other trees, forming hierarchical project snapshots.

**Unified Diff Format**: Industry standard for presenting file differences with context lines, hunk headers, and change markers. Provides human-readable representation of modifications suitable for code review and patch application.

### Advanced Concepts and Extensions

**Change Attribution**: Process of determining which branch made each modification during merge operations. Essential for three-way merge algorithm to distinguish between conflicting and complementary changes from different development lines.

**Delta Compression**: Storage optimization technique using differences between similar objects rather than storing complete content. Reduces repository size by expressing objects as modifications to base objects, used in Git's pack file format.

**Fanout Table**: Index structure in pack files providing efficient object lookup by organizing objects by hash prefix. Contains 256 entries corresponding to first byte values, enabling binary search optimization for object retrieval.

**Fast-Forward Update**: Reference update where new commit is descendant of old commit, requiring only pointer movement without merge. Common in linear development workflows and during fetch operations with no local changes.

**Fetch Operation**: Retrieval of objects and references from remote repository without merging into local branches. Updates remote-tracking branches to reflect remote state while leaving local development branches unchanged.

**Garbage Collection**: Process of removing unreachable objects and optimizing repository storage. Identifies objects not reachable from any reference, removes them to reclaim disk space, and reorganizes remaining objects for efficiency.

**Greedy Extension**: Myers algorithm optimization that follows matching elements as far as possible before considering insertions or deletions. Improves algorithm performance by reducing the search space for optimal solutions.

**Object Relationships**: References between commits, trees, and blobs forming Git's directed acyclic graph structure. Enables repository validation, garbage collection, and history traversal by following hash-based connections.

**Pack File**: Compressed bundle of Git objects optimized for storage efficiency and network transfer. Uses delta compression and zlib to minimize space, essential for large repositories and efficient remote operations.

**Pack Index**: Lookup table enabling efficient random access to objects within pack files. Contains sorted object hashes with corresponding pack file offsets, supporting binary search for fast object retrieval.

**Push Operation**: Sending local objects and reference updates to remote repository. Includes object transfer optimization and reference update validation to maintain consistency across distributed repositories.

**Remote-Tracking Branch**: Local reference showing last known state of remote branch without affecting local development branches. Updated during fetch operations to track remote repository evolution.

**Want/Have Negotiation**: Network protocol optimization to minimize object transfer by identifying objects already present in destination repository. Reduces bandwidth usage during fetch and push operations.

### Implementation Details

**Working Directory**: File system directory containing checked-out project files that users can modify. Represents current project state and serves as source for staging operations, distinct from immutable repository history.

**Git Directory**: The `.git` folder containing all repository metadata, object storage, references, and configuration. Hidden from normal file operations while containing all information needed to reconstruct project history.

**Diagnostic Tools**: Utilities for inspecting and validating Git repository internals during development and debugging. Include object inspection, reference validation, and repository consistency checking capabilities.

**Repository Corruption**: Data integrity failures in Git repositories including corrupted objects, missing references, and invalid repository states. Requires detection mechanisms and recovery strategies to maintain reliable operation.

**Concurrent Access Patterns**: Strategies for handling multiple processes accessing Git repository simultaneously without corruption. Includes file locking, atomic operations, and coordination mechanisms for safe concurrent operation.

**File System Permissions**: Access control settings for Git repository files ensuring security while enabling necessary operations. Critical for shared repositories and preventing unauthorized modification of repository data.

**Binary Detection Size**: Number of bytes analyzed when determining if file content is binary or text, typically 1024 bytes. Optimization balancing accuracy of detection with performance of file analysis.

**Printable Ratio Threshold**: Minimum proportion of printable characters required to classify content as text rather than binary, typically 75%. Used in heuristic binary detection algorithms.

### Data Structures and Formats

| Structure | Purpose | Key Components |
|-----------|---------|----------------|
| `Repository` | Main repository interface | git_dir, work_tree paths |
| `ObjectStore` | Content-addressable storage | objects_dir, hash-to-path mapping |
| `Index` | Staging area implementation | entries list, binary serialization |
| `ReferenceManager` | Branch and reference handling | refs_dir, heads_dir, head_file |
| `TreeEntry` | Directory entry representation | mode, name, hash tuple |
| `IndexEntry` | Staged file metadata | timestamps, size, hash, path |
| `GitObject` | Base object interface | content bytes, type identification |
| `BlobObject` | File content storage | raw content bytes |
| `TreeObject` | Directory structure | sorted entries list |
| `CommitObject` | Project snapshot | tree, parents, metadata |

### Error Conditions and Recovery

| Error Type | Symptoms | Detection Method | Recovery Strategy |
|------------|----------|------------------|-------------------|
| Hash Mismatch | Object corruption | SHA-1 verification | Recompute from source |
| Missing Object | Reference errors | Object existence check | Fetch from remote |
| Index Corruption | Staging failures | Checksum validation | Rebuild from working tree |
| Reference Invalid | Branch operations fail | Path validation | Reset to known good state |
| Merge Conflicts | Overlapping changes | Three-way comparison | Manual resolution |
| Lock Contention | Concurrent access | File lock timeout | Retry with backoff |

### Algorithm Categories

| Algorithm Type | Primary Use | Key Characteristics | Implementation Notes |
|----------------|-------------|-------------------|---------------------|
| Myers Diff | File comparison | Optimal edit distance | Grid-based dynamic programming |
| Breadth-First Search | Merge base finding | Graph traversal | Queue-based exploration |
| Three-Way Merge | Branch combination | Conflict detection | Base-relative comparison |
| SHA-1 Hashing | Object identification | Cryptographic integrity | Content-addressable keys |
| Zlib Compression | Storage optimization | Size reduction | Standard deflate algorithm |

### Implementation Guidance

This glossary serves as a living reference throughout implementation, with definitions becoming more concrete as you implement each milestone. The terms progress from abstract concepts to specific data structures and algorithms as your understanding deepens.

#### Essential Reference Patterns

When implementing Git operations, several reference patterns appear repeatedly:

**Object Hash Calculation**: Always format as `{type} {size}\0{content}` before computing SHA-1 hash. The null byte separator is critical for proper hash computation and object integrity.

**Path Resolution**: Convert object hashes to file system paths by splitting into 2-character directory prefix and 38-character filename. This distribution ensures reasonable directory sizes even with millions of objects.

**Reference Chain Following**: When resolving references, follow symbolic references iteratively until reaching a direct hash reference. Implement cycle detection to prevent infinite loops from corrupted references.

**Binary vs Text Detection**: Sample initial bytes of file content, count printable characters, and apply threshold ratio. This heuristic approach balances accuracy with performance for large repositories.

#### Common Implementation Mistakes

⚠️ **Pitfall: Hash Format Confusion**
Many implementations incorrectly store object hashes as 40-character hex strings in binary formats like the index. Git stores hashes as 20-byte binary data in most internal formats, converting to hex only for display and file names.

⚠️ **Pitfall: Reference File Encoding**
Reference files must end with newlines and contain only the hash or symbolic reference. Missing newlines cause parsing errors, while extra whitespace breaks hash validation.

⚠️ **Pitfall: Tree Entry Sorting**
Tree entries must be sorted using Git's specific comparison rules, treating directories as if they end with '/' for sorting purposes. Standard string sorting produces incorrect tree hashes.

⚠️ **Pitfall: Merge Base Edge Cases**
When multiple common ancestors exist, the merge base algorithm must select the lowest common ancestor, not just any shared commit. This affects merge conflict detection and resolution.

#### Debugging Reference Guide

| Issue | Likely Cause | Investigation Steps | Resolution |
|-------|--------------|-------------------|-----------|
| Invalid object hash | Header format wrong | Check null byte placement | Reformat as `type size\0content` |
| Missing objects | Storage path incorrect | Verify `.git/objects/xx/yyyy` structure | Fix path computation logic |
| Index corruption | Binary format error | Validate header and checksum | Implement proper serialization |
| Reference resolution fails | Symbolic ref chain broken | Trace reference chain manually | Repair or reset references |
| Diff output wrong | Algorithm implementation bug | Compare with known good output | Debug Myers algorithm step-by-step |
| Merge conflicts incorrect | Base calculation wrong | Verify merge base commit | Fix breadth-first search logic |

This glossary evolves as your implementation progresses, with initially abstract concepts becoming concrete data structures and algorithms. Use it as both a learning tool and a reference during debugging sessions.

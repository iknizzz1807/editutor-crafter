# AUDIT & FIX: build-git

## CRITIQUE
- **Audit Finding 1 (Index/Staging Ordering):** This is a nuanced finding. In the real Git implementation, `write-tree` reads from the index. However, for a learning project, understanding tree objects conceptually (how they represent directory snapshots) can precede understanding the index (how changes are staged). The original ordering (M3: Trees, M6: Index) teaches the storage layer first, then the staging workflow. The finding's suggested reorder (Index before Trees) would mean students implement the index binary format before understanding what tree objects it produces. I disagree with the finding - the current ordering is pedagogically sound. However, M3's AC for `write-tree` should be adjusted: instead of 'from the current index,' it should say 'from a given directory structure' since the index doesn't exist yet at M3.
- **Audit Finding 2 (Missing Packfiles):** Valid for completeness but this is already an expert 30-50 hour project with 8 milestones. Packfiles with delta compression would add 10-15 hours minimum. This is better as an optional extension milestone. However, the project *should* acknowledge this gap and explain why loose objects are sufficient for learning.
- **M1 Estimated Hours (1-2):** Using ranges is inconsistent. Also, M1 is too trivial for a milestone - creating directories is not a meaningful learning experience. Consider merging M1 with M2.
- **M7 Myers Diff:** The diff algorithm is specified as Myers' but the AC doesn't require the *optimal* shortest edit script - just 'line-by-line differences.' Myers' algorithm is O(ND) where D is the edit distance; this should be specified.
- **M8 Merge Base Algorithm:** 'Finding the lowest common ancestor' is underspecified. In Git, this is done with `merge-base` which handles multiple common ancestors. The AC should specify which algorithm (e.g., recursive merge base finding).
- **Missing: clone/fetch/push (Remote Operations):** The project builds a local Git only. No remote operations are covered. This is acceptable for scope but should be explicitly stated.
- **Missing: .gitignore:** No milestone handles ignore patterns, which is a basic Git feature.
- **SHA-1 vs SHA-256:** The essence mentions both but all milestones use only SHA-1. SHA-256 transition is a real Git topic but shouldn't be in scope for this project.
- **Estimated Hours:** Sum of ranges: 1-2 + 2-3 + 3-4 + 2-3 + 2-3 + 4-6 + 4-6 + 6-10 = 24-37. Project says 30-50. The upper bound seems high unless packfiles are included.

## FIXED YAML
```yaml
id: build-git
name: Build Your Own Git
description: >-
  Version control system implementing content-addressable object storage,
  commit DAG, staging area, diff, and three-way merge.
difficulty: expert
estimated_hours: 45
essence: >-
  Content-addressable storage using SHA-1 hashing to create an immutable
  object database where blobs, trees, and commits form a directed acyclic
  graph representing versioned filesystem snapshots with cryptographic
  integrity guarantees and deduplication.
why_important: >-
  Building this teaches fundamental data structures used in distributed
  systems, how content-addressable storage enables efficient deduplication
  and integrity verification, and the graph algorithms underlying the
  version control tool used daily by millions of developers.
learning_outcomes:
  - Implement content-addressable object storage using SHA-1 hashing with zlib compression
  - Design blob, tree, and commit objects forming a directed acyclic graph
  - Build a binary index (staging area) for tracking changes between working tree and repository
  - Implement Myers diff algorithm for efficient line-level change detection
  - Build three-way merge with automatic conflict detection and resolution markers
  - Design a reference system for branches and HEAD pointer management
  - Understand the fundamental tradeoff of loose object storage vs packfile compression
skills:
  - Content-Addressable Storage
  - Cryptographic Hashing (SHA-1)
  - Directed Acyclic Graphs
  - Myers Diff Algorithm
  - Three-way Merge
  - Binary File Format Design
  - Object Serialization with zlib
  - Reference Management
tags:
  - build-from-scratch
  - diff
  - expert
  - merkle-tree
  - objects
  - version-control
architecture_doc: architecture-docs/build-git/index.md
languages:
  recommended:
    - Python
    - Rust
    - Go
    - C
  also_possible: []
resources:
  - name: Write yourself a Git!
    url: https://wyag.thb.lt/
    type: tutorial
  - name: CodeCrafters Git Challenge
    url: https://app.codecrafters.io/courses/git/overview
    type: tool
  - name: Git Internals - Git Objects
    url: https://git-scm.com/book/en/v2/Git-Internals-Git-Objects
    type: documentation
  - name: Git Index Format
    url: https://git-scm.com/docs/index-format
    type: documentation
  - name: Myers Diff Algorithm Paper
    url: http://www.xmailserver.org/diff2.pdf
    type: paper
prerequisites:
  - type: skill
    name: File I/O and cryptographic hashing (SHA-1)
  - type: skill
    name: Tree data structures and graph traversal
  - type: skill
    name: Zlib compression
  - type: skill
    name: Binary file format parsing
note: >-
  This project builds a local-only Git implementation. Remote operations
  (clone, fetch, push) and packfile compression are out of scope. All objects
  are stored as loose compressed files. Packfiles would be a natural extension
  for scaling to large repositories.
milestones:
  - id: build-git-m1
    name: Repository Init & Blob Storage
    description: >-
      Initialize a .git repository structure and implement content-addressable
      blob storage with hash-object and cat-file commands.
    acceptance_criteria:
      - init command creates .git directory with objects/, refs/heads/, and HEAD file containing 'ref: refs/heads/master\n'
      - .git/objects/ directory is created with info/ and pack/ subdirectories
      - hash-object computes SHA-1 of 'blob {size}\0{content}' and stores the zlib-compressed object at .git/objects/{xx}/{38-char-remainder}
      - hash-object output matches real git's hash-object output for identical input content
      - cat-file retrieves and decompresses a stored object by its SHA-1 hash, outputting the raw content (without the header)
      - cat-file -t prints the object type (blob), cat-file -s prints the content size, cat-file -p prints the content
      - Binary content (non-text files) is handled correctly - blob storage is content-agnostic
      - Attempting to init inside an existing repo prints a warning and does not overwrite
    pitfalls:
      - Forgetting the null byte (\0) between header and content produces a different hash than real git - binary exactness matters
      - Content must be zlib-compressed before writing to disk; storing uncompressed objects wastes space and is incompatible with real git
      - Binary vs text content distinction - blobs store raw bytes, not text; newline conversion must not be applied during storage
      - The .git directory permissions should be restrictive (0755 for dirs, 0644 for files) to prevent accidental modification
      - HEAD file format is exactly 'ref: refs/heads/master\n' with a trailing newline - missing the newline breaks ref parsing
    concepts:
      - Content-addressable storage - identical content always produces identical hash
      - Git object format - type tag, space, decimal size, null byte, raw content
      - SHA-1 hash as a unique object identifier
      - Zlib compression for storage efficiency
      - Repository directory structure conventions
    skills:
      - SHA-1 hashing
      - Zlib compression
      - Binary file I/O
      - Directory structure creation
    deliverables:
      - init command creating complete .git directory structure
      - hash-object computing SHA-1 and storing compressed blob objects
      - cat-file retrieving and decompressing objects by hash
      - Object path derivation splitting hash into 2-char directory and 38-char filename
    estimated_hours: 4

  - id: build-git-m2
    name: Tree Objects
    description: >-
      Implement tree objects representing directory snapshots with nested
      subdirectory support.
    acceptance_criteria:
      - Tree object stores a sorted list of entries, each containing mode (as ASCII bytes), name, and SHA-1 hash (as raw 20 bytes)
      - Tree entry format is 'mode name\0{20-byte-binary-hash}' with entries concatenated, then wrapped in 'tree {size}\0{entries}'
      - ls-tree command displays tree contents showing mode, object type (blob or tree), hex hash, and filename for each entry
      - write-tree command recursively builds tree objects from a given directory structure (simplified: from working directory, not index yet)
      - Nested subdirectories produce nested tree objects - parent tree references child tree by hash
      - Entries are sorted by name with the Git sorting rule: directories are sorted as if their name has a trailing '/'
      - Tree hash matches real git's tree hash for identical directory contents
    pitfalls:
      - SHA-1 hash in tree entries is stored as raw binary (20 bytes), NOT as hex string (40 chars) - this is the most common mistake
      - Git's sorting rule for tree entries is not simple lexicographic - directories sort as 'dirname/' while files sort as 'filename'
      - Mode is stored as ASCII digits without leading zeros for files (100644) but with specific values for trees (40000), symlinks (120000), etc.
      - Forgetting to recursively build tree objects for subdirectories produces a flat tree that misrepresents the directory structure
      - Empty directories are not stored in Git - only directories containing at least one tracked file get a tree object
    concepts:
      - Tree data structure representing a directory snapshot
      - Binary tree entry format with raw hash bytes
      - Recursive tree construction for nested directories
      - Content-addressable tree objects for deduplication of identical directories
    skills:
      - Binary format implementation
      - Tree traversal and recursion
      - Sorting with custom comparators
      - Packed binary format construction
    deliverables:
      - Tree object format implementation with binary hash storage
      - write-tree building tree objects recursively from directory structure
      - ls-tree displaying tree contents in human-readable format
      - Nested tree object creation for subdirectories
    estimated_hours: 4

  - id: build-git-m3
    name: Commit Objects & History
    description: >-
      Implement commit objects linking tree snapshots into a history DAG
      with author, committer, and message metadata.
    acceptance_criteria:
      - commit-tree creates a commit object referencing a tree hash, optional parent commit hash(es), author, committer, and message
      - Commit format contains 'tree {hash}\n', 'parent {hash}\n' (zero or more), 'author {name} <{email}> {timestamp} {timezone}\n', 'committer ...\n', blank line, then message
      - Timestamps are stored as Unix epoch seconds with timezone offset (e.g., '1234567890 +0000')
      - Commit with no parent is the root commit; commit with one parent extends linear history; commit with two parents is a merge commit
      - Commit hash matches real git's commit hash for identical tree, parent, author, committer, timestamp, and message
      - log command traverses the parent chain from a given commit, displaying hash, author, date, and message for each ancestor
    pitfalls:
      - Timestamp format is precisely '{unix_epoch} {+/-}{HHMM}' with a space between epoch and timezone - any deviation changes the hash
      - Multi-line commit messages are terminated by EOF, not by a delimiter - the entire remainder after the blank line is the message
      - Merge commits have multiple 'parent' lines, one per parent - order matters (first parent is the branch you merged into)
      - Author vs committer distinction: author is who wrote the change, committer is who applied it (matters for cherry-pick, rebase)
      - Commit traversal for log must handle merge commits by following only the first parent for linear history display
    concepts:
      - Commit object as a snapshot reference plus metadata
      - Directed acyclic graph formed by parent references
      - Immutable history - commits cannot be changed, only new commits created
      - Author vs committer distinction
    skills:
      - Graph construction
      - Timestamp and timezone handling
      - Multi-line text serialization
      - DAG traversal
    deliverables:
      - commit-tree creating commit objects with tree, parent(s), author, committer, and message
      - Commit hash computation matching real git for identical inputs
      - log command traversing parent chain and displaying history
      - Support for root commits (no parent), linear commits (one parent), and merge commits (two parents)
    estimated_hours: 4

  - id: build-git-m4
    name: References & Branches
    description: >-
      Implement branch management as file-based references to commit hashes
      with HEAD tracking.
    acceptance_criteria:
      - branch command creates .git/refs/heads/{name} containing the 40-character commit hash followed by newline
      - branch -d deletes the ref file after verifying the branch is not the current branch and the branch is fully merged
      - branch (no args) lists all branches, marking the current branch with an asterisk
      - HEAD contains a symbolic reference 'ref: refs/heads/{branch}\n' for normal operation
      - checkout {branch} updates HEAD to point to the new branch's symbolic reference
      - Detached HEAD state writes a raw commit hash to HEAD when checking out a specific commit (not a branch)
      - checkout also updates the working directory to match the tree of the target commit
      - Ref file updates are atomic (write to temp file, then rename) to prevent corruption on crash
    pitfalls:
      - Symbolic refs ('ref: refs/heads/main') vs direct refs (raw commit hash) - HEAD can be either, all ref-reading code must handle both
      - Deleting the currently checked-out branch would leave HEAD dangling - always prevent this
      - Ref file locking prevents concurrent writes; without it, simultaneous operations corrupt the ref
      - checkout must update the working directory files, not just HEAD - this requires reading the target commit's tree and writing files
      - Switching branches with uncommitted changes should warn or abort to prevent data loss
    concepts:
      - References as lightweight pointers to commit objects
      - Symbolic references (HEAD -> refs/heads/main -> commit hash)
      - Detached HEAD as a direct commit reference
      - Atomic file operations for crash safety
    skills:
      - File-based reference management
      - Symbolic vs direct reference resolution
      - Atomic file writes
      - Working directory manipulation
    deliverables:
      - Branch creation writing ref files under .git/refs/heads/
      - Branch listing with current branch indicator
      - HEAD management supporting symbolic and detached states
      - checkout updating HEAD and working directory to target commit's tree
      - Atomic ref file updates via temp file + rename
    estimated_hours: 5

  - id: build-git-m5
    name: Index (Staging Area)
    description: >-
      Implement the binary index file for staging changes between the
      working directory and the next commit.
    acceptance_criteria:
      - add command stages a file by computing its blob hash, storing the blob object, and adding/updating an entry in the index
      - Index is stored as a binary file at .git/index with a 12-byte header (DIRC signature, version 2, entry count), sorted entries, and a trailing SHA-1 checksum
      - Each index entry stores ctime, mtime, device, inode, mode, uid, gid, file size, SHA-1 hash, flags (name length), and null-terminated path name, padded to 8-byte boundary
      - status command compares index vs HEAD tree (staged changes) and index vs working directory (unstaged changes) to show modified, added, and deleted files
      - write-tree (from M2) is updated to build tree objects from the index entries rather than the working directory
      - Removing a file from the index (rm --cached) removes the entry without deleting the working directory file
      - Index entries are sorted by path name for binary search during lookup
    pitfalls:
      - Index is a complex binary format - byte-level alignment and padding to 8-byte boundaries is critical for compatibility with real git
      - Stat info (ctime, mtime, size) is used to detect working directory changes without re-hashing - this is a performance optimization, not a correctness mechanism
      - Path encoding uses forward slashes on all platforms; backslashes must be normalized
      - The trailing SHA-1 checksum covers all bytes from the header to the last entry - any byte error corrupts the index
      - Adding a file that's already in the index should update the existing entry, not create a duplicate
    concepts:
      - Staging area as an intermediate state between working directory and commit
      - Binary file format with header, variable-length entries, and integrity checksum
      - File metadata caching for efficient change detection
      - Three-state comparison: HEAD tree, index, working directory
    skills:
      - Binary file format parsing and writing
      - Filesystem metadata extraction
      - Byte-level data packing and alignment
      - Path normalization
    deliverables:
      - Index file reader parsing binary format with header, entries, and checksum validation
      - add command staging files by creating blob and updating index entry
      - status command showing staged and unstaged changes via three-way comparison
      - write-tree updated to build trees from index entries
      - Index entry removal for unstaging files
    estimated_hours: 6

  - id: build-git-m6
    name: Diff Algorithm
    description: >-
      Implement the Myers diff algorithm for efficient line-level change
      detection with unified diff output format.
    acceptance_criteria:
      - Myers diff algorithm finds the shortest edit script (minimum number of insertions + deletions) between two sequences of lines
      - Output uses unified diff format with '---' and '+++' file headers and '@@ -a,b +c,d @@' hunk headers
      - Context lines (default 3) surround each change hunk for readability
      - diff command compares working directory file against index version showing unstaged changes
      - diff --cached compares index against HEAD tree showing staged changes
      - diff {commit1} {commit2} compares two commits' tree objects showing all changed files
      - Binary files are detected and reported as 'Binary files differ' rather than attempting line diff
    pitfalls:
      - Myers algorithm uses O(ND) time and O(N) space where D is the edit distance - for very different files, D approaches N making it O(N^2)
      - Line ending differences (\n vs \r\n) can produce misleading diffs if not normalized
      - Binary file detection must check for null bytes in content - attempting to diff binary produces garbage output
      - Hunk header line numbers must be correct relative to both the old and new file versions - off-by-one errors are common
      - Large files with many changes produce enormous diffs - consider a maximum diff size or binary threshold
    concepts:
      - Myers diff algorithm finding shortest edit script on the edit graph
      - Edit distance and the relationship between insertions, deletions, and common subsequences
      - Unified diff format with hunk headers and context lines
      - Tree diff comparing two tree objects to find changed files
    skills:
      - Dynamic programming / greedy algorithm implementation
      - Text processing and line-based comparison
      - Output formatting with hunk headers and context
      - Tree comparison for file-level changes
    deliverables:
      - Myers diff algorithm producing shortest edit script between two line sequences
      - Unified diff formatter with file headers, hunk headers, and context lines
      - Working directory vs index diff (unstaged changes)
      - Index vs HEAD diff (staged changes)
      - Commit-to-commit diff comparing tree objects
      - Binary file detection skipping line diff for non-text content
    estimated_hours: 7

  - id: build-git-m7
    name: Commit Workflow & Log
    description: >-
      Implement the full commit workflow (add -> commit) and enhanced
      history traversal with log.
    acceptance_criteria:
      - commit command reads the index, creates a tree object via write-tree, creates a commit object with the current HEAD as parent, and updates the current branch ref
      - Commit message is provided via -m flag or opened in $EDITOR if no message is provided
      - After commit, the index matches the new commit's tree (no staged changes remain)
      - log command displays commit history with hash, author, date, and message, following parent chain from HEAD
      - log --oneline shows abbreviated hash and first line of commit message
      - Empty commits (no changes staged since last commit) are rejected with an error message
    pitfalls:
      - The commit must update the branch ref that HEAD points to, not HEAD itself (unless in detached HEAD state)
      - First commit has no parent - handle the initial commit case where HEAD points to a nonexistent ref
      - Commit must be atomic - if any step fails (tree creation, commit creation, ref update), no partial state should remain
      - Editor-based commit message must handle the user aborting (empty message = abort commit)
    concepts:
      - Commit workflow as a multi-step atomic operation
      - Index-to-tree-to-commit pipeline
      - Branch ref update after commit
      - Initial commit handling (no parent)
    skills:
      - Multi-step atomic operations
      - Process spawning for editor
      - Ref update coordination
    deliverables:
      - commit command creating tree and commit objects from current index
      - Branch ref update pointing to new commit
      - Commit message handling via -m flag and editor fallback
      - Empty commit detection and rejection
      - Enhanced log with formatting options
    estimated_hours: 5

  - id: build-git-m8
    name: Three-Way Merge
    description: >-
      Implement three-way merge with common ancestor finding, automatic
      conflict resolution, and conflict markers for manual resolution.
    acceptance_criteria:
      - Merge base algorithm finds the common ancestor commit of two branch tips by traversing both parent chains
      - For each file, three-way comparison uses the base version, current branch version, and target branch version to determine changes
      - Files changed only in one branch are automatically applied to the merge result
      - Files changed in both branches with non-overlapping changes are automatically merged
      - Overlapping changes produce conflict markers (<<<<<<< HEAD, =======, >>>>>>> branch) in the working directory file
      - Successful merge (no conflicts) creates a merge commit with two parent hashes
      - Conflicted merge leaves the working directory in a conflicted state; the user must resolve and commit manually
      - merge --abort restores the pre-merge state
    pitfalls:
      - Finding the merge base in complex history with multiple common ancestors requires careful graph traversal (BFS from both tips, find first intersection)
      - File renames between base and tip make three-way merge significantly more complex - handle the simple case first (no renames)
      - Conflict markers must be exactly '<<<<<<< HEAD\n', '=======\n', '>>>>>>> {branch}\n' for compatibility with standard merge tools
      - Nested conflicts (conflicts within conflicts from multiple merge attempts) must be prevented
      - Both-deleted files should be silently removed; add/add conflicts (new file in both branches) need special handling
    concepts:
      - Three-way merge using common ancestor as the base version
      - Merge base finding via graph traversal (lowest common ancestor)
      - Automatic merge of non-conflicting changes
      - Conflict detection and marker insertion for overlapping changes
      - Merge commit with two parents preserving branch topology
    skills:
      - Graph traversal for ancestor finding
      - Three-way comparison algorithm
      - Conflict detection and marker generation
      - State management for merge in progress
    deliverables:
      - Merge base finder locating common ancestor commit of two branches
      - Three-way file merge applying non-conflicting changes automatically
      - Conflict marker insertion for overlapping changes
      - Merge commit creation with two parent hashes
      - Merge abort restoring pre-merge state
      - Handling of added, deleted, and modified files across branches
    estimated_hours: 10
```
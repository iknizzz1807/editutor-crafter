# AUDIT & FIX: file-upload-service

## CRITIQUE
- **Audit Finding Confirmed (Missing Metadata Layer - CRITICAL):** There is no database or metadata store anywhere in the milestones. File uploads need persistent metadata tracking: upload ID → storage path, file status (uploading, scanning, complete, quarantined), file owner, original filename, size, checksum, timestamps. Without this, the system has no way to answer 'what files exist?' or 'who uploaded this?' or 'what's the status of upload X?'
- **Audit Finding Confirmed (Scan Timing):** M3 says 'Scan uploaded files with ClamAV...before storage' but the milestone ordering has storage abstraction (M2) before virus scanning (M3). The architectural flow should be: chunks arrive → assemble in temporary/quarantine storage → scan → on clean, move to final storage. The current ordering implies files are already in final storage before scanning.
- **M1 Protocol Confusion:** The description says 'tus.io-style resumable upload protocol' but the ACs describe a multipart upload with part numbers (more like S3 multipart), not tus.io semantics (which uses byte offsets with PATCH requests and Upload-Offset headers). These are fundamentally different protocols. Pick one and be precise.
- **Estimated Hours:** All three milestones are exactly 11.5 hours each, totaling 34.5h against a project estimate of 35h. This suspiciously uniform distribution suggests no thought was given to relative complexity. Virus scanning integration is simpler than building a resumable upload protocol.
- **M2 S3 Consistency Pitfall:** The pitfall says 'S3 eventually consistent for overwrite - use versioning.' Since December 2020, S3 provides strong read-after-write consistency for all operations. This is factually outdated.
- **Missing Garbage Collection:** The learning outcomes mention 'orphaned chunk garbage collection' but no milestone has an AC for cleaning up incomplete/abandoned uploads.
- **Missing Rate Limiting/Quotas:** The learning outcomes mention 'rate limiting and quota management' but no AC addresses this.
- **Only 3 Milestones:** For an advanced project estimated at 35 hours, three milestones is too few. The metadata layer and cleanup/quota management deserve their own milestone.

## FIXED YAML
```yaml
id: file-upload-service
name: File Upload Service
description: >-
  Build a production-grade file upload service with resumable chunked uploads,
  multi-backend storage abstraction, virus scanning, and file metadata management.
difficulty: advanced
estimated_hours: "35-50"
essence: >-
  Stateful chunked binary transfer protocol with byte-range offset tracking
  and resumption semantics, unified storage backend abstraction across
  heterogeneous cloud provider APIs, metadata-driven file lifecycle management,
  and stream-based malware detection in a quarantine-first architecture.
why_important: >-
  Building this teaches production-grade file handling patterns used by
  services like Dropbox and Google Drive, covering critical real-world
  challenges like unreliable networks, multi-cloud storage strategies,
  security validation, and file lifecycle management.
learning_outcomes:
  - Implement a resumable upload protocol with chunk offset tracking and resume capability
  - Design a metadata layer tracking file ownership, status, and storage location
  - Build a storage abstraction supporting multiple backends with a unified interface
  - Integrate ClamAV for stream-based virus scanning in a quarantine-first architecture
  - Implement MIME type validation using magic bytes, not file extensions
  - Handle orphaned upload cleanup with garbage collection for abandoned sessions
  - Design idempotent upload operations with checksum verification for data integrity
  - Implement upload quotas and rate limiting per user
skills:
  - HTTP Protocol Design
  - Storage Abstraction Patterns
  - Stream Processing
  - Virus Scanning Integration
  - Chunked Upload Handling
  - Checksum Verification
  - Binary Data Handling
  - Cloud Storage APIs
  - Metadata Management
tags:
  - advanced
  - chunked
  - file-transfer
  - resumable
  - s3
  - scanning
  - service
  - storage
  - uploads
architecture_doc: architecture-docs/file-upload-service/index.md
languages:
  recommended:
    - Go
    - Python
    - Rust
  also_possible:
    - Java
    - Node.js
resources:
  - name: tus.io Resumable Upload Protocol
    url: https://tus.io/protocols/resumable-upload
    type: documentation
  - name: AWS S3 Multipart Upload Guide
    url: https://docs.aws.amazon.com/AmazonS3/latest/userguide/mpuoverview.html
    type: documentation
  - name: ClamAV Documentation
    url: https://docs.clamav.net/
    type: documentation
prerequisites:
  - type: project
    id: http-server-basic
    name: HTTP Server (Basic)
  - type: skill
    name: Database basics
milestones:
  - id: file-upload-service-m1
    name: Upload Metadata & Session Management
    description: >-
      Design the metadata database schema and implement upload session
      creation, tracking, and lifecycle management.
    acceptance_criteria:
      - "Database schema defines tables for files (id, owner_id, original_filename, size_bytes, mime_type, checksum_sha256, status, storage_path, created_at, updated_at) and upload_sessions (id, file_id, total_size, bytes_received, chunk_size, status, expires_at, created_at)"
      - "POST /uploads initializes a new upload session returning an upload_id, expected chunk size, and upload endpoint URL"
      - "GET /uploads/:id returns the current upload session state including bytes_received and status (pending, uploading, assembling, scanning, complete, quarantined, failed)"
      - "Upload sessions expire after a configurable timeout (e.g., 24 hours); expired sessions are marked as failed"
      - "File metadata records are created with status 'pending' at upload initialization and updated through the lifecycle"
      - "Per-user upload quota is enforced; requests exceeding the quota are rejected with 413 Payload Too Large"
    pitfalls:
      - "Not tracking upload session state in a database means server restarts lose all in-progress uploads"
      - "Expiry cleanup must not delete sessions that are actively receiving chunks; check last_activity timestamp"
      - "Not enforcing upload quotas allows a single user to exhaust storage"
      - "File status enum must cover the full lifecycle; missing states cause ambiguous file conditions"
    concepts:
      - File lifecycle state machine
      - Upload session management
      - Quota enforcement
      - Database schema for file metadata
    skills:
      - Database schema design
      - Session lifecycle management
      - Quota and rate limiting
      - REST API design
    deliverables:
      - Database schema DDL for files and upload_sessions tables
      - Upload session creation endpoint
      - Session status query endpoint
      - Quota enforcement middleware
      - Session expiry background job
    estimated_hours: "7-10"

  - id: file-upload-service-m2
    name: Resumable Chunked Upload Protocol
    description: >-
      Implement a resumable chunked upload protocol where clients upload
      file data in sequential byte-range chunks with resume-on-failure capability.
    acceptance_criteria:
      - "PATCH /uploads/:id accepts a chunk with Upload-Offset and Content-Length headers; server validates offset matches bytes_received"
      - "Server responds with Upload-Offset header indicating the next expected byte offset after each successful chunk"
      - "If the client sends an offset that doesn't match server's bytes_received, the server returns 409 Conflict with the correct offset"
      - "HEAD /uploads/:id returns the current Upload-Offset so clients can resume from the correct position after a failure"
      - "Each chunk's integrity is verified using a Content-MD5 or checksum header; mismatched chunks are rejected with 400"
      - "When all bytes are received (bytes_received == total_size), chunks are assembled into a complete file in temporary/quarantine storage"
      - "Assembled file's SHA-256 checksum is computed and verified against the client-provided checksum if supplied"
      - "Concurrent uploads to the same session are serialized; only one chunk is processed at a time per session"
    pitfalls:
      - "Offset mismatch between client and server causes data corruption; always validate offset before accepting chunk data"
      - "Not verifying per-chunk checksums means network corruption goes undetected until final assembly"
      - "Assembling chunks into the final file must be atomic; partial assembly on crash leaves corrupted files"
      - "Pre-allocating the target file avoids filesystem fragmentation on sequential chunk writes"
      - "Chunks must be written to temporary/quarantine storage, NOT final storage (scanning happens first)"
    concepts:
      - Byte-range offset tracking and resume semantics
      - Chunk integrity verification with checksums
      - Atomic file assembly from sequential chunks
      - Content-Range and Upload-Offset HTTP headers
    skills:
      - Chunked upload protocol implementation
      - Binary data handling and streaming
      - Checksum verification
      - HTTP header management
    deliverables:
      - Chunk upload endpoint with offset validation
      - Resume query endpoint (HEAD) returning current offset
      - Per-chunk checksum verification
      - File assembly from chunks into quarantine storage
      - Final file SHA-256 checksum computation and verification
    estimated_hours: "10-14"

  - id: file-upload-service-m3
    name: Virus Scanning & File Validation
    description: >-
      Validate file type using magic bytes, scan assembled files for malware
      in quarantine storage, and transition clean files to final storage.
    acceptance_criteria:
      - "File type is validated using magic bytes (file signature) from the first chunk; files not matching an allowlist of permitted types are rejected immediately"
      - "Assembled files in quarantine storage are scanned using ClamAV (via clamd socket) or equivalent before being moved to final storage"
      - "Clean files are moved from quarantine to final storage and their metadata status is updated to 'complete'"
      - "Infected files remain in quarantine storage with metadata status set to 'quarantined'; they are never moved to final storage"
      - "Configurable maximum file size is enforced; uploads exceeding the limit are rejected at session creation and chunk upload"
      - "Scan timeout is configured; files that exceed the scan timeout are retried up to a configurable limit before being marked as 'scan_failed'"
      - "Scan results (clean, infected, scan_failed) are logged with the file ID for audit purposes"
    pitfalls:
      - "Never trust file extensions alone; always check magic bytes for file type identification"
      - "Scanning must happen BEFORE moving to final storage; the quarantine-first architecture prevents serving infected files"
      - "ClamAV can timeout on large files; set appropriate scan timeouts and implement retry with backoff"
      - "Quarantined files should be retained for forensic analysis, not immediately deleted"
      - "Streaming the file to ClamAV via the clamd socket avoids loading the entire file into memory"
    concepts:
      - Magic number/file signature detection
      - Quarantine-first architecture
      - clamd socket protocol for streaming scans
      - File lifecycle state transitions
    skills:
      - File type validation
      - Antivirus integration
      - Quarantine storage management
      - Stream processing
    deliverables:
      - Magic byte validation on first chunk receipt
      - ClamAV integration via clamd socket for streaming scan
      - Quarantine-to-final storage promotion for clean files
      - Quarantine retention for infected files with audit logging
      - Scan retry logic with configurable timeout and attempt limit
    estimated_hours: "8-12"

  - id: file-upload-service-m4
    name: Storage Abstraction & Cleanup
    description: >-
      Implement a pluggable storage backend abstraction supporting local
      filesystem and S3-compatible object storage, plus garbage collection
      for orphaned uploads.
    acceptance_criteria:
      - "Storage interface defines read, write, delete, and generate_signed_url methods implemented by each backend"
      - "Local filesystem backend implements the storage interface with path-traversal-safe key sanitization"
      - "S3-compatible backend implements the storage interface using multipart upload for files exceeding a configurable threshold (e.g., 5MB minimum part size)"
      - "Storage backend is selected via configuration; switching backends requires no code changes"
      - "Signed download URLs are generated with configurable expiration for secure file access without exposing storage credentials"
      - "Garbage collection job runs periodically, cleaning up: (a) expired upload sessions and their temporary chunks, (b) orphaned files in quarantine older than a configurable retention period"
      - "Files are streamed to/from storage without loading the entire file into memory"
    pitfalls:
      - "S3 multipart upload requires minimum 5MB parts (except the last); violating this causes API errors"
      - "Local filesystem storage must sanitize keys to prevent path traversal attacks (e.g., ../../etc/passwd)"
      - "Streaming large files through memory causes OOM; always use buffered I/O with bounded buffer sizes"
      - "Garbage collection must not delete files that are actively being scanned or assembled; check status before cleanup"
      - "S3 has been strongly consistent since December 2020; the old eventual consistency concern for overwrites no longer applies"
    concepts:
      - Interface-based storage abstraction
      - S3 multipart upload protocol
      - Signed URL generation for secure access
      - Garbage collection for orphaned resources
      - Streaming I/O with bounded buffers
    skills:
      - Storage abstraction design
      - S3 API integration
      - Garbage collection scheduling
      - Stream-based I/O
      - Secure URL generation
    deliverables:
      - Storage interface with read, write, delete, and signed URL methods
      - Local filesystem backend with path sanitization
      - S3-compatible backend with multipart upload support
      - Configuration-based backend selection
      - Garbage collection job for expired sessions and orphaned quarantine files
    estimated_hours: "10-14"
```
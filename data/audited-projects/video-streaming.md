# AUDIT & FIX: video-streaming

## CRITIQUE
- CRITICAL: The audit correctly identifies that chunked upload (M1) to transcoding (M2) has no chunk assembly step. If you upload in chunks, you must reassemble and validate the complete file before FFmpeg can process it. This is a missing AC.
- M1 AC says 'Chunked upload support allows resumable uploads' but provides no measurable definition of 'resumable' — should specify that uploads can resume from the last successfully received chunk after a connection drop.
- M1 deliverables mention 'Metadata extraction service reading duration, resolution, and codec from uploaded files' but this can only work after chunk assembly, which is not a deliverable.
- M2 AC says 'FFmpeg integration transcodes uploaded videos into target format and bitrate' — this is vague. Should specify which codecs, containers, and bitrate targets.
- M2 AC says 'Progress monitoring reports transcoding percentage and estimated time remaining' — FFmpeg progress is typically derived from the frame count vs total frames, which should be specified.
- M2 deliverables say 'Codec selection logic choosing optimal output codec based on target device compatibility' but no AC tests this selection logic.
- M3 AC says 'Byte-range requests support partial content delivery for efficient seeking' — this is HTTP Range requests (206 Partial Content), which is about serving, not about HLS per se. HLS segments are discrete files; byte-range is for progressive download. These are conflated.
- M3 is titled 'Adaptive Streaming' but the AC doesn't actually test adaptive behavior — no AC verifies that the player switches between quality levels based on bandwidth.
- M3 deliverables say 'HLS segment generation' but this should logically happen during transcoding (M2), not during serving (M3). The pipeline order is confused.
- M4 estimated hours (11-20) is a very wide range suggesting unclear scope.
- M4 AC 'Playback analytics record view duration, quality switches, and buffering events' — no measurable threshold for what constitutes a 'buffering event.'
- No mention of audio track handling anywhere — video without audio consideration is incomplete.
- Total estimated hours: 30-50, but the milestones individually sum to 30-50 which is fine.
- No security considerations for upload endpoint (file type validation beyond format, malicious file scanning).

## FIXED YAML
```yaml
id: video-streaming
name: Video Streaming Platform
description: >-
  Video upload, transcoding, HLS adaptive bitrate streaming with player
  integration.
difficulty: intermediate
estimated_hours: 45
essence: >-
  Chunked resumable upload with server-side assembly, CPU-intensive FFmpeg
  transcoding pipelines generating multi-bitrate HLS variants with proper
  segmentation, and HTTP-based adaptive streaming delivery where the client
  player switches quality levels based on bandwidth conditions.
why_important: >-
  Building this teaches production-grade video infrastructure patterns including
  handling large file uploads, CPU-intensive transcoding workflows, and adaptive
  streaming protocols critical for modern web applications.
learning_outcomes:
  - Implement chunked file upload with resumable progress tracking for large media files
  - Build FFmpeg transcoding pipelines to generate multiple quality variants from source videos
  - Design HLS manifest generation with multi-bitrate playlist configuration
  - Implement adaptive bitrate streaming with quality ladder (360p, 720p, 1080p)
  - Handle video processing job queues and background worker architecture
  - Build custom video player with quality switching and playback analytics
  - Optimize video delivery using CDN integration with cache-control headers
skills:
  - HLS Protocol
  - FFmpeg Transcoding
  - Chunked Upload
  - Adaptive Bitrate Streaming
  - Media Processing Pipelines
  - Background Job Queues
  - Video Codec Handling
  - CDN Integration
tags:
  - adaptive-bitrate
  - file-transfer
  - hls
  - intermediate
  - manifest
  - segments
architecture_doc: architecture-docs/video-streaming/index.md
languages:
  recommended:
    - Node.js
    - Python
    - Go
  also_possible:
    - Java
    - Rust
resources:
  - type: article
    name: HLS Streaming Explained
    url: https://www.cloudflare.com/learning/video/what-is-hls-streaming/
  - type: tool
    name: FFmpeg Documentation
    url: https://ffmpeg.org/documentation.html
  - type: article
    name: "tus: Resumable Upload Protocol"
    url: https://tus.io/
prerequisites:
  - type: skill
    name: HTTP server implementation
  - type: skill
    name: File handling and streaming I/O
  - type: skill
    name: Basic frontend (HTML/JS)
milestones:
  - id: video-streaming-m1
    name: Chunked Upload & Assembly
    description: >-
      Handle large video file uploads via chunked transfer with resumable
      progress, server-side assembly, and validation before downstream
      processing.
    estimated_hours: 8
    concepts:
      - Chunked upload protocols (tus.io or custom)
      - Resumable uploads with chunk tracking and offset management
      - Server-side chunk assembly: ordered concatenation with integrity verification
      - File type validation (magic bytes, not just extension)
    skills:
      - Multipart/chunked upload handling
      - Stream-based file processing
      - Resumable upload state management
      - File type and integrity validation
    acceptance_criteria:
      - "Chunked upload endpoint accepts sequential chunks (configurable chunk size, default 5MB) and tracks upload progress by offset; the client can query the current offset to resume after disconnection"
      - "After all chunks are received, the server assembles them into a complete file and verifies integrity using a client-provided checksum (SHA-256 or MD5)"
      - "File validation rejects files that are not valid video containers (MP4, MKV, WebM, MOV) based on magic bytes, and rejects files exceeding a configurable size limit (default 5GB)"
      - "Progress tracking API returns bytes uploaded, total expected bytes, and completion percentage for each in-progress upload"
      - "Metadata extraction (duration, resolution, codec, audio tracks) is performed on the assembled file and stored in the database, and the video is marked as 'ready for transcoding'"
      - Incomplete uploads older than 24 hours are automatically cleaned up by a background process
    pitfalls:
      - Loading entire file into memory instead of streaming chunks to disk
      - Not validating chunk ordering or detecting missing/duplicate chunks
      - Trusting file extension instead of checking magic bytes for type validation
      - Not cleaning up incomplete upload chunks on abandonment
      - Skipping checksum verification and accepting corrupted files
    deliverables:
      - Chunked upload endpoint accepting sequential parts with offset tracking
      - Upload resume API returning current offset for a given upload ID
      - Chunk assembly service concatenating chunks and verifying integrity
      - File type validator checking magic bytes and enforcing size limits
      - Metadata extraction service reading video properties from assembled file
      - Stale upload cleanup job removing incomplete uploads after configurable timeout

  - id: video-streaming-m2
    name: FFmpeg Transcoding & HLS Segmentation
    description: >-
      Transcode uploaded videos into multiple quality levels and generate HLS
      segments and manifests for adaptive streaming.
    estimated_hours: 12
    concepts:
      - Container formats vs codecs: MP4/fMP4 containers with H.264/H.265/VP9 codecs
      - "Constant Rate Factor (CRF) for quality-based encoding vs target bitrate"
      - HLS segmentation: splitting transcoded output into fixed-duration TS or fMP4 segments
      - Keyframe interval (GOP) alignment across quality levels for seamless switching
      - Audio track handling: AAC encoding, preserving or re-encoding audio streams
    skills:
      - FFmpeg command-line integration
      - Video codec parameters and profiles
      - HLS manifest and segment generation
      - Background job queue management
    acceptance_criteria:
      - FFmpeg transcodes each uploaded video into at least 3 quality variants: 360p (800kbps), 720p (2500kbps), and 1080p (5000kbps) using H.264 baseline/main profile with AAC audio
      - "Each quality variant is segmented into fixed-duration HLS segments (default 6 seconds) with aligned keyframes across all variants"
      - "A master M3U8 playlist is generated referencing variant playlists for each quality level with BANDWIDTH and RESOLUTION tags"
      - "Each variant has its own M3U8 playlist listing all segments with correct duration (EXT-X-TARGETDURATION) and sequence numbers"
      - "Transcoding runs as a background job with configurable concurrency (max N simultaneous transcodes); new uploads are queued when workers are busy"
      - "Transcoding progress is reported (based on frame count / total frames) and queryable via API, updated at least every 5 seconds"
      - "Transcoding failures are retried up to 3 times; permanently failed jobs are marked with error details"
    pitfalls:
      - Not using H.264 baseline profile for maximum device compatibility
      - Misaligned keyframes across quality levels causing buffering on quality switch
      - "Forgetting -movflags +faststart for progressive MP4 download (for non-HLS fallback)"
      - HLS segment duration too short (many HTTP requests) or too long (high startup latency)
      - Not handling audio-only or video-only input files
      - FFmpeg process consuming all system memory on high-resolution source files
    deliverables:
      - FFmpeg wrapper executing transcoding as subprocess with stdout progress parsing
      - Quality ladder configuration defining resolution, bitrate, and codec per variant
      - HLS segment generator producing TS segments with aligned keyframes
      - "Master M3U8 manifest generator with variant playlists (EXT-X-STREAM-INF)"
      - Transcoding job queue with configurable concurrency and retry logic
      - Transcoding progress API reporting percentage and ETA

  - id: video-streaming-m3
    name: Streaming Delivery
    description: >-
      Serve HLS streams with correct headers, CORS configuration, and
      CDN-friendly caching for adaptive playback.
    estimated_hours: 8
    concepts:
      - HLS serving: M3U8 manifests and TS/fMP4 segments over HTTP
      - "Cache-control headers for segments (immutable, long max-age) vs manifests (short max-age for live)"
      - CORS headers for cross-origin player access
      - CDN integration for edge caching of segments
    skills:
      - HTTP content serving
      - CORS configuration
      - CDN cache header design
      - Content-type handling for media
    acceptance_criteria:
      - Master M3U8 playlists are served with Content-Type: application/vnd.apple.mpegurl and CORS headers allowing the player origin
      - TS segments are served with Content-Type: video/MP2T and cache-control headers (max-age >= 1 year, immutable) since segment content never changes
      - "Variant M3U8 playlists are served with shorter cache-control (max-age 60s) to allow for potential updates"
      - "A standard HLS player (hls.js, Safari native) successfully plays the stream and switches between quality levels based on simulated bandwidth conditions"
      - "404 responses are returned within 50ms for non-existent video IDs, preventing enumeration attacks"
      - Content-Length header is accurate for all served files
    pitfalls:
      - Missing CORS headers causing player to fail on cross-origin requests
      - Caching manifest files too aggressively preventing quality switch updates
      - Wrong Content-Type headers causing player parsing failures
      - Serving segments from application server instead of CDN/object storage under load
      - Confusing HLS segment serving with HTTP Range requests (they serve different purposes)
    deliverables:
      - Manifest serving endpoint returning M3U8 files with correct content type
      - Segment serving endpoint returning TS files with immutable cache headers
      - CORS middleware configured for player origins
      - CDN configuration guide or integration for segment edge caching
      - Video detail API returning master playlist URL and video metadata

  - id: video-streaming-m4
    name: Video Player Integration
    description: >-
      Build a frontend video player with HLS.js, quality switching controls,
      and playback analytics.
    estimated_hours: 17
    concepts:
      - HLS.js: JavaScript HLS client for browsers without native HLS support
      - Adaptive bitrate switching (ABR) based on measured download throughput
      - Media Source Extensions (MSE) API for programmatic media feeding
      - Playback event-driven analytics collection
    skills:
      - HLS.js integration
      - Video player UI development
      - Playback state management
      - Analytics event collection
    acceptance_criteria:
      - "HLS.js is initialized with the master playlist URL and plays adaptive streams in Chrome, Firefox, and Edge; Safari uses native HLS"
      - "Quality selector UI shows available renditions (360p, 720p, 1080p, Auto) and allows manual selection; switching occurs within 2 segments"
      - "Progress bar displays current playback position, buffered ranges, and total duration; click-to-seek navigates to the target time within 1 second"
      - Playback analytics events are recorded: view start, view duration at 5-second intervals, quality switches (from/to level and timestamp), and buffering events (start time, duration >500ms)
      - Player handles network errors gracefully: displays error state and allows retry without page reload
      - Play, pause, volume, and fullscreen controls function correctly
    pitfalls:
      - Not destroying HLS.js instance on component unmount causing memory leaks
      - Browser compatibility: Safari has native HLS but doesn't support MSE-based hls.js the same way
      - Bandwidth estimation inaccuracy in low-bandwidth conditions causing quality thrashing
      - Not handling seek to unbuffered position (need to wait for segment download)
    deliverables:
      - HLS.js initialization with master playlist URL and error handling
      - Quality switching UI with Auto and manual rendition selection
      - Playback controls (play, pause, seek, volume, fullscreen)
      - Progress bar with buffered range visualization
      - Analytics collector recording view events, quality switches, and buffering
      - Error handling overlay with retry functionality
```